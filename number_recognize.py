import torch
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from utils.general import non_max_suppression, scale_boxes
from ultralytics.utils import ops
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from flask import Flask, request, jsonify
from collections import Counter

def most_common_string(array):
    # Фильтрация: только строки
    strings = [str(s) for s in array if isinstance(s, str)]

    if not strings:
        return ""

    # Вычисляем среднюю длину
    avg_len = round(sum(len(s) for s in strings) / len(strings))

    result = []

    for i in range(avg_len):
        chars_at_i = [s[i] for s in strings if len(s) > i]
        if chars_at_i:
            most_common_char, _ = Counter(chars_at_i).most_common(1)[0]
            result.append(most_common_char)
        else:
            result.append('')  # Если на позиции ничего нет — добавим пустоту или что-то по дефолту

    return ''.join(result)

app = Flask(__name__)


weights = 'best.pt'
classnames = ['number']

label = {}
for i, name in enumerate(classnames):
    label[i]=name

# load pre-trained model
weights = weights

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
model = DetectMultiBackend(weights, device=torch.device('cpu'), dnn=False, data="data/coco128.yaml", fp16=False)
stride, names, pt = model.stride, model.names, model.pt
model.warmup(imgsz=(1 if pt or model.triton else 1, 3, 640, 640))  # warmup

def image_loader(img,imsize):
    '''
    processes input image for inference 
    '''
    h, w = img.shape[:2]
    img = cv2.resize(img,(imsize,imsize))
    img = img[:, :, ::-1].transpose(2, 0, 1) 
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()
    img /= 255.0
    img = img.unsqueeze(0)
    return img, h, w 


def get_pred(img, stride):
    '''
    returns prediction in numpy array
    '''
    im0 = img.copy()
    im = letterbox(im0, 640, stride=stride, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    return pred, im.shape

def inference():
    try:
        img = cv2.imread('image.jpg')
    except:
        return "Image don't readed"
    im0 = img.copy()

    prediction, im_shape = get_pred(img, stride)
    # print(prediction)

    if prediction is not None:
        for i, det in enumerate(prediction):
            det[:, :4] = scale_boxes(im_shape[2:], det[:, :4], im0.shape).round()
            # print(det)

            # Write results
            if len(det) == 0:
                return "Number don't find"
            texts = []
            for *xyxy, conf, cls in reversed(det):
                if not isinstance(xyxy, torch.Tensor):  # may be list
                    xyxy = torch.stack(xyxy)
                b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
                b[:, 2:] = b[:, 2:] * 1.02 + 10  # box wh * gain + pad
                xyxy = ops.xywh2xyxy(b).long()
                # print(xyxy, '\n\n\n')
                # print(int(xyxy[0, 1]), int(xyxy[0, 3]), int(xyxy[0, 0]), int(xyxy[0, 2]))
                crop = img[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: 1]
                cr_shape = crop.shape
                if cr_shape[0] > cr_shape[1]:
                    crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite('result.jpg', crop)

                image = Image.open('result.jpg')

                pixel_values = processor(image, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                net_str = ''
                for i in range(len(recognized_text)):
                    try:
                        symbol = int(recognized_text[i])
                        net_str += str(symbol)
                    except:
                        pass
                texts.append(net_str)
            max_len = 0
            ret = ''
            for text in texts:
                if len(text) > max_len:
                    ret = text
                    max_len = len(text)
            return ret

@app.route('/recognize', methods=['POST'])
def recognize():
    oldPhotos = request.files.getlist('oldPhotos')
    newPhotos = request.files.getlist('newPhotos')

    if not oldPhotos or not newPhotos:
        return jsonify({"oldNumber": '', "newNumber": '', "errorMessage": "No files provided"})

    resOld = []
    resNew = []

    for file in oldPhotos:
        file.save('./image.jpg')
        im = cv2.imread('./image.jpg')
        cv2.imwrite('./image.jpg', im)
        inf_res = inference()
        if inf_res == "Number don't find":
            return jsonify({"oldNumber": '', "newNumber": '', "errorMessage": "Old number don't find"})
        if inf_res:
            resOld.append(inf_res)
    for file in newPhotos:
        file.save('./image.jpg')
        im = cv2.imread('./image.jpg')
        cv2.imwrite('./image.jpg', im)
        inf_res = inference()
        if inf_res == "Number don't find":
            return jsonify({"oldNumber": '', "newNumber": '', "errorMessage": "New number don't find"})
        if inf_res:
            resNew.append(inf_res)
    mostOld = most_common_string(resOld)
    mostNew = most_common_string(resNew)
    if mostOld == '' and mostNew == '':
        return jsonify({"oldNumber": '', "newNumber": '', "errorMessage": "Old number and new number don't recognize"})
    if mostOld == '':
        return jsonify({"oldNumber": '', "newNumber": '', "errorMessage": "Old number don't recognize"})
    if mostNew == '':
        return jsonify({"oldNumber": '', "newNumber": '', "errorMessage": "New number don't recognize"})

    return jsonify({"oldNumber": most_common_string(resOld), "newNumber": most_common_string(resNew)})

app.run(host="0.0.0.0", port=5555)
