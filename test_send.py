import requests

imgs = []
for c in range(100, 106):
    try:
        imgs.append((str(c) + '.jpg', open('/home/alexey/yolov5/hakaton/train/images/' + str(c) + '.jpg', 'rb'), 'image/jpeg'))
        # imgs.append(im.tobytes())
    except:
        pass
print(requests.post('http://127.0.0.1:5555/recognize', files={'files': imgs}).json())