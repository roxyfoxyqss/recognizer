import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/alexey/yolov5/runs/train/numbers2/weights/best.pt', force_reload=True)
