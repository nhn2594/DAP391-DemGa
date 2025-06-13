import torch
import cv2
import os

# Load model đã huấn luyện
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/admin/Desktop/Code/yolov5/runs/train/chicken_detector/weights/best.pt')

# Thư mục chứa ảnh
img_folder = 'chicken_images/'

# Lặp qua các ảnh trong thư mục
for filename in os.listdir(img_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(img_folder, filename)
        img = cv2.imread(path)
        results = model(img)
        labels = results.pandas().xyxy[0]['name']
        chicken_count = (labels == 'chicken').sum()
        print(f'{filename}: {chicken_count} con gà')
