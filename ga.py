#model 
import torch
import cv2
from matplotlib import pyplot as plt

# Load mô hình YOLOv5 pretrained (có thể nhận diện gà nếu dùng custom model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load ảnh
img_path = 'chickens.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Nhận diện đối tượng
results = model(img_rgb)

# Hiển thị kết quả (nếu muốn)
results.print()
results.show()

# Đếm số "chickens" nếu dùng mô hình đã huấn luyện custom
# Với YOLOv5 mặc định, bạn cần huấn luyện lại model để phân biệt được 'chicken'

# Ví dụ đếm số đối tượng theo class
labels = results.pandas().xyxy[0]['name']
chicken_count = (labels == 'bird').sum()  # nếu mô hình nhận chicken là bird
print(f'Số con gà phát hiện: {chicken_count}')
