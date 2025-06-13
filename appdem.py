import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/admin/Desktop/Code/yolov5/runs/train/chicken_detector/weights/best.pt')

st.title("Ứng dụng đếm gà trong ảnh 🐔")

uploaded_file = st.file_uploader("Tải ảnh lên", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    st.image(image, caption='Ảnh bạn vừa tải lên', use_column_width=True)

    # Nhận diện
    results = model(img_np)
    df = results.pandas().xyxy[0]
    chicken_count = (df['name'] == 'chicken').sum()
    st.write(f'👉 Số con gà phát hiện: **{chicken_count}**')

    # Hiển thị ảnh có bounding boxes
    results.render()  # gắn box vào ảnh
    st.image(results.ims[0], caption='Kết quả nhận diện', use_column_width=True)
