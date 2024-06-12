import cv2
import os
import numpy as np

def rgb_to_hsv(pixel):
    r, g, b = pixel 
    r, g, b = b / 255.0, g / 255.0, r / 255.0
    
    v = max(r, g, b)
    delta = v - min(r, g, b)
    
    if delta == 0:
        h = 0
        s = 0
    else:
        s = delta / v
        if r == v:
            h = (g - b) / delta
        elif g == v:
            h = 2 + (b - r) / delta
        else:
            h = 4 + (r - g) / delta
        h = (h / 6) % 1.0
        
    return [int(h * 180), int(s * 255), int(v * 255)]

def convert_image_rgb_to_hsv(img):
    hsv_image = []
    for row in img:
        hsv_row = []
        for pixel in row:
            new_color = rgb_to_hsv(pixel)
            hsv_row.append(new_color)
        hsv_image.append(hsv_row)
    hsv_image = np.array(hsv_image)
    return hsv_image

def my_calcHist(image, channels, histSize, ranges):
    hist = cv2.calcHist(
        [image.astype(np.uint8)], channels, None, histSize, ranges
    )
    return hist


# Đường dẫn tới thư mục chứa ảnh
folder_path = "static/data_new"

# Tạo một danh sách để lưu trữ các đặc trưng màu sắc
color_features = []

# Lặp qua tất cả các folder con trong thư mục chính
for subdir in os.listdir(folder_path):
    subdir_path = os.path.join(folder_path, subdir)
    if os.path.isdir(subdir_path):
        # Lặp qua tất cả các file ảnh trong mỗi folder con
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            # Đọc ảnh bằng OpenCV
            img = cv2.imread(img_path)
            # Chuyển đổi ảnh sang không gian màu HSV
            img_hsv = convert_image_rgb_to_hsv(img)
            # Tính toán histogram màu sắc
            hist = my_calcHist(img_hsv, channels=[0, 1, 2], histSize=[8, 8, 8], ranges=[0, 180, 0, 255, 0, 255])
            # Thêm histogram vào danh sách đặc trưng màu sắc
            color_features.append(hist.flatten())

# Lưu mảng đặc trưng màu sắc vào file "mausac.npy"
np.save("mausac.npy", color_features)

print(color_features)
