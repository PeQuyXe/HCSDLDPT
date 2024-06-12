import os
import numpy as np
import cv2
from skimage import feature

def hog_feature_extraction(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hog_vector = feature.hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")

    return hog_vector

# Đường dẫn tới thư mục chứa ảnh
folder_path = "data_new"

# Tạo một danh sách để lưu trữ các đặc trưng hình dạng
shape_features_list = []

# Lặp qua tất cả các folder con trong thư mục chính
for subdir in os.listdir(folder_path):
    subdir_path = os.path.join(folder_path, subdir)
    if os.path.isdir(subdir_path):
        # Lặp qua tất cả các file ảnh trong mỗi folder con
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            # Đọc ảnh bằng OpenCV
            img = cv2.imread(img_path)
            hog_feats = hog_feature_extraction(img)

# Lưu mảng đặc trưng hình dạng vào tệp "hinhdang.npy"
np.save("hinhdang.npy", shape_features_list)

