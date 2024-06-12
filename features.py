import os
import cv2
import numpy as np

# Trích xuất đặc trưng màu sắc và hình dạng
def extract_features(image):
    color_feature = extract_color_features(image)
    shape_feature = extract_shape_features(image)
    # Kết hợp các đặc trưng lại thành một vector đặc trưng
    features = np.concatenate((color_feature, shape_feature))
    return features

# Trích xuất đặc trưng màu sắc
def extract_color_features(image):
    # Chuyển đổi ảnh từ RGB sang HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Tạo histogram màu sắc theo các kênh Hue, Saturation và Value
    hist = cv2.calcHist(
        [hsv_image], [0, 1, 2], None, [12, 12, 3], [0, 180, 0, 256, 0, 256]
    )
    # Chuẩn hóa histogram (tính xác suất xuất hiện của mỗi giá trị màu)
    hist = cv2.normalize(hist, hist).flatten()
    #print(hist) 
    #print("+++++++++")
    return hist

# Trích xuất đặc trưng hình dạng
def extract_shape_features(image):
    # Chuyển đổi ảnh sang ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Phát hiện biên cạnh bằng phương pháp Canny
    edges = cv2.Canny(gray_image, 100, 200)
    
    # Tính toán số lượng điểm ảnh biên cạnh
    num_edges = np.sum(edges)
    
    # Tìm các đường viền (contours)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tính toán tổng diện tích của các đường viền
    contour_area = np.sum([cv2.contourArea(c) for c in contours])
    
    # Tính toán số lượng đường viền
    num_contours = len(contours)
    
    # Tỷ lệ phần trăm các điểm ảnh biên cạnh so với tổng số điểm ảnh
    edge_density = num_edges / (image.shape[0] * image.shape[1])
    
    # Kết hợp các đặc trưng hình dạng vào một mảng 1 chiều
    shape_feature = np.array([num_edges, contour_area, num_contours, edge_density])
    
    #print(shape_feature)
    #print("---------")
    return shape_feature

# Đường dẫn tới thư mục chứa ảnh
folder_path = "static/data_new"

# Tạo một danh sách để lưu trữ các đặc trưng của tất cả các ảnh
all_features = []
all_labels = []

# Lặp qua tất cả các folder con trong thư mục chính
for subdir in os.listdir(folder_path):
    subdir_path = os.path.join(folder_path, subdir)
    if os.path.isdir(subdir_path):
        # Lặp qua tất cả các file ảnh trong mỗi folder con
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            label = os.path.join(subdir, filename).replace("\\", "/")
            # Đọc ảnh bằng OpenCV
            image = cv2.imread(img_path)
            # Trích xuất đặc trưng của ảnh và thêm vào danh sách các đặc trưng
            feature = extract_features(image)
            all_features.append(feature)
            all_labels.append(label)  # Thêm nhãn của ảnh vào danh sách nhãn

#print(all_labels)
print(all_features)

# Chuyển danh sách các nhãn thành mảng numpy
all_labels = np.array(all_labels)

# Lưu mảng đặc trưng vào file "features.npy"
np.save("all_features.npy", all_features)
# Lưu mảng nhãn vào file "labels.npy"
np.save("all_labels.npy", all_labels)
