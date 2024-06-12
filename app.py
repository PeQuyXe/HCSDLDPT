import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Tải các đặc trưng và nhãn đã được lưu trước đó
all_features = np.load("all_features.npy")
all_labels = np.load("all_labels.npy")

# Sử dụng KNN để dự đoán nhãn của ảnh đầu vào
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(all_features, all_labels)

# Trích xuất đặc trưng màu sắc và hình dạng từ ảnh đầu vào
def extract_features(image):
    color_feature = extract_color_features(image)
    shape_feature = extract_shape_features(image)
    features = np.concatenate((color_feature, shape_feature))
    return features

# Trích xuất đặc trưng màu sắc
def extract_color_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv_image], [0, 1, 2], None, [12, 12, 3], [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Trích xuất đặc trưng hình dạng
def extract_shape_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    shape_feature = np.array([np.sum(edges)])
    return shape_feature

# Tìm kiếm ảnh gần nhất
def find_nearest_images(image_features, k=3):
    image_features = np.array(image_features)
    distances, indices = knn_model.kneighbors(image_features.reshape(1, -1), n_neighbors=k)
    
    # Tạo danh sách các ảnh gần nhất với nhãn tương ứng
    nearest_images = [(f"data_new/{all_labels[i]}") for i in indices[0]]
    
    return nearest_images


# Tính toán độ chính xác khi đưa ra ảnh gần nhất
def calculate_accuracy(nearest_images, true_label):
    nearest_labels = [label for label, _ in nearest_images]
    accuracy = nearest_labels.count(true_label) / len(nearest_labels)
    return accuracy

# Tìm kiếm ảnh
@app.route('/', methods=['GET'])
def predict():
    return render_template('predict.html', prediction=None)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files['file']
    if file:
        # Lưu file ảnh mới vào thư mục test2
        file_path = "test2/input.jpg"
        file.save(file_path)
        # Trích xuất đặc trưng về màu sắc và hình dạng của ảnh đầu vào
        image = cv2.imread(file_path)
        image_features = extract_features(image).reshape(1, -1)

        # Tìm ảnh gần nhất
        nearest_images = find_nearest_images(image_features)

        return render_template('result.html', nearest_images=nearest_images)
    else:
        return jsonify({'error': 'Không có ảnh nào được tìm thấy!'})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
