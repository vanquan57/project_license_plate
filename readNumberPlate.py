import string
import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)


def preprocess_image(image):
    # Kiểm tra kích thước ảnh đầu vào
    height, width = image.shape[:2]
    if height < 50 or width < 100:  # Ngưỡng kích thước tối thiểu
        scale_factor = max(3, 300 / min(height, width))  # Tăng tỷ lệ nếu ảnh quá nhỏ
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Chuyển sang thang độ xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Cân bằng histogram để cải thiện độ tương phản
    gray = cv2.equalizeHist(gray)

    # Làm mờ để giảm nhiễu
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding thay vì Otsu để xử lý ánh sáng không đồng đều
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Morphological closing để loại bỏ lỗ nhỏ và kết nối vùng ký tự
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Xóa nhiễu nhỏ bằng cách mở (morphological opening)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

    return processed


def read_license_plate(license_plate_crop):

    print("Đang đọc biển số...")
    # Tiền xử lý ảnh
    processed_img = preprocess_image(license_plate_crop)

    detections = reader.readtext(processed_img)
    converted_text = ''
    score = 0.0

    for detection in detections:
        bbox, text, detection_score = detection
        
        text = text.upper().replace(' ', '')
        converted_text += text
        score = detection_score

    return converted_text, score
