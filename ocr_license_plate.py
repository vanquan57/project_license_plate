import cv2
import numpy as np
import os
from ultralytics import YOLO
from readNumberPlate import read_license_plate


# Load mô hình YOLO
yolo_model = YOLO('model/best.pt')

# Đường dẫn đến video
video_file = "Video_test.mp4"

# Tạo thư mục lưu ảnh nếu chưa tồn tại
save_dir = "data_image"
os.makedirs(save_dir, exist_ok=True)

# Kích thước tối đa của cửa sổ hiển thị
MAX_WINDOW_WIDTH = 1000
MAX_WINDOW_HEIGHT = 800

# Mở video
video = cv2.VideoCapture(video_file)

if not video.isOpened():
    print("Không thể mở video.")
    exit()

# Lấy FPS
fps = video.get(cv2.CAP_PROP_FPS)

# Tỷ lệ scale khung hình
scale_factor = 0.5

frame_count = 0
box_count = 0  # Đếm số lượng bounding box

while True:
    ret, frame = video.read()
    if not ret:
        print("Đã kết thúc video.")
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # Resize khung hình để xử lý nhanh hơn
    h, w, _ = frame.shape
    frame_resized = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))

    # Phát hiện đối tượng bằng YOLO
    results = yolo_model(frame_resized, verbose=False)

    for result in results:
        for box in result.boxes:
            # Lấy toạ độ bounding box (sau khi scale)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Chuyển toạ độ về kích thước gốc
            x1 = int(x1 / scale_factor)
            y1 = int(y1 / scale_factor)
            x2 = int(x2 / scale_factor)
            y2 = int(y2 / scale_factor)

            # # Cắt ảnh từ frame gốc
            cropped_img = frame[y1:y2, x1:x2]

            # # Kiểm tra để tránh lỗi khi toạ độ sai
            # if cropped_img.size > 0:
            #     license_text, score = read_license_plate(cropped_img)

            #     if license_text and score > 0.5:
            #         cv2.putText(frame, license_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            #                     0.9, (0, 255, 0), 2, cv2.LINE_AA)
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Resize frame để hiển thị
    h, w, _ = frame.shape
    scale = min(MAX_WINDOW_WIDTH / w, MAX_WINDOW_HEIGHT / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))

    # Hiển thị
    cv2.namedWindow("Detected Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected Video", new_w, new_h)
    cv2.imshow("Detected Video", resized_frame)

    if cv2.waitKey(1) == 27:  # ESC để thoát
        break

# Đóng tất cả
video.release()
cv2.destroyAllWindows()
