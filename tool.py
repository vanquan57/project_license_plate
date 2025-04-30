import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO

# Load mô hình YOLO
yolo_model = YOLO('best.pt')

# Đường dẫn đến thư mục chứa ảnh

image_dir = r"E:\HK6\học máy\toolsHandleImage\images"
image2_dir = r"E:\HK6\học máy\toolsHandleImage\imagesAfterHandle\train\images"  # Thư mục ảnh đã xử lý
label_dir = r"E:\HK6\học máy\toolsHandleImage\imagesAfterHandle\train\labels"

# Kích thước tối đa của cửa sổ hiển thị
MAX_WINDOW_WIDTH = 1000
MAX_WINDOW_HEIGHT = 800

# Lấy danh sách ảnh
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
current_index = 0  # Chỉ số ảnh hiện tại

# Hàm tạo label YOLO
def create_yolo_label(box, keypoints, image_name):
    # Chuyển đổi tọa độ box về dạng số nguyên
    x1, y1, x2, y2 = map(int, box)

    # Kiểm tra và chuẩn hóa các keypoints
    keypoints = keypoints.xy.cpu().numpy().astype(int)  # Lấy keypoints và chuyển thành số nguyên

    # Lấy chiều rộng và chiều cao của ảnh để chuẩn hóa tọa độ
    img = cv2.imread(os.path.join(image_dir, image_name))
    h, w, _ = img.shape

    # Mở file label, sẽ ghi thêm dòng mới nếu đã có
    label_file = os.path.join(label_dir, f"{os.path.splitext(image_name)[0]}.txt")
    
    with open(label_file, 'a') as f:  # Mở file ở chế độ 'append' (ghi thêm)
        for kps in keypoints:
            # Kiểm tra nếu có ít nhất 4 keypoints
            if kps.shape[0] < 4:
                print(f"Lỗi: Không đủ keypoints cho ảnh {image_name} (Số keypoints: {kps.shape[0]})")
                continue  # Bỏ qua nếu không đủ keypoints

            corners = kps.flatten()  # Lấy 4 điểm góc biển số

            # Chuẩn hóa tọa độ theo tỷ lệ
            x1, y1 = x1 / w, y1 / h
            x2, y2 = x2 / w, y2 / h
            corners = corners / np.array([w, h, w, h, w, h, w, h])

            # Tạo chuỗi label cho đối tượng
            label_str = f"1 {corners[0]} {corners[1]} {corners[2]} {corners[3]} {corners[4]} {corners[5]} {corners[6]} {corners[7]}\n"
            f.write(label_str)  # Ghi label vào file

            print(f"Đã lưu label cho {image_name}: {label_str.strip()}")


# Hàm load ảnh mới
def load_new_image():
    global img, image_path, results

    if len(image_files) == 0:
        print("Không còn ảnh để hiển thị.")
        cv2.destroyAllWindows()
        return

    image_path = os.path.join(image_dir, image_files[current_index])
    img = cv2.imread(image_path)

    if img is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return

    # Chạy mô hình YOLO trên ảnh
    results = yolo_model.predict(image_path, verbose=False)

    for result in results:
        for box, kps in zip(result.boxes, result.keypoints):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            corners = np.array(kps.xy.cpu().numpy(), dtype=np.int32)[:4]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.polylines(img, [corners], isClosed=True, color=(0, 0, 255), thickness=2)

    # Điều chỉnh kích thước cửa sổ nếu ảnh quá lớn
    h, w, _ = img.shape
    scale = min(MAX_WINDOW_WIDTH / w, MAX_WINDOW_HEIGHT / h, 1.0)  # Không phóng to ảnh
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))

    # Hiển thị ảnh mới
    cv2.namedWindow("Detected License Plates", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected License Plates", new_w, new_h)
    cv2.imshow("Detected License Plates", resized_img)

# Bắt đầu chương trình
if len(image_files) > 0:
    load_new_image()

    while True:
        key = cv2.waitKey(0)

        if key == 27:  # Nhấn ESC để thoát
            break

        elif key == ord('w') or key == 2490368:  # Phím W hoặc mũi tên lên (Lưu label & di chuyển ảnh)
            print("Đã nhấn mũi tên lên (Lưu label & Di chuyển ảnh).")
            for result in results:
                for box, kps in zip(result.boxes, result.keypoints):
                    create_yolo_label(box.xyxy[0].cpu().numpy(), kps, image_files[current_index])

            shutil.move(image_path, os.path.join(image2_dir, image_files[current_index]))
            print(f"Đã di chuyển ảnh {image_files[current_index]} sang thư mục images2")

            del image_files[current_index]
            if current_index >= len(image_files):
                current_index -= 1
            load_new_image()

        elif key == ord('s') or key == 2621440:  # Phím S hoặc mũi tên xuống (Xóa ảnh)
            print("Đã nhấn mũi tên xuống (Xóa ảnh).")
            os.remove(image_path)
            print(f"Đã xóa ảnh {image_files[current_index]} khỏi thư mục images")

            del image_files[current_index]
            if current_index >= len(image_files):
                current_index -= 1
            load_new_image()

        elif key == ord('d') or key == 2555904:  # Phím D hoặc mũi tên phải (Ảnh tiếp theo)
            print("Đã nhấn mũi tên phải (Chuyển ảnh tiếp theo và xóa ảnh).")

            # Xóa ảnh khỏi thư mục gốc
            os.remove(image_path)
            print(f"Đã xóa ảnh {image_files[current_index]} khỏi thư mục images")

            # Cập nhật danh sách ảnh và chỉ số hiện tại
            del image_files[current_index]
            if current_index >= len(image_files):
                current_index -= 1
            load_new_image()


        elif key == ord('a') or key == 2424832:  # Phím A hoặc mũi tên trái (Ảnh trước)
            print("Đã nhấn mũi tên trái (Quay lại ảnh trước).")
            if current_index > 0:
                current_index -= 1
                load_new_image()

    cv2.destroyAllWindows()
else:
    print("Không có ảnh nào trong thư mục images.")
