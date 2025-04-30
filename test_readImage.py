from readNumberPlate import read_license_plate
import cv2

image_path = r'images\clip17_new_7.jpg'

if image_path is None:
    print("Không đọc được ảnh. Vui lòng kiểm tra lại đường dẫn.")
    exit()

image = cv2.imread(image_path)

plate_text, confidence = read_license_plate(image)

if plate_text:
    print(f"Biển số đọc được: {plate_text}, Độ tin cậy: {confidence}")
else:
    print("Không đọc được biển số.")