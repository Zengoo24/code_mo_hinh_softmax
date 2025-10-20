import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import sys
import time
from ultralytics import YOLO

# ======================================================================
# 1. CẤU HÌNH VÀ TẢI MODEL
# ======================================================================

# 🛑 CẤU HÌNH ĐƯỜNG DẪN CỦA BẠN 🛑
# Vui lòng thay đổi các đường dẫn này cho phù hợp với máy tính của bạn
YOLO_MODEL_PATH = r"E:\PythonProject\data\New folder\best (1).pt"
MY_IMAGE_PATH = r"E:\PythonProject\data\New folder\off-wheel\2274.jpg" # ⚠️ THAY ĐỔI ĐƯỜNG DẪN NÀY
# ----------------------------------

try:
    # Tải các model cần thiết (YÊU CẦU PHẢI HUẤN LUYỆN LẠI VỚI 128 ĐẶC TRƯNG MỚI)
    model_data = joblib.load("softmax_wheel_model.pkl")
    scaler = joblib.load("scaler_wheel.pkl")
    YOLO_MODEL = YOLO(YOLO_MODEL_PATH)

except FileNotFoundError as e:
    print(f"❌ Lỗi: Không tìm thấy file cần thiết: {e.filename}. Hãy kiểm tra lại.")
    sys.exit()
except Exception as e:
    print(f"❌ Lỗi tải mô hình YOLO hoặc Joblib: {e}")
    sys.exit()

W = model_data["W"]
b = model_data["b"]
CLASS_NAMES = model_data["classes"]
X_mean = scaler["X_mean"]
X_std = scaler["X_std"]

# CÁC HẰNG SỐ VÀ KHAI BÁO MP
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
EPS = 1e-8
# ĐÃ ĐẶT EXPECTED_FEATURES = 128 (phải khớp với logic trích xuất mới)
EXPECTED_FEATURES = 128
if W.shape[0] != EXPECTED_FEATURES:
    print(f"❌ LỖI: Mô hình Softmax cần {EXPECTED_FEATURES} đặc trưng nhưng file .pkl chỉ có {W.shape[0]}.")
    print("Vui lòng HUẤN LUYỆN LẠI mô hình Softmax bằng dữ liệu 128 chiều mới.")
    sys.exit()


# ======================================================================
# 2. HÀM CỐT LÕI (SOFTMAX, YOLO, VÀ TRÍCH XUẤT ĐẶC TRƯNG 128 CHIỀU)
# ======================================================================

def softmax(z):
    """Tính toán hàm Softmax."""
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def detect_wheel_yolo(frame, yolo_model):
    """Phát hiện vô lăng bằng YOLOv8 và trả về (bbox, x, y, r)."""
    results = yolo_model(frame, verbose=False, conf=0.5, classes=[0])

    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            box = boxes[0].xyxy[0].cpu().numpy().astype(int)
            x_min, y_min, x_max, y_max = box

            x_w = (x_min + x_max) // 2
            y_w = (y_min + y_max) // 2
            r_w = int((x_max - x_min + y_max - y_min) / 4)

            return (x_min, y_min, x_max, y_max), (x_w, y_w, r_w)

    return None, None


def extract_features(image, wheel_coords):
    """
    Trích xuất 128 đặc trưng tay, bao gồm tọa độ, khoảng cách tương đối và góc độ.
    Logic này phải khớp hoàn toàn với script tạo dữ liệu NPZ.
    """
    xw, yw, rw = wheel_coords
    h, w, _ = image.shape
    feats_all = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if not res.multi_hand_landmarks:
            return None

        for hand_landmarks in res.multi_hand_landmarks:
            feats = []
            normalized_coords = []

            # 1. Trích xuất Tọa độ chuẩn hóa (63 đặc trưng: 21 * x,y,z)
            for lm in hand_landmarks.landmark:
                feats.extend([lm.x, lm.y, lm.z])
                normalized_coords.append(np.array([lm.x, lm.y]))

                # 2. Đặc trưng Khoảng cách đến tâm vô lăng (1 đặc trưng)
            hx = hand_landmarks.landmark[0].x * w
            hy = hand_landmarks.landmark[0].y * h
            dist_to_center = np.sqrt((xw - hx) ** 2 + (yw - hy) ** 2)
            feats.append(dist_to_center / (rw + EPS))

            # --- THÊM CÁC ĐẶC TRƯNG MỚI (Dấu hiệu của việc Nắm) ---

            # a) Đặc trưng vị trí tương đối của các đầu ngón tay so với tâm vô lăng (10 đặc trưng)
            tip_indices = [4, 8, 12, 16, 20]

            for i in tip_indices:
                lm_tip = hand_landmarks.landmark[i]

                tip_x = lm_tip.x * w
                tip_y = lm_tip.y * h

                # Khoảng cách tương đối
                rel_dist = np.sqrt((xw - tip_x) ** 2 + (yw - tip_y) ** 2)
                feats.append(rel_dist / (rw + EPS))

                # Góc tương đối
                angle = np.arctan2(tip_y - yw, tip_x - xw) / np.pi
                feats.append(angle)

            # b) Đặc trưng Khoảng cách giữa các ngón tay (10 đặc trưng)
            pairs = [(5, 8), (9, 12), (13, 16), (17, 20), (0, 5)]
            for i, j in pairs:
                p_i = normalized_coords[i]
                p_j = normalized_coords[j]

                distance = np.linalg.norm(p_i - p_j)
                feats.append(distance)

            feats_all.extend(feats)

        # Cắt hoặc thêm 0.0 để đảm bảo đúng EXPECTED_FEATURES (128)
        if len(feats_all) < EXPECTED_FEATURES:
            feats_all.extend([0.0] * (EXPECTED_FEATURES - len(feats_all)))

        feats_all = feats_all[:EXPECTED_FEATURES]

    return np.array(feats_all, dtype=np.float32)


# ======================================================================
# 3. HÀM DỰ ĐOÁN VÀ HIỂN THỊ CHÍNH
# ======================================================================
def predict_image_and_show(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không thể đọc ảnh tại: {image_path}")
        return

    print(f"\nẢnh đang kiểm tra: {os.path.basename(image_path)}")

    # 1. PHÁT HIỆN VÔ LĂNG BẰNG YOLO
    bbox, wheel_coords = detect_wheel_yolo(img, YOLO_MODEL)

    if wheel_coords is None:
        cv2.putText(img, "WHEEL NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        print("⚠️ Bỏ qua: YOLO không phát hiện được vô-lăng.")
        cv2.imshow("YOLOv8 Test", img);
        cv2.waitKey(0);
        cv2.destroyAllWindows()
        return

    # 2. TRÍCH XUẤT ĐẶC TRƯNG (Sử dụng 128 đặc trưng mới)
    features = extract_features(img, wheel_coords)

    # 🛑 LUẬT CỨNG (KHÔNG TAY) -> RỜI 🛑
    if features is None:
        final_predicted_class = "off-wheel"
        display_label = "ROI (OFF-WHEEL)"
        final_color = (0, 0, 255)  # Đỏ/Blue cho OFF

    else:
        # 3. DỰ ĐOÁN SOFTMAX
        X_sample = features.reshape(1, -1)
        X_scaled = (X_sample - X_mean) / X_std
        z = X_scaled @ W + b
        probabilities = softmax(z)[0]
        predicted_index = np.argmax(probabilities)
        final_predicted_class = CLASS_NAMES[predicted_index]

        # 4. Gán nhãn hiển thị
        display_label = f"{final_predicted_class.upper()} ({probabilities[predicted_index] * 100:.2f}%)"
        final_color = (0, 255, 0) if final_predicted_class == "on-wheel" else (0, 0, 255)  # Xanh lá cho ON, Đỏ cho OFF

    print(f"-> KẾT QUẢ CUỐI: {display_label}")

    # 5. HIỂN THỊ KẾT QUẢ TRỰC QUAN

    # Vẽ Vô lăng
    x_min, y_min, x_max, y_max = bbox
    xw, yw, rw = wheel_coords

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Bounding Box YOLO
    cv2.circle(img, (xw, yw), rw, (255, 0, 255), 2)  # Vòng tròn ước tính (Magenta)
    cv2.circle(img, (xw, yw), 5, (0, 0, 255), -1)  # Tâm (Đỏ)

    # Vẽ Tay (landmarks)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                # Đổi màu vẽ tay thành màu Xanh lơ (Cyan) như trong ảnh gốc để dễ nhìn
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=2))

    # 🛑 HIỂN THỊ KẾT QUẢ PHÂN LOẠI CUỐI 🛑
    cv2.putText(img, display_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, final_color, 3, cv2.LINE_AA)

    cv2.imshow("Wheel Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================================================================
# 4. THỰC THI CHÍNH
# ======================================================================
if __name__ == "__main__":
    predict_image_and_show(MY_IMAGE_PATH)
