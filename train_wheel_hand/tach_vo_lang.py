import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import joblib
from collections import defaultdict
import sys

# ==========================================================
# CẤU HÌNH VÀ HẰNG SỐ
# ==========================================================
# 🛑 CHỈNH SỬA CÁC ĐƯỜNG DẪN NÀY 🛑
YOLO_MODEL_PATH = r"E:\PythonProject\data\New folder\best (1).pt"  # File YOLO đã train
DATA_DIR = r"E:\PythonProject\data\New folder"  # Thư mục gốc chứa ảnh (on_wheel, off_wheel)
OUTPUT_FILE = "wheel_features.npz"  # File đầu ra mới
# ----------------------------------

EPS = 1e-8
# ĐÃ CẬP NHẬT: Kích thước đặc trưng mục tiêu (2 tay, hoặc 1 tay + padding)
EXPECTED_FEATURES = 128
mp_hands = mp.solutions.hands


# ==========================================================
# CÁC HÀM XỬ LÝ VÔ LĂNG & TAY
# ==========================================================

def load_yolo_wheel_model(model_path):
    """Tải mô hình YOLOv8 đã train."""
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"❌ LỖI TẢI YOLO: {e}")
        return None


def detect_wheel_yolo(frame, yolo_model):
    """Phát hiện vô lăng bằng mô hình YOLOv8 và trả về (x, y, r)."""
    if yolo_model is None: return None
    results = yolo_model(frame, verbose=False, conf=0.5, classes=[0])

    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            box = boxes[0].xyxy[0].cpu().numpy().astype(int)
            x_min, y_min, x_max, y_max = box

            x_w = (x_min + x_max) // 2
            y_w = (y_min + y_max) // 2
            r_w = int((x_max - x_min + y_max - y_min) / 4)

            return (x_w, y_w, r_w)

    return None


def get_mp_hands_instance():
    """Tạo instance MediaPipe Hands."""
    return mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


def extract_features(image, hands_processor, wheel):
    """
    Trích xuất 128 đặc trưng (tổng) tay, bao gồm các đặc trưng tương đối và góc độ.
    Logic này phải khớp hoàn toàn với hàm extract_features trong file dự đoán.
    """
    if wheel is None: return None
    xw, yw, rw = wheel
    h, w, _ = image.shape
    feats_all = []

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = hands_processor.process(rgb)

    if not res.multi_hand_landmarks:
        return None

    for hand_landmarks in res.multi_hand_landmarks:
        feats = []
        normalized_coords = []

        # 1. Trích xuất Tọa độ chuẩn hóa (63 đặc trưng: 21 * x,y,z)
        for lm in hand_landmarks.landmark:
            feats.extend([lm.x, lm.y, lm.z])
            normalized_coords.append(np.array([lm.x, lm.y]))  # Dùng cho tính toán khoảng cách

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
        # Nắm chặt sẽ làm các khoảng cách này thay đổi
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

    return np.array(feats_all[:EXPECTED_FEATURES], dtype=np.float32)


# ==========================================================
# HÀM CÂN BẰNG DỮ LIỆU (UPSAMPLING)
# ==========================================================

def balance_data(X_list, Y_list, class_names):
    """Cân bằng dữ liệu bằng cách Upsampling lớp thiểu số."""

    # 1. Nhóm dữ liệu theo lớp
    class_data = defaultdict(lambda: {'X': [], 'Y': []})
    for x, y in zip(X_list, Y_list):
        class_data[y]['X'].append(x)
        class_data[y]['Y'].append(y)

    # 2. Tìm kích thước của lớp lớn nhất
    max_size = 0
    for class_id in class_data:
        max_size = max(max_size, len(class_data[class_id]['X']))

    # 3. Thực hiện Upsampling
    X_balanced = []
    Y_balanced = []

    print(f"\n--- Cân bằng dữ liệu (Upsampling) ---")

    for class_id in class_data:
        X_class = np.array(class_data[class_id]['X'])
        Y_class = np.array(class_data[class_id]['Y'])
        current_size = len(X_class)

        if current_size == 0:
            continue

        num_repeats = max_size // current_size
        remainder = max_size % current_size

        X_repeated = np.repeat(X_class, num_repeats, axis=0)
        Y_repeated = np.repeat(Y_class, num_repeats)

        if remainder > 0:
            # Chọn ngẫu nhiên các mẫu để thêm vào phần dư
            indices = np.random.choice(current_size, size=remainder, replace=False)
            X_repeated = np.concatenate([X_repeated, X_class[indices]])
            Y_repeated = np.concatenate([Y_repeated, Y_class[indices]])

        X_balanced.extend(X_repeated.tolist())
        Y_balanced.extend(Y_repeated.tolist())

        print(
            f"Lớp {class_names[class_id]} (ID {class_id}): Ban đầu {current_size} mẫu -> Sau cân bằng {len(X_repeated)} mẫu.")

    # 4. Chuyển về Numpy Array và xáo trộn (Shuffle)
    X_balanced = np.array(X_balanced)
    Y_balanced = np.array(Y_balanced)

    indices = np.arange(len(X_balanced))
    np.random.shuffle(indices)

    return X_balanced[indices], Y_balanced[indices]


# ==========================================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================================

def create_npz_data():
    all_features = []
    all_labels = []

    if not os.path.exists(DATA_DIR):
        print(f"❌ LỖI: Không tìm thấy thư mục dữ liệu '{DATA_DIR}'.")
        return

    # Tải model YOLO và tạo map nhãn
    yolo_model = load_yolo_wheel_model(YOLO_MODEL_PATH)
    if yolo_model is None: return

    # 1. Tạo label map và class names
    class_dirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    if not class_dirs:
        print(f"❌ LỖI: Không tìm thấy thư mục con nào (ví dụ: on_wheel, off_wheel) trong '{DATA_DIR}'.")
        return

    label_map = {name: i for i, name in enumerate(class_dirs)}
    class_names = {i: name for name, i in label_map.items()}  # Map ID -> Name

    hands_processor = get_mp_hands_instance()

    print(f"✅ Model YOLO đã tải. Bắt đầu trích xuất đặc trưng ({EXPECTED_FEATURES} chiều)...")
    print(f"Bản đồ nhãn: {label_map}")

    # 2. Vòng lặp trích xuất
    for label_name, label_id in label_map.items():
        class_path = os.path.join(DATA_DIR, label_name)

        for filename in os.listdir(class_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(class_path, filename)

                try:
                    img = np.array(Image.open(file_path).convert('RGB'))
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    wheel_coords = detect_wheel_yolo(img_bgr, yolo_model)

                    if wheel_coords is None: continue

                    # SỬ DỤNG HÀM TRÍCH XUẤT CẢI TIẾN 128 CHIỀU
                    features = extract_features(img_bgr, hands_processor, wheel_coords)

                    if features is not None:
                        all_features.append(features)
                        all_labels.append(label_id)

                    # NOTE: Xử lý trường hợp off_wheel không tìm thấy tay
                    # Luật cứng này có thể gây nhiễu, nhưng giữ lại theo logic cũ
                    elif label_name.lower() == 'off_wheel':
                        # Gán mảng 0.0 (128 đặc trưng) nếu không tìm thấy tay trong class off_wheel
                        zero_features = np.zeros(EXPECTED_FEATURES, dtype=np.float32)
                        all_features.append(zero_features)
                        all_labels.append(label_id)

                except Exception as e:
                    # print(f"  [LỖI] Xử lý {filename} thất bại: {e}")
                    pass

    print("\n--- Hoàn tất trích xuất thô. Bắt đầu cân bằng dữ liệu... ---")

    # 3. Cân bằng dữ liệu
    X_balanced, Y_balanced = balance_data(all_features, all_labels, class_names)

    # 4. Lưu trữ
    np.savez_compressed(OUTPUT_FILE, X=X_balanced, Y=Y_balanced, classes=list(label_map.keys()))

    print("\n" + "=" * 50)
    print(f"✅ HOÀN TẤT LƯU TRỮ DỮ LIỆU ĐÃ CÂN BẰNG!")
    print(f"File đầu ra: {OUTPUT_FILE}")
    print(f"Tổng số mẫu sau cân bằng: {X_balanced.shape[0]}")
    print(f"Kích thước đặc trưng: {X_balanced.shape[1]}")
    print("=" * 50)


if __name__ == "__main__":
    create_npz_data()
