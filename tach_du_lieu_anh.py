import cv2
import mediapipe as mp
import numpy as np
import os, random
from tqdm import tqdm
from sklearn.utils import resample
import json
import warnings

# ==============================
# CẤU HÌNH
# ==============================
VIDEOS_DIR = ("train")
OUTPUT_FILE = "features_static_10feats.npz"  # Tên file mới
LABEL_MAP_FILE = "label_map_6cls.json"

# DANH SÁCH LỚP 6:
CLASSES = ["left", "right", "yawn", "blink", "normal", "nod"]

# KHÔNG CÓ WINDOW_SIZE, OVERLAP, FRAME_STEP
AUGMENT_FRAME_PROB = 0.1
BALANCE_DATA = True
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==============================
# CÁC HÀM AUGMENT (Giữ nguyên)
# ==============================
def motion_blur(frame, k=5):
    kernel = np.zeros((k, k))
    kernel[int((k - 1) / 2), :] = np.ones(k)
    return cv2.filter2D(frame, -1, kernel / k)


def add_noise(frame, sigma=10):
    noise = np.random.randn(*frame.shape) * sigma
    return np.clip(frame + noise, 0, 255).astype(np.uint8)


def change_brightness(frame, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def random_augmentation(frame):
    op = random.choice(["blur", "noise", "bright", "flip", "rotate"])
    if op == "blur":
        return motion_blur(frame, k=random.choice([3, 5, 7]))
    elif op == "noise":
        return add_noise(frame, sigma=random.choice([5, 10, 15]))
    elif op == "flip":
        return cv2.flip(frame, 1)
    elif op == "rotate":
        h, w = frame.shape[:2]
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(frame, M, (w, h))
    else:
        return change_brightness(frame, alpha=random.uniform(0.7, 1.3),
                                 beta=random.randint(-30, 30))


# ==============================
# HÀM TÍNH ĐẶC TRƯNG (Giữ nguyên)
# ==============================
EPS = 1e-6
EYE_LEFT_IDX = np.array([33, 159, 145, 133, 153, 144])
EYE_RIGHT_IDX = np.array([362, 386, 374, 263, 380, 385])
MOUTH_IDX = np.array([61, 291, 0, 17, 78, 308])


def eye_aspect_ratio(landmarks, left=True):
    idx = EYE_LEFT_IDX if left else EYE_RIGHT_IDX
    pts = landmarks[idx, :2]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * (C + EPS))


def mouth_aspect_ratio(landmarks):
    pts = landmarks[MOUTH_IDX, :2]
    A = np.linalg.norm(pts[0] - pts[1])
    B = np.linalg.norm(pts[4] - pts[5])
    C = np.linalg.norm(pts[2] - pts[3])
    return (A + B) / (2.0 * (C + EPS))


def head_pose_yaw_pitch_roll(landmarks):
    left_eye = landmarks[33][:2]
    right_eye = landmarks[263][:2]
    nose = landmarks[1][:2]
    chin = landmarks[152][:2]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll = np.degrees(np.arctan2(dy, dx + EPS))

    interocular = np.linalg.norm(right_eye - left_eye) + EPS
    eyes_center = (left_eye + right_eye) / 2.0
    yaw = np.degrees(np.arctan2((nose[0] - eyes_center[0]), interocular))

    baseline = chin - eyes_center
    pitch = np.degrees(np.arctan2((nose[1] - eyes_center[1]),
                                  (np.linalg.norm(baseline) + EPS)))
    return yaw, pitch, roll


def get_extra_features(landmarks):
    nose, chin = landmarks[1], landmarks[152]
    angle_pitch_extra = np.degrees(np.arctan2(chin[1] - nose[1], (chin[2] - nose[2]) + EPS))
    forehead_y = np.mean(landmarks[[10, 338, 297, 332, 284], 1])
    cheek_dist = np.linalg.norm(landmarks[50] - landmarks[280])
    return angle_pitch_extra, forehead_y, cheek_dist


# ==============================
# KHỞI TẠO MEDIAPIPE
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# TRÍCH XUẤT 10 ĐẶC TRƯNG (9 Static + Delta EAR + Delta Pitch)
# ==============================
X, y = [], []
label_map = {cls: i for i, cls in enumerate(CLASSES)}
label_counts = {cls: 0 for cls in CLASSES}

for label in CLASSES:
    folder = os.path.join(VIDEOS_DIR, label)
    if not os.path.exists(folder):
        print(f"[!] Bỏ qua '{folder}' (không tồn tại)")
        continue

    print(f"\n[+] Xử lý nhãn: {label}")
    num_aug_versions = 2 if label in ["blink", "yawn", "left", "right", "nod"] else 1
    total_samples = 0

    # Khởi tạo lịch sử EAR và Pitch cho Delta EAR và Delta Pitch
    last_ear_avg = 0.4
    last_pitch = 0.0

    # Lọc ra các file ảnh
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print(f"[!] Thư mục '{folder}' không chứa file ảnh nào.")
        continue

    # Sắp xếp file để duy trì thứ tự thời gian nếu đây là chuỗi ảnh
    files.sort()

    for file in tqdm(files, desc=label, leave=False):
        path = os.path.join(folder, file)

        original_frame = cv2.imread(path)
        if original_frame is None:
            warnings.warn(f"Không đọc được file ảnh {file}")
            continue

        for aug_idx in range(num_aug_versions):
            current_frame = original_frame.copy()

            # Augmentation
            if aug_idx > 0 and label in ["blink", "yawn", "left", "right", "nod"]:
                current_frame = random_augmentation(current_frame)
            elif random.random() < AUGMENT_FRAME_PROB:
                current_frame = random_augmentation(current_frame)

            h, w = current_frame.shape[:2]
            rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                # Nếu mất mặt, sử dụng EAR và Pitch lịch sử để tránh Delta quá lớn
                delta_ear_value = 0.0
                delta_pitch_value = 0.0
                continue

            face = results.multi_face_landmarks[0]
            landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in face.landmark])

            # 1. TÍNH TOÁN CÁC ĐẶC TRƯNG TĨNH
            ear_l = eye_aspect_ratio(landmarks, True)
            ear_r = eye_aspect_ratio(landmarks, False)
            mar = mouth_aspect_ratio(landmarks)
            yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
            angle_pitch_extra, forehead_y, cheek_dist = get_extra_features(landmarks)

            # 2. TÍNH ĐẶC TRƯNG ĐỘNG (Delta EAR & Delta Pitch)
            ear_avg = (ear_l + ear_r) / 2.0

            delta_ear_value = ear_avg - last_ear_avg
            delta_pitch_value = pitch - last_pitch  # Đặc trưng động mới cho NOD

            # 3. CẬP NHẬT LỊCH SỬ
            last_ear_avg = ear_avg
            last_pitch = pitch

            # Mảng 10 đặc trưng: [EAR_L, EAR_R, MAR, YAW, PITCH, ROLL, ANGLE_PITCH_EXTRA, DELTA_EAR, FOREHEAD_Y, DELTA_PITCH]
            # cheek_dist đã bị thay thế bằng DELTA_PITCH
            feats = [ear_l, ear_r, mar, yaw, pitch, roll,
                     angle_pitch_extra, delta_ear_value, forehead_y, delta_pitch_value]

            X.append(feats)
            y.append(label_map[label])
            total_samples += 1
            label_counts[label] += 1

    print(f"=> Tổng mẫu cho '{label}': {total_samples}")

# ==============================
# CÂN BẰNG DỮ LIỆU
# ==============================
X, y = np.array(X, np.float32), np.array(y, np.int32)
if BALANCE_DATA:
    print("\n[⚖️] Đang cân bằng dữ liệu...")
    counts = {cls: int(np.sum(y == label_map[cls])) for cls in CLASSES}
    min_samples = min(counts[c] for c in counts if counts[c] > 0)

    if min_samples > 0:
        X_bal, y_bal = [], []
        for cls in CLASSES:
            mask = y == label_map[cls]
            if np.sum(mask) == 0:
                continue
            X_cls, y_cls = X[mask], y[mask]
            replace = X_cls.shape[0] < min_samples
            X_res, y_res = resample(X_cls, y_cls, n_samples=min_samples, replace=replace, random_state=RANDOM_SEED)
            X_bal.append(X_res)
            y_bal.append(y_res)
        X = np.vstack(X_bal)
        y = np.concatenate(y_bal)
    else:
        print("[!] Không thể cân bằng dữ liệu vì tất cả các lớp đều không có mẫu.")

# ==============================
# LƯU FILE
# ==============================
print("\n========== TỔNG KẾ ==========")
for lbl, cnt in label_counts.items():
    print(f" - {lbl:<8}: {cnt} mẫu (Gốc)")

print(f"Tổng cộng (sau cân bằng): {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")
print(f"Số lượng đặc trưng trích xuất: {X.shape[1]} (Phải là 10)")

assert not np.isnan(X).any(), "❌ Dữ liệu có NaN!"
assert not np.isinf(X).any(), "❌ Dữ liệu có vô cực!"

with open(LABEL_MAP_FILE, "w") as f:
    json.dump(label_map, f, indent=2)

np.savez(OUTPUT_FILE, X=X, y=y)
print(f"\n✅ Đã lưu '{OUTPUT_FILE}' và '{LABEL_MAP_FILE}'")

face_mesh.close()