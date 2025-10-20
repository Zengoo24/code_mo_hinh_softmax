import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import time
import json
import warnings
import os
import sys

# ==============================
# CẤU HÌNH
# ==============================
MODEL_PATH = "softmax_model_best1.pkl"  # Cập nhật tên file model mới
SCALER_PATH = "scale1.pkl"  # Cập nhật tên file scaler mới
SMOOTH_WINDOW = 5  # Tăng lên 5 để ổn định Nod và các hành vi khác
BLINK_THRESHOLD = 0.20 #Luật cứng cho Blink
EPS = 1e-8
FPS_SMOOTH = 0.9
N_FEATURES = 10  # SỐ LƯỢNG ĐẶC TRƯNG MONG ĐỢI: 10

# ==============================
# LOAD MODEL VÀ SCALER
# ==============================
try:
    print("🔹 Đang load model softmax...")
    model_data = joblib.load(MODEL_PATH)
    W = model_data["W"]
    b = model_data["b"]
    CLASSES = model_data["classes"]

    scaler_data = joblib.load(SCALER_PATH)
    X_mean = scaler_data["X_mean"]
    X_std = scaler_data["X_std"]

    idx2label = {i: lbl for i, lbl in enumerate(CLASSES)}
    print(f"✅ Model: {CLASSES} ({W.shape[1]} classes)")

    if W.shape[0] != N_FEATURES:
        warnings.warn(
            f"Lỗi: Kích thước đặc trưng của mô hình ({W.shape[0]}) không khớp với số đặc trưng ({N_FEATURES}) được trích xuất trong code.")
        sys.exit()

except FileNotFoundError as e:
    print(f"LỖI: Không tìm thấy file tài nguyên. Vui lòng kiểm tra đường dẫn: {e}")
    sys.exit()
except Exception as e:
    print(f"LỖI LOAD DỮ LIỆU: {e}")
    sys.exit()


# ==============================
# HÀM DỰ ĐOÁN
# ==============================
def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def predict_proba(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    z = np.dot(x, W) + b
    return softmax(z)


# ==============================
# FACE MESH KHỞI TẠO
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# HÀM TÍNH ĐẶC TRƯNG (10 Đặc trưng)
# ==============================
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
    pitch = np.degrees(np.arctan2((nose[1] - eyes_center[1]), (np.linalg.norm(baseline) + EPS)))
    return yaw, pitch, roll


def get_extra_features(landmarks):
    nose, chin = landmarks[1], landmarks[152]
    angle_pitch_extra = np.degrees(np.arctan2(chin[1] - nose[1], (chin[2] - nose[2]) + EPS))
    forehead_y = np.mean(landmarks[[10, 338, 297, 332, 284], 1])
    cheek_dist = np.linalg.norm(landmarks[50] - landmarks[280])
    # Chỉ trả về angle_pitch_extra và forehead_y (Vì cheek_dist đã bị thay thế)
    return angle_pitch_extra, forehead_y, cheek_dist


# ==============================
# VÒNG LẶP CAMERA CHÍNH
# ==============================
cap = cv2.VideoCapture(0)
pred_queue = deque(maxlen=SMOOTH_WINDOW)
last_ear_avg = 0.4  # Lịch sử EAR ban đầu cho Delta EAR
last_pitch = 0.0  # Lịch sử Pitch ban đầu cho Delta Pitch

pTime = 0
fps = 0
print("📷 Bắt đầu camera (nhấn Q để thoát)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Lật ngang để tạo hiệu ứng gương
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_pred_label = "unknown"
    delta_ear_value = 0.0
    delta_pitch_value = 0.0

    if results.multi_face_landmarks:
        landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in results.multi_face_landmarks[0].landmark])

        # 1. TÍNH TOÁN CÁC ĐẶC TRƯNG TĨNH
        ear_l = eye_aspect_ratio(landmarks, True)
        ear_r = eye_aspect_ratio(landmarks, False)
        mar = mouth_aspect_ratio(landmarks)
        yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)

        angle_pitch_extra, forehead_y, cheek_dist = get_extra_features(
            landmarks)  # Tính nhưng chỉ dùng angle_pitch_extra và forehead_y

        # 2. TÍNH ĐẶC TRƯNG ĐỘNG (Delta EAR và Delta Pitch)
        ear_avg = (ear_l + ear_r) / 2.0
        delta_ear_value = ear_avg - last_ear_avg
        delta_pitch_value = pitch - last_pitch  # Đặc trưng động cho NOD

        # 3. CẬP NHẬT LỊCH SỬ
        last_ear_avg = ear_avg
        last_pitch = pitch

        # 4. ÁP DỤNG LUẬT CỨNG (HEURISTIC) CHO BLINK
        if ear_avg < BLINK_THRESHOLD:
            current_pred_label = "blink"
        else:
            # Dùng Softmax cho các hành vi khác (bao gồm Nod)

            # Mảng 10 đặc trưng: [EAR_L, EAR_R, MAR, YAW, PITCH, ROLL, ANGLE_PITCH_EXTRA, DELTA_EAR, FOREHEAD_Y, DELTA_PITCH]
            feats = np.array([ear_l, ear_r, mar, yaw, pitch, roll,
                              angle_pitch_extra, delta_ear_value, forehead_y, delta_pitch_value], dtype=np.float32)

            # CHUẨN HÓA
            feats_scaled = (feats - X_mean[:N_FEATURES]) / (X_std[:N_FEATURES] + EPS)

            # DỰ ĐOÁN Softmax
            probs = predict_proba(feats_scaled)
            probs = np.array(probs).flatten()
            pred_idx = np.argmax(probs)
            current_pred_label = idx2label[pred_idx]

        # LÀM MƯỢT KẾT QUẢ
        pred_queue.append(current_pred_label)

    else:
        # Nếu mất mặt, reset lịch sử EAR và Pitch
        last_ear_avg = 0.4
        last_pitch = 0.0

    # ======== SMOOTH PREDICTION ========
    if len(pred_queue) > 0:
        final_label = max(set(pred_queue), key=pred_queue.count)
    else:
        final_label = "unknown"

    # ======== HIỂN THỊ ========
    cTime = time.time()
    fps = FPS_SMOOTH * fps + (1 - FPS_SMOOTH) * (1 / (cTime - pTime + EPS))
    pTime = cTime

    # Hiển thị FPS và Trạng thái
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"State: {final_label.upper()}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)
    cv2.putText(frame, f"Delta EAR: {delta_ear_value:.3f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f"Delta Pitch: {delta_pitch_value:.3f}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Blink Thresh: <{BLINK_THRESHOLD}", (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Softmax Driver Monitor (10 Feats)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("🛑 Kết thúc.")