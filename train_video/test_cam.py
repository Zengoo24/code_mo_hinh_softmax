import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import time
import json

# ==============================
# CẤU HÌNH
# ==============================
MODEL_PATH = "softmax_model_best.pkl"
SCALER_PATH = "scale.pkl"
SMOOTH_WINDOW = 8
CONF_THRESHOLD = 0.5
FPS_SMOOTH = 0.9

# ==============================
# LOAD MODEL
# ==============================
print("🔹 Đang load model softmax...")
model_data = joblib.load(MODEL_PATH)
W = model_data["W"]
b = model_data["b"]
CLASSES = model_data["classes"]

scaler_data = joblib.load(SCALER_PATH)
X_mean = scaler_data["X_mean"]
X_std = scaler_data["X_std"]

idx2label = {i: lbl for i, lbl in enumerate(CLASSES)}
print("✅ Model:", CLASSES)

# ==============================
# HÀM DỰ ĐOÁN
# ==============================
def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def predict_proba(x):
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
# HÀM TÍNH ĐẶC TRƯNG
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
    pitch = np.degrees(np.arctan2((nose[1] - eyes_center[1]), (np.linalg.norm(baseline) + EPS)))
    return yaw, pitch, roll

# ==============================
# VÒNG LẶP CAMERA
# ==============================
cap = cv2.VideoCapture(0)
frame_queue = deque(maxlen=30)
pred_queue = deque(maxlen=SMOOTH_WINDOW)

pTime = 0
fps = 0
print("📷 Bắt đầu camera (nhấn Q để thoát)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # <--- Thêm dòng này (lật ngang)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in face.landmark])

        ear_l = eye_aspect_ratio(landmarks, True)
        ear_r = eye_aspect_ratio(landmarks, False)
        mar = mouth_aspect_ratio(landmarks)
        yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)

        nose, chin = landmarks[1], landmarks[152]
        angle_pitch_extra = np.degrees(np.arctan2(chin[1] - nose[1], (chin[2] - nose[2]) + EPS))
        forehead_y = np.mean(landmarks[[10, 338, 297, 332, 284], 1])
        cheek_dist = np.linalg.norm(landmarks[50] - landmarks[280])

        feat = np.array([ear_l, ear_r, mar, yaw, pitch, roll,
                         angle_pitch_extra, forehead_y, cheek_dist], dtype=np.float32)

        frame_queue.append(feat)

        if len(frame_queue) == 30:
            window = np.array(frame_queue)
            mean_feats = window.mean(axis=0)
            std_feats = window.std(axis=0)
            yaw_diff = np.mean(np.abs(np.diff(window[:, 3])))
            pitch_diff = np.mean(np.abs(np.diff(window[:, 4])))
            roll_diff = np.mean(np.abs(np.diff(window[:, 5])))
            mar_max = np.max(window[:, 2])
            mar_mean = np.mean(window[:, 2])
            ear_mean = np.mean((window[:, 0] + window[:, 1]) / 2.0)
            mar_ear_ratio = mar_mean / (ear_mean + EPS)
            yaw_pitch_ratio = np.mean(np.abs(window[:, 3])) / (np.mean(np.abs(window[:, 4])) + EPS)

            feats = np.concatenate([mean_feats, std_feats,
                                    [yaw_diff, pitch_diff, roll_diff,
                                     mar_max, mar_ear_ratio, yaw_pitch_ratio]])

            # Chuẩn hóa
            feats_scaled = (feats - X_mean) / (X_std + EPS)

            # Dự đoán Softmax
            probs = predict_proba(feats_scaled)
            probs = np.array(probs).flatten()  # đảm bảo thành mảng 1 chiều
            pred_idx = np.argmax(probs)
            pred_conf = float(probs[pred_idx])  # Lấy xác suất cao nhất (dạng float)
            pred_label = idx2label[pred_idx]

            # Chỉ thêm vào queue nếu xác suất đủ cao
            if pred_conf > CONF_THRESHOLD:
                pred_queue.append(pred_label)

    # ======== SMOOTH PREDICTION ========
    if len(pred_queue) > 0:
        final_label = max(set(pred_queue), key=pred_queue.count)
    else:
        final_label = "unknown"

    # ======== HIỂN THỊ ========
    cTime = time.time()
    fps = FPS_SMOOTH * fps + (1 - FPS_SMOOTH) * (1 / (cTime - pTime + EPS))
    pTime = cTime

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"State: {final_label}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)

    cv2.imshow("Softmax Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
print("🛑 Kết thúc.")
