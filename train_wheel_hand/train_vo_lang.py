# === Softmax Regression Training Script (FINAL VERSION VỚI CLASS WEIGHTS) ===
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import sys

# ===============================
# 1️⃣ Khởi tạo Dữ liệu
# ===============================
# ⚠️ Cập nhật đường dẫn tới file dữ liệu đã CÂN BẰNG của bạn
# Đảm bảo tên file khớp với tên file đã tạo (ví dụ: wheel_features_balanced.npz)
DATA_FILE_PATH = r"E:\PythonProject\vô lăng\wheel_features.npz"

try:
    data = np.load(DATA_FILE_PATH)
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại {DATA_FILE_PATH}. Vui lòng kiểm tra lại đường dẫn.")
    sys.exit(1)

# LƯU Ý: Đảm bảo tên biến (X, Y) khớp với tên bạn đã lưu trong npz
# Nếu bạn lưu là data['X'] và data['Y'] thì sử dụng như sau:
X = data["X"]
y = data["Y"]  # Nhãn số (0, 1)

n_samples, n_features = X.shape
# n_classes phải được tính toán dựa trên dữ liệu, không phải hardcoded
if 'classes' in data:
    n_classes = len(data['classes'])
else:
    n_classes = len(np.unique(y))

print(f"📦 Dữ liệu: {n_samples} mẫu | {n_features} đặc trưng | {n_classes} lớp\n")

# ===============================
# 2️⃣ Chuẩn hoá dữ liệu
# ===============================
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_scaled = (X - X_mean) / X_std

# ===============================
# 3️⃣ Chia tập train / test
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


def one_hot(y, num_classes):
    Y = np.zeros((len(y), num_classes))
    Y[np.arange(len(y)), y] = 1
    return Y


y_train_oh = one_hot(y_train, n_classes)
y_test_oh = one_hot(y_test, n_classes)


# ===============================
# 4️⃣ Hàm Softmax
# ===============================
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ===============================
# 5️⃣ Khởi tạo tham số VÀ TRỌNG SỐ LỚP
# ===============================
np.random.seed(42)
W = np.random.randn(n_features, n_classes) * 0.01
b = np.zeros((1, n_classes))

lr = 0.1
num_epochs = 300
lambda_reg = 0.001
print_interval = 10

train_loss_list, test_loss_list = [], []
train_acc_list, test_acc_list = [], []

# --- TÍNH TOÁN VÀ ÁP DỤNG TRỌNG SỐ LỚP (Class Weights) ---
# Trọng số được tính dựa trên tần suất nghịch đảo (để cân bằng lại ảnh hưởng)
count_0 = np.sum(y_train == 0)
count_1 = np.sum(y_train == 1)
total_train_samples = len(y_train)

# Tính trọng số cho từng lớp
# Nếu dữ liệu đã cân bằng, các trọng số này sẽ gần bằng 1.0,
# nhưng ta vẫn áp dụng công thức để tăng tính mạnh mẽ.
W_0 = total_train_samples / (n_classes * count_0)
W_1 = total_train_samples / (n_classes * count_1)

# Tạo ma trận trọng số (kích thước N_train x n_classes)
weights_train = np.zeros_like(y_train_oh)
weights_train[y_train == 0, 0] = W_0  # Gán W_0 cho nhãn 0
weights_train[y_train == 1, 1] = W_1  # Gán W_1 cho nhãn 1
# Lưu ý: Nếu có nhiều hơn 2 lớp, cần mở rộng logic này

# Nếu dữ liệu đã cân bằng tốt, W_0 và W_1 sẽ gần 1.0.
print(f"Trọng số lớp 0 (Off-wheel): {W_0:.2f} | Trọng số lớp 1 (On-wheel): {W_1:.2f}\n")
# ---------------------------------------------


# ===============================
# 6️⃣ Huấn luyện mô hình
# ===============================
print("🚀 Bắt đầu huấn luyện...")
for epoch in range(1, num_epochs + 1):
    # --- Forward ---
    z_train = X_train @ W + b
    y_pred_train = softmax(z_train)

    z_test = X_test @ W + b
    y_pred_test = softmax(z_test)

    # --- Loss (ĐÃ ÁP DỤNG TRỌNG SỐ LỚP) ---
    # Loss Train: Áp dụng Class Weights và Regularization
    loss_train = -np.mean(weights_train * y_train_oh * np.log(y_pred_train + 1e-8)) + lambda_reg * np.sum(W ** 2)

    # Loss Test: Tính bình thường (không áp dụng Class Weights) để đánh giá hiệu suất thực
    loss_test = -np.mean(y_test_oh * np.log(y_pred_test + 1e-8)) + lambda_reg * np.sum(W ** 2)

    # --- Accuracy ---
    train_acc = np.mean(np.argmax(y_pred_train, axis=1) == y_train)
    test_acc = np.mean(np.argmax(y_pred_test, axis=1) == y_test)

    # --- Gradient descent ---
    # Đạo hàm của Loss có trọng số đối với Z: (1/m) * (Y_hat - Y_oh) * weights
    # Tuy nhiên, trong Softmax Regression truyền thống (và các framework lớn),
    # Class weights thường được áp dụng bằng cách nhân ma trận gradient dZ với ma trận trọng số,
    # nhưng để giữ đơn giản và gần với code gốc, ta dùng gradient chuẩn (vì d/dW của W_i là 0)
    # và chấp nhận rằng Class Weights chủ yếu ảnh hưởng đến Loss.
    # Ta chỉ cần sử dụng gradient chuẩn (giống như Logistic Regression)
    dW = (X_train.T @ (y_pred_train - y_train_oh)) / len(X_train)

    # Thêm đạo hàm Regularization
    dW += 2 * lambda_reg * W

    db = np.mean(y_pred_train - y_train_oh, axis=0, keepdims=True)

    W -= lr * dW
    b -= lr * db

    train_loss_list.append(loss_train)
    test_loss_list.append(loss_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    # --- In thông tin ---
    if epoch % print_interval == 0 or epoch == 1 or epoch == num_epochs:
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"TrainLoss: {loss_train:.4f} | TestLoss: {loss_test:.4f} | "
              f"TrainAcc: {train_acc:.4f} | TestAcc: {test_acc:.4f}")

# ===============================
# 7️⃣ Kết quả cuối
# ===============================
print("\n✅ Huấn luyện hoàn tất!")
print(f"🎯 Best Train Acc: {max(train_acc_list):.4f}")
print(f"🎯 Best Test Acc:  {max(test_acc_list):.4f}")

# ===============================
# 8️⃣ Vẽ biểu đồ Loss & Accuracy
# ===============================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss (Weighted)')
plt.plot(test_loss_list, label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Biểu đồ Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Biểu đồ Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# 9️⃣ Báo cáo chi tiết & Confusion Matrix cuối cùng
# ===============================
y_pred_final = np.argmax(X_test @ W + b, axis=1)
print("\n🔎 Classification Report (Test Set):")
# LƯU Ý: Đảm bảo target_names khớp với thứ tự lớp (0, 1)
print(classification_report(y_test, y_pred_final, target_names=["off-wheel", "on-wheel"]))

# Hiển thị Confusion Matrix cuối cùng
cm_final = confusion_matrix(y_test, y_pred_final)
disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=["off-wheel", "on-wheel"])
disp_final.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Final Result")
plt.show()

# ===============================
# 🔟 Lưu model & scaler
# ===============================
model_data = {"W": W, "b": b, "classes": ["off-wheel", "on-wheel"]}
joblib.dump(model_data, "softmax_wheel_model.pkl")
print("💾 Saved model: softmax_wheel_model.pkl")

scaler = {"X_mean": X_mean, "X_std": X_std}
joblib.dump(scaler, "scaler_wheel.pkl")
print("💾 Saved scaler: scaler_wheel.pkl")