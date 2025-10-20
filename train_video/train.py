import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import joblib

# ============================
# 1. Load dữ liệu
# ============================
data = np.load("features_aug.npz")
X = data["X"]
y = data["y"]

n_samples, n_features = X.shape
n_classes = len(np.unique(y))
print(f"Dữ liệu: {n_samples} mẫu, {n_features} đặc trưng, {n_classes} lớp")

# ============================
# 2. Chuẩn hóa dữ liệu
# ============================
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X = (X - X_mean) / X_std

# ============================
# 3. Chia train / test
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

# One-hot encoding cho nhãn
def one_hot(y, num_classes):
    y_onehot = np.zeros((len(y), num_classes))
    y_onehot[np.arange(len(y)), y] = 1
    return y_onehot

y_train_onehot = one_hot(y_train, n_classes)
y_test_onehot = one_hot(y_test, n_classes)

# ============================
# 4. Hàm softmax
# ============================
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)  # tránh tràn số
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ============================
# 5. Khởi tạo tham số
# ============================
np.random.seed(42)
W = np.random.randn(n_features, n_classes) * 0.01
b = np.zeros((1, n_classes))

lr = 0.1           # learning rate
num_epochs = 500   # số vòng lặp huấn luyện
lambda_reg = 0.001 # hệ số regularization L2

train_loss_list, test_loss_list = [], []
train_acc_list, test_acc_list = [], []

# ============================
# 6. Vòng lặp huấn luyện
# ============================
for epoch in range(num_epochs):
    # --- Forward ---
    logits_train = X_train @ W + b
    y_pred_train = softmax(logits_train)

    logits_test = X_test @ W + b
    y_pred_test = softmax(logits_test)

    # --- Loss có Regularization ---
    loss_train = -np.mean(y_train_onehot * np.log(y_pred_train + 1e-8)) + lambda_reg * np.sum(W**2)
    loss_test = -np.mean(y_test_onehot * np.log(y_pred_test + 1e-8)) + lambda_reg * np.sum(W**2)

    # --- Accuracy ---
    train_acc = np.mean(np.argmax(y_pred_train, axis=1) == y_train)
    test_acc = np.mean(np.argmax(y_pred_test, axis=1) == y_test)

    # --- Gradient ---
    dW = (X_train.T @ (y_pred_train - y_train_onehot)) / len(X_train)
    dW += 2 * lambda_reg * W  # thêm phần regularization
    db = np.mean(y_pred_train - y_train_onehot, axis=0, keepdims=True)

    # --- Update trọng số ---
    W -= lr * dW
    b -= lr * db

    # --- Lưu lại kết quả ---
    train_loss_list.append(loss_train)
    test_loss_list.append(loss_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    # --- In thông tin ---
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"TrainLoss: {loss_train:.4f} | TestLoss: {loss_test:.4f} | "
              f"TrainAcc: {train_acc:.4f} | TestAcc: {test_acc:.4f}")

# ============================
# 7. Kết quả cuối cùng
# ============================
print("\n✅ Training hoàn tất!")
print(f"Best Train Acc: {max(train_acc_list):.4f}")
print(f"Best Test Acc:  {max(test_acc_list):.4f}")

# ============================
# 8. Vẽ biểu đồ
# ============================
plt.figure(figsize=(12, 5))

# ---- Biểu đồ Loss ----
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Biểu đồ Loss")
plt.legend()
plt.grid(True)

# ---- Biểu đồ Accuracy ----
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Biểu đồ Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================
# 9. Đánh giá mô hình
# ============================
y_pred_final = np.argmax(X_test @ W + b, axis=1)
overall_acc = np.mean(y_pred_final == y_test)
print(f"\n🎯 Overall Accuracy: {overall_acc:.4f}")

# Confusion Matrix
CLASSES = ["left", "right", "yawn", "blink", "normal","nod"]  # thay tên theo dataset của bạn
cm = confusion_matrix(y_test, y_pred_final)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()

# Báo cáo chi tiết
print("\n🔎 Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=CLASSES))

# ============================
# 10. Lưu model & scaler
# ============================
model_data = {
    "W": W,
    "b": b,
    "classes": CLASSES
}
joblib.dump(model_data, "softmax_model_best.pkl")
print("✅ Đã lưu model vào 'softmax_model_best.pkl'")

scaler = {
    "X_mean": X_mean,
    "X_std": X_std
}
joblib.dump(scaler, "scale.pkl")
print("✅ Đã lưu scaler vào 'scale.pkl'")
