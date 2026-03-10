import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure outputs directory exists
output_dir = 'submission/outputs(4x-7)'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("TensorFlow Version:", tf.__version__)

# ================================================================
# 원본(y = 2x - 1)에서 y = 4x - 7 로 변경하여 실험
# ================================================================

# 1. 데이터 준비
# 학습할 관계: y = 4x - 7
# x: [-1, 0, 1, 2, 3, 4] → y: [-11, -7, -3, 1, 5, 9]
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_clean = np.array([-11.0, -7.0, -3.0, 1.0, 5.0, 9.0], dtype=float)  # 4x - 7

# 노이즈 추가 (scale=1.0)
np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=1.0, size=len(X))
y = y_clean + noise

print("\nTraining Data (y = 4x - 7):")
print("X:", X)
print("y (clean):", y_clean)
print("y (noisy):", y)

# 2. 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 3. 컴파일
model.compile(optimizer='sgd', loss='mean_squared_error')

# 4. 학습
print("\nStarting training...")
history = model.fit(X, y, epochs=500, verbose=0)
print("Training finished!")

# 5. 예측
new_x = 10.0
prediction = model.predict(np.array([[new_x]]))
print(f"\nPrediction for x={new_x}: {prediction[0][0]:.4f}")
print(f"Expected value: {4 * new_x - 7}")  # 33.0

# 6. 학습 가중치 확인
weights = model.get_weights()
w = weights[0][0][0]
b = weights[1][0]
print(f"\nLearned Parameters:")
print(f"Weight (w): {w:.4f} (Expected: 4.0)")
print(f"Bias (b):   {b:.4f} (Expected: -7.0)")
print(f"Formula: y = {w:.4f}x + {b:.4f}")

# 7. 시각화
# Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'])
plt.title('Model Training Loss (y = 4x - 7)')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.grid(True)
plt.savefig(os.path.join(output_dir, '01_training_loss_y=4x-7.png'))
print(f"\nLoss plot saved.")

# Model Fit Graph
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Noisy Data', s=100)
plt.plot(X, y_clean, 'k:', label='True Function (y=4x-7)', alpha=0.5)

x_range = np.linspace(-2, 5, 100)
y_pred = model.predict(x_range.reshape(-1, 1), verbose=0)
plt.plot(x_range, y_pred, label='Neural Network Fit', color='blue')

plt.title('Neural Network Regression: y = 4x - 7')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, '01_model_fit_y=4x-7.png'))
print("Model fit plot saved.")
