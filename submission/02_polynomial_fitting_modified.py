import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Ensure outputs directory exists
output_dir = 'submission/outputs(4x-7)'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=" * 50)
print("Numerical Methods vs Neural Networks")
print("Changed formula: y = 4x - 7")
print("=" * 50)

# ================================================================
# 원본(y = 2x - 1)에서 y = 4x - 7 로 변경하여 실험
# 세 방법(Polyfit, Curve Fit, 신경망)이 모두 w≈4, b≈-7을 찾는지 확인
# ================================================================

# 1. 데이터 준비 (01_hello_nn_modified.py와 동일)
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_clean = np.array([-11.0, -7.0, -3.0, 1.0, 5.0, 9.0], dtype=float)  # 4x - 7

np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=1.0, size=len(X))
y = y_clean + noise

print("Data (y = 4x - 7):")
print(f"X:        {X}")
print(f"y (clean): {y_clean}")
print(f"y (noisy): {y}")
print("-" * 50)

new_x = 10.0
expected = 4 * new_x - 7  # 33.0

# ---------------------------------------------------------
# Method 1: NumPy Polyfit (해석적 최소자승법)
# ---------------------------------------------------------
print("\n[Method 1] NumPy Polyfit (Least Squares)")
coefficients = np.polyfit(X, y, deg=1)
slope_poly = coefficients[0]
intercept_poly = coefficients[1]

pred_poly = slope_poly * new_x + intercept_poly
print(f"Result:     y = {slope_poly:.4f}x + {intercept_poly:.4f}")
print(f"Expected:   y = 4.0000x + -7.0000")
print(f"Prediction for x={new_x}: {pred_poly:.4f}  (Expected: {expected})")

# ---------------------------------------------------------
# Method 2: SciPy Curve Fit (수치 최적화)
# ---------------------------------------------------------
print("\n[Method 2] SciPy Curve Fit (Optimization)")

def linear_function(x, w, b):
    return w * x + b

popt, pcov = curve_fit(linear_function, X, y, p0=[0.5, 0.5])
w_opt, b_opt = popt

pred_scipy = linear_function(new_x, w_opt, b_opt)
print(f"Result:     y = {w_opt:.4f}x + {b_opt:.4f}")
print(f"Prediction for x={new_x}: {pred_scipy:.4f}  (Expected: {expected})")

# ---------------------------------------------------------
# 노이즈 크기 변화 실험 (scale 변경)
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("Noise Experiment: scale 변화에 따른 영향")
print("=" * 50)

results = {}
for scale in [0.1, 1.0, 5.0]:
    np.random.seed(42)
    noisy_y = y_clean + np.random.normal(0, scale, size=len(X))
    coef = np.polyfit(X, noisy_y, deg=1)
    results[scale] = {'w': coef[0], 'b': coef[1]}
    print(f"scale={scale:.1f} → w={coef[0]:.4f} (expected 4.0), b={coef[1]:.4f} (expected -7.0)")

# ---------------------------------------------------------
# 시각화 1: 세 방법 비교
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Noisy Data (scale=1.0)', s=100, zorder=5)
plt.plot(X, y_clean, 'k:', label='True Function (y=4x-7)', alpha=0.6)

x_range = np.linspace(-2, 11, 100)
y_poly_line = slope_poly * x_range + intercept_poly
y_scipy_line = linear_function(x_range, w_opt, b_opt)

plt.plot(x_range, y_poly_line, label=f'Polyfit:   y={slope_poly:.2f}x+{intercept_poly:.2f}', color='blue', linestyle='--')
plt.plot(x_range, y_scipy_line, label=f'CurveFit: y={w_opt:.2f}x+{b_opt:.2f}', color='orange', linestyle='-.')
plt.scatter([new_x], [pred_poly], color='green', marker='*', s=200, label=f'Prediction x={new_x}', zorder=5)

plt.title('Method Comparison: y = 4x - 7')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, '02_method_comparison_y=4x-7.png'))
print(f"\nComparison plot saved.")

# ---------------------------------------------------------
# 시각화 2: 노이즈 크기 비교
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
scales = [0.1, 1.0, 5.0]

for ax, scale in zip(axes, scales):
    np.random.seed(42)
    noisy_y = y_clean + np.random.normal(0, scale, size=len(X))
    coef = np.polyfit(X, noisy_y, deg=1)

    ax.scatter(X, noisy_y, color='red', s=80, label='Noisy Data')
    ax.plot(X, y_clean, 'k:', alpha=0.5, label='True (y=4x-7)')
    ax.plot(X, coef[0] * X + coef[1], 'b--', label=f'Fit: y={coef[0]:.2f}x+{coef[1]:.2f}')
    ax.set_title(f'Noise scale = {scale}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend(fontsize=7)
    ax.grid(True)

plt.suptitle('Effect of Noise on Polyfit (y = 4x - 7)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_noise_comparison_y=4x-7.png'))
print("Noise comparison plot saved.")

# ---------------------------------------------------------
# 최종 요약
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("Summary")
print("=" * 50)
print(f"{'Method':<20} {'w (expected 4.0)':<22} {'b (expected -7.0)':<20} {'pred x=10 (exp 33.0)'}")
print("-" * 80)
print(f"{'NumPy Polyfit':<20} {slope_poly:<22.4f} {intercept_poly:<20.4f} {pred_poly:.4f}")
print(f"{'SciPy CurveFit':<20} {w_opt:<22.4f} {b_opt:<20.4f} {pred_scipy:.4f}")
