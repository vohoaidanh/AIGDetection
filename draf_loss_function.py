# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def mean_square_error(y_true, y_pred):
    """
    Tính Mean Square Error giữa các giá trị thực và giá trị dự đoán.

    Args:
    - y_true (numpy array): Mảng chứa các giá trị thực tế.
    - y_pred (numpy array): Mảng chứa các giá trị dự đoán.

    Returns:
    - float: Giá trị MSE.
    """
    n = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / n
    return mse


# Số lượng điểm dữ liệu
num_points = 100

# Tạo dữ liệu ngẫu nhiên cho x
np.random.seed(0)  # Để có kết quả như nhau mỗi lần chạy
x = np.random.rand(num_points) * 10  # x nằm trong khoảng từ 0 đến 10
noise = (np.random.rand(num_points) - 0.5) * 5
# Tính giá trị y tương ứng
y_true = (0*x**2 + 4 * x + 11) + noise
#y_pred = 2*x + 4

#plt.plot(x, y_true, 'b.')
#plt.plot(x, y_pred, 'g.')

#loss = mean_square_error(y_true, y_pred)
#print(f'Mean Square Error: {loss}')


a_values = np.linspace(-50, 50, 101)  # 10 giá trị từ 2 đến 4
b_values = np.linspace(-50, 50, 101)  # 10 giá trị từ 1 đến 6

A, B = np.meshgrid(a_values, b_values)


mse_values = np.zeros((len(a_values), len(b_values)))
plt.figure(figsize=(12, 8))


for i in range(len(a_values)):
    for j in range(len(b_values)):
        y_pred = A[i, j] * x + B[i, j]
        mse_values[i, j] = mean_square_error(y_true, y_pred)

# Vẽ biểu đồ contour của MSE
plt.figure(figsize=(10, 6))
plt.contourf(A, B, mse_values, levels=20, cmap='viridis')
plt.colorbar(label='MSE')
plt.title('Mean Square Error Contour Plot')
plt.xlabel('a')
plt.ylabel('b')
plt.show()


# Tìm vị trí (index) của giá trị MSE nhỏ nhất
min_idx = np.unravel_index(np.argmin(mse_values), mse_values.shape)
best_a = A[min_idx]
best_b = B[min_idx]
min_mse = mse_values[min_idx]

print(f"Best parameters: a = {best_a:.2f}, b = {best_b:.2f}")
print(f"Minimum MSE: {min_mse:.4f}")



# Vẽ biểu đồ 3D của MSE
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(A, B, mse_values, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, label='MSE')
ax.set_title('Mean Square Error in 3D')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
plt.show()



import numpy as np

def generate_orthogonal_vectors(n):
    # Tạo vector ngẫu nhiên
    v1 = np.random.randn(n)
    
    # Chuẩn hóa vector đầu tiên
    v1_normalized = v1 / np.linalg.norm(v1)
    
    # Tạo vector ngẫu nhiên thứ hai
    v2 = np.random.randn(n)
    
    # Chiếu v2 để nó trực giao với v1
    v2 -= np.dot(v2, v1_normalized) * v1_normalized
    v2_normalized = v2 / np.linalg.norm(v2)
    
    return v1_normalized, v2_normalized

# Sử dụng hàm để tạo hai vector trực giao trong không gian 3 chiều
n_dim = 100
v1, v2 = generate_orthogonal_vectors(n_dim)

print("Vector 1:", v1)
print("Vector 2:", v2)
print("Inner product (should be close to 0):", np.dot(v1, v2))

np.dot(v1, v2)

np.sum(v2**2)


np.array([3,4,0]).dot(np.array([0,4,5]))




