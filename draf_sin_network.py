import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
# Generate synthetic data
np.random.seed(0)
torch.manual_seed(0)

# Class 0: Gaussian distribution centered at (-2, -2)
class_0 = np.random.randn(1000, 2) - 2

# Class 1: Gaussian distribution centered at (2, 2)
class_1 = np.random.randn(1000, 2) + 2

from sklearn.mixture import GaussianMixture

n_samples = 1000

# Define mixture components with elliptical covariance matrices
cov1 = np.array([[1.0, 0.8], [0.8, 1.0]])  # Elliptical covariance matrix
cov2 = np.array([[1.0, -0.6], [-0.6, 1.0]])  # Elliptical covariance matrix
cov3 = np.array([[1.0, 0.0], [0.0, 0.1]])  # More elongated covariance matrix

means = np.array([[-2, 0], [0, 3], [3, -2]])

# Fit GaussianMixture model
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.means_init = means
gmm.covariances_init = [cov1, cov2, cov3]
gmm.fit(np.concatenate([np.random.multivariate_normal(means[i], cov, int(n_samples/3)) for i, cov in enumerate([cov1, cov2, cov3])]))

# Generate samples
class_2, _ = gmm.sample(n_samples)

# Combine the data
data = np.vstack((class_0, class_2)).astype(np.float32)
labels = np.hstack((np.zeros(1000), np.ones(1000))).astype(np.float32)

# Convert to PyTorch tensors
data_tensor = torch.tensor(data)
data_tensor = data_tensor/(np.pi*2) + np.pi
labels_tensor = torch.tensor(labels).unsqueeze(1)


class SinLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SinLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize the learnable parameter theta
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.theta, a=0.0, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        # Perform the SinLinear operation
        # x: input tensor (batch_size, in_features)
        
        # Perform batched matrix multiplication between input and theta
        #z = torch.matmul(x, self.theta.t())
        
        output =  torch.sin(x.unsqueeze(1) + self.theta.unsqueeze(0))
        return output
    
    
    

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        #self.fc1 = nn.Linear(2, 4)  # First fully connected layer
        #self.fc2 = nn.Linear(4, 1)  # Second fully connected layer (output layer)
        
        self.fc1 = SinLinear(2,3)  # First fully connected layer
        self.fc2 = SinLinear(2, 1)  # Second fully connected layer (output layer)

    def forward(self, x):
        #x = torch.relu(self.fc1(x))  # Apply ReLU activation after first layer
        #x = torch.sigmoid(self.fc2(x))  # Apply Sigmoid activation for binary classification
        x = self.fc1(x) 
        x = self.fc2(x)
        return x

# Create an instance of the model
model = SimpleNN()

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 2000
batch_size = 64

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(data_tensor.size()[0])
    running_loss = 0.0

    for i in range(0, data_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = data_tensor[indices], labels_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_tensor):.4f}")


model.eval()
with torch.no_grad():
    outputs = model(data_tensor)
    predictions = (outputs > 0.5).float()
    accuracy = (predictions == labels_tensor).float().mean()
    print(f'Accuracy: {accuracy:.2f}')

# Visualization of the decision boundary
h = 0.02
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

model.eval()
with torch.no_grad():
    probs = model(grid).reshape(xx.shape)

plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], cmap='RdYlBu', alpha=0.6)
plt.scatter(class_0[:, 0], class_0[:, 1], label='Class 0', edgecolor='k')
plt.scatter(class_2[:, 0], class_2[:, 1], label='Class 1', edgecolor='k')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Function to generate half-moon shape
def generate_half_moon(radius, thickness, distance, n_samples):
    theta = np.linspace(np.pi, 0, n_samples)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta) + distance
    moon = np.vstack((x, y)).T
    noise = np.random.normal(scale=thickness, size=moon.shape)
    moon += noise
    return moon

# Generate half-moon shapes for two classes
n_samples = 1000
outer_moon = generate_half_moon(radius=10, thickness=2, distance=0, n_samples=n_samples)
inner_moon = generate_half_moon(radius=6, thickness=1, distance=0, n_samples=n_samples)

# Adjust positions of inner moon to nest inside outer moon
inner_moon[:, 1] -= 6  # Shift inner moon downwards

# Plot the nested half-moons
plt.figure(figsize=(8, 6))
plt.scatter(outer_moon[:, 0], outer_moon[:, 1], color='blue', s=10, label='Class 1 (Outer Moon)')
plt.scatter(inner_moon[:, 0], inner_moon[:, 1], color='red', s=10, label='Class 2 (Inner Moon)')
plt.title('Nested Half-Moons Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()


class_0 = outer_moon
class_2 = inner_moon





# Khởi tạo mẫu dữ liệu
batch_size = 4
input_matrix = torch.randn(batch_size, 2)  # Input matrix of size [batch_size, 2]
theta_matrix = torch.randn(3, 2)           # Theta matrix of size [3, 2]

# Sử dụng broadcasting để thực hiện phép cộng
result =  theta_matrix.unsqueeze(0) + input_matrix.unsqueeze(1)

# Kiểm tra kết quả
print(result.shape)  # Output shape: [3, batch_size, 2]

# In ra một vài giá trị để kiểm tra
print(result[0])     # In ra kết quả input + theta[0,:]
print(result[1])     # In ra kết quả input + theta[1,:]
print(result[2])     # In ra kết quả input + theta[2,:]




input_matrix + theta_matrix[0,:]

a = theta_matrix.unsqueeze(1) 




sinlayer = SinLinear(2,3)
out = sinlayer(input_matrix)


import sympy as sp

# Khai báo các biến và hàm
x = sp.symbols('x')
a, b, c  = sp.symbols('a b c')
f1 = sp.sin(a * x + b)
f2 = c*sp.sin(a * x + b)
f = f1 + f2
f = (c+1)*sp.sin(a*x+b)
# Tính đạo hàm của hàm f theo x
f_prime = sp.diff(f, x)

# Hiển thị kết quả
print(f_prime)


import numpy as np
import matplotlib.pyplot as plt

# Định nghĩa các hệ số a và b
a = 1  # Bạn có thể thay đổi giá trị của a
b = 0  # Bạn có thể thay đổi giá trị của b

# Tạo mảng các giá trị x
x = np.linspace(-np.pi, np.pi, 100)

# Tính giá trị của hàm f(x) = sin(ax + b)
y = np.sin(1*a * x)
y1 = np.sin(2*a * x)
y2 = np.sin(2.8*a * x)

#y2 = np.sin(a * x + np.pi/2)


# Vẽ đồ thị
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f'sin({a}x + {b})')
plt.plot(x, y1, label=f'sin({a}x + {b})')
plt.plot(x, y2, label=f'sin({a}x + {b+1})')

plt.title('Đồ thị của hàm f(x) = sin(ax + b)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()














