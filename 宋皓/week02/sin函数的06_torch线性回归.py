import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(0)
X_numpy = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)  # 在[-π, π]区间内均匀采样
y_numpy = np.sin(X_numpy)  # 计算对应的sin值
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

# 定义一个多层感知器 (MLP) 模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),  # 输入层到隐藏层
            nn.ReLU(),
            nn.Linear(64, 64),  # 隐藏层到隐藏层
            nn.ReLU(),
            nn.Linear(64, 1)  # 隐藏层到输出层
        )

    def forward(self, x):
        return self.layers(x)

# 初始化模型、损失函数和优化器
model = MLP()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam优化器

# 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 使用模型进行预测
with torch.no_grad():
    y_predicted = model(X).numpy()

# 绘制原始数据与模型预测
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label='Model prediction', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Fitting sin(x) with a Multi-layer Perceptron')
plt.show()
