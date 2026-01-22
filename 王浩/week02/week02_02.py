import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数的模拟数据
# 生成在[0, 2π]范围内的均匀分布数据，增加噪声让拟合更贴近真实场景
X_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)  # (200, 1)
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)  # 加噪声

# 转换为torch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成，数据形状：")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print("---" * 10)


# 2. 定义多层神经网络模型
class SinFittingNet(nn.Module):
    def __init__(self):
        super(SinFittingNet, self).__init__()
        # 构建多层全连接网络：输入层(1) -> 隐藏层1(32) -> 隐藏层2(16) -> 输出层(1)
        self.layers = nn.Sequential(
            nn.Linear(1, 32),  # 输入层：1个特征（x值）映射到32维隐藏特征
            nn.ReLU(),  # 激活函数，引入非线性
            nn.Linear(32, 16),  # 隐藏层1：32维映射到16维
            nn.ReLU(),  # 激活函数
            nn.Linear(16, 1)  # 输出层：16维映射到1个输出（sin(x)预测值）
        )

    def forward(self, x):
        # 前向传播
        return self.layers(x)


# 初始化模型、损失函数、优化器
model = SinFittingNet()
loss_fn = nn.MSELoss()  # 回归任务仍使用均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam优化器收敛更快

# 3. 训练模型
num_epochs = 1000
loss_history = []  # 记录损失变化

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录损失
    loss_history.append(loss.item())

    # 每50个epoch打印一次
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 4. 模型预测（关闭梯度计算加速）
with torch.no_grad():
    y_pred = model(X).numpy()  # 转换为numpy数组用于绘图

# 5. 可视化结果
plt.figure(figsize=(12, 8))

# 子图1：损失变化曲线
plt.subplot(2, 1, 1)
plt.plot(loss_history, color='orange', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)

# 子图2：sin函数拟合结果
plt.subplot(2, 1, 2)
plt.scatter(X_numpy, y_numpy, label='Raw Data (with noise)', color='blue', alpha=0.5, s=10)
plt.plot(X_numpy, y_pred, label='Fitted Curve (MLP)', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='True sin(x)', color='green', linestyle='--', linewidth=2)
plt.xlabel('X (0 to 2π)')
plt.ylabel('y')
plt.title('MLP Fitting sin(x) Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()  # 自动调整子图间距
plt.show()

# 打印模型结构
print("\n--- Model Structure ---")
print(model)