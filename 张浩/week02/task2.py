# 2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据：y = sin(x) + 噪声
x_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)  # (200, 1)
y_numpy = np.sin(x_numpy) + 0.1 * np.random.randn(*x_numpy.shape)  # 加入小噪声

X = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络模型（使用 nn.Module）
class SinRegressor(torch.nn.Module):
    def __init__(self):
        super(SinRegressor, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SinRegressor()
print("模型结构：")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 可视化结果
model.eval()
with torch.no_grad():
    y_pred_final = model(X).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(x_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6, s=15)
plt.plot(x_numpy, np.sin(x_numpy), label='sin(x)', color='red', linewidth=2, linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
