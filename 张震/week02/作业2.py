import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)  # 生成-2π到2π的数据

y_numpy = np.sin(X_numpy) + np.random.randn(1000, 1) * 0.1

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

class ZSin(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ZSin, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size)
        )

    def forward(self, x):
        return self.network(x)

model = ZSin(1, 64, 1)

# 定义损失函数 (均方误差)
loss_fn = nn.MSELoss()

# 定义优化器 (随机梯度下降)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

num_epochs = 5000
for epoch in range(num_epochs):
    model.train()

    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    y_pred = model(X).numpy()


plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy)
plt.scatter(X_numpy, y_pred)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()