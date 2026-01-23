import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# 定义神经网络模型
class SinNet(nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        # 多层感知机，用于拟合sin函数
        self.hidden1 = nn.Linear(1, 50)  # 输入层到第一个隐藏层
        self.hidden2 = nn.Linear(50, 30)  # 第一个隐藏层到第二个隐藏层
        self.hidden3 = nn.Linear(30, 10)  # 第二个隐藏层到第三个隐藏层
        self.output = nn.Linear(10, 1)  # 输出层

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.output(x)
        return x


# 生成训练数据
def generate_data(n_samples=1000):
    # 在[-2π, 2π]范围内生成随机x值
    X = np.random.uniform(-2 * np.pi, 2 * np.pi, (n_samples, 1)).astype(np.float32)
    y = np.sin(X) + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)  # 加入少量噪声

    return torch.from_numpy(X), torch.from_numpy(y)


# 训练函数
def train_model(model, X_train, y_train, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam优化器

    losses = []

    for epoch in range(epochs):
        # 前向传播
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return losses


# 主程序
if __name__ == "__main__":
    # 生成训练数据
    X_train, y_train = generate_data(1000)

    # 创建模型实例
    model = SinNet()

    # 训练模型
    print("开始训练模型...")
    losses = train_model(model, X_train, y_train, epochs=1000, lr=0.01)

    # 生成测试数据进行预测
    X_test = np.linspace(-2 * np.pi, 2 * np.pi, 400).reshape(-1, 1).astype(np.float32)
    X_test_tensor = torch.from_numpy(X_test)

    # 模型预测
    with torch.no_grad():
        model.eval()
        y_pred = model(X_test_tensor).numpy()

    # 绘制结果
    plt.figure(figsize=(12, 8))

    # 子图1：原始数据和预测结果
    plt.subplot(2, 1, 1)
    plt.scatter(X_train.numpy(), y_train.numpy(), alpha=0.5, label='Training data', s=10)
    plt.plot(X_test, np.sin(X_test), 'g-', label='True sin(x)', linewidth=2)
    plt.plot(X_test, y_pred, 'r-', label='Predicted', linewidth=2)
    plt.title('Neural Network Fitting to sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：训练损失曲线
    plt.subplot(2, 1, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 计算最终误差
    with torch.no_grad():
        train_pred = model(X_train)
        final_loss = nn.MSELoss()(train_pred, y_train)
        print(f"\nFinal training MSE: {final_loss.item():.6f}")
