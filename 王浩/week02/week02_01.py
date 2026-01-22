import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)  # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)  # 读取批量数据集 -》 batch数据

def loss_fun(hidden_dim: int):
    #hidden_dim = 128
    # 初始化列表存储每轮Loss
    epoch_losses = []
    output_dim = len(label_to_index)
    model = SimpleClassifier(vocab_size, hidden_dim, output_dim)  # 维度和精度有什么关系？
    criterion = nn.CrossEntropyLoss()  # 损失函数 内部自带激活函数，softmax
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # epoch： 将数据集整体迭代训练一次
    # batch： 数据集汇总为一批训练一次

    num_epochs = 10
    for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # running_loss += loss.item()
            # if idx % 50 == 0:
            #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
            epoch_loss = loss.item()
            epoch_losses.append(epoch_loss)

    return epoch_losses

# ---------------------- 5. Loss可视化绘图 ----------------------
def plot_loss_curves(loss_dict, title="Loss变化对比图"):
    """
    绘制多组Loss曲线
    :param loss_dict: 字典，key=配置名称，value=Loss列表
    """
    # 设置中文字体（避免中文乱码）
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建画布
    plt.figure(figsize=(10, 6))

    # 遍历绘制每组Loss曲线
    colors = ["red", "blue", "green"]  # 不同曲线的颜色
    markers = ["o", "s", "^"]  # 不同曲线的标记点
    for idx, (config, losses) in enumerate(loss_dict.items()):
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, color=colors[idx], marker=markers[idx], label=config)

    # 设置图表属性
    plt.xlabel("Epoch（训练轮数）", fontsize=12)
    plt.ylabel("Loss（损失值）", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)  # 显示网格
    plt.legend(loc="best")  # 显示图例

    # 保存图片（可选，建议保存方便对比）
    plt.savefig("loss_comparison.png", dpi=300, bbox_inches="tight")
    # 显示图片
    plt.show()

if __name__ == '__main__':
    # 调用绘图函数（传入多组Loss数据）
    loss_dict = {
        "隐藏层节点数=64": loss_fun(64),
        "隐藏层节点数=128": loss_fun(128),
        "隐藏层节点数=256": loss_fun(256)
    }
    plot_loss_curves(loss_dict)

