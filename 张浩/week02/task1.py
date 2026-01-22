# 1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 数据加载与预处理
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
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

# 创建数据集和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)
output_dim = len(label_to_index)

# 定义通用的多层感知机模型
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        # 根据 hidden_dims 列表动态添加线性层和激活函数
        for hidden_dim 在 hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 训练函数
def train_model(model, dataloader, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Model: {model.__class__.__name__} | Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    return epoch_losses

# 定义要测试的模型配置
configs = {
    "1-Layer (128)": [128],
    "2-Layer (128->64)": [128, 64],
    "3-Layer (256->128->64)": [256, 128, 64],
}

all_losses = {}

print("开始训练不同结构的模型...\n")
for name, hidden_dims in configs.items():
    print(f"\n--- 训练模型: {name} ---")
    model = MLPClassifier(vocab_size, hidden_dims, output_dim)
    losses = train_model(model, dataloader, num_epochs=10, lr=0.01)
    all_losses[name] = losses

# 结果对比
print("\n" + "=" * 50)
print("最终 Epoch 的 Loss 对比:")
print("=" * 50)
for name, losses in all_losses.items():
    final_loss = losses[-1]
    print(f"{name:<25}: {final_loss:.4f}")
