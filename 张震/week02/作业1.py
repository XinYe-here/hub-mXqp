import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
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


class SimpleClassifier_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier_1, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class SimpleClassifier_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier_2, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.relu(self.fc5(out))
        out = self.fc6(out)
        return out


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

hidden_dim = 128
output_dim = len(label_to_index)
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

# Epoch [50/50], Loss: 0.1975
# Epoch [100/100], Loss: 0.1095
model_1 = SimpleClassifier_1(vocab_size, 128, output_dim) # 维度和精度有什么关系？
optimizer_1 = optim.SGD(model_1.parameters(), lr=0.01)
losses_1 = []
num_epochs = 100
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model_1.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer_1.zero_grad()
        outputs = model_1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_1.step()
        running_loss += loss.item()
        losses_1.append(loss.item())
        if idx % 100 == 0:
            print(f"Batch1 个数 {idx}, 当前Batch Loss: {loss.item()}")


    print(f"Epoch1 [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

model_2 = SimpleClassifier_2(vocab_size, 128, output_dim) # 维度和精度有什么关系？
optimizer_2 = optim.SGD(model_2.parameters(), lr=0.01)
losses_2 = []
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model_2.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer_2.zero_grad()
        outputs = model_2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_2.step()
        running_loss += loss.item()
        losses_2.append(loss.item())
        if idx % 100 == 0:
            print(f"Batch2 个数 {idx}, 当前Batch Loss: {loss.item()}")


    print(f"Epoch2 [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


model_3 = SimpleClassifier_2(vocab_size, 8, output_dim) # 维度和精度有什么关系？
optimizer_3 = optim.SGD(model_3.parameters(), lr=0.01)
losses_3 = []
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model_3.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer_3.zero_grad()
        outputs = model_3(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_3.step()
        running_loss += loss.item()
        losses_3.append(loss.item())
        if idx % 100 == 0:
            print(f"Batch3 个数 {idx}, 当前Batch Loss: {loss.item()}")


    print(f"Epoch3 [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model_1, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model_1, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")


# 设置全局字体为支持中文的字体，例如使用“SimHei”
matplotlib.rcParams['font.family'] = 'SimHei'  # 或 'Microsoft YaHei' 等
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# plt.subplot(1, 3, 2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失下降曲线')
plt.plot(losses_1, label='2层128节点', color='green')
plt.plot(losses_2, label='6层128节点', color='blue')
plt.plot(losses_3, label='6层16节点', color='red')
#plt.yscale('log')  # 使用对数坐标更清晰
plt.legend()
plt.grid(True)
plt.show()
