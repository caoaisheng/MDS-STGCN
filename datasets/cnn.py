import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# 假设你的数据文件是以某种形式存储的，比如numpy数组
# 请替换下面的加载方式为你实际的数据加载方式

# 加载训练数据

# 加载数据
train_data = torch.load('E:/Fogvideo/unpreprocessed_data/training/train.pt')
val_data = torch.load('E:/Fogvideo/unpreprocessed_data/training/val.pt')
test_data = torch.load('E:/Fogvideo/unpreprocessed_data/training/test.pt')

# 从加载的数据中提取样本和标签
train_samples, train_labels = train_data['samples'].float(), train_data['labels'].long()
val_samples, val_labels = val_data['samples'].float(), val_data['labels'].long()
test_samples, test_labels = test_data['samples'].float(), test_data['labels'].long()
# 检查数据是否正确加载
assert train_samples.dim() > 0, "Train samples tensor has no dimensions"
assert train_labels.dim() > 0, "Train labels tensor has no dimensions"

# 确保样本和标签数量匹配
assert train_samples.size(0) == train_labels.size(0), "Sample and label counts do not match"

# 创建数据集和数据加载器
train_dataset = TensorDataset(train_samples, train_labels)
val_dataset = TensorDataset(val_samples, val_labels)
test_dataset = TensorDataset(test_samples, test_labels)

# 根据数据集大小调整batch_size
batch_size = min(64, train_samples.size(0))  # 假设64是你想要的batch_size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 接下来是模型训练和评估的代码...

# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 64, 64)  # 16 * 64 是根据输入数据维度计算得出的，可以根据实际情况调整
        self.fc2 = nn.Linear(64, 2)  # 输出维度为2，对应两个类别

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 64)  # 展开成一维向量
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
model = SimpleCNN().float()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # 在验证集上验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')


# 在训练集上训练模型
train(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# 在测试集上评估模型
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.4f}')
