import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import nni


# -----------------------------
# 1. 深度可分离卷积
# -----------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# -----------------------------
# 2. 根据参数构建模型
# -----------------------------
class MNISTNet(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)

        # 搜索点1：卷积类型
        if params['conv_type'] == 'conv':
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        else:
            self.conv2 = DepthwiseSeparableConv(32, 64)

        # 搜索点2：dropout
        self.dropout = nn.Dropout(params['dropout'])

        # 搜索点3：fc宽度
        fc_dim = params['fc_size']

        self.fc1 = nn.Linear(2304, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# -----------------------------
# 3. 训练 + 上报
# -----------------------------
def main():
    # 获取搜索参数
    params = nni.get_next_parameter()
    print("Params:", params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MNISTNet(params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(
        MNIST('data', download=True, transform=transf),
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        MNIST('data', download=True, train=False, transform=transf),
        batch_size=32,
        num_workers=0
    )

    for epoch in range(2):
        model.train()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(1)
                correct += pred.eq(target).sum().item()

        acc = 100. * correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1}, Acc: {acc:.2f}%")

        nni.report_intermediate_result(acc)

    nni.report_final_result(acc)


if __name__ == '__main__':
    main()