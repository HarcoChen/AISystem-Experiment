# train.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self, num_conv_layers=3, dropout_rate=0.5):
        super().__init__()
        layers = []
        in_channel = 3

        for i in range(num_conv_layers):
            out_channel = min(32 * (2 ** i), 256)
            layers.append(nn.Conv2d(in_channel, out_channel, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool2d(2, 2))
            in_channel = out_channel

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            out = self.features(dummy)
            flatten_dim = out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(flatten_dim, 256),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train_model(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # 四周填充4像素后随机裁剪
    transforms.RandomHorizontalFlip(),    # 50%概率水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
])
    test_transform = transforms.Compose([
        # 把原始的 PIL 图片或 numpy 数组转成Tensor，
        transforms.ToTensor(), 
        
        # 图像归一化 (Normalize)
        # 这里的两个元组分别是 CIFAR-10 数据集在 RGB 三个通道上的标准“均值”和“标准差”
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=params['batch_size'], shuffle=False, num_workers=0)

    model = MyCNN(
        num_conv_layers=params['num_conv_layers'],
        dropout_rate=params['dropout_rate']
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    opt_cfg = params['optimizer']
    opt_name = opt_cfg['_name']
    lr = opt_cfg['lr']
    weight_decay = opt_cfg.get('weight_decay', 0.0)

    if opt_name == 'Adam':
        optimizer = torch.optim.Adam(
            mymodel.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            mymodel.parameters(),
            lr=lr,
            momentum=opt_cfg.get('momentum', 0.9),
            weight_decay=weight_decay
        )

    best_acc = 0.0

    for epoch in range(20):  # 控制时间
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        acc = 100 * correct / total
        best_acc = max(best_acc, acc)

    return best_acc