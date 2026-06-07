import torch

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.profiler import profile,ProfilerActivity,schedule,tensorboard_trace_handler
from torch.utils.tensorboard.writer import SummaryWriter
class myCNN(nn.Module):
    def __init__(self,num_conv_layers=3):
        super(myCNN,self).__init__()
        layers=[]
        in_channel=3
        for i in range(num_conv_layers):
            out_channel = 32*(2**i)
            layers.append(nn.Conv2d(in_channel,out_channel,3,padding=1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool2d(2,2))
            in_channel=out_channel
        self.features = nn.Sequential(*layers)
        
        with torch.no_grad():
            # 动态算dim，因为卷积有几层不知道
            dummy_input = torch.zeros(1, 3, 32, 32) 
            # 让假输入跑一遍刚刚定义好的卷积层
            dummy_output = self.features(dummy_input) 
            
            flatten_dim = dummy_output.numel()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
#———————————————————常量区—————————————————————
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


# CIFAR-10 的 10 个类别标签（打印测试结果时可以用到）
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

epochs = 15
learningrate=0.001
mybatch_size=256
# ——————————————————————主程序——————————————————————
if __name__=="__main__":
    device = torch.device("cuda")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=train_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mybatch_size,shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=test_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=mybatch_size,shuffle=False, num_workers=4)
    mymodel = myCNN().to(device)
    myloss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mymodel.parameters(),lr=learningrate)
    Writer=SummaryWriter('runs/cifar10_baseline')
    with profile(
        activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
        schedule=schedule(wait=1,warmup=1,active=3,repeat=2),
        on_trace_ready=tensorboard_trace_handler('runs/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(epochs):
            mymodel.train()
            running_loss =0
            for i,(images,labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = mymodel(images)
                loss = myloss(outputs,labels)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
                if i % 100 == 99:
                    Writer.add_scalar('Train/Loss',running_loss/100,epoch*len(trainloader)+i)
                    running_loss=0
                prof.step()
            correct = 0
            total = 0
            mymodel.eval()  # 切换到评估模式（停用 Dropout）
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = mymodel(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'Accuracy: {100 * correct / total:.2f}%')
            Writer.add_scalar('Test/Acc',100 * correct / total,epoch)
            mymodel.train()
        Writer.close()