import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import nni
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class MyCNN(nn.Module):
    def __init__(self, num_conv_layers=3, dropout_rate=0.5):
        super().__init__()
        layers = []
        in_channels = 3

        for i in range(num_conv_layers):
            out_channels = min(32 * (2 ** i), 256)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            flattened_dim = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(flattened_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_optimizer(model, params):
    optimizer_name = params["optimizer"]
    lr = params["lr"]

    if isinstance(optimizer_name, dict):
        optimizer_cfg = optimizer_name
        optimizer_name = optimizer_cfg.get("_name", "Adam")
        lr = optimizer_cfg.get("lr", lr)
        weight_decay = optimizer_cfg.get("weight_decay", 0.0)
        momentum = optimizer_cfg.get("momentum", 0.9)
    else:
        weight_decay = params.get("weight_decay", 0.0)
        momentum = params.get("momentum", 0.9)

    if optimizer_name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return 100.0 * correct / total


def main():
    default_params = {
        "lr": 0.001,
        "batch_size": 128,
        "optimizer": "Adam",
        "num_conv_layers": 3,
        "dropout_rate": 0.5,
    }

    tuned_params = nni.get_next_parameter()
    params = {**default_params, **tuned_params}

    trial_budget = max(1, int(params.pop("TRIAL_BUDGET", 9)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "./data"
    should_download = not os.path.exists(os.path.join(data_root, "cifar-10-batches-py"))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=should_download,
        transform=train_transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=should_download,
        transform=test_transform,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = MyCNN(
        num_conv_layers=params["num_conv_layers"],
        dropout_rate=params["dropout_rate"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, params)

    best_acc = 0.0

    for epoch in range(trial_budget):
        model.train()
        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        accuracy = evaluate(model, testloader, device)
        best_acc = max(best_acc, accuracy)
        nni.report_intermediate_result(
            {
                "default": accuracy,
                "best_acc": best_acc,
                "epoch": epoch + 1,
                "budget": trial_budget,
            }
        )

    nni.report_final_result(
        {
            "default": best_acc,
            "best_acc": best_acc,
            "budget": trial_budget,
        }
    )


if __name__ == "__main__":
    main()
