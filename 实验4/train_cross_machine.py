import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

torch.set_num_threads(8)

params = {
    "model_name": "resnet50",
    "batch_size": 128,
    "epochs": 5,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "num_workers": 0,
    "comm_batches": 5,
    "backend": "nccl",   
}


def build_model(model_name):
    if model_name == "resnet18":
        return models.resnet18(num_classes=10)
    elif model_name == "resnet50":
        return models.resnet50(num_classes=10)


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    correct_tensor = torch.tensor(correct, device=device)
    total_tensor = torch.tensor(total, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    return 100.0 * correct_tensor.item() / total_tensor.item()


def measure_comm_time(
    model, criterion, optimizer, images, labels, device
):  # 第一个轮次的通信时间和计算时间，受CPU DataLoader的影响较大，取后续4轮次
    optimizer.zero_grad()

    torch.cuda.synchronize(device)
    start_total = time.time()  # 把之前的任务执行完
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    torch.cuda.synchronize(device)
    total_time = time.time() - start_total

    optimizer.zero_grad()

    torch.cuda.synchronize(device)
    start_compute = time.time()
    with model.no_sync():
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
    torch.cuda.synchronize(device)
    compute_time = time.time() - start_compute

    total_tensor = torch.tensor(total_time, device=device)
    compute_tensor = torch.tensor(compute_time, device=device)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(compute_tensor, op=dist.ReduceOp.MAX)

    total_time = total_tensor.item()
    compute_time = compute_tensor.item()
    comm_time = max(total_time - compute_time, 0.0)
    comm_ratio = comm_time / total_time if total_time > 0 else 0.0
    return total_time, compute_time, comm_time, comm_ratio


def measure_comm_time_avg(model, criterion, optimizer, sample_batches, device):
    if not sample_batches:
        return 0.0, 0.0, 0.0, 0.0

    total_times = []
    compute_times = []
    comm_times = []
    comm_ratios = []

    for images, labels in sample_batches:
        total_time, compute_time, comm_time, comm_ratio = measure_comm_time(
            model, criterion, optimizer, images, labels, device
        )
        total_times.append(total_time)
        compute_times.append(compute_time)
        comm_times.append(comm_time)
        comm_ratios.append(comm_ratio)

    n = len(sample_batches)
    return (
        sum(total_times) / n,
        sum(compute_times) / n,
        sum(comm_times) / n,
        sum(comm_ratios) / n,
    )


def train_model(params, rank, local_rank, world_size):
    device = torch.device(f"cuda:{local_rank}")
    writer = None
    if rank == 0:
        writer = SummaryWriter(
            f"runs/ws{world_size}_{params['model_name']}_bs{params['batch_size']}_{params['backend']}"
        )
        print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}", flush=True)
        print("[rank 0] preparing datasets", flush=True)

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
    is_local_main = local_rank == 0
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=is_local_main, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=test_transform
    )
    if rank == 0:
        print(
            f"[rank 0] datasets ready: train={len(trainset)}, test={len(testset)}; continuing without barrier",
            flush=True,
        )
    if rank == 0:
        print("[rank 0] building dataloaders", flush=True)
    train_sampler = DistributedSampler(trainset)
    test_sampler = DistributedSampler(testset, shuffle=False)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=params["batch_size"],
        sampler=train_sampler,
        num_workers=params["num_workers"],
        pin_memory=True,
        persistent_workers=params["num_workers"] > 0,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=params["batch_size"],
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
    )
    if rank == 0:
        print("[rank 0] dataloaders ready, building model", flush=True)
    model = build_model(params["model_name"]).to(device)
    if rank == 0:
        print("[rank 0] model moved to cuda, wrapping with DDP", flush=True)
    model = DDP(model, device_ids=[local_rank])
    if rank == 0:
        print("[rank 0] DDP ready", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params["lr"],
        momentum=params["momentum"],
        weight_decay=params["weight_decay"],
    )

    best_acc = 0.0
    total_start = time.time()

    for epoch in range(params["epochs"]):
        model.train()
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print(f"[rank 0] epoch {epoch} started", flush=True)

        torch.cuda.synchronize(device)
        epoch_start = time.time()
        total_samples = 0
        sample_batches = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_samples += images.size(0)
            if rank == 0 and (
                batch_idx == 0 or (batch_idx + 1) % 50 == 0 or batch_idx + 1 == len(trainloader)
            ):
                print(
                    f"[rank 0] epoch {epoch} batch {batch_idx + 1}/{len(trainloader)} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

            if (world_size > 1 and len(sample_batches) < params["comm_batches"]):  # 单卡测什么通信时间
                sample_batches.append((images.detach(), labels.detach()))

        torch.cuda.synchronize(device)
        local_epoch_time = time.time() - epoch_start

        epoch_time_tensor = torch.tensor(local_epoch_time, device=device)
        sample_tensor = torch.tensor(total_samples, device=device)
        dist.all_reduce(epoch_time_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)

        epoch_time = epoch_time_tensor.item()
        throughput = sample_tensor.item() / epoch_time
        if rank == 0:
            print(f"[rank 0] epoch {epoch} finished training, starting evaluation", flush=True)
        acc = evaluate(model, testloader, device)
        best_acc = max(best_acc, acc)

        if world_size > 1 and sample_batches:
            _, _, _, comm_ratio = measure_comm_time_avg(
                model, criterion, optimizer, sample_batches, device
            )
        else:
            comm_ratio = 0.0

        if rank == 0 and writer is not None:
            print(
                f"Epoch {epoch} | Epoch Time {epoch_time:.4f}s | "
                f"Throughput {throughput:.2f} | Acc {acc:.2f}% | "
                f"Comm Ratio {comm_ratio:.2%}"
            )
            writer.add_scalar("Perf/Throughput", throughput, epoch)
            writer.add_scalar("Perf/Epoch_Time", epoch_time, epoch)
            writer.add_scalar("Perf/Comm_Ratio", comm_ratio, epoch)
            writer.add_scalar("Test/Accuracy", acc, epoch)

    torch.cuda.synchronize(device)
    total_train_time = time.time() - total_start
    total_time_tensor = torch.tensor(total_train_time, device=device)
    dist.all_reduce(total_time_tensor, op=dist.ReduceOp.MAX)
    total_train_time = total_time_tensor.item()

    if rank == 0:
        print(
            f"Model: {params['model_name']} | Batch Size: {params['batch_size']} | World Size: {world_size}   "
        )
        print(
            f"Total Train Time: {total_train_time:.4f}s| Throughput: {sample_tensor.item() / total_train_time:.2f} | Final Acc: {acc:.2f}%"
        )
        print(f"Best Accuracy: {best_acc:.2f}%")

    if writer:
        writer.close()


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "8"
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=params["backend"])

    train_model(params, rank, local_rank, world_size)
    
    dist.destroy_process_group()
