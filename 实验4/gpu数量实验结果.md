# GPU 数量对训练性能的影响

## 实验设置

- 模型：ResNet50
- 数据集：CIFAR-10
- 训练轮数：5 epoch
- 每个进程 batch size：128
- 分布式方式：PyTorch DDP
- 通信后端：NCCL

说明：这里的 batch size 是每个 GPU 进程上的 batch size，因此 GPU 数量增加时，全局 batch size 也会随之增大。

## 原始结果

### 1 卡

```text
Epoch 0 | Epoch Time 18.5042s | Throughput 2702.08 | Acc 22.44% | Comm Ratio 0.00%
Epoch 1 | Epoch Time 17.9558s | Throughput 2784.62 | Acc 26.51% | Comm Ratio 0.00%
Epoch 2 | Epoch Time 17.9194s | Throughput 2790.28 | Acc 30.31% | Comm Ratio 0.00%
Epoch 3 | Epoch Time 17.9108s | Throughput 2791.61 | Acc 35.97% | Comm Ratio 0.00%
Epoch 4 | Epoch Time 17.9060s | Throughput 2792.36 | Acc 38.80% | Comm Ratio 0.00%
Model: resnet50 | Params: 23528522
Total Train Time: 102.2838s
Best Accuracy: 38.80%
```

### 2 卡

```text
Epoch 0 | Epoch Time 9.9540s | Throughput 5023.12 | Acc 14.39% | Comm Ratio 24.14%
Epoch 1 | Epoch Time 9.6014s | Throughput 5207.59 | Acc 20.81% | Comm Ratio 19.08%
Epoch 2 | Epoch Time 9.3775s | Throughput 5331.90 | Acc 26.86% | Comm Ratio 19.08%
Epoch 3 | Epoch Time 9.3843s | Throughput 5328.04 | Acc 33.47% | Comm Ratio 19.88%
Epoch 4 | Epoch Time 9.3376s | Throughput 5354.68 | Acc 38.95% | Comm Ratio 20.41%
Model: resnet50 | Params: 23528522
Total Train Time: 55.6765s
Best Accuracy: 38.95%
```

### 4 卡

```text
Epoch 0 | Epoch Time 6.8735s | Throughput 7274.35 | Acc 14.44% | Comm Ratio 61.57%
Epoch 1 | Epoch Time 6.1357s | Throughput 8148.97 | Acc 15.97% | Comm Ratio 54.96%
Epoch 2 | Epoch Time 6.1399s | Throughput 8143.46 | Acc 20.86% | Comm Ratio 57.10%
Epoch 3 | Epoch Time 6.1304s | Throughput 8156.04 | Acc 17.95% | Comm Ratio 58.05%
Epoch 4 | Epoch Time 6.1453s | Throughput 8136.24 | Acc 25.97% | Comm Ratio 58.36%
Model: resnet50 | Params: 23528522
Total Train Time: 37.4257s
Best Accuracy: 25.97%
```

### 6 卡

```text
Epoch 0 | Epoch Time 5.1937s | Throughput 9627.89 | Acc 12.07% | Comm Ratio 55.91%
Epoch 1 | Epoch Time 4.2413s | Throughput 11789.68 | Acc 11.13% | Comm Ratio 52.83%
Epoch 2 | Epoch Time 4.2444s | Throughput 11781.24 | Acc 10.44% | Comm Ratio 60.00%
Epoch 3 | Epoch Time 4.2214s | Throughput 11845.26 | Acc 14.10% | Comm Ratio 59.51%
Epoch 4 | Epoch Time 4.2367s | Throughput 11802.47 | Acc 18.06% | Comm Ratio 58.78%
Model: resnet50 | Batch Size: 128 | World Size: 6
Total Train Time: 26.9158s| Throughput: 1857.79 | Final Acc: 18.06%
Best Accuracy: 18.06%
```

## 汇总表

| GPU 数量 | 总训练时间/s | 稳定吞吐率 samples/s | 加速比 | 理想加速比 | 并行效率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 102.2838 | 约 2790 | 1.00 | 1.00 | 100.0% |
| 2 | 55.6765 | 约 5330 | 1.84 | 2.00 | 91.8% |
| 4 | 37.4257 | 约 8145 | 2.73 | 4.00 | 68.3% |
| 6 | 26.9158 | 约 11805 | 3.80 | 6.00 | 63.3% |

加速比计算方式：

```text
Speedup = 单卡总训练时间 / 当前 GPU 数量下的总训练时间
```

因此：

```text
2 卡加速比 = 102.2838 / 55.6765 = 1.84
4 卡加速比 = 102.2838 / 37.4257 = 2.73
6 卡加速比 = 102.2838 / 26.9158 = 3.80
```

并行效率计算方式：

```text
Efficiency = Speedup / GPU 数量
```

## 结果分析

从实验结果可以看出，随着 GPU 数量增加，单轮训练时间明显下降，吞吐率明显上升。单卡训练的总时间为 102.2838s，两卡训练降低到 55.6765s，四卡训练进一步降低到 37.4257s，六卡训练继续下降到 26.9158s，说明 DDP 能够有效利用多张 GPU 提升训练性能。

两卡训练的加速比为 1.84，接近理想的 2 倍线性加速，并行效率约为 91.8\%。四卡训练的加速比为 2.73，并行效率下降到约 68.3\%；六卡训练的加速比进一步提升到 3.80，但并行效率继续下降到约 63.3\%。这说明 GPU 数量增加后，训练速度虽然继续提升，但扩展效率会逐渐下降。

造成无法达到理想线性加速的主要原因是分布式训练中存在额外开销。DDP 在反向传播过程中需要在不同 GPU 之间同步梯度，GPU 数量越多，通信和同步开销越明显。同时，数据加载、进程调度、测试评估等部分也不会随着 GPU 数量增加而完全线性缩短。

从通信开销占比看，四卡训练中通信占比大多稳定在 56% 到 58% 左右，明显高于两卡训练约 19% 到 20% 的通信占比；六卡训练的后 4 轮通信占比进一步达到约 57.78%，整体处于 53% 到 60% 区间。这说明当 GPU 数量增加时，每张 GPU 分到的计算量减少，但梯度同步的通信需求仍然存在，因此通信开销在总训练时间中的占比会上升，最终导致并行效率下降。

需要注意的是，本实验中 `batch_size=128` 是每个 GPU 进程上的 batch size，因此 GPU 数量增加时，全局 batch size 也随之增大。全局 batch size 的变化会影响模型收敛，因此不同 GPU 数量下的准确率不适合直接作为模型效果对比，实验重点应放在训练时间、吞吐率和加速比等性能指标上。
