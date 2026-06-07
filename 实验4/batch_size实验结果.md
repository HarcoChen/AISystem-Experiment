# Batch Size 对通信开销的影响

## 实验设置

- 模型：ResNet50
- 数据集：CIFAR-10
- GPU 数量：2
- 训练轮数：5 epoch
- 分布式方式：PyTorch DDP
- 通信后端：NCCL

本实验固定 GPU 数量和模型结构，只改变每个 GPU 进程上的 `batch_size`，观察通信开销占比变化。

## 原始结果

### batch size = 64

```text
Epoch 0 | Epoch Time 20.2840s | Throughput 2465.00 | Acc 17.23% | Comm Ratio 53.84%
Epoch 1 | Epoch Time 19.6003s | Throughput 2550.99 | Acc 22.77% | Comm Ratio 14.18%
Epoch 2 | Epoch Time 19.4206s | Throughput 2574.58 | Acc 29.09% | Comm Ratio 22.31%
Epoch 3 | Epoch Time 19.6784s | Throughput 2540.86 | Acc 32.15% | Comm Ratio 25.33%
Epoch 4 | Epoch Time 19.7593s | Throughput 2530.45 | Acc 38.61% | Comm Ratio 8.52%
Model: resnet50 | Params: 23528522
Total Train Time: 107.3541s
Best Accuracy: 38.61%
```

### batch size = 128

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

### batch size = 256

```text
Epoch 0 | Epoch Time 8.5256s | Throughput 5864.68 | Acc 12.55% | Comm Ratio 53.39%
Epoch 1 | Epoch Time 7.9310s | Throughput 6304.41 | Acc 10.62% | Comm Ratio 23.01%
Epoch 2 | Epoch Time 7.9511s | Throughput 6288.40 | Acc 22.43% | Comm Ratio 28.24%
Epoch 3 | Epoch Time 8.4980s | Throughput 5883.75 | Acc 24.20% | Comm Ratio 30.83%
Epoch 4 | Epoch Time 7.9422s | Throughput 6295.46 | Acc 31.45% | Comm Ratio 27.73%
Model: resnet50 | Params: 23528522
Total Train Time: 47.6404s
Best Accuracy: 31.45%
```

## 当前汇总

| Batch Size | GPU 数量 | 总训练时间/s | 稳定吞吐率 samples/s | 后 4 轮平均通信占比 |
|---:|---:|---:|---:|---:|
| 64 | 2 | 107.3541 | 约 2550 | 17.59% |
| 128 | 2 | 55.6765 | 约 5300 | 19.61% |
| 256 | 2 | 47.6404 | 约 6200 | 27.45% |

后 4 轮平均通信占比计算：

```text
(14.18% + 22.31% + 25.33% + 8.52%) / 4 = 17.59%
(19.08% + 19.08% + 19.88% + 20.41%) / 4 = 19.61%
(23.01% + 28.24% + 30.83% + 27.73%) / 4 = 27.45%
```

## 初步分析

从总训练时间和吞吐率看，随着 batch size 从 64 增大到 128，再增大到 256，总训练时间总体下降，吞吐率明显提高，说明更大的 batch size 能更充分地利用 GPU 的计算能力。三组实验中，`batch_size=64` 的稳定吞吐率约为 2550 samples/s，`batch_size=128` 提升到约 5350 samples/s，`batch_size=256` 进一步提升到约 6200 samples/s。

从通信开销占比看，当前测得的后 4 轮平均值分别为：`batch_size=64` 时约 17.59%，`batch_size=128` 时约 19.61%，`batch_size=256` 时约 27.45%。这一结果没有完全呈现出“batch size 增大，通信占比下降”的理想趋势，说明本实验中的通信测量存在一定波动。原因在于这里的通信时间是通过“总时间减去 no_sync 下纯计算时间”近似估算的，而实际训练过程中计算与通信可能存在重叠，此外单次测量还会受到 CUDA 预热、缓存状态和系统调度等因素影响。

尽管通信占比结果存在波动，但吞吐率随 batch size 增大而上升这一现象比较稳定，说明增大 batch size 确实能提高 GPU 利用率。报告中可以说明：理论上 batch size 增大后，计算量增加得更明显，而通信量变化相对较小，因此通信占比通常会下降；但本次实验由于测量方法为近似方法，结果没有完全呈现理想趋势。
