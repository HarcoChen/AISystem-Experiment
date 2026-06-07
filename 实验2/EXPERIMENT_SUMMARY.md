# MNIST 自定义算子实验总结

更新日期：2026-03-26

## 1. 实验目标

对比不同实现平台与算子组合在 MNIST 任务上的训练表现，重点关注：

1. 平台差异：Python vs C++ 扩展。
2. 算子差异：PyTorch 内置 vs 自定义 `Linear` / `BentIdentity`。
3. 可解释性：通过 Profiler trace 观察热点，而不是只看最终精度。

## 2. 当前脚本矩阵（与文件名一致）

| 脚本 | 平台 | Linear | Activation | TB 日志目录 |
| --- | --- | --- | --- | --- |
| `mnist_basic.py` | PyTorch 内置 | `nn.Linear` | `F.relu` | `runs/mnist_basic` |
| `mnist_custom_linear.py` | Python | 自定义 Python Linear | `F.relu` | `runs/mnist_custom_linear` |
| `mnist_custom_linear+op.py` | Python | 自定义 Python Linear | 自定义 Python BentIdentity | `runs/mnist_custom_linear+op` |
| `mnist_custom_linear_cpp.py` | C++ 扩展 | 自定义 C++ Linear | `F.relu` | `runs/mnist_custom_linear_cpp` |
| `mnist_custom_linear+op_cpp.py` | C++ 扩展 | 自定义 C++ Linear | 自定义 C++ BentIdentity | `runs/mnist_custom_linear+op_cpp` |

## 3. 统一训练配置

### 3.1 脚本默认参数（5 个脚本一致）

1. `batch_size=64`
2. `test_batch_size=1000`
3. `epochs=3`
4. `lr=1.0`（Adadelta）
5. `gamma=0.7`（StepLR）
6. `seed=1`
7. `log_interval=10`

### 3.2 一键脚本实际覆盖

`run_all_mnist_3epochs.sh` 当前会显式传参 `--epochs 5`，即：

1. 单独运行某个脚本：默认 3 epoch。
2. 用一键脚本运行：实际 5 epoch。

## 4. 监控与可观测性（当前版本）

### 4.1 TensorBoard 标量（已精简）

5 个脚本统一只保留两项：

1. `train/loss_epoch`
2. `test/accuracy`

说明：

1. 已移除 step 级大部分监控（如 step 耗时、step 吞吐、step_eval、to95/to99 等），减少写盘与观测干扰。
2. 当前更适合做简洁对比和报告展示。

### 4.2 Profiler（5 个脚本全部具备）

Profiler 参数统一为：

1. `PROFILE_WAIT=20`
2. `PROFILE_WARMUP=20`
3. `PROFILE_ACTIVE=30`
4. `PROFILE_REPEAT=1`

Trace 输出目录：

`runs/<exp_name>/profiler_traces/*.pt.trace.json`

## 5. 运行方式

### 5.1 一键执行

```bash
bash class2/run_all_mnist_3epochs.sh
```

按顺序运行以下 5 个脚本（并覆盖为 5 epoch）：

1. `mnist_basic.py`
2. `mnist_custom_linear.py`
3. `mnist_custom_linear+op.py`
4. `mnist_custom_linear_cpp.py`
5. `mnist_custom_linear+op_cpp.py`

### 5.2 TensorBoard 查看

```bash
tensorboard --logdir=class2/runs
```

建议重点看：

1. `train/loss_epoch`
2. `test/accuracy`
3. `Profile` 标签页（算子级热点）

若缺 Profile 标签：

```bash
pip install tensorboard_plugin_profile torch_tb_profiler
```

## 6. 当前阶段结论

1. MNIST 任务较简单，最终精度容易饱和，`test/accuracy` 的区分度有限。
2. 在小模型场景里，Python 调度、数据加载、日志和 profiler 开销会放大，导致 step 级直觉不稳定。
3. 现阶段先用“epoch 级简化指标 + trace 定位”更稳；若要做严格速度结论，建议单独做无 profiler 的纯吞吐基准。

## 7. BentIdentity 热点分析（本轮补充）

根据 `runs/processed/profile_top_cpu_ops.csv`，引入 BentIdentity 后，热点从卷积反向转移到了激活函数前后向本身：

1. `mnist_custom_linear+op`：
   - `autograd::engine::evaluate_function: BentIdentityFuncBackward`（`total_ms=40.46`）
   - `BentIdentityFuncBackward`（`total_ms=38.03`）
   - `BentIdentityFunc`（`total_ms=36.79`）
2. `mnist_custom_linear+op_cpp`：
   - `autograd::engine::evaluate_function: myBentIdentityFuctBackward`（`total_ms=34.97`）
   - `myBentIdentityFuctBackward`（`total_ms=32.64`）
   - `myBentIdentityFuct`（`total_ms=31.94`）

这属于预期现象，主要原因：

1. BentIdentity 的前后向都包含 `square/sqrt/div` 等连续算子链，计算复杂度高于 ReLU。
2. 网络里激活调用次数较多（`conv1` 后、`conv2` 后、`fc1` 后），累计开销明显。
3. 当前 C++ 实现是 ATen 表达式拼接，不是融合 CUDA kernel，因此不会天然显著快于 Python 版的算子组合。
4. 在 MNIST 小模型场景里，profiler 与框架调度开销占比更高，容易放大“热点占比”观感。

因此，本实验里的“BentIdentity 变成热点”结论是可信的；若要进一步比较“纯算子速度”，建议补做：

1. 关闭 profiler 的纯训练吞吐测试（固定 seed，多次重复，取均值）。
2. 独立 activation microbenchmark（仅测 ReLU vs BentIdentity 前后向，统一 shape 与 dtype）。
