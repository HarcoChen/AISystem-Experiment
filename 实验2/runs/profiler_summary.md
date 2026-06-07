# Profiler 数据总结

## 统计口径

- 数据来源：`runs/*/profiler_traces/*.pt.trace.json`
- 统计对象：每个实验中的 `ProfilerStep#40` 到 `ProfilerStep#69`
- 核心指标：`Average Step Time`，即所有 `ProfilerStep` 的 `dur` 平均值
- 单位说明：`dur` 原始单位为微秒，本文统一换算为毫秒（ms）

## Average Step Time 排名

| 排名 | 实验 | Average Step Time (ms) | 相对 `mnist_basic` |
| --- | --- | ---: | ---: |
| 1 | `mnist_custom_linear_cuda_nobias` | 9.421 | -10.70% |
| 2 | `mnist_basic_nobias` | 9.458 | -10.35% |
| 3 | `mnist_custom_linear_cpp` | 9.468 | -10.25% |
| 4 | `mnist_custom_linear` | 9.513 | -9.83% |
| 5 | `mnist_custom_linear_cuda` | 9.683 | -8.21% |
| 6 | `mnist_basic` | 10.550 | baseline |
| 7 | `mnist_custom_linear+op_cpp` | 10.988 | +4.15% |
| 8 | `mnist_custom_linear+op` | 11.195 | +6.11% |

## 详细指标

| 实验 | Step 数 | Step 范围 | Avg (ms) | Min (ms) | Max (ms) | Std (ms) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `mnist_custom_linear_cuda_nobias` | 60 | `40-69` | 9.421 | 7.245 | 13.110 | 1.859 |
| `mnist_basic_nobias` | 60 | `40-69` | 9.458 | 7.370 | 13.172 | 1.811 |
| `mnist_custom_linear_cpp` | 60 | `40-69` | 9.468 | 7.254 | 13.450 | 1.913 |
| `mnist_custom_linear` | 60 | `40-69` | 9.513 | 7.348 | 13.452 | 1.869 |
| `mnist_custom_linear_cuda` | 60 | `40-69` | 9.683 | 7.595 | 13.562 | 1.794 |
| `mnist_basic` | 60 | `40-69` | 10.550 | 5.258 | 24.861 | 3.826 |
| `mnist_custom_linear+op_cpp` | 60 | `40-69` | 10.988 | 9.918 | 15.374 | 1.065 |
| `mnist_custom_linear+op` | 60 | `40-69` | 11.195 | 9.687 | 15.174 | 0.924 |

## 结论提炼

1. `Average Step Time` 最优的是 `mnist_custom_linear_cuda_nobias`，平均每步 9.421 ms，比 `mnist_basic` 快约 10.70%，也略快于 `mnist_basic_nobias`。
2. 在不含 Bent Identity 的自定义 Linear 中，`mnist_custom_linear_cuda_nobias` 最快；其后是 `mnist_custom_linear_cpp`、`mnist_custom_linear`、`mnist_custom_linear_cuda`，几组结果整体接近。
3. `mnist_custom_linear+op_cpp` 和 `mnist_custom_linear+op` 明显更慢，说明引入 `+op` 版本后，单步耗时反而上升；其中 `mnist_custom_linear+op` 最慢。
4. 从稳定性看，`mnist_basic` 的波动最大，标准差 3.826 ms，最大单步耗时达到 24.861 ms，显著高于其他实验。
5. `+op` 两组虽然平均耗时较差，但波动更小，说明其问题更像是“稳定地慢”，而不是偶发抖动导致。

## 建议结论

- 如果只看 `Average Step Time`，当前最优方案是 `mnist_custom_linear_cuda_nobias`。
- 如果需要在“不带 Bent Identity 的自定义实现”中选方案，`mnist_custom_linear_cuda_nobias` 是当前最优候选；若只比较原先的三类自定义 Linear，则 `mnist_custom_linear_cpp` 仍略优于 Python 版和含 bias 的 CUDA 版。
- `mnist_custom_linear+op` 与 `mnist_custom_linear+op_cpp` 暂时不适合作为性能优化方案，建议优先排查额外算子封装带来的开销。
