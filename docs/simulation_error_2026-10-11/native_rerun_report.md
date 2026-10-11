# 目标 native 重测与后端定位报告（2026-10-11）

本报告把新仿真矩阵中的四个关键场景换成同一机器、同一 llama.cpp 构建的最新 native 观测，再重新计算误差。仿真结果没有被重新拟合；native 重测只用于判断旧 denominator 是否发生了时间漂移。

仿真矩阵来自提交 `9733a5a8cb220d4c5f1c9ffd81fc33dcb2d2f636` 的隔离工作树；本报告及其证据文件由提交 `e9e6008` 保存。native server 是 `d3146f2b56c2db4711ac8391871c9e529d1946d7` 构建的未插桩 `llama-server.exe`。四组按固定交错顺序运行：Qwen2.5 off → Qwen3 off → Qwen2.5 on → Qwen3 on；每组 2 次预热、5 次正式测量，Qwen3 on 因初始 5 次 CV 超过 5% 追加为 9 次正式测量。

## 重测门槛

每个阶段比较新旧中位数。接受条件是绝对中位数漂移不超过 `max(3 × 旧 MAD, 旧中位数的 5%)`，且新正式样本 CV 不超过 5%。四组在扩展后全部通过；因此没有理由把剩余 16 个场景的 native denominator 全部重测。

## 新 native 中位数与仿真残差

单位均为毫秒；残差为“仿真 − 新 native”，负值表示仿真偏低。

| 模型 / Graph | TTFT 仿真 / native | TPOT 仿真 / native | E2E 仿真 / native | E2E APE |
|---|---:|---:|---:|---:|
| Qwen2.5 0.5B / off | 8.043 / 13.180 | 1.436 / 3.310 | 190.450 / 434.131 | 56.13% |
| Qwen2.5 0.5B / on | 8.043 / 13.323 | 1.438 / 1.778 | 190.668 / 238.677 | 20.11% |
| Qwen3 0.6B / off | 12.577 / 19.170 | 3.543 / 3.950 | 462.569 / 520.348 | 11.10% |
| Qwen3 0.6B / on | 12.577 / 19.142 | 3.545 / 2.770 | 462.810 / 370.884 | 24.79% |

按 E2E 绝对百分比误差排名：Qwen2.5 off（56.13%）> Qwen3 on（24.79%）> Qwen2.5 on（20.11%）> Qwen3 off（11.10%）。按阶段看，最大 TTFT APE 为 Qwen2.5 on（39.63%），最大 TPOT APE 与最大 E2E APE 都是 Qwen2.5 off（56.61%、56.13%）。

新旧 native 中位数变化没有解释掉主要残差：Qwen2.5 off 的 E2E 中位数从 423.387 ms 变为 434.131 ms（+2.54%），但 E2E APE 仍为 56.13%；Qwen2.5 on 几乎不变（238.704 → 238.677 ms）。Qwen3 on 经过 9 次测量后为 370.884 ms，较旧中位数上升 4.66%，仍在门槛内，残差仍为 24.79%。这说明主要问题在仿真 cost model 与 native 行为之间，而不是复用旧 native 结果造成的假象。

Graph 的 native E2E 降幅为 Qwen2.5 45.02%、Qwen3 28.72%；对应仿真 E2E 变化约为 -0.11%、-0.05%。因此 Graph benefit 没有在当前仿真中体现，仍是最明确的结构性残差。这里的 Graph 结论来自同一轮 off/on 的 native 中位数配对，未用单个异常样本替代中位数。

## 后端粒度定位

对 Qwen2.5 0.5B Graph on 另跑了一次独立 Nsight Systems 图级采集（128/32 短请求，1 次预热 + 1 次正式请求）。采集文件为 [`qwen2_5_0_5b_graph_on.nsys-rep`](native_rerun/qwen2_5_0_5b_graph_on.nsys-rep)，统计摘要为 [`qwen2_5_0_5b_graph_on_nsys_stats.txt`](native_rerun/qwen2_5_0_5b_graph_on_nsys_stats.txt)。它显示了 `cudaGraphLaunch` 4 次、`cudaGraphInstantiate` 2 次、`cudaGraphExecUpdate` 2 次，并列出了 CUDA API 与 kernel 汇总；这能确认 Graph 生命周期和 kernel 时间存在，但插桩时间不参与上面的 normal timing。

现有 native JSON 和旧 Graph diagnostic 仍没有每个 GGML op/CUDA node 的开始结束时间，也没有 DRAM/HBM transaction 计数；旧 diagnostic 明确 `latency_values_retained=false`。因此本报告可以把误差定位到 TTFT、TPOT、E2E 和 Graph off/on 层级，不能据此声称某个 kernel 或物理 DRAM 路径造成了多少毫秒。若要继续下钻，需要单独的 node-level trace，并把 source dispatch 结构映射与同请求 CUDA timeline 对齐；Nsight Compute 的单 kernel 计数只能作为诊断证据，不能回填 E2E 真值。

## 数据来源边界

native 数据是 llama.cpp 引擎边界的 prompt/decode/request timing，不是仿真输出，也不是实测 DRAM/NAND 物理账本。仿真报告中的物理路径账本来自 simulator 的场景计数和 cost model；它可用于内部路径归属和一致性检查，不能表述为硬件实测吞吐、流量或能耗。

原始 native JSON、9 次扩展结果、分析 JSON 和 Nsight 证据均在 [`native_rerun`](native_rerun/) 目录中；分析脚本为 [`analyze_targeted_native_rerun.py`](../../tools/analyze_targeted_native_rerun.py)。
