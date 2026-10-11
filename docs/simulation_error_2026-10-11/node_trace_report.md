# Qwen2.5 0.5B 节点级 CUDA 追踪（2026-10-11）

本次采集独立于 normal native timing：使用同一未插桩 llama.cpp server，Nsight Systems 的 node 级 Graph trace，每种 Graph 模式 2 次预热 + 1 次正式请求，正式请求由 NVTX 标记。插桩时间不回填到前一份误差报告。

证据文件在 [node_trace](node_trace/)；解析结果是 [node_evidence.json](node_trace/node_evidence.json)。Graph-on 正式窗口包含 510 个逻辑 Graph node event，其中 486 个 node 有 kernel activity 关联；Graph-off 没有 Graph node ID，只能做正式窗口内的 kernel-family 汇总。

## 可确认的 kernel family

Nsight 的模板类型与 llama.cpp 的 GGML 类型对应为：type 6 = Q5_0，type 8 = Q8_0，type 12 = Q4_K，type 14 = Q6_K。Graph-on 正式窗口的 GPU activity 汇总如下。这里的 duration 是 kernel activity 的总时长，不是 E2E，也不能把各行直接相加成请求时延。

| kernel family | instances | activity time | 占 kernel activity |
|---|---:|---:|---:|
| Q5_0 mul_mat_vec_q | 12,765 | 72.464 ms | 35.5% |
| quantize_q8_1 | 17,141 | 28.729 ms | 14.1% |
| Q8_0 mul_mat_vec_q | 1,538 | 26.062 ms | 12.8% |
| rms_norm_f32 | 5,840 | 17.075 ms | 8.4% |
| Q6_K mul_mat_vec_q | 1,419 | 11.147 ms | 5.5% |
| Q4_K mul_mat_vec_q | 1,418 | 10.088 ms | 4.9% |
| dense FP16 matvec/matmul | 5,674 | 13.658 ms | 6.7% |
| RoPE + softmax + set_rows | 11,444 | 17.758 ms | 8.7% |

因此，当前证据能把残差候选缩小到量化矩阵乘、输入量化、RMSNorm、attention 的 QK/softmax/AV、RoPE/KV-cache 写入、FFN 以及最终词表投影这些 family。它不能证明某一个 family 单独造成了全部误差。

## 能对齐到 source-dispatch 的节点

对 Graph-on，按同一次 trace 的低 32 位 node ordinal，并用 grid/block geometry 与 capture-only ownership 结构交叉检查，得到以下可复核的结构对齐：

| node ordinal | source owner | kernel / geometry | formal kernel activity |
|---:|---|---|---:|
| 509 | result_output，MUL_MAT，ne=[151936,1,1,1] | Q8_0 mul_mat_vec_q，grid [37984,1,1]，block [32,4,1] | 117 次，22.339 ms |
| 508 | result_output 同一 dispatch | quantize_q8_1，grid [4,1,1]，block [256,1,1] | 117 次，0.526 ms |
| 207 | ffn_gate-9 + ffn_up-9 + ffn_swiglu-9 | Q5_0 mul_mat_vec_q，grid [1216,1,1]，block [32,4,1] | 117 次，2.373 ms |
| 206 | 上述 FFN dispatch 的输入量化节点 | quantize_q8_1，grid [4,1,1]，block [256,1,1] | 117 次，0.325 ms |

这说明最终词表投影和 FFN 第 9 层的结构 owner 可以在节点序列上对齐到具体 kernel 形状。Q/K/V、RoPE、KV-cache、QK、softmax、AV、down/residual 等节点也存在同样的结构序列，但同名模板会在多层复用，不能仅凭 kernel 名称确认具体层。

## 对仿真误差的定位

Qwen2.5 的新 native 中位数与仿真之间，Graph-off 的 E2E 差为 243.681 ms，其中 127 × TPOT 差约 237.998 ms，TTFT 差约 5.137 ms；Graph-on 的 E2E 差为 48.009 ms，其中 127 × TPOT 差约 43.231 ms，TTFT 差约 5.280 ms。误差主要重复出现在 decode 路径，而非单次 prefill。

仿真的 measured cohort 给出的 decode family service prediction 是分析模型输出：Q5_0 220.888 µs、Q8_0 191.716 µs、dense FP16 197.600 µs、Q6_K 56.220 µs、Q4_K 35.442 µs；记录明确 confidence=low、reason=no_measurements、validated_llm_accuracy=false。因此这些数值只能用来说明 simulator 在哪些 family 上做了预测，不能拿来计算每个 kernel 的实测残差，也不能用来校准模型。

## 仍无法归因的残差

- Graph-off 没有 node ID，无法把其 kernel activity 归到具体 CUDA node。
- Graph-on 的 graphNodeId 与 source capture 通过 ordinal/geometry 对齐；它不是跨进程持久指针。在本次正式窗口里，逻辑 node event 为 510 个，而每个 replay cluster 约有 486 条 kernel activity 记录；两者不是一一对应，且部分节点可能被融合或表现为非 kernel activity，不能按最近时间配对。
- 无法从这次 trace 得出某一层单 kernel 的独立 E2E 贡献，或把一个 fused kernel 拆回多个 GGML op。
- 没有 DRAM/HBM bytes、cache miss、occupancy、功耗或真实内存路径计数；不能把任何 kernel activity 直接解释成物理 DRAM/NAND 账本。
- CPU 调度、Graph replay/update/launch、同步、分配器、KV-cache 流量、时钟/功耗状态以及 simulator 的物理路径假设，仍未被这次节点级采集分别测量。
- 因此 Graph-off 与 Graph-on 的 195.454 ms 正常 E2E 差，不能按本诊断 trace 的 kernel activity 差额拆分；插桩改变了执行开销，且两个模式的 logical node/fusion 结构不同。

原始 nsys-rep、NVTX 标记请求和统计摘要均已保留；SQLite 是 Nsight 的派生导出，没有作为仓库证据提交。
