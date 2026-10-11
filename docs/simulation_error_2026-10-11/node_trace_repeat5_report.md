# Qwen2.5 0.5B 重复 node trace 结果

这次测量使用当前 native binary 和同一个 GGUF，Graph-off 与 Graph-on 各执行 2 次预热、5 次正式请求。Nsight Systems 只在正式请求范围内统计 CUDA activity；trace 仍是诊断证据，不会自动进入模拟器成本表。

## 身份与边界

- runtime source commit：`d3146f2b56c2db4711ac8391871c9e529d1946d7`
- llama-server SHA-256：`9abb694e0ac4ba471f9749649a93304e1033278276b9ea25808c05230f038562`
- GGUF SHA-256：`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- GPU：RTX 5080，SM120；prompt 512，decode 128，batch/ubatch 512，threads 16，F16 KV，flash attention off
- 原始诊断 JSON：`node_trace_repeat5/qwen25_off_repeat5_v2.json`、`node_trace_repeat5/qwen25_on_repeat5_v2.json`
- Nsight 原始 trace：`node_trace_repeat5/qwen25_off_repeat5_v2.nsys-rep`、`node_trace_repeat5/qwen25_on_repeat5_v2.nsys-rep`
- 可复核汇总：`node_trace_repeat5/node_repeat_evidence.json`

## 重复性与粒度

Graph-off 没有 CUDA graph node ID，因此只能按正式请求窗口给出 kernel family 汇总。Graph-on 每次都有 510 个逻辑 graph-node event，其中 486 个节点有 kernel activity 关联；5 次请求的这两个计数都保持不变。用上一份 capture 的 node ordinal、family、grid/block 做结构交叉核对时，节点都能找到对应的 capture-only source owner，但这个跨 capture 连接不具有持久指针语义，也不产生 timing qualification。

正式窗口内的 activity 汇总如下。时长是 Nsight kernel activity duration 的总和，不是请求 E2E：

| family | Graph-off 中位数（MAD） | Graph-on 中位数（MAD） |
|---|---:|---:|
| Q5_0 `mul_mat_vec_q:type_6` | 58.249 ms（1.230 ms） | 316.681 ms（95.915 ms） |
| Q8_0 `mul_mat_vec_q:type_8` | 25.216 ms（1.597 ms） | 103.571 ms（25.976 ms） |
| `quantize_q8_1` | 17.773 ms（0.108 ms） | 120.365 ms（40.309 ms） |

Graph-on activity 的波动和数量级不能解释为 Graph-on 的 E2E 变慢；它说明 profiler activity、graph replay、并发与请求窗口之间存在额外边界，不能把这些时长相加后写回模拟器。

## 能定位到的误差来源

当前 native source 的 dispatch geometry 与 trace 一致：Qwen2.5 decode 的 Q5_0 形状为 `(M,N,K)=(1,896,896)`、`(1,4864,896)`，Q8_0 形状为 `(1,128,896)`、`(1,151936,896)`；对应 MMVQ grid 分别为 `N/4`，block 为 `(32,4,1)`。因此没有证据表明误差来自 Q5/Q8 dispatch 选择或 grid 计算。

误差最可信的候选是模拟器尚未校准的 MMVQ/Q8_1 service rate，以及 Graph-off 的 host/driver queue 供给成本。现有 trace 能证明它们在请求中占据主要 activity，但不能在单个 fused kernel 内拆开计算、解包、scale、reduction、occupancy 或真实内存流量。

## 不能由本次数据闭合的部分

1. Graph-off 没有 node ID，只能保留 family 级别。
2. Graph-on logical node 与 kernel activity 不是一一对应；本报告只记录结构 owner，不按最近时间强行配对。
3. 没有 DRAM/HBM bytes、cache miss、occupancy、功耗、CPU 调度和 allocator counters。
4. Nsight activity 不是 E2E，不能用它拆分正常 Graph-off/Graph-on E2E 差异。
5. 还没有同一 binary/GGUF 下、带数值校验的 standalone Q5_0/Q8_0 MMVQ harness，因此不能把本次 family duration 写成 Level-2 rate surface。

另外，独立 `llama-bench` 只作整体边界控制：同一 512/128 配置下，Graph-off 的 aggregate 平均耗时为 5.219 s，Graph-on 为 1.272 s。它和 llama-server 的 HTTP/server timing 边界不同，不能直接与服务 E2E 或模拟器 TPOT 相减；原始 JSONL 与身份收据在 `qwen25_exact_decode_bench_*` 文件中。

## 后端改动

- 统一 MMQ source contract 的 pinned revision，并修正过期的 `mmq.cuh` 内容哈希；source identity 不再误报旧提交。
- 给模拟任务增加 `cuda_source_dispatch` capture-only ownership annotation，显式记录一对一、融合多对一和一对多关系；annotation 不改变 service、bytes、energy、work 或 dependencies。
- 增加重复 node trace 分窗分析器，固定输出 family 中位数/MAD、Graph-on node 数量和结构 owner，并将 timing/physical-ledger qualification 固定为 false。
- Graph device replay envelope 继续 fail-closed；没有拓扑匹配的独立 device dispatch 测量就不填写 replay latency。

下一步若要真正闭合 Q5/Q8 残差，需要在同一 source/binary/GGUF 身份下做 standalone MMVQ probe：至少覆盖上述 4 个 decode shape 和对应 `M=512` MMQ 控制组，2 次预热、9 次正式测量，记录 CUDA event、数值校验、grid/block 及 dispatch signature。Graph replay 则需要把 host replay submit、首节点 device dispatch gap、真实 body 分开测量。
