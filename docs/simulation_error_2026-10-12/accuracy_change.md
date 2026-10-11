# 优化后整体准确率变化

当前源码提交：`97e4c424`。

本轮没有可报告的“优化后准确率变化”：新增 MMVQ 源码几何契约在旧的 20 组 Graph off/on 场景中实际覆盖为 0/20，因此继续跑同一矩阵只能重复 generic 路径。按你的要求，已停止这轮无关的完整矩阵。

## 可确认的基线

|范围|TTFT MAPE|TPOT MAPE|E2E MAPE|
|---|---:|---:|---:|
|off|23.961199%|20.972558%|21.128834%|
|on|23.878974%|10.843977%|10.064187%|
|all|23.920087%|15.908267%|15.596511%|

上述基线来自已完成的 9733a5a 矩阵；它不是本次修复后的结果。因新路径未启用，修复后的 MAPE 不能严谨地写成基线值，也不能写成 0% 改善。

## 为什么没有启用

准备好的 Qwen2.5 0.5B Q5/Q8 单段候选和 Qwen3 1.7B Q8_0 候选虽然存在，但场景缺少 `llama_cpp_gpu_native_invocations` 及 GPU 组件上的 `llama_cpp_mmvq_source_contract`。`prepare_llama_scenario` 只解析常规 llama.cpp runtime，不会凭模型名或 GGUF 格式自动猜测源契约。当前 GPU 实测 receipt 也只有 dispatch/几何证据，没有能供 `derive` 使用的当前 runtime source binding，因此不能事后把契约塞进已提交场景。

## 新测量能说明什么

独立 MMVQ receipt 覆盖 4 个 M=1 Q5/Q8 tensor 和 4 个 M=512 控制。它记录了 CUDA event、CUPTI kernel、grid/block、dispatch signature 与重复性；M=512 保持控制用途，未安装为 MMVQ rate surface。当前没有绑定相同 shape/dispatch 的独立 analytical_ns，所以 prediction error comparison 明确为 unavailable；没有把测量时延写进源码。

下一步必须先为当前 d3146f2 runtime 建立完整、可核验的 source/build binding，再在场景准备阶段应用契约并重新规划。那之后只需重测受影响的 Q5_0/Q8_0 M=1 decode 和一个未受影响控制组，再给出真正的 matched accuracy delta。
