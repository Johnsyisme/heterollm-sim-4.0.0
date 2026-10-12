# MMVQ 共享几何覆盖盘点（2026-10-12）

输入：`docs/gguf_preset_native_validation_2026-10-08/base/` 的 10 个 512/128 preset；只读取已有 projection descriptor，不使用 native timing，不写拟合参数。

|模型|去重后物理 segment|Q5_0/Q8_0|K-format|多 segment projection|
|---|---:|---:|---:|---|
|llama3_1_8b_q4_k_m|12|0|12|attention.qkv,mlp.down,mlp.up_gate|
|llama3_2_1b_q4_k_m|12|0|12|attention.qkv,mlp.down,mlp.up_gate|
|qwen2_5_0_5b_q4_k_m|12|10|2|attention.qkv,mlp.down,mlp.up_gate|
|qwen2_5_7b_q4_k_m|12|0|12|attention.qkv,mlp.down,mlp.up_gate|
|qwen3_0_6b_f16|0|0|0||
|qwen3_14b_q4_k_m|12|0|12|attention.qkv,mlp.down,mlp.up_gate|
|qwen3_1_7b_q8_0|10|10|0|attention.qkv,mlp.up_gate|
|qwen3_4b_q4_k_m|12|0|12|attention.qkv,mlp.down,mlp.up_gate|
|qwen3_8b_q4_k_m|12|0|12|attention.qkv,mlp.down,mlp.up_gate|
|qwen3_8_27b_mixed|15|0|3|attention.qkv,mlp.up_gate|

判定规则：

- **可共享 MMVQ 几何**：普通 contiguous 2-D、一个已被 lowering 选中的物理权重 child、源码支持的 quant format、K 是格式 block 的整数倍、M 不超过该架构的 source batch limit；只复用 `derive_mmvq_geometry` 的 launch geometry，不共享 timing/rate。架构不再按产品名绑定：目标可以使用显式 `compute_capability`、`highest_compiled_arch`、`warp_size`，旧 RTX 5080 catalog 记录保留兼容回退。当前 RTX 5080 analytical profile 对 23 个 CUDA 源格式都登记了无 samples 的 analytical descriptor。
- **QKV 多 segment**：不能把 q/k/v 合并成一次 MMVQ；planner 会按 descriptor 中唯一匹配的物理 child 拆成独立调用，各自复用同一几何函数。每个 segment 保留自己的 weight/output slice 和 source audit；同形状的两个 child 没有 child index 时仍拒绝。
- **Q4_K/Q5_K/Q6_K**：Blackwell 普通 vector dispatch 的 M 上限分别为 5、6、7；其余源格式按 `mmvq.cu` 表使用 8 或架构特定上限。M 上限不是 timing rate，也不会把 issue-bound 自动打开。
- **M=512**：超过 MMVQ batch limit，必须回退 MMQ；不能用 M1 的 work 或缩放 rate。
- **F32 存储证据**：全局 `llama_cpp_f32_hidden_storage` 或操作级 `runtime_input_storage_bits/runtime_output_storage_bits=32`（并与实际字节数相符）均可作为输入/输出存储证明；只凭字节数不能放宽。
- **IQ3/IQ4**：普通 contiguous segment 纳入共享几何；expert、dynamic RHS、fused epilogue、scatter 仍保持 fail-closed。


全库去重格式总数（10 个 preset，尚未乘 graph off/on）：Q4_K=55，Q6_K=20，Q5_K=2，Q5_0=8，Q8_0=12，IQ4_XS=8，IQ3_S=4。当前共享 source-rule 已覆盖这些实际出现的普通 segment；这是静态覆盖，不代表 native dispatch 或 timing 已验证。每次调用仍验证 M 上限、K block 整除、N/CTA 尾部和物理 segment。

旧 20-case（graph off/on）中，Q5/Q8 candidate 共 40 个实体 segment；K-format candidate 共 154 个实体 segment。M=512 的 prefill 请求不属于这个上限。
