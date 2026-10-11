# Qwen2.5 0.5B standalone MMVQ/MMQ measurement

- GGUF SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- ggml-cuda.dll SHA-256: `2224f2732b4dcb946bb10fcc9af1b482739d14f9b40d09241d182d2a6770f099`
- runtime source: `d3146f2b56c2db4711ac8391871c9e529d1946d7`
- GPU: `NVIDIA GeForce RTX 5080, GPU-83b80720-113c-3f3d-c624-f1dc642b3f8b, 12.0, 617.14, 16303`
- Harness: one GGUF quantized tensor + one `ggml_mul_mat` graph; no `llama_decode`.
- Formal protocol: 2 warmups, 9 formal repetitions, 3 holdout repetitions; one graph and one tensor allocation are reused within each case. The canonical receipt for this report is `standalone_mmvq_qwen25_final.json`.
- Cache protocol: `warm_reuse`. The canonical receipt predates the later
  validation-readback reduction, so each formal repeat in this receipt also
  copied the output back outside the CUDA event. The harness now records that
  validation readback separately. Neither protocol claims L2 eviction or DRAM
  coldness.
- Environment gate: the final run saw active/unknown graphics contexts in the before/after `nvidia-smi pmon` samples, so it remains diagnostic-only and is not a production timing surface.

## M=1 MMVQ results

| format / tensor | shape (M,N,K) | grid / block | formal median ns | formal CV | holdout median ns | holdout error | decision |
|---|---:|---|---:|---:|---:|---:|---|
| q5_0 / q5_0_attn_q | (1, 896, 896) | `(224, 1, 1) / (32, 4, 1)` | 1472 | 1.09% | 1472 | 0.00% | diagnostic only; no prediction update |
| q5_0 / q5_0_ffn_gate | (1, 4864, 896) | `(1216, 1, 1) / (32, 4, 1)` | 3009 | 1.20% | 3040 | 1.02% | diagnostic only; no prediction update |
| q8_0 / q8_0_attn_v | (1, 128, 896) | `(32, 1, 1) / (32, 4, 1)` | 1120 | 0.00% | 1120 | 0.00% | diagnostic only; no prediction update |
| q8_0 / q8_0_output | (1, 151936, 896) | `(37984, 1, 1) / (32, 4, 1)` | 157798 | 0.35% | 157701 | 0.06% | diagnostic only; no prediction update |

## M=512 MMQ controls

The same four tensors were measured at M=512. These rows are retained as MMQ controls and are never installed into the MMVQ surface.

| format / tensor | grid / block | formal samples | holdout samples |
|---|---|---:|---:|
| q5_0 / q5_0_attn_q | `(84, 1, 1) / (32, 8, 1)` | 9 CUPTI intervals | 3 CUPTI intervals |
| q5_0 / q5_0_ffn_gate | `(152, 1, 1) / (32, 8, 1)` | 9 CUPTI intervals | 3 CUPTI intervals |
| q8_0 / q8_0_attn_v | `(84, 1, 1) / (32, 8, 1)` | 9 CUPTI intervals | 3 CUPTI intervals |
| q8_0 / q8_0_output | `(4748, 1, 1) / (32, 8, 1)` | 9 CUPTI intervals | 3 CUPTI intervals |

## Calibration decision

- The four M=1 rows have exact source geometry and canonical dispatch signatures; numerical CPU-vs-CUDA checks passed for all rows.
- After correcting the harness event/cache/graph semantics, q8_0/output is repeatable (formal CV 0.35%, holdout relative error 0.06%). The earlier 161/347 µs bimodality came from the invalid old measurement protocol and is not evidence of a CUDA source defect.
- The receipt is diagnostic evidence only. It is not installed as a rate surface: the measured `formal_median_ns` is never written into `KernelSample` or used as a correction ratio. The importer hard-fails closed for measurement-derived installation, so the analytical fallback remains in force.
- Physical bytes/work and runtime submission ownership remain unchanged; no E2E latency fit, per-kernel fitted parameter, or M=512 interpolation was applied. No `mmvq.cu` compute or dispatch correction is justified by this timing evidence.

## Source-level correction made after the measurement

The independent geometry audit found a structural prediction error that does
not require a timing fit. When an exact source/runtime contract is present,
the planner now binds `MMVQWork` per workload before kernel-model estimation;
the optional MMQ matrix-cost switch no longer gates this geometry-only path.
The contract requires the pinned `mmvq.cu`/`mmvq.cuh`/`vecdotq.cuh` hashes,
the same CUDA binary identity, CC 12.0/warp 32, ordinary contiguous 2-D
layout, the physical tensor segment and the native invocation capture. If any
piece is missing, the historical generic analytical path remains active.

For Q8_0 `(M,N,K)=(1,128,896)`, the generic descriptor had 128 CTAs while
the source-derived launch has 32 CTAs, four rows per CTA and 4 warps per CTA.
The same source derivation covers the M=2/4/5/8 branches without using their
measured durations. The change carries only launch geometry and the reduction
shared-memory lower bound; it does not add registers, efficiency, bandwidth,
latency, or occupancy coefficients from the standalone run. The CUDA kernel
source itself is unchanged.

This receipt does not bind the simulator's analytical prediction for any row,
so it cannot produce a simulator-versus-native residual or an error ranking.
The CV and formal/holdout differences above are measurement repeatability
statistics; they are not prediction errors.  A future residual report must
join an independently generated simulator estimate to these exact
`(M,N,K,format,dispatch_signature)` rows before attributing any discrepancy to
the backend.  The native timing must not be used to create that estimate.

Evidence: `standalone_mmvq_qwen25_final.json`, `standalone_mmvq_qwen25_final.sqlite`, `standalone_mmvq_qwen25_corrected.json`, `standalone_mmvq_geometry_audit.json`, and the strict importer/gate tests. The corrected receipt is retained as a semantic cross-check only; its earlier environment parser did not yet classify unknown graphics contexts, so the final receipt is the authoritative qualification decision.

## Attribution boundary

The CUPTI rows provide a source-bound dispatch observation, not a proof of a
source defect.  For all four M=1 cases the observed activity is the same
single-node `mul_mat_vec_q` family with `ncols_dst=1`, `fusion=0`,
`small_k=1`, `rows=4`, and a stable grid/block pair.  This is enough to bind
the observation to the `mmvq.cu` launch decision and the instantiated
`mul_mat_vec_q<GGML_TYPE_Q5_0|GGML_TYPE_Q8_0, ...>` kernel family.  It does not
identify a particular `mmvq.cu` branch, `vecdotq.cuh` instruction sequence, or
memory-system event as the cause of an error.

The source-side launch arithmetic is consistent with the trace: `K=896`
contains 28 quantization blocks, and the current `QI5_0`/`QI8_0` and VDR
constants make the `should_use_small_k` predicate true for both formats on
the pinned CC 12.0 build.  The same calculation leaves `halve_iters` false.
The CC 12.0 path uses the generic parameter table; the separate GB10 table is
not applicable to this GPU.  This is a cross-check of dispatch selection, not
evidence that the predicate is wrong.

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| GGUF tensor name, type, and shape | The measured quantized input and exact decode geometry | That every production decode uses this isolated tensor path |
| CUPTI kernel name, grid, and block | The actual CUDA launch and the kernel family selected by the binary | Occupancy, cache misses, DRAM traffic, clock throttling, or per-instruction cost |
| `dispatch_signature` and source hashes | The dispatch contract and the pinned `mmvq.cu`/`vecdotq.cuh` build identity | Which source line is defective |
| CPU-vs-CUDA numerical check | No correctness mismatch was observed for these isolated outputs | That simulator timing is correct, or that the full model has no numerical issue |
| Formal/holdout timing | Repeatability of the isolated activity interval | A causal explanation for q8 output's same-signature multimodal latency |

The corrected SQLite trace contains one `quantize_q8_1` activity followed by
the same `mul_mat_vec_q<q8_0,m=1,fusion=0,small_k=1,halve_iters=0>` kernel for
each q8 output M=1 range.  It selects no alternate grid, block, or kernel,
and its main-kernel interval is repeatable.  The earlier 161/347-us and
5.7-ms modes came from the invalid old harness protocol and are retained only
as debugging history; they are not used as evidence of a backend source bug.

For M=512 controls, CUPTI also records `mul_mat_q_stream_k_fixup` with a
separate `(84,4,1)/(32,4,1)` launch for the narrow Q8/Q5 control shapes. The
parser now preserves these activities as `stage=fixup` diagnostic rows and
keeps them out of the main MMQ distribution. This separates a real second
dispatch from the MMQ main kernel without treating the fixup as an MMVQ rate
sample.

An additional `GGML_CUDA_PDL=0` q8/output attempt was made as a falsifiable
execution-condition check.  That run observed active/unknown graphics
contexts and is rejected by the environment gate; it is retained only as a
lead and supplies no basis for a PDL-related source change.

One source-controlled discriminator remains unverified: this build calls
`ggml_cuda_pdl_sync()` in the MMVQ kernel and enables programmatic dependent
launches by default when `GGML_CUDA_PDL` is unset.  The rejected
`GGML_CUDA_PDL=0` attempt was contaminated by active/unknown graphics
contexts, so PDL remains a hypothesis for a future clean A/B test, not an
identified source bug or a calibration parameter.

## Non-fitting rule

These timings are retained as diagnostic evidence only.  No measured median,
ratio, percentile, or long-tail value is inserted as a simulator coefficient,
multiplier, or lookup-table entry.  The receipt has no bound analytical
baseline and the strict importer consequently installs no measured rate; the
existing analytical fallback remains active.  Any future backend change must
be justified by an independently testable source-level invariant (for
example, a demonstrably wrong dispatch predicate or indexing rule), followed
by a fresh measurement on the changed binary and a separate holdout.
