"""Source-derived CC1200 MMVQ geometry, with no latency or bandwidth model.

Ordinary contiguous Q5_0/Q8_0 MUL_MAT is described by default. An explicit
issue-bound treatment can also derive the locked Q4_K/Q6_K geometry.  An explicit runtime
contract is required; this module never chooses MMVQ from a model name and
never assumes that a CTA count is an HBM utilization percentage.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

SOURCE_SHA256 = MappingProxyType({
    'ggml-cuda/mmvq.cu': '14026871030393662628abdbd4937d5cab72031e20ddf582c9de1d7b424bb368',
    'ggml-cuda/mmvq.cuh': '93ef0ea631585f93c35b00087d265e4ca45f82aaca8e7c363d382dc561d85fbd',
    'ggml-cuda/vecdotq.cuh': '9d165e8e36db9bdfb69bfc2301e550a58c4a3057907bbf4d075b5f3eb72347a2',
    'ggml-common.h': '3ac6eed12695ceea1acd18f556845023a45f92f6ce0530c18c70bb10450207ea',
    'ggml-cuda/common.cuh': '1cc3186a56426d5b929e9c29f9fae2d61d583f5f7abf0764d363c21cbbcbe932',
})

class UnsupportedMMVQ(ValueError):
    """The narrow source/runtime contract cannot describe this invocation."""


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise UnsupportedMMVQ(name + ' must be a positive integer')
    return value


@dataclass(frozen=True)
class MMVQSourceContract:
    compute_capability: int
    highest_compiled_arch: int
    warp_size: int
    source_hashes: Mapping[str, str]
    runtime_binary_sha256: str
    ordinary_contiguous_2d: bool
    force_cublas: bool
    mmvq_dispatch_enabled: bool
    has_ids: bool = False
    has_fusion: bool = False
    channels: int = 1
    samples: int = 1

    def __post_init__(self) -> None:
        if dict(self.source_hashes) != dict(SOURCE_SHA256):
            raise UnsupportedMMVQ('immutable MMVQ source identity differs')
        object.__setattr__(self, 'source_hashes', MappingProxyType(dict(self.source_hashes)))
        if (type(self.compute_capability) is not int or self.compute_capability != 1200
                or type(self.highest_compiled_arch) is not int or self.highest_compiled_arch != 1200
                or type(self.warp_size) is not int or self.warp_size != 32):
            raise UnsupportedMMVQ('requires CC1200 compiled CC1200 and warp32')
        if (not isinstance(self.runtime_binary_sha256, str) or len(self.runtime_binary_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.runtime_binary_sha256)):
            raise UnsupportedMMVQ('runtime binary SHA required')
        if (self.ordinary_contiguous_2d is not True or self.force_cublas is not False
                or self.mmvq_dispatch_enabled is not True or self.has_ids is not False
                or self.has_fusion is not False or type(self.channels) is not int or self.channels != 1
                or type(self.samples) is not int or self.samples != 1):
            raise UnsupportedMMVQ('only declared ordinary unfused single-channel/sample MMVQ supported')


def verify_mmvq_source_tree(src_directory: str | Path) -> Mapping[str, str]:
    """Optional explicit read-only identity check; importing performs no IO."""
    root = Path(src_directory)
    actual = {}
    for relative, expected in SOURCE_SHA256.items():
        with (root / relative).open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != expected:
            raise UnsupportedMMVQ('source hash mismatch: ' + relative)
        actual[relative] = digest
    return MappingProxyType(actual)


@dataclass(frozen=True)
class MMVQWork:
    m: int
    k: int
    n: int
    weight_format: str
    runtime_binary_sha256: str
    qk: int
    qi: int
    vdr: int
    warps_per_cta: int
    rows_per_cta: int
    small_k: bool
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    blocks_per_row: int
    blocks_per_loop_step: int
    loop_iterations_by_thread: tuple[int, ...]
    weight_block_bytes: int

    @property
    def cta_count(self) -> int:
        return self.grid[0]

    @property
    def launched_warps(self) -> int:
        return self.cta_count * self.warps_per_cta

    @property
    def active_k_threads_per_cta(self) -> int:
        return sum(count > 0 for count in self.loop_iterations_by_thread)

    @property
    def maximum_thread_k_iterations(self) -> int:
        return max(self.loop_iterations_by_thread)

    @property
    def source_vector_dot_calls(self) -> int:
        return sum(self.loop_iterations_by_thread) * self.m * self.rows_per_cta * self.cta_count

    @property
    def logical_weight_bytes(self) -> int:
        return self.n * self.blocks_per_row * self.weight_block_bytes

    @property
    def logical_input_f32_bytes(self) -> int:
        return 4 * self.m * self.k

    @property
    def consumer_q8_1_unique_bytes(self) -> int:
        return 36 * self.m * self.k // 32

    @property
    def output_f32_bytes(self) -> int:
        return 4 * self.m * self.n

    @property
    def reduction_shared_array_bytes(self) -> int:
        # Main tmp_shared only. The unused gate array and compiler elimination
        # are not a proof of final cubin static allocation.
        return 4 * max(1, self.warps_per_cta - 1) * self.m * self.rows_per_cta * 32

    def to_metadata(self) -> dict[str, object]:
        return {**asdict(self), 'schema': 'heterollm.source-mmvq-work/v1',
            'source_hashes': dict(SOURCE_SHA256), 'parameter_table': 'MMVQ_PARAMETERS_GENERIC',
            'template_batch': self.m, 'has_fusion': False, 'halve_iters': False,
            'cta_count': self.cta_count, 'launched_warps': self.launched_warps,
            'active_k_threads_per_cta': self.active_k_threads_per_cta,
            'maximum_thread_k_iterations': self.maximum_thread_k_iterations,
            'source_vector_dot_calls': self.source_vector_dot_calls,
            'logical_weight_bytes': self.logical_weight_bytes,
            'logical_input_f32_bytes': self.logical_input_f32_bytes,
            'consumer_q8_1_unique_bytes': self.consumer_q8_1_unique_bytes,
            'output_f32_bytes': self.output_f32_bytes,
            'reduction_shared_array_bytes': self.reduction_shared_array_bytes,
            'dynamic_shared_launch_bytes': 0, 'cta_barriers': 1,
            'final_warp_reduce_rounds': 5, 'native_dispatch_proven': False,
            'binary_source_equivalence_proven': False, 'cost_model_applied': False,
            'hbm_efficiency': None, 'measured_occupancy': None,
            'work_units': 'source vector-dot calls and logical bytes; not SASS instructions or DRAM transactions',
            'timing_completeness': 'source_geometry_only_no_cost_rates',
            'utilization_candidate': 'Use vector CTA/warp/K-loop geometry instead of MMA tiles only after independent bandwidth/occupancy validation.'}


# The following table is copied from the source-level type traits, vec_dot
# dispatch, and block declarations in the pinned llama.cpp tree.  It is used
# only by ``derive_mmvq_geometry`` below.  Unlike MMVQWork, the analytical
# descriptor deliberately has no runtime binary identity and never carries a
# measured rate.
_ANALYTICAL_FORMATS = {
    # name: qk, qi, vdr, sizeof(block_*)
    'Q1_0': (128, 4, 1, 18),
    'Q2_0': (64, 2, 1, 18),
    'Q4_0': (32, 4, 2, 18),
    'Q4_1': (32, 4, 2, 20),
    'Q5_0': (32, 4, 2, 22),
    'Q5_1': (32, 4, 2, 24),
    'Q8_0': (32, 8, 2, 34),
    'MXFP4': (32, 4, 2, 17),
    'NVFP4': (64, 8, 4, 36),
    'Q2_K': (256, 16, 1, 84),
    'Q3_K': (256, 16, 1, 110),
    'Q4_K': (256, 32, 2, 144),
    'Q5_K': (256, 32, 2, 176),
    'Q6_K': (256, 32, 1, 210),
    'IQ2_XXS': (256, 16, 2, 66),
    'IQ2_XS': (256, 16, 2, 74),
    'IQ2_S': (256, 16, 2, 82),
    'IQ3_XXS': (256, 16, 2, 98),
    'IQ1_S': (256, 8, 1, 50),
    'IQ1_M': (256, 8, 1, 56),
    'IQ4_NL': (32, 4, 2, 18),
    'IQ4_XS': (256, 32, 4, 136),
    'IQ3_S': (256, 16, 2, 110),
}
# Public read-only format inventory for planner qualification.  Keep the
# detailed tuple table private so callers cannot accidentally alter source
# geometry; only the format names are needed by an analytical gate.
ANALYTICAL_MMVQ_FORMATS = frozenset(_ANALYTICAL_FORMATS)

_MMVQ_MAX_BATCH_SIZE = 8
_NVIDIA_CC_VOLTA = 700
_NVIDIA_CC_TURING = 750
_NVIDIA_CC_AMPERE = 800
_NVIDIA_CC_ADA = 890
_NVIDIA_CC_BLACKWELL = 1200
_NVIDIA_CC_DGX_SPARK = 1210
_AMD_OFFSET = 0x1000000
_AMD_CDNA1 = _AMD_OFFSET + 0x908
_AMD_CDNA2 = _AMD_OFFSET + 0x90A
_AMD_CDNA3 = _AMD_OFFSET + 0x942
_AMD_CDNA4 = _AMD_OFFSET + 0x950
_AMD_RDNA1 = _AMD_OFFSET + 0x1010
_AMD_RDNA2 = _AMD_OFFSET + 0x1030
_AMD_RDNA3 = _AMD_OFFSET + 0x1100
_AMD_RDNA3_5 = _AMD_OFFSET + 0x1150
_AMD_RDNA4 = _AMD_OFFSET + 0x1200

_MMVQ_MAX_BATCH_PASCAL = {
    'IQ1_S': 6, 'IQ1_M': 6, 'IQ2_S': 4, 'IQ2_XS': 5, 'IQ2_XXS': 5,
    'IQ3_S': 4, 'IQ3_XXS': 4, 'IQ4_NL': 6, 'IQ4_XS': 5,
    'Q2_K': 4, 'Q3_K': 4, 'Q4_0': 6, 'Q4_1': 6, 'Q4_K': 5,
    'Q5_0': 6, 'Q5_1': 6, 'Q5_K': 5, 'Q6_K': 4, 'Q8_0': 4,
}
_MMVQ_MAX_BATCH_TURING = {'IQ2_S': 7, 'IQ3_S': 6, 'IQ3_XXS': 7,
                          'Q2_K': 7, 'Q3_K': 5}
_MMVQ_MAX_BATCH_GCN = {
    'IQ1_S': 5, 'IQ1_M': 5, 'IQ2_S': 4, 'IQ2_XS': 4, 'IQ2_XXS': 4,
    'IQ3_S': 4, 'IQ3_XXS': 4, 'IQ4_NL': 6, 'IQ4_XS': 4,
    'Q2_K': 4, 'Q3_K': 4, 'Q4_0': 5, 'Q4_1': 5, 'Q4_K': 4,
    'Q5_K': 4, 'Q6_K': 4, 'Q8_0': 4,
}
_MMVQ_MAX_BATCH_RDNA2 = {
    'IQ2_S': 4, 'IQ2_XS': 4, 'IQ2_XXS': 4, 'IQ3_S': 4, 'IQ3_XXS': 4,
    'Q2_K': 7, 'Q3_K': 4, 'Q4_K': 5, 'Q5_K': 6, 'Q6_K': 5,
}
_MMVQ_MAX_BATCH_RDNA3 = {
    'IQ1_S': 6, 'IQ1_M': 6, 'IQ2_S': 4, 'IQ2_XS': 4, 'IQ2_XXS': 4,
    'IQ3_S': 4, 'IQ3_XXS': 4, 'IQ4_NL': 6, 'IQ4_XS': 6,
    'Q3_K': 4, 'Q4_K': 4, 'Q5_K': 4, 'Q6_K': 4,
}
_MMVQ_MAX_BATCH_RDNA4 = {
    'IQ1_S': 7, 'IQ1_M': 7, 'IQ2_S': 4, 'IQ2_XS': 4, 'IQ2_XXS': 4,
    'IQ3_S': 4, 'IQ3_XXS': 4, 'IQ4_NL': 7, 'IQ4_XS': 5,
    'Q3_K': 4, 'Q4_K': 4, 'Q5_K': 4, 'Q6_K': 5,
}
_MMVQ_MAX_BATCH_CDNA1 = {
    'IQ1_S': 5, 'IQ2_XXS': 6, 'IQ3_S': 6, 'IQ4_XS': 6,
    'Q2_K': 4, 'Q3_K': 3, 'Q4_0': 7, 'Q4_1': 7,
    'Q4_K': 2, 'Q5_1': 7, 'Q5_K': 3, 'Q6_K': 4, 'Q8_0': 6,
}
_MMVQ_MAX_BATCH_CDNA = {
    'IQ2_S': 5, 'IQ2_XS': 5, 'IQ2_XXS': 5,
    'IQ3_S': 4, 'IQ3_XXS': 5,
}


@dataclass(frozen=True)
class MMVQGeometry:
    """Source-rule MMVQ geometry without binary authentication or timing.

    This descriptor intentionally cannot be passed where ``MMVQWork`` is
    required.  It is an analytical dispatch fact: it describes the source
    launch shape, K-loop partition and output-tail guard, while keeping
    runtime binary identity and performance calibration out of the result.
    """

    m: int
    k: int
    n: int
    weight_format: str
    compute_capability: int
    highest_compiled_arch: int
    warp_size: int
    architecture_table: str
    dispatch_kind: str
    has_ids: bool
    has_fusion: bool
    qk: int
    qi: int
    vdr: int
    warps_per_cta: int
    rows_per_cta: int
    small_k: bool
    halve_iters: bool
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    blocks_per_row: int
    blocks_per_loop_step: int
    loop_iterations_by_thread: tuple[int, ...]
    weight_block_bytes: int
    max_batch_size: int

    @property
    def cta_count(self) -> int:
        return self.grid[0]

    @property
    def runtime_binary_sha256(self) -> None:
        """Analytical rules intentionally have no binary identity."""
        return None

    @property
    def output_tail_rows(self) -> int:
        return self.n % self.rows_per_cta

    @property
    def launched_warps(self) -> int:
        # The MoE path has one independent warp per token in threadIdx.y.
        return self.cta_count * (self.m if self.dispatch_kind == 'moe' else self.warps_per_cta)

    @property
    def active_k_threads_per_cta(self) -> int:
        return sum(count > 0 for count in self.loop_iterations_by_thread)

    @property
    def maximum_thread_k_iterations(self) -> int:
        return max(self.loop_iterations_by_thread)

    @property
    def source_vector_dot_calls(self) -> int:
        return (sum(self.loop_iterations_by_thread) * self.m * self.rows_per_cta * self.cta_count)

    @property
    def logical_weight_bytes(self) -> int:
        return self.n * self.blocks_per_row * self.weight_block_bytes

    @property
    def logical_input_f32_bytes(self) -> int:
        return 4 * self.m * self.k

    @property
    def consumer_q8_1_unique_bytes(self) -> int:
        return 36 * self.m * self.k // 32

    @property
    def output_f32_bytes(self) -> int:
        return 4 * self.m * self.n

    @property
    def reduction_shared_array_bytes(self) -> int:
        # Keep the same structural estimate used by MMVQWork.  This is a
        # resource-shape fact, not a measured occupancy or timing rate.
        return 4 * max(1, self.warps_per_cta - 1) * self.m * self.rows_per_cta * 32

    def to_metadata(self) -> dict[str, object]:
        """Serialize facts while explicitly marking their analytical status."""
        return {
            **asdict(self),
            'schema': 'heterollm.analytical-mmvq-geometry/v1',
            'source_geometry_kind': 'analytical_source_rule',
            'source_rule_basis': 'mmvq.cu + mmvq.cuh + vecdotq.cuh + common.cuh + ggml-common.h',
            'provenance': 'source_analytical_rule_no_binary_identity',
            'native_dispatch_proven': False,
            'binary_source_equivalence_proven': False,
            'cost_model_applied': False,
            'measured_rate': None,
            'runtime_binary_sha256': None,
            'output_tail_rows': self.output_tail_rows,
            'cta_count': self.cta_count,
            'launched_warps': self.launched_warps,
            'active_k_threads_per_cta': self.active_k_threads_per_cta,
            'maximum_thread_k_iterations': self.maximum_thread_k_iterations,
            'source_vector_dot_calls': self.source_vector_dot_calls,
            'logical_weight_bytes': self.logical_weight_bytes,
            'logical_input_f32_bytes': self.logical_input_f32_bytes,
            'consumer_q8_1_unique_bytes': self.consumer_q8_1_unique_bytes,
            'output_f32_bytes': self.output_f32_bytes,
        }


def _analytical_architecture_table(compute_capability: int, highest_compiled_arch: int) -> str:
    """Mirror ``get_device_table_id`` for the source's CUDA/AMD families."""
    if type(compute_capability) is not int or type(highest_compiled_arch) is not int:
        raise UnsupportedMMVQ('compute capabilities must be integers')
    if compute_capability >= _AMD_OFFSET:
        if compute_capability in {_AMD_CDNA1, _AMD_CDNA2, _AMD_CDNA3, _AMD_CDNA4}:
            return 'cdna'
        if _AMD_CDNA1 <= compute_capability < _AMD_RDNA1 and compute_capability not in {
                _AMD_OFFSET + 0x909, _AMD_OFFSET + 0x90C}:
            return 'cdna'
        if compute_capability >= _AMD_RDNA4:
            return 'rdna4'
        if _AMD_RDNA3 <= compute_capability < _AMD_RDNA3_5:
            return 'rdna3_0'
        if (_AMD_RDNA2 <= compute_capability < _AMD_RDNA3) or (_AMD_RDNA3_5 <= compute_capability < _AMD_RDNA4):
            return 'rdna2'
        return 'gcn'
    if highest_compiled_arch == _NVIDIA_CC_DGX_SPARK:
        return 'gb10'
    if _NVIDIA_CC_TURING <= highest_compiled_arch < _NVIDIA_CC_AMPERE:
        return 'turing'
    return 'generic'


def _analytical_mmvq_mmid_max_batch(weight_format: str, compute_capability: int, table: str) -> int:
    if compute_capability < _AMD_OFFSET:
        if compute_capability == _NVIDIA_CC_VOLTA or compute_capability >= _NVIDIA_CC_ADA:
            return _MMVQ_MAX_BATCH_SIZE
        if compute_capability >= _NVIDIA_CC_TURING:
            return _MMVQ_MAX_BATCH_TURING.get(weight_format, _MMVQ_MAX_BATCH_SIZE)
        return _MMVQ_MAX_BATCH_PASCAL.get(weight_format, _MMVQ_MAX_BATCH_SIZE)
    if table == 'rdna4':
        return _MMVQ_MAX_BATCH_RDNA4.get(weight_format, _MMVQ_MAX_BATCH_SIZE)
    if table == 'cdna':
        if compute_capability == _AMD_CDNA1:
            return _MMVQ_MAX_BATCH_CDNA1.get(weight_format, _MMVQ_MAX_BATCH_SIZE)
        return _MMVQ_MAX_BATCH_CDNA.get(weight_format, _MMVQ_MAX_BATCH_SIZE)
    if table == 'rdna3_0' or table == 'rdna2':
        # Source calc_nwarps groups RDNA3.5 with RDNA2, while its host
        # MUL_MAT_ID batch table groups RDNA3.5 with RDNA3.
        batch_table = (_MMVQ_MAX_BATCH_RDNA3
                       if table == 'rdna3_0' or compute_capability >= _AMD_RDNA3_5
                       else _MMVQ_MAX_BATCH_RDNA2)
        return batch_table.get(weight_format, _MMVQ_MAX_BATCH_SIZE)
    return _MMVQ_MAX_BATCH_GCN.get(weight_format, _MMVQ_MAX_BATCH_SIZE)


def _analytical_nwarps(weight_format: str, m: int, table: str, *, small_k: bool = False,
                       halve_iters: bool = False) -> int:
    if table == 'generic':
        if 1 <= m <= 4:
            return 4
        if 5 <= m <= 8:
            return 2
        return 1
    if table in {'gcn', 'cdna'}:
        return 2 if 1 <= m <= 4 else 1
    if table == 'rdna4':
        return 8 if m == 1 and weight_format in {
            'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'Q2_K', 'Q4_K', 'Q5_K', 'Q6_K',
            'IQ4_NL', 'IQ4_XS'} else 1
    if table == 'rdna3_0':
        if m == 1 and weight_format in {'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'IQ4_NL'}:
            return 8
        if m == 1 and weight_format == 'Q6_K':
            return 2
        return 1
    if table == 'turing':
        if m == 1:
            return 2 if weight_format in {'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K'} else 4
        if 2 <= m <= 4:
            return 4
        if 5 <= m <= 8:
            return 2
        return 1
    if table == 'gb10':
        generic = _analytical_nwarps(weight_format, m, 'generic')
        if (m == 1 and not small_k and halve_iters and weight_format in {
                'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'Q4_K', 'Q5_K', 'Q6_K', 'IQ4_NL'}):
            return 2 * generic
        return generic
    # RDNA2 has no special calc_nwarps branch in mmvq.cu.
    return 1


def _analytical_rows(m: int, table: str, small_k: bool, nwarps: int) -> int:
    if table in {'generic', 'gcn', 'cdna', 'turing', 'gb10'}:
        if m == 1:
            return nwarps if small_k else 1
        if 2 <= m <= 8:
            return 2
    return 1


def _analytical_small_k(weight_format: str, m: int, blocks_per_row: int, table: str,
                        compute_capability: int, warp_size: int, vdr: int, qi: int) -> bool:
    baseline_warps = _analytical_nwarps(weight_format, m, table)
    use = baseline_warps > 1 and blocks_per_row < baseline_warps * (vdr * warp_size // qi)
    if m != 1:
        return use
    is_nvidia_turing_plus = compute_capability < _AMD_OFFSET and compute_capability >= _NVIDIA_CC_TURING
    is_nvidia_pascal_older = compute_capability < _AMD_OFFSET and compute_capability < _NVIDIA_CC_VOLTA
    if is_nvidia_turing_plus:
        # The source has a narrower Turing-plus exception: only IQ3_XXS/IQ3_S
        # avoid small-K promotion on that branch.  The broader slow-IQ list
        # below is an ``else if`` for non-Turing NVIDIA/AMD paths.
        if table == 'turing' and weight_format in {'IQ3_XXS', 'IQ3_S'}:
            return False
    elif ((weight_format in {'IQ1_S', 'IQ1_M', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ3_XXS', 'IQ3_S', 'IQ4_XS'})
          or (is_nvidia_pascal_older and weight_format in {'Q2_K', 'Q3_K'})
          or (compute_capability >= _AMD_RDNA1)):
        return False
    return use


def _analytical_halve_iters(weight_format: str, m: int, blocks_per_row: int, table: str,
                            has_ids: bool, warp_size: int, qi: int, vdr: int) -> bool:
    if table != 'gb10' or m != 1 or has_ids:
        return False
    base_warps = _analytical_nwarps(weight_format, 1, table, halve_iters=False)
    blocks_per_iter_1warp = vdr * warp_size // qi
    blocks_per_iter = base_warps * blocks_per_iter_1warp
    iters = (blocks_per_row + blocks_per_iter - 1) // blocks_per_iter
    iters_wide = (blocks_per_row + blocks_per_iter * 2 - 1) // (blocks_per_iter * 2)
    idle = iters_wide * 2 - iters
    return (weight_format in {
        'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'Q4_K', 'Q5_K', 'Q6_K', 'IQ4_NL'}
            and idle * 8 <= iters_wide * 2)


def derive_mmvq_geometry(*, m: int, k: int, n: int, weight_format: str,
                         compute_capability: int = _NVIDIA_CC_BLACKWELL,
                         highest_compiled_arch: int | None = None,
                         warp_size: int = 32, has_ids: bool = False,
                         has_fusion: bool = False, channels: int = 1,
                         samples: int = 1) -> MMVQGeometry:
    """Derive shared source dispatch geometry without binary/timing claims.

    ``m`` is the source ``ncols_dst`` dispatch domain (1..8, with the
    architecture/type-specific limit used by ``MUL_MAT_ID`` when ``has_ids``
    is true), ``n`` is the output-row count, and ``k`` must be an integral
    number of source quantization blocks.  Partial output rows are represented
    with a ceil grid and the source's write-boundary guard; they are not
    rejected as if the source required exact CTA tiling.
    """
    for value, name in ((m, 'M'), (k, 'K'), (n, 'N'), (channels, 'channels'), (samples, 'samples')):
        _positive(value, name)
    if not isinstance(weight_format, str):
        raise UnsupportedMMVQ('weight_format must be a string')
    weight_format = weight_format.upper()
    if weight_format not in _ANALYTICAL_FORMATS:
        raise UnsupportedMMVQ('format is not dispatched by source MMVQ')
    for value, name in ((has_ids, 'has_ids'), (has_fusion, 'has_fusion')):
        if type(value) is not bool:
            raise UnsupportedMMVQ(name + ' must be boolean')
    if type(warp_size) is not int or warp_size != 32:
        raise UnsupportedMMVQ('source MMVQ analytical rule currently supports warp32 only')
    if highest_compiled_arch is None:
        highest_compiled_arch = compute_capability
    _positive(compute_capability, 'compute_capability')
    _positive(highest_compiled_arch, 'highest_compiled_arch')
    if type(highest_compiled_arch) is not int:
        raise UnsupportedMMVQ('highest_compiled_arch must be an integer')
    table = _analytical_architecture_table(compute_capability, highest_compiled_arch)
    qk, qi, vdr, block_bytes = _ANALYTICAL_FORMATS[weight_format]
    if k % qk:
        raise UnsupportedMMVQ('K must be divisible by the source quantization block')
    max_batch = (_analytical_mmvq_mmid_max_batch(weight_format, compute_capability, table)
                 if has_ids else _MMVQ_MAX_BATCH_SIZE)
    # Blackwell's source dispatch has narrower K-format limits for both the
    # ordinary and ID entry points; Q5_0/Q8_0 and IQ formats retain eight.
    if compute_capability == _NVIDIA_CC_BLACKWELL:
        max_batch = {
            "Q2_K": 5, "Q3_K": 5, "Q4_K": 5,
            "Q5_K": 6, "Q6_K": 7,
        }.get(weight_format, _MMVQ_MAX_BATCH_SIZE)
    if m > max_batch:
        raise UnsupportedMMVQ('M exceeds source MMVQ dispatch batch limit')
    if has_fusion and m != 1 and not (has_ids and m > 1):
        raise UnsupportedMMVQ('source vector fusion specialization is only instantiated for M=1')
    if has_ids and m > 1 and samples != 1:
        raise UnsupportedMMVQ('source MMVQ MoE dispatch has no sample grid dimension')

    blocks_per_row = k // qk
    if has_ids and m > 1:
        # mul_mat_vec_q_moe: fixed c_rows_per_block=2, one warp per token.
        dispatch_kind = 'moe'
        small_k = False
        halve_iters = False
        warps = 1
        rows = 2
        blocks_per_step = vdr * warp_size // qi
        block = (warp_size, m, 1)
        grid = ((n + rows - 1) // rows, channels, 1)
        thread_count = warp_size
    else:
        dispatch_kind = 'vector'
        small_k = _analytical_small_k(weight_format, m, blocks_per_row, table,
                                      compute_capability, warp_size, vdr, qi)
        halve_iters = _analytical_halve_iters(weight_format, m, blocks_per_row, table,
                                              has_ids, warp_size, qi, vdr)
        warps = _analytical_nwarps(weight_format, m, table, small_k=small_k,
                                   halve_iters=halve_iters)
        rows = _analytical_rows(m, table, small_k, warps)
        blocks_per_step = vdr * warps * warp_size // qi
        block = (warp_size, warps, 1)
        grid = ((n + rows - 1) // rows, channels, samples)
        thread_count = warp_size * warps
    group = qi // vdr
    iterations = tuple(max(0, (blocks_per_row - 1 - tid // group) // blocks_per_step + 1)
                       for tid in range(thread_count))
    return MMVQGeometry(
        m, k, n, weight_format, compute_capability, highest_compiled_arch, warp_size,
        table, dispatch_kind, has_ids, has_fusion, qk, qi, vdr, warps, rows,
        small_k, halve_iters, grid, block, blocks_per_row, blocks_per_step,
        iterations, block_bytes, max_batch,
    )


def derive_mmvq_work(*, m: int, k: int, n: int, weight_format: str,
                     contract: MMVQSourceContract, allow_k_formats: bool = False, allow_iq4_xs: bool = False) -> MMVQWork:
    """Derive exact source launch geometry, not GPU performance.

    Default Q5_0/Q8_0 generic CC1200 dispatch accepts M<=8; explicit
    K-format opt-in admits Q4_K M<=5 and Q6_K M<=7. M1 may
    increase rows/CTA to four when logical K-blocks fit the strict small-K
    predicate. M2..4 use four warps/two rows; M5..8 two warps/two rows.
    Odd/partial output tiles require independent weight-tail allocation proof
    and are deliberately not handled by this initial contract.
    """
    if not isinstance(contract, MMVQSourceContract):
        raise UnsupportedMMVQ('explicit source/runtime contract required')
    for value, name in ((m, 'M'), (k, 'K'), (n, 'N')):
        _positive(value, name)
    if type(allow_k_formats) is not bool:
        raise UnsupportedMMVQ('K-format geometry requires explicit boolean opt-in')
    parameters = {'Q5_0': (32, 4, 2, 22, 8), 'Q8_0': (32, 8, 2, 34, 8)}
    if allow_k_formats:
        # Generic CC1200 launch table, ggml-common QI/QR and vecdotq VDR.
        parameters.update(Q4_K=(256, 32, 2, 144, 5), Q5_K=(256, 32, 2, 176, 6), Q6_K=(256, 32, 1, 210, 7))
    if type(allow_iq4_xs) is not bool:
        raise UnsupportedMMVQ('IQ4_XS geometry requires explicit boolean opt-in')
    if allow_iq4_xs:
        # Locked common.h QK=256, QI=32, block=136; vecdotq VDR=4.
        parameters['IQ4_XS'] = (256, 32, 4, 136, 8)
    if not isinstance(weight_format, str) or weight_format not in parameters:
        raise UnsupportedMMVQ('format outside explicitly enabled MMVQ geometry scope')
    qk, qi, vdr, block_bytes, max_m = parameters[weight_format]
    if m > max_m or k % qk:
        raise UnsupportedMMVQ('outside source MMVQ batch or logical K block domain')
    warp = 32
    warps = 4 if m <= 4 else 2
    qblocks = k // qk
    small = m == 1 and qblocks < warps * vdr * warp // qi
    rows = warps if small else 1 if m == 1 else 2
    if n % rows:
        raise UnsupportedMMVQ('partial output CTA weight-tail allocation not proven')
    step = vdr * warps * warp // qi
    iterations = tuple(max(0, (qblocks - 1 - tid // (qi // vdr)) // step + 1)
                       for tid in range(warp * warps))
    return MMVQWork(m, k, n, weight_format, contract.runtime_binary_sha256, qk, qi, vdr,
                   warps, rows, small, (n // rows, 1, 1), (warp, warps, 1),
                   qblocks, step, iterations, block_bytes)
