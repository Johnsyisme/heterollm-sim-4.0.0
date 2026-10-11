import json

import pytest

from heterollm_sim.kernel_model import KernelCapability, KernelModelProfile
from heterollm_sim.mmvq_work import SOURCE_SHA256
from heterollm_sim.standalone_mmvq_calibration import (
    CUDA_EVENT_BOUNDARY,
    install_standalone_mmvq_surfaces,
    load_standalone_mmvq_receipt,
)


RUNTIME_SHA = "a" * 64
GGUF_SHA = "b" * 64


def _row(*, m=1, n=896, k=896, formal=None, holdout=None, family="mmvq", stage="main", phase="decode"):
    return {
        "case_id": f"{family}-{m}-{n}-{k}",
        "stage": stage,
        "phase": phase,
        "kernel_family": family,
        "weight_format": "q5_0",
        "shape": [m, n, k],
        "analytical_ns": 100.0,
        "formal_device_ns": formal or [80.0, 82.0, 81.0, 80.5, 81.5],
        "holdout_device_ns": [81.0, 82.0, 80.5] if holdout is None else holdout,
        "grid": [n // 4, 1, 1],
        "block": [32, 4, 1],
        "dispatch_signature": "mmvq:type=6:m=1:fusion=0:small_k=1:halve_iters=0:warps=4:rows=4",
        "timing_boundary": CUDA_EVENT_BOUNDARY,
        "cache_protocol": "cold_streaming",
        "fusion_mode": "none",
        "numerical_validation": {"passed": True, "count": 1024},
    }


def _receipt(tmp_path, rows, *, clean=True):
    raw = {
        "schema": "heterollm.native-mmvq-standalone/v1",
        "status": "measured",
        "hardware_id": "nvidia-rtx-5080",
        "runtime_binary_sha256": RUNTIME_SHA,
        "gguf_sha256": GGUF_SHA,
        "runtime_components": [{"path": "ggml-cuda.dll", "sha256": "c" * 64}],
        "source_hashes": dict(SOURCE_SHA256),
        "cache_protocol": "cold_streaming",
        "qualification": {
            "exclusive_gpu": clean,
            "active_compute_processes": 0 if clean else 1,
            "diagnostics_clean": clean,
        },
        "samples": rows,
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_unqualified_environment_is_preserved_but_not_installable(tmp_path):
    receipt = load_standalone_mmvq_receipt(_receipt(tmp_path, [_row()], clean=False))
    assert receipt.samples
    assert receipt.qualified_samples == ()


def test_unbound_analytical_baseline_remains_diagnostic(tmp_path):
    row = _row()
    row.pop("analytical_ns")
    receipt = load_standalone_mmvq_receipt(_receipt(tmp_path, [row]))
    assert receipt.samples[0].analytical_ns is None
    assert receipt.samples[0].mmvq_surface_eligible is False
    assert "analytical_baseline_unbound" in receipt.samples[0].eligibility_reasons
    assert receipt.qualified_samples == ()


def test_runner_rows_alias_and_control_phase_are_loadable(tmp_path):
    mmvq = _row()
    mmvq.pop("analytical_ns")
    control = _row(m=512, family="mmq")
    control.pop("analytical_ns")
    control["phase"] = "control"
    raw_path = _receipt(tmp_path, [mmvq, control])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["rows"] = raw.pop("samples")  # runner spelling
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    receipt = load_standalone_mmvq_receipt(raw_path)
    assert len(receipt.samples) == 2
    assert all(sample.analytical_ns is None for sample in receipt.samples)
    assert receipt.qualified_samples == ()


def test_m512_mmq_control_is_not_installed(tmp_path):
    receipt = load_standalone_mmvq_receipt(_receipt(
        tmp_path, [_row(m=512, family="mmq", stage="control", phase="control")]
    ))
    assert receipt.samples[0].mmvq_surface_eligible is False
    assert receipt.qualified_samples == ()


def test_measurements_are_diagnostic_only_and_never_added_to_profile(tmp_path):
    receipt = load_standalone_mmvq_receipt(_receipt(tmp_path, [_row()]))
    owner = KernelCapability(
        "cuda_mmvq_q5_0", ("q5_0",), "fp32", "decode", "dp4a", "analytical",
        min_shape=(1, 1, 32), max_shape=(4, 2_000, 2_000),
        cta_geometry=(1, 1, 32), warps_per_cta=4, output_bits=32,
        calibration_source_bound=False,
    )
    profile = KernelModelProfile(
        "nvidia-rtx-5080", "runtime", "sm120",
        kernels=(owner,), calibration_hardware_id="nvidia-rtx-5080",
        calibration_runtime_sha256=RUNTIME_SHA,
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        install_standalone_mmvq_surfaces(
            profile, receipt, analytical_baselines={(1, 896, 896): 100.0},
            expected_gguf_sha256=GGUF_SHA,
        )
    assert profile.kernels == (owner,)


def test_missing_holdout_fails_closed(tmp_path):
    row = _row(holdout=[])
    receipt = load_standalone_mmvq_receipt(_receipt(tmp_path, [row]))
    assert receipt.qualified_samples == ()
    with pytest.raises(ValueError, match="diagnostic-only"):
        install_standalone_mmvq_surfaces(
            KernelModelProfile(
                "nvidia-rtx-5080", "runtime", "sm120",
                calibration_hardware_id="nvidia-rtx-5080",
                calibration_runtime_sha256=RUNTIME_SHA,
                kernels=(),
            ),
            receipt,
            analytical_baselines={(1, 896, 896): 100.0},
            expected_gguf_sha256=GGUF_SHA,
        )


def test_analytical_baseline_mismatch_fails_closed(tmp_path):
    receipt = load_standalone_mmvq_receipt(_receipt(tmp_path, [_row()]))
    owner = KernelCapability(
        "cuda_mmvq_q5_0", ("q5_0",), "fp32", "decode", "dp4a", "analytical",
        min_shape=(1, 1, 32), max_shape=(4, 2_000, 2_000),
        cta_geometry=(1, 1, 32), warps_per_cta=4, output_bits=32,
    )
    profile = KernelModelProfile(
        "nvidia-rtx-5080", "runtime", "sm120", kernels=(owner,),
        calibration_hardware_id="nvidia-rtx-5080",
        calibration_runtime_sha256=RUNTIME_SHA,
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        install_standalone_mmvq_surfaces(
            profile, receipt, analytical_baselines={(1, 896, 896): 99.0},
            expected_gguf_sha256=GGUF_SHA,
        )
