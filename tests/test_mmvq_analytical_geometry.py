import pytest

from heterollm_sim.mmvq_work import UnsupportedMMVQ, derive_mmvq_geometry


FORMATS = {
    # qk, qi, vdr, block bytes from the pinned source traits/structs.
    "Q1_0": (128, 4, 1, 18), "Q2_0": (64, 2, 1, 18),
    "Q4_0": (32, 4, 2, 18), "Q4_1": (32, 4, 2, 20),
    "Q5_0": (32, 4, 2, 22), "Q5_1": (32, 4, 2, 24),
    "Q8_0": (32, 8, 2, 34), "MXFP4": (32, 4, 2, 17),
    "NVFP4": (64, 8, 4, 36), "Q2_K": (256, 16, 1, 84),
    "Q3_K": (256, 16, 1, 110), "Q4_K": (256, 32, 2, 144),
    "Q5_K": (256, 32, 2, 176), "Q6_K": (256, 32, 1, 210),
    "IQ2_XXS": (256, 16, 2, 66), "IQ2_XS": (256, 16, 2, 74),
    "IQ2_S": (256, 16, 2, 82), "IQ3_XXS": (256, 16, 2, 98),
    "IQ1_S": (256, 8, 1, 50), "IQ1_M": (256, 8, 1, 56),
    "IQ4_NL": (32, 4, 2, 18), "IQ4_XS": (256, 32, 4, 136),
    "IQ3_S": (256, 16, 2, 110),
}


@pytest.mark.parametrize("weight_format,expected", FORMATS.items())
def test_analytical_rules_cover_source_mmvq_formats(weight_format, expected):
    qk, qi, vdr, block_bytes = expected
    geometry = derive_mmvq_geometry(
        m=1, k=qk, n=3, weight_format=weight_format,
    )
    assert (geometry.qk, geometry.qi, geometry.vdr, geometry.weight_block_bytes) == expected
    assert geometry.grid[0] == (3 + geometry.rows_per_cta - 1) // geometry.rows_per_cta
    assert geometry.output_tail_rows == 3 % geometry.rows_per_cta
    metadata = geometry.to_metadata()
    assert metadata["provenance"] == "source_analytical_rule_no_binary_identity"
    assert metadata["runtime_binary_sha256"] is None
    assert metadata["measured_rate"] is None


def test_generic_m_dispatch_and_ceil_output_tail():
    geometry = derive_mmvq_geometry(m=5, k=32, n=5, weight_format="Q5_0")
    assert geometry.dispatch_kind == "vector"
    assert geometry.warps_per_cta == 2
    assert geometry.rows_per_cta == 2
    assert geometry.grid == (3, 1, 1)
    assert geometry.output_tail_rows == 1
    assert geometry.block == (32, 2, 1)


def test_turing_and_rdna_table_differences_are_explicit():
    turing = derive_mmvq_geometry(
        m=1, k=32, n=7, weight_format="Q4_0",
        compute_capability=750, highest_compiled_arch=750,
    )
    assert turing.architecture_table == "turing"
    assert turing.warps_per_cta == 4
    assert turing.rows_per_cta == 4

    rdna4 = derive_mmvq_geometry(
        m=1, k=32, n=7, weight_format="Q4_0",
        compute_capability=0x1000000 + 0x1200,
    )
    assert rdna4.architecture_table == "rdna4"
    assert rdna4.warps_per_cta == 8
    # RDNA tables intentionally do not use the generic small-K row promotion.
    assert rdna4.rows_per_cta == 1
    assert not rdna4.small_k

    cdna = derive_mmvq_geometry(
        m=1, k=32, n=7, weight_format="Q4_0",
        compute_capability=0x1000000 + 0x908,
    )
    # CDNA uses the GCN launch table but has its own source ID batch table.
    assert cdna.architecture_table == "cdna"
    assert cdna.warps_per_cta == 2
    assert cdna.rows_per_cta == 2

    rdna35 = derive_mmvq_geometry(
        m=1, k=32, n=7, weight_format="Q4_0",
        compute_capability=0x1000000 + 0x1150,
    )
    assert rdna35.architecture_table == "rdna2"
    with pytest.raises(UnsupportedMMVQ, match="batch limit"):
        derive_mmvq_geometry(
            m=5, k=256, n=7, weight_format="Q4_K", has_ids=True,
            compute_capability=0x1000000 + 0x1150,
        )

    # mmvq.cu's Turing-plus branch suppresses only IQ3 small-K promotion;
    # IQ2 remains on the normal promotion path.
    turing_iq2 = derive_mmvq_geometry(
        m=1, k=256, n=7, weight_format="IQ2_XXS",
        compute_capability=750, highest_compiled_arch=750,
    )
    turing_iq3 = derive_mmvq_geometry(
        m=1, k=256, n=7, weight_format="IQ3_XXS",
        compute_capability=750, highest_compiled_arch=750,
    )
    assert turing_iq2.small_k and turing_iq2.rows_per_cta == 4
    assert not turing_iq3.small_k and turing_iq3.rows_per_cta == 1


def test_gb10_halve_iters_is_source_branch_not_a_rate_fit():
    widened = derive_mmvq_geometry(
        m=1, k=32 * 128, n=3, weight_format="Q4_0",
        compute_capability=1210, highest_compiled_arch=1210,
    )
    assert widened.architecture_table == "gb10"
    assert widened.halve_iters
    assert widened.warps_per_cta == 8
    assert widened.block == (32, 8, 1)

    odd_tail = derive_mmvq_geometry(
        m=1, k=32 * 192, n=3, weight_format="Q4_0",
        compute_capability=1210, highest_compiled_arch=1210,
    )
    assert not odd_tail.halve_iters
    assert odd_tail.warps_per_cta == 4


def test_moe_dispatch_has_ceil_rows_and_architecture_batch_limit():
    geometry = derive_mmvq_geometry(
        m=3, k=256, n=5, weight_format="Q4_K", has_ids=True,
        channels=2,
    )
    assert geometry.dispatch_kind == "moe"
    assert geometry.grid == (3, 2, 1)
    assert geometry.block == (32, 3, 1)
    assert geometry.rows_per_cta == 2
    assert geometry.output_tail_rows == 1

    fused_moe = derive_mmvq_geometry(
        m=3, k=256, n=5, weight_format="NVFP4", has_ids=True,
        has_fusion=True,
    )
    assert fused_moe.dispatch_kind == "moe" and fused_moe.has_fusion

    with pytest.raises(UnsupportedMMVQ, match="batch limit"):
        derive_mmvq_geometry(
            m=7, k=32, n=5, weight_format="Q4_0", has_ids=True,
            compute_capability=600, highest_compiled_arch=600,
        )


def test_fusion_and_invalid_shapes_remain_source_guarded():
    fused = derive_mmvq_geometry(m=1, k=32, n=2, weight_format="Q8_0", has_fusion=True)
    assert fused.has_fusion and fused.dispatch_kind == "vector"
    with pytest.raises(UnsupportedMMVQ, match="fusion"):
        derive_mmvq_geometry(m=2, k=32, n=2, weight_format="Q8_0", has_fusion=True)
    with pytest.raises(UnsupportedMMVQ, match="divisible"):
        derive_mmvq_geometry(m=1, k=31, n=2, weight_format="Q8_0")

