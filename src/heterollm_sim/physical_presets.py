"""Explicit equivalent geometries for authored hardware presets.

These builders live below component/architecture catalogs. They never run as
an execution fallback: a simulation component must carry the complete result
in ``metadata.physical_memory_config`` and pass ``physical_contract``.
Geometry and array timings here are analytical assumptions, not a DRAM/NAND
part database. Low-level ``memory_types.make_*`` helpers remain toy defaults.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .memory_types import DramConfig, NandConfig


def make_analytical_dram_config(
    *, kind: str, generation: str, capacity_bytes: int,
    read_bandwidth_gbps: float, write_bandwidth_gbps: float,
    channels: int = 1, data_width_bits: int = 64,
    read_latency_ns: float = 35.0, write_latency_ns: float = 35.0,
    scope: str = "equivalent_memory_subsystem",
) -> Dict[str, Any]:
    """Represent published capacity/bandwidth with explicitly synthetic geometry."""
    groups, banks, row_bytes, burst_bytes = 4, 4, 8192, 64
    bank_count = channels * groups * banks
    rows = (capacity_bytes + bank_count * row_bytes - 1) // (bank_count * row_bytes)
    interface_gb_s = max(read_bandwidth_gbps, write_bandwidth_gbps) / 8.0
    lane_gb_s = interface_gb_s / channels
    return asdict(DramConfig(
        kind=kind, generation=generation, channels=channels,
        ranks_per_channel=1, bank_groups_per_rank=groups,
        banks_per_group=banks, rows_per_bank=rows, row_bytes=row_bytes,
        burst_bytes=burst_bytes, interleave_bytes=burst_bytes,
        data_width_bits=data_width_bits,
        data_rate_mt_s=interface_gb_s * 8_000.0 / (channels * data_width_bits),
        interface_bandwidth_gb_s=interface_gb_s,
        read_bandwidth_gb_s=read_bandwidth_gbps / 8.0,
        write_bandwidth_gb_s=write_bandwidth_gbps / 8.0,
        open_ns=14.0, close_ns=14.0,
        read_latency_ns=read_latency_ns, write_latency_ns=write_latency_ns,
        burst_interval_ns=burst_bytes / lane_gb_s,
        capacity_bytes=capacity_bytes, max_outstanding_requests=64,
        metadata={
            "parameter_basis": "analytical_equivalent_assumptions",
            "configuration_scope": scope,
            "known_values": "component capacity and one-way directional bandwidth",
            "analytical_assumptions": [
                "channel count unless separately cited by the preset",
                "rank, bank, row and burst organization",
                "open/close, command-to-data and recovery timings",
                "burst interval is a payload scheduling budget, not measured tCCD",
                "bank-group column/ACT/WRITE-to-READ timings and four-ACT window are unspecified (zero disables each constraint)",
            ],
            "capacity_basis": "declared capacity with an equivalent geometry rounded upward to whole rows",
            "bandwidth_basis": "explicit component one-way bandwidth; synthetic lane width does not infer pin speed",
            "data_rate_basis": "equivalent interface rate derived from declared aggregate bandwidth and synthetic width; not a measured pin speed",
        },
    ))


def make_analytical_nand_config(
    *, kind: str, capacity_bytes: int, read_bandwidth_gbps: float,
    write_bandwidth_gbps: float, read_latency_ns: float,
    write_latency_ns: float, scope: str = "equivalent_flash_device",
) -> Dict[str, Any]:
    """Model a NAND-backed device without claiming undisclosed die layout."""
    channels, targets, dies, luns, planes = 8, 2, 2, 1, 2
    pages, page_bytes = 256, 16384
    block_bytes = channels * targets * dies * luns * planes * pages * page_bytes
    blocks = (capacity_bytes + block_bytes - 1) // block_bytes
    return asdict(NandConfig(
        kind=kind, channels=channels, targets_per_channel=targets,
        dies_per_target=dies, luns_per_die=luns, planes_per_lun=planes,
        blocks_per_plane=blocks, pages_per_block=pages, page_bytes=page_bytes,
        host_granularity_bytes=4096,
        host_bandwidth_gb_s=max(read_bandwidth_gbps, write_bandwidth_gbps) / 8.0,
        internal_bandwidth_gb_s=max(read_bandwidth_gbps, write_bandwidth_gbps) / 8.0,
        host_read_bandwidth_gb_s=read_bandwidth_gbps / 8.0,
        host_write_bandwidth_gb_s=write_bandwidth_gbps / 8.0,
        page_read_ns=read_latency_ns, page_program_ns=write_latency_ns,
        block_erase_ns=3_000_000.0,
        capacity_bytes=capacity_bytes, max_outstanding_requests=64,
        write_buffer_bytes=0,
        metadata={
            "parameter_basis": "analytical_equivalent_assumptions",
            "configuration_scope": scope,
            "known_values": "component capacity and sequential read/write bandwidth",
            "analytical_assumptions": [
                "channel, target, die, LUN, plane, block and page organization",
                "page read/program and block erase timings",
                "internal bandwidth and scheduling parallelism",
            ],
            "directional_bandwidth_gb_s": {
                "read": read_bandwidth_gbps / 8.0,
                "write": write_bandwidth_gbps / 8.0,
            },
            "capacity_basis": "declared capacity with equivalent array rounded upward to whole blocks",
        },
    ))
