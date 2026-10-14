"""Required physical contracts at simulation and memory-service boundaries.

Incomplete components may exist in the editor. They must never execute by
substituting a bandwidth-only service or the memory core's example defaults.
"""

from dataclasses import asdict, is_dataclass
from typing import Mapping

from .memory_types import (
    DRAM_BANK_GROUP_TIMING_FIELDS,
    DramConfig, NandConfig, Operation, parse_physical_memory_config,
)


DRAM_COMPONENT_KINDS = frozenset({
    "hbm", "hbm_stack", "gddr", "gddr_memory", "dram", "ddr", "ddr_memory",
    "lpddr", "lpddr_memory", "cxl_memory", "host_memory", "memory",
})
NAND_COMPONENT_KINDS = frozenset({"hbf", "ssd", "high_io_ssd", "nvme"})
PHYSICAL_MEMORY_COMPONENT_KINDS = DRAM_COMPONENT_KINDS | NAND_COMPONENT_KINDS

# This additive group/rank timing extension is intentionally optional for old
# serialized configurations. Each field is validated by DramConfig when
# present; omission means zero/disabled, never a family-specific timing guess.
DRAM_OPTIONAL_BANK_GROUP_TIMING_FIELDS = frozenset(DRAM_BANK_GROUP_TIMING_FIELDS)

_DRAM_REQUIRED = frozenset({
    "kind", "channels", "subchannels_per_channel", "pseudo_channels_per_channel",
    "stacks", "dies_per_stack", "ranks_per_channel", "bank_groups_per_rank",
    "banks_per_group", "rows_per_bank", "row_bytes", "burst_bytes",
    "open_ns", "close_ns", "read_latency_ns", "write_latency_ns",
    "burst_interval_ns", "read_recovery_ns", "write_recovery_ns",
    "read_to_write_ns", "write_to_read_ns", "max_outstanding_requests",
})
_NAND_REQUIRED = frozenset({
    "kind", "channels", "targets_per_channel", "dies_per_target", "luns_per_die",
    "planes_per_lun", "blocks_per_plane", "pages_per_block", "page_bytes",
    "host_granularity_bytes", "host_bandwidth_gb_s", "internal_bandwidth_gb_s",
    "page_read_ns", "page_program_ns", "block_erase_ns", "front_ns",
    "planes_independent", "partial_page_policy", "max_outstanding_requests",
})


def parse_required_physical_config(component_id, kind, raw):
    """Validate authored fields before dataclass defaults can fill omissions."""
    prefix = "component {} ({}): ".format(component_id, kind)
    if kind not in PHYSICAL_MEMORY_COMPONENT_KINDS:
        raise ValueError(prefix + "no supported physical memory core; select a DRAM or NAND component")
    if is_dataclass(raw):
        raw = asdict(raw)
    if not isinstance(raw, Mapping):
        raise ValueError(prefix + "physical_memory_config is required; reload a complete hardware preset or configure the physical memory core. Aggregate-cost fallback is disabled")
    required = _NAND_REQUIRED if kind in NAND_COMPONENT_KINDS else _DRAM_REQUIRED
    missing = sorted(name for name in required if raw.get(name) is None)
    if kind in DRAM_COMPONENT_KINDS and not (
        raw.get("interface_bandwidth_gb_s") is not None
        or raw.get("lane_bandwidth_gb_s") is not None
        or (raw.get("data_width_bits") is not None and (
            raw.get("data_rate_mt_s") is not None or raw.get("effective_pin_data_rate_gbps") is not None
        ))
    ):
        missing.append("interface bandwidth or data width and data rate")
    if missing:
        raise ValueError(prefix + "physical_memory_config missing explicit fields: " + ", ".join(missing))
    try:
        parsed = parse_physical_memory_config(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(prefix + "invalid physical_memory_config: " + str(exc)) from exc
    allowed = (
        {"HBF"} if kind == "hbf" else {"SSD"} if kind in NAND_COMPONENT_KINDS
        else {"HBM"} if kind in {"hbm", "hbm_stack"}
        else {"GDDR"} if kind in {"gddr", "gddr_memory"}
        else {"DDR"} if kind in {"ddr", "ddr_memory"}
        else {"LPDDR"} if kind in {"lpddr", "lpddr_memory"}
        else {"DDR", "LPDDR", "HBM", "GDDR"}
    )
    if parsed.kind.value not in allowed:
        raise ValueError(prefix + "physical_memory_config.kind must be " + "/".join(sorted(allowed)))
    return parsed


def require_physical_memory_config(component):
    """Return a validated core config, or identify the incomplete component."""
    prefix = "component {} ({}): ".format(component.component_id, component.kind)
    legacy_mode = component.metadata.get("memory_execution_mode")
    if legacy_mode is not None and str(legacy_mode).strip().lower() != "physical":
        raise ValueError(prefix + "memory_execution_mode is no longer configurable; remove the obsolete setting. All memory execution requires its physical DRAM/NAND core")
    parsed = parse_required_physical_config(
        component.component_id, component.normalized_kind,
        component.metadata.get("physical_memory_config"),
    )
    capacity = component.capacity_bytes
    if capacity <= 0 or parsed.capacity_bytes != capacity:
        raise ValueError(prefix + "capacity_bytes must be positive and equal physical_memory_config capacity (component={}, physical={})".format(capacity, parsed.capacity_bytes))
    physical_memory_bandwidth_caps_gbps(component, physical_config=parsed)
    return parsed


def physical_memory_bandwidth_caps_gbps(component, *, physical_config=None):
    """Return the core's shared/read/write ceilings in Gb/s.

    A prevalidated typed config may be supplied by a caller that already used
    ``require_physical_memory_config``. Component bandwidth declarations are
    validated ceilings for effective profiles; they do not retime the core or
    replace its physical interface values in reports.
    """
    parsed = (require_physical_memory_config(component)
              if physical_config is None else physical_config)
    if isinstance(parsed, DramConfig):
        caps = (parsed.physical_interface_bandwidth_gb_s,
                parsed.directional_bandwidth_gb_s(Operation.READ),
                parsed.directional_bandwidth_gb_s(Operation.WRITE))
    elif isinstance(parsed, NandConfig):
        caps = (parsed.host_bandwidth_gb_s,
                parsed.host_bandwidth_for(Operation.READ),
                parsed.host_bandwidth_for(Operation.WRITE))
    else:
        raise TypeError("physical_config must be a validated DramConfig or NandConfig")
    caps = tuple(float(value) * 8.0 for value in caps)
    for label, declared, physical in zip(
        ("shared", "read", "write"),
        (component.shared_bandwidth_gbps, component.read_bandwidth_gbps,
         component.write_bandwidth_gbps), caps,
    ):
        if declared > physical * (1.0 + 1e-12):
            raise ValueError(
                "component {} ({}): component {} bandwidth exceeds physical memory {} bandwidth".format(
                    component.component_id, component.kind, label,
                    "interface" if label == "shared" else label))
    return caps


def parse_execution_physical_config(raw):
    """Accept a typed core config or require complete serialized execution input.

    Convenience constructors may supply analytical defaults before this
    boundary. A serialized execution request must carry those choices rather
    than silently acquiring a different geometry or timing configuration.
    """
    if isinstance(raw, (DramConfig, NandConfig)):
        return raw
    if not isinstance(raw, Mapping) or raw.get("kind") is None:
        raise ValueError("physical_memory_config missing explicit fields: kind")
    value = raw["kind"]
    kind = str(getattr(value, "value", value)).strip().upper().replace("-", "")
    component_kind = "gddr" if kind in {"GDDR6", "GDDR6X", "GDDR7"} else kind.lower()
    return parse_required_physical_config("execution", component_kind, raw)
