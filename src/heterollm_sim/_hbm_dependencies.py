"""Pure HBM organization/timing dependencies shared by all config entrances.

This module deliberately imports no configuration dataclasses. Provenance
identifies derived/reference values; explicit experiments retain their values.
Geometry is insufficient to infer vendor array delays or a valid speed bin.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Iterable, Mapping


SOURCES = {
    "HBM2": "https://gitcode.com/Open-source-documentation-tutorial/a1451",
    "HBM3": "https://gitcode.com/Open-source-documentation-tutorial/69c72",
    "HBM4": "https://github.com/Adilohit/HBM4_verification/blob/main/resources/JEDECJESD270-4A-2025.pdf",
}
DOCUMENTS = {
    "HBM2": ("JESD235B", ".refs/jedec/JESD235B-2018.pdf",
             "JESD235B (November 2018), sections 3.2.1.2/3.2.3/3.2.4, Tables 4/5, printed pp. 6-8",
             "JESD235B pp.30-41,107-113"),
    "HBM3": ("JESD238", ".refs/jedec/JESD238-2022.pdf",
             "JESD238 (January 2022), sections 3.1.2/3.2, Tables 4/5, printed pp. 6-7",
             "JESD238 pp.26,34-46,161-165"),
    "HBM4": ("JESD270-4A", ".refs/jedec/JESD270-4A-2025.pdf",
             "JESD270-4A (2025), sections 3.1.2/3.2, Tables 4/5, printed pp. 6-7",
             "JESD270-4A pp.33,41-55,194-199"),
}
AUTOMATIC = {"standard_organization", "derived_standard_organization", "standard_reference", "standard_minimum",
             "standard_command_semantics", "derived_from_input", "jedec_idd_example", "selected_reference"}
GEOMETRY_FIELDS = {"channels", "subchannels_per_channel", "pseudo_channels_per_channel",
    "ranks_per_channel", "bank_groups_per_rank", "banks_per_group", "rows_per_bank",
    "row_bytes", "burst_bytes", "data_width_bits", "bank_geometry_scope", "stack_interface_mode"}
ARRAY_UNKNOWN = {"activate_to_read_ns", "activate_to_write_ns", "write_to_precharge_ns",
    "activate_same_group_ns", "activate_different_group_ns", "four_activate_window_ns",
    "read_to_precharge_ns", "write_to_read_same_group_ns", "write_to_read_different_group_ns",
    "read_to_write_command_ns"}
TIMING_FIELDS = ARRAY_UNKNOWN | {"command_clock_ns", "activate_command_cycles",
    "precharge_command_cycles", "column_command_cycles", "precharge_to_activate_ns",
    "minimum_active_ns", "maximum_active_ns", "activate_cycle_ns", "precharge_to_precharge_ns",
    "column_same_group_ns", "column_different_group_ns", "column_different_sid_ns",
    "refresh_interval_ns", "refresh_cycle_ns", "read_latency_ns", "write_latency_ns",
    "refresh_policy", "activated_row_msb", "close_ns", "burst_interval_ns"}


def family(generation):
    return "HBM2" if generation in {"HBM2", "HBM2E"} else "HBM3" if generation in {"HBM3", "HBM3E"} else generation


def organization_for(generation: str, density: int, height: int) -> dict:
    """Return the imported JEDEC organization, or raise for unsupported inputs."""
    gen = family(generation)
    densities = (16,) if gen == "HBM2" else (8, 16, 24, 32) if gen == "HBM3" else (24, 32)
    heights = (4, 8) if gen == "HBM2" else (4, 8, 12, 16)
    if gen not in SOURCES or density not in densities or height not in heights:
        raise ValueError("unsupported HBM standard density/height organization")
    return dict(channels=8 if gen == "HBM2" else 16 if gen == "HBM3" else 32,
        subchannels_per_channel=1, pseudo_channels_per_channel=2, ranks_per_channel=1,
        bank_groups_per_rank=height // 2 if gen == "HBM4" else height,
        banks_per_group=8 if gen == "HBM4" else 4,
        rows_per_bank=density * (2048 if gen == "HBM2" else 1024 if gen == "HBM3" else 512),
        row_bytes=1024, burst_bytes=32, data_width_bits=64 if gen == "HBM2" else 32,
        bank_geometry_scope="per_stack", stack_interface_mode="independent")


def refresh_cycle_ns_for(generation: str, density: int, height: int):
    """JEDEC tRFCab table lookup; None means unavailable, never a zero delay."""
    gen = family(generation)
    if gen == "HBM4":
        return {24: {4: 360., 8: 410., 12: 450., 16: 490.},
                32: {4: 400., 8: 450., 12: 490., 16: 530.}}.get(density, {}).get(height)
    if gen == "HBM3":
        return {(8, 8): 260., (8, 12): 310., (8, 16): 350.,
                (16, 4): 260., (16, 8): 350., (16, 12): 410., (16, 16): 450.}.get((density, height))
    if gen == "HBM2" and density == 16:
        return {4: 350., 8: 450.}.get(height)
    return None


def clock_ns_for(generation: str, width: int, lanes: int, interface_gb_s: float) -> float:
    return (2 if family(generation) == "HBM2" else 4) * width * lanes / (8.0 * interface_gb_s)


def reference_timings(generation: str, clock: float) -> dict:
    """Protocol minima and the separately labelled IDD example condition."""
    hbm2 = family(generation) == "HBM2"
    ras, rp = math.ceil(33.0 / clock) * clock, math.ceil(15.0 / clock) * clock
    return dict(activate_command_cycles=2. if hbm2 else 1.5,
        precharge_command_cycles=1. if hbm2 else .5, column_command_cycles=1.,
        column_same_group_ns=max(4 * clock, 2.8 if hbm2 else 2.5),
        column_different_group_ns=2 * clock, column_different_sid_ns=4 * clock,
        precharge_to_precharge_ns=2 * clock, minimum_active_ns=ras,
        precharge_to_activate_ns=rp, close_ns=rp, activate_cycle_ns=ras + rp)


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def normalize_hbm_dependencies(payload: Mapping, *, previous: Mapping | None = None,
                               explicit_fields: Iterable[str] = ()) -> dict:
    """Normalize managed HBM values; preserve explicitly authored experiments.

    Bandwidth mode uses the aggregate interface as its primary operating
    point. Data-rate mode uses per-pin rate. A clock-only reconfiguration
    changes that operating point. Raw derived fields follow their declared
    provenance; use explicit assumption provenance to author a different value.
    """
    result = deepcopy(dict(payload))
    if result.get("control_mode") != "jedec_hbm":
        return result
    generation = str(result.get("generation", "")).upper()
    gen = family(generation)
    if gen not in SOURCES:
        return result
    explicit = set(explicit_fields)
    prior = dict(previous or {})
    if not isinstance(result.get("metadata", {}), Mapping):
        raise ValueError("HBM metadata must be a mapping")
    meta = deepcopy(dict(result.get("metadata") or {}))
    prior_meta = dict(prior.get("metadata") or {})
    if not isinstance(meta.get("parameter_provenance", {}), Mapping):
        raise ValueError("HBM parameter_provenance must be a mapping")
    provenance = deepcopy(dict(meta.get("parameter_provenance") or {}))
    if any(not isinstance(entry, Mapping) for entry in provenance.values()):
        raise ValueError("HBM parameter provenance entries must be mappings")
    managed = meta.get("geometry_status") in {"standard_organization", "explicit_assumption"} and (
        bool(provenance) or meta.get("organization_source") or meta.get("jedec_standard"))
    source = SOURCES[gen]
    unknown = set(meta.get("unknown_parameters") or ())
    meta["vendor_verified"] = False

    def record(name, status, basis, **extra):
        provenance[name] = dict(status=status, source=source if status in AUTOMATIC else "physical_config_override",
                                basis=basis, **extra)

    def automatic(name):
        return name not in explicit and (provenance.get(name, {}).get("status") in AUTOMATIC
                                        or (managed and name not in provenance))

    for name in explicit & TIMING_FIELDS:
        record(name, "explicit_assumption", "explicit timing/control input; not a certified vendor value")
    if not managed:
        # Synthetic configurations remain executable with their authored
        # numbers, while missing manufacturer evidence remains visible.
        for name in TIMING_FIELDS & result.keys():
            if name not in provenance:
                record(name, "explicit_assumption" if result[name] else "unknown_disabled",
                       "authored synthetic control configuration; vendor timing is unknown")
        for name in ARRAY_UNKNOWN:
            unknown.add(name)
            if result.get(name):
                record(name, "explicit_assumption" if name in explicit else "unknown_with_assumption",
                       "device minimum unknown; explicit analytical value is executed", vendor_value_status="unknown")
        if "activate_cycle_ns" not in explicit and explicit & {"minimum_active_ns", "precharge_to_activate_ns"}:
            result["activate_cycle_ns"] = result["minimum_active_ns"] + result["precharge_to_activate_ns"]
        if "maximum_active_ns" not in explicit and "refresh_interval_ns" in explicit and _positive(result.get("refresh_interval_ns")):
            result["maximum_active_ns"] = 9 * result["refresh_interval_ns"]
        meta.update(parameter_provenance=provenance, unknown_parameters=sorted(unknown))
        result["metadata"] = meta
        return result

    for name in ("channels", "subchannels_per_channel", "pseudo_channels_per_channel",
                 "stacks", "dies_per_stack", "ranks_per_channel", "bank_groups_per_rank",
                 "banks_per_group", "rows_per_bank", "row_bytes", "burst_bytes", "data_width_bits"):
        if isinstance(result.get(name), bool) or not isinstance(result.get(name), int) or result[name] <= 0:
            raise ValueError(name + " must be a positive integer")

    old_gen = family(str(prior.get("generation", generation)))
    if not prior:
        old_gen = {"JESD235B": "HBM2", "JESD238": "HBM3", "JESD270-4A": "HBM4"}.get(meta.get("jedec_standard"), gen)
    old_density = prior_meta.get("die_density_gbits", meta.get("die_density_gbits"))
    old_height = prior.get("dies_per_stack", int(meta.get("sid_count", 0)) * 4 or result.get("dies_per_stack"))
    if not prior and _positive(meta.get("raw_stack_capacity_bytes")) and _positive(old_height):
        capacity_density = meta["raw_stack_capacity_bytes"] * 8 / (old_height * (1 << 30))
        if float(capacity_density).is_integer():
            old_density = int(capacity_density)
    try:
        old_org = organization_for(old_gen, old_density, old_height)
    except (TypeError, ValueError):
        old_org = {}
    density = meta.get("die_density_gbits")
    height = result.get("dies_per_stack")
    rows = result.get("rows_per_bank")
    rows_authored = "rows_per_bank" in explicit or density is None or (not prior and old_org and rows != old_org["rows_per_bank"])
    density_authored = (("metadata" in explicit and density != prior_meta.get("die_density_gbits"))
                        or (not prior and density is not None and density != old_density))
    factor = 2048 if gen == "HBM2" else 1024 if gen == "HBM3" else 512
    if rows_authored:
        inferred = rows // factor if isinstance(rows, int) and rows > 0 and rows % factor == 0 else None
        if density_authored and inferred != density:
            raise ValueError("explicit HBM die density and row organization conflict")
        density = inferred
    try:
        expected = organization_for(generation, density, height)
    except (TypeError, ValueError):
        expected = None
    if expected:
        for name, value in expected.items():
            authored = name in explicit or (not prior and name in old_org and result.get(name) != old_org[name])
            if name == "rows_per_bank":
                authored = rows_authored
            if not authored and (automatic(name) or name in {"bank_geometry_scope", "stack_interface_mode"}):
                result[name] = value
        standard = all(result.get(name) == value for name, value in expected.items())
        standard = standard and result.get("data_lanes") in (None, expected["channels"] * 2)
    else:
        standard = False
    if standard:
        standard_name, source_file, organization_basis, control_basis = DOCUMENTS[gen]
        limitations = ["linear mixed-radix simulation mapping is not a vendor address mapping",
                      "standard organization does not establish the target vendor part or timing"]
        if gen == "HBM2":
            limitations.append("catalog imports only 16Gb base/8H HBM2 organizations; other legal JEDEC organizations are not yet imported")
        if height == 4 and gen != "HBM4":
            limitations.append(f"4H instantiates the no-SID base organization; {standard_name} Table 4 leaves base-organization physical stack height vendor-specific")
        meta.update(geometry_status="standard_organization", die_density_gbits=density,
            sid_count=height // 4, physical_channels_per_stack=expected["channels"],
            physical_channel_width_bits=expected["data_width_bits"] * 2,
            pseudo_channel_width_bits=expected["data_width_bits"],
            stack_data_width_bits=expected["channels"] * expected["data_width_bits"] * 2,
            banks_per_pseudo_channel=expected["bank_groups_per_rank"] * expected["banks_per_group"],
            bank_groups_per_pseudo_channel=expected["bank_groups_per_rank"],
            effective_addressed_page_bytes=1024, organization_source=source,
            jedec_standard=standard_name, jedec_reference_file=source_file,
            organization_basis=organization_basis, control_basis=control_basis,
            geometry_basis=organization_basis, geometry_limitations=limitations,
            channel_operation_mode="pseudo_channel",
            channel_operation_mode_basis=(f"{standard_name} section 3.1.2 defines two pseudo-channels per channel; no alternative Legacy mode is defined for this generation"
                if gen != "HBM2" else "JESD235B section 3.2.1 defines Legacy (HBM1) and Pseudo Channel (HBM2); this catalog imports only PC-mode organizations. Mode support is fixed by device design: DEVICE_ID[17:16] is 01 for PC-only or 10 for Legacy-only; 00 and 11 are illegal. This is not a runtime Legacy/PC switch."),
            capacity_formula="stacks * die_density_gbits * dies_per_stack * 2**30 / 8",
            source_documents=[dict(source=source, basis=organization_basis, local_file=source_file,
                                   document_author="JEDEC", access_method="public mirror of original protocol PDF")],
            parameter_basis="sourced_organization_with_explicit_timing_inputs")
        meta["row_bytes_basis"] = (f"{standard_name} Table 4 note 3: row_bytes is the 1KiB effective addressed PC page; the MSB of RA selects half of an open 2KiB page"
                                  if gen != "HBM2" else "JESD235B Table 4: effective PC page is 1KiB; physical activation organization is not asserted")
        if gen != "HBM2":
            meta["activated_page_bytes"] = 2048
        else:
            meta.pop("activated_page_bytes", None)
            provenance.pop("activated_page_bytes", None)
        for name in GEOMETRY_FIELDS:
            record(name, "standard_organization", "JEDEC channel addressing table for selected density/height")
        for name in ("effective_addressed_page_bytes", "activated_page_bytes"):
            if name in meta:
                record(name, "standard_organization", meta["row_bytes_basis"])
        for name, entry in provenance.items():
            if entry.get("status") in AUTOMATIC:
                entry["source"] = source
    else:
        meta.update(geometry_status="explicit_assumption", control_status="jedec_command_model_on_custom_geometry",
                    parameter_basis="explicit_organization_with_analytical_timing")
        for name in ("die_density_gbits", "sid_count", "organization_source", "organization_basis",
                     "organization_selection", "jedec_standard", "jedec_reference_file", "geometry_standard",
                     "geometry_source", "geometry_verified", "source_provenance", "capacity_formula",
                     "physical_channels_per_stack", "physical_channel_width_bits", "pseudo_channel_width_bits",
                     "stack_data_width_bits", "banks_per_pseudo_channel", "bank_groups_per_pseudo_channel"):
            meta.pop(name, None)
        for name in GEOMETRY_FIELDS:
            if name in explicit or not expected or result.get(name) != expected[name]:
                record(name, "explicit_assumption", "custom organization; no standard device organization is asserted")
        unknown.add("die_density_gbits")
        # The executed command protocol still follows this generation, while
        # its source table no longer certifies the custom organization.
        _, source_file, _, control_basis = DOCUMENTS[gen]
        meta["control_basis"] = control_basis
        meta["source_documents"] = [dict(source=source, basis=control_basis,
            local_file=source_file, document_author="JEDEC",
            access_method="public mirror of original protocol PDF")]
        if gen != old_gen:
            for name in ("row_bytes_basis", "geometry_limitations", "channel_operation_mode_basis",
                         "effective_addressed_page_bytes", "activated_page_bytes"):
                meta.pop(name, None)
                provenance.pop(name, None)

    lanes = (result.get("data_lanes") or result["channels"] * result["subchannels_per_channel"] * result["pseudo_channels_per_channel"])
    if result.get("stack_interface_mode") == "independent":
        lanes *= result["stacks"]
    width = result["data_width_bits"]
    prior_clock = prior.get("command_clock_ns", result.get("command_clock_ns"))
    old_interface = prior.get("interface_bandwidth_gb_s", result.get("interface_bandwidth_gb_s"))
    frequency_fields = {"interface_bandwidth_gb_s", "lane_bandwidth_gb_s", "effective_pin_data_rate_gbps", "data_rate_mt_s"}
    clock_authored = "command_clock_ns" in explicit
    mode = result.get("bandwidth_input_mode", "bandwidth")
    for name in explicit & (frequency_fields | {"command_clock_ns"}):
        if result.get(name) is not None and not _positive(result[name]):
            raise ValueError(name + " must be finite and positive")
    if clock_authored and not _positive(result.get("command_clock_ns")):
        raise ValueError("command_clock_ns must be finite and positive")
    if "bandwidth_input_mode" not in explicit:
        if explicit & {"effective_pin_data_rate_gbps", "data_rate_mt_s"}:
            mode = "data_rate"
        elif explicit & {"interface_bandwidth_gb_s", "lane_bandwidth_gb_s"}:
            mode = "bandwidth"
    result["bandwidth_input_mode"] = mode
    declared_interfaces = []
    for name in explicit & frequency_fields:
        value = result.get(name)
        if value is None:
            continue
        supplied = value * lanes if name == "lane_bandwidth_gb_s" else value * width * lanes / (8.0 * (1000 if name == "data_rate_mt_s" else 1)) if name in {"data_rate_mt_s", "effective_pin_data_rate_gbps"} else value
        declared_interfaces.append(supplied)
    if any(not math.isclose(value, declared_interfaces[0], rel_tol=1e-10) for value in declared_interfaces[1:]):
        raise ValueError("explicit HBM interface and data-rate inputs conflict")
    if clock_authored and _positive(result.get("command_clock_ns")):
        interface = (2 if gen == "HBM2" else 4) * width * lanes / (8.0 * result["command_clock_ns"])
        for name in explicit & frequency_fields:
            value = result.get(name)
            if value is None:
                continue
            supplied = value * lanes if name == "lane_bandwidth_gb_s" else value * width * lanes / (8.0 * (1000 if name == "data_rate_mt_s" else 1)) if name in {"data_rate_mt_s", "effective_pin_data_rate_gbps"} else value
            if not math.isclose(interface, supplied, rel_tol=1e-10):
                raise ValueError("explicit HBM command clock and interface/data-rate inputs conflict")
        result["interface_bandwidth_gb_s"] = interface
        result["lane_bandwidth_gb_s"] = None
        operating_basis = "explicit command_clock_ns; interface and per-pin rate follow JEDEC CK/DQ ratio"
    elif mode == "data_rate":
        rate = result["data_rate_mt_s"] / 1000.0 if "data_rate_mt_s" in explicit and "effective_pin_data_rate_gbps" not in explicit else result.get("effective_pin_data_rate_gbps")
        rate = rate if rate is not None else result["data_rate_mt_s"] / 1000.0
        interface = rate * width * lanes / 8.0
        if "interface_bandwidth_gb_s" in explicit and result.get("interface_bandwidth_gb_s") is not None and not math.isclose(interface, result["interface_bandwidth_gb_s"], rel_tol=1e-10):
            raise ValueError("explicit HBM data-rate and interface bandwidth inputs conflict")
        result["interface_bandwidth_gb_s"] = interface
        result["lane_bandwidth_gb_s"] = None
        operating_basis = "data_rate mode: per-pin rate is primary; aggregate interface and CK are derived"
    else:
        if "lane_bandwidth_gb_s" in explicit and "interface_bandwidth_gb_s" not in explicit:
            result["interface_bandwidth_gb_s"] = None
        if "interface_bandwidth_gb_s" in explicit and "lane_bandwidth_gb_s" not in explicit:
            result["lane_bandwidth_gb_s"] = None
        interface = result.get("interface_bandwidth_gb_s")
        if interface is None:
            interface = result.get("lane_bandwidth_gb_s")
            interface = interface * lanes if interface is not None else result["data_rate_mt_s"] * width * lanes / 8000.0
        operating_basis = "bandwidth mode: aggregate interface bandwidth is primary; per-PC rate and CK are derived"
    if not _positive(interface) or not _positive(lanes):
        raise ValueError("HBM physical interface bandwidth and lane count must be positive")
    clock = clock_ns_for(generation, width, lanes, interface)
    result["command_clock_ns"] = clock
    result["data_rate_mt_s"] = interface * 8000.0 / (width * lanes)
    if mode == "data_rate" or result.get("effective_pin_data_rate_gbps") is not None:
        result["effective_pin_data_rate_gbps"] = result["data_rate_mt_s"] / 1000.0
    if previous and old_interface is not None and not math.isclose(interface, old_interface, rel_tol=1e-10):
        for name in ("read_bandwidth_gb_s", "write_bandwidth_gb_s"):
            if name not in explicit and result.get(name) is not None:
                if math.isclose(result[name], old_interface, rel_tol=1e-10):
                    result[name] = interface
                else:
                    result[name] = min(result[name], interface)
    for name in ("command_clock_ns", "data_rate_mt_s"):
        record(name, "derived_from_input", operating_basis)
    meta["operating_point_basis"] = operating_basis
    result["burst_interval_ns"] = result["burst_bytes"] * lanes / interface
    record("burst_interval_ns", "derived_from_input", "full DQ burst duration from physical interface; direction caps only pace starts")
    if automatic("activated_row_msb"):
        result["activated_row_msb"] = None if gen == "HBM2" else (result["rows_per_bank"] - 1).bit_length() - 1
        record("activated_row_msb", "standard_command_semantics", "effective RA highest bit chooses the activated page half")

    for name, value in reference_timings(generation, clock).items():
        if automatic(name):
            entry = provenance.get(name, {})
            floor = entry.get("analytical_floor_ns")
            floor_rule = entry.get("rule") == "max_standard_minimum_and_analytical_floor"
            if floor_rule:
                if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor < 0:
                    raise ValueError("analytical_floor_ns must be finite and non-negative")
                value = max(value, floor)
            result[name] = value
            status = "jedec_idd_example" if name in {"minimum_active_ns", "precharge_to_activate_ns", "close_ns", "activate_cycle_ns"} else "selected_reference" if name == "column_different_sid_ns" else "standard_command_semantics" if name.endswith("command_cycles") else "standard_minimum"
            record(name, "derived_from_input" if floor_rule else status,
                   "maximum of the current standard minimum and an explicit analytical floor" if floor_rule else
                   "IDD example 33ns tRAS/15ns tRP quantized in CK" if status == "jedec_idd_example" else "JEDEC command clocks; tCCDR selects the 4CK reference within vendor range",
                   **({"rule": "max_standard_minimum_and_analytical_floor", "analytical_floor_ns": floor} if floor_rule else {}))
    for name, default in (("read_latency_ns", 32), ("write_latency_ns", 8)):
        entry = provenance.get(name, {})
        cycles = entry.get("cycles")
        if automatic(name) or (name not in explicit and entry.get("input_unit") == "cycles"):
            if cycles is None:
                cycles = result.get(name, 0) / prior_clock if _positive(prior_clock) else default
            result[name] = cycles * clock
            record(name, "caller_supplied" if entry.get("input_unit") == "cycles" and entry.get("status") == "caller_supplied" else "selected_reference",
                   "selected whole-CK MR latency; vendor speed-bin support unknown", cycles=cycles, input_unit="cycles")
        else:
            record(name, "explicit_assumption", "caller-authored ns latency; legal MR encoding validated separately",
                   cycles=result.get(name, 0) / clock, input_unit="ns")
        unknown.add(name)
    if automatic("activate_cycle_ns"):
        if "activate_cycle_ns" not in explicit:
            result["activate_cycle_ns"] = result["minimum_active_ns"] + result["precharge_to_activate_ns"]
            record("activate_cycle_ns", "derived_from_input", "tRC = configured tRAS + tRP, preserving their clock quantization")
    refresh = refresh_cycle_ns_for(generation, density, height) if standard else None
    if automatic("refresh_cycle_ns") or provenance.get("refresh_cycle_ns", {}).get("status") in {"unknown", "unknown_disabled"}:
        result["refresh_cycle_ns"] = refresh or 0.0
        record("refresh_cycle_ns", "standard_reference" if refresh is not None else "unknown",
               "JEDEC tRFCab table selected by generation, density and height" if refresh is not None else "no applicable known tRFCab entry; refresh duration is unknown")
    elif refresh is None:
        record("refresh_cycle_ns", "explicit_assumption", "no known device tRFCab; explicit analytical duration retained",
               vendor_value_status="unknown")
    if refresh is None:
        unknown.add("refresh_cycle_ns")
    else:
        unknown.discard("refresh_cycle_ns")
    if automatic("refresh_interval_ns") or provenance.get("refresh_interval_ns", {}).get("status") in {"unknown", "unknown_disabled"}:
        interval = result.get("refresh_interval_ns")
        if _positive(interval) and interval != 3900.0:
            record("refresh_interval_ns", "explicit_assumption", "authored refresh/temperature operating interval; vendor conditions unknown")
        else:
            result["refresh_interval_ns"] = 3900.0 if result.get("refresh_cycle_ns", 0) > 0 else 0.0
            record("refresh_interval_ns", "standard_reference" if result["refresh_interval_ns"] else "unknown",
                   "selected normal-temperature 3900ns tREFI; actual temperature/vendor conditions unknown")
    if "refresh_policy" not in explicit and provenance.get("refresh_policy", {}).get("status") in AUTOMATIC | {"controller_policy", "unknown_disabled"}:
        result["refresh_policy"] = "all_bank" if result.get("refresh_cycle_ns", 0) > 0 and result.get("refresh_interval_ns", 0) > 0 else "disabled"
    if automatic("maximum_active_ns"):
        result["maximum_active_ns"] = 9 * (result.get("refresh_interval_ns") or 3900.0)
        record("maximum_active_ns", "derived_from_input", "tRASmax = 9*tREFI; 3900ns reference remains when automatic refresh is unavailable")
    for name in ARRAY_UNKNOWN:
        unknown.add(name)
        status = provenance.get(name, {}).get("status")
        record(name, "explicit_assumption" if status == "explicit_assumption" else "unknown_with_assumption" if result.get(name) else "unknown_disabled",
               "device timing cannot be inferred from geometry; explicit analytical value retained" if result.get(name) else "device timing not supplied; zero disables this gate",
               vendor_value_status="unknown")
    unknown.update({"minimum_active_ns", "precharge_to_activate_ns", "activate_cycle_ns", "column_different_sid_ns"})
    if standard:
        raw_stack = density * height * (1 << 30) // 8
        raw = raw_stack * result["stacks"]
        meta.update(raw_stack_capacity_bytes=raw_stack, raw_capacity_bytes=raw,
                    visible_capacity_bytes=result.get("capacity_bytes"),
                    unexposed_capacity_bytes=raw - (result.get("capacity_bytes") or raw),
                    control_status="jedec_normal_operation_subset")
    meta.update(parameter_provenance=provenance, unknown_parameters=sorted(unknown),
                unconfigured_timing_parameters=sorted(name for name in ARRAY_UNKNOWN | {"refresh_cycle_ns"} if not result.get(name)),
                timing_validation="not_a_vendor_speed_bin")
    result["metadata"] = meta
    return result
