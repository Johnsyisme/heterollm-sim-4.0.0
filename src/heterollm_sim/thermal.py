"""Opt-in static operating points and explicitly assumed dynamic NAND RC state.

Factors are supplied by the caller (measurement or labeled sensitivity input).
No calibration is inferred. Dynamic NAND parameters are authored separately.
Always apply to a baseline scenario: capacities, placement and ownership stay
fixed while cloned profiles and link/endpoint rates change together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from numbers import Real

from .config import ScenarioConfig
from .cost_models import (
    CPUProfile, DigitalSramCimProfile, GPUProfile, HBMProfile, HostMemoryProfile,
)
from .memory_types import DramConfig, parse_physical_memory_config


def advance_nand_thermal(state, config, start_ns: float, work_ns: float):
    """Integrate one serial physical-plane operation, including threshold events.

    ``work_ns`` is unthrottled media work; thermal service-rate changes alter
    its actual completion time. Idle gaps cool the same persistent plane.
    Different planes are explicitly uncoupled; this is an RC sensitivity model.
    """
    previous = state or {"end_ns": 0., "temperature_c": config.initial_c, "throttled": False}
    if start_ns < previous["end_ns"]-1e-6:
        raise ValueError("dynamic thermal operations must be ordered within their physical plane")
    tau_ns = config.resistance_k_per_w*config.capacitance_j_per_k*1e9
    def evolve(temperature, power, duration):
        equilibrium = config.ambient_c + power*config.resistance_k_per_w
        return equilibrium + (temperature-equilibrium)*math.exp(-duration/tau_ns)
    temperature = evolve(previous["temperature_c"], config.idle_power_w,
                         max(0., start_ns-previous["end_ns"]))
    throttled = previous["throttled"]
    release = config.throttle_threshold_c-config.hysteresis_c
    idle_recovery = int(throttled and temperature <= release)
    if idle_recovery:
        throttled = False
    if temperature >= config.throttle_threshold_c:
        throttled = True
    initial = temperature
    # This event describes the current array-operation window. A previous
    # hot operation can have cooled during the unobserved idle gap; its peak
    # remains in that earlier event, not in this operation's temperature range.
    peak = temperature
    remaining, elapsed, throttle_ns, switches = float(work_ns), 0., 0., 0
    while remaining > 1e-12:
        rate = config.throttled_service_rate if throttled else 1.
        power = config.active_power_w*(config.throttled_power_fraction if throttled else 1.)
        equilibrium = config.ambient_c+power*config.resistance_k_per_w
        finish_time = remaining/rate
        target = release if throttled else config.throttle_threshold_c
        crossing = math.inf
        if (throttled and equilibrium < target < temperature) or (not throttled and temperature < target < equilibrium):
            crossing = -tau_ns*math.log((target-equilibrium)/(temperature-equilibrium))
        duration = min(finish_time, crossing)
        temperature = evolve(temperature, power, duration)
        peak = max(peak, temperature)
        elapsed += duration
        throttle_ns += duration if throttled else 0.
        remaining = max(0., remaining-duration*rate)
        if crossing < finish_time:
            throttled = not throttled
            switches += 1
        if switches > 100000:
            raise ValueError("thermal RC assumptions cause over 100000 threshold crossings in one page operation")
    return {"end_ns": start_ns+elapsed, "temperature_c": temperature, "throttled": throttled}, {
        "duration_ns": elapsed, "temperature_start_c": initial, "temperature_end_c": temperature,
        "peak_temperature_c": peak, "thermal_throttle_ns": throttle_ns,
        "thermal_extra_ns": max(0., elapsed-work_ns), "thermal_switches": switches,
        "idle_recoveries": idle_recovery,
    }


@dataclass(frozen=True)
class ThermalOperatingPoint:
    domain_id: str
    enabled: bool = False
    frequency_scale: float = 1.0
    memory_bandwidth_scale: float = 1.0
    link_bandwidth_scale: float = 1.0
    latency_scale: float = 1.0
    evidence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.domain_id, str) or not self.domain_id.strip():
            raise ValueError("thermal domain_id must be non-empty text")
        if not isinstance(self.enabled, bool):
            raise ValueError("thermal enabled must be boolean")
        for name in ("frequency_scale", "memory_bandwidth_scale", "link_bandwidth_scale", "latency_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
            if (name == "latency_scale" and value < 1) or (name != "latency_scale" and value > 1):
                raise ValueError(name + " must derate, not accelerate, the baseline")
        if not isinstance(self.evidence, str) or (self.enabled and not self.evidence.strip()):
            raise ValueError("enabled thermal point requires explicit evidence or a labeled analytical assumption")


def _invalidate_kernel_samples(profile, reason):
    model = profile.kernel_model
    if model is None or not any(kernel.samples for kernel in model.kernels):
        return profile
    return replace(profile, kernel_model=replace(model, kernels=tuple(
        replace(kernel, samples=(), calibration_source_bound=False,
                calibration_cells=(), validation_relative_error=None,
                evidence=kernel.evidence + "; " + reason)
        if kernel.samples else kernel for kernel in model.kernels)))


def _derate_profile(profile, point: ThermalOperatingPoint):
    """Clone known typed profiles, retaining capacities, precision and resource IDs."""

    def cache(hierarchy):
        return replace(hierarchy, levels=tuple(
            replace(level, bandwidth_gb_s=level.bandwidth_gb_s * point.memory_bandwidth_scale,
                    hit_latency_ns=level.hit_latency_ns * point.latency_scale)
            for level in hierarchy.levels
        ))

    if isinstance(profile, GPUProfile):
        if (point.frequency_scale != 1.0 or point.memory_bandwidth_scale != 1.0
                or point.latency_scale != 1.0):
            profile = _invalidate_kernel_samples(profile,
                "thermal operating point changed: measured surface invalidated")
        return replace(profile,
                       tensor_core=replace(profile.tensor_core, frequency_ghz=profile.tensor_core.frequency_ghz * point.frequency_scale),
                       cache_hierarchy=cache(profile.cache_hierarchy))
    if isinstance(profile, CPUProfile):
        return replace(profile,
                       pipeline=replace(profile.pipeline, frequency_ghz=profile.pipeline.frequency_ghz * point.frequency_scale,
                                        branch_mispredict_ns=profile.pipeline.branch_mispredict_ns * point.latency_scale),
                       cache_hierarchy=cache(profile.cache_hierarchy))
    if isinstance(profile, (HBMProfile, HostMemoryProfile)):
        changes = {"bandwidth_gb_s": profile.bandwidth_gb_s * point.memory_bandwidth_scale}
        # Reference registries may predate the additive read/write timing fields.
        for field in fields(profile):
            if field.name in {"read_latency_ns", "write_latency_ns"}:
                changes[field.name] = getattr(profile, field.name) * point.latency_scale
            elif field.name in {"read_bandwidth_gb_s", "write_bandwidth_gb_s", "measured_effective_bandwidth_gb_s"}:
                value = getattr(profile, field.name)
                if value is not None:
                    changes[field.name] = value * point.memory_bandwidth_scale
        return replace(profile, **changes)
    if isinstance(profile, DigitalSramCimProfile):
        return replace(profile,
                       frequency_ghz=profile.frequency_ghz * point.frequency_scale,
                       load_bandwidth_gb_s=profile.load_bandwidth_gb_s * point.memory_bandwidth_scale,
                       activation_bandwidth_gb_s=profile.activation_bandwidth_gb_s * point.memory_bandwidth_scale,
                       output_bandwidth_gb_s=profile.output_bandwidth_gb_s * point.memory_bandwidth_scale,
                       noc_bandwidth_gb_s=profile.noc_bandwidth_gb_s * point.link_bandwidth_scale,
                       load_latency_ns=profile.load_latency_ns * point.latency_scale,
                       noc_hop_latency_ns=profile.noc_hop_latency_ns * point.latency_scale,
                       peripheral_latency_ns=profile.peripheral_latency_ns * point.latency_scale)
    raise TypeError("unsupported thermal cost profile: " + type(profile).__name__)


def _derate_physical_memory_config(config, point: ThermalOperatingPoint):
    """Scale the Core's explicit rates/timings without changing its geometry.

    Bandwidth and latency factors are independent authored sensitivities.
    All declared equivalent DRAM rate representations must change together,
    including GDDR pin rate, or the resulting physical contract is inconsistent.
    Dynamic NAND RC/wear parameters are retained: if enabled, their feedback
    acts on this statically derated baseline rather than being recomputed here.
    """
    config = parse_physical_memory_config(config)
    if isinstance(config, DramConfig):
        rates = ("data_rate_mt_s", "effective_pin_data_rate_gbps",
                 "lane_bandwidth_gb_s", "interface_bandwidth_gb_s",
                 "read_bandwidth_gb_s", "write_bandwidth_gb_s")
        timings = ("open_ns", "close_ns", "read_latency_ns", "write_latency_ns",
                   "burst_interval_ns", "read_recovery_ns", "write_recovery_ns",
                   "read_to_write_ns", "write_to_read_ns")
    else:
        rates = ("host_bandwidth_gb_s", "host_read_bandwidth_gb_s",
                 "host_write_bandwidth_gb_s", "internal_bandwidth_gb_s")
        timings = ("page_read_ns", "page_program_ns", "block_erase_ns", "front_ns")
    changes = {name: getattr(config, name) * point.memory_bandwidth_scale
               for name in rates if getattr(config, name) is not None}
    changes.update({name: getattr(config, name) * point.latency_scale for name in timings})
    return replace(config, **changes)


def apply_thermal_operating_point(
    scenario: ScenarioConfig, operating_point: ThermalOperatingPoint,
) -> ScenarioConfig:
    """Return a derated copy; disabled is an identity/no-op.

    Domain membership is explicit on components and links via
    ``metadata['thermal_domain_id']``. Profiles shared with another domain are
    cloned per component, never edited in place. Route caps on the associated
    ports change along with links; off-domain host CPU/memory remain untouched.
    Memory Core rates and timings change with the corresponding endpoint;
    capacities, address geometry and optional dynamic NAND feedback stay fixed.
    This is a sensitivity operation, not a calibrated prediction of throttling.
    """

    point = operating_point
    if not point.enabled:
        return scenario
    hardware = scenario.hardware
    if hardware.metadata.get("thermal_operating_point", {}).get("enabled"):
        raise ValueError("apply a thermal operating point to the unmodified baseline, not an already derated scenario")
    affected = {component.component_id for component in hardware.components
                if component.metadata.get("thermal_domain_id") == point.domain_id}
    link_ids = {link.link_id for link in hardware.links
                if link.metadata.get("thermal_domain_id") == point.domain_id}
    if not affected and not link_ids:
        raise ValueError("unknown thermal domain: " + point.domain_id)
    ports = {(component_id, port_id)
             for link in hardware.links if link.link_id in link_ids
             for component_id, port_id in ((link.source_component, link.source_port), (link.target_component, link.target_port))}
    profiles = {kind: dict(registry) for kind, registry in scenario.component_profiles.items()}
    components = []
    for component in hardware.components:
        updates = {}
        if component.component_id in affected:
            kind = scenario.component_profile_kind(component)
            if kind is not None:
                profile_id = "{}.thermal.{}".format(component.cost_profile_id, component.component_id)
                if profile_id in profiles[kind]:
                    raise ValueError("thermal profile id already exists: " + profile_id)
                profiles[kind][profile_id] = _derate_profile(scenario.resolve_component_profile(component), point)
                updates["cost_profile_id"] = profile_id
            metadata = dict(component.metadata)
            if metadata.get("physical_memory_config") is not None:
                metadata["physical_memory_config"] = asdict(
                    _derate_physical_memory_config(metadata["physical_memory_config"], point)
                )
            for name in ("read_latency_ns", "write_latency_ns"):
                if name in metadata:
                    metadata[name] *= point.latency_scale
            metadata["thermal_derating"] = {"domain_id": point.domain_id, "mode": "static_operating_point", "evidence": point.evidence}
            updates.update(peak_ops_per_s=component.peak_ops_per_s * point.frequency_scale,
                           bandwidth_gbps=component.bandwidth_gbps * point.memory_bandwidth_scale,
                           read_bandwidth_gbps=component.read_bandwidth_gbps * point.memory_bandwidth_scale,
                           write_bandwidth_gbps=component.write_bandwidth_gbps * point.memory_bandwidth_scale,
                           metadata=metadata)
        if any((component.component_id, port.port_id) in ports for port in component.ports):
            updates["ports"] = tuple(
                replace(port, bandwidth_gbps=port.bandwidth_gbps * point.link_bandwidth_scale)
                if (component.component_id, port.port_id) in ports else port
                for port in component.ports
            )
        components.append(replace(component, **updates) if updates else component)
    links = tuple(
        replace(link, bandwidth_gbps=link.bandwidth_gbps * point.link_bandwidth_scale,
                latency_ns=link.latency_ns * point.latency_scale)
        if link.link_id in link_ids else link
        for link in hardware.links
    )
    hardware = replace(hardware, components=tuple(components), links=links,
                       metadata={**hardware.metadata, "thermal_operating_point": {
                           **asdict(point), "mode": "static_derating_only", "calibrated": False,
                           "dynamic_temperature_model": False,
                       }})
    derated = replace(scenario, hardware=hardware, component_profiles=profiles,
                      assumptions=scenario.assumptions + ("Static thermal derating only: " + point.evidence,))
    changed_memories = tuple(component.component_id for component in scenario.hardware.components
        if component.is_active_memory and any(
            component.metadata.get(key) !=
            derated.hardware.get_component(component.component_id).metadata.get(key)
            for key in ("memory_service", "physical_memory_config")))
    changed_links = tuple(link.link_id for link, updated in zip(scenario.hardware.links, derated.hardware.links)
                          if link != updated)
    if changed_memories or changed_links:
        # Profiles do not yet bind samples to a complete memory/route set.
        # Invalidate all GPU surfaces conservatively rather than guess which
        # remote/aggregate service a measured kernel may have consumed.
        reason = ("thermal memory/route changed: measured surface invalidated; "
                  "dependency binding unavailable, conservative all-GPU scope")
        registry = profiles.get("gpu", {})
        revised = {key: _invalidate_kernel_samples(value, reason)
                   for key, value in registry.items()}
        if revised != registry:
            metadata = {**derated.hardware.metadata, "kernel_calibration_invalidation": {
                "reason": reason, "memory_components": changed_memories,
                "links": changed_links, "scope": "all_gpu_profiles"}}
            derated = replace(derated, component_profiles={**profiles, "gpu": revised},
                              hardware=replace(derated.hardware, metadata=metadata))
    return derated
