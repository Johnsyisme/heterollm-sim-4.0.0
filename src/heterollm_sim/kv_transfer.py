"""Physical KV-page transfer executor.

This module owns the page-to-address mapping used by the paged KV pool.  It
never derives an address from a logical page number: every live page is
materialised through :class:`PhysicalAddressAllocator`.  A transfer reserves
source-memory, link/relay, and target-memory work on one shared
``PhysicalRuntimeContext``.  The returned ``KvPoolTransferPlan`` makes that
reservation reversible until the pool commits its capacity ledger.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .data_motion import AccessKind, PhysicalRuntimeContext, PhysicalService, resolve_service
from .memory_allocator import PhysicalAddressAllocator, PhysicalAllocation
from .memory_types import DramConfig, NandConfig, parse_physical_memory_config
from .kv_pool import KvPoolError, KvPoolTransferPlan, KvPoolTransferRequest


@dataclass
class _PageBinding:
    page_id: int
    request_id: str
    layer_group: str
    component_id: str
    owner: str
    bytes: int
    generation: int
    allocation: PhysicalAllocation


class PhysicalKVTransferExecutor:
    """Lower KV page movement and ordinary page access into physical events.

    ``hardware`` can be a ``HardwareSpec`` (or an object exposing one through
    ``.hardware``), a component mapping, or a sequence of ``ComponentSpec``
    values.  ``topology`` is normally a ``TopologyRouter``.  The optional
    ``relay_components`` sequence forces transfers through those components;
    each relay is written and then read, so relay memory and its own queue are
    charged explicitly.
    """

    def __init__(
        self,
        hardware: Any,
        *,
        runtime: Optional[PhysicalRuntimeContext] = None,
        topology: Any = None,
        relay_components: Sequence[str] = (),
        arrival_ns: Optional[float | Callable[[KvPoolTransferRequest], float]] = 0.0,
        namespace: str = "kv-pool",
    ) -> None:
        self.runtime = runtime or PhysicalRuntimeContext()
        self.topology = topology
        self.arrival_ns = arrival_ns
        self.namespace = str(namespace)
        if not self.namespace:
            raise ValueError("namespace must be non-empty")
        self.components = self._component_map(hardware)
        self.relay_components = tuple(str(value) for value in relay_components)
        unknown = [value for value in self.relay_components if value not in self.components]
        if unknown:
            raise ValueError("unknown relay component(s): " + ", ".join(unknown))
        self.services: dict[str, PhysicalService] = {}
        self._configs: dict[str, DramConfig | NandConfig] = {}
        self._bindings: dict[int, _PageBinding] = {}
        self.events: list[dict[str, Any]] = []
        self.counters = {"read_bytes": 0, "write_bytes": 0, "link_bytes": 0,
                         "relay_bytes": 0, "transfer_count": 0,
                         "service_ns": 0.0, "link_ns": 0.0}
        self._validate_components()

    @staticmethod
    def _component_map(hardware: Any) -> dict[str, Any]:
        if hasattr(hardware, "hardware"):
            hardware = hardware.hardware
        getter = getattr(hardware, "component_map", None)
        if callable(getter):
            return dict(getter())
        if isinstance(hardware, Mapping):
            values = hardware.get("components", hardware)
            if isinstance(values, Mapping):
                return {str(key): value for key, value in values.items()}
            return {str(getattr(item, "component_id", getattr(item, "id", ""))): item for item in values}
        return {str(getattr(item, "component_id", getattr(item, "id", ""))): item for item in hardware}

    @staticmethod
    def _metadata(component: Any) -> Mapping[str, Any]:
        return getattr(component, "metadata", {}) or {}

    @staticmethod
    def _component_id(component: Any) -> str:
        return str(getattr(component, "component_id", getattr(component, "id", "")))

    def _validate_components(self) -> None:
        # Hardware graphs commonly include compute/IO components without a
        # physical memory core. Validate memory descriptors eagerly when they
        # are present, while resolving only page source/target components on
        # demand.
        for ident, component in self.components.items():
            if not ident:
                raise ValueError("physical component id must be non-empty")
            raw = self._metadata(component).get("physical_memory_config")
            if raw is None:
                continue
            self._ensure_service(ident)

    def _ensure_service(self, component_id: str) -> PhysicalService:
        ident = str(component_id)
        if ident in self.services:
            return self.services[ident]
        component = self.components.get(ident)
        if component is None:
            raise KvPoolError("unknown physical component " + ident)
        raw = self._metadata(component).get("physical_memory_config")
        if raw is None:
            raise KvPoolError("component {} requires physical_memory_config".format(ident))
        config = raw if isinstance(raw, (DramConfig, NandConfig)) else parse_physical_memory_config(raw)
        service = resolve_service(component)
        self._configs[ident] = config
        self.services[ident] = service
        self.runtime.runtime(config, service.physical_owner)
        return service

    def _page_id(self, page: Any) -> int:
        value = getattr(page, "logical_page_id", page)
        if isinstance(value, bool):
            raise KvPoolError("logical page id must be an integer")
        return int(value)

    def _binding(self, page: Any) -> _PageBinding:
        try:
            return self._bindings[self._page_id(page)]
        except KeyError as error:
            raise KvPoolError("KV page is not bound to a physical allocation") from error

    def _allocator(self, owner: str) -> PhysicalAddressAllocator:
        allocator = self.runtime.allocators.get(owner)
        if allocator is None:
            raise KvPoolError("physical owner {} has no address allocator".format(owner))
        return allocator

    def _buffer_id(self, page: Any) -> str:
        request = str(getattr(page, "request_id", "request"))
        return "{}:{}:page:{}".format(self.namespace, request, self._page_id(page))

    def bind_page(self, page: Any) -> int:
        """Allocate and bind one page at its declared owner component."""
        page_id = self._page_id(page)
        component_id = str(getattr(page, "owner_component", ""))
        if component_id not in self.components:
            raise KvPoolError("unknown page owner component " + component_id)
        size = int(getattr(page, "bytes", 0))
        generation = int(getattr(page, "allocation_generation", 0))
        if size <= 0 or generation < 0:
            raise KvPoolError("page bytes and allocation generation are invalid")
        existing = self._bindings.get(page_id)
        if existing is not None:
            if existing.component_id != component_id or existing.bytes != size or existing.generation != generation:
                raise KvPoolError("page binding changed without migration commit")
            return existing.allocation.base_address
        service = self._ensure_service(component_id)
        allocator = self._allocator(service.physical_owner)
        allocation = allocator.allocate(self._buffer_id(page), size, generation)
        self._bindings[page_id] = _PageBinding(
            page_id, str(getattr(page, "request_id", "request")),
            str(getattr(page, "layer_group", "kv")), component_id,
            service.physical_owner, size, generation, allocation,
        )
        return allocation.base_address

    def sync_pages(self, pages: Sequence[Any]) -> None:
        """Bind pages that were allocated by a pool after a resize."""
        current = {self._page_id(page) for page in pages}
        # Pool release can remove a page without involving this executor.  Drop
        # the corresponding physical allocation before the next migration so
        # allocator capacity and the logical pool stay in lockstep.
        for page_id in tuple(self._bindings):
            if page_id not in current:
                self.release_page(page_id)
        for page in pages:
            self.bind_page(page)

    def address(self, page: Any, offset_bytes: int = 0, size_bytes: Optional[int] = None) -> int:
        binding = self._binding(page)
        size = binding.bytes - int(offset_bytes) if size_bytes is None else int(size_bytes)
        return binding.allocation.address(int(offset_bytes), size)

    def release_page(self, page: Any) -> None:
        page_id = self._page_id(page)
        binding = self._bindings.pop(page_id, None)
        if binding is None:
            return
        self._allocator(binding.owner).release(binding.allocation.buffer_id, binding.generation)

    def _arrival(self, request: Optional[KvPoolTransferRequest], value: Optional[float]) -> float:
        if value is not None:
            result = float(value)
        elif callable(self.arrival_ns):
            result = float(self.arrival_ns(request))
        elif self.arrival_ns is None:
            result = 0.0
        else:
            result = float(self.arrival_ns)
        if not math.isfinite(result) or result < 0:
            raise ValueError("arrival_ns must be finite and non-negative")
        return result

    @staticmethod
    def _sum_counters(total: dict[str, Any], result: Mapping[str, Any]) -> None:
        for key, value in result.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value

    def _price_chunks(self, component_id: str, kind: AccessKind, address: int,
                      size: int, *, arrival_ns: float) -> tuple[dict[str, Any], ...]:
        service = self._ensure_service(component_id)
        config = self._configs[component_id]
        limit = int(getattr(config, "write_buffer_bytes", 0)) if kind is AccessKind.WRITE else 0
        granularity = int(getattr(config, "host_granularity_bytes", 0) or getattr(config, "burst_bytes", 0) or 1)
        if limit > 0 and limit < granularity:
            raise KvPoolError("write buffer is smaller than one host transaction granularity")
        chunk = min(size, limit) if limit > 0 else size
        if limit > 0:
            chunk = max(granularity, (chunk // granularity) * granularity)
        chunk = max(1, chunk)
        results = []
        offset = 0
        current = arrival_ns
        while offset < size:
            amount = min(chunk, size - offset)
            bill = service.price(kind, amount, page_offset_bytes=address + offset,
                                 runtime=self.runtime, arrival_ns=current)
            results.append(dict(bill))
            current = float(bill.get("completion_ns", current))
            offset += amount
        if kind is AccessKind.READ:
            self.counters["read_bytes"] += size
        elif kind is AccessKind.WRITE:
            self.counters["write_bytes"] += size
        self.counters["service_ns"] += sum(
            float(item.get("service_ns", 0.0)) for item in results
        )
        return tuple(results)

    def read_page(self, page: Any, *, arrival_ns: Optional[float] = None,
                  offset_bytes: int = 0, size_bytes: Optional[int] = None) -> Mapping[str, Any]:
        binding = self._binding(page)
        size = binding.bytes - int(offset_bytes) if size_bytes is None else int(size_bytes)
        address = binding.allocation.address(int(offset_bytes), size)
        total: dict[str, Any] = {"operation": "READ", "logical_bytes": 0}
        for bill in self._price_chunks(binding.component_id, AccessKind.READ, address, size,
                                       arrival_ns=self._arrival(None, arrival_ns)):
            self._sum_counters(total, bill)
        if total.get("completion_ns") is None:
            total["completion_ns"] = total.get("arrival_ns", 0.0)
        return total

    def write_page(self, page: Any, *, arrival_ns: Optional[float] = None,
                   offset_bytes: int = 0, size_bytes: Optional[int] = None) -> Mapping[str, Any]:
        binding = self._binding(page)
        size = binding.bytes - int(offset_bytes) if size_bytes is None else int(size_bytes)
        address = binding.allocation.address(int(offset_bytes), size)
        total: dict[str, Any] = {"operation": "WRITE", "logical_bytes": 0}
        for bill in self._price_chunks(binding.component_id, AccessKind.WRITE, address, size,
                                       arrival_ns=self._arrival(None, arrival_ns)):
            self._sum_counters(total, bill)
        if total.get("completion_ns") is None:
            total["completion_ns"] = total.get("arrival_ns", 0.0)
        return total

    def _route(self, source: str, target: str, size: int) -> Sequence[Any]:
        if source == target:
            return ()
        if self.topology is None:
            return ()
        route = getattr(self.topology, "route", None)
        if route is None:
            raise KvPoolError("topology must expose route(source, target, bytes)")
        try:
            return tuple(route(source, target, size))
        except TypeError:
            return tuple(route(source, target, size, policy="lowest_latency"))

    def _link_segment(self, source: str, target: str, size: int, *, arrival_ns: float) -> tuple[float, list[dict[str, Any]]]:
        current = arrival_ns
        phases: list[dict[str, Any]] = []
        for hop in self._route(source, target, size):
            # TopologyRouter exposes a local memory-service view as a route
            # hop with ``service_ref``.  The source/target PhysicalService
            # bills already charge that queue; reserving the view again would
            # double-count the same medium.  Only independent fabric links
            # consume link timeline capacity here.
            if getattr(hop, "service_ref", None):
                continue
            duration = float(hop.transfer_ns(size))
            transfer = self.runtime.timeline.reserve(hop.resource_id, current, duration,
                                                     direction="read_write")
            self.runtime.timeline.bytes_moved[hop.resource_id] = self.runtime.timeline.bytes_moved.get(hop.resource_id, 0) + size
            current = transfer.end_ns
            phases.append({"kind": "link", "link_id": hop.link_id,
                           "resource_id": hop.resource_id, "bytes": size,
                           "start_ns": transfer.start_ns, "completion_ns": transfer.end_ns,
                           "duration_ns": duration})
            self.counters["link_bytes"] += size
            self.counters["link_ns"] += duration
        return current, phases

    def _transfer(self, request: KvPoolTransferRequest) -> KvPoolTransferPlan:
        page_id = int(request.logical_page_id)
        binding = self._bindings.get(page_id)
        if binding is None:
            raise KvPoolError("KV page {} is not physically bound".format(page_id))
        if binding.component_id != request.source_component or binding.bytes != int(request.bytes):
            raise KvPoolError("transfer request does not match the bound source page")
        if int(request.allocation_generation) != binding.generation:
            raise KvPoolError("transfer request generation does not match bound page")
        target_component = str(request.target_component)
        if target_component not in self.components:
            raise KvPoolError("unknown transfer target " + target_component)
        target_service = self._ensure_service(target_component)
        target_owner = target_service.physical_owner
        snapshot = self.runtime.snapshot()
        bindings_snapshot = copy.deepcopy(self._bindings)
        events_snapshot = len(self.events)
        counters_snapshot = copy.deepcopy(self.counters)
        temp_allocations: list[tuple[PhysicalAddressAllocator, str, int]] = []
        target_allocator = self._allocator(target_owner)
        target_generation = binding.generation + 1
        start = self._arrival(request, None)
        phases: list[dict[str, Any]] = []
        try:
            target_alloc = target_allocator.allocate(self._buffer_id(request), binding.bytes, target_generation)
            source_addr = binding.allocation.base_address
            read_bills = self._price_chunks(binding.component_id, AccessKind.READ, source_addr,
                                            binding.bytes, arrival_ns=start)
            current = float(read_bills[-1].get("completion_ns", start)) if read_bills else start
            for bill in read_bills:
                phases.append({"kind": "source_read", **bill})
            route_stages = (self.relay_components + (target_component,)) if self.relay_components else (target_component,)
            current_component = binding.component_id
            for stage in route_stages[:-1]:
                current, link_phases = self._link_segment(current_component, stage, binding.bytes, arrival_ns=current)
                phases.extend(link_phases)
                relay_allocator = self._allocator(self._ensure_service(stage).physical_owner)
                relay_allocation = relay_allocator.allocate(
                    self._buffer_id(request) + ":relay:" + stage, binding.bytes, target_generation)
                temp_allocations.append((relay_allocator, relay_allocation.buffer_id, target_generation))
                relay_bills = self._price_chunks(stage, AccessKind.WRITE, relay_allocation.base_address,
                    binding.bytes, arrival_ns=current)
                for bill in relay_bills:
                    phases.append({"kind": "relay_write", "component": stage, **bill})
                current = float(relay_bills[-1].get("completion_ns", current)) if relay_bills else current
                relay_read = self._price_chunks(stage, AccessKind.READ,
                    relay_allocation.base_address,
                    binding.bytes, arrival_ns=current)
                for bill in relay_read:
                    phases.append({"kind": "relay_read", "component": stage, **bill})
                current = float(relay_read[-1].get("completion_ns", current)) if relay_read else current
                self.counters["relay_bytes"] += 2 * binding.bytes
                current_component = stage
            current, link_phases = self._link_segment(current_component, target_component, binding.bytes, arrival_ns=current)
            phases.extend(link_phases)
            target_bills = self._price_chunks(target_component, AccessKind.WRITE, target_alloc.base_address,
                                              binding.bytes, arrival_ns=current)
            for bill in target_bills:
                phases.append({"kind": "target_write", **bill})
            completion = float(target_bills[-1].get("completion_ns", current)) if target_bills else current
        except Exception:
            self.runtime.restore(snapshot)
            self._bindings = bindings_snapshot
            self.counters = counters_snapshot
            raise
        state = {"done": False}
        plan = KvPoolTransferPlan(
            duration_ns=max(0.0, completion - start), start_ns=start,
            completion_ns=completion,
            commit=lambda: self._commit_transfer(page_id, target_component, target_owner, target_alloc,
                                                 target_generation, phases, temp_allocations, state),
            rollback=lambda: self._rollback_transfer(snapshot, bindings_snapshot, counters_snapshot, events_snapshot, target_component,
                                                     target_owner, target_alloc, temp_allocations, state),
        )
        return plan

    def _commit_transfer(self, page_id: int, target_component: str, target_owner: str,
                         target_alloc: PhysicalAllocation, target_generation: int,
                         phases: list[dict[str, Any]], temp_allocations: list[tuple[PhysicalAddressAllocator, str, int]],
                         state: dict[str, bool]) -> None:
        if state["done"]:
            return
        binding = self._bindings[page_id]
        old = binding
        self._allocator(old.owner).release(old.allocation.buffer_id, old.generation)
        self._bindings[page_id] = _PageBinding(old.page_id, old.request_id, old.layer_group,
                                               target_component, target_owner, old.bytes,
                                               target_generation, target_alloc)
        for allocator, buffer_id, generation in temp_allocations:
            allocator.release(buffer_id, generation)
        self.events.append({"logical_page_id": page_id, "source_component": old.component_id,
                            "target_component": target_component, "bytes": old.bytes,
                            "start_ns": phases[0].get("arrival_ns", 0.0) if phases else 0.0,
                            "completion_ns": phases[-1].get("completion_ns", 0.0) if phases else 0.0,
                            "duration_ns": 0.0 if not phases else phases[-1].get("completion_ns", 0.0) - phases[0].get("arrival_ns", 0.0),
                            "phases": tuple(phases)})
        self.counters["transfer_count"] += 1
        state["done"] = True

    def _rollback_transfer(self, snapshot: Mapping[str, Any], bindings: Mapping[int, _PageBinding],
                           counters: Mapping[str, Any], events_len: int, target_component: str, target_owner: str,
                           target_alloc: PhysicalAllocation, temp_allocations: list[tuple[PhysicalAddressAllocator, str, int]],
                           state: dict[str, bool]) -> None:
        if state["done"]:
            return
        self.runtime.restore(snapshot)
        self._bindings = copy.deepcopy(bindings)
        self.counters = copy.deepcopy(counters)
        del self.events[events_len:]
        state["done"] = True

    def __call__(self, request: KvPoolTransferRequest) -> KvPoolTransferPlan:
        return self._transfer(request)

    def snapshot(self) -> dict[str, Any]:
        return {"runtime": self.runtime.snapshot(), "bindings": copy.deepcopy(self._bindings),
                "events": copy.deepcopy(self.events), "counters": copy.deepcopy(self.counters)}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.runtime.restore(snapshot["runtime"])
        self._bindings = copy.deepcopy(snapshot["bindings"])
        self.events = copy.deepcopy(snapshot["events"])
        self.counters = copy.deepcopy(snapshot["counters"])

    def stats(self) -> Mapping[str, Any]:
        return {**self.counters, "events": len(self.events)}


__all__ = ["PhysicalKVTransferExecutor"]
