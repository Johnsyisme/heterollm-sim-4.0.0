"""Shared page-level NAND core for SSD and HBF configurations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional

from .memory_mapping import map_nand_address, split_nand_request, validate_nand_request
from .memory_transfer import ResourceTimeline
from .memory_types import AccessRequest, NandConfig, Operation, StageTiming, TransactionResult


@dataclass
class _StagedPage:
    """Host-side state for one dirty NAND page.

    The simulator intentionally does not model cell contents.  Keeping the
    physical mapping and byte counters is enough to preserve the important
    contract: repeated small writes merge into one eventual page program and
    reads observe the staged copy until the program drains to media.
    """

    mapping: object
    host_bytes: int = 0
    logical_bytes: int = 0
    request_count: int = 0


class NandCore:
    """Model page read, page program and block erase without FTL/GC state."""

    def __init__(
        self,
        config: NandConfig,
        timeline: Optional[ResourceTimeline] = None,
        *,
        capture_details: bool = True,
    ) -> None:
        if not isinstance(config, NandConfig):
            raise TypeError("config must be NandConfig")
        if not isinstance(capture_details, bool):
            raise TypeError("capture_details must be a boolean")
        self.config = config
        self.timeline = timeline or ResourceTimeline()
        # Keep exact stage/mapping traces for normal runs; aggregate serving
        # estimates can disable them while preserving scalar timing/counters.
        self.capture_details = capture_details
        self._array_ready: Dict[str, float] = {}
        self._buffer_ready: Dict[str, float] = {}
        self._inflight: list[float] = []
        self._acceptance_ns = 0.0
        # A positive write_buffer_bytes models a shared host-side staging
        # budget.  Entries are released when their media program completes;
        # zero preserves the existing write-through path.
        self._write_buffer_pending: list[tuple[float, int]] = []
        self._write_buffer_occupancy = 0
        # ``buffered_write_mode`` is an explicit opt-in.  Keeping the state
        # separate means legacy positive write_buffer_bytes configurations
        # retain their write-through timing and completion semantics.
        self._buffered_write_mode = config.buffered_write_mode
        self._staged_pages: Dict[tuple[int, ...], _StagedPage] = {}
        self._buffer_merge_bytes = 0

    @property
    def buffered_write_mode(self) -> bool:
        """Return whether this core exposes host-ack/media-drain semantics."""

        return self._buffered_write_mode

    @property
    def pending_write_buffer_bytes(self) -> int:
        """Bytes currently held in the opt-in host staging buffer."""

        return int(self._write_buffer_occupancy if self._buffered_write_mode else 0)

    @property
    def buffered_page_count(self) -> int:
        """Number of dirty physical pages awaiting a media program."""

        return len(self._staged_pages)

    def _array_id(self, mapping) -> str:
        prefix = str(self.config.metadata.get("array_resource_prefix", "nand:array"))
        return f"{prefix}:{mapping.unit}"

    def _array_resource_id(self, mapping) -> str:
        """Return the execution slot, without changing physical mapping."""
        array_id = self._array_id(mapping)
        if self.config.parallel_units is None:
            return array_id
        prefix = str(self.config.metadata.get("parallel_resource_prefix", "nand:parallel"))
        return f"{prefix}:{mapping.unit % self.config.parallel_units}"

    def _channel_id(self, mapping) -> str:
        prefix = str(self.config.metadata.get("channel_resource_prefix", "nand:channel"))
        return f"{prefix}:{mapping.channel}"

    def _host_id(self) -> str:
        return str(self.config.metadata.get("host_resource_id", "nand:host"))

    def _bandwidth_ceiling_gb_s(self, operation: Optional[Operation | str] = None) -> float:
        """Return the end-to-end payload ceiling for the NAND data path.

        A request crosses both the host link and the NAND internal channel.
        The slower leg is therefore the ceiling exposed to reports.  The
        reservation timeline still models each leg separately; this helper
        only fixes the derived bottleneck/utilization metric.
        """
        # Keep the no-operation form as the legacy aggregate ceiling used by
        # batch summaries. A transaction uses its own directional host link.
        host_bandwidth = (
            float(self.config.host_bandwidth_gb_s)
            if operation is None
            else self.config.host_bandwidth_for(operation)
        )
        return min(host_bandwidth, float(self.config.internal_bandwidth_gb_s))

    def _array_stage(self, name: str, resource: str, earliest: float, duration: float, bytes: int = 0) -> StageTiming:
        reservation = self.timeline.reserve(resource, earliest, duration)
        return StageTiming(name, reservation.start_ns, reservation.end_ns, resource, bytes)

    def _write_buffer_payload_bytes(self, request: AccessRequest) -> int:
        if request.operation is not Operation.WRITE or self.config.write_buffer_bytes <= 0:
            return 0
        return sum(
            int(segment.host_transfer_bytes)
            for segment in split_nand_request(request, self.config)
        )

    @staticmethod
    def _staged_key(mapping) -> tuple[int, ...]:
        """Build a stable physical-page key from an address mapping."""

        return (
            int(mapping.stack), int(mapping.lane), int(mapping.subchannel),
            int(mapping.pseudo_channel), int(mapping.rank),
            int(mapping.bank_group), int(mapping.bank),
            int(mapping.channel), int(mapping.target), int(mapping.die),
            int(mapping.lun), int(mapping.plane), int(mapping.block), int(mapping.page),
        )

    def _stage_page(self, segment) -> tuple[int, int]:
        """Merge one write segment into staging and return (merge, new)."""

        key = self._staged_key(segment.mapping)
        page = self._staged_pages.get(key)
        merge_bytes = int(segment.host_transfer_bytes) if page is not None else 0
        if page is None:
            page = _StagedPage(mapping=segment.mapping)
            self._staged_pages[key] = page
        page.host_bytes += int(segment.host_transfer_bytes)
        page.logical_bytes += int(segment.logical_bytes)
        page.request_count += 1
        self._buffer_merge_bytes += merge_bytes
        return merge_bytes, 1 if merge_bytes else 0

    def _release_write_buffer(self, now_ns: float) -> None:
        if not self._write_buffer_pending:
            return
        retained = []
        released = 0
        for release_ns, byte_count in self._write_buffer_pending:
            if release_ns <= now_ns:
                released += byte_count
            else:
                retained.append((release_ns, byte_count))
        if released:
            self._write_buffer_occupancy = max(0, self._write_buffer_occupancy - released)
            self._write_buffer_pending = retained

    def _admit_write_buffer(
        self, request: AccessRequest, effective_arrival: float,
    ) -> tuple[float, int, float]:
        """Delay admission until a finite write buffer has room.

        Capacity is measured in host payload bytes, matching the host transfer
        stages and avoiding a second page-level accounting convention.
        """
        payload_bytes = self._write_buffer_payload_bytes(request)
        if not payload_bytes:
            self._release_write_buffer(effective_arrival)
            return effective_arrival, 0, 0.0
        capacity = self.config.write_buffer_bytes
        if payload_bytes > capacity:
            raise ValueError(
                "write request requires {} write-buffer bytes, exceeds configured capacity {}"
                .format(payload_bytes, capacity)
            )
        initial_arrival = effective_arrival
        if self._buffered_write_mode:
            # In buffered mode occupancy persists until an explicit or
            # automatic drain.  A full buffer therefore stalls admission at
            # the media-drain completion point instead of pretending that a
            # host acknowledgement freed the bytes.
            while self._write_buffer_occupancy + payload_bytes > capacity:
                drained = self.flush(now_ns=effective_arrival)
                drain_end = float(drained.get("media_drain_completion_ns", effective_arrival))
                if drain_end <= effective_arrival and self._write_buffer_occupancy:
                    raise RuntimeError("write buffer occupancy cannot be drained")
                effective_arrival = max(effective_arrival, drain_end)
            return effective_arrival, payload_bytes, max(0.0, effective_arrival - initial_arrival)
        while True:
            self._release_write_buffer(effective_arrival)
            if self._write_buffer_occupancy + payload_bytes <= capacity:
                return effective_arrival, payload_bytes, max(0.0, effective_arrival - initial_arrival)
            future = [
                release_ns for release_ns, _ in self._write_buffer_pending
                if release_ns > effective_arrival
            ]
            if not future:
                raise RuntimeError("write buffer occupancy cannot be drained")
            effective_arrival = min(future)

    def _reserve_write_buffer(self, completion_ns: float, payload_bytes: int) -> None:
        if payload_bytes <= 0:
            return
        if self._buffered_write_mode:
            self._write_buffer_occupancy += payload_bytes
            return
        self._write_buffer_pending.append((completion_ns, payload_bytes))
        self._write_buffer_occupancy += payload_bytes

    def flush_write_buffer(self, *, now_ns: Optional[float] = None) -> dict:
        """Program all dirty staged pages and return drain accounting.

        Host acknowledgements returned by buffered writes are deliberately
        separate from this media completion.  The return value is a small
        JSON-friendly mapping so serving/reporting code can include it without
        introducing a synthetic request into the event stream.
        """

        if not self._buffered_write_mode:
            return {
                "buffered_write_mode": False,
                "flushed_pages": 0,
                "merge_bytes": 0,
                "physical_programs": 0,
                "physical_write_bytes": 0,
                "media_drain_start_ns": float(now_ns or 0.0),
                "media_drain_completion_ns": float(now_ns or 0.0),
                "write_completion": "media_program_complete",
            }
        if not self._staged_pages:
            value = float(
                now_ns if now_ns is not None else max(
                    self._acceptance_ns, max(self.timeline.ready_ns.values(), default=0.0)
                )
            )
            return {
                "buffered_write_mode": True,
                "flushed_pages": 0,
                "merge_bytes": 0,
                "physical_programs": 0,
                "physical_write_bytes": 0,
                "media_drain_start_ns": value,
                "media_drain_completion_ns": value,
                "write_completion": "media_program_complete",
            }
        cursor = float(
            now_ns if now_ns is not None else max(
                self._acceptance_ns, max(self.timeline.ready_ns.values(), default=0.0)
            )
        )
        start_ns: Optional[float] = None
        physical_programs = 0
        physical_write_bytes = 0
        internal_transfer_bytes = 0
        host_bytes = 0
        for staged in tuple(self._staged_pages.values()):
            m = staged.mapping
            array_id = self._array_id(m)
            array_resource = self._array_resource_id(m)
            dependency = max(cursor, self._array_ready.get(array_id, 0.0), self._buffer_ready.get(array_id, 0.0))
            internal = self.timeline.transfer(
                self._channel_id(m), self.config.transfer_page_bytes,
                dependency, self.config.internal_bandwidth_gb_s,
                direction="write",
            )
            program = self._array_stage(
                "PAGE_PROGRAM", array_resource, internal.end_ns, self.config.page_program_ns
            )
            self._array_ready[array_id] = program.end_ns
            self._buffer_ready[array_id] = program.end_ns
            self.timeline.ready_ns[array_id] = program.end_ns
            start_ns = internal.start_ns if start_ns is None else min(start_ns, internal.start_ns)
            cursor = max(cursor, program.end_ns)
            physical_programs += 1
            physical_write_bytes += self.config.page_bytes
            internal_transfer_bytes += internal.bytes
            host_bytes += staged.host_bytes
        merge_bytes = int(self._buffer_merge_bytes)
        flushed_pages = len(self._staged_pages)
        self._staged_pages.clear()
        self._buffer_merge_bytes = 0
        self._write_buffer_occupancy = 0
        return {
            "buffered_write_mode": True,
            "flushed_pages": flushed_pages,
            "merge_bytes": merge_bytes,
            "physical_programs": physical_programs,
            "physical_write_bytes": physical_write_bytes,
            "host_transfer_bytes": host_bytes,
            "internal_transfer_bytes": internal_transfer_bytes,
            "media_drain_start_ns": float(start_ns if start_ns is not None else cursor),
            "media_drain_completion_ns": float(cursor),
            "write_completion": "media_program_complete",
        }

    def drain(self, *, now_ns: Optional[float] = None) -> dict:
        """Alias for :meth:`flush_write_buffer` used by serving callers."""

        return self.flush_write_buffer(now_ns=now_ns)

    def flush(self, *, now_ns: Optional[float] = None) -> dict:
        """Short alias for explicit write-buffer draining."""

        return self.flush_write_buffer(now_ns=now_ns)

    def execute(self, request: AccessRequest) -> TransactionResult:
        """Submit one request through the same admission gate as batches."""
        return self.submit(request)

    def submit(self, request: AccessRequest) -> TransactionResult:
        validate_nand_request(request, self.config)
        effective_arrival = max(request.arrival_ns, self._acceptance_ns)
        inflight = [end for end in self._inflight if end > effective_arrival]
        if len(inflight) >= self.config.max_outstanding_requests:
            effective_arrival = max(effective_arrival, min(inflight))
            inflight = [end for end in inflight if end > effective_arrival]
        effective_arrival, write_buffer_bytes, write_buffer_wait_ns = self._admit_write_buffer(
            request, effective_arrival
        )
        before_metrics = self.timeline.metrics_snapshot(
            max_intervals=(
                self.config.max_expanded_segments
                if self.capture_details
                else 0
            )
        )
        result = self._execute_accepted(
            request if effective_arrival == request.arrival_ns
            else replace(request, arrival_ns=effective_arrival)
        )
        result = replace(result, arrival_ns=request.arrival_ns, counters={
            **result.counters,
            "queue_wait_ns": effective_arrival - request.arrival_ns,
            "write_buffer_wait_ns": write_buffer_wait_ns,
            "write_buffer_bytes": write_buffer_bytes,
            "write_buffer_occupancy_bytes": self._write_buffer_occupancy + write_buffer_bytes,
            "write_buffer_peak_bytes": self._write_buffer_occupancy + write_buffer_bytes,
            "buffered_write_mode": self._buffered_write_mode,
            **self.timeline.metrics_delta(before_metrics),
        })
        self._reserve_write_buffer(result.completion_ns, write_buffer_bytes)
        self._acceptance_ns = effective_arrival
        self._inflight = [*inflight, result.completion_ns]
        return result

    def _execute_accepted(self, request: AccessRequest) -> TransactionResult:
        first_page = request.address // self.config.page_bytes
        last_page = (request.address + request.byte_count - 1) // self.config.page_bytes
        page_count = last_page - first_page + 1
        if (
            not self.capture_details
            and request.operation is not Operation.ERASE
            and page_count >= 128
            # Positive write buffers are admitted at request granularity so
            # every buffered write observes the same capacity gate.
            and not (
                request.operation is Operation.WRITE
                and self.config.write_buffer_bytes > 0
            )
        ):
            return self._execute_accelerated_accepted(request, page_count)
        return self._execute_detailed_accepted(request)

    def _execute_detailed_accepted(self, request: AccessRequest) -> TransactionResult:
        segments = split_nand_request(request, self.config)
        stages = []
        mappings = []
        segment_count = 0
        details_truncated = False
        def record(stage):
            if self.capture_details and not details_truncated:
                stages.append(stage)
        completion = request.arrival_ns + self.config.front_ns
        logical_bytes = 0
        host_bytes = internal_bytes = 0
        pages_read = pages_programmed = erases = 0
        merge_bytes = 0
        merged_pages = 0
        staging_write_pages = 0
        staging_read_hits = 0
        for segment in segments:
            segment_count += 1
            if segment_count > self.config.max_expanded_segments and not details_truncated:
                stages.clear(); mappings.clear(); details_truncated = True
            m = segment.mapping
            if self.capture_details and not details_truncated:
                mappings.append(m)
            logical_bytes += segment.logical_bytes
            array_id = self._array_id(m)
            array_resource = self._array_resource_id(m)
            array_ready = self._array_ready.get(array_id, 0.0)
            buffer_ready = self._buffer_ready.get(array_id, 0.0)
            dependency = max(request.arrival_ns + self.config.front_ns, array_ready, buffer_ready)
            if request.operation is Operation.ERASE:
                stage = self._array_stage("BLOCK_ERASE", array_resource, dependency, self.config.block_erase_ns)
                record(stage)
                self._array_ready[array_id] = stage.end_ns
                self.timeline.ready_ns[array_id] = stage.end_ns
                completion = max(completion, stage.end_ns)
                erases += 1
                continue

            host_bytes += segment.host_transfer_bytes
            internal_bytes += segment.internal_transfer_bytes
            partial = segment.logical_bytes != self.config.page_bytes
            if (
                request.operation is Operation.WRITE
                and self._buffered_write_mode
                and self.config.write_buffer_bytes > 0
            ):
                # Host acknowledgement completes when the payload reaches the
                # staging buffer.  A later flush performs the one physical
                # page program for each distinct dirty page.
                internal_bytes -= segment.internal_transfer_bytes
                host = self.timeline.transfer(
                    self._host_id(), segment.host_transfer_bytes,
                    request.arrival_ns + self.config.front_ns,
                    self.config.host_bandwidth_for(Operation.WRITE),
                    direction="write",
                )
                record(StageTiming("HOST_TRANSFER", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                merged, merged_page = self._stage_page(segment)
                merge_bytes += merged
                merged_pages += merged_page
                staging_write_pages += 1
                completion = max(completion, host.end_ns)
                continue
            if request.operation is Operation.WRITE and partial and self.config.partial_page_policy == "read_modify_write":
                # Boundary pages need a simplified old-data read before the
                # new bytes can be programmed.  This is intentionally not a
                # persistent page cache.
                read = self._array_stage("RMW_PAGE_READ", array_resource, dependency, self.config.page_read_ns)
                record(read)
                dependency = read.end_ns
                read_xfer = self.timeline.transfer(
                    self._channel_id(m), self.config.transfer_page_bytes,
                    dependency, self.config.internal_bandwidth_gb_s,
                    direction="read",
                )
                record(StageTiming("RMW_INTERNAL_TRANSFER", read_xfer.start_ns, read_xfer.end_ns, read_xfer.resource_id, read_xfer.bytes))
                internal_bytes += read_xfer.bytes
                dependency = read_xfer.end_ns
                self._buffer_ready[array_id] = read_xfer.end_ns
                pages_read += 1
            if request.operation is Operation.READ:
                staged = self._staged_pages.get(self._staged_key(m)) if self._buffered_write_mode else None
                if staged is not None:
                    # The host copy is authoritative until media drain.  Do
                    # not charge a stale PAGE_READ or internal NAND transfer.
                    internal_bytes -= segment.internal_transfer_bytes
                    host = self.timeline.transfer(
                        self._host_id(), segment.host_transfer_bytes,
                        request.arrival_ns + self.config.front_ns,
                        self.config.host_bandwidth_for(Operation.READ),
                        direction="read",
                    )
                    record(StageTiming("STAGING_READ", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                    completion = max(completion, host.end_ns)
                    staging_read_hits += 1
                    continue
                read = self._array_stage("PAGE_READ", array_resource, dependency, self.config.page_read_ns)
                record(read)
                pages_read += 1
                dependency = read.end_ns
                internal = self.timeline.transfer(
                    self._channel_id(m), segment.internal_transfer_bytes,
                    dependency, self.config.internal_bandwidth_gb_s,
                    direction="read",
                )
                record(StageTiming("INTERNAL_TRANSFER", internal.start_ns, internal.end_ns, internal.resource_id, internal.bytes))
                host = self.timeline.transfer(
                    self._host_id(), segment.host_transfer_bytes,
                    internal.end_ns, self.config.host_bandwidth_for(Operation.READ),
                    direction="read",
                )
                record(StageTiming("HOST_TRANSFER", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                completion = max(completion, host.end_ns)
                self._array_ready[array_id] = read.end_ns
                self._buffer_ready[array_id] = internal.end_ns
                self.timeline.ready_ns[array_id] = read.end_ns
            else:
                host = self.timeline.transfer(
                    self._host_id(), segment.host_transfer_bytes,
                    request.arrival_ns + self.config.front_ns,
                    self.config.host_bandwidth_for(Operation.WRITE),
                    direction="write",
                )
                record(StageTiming("HOST_TRANSFER", host.start_ns, host.end_ns, host.resource_id, host.bytes))
                internal = self.timeline.transfer(
                    self._channel_id(m), segment.internal_transfer_bytes,
                    max(host.end_ns, self._buffer_ready.get(array_id, 0.0)),
                    self.config.internal_bandwidth_gb_s,
                    direction="write",
                )
                record(StageTiming("INTERNAL_TRANSFER", internal.start_ns, internal.end_ns, internal.resource_id, internal.bytes))
                program = self._array_stage("PAGE_PROGRAM", array_resource, internal.end_ns, self.config.page_program_ns)
                record(program)
                pages_programmed += 1
                completion = max(completion, program.end_ns)
                self._array_ready[array_id] = program.end_ns
                self._buffer_ready[array_id] = program.end_ns
                self.timeline.ready_ns[array_id] = program.end_ns
        return TransactionResult(
            request_id=request.request_id, operation=request.operation,
            arrival_ns=request.arrival_ns, completion_ns=completion,
            logical_bytes=logical_bytes,
            transfer_bytes=internal_bytes,
            mapping=tuple(mappings), stages=tuple(stages),
            counters={
                "host_transfer_bytes": host_bytes,
                "internal_transfer_bytes": internal_bytes,
                "physical_read_bytes": pages_read * self.config.page_bytes,
                "physical_write_bytes": pages_programmed * self.config.page_bytes,
                "pages_read": pages_read,
                "pages_programmed": pages_programmed,
                "merge_bytes": merge_bytes,
                "merged_pages": merged_pages,
                "staging_write_pages": staging_write_pages,
                "staging_read_hits": staging_read_hits,
                "erase_operations": erases,
                "page_count": segment_count,
                "details_truncated": details_truncated,
                "bandwidth_ceiling_gb_s": self._bandwidth_ceiling_gb_s(request.operation),
                "array_units": self.config.array_units,
                "write_completion": (
                    "host_buffer_ack_pending_media_drain"
                    if request.operation is Operation.WRITE and staging_write_pages
                    else "media_program_complete" if request.operation is Operation.WRITE else "n/a"
                ),
                "host_ack_completion_ns": completion if staging_write_pages else None,
                "media_drain_completion_ns": (
                    None if staging_write_pages
                    else completion if request.operation is Operation.WRITE else None
                ),
            },
        )

    def _execute_accelerated_accepted(self, request: AccessRequest, page_count: int) -> TransactionResult:
        from ._nand_acceleration import execute_full_pages

        page = self.config.page_bytes
        end = request.address + request.byte_count
        cursor = request.address
        parts = []
        accelerated_pages = 0
        # Partial boundary pages retain the original RMW/transfer transitions.
        if cursor % page:
            length = min(end - cursor, page - cursor % page)
            parts.append(self._execute_detailed_accepted(replace(request, address=cursor, byte_count=length)))
            cursor += length
        full_pages = (end - cursor) // page
        if full_pages:
            result = execute_full_pages(self, request, cursor // page, full_pages)
            if result is not None:
                accelerated_pages = result["accelerated_pages"]
                count = result["page_count"]
                parts.append(TransactionResult(
                    request_id=request.request_id, operation=request.operation,
                    arrival_ns=request.arrival_ns, completion_ns=result["completion_ns"],
                    logical_bytes=count * page, transfer_bytes=result["internal_transfer_bytes"],
                    counters={
                        "host_transfer_bytes": result["host_transfer_bytes"],
                        "internal_transfer_bytes": result["internal_transfer_bytes"],
                        "pages_read": count if request.operation is Operation.READ else 0,
                        "pages_programmed": count if request.operation is Operation.WRITE else 0,
                    },
                ))
                cursor += count * page
        if cursor < end:
            parts.append(self._execute_detailed_accepted(replace(request, address=cursor, byte_count=end - cursor)))
        pages_read = sum(part.counters["pages_read"] for part in parts)
        pages_programmed = sum(part.counters["pages_programmed"] for part in parts)
        host_bytes = sum(part.counters["host_transfer_bytes"] for part in parts)
        internal_bytes = sum(part.counters["internal_transfer_bytes"] for part in parts)
        return TransactionResult(
            request_id=request.request_id, operation=request.operation,
            arrival_ns=request.arrival_ns, completion_ns=max(part.completion_ns for part in parts),
            logical_bytes=request.byte_count, transfer_bytes=internal_bytes,
            counters={
                "host_transfer_bytes": host_bytes, "internal_transfer_bytes": internal_bytes,
                "physical_read_bytes": pages_read * page, "physical_write_bytes": pages_programmed * page,
                "pages_read": pages_read, "pages_programmed": pages_programmed,
                "erase_operations": 0, "page_count": page_count,
                "details_truncated": page_count > self.config.max_expanded_segments,
                "accelerated_pages": accelerated_pages,
                "bandwidth_ceiling_gb_s": self._bandwidth_ceiling_gb_s(request.operation),
                "array_units": self.config.array_units,
                "write_completion": "media_program_complete" if request.operation is Operation.WRITE else "n/a",
            },
        )

    def execute_batch(self, requests: Iterable[AccessRequest]) -> tuple[TransactionResult, ...]:
        return tuple(self.submit(request) for request in requests)

    def run(self, requests: Iterable[AccessRequest]):
        """Execute a batch and return one aggregate with achieved bandwidth."""
        from .memory_transfer import summarize_batch
        return summarize_batch(
            self.execute_batch(requests),
            bandwidth_ceiling_gb_s=self._bandwidth_ceiling_gb_s(),
        )


def nand_service(config: NandConfig, request: AccessRequest, *, timeline: Optional[ResourceTimeline] = None) -> TransactionResult:
    return NandCore(config, timeline).execute(request)


__all__ = ["NandCore", "nand_service"]
