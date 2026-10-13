"""Physical page access must preserve byte and timestamp accounting."""
from dataclasses import replace
from types import SimpleNamespace

from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
import test_paged_pool_serving as serving_fixtures


def _executor():
    hardware = serving_fixtures.PagedPoolServingTests._scenario_with_hbf().hardware
    component = hardware.get_component("hbf0")
    metadata = dict(component.metadata)
    metadata["physical_memory_config"] = replace(metadata["physical_memory_config"], write_buffer_bytes=4096)
    metadata["write_buffer_bytes"] = 4096
    executor = PhysicalKVTransferExecutor([replace(component, metadata=metadata)])
    page = SimpleNamespace(logical_page_id=0, owner_component="hbf0", bytes=16384,
                           allocation_generation=0, request_id="test", layer_group="kv")
    executor.bind_page(page)
    return executor, page


def test_chunked_write_uses_last_completion_instead_of_sum():
    executor, page = _executor()
    result = executor.write_page(page, arrival_ns=1000.0)
    active = executor.runtime.runtimes[executor.services["hbf0"].physical_owner]
    assert result["arrival_ns"] == 1000.0
    assert result["completion_ns"] == active.clock_ns
    assert result["service_ns"] == active.clock_ns - 1000.0
    assert result["logical_bytes"] == page.bytes
    assert result["bandwidth_ceiling_gb_s"] == 27.2


def test_unaligned_write_chunks_fit_rounded_buffer_budget():
    executor, page = _executor()
    result = executor.write_page(page, arrival_ns=0.0, offset_bytes=1, size_bytes=4096)
    assert result["logical_bytes"] == 4096
    assert result["host_transfer_bytes"] == 8192
    assert result["write_buffer_peak_bytes"] <= 4096
