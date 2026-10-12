"""Serving-level regression coverage for physical paged KV migration."""

from dataclasses import replace
import unittest

from heterollm_sim.ir import ComponentSpec, KVCachePolicy, LinkSpec, PortSpec
from heterollm_sim.memory_types import make_hbf_config
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serving import BatchCost, _OnlineRuntime, compile_serving_plan


class PagedPoolServingTests(unittest.TestCase):
    @staticmethod
    def _scenario_with_hbf():
        base = build_reference_scenario()
        capacity = 8 * 1024**3
        physical = make_hbf_config(
            capacity_bytes=capacity,
            channels=8,
            parallel_units=8,
            page_bytes=16 * 1024,
            host_bandwidth_gb_s=488.0,
            host_read_bandwidth_gb_s=488.0,
            host_write_bandwidth_gb_s=27.2,
        )
        hbf = ComponentSpec(
            component_id="hbf0",
            kind="hbf",
            ports=(
                PortSpec(
                    port_id="pcie0",
                    protocol="PCIe",
                    role="endpoint",
                    version="5.0",
                    lanes=16,
                    bandwidth_gbps=512.0,
                    payload="coherent_dma",
                ),
            ),
            capacity_bytes=capacity,
            read_bandwidth_gbps=488.0 * 8.0,
            write_bandwidth_gbps=27.2 * 8.0,
            metadata={
                "physical_memory_config": physical,
                "access_mode": "remote_flash",
            },
        )
        hardware = replace(
            base.hardware,
            components=base.hardware.components + (hbf,),
            links=base.hardware.links
            + (
                LinkSpec(
                    link_id="cpu-hbf-pcie",
                    source_component="cpu0",
                    source_port="pcie0",
                    target_component="hbf0",
                    target_port="pcie0",
                    protocol="PCIe",
                    version="5.0",
                    lanes=16,
                    bandwidth_gbps=512.0,
                    latency_ns=100.0,
                    payload="coherent_dma",
                ),
            ),
            require_connected=False,
        )
        policy = KVCachePolicy(
            cache_component="hbm0",
            offload_component="hbf0",
            tokens_per_page=16,
            dtype="int8",
            layout_mode="paged_pool",
            pool_components=("hbm0", "hbf0"),
            preemption_mode="swap",
        )
        return replace(
            base,
            hardware=hardware,
            placement=replace(
                base.placement,
                hardware_name=hardware.name,
                kv_policy=policy,
            ),
        )

    def test_paged_pool_offload_restore_uses_physical_executor(self):
        plan = compile_serving_plan(self._scenario_with_hbf())
        runtime = _OnlineRuntime(
            plan,
            lambda *_: BatchCost(0.0),
            apply_measurement_start_residency=False,
        )
        request = runtime.states[plan.requests[0].request_id]
        pool = runtime.ledger.paged_pool
        self.assertIsNotNone(pool)
        self.assertIsNotNone(pool.transfer_executor)

        self.assertTrue(runtime.ledger.resize(request, 1))
        page = pool.request_pages(request.spec.request_id)[0]
        self.assertEqual(page.owner_component, "hbm0")

        self.assertTrue(runtime.ledger.offload(request))
        runtime._sync_paged_pool_events()
        self.assertEqual(request.kv_pages, 0)
        self.assertEqual(request.swapped_pages, 1)
        self.assertEqual(page.owner_component, "hbf0")
        self.assertEqual(len(pool.transfer_executor.events), 1)

        self.assertTrue(runtime._resume_swap(request))
        self.assertEqual(request.kv_pages, 1)
        self.assertEqual(request.swapped_pages, 0)
        self.assertEqual(page.owner_component, "hbm0")
        self.assertEqual(len(pool.transfer_executor.events), 2)
        self.assertGreater(runtime.swap_transfer_time_ns, 0.0)


if __name__ == "__main__":
    unittest.main()
