"""Full serving checks for real ablations and fixed-budget routed KV copies."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from hbf_hbm_ablation_paths import build_path_scenario, run_ablation_case, run_path_case


class FullServingAblationTests(unittest.TestCase):
    def test_ideal_media_match_and_single_factors_execute(self):
        parameters=dict(batch=1,prompt=32,output=3)
        hbm=run_ablation_case("hbm","ideal",**parameters)
        hbf=run_ablation_case("hbf","ideal",**parameters)
        self.assertTrue(math.isclose(hbm["makespan_ns"],hbf["makespan_ns"],rel_tol=1e-7))
        self.assertEqual(hbm["visible_output_tokens"],hbf["visible_output_tokens"])
        for factor in ("read_latency","write_restriction"):
            changed=run_ablation_case("hbf",factor,**parameters)
            self.assertEqual(changed["completed_requests"],1)
            self.assertGreater(changed["makespan_ns"],hbf["makespan_ns"])
        finite=run_ablation_case("hbf","finite_concurrency",**parameters)
        # Finite queues do not create an invented latency at zero media latency.
        self.assertTrue(math.isclose(finite["makespan_ns"],hbf["makespan_ns"],rel_tol=1e-7))

    def test_unique_physical_link_budget_and_partition_inventory(self):
        inventories=[]
        for mode,share in (("direct",.5),("relay",.5),("dual",.25),("dual",.75)):
            scenario,contract=build_path_scenario(mode,weight_share=share)
            physical={}
            for link in scenario.hardware.links:
                if link.link_id.startswith("path-"):
                    ident=link.metadata["bandwidth_resource_id"]
                    if ident in physical:
                        self.assertEqual(physical[ident],link.bandwidth_gbps)
                    physical[ident]=link.bandwidth_gbps
            self.assertAlmostEqual(sum(physical.values())/8,64.)
            self.assertAlmostEqual(contract["applied_sum_link_gb_s"],64.)
            inventories.append([(c.component_id,c.capacity_bytes,c.read_bandwidth_gbps,c.write_bandwidth_gbps)
                for c in scenario.hardware.components if c.component_id in ("hbf0","hbf_weights","relaymem")])
        self.assertTrue(all(inventory==inventories[0] for inventory in inventories))

    def test_full_serving_relay_pays_memory_and_dual_executes_both_paths(self):
        parameters=dict(prompt=32,output=12,interval_ns=500_000.)
        direct=run_path_case("direct",**parameters)
        dual=run_path_case("dual",**parameters)
        for row in (direct,dual):
            self.assertEqual(row["completed_requests"],2)
            self.assertGreater(row["path_observations"]["gpu_operator_stage_count"],0)
            self.assertGreater(row["path_observations"]["migration_events"],0)
            self.assertEqual(row["prefetch_overlap"]["premature_consumer_overlap_ns"],0)
        self.assertEqual(direct["path_observations"]["relay_write_bytes"],0)
        observed=dual["path_observations"]
        self.assertGreater(observed["relay_write_bytes"],0)
        self.assertEqual(observed["relay_read_bytes"],observed["relay_write_bytes"])
        self.assertGreater(observed["executed_path_resources"]["link.path-direct"],0)
        for link in ("path-relay-in","path-relay-out"):
            self.assertGreater(observed["migration_link_bytes"][link],0)
        total=sum(event["bytes"] for event in observed["physical_transfers"])
        self.assertEqual(observed["relay_write_bytes"],total)
        self.assertEqual(sum(observed["migration_link_bytes"].values()),2*total)
        # Physical dependencies preserve read->link->relay write/read->link->write.
        for event in observed["physical_transfers"]:
            phases=event["phases"]
            previous=event["start_ns"]
            for phase in phases:
                start=phase.get("start_ns",phase.get("arrival_ns"))
                self.assertGreaterEqual(start+1e-6,previous)
                previous=phase["completion_ns"]
            self.assertAlmostEqual(previous,event["completion_ns"])


if __name__=="__main__":
    unittest.main()
