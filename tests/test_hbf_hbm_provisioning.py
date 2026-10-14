from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hbf_hbm_provisioning import build_package_scenario, budget_feasibility
from hbf_hbm_economics import conditional_node_cost
from hbf_hbm_experiments import serving_row
from heterollm_sim.memory_types import parse_physical_memory_config


class ProvisioningTests(unittest.TestCase):
    def test_package_inventory_drives_physical_and_analytical_budgets(self):
        s=build_package_scenario(2,2,'D_hbf_weights',model_id='small')
        hbm=s.hardware.get_component('hbm0'); hbf=s.hardware.get_component('hbf0')
        physical=parse_physical_memory_config(hbf.metadata['physical_memory_config'])
        self.assertEqual(hbm.capacity_bytes,72_000_000_000)
        self.assertEqual(hbf.capacity_bytes,1_024_000_000_000)
        self.assertEqual(physical.parallel_units,128)
        self.assertEqual(physical.host_read_bandwidth_gb_s,3200.)
        self.assertEqual(s.resolve_component_profile('hbf0').read_bandwidth_gb_s,3200.)
        self.assertEqual(s.resolve_component_profile('hbm0').read_bandwidth_gb_s,3200.)
        self.assertEqual(parse_physical_memory_config(hbm.metadata['physical_memory_config']).max_outstanding_requests,
                         s.resolve_component_profile('hbm0').max_outstanding_requests)
        baseline=build_package_scenario(4,0,'A_hbm_only',model_id='small')
        self.assertNotIn('hbf0',{c.component_id for c in baseline.hardware.components})
        self.assertTrue(all('hbf0' not in (l.source_component,l.target_component) for l in baseline.hardware.links))

    def test_independent_budget_limits_can_reject_different_candidates(self):
        hbm=budget_feasibility(conditional_node_cost(hbm_packages=4,hbf_packages=0))
        flash=budget_feasibility(conditional_node_cost(hbm_packages=0,hbf_packages=4))
        self.assertTrue(hbm['cost_ceiling']); self.assertTrue(hbm['capacity_ceiling'])
        self.assertTrue(flash['cost_ceiling']); self.assertTrue(flash['area_ceiling'])
        self.assertFalse(flash['capacity_ceiling'])
        expensive=budget_feasibility(conditional_node_cost(hbm_packages=4,hbf_packages=0,
            assumptions={'gpu_price_usd':10000.}))
        self.assertFalse(expensive['cost_ceiling'])
        self.assertTrue(expensive['capacity_ceiling'])

    def test_long_indexed_embedding_reads_do_not_enlarge_remote_table(self):
        s=build_package_scenario(2,2,'D_hbf_weights',model_id='small',output=4)
        row=serving_row(s,'test-long-gather',placement='D_hbf_weights')
        self.assertEqual(row['completed_requests'],1)
        self.assertGreater(row['resources']['storage_traffic']['logical_read_bytes'],524288)
        self.assertEqual(row['resolved_placement']['tensor_to_component']['embedding_weights'],'hbf0')


if __name__=='__main__': unittest.main()
