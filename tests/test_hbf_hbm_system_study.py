from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from hbf_hbm_system_study import build_multi_gpu, run_multi_case
from heterollm_sim.planner import _direct_device_memory


class MultiGPUStudyTests(unittest.TestCase):
    def test_tp_runs_every_rank_with_collectives_and_rank_local_lm_head(self):
        row=run_multi_case('A_hbm_only',2,prompt=32,output=2)
        self.assertEqual(row['completed_requests'],4)
        execution=row['physical_execution']
        self.assertEqual(set(execution['gpu_compute_resources']),{'gpu0.tensor_core','gpu1.tensor_core'})
        self.assertTrue(all(v>0 for v in execution['gpu_compute_resources'].values()))
        self.assertTrue(any(v>0 for v in execution['peer_link_resources'].values()))
        self.assertEqual({x['component_id'] for x in execution['operator_execution_targets']['lm_head']},{'gpu0','gpu1'})
        self.assertEqual(execution['execution_schedule_modes'],['resource_dag'])
        self.assertAlmostEqual(row['per_gpu_tps']*2,row['node_tps'])

    def test_shared_direct_exposure_requires_explicit_permission_and_executes(self):
        s=build_multi_gpu('D_hbf_weights',2,prompt=32,output=2)
        self.assertTrue(_direct_device_memory(s,'hbf0','gpu1'))
        self.assertFalse(_direct_device_memory(s,'hbf0','gpu9'))
        row=run_multi_case('D_hbf_weights',2,prompt=32,output=2)
        self.assertEqual(row['completed_requests'],4)
        self.assertGreater(row['resources']['storage_traffic']['logical_read_bytes'],0)
        self.assertEqual(row['resources']['storage_traffic']['logical_write_bytes'],0)

    def test_slow_rack_collectives_change_measured_service(self):
        local=run_multi_case('G_cxl_dram',2,prompt=32,output=2)
        rack=run_multi_case('G_cxl_dram',2,prompt=32,output=2,rack=True)
        self.assertGreater(rack['makespan_ns'],local['makespan_ns'])
        self.assertEqual(rack['completed_requests'],local['completed_requests'])


if __name__=='__main__': unittest.main()
