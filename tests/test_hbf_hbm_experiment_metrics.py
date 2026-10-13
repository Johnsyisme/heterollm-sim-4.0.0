"""Regression checks for experiment reporting and isolated capacity controls."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import hbf_hbm_experiments as experiments


class ExperimentMetricsTests(unittest.TestCase):
    def test_metrics_can_skip_ui_trace_without_changing_statistics(self):
        result=experiments.run_scenario(experiments.scenario(count=1,output=2))
        with patch('heterollm_sim.reporting._visualization_payload',side_effect=AssertionError('UI trace requested')):
            without=experiments.report_dict(result,include_visualization=False)
        full=experiments.report_dict(result,visualization_limit=1)
        for key in ('summary','kv_cache','storage_traffic','dram_traffic','scheduler'):
            self.assertEqual(without[key],full[key])
        self.assertEqual(without['visualization']['status'],'not_requested')

    def test_finished_requests_count_and_goodput_share_denominator(self):
        row=experiments.serving_row(experiments.scenario(hbf=True,count=2,output=4),"test")
        self.assertEqual(row["completed_requests"],2)
        self.assertEqual(row["unfinished_requests"],0)
        self.assertAlmostEqual(row["node_tps"],8e9/row["makespan_ns"])
        self.assertGreater(row["tpot_ns"]["p50"],0)
        self.assertEqual(row["tbt_ns"]["count"],6)
        self.assertLessEqual(row["goodput_tps"],row["node_tps"])

    def test_dedicated_kv_capacity_has_independent_physical_owner(self):
        scenario=experiments.scenario(kv_budget=65536,count=2,slots=2)
        original=scenario.hardware.get_component("hbm0")
        kv=scenario.hardware.get_component("hbm_kv")
        self.assertNotEqual(original.metadata["memory_service_owner"],kv.metadata["memory_service_owner"])
        row=experiments.serving_row(scenario,"isolated-kv")
        self.assertEqual(row["completed_requests"],2)
        self.assertLessEqual(row["kv"]["peak_used_bytes"],65536)

    def test_nonzero_initialization_is_in_request_window(self):
        row=experiments.serving_row(experiments.scenario(count=1,output=4),"window")
        windows=row['measurement_windows']
        self.assertGreater(windows['initialization_ns'],0)
        self.assertAlmostEqual(row['makespan_ns'],row['e2e_ns']['p50'])
        self.assertAlmostEqual(row['makespan_ns']-windows['runtime_window']['duration_ns'],
                               windows['initialization_ns'])
        self.assertLess(row['node_tps'],row['runtime_tps'])
        self.assertAlmostEqual(row['phase_service_rates']['prefill']['wall_clock_tokens_per_second'],
                               128e9/row['makespan_ns'])

    def test_four_requested_sequences_really_batch_together(self):
        row=experiments.serving_row(experiments.scenario(prompt=2048,count=4,output=4),"batch")
        self.assertEqual(row['max_observed_batch_requests'],4)
        self.assertEqual(row['completed_requests'],4)

    def test_hbf_directional_bandwidth_uses_named_window(self):
        row=experiments.serving_row(experiments.scenario(hbf=True,count=1,output=3),"memory")
        memory=row['memory_metrics']
        storage=row['resources']['storage_traffic']
        self.assertGreater(memory['page_access_count'],0)
        self.assertEqual(storage['read_operations'],storage['pages_read'])
        self.assertEqual(storage['program_operations'],storage['pages_programmed'])
        self.assertGreater(storage['rmw_read_operations'],0)
        for direction in ('read','write'):
            metric=memory['directional_bandwidth'][direction]
            self.assertGreater(metric['logical_bytes'],0)
            self.assertAlmostEqual(metric['effective_logical_gb_s'],metric['logical_bytes']/row['makespan_ns'])


if __name__=="__main__":
    unittest.main()
