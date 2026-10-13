"""Tests for the paper-parameterized HBF physical profiles."""
import json
import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrated_hbf_profiles.py"
SPEC = importlib.util.spec_from_file_location("calibrated_hbf_profiles", SCRIPT)
profiles = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profiles
SPEC.loader.exec_module(profiles)


class CalibratedHbfProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = profiles.build_report()

    def test_public_profile_parameters_are_explicit(self):
        rows = {item.profile_id: item for item in profiles._profile_specs()}
        self.assertEqual(set(rows), {"son_hbf", "son_hbf_plus", "hbf_sucks_hbf2_b200", "hot_cold_b200"})
        self.assertEqual(rows["son_hbf"].granularity["page_bytes"], 4096)
        self.assertEqual(rows["son_hbf"].latency["hbf_read_us"], 1.0)
        self.assertEqual(rows["son_hbf"].latency["hbf_write_us"], 100.0)
        self.assertEqual(rows["hbf_sucks_hbf2_b200"].granularity["write_page_bytes"], 4096)
        self.assertEqual(rows["hot_cold_b200"].granularity["kv_block_bytes"], 1_572_864)
        self.assertIsNone(rows["hot_cold_b200"].granularity["physical_page_bytes"])
        self.assertIsNone(rows["hbf_sucks_hbf2_b200"].slo["threshold_s"])

    def test_profile_runner_emits_paper_separated_results(self):
        report = self.report
        self.assertEqual(report["schema_version"], 1)
        fixed = [item for item in report["results"] if "hbm_only" in item]
        self.assertEqual({item["profile_id"] for item in fixed}, {
            "son_hbf", "son_hbf_plus", "hbf_sucks_hbf2_b200", "hot_cold_b200",
        })
        son = next(item for item in fixed if item["profile_id"] == "son_hbf")
        self.assertGreater(son["all_hbf"]["service_ns"], son["hbm_only"]["service_ns"])
        self.assertTrue(son["full_backend"]["full_backend_used"])
        self.assertGreater(
            son["full_backend"]["summary"]["throughput_visible_output_tokens_per_s"],
            0.0,
        )
        self.assertTrue(son["full_backend"]["hbm_baseline"]["full_backend_used"])
        self.assertEqual(
            son["full_backend"]["hbm_baseline"]["resources"]["storage_traffic"].get("task_count", 0),
            0,
        )
        self.assertGreater(
            son["full_backend"]["resources"]["storage_traffic"]["task_count"],
            0,
        )
        self.assertLess(
            son["full_backend"]["comparison"]["hbf_vs_hbm_visible_tps_ratio"],
            1.0,
        )
        plus = next(item for item in fixed if item["profile_id"] == "son_hbf_plus")
        self.assertEqual(plus["hbm_only"]["baseline_label"], "son_hbm4_reference_not_same_profile")
        hbf_sucks = next(item for item in fixed if item["profile_id"] == "hbf_sucks_hbf2_b200")
        self.assertEqual(hbf_sucks["profile_id"], "hbf_sucks_hbf2_b200")
        self.assertIsNone(hbf_sucks["all_hbf"]["memory_service_slo_met"])
        sweep = next(item for item in report["results"] if item.get("experiment") == "media_latency_sweep")
        self.assertEqual(len(sweep["latency_sweep"]), 4)
        self.assertEqual(sweep["latency_sweep"][0]["read_us"], 8)
        self.assertGreater(sweep["latency_sweep"][-1]["all_hbf_service_us"], sweep["latency_sweep"][0]["all_hbf_service_us"])

    def test_hot_cold_block_fetch_is_same_unit_diagnostic_only(self):
        report = self.report
        row = next(item for item in report["results"] if item.get("experiment") == "resume_block_fetch")
        block48 = next(item for item in row["block_fetch"] if item["blocks"] == 48)
        self.assertGreater(block48["resume_service_us"], 0.0)
        self.assertLess(block48["resume_service_us"], 200.0)
        self.assertEqual(row["paper_resume_overhead_us"], 84)
        self.assertIn("missing-block", row["comparability"])

    def test_output_is_json_safe(self):
        report = self.report
        json.dumps(report, allow_nan=False)
        self.assertTrue(all(item["comparability"]["full_paper_tps_comparable"] is False
                            for item in report["results"] if "hbm_only" in item))

    def test_full_backend_probe_uses_gpu_and_serving_models(self):
        profile = profiles._profile_by_id("son_hbf")
        result = profiles._full_backend_summary(
            profile, request_count=1, prompt_tokens=32, output_tokens=4
        )
        self.assertTrue(result["full_backend_used"])
        self.assertEqual(result["memory_placement"], "hbf_direct_kv")
        self.assertEqual(result["execution_mode"], "continuous_batching")
        self.assertEqual(
            result["backend_model"],
            "TopologyAwareBatchCostProvider + simulate_online",
        )
        self.assertGreater(
            result["summary"]["throughput_visible_output_tokens_per_s"], 0.0
        )
        self.assertGreater(
            result["resources"]["storage_traffic"].get("task_count", 0), 0
        )
        self.assertIn("gpu0.tensor_core", result["resources"]["gpu_resource_utilization"])
        breakdown = result["resources"]["resource_breakdown"]
        self.assertGreater(breakdown["gpu"]["max_utilization"], 0.0)
        self.assertGreater(breakdown["cpu"]["max_utilization"], 0.0)
        self.assertEqual(breakdown["cim"]["resource_count"], 0)
        self.assertEqual(result["backend_policy"]["cim_mode"], "disabled")
        self.assertEqual(result["gpu_model"]["source"], "bundled reference GPU profile")
        self.assertIn("bottleneck_resource", result["summary"])

    def test_backend_policy_parameters_change_only_their_intended_path(self):
        profile = profiles._profile_by_id("son_hbf")
        baseline = profiles._full_backend_summary(
            profile, use_hbf=False, request_count=1, prompt_tokens=32, output_tokens=4
        )
        slower_gpu = profiles._full_backend_summary(
            profile, use_hbf=False, request_count=1, prompt_tokens=32, output_tokens=4,
            gpu_compute_pflops=0.06,
        )
        slower_host = profiles._full_backend_summary(
            profile, use_hbf=False, request_count=1, prompt_tokens=32, output_tokens=4,
            host_control_scale=100.0,
        )
        slow_link = profiles._full_backend_summary(
            profile, use_hbf=True, request_count=1, prompt_tokens=32, output_tokens=4,
            interconnect_bandwidth_gbps=8.0,
        )
        self.assertNotEqual(
            baseline["summary"]["makespan_ns"], slower_gpu["summary"]["makespan_ns"]
        )
        self.assertEqual(
            baseline["resources"]["dram_traffic"]["logical_read_bytes"],
            slower_gpu["resources"]["dram_traffic"]["logical_read_bytes"],
        )
        self.assertGreater(
            slower_host["summary"]["makespan_ns"], baseline["summary"]["makespan_ns"]
        )
        self.assertGreater(
            slow_link["summary"]["makespan_ns"],
            profiles._full_backend_summary(
                profile, use_hbf=True, request_count=1, prompt_tokens=32, output_tokens=4
            )["summary"]["makespan_ns"],
        )

    def test_paper_compute_is_applied_only_when_published(self):
        son = profiles._full_backend_summary(
            profiles._profile_by_id("son_hbf"), request_count=1, prompt_tokens=32, output_tokens=4
        )
        hot_cold = profiles._full_backend_summary(
            profiles._profile_by_id("hot_cold_b200"), request_count=1, prompt_tokens=32, output_tokens=4
        )
        self.assertFalse(son["paper_compute_applied"])
        self.assertTrue(hot_cold["paper_compute_applied"])

    def test_hbm_full_backend_defaults_to_aggregate_service_mode(self):
        profile = profiles._profile_by_id("son_hbf")
        aggregate = profiles._full_backend_summary(
            profile, use_hbf=False, request_count=1, prompt_tokens=32, output_tokens=4
        )
        physical = profiles._full_backend_summary(
            profile, use_hbf=False, request_count=1, prompt_tokens=32, output_tokens=4,
            hbm_execution_mode="physical",
        )
        self.assertEqual(aggregate["hbm_execution_mode"], "aggregate")
        self.assertGreater(aggregate["summary"]["throughput_visible_output_tokens_per_s"], 0.0)
        self.assertGreater(
            physical["resources"]["dram_traffic"]["task_count"],
            aggregate["resources"]["dram_traffic"]["task_count"],
        )


if __name__ == "__main__":
    unittest.main()
