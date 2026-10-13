"""Full backend checks: labels must correspond to real placement and traffic."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
import hbf_hbm_serving_study as study


class ServingStudyTests(unittest.TestCase):
    def test_named_architecture_and_context_extension(self):
        eight=study.build_study_scenario('A_hbm_only',model_id='llama3_1-8b',prompt=2048)
        seventy=study.build_study_scenario('B_all_kv_hbf',model_id='llama3_1-70b',prompt=1048576)
        self.assertEqual(eight.model.num_layers,32)
        self.assertEqual(seventy.model.num_layers,80)
        self.assertGreater(seventy.model.total_declared_weight_bytes,100*study.GIB)
        self.assertEqual(seventy.placement.kv_policy.dtype,'bf16')
        self.assertEqual(seventy.model.metadata['study_context_extrapolation']['native_max_tokens'],131072)
        self.assertFalse(seventy.model.metadata['study_context_extrapolation']['accuracy_claim'])

    def test_c_uses_real_physical_swap_and_resume_in_full_serving(self):
        row=study.run_case('C_hot_cold',row_id='test/C',prompt=32,output=4,batch=1,
            request_count=2,interval_ns=10000.,priority_preemption=True)
        self.assertEqual(row['completed_requests'],2)
        self.assertGreater(row['kv']['offload_bytes'],0)
        self.assertGreater(row['kv']['peak_bytes_by_component']['hbf0'],0)
        self.assertEqual(row['kv']['offload_bytes'],row['kv']['swap_in_bytes'])
        self.assertEqual(row['kv']['recompute_events'],0)
        directions={(e['source_component'],e['target_component']) for e in row['migration_ledger']['events']}
        self.assertEqual(directions,{('hbm0','hbf0'),('hbf0','hbm0')})
        self.assertTrue(row['placement_verification']['resolved_kv_on_requested_tier'])
        self.assertTrue(row['placement_verification']['all_resolved_model_ops_on_gpu'])

    def test_d_executes_hbf_weight_reads_with_hbm_active_kv(self):
        row=study.run_case('D_hbf_weights',row_id='test/D',prompt=32,output=4)
        self.assertEqual(row['completed_requests'],1)
        storage=row['resources']['storage_traffic']
        self.assertGreater(storage['logical_read_bytes'],0)
        self.assertEqual(storage['logical_write_bytes'],0)
        self.assertGreater(row['kv']['logical_decode_write_bytes'],0)
        self.assertEqual(set(row['kv']['peak_bytes_by_component']),{'hbm0'})
        self.assertTrue(row['placement_verification']['all_resolved_weights_on_requested_tier'])
        self.assertTrue(row['placement_verification']['all_resolved_model_ops_on_gpu'])

    def test_nvme_baseline_is_storage_not_relabelled_hbf(self):
        scenario=study.build_study_scenario('F_nvme',prompt=32,output=4)
        self.assertEqual(scenario.hardware.get_component('hbf0').kind,'ssd')
        row=study.run_case('F_nvme',row_id='test/F',prompt=32,output=4,batch=1,
            request_count=2,interval_ns=10000.,priority_preemption=True)
        self.assertEqual(row['completed_requests'],2)
        self.assertGreater(row['kv']['swap_in_bytes'],0)
        self.assertGreater(row['node_tps'],0)

    def test_extended_presets_keep_real_dense_and_moe_geometry(self):
        dense=study.build_study_scenario('A_hbm_only',model_id='llama3_1-405b',
            prompt=2048,hbm_capacity_bytes=1024*study.GIB)
        moe=study.build_study_scenario('D_hbf_weights',model_id='qwen3-30b-a3b',prompt=2048)
        self.assertEqual(dense.model.num_layers,126)
        self.assertGreater(dense.model.total_declared_weight_bytes,512*study.GIB)
        self.assertEqual(dense.hardware.get_component('hbm0').capacity_bytes,1024*study.GIB)
        self.assertEqual(moe.model.num_layers,48)
        layers=study.model_graph_execution_layers(moe.model.graph)
        self.assertTrue(all(layer.is_moe for layer in layers))
        self.assertTrue(all(layer.num_experts>layer.experts_per_token>1 for layer in layers))

    def test_speculative_full_serving_observes_acceptance_and_rejection(self):
        report=study.run_serving_study(sections=('speculative',),progress=False)
        self.assertEqual(report['status'],'passed')
        self.assertEqual(len(report['rows']),8)
        for row in report['rows']:
            self.assertEqual(row['completed_requests'],4)
            self.assertEqual(row['model_structure']['mtp_descriptor_count'],2)
            events=row['speculative']
            probability=row['parameters']['speculative_acceptance']
            if probability is None:
                self.assertEqual(events['drafted_tokens'],0)
            else:
                self.assertGreater(events['drafted_tokens'],0)
                self.assertTrue(events['rounds'])
                self.assertEqual(events['drafted_tokens'],events['accepted_draft_tokens']+events['rejected_draft_tokens'])
                self.assertGreater(events['rejected_draft_tokens'],0)
                self.assertEqual(events['accepted_draft_tokens']==0,probability==0.)

    def test_parallel_points_equal_serial_and_resume_only_matching_cache(self):
        cases=(("A_hbm_only","parallel/A",dict(prompt=32,output=4)),
               ("B_all_kv_hbf","parallel/B",dict(prompt=32,output=4)))
        def measured(rows):
            # JSON normalizes tuple/list representation after cache loading;
            # every reported value except real execution wall time is compared.
            return json.loads(json.dumps([{k:v for k,v in row.items() if k!='wall_seconds'}
                                          for row in rows]))
        serial=study.study_cases(cases,workers=1,progress=False)
        with tempfile.TemporaryDirectory(prefix='hbf-study-parallel-') as directory:
            parallel=study.study_cases(cases,workers=2,point_cache_dir=directory,progress=False)
            self.assertEqual([r['id'] for r in parallel],['parallel/A','parallel/B'])
            self.assertEqual(measured(serial),measured(parallel))
            with mock.patch.object(study,'run_case',side_effect=AssertionError('matching cache must resume')), \
                 mock.patch.object(study,'ProcessPoolExecutor',side_effect=AssertionError('no workers needed')):
                resumed=study.study_cases(cases,workers=2,point_cache_dir=directory,resume=True,progress=False)
            self.assertEqual(measured(parallel),measured(resumed))
            first=Path(directory)/'parallel__A.json'
            second=Path(directory)/'parallel__B.json'
            cached=json.loads(first.read_text(encoding='utf-8'))
            cached['parameters']['prompt']=99
            first.write_text(json.dumps(cached),encoding='utf-8')
            cached=json.loads(second.read_text(encoding='utf-8'))
            cached['point_result_version']=study.POINT_RESULT_VERSION-1
            second.write_text(json.dumps(cached),encoding='utf-8')
            with mock.patch.object(study,'run_case',wraps=study.run_case) as execute:
                refreshed=study.study_cases(cases,workers=1,point_cache_dir=directory,resume=True,progress=False)
            self.assertEqual(execute.call_count,2)
            self.assertEqual(measured(serial),measured(refreshed))

    def test_parallel_failure_retains_successful_points_and_raises(self):
        cases=(("A_hbm_only","failure/A",dict(prompt=32,output=4)),
               ("invalid_architecture","failure/bad",dict(prompt=32,output=4)),
               ("B_all_kv_hbf","failure/B",dict(prompt=32,output=4)))
        with tempfile.TemporaryDirectory(prefix='hbf-study-failure-') as directory:
            with self.assertRaises(study.StudyCasesError) as captured:
                study.study_cases(cases,workers=2,point_cache_dir=directory,progress=False)
            self.assertIn('failure/bad',str(captured.exception))
            self.assertEqual([r['id'] for r in captured.exception.completed_rows],['failure/A','failure/B'])
            self.assertFalse((Path(directory)/'failure__bad.json').exists())
            for name in ('A','B'):
                cached=json.loads((Path(directory)/f'failure__{name}.json').read_text(encoding='utf-8'))
                self.assertTrue(cached['executed_serving'])
                self.assertEqual(cached['completed_requests'],1)


if __name__=='__main__':
    unittest.main()
