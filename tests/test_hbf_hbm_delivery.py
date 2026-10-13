import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from hbf_hbm_experiments import completion_matrix,extended_model_matrix_complete,decode_matrix_complete,memory_core_coverage


class DeliveryCoverageTests(unittest.TestCase):
    def groups(self):
        original={'id':'09_serving_study','status':'passed','coverage':{'sections':[
            'placement','models','named_extended','long_context','baselines','slo','tail_sample','prefetch','speculative']}}
        rows=[]
        for c in (2048,32768):
            for a in ('A_hbm_only','B_all_kv_hbf','D_hbf_weights'):
                for m in ('qwen3-coder-30b-a3b','synthetic-1t-gqa-moe'):
                    rows.append({'mode':'full_operator_serving','parameters':{'model_id':m},
                        'prompt_tokens':c,'placement':a,'completed_requests':1})
                for m in ('deepseek-v3','deepseek-v4-flash-0731','llama4-scout','llama4-maverick'):
                    rows.append({'mode':'structural_physical_probe','model_id':m,'context':c,'placement':a,'node_tps':None})
        return [original,{'id':'17_extended_models','status':'passed','sections':['full','probes'],'rows':rows}]

    def test_probe_only_success_does_not_close_model_work(self):
        groups=self.groups()
        self.assertTrue(extended_model_matrix_complete(groups))
        groups[1]['sections']=['probes']
        self.assertFalse(extended_model_matrix_complete(groups))
        item=next(x for x in completion_matrix(groups) if 17 in x['groups'])
        self.assertEqual(item['status'],'本次未完成')

    def test_declared_sections_cannot_hide_missing_execution_or_base_groups(self):
        groups=self.groups()
        groups[1]['rows'].pop(0)
        self.assertFalse(extended_model_matrix_complete(groups))
        baseline=next(x for x in completion_matrix(groups) if x['groups']==list(range(1,9)))
        self.assertEqual(baseline['status'],'本次未完成')

    def test_successful_decode_subset_cannot_close_full_matrix(self):
        rows=[{'parameters':{'model_id':m},'prompt_tokens':c,'placement':a,
               'visible_output_tokens':64,'decode_measurement':{'decode_tokens':63},'decode_core_checks':{'actual':True}}
              for m in ('llama3_1-8b','llama3_1-70b') for c in (2048,32768,131072,1048576)
              for a in ('A_hbm_only','B_all_kv_hbf')]
        group={'id':'19_decode_physical','status':'passed','rows':rows}
        self.assertTrue(decode_matrix_complete([group]))
        rows.pop()
        self.assertFalse(decode_matrix_complete([group]))
        self.assertEqual(next(x for x in completion_matrix([group]) if x['groups']==[19])['status'],'本次未完成')

    def test_host_dram_and_invalid_audit_do_not_prove_hbm_execution(self):
        audit={'valid_for_committed_byte_totals':True,'owners':[
            {'memory_kind':'DDR','core':'DramCore','transactions':12}]}
        row={'id':'case','mode':'full_operator_serving','core_execution_audit':audit}
        groups=[{'rows':[row]}]
        self.assertFalse(memory_core_coverage(groups)['all_full_serving_hbm_verified'])
        audit['owners'].append({'memory_kind':'HBM','core':'DramCore','transactions':12})
        self.assertTrue(memory_core_coverage(groups)['all_full_serving_hbm_verified'])
        audit['valid_for_committed_byte_totals']=False
        self.assertFalse(memory_core_coverage(groups)['all_full_serving_hbm_verified'])


if __name__=='__main__':unittest.main()
