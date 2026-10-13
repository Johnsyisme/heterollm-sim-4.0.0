import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from hbf_hbm_experiments import completion_matrix,extended_model_matrix_complete


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


if __name__=='__main__':unittest.main()
