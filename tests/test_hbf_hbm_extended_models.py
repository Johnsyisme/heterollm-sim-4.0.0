"""Architecture conservation checks for the explicit non-native probes."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from hbf_hbm_extended_models import structural_workload, run_structural_probe
from heterollm_sim.config import model_from_dict
from heterollm_sim.model_presets import (EXTENDED_MODEL_SPECS, UnsupportedPresetError,
    materialize_model_payload, get_model_preset, model_preset_metadata)


class ExtendedModelTests(unittest.TestCase):
    def test_unsupported_native_models_cannot_silently_materialize_gqa(self):
        for model in EXTENDED_MODEL_SPECS:
            with self.assertRaises(UnsupportedPresetError):
                materialize_model_payload(model)
        synthetic=model_from_dict(materialize_model_payload("synthetic-1t-gqa-moe"))
        self.assertGreater(synthetic.total_declared_weight_bytes,2e12)
        self.assertEqual(synthetic.num_layers,40)
        metadata=model_preset_metadata(get_model_preset("synthetic-1t-gqa-moe"))
        self.assertEqual(metadata["provenance_status"],"explicit_synthetic_design")
        self.assertEqual(metadata["architecture_evidence"]["source_url"],"")

    def test_v3_latent_cache_and_absorbed_contractions(self):
        w=structural_workload("deepseek-v3",2048)
        # Latent rank and shared RoPE, not 128 pairs of full-head K/V.
        self.assertEqual(w["totals"]["cache_resident_bytes"],61*2048*(512+64)*2)
        self.assertEqual(w["totals"]["attention_and_compression_flops"],61*2*128*2048*(2*512+64))
        self.assertEqual(sum(l["moe"] for l in w["layers"]),58)
        self.assertEqual(w["totals"]["matrix_resident_parameters"],671025397760)

    def test_v4_compressed_shared_storage_and_index_selection(self):
        w=structural_workload("deepseek-v4-flash-0731",32768)
        kinds=[l["attention"] for l in w["layers"]]
        self.assertEqual(kinds.count("sliding_shared_kv"),2)
        self.assertEqual(kinds.count("compressed_sparse_attention"),21)
        self.assertEqual(kinds.count("heavily_compressed_attention"),20)
        c={v["name"]:v for v in w["layers"][2]["caches"]}
        self.assertEqual(c["shared_kv_window"]["resident_bytes"],128*512*2)
        self.assertEqual(c["compressed_shared_kv"]["resident_bytes"],8192*512*2)
        self.assertEqual(c["compressed_shared_kv"]["read_bytes"],512*512*2)
        self.assertEqual(c["compressed_index_keys"]["read_bytes"],8192*128*2)
        self.assertEqual(c["compressor_previous_ca_overlap"]["resident_bytes"],4*512*2*2)
        self.assertEqual(c["index_previous_ca_overlap"]["resident_bytes"],4*128*2*2)
        between=structural_workload("deepseek-v4-flash-0731",32769)
        caches={v["name"]:v for v in between["layers"][2]["caches"]}
        self.assertEqual(caches["compressed_shared_kv"]["write_bytes"],0)
        self.assertEqual(caches["compressor_previous_ca_overlap"]["read_bytes"],0)

    def test_llama_chunk_boundary_resets_and_moe_interleaves(self):
        scout=structural_workload("llama4-scout",32768)
        after=structural_workload("llama4-scout",32769)
        self.assertEqual(scout["totals"]["cache_resident_bytes"],(36*8192+12*32768)*8*128*2*2)
        self.assertEqual(after["layers"][0]["caches"][0]["entries"],1)
        self.assertEqual(after["layers"][3]["caches"][0]["entries"],32769)
        maverick=structural_workload("llama4-maverick",2048)
        self.assertEqual(sum(l["moe"] for l in maverick["layers"]),24)
        self.assertTrue(maverick["spec"]["assumptions"])

    def test_v4_first_overlap_and_producer_consumer_order(self):
        first=structural_workload("deepseek-v4-flash-0731",4)
        c={v["name"]:v for v in first["layers"][2]["caches"]}
        self.assertEqual(c["compressor_previous_ca_overlap"]["read_bytes"],0)
        self.assertEqual(c["index_previous_ca_overlap"]["read_bytes"],0)
        row=run_structural_probe("deepseek-v4-flash-0731",8,"B_all_kv_hbf")
        events={t["name"]:t for t in row["transactions"] if t["name"].startswith("layer2.")}
        def before(left,right):
            self.assertLessEqual(events["layer2."+left]["completion_ns"],events["layer2."+right]["start_ns"])
        for branch,compressed in (("compressor","compressed_shared_kv"),("index","compressed_index_keys")):
            before(branch+"_source_window.read",compressed+".append")
            before(branch+"_previous_ca_overlap.read",compressed+".append")
            before(branch+"_previous_ca_overlap.read",branch+"_previous_ca_overlap.append")
            before(compressed+".append",compressed+".read")
        before("compressed_index_keys.read","compressed_shared_kv.read")
        before("compressed_shared_kv.read","grouped_output_down")
        before("grouped_output_up","routed_experts")

    def test_real_nand_probe_executes_bytes_without_claiming_tps(self):
        row=run_structural_probe("deepseek-v4-flash-0731",128,"B_all_kv_hbf")
        self.assertFalse(row["executed_serving"])
        self.assertIsNone(row["node_tps"])
        traffic=row["physical_traffic"]["hbf0"]
        self.assertGreater(traffic["physical_write_bytes"],0)
        self.assertGreater(traffic["physical_read_bytes"],0)
        totals=row["workload"]["totals"]
        self.assertEqual(sum(t["logical_bytes"] for t in row["transactions"]),
            totals["matrix_read_bytes"]+totals["cache_read_bytes"]+totals["cache_write_bytes"])
        previous=0.
        for t in row["transactions"]:
            self.assertGreaterEqual(t["start_ns"],previous)
            self.assertGreater(t["completion_ns"],t["start_ns"])
            previous=t["completion_ns"]


if __name__=="__main__":
    unittest.main()
