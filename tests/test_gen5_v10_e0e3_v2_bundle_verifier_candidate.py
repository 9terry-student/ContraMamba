"""Strict CPU synthetic V2 schema and failure tests; no CUDA/real data."""
from __future__ import annotations

import copy
import unittest

from scripts import gen5_v10_e0e3_v2_bundle_verifier_candidate as v2
from scripts import gen5_v10_bounded_worker_export_candidate as worker
from scripts import gen5_v10_bounded_tensor_provenance_candidate as stage_a
from test_gen5_v10_bounded_tensor_provenance_candidate import TestStageACanonicalProvenance


class TestV2StrictCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    @staticmethod
    def row(i,*,candidate=False):
        coarse={p:[0.0]*127 for p in v2.REGION}
        fine={p:{cell:[0.0]*127 for cell in v2.FINE} for p in v2.PARENTS}
        # Deliberate small numerical discrepancy with authentic rel-max diff.
        parent=v2.PARENTS[0]
        coarse[parent][5]=1e-12*(i+1)
        fine[parent][v2.FINE[0]][5]=3e-11*(i+1) if candidate else 0.0
        diff=abs(coarse[parent][5]-fine[parent][v2.FINE[0]][5])
        rel=diff/abs(coarse[parent][5])
        return {
            'R':3.0, 'S':1.0,'E':2.0,'C':0.3,'Q':0.1,
            'pair_gap_C':[0.0]*127,'pair_gap_C_reconstruction_abs':0.0,
            'pair_gap_Q_reconstruction_abs':0.0,'pair_gap_fft_half_log_span_max':92.25,
            'region_pair_C_by_gap':coarse,'region_pair_gap_reconstruction_abs_max':0.0,
            'region_pair_gap_reconstruction_rel_max':0.0,'region_pair_fft_half_log_span_max':92.25,
            'fine_pair_C_by_gap':fine, 'fine_parent_reconstruction_abs_max':diff,
            'fine_parent_reconstruction_rel_max':rel,'fine_pair_fft_half_log_span_max':92.25,
        }

    def fixture(self):
        kw=TestStageACanonicalProvenance().fixture(b=32,w=8)
        manifest=stage_a.manifest_for_cpu_inputs(**kw)
        source=worker.SOURCE_CELLS
        orient=worker.FROZEN_ORIENTATIONS
        locus=worker.choose_bounded_locus(
            worker_id=1,batch_index=13,row_start=800,row_stop=832,
            row_ids=[f'ROW_{i}' for i in range(800,832)],source_cells=source,
            orientations=orient,selected_source=(6201,6201),selected_targets=((6201,6202),(6201,6203)))
        frozen=[self.row(i) for i in range(4)]
        candidate=[self.row(i,candidate=True) for i in range(4)]
        bundle={
            'schema_version':v2.SCHEMA,
            'candidate_base':v2.AUTH_BASE,
            'selector':locus,
            'runtime_claims':{'source_kind':'CPU_SYNTHETIC_ONLY','execution':'NO',
                  'model_revision':'NOT_AUTHENTICATED','checkpoint_sha256':'NOT_AUTHENTICATED',
                  'dev_order_sha256':'NOT_AUTHENTICATED',
                  'active_encoding_sha256':'NOT_AUTHENTICATED','code_head':v2.AUTH_BASE,
                  'physical_device':'CPU','cuda_runtime':None,'torch_version':'NOT_AUTHENTICATED',
                  'tf32':False},
            'input_capture':{'pre_frozen':copy.deepcopy(manifest),
                     'post_frozen':copy.deepcopy(manifest),'post_v10':copy.deepcopy(manifest)},
            'engines':{
                'frozen':{'engine_id':'FROZEN_FFT_V1','records':frozen,
                          'records_sha256':v2.digest(frozen),
                          'dispatch':{'mode':'FROZEN_WHOLE','batch_rows':32},
                          'guards':{'fine_parent_replay_claim':'PASS',
                                    'upstream_guard_coverage':'NOT_AUTHENTICATED'}},
                'v10':{'engine_id':'V10_HYBRID_CANDIDATE','records':candidate,
                       'records_sha256':v2.digest(candidate),
                       'dispatch':{'mode':'CHUNKWISE_HYBRID','batch_rows':32,
                                   'fast_chunks':1,'gemm_chunks':1,'frozen_tail_calls':0,
                                   'total_chunks':2,'max_half_span':92.25},
                       'guards':{'fine_parent_replay_claim':'PASS',
                                 'upstream_guard_coverage':'NOT_AUTHENTICATED'}},
            },
            'evidence_status':'CPU_SYNTHETIC_STATIC_ONLY_NOT_APPROVED',
        }
        return v2.seal_synthetic_bundle(bundle)

    @staticmethod
    def reseal(b):
        b.pop('bundle_sha256',None)
        b['engines']['frozen']['records_sha256']=v2.digest(b['engines']['frozen']['records'])
        b['engines']['v10']['records_sha256']=v2.digest(b['engines']['v10']['records'])
        return v2.seal_synthetic_bundle(b)

    def test_legacy_fail_never_promotes(self):
        r=v2.validate(self.fixture())
        self.assertEqual(r['LEGACY_FULL_RECORD'],'FAIL')
        self.assertEqual(r['V10_E2_BOUNDED_NUMERIC'],'PASS_PROVISIONAL')
        self.assertEqual(r['V10_E0_PROVENANCE'],'NOT_EVALUATED')
        self.assertEqual(r['V10_E1_FROZEN_GUARDS'],'NOT_EVALUATED')
        self.assertEqual(r['V10_E2_PAYLOAD'],'NOT_EVALUATED')
        self.assertEqual(r['V10_E3_DIAGNOSTICS'],'NOT_EVALUATED')
        self.assertEqual(r['V10_PAYLOAD_EQUIVALENCE'],'NOT_EVALUATED')
        self.assertEqual(r['comparison']['diagnostic_differences'],4)
        self.assertEqual(r['derived_H']['fail'],0)

    def test_source_identity_forgery_blocks(self):
        b=self.fixture();b['selector']['source_cell']='6202_6202'
        with self.assertRaisesRegex(v2.V2Blocked,'LOCUS_SOURCE'):
            v2.validate(self.reseal(b))

    def test_target_order_forgery_blocks(self):
        b=self.fixture();b['selector']['target_cells'].reverse()
        with self.assertRaisesRegex(v2.V2Blocked,'LOCUS_TARGET_ORDER'):
            v2.validate(self.reseal(b))

    def test_bad_full_bundle_digest_blocks(self):
        b=self.fixture();b['selector']['row_ids'][0]='TAMPER'
        with self.assertRaisesRegex(v2.V2Blocked,'BUNDLE_HASH_FORGERY'):
            v2.validate(b)

    def test_row_order_identity_blocks_duplicate(self):
        b=self.fixture();b['selector']['row_ids'][0]=b['selector']['row_ids'][1]
        with self.assertRaisesRegex(v2.V2Blocked,'LOCUS_ROW_IDS'):
            v2.validate(self.reseal(b))

    def test_wrong_batch_blocks(self):
        b=self.fixture();b['selector']['batch_index_one_based']=14
        with self.assertRaisesRegex(v2.V2Blocked,'LOCUS_GEOMETRY'):
            v2.validate(self.reseal(b))

    def test_missing_fine_cell_blocks(self):
        b=self.fixture();b['engines']['v10']['records'][0]['fine_pair_C_by_gap'][v2.PARENTS[0]].pop(v2.FINE[-1])
        with self.assertRaisesRegex(v2.V2Blocked,'FINE_CELLS'):
            v2.validate(self.reseal(b))

    def test_injected_row_field_blocks(self):
        b=self.fixture();b['engines']['v10']['records'][0]['bogus']=0
        with self.assertRaisesRegex(v2.V2Blocked,'RECORD_FIELDS'):
            v2.validate(self.reseal(b))

    def test_forged_diagnostic_blocks(self):
        b=self.fixture();b['engines']['v10']['records'][0]['fine_parent_reconstruction_rel_max']=0.0
        with self.assertRaisesRegex(v2.V2Blocked,'FORGED_REL_DIAGNOSTIC'):
            v2.validate(self.reseal(b))

    def test_parent_replay_failure_blocks(self):
        b=self.fixture()
        b['engines']['v10']['records'][0]['fine_pair_C_by_gap'][v2.PARENTS[0]][v2.FINE[0]][5]=0.05
        with self.assertRaisesRegex(v2.V2Blocked,'FROZEN_PARENT_REPLAY'):
            v2.validate(self.reseal(b))

    def test_payload_mismatch_returns_fail(self):
        b=self.fixture();b['engines']['v10']['records'][0]['R']=5.0
        r=v2.validate(self.reseal(b))
        self.assertEqual(r['V10_E2_PAYLOAD'],'FAIL')
        self.assertEqual(r['V10_PAYLOAD_EQUIVALENCE'],'FAIL')
        self.assertEqual(r['LEGACY_FULL_RECORD'],'FAIL')

    def test_nan_rejected(self):
        b=self.fixture();b['engines']['v10']['records'][0]['E']=float('nan')
        with self.assertRaises(v2.V2Blocked):v2.validate(self.reseal(b))

    def test_manifest_drift_blocks(self):
        b=self.fixture()
        b['input_capture']['post_v10']=copy.deepcopy(b['input_capture']['post_v10'])
        b['input_capture']['post_v10']['tensors'][0]['content_sha256']='0'*64
        with self.assertRaises(Exception):v2.validate(self.reseal(b))

    def test_engine_swap_blocks(self):
        b=self.fixture();b['engines']['v10']['engine_id']='FROZEN_FFT_V1'
        with self.assertRaisesRegex(v2.V2Blocked,'ENGINE_ID'):
            v2.validate(self.reseal(b))

    def test_missing_upstream_guard_blocks(self):
        b=self.fixture();b['engines']['frozen']['guards'].pop('upstream_guard_coverage')
        with self.assertRaisesRegex(v2.V2Blocked,'GUARD_FIELDS'):
            v2.validate(self.reseal(b))

    def test_false_gpu_provenance_claim_blocks(self):
        b=self.fixture();b['runtime_claims']['physical_device']='T4'
        with self.assertRaisesRegex(v2.V2Blocked,'RUNTIME_NOT_STATIC'):
            v2.validate(self.reseal(b))

    def test_production_geometry_claim_blocks(self):
        b=self.fixture()
        b['input_capture']['pre_frozen']['geometry']['synthetic_cpu_geometry']=False
        with self.assertRaises(Exception):v2.validate(self.reseal(b))

    def test_dispatch_chunk_coverage_blocks(self):
        b=self.fixture();b['engines']['v10']['dispatch']['gemm_chunks']=0
        with self.assertRaisesRegex(v2.V2Blocked,'ROUTE_COVERAGE'):
            v2.validate(self.reseal(b))

    def test_record_hash_blocks_without_bundle_rehash(self):
        b=self.fixture()
        b['engines']['v10']['records'][0]['R']=5.0
        b['bundle_sha256']=v2.digest({k:v for k,v in b.items() if k!='bundle_sha256'})
        with self.assertRaisesRegex(v2.V2Blocked,'RECORD_HASH'):
            v2.validate(b)

    def test_unknown_schema_blocks(self):
        b=self.fixture();b['schema_version']='GEN5_SHARED_GEMM_CPU_EQUIVALENCE_DRAFT_V1'
        with self.assertRaisesRegex(v2.V2Blocked,'BUNDLE_SCHEMA'):
            v2.validate(self.reseal(b))

    def test_missing_field_blocks(self):
        b=self.fixture();b.pop('evidence_status')
        with self.assertRaisesRegex(v2.V2Blocked,'BUNDLE_TOPLEVEL'):
            v2.validate(self.reseal(b))


if __name__=='__main__': unittest.main()
