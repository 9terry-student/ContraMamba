#!/usr/bin/env python3
"""V3 offline runtime-bundle fail-closed negative tests, CPU only."""
from __future__ import annotations

import copy
import math
import unittest

import torch
from scripts import gen5_v10_bounded_worker_export_candidate as b
from scripts import gen5_v10_e0e3_v3_runtime_verifier_candidate as v3
from scripts import gen5_v10_e0e3_v2_bundle_verifier_candidate as v2


class TestV3UntrustedCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        from test_gen5_v10_bounded_tensor_provenance_candidate import TestStageACanonicalProvenance
        from test_gen5_v10_e0e3_v2_bundle_verifier_candidate import TestV2StrictCPU
        kw = TestStageACanonicalProvenance().fixture(b=32,w=8)
        inputs = {'component':kw['component'],'log_a':kw['log_a'],
                  'survival':kw['survival'],'attention_mask':kw['attention_mask']}
        inputs.update({f'role_masks.{k}':v for k,v in kw['role_masks'].items()})
        inputs.update({f'fine_masks.{side}.{k}':v
                       for side, masks in kw['fine_masks'].items() for k,v in masks.items()})
        cap = {phase:{key:b.capture_value_digest(t,phase=label) for key,t in inputs.items()}
               for phase,label in (('pre_frozen','PRE_FROZEN'),('post_frozen','POST_FROZEN'),
                                    ('post_v10','POST_V10'))}
        locus = b.choose_bounded_locus(
            worker_id=1,batch_index=13,row_start=800,row_stop=832,
            row_ids=[f'ROW_{i}' for i in range(800,832)],
            source_cells=b.SOURCE_CELLS,orientations=b.FROZEN_ORIENTATIONS,
            selected_source=b.SOURCE_CELLS[0], selected_targets=(b.SOURCE_CELLS[1],b.SOURCE_CELLS[2]))
        raw_frozen = [TestV2StrictCPU.row(i) for i in range(4)]
        raw_v10 = [TestV2StrictCPU.row(i,candidate=True) for i in range(4)]
        payload = {
            'schema_version':v3.SCHEMA,'candidate_base':v3.BASE,'selector':locus,
            'code_identity':{
                'branch':'gen5-causal-role-state-ownership','frozen_base':v3.BASE,
                'code_head':'NOT_AUTHENTICATED','frozen_blobs':copy.deepcopy(v3.FROZEN_BLOBS),
                'identity_status':'NOT_AUTHENTICATED',
            },
            'runtime_claims':{
                'source_kind':'CPU_MOCK_ONLY','execution_status':'NO',
                'physical_device':'CPU','model_revision':'NOT_AUTHENTICATED',
                'checkpoint_sha256':'NOT_AUTHENTICATED',
                'dev_order_sha256':'NOT_AUTHENTICATED',
                'active_encoding_sha256':'NOT_AUTHENTICATED',
                'torch_version':'NOT_AUTHENTICATED','cuda_runtime':'NOT_AUTHENTICATED',
                'cublas_version':'NOT_AUTHENTICATED','tf32':False,
                'determinism':'NOT_AUTHENTICATED',
            },
            'input_capture':cap,
            'engines':{
                'frozen':{'engine_id':'FROZEN_FFT_V1','records':raw_frozen,
                          'records_sha256':v3.digest(raw_frozen),
                          'dispatch':{'mode':'FROZEN_WHOLE','batch_rows':32},
                          'guard_status':'UNAUTHENTICATED_RUNTIME_CLAIM'},
                'v10':{'engine_id':'V10_HYBRID_CANDIDATE','records':raw_v10,
                       'records_sha256':v3.digest(raw_v10),
                       'dispatch':{'mode':'CHUNKWISE_HYBRID','batch_rows':32,
                                   'fast_chunks':1,'gemm_chunks':1,'frozen_tail_calls':0,
                                   'total_chunks':2,'max_half_span':95.},
                       'guard_status':'UNAUTHENTICATED_RUNTIME_CLAIM'},
            },
            'guard_trace':{name:'NOT_AUTHENTICATED' for name in v3.GUARD_KEYS},
            'failure_evidence':[],
            'evidence_status':'UNAUTHENTICATED_BOUNDED_RECORDS_NO_PROMOTION',
        }
        return v3.seal(payload)

    @staticmethod
    def reseal(bundle):
        bundle.pop('bundle_sha256',None);bundle.pop('sidecar',None)
        for name in ('frozen','v10'):
            bundle['engines'][name]['records_sha256']=v3.digest(bundle['engines'][name]['records'])
        return v3.seal(bundle)

    def check_blocks(self, mutate, code):
        bundle=self.fixture();mutate(bundle)
        with self.assertRaisesRegex(v3.V3Blocked, code):
            v3.validate(self.reseal(bundle))

    def test_bounded_synthetic_retains_legacy_failure(self):
        actual=v3.validate(self.fixture())
        self.assertEqual(actual['LEGACY_FULL_RECORD'],'FAIL')
        self.assertEqual(actual['raw_comparison']['diagnostic_failures'],4)
        self.assertEqual(actual['raw_comparison']['payload_failures'],0)
        self.assertEqual(actual['normalized_H']['H_by_gap'],38100)
        self.assertEqual(actual['normalized_H']['H_windows'],1500)
        self.assertEqual(actual['normalized_H']['failures'],0)
        for k in ('V10_E0_PROVENANCE','V10_E1_FROZEN_GUARDS','V10_E2_PAYLOAD',
                  'V10_E3_DIAGNOSTICS','V10_PAYLOAD_EQUIVALENCE'):
            self.assertEqual(actual[k],'NOT_EVALUATED')

    def test_wrong_v2_schema(self):
        self.check_blocks(lambda x:x.__setitem__('schema_version',v2.SCHEMA),'SCHEMA_OR_BASE')

    def test_tampered_top_level_bundle_hash(self):
        x=self.fixture();x['selector']['row_ids'][0]='TAMPER'
        with self.assertRaisesRegex(v3.V3Blocked,'BUNDLE_HASH'):
            v3.validate(x)

    def test_sidecar_hash_blocks(self):
        x=self.fixture();x['sidecar']['sidecar_sha256']='0'*64
        with self.assertRaisesRegex(v3.V3Blocked,'SIDECAR_HASH'):v3.validate(x)

    def test_source_mismatch(self):
        self.check_blocks(lambda x:x['selector'].__setitem__('source_cell','6202_6202'),'SELECTOR_IDENTITY')

    def test_target_order_mismatch(self):
        self.check_blocks(lambda x:x['selector']['target_cells'].reverse(),'SELECTOR_IDENTITY')

    def test_row_duplicate(self):
        self.check_blocks(lambda x:x['selector']['row_ids'].__setitem__(0,'ROW_801'),'ROW_IDS')

    def test_wrong_worker_batch(self):
        self.check_blocks(lambda x:x['selector'].__setitem__('batch_index_one_based',12),'BATCH_ID')

    def test_omitted_code_blob(self):
        def mutate(x):x['code_identity']['frozen_blobs'].pop(next(iter(v3.FROZEN_BLOBS)))
        self.check_blocks(mutate,'CODE_IDENTITY_FALSE_ATTESTATION')

    def test_bad_execution_head(self):
        self.check_blocks(lambda x:x['code_identity'].__setitem__('code_head',v3.BASE),'CODE_IDENTITY_FALSE_ATTESTATION')

    def test_fake_cuda_source_claim(self):
        self.check_blocks(lambda x:x['runtime_claims'].__setitem__('source_kind','RUNTIME_CLAIM_UNAUTHENTICATED'),'EXECUTION_CLAIM')

    def test_false_authenticated_checkpoint(self):
        self.check_blocks(lambda x:x['runtime_claims'].__setitem__('checkpoint_sha256','1'*64),'RUNTIME_FALSE_ATTEST')

    def test_tensor_digest_drift(self):
        self.check_blocks(lambda x:x['input_capture']['post_frozen']['component'].__setitem__('content_sha256','0'*64),'INPUT_VALUE_DRIFT')

    def test_mask_digest_drift(self):
        self.check_blocks(lambda x:x['input_capture']['post_v10']['fine_masks.CLAIM.A_ROLE'].__setitem__('content_sha256','0'*64),'INPUT_VALUE_DRIFT')

    def test_missing_capture_key(self):
        self.check_blocks(lambda x:x['input_capture']['pre_frozen'].pop('component'),'CAPTURE_TENSOR_KEYS')

    def test_bad_capture_order(self):
        self.check_blocks(lambda x:x['input_capture']['pre_frozen']['component'].__setitem__('phase','POST_V10'),'CAPTURE_CLAIM')

    def test_bad_capture_byte_count(self):
        self.check_blocks(lambda x:x['input_capture']['pre_frozen']['component'].__setitem__('byte_count',100),'CAPTURE_HASH_SIZE')

    def test_missing_guard_trace(self):
        self.check_blocks(lambda x:x['guard_trace'].pop('raw_reconstruction_auth'),'GUARD_COVERAGE')

    def test_forged_upstream_guard_pass(self):
        self.check_blocks(lambda x:x['guard_trace'].__setitem__('source_gradient_projector','PASS'),'GUARD_FALSE_ATTEST')

    def test_nonempty_failure_blocks_e1(self):
        x=self.fixture();x['failure_evidence']=['RAW_RECONSTRUCTION_AUTH:FAIL']
        report=v3.validate(self.reseal(x))
        self.assertEqual(report['V10_E1_FROZEN_GUARDS'],'FAIL')
        self.assertEqual(report['V10_PAYLOAD_EQUIVALENCE'],'FAIL')

    def test_missing_record_field(self):
        self.check_blocks(lambda x:x['engines']['v10']['records'][0].pop('R'),'RECORD_SCHEMA')

    def test_extra_record_field(self):
        self.check_blocks(lambda x:x['engines']['v10']['records'][0].__setitem__('invented',0),'RECORD_SCHEMA')

    def test_forged_diagnostic(self):
        self.check_blocks(lambda x:x['engines']['v10']['records'][0].__setitem__('fine_parent_reconstruction_rel_max',0.),'DIAGNOSTIC_FORGERY')

    def test_fine_cell_omission(self):
        self.check_blocks(lambda x:x['engines']['v10']['records'][0]['fine_pair_C_by_gap'][v3.PARENTS[0]].pop(v3.FINE[-1]),'FINE_CELLS')

    def test_parent_replay_drift(self):
        self.check_blocks(lambda x:x['engines']['v10']['records'][0]['fine_pair_C_by_gap'][v3.PARENTS[0]][v3.FINE[0]].__setitem__(8,0.2),'FROZEN_PARENT_REPLAY')

    def test_nan(self):
        self.check_blocks(lambda x:x['engines']['v10']['records'][0].__setitem__('E',float('nan')),'NONCANONICAL_JSON')

    def test_payload_fail_not_promoted(self):
        x=self.fixture();x['engines']['v10']['records'][0]['R']=7.0
        report=v3.validate(self.reseal(x))
        self.assertEqual(report['V10_E2_PAYLOAD'],'FAIL')
        self.assertEqual(report['V10_PAYLOAD_EQUIVALENCE'],'FAIL')

    def test_dispatch_coverage(self):
        self.check_blocks(lambda x:x['engines']['v10']['dispatch'].__setitem__('gemm_chunks',0),'V10_ROUTE_COVERAGE')

    def test_negative_half_span_blocks(self):
        self.check_blocks(lambda x:x['engines']['v10']['dispatch']
                          .__setitem__('max_half_span',-1.0),'V10_HALF_SPAN_NEGATIVE')

    def test_guard_false_positive(self):
        self.check_blocks(lambda x:x['engines']['frozen'].__setitem__('guard_status','PASS'),'GUARD_FALSE_CLAIM')

    def test_record_hash_without_reseal(self):
        x=self.fixture();x['engines']['v10']['records'][0]['R']=7.0
        x['bundle_sha256']=v3.digest({k:v for k,v in x.items() if k not in ('bundle_sha256','sidecar')})
        sc={"domain":"GEN5_V10_V3_SIDECAR_V1","bundle_sha256":x['bundle_sha256']}
        import hashlib
        x['sidecar']=dict(sc,sidecar_sha256=hashlib.sha256(v3.SIDECAR_DOMAIN+v3.canonical(sc)).hexdigest())
        with self.assertRaisesRegex(v3.V3Blocked,'RECORD_HASH'):v3.validate(x)

    def pseudo_runtime_claim_fixture(self):
        """Only fake metadata; no full-width tensor allocation or GPU execution."""
        v=self.fixture()
        v['code_identity']['code_head']='a'*40
        v['code_identity']['identity_status']='SELF_ASSERTED_NOT_AUTHENTICATED'
        r=v['runtime_claims']
        r.update({
            'source_kind':'RUNTIME_CLAIM_UNAUTHENTICATED',
            'execution_status':'UNVERIFIED',
            'physical_device':'NVIDIA Tesla T4 (SELF_ASSERTED)',
            'model_revision':'40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37',
            'checkpoint_sha256':'1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f',
            'dev_order_sha256':'b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25',
            'active_encoding_sha256':'e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51',
            'torch_version':'2.x.self.asserted',
            'cuda_runtime':'12.x.self.asserted',
            'cublas_version':'self.asserted',
            'determinism':'DISABLED_CLAIM',
        })
        for phase in v['input_capture'].values():
            for key, meta in phase.items():
                if key == 'component': meta['shape']=[4,32,128,24576]
                elif key == 'log_a': meta['shape']=[32,128,1,24576]
                elif key == 'survival': meta['shape']=[32,128,24576]
                meta['byte_count']=math.prod(meta['shape'])*(4 if meta['dtype']=='float32' else 1)
                stride=1
                strides=[]
                for dim in reversed(meta['shape']):
                    strides.insert(0,stride)
                    stride*=dim
                meta['source_strides']=strides
                meta['source_device']='cuda:0'

        v['engines']['v10']['dispatch'].update({'fast_chunks':0,'gemm_chunks':96,
                                                 'total_chunks':96})
        return self.reseal(v)

    def test_self_asserted_runtime_metadata_still_does_not_promote(self):
        report=v3.validate(self.pseudo_runtime_claim_fixture())
        self.assertEqual(report['geometry']['W'],24576)
        self.assertFalse(report['input_source_authenticated'])
        for field in ('V10_E0_PROVENANCE','V10_E1_FROZEN_GUARDS',
                      'V10_E2_PAYLOAD','V10_E3_DIAGNOSTICS',
                      'V10_PAYLOAD_EQUIVALENCE'):
            self.assertEqual(report[field], 'NOT_EVALUATED')
        self.assertEqual(report['LEGACY_FULL_RECORD'],'FAIL')

    def test_untrusted_runtime_wrong_checkpoint_blocks(self):
        x=self.pseudo_runtime_claim_fixture()
        x['runtime_claims']['checkpoint_sha256']='f'*64
        with self.assertRaisesRegex(v3.V3Blocked,'RUNTIME_PINNED_SOURCE_CLAIMS'):
            v3.validate(self.reseal(x))

    def test_untrusted_runtime_base_head_not_executable(self):
        x=self.pseudo_runtime_claim_fixture()
        x['code_identity']['code_head']=v3.BASE
        with self.assertRaisesRegex(v3.V3Blocked,'RUNTIME_CODE_NOT_VERIFIED'):
            v3.validate(self.reseal(x))

    def test_cli_has_no_execution_flags(self):
        import subprocess,sys,os
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        out=subprocess.run([sys.executable,'-B','-m','scripts.gen5_v10_e0e3_v3_runtime_verifier_candidate',
                            '--run-bounded'],capture_output=True,text=True,env=env)
        self.assertNotEqual(out.returncode,0)
        self.assertTrue('unrecognized arguments' in out.stderr or 'required' in out.stderr)


if __name__ == '__main__':unittest.main()
