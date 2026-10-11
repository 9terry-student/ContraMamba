#!/usr/bin/env python3
"""D4: in-process CPU callback wiring, no frozen/GPU execution."""
from __future__ import annotations

import copy
import contextlib
import io
import os
import sys
import unittest
from unittest.mock import patch

import torch

from scripts import gen5_v10_stage_d4_inprocess_cpu_wiring_candidate as d4
from scripts import gen5_v10_stage_d3_untrusted_instrumentation_candidate as d3
from scripts import gen5_v10_bounded_tensor_provenance_candidate as a
from test_gen5_v10_bounded_tensor_provenance_candidate import TestStageACanonicalProvenance
from test_gen5_v10_e0e3_v2_bundle_verifier_candidate import TestV2StrictCPU


class TestD4WiredCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        kw = TestStageACanonicalProvenance().fixture(b=32,w=8)
        vis = kw['component'][:2].clone()
        comp = kw['component'][2:].clone()
        component = torch.cat((vis,comp),0)
        kw['component'] = component
        capture = {k:v for k,v in kw.items() if k != 'synthetic_cpu_geometry'}
        old = [TestV2StrictCPU.row(i) for i in range(4)]
        new = [TestV2StrictCPU.row(i, candidate=True) for i in range(4)]
        timeline = []
        def callback(name,value):
            def run():
                timeline.append(name)
                return value
            return run
        outputs = {
            'CODE_SOURCE_CHECKED':d4.SOURCE_MARKER,
            'SELECTOR_LOCKED':copy.deepcopy(d3.BOUNDS),
            'BATCH_FEATURES_MASKS_READY':capture['attention_mask'],
            'COMMON_CONTEXT_LOG_A_SURVIVAL_READY':{'log_a':capture['log_a'],
                                                  'survival':capture['survival']},
            'RAW_WRITE_ALL_NINE_READY':{k:torch.ones(1) for k in d4.SOURCE_ORDER},
            'FIRST_SOURCE_GRADIENT_READY':(torch.zeros(1),torch.ones(1)),
            'FIRST_PROJECTOR_READY':'CPU_MOCK_PROJECTOR_READY',
            'FIRST_TARGET_GROUP_PROJECTED':(vis,comp),
            'K4_COMBINED_READY':component,
            'PRE_FROZEN_CAPTURED':capture,
            'FROZEN_FULL_REDUCER_RETURNED':old,
            'POST_FROZEN_CAPTURED':capture,
            'V10_HYBRID_REDUCER_RETURNED':new,
            'POST_V10_CAPTURED':capture,
            'ORIGINAL_RAW_PROJECTOR_GUARDS_EVALUATED':'CPU_MOCK_ORIGINAL_GUARDS_RETURNED',
            'V3_OFFLINE_RECORDS_VERIFIED':'CPU_MOCK_OFFLINE_REPLAY',
            'D1_ARTIFACT_PUBLISHED':d4.NO_ARTIFACT,
        }
        callbacks = {key:callback(key,outputs[key]) for key in d3.EVENTS}
        return callbacks,outputs,capture,timeline

    def run_fixture(self,callbacks):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            return d4.run_injected_cpu_fixture(callbacks=callbacks)

    def test_wired_order_calls_each_callback_once(self):
        cb,out,inputs,called=self.fixture()
        r=self.run_fixture(cb)
        self.assertEqual(called,list(d3.EVENTS))
        self.assertEqual(r['callbacks_invoked'],17)
        self.assertEqual(d3.validate(r['observation'])['event_count'],17)
        self.assertEqual(r['observation']['source_kind'],'CPU_SYNTHETIC_FIXTURE')

    def test_manifest_matches_stage_a_exact_cpu_values(self):
        cb,out,inp,called=self.fixture()
        r=self.run_fixture(cb)
        expected=a.manifest_for_cpu_inputs(**inp,synthetic_cpu_geometry=True)
        self.assertEqual(r['capture_manifest_sha256'],expected['manifest_sha256'])
        self.assertEqual([row['capture_manifest_sha256'] for row in r['observation']['events'] if row['capture_manifest_sha256'] is not None],[expected['manifest_sha256']]*3)

    def test_e0_e3_never_promote_and_no_artifact(self):
        cb,out,inp,called=self.fixture()
        r=self.run_fixture(cb)
        for field in ('V10_E0_PROVENANCE','V10_E1_FROZEN_GUARDS','V10_E2_PAYLOAD','V10_E3_DIAGNOSTICS'):
            self.assertEqual(r[field],'NOT_EVALUATED')
        self.assertFalse(r['artifact_written'])
        self.assertFalse(r['input_source_authenticated'])
        self.assertFalse(r['upstream_guard_authenticated'])
        self.assertFalse(r['model_loaded'])

    def test_all_guards_remain_claims(self):
        cb,_,_,_=self.fixture()
        r=self.run_fixture(cb)
        self.assertEqual(set(r['observation']['guard_trace'].values()),{'OBSERVED_CLAIM_ONLY'})
        self.assertEqual(r['trace_verdict']['E1'],'NOT_EVALUATED')

    def test_missing_callback_blocks_before_execution(self):
        cb,_,_,called=self.fixture();cb.pop('V10_HYBRID_REDUCER_RETURNED')
        with self.assertRaisesRegex(d4.D4Blocked,'CALLBACK_EXACT_SET'): self.run_fixture(cb)
        self.assertEqual(called,[])

    def test_extra_callback_blocks(self):
        cb,_,_,called=self.fixture();cb['invented']=lambda:None
        with self.assertRaisesRegex(d4.D4Blocked,'CALLBACK_EXACT_SET'): self.run_fixture(cb)
        self.assertEqual(called,[])

    def test_inert_callback_blocks(self):
        cb,_,_,called=self.fixture();cb['FIRST_PROJECTOR_READY']='NOT_CALLABLE'
        with self.assertRaisesRegex(d4.D4Blocked,'CALLBACK_NOT_CALLABLE'): self.run_fixture(cb)
        self.assertEqual(called,[])

    def test_engine_callback_identity_independent(self):
        cb,_,_,called=self.fixture();cb['V10_HYBRID_REDUCER_RETURNED']=cb['FROZEN_FULL_REDUCER_RETURNED']
        with self.assertRaisesRegex(d4.D4Blocked,'ENGINE_CALLBACK_NOT_INDEPENDENT'):self.run_fixture(cb)
        self.assertEqual(called,[])

    def test_cuda_visible_blocks(self):
        cb,_,_,called=self.fixture()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(d4.D4Blocked,'CUDA_NOT_DISABLED'):d4.run_injected_cpu_fixture(callbacks=cb)
        self.assertEqual(called,[])

    def test_source_marker_blocks(self):
        cb,out,_,called=self.fixture();cb['CODE_SOURCE_CHECKED']=lambda:'GPU_AUTHENTICATED'
        with self.assertRaisesRegex(d4.D4Blocked,'CPU_SOURCE_MARKER'):self.run_fixture(cb)
        self.assertEqual(called,[])

    def test_selector_drift_blocks(self):
        cb,out,_,called=self.fixture();bad=copy.deepcopy(d3.BOUNDS);bad['row_start']=799
        cb['SELECTOR_LOCKED']=lambda:bad
        with self.assertRaisesRegex(d4.D4Blocked,'SELECTOR_MISMATCH'):self.run_fixture(cb)
        self.assertEqual(called,['CODE_SOURCE_CHECKED'])

    def test_batch_dtype_blocks(self):
        cb,out,_,called=self.fixture();cb['BATCH_FEATURES_MASKS_READY']=lambda:torch.zeros((32,128),dtype=torch.float32)
        with self.assertRaisesRegex(d4.D4Blocked,'BATCH_MASK_GEOMETRY'):self.run_fixture(cb)

    def test_batch_cuda_or_nontensor_blocks(self):
        cb,out,_,called=self.fixture();cb['BATCH_FEATURES_MASKS_READY']=lambda:'not cpu tensor'
        with self.assertRaisesRegex(d4.D4Blocked,'CPU_ONLY'):self.run_fixture(cb)

    def test_dynamics_missing_blocks(self):
        cb,out,_,called=self.fixture();cb['COMMON_CONTEXT_LOG_A_SURVIVAL_READY']=lambda:{'log_a':out['COMMON_CONTEXT_LOG_A_SURVIVAL_READY']['log_a']}
        with self.assertRaisesRegex(d4.D4Blocked,'DYNAMICS_KEYS'):self.run_fixture(cb)

    def test_raw_nine_missing_blocks(self):
        cb,out,_,called=self.fixture();raw=copy.copy(out['RAW_WRITE_ALL_NINE_READY']);raw.pop(next(iter(raw)))
        cb['RAW_WRITE_ALL_NINE_READY']=lambda:raw
        with self.assertRaisesRegex(d4.D4Blocked,'NINE_RAW_CELLS_ORDER'):self.run_fixture(cb)

    def test_raw_nine_order_blocks(self):
        cb,out,_,called=self.fixture();raw=dict(reversed(list(out['RAW_WRITE_ALL_NINE_READY'].items())))
        cb['RAW_WRITE_ALL_NINE_READY']=lambda:raw
        with self.assertRaisesRegex(d4.D4Blocked,'NINE_RAW_CELLS_ORDER'):self.run_fixture(cb)

    def test_gradient_pair_blocks(self):
        cb,out,_,called=self.fixture();cb['FIRST_SOURCE_GRADIENT_READY']=lambda:(torch.zeros(1),)
        with self.assertRaisesRegex(d4.D4Blocked,'GRADIENT_PAIR'):self.run_fixture(cb)

    def test_projector_marker_blocks(self):
        cb,out,_,called=self.fixture();cb['FIRST_PROJECTOR_READY']=lambda:'PASS'
        with self.assertRaisesRegex(d4.D4Blocked,'PROJECTOR_MOCK_MARKER'):self.run_fixture(cb)

    def test_wrong_projected_shape_blocks(self):
        cb,out,_,called=self.fixture();cb['FIRST_TARGET_GROUP_PROJECTED']=lambda:(torch.zeros(1),torch.zeros(1))
        with self.assertRaisesRegex(d4.D4Blocked,'PROJECTED_GEOMETRY'):self.run_fixture(cb)

    def test_k4_changed_component_blocks(self):
        cb,out,_,called=self.fixture();cb['K4_COMBINED_READY']=lambda:out['K4_COMBINED_READY'].clone().add(1)
        with self.assertRaisesRegex(d4.D4Blocked,'K4_MISMATCH'):self.run_fixture(cb)

    def test_capture_missing_key_blocks(self):
        cb,out,_,called=self.fixture();bad=dict(out['PRE_FROZEN_CAPTURED']);bad.pop('fine_masks')
        cb['PRE_FROZEN_CAPTURED']=lambda:bad
        with self.assertRaisesRegex(d4.D4Blocked,'CAPTURE_INPUT_FIELDS'):self.run_fixture(cb)

    def test_capture_clone_not_shared_blocks(self):
        cb,out,_,called=self.fixture();bad=dict(out['PRE_FROZEN_CAPTURED']);bad['component']=bad['component'].clone()
        cb['PRE_FROZEN_CAPTURED']=lambda:bad
        with self.assertRaisesRegex(d4.D4Blocked,'CAPTURE_COMPONENT_NOT_SHARED'):self.run_fixture(cb)

    def test_capture_wrong_mask_identity_blocks(self):
        cb,out,_,called=self.fixture();bad=dict(out['PRE_FROZEN_CAPTURED']);bad['attention_mask']=bad['attention_mask'].clone()
        cb['PRE_FROZEN_CAPTURED']=lambda:bad
        with self.assertRaisesRegex(d4.D4Blocked,'CAPTURE_MASK_NOT_SHARED'):self.run_fixture(cb)

    def test_frozen_mutation_blocks_before_v10(self):
        cb,out,inp,called=self.fixture()
        original=cb['FROZEN_FULL_REDUCER_RETURNED']
        def mutate():
            inp['component'][0,0,0,1] += .1
            return original()
        cb['FROZEN_FULL_REDUCER_RETURNED']=mutate
        with self.assertRaisesRegex(d4.D4Blocked,'INPUT_MUTATION:POST_FROZEN_CAPTURED'):self.run_fixture(cb)
        self.assertNotIn('V10_HYBRID_REDUCER_RETURNED',called)

    def test_v10_mutation_blocks(self):
        cb,out,inp,called=self.fixture()
        original=cb['V10_HYBRID_REDUCER_RETURNED']
        def mutate():
            inp['fine_masks']['CLAIM']['A_TITLE'][0,0]=False
            return original()
        cb['V10_HYBRID_REDUCER_RETURNED']=mutate
        with self.assertRaises(d4.D4Blocked):self.run_fixture(cb)
        self.assertNotIn('ORIGINAL_RAW_PROJECTOR_GUARDS_EVALUATED',called)

    def test_bad_frozen_records_blocks(self):
        cb,out,inp,called=self.fixture();cb['FROZEN_FULL_REDUCER_RETURNED']=lambda:[{'R':1}]*4
        with self.assertRaisesRegex(d4.D4Blocked,'V3_RECORD'):self.run_fixture(cb)

    def test_forged_diagnostic_blocks(self):
        cb,out,inp,called=self.fixture();wrong=copy.deepcopy(out['V10_HYBRID_REDUCER_RETURNED']);wrong[0]['fine_parent_reconstruction_rel_max']=0.0
        cb['V10_HYBRID_REDUCER_RETURNED']=lambda:wrong
        with self.assertRaisesRegex(d4.D4Blocked,'V3_RECORD'):self.run_fixture(cb)

    def test_guard_exception_propagates_and_no_publish(self):
        cb,out,inp,called=self.fixture()
        def guard(): raise RuntimeError('RAW_GUARD_FAILED')
        cb['ORIGINAL_RAW_PROJECTOR_GUARDS_EVALUATED']=guard
        with self.assertRaisesRegex(RuntimeError,'RAW_GUARD_FAILED'):self.run_fixture(cb)
        self.assertNotIn('D1_ARTIFACT_PUBLISHED',called)

    def test_guard_callback_late_mutation_blocks_at_final_check(self):
        cb,out,inp,called=self.fixture()
        def mutate():
            inp['survival'][0,0,0] += 0.125
            return 'CPU_MOCK_ORIGINAL_GUARDS_RETURNED'
        cb['ORIGINAL_RAW_PROJECTOR_GUARDS_EVALUATED'] = mutate
        with self.assertRaisesRegex(d4.D4Blocked,'INPUT_MUTATION:FINAL_CALLBACKS'):
            self.run_fixture(cb)
        self.assertIn('D1_ARTIFACT_PUBLISHED',called)

    def test_publication_callback_mutation_blocks_at_final_check(self):
        cb,out,inp,called=self.fixture()
        def mutate():
            inp['component'][1,0,0,0] += 0.125
            return d4.NO_ARTIFACT
        cb['D1_ARTIFACT_PUBLISHED'] = mutate
        with self.assertRaisesRegex(d4.D4Blocked,'INPUT_MUTATION:FINAL_CALLBACKS'):
            self.run_fixture(cb)

    def test_false_publication_blocks(self):
        cb,out,inp,called=self.fixture();cb['D1_ARTIFACT_PUBLISHED']=lambda:'REAL_ARTIFACT_PUBLISHED'
        with self.assertRaisesRegex(d4.D4Blocked,'FALSE_PUBLICATION'):self.run_fixture(cb)

    def test_constant_clock_blocks(self):
        cb,out,inp,called=self.fixture()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(d4.D4Blocked,'NONMONOTONIC_CLOCK'):
                d4.run_injected_cpu_fixture(callbacks=cb,clock_ns=lambda:500)

    def test_zero_or_bad_clock_blocks(self):
        cb,out,inp,called=self.fixture()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(d4.D4Blocked,'CLOCK_START'):
                d4.run_injected_cpu_fixture(callbacks=cb,clock_ns=lambda:-1)

    def test_static_cli(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with patch.object(sys,'argv',['d4','--static-contract-check']),contextlib.redirect_stdout(io.StringIO()) as cap:
                self.assertEqual(d4.main(),0)
            self.assertIn('GPU=OFF',cap.getvalue())
            self.assertIn('NOT_AUTHENTICATED',cap.getvalue())

    def test_cli_run_bounded_rejected(self):
        with patch.object(sys,'argv',['d4','--run-bounded']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:d4.main()
        self.assertNotEqual(e.exception.code,0)


if __name__=='__main__': unittest.main()
