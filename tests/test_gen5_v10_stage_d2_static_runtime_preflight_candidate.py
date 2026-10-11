#!/usr/bin/env python3
"""D2 purely CPU/static memory-floor and local Git metadata gate tests."""
from __future__ import annotations

import os
import io
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import gen5_v10_stage_d2_static_runtime_preflight_candidate as d2


class TestD2MemoryStatic(unittest.TestCase):
    def make(self):
        return d2.memory_floor(intermediate_size=48, state_size=512)

    def test_memory_exact_b32(self):
        r=self.make()
        self.assertEqual(r['float32_input_bytes'], {
            'component':1610612736, 'log_a':402653184,'survival':402653184})
        self.assertEqual(r['mask_input_bytes'], 57344)
        self.assertEqual(r['logical_input_bytes_per_capture'],2415976448)
        self.assertEqual(r['logical_input_bytes_three_captures'],7247929344)
        self.assertEqual(r['logical_tensor_keys'],17)
        self.assertEqual(r['capture_phase_count'],3)
        d2.validate_memory_floor(r)

    def test_factorization_equivalent(self):
        a=self.make()
        b=d2.memory_floor(intermediate_size=1,state_size=24576)
        self.assertEqual(a['logical_input_bytes_per_capture'],b['logical_input_bytes_per_capture'])
        self.assertNotEqual(a['static_contract_sha256'],b['static_contract_sha256'])

    def test_no_peak_memory_claim(self):
        r=self.make()
        self.assertIsNone(r['observed_gpu_free_bytes'])
        self.assertIsNone(r['observed_cuda_peak_bytes'])
        self.assertFalse(r['maximum_resident_gpu_memory_certified'])
        self.assertFalse(r['additional_full_cuda_clone_allowed'])
        self.assertEqual(r['GPU_EXECUTION'],'NOT_AUTHORIZED')
        self.assertEqual(r['readiness'],'BLOCKED_PENDING_INDEPENDENT_RUNTIME_EVIDENCE')
        self.assertEqual(len(r['missing_evidence']),len(d2.STOP_CODES))

    def test_bad_width_blocks(self):
        for args in [(2,10),(10,100),(256,256)]:
            with self.subTest(args=args),self.assertRaisesRegex(d2.D2Blocked,'EXACT_STATE_WIDTH'):
                d2.memory_floor(intermediate_size=args[0],state_size=args[1])

    def test_non_integer_and_bool_geometry_blocks(self):
        for x in (True,0,-1,1.0,'48',None):
            with self.subTest(x=x),self.assertRaises(d2.D2Blocked):
                d2.memory_floor(intermediate_size=x,state_size=512)

    def test_bad_capture_chunk_blocks(self):
        for x in (0,3,5,4095,8193,8388612,16*1024*1024,True,4.0):
            with self.subTest(x=x),self.assertRaisesRegex(d2.D2Blocked,'CAPTURE_CHUNK_POLICY'):
                d2.memory_floor(intermediate_size=48,state_size=512,capture_chunk_bytes=x)

    def test_allowed_small_capture_chunk(self):
        r=d2.memory_floor(intermediate_size=48,state_size=512,capture_chunk_bytes=4096)
        self.assertEqual(r['max_leaf_payload_bytes'],4096)
        d2.validate_memory_floor(r)

    def test_report_tamper_falsifies_digest(self):
        r=self.make()
        r['logical_input_bytes_per_capture']+=1
        with self.assertRaisesRegex(d2.D2Blocked,'MEMORY_REPORT_MISMATCH'):
            d2.validate_memory_floor(r)

    def test_self_consistent_rehash_cannot_promote(self):
        r=self.make()
        r['GPU_EXECUTION']='PASS'
        r.pop('static_contract_sha256')
        import hashlib
        r['static_contract_sha256']=hashlib.sha256(d2.canonical_bytes(r)).hexdigest()
        with self.assertRaisesRegex(d2.D2Blocked,'MEMORY_REPORT_MISMATCH'):
            d2.validate_memory_floor(r)

    def test_injected_report_fields_block(self):
        r=self.make();r['actual_gpu_peak_bytes']=100
        with self.assertRaisesRegex(d2.D2Blocked,'MEMORY_REPORT_MISMATCH'):
            d2.validate_memory_floor(r)

    def test_missing_memory_report_fields_block(self):
        r=self.make();r.pop('mask_input_bytes')
        with self.assertRaisesRegex(d2.D2Blocked,'MEMORY_REPORT_MISMATCH'):
            d2.validate_memory_floor(r)

    def test_missing_shape_blocks(self):
        r=self.make();del r['shape']
        with self.assertRaisesRegex(d2.D2Blocked,'MEMORY_REPORT_SHAPE'):
            d2.validate_memory_floor(r)

    def test_stable_digest(self):
        self.assertEqual(self.make(),self.make())

    def test_json_nan_disallowed(self):
        with self.assertRaises(d2.D2Blocked): d2.canonical_bytes({'bad':float('nan')})


class TestD2LineageClaims(unittest.TestCase):
    @staticmethod
    def fixture():
        return {
            'checkpoint_sha256':d2.CHECKPOINT_SHA256,
            'dev_order_sha256':d2.DEV_ORDER_SHA256,
            'active_encoding_sha256':d2.ACTIVE_ENCODING_SHA256,
            'model_revision':d2.MODEL_REVISION,
            'worker_id':1,'batch_index_one_based':13,'row_start':800,'row_stop':832,
            'source_cell':'6201_6201','target_cells':['6201_6202','6201_6203'],
            'code_head':'0'*40,'claimed_branch':d2.BRANCH,
            'claimed_clean_worktree':True,'source_status':'SELF_ASSERTED_NOT_AUTHENTICATED',
            'physical_device':'NVIDIA Tesla T4 user claim','torch_version':'claimed 2.x',
            'cuda_runtime':'claimed 12.x','cublas_version':'claimed 12.x',
            'allow_tf32':False,
        }

    def test_valid_claim_never_becomes_authenticated(self):
        result=d2.validate_source_claims(self.fixture())
        self.assertFalse(result['source_authenticated'])
        self.assertFalse(result['metadata_independently_observed'])
        self.assertEqual(result['E0'],'NOT_EVALUATED')
        self.assertEqual(result['GPU_EXECUTION'],'NOT_AUTHORIZED')

    def test_source_pin_drift_blocks(self):
        for key in ('checkpoint_sha256','dev_order_sha256','active_encoding_sha256',
                    'model_revision','source_cell','worker_id','row_stop','claimed_branch'):
            kw=self.fixture(); kw[key]=None
            with self.subTest(key=key), self.assertRaisesRegex(d2.D2Blocked,'PINNED_CLAIM'):
                d2.validate_source_claims(kw)

    def test_target_reorder_blocks(self):
        kw=self.fixture();kw['target_cells'].reverse()
        with self.assertRaisesRegex(d2.D2Blocked,'TARGET_ORDER'):
            d2.validate_source_claims(kw)

    def test_self_asserted_false_status_blocks(self):
        kw=self.fixture();kw['source_status']='AUTHENTICATED'
        with self.assertRaisesRegex(d2.D2Blocked,'PINNED_CLAIM'):
            d2.validate_source_claims(kw)

    def test_false_clean_tf32_and_head_blocks(self):
        variants=(('claimed_clean_worktree',False,'CLEAN_WORKTREE_CLAIM'),
                  ('allow_tf32',True,'TF32_DISABLED_CLAIM'),
                  ('code_head',d2.BASE,'EXECUTION_HEAD_NOT_NEW'),
                  ('code_head','bad','EXECUTION_HEAD_NOT_NEW'))
        for field,value,code in variants:
            kw=self.fixture();kw[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(d2.D2Blocked,code):
                d2.validate_source_claims(kw)

    def test_missing_or_extra_field_blocks(self):
        kw=self.fixture();kw['e0']='PASS'
        with self.assertRaisesRegex(d2.D2Blocked,'SOURCE_CLAIMS_KEYS'):
            d2.validate_source_claims(kw)
        kw.pop('e0');kw.pop('cublas_version')
        with self.assertRaisesRegex(d2.D2Blocked,'SOURCE_CLAIMS_KEYS'):
            d2.validate_source_claims(kw)

    def test_source_runtime_blank_blocks(self):
        for field in ('physical_device','torch_version','cuda_runtime','cublas_version'):
            kw=self.fixture();kw[field]=' '
            with self.subTest(field=field),self.assertRaisesRegex(d2.D2Blocked,'RUNTIME_CLAIM'):
                d2.validate_source_claims(kw)


class TestD2LocalGit(unittest.TestCase):
    def git_runner(self, *, head=None, branch=None, status='', drift=None, ancestor=True):
        head=head or 'b'*40
        branch=branch or d2.BRANCH
        def run(argv, **kwargs):
            self.assertEqual(argv[0],'git')
            self.assertEqual(argv[1],'-C')
            self.assertTrue(kwargs['capture_output'])
            self.assertTrue(kwargs['text'])
            args=argv[3:]
            if args==['rev-parse','HEAD']:
                output=head
            elif args==['branch','--show-current']:
                output=branch
            elif args==['status','--porcelain=v1','--untracked-files=all']:
                output=status
            elif args==['merge-base','--is-ancestor',d2.BASE,head]:
                return SimpleNamespace(returncode=(0 if ancestor else 1),stdout='')
            elif len(args)==2 and args[0]=='rev-parse' and args[1].startswith('HEAD:'):
                path=args[1][len('HEAD:'):]
                self.assertIn(path,d2.EXPECTED_BLOBS)
                output=('0'*40 if drift==path else d2.EXPECTED_BLOBS[path])
            else:
                self.fail('UNEXPECTED GIT QUERY:'+str(args))
            return SimpleNamespace(returncode=0,stdout=output+'\n')
        return run

    def call(self, *, expected_head=None, **kw):
        head=expected_head or 'b'*40
        with tempfile.TemporaryDirectory() as temp, patch.object(d2.subprocess,'run',side_effect=self.git_runner(head=head,**kw)):
            return d2.inspect_clean_local_git(Path(temp),expected_head=head)

    def test_clean_git_matches_pinned_blobs(self):
        r=self.call()
        self.assertTrue(r['local_git_source_checked'])
        self.assertEqual(r['pinned_blobs_checked'],len(d2.EXPECTED_BLOBS))
        self.assertFalse(r['runtime_tensor_origin_authenticated'])
        self.assertFalse(r['model_checkpoint_authenticated'])
        self.assertEqual(r['E0'],'NOT_EVALUATED')

    def test_dirty_blocks(self):
        with self.assertRaisesRegex(d2.D2Blocked,'DIRTY_WORKTREE'):
            self.call(status='?? unrelated.txt')

    def test_branch_blocks(self):
        with self.assertRaisesRegex(d2.D2Blocked,'BRANCH_MISMATCH'):
            self.call(branch='main')

    def test_blob_drift_blocks(self):
        path=next(iter(d2.EXPECTED_BLOBS))
        with self.assertRaisesRegex(d2.D2Blocked,'FROZEN_GIT_BLOB_DRIFT'):
            self.call(drift=path)

    def test_ancestor_blocks(self):
        with self.assertRaisesRegex(d2.D2Blocked,'GIT_COMMAND_FAILED:merge-base'):
            self.call(ancestor=False)

    def test_git_io_error_blocks(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(d2.subprocess,'run',side_effect=OSError('unavailable')):
            with self.assertRaisesRegex(d2.D2Blocked,'GIT_IO_FAILURE'):
                d2.inspect_clean_local_git(Path(temp),expected_head='b'*40)

    def test_invalid_head_blocks_before_git(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(d2.subprocess,'run') as mocked:
            with self.assertRaisesRegex(d2.D2Blocked,'EXPECTED_HEAD_FORMAT'):
                d2.inspect_clean_local_git(Path(temp),expected_head='not a sha')
            mocked.assert_not_called()

    def test_git_path_not_directory_blocks(self):
        with self.assertRaisesRegex(d2.D2Blocked,'GIT_REPOSITORY_PATH'):
            d2.inspect_clean_local_git(Path('/non-existent-d2-not-a-repo'),expected_head='a'*40)

    def test_static_cli_never_queries_git_or_gpu(self):
        with patch.object(d2.subprocess,'run') as git,patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':''}):
            with patch.object(sys,'argv',['d2','--static-contract-check']):
                self.assertEqual(d2.main(),0)
            git.assert_not_called()

    def test_cli_unknown_execution_flag_rejected(self):
        with patch.object(sys,'argv',['d2','--static-contract-check','--run-bounded']):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx: d2.main()
            self.assertNotEqual(ctx.exception.code,0)

    def test_cli_cuda_visible_rejected(self):
        with patch.object(sys,'argv',['d2','--static-contract-check']),patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(d2.D2Blocked,'CUDA_MUST_BE_DISABLED'):
                d2.main()


if __name__=='__main__':
    unittest.main()
