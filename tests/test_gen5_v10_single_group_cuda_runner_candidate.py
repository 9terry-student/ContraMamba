"""CPU-only negative tests for a real-run-capable but approval-gated candidate."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import gen5_v10_single_group_cuda_runner_candidate as r

class TestGen5BoundedGpuRunnerNoGpu(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name)
        (self.repo/'scripts').mkdir()
    def fake_git(self, repo, *args):
        if args==('rev-parse','HEAD'): return 'a'*40
        if args==('branch','--show-current'): return r.BRANCH
        if args==('status','--porcelain=v1','--untracked-files=all'): return ''
        if args==('merge-base','--is-ancestor',r.PREDECESSOR,'HEAD'): return ''
        if len(args)==2 and args[0]=='rev-parse' and args[1].startswith('HEAD:'): return r.FROZEN.get(args[1][5:],'f'*40) or 'f'*40
        raise AssertionError(str(args))
    def test_locus_constants(self):
        self.assertEqual((r.EXPECTED['worker'],r.EXPECTED['batch'],r.EXPECTED['start'],r.EXPECTED['stop']), (1,13,800,832))
        self.assertEqual(r.EXPECTED['targets'],((6201,6202),(6201,6203)))
    def test_approval_is_closed_default(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(r.BoundedBlocked,'OPERATOR_GPU_APPROVAL_REQUIRED'):
                r._gpu_entry_guard(argparse.Namespace(run_bounded=True,explicit_gpu_approval=None))
    def test_approval_env_missing(self):
        a=argparse.Namespace(run_bounded=True,explicit_gpu_approval='I_APPROVE_GEN5_V10_BOUNDED_GPU')
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'},clear=True):
            with self.assertRaisesRegex(r.BoundedBlocked,'EXTERNAL_EXECUTION_AUTHORITY_REQUIRED'):
                r._gpu_entry_guard(a)
    def test_approval_gpu_mapping_strict(self):
        a=argparse.Namespace(run_bounded=True,explicit_gpu_approval='I_APPROVE_GEN5_V10_BOUNDED_GPU')
        with patch.dict(os.environ,{'CONTRAMAMBA_GEN5_V10_RUN_AUTHORIZED':'YES','CUDA_VISIBLE_DEVICES':'1'},clear=True):
            with self.assertRaisesRegex(r.BoundedBlocked,'EXACT_ONE_VISIBLE_DEVICE_REQUEST'):
                r._gpu_entry_guard(a)
    def test_approval_head_equal_freeze(self):
        a=argparse.Namespace(run_bounded=True,explicit_gpu_approval='I_APPROVE_GEN5_V10_BOUNDED_GPU',expected_head='a'*40,implementation_freeze_commit='b'*40,model_snapshot=Path('a'),tokenizer_snapshot=Path('b'),checkpoint=Path('c'),output_dir=Path('d'))
        with patch.dict(os.environ,{'CONTRAMAMBA_GEN5_V10_RUN_AUTHORIZED':'YES','CUDA_VISIBLE_DEVICES':'0'},clear=True):
            with self.assertRaisesRegex(r.BoundedBlocked,'EXECUTION_HEAD_NOT_FREEZE'):
                r._gpu_entry_guard(a)
    def test_no_run_bounded_static_import(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''},clear=True), patch.object(sys,'stdout',new_callable=io.StringIO) as capture:
            self.assertEqual(r.main(['--static-contract-check']),0)
            self.assertIn('GPU=OFF',capture.getvalue())
    def test_static_rejects_cuda_visibility(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(r.BoundedBlocked,'STATIC_REQUIRES_GPU_OFF'):
                r.main(['--static-contract-check'])
    def test_static_rejects_run_args(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(r.BoundedBlocked,'STATIC_EXTRA_ARGS_FORBIDDEN'):
                r.main(['--static-contract-check','--expected-head','a'*40])
    def test_unknown_cli_rejected(self):
        with self.assertRaises(SystemExit):
            r.main(['--run-worker'])
    def test_bad_hash_format(self):
        with self.assertRaisesRegex(r.BoundedBlocked,'EXPECTED_HEAD_FORMAT'):
            r.verify_committed_execution(self.repo,'abc')
    def test_wrong_head(self):
        with patch.object(r,'_git',side_effect=self.fake_git):
            with self.assertRaisesRegex(r.BoundedBlocked,'HEAD_DRIFT'):
                r.verify_committed_execution(self.repo,'b'*40,require_current_file=False)
    def test_clean_committed_metadata_positive(self):
        with patch.object(r,'_git',side_effect=self.fake_git):
            ret=r.verify_committed_execution(self.repo,'a'*40,require_current_file=False)
        self.assertEqual(ret['head'],'a'*40)
        self.assertIn('frozen_git_blobs',ret)
    def test_head_in_past_not_accepted(self):
        with patch.object(r,'_git',side_effect=lambda repo,*args:'\n' if args[0]=='status' else self.fake_git(repo,*args)):
            with self.assertRaisesRegex(r.BoundedBlocked,'WORKTREE_NOT_CLEAN'):
                r.verify_committed_execution(self.repo,'a'*40,require_current_file=False)
    def test_pinned_blob_drift(self):
        def fake(repo,*args):
            if args==('rev-parse','HEAD:scripts/audit_native_recurrent_generator_anchor_pair_localization.py'):
                return '0'*40
            return self.fake_git(repo,*args)
        with patch.object(r,'_git',side_effect=fake):
            with self.assertRaisesRegex(r.BoundedBlocked,'FROZEN_BLOB_DRIFT'):
                r.verify_committed_execution(self.repo,'a'*40,require_current_file=False)
    def test_output_location_must_be_direct_child(self):
        root=self.repo/'reports'/'native_recurrent_generator_anchor_pair_localization_v10_bounded_cuda_runs'
        with self.assertRaisesRegex(r.BoundedBlocked,'OUTPUT_DIR_OUTSIDE_APPROVED_ROOT'):
            r._output_dir(self.repo,self.repo/'escape')
        self.assertEqual(r._output_dir(self.repo,root/'one'),root/'one')
        (root/'one').mkdir(parents=True)
        with self.assertRaisesRegex(r.BoundedBlocked,'OUTPUT_COLLISION'):
            r._output_dir(self.repo,root/'one')
    def test_exact_capture_compare(self):
        x={'component':{'content_sha256':'a','byte_count':10,'shape':[1],'dtype':'float32','source_device':'cuda:0'}}
        r._equal_capture(x,{'component':dict(x['component'])},'X')
        y={'component':dict(x['component'],content_sha256='b')}
        with self.assertRaisesRegex(r.BoundedBlocked,'INPUT_MUTATED:X:component'):
            r._equal_capture(x,y,'X')
    def test_capture_keys_mismatch(self):
        with self.assertRaisesRegex(r.BoundedBlocked,'CAPTURE_KEYS'):
            r._equal_capture({'a':1},{},'test')
    def test_atomic_write_and_hash(self):
        r._write_one(self.repo,'bounded_result.json',{'a':1})
        data=(self.repo/'bounded_result.json').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),(self.repo/'bounded_result.json.sha256').read_text().strip())
        with self.assertRaises(FileExistsError):
            r._write_one(self.repo,'bounded_result.json',{'a':2})
    def test_static_module_lacks_full_worker_dispatch(self):
        s=Path(r.__file__).read_text(encoding='utf-8')
        # Textual mention in comments allowed; the callable must never be invoked.
        self.assertNotIn('frozen.run_worker(',s)
        self.assertNotIn('frozen.run_full(',s)
        self.assertNotIn('torch.cuda.empty_cache(',s)
    def test_full_gpu_path_not_entered_when_not_approved(self):
        with patch.dict(os.environ,{},clear=True),patch.object(r,'_run_measured',side_effect=AssertionError('MUST_NOT_RUN')):
            with self.assertRaises(r.BoundedBlocked):
                r.main(['--run-bounded'])
    def test_gpu_memory_consistency_with_injected_mock(self):
        class Cuda:
            @staticmethod
            def synchronize(dev): pass
            @staticmethod
            def mem_get_info(dev): return (15*1024**3,16*1024**3)
            @staticmethod
            def memory_allocated(i): return 200
            @staticmethod
            def memory_reserved(i): return 300
            @staticmethod
            def max_memory_allocated(i): return 400
            @staticmethod
            def max_memory_reserved(i): return 500
        class T:
            cuda=Cuda
            @staticmethod
            def device(x): return x
        obs=[];r.physical_sample(T,'test',samples=obs,min_free=14*1024**3)
        self.assertEqual(obs[0]['driver_free_bytes'],15*1024**3)
        with self.assertRaisesRegex(r.BoundedBlocked,'INSUFFICIENT_GPU_FREE_AT'):
            r.physical_sample(T,'test2',samples=obs,min_free=16*1024**3)
    def test_execution_uses_pins_not_moving_refs(self):
        self.assertRegex(r.PREDECESSOR,r'^[0-9a-f]{40}$')
        self.assertEqual(len(r.FROZEN),7)

if __name__=='__main__': unittest.main()
