#!/usr/bin/env python3
"""E2 local CPU byte measurements: fixtures must never become provenance claims."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from scripts import gen5_v10_stage_e2_cpu_source_byte_verifier_candidate as e2


class TestE2LocalSourceBytes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def fixture():
        n, s = 840, 128
        row = []
        for i in range(n):
            row.append({"id":f"internal-{i}","pair_id":f"pair-{i//7}",
                        "row_id":f"dev-row-{i}","source_pair_id":f"pair-{i//7}",
                        "contrast_cell_id":"C1", "final_label":"REFUTE",
                        "final_label_id":0, "stressor_domain":i % 2 == 0})
        ids=torch.zeros((n,s),dtype=torch.int64)
        att=torch.zeros((n,s),dtype=torch.bool)
        claim=torch.zeros_like(att)
        ev=torch.zeros_like(att)
        for i in range(n):
            width=72 if i < 454 else 71
            att[i,:width]=True
            claim[i,:30]=True
            ev[i,31:width]=True
            ids[i,:30]=42
            ids[i,31:width]=43
        active=torch.tensor([r['stressor_domain'] for r in row],dtype=torch.bool)
        targets=torch.tensor([2 if r['stressor_domain'] else -1 for r in row],dtype=torch.int64)
        bundle={"model_inputs":{"input_ids":ids,"attention_mask":att,"claim_mask":claim,
                 "evidence_mask":ev,"final_labels":torch.zeros(n,dtype=torch.int64)},
                "stressor_active":active,"target_indices":targets,
                "row_ids":[r['row_id'] for r in row],
                "pair_ids":[r['source_pair_id'] for r in row],
                "contrast_cell_ids":[r['contrast_cell_id'] for r in row]}
        return row,bundle

    @classmethod
    def fixture_files(cls,tmp):
        root=Path(tmp)
        cp=root/'parent.pt';cp.write_bytes(b'CPU-FAKE-CHECKPOINT')
        snap=root/e2.MODEL_REVISION;snap.mkdir()
        files={"config.json":b'{"cpu":true}',"tokenizer.json":b'{"not":"a-tokenizer"}',
               "tokenizer_config.json":b'{"x":1}',"special_tokens_map.json":b'{"y":2}'}
        for k,v in files.items():(snap/k).write_bytes(v)
        pins={k:(len(v) if k=='config.json' else None,hashlib.sha256(v).hexdigest()) for k,v in files.items()}
        return cp,snap,pins

    def test_p3a_row_order_exact_canonical_algorithm(self):
        rows,_=self.fixture()
        expected=[]
        for r in rows:
            expected.append({"id":r['id'],"pair_id":r['pair_id'],
                  "contrast_cell_id":r['contrast_cell_id'],"final_label":r['final_label'],
                  "final_label_id":r['final_label_id'],"stressor_domain":r['stressor_domain']})
        digest=hashlib.sha256(e2.canonical_bytes(expected)).hexdigest()
        self.assertEqual(e2.dev_order_digest(rows),digest)

    def test_encoded_canonical_digest_deterministic(self):
        r,b=self.fixture();e2.validate_encoded_cpu_bundle(b,r)
        self.assertEqual(e2.encoded_bundle_digest(b),e2.encoded_bundle_digest(b))
        self.assertEqual(int(b['model_inputs']['attention_mask'].sum()),60094)

    def test_actual_file_hash_measures_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            cp,_,_=self.fixture_files(td)
            sha,count=e2._actual_file_sha256(cp)
            self.assertEqual(count,len(b'CPU-FAKE-CHECKPOINT'))
            self.assertEqual(sha,hashlib.sha256(cp.read_bytes()).hexdigest())

    def test_wrong_parent_file_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            cp,_,_=self.fixture_files(td)
            with self.assertRaisesRegex(e2.E2Blocked,'CHECKPOINT_REAL_BYTE_DIGEST_MISMATCH'):
                e2.verify_checkpoint_file(cp)

    def test_symlink_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            cp,_,_=self.fixture_files(td)
            # Deterministic across unprivileged Windows/Linux: exercise the
            # symlink refusal even if OS symlink creation is not permitted.
            with patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaisesRegex(e2.E2Blocked,'SOURCE_SYMLINK_FORBIDDEN'):
                    e2.verify_checkpoint_file(cp)

    def test_missing_checkpoint_rejected(self):
        with self.assertRaisesRegex(e2.E2Blocked,'SOURCE_NOT_REGULAR_FILE'):
            e2.verify_checkpoint_file(Path('/definitely-missing-e2-checkpoint.pt'))

    def test_snapshot_actual_bytes_pinned(self):
        with tempfile.TemporaryDirectory() as td:
            _,snap,pins=self.fixture_files(td)
            with patch.object(e2,'SNAPSHOT_FILES',pins):
                r=e2.verify_snapshot_bytes(snap)
            self.assertEqual(set(r['verified_files']),set(pins))
            self.assertFalse(r['tokenizer_loaded'])

    def test_snapshot_tamper_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            _,snap,pins=self.fixture_files(td)
            (snap/'tokenizer.json').write_bytes(b'tampered')
            with patch.object(e2,'SNAPSHOT_FILES',pins):
                with self.assertRaisesRegex(e2.E2Blocked,'SNAPSHOT_REAL_BYTE_DIGEST'):
                    e2.verify_snapshot_bytes(snap)

    def test_snapshot_config_length_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            _,snap,pins=self.fixture_files(td)
            pins['config.json']=(0,pins['config.json'][1])
            with patch.object(e2,'SNAPSHOT_FILES',pins):
                with self.assertRaisesRegex(e2.E2Blocked,'SNAPSHOT_SIZE'):
                    e2.verify_snapshot_bytes(snap)

    def test_snapshot_revision_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            _,snap,_=self.fixture_files(td)
            with patch.object(e2,'MODEL_REVISION','wrong'):
                with self.assertRaisesRegex(e2.E2Blocked,'MODEL_SNAPSHOT_REVISION'):
                    e2.verify_snapshot_bytes(snap)

    def test_dev_row_order_pin(self):
        r,_=self.fixture()
        with patch.object(e2,'DEV_ORDER_SHA256',e2.dev_order_digest(r)):
            out=e2.verify_dev_rows(r)
        self.assertEqual(out['bounded_row_ids'][0],'dev-row-800')
        self.assertEqual(len(out['bounded_row_ids']),32)

    def test_dev_row_order_mismatch_blocks(self):
        r,_=self.fixture()
        with self.assertRaisesRegex(e2.E2Blocked,'DEV_ORDER_CANONICAL_DIGEST_MISMATCH'):
            e2.verify_dev_rows(r)

    def test_dev_row_reorder_blocks(self):
        r,_=self.fixture();baseline=e2.dev_order_digest(r)
        r[0],r[1]=r[1],r[0]
        with patch.object(e2,'DEV_ORDER_SHA256',baseline):
            with self.assertRaisesRegex(e2.E2Blocked,'DEV_ORDER_CANONICAL_DIGEST_MISMATCH'):
                e2.verify_dev_rows(r)

    def test_dev_duplicate_row_ids_blocks(self):
        r,_=self.fixture();r[1]['row_id']=r[0]['row_id']
        with self.assertRaisesRegex(e2.E2Blocked,'DUPLICATE_DEV_ROW_IDS'):
            e2.verify_dev_rows(r)

    def test_bundle_digest_pin(self):
        r,b=self.fixture()
        with patch.object(e2,'DEV_ENCODING_SHA256',e2.encoded_bundle_digest(b)):
            d=e2.verify_encoded_cpu_bundle(b,r)
        self.assertEqual(d['checked_cpu_tensors'],7)
        self.assertEqual(d['tokenizer_invocations'],0)

    def test_encoded_token_value_change_detected_by_digest(self):
        rows,bundle=self.fixture()
        original=e2.encoded_bundle_digest(bundle)
        bundle["model_inputs"]["input_ids"][0,0]=91
        self.assertNotEqual(original,e2.encoded_bundle_digest(bundle))
        with patch.object(e2, 'DEV_ENCODING_SHA256', original):
            with self.assertRaisesRegex(e2.E2Blocked,'ENCODED_CANONICAL_DIGEST_MISMATCH'):
                e2.verify_encoded_cpu_bundle(bundle,rows)

    def test_bundle_digest_mismatch_blocks(self):
        r,b=self.fixture()
        with self.assertRaisesRegex(e2.E2Blocked,'ENCODED_CANONICAL_DIGEST_MISMATCH'):
            e2.verify_encoded_cpu_bundle(b,r)

    def test_encoded_target_value_change_detected_by_digest(self):
        rows,bundle=self.fixture()
        original=e2.encoded_bundle_digest(bundle)
        bundle["target_indices"][0]=3
        self.assertNotEqual(original,e2.encoded_bundle_digest(bundle))
        with patch.object(e2,'DEV_ENCODING_SHA256',original):
            with self.assertRaisesRegex(e2.E2Blocked,'ENCODED_CANONICAL_DIGEST_MISMATCH'):
                e2.verify_encoded_cpu_bundle(bundle,rows)

    def test_bundle_row_id_drift_blocks_before_hash(self):
        r,b=self.fixture();b['row_ids'][800]='forged'
        with self.assertRaisesRegex(e2.E2Blocked,'ENCODED_ROW_LINEAGE_INCONSISTENT'):
            e2.verify_encoded_cpu_bundle(b,r)

    def test_bundle_cpu_only(self):
        r,b=self.fixture()
        b['model_inputs']['input_ids']=b['model_inputs']['input_ids'].to(torch.float32)
        with self.assertRaisesRegex(e2.E2Blocked,'CPU_ENCODED_TENSOR'):
            e2.verify_encoded_cpu_bundle(b,r)

    def test_bundle_target_mismatch_blocks(self):
        r,b=self.fixture();b['target_indices'][0]=128
        with self.assertRaisesRegex(e2.E2Blocked,'ENCODED_TARGET_INDEX'):
            e2.verify_encoded_cpu_bundle(b,r)

    def test_bundle_eos_shift_blocks(self):
        r,b=self.fixture();b['model_inputs']['input_ids'][0,30]=5
        with self.assertRaisesRegex(e2.E2Blocked,'ENCODED_SERIALIZATION_LAYOUT'):
            e2.verify_encoded_cpu_bundle(b,r)

    def test_bundle_total_tokens_blocks(self):
        r,b=self.fixture();b['model_inputs']['attention_mask'][0,80]=True
        with self.assertRaises(e2.E2Blocked):
            e2.verify_encoded_cpu_bundle(b,r)

    def test_bundle_mask_overlap_blocks(self):
        r,b=self.fixture();b['model_inputs']['claim_mask'][0,31]=True
        with self.assertRaisesRegex(e2.E2Blocked,'ENCODED_MASK_OVERLAP'):
            e2.verify_encoded_cpu_bundle(b,r)

    def test_bundle_extra_keys_blocks(self):
        r,b=self.fixture();b['provenance']='PASS'
        with self.assertRaisesRegex(e2.E2Blocked,'ENCODED_BUNDLE_EXACT_KEYS'):
            e2.verify_encoded_cpu_bundle(b,r)

    def test_all_four_sources_real_bytes_cpu_floor(self):
        r,b=self.fixture()
        with tempfile.TemporaryDirectory() as td:
            cp,snap,pins=self.fixture_files(td)
            with patch.object(e2,'CHECKPOINT_SHA256',hashlib.sha256(cp.read_bytes()).hexdigest()),\
                 patch.object(e2,'SNAPSHOT_FILES',pins),\
                 patch.object(e2,'DEV_ORDER_SHA256',e2.dev_order_digest(r)),\
                 patch.object(e2,'DEV_ENCODING_SHA256',e2.encoded_bundle_digest(b)),\
                 patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
                result=e2.audit_local_source_bytes(checkpoint=cp,snapshot=snap,dev_rows=r,dev_bundle=b)
        self.assertTrue(result['local_source_bytes_match_pins'])
        self.assertEqual(len(result['bounded_row_ids']),32)
        for k in ('GPU_EXECUTION','SCIENTIFIC_EXECUTION'):
            self.assertEqual(result[k],'NOT_AUTHORIZED')
        for k in ('E0','E1','E2','E3'):
            self.assertEqual(result[k],'NOT_EVALUATED')
        self.assertFalse(result['active_gpu_tensor_lineage_authenticated'])
        self.assertFalse(result['checkpoint_bytes_loaded_into_model_authenticated'])

    def test_cuda_visibility_blocks_without_loading(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(e2.E2Blocked,'GPU_MUST_BE_HIDDEN'):
                e2.audit_local_source_bytes(checkpoint=Path('bad'),snapshot=Path('bad'),dev_rows=[],dev_bundle={})

    def test_cli_static_without_reading_sources(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),\
             patch.object(sys,'argv',['e2','--static-contract-check']),\
             patch.object(e2,'inspect_pinned_git') as git,\
             contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(e2.main(),0)
            self.assertIn('MODEL_LOAD=NO',stream.getvalue())
            git.assert_not_called()

    def test_cli_unknown_execution_rejected(self):
        with patch.object(sys,'argv',['e2','--run-bounded']),\
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):e2.main()

    def test_cli_cuda_visible_rejected(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),\
             patch.object(sys,'argv',['e2','--static-contract-check']):
            with self.assertRaisesRegex(e2.E2Blocked,'GPU_MUST_BE_HIDDEN'):
                e2.main()

    def test_pinned_git_all_five_blobs(self):
        head=e2.BASE
        def fake_run(argv, **kwargs):
            args=argv[3:]
            if args == ['rev-parse', 'HEAD']:
                output=head
            elif args == ['branch', '--show-current']:
                output=e2.BRANCH
            elif args in (['diff','--name-only'],['diff','--cached','--name-only']):
                output=''
            elif args[0]=='rev-parse' and len(args)==2 and args[1].startswith('HEAD:'):
                output=e2.FROZEN_BLOBS[args[1][5:]]
            else:
                self.fail('unexpected Git command '+str(args))
            return type('R',(),{'returncode':0,'stdout':output+'\n'})()
        with tempfile.TemporaryDirectory() as td, patch.object(e2.subprocess,'run',side_effect=fake_run):
            result=e2.inspect_pinned_git(Path(td))
        self.assertEqual(result['pinned_blobs'],5)
        self.assertFalse(result['loaded_runtime_modules_authenticated'])

    def test_git_read_failure_blocks(self):
        with tempfile.TemporaryDirectory() as td, patch.object(e2.subprocess,'run',side_effect=OSError('git')):
            with self.assertRaisesRegex(e2.E2Blocked,'GIT_READ_ERROR'):
                e2.inspect_pinned_git(Path(td))

    def test_git_head_mismatch_blocks(self):
        with tempfile.TemporaryDirectory() as td, patch.object(e2.subprocess,'run') as run:
            run.return_value.returncode=0
            run.return_value.stdout='0'*40+'\n'
            with self.assertRaisesRegex(e2.E2Blocked,'GIT_HEAD'):
                e2.inspect_pinned_git(Path(td))


if __name__ == '__main__':unittest.main()
