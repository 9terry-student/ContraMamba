"""CPU synthetic tests for Stage B seam and canonical chunk capture; no GPU."""
from __future__ import annotations

import copy
import hashlib
import os
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from scripts import gen5_v10_bounded_worker_export_candidate as w


class TestBoundedStaticWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_static_only_cli_is_not_scientific(self):
        self.assertEqual(w.SCHEMA, 'GEN5_V10_BOUNDED_PAIRED_EXPORT_V2')
        self.assertFalse(hasattr(w, 'run_full'))
        self.assertFalse(hasattr(w, 'run_bounded'))

    def sources(self):
        return w.SOURCE_CELLS, w.FROZEN_ORIENTATIONS

    def select(self,**replace):
        sources,orientations=self.sources()
        kw=dict(worker_id=1,batch_index=13,row_start=800,row_stop=832,
                row_ids=[f'row-{i}' for i in range(800,832)],source_cells=sources,
                orientations=orientations,selected_source=sources[0],
                selected_targets=((6201,6202),(6201,6203)),target_group_index=0)
        kw.update(replace)
        return w.choose_bounded_locus(**kw)

    def test_first_source_target_and_rows(self):
        d=self.select()
        self.assertEqual(d['worker_id'],1)
        self.assertEqual(d['source_cell'],'6201_6201')
        self.assertEqual(d['target_cells'], ['6201_6202','6201_6203'])
        self.assertEqual(len(d['row_ids']),32)

    def test_wrong_batch_blocks(self):
        with self.assertRaisesRegex(w.StageBBlocked,'BATCH_ID'):self.select(batch_index=14)

    def test_wrong_shard_blocks(self):
        with self.assertRaisesRegex(w.StageBBlocked,'ROW_INTERVAL'):self.select(row_start=832)

    def test_second_group_blocks(self):
        with self.assertRaisesRegex(w.StageBBlocked,'FIRST_GROUP_ONLY'):self.select(target_group_index=1)

    def test_wrong_source_blocks(self):
        with self.assertRaisesRegex(w.StageBBlocked,'SOURCE_NOT_FIRST'):self.select(selected_source=(6201,6202))

    def test_wrong_target_order_blocks(self):
        with self.assertRaisesRegex(w.StageBBlocked,'TARGET_GROUP_IDENTITY'):
            self.select(selected_targets=((6201,6203),(6201,6202)))

    def test_unfrozen_source_enumeration_blocks(self):
        cells,_=self.sources()
        with self.assertRaisesRegex(w.StageBBlocked,'SOURCE_ORDER'):
            self.select(source_cells=tuple(reversed(cells)))

    def test_unfrozen_orientation_order_blocks(self):
        _,rows=self.sources()
        with self.assertRaisesRegex(w.StageBBlocked,'ORIENTATION_ORDER'):
            self.select(orientations=tuple(reversed(rows)))

    def test_duplicate_row_ids_blocks(self):
        with self.assertRaisesRegex(w.StageBBlocked,'ROW_IDS'):
            self.select(row_ids=['x']*32)

    def test_stream_matches_stage_a_bytes_on_float(self):
        a=torch.randn(2,3,7,dtype=torch.float32)
        got=w.capture_value_digest(a,phase='PRE_FROZEN',max_chunk_bytes=16)
        expected=hashlib.sha256(a.numpy().astype('<f4').tobytes(order='C')).hexdigest()
        self.assertEqual(got['content_sha256'],expected)
        self.assertEqual(got['byte_count'],a.numel()*4)

    def test_stream_noncontiguous_true_logical_order(self):
        a=torch.randn(3,2,18,dtype=torch.float32).transpose(0,1)
        self.assertFalse(a.is_contiguous())
        got=w.capture_value_digest(a,phase='POST_FROZEN',max_chunk_bytes=32)
        exp=hashlib.sha256(a.contiguous().numpy().astype('<f4').tobytes(order='C')).hexdigest()
        self.assertEqual(got['content_sha256'],exp)

    def test_stream_bool_01(self):
        a=torch.tensor([[True,False,True],[False,True,False]],dtype=torch.bool)
        got=w.capture_value_digest(a,phase='POST_V10',max_chunk_bytes=4)
        exp=hashlib.sha256(bytes([1,0,1,0,1,0])).hexdigest()
        self.assertEqual(got['content_sha256'],exp)

    def test_byte_tamper_digest_changes(self):
        a=torch.ones((4,8),dtype=torch.float32)
        x=w.capture_value_digest(a,phase='PRE_FROZEN',max_chunk_bytes=20)
        a[0,0]=-1.0
        y=w.capture_value_digest(a,phase='POST_FROZEN',max_chunk_bytes=20)
        self.assertNotEqual(x['content_sha256'],y['content_sha256'])

    def test_bad_capture_phase_blocks(self):
        with self.assertRaisesRegex(w.StageBBlocked,'CAPTURE_PHASE'):
            w.capture_value_digest(torch.zeros(2,2),phase='GUESSED')

    def test_cpu_fake_paired_seam_blocks_drift(self):
        from test_gen5_v10_bounded_tensor_provenance_candidate import TestStageACanonicalProvenance
        kw=TestStageACanonicalProvenance().fixture(b=32)
        inputs={k:kw[k] for k in ('component','log_a','survival','attention_mask','role_masks','fine_masks')}
        calls=[]
        def old(**data):
            calls.append('frozen')
            return [{'S':1.0}]*4
        def new(**data):
            calls.append('v10')
            return [{'S':1.0}]*4
        result=w.cpu_stub_paired_capture(inputs=inputs,shape=kw['shape'],frozen_fn=old,v10_fn=new)
        self.assertEqual(calls,['frozen','v10'])
        self.assertTrue(result['capture_equal'])
        self.assertEqual(result['V10_E0_PROVENANCE'],'NOT_EVALUATED')
        def mutator(**data):
            data['component'][0,0,0,0]+=1.0
            return [{}]*4
        with self.assertRaisesRegex(w.StageBBlocked,'FROZEN_MUTATED_INPUTS'):
            w.cpu_stub_paired_capture(inputs=inputs,shape=kw['shape'],frozen_fn=mutator,v10_fn=new)

    def test_v10_mutation_blocks(self):
        from test_gen5_v10_bounded_tensor_provenance_candidate import TestStageACanonicalProvenance
        kw=TestStageACanonicalProvenance().fixture(b=32)
        inputs={k:kw[k] for k in ('component','log_a','survival','attention_mask','role_masks','fine_masks')}
        def old(**data):return [{}]*4
        def new(**data):
            data['attention_mask'][0,1]=False
            return [{}]*4
        with self.assertRaises(Exception):
            w.cpu_stub_paired_capture(inputs=inputs,shape=kw['shape'],frozen_fn=old,v10_fn=new)


if __name__ == '__main__':unittest.main()
