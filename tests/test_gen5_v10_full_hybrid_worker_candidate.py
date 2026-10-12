import unittest
from importlib import import_module
from unittest.mock import MagicMock
from pathlib import Path
import os
from types import SimpleNamespace

m = import_module('scripts.gen5_v10_full_hybrid_worker_candidate')

class TestFullV10Cpu(unittest.TestCase):
    def test_fixed_counts(self):
        self.assertEqual(m.tally_start(0)['expected_groups'],234)
        self.assertEqual(m.tally_start(1)['expected_groups'],252)
        self.assertRaises(m.FullBlocked,m.tally_start,2)

    def test_full_b32_tally(self):
        t=m.tally_start(0)
        t.update(calls=234,b32_calls=234,gemm_chunks=234*96)
        m.verify_tally(t)

    def test_tail8_tally(self):
        t=m.tally_start(1)
        t.update(calls=252,b32_calls=234,b8_calls=18,gemm_chunks=234*96,
                 frozen_tail_calls=18)
        m.verify_tally(t)

    def test_missing_tail8_blocks(self):
        t=m.tally_start(1)
        t.update(calls=252,b32_calls=234,b8_calls=18,gemm_chunks=234*96)
        self.assertRaises(m.FullBlocked,m.verify_tally,t)

    def test_96_chunk_coverage(self):
        t=m.tally_start(0)
        t.update(calls=234,b32_calls=234,gemm_chunks=234*95)
        self.assertRaises(m.FullBlocked,m.verify_tally,t)

    def test_worker_gate_before_import(self):
        old=os.environ.pop('CONTRAMAMBA_GEN5_V10_FULL_GPU_APPROVED',None)
        try:
            a=SimpleNamespace(approval='I_APPROVE_GEN5_V10_FULL_GPU')
            self.assertRaises(m.FullBlocked,m.run_worker,a)
        finally:
            if old is not None:os.environ['CONTRAMAMBA_GEN5_V10_FULL_GPU_APPROVED']=old

    def test_bad_sha(self):
        self.assertRaises(m.FullBlocked,m.verify_source,'abc')

    def test_no_worker_tag_confusion(self):
        self.assertNotIn('EXACT_INTERFERENCE_V1',m.ENGINE)
        self.assertIn('CANDIDATE',m.SCHEMA)

    def test_import_no_gpu(self):
        self.assertNotIn('torch',m.__dict__)

    def test_receipt_sha(self):
        import tempfile,json,hashlib
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'receipt.json'
            m.write_json_with_sha(p,{'status':'FAIL'})
            self.assertEqual(json.loads(p.read_text())['status'],'FAIL')
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),
                             p.with_suffix('.json.sha256').read_text().strip())
            self.assertRaises(m.FullBlocked,m.write_json_with_sha,p,{'status':'PASS'})

    def test_view_numerics_no_change_with_mocks(self):
        # Inject mocks; the wrapper must use coarse and hybrid, never original full FFT reducer.
        frozen=MagicMock()
        frozen.p3a=SimpleNamespace(MAX_LENGTH=128)
        frozen.FFT_CHUNK_WIDTH=256
        frozen.PARENT_PAIRS=(("CLAIM","CLAIM"),)
        frozen.PARENT_PAIR_NAMES=("CLAIM->CLAIM",)
        frozen.FINE_CLASSES=("A",)
        frozen.FINE_PAIR_NAMES=("A->A",)
        frozen.FAST_FLOAT32_HALF_SPAN_LIMIT=60.0
        frozen.coarse._component_region_pair_stats.return_value=[{'a':1}]*4
        hybrid=MagicMock()
        hybrid.hybrid_fine_pair_gap.return_value=({'x':3},300.0,
            {'batch_rows':32,'total_chunks':96,'gemm_chunks':96,'fast_chunks':0,'frozen_tail_calls':0})
        bridge=MagicMock()
        bridge._post_original_reducer.return_value=[{'r':1}]*4
        torch=MagicMock()
        torch.float32='float32'
        torch.cuda.max_memory_allocated.return_value=100
        torch.no_grad.return_value.__enter__.return_value=None
        t=m.tally_start(0)
        m.install_reducer(frozen,hybrid,bridge,torch,t)
        component=MagicMock()
        component.shape=(4,32,128,24576)
        component.is_cuda=True
        component.dtype='float32'
        ret=frozen._component_fine_pair_stats(component=component,log_a=MagicMock(),
            survival=MagicMock(),attention_mask=MagicMock(),role_masks={},fine_masks={},
            shape=MagicMock())
        self.assertEqual(len(ret),4)
        self.assertEqual(t['calls'],1)
        bridge._post_original_reducer.assert_called_once()
        hybrid.hybrid_fine_pair_gap.assert_called_once()
        frozen.coarse._component_region_pair_stats.assert_called_once()

if __name__=='__main__':unittest.main()
