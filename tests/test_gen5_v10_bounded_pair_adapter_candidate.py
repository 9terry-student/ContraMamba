#!/usr/bin/env python3
"""CPU-only bounded paired adapter: genuine frozen integration where installed."""
from __future__ import annotations

import copy
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from scripts import gen5_v10_bounded_pair_adapter_candidate as adapter
from test_gen5_v10_hybrid_candidate import (
    FrozenStub, PARENTS, CLASSES, PAIRS, exact_causal_oracle, input_fixture,
)


class ReplaySpy:
    def __init__(self, fine_masks):
        self.fine_masks = fine_masks
        self.coarse_calls = 0
        self.replay_names = []

    def _component_region_pair_stats(self, **kw):
        self.coarse_calls += 1
        oracle = exact_causal_oracle(
            component=kw['component'], log_a=kw['log_a'],
            survival=kw['survival'], attention_mask=kw['attention_mask'],
            fine_masks=self.fine_masks, shape=kw['shape'],
        )
        rows = []
        for k in range(4):
            parent = {}
            for name in PARENTS:
                values = torch.zeros(128, dtype=torch.float64)
                for cell in PAIRS:
                    values += oracle[name][cell][k]
                parent[name] = values[1:].tolist()
            rows.append({'R': 1.0, 'S': 1.0, 'E': 0.5,
                         'region_pair_C_by_gap': parent})
        return rows

    def _replay_vector(self, *, name, observed, expected):
        self.replay_names.append(name)
        assert len(observed) == len(expected)
        mx_a = 0.0
        mx_r = 0.0
        for obs, exp in zip(observed, expected):
            a = abs(obs-exp)
            r = a/max(abs(exp), 1e-30)
            if not (a <= 2e-5 or r <= 2e-5):
                raise RuntimeError('VALIDATED_PAIR_GAP_VECTOR_REPLAY:' + name)
            mx_a=max(mx_a,a);mx_r=max(mx_r,r)
        return {'pass':True,'max_abs_error':mx_a,'max_rel_error':mx_r}


class SyntheticAudit(FrozenStub):
    FFT_CHUNK_WIDTH = 256

    def __init__(self, fine_masks):
        super().__init__()
        self.coarse = ReplaySpy(fine_masks)


def synth_fixture(batch=32, kind='mixed', width=8):
    assert width == 8
    kw = input_fixture(batch=batch, width=width, kind=kind, seed=73)
    attention = kw['attention_mask']
    seq = kw['component'].shape[2]
    claim = torch.arange(seq)[None, :] < 50
    evidence = torch.arange(seq)[None, :] > 50
    kw['role_masks']={
        'CLAIM': claim.expand(batch,seq) & attention,
        'EOS': (torch.arange(seq)[None,:] == 50).expand(batch,seq) & attention,
        'EVIDENCE': evidence.expand(batch,seq) & attention,
    }
    return kw


def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        return all(compare(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return len(a)==len(b) and all(compare(x,y) for x,y in zip(a,b))
    return a==b


class TestBoundedAdapterCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''

    def _run(self,b=32,kind='mixed'):
        kw=synth_fixture(batch=b,kind=kind)
        audit=SyntheticAudit(kw['fine_masks'])
        before=kw['component'].clone()
        result=adapter.paired_engine_records(audit,**kw,synthetic_cpu_geometry=True)
        self.assertTrue(torch.equal(before,kw['component']))
        self.assertEqual(audit.coarse.coarse_calls,1)
        self.assertEqual(len(audit.coarse.replay_names),24)
        self.assertEqual(len(result['frozen_records']),4)
        self.assertEqual(len(result['v10_records']),4)
        self.assertEqual(result['V10_E2_PAYLOAD'],'NOT_EVALUATED')
        self.assertEqual(result['PRODUCTION'],'BLOCKED')
        self.assertEqual(result['SCIENTIFIC_EXECUTION'],'NOT_AUTHORIZED')
        self.assertTrue(all('fine_parent_reconstruction_rel_max' in x
                            for x in result['v10_records']))
        return result

    def test_mixed_32_cpu(self):
        x=self._run(32,'mixed')
        self.assertEqual((x['v10_route']['fast_chunks'], x['v10_route']['gemm_chunks']),(1,1))
        for old,new in zip(x['frozen_records'],x['v10_records']):
            for parent in PARENTS:
                for cell in PAIRS:
                    a=old['fine_pair_C_by_gap'][parent][cell]
                    b=new['fine_pair_C_by_gap'][parent][cell]
                    for obs,exp in zip(a,b):
                        d=abs(obs-exp)
                        self.assertTrue(d<=2e-5 or d/max(abs(obs),1e-30)<=2e-5)

    def test_tail_8_original_frozen(self):
        x=self._run(8,'mixed')
        self.assertEqual(x['v10_route']['frozen_tail_calls'],1)
        self.assertTrue(compare(x['frozen_records'],x['v10_records']))

    def test_all_fast_32(self):
        x=self._run(32,'fast')
        self.assertEqual((x['v10_route']['fast_chunks'],x['v10_route']['gemm_chunks']),(2,0))

    def test_all_gemm_32(self):
        x=self._run(32,'large')
        self.assertEqual((x['v10_route']['fast_chunks'],x['v10_route']['gemm_chunks']),(0,2))

    def test_wrong_batch_rejected(self):
        kw=synth_fixture(batch=16)
        with self.assertRaisesRegex(adapter.BoundedAdapterBlocked,'K_BATCH_SEQUENCE'):
            adapter.paired_engine_records(SyntheticAudit(kw['fine_masks']),**kw,synthetic_cpu_geometry=True)

    def test_production_width_rejected(self):
        kw=synth_fixture()
        with self.assertRaisesRegex(adapter.BoundedAdapterBlocked,'PRODUCTION_WIDTH'):
            adapter.paired_engine_records(SyntheticAudit(kw['fine_masks']),**kw)

    def test_nan_rejected(self):
        kw=synth_fixture()
        kw['component'][0,0,0,0]=float('nan')
        with self.assertRaisesRegex(adapter.BoundedAdapterBlocked,'INPUT_NONFINITE'):
            adapter.paired_engine_records(SyntheticAudit(kw['fine_masks']),**kw,synthetic_cpu_geometry=True)

    def test_overlap_rejected(self):
        kw=synth_fixture()
        kw['fine_masks']['CLAIM']['A_NAME'] |= kw['fine_masks']['CLAIM']['A_TITLE']
        with self.assertRaisesRegex(adapter.BoundedAdapterBlocked,'FINE_MASK_OVERLAP'):
            adapter.paired_engine_records(SyntheticAudit(kw['fine_masks']),**kw,synthetic_cpu_geometry=True)

    def test_parent_tamper_blocked(self):
        kw=synth_fixture(batch=8)
        audit=SyntheticAudit(kw['fine_masks'])
        original=audit._fine_pair_gap_interference_fft
        def tampered(**args):
            g,span=original(**args)
            g[PARENTS[0]][PAIRS[0]][:,1] += 1.0
            return g,span
        audit._fine_pair_gap_interference_fft=tampered
        with self.assertRaisesRegex(RuntimeError,'VALIDATED_PAIR_GAP_VECTOR_REPLAY'):
            adapter.paired_engine_records(audit,**kw,synthetic_cpu_geometry=True)

    def test_missing_cell_blocked(self):
        kw=synth_fixture(batch=8)
        audit=SyntheticAudit(kw['fine_masks'])
        original=audit._fine_pair_gap_interference_fft
        def tampered(**args):
            g,span=original(**args)
            g[PARENTS[0]].pop(PAIRS[-1])
            return g,span
        audit._fine_pair_gap_interference_fft=tampered
        with self.assertRaisesRegex(adapter.BoundedAdapterBlocked,'FROZEN_FINE_KEYS'):
            adapter.paired_engine_records(audit,**kw,synthetic_cpu_geometry=True)

    def test_wrong_output_shape_blocked(self):
        kw=synth_fixture(batch=8)
        audit=SyntheticAudit(kw['fine_masks'])
        original=audit._fine_pair_gap_interference_fft
        def tampered(**args):
            g,span=original(**args)
            g[PARENTS[0]][PAIRS[0]]=torch.zeros((4,127),dtype=torch.float64)
            return g,span
        audit._fine_pair_gap_interference_fft=tampered
        with self.assertRaisesRegex(adapter.BoundedAdapterBlocked,'FROZEN_GAP_SHAPE_DTYPE'):
            adapter.paired_engine_records(audit,**kw,synthetic_cpu_geometry=True)

    def test_no_execution_entrypoint(self):
        text=Path(adapter.__file__).read_text('utf-8')
        self.assertNotIn('def run_worker(',text)
        self.assertNotIn('def run_full(',text)
        self.assertNotIn('subprocess.',text)
        self.assertNotIn('torch.cuda.',text)


class TestRealFrozenModuleCPU(unittest.TestCase):
    """Actual source integration, skipped only in standalone bundle without repo."""

    @classmethod
    def setUpClass(cls):
        script=Path(__file__).resolve().parents[1] / 'scripts' / 'audit_native_recurrent_generator_anchor_pair_localization.py'
        if not script.is_file():
            raise unittest.SkipTest('Frozen audit module is not in standalone package')
        from scripts import audit_native_recurrent_generator_anchor_pair_localization as frozen
        cls.frozen=frozen

    def fixture(self,b):
        from types import SimpleNamespace
        fine=self.frozen
        torch.manual_seed(6163+b)
        seq,width,k=128,8,4
        att=torch.zeros(b,seq,dtype=torch.bool)
        roles={key:torch.zeros(b,seq,dtype=torch.bool) for key in ('CLAIM','EOS','EVIDENCE')}
        classes=torch.randint(0,5,(b,seq))
        masks={side:{cell:torch.zeros(b,seq,dtype=torch.bool) for cell in fine.FINE_CLASSES}
               for side in ('CLAIM','EVIDENCE')}
        for i in range(b):
            end=117+i%9
            att[i,:end]=True;roles['CLAIM'][i,:52]=True
            roles['EOS'][i,52]=True;roles['EVIDENCE'][i,53:end]=True
        for side in ('CLAIM','EVIDENCE'):
            for ci,cell in enumerate(fine.FINE_CLASSES):
                masks[side][cell]=roles[side] & (classes==ci)
        component=(.002*torch.randn(k,b,seq,width)).to(torch.float32)
        component*=att[None,:,:,None]
        log_a=torch.full((b,seq,1,width),-.04,dtype=torch.float32)
        log_a[:,:,:,4:]=-1.5
        survival=fine.coarse.pairgap._backward_survival_factor(
            log_a_flat=log_a.reshape(b,seq,width),attention_mask=att)
        shape=SimpleNamespace(intermediate_size=1,state_size=width,state_width=width)
        return dict(component=component,log_a=log_a,survival=survival,
                    attention_mask=att,role_masks=roles,fine_masks=masks,shape=shape)

    def test_real_frozen_outer_b32(self):
        self._run(32)

    def test_real_frozen_outer_b8(self):
        self._run(8)

    def _run(self,b):
        kw=self.fixture(b)
        frozen=self.frozen
        original=frozen._component_fine_pair_stats(**kw)
        original_fine=frozen._fine_pair_gap_interference_fft
        result=adapter.paired_engine_records(frozen,**kw,synthetic_cpu_geometry=True)
        self.assertIs(frozen._fine_pair_gap_interference_fft,original_fine)
        self.assertEqual(original,result['frozen_records'])
        self.assertEqual(len(result['v10_records']),4)
        self.assertEqual(result['v10_route']['mode'],
                         'CHUNKWISE_HYBRID' if b==32 else 'FROZEN_TAIL8')
        if b==8:
            self.assertEqual(original,result['v10_records'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
