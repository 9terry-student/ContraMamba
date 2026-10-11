#!/usr/bin/env python3
"""Stage C bridge: selected frozen seam mock, CPU-only negative and ordering tests."""
from __future__ import annotations

import copy
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from scripts import gen5_v10_bounded_runtime_bridge_candidate as c
from scripts import gen5_v10_hybrid_fine_pair_candidate as hybrid
from scripts import gen5_v10_bounded_worker_export_candidate as b
from test_gen5_v10_bounded_pair_adapter_candidate import SyntheticAudit, synth_fixture


class FrozenMock(SyntheticAudit):
    FULL_FACTORIAL_CELLS = b.SOURCE_CELLS
    TARGETS_PER_TRANSPORT = 2
    RAW_COMPONENT_MATCH_RTOL = 1e-12
    def __init__(self,kw,events):
        super().__init__(kw['fine_masks'])
        self.events=events;self.kw=kw
        self.RAW_COMPONENT_MATCH_RTOL=1e-12
        self._relative_error=lambda x,y:abs(x-y)/max(abs(y),1e-30)
        old_coarse = self.coarse._component_region_pair_stats
        def complete_coarse(**kwargs):
            rows = old_coarse(**kwargs)
            for rec in rows:
                rec.update({
                    'C': 0.0, 'Q': 0.0, 'pair_gap_C': [0.0]*127,
                    'pair_gap_C_reconstruction_abs': 0.0,
                    'pair_gap_Q_reconstruction_abs': 0.0,
                    'pair_gap_fft_half_log_span_max': 90.0,
                    'region_pair_gap_reconstruction_abs_max': 0.0,
                    'region_pair_gap_reconstruction_rel_max': 0.0,
                    'region_pair_fft_half_log_span_max': 90.0,
                })
                for region in ('CLAIM->EOS','EOS->EVIDENCE'):
                    rec['region_pair_C_by_gap'][region] = [0.0]*127
            return rows
        self.coarse._component_region_pair_stats = complete_coarse
        self.coarse._serialization_role_masks=self._role_masks
        self.coarse._pair_opportunity_counts=lambda _roles:self._event('coarse_counts',{})
        self._build_anchor_partition=self._anchor_partition
        self._batch_anchor_masks=self._batch_masks
        self._fine_pair_opportunity_counts=lambda masks,counts:self._event('fine_counts',{})
        self.transport=SimpleNamespace(
            RAW_RECON_ATOL=5e-5,
            legacy=SimpleNamespace(
                _phase_b_batch_features=self._features,
                _phase_b_prepare_common_context=self._context,
                _phase_b_raw_write=self._raw_write,
            ),
            _cell_weights=self._weights,
            _source_gradient_bundle=self._grad_bundle,
            _prepare_projector_pinv=self._pinv,
            _empty_orientation=self._acc,
            _project_visible_group=self._project,
        )
        self.kernel=SimpleNamespace(_discrete_log_a=self._log_a)
        self.pairgap=SimpleNamespace(_backward_survival_factor=self._survival)

    @staticmethod
    def all_orientations():return b.FROZEN_ORIENTATIONS
    @staticmethod
    def worker_row_range(worker):return {0:(0,416),1:(416,840)}[worker]

    def _event(self,label,result):
        self.events.append(label)
        return result
    def _features(self,features,start,stop):
        assert (start,stop)==(800,832)
        return self._event('batch_features',{'attention_mask':self.kw['attention_mask']})
    def _role_masks(self,_features):return self._event('roles',self.kw['role_masks'])
    def _anchor_partition(self,**kwargs):
        assert (kwargs['row_start'],kwargs['row_stop'])==(416,840)
        partition={side:{cell:torch.cat((torch.zeros(384,128,dtype=torch.bool),v))
                    for cell,v in cells.items()} for side,cells in self.kw['fine_masks'].items()}
        return self._event('anchor_partition',(partition,{'mock':True}))
    def _batch_masks(self,partition,*,local_start,local_stop,device):
        assert (local_start,local_stop)==(384,416)
        return self._event('fine_masks',{
            side:{cell:mask[384:416] for cell,mask in fields.items()}
            for side,fields in partition.items()})
    def _context(self,**kwargs):
        return self._event('context',{'mixer_input':torch.zeros(32,128,8)})
    def _log_a(self,context,*,shape):return self._event('log_a',self.kw['log_a'])
    def _survival(self,**kwargs):return self._event('survival',self.kw['survival'])
    def _weights(self):return self._event('weights',{k:(torch.tensor(1.),torch.tensor(1.))
                                                          for k in b.SOURCE_CELLS})
    def _raw_write(self,*args):return self._event('raw_write',torch.zeros(32,128,8))
    def _grad_bundle(self,**kw):return self._event('source_grad',('ignored','grad_r','grad_s'))
    def _pinv(self,**kw):return self._event('projector_pinv',('grad_r','grad_s','pinv'))
    def _acc(self,*row):
        return self._event('projector_acc',{
            'raw_reconstruction_max_abs':0.,
            'stage':{'raw_write':{'D_visible':1.,'D_complement':1.}}})
    def _project(self,**kw):
        assert len(kw['raw_targets'])==2 and len(kw['accumulators'])==2
        return self._event('projector',
                           (self.kw['component'][:2].clone(),self.kw['component'][2:].clone()))
    def _component_fine_pair_stats(self,**kwargs):
        self.events.append('original_full_fine_stats')
        coarse=self.coarse._component_region_pair_stats(
            **{k:kwargs[k] for k in ('component','log_a','survival',
                                     'attention_mask','role_masks','shape')})
        gap,span=self._fine_pair_gap_interference_fft(
            **{k:kwargs[k] for k in ('component','log_a','survival',
                                     'attention_mask','fine_masks','shape')})
        return c._post_original_reducer(self,coarse,gap,span,kwargs['component'])


class TestStageCBridgeCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        kw=synth_fixture()
        events=[]
        obj=FrozenMock(kw,events)
        ids=[f'ROW_{i}' for i in range(800,832)]
        return kw,obj,ids,events

    def test_selected_locus(self):
        _,obj,ids,_=self.fixture()
        sel=c.bounded_selector(obj,ids)
        self.assertEqual((sel['worker_id'],sel['batch_index_one_based']), (1,13))
        self.assertEqual(sel['target_cells'],['6201_6202','6201_6203'])

    def test_static_seam_b32_and_auth_floor(self):
        kw,obj,ids,events=self.fixture()
        selector=c.bounded_selector(obj,ids)
        x=c.paired_cpu_seam(frozen=obj,hybrid=hybrid,selector=selector,
                            source_row_ids=ids,**kw)
        self.assertEqual(len(x['frozen_records']),4)
        self.assertEqual(len(x['v10_records']),4)
        self.assertEqual(len(x['input_capture']['pre_frozen']),17)
        self.assertEqual(x['V10_E0_PROVENANCE'],'NOT_EVALUATED')
        self.assertEqual(x['V10_E1_FROZEN_GUARDS'],'NOT_EVALUATED')
        self.assertIn('original_full_fine_stats',events)
        self.assertEqual(obj.coarse.coarse_calls,2)
        self.assertEqual(len(obj.coarse.replay_names),24)
        self.assertEqual(x['v10_dispatch']['total_chunks'],2)

    def test_one_group_precursor_exact_call_order(self):
        _,obj,ids,events=self.fixture()
        result=c.construct_one_group_from_mocked_frozen_worker(
            frozen=obj,row_ids=ids,static={},encoded={},tokenizer_snapshot=None,
            model=SimpleNamespace(parameters=lambda:[]),
            wrapper=SimpleNamespace(correction=SimpleNamespace(shape=obj.kw['shape'])),
            strong_mask=None,planes=None,features=None,active=[0]*840,targets=[0]*840)
        self.assertEqual(result['selector']['source_cell'],'6201_6201')
        self.assertEqual(result['component'].shape,(4,32,128,8))
        self.assertEqual(events.count('raw_write'),9)
        self.assertEqual(events.count('source_grad'),1)
        self.assertEqual(events.count('projector'),1)
        self.assertLess(events.index('anchor_partition'),events.index('source_grad'))
        self.assertEqual(result['SCIENTIFIC_EXECUTION'],'NOT_AUTHORIZED')

    def test_complete_mock_worker_order(self):
        _,obj,ids,events=self.fixture()
        result=c.execute_one_group_cpu_mock(
            frozen=obj,hybrid=hybrid,row_ids=ids,static={},encoded={},tokenizer_snapshot=None,
            model=SimpleNamespace(parameters=lambda:[]),
            wrapper=SimpleNamespace(correction=SimpleNamespace(shape=obj.kw['shape'])),
            strong_mask=None,planes=None,features=None,active=[0]*840,targets=[0]*840)
        self.assertEqual(result['raw_projector_guard_status'],
                         'PASS_CPU_MOCK_ONLY_NOT_AUTHENTICATED')
        self.assertEqual(result['SCIENTIFIC_EXECUTION'],'NOT_AUTHORIZED')
        self.assertLess(events.index('source_grad'),events.index('original_full_fine_stats'))

    def test_wrong_selector_prior_to_gradients(self):
        kw,obj,ids,events=self.fixture();ids[0]=ids[1]
        with self.assertRaises(Exception):c.bounded_selector(obj,ids)
        self.assertEqual(events,[])

    def test_changed_original_source_order_blocks(self):
        kw,obj,ids,_=self.fixture()
        obj.FULL_FACTORIAL_CELLS=tuple(reversed(b.SOURCE_CELLS))
        with self.assertRaises(Exception):c.bounded_selector(obj,ids)

    def test_frozen_mutation_after_call_blocks(self):
        kw,obj,ids,_=self.fixture()
        def mutate(**inp):
            kw['component'][0,0,1,1]+=0.003
            return [{'R':1.} for _ in range(4)]
        obj._component_fine_pair_stats=mutate
        with self.assertRaisesRegex(c.StageCBlocked,'INPUT_MUTATION'):
            c.paired_cpu_seam(frozen=obj,hybrid=hybrid,selector=c.bounded_selector(obj,ids),
                              source_row_ids=ids,**kw)

    def test_v10_mutation_after_call_blocks(self):
        kw,obj,ids,_=self.fixture()
        class Malicious:
            def hybrid_fine_pair_gap(self,*args,**kwargs):
                kw['component'][0,0,1,1]+=0.003
                return {},0,{}
        with self.assertRaisesRegex(c.StageCBlocked,'INPUT_MUTATION'):
            c.paired_cpu_seam(frozen=obj,hybrid=Malicious(),
                selector=c.bounded_selector(obj,ids),source_row_ids=ids,**kw)

    def test_frozen_exception_propagates(self):
        kw,obj,ids,_=self.fixture()
        def boom(**kw):raise RuntimeError('FROZEN_COARSE_GUARD_FAIL')
        obj._component_fine_pair_stats=boom
        with self.assertRaisesRegex(RuntimeError,'FROZEN_COARSE_GUARD_FAIL'):
            c.paired_cpu_seam(frozen=obj,hybrid=hybrid,selector=c.bounded_selector(obj,ids),
                              source_row_ids=ids,**kw)

    def test_missing_parent_blocks(self):
        kw,obj,ids,_=self.fixture()
        def broken(**kw):return [{}]*4
        obj.coarse._component_region_pair_stats=broken
        with self.assertRaises(Exception):
            c.paired_cpu_seam(frozen=obj,hybrid=hybrid,selector=c.bounded_selector(obj,ids),
                              source_row_ids=ids,**kw)

    def test_projector_original_raw_guard_blocks(self):
        kw,obj,ids,_=self.fixture()
        rows=[{'R':1.}]*4
        acc=[{'raw_reconstruction_max_abs':0.1,
              'stage':{'raw_write':{'D_visible':1.,'D_complement':1.}}}]*2
        with self.assertRaisesRegex(c.StageCBlocked,'RAW_RECONSTRUCTION_AUTH'):
            c.verify_projector_original_guards(frozen=obj,
                projector_accumulators=acc,frozen_records=rows)

    def test_projector_component_energy_blocks(self):
        kw,obj,ids,_=self.fixture()
        rows=[{'R':1.}]*4
        acc=[{'raw_reconstruction_max_abs':0.,
              'stage':{'raw_write':{'D_visible':9.,'D_complement':1.}}}]*2
        with self.assertRaisesRegex(c.StageCBlocked,'VISIBLE_RAW_ENERGY_INTERNAL_REPLAY'):
            c.verify_projector_original_guards(frozen=obj,
                projector_accumulators=acc,frozen_records=rows)

    def test_no_runtime_entrypoint(self):
        content=Path(c.__file__).read_text(encoding='utf-8')
        self.assertNotIn('def run_full(',content)
        self.assertNotIn('def run_worker(',content)
        self.assertNotIn('subprocess.',content)
        self.assertNotIn('torch.cuda.',content)


class TestStageCActualFrozenOuterCPU(unittest.TestCase):
    """Executed only from the actual repository, not the standalone ZIP."""
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        if not (root / 'scripts/audit_native_recurrent_generator_anchor_pair_localization.py').is_file():
            raise unittest.SkipTest('original frozen audit unavailable in standalone bundle')
        from scripts import audit_native_recurrent_generator_anchor_pair_localization as frozen
        cls.original = frozen
        torch.set_num_threads(2)

    def fixture(self):
        old = self.original
        torch.manual_seed(6165)
        k,batch,seq,width = 4,32,128,8
        attention = torch.zeros(batch,seq,dtype=torch.bool)
        roles = {key:torch.zeros(batch,seq,dtype=torch.bool)
                 for key in ('CLAIM','EOS','EVIDENCE')}
        classes = torch.randint(0,5,(batch,seq))
        masks={side:{cell:torch.zeros(batch,seq,dtype=torch.bool)
                     for cell in old.FINE_CLASSES} for side in ('CLAIM','EVIDENCE')}
        for i in range(batch):
            end=117+i%9
            attention[i,:end]=True
            roles['CLAIM'][i,:52]=True
            roles['EOS'][i,52]=True
            roles['EVIDENCE'][i,53:end]=True
        for side in ('CLAIM','EVIDENCE'):
            for j,cell in enumerate(old.FINE_CLASSES):
                masks[side][cell]=roles[side] & (classes==j)
        component=(.002*torch.randn(k,batch,seq,width)).float()
        component*=attention[None,:,:,None]
        log_a=torch.full((batch,seq,1,width),-.04,dtype=torch.float32)
        log_a[:,:,:,4:]=-1.5
        survival=old.coarse.pairgap._backward_survival_factor(
            log_a_flat=log_a.reshape(batch,seq,width),attention_mask=attention)
        shape=SimpleNamespace(intermediate_size=1,state_size=width,state_width=width)
        return dict(component=component,log_a=log_a,survival=survival,
                    attention_mask=attention,role_masks=roles,fine_masks=masks,shape=shape)

    def test_original_full_stats_exact_and_no_monkeypatch(self):
        kw=self.fixture()
        original=self.original
        row_ids=[f'ROW_{i}' for i in range(800,832)]
        seam=c.bounded_selector(original,row_ids)
        frozen_fine_func=original._fine_pair_gap_interference_fft
        expected=original._component_fine_pair_stats(**kw)
        result=c.paired_cpu_seam(frozen=original,hybrid=hybrid,selector=seam,
                                 source_row_ids=row_ids,**kw)
        self.assertEqual(expected,result['frozen_records'])
        self.assertIs(frozen_fine_func,original._fine_pair_gap_interference_fft)
        self.assertEqual(len(result['v10_records']),4)
        self.assertEqual(result['V10_E1_FROZEN_GUARDS'],'NOT_EVALUATED')
        self.assertEqual(result['v10_dispatch']['total_chunks'],2)

    def test_original_replay_guard_rejects_parent_tamper(self):
        kw=self.fixture()
        old=self.original
        fine_gap,span=old._fine_pair_gap_interference_fft(
            **{key:kw[key] for key in ('component','log_a','survival',
                                     'attention_mask','fine_masks','shape')})
        coarse=old.coarse._component_region_pair_stats(
            **{key:kw[key] for key in ('component','log_a','survival',
                                     'attention_mask','role_masks','shape')})
        parent=old.PARENT_PAIR_NAMES[0];cell=old.FINE_PAIR_NAMES[0]
        fine_gap[parent][cell][0,1]+=1.0
        with self.assertRaisesRegex(Exception,'VALIDATED_PAIR_GAP_VECTOR_REPLAY'):
            c._post_original_reducer(old,coarse,fine_gap,span,kw['component'])


if __name__ == "__main__":
    unittest.main()
