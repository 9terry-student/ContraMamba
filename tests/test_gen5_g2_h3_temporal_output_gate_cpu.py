"""Synthetic CPU verification for Gen5 H3 output gate. NO real model/forward."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contramamba.gen5_g2_h3_temporal_output_gate_cpu import (
    REGIMES, H3GateContractError, H3TemporalOutputGateCPU, frozen_window_mask,
)


class FakeNative(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.125 + 0.25


class FakeFrozenFullWrapper(nn.Module):
    arm = 'FULL'

    def __init__(self):
        super().__init__()
        self.native_mixer = FakeNative()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        native = self.native_mixer(x)
        q = (x * 0 + 0.5)
        return native + q


def fixture(pairs: int = 1):
    b = pairs * 3
    ids = torch.zeros((b, 128), dtype=torch.long)
    valid = torch.zeros((b, 128), dtype=torch.bool)
    claim = torch.zeros((b, 128), dtype=torch.bool)
    evidence = torch.zeros((b, 128), dtype=torch.bool)
    pairs_ids, conditions, rows, positions = [], [], [], []
    for p in range(pairs):
        pid = f'test-pair-{p:03d}'
        claim_len = 4 + p % 4
        e_start = claim_len + 1
        length = e_start + 19
        t = e_start + 2
        for j, (cond, token) in enumerate(zip(('NEG_DENIES','POS_CONFIRMS','POS_ASSERTS'),
                                               (31647,23849,17086))):
            i = 3*p + j
            ids[i, :claim_len] = torch.arange(101, 101 + claim_len)
            ids[i, e_start:length] = 200
            ids[i, t] = token
            valid[i, :length] = True
            claim[i, :claim_len] = True
            evidence[i, e_start:length] = True
            pairs_ids.append(pid)
            conditions.append(cond)
            rows.append(pid + '__g2_h2__' + cond.lower())
            positions.append(t)
    kwargs = dict(input_ids=ids, attention_mask=valid, claim_mask=claim,
                  evidence_mask=evidence, pair_ids=pairs_ids,
                  conditions=conditions, row_ids=rows, verb_indices=positions)
    return kwargs


class TemporalGateCPUContracts(unittest.TestCase):
    def setUp(self):
        self.kwargs = fixture()
        self.mask = frozen_window_mask(**self.kwargs)
        self.model = FakeFrozenFullWrapper().eval()
        self.x = torch.arange(3*128*768, dtype=torch.float32).reshape(3,128,768) / 10000

    def _run(self, regime: str, model=None):
        model = model or self.model
        with torch.no_grad():
            gate = H3TemporalOutputGateCPU(model, regime=regime,
                                           **self.kwargs)
            with gate:
                out = model(self.x)
            audit = gate.audit()
        self.assertEqual(audit['native_calls'], 1)
        self.assertEqual(audit['wrapper_calls'], 1)
        self.assertEqual(audit['window_tokens'], 27)
        self.assertEqual(len(model._forward_hooks), 0)
        self.assertEqual(len(model.native_mixer._forward_hooks), 0)
        return out

    def test_frozen_metadata_and_exact_window(self):
        self.assertEqual(tuple(self.mask.shape), (3,128))
        self.assertEqual(int(self.mask.sum().item()), 27)
        t = self.kwargs['verb_indices'][0]
        for row in range(3):
            self.assertFalse(bool(self.mask[row,:t].any()))
            self.assertTrue(bool(self.mask[row,t:t+9].all()))
            self.assertFalse(bool(self.mask[row,t+9:].any()))

    def test_none_endpoint_bitwise_native(self):
        with torch.no_grad():
            native = self.model.native_mixer(self.x)
        self.assertTrue(torch.equal(self._run('NONE'), native))

    def test_all_endpoint_bitwise_full(self):
        with torch.no_grad():
            full = self.model(self.x)
        self.assertTrue(torch.equal(self._run('ALL'), full))

    def test_only_window_exact_selection(self):
        with torch.no_grad():
            native, full = self.model.native_mixer(self.x), self.model(self.x)
            expected = torch.where(self.mask.unsqueeze(-1), full, native)
        self.assertTrue(torch.equal(self._run('ONLY_WINDOW'), expected))

    def test_drop_window_exact_selection(self):
        with torch.no_grad():
            native, full = self.model.native_mixer(self.x), self.model(self.x)
            expected = torch.where(self.mask.unsqueeze(-1), native, full)
        self.assertTrue(torch.equal(self._run('DROP_WINDOW'), expected))

    def test_four_way_complement_without_roundoff_assumptions(self):
        outputs = {r: self._run(r) for r in REGIMES}
        w = self.mask.unsqueeze(-1).expand_as(outputs['ALL'])
        self.assertTrue(torch.equal(outputs['ONLY_WINDOW'][w], outputs['ALL'][w]))
        self.assertTrue(torch.equal(outputs['ONLY_WINDOW'][~w], outputs['NONE'][~w]))
        self.assertTrue(torch.equal(outputs['DROP_WINDOW'][w], outputs['NONE'][w]))
        self.assertTrue(torch.equal(outputs['DROP_WINDOW'][~w], outputs['ALL'][~w]))
        for r in REGIMES:
            self.assertTrue(torch.equal(outputs[r][:,:self.kwargs['verb_indices'][0]],
                                        outputs['NONE' if r in ('NONE','ONLY_WINDOW') else 'ALL'][:,:self.kwargs['verb_indices'][0]]))

    def test_many_triples_and_pair_alignment(self):
        kw = fixture(10)
        mask = frozen_window_mask(**kw)
        self.assertEqual(tuple(mask.shape), (30,128))
        self.assertEqual(int(mask.sum().item()), 270)
        kw['pair_ids'][1] = 'wrong'
        with self.assertRaisesRegex(H3GateContractError, 'PAIR_OR_ROW_DUPLICATE'):
            frozen_window_mask(**kw)

    def test_reject_wrong_verb_and_silent_second_token_change(self):
        kw = fixture()
        kw['input_ids'][1,kw['verb_indices'][1]] = 31647
        with self.assertRaisesRegex(H3GateContractError, 'VERB_TOKEN_ID'):
            frozen_window_mask(**kw)
        kw = fixture()
        kw['input_ids'][1,kw['verb_indices'][1]+10] += 1
        with self.assertRaisesRegex(H3GateContractError, 'NOT_SINGLE_TOKEN_SWAP'):
            frozen_window_mask(**kw)

    def test_reject_bad_mask_position_and_proposed_row_ids(self):
        kw = fixture()
        kw['attention_mask'][1,0] = False
        with self.assertRaisesRegex(H3GateContractError, 'TRIPLET_MASK_MISMATCH'):
            frozen_window_mask(**kw)
        kw = fixture()
        kw['row_ids'][1] = kw['row_ids'][0]
        with self.assertRaisesRegex(H3GateContractError, 'PAIR_OR_ROW_DUPLICATE'):
            frozen_window_mask(**kw)
        kw = fixture()
        kw['verb_indices'][0] += 1
        with self.assertRaisesRegex(H3GateContractError, 'TRIPLET_VERB_POSITION'):
            frozen_window_mask(**kw)

    def test_reject_autograd_training_and_wrong_arm(self):
        with self.assertRaisesRegex(H3GateContractError, 'AUTOGRAD_ENABLED'):
            with H3TemporalOutputGateCPU(self.model, regime='ALL',
                                         **self.kwargs):
                pass
        self.model.train()
        with self.assertRaisesRegex(H3GateContractError, 'FROZEN_EVAL_ONLY'):
            H3TemporalOutputGateCPU(self.model, regime='ALL', **self.kwargs)
        self.model.eval()
        self.model.arm = 'ZERO'
        with self.assertRaisesRegex(H3GateContractError, 'FROZEN_FULL_ARM_REQUIRED'):
            H3TemporalOutputGateCPU(self.model, regime='ALL', **self.kwargs)

    def test_reject_nested_hook_or_double_forward(self):
        with torch.no_grad():
            other = self.model.register_forward_hook(lambda _m, _a, o: o)
            try:
                with self.assertRaisesRegex(H3GateContractError, 'PREEXISTING_HOOKS'):
                    with H3TemporalOutputGateCPU(self.model, regime='ALL',
                                                 **self.kwargs):
                        self.model(self.x)
            finally:
                other.remove()
            gate = H3TemporalOutputGateCPU(self.model, regime='ALL',
                                           **self.kwargs)
            with self.assertRaisesRegex(H3GateContractError, 'NATIVE_CALL_ORDER'):
                with gate:
                    self.model(self.x)
                    self.model(self.x)
            self.assertEqual(len(self.model._forward_hooks), 0)
            self.assertEqual(len(self.model.native_mixer._forward_hooks), 0)

    def test_native_inplace_mutation_is_blocked(self):
        class WrongWrapper(FakeFrozenFullWrapper):
            def forward(self, x):
                native = self.native_mixer(x)
                native.add_(0.25)  # deliberate source corruption
                return native
        model = WrongWrapper().eval()
        with torch.no_grad():
            gate = H3TemporalOutputGateCPU(model, regime='NONE', **self.kwargs)
            with self.assertRaisesRegex(H3GateContractError, 'NATIVE_OUTPUT_MUTATED'):
                with gate:
                    model(self.x)
            self.assertFalse(model._forward_hooks)
            self.assertFalse(model.native_mixer._forward_hooks)

    def test_exception_restores_hooks_and_no_artifact(self):
        with torch.no_grad():
            gate = H3TemporalOutputGateCPU(self.model, regime='ONLY_WINDOW',
                                           **self.kwargs)
            with self.assertRaisesRegex(RuntimeError, 'simulated-failure'):
                with gate:
                    raise RuntimeError('simulated-failure')
            self.assertFalse(self.model._forward_hooks)
            self.assertFalse(self.model.native_mixer._forward_hooks)
            self.assertFalse(gate._entered)
            self.assertTrue(gate._finished)


if __name__ == '__main__':
    unittest.main()
