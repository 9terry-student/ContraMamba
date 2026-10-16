"""CPU-only synthetic G2-H2 read-only hook adapter contract regression.

No real Mamba/parent, checkpoint, tokenizer, training, Kaggle or CUDA. Uses
synthetic forward that mimics G2 wrapper's order and mask provenance.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contramamba.gen5_g2_h2_readonly_output_adapter import (  # noqa: E402
    H2Layer22ReadOnlyAdapter, H2CaptureContractError, VERB_IDS,
)
from scripts.gen5_g2_h2_three_condition_cpu_metrics import output_readout  # noqa: E402


class SmallNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2, dtype=torch.float64)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor([[0.25, 0.4], [-0.2, 0.8]], dtype=torch.float64))
            self.lin.bias.copy_(torch.tensor([0.1, -0.1], dtype=torch.float64))
        self.seen_masks = []

    def forward(self, hidden_states, cache_params=None, cache_position=None,
                attention_mask=None):
        assert cache_params is None and cache_position is None
        self.seen_masks.append(attention_mask)
        result = self.lin(hidden_states)
        return result if attention_mask is None else result + attention_mask.unsqueeze(-1)


class SmallG2(nn.Module):
    def __init__(self, arm='FULL', *, corrupt_forward_mask=False,
                 corrupt_zero=False):
        super().__init__()
        self.native_mixer = SmallNative()
        self._phase2_active_mask = None
        self.arm = arm
        self.corrupt_forward_mask = corrupt_forward_mask
        self.corrupt_zero = corrupt_zero

    @contextmanager
    def active_mask(self, mask):
        assert self._phase2_active_mask is None
        self._phase2_active_mask = mask
        try:
            yield
        finally:
            self._phase2_active_mask = None

    def forward(self, hidden_states, cache_params=None, cache_position=None,
                attention_mask=None):
        effective = (attention_mask if attention_mask is not None
                     else self._phase2_active_mask)
        native_mask = effective if self.corrupt_forward_mask else attention_mask
        native = self.native_mixer(hidden_states, cache_params=cache_params,
                                   cache_position=cache_position,
                                   attention_mask=native_mask)
        if self.arm == 'ZERO':
            return native + 0.001 if self.corrupt_zero else native
        return native + 0.2 * hidden_states * effective.unsqueeze(-1)


def sample_data(*, real_rows=None):
    if real_rows is None:
        claim = [11, 12, 13]
        n = 14
        pairs = ['p0', 'p1']
        rows = []
        for pid in pairs:
            for condition in ('POS_ASSERTS', 'NEG_DENIES', 'POS_CONFIRMS'):
                evidence = [20, 21, VERB_IDS[condition]] + list(range(22, 22 + n-3))
                serial = claim + [0] + evidence
                rows.append((pid, condition, serial, len(claim) + 1 + 2, len(claim)))
        rows = [rows[i] for i in [1, 3, 5, 0, 4, 2]]
    else:
        rows = [(r['pair_id'], c, r['conditions'][c]['serialized_token_ids'],
                 r['conditions'][c]['verb_serialized_token_index'],
                 len(r['claim_token_ids']))
                for r in real_rows for c in ('NEG_DENIES', 'POS_CONFIRMS', 'POS_ASSERTS')]
        rows = [rows[i] for i in (4, 0, 5, 2, 1, 3)]
    count = len(rows)
    ids = torch.zeros((count, 128), dtype=torch.long)
    att = torch.zeros((count, 128), dtype=torch.bool)
    claim = torch.zeros_like(att)
    evidence = torch.zeros_like(att)
    pairs, conditions, row_ids, verbs = [], [], [], []
    for i, (pid, condition, serial, verb, clen) in enumerate(rows):
        length = len(serial)
        ids[i, :length] = torch.tensor(serial, dtype=torch.long)
        att[i, :length] = True
        claim[i, :clen] = True
        evidence[i, clen+1:length] = True
        pairs.append(pid)
        conditions.append(condition)
        row_ids.append(pid + '__g2_h2__' + condition.lower())
        verbs.append(verb)
    x = torch.stack((ids.to(torch.float64) / 25000. + 0.1,
                     ids.to(torch.float64) / 50000. - 0.03), dim=-1)
    return dict(input_ids=ids, attention_mask=att, claim_mask=claim,
                evidence_mask=evidence, pair_ids=pairs, conditions=conditions,
                row_ids=row_ids, verb_indices=verbs), x


def obs(wrapper, data):
    return H2Layer22ReadOnlyAdapter(wrapper, arm=wrapper.arm, **data)


class AdapterTest(unittest.TestCase):
    def test_forward_noop_and_row_identity_and_metrics(self):
        data, x = sample_data()
        wrapper = SmallG2().eval().requires_grad_(False)
        head = nn.Linear(2, 3, dtype=torch.float64).eval().requires_grad_(False)
        initial_params = [p.detach().clone() for p in tuple(wrapper.parameters()) + tuple(head.parameters())]
        with torch.inference_mode(), wrapper.active_mask(data['attention_mask']):
            plain = wrapper(x)
            plain_logits = head(plain).sum(dim=1)
        observer = obs(wrapper, data)
        with torch.inference_mode(), wrapper.active_mask(data['attention_mask']), observer:
            watched = wrapper(x)
            watched_logits = head(watched).sum(dim=1)
        self.assertTrue(torch.equal(plain, watched))
        self.assertTrue(torch.equal(plain_logits, watched_logits))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(initial_params,
            tuple(wrapper.parameters()) + tuple(head.parameters()))))
        self.assertEqual(wrapper.native_mixer.seen_masks, [None, None])
        self.assertEqual(len(wrapper._forward_hooks), 0)
        self.assertEqual(len(wrapper.native_mixer._forward_hooks), 0)
        self.assertIsNone(wrapper._phase2_active_mask)
        record = observer.record()
        self.assertEqual(set(record.conditions), set(data['conditions']))
        self.assertEqual(record.row_ids, tuple(data['row_ids']))
        self.assertTrue(torch.equal(record.wrapped, watched))
        self.assertFalse(record.native.requires_grad)
        self.assertNotEqual(record.native.data_ptr(), record.wrapped.data_ptr())
        self.assertNotEqual(record.native.data_ptr(), record.contribution.data_ptr())
        # Join H2 conditions by PAIR identity, not batch adjacency/order.
        for pair in ('p0','p1'):
            indexes = {c: data['pair_ids'].index(pair) if False else next(
                i for i, (p,co) in enumerate(zip(data['pair_ids'], data['conditions']))
                if p == pair and co == c)
                for c in VERB_IDS}
            trimmed = {c: record.metric_row(i) for c,i in indexes.items()}
            outputs = {c: {'native': r['native'], 'wrapped': r['wrapped']}
                       for c,r in trimmed.items()}
            lengths = {r['length'] for r in trimmed.values()}
            evidence_starts = {r['evidence_start'] for r in trimmed.values()}
            self.assertEqual(len(lengths), 1)
            self.assertEqual(len(evidence_starts), 1)
            i = next(iter(indexes.values()))
            vals = output_readout(outputs, length=lengths.pop(),
                  evidence_start=evidence_starts.pop(),
                  verb_index=record.verb_indices[i], arm='FULL')
            self.assertEqual(vals['prefix_max_abs_deviations']['NATIVE_PRE_NC'], 0.)
            self.assertEqual(vals['prefix_max_abs_deviations']['Q_PRE_NC'], 0.)
            self.assertGreater(vals['distance_rms']['VERB']['NC'], 0.)
            self.assertEqual(vals['distance_rms']['POST8']['NC'], 0.)

    def test_zero_contribution_and_no_forward_twice(self):
        data, x = sample_data()
        wrapper = SmallG2('ZERO').eval().requires_grad_(False)
        o = obs(wrapper, data)
        with torch.inference_mode(), wrapper.active_mask(data['attention_mask']), o:
            wrapper(x)
        self.assertEqual(int(torch.count_nonzero(o.record().contribution)), 0)
        for i in range(len(data['pair_ids'])):
            self.assertTrue(torch.equal(o.record().native[i], o.record().wrapped[i]))
        with self.assertRaisesRegex(H2CaptureContractError, 'OBSERVER_NOT_REUSABLE'):
            with o:
                pass

    def test_corrupt_zero_fail_closed(self):
        data, x = sample_data()
        wrapper = SmallG2('ZERO', corrupt_zero=True).eval().requires_grad_(False)
        o = obs(wrapper, data)
        with torch.inference_mode(), wrapper.active_mask(data['attention_mask']), o:
            wrapper(x)
        with self.assertRaisesRegex(H2CaptureContractError, 'ZERO_CONTRIBUTION_NONZERO'):
            o.record()

    def test_native_mask_forwarding_not_modified(self):
        data, x = sample_data()
        wrapper = SmallG2(corrupt_forward_mask=True).eval().requires_grad_(False)
        o = obs(wrapper, data)
        with torch.inference_mode(), wrapper.active_mask(data['attention_mask']):
            with self.assertRaisesRegex(H2CaptureContractError,
                                        'NATIVE_MASK_FORWARDING_CHANGED'):
                with o:
                    wrapper(x)
        self.assertEqual(len(wrapper._forward_hooks), 0)
        self.assertEqual(len(wrapper.native_mixer._forward_hooks), 0)

    def test_nonbinary_mask_invalid(self):
        data, _ = sample_data()
        data['attention_mask'] = data['attention_mask'].to(torch.int64)
        data['attention_mask'][0,0] = 2
        with self.assertRaisesRegex(H2CaptureContractError, 'ATTENTION_NONBINARY'):
            obs(SmallG2().eval().requires_grad_(False), data)

    def test_wrong_conditions_row_identity_invalid(self):
        data, _ = sample_data()
        data['row_ids'][0] = data['row_ids'][0].replace('g2_h2', 'phase3')
        with self.assertRaisesRegex(H2CaptureContractError, 'H2_ROW_ID_PROVENANCE'):
            obs(SmallG2().eval().requires_grad_(False), data)

    def test_wrong_verb_id_and_position(self):
        data, _ = sample_data()
        data['input_ids'][0, data['verb_indices'][0]] = 777
        with self.assertRaisesRegex(H2CaptureContractError, 'VERB_FROZEN_TOKEN_ID'):
            obs(SmallG2().eval().requires_grad_(False), data)

    def test_pair_mask_mismatch(self):
        data, _ = sample_data()
        idx = data['pair_ids'].index('p0')
        index2 = next(i for i in range(len(data['pair_ids'])) if data['pair_ids'][i]=='p0' and i != idx)
        data['verb_indices'][index2] += 1
        with self.assertRaisesRegex(H2CaptureContractError, 'VERB_FROZEN_TOKEN_ID|PAIR_MASK_POSITION_MISMATCH'):
            obs(SmallG2().eval().requires_grad_(False), data)

    def test_duplicate_pair_condition(self):
        data, _ = sample_data()
        data['pair_ids'][1] = data['pair_ids'][0]
        data['conditions'][1] = data['conditions'][0]
        data['row_ids'][1] = data['row_ids'][0]
        with self.assertRaisesRegex(H2CaptureContractError, 'DUPLICATE_ROW_ID'):
            obs(SmallG2().eval().requires_grad_(False), data)

    def test_bad_eos_and_padding_fail(self):
        data, _ = sample_data()
        data['input_ids'][0,3] = 6
        with self.assertRaisesRegex(H2CaptureContractError, 'EOS_PAD_TOKEN_ID'):
            obs(SmallG2().eval().requires_grad_(False), data)

    def test_grad_or_train_rejected(self):
        data, _ = sample_data()
        with self.assertRaisesRegex(H2CaptureContractError, 'EVAL_REQUIRED'):
            obs(SmallG2().train(), data)
        with self.assertRaisesRegex(H2CaptureContractError, 'FROZEN_PARAMETERS_REQUIRED'):
            obs(SmallG2().eval(), data)

    def test_multiple_and_missing_forward_rejected(self):
        data, x = sample_data()
        wrapper = SmallG2().eval().requires_grad_(False)
        with self.assertRaisesRegex(H2CaptureContractError, 'EXACTLY_ONE_FORWARD_REQUIRED'):
            with obs(wrapper, data):
                pass
        with self.assertRaisesRegex(H2CaptureContractError, 'NATIVE_FORWARD_ORDER'):
            with torch.inference_mode(), wrapper.active_mask(data['attention_mask']), obs(wrapper, data):
                wrapper(x)
                wrapper(x)
        self.assertEqual(len(wrapper._forward_hooks), 0)
        self.assertEqual(len(wrapper.native_mixer._forward_hooks), 0)

    def test_duplicate_observer_rejected_and_cleaned(self):
        data, x = sample_data()
        wrapper = SmallG2().eval().requires_grad_(False)
        a, b = obs(wrapper, data), obs(wrapper, data)
        with torch.inference_mode(), wrapper.active_mask(data['attention_mask']), a:
            with self.assertRaisesRegex(H2CaptureContractError, 'DUPLICATE_OBSERVER'):
                with b:
                    pass
            wrapper(x)
        a.record()
        self.assertEqual(len(wrapper._forward_hooks), 0)
        self.assertEqual(len(wrapper.native_mixer._forward_hooks), 0)

    def test_effective_mask_mismatch(self):
        data, x = sample_data()
        wrapper = SmallG2().eval().requires_grad_(False)
        bad_mask = data['attention_mask'].clone()
        bad_mask[0,1] = False
        with torch.inference_mode(), wrapper.active_mask(bad_mask):
            with self.assertRaisesRegex(H2CaptureContractError, 'EFFECTIVE_MASK_IDENTITY'):
                with obs(wrapper, data):
                    wrapper(x)
        self.assertEqual(len(wrapper._forward_hooks), 0)

    def test_real_frozen_input_subset_if_present(self):
        source = ROOT / 'reports/gen5_g2_h2_static_matched_verb_candidates/' \
            'g2h2-static-matched-verbs-40a279a-r1/matched_verb_candidate.jsonl'
        if not source.is_file():
            self.skipTest('real frozen candidate file not present (synthetic tests still execute)')
        with source.open(encoding='utf8') as file:
            records = [json.loads(file.readline()) for _ in range(2)]
        data, x = sample_data(real_rows=records)
        wrapper = SmallG2().eval().requires_grad_(False)
        o = obs(wrapper, data)
        with torch.inference_mode(), wrapper.active_mask(data['attention_mask']), o:
            wrapper(x)
        out = o.record()
        self.assertEqual(out.native.shape, (6,128,2))
        self.assertEqual(len(set(out.row_ids)), 6)
        for i in range(6):
            self.assertEqual(len(out.metric_row(i)['native']), out.valid_lengths[i])
            self.assertEqual(out.evidence_starts[i], len(records[0]['claim_token_ids'])+1)


if __name__ == '__main__':
    unittest.main()
