"""CPU-only synthetic parity for the distinct G2-H2 CUDA capture boundary.

Zero CUDA calls. A PASS is not a real-Mamba/CUDA numerical gate.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'tests'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from contramamba.gen5_g2_h2_device_output_capture_bridge import (  # noqa: E402
    H2DeviceOutputCaptureBridge, verify_h2_runtime_transfer,
)
from contramamba.gen5_g2_h2_readonly_output_adapter import (  # noqa: E402
    H2Layer22ReadOnlyAdapter, H2CaptureContractError,
)
from scripts.gen5_g2_h2_three_condition_cpu_metrics import output_readout  # noqa: E402
from test_gen5_g2_h2_readonly_output_adapter import (  # noqa: E402
    SmallG2, sample_data,
)


def observe(wrapper, data, x, *, new: bool):
    kwargs = {'arm': wrapper.arm, **data}
    adapter = (H2DeviceOutputCaptureBridge(wrapper, expected_output_device='cpu',
                                           **kwargs)
               if new else H2Layer22ReadOnlyAdapter(wrapper, **kwargs))
    with torch.inference_mode(), wrapper.active_mask(data['attention_mask']), adapter:
        y = wrapper(x)
    return y, adapter.record(), adapter


class DeviceBoundaryTest(unittest.TestCase):
    def test_cpu_parity_full_several_pairs(self):
        data, x = sample_data()
        w = SmallG2('FULL').eval().requires_grad_(False)
        y1, old, _ = observe(w, data, x, new=False)
        y2, new, _ = observe(w, data, x, new=True)
        self.assertTrue(torch.equal(y1, y2))
        for key in ('native', 'wrapped', 'contribution'):
            self.assertTrue(torch.equal(getattr(old, key), getattr(new, key)))
        self.assertEqual(old.row_ids, new.row_ids)
        for i in range(len(old.row_ids)):
            self.assertEqual(old.metric_row(i), new.metric_row(i))
        self.assertEqual(len(w._forward_hooks), 0)
        self.assertEqual(len(w.native_mixer._forward_hooks), 0)

    def test_cpu_parity_zero(self):
        data, x = sample_data()
        w = SmallG2('ZERO').eval().requires_grad_(False)
        y1, old, _ = observe(w, data, x, new=False)
        y2, new, _ = observe(w, data, x, new=True)
        self.assertTrue(torch.equal(y1, y2))
        self.assertTrue(torch.equal(old.contribution, new.contribution))
        self.assertEqual(torch.count_nonzero(new.contribution).item(), 0)

    def test_cpu_parity_with_frozen_record_subset(self):
        import json
        p = (ROOT / 'reports/gen5_g2_h2_static_matched_verb_candidates/'
             'g2h2-static-matched-verbs-40a279a-r1/matched_verb_candidate.jsonl')
        if not p.exists():
            self.skipTest('frozen cohort only available in full repo')
        rows = [json.loads(x) for x in p.read_text().splitlines()[:2]]
        data, x = sample_data(real_rows=rows)
        w = SmallG2('FULL').eval().requires_grad_(False)
        _, base, _ = observe(w, data, x, new=False)
        _, capture, _ = observe(w, data, x, new=True)
        self.assertTrue(torch.equal(base.wrapped, capture.wrapped))
        joined = {c: capture.metric_row(next(i for i, (p,cc) in enumerate(zip(
                   capture.pair_ids, capture.conditions)) if p == rows[0]['pair_id']
                   and cc == c)) for c in ('NEG_DENIES', 'POS_CONFIRMS', 'POS_ASSERTS')}
        output = {c: {'native': v['native'], 'wrapped': v['wrapped']}
                  for c, v in joined.items()}
        pos = joined['NEG_DENIES']
        measured = output_readout(output, length=pos['length'],
                                  evidence_start=pos['evidence_start'],
                                  verb_index=pos['verb_index'], arm='FULL')
        self.assertEqual(measured['prefix_max_abs_deviations']['Q_PRE_NC'], 0.0)

    def test_no_forward_side_effects_parameters_masks_and_logits(self):
        data, x = sample_data()
        w = SmallG2('FULL').eval().requires_grad_(False)
        head = nn.Linear(2, 3, dtype=torch.float64).eval().requires_grad_(False)
        before = [p.detach().clone() for p in w.parameters()]
        with torch.inference_mode(), w.active_mask(data['attention_mask']):
            plain_logits = head(w(x)).sum(dim=1)
        _, record, observer = observe(w, data, x, new=True)
        with torch.inference_mode(), w.active_mask(data['attention_mask']):
            after_logits = head(w(x)).sum(dim=1)
        self.assertTrue(torch.equal(plain_logits, after_logits))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, w.parameters())))
        self.assertIsNone(w._phase2_active_mask)
        self.assertFalse(record.wrapped.requires_grad)
        self.assertFalse(observer._handles)

    def test_corrupt_zero_rejected(self):
        data, x = sample_data()
        w = SmallG2('ZERO', corrupt_zero=True).eval().requires_grad_(False)
        with self.assertRaisesRegex(H2CaptureContractError, 'ZERO_CONTRIBUTION_NONZERO'):
            observe(w, data, x, new=True)
        self.assertFalse(w._forward_hooks)
        self.assertFalse(w.native_mixer._forward_hooks)

    def test_native_mask_identity_rejected(self):
        data, x = sample_data()
        w = SmallG2('FULL', corrupt_forward_mask=True).eval().requires_grad_(False)
        with self.assertRaisesRegex(H2CaptureContractError, 'NATIVE_MASK_FORWARDING_CHANGED'):
            observe(w, data, x, new=True)
        self.assertFalse(w._forward_hooks)
        self.assertFalse(w.native_mixer._forward_hooks)

    def test_effective_mask_mismatch_rejected(self):
        data, x = sample_data()
        w = SmallG2('FULL').eval().requires_grad_(False)
        o = H2DeviceOutputCaptureBridge(w, expected_output_device='cpu',
                                        arm=w.arm, **data)
        bad = data['attention_mask'].clone()
        bad[0, -1] = True
        with self.assertRaisesRegex(H2CaptureContractError, 'EFFECTIVE_MASK_IDENTITY'):
            with torch.inference_mode(), w.active_mask(bad), o:
                w(x)
        self.assertFalse(w._forward_hooks)
        self.assertFalse(w.native_mixer._forward_hooks)

    def test_cuda_mode_fails_on_cpu_without_cuda_invocation(self):
        data, _ = sample_data()
        w = SmallG2('FULL').eval().requires_grad_(False)
        with self.assertRaisesRegex(H2CaptureContractError, 'WRAPPER_DEVICE_MISMATCH'):
            H2DeviceOutputCaptureBridge(w, expected_output_device='cuda',
                                        arm=w.arm, **data)

    def test_device_mode_invalid(self):
        data, _ = sample_data()
        w = SmallG2('FULL').eval().requires_grad_(False)
        with self.assertRaisesRegex(H2CaptureContractError, 'DEVICE_MODE_UNSUPPORTED'):
            H2DeviceOutputCaptureBridge(w, expected_output_device='any',
                                        arm=w.arm, **data)

    def test_transfer_exact_feature_identity(self):
        data, _ = sample_data()
        keys = ('input_ids', 'attention_mask', 'claim_mask', 'evidence_mask')
        features = {k: data[k] for k in keys}
        runtime = {k: features[k].clone() for k in keys}
        verify_h2_runtime_transfer(features, runtime, expected_device='cpu')
        runtime['input_ids'][1, 3] += 1
        with self.assertRaisesRegex(H2CaptureContractError,
                                    'TRANSFER_TOKEN_MASK_DRIFT_input_ids'):
            verify_h2_runtime_transfer(features, runtime, expected_device='cpu')

    def test_transfer_rejects_missing_or_added_labels(self):
        data, _ = sample_data()
        keys = ('input_ids', 'attention_mask', 'claim_mask', 'evidence_mask')
        features = {k: data[k] for k in keys}
        runtime = {k: features[k].clone() for k in keys}
        runtime['final_labels'] = torch.zeros(6, dtype=torch.long)
        with self.assertRaisesRegex(H2CaptureContractError, 'TRANSFER_FEATURE_KEYS'):
            verify_h2_runtime_transfer(features, runtime, expected_device='cpu')

    def test_transfer_rejects_wrong_device_mode(self):
        data, _ = sample_data()
        keys = ('input_ids', 'attention_mask', 'claim_mask', 'evidence_mask')
        features = {k: data[k] for k in keys}
        runtime = {k: features[k].clone() for k in keys}
        with self.assertRaisesRegex(H2CaptureContractError,
                                    'TRANSFER_DEVICE_attention_mask'):
            verify_h2_runtime_transfer(features, runtime, expected_device='cuda')

    def test_multiple_forward_rejected_and_hooks_cleaned(self):
        data, x = sample_data()
        w = SmallG2('FULL').eval().requires_grad_(False)
        o = H2DeviceOutputCaptureBridge(w, expected_output_device='cpu',
                                        arm=w.arm, **data)
        with self.assertRaisesRegex(H2CaptureContractError, 'NATIVE_FORWARD_ORDER'):
            with torch.inference_mode(), w.active_mask(data['attention_mask']), o:
                w(x); w(x)
        self.assertFalse(w._forward_hooks)
        self.assertFalse(w.native_mixer._forward_hooks)

    def test_absent_forward_rejected_and_hooks_cleaned(self):
        data, _ = sample_data()
        w = SmallG2('FULL').eval().requires_grad_(False)
        o = H2DeviceOutputCaptureBridge(w, expected_output_device='cpu',
                                        arm=w.arm, **data)
        with self.assertRaisesRegex(H2CaptureContractError, 'EXACTLY_ONE_FORWARD_REQUIRED'):
            with torch.inference_mode(), w.active_mask(data['attention_mask']), o:
                pass
        self.assertFalse(w._forward_hooks)
        self.assertFalse(w.native_mixer._forward_hooks)


if __name__ == '__main__':
    unittest.main()
