#!/usr/bin/env python3
"""CPU-only Stage A tensor provenance contract and tamper tests."""
from __future__ import annotations

import copy
import hashlib
import json
import unittest
from types import SimpleNamespace

import torch

from scripts import gen5_v10_bounded_tensor_provenance_candidate as prov


class TestStageACanonicalProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, b=8, w=6):
        gen = torch.Generator().manual_seed(475)
        seq = 128
        comp = torch.randn((4, b, seq, w), generator=gen, dtype=torch.float32) * 0.05
        la = -torch.rand((b, seq, 1, w), generator=gen, dtype=torch.float32) * 0.07
        surv = torch.rand((b, seq, w), generator=gen, dtype=torch.float32)
        mask = torch.zeros(b, seq, dtype=torch.bool)
        claim = torch.zeros_like(mask)
        evidence = torch.zeros_like(mask)
        eos = torch.zeros_like(mask)
        for j in range(b):
            stop = 110 + j % 7
            mask[j, :stop] = True
            claim[j, :53] = True
            eos[j, 53] = True
            evidence[j, 54:stop] = True
        role_masks = {"CLAIM": claim, "EVIDENCE": evidence, "EOS": eos}
        fine_masks = {}
        pos = torch.arange(seq).unsqueeze(0).expand(b, -1)
        for side, role in (("CLAIM", claim), ("EVIDENCE", evidence)):
            fine_masks[side] = {name: role & (pos % 5 == index)
                                for index, name in enumerate(prov.FINE_CLASSES)}
        return {"component": comp, "log_a": la, "survival": surv,
                "attention_mask": mask, "role_masks": role_masks,
                "fine_masks": fine_masks,
                "shape": SimpleNamespace(intermediate_size=1, state_size=w),
                "synthetic_cpu_geometry": True}

    def make(self, kw=None):
        return prov.manifest_for_cpu_inputs(**(kw or self.fixture()))

    def test_sorted_keys_and_geometry(self):
        m = self.make()
        self.assertEqual(m['schema_version'], prov.SCHEMA)
        self.assertEqual([e['key'] for e in m['tensors']], list(prov.TENSOR_KEYS))
        self.assertEqual(len(m['tensors']), 17)
        self.assertEqual(m['V10_E0_PROVENANCE'], 'NOT_EVALUATED')
        self.assertEqual(prov.validate_manifest_structure(m)['status'],
                         'STRUCTURAL_HASHES_VALID_CONTENT_UNAUTHENTICATED')

    def test_repeatability_and_reverification(self):
        kw = self.fixture()
        a = self.make(kw)
        b = self.make(kw)
        self.assertEqual(a, b)
        self.assertEqual(prov.verify_cpu_inputs(a, **kw)['manifest_sha256'],
                         a['manifest_sha256'])
        self.assertEqual(prov.compare_manifest_identity(a, b)['manifest_sha256'],
                         a['manifest_sha256'])

    def test_float32_little_endian_content(self):
        kw = self.fixture()
        kw['component'].zero_()
        kw['component'][0, 0, 0, :4] = torch.tensor([1.0, -2.5, 0.0, -0.0])
        m = self.make(kw)
        import numpy as np
        expected = hashlib.sha256(kw['component'].contiguous().numpy()
                                  .astype('<f4', copy=False).tobytes(order='C')).hexdigest()
        entry = next(x for x in m['tensors'] if x['key'] == 'component')
        self.assertEqual(entry['content_sha256'], expected)

    def test_bool_is_01_encoding(self):
        kw = self.fixture()
        m = self.make(kw)
        key = 'attention_mask'
        entry = next(x for x in m['tensors'] if x['key'] == key)
        raw = kw[key].contiguous().numpy().astype('uint8', copy=False).tobytes()
        self.assertEqual(set(raw), {0, 1})
        self.assertEqual(entry['byte_count'], kw[key].numel())
        self.assertEqual(entry['content_sha256'], hashlib.sha256(raw).hexdigest())

    def test_noncontiguous_values_canonicalize_without_mutation(self):
        kw = self.fixture()
        contiguous = kw['component'].clone()
        kw['component'] = contiguous.transpose(-1, -2).contiguous().transpose(-1, -2)
        self.assertFalse(kw['component'].is_contiguous())
        a = self.make(kw)
        self.assertTrue(torch.equal(kw['component'], contiguous))
        kw['component'] = contiguous
        self.assertEqual(a, self.make(kw))

    def test_float_content_tampering_detected(self):
        kw = self.fixture()
        m = self.make(kw)
        kw['component'][0, 0, 2, 1] += 1e-3
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'CPU_TENSOR_CONTENT'):
            prov.verify_cpu_inputs(m, **kw)

    def test_sequence_order_tampering_detected(self):
        kw = self.fixture()
        m = self.make(kw)
        kw['component'] = kw['component'].flip(2)
        with self.assertRaises(prov.ProvenanceBlocked):
            prov.verify_cpu_inputs(m, **kw)

    def test_role_mask_content_tampering_blocks(self):
        kw = self.fixture()
        m = self.make(kw)
        kw['role_masks']['CLAIM'][0, 0] = False
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'ROLE_PARTITION'):
            prov.verify_cpu_inputs(m, **kw)

    def test_fine_mask_tampering_blocks(self):
        kw = self.fixture()
        kw['fine_masks']['CLAIM']['A_TITLE'][0, 0] = False
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'FINE_MASK_PARTITION'):
            self.make(kw)

    def test_missing_extra_mask_keys_block(self):
        kw = self.fixture()
        kw['fine_masks']['EVIDENCE'].pop('RESIDUAL')
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'FINE_MASK_CLASSES'):
            self.make(kw)
        kw = self.fixture()
        kw['role_masks']['OTHER'] = torch.zeros_like(kw['attention_mask'])
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'ROLE_MASK_KEYS'):
            self.make(kw)

    def test_wrong_float_and_bool_dtype_block(self):
        kw = self.fixture()
        kw['component'] = kw['component'].double()
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'TENSOR_DTYPE'):
            self.make(kw)
        kw = self.fixture()
        kw['attention_mask'] = kw['attention_mask'].to(torch.uint8)
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'TENSOR_DTYPE'):
            self.make(kw)

    def test_shape_and_batch_mismatch_block(self):
        kw = self.fixture()
        kw['shape'].state_size += 1
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'STATE_WIDTH'):
            self.make(kw)
        kw = self.fixture(b=16)
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'K_BATCH_SEQUENCE'):
            self.make(kw)

    def test_nonfinite_and_invalid_sign_block(self):
        kw = self.fixture()
        kw['component'][0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'NONFINITE'):
            self.make(kw)
        kw = self.fixture()
        kw['log_a'][0, 0, 0, 0] = 0.01
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'POSITIVE_LOG_A'):
            self.make(kw)
        kw = self.fixture()
        kw['survival'][0, 0, 0] = -.1
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'NEGATIVE_SURVIVAL'):
            self.make(kw)

    def test_gpu_or_non_cpu_device_blocks(self):
        kw = self.fixture()
        kw['survival'] = torch.empty(kw['survival'].shape, device='meta')
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'CPU_TENSOR'):
            self.make(kw)

    def test_production_width_cannot_be_spoofed(self):
        kw = self.fixture()
        kw['synthetic_cpu_geometry'] = False
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'PRODUCTION_WIDTH'):
            self.make(kw)
        kw = self.fixture()
        kw['synthetic_cpu_geometry'] = 1
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'SYNTHETIC_FLAG_TYPE'):
            self.make(kw)

    def test_manifest_key_omission_injection_and_order_block(self):
        m = self.make()
        for mode in ('omit','extra','reorder'):
            t = copy.deepcopy(m)
            if mode == 'omit':t['tensors'].pop()
            if mode == 'extra':t['tensors'].append(copy.deepcopy(t['tensors'][-1]))
            if mode == 'reorder':t['tensors'][0],t['tensors'][1]=t['tensors'][1],t['tensors'][0]
            with self.assertRaises(prov.ProvenanceBlocked):
                prov.validate_manifest_structure(t)

    def test_binding_and_manifest_digest_forgery_block(self):
        m = self.make()
        t = copy.deepcopy(m)
        t['tensors'][0]['content_sha256'] = '0'*64
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'ENTRY_BINDING_DIGEST'):
            prov.validate_manifest_structure(t)
        t = copy.deepcopy(m)
        t['manifest_sha256'] = 'a'*64
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'MANIFEST_DIGEST_MISMATCH'):
            prov.validate_manifest_structure(t)

    def test_geometry_count_bool_and_device_forgery_block(self):
        m = self.make()
        t = copy.deepcopy(m)
        t['geometry']['state_size'] = True
        with self.assertRaises(prov.ProvenanceBlocked):
            prov.validate_manifest_structure(t)
        t = copy.deepcopy(m)
        t['tensors'][0]['numel'] = 0
        with self.assertRaises(prov.ProvenanceBlocked):
            prov.validate_manifest_structure(t)
        t = copy.deepcopy(m)
        t['tensors'][0]['device_at_capture'] = 'cuda:0'
        with self.assertRaises(prov.ProvenanceBlocked):
            prov.validate_manifest_structure(t)

    def test_manifest_divergence_and_source_claims(self):
        kw = self.fixture()
        m1 = self.make(kw)
        kw['component'] = kw['component'] + 0.1
        m2 = self.make(kw)
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'INPUT_MANIFEST_IDENTITY'):
            prov.compare_manifest_identity(m1, m2)
        t = copy.deepcopy(m1)
        t['V10_E0_PROVENANCE'] = 'PASS'
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'FORBIDDEN_PROMOTION'):
            prov.validate_manifest_structure(t)
        self.assertEqual(m1['SCIENTIFIC_EXECUTION'], 'NOT_AUTHORIZED')

    def test_self_consistent_forged_manifest_still_fails_cpu_reverification(self):
        # A malicious editor can rehash metadata, but without matching CPU
        # tensors that is not evidence of identical input bytes.
        kw = self.fixture()
        original = self.make(kw)
        fake = copy.deepcopy(original)
        record = next(e for e in fake['tensors'] if e['key'] == 'component')
        record['content_sha256'] = '0' * 64
        core = {key: value for key, value in record.items()
                if key != 'binding_sha256'}
        record['binding_sha256'] = prov._digest(prov.BINDING_DOMAIN, core)
        base = {key: value for key, value in fake.items()
                if key != 'manifest_sha256'}
        fake['manifest_sha256'] = prov._digest(prov.MANIFEST_DOMAIN, base)
        self.assertEqual(prov.validate_manifest_structure(fake)['status'],
                         'STRUCTURAL_HASHES_VALID_CONTENT_UNAUTHENTICATED')
        with self.assertRaisesRegex(prov.ProvenanceBlocked, 'CPU_TENSOR_CONTENT'):
            prov.verify_cpu_inputs(fake, **kw)

    def test_binding_hash_prevents_key_relabel_even_with_same_bytes(self):
        kw = self.fixture()
        m = self.make(kw)
        entries = {e['key']:e for e in m['tensors']}
        self.assertNotEqual(entries['role_masks.CLAIM']['binding_sha256'],
                            entries['role_masks.EVIDENCE']['binding_sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
