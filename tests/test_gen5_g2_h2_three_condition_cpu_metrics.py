"""CPU-only H2 frozen candidate and numerical contracts. No GPU, model or tokenizer."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import gen5_g2_h2_three_condition_cpu_metrics as h2

ROOT = Path(__file__).resolve().parents[1]


class InputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort = ROOT / h2.COHORT / 'matched_verb_candidate.jsonl'
        cls.manifest = ROOT / h2.COHORT / 'static_candidate_manifest.json'

    def test_frozen_candidate_identity(self):
        records = h2.validate_candidate(self.cohort, self.manifest)
        self.assertEqual(len(records), 120)
        self.assertEqual(records[0]['pair_id'], 'xg1_fact_9012')
        self.assertEqual(records[0]['conditions']['NEG_DENIES']['evidence_token_ids'][25], 31647)
        self.assertEqual(records[0]['conditions']['POS_CONFIRMS']['evidence_token_ids'][25], 23849)
        self.assertEqual(records[0]['conditions']['POS_ASSERTS']['evidence_token_ids'][25], 17086)

    def test_frozen_original_source_match(self):
        source_root = ROOT / h2.SOURCE
        if not (source_root / 'pair_split_manifest.json').exists():
            self.skipTest('Original XG1 sources not in disposable local fixture')
        records = h2.validate_candidate(self.cohort, self.manifest)
        h2.validate_original_source(ROOT, records)

    def test_missing_file_and_hash_fail_closed(self):
        with self.assertRaisesRegex(h2.ContractError, 'CANDIDATE_FILES_MISSING'):
            h2.validate_candidate(ROOT / 'missing.no', self.manifest)
        with patch.object(h2, 'COHORT_SHA', '0' * 64):
            with self.assertRaisesRegex(h2.ContractError, 'COHORT_HASH'):
                h2.validate_candidate(self.cohort, self.manifest)


class MetricTest(unittest.TestCase):
    @staticmethod
    def fixture(arm='FULL'):
        size, verb = 18, 5
        outputs = {}
        for condition, value in zip(h2.CONDITIONS, (-2., 1., .5)):
            native = [[.1, -.1] for _ in range(size)]
            wrapped = deepcopy(native)
            if arm == 'FULL':
                for i in range(verb, size):
                    wrapped[i] = [native[i][0] + value, native[i][1] + value]
            outputs[condition] = {'native': native, 'wrapped': wrapped}
        return outputs

    def test_plan_regions_and_fixed_positions(self):
        r = h2.region_positions(18, 2, 5)
        self.assertEqual(r['PRE'], list(range(5)))
        self.assertEqual(r['VERB'], [5])
        self.assertEqual(r['POST8'], list(range(6, 14)))
        self.assertEqual(r['EVIDENCE'], list(range(2, 18)))
        with self.assertRaisesRegex(h2.ContractError, 'REGION_INDICES'):
            h2.region_positions(12, 2, 5)

    def test_arm_logit_interaction_and_additive_identity(self):
        logits = lambda v: [v, 0., 0.]
        zero = dict(zip(h2.CONDITIONS, [logits(4), logits(1), logits(1.2)]))
        full = dict(zip(h2.CONDITIONS, [logits(1), logits(0), logits(.2)]))
        x = h2.arm_interactions(zero, full)
        self.assertAlmostEqual(x['DELTA']['NC'], -2)
        self.assertAlmostEqual(x['DELTA']['NA'], -2)
        self.assertAlmostEqual(x['DELTA']['CA'], 0)
        self.assertAlmostEqual(x['DELTA']['NA'], x['DELTA']['NC'] + x['DELTA']['CA'])

    def test_logit_shape_nonfinite_or_missing_condition_rejected(self):
        logits = {c: [0., 0., 0.] for c in h2.CONDITIONS}
        for mutation in (lambda x: x.pop('POS_ASSERTS'),
                         lambda x: x['NEG_DENIES'].pop(),
                         lambda x: x['NEG_DENIES'].__setitem__(0, float('nan'))):
            b = deepcopy(logits)
            mutation(b)
            with self.assertRaises(h2.ContractError):
                h2.decision(b)

    def test_exact_zero_and_full_output_distances(self):
        q = h2.output_readout(self.fixture('FULL'), length=18, evidence_start=2,
                              verb_index=5, arm='FULL')
        self.assertEqual(q['distance_rms']['VERB']['NC'], 3.)
        self.assertEqual(q['distance_rms']['POST8']['NA'], 2.5)
        self.assertEqual(q['distance_rms']['POST8']['CA'], .5)
        self.assertEqual(q['distance_rms']['PRE']['NC'], 0.)
        self.assertEqual(q['prefix_max_abs_deviations']['NATIVE_PRE_NC'], 0.)
        self.assertEqual(q['q_stat_differences']['VERB']['NC']['mean'], -3.)
        z = h2.output_readout(self.fixture('ZERO'), length=18, evidence_start=2,
                              verb_index=5, arm='ZERO')
        self.assertTrue(all(v == 0 for r in z['distance_rms'].values() for v in r.values()))

    def test_zero_contribution_must_be_exact(self):
        out = self.fixture('ZERO')
        out['NEG_DENIES']['wrapped'][5][0] += 1e-9
        with self.assertRaisesRegex(h2.ContractError, 'ZERO_CONTRIBUTION_NONZERO'):
            h2.output_readout(out, length=18, evidence_start=2,
                              verb_index=5, arm='ZERO')

    def test_native_and_correction_prefix_control(self):
        out = self.fixture('FULL')
        out['POS_CONFIRMS']['native'][2][0] += 0.01
        with self.assertRaisesRegex(h2.ContractError, 'PREFIX_NUMERICAL_GATE'):
            h2.output_readout(out, length=18, evidence_start=2,
                              verb_index=5, arm='FULL')
        out = self.fixture('FULL')
        out['POS_ASSERTS']['wrapped'][1][0] += 0.01
        with self.assertRaisesRegex(h2.ContractError, 'PREFIX_NUMERICAL_GATE'):
            h2.output_readout(out, length=18, evidence_start=2,
                              verb_index=5, arm='FULL')

    def test_invalid_tensors_rejected(self):
        for edit in (lambda a: a['NEG_DENIES']['wrapped'].pop(),
                     lambda a: a['NEG_DENIES']['native'][0].append(7.),
                     lambda a: a['NEG_DENIES']['wrapped'][4].__setitem__(0, math.inf)):
            out = self.fixture('FULL')
            edit(out)
            with self.assertRaises(h2.ContractError):
                h2.output_readout(out, length=18, evidence_start=2,
                                  verb_index=5, arm='FULL')

    def test_per_pair_summary_and_failure_rules(self):
        zero = {c: [v, 0., 0.] for c, v in zip(h2.CONDITIONS, [4., 1., 1.2])}
        full = {c: [v, 0., 0.] for c, v in zip(h2.CONDITIONS, [1., 0., .2])}
        rows = {f'p{i}': h2.arm_interactions(zero, full) for i in range(120)}
        a = h2.summarize_pairs(rows)
        self.assertEqual(a['NC']['DELTA']['negative'], 120)
        self.assertEqual(a['NC']['DELTA']['positive'], 0)
        self.assertEqual(a['NC']['DELTA']['mean'], -2.)
        self.assertEqual(a['NC']['DELTA']['q90'], -2.)
        d = {'VERB': {'NC': 3., 'NA': 2.5, 'CA': .5},
             'POST8': {'NC': 3., 'NA': 2.5, 'CA': .5}}
        status = h2.descriptive_disposition(a, d)
        self.assertTrue(status['observational_selectivity_compatible_only'])
        self.assertEqual(status['causal_mediation'], 'NOT_ESTABLISHED')
        self.assertEqual(status['phase3b'], 'BLOCKED_UNCHANGED')
        d['POST8']['CA'] = 10.
        self.assertFalse(h2.descriptive_disposition(a, d)['observational_selectivity_compatible_only'])
        rows.pop('p0')
        with self.assertRaisesRegex(h2.ContractError, 'AGGREGATE_PAIR_COUNT'):
            h2.summarize_pairs(rows)


if __name__ == '__main__':
    unittest.main()
