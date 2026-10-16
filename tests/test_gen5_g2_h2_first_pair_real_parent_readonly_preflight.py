"""CPU-only runner contracts for H2 one-pair future CUDA numerical preflight.

No pretrained model, tokenizer execution, CUDA, optimizer, or user data edits.
H1 six-reference checks are exact-byte verified against committed handoff.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gen5_g2_h2_first_pair_real_parent_readonly_preflight as runner


def example_pair():
    data = {'pair_id':'example','claim_token_ids':[12,13],
            'frozen_c6_row_id':'h1-rid','conditions':{}}
    for name,verb in [('NEG_DENIES',31647),('POS_CONFIRMS',23849),('POS_ASSERTS',17086)]:
        evidence = [31,44,verb]+list(range(100,114))
        data['conditions'][name] = {
            'serialized_token_ids':[12,13,0]+evidence,
            'verb_serialized_token_index':5,
        }
    return data


class PureRunnerTest(unittest.TestCase):
    def test_pure_selftest(self):
        s=StringIO()
        with redirect_stdout(s):
            runner.selftest()
        self.assertIn('PURE_CPU_SELFTEST_PASS',s.getvalue())

    def test_128_width_feature_layout_and_runtime_label_exclusion(self):
        tables,meta=runner.first_pair_features(example_pair())
        self.assertEqual(set(tables),set(runner.FEATURES))
        self.assertEqual(meta['valid_lengths'],[20,20,20])
        self.assertEqual(meta['verb_index'],5)
        self.assertEqual(tables['attention_mask'][0],[1]*20+[0]*108)
        self.assertEqual(tables['claim_mask'][0],[1,1]+[0]*126)
        self.assertEqual(tables['evidence_mask'][0],[0,0,0]+[1]*17+[0]*108)
        self.assertEqual(tables['input_ids'][0][5],31647)
        self.assertEqual(tables['input_ids'][1][5],23849)
        self.assertEqual(tables['input_ids'][2][5],17086)
        self.assertNotIn('final_labels',tables)
        self.assertNotIn('labels',tables)

    def test_payload_one_token_drift_rejected(self):
        pair=example_pair()
        pair['conditions']['POS_CONFIRMS']['serialized_token_ids'][14]+=1
        with self.assertRaisesRegex(RuntimeError,'NOT_ONE_TOKEN_SWAP'):
            runner.first_pair_features(pair)

    def test_bad_token_count_or_verbs_rejected(self):
        pair=example_pair()
        pair['conditions']['POS_ASSERTS']['serialized_token_ids'].extend([0]*130)
        with self.assertRaisesRegex(RuntimeError,'FROZEN_SEQUENCE_LENGTH'):
            runner.first_pair_features(pair)
        pair=example_pair()
        pair['conditions']['POS_ASSERTS']['verb_serialized_token_index']=6
        with self.assertRaisesRegex(RuntimeError,'VERB_OR_ROW_ID_ALIGNMENT'):
            runner.first_pair_features(pair)

    def test_strict_logit_gate(self):
        self.assertEqual(runner.check_neg_against_h1([0.,2.,4.],[0.,2.,4.]),0.)
        self.assertLess(runner.check_neg_against_h1([0.,2.,4.],[0.,2.000005,4.]),1e-5)
        with self.assertRaisesRegex(RuntimeError,'NEG_REFERENCE_DRIFT'):
            runner.check_neg_against_h1([0.,2.,4.],[0.,2.0001,4.])
        with self.assertRaisesRegex(RuntimeError,'NEG_LOGIT_FORMAT'):
            runner.check_neg_against_h1([0.,2.,4.],[0.,float('nan'),4.])

    def test_run_root_scope_and_commit_pinning(self):
        repo=runner.ROOT
        with patch.dict(os.environ,{'CONTRAMAMBA_RUN_ARTIFACT_ROOT':
                         'reports/gen5_g2_h2_real_parent_preflight_runs/g2h2-firstpair-c1cfe06-r1'}):
            out=(repo/'reports/gen5_g2_h2_real_parent_preflight_runs/'
                 'g2h2-firstpair-c1cfe06-r1')
            self.assertEqual(runner.output_dir_contract(out,runner.BASE_HEAD),
                'reports/gen5_g2_h2_real_parent_preflight_runs/g2h2-firstpair-c1cfe06-r1')
            with self.assertRaisesRegex(RuntimeError,'RUN_NAME_OR_COMMIT'):
                runner.output_dir_contract(out, 'ff'*20)
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(RuntimeError,'OUTPUT_OUTSIDE_REPO'):
                    runner.output_dir_contract(Path(tmp)/'new',runner.BASE_HEAD)

    def test_no_gpu_in_any_default_mode(self):
        from argparse import Namespace
        with self.assertRaisesRegex(RuntimeError, 'EXPLICIT_GPU_AUTH_REQUIRED'):
            runner.runtime_contract(Namespace(authorize_g2h2_numeric_preflight=False))
        self.assertNotIn('torch', runner.__dict__)


class FrozenArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.found = ((runner.ROOT / 'reports/gen5_g2_h2_static_matched_verb_candidates/'
            'g2h2-static-matched-verbs-40a279a-r1/matched_verb_candidate.jsonl').is_file()
            and (runner.H1_ROOT / 'observation_summary.json').is_file())

    def test_real_frozen_h1_6_references_when_present(self):
        if not self.found:
            self.skipTest('Frozen H2/H1 files absent outside user repository')
        from scripts.gen5_g2_h2_three_condition_cpu_metrics import (
            COHORT, validate_candidate,
        )
        root=runner.ROOT/COHORT
        pairs=validate_candidate(root/'matched_verb_candidate.jsonl',
                                 root/'static_candidate_manifest.json')
        self.assertEqual(len(pairs),120)
        first=pairs[0]
        refs, hashes=runner.h1_references(first)
        self.assertEqual(set(refs),{f'{s}/{a}' for s in runner.SEEDS for a in runner.ARMS})
        self.assertEqual(len(hashes),6)
        self.assertTrue(all(len(x)==3 for x in refs.values()))

    def test_bad_h1_summary_sha_rejected(self):
        if not self.found:
            self.skipTest('Frozen H1 summary not installed')
        with patch.object(runner,'H1_SUMMARY_SHA','0'*64):
            with self.assertRaisesRegex(RuntimeError,'H1_SUMMARY_SHA'):
                runner.h1_references(example_pair())

    def test_static_plan_independent_of_gpu(self):
        if not self.found:
            self.skipTest('Frozen H2/H1 files absent outside user repository')
        # Original XG1 source is checked by the production --static-plan CLI.
        with patch.dict(sys.modules, {'torch': None}):
            p=runner.plan(authenticate_source=False)
        self.assertEqual(p['expected_forwards'],4)
        self.assertEqual(p['seed'],6201)
        self.assertEqual(p['arms'],['ZERO','FULL'])
        self.assertEqual(len(p['cpu_features']['input_ids']),3)
        self.assertEqual(len(p['references']),6)
        self.assertIsNone(p['science_result'])


if __name__ == '__main__':
    unittest.main()
