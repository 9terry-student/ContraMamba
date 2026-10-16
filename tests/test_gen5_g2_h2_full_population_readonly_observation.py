"""Pure CPU full runner contract tests; no actual model, GPU or scientific output."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0,str(path))
from scripts import gen5_g2_h2_full_population_readonly_observation as full


class FullRunnerTests(unittest.TestCase):
    def test_batch_budget_and_pair_alignment(self):
        batches = full.batch_slices()
        self.assertEqual(len(batches), 12)
        self.assertEqual((full.PAIRS, full.ROWS, full.BATCH_ROWS), (120,360,30))
        self.assertEqual(full.EXPECTED_FORWARDS, 144)
        self.assertEqual(batches[0], (0,10,0,30))
        self.assertEqual(batches[-1], (110,120,330,360))
        self.assertEqual(sum(hi-lo for lo,hi,_,_ in batches),120)
        self.assertEqual(sum(rhi-rlo for _,_,rlo,rhi in batches),360)
        for plo,phi,rlo,rhi in batches:
            self.assertEqual((rlo,rhi),(3*plo,3*phi))

    def test_exact_science_scope_preserved(self):
        self.assertEqual(full.CONDITIONS, ('NEG_DENIES','POS_CONFIRMS','POS_ASSERTS'))
        self.assertEqual(full.SEEDS,(6201,6202,6203))
        self.assertEqual(full.ARMS,('ZERO','FULL'))
        self.assertEqual(full.REFERENCE_ATOL,1e-5)
        self.assertEqual(full.BASE_HEAD,
             'a5a244b9ba998af90397867e1a10f25d40acdfef')

    def test_output_scope_and_no_overwrite(self):
        name='g2h2-full-a5a244b-r1'
        relative=f'reports/{full.FOLDER}/{name}'
        with patch.dict(os.environ,{'CONTRAMAMBA_RUN_ARTIFACT_ROOT': relative}):
            target=full.ROOT/relative
            self.assertEqual(full.output_contract(target,full.BASE_HEAD),relative)
            with self.assertRaisesRegex(RuntimeError,'OUTPUT_RUN_NAME_HEAD'):
                full.output_contract(full.ROOT/'reports'/full.FOLDER/'wrong',full.BASE_HEAD)
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaisesRegex(RuntimeError,'OUTPUT_OUTSIDE_REPO'):
                    full.output_contract(Path(td)/name,full.BASE_HEAD)
                local=Path(td)/'summary.json'
                full.write_json_once(local,{'x':1})
                with self.assertRaisesRegex(RuntimeError,'OUTPUT_COLLISION'):
                    full.write_json_once(local,{'x':2})

    def test_arm_join_with_synthetic_120_pairs(self):
        order=[f'pair-{i:03d}' for i in range(120)]
        parts={}
        for arm in full.ARMS:
            by={}
            for pair in order:
                positive=(arm=='FULL')
                logits={
                    'NEG_DENIES':[2.0 if not positive else 1.5,0.,0.],
                    'POS_CONFIRMS':[1.0,0.,0.],
                    'POS_ASSERTS':[0.8,0.,0.],
                }
                distances={r:{c:(0.0 if not positive else {'NC':3.,'NA':2.8,'CA':0.2}[c])
                              for c in ('NC','NA','CA')}
                           for r in ('PRE','VERB','POST8','EVIDENCE')}
                by[pair]={'logits':logits,'output':{'distance_rms':distances}}
            parts[arm]=by
        rows, agg, disp=full.merge_arm_pair_metrics(parts, order)
        self.assertEqual(len(rows),120)
        self.assertAlmostEqual(agg['logits']['NC']['DELTA']['mean'], -0.5)
        self.assertAlmostEqual(agg['logits']['NA']['DELTA']['mean'], -0.5)
        self.assertTrue(disp['direction'])
        self.assertFalse(disp['lexical_control_dominance'] is False)
        self.assertTrue(disp['observational_selectivity_compatible_only'])
        self.assertEqual(disp['causal_mediation'],'NOT_ESTABLISHED')
        self.assertEqual(disp['phase3b'],'BLOCKED_UNCHANGED')

    def test_wrong_pair_or_zero_correction_blocks(self):
        order=[f'p{i}' for i in range(120)]
        blank={c:[0.,0.,0.] for c in full.CONDITIONS}
        q={r:{c:0.0 for c in ('NC','NA','CA')}
           for r in ('PRE','VERB','POST8','EVIDENCE')}
        rows={pid:{'logits':blank,'output':{'distance_rms':q}} for pid in order}
        with self.assertRaisesRegex(RuntimeError,'PAIR_ARM_MISSING'):
            full.merge_arm_pair_metrics({'ZERO':rows,'FULL':{}},order)
        q_bad={r:dict(v) for r,v in q.items()}
        q_bad['VERB']['NC']=1.
        bad=dict(rows)
        bad['p0']={'logits':blank,'output':{'distance_rms':q_bad}}
        with self.assertRaisesRegex(RuntimeError,'ZERO_OUTPUT_DISTANCE'):
            full.merge_arm_pair_metrics({'ZERO':bad,'FULL':rows},order)

    def test_module_does_not_import_torch(self):
        self.assertNotIn('torch', full.__dict__)


if __name__=='__main__':
    unittest.main()
