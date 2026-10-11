#!/usr/bin/env python3
"""E1 original-helper call-budget tests, CPU synthetic injection only."""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from scripts import gen5_v10_stage_e1_bounded_cpu_dispatch_candidate as e1
from scripts import gen5_v10_bounded_runtime_bridge_candidate as c
from scripts import gen5_v10_hybrid_fine_pair_candidate as hybrid
from test_gen5_v10_bounded_runtime_bridge_candidate import TestStageCBridgeCPU


class TestE1BoundedCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        kw, frozen, ids, events = TestStageCBridgeCPU().fixture()
        d = dict(frozen=frozen, hybrid=hybrid, row_ids=ids,
                 static={}, encoded={}, tokenizer_snapshot=None,
                 model=SimpleNamespace(parameters=lambda: []),
                 wrapper=SimpleNamespace(correction=SimpleNamespace(shape=kw['shape'])),
                 strong_mask=None, planes=None, features=None,
                 active=[0] * 840, targets=[0] * 840)
        return d, events

    def run_cpu(self, kw):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            return e1.execute_injected_cpu_one_group(**kw)

    def test_exact_one_group_and_original_helper_counts(self):
        kw, events = self.fixture()
        r = self.run_cpu(kw)
        self.assertEqual(tuple(r['original_helper_trace']), e1.ORDER)
        self.assertEqual(r['original_helper_counts'], e1.CALL_BUDGET)
        self.assertEqual(events.count('raw_write'), 9)
        self.assertEqual(events.count('source_grad'), 1)
        self.assertEqual(events.count('projector_pinv'), 1)
        self.assertEqual(events.count('projector'), 1)
        self.assertEqual(events.count('original_full_fine_stats'), 1)

    def test_selector_and_numerical_differential_with_stage_c(self):
        a, _ = self.fixture()
        b, _ = self.fixture()
        observed = self.run_cpu(a)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            expected = c.execute_one_group_cpu_mock(**b)
        self.assertEqual(observed['selected_locus'], expected['selector'])
        self.assertEqual(observed['input_manifest'], expected['input_manifest'])
        self.assertEqual(observed['frozen_records'], expected['frozen_records'])
        self.assertEqual(observed['v10_records'], expected['v10_records'])
        self.assertEqual(observed['v10_dispatch'], expected['v10_dispatch'])

    def test_unverified_science_floor(self):
        kw, _ = self.fixture()
        r = self.run_cpu(kw)
        self.assertEqual(r['SCIENTIFIC_EXECUTION'], 'NOT_AUTHORIZED')
        self.assertEqual(r['PRODUCTION'], 'BLOCKED')
        self.assertFalse(r['runtime_origin_attested'])
        for k in ('V10_E0_PROVENANCE', 'V10_E1_FROZEN_GUARDS',
                  'V10_E2_PAYLOAD', 'V10_E3_DIAGNOSTICS'):
            self.assertEqual(r[k], 'NOT_EVALUATED')
        self.assertEqual(len(r['input_manifest']['tensors']), 17)

    def test_no_original_dispatch_called(self):
        kw, _ = self.fixture()
        kw['frozen'].run_worker = lambda args: (_ for _ in ()).throw(RuntimeError('FORBIDDEN_RUN_WORKER'))
        kw['frozen'].run_full = lambda args: (_ for _ in ()).throw(RuntimeError('FORBIDDEN_RUN_FULL'))
        self.run_cpu(kw)

    def test_proxy_does_not_patch_original(self):
        kw, _ = self.fixture()
        original = kw['frozen']
        raw = original.transport.legacy._phase_b_raw_write
        grad = original.transport._source_gradient_bundle
        proxy = e1._BudgetedFrozen(original)
        self.assertIs(original.transport.legacy._phase_b_raw_write, raw)
        self.assertIs(original.transport._source_gradient_bundle, grad)
        self.assertIsNot(proxy.transport.legacy._phase_b_raw_write, raw)
        self.run_cpu(kw)
        self.assertIs(original.transport.legacy._phase_b_raw_write, raw)

    def test_wrong_order_fails_before_helper(self):
        kw, _ = self.fixture()
        proxy = e1._BudgetedFrozen(kw['frozen'])
        with self.assertRaisesRegex(e1.E1Blocked, 'ORIGINAL_HELPER_ORDER'):
            proxy.transport._source_gradient_bundle()
        self.assertEqual(proxy.trace, [])

    def test_extra_raw_call_after_nine_fails(self):
        kw, _ = self.fixture()
        proxy = e1._BudgetedFrozen(kw['frozen'])
        proxy.trace = list(e1.ORDER[:13])
        proxy.counts.update({'raw_write': 9})
        with self.assertRaisesRegex(e1.E1Blocked, 'ORIGINAL_HELPER_ORDER'):
            proxy.transport.legacy._phase_b_raw_write(None, None, None, None)

    def test_early_abnormal_return_does_not_record(self):
        kw, _ = self.fixture()
        def fail(*args, **kwargs):
            raise RuntimeError('READ_FAILED')
        kw['frozen'].transport.legacy._phase_b_batch_features = fail
        proxy = e1._BudgetedFrozen(kw['frozen'])
        with self.assertRaisesRegex(RuntimeError, 'READ_FAILED'):
            proxy.transport.legacy._phase_b_batch_features(None, 800, 832)
        self.assertEqual(proxy.trace, [])

    def test_incomplete_budget_never_passes(self):
        kw, _ = self.fixture()
        proxy = e1._BudgetedFrozen(kw['frozen'])
        with self.assertRaisesRegex(e1.E1Blocked, 'INCOMPLETE_ORIGINAL_HELPER_TRACE'):
            proxy.finish()

    def test_duplicate_ids_block_before_helpers(self):
        kw, events = self.fixture()
        kw['row_ids'][1] = kw['row_ids'][0]
        with self.assertRaisesRegex(e1.E1Blocked, 'ROW_IDS_EXACT'):
            self.run_cpu(kw)
        self.assertEqual(events, [])

    def test_invalid_row_ids_type_blocks(self):
        for value in ('abc', tuple(str(x) for x in range(32)), [1]*32, []):
            with self.subTest(value=type(value).__name__):
                kw, events = self.fixture()
                kw['row_ids'] = value
                with self.assertRaisesRegex(e1.E1Blocked, 'ROW_IDS_EXACT'):
                    self.run_cpu(kw)
                self.assertEqual(events, [])

    def test_insufficient_active_blocks_prework(self):
        kw, events = self.fixture()
        kw['active'] = [0]*831
        with self.assertRaisesRegex(e1.E1Blocked, 'INJECTED_ROW_CAPACITY'):
            self.run_cpu(kw)
        self.assertEqual(events, [])

    def test_insufficient_targets_blocks_prework(self):
        kw, events = self.fixture()
        kw['targets'] = [0]*831
        with self.assertRaisesRegex(e1.E1Blocked, 'INJECTED_ROW_CAPACITY'):
            self.run_cpu(kw)
        self.assertEqual(events, [])

    def test_prod_geometry_forbidden_prework(self):
        kw, events = self.fixture()
        kw['wrapper'].correction.shape = SimpleNamespace(intermediate_size=1, state_size=24576)
        with self.assertRaisesRegex(e1.E1Blocked, 'CPU_SYNTHETIC_WIDTH_ONLY'):
            self.run_cpu(kw)
        self.assertEqual(events, [])

    def test_bool_shape_blocks_before_work(self):
        kw, events = self.fixture()
        kw['wrapper'].correction.shape = SimpleNamespace(intermediate_size=True, state_size=8)
        with self.assertRaisesRegex(e1.E1Blocked, 'CPU_SYNTHETIC_SHAPE_INTEGER'):
            self.run_cpu(kw)
        self.assertEqual(events, [])

    def test_no_shape_blocks_prework(self):
        kw, events = self.fixture()
        kw['wrapper'] = SimpleNamespace()
        with self.assertRaisesRegex(e1.E1Blocked, 'INJECTED_SHAPE'):
            self.run_cpu(kw)
        self.assertEqual(events, [])

    def test_cuda_visibility_forbidden_prework(self):
        kw, events = self.fixture()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(e1.E1Blocked, 'GPU_VISIBLE'):
                e1.execute_injected_cpu_one_group(**kw)
        self.assertEqual(events, [])

    def test_invalid_orientation_blocks_without_raw(self):
        kw, events = self.fixture()
        kw['frozen'].FULL_FACTORIAL_CELLS = tuple(reversed(kw['frozen'].FULL_FACTORIAL_CELLS))
        with self.assertRaises(Exception):
            self.run_cpu(kw)
        self.assertNotIn('raw_write', events)

    def test_raw_failure_stops_gradient(self):
        kw, events = self.fixture()
        def fail(*args):
            raise RuntimeError('RAW_WRITE_BROKEN')
        kw['frozen'].transport.legacy._phase_b_raw_write = fail
        with self.assertRaisesRegex(RuntimeError, 'RAW_WRITE_BROKEN'):
            self.run_cpu(kw)
        self.assertNotIn('source_grad', events)

    def test_gradient_failure_stops_projector(self):
        kw, events = self.fixture()
        def fail(**args):
            raise RuntimeError('SOURCE_GRADIENT_BROKEN')
        kw['frozen'].transport._source_gradient_bundle = fail
        with self.assertRaisesRegex(RuntimeError, 'SOURCE_GRADIENT_BROKEN'):
            self.run_cpu(kw)
        self.assertNotIn('projector_pinv', events)
        self.assertNotIn('projector', events)

    def test_projector_failure_stops_frozen(self):
        kw, events = self.fixture()
        def fail(**args):
            raise RuntimeError('PROJECTION_BROKEN')
        kw['frozen'].transport._project_visible_group = fail
        with self.assertRaisesRegex(RuntimeError, 'PROJECTION_BROKEN'):
            self.run_cpu(kw)
        self.assertNotIn('original_full_fine_stats', events)

    def test_frozen_failure_stops_v10(self):
        kw, events = self.fixture()
        def fail(**args):
            raise RuntimeError('FROZEN_FFT_BROKEN')
        kw['frozen']._component_fine_pair_stats = fail
        with self.assertRaisesRegex(RuntimeError, 'FROZEN_FFT_BROKEN'):
            self.run_cpu(kw)
        self.assertNotIn('original_full_fine_stats', events)

    def test_guard_failure_never_returns_report(self):
        kw, events = self.fixture()
        old_acc = kw['frozen'].transport._empty_orientation
        def fake_acc(*args):
            acc = old_acc(*args)
            acc['raw_reconstruction_max_abs'] = 100.0
            return acc
        kw['frozen'].transport._empty_orientation = fake_acc
        with self.assertRaisesRegex(c.StageCBlocked, 'RAW_RECONSTRUCTION_AUTH'):
            self.run_cpu(kw)
        self.assertIn('original_full_fine_stats', events)

    def test_no_unknown_extra_executor_option(self):
        with patch.object(sys, 'argv', ['e1', '--run-bounded']), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as err:
                e1.main()
        self.assertNotEqual(err.exception.code, 0)

    def test_static_cli_inert(self):
        with patch.object(sys, 'argv', ['e1', '--static-contract-check']), \
             patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(e1.main(), 0)
        self.assertIn('GPU=OFF', output.getvalue())
        self.assertIn('E0_E3=NOT_EVALUATED', output.getvalue())

    def test_static_cli_blocks_cuda(self):
        with patch.object(sys, 'argv', ['e1', '--static-contract-check']), \
             patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(e1.E1Blocked, 'GPU_VISIBLE'):
                e1.main()

    def test_static_blob_inspector_on_injected_frozen_snippet(self):
        from scripts import gen5_v10_stage_e0_frozen_locus_adapter_candidate as e0
        self.assertEqual(e1.ORIGINAL_BLOB, e0.ORIGINAL_BLOB)
        self.assertEqual(e1.BRANCH, e0.BRANCH)
        self.assertEqual(e1.BASE, 'c3c75af0e232252c6e1c0ec0ff3f0d1544c3a8fd')

    def test_git_source_head_mismatch_blocks(self):
        from pathlib import Path
        from unittest.mock import MagicMock
        def fake_run(argv, **kw):
            return SimpleNamespace(returncode=0, stdout=b'0'*40+b'\n')
        with patch.object(e1, '__file__', str(Path('/tmp/repo/scripts/e1.py'))), \
             patch('subprocess.run', side_effect=fake_run):
            with self.assertRaisesRegex(e1.E1Blocked, 'LOCAL_HEAD_MISMATCH'):
                e1.inspect_local_source()

    def test_git_source_blob_mismatch_blocks(self):
        def fake_run(argv, **kw):
            args=argv[3:]
            if args==['rev-parse','HEAD']: out=e1.BASE.encode()
            elif args==['branch','--show-current']:out=e1.BRANCH.encode()
            else:out=b'0'*40
            return SimpleNamespace(returncode=0,stdout=out+b'\n')
        with patch('subprocess.run', side_effect=fake_run):
            with self.assertRaisesRegex(e1.E1Blocked, 'ORIGINAL_BLOB_MISMATCH'):
                e1.inspect_local_source()

    def test_git_source_valid_report_not_runtime(self):
        from scripts import gen5_v10_stage_e0_frozen_locus_adapter_candidate as e0
        def fake_run(argv, **kw):
            args=argv[3:]
            if args==['rev-parse','HEAD']:out=e1.BASE.encode()+b'\n'
            elif args==['branch','--show-current']:out=e1.BRANCH.encode()+b'\n'
            elif args[0]=='rev-parse':out=e1.ORIGINAL_BLOB.encode()+b'\n'
            elif args[0]=='diff':out=b''
            elif args[0]=='show':out=b'original mocked body'
            else:raise AssertionError(args)
            return SimpleNamespace(returncode=0,stdout=out)
        with patch('subprocess.run', side_effect=fake_run), \
             patch.object(e0, 'inspect_frozen_source', return_value={
                'frozen_git_blob':e1.ORIGINAL_BLOB,
                'frozen_worker_direct_invocation_allowed':False,
                'E0':'NOT_EVALUATED'}) as inspect:
            r=e1.inspect_local_source()
            self.assertEqual(r['E0'],'NOT_EVALUATED')
            inspect.assert_called_once_with(b'original mocked body')

    def test_unsupported_gpu_action_not_present(self):
        from pathlib import Path
        source = Path(e1.__file__).read_text(encoding='utf-8')
        self.assertNotIn('def run_worker(', source)
        self.assertNotIn('def run_full(', source)
        self.assertNotIn('torch.cuda.', source)
        self.assertNotIn('transport._runtime_inputs(', source)


if __name__ == '__main__':
    unittest.main()
