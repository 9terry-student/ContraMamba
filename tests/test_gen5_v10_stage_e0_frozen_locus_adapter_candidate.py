#!/usr/bin/env python3
"""CPU/source-AST only. Never imports or runs the original worker."""
from __future__ import annotations

import contextlib
import io
import os
import pathlib
import sys
import unittest
from unittest.mock import patch

from scripts import gen5_v10_stage_e0_frozen_locus_adapter_candidate as e0

SOURCE = '''
def run_worker(args):
    _validate_runtime_head(args)
    validate_static_contract()
    require(args.worker_id in (0, 1), "WORKER_ID_REQUIRED")
    require(torch.cuda.is_available(), "CUDA_REQUIRED")
    static, encoded, model, wrapper, strong_mask, planes = transport._runtime_inputs(args)
    features, labels, active, targets = p3a._feature_batch_to_device(encoded["dev_bundle"], device)
    row_start, row_stop = worker_row_range(worker_id)
    for batch_index, start in enumerate(range(row_start, row_stop, BATCH_ROWS), start=1):
        stop = min(start + BATCH_ROWS, row_stop)
        batch_features = transport.legacy._phase_b_batch_features(features,start,stop)
        batch_mask = batch_features["attention_mask"]
        role_masks = coarse._serialization_role_masks(batch_features)
        fine_masks = _batch_anchor_masks(partition,local_start=0,local_stop=32,device=device)
        context = transport.legacy._phase_b_prepare_common_context(model=model,wrapper=wrapper,features=batch_features,stressor_active=active[start:stop],target_indices=targets[start:stop],strong_mask=strong_mask,planes=planes)
        pair_log_a = kernel._discrete_log_a(context,shape=wrapper.correction.shape)
        pair_survival = pairgap._backward_survival_factor(log_a_flat=pair_log_a,attention_mask=batch_mask)
        raw_by_cell = {}
        with torch.no_grad():
            for cell in FULL_FACTORIAL_CELLS:
                raw_by_cell[cell] = transport.legacy._phase_b_raw_write(context["mixer_input"],batch_mask,weights[cell][0],weights[cell][1]).detach()
        for source in local_sources:
            source_rows = by_source[source]
            _, g_refute, g_support = transport._source_gradient_bundle(model=model,wrapper=wrapper,features=batch_features,context=context,raw_source=raw_by_cell[source])
            g_refute,g_support,pinv_cpu = transport._prepare_projector_pinv(grad_refute=g_refute,grad_support=g_support,attention_mask=batch_mask)
            for group_start in range(0, len(source_rows), TARGETS_PER_TRANSPORT):
                group_rows = source_rows[group_start:group_start + TARGETS_PER_TRANSPORT]
                visible, complement = transport._project_visible_group(grad_refute=g_refute,grad_support=g_support,pinv_cpu=pinv_cpu,attention_mask=batch_mask,raw_source=raw_by_cell[source],raw_targets=raw_targets,accumulators=accumulators)
                combined = torch.cat((visible, complement), dim=0)
                stats = _component_fine_pair_stats(component=combined,log_a=pair_log_a,survival=pair_survival,attention_mask=batch_mask,role_masks=role_masks,fine_masks=fine_masks,shape=wrapper.correction.shape)
                target_count = len(group_rows)
                for index, row in enumerate(group_rows):
                    accumulator = accumulators[row]
                    visible_stats = stats[index]
                    complement_stats = stats[target_count + index]
                    _accumulate_component_row(accumulator,prefix="visible",stats=visible_stats)
                    _accumulate_component_row(accumulator,prefix="complement",stats=complement_stats)
                    require(raw_error <= threshold, "RAW_RECONSTRUCTION_AUTH")
                    require(raw_visible <= threshold, "VISIBLE_RAW_ENERGY_INTERNAL_REPLAY")
                    require(raw_complement <= threshold, "COMPLEMENT_RAW_ENERGY_INTERNAL_REPLAY")
'''

class TestE0SourceBoundCPU(unittest.TestCase):
    def ok(self, value: str = SOURCE):
        return e0.inspect_frozen_source(value.encode('utf-8'), enforce_blob=False)

    def blocked(self, value: str, code: str):
        with self.assertRaisesRegex(e0.E0Blocked, code): self.ok(value)

    def test_source_ast_pass_not_runtime(self):
        x=self.ok()
        self.assertEqual(len(x['static_callsites']), len(e0.ORDER))
        self.assertLess(x['component_accumulation_callsites']['visible'],
                        x['component_accumulation_callsites']['complement'])
        self.assertEqual(len(x['frozen_raw_guard_literals']),3)
        self.assertFalse(x['frozen_worker_direct_invocation_allowed'])
        self.assertEqual(x['readiness'],'BLOCKED_BOUND_EXECUTION_AND_INDEPENDENT_EVIDENCE')
        self.assertEqual([x[z] for z in ('E0','E1','E2','E3')],['NOT_EVALUATED']*4)
        self.assertFalse(x['real_cuda_peak_measured'])

    def test_deterministic_anchor_report(self):
        self.assertEqual(self.ok(),self.ok())

    def test_frozen_blob_wrong_blocks(self):
        with self.assertRaisesRegex(e0.E0Blocked,'FROZEN_SOURCE_BLOB_MISMATCH'):
            e0.inspect_frozen_source(SOURCE.encode(), enforce_blob=True)

    def test_git_blob_sha_stable(self):
        self.assertEqual(e0.git_blob_sha(b'hello'), 'b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0')

    def test_bad_source_bytes(self):
        for x in (None,'str',b''):
            with self.subTest(x=x),self.assertRaises(e0.E0Blocked):e0.inspect_frozen_source(x)

    def test_invalid_python_blocks(self):self.blocked('def run_worker(:','INVALID_SOURCE')
    def test_missing_entrypoint(self):self.blocked(SOURCE.replace('run_worker(args)','run_all(args)'), 'ORIGINAL_WORKER_ENTRYPOINT')
    def test_duplicate_entrypoint(self):self.blocked(SOURCE+SOURCE,'ORIGINAL_WORKER_ENTRYPOINT')
    def test_missing_batch_loop(self):self.blocked(SOURCE.replace('for batch_index, start in enumerate(range(row_start, row_stop, BATCH_ROWS), start=1):','for start in range(1):'),'EXACT_FROZEN_LOOP_NESTING')
    def test_wrong_batch_loop(self):self.blocked(SOURCE.replace('range(row_start, row_stop, BATCH_ROWS)','range(0, 840, BATCH_ROWS)'),'EXACT_FROZEN_LOOP_NESTING')
    def test_missing_full_raw_loop(self):self.blocked(SOURCE.replace('for cell in FULL_FACTORIAL_CELLS:','for cell in selected_cells:'),'EXACT_FROZEN_LOOP_NESTING')
    def test_missing_source_loop(self):self.blocked(SOURCE.replace('for source in local_sources:','for source in [local_sources[0]]:'),'EXACT_FROZEN_LOOP_NESTING')
    def test_missing_group_loop(self):self.blocked(SOURCE.replace('range(0, len(source_rows), TARGETS_PER_TRANSPORT)','range(0, 4, 2)'),'EXACT_FROZEN_LOOP_NESTING')
    def test_wrong_k4_cat(self):self.blocked(SOURCE.replace('torch.cat((visible, complement), dim=0)','torch.cat((complement, visible), dim=0)'),'EXACT_K4_CONCAT')
    def test_wrong_cat_dimension(self):self.blocked(SOURCE.replace('torch.cat((visible, complement), dim=0)','torch.cat((visible, complement), dim=1)'),'EXACT_K4_CONCAT')
    def test_frozen_stats_replaced(self):self.blocked(SOURCE.replace('stats = _component_fine_pair_stats','stats = synthetic_stats'),'CALLSITE_CARDINALITY')
    def test_frozen_stats_wrong_inputs(self):self.blocked(SOURCE.replace('component=combined,log_a=pair_log_a','component=combined.clone(),log_a=pair_log_a'),'FROZEN_STATS_ARGUMENT_BINDING')
    def test_frozen_stats_wrong_mask(self):self.blocked(SOURCE.replace('fine_masks=fine_masks','fine_masks=other_fine_masks'),'FROZEN_STATS_ARGUMENT_BINDING')
    def test_raw_write_replaced(self):self.blocked(SOURCE.replace('transport.legacy._phase_b_raw_write','cached_raw'), 'CALLSITE_CARDINALITY')
    def test_projector_replaced(self):self.blocked(SOURCE.replace('transport._project_visible_group','some_projection'), 'CALLSITE_CARDINALITY')
    def test_gradient_replaced(self):self.blocked(SOURCE.replace('transport._source_gradient_bundle','fake_grad'), 'CALLSITE_CARDINALITY')
    def test_no_guard_raw(self):self.blocked(SOURCE.replace('"RAW_RECONSTRUCTION_AUTH"','"RAW_AUTH_SKIPPED"'), 'FROZEN_GUARD_MISSING')
    def test_no_guard_visible(self):self.blocked(SOURCE.replace('"VISIBLE_RAW_ENERGY_INTERNAL_REPLAY"','"VISIBLE_SKIPPED"'), 'FROZEN_GUARD_MISSING')
    def test_no_guard_complement(self):self.blocked(SOURCE.replace('"COMPLEMENT_RAW_ENERGY_INTERNAL_REPLAY"','"COMP_SKIPPED"'), 'FROZEN_GUARD_MISSING')
    def test_no_model_preload_gate(self):self.blocked(SOURCE.replace('_validate_runtime_head(args)','pass'), 'PRELOAD_SOURCE_GATE')
    def test_reducer_before_projection(self):
        old='                visible, complement = transport._project_visible_group'
        move='                stats = _component_fine_pair_stats(component=combined,log_a=pair_log_a,survival=pair_survival,attention_mask=batch_mask,role_masks=role_masks,fine_masks=fine_masks,shape=wrapper.correction.shape)\n'
        text=SOURCE.replace(move,'').replace(old,move+old)
        self.blocked(text,'CALLSITE_ORDER')
    def test_order_moved_stats_without_guards(self):self.blocked(SOURCE.replace('"COMPLEMENT_RAW_ENERGY_INTERNAL_REPLAY"','"INCORRECT"'), 'FROZEN_GUARD_MISSING')
    def test_missing_complement_accumulation_blocks(self):
        self.blocked(SOURCE.replace(
            '                    _accumulate_component_row(accumulator,prefix="complement",stats=complement_stats)\n',
            ''), 'COMPONENT_ACCUMULATION_CARDINALITY')

    def test_duplicate_visible_accumulation_blocks(self):
        self.blocked(SOURCE.replace(
            'prefix="complement",stats=complement_stats',
            'prefix="visible",stats=visible_stats'), 'COMPONENT_ACCUMULATION_BINDING')

    def test_swapped_component_accumulation_order_blocks(self):
        a='                    _accumulate_component_row(accumulator,prefix="visible",stats=visible_stats)\n'
        b='                    _accumulate_component_row(accumulator,prefix="complement",stats=complement_stats)\n'
        self.blocked(SOURCE.replace(a+b,b+a), 'COMPONENT_ACCUMULATION_BINDING')

    def test_wrong_complement_index_binding_blocks(self):
        self.blocked(SOURCE.replace('stats[target_count + index]', 'stats[index + 1]'),
                     'COMPONENT_INDEX_BINDING')

    def test_cpu_contract_no_model_or_gpu_import(self):
        import inspect
        src=inspect.getsource(e0)
        self.assertNotIn('import torch',src)
        self.assertNotIn('torch.cuda.is_available()',src)
        self.assertNotIn('eval(',src)
        self.assertNotIn('exec(',src)
    def test_cli_static(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(sys,'argv',['e0','--static-contract-check']),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(e0.main(),0)
        self.assertIn('NOT_EVALUATED',output.getvalue())
    def test_cli_no_run_bounded(self):
        with patch.object(sys,'argv',['e0','--run-bounded']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):e0.main()
    def test_cli_gpu_visible_blocks(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(sys,'argv',['e0','--static-contract-check']):
            with self.assertRaisesRegex(e0.E0Blocked,'CUDA_VISIBLE_DISABLED_REQUIRED'):e0.main()
    def test_git_metadata_mismatch(self):
        with patch.object(e0,'_git',return_value=b'bad'),patch.object(e0.Path,'is_dir',return_value=True):
            with self.assertRaisesRegex(e0.E0Blocked,'HEAD_MISMATCH'):e0.inspect_pinned_repo(pathlib.Path('.'))
    def test_git_missing_directory(self):
        with self.assertRaisesRegex(e0.E0Blocked,'REPO_DIRECTORY'): e0.inspect_pinned_repo(pathlib.Path('/e0-missing-repo'))
    def test_pinned_git_source_calls_inspector_without_execute(self):
        vals={('rev-parse','HEAD'):e0.BASE.encode(),('branch','--show-current'):e0.BRANCH.encode(),
              ('diff','--name-only'):b'',('diff','--cached','--name-only'):b'',
              ('rev-parse','HEAD:'+e0.ORIGINAL):e0.ORIGINAL_BLOB.encode(),
              ('show','HEAD:'+e0.ORIGINAL):SOURCE.encode()}
        def git(repo,*args):return vals[args]
        with patch.object(e0,'_git',side_effect=git),patch.object(e0,'inspect_frozen_source',side_effect=lambda raw:{'frozen_git_blob':'fixture','static_callsites':{},'frozen_raw_guard_literals':{}}) as inspect:
            out=e0.inspect_pinned_repo(pathlib.Path('.'))
            self.assertEqual(out['source_read_mode'],'GIT_SHOW_COMMITTED_BLOB_NOT_RUNTIME_LOADED_SOURCE')
            inspect.assert_called_once_with(SOURCE.encode())
    def test_git_io_failure(self):
        with patch.object(e0.subprocess,'run',side_effect=OSError('offline')):
            with self.assertRaisesRegex(e0.E0Blocked,'GIT_IO'): e0._git(pathlib.Path('.'),'rev-parse','HEAD')

if __name__ == '__main__':unittest.main()
