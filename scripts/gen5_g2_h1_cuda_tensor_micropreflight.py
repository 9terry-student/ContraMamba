#!/usr/bin/env python3
"""Opt-in CUDA *synthetic tensor* preflight for the Gen5 G2-H1 output observer.

No parent checkpoint, tokenizer, real data, trainer, task evaluation or G2 science.
The --static option checks imports only and never runs a forward or asks for CUDA.
The --cuda option is explicitly GPU dependent and must be authorized separately.

Install as scripts/gen5_g2_h1_cuda_tensor_micropreflight.py.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

import torch
from torch import nn


class TinyNative(nn.Module):
    """Small stand-in for only the native-mixer module-call interface."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, dtype=torch.float64)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[0.5, 0.1], [-0.2, 0.25]], dtype=torch.float64))
            self.proj.bias.copy_(torch.tensor([0.15, -0.3], dtype=torch.float64))
        self.masks: list[torch.Tensor | None] = []

    def forward(self, hidden_states: torch.Tensor, cache_params=None,
                cache_position=None, attention_mask=None) -> torch.Tensor:
        assert cache_params is None and cache_position is None
        self.masks.append(attention_mask)
        return self.proj(hidden_states)


class TinyWrapper(nn.Module):
    """Stand-in for G2's single native call plus out-of-band masked addition."""

    def __init__(self) -> None:
        super().__init__()
        self.native_mixer = TinyNative()
        self._phase2_active_mask: torch.Tensor | None = None

    @contextmanager
    def active_mask(self, mask: torch.Tensor):
        assert self._phase2_active_mask is None
        self._phase2_active_mask = mask
        try:
            yield
        finally:
            self._phase2_active_mask = None

    def forward(self, hidden_states: torch.Tensor, cache_params=None,
                cache_position=None, attention_mask=None) -> torch.Tensor:
        native = self.native_mixer(
            hidden_states, cache_params=cache_params,
            cache_position=cache_position, attention_mask=attention_mask,
        )
        effective = attention_mask if attention_mask is not None else self._phase2_active_mask
        if effective is None:
            raise RuntimeError('MISSING_EFFECTIVE_MASK')
        correction = (0.37 * hidden_states[..., [1, 0]]) * effective.to(
            hidden_states.dtype).unsqueeze(-1)
        return native + correction


def require(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('G2_H1_CUDA_MICRO_BLOCKED:' + code)


def load_observer():
    # Normal project import, with a fallback for direct invocation from scripts/.
    root = Path(__file__).resolve().parent.parent
    source = root / 'src'
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from contramamba.gen5_g2_h1_layer22_output_observer import (
        CELLS, G2H1PairSummaryAccumulator, Layer22OutputObserver,
        ObservationContractError,
    )
    return CELLS, G2H1PairSummaryAccumulator, Layer22OutputObserver, ObservationContractError


def run_tensor_smoke(*, use_cuda: bool) -> None:
    CELLS, Accumulator, Observer, ContractError = load_observer()
    if use_cuda:
        require(torch.cuda.is_available(), 'CUDA_UNAVAILABLE')
        require(torch.cuda.device_count() >= 1, 'NO_CUDA_DEVICE')
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.manual_seed(1022)
    model = TinyWrapper().to(device).eval().requires_grad_(False)
    parameters_before = [p.detach().clone() for p in model.parameters()]
    ordered_cells = (
        'C6_EXPLICIT_DENIAL', 'C2_NAME', 'C0_SHAM', 'C3_ROLE',
        'C5_TITLE_NAME', 'C1_TITLE', 'C4_PREDICATE',
    )
    require(set(ordered_cells) == CELLS, 'CELL_SET')
    accumulator = Accumulator()
    original_hook_counts = (
        len(model.native_mixer._forward_hooks), len(model._forward_hooks),
    )
    require(original_hook_counts == (0, 0), 'PREEXISTING_HOOK')

    max_output_abs_diff = 0.0
    max_native_abs_diff = 0.0
    with torch.inference_mode():
        for i, cell in enumerate(ordered_cells):
            x = torch.tensor([[[1., 2.], [3., -4.], [5., 6.], [99., 100.]]],
                             dtype=torch.float64, device=device) + float(i) * 0.01
            valid = torch.tensor([[1, 1, 1, 0]], device=device, dtype=torch.long)
            claim = torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.long)
            evidence = torch.tensor([[0, 1, 0, 0]], device=device, dtype=torch.long)
            input_copy, mask_copy = x.clone(), valid.clone()
            with model.active_mask(valid):
                native_reference = model.native_mixer(x)
                plain = model(x)
            # Exactly one wrapper invocation with two hooked module outputs.
            observed = Observer(
                model, attention_mask=valid, claim_mask=claim,
                evidence_mask=evidence, pair_ids=['p0'], cell_ids=[cell],
                row_ids=['synthetic_' + cell], allow_partial_pairs=True,
                allow_cuda_capture=True,
            )
            with model.active_mask(valid), observed:
                hooked = model(x)
            if use_cuda:
                torch.cuda.synchronize(device)
            record = observed.record()

            require(torch.equal(plain, hooked), 'HOOK_CHANGES_FORWARD')
            require(torch.equal(x, input_copy) and torch.equal(valid, mask_copy),
                    'INPUT_OR_MASK_MUTATION')
            require(model.native_mixer.masks[-1] is None, 'NATIVE_MASK_CHANGED')
            require(record.native.device.type == record.wrapped.device.type ==
                    record.contribution.device.type == 'cpu', 'RECORD_NOT_HOST')
            require(not record.native.requires_grad and not record.wrapped.requires_grad,
                    'RECORD_REQUIRES_GRAD')
            require(record.per_row[0]['contribution']['valid']['tokens'] == 3,
                    'VALID_TOKEN_COUNT')
            require(record.per_row[0]['contribution']['claim']['tokens'] == 1,
                    'CLAIM_TOKEN_COUNT')
            require(record.per_row[0]['contribution']['evidence']['tokens'] == 1,
                    'EVIDENCE_TOKEN_COUNT')
            require(record.per_row[0]['contribution']['other']['tokens'] == 1,
                    'OTHER_TOKEN_COUNT')
            require(not model.native_mixer._forward_hooks and not model._forward_hooks,
                    'HOOK_LEAK')
            require(model._phase2_active_mask is None, 'MASK_CONTEXT_LEAK')
            require(record.pair_differences == {}, 'UNEXPECTED_BATCH_PAIR_RESULT')

            diff = float((record.wrapped - plain.cpu()).abs().max().item())
            native_diff = float((record.native - native_reference.cpu()).abs().max().item())
            max_output_abs_diff = max(max_output_abs_diff, diff)
            max_native_abs_diff = max(max_native_abs_diff, native_diff)
            require(diff == 0.0 and native_diff == 0.0, 'CAPTURE_VALUE_CHANGED')
            torch.testing.assert_close(
                record.contribution, record.wrapped - record.native,
                atol=0.0, rtol=0.0,
            )
            accumulator.add_batch(
                record.per_row, pair_ids=['p0'], cell_ids=[cell],
                row_ids=['synthetic_' + cell],
            )
        pairs = accumulator.finalize(expected_pairs=1, require_all_seven_cells=True)
        require(set(pairs) == {'p0'}, 'PAIR_ACCUMULATION_MISMATCH')

        # Fail-closed negative control: caller supplies a different effective mask.
        x = torch.ones((1, 4, 2), dtype=torch.float64, device=device)
        valid = torch.tensor([[1, 1, 1, 0]], dtype=torch.long, device=device)
        bad_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long, device=device)
        claim = torch.tensor([[1, 0, 0, 0]], dtype=torch.long, device=device)
        evidence = torch.tensor([[0, 1, 0, 0]], dtype=torch.long, device=device)
        bad_observer = Observer(
            model, attention_mask=valid, claim_mask=claim,
            evidence_mask=evidence, pair_ids=['p1'], cell_ids=['C0_SHAM'],
            allow_partial_pairs=True, allow_cuda_capture=True,
        )
        got_expected_exception = False
        try:
            with model.active_mask(bad_mask), bad_observer:
                model(x)
        except ContractError as exc:
            got_expected_exception = 'OBSERVER_EFFECTIVE_MASK_MISMATCH' in str(exc)
        require(got_expected_exception, 'NEGATIVE_MASK_CHECK_NOT_ENFORCED')
        require(not model.native_mixer._forward_hooks and not model._forward_hooks,
                'NEGATIVE_PATH_HOOK_LEAK')
        require(model._phase2_active_mask is None, 'NEGATIVE_PATH_MASK_LEAK')
        require(all(torch.equal(a, b) for a, b in zip(parameters_before, model.parameters())),
                'PARAMETER_MUTATION')

    print('G2_H1_' + ('CUDA' if use_cuda else 'CPU') + '_TENSOR_MICROPREFLIGHT_PASS')
    print('DEVICE=' + (str(torch.cuda.get_device_name(device)) if use_cuda else 'cpu'))
    print('TOY_FORWARD_PAIRS=7')
    print('MAX_WRAPPED_CAPTURE_ABS_DIFF=' + str(max_output_abs_diff))
    print('MAX_NATIVE_CAPTURE_ABS_DIFF=' + str(max_native_abs_diff))
    print('PRETRAINED_MODEL=NOT_LOADED')
    print('TASK_EVALUATION=NOT_RUN')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--static', action='store_true',
                      help='Import-only check; no forwards, no CUDA calls')
    mode.add_argument('--cpu', action='store_true',
                      help='Run the tiny synthetic forward/capture on CPU only')
    mode.add_argument('--cuda', action='store_true',
                      help='Explicitly execute tiny CUDA synthetic forward/capture')
    args = parser.parse_args()
    if args.static:
        load_observer()
        print('G2_H1_CUDA_MICROPREFLIGHT_STATIC_PASS')
        print('CUDA=NOT_USED;MODEL_FORWARD=NOT_RUN')
    elif args.cpu:
        run_tensor_smoke(use_cuda=False)
    elif args.cuda:
        run_tensor_smoke(use_cuda=True)


if __name__ == '__main__':
    main()
