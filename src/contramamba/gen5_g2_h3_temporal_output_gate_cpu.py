"""Gen5 G2-H3 temporal mixer-output gate: CPU synthetic contract ONLY.

No pretrained model, tokenizer, CUDA, training, or scientific execution.
An eventual real-parent intervention requires separate CUDA equivalence gates and
specific scientific execution authorization. Do not call this raw recurrent-state
intervention or causal mediation evidence.

At the L22 *wrapper output* boundary, h_native=N and h_full=F.
For each row, W={verb_index,...,verb_index+8}. Output is selected without
arithmetic mixing to preserve exact bitwise NONE=N and ALL=F endpoints:
  NONE         N everywhere
  ALL          F everywhere
  ONLY_WINDOW  F on W, N elsewhere
  DROP_WINDOW  N on W, F elsewhere

The mask spans the full 128-column tensor; for DROP_WINDOW it is FULL even on
padding positions, preserving the global ALL/ONLY/DROP/NONE complement. No
in-place mutation, no model-weight or recurrent-state writes are allowed.
"""
from __future__ import annotations

from collections.abc import Sequence
from weakref import WeakSet

import torch
from torch import nn

REGIMES = ('NONE', 'ALL', 'ONLY_WINDOW', 'DROP_WINDOW')
CONDITIONS = ('NEG_DENIES', 'POS_CONFIRMS', 'POS_ASSERTS')
VERB_IDS = {'NEG_DENIES': 31647, 'POS_CONFIRMS': 23849, 'POS_ASSERTS': 17086}
WIDTH = 128
CHANNELS = 768
WINDOW_LENGTH = 9
MAX_PAIR_BATCH = 10
_ACTIVE: WeakSet[nn.Module] = WeakSet()


class H3GateContractError(RuntimeError):
    """Any failure is a stop condition. Never auto-rerun a failed gate."""


def require(ok: bool, code: str) -> None:
    if not ok:
        raise H3GateContractError('G2_H3_TEMPORAL_GATE_BLOCKED:' + code)


def _binary_cpu_mask(value: torch.Tensor, shape: tuple[int, int], name: str) -> torch.Tensor:
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and tuple(value.shape) == shape and value.ndim == 2
            and value.dtype in (torch.bool, torch.uint8, torch.int16,
                                torch.int32, torch.int64), name + '_SHAPE_DTYPE_DEVICE')
    result = value.detach().clone()
    require(bool(((result == 0) | (result == 1)).all().item()), name + '_NONBINARY')
    return result.bool()


def frozen_window_mask(*, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor, claim_mask: torch.Tensor,
                       evidence_mask: torch.Tensor, pair_ids: Sequence[str],
                       conditions: Sequence[str], row_ids: Sequence[str],
                       verb_indices: Sequence[int]) -> torch.Tensor:
    """Exact CPU metadata contract, rechecks triplets, masks and single-token swap.

    The frozen cohort's SHA/source identity is a SEPARATE prerequisite. This
    function never retokenizes or reads files; its output is a detached CPU bool
    mask with nine true values per row.
    """
    require(isinstance(input_ids, torch.Tensor) and input_ids.device.type == 'cpu'
            and input_ids.ndim == 2 and input_ids.dtype == torch.long,
            'INPUT_CPU_LONG')
    b, l = tuple(input_ids.shape)
    require(l == WIDTH and b in range(3, MAX_PAIR_BATCH * 3 + 1, 3),
            'BATCH_SHAPE_COMPLETE_TRIPLETS')
    shape = (b, WIDTH)
    ids = input_ids.detach().clone()
    valid = _binary_cpu_mask(attention_mask, shape, 'ATTENTION')
    claim = _binary_cpu_mask(claim_mask, shape, 'CLAIM')
    evidence = _binary_cpu_mask(evidence_mask, shape, 'EVIDENCE')
    require(all(len(seq) == b for seq in (pair_ids, conditions, row_ids, verb_indices)),
            'METADATA_LENGTH')
    require(len(set(row_ids)) == b and len(set(pair_ids)) == b // 3,
            'PAIR_OR_ROW_DUPLICATE')
    windows = torch.zeros(shape, dtype=torch.bool)
    for base in range(0, b, 3):
        pid = pair_ids[base]
        require(type(pid) is str and bool(pid) and
                tuple(pair_ids[base:base + 3]) == (pid, pid, pid),
                'TRIPLET_PAIR_ID')
        require(tuple(conditions[base:base + 3]) == CONDITIONS,
                'TRIPLET_CONDITION_ORDER')
        require(all(row_ids[base + j] == pid + '__g2_h2__' + c.lower()
                    for j, c in enumerate(CONDITIONS)), 'H2_ROW_ID')
        t = verb_indices[base]
        require(type(t) is int and
                all(type(verb_indices[base + j]) is int and
                    verb_indices[base + j] == t for j in range(3)),
                'TRIPLET_VERB_POSITION')
        first = ids[base]
        valid_len = int(valid[base].sum().item())
        claim_len = int(claim[base].sum().item())
        evidence_len = int(evidence[base].sum().item())
        require(1 <= claim_len <= 63 and 1 <= evidence_len <= 64 and
                valid_len == claim_len + 1 + evidence_len and
                valid_len <= WIDTH and claim_len + 1 <= t and
                t + WINDOW_LENGTH <= valid_len,
                'SERIALIZATION_OR_WINDOW_BUDGET')
        require(int(first[claim_len].item()) == 0 and
                not bool(first[valid_len:].any().item()), 'EOS_AND_PADDING')
        for j, cond in enumerate(CONDITIONS):
            idx = base + j
            require(torch.equal(valid[idx], valid[base]) and
                    torch.equal(claim[idx], claim[base]) and
                    torch.equal(evidence[idx], evidence[base]),
                    'TRIPLET_MASK_MISMATCH')
            require(bool(valid[idx, :valid_len].all().item()) and
                    not bool(valid[idx, valid_len:].any().item()) and
                    bool(claim[idx, :claim_len].all().item()) and
                    not bool(claim[idx, claim_len:].any().item()) and
                    not bool(evidence[idx, :claim_len + 1].any().item()) and
                    bool(evidence[idx, claim_len + 1:valid_len].all().item()) and
                    not bool(evidence[idx, valid_len:].any().item()),
                    'TRIPLET_MASK_ALIGNMENT')
            require(int(ids[idx, t].item()) == VERB_IDS[cond], 'VERB_TOKEN_ID')
            require(torch.equal(ids[idx, :t], first[:t]) and
                    torch.equal(ids[idx, t+1:], first[t+1:]),
                    'NOT_SINGLE_TOKEN_SWAP')
            windows[idx, t:t + WINDOW_LENGTH] = True
    require(int(windows.sum().item()) == b * WINDOW_LENGTH and
            not bool((windows & ~valid).any().item()), 'WINDOW_COUNT_OR_VALIDITY')
    return windows


class H3TemporalOutputGateCPU:
    """One-forward context manager for an already-frozen FULL G2 wrapper.

    Hook sites: native_mixer post-forward records N; wrapper post-forward
    selects native/FULL output, so the downstream backbone sees the intervention.
    The hook never rewrites the recurrence, parameters, native output or q.

    CPU ONLY. Never use in a GPU experiment without a new explicitly verified
    adapter/runtime gate. The prototype is not a full scientific runner.
    """

    def __init__(self, wrapper: nn.Module, *, regime: str,
                 input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 claim_mask: torch.Tensor, evidence_mask: torch.Tensor,
                 pair_ids: Sequence[str], conditions: Sequence[str],
                 row_ids: Sequence[str], verb_indices: Sequence[int]):
        require(regime in REGIMES, 'REGIME_UNKNOWN')
        require(isinstance(wrapper, nn.Module) and
                isinstance(getattr(wrapper, 'native_mixer', None), nn.Module)
                and wrapper.native_mixer is not wrapper, 'FROZEN_WRAPPER_REQUIRED')
        require(getattr(wrapper, 'arm', None) == 'FULL', 'FROZEN_FULL_ARM_REQUIRED')
        require(not wrapper.training and not wrapper.native_mixer.training and
                all(not p.requires_grad for p in wrapper.parameters()),
                'FROZEN_EVAL_ONLY')
        window_mask = frozen_window_mask(
            input_ids=input_ids, attention_mask=attention_mask,
            claim_mask=claim_mask, evidence_mask=evidence_mask,
            pair_ids=pair_ids, conditions=conditions, row_ids=row_ids,
            verb_indices=verb_indices)
        require(all(t.device.type == 'cpu' for t in
                    list(wrapper.parameters()) + list(wrapper.buffers())),
                'MODEL_NOT_CPU')
        self.wrapper = wrapper
        self.native = wrapper.native_mixer
        self.regime = regime
        self.window = window_mask.detach().clone()
        self._native_handle = None
        self._wrapper_handle = None
        self._native_value: torch.Tensor | None = None
        self._native_live: torch.Tensor | None = None
        self._native_calls = 0
        self._wrapped_calls = 0
        self._entered = False
        self._finished = False

    def _tensor(self, value: object, label: str) -> torch.Tensor:
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and tuple(value.shape) == (self.window.shape[0], WIDTH, CHANNELS)
                and value.is_floating_point() and not value.requires_grad,
                label + '_SHAPE_DEVICE_GRAD')
        require(bool(torch.isfinite(value).all().item()), label + '_NONFINITE')
        return value

    def _native_hook(self, _module: nn.Module, _args: tuple,
                     _kwargs: dict, output: object) -> None:
        require(self._entered and self._native_calls == 0 and
                self._wrapped_calls == 0, 'NATIVE_CALL_ORDER')
        self._native_live = self._tensor(output, 'NATIVE')
        self._native_value = self._native_live.detach().clone()
        self._native_calls += 1

    def _wrapper_hook(self, _module: nn.Module, _args: tuple,
                      _kwargs: dict, output: object) -> torch.Tensor:
        require(self._native_calls == 1 and self._wrapped_calls == 0,
                'WRAPPER_CALL_ORDER')
        wrapped = self._tensor(output, 'FULL')
        native = self._native_value
        require(native is not None and wrapped.dtype == native.dtype,
                'NATIVE_FULL_DTYPE')
        require(self._native_live is not None and
                torch.equal(native, self._native_live), 'NATIVE_OUTPUT_MUTATED')
        self._wrapped_calls += 1
        if self.regime == 'ALL':
            return wrapped
        if self.regime == 'NONE':
            return native
        if self.regime == 'ONLY_WINDOW':
            return torch.where(self.window.unsqueeze(-1), wrapped, native)
        return torch.where(self.window.unsqueeze(-1), native, wrapped)

    def __enter__(self) -> 'H3TemporalOutputGateCPU':
        require(not torch.is_grad_enabled(), 'AUTOGRAD_ENABLED')
        require(not self._entered and not self._finished and
                self.wrapper not in _ACTIVE, 'REENTRANT_GATE')
        require(not self.wrapper._forward_hooks and not self.native._forward_hooks,
                'PREEXISTING_HOOKS')
        self._entered = True
        _ACTIVE.add(self.wrapper)
        try:
            self._native_handle = self.native.register_forward_hook(
                self._native_hook, with_kwargs=True)
            self._wrapper_handle = self.wrapper.register_forward_hook(
                self._wrapper_hook, with_kwargs=True)
        except BaseException:
            self._remove_hooks()
            self._entered = False
            _ACTIVE.discard(self.wrapper)
            raise
        return self

    def _remove_hooks(self) -> None:
        if self._wrapper_handle is not None:
            self._wrapper_handle.remove()
            self._wrapper_handle = None
        if self._native_handle is not None:
            self._native_handle.remove()
            self._native_handle = None
        self._native_value = None
        self._native_live = None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        require(self._entered and not self._finished, 'GATE_EXIT_STATE')
        self._remove_hooks()
        _ACTIVE.discard(self.wrapper)
        self._entered = False
        self._finished = True
        if exc_type is None:
            require(self._native_calls == self._wrapped_calls == 1,
                    'EXACT_ONE_FORWARD_REQUIRED')
        require(not self.wrapper._forward_hooks and not self.native._forward_hooks,
                'HOOK_NOT_RESTORED')
        return False

    def audit(self) -> dict[str, object]:
        require(self._finished and not self._entered and
                self._native_calls == self._wrapped_calls == 1,
                'AUDIT_REQUIRES_ONE_COMPLETED_FORWARD')
        return {'regime': self.regime, 'rows': self.window.shape[0],
                'window_tokens': int(self.window.sum().item()),
                'native_calls': self._native_calls,
                'wrapper_calls': self._wrapped_calls,
                'device': 'cpu', 'scientific_conclusion': None,
                'training': False, 'phase3b': 'BLOCKED_UNCHANGED'}
