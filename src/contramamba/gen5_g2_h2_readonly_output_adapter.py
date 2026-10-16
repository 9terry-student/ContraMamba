"""G2-H2 three-condition frozen-wrapper output capture, CPU-only contract.

No pretrained model, tokenizer, GPU, labels, loss, or forward call here. The
caller owns one forward under frozen/eval/no-grad and G2 active_mask. Captured
"contribution" is wrapped mixer output minus native mixer output; NOT raw
recurrent state write. This adapter does not alter H1's seven-cell observer.

CPU-only by design: do not widen device support without a separate equivalence
and execution gate. Partial pair batches are intentional; pair joining belongs
to the frozen G2-H2 CPU metrics module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from weakref import WeakSet

import torch
from torch import nn

CONDITIONS = frozenset(('NEG_DENIES', 'POS_CONFIRMS', 'POS_ASSERTS'))
VERB_IDS = {'NEG_DENIES': 31647, 'POS_CONFIRMS': 23849, 'POS_ASSERTS': 17086}
ARMS = frozenset(('ZERO', 'FULL'))
_ACTIVE: WeakSet[nn.Module] = WeakSet()


class H2CaptureContractError(RuntimeError):
    pass


def need(condition: bool, code: str) -> None:
    if not condition:
        raise H2CaptureContractError('G2_H2_ADAPTER_BLOCKED:' + code)


def _cpu_mask(value: Any, shape: tuple[int, int], name: str) -> torch.Tensor:
    need(isinstance(value, torch.Tensor), name + '_NOT_TENSOR')
    need(value.device.type == 'cpu' and value.ndim == 2 and
         tuple(value.shape) == shape, name + '_CPU_SHAPE')
    need(value.dtype in (torch.bool, torch.uint8, torch.int16,
                          torch.int32, torch.int64), name + '_DTYPE')
    copy = value.detach().clone()
    need(bool(((copy == 0) | (copy == 1)).all().item()), name + '_NONBINARY')
    return copy.bool()


def _copy_output(value: Any, shape: tuple[int, int], name: str) -> torch.Tensor:
    need(isinstance(value, torch.Tensor), name + '_NOT_TENSOR')
    need(value.device.type == 'cpu' and value.ndim == 3 and
         tuple(value.shape[:2]) == shape and value.shape[-1] > 0 and
         value.is_floating_point(), name + '_CPU_SHAPE')
    need(not value.requires_grad, name + '_GRAD_ENABLED')
    snapshot = value.detach().clone()
    need(bool(torch.isfinite(snapshot).all().item()), name + '_NONFINITE')
    return snapshot


@dataclass(frozen=True)
class H2OutputCapture:
    """One forward only. CPU snapshots are not a raw SSM state observation."""
    pair_ids: tuple[str, ...]
    conditions: tuple[str, ...]
    row_ids: tuple[str, ...]
    verb_indices: tuple[int, ...]
    valid_lengths: tuple[int, ...]
    evidence_starts: tuple[int, ...]
    native: torch.Tensor
    wrapped: torch.Tensor
    contribution: torch.Tensor

    def metric_row(self, index: int) -> dict[str, Any]:
        """Return ONE VALID-ONLY row for frozen H2 output_readout.

        Deliberately omit padding. Passing 128 padded steps as EVIDENCE would
        incorrectly include padding in the prespecified evidence-region RMS.
        """
        need(type(index) is int and 0 <= index < len(self.row_ids),
             'METRIC_ROW_INDEX')
        valid = self.valid_lengths[index]
        return {
            'row_id': self.row_ids[index], 'pair_id': self.pair_ids[index],
            'condition': self.conditions[index], 'length': valid,
            'evidence_start': self.evidence_starts[index],
            'verb_index': self.verb_indices[index],
            'native': self.native[index, :valid].tolist(),
            'wrapped': self.wrapped[index, :valid].tolist(),
        }


class H2Layer22ReadOnlyAdapter:
    """Capture two module outputs without changing wrapper, args or result.

    The inputs, padding and token-identity checks use the already frozen H2
    token IDs. A batch may contain an incomplete pair; inter-batch joins are
    deliberately handled outside the observer. Does not invoke a model.
    """

    def __init__(
        self,
        wrapper: nn.Module,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
        pair_ids: Sequence[str],
        conditions: Sequence[str],
        row_ids: Sequence[str],
        verb_indices: Sequence[int],
        arm: str,
    ) -> None:
        need(isinstance(wrapper, nn.Module) and
             isinstance(getattr(wrapper, 'native_mixer', None), nn.Module),
             'G2_WRAPPER_REQUIRED')
        need(wrapper.native_mixer is not wrapper, 'RECURSIVE_WRAPPER')
        need(not wrapper.training and not wrapper.native_mixer.training,
             'EVAL_REQUIRED')
        need(all(not p.requires_grad for p in wrapper.parameters()),
             'FROZEN_PARAMETERS_REQUIRED')
        need(arm in ARMS and getattr(wrapper, 'arm', None) == arm,
             'FROZEN_ARM_MISMATCH')
        need(isinstance(input_ids, torch.Tensor) and
             input_ids.device.type == 'cpu' and input_ids.ndim == 2 and
             input_ids.dtype == torch.long, 'INPUT_IDS_CPU_LONG')
        shape = tuple(input_ids.shape)
        need(shape[0] > 0 and shape[1] == 128, 'FROZEN_INPUT_SHAPE')
        valid = _cpu_mask(attention_mask, shape, 'ATTENTION')
        claim = _cpu_mask(claim_mask, shape, 'CLAIM')
        evidence = _cpu_mask(evidence_mask, shape, 'EVIDENCE')
        b, _ = shape
        ids = input_ids.detach().clone()
        self.pair_ids = tuple(pair_ids)
        self.conditions = tuple(conditions)
        self.row_ids = tuple(row_ids)
        self.verb_indices = tuple(verb_indices)
        need(all(len(x) == b for x in (self.pair_ids, self.conditions,
                                       self.row_ids, self.verb_indices)),
             'ROW_METADATA_SHAPE')
        need(all(type(p) is str and bool(p) for p in self.pair_ids),
             'PAIR_ID_INVALID')
        need(all(c in CONDITIONS for c in self.conditions), 'CONDITION_INVALID')
        need(len(set(self.row_ids)) == b, 'DUPLICATE_ROW_ID')
        need(len(set(zip(self.pair_ids, self.conditions))) == b,
             'DUPLICATE_PAIR_CONDITION')
        # H1 C6 IDs are never reused for H2.
        need(all(self.row_ids[i] ==
                 self.pair_ids[i] + '__g2_h2__' + self.conditions[i].lower()
                 for i in range(b)), 'H2_ROW_ID_PROVENANCE')
        valid_lengths: list[int] = []
        evidence_starts: list[int] = []
        for i in range(b):
            v = int(valid[i].sum().item())
            c = int(claim[i].sum().item())
            e = int(evidence[i].sum().item())
            need(1 <= c <= 63 and 1 <= e <= 64 and v == c + 1 + e,
                 'FROZEN_SERIALIZATION_BUDGET')
            need(bool(valid[i, :v].all()) and
                 not bool(valid[i, v:].any()) and
                 bool(claim[i, :c].all()) and
                 not bool(claim[i, c:].any()) and
                 not bool(evidence[i, :c+1].any()) and
                 bool(evidence[i, c+1:v].all()) and
                 not bool(evidence[i, v:].any()),
                 'CLAIM_EVIDENCE_MASK_ALIGNMENT')
            need(int(ids[i, c]) == 0 and
                 not bool(ids[i, v:].any().item()), 'EOS_PAD_TOKEN_ID')
            position = self.verb_indices[i]
            need(type(position) is int and c + 1 <= position < v - 8,
                 'VERB_POSITION_POST8')
            need(int(ids[i, position]) == VERB_IDS[self.conditions[i]],
                 'VERB_FROZEN_TOKEN_ID')
            valid_lengths.append(v)
            evidence_starts.append(c + 1)
        # Check all complete or partial within-batch comparisons by identity.
        seen: dict[str, tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for i, pair in enumerate(self.pair_ids):
            cur = (ids[i], self.verb_indices[i], valid[i], claim[i], evidence[i])
            if pair in seen:
                prev, prev_pos, va, cl, ev = seen[pair]
                need(cur[1] == prev_pos and torch.equal(cur[2], va) and
                     torch.equal(cur[3], cl) and torch.equal(cur[4], ev),
                     'PAIR_MASK_POSITION_MISMATCH')
                differences = torch.nonzero(prev != ids[i], as_tuple=True)[0].tolist()
                need(differences == [prev_pos], 'PAIR_NOT_SINGLE_TOKEN_SWAP')
            else:
                seen[pair] = cur
        self.valid_lengths = tuple(valid_lengths)
        self.evidence_starts = tuple(evidence_starts)
        self.wrapper = wrapper
        self.valid = valid
        self.arm = arm
        self._shape = shape
        self._native: torch.Tensor | None = None
        self._wrapped: torch.Tensor | None = None
        self._native_mask: Any = None
        self._native_device: torch.device | None = None
        self._native_count = 0
        self._wrapped_count = 0
        self._entered = False
        self._finished = False
        self._failed = False
        self._handles: list[Any] = []

    @staticmethod
    def _forward_mask(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return kwargs['attention_mask'] if 'attention_mask' in kwargs else (
            args[3] if len(args) >= 4 else None)

    def _native_hook(self, _module: nn.Module, args: tuple[Any, ...],
                     kwargs: dict[str, Any], output: Any) -> None:
        self._native_count += 1
        need(self._native_count == 1 and self._wrapped_count == 0,
             'NATIVE_FORWARD_ORDER')
        self._native_mask = self._forward_mask(args, kwargs)
        self._native_device = output.device if isinstance(output, torch.Tensor) else None
        self._native = _copy_output(output, self._shape, 'NATIVE')

    def _wrapper_hook(self, _module: nn.Module, args: tuple[Any, ...],
                      kwargs: dict[str, Any], output: Any) -> None:
        self._wrapped_count += 1
        need(self._native_count == 1 and self._wrapped_count == 1,
             'WRAPPER_FORWARD_ORDER')
        explicit = self._forward_mask(args, kwargs)
        need(self._native_mask is explicit, 'NATIVE_MASK_FORWARDING_CHANGED')
        need(isinstance(output, torch.Tensor) and output.device == self._native_device,
             'NATIVE_WRAPPED_DEVICE_MISMATCH')
        effective = (explicit if explicit is not None else
                     getattr(self.wrapper, '_phase2_active_mask', None))
        need(effective is not None, 'EFFECTIVE_MASK_MISSING')
        need(torch.equal(_cpu_mask(effective, self._shape, 'EFFECTIVE'), self.valid),
             'EFFECTIVE_MASK_IDENTITY')
        self._wrapped = _copy_output(output, self._shape, 'WRAPPED')
        need(self._native is not None and self._wrapped.shape == self._native.shape and
             self._wrapped.dtype == self._native.dtype,
             'NATIVE_WRAPPED_OUTPUT_MISMATCH')

    def __enter__(self) -> 'H2Layer22ReadOnlyAdapter':
        need(not self._entered and not self._finished, 'OBSERVER_NOT_REUSABLE')
        need(self.wrapper not in _ACTIVE, 'DUPLICATE_OBSERVER')
        need(not self.wrapper.training and not self.wrapper.native_mixer.training,
             'EVAL_REQUIRED')
        _ACTIVE.add(self.wrapper)
        try:
            self._handles.append(self.wrapper.native_mixer.register_forward_hook(
                self._native_hook, with_kwargs=True))
            self._handles.append(self.wrapper.register_forward_hook(
                self._wrapper_hook, with_kwargs=True))
            self._entered = True
        except BaseException:
            for h in reversed(self._handles):
                h.remove()
            self._handles.clear()
            _ACTIVE.discard(self.wrapper)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        _ACTIVE.discard(self.wrapper)
        self._finished = True
        self._failed = exc_type is not None
        if exc_type is None:
            need(self._native_count == 1 and self._wrapped_count == 1,
                 'EXACTLY_ONE_FORWARD_REQUIRED')
        return False

    def record(self) -> H2OutputCapture:
        need(self._finished and not self._failed and
             self._native_count == 1 and self._wrapped_count == 1 and
             self._native is not None and self._wrapped is not None,
             'CAPTURE_NOT_COMPLETE')
        native = self._native.clone()
        wrapped = self._wrapped.clone()
        contribution = wrapped - native
        need(bool(torch.isfinite(contribution).all().item()),
             'CONTRIBUTION_NONFINITE')
        if self.arm == 'ZERO':
            need(bool(torch.equal(contribution, torch.zeros_like(contribution))),
                 'ZERO_CONTRIBUTION_NONZERO')
        return H2OutputCapture(
            pair_ids=self.pair_ids, conditions=self.conditions,
            row_ids=self.row_ids, verb_indices=self.verb_indices,
            valid_lengths=self.valid_lengths,
            evidence_starts=self.evidence_starts,
            native=native, wrapped=wrapped, contribution=contribution,
        )
