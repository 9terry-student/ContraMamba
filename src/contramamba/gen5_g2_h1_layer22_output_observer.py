"""Read-only, CPU-only Gen5 G2-H1 layer-22 *output* observer.

Observe exactly one frozen/eval wrapper forward. No model calls are performed by
this module. It does NOT expose the internal recurrent raw write, and it does
NOT establish causal attribution.

Install in: src/contramamba/gen5_g2_h1_layer22_output_observer.py
"""
from __future__ import annotations

from copy import deepcopy
import math
from dataclasses import dataclass
from typing import Any, Sequence
from weakref import WeakSet

import torch
from torch import nn


class ObservationContractError(RuntimeError):
    """A non-scientific observer contract was violated."""


_ACTIVE_WRAPPERS: WeakSet[nn.Module] = WeakSet()
CELLS = frozenset((
    "C0_SHAM", "C1_TITLE", "C2_NAME", "C3_ROLE", "C4_PREDICATE",
    "C5_TITLE_NAME", "C6_EXPLICIT_DENIAL",
))
REGIONS = ("valid", "claim", "evidence", "other")
FIELDS = ("mean", "mean_abs", "rms")


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ObservationContractError(reason)


def _binary_cpu_mask(value: torch.Tensor, shape: tuple[int, int],
                     name: str) -> torch.Tensor:
    _need(isinstance(value, torch.Tensor), f"{name}_NOT_TENSOR")
    _need(value.device.type == "cpu", f"{name}_CPU_ONLY")
    _need(tuple(value.shape) == shape, f"{name}_SHAPE")
    _need(value.dtype in (torch.bool, torch.uint8, torch.int16, torch.int32,
                          torch.int64), f"{name}_DTYPE")
    _need(bool(torch.all((value == 0) | (value == 1)).item()),
          f"{name}_NON_BINARY")
    return value.to(dtype=torch.bool).clone()


def _group_identity_rows(
    pair_ids: Sequence[str], cell_ids: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Validate row identities without assuming a pair fits in one batch."""
    _need(len(pair_ids) == len(cell_ids), "IDENTITY_LENGTH")
    grouped: dict[str, dict[str, int]] = {}
    for i, (pair, cell) in enumerate(zip(pair_ids, cell_ids)):
        _need(isinstance(pair, str) and bool(pair), "PAIR_ID_INVALID")
        _need(isinstance(cell, str) and cell in CELLS, "CELL_ID_INVALID")
        group = grouped.setdefault(pair, {})
        _need(cell not in group, "DUPLICATE_PAIR_CELL")
        group[cell] = i
    return grouped


def _row_indices(pair_ids: Sequence[str], cell_ids: Sequence[str]
                 ) -> dict[str, tuple[int, int]]:
    grouped = _group_identity_rows(pair_ids, cell_ids)
    for pair, group in grouped.items():
        _need("C0_SHAM" in group and "C6_EXPLICIT_DENIAL" in group,
              f"MISSING_C0_OR_C6:{pair}")
    return {
        pair: (group["C0_SHAM"], group["C6_EXPLICIT_DENIAL"])
        for pair, group in grouped.items()
    }


def _tensor_output(value: Any, shape: tuple[int, int],
                   name: str) -> torch.Tensor:
    _need(isinstance(value, torch.Tensor), f"{name}_NOT_TENSOR")
    _need(value.device.type == "cpu", f"{name}_CPU_ONLY")
    _need(value.ndim == 3 and tuple(value.shape[:2]) == shape,
          f"{name}_SHAPE")
    _need(value.shape[-1] > 0 and value.is_floating_point(),
          f"{name}_FEATURES")
    _need(bool(torch.isfinite(value).all().item()), f"{name}_NONFINITE")
    _need(not value.requires_grad, f"{name}_REQUIRES_GRAD")
    return value.detach().clone()


def masked_token_statistics(
    tensor: torch.Tensor,
    valid_mask: torch.Tensor,
    claim_mask: torch.Tensor,
    evidence_mask: torch.Tensor,
) -> list[dict[str, dict[str, float | int | None]]]:
    """Per-row token-region descriptive statistics, not decision attribution.

    Region means average *all channel values at valid selected tokens*.
    Region RMS is sqrt(mean(value**2)) across tokens AND channels.
    Claim/evidence may overlap; 'other' excludes their union.
    Empty region fields are None (rather than misleading zero).
    """
    _need(isinstance(tensor, torch.Tensor), "VALUES_NOT_TENSOR")
    _need(tensor.device.type == "cpu" and tensor.ndim == 3 and
          tensor.is_floating_point() and tensor.shape[-1] > 0, "VALUES_SHAPE")
    b, t, _ = tensor.shape
    shape = (b, t)
    valid = _binary_cpu_mask(valid_mask, shape, "VALID_MASK")
    claim = _binary_cpu_mask(claim_mask, shape, "CLAIM_MASK")
    evidence = _binary_cpu_mask(evidence_mask, shape, "EVIDENCE_MASK")
    _need(not bool((claim & ~valid).any().item()), "CLAIM_IN_PADDING")
    _need(not bool((evidence & ~valid).any().item()), "EVIDENCE_IN_PADDING")
    _need(bool(valid.any(dim=1).all().item()), "ROW_WITH_NO_VALID_TOKENS")
    _need(bool(torch.isfinite(tensor).all().item()), "VALUES_NONFINITE")
    region_masks = {
        "valid": valid, "claim": claim, "evidence": evidence,
        "other": valid & ~(claim | evidence),
    }
    result: list[dict[str, dict[str, float | int | None]]] = []
    # All descriptive arithmetic is on detached CPU float64 copies.
    values = tensor.detach().to(dtype=torch.float64)
    for row in range(b):
        record: dict[str, dict[str, float | int | None]] = {}
        for region in REGIONS:
            mask = region_masks[region][row]
            count = int(mask.sum().item())
            if count == 0:
                record[region] = {
                    "tokens": 0, "mean": None, "mean_abs": None, "rms": None,
                }
            else:
                selected = values[row, mask, :]
                record[region] = {
                    "tokens": count,
                    "mean": float(selected.mean().item()),
                    "mean_abs": float(selected.abs().mean().item()),
                    "rms": float(selected.square().mean().sqrt().item()),
                }
        result.append(record)
    return result


def pair_region_differences(
    per_row: Sequence[dict[str, Any]],
    pair_ids: Sequence[str],
    cell_ids: Sequence[str],
) -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    """C6 minus C0 summary differences; never mistake RMS for causal effect."""
    _need(len(per_row) == len(pair_ids), "SUMMARY_IDENTITY_LENGTH")
    indices = _row_indices(pair_ids, cell_ids)
    results: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}
    for pair, (c0, c6) in indices.items():
        sources: dict[str, dict[str, dict[str, float | None]]] = {}
        for source in ("native", "wrapped", "contribution"):
            regions: dict[str, dict[str, float | None]] = {}
            for region in REGIONS:
                a = per_row[c0][source][region]
                b = per_row[c6][source][region]
                values: dict[str, float | None] = {}
                for field in FIELDS:
                    values[field + "_C6_minus_C0"] = (
                        None if a[field] is None or b[field] is None
                        else float(b[field] - a[field])
                    )
                regions[region] = values
            sources[source] = regions
        results[pair] = sources
    return results


def final_logit_rs_contrasts(
    logits: torch.Tensor,
    pair_ids: Sequence[str],
    cell_ids: Sequence[str],
) -> dict[str, float]:
    """Pure calculation on supplied logits; does not invoke/evaluate a model.

    Frozen G2 class indices: REFUTE=0, NOT_ENTITLED=1, SUPPORT=2.
    """
    _need(isinstance(logits, torch.Tensor) and logits.device.type == "cpu",
          "LOGITS_CPU_ONLY")
    _need(logits.ndim == 2 and tuple(logits.shape) == (len(pair_ids), 3) and
          logits.is_floating_point(), "LOGITS_SHAPE")
    _need(bool(torch.isfinite(logits).all().item()), "LOGITS_NONFINITE")
    indices = _row_indices(pair_ids, cell_ids)
    scores = (logits[:, 0] - logits[:, 2]).detach().to(torch.float64)
    return {p: float((scores[i6] - scores[i0]).item())
            for p, (i0, i6) in indices.items()}


@dataclass(frozen=True)
class Layer22OutputRecord:
    """Detached copies made during a single observed synthetic forward."""
    native: torch.Tensor
    wrapped: torch.Tensor
    contribution: torch.Tensor  # wrapped - native, not raw recurrent write
    per_row: list[dict[str, Any]]
    pair_differences: dict[str, Any]


class Layer22OutputObserver:
    """Register native + wrapper hooks for *one* no-grad CPU forward only.

    The caller owns all model calls, masks and the active-mask context.
    Hooks only read/clone output tensors; they never modify outputs or args.
    """

    def __init__(
        self,
        wrapper: nn.Module,
        *,
        attention_mask: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
        pair_ids: Sequence[str],
        cell_ids: Sequence[str],
        row_ids: Sequence[str] | None = None,
        allow_partial_pairs: bool = False,
    ) -> None:
        _need(isinstance(wrapper, nn.Module) and
              isinstance(getattr(wrapper, "native_mixer", None), nn.Module),
              "G2_WRAPPER_WITH_NATIVE_REQUIRED")
        _need(wrapper.native_mixer is not wrapper, "NESTED_SELF_WRAPPER")
        _need(not wrapper.training and not wrapper.native_mixer.training,
              "EVAL_ONLY")
        shape = tuple(attention_mask.shape)
        _need(len(shape) == 2 and shape[0] > 0 and shape[1] > 0,
              "MASK_SHAPE")
        self.valid = _binary_cpu_mask(attention_mask, shape, "ATTENTION_MASK")
        self.claim = _binary_cpu_mask(claim_mask, shape, "CLAIM_MASK")
        self.evidence = _binary_cpu_mask(evidence_mask, shape, "EVIDENCE_MASK")
        _need(not bool((self.claim & ~self.valid).any().item()),
              "CLAIM_IN_PADDING")
        _need(not bool((self.evidence & ~self.valid).any().item()),
              "EVIDENCE_IN_PADDING")
        _need(bool(self.valid.any(dim=1).all().item()),
              "ROW_WITH_NO_VALID_TOKENS")
        self.pair_ids = tuple(pair_ids)
        self.cell_ids = tuple(cell_ids)
        _need(len(self.pair_ids) == shape[0], "PAIR_ROW_COUNT")
        _need(type(allow_partial_pairs) is bool, "PAIR_MODE_MUST_BE_BOOL")
        self.allow_partial_pairs = allow_partial_pairs
        if allow_partial_pairs:
            _group_identity_rows(self.pair_ids, self.cell_ids)
        else:
            _row_indices(self.pair_ids, self.cell_ids)
        if row_ids is not None:
            _need(len(row_ids) == shape[0] and
                  len(set(row_ids)) == shape[0], "ROW_IDS_NOT_UNIQUE")
        self.wrapper = wrapper
        self._hooks: list[Any] = []
        self._entered = False
        self._finished = False
        self._failed = False
        self._native_count = 0
        self._wrapped_count = 0
        self._native: torch.Tensor | None = None
        self._wrapped: torch.Tensor | None = None
        self._native_mask: Any = None

    @staticmethod
    def _forward_attention_mask(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        # G2 forward(hidden_states, cache_params, cache_position, attention_mask).
        # kwargs precedence is exactly Python call semantics.
        return kwargs["attention_mask"] if "attention_mask" in kwargs else (
            args[3] if len(args) >= 4 else None
        )

    def _on_native(self, module: nn.Module, args: tuple[Any, ...],
                   kwargs: dict[str, Any], output: Any) -> None:
        self._native_count += 1
        _need(self._native_count == 1 and self._wrapped_count == 0,
              "NATIVE_CALLED_MULTIPLE_OR_OUT_OF_ORDER")
        self._native_mask = self._forward_attention_mask(args, kwargs)
        self._native = _tensor_output(
            output, tuple(self.valid.shape), "NATIVE_OUTPUT"
        )

    def _on_wrapper(self, module: nn.Module, args: tuple[Any, ...],
                    kwargs: dict[str, Any], output: Any) -> None:
        self._wrapped_count += 1
        _need(self._wrapped_count == 1 and self._native_count == 1,
              "WRAPPER_CALLED_MULTIPLE_OR_NATIVE_MISSING")
        explicit = self._forward_attention_mask(args, kwargs)
        _need(self._native_mask is explicit,
              "NATIVE_ATTENTION_MASK_FORWARDING_CHANGED")
        effective = (explicit if explicit is not None else
                     getattr(module, "_phase2_active_mask", None))
        _need(effective is not None, "EFFECTIVE_MASK_MISSING")
        checked = _binary_cpu_mask(
            effective, tuple(self.valid.shape), "EFFECTIVE_MASK")
        _need(torch.equal(checked, self.valid),
              "OBSERVER_EFFECTIVE_MASK_MISMATCH")
        self._wrapped = _tensor_output(
            output, tuple(self.valid.shape), "WRAPPED_OUTPUT"
        )
        _need(self._native is not None and
              tuple(self._native.shape) == tuple(self._wrapped.shape) and
              self._native.dtype == self._wrapped.dtype,
              "NATIVE_WRAPPED_OUTPUT_MISMATCH")

    def __enter__(self) -> "Layer22OutputObserver":
        _need(not self._entered and not self._finished,
              "OBSERVER_NOT_REUSABLE")
        _need(self.wrapper not in _ACTIVE_WRAPPERS,
              "DUPLICATE_OBSERVER_REGISTRATION")
        _need(not self.wrapper.training and
              not self.wrapper.native_mixer.training, "EVAL_ONLY")
        _ACTIVE_WRAPPERS.add(self.wrapper)
        try:
            self._hooks.append(
                self.wrapper.native_mixer.register_forward_hook(
                    self._on_native, with_kwargs=True)
            )
            self._hooks.append(
                self.wrapper.register_forward_hook(
                    self._on_wrapper, with_kwargs=True)
            )
            self._entered = True
        except BaseException:
            for handle in reversed(self._hooks):
                handle.remove()
            self._hooks.clear()
            _ACTIVE_WRAPPERS.discard(self.wrapper)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for handle in reversed(self._hooks):
            handle.remove()
        self._hooks.clear()
        _ACTIVE_WRAPPERS.discard(self.wrapper)
        self._finished = True
        self._failed = exc_type is not None
        if exc_type is None:
            _need(self._native_count == 1 and self._wrapped_count == 1,
                  "EXPECTED_EXACTLY_ONE_FORWARD")
        return False

    def record(self) -> Layer22OutputRecord:
        _need(self._finished and not self._failed and
              self._native_count == 1 and self._wrapped_count == 1,
              "OBSERVATION_NOT_COMPLETE")
        _need(self._native is not None and self._wrapped is not None,
              "OBSERVATION_OUTPUTS_MISSING")
        # The subtraction is in the same output dtype. Rounding means it may
        # not be bitwise identical to the separately computed correction term.
        contribution = self._wrapped - self._native
        sources = (self._native, self._wrapped, contribution)
        stats = [masked_token_statistics(
            tensor, self.valid, self.claim, self.evidence
        ) for tensor in sources]
        per_row = [
            dict(native=stats[0][i], wrapped=stats[1][i],
                 contribution=stats[2][i])
            for i in range(self.valid.shape[0])
        ]
        return Layer22OutputRecord(
            native=self._native.clone(), wrapped=self._wrapped.clone(),
            contribution=contribution.clone(), per_row=per_row,
            pair_differences=(
                {} if self.allow_partial_pairs else pair_region_differences(
                    per_row, self.pair_ids, self.cell_ids,
                )
            ),
        )


class G2H1PairSummaryAccumulator:
    """CPU-only streaming join of detached row summaries across batches/shards.

    No tensor/model forward, checkpoint access, or GPU execution occurs here.
    Identity is (row_id, pair_id, contrast_cell_id), never row adjacency.
    Finalization is fail-closed on missing or duplicate contrast cells.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._pairs: list[str] = []
        self._cells: list[str] = []
        self._seen_rows: set[str] = set()
        self._seen_pair_cells: set[tuple[str, str]] = set()
        self._finalized = False

    def add_batch(
        self,
        per_row: Sequence[dict[str, Any]],
        *,
        pair_ids: Sequence[str],
        cell_ids: Sequence[str],
        row_ids: Sequence[str],
    ) -> None:
        _need(not self._finalized, "ACCUMULATOR_ALREADY_FINALIZED")
        n = len(per_row)
        _need(n > 0 and len(pair_ids) == len(cell_ids) == len(row_ids) == n,
              "BATCH_IDENTITY_LENGTH")
        _group_identity_rows(pair_ids, cell_ids)
        pending_rows: set[str] = set()
        pending_cells: set[tuple[str, str]] = set()
        for rid, pid, cell in zip(row_ids, pair_ids, cell_ids):
            _need(isinstance(rid, str) and bool(rid), "ROW_ID_INVALID")
            _need(rid not in self._seen_rows and rid not in pending_rows,
                  "DUPLICATE_ROW_ID")
            key = (pid, cell)
            _need(key not in self._seen_pair_cells and key not in pending_cells,
                  "DUPLICATE_PAIR_CELL")
            pending_rows.add(rid)
            pending_cells.add(key)
        # Snapshot only small per-row descriptive summaries; never retain
        # full [batch, sequence, hidden] native/wrapped/correction tensors.
        snapshot = deepcopy(list(per_row))
        for item in snapshot:
            _need(isinstance(item, dict) and
                  all(src in item for src in ("native", "wrapped", "contribution")),
                  "ROW_SUMMARY_SHAPE")
            for source in ("native", "wrapped", "contribution"):
                _need(isinstance(item[source], dict), "SOURCE_SUMMARY_SHAPE")
                for region in REGIONS:
                    fields = item[source].get(region)
                    _need(isinstance(fields, dict), "REGION_SUMMARY_SHAPE")
                    _need(type(fields.get("tokens")) is int and
                          fields["tokens"] >= 0, "REGION_TOKEN_COUNT")
                    for field in FIELDS:
                        value = fields.get(field)
                        _need(value is None or (isinstance(value, (int, float)) and
                              math.isfinite(float(value))), "REGION_NONFINITE")
                        _need((value is None) == (fields["tokens"] == 0),
                              "EMPTY_REGION_VALUE_INCONSISTENT")
        self._rows.extend(snapshot)
        self._pairs.extend(pair_ids)
        self._cells.extend(cell_ids)
        self._seen_rows.update(pending_rows)
        self._seen_pair_cells.update(pending_cells)

    def finalize(
        self,
        *,
        expected_pairs: int,
        require_all_seven_cells: bool = True,
    ) -> dict[str, dict[str, Any]]:
        _need(not self._finalized, "ACCUMULATOR_ALREADY_FINALIZED")
        _need(type(expected_pairs) is int and expected_pairs > 0,
              "EXPECTED_PAIRS_INVALID")
        _need(type(require_all_seven_cells) is bool, "FULL_CELLS_FLAG_INVALID")
        grouped = _group_identity_rows(self._pairs, self._cells)
        _need(len(grouped) == expected_pairs, "PAIR_COUNT_MISMATCH")
        for pair, group in grouped.items():
            required = CELLS if require_all_seven_cells else (
                "C0_SHAM", "C6_EXPLICIT_DENIAL")
            _need(set(required).issubset(group), f"INCOMPLETE_PAIR:{pair}")
            if require_all_seven_cells:
                _need(len(group) == len(CELLS), f"PAIR_CELL_COUNT:{pair}")
        result = pair_region_differences(self._rows, self._pairs, self._cells)
        self._finalized = True
        return result
