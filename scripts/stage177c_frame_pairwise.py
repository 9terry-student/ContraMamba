"""Stage177-C clean-train frame-logit pairwise objective.

Targets are read only from the row-level ``frame_compatible_label`` field.
The builder preserves the supplied row ordering, and the loss gives every
validated pair equal weight regardless of its number of comparisons.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Sequence

import torch
import torch.nn.functional as F


FRAME_TARGET_FIELD = "frame_compatible_label"
OBJECTIVE_NAME = "pair_normalized_softplus_negative_gap"


def build_stage177c_train_pair_index(
    train_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build and validate a stable clean-train pair index.

    A pair is malformed when it has no compatible or no incompatible row.
    Malformed pairs are never skipped: their complete diagnostics are included
    in the exception message and construction fails before training starts.
    """

    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row_index, row in enumerate(train_rows):
        if not isinstance(row, dict):
            raise TypeError(f"[stage177c] train row {row_index} is not a mapping")
        pair_id = row.get("pair_id")
        if pair_id is None or str(pair_id) == "":
            raise ValueError(f"[stage177c] train row {row_index} lacks pair_id")
        if FRAME_TARGET_FIELD not in row:
            raise ValueError(
                f"[stage177c] train row {row_index} lacks {FRAME_TARGET_FIELD}"
            )
        raw_label = row[FRAME_TARGET_FIELD]
        if isinstance(raw_label, bool):
            label = int(raw_label)
        elif isinstance(raw_label, int) and raw_label in (0, 1):
            label = raw_label
        else:
            raise ValueError(
                f"[stage177c] train row {row_index} has invalid "
                f"{FRAME_TARGET_FIELD}={raw_label!r}; expected integer 0 or 1"
            )
        key = str(pair_id)
        group = groups.setdefault(
            key,
            {
                "pair_id": key,
                "compatible_indices": [],
                "incompatible_indices": [],
            },
        )
        target = "compatible_indices" if label == 1 else "incompatible_indices"
        group[target].append(row_index)

    pairs: list[dict[str, Any]] = []
    malformed_pairs: list[dict[str, Any]] = []
    for group in groups.values():
        compatible_count = len(group["compatible_indices"])
        incompatible_count = len(group["incompatible_indices"])
        record = {
            **group,
            "compatible_count": compatible_count,
            "incompatible_count": incompatible_count,
            "comparison_count": compatible_count * incompatible_count,
        }
        if compatible_count == 0 or incompatible_count == 0:
            malformed_pairs.append(record)
        else:
            pairs.append(record)

    if malformed_pairs:
        details = ", ".join(
            f"{item['pair_id']}(compatible={item['compatible_count']},"
            f" incompatible={item['incompatible_count']})"
            for item in malformed_pairs
        )
        raise ValueError(
            f"[stage177c] malformed clean-train pairs are forbidden: {details}"
        )

    compatible_row_count = sum(item["compatible_count"] for item in pairs)
    incompatible_row_count = sum(item["incompatible_count"] for item in pairs)
    raw_comparison_count = sum(item["comparison_count"] for item in pairs)
    return {
        "target_field": FRAME_TARGET_FIELD,
        "row_count": len(train_rows),
        "total_pair_count": len(groups),
        "eligible_pair_count": len(pairs),
        "malformed_pair_count": 0,
        "malformed_pairs": [],
        "compatible_row_count": compatible_row_count,
        "incompatible_row_count": incompatible_row_count,
        "raw_comparison_count": raw_comparison_count,
        "pairs": pairs,
    }


def _frame_logit_vector(frame_logits: torch.Tensor, row_count: int) -> torch.Tensor:
    if not torch.is_tensor(frame_logits):
        raise TypeError("[stage177c] frame_logits must be a torch.Tensor")
    if frame_logits.ndim == 1:
        vector = frame_logits
    elif frame_logits.ndim == 2 and frame_logits.shape[1] == 1:
        vector = frame_logits[:, 0]
    else:
        raise ValueError(
            "[stage177c] frame_logits must have shape [rows] or [rows, 1], "
            f"got {tuple(frame_logits.shape)}"
        )
    if vector.shape[0] != row_count:
        raise ValueError(
            f"[stage177c] frame-logit row count {vector.shape[0]} != pair-index "
            f"row count {row_count}"
        )
    if not vector.is_floating_point():
        raise TypeError("[stage177c] frame_logits must have a floating dtype")
    return vector


def compute_stage177c_frame_pairwise_loss(
    frame_logits: torch.Tensor,
    pair_index: dict[str, Any],
    mode: str = "pair_softplus",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the differentiable unweighted objective and detached diagnostics."""

    if mode != "pair_softplus":
        raise ValueError(f"[stage177c] unsupported loss mode: {mode!r}")
    pairs = pair_index.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("[stage177c] validated pair index has no eligible pairs")
    if int(pair_index.get("malformed_pair_count", -1)) != 0:
        raise ValueError("[stage177c] malformed pairs cannot enter loss computation")

    vector = _frame_logit_vector(frame_logits, int(pair_index["row_count"]))
    pair_losses: list[torch.Tensor] = []
    pair_accuracies: list[torch.Tensor] = []
    all_gaps: list[torch.Tensor] = []
    for pair in pairs:
        compatible_indices = pair.get("compatible_indices") or []
        incompatible_indices = pair.get("incompatible_indices") or []
        if not compatible_indices or not incompatible_indices:
            raise ValueError(
                f"[stage177c] empty side in pair {pair.get('pair_id')!r}"
            )
        compatible_index = torch.as_tensor(
            compatible_indices, dtype=torch.long, device=vector.device
        )
        incompatible_index = torch.as_tensor(
            incompatible_indices, dtype=torch.long, device=vector.device
        )
        gaps = (
            vector.index_select(0, compatible_index)[:, None]
            - vector.index_select(0, incompatible_index)[None, :]
        ).reshape(-1)
        comparison_losses = F.softplus(-gaps)
        pair_losses.append(comparison_losses.mean())
        pair_accuracies.append((gaps > 0).to(vector.dtype).mean())
        all_gaps.append(gaps)

    loss = torch.stack(pair_losses).mean()
    if loss.ndim != 0 or not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("[stage177c] pairwise loss is not a finite scalar")

    detached_gaps = torch.cat([item.detach().reshape(-1) for item in all_gaps])
    detached_pair_losses = torch.stack([item.detach() for item in pair_losses])
    sorted_gaps = torch.sort(detached_gaps).values
    gap_count = int(sorted_gaps.numel())
    middle = gap_count // 2
    median_gap = (
        sorted_gaps[middle]
        if gap_count % 2 == 1
        else (sorted_gaps[middle - 1] + sorted_gaps[middle]) / 2
    )
    positive_count = int((detached_gaps > 0).sum().item())
    zero_count = int((detached_gaps == 0).sum().item())
    negative_count = int((detached_gaps < 0).sum().item())
    raw_count = int(detached_gaps.numel())
    diagnostics = {
        "eligible_pair_count": len(pairs),
        "malformed_pair_count": 0,
        "compatible_row_count": int(pair_index["compatible_row_count"]),
        "incompatible_row_count": int(pair_index["incompatible_row_count"]),
        "raw_comparison_count": raw_count,
        "pair_loss_mean": float(detached_pair_losses.mean().item()),
        "comparison_loss_mean": float(F.softplus(-detached_gaps).mean().item()),
        "mean_gap": float(detached_gaps.mean().item()),
        "median_gap": float(median_gap.item()),
        "positive_gap_count": positive_count,
        "zero_gap_count": zero_count,
        "negative_gap_count": negative_count,
        "comparison_ranking_accuracy": positive_count / raw_count,
        "pair_normalized_ranking_accuracy": float(
            torch.stack([item.detach() for item in pair_accuracies]).mean().item()
        ),
        "finite": bool(math.isfinite(float(loss.detach().item()))),
        "gradient_enabled": bool(loss.requires_grad and torch.is_grad_enabled()),
    }
    return loss, diagnostics
