"""Train ContraMamba-v6B-minimal on controlled intervention data.

Minimal v6B wrapper: reuses v5 training infrastructure, adds temporal/predicate
comparator alphas with learnable scaling. No composer, no product_final_loss.
All CE/pairwise/intervention losses consume final calibrated logits.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contramamba.comparator_flags import (  # noqa: E402
    predicate_mismatch_flags_from_intervention_type,
    predicate_mismatch_flags_from_probe,
    temporal_mismatch_flags_from_probe,
    temporal_mismatch_flags_none,
)
from contramamba.modeling_v6b_minimal import ContraMambaV6BMinimal  # noqa: E402
from scripts import train_controlled_v5 as v5  # noqa: E402


def build_model(
    vocab_size: int, max_length: int, hidden_size: int = 48,
    use_boundary_head: bool = False,
    use_frame_violation_head: bool = False,
    use_predicate_isolation_head: bool = False,
    use_preservation_entitlement_head: bool = False,
    use_temporal_diagnostic_head: bool = False,
    use_temporal_residual_adapter: bool = False,
    temporal_adapter_detach_input: bool = True,
    use_temporal_channel: bool = False,
    temporal_channel_detach_input: bool = True,
    use_temporal_channel_loss: bool = False,
    temporal_channel_loss_weight: float = 0.0,
    temporal_channel_loss_pos_weight: float = 1.0,
    use_temporal_channel_gated_penalty: bool = False,
    temporal_channel_gated_penalty_scale: float = 0.0,
) -> ContraMambaV6BMinimal:
    backbone = v5.ControlledDummyBackbone(vocab_size, hidden_size, max_length)
    return ContraMambaV6BMinimal(
        backbone=backbone,
        frame_size=32,
        predicate_size=32,
        sufficiency_size=32,
        energy_size=24,
        dropout=0.0,
        decision_mode="explicit_product",
        use_temporal_comparator=True,
        use_predicate_comparator=True,
        alpha_temporal_init=1.25,
        alpha_predicate_init=1.25,
        use_boundary_head=use_boundary_head,
        use_frame_violation_head=use_frame_violation_head,
        use_predicate_isolation_head=use_predicate_isolation_head,
        use_preservation_entitlement_head=use_preservation_entitlement_head,
        use_temporal_diagnostic_head=use_temporal_diagnostic_head,
        use_temporal_residual_adapter=use_temporal_residual_adapter,
        temporal_adapter_detach_input=temporal_adapter_detach_input,
        use_temporal_channel=use_temporal_channel,
        temporal_channel_detach_input=temporal_channel_detach_input,
        use_temporal_channel_loss=use_temporal_channel_loss,
        temporal_channel_loss_weight=temporal_channel_loss_weight,
        temporal_channel_loss_pos_weight=temporal_channel_loss_pos_weight,
        use_temporal_channel_gated_penalty=use_temporal_channel_gated_penalty,
        temporal_channel_gated_penalty_scale=temporal_channel_gated_penalty_scale,
    )


def build_mamba_model(
    model_name: str,
    freeze_encoder: bool,
    freeze_a_log: bool,
    use_boundary_head: bool = False,
    use_frame_violation_head: bool = False,
    use_predicate_isolation_head: bool = False,
    use_preservation_entitlement_head: bool = False,
    use_temporal_diagnostic_head: bool = False,
    use_temporal_residual_adapter: bool = False,
    temporal_adapter_detach_input: bool = True,
    use_temporal_channel: bool = False,
    temporal_channel_detach_input: bool = True,
    use_temporal_channel_loss: bool = False,
    temporal_channel_loss_weight: float = 0.0,
    temporal_channel_loss_pos_weight: float = 1.0,
    use_temporal_channel_gated_penalty: bool = False,
    temporal_channel_gated_penalty_scale: float = 0.0,
) -> ContraMambaV6BMinimal:
    model = ContraMambaV6BMinimal(
        model_name=model_name,
        frame_size=128,
        predicate_size=128,
        sufficiency_size=128,
        energy_size=64,
        dropout=0.1,
        freeze_a_log=freeze_a_log,
        decision_mode="explicit_product",
        use_temporal_comparator=True,
        use_predicate_comparator=True,
        alpha_temporal_init=1.25,
        alpha_predicate_init=1.25,
        use_boundary_head=use_boundary_head,
        use_frame_violation_head=use_frame_violation_head,
        use_predicate_isolation_head=use_predicate_isolation_head,
        use_preservation_entitlement_head=use_preservation_entitlement_head,
        use_temporal_diagnostic_head=use_temporal_diagnostic_head,
        use_temporal_residual_adapter=use_temporal_residual_adapter,
        temporal_adapter_detach_input=temporal_adapter_detach_input,
        use_temporal_channel=use_temporal_channel,
        temporal_channel_detach_input=temporal_channel_detach_input,
        use_temporal_channel_loss=use_temporal_channel_loss,
        temporal_channel_loss_weight=temporal_channel_loss_weight,
        temporal_channel_loss_pos_weight=temporal_channel_loss_pos_weight,
        use_temporal_channel_gated_penalty=use_temporal_channel_gated_penalty,
        temporal_channel_gated_penalty_scale=temporal_channel_gated_penalty_scale,
    )
    for parameter in model.mamba.parameters():
        parameter.requires_grad = not freeze_encoder
    if freeze_a_log:
        for name, parameter in model.mamba.named_parameters():
            if "A_log" in name:
                parameter.requires_grad = False
    return model


def build_v7_model(
    vocab_size: int,
    max_length: int,
    hidden_size: int = 48,
    v7_disable_frame_channel: bool = False,
    v7_disable_predicate_channel: bool = False,
    v7_disable_sufficiency_channel: bool = False,
    v7_disable_temporal_channel: bool = False,
    v7_flat_arbiter: bool = False,
    v7_no_entitlement_polarity_conditioning: bool = False,
    v7_no_aux_losses: bool = False,
    v7_initial_ne_bias: float = -0.5,
) -> "ContraMambaV7Hierarchical":
    """Build a ContraMambaV7Hierarchical with dummy backbone for plumbing validation."""
    from contramamba.modeling_v7_hierarchical import ContraMambaV7Hierarchical
    backbone = v5.ControlledDummyBackbone(vocab_size, hidden_size, max_length)
    return ContraMambaV7Hierarchical(
        backbone=backbone,
        frame_size=32,
        predicate_size=32,
        sufficiency_size=32,
        polarity_size=24,
        dropout=0.0,
        v7_disable_frame_channel=v7_disable_frame_channel,
        v7_disable_predicate_channel=v7_disable_predicate_channel,
        v7_disable_sufficiency_channel=v7_disable_sufficiency_channel,
        v7_disable_temporal_channel=v7_disable_temporal_channel,
        v7_flat_arbiter=v7_flat_arbiter,
        v7_no_entitlement_polarity_conditioning=v7_no_entitlement_polarity_conditioning,
        v7_no_aux_losses=v7_no_aux_losses,
        v7_initial_ne_bias=v7_initial_ne_bias,
    )


def build_v7_mamba_model(
    model_name: str,
    freeze_encoder: bool,
    freeze_a_log: bool,
    v7_disable_frame_channel: bool = False,
    v7_disable_predicate_channel: bool = False,
    v7_disable_sufficiency_channel: bool = False,
    v7_disable_temporal_channel: bool = False,
    v7_flat_arbiter: bool = False,
    v7_no_entitlement_polarity_conditioning: bool = False,
    v7_no_aux_losses: bool = False,
    v7_initial_ne_bias: float = -0.5,
) -> "ContraMambaV7Hierarchical":
    """Build a ContraMambaV7Hierarchical with real Mamba backbone."""
    from contramamba.modeling_v7_hierarchical import ContraMambaV7Hierarchical
    model = ContraMambaV7Hierarchical(
        model_name=model_name,
        frame_size=128,
        predicate_size=128,
        sufficiency_size=128,
        polarity_size=64,
        dropout=0.1,
        freeze_a_log=freeze_a_log,
        v7_disable_frame_channel=v7_disable_frame_channel,
        v7_disable_predicate_channel=v7_disable_predicate_channel,
        v7_disable_sufficiency_channel=v7_disable_sufficiency_channel,
        v7_disable_temporal_channel=v7_disable_temporal_channel,
        v7_flat_arbiter=v7_flat_arbiter,
        v7_no_entitlement_polarity_conditioning=v7_no_entitlement_polarity_conditioning,
        v7_no_aux_losses=v7_no_aux_losses,
        v7_initial_ne_bias=v7_initial_ne_bias,
    )
    for parameter in model.mamba.parameters():
        parameter.requires_grad = not freeze_encoder
    if freeze_a_log:
        for name, parameter in model.mamba.named_parameters():
            if "A_log" in name:
                parameter.requires_grad = False
    return model


def extract_flags(
    records: list[dict],
    flag_source: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract temporal and predicate flags based on source."""
    if flag_source == "stage15_probe_type":
        temporal_flags = temporal_mismatch_flags_from_probe(records, device)
        predicate_flags = predicate_mismatch_flags_from_probe(records, device)
    elif flag_source == "controlled_heuristic":
        temporal_flags = temporal_mismatch_flags_none(records, device)
        predicate_flags = predicate_mismatch_flags_from_intervention_type(records, device)
    elif flag_source == "none":
        temporal_flags = temporal_mismatch_flags_none(records, device)
        predicate_flags = temporal_mismatch_flags_none(records, device)
    else:
        raise ValueError(f"unknown flag_source: {flag_source}")
    return temporal_flags, predicate_flags


def _make_ablation_flags(
    mode: str,
    temporal_flags: torch.Tensor,
    predicate_flags: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (temporal, predicate) flag tensors for a given ablation mode."""
    zeros_t = torch.zeros_like(temporal_flags)
    zeros_p = torch.zeros_like(predicate_flags)
    if mode == "current":
        return temporal_flags, predicate_flags
    if mode == "no_flags":
        return zeros_t, zeros_p
    if mode == "temporal_only":
        return temporal_flags, zeros_p
    if mode == "predicate_only":
        return zeros_t, predicate_flags
    raise ValueError(f"unknown ablation mode: {mode!r}")


def _apply_ne_shift_and_eval(
    logits_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    unflagged_mask: torch.Tensor,
    records: list[dict],
    ne_idx: int,
    shift: float,
) -> dict[str, Any]:
    """Post-hoc NOT_ENTITLED logit shift for unflagged records; recompute predictions/metrics."""
    from collections import defaultdict

    adjusted = logits_cpu.clone()
    if shift != 0.0:
        adjusted[unflagged_mask, ne_idx] -= shift
    predictions = adjusted.argmax(dim=-1)

    # overall metrics
    pred_dist_overall: dict[str, int] = {}
    for pred_id in predictions.tolist():
        name = v5.ID_TO_FINAL_LABEL[pred_id]
        pred_dist_overall[name] = pred_dist_overall.get(name, 0) + 1
    overall_acc = (predictions == labels_cpu).float().mean().item()
    f1_overall: list[float] = []
    for label in v5.FinalLabel:
        lid = int(label)
        pred = predictions == lid
        actual = labels_cpu == lid
        tp = (pred & actual).sum().item()
        pd = pred.sum().item()
        rd = actual.sum().item()
        p = tp / pd if pd else 0.0
        r = tp / rd if rd else 0.0
        f1_overall.append(2.0 * p * r / (p + r) if p + r else 0.0)
    overall_metrics = {
        "final_accuracy": overall_acc,
        "final_macro_f1": sum(f1_overall) / len(f1_overall) if f1_overall else 0.0,
        "prediction_distribution": dict(sorted(pred_dist_overall.items())),
    }

    # per-group metrics (same grouping key as evaluate_ood_v6b)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[record.get("stage15_probe_type", "unknown")].append(idx)

    group_metrics: dict[str, Any] = {}
    for group_name, indices in sorted(groups.items()):
        g_preds = predictions[indices]
        g_labels = labels_cpu[indices]
        n = len(indices)
        acc = (g_preds == g_labels).float().mean().item()
        f1_vals: list[float] = []
        for label in v5.FinalLabel:
            lid = int(label)
            pred = g_preds == lid
            actual = g_labels == lid
            tp = (pred & actual).sum().item()
            pd = pred.sum().item()
            rd = actual.sum().item()
            p = tp / pd if pd else 0.0
            r = tp / rd if rd else 0.0
            f1_vals.append(2.0 * p * r / (p + r) if p + r else 0.0)
        macro_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0
        pred_dist: dict[str, int] = {}
        for pred_id in g_preds.tolist():
            name = v5.ID_TO_FINAL_LABEL[pred_id]
            pred_dist[name] = pred_dist.get(name, 0) + 1
        fe_count = fne_count = gold_entitled = 0
        for pred_id, gold_id in zip(g_preds.tolist(), g_labels.tolist()):
            if gold_id == ne_idx and pred_id != ne_idx:
                fe_count += 1
            if gold_id != ne_idx:
                gold_entitled += 1
                if pred_id == ne_idx:
                    fne_count += 1
        group_metrics[group_name] = {
            "n": n,
            "final_accuracy": acc,
            "final_macro_f1": macro_f1,
            "prediction_distribution": dict(sorted(pred_dist.items())),
            "false_entitled_count": fe_count,
            "false_entitled_rate": fe_count / n if n > 0 else 0.0,
            "false_not_entitled_count": fne_count,
            "false_not_entitled_rate": fne_count / gold_entitled if gold_entitled > 0 else None,
        }

    return {
        "n_records": len(records),
        "overall_metrics": overall_metrics,
        "group_metrics": group_metrics,
    }


def _build_gate_mask(
    gate_name: str,
    unflagged_mask: torch.Tensor,
    aux_probs: dict[str, "torch.Tensor | None"],
    threshold: float,
) -> torch.Tensor:
    """Return bool mask for records chosen by a gate (always a subset of unflagged_mask).

    aux_probs must contain the keys the gate needs; raises ValueError if missing.
    The gate does NOT use gold group labels or stage15_probe_type.
    """
    _GATE_REQUIRED: dict[str, list[str]] = {
        "high_sufficiency": ["sufficiency_prob"],
        "high_frame": ["frame_prob"],
        "high_frame_sufficiency": ["frame_prob", "sufficiency_prob"],
        "high_frame_suff_predicate": [
            "frame_prob", "sufficiency_prob", "predicate_coverage_prob"
        ],
    }
    if gate_name not in _GATE_REQUIRED:
        raise ValueError(
            f"Unknown selective gate {gate_name!r}. "
            f"Valid gates: {sorted(_GATE_REQUIRED)}"
        )
    for req_key in _GATE_REQUIRED[gate_name]:
        if aux_probs.get(req_key) is None:
            raise ValueError(
                f"Gate {gate_name!r} requires model output key {req_key!r} "
                f"but the model did not return it. "
                f"Ensure the v6B forward pass produces this auxiliary probability."
            )
    mask = unflagged_mask.clone()
    for req_key in _GATE_REQUIRED[gate_name]:
        mask = mask & (aux_probs[req_key] >= threshold)
    return mask


# Stage22-A: boundary label mapping
# preservation-positive: surface-form-preserving interventions (gold = SUPPORT or REFUTE)
# frame-mismatch-negative: structural frame/slot swaps (gold = NOT_ENTITLED)
# excluded: sufficiency, polarity, predicate, unknown
_BOUNDARY_POSITIVE: frozenset = frozenset({"none", "paraphrase"})
_BOUNDARY_NEGATIVE: frozenset = frozenset({
    "location_swap", "role_swap", "entity_swap", "event_swap", "title_name_swap",
})

# Stage22-A3: frame violation label mapping
# positive/violation=1: frame-slot swaps that produce a structural frame mismatch
# negative/non-violation=0: preservation controls + sufficiency/polarity types (all non-frame)
# excluded/masked: predicate_swap (predicate mismatch, not frame), time_swap (absent in
#   controlled_v5_v3_without_time_swap.jsonl), and any unknown intervention_type
# Class balance in controlled_v5_v3_without_time_swap.jsonl:
#   positive: 5 types x 300 = 1500; negative: 6 types x 300 = 1800; ratio neg/pos = 1.2
_FRAME_VIOLATION_POSITIVE: frozenset = frozenset({
    "entity_swap", "event_swap", "location_swap", "role_swap", "title_name_swap",
})
_FRAME_VIOLATION_NEGATIVE: frozenset = frozenset({
    "none", "paraphrase",
    "evidence_deletion", "evidence_truncation", "irrelevant_evidence",
    "polarity_flip",
})

# Predicate isolation label mapping
# positive/noncoverage=1: predicate_swap (predicate mismatch — not frame, not sufficiency)
# negative/covered=0: none, paraphrase (predicate coverage present in both)
# excluded/masked: all frame-swap, evidence, polarity intervention types, and unknown
# Class balance in controlled_v5_v3_without_time_swap.jsonl:
#   positive: 300 predicate_swap; negative: 600 (none+paraphrase); ratio neg/pos = 2.0
# Supervision target: predicate_pair_repr only (NOT frame_pair_repr, NOT sufficiency_repr).
# This keeps predicate-noncoverage supervision separate from FrameGate.
_PRED_ISOLATION_POSITIVE: frozenset = frozenset({"predicate_swap"})
_PRED_ISOLATION_NEGATIVE: frozenset = frozenset({"none", "paraphrase"})

# Preservation entitlement label mapping
# positive/entitled=1: none, paraphrase (preservation-positive; should remain entitled)
# negative/rejected=0: entity_swap, event_swap, location_swap, role_swap, title_name_swap
#   (narrow true frame-mismatch rejections — these and only these are valid frame negatives)
# excluded/masked:
#   predicate_swap — predicate noncoverage path, kept separate from FrameGate supervision
#   evidence_deletion, evidence_truncation, irrelevant_evidence — sufficiency failures;
#     masking avoids conflating frame-rejection with evidence-insufficiency
#   polarity_flip — polarity failure, not frame/entitlement failure
# Class balance in controlled_v5_v3_without_time_swap.jsonl:
#   positive: 600 (none+paraphrase); negative: 1500 (5 frame-swap types x 300)
#   ratio neg/pos = 2.5
# Supervision target: sufficiency_repr only (entitlement-level gate output combining
# frame+predicate info). Distinct from boundary_head (full [frame+pred+suff] concat)
# and predicate_isolation_head (predicate_pair_repr only).
_PRES_ENT_POSITIVE: frozenset = frozenset({"none", "paraphrase"})
_PRES_ENT_NEGATIVE: frozenset = frozenset({
    "entity_swap", "event_swap", "location_swap", "role_swap", "title_name_swap",
})

# Temporal diagnostic label mapping
# Loaded from a SEPARATE temporal diagnostic JSONL built by
#   scripts/make_temporal_diagnostic_from_controlled.py from controlled_v5_v3.jsonl.
# positive/temporal_mismatch=1: time_swap records
#   (primary_failure_type='frame', frame_compatible_label=0)
# negative/temporal_control=0: none, paraphrase records (temporally safe controls)
# excluded: all other intervention types are not present in the temporal diagnostic file
# Supervision target: temporal_diagnostic_head on frame_pair_repr only.
#   time_swap is a frame-compatibility failure → frame_pair_repr is the most specific
#   available representation. Distinct from predicate_isolation_head (predicate_pair_repr)
#   and preservation_entitlement_head (sufficiency_repr).
# Class balance in temporal diagnostic file (default with paraphrase controls):
#   positive: 300 time_swap; negative: 600 (none+paraphrase); ratio neg/pos = 2.0
# Stage15 OOD records must NOT be present in the temporal diagnostic file.
# This dataset must NOT be mixed into the main clean train/eval classification data.
_TEMPORAL_DIAG_MISMATCH_ROLE: str = "temporal_mismatch"
_TEMPORAL_DIAG_CONTROL_ROLE: str = "temporal_control"
_TEMPORAL_DIAG_ALLOWED_ROLES: frozenset = frozenset({
    _TEMPORAL_DIAG_MISMATCH_ROLE, _TEMPORAL_DIAG_CONTROL_ROLE
})
_TEMPORAL_DIAG_REQUIRED_FIELDS: frozenset = frozenset({
    "temporal_diagnostic_label",
    "temporal_diagnostic_role",
    "source_intervention_type",
    "diagnostic_dataset",
    "leakage_note",
    "usage_note",
})


def load_temporal_diagnostic_jsonl(path: "Path") -> list[dict]:
    """Load and validate temporal diagnostic JSONL records.

    Validates schema fields, roles, labels, and that the path does not reference Stage15.
    Raises ValueError or FileNotFoundError on any validation failure.
    """
    path_str = str(path).lower()
    if "stage15" in path_str:
        raise ValueError(
            f"[td_head] temporal diagnostic path contains 'stage15': {path}\n"
            "Stage15 OOD records must not be used for temporal diagnostic training."
        )
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"[td_head] Malformed JSON on line {lineno}: {exc}"
                ) from exc
            missing = _TEMPORAL_DIAG_REQUIRED_FIELDS - set(r.keys())
            if missing:
                raise ValueError(
                    f"[td_head] Record {lineno} missing required fields: {sorted(missing)}.\n"
                    "Ensure temporal diagnostic file was built by "
                    "scripts/make_temporal_diagnostic_from_controlled.py."
                )
            role = r.get("temporal_diagnostic_role", "")
            if role not in _TEMPORAL_DIAG_ALLOWED_ROLES:
                raise ValueError(
                    f"[td_head] Record {lineno} has invalid temporal_diagnostic_role: {role!r}. "
                    f"Expected one of {sorted(_TEMPORAL_DIAG_ALLOWED_ROLES)}."
                )
            lbl = r.get("temporal_diagnostic_label")
            if lbl not in (0, 1):
                raise ValueError(
                    f"[td_head] Record {lineno} has invalid temporal_diagnostic_label: {lbl!r}. "
                    "Expected 0 or 1."
                )
            records.append(r)
    return records


def encode_temporal_diagnostic_labels(
    records: list[dict],
    device: "torch.device",
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Encode temporal diagnostic labels from temporal diagnostic file records.

    Reads temporal_diagnostic_label field directly (0 or 1).
    All records with a valid label are included (mask=1).

    Returns (labels, mask) both of shape [B].
      label=1, mask=1: temporal_mismatch (time_swap)
      label=0, mask=1: temporal_control (none, paraphrase)
      label=0, mask=0: records missing temporal_diagnostic_label (should not occur in valid files)
    """
    labels: list[int] = []
    mask: list[int] = []
    for r in records:
        lbl = r.get("temporal_diagnostic_label")
        if lbl is not None:
            labels.append(int(lbl))
            mask.append(1)
        else:
            labels.append(0)
            mask.append(0)
    return (
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
    )


def compute_temporal_diagnostic_metrics(
    output: dict[str, Any],
    td_labels: "torch.Tensor | None",
    td_mask: "torch.Tensor | None",
) -> dict[str, Any]:
    """Compute temporal diagnostic head metrics.

    Returns empty dict when the head is disabled or no valid examples.
    Keys: td_accuracy, td_mismatch_recall, td_control_acceptance, td_valid_count, td_mean_prob.
    """
    if (
        output.get("temporal_diagnostic_prob") is None
        or td_labels is None
        or td_mask is None
    ):
        return {}
    probs = output["temporal_diagnostic_prob"].detach().cpu()
    labels = td_labels.detach().cpu()
    mask = td_mask.detach().cpu().bool()
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return {"td_valid_count": 0}
    probs_v = probs[mask]
    labels_v = labels[mask]
    preds_v = (probs_v >= 0.5).float()
    accuracy = (preds_v == labels_v).float().mean().item()
    mean_prob = probs_v.mean().item()
    pos_mask = labels_v == 1
    neg_mask = labels_v == 0
    mismatch_recall = (
        (preds_v[pos_mask] == 1).float().mean().item() if pos_mask.any() else float("nan")
    )
    control_acceptance = (
        (preds_v[neg_mask] == 0).float().mean().item() if neg_mask.any() else float("nan")
    )
    return {
        "td_accuracy": round(accuracy, 4),
        "td_mismatch_recall": (
            round(mismatch_recall, 4) if mismatch_recall == mismatch_recall else mismatch_recall
        ),
        "td_control_acceptance": (
            round(control_acceptance, 4)
            if control_acceptance == control_acceptance else control_acceptance
        ),
        "td_valid_count": valid_count,
        "td_mean_prob": round(mean_prob, 4),
    }


def compute_temporal_adapter_metrics(
    output: dict[str, Any],
    td_labels: "torch.Tensor | None",
    td_mask: "torch.Tensor | None",
) -> dict[str, Any]:
    """Compute temporal residual adapter metrics.

    Returns empty dict when adapter is disabled or no valid examples.
    Keys: ta_accuracy, ta_mismatch_recall, ta_control_acceptance, ta_valid_count, ta_mean_prob.
    Uses temporal_adapter_prob (output of the 2-layer MLP adapter) — not temporal_diagnostic_prob.
    Labels: 1=temporal_mismatch (time_swap), 0=temporal_control (none/paraphrase).
    """
    if (
        output.get("temporal_adapter_prob") is None
        or td_labels is None
        or td_mask is None
    ):
        return {}
    probs = output["temporal_adapter_prob"].detach().cpu()
    labels = td_labels.detach().cpu()
    mask = td_mask.detach().cpu().bool()
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return {"ta_valid_count": 0}
    probs_v = probs[mask]
    labels_v = labels[mask]
    preds_v = (probs_v >= 0.5).float()
    accuracy = (preds_v == labels_v).float().mean().item()
    mean_prob = probs_v.mean().item()
    pos_mask = labels_v == 1
    neg_mask = labels_v == 0
    mismatch_recall = (
        (preds_v[pos_mask] == 1).float().mean().item() if pos_mask.any() else float("nan")
    )
    control_acceptance = (
        (preds_v[neg_mask] == 0).float().mean().item() if neg_mask.any() else float("nan")
    )
    return {
        "ta_accuracy": round(accuracy, 4),
        "ta_mismatch_recall": (
            round(mismatch_recall, 4) if mismatch_recall == mismatch_recall else mismatch_recall
        ),
        "ta_control_acceptance": (
            round(control_acceptance, 4)
            if control_acceptance == control_acceptance else control_acceptance
        ),
        "ta_valid_count": valid_count,
        "ta_mean_prob": round(mean_prob, 4),
    }


def compute_td_final_decision_metrics(
    output: dict[str, Any],
    td_labels: "torch.Tensor | None",
    td_mask: "torch.Tensor | None",
) -> dict[str, Any]:
    """Compute final classification decision behavior on temporal diagnostic dev records.

    Uses output["logits"] (final calibrated logits) to measure how well the classifier
    already rejects temporal mismatches and preserves controls, without any new loss.

    Label convention (temporal diagnostic): 1 = temporal_mismatch, 0 = temporal_control.
    A temporal_mismatch (label=1) is correctly handled if the final prediction is NOT_ENTITLED.
    A temporal_control (label=0) is correctly handled if the final prediction is NOT NOT_ENTITLED.

    Keys returned:
      td_final_temporal_rejection_rate  — among label=1 records, frac predicted NOT_ENTITLED
      td_final_control_preservation_rate — among label=0 records, frac predicted non-NOT_ENTITLED
      td_final_binary_accuracy           — overall binary accuracy under the above mapping
      td_final_temporal_count            — number of label=1 records with valid mask
      td_final_control_count             — number of label=0 records with valid mask

    Returns empty dict when logits are absent or no valid examples.
    Stage15/OOD data is never passed here; this runs only on --temporal-diagnostic-data.
    """
    if (
        output.get("logits") is None
        or td_labels is None
        or td_mask is None
    ):
        return {}
    logits = output["logits"].detach().cpu()
    labels = td_labels.detach().cpu()
    mask = td_mask.detach().cpu().bool()
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return {"td_final_temporal_count": 0, "td_final_control_count": 0}
    ne_idx: int = v5.FINAL_LABEL_TO_ID.get("NOT_ENTITLED", 1)
    preds = logits.argmax(dim=-1)
    labels_v = labels[mask]
    preds_v = preds[mask]
    pos_mask = labels_v == 1  # temporal_mismatch
    neg_mask = labels_v == 0  # temporal_control
    td_final_temporal_count = int(pos_mask.sum().item())
    td_final_control_count = int(neg_mask.sum().item())
    temporal_rejection_rate = (
        (preds_v[pos_mask] == ne_idx).float().mean().item()
        if pos_mask.any() else float("nan")
    )
    control_preservation_rate = (
        (preds_v[neg_mask] != ne_idx).float().mean().item()
        if neg_mask.any() else float("nan")
    )
    # Binary accuracy: mismatch correct if NOT_ENTITLED, control correct if not NOT_ENTITLED
    correct_pos = (preds_v[pos_mask] == ne_idx).float().sum() if pos_mask.any() else torch.tensor(0.0)
    correct_neg = (preds_v[neg_mask] != ne_idx).float().sum() if neg_mask.any() else torch.tensor(0.0)
    binary_accuracy = (correct_pos + correct_neg).item() / valid_count
    return {
        "td_final_temporal_rejection_rate": (
            round(temporal_rejection_rate, 4)
            if temporal_rejection_rate == temporal_rejection_rate else temporal_rejection_rate
        ),
        "td_final_control_preservation_rate": (
            round(control_preservation_rate, 4)
            if control_preservation_rate == control_preservation_rate else control_preservation_rate
        ),
        "td_final_binary_accuracy": round(binary_accuracy, 4),
        "td_final_temporal_count": td_final_temporal_count,
        "td_final_control_count": td_final_control_count,
    }


def encode_boundary_labels(
    records: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive preservation boundary binary labels from intervention_type.

    Returns (labels, mask) both of shape [B].
      label=1, mask=1: none, paraphrase  (preservation-positive)
      label=0, mask=1: location_swap, role_swap, entity_swap, event_swap,
                        title_name_swap  (frame-mismatch-negative)
      label=0, mask=0: evidence_deletion, evidence_truncation, irrelevant_evidence,
                        polarity_flip, predicate_swap, unknown  (excluded)
    BCE loss is computed only on mask==1 examples.
    Does NOT use OOD group names or gold labels at inference.
    """
    labels: list[int] = []
    mask: list[int] = []
    for r in records:
        it = r.get("intervention_type", "")
        if it in _BOUNDARY_POSITIVE:
            labels.append(1)
            mask.append(1)
        elif it in _BOUNDARY_NEGATIVE:
            labels.append(0)
            mask.append(1)
        else:
            labels.append(0)
            mask.append(0)
    return (
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
    )


def encode_frame_violation_labels(
    records: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive frame violation binary labels from intervention_type.

    Returns (labels, mask) both of shape [B].
      label=1, mask=1: entity_swap, event_swap, location_swap, role_swap,
                        title_name_swap  (frame-violation-positive)
      label=0, mask=1: none, paraphrase, evidence_deletion, evidence_truncation,
                        irrelevant_evidence, polarity_flip  (non-violation-negative)
      label=0, mask=0: predicate_swap, time_swap, unknown  (excluded)
    predicate_swap is excluded because it represents predicate mismatch (not frame).
    time_swap is excluded because it is absent from the filtered training dataset and
    would require a separate temporal comparator signal.
    BCE loss is computed only on mask==1 examples.
    Does NOT use OOD group names or gold labels at inference.
    """
    labels: list[int] = []
    mask: list[int] = []
    for r in records:
        it = r.get("intervention_type", "")
        if it in _FRAME_VIOLATION_POSITIVE:
            labels.append(1)
            mask.append(1)
        elif it in _FRAME_VIOLATION_NEGATIVE:
            labels.append(0)
            mask.append(1)
        else:
            labels.append(0)
            mask.append(0)
    return (
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
    )


def encode_predicate_isolation_labels(
    records: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive predicate isolation binary labels from intervention_type.

    Returns (labels, mask) both of shape [B].
      label=1, mask=1: predicate_swap  (predicate noncoverage)
      label=0, mask=1: none, paraphrase  (predicate covered)
      label=0, mask=0: all other intervention types  (excluded)
    BCE loss is computed only on mask==1 examples.
    Routes predicate-noncoverage supervision to predicate_pair_repr only;
    does NOT route predicate_swap into frame violation supervision.
    Does NOT use OOD group names or gold labels at inference.
    """
    labels: list[int] = []
    mask: list[int] = []
    for r in records:
        it = r.get("intervention_type", "")
        if it in _PRED_ISOLATION_POSITIVE:
            labels.append(1)
            mask.append(1)
        elif it in _PRED_ISOLATION_NEGATIVE:
            labels.append(0)
            mask.append(1)
        else:
            labels.append(0)
            mask.append(0)
    return (
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
    )


def compute_frame_violation_metrics(
    output: dict[str, Any],
    fv_labels: "torch.Tensor | None",
    fv_mask: "torch.Tensor | None",
) -> dict[str, Any]:
    """Compute frame violation head diagnostic metrics.

    Returns empty dict when the frame violation head is disabled or no valid examples.
    """
    if output.get("frame_violation_prob") is None or fv_labels is None or fv_mask is None:
        return {}
    probs = output["frame_violation_prob"].detach().cpu()
    labels = fv_labels.detach().cpu()
    mask = fv_mask.detach().cpu().bool()
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return {"fv_valid_count": 0}
    probs_v = probs[mask]
    labels_v = labels[mask]
    preds_v = (probs_v >= 0.5).float()
    accuracy = (preds_v == labels_v).float().mean().item()
    pos_rate = labels_v.mean().item()
    mean_prob = probs_v.mean().item()
    return {
        "fv_accuracy": round(accuracy, 4),
        "fv_pos_rate": round(pos_rate, 4),
        "fv_valid_count": valid_count,
        "fv_mean_prob": round(mean_prob, 4),
    }


def compute_boundary_metrics(
    output: dict[str, Any],
    boundary_labels: "torch.Tensor | None",
    boundary_mask: "torch.Tensor | None",
) -> dict[str, Any]:
    """Compute boundary head diagnostic metrics (accuracy, pos rate, valid count, mean prob).

    Returns empty dict when boundary head is disabled or no valid examples exist.
    """
    if output.get("boundary_prob") is None or boundary_labels is None or boundary_mask is None:
        return {}
    probs = output["boundary_prob"].detach().cpu()
    labels = boundary_labels.detach().cpu()
    mask = boundary_mask.detach().cpu().bool()
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return {"boundary_valid_count": 0}
    probs_v = probs[mask]
    labels_v = labels[mask]
    preds_v = (probs_v >= 0.5).float()
    accuracy = (preds_v == labels_v).float().mean().item()
    pos_rate = labels_v.mean().item()
    mean_prob = probs_v.mean().item()
    return {
        "boundary_accuracy": round(accuracy, 4),
        "boundary_pos_rate": round(pos_rate, 4),
        "boundary_valid_count": valid_count,
        "boundary_mean_prob": round(mean_prob, 4),
    }


def compute_predicate_isolation_metrics(
    output: dict[str, Any],
    pi_labels: "torch.Tensor | None",
    pi_mask: "torch.Tensor | None",
) -> dict[str, Any]:
    """Compute predicate isolation head diagnostic metrics.

    Returns empty dict when the head is disabled or no valid examples exist.
    """
    if (
        output.get("predicate_noncoverage_prob") is None
        or pi_labels is None
        or pi_mask is None
    ):
        return {}
    probs = output["predicate_noncoverage_prob"].detach().cpu()
    labels = pi_labels.detach().cpu()
    mask = pi_mask.detach().cpu().bool()
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return {"pi_valid_count": 0}
    probs_v = probs[mask]
    labels_v = labels[mask]
    preds_v = (probs_v >= 0.5).float()
    accuracy = (preds_v == labels_v).float().mean().item()
    pos_rate = labels_v.mean().item()
    mean_prob = probs_v.mean().item()
    return {
        "pi_accuracy": round(accuracy, 4),
        "pi_pos_rate": round(pos_rate, 4),
        "pi_valid_count": valid_count,
        "pi_mean_prob": round(mean_prob, 4),
    }


def encode_preservation_entitlement_labels(
    records: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive preservation entitlement binary labels from intervention_type.

    Returns (labels, mask) both of shape [B].
      label=1, mask=1: none, paraphrase  (preservation-entitled; should remain entitled)
      label=0, mask=1: entity_swap, event_swap, location_swap, role_swap,
                        title_name_swap  (narrow frame-mismatch rejections)
      label=0, mask=0: predicate_swap, evidence_deletion, evidence_truncation,
                        irrelevant_evidence, polarity_flip, unknown  (excluded)
    predicate_swap is excluded to keep predicate-noncoverage on the predicate path.
    Evidence-sufficiency types (deletion/truncation/irrelevant) are excluded to avoid
    conflating frame-rejection with evidence-insufficiency in the sufficiency_repr probe.
    polarity_flip is excluded as a polarity/refutation failure, not a frame rejection.
    BCE loss is computed only on mask==1 examples.
    Supervision target: preservation_entitlement_head on sufficiency_repr only.
    Does NOT use OOD group names or gold labels at inference.
    """
    labels: list[int] = []
    mask: list[int] = []
    for r in records:
        it = r.get("intervention_type", "")
        if it in _PRES_ENT_POSITIVE:
            labels.append(1)
            mask.append(1)
        elif it in _PRES_ENT_NEGATIVE:
            labels.append(0)
            mask.append(1)
        else:
            labels.append(0)
            mask.append(0)
    return (
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
    )


def compute_preservation_entitlement_metrics(
    output: dict[str, Any],
    pe_labels: "torch.Tensor | None",
    pe_mask: "torch.Tensor | None",
) -> dict[str, Any]:
    """Compute preservation entitlement head diagnostic metrics.

    Returns empty dict when the head is disabled or no valid examples exist.
    """
    if (
        output.get("preservation_entitlement_prob") is None
        or pe_labels is None
        or pe_mask is None
    ):
        return {}
    probs = output["preservation_entitlement_prob"].detach().cpu()
    labels = pe_labels.detach().cpu()
    mask = pe_mask.detach().cpu().bool()
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return {"pe_valid_count": 0}
    probs_v = probs[mask]
    labels_v = labels[mask]
    preds_v = (probs_v >= 0.5).float()
    accuracy = (preds_v == labels_v).float().mean().item()
    pos_rate = labels_v.mean().item()
    mean_prob = probs_v.mean().item()
    return {
        "pe_accuracy": round(accuracy, 4),
        "pe_pos_rate": round(pos_rate, 4),
        "pe_valid_count": valid_count,
        "pe_mean_prob": round(mean_prob, 4),
    }


# ---------------------------------------------------------------------------
# Stage22-A4c/A4e: pair-group contrastive frame helpers
# ---------------------------------------------------------------------------

# A4b2 use-case values (filter via contrastive_use_case field)
_PC_USE_CASES_A4B2 = frozenset({
    "frame_violation_contrastive",
    "support_safe_frame_contrastive",
})
# A4d OOD-matched use-case values (filter via target / source / preservation_construction_type)
_PC_USE_CASES_A4D = frozenset({
    "ood_matched",
    "ood_matched_surface",
    "ood_matched_temporal",
})
_PC_USE_CASES = _PC_USE_CASES_A4B2 | _PC_USE_CASES_A4D | frozenset({"all"})

_A4D_TARGET = "frame_more_violating_than_ood_matched_preservation"
_A4D_SOURCE = "controlled_ood_matched_pair_builder"


def _detect_pc_schema(r: dict[str, Any]) -> str:
    """Return 'a4d' when the record uses A4d OOD-matched schema, else 'a4b2'."""
    if r.get("source") == _A4D_SOURCE or r.get("target") == _A4D_TARGET:
        return "a4d"
    if "preservation_construction_type" in r or "frame_construction_type" in r:
        return "a4d"
    return "a4b2"


def _normalize_pair_record(r: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of r with canonical intervention_type fields filled in.

    A4b2 records already have preservation_intervention_type / frame_intervention_type.
    A4d records store these as preservation_source_intervention_type /
    frame_source_intervention_type; this function aliases them into the canonical names
    so that _pair_record_to_virtual_records works for both schemas without modification.
    """
    schema = _detect_pc_schema(r)
    if schema == "a4b2":
        return r  # already has canonical fields
    # A4d: alias source_intervention_type → intervention_type
    out = dict(r)
    if "preservation_intervention_type" not in out and "preservation_source_intervention_type" in out:
        out["preservation_intervention_type"] = out["preservation_source_intervention_type"]
    if "frame_intervention_type" not in out and "frame_source_intervention_type" in out:
        out["frame_intervention_type"] = out["frame_source_intervention_type"]
    return out


def _pc_filter(r: dict[str, Any], use_case_filter: str) -> bool:
    """Return True when record r passes the use_case_filter."""
    if use_case_filter == "all":
        return True
    schema = _detect_pc_schema(r)
    if use_case_filter in _PC_USE_CASES_A4B2:
        # A4b2 path: match contrastive_use_case field
        return schema == "a4b2" and r.get("contrastive_use_case") == use_case_filter
    if use_case_filter == "ood_matched":
        return schema == "a4d"
    if use_case_filter == "ood_matched_surface":
        return schema == "a4d" and r.get("preservation_construction_type") == "surface_like_preservation"
    if use_case_filter == "ood_matched_temporal":
        return schema == "a4d" and r.get("preservation_construction_type") == "temporal_erased_like_preservation"
    return False


def load_pair_contrastive_jsonl(
    path: Path,
    *,
    use_case_filter: str = "frame_violation_contrastive",
) -> list[dict[str, Any]]:
    """Load pair contrastive JSONL, detect schema (A4b2 or A4d), filter, and normalize.

    Filtering rules:
      frame_violation_contrastive / support_safe_frame_contrastive
          → A4b2 records matched by contrastive_use_case field
      ood_matched        → all A4d records (target or source tag present)
      ood_matched_surface  → A4d records where preservation_construction_type == surface_like_preservation
      ood_matched_temporal → A4d records where preservation_construction_type == temporal_erased_like_preservation
      all                → all records from either schema

    Returns normalized records (preservation_intervention_type and frame_intervention_type
    always populated). Stage15 OOD records are never loaded here.
    """
    raw: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    return [_normalize_pair_record(r) for r in raw if _pc_filter(r, use_case_filter)]


def compute_pair_contrastive_metrics(
    pres_output: dict[str, Any],
    frame_output: dict[str, Any],
    *,
    margin: float,
) -> dict[str, Any]:
    """Compute pair contrastive ranking metrics for the frame_violation_head.

    Returns empty dict when frame_violation_logit is absent from either output.
    Does NOT modify logits or predictions; purely diagnostic.
    """
    if (
        pres_output.get("frame_violation_logit") is None
        or frame_output.get("frame_violation_logit") is None
    ):
        return {}
    pres_logits = pres_output["frame_violation_logit"].detach().cpu().squeeze(-1)
    frame_logits = frame_output["frame_violation_logit"].detach().cpu().squeeze(-1)
    n = int(pres_logits.shape[0])
    if n == 0:
        return {"pair_contrastive_frame_valid_count": 0}
    diff = frame_logits - pres_logits
    accuracy = float((diff > 0).float().mean().item())
    mean_margin = float(diff.mean().item())
    ranking_loss = float(F.relu(margin - diff).mean().item())
    pres_probs = torch.sigmoid(pres_logits)
    frame_probs = torch.sigmoid(frame_logits)
    return {
        "pair_contrastive_frame_valid_count": n,
        "pair_contrastive_frame_accuracy": round(accuracy, 4),
        "pair_contrastive_frame_margin_mean": round(mean_margin, 4),
        "pair_contrastive_frame_loss": round(ranking_loss, 4),
        "pair_contrastive_frame_mean_pres_fv_prob": round(float(pres_probs.mean().item()), 4),
        "pair_contrastive_frame_mean_frame_fv_prob": round(float(frame_probs.mean().item()), 4),
    }


# Required fields for v5.encode_records / v5.encode_mamba_records.
# frame_compatible_label, sufficiency_label, predicate_covered_label must be int 0/1.
# polarity_label must be a key in v5.POLARITY_LABEL_TO_ID: NONE, REFUTE, SUPPORT.
_PC_PRES_REQUIRED = ("claim", "preservation_evidence", "preservation_intervention_type")
_PC_FRAME_REQUIRED = ("claim", "frame_evidence", "frame_intervention_type")
_PC_BINARY_DEFAULT = 0       # conservative default for binary label fields
_PC_POLARITY_DEFAULT = "NONE"
_PC_FINAL_LABEL_PRES_DEFAULT = "SUPPORT"
_PC_FINAL_LABEL_FRAME_DEFAULT = "NOT_ENTITLED"
_PC_FAILURE_DEFAULT = "none"


def _pair_record_to_virtual_records(
    r: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one pair-contrastive record to (preservation_virtual, frame_virtual).

    Both virtual records satisfy the full schema expected by v5.encode_records and
    v5.encode_mamba_records, including all auxiliary label fields.

    Required pair fields: claim, preservation_evidence, frame_evidence,
    preservation_intervention_type, frame_intervention_type.
    Missing required fields raise a clear ValueError naming the field and contrastive_id.

    Optional label fields fall back to conservative defaults when absent rather than
    crashing with KeyError.  These defaults are used only for tensor construction —
    the pair-contrastive loss reads frame_violation_logit, not these label tensors.
    """
    cid = r.get("contrastive_id", "<unknown>")
    _schema = _detect_pc_schema(r)
    missing = [f for f in (*_PC_PRES_REQUIRED, *_PC_FRAME_REQUIRED) if not r.get(f)]
    if missing:
        raise ValueError(
            f"[pc22a4e] pair record {cid!r} (schema={_schema!r}) is missing required "
            f"field(s): {missing}. "
            "A4b2 records must have preservation_intervention_type / frame_intervention_type. "
            "A4d records must be normalized first via load_pair_contrastive_jsonl."
        )

    pres: dict[str, Any] = {
        "id": r.get("preservation_source_id") or f"{cid}__pres",
        "pair_id": r.get("pair_id", cid),
        "claim": r["claim"],
        "evidence": r["preservation_evidence"],
        "intervention_type": r["preservation_intervention_type"],
        "final_label": r.get("preservation_final_label") or _PC_FINAL_LABEL_PRES_DEFAULT,
        "frame_compatible_label": int(r.get("frame_compatible_label", _PC_BINARY_DEFAULT)),
        "sufficiency_label": int(r.get("sufficiency_label", _PC_BINARY_DEFAULT)),
        "predicate_covered_label": int(r.get("predicate_covered_label", _PC_BINARY_DEFAULT)),
        "polarity_label": r.get("polarity_label") or _PC_POLARITY_DEFAULT,
        "primary_failure_type": r.get("primary_failure_type") or _PC_FAILURE_DEFAULT,
    }

    frame: dict[str, Any] = {
        "id": r.get("frame_source_id") or f"{cid}__frame",
        "pair_id": r.get("pair_id", cid),
        "claim": r["claim"],
        "evidence": r["frame_evidence"],
        "intervention_type": r["frame_intervention_type"],
        "final_label": r.get("frame_final_label") or _PC_FINAL_LABEL_FRAME_DEFAULT,
        "frame_compatible_label": int(r.get("frame_frame_compatible_label", _PC_BINARY_DEFAULT)),
        "sufficiency_label": int(r.get("frame_sufficiency_label", _PC_BINARY_DEFAULT)),
        "predicate_covered_label": int(r.get("frame_predicate_covered_label", _PC_BINARY_DEFAULT)),
        "polarity_label": r.get("frame_polarity_label") or _PC_POLARITY_DEFAULT,
        "primary_failure_type": r.get("frame_primary_failure_type") or _PC_FAILURE_DEFAULT,
    }

    return pres, frame


def compute_class_weights_v6b(
    records: list[dict],
    mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    n_classes = len(v5.FinalLabel)
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for record in records:
        counts[v5.FINAL_LABEL_TO_ID[record["final_label"]]] += 1.0
    counts = counts.clamp_min(1.0)
    if mode == "inverse_freq":
        weights = 1.0 / counts
    elif mode == "sqrt_inverse_freq":
        weights = 1.0 / counts.sqrt()
    else:
        raise ValueError(f"unknown class_weighting mode: {mode!r}")
    weights = weights / weights.mean()
    return weights.to(device)


def evaluate_ood_v6b(
    model: ContraMambaV6BMinimal,
    records: list[dict],
    inputs: dict[str, torch.Tensor],
    temporal_flags: torch.Tensor,
    predicate_flags: torch.Tensor,
) -> tuple[dict[str, Any], list[dict]]:
    """Evaluate trained model on OOD records using final output logits/predictions."""
    from collections import defaultdict

    model.eval()
    with torch.no_grad():
        output = model(
            **v5.model_feature_inputs(inputs),
            temporal_mismatch_flags=temporal_flags,
            predicate_mismatch_flags=predicate_flags,
        )

    overall_metrics = v5.compute_metrics(output, inputs)
    predictions_export = prediction_records_v6b(records, output)

    predictions_cpu = output["predictions"].detach().cpu()
    labels_cpu = inputs["final_labels"].detach().cpu()
    not_entitled_id = v5.FINAL_LABEL_TO_ID.get("NOT_ENTITLED")

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[record.get("stage15_probe_type", "unknown")].append(idx)

    group_metrics: dict[str, dict[str, Any]] = {}
    for group_name, indices in sorted(groups.items()):
        g_preds = predictions_cpu[indices]
        g_labels = labels_cpu[indices]
        n = len(indices)
        accuracy = (g_preds == g_labels).float().mean().item()
        f1_values = []
        for label in v5.FinalLabel:
            label_id = int(label)
            predicted = g_preds == label_id
            actual = g_labels == label_id
            tp = (predicted & actual).sum().item()
            prec_denom = predicted.sum().item()
            rec_denom = actual.sum().item()
            prec = tp / prec_denom if prec_denom else 0.0
            rec = tp / rec_denom if rec_denom else 0.0
            f1_values.append(2.0 * prec * rec / (prec + rec) if prec + rec else 0.0)
        macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
        pred_dist: dict[str, int] = {}
        for pred_id in g_preds.tolist():
            label_name = v5.ID_TO_FINAL_LABEL[pred_id]
            pred_dist[label_name] = pred_dist.get(label_name, 0) + 1
        false_entitled_count = 0
        false_not_entitled_count = 0
        gold_entitled_count = 0
        if not_entitled_id is not None:
            for pred_id, gold_id in zip(g_preds.tolist(), g_labels.tolist()):
                if gold_id == not_entitled_id and pred_id != not_entitled_id:
                    false_entitled_count += 1
                if gold_id != not_entitled_id:
                    gold_entitled_count += 1
                    if pred_id == not_entitled_id:
                        false_not_entitled_count += 1
        false_entitled_rate = false_entitled_count / n if n > 0 else 0.0
        false_not_entitled_rate = (
            false_not_entitled_count / gold_entitled_count if gold_entitled_count > 0 else None
        )
        group_entry: dict[str, Any] = {
            "n": n,
            "final_accuracy": accuracy,
            "final_macro_f1": macro_f1,
            "prediction_distribution": dict(sorted(pred_dist.items())),
            "false_entitled_count": false_entitled_count,
            "false_entitled_rate": false_entitled_rate,
            "false_not_entitled_count": false_not_entitled_count,
            "false_not_entitled_rate": false_not_entitled_rate,
        }
        # Stage22-A: per-group boundary_prob diagnostics (absent when head disabled)
        if output.get("boundary_prob") is not None:
            bp_cpu = output["boundary_prob"].detach().cpu()
            bp_group = bp_cpu[indices]
            group_entry["boundary_prob_mean"] = round(float(bp_group.mean().item()), 4)
            group_entry["boundary_prob_std"] = round(float(bp_group.std().item()), 4)
        # Stage22-A3: per-group frame_violation_prob diagnostics (absent when head disabled)
        if output.get("frame_violation_prob") is not None:
            fv_cpu = output["frame_violation_prob"].detach().cpu()
            fv_group = fv_cpu[indices]
            group_entry["frame_violation_prob_mean"] = round(float(fv_group.mean().item()), 4)
            group_entry["frame_violation_prob_std"] = round(float(fv_group.std().item()), 4)
        group_metrics[group_name] = group_entry

    return {
        "n_records": len(records),
        "overall_metrics": overall_metrics,
        "group_metrics": group_metrics,
    }, predictions_export


def prediction_records_v6b(records: list[dict], output: dict[str, Any]) -> list[dict]:
    """Export predictions with v6b metadata."""
    probabilities = torch.softmax(output["logits"], dim=-1).detach().cpu()
    predictions = output["predictions"].detach().cpu()
    scalar_keys = (
        "frame_prob",
        "predicate_coverage_prob",
        "sufficiency_prob",
        "entitlement_prob",
        "polarity_margin",
        "boundary_prob",        # Stage22-A:  None when boundary head is disabled
        "frame_violation_prob", # Stage22-A3: None when frame violation head is disabled
    )
    scalars = {
        key: output[key].detach().cpu()
        for key in scalar_keys
        if key in output and output[key] is not None
    }
    exported: list[dict] = []
    for index, record in enumerate(records):
        item = {
            "id": record["id"],
            "pair_id": record["pair_id"],
            "intervention_type": record["intervention_type"],
            "claim": record["claim"],
            "evidence": record["evidence"],
            "gold_final_label": record["final_label"],
            "pred_final_label": v5.ID_TO_FINAL_LABEL[int(predictions[index])],
            "final_probs": probabilities[index].tolist(),
            **{key: float(scalars[key][index]) for key in scalar_keys if key in scalars},
        }
        exported.append(item)
    return exported


def intervention_diagnostics_v6b(
    records: list[dict], output: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """v6b extension of v5.intervention_diagnostics: adds auxiliary head probs when available.

    Delegates grouping and standard scalar aggregation to v5, then injects:
      - boundary_prob_mean per intervention type (Stage22-A, when head is active)
      - frame_violation_prob_mean per intervention type (Stage22-A3, when head is active)
    When a head is disabled, its prob_mean is omitted entirely -- no crash, no None.
    """
    from collections import defaultdict
    result = v5.intervention_diagnostics(records, output)
    has_boundary = output.get("boundary_prob") is not None
    has_fv = output.get("frame_violation_prob") is not None
    if has_boundary or has_fv:
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, record in enumerate(records):
            grouped[record["intervention_type"]].append(idx)
        bp_cpu = output["boundary_prob"].detach().cpu() if has_boundary else None
        fv_cpu = output["frame_violation_prob"].detach().cpu() if has_fv else None
        for intervention, indices in grouped.items():
            if intervention not in result:
                continue
            if bp_cpu is not None:
                result[intervention]["boundary_prob_mean"] = round(
                    float(bp_cpu[indices].mean().item()), 4
                )
            if fv_cpu is not None:
                result[intervention]["frame_violation_prob_mean"] = round(
                    float(fv_cpu[indices].mean().item()), 4
                )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = v5.build_parser()
    parser.add_argument(
        "--use-temporal-comparator",
        action="store_true",
        default=True,
        help="Use learnable temporal comparator alpha",
    )
    parser.add_argument(
        "--use-predicate-comparator",
        action="store_true",
        default=True,
        help="Use learnable predicate comparator alpha",
    )
    parser.add_argument(
        "--flag-source",
        choices=("stage15_probe_type", "controlled_heuristic", "none"),
        default="controlled_heuristic",
        help="Source for temporal/predicate flags",
    )
    parser.add_argument(
        "--max-train-records",
        type=int,
        default=None,
        help="Max train records (for smoke testing)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: tiny settings, small data",
    )
    parser.add_argument(
        "--allow-dummy-backbone",
        action="store_true",
        default=False,
        help=(
            "Permit backbone=dummy to run. "
            "Dummy backbone is for SMOKE / PLUMBING VALIDATION ONLY and produces "
            "results that are NOT claim-worthy as model performance evidence. "
            "Use --backbone mamba for any claim-worthy experiment. "
            "This flag must be passed explicitly to prevent accidental dummy runs."
        ),
    )
    parser.add_argument(
        "--class-weighting",
        choices=("none", "inverse_freq", "sqrt_inverse_freq"),
        default="none",
        help="Class weighting mode for CE classification loss (none preserves existing behavior)",
    )
    parser.add_argument(
        "--ood-ablation-modes",
        type=str,
        default=None,
        help=(
            "Comma-separated OOD ablation modes to evaluate after training. "
            "Valid values: current,no_flags,temporal_only,predicate_only. "
            "When set, report includes an ood_ablation object keyed by mode. "
            "When unset, existing single-mode OOD evaluation is used."
        ),
    )
    parser.add_argument(
        "--ood-unflagged-ne-shift-sweep",
        type=str,
        default=None,
        help=(
            "Comma-separated float shift values for post-hoc NOT_ENTITLED logit calibration "
            "applied to unflagged OOD records (where both temporal_flag==0 and "
            "predicate_flag==0). E.g. '0,0.25,0.5,0.75,1.0'. Subtracts each shift from "
            "the NOT_ENTITLED final logit and recomputes predictions. Eval-only; does not "
            "affect training, losses, or checkpoint selection. "
            "Cannot be combined with --ood-ablation-modes."
        ),
    )
    parser.add_argument(
        "--ood-selective-ne-shift-sweep",
        type=str,
        default=None,
        help=(
            "Comma-separated float shift values for selective post-hoc NOT_ENTITLED logit "
            "calibration. Applies the shift only to unflagged OOD records that also pass "
            "a preservation-like gate defined by --ood-selective-ne-gates and "
            "--ood-selective-ne-thresholds. E.g. '0,0.1,0.2,0.25,0.3,0.4'. "
            "Eval-only; does not affect training. "
            "Mutually exclusive with --ood-ablation-modes and --ood-unflagged-ne-shift-sweep."
        ),
    )
    parser.add_argument(
        "--ood-selective-ne-gates",
        type=str,
        default=None,
        help=(
            "Comma-separated gate names for --ood-selective-ne-shift-sweep. "
            "Default when sweep is set: 'high_sufficiency,high_frame_sufficiency,"
            "high_frame_suff_predicate'. "
            "Valid gates: high_sufficiency, high_frame, high_frame_sufficiency, "
            "high_frame_suff_predicate. "
            "Each gate uses model-internal auxiliary probabilities (not gold labels) "
            "to select unflagged records for the NOT_ENTITLED logit shift."
        ),
    )
    parser.add_argument(
        "--ood-selective-ne-thresholds",
        type=str,
        default=None,
        help=(
            "Comma-separated float thresholds for --ood-selective-ne-shift-sweep. "
            "Default: '0.7'. Applied as a lower bound on each auxiliary probability "
            "required by the gate (e.g. sufficiency_prob >= threshold). "
            "Each threshold is swept over all gates and shifts."
        ),
    )
    # Stage22-G2: dev-calibrated selective NE shift
    parser.add_argument(
        "--dev-calibrated-ne-shift-candidates",
        type=str,
        default=None,
        help=(
            "Comma-separated float shift values for dev-calibrated selective NE logit "
            "calibration. E.g. '0,0.25,0.5,0.75,1.0'. "
            "Selects the best shift on controlled dev only via: "
            "score = pres_accept_rate - frame_penalty * frame_false_entitled_rate. "
            "The selected shift is then applied to OOD evaluation (if --ood-data is set) "
            "without inspecting Stage15 OOD labels. "
            "stage15_used_for_shift_selection is always false. "
            "When unset, dev calibration is skipped."
        ),
    )
    parser.add_argument(
        "--dev-calibrated-ne-gate",
        choices=("high_sufficiency", "high_frame_sufficiency", "high_frame_suff_predicate"),
        default="high_sufficiency",
        help=(
            "Gate used for dev-calibrated selective NE shift. "
            "Selects unflagged records for the NOT_ENTITLED logit shift using "
            "model-internal auxiliary probabilities (not gold labels). "
            "high_sufficiency: sufficiency_prob >= threshold. "
            "high_frame_sufficiency: frame_prob AND sufficiency_prob >= threshold. "
            "high_frame_suff_predicate: all three aux probs >= threshold. "
            "Default: high_sufficiency."
        ),
    )
    parser.add_argument(
        "--dev-calibrated-ne-threshold",
        type=float,
        default=0.6,
        help=(
            "Auxiliary probability threshold for dev-calibrated NE gate. "
            "Lower bound applied to each probability required by the gate. "
            "Default: 0.6."
        ),
    )
    parser.add_argument(
        "--dev-calibrated-ne-frame-penalty",
        type=float,
        default=2.0,
        help=(
            "Frame false-entitlement penalty weight in the dev calibration objective. "
            "score = pres_accept_rate - frame_penalty * frame_false_entitled_rate. "
            "Higher values penalize predicting SUPPORT/REFUTE for frame-mismatch records. "
            "Ignored when --dev-calibrated-ne-frame-penalty-candidates is set. "
            "Default: 2.0."
        ),
    )
    parser.add_argument(
        "--dev-calibrated-ne-calibration-source",
        choices=("dev", "train", "train_dev"),
        default="dev",
        help=(
            "Controlled data split used to calibrate the selective NE shift (Stage22-G3). "
            "dev: use held-out controlled dev records only (G2 behavior, default). "
            "train: use controlled train records for a larger calibration pool. "
            "train_dev: concatenate train and dev records for the largest pool. "
            "Stage15/OOD records are NEVER used regardless of this setting. "
            "stage15_used_for_shift_selection is always false."
        ),
    )
    parser.add_argument(
        "--dev-calibrated-ne-frame-penalty-candidates",
        type=str,
        default=None,
        help=(
            "Comma-separated float penalty candidates for G3 joint (penalty, shift) selection. "
            "E.g. '0.5,1.0,1.5,2.0'. When set, sweeps all combinations of "
            "--dev-calibrated-ne-shift-candidates x frame-penalty-candidates and selects "
            "the best pair by: (1) highest objective score, then tie-breaks: "
            "(2) lower frame_false_entitled_rate, (3) higher pres_accept_rate, "
            "(4) lower shift, (5) higher frame_penalty. "
            "When absent, uses the single --dev-calibrated-ne-frame-penalty (G2 behavior)."
        ),
    )
    # Stage22-A: preservation boundary head
    parser.add_argument(
        "--use-boundary-loss",
        action="store_true",
        default=False,
        help=(
            "Enable Stage22-A preservation boundary head. "
            "Adds a linear head on [frame_pair_repr, predicate_pair_repr, sufficiency_repr] "
            "trained to distinguish preservation-like records (none, paraphrase) from "
            "frame-mismatch records (location_swap, role_swap, entity_swap, event_swap, "
            "title_name_swap) using BCE loss. "
            "Does NOT modify output['logits'] or affect predictions."
        ),
    )
    parser.add_argument(
        "--boundary-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the boundary BCE loss added to the total training loss. "
            "Default 0.0 means the head is built but its loss is not included. "
            "Requires --use-boundary-loss. Suggested starting value: 0.5."
        ),
    )
    parser.add_argument(
        "--boundary-loss-pos-weight",
        type=float,
        default=2.5,
        help=(
            "Positive class weight for the boundary BCE loss (pos_weight in "
            "F.binary_cross_entropy_with_logits). Compensates for class imbalance: "
            "preservation-positive (none+paraphrase=600) vs frame-mismatch (5 types x 300=1500). "
            "Default 2.5 = 1500/600."
        ),
    )
    # Stage22-A3: frame violation head
    parser.add_argument(
        "--use-frame-violation-loss",
        action="store_true",
        default=False,
        help=(
            "Enable Stage22-A3 frame violation head. "
            "Adds a linear head on [frame_pair_repr, predicate_pair_repr, sufficiency_repr] "
            "trained to distinguish frame-violating interventions (entity_swap, event_swap, "
            "location_swap, role_swap, title_name_swap; violation=1) from non-violating "
            "interventions (none, paraphrase, evidence_deletion, evidence_truncation, "
            "irrelevant_evidence, polarity_flip; violation=0). predicate_swap and time_swap "
            "are excluded from the BCE loss. "
            "Does NOT modify output['logits'] or affect predictions."
        ),
    )
    parser.add_argument(
        "--frame-violation-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the frame violation BCE loss added to the total training loss. "
            "Default 0.0 means the head is built but its loss is not included. "
            "Requires --use-frame-violation-loss. Suggested starting value: 0.5."
        ),
    )
    parser.add_argument(
        "--frame-violation-loss-pos-weight",
        type=float,
        default=1.2,
        help=(
            "Positive class weight for the frame violation BCE loss (pos_weight in "
            "F.binary_cross_entropy_with_logits). Compensates for class imbalance: "
            "violation-positive (5 types x 300=1500) vs non-violation-negative "
            "(6 types x 300=1800). Default 1.2 = 1800/1500."
        ),
    )
    # Stage22-A4c: pair-group contrastive frame ranking loss
    parser.add_argument(
        "--pair-contrastive-frame-data",
        type=str,
        default=None,
        help=(
            "Path to Stage22-A4b pair contrastive JSONL "
            "(e.g. data/stage22a4_pair_contrastive_frame.jsonl). "
            "Must have been constructed from controlled data only "
            "(leakage_note = constructed_from_controlled_data_only). "
            "Stage15 OOD records must NOT be present. "
            "Required when --use-pair-contrastive-frame-loss is set."
        ),
    )
    parser.add_argument(
        "--use-pair-contrastive-frame-loss",
        action="store_true",
        default=False,
        help=(
            "Enable Stage22-A4c pair-group contrastive frame ranking loss. "
            "Trains the frame_violation_head to score frame_evidence higher than "
            "preservation_evidence within the same pair_id using a margin ranking loss. "
            "Requires --use-frame-violation-loss and --pair-contrastive-frame-data. "
            "Does NOT modify output['logits'] or affect predictions."
        ),
    )
    parser.add_argument(
        "--pair-contrastive-frame-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the pair contrastive frame ranking loss added to total training loss. "
            "Default 0.0 (disabled). Requires --use-pair-contrastive-frame-loss."
        ),
    )
    parser.add_argument(
        "--pair-contrastive-frame-margin",
        type=float,
        default=0.2,
        help=(
            "Margin for the pair contrastive ranking loss: "
            "relu(margin - (frame_fvl - pres_fvl)).mean(). "
            "Default 0.2. A positive margin enforces a minimum separation between "
            "frame_violation_logit(frame_evidence) and frame_violation_logit(pres_evidence)."
        ),
    )
    parser.add_argument(
        "--pair-contrastive-use-case",
        choices=(
            "frame_violation_contrastive",
            "support_safe_frame_contrastive",
            "ood_matched",
            "ood_matched_surface",
            "ood_matched_temporal",
            "all",
        ),
        default="frame_violation_contrastive",
        help=(
            "Filter pair contrastive records by schema and use_case. "
            "A4b2 schema filters (contrastive_use_case field): "
            "  frame_violation_contrastive: frame valid + pres non-frame (may be REFUTE); "
            "  support_safe_frame_contrastive: strict subset where pres is SUPPORT-safe. "
            "A4d OOD-matched schema filters (target/source/preservation_construction_type): "
            "  ood_matched: all A4d OOD-matched records; "
            "  ood_matched_surface: A4d records with surface_like_preservation only; "
            "  ood_matched_temporal: A4d records with temporal_erased_like_preservation only. "
            "all: all records from either schema. "
            "Default: frame_violation_contrastive."
        ),
    )
    # Predicate isolation head
    parser.add_argument(
        "--use-predicate-isolation-loss",
        action="store_true",
        default=False,
        help=(
            "Enable predicate isolation diagnostic head. "
            "Adds a linear head on predicate_pair_repr only (not frame or sufficiency repr), "
            "trained to distinguish predicate-noncoverage (predicate_swap; label=1) from "
            "predicate-covered (none, paraphrase; label=0) using BCE loss. "
            "All other intervention types are excluded from the loss (masked). "
            "Keeps predicate-noncoverage supervision separate from FrameGate and from the "
            "existing V5 predicate_loss (which uses per-record predicate_covered_label). "
            "Does NOT modify output['logits'] or affect predictions. "
            "Stage15 OOD records are never used."
        ),
    )
    parser.add_argument(
        "--predicate-isolation-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the predicate isolation BCE loss added to the total training loss. "
            "Default 0.0 means the head is built but its loss is not included. "
            "Requires --use-predicate-isolation-loss. Suggested starting value: 0.1."
        ),
    )
    parser.add_argument(
        "--predicate-isolation-loss-pos-weight",
        type=float,
        default=2.0,
        help=(
            "Positive class weight for the predicate isolation BCE loss (pos_weight in "
            "F.binary_cross_entropy_with_logits). Compensates for class imbalance: "
            "negative (none+paraphrase=600) vs positive (predicate_swap=300). "
            "Default 2.0 = 600/300."
        ),
    )
    # Preservation entitlement head
    parser.add_argument(
        "--use-preservation-entitlement-loss",
        action="store_true",
        default=False,
        help=(
            "Enable preservation entitlement diagnostic head. "
            "Adds a linear head on sufficiency_repr only (the entitlement-level gate output "
            "combining frame+predicate info), trained to distinguish preservation-entitled "
            "records (none, paraphrase; label=1) from narrow frame-rejection records "
            "(entity_swap, event_swap, location_swap, role_swap, title_name_swap; label=0). "
            "Excluded (masked): predicate_swap (predicate path), evidence-sufficiency types "
            "(evidence_deletion, evidence_truncation, irrelevant_evidence), polarity_flip. "
            "Probes the sufficiency_repr space specifically for the preservation/rejection "
            "distinction, distinct from boundary_head (full concat) and "
            "predicate_isolation_head (predicate_pair_repr). "
            "Does NOT modify output['logits'] or affect predictions. "
            "Stage15 OOD records are never used."
        ),
    )
    parser.add_argument(
        "--preservation-entitlement-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the preservation entitlement BCE loss added to the total training loss. "
            "Default 0.0 means the head is built but its loss is not included. "
            "Requires --use-preservation-entitlement-loss. Suggested starting value: 0.1."
        ),
    )
    parser.add_argument(
        "--preservation-entitlement-loss-pos-weight",
        type=float,
        default=1.0,
        help=(
            "Positive class weight for the preservation entitlement BCE loss (pos_weight in "
            "F.binary_cross_entropy_with_logits). True class ratio is neg/pos = 2.5 "
            "(1500 frame-rejection negatives vs 600 preservation-entitled positives); "
            "default 1.0 is conservative and may need tuning upward."
        ),
    )
    # Temporal diagnostic head
    parser.add_argument(
        "--temporal-diagnostic-data",
        type=str,
        default=None,
        help=(
            "Path to temporal diagnostic JSONL "
            "(e.g. data/temporal_diagnostic_v1_from_controlled_v5_v3.jsonl). "
            "Must be built by scripts/make_temporal_diagnostic_from_controlled.py "
            "from controlled_v5_v3.jsonl. Stage15 OOD records must NOT be present. "
            "Must not be the same as the main --data argument. "
            "Required when --use-temporal-diagnostic-loss is set."
        ),
    )
    parser.add_argument(
        "--use-temporal-diagnostic-loss",
        action="store_true",
        default=False,
        help=(
            "Enable temporal diagnostic head. "
            "Adds a linear head on frame_pair_repr (narrowest frame-level representation; "
            "time_swap has primary_failure_type='frame' and frame_compatible_label=0), "
            "trained to distinguish temporal mismatch records (time_swap; label=1) from "
            "temporal control records (none, paraphrase; label=0) using BCE loss. "
            "Loaded from a SEPARATE temporal diagnostic JSONL (--temporal-diagnostic-data) "
            "that must not be mixed into the main clean controlled train/eval classification data. "
            "Stage15 OOD records must not be present in the temporal diagnostic file. "
            "Does NOT modify output['logits'] or affect predictions. "
            "Requires --temporal-diagnostic-data."
        ),
    )
    parser.add_argument(
        "--temporal-diagnostic-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the temporal diagnostic BCE loss added to the total training loss. "
            "Default 0.0 means the head is built but its loss is not included. "
            "Requires --use-temporal-diagnostic-loss and --temporal-diagnostic-data."
        ),
    )
    parser.add_argument(
        "--temporal-diagnostic-loss-pos-weight",
        type=float,
        default=2.0,
        help=(
            "Positive class weight for the temporal diagnostic BCE loss (pos_weight in "
            "F.binary_cross_entropy_with_logits). Compensates for class imbalance: "
            "negative controls (none+paraphrase=600) vs positive (time_swap=300). "
            "Default 2.0 = 600/300."
        ),
    )
    parser.add_argument(
        "--use-td-constrained-selection",
        action="store_true",
        default=False,
        help=(
            "Enable TD-aware constrained checkpoint selection. "
            "An epoch is eligible only when clean-dev paraphrase_preserved pass_rate >= "
            "--td-selection-min-paraphrase-preserved AND dev td_mismatch_recall >= "
            "--td-selection-min-mismatch-recall AND dev td_control_acceptance >= "
            "--td-selection-min-control-acceptance. Among eligible epochs the one with "
            "highest clean-dev final_macro_f1 is selected. Falls back to unconstrained "
            "final_macro_f1 selection if no epoch is eligible. "
            "Never uses Stage15/OOD metrics. Default off."
        ),
    )
    parser.add_argument(
        "--td-selection-min-paraphrase-preserved",
        type=float,
        default=0.80,
        help=(
            "Minimum clean-dev paraphrase_preserved pass_rate required for an epoch to be "
            "eligible under --use-td-constrained-selection. Default 0.80."
        ),
    )
    parser.add_argument(
        "--td-selection-min-mismatch-recall",
        type=float,
        default=0.50,
        help=(
            "Minimum temporal diagnostic dev td_mismatch_recall required for an epoch to be "
            "eligible under --use-td-constrained-selection. Default 0.50."
        ),
    )
    parser.add_argument(
        "--td-selection-min-control-acceptance",
        type=float,
        default=0.80,
        help=(
            "Minimum temporal diagnostic dev td_control_acceptance required for an epoch to be "
            "eligible under --use-td-constrained-selection. Default 0.80."
        ),
    )
    parser.add_argument(
        "--td-selection-use-final-decision",
        action="store_true",
        default=False,
        help=(
            "Extend --use-td-constrained-selection to also require that the final "
            "classification decision satisfies temporal rejection and control preservation "
            "constraints on the TD dev set. Uses output['logits'].argmax() only; adds no loss. "
            "Requires --use-td-constrained-selection. Default off."
        ),
    )
    parser.add_argument(
        "--td-selection-min-final-temporal-rejection",
        type=float,
        default=0.50,
        help=(
            "Minimum td_final_temporal_rejection_rate (frac of temporal_mismatch TD dev records "
            "predicted NOT_ENTITLED) required when --td-selection-use-final-decision is on. "
            "Default 0.50."
        ),
    )
    parser.add_argument(
        "--td-selection-min-final-control-preservation",
        type=float,
        default=0.80,
        help=(
            "Minimum td_final_control_preservation_rate (frac of temporal_control TD dev records "
            "predicted non-NOT_ENTITLED) required when --td-selection-use-final-decision is on. "
            "Default 0.80."
        ),
    )

    # ── Temporal residual adapter ──────────────────────────────────────────────────────────────
    # A 2-layer MLP adapter that absorbs temporal diagnostic supervision without propagating
    # gradients into the shared frame_pair_repr / FrameGate representation.
    # Architecture: Linear(frame_size, frame_size//2) → GELU → Linear(frame_size//2, 1)
    # Default: disabled. When enabled, adapter is trained only via --use-temporal-adapter-loss.
    # Stage15 OOD is eval-only and is NEVER used for adapter loss, calibration, or selection.
    parser.add_argument(
        "--use-temporal-residual-adapter",
        action="store_true",
        default=False,
        help=(
            "Enable the temporal residual adapter branch. Default off. When enabled, adds a "
            "2-layer MLP adapter reading frame_pair_repr (detached by default) to produce "
            "temporal_adapter_logit and temporal_adapter_prob. The adapter absorbs temporal "
            "diagnostic supervision without coupling gradients into shared FrameGate."
        ),
    )
    parser.add_argument(
        "--temporal-adapter-detach-input",
        action="store_true",
        default=True,
        dest="temporal_adapter_detach_input",
        help=(
            "Detach frame_pair_repr before the temporal residual adapter (default: true). "
            "This prevents adapter BCE gradients from propagating into FrameGate. "
            "Disable with --no-temporal-adapter-detach-input for experimentation."
        ),
    )
    parser.add_argument(
        "--no-temporal-adapter-detach-input",
        action="store_false",
        dest="temporal_adapter_detach_input",
        help="Allow adapter gradients to propagate into frame_pair_repr / FrameGate (stage23 failure mode).",
    )
    parser.add_argument(
        "--use-temporal-adapter-loss",
        action="store_true",
        default=False,
        help=(
            "Supervise the temporal residual adapter with BCE loss on temporal diagnostic data. "
            "Requires --use-temporal-residual-adapter and temporal diagnostic data. "
            "Loss is added to total_loss scaled by --temporal-adapter-loss-weight. Default off."
        ),
    )
    parser.add_argument(
        "--temporal-adapter-loss-weight",
        type=float,
        default=0.0,
        help="Weight for temporal adapter BCE loss. Applied only when --use-temporal-adapter-loss is on. Default 0.0.",
    )
    parser.add_argument(
        "--temporal-adapter-loss-pos-weight",
        type=float,
        default=2.0,
        help=(
            "Positive-class weight for temporal adapter BCE loss (time_swap examples). "
            "Same role as --td-loss-pos-weight but scoped to the adapter. Default 2.0."
        ),
    )
    parser.add_argument(
        "--use-temporal-adapter-final-penalty",
        action="store_true",
        default=False,
        help=(
            "Apply a per-example NOT_ENTITLED logit penalty driven by temporal_adapter_prob. "
            "Penalty = sigmoid(adapter_logit).detach() * --temporal-adapter-final-penalty-scale. "
            "No gradient flows back into the adapter from this penalty path. NOT OOD calibration. "
            "Stage15 is never used to set penalty scale. Default off."
        ),
    )
    parser.add_argument(
        "--temporal-adapter-final-penalty-scale",
        type=float,
        default=0.0,
        help=(
            "Scale for per-example NOT_ENTITLED penalty from adapter confidence. "
            "Applied only when --use-temporal-adapter-final-penalty is on. Default 0.0."
        ),
    )

    # ── TemporalChannel V1 (v6C Lean) ────────────────────────────────────────────────────────
    # Default off. Independent temporal channel reading from cat([claim_frame_state,
    # evidence_frame_state]) — NOT frame_pair_repr — to avoid Stage23 gradient coupling.
    # Stage15 OOD is eval-only and is NEVER used for TC loss, calibration, or penalty selection.
    # Gated penalty requires --use-preservation-entitlement-loss (raises clear error otherwise).
    # Cannot be stacked with --use-temporal-adapter-final-penalty (raises clear error).
    parser.add_argument(
        "--use-temporal-channel",
        action="store_true",
        default=False,
        help=(
            "Enable TemporalChannel V1. Default off. Adds a 2-layer MLP reading "
            "cat([claim_frame_state, evidence_frame_state]) — pre-pair-projector slot states "
            "from FrameGate, NOT frame_pair_repr — to produce temporal_channel_logit and "
            "temporal_channel_prob. With --temporal-channel-detach-input (default), TC loss "
            "gradients cannot propagate into FrameGate parameters."
        ),
    )
    parser.add_argument(
        "--temporal-channel-detach-input",
        action="store_true",
        default=True,
        dest="temporal_channel_detach_input",
        help=(
            "Detach cat([claim_frame_state, evidence_frame_state]) before TemporalChannel V1 "
            "(default: true). Prevents TC BCE gradients from propagating into FrameGate. "
            "Disable with --no-temporal-channel-detach-input for experimentation."
        ),
    )
    parser.add_argument(
        "--no-temporal-channel-detach-input",
        action="store_false",
        dest="temporal_channel_detach_input",
        help=(
            "Allow TC gradients to propagate into claim_frame_state / evidence_frame_state / "
            "FrameGate projection. Risks Stage23-style gradient coupling. Use only for ablation."
        ),
    )
    parser.add_argument(
        "--use-temporal-channel-loss",
        action="store_true",
        default=False,
        help=(
            "Supervise TemporalChannel V1 with BCE loss on temporal diagnostic data. "
            "Requires --use-temporal-channel and temporal diagnostic data. "
            "Supervises temporal_channel_logit only; does not touch output['logits']. "
            "Loss scaled by --temporal-channel-loss-weight. Default off."
        ),
    )
    parser.add_argument(
        "--temporal-channel-loss-weight",
        type=float,
        default=0.0,
        help="Weight for TemporalChannel BCE loss. Applied only when --use-temporal-channel-loss is on. Default 0.0.",
    )
    parser.add_argument(
        "--temporal-channel-loss-pos-weight",
        type=float,
        default=1.0,
        help=(
            "Positive-class weight for TemporalChannel BCE loss (time_swap examples). "
            "Default 1.0 (balanced). Increase to weight mismatch recall higher."
        ),
    )
    parser.add_argument(
        "--use-temporal-channel-gated-penalty",
        action="store_true",
        default=False,
        help=(
            "Apply a per-example NOT_ENTITLED penalty gated by preservation entitlement. "
            "Formula: scale * sigmoid(tc_logit).detach() * (1 - pe_prob).detach(). "
            "Fires only when TC detects temporal mismatch AND PE signals non-entitlement. "
            "Requires --use-temporal-channel and --use-preservation-entitlement-loss. "
            "Cannot be combined with --use-temporal-adapter-final-penalty. Default off."
        ),
    )
    parser.add_argument(
        "--temporal-channel-gated-penalty-scale",
        type=float,
        default=0.0,
        help=(
            "Scale for gated per-example NOT_ENTITLED boost from TemporalChannel. "
            "Applied only when --use-temporal-channel-gated-penalty is on. Default 0.0."
        ),
    )

    # ── Stage26-A: v7 Hierarchical Entitlement architecture ──────────────────────────────────
    # Default: architecture=v6b_minimal (full backward compatibility; no v6B behavior changes).
    # Set --architecture v7_hierarchical to use ContraMambaV7Hierarchical instead.
    # v7 flags below are only consulted when architecture==v7_hierarchical; they have no
    # effect on v6B runs and do not change any default v6B behavior.
    parser.add_argument(
        "--architecture",
        choices=("v6b_minimal", "v7_hierarchical"),
        default="v6b_minimal",
        help=(
            "Model architecture to use. Default: v6b_minimal (existing v6B behavior, "
            "fully backward compatible). v7_hierarchical: Stage26-A ContraMambaV7Hierarchical "
            "(new hierarchical entitlement pipeline; requires explicit selection)."
        ),
    )
    parser.add_argument(
        "--v7-disable-frame-channel",
        action="store_true",
        default=False,
        help=(
            "v7 ablation: EntitlementGate sees frame_prob=1.0 (no frame signal). "
            "FrameGate still runs so downstream heads get frame representations. "
            "Only valid when --architecture v7_hierarchical."
        ),
    )
    parser.add_argument(
        "--v7-disable-predicate-channel",
        action="store_true",
        default=False,
        help=(
            "v7 ablation: EntitlementGate sees predicate_prob=1.0 (no predicate signal). "
            "Only valid when --architecture v7_hierarchical."
        ),
    )
    parser.add_argument(
        "--v7-disable-sufficiency-channel",
        action="store_true",
        default=False,
        help=(
            "v7 ablation: EntitlementGate sees sufficiency_prob=1.0 (no sufficiency signal). "
            "Only valid when --architecture v7_hierarchical."
        ),
    )
    parser.add_argument(
        "--v7-disable-temporal-channel",
        action="store_true",
        default=False,
        help=(
            "v7 ablation: TemporalChannelV2 not instantiated; EntitlementGate uses 3-input "
            "MLP instead of 4-input. In Stage26-A, temporal channel trains through CE only. "
            "Only valid when --architecture v7_hierarchical."
        ),
    )
    parser.add_argument(
        "--v7-flat-arbiter",
        action="store_true",
        default=False,
        help=(
            "v7 ablation: EntitlementGate uses explicit product formula (v6B-like) instead of "
            "learned MLP. Only valid when --architecture v7_hierarchical."
        ),
    )
    parser.add_argument(
        "--v7-no-entitlement-polarity-conditioning",
        action="store_true",
        default=False,
        help=(
            "v7 ablation: Final composition ignores entitlement_logit; polarity logits alone "
            "determine SUPPORT/REFUTE; NE uses fixed ne_bias. "
            "Only valid when --architecture v7_hierarchical."
        ),
    )
    parser.add_argument(
        "--v7-no-aux-losses",
        action="store_true",
        default=False,
        help=(
            "v7: Disable all v7 auxiliary losses. Stage26-A no-op (no v7 aux losses exist yet). "
            "Placeholder for Stage26-B auxiliary loss ablations. "
            "Only valid when --architecture v7_hierarchical."
        ),
    )
    # v7 auxiliary loss flags — all off by default; no v7 aux losses active in Stage26-A.
    # These are reserved for Stage26-B when channel-level supervision targets are defined.
    parser.add_argument("--v7-use-aux-losses", action="store_true", default=False,
        help="v7: Enable v7 auxiliary channel losses. Default off (Stage26-A).")
    parser.add_argument("--v7-frame-loss-weight", type=float, default=0.0,
        help="v7: Frame channel auxiliary BCE loss weight. Default 0.0.")
    parser.add_argument("--v7-predicate-loss-weight", type=float, default=0.0,
        help="v7: Predicate channel auxiliary BCE loss weight. Default 0.0.")
    parser.add_argument("--v7-sufficiency-loss-weight", type=float, default=0.0,
        help="v7: Sufficiency channel auxiliary BCE loss weight. Default 0.0.")
    parser.add_argument("--v7-temporal-loss-weight", type=float, default=0.0,
        help="v7: Temporal channel auxiliary BCE loss weight. Default 0.0.")
    parser.add_argument("--v7-entitlement-loss-weight", type=float, default=0.0,
        help="v7: EntitlementGate auxiliary BCE loss weight. Default 0.0.")

    # ── Stage26-G: v7 stabilization options ──────────────────────────────────────────────────────
    # All off by default. Clean-data supervision only. No Stage15. No OOD. No time_swap.
    parser.add_argument("--v7-use-polarity-margin-loss", action="store_true", default=False,
        help=(
            "v7 Stage26-G: Enable polarity margin loss on v7_polarity_support/refute_logit. "
            "Applied only to gold SUPPORT and REFUTE examples. NOT applied to NOT_ENTITLED. "
            "Pushes support/refute logit separation by at least --v7-polarity-margin. "
            "Default off."
        ),
    )
    parser.add_argument("--v7-polarity-margin-loss-weight", type=float, default=0.0,
        help="v7 Stage26-G: Weight for polarity margin loss. Default 0.0.")
    parser.add_argument("--v7-polarity-margin", type=float, default=0.5,
        help="v7 Stage26-G: Polarity margin for margin loss. Default 0.5.")
    parser.add_argument("--v7-use-entitlement-bce-loss", action="store_true", default=False,
        help=(
            "v7 Stage26-G: Enable entitlement BCE auxiliary loss on v7_entitlement_logit. "
            "Target: entitled=1 for gold SUPPORT and REFUTE, entitled=0 for NOT_ENTITLED. "
            "Clean-data supervision only. No Stage15. No OOD. Default off."
        ),
    )
    parser.add_argument("--v7-entitlement-bce-loss-weight", type=float, default=0.0,
        help="v7 Stage26-G: Weight for entitlement BCE loss. Default 0.0.")
    parser.add_argument("--v7-entitlement-bce-pos-weight", type=float, default=1.0,
        help=(
            "v7 Stage26-G: BCE pos_weight for entitlement BCE loss. "
            "Increase if SUPPORT+REFUTE class is underrepresented. Default 1.0."
        ),
    )
    parser.add_argument("--v7-use-entitled-class-balanced-ce", action="store_true", default=False,
        help=(
            "v7 Stage26-G: Enable auxiliary CE over SUPPORT/REFUTE examples using "
            "v7_polarity_logits (local label: REFUTE=0, SUPPORT=1). "
            "Intended to stabilize polarity direction. NOT_ENTITLED excluded. "
            "Does NOT use output['logits']. Default off."
        ),
    )
    parser.add_argument("--v7-entitled-class-balanced-ce-weight", type=float, default=0.0,
        help="v7 Stage26-G: Weight for entitled class-balanced CE loss. Default 0.0.")
    parser.add_argument("--v7-initial-ne-bias", type=float, default=-0.5,
        help=(
            "v7 Stage26-G: Initial value for the learnable ne_bias parameter in "
            "ContraMambaV7Hierarchical. Default -0.5 (reduces early NOT_ENTITLED collapse "
            "observed in Stage26-F: NE dominated 98%% of predictions at epoch 2). "
            "Use 0.0 to restore Stage26-A behavior."
        ),
    )

    # ── Generic preservation-constrained checkpoint selection ─────────────────────────────────
    # Default off. Uses ONLY clean dev pairwise checks — no Stage15/OOD, no temporal diagnostic
    # metrics. Compatible with baseline, TD, TA, PE, and future runs. Cannot be combined with
    # --use-td-constrained-selection (raises a clear error).
    parser.add_argument(
        "--use-preservation-constrained-selection",
        action="store_true",
        default=False,
        help=(
            "Enable generic preservation-constrained checkpoint selection. Default off. "
            "Among all epochs, select the one with highest clean-dev final_macro_f1 that also "
            "satisfies paraphrase-preserved and predicate-disentangled pass-rate thresholds "
            "(from clean dev pairwise checks only — Stage15/OOD is never consulted). "
            "Falls back to unconstrained final_macro_f1 if no epoch is eligible. "
            "Cannot be combined with --use-td-constrained-selection."
        ),
    )
    parser.add_argument(
        "--selection-min-paraphrase-preserved",
        type=float,
        default=0.70,
        help=(
            "Minimum clean-dev paraphrase_preserved pass_rate required for an epoch to be "
            "eligible under --use-preservation-constrained-selection. Default 0.70."
        ),
    )
    parser.add_argument(
        "--selection-min-predicate-disentangled",
        type=float,
        default=0.85,
        help=(
            "Minimum clean-dev predicate_disentangled pass_rate required for an epoch to be "
            "eligible under --use-preservation-constrained-selection. Default 0.85."
        ),
    )
    parser.add_argument(
        "--selection-fallback",
        type=str,
        default="final_macro_f1",
        help=(
            "Fallback scoring metric used if no epoch satisfies preservation constraints "
            "under --use-preservation-constrained-selection. Default 'final_macro_f1'."
        ),
    )

    return parser


def load_ood_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prediction_distribution_from_records(records: list[dict]) -> dict[str, int]:
    """Compute prediction distribution from exported prediction records."""
    from collections import Counter
    predictions = [record.get("pred_final_label") for record in records]
    return dict(sorted(Counter(predictions).items()))


# ---------------------------------------------------------------------------
# Stage26-F extended: v7 collapse and logit diagnostic helpers
# ---------------------------------------------------------------------------
# These are pure-compute helpers: no training, no optimizer, no loss.
# All operate on already-computed, detached CPU tensors.
# ---------------------------------------------------------------------------

_V7_DIAG_CAPTURE_KEYS: tuple[str, ...] = (
    "logits",
    "predictions",
    "v7_entitlement_logit",
    "v7_entitlement_prob",
    "v7_polarity_support_logit",
    "v7_polarity_refute_logit",
    "v7_temporal_prob",
    "v7_frame_prob",
    "v7_predicate_prob",
    "v7_sufficiency_prob",
    "v7_final_logit_composition",
)


def _v7_capture_dev_output(out: "dict[str, Any]") -> "dict[str, Any]":
    """Clone v7-relevant tensors to CPU for post-epoch diagnostics."""
    captured: dict[str, Any] = {}
    for k in _V7_DIAG_CAPTURE_KEYS:
        v = out.get(k)
        if v is not None:
            captured[k] = v.detach().cpu().clone() if hasattr(v, "detach") else v
    return captured


def _v7_tensor_stats(t: "torch.Tensor") -> "dict[str, float]":
    """Return mean/std/min/max for a 1-D or batched tensor."""
    tf = t.detach().cpu().float()
    n = tf.numel()
    return {
        "mean": float(tf.mean().item()),
        "std": float(tf.std().item()) if n > 1 else 0.0,
        "min": float(tf.min().item()),
        "max": float(tf.max().item()),
    }


def _v7_make_logit_summary(out: "dict[str, Any] | None") -> "dict[str, Any] | None":
    """Build per-tensor stats for v7 logit/prob keys from a captured output dict.

    Covers: v7_entitlement_logit/prob, v7_polarity_support/refute_logit, v7_temporal_prob,
    v7_frame/predicate/sufficiency_prob, per-class logits, and derived margins.
    """
    if out is None:
        return None
    result: dict[str, Any] = {}

    for key in (
        "v7_entitlement_logit",
        "v7_entitlement_prob",
        "v7_polarity_support_logit",
        "v7_polarity_refute_logit",
        "v7_temporal_prob",
        "v7_frame_prob",
        "v7_predicate_prob",
        "v7_sufficiency_prob",
    ):
        val = out.get(key)
        if val is not None and hasattr(val, "mean"):
            result[key] = _v7_tensor_stats(val)

    logits = out.get("logits")
    if (
        logits is not None
        and hasattr(logits, "shape")
        and len(logits.shape) >= 2
        and logits.shape[-1] == 3
    ):
        result["logits_refute"] = _v7_tensor_stats(logits[:, 0])
        result["logits_not_entitled"] = _v7_tensor_stats(logits[:, 1])
        result["logits_support"] = _v7_tensor_stats(logits[:, 2])
        # Derived margins (class order: REFUTE=0 NE=1 SUPPORT=2)
        lf = logits.detach().cpu().float()
        ne, sup, ref = lf[:, 1], lf[:, 2], lf[:, 0]
        result["ne_minus_support_mean"] = float((ne - sup).mean().item())
        result["ne_minus_refute_mean"] = float((ne - ref).mean().item())
        result["support_minus_refute_mean"] = float((sup - ref).mean().item())

    ent = out.get("v7_entitlement_logit")
    if ent is not None and hasattr(ent, "mean"):
        ef = ent.detach().cpu().float()
        result["entitlement_logit_mean"] = float(ef.mean().item())
        result["entitlement_logit_std"] = float(ef.std().item()) if ef.numel() > 1 else 0.0

    return result


def _v7_make_per_gold_summary(
    out: "dict[str, Any] | None",
    final_labels: "torch.Tensor | None",
) -> "dict[str, Any] | None":
    """Per-gold-label breakdown of v7 channel probs and logits.

    `out` must be a CPU-tensor dict (from _v7_capture_dev_output).
    `final_labels` may be on any device; moved to CPU here.
    Label order: REFUTE=0, NOT_ENTITLED=1, SUPPORT=2 (FinalLabel in labels.py).
    """
    if out is None or final_labels is None:
        return None
    logits = out.get("logits")
    if logits is None:
        return None

    labels_cpu = final_labels.detach().cpu()
    logits_cpu = logits.detach().cpu().float()
    _LNAMES = ("REFUTE", "NOT_ENTITLED", "SUPPORT")

    result: dict[str, Any] = {}
    for label_id, label_name in enumerate(_LNAMES):
        mask = labels_cpu == label_id
        count = int(mask.sum().item())
        entry: dict[str, Any] = {"count": count}

        if count > 0:
            preds = out.get("predictions")
            if preds is not None and hasattr(preds, "__len__"):
                preds_cpu = preds.detach().cpu() if hasattr(preds, "detach") else preds
                dist: dict[str, int] = {}
                for p_id, p_name in enumerate(_LNAMES):
                    n = int((preds_cpu[mask] == p_id).sum().item())
                    if n:
                        dist[p_name] = n
                entry["prediction_distribution"] = dist

            for field, key in (
                ("mean_entitlement_prob", "v7_entitlement_prob"),
                ("mean_frame_prob", "v7_frame_prob"),
                ("mean_predicate_prob", "v7_predicate_prob"),
                ("mean_sufficiency_prob", "v7_sufficiency_prob"),
                ("mean_temporal_prob", "v7_temporal_prob"),
                ("mean_polarity_support_logit", "v7_polarity_support_logit"),
                ("mean_polarity_refute_logit", "v7_polarity_refute_logit"),
            ):
                val = out.get(key)
                if val is not None and hasattr(val, "detach"):
                    vc = val.detach().cpu().float()
                    entry[field] = float(vc[mask].mean().item())

            entry["mean_logit_refute"] = float(logits_cpu[mask, 0].mean().item())
            entry["mean_logit_not_entitled"] = float(logits_cpu[mask, 1].mean().item())
            entry["mean_logit_support"] = float(logits_cpu[mask, 2].mean().item())

        result[label_name] = entry

    return result


# ---------------------------------------------------------------------------
# Stage26-D: report schema aliases
# ---------------------------------------------------------------------------
# Root-level aliases make v7 architecture metadata and dev metric scalars
# directly accessible without navigating nested dicts.  This helper is called
# once, immediately before the final JSON is printed/written.
#
# Rules:
#   - Only adds a key if it is absent from root (setdefault semantics).
#   - Works for both v6B and v7 runs.
#   - Uses .get() throughout so unusual report paths cannot crash.
# ---------------------------------------------------------------------------
_LIFT_CONFIG_KEYS: tuple[str, ...] = (
    # Model version / architecture identity
    "model_version",
    "architecture",
    "use_v7_hierarchical",
    # v7 composition / output-contract fields
    "v7_final_logit_composition",
    "v7_channel_output_keys",
    "v7_aux_losses_active",
    # v7 ablation flags
    "v7_disable_frame_channel",
    "v7_disable_predicate_channel",
    "v7_disable_sufficiency_channel",
    "v7_disable_temporal_channel",
    "v7_flat_arbiter",
    "v7_no_entitlement_polarity_conditioning",
    "v7_no_aux_losses",
    # Stage26-G: v7 stabilization option flags and hyperparameters
    "v7_use_polarity_margin_loss",
    "v7_polarity_margin_loss_weight",
    "v7_polarity_margin",
    "v7_use_entitlement_bce_loss",
    "v7_entitlement_bce_loss_weight",
    "v7_entitlement_bce_pos_weight",
    "v7_use_entitled_class_balanced_ce",
    "v7_entitled_class_balanced_ce_weight",
    "v7_initial_ne_bias",
    # v7 Stage15 / time_swap provenance (also lifted from audit_ledger elsewhere;
    # this covers the configuration copy when the ledger path is absent)
    "stage15_used_for_v7_training",
    "stage15_used_for_v7_selection",
    "stage15_used_for_v7_aux_loss_targets",
    "time_swap_used_in_v7_main_clean_data",
)


def lift_report_aliases(report: dict[str, Any]) -> None:
    """Lift v7 architecture metadata and dev metric aliases to report root.

    Mutates `report` in place.  Existing root values are never overwritten.
    Safe to call on both v6B and v7 reports.
    """
    config: dict[str, Any] = report.get("configuration") or {}

    # Lift architecture / v7 fields from configuration to root
    for key in _LIFT_CONFIG_KEYS:
        if key not in report and key in config:
            report[key] = config[key]

    # Lift dev metric scalars as convenient root-level aliases
    dev_metrics: dict[str, Any] = report.get("best_dev_metrics") or {}
    if "best_dev_acc" not in report:
        _acc = dev_metrics.get("final_accuracy")
        if _acc is not None:
            report["best_dev_acc"] = _acc
    if "best_dev_macro_f1" not in report:
        _mf1 = dev_metrics.get("final_macro_f1")
        if _mf1 is not None:
            report["best_dev_macro_f1"] = _mf1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ---------------------------------------------------------------------------
    # Fail-fast: dummy backbone guard
    # Dummy backbone is valid ONLY for explicit smoke/plumbing diagnostics.
    # It has no text comprehension capacity; metrics it produces are not
    # claim-worthy. Require --allow-dummy-backbone to proceed.
    # ---------------------------------------------------------------------------
    if args.backbone == "dummy" and not args.allow_dummy_backbone:
        raise ValueError(
            "[DUMMY BACKBONE BLOCKED] backbone=dummy is permitted only for explicit "
            "smoke/plumbing validation.\n"
            "  - For a claim-worthy experiment use: --backbone mamba\n"
            "  - For intentional dummy smoke/plumbing: add --allow-dummy-backbone\n"
            "Dummy results are NOT model performance evidence and must not be cited "
            "as such in papers or Kaggle submissions."
        )
    if args.backbone == "dummy" and args.allow_dummy_backbone:
        print(
            "[DUMMY BACKBONE WARNING] backbone=dummy is active. "
            "This run is NOT claim-worthy. "
            "Results are valid for plumbing/smoke validation only."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    # Smoke mode overrides
    if args.smoke:
        args.epochs = 2
        args.max_train_records = 16
        print("[SMOKE MODE] epochs=2, max_train_records=16")

    records = v5.load_jsonl(args.data)
    if args.max_train_records is not None:
        records = records[: args.max_train_records]

    train_records, dev_records = v5.split_by_pair_id(
        records, dev_ratio=args.dev_ratio, seed=args.seed
    )

    ce_class_weights = compute_class_weights_v6b(train_records, args.class_weighting, device)
    label_counts: dict[str, int] = {name: 0 for name in v5.ID_TO_FINAL_LABEL.values()}
    for _record in train_records:
        label_counts[_record["final_label"]] += 1

    if args.backbone == "dummy":
        vocab = v5.build_vocab(records)
        train_bundle = v5.encode_records(train_records, vocab)
        dev_bundle = v5.encode_records(dev_records, vocab)
        model = None
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Mamba tokenizer has neither pad_token nor eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        train_bundle = v5.encode_mamba_records(train_records, tokenizer, args.max_length)
        dev_bundle = v5.encode_mamba_records(dev_records, tokenizer, args.max_length)
        if args.architecture == "v7_hierarchical":
            model = build_v7_mamba_model(
                args.model_name,
                freeze_encoder=args.freeze_encoder,
                freeze_a_log=args.freeze_a_log,
                v7_disable_frame_channel=args.v7_disable_frame_channel,
                v7_disable_predicate_channel=args.v7_disable_predicate_channel,
                v7_disable_sufficiency_channel=args.v7_disable_sufficiency_channel,
                v7_disable_temporal_channel=args.v7_disable_temporal_channel,
                v7_flat_arbiter=args.v7_flat_arbiter,
                v7_no_entitlement_polarity_conditioning=args.v7_no_entitlement_polarity_conditioning,
                v7_no_aux_losses=args.v7_no_aux_losses,
                v7_initial_ne_bias=args.v7_initial_ne_bias,
            )
        else:
            model = build_mamba_model(
                args.model_name,
                freeze_encoder=args.freeze_encoder,
                freeze_a_log=args.freeze_a_log,
                use_boundary_head=args.use_boundary_loss,
                use_frame_violation_head=args.use_frame_violation_loss,
                use_predicate_isolation_head=args.use_predicate_isolation_loss,
                use_preservation_entitlement_head=args.use_preservation_entitlement_loss,
                use_temporal_diagnostic_head=args.use_temporal_diagnostic_loss,
                use_temporal_residual_adapter=args.use_temporal_residual_adapter,
                temporal_adapter_detach_input=args.temporal_adapter_detach_input,
                use_temporal_channel=args.use_temporal_channel,
                temporal_channel_detach_input=args.temporal_channel_detach_input,
                use_temporal_channel_loss=args.use_temporal_channel_loss,
                temporal_channel_loss_weight=(
                    args.temporal_channel_loss_weight
                    if args.use_temporal_channel_loss else 0.0
                ),
                temporal_channel_loss_pos_weight=args.temporal_channel_loss_pos_weight,
                use_temporal_channel_gated_penalty=args.use_temporal_channel_gated_penalty,
                temporal_channel_gated_penalty_scale=(
                    args.temporal_channel_gated_penalty_scale
                    if args.use_temporal_channel_gated_penalty else 0.0
                ),
            )

    train_inputs = v5.move_inputs(train_bundle["model_inputs"], device)
    dev_inputs = v5.move_inputs(dev_bundle["model_inputs"], device)
    max_length = max(
        train_inputs["input_ids"].shape[1], dev_inputs["input_ids"].shape[1]
    )
    for inputs in (train_inputs, dev_inputs):
        difference = max_length - inputs["input_ids"].shape[1]
        if difference:
            for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                inputs[key] = F.pad(inputs[key], (0, difference), value=0)

    if model is None:
        if args.architecture == "v7_hierarchical":
            model = build_v7_model(
                len(vocab), max_length,
                v7_disable_frame_channel=args.v7_disable_frame_channel,
                v7_disable_predicate_channel=args.v7_disable_predicate_channel,
                v7_disable_sufficiency_channel=args.v7_disable_sufficiency_channel,
                v7_disable_temporal_channel=args.v7_disable_temporal_channel,
                v7_flat_arbiter=args.v7_flat_arbiter,
                v7_no_entitlement_polarity_conditioning=args.v7_no_entitlement_polarity_conditioning,
                v7_no_aux_losses=args.v7_no_aux_losses,
                v7_initial_ne_bias=args.v7_initial_ne_bias,
            )
        else:
            model = build_model(
                len(vocab), max_length,
                use_boundary_head=args.use_boundary_loss,
                use_frame_violation_head=args.use_frame_violation_loss,
                use_predicate_isolation_head=args.use_predicate_isolation_loss,
                use_preservation_entitlement_head=args.use_preservation_entitlement_loss,
                use_temporal_diagnostic_head=args.use_temporal_diagnostic_loss,
                use_temporal_residual_adapter=args.use_temporal_residual_adapter,
                temporal_adapter_detach_input=args.temporal_adapter_detach_input,
                use_temporal_channel=args.use_temporal_channel,
                temporal_channel_detach_input=args.temporal_channel_detach_input,
                use_temporal_channel_loss=args.use_temporal_channel_loss,
                temporal_channel_loss_weight=(
                    args.temporal_channel_loss_weight
                    if args.use_temporal_channel_loss else 0.0
                ),
                temporal_channel_loss_pos_weight=args.temporal_channel_loss_pos_weight,
                use_temporal_channel_gated_penalty=args.use_temporal_channel_gated_penalty,
                temporal_channel_gated_penalty_scale=(
                    args.temporal_channel_gated_penalty_scale
                    if args.use_temporal_channel_gated_penalty else 0.0
                ),
            )
    model = model.to(device)

    # Stage26-C TODO: add a one-shot contract check for v7 before the training loop.
    # Call validate_v7_output_contract(output) after the first model forward to catch
    # missing keys early.  Not wired here to avoid importing v7 code on v6B runs.
    # Example (in a test or a separate validation script):
    #   from contramamba.modeling_v7_hierarchical import validate_v7_output_contract
    #   with torch.no_grad(): out = model(**model_feature_inputs(dev_inputs))
    #   validate_v7_output_contract(out)

    if args.backbone == "mamba" and args.freeze_encoder:
        print("Caching frozen Mamba token states for train/dev...")
        v5.cache_frozen_encoder_states(model, train_inputs)
        v5.cache_frozen_encoder_states(model, dev_inputs)

    print(
        f"controlled {args.architecture} | backbone={args.backbone} "
        f"train={len(train_records)} dev={len(dev_records)} "
        f"flag_source={args.flag_source} freeze_encoder={args.freeze_encoder}"
    )

    # Extract flags aligned to train/dev records
    train_temporal_flags, train_predicate_flags = extract_flags(
        train_records, args.flag_source, device
    )
    dev_temporal_flags, dev_predicate_flags = extract_flags(
        dev_records, args.flag_source, device
    )

    # Stage22-A: boundary labels derived from intervention_type (only used when head is active)
    train_boundary_labels, train_boundary_mask = encode_boundary_labels(train_records, device)
    dev_boundary_labels, dev_boundary_mask = encode_boundary_labels(dev_records, device)
    _train_b_pos = int(train_boundary_mask.sum().item())
    _train_b_valid = int(train_boundary_mask.sum().item())
    if args.use_boundary_loss:
        _train_b_valid = int(train_boundary_mask.sum().item())
        _train_b_pos_only = int((train_boundary_labels * train_boundary_mask.float()).sum().item())
        print(
            f"[boundary22a] enabled weight={args.boundary_loss_weight}"
            f" pos_weight={args.boundary_loss_pos_weight}"
            f" train_valid={_train_b_valid}"
            f" train_pos={_train_b_pos_only}"
        )

    # Stage22-A3: frame violation labels derived from intervention_type
    train_fv_labels, train_fv_mask = encode_frame_violation_labels(train_records, device)
    dev_fv_labels, dev_fv_mask = encode_frame_violation_labels(dev_records, device)
    if args.use_frame_violation_loss:
        _train_fv_valid = int(train_fv_mask.sum().item())
        _train_fv_pos_only = int((train_fv_labels * train_fv_mask.float()).sum().item())
        print(
            f"[fv22a3] enabled weight={args.frame_violation_loss_weight}"
            f" pos_weight={args.frame_violation_loss_pos_weight}"
            f" train_valid={_train_fv_valid}"
            f" train_pos={_train_fv_pos_only}"
        )

    # Predicate isolation labels derived from intervention_type (only used when head is active)
    train_pi_labels, train_pi_mask = encode_predicate_isolation_labels(train_records, device)
    dev_pi_labels, dev_pi_mask = encode_predicate_isolation_labels(dev_records, device)
    if args.use_predicate_isolation_loss:
        _train_pi_valid = int(train_pi_mask.sum().item())
        _train_pi_pos_only = int((train_pi_labels * train_pi_mask.float()).sum().item())
        print(
            f"[pi_head] enabled weight={args.predicate_isolation_loss_weight}"
            f" pos_weight={args.predicate_isolation_loss_pos_weight}"
            f" train_valid={_train_pi_valid}"
            f" train_pos={_train_pi_pos_only}"
        )

    # Preservation entitlement labels derived from intervention_type
    train_pe_labels, train_pe_mask = encode_preservation_entitlement_labels(
        train_records, device
    )
    dev_pe_labels, dev_pe_mask = encode_preservation_entitlement_labels(dev_records, device)
    if args.use_preservation_entitlement_loss:
        _train_pe_valid = int(train_pe_mask.sum().item())
        _train_pe_pos_only = int(
            (train_pe_labels * train_pe_mask.float()).sum().item()
        )
        print(
            f"[pe_head] enabled weight={args.preservation_entitlement_loss_weight}"
            f" pos_weight={args.preservation_entitlement_loss_pos_weight}"
            f" train_valid={_train_pe_valid}"
            f" train_pos={_train_pe_pos_only}"
        )

    # Temporal diagnostic data loading — separate dataset, never mixed into main train/dev.
    # Records may include time_swap; they must not be part of the main classification CE.
    _td_train_records: list[dict] = []
    _td_dev_records: list[dict] = []
    _td_train_inputs: "dict[str, torch.Tensor] | None" = None
    _td_dev_inputs: "dict[str, torch.Tensor] | None" = None
    _td_train_labels: "torch.Tensor | None" = None
    _td_train_mask: "torch.Tensor | None" = None
    _td_dev_labels: "torch.Tensor | None" = None
    _td_dev_mask: "torch.Tensor | None" = None

    if (
        args.use_temporal_diagnostic_loss
        or args.use_temporal_channel_loss
        or args.use_temporal_adapter_loss
        or args.temporal_diagnostic_data is not None
    ):
        if args.temporal_diagnostic_data is None:
            _td_need = [
                f
                for f, v in [
                    ("--use-temporal-diagnostic-loss", args.use_temporal_diagnostic_loss),
                    ("--use-temporal-channel-loss", args.use_temporal_channel_loss),
                    ("--use-temporal-adapter-loss", args.use_temporal_adapter_loss),
                ]
                if v
            ]
            raise ValueError(
                f"{', '.join(_td_need)} require --temporal-diagnostic-data. "
                "Provide the temporal diagnostic JSONL (time_swap records) via "
                "--temporal-diagnostic-data. Do not use --data; temporal diagnostic "
                "records must not be merged into the main clean train/eval data."
            )
        _td_path = Path(args.temporal_diagnostic_data)
        if not _td_path.exists():
            raise FileNotFoundError(
                f"[td_head] --temporal-diagnostic-data not found: {_td_path}"
            )
        if _td_path.resolve() == Path(args.data).resolve():
            raise ValueError(
                "[td_head] --temporal-diagnostic-data must not be the same as --data.\n"
                "Temporal diagnostic records (including time_swap) must not be mixed "
                "into the main clean controlled train/eval classification data."
            )
        _td_all_records = load_temporal_diagnostic_jsonl(_td_path)
        _td_train_records, _td_dev_records = v5.split_by_pair_id(
            _td_all_records, dev_ratio=args.dev_ratio, seed=args.seed
        )
        _td_train_labels, _td_train_mask = encode_temporal_diagnostic_labels(
            _td_train_records, device
        )
        _td_dev_labels, _td_dev_mask = encode_temporal_diagnostic_labels(
            _td_dev_records, device
        )
        if args.backbone == "dummy":
            _td_train_bundle = v5.encode_records(_td_train_records, vocab)
            _td_dev_bundle = v5.encode_records(_td_dev_records, vocab)
        else:
            _td_train_bundle = v5.encode_mamba_records(
                _td_train_records, tokenizer, args.max_length
            )
            _td_dev_bundle = v5.encode_mamba_records(
                _td_dev_records, tokenizer, args.max_length
            )
        _td_train_inputs = v5.move_inputs(_td_train_bundle["model_inputs"], device)
        _td_dev_inputs = v5.move_inputs(_td_dev_bundle["model_inputs"], device)
        for _td_inp in (_td_train_inputs, _td_dev_inputs):
            _td_seq = _td_inp["input_ids"].shape[1]
            if _td_seq < max_length:
                _diff = max_length - _td_seq
                for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _td_inp[_key] = F.pad(_td_inp[_key], (0, _diff), value=0)
            elif _td_seq > max_length:
                for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _td_inp[_key] = _td_inp[_key][:, :max_length]
        if args.backbone == "mamba" and args.freeze_encoder:
            print(
                f"Caching frozen Mamba token states for temporal diagnostic "
                f"train/dev (td_train={len(_td_train_records)}"
                f" td_dev={len(_td_dev_records)})..."
            )
            v5.cache_frozen_encoder_states(model, _td_train_inputs)
            v5.cache_frozen_encoder_states(model, _td_dev_inputs)
        _td_train_pos = int(_td_train_labels.sum().item())
        _td_train_neg = int(
            (_td_train_mask.float() - _td_train_labels * _td_train_mask.float()).sum().item()
        )
        print(
            f"[td_head] enabled weight={args.temporal_diagnostic_loss_weight}"
            f" pos_weight={args.temporal_diagnostic_loss_pos_weight}"
            f" td_train={len(_td_train_records)}"
            f" td_dev={len(_td_dev_records)}"
            f" td_train_pos={_td_train_pos}"
            f" td_train_neg={_td_train_neg}"
        )

    # Stage22-A4c/A4e: pair contrastive frame data loading and encoding
    _pc_pair_records: list[dict[str, Any]] = []
    _pc_pres_inputs: "dict[str, torch.Tensor] | None" = None
    _pc_frame_inputs: "dict[str, torch.Tensor] | None" = None
    _pc_pres_type_counts: dict[str, int] = {}
    _pc_frame_type_counts: dict[str, int] = {}
    if (
        args.use_pair_contrastive_frame_loss
        and args.pair_contrastive_frame_data is not None
        and args.use_frame_violation_loss
    ):
        _pc_path = Path(args.pair_contrastive_frame_data)
        if not _pc_path.exists():
            raise FileNotFoundError(
                f"--pair-contrastive-frame-data not found: {_pc_path}"
            )
        _pc_pair_records = load_pair_contrastive_jsonl(
            _pc_path, use_case_filter=args.pair_contrastive_use_case
        )
        # Count by construction type for A4d records; empty for A4b2
        from collections import Counter as _Counter
        _pc_pres_type_counts: dict[str, int] = dict(
            _Counter(r.get("preservation_construction_type", "a4b2_record") for r in _pc_pair_records)
        )
        _pc_frame_type_counts: dict[str, int] = dict(
            _Counter(r.get("frame_construction_type", "a4b2_record") for r in _pc_pair_records)
        )
        print(
            f"[pc22a4e] loaded {len(_pc_pair_records)} pair records"
            f" use_case={args.pair_contrastive_use_case}"
            f" weight={args.pair_contrastive_frame_loss_weight}"
            f" margin={args.pair_contrastive_frame_margin}"
        )
        if _pc_pres_type_counts:
            print(f"  [pc22a4e] pres_types={_pc_pres_type_counts} frame_types={_pc_frame_type_counts}")
        if _pc_pair_records:
            _pc_pres_virtual: list[dict[str, Any]] = []
            _pc_frame_virtual: list[dict[str, Any]] = []
            for _r in _pc_pair_records:
                _pv, _fv = _pair_record_to_virtual_records(_r)
                _pc_pres_virtual.append(_pv)
                _pc_frame_virtual.append(_fv)
            if args.backbone == "dummy":
                _pres_bundle = v5.encode_records(_pc_pres_virtual, vocab)
                _frame_bundle = v5.encode_records(_pc_frame_virtual, vocab)
            else:
                _pres_bundle = v5.encode_mamba_records(
                    _pc_pres_virtual, tokenizer, args.max_length
                )
                _frame_bundle = v5.encode_mamba_records(
                    _pc_frame_virtual, tokenizer, args.max_length
                )
            _pc_pres_inputs = v5.move_inputs(_pres_bundle["model_inputs"], device)
            _pc_frame_inputs = v5.move_inputs(_frame_bundle["model_inputs"], device)
            # Align sequence length to max_length
            for _pc_inp in (_pc_pres_inputs, _pc_frame_inputs):
                _pc_seq = _pc_inp["input_ids"].shape[1]
                if _pc_seq < max_length:
                    _diff = max_length - _pc_seq
                    for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                        _pc_inp[_key] = F.pad(_pc_inp[_key], (0, _diff), value=0)
                elif _pc_seq > max_length:
                    for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                        _pc_inp[_key] = _pc_inp[_key][:, :max_length]

    # Wrap v5 training to accept flags
    original_run_training = v5.run_training

    def run_training_v6b(
        model,
        train_inputs,
        dev_inputs,
        train_records,
        dev_records,
        train_bundle,
        *,
        epochs,
        lr,
        head_lr,
        encoder_lr,
        weighted_label_loss,
        balanced_sampler,
        use_intervention_loss,
        ranking_weight,
        loss_config,
        seed,
        run_name,
        select_metric="final_macro_f1",
        capture_best_trainable_state=False,
        smoke_mode=False,
        ce_class_weights=None,
        train_boundary_labels=None,
        train_boundary_mask=None,
        dev_boundary_labels=None,
        dev_boundary_mask=None,
        boundary_loss_weight=0.0,
        boundary_loss_pos_weight=2.5,
        train_fv_labels=None,
        train_fv_mask=None,
        dev_fv_labels=None,
        dev_fv_mask=None,
        fv_loss_weight=0.0,
        fv_loss_pos_weight=1.2,
        # Stage22-A4c: pair contrastive frame inputs (pre-encoded; None when disabled)
        pc_pres_inputs=None,
        pc_frame_inputs=None,
        pc_loss_weight=0.0,
        pc_margin=0.2,
        pc_valid_count=0,
        train_pi_labels=None,
        train_pi_mask=None,
        dev_pi_labels=None,
        dev_pi_mask=None,
        pi_loss_weight=0.0,
        pi_loss_pos_weight=2.0,
        train_pe_labels=None,
        train_pe_mask=None,
        dev_pe_labels=None,
        dev_pe_mask=None,
        pe_loss_weight=0.0,
        pe_loss_pos_weight=1.0,
        # Temporal diagnostic head: separate batch forward pass (time_swap not in main data)
        td_train_inputs=None,
        td_dev_inputs=None,
        td_train_labels=None,
        td_train_mask=None,
        td_dev_labels=None,
        td_dev_mask=None,
        td_loss_weight=0.0,
        td_loss_pos_weight=2.0,
        # TD-constrained checkpoint selection (default off; never uses Stage15/OOD metrics)
        use_td_constrained_selection=False,
        td_sel_min_paraphrase_preserved=0.80,
        td_sel_min_mismatch_recall=0.50,
        td_sel_min_control_acceptance=0.80,
        use_td_final_decision_selection=False,
        td_sel_min_final_temporal_rejection=0.50,
        td_sel_min_final_control_preservation=0.80,
        # Temporal residual adapter (default off; gradient-isolated from frame_pair_repr by default)
        use_temporal_residual_adapter=False,
        ta_loss_weight=0.0,
        ta_loss_pos_weight=2.0,
        ta_final_penalty_scale=0.0,
        # TemporalChannel V1 (default off; reads cat([claim_frame_state, evidence_frame_state]))
        use_temporal_channel=False,
        tc_loss_weight=0.0,
        tc_loss_pos_weight=1.0,
        use_temporal_channel_gated_penalty=False,
        tc_gated_penalty_scale=0.0,
        # Generic preservation-constrained checkpoint selection (default off; clean dev only)
        use_preservation_constrained_selection=False,
        sel_min_paraphrase_preserved=0.70,
        sel_min_predicate_disentangled=0.85,
        sel_fallback="final_macro_f1",
    ):
        """Modified run_training that passes flags to v6b model."""
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if use_preservation_constrained_selection and use_td_constrained_selection:
            raise ValueError(
                "--use-preservation-constrained-selection and --use-td-constrained-selection "
                "cannot both be enabled in the same run. Enable only one. "
                "Both selectors override best_* after training; combining them would create "
                "ambiguous behavior. Use one or the other."
            )
        if use_temporal_channel_gated_penalty and ta_final_penalty_scale > 0.0:
            raise ValueError(
                "--use-temporal-channel-gated-penalty and --temporal-adapter-final-penalty-scale "
                "cannot both be active in the same run. "
                "Each is a true post-hoc final-logit modifier; stacking two violates the "
                "active-component policy (at most one final-logit modifier per run)."
            )
        if use_temporal_channel_gated_penalty and not use_temporal_channel:
            raise ValueError(
                "--use-temporal-channel-gated-penalty requires --use-temporal-channel. "
                "The gated penalty reads temporal_channel_logit which requires the TC head."
            )
        if tc_loss_weight > 0.0 and not use_temporal_channel:
            raise ValueError(
                "--temporal-channel-loss-weight > 0 requires --use-temporal-channel. "
                "Enable the TC head with --use-temporal-channel."
            )
        if tc_loss_weight > 0.0 and td_train_inputs is None:
            raise ValueError(
                "--temporal-channel-loss-weight > 0 requires temporal diagnostic data. "
                "Pass --temporal-diagnostic-data with the temporal diagnostic JSONL "
                "(time_swap records). Without it, TC BCE cannot be computed."
            )
        if use_temporal_channel_gated_penalty and (
            not hasattr(model, "preservation_entitlement_head")
            or model.preservation_entitlement_head is None
        ):
            raise ValueError(
                "--use-temporal-channel-gated-penalty requires "
                "--use-preservation-entitlement-loss. "
                "The gated penalty formula requires preservation_entitlement_prob; "
                "it cannot fire without the PE head active."
            )
        optimizer = v5.build_optimizer(model, lr, head_lr, encoder_lr)
        sampling_generator = torch.Generator().manual_seed(seed)
        best_epoch = 0
        best_score = float("-inf")
        best_dev_metrics = None
        best_dev_interventions = None
        best_dev_pairwise_checks = None
        best_dev_predictions = None
        best_trainable_state = None
        best_pc_metrics: dict[str, Any] = {}
        best_state: dict[str, torch.Tensor] | None = None
        # Preservation-constrained selection tracking (parallel; never uses Stage15/OOD)
        # Only clean dev pairwise checks (paraphrase_preserved, predicate_disentangled).
        # score = final_macro_f1 among eligible epochs; fallback = unconstrained best_*.
        _pcs_epoch: int = -1
        _pcs_score: float = float("-inf")
        _pcs_state: "dict[str, torch.Tensor] | None" = None
        _pcs_dev_metrics: "dict | None" = None
        _pcs_dev_interventions: "dict | None" = None
        _pcs_dev_pairwise_checks: "dict | None" = None
        _pcs_dev_predictions: "list | None" = None
        _pcs_pc_metrics: dict[str, Any] = {}
        _pcs_paraphrase_preserved: float = float("nan")
        _pcs_predicate_disentangled: float = float("nan")
        _pcs_eligible_count: int = 0
        # TD-constrained selection tracking (parallel to unconstrained; fallback is existing best_*)
        _tc_epoch: int = -1
        _tc_score: float = float("-inf")
        _tc_state: "dict[str, torch.Tensor] | None" = None
        _tc_dev_metrics: "dict | None" = None
        _tc_dev_interventions: "dict | None" = None
        _tc_dev_pairwise_checks: "dict | None" = None
        _tc_dev_predictions: "list | None" = None
        _tc_pc_metrics: dict[str, Any] = {}
        _tc_mismatch_recall: float = float("nan")
        _tc_ctrl_acc: float = float("nan")
        _tc_paraphrase_preserved: float = float("nan")
        _tc_td_final_temporal_rejection: float = float("nan")
        _tc_td_final_control_preservation: float = float("nan")
        _tc_td_final_binary_accuracy: float = float("nan")

        # ── Audit ledger: loss accumulators (reporting only; no effect on training) ──────────────
        # Per-epoch raw (pre-weight) and weighted (actual contribution to total_loss) loss values.
        # Both lists grow one entry per epoch and are indexed by epoch-1 at ledger build time.
        _audit_per_epoch_raw: list[dict] = []
        _audit_per_epoch_weighted: list[dict] = []
        _audit_epoch_count: int = 0

        # ── Stage26-F: v7 epoch diagnostic history ─────────────────────────────────────────────
        # Per-epoch dev metric snapshots stored for post-hoc diagnosis (e.g. label collapse
        # trajectory, channel prob trends).  Reporting only; no effect on training or selection.
        _v7_epoch_history: list[dict[str, Any]] = []

        # Stage26-F extended: captured v7 output tensors for best-epoch logit / per-gold summaries.
        # Updated inside the epoch loop whenever score > best_score.
        # Note: if TD/PCS constrained selection overrides best_epoch after the loop, the logit
        # summary may be from the unconstrained best epoch rather than the checkpointed one.
        # Collapse / recall fields are always from best_dev_metrics (correctly overridden).
        _best_dev_output_v7: "dict[str, Any] | None" = None

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            # CRITICAL: Pass flags to v6b model forward
            _ta_pen = ta_final_penalty_scale if use_temporal_residual_adapter and ta_final_penalty_scale > 0.0 else 0.0
            _tc_pen = tc_gated_penalty_scale if use_temporal_channel_gated_penalty and tc_gated_penalty_scale > 0.0 else 0.0
            output = model(
                **v5.model_feature_inputs(train_inputs),
                temporal_mismatch_flags=train_temporal_flags,
                predicate_mismatch_flags=train_predicate_flags,
                temporal_adapter_final_penalty_scale=_ta_pen,
                temporal_channel_gated_penalty_scale=_tc_pen,
            )

            indices = v5.sample_indices(
                train_inputs["final_labels"], balanced_sampler, sampling_generator
            )
            losses = v5.controlled_losses(
                output, train_inputs, indices,
                False if ce_class_weights is not None else weighted_label_loss,
            )
            if ce_class_weights is not None:
                selected_labels = train_inputs["final_labels"].index_select(0, indices)
                new_label_loss = F.cross_entropy(
                    output["logits"].index_select(0, indices),
                    selected_labels,
                    weight=ce_class_weights,
                )
                non_label_total = losses["total"] - losses["label"]
                losses = dict(losses)
                losses["label"] = new_label_loss
                losses["total"] = non_label_total + new_label_loss

            if use_intervention_loss:
                from contramamba import intervention_pairwise_losses

                # Pairwise losses consume output["logits"] (final logits from v6b)
                pairwise_losses = intervention_pairwise_losses(
                    output,
                    train_bundle["pair_ids"],
                    train_bundle["intervention_types"],
                    train_inputs["final_labels"],
                    **loss_config,
                )
                active_intervention_loss = pairwise_losses["total"]
            else:
                active_intervention_loss = (
                    ranking_weight * v5.intervention_objective(output, train_records)
                )
            total_loss = losses["total"] + active_intervention_loss

            # Stage22-A: boundary head BCE loss (training signal only; output["logits"] unchanged)
            _bdry_loss = torch.tensor(0.0)
            if (
                boundary_loss_weight > 0.0
                and train_boundary_labels is not None
                and train_boundary_mask is not None
                and output.get("boundary_logit") is not None
            ):
                _active_b = train_boundary_mask.bool()
                if torch.any(_active_b):
                    _pos_w = torch.tensor(
                        boundary_loss_pos_weight,
                        dtype=torch.float32,
                        device=output["boundary_logit"].device,
                    )
                    _bdry_loss = F.binary_cross_entropy_with_logits(
                        output["boundary_logit"][_active_b],
                        train_boundary_labels[_active_b],
                        pos_weight=_pos_w,
                    )
                    total_loss = total_loss + boundary_loss_weight * _bdry_loss

            # Stage22-A3: frame violation BCE loss (training signal only; output["logits"] unchanged)
            _fv_loss = torch.tensor(0.0)
            if (
                fv_loss_weight > 0.0
                and train_fv_labels is not None
                and train_fv_mask is not None
                and output.get("frame_violation_logit") is not None
            ):
                _active_fv = train_fv_mask.bool()
                if torch.any(_active_fv):
                    _fv_pos_w = torch.tensor(
                        fv_loss_pos_weight,
                        dtype=torch.float32,
                        device=output["frame_violation_logit"].device,
                    )
                    _fv_loss = F.binary_cross_entropy_with_logits(
                        output["frame_violation_logit"][_active_fv],
                        train_fv_labels[_active_fv],
                        pos_weight=_fv_pos_w,
                    )
                    total_loss = total_loss + fv_loss_weight * _fv_loss

            # Predicate isolation BCE loss (diagnostic only; output["logits"] unchanged)
            # Supervises predicate_noncoverage_logit on predicate_pair_repr only.
            _pi_loss = torch.tensor(0.0)
            if (
                pi_loss_weight > 0.0
                and train_pi_labels is not None
                and train_pi_mask is not None
                and output.get("predicate_noncoverage_logit") is not None
            ):
                _active_pi = train_pi_mask.bool()
                if torch.any(_active_pi):
                    _pi_pos_w = torch.tensor(
                        pi_loss_pos_weight,
                        dtype=torch.float32,
                        device=output["predicate_noncoverage_logit"].device,
                    )
                    _pi_loss = F.binary_cross_entropy_with_logits(
                        output["predicate_noncoverage_logit"][_active_pi],
                        train_pi_labels[_active_pi],
                        pos_weight=_pi_pos_w,
                    )
                    total_loss = total_loss + pi_loss_weight * _pi_loss

            # Preservation entitlement BCE loss (diagnostic only; output["logits"] unchanged)
            # Supervises preservation_entitlement_logit on sufficiency_repr only.
            _pe_loss = torch.tensor(0.0)
            if (
                pe_loss_weight > 0.0
                and train_pe_labels is not None
                and train_pe_mask is not None
                and output.get("preservation_entitlement_logit") is not None
            ):
                _active_pe = train_pe_mask.bool()
                if torch.any(_active_pe):
                    _pe_pos_w = torch.tensor(
                        pe_loss_pos_weight,
                        dtype=torch.float32,
                        device=output["preservation_entitlement_logit"].device,
                    )
                    _pe_loss = F.binary_cross_entropy_with_logits(
                        output["preservation_entitlement_logit"][_active_pe],
                        train_pe_labels[_active_pe],
                        pos_weight=_pe_pos_w,
                    )
                    total_loss = total_loss + pe_loss_weight * _pe_loss

            # Temporal diagnostic BCE loss (diagnostic only; output["logits"] unchanged)
            # Separate forward pass on temporal diagnostic records (which include time_swap).
            # These records are NOT in the main clean train/eval tensors; no classification
            # CE is computed on them here. Only temporal_diagnostic_head is supervised.
            _td_loss = torch.tensor(0.0)
            if (
                td_loss_weight > 0.0
                and td_train_inputs is not None
                and td_train_labels is not None
                and td_train_mask is not None
            ):
                _td_n = td_train_inputs["input_ids"].shape[0]
                _td_zero_t = torch.zeros(_td_n, dtype=torch.float32, device=device)
                _td_zero_p = torch.zeros(_td_n, dtype=torch.float32, device=device)
                _td_train_out = model(
                    **v5.model_feature_inputs(td_train_inputs),
                    temporal_mismatch_flags=_td_zero_t,
                    predicate_mismatch_flags=_td_zero_p,
                )
                if _td_train_out.get("temporal_diagnostic_logit") is not None:
                    _active_td = td_train_mask.bool()
                    if torch.any(_active_td):
                        _td_pos_w = torch.tensor(
                            td_loss_pos_weight,
                            dtype=torch.float32,
                            device=_td_train_out["temporal_diagnostic_logit"].device,
                        )
                        _td_loss = F.binary_cross_entropy_with_logits(
                            _td_train_out["temporal_diagnostic_logit"][_active_td],
                            td_train_labels[_active_td],
                            pos_weight=_td_pos_w,
                        )
                        total_loss = total_loss + td_loss_weight * _td_loss

            # Temporal residual adapter BCE loss (diagnostic only; output["logits"] unchanged)
            # Same temporal diagnostic data as the TD head loss above, but supervises
            # temporal_residual_adapter via temporal_adapter_logit. The adapter reads
            # frame_pair_repr (detached by default) so these gradients cannot reach FrameGate.
            # Stage15 / OOD is eval-only and is NEVER used here.
            # _td_train_out may already exist (from the TD head loss block above); reuse it if
            # available rather than running a second forward pass on the same batch.
            _ta_loss = torch.tensor(0.0)
            if (
                ta_loss_weight > 0.0
                and use_temporal_residual_adapter
                and td_train_inputs is not None
                and td_train_labels is not None
                and td_train_mask is not None
            ):
                # Reuse _td_train_out if the TD head block ran a forward pass this step.
                # Otherwise run a fresh pass (td_loss_weight==0 but adapter loss is enabled).
                _ta_need_forward = not (
                    td_loss_weight > 0.0
                    and td_train_inputs is not None
                    and td_train_labels is not None
                    and td_train_mask is not None
                )
                if _ta_need_forward:
                    _ta_n = td_train_inputs["input_ids"].shape[0]
                    _ta_zero_t = torch.zeros(_ta_n, dtype=torch.float32, device=device)
                    _ta_zero_p = torch.zeros(_ta_n, dtype=torch.float32, device=device)
                    _ta_train_out = model(
                        **v5.model_feature_inputs(td_train_inputs),
                        temporal_mismatch_flags=_ta_zero_t,
                        predicate_mismatch_flags=_ta_zero_p,
                    )
                else:
                    _ta_train_out = _td_train_out
                if _ta_train_out.get("temporal_adapter_logit") is not None:
                    _active_ta = td_train_mask.bool()
                    if torch.any(_active_ta):
                        _ta_pos_w = torch.tensor(
                            ta_loss_pos_weight,
                            dtype=torch.float32,
                            device=_ta_train_out["temporal_adapter_logit"].device,
                        )
                        _ta_loss = F.binary_cross_entropy_with_logits(
                            _ta_train_out["temporal_adapter_logit"][_active_ta],
                            td_train_labels[_active_ta],
                            pos_weight=_ta_pos_w,
                        )
                        total_loss = total_loss + ta_loss_weight * _ta_loss

            # TemporalChannel V1 BCE loss (diagnostic only; output["logits"] unchanged)
            # Same temporal diagnostic data as TD/TA above, but supervises temporal_channel_v1
            # via temporal_channel_logit. The channel reads cat([claim_frame_state,
            # evidence_frame_state]) (pre-pair-projector; NOT frame_pair_repr), detached by
            # default. Stage15 / OOD is eval-only and is NEVER used here.
            # Reuse existing TD-batch forward output if TD or TA block already ran; otherwise
            # run a fresh pass on the temporal diagnostic batch.
            _tc_loss = torch.tensor(0.0)
            _td_ran = (
                td_loss_weight > 0.0
                and td_train_inputs is not None
                and td_train_labels is not None
                and td_train_mask is not None
            )
            _ta_ran = (
                ta_loss_weight > 0.0
                and use_temporal_residual_adapter
                and td_train_inputs is not None
                and td_train_labels is not None
                and td_train_mask is not None
            )
            if (
                tc_loss_weight > 0.0
                and use_temporal_channel
                and td_train_inputs is not None
                and td_train_labels is not None
                and td_train_mask is not None
            ):
                if _td_ran:
                    _tc_train_out = _td_train_out
                elif _ta_ran:
                    _tc_train_out = _ta_train_out
                else:
                    _tc_n = td_train_inputs["input_ids"].shape[0]
                    _tc_zero_t = torch.zeros(_tc_n, dtype=torch.float32, device=device)
                    _tc_zero_p = torch.zeros(_tc_n, dtype=torch.float32, device=device)
                    _tc_train_out = model(
                        **v5.model_feature_inputs(td_train_inputs),
                        temporal_mismatch_flags=_tc_zero_t,
                        predicate_mismatch_flags=_tc_zero_p,
                    )
                if _tc_train_out.get("temporal_channel_logit") is None:
                    raise RuntimeError(
                        "TemporalChannel loss is enabled but model output has no "
                        "temporal_channel_logit. Check --use-temporal-channel and "
                        "model.temporal_channel_v1 initialization."
                    )
                _active_tc = td_train_mask.bool()
                if torch.any(_active_tc):
                    _tc_pos_w = torch.tensor(
                        tc_loss_pos_weight,
                        dtype=torch.float32,
                        device=_tc_train_out["temporal_channel_logit"].device,
                    )
                    _tc_loss = F.binary_cross_entropy_with_logits(
                        _tc_train_out["temporal_channel_logit"][_active_tc],
                        td_train_labels[_active_tc],
                        pos_weight=_tc_pos_w,
                    )
                    total_loss = total_loss + tc_loss_weight * _tc_loss

            # Stage22-A4c: pair contrastive frame ranking loss
            # Supervises frame_violation_logit only; output["logits"] and predictions unchanged.
            _pc_loss = torch.tensor(0.0)
            if (
                pc_loss_weight > 0.0
                and pc_pres_inputs is not None
                and pc_frame_inputs is not None
                and output.get("frame_violation_logit") is not None
            ):
                _pc_n = pc_pres_inputs["input_ids"].shape[0]
                _pc_zero_t = torch.zeros(_pc_n, dtype=torch.float32, device=device)
                _pc_zero_p = torch.zeros(_pc_n, dtype=torch.float32, device=device)
                _pc_pres_out = model(
                    **v5.model_feature_inputs(pc_pres_inputs),
                    temporal_mismatch_flags=_pc_zero_t,
                    predicate_mismatch_flags=_pc_zero_p,
                )
                _pc_frame_out = model(
                    **v5.model_feature_inputs(pc_frame_inputs),
                    temporal_mismatch_flags=_pc_zero_t,
                    predicate_mismatch_flags=_pc_zero_p,
                )
                if (
                    _pc_pres_out.get("frame_violation_logit") is not None
                    and _pc_frame_out.get("frame_violation_logit") is not None
                ):
                    _pres_fvl = _pc_pres_out["frame_violation_logit"].squeeze(-1)
                    _frame_fvl = _pc_frame_out["frame_violation_logit"].squeeze(-1)
                    _margins = _frame_fvl - _pres_fvl
                    _pc_loss = F.relu(pc_margin - _margins).mean()
                    total_loss = total_loss + pc_loss_weight * _pc_loss

            # ── Stage26-G: v7 polarity margin auxiliary loss ──────────────────────────────────
            # Hinge margin loss on v7_polarity_support/refute_logit.
            # Applied only to gold SUPPORT and REFUTE examples; NOT_ENTITLED excluded.
            # Does NOT replace CE. CE continues using output["logits"]. No loss_logits.
            # Clean-data only (main train split). No Stage15. No OOD. No time_swap.
            _v7_pm_loss = torch.tensor(0.0, device=device)
            if (
                args.architecture == "v7_hierarchical"
                and args.v7_use_polarity_margin_loss
                and args.v7_polarity_margin_loss_weight > 0.0
                and not args.v7_no_aux_losses
            ):
                _pm_sup_logit = output.get("v7_polarity_support_logit")
                _pm_ref_logit = output.get("v7_polarity_refute_logit")
                if _pm_sup_logit is not None and _pm_ref_logit is not None:
                    _pm_final_labels = train_inputs["final_labels"]
                    _pm_margin = args.v7_polarity_margin
                    _pm_parts: list[torch.Tensor] = []
                    _sup_mask = _pm_final_labels == 2  # SUPPORT
                    _ref_mask = _pm_final_labels == 0  # REFUTE
                    if _sup_mask.any():
                        # Push (support_logit - refute_logit) >= margin for gold SUPPORT
                        _pm_parts.append(
                            F.relu(_pm_margin - (_pm_sup_logit[_sup_mask] - _pm_ref_logit[_sup_mask]))
                        )
                    if _ref_mask.any():
                        # Push (refute_logit - support_logit) >= margin for gold REFUTE
                        _pm_parts.append(
                            F.relu(_pm_margin - (_pm_ref_logit[_ref_mask] - _pm_sup_logit[_ref_mask]))
                        )
                    if _pm_parts:
                        _v7_pm_loss = torch.cat(_pm_parts).mean()
                        total_loss = total_loss + args.v7_polarity_margin_loss_weight * _v7_pm_loss

            # ── Stage26-G: v7 entitlement BCE auxiliary loss ──────────────────────────────────
            # BCE on v7_entitlement_logit with ground-truth entitled target derived from labels:
            #   entitled=1 for SUPPORT (2) and REFUTE (0) — these require a polarity judgment
            #   entitled=0 for NOT_ENTITLED (1) — evidence fails to entitle a polarity judgment
            # Clean-data only (main train split). No Stage15. No OOD. No time_swap.
            _v7_ent_bce_loss = torch.tensor(0.0, device=device)
            if (
                args.architecture == "v7_hierarchical"
                and args.v7_use_entitlement_bce_loss
                and args.v7_entitlement_bce_loss_weight > 0.0
                and not args.v7_no_aux_losses
            ):
                _ent_logit = output.get("v7_entitlement_logit")
                if _ent_logit is not None:
                    _ent_labels = train_inputs["final_labels"]
                    # SUPPORT=2 and REFUTE=0 → entitled=1; NOT_ENTITLED=1 → entitled=0
                    _ent_target = (_ent_labels != 1).float()
                    _ent_pos_w = torch.tensor(
                        args.v7_entitlement_bce_pos_weight,
                        dtype=_ent_logit.dtype,
                        device=device,
                    )
                    _v7_ent_bce_loss = F.binary_cross_entropy_with_logits(
                        _ent_logit, _ent_target, pos_weight=_ent_pos_w
                    )
                    total_loss = total_loss + args.v7_entitlement_bce_loss_weight * _v7_ent_bce_loss

            # ── Stage26-G: v7 entitled class-balanced CE ──────────────────────────────────────
            # Auxiliary CE over SUPPORT/REFUTE examples only, using v7_polarity_logits [B, 2].
            # Local labels: REFUTE (gold=0) → local 0; SUPPORT (gold=2) → local 1.
            # NOT_ENTITLED examples are excluded entirely.
            # Intended to anchor polarity direction independently of the entitlement gate.
            # Does NOT use output["logits"]. Does NOT create loss_logits.
            _v7_ecb_loss = torch.tensor(0.0, device=device)
            if (
                args.architecture == "v7_hierarchical"
                and args.v7_use_entitled_class_balanced_ce
                and args.v7_entitled_class_balanced_ce_weight > 0.0
                and not args.v7_no_aux_losses
            ):
                _ecb_polarity_logits = output.get("v7_polarity_logits")  # [B, 2]: [refute, support]
                if _ecb_polarity_logits is not None:
                    _ecb_labels = train_inputs["final_labels"]
                    _ecb_mask = _ecb_labels != 1  # True for SUPPORT (2) and REFUTE (0)
                    if _ecb_mask.any():
                        _ecb_pol_sub = _ecb_polarity_logits[_ecb_mask]
                        # REFUTE (gold=0) → local 0; SUPPORT (gold=2) → local 1
                        _ecb_local_labels = (_ecb_labels[_ecb_mask] == 2).long()
                        _v7_ecb_loss = F.cross_entropy(_ecb_pol_sub, _ecb_local_labels)
                        total_loss = total_loss + args.v7_entitled_class_balanced_ce_weight * _v7_ecb_loss

            # ── Audit ledger accumulation (reporting only; does not affect gradients) ─────────
            # raw = loss value before weight multiplication; weighted = actual total_loss contribution
            _ep_ce_raw = float(losses["label"].item())
            _ep_ai_val = (
                float(active_intervention_loss.item())
                if hasattr(active_intervention_loss, "item")
                else float(active_intervention_loss)
            )
            _ep_bdry_raw = float(_bdry_loss.item()) if hasattr(_bdry_loss, "item") else 0.0
            _ep_fv_raw = float(_fv_loss.item()) if hasattr(_fv_loss, "item") else 0.0
            _ep_pc_raw = float(_pc_loss.item()) if hasattr(_pc_loss, "item") else 0.0
            _ep_pi_raw = float(_pi_loss.item()) if hasattr(_pi_loss, "item") else 0.0
            _ep_pe_raw = float(_pe_loss.item()) if hasattr(_pe_loss, "item") else 0.0
            _ep_td_raw = float(_td_loss.item()) if hasattr(_td_loss, "item") else 0.0
            _ep_ta_raw = float(_ta_loss.item()) if hasattr(_ta_loss, "item") else 0.0
            _ep_tc_raw = float(_tc_loss.item()) if hasattr(_tc_loss, "item") else 0.0
            _ep_v7_pm_raw = float(_v7_pm_loss.item()) if hasattr(_v7_pm_loss, "item") else 0.0
            _ep_v7_ent_bce_raw = float(_v7_ent_bce_loss.item()) if hasattr(_v7_ent_bce_loss, "item") else 0.0
            _ep_v7_ecb_raw = float(_v7_ecb_loss.item()) if hasattr(_v7_ecb_loss, "item") else 0.0
            _ep_total = float(total_loss.item())
            # For ranking path: active_intervention_loss = ranking_weight * raw_ranking.
            # Recover raw by dividing. For intervention path: pairwise_losses["total"] is
            # already a weighted composite; report raw == weighted (no scalar unwinding).
            if use_intervention_loss:
                _ep_ranking_raw, _ep_ranking_w = 0.0, 0.0
                _ep_intervention_raw = _ep_ai_val
                _ep_intervention_w = _ep_ai_val
            else:
                _ep_ranking_raw = (_ep_ai_val / ranking_weight) if ranking_weight > 0.0 else 0.0
                _ep_ranking_w = _ep_ai_val  # = ranking_weight * _ep_ranking_raw
                _ep_intervention_raw, _ep_intervention_w = 0.0, 0.0
            _audit_per_epoch_raw.append({
                "ce_loss": _ep_ce_raw,
                "ranking_loss": _ep_ranking_raw,
                "intervention_loss": _ep_intervention_raw,
                "boundary_loss": _ep_bdry_raw,
                "frame_violation_loss": _ep_fv_raw,
                "pair_contrastive_frame_loss": _ep_pc_raw,
                "predicate_isolation_loss": _ep_pi_raw,
                "preservation_entitlement_loss": _ep_pe_raw,
                "temporal_diagnostic_loss": _ep_td_raw,
                "temporal_adapter_loss": _ep_ta_raw,
                "temporal_channel_loss": _ep_tc_raw,
                # Stage26-G: v7 stabilization losses (0.0 when disabled or architecture=v6b_minimal)
                "v7_polarity_margin_loss": _ep_v7_pm_raw,
                "v7_entitlement_bce_loss": _ep_v7_ent_bce_raw,
                "v7_entitled_class_balanced_ce_loss": _ep_v7_ecb_raw,
                "total_loss": _ep_total,
            })
            _audit_per_epoch_weighted.append({
                "ce_loss": _ep_ce_raw,  # CE weight = 1.0
                "ranking_loss": _ep_ranking_w,
                "intervention_loss": _ep_intervention_w,
                "boundary_loss": boundary_loss_weight * _ep_bdry_raw,
                "frame_violation_loss": fv_loss_weight * _ep_fv_raw,
                "pair_contrastive_frame_loss": pc_loss_weight * _ep_pc_raw,
                "predicate_isolation_loss": pi_loss_weight * _ep_pi_raw,
                "preservation_entitlement_loss": pe_loss_weight * _ep_pe_raw,
                "temporal_diagnostic_loss": td_loss_weight * _ep_td_raw,
                "temporal_adapter_loss": ta_loss_weight * _ep_ta_raw,
                "temporal_channel_loss": tc_loss_weight * _ep_tc_raw,
                # Stage26-G: v7 stabilization losses weighted contributions
                "v7_polarity_margin_loss": args.v7_polarity_margin_loss_weight * _ep_v7_pm_raw,
                "v7_entitlement_bce_loss": args.v7_entitlement_bce_loss_weight * _ep_v7_ent_bce_raw,
                "v7_entitled_class_balanced_ce_loss": (
                    args.v7_entitled_class_balanced_ce_weight * _ep_v7_ecb_raw
                ),
                "total_loss": _ep_total,
            })
            _audit_epoch_count += 1
            # ── end audit accumulation ────────────────────────────────────────────────────────

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # Evaluate with flags
            model.eval()
            with torch.no_grad():
                train_output = model(
                    **train_inputs,
                    temporal_mismatch_flags=train_temporal_flags,
                    predicate_mismatch_flags=train_predicate_flags,
                    temporal_adapter_final_penalty_scale=_ta_pen,
                    temporal_channel_gated_penalty_scale=_tc_pen,
                )
                dev_output = model(
                    **dev_inputs,
                    temporal_mismatch_flags=dev_temporal_flags,
                    predicate_mismatch_flags=dev_predicate_flags,
                    temporal_adapter_final_penalty_scale=_ta_pen,
                    temporal_channel_gated_penalty_scale=_tc_pen,
                )
                # Temporal diagnostic eval: separate forward passes on td train/dev records.
                # Batches are kept separate from main train/dev; no classification CE here.
                _ttdm: dict[str, Any] = {}
                _dtdm: dict[str, Any] = {}
                if td_train_inputs is not None and td_dev_inputs is not None:
                    _td_et = td_train_inputs["input_ids"].shape[0]
                    _td_zt = torch.zeros(_td_et, dtype=torch.float32, device=device)
                    _td_zp = torch.zeros(_td_et, dtype=torch.float32, device=device)
                    _td_eval_train_out = model(
                        **v5.model_feature_inputs(td_train_inputs),
                        temporal_mismatch_flags=_td_zt,
                        predicate_mismatch_flags=_td_zp,
                    )
                    _ttdm = compute_temporal_diagnostic_metrics(
                        _td_eval_train_out, td_train_labels, td_train_mask
                    )
                    _td_ed = td_dev_inputs["input_ids"].shape[0]
                    _td_ztd = torch.zeros(_td_ed, dtype=torch.float32, device=device)
                    _td_zpd = torch.zeros(_td_ed, dtype=torch.float32, device=device)
                    _td_eval_dev_out = model(
                        **v5.model_feature_inputs(td_dev_inputs),
                        temporal_mismatch_flags=_td_ztd,
                        predicate_mismatch_flags=_td_zpd,
                    )
                    _dtdm = compute_temporal_diagnostic_metrics(
                        _td_eval_dev_out, td_dev_labels, td_dev_mask
                    )
                    # Final classification decision behavior on TD dev (no new loss)
                    _dtd_fdm = compute_td_final_decision_metrics(
                        _td_eval_dev_out, td_dev_labels, td_dev_mask
                    )
                    # Temporal residual adapter metrics (reuses same eval forward passes)
                    _ttatm = compute_temporal_adapter_metrics(
                        _td_eval_train_out, td_train_labels, td_train_mask
                    )
                    _dtatm = compute_temporal_adapter_metrics(
                        _td_eval_dev_out, td_dev_labels, td_dev_mask
                    )
                else:
                    _dtd_fdm: dict[str, Any] = {}
                    _ttatm: dict[str, Any] = {}
                    _dtatm: dict[str, Any] = {}
            train_metrics = v5.compute_metrics(train_output, train_inputs)
            dev_metrics = v5.compute_metrics(dev_output, dev_inputs)

            # Stage22-A: boundary head metrics
            _tbm = compute_boundary_metrics(train_output, train_boundary_labels, train_boundary_mask)
            _dbm = compute_boundary_metrics(dev_output, dev_boundary_labels, dev_boundary_mask)
            # Stage22-A3: frame violation head metrics
            _tfvm = compute_frame_violation_metrics(train_output, train_fv_labels, train_fv_mask)
            _dfvm = compute_frame_violation_metrics(dev_output, dev_fv_labels, dev_fv_mask)

            # Predicate isolation head metrics
            _tpim = compute_predicate_isolation_metrics(train_output, train_pi_labels, train_pi_mask)
            _dpim = compute_predicate_isolation_metrics(dev_output, dev_pi_labels, dev_pi_mask)
            # Preservation entitlement head metrics
            _tpem = compute_preservation_entitlement_metrics(
                train_output, train_pe_labels, train_pe_mask
            )
            _dpem = compute_preservation_entitlement_metrics(
                dev_output, dev_pe_labels, dev_pe_mask
            )

            # Stage22-A4c: pair contrastive ranking metrics (eval; no_grad already active)
            _pc_eval_metrics: dict[str, Any] = {}
            if (
                pc_pres_inputs is not None
                and pc_frame_inputs is not None
                and output.get("frame_violation_logit") is not None
            ):
                _pc_n = pc_pres_inputs["input_ids"].shape[0]
                _pc_zero_t = torch.zeros(_pc_n, dtype=torch.float32, device=device)
                _pc_zero_p = torch.zeros(_pc_n, dtype=torch.float32, device=device)
                _pc_pres_eval = model(
                    **v5.model_feature_inputs(pc_pres_inputs),
                    temporal_mismatch_flags=_pc_zero_t,
                    predicate_mismatch_flags=_pc_zero_p,
                )
                _pc_frame_eval = model(
                    **v5.model_feature_inputs(pc_frame_inputs),
                    temporal_mismatch_flags=_pc_zero_t,
                    predicate_mismatch_flags=_pc_zero_p,
                )
                _pc_eval_metrics = compute_pair_contrastive_metrics(
                    _pc_pres_eval, _pc_frame_eval, margin=pc_margin
                )

            if select_metric not in dev_metrics or not isinstance(
                dev_metrics[select_metric], (int, float)
            ):
                raise ValueError(f"unsupported select_metric: {select_metric!r}")
            score = float(dev_metrics[select_metric])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_dev_metrics = dev_metrics
                best_dev_interventions = intervention_diagnostics_v6b(
                    dev_records, dev_output
                )
                # Skip pairwise checks in smoke mode (may have incomplete variants)
                if not smoke_mode:
                    best_dev_pairwise_checks = v5.pairwise_checks(dev_records, dev_output)
                best_dev_predictions = prediction_records_v6b(dev_records, dev_output)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_pc_metrics = {
                    **_pc_eval_metrics,
                    "pair_contrastive_use_case": args.pair_contrastive_use_case,
                    "count_by_preservation_construction_type": _pc_pres_type_counts,
                    "count_by_frame_construction_type": _pc_frame_type_counts,
                }
                if capture_best_trainable_state:
                    best_trainable_state = v5.capture_trainable_state(model)
                # Stage26-F extended: snapshot v7 diagnostic tensors from the new best epoch
                if args.architecture == "v7_hierarchical":
                    _best_dev_output_v7 = _v7_capture_dev_output(dev_output)

            # TD-constrained checkpoint selection (parallel, default off, never uses Stage15/OOD)
            if use_td_constrained_selection and not smoke_mode:
                _ep_pw = v5.pairwise_checks(dev_records, dev_output)
                _ep_pp = _ep_pw.get("paraphrase_preserved", {}).get("pass_rate", 0.0)
                _ep_mr = _dtdm.get("td_mismatch_recall", float("nan")) if _dtdm else float("nan")
                _ep_ca = _dtdm.get("td_control_acceptance", float("nan")) if _dtdm else float("nan")
                # NaN means metric absent; treat as failing the constraint
                _ep_mr_v = _ep_mr if _ep_mr == _ep_mr else 0.0
                _ep_ca_v = _ep_ca if _ep_ca == _ep_ca else 0.0
                _ep_eligible = (
                    _ep_pp >= td_sel_min_paraphrase_preserved
                    and _ep_mr_v >= td_sel_min_mismatch_recall
                    and _ep_ca_v >= td_sel_min_control_acceptance
                )
                # Optional final-decision constraints on TD dev
                if _ep_eligible and use_td_final_decision_selection:
                    _ep_ftr = _dtd_fdm.get(
                        "td_final_temporal_rejection_rate", float("nan")
                    ) if _dtd_fdm else float("nan")
                    _ep_fcp = _dtd_fdm.get(
                        "td_final_control_preservation_rate", float("nan")
                    ) if _dtd_fdm else float("nan")
                    _ep_ftr_v = _ep_ftr if _ep_ftr == _ep_ftr else 0.0
                    _ep_fcp_v = _ep_fcp if _ep_fcp == _ep_fcp else 0.0
                    _ep_eligible = (
                        _ep_ftr_v >= td_sel_min_final_temporal_rejection
                        and _ep_fcp_v >= td_sel_min_final_control_preservation
                    )
                else:
                    _ep_ftr = _dtd_fdm.get(
                        "td_final_temporal_rejection_rate", float("nan")
                    ) if _dtd_fdm else float("nan")
                    _ep_fcp = _dtd_fdm.get(
                        "td_final_control_preservation_rate", float("nan")
                    ) if _dtd_fdm else float("nan")
                if _ep_eligible and score > _tc_score:
                    _tc_score = score
                    _tc_epoch = epoch
                    _tc_dev_metrics = dev_metrics
                    _tc_dev_interventions = intervention_diagnostics_v6b(dev_records, dev_output)
                    _tc_dev_pairwise_checks = _ep_pw
                    _tc_dev_predictions = prediction_records_v6b(dev_records, dev_output)
                    _tc_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                    _tc_pc_metrics = {
                        **_pc_eval_metrics,
                        "pair_contrastive_use_case": args.pair_contrastive_use_case,
                        "count_by_preservation_construction_type": _pc_pres_type_counts,
                        "count_by_frame_construction_type": _pc_frame_type_counts,
                    }
                    _tc_mismatch_recall = _ep_mr
                    _tc_ctrl_acc = _ep_ca
                    _tc_paraphrase_preserved = _ep_pp
                    _tc_td_final_temporal_rejection = _ep_ftr
                    _tc_td_final_control_preservation = _ep_fcp
                    _tc_td_final_binary_accuracy = _dtd_fdm.get(
                        "td_final_binary_accuracy", float("nan")
                    ) if _dtd_fdm else float("nan")

            # Generic preservation-constrained checkpoint selection (clean dev only; no Stage15)
            # Uses paraphrase_preserved and predicate_disentangled pass_rate from clean dev
            # pairwise checks. Stage15/OOD is never consulted here.
            if use_preservation_constrained_selection and not smoke_mode:
                _ep_pcs_pw = v5.pairwise_checks(dev_records, dev_output)
                _ep_pcs_pp = _ep_pcs_pw.get("paraphrase_preserved", {}).get("pass_rate", 0.0)
                _ep_pcs_pd = _ep_pcs_pw.get("predicate_disentangled", {}).get("pass_rate", 0.0)
                _ep_pcs_eligible = (
                    _ep_pcs_pp >= sel_min_paraphrase_preserved
                    and _ep_pcs_pd >= sel_min_predicate_disentangled
                )
                if _ep_pcs_eligible:
                    _pcs_eligible_count += 1
                    if score > _pcs_score:
                        _pcs_score = score
                        _pcs_epoch = epoch
                        _pcs_paraphrase_preserved = _ep_pcs_pp
                        _pcs_predicate_disentangled = _ep_pcs_pd
                        _pcs_dev_metrics = dev_metrics
                        _pcs_dev_interventions = intervention_diagnostics_v6b(
                            dev_records, dev_output
                        )
                        _pcs_dev_pairwise_checks = _ep_pcs_pw
                        _pcs_dev_predictions = prediction_records_v6b(dev_records, dev_output)
                        _pcs_state = {
                            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                        }
                        _pcs_pc_metrics = {
                            **_pc_eval_metrics,
                            "pair_contrastive_use_case": args.pair_contrastive_use_case,
                            "count_by_preservation_construction_type": _pc_pres_type_counts,
                            "count_by_frame_construction_type": _pc_frame_type_counts,
                        }

            # ── Stage26-F: record per-epoch dev snapshot ─────────────────────────────────────
            # All values come from dev_metrics (already computed) and dev_output (already
            # available in no_grad context).  No new forward passes.  Reporting only.
            _v7_ep_snap: dict[str, Any] = {
                "epoch": epoch,
                "dev_final_accuracy": dev_metrics.get("final_accuracy"),
                "dev_final_macro_f1": dev_metrics.get("final_macro_f1"),
                "dev_prediction_distribution": dev_metrics.get("prediction_distribution"),
                "dev_frame_accuracy": dev_metrics.get("frame_accuracy"),
                "dev_predicate_accuracy": dev_metrics.get("predicate_accuracy"),
                "dev_sufficiency_accuracy": dev_metrics.get("sufficiency_accuracy"),
                "dev_polarity_accuracy_entitled": dev_metrics.get("polarity_accuracy_entitled"),
            }
            # v7_logit_summary: lightweight channel prob means from dev_output.
            # Only present for v7 runs; zero-cost (tensors already computed).
            if args.architecture == "v7_hierarchical":
                _v7_ep_snap["v7_logit_summary"] = {
                    k: float(dev_output[k].mean().item())
                    for k in (
                        "v7_frame_prob",
                        "v7_predicate_prob",
                        "v7_sufficiency_prob",
                        "v7_entitlement_prob",
                    )
                    if k in dev_output and dev_output[k] is not None
                }
                # Temporal prob is optional (None when temporal channel disabled)
                _v7_tp = dev_output.get("v7_temporal_prob")
                if _v7_tp is not None:
                    _v7_ep_snap["v7_logit_summary"]["v7_temporal_prob"] = float(_v7_tp.mean().item())
                _v7_ep_snap["v7_logit_summary"]["v7_final_logit_composition"] = dev_output.get(
                    "v7_final_logit_composition"
                )
            else:
                _v7_ep_snap["v7_logit_summary"] = None
            _v7_epoch_history.append(_v7_ep_snap)
            # ── end Stage26-F epoch snapshot ──────────────────────────────────────────────────

            print(
                f"run={run_name} "
                + v5.format_epoch(
                    epoch,
                    total_loss.item(),
                    losses,
                    train_metrics,
                    dev_metrics,
                    active_intervention_loss,
                )
            )
            if _tbm:
                _bl_val = float(_bdry_loss.item()) if hasattr(_bdry_loss, "item") else 0.0
                print(
                    f"  [boundary22a] loss={_bl_val:.4f}"
                    f" train_acc={_tbm.get('boundary_accuracy', 0.0):.3f}"
                    f" dev_acc={_dbm.get('boundary_accuracy', 0.0):.3f}"
                    f" train_mean_prob={_tbm.get('boundary_mean_prob', 0.0):.3f}"
                    f" dev_mean_prob={_dbm.get('boundary_mean_prob', 0.0):.3f}"
                    f" valid={_tbm.get('boundary_valid_count', 0)}"
                )
            if _tfvm:
                _fv_val = float(_fv_loss.item()) if hasattr(_fv_loss, "item") else 0.0
                print(
                    f"  [fv22a3] loss={_fv_val:.4f}"
                    f" train_acc={_tfvm.get('fv_accuracy', 0.0):.3f}"
                    f" dev_acc={_dfvm.get('fv_accuracy', 0.0):.3f}"
                    f" train_mean_prob={_tfvm.get('fv_mean_prob', 0.0):.3f}"
                    f" dev_mean_prob={_dfvm.get('fv_mean_prob', 0.0):.3f}"
                    f" valid={_tfvm.get('fv_valid_count', 0)}"
                )
            if _tpim:
                _pi_loss_val = float(_pi_loss.item()) if hasattr(_pi_loss, "item") else 0.0
                print(
                    f"  [pi_head] loss={_pi_loss_val:.4f}"
                    f" train_acc={_tpim.get('pi_accuracy', 0.0):.3f}"
                    f" dev_acc={_dpim.get('pi_accuracy', 0.0):.3f}"
                    f" train_mean_prob={_tpim.get('pi_mean_prob', 0.0):.3f}"
                    f" dev_mean_prob={_dpim.get('pi_mean_prob', 0.0):.3f}"
                    f" valid={_tpim.get('pi_valid_count', 0)}"
                )
            if _tpem:
                _pe_loss_val = float(_pe_loss.item()) if hasattr(_pe_loss, "item") else 0.0
                print(
                    f"  [pe_head] loss={_pe_loss_val:.4f}"
                    f" train_acc={_tpem.get('pe_accuracy', 0.0):.3f}"
                    f" dev_acc={_dpem.get('pe_accuracy', 0.0):.3f}"
                    f" train_mean_prob={_tpem.get('pe_mean_prob', 0.0):.3f}"
                    f" dev_mean_prob={_dpem.get('pe_mean_prob', 0.0):.3f}"
                    f" valid={_tpem.get('pe_valid_count', 0)}"
                )
            if _ttdm:
                _td_loss_val = float(_td_loss.item()) if hasattr(_td_loss, "item") else 0.0
                print(
                    f"  [td_head] loss={_td_loss_val:.4f}"
                    f" train_acc={_ttdm.get('td_accuracy', 0.0):.3f}"
                    f" dev_acc={_dtdm.get('td_accuracy', 0.0):.3f}"
                    f" train_mismatch_recall={_ttdm.get('td_mismatch_recall', 0.0)}"
                    f" dev_mismatch_recall={_dtdm.get('td_mismatch_recall', 0.0)}"
                    f" train_ctrl_acc={_ttdm.get('td_control_acceptance', 0.0)}"
                    f" dev_ctrl_acc={_dtdm.get('td_control_acceptance', 0.0)}"
                    f" valid={_ttdm.get('td_valid_count', 0)}"
                )
                if _dtd_fdm:
                    print(
                        f"  [td_final] dev_temporal_reject="
                        f"{_dtd_fdm.get('td_final_temporal_rejection_rate', float('nan'))}"
                        f" dev_ctrl_pres="
                        f"{_dtd_fdm.get('td_final_control_preservation_rate', float('nan'))}"
                        f" dev_bin_acc={_dtd_fdm.get('td_final_binary_accuracy', 0.0):.3f}"
                        f" n_temporal={_dtd_fdm.get('td_final_temporal_count', 0)}"
                        f" n_ctrl={_dtd_fdm.get('td_final_control_count', 0)}"
                    )
            if _ttatm:
                _ta_loss_val = float(_ta_loss.item()) if hasattr(_ta_loss, "item") else 0.0
                print(
                    f"  [ta_adapter] loss={_ta_loss_val:.4f}"
                    f" train_acc={_ttatm.get('ta_accuracy', 0.0):.3f}"
                    f" dev_acc={_dtatm.get('ta_accuracy', 0.0):.3f}"
                    f" train_mismatch_recall={_ttatm.get('ta_mismatch_recall', float('nan'))}"
                    f" dev_mismatch_recall={_dtatm.get('ta_mismatch_recall', float('nan'))}"
                    f" train_ctrl_acc={_ttatm.get('ta_control_acceptance', float('nan'))}"
                    f" dev_ctrl_acc={_dtatm.get('ta_control_acceptance', float('nan'))}"
                    f" train_mean_prob={_ttatm.get('ta_mean_prob', 0.0):.3f}"
                    f" dev_mean_prob={_dtatm.get('ta_mean_prob', 0.0):.3f}"
                    f" valid={_ttatm.get('ta_valid_count', 0)}"
                )
            if _pc_eval_metrics:
                _pc_loss_val = float(_pc_loss.item()) if hasattr(_pc_loss, "item") else 0.0
                print(
                    f"  [pc22a4e] loss={_pc_loss_val:.4f}"
                    f" acc={_pc_eval_metrics.get('pair_contrastive_frame_accuracy', 0.0):.3f}"
                    f" margin={_pc_eval_metrics.get('pair_contrastive_frame_margin_mean', 0.0):.3f}"
                    f" pres_prob={_pc_eval_metrics.get('pair_contrastive_frame_mean_pres_fv_prob', 0.0):.3f}"
                    f" frame_prob={_pc_eval_metrics.get('pair_contrastive_frame_mean_frame_fv_prob', 0.0):.3f}"
                    f" n={_pc_eval_metrics.get('pair_contrastive_frame_valid_count', 0)}"
                    f" use_case={args.pair_contrastive_use_case}"
                )
                if _pc_pres_type_counts:
                    print(
                        f"  [pc22a4e] pres={_pc_pres_type_counts}"
                        f" frame={_pc_frame_type_counts}"
                    )

        # Apply TD-constrained checkpoint selection after all epochs
        _tc_used = False
        _tc_fallback_used = False
        _tc_reason = "unconstrained_default"
        if use_td_constrained_selection:
            _tc_used = True
            if _tc_epoch >= 1:
                # At least one epoch satisfied all constraints — override best_* with constrained
                print(
                    f"  [td_constrained_sel] selected epoch={_tc_epoch}"
                    f" clean_macro={_tc_score:.4f}"
                    f" paraphrase_preserved={_tc_paraphrase_preserved:.3f}"
                    f" td_mismatch_recall={_tc_mismatch_recall}"
                    f" td_ctrl_acc={_tc_ctrl_acc}"
                    f" (unconstrained_fallback_epoch={best_epoch})"
                )
                best_epoch = _tc_epoch
                best_dev_metrics = _tc_dev_metrics
                best_dev_interventions = _tc_dev_interventions
                best_dev_pairwise_checks = _tc_dev_pairwise_checks
                best_dev_predictions = _tc_dev_predictions
                best_state = _tc_state
                best_pc_metrics = _tc_pc_metrics
                _tc_reason = "constrained_selection_applied"
            else:
                # No epoch met all constraints — fall back to unconstrained best
                _tc_fallback_used = True
                _tc_reason = "constrained_fallback_no_eligible_epoch"
                _fd_note = (
                    f" min_ftr={td_sel_min_final_temporal_rejection}"
                    f" min_fcp={td_sel_min_final_control_preservation}"
                    if use_td_final_decision_selection else ""
                )
                print(
                    f"  [td_constrained_sel] WARNING: no eligible epoch found "
                    f"(min_pp={td_sel_min_paraphrase_preserved}"
                    f" min_mr={td_sel_min_mismatch_recall}"
                    f" min_ca={td_sel_min_control_acceptance}{_fd_note}); "
                    f"falling back to unconstrained best epoch={best_epoch}"
                )
        _tc_constrained_applied = _tc_used and not _tc_fallback_used
        _tc_selection_info: dict[str, Any] = {
            "use_td_constrained_selection": use_td_constrained_selection,
            "td_selection_min_paraphrase_preserved": td_sel_min_paraphrase_preserved,
            "td_selection_min_mismatch_recall": td_sel_min_mismatch_recall,
            "td_selection_min_control_acceptance": td_sel_min_control_acceptance,
            "td_selection_use_final_decision": use_td_final_decision_selection,
            "td_selection_min_final_temporal_rejection": td_sel_min_final_temporal_rejection,
            "td_selection_min_final_control_preservation": td_sel_min_final_control_preservation,
            "td_selection_fallback": "final_macro_f1",
            "stage15_used_for_td_constrained_selection": False,
            "td_constrained_selection_used": _tc_used,
            "td_constrained_selection_fallback_used": _tc_fallback_used,
            "td_constrained_selection_reason": _tc_reason,
            "td_constrained_selection_eligible_epoch_count": (
                -1 if not use_td_constrained_selection else (0 if _tc_epoch < 1 else 1)
            ),
            "td_constrained_selection_selected_epoch": best_epoch if _tc_used else None,
            "td_constrained_selection_best_clean_macro": (
                _tc_score if _tc_constrained_applied else None
            ),
            "td_constrained_selection_selected_td_mismatch_recall": (
                _tc_mismatch_recall if _tc_constrained_applied else None
            ),
            "td_constrained_selection_selected_td_control_acceptance": (
                _tc_ctrl_acc if _tc_constrained_applied else None
            ),
            "td_constrained_selection_selected_paraphrase_preserved": (
                _tc_paraphrase_preserved if _tc_constrained_applied else None
            ),
            "td_constrained_selection_selected_td_final_temporal_rejection_rate": (
                _tc_td_final_temporal_rejection if _tc_constrained_applied else None
            ),
            "td_constrained_selection_selected_td_final_control_preservation_rate": (
                _tc_td_final_control_preservation if _tc_constrained_applied else None
            ),
            "td_constrained_selection_selected_td_final_binary_accuracy": (
                _tc_td_final_binary_accuracy if _tc_constrained_applied else None
            ),
        }

        # Apply generic preservation-constrained checkpoint selection after all epochs
        _pcs_used = False
        _pcs_fallback_used = False
        _pcs_reason = "unconstrained_default"
        if use_preservation_constrained_selection:
            _pcs_used = True
            if _pcs_epoch >= 1:
                # At least one epoch satisfied preservation constraints — override best_*
                print(
                    f"  [pres_constrained_sel] selected epoch={_pcs_epoch}"
                    f" clean_macro={_pcs_score:.4f}"
                    f" paraphrase_preserved={_pcs_paraphrase_preserved:.3f}"
                    f" predicate_disentangled={_pcs_predicate_disentangled:.3f}"
                    f" eligible_epochs={_pcs_eligible_count}"
                    f" (unconstrained_fallback_epoch={best_epoch})"
                )
                best_epoch = _pcs_epoch
                best_dev_metrics = _pcs_dev_metrics
                best_dev_interventions = _pcs_dev_interventions
                best_dev_pairwise_checks = _pcs_dev_pairwise_checks
                best_dev_predictions = _pcs_dev_predictions
                best_state = _pcs_state
                best_pc_metrics = _pcs_pc_metrics
                _pcs_reason = "constrained_selection_applied"
            else:
                # No epoch met preservation constraints — fall back to unconstrained best
                _pcs_fallback_used = True
                _pcs_reason = "constrained_fallback_no_eligible_epoch"
                print(
                    f"  [pres_constrained_sel] WARNING: no eligible epoch found "
                    f"(min_pp={sel_min_paraphrase_preserved}"
                    f" min_pd={sel_min_predicate_disentangled}); "
                    f"falling back to unconstrained best epoch={best_epoch}"
                )
        _pcs_applied = _pcs_used and not _pcs_fallback_used
        _pcs_selection_info: dict[str, Any] = {
            "use_preservation_constrained_selection": use_preservation_constrained_selection,
            "selection_min_paraphrase_preserved": sel_min_paraphrase_preserved,
            "selection_min_predicate_disentangled": sel_min_predicate_disentangled,
            "selection_fallback": sel_fallback,
            "stage15_used_for_preservation_constrained_selection": False,
            "preservation_constrained_selection_used": _pcs_used,
            "preservation_constrained_selection_fallback_used": _pcs_fallback_used,
            "preservation_constrained_selection_reason": _pcs_reason,
            "preservation_constrained_selection_eligible_epoch_count": (
                -1 if not use_preservation_constrained_selection else _pcs_eligible_count
            ),
            "preservation_constrained_selection_selected_epoch": (
                best_epoch if _pcs_used else None
            ),
            "preservation_constrained_selection_selected_clean_macro": (
                _pcs_score if _pcs_applied else None
            ),
            "preservation_constrained_selection_selected_paraphrase_preserved": (
                _pcs_paraphrase_preserved if _pcs_applied else None
            ),
            "preservation_constrained_selection_selected_predicate_disentangled": (
                _pcs_predicate_disentangled if _pcs_applied else None
            ),
        }

        # ── Build run-level audit ledger (reporting only; no model/loss/logit change) ──────────
        _n = max(_audit_epoch_count, 1)

        def _avg_epoch_dicts(dicts: list[dict]) -> dict[str, float]:
            if not dicts:
                return {}
            keys = list(dicts[0].keys())
            return {k: round(sum(d[k] for d in dicts) / len(dicts), 6) for k in keys}

        _loss_epoch_avg_raw = _avg_epoch_dicts(_audit_per_epoch_raw)
        _loss_epoch_avg_weighted = _avg_epoch_dicts(_audit_per_epoch_weighted)

        # selected_epoch_loss: the epoch that was actually checkpointed (best_epoch, post-selector)
        # final_epoch_loss: the last epoch of training (index -1)
        _sel_idx = (best_epoch - 1) if 0 <= best_epoch - 1 < len(_audit_per_epoch_raw) else None
        _fin_idx = len(_audit_per_epoch_raw) - 1 if _audit_per_epoch_raw else None
        _selected_epoch_raw = _audit_per_epoch_raw[_sel_idx] if _sel_idx is not None else None
        _selected_epoch_weighted = _audit_per_epoch_weighted[_sel_idx] if _sel_idx is not None else None
        _final_epoch_raw = _audit_per_epoch_raw[_fin_idx] if _fin_idx is not None else None
        _final_epoch_weighted = _audit_per_epoch_weighted[_fin_idx] if _fin_idx is not None else None

        # Ratios using weighted values (weighted matches what total_loss actually accumulated)
        _ce_avg_w = _loss_epoch_avg_weighted.get("ce_loss", 0.0)
        _total_avg_w = _loss_epoch_avg_weighted.get("total_loss", 0.0)
        _aux_weighted_sum = round(_total_avg_w - _ce_avg_w, 6)
        _aux_to_ce_w = round(_aux_weighted_sum / _ce_avg_w, 4) if _ce_avg_w > 0.0 else None

        _ce_avg_r = _loss_epoch_avg_raw.get("ce_loss", 0.0)
        _total_avg_r = _loss_epoch_avg_raw.get("total_loss", 0.0)
        _aux_raw_sum = round(_total_avg_r - _ce_avg_r, 6)
        _aux_to_ce_r = round(_aux_raw_sum / _ce_avg_r, 4) if _ce_avg_r > 0.0 else None

        _active_training_losses: dict[str, Any] = {
            "main_ce": {
                "enabled": True,
                "weight": 1.0,
                "target": "output['logits'] (final_logits)",
                "diagnostic_head_only": False,
                "raw_loss_key": "ce_loss",
                "weighted_loss_key": "ce_loss",
                "note": "always active; weight=1.0 so raw == weighted",
            },
            "ranking_loss": {
                "enabled": not use_intervention_loss,
                "weight": ranking_weight if not use_intervention_loss else 0.0,
                "target": "output['logits'] (ranking objective via v5.intervention_objective)",
                "diagnostic_head_only": False,
                "raw_loss_key": "ranking_loss",
                "weighted_loss_key": "ranking_loss",
                "note": (
                    f"ranking_weight={ranking_weight}; weighted_loss = ranking_weight * raw_loss. "
                    "Baseline is NOT CE-only when this is enabled — ranking loss is active by default."
                    if not use_intervention_loss else "disabled; use_intervention_loss=True"
                ),
            },
            "intervention_loss": {
                "enabled": use_intervention_loss,
                "weight": 1.0 if use_intervention_loss else 0.0,
                "target": "output['logits'] (structured pairwise via intervention_pairwise_losses)",
                "diagnostic_head_only": False,
                "raw_loss_key": "intervention_loss",
                "weighted_loss_key": "intervention_loss",
                "note": (
                    "pairwise_losses['total']; internally weighted composite — "
                    "raw and weighted are identical in the audit (no outer scalar)"
                    if use_intervention_loss else "disabled; use_intervention_loss=False"
                ),
            },
            "boundary_loss": {
                "enabled": boundary_loss_weight > 0.0,
                "weight": boundary_loss_weight,
                "pos_weight": boundary_loss_pos_weight,
                "target": "boundary_logit",
                "diagnostic_head_only": True,
                "raw_loss_key": "boundary_loss",
                "weighted_loss_key": "boundary_loss",
            },
            "frame_violation_loss": {
                "enabled": fv_loss_weight > 0.0,
                "weight": fv_loss_weight,
                "pos_weight": fv_loss_pos_weight,
                "target": "frame_violation_logit",
                "diagnostic_head_only": True,
                "raw_loss_key": "frame_violation_loss",
                "weighted_loss_key": "frame_violation_loss",
            },
            "pair_contrastive_frame_loss": {
                "enabled": pc_loss_weight > 0.0,
                "weight": pc_loss_weight,
                "target": "frame_violation_logit (margin ranking between pres/frame pairs)",
                "diagnostic_head_only": True,
                "raw_loss_key": "pair_contrastive_frame_loss",
                "weighted_loss_key": "pair_contrastive_frame_loss",
            },
            "predicate_isolation_loss": {
                "enabled": pi_loss_weight > 0.0,
                "weight": pi_loss_weight,
                "pos_weight": pi_loss_pos_weight,
                "target": "predicate_noncoverage_logit",
                "diagnostic_head_only": True,
                "raw_loss_key": "predicate_isolation_loss",
                "weighted_loss_key": "predicate_isolation_loss",
            },
            "preservation_entitlement_loss": {
                "enabled": pe_loss_weight > 0.0,
                "weight": pe_loss_weight,
                "pos_weight": pe_loss_pos_weight,
                "target": "preservation_entitlement_logit",
                "diagnostic_head_only": True,
                "raw_loss_key": "preservation_entitlement_loss",
                "weighted_loss_key": "preservation_entitlement_loss",
            },
            "temporal_diagnostic_loss": {
                "enabled": td_loss_weight > 0.0,
                "weight": td_loss_weight,
                "pos_weight": td_loss_pos_weight,
                "target": "temporal_diagnostic_logit",
                "diagnostic_head_only": True,
                "raw_loss_key": "temporal_diagnostic_loss",
                "weighted_loss_key": "temporal_diagnostic_loss",
                "note": (
                    "WARNING: supervised via frame_pair_repr (shared); "
                    "gradient coupling risk confirmed in Stage23 at weight >= 0.05"
                    if td_loss_weight > 0.0 else "disabled"
                ),
            },
            "temporal_adapter_loss": {
                "enabled": ta_loss_weight > 0.0,
                "weight": ta_loss_weight,
                "pos_weight": ta_loss_pos_weight,
                "target": "temporal_adapter_logit",
                "diagnostic_head_only": True,
                "gradient_isolated": True,
                "raw_loss_key": "temporal_adapter_loss",
                "weighted_loss_key": "temporal_adapter_loss",
                "note": (
                    "adapter input is frame_pair_repr.detach(); no gradient to FrameGate"
                    if ta_loss_weight > 0.0 else "disabled"
                ),
            },
            "temporal_channel_loss": {
                "enabled": getattr(model, "use_temporal_channel_loss", tc_loss_weight > 0.0),
                "weight": getattr(model, "temporal_channel_loss_weight", tc_loss_weight),
                "pos_weight": getattr(model, "temporal_channel_loss_pos_weight", tc_loss_pos_weight),
                "target": "temporal_channel_logit",
                "diagnostic_head_only": True,
                "gradient_isolated": bool(
                    getattr(model, "temporal_channel_detach_input", True)
                ),
                "head_active": getattr(model, "temporal_channel_v1", None) is not None,
                "input_representation": "cat([claim_frame_state, evidence_frame_state])",
                "input_representation_note": (
                    "pre-pair-projector slot states from FrameGate — NOT frame_pair_repr. "
                    "With detach=True (default), TC loss cannot propagate into FrameGate parameters."
                ),
                "raw_loss_key": "temporal_channel_loss",
                "weighted_loss_key": "temporal_channel_loss",
                "note": (
                    f"temporal_channel_detach_input={getattr(model, 'temporal_channel_detach_input', True)}; "
                    "TC V1 reads cat([claim_frame_state, evidence_frame_state]), pre-pair-projector"
                    if tc_loss_weight > 0.0 else "disabled"
                ),
            },
            # Stage26-G: v7 stabilization losses — all off by default, v7_hierarchical only
            "v7_polarity_margin_loss": {
                "enabled": (
                    args.architecture == "v7_hierarchical"
                    and getattr(args, "v7_use_polarity_margin_loss", False)
                    and getattr(args, "v7_polarity_margin_loss_weight", 0.0) > 0.0
                    and not getattr(args, "v7_no_aux_losses", False)
                ),
                "weight": getattr(args, "v7_polarity_margin_loss_weight", 0.0),
                "margin": getattr(args, "v7_polarity_margin", 0.5),
                "target": "v7_polarity_support_logit vs v7_polarity_refute_logit",
                "scope": "SUPPORT and REFUTE examples only (NOT_ENTITLED excluded)",
                "stage15_used_for_selection_or_calibration": False,
                "ood_used": False,
                "raw_loss_key": "v7_polarity_margin_loss",
                "weighted_loss_key": "v7_polarity_margin_loss",
                "note": (
                    "Hinge margin on polarity logits: pushes |support_logit - refute_logit| >= margin "
                    "in the correct direction for gold SUPPORT and REFUTE. NOT_ENTITLED excluded. "
                    "CE unchanged (still uses output['logits']). No loss_logits. No OOD. No Stage15."
                    if (
                        getattr(args, "v7_use_polarity_margin_loss", False)
                        and getattr(args, "v7_polarity_margin_loss_weight", 0.0) > 0.0
                    ) else "disabled"
                ),
            },
            "v7_entitlement_bce_loss": {
                "enabled": (
                    args.architecture == "v7_hierarchical"
                    and getattr(args, "v7_use_entitlement_bce_loss", False)
                    and getattr(args, "v7_entitlement_bce_loss_weight", 0.0) > 0.0
                    and not getattr(args, "v7_no_aux_losses", False)
                ),
                "weight": getattr(args, "v7_entitlement_bce_loss_weight", 0.0),
                "pos_weight": getattr(args, "v7_entitlement_bce_pos_weight", 1.0),
                "target": "v7_entitlement_logit",
                "target_derivation": "entitled=1 for SUPPORT/REFUTE, entitled=0 for NOT_ENTITLED",
                "stage15_used_for_selection_or_calibration": False,
                "ood_used": False,
                "raw_loss_key": "v7_entitlement_bce_loss",
                "weighted_loss_key": "v7_entitlement_bce_loss",
                "note": (
                    "BCE on EntitlementGate output. Target is a hard 0/1 derived from gold labels "
                    "on clean train data only. No OOD. No Stage15."
                    if (
                        getattr(args, "v7_use_entitlement_bce_loss", False)
                        and getattr(args, "v7_entitlement_bce_loss_weight", 0.0) > 0.0
                    ) else "disabled"
                ),
            },
            "v7_entitled_class_balanced_ce": {
                "enabled": (
                    args.architecture == "v7_hierarchical"
                    and getattr(args, "v7_use_entitled_class_balanced_ce", False)
                    and getattr(args, "v7_entitled_class_balanced_ce_weight", 0.0) > 0.0
                    and not getattr(args, "v7_no_aux_losses", False)
                ),
                "weight": getattr(args, "v7_entitled_class_balanced_ce_weight", 0.0),
                "target": "v7_polarity_logits",
                "target_derivation": "REFUTE→local_0, SUPPORT→local_1 (NOT_ENTITLED excluded)",
                "stage15_used_for_selection_or_calibration": False,
                "ood_used": False,
                "raw_loss_key": "v7_entitled_class_balanced_ce_loss",
                "weighted_loss_key": "v7_entitled_class_balanced_ce_loss",
                "note": (
                    "Auxiliary CE over v7_polarity_logits on SUPPORT/REFUTE examples only. "
                    "NOT_ENTITLED excluded. Does NOT use output['logits']. No OOD. No Stage15."
                    if (
                        getattr(args, "v7_use_entitled_class_balanced_ce", False)
                        and getattr(args, "v7_entitled_class_balanced_ce_weight", 0.0) > 0.0
                    ) else "disabled"
                ),
            },
        }

        # True post-hoc final-logit modifiers only.
        # Comparator alphas are learned architectural parameters inside model.forward() and are
        # recorded separately under active_architectural_logit_components.
        _active_final_logit_modifiers: dict[str, Any] = {
            "temporal_adapter_final_penalty": {
                "enabled": ta_final_penalty_scale > 0.0,
                "scale": ta_final_penalty_scale,
                "type": "local_example_dependent",
                "stage15_used_for_selection_or_calibration": False,
                "note": (
                    "per-example NOT_ENTITLED boost proportional to sigmoid(adapter_logit).detach(). "
                    "Scale is a fixed hyperparameter — not OOD-calibrated. Stage15 never used."
                    if ta_final_penalty_scale > 0.0 else "disabled"
                ),
            },
            "temporal_channel_gated_penalty": {
                "enabled": (
                    getattr(model, "use_temporal_channel_gated_penalty", use_temporal_channel_gated_penalty)
                    and tc_gated_penalty_scale > 0.0
                ),
                "scale": getattr(model, "temporal_channel_gated_penalty_scale", tc_gated_penalty_scale),
                "type": "local_example_dependent_gated",
                "gating_formula": (
                    "scale * sigmoid(tc_logit).detach() * (1 - pe_prob).detach()"
                ),
                "gating_requirement": "preservation_entitlement_head must be active",
                "stage15_used_for_selection_or_calibration": False,
                "note": (
                    "PE-gated NOT_ENTITLED boost. Fires only when TC detects temporal mismatch "
                    "AND PE signals non-entitlement. Scale is a fixed hyperparameter — not "
                    "OOD-calibrated. Stage15 never used. Cannot be combined with "
                    "temporal_adapter_final_penalty in the same run."
                    if use_temporal_channel_gated_penalty and tc_gated_penalty_scale > 0.0 else "disabled"
                ),
            },
            "dev_calibrated_ne_shift": {
                "enabled": args.dev_calibrated_ne_shift_candidates is not None,
                "candidates": args.dev_calibrated_ne_shift_candidates,
                "type": "global_shift",
                "calibration_source": (
                    f"controlled_{args.dev_calibrated_ne_calibration_source}_only"
                    if args.dev_calibrated_ne_shift_candidates is not None else None
                ),
                "stage15_used_for_selection_or_calibration": False,
                "note": (
                    "post-hoc global NE logit shift; selected on controlled data only — not Stage15. "
                    "Diagnostic method; not a validated final-model component."
                    if args.dev_calibrated_ne_shift_candidates is not None else "disabled"
                ),
            },
            "ood_unflagged_ne_shift": {
                "enabled": args.ood_unflagged_ne_shift_sweep is not None,
                "type": "ood_tuned_eval_only",
                "stage15_used_for_selection_or_calibration": (
                    args.ood_unflagged_ne_shift_sweep is not None
                ),
                "note": (
                    "OOD-tuned global NE shift — Stage15 is consulted to select the shift. "
                    "NOT a valid final-model component. Diagnostic upper bound only."
                    if args.ood_unflagged_ne_shift_sweep is not None else "disabled"
                ),
            },
            "ood_selective_ne_shift": {
                "enabled": args.ood_selective_ne_shift_sweep is not None,
                "type": "ood_tuned_eval_only",
                "stage15_used_for_selection_or_calibration": (
                    args.ood_selective_ne_shift_sweep is not None
                ),
                "note": (
                    "OOD-tuned selective NE shift — Stage15 is consulted to select the shift. "
                    "NOT a valid final-model component. Diagnostic upper bound only."
                    if args.ood_selective_ne_shift_sweep is not None else "disabled"
                ),
            },
        }

        # Architectural logit components: learned parameters inside model.forward().
        # These modify final_logits as part of the model graph — they are trained through
        # backprop and gated on external flag inputs, NOT post-hoc fixed-scale modifiers.
        _active_architectural_logit_components: dict[str, Any] = {
            "temporal_comparator_alpha": {
                "enabled": bool(getattr(model, "use_temporal_comparator", False)),
                "type": "learned_gated_scalar",
                "gating": "temporal_mismatch_flags (per-example; flagged examples only)",
                "effect": "shifts final_logits[active, NOT_ENTITLED] by +alpha; SUPPORT/REFUTE by -alpha",
                "trained_via_backprop": True,
                "stage15_used": False,
                "note": "learned scalar alpha_temporal(); applies to flagged examples; not a post-hoc modifier",
            },
            "predicate_comparator_alpha": {
                "enabled": bool(getattr(model, "use_predicate_comparator", False)),
                "type": "learned_gated_scalar",
                "gating": "predicate_mismatch_flags (per-example; flagged examples only)",
                "effect": "shifts final_logits[active, NOT_ENTITLED] by +alpha; SUPPORT/REFUTE by -alpha",
                "trained_via_backprop": True,
                "stage15_used": False,
                "note": "learned scalar alpha_predicate(); applies to flagged examples; not a post-hoc modifier",
            },
        }

        _active_selection_rules: dict[str, Any] = {
            "standard_clean_dev": {
                "enabled": True,
                "metric": select_metric,
                "stage15_used": False,
                "note": "always active; primary checkpoint metric on clean controlled dev",
            },
            "td_constrained_selection": {
                "enabled": use_td_constrained_selection,
                "constraints": {
                    "min_paraphrase_preserved": td_sel_min_paraphrase_preserved,
                    "min_td_mismatch_recall": td_sel_min_mismatch_recall,
                    "min_td_control_acceptance": td_sel_min_control_acceptance,
                    "use_final_decision": use_td_final_decision_selection,
                } if use_td_constrained_selection else None,
                "fallback": "final_macro_f1",
                "stage15_used": False,
                "fallback_triggered": _tc_fallback_used,
            },
            "preservation_constrained_selection": {
                "enabled": use_preservation_constrained_selection,
                "constraints": {
                    "min_paraphrase_preserved": sel_min_paraphrase_preserved,
                    "min_predicate_disentangled": sel_min_predicate_disentangled,
                } if use_preservation_constrained_selection else None,
                "fallback": sel_fallback,
                "stage15_used": False,
                "fallback_triggered": _pcs_fallback_used,
                "eligible_epoch_count": (
                    _pcs_eligible_count if use_preservation_constrained_selection else None
                ),
            },
        }

        # Audit warnings (reporting only; using weighted ratios as primary signal)
        _audit_warnings: list[str] = []
        if _aux_to_ce_w is not None and _aux_to_ce_w > 0.5:
            _audit_warnings.append(
                f"aux_to_ce_loss_ratio_weighted={_aux_to_ce_w:.3f} > 0.5: "
                "weighted auxiliary losses exceed 50% of CE — total_loss is not CE-dominated"
            )
        if not use_intervention_loss and ranking_weight > 2.0:
            _audit_warnings.append(
                f"ranking_loss enabled with large weight={ranking_weight}: "
                "baseline is not CE-only; ranking loss contributes substantially to total_loss"
            )
        elif not use_intervention_loss and ranking_weight > 0.0:
            _audit_warnings.append(
                f"ranking_loss is active (ranking_weight={ranking_weight}): "
                "baseline is not CE-only; see loss_component_epoch_avg_weighted['ranking_loss']"
            )
        # Count only true post-hoc modifiers (not architectural components)
        _active_modifier_count = sum(
            1 for _mod in _active_final_logit_modifiers.values() if _mod.get("enabled")
        )
        if _active_modifier_count > 1:
            _audit_warnings.append(
                f"{_active_modifier_count} post-hoc final-logit modifiers enabled simultaneously; "
                "verify priority and interaction: "
                + ", ".join(
                    k for k, v in _active_final_logit_modifiers.items() if v.get("enabled")
                )
            )
        if use_td_constrained_selection and use_preservation_constrained_selection:
            _audit_warnings.append(
                "CRITICAL: both td_constrained_selection and preservation_constrained_selection "
                "are enabled — this should have been caught at startup"
            )
        if args.ood_unflagged_ne_shift_sweep is not None or args.ood_selective_ne_shift_sweep is not None:
            _audit_warnings.append(
                "OOD-tuned NE shift is active during OOD eval: "
                "this is a diagnostic upper bound, NOT a valid final-model method. "
                "Stage15 was used to select or sweep the shift value."
            )
        if td_loss_weight > 0.0:
            _audit_warnings.append(
                "temporal_diagnostic_loss is active: supervises via frame_pair_repr (shared); "
                "Stage23 confirmed this causes preservation/predicate collapse at weight >= 0.05"
            )
        if ta_final_penalty_scale > 0.0 and ta_loss_weight <= 0.0:
            _audit_warnings.append(
                "temporal_adapter_final_penalty is enabled but temporal_adapter_loss=0: "
                "adapter was not trained this run; penalty applies untrained adapter probabilities"
            )
        if use_temporal_channel_gated_penalty and tc_gated_penalty_scale > 0.0 and tc_loss_weight <= 0.0:
            _audit_warnings.append(
                "temporal_channel_gated_penalty is enabled but temporal_channel_loss=0: "
                "TC head was not trained this run; gated penalty applies untrained TC probabilities. "
                "Enable --use-temporal-channel-loss with a nonzero weight to train the head."
            )
        if use_temporal_channel and not any([
            tc_loss_weight > 0.0,
            use_temporal_channel_gated_penalty and tc_gated_penalty_scale > 0.0,
        ]):
            _audit_warnings.append(
                "temporal_channel_v1 is instantiated but neither tc_loss nor tc_gated_penalty is enabled: "
                "TC head adds parameters with no training signal and no effect on final logits"
            )
        if tc_loss_weight > 0.0 and _loss_epoch_avg_raw.get("temporal_channel_loss", 0.0) == 0.0:
            _audit_warnings.append(
                "temporal_channel_loss is enabled (weight > 0) but epoch-average raw loss is 0.0: "
                "TC BCE was not computed. Check that --temporal-diagnostic-data was provided "
                "and that the TC head produced temporal_channel_logit."
            )
        if tc_loss_weight > 0.0 and _loss_epoch_avg_weighted.get("temporal_channel_loss", 0.0) == 0.0:
            _audit_warnings.append(
                "temporal_channel_loss weighted average is 0.0 despite weight > 0: "
                "TC BCE contributed nothing to total_loss this run."
            )
        if _tc_fallback_used:
            _audit_warnings.append(
                "td_constrained_selection fallback triggered: "
                "no epoch satisfied TD constraints; using unconstrained best epoch"
            )
        if _pcs_fallback_used:
            _audit_warnings.append(
                "preservation_constrained_selection fallback triggered: "
                "no epoch satisfied preservation constraints; using unconstrained best epoch"
            )

        # ── Stage26-F extended: build v7 collapse and logit diagnostics ──────────────────────
        # No new forward passes.  All values derived from already-computed variables.
        # _best_dev_output_v7 : CPU tensors from the unconstrained-best epoch (may differ
        #                       from checkpointed epoch if TC/PCS selection applied).
        # dev_output           : last epoch's output (still in scope after the for-loop).
        # best_dev_metrics     : always reflects the actually selected epoch (after TC/PCS).
        _v7_ext_diagnostics: dict[str, Any] = {}
        if args.architecture == "v7_hierarchical":
            _v7_best_logit_summary = _v7_make_logit_summary(_best_dev_output_v7)

            # Final-epoch logit summary — dev_output is the last iteration's output
            _v7_final_logit_summary: "dict[str, Any] | None" = None
            if epochs > 0:
                _fin_out_v7 = {
                    k: v.detach().cpu() if hasattr(v, "detach") else v
                    for k, v in dev_output.items()
                    if v is not None
                }
                _v7_final_logit_summary = _v7_make_logit_summary(_fin_out_v7)

            # Per-gold-label breakdown from best-epoch tensors + ground-truth dev labels
            _v7_per_gold_summary = _v7_make_per_gold_summary(
                _best_dev_output_v7,
                dev_inputs.get("final_labels"),
            )

            # Collapse / recall fields from best_dev_metrics (correct after TC/PCS overrides)
            _bm = best_dev_metrics or {}
            _pred_dist: dict[str, int] = _bm.get("prediction_distribution") or {}
            _total_preds = sum(_pred_dist.values()) if _pred_dist else 0
            _per_label: dict[str, Any] = _bm.get("per_label") or {}
            _maj_class: "str | None" = None
            if _pred_dist and _total_preds > 0:
                _maj_class = max(_pred_dist, key=lambda _k: _pred_dist[_k])

            _v7_ext_diagnostics = {
                "v7_best_dev_logit_summary": _v7_best_logit_summary,
                "v7_final_epoch_logit_summary": _v7_final_logit_summary,
                "v7_best_dev_per_gold_label_summary": _v7_per_gold_summary,
                "v7_predicted_single_class": (
                    (len(_pred_dist) == 1) if _maj_class is not None else None
                ),
                "v7_predicted_majority_class": _maj_class,
                "v7_predicted_majority_fraction": (
                    _pred_dist[_maj_class] / _total_preds
                    if _maj_class is not None else None
                ),
                "v7_support_prediction_count": (
                    _pred_dist.get("SUPPORT", 0) if _pred_dist else None
                ),
                "v7_refute_prediction_count": (
                    _pred_dist.get("REFUTE", 0) if _pred_dist else None
                ),
                "v7_ne_prediction_count": (
                    _pred_dist.get("NOT_ENTITLED", 0) if _pred_dist else None
                ),
                "v7_support_recall": _per_label.get("SUPPORT", {}).get("recall"),
                "v7_refute_recall": _per_label.get("REFUTE", {}).get("recall"),
                "v7_ne_recall": _per_label.get("NOT_ENTITLED", {}).get("recall"),
            }
        # ── end Stage26-F extended ─────────────────────────────────────────────────────────

        _run_audit_ledger: dict[str, Any] = {
            "active_training_losses": _active_training_losses,
            "active_final_logit_modifiers": _active_final_logit_modifiers,
            "active_architectural_logit_components": _active_architectural_logit_components,
            "active_selection_rules": _active_selection_rules,
            # Loss component averages — raw (pre-weight) and weighted (contribution to total_loss)
            "loss_component_epoch_avg_raw": _loss_epoch_avg_raw,
            "loss_component_epoch_avg_weighted": _loss_epoch_avg_weighted,
            "loss_component_epoch_avg": _loss_epoch_avg_weighted,   # backward-compat alias
            "loss_component_epoch_avg_semantics": "weighted",
            # Selected epoch loss (epoch actually checkpointed, after all selectors ran)
            "selected_epoch_loss_component_avg_raw": _selected_epoch_raw,
            "selected_epoch_loss_component_avg_weighted": _selected_epoch_weighted,
            "selected_epoch_loss_component_avg": _selected_epoch_weighted,
            # Final epoch loss (last epoch of training)
            "final_epoch_loss_component_avg_raw": _final_epoch_raw,
            "final_epoch_loss_component_avg_weighted": _final_epoch_weighted,
            "final_epoch_loss_component_avg": _final_epoch_weighted,
            # Ratios — weighted is primary; raw is supplemental
            "aux_weighted_loss_sum": _aux_weighted_sum,
            "aux_to_ce_loss_ratio_weighted": _aux_to_ce_w,
            "aux_raw_loss_sum": _aux_raw_sum,
            "aux_to_ce_loss_ratio_raw": _aux_to_ce_r,
            "aux_to_ce_loss_ratio": _aux_to_ce_w,   # backward-compat alias; equals _weighted
            "audit_warnings": _audit_warnings,
            "audit_epoch_count": _audit_epoch_count,
            "selected_epoch": best_epoch,
            "final_epoch": epochs,
            # Static provenance
            "stage15_used_for_training": False,
            "stage15_used_for_loss_selection": False,
            "stage15_used_for_final_logit_modifier_selection": False,
            "stage15_used_for_checkpoint_selection": False,
            "stage15_used_for_temporal_channel_training": False,
            "stage15_used_for_temporal_channel_penalty_selection": False,
            "time_swap_used_in_main_clean_data": False,
            # Stage26-A: v7 hierarchical architecture provenance (always False in Stage26-A)
            "stage15_used_for_v7_training": False,
            "stage15_used_for_v7_selection": False,
            "stage15_used_for_v7_aux_loss_targets": False,
            "time_swap_used_in_v7_main_clean_data": False,
        }

        report = {
            "run_name": run_name,
            "final_epoch": epochs,
            "best_epoch": best_epoch,
            "select_metric": select_metric,
            "best_dev_metrics": best_dev_metrics,
            "best_dev_interventions": best_dev_interventions,
            "best_dev_pairwise_checks": best_dev_pairwise_checks,
            "_best_dev_predictions": best_dev_predictions,
            "loss_config": loss_config,
            "best_pair_contrastive_frame_metrics": best_pc_metrics,
            **_tc_selection_info,
            **_pcs_selection_info,
            "audit_ledger": _run_audit_ledger,
            # Stage26-F: per-epoch dev metric history for post-hoc diagnosis
            "v7_epoch_diagnostic_history": _v7_epoch_history,
            # Stage26-F extended: v7 logit summaries, per-gold breakdown, collapse/recall fields
            **_v7_ext_diagnostics,
        }
        report["_best_state"] = best_state
        if best_trainable_state is not None:
            report["best_trainable_state"] = best_trainable_state
        return report

    requested_loss_config = {
        "lambda_frame_preserve": args.lambda_frame_preserve,
        "lambda_frame_anchor": args.lambda_frame_anchor,
        "lambda_predicate_contrast": args.lambda_predicate_contrast,
        "lambda_predicate_anchor": args.lambda_predicate_anchor,
        "lambda_sufficiency_contrast": args.lambda_sufficiency_contrast,
        "lambda_polarity_flip": args.lambda_polarity_flip,
        "lambda_polarity_margin_anchor": args.lambda_polarity_margin_anchor,
        "lambda_paraphrase_preserve": args.lambda_paraphrase_preserve,
        "lambda_entitlement_preserve": args.lambda_entitlement_preserve,
        "lambda_logit_preserve": args.lambda_logit_preserve,
        "ranking_margin": args.ranking_margin,
        "polarity_margin_min": args.polarity_margin_min,
    }
    configurations = (
        v5.sweep_presets(args.ranking_margin)
        if args.loss_sweep
        else {"single": requested_loss_config}
    )
    initial_head_state = v5.capture_head_state(model)
    reports: dict[str, dict[str, Any]] = {}
    # TD data is shared by temporal_diagnostic_loss, temporal_adapter_loss, and
    # temporal_channel_loss. Pass it whenever any of those needs it.
    _td_data_needed = (
        args.use_temporal_diagnostic_loss
        or args.use_temporal_channel_loss
        or args.use_temporal_adapter_loss
    )

    for run_name, loss_config in configurations.items():
        v5.restore_head_state(model, initial_head_state)
        torch.manual_seed(args.seed)
        reports[run_name] = run_training_v6b(
            model,
            train_inputs,
            dev_inputs,
            train_records,
            dev_records,
            train_bundle,
            epochs=args.epochs,
            lr=args.lr,
            head_lr=args.head_lr,
            encoder_lr=args.encoder_lr,
            weighted_label_loss=args.weighted_label_loss,
            balanced_sampler=args.balanced_sampler,
            use_intervention_loss=args.use_intervention_loss or args.loss_sweep,
            ranking_weight=args.ranking_weight,
            loss_config=loss_config,
            seed=args.seed,
            run_name=run_name,
            select_metric=args.select_metric,
            smoke_mode=args.smoke,
            ce_class_weights=ce_class_weights,
            train_boundary_labels=train_boundary_labels if args.use_boundary_loss else None,
            train_boundary_mask=train_boundary_mask if args.use_boundary_loss else None,
            dev_boundary_labels=dev_boundary_labels if args.use_boundary_loss else None,
            dev_boundary_mask=dev_boundary_mask if args.use_boundary_loss else None,
            boundary_loss_weight=args.boundary_loss_weight if args.use_boundary_loss else 0.0,
            boundary_loss_pos_weight=args.boundary_loss_pos_weight,
            train_fv_labels=train_fv_labels if args.use_frame_violation_loss else None,
            train_fv_mask=train_fv_mask if args.use_frame_violation_loss else None,
            dev_fv_labels=dev_fv_labels if args.use_frame_violation_loss else None,
            dev_fv_mask=dev_fv_mask if args.use_frame_violation_loss else None,
            fv_loss_weight=args.frame_violation_loss_weight if args.use_frame_violation_loss else 0.0,
            fv_loss_pos_weight=args.frame_violation_loss_pos_weight,
            train_pi_labels=train_pi_labels if args.use_predicate_isolation_loss else None,
            train_pi_mask=train_pi_mask if args.use_predicate_isolation_loss else None,
            dev_pi_labels=dev_pi_labels if args.use_predicate_isolation_loss else None,
            dev_pi_mask=dev_pi_mask if args.use_predicate_isolation_loss else None,
            pi_loss_weight=(
                args.predicate_isolation_loss_weight
                if args.use_predicate_isolation_loss else 0.0
            ),
            pi_loss_pos_weight=args.predicate_isolation_loss_pos_weight,
            train_pe_labels=train_pe_labels if args.use_preservation_entitlement_loss else None,
            train_pe_mask=train_pe_mask if args.use_preservation_entitlement_loss else None,
            dev_pe_labels=dev_pe_labels if args.use_preservation_entitlement_loss else None,
            dev_pe_mask=dev_pe_mask if args.use_preservation_entitlement_loss else None,
            pe_loss_weight=(
                args.preservation_entitlement_loss_weight
                if args.use_preservation_entitlement_loss else 0.0
            ),
            pe_loss_pos_weight=args.preservation_entitlement_loss_pos_weight,
            td_train_inputs=_td_train_inputs if _td_data_needed else None,
            td_dev_inputs=_td_dev_inputs if _td_data_needed else None,
            td_train_labels=_td_train_labels if _td_data_needed else None,
            td_train_mask=_td_train_mask if _td_data_needed else None,
            td_dev_labels=_td_dev_labels if _td_data_needed else None,
            td_dev_mask=_td_dev_mask if _td_data_needed else None,
            td_loss_weight=(
                args.temporal_diagnostic_loss_weight
                if args.use_temporal_diagnostic_loss else 0.0
            ),
            td_loss_pos_weight=args.temporal_diagnostic_loss_pos_weight,
            use_td_constrained_selection=args.use_td_constrained_selection,
            td_sel_min_paraphrase_preserved=args.td_selection_min_paraphrase_preserved,
            td_sel_min_mismatch_recall=args.td_selection_min_mismatch_recall,
            td_sel_min_control_acceptance=args.td_selection_min_control_acceptance,
            use_td_final_decision_selection=args.td_selection_use_final_decision,
            td_sel_min_final_temporal_rejection=args.td_selection_min_final_temporal_rejection,
            td_sel_min_final_control_preservation=args.td_selection_min_final_control_preservation,
            use_temporal_residual_adapter=args.use_temporal_residual_adapter,
            ta_loss_weight=(
                args.temporal_adapter_loss_weight
                if args.use_temporal_adapter_loss else 0.0
            ),
            ta_loss_pos_weight=args.temporal_adapter_loss_pos_weight,
            ta_final_penalty_scale=(
                args.temporal_adapter_final_penalty_scale
                if args.use_temporal_adapter_final_penalty else 0.0
            ),
            use_temporal_channel=args.use_temporal_channel,
            tc_loss_weight=(
                args.temporal_channel_loss_weight
                if args.use_temporal_channel_loss else 0.0
            ),
            tc_loss_pos_weight=args.temporal_channel_loss_pos_weight,
            use_temporal_channel_gated_penalty=args.use_temporal_channel_gated_penalty,
            tc_gated_penalty_scale=(
                args.temporal_channel_gated_penalty_scale
                if args.use_temporal_channel_gated_penalty else 0.0
            ),
            use_preservation_constrained_selection=args.use_preservation_constrained_selection,
            sel_min_paraphrase_preserved=args.selection_min_paraphrase_preserved,
            sel_min_predicate_disentangled=args.selection_min_predicate_disentangled,
            sel_fallback=args.selection_fallback,
            pc_pres_inputs=_pc_pres_inputs if args.use_pair_contrastive_frame_loss else None,
            pc_frame_inputs=_pc_frame_inputs if args.use_pair_contrastive_frame_loss else None,
            pc_loss_weight=(
                args.pair_contrastive_frame_loss_weight
                if args.use_pair_contrastive_frame_loss else 0.0
            ),
            pc_margin=args.pair_contrastive_frame_margin,
            pc_valid_count=len(_pc_pair_records),
        )

    # Capture learned alphas (v6B-specific; v7 has no comparator alphas)
    alpha_temporal = (
        float(model.alpha_temporal().detach())
        if getattr(model, "alpha_temporal_raw", None) is not None else 0.0
    )
    alpha_predicate = (
        float(model.alpha_predicate().detach())
        if getattr(model, "alpha_predicate_raw", None) is not None else 0.0
    )
    temporal_flag_count = int(train_temporal_flags.sum().item())
    predicate_flag_count = int(train_predicate_flags.sum().item())

    report = {
        "configuration": {
            "seed": args.seed,
            "random_seed": args.seed,
            "numpy_seed": args.seed,
            "torch_seed": args.seed,
            "cuda_seed": args.seed if torch.cuda.is_available() else None,
            "data_seed": args.seed,
            "backbone": args.backbone,
            "model_name": args.model_name if args.backbone == "mamba" else None,
            "freeze_encoder": args.freeze_encoder,
            "freeze_a_log": args.freeze_a_log,
            "device": str(args.device),
            "allow_dummy_backbone": args.allow_dummy_backbone,
            "dummy_result_claim_policy": (
                "smoke_plumbing_only_not_claim_worthy"
                if args.backbone == "dummy"
                else "real_backbone_claim_candidate"
            ),
            "weighted_label_loss": args.weighted_label_loss,
            "balanced_sampler": args.balanced_sampler,
            "use_intervention_loss": args.use_intervention_loss,
            "loss_sweep": args.loss_sweep,
            "model_version": (
                "v7_hierarchical" if args.architecture == "v7_hierarchical" else "v6b_minimal"
            ),
            "architecture": args.architecture,
            "use_temporal_comparator": args.use_temporal_comparator,
            "use_predicate_comparator": args.use_predicate_comparator,
            "flag_source": args.flag_source,
            "alpha_temporal": alpha_temporal,
            "alpha_predicate": alpha_predicate,
            "temporal_flag_count": temporal_flag_count,
            "predicate_flag_count": predicate_flag_count,
            "final_logits_used": True,
            "time_swap_used": False,
            "pairwise_checks_skipped": args.smoke,
            "pairwise_checks_skip_reason": "incomplete smoke subset" if args.smoke else None,
            "class_weighting": args.class_weighting,
            "use_boundary_head": args.use_boundary_loss,
            "boundary_loss_weight": args.boundary_loss_weight if args.use_boundary_loss else 0.0,
            "boundary_loss_pos_weight": args.boundary_loss_pos_weight,
            "boundary_label_mapping": {
                "positive": sorted(_BOUNDARY_POSITIVE),
                "negative": sorted(_BOUNDARY_NEGATIVE),
                "excluded": "evidence_deletion,evidence_truncation,irrelevant_evidence,"
                            "polarity_flip,predicate_swap,unknown",
            },
            "use_frame_violation_head": args.use_frame_violation_loss,
            "frame_violation_loss_weight": (
                args.frame_violation_loss_weight if args.use_frame_violation_loss else 0.0
            ),
            "frame_violation_loss_pos_weight": args.frame_violation_loss_pos_weight,
            "frame_violation_label_mapping": {
                "positive": sorted(_FRAME_VIOLATION_POSITIVE),
                "negative": sorted(_FRAME_VIOLATION_NEGATIVE),
                "excluded": "predicate_swap,time_swap,unknown",
            },
            "use_pair_contrastive_frame_loss": args.use_pair_contrastive_frame_loss,
            "pair_contrastive_frame_data": args.pair_contrastive_frame_data,
            "pair_contrastive_frame_loss_weight": (
                args.pair_contrastive_frame_loss_weight
                if args.use_pair_contrastive_frame_loss else 0.0
            ),
            "pair_contrastive_frame_margin": args.pair_contrastive_frame_margin,
            "pair_contrastive_use_case": args.pair_contrastive_use_case,
            "pair_contrastive_valid_count": len(_pc_pair_records),
            "pair_contrastive_leakage_constraint": (
                "pair contrastive data must be constructed from controlled data only; "
                "Stage15 OOD records are not used for training"
            ),
            "use_predicate_isolation_head": args.use_predicate_isolation_loss,
            "predicate_isolation_loss_weight": (
                args.predicate_isolation_loss_weight
                if args.use_predicate_isolation_loss else 0.0
            ),
            "predicate_isolation_loss_pos_weight": args.predicate_isolation_loss_pos_weight,
            "predicate_isolation_label_mapping": {
                "positive": sorted(_PRED_ISOLATION_POSITIVE),
                "negative": sorted(_PRED_ISOLATION_NEGATIVE),
                "excluded": "entity_swap,event_swap,evidence_deletion,evidence_truncation,"
                            "irrelevant_evidence,location_swap,polarity_flip,role_swap,"
                            "title_name_swap,unknown",
            },
            "use_preservation_entitlement_head": args.use_preservation_entitlement_loss,
            "preservation_entitlement_loss_weight": (
                args.preservation_entitlement_loss_weight
                if args.use_preservation_entitlement_loss else 0.0
            ),
            "preservation_entitlement_loss_pos_weight": (
                args.preservation_entitlement_loss_pos_weight
            ),
            "preservation_entitlement_label_mapping": {
                "positive": sorted(_PRES_ENT_POSITIVE),
                "negative": sorted(_PRES_ENT_NEGATIVE),
                "excluded": "evidence_deletion,evidence_truncation,irrelevant_evidence,"
                            "polarity_flip,predicate_swap,unknown",
            },
            "use_temporal_diagnostic_head": args.use_temporal_diagnostic_loss,
            "temporal_diagnostic_data": args.temporal_diagnostic_data,
            "temporal_diagnostic_loss_weight": (
                args.temporal_diagnostic_loss_weight
                if args.use_temporal_diagnostic_loss else 0.0
            ),
            "temporal_diagnostic_loss_pos_weight": args.temporal_diagnostic_loss_pos_weight,
            "temporal_diagnostic_label_mapping": {
                "positive_role": _TEMPORAL_DIAG_MISMATCH_ROLE,
                "negative_role": _TEMPORAL_DIAG_CONTROL_ROLE,
                "positive_source_intervention_type": "time_swap",
                "negative_source_intervention_types": ["none", "paraphrase"],
                "note": (
                    "loaded from separate temporal diagnostic file; "
                    "time_swap is NOT in the main clean train/eval data"
                ),
            },
            "temporal_diagnostic_train_count": len(_td_train_records),
            "temporal_diagnostic_dev_count": len(_td_dev_records),
            "stage15_used_for_temporal_diagnostic_training": False,
            "use_td_constrained_selection": args.use_td_constrained_selection,
            "td_selection_min_paraphrase_preserved": args.td_selection_min_paraphrase_preserved,
            "td_selection_min_mismatch_recall": args.td_selection_min_mismatch_recall,
            "td_selection_min_control_acceptance": args.td_selection_min_control_acceptance,
            "td_selection_use_final_decision": args.td_selection_use_final_decision,
            "td_selection_min_final_temporal_rejection": (
                args.td_selection_min_final_temporal_rejection
            ),
            "td_selection_min_final_control_preservation": (
                args.td_selection_min_final_control_preservation
            ),
            "td_selection_fallback": "final_macro_f1",
            "stage15_used_for_td_constrained_selection": False,
            "use_temporal_residual_adapter": args.use_temporal_residual_adapter,
            "temporal_adapter_detach_input": args.temporal_adapter_detach_input,
            "use_temporal_adapter_loss": args.use_temporal_adapter_loss,
            "temporal_adapter_loss_weight": (
                args.temporal_adapter_loss_weight
                if args.use_temporal_adapter_loss else 0.0
            ),
            "temporal_adapter_loss_pos_weight": args.temporal_adapter_loss_pos_weight,
            "use_temporal_adapter_final_penalty": args.use_temporal_adapter_final_penalty,
            "temporal_adapter_final_penalty_scale": (
                args.temporal_adapter_final_penalty_scale
                if args.use_temporal_adapter_final_penalty else 0.0
            ),
            "stage15_used_for_temporal_adapter_training": False,
            "use_temporal_channel": args.use_temporal_channel,
            "temporal_channel_detach_input": args.temporal_channel_detach_input,
            "use_temporal_channel_loss": args.use_temporal_channel_loss,
            "temporal_channel_loss_weight": (
                args.temporal_channel_loss_weight
                if args.use_temporal_channel_loss else 0.0
            ),
            "temporal_channel_loss_pos_weight": args.temporal_channel_loss_pos_weight,
            "use_temporal_channel_gated_penalty": args.use_temporal_channel_gated_penalty,
            "temporal_channel_gated_penalty_scale": (
                args.temporal_channel_gated_penalty_scale
                if args.use_temporal_channel_gated_penalty else 0.0
            ),
            "stage15_used_for_temporal_channel_training": False,
            "stage15_used_for_temporal_channel_penalty_selection": False,
            # Stage26-A: v7 hierarchical architecture fields
            "architecture": args.architecture,
            "use_v7_hierarchical": args.architecture == "v7_hierarchical",
            "v7_disable_frame_channel": getattr(args, "v7_disable_frame_channel", False),
            "v7_disable_predicate_channel": getattr(args, "v7_disable_predicate_channel", False),
            "v7_disable_sufficiency_channel": getattr(args, "v7_disable_sufficiency_channel", False),
            "v7_disable_temporal_channel": getattr(args, "v7_disable_temporal_channel", False),
            "v7_flat_arbiter": getattr(args, "v7_flat_arbiter", False),
            "v7_no_entitlement_polarity_conditioning": getattr(
                args, "v7_no_entitlement_polarity_conditioning", False
            ),
            "v7_no_aux_losses": getattr(args, "v7_no_aux_losses", False),
            "v7_aux_losses_active": (
                args.architecture == "v7_hierarchical"
                and not getattr(args, "v7_no_aux_losses", False)
                and (
                    (getattr(args, "v7_use_polarity_margin_loss", False)
                     and getattr(args, "v7_polarity_margin_loss_weight", 0.0) > 0.0)
                    or (getattr(args, "v7_use_entitlement_bce_loss", False)
                        and getattr(args, "v7_entitlement_bce_loss_weight", 0.0) > 0.0)
                    or (getattr(args, "v7_use_entitled_class_balanced_ce", False)
                        and getattr(args, "v7_entitled_class_balanced_ce_weight", 0.0) > 0.0)
                )
            ),
            # Stage26-G: v7 stabilization options
            "v7_use_polarity_margin_loss": getattr(args, "v7_use_polarity_margin_loss", False),
            "v7_polarity_margin_loss_weight": getattr(args, "v7_polarity_margin_loss_weight", 0.0),
            "v7_polarity_margin": getattr(args, "v7_polarity_margin", 0.5),
            "v7_use_entitlement_bce_loss": getattr(args, "v7_use_entitlement_bce_loss", False),
            "v7_entitlement_bce_loss_weight": getattr(args, "v7_entitlement_bce_loss_weight", 0.0),
            "v7_entitlement_bce_pos_weight": getattr(args, "v7_entitlement_bce_pos_weight", 1.0),
            "v7_use_entitled_class_balanced_ce": getattr(
                args, "v7_use_entitled_class_balanced_ce", False
            ),
            "v7_entitled_class_balanced_ce_weight": getattr(
                args, "v7_entitled_class_balanced_ce_weight", 0.0
            ),
            "v7_initial_ne_bias": getattr(args, "v7_initial_ne_bias", -0.5),
            "v7_final_logit_composition": (
                "flat"
                if getattr(args, "v7_no_entitlement_polarity_conditioning", False)
                else "hierarchical_additive"
            ) if args.architecture == "v7_hierarchical" else None,
            # Stage26-B: emit the v7 output contract key list for traceability.
            # See V7_REQUIRED_OUTPUT_KEYS in modeling_v7_hierarchical.py for definition.
            "v7_channel_output_keys": (
                [
                    "v7_frame_logit", "v7_frame_prob",
                    "v7_predicate_logit", "v7_predicate_prob",
                    "v7_sufficiency_logit", "v7_sufficiency_prob",
                    "v7_entitlement_logit", "v7_entitlement_prob",
                    "v7_polarity_support_logit", "v7_polarity_refute_logit",
                    "v7_temporal_logit", "v7_temporal_prob",
                ]
                if args.architecture == "v7_hierarchical" else None
            ),
            "stage15_used_for_v7_training": False,
            "stage15_used_for_v7_selection": False,
            "stage15_used_for_v7_aux_loss_targets": False,
            "time_swap_used_in_v7_main_clean_data": False,
            "use_preservation_constrained_selection": args.use_preservation_constrained_selection,
            "selection_min_paraphrase_preserved": args.selection_min_paraphrase_preserved,
            "selection_min_predicate_disentangled": args.selection_min_predicate_disentangled,
            "selection_fallback": args.selection_fallback,
            "preservation_constrained_selection_used": (
                next(iter(reports.values()), {}).get(
                    "preservation_constrained_selection_used", False
                )
                if len(reports) == 1 else None
            ),
            "preservation_constrained_selection_fallback_used": (
                next(iter(reports.values()), {}).get(
                    "preservation_constrained_selection_fallback_used", False
                )
                if len(reports) == 1 else None
            ),
            "stage15_used_for_preservation_constrained_selection": False,
            "dev_calibrated_ne_shift_candidates": args.dev_calibrated_ne_shift_candidates,
            "dev_calibrated_ne_gate": args.dev_calibrated_ne_gate,
            "dev_calibrated_ne_threshold": args.dev_calibrated_ne_threshold,
            "dev_calibrated_ne_frame_penalty": args.dev_calibrated_ne_frame_penalty,
            "dev_calibrated_ne_calibration_source": args.dev_calibrated_ne_calibration_source,
            "dev_calibrated_ne_frame_penalty_candidates": args.dev_calibrated_ne_frame_penalty_candidates,
            "class_weights": ce_class_weights.tolist() if ce_class_weights is not None else None,
            "class_counts": label_counts,
            # Audit provenance summary in configuration (reflects single-run state when 1 run)
            "stage15_used_for_training": False,
            "stage15_used_for_loss_selection": False,
            "stage15_used_for_final_logit_modifier_selection": False,
            "stage15_used_for_checkpoint_selection": False,
            "time_swap_used_in_main_clean_data": False,
            "loss_component_epoch_avg_semantics": "weighted",
            "audit_ledger_note": (
                "active_training_losses, active_final_logit_modifiers, "
                "active_architectural_logit_components, active_selection_rules, "
                "loss_component_epoch_avg_raw/weighted, selected/final_epoch_loss_component_avg, "
                "aux_to_ce_loss_ratio_weighted/raw available in runs[*].audit_ledger and lifted "
                "to top-level when len(runs)==1"
            ),
        },
        "runs": reports,
    }

    prediction_exports = {
        name: run_report.pop("_best_dev_predictions")
        for name, run_report in reports.items()
    }
    _ood_best_state: dict[str, torch.Tensor] | None = None
    _ood_best_epoch: int = args.epochs
    if len(reports) == 1:
        _single_run = next(iter(reports.values()))
        _ood_best_state = _single_run.pop("_best_state", None)
        _ood_best_epoch = _single_run.get("best_epoch", args.epochs)
    else:
        for _rpt in reports.values():
            _rpt.pop("_best_state", None)

    if args.output_predictions_json is not None:
        if len(reports) != 1:
            parser.error("--output-predictions-json requires a single non-sweep run")
        run_name, run_report = next(iter(reports.items()))
        metadata = {
            "data_path": str(args.data),
            "seed": args.seed,
            "best_epoch": run_report["best_epoch"],
            "backbone": args.backbone,
            "model_name": args.model_name if args.backbone == "mamba" else None,
            "freeze_encoder": args.freeze_encoder,
            "weighted_label_loss": args.weighted_label_loss,
            "balanced_sampler": args.balanced_sampler,
            "use_intervention_loss": args.use_intervention_loss,
            "model_version": (
                "v7_hierarchical" if args.architecture == "v7_hierarchical" else "v6b_minimal"
            ),
            "architecture": args.architecture,
            "use_temporal_comparator": args.use_temporal_comparator,
            "use_predicate_comparator": args.use_predicate_comparator,
            "flag_source": args.flag_source,
            "alpha_temporal": alpha_temporal,
            "alpha_predicate": alpha_predicate,
            "final_logits_used": True,
        }
        v5.write_predictions_json(
            args.output_predictions_json,
            metadata,
            prediction_exports[run_name],
        )

    if len(reports) == 1:
        single = next(iter(reports.values()))
        for key in (
            "final_epoch",
            "best_epoch",
            "best_dev_metrics",
            "best_dev_interventions",
            "best_dev_pairwise_checks",
            # Stage26-F: per-epoch diagnostic history lifted to root for single-run reports
            "v7_epoch_diagnostic_history",
            # Stage26-F extended: v7 logit summaries, per-gold breakdown, collapse/recall fields
            "v7_best_dev_logit_summary",
            "v7_final_epoch_logit_summary",
            "v7_best_dev_per_gold_label_summary",
            "v7_predicted_single_class",
            "v7_predicted_majority_class",
            "v7_predicted_majority_fraction",
            "v7_support_prediction_count",
            "v7_refute_prediction_count",
            "v7_ne_prediction_count",
            "v7_support_recall",
            "v7_refute_recall",
            "v7_ne_recall",
            # TD constrained selection results — lifted from run report to top level
            "use_td_constrained_selection",
            "td_selection_min_paraphrase_preserved",
            "td_selection_min_mismatch_recall",
            "td_selection_min_control_acceptance",
            "td_selection_use_final_decision",
            "td_selection_min_final_temporal_rejection",
            "td_selection_min_final_control_preservation",
            "td_selection_fallback",
            "stage15_used_for_td_constrained_selection",
            "td_constrained_selection_used",
            "td_constrained_selection_fallback_used",
            "td_constrained_selection_reason",
            "td_constrained_selection_eligible_epoch_count",
            "td_constrained_selection_selected_epoch",
            "td_constrained_selection_best_clean_macro",
            "td_constrained_selection_selected_td_mismatch_recall",
            "td_constrained_selection_selected_td_control_acceptance",
            "td_constrained_selection_selected_paraphrase_preserved",
            "td_constrained_selection_selected_td_final_temporal_rejection_rate",
            "td_constrained_selection_selected_td_final_control_preservation_rate",
            "td_constrained_selection_selected_td_final_binary_accuracy",
            # Generic preservation-constrained selection results
            "use_preservation_constrained_selection",
            "selection_min_paraphrase_preserved",
            "selection_min_predicate_disentangled",
            "selection_fallback",
            "stage15_used_for_preservation_constrained_selection",
            "preservation_constrained_selection_used",
            "preservation_constrained_selection_fallback_used",
            "preservation_constrained_selection_reason",
            "preservation_constrained_selection_eligible_epoch_count",
            "preservation_constrained_selection_selected_epoch",
            "preservation_constrained_selection_selected_clean_macro",
            "preservation_constrained_selection_selected_paraphrase_preserved",
            "preservation_constrained_selection_selected_predicate_disentangled",
            # Audit ledger (nested dict; lifted from run report)
            "audit_ledger",
        ):
            if key in single:
                report[key] = single[key]

        # Also lift flat audit fields to top-level for easy reading
        _single_ledger = single.get("audit_ledger", {})
        for _audit_key in (
            "loss_component_epoch_avg",
            "loss_component_epoch_avg_raw",
            "loss_component_epoch_avg_weighted",
            "loss_component_epoch_avg_semantics",
            "selected_epoch_loss_component_avg",
            "selected_epoch_loss_component_avg_raw",
            "selected_epoch_loss_component_avg_weighted",
            "final_epoch_loss_component_avg",
            "final_epoch_loss_component_avg_raw",
            "final_epoch_loss_component_avg_weighted",
            "aux_weighted_loss_sum",
            "aux_to_ce_loss_ratio_weighted",
            "aux_raw_loss_sum",
            "aux_to_ce_loss_ratio_raw",
            "aux_to_ce_loss_ratio",
            "active_training_losses",
            "active_final_logit_modifiers",
            "active_architectural_logit_components",
            "active_selection_rules",
            "audit_warnings",
            "stage15_used_for_training",
            "stage15_used_for_loss_selection",
            "stage15_used_for_final_logit_modifier_selection",
            "stage15_used_for_checkpoint_selection",
            "stage15_used_for_temporal_channel_training",
            "stage15_used_for_temporal_channel_penalty_selection",
            "time_swap_used_in_main_clean_data",
            "stage15_used_for_v7_training",
            "stage15_used_for_v7_selection",
            "stage15_used_for_v7_aux_loss_targets",
            "time_swap_used_in_v7_main_clean_data",
        ):
            if _audit_key in _single_ledger:
                report[_audit_key] = _single_ledger[_audit_key]

    for run_name, run_report in reports.items():
        distribution = prediction_distribution_from_records(prediction_exports[run_name])
        if len(distribution) == 1:
            collapsed_label = next(iter(distribution))
            print(
                f"WARNING: run {run_name} dev predictions collapsed to "
                f"the single label {collapsed_label}",
                file=sys.stderr,
            )

    # ---------------------------------------------------------------------------
    # Stage22-G2/G3: dev-calibrated selective NE shift
    # Shift (and optionally frame penalty) is selected on controlled data only.
    # Stage15 OOD labels are NEVER consulted during selection.
    # G2: source=dev (default) — same pool as before.
    # G3: source=train or train_dev — larger pool without any OOD records.
    # ---------------------------------------------------------------------------
    _selected_dev_ne_shift: float | None = None
    _selected_dev_ne_frame_penalty: float | None = None
    _dev_cal_ne_result: dict[str, Any] | None = None

    if args.dev_calibrated_ne_shift_candidates is not None:
        _dc_ne_idx = v5.FINAL_LABEL_TO_ID.get("NOT_ENTITLED")
        if _dc_ne_idx is None:
            raise ValueError(
                "NOT_ENTITLED label index not found in FINAL_LABEL_TO_ID; "
                "cannot apply dev-calibrated NE shift"
            )
        # Restore best checkpoint (eval needs best weights, same as OOD block)
        if _ood_best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in _ood_best_state.items()})
        model.eval()

        _dc_cal_source = args.dev_calibrated_ne_calibration_source  # "dev", "train", "train_dev"
        _DC_PRES_ITYPES = frozenset({"none", "paraphrase"})
        _DC_FRAME_ITYPES = frozenset({
            "entity_swap", "event_swap", "location_swap", "role_swap", "title_name_swap",
        })
        _dc_aux_keys = ("frame_prob", "sufficiency_prob", "predicate_coverage_prob")

        # --- Run model on needed splits; collect (logits, records, flags) per split ---
        def _dc_run_split(
            inputs: dict, t_flags: "torch.Tensor", p_flags: "torch.Tensor"
        ) -> tuple["torch.Tensor", dict[str, "torch.Tensor | None"], "torch.Tensor"]:
            with torch.no_grad():
                _out = model(
                    **v5.model_feature_inputs(inputs),
                    temporal_mismatch_flags=t_flags,
                    predicate_mismatch_flags=p_flags,
                )
            _logits = _out["logits"].detach().cpu()
            _aux = {k: _out[k].detach().cpu() if k in _out else None for k in _dc_aux_keys}
            _t = t_flags.detach().cpu()
            _p = p_flags.detach().cpu()
            _unflagged = (_t == 0) & (_p == 0)
            return _logits, _aux, _unflagged

        if _dc_cal_source == "dev":
            _dc_logits, _dc_aux, _dc_unflagged = _dc_run_split(
                dev_inputs, dev_temporal_flags, dev_predicate_flags
            )
            _dc_records = dev_records
        elif _dc_cal_source == "train":
            _dc_logits, _dc_aux, _dc_unflagged = _dc_run_split(
                train_inputs, train_temporal_flags, train_predicate_flags
            )
            _dc_records = train_records
        else:  # train_dev
            _dc_tr_logits, _dc_tr_aux, _dc_tr_unflagged = _dc_run_split(
                train_inputs, train_temporal_flags, train_predicate_flags
            )
            _dc_dv_logits, _dc_dv_aux, _dc_dv_unflagged = _dc_run_split(
                dev_inputs, dev_temporal_flags, dev_predicate_flags
            )
            _dc_logits = torch.cat([_dc_tr_logits, _dc_dv_logits], dim=0)
            _dc_aux = {
                k: (
                    torch.cat([_dc_tr_aux[k], _dc_dv_aux[k]], dim=0)
                    if _dc_tr_aux[k] is not None and _dc_dv_aux[k] is not None
                    else None
                )
                for k in _dc_aux_keys
            }
            _dc_unflagged = torch.cat([_dc_tr_unflagged, _dc_dv_unflagged], dim=0)
            _dc_records = train_records + dev_records

        _dc_unflagged_count = int(_dc_unflagged.sum().item())
        _dc_gate_mask = _build_gate_mask(
            args.dev_calibrated_ne_gate,
            _dc_unflagged,
            _dc_aux,
            args.dev_calibrated_ne_threshold,
        )
        _dc_selected_count = int(_dc_gate_mask.sum().item())
        _dc_selected_rate = (
            _dc_selected_count / _dc_unflagged_count if _dc_unflagged_count > 0 else 0.0
        )

        # Record sets derived from intervention_type only — no Stage15 labels used
        _dc_pres_idx = [
            i for i, r in enumerate(_dc_records)
            if r.get("intervention_type", "") in _DC_PRES_ITYPES
        ]
        _dc_frame_idx = [
            i for i, r in enumerate(_dc_records)
            if r.get("intervention_type", "") in _DC_FRAME_ITYPES
        ]

        # Shift and penalty candidates (sorted ascending for tie-break ordering)
        _dc_shift_cands = sorted(
            float(s.strip()) for s in args.dev_calibrated_ne_shift_candidates.split(",")
        )
        _dc_penalty_cands: list[float]
        if args.dev_calibrated_ne_frame_penalty_candidates is not None:
            _dc_penalty_cands = sorted(
                float(p.strip())
                for p in args.dev_calibrated_ne_frame_penalty_candidates.split(",")
            )
        else:
            _dc_penalty_cands = [args.dev_calibrated_ne_frame_penalty]

        # Joint (penalty, shift) sweep — nested dict keyed by penalty then shift
        _dc_sweep: dict[str, Any] = {}
        # Best-so-far tracking; tie-breaks applied in order:
        # 1 higher score  2 lower frame_fe_rate  3 higher pres_rate  4 lower shift  5 higher penalty
        _dc_best_shift: float = _dc_shift_cands[0]
        _dc_best_penalty: float = _dc_penalty_cands[0]
        _dc_best_score: float = float("-inf")
        _dc_best_fe_rate: float = 1.0
        _dc_best_pres_rate: float = 0.0

        _dc_pres_total = len(_dc_pres_idx)
        _dc_frame_total = len(_dc_frame_idx)

        for _dc_penalty in _dc_penalty_cands:
            _dc_penalty_key = f"penalty={_dc_penalty:g}"
            _dc_sweep[_dc_penalty_key] = {}
            for _dc_shift in _dc_shift_cands:
                _dc_adj = _dc_logits.clone()
                if _dc_shift != 0.0:
                    _dc_adj[_dc_gate_mask, _dc_ne_idx] -= _dc_shift
                _dc_preds = _dc_adj.argmax(dim=-1)

                _dc_pres_accept = sum(
                    1 for i in _dc_pres_idx if _dc_preds[i].item() != _dc_ne_idx
                )
                _dc_pres_rate = (
                    _dc_pres_accept / _dc_pres_total if _dc_pres_total > 0 else 0.0
                )
                _dc_frame_fe = sum(
                    1 for i in _dc_frame_idx if _dc_preds[i].item() != _dc_ne_idx
                )
                _dc_frame_fe_rate = (
                    _dc_frame_fe / _dc_frame_total if _dc_frame_total > 0 else 0.0
                )
                _dc_score = _dc_pres_rate - _dc_penalty * _dc_frame_fe_rate

                _dc_sweep[_dc_penalty_key][f"shift={_dc_shift:g}"] = {
                    "shift": _dc_shift,
                    "frame_penalty": _dc_penalty,
                    "pres_accept_rate": _dc_pres_rate,
                    "frame_false_entitled_rate": _dc_frame_fe_rate,
                    "pres_total": _dc_pres_total,
                    "frame_total": _dc_frame_total,
                    "objective_score": _dc_score,
                }

                # Tie-break order: (1) higher score; (2) lower fe_rate; (3) higher pres_rate;
                # (4) lower shift (ascending loop handles this); (5) higher penalty (desc loop
                # would handle — instead we check explicitly).
                def _dc_beats_best() -> bool:
                    if _dc_score > _dc_best_score:
                        return True
                    if _dc_score < _dc_best_score:
                        return False
                    if _dc_frame_fe_rate < _dc_best_fe_rate:
                        return True
                    if _dc_frame_fe_rate > _dc_best_fe_rate:
                        return False
                    if _dc_pres_rate > _dc_best_pres_rate:
                        return True
                    if _dc_pres_rate < _dc_best_pres_rate:
                        return False
                    if _dc_shift < _dc_best_shift:
                        return True
                    if _dc_shift > _dc_best_shift:
                        return False
                    # shift equal: higher penalty wins
                    return _dc_penalty > _dc_best_penalty

                if _dc_beats_best():
                    _dc_best_score = _dc_score
                    _dc_best_fe_rate = _dc_frame_fe_rate
                    _dc_best_pres_rate = _dc_pres_rate
                    _dc_best_shift = _dc_shift
                    _dc_best_penalty = _dc_penalty

        _selected_dev_ne_shift = _dc_best_shift
        _selected_dev_ne_frame_penalty = _dc_best_penalty
        _dc_cal_source_label = f"controlled_{_dc_cal_source}_only"
        _dev_cal_ne_result = {
            "selected_dev_calibrated_ne_shift": _selected_dev_ne_shift,
            "selected_dev_calibrated_ne_frame_penalty": _selected_dev_ne_frame_penalty,
            "dev_calibrated_ne_shift_candidates": args.dev_calibrated_ne_shift_candidates,
            "dev_calibrated_ne_frame_penalty_candidates": (
                args.dev_calibrated_ne_frame_penalty_candidates
            ),
            "dev_calibrated_ne_gate": args.dev_calibrated_ne_gate,
            "dev_calibrated_ne_threshold": args.dev_calibrated_ne_threshold,
            "dev_calibrated_ne_calibration_source": _dc_cal_source,
            "calibration_source": _dc_cal_source_label,
            "stage15_used_for_shift_selection": False,
            "best_objective_score": _dc_best_score,
            "calibration_pres_record_count": _dc_pres_total,
            "calibration_frame_record_count": _dc_frame_total,
            "calibration_unflagged_count": _dc_unflagged_count,
            "calibration_selected_count": _dc_selected_count,
            "calibration_selected_rate": _dc_selected_rate,
            "shift_sweep": _dc_sweep,
        }
        report["dev_calibrated_ne_shift"] = _selected_dev_ne_shift
        report["dev_calibrated_ne_frame_penalty"] = _selected_dev_ne_frame_penalty
        report["dev_calibrated_ne_calibration_source"] = _dc_cal_source
        report["dev_calibrated_ne_shift_sweep"] = _dev_cal_ne_result
        print(
            f"[DEV CAL NE SHIFT G3] source={_dc_cal_source} "
            f"gate={args.dev_calibrated_ne_gate} "
            f"thr={args.dev_calibrated_ne_threshold} "
            f"selected_shift={_selected_dev_ne_shift:g} "
            f"selected_penalty={_selected_dev_ne_frame_penalty:g} "
            f"score={_dc_best_score:.4f} "
            f"selected={_dc_selected_count}/{_dc_unflagged_count} "
            f"pres_n={_dc_pres_total} frame_n={_dc_frame_total}"
        )

    if args.ood_data is not None:
        ood_flag_source = args.ood_flag_source if args.ood_flag_source is not None else args.flag_source
        if _ood_best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in _ood_best_state.items()})
            model.eval()
            ood_eval_state = "best_dev"
            ood_eval_epoch = _ood_best_epoch
        else:
            ood_eval_state = "final_epoch_fallback"
            ood_eval_epoch = args.epochs
        print(
            f"[OOD EVAL] loading {args.ood_data} flag_source={ood_flag_source} "
            f"ood_eval_state={ood_eval_state} epoch={ood_eval_epoch}"
        )
        ood_records = load_ood_jsonl(args.ood_data)
        if args.backbone == "dummy":
            ood_bundle = v5.encode_records(ood_records, vocab)
        else:
            ood_bundle = v5.encode_mamba_records(ood_records, tokenizer, args.max_length)
        ood_inputs = v5.move_inputs(ood_bundle["model_inputs"], device)
        ood_seq_len = ood_inputs["input_ids"].shape[1]
        if ood_seq_len < max_length:
            for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                ood_inputs[key] = F.pad(ood_inputs[key], (0, max_length - ood_seq_len), value=0)
        elif ood_seq_len > max_length:
            for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                ood_inputs[key] = ood_inputs[key][:, :max_length]
        ood_temporal_flags, ood_predicate_flags = extract_flags(
            ood_records, ood_flag_source, device
        )

        # Provenance block written to every --output-ood-json file so the backbone
        # audit script can classify OOD-only JSONs without the main seed JSON.
        # This is logging only — no model behavior, loss, or logit change.
        _ood_provenance: dict[str, Any] = {
            "backbone": args.backbone,
            "freeze_encoder": getattr(args, "freeze_encoder", None),
            "freeze_a_log": getattr(args, "freeze_a_log", None),
            "model_name": getattr(args, "model_name", None),
            "device": str(args.device),
            "allow_dummy_backbone": args.allow_dummy_backbone,
            "dummy_result_claim_policy": (
                "smoke_plumbing_only_not_claim_worthy"
                if args.backbone == "dummy"
                else "real_backbone_claim_candidate"
            ),
            "seed": args.seed,
            "data": str(args.data),
            "ood_data": str(args.ood_data),
            "ood_flag_source": ood_flag_source,
            "use_boundary_loss": getattr(args, "use_boundary_loss", False),
            "boundary_loss_weight": getattr(args, "boundary_loss_weight", 0.0),
            "use_frame_violation_loss": getattr(args, "use_frame_violation_loss", False),
            "frame_violation_loss_weight": getattr(args, "frame_violation_loss_weight", 0.0),
            "use_predicate_isolation_loss": getattr(args, "use_predicate_isolation_loss", False),
            "use_predicate_isolation_head": getattr(args, "use_predicate_isolation_loss", False),
            "predicate_isolation_loss_weight": getattr(
                args, "predicate_isolation_loss_weight", 0.0
            ),
            "predicate_isolation_loss_pos_weight": getattr(
                args, "predicate_isolation_loss_pos_weight", 2.0
            ),
            "use_preservation_entitlement_loss": getattr(
                args, "use_preservation_entitlement_loss", False
            ),
            "use_preservation_entitlement_head": getattr(
                args, "use_preservation_entitlement_loss", False
            ),
            "preservation_entitlement_loss_weight": getattr(
                args, "preservation_entitlement_loss_weight", 0.0
            ),
            "preservation_entitlement_loss_pos_weight": getattr(
                args, "preservation_entitlement_loss_pos_weight", 1.0
            ),
            "use_temporal_diagnostic_loss": getattr(
                args, "use_temporal_diagnostic_loss", False
            ),
            "use_temporal_diagnostic_head": getattr(
                args, "use_temporal_diagnostic_loss", False
            ),
            "temporal_diagnostic_data": getattr(args, "temporal_diagnostic_data", None),
            "temporal_diagnostic_loss_weight": getattr(
                args, "temporal_diagnostic_loss_weight", 0.0
            ),
            "temporal_diagnostic_loss_pos_weight": getattr(
                args, "temporal_diagnostic_loss_pos_weight", 2.0
            ),
            "stage15_used_for_temporal_diagnostic_training": False,
            "use_td_constrained_selection": getattr(
                args, "use_td_constrained_selection", False
            ),
            "td_selection_use_final_decision": getattr(
                args, "td_selection_use_final_decision", False
            ),
            "td_constrained_selection_used": (
                next(iter(reports.values()), {}).get("td_constrained_selection_used", False)
                if len(reports) == 1 else None
            ),
            "td_constrained_selection_fallback_used": (
                next(iter(reports.values()), {}).get(
                    "td_constrained_selection_fallback_used", False
                )
                if len(reports) == 1 else None
            ),
            "stage15_used_for_td_constrained_selection": False,
            "use_temporal_residual_adapter": getattr(
                args, "use_temporal_residual_adapter", False
            ),
            "temporal_adapter_detach_input": getattr(
                args, "temporal_adapter_detach_input", True
            ),
            "use_temporal_adapter_loss": getattr(
                args, "use_temporal_adapter_loss", False
            ),
            "temporal_adapter_loss_weight": getattr(
                args, "temporal_adapter_loss_weight", 0.0
            ),
            "temporal_adapter_loss_pos_weight": getattr(
                args, "temporal_adapter_loss_pos_weight", 2.0
            ),
            "use_temporal_adapter_final_penalty": getattr(
                args, "use_temporal_adapter_final_penalty", False
            ),
            "temporal_adapter_final_penalty_scale": getattr(
                args, "temporal_adapter_final_penalty_scale", 0.0
            ),
            "stage15_used_for_temporal_adapter_training": False,
            "use_temporal_channel": getattr(args, "use_temporal_channel", False),
            "temporal_channel_detach_input": getattr(
                args, "temporal_channel_detach_input", True
            ),
            "use_temporal_channel_loss": getattr(args, "use_temporal_channel_loss", False),
            "temporal_channel_loss_weight": getattr(
                args, "temporal_channel_loss_weight", 0.0
            ),
            "temporal_channel_loss_pos_weight": getattr(
                args, "temporal_channel_loss_pos_weight", 1.0
            ),
            "use_temporal_channel_gated_penalty": getattr(
                args, "use_temporal_channel_gated_penalty", False
            ),
            "temporal_channel_gated_penalty_scale": getattr(
                args, "temporal_channel_gated_penalty_scale", 0.0
            ),
            "stage15_used_for_temporal_channel_training": False,
            "stage15_used_for_temporal_channel_penalty_selection": False,
            # Stage26-A: v7 hierarchical architecture fields (OOD-sweep config)
            "architecture": getattr(args, "architecture", "v6b_minimal"),
            "use_v7_hierarchical": getattr(args, "architecture", "v6b_minimal") == "v7_hierarchical",
            "v7_disable_frame_channel": getattr(args, "v7_disable_frame_channel", False),
            "v7_disable_predicate_channel": getattr(args, "v7_disable_predicate_channel", False),
            "v7_disable_sufficiency_channel": getattr(args, "v7_disable_sufficiency_channel", False),
            "v7_disable_temporal_channel": getattr(args, "v7_disable_temporal_channel", False),
            "v7_flat_arbiter": getattr(args, "v7_flat_arbiter", False),
            "v7_no_entitlement_polarity_conditioning": getattr(
                args, "v7_no_entitlement_polarity_conditioning", False
            ),
            "v7_no_aux_losses": getattr(args, "v7_no_aux_losses", False),
            "v7_aux_losses_active": False,
            # Stage26-B: emit the v7 output contract key list for traceability (OOD-sweep config).
            "v7_channel_output_keys": (
                [
                    "v7_frame_logit", "v7_frame_prob",
                    "v7_predicate_logit", "v7_predicate_prob",
                    "v7_sufficiency_logit", "v7_sufficiency_prob",
                    "v7_entitlement_logit", "v7_entitlement_prob",
                    "v7_polarity_support_logit", "v7_polarity_refute_logit",
                    "v7_temporal_logit", "v7_temporal_prob",
                ]
                if getattr(args, "architecture", "v6b_minimal") == "v7_hierarchical" else None
            ),
            "stage15_used_for_v7_training": False,
            "stage15_used_for_v7_selection": False,
            "stage15_used_for_v7_aux_loss_targets": False,
            "time_swap_used_in_v7_main_clean_data": False,
            "use_preservation_constrained_selection": getattr(
                args, "use_preservation_constrained_selection", False
            ),
            "selection_min_paraphrase_preserved": getattr(
                args, "selection_min_paraphrase_preserved", 0.70
            ),
            "selection_min_predicate_disentangled": getattr(
                args, "selection_min_predicate_disentangled", 0.85
            ),
            "selection_fallback": getattr(args, "selection_fallback", "final_macro_f1"),
            "preservation_constrained_selection_used": (
                next(iter(reports.values()), {}).get(
                    "preservation_constrained_selection_used", False
                )
                if len(reports) == 1 else None
            ),
            "preservation_constrained_selection_fallback_used": (
                next(iter(reports.values()), {}).get(
                    "preservation_constrained_selection_fallback_used", False
                )
                if len(reports) == 1 else None
            ),
            "stage15_used_for_preservation_constrained_selection": False,
            "use_pair_contrastive_frame_loss": getattr(
                args, "use_pair_contrastive_frame_loss", False
            ),
            "pair_contrastive_frame_data": getattr(
                args, "pair_contrastive_frame_data", None
            ),
            "pair_contrastive_use_case": getattr(
                args, "pair_contrastive_use_case", None
            ),
            "pair_contrastive_valid_count": len(_pc_pair_records),
            "pair_contrastive_frame_loss_weight": getattr(
                args, "pair_contrastive_frame_loss_weight", 0.0
            ),
            "pair_contrastive_frame_margin": getattr(
                args, "pair_contrastive_frame_margin", 0.0
            ),
            "dev_calibrated_ne_shift_candidates": args.dev_calibrated_ne_shift_candidates,
            "dev_calibrated_ne_frame_penalty_candidates": (
                args.dev_calibrated_ne_frame_penalty_candidates
            ),
            "dev_calibrated_ne_calibration_source": args.dev_calibrated_ne_calibration_source,
            "dev_calibrated_ne_gate": args.dev_calibrated_ne_gate,
            "dev_calibrated_ne_threshold": args.dev_calibrated_ne_threshold,
            "dev_calibrated_ne_frame_penalty": args.dev_calibrated_ne_frame_penalty,
            "selected_dev_calibrated_ne_shift": _selected_dev_ne_shift,
            "selected_dev_calibrated_ne_frame_penalty": _selected_dev_ne_frame_penalty,
            "calibration_source": (
                f"controlled_{args.dev_calibrated_ne_calibration_source}_only"
                if args.dev_calibrated_ne_shift_candidates is not None
                else None
            ),
            "stage15_used_for_shift_selection": (
                False if args.dev_calibrated_ne_shift_candidates is not None else None
            ),
            # Audit provenance in OOD block
            "stage15_used_for_training": False,
            "stage15_used_for_loss_selection": False,
            "stage15_used_for_final_logit_modifier_selection": False,
            "stage15_used_for_checkpoint_selection": False,
            "time_swap_used_in_main_clean_data": False,
            "ood_tuned_ne_shift_active": (
                args.ood_unflagged_ne_shift_sweep is not None
                or args.ood_selective_ne_shift_sweep is not None
            ),
            "ood_tuned_ne_shift_diagnostic_only": True,
        }

        ablation_modes = (
            [m.strip() for m in args.ood_ablation_modes.split(",")]
            if args.ood_ablation_modes is not None
            else None
        )
        ne_shift_vals = (
            [float(s.strip()) for s in args.ood_unflagged_ne_shift_sweep.split(",")]
            if args.ood_unflagged_ne_shift_sweep is not None
            else None
        )
        selective_shift_vals = (
            [float(s.strip()) for s in args.ood_selective_ne_shift_sweep.split(",")]
            if args.ood_selective_ne_shift_sweep is not None
            else None
        )
        _active_sweep_count = sum([
            ablation_modes is not None,
            ne_shift_vals is not None,
            selective_shift_vals is not None,
        ])
        if _active_sweep_count > 1:
            raise ValueError(
                "--ood-ablation-modes, --ood-unflagged-ne-shift-sweep, and "
                "--ood-selective-ne-shift-sweep are mutually exclusive; "
                "use at most one per invocation"
            )

        if selective_shift_vals is not None:
            # --- Selective gate NE shift sweep branch ---
            ne_idx = v5.FINAL_LABEL_TO_ID.get("NOT_ENTITLED")
            if ne_idx is None:
                raise ValueError(
                    "NOT_ENTITLED label index not found in FINAL_LABEL_TO_ID; "
                    "cannot apply selective NE logit shift"
                )
            model.eval()
            with torch.no_grad():
                ood_output = model(
                    **v5.model_feature_inputs(ood_inputs),
                    temporal_mismatch_flags=ood_temporal_flags,
                    predicate_mismatch_flags=ood_predicate_flags,
                )
            logits_cpu = ood_output["logits"].detach().cpu()
            labels_cpu = ood_inputs["final_labels"].detach().cpu()
            t_cpu = ood_temporal_flags.detach().cpu()
            p_cpu = ood_predicate_flags.detach().cpu()
            unflagged_mask = (t_cpu == 0) & (p_cpu == 0)
            unflagged_count = int(unflagged_mask.sum().item())
            temporal_flag_count = int(ood_temporal_flags.sum().item())
            predicate_flag_count = int(ood_predicate_flags.sum().item())
            _aux_keys = ("frame_prob", "sufficiency_prob", "predicate_coverage_prob")
            aux_probs: dict[str, torch.Tensor | None] = {
                k: ood_output[k].detach().cpu() if k in ood_output else None
                for k in _aux_keys
            }
            gate_names = (
                [g.strip() for g in args.ood_selective_ne_gates.split(",")]
                if args.ood_selective_ne_gates is not None
                else ["high_sufficiency", "high_frame_sufficiency", "high_frame_suff_predicate"]
            )
            thresholds = (
                [float(t_val.strip()) for t_val in args.ood_selective_ne_thresholds.split(",")]
                if args.ood_selective_ne_thresholds is not None
                else [0.7]
            )
            selective_sweep: dict[str, Any] = {}
            for gate_name in gate_names:
                for threshold in thresholds:
                    gate_mask = _build_gate_mask(
                        gate_name, unflagged_mask, aux_probs, threshold
                    )
                    selected_count = int(gate_mask.sum().item())
                    selected_rate = (
                        selected_count / unflagged_count if unflagged_count > 0 else 0.0
                    )
                    for shift in selective_shift_vals:
                        cond_key = (
                            f"gate={gate_name}|thr={threshold:.2f}|shift={shift:g}"
                        )
                        cond_summary = _apply_ne_shift_and_eval(
                            logits_cpu, labels_cpu, gate_mask, ood_records, ne_idx, shift
                        )
                        cond_summary["ood_eval_state"] = ood_eval_state
                        cond_summary["ood_eval_epoch"] = ood_eval_epoch
                        cond_summary["ood_flag_source"] = ood_flag_source
                        cond_summary["train_flag_source"] = args.flag_source
                        cond_summary["selective_gate"] = gate_name
                        cond_summary["selective_threshold"] = threshold
                        cond_summary["unflagged_ne_shift"] = shift
                        cond_summary["temporal_flag_count"] = temporal_flag_count
                        cond_summary["predicate_flag_count"] = predicate_flag_count
                        cond_summary["unflagged_count"] = unflagged_count
                        cond_summary["selected_count"] = selected_count
                        cond_summary["selected_rate_among_unflagged"] = selected_rate
                        selective_sweep[cond_key] = cond_summary
                        print(
                            f"[OOD SELECTIVE] {cond_key} "
                            f"selected={selected_count}/{unflagged_count} "
                            f"accuracy={cond_summary['overall_metrics']['final_accuracy']:.4f}"
                        )
            report["ood_selective_ne_shift_sweep"] = selective_sweep
            if args.output_ood_json is not None:
                v5.write_report_json(
                    {
                        "ood_provenance": _ood_provenance,
                        "ood_selective_ne_shift_sweep": selective_sweep,
                    },
                    args.output_ood_json,
                )

        elif ne_shift_vals is not None:
            # --- NE shift sweep branch ---
            ne_idx = v5.FINAL_LABEL_TO_ID.get("NOT_ENTITLED")
            if ne_idx is None:
                raise ValueError(
                    "NOT_ENTITLED label index not found in FINAL_LABEL_TO_ID; "
                    "cannot apply NE logit shift"
                )
            # single forward pass with full OOD flags
            model.eval()
            with torch.no_grad():
                ood_output = model(
                    **v5.model_feature_inputs(ood_inputs),
                    temporal_mismatch_flags=ood_temporal_flags,
                    predicate_mismatch_flags=ood_predicate_flags,
                )
            logits_cpu = ood_output["logits"].detach().cpu()
            labels_cpu = ood_inputs["final_labels"].detach().cpu()
            t_cpu = ood_temporal_flags.detach().cpu()
            p_cpu = ood_predicate_flags.detach().cpu()
            unflagged_mask = (t_cpu == 0) & (p_cpu == 0)
            unflagged_count = int(unflagged_mask.sum().item())
            temporal_flag_count = int(ood_temporal_flags.sum().item())
            predicate_flag_count = int(ood_predicate_flags.sum().item())

            shift_sweep: dict[str, Any] = {}
            for shift in ne_shift_vals:
                shift_key = f"{shift:g}"
                shift_summary = _apply_ne_shift_and_eval(
                    logits_cpu, labels_cpu, unflagged_mask, ood_records, ne_idx, shift
                )
                shift_summary["ood_eval_state"] = ood_eval_state
                shift_summary["ood_eval_epoch"] = ood_eval_epoch
                shift_summary["ood_flag_source"] = ood_flag_source
                shift_summary["train_flag_source"] = args.flag_source
                shift_summary["unflagged_ne_shift"] = shift
                shift_summary["temporal_flag_count"] = temporal_flag_count
                shift_summary["predicate_flag_count"] = predicate_flag_count
                shift_summary["unflagged_count"] = unflagged_count
                shift_sweep[shift_key] = shift_summary
                print(
                    f"[OOD NE SHIFT] shift={shift_key} unflagged={unflagged_count} "
                    f"accuracy={shift_summary['overall_metrics']['final_accuracy']:.4f}"
                )
            report["ood_unflagged_ne_shift_sweep"] = shift_sweep
            if args.output_ood_json is not None:
                v5.write_report_json(
                    {
                        "ood_provenance": _ood_provenance,
                        "ood_unflagged_ne_shift_sweep": shift_sweep,
                    },
                    args.output_ood_json,
                )

        elif ablation_modes is not None:
            # --- ablation branch: evaluate each flag mode, build ood_ablation dict ---
            ood_ablation: dict[str, Any] = {}
            for mode in ablation_modes:
                abl_temporal, abl_predicate = _make_ablation_flags(
                    mode, ood_temporal_flags, ood_predicate_flags
                )
                abl_summary, _ = evaluate_ood_v6b(
                    model, ood_records, ood_inputs, abl_temporal, abl_predicate
                )
                abl_summary["ood_eval_state"] = ood_eval_state
                abl_summary["ood_eval_epoch"] = ood_eval_epoch
                abl_summary["ood_flag_source"] = ood_flag_source
                abl_summary["train_flag_source"] = args.flag_source
                abl_summary["ood_ablation_mode"] = mode
                abl_summary["temporal_flag_count"] = int(abl_temporal.sum().item())
                abl_summary["predicate_flag_count"] = int(abl_predicate.sum().item())
                ood_ablation[mode] = abl_summary
                print(
                    f"[OOD ABLATION] mode={mode} "
                    f"temporal_flags={abl_summary['temporal_flag_count']} "
                    f"predicate_flags={abl_summary['predicate_flag_count']} "
                    f"accuracy={abl_summary['overall_metrics']['final_accuracy']:.4f}"
                )
            report["ood_ablation"] = ood_ablation
            if args.output_ood_json is not None:
                v5.write_report_json(
                    {
                        "ood_provenance": _ood_provenance,
                        "ood_ablation": ood_ablation,
                    },
                    args.output_ood_json,
                )

        else:
            # --- single-mode branch: existing behaviour, unchanged ---
            ood_summary, ood_predictions = evaluate_ood_v6b(
                model, ood_records, ood_inputs, ood_temporal_flags, ood_predicate_flags
            )
            ood_summary["train_flag_source"] = args.flag_source
            ood_summary["ood_flag_source"] = ood_flag_source
            ood_summary["ood_eval_state"] = ood_eval_state
            ood_summary["ood_eval_epoch"] = ood_eval_epoch
            report["ood_evaluation"] = ood_summary
            if args.output_ood_json is not None:
                v5.write_report_json(
                    {"ood_provenance": _ood_provenance, **ood_summary},
                    args.output_ood_json,
                )
            if args.output_ood_predictions_json is not None:
                ood_metadata = {
                    "ood_data_path": str(args.ood_data),
                    "seed": args.seed,
                    "backbone": args.backbone,
                    "model_version": (
                        "v7_hierarchical" if args.architecture == "v7_hierarchical" else "v6b_minimal"
                    ),
                    "architecture": args.architecture,
                    "train_flag_source": args.flag_source,
                    "ood_flag_source": ood_flag_source,
                    "final_logits_used": True,
                }
                v5.write_predictions_json(
                    args.output_ood_predictions_json, ood_metadata, ood_predictions
                )

        # Stage22-G2: apply dev-calibrated shift to OOD (always runs if shift was selected,
        # independent of which sweep branch above was active; adds ood_dev_calibrated_ne_shift
        # to the main report — Stage15 OOD labels never used for shift selection)
        if _selected_dev_ne_shift is not None:
            _dc_ood_ne_idx = v5.FINAL_LABEL_TO_ID.get("NOT_ENTITLED")
            if _dc_ood_ne_idx is None:
                raise ValueError(
                    "NOT_ENTITLED label index not found; "
                    "cannot apply dev-calibrated NE shift to OOD"
                )
            model.eval()
            with torch.no_grad():
                _dc_ood_out = model(
                    **v5.model_feature_inputs(ood_inputs),
                    temporal_mismatch_flags=ood_temporal_flags,
                    predicate_mismatch_flags=ood_predicate_flags,
                )
            _dc_ood_logits = _dc_ood_out["logits"].detach().cpu()
            _dc_ood_labels = ood_inputs["final_labels"].detach().cpu()
            _dc_ood_t = ood_temporal_flags.detach().cpu()
            _dc_ood_p = ood_predicate_flags.detach().cpu()
            _dc_ood_unflagged = (_dc_ood_t == 0) & (_dc_ood_p == 0)
            _dc_ood_unflagged_count = int(_dc_ood_unflagged.sum().item())
            _dc_ood_aux: dict[str, torch.Tensor | None] = {
                k: _dc_ood_out[k].detach().cpu() if k in _dc_ood_out else None
                for k in ("frame_prob", "sufficiency_prob", "predicate_coverage_prob")
            }
            _dc_ood_gate_mask = _build_gate_mask(
                args.dev_calibrated_ne_gate,
                _dc_ood_unflagged,
                _dc_ood_aux,
                args.dev_calibrated_ne_threshold,
            )
            _dc_ood_selected_count = int(_dc_ood_gate_mask.sum().item())
            _dc_ood_selected_rate = (
                _dc_ood_selected_count / _dc_ood_unflagged_count
                if _dc_ood_unflagged_count > 0
                else 0.0
            )
            _dc_ood_eval = _apply_ne_shift_and_eval(
                _dc_ood_logits,
                _dc_ood_labels,
                _dc_ood_gate_mask,
                ood_records,
                _dc_ood_ne_idx,
                _selected_dev_ne_shift,
            )
            _dc_ood_eval.update({
                "ood_eval_state": ood_eval_state,
                "ood_eval_epoch": ood_eval_epoch,
                "ood_flag_source": ood_flag_source,
                "selected_shift": _selected_dev_ne_shift,
                "selected_frame_penalty": _selected_dev_ne_frame_penalty,
                "gate": args.dev_calibrated_ne_gate,
                "threshold": args.dev_calibrated_ne_threshold,
                "dev_calibrated_ne_calibration_source": args.dev_calibrated_ne_calibration_source,
                "dev_objective_score": (
                    _dev_cal_ne_result["best_objective_score"]
                    if _dev_cal_ne_result is not None else None
                ),
                "calibration_pres_record_count": (
                    _dev_cal_ne_result["calibration_pres_record_count"]
                    if _dev_cal_ne_result is not None else None
                ),
                "calibration_frame_record_count": (
                    _dev_cal_ne_result["calibration_frame_record_count"]
                    if _dev_cal_ne_result is not None else None
                ),
                "calibration_unflagged_count": (
                    _dev_cal_ne_result["calibration_unflagged_count"]
                    if _dev_cal_ne_result is not None else None
                ),
                "selected_count": _dc_ood_selected_count,
                "selected_rate": _dc_ood_selected_rate,
                "calibration_source": (
                    f"controlled_{args.dev_calibrated_ne_calibration_source}_only"
                ),
                "stage15_used_for_shift_selection": False,
            })
            report["ood_dev_calibrated_ne_shift"] = _dc_ood_eval
            print(
                f"[OOD DEV CAL NE SHIFT G3] shift={_selected_dev_ne_shift:g} "
                f"penalty={_selected_dev_ne_frame_penalty:g} "
                f"gate={args.dev_calibrated_ne_gate} "
                f"source={args.dev_calibrated_ne_calibration_source} "
                f"selected={_dc_ood_selected_count}/{_dc_ood_unflagged_count} "
                f"accuracy={_dc_ood_eval['overall_metrics']['final_accuracy']:.4f}"
            )

    # Stage26-D: lift architecture metadata and dev metric aliases to root level.
    lift_report_aliases(report)

    print("\nFINAL_REPORT")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json is not None:
        v5.write_report_json(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
