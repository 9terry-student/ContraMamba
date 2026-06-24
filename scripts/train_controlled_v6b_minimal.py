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
    )


def build_mamba_model(
    model_name: str,
    freeze_encoder: bool,
    freeze_a_log: bool,
    use_boundary_head: bool = False,
    use_frame_violation_head: bool = False,
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


# ---------------------------------------------------------------------------
# Stage22-A4c: pair-group contrastive frame helpers
# ---------------------------------------------------------------------------

_PC_USE_CASES = frozenset({
    "frame_violation_contrastive",
    "support_safe_frame_contrastive",
    "all",
})


def load_pair_contrastive_jsonl(
    path: Path,
    *,
    use_case_filter: str = "frame_violation_contrastive",
) -> list[dict[str, Any]]:
    """Load pair contrastive JSONL and filter by contrastive_use_case.

    use_case_filter="all" keeps all records regardless of contrastive_use_case.
    Stage15 OOD records are never loaded here; this file must have been constructed
    from controlled data only (leakage_note = "constructed_from_controlled_data_only").
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if use_case_filter == "all":
        return records
    return [r for r in records if r.get("contrastive_use_case") == use_case_filter]


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
        choices=("frame_violation_contrastive", "support_safe_frame_contrastive", "all"),
        default="frame_violation_contrastive",
        help=(
            "Filter pair contrastive records by contrastive_use_case field. "
            "frame_violation_contrastive: frame valid + pres non-frame (may be REFUTE). "
            "support_safe_frame_contrastive: strict subset where pres is SUPPORT-safe. "
            "all: use all records regardless of use_case. "
            "Default: frame_violation_contrastive."
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
        model = build_mamba_model(
            args.model_name,
            freeze_encoder=args.freeze_encoder,
            freeze_a_log=args.freeze_a_log,
            use_boundary_head=args.use_boundary_loss,
            use_frame_violation_head=args.use_frame_violation_loss,
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
        model = build_model(
            len(vocab), max_length,
            use_boundary_head=args.use_boundary_loss,
            use_frame_violation_head=args.use_frame_violation_loss,
        )
    model = model.to(device)

    if args.backbone == "mamba" and args.freeze_encoder:
        print("Caching frozen Mamba token states for train/dev...")
        v5.cache_frozen_encoder_states(model, train_inputs)
        v5.cache_frozen_encoder_states(model, dev_inputs)

    print(
        f"controlled v6b_minimal | backbone={args.backbone} "
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

    # Stage22-A4c: pair contrastive frame data loading and encoding
    _pc_pair_records: list[dict[str, Any]] = []
    _pc_pres_inputs: "dict[str, torch.Tensor] | None" = None
    _pc_frame_inputs: "dict[str, torch.Tensor] | None" = None
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
        print(
            f"[pc22a4c] loaded {len(_pc_pair_records)} pair records"
            f" use_case={args.pair_contrastive_use_case}"
            f" weight={args.pair_contrastive_frame_loss_weight}"
            f" margin={args.pair_contrastive_frame_margin}"
        )
        if _pc_pair_records:
            _pc_pres_virtual = [
                {
                    "claim": r["claim"],
                    "evidence": r["preservation_evidence"],
                    "final_label": r.get("preservation_final_label") or "SUPPORT",
                }
                for r in _pc_pair_records
            ]
            _pc_frame_virtual = [
                {
                    "claim": r["claim"],
                    "evidence": r["frame_evidence"],
                    "final_label": r.get("frame_final_label") or "NOT_ENTITLED",
                }
                for r in _pc_pair_records
            ]
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
    ):
        """Modified run_training that passes flags to v6b model."""
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
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

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            # CRITICAL: Pass flags to v6b model forward
            output = model(
                **v5.model_feature_inputs(train_inputs),
                temporal_mismatch_flags=train_temporal_flags,
                predicate_mismatch_flags=train_predicate_flags,
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
                )
                dev_output = model(
                    **dev_inputs,
                    temporal_mismatch_flags=dev_temporal_flags,
                    predicate_mismatch_flags=dev_predicate_flags,
                )
            train_metrics = v5.compute_metrics(train_output, train_inputs)
            dev_metrics = v5.compute_metrics(dev_output, dev_inputs)

            # Stage22-A: boundary head metrics
            _tbm = compute_boundary_metrics(train_output, train_boundary_labels, train_boundary_mask)
            _dbm = compute_boundary_metrics(dev_output, dev_boundary_labels, dev_boundary_mask)
            # Stage22-A3: frame violation head metrics
            _tfvm = compute_frame_violation_metrics(train_output, train_fv_labels, train_fv_mask)
            _dfvm = compute_frame_violation_metrics(dev_output, dev_fv_labels, dev_fv_mask)

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
                best_pc_metrics = _pc_eval_metrics
                if capture_best_trainable_state:
                    best_trainable_state = v5.capture_trainable_state(model)

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
            if _pc_eval_metrics:
                _pc_loss_val = float(_pc_loss.item()) if hasattr(_pc_loss, "item") else 0.0
                print(
                    f"  [pc22a4c] loss={_pc_loss_val:.4f}"
                    f" acc={_pc_eval_metrics.get('pair_contrastive_frame_accuracy', 0.0):.3f}"
                    f" margin={_pc_eval_metrics.get('pair_contrastive_frame_margin_mean', 0.0):.3f}"
                    f" pres_prob={_pc_eval_metrics.get('pair_contrastive_frame_mean_pres_fv_prob', 0.0):.3f}"
                    f" frame_prob={_pc_eval_metrics.get('pair_contrastive_frame_mean_frame_fv_prob', 0.0):.3f}"
                    f" n={_pc_eval_metrics.get('pair_contrastive_frame_valid_count', 0)}"
                )

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
            pc_pres_inputs=_pc_pres_inputs if args.use_pair_contrastive_frame_loss else None,
            pc_frame_inputs=_pc_frame_inputs if args.use_pair_contrastive_frame_loss else None,
            pc_loss_weight=(
                args.pair_contrastive_frame_loss_weight
                if args.use_pair_contrastive_frame_loss else 0.0
            ),
            pc_margin=args.pair_contrastive_frame_margin,
            pc_valid_count=len(_pc_pair_records),
        )

    # Capture learned alphas
    alpha_temporal = float(model.alpha_temporal().detach()) if model.alpha_temporal_raw else 0.0
    alpha_predicate = float(model.alpha_predicate().detach()) if model.alpha_predicate_raw else 0.0
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
            "weighted_label_loss": args.weighted_label_loss,
            "balanced_sampler": args.balanced_sampler,
            "use_intervention_loss": args.use_intervention_loss,
            "loss_sweep": args.loss_sweep,
            "model_version": "v6b_minimal",
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
            "class_weights": ce_class_weights.tolist() if ce_class_weights is not None else None,
            "class_counts": label_counts,
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
            "model_version": "v6b_minimal",
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
        ):
            report[key] = single[key]

    for run_name, run_report in reports.items():
        distribution = prediction_distribution_from_records(prediction_exports[run_name])
        if len(distribution) == 1:
            collapsed_label = next(iter(distribution))
            print(
                f"WARNING: run {run_name} dev predictions collapsed to "
                f"the single label {collapsed_label}",
                file=sys.stderr,
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
                    {"ood_selective_ne_shift_sweep": selective_sweep}, args.output_ood_json
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
                    {"ood_unflagged_ne_shift_sweep": shift_sweep}, args.output_ood_json
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
                v5.write_report_json({"ood_ablation": ood_ablation}, args.output_ood_json)

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
                v5.write_report_json(ood_summary, args.output_ood_json)
            if args.output_ood_predictions_json is not None:
                ood_metadata = {
                    "ood_data_path": str(args.ood_data),
                    "seed": args.seed,
                    "backbone": args.backbone,
                    "model_version": "v6b_minimal",
                    "train_flag_source": args.flag_source,
                    "ood_flag_source": ood_flag_source,
                    "final_logits_used": True,
                }
                v5.write_predictions_json(
                    args.output_ood_predictions_json, ood_metadata, ood_predictions
                )

    print("\nFINAL_REPORT")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json is not None:
        v5.write_report_json(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
