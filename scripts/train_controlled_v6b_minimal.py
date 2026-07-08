"""Train ContraMamba-v6B-minimal on controlled intervention data.

Minimal v6B wrapper: reuses v5 training infrastructure, adds temporal/predicate
comparator alphas with learnable scaling. No composer, no product_final_loss.
All CE/pairwise/intervention losses consume final calibrated logits.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch import nn
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
from contramamba.modeling_vnext_minimal import ContraMambaVNextMinimal  # noqa: E402
from scripts import train_controlled_v5 as v5  # noqa: E402
from scripts.stage43_external_factver_eval_utils import (  # noqa: E402
    analyze_stage43_predictions,
    build_aggregate_report,
    load_stage43_jsonl,
    render_aggregate_markdown,
    render_report_markdown,
    stage43_rows_to_controlled_records,
    summarize_numeric,
    write_json,
    write_jsonl,
    write_text,
)
from scripts.stage45_internal_family_utils import split_leave_family_out  # noqa: E402
from scripts.stage45_support_recovery_utils import (  # noqa: E402
    build_stage45c_report,
    compute_support_recovery_terms,
    label_counts as stage45c_label_counts,
)

STAGE31C_COVERAGE_LABELS = [
    "ENTAILS_SUPPORT",
    "OVERCLAIM_NOT_ENTITLED",
    "CONTRADICTS_REFUTE",
    "OTHER_RESIDUAL",
]
STAGE31C_COVERAGE_LABEL_TO_ID = {
    label: idx for idx, label in enumerate(STAGE31C_COVERAGE_LABELS)
}
STAGE31C_DEFAULT_NUM_CLASSES = 3
STAGE31C_INPUT_MODES = ("current", "raw_pair", "hybrid")
VNEXT_EVIDENCE_INTERFACE_CHOICES: tuple[str, ...] = (
    "full_evidence",
    "core_only",
    "core_first_context_suffix",
    "context_prefix_core",
    "core_marker_context_suffix",
    "segmented_dual_pass_scaffold",
    "segmented_dual_pass",
)
STAGE118_DIAGNOSTIC_EVIDENCE_INTERFACE_CHOICES: tuple[str, ...] = (
    "same_as_vnext",
    *VNEXT_EVIDENCE_INTERFACE_CHOICES,
)
STAGE128_LOCATION_SLOT_GUARD_MODES: tuple[str, ...] = (
    "off",
    "controlled_in_during_location_mismatch",
)
_STAGE128_IN_DURING_LOCATION_RE = re.compile(
    r"\bin\s+([A-Z][A-Za-z]*(?:[ -][A-Z][A-Za-z]*)*)\s+during\b"
)
_STAGE128_LOCATION_SLOT_GUARD_EXPORT_FIELDS: tuple[str, ...] = (
    "stage128_location_slot_guard_enabled",
    "stage128_location_slot_guard_mode",
    "stage128_claim_location",
    "stage128_evidence_location",
    "stage128_location_mismatch",
    "stage128_prediction_before_location_guard",
    "stage128_prediction_after_location_guard",
    "stage128_location_guard_applied",
    "stage128_location_guard_notes",
)


class Stage31CCoverageEntailmentHead(nn.Module):
    """Readout-only directional Coverage/Entailment diagnostic head."""

    def __init__(
        self,
        frame_size: int,
        predicate_size: int,
        sufficiency_size: int,
        num_classes: int = STAGE31C_DEFAULT_NUM_CLASSES,
        input_mode: str = "current",
        detach_input: bool = False,
    ) -> None:
        super().__init__()
        if num_classes not in (3, 4):
            raise ValueError(f"Stage31-C coverage entailment num_classes must be 3 or 4, got {num_classes}")
        if input_mode not in STAGE31C_INPUT_MODES:
            raise ValueError(
                f"Stage31-C coverage entailment input_mode must be one of "
                f"{STAGE31C_INPUT_MODES}, got {input_mode!r}"
            )
        current_size = frame_size + predicate_size + sufficiency_size
        raw_pair_size = frame_size * 4
        if input_mode == "current":
            input_size = current_size
        elif input_mode == "raw_pair":
            input_size = raw_pair_size
        else:
            input_size = raw_pair_size + current_size
        hidden = max(input_size // 2, 16)
        self.num_classes = num_classes
        self.input_mode = input_mode
        self.detach_input = detach_input
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def _current_features(self, output: dict[str, Any]) -> torch.Tensor:
        return torch.cat(
            [
                output["frame_pair_repr"],
                output["predicate_pair_repr"],
                output["sufficiency_repr"],
            ],
            dim=-1,
        )

    def _raw_pair_features(self, output: dict[str, Any]) -> torch.Tensor:
        claim_repr = output["claim_frame_state"]
        evidence_repr = output["evidence_frame_state"]
        return torch.cat(
            [
                claim_repr,
                evidence_repr,
                torch.abs(claim_repr - evidence_repr),
                claim_repr * evidence_repr,
            ],
            dim=-1,
        )

    def forward(self, output: dict[str, Any]) -> torch.Tensor:
        if self.input_mode == "current":
            features = self._current_features(output)
        elif self.input_mode == "raw_pair":
            features = self._raw_pair_features(output)
        else:
            features = torch.cat(
                [self._raw_pair_features(output), self._current_features(output)],
                dim=-1,
            )
        if self.detach_input:
            features = features.detach()
        return self.mlp(features)


def _stage31c_module_output_size(module: nn.Module, classifier_name: str) -> int:
    classifier = getattr(module, classifier_name)
    return int(classifier.in_features)


def install_stage31c_coverage_entailment_head(
    model: nn.Module,
    *,
    num_classes: int = STAGE31C_DEFAULT_NUM_CLASSES,
    input_mode: str = "current",
    detach_input: bool,
) -> None:
    """Register and wrap a diagnostic head without changing final logits."""
    if getattr(model, "stage31c_coverage_entailment_head", None) is not None:
        return
    frame_size = _stage31c_module_output_size(model.frame_gate, "frame_classifier")
    predicate_size = _stage31c_module_output_size(
        model.predicate_coverage_head, "coverage_classifier"
    )
    sufficiency_size = _stage31c_module_output_size(model.sufficiency_gate, "classifier")
    model.stage31c_coverage_entailment_head = Stage31CCoverageEntailmentHead(
        frame_size=frame_size,
        predicate_size=predicate_size,
        sufficiency_size=sufficiency_size,
        num_classes=num_classes,
        input_mode=input_mode,
        detach_input=detach_input,
    )
    original_forward = model.forward

    def _forward_with_stage31c(*args: Any, **kwargs: Any) -> dict[str, Any]:
        output = original_forward(*args, **kwargs)
        return add_stage31c_coverage_entailment_outputs(model, output)

    model.forward = _forward_with_stage31c  # type: ignore[method-assign]


def add_stage31c_coverage_entailment_outputs(
    model: nn.Module,
    output: dict[str, Any],
) -> dict[str, Any]:
    head = getattr(model, "stage31c_coverage_entailment_head", None)
    if head is None:
        return output
    input_mode = getattr(head, "input_mode", "current")
    required = ["frame_pair_repr", "predicate_pair_repr", "sufficiency_repr"]
    if input_mode in ("raw_pair", "hybrid"):
        required.extend(["claim_frame_state", "evidence_frame_state"])
    if any(output.get(key) is None for key in required):
        return output
    logits = head(output)
    probs = torch.softmax(logits, dim=-1)
    pred_id = probs.argmax(dim=-1)
    confidence = probs.max(dim=-1).values
    output["coverage_entailment_logits"] = logits
    output["coverage_entails_support_logit"] = logits[:, 0]
    output["coverage_overclaim_ne_logit"] = logits[:, 1]
    output["coverage_contradicts_refute_logit"] = logits[:, 2]
    output["coverage_entails_support_prob"] = probs[:, 0]
    output["coverage_overclaim_ne_prob"] = probs[:, 1]
    output["coverage_contradicts_refute_prob"] = probs[:, 2]
    if logits.shape[-1] > 3:
        output["coverage_other_residual_logit"] = logits[:, 3]
        output["coverage_other_residual_prob"] = probs[:, 3]
    output["coverage_entailment_pred_id"] = pred_id
    labels = STAGE31C_COVERAGE_LABELS[: int(logits.shape[-1])]
    output["coverage_entailment_pred_label"] = [
        labels[int(i)] for i in pred_id.detach().cpu().tolist()
    ]
    output["coverage_entailment_confidence"] = confidence
    output["coverage_entailment_input_mode"] = input_mode
    return output


def load_stage31c_coverage_entailment_jsonl(path: Path) -> list[dict]:
    if path.name == "stage31_coverage_entailment_probe.jsonl":
        raise ValueError(
            "Stage31-C coverage-entailment loss must not use the Stage31-A/B "
            "evaluation probe. Use data/stage31c_coverage_entailment_aux.jsonl."
        )
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("coverage_direction_label")
            label_id = row.get("coverage_direction_id")
            split = row.get("split")
            if label not in STAGE31C_COVERAGE_LABEL_TO_ID:
                raise ValueError(
                    f"[stage31c] row {lineno} invalid coverage_direction_label={label!r}"
                )
            if label_id != STAGE31C_COVERAGE_LABEL_TO_ID[label]:
                raise ValueError(
                    f"[stage31c] row {lineno} coverage_direction_id={label_id!r} "
                    f"does not match label {label!r}"
                )
            if split not in ("train", "dev"):
                raise ValueError(f"[stage31c] row {lineno} invalid split={split!r}")
            records.append(row)
    return records


def encode_stage31c_coverage_entailment_labels(
    records: list[dict],
    device: torch.device,
    num_classes: int = STAGE31C_DEFAULT_NUM_CLASSES,
) -> torch.Tensor:
    labels = [int(r["coverage_direction_id"]) for r in records]
    invalid = [label for label in labels if label < 0 or label >= num_classes]
    if invalid:
        raise ValueError(
            f"[stage31c] coverage_direction_id values {sorted(set(invalid))} "
            f"are invalid for --v7-coverage-entailment-num-classes {num_classes}."
        )
    return torch.tensor(
        labels,
        dtype=torch.long,
        device=device,
    )


def _stage31c_macro_f1(
    labels: list[int],
    preds: list[int],
    num_classes: int,
) -> tuple[float, dict[str, Any]]:
    per_class: dict[str, Any] = {}
    f1s: list[float] = []
    active_labels = STAGE31C_COVERAGE_LABELS[:num_classes]
    for idx, name in enumerate(active_labels):
        tp = sum(g == idx and p == idx for g, p in zip(labels, preds))
        fp = sum(g != idx and p == idx for g, p in zip(labels, preds))
        fn = sum(g == idx and p != idx for g, p in zip(labels, preds))
        support = sum(g == idx for g in labels)
        predicted = sum(p == idx for p in preds)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
            "predicted": predicted,
        }
    return (sum(f1s) / len(f1s) if f1s else 0.0), per_class


def compute_stage31c_coverage_entailment_metrics(
    output: dict[str, Any],
    labels: torch.Tensor,
) -> dict[str, Any]:
    logits = output.get("coverage_entailment_logits")
    if logits is None or labels is None or labels.numel() == 0:
        return {}
    preds_t = logits.detach().argmax(dim=-1).cpu()
    labels_t = labels.detach().cpu()
    labels_l = [int(x) for x in labels_t.tolist()]
    preds_l = [int(x) for x in preds_t.tolist()]
    accuracy = sum(g == p for g, p in zip(labels_l, preds_l)) / len(labels_l)
    num_classes = int(logits.shape[-1])
    active_labels = STAGE31C_COVERAGE_LABELS[:num_classes]
    macro, per_class = _stage31c_macro_f1(labels_l, preds_l, num_classes)
    confusion = {
        gold: {pred: 0 for pred in active_labels}
        for gold in active_labels
    }
    for gold, pred in zip(labels_l, preds_l):
        confusion[STAGE31C_COVERAGE_LABELS[gold]][STAGE31C_COVERAGE_LABELS[pred]] += 1
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro, 4),
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def _stage44b_clean_dev_selection_metrics(
    *,
    epoch: int,
    score: float,
    dev_metrics: dict[str, Any],
    dev_inputs: dict[str, torch.Tensor],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    labels_cpu = dev_inputs["final_labels"].detach().cpu().tolist()
    gold_counts = {
        label_name: sum(
            1
            for label_id in labels_cpu
            if v5.ID_TO_FINAL_LABEL[int(label_id)] == label_name
        )
        for label_name in ("SUPPORT", "REFUTE", "NOT_ENTITLED")
    }
    total_gold = sum(gold_counts.values())
    gold_rates = {
        label_name: (count / total_gold if total_gold else None)
        for label_name, count in gold_counts.items()
    }
    prediction_counts = {
        label_name: int((dev_metrics.get("prediction_distribution") or {}).get(label_name, 0))
        for label_name in ("SUPPORT", "REFUTE", "NOT_ENTITLED")
    }
    total_predictions = sum(prediction_counts.values())
    per_label = dev_metrics.get("per_label") or {}
    support_recall = per_label.get("SUPPORT", {}).get("recall")
    refute_recall = per_label.get("REFUTE", {}).get("recall")
    support_precision = per_label.get("SUPPORT", {}).get("precision")
    refute_precision = per_label.get("REFUTE", {}).get("precision")
    ne_pred_rate = (
        prediction_counts["NOT_ENTITLED"] / total_predictions
        if total_predictions
        else None
    )
    gold_ne_rate = gold_rates.get("NOT_ENTITLED")
    ne_pred_minus_gold_ne_rate = (
        ne_pred_rate - gold_ne_rate
        if ne_pred_rate is not None and gold_ne_rate is not None
        else None
    )
    accuracy = dev_metrics.get("final_accuracy")
    macro_f1 = dev_metrics.get("final_macro_f1")
    prior_aware_enabled = bool(constraints.get("use_prior_aware_ne_constraint"))
    prior_delta = constraints.get("max_ne_gold_prior_delta")
    checks = {
        "min_support_recall": (
            True if constraints.get("min_support_recall") is None
            else (support_recall is not None and support_recall >= constraints["min_support_recall"])
        ),
        "min_refute_recall": (
            True if constraints.get("min_refute_recall") is None
            else (refute_recall is not None and refute_recall >= constraints["min_refute_recall"])
        ),
        "max_not_entitled_pred_rate": (
            True if constraints.get("max_not_entitled_pred_rate") is None
            else (ne_pred_rate is not None and ne_pred_rate <= constraints["max_not_entitled_pred_rate"])
        ),
        "prior_aware_ne_pred_rate": (
            True if (not prior_aware_enabled or prior_delta is None)
            else (
                ne_pred_rate is not None
                and gold_ne_rate is not None
                and ne_pred_rate <= gold_ne_rate + prior_delta
            )
        ),
        "min_clean_dev_accuracy": (
            True if constraints.get("min_clean_dev_accuracy") is None
            else (accuracy is not None and accuracy >= constraints["min_clean_dev_accuracy"])
        ),
        "min_macro_f1": (
            True if (not prior_aware_enabled or constraints.get("min_macro_f1") is None)
            else (macro_f1 is not None and macro_f1 >= constraints["min_macro_f1"])
        ),
        "min_support_precision": (
            True if (not prior_aware_enabled or constraints.get("min_support_precision") is None)
            else (support_precision is not None and support_precision >= constraints["min_support_precision"])
        ),
        "min_refute_precision": (
            True if (not prior_aware_enabled or constraints.get("min_refute_precision") is None)
            else (refute_precision is not None and refute_precision >= constraints["min_refute_precision"])
        ),
    }
    return {
        "epoch": epoch,
        "score": score,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "per_label": per_label,
        "prediction_counts": prediction_counts,
        "gold_label_counts": gold_counts,
        "gold_label_rates": gold_rates,
        "not_entitled_prediction_rate": ne_pred_rate,
        "stage44b2_ne_pred_minus_gold_ne_rate": ne_pred_minus_gold_ne_rate,
        "support_recall": support_recall,
        "refute_recall": refute_recall,
        "support_precision": support_precision,
        "refute_precision": refute_precision,
        "constraint_checks": checks,
        "constraints_satisfied": all(bool(value) for value in checks.values()),
    }

def parse_stage31c_class_weights(
    raw: str | None,
    device: torch.device,
    num_classes: int = STAGE31C_DEFAULT_NUM_CLASSES,
) -> "torch.Tensor | None":
    if raw is None:
        return None
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != num_classes:
        raise ValueError(
            "--v7-coverage-entailment-loss-class-weights must contain exactly "
            f"{num_classes} comma-separated values."
        )
    return torch.tensor(values, dtype=torch.float32, device=device)


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



_VNEXT_STAGE124_125_MODEL_KWARGS: tuple[str, ...] = (
    "vnext_enable_segmented_dual_pass",
    "vnext_segmented_context_role",
    "vnext_context_risk_cap_alpha",
    "vnext_context_risk_threshold",
    "vnext_context_risk_source",
)


def _construct_vnext_minimal_with_aligned_kwargs(
    kwargs: dict[str, Any],
) -> ContraMambaVNextMinimal:
    signature = inspect.signature(ContraMambaVNextMinimal.__init__)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_var_kwargs:
        ctor_kwargs = dict(kwargs)
    else:
        ctor_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    dropped = sorted(set(kwargs) - set(ctor_kwargs))
    unexpected = [
        key for key in dropped
        if key not in _VNEXT_STAGE124_125_MODEL_KWARGS
    ]
    if unexpected:
        raise TypeError(
            "ContraMambaVNextMinimal.__init__ does not accept builder kwargs: "
            + ", ".join(unexpected)
        )
    model = ContraMambaVNextMinimal(**ctor_kwargs)
    model.vnext_enable_segmented_dual_pass = bool(
        kwargs.get("vnext_enable_segmented_dual_pass", False)
    )
    model.vnext_segmented_context_role = str(
        kwargs.get("vnext_segmented_context_role", "diagnostic_only")
    )
    model.vnext_context_risk_cap_alpha = float(
        kwargs.get("vnext_context_risk_cap_alpha", 0.0)
    )
    model.vnext_context_risk_threshold = float(
        kwargs.get("vnext_context_risk_threshold", 0.5)
    )
    model.vnext_context_risk_source = str(
        kwargs.get("vnext_context_risk_source", "context_not_entitled_prob")
    )
    return model

def build_vnext_model(
    vocab_size: int,
    max_length: int,
    hidden_size: int = 48,
    vnext_router_mode: str = "learned_x_product",
    vnext_enable_segmented_dual_pass: bool = False,
    vnext_segmented_context_role: str = "diagnostic_only",
    vnext_context_risk_cap_alpha: float = 0.0,
    vnext_context_risk_threshold: float = 0.5,
    vnext_context_risk_source: str = "context_not_entitled_prob",
) -> ContraMambaVNextMinimal:
    """Build a ContraMambaVNextMinimal with dummy backbone for plumbing validation."""
    backbone = v5.ControlledDummyBackbone(vocab_size, hidden_size, max_length)
    return _construct_vnext_minimal_with_aligned_kwargs({
        "backbone": backbone,
        "frame_size": 32,
        "predicate_size": 32,
        "sufficiency_size": 32,
        "energy_size": 24,
        "dropout": 0.0,
        "vnext_router_mode": vnext_router_mode,
        "vnext_enable_segmented_dual_pass": vnext_enable_segmented_dual_pass,
        "vnext_segmented_context_role": vnext_segmented_context_role,
        "vnext_context_risk_cap_alpha": vnext_context_risk_cap_alpha,
        "vnext_context_risk_threshold": vnext_context_risk_threshold,
        "vnext_context_risk_source": vnext_context_risk_source,
    })


def build_vnext_mamba_model(
    model_name: str,
    freeze_encoder: bool,
    freeze_a_log: bool,
    vnext_router_mode: str = "learned_x_product",
    vnext_enable_segmented_dual_pass: bool = False,
    vnext_segmented_context_role: str = "diagnostic_only",
    vnext_context_risk_cap_alpha: float = 0.0,
    vnext_context_risk_threshold: float = 0.5,
    vnext_context_risk_source: str = "context_not_entitled_prob",
) -> ContraMambaVNextMinimal:
    """Build a ContraMambaVNextMinimal with real Mamba backbone."""
    model = _construct_vnext_minimal_with_aligned_kwargs({
        "model_name": model_name,
        "frame_size": 128,
        "predicate_size": 128,
        "sufficiency_size": 128,
        "energy_size": 64,
        "dropout": 0.1,
        "freeze_a_log": freeze_a_log,
        "vnext_router_mode": vnext_router_mode,
        "vnext_enable_segmented_dual_pass": vnext_enable_segmented_dual_pass,
        "vnext_segmented_context_role": vnext_segmented_context_role,
        "vnext_context_risk_cap_alpha": vnext_context_risk_cap_alpha,
        "vnext_context_risk_threshold": vnext_context_risk_threshold,
        "vnext_context_risk_source": vnext_context_risk_source,
    })
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
    v7_use_v6b_style_final_decision: bool = False,
    v7_use_learnable_ne_alpha: bool = False,
    v7_ne_alpha_init: float = 1.0,
    v7_h1_entitlement_decision_signal: str = "learned",
    v7_h1_entitlement_product_power: float = 1.0,
    v7_h1_hybrid_residual_beta: float = 0.25,
    v7_use_location_boundary_head: bool = False,
    v7_location_boundary_cap_mode: str = "none",
    v7_location_boundary_cap_gamma: float = 1.0,
    v7_location_boundary_cap_detach: bool = False,
    v7_use_temporal_safety_head: bool = False,
    v7_temporal_safety_cap_mode: str = "none",
    v7_temporal_safety_cap_gamma: float = 1.0,
    v7_temporal_safety_cap_detach: bool = False,
    v7_use_temporal_mismatch_multihead: bool = False,
    v7_temporal_mismatch_multihead_cap_mode: str = "none",
    v7_temporal_mismatch_multihead_cap_gamma: float = 1.0,
    v7_temporal_mismatch_multihead_cap_detach: bool = False,
    v7_temporal_mismatch_multihead_fusion: str = "frame_only",
    v7_use_temporal_preservation_head: bool = False,
    v7_use_temporal_preservation_aware_cap: bool = False,
    v7_temporal_preservation_cap_gamma: float = 1.0,
    v7_temporal_preservation_cap_detach: bool = False,
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
        v7_use_v6b_style_final_decision=v7_use_v6b_style_final_decision,
        v7_use_learnable_ne_alpha=v7_use_learnable_ne_alpha,
        v7_ne_alpha_init=v7_ne_alpha_init,
        v7_h1_entitlement_decision_signal=v7_h1_entitlement_decision_signal,
        v7_h1_entitlement_product_power=v7_h1_entitlement_product_power,
        v7_h1_hybrid_residual_beta=v7_h1_hybrid_residual_beta,
        v7_use_location_boundary_head=v7_use_location_boundary_head,
        v7_location_boundary_cap_mode=v7_location_boundary_cap_mode,
        v7_location_boundary_cap_gamma=v7_location_boundary_cap_gamma,
        v7_location_boundary_cap_detach=v7_location_boundary_cap_detach,
        v7_use_temporal_safety_head=v7_use_temporal_safety_head,
        v7_temporal_safety_cap_mode=v7_temporal_safety_cap_mode,
        v7_temporal_safety_cap_gamma=v7_temporal_safety_cap_gamma,
        v7_temporal_safety_cap_detach=v7_temporal_safety_cap_detach,
        v7_use_temporal_mismatch_multihead=v7_use_temporal_mismatch_multihead,
        v7_temporal_mismatch_multihead_cap_mode=v7_temporal_mismatch_multihead_cap_mode,
        v7_temporal_mismatch_multihead_cap_gamma=v7_temporal_mismatch_multihead_cap_gamma,
        v7_temporal_mismatch_multihead_cap_detach=v7_temporal_mismatch_multihead_cap_detach,
        v7_temporal_mismatch_multihead_fusion=v7_temporal_mismatch_multihead_fusion,
        v7_use_temporal_preservation_head=v7_use_temporal_preservation_head,
        v7_use_temporal_preservation_aware_cap=v7_use_temporal_preservation_aware_cap,
        v7_temporal_preservation_cap_gamma=v7_temporal_preservation_cap_gamma,
        v7_temporal_preservation_cap_detach=v7_temporal_preservation_cap_detach,
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
    v7_use_v6b_style_final_decision: bool = False,
    v7_use_learnable_ne_alpha: bool = False,
    v7_ne_alpha_init: float = 1.0,
    v7_h1_entitlement_decision_signal: str = "learned",
    v7_h1_entitlement_product_power: float = 1.0,
    v7_h1_hybrid_residual_beta: float = 0.25,
    v7_use_location_boundary_head: bool = False,
    v7_location_boundary_cap_mode: str = "none",
    v7_location_boundary_cap_gamma: float = 1.0,
    v7_location_boundary_cap_detach: bool = False,
    v7_use_temporal_safety_head: bool = False,
    v7_temporal_safety_cap_mode: str = "none",
    v7_temporal_safety_cap_gamma: float = 1.0,
    v7_temporal_safety_cap_detach: bool = False,
    v7_use_temporal_mismatch_multihead: bool = False,
    v7_temporal_mismatch_multihead_cap_mode: str = "none",
    v7_temporal_mismatch_multihead_cap_gamma: float = 1.0,
    v7_temporal_mismatch_multihead_cap_detach: bool = False,
    v7_temporal_mismatch_multihead_fusion: str = "frame_only",
    v7_use_temporal_preservation_head: bool = False,
    v7_use_temporal_preservation_aware_cap: bool = False,
    v7_temporal_preservation_cap_gamma: float = 1.0,
    v7_temporal_preservation_cap_detach: bool = False,
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
        v7_use_v6b_style_final_decision=v7_use_v6b_style_final_decision,
        v7_use_learnable_ne_alpha=v7_use_learnable_ne_alpha,
        v7_ne_alpha_init=v7_ne_alpha_init,
        v7_h1_entitlement_decision_signal=v7_h1_entitlement_decision_signal,
        v7_h1_entitlement_product_power=v7_h1_entitlement_product_power,
        v7_h1_hybrid_residual_beta=v7_h1_hybrid_residual_beta,
        v7_use_location_boundary_head=v7_use_location_boundary_head,
        v7_location_boundary_cap_mode=v7_location_boundary_cap_mode,
        v7_location_boundary_cap_gamma=v7_location_boundary_cap_gamma,
        v7_location_boundary_cap_detach=v7_location_boundary_cap_detach,
        v7_use_temporal_safety_head=v7_use_temporal_safety_head,
        v7_temporal_safety_cap_mode=v7_temporal_safety_cap_mode,
        v7_temporal_safety_cap_gamma=v7_temporal_safety_cap_gamma,
        v7_temporal_safety_cap_detach=v7_temporal_safety_cap_detach,
        v7_use_temporal_mismatch_multihead=v7_use_temporal_mismatch_multihead,
        v7_temporal_mismatch_multihead_cap_mode=v7_temporal_mismatch_multihead_cap_mode,
        v7_temporal_mismatch_multihead_cap_gamma=v7_temporal_mismatch_multihead_cap_gamma,
        v7_temporal_mismatch_multihead_cap_detach=v7_temporal_mismatch_multihead_cap_detach,
        v7_temporal_mismatch_multihead_fusion=v7_temporal_mismatch_multihead_fusion,
        v7_use_temporal_preservation_head=v7_use_temporal_preservation_head,
        v7_use_temporal_preservation_aware_cap=v7_use_temporal_preservation_aware_cap,
        v7_temporal_preservation_cap_gamma=v7_temporal_preservation_cap_gamma,
        v7_temporal_preservation_cap_detach=v7_temporal_preservation_cap_detach,
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
# positive/noncoverage=1: predicate_swap (predicate mismatch ??not frame, not sufficiency)
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
#   (narrow true frame-mismatch rejections ??these and only these are valid frame negatives)
# excluded/masked:
#   predicate_swap ??predicate noncoverage path, kept separate from FrameGate supervision
#   evidence_deletion, evidence_truncation, irrelevant_evidence ??sufficiency failures;
#     masking avoids conflating frame-rejection with evidence-insufficiency
#   polarity_flip ??polarity failure, not frame/entitlement failure
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
#   time_swap is a frame-compatibility failure ??frame_pair_repr is the most specific
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
    Uses temporal_adapter_prob (output of the 2-layer MLP adapter) ??not temporal_diagnostic_prob.
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
      td_final_temporal_rejection_rate  ??among label=1 records, frac predicted NOT_ENTITLED
      td_final_control_preservation_rate ??among label=0 records, frac predicted non-NOT_ENTITLED
      td_final_binary_accuracy           ??overall binary accuracy under the above mapping
      td_final_temporal_count            ??number of label=1 records with valid mask
      td_final_control_count             ??number of label=0 records with valid mask

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


def load_temporal_safety_jsonl(path: "Path") -> list[dict]:
    """Load temporal safety diagnostic JSONL for Stage30-C2 BCE loss.

    Validates that the path does not reference Stage15.
    Records may include time_swap (label=0), none/paraphrase (label=1).
    Do NOT mix these into main train/dev classification data.
    """
    path_str = str(path).lower()
    if "stage15" in path_str:
        raise ValueError(
            f"[ts_head] temporal safety data path contains 'stage15': {path}\n"
            "Stage15 OOD records must not be used for temporal safety training."
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
                    f"[ts_head] Malformed JSON on line {lineno}: {exc}"
                ) from exc
            records.append(r)
    return records


def encode_temporal_safety_labels(
    records: list[dict],
    device: "torch.device",
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Encode temporal safety labels for Stage30-C2 BCE loss.

    Primary: reads `stage30_temporal_safe_label` (1=safe, 0=mismatch) if present.
    Fallback derivation from intervention_type:
        time_swap  ??label=0 (temporal mismatch; unsafe)
        none, paraphrase ??label=1 (temporally safe)
        all others ??masked (excluded from loss)

    Returns (labels, mask) both of shape [B].
    Stage15 OOD is never passed here.
    """
    labels: list[int] = []
    mask: list[int] = []
    for r in records:
        explicit = r.get("stage30_temporal_safe_label")
        if explicit is not None:
            labels.append(int(explicit))
            mask.append(1)
        else:
            itype = r.get("intervention_type", "")
            if itype == "time_swap":
                labels.append(0)
                mask.append(1)
            elif itype in ("none", "paraphrase"):
                labels.append(1)
                mask.append(1)
            else:
                labels.append(0)
                mask.append(0)
    return (
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
    )


def encode_temporal_mismatch_multihead_labels(
    records: list[dict],
    device: "torch.device",
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Encode temporal mismatch labels for Stage30-D multihead BCE loss.

    Convention: temporal mismatch positive = 1, temporal safe/control = 0.
    This is the INVERSE of encode_temporal_safety_labels (where safe=1).

    Primary: reads `stage30_temporal_safe_label` (1=safe, 0=mismatch) if present.
        safe_label 0 ??mismatch target = 1
        safe_label 1 ??mismatch target = 0
    Fallback derivation from intervention_type:
        time_swap          ??label=1 (temporal mismatch positive)
        none, paraphrase   ??label=0 (temporal safe/control)
        all others         ??masked (excluded from loss)

    Returns (labels, mask) both of shape [B].
    Stage15 OOD is never passed here.
    """
    labels: list[int] = []
    mask: list[int] = []
    for r in records:
        explicit = r.get("stage30_temporal_safe_label")
        if explicit is not None:
            # Invert: safe_label=0 means mismatch ??target 1; safe_label=1 means safe ??target 0
            labels.append(1 - int(explicit))
            mask.append(1)
        else:
            itype = r.get("intervention_type", "")
            if itype == "time_swap":
                labels.append(1)
                mask.append(1)
            elif itype in ("none", "paraphrase"):
                labels.append(0)
                mask.append(1)
            else:
                labels.append(0)
                mask.append(0)
    return (
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
    )


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
    # A4d: alias source_intervention_type ??intervention_type
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
          ??A4b2 records matched by contrastive_use_case field
      ood_matched        ??all A4d records (target or source tag present)
      ood_matched_surface  ??A4d records where preservation_construction_type == surface_like_preservation
      ood_matched_temporal ??A4d records where preservation_construction_type == temporal_erased_like_preservation
      all                ??all records from either schema

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
    crashing with KeyError.  These defaults are used only for tensor construction ??    the pair-contrastive loss reads frame_violation_logit, not these label tensors.
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
            **_vnext_model_feature_inputs(inputs),
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


# ---------------------------------------------------------------------------
# Stage28-E: prediction export helpers
# ---------------------------------------------------------------------------

_S28E_INTERVENTION_NORMALIZE: dict[str, str] = {
    "location_swap": "location_swap",
    "role_swap": "role_swap",
    "predicate_swap": "predicate_swap",
    "entity_swap": "entity_swap",
    "event_swap": "event_swap",
    "title_name_swap": "title_name_swap",
    "evidence_deletion": "evidence_deletion",
    "evidence_truncation": "evidence_truncation",
    "irrelevant_evidence": "irrelevant_evidence",
    "none": "none",
    "paraphrase": "paraphrase",
    "polarity_flip": "polarity_flip",
}

_S28E_INTERVENTION_SUFFIXES: tuple[str, ...] = (
    "__location_swap",
    "__role_swap",
    "__predicate_swap",
    "__entity_swap",
    "__event_swap",
    "__title_name_swap",
    "__evidence_deletion",
    "__evidence_truncation",
    "__irrelevant_evidence",
    "__none",
    "__paraphrase",
    "__polarity_flip",
)

_S28E_DIAGNOSTIC_AXIS: dict[str, str] = {
    "location_swap": "location",
    "role_swap": "role",
    "predicate_swap": "predicate",
    "evidence_deletion": "missing_evidence",
    "evidence_truncation": "missing_evidence",
    "irrelevant_evidence": "missing_evidence",
    "entity_swap": "other_frame",
    "event_swap": "other_frame",
    "title_name_swap": "other_frame",
    "none": "control",
    "paraphrase": "control",
    "polarity_flip": "control",
}

_S28E_LABEL_NORMALIZE: dict = {
    "REFUTE": "REFUTE", "REFUTES": "REFUTE", 0: "REFUTE",
    "NOT_ENTITLED": "NOT_ENTITLED", "NE": "NOT_ENTITLED",
    "NOT_ENOUGH_INFO": "NOT_ENTITLED", 1: "NOT_ENTITLED",
    "SUPPORT": "SUPPORT", "SUPPORTS": "SUPPORT", 2: "SUPPORT",
}

_S28E_LABEL_TO_ID: dict[str, int] = {"REFUTE": 0, "NOT_ENTITLED": 1, "SUPPORT": 2}

_S28E_V7_SCALAR_KEYS: tuple[str, ...] = (
    "v7_entitlement_logit",
    "v7_entitlement_prob",
    "v7_polarity_support_logit",
    "v7_polarity_refute_logit",
    "v7_temporal_prob",
    "v7_frame_prob",
    "v7_predicate_prob",
    "v7_sufficiency_prob",
    "v7_h1_entitlement_for_decision",
    "v7_polarity_positive_energy",
    "v7_polarity_negative_energy",
    "v7_polarity_energy_margin",
    # Stage28-I-A: location boundary head scalars (absent when head is disabled)
    "location_boundary_logit",
    "location_boundary_prob",
    "v7_location_boundary_logit",
    "v7_location_boundary_prob",
    "v7_h1_entitlement_before_location_cap",
    "v7_h1_entitlement_after_location_cap",
    # Stage30-C2: temporal safety head scalars (absent when head is disabled)
    "temporal_safety_logit",
    "temporal_safety_prob",
    "v7_temporal_safety_logit",
    "v7_temporal_safety_prob",
    "v7_h1_entitlement_before_temporal_cap",
    "v7_h1_entitlement_after_temporal_cap",
    # Stage30-D: temporal mismatch multihead scalars (absent when head is disabled)
    "temporal_frame_mismatch_logit",
    "temporal_predicate_mismatch_logit",
    "temporal_sufficiency_mismatch_logit",
    "temporal_frame_mismatch_prob",
    "temporal_predicate_mismatch_prob",
    "temporal_sufficiency_mismatch_prob",
    "temporal_mismatch_fused_prob",
    "temporal_mismatch_safe_factor",
    "v7_h1_entitlement_before_temporal_mismatch_cap",
    "v7_h1_entitlement_after_temporal_mismatch_cap",
    # Stage30-E: temporal preservation signal and cap scalars (absent when head/cap is disabled)
    "temporal_preservation_logit",
    "temporal_preservation_prob",
    "v7_temporal_preservation_logit",
    "v7_temporal_preservation_prob",
    "effective_temporal_penalty",
    "temporal_preservation_safe_factor",
    "entitlement_before_temporal_preservation_cap",
    "entitlement_after_temporal_preservation_cap",
    # Stage31-C: directional Coverage/Entailment diagnostic readout
    "coverage_entails_support_logit",
    "coverage_overclaim_ne_logit",
    "coverage_contradicts_refute_logit",
    "coverage_other_residual_logit",
    "coverage_entails_support_prob",
    "coverage_overclaim_ne_prob",
    "coverage_contradicts_refute_prob",
    "coverage_other_residual_prob",
    "coverage_entailment_confidence",
)

_S28E_AUX_LABEL_KEYS: tuple[str, ...] = (
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "primary_failure_type",
)

_S28E_RAW_RECORD_KEYS: tuple[str, ...] = (
    "id", "pair_id", "source_id", "original_id",
    "claim", "evidence", "final_label",
    "intervention", "intervention_type", "perturbation",
    "frame_compatible_label", "predicate_covered_label",
    "sufficiency_label", "polarity_label", "primary_failure_type",
)

_S28E_PRESERVED_METADATA_KEYS: tuple[str, ...] = (
    "id",
    "pair_id",
    "claim",
    "evidence",
    "group",
    "intervention_type",
    "normalized_intervention",
    "primary_failure_type",
    "failure_type",
    "source",
    "split",
    "stage34_family",
    "stage34_relation",
    "stage34_expected_route",
    "stage34_is_heldout",
    "final_label",
    "gold_label",
    "label",
)


def _s28e_normalize_label(value: Any) -> "str | None":
    if value is None:
        return None
    result = _S28E_LABEL_NORMALIZE.get(value)
    if result is not None:
        return result
    if isinstance(value, str):
        result = _S28E_LABEL_NORMALIZE.get(value.upper())
    return result


def _s28e_label_to_id(label: "str | None") -> "int | None":
    if label is None:
        return None
    return _S28E_LABEL_TO_ID.get(label)


def _s28e_normalize_intervention(record: dict) -> "str | None":
    for key in ("intervention", "intervention_type", "perturbation", "probe_type"):
        raw = record.get(key)
        if raw is not None:
            mapped = _S28E_INTERVENTION_NORMALIZE.get(str(raw).lower().strip())
            if mapped:
                return mapped
    meta = record.get("metadata") or {}
    for key in ("intervention", "intervention_type"):
        raw = meta.get(key)
        if raw is not None:
            mapped = _S28E_INTERVENTION_NORMALIZE.get(str(raw).lower().strip())
            if mapped:
                return mapped
    return None


def _s28e_derive_source_id(stable_id: str) -> str:
    for suffix in _S28E_INTERVENTION_SUFFIXES:
        if stable_id.endswith(suffix):
            return stable_id[: -len(suffix)]
    return stable_id


def _s28e_safe_float(value: Any) -> "float | None":
    if value is None:
        return None
    try:
        return float(value.item()) if hasattr(value, "item") else float(value)
    except (TypeError, ValueError):
        return None


def _s28e_safe_list_float(value: Any) -> "list[float] | None":
    if value is None:
        return None
    try:
        seq = value.tolist() if hasattr(value, "tolist") else list(value)
        return [float(x) for x in seq]
    except (TypeError, ValueError):
        return None


def _stage32_output_value(output: dict[str, Any], key: str, index: int) -> Any:
    value = output.get(key)
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return value[index] if index < len(value) else None
    try:
        row_value = value.detach().cpu()[index]
    except (AttributeError, IndexError, TypeError):
        return value
    return _s28e_safe_float(row_value)


def _stage32_bool_label(value: bool | None) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "unavailable"


def _stage32_current_final_label(output: dict[str, Any], index: int) -> str:
    pred_id = _stage32_output_value(output, "predictions", index)
    try:
        return v5.ID_TO_FINAL_LABEL[int(pred_id)]
    except (KeyError, TypeError, ValueError):
        return "UNKNOWN"


_STAGE33_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "their", "there", "to",
    "was", "were", "with",
}

_STAGE33_RULE_STRENGTHS = {
    "quantifier_all_to_some": "high_precision",
    "quantifier_some_to_all": "high_precision",
    "only_to_base": "high_precision",
    "also_to_only": "high_precision",
    "none_to_some": "high_precision",
    "some_to_none": "high_precision",
    "specific_to_general_proxy": "proxy",
    "general_to_specific_proxy": "proxy",
    "whole_to_part_proxy": "proxy",
    "part_to_whole_proxy": "proxy",
    "no_structured_rule_fired": "unresolved",
    "disabled": "unresolved",
}

_STAGE33_DEFAULT_DIRECT_SUPPORT_RULES = "quantifier_all_to_some,only_to_base"

_STAGE33_WHOLE_PART_PAIRS: tuple[tuple[str, str], ...] = (
    ("employees", "engineers"),
    ("research projects", "biology projects"),
    ("services", "payment service"),
    ("projects", "biology projects"),
    ("company", "engineers at the company"),
    ("department", "biology projects in the department"),
    ("platform", "payment service on the platform"),
)

_STAGE33_WHOLE_PART_V2_PAIRS: tuple[tuple[str, str], ...] = (
    ("vehicles", "trucks"),
    ("members", "senior members"),
    ("models", "mark ii model"),
    ("floors", "third floor"),
    ("performing arts groups", "regional dance troupe"),
    ("financial records", "expense reports"),
    ("staff", "contract staff"),
    ("access roads", "eastern access road"),
    ("new hires", "new hires in operations"),
    ("top-ranked competitors", "top-ranked competitor from zone a"),
    ("subscribers", "premium subscribers"),
    ("nodes", "gateway nodes"),
    ("contracts", "service contracts"),
    ("registered participants", "registered participants from abroad"),
    ("residents in the zone", "residents in the northern sector of the zone"),
    ("construction projects", "residential construction projects"),
    ("persons", "foreign nationals"),
)


def _stage33_text(record: dict[str, Any], field: str) -> str:
    raw = record.get(field)
    if raw is None and isinstance(record.get("raw_record"), dict):
        raw = record["raw_record"].get(field)
    return str(raw or "")


def _stage33_word_set(text: str) -> set[str]:
    return {
        tok for tok in re.findall(r"[a-z0-9']+", text.lower())
        if tok not in _STAGE33_STOPWORDS
    }


def _stage33_has_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(
        re.search(rf"\b{re.escape(phrase)}\b", lowered) is not None
        for phrase in phrases
    )


def _stage33_extract_cues(text: str) -> dict[str, Any]:
    lowered = text.lower()
    tokens = _stage33_word_set(text)
    cues = {
        "universal": _stage33_has_phrase(lowered, ("all", "every", "each", "any")),
        "existential": _stage33_has_phrase(
            lowered, ("some", "at least one", "one or more")
        ),
        "negative": _stage33_has_phrase(
            lowered, ("no", "none", "never", "not any")
        ),
        "exclusive": _stage33_has_phrase(
            lowered, ("only", "solely", "exclusively", "alone")
        ),
        "additive": _stage33_has_phrase(
            lowered, ("also", "additionally", "as well as", "along with", "including")
        ),
        "partwhole": _stage33_has_phrase(
            lowered,
            (
                "part of", "member of", "component of", "located in",
                "includes", "contains", "consists of",
            ),
        ),
        "token_count": len(tokens),
        "tokens": tokens,
    }
    return cues


def _stage33_compact_cues(cues: dict[str, Any]) -> str:
    names = [
        name for name in (
            "universal", "existential", "negative", "exclusive", "additive", "partwhole"
        )
        if cues.get(name)
    ]
    names.append(f"tokens={cues.get('token_count', 0)}")
    return ",".join(names)


def _stage33_parse_csv_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (set, list, tuple)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _stage33_normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _stage33_phrase_variants(phrase: str) -> set[str]:
    normalized = _stage33_normalize_phrase(phrase)
    variants = {normalized}
    replacements = (
        ("services", "service"),
        ("service", "services"),
        ("projects", "project"),
        ("project", "projects"),
        ("employees", "employee"),
        ("employee", "employees"),
    )
    for src, dst in replacements:
        variants.add(re.sub(rf"\b{re.escape(src)}\b", dst, normalized))
    return {variant for variant in variants if variant}


def _stage33_contains_phrase(text: str, phrase: str) -> bool:
    normalized = _stage33_normalize_phrase(text)
    return any(
        re.search(rf"\b{re.escape(variant)}\b", normalized) is not None
        for variant in _stage33_phrase_variants(phrase)
    )


def _stage33_parse_whole_part_pairs(
    value: Any,
    *,
    expanded: bool = False,
) -> list[tuple[str, str]]:
    pairs = list(_STAGE33_WHOLE_PART_PAIRS)
    if expanded:
        pairs.extend(_STAGE33_WHOLE_PART_V2_PAIRS)
    for item in sorted(_stage33_parse_csv_set(value)):
        if "->" in item:
            whole, part = item.split("->", 1)
        elif ":" in item:
            whole, part = item.split(":", 1)
        else:
            continue
        whole = _stage33_normalize_phrase(whole)
        part = _stage33_normalize_phrase(part)
        if whole and part:
            pairs.append((whole, part))
    return pairs


def _stage33_content_tokens(text: str) -> set[str]:
    return _stage33_word_set(text)


def _stage33_after_universal_phrase(text: str) -> str:
    normalized = _stage33_normalize_phrase(text)
    match = re.search(r"\b(?:all|every|each)\s+(.+)", normalized)
    return match.group(1).strip() if match else ""


def _stage33_suffix_tokens(text: str) -> set[str]:
    normalized = _stage33_normalize_phrase(text)
    parts = re.split(r"\b(?:at|in|on|after|before|from|of|with|for)\b", normalized, maxsplit=1)
    if len(parts) < 2:
        return set()
    return _stage33_content_tokens(parts[1])


def _stage33_whole_part_pattern_match(evidence: str, claim: str) -> dict[str, str]:
    evidence_np = _stage33_after_universal_phrase(evidence)
    claim_np = _stage33_normalize_phrase(claim)
    if not evidence_np or not claim_np:
        return {"relation": "none", "match": ""}
    evidence_tokens = _stage33_content_tokens(evidence_np)
    claim_tokens = _stage33_content_tokens(claim_np)
    if len(claim_tokens) <= len(evidence_tokens.intersection(claim_tokens)):
        return {"relation": "none", "match": ""}
    shared_suffix = _stage33_suffix_tokens(evidence_np).intersection(
        _stage33_suffix_tokens(claim_np)
    )
    if len(shared_suffix) < 2 and not _stage33_local_domains(evidence).intersection(
        _stage33_local_domains(claim)
    ):
        return {"relation": "none", "match": ""}
    if not evidence_tokens.intersection(claim_tokens):
        return {"relation": "none", "match": ""}
    whole_head = " ".join(list(evidence_tokens)[:3])
    part_head = " ".join(list(claim_tokens - evidence_tokens)[:3] or list(claim_tokens)[:3])
    return {
        "relation": "whole_to_part",
        "match": f"pattern:{whole_head}->{part_head}",
    }


def _stage33_local_domains(text: str) -> set[str]:
    normalized = _stage33_normalize_phrase(text)
    domains: set[str] = set()
    for match in re.finditer(r"\b(?:at|in|on)\s+(?:the\s+)?([a-z0-9' -]+)", normalized):
        domain = match.group(1)
        domain = re.split(r"\b(?:that|which|where|and|but|,|\.)\b", domain)[0]
        words = [
            word for word in re.findall(r"[a-z0-9']+", domain)
            if word not in _STAGE33_STOPWORDS
        ]
        if words:
            domains.add(" ".join(words[:4]))
            domains.add(words[0])
    return domains


def _stage33_same_local_domain(evidence: str, claim: str, whole: str, part: str) -> bool:
    evidence_domains = _stage33_local_domains(evidence)
    claim_domains = _stage33_local_domains(claim)
    if evidence_domains and claim_domains:
        return bool(evidence_domains.intersection(claim_domains))
    return (
        _stage33_contains_phrase(evidence, whole)
        and _stage33_contains_phrase(claim, part)
    ) or (
        _stage33_contains_phrase(evidence, part)
        and _stage33_contains_phrase(claim, whole)
    )


def _stage33_whole_part_match(
    evidence: str,
    claim: str,
    *,
    enabled: bool,
    lexicon: Any = "",
    expanded_lexicon: bool = False,
    pattern_v2: bool = False,
) -> dict[str, str]:
    if not enabled:
        return {"relation": "none", "match": ""}
    for whole, part in _stage33_parse_whole_part_pairs(
        lexicon,
        expanded=expanded_lexicon,
    ):
        evidence_has_whole = _stage33_contains_phrase(evidence, whole)
        evidence_has_part = _stage33_contains_phrase(evidence, part)
        claim_has_whole = _stage33_contains_phrase(claim, whole)
        claim_has_part = _stage33_contains_phrase(claim, part)
        if not _stage33_same_local_domain(evidence, claim, whole, part):
            continue
        if evidence_has_whole and claim_has_part:
            return {"relation": "whole_to_part", "match": f"{whole}->{part}"}
        if evidence_has_part and claim_has_whole:
            return {"relation": "part_to_whole", "match": f"{part}->{whole}"}
    if pattern_v2:
        return _stage33_whole_part_pattern_match(evidence, claim)
    return {"relation": "none", "match": ""}


def _stage33_rule_strength(reason: str) -> str:
    if str(reason).startswith("weak_rule_forced_to_residual:"):
        reason = str(reason).split(":", 1)[1]
    return _STAGE33_RULE_STRENGTHS.get(str(reason), "unknown")


def build_stage33_structured_coverage_owner_state(
    record: dict[str, Any],
    *,
    enabled: bool,
    weak_rules_to_residual: set[str] | None = None,
    whole_part_enabled: bool = False,
    whole_part_lexicon: str = "",
    whole_part_direct_support_enabled: bool = False,
    whole_part_v2_enabled: bool = False,
    whole_part_v2_expanded_lexicon: bool = False,
    whole_part_v2_direct_support_policy: str = "hard_core_required",
) -> dict[str, Any]:
    """Build deterministic Stage33 structured coverage owner diagnostics."""
    claim = _stage33_text(record, "claim")
    evidence = _stage33_text(record, "evidence")
    claim_cues = _stage33_extract_cues(claim)
    evidence_cues = _stage33_extract_cues(evidence)
    label = "STRUCT_UNRESOLVED"
    route = "RESIDUAL"
    reason = "disabled" if not enabled else "no_structured_rule_fired"
    original_reason = reason
    confidence = 0.0
    rule_fired = False
    weak_rules_to_residual = weak_rules_to_residual or set()
    whole_part = _stage33_whole_part_match(
        evidence,
        claim,
        enabled=enabled and whole_part_enabled,
        lexicon=whole_part_lexicon,
        expanded_lexicon=whole_part_v2_enabled and whole_part_v2_expanded_lexicon,
        pattern_v2=whole_part_v2_enabled,
    )

    claim_tokens = claim_cues["tokens"]
    evidence_tokens = evidence_cues["tokens"]
    claim_subset_evidence = bool(claim_tokens) and claim_tokens.issubset(evidence_tokens)
    evidence_subset_claim = bool(evidence_tokens) and evidence_tokens.issubset(claim_tokens)

    if enabled:
        if evidence_cues["negative"] and claim_cues["existential"]:
            label = "STRUCT_CONTRADICTION_REFUTE"
            route = "CONTRADICTION_REFUTE"
            reason = "none_to_some"
            confidence = 1.0
            rule_fired = True
        elif evidence_cues["existential"] and claim_cues["negative"]:
            label = "STRUCT_CONTRADICTION_REFUTE"
            route = "CONTRADICTION_REFUTE"
            reason = "some_to_none"
            confidence = 1.0
            rule_fired = True
        elif evidence_cues["universal"] and claim_cues["existential"]:
            label = "STRUCT_ENTAILMENT_PRESERVE"
            route = "ENTAILMENT_PRESERVE"
            reason = "quantifier_all_to_some"
            confidence = 1.0
            rule_fired = True
        elif evidence_cues["existential"] and claim_cues["universal"]:
            label = "STRUCT_OVERCLAIM_NE"
            route = "OVERCLAIM_NE"
            reason = "quantifier_some_to_all"
            confidence = 1.0
            rule_fired = True
        elif evidence_cues["exclusive"] and not claim_cues["exclusive"]:
            label = "STRUCT_ENTAILMENT_PRESERVE"
            route = "ENTAILMENT_PRESERVE"
            reason = "only_to_base"
            confidence = 1.0
            rule_fired = True
        elif evidence_cues["additive"] and claim_cues["exclusive"]:
            label = "STRUCT_OVERCLAIM_NE"
            route = "OVERCLAIM_NE"
            reason = "also_to_only"
            confidence = 1.0
            rule_fired = True
        elif evidence_cues["universal"] and whole_part["relation"] == "whole_to_part":
            label = "STRUCT_ENTAILMENT_PRESERVE"
            route = "ENTAILMENT_PRESERVE"
            reason = "whole_to_part_proxy"
            confidence = 0.75
            rule_fired = True
        elif claim_cues["universal"] and whole_part["relation"] == "part_to_whole":
            label = "STRUCT_OVERCLAIM_NE"
            route = "OVERCLAIM_NE"
            reason = "part_to_whole_proxy"
            confidence = 0.75
            rule_fired = True
        elif claim_subset_evidence and len(evidence_tokens - claim_tokens) >= 2:
            label = "STRUCT_ENTAILMENT_PRESERVE"
            route = "ENTAILMENT_PRESERVE"
            reason = "specific_to_general_proxy"
            confidence = 0.75
            rule_fired = True
        elif evidence_subset_claim and len(claim_tokens - evidence_tokens) >= 2:
            label = "STRUCT_OVERCLAIM_NE"
            route = "OVERCLAIM_NE"
            reason = "general_to_specific_proxy"
            confidence = 0.75
            rule_fired = True

    original_reason = reason
    if enabled and reason in weak_rules_to_residual:
        label = "STRUCT_UNRESOLVED"
        route = "RESIDUAL"
        reason = f"weak_rule_forced_to_residual:{original_reason}"

    rule_strength = _stage33_rule_strength(original_reason)
    selected_whole_part = original_reason in {
        "whole_to_part_proxy",
        "part_to_whole_proxy",
    }
    exported_whole_part_relation = whole_part["relation"] if selected_whole_part else "none"
    exported_whole_part_match = whole_part["match"] if selected_whole_part else ""
    return {
        "enabled": bool(enabled),
        "label": label,
        "route": route,
        "reason": reason,
        "original_reason": original_reason,
        "rule_strength": rule_strength,
        "confidence": confidence,
        "whole_part_enabled": bool(whole_part_enabled),
        "whole_part_relation": exported_whole_part_relation,
        "whole_part_match": exported_whole_part_match,
        "whole_part_direct_support_enabled": bool(whole_part_direct_support_enabled),
        "whole_part_v2_enabled": bool(whole_part_v2_enabled),
        "whole_part_v2_expanded_lexicon": bool(whole_part_v2_expanded_lexicon),
        "whole_part_v2_direct_support_policy": whole_part_v2_direct_support_policy,
        "whole_part_direct_support_candidate": (
            reason == "whole_to_part_proxy" and route == "ENTAILMENT_PRESERVE"
        ),
        "whole_part_direct_support_allowed": False,
        "whole_part_direct_support_block_reason": (
            "not_whole_to_part" if reason != "whole_to_part_proxy" else "not_evaluated"
        ),
        "whole_part_direct_support_action_block_reason": "none",
        "whole_part_conditional_safe_override_hard_core_enabled": False,
        "whole_part_hard_core_pass": None,
        "whole_part_original_current_label": None,
        "whole_part_conditional_action": None,
        "claim_cues": _stage33_compact_cues(claim_cues),
        "evidence_cues": _stage33_compact_cues(evidence_cues),
        "rule_fired": rule_fired,
        "direct_support_allowed": False,
        "direct_support_block_reason": "not_evaluated",
        "priority_trace": [
            f"structured:{route}",
            f"reason:{reason}",
            f"strength:{rule_strength}",
        ],
    }


def build_stage32_hard_core_owner_state(
    output: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Build the Stage32-B Hard Core owner proxy from existing signals."""
    hard_core_prob = _stage32_output_value(output, "frame_prob", index)
    hard_core_pass = (
        bool(hard_core_prob >= 0.5) if hard_core_prob is not None else None
    )
    source_fields = ["frame_prob"] if hard_core_prob is not None else []
    for optional_source in (
        "v7_location_boundary_prob",
        "v7_temporal_prob",
        "temporal_mismatch_fused_prob",
    ):
        if output.get(optional_source) is not None:
            source_fields.append(optional_source)
    return {
        "prob": hard_core_prob,
        "pass": hard_core_pass,
        "block_reason": (
            "none"
            if hard_core_pass is True
            else "low_frame_proxy"
            if hard_core_pass is False
            else "unavailable_frame_proxy"
        ),
        "source_fields": source_fields,
        "notes": (
            "Proxy-derived Hard Core owner. Uses frame_prob as the validity "
            "probability; location/temporal fields are annotated only when present."
        ),
    }


def build_stage32_coverage_entailment_owner_state(
    output: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Build the Stage32-B Coverage/Entailment owner proxy."""
    cov_source_fields = [
        field
        for field in (
            "coverage_entails_support_prob",
            "coverage_overclaim_ne_prob",
            "coverage_contradicts_refute_prob",
            "coverage_entailment_pred_label",
            "coverage_entailment_pred_id",
            "coverage_entailment_confidence",
            "coverage_entailment_input_mode",
        )
        if output.get(field) is not None
    ]
    coverage_pred_label = _stage32_output_value(
        output, "coverage_entailment_pred_label", index
    )
    coverage_pred_id = _stage32_output_value(output, "coverage_entailment_pred_id", index)
    return {
        "entails_support_prob": _stage32_output_value(
            output, "coverage_entails_support_prob", index
        ),
        "overclaim_ne_prob": _stage32_output_value(
            output, "coverage_overclaim_ne_prob", index
        ),
        "contradicts_refute_prob": _stage32_output_value(
            output, "coverage_contradicts_refute_prob", index
        ),
        "pred_label": coverage_pred_label if coverage_pred_label is not None else "UNAVAILABLE",
        "pred_id": int(coverage_pred_id) if coverage_pred_id is not None else None,
        "confidence": _stage32_output_value(
            output, "coverage_entailment_confidence", index
        ),
        "input_mode": _stage32_output_value(
            output, "coverage_entailment_input_mode", index
        ),
        "source_fields": cov_source_fields,
        "notes": (
            "Proxy-derived Coverage/Entailment owner. Populated from the Stage31 "
            "diagnostic coverage head when enabled; otherwise unavailable."
        ),
    }


def _stage32_coverage_route_from_label(label: Any) -> str:
    if label == "ENTAILS_SUPPORT":
        return "ENTAILMENT_PRESERVE"
    if label == "OVERCLAIM_NOT_ENTITLED":
        return "OVERCLAIM_NE"
    if label == "CONTRADICTS_REFUTE":
        return "CONTRADICTION_REFUTE"
    return "RESIDUAL"


def build_stage32_coverage_owner_v2_state(
    coverage_entailment: dict[str, Any],
    *,
    enabled: bool,
    min_confidence: float,
    min_margin: float,
    allow_abstain: bool,
) -> dict[str, Any]:
    """Build Coverage Owner v2 abstain-aware shadow route fields."""
    probs = [
        (
            "ENTAILS_SUPPORT",
            coverage_entailment.get("entails_support_prob"),
        ),
        (
            "OVERCLAIM_NOT_ENTITLED",
            coverage_entailment.get("overclaim_ne_prob"),
        ),
        (
            "CONTRADICTS_REFUTE",
            coverage_entailment.get("contradicts_refute_prob"),
        ),
    ]
    numeric_probs = [
        (label, float(prob))
        for label, prob in probs
        if prob is not None
    ]
    numeric_probs.sort(key=lambda item: item[1], reverse=True)
    top_label = numeric_probs[0][0] if numeric_probs else coverage_entailment.get("pred_label")
    top_prob = numeric_probs[0][1] if numeric_probs else coverage_entailment.get("confidence")
    second_prob = numeric_probs[1][1] if len(numeric_probs) > 1 else None
    margin = (
        float(top_prob) - float(second_prob)
        if top_prob is not None and second_prob is not None
        else None
    )
    original_label = coverage_entailment.get("pred_label")
    pred_label = original_label
    route = _stage32_coverage_route_from_label(original_label)
    reason = "v2_disabled_uses_v1"
    abstained = False

    if enabled:
        if allow_abstain and (top_prob is None or float(top_prob) < min_confidence):
            pred_label = "UNRESOLVED_COVERAGE"
            route = "RESIDUAL"
            reason = "low_confidence_abstain"
            abstained = True
        elif allow_abstain and (margin is None or float(margin) < min_margin):
            pred_label = "UNRESOLVED_COVERAGE"
            route = "RESIDUAL"
            reason = "low_margin_abstain"
            abstained = True
        else:
            pred_label = original_label if original_label is not None else top_label
            route = _stage32_coverage_route_from_label(pred_label)
            reason = "confident_coverage_prediction"

    return {
        "enabled": bool(enabled),
        "top_prob": top_prob,
        "second_prob": second_prob,
        "margin": margin,
        "min_confidence": min_confidence,
        "min_margin": min_margin,
        "pred_label": pred_label if pred_label is not None else "UNAVAILABLE",
        "route": route,
        "reason": reason,
        "abstained": abstained,
        "allow_abstain": bool(allow_abstain),
    }


def build_stage32_residual_adjudication_owner_state(
    output: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Build the reserved Stage32-B Residual Adjudication owner proxy."""
    del output, index
    return {
        "residual_prob": None,
        "ambiguous_prob": None,
        "underspecified_prob": None,
        "pred_label": "UNIMPLEMENTED_PROXY",
        "source_fields": [],
        "notes": (
            "Reserved interface only. No stable residual owner exists in Stage32-B."
        ),
    }


def build_stage32_ani_diagnostic_state(
    output: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Build the reserved Stage32-B ANI diagnostic readout proxy."""
    del output, index
    return {
        "novelty_prob": None,
        "ambiguity_prob": None,
        "ignorance_prob": None,
        "pred_label": "UNIMPLEMENTED_PROXY",
        "source_fields": [],
        "notes": "Reserved diagnostic readout only. No ANI owner is implemented in Stage32-B.",
    }


def build_stage32_polarity_owner_state(
    output: dict[str, Any],
    index: int,
    hard_core: dict[str, Any],
) -> dict[str, Any]:
    """Build the Stage32-B Polarity owner proxy from existing energies."""
    support_energy = _stage32_output_value(output, "positive_energy", index)
    refute_energy = _stage32_output_value(output, "negative_energy", index)
    support_prob: float | None = None
    refute_prob: float | None = None
    polarity_pred = "NEUTRAL_OR_BLOCKED"
    if support_energy is not None and refute_energy is not None:
        energy_t = torch.tensor([refute_energy, support_energy], dtype=torch.float32)
        probs = torch.softmax(energy_t, dim=-1)
        refute_prob = float(probs[0].item())
        support_prob = float(probs[1].item())
        if hard_core.get("pass") is not False:
            polarity_pred = "SUPPORT" if support_prob >= refute_prob else "REFUTE"
    return {
        "support_energy": support_energy,
        "refute_energy": refute_energy,
        "support_prob": support_prob,
        "refute_prob": refute_prob,
        "pred_label": polarity_pred,
        "source_fields": (
            ["positive_energy", "negative_energy"]
            if support_energy is not None and refute_energy is not None
            else []
        ),
        "notes": (
            "Proxy-derived Polarity owner. Uses existing positive/negative energies; "
            "does not change polarity loss or final prediction."
        ),
    }


def build_stage32_shadow_composer_state(
    *,
    hard_core: dict[str, Any],
    coverage_entailment: dict[str, Any],
    coverage_v2: dict[str, Any],
    structured_coverage: dict[str, Any],
    structured_shadow_mode: bool,
    structured_preserve_can_support: bool,
    structured_direct_support_rules: set[str],
    structured_conditional_fallback: bool,
    structured_conditional_fallback_source: str,
    structured_whole_part_direct_support: bool,
    structured_whole_part_v2_enabled: bool,
    structured_whole_part_v2_direct_support_policy: str,
    structured_whole_part_conditional_safe_overrides_hard_core: bool,
    current_final_label: str,
    residual_adjudication: dict[str, Any],
    ani_diagnostic: dict[str, Any],
    polarity: dict[str, Any],
) -> dict[str, Any]:
    """Build the Stage32-B shadow composer route without applying it."""
    del residual_adjudication, ani_diagnostic
    priority_trace: list[str] = []
    conditional_enabled = (
        bool(structured_conditional_fallback)
        and bool(structured_shadow_mode)
        and bool(structured_coverage.get("enabled"))
    )
    conditional_action = "not_applicable"
    conditional_fallback_used = False
    conditional_override_applied = False
    conditional_override_type = "none"
    route = structured_coverage.get("route")
    structured_reason = str(
        structured_coverage.get("original_reason") or structured_coverage.get("reason")
    )
    structured_strength = str(structured_coverage.get("rule_strength", "unknown"))
    direct_support_allowed = (
        route == "ENTAILMENT_PRESERVE"
        and structured_preserve_can_support
        and structured_reason in structured_direct_support_rules
        and structured_strength == "high_precision"
        and hard_core.get("pass") is True
    )
    whole_part_candidate = (
        route == "ENTAILMENT_PRESERVE"
        and structured_reason == "whole_to_part_proxy"
    )
    if structured_whole_part_v2_enabled:
        whole_part_policy = structured_whole_part_v2_direct_support_policy
    else:
        whole_part_policy = (
            "hard_core_required" if structured_whole_part_direct_support else "off"
        )
    conflicting_hp_route = (
        route in {"CONTRADICTION_REFUTE", "OVERCLAIM_NE"}
        and structured_strength == "high_precision"
    )
    if not whole_part_candidate:
        whole_part_block_reason = "not_whole_to_part"
        whole_part_direct_support_allowed = False
    elif whole_part_policy == "off":
        whole_part_block_reason = "policy_off"
        whole_part_direct_support_allowed = False
    elif hard_core.get("pass") is not True and whole_part_policy == "hard_core_required":
        whole_part_block_reason = "hard_core_not_true"
        whole_part_direct_support_allowed = False
    elif not conditional_enabled and whole_part_policy == "conditional_safe":
        whole_part_block_reason = "conditional_fallback_not_enabled"
        whole_part_direct_support_allowed = False
    elif conflicting_hp_route:
        whole_part_block_reason = "conflicting_high_precision_route"
        whole_part_direct_support_allowed = False
    elif whole_part_policy == "conditional_safe":
        whole_part_block_reason = "allowed"
        whole_part_direct_support_allowed = conditional_enabled
    else:
        whole_part_block_reason = "allowed"
        whole_part_direct_support_allowed = hard_core.get("pass") is True
    structured_coverage["whole_part_direct_support_candidate"] = whole_part_candidate
    structured_coverage["whole_part_direct_support_allowed"] = (
        whole_part_direct_support_allowed
    )
    structured_coverage["whole_part_direct_support_block_reason"] = (
        whole_part_block_reason
    )
    structured_coverage["whole_part_direct_support_action_block_reason"] = "none"
    structured_coverage[
        "whole_part_conditional_safe_override_hard_core_enabled"
    ] = bool(structured_whole_part_conditional_safe_overrides_hard_core)
    structured_coverage["whole_part_hard_core_pass"] = hard_core.get("pass")
    structured_coverage["whole_part_original_current_label"] = current_final_label
    if route == "ENTAILMENT_PRESERVE":
        if direct_support_allowed or whole_part_direct_support_allowed:
            direct_block_reason = "allowed"
        elif not structured_preserve_can_support:
            direct_block_reason = "preserve_can_support_disabled"
        elif hard_core.get("pass") is not True:
            direct_block_reason = "hard_core_not_true"
        elif structured_strength != "high_precision":
            direct_block_reason = f"rule_strength:{structured_strength}"
        else:
            direct_block_reason = f"rule_not_allowed:{structured_reason}"
        structured_coverage["direct_support_allowed"] = (
            direct_support_allowed or whole_part_direct_support_allowed
        )
        structured_coverage["direct_support_block_reason"] = direct_block_reason

    if conditional_enabled:
        priority_trace.extend([
            "conditional_fallback:on",
            f"route:{route}",
            f"strength:{structured_strength}",
        ])
        if (
            hard_core.get("pass") is False
            and whole_part_direct_support_allowed
            and whole_part_policy == "conditional_safe"
            and structured_whole_part_conditional_safe_overrides_hard_core
        ):
            shadow_label = "SUPPORT"
            shadow_reason = (
                "stage33_conditional_whole_part_conditional_safe_direct_support"
            )
            conditional_action = "SUPPORT"
            conditional_override_applied = True
            conditional_override_type = "whole_part_conditional_safe_direct_support"
        elif hard_core.get("pass") is False:
            shadow_label = current_final_label
            shadow_reason = "stage33_conditional_fallback_hard_core_block"
            conditional_action = "fallback_current_final"
            conditional_fallback_used = True
            if (
                whole_part_direct_support_allowed
                and whole_part_policy == "conditional_safe"
            ):
                structured_coverage["whole_part_direct_support_block_reason"] = (
                    "hard_core_priority_blocks_action"
                )
                structured_coverage["whole_part_direct_support_action_block_reason"] = (
                    "hard_core_priority_blocks_action"
                )
        elif route == "CONTRADICTION_REFUTE" and structured_strength == "high_precision":
            shadow_label = "REFUTE"
            shadow_reason = "stage33_conditional_high_precision_contradiction"
            conditional_action = "REFUTE"
            conditional_override_applied = shadow_label != current_final_label
            conditional_override_type = "high_precision_contradiction"
        elif route == "OVERCLAIM_NE" and structured_strength == "high_precision":
            shadow_label = "NOT_ENTITLED"
            shadow_reason = "stage33_conditional_high_precision_overclaim"
            conditional_action = "NOT_ENTITLED"
            conditional_override_applied = shadow_label != current_final_label
            conditional_override_type = "high_precision_overclaim"
        elif route == "ENTAILMENT_PRESERVE" and direct_support_allowed:
            shadow_label = "SUPPORT"
            shadow_reason = "stage33_conditional_high_precision_direct_support"
            conditional_action = "SUPPORT"
            conditional_override_applied = shadow_label != current_final_label
            conditional_override_type = "high_precision_direct_support"
        elif route == "ENTAILMENT_PRESERVE" and whole_part_direct_support_allowed:
            shadow_label = "SUPPORT"
            shadow_reason = "stage33_conditional_whole_part_direct_support"
            conditional_action = "SUPPORT"
            conditional_override_applied = shadow_label != current_final_label
            conditional_override_type = "whole_part_direct_support"
        elif (
            route == "ENTAILMENT_PRESERVE"
            and polarity["pred_label"] == "SUPPORT"
            and structured_strength != "unknown"
        ):
            structured_coverage["direct_support_allowed"] = False
            structured_coverage["direct_support_block_reason"] = (
                "not_needed_positive_polarity"
            )
            shadow_label = "SUPPORT"
            shadow_reason = "stage33_conditional_entailment_positive_polarity"
            conditional_action = "SUPPORT"
            conditional_override_applied = shadow_label != current_final_label
            conditional_override_type = "entailment_positive_polarity"
        else:
            shadow_label = current_final_label
            shadow_reason = "stage33_conditional_fallback_current_final"
            conditional_action = "fallback_current_final"
            conditional_fallback_used = True
            if whole_part_direct_support_allowed and whole_part_policy == "conditional_safe":
                structured_coverage["whole_part_direct_support_action_block_reason"] = (
                    "fallback_priority_blocks_action"
                )
        structured_coverage["whole_part_conditional_action"] = conditional_action
        priority_trace.append(f"action:{conditional_action}")
        would_block_support = hard_core.get("pass") is False
        would_route_ne = shadow_label == "NOT_ENTITLED"
        would_route_refute = shadow_label == "REFUTE"
    elif hard_core.get("pass") is False:
        priority_trace.extend(["hard_core:fail", "route:NOT_ENTITLED"])
        shadow_label = "NOT_ENTITLED"
        shadow_reason = "hard_core_block"
        would_block_support = True
        would_route_ne = True
        would_route_refute = False
    elif structured_shadow_mode and structured_coverage.get("enabled"):
        if route == "OVERCLAIM_NE":
            priority_trace.extend([
                "hard_core:pass",
                "stage33_structured:OVERCLAIM_NE",
                "route:NOT_ENTITLED",
            ])
            shadow_label = "NOT_ENTITLED"
            shadow_reason = "stage33_structured_overclaim"
            would_block_support = False
            would_route_ne = True
            would_route_refute = False
        elif route == "CONTRADICTION_REFUTE":
            priority_trace.extend([
                "hard_core:pass",
                "stage33_structured:CONTRADICTION_REFUTE",
                "route:REFUTE",
            ])
            shadow_label = "REFUTE"
            shadow_reason = "stage33_structured_contradiction"
            would_block_support = False
            would_route_ne = False
            would_route_refute = True
        elif route == "ENTAILMENT_PRESERVE" and polarity["pred_label"] == "SUPPORT":
            structured_coverage["direct_support_allowed"] = False
            structured_coverage["direct_support_block_reason"] = (
                "not_needed_positive_polarity"
            )
            priority_trace.extend([
                "hard_core:pass",
                "stage33_structured:ENTAILMENT_PRESERVE",
                "polarity:SUPPORT",
                "route:SUPPORT",
            ])
            shadow_label = "SUPPORT"
            shadow_reason = "stage33_structured_entails_support_with_positive_polarity"
            would_block_support = False
            would_route_ne = False
            would_route_refute = False
        elif route == "ENTAILMENT_PRESERVE" and structured_preserve_can_support:
            if direct_support_allowed:
                block_reason = "allowed"
            elif hard_core.get("pass") is not True:
                block_reason = "hard_core_not_true"
            elif structured_strength != "high_precision":
                block_reason = f"rule_strength:{structured_strength}"
            else:
                block_reason = f"rule_not_allowed:{structured_reason}"
            structured_coverage["direct_support_allowed"] = direct_support_allowed
            structured_coverage["direct_support_block_reason"] = block_reason
            if direct_support_allowed:
                priority_trace.extend([
                    "hard_core:pass",
                    "stage33_structured:ENTAILMENT_PRESERVE",
                    f"rule:{structured_reason}",
                    "direct_support:allowed",
                    f"polarity:{polarity.get('pred_label')}",
                    "route:SUPPORT",
                ])
                shadow_label = "SUPPORT"
                shadow_reason = "stage33_structured_entails_support_direct_preserve"
                would_block_support = False
                would_route_ne = False
                would_route_refute = False
            else:
                priority_trace.extend([
                    "hard_core:pass",
                    "stage33_structured:ENTAILMENT_PRESERVE",
                    f"rule:{structured_reason}",
                    f"polarity:{polarity.get('pred_label')}",
                    f"direct_support:blocked:{block_reason}",
                    "route:NOT_ENTITLED",
                ])
                shadow_label = "NOT_ENTITLED"
                shadow_reason = "stage33_structured_entailment_direct_support_blocked"
                would_block_support = False
                would_route_ne = True
                would_route_refute = False
        elif route == "ENTAILMENT_PRESERVE":
            structured_coverage["direct_support_allowed"] = False
            structured_coverage["direct_support_block_reason"] = (
                "preserve_can_support_disabled"
            )
            priority_trace.extend([
                "hard_core:pass",
                "stage33_structured:ENTAILMENT_PRESERVE",
                f"polarity:{polarity.get('pred_label')}",
                "direct_support:blocked:preserve_can_support_disabled",
                "route:NOT_ENTITLED",
            ])
            shadow_label = "NOT_ENTITLED"
            shadow_reason = "stage33_structured_entailment_without_positive_polarity"
            would_block_support = False
            would_route_ne = True
            would_route_refute = False
        else:
            priority_trace.extend([
                "hard_core:pass",
                "stage33_structured:RESIDUAL",
                "route:NOT_ENTITLED",
            ])
            shadow_label = "NOT_ENTITLED"
            shadow_reason = "stage33_structured_unresolved_to_residual"
            would_block_support = False
            would_route_ne = True
            would_route_refute = False
    elif coverage_v2.get("enabled"):
        route = coverage_v2.get("route")
        pred_label = coverage_v2.get("pred_label")
        if route == "OVERCLAIM_NE":
            priority_trace.extend([
                "hard_core:pass",
                "coverage_v2:OVERCLAIM_NE",
                "route:NOT_ENTITLED",
            ])
            shadow_label = "NOT_ENTITLED"
            shadow_reason = "coverage_v2_overclaim"
            would_block_support = False
            would_route_ne = True
            would_route_refute = False
        elif route == "CONTRADICTION_REFUTE":
            priority_trace.extend([
                "hard_core:pass",
                "coverage_v2:CONTRADICTION_REFUTE",
                "route:REFUTE",
            ])
            shadow_label = "REFUTE"
            shadow_reason = "coverage_v2_contradiction"
            would_block_support = False
            would_route_ne = False
            would_route_refute = True
        elif route == "ENTAILMENT_PRESERVE" and polarity["pred_label"] == "SUPPORT":
            priority_trace.extend([
                "hard_core:pass",
                "coverage_v2:ENTAILMENT_PRESERVE",
                "polarity:SUPPORT",
                "route:SUPPORT",
            ])
            shadow_label = "SUPPORT"
            shadow_reason = "coverage_v2_entails_support_with_positive_polarity"
            would_block_support = False
            would_route_ne = False
            would_route_refute = False
        elif route == "ENTAILMENT_PRESERVE":
            priority_trace.extend([
                "hard_core:pass",
                "coverage_v2:ENTAILMENT_PRESERVE",
                f"polarity:{polarity.get('pred_label')}",
                "route:NOT_ENTITLED",
            ])
            shadow_label = "NOT_ENTITLED"
            shadow_reason = "coverage_v2_entailment_without_positive_polarity"
            would_block_support = False
            would_route_ne = True
            would_route_refute = False
        else:
            priority_trace.extend([
                "hard_core:pass",
                f"coverage_v2:{pred_label}",
                "route:NOT_ENTITLED",
            ])
            shadow_label = "NOT_ENTITLED"
            shadow_reason = "coverage_v2_unresolved_to_residual"
            would_block_support = False
            would_route_ne = True
            would_route_refute = False
    elif coverage_entailment["pred_label"] == "OVERCLAIM_NOT_ENTITLED":
        priority_trace.extend([
            "hard_core:pass",
            "coverage:OVERCLAIM_NOT_ENTITLED",
            "route:NOT_ENTITLED",
        ])
        shadow_label = "NOT_ENTITLED"
        shadow_reason = "coverage_overclaim"
        would_block_support = False
        would_route_ne = True
        would_route_refute = False
    elif coverage_entailment["pred_label"] == "CONTRADICTS_REFUTE":
        priority_trace.extend([
            "hard_core:pass",
            "coverage:CONTRADICTS_REFUTE",
            "route:REFUTE",
        ])
        shadow_label = "REFUTE"
        shadow_reason = "coverage_contradiction"
        would_block_support = False
        would_route_ne = False
        would_route_refute = True
    elif (
        coverage_entailment["pred_label"] == "ENTAILS_SUPPORT"
        and polarity["pred_label"] == "SUPPORT"
    ):
        priority_trace.extend([
            "hard_core:pass",
            "coverage:ENTAILS_SUPPORT",
            "polarity:SUPPORT",
            "route:SUPPORT",
        ])
        shadow_label = "SUPPORT"
        shadow_reason = "coverage_entails_support_with_positive_polarity"
        would_block_support = False
        would_route_ne = False
        would_route_refute = False
    else:
        priority_trace.extend([
            "hard_core:" + _stage32_bool_label(hard_core.get("pass")),
            f"coverage:{coverage_entailment.get('pred_label')}",
            f"polarity:{polarity.get('pred_label')}",
            "route:NOT_ENTITLED",
        ])
        shadow_label = "NOT_ENTITLED"
        shadow_reason = "residual_or_unresolved"
        would_block_support = False
        would_route_ne = True
        would_route_refute = False
    return {
        "would_block_support": would_block_support,
        "would_route_ne": would_route_ne,
        "would_route_refute": would_route_refute,
        "shadow_label": shadow_label,
        "shadow_reason": shadow_reason,
        "priority_trace": priority_trace,
        "stage33_conditional_fallback_enabled": conditional_enabled,
        "stage33_conditional_fallback_source": structured_conditional_fallback_source,
        "stage33_conditional_action": conditional_action,
        "stage33_conditional_fallback_used": conditional_fallback_used,
        "stage33_conditional_override_applied": conditional_override_applied,
        "stage33_conditional_override_type": conditional_override_type,
        "stage33_conditional_original_current_label": current_final_label,
        "stage33_conditional_shadow_label": shadow_label if conditional_enabled else None,
        "note": "Stage32-B shadow only; not used for logits, loss, predictions, or selection.",
    }


def build_stage32_owner_state(
    record: dict[str, Any],
    output: dict[str, Any],
    index: int,
    *,
    coverage_owner_v2_enabled: bool = False,
    coverage_owner_v2_min_confidence: float = 0.50,
    coverage_owner_v2_min_margin: float = 0.05,
    coverage_owner_v2_allow_abstain: bool = False,
    structured_coverage_enabled: bool = False,
    structured_coverage_shadow_mode: bool = False,
    structured_coverage_preserve_can_support: bool = False,
    structured_coverage_direct_support_rules: set[str] | None = None,
    structured_coverage_weak_rules_to_residual: set[str] | None = None,
    structured_coverage_conditional_fallback: bool = False,
    structured_coverage_fallback_source: str = "current_final",
    structured_coverage_whole_part_enabled: bool = False,
    structured_coverage_whole_part_direct_support: bool = False,
    structured_coverage_whole_part_lexicon: str = "",
    structured_coverage_whole_part_v2: bool = False,
    structured_coverage_whole_part_v2_expanded_lexicon: bool = False,
    structured_coverage_whole_part_v2_direct_support_policy: str = "hard_core_required",
    structured_whole_part_conditional_safe_overrides_hard_core: bool = False,
) -> dict[str, Any]:
    """Build Stage32-B owner-state proxies without changing model outputs."""
    hard_core = build_stage32_hard_core_owner_state(output, index)
    coverage_entailment = build_stage32_coverage_entailment_owner_state(output, index)
    coverage_v2 = build_stage32_coverage_owner_v2_state(
        coverage_entailment,
        enabled=coverage_owner_v2_enabled,
        min_confidence=coverage_owner_v2_min_confidence,
        min_margin=coverage_owner_v2_min_margin,
        allow_abstain=coverage_owner_v2_allow_abstain,
    )
    structured_coverage = build_stage33_structured_coverage_owner_state(
        record,
        enabled=structured_coverage_enabled,
        weak_rules_to_residual=structured_coverage_weak_rules_to_residual,
        whole_part_enabled=structured_coverage_whole_part_enabled,
        whole_part_lexicon=structured_coverage_whole_part_lexicon,
        whole_part_direct_support_enabled=structured_coverage_whole_part_direct_support,
        whole_part_v2_enabled=(
            structured_coverage_whole_part_enabled
            and structured_coverage_whole_part_v2
        ),
        whole_part_v2_expanded_lexicon=structured_coverage_whole_part_v2_expanded_lexicon,
        whole_part_v2_direct_support_policy=(
            structured_coverage_whole_part_v2_direct_support_policy
        ),
    )
    residual_adjudication = build_stage32_residual_adjudication_owner_state(output, index)
    ani_diagnostic = build_stage32_ani_diagnostic_state(output, index)
    polarity = build_stage32_polarity_owner_state(output, index, hard_core)
    current_final_label = _stage32_current_final_label(output, index)
    composer_shadow = build_stage32_shadow_composer_state(
        hard_core=hard_core,
        coverage_entailment=coverage_entailment,
        coverage_v2=coverage_v2,
        structured_coverage=structured_coverage,
        structured_shadow_mode=structured_coverage_shadow_mode,
        structured_preserve_can_support=structured_coverage_preserve_can_support,
        structured_direct_support_rules=(
            structured_coverage_direct_support_rules
            or _stage33_parse_csv_set(_STAGE33_DEFAULT_DIRECT_SUPPORT_RULES)
        ),
        structured_conditional_fallback=structured_coverage_conditional_fallback,
        structured_conditional_fallback_source=structured_coverage_fallback_source,
        structured_whole_part_direct_support=structured_coverage_whole_part_direct_support,
        structured_whole_part_v2_enabled=(
            structured_coverage_whole_part_enabled
            and structured_coverage_whole_part_v2
        ),
        structured_whole_part_v2_direct_support_policy=(
            structured_coverage_whole_part_v2_direct_support_policy
        ),
        structured_whole_part_conditional_safe_overrides_hard_core=(
            structured_whole_part_conditional_safe_overrides_hard_core
        ),
        current_final_label=current_final_label,
        residual_adjudication=residual_adjudication,
        ani_diagnostic=ani_diagnostic,
        polarity=polarity,
    )
    return {
        "hard_core": hard_core,
        "coverage_entailment": coverage_entailment,
        "coverage_v2": coverage_v2,
        "structured_coverage": structured_coverage,
        "residual_adjudication": residual_adjudication,
        "ani_diagnostic": ani_diagnostic,
        "polarity": polarity,
        "composer_shadow": composer_shadow,
    }


# ---------------------------------------------------------------------------
# Stage36-A: conservative support-safety blockers (shadow-only)
# ---------------------------------------------------------------------------
# These blockers fire *before* a Stage33 structured owner SUPPORT override is
# treated as safe. They never touch final logits or final classifier
# predictions -- they only affect the exported Stage32/Stage33 shadow label
# when explicitly enabled via --stage36-support-safety-shadow-mode. All
# heuristics below are intentionally conservative: they only fire on the
# SUPPORT side of specific, narrow failure modes observed in Stage35-A.

_STAGE36_EXCEPTION_RE = re.compile(
    r"\ball\s+(?P<whole>.+?)\s+except\s+(?:the\s+)?(?P<excluded>.+?)"
    r"(?=\s+(?:were|was|is|are|had|have|has)\b|[.,;]|$)"
)

_STAGE36_NOT_ALL_RE = re.compile(
    r"\bnot\s+(?:all|every)\s+(?:of\s+the\s+)?(?P<subject>.+?)"
    r"(?=\s+(?:were|was|is|are|had|have|has)\b|[.,;]|$)"
)

_STAGE36_SOME_RE = re.compile(
    r"\b(?:at\s+least\s+)?some\s+(?:of\s+the\s+)?(?P<subject>.+?)"
    r"(?=\s+(?:were|was|is|are|had|have|has)\b|[.,;]|$)"
)

_STAGE36_LOCATION_DIRECTIONS = {
    "east", "west", "north", "south", "eastern", "western", "northern", "southern",
}

_STAGE36_LOCATION_NOUNS = {
    "district", "zone", "region", "county", "campus", "depot", "warehouse",
    "clinic", "platform", "network", "catalog", "archive", "province", "league",
    "shelter", "plant", "building",
}

_STAGE36_LOCATION_PHRASES = ("transit plan", "support system")

_STAGE36_WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

_STAGE36_QUARTERS = {"q1", "q2", "q3", "q4"}

_STAGE36_TEMPORAL_PHRASES = (
    "last year", "this year", "next year", "today", "yesterday", "tomorrow",
)

_STAGE36_YEAR_RE = re.compile(r"\b(20[2-3][0-9])\b")


def _stage36_find_exception_clause(evidence: str) -> "dict[str, str] | None":
    normalized = _stage33_normalize_phrase(evidence).rstrip(".")
    match = _STAGE36_EXCEPTION_RE.search(normalized)
    if not match:
        return None
    return {
        "whole": match.group("whole").strip(),
        "excluded": match.group("excluded").strip(),
    }


def _stage36_exception_blocker(claim: str, evidence: str) -> dict[str, Any]:
    """Block SUPPORT when the claim targets an 'all X except Y' excluded subset."""
    clause = _stage36_find_exception_clause(evidence)
    if clause is None:
        return {"fired": False}
    excluded_words = _stage33_word_set(clause["excluded"])
    claim_words = _stage33_word_set(claim)
    if not excluded_words:
        return {"fired": False}
    overlap = excluded_words.intersection(claim_words)
    ratio = len(overlap) / len(excluded_words)
    fired = bool(overlap) and ratio >= 0.6
    return {
        "fired": fired,
        "excluded_phrase": clause["excluded"],
        "overlap": sorted(overlap),
    }


def _stage36_not_all_existential_blocker(claim: str, evidence: str) -> dict[str, Any]:
    """Block SUPPORT when 'not all/every X' is claimed to entail 'some X'."""
    evidence_norm = _stage33_normalize_phrase(evidence).rstrip(".")
    claim_norm = _stage33_normalize_phrase(claim).rstrip(".")
    evidence_match = _STAGE36_NOT_ALL_RE.search(evidence_norm)
    claim_match = _STAGE36_SOME_RE.search(claim_norm)
    if not evidence_match or not claim_match:
        return {"fired": False}
    evidence_subject = _stage33_word_set(evidence_match.group("subject"))
    claim_subject = _stage33_word_set(claim_match.group("subject"))
    if not evidence_subject or not claim_subject:
        return {"fired": False}
    overlap = evidence_subject.intersection(claim_subject)
    ratio = len(overlap) / len(evidence_subject)
    fired = bool(overlap) and ratio >= 0.5
    return {
        "fired": fired,
        "evidence_subject": evidence_match.group("subject").strip(),
        "claim_subject": claim_match.group("subject").strip(),
    }


def _stage36_extract_location_scopes(text: str) -> dict[str, str]:
    normalized = _stage33_normalize_phrase(text)
    tokens = normalized.split()
    scopes: dict[str, str] = {}
    for i, tok in enumerate(tokens):
        clean = re.sub(r"[^a-z0-9]", "", tok)
        if clean not in _STAGE36_LOCATION_NOUNS:
            continue
        qualifier = ""
        if i > 0:
            prev = re.sub(r"[^a-z0-9]", "", tokens[i - 1])
            if prev in _STAGE36_LOCATION_DIRECTIONS:
                qualifier = prev
            elif prev.isalpha() and len(prev) <= 2:
                qualifier = prev
        if not qualifier and i + 1 < len(tokens):
            nxt = re.sub(r"[^a-z0-9]", "", tokens[i + 1])
            if nxt.isalpha() and len(nxt) <= 2:
                qualifier = nxt
        if qualifier:
            scopes[clean] = qualifier
    for phrase in _STAGE36_LOCATION_PHRASES:
        idx = normalized.find(phrase)
        if idx < 0:
            continue
        preceding = normalized[:idx].split()
        if preceding and preceding[-1] in _STAGE36_LOCATION_DIRECTIONS:
            scopes[phrase.replace(" ", "_")] = preceding[-1]
    return scopes


def _stage36_location_scope_blocker(claim: str, evidence: str) -> dict[str, Any]:
    """Block SUPPORT when claim/evidence location scopes both exist and conflict."""
    claim_scopes = _stage36_extract_location_scopes(claim)
    evidence_scopes = _stage36_extract_location_scopes(evidence)
    conflicts = {}
    for marker, claim_qualifier in claim_scopes.items():
        evidence_qualifier = evidence_scopes.get(marker)
        if evidence_qualifier and evidence_qualifier != claim_qualifier:
            conflicts[marker] = {
                "claim": claim_qualifier,
                "evidence": evidence_qualifier,
            }
    return {
        "fired": bool(conflicts),
        "claim_values": claim_scopes,
        "evidence_values": evidence_scopes,
        "conflicts": conflicts,
    }


def _stage36_extract_temporal_markers(text: str) -> set[str]:
    normalized = _stage33_normalize_phrase(text)
    markers: set[str] = set(_STAGE36_YEAR_RE.findall(normalized))
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    markers.update(tokens.intersection(_STAGE36_WEEKDAYS))
    markers.update(tokens.intersection(_STAGE36_QUARTERS))
    for phrase in _STAGE36_TEMPORAL_PHRASES:
        if phrase in normalized:
            markers.add(phrase)
    return markers


def _stage36_temporal_scope_blocker(claim: str, evidence: str) -> dict[str, Any]:
    """Block SUPPORT when claim/evidence temporal markers both exist and conflict."""
    claim_markers = _stage36_extract_temporal_markers(claim)
    evidence_markers = _stage36_extract_temporal_markers(evidence)
    fired = (
        bool(claim_markers)
        and bool(evidence_markers)
        and not claim_markers.intersection(evidence_markers)
    )
    return {
        "fired": fired,
        "claim_values": sorted(claim_markers),
        "evidence_values": sorted(evidence_markers),
    }


def compute_stage36_support_safety_blocker(
    claim: str,
    evidence: str,
    proposed_shadow_label: str,
    proposed_reason: str,
    proposed_route: str,
    *,
    enabled: bool,
    block_exception_scope: bool = False,
    block_not_all_existential: bool = False,
    block_location_scope_mismatch: bool = False,
    block_temporal_scope_mismatch: bool = False,
    blocker_action: str = "fallback_current_final",
    current_final_label: str = "NOT_ENTITLED",
) -> dict[str, Any]:
    """Stage36-A: conservative safety blockers for structured SUPPORT overrides.

    Only fires when `proposed_shadow_label` is SUPPORT. Never blocks REFUTE or
    NOT_ENTITLED, and never fires when the proposed label is already the
    fallback_current_final non-SUPPORT label. Shadow/diagnostic only -- callers
    are responsible for keeping this out of final logits/predictions.
    """
    del proposed_reason, proposed_route  # reserved for future rule-specific gating
    result: dict[str, Any] = {
        "stage36_support_blocker_fired": False,
        "stage36_support_blocker_reasons": [],
        "stage36_support_blocker_action": blocker_action,
        "stage36_support_blocker_original_shadow_label": proposed_shadow_label,
        "stage36_support_blocker_final_shadow_label": proposed_shadow_label,
        "stage36_exception_blocker_fired": False,
        "stage36_not_all_blocker_fired": False,
        "stage36_location_scope_blocker_fired": False,
        "stage36_temporal_scope_blocker_fired": False,
        "stage36_scope_claim_values": {},
        "stage36_scope_evidence_values": {},
    }
    if not enabled or proposed_shadow_label != "SUPPORT":
        return result

    reasons: list[str] = []
    scope_claim_values: dict[str, Any] = {}
    scope_evidence_values: dict[str, Any] = {}

    if block_exception_scope:
        exception_result = _stage36_exception_blocker(claim, evidence)
        if exception_result.get("fired"):
            result["stage36_exception_blocker_fired"] = True
            reasons.append("exception_excluded_subset_block")

    if block_not_all_existential:
        not_all_result = _stage36_not_all_existential_blocker(claim, evidence)
        if not_all_result.get("fired"):
            result["stage36_not_all_blocker_fired"] = True
            reasons.append("not_all_does_not_entail_some_block")

    if block_location_scope_mismatch:
        location_result = _stage36_location_scope_blocker(claim, evidence)
        scope_claim_values["location"] = location_result.get("claim_values", {})
        scope_evidence_values["location"] = location_result.get("evidence_values", {})
        if location_result.get("fired"):
            result["stage36_location_scope_blocker_fired"] = True
            reasons.append("location_scope_mismatch_block")

    if block_temporal_scope_mismatch:
        temporal_result = _stage36_temporal_scope_blocker(claim, evidence)
        scope_claim_values["temporal"] = temporal_result.get("claim_values", [])
        scope_evidence_values["temporal"] = temporal_result.get("evidence_values", [])
        if temporal_result.get("fired"):
            result["stage36_temporal_scope_blocker_fired"] = True
            reasons.append("temporal_scope_mismatch_block")

    result["stage36_scope_claim_values"] = scope_claim_values
    result["stage36_scope_evidence_values"] = scope_evidence_values
    result["stage36_support_blocker_reasons"] = reasons
    result["stage36_support_blocker_fired"] = bool(reasons)

    if reasons:
        if blocker_action == "force_not_entitled":
            final_label = "NOT_ENTITLED"
        else:
            final_label = (
                current_final_label if current_final_label != "SUPPORT" else "NOT_ENTITLED"
            )
        result["stage36_support_blocker_final_shadow_label"] = final_label

    return result


def _stage36_config_from_args(args: "argparse.Namespace") -> dict[str, Any]:
    """Build the Stage36-A blocker config dict from CLI args. All defaults off."""
    return {
        "enabled": getattr(args, "stage36_use_support_safety_blockers", False),
        "export": getattr(args, "stage36_support_safety_export", False),
        "shadow_mode": getattr(args, "stage36_support_safety_shadow_mode", False),
        "block_exception_scope": getattr(args, "stage36_block_exception_scope", False),
        "block_not_all_existential": getattr(
            args, "stage36_block_not_all_existential", False
        ),
        "block_location_scope_mismatch": getattr(
            args, "stage36_block_location_scope_mismatch", False
        ),
        "block_temporal_scope_mismatch": getattr(
            args, "stage36_block_temporal_scope_mismatch", False
        ),
        "blocker_action": getattr(
            args, "stage36_support_blocker_action", "fallback_current_final"
        ),
    }


# ---------------------------------------------------------------------------
# Stage37-A: conservative safe SUPPORT recovery (shadow-only)
# ---------------------------------------------------------------------------
# Stage37 runs strictly *after* Stage36's post-blocker shadow label is known.
# It never overrides a fired Stage36 blocker and never touches final logits or
# final classifier predictions. It only recovers SUPPORT for a small set of
# conservative, narrowly-scoped patterns left over from Stage35-A/Stage36-A:
#   (1) "no X except Y" -> Y SUPPORT  (included-subset double-negative)
#   (2) "all X and all Z" -> subset of X or Z SUPPORT (coordination universal)
#   (3) "all N X" -> subset among X SUPPORT (numeric universal)
# All heuristics reuse the Stage36 hazard detectors so hazard logic is never
# duplicated or allowed to drift out of sync.

_STAGE37_VERB_SPLIT_RE = re.compile(r"\b(were|was|is|are|had|have|has|received)\b")

_STAGE37_NO_EXCEPT_RE = re.compile(
    r"\bno\s+(?P<whole>.+?)\s+except\s+(?:the\s+)?(?P<included>.+?)\s+"
    r"(?P<predicate>(?:were|was|is|are|had|have|has)\b.+?)[.,;]?$"
)

_STAGE37_COORD_UNIVERSAL_RE = re.compile(
    r"\ball\s+(?P<first>.+?)\s+and\s+all\s+(?P<second>.+?)\s+"
    r"(?P<predicate>(?:were|was|is|are|had|have|has)\b.+?)[.,;]?$"
)

_STAGE37_NUMERIC_UNIVERSAL_RE = re.compile(
    r"\ball\s+(?P<number>\d+)\s+(?P<whole>.+?)\s+"
    r"(?P<predicate>(?:were|was|is|are|had|have|has|received)\b.+?)[.,;]?$"
)

_STAGE37_AMONG_RE = re.compile(
    r"\bamong\s+(?:the\s+)?(?P<whole>.+?)"
    r"(?=\s+(?:were|was|is|are|had|have|has)\b|[.,;]|$)"
)


def has_excluded_subset_hazard(claim: str, evidence: str) -> bool:
    """Stage37-A: reuse the Stage36 'all X except Y' excluded-subset blocker."""
    return bool(_stage36_exception_blocker(claim, evidence).get("fired"))


def has_not_all_existential_hazard(claim: str, evidence: str) -> bool:
    """Stage37-A: reuse the Stage36 'not all/every X' -> 'some X' blocker."""
    return bool(_stage36_not_all_existential_blocker(claim, evidence).get("fired"))


def has_location_scope_mismatch(claim: str, evidence: str) -> bool:
    """Stage37-A: reuse the Stage36 conflicting location-scope blocker."""
    return bool(_stage36_location_scope_blocker(claim, evidence).get("fired"))


def has_temporal_scope_mismatch(claim: str, evidence: str) -> bool:
    """Stage37-A: reuse the Stage36 conflicting temporal-scope blocker."""
    return bool(_stage36_temporal_scope_blocker(claim, evidence).get("fired"))


def _stage37_split_subject_predicate(text: str) -> "tuple[str, str] | None":
    normalized = _stage33_normalize_phrase(text).rstrip(".")
    match = _STAGE37_VERB_SPLIT_RE.search(normalized)
    if not match:
        return None
    subject = normalized[: match.start()].strip()
    predicate = normalized[match.start():].strip()
    if not subject or not predicate:
        return None
    return subject, predicate


def _stage37_phrase_head(phrase: str) -> str:
    words = re.findall(r"[a-z0-9']+", phrase.lower())
    return words[-1] if words else ""


def _stage37_is_subset_of_whole(claim_subject: str, whole_phrase: str) -> bool:
    """Conservative subset check: claim subject must contain the whole's head noun."""
    claim_words = _stage33_word_set(claim_subject)
    whole_head = _stage37_phrase_head(whole_phrase)
    if not whole_head or not claim_words:
        return False
    return whole_head in claim_words


def _stage37_predicate_overlap_ratio(evidence_predicate: str, claim_predicate: str) -> float:
    predicate_words = _stage33_word_set(evidence_predicate)
    claim_predicate_words = _stage33_word_set(claim_predicate)
    if not claim_predicate_words:
        return 0.0
    overlap = predicate_words.intersection(claim_predicate_words)
    return len(overlap) / len(claim_predicate_words)


def _stage37_no_except_included_subset_recovery(claim: str, evidence: str) -> dict[str, Any]:
    """Part C: 'no X except Y ...' -> Y SUPPORT when claim targets the included subset Y."""
    evidence_norm = _stage33_normalize_phrase(evidence).rstrip(".")
    match = _STAGE37_NO_EXCEPT_RE.search(evidence_norm)
    if not match:
        return {"fired": False}
    included_phrase = match.group("included").strip()
    evidence_predicate = match.group("predicate").strip()
    claim_split = _stage37_split_subject_predicate(claim)
    if claim_split is None:
        return {"fired": False}
    claim_subject, claim_predicate = claim_split
    included_words = _stage33_word_set(included_phrase)
    claim_subject_words = _stage33_word_set(claim_subject)
    if not included_words or not claim_subject_words:
        return {"fired": False}
    subject_overlap = included_words.intersection(claim_subject_words)
    # Require exact word-set equality (not mere overlap) so that a distinguishing
    # modifier swap (e.g. "night-shift" -> "day-shift") -- which shares tokens
    # like "shift"/"workers" with the included phrase -- is never conflated with
    # the actually-included subset. This is deliberately conservative.
    subject_exact_match = claim_subject_words == included_words
    predicate_ratio = _stage37_predicate_overlap_ratio(evidence_predicate, claim_predicate)
    fired = subject_exact_match and predicate_ratio >= 0.5
    return {
        "fired": fired,
        "included_phrase": included_phrase,
        "subject_overlap": sorted(subject_overlap),
    }


def _stage37_coordination_universal_subset_recovery(claim: str, evidence: str) -> dict[str, Any]:
    """Part D: 'all X and all Z were P' -> subset of X or Z SUPPORT."""
    evidence_norm = _stage33_normalize_phrase(evidence).rstrip(".")
    match = _STAGE37_COORD_UNIVERSAL_RE.search(evidence_norm)
    if not match:
        return {"fired": False}
    first_whole = match.group("first").strip()
    second_whole = match.group("second").strip()
    evidence_predicate = match.group("predicate").strip()
    claim_split = _stage37_split_subject_predicate(claim)
    if claim_split is None:
        return {"fired": False}
    claim_subject, claim_predicate = claim_split
    matched_whole = None
    if _stage37_is_subset_of_whole(claim_subject, first_whole):
        matched_whole = first_whole
    elif _stage37_is_subset_of_whole(claim_subject, second_whole):
        matched_whole = second_whole
    if matched_whole is None:
        return {"fired": False}
    predicate_ratio = _stage37_predicate_overlap_ratio(evidence_predicate, claim_predicate)
    fired = predicate_ratio >= 0.5
    return {"fired": fired, "matched_whole": matched_whole}


def _stage37_numeric_universal_subset_recovery(claim: str, evidence: str) -> dict[str, Any]:
    """Part E: 'all N X were P' -> subset among X SUPPORT."""
    evidence_norm = _stage33_normalize_phrase(evidence).rstrip(".")
    match = _STAGE37_NUMERIC_UNIVERSAL_RE.search(evidence_norm)
    if not match:
        return {"fired": False}
    whole_phrase = match.group("whole").strip()
    evidence_predicate = match.group("predicate").strip()
    claim_norm = _stage33_normalize_phrase(claim).rstrip(".")
    whole_head = _stage37_phrase_head(whole_phrase)
    subset_ok = False
    among_match = _STAGE37_AMONG_RE.search(claim_norm)
    if among_match:
        among_words = _stage33_word_set(among_match.group("whole").strip())
        if whole_head and whole_head in among_words:
            subset_ok = True
    claim_split = _stage37_split_subject_predicate(claim)
    if claim_split is None:
        return {"fired": False}
    claim_subject, claim_predicate = claim_split
    if not subset_ok and _stage37_is_subset_of_whole(claim_subject, whole_phrase):
        subset_ok = True
    if not subset_ok:
        return {"fired": False}
    predicate_ratio = _stage37_predicate_overlap_ratio(evidence_predicate, claim_predicate)
    fired = predicate_ratio >= 0.5
    return {"fired": fired, "whole_phrase": whole_phrase}


def compute_stage37_safe_support_recovery(
    claim: str,
    evidence: str,
    current_shadow_label: str,
    args: "argparse.Namespace",
    stage36_info: dict[str, Any],
) -> dict[str, Any]:
    """Stage37-A: conservative safe SUPPORT recovery after Stage36 blockers.

    Only fires when Stage37 is enabled, the post-Stage36 shadow label is not
    already SUPPORT, no Stage36 support blocker fired, and none of the
    conservative hazard checks (excluded-subset, not-all-existential,
    location-scope, temporal-scope) trip. Never overrides a fired Stage36
    blocker. Shadow/diagnostic only -- never touches final logits or final
    classifier predictions.
    """
    result: dict[str, Any] = {
        "stage37_safe_recovery_fired": False,
        "stage37_safe_recovery_reasons": [],
        "stage37_safe_recovery_action": "none",
        "stage37_original_shadow_label": current_shadow_label,
        "stage37_final_shadow_label": current_shadow_label,
        "stage37_no_except_included_subset_fired": False,
        "stage37_coordination_universal_subset_fired": False,
        "stage37_numeric_universal_subset_fired": False,
        "stage37_recovered_from_label": None,
        "stage37_recovered_to_label": None,
        "stage37_blocked_by_stage36": False,
        "stage37_blocked_by_scope_hazard": False,
        "stage37_blocked_by_exception_hazard": False,
        "stage37_blocked_by_not_all_hazard": False,
    }

    if not getattr(args, "stage37_use_safe_support_recovery", False):
        return result
    if current_shadow_label not in ("NOT_ENTITLED", "REFUTE"):
        return result
    if stage36_info.get("stage36_support_blocker_fired"):
        result["stage37_blocked_by_stage36"] = True
        return result
    if current_shadow_label == "REFUTE" and not getattr(
        args, "stage37_allow_recover_from_refute", False
    ):
        return result

    if has_excluded_subset_hazard(claim, evidence):
        result["stage37_blocked_by_exception_hazard"] = True
        return result
    if has_not_all_existential_hazard(claim, evidence):
        result["stage37_blocked_by_not_all_hazard"] = True
        return result
    if has_location_scope_mismatch(claim, evidence) or has_temporal_scope_mismatch(
        claim, evidence
    ):
        result["stage37_blocked_by_scope_hazard"] = True
        return result

    reasons: list[str] = []
    fired_rule: "str | None" = None

    if getattr(args, "stage37_recover_no_except_included_subset", False):
        no_except_result = _stage37_no_except_included_subset_recovery(claim, evidence)
        if no_except_result.get("fired"):
            result["stage37_no_except_included_subset_fired"] = True
            reasons.append("no_except_included_subset_support")
            fired_rule = fired_rule or "no_except_included_subset_support"

    if getattr(args, "stage37_recover_coordination_universal_subset", False):
        coord_result = _stage37_coordination_universal_subset_recovery(claim, evidence)
        if coord_result.get("fired"):
            result["stage37_coordination_universal_subset_fired"] = True
            reasons.append("coordination_universal_subset_support")
            fired_rule = fired_rule or "coordination_universal_subset_support"

    if getattr(args, "stage37_recover_numeric_universal_subset", False):
        numeric_result = _stage37_numeric_universal_subset_recovery(claim, evidence)
        if numeric_result.get("fired"):
            result["stage37_numeric_universal_subset_fired"] = True
            reasons.append("numeric_universal_subset_support")
            fired_rule = fired_rule or "numeric_universal_subset_support"

    result["stage37_safe_recovery_reasons"] = reasons
    if not reasons:
        return result

    result["stage37_safe_recovery_fired"] = True
    result["stage37_safe_recovery_action"] = fired_rule or "safe_support_recovery"
    result["stage37_recovered_from_label"] = current_shadow_label
    result["stage37_recovered_to_label"] = "SUPPORT"
    result["stage37_final_shadow_label"] = "SUPPORT"
    return result


def _stage37_config_from_args(args: "argparse.Namespace") -> dict[str, Any]:
    """Build the Stage37-A safe SUPPORT recovery config dict from CLI args. All defaults off."""
    return {
        "enabled": getattr(args, "stage37_use_safe_support_recovery", False),
        "export": getattr(args, "stage37_safe_support_export", False),
        "shadow_mode": getattr(args, "stage37_safe_support_shadow_mode", False),
        "recover_no_except_included_subset": getattr(
            args, "stage37_recover_no_except_included_subset", False
        ),
        "recover_coordination_universal_subset": getattr(
            args, "stage37_recover_coordination_universal_subset", False
        ),
        "recover_numeric_universal_subset": getattr(
            args, "stage37_recover_numeric_universal_subset", False
        ),
        "allow_recover_from_refute": getattr(
            args, "stage37_allow_recover_from_refute", False
        ),
    }


#  Stage39-A: opt-in final composer (prediction/export-time only) 
#
# Off by default. Reuses the Stage32/Stage33/Stage36/Stage37 shadow labels
# already computed above; never touches final logits, training, or loss.
# When `--stage39-use-final-composer-opt-in` is absent, behavior, metrics and
# exported labels are identical to Stage38/Stage37. Only combining
# `--stage39-use-final-composer-opt-in` with
# `--stage39-final-composer-output-mode replace_pred_final_label` may replace
# the exported final prediction; `export_only` (the default) only exports
# `stage39_composed_final_label` alongside the unchanged `pred_final_label`.
_STAGE39_VALID_FINAL_LABELS = frozenset({"SUPPORT", "REFUTE", "NOT_ENTITLED"})


def _stage39_is_high_precision_contradiction(row: dict) -> bool:
    """Stage39-A: safe_structured REFUTE gate -- all three Stage33 fields must agree."""
    return (
        str(row.get("stage33_structured_coverage_reason") or "") == "none_to_some"
        and str(row.get("stage33_structured_coverage_route") or "") == "CONTRADICTION_REFUTE"
        and str(row.get("stage33_conditional_override_type") or "")
        == "high_precision_contradiction"
    )


def _stage39_high_precision_contradiction_trigger_v2(row: dict) -> dict[str, Any]:
    """Stage39-C: safe_structured_v2 REFUTE gate diagnostics.

    `_stage39_is_high_precision_contradiction` requires Stage33 route,
    override, and reason=="none_to_some" to *all* agree, which silently
    excludes the equally high-precision "some_to_none" contradiction group.
    This broadens the gate to an OR across four independent high-precision-
    contradiction signals (Stage33 route, Stage33 structured label, a
    Stage33/Stage36/Stage37 conditional override, or the Stage33 reason
    being none_to_some/some_to_none), any one of which is sufficient.
    Never reads gold labels.
    """
    route = str(row.get("stage33_structured_coverage_route") or "")
    label = str(row.get("stage33_structured_coverage_label") or "")
    reason = str(row.get("stage33_structured_coverage_reason") or "")
    override_value = next(
        (
            str(row.get(key))
            for key in (
                "stage33_conditional_override_type",
                "stage36_conditional_override_type",
                "stage37_conditional_override_type",
            )
            if str(row.get(key) or "") == "high_precision_contradiction"
        ),
        None,
    )

    route_fired = route == "CONTRADICTION_REFUTE"
    label_fired = label == "STRUCT_CONTRADICTION_REFUTE"
    override_fired = override_value is not None
    reason_fired = reason in {"none_to_some", "some_to_none"}
    fired = route_fired or label_fired or override_fired or reason_fired

    return {
        "fired": fired,
        "reason": reason if fired and reason else None,
        "route": route if fired and route else None,
        "override": override_value if fired else None,
    }


def _stage39_blocked_action_reason(result: dict[str, Any]) -> "tuple[str, str]":
    if result["stage39_blocked_by_stage36"]:
        return "blocked_by_stage36", "stage36_support_safety_blocker_fired"
    if result["stage39_blocked_by_refute_to_support_guard"]:
        return "blocked_by_refute_to_support_guard", "refute_to_support_guard_active"
    if result["stage39_blocked_by_stage37_from_refute_guard"]:
        return (
            "blocked_by_stage37_from_refute_guard",
            "stage37_recovered_from_refute_guard_active",
        )
    return "no_change", "blocked_unknown"


def compute_stage39_final_composer(row: dict, args: "argparse.Namespace") -> dict[str, Any]:
    """Stage39-A: deterministic opt-in final composer.

    Never trains on, calibrates against, or selects checkpoints from this
    computation -- prediction/export-time composition only. `row` must carry
    the current final label (`pred_final_label`), the candidate shadow
    source labels (`stage37_final_shadow_label`, `stage36_final_shadow_label`,
    `stage32_shadow_label`), the Stage36 blocker flag
    (`stage36_support_blocker_fired`), the Stage37 recovery provenance
    (`stage37_recovered_from_label`), and the Stage33 structured fields used
    for the safe_structured / safe_structured_v2 high-precision-contradiction
    gates (`stage33_structured_coverage_reason`, `_route`, `_label`,
    `stage33_conditional_override_type`, and, if present,
    `stage36_conditional_override_type` / `stage37_conditional_override_type`).
    `row` may also carry a gold label for reporting only -- it is never read
    here for any decision.
    """
    enabled = bool(getattr(args, "stage39_use_final_composer_opt_in", False))
    policy = getattr(args, "stage39_final_composer_policy", "support_only")
    output_mode = getattr(args, "stage39_final_composer_output_mode", "export_only")
    source_name = getattr(
        args, "stage39_final_composer_source", "stage37_final_shadow_label"
    )
    disallow_refute_to_support = bool(
        getattr(args, "stage39_disallow_refute_to_support", True)
    )
    require_stage36_clear = bool(
        getattr(args, "stage39_require_stage36_safety_clear", True)
    )
    require_stage37_not_from_refute = bool(
        getattr(args, "stage39_require_stage37_not_from_refute", True)
    )

    original_final_label = str(row.get("pred_final_label") or "")

    result: dict[str, Any] = {
        "stage39_final_composer_enabled": enabled,
        "stage39_final_composer_policy": policy,
        "stage39_final_composer_output_mode": output_mode,
        "stage39_original_final_label": original_final_label,
        "stage39_source_shadow_label": None,
        "stage39_composed_final_label": original_final_label,
        "stage39_final_label_changed": False,
        "stage39_composer_action": "disabled",
        "stage39_composer_reason": "stage39_disabled",
        "stage39_blocked_by_stage36": False,
        "stage39_blocked_by_refute_to_support_guard": False,
        "stage39_blocked_by_stage37_from_refute_guard": False,
        "stage39_blocked_by_missing_source": False,
        "stage39_refute_trigger": False,
        "stage39_refute_trigger_reason": None,
        "stage39_refute_trigger_route": None,
        "stage39_refute_trigger_override": None,
    }
    if not enabled:
        return result

    source_lookup = {
        "stage37_final_shadow_label": row.get("stage37_final_shadow_label"),
        "stage36_final_shadow_label": row.get("stage36_final_shadow_label"),
        "stage32_shadow_label": row.get("stage32_shadow_label"),
    }
    source_label_raw = source_lookup.get(source_name)
    source_label = (
        str(source_label_raw)
        if source_label_raw is not None
        and str(source_label_raw) in _STAGE39_VALID_FINAL_LABELS
        else None
    )
    result["stage39_source_shadow_label"] = source_label

    if source_label is None:
        result["stage39_blocked_by_missing_source"] = True
        result["stage39_composer_action"] = "blocked_by_missing_source"
        result["stage39_composer_reason"] = "missing_source_shadow_label"
        return result

    stage36_blocker_fired = bool(row.get("stage36_support_blocker_fired", False))
    stage37_recovered_from_refute = (
        str(row.get("stage37_recovered_from_label") or "") == "REFUTE"
    )

    def _guard_support_composition() -> bool:
        allowed = True
        if original_final_label == "REFUTE" and disallow_refute_to_support:
            result["stage39_blocked_by_refute_to_support_guard"] = True
            allowed = False
        if require_stage36_clear and stage36_blocker_fired:
            result["stage39_blocked_by_stage36"] = True
            allowed = False
        if require_stage37_not_from_refute and stage37_recovered_from_refute:
            result["stage39_blocked_by_stage37_from_refute_guard"] = True
            allowed = False
        return allowed

    candidate_label = original_final_label
    action = "no_change"
    reason = "source_matches_original"

    if policy == "support_only":
        if source_label != "SUPPORT":
            action, reason = "no_change", "support_only_policy_restricts_to_support"
        elif original_final_label == "SUPPORT":
            action, reason = "no_change", "already_support"
        elif _guard_support_composition():
            candidate_label = "SUPPORT"
            action, reason = "composed_to_support", "support_only_from_stage37"
        else:
            action, reason = _stage39_blocked_action_reason(result)

    elif policy == "safe_structured":
        if source_label == "SUPPORT" and original_final_label != "SUPPORT":
            if _guard_support_composition():
                candidate_label = "SUPPORT"
                action, reason = "composed_to_support", "support_only_from_stage37"
            else:
                action, reason = _stage39_blocked_action_reason(result)
        elif source_label == "NOT_ENTITLED" and original_final_label == "SUPPORT":
            candidate_label = "NOT_ENTITLED"
            action, reason = (
                "composed_to_not_entitled",
                "overclaim_not_entitled_from_stage37_shadow",
            )
        elif original_final_label in {
            "SUPPORT",
            "NOT_ENTITLED",
        } and _stage39_is_high_precision_contradiction(row):
            candidate_label = "REFUTE"
            action, reason = (
                "composed_to_refute",
                "high_precision_contradiction_from_stage33",
            )
        else:
            action, reason = "no_change", "safe_structured_no_qualifying_transition"

    elif policy == "safe_structured_v2":
        trigger = _stage39_high_precision_contradiction_trigger_v2(row)
        result["stage39_refute_trigger"] = trigger["fired"]
        result["stage39_refute_trigger_reason"] = trigger["reason"]
        result["stage39_refute_trigger_route"] = trigger["route"]
        result["stage39_refute_trigger_override"] = trigger["override"]

        if source_label == "SUPPORT" and original_final_label != "SUPPORT":
            if _guard_support_composition():
                candidate_label = "SUPPORT"
                action, reason = "composed_to_support", "support_only_from_stage37"
            else:
                action, reason = _stage39_blocked_action_reason(result)
        elif source_label == "NOT_ENTITLED" and original_final_label == "SUPPORT":
            candidate_label = "NOT_ENTITLED"
            action, reason = (
                "composed_to_not_entitled",
                "overclaim_not_entitled_from_stage37_shadow",
            )
        elif original_final_label == "NOT_ENTITLED" and trigger["fired"]:
            candidate_label = "REFUTE"
            action, reason = (
                "composed_to_refute",
                "safe_structured_v2_high_precision_contradiction",
            )
        else:
            action, reason = "no_change", "safe_structured_v2_no_qualifying_transition"

    elif policy == "full_shadow":
        candidate_label = source_label
        if candidate_label == original_final_label:
            action, reason = "no_change", "source_matches_original"
        elif candidate_label == "SUPPORT":
            if _guard_support_composition():
                action, reason = "composed_to_support", "full_shadow_support_adoption"
            else:
                candidate_label = original_final_label
                action, reason = _stage39_blocked_action_reason(result)
        elif candidate_label == "REFUTE":
            action, reason = "composed_to_refute", "full_shadow_refute_adoption"
        elif candidate_label == "NOT_ENTITLED":
            action, reason = (
                "composed_to_not_entitled",
                "full_shadow_not_entitled_adoption",
            )
        else:
            candidate_label = original_final_label
            action, reason = "no_change", "full_shadow_invalid_source_label"

    result["stage39_composed_final_label"] = candidate_label
    result["stage39_final_label_changed"] = candidate_label != original_final_label
    result["stage39_composer_action"] = action
    result["stage39_composer_reason"] = reason
    return result


def flatten_stage32_owner_state(state: dict[str, Any]) -> dict[str, Any]:
    hard_core = state["hard_core"]
    coverage = state["coverage_entailment"]
    coverage_v2 = state["coverage_v2"]
    structured = state["structured_coverage"]
    residual = state["residual_adjudication"]
    ani = state["ani_diagnostic"]
    polarity = state["polarity"]
    shadow = state["composer_shadow"]
    return {
        "stage32_hard_core_prob": hard_core["prob"],
        "stage32_hard_core_pass": hard_core["pass"],
        "stage32_hard_core_block_reason": hard_core["block_reason"],
        "stage32_hard_core_notes": hard_core.get("notes"),
        "stage32_coverage_entails_support_prob": coverage["entails_support_prob"],
        "stage32_coverage_overclaim_ne_prob": coverage["overclaim_ne_prob"],
        "stage32_coverage_contradicts_refute_prob": coverage["contradicts_refute_prob"],
        "stage32_coverage_pred_label": coverage["pred_label"],
        "stage32_coverage_pred_id": coverage["pred_id"],
        "stage32_coverage_confidence": coverage["confidence"],
        "stage32_coverage_input_mode": coverage["input_mode"],
        "stage32_coverage_notes": coverage.get("notes"),
        "stage32_coverage_v2_enabled": coverage_v2["enabled"],
        "stage32_coverage_v2_top_prob": coverage_v2["top_prob"],
        "stage32_coverage_v2_second_prob": coverage_v2["second_prob"],
        "stage32_coverage_v2_margin": coverage_v2["margin"],
        "stage32_coverage_v2_min_confidence": coverage_v2["min_confidence"],
        "stage32_coverage_v2_min_margin": coverage_v2["min_margin"],
        "stage32_coverage_v2_pred_label": coverage_v2["pred_label"],
        "stage32_coverage_v2_route": coverage_v2["route"],
        "stage32_coverage_v2_reason": coverage_v2["reason"],
        "stage32_coverage_v2_abstained": coverage_v2["abstained"],
        "stage33_structured_coverage_enabled": structured["enabled"],
        "stage33_structured_coverage_label": structured["label"],
        "stage33_structured_coverage_route": structured["route"],
        "stage33_structured_coverage_reason": structured["reason"],
        "stage33_structured_coverage_original_reason": structured["original_reason"],
        "stage33_structured_coverage_rule_strength": structured["rule_strength"],
        "stage33_structured_coverage_confidence": structured["confidence"],
        "stage33_structured_coverage_whole_part_enabled": structured[
            "whole_part_enabled"
        ],
        "stage33_structured_coverage_whole_part_relation": structured[
            "whole_part_relation"
        ],
        "stage33_structured_coverage_whole_part_match": structured[
            "whole_part_match"
        ],
        "stage33_structured_coverage_whole_part_direct_support_enabled": structured[
            "whole_part_direct_support_enabled"
        ],
        "stage33_structured_coverage_whole_part_v2_enabled": structured[
            "whole_part_v2_enabled"
        ],
        "stage33_structured_coverage_whole_part_v2_expanded_lexicon": structured[
            "whole_part_v2_expanded_lexicon"
        ],
        "stage33_structured_coverage_whole_part_v2_direct_support_policy": structured[
            "whole_part_v2_direct_support_policy"
        ],
        "stage33_whole_part_direct_support_candidate": structured[
            "whole_part_direct_support_candidate"
        ],
        "stage33_whole_part_direct_support_allowed": structured[
            "whole_part_direct_support_allowed"
        ],
        "stage33_whole_part_direct_support_block_reason": structured[
            "whole_part_direct_support_block_reason"
        ],
        "stage33_whole_part_direct_support_action_block_reason": structured[
            "whole_part_direct_support_action_block_reason"
        ],
        "stage33_whole_part_conditional_safe_override_hard_core_enabled": structured[
            "whole_part_conditional_safe_override_hard_core_enabled"
        ],
        "stage33_whole_part_hard_core_pass": structured["whole_part_hard_core_pass"],
        "stage33_whole_part_original_current_label": structured[
            "whole_part_original_current_label"
        ],
        "stage33_whole_part_conditional_action": structured[
            "whole_part_conditional_action"
        ],
        "stage33_structured_coverage_direct_support_allowed": structured[
            "direct_support_allowed"
        ],
        "stage33_structured_coverage_direct_support_block_reason": structured[
            "direct_support_block_reason"
        ],
        "stage33_structured_coverage_claim_cues": structured["claim_cues"],
        "stage33_structured_coverage_evidence_cues": structured["evidence_cues"],
        "stage33_structured_coverage_rule_fired": structured["rule_fired"],
        "stage33_structured_coverage_priority_trace": " | ".join(
            structured.get("priority_trace", [])
        ),
        "stage32_residual_prob": residual["residual_prob"],
        "stage32_residual_pred_label": residual["pred_label"],
        "stage32_residual_notes": residual.get("notes"),
        "stage32_ani_novelty_prob": ani["novelty_prob"],
        "stage32_ani_ambiguity_prob": ani["ambiguity_prob"],
        "stage32_ani_ignorance_prob": ani["ignorance_prob"],
        "stage32_ani_pred_label": ani["pred_label"],
        "stage32_ani_notes": ani.get("notes"),
        "stage32_polarity_support_energy": polarity["support_energy"],
        "stage32_polarity_refute_energy": polarity["refute_energy"],
        "stage32_polarity_support_prob": polarity["support_prob"],
        "stage32_polarity_refute_prob": polarity["refute_prob"],
        "stage32_polarity_pred_label": polarity["pred_label"],
        "stage32_polarity_notes": polarity.get("notes"),
        "stage32_shadow_label": shadow["shadow_label"],
        "stage32_shadow_reason": shadow["shadow_reason"],
        "stage32_shadow_priority_trace": " | ".join(shadow.get("priority_trace", [])),
        "stage32_shadow_would_block_support": shadow["would_block_support"],
        "stage32_shadow_would_route_ne": shadow["would_route_ne"],
        "stage32_shadow_would_route_refute": shadow["would_route_refute"],
        "stage33_conditional_fallback_enabled": shadow[
            "stage33_conditional_fallback_enabled"
        ],
        "stage33_conditional_fallback_source": shadow[
            "stage33_conditional_fallback_source"
        ],
        "stage33_conditional_action": shadow["stage33_conditional_action"],
        "stage33_conditional_fallback_used": shadow[
            "stage33_conditional_fallback_used"
        ],
        "stage33_conditional_override_applied": shadow[
            "stage33_conditional_override_applied"
        ],
        "stage33_conditional_override_type": shadow[
            "stage33_conditional_override_type"
        ],
        "stage33_conditional_original_current_label": shadow[
            "stage33_conditional_original_current_label"
        ],
        "stage33_conditional_shadow_label": shadow[
            "stage33_conditional_shadow_label"
        ],
    }


_STAGE113_VNEXT_SCALAR_FIELDS: tuple[str, ...] = (
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "entitlement_prob",
    "entitlement_for_decision",
    "compositional_entitlement_prob",
    "learned_entitlement_prob",
    "learned_entitlement_logit",
    "polarity_margin",
    "positive_energy",
    "negative_energy",
)

_STAGE113_VNEXT_METADATA_FIELDS: tuple[str, ...] = (
    "vnext_router_mode",
    "vnext_final_logit_order",
)

_STAGE113_VNEXT_EXPORT_FIELDS: tuple[str, ...] = (
    *_STAGE113_VNEXT_SCALAR_FIELDS,
    *_STAGE113_VNEXT_METADATA_FIELDS,
)

_STAGE123_VNEXT_EVIDENCE_EXPORT_FIELDS: tuple[str, ...] = (
    "vnext_evidence_interface",
    "stage118_diagnostic_evidence_interface",
    "vnext_resolved_evidence",
    "vnext_evidence_core_text",
    "vnext_evidence_context_text",
    "vnext_evidence_interface_fallback_used",
    "vnext_evidence_interface_notes",
    "vnext_segmented_dual_pass_active",
    "vnext_segmented_context_role",
    "vnext_primary_rep_source",
    "vnext_core_text_for_encoding",
    "vnext_context_text_for_encoding",
    "vnext_context_empty",
    "vnext_core_rep_norm",
    "vnext_context_rep_norm",
    "vnext_core_context_cosine",
    "vnext_context_risk_cap_active",
    "vnext_context_risk_source",
    "vnext_context_risk",
    "vnext_context_risk_threshold",
    "vnext_context_risk_cap_alpha",
    "vnext_context_risk_excess",
    "vnext_context_cap_factor",
    "vnext_context_cap_applied",
    "vnext_logits_before_context_cap",
    "vnext_logits_after_context_cap",
    "vnext_prediction_before_context_cap",
    "vnext_prediction_after_context_cap",
    "vnext_context_only_logits",
    "vnext_context_only_prediction",
    "vnext_context_cap_notes",
)

_STAGE118_PRESERVED_FIELD_PREFIXES: tuple[str, ...] = (
    "stage",
    "metadata",
    "source_id",
    "intervention_type",
    "primary_failure_type",
    "pair_id",
)

_STAGE118_CORE_PREDICTION_FIELDS: set[str] = {
    "claim",
    "evidence",
    "gold_label",
    "base_prediction",
    "prediction",
    "logits",
    "final_logits",
    "pred_final_label",
    "pred_label",
    *_STAGE113_VNEXT_EXPORT_FIELDS,
    *_STAGE123_VNEXT_EVIDENCE_EXPORT_FIELDS,
}

_STAGE118_BUCKETS: tuple[str, ...] = (
    "correct_SUPPORT",
    "correct_REFUTE",
    "correct_NE",
    "false_NE_SUPPORT",
    "false_NE_REFUTE",
    "false_entitlement",
    "SUPPORT_to_REFUTE",
    "REFUTE_to_SUPPORT",
)


def _vnext_clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _vnext_first_text(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, str | None]:
    for field in fields:
        text = _vnext_clean_text(record.get(field))
        if text:
            return text, field
    return "", None


def _stage128_extract_controlled_location_slot(text: Any) -> str:
    """Extract the controlled `in <Location> during` slot; not general NER."""
    cleaned = _vnext_clean_text(text)
    if not cleaned:
        return ""
    match = _STAGE128_IN_DURING_LOCATION_RE.search(cleaned)
    return _vnext_clean_text(match.group(1)) if match else ""


def _stage128_resolve_evidence_core_text(record: dict[str, Any]) -> tuple[str, str | None]:
    return _vnext_first_text(
        record,
        (
            "stage122_evidence_core",
            "stage122_core_text",
            "stage121_evidence_core",
            "evidence_core",
            "evidence",
        ),
    )


def _stage128_location_slot_guard_exports(
    record: dict[str, Any],
    *,
    prediction_before_guard: str,
    args: argparse.Namespace | None,
) -> dict[str, Any]:
    enabled = bool(
        getattr(args, "stage128_enable_location_slot_guard", False)
    ) if args is not None else False
    mode = (
        getattr(args, "stage128_location_slot_guard_mode", "off")
        if args is not None else "off"
    )
    if not enabled or mode == "off":
        return {}

    claim_location = ""
    evidence_location = ""
    mismatch = False
    applied = False
    prediction_after_guard = prediction_before_guard
    notes: list[str] = ["eval_only_controlled_pattern_not_general_ner"]

    if mode == "controlled_in_during_location_mismatch":
        evidence_text, evidence_source = _stage128_resolve_evidence_core_text(record)
        claim_location = _stage128_extract_controlled_location_slot(record.get("claim"))
        evidence_location = _stage128_extract_controlled_location_slot(evidence_text)
        mismatch = bool(
            claim_location
            and evidence_location
            and claim_location != evidence_location
        )
        notes.append(f"evidence_source={evidence_source or 'none'}")
        if prediction_before_guard == "SUPPORT" and mismatch:
            prediction_after_guard = "NOT_ENTITLED"
            applied = True
            notes.append("support_to_not_entitled_location_mismatch")
        elif prediction_before_guard != "SUPPORT":
            notes.append("non_support_prediction_unchanged")
        elif not claim_location or not evidence_location:
            notes.append("missing_controlled_location_slot")
        else:
            notes.append("locations_match_or_no_mismatch")
    else:
        notes.append(f"unsupported_mode={mode}")

    return {
        "stage128_location_slot_guard_enabled": enabled,
        "stage128_location_slot_guard_mode": mode,
        "stage128_claim_location": claim_location,
        "stage128_evidence_location": evidence_location,
        "stage128_location_mismatch": mismatch,
        "stage128_prediction_before_location_guard": prediction_before_guard,
        "stage128_prediction_after_location_guard": prediction_after_guard,
        "stage128_location_guard_applied": applied,
        "stage128_location_guard_notes": ";".join(notes),
    }


def resolve_vnext_evidence_text(record: dict[str, Any], evidence_interface: str) -> dict[str, Any]:
    """Resolve opt-in vNext evidence text without reading labels or decisions."""
    if evidence_interface not in VNEXT_EVIDENCE_INTERFACE_CHOICES:
        raise ValueError(f"unknown vNext evidence interface: {evidence_interface!r}")

    claim = record.get("claim")
    evidence_text = _vnext_clean_text(record.get("evidence"))
    core_text, core_source = _vnext_first_text(
        record,
        ("evidence_core", "stage122_evidence_core", "stage121_original_evidence", "evidence"),
    )
    context_text, context_source = _vnext_first_text(
        record,
        ("evidence_context", "context_prefix", "stage122_prefix_text", "stage121_prefix_text"),
    )
    fallback_used = False
    notes: list[str] = []

    if evidence_interface == "full_evidence":
        resolved = evidence_text
        notes.append("full_evidence_uses_evidence")
    elif evidence_interface == "core_only":
        resolved = core_text
        fallback_used = core_source != "evidence_core"
        notes.append(f"core_source={core_source or 'none'}")
    elif evidence_interface == "core_first_context_suffix":
        resolved = " ".join(part for part in (core_text, context_text) if part)
        fallback_used = core_source != "evidence_core" or context_source != "evidence_context"
        notes.append(f"core_source={core_source or 'none'}")
        notes.append(f"context_source={context_source or 'none'}")
    elif evidence_interface == "context_prefix_core":
        resolved = " ".join(part for part in (context_text, core_text) if part)
        fallback_used = core_source != "evidence_core" or context_source != "evidence_context"
        notes.append("prefix_control_interface")
        notes.append(f"core_source={core_source or 'none'}")
        notes.append(f"context_source={context_source or 'none'}")
    elif evidence_interface == "core_marker_context_suffix":
        resolved = f"Evidence: {core_text}" if core_text else ""
        if context_text:
            resolved = f"{resolved} Context: {context_text}" if resolved else f"Context: {context_text}"
        fallback_used = core_source != "evidence_core" or context_source != "evidence_context"
        notes.append(f"core_source={core_source or 'none'}")
        notes.append(f"context_source={context_source or 'none'}")
    elif evidence_interface == "segmented_dual_pass_scaffold":
        resolved = core_text
        fallback_used = core_source != "evidence_core" or context_source != "evidence_context"
        notes.append("segmented_dual_pass_scaffold_single_pass_core_sequence")
        notes.append(f"core_source={core_source or 'none'}")
        notes.append(f"context_source={context_source or 'none'}")
    elif evidence_interface == "segmented_dual_pass":
        resolved = core_text
        fallback_used = core_source != "evidence_core" or context_source != "evidence_context"
        notes.append("segmented_dual_pass_core_primary_dual_encoder")
        notes.append(f"core_source={core_source or 'none'}")
        notes.append(f"context_source={context_source or 'none'}")
    else:
        resolved = evidence_text

    if not resolved:
        resolved = evidence_text
        fallback_used = True
        notes.append("empty_resolved_evidence_fell_back_to_evidence")

    return {
        "resolved_claim": claim,
        "resolved_evidence": resolved,
        "evidence_core_text": core_text,
        "evidence_context_text": context_text,
        "core_sequence_text": core_text,
        "context_sequence_text": context_text,
        "evidence_interface": evidence_interface,
        "evidence_interface_fallback_used": bool(fallback_used),
        "evidence_interface_notes": ";".join(notes),
    }


def resolve_stage118_diagnostic_evidence_interface(args: argparse.Namespace) -> str:
    requested = getattr(args, "stage118_diagnostic_evidence_interface", "same_as_vnext")
    if requested == "same_as_vnext":
        return getattr(args, "vnext_evidence_interface", "full_evidence")
    return requested


def parse_stage118_diagnostic_evidence_interface_sweep(raw_value: str | None) -> list[str]:
    if raw_value is None or not raw_value.strip():
        return []
    interfaces: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_value.split(","):
        interface = raw_item.strip()
        if not interface:
            continue
        if interface not in VNEXT_EVIDENCE_INTERFACE_CHOICES:
            allowed = ", ".join(VNEXT_EVIDENCE_INTERFACE_CHOICES)
            raise ValueError(
                "--stage118-diagnostic-evidence-interface-sweep contains "
                f"invalid interface {interface!r}; allowed values: {allowed}"
            )
        if interface in seen:
            continue
        interfaces.append(interface)
        seen.add(interface)
    return interfaces


def apply_vnext_evidence_interface_to_records(
    records: list[dict[str, Any]],
    evidence_interface: str,
) -> list[dict[str, Any]]:
    resolved_records: list[dict[str, Any]] = []
    for record in records:
        resolved = resolve_vnext_evidence_text(record, evidence_interface)
        item = dict(record)
        item["claim"] = resolved["resolved_claim"]
        item["evidence"] = resolved["resolved_evidence"]
        item["vnext_evidence_interface"] = resolved["evidence_interface"]
        item["vnext_resolved_evidence"] = resolved["resolved_evidence"]
        item["vnext_evidence_core_text"] = resolved["evidence_core_text"]
        item["vnext_evidence_context_text"] = resolved["evidence_context_text"]
        item["vnext_core_sequence_text"] = resolved["core_sequence_text"]
        item["vnext_context_sequence_text"] = resolved["context_sequence_text"]
        item["vnext_evidence_interface_fallback_used"] = resolved[
            "evidence_interface_fallback_used"
        ]
        item["vnext_evidence_interface_notes"] = resolved["evidence_interface_notes"]
        resolved_records.append(item)
    return resolved_records

def _vnext_segmented_dual_pass_active(
    args: argparse.Namespace,
    evidence_interface: str | None = None,
) -> bool:
    selected_interface = evidence_interface or getattr(args, "vnext_evidence_interface", "full_evidence")
    return (
        getattr(args, "architecture", None) == "vnext_minimal"
        and bool(getattr(args, "vnext_enable_segmented_dual_pass", False))
        and selected_interface == "segmented_dual_pass"
    )


def _vnext_segmented_record_view(
    records: list[dict[str, Any]],
    *,
    text_key: str,
    empty_placeholder: str = "[empty_context]",
) -> tuple[list[dict[str, Any]], list[bool], list[str]]:
    encoded_records: list[dict[str, Any]] = []
    empty_flags: list[bool] = []
    texts: list[str] = []
    for record in records:
        text = _vnext_clean_text(record.get(text_key))
        empty = not bool(text)
        if empty:
            text = empty_placeholder
        item = dict(record)
        item["evidence"] = text
        encoded_records.append(item)
        empty_flags.append(empty)
        texts.append("" if empty else text)
    return encoded_records, empty_flags, texts


def _vnext_encode_segmented_features(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    vocab: dict[str, int] | None,
    tokenizer: Any | None,
) -> tuple[dict[str, torch.Tensor], list[bool], list[str], list[str]]:
    core_records, _, core_texts = _vnext_segmented_record_view(
        records, text_key="vnext_core_sequence_text", empty_placeholder="[empty_core]"
    )
    context_records, context_empty, context_texts = _vnext_segmented_record_view(
        records, text_key="vnext_context_sequence_text", empty_placeholder="[empty_context]"
    )
    if args.backbone == "dummy":
        if vocab is None:
            raise ValueError("segmented dual-pass dummy encoding requires vocab")
        core_bundle = v5.encode_records(core_records, vocab)
        context_bundle = v5.encode_records(context_records, vocab)
    else:
        if tokenizer is None:
            raise ValueError("segmented dual-pass Mamba encoding requires tokenizer")
        core_bundle = v5.encode_mamba_records(core_records, tokenizer, args.max_length)
        context_bundle = v5.encode_mamba_records(context_records, tokenizer, args.max_length)
    features: dict[str, torch.Tensor] = {}
    for source_key, target_key in (
        ("input_ids", "core_input_ids"),
        ("attention_mask", "core_attention_mask"),
        ("claim_mask", "core_claim_mask"),
        ("evidence_mask", "core_evidence_mask"),
    ):
        features[target_key] = core_bundle["model_inputs"][source_key]
    for source_key, target_key in (
        ("input_ids", "context_input_ids"),
        ("attention_mask", "context_attention_mask"),
        ("claim_mask", "context_claim_mask"),
        ("evidence_mask", "context_evidence_mask"),
    ):
        features[target_key] = context_bundle["model_inputs"][source_key]
    features["context_empty"] = torch.tensor(context_empty, dtype=torch.bool)
    return features, context_empty, core_texts, context_texts


def attach_vnext_segmented_dual_pass_inputs(
    inputs: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    vocab: dict[str, int] | None,
    tokenizer: Any | None,
    device: torch.device,
    evidence_interface: str | None = None,
) -> None:
    if not _vnext_segmented_dual_pass_active(args, evidence_interface):
        return
    features, context_empty, core_texts, context_texts = _vnext_encode_segmented_features(
        records, args=args, vocab=vocab, tokenizer=tokenizer
    )
    for key, value in features.items():
        inputs[key] = value.to(device)
    for record, is_empty, core_text, context_text in zip(
        records, context_empty, core_texts, context_texts
    ):
        record["vnext_segmented_dual_pass_active"] = True
        record["vnext_segmented_context_role"] = getattr(
            args, "vnext_segmented_context_role", "diagnostic_only"
        )
        record["vnext_primary_rep_source"] = "core_rep"
        record["vnext_core_text_for_encoding"] = core_text
        record["vnext_context_text_for_encoding"] = context_text
        record["vnext_context_empty"] = bool(is_empty)


_VNEXT_OPTIONAL_MODEL_FEATURE_KEYS: tuple[str, ...] = (
    "core_input_ids",
    "core_attention_mask",
    "core_claim_mask",
    "core_evidence_mask",
    "context_input_ids",
    "context_attention_mask",
    "context_claim_mask",
    "context_evidence_mask",
    "context_empty",
)


def _vnext_model_feature_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Forward legacy model features plus opt-in segmented dual-pass tensors."""
    result = {key: inputs[key] for key in v5.MODEL_FEATURE_KEYS}
    for key in _VNEXT_OPTIONAL_MODEL_FEATURE_KEYS:
        if key in inputs:
            result[key] = inputs[key]
    if "encoder_hidden_states" in inputs:
        result["encoder_hidden_states"] = inputs["encoder_hidden_states"]
    return result


def _assert_model_accepts_feature_kwargs(
    model: nn.Module,
    feature_inputs: dict[str, torch.Tensor],
    *,
    context: str,
) -> None:
    optional_keys = [key for key in _VNEXT_OPTIONAL_MODEL_FEATURE_KEYS if key in feature_inputs]
    if not optional_keys:
        return
    signature = inspect.signature(model.forward)
    parameters = signature.parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return
    unsupported = sorted(key for key in optional_keys if key not in parameters)
    if unsupported:
        raise TypeError(
            f"{type(model).__name__}.forward does not accept Stage126 segmented "
            f"kwargs in {context}: " + ", ".join(unsupported)
        )

def _stage113_jsonable_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            item = detached.item()
            return round(float(item), 6) if isinstance(item, (int, float)) else item
        return detached.tolist()
    if isinstance(value, (list, tuple)):
        return [_stage113_jsonable_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stage113_jsonable_metadata(item) for key, item in value.items()}
    return str(value)


def _stage113_scalar_value(output: dict[str, Any], key: str, index: int) -> float | None:
    value = output.get(key)
    if value is None:
        return None
    try:
        if torch.is_tensor(value):
            tensor_value = value.detach().cpu()
            if tensor_value.ndim == 0:
                if index != 0:
                    return None
                row_value = tensor_value
            else:
                if index >= int(tensor_value.shape[0]):
                    return None
                row_value = tensor_value[index]
            if not torch.is_tensor(row_value) or row_value.numel() != 1:
                return None
            return round(float(row_value.item()), 6)
        if isinstance(value, (list, tuple)):
            if index >= len(value):
                return None
            row_value = value[index]
            if isinstance(row_value, (int, float)):
                return round(float(row_value), 6)
    except (TypeError, ValueError, IndexError, RuntimeError):
        return None
    return None


def _stage113_vnext_export_values(output: dict[str, Any], index: int) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in _STAGE113_VNEXT_SCALAR_FIELDS:
        scalar = _stage113_scalar_value(output, key, index)
        if scalar is not None:
            values[key] = scalar
    for key in _STAGE113_VNEXT_METADATA_FIELDS:
        if key in output and output.get(key) is not None:
            values[key] = _stage113_jsonable_metadata(output.get(key))
    return values


def _stage113_add_vnext_scalars(row: dict[str, Any], output: dict[str, Any], index: int) -> None:
    row.update(_stage113_vnext_export_values(output, index))


def _stage113_prediction_row_exports(prediction_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    exports: dict[str, dict[str, Any]] = {}
    for row in prediction_rows:
        row_id = row.get("id")
        if row_id is None:
            continue
        exported = {key: row[key] for key in _STAGE113_VNEXT_EXPORT_FIELDS if key in row}
        if exported:
            exports[str(row_id)] = exported
    return exports


def _stage113_merge_prediction_exports(
    stage43_predictions: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    exports_by_id = _stage113_prediction_row_exports(prediction_rows)
    if not exports_by_id:
        return
    for row in stage43_predictions:
        row_exports = exports_by_id.get(str(row.get("id")))
        if row_exports:
            row.update(row_exports)


def _stage113_vnext_scalar_report(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = sorted(
        key
        for key in _STAGE113_VNEXT_EXPORT_FIELDS
        if any(key in row for row in prediction_rows)
    )
    missing = sorted(key for key in _STAGE113_VNEXT_EXPORT_FIELDS if key not in present)
    return {
        "stage113_vnext_scalar_export_enabled": bool(present),
        "stage113_vnext_scalar_fields_present": present,
        "stage113_vnext_scalar_fields_missing": missing,
    }


def _stage115_clean_dev_scalar_rows(
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in prediction_rows:
        item: dict[str, Any] = {}
        if row.get("id") is not None:
            item["id"] = row.get("id")
        item["claim"] = row.get("claim")
        item["evidence"] = row.get("evidence")
        item["gold_label"] = row.get("gold_label")
        item["prediction"] = (
            row.get("prediction")
            if row.get("prediction") is not None
            else row.get(
                "base_prediction",
                row.get("pred_label", row.get("pred_final_label")),
            )
        )
        for key in _STAGE113_VNEXT_EXPORT_FIELDS:
            if key in row:
                item[key] = row[key]
        exported.append(item)
    return exported


def _stage115_clean_dev_scalar_report(
    prediction_rows: list[dict[str, Any]],
    output_jsonl: Path,
) -> dict[str, Any]:
    present = sorted(
        key
        for key in _STAGE113_VNEXT_EXPORT_FIELDS
        if any(key in row for row in prediction_rows)
    )
    missing = sorted(key for key in _STAGE113_VNEXT_EXPORT_FIELDS if key not in present)
    return {
        "stage115_clean_dev_scalar_output_jsonl": str(output_jsonl),
        "stage115_clean_dev_scalar_export_enabled": True,
        "stage115_clean_dev_scalar_fields_present": present,
        "stage115_clean_dev_scalar_fields_missing": missing,
        "stage115_clean_dev_scalar_row_count": len(prediction_rows),
    }



_STAGE125_RISK_CAP_REQUIRED_EXPORT_FIELDS: tuple[str, ...] = (
    "vnext_context_risk_cap_active",
    "vnext_context_risk_source",
    "vnext_context_risk",
    "vnext_context_risk_threshold",
    "vnext_context_risk_cap_alpha",
    "vnext_context_risk_excess",
    "vnext_context_cap_factor",
    "vnext_context_cap_applied",
    "vnext_logits_before_context_cap",
    "vnext_logits_after_context_cap",
    "vnext_prediction_before_context_cap",
    "vnext_prediction_after_context_cap",
    "vnext_context_only_logits",
    "vnext_context_only_prediction",
    "vnext_context_cap_notes",
)


def _stage125_assert_risk_cap_exports(row: dict[str, Any]) -> None:
    interface = row.get("stage118_diagnostic_evidence_interface") or row.get(
        "vnext_evidence_interface"
    )
    segmented_interface = interface == "segmented_dual_pass"
    risk_cap_row = (
        row.get("vnext_segmented_context_role") == "risk_cap"
        or row.get("vnext_context_risk_cap_active") is True
    )
    if not (segmented_interface and risk_cap_row):
        return
    missing_or_null = [
        key
        for key in _STAGE125_RISK_CAP_REQUIRED_EXPORT_FIELDS
        if key not in row or row.get(key) is None
    ]
    if missing_or_null:
        raise RuntimeError(
            "Stage126 risk-cap export missing/null fields: "
            + ", ".join(missing_or_null)
        )

def _stage125_output_value(
    output: dict[str, Any],
    key: str,
    index: int,
    *,
    global_scalar: bool = False,
) -> Any:
    value = output.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor_value = value.detach().cpu()
        if tensor_value.ndim == 0:
            if global_scalar or index == 0:
                item = tensor_value.item()
                return round(float(item), 6) if isinstance(item, (int, float)) else item
            return None
        if index >= int(tensor_value.shape[0]):
            return None
        row_value = tensor_value[index]
        if row_value.numel() == 1:
            return round(float(row_value.item()), 6)
        return [round(float(item), 6) for item in row_value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) > index:
            row_value = value[index]
            if torch.is_tensor(row_value):
                row_value = row_value.detach().cpu()
                if row_value.numel() == 1:
                    return round(float(row_value.item()), 6)
                return [round(float(item), 6) for item in row_value.reshape(-1).tolist()]
            return row_value
        return None
    if isinstance(value, (str, int, float, bool)):
        if global_scalar or not isinstance(value, bool):
            return value
    return None


def _stage125_prediction_label(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return v5.ID_TO_FINAL_LABEL.get(int(value), int(value))
    except (TypeError, ValueError):
        return value


def _stage125_merge_risk_cap_exports(
    item: dict[str, Any],
    output: dict[str, Any],
    index: int,
    *,
    args: argparse.Namespace | None,
) -> None:
    scalar_keys = (
        "vnext_context_risk",
        "vnext_context_risk_excess",
        "vnext_context_cap_factor",
    )
    for key in scalar_keys:
        value = _stage125_output_value(output, key, index)
        if value is not None:
            item[key] = value
    for key in (
        "vnext_context_risk_threshold",
        "vnext_context_risk_cap_alpha",
    ):
        value = _stage125_output_value(output, key, index, global_scalar=True)
        if value is not None:
            item[key] = value
    value = _stage125_output_value(
        output, "vnext_context_risk_cap_active", index, global_scalar=True
    )
    if value is not None:
        item["vnext_context_risk_cap_active"] = bool(value)
    value = _stage125_output_value(output, "vnext_context_risk_source", index, global_scalar=True)
    if value is not None:
        item["vnext_context_risk_source"] = value
    value = _stage125_output_value(output, "vnext_context_cap_notes", index, global_scalar=True)
    if value is not None:
        item["vnext_context_cap_notes"] = value
    value = _stage125_output_value(output, "vnext_context_cap_applied", index)
    if value is not None:
        item["vnext_context_cap_applied"] = bool(value)
    for key in (
        "vnext_logits_before_context_cap",
        "vnext_logits_after_context_cap",
        "vnext_context_only_logits",
    ):
        value = _stage125_output_value(output, key, index)
        if value is not None:
            item[key] = value
    for key in (
        "vnext_prediction_before_context_cap",
        "vnext_prediction_after_context_cap",
        "vnext_context_only_prediction",
    ):
        value = _stage125_output_value(output, key, index)
        if value is not None:
            item[key] = _stage125_prediction_label(value)
    role = item.get("vnext_segmented_context_role")
    risk_source = item.get(
        "vnext_context_risk_source",
        getattr(args, "vnext_context_risk_source", "context_not_entitled_prob")
        if args is not None else "context_not_entitled_prob",
    )
    alpha = item.get(
        "vnext_context_risk_cap_alpha",
        getattr(args, "vnext_context_risk_cap_alpha", 0.0) if args is not None else 0.0,
    )
    threshold = item.get(
        "vnext_context_risk_threshold",
        getattr(args, "vnext_context_risk_threshold", 0.5) if args is not None else 0.5,
    )
    try:
        alpha_float = float(alpha)
    except (TypeError, ValueError):
        alpha_float = 0.0
    try:
        threshold_float = float(threshold)
    except (TypeError, ValueError):
        threshold_float = 0.5
    if role == "risk_cap":
        item.setdefault("vnext_context_risk_cap_alpha", round(alpha_float, 6))
        item.setdefault("vnext_context_risk_threshold", round(threshold_float, 6))
        item.setdefault("vnext_context_risk_source", risk_source)
        final_logits = item.get("final_logits")
        if final_logits is not None:
            item.setdefault("vnext_logits_before_context_cap", final_logits)
            item.setdefault("vnext_logits_after_context_cap", final_logits)
        item.setdefault(
            "vnext_prediction_before_context_cap", item.get("pred_final_label", item.get("pred_label"))
        )
        item.setdefault(
            "vnext_prediction_after_context_cap", item.get("pred_final_label", item.get("pred_label"))
        )
        if alpha_float == 0.0:
            item.setdefault("vnext_context_risk", 0.0)
            item.setdefault("vnext_context_risk_excess", 0.0)
            item.setdefault("vnext_context_cap_factor", 1.0)
            item.setdefault("vnext_context_cap_applied", False)
            item.setdefault("vnext_context_risk_cap_active", False)
            item.setdefault(
                "vnext_context_cap_notes",
                f"risk_source={risk_source};alpha_zero_noop;export_noop_fallback",
            )
        item.setdefault("vnext_context_only_logits", None)
        item.setdefault("vnext_context_only_prediction", None)

def prediction_records_v6b(
    records: list[dict],
    output: dict[str, Any],
    *,
    stage32_owner_state_export: bool = False,
    stage32_owner_state_shadow_mode: bool = False,
    stage32_coverage_owner_v2: bool = False,
    stage32_coverage_owner_v2_min_confidence: float = 0.50,
    stage32_coverage_owner_v2_min_margin: float = 0.05,
    stage32_coverage_owner_v2_allow_abstain: bool = False,
    stage33_structured_coverage_owner: bool = False,
    stage33_structured_coverage_owner_export: bool = False,
    stage33_structured_coverage_owner_shadow_mode: bool = False,
    stage33_structured_coverage_preserve_can_support: bool = False,
    stage33_structured_coverage_direct_support_rules: str = _STAGE33_DEFAULT_DIRECT_SUPPORT_RULES,
    stage33_structured_coverage_disable_specific_general_direct_support: bool = False,
    stage33_structured_coverage_weak_rules_to_residual: str = "",
    stage33_structured_coverage_conditional_fallback: bool = False,
    stage33_structured_coverage_fallback_source: str = "current_final",
    stage33_structured_coverage_enable_whole_part_rules: bool = False,
    stage33_structured_coverage_whole_part_direct_support: bool = False,
    stage33_structured_coverage_whole_part_lexicon: str = "",
    stage33_structured_coverage_whole_part_v2: bool = False,
    stage33_structured_coverage_whole_part_v2_use_expanded_lexicon: bool = False,
    stage33_structured_coverage_whole_part_v2_direct_support_policy: str = "hard_core_required",
    stage33_whole_part_conditional_safe_overrides_hard_core: bool = False,
    stage36_support_safety_config: "dict[str, Any] | None" = None,
    stage37_safe_support_recovery_config: "dict[str, Any] | None" = None,
    args: "argparse.Namespace | None" = None,
) -> list[dict]:
    """Export predictions with Stage28-E enriched schema (additive; preserves all legacy fields)."""
    logits_cpu = output["logits"].detach().cpu()
    probabilities = torch.softmax(logits_cpu, dim=-1)
    predictions = output["predictions"].detach().cpu()
    stage32_shadow_logits_before = (
        logits_cpu.clone()
        if stage32_owner_state_shadow_mode and stage32_owner_state_export
        else None
    )
    stage32_shadow_predictions_before = (
        predictions.clone()
        if stage32_owner_state_shadow_mode and stage32_owner_state_export
        else None
    )
    structured_direct_support_rules = _stage33_parse_csv_set(
        stage33_structured_coverage_direct_support_rules
    )
    if stage33_structured_coverage_disable_specific_general_direct_support:
        structured_direct_support_rules.discard("specific_to_general_proxy")
    structured_weak_rules_to_residual = _stage33_parse_csv_set(
        stage33_structured_coverage_weak_rules_to_residual
    )

    # Existing v6b scalar outputs
    scalar_keys = (
        "frame_prob",
        "predicate_coverage_prob",
        "sufficiency_prob",
        "entitlement_prob",
        "polarity_margin",
        "boundary_prob",        # Stage22-A:  None when boundary head is disabled
        "frame_violation_prob", # Stage22-A3: None when frame violation head is disabled
    )
    exported: list[dict] = []
    for index, record in enumerate(records):
        stage32_owner_state = (
            build_stage32_owner_state(
                record,
                output,
                index,
                coverage_owner_v2_enabled=stage32_coverage_owner_v2,
                coverage_owner_v2_min_confidence=stage32_coverage_owner_v2_min_confidence,
                coverage_owner_v2_min_margin=stage32_coverage_owner_v2_min_margin,
                coverage_owner_v2_allow_abstain=stage32_coverage_owner_v2_allow_abstain,
                structured_coverage_enabled=(
                    stage33_structured_coverage_owner
                    and stage33_structured_coverage_owner_export
                ),
                structured_coverage_shadow_mode=(
                    stage33_structured_coverage_owner
                    and stage33_structured_coverage_owner_shadow_mode
                ),
                structured_coverage_preserve_can_support=(
                    stage33_structured_coverage_preserve_can_support
                ),
                structured_coverage_direct_support_rules=structured_direct_support_rules,
                structured_coverage_weak_rules_to_residual=structured_weak_rules_to_residual,
                structured_coverage_conditional_fallback=(
                    stage33_structured_coverage_conditional_fallback
                ),
                structured_coverage_fallback_source=(
                    stage33_structured_coverage_fallback_source
                ),
                structured_coverage_whole_part_enabled=(
                    stage33_structured_coverage_enable_whole_part_rules
                ),
                structured_coverage_whole_part_direct_support=(
                    stage33_structured_coverage_whole_part_direct_support
                ),
                structured_coverage_whole_part_lexicon=(
                    stage33_structured_coverage_whole_part_lexicon
                ),
                structured_coverage_whole_part_v2=(
                    stage33_structured_coverage_whole_part_v2
                ),
                structured_coverage_whole_part_v2_expanded_lexicon=(
                    stage33_structured_coverage_whole_part_v2_use_expanded_lexicon
                ),
                structured_coverage_whole_part_v2_direct_support_policy=(
                    stage33_structured_coverage_whole_part_v2_direct_support_policy
                ),
                structured_whole_part_conditional_safe_overrides_hard_core=(
                    stage33_whole_part_conditional_safe_overrides_hard_core
                ),
            )
            if stage32_owner_state_export
            else None
        )
        pred_id = int(predictions[index])
        pred_label = v5.ID_TO_FINAL_LABEL[pred_id]

        # Gold label
        gold_raw = record.get("final_label")
        gold_label = _s28e_normalize_label(gold_raw)
        gold_label_id = _s28e_label_to_id(gold_label)

        # Stable identity
        stable_id = str(record.get("id", index))
        source_id_raw = (
            record.get("pair_id")
            or record.get("source_id")
            or record.get("original_id")
        )
        if source_id_raw is None:
            meta = record.get("metadata") or {}
            source_id_raw = (
                meta.get("pair_id")
                or meta.get("source_id")
                or meta.get("original_id")
            )
        source_id = (
            str(source_id_raw) if source_id_raw is not None
            else _s28e_derive_source_id(stable_id)
        )
        pair_id_val = record.get("pair_id")
        pair_id = str(pair_id_val) if pair_id_val is not None else source_id

        # Intervention
        norm_intervention = _s28e_normalize_intervention(record)
        diagnostic_axis = (
            _S28E_DIAGNOSTIC_AXIS.get(norm_intervention) if norm_intervention else None
        )

        # Correctness / false-support flags
        is_correct: "bool | None" = (
            (pred_id == gold_label_id) if gold_label_id is not None else None
        )
        is_false_support: "bool | None" = (
            (gold_label != "SUPPORT" and pred_label == "SUPPORT")
            if gold_label is not None else None
        )
        is_location_false_support = (
            norm_intervention == "location_swap" and pred_label == "SUPPORT"
        )
        is_role_false_support = (
            norm_intervention == "role_swap" and pred_label == "SUPPORT"
        )

        # Per-class logits and probabilities
        logits_row = logits_cpu[index]
        probs_row = probabilities[index]
        final_logits = _s28e_safe_list_float(logits_row)
        final_probs_list = _s28e_safe_list_float(probs_row)
        refute_logit = _s28e_safe_float(logits_row[0])
        ne_logit = _s28e_safe_float(logits_row[1])
        support_logit = _s28e_safe_float(logits_row[2])
        refute_prob = _s28e_safe_float(probs_row[0])
        ne_prob = _s28e_safe_float(probs_row[1])
        support_prob = _s28e_safe_float(probs_row[2])

        item: dict[str, Any] = {
            #  Stable identity 
            "stable_id": stable_id,
            "source_id": source_id,
            "pair_id": pair_id,
            "example_index": index,
            #  Text 
            "claim": record.get("claim"),
            "evidence": record.get("evidence"),
            #  Intervention 
            "intervention": record.get("intervention_type"),
            "normalized_intervention": norm_intervention,
            "diagnostic_axis": diagnostic_axis,
            #  Gold labels 
            "gold_label_raw": gold_raw,
            "gold_label": gold_label,
            "gold_label_id": gold_label_id,
            #  Predicted labels 
            "pred_label_id": pred_id,
            "pred_label": pred_label,
            "pred_label_raw": pred_label,
            "is_correct": is_correct,
            "is_false_support": is_false_support,
            "is_location_false_support": is_location_false_support,
            "is_role_false_support": is_role_false_support,
            #  Final logits / probs (order: REFUTE=0, NOT_ENTITLED=1, SUPPORT=2) 
            "final_logits": final_logits,
            "final_probs": final_probs_list,
            "refute_logit": refute_logit,
            "ne_logit": ne_logit,
            "support_logit": support_logit,
            "refute_prob": refute_prob,
            "ne_prob": ne_prob,
            "support_prob": support_prob,
            #  Existing v6b diagnostic scalars 
            **{
                key: scalar_value
                for key in scalar_keys
                for scalar_value in (_stage113_scalar_value(output, key, index),)
                if scalar_value is not None
            },
            #  Gold auxiliary labels (when present in source record) 
            **{key: record[key] for key in _S28E_AUX_LABEL_KEYS if key in record},
            #  V7/H1 diagnostic scalars (absent on v6b_minimal runs) 
            **{
                key: scalar_value
                for key in _S28E_V7_SCALAR_KEYS
                for scalar_value in (_stage113_scalar_value(output, key, index),)
                if scalar_value is not None
            },
            **(
                {
                    "coverage_entailment_pred_id": int(
                        output["coverage_entailment_pred_id"].detach().cpu()[index]
                    ),
                    "coverage_entailment_pred_label": output[
                        "coverage_entailment_pred_label"
                    ][index],
                    "coverage_entailment_input_mode": output.get(
                        "coverage_entailment_input_mode"
                    ),
                }
                if output.get("coverage_entailment_pred_id") is not None
                and output.get("coverage_entailment_pred_label") is not None
                else {}
            ),
            #  Legacy backward-compat fields 
            **(
                flatten_stage32_owner_state(stage32_owner_state)
                if stage32_owner_state is not None
                else {}
            ),
            "id": record.get("id"),
            "intervention_type": record.get("intervention_type"),
            "gold_final_label": gold_raw,
            "pred_final_label": pred_label,
            #  Shallow raw record snapshot 
            "raw_record": {k: record[k] for k in _S28E_RAW_RECORD_KEYS if k in record},
        }
        for metadata_key in _S28E_PRESERVED_METADATA_KEYS:
            if metadata_key in record:
                item[metadata_key] = record[metadata_key]
        _stage113_add_vnext_scalars(item, output, index)
        output_segmented_active = bool(output.get("vnext_segmented_dual_pass_active", False))
        record_segmented_active = bool(record.get("vnext_segmented_dual_pass_active", False))
        vnext_segmented_active = record_segmented_active or output_segmented_active
        if vnext_segmented_active:
            item["vnext_segmented_dual_pass_active"] = True
            item.setdefault(
                "vnext_segmented_context_role",
                getattr(args, "vnext_segmented_context_role", record.get("vnext_segmented_context_role", "diagnostic_only"))
                if args is not None
                else record.get("vnext_segmented_context_role", "diagnostic_only"),
            )
            primary_source = output.get("vnext_primary_rep_source")
            if primary_source is not None:
                item["vnext_primary_rep_source"] = primary_source
        for key in (
            "vnext_segmented_dual_pass_active",
            "vnext_segmented_context_role",
            "vnext_primary_rep_source",
            "vnext_core_text_for_encoding",
            "vnext_context_text_for_encoding",
            "vnext_context_empty",
        ):
            if key in record:
                item[key] = record[key]
        for key in (
            "vnext_core_rep_norm",
            "vnext_context_rep_norm",
            "vnext_core_context_cosine",
        ):
            scalar_value = _stage113_scalar_value(output, key, index)
            if scalar_value is not None:
                item[key] = scalar_value

        if vnext_segmented_active:
            _stage125_merge_risk_cap_exports(item, output, index, args=args)


        #  Stage36-A: conservative support-safety blockers (shadow-only) 
        _stage37_stage36_info: dict[str, Any] = {}
        if (
            stage36_support_safety_config is not None
            and stage36_support_safety_config.get("enabled")
            and stage32_owner_state is not None
        ):
            _stage36_composer = stage32_owner_state["composer_shadow"]
            _stage36_structured = stage32_owner_state["structured_coverage"]
            _stage36_result = compute_stage36_support_safety_blocker(
                claim=str(record.get("claim") or ""),
                evidence=str(record.get("evidence") or ""),
                proposed_shadow_label=_stage36_composer["shadow_label"],
                proposed_reason=_stage36_composer["shadow_reason"],
                proposed_route=_stage36_structured.get("route", "RESIDUAL"),
                enabled=True,
                block_exception_scope=stage36_support_safety_config.get(
                    "block_exception_scope", False
                ),
                block_not_all_existential=stage36_support_safety_config.get(
                    "block_not_all_existential", False
                ),
                block_location_scope_mismatch=stage36_support_safety_config.get(
                    "block_location_scope_mismatch", False
                ),
                block_temporal_scope_mismatch=stage36_support_safety_config.get(
                    "block_temporal_scope_mismatch", False
                ),
                blocker_action=stage36_support_safety_config.get(
                    "blocker_action", "fallback_current_final"
                ),
                current_final_label=pred_label,
            )
            _stage37_stage36_info = _stage36_result
            if stage36_support_safety_config.get("export"):
                item["stage36_original_shadow_label"] = _stage36_result[
                    "stage36_support_blocker_original_shadow_label"
                ]
                item["stage36_final_shadow_label"] = _stage36_result[
                    "stage36_support_blocker_final_shadow_label"
                ]
                item["stage36_support_blocker_fired"] = _stage36_result[
                    "stage36_support_blocker_fired"
                ]
                item["stage36_support_blocker_reasons"] = _stage36_result[
                    "stage36_support_blocker_reasons"
                ]
                item["stage36_support_blocker_action"] = _stage36_result[
                    "stage36_support_blocker_action"
                ]
                item["stage36_exception_blocker_fired"] = _stage36_result[
                    "stage36_exception_blocker_fired"
                ]
                item["stage36_not_all_blocker_fired"] = _stage36_result[
                    "stage36_not_all_blocker_fired"
                ]
                item["stage36_location_scope_blocker_fired"] = _stage36_result[
                    "stage36_location_scope_blocker_fired"
                ]
                item["stage36_temporal_scope_blocker_fired"] = _stage36_result[
                    "stage36_temporal_scope_blocker_fired"
                ]
                item["stage36_scope_claim_values"] = _stage36_result[
                    "stage36_scope_claim_values"
                ]
                item["stage36_scope_evidence_values"] = _stage36_result[
                    "stage36_scope_evidence_values"
                ]
            if (
                stage36_support_safety_config.get("shadow_mode")
                and _stage36_result["stage36_support_blocker_fired"]
            ):
                _stage36_final_label = _stage36_result[
                    "stage36_support_blocker_final_shadow_label"
                ]
                item["stage32_shadow_label"] = _stage36_final_label
                item["stage32_shadow_reason"] = (
                    "stage36_support_safety_blocked:"
                    + ",".join(_stage36_result["stage36_support_blocker_reasons"])
                )
                if item.get("stage33_conditional_shadow_label") is not None:
                    item["stage33_conditional_shadow_label"] = _stage36_final_label

        #  Stage37-A: conservative safe SUPPORT recovery (shadow-only) 
        # Runs strictly after Stage36's post-blocker shadow label is known and
        # never overrides a fired Stage36 blocker.
        _stage39_stage37_info: dict[str, Any] = {}
        if (
            stage37_safe_support_recovery_config is not None
            and stage37_safe_support_recovery_config.get("enabled")
            and stage32_owner_state is not None
            and args is not None
        ):
            if _stage37_stage36_info:
                _stage37_post_stage36_label = _stage37_stage36_info[
                    "stage36_support_blocker_final_shadow_label"
                ]
            else:
                _stage37_post_stage36_label = stage32_owner_state["composer_shadow"][
                    "shadow_label"
                ]
            _stage37_result = compute_stage37_safe_support_recovery(
                claim=str(record.get("claim") or ""),
                evidence=str(record.get("evidence") or ""),
                current_shadow_label=_stage37_post_stage36_label,
                args=args,
                stage36_info=_stage37_stage36_info,
            )
            _stage39_stage37_info = _stage37_result
            if stage37_safe_support_recovery_config.get("export"):
                item["stage37_original_shadow_label"] = _stage37_result[
                    "stage37_original_shadow_label"
                ]
                item["stage37_final_shadow_label"] = _stage37_result[
                    "stage37_final_shadow_label"
                ]
                item["stage37_safe_recovery_fired"] = _stage37_result[
                    "stage37_safe_recovery_fired"
                ]
                item["stage37_safe_recovery_reasons"] = _stage37_result[
                    "stage37_safe_recovery_reasons"
                ]
                item["stage37_safe_recovery_action"] = _stage37_result[
                    "stage37_safe_recovery_action"
                ]
                item["stage37_no_except_included_subset_fired"] = _stage37_result[
                    "stage37_no_except_included_subset_fired"
                ]
                item["stage37_coordination_universal_subset_fired"] = _stage37_result[
                    "stage37_coordination_universal_subset_fired"
                ]
                item["stage37_numeric_universal_subset_fired"] = _stage37_result[
                    "stage37_numeric_universal_subset_fired"
                ]
                item["stage37_recovered_from_label"] = _stage37_result[
                    "stage37_recovered_from_label"
                ]
                item["stage37_recovered_to_label"] = _stage37_result[
                    "stage37_recovered_to_label"
                ]
                item["stage37_blocked_by_stage36"] = _stage37_result[
                    "stage37_blocked_by_stage36"
                ]
                item["stage37_blocked_by_scope_hazard"] = _stage37_result[
                    "stage37_blocked_by_scope_hazard"
                ]
                item["stage37_blocked_by_exception_hazard"] = _stage37_result[
                    "stage37_blocked_by_exception_hazard"
                ]
                item["stage37_blocked_by_not_all_hazard"] = _stage37_result[
                    "stage37_blocked_by_not_all_hazard"
                ]
            if (
                stage37_safe_support_recovery_config.get("shadow_mode")
                and _stage37_result["stage37_safe_recovery_fired"]
            ):
                _stage37_final_label = _stage37_result["stage37_final_shadow_label"]
                item["stage32_shadow_label"] = _stage37_final_label
                item["stage32_shadow_reason"] = (
                    "stage37_safe_support_recovery:"
                    + ",".join(_stage37_result["stage37_safe_recovery_reasons"])
                )
                if item.get("stage33_conditional_shadow_label") is not None:
                    item["stage33_conditional_shadow_label"] = _stage37_final_label

        #  Stage39-A: opt-in final composer (prediction/export-time only) 
        # Runs strictly after Stage37's final shadow label is known. Off by
        # default: no stage39_* fields are added and pred_final_label is
        # untouched unless --stage39-use-final-composer-opt-in (and, to
        # replace the exported final label, --stage39-final-composer-output-
        # mode replace_pred_final_label) are explicitly set. Never mutates
        # logits/predictions tensors -- export dict only.
        _stage39_opt_in_enabled = bool(
            getattr(args, "stage39_use_final_composer_opt_in", False)
        ) if args is not None else False
        _stage39_export_flag = bool(
            getattr(args, "stage39_final_composer_export", False)
        ) if args is not None else False
        if _stage39_opt_in_enabled or _stage39_export_flag:
            _stage39_row: dict[str, Any] = {
                "pred_final_label": item.get("pred_final_label"),
                "stage37_final_shadow_label": _stage39_stage37_info.get(
                    "stage37_final_shadow_label"
                ),
                "stage36_final_shadow_label": _stage37_stage36_info.get(
                    "stage36_support_blocker_final_shadow_label"
                ),
                "stage32_shadow_label": item.get("stage32_shadow_label"),
                "stage36_support_blocker_fired": _stage37_stage36_info.get(
                    "stage36_support_blocker_fired", False
                ),
                "stage37_recovered_from_label": _stage39_stage37_info.get(
                    "stage37_recovered_from_label"
                ),
                "stage33_structured_coverage_reason": item.get(
                    "stage33_structured_coverage_reason"
                ),
                "stage33_structured_coverage_route": item.get(
                    "stage33_structured_coverage_route"
                ),
                "stage33_structured_coverage_label": item.get(
                    "stage33_structured_coverage_label"
                ),
                "stage33_conditional_override_type": item.get(
                    "stage33_conditional_override_type"
                ),
                "stage36_conditional_override_type": item.get(
                    "stage36_conditional_override_type"
                ),
                "stage37_conditional_override_type": item.get(
                    "stage37_conditional_override_type"
                ),
                "gold_final_label": item.get("gold_final_label"),
            }
            _stage39_result = compute_stage39_final_composer(_stage39_row, args)
            item["stage39_final_composer_enabled"] = _stage39_result[
                "stage39_final_composer_enabled"
            ]
            item["stage39_final_composer_policy"] = _stage39_result[
                "stage39_final_composer_policy"
            ]
            item["stage39_final_composer_output_mode"] = _stage39_result[
                "stage39_final_composer_output_mode"
            ]
            item["stage39_original_final_label"] = _stage39_result[
                "stage39_original_final_label"
            ]
            item["stage39_original_pred_final_label"] = item["pred_final_label"]
            item["stage39_source_shadow_label"] = _stage39_result[
                "stage39_source_shadow_label"
            ]
            item["stage39_composed_final_label"] = _stage39_result[
                "stage39_composed_final_label"
            ]
            item["stage39_final_label_changed"] = _stage39_result[
                "stage39_final_label_changed"
            ]
            item["stage39_composer_action"] = _stage39_result["stage39_composer_action"]
            item["stage39_composer_reason"] = _stage39_result["stage39_composer_reason"]
            item["stage39_blocked_by_stage36"] = _stage39_result[
                "stage39_blocked_by_stage36"
            ]
            item["stage39_blocked_by_refute_to_support_guard"] = _stage39_result[
                "stage39_blocked_by_refute_to_support_guard"
            ]
            item["stage39_blocked_by_stage37_from_refute_guard"] = _stage39_result[
                "stage39_blocked_by_stage37_from_refute_guard"
            ]
            item["stage39_blocked_by_missing_source"] = _stage39_result[
                "stage39_blocked_by_missing_source"
            ]
            item["stage39_refute_trigger"] = _stage39_result["stage39_refute_trigger"]
            item["stage39_refute_trigger_reason"] = _stage39_result[
                "stage39_refute_trigger_reason"
            ]
            item["stage39_refute_trigger_route"] = _stage39_result[
                "stage39_refute_trigger_route"
            ]
            item["stage39_refute_trigger_override"] = _stage39_result[
                "stage39_refute_trigger_override"
            ]
            if (
                _stage39_opt_in_enabled
                and _stage39_result["stage39_final_composer_output_mode"]
                == "replace_pred_final_label"
            ):
                item["pred_final_label"] = _stage39_result[
                    "stage39_composed_final_label"
                ]

        _stage128_guard_exports = _stage128_location_slot_guard_exports(
            record,
            prediction_before_guard=item.get("pred_final_label", pred_label),
            args=args,
        )
        if _stage128_guard_exports:
            item.update(_stage128_guard_exports)
            item["base_prediction"] = _stage128_guard_exports[
                "stage128_prediction_before_location_guard"
            ]
            item["prediction"] = _stage128_guard_exports[
                "stage128_prediction_after_location_guard"
            ]
            item["pred_final_label"] = _stage128_guard_exports[
                "stage128_prediction_after_location_guard"
            ]

        _stage125_assert_risk_cap_exports(item)
        exported.append(item)
    if stage32_shadow_logits_before is not None and not torch.equal(
        stage32_shadow_logits_before, output["logits"].detach().cpu()
    ):
        raise RuntimeError("Stage32-A owner-state export modified final logits.")
    if stage32_shadow_predictions_before is not None and not torch.equal(
        stage32_shadow_predictions_before, output["predictions"].detach().cpu()
    ):
        raise RuntimeError("Stage32-A owner-state export modified final predictions.")
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


# ---------------------------------------------------------------------------
# Stage27-H2A helpers
# ---------------------------------------------------------------------------

#: Human-readable formula descriptions for each H1 entitlement decision signal.
_V7_H1_DECISION_SIGNAL_SOURCE: dict[str, str] = {
    "learned":               "v7_entitlement_prob (EntitlementGate)",
    "product":               "frame_prob * predicate_coverage_prob * sufficiency_prob",
    "min":                   "min(frame_prob, predicate_coverage_prob, sufficiency_prob)",
    "frame_predicate_product": "frame_prob * predicate_coverage_prob",
    "frame_predicate_min":   "min(frame_prob, predicate_coverage_prob)",
    "product_learned_residual":
        "product_base + beta * (v7_entitlement_prob - product_base.detach())",
}


def _resolve_v7_final_logit_composition(args: "argparse.Namespace") -> "str | None":
    """Return the actual v7 final-logit composition mode string for reporting.

    This fixes the Stage26 reporting bug where H1 runs incorrectly reported
    'hierarchical_additive' even though v6b_style_softplus_multiplicative was active.
    """
    if args.architecture != "v7_hierarchical":
        return None
    if getattr(args, "v7_use_v6b_style_final_decision", False):
        return "v6b_style_softplus_multiplicative"
    if getattr(args, "v7_no_entitlement_polarity_conditioning", False):
        return "flat"
    return "hierarchical_additive"


# ---------------------------------------------------------------------------
# Stage48: load the Stage47-validated frozen recovery config
# ---------------------------------------------------------------------------

STAGE47_DECISION_READY = "STAGE47_SELECTED_RECOVERY_CONFIG_READY"
STAGE47_SELECTED_CONFIG_NAME = "recovery_w01_ne01"
STAGE47_SELECTED_SUPPORT_W = 0.1
STAGE47_SELECTED_NE_W = 0.1


def load_stage47_selected_recovery_weights(path: Path) -> tuple[float, float]:
    """Read and validate the Stage47 selected-recovery-config-check JSON.

    Returns (support_w, ne_w) for the frozen recovery_w01_ne01 selection.
    Raises ValueError with a clear message if the file is missing, malformed,
    or does not match the frozen Stage47 selection. Never falls back silently.
    """
    if not path.exists():
        raise ValueError(
            f"[stage48] --use-stage47-selected-recovery-config: Stage47 config file "
            f"not found at {path}. Run scripts/write_stage47_selected_recovery_config_check.py "
            "first, or pass --stage47-recovery-config-path to point at it."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"[stage48] --use-stage47-selected-recovery-config: failed to parse "
            f"Stage47 config file at {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"[stage48] --use-stage47-selected-recovery-config: Stage47 config file "
            f"at {path} did not contain a JSON object."
        )

    if payload.get("decision") != STAGE47_DECISION_READY:
        raise ValueError(
            "[stage48] --use-stage47-selected-recovery-config: Stage47 decision is "
            f"{payload.get('decision')!r}, not {STAGE47_DECISION_READY!r}. "
            f"Refusing to guess recovery weights from {path}."
        )
    if payload.get("selected_config_name") != STAGE47_SELECTED_CONFIG_NAME:
        raise ValueError(
            "[stage48] --use-stage47-selected-recovery-config: Stage47 "
            f"selected_config_name is {payload.get('selected_config_name')!r}, not "
            f"{STAGE47_SELECTED_CONFIG_NAME!r}."
        )
    if payload.get("selected_support_w") != STAGE47_SELECTED_SUPPORT_W:
        raise ValueError(
            "[stage48] --use-stage47-selected-recovery-config: Stage47 "
            f"selected_support_w is {payload.get('selected_support_w')!r}, not "
            f"{STAGE47_SELECTED_SUPPORT_W!r}."
        )
    if payload.get("selected_ne_w") != STAGE47_SELECTED_NE_W:
        raise ValueError(
            "[stage48] --use-stage47-selected-recovery-config: Stage47 "
            f"selected_ne_w is {payload.get('selected_ne_w')!r}, not "
            f"{STAGE47_SELECTED_NE_W!r}."
        )

    return STAGE47_SELECTED_SUPPORT_W, STAGE47_SELECTED_NE_W


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
        "--vnext-evidence-interface",
        choices=VNEXT_EVIDENCE_INTERFACE_CHOICES,
        default="full_evidence",
        help=(
            "Stage123-A experimental vNext evidence interface. "
            "Default full_evidence preserves existing claim+evidence encoding."
        ),
    )
    parser.add_argument(
        "--vnext-enable-segmented-dual-pass",
        action="store_true",
        default=False,
        help=(
            "Stage124-A: enable segmented dual-pass vNext encoding when "
            "--vnext-evidence-interface segmented_dual_pass is selected."
        ),
    )
    parser.add_argument(
        "--vnext-segmented-context-role",
        choices=("ignore", "diagnostic_only", "risk_cap"),
        default="diagnostic_only",
        help=(
            "Stage125-A: context role for segmented dual-pass. ignore and diagnostic_only "
            "leave logits unchanged; risk_cap may symmetrically cap REFUTE/SUPPORT."
        ),
    )
    parser.add_argument(
        "--vnext-context-risk-cap-alpha",
        type=float,
        default=0.0,
        help=(
            "Stage125-A: strength of segmented context entitlement cap. "
            "Default 0.0 is an exact no-op."
        ),
    )
    parser.add_argument(
        "--vnext-context-risk-threshold",
        type=float,
        default=0.5,
        help=(
            "Stage125-A: context risk threshold below which no cap is applied. "
            "Default: 0.5."
        ),
    )
    parser.add_argument(
        "--vnext-context-risk-source",
        choices=("context_not_entitled_prob", "context_uncertainty"),
        default="context_not_entitled_prob",
        help=(
            "Stage125-A: source for segmented context risk. "
            "context_not_entitled_prob uses the context-only NOT_ENTITLED probability; "
            "context_uncertainty uses 1 - max(context-only probability)."
        ),
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

    #  Temporal residual adapter 
    # A 2-layer MLP adapter that absorbs temporal diagnostic supervision without propagating
    # gradients into the shared frame_pair_repr / FrameGate representation.
    # Architecture: Linear(frame_size, frame_size//2) ??GELU ??Linear(frame_size//2, 1)
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

    #  TemporalChannel V1 (v6C Lean) 
    # Default off. Independent temporal channel reading from cat([claim_frame_state,
    # evidence_frame_state]) ??NOT frame_pair_repr ??to avoid Stage23 gradient coupling.
    # Stage15 OOD is eval-only and is NEVER used for TC loss, calibration, or penalty selection.
    # Gated penalty requires --use-preservation-entitlement-loss (raises clear error otherwise).
    # Cannot be stacked with --use-temporal-adapter-final-penalty (raises clear error).
    parser.add_argument(
        "--use-temporal-channel",
        action="store_true",
        default=False,
        help=(
            "Enable TemporalChannel V1. Default off. Adds a 2-layer MLP reading "
            "cat([claim_frame_state, evidence_frame_state]) ??pre-pair-projector slot states "
            "from FrameGate, NOT frame_pair_repr ??to produce temporal_channel_logit and "
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

    #  Stage26-A: v7 Hierarchical Entitlement architecture 
    # Default: architecture=v6b_minimal (full backward compatibility; no v6B behavior changes).
    # Set --architecture v7_hierarchical to use ContraMambaV7Hierarchical instead.
    # v7 flags below are only consulted when architecture==v7_hierarchical; they have no
    # effect on v6B runs and do not change any default v6B behavior.
    parser.add_argument(
        "--architecture",
        choices=("v6b_minimal", "v7_hierarchical", "vnext_minimal"),
        default="v6b_minimal",
        help=(
            "Model architecture to use. Default: v6b_minimal (existing v6B behavior, "
            "fully backward compatible). v7_hierarchical: Stage26-A ContraMambaV7Hierarchical "
            "(new hierarchical entitlement pipeline; requires explicit selection). "
            "vnext_minimal: Stage109 entitlement-first minimal vNext surface."
        ),
    )
    parser.add_argument(
        "--vnext-router-mode",
        choices=(
            "learned_only",
            "product",
            "min",
            "learned_x_product",
            "learned_x_sufficiency",
            "sufficiency_only",
            "learned_x_frame_sufficiency",
            "learned_x_predicate_sufficiency",
        ),
        default="learned_x_product",
        help=(
            "vNext minimal entitlement composition. Default learned_x_product: "
            "learned entitlement probability gated by frame*predicate*sufficiency. "
            "Ablations include learned_x_sufficiency, sufficiency_only, learned_only, "
            "learned_x_frame_sufficiency, and learned_x_predicate_sufficiency. "
            "Only used when --architecture vnext_minimal."
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
        "--v7-use-v6b-style-final-decision",
        action="store_true",
        default=False,
        help=(
            "Stage26-H1: Use v6B-style softplus polarity energy plus multiplicative "
            "entitlement_prob final decision inside v7. Default off."
        ),
    )

    parser.add_argument(
        "--v7-use-learnable-ne-alpha",
        action="store_true",
        default=False,
        help=(
            "Stage26-H1: Use learnable alpha for NOT_ENTITLED residual "
            "ne_bias + alpha * (1 - entitlement_prob). Default off."
        ),
    )

    parser.add_argument(
        "--v7-ne-alpha-init",
        type=float,
        default=1.0,
        help="Stage26-H1: Initial value for learnable NE alpha. Default 1.0.",
    )
    parser.add_argument(
        "--v7-h1-entitlement-decision-signal",
        choices=(
            "learned", "product", "min",
            "frame_predicate_product", "frame_predicate_min",
            "product_learned_residual",
        ),
        default="learned",
        help=(
            "Stage27-H2A: Decision-time entitlement signal used in the H1 final-decision path "
            "(--v7-use-v6b-style-final-decision). Only consulted when H1 is active. "
            "Default 'learned' preserves existing H1 behavior (v7_entitlement_prob). "
            "'product' = frame_prob * predicate_coverage_prob * sufficiency_prob. "
            "'min' = min(frame_prob, predicate_coverage_prob, sufficiency_prob). "
            "'frame_predicate_product' = frame_prob * predicate_coverage_prob. "
            "'frame_predicate_min' = min(frame_prob, predicate_coverage_prob). "
            "'product_learned_residual' = product_base + beta * (learned - product_base.detach()), "
            "where beta is --v7-h1-hybrid-residual-beta (Stage27-H2E)."
        ),
    )
    parser.add_argument(
        "--v7-h1-entitlement-product-power",
        type=float,
        default=1.0,
        help=(
            "Stage27-H2B: Power exponent applied to the 'product' H1 entitlement signal. "
            "Consulted when --v7-h1-entitlement-decision-signal is 'product' or "
            "'product_learned_residual'. "
            "Default 1.0 preserves exact H2A product behavior. "
            "Values < 1.0 soften the product gate (less SUPPORT suppression). "
            "Values > 1.0 sharpen it."
        ),
    )
    parser.add_argument(
        "--v7-h1-hybrid-residual-beta",
        type=float,
        default=0.25,
        help=(
            "Stage27-H2E: residual strength for product_learned_residual decision signal. "
            "The base is the product entitlement signal after product_power; "
            "the residual is learned_entitlement_prob - detached_base. "
            "beta=0 recovers pure product. Suggested sweep: 0.1,0.2,0.3,0.5."
        ),
    )

    #  Stage28-I-A: independent location-boundary cap/head 
    # All off by default. Preserves all current Stage27/Stage28 behavior when disabled.
    # Only meaningful for --architecture v7_hierarchical.
    parser.add_argument(
        "--v7-use-location-boundary-head",
        action="store_true",
        default=False,
        help=(
            "Stage28-I-A: Enable independent location-boundary head for v7_hierarchical. "
            "Adds a small MLP over [frame_pair_repr, predicate_pair_repr, sufficiency_repr] "
            "producing location_boundary_prob in [0,1]. "
            "High prob = location-safe; low prob = potential location mismatch. "
            "Disabled by default; no effect on existing behavior when off."
        ),
    )
    parser.add_argument(
        "--v7-use-location-boundary-loss",
        action="store_true",
        default=False,
        help=(
            "Stage28-I-A: Enable auxiliary BCE training for the location boundary head. "
            "Target: 0 for location_swap, 1 for none/paraphrase/polarity_flip records. "
            "All other intervention types are excluded from this loss. "
            "Requires --v7-use-location-boundary-head. "
            "Stage15/OOD is not used for this loss. Default off."
        ),
    )
    parser.add_argument(
        "--v7-location-boundary-loss-weight",
        type=float,
        default=0.0,
        help="Stage28-I-A: Weight for location boundary BCE loss. Default 0.0.",
    )
    parser.add_argument(
        "--v7-location-boundary-cap-mode",
        choices=("none", "hard", "soft"),
        default="none",
        help=(
            "Stage28-I-A: Final-decision cap mode for the location boundary head. "
            "'none' (default): no cap applied; existing behavior is unchanged. "
            "'hard': entitlement_for_decision = min(entitlement, location_boundary_prob). "
            "'soft': entitlement_for_decision = entitlement * location_boundary_prob^gamma. "
            "Requires --v7-use-location-boundary-head when not 'none'."
        ),
    )
    parser.add_argument(
        "--v7-location-boundary-cap-gamma",
        type=float,
        default=1.0,
        help=(
            "Stage28-I-A: Gamma exponent for soft location boundary cap. "
            "Used only when --v7-location-boundary-cap-mode soft. "
            "Must be > 0. Default 1.0."
        ),
    )
    parser.add_argument(
        "--v7-location-boundary-cap-detach",
        action="store_true",
        default=False,
        help=(
            "Stage28-I-A: Detach location_boundary_prob before applying the cap. "
            "Isolates cap effect from CE gradients. Default off."
        ),
    )

    #  Stage30-C2: independent temporal-safety cap/head 
    # All off by default. Preserves all current Stage28-I behavior when disabled.
    # Only meaningful for --architecture v7_hierarchical with --v7-use-v6b-style-final-decision.
    parser.add_argument(
        "--v7-use-temporal-safety-head",
        action="store_true",
        default=False,
        help=(
            "Stage30-C2: Enable independent temporal-safety head for v7_hierarchical. "
            "Adds a small MLP over [frame_pair_repr, predicate_pair_repr, sufficiency_repr] "
            "producing temporal_safety_prob in [0,1]. "
            "High prob = temporally safe; low prob = temporal mismatch risk. "
            "Disabled by default; no effect on existing behavior when off."
        ),
    )
    parser.add_argument(
        "--v7-use-temporal-safety-loss",
        action="store_true",
        default=False,
        help=(
            "Stage30-C2: Enable auxiliary BCE training for the temporal safety head. "
            "Target: 0 for time_swap, 1 for none/paraphrase (from --v7-temporal-safety-data). "
            "Reads stage30_temporal_safe_label if present; falls back to intervention_type. "
            "Requires --v7-use-temporal-safety-head and --v7-temporal-safety-data. "
            "Stage15/OOD is not used for this loss. Default off."
        ),
    )
    parser.add_argument(
        "--v7-temporal-safety-loss-weight",
        type=float,
        default=0.0,
        help="Stage30-C2: Weight for temporal safety BCE loss. Default 0.0.",
    )
    parser.add_argument(
        "--v7-temporal-safety-loss-pos-weight",
        type=float,
        default=None,
        help=(
            "Stage30-C2: BCE pos_weight for temporal safety loss (positive = time_swap). "
            "If omitted, no pos_weight is applied. Default None."
        ),
    )
    parser.add_argument(
        "--v7-temporal-safety-data",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Stage30-C2: path to temporal safety diagnostic JSONL. "
            "Must contain none/paraphrase (temporal_safe=1) and time_swap (temporal_safe=0). "
            "Never added to main train/dev; never used for checkpoint selection. "
            "Required when --v7-use-temporal-safety-loss is enabled."
        ),
    )
    parser.add_argument(
        "--v7-temporal-safety-cap-mode",
        choices=("none", "hard", "soft"),
        default="none",
        help=(
            "Stage30-C2: Final-decision cap mode for the temporal safety head. "
            "'none' (default): no cap applied; existing behavior is unchanged. "
            "'hard': entitlement = min(entitlement, temporal_safety_prob). "
            "'soft': entitlement = entitlement * temporal_safety_prob^gamma. "
            "Applied after location-boundary cap when both are enabled. "
            "Requires --v7-use-temporal-safety-head when not 'none'."
        ),
    )
    parser.add_argument(
        "--v7-temporal-safety-cap-gamma",
        type=float,
        default=1.0,
        help=(
            "Stage30-C2: Gamma exponent for soft temporal safety cap. "
            "Used only when --v7-temporal-safety-cap-mode soft. "
            "Must be > 0. Default 1.0."
        ),
    )
    parser.add_argument(
        "--v7-temporal-safety-cap-detach",
        action="store_true",
        default=False,
        help=(
            "Stage30-C2: Detach temporal_safety_prob before applying the cap. "
            "Isolates cap effect from CE gradients. Default off."
        ),
    )

    #  Stage30-D: representation-decomposed temporal mismatch multihead 
    parser.add_argument(
        "--v7-use-temporal-mismatch-multihead",
        action="store_true",
        default=False,
        help=(
            "Stage30-D: Enable TemporalMismatchMultiHead for v7_hierarchical. "
            "Three independent heads: frame_head on frame_pair_repr, predicate_head on "
            "predicate_pair_repr, sufficiency_head on sufficiency_repr. "
            "Positive=1 means temporal mismatch (opposite of Stage30-C2 convention). "
            "Cannot combine Stage30-D cap with Stage30-C2 cap. Default off."
        ),
    )
    parser.add_argument(
        "--v7-use-temporal-mismatch-multihead-loss",
        action="store_true",
        default=False,
        help=(
            "Stage30-D: Enable auxiliary BCE training for the temporal mismatch multihead. "
            "Reads stage30_temporal_safe_label (0->mismatch=1, 1->safe=0) or "
            "falls back to intervention_type (time_swap??, none/paraphrase??). "
            "Requires --v7-use-temporal-mismatch-multihead and "
            "--v7-temporal-mismatch-multihead-data. Default off."
        ),
    )
    parser.add_argument(
        "--v7-temporal-mismatch-multihead-loss-weight",
        type=float,
        default=0.0,
        help="Stage30-D: Weight for temporal mismatch multihead BCE loss. Default 0.0.",
    )
    parser.add_argument(
        "--v7-temporal-mismatch-multihead-loss-pos-weight",
        type=float,
        default=None,
        help=(
            "Stage30-D: BCE pos_weight for temporal mismatch multihead loss "
            "(positive = time_swap / temporal mismatch). If omitted, no pos_weight. Default None."
        ),
    )
    parser.add_argument(
        "--v7-temporal-mismatch-multihead-data",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Stage30-D: Path to temporal mismatch multihead auxiliary JSONL. "
            "Same format as Stage30-C2 temporal safety data; labels are inverted. "
            "Never added to main train/dev; never used for checkpoint selection. "
            "Required when --v7-use-temporal-mismatch-multihead-loss is enabled."
        ),
    )
    parser.add_argument(
        "--v7-temporal-mismatch-multihead-cap-mode",
        choices=("none", "hard", "soft"),
        default="none",
        help=(
            "Stage30-D: Final-decision cap mode for the temporal mismatch multihead. "
            "'none' (default): no cap applied. "
            "'hard': entitlement = min(entitlement, safe_factor). "
            "'soft': entitlement = entitlement * safe_factor.clamp_min(1e-8).pow(gamma). "
            "safe_factor = 1 - fused_mismatch_prob. "
            "Cannot combine with Stage30-C2 --v7-temporal-safety-cap-mode != none. "
            "Requires --v7-use-temporal-mismatch-multihead when not 'none'."
        ),
    )
    parser.add_argument(
        "--v7-temporal-mismatch-multihead-cap-gamma",
        type=float,
        default=1.0,
        help=(
            "Stage30-D: Gamma exponent for soft temporal mismatch multihead cap. "
            "Used only when --v7-temporal-mismatch-multihead-cap-mode soft. "
            "Must be > 0. Default 1.0."
        ),
    )
    parser.add_argument(
        "--v7-temporal-mismatch-multihead-cap-detach",
        action="store_true",
        default=False,
        help=(
            "Stage30-D: Detach safe_factor before applying the cap. "
            "Isolates cap effect from BCE gradients. Default off."
        ),
    )
    parser.add_argument(
        "--v7-temporal-mismatch-multihead-fusion",
        choices=("frame_only", "predicate_only", "sufficiency_only", "max", "noisy_or", "mean"),
        default="frame_only",
        help=(
            "Stage30-D: Fusion mode combining per-head mismatch probabilities. "
            "frame_only (default): fused = p_frame. "
            "predicate_only: fused = p_predicate. "
            "sufficiency_only: fused = p_sufficiency. "
            "max: fused = max(p_f, p_p, p_s). "
            "mean: fused = (p_f + p_p + p_s) / 3. "
            "noisy_or: fused = 1 - (1-p_f)(1-p_p)(1-p_s). "
            "Used for cap and for temporal_mismatch_fused_prob export."
        ),
    )

    #  Stage30-E: temporal residual preservation-aware cap 
    # Reuses Stage30-D temporal_mismatch_fused_prob as the soft risk signal.
    # Adds a narrow TemporalPreservationSignal head for the preservation signal.
    # Cannot combine Stage30-E preservation-aware cap with Stage30-D direct cap.
    parser.add_argument(
        "--v7-use-temporal-preservation-head",
        action="store_true",
        default=False,
        help=(
            "Stage30-E: Enable narrow TemporalPreservationSignal head for v7_hierarchical. "
            "Adds a small MLP over [frame_pair_repr, predicate_pair_repr, sufficiency_repr] "
            "producing temporal_preservation_prob in [0,1]. "
            "High prob = temporal relationship preserved (none/paraphrase). "
            "Low prob = not preserved (time_swap). "
            "Disabled by default; no effect on existing behavior when off. "
            "Requires --v7-use-temporal-mismatch-multihead for cap to function."
        ),
    )
    parser.add_argument(
        "--v7-use-temporal-preservation-loss",
        action="store_true",
        default=False,
        help=(
            "Stage30-E: Enable auxiliary BCE training for the temporal preservation head. "
            "Target: 1 for none/paraphrase (preserved), 0 for time_swap (not preserved). "
            "Reads stage30_temporal_safe_label if present (1=safe=preserved, 0=mismatch=not); "
            "falls back to intervention_type. "
            "Requires --v7-use-temporal-preservation-head and --v7-temporal-preservation-data. "
            "Stage15/OOD data must not be present in the data file. Default off."
        ),
    )
    parser.add_argument(
        "--v7-temporal-preservation-loss-weight",
        type=float,
        default=0.0,
        help="Stage30-E: Weight for temporal preservation BCE loss. Default 0.0.",
    )
    parser.add_argument(
        "--v7-temporal-preservation-loss-pos-weight",
        type=float,
        default=None,
        help=(
            "Stage30-E: BCE pos_weight for temporal preservation loss "
            "(positive = none/paraphrase / preserved). "
            "If omitted, no pos_weight is applied. Default None."
        ),
    )
    parser.add_argument(
        "--v7-temporal-preservation-data",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Stage30-E: Path to temporal preservation auxiliary JSONL. "
            "Same format as Stage30-C2/D temporal diagnostic data. "
            "none/paraphrase ??preserved=1; time_swap ??preserved=0. "
            "Never added to main train/dev; never used for checkpoint selection. "
            "Required when --v7-use-temporal-preservation-loss is enabled. "
            "Do not use Stage15 OOD data here."
        ),
    )
    parser.add_argument(
        "--v7-use-temporal-preservation-aware-cap",
        action="store_true",
        default=False,
        help=(
            "Stage30-E: Enable preservation-aware temporal residual cap. "
            "Formula: "
            "  effective_penalty = temporal_mismatch_fused_prob * (1 - temporal_preservation_prob); "
            "  safe_factor = (1 - effective_penalty).clamp(0,1) ** gamma; "
            "  entitlement_after = entitlement_after_location_cap * safe_factor. "
            "Requires --v7-use-temporal-mismatch-multihead (for risk signal) and "
            "--v7-use-temporal-preservation-head (for preservation signal). "
            "Cannot be combined with --v7-temporal-mismatch-multihead-cap-mode != none. "
            "Applied after Stage28-I location cap; does not apply before it. "
            "Does not increase entitlement above the pre-cap value. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--v7-temporal-preservation-cap-gamma",
        type=float,
        default=1.0,
        help=(
            "Stage30-E: Gamma exponent for temporal preservation-aware cap safe_factor. "
            "safe_factor = (1 - effective_penalty).clamp(0,1) ** gamma. "
            "Must be > 0. Default 1.0 (no additional sharpening)."
        ),
    )
    parser.add_argument(
        "--v7-temporal-preservation-cap-detach",
        action="store_true",
        default=False,
        help=(
            "Stage30-E: Detach both temporal_mismatch_fused_prob and "
            "temporal_preservation_prob before computing effective_penalty. "
            "Isolates cap effect from BCE/CE gradients. Default off."
        ),
    )

    # Stage31-C: directional Coverage/Entailment diagnostic owner.
    # Readout-only in this patch: no final-logit edits, no composer wiring, no cap.
    parser.add_argument(
        "--v7-use-coverage-entailment-head",
        action="store_true",
        default=False,
        help=(
            "Stage31-C: Enable a readout-only directional Coverage/Entailment "
            "diagnostic head over concat([frame_pair_repr, predicate_pair_repr, "
            "sufficiency_repr]). Exports diagnostic probabilities. Does not modify "
            "output['logits'], entitlement, H1 composer, caps, or final predictions."
        ),
    )
    parser.add_argument(
        "--v7-coverage-entailment-num-classes",
        type=int,
        choices=(3, 4),
        default=3,
        help=(
            "Stage31-C2: output dimension for the diagnostic coverage-entailment "
            "head. Default 3 for ENTAILS_SUPPORT, OVERCLAIM_NOT_ENTITLED, "
            "CONTRADICTS_REFUTE. Use 4 only for legacy OTHER_RESIDUAL exports."
        ),
    )
    parser.add_argument(
        "--v7-coverage-entailment-input-mode",
        choices=STAGE31C_INPUT_MODES,
        default="current",
        help=(
            "Stage31-C3: representation access ablation for the diagnostic "
            "coverage-entailment head. current uses frame/predicate/sufficiency "
            "pair representations; raw_pair uses claim/evidence frame states plus "
            "abs-diff and product; hybrid concatenates both."
        ),
    )
    parser.add_argument(
        "--v7-use-coverage-entailment-loss",
        action="store_true",
        default=False,
        help=(
            "Stage31-C: Enable auxiliary CE loss for the diagnostic "
            "Coverage/Entailment head using --v7-coverage-entailment-data only. "
            "Never uses data/stage31_coverage_entailment_probe.jsonl and never "
            "mixes auxiliary rows into main final-label batches."
        ),
    )
    parser.add_argument(
        "--v7-coverage-entailment-loss-weight",
        type=float,
        default=0.0,
        help="Stage31-C: Weight for auxiliary coverage-entailment CE loss.",
    )
    parser.add_argument(
        "--v7-coverage-entailment-data",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Stage31-C: Path to data/stage31c_coverage_entailment_aux.jsonl. "
            "The file's train/dev split is respected. Do not pass the Stage31 "
            "evaluation probe."
        ),
    )
    parser.add_argument(
        "--v7-coverage-entailment-loss-class-weights",
        type=str,
        default=None,
        help=(
            "Stage31-C: optional comma-separated CE class weights for "
            "the active coverage-entailment classes. Default mode expects "
            "ENTAILS_SUPPORT,OVERCLAIM_NOT_ENTITLED,CONTRADICTS_REFUTE."
        ),
    )
    parser.add_argument(
        "--v7-coverage-entailment-detach-input",
        action="store_true",
        default=False,
        help=(
            "Stage31-C: detach the selected diagnostic-head representation before "
            "the coverage-entailment head. Default off."
        ),
    )
    parser.add_argument(
        "--stage32-use-owner-state-schema",
        action="store_true",
        default=False,
        help=(
            "Stage32-A: enable diagnostic owner-state schema construction in "
            "shadow mode. Does not change logits, predictions, losses, caps, "
            "entitlement, composer logic, or checkpoint selection."
        ),
    )
    parser.add_argument(
        "--stage32-use-owner-interfaces",
        action="store_true",
        default=False,
        help=(
            "Stage32-B: build owner-state fields through explicit owner interface "
            "functions. Shadow/export-only; does not change final logits, final "
            "predictions, loss, caps, entitlement, composer logic, or selection."
        ),
    )
    parser.add_argument(
        "--stage32-owner-state-export",
        action="store_true",
        default=False,
        help=(
            "Stage32-A: include flattened owner-state proxy fields in prediction "
            "exports. Export-only; no effect on training or selection."
        ),
    )
    parser.add_argument(
        "--stage32-owner-state-shadow-mode",
        action="store_true",
        default=False,
        help=(
            "Stage32-A: mark owner-state schema as shadow-only and enforce the "
            "contract that Stage32 owner states do not modify final logits, final "
            "predictions, loss, entitlement, caps, or checkpoint selection."
        ),
    )
    parser.add_argument(
        "--stage32-coverage-owner-v2",
        action="store_true",
        default=False,
        help=(
            "Stage32-D: enable abstain-aware Coverage Owner v2 for Stage32 "
            "shadow owner-state export/composer fields only. Does not affect final "
            "logits, final predictions, losses, caps, entitlement, or selection."
        ),
    )
    parser.add_argument(
        "--stage32-coverage-owner-v2-min-confidence",
        type=float,
        default=0.50,
        help="Stage32-D: minimum top probability for Coverage Owner v2 confidence gating.",
    )
    parser.add_argument(
        "--stage32-coverage-owner-v2-min-margin",
        type=float,
        default=0.05,
        help="Stage32-D: minimum top-minus-second probability margin for Coverage Owner v2.",
    )
    parser.add_argument(
        "--stage32-coverage-owner-v2-allow-abstain",
        action="store_true",
        default=False,
        help=(
            "Stage32-D: allow low-confidence or low-margin Coverage Owner v2 "
            "predictions to abstain as UNRESOLVED_COVERAGE/RESIDUAL in the "
            "shadow composer only."
        ),
    )
    parser.add_argument(
        "--stage33-use-structured-coverage-owner",
        action="store_true",
        default=False,
        help=(
            "Stage33-A: compute deterministic structured Coverage Owner v0 "
            "diagnostics from claim/evidence text. Shadow/export-only."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-owner-export",
        action="store_true",
        default=False,
        help=(
            "Stage33-A: include structured coverage owner fields in Stage32 "
            "owner-state prediction exports."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-owner-shadow-mode",
        action="store_true",
        default=False,
        help=(
            "Stage33-A: allow structured coverage route to drive only the exported "
            "Stage32 shadow label/reason. Does not affect final logits/predictions."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-preserve-can-support",
        action="store_true",
        default=False,
        help=(
            "Stage33-A: in shadow mode only, allow structured entailment-preserve "
            "routes to recover SUPPORT even when polarity proxy is not SUPPORT."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-direct-support-rules",
        type=str,
        default=_STAGE33_DEFAULT_DIRECT_SUPPORT_RULES,
        help=(
            "Stage33-B: comma-separated structured rule reasons allowed to directly "
            "recover SUPPORT when preserve-can-support is enabled. Default: "
            f"{_STAGE33_DEFAULT_DIRECT_SUPPORT_RULES}."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-disable-specific-general-direct-support",
        action="store_true",
        default=False,
        help=(
            "Stage33-B: prevent specific_to_general_proxy from directly recovering "
            "SUPPORT in the structured shadow composer."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-weak-rules-to-residual",
        type=str,
        default="",
        help=(
            "Stage33-B: comma-separated structured rule reasons to force to "
            "STRUCT_UNRESOLVED/RESIDUAL while preserving the original reason field."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-conditional-fallback",
        action="store_true",
        default=False,
        help=(
            "Stage33-C: in structured shadow mode, apply only high-precision local "
            "structured overrides and fall back to the current final prediction for "
            "residual, unresolved, weak, or blocked cases."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-fallback-source",
        choices=("current_final", "h1_current"),
        default="current_final",
        help=(
            "Stage33-C: source label for conditional fallback. h1_current is treated "
            "as the current final prediction in this implementation."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-enable-whole-part-rules",
        action="store_true",
        default=False,
        help=(
            "Stage33-D: enable conservative whole/part structured coverage rules "
            "for known Stage31-style lexicon pairs. Shadow/export-only."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-whole-part-direct-support",
        action="store_true",
        default=False,
        help=(
            "Stage33-D: allow whole_to_part_proxy to recover SUPPORT directly only "
            "inside conditional fallback shadow mode."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-whole-part-lexicon",
        type=str,
        default="",
        help=(
            "Stage33-D: optional comma-separated custom whole/part pairs using "
            "'whole->part' or 'whole:part' syntax."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-whole-part-v2",
        action="store_true",
        default=False,
        help=(
            "Stage33-E: enable whole/part structured coverage v2. Requires "
            "--stage33-structured-coverage-enable-whole-part-rules."
        ),
    )
    parser.add_argument(
        "--stage33-structured-coverage-whole-part-v2-use-expanded-lexicon",
        action="store_true",
        default=False,
        help="Stage33-E: include the expanded built-in whole/part v2 lexicon.",
    )
    parser.add_argument(
        "--stage33-structured-coverage-whole-part-v2-direct-support-policy",
        choices=("off", "hard_core_required", "conditional_safe"),
        default="hard_core_required",
        help=(
            "Stage33-E: direct SUPPORT policy for matched whole_to_part_proxy in "
            "shadow mode."
        ),
    )
    parser.add_argument(
        "--stage33-whole-part-conditional-safe-overrides-hard-core",
        action="store_true",
        default=False,
        help=(
            "Stage33-F: ablation flag allowing conditional-safe whole/part direct "
            "SUPPORT to override hard-core in shadow mode only."
        ),
    )

    #  Stage36-A: conservative support-safety blockers (shadow-only) 
    # All off by default. When off, behavior is identical to Stage33-F. When on,
    # blockers only affect exported shadow/diagnostic SUPPORT overrides -- never
    # final logits or final classifier predictions.
    parser.add_argument(
        "--stage36-use-support-safety-blockers",
        action="store_true",
        default=False,
        help=(
            "Stage36-A: compute deterministic conservative safety blockers that "
            "fire before Stage33 structured owner SUPPORT overrides are treated "
            "as safe. Shadow/diagnostic only. Default off (identical to Stage33-F)."
        ),
    )
    parser.add_argument(
        "--stage36-support-safety-export",
        action="store_true",
        default=False,
        help="Stage36-A: include stage36_* diagnostic fields in prediction exports.",
    )
    parser.add_argument(
        "--stage36-support-safety-shadow-mode",
        action="store_true",
        default=False,
        help=(
            "Stage36-A: allow a fired support-safety blocker to replace the "
            "exported Stage32/Stage33 shadow label. Does not affect final "
            "logits/predictions."
        ),
    )
    parser.add_argument(
        "--stage36-block-exception-scope",
        action="store_true",
        default=False,
        help=(
            "Stage36-A: block SUPPORT when evidence has an 'all X except Y' "
            "exclusion clause and the claim targets the excluded subset Y."
        ),
    )
    parser.add_argument(
        "--stage36-block-not-all-existential",
        action="store_true",
        default=False,
        help=(
            "Stage36-A: block SUPPORT when evidence asserts 'not all/every X' and "
            "the claim asserts 'some X' (not-all does not entail some)."
        ),
    )
    parser.add_argument(
        "--stage36-block-location-scope-mismatch",
        action="store_true",
        default=False,
        help=(
            "Stage36-A: block SUPPORT when claim and evidence both carry explicit "
            "location/scope markers (e.g. east vs west district) that conflict."
        ),
    )
    parser.add_argument(
        "--stage36-block-temporal-scope-mismatch",
        action="store_true",
        default=False,
        help=(
            "Stage36-A: block SUPPORT when claim and evidence both carry explicit "
            "temporal markers (year, weekday, quarter, relative phrase) that "
            "conflict."
        ),
    )
    parser.add_argument(
        "--stage36-support-blocker-action",
        choices=("fallback_current_final", "force_not_entitled"),
        default="fallback_current_final",
        help=(
            "Stage36-A: action to take when a support-safety blocker fires. "
            "fallback_current_final uses the current final label (or "
            "NOT_ENTITLED if that is also SUPPORT); force_not_entitled always "
            "sets NOT_ENTITLED."
        ),
    )

    #  Stage37-A: conservative safe SUPPORT recovery (shadow-only) 
    # All off by default. When off, behavior is identical to Stage36-A. When on,
    # recovery only affects exported shadow/diagnostic SUPPORT overrides -- never
    # final logits or final classifier predictions, and never overrides a fired
    # Stage36 blocker.
    parser.add_argument(
        "--stage37-use-safe-support-recovery",
        action="store_true",
        default=False,
        help=(
            "Stage37-A: compute deterministic conservative safe SUPPORT recovery "
            "rules that fire after Stage36's post-blocker shadow label is known. "
            "Shadow/diagnostic only. Default off (identical to Stage36-A)."
        ),
    )
    parser.add_argument(
        "--stage37-safe-support-export",
        action="store_true",
        default=False,
        help="Stage37-A: include stage37_* diagnostic fields in prediction exports.",
    )
    parser.add_argument(
        "--stage37-safe-support-shadow-mode",
        action="store_true",
        default=False,
        help=(
            "Stage37-A: allow a fired safe SUPPORT recovery rule to replace the "
            "exported Stage32/Stage33/Stage36 shadow label. Does not affect "
            "final logits/predictions."
        ),
    )
    parser.add_argument(
        "--stage37-recover-no-except-included-subset",
        action="store_true",
        default=False,
        help=(
            "Stage37-A: recover SUPPORT for 'no X except Y ...' evidence when "
            "the claim targets the included subset Y."
        ),
    )
    parser.add_argument(
        "--stage37-recover-coordination-universal-subset",
        action="store_true",
        default=False,
        help=(
            "Stage37-A: recover SUPPORT for 'all X and all Z were P' evidence "
            "when the claim targets a subset of X or Z."
        ),
    )
    parser.add_argument(
        "--stage37-recover-numeric-universal-subset",
        action="store_true",
        default=False,
        help=(
            "Stage37-A: recover SUPPORT for 'all N X were P' evidence when the "
            "claim targets a clear subset among X (e.g. 'among the X')."
        ),
    )
    parser.add_argument(
        "--stage37-allow-recover-from-refute",
        action="store_true",
        default=False,
        help=(
            "Stage37-A: allow safe SUPPORT recovery to fire when the post-"
            "Stage36 shadow label is REFUTE, not just NOT_ENTITLED. Default "
            "off: recovery only fires from NOT_ENTITLED."
        ),
    )

    #  Stage39-A: opt-in final composer validation (prediction/export-time) 
    # All off by default. When off, final predictions/metrics/exported labels
    # are identical to Stage38/Stage37. When on, composes a candidate final
    # label from a Stage37/Stage36/Stage32 shadow label under deterministic
    # safety guards. Never touches logits, training, or loss.
    parser.add_argument(
        "--stage39-use-final-composer-opt-in",
        action="store_true",
        default=False,
        help=(
            "Stage39-A: enable deterministic final composer candidate "
            "generation. Shadow/diagnostic only unless combined with "
            "--stage39-final-composer-output-mode replace_pred_final_label. "
            "Default off (identical to Stage38/Stage37)."
        ),
    )
    parser.add_argument(
        "--stage39-final-composer-export",
        action="store_true",
        default=False,
        help=(
            "Stage39-A: include stage39_* diagnostic fields in prediction "
            "exports even if the final prediction is not replaced."
        ),
    )
    parser.add_argument(
        "--stage39-final-composer-policy",
        choices=("support_only", "safe_structured", "full_shadow", "safe_structured_v2"),
        default="support_only",
        help=(
            "Stage39-A: composition policy. support_only only composes "
            "SUPPORT from the source shadow label; safe_structured also "
            "allows high-precision-contradiction REFUTE; full_shadow "
            "diagnostically adopts the source shadow label outright, still "
            "subject to hard safety guards. Stage39-C: safe_structured_v2 "
            "keeps all safe_structured SUPPORT behavior and safety guards, "
            "and broadens high-precision-contradiction REFUTE composition "
            "(OR across Stage33 route/label/override/reason signals, "
            "restricted to original final label NOT_ENTITLED only -- never "
            "SUPPORT->REFUTE)."
        ),
    )
    parser.add_argument(
        "--stage39-final-composer-output-mode",
        choices=("export_only", "replace_pred_final_label"),
        default="export_only",
        help=(
            "Stage39-A: export_only exports stage39_composed_final_label "
            "without changing pred_final_label; replace_pred_final_label "
            "replaces the exported pred_final_label with the composed label "
            "(only takes effect together with --stage39-use-final-composer-"
            "opt-in)."
        ),
    )
    parser.add_argument(
        "--stage39-final-composer-source",
        choices=(
            "stage37_final_shadow_label",
            "stage36_final_shadow_label",
            "stage32_shadow_label",
        ),
        default="stage37_final_shadow_label",
        help=(
            "Stage39-A: which shadow label the composer treats as the "
            "candidate source."
        ),
    )
    parser.add_argument(
        "--stage39-disallow-refute-to-support",
        action="store_true",
        default=True,
        help=(
            "Stage39-A: if the current final label is REFUTE and the source "
            "shadow label is SUPPORT, do not compose to SUPPORT. Default on."
        ),
    )
    parser.add_argument(
        "--stage39-require-stage36-safety-clear",
        action="store_true",
        default=True,
        help=(
            "Stage39-A: if a Stage36 support-safety blocker fired on a row, "
            "do not compose to SUPPORT. Default on."
        ),
    )
    parser.add_argument(
        "--stage39-require-stage37-not-from-refute",
        action="store_true",
        default=True,
        help=(
            "Stage39-A: if Stage37 recovered the shadow label from REFUTE, "
            "do not compose to SUPPORT. Default on."
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
    # v7 auxiliary loss flags ??all off by default; no v7 aux losses active in Stage26-A.
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

    #  Stage26-G: v7 stabilization options 
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

    #  Generic preservation-constrained checkpoint selection 
    # Default off. Uses ONLY clean dev pairwise checks ??no Stage15/OOD, no temporal diagnostic
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
            "(from clean dev pairwise checks only ??Stage15/OOD is never consulted). "
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

    #  Stage44-B: internal-only anti-collapse checkpoint selection
    parser.add_argument(
        "--stage44-use-anti-collapse-selection",
        action="store_true",
        default=False,
        help=(
            "Stage44-B: enable internal clean-dev-only checkpoint selection constraints "
            "for SUPPORT/REFUTE recall and NOT_ENTITLED prediction rate. Default off."
        ),
    )
    parser.add_argument(
        "--stage44-min-support-recall",
        type=float,
        default=None,
        help="Stage44-B: optional minimum internal clean-dev SUPPORT recall.",
    )
    parser.add_argument(
        "--stage44-min-refute-recall",
        type=float,
        default=None,
        help="Stage44-B: optional minimum internal clean-dev REFUTE recall.",
    )
    parser.add_argument(
        "--stage44-max-not-entitled-pred-rate",
        type=float,
        default=None,
        help="Stage44-B: optional maximum internal clean-dev NOT_ENTITLED prediction rate.",
    )
    parser.add_argument(
        "--stage44-min-clean-dev-accuracy",
        type=float,
        default=None,
        help="Stage44-B: optional minimum internal clean-dev final accuracy.",
    )
    parser.add_argument(
        "--stage44-selection-fallback",
        choices=("best_metric", "fail_incomplete"),
        default="best_metric",
        help=(
            "Stage44-B: fallback when no epoch satisfies anti-collapse constraints. "
            "best_metric keeps the original best-metric checkpoint; fail_incomplete "
            "marks Stage44-B incomplete while preserving normal output behavior."
        ),
    )
    parser.add_argument(
        "--stage44-selection-report-json",
        type=Path,
        default=None,
        help=(
            "Stage44-B: optional path for a standalone internal-only selection report JSON. "
            "If omitted, fields are embedded in the normal output JSON only."
        ),
    )
    parser.add_argument(
        "--stage44-use-prior-aware-ne-constraint",
        action="store_true",
        default=False,
        help=(
            "Stage44-B2: enable prior-aware NOT_ENTITLED prediction-rate constraint "
            "using the internal clean-dev gold NOT_ENTITLED rate plus optional delta. "
            "Default off."
        ),
    )
    parser.add_argument(
        "--stage44-max-ne-gold-prior-delta",
        type=float,
        default=None,
        help=(
            "Stage44-B2: optional maximum allowed excess of internal clean-dev "
            "NOT_ENTITLED prediction rate over the internal clean-dev gold NOT_ENTITLED rate."
        ),
    )
    parser.add_argument(
        "--stage44-min-macro-f1",
        type=float,
        default=None,
        help="Stage44-B2: optional minimum internal clean-dev final macro-F1.",
    )
    parser.add_argument(
        "--stage44-min-relative-macro-f1-of-best",
        type=float,
        default=None,
        help=(
            "Stage44-B2: optional minimum selected macro-F1 as a fraction of the "
            "original best-metric checkpoint macro-F1."
        ),
    )
    parser.add_argument(
        "--stage44-min-support-precision",
        type=float,
        default=None,
        help="Stage44-B2: optional minimum internal clean-dev SUPPORT precision.",
    )
    parser.add_argument(
        "--stage44-min-refute-precision",
        type=float,
        default=None,
        help="Stage44-B2: optional minimum internal clean-dev REFUTE precision.",
    )

    # Stage45-B: internal leave-family-out validation scaffold
    parser.add_argument(
        "--stage45-use-family-holdout",
        action="store_true",
        default=False,
        help=(
            "Stage45-B: replace the normal internal train/dev split with an "
            "internal leave-family-out split. Default off. Does not read external data."
        ),
    )
    parser.add_argument(
        "--stage45-family-field",
        default="auto",
        help=(
            "Stage45-B: family metadata field to use for internal holdout splitting, "
            "or 'auto' for the shared preferred-field resolver. Also accepts the "
            "Stage45-B1 recovered fields 'intervention_type' and "
            "'primary_failure_type', or composite fields "
            "'intervention_type+primary_failure_type', "
            "'intervention_type+final_label', 'primary_failure_type+final_label'. "
            "Default auto."
        ),
    )
    parser.add_argument(
        "--stage45-holdout-family",
        default=None,
        help="Stage45-B: internal transformation family to hold out as dev/validation.",
    )
    parser.add_argument(
        "--stage45-min-holdout-size",
        type=int,
        default=20,
        help="Stage45-B: minimum required internal holdout rows. Default 20.",
    )
    parser.add_argument(
        "--stage45-family-holdout-report-json",
        type=Path,
        default=None,
        help="Stage45-B: optional standalone internal family-holdout report JSON.",
    )
    parser.add_argument(
        "--stage45-family-holdout-report-md",
        type=Path,
        default=None,
        help="Stage45-B: optional standalone internal family-holdout report Markdown.",
    )

    # Stage45-C: internal-only SUPPORT entitlement recovery scaffold
    parser.add_argument(
        "--stage45c-enable-support-recovery",
        action="store_true",
        default=False,
        help=(
            "Stage45-C: enable an optional internal-only auxiliary loss targeting "
            "SUPPORT under-recall and entitled-to-NOT_ENTITLED over-rejection, "
            "computed only from the internal training split. Default off; leaves "
            "existing training behavior unchanged when off."
        ),
    )
    parser.add_argument(
        "--stage45c-support-recovery-weight",
        type=float,
        default=0.0,
        help=(
            "Stage45-C: weight for the SUPPORT recovery auxiliary term "
            "(penalizes low predicted --stage45c-target-label probability on gold "
            "--stage45c-target-label training rows). Default 0.0 (inactive)."
        ),
    )
    parser.add_argument(
        "--stage45c-entitled-ne-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Stage45-C: weight for the entitled-NOT_ENTITLED over-rejection penalty "
            "(penalizes high predicted NOT_ENTITLED probability on gold "
            "--stage45c-entitled-labels training rows). Default 0.0 (inactive)."
        ),
    )
    parser.add_argument(
        "--stage45c-target-label",
        default="SUPPORT",
        help="Stage45-C: gold label targeted by the SUPPORT recovery term. Default SUPPORT.",
    )
    parser.add_argument(
        "--stage45c-entitled-labels",
        default="SUPPORT,REFUTE",
        help=(
            "Stage45-C: comma-separated gold labels considered 'entitled' for the "
            "over-rejection penalty. Default SUPPORT,REFUTE."
        ),
    )
    parser.add_argument(
        "--stage45c-report-json",
        type=Path,
        default=None,
        help="Stage45-C: optional standalone internal support-recovery report JSON.",
    )
    parser.add_argument(
        "--stage45c-report-md",
        type=Path,
        default=None,
        help="Stage45-C: optional standalone internal support-recovery report Markdown.",
    )

    # Stage48: optional load of the Stage47-validated frozen recovery config
    parser.add_argument(
        "--use-stage47-selected-recovery-config",
        action="store_true",
        default=False,
        help=(
            "Stage48: load the Stage47-validated frozen recovery config "
            "(recovery_w01_ne01: support_w=0.1, ne_w=0.1) and apply it to "
            "--stage45c-support-recovery-weight / --stage45c-entitled-ne-penalty-weight, "
            "instead of retyping those weights manually. Fails fast if the Stage47 "
            "file is missing or does not match the frozen selection. Default off; "
            "leaves existing training behavior unchanged when off."
        ),
    )
    parser.add_argument(
        "--stage47-recovery-config-path",
        type=Path,
        default=Path("results/stage47_selected_recovery_config_check.json"),
        help=(
            "Stage48: path to the Stage47 selected recovery config-check JSON. "
            "Only read when --use-stage47-selected-recovery-config is set. "
            "Default: results/stage47_selected_recovery_config_check.json"
        ),
    )

    # Stage28-E: prediction export schema version
    parser.add_argument(
        "--prediction-export-schema",
        choices=("legacy", "stage28e_v1"),
        default="stage28e_v1",
        help=(
            "Stage28-E: prediction export schema version. "
            "'stage28e_v1' (default) writes enriched per-record fields and adds "
            "label_space/config_summary to export metadata. "
            "'legacy' omits Stage28-E metadata additions (per-record enrichment is "
            "always written since it is additive and backward-compatible)."
        ),
    )
    parser.add_argument(
        "--stage128-enable-location-slot-guard",
        action="store_true",
        default=False,
        help=(
            "Stage128-B: enable eval/export-only controlled location-slot guard. "
            "Default off; never affects training, loss computation, logits, or "
            "checkpoint selection."
        ),
    )
    parser.add_argument(
        "--stage128-location-slot-guard-mode",
        choices=STAGE128_LOCATION_SLOT_GUARD_MODES,
        default="off",
        help=(
            "Stage128-B guard mode. off preserves existing exports; "
            "controlled_in_during_location_mismatch extracts controlled "
            "'in <Location> during' slots from claim and evidence/core text "
            "and changes exported SUPPORT predictions to NOT_ENTITLED only "
            "when both slots are present and unequal."
        ),
    )

    parser.add_argument(
        "--stage115-clean-dev-scalar-output-jsonl",
        type=Path,
        default=None,
        help=(
            "Stage115-B: optional JSONL path for best internal clean-dev per-row "
            "vNext scalar diagnostics. Eval/export only; default off."
        ),
    )
    parser.add_argument(
        "--stage118-diagnostic-jsonl",
        type=Path,
        default=None,
        help=(
            "Stage118: optional generic controlled-style diagnostic JSONL to evaluate "
            "after best clean-dev state restoration. Does not use Stage43/VitaminC "
            "external filtering."
        ),
    )
    parser.add_argument(
        "--stage118-diagnostic-output-jsonl",
        type=Path,
        default=None,
        help="Stage118: output JSONL path for generic diagnostic predictions.",
    )
    parser.add_argument(
        "--stage118-diagnostic-summary-json",
        type=Path,
        default=None,
        help="Stage118: output JSON path for generic diagnostic summary metrics.",
    )
    parser.add_argument(
        "--stage118-diagnostic-batch-size",
        type=int,
        default=16,
        help="Stage118: eval batch size for generic diagnostic JSONL. Default: 16.",
    )
    parser.add_argument(
        "--stage118-diagnostic-name",
        type=str,
        default="stage118_generic_diagnostic",
        help="Stage118: diagnostic name stamped into predictions and summary.",
    )
    parser.add_argument(
        "--stage118-diagnostic-evidence-interface",
        choices=STAGE118_DIAGNOSTIC_EVIDENCE_INTERFACE_CHOICES,
        default="same_as_vnext",
        help=(
            "Stage123-A2: evidence interface for Stage118 generic diagnostics. "
            "same_as_vnext uses --vnext-evidence-interface; other choices override "
            "diagnostic input evidence only."
        ),
    )
    parser.add_argument(
        "--stage118-diagnostic-evidence-interface-sweep",
        type=str,
        default="",
        help=(
            "Stage123-A3: comma-separated Stage118 diagnostic evidence interfaces "
            "to evaluate in one process after the selected model is ready. Valid "
            "values: full_evidence, core_only, core_first_context_suffix, "
            "context_prefix_core, core_marker_context_suffix, "
            "segmented_dual_pass_scaffold, segmented_dual_pass. Empty preserves existing behavior."
        ),
    )
    parser.add_argument(
        "--stage118-diagnostic-sweep-output-dir",
        type=Path,
        default=None,
        help=(
            "Stage123-A3: output directory for Stage118 diagnostic sweep prediction, "
            "summary, and manifest files. Required when "
            "--stage118-diagnostic-evidence-interface-sweep is non-empty."
        ),
    )
    parser.add_argument(
        "--stage126-preflight-export-only",
        action="store_true",
        default=False,
        help=(
            "Stage126: run only a tiny segmented_dual_pass+risk_cap Stage118 "
            "prediction export preflight, then exit before training."
        ),
    )
    parser.add_argument(
        "--stage126-preflight-max-rows",
        type=int,
        default=32,
        help="Stage126: maximum diagnostic rows for --stage126-preflight-export-only. Default: 32.",
    )
    parser.add_argument(
        "--stage126-preflight-output-dir",
        type=Path,
        default=None,
        help=(
            "Stage126: optional output directory for preflight prediction/summary files. "
            "Default: results/stage126_preflight."
        ),
    )

    # Stage29-B: external probe evaluation (eval-only; no effect on training or selection)
    parser.add_argument(
        "--external-eval-jsonl",
        action="append",
        default=[],
        dest="external_eval_jsonl",
        metavar="PATH",
        help=(
            "Stage29-B: path to an external probe JSONL to evaluate on the selected best "
            "checkpoint. May be repeated for multiple probe files. "
            "External probe records are NEVER added to train or dev; NEVER used for "
            "checkpoint selection, calibration, or loss computation. "
            "Example: --external-eval-jsonl data/stage15_slot_sensitivity_probe.jsonl "
            "--external-eval-jsonl data/stage10a_number_swap_probe.jsonl"
        ),
    )
    parser.add_argument(
        "--external-eval-name",
        action="append",
        default=[],
        dest="external_eval_name",
        metavar="NAME",
        help=(
            "Stage29-B: human-readable name for the corresponding --external-eval-jsonl file. "
            "If provided, count must match --external-eval-jsonl count. "
            "If omitted, names are derived from the file stem. "
            "Example: --external-eval-name stage15_slot --external-eval-name stage10a_number"
        ),
    )
    parser.add_argument(
        "--external-output-dir",
        type=str,
        default=None,
        dest="external_output_dir",
        metavar="DIR",
        help=(
            "Stage29-B: directory to write per-probe prediction JSON files. "
            "One file per --external-eval-jsonl, named "
            "external_probe_{name}_predictions.json. "
            "If omitted, prediction records are not written to disk."
        ),
    )


    # Stage43-C0: post-training external fact-verification evaluation hook.
    # Explicit opt-in only; runs after normal best-state restoration and never
    # participates in training, calibration, threshold tuning, or selection.
    parser.add_argument(
        "--stage43-external-factver-jsonl",
        action="append",
        default=[],
        dest="stage43_external_factver_jsonl",
        metavar="PATH",
        help=(
            "Stage43-C0: path to an external fact-verification JSONL file "
            "(e.g. VitaminC or Climate-FEVER acquisition output). May be repeated. "
            "Only evaluated when --enable-stage43-external-eval is set."
        ),
    )
    parser.add_argument(
        "--stage43-external-output-dir",
        type=str,
        default="reports",
        help="Stage43-C0: output directory for external fact-verification reports.",
    )
    parser.add_argument(
        "--stage43-external-run-prefix",
        type=str,
        default="stage43c0",
        help="Stage43-C0: prefix for per-dataset and aggregate report files.",
    )
    parser.add_argument(
        "--stage43-external-max-rows",
        type=int,
        default=None,
        help="Stage43-C0: optional maximum rows to read from each external JSONL.",
    )
    parser.add_argument(
        "--stage43-external-batch-size",
        type=int,
        default=None,
        help=(
            "Stage43-C0: eval batch size for external fact-verification files. "
            "Defaults to --batch-size when available, otherwise 8."
        ),
    )
    parser.add_argument(
        "--stage43-external-enable-shadow-export",
        action="store_true",
        default=False,
        help=(
            "Stage43-C2: during external fact-verification eval only, export the "
            "Stage32/33/36/37 shadow fields required by Stage39 safe_structured_v2. "
            "Export/composer diagnostic only; does not alter base predictions, "
            "training, calibration, thresholds, or checkpoint selection."
        ),
    )
    parser.add_argument(
        "--enable-stage43-external-eval",
        action="store_true",
        default=False,
        help=(
            "Stage43-C0: explicitly enable post-training external fact-verification "
            "evaluation. Default off."
        ),
    )
    parser.add_argument(
        "--stage57-bridge-train-jsonl",
        type=str,
        default=None,
        help=(
            "Stage60: optional path to the Stage57 non-leaking external bridge JSONL "
            "(e.g. data/stage57_nonleaking_external_bridge.jsonl). Only used when "
            "--stage57-bridge-train-mode=append_train_only. Default: None, which "
            "preserves current training/data split behavior exactly."
        ),
    )
    parser.add_argument(
        "--stage57-bridge-train-mode",
        choices=("none", "append_train_only"),
        default="none",
        help=(
            "Stage60: whether to append Stage57 bridge rows to the train split only, "
            "after the clean main train/dev split is created. 'none' (default) leaves "
            "current behavior unchanged; the clean dev split always remains the "
            "checkpoint-selection/dev source. 'append_train_only' appends the rows "
            "loaded from --stage57-bridge-train-jsonl to train only."
        ),
    )
    parser.add_argument(
        "--stage66-bridge-train-jsonl",
        type=str,
        default=None,
        help=(
            "Stage69/Stage70: optional path to the Stage66 residual bridge JSONL "
            "(e.g. data/stage66_residual_bridge.jsonl). Only used when "
            "--stage66-bridge-train-mode=append_train_only. Default: None, which "
            "preserves current training/data split behavior exactly."
        ),
    )
    parser.add_argument(
        "--stage66-bridge-train-mode",
        choices=("none", "append_train_only"),
        default="none",
        help=(
            "Stage69/Stage70: whether to append Stage66 residual bridge rows to the "
            "train split only, after the clean main train/dev split is created (and "
            "after any Stage57 bridge rows are appended). 'none' (default) leaves "
            "current behavior unchanged; the clean dev split always remains the "
            "checkpoint-selection/dev source. 'append_train_only' appends the rows "
            "loaded from --stage66-bridge-train-jsonl to train only."
        ),
    )
    parser.add_argument(
        "--stage75-bridge-train-jsonl",
        type=str,
        default=None,
        help=(
            "Stage75C: optional path to the Stage75 targeted residual bridge JSONL "
            "(e.g. data/stage75_targeted_residual_bridge.jsonl). Only used when "
            "--stage75-bridge-train-mode=append_train_only. Default: None, which "
            "preserves current training/data split behavior exactly."
        ),
    )
    parser.add_argument(
        "--stage75-bridge-train-mode",
        choices=("none", "append_train_only"),
        default="none",
        help=(
            "Stage75C: whether to append Stage75 targeted residual bridge rows to "
            "the train split only, after the clean main train/dev split is created "
            "(and after any Stage57/Stage66 bridge rows are appended). 'none' "
            "(default) leaves current behavior unchanged; the clean dev split always "
            "remains the checkpoint-selection/dev source. 'append_train_only' "
            "appends the rows loaded from --stage75-bridge-train-jsonl to train only."
        ),
    )
    parser.add_argument(
        "--stage80a-bridge-train-jsonl",
        type=str,
        default=None,
        help=(
            "Stage80D: optional path to the Stage80A conservative Stage75v2 bridge "
            "JSONL (e.g. data/stage80a_conservative_stage75v2_bridge.jsonl). Only "
            "used when --stage80a-bridge-train-mode=append_train_only. Default: "
            "None, which preserves current training/data split behavior exactly."
        ),
    )
    parser.add_argument(
        "--stage80a-bridge-train-mode",
        choices=("none", "append_train_only"),
        default="none",
        help=(
            "Stage80D: whether to append Stage80A conservative Stage75v2 bridge "
            "rows to the train split only, after the clean main train/dev split is "
            "created (and after any Stage57/Stage66/Stage75 bridge rows are "
            "appended). 'none' (default) leaves current behavior unchanged; the "
            "clean dev split always remains the checkpoint-selection/dev source. "
            "'append_train_only' appends the rows loaded from "
            "--stage80a-bridge-train-jsonl to train only."
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
# Stage29-B: external probe evaluation helpers
# ---------------------------------------------------------------------------


def load_external_probe_jsonl(path: "Path") -> list[dict]:
    """Load external probe JSONL records tolerantly (no schema enforcement)."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Stage73: external fact-verification schema normalization (eval-only)
#
# Fixes external eval crashing with `KeyError: 'final_label'` when
# --external-eval-jsonl points at a Stage43B1-style external fact-verification
# file (e.g. data/stage43b1_vitaminc_validation_sample1000.jsonl) whose schema
# differs from the controlled v5/v6 training/dev schema expected by
# v5.encode_mamba_records / v5.encode_label_tensors. This normalization is
# runner-side only: it never touches src/contramamba/losses.py, never affects
# controlled train/dev construction, and derived labels are used solely to
# satisfy the encoder's tensor-construction requirements for eval-only
# forward passes (never for training, calibration, threshold selection, or
# checkpoint selection).
# ---------------------------------------------------------------------------

_STAGE73_LABEL_FIELD_CANDIDATES: tuple[str, ...] = (
    "final_label",
    "label",
    "gold",
    "gold_label",
    "answer",
    "verdict",
    "fact_label",
    "vitaminc_label",
    "stage43_label",
    "original_label",
)

_STAGE73_CLAIM_FIELD_CANDIDATES: tuple[str, ...] = (
    "claim",
    "hypothesis",
    "statement",
    "query",
)

_STAGE73_EVIDENCE_FIELD_CANDIDATES: tuple[str, ...] = (
    "evidence",
    "premise",
    "context",
    "passage",
    "document",
    "evidence_text",
)

# Case-insensitive canonicalization: raw values are upper-cased and any run of
# whitespace/hyphen characters is collapsed to a single underscore before
# lookup, so "Not Enough Info", "not-enough-info", and "NOT_ENOUGH_INFO" all
# resolve to the same key.
_STAGE73_LABEL_CANONICAL_MAP: dict[str, str] = {
    "SUPPORT": "SUPPORT",
    "SUPPORTS": "SUPPORT",
    "ENTAILMENT": "SUPPORT",
    "ENTAILED": "SUPPORT",
    "REFUTE": "REFUTE",
    "REFUTES": "REFUTE",
    "CONTRADICTION": "REFUTE",
    "CONTRADICTS": "REFUTE",
    "NOT_ENTITLED": "NOT_ENTITLED",
    "NEI": "NOT_ENTITLED",
    "NOT_ENOUGH_INFO": "NOT_ENTITLED",
    "UNKNOWN": "NOT_ENTITLED",
    "UNVERIFIABLE": "NOT_ENTITLED",
}

# Stage73 requirement 7: recommended auxiliary-label defaults, keyed by the
# canonicalized final_label. Only used to satisfy v5.encode_label_tensors for
# external eval; never used for training losses or checkpoint selection.
_STAGE73_AUX_LABEL_DEFAULTS_BY_FINAL_LABEL: dict[str, dict[str, Any]] = {
    "SUPPORT": {
        "frame_compatible_label": 1,
        "predicate_covered_label": 1,
        "sufficiency_label": 1,
        "polarity_label": "SUPPORT",
    },
    "REFUTE": {
        "frame_compatible_label": 1,
        "predicate_covered_label": 1,
        "sufficiency_label": 1,
        "polarity_label": "REFUTE",
    },
    "NOT_ENTITLED": {
        "frame_compatible_label": 0,
        "predicate_covered_label": 0,
        "sufficiency_label": 0,
        "polarity_label": "NONE",
    },
}


def _stage73_canonicalize_label(raw: Any) -> "str | None":
    """Case-insensitive canonicalization of an external label value.

    Returns None (rather than raising) when the value is unmapped so callers
    can produce a clear, contextualized ValueError.
    """
    if raw is None:
        return None
    key = re.sub(r"[\s\-]+", "_", str(raw).strip().upper())
    return _STAGE73_LABEL_CANONICAL_MAP.get(key)


def _stage73_first_present_field(
    record: dict, candidates: tuple[str, ...]
) -> "tuple[str | None, Any]":
    """Return the (field_name, value) of the first candidate with a non-empty value."""
    for field in candidates:
        if field not in record:
            continue
        value = record[field]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return field, value
    return None, None


def normalize_external_factver_record(
    record: dict[str, Any], index: int
) -> "tuple[dict[str, Any], dict[str, Any]]":
    """Normalize one external fact-verification record (Stage43B1 / VitaminC-style
    schema) into the controlled v5/v6 schema expected by v5.encode_mamba_records
    and v5.encode_label_tensors.

    Never mutates the input record. Preserves all original fields on the
    returned copy and adds provenance metadata
    (external_original_label / external_schema_normalized /
    external_schema_source). Raises ValueError with the record id and
    available keys when a required field cannot be resolved.

    Returns (normalized_record, stats) where stats reports what this record
    needed: which label/claim/evidence field was used, whether final_label
    was missing beforehand, whether auxiliary labels were defaulted, and
    whether the record required any normalization at all.
    """
    out = dict(record)
    available_keys = sorted(record.keys())
    record_id = record.get("id")
    if record_id is None:
        record_id = record.get("pair_id")
    if record_id is None:
        record_id = f"external_row_{index}"

    # --- final_label -------------------------------------------------------
    final_label_was_missing = "final_label" not in record or record.get("final_label") is None
    label_field, label_value = _stage73_first_present_field(
        record, _STAGE73_LABEL_FIELD_CANDIDATES
    )
    if label_field is None:
        raise ValueError(
            "External eval record has no usable label field: "
            f"record_id={record_id!r} "
            f"candidate_label_fields={list(_STAGE73_LABEL_FIELD_CANDIDATES)} "
            f"available_keys={available_keys}"
        )
    canonical_label = _stage73_canonicalize_label(label_value)
    if canonical_label is None:
        raise ValueError(
            "External eval label cannot be canonicalized: "
            f"record_id={record_id!r} label_value={label_value!r} "
            f"label_field={label_field!r} "
            f"candidate_label_fields={list(_STAGE73_LABEL_FIELD_CANDIDATES)} "
            f"available_keys={available_keys}"
        )
    out["final_label"] = canonical_label

    # --- claim / evidence ---------------------------------------------------
    claim_field, claim_value = _stage73_first_present_field(
        record, _STAGE73_CLAIM_FIELD_CANDIDATES
    )
    evidence_field, evidence_value = _stage73_first_present_field(
        record, _STAGE73_EVIDENCE_FIELD_CANDIDATES
    )
    if claim_field is None or evidence_field is None:
        raise ValueError(
            "External eval record is missing required claim/evidence text: "
            f"record_id={record_id!r} "
            f"claim_field_found={claim_field!r} evidence_field_found={evidence_field!r} "
            f"candidate_claim_fields={list(_STAGE73_CLAIM_FIELD_CANDIDATES)} "
            f"candidate_evidence_fields={list(_STAGE73_EVIDENCE_FIELD_CANDIDATES)} "
            f"available_keys={available_keys}"
        )
    out["claim"] = str(claim_value)
    out["evidence"] = str(evidence_value)

    # --- auxiliary labels (external-eval-only; never used for training) ----
    aux_defaults = _STAGE73_AUX_LABEL_DEFAULTS_BY_FINAL_LABEL[canonical_label]
    aux_labels_added = False
    for aux_field, default_value in aux_defaults.items():
        if aux_field not in out or out[aux_field] is None:
            out[aux_field] = default_value
            aux_labels_added = True

    # --- identity / pairing fields required by v5.encode_mamba_records -----
    id_defaulted = "id" not in out or out["id"] is None
    if id_defaulted:
        out["id"] = record_id
    pair_id_defaulted = "pair_id" not in out or out["pair_id"] is None
    if pair_id_defaulted:
        out["pair_id"] = out["id"]
    intervention_type_defaulted = (
        "intervention_type" not in out or out["intervention_type"] is None
    )
    if intervention_type_defaulted:
        out["intervention_type"] = "stage43b1_external_factver"
    out.setdefault("normalized_intervention", out["intervention_type"])
    out.setdefault("primary_failure_type", "stage43b1_external_factver")
    out.setdefault("source_intervention_type", "")

    # --- provenance metadata (additive; does not break prediction export) --
    out["external_original_label"] = label_value
    out["external_schema_normalized"] = True
    out["external_schema_source"] = "stage43b1_factver"

    changed = (
        final_label_was_missing
        or label_field != "final_label"
        or claim_field != "claim"
        or evidence_field != "evidence"
        or aux_labels_added
        or id_defaulted
        or pair_id_defaulted
        or intervention_type_defaulted
    )
    stats = {
        "label_field_used": label_field,
        "claim_field_used": claim_field,
        "evidence_field_used": evidence_field,
        "final_label_was_missing": final_label_was_missing,
        "aux_labels_added": aux_labels_added,
        "changed": changed,
    }
    return out, stats


def normalize_external_factver_records(
    records: list[dict[str, Any]],
) -> "tuple[list[dict[str, Any]], dict[str, Any]]":
    """Batch wrapper around normalize_external_factver_record.

    Returns (normalized_records, schema_report) where schema_report supplies
    the Stage73 external-eval reporting fields (requirement 9):
    external_schema_normalized, external_schema_normalization_source,
    external_schema_missing_final_label_fixed, external_schema_label_field_used,
    external_schema_records_normalized, external_schema_records_with_added_aux_labels.
    """
    normalized: list[dict[str, Any]] = []
    label_fields_used: set[str] = set()
    records_normalized = 0
    records_with_added_aux_labels = 0
    missing_final_label_fixed = False

    for index, record in enumerate(records):
        norm_record, rec_stats = normalize_external_factver_record(record, index)
        normalized.append(norm_record)
        if rec_stats["label_field_used"] is not None:
            label_fields_used.add(rec_stats["label_field_used"])
        if rec_stats["changed"]:
            records_normalized += 1
        if rec_stats["aux_labels_added"]:
            records_with_added_aux_labels += 1
        if rec_stats["final_label_was_missing"]:
            missing_final_label_fixed = True

    schema_report = {
        "external_schema_normalized": True,
        "external_schema_normalization_source": "runner_external_eval",
        "external_schema_missing_final_label_fixed": missing_final_label_fixed,
        "external_schema_label_field_used": sorted(label_fields_used),
        "external_schema_records_normalized": records_normalized,
        "external_schema_records_with_added_aux_labels": records_with_added_aux_labels,
    }
    return normalized, schema_report


def evaluate_external_probe(
    model: "ContraMambaV6BMinimal | Any",
    probe_records: list[dict],
    probe_inputs: dict[str, "torch.Tensor"],
    flag_source: str,
    device: "torch.device",
    eval_name: str,
    eval_path: str,
    args: "argparse.Namespace",
) -> "tuple[dict[str, Any], list[dict]]":
    """Eval-only forward pass on an external probe dataset.

    No training, no checkpoint selection, no calibration.
    Returns (metrics_dict, prediction_records) using the stage28e_v1 schema.

    Label handling:
      - Final labels REFUTE / NOT_ENTITLED / SUPPORT are recognised when present.
      - Per-label F1 is computed over the full known label space (v5.FinalLabel);
        absent classes contribute 0.0 to their F1 slot but do NOT crash.
      - Macro F1 averages over the full known label space for consistency.
    """
    probe_temporal_flags, probe_predicate_flags = extract_flags(
        probe_records, flag_source, device
    )

    model.eval()
    with torch.no_grad():
        output = model(
            **_vnext_model_feature_inputs(probe_inputs),
            temporal_mismatch_flags=probe_temporal_flags,
            predicate_mismatch_flags=probe_predicate_flags,
        )

    prediction_recs = prediction_records_v6b(
        probe_records,
        output,
        stage32_owner_state_export=getattr(args, "stage32_owner_state_export", False),
        stage32_owner_state_shadow_mode=getattr(
            args, "stage32_owner_state_shadow_mode", False
        ),
        stage32_coverage_owner_v2=getattr(args, "stage32_coverage_owner_v2", False),
        stage32_coverage_owner_v2_min_confidence=getattr(
            args, "stage32_coverage_owner_v2_min_confidence", 0.50
        ),
        stage32_coverage_owner_v2_min_margin=getattr(
            args, "stage32_coverage_owner_v2_min_margin", 0.05
        ),
        stage32_coverage_owner_v2_allow_abstain=getattr(
            args, "stage32_coverage_owner_v2_allow_abstain", False
        ),
        stage33_structured_coverage_owner=getattr(
            args, "stage33_use_structured_coverage_owner", False
        ),
        stage33_structured_coverage_owner_export=getattr(
            args, "stage33_structured_coverage_owner_export", False
        ),
        stage33_structured_coverage_owner_shadow_mode=getattr(
            args, "stage33_structured_coverage_owner_shadow_mode", False
        ),
        stage33_structured_coverage_preserve_can_support=getattr(
            args, "stage33_structured_coverage_preserve_can_support", False
        ),
        stage33_structured_coverage_direct_support_rules=getattr(
            args,
            "stage33_structured_coverage_direct_support_rules",
            _STAGE33_DEFAULT_DIRECT_SUPPORT_RULES,
        ),
        stage33_structured_coverage_disable_specific_general_direct_support=getattr(
            args,
            "stage33_structured_coverage_disable_specific_general_direct_support",
            False,
        ),
        stage33_structured_coverage_weak_rules_to_residual=getattr(
            args, "stage33_structured_coverage_weak_rules_to_residual", ""
        ),
        stage33_structured_coverage_conditional_fallback=getattr(
            args, "stage33_structured_coverage_conditional_fallback", False
        ),
        stage33_structured_coverage_fallback_source=getattr(
            args, "stage33_structured_coverage_fallback_source", "current_final"
        ),
        stage33_structured_coverage_enable_whole_part_rules=getattr(
            args, "stage33_structured_coverage_enable_whole_part_rules", False
        ),
        stage33_structured_coverage_whole_part_direct_support=getattr(
            args, "stage33_structured_coverage_whole_part_direct_support", False
        ),
        stage33_structured_coverage_whole_part_lexicon=getattr(
            args, "stage33_structured_coverage_whole_part_lexicon", ""
        ),
        stage33_structured_coverage_whole_part_v2=getattr(
            args, "stage33_structured_coverage_whole_part_v2", False
        ),
        stage33_structured_coverage_whole_part_v2_use_expanded_lexicon=getattr(
            args,
            "stage33_structured_coverage_whole_part_v2_use_expanded_lexicon",
            False,
        ),
        stage33_structured_coverage_whole_part_v2_direct_support_policy=getattr(
            args,
            "stage33_structured_coverage_whole_part_v2_direct_support_policy",
            "hard_core_required",
        ),
        stage33_whole_part_conditional_safe_overrides_hard_core=getattr(
            args, "stage33_whole_part_conditional_safe_overrides_hard_core", False
        ),
        stage36_support_safety_config=_stage36_config_from_args(args),
        stage37_safe_support_recovery_config=_stage37_config_from_args(args),
        args=args,
    )

    predictions_cpu = output["predictions"].detach().cpu()
    labels_cpu = probe_inputs["final_labels"].detach().cpu()
    n = len(probe_records)

    # Label counts (from records, not from tensors, to preserve original string values)
    label_counts: dict[str, int] = {}
    for r in probe_records:
        lbl = str(r.get("final_label", "UNKNOWN"))
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # Intervention counts
    intervention_counts: dict[str, int] = {}
    for r in probe_records:
        it = str(r.get("intervention_type", "unknown"))
        intervention_counts[it] = intervention_counts.get(it, 0) + 1

    # Prediction distribution
    pred_dist: dict[str, int] = {}
    for pred_id in predictions_cpu.tolist():
        name = v5.ID_TO_FINAL_LABEL[pred_id]
        pred_dist[name] = pred_dist.get(name, 0) + 1

    # Accuracy
    accuracy = float((predictions_cpu == labels_cpu).float().mean().item())

    # Per-label metrics and macro F1 over known label space (safe for absent labels)
    f1_vals: list[float] = []
    per_label: dict[str, Any] = {}
    for label in v5.FinalLabel:
        lid = int(label)
        label_name = v5.ID_TO_FINAL_LABEL[lid]
        pred = predictions_cpu == lid
        actual = labels_cpu == lid
        tp = float((pred & actual).sum().item())
        pd_ = float(pred.sum().item())
        rd = float(actual.sum().item())
        prec = tp / pd_ if pd_ else 0.0
        rec = tp / rd if rd else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1_vals.append(f1)
        per_label[label_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(rd),
            "predicted": int(pd_),
        }
    macro_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0

    # False SUPPORT metrics
    false_support_total = 0
    false_support_by_intervention: dict[str, int] = {}
    false_support_by_axis: dict[str, int] = {}
    true_support_correct = 0

    for i, r in enumerate(probe_records):
        pred_id = int(predictions_cpu[i].item())
        pred_label = v5.ID_TO_FINAL_LABEL[pred_id]
        gold_label = _s28e_normalize_label(r.get("final_label"))
        it = str(r.get("intervention_type", "unknown"))
        axis = _S28E_DIAGNOSTIC_AXIS.get(it)

        if gold_label is not None and gold_label != "SUPPORT" and pred_label == "SUPPORT":
            false_support_total += 1
            false_support_by_intervention[it] = false_support_by_intervention.get(it, 0) + 1
            if axis:
                false_support_by_axis[axis] = false_support_by_axis.get(axis, 0) + 1

        if gold_label == "SUPPORT" and pred_label == "SUPPORT":
            true_support_correct += 1

    config_summary = {
        "v7_use_location_boundary_head": getattr(args, "v7_use_location_boundary_head", False),
        "v7_use_location_boundary_loss": getattr(args, "v7_use_location_boundary_loss", False),
        "v7_location_boundary_cap_mode": getattr(args, "v7_location_boundary_cap_mode", "none"),
        "v7_location_boundary_cap_gamma": getattr(args, "v7_location_boundary_cap_gamma", 1.0),
        "v7_location_boundary_cap_detach": getattr(args, "v7_location_boundary_cap_detach", False),
        "v7_use_temporal_safety_head": getattr(args, "v7_use_temporal_safety_head", False),
        "v7_use_temporal_safety_loss": getattr(args, "v7_use_temporal_safety_loss", False),
        "v7_temporal_safety_cap_mode": getattr(args, "v7_temporal_safety_cap_mode", "none"),
        "v7_temporal_safety_cap_gamma": getattr(args, "v7_temporal_safety_cap_gamma", 1.0),
        "v7_temporal_safety_cap_detach": getattr(args, "v7_temporal_safety_cap_detach", False),
        "v7_use_temporal_mismatch_multihead": getattr(
            args, "v7_use_temporal_mismatch_multihead", False
        ),
        "v7_use_temporal_mismatch_multihead_loss": getattr(
            args, "v7_use_temporal_mismatch_multihead_loss", False
        ),
        "v7_temporal_mismatch_multihead_cap_mode": getattr(
            args, "v7_temporal_mismatch_multihead_cap_mode", "none"
        ),
        "v7_temporal_mismatch_multihead_fusion": getattr(
            args, "v7_temporal_mismatch_multihead_fusion", "frame_only"
        ),
        "v7_h1_entitlement_decision_signal": getattr(
            args, "v7_h1_entitlement_decision_signal", None
        ),
        "v7_use_v6b_style_final_decision": getattr(
            args, "v7_use_v6b_style_final_decision", False
        ),
        "architecture": getattr(args, "architecture", "v6b_minimal"),
        "flag_source": getattr(args, "flag_source", "controlled_heuristic"),
    }

    result: dict[str, Any] = {
        "external_eval_name": eval_name,
        "external_eval_path": eval_path,
        "n_records": n,
        "label_counts": label_counts,
        "intervention_counts": dict(sorted(intervention_counts.items())),
        "prediction_distribution": dict(sorted(pred_dist.items())),
        "final_accuracy": round(accuracy, 4),
        "final_macro_f1": round(macro_f1, 4),
        "per_label": per_label,
        "false_SUPPORT_total": false_support_total,
        "false_SUPPORT_by_intervention": dict(sorted(false_support_by_intervention.items())),
        "false_SUPPORT_by_axis": dict(sorted(false_support_by_axis.items())),
        "true_SUPPORT_correct": true_support_correct,
        "config_summary": config_summary,
        # Provenance guards
        "stage15_used_for_training": False,
        "stage15_used_for_checkpoint_selection": False,
        "external_probe_used_for_checkpoint_selection": False,
        "external_probe_used_for_calibration": False,
        "external_probe_used_for_training": False,
    }

    return result, prediction_recs


# ---------------------------------------------------------------------------
# Stage43-C0: post-training external fact-verification evaluation hook
# ---------------------------------------------------------------------------


def _stage43_factver_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    """Return an eval-only args clone with Stage39-C composer export enabled.

    This does not mutate the training args and does not alter logits, losses,
    training data, checkpoint selection, or model weights. It only requests the
    same export-time Stage39-C safe_structured_v2 diagnostics used by the normal
    prediction export path.
    """
    eval_args = argparse.Namespace(**vars(args))
    shadow_export = bool(getattr(args, "stage43_external_enable_shadow_export", False))
    if shadow_export:
        eval_args.stage32_owner_state_export = True
        eval_args.stage32_owner_state_shadow_mode = True
        eval_args.stage33_use_structured_coverage_owner = True
        eval_args.stage33_structured_coverage_owner_export = True
        eval_args.stage33_structured_coverage_owner_shadow_mode = True
        eval_args.stage33_structured_coverage_conditional_fallback = True
        eval_args.stage36_use_support_safety_blockers = True
        eval_args.stage36_support_safety_export = True
        eval_args.stage36_support_safety_shadow_mode = True
        eval_args.stage37_use_safe_support_recovery = True
        eval_args.stage37_safe_support_export = True
        eval_args.stage37_safe_support_shadow_mode = True
    eval_args.stage39_use_final_composer_opt_in = True
    eval_args.stage39_final_composer_export = True
    eval_args.stage39_final_composer_policy = "safe_structured_v2"
    eval_args.stage39_final_composer_output_mode = "export_only"
    if not getattr(eval_args, "stage39_final_composer_source", None):
        eval_args.stage39_final_composer_source = "stage37_final_shadow_label"
    eval_args.stage43c2_shadow_export_enabled = shadow_export
    return eval_args


def _stage43_slice_inputs(inputs: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    return {key: value[start:end] for key, value in inputs.items()}


def _stage43_input_path_diagnostics(args: argparse.Namespace, max_length: int) -> dict[str, Any]:
    if args.backbone == "dummy":
        template = "v5.encode_records: lowercase tokenized claim + <sep> + lowercase tokenized evidence; claim/evidence masks over their spans; padded/truncated to training max_length"
    else:
        template = "v5.encode_mamba_records: tokenizer claim span truncated to floor((max_length - 1) / 2), separator token, tokenizer evidence span truncated to remaining budget; exact claim/evidence masks"
    return {
        "external_input_template": template,
        "controlled_dev_input_template": template,
        "external_uses_same_prediction_path_as_dev": True,
        "external_uses_same_label_mapping_as_dev": True,
        "label_id_to_name": {str(key): value for key, value in sorted(v5.ID_TO_FINAL_LABEL.items())},
        "name_to_label_id": dict(sorted(v5.FINAL_LABEL_TO_ID.items())),
        "model_forward_path": "model(**_vnext_model_feature_inputs(inputs), temporal_mismatch_flags=extract_flags(...), predicate_mismatch_flags=extract_flags(...))",
        "prediction_export_path": "prediction_records_v6b with Stage39 export-only composer diagnostics",
        "tokenizer_source": "dummy_vocab" if args.backbone == "dummy" else getattr(args, "model_name", None),
        "max_length": max_length,
        "stage43c2_shadow_export_enabled": bool(getattr(args, "stage43_external_enable_shadow_export", False)),
        "stage43c2_reused_internal_export_path": "prediction_records_v6b -> build_stage32_owner_state -> compute_stage36_support_safety_blocker -> compute_stage37_safe_support_recovery -> compute_stage39_final_composer",
        "stage43c2_forced_eval_only_exports": [
            "stage32_owner_state_export",
            "stage32_owner_state_shadow_mode",
            "stage33_use_structured_coverage_owner",
            "stage33_structured_coverage_owner_export",
            "stage33_structured_coverage_owner_shadow_mode",
            "stage33_structured_coverage_conditional_fallback",
            "stage36_use_support_safety_blockers",
            "stage36_support_safety_export",
            "stage36_support_safety_shadow_mode",
            "stage37_use_safe_support_recovery",
            "stage37_safe_support_export",
            "stage37_safe_support_shadow_mode",
            "stage39_use_final_composer_opt_in",
            "stage39_final_composer_export",
            "stage39_final_composer_policy=safe_structured_v2",
            "stage39_final_composer_output_mode=export_only",
        ] if bool(getattr(args, "stage43_external_enable_shadow_export", False)) else [],
    }


def _stage43_token_diagnostics(
    records: list[dict],
    *,
    args: argparse.Namespace,
    tokenizer: Any | None,
    max_length: int,
) -> dict[str, Any]:
    lengths: list[int] = []
    truncation_count = 0
    claim_budget = max(1, (int(args.max_length) - 1) // 2) if args.backbone != "dummy" else None
    evidence_budget = int(args.max_length) - int(claim_budget) - 1 if claim_budget is not None else None
    for record in records:
        claim = str(record.get("claim") or "")
        evidence = str(record.get("evidence") or "")
        if args.backbone == "dummy":
            claim_len = len(v5.tokenize(claim))
            evidence_len = len(v5.tokenize(evidence))
            pair_len = claim_len + 1 + evidence_len
            truncated = pair_len > max_length
        else:
            if tokenizer is None:
                claim_len = 0
                evidence_len = 0
            else:
                claim_len = len(tokenizer.encode(claim, add_special_tokens=False, truncation=False))
                evidence_len = len(tokenizer.encode(evidence, add_special_tokens=False, truncation=False))
            pair_len = claim_len + 1 + evidence_len
            truncated = (
                claim_budget is not None
                and evidence_budget is not None
                and (claim_len > claim_budget or evidence_len > evidence_budget)
            )
        lengths.append(pair_len)
        if truncated:
            truncation_count += 1
    row_count = len(records)
    return {
        "external_token_length_summary": summarize_numeric(lengths),
        "external_truncation_count": truncation_count,
        "external_truncation_rate": round(truncation_count / row_count, 6) if row_count else None,
        "token_length_definition": "untruncated claim token count + separator + untruncated evidence token count",
        "claim_budget": claim_budget,
        "evidence_budget": evidence_budget,
    }


def _stage43_enrich_prediction_confidence(prediction_rows: list[dict]) -> None:
    for row in prediction_rows:
        probs = row.get("final_probs")
        if not isinstance(probs, list) or not probs:
            continue
        clean_probs = [float(value) for value in probs if isinstance(value, (int, float))]
        if not clean_probs:
            continue
        max_prob = max(clean_probs)
        entropy = 0.0
        for prob in clean_probs:
            if prob > 0.0:
                entropy -= prob * float(np.log(prob))
        row["stage43c1_max_probability"] = round(max_prob, 6)
        row["stage43c1_prediction_entropy"] = round(entropy, 6)

def _stage43_prediction_records_from_model(
    model: "ContraMambaV6BMinimal | Any",
    records: list[dict],
    model_inputs: dict[str, torch.Tensor],
    flag_source: str,
    device: torch.device,
    args: argparse.Namespace,
    batch_size: int,
) -> list[dict]:
    """Eval-only Stage43 forward/export loop over an already-restored model."""
    eval_args = _stage43_factver_eval_args(args)
    batch_size = max(1, int(batch_size))
    exported: list[dict] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            end = min(len(records), start + batch_size)
            batch_records = records[start:end]
            batch_inputs = _stage43_slice_inputs(model_inputs, start, end)
            temporal_flags, predicate_flags = extract_flags(batch_records, flag_source, device)
            feature_inputs = _vnext_model_feature_inputs(batch_inputs)
            _assert_model_accepts_feature_kwargs(
                model, feature_inputs, context="Stage118/Stage43 diagnostic export"
            )
            output = model(
                **feature_inputs,
                temporal_mismatch_flags=temporal_flags,
                predicate_mismatch_flags=predicate_flags,
            )
            exported.extend(
                prediction_records_v6b(
                    batch_records,
                    output,
                    stage32_owner_state_export=getattr(eval_args, "stage32_owner_state_export", False),
                    stage32_owner_state_shadow_mode=getattr(eval_args, "stage32_owner_state_shadow_mode", False),
                    stage32_coverage_owner_v2=getattr(eval_args, "stage32_coverage_owner_v2", False),
                    stage32_coverage_owner_v2_min_confidence=getattr(eval_args, "stage32_coverage_owner_v2_min_confidence", 0.50),
                    stage32_coverage_owner_v2_min_margin=getattr(eval_args, "stage32_coverage_owner_v2_min_margin", 0.05),
                    stage32_coverage_owner_v2_allow_abstain=getattr(eval_args, "stage32_coverage_owner_v2_allow_abstain", False),
                    stage33_structured_coverage_owner=getattr(eval_args, "stage33_use_structured_coverage_owner", False),
                    stage33_structured_coverage_owner_export=getattr(eval_args, "stage33_structured_coverage_owner_export", False),
                    stage33_structured_coverage_owner_shadow_mode=getattr(eval_args, "stage33_structured_coverage_owner_shadow_mode", False),
                    stage33_structured_coverage_preserve_can_support=getattr(eval_args, "stage33_structured_coverage_preserve_can_support", False),
                    stage33_structured_coverage_direct_support_rules=getattr(eval_args, "stage33_structured_coverage_direct_support_rules", _STAGE33_DEFAULT_DIRECT_SUPPORT_RULES),
                    stage33_structured_coverage_disable_specific_general_direct_support=getattr(eval_args, "stage33_structured_coverage_disable_specific_general_direct_support", False),
                    stage33_structured_coverage_weak_rules_to_residual=getattr(eval_args, "stage33_structured_coverage_weak_rules_to_residual", ""),
                    stage33_structured_coverage_conditional_fallback=getattr(eval_args, "stage33_structured_coverage_conditional_fallback", False),
                    stage33_structured_coverage_fallback_source=getattr(eval_args, "stage33_structured_coverage_fallback_source", "current_final"),
                    stage33_structured_coverage_enable_whole_part_rules=getattr(eval_args, "stage33_structured_coverage_enable_whole_part_rules", False),
                    stage33_structured_coverage_whole_part_direct_support=getattr(eval_args, "stage33_structured_coverage_whole_part_direct_support", False),
                    stage33_structured_coverage_whole_part_lexicon=getattr(eval_args, "stage33_structured_coverage_whole_part_lexicon", ""),
                    stage33_structured_coverage_whole_part_v2=getattr(eval_args, "stage33_structured_coverage_whole_part_v2", False),
                    stage33_structured_coverage_whole_part_v2_use_expanded_lexicon=getattr(eval_args, "stage33_structured_coverage_whole_part_v2_use_expanded_lexicon", False),
                    stage33_structured_coverage_whole_part_v2_direct_support_policy=getattr(eval_args, "stage33_structured_coverage_whole_part_v2_direct_support_policy", "hard_core_required"),
                    stage33_whole_part_conditional_safe_overrides_hard_core=getattr(eval_args, "stage33_whole_part_conditional_safe_overrides_hard_core", False),
                    stage36_support_safety_config=_stage36_config_from_args(eval_args),
                    stage37_safe_support_recovery_config=_stage37_config_from_args(eval_args),
                    args=eval_args,
                )
            )
    return exported


def _stage118_normalize_label(value: Any) -> str | None:
    if isinstance(value, str):
        compact = value.strip().upper().replace(" ", "_")
        normalized = _s28e_normalize_label(compact)
        if normalized is not None:
            return normalized
        if compact == "NEI":
            return "NOT_ENTITLED"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if int(value) == value:
            return _s28e_normalize_label(int(value))
    return _s28e_normalize_label(value)


def load_stage118_diagnostic_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    records: list[dict[str, Any]] = []
    skip_reasons: dict[str, int] = {}
    n_input_rows = 0

    def skip(reason: str) -> None:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    with open(path, encoding="utf-8") as fh:
        for line_index, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            n_input_rows += 1
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                skip("invalid_json")
                continue
            if not isinstance(row, dict):
                skip("not_object")
                continue
            claim = row.get("claim")
            evidence = row.get("evidence")
            if claim is None or evidence is None:
                skip("missing_claim_or_evidence")
                continue
            raw_label = row.get("gold_label", row.get("label"))
            gold_label = _stage118_normalize_label(raw_label)
            if gold_label is None:
                skip("invalid_or_missing_label")
                continue
            record = dict(row)
            record["claim"] = str(claim)
            record["evidence"] = str(evidence)
            record["gold_label"] = gold_label
            record["final_label"] = gold_label
            record.setdefault("id", row.get("id", f"stage118_row_{line_index}"))
            record.setdefault("intervention_type", row.get("intervention_type", "stage118_generic_diagnostic"))
            record["stage118_valid_row_index"] = len(records)
            records.append(record)
    return records, dict(sorted(skip_reasons.items())), n_input_rows


def _stage118_encode_inputs(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    vocab: dict[str, int] | None,
    tokenizer: Any | None,
    max_length: int,
    device: torch.device,
    evidence_interface: str | None = None,
) -> dict[str, torch.Tensor]:
    if args.backbone == "dummy":
        if vocab is None:
            raise ValueError("Stage118 diagnostic eval with dummy backbone requires vocab.")
        bundle = v5.encode_records(records, vocab)
    else:
        if tokenizer is None:
            raise ValueError("Stage118 diagnostic eval with Mamba backbone requires tokenizer.")
        bundle = v5.encode_mamba_records(records, tokenizer, args.max_length)
    inputs = v5.move_inputs(bundle["model_inputs"], device)
    attach_vnext_segmented_dual_pass_inputs(
        inputs,
        records,
        args=args,
        vocab=vocab,
        tokenizer=tokenizer,
        device=device,
        evidence_interface=evidence_interface,
    )
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < max_length:
        for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
            inputs[key] = F.pad(inputs[key], (0, max_length - seq_len), value=0)
    elif seq_len > max_length:
        for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
            inputs[key] = inputs[key][:, :max_length]
    return inputs


def _stage118_preserved_metadata_fields(record: dict[str, Any]) -> dict[str, Any]:
    preserved: dict[str, Any] = {}
    for key, value in record.items():
        if key in _STAGE118_CORE_PREDICTION_FIELDS:
            continue
        if key.startswith(_STAGE118_PRESERVED_FIELD_PREFIXES):
            preserved[key] = value
    return preserved

def _stage118_prediction_rows(
    records: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    diagnostic_name: str,
    source_jsonl: Path,
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for record, pred in zip(records, prediction_rows):
        prediction = pred.get("pred_final_label") or pred.get("pred_label")
        base_prediction = pred.get(
            "stage128_prediction_before_location_guard",
            pred.get("base_prediction", prediction),
        )
        final_prediction = pred.get(
            "stage128_prediction_after_location_guard",
            pred.get("prediction", prediction),
        )
        row: dict[str, Any] = {
            "id": record.get("id"),
            "claim": record.get("claim"),
            "evidence": record.get("evidence"),
            "gold_label": record.get("gold_label", record.get("final_label")),
            "base_prediction": base_prediction,
            "prediction": final_prediction,
            "stage118_diagnostic_name": diagnostic_name,
            "stage118_source_jsonl": str(source_jsonl),
            "stage118_valid_row_index": record.get("stage118_valid_row_index"),
        }
        for key, value in _stage118_preserved_metadata_fields(record).items():
            row.setdefault(key, value)
        if record.get("source_id") is None and pred.get("source_id") is not None:
            row["source_id"] = pred.get("source_id")
        if pred.get("final_logits") is not None:
            row["logits"] = pred.get("final_logits")
        for key in _STAGE113_VNEXT_EXPORT_FIELDS:
            if key in pred:
                row[key] = pred[key]
        for key in _STAGE123_VNEXT_EVIDENCE_EXPORT_FIELDS:
            if key in pred:
                row[key] = pred[key]
            if key in record:
                row[key] = record[key]
        for key in _STAGE128_LOCATION_SLOT_GUARD_EXPORT_FIELDS:
            if key in pred:
                row[key] = pred[key]
        _stage125_assert_risk_cap_exports(row)
        exported.append(row)
    return exported


def _stage118_scalar_medians(rows: list[dict[str, Any]]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for key in _STAGE113_VNEXT_SCALAR_FIELDS:
        values = [row.get(key) for row in rows]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        if numeric:
            medians[key] = round(float(np.median(numeric)), 6)
    return medians


def _stage118_build_summary(
    *,
    diagnostic_name: str,
    input_jsonl: Path,
    output_jsonl: Path,
    n_input_rows: int,
    skip_reasons: dict[str, int],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
    pred_counts = {label: 0 for label in labels}
    gold_counts = {label: 0 for label in labels}
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get("prediction")
        if gold in gold_counts:
            gold_counts[gold] += 1
        if pred in pred_counts:
            pred_counts[pred] += 1

    per_label: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for row in rows if row.get("gold_label") == label and row.get("prediction") == label)
        pred_n = pred_counts[label]
        gold_n = gold_counts[label]
        precision = tp / pred_n if pred_n else 0.0
        recall = tp / gold_n if gold_n else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(gold_n),
        }

    n_valid = len(rows)
    correct = sum(1 for row in rows if row.get("gold_label") == row.get("prediction"))
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in _STAGE118_BUCKETS}
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get("prediction")
        if gold == pred == "SUPPORT":
            buckets["correct_SUPPORT"].append(row)
        if gold == pred == "REFUTE":
            buckets["correct_REFUTE"].append(row)
        if gold == pred == "NOT_ENTITLED":
            buckets["correct_NE"].append(row)
        if gold == "SUPPORT" and pred == "NOT_ENTITLED":
            buckets["false_NE_SUPPORT"].append(row)
        if gold == "REFUTE" and pred == "NOT_ENTITLED":
            buckets["false_NE_REFUTE"].append(row)
        if gold == "NOT_ENTITLED" and pred in {"SUPPORT", "REFUTE"}:
            buckets["false_entitlement"].append(row)
        if gold == "SUPPORT" and pred == "REFUTE":
            buckets["SUPPORT_to_REFUTE"].append(row)
        if gold == "REFUTE" and pred == "SUPPORT":
            buckets["REFUTE_to_SUPPORT"].append(row)

    scalar_medians_by_gold_label = {
        label: _stage118_scalar_medians([row for row in rows if row.get("gold_label") == label])
        for label in labels
    }
    scalar_medians_by_bucket = {
        bucket: _stage118_scalar_medians(bucket_rows)
        for bucket, bucket_rows in buckets.items()
    }

    return {
        "stage": "Stage118",
        "diagnostic_name": diagnostic_name,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "n_input_rows": int(n_input_rows),
        "n_valid_rows": int(n_valid),
        "n_skipped_rows": int(sum(skip_reasons.values())),
        "skip_reasons": skip_reasons,
        "accuracy": round(correct / n_valid, 4) if n_valid else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "per_label": per_label,
        "pred_counts": dict(sorted(pred_counts.items())),
        "gold_counts": dict(sorted(gold_counts.items())),
        "false_NE_total": int(len(buckets["false_NE_SUPPORT"]) + len(buckets["false_NE_REFUTE"])),
        "false_entitlement_total": int(len(buckets["false_entitlement"])),
        "polarity_error_total": int(len(buckets["SUPPORT_to_REFUTE"]) + len(buckets["REFUTE_to_SUPPORT"])),
        "scalar_medians_by_gold_label": scalar_medians_by_gold_label,
        "scalar_medians_by_bucket": scalar_medians_by_bucket,
        "decision": "STAGE118_GENERIC_DIAGNOSTIC_EVAL_COMPLETE",
    }


def run_stage118_generic_diagnostic_eval(
    *,
    model: "ContraMambaV6BMinimal | Any",
    input_jsonl: Path,
    output_jsonl: Path,
    summary_json: Path,
    diagnostic_name: str,
    batch_size: int,
    args: argparse.Namespace,
    vocab: dict[str, int] | None,
    tokenizer: Any | None,
    max_length: int,
    device: torch.device,
    diagnostic_evidence_interface_override: str | None = None,
) -> dict[str, Any]:
    records, skip_reasons, n_input_rows = load_stage118_diagnostic_jsonl(input_jsonl)
    stage118_evidence_interface = (
        diagnostic_evidence_interface_override
        if diagnostic_evidence_interface_override is not None
        else resolve_stage118_diagnostic_evidence_interface(args)
    )
    records = apply_vnext_evidence_interface_to_records(
        records, stage118_evidence_interface
    )
    for record in records:
        record["stage118_diagnostic_evidence_interface"] = (
            stage118_evidence_interface
            if diagnostic_evidence_interface_override is not None
            else getattr(args, "stage118_diagnostic_evidence_interface", "same_as_vnext")
        )
    inputs = _stage118_encode_inputs(
        records,
        args=args,
        vocab=vocab,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
        evidence_interface=stage118_evidence_interface,
    ) if records else {}
    prediction_rows = (
        _stage43_prediction_records_from_model(
            model=model,
            records=records,
            model_inputs=inputs,
            flag_source=args.flag_source,
            device=device,
            args=args,
            batch_size=batch_size,
        )
        if records else []
    )
    exported_rows = _stage118_prediction_rows(
        records,
        prediction_rows,
        diagnostic_name=diagnostic_name,
        source_jsonl=input_jsonl,
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, exported_rows)
    summary = _stage118_build_summary(
        diagnostic_name=diagnostic_name,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        n_input_rows=n_input_rows,
        skip_reasons=skip_reasons,
        rows=exported_rows,
    )
    write_json(summary_json, summary)
    return summary


def run_stage118_diagnostic_evidence_interface_sweep(
    *,
    model: "ContraMambaV6BMinimal | Any",
    input_jsonl: Path,
    output_dir: Path,
    diagnostic_name: str,
    batch_size: int,
    sweep_interfaces: list[str],
    args: argparse.Namespace,
    vocab: dict[str, int] | None,
    tokenizer: Any | None,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    prediction_paths: dict[str, str] = {}
    summary_paths: dict[str, str] = {}

    for evidence_interface in sweep_interfaces:
        output_jsonl = output_dir / f"stage118_diagnostic_{evidence_interface}_predictions.jsonl"
        summary_json = output_dir / f"stage118_diagnostic_{evidence_interface}_summary.json"
        summary = run_stage118_generic_diagnostic_eval(
            model=model,
            input_jsonl=input_jsonl,
            output_jsonl=output_jsonl,
            summary_json=summary_json,
            diagnostic_name=diagnostic_name,
            batch_size=batch_size,
            args=args,
            vocab=vocab,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
            diagnostic_evidence_interface_override=evidence_interface,
        )
        summaries[evidence_interface] = summary
        prediction_paths[evidence_interface] = str(output_jsonl)
        summary_paths[evidence_interface] = str(summary_json)

    manifest = {
        "stage": "Stage118",
        "sweep_interfaces": sweep_interfaces,
        "prediction_paths": prediction_paths,
        "summary_paths": summary_paths,
        "train_dev_evidence_interface": getattr(args, "vnext_evidence_interface", "full_evidence"),
        "diagnostic_source_jsonl_path": str(input_jsonl),
        "model_architecture": (
            "v7_hierarchical" if getattr(args, "architecture", None) == "v7_hierarchical" else "v6b_minimal"
        ),
        "architecture": getattr(args, "architecture", None),
        "router_mode": getattr(args, "vnext_router_mode", None),
        "seed": getattr(args, "seed", None),
    }
    manifest_path = output_dir / "stage118_diagnostic_sweep_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "summaries": summaries,
    }



_STAGE126_PREFLIGHT_REQUIRED_NON_NULL_FIELDS: tuple[str, ...] = (
    "vnext_context_risk",
    "vnext_logits_before_context_cap",
    "vnext_logits_after_context_cap",
    "vnext_prediction_before_context_cap",
    "vnext_prediction_after_context_cap",
    "vnext_context_only_logits",
    "vnext_context_only_prediction",
)



_STAGE126_PREFLIGHT_REQUIRED_SEGMENTED_VALUES: dict[str, Any] = {
    "stage118_diagnostic_evidence_interface": "segmented_dual_pass",
    "vnext_segmented_dual_pass_active": True,
    "vnext_segmented_context_role": "risk_cap",
    "vnext_primary_rep_source": "core_rep",
}


def _stage126_preflight_export_audit(
    rows: list[dict[str, Any]],
    *,
    evidence_interface: str,
) -> dict[str, Any]:
    fields = list(_STAGE126_PREFLIGHT_REQUIRED_NON_NULL_FIELDS)
    segmented_fields = list(_STAGE126_PREFLIGHT_REQUIRED_SEGMENTED_VALUES)
    field_counts: dict[str, dict[str, int]] = {}
    for key in fields + segmented_fields:
        present = sum(1 for row in rows if key in row)
        non_null = sum(1 for row in rows if row.get(key) is not None)
        field_counts[key] = {"present": int(present), "non_null": int(non_null)}
    segmented_value_mismatches: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for key, expected in _STAGE126_PREFLIGHT_REQUIRED_SEGMENTED_VALUES.items():
            actual = row.get(key)
            if actual != expected:
                segmented_value_mismatches.append(
                    {
                        "row_index": row_index,
                        "field": key,
                        "expected": expected,
                        "actual": actual,
                    }
                )
    sample_row = rows[0] if rows else {}
    return {
        "row_count": int(len(rows)),
        "evidence_interface_used": evidence_interface,
        "field_counts": field_counts,
        "required_fields_all_present_non_null": all(
            field_counts[key]["present"] == len(rows)
            and field_counts[key]["non_null"] == len(rows)
            for key in fields
        ) if rows else False,
        "segmented_values_all_match": not segmented_value_mismatches and bool(rows),
        "segmented_value_mismatches": segmented_value_mismatches[:20],
        "sample_vnext_context_risk": sample_row.get("vnext_context_risk"),
        "sample_vnext_context_only_prediction": sample_row.get(
            "vnext_context_only_prediction"
        ),
        "sample_stage118_diagnostic_evidence_interface": sample_row.get(
            "stage118_diagnostic_evidence_interface"
        ),
        "sample_vnext_segmented_dual_pass_active": sample_row.get(
            "vnext_segmented_dual_pass_active"
        ),
        "sample_vnext_segmented_context_role": sample_row.get(
            "vnext_segmented_context_role"
        ),
        "sample_vnext_primary_rep_source": sample_row.get(
            "vnext_primary_rep_source"
        ),
    }

def run_stage126_preflight_export_only(
    *,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    device: torch.device,
) -> dict[str, Any]:
    if args.architecture != "vnext_minimal":
        parser.error("--stage126-preflight-export-only requires --architecture vnext_minimal")
    if args.stage118_diagnostic_jsonl is None:
        parser.error("--stage126-preflight-export-only requires --stage118-diagnostic-jsonl")
    max_rows = int(args.stage126_preflight_max_rows)
    if max_rows <= 0:
        parser.error("--stage126-preflight-max-rows must be > 0")

    output_dir = args.stage126_preflight_output_dir or Path("results/stage126_preflight")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capped_input_jsonl = output_dir / "stage126_preflight_input.jsonl"
    output_jsonl = output_dir / "stage126_preflight_predictions.jsonl"
    summary_json = output_dir / "stage126_preflight_summary.json"
    report_json = output_dir / "stage126_preflight_report.json"

    records, skip_reasons, n_input_rows = load_stage118_diagnostic_jsonl(
        Path(args.stage118_diagnostic_jsonl)
    )
    capped_records = [dict(record) for record in records[:max_rows]]
    if not capped_records:
        raise RuntimeError(
            "Stage126 preflight found no valid diagnostic rows in "
            f"{args.stage118_diagnostic_jsonl}"
        )
    for record in capped_records:
        record["stage118_diagnostic_evidence_interface"] = "segmented_dual_pass"
        record["vnext_evidence_interface"] = "segmented_dual_pass"
        record["vnext_segmented_dual_pass_active"] = True
        record["vnext_segmented_context_role"] = "risk_cap"
        record["vnext_primary_rep_source"] = "core_rep"
    write_jsonl(capped_input_jsonl, capped_records)

    preflight_args = argparse.Namespace(**vars(args))
    preflight_args.architecture = "vnext_minimal"
    preflight_args.vnext_evidence_interface = "segmented_dual_pass"
    preflight_args.vnext_enable_segmented_dual_pass = True
    preflight_args.vnext_segmented_context_role = "risk_cap"
    preflight_args.stage118_diagnostic_evidence_interface = "segmented_dual_pass"
    preflight_args.stage118_diagnostic_evidence_interface_sweep = ""
    preflight_args.stage118_diagnostic_evidence_interface_sweep_list = []

    vocab: dict[str, int] | None = None
    tokenizer: Any | None = None
    if args.backbone == "dummy":
        vocab_records = apply_vnext_evidence_interface_to_records(
            [dict(record) for record in capped_records], "segmented_dual_pass"
        )
        vocab = v5.build_vocab(vocab_records)
        model = build_vnext_model(
            len(vocab),
            args.max_length,
            vnext_router_mode=args.vnext_router_mode,
            vnext_enable_segmented_dual_pass=True,
            vnext_segmented_context_role="risk_cap",
            vnext_context_risk_cap_alpha=args.vnext_context_risk_cap_alpha,
            vnext_context_risk_threshold=args.vnext_context_risk_threshold,
            vnext_context_risk_source=args.vnext_context_risk_source,
        )
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Mamba tokenizer has neither pad_token nor eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        model = build_vnext_mamba_model(
            args.model_name,
            freeze_encoder=args.freeze_encoder,
            freeze_a_log=args.freeze_a_log,
            vnext_router_mode=args.vnext_router_mode,
            vnext_enable_segmented_dual_pass=True,
            vnext_segmented_context_role="risk_cap",
            vnext_context_risk_cap_alpha=args.vnext_context_risk_cap_alpha,
            vnext_context_risk_threshold=args.vnext_context_risk_threshold,
            vnext_context_risk_source=args.vnext_context_risk_source,
        )
    model.to(device)
    model.eval()

    summary = run_stage118_generic_diagnostic_eval(
        model=model,
        input_jsonl=capped_input_jsonl,
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        diagnostic_name="stage126_preflight",
        batch_size=max(1, min(int(args.stage118_diagnostic_batch_size), len(capped_records))),
        args=preflight_args,
        vocab=vocab,
        tokenizer=tokenizer,
        max_length=args.max_length,
        device=device,
        diagnostic_evidence_interface_override="segmented_dual_pass",
    )

    exported_rows: list[dict[str, Any]] = []
    with open(output_jsonl, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                exported_rows.append(json.loads(stripped))
    export_audit = _stage126_preflight_export_audit(
        exported_rows, evidence_interface="segmented_dual_pass"
    )

    report = {
        "stage": "Stage126",
        "decision": "STAGE126_PREFLIGHT_EXPORT_COMPLETE",
        "input_jsonl": str(args.stage118_diagnostic_jsonl),
        "capped_input_jsonl": str(capped_input_jsonl),
        "output_jsonl": str(output_jsonl),
        "prediction_path": str(output_jsonl),
        "summary_json": str(summary_json),
        "n_input_rows": n_input_rows,
        "n_valid_rows_loaded": len(records),
        "n_preflight_rows": len(capped_records),
        "n_exported_rows": len(exported_rows),
        "skip_reasons": skip_reasons,
        "summary": summary,
        "required_non_null_fields": list(_STAGE126_PREFLIGHT_REQUIRED_NON_NULL_FIELDS),
        "required_segmented_values": dict(_STAGE126_PREFLIGHT_REQUIRED_SEGMENTED_VALUES),
        "export_audit": export_audit,
    }
    write_json(report_json, report)
    report["report_json"] = str(report_json)

    if not exported_rows:
        raise RuntimeError(
            "Stage126 preflight produced no prediction rows; report="
            f"{report_json}"
        )
    for row_index, row in enumerate(exported_rows):
        missing_or_null = [
            key for key in _STAGE126_PREFLIGHT_REQUIRED_NON_NULL_FIELDS
            if key not in row or row.get(key) is None
        ]
        segmented_mismatches = [
            f"{key}={row.get(key)!r} expected {expected!r}"
            for key, expected in _STAGE126_PREFLIGHT_REQUIRED_SEGMENTED_VALUES.items()
            if row.get(key) != expected
        ]
        if missing_or_null or segmented_mismatches:
            details: list[str] = []
            if missing_or_null:
                details.append("missing/null fields: " + ", ".join(missing_or_null))
            if segmented_mismatches:
                details.append("segmented path mismatches: " + "; ".join(segmented_mismatches))
            raise RuntimeError(
                f"Stage126 preflight row {row_index} did not exercise "
                "segmented_dual_pass+risk_cap export: "
                + " | ".join(details)
                + f"; report={report_json}"
            )
        _stage125_assert_risk_cap_exports(row)
    return report

def run_stage43_external_factver_hook(
    *,
    model: "ContraMambaV6BMinimal | Any",
    jsonl_paths: list[str],
    output_dir: Path,
    run_prefix: str,
    max_rows: int | None,
    batch_size: int,
    args: argparse.Namespace,
    vocab: dict[str, int] | None,
    tokenizer: Any | None,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run Stage43-C0 external fact-verification eval after best-state restore."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_reports: list[dict[str, Any]] = []
    for jsonl_path_str in jsonl_paths:
        jsonl_path = Path(jsonl_path_str)
        dataset_stem = jsonl_path.stem
        run_name = f"{run_prefix}_{dataset_stem}"
        report_json = output_dir / f"{run_prefix}_{dataset_stem}_external_factver_report.json"
        report_md = output_dir / f"{run_prefix}_{dataset_stem}_external_factver_report.md"
        pred_jsonl = output_dir / f"{run_prefix}_{dataset_stem}_external_factver_predictions.jsonl"

        rows = load_stage43_jsonl(jsonl_path, max_rows=max_rows)
        records = stage43_rows_to_controlled_records(rows)
        stage43_path_diagnostics = _stage43_input_path_diagnostics(args, max_length)
        stage43_token_diagnostics = _stage43_token_diagnostics(
            records,
            args=args,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        prediction_rows: list[dict] = []
        stage43_prediction_error: str | None = None
        if records:
            try:
                if args.backbone == "dummy":
                    if vocab is None:
                        raise ValueError(
                            "Stage43-C0 dummy external eval requires the dummy vocab."
                        )
                    bundle = v5.encode_records(records, vocab)
                else:
                    if tokenizer is None:
                        raise ValueError(
                            "Stage43-C0 Mamba external eval requires the Mamba tokenizer."
                        )
                    bundle = v5.encode_mamba_records(records, tokenizer, args.max_length)
                model_inputs = v5.move_inputs(bundle["model_inputs"], device)
                seq_len = model_inputs["input_ids"].shape[1]
                if seq_len < max_length:
                    for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                        model_inputs[key] = F.pad(model_inputs[key], (0, max_length - seq_len), value=0)
                elif seq_len > max_length:
                    for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                        model_inputs[key] = model_inputs[key][:, :max_length]
                prediction_rows = _stage43_prediction_records_from_model(
                    model=model,
                    records=records,
                    model_inputs=model_inputs,
                    flag_source=args.flag_source,
                    device=device,
                    args=args,
                    batch_size=batch_size,
                )
                _stage43_enrich_prediction_confidence(prediction_rows)
            except Exception as exc:
                stage43_prediction_error = f"{type(exc).__name__}: {exc}"
                prediction_rows = []

        report_payload, stage43_predictions = analyze_stage43_predictions(
            input_jsonl=jsonl_path,
            run_name=run_name,
            rows=rows,
            prediction_rows=prediction_rows,
            output_predictions_path=str(pred_jsonl),
            token_diagnostics=stage43_token_diagnostics,
            path_diagnostics=stage43_path_diagnostics,
        )
        _stage113_merge_prediction_exports(stage43_predictions, prediction_rows)
        report_payload.update(_stage113_vnext_scalar_report(prediction_rows))
        report_payload["output_report_json"] = str(report_json)
        report_payload["output_report_md"] = str(report_md)
        if stage43_prediction_error is not None:
            report_payload["decision"] = "STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE"
            report_payload.setdefault("sample_error_rows", []).insert(
                0,
                {
                    "error": "prediction_generation_failed",
                    "detail": stage43_prediction_error,
                },
            )
            report_payload.setdefault("risks", []).insert(0, stage43_prediction_error)
            report_payload["recommendation"] = (
                "Stage43-C0 model predictions could not be produced for this external file; "
                "treat the dataset evaluation as incomplete."
            )
        write_json(report_json, report_payload)
        write_text(report_md, render_report_markdown(report_payload))
        write_jsonl(pred_jsonl, stage43_predictions)
        dataset_reports.append(report_payload)
        print(
            f"[STAGE43-C0 EXTERNAL FACTVER] {dataset_stem} "
            f"decision={report_payload['decision']} rows={report_payload['row_count']} "
            f"base_f1={report_payload['base_macro_f1']} composed_f1={report_payload['composed_macro_f1']}"
        )

    aggregate = build_aggregate_report(run_prefix, dataset_reports)
    aggregate_json = output_dir / f"{run_prefix}_external_factver_aggregate_report.json"
    aggregate_md = output_dir / f"{run_prefix}_external_factver_aggregate_report.md"
    aggregate["output_report_json"] = str(aggregate_json)
    aggregate["output_report_md"] = str(aggregate_md)
    write_json(aggregate_json, aggregate)
    write_text(aggregate_md, render_aggregate_markdown(aggregate))
    return {
        "stage43_external_factver_reports": dataset_reports,
        "stage43_external_factver_aggregate": aggregate,
    }


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
    "v7_polarity_positive_energy",
    "v7_polarity_negative_energy",
    "v7_polarity_energy_margin",
    # Stage27-H2A: actual entitlement_for_decision used in H1 (None when H1 is inactive)
    "v7_h1_entitlement_for_decision",
    # Stage28-I-A: location boundary head (None when head is disabled)
    "v7_location_boundary_logit",
    "v7_location_boundary_prob",
    "v7_h1_entitlement_before_location_cap",
    "v7_h1_entitlement_after_location_cap",
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
        # Stage27-H2A: actual H1 decision signal (None when H1 inactive)
        "v7_h1_entitlement_for_decision",
        # Stage28-I-A: location boundary head (None when head is disabled)
        "v7_location_boundary_logit",
        "v7_location_boundary_prob",
        "v7_h1_entitlement_before_location_cap",
        "v7_h1_entitlement_after_location_cap",
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
    "v7_use_coverage_entailment_head",
    "v7_coverage_entailment_num_classes",
    "v7_coverage_entailment_input_mode",
    "v7_use_coverage_entailment_loss",
    "v7_coverage_entailment_loss_weight",
    "v7_coverage_entailment_data",
    "v7_coverage_entailment_detach_input",
    "stage31_probe_used_for_v7_coverage_entailment_loss",
    "v7_coverage_entailment_modifies_final_predictions",
    "stage32_use_owner_state_schema",
    "stage32_use_owner_interfaces",
    "stage32_owner_state_export",
    "stage32_owner_state_shadow_mode",
    "stage32_owner_state_modifies_final_logits",
    "stage32_owner_state_modifies_final_predictions",
    "stage32_coverage_owner_v2",
    "stage32_coverage_owner_v2_min_confidence",
    "stage32_coverage_owner_v2_min_margin",
    "stage32_coverage_owner_v2_allow_abstain",
    "stage33_use_structured_coverage_owner",
    "stage33_structured_coverage_owner_export",
    "stage33_structured_coverage_owner_shadow_mode",
    "stage33_structured_coverage_preserve_can_support",
    "stage33_structured_coverage_direct_support_rules",
    "stage33_structured_coverage_disable_specific_general_direct_support",
    "stage33_structured_coverage_weak_rules_to_residual",
    "stage33_structured_coverage_conditional_fallback",
    "stage33_structured_coverage_fallback_source",
    "stage33_structured_coverage_enable_whole_part_rules",
    "stage33_structured_coverage_whole_part_direct_support",
    "stage33_structured_coverage_whole_part_lexicon",
    "stage33_structured_coverage_whole_part_v2",
    "stage33_structured_coverage_whole_part_v2_use_expanded_lexicon",
    "stage33_structured_coverage_whole_part_v2_direct_support_policy",
    "stage33_whole_part_conditional_safe_overrides_hard_core",
    # Stage36-A: support-safety blocker flags (shadow-only)
    "stage36_use_support_safety_blockers",
    "stage36_support_safety_export",
    "stage36_support_safety_shadow_mode",
    "stage36_block_exception_scope",
    "stage36_block_not_all_existential",
    "stage36_block_location_scope_mismatch",
    "stage36_block_temporal_scope_mismatch",
    "stage36_support_blocker_action",
    # Stage37-A: safe SUPPORT recovery flags (shadow-only)
    "stage37_use_safe_support_recovery",
    "stage37_safe_support_export",
    "stage37_safe_support_shadow_mode",
    "stage37_recover_no_except_included_subset",
    "stage37_recover_coordination_universal_subset",
    "stage37_recover_numeric_universal_subset",
    "stage37_allow_recover_from_refute",
    # Stage39-A: opt-in final composer flags (prediction/export-time only)
    "stage39_use_final_composer_opt_in",
    "stage39_final_composer_export",
    "stage39_final_composer_policy",
    "stage39_final_composer_output_mode",
    "stage39_final_composer_source",
    "stage39_disallow_refute_to_support",
    "stage39_require_stage36_safety_clear",
    "stage39_require_stage37_not_from_refute",
    # v7 Stage15 / time_swap provenance (also lifted from audit_ledger elsewhere;
    # this covers the configuration copy when the ledger path is absent)
    "stage15_used_for_v7_training",
    "stage15_used_for_v7_selection",
    "stage15_used_for_v7_aux_loss_targets",
    "time_swap_used_in_v7_main_clean_data",
    "v7_use_v6b_style_final_decision",
    "v7_use_learnable_ne_alpha",
    "v7_ne_alpha_init",
    # Stage27-H2A/H2B/H2E: H1-path entitlement decision signal, product power, residual beta
    "v7_h1_entitlement_decision_signal",
    "v7_h1_entitlement_for_decision_source",
    "v7_h1_entitlement_product_power",
    "v7_h1_hybrid_residual_beta",
    # Stage28-I-A: location boundary head / cap configuration
    "v7_use_location_boundary_head",
    "v7_use_location_boundary_loss",
    "v7_location_boundary_loss_weight",
    "v7_location_boundary_cap_mode",
    "v7_location_boundary_cap_gamma",
    "v7_location_boundary_cap_detach",
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


# ---------------------------------------------------------------------------
# Stage60: optional train-only Stage57 non-leaking external bridge integration.
# Stage69/Stage70: extended with an analogous train-only Stage66 residual
# bridge integration path (see load_stage66_bridge_train_rows below), which
# shares the same forbidden-source guard.
#
# Policy (frozen by Stage59, extended by Stage69):
#   - Main clean data / dev split behavior is unchanged unless the user opts in
#     via --stage57-bridge-train-mode append_train_only and/or
#     --stage66-bridge-train-mode append_train_only.
#   - Bridge rows may only be appended to the train split, and only AFTER the
#     clean main train/dev split has already been created.
#   - Bridge rows never enter dev / checkpoint selection.
#   - No external datasets (VitaminC, Climate-FEVER, FEVEROUS), no
#     Stage43/53/55/63/65 outputs, and no time_swap data may be pulled in
#     through this path.
# ---------------------------------------------------------------------------
STAGE60_FORBIDDEN_SOURCE_TOKENS = (
    "vitaminc",
    "vitamin-c",
    "climate_fever",
    "climate-fever",
    "feverous",
    "stage43",
    "stage53",
    "stage55",
    "stage63",
    "stage65",
    "time_swap",
)
# This exact Stage57 metadata string is expected on every bridge row (it records
# that no VitaminC text/labels were used to build the bridge dataset) and must
# not itself be treated as a forbidden-source violation.
STAGE60_ALLOWED_METADATA_EXCEPTION = "no_vitaminc_text_or_labels_used"
STAGE60_BRIDGE_REQUIRED_FIELDS = ("id", "claim", "evidence", "label")


def _stage60_check_forbidden_source(value: str, context: str) -> None:
    """Hard fail if `value` references a prohibited external source/stage.

    The single allowed exception is the literal Stage57 leakage-policy string
    STAGE60_ALLOWED_METADATA_EXCEPTION, which itself contains "vitaminc" but is
    not a leak: it documents that VitaminC was NOT used.
    """
    if value == STAGE60_ALLOWED_METADATA_EXCEPTION:
        return
    lowered = value.lower()
    for token in STAGE60_FORBIDDEN_SOURCE_TOKENS:
        if token in lowered:
            raise ValueError(
                f"[stage60] forbidden external source token {token!r} found in "
                f"{context}: {value!r}. Stage60 bridge-train-only integration may not "
                "use VitaminC, Climate-FEVER, FEVEROUS, Stage43/53/55 outputs, or "
                "time_swap data."
            )


def load_stage57_bridge_train_rows(
    bridge_path: Path,
    existing_ids: set[str],
) -> tuple[list[dict], dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    """Load, validate, and normalize Stage57 bridge rows for train-only append.

    Returns (normalized_records, label_counts, family_counts, family_label_counts).
    Raises ValueError/FileNotFoundError on any Stage60 safety-check violation.
    """
    _stage60_check_forbidden_source(str(bridge_path), "--stage57-bridge-train-jsonl path")

    if not bridge_path.exists():
        raise FileNotFoundError(
            f"[stage60] --stage57-bridge-train-jsonl not found: {bridge_path}"
        )

    raw_rows: list[dict] = []
    with bridge_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"[stage60] invalid JSON on line {line_number} of {bridge_path}: {exc}"
                ) from exc

    if not raw_rows:
        raise ValueError(f"[stage60] --stage57-bridge-train-jsonl is empty: {bridge_path}")

    normalized: list[dict] = []
    seen_bridge_ids: set[str] = set()
    label_counts: dict[str, int] = {name: 0 for name in v5.ID_TO_FINAL_LABEL.values()}
    family_counts: dict[str, int] = {}
    family_label_counts: dict[str, dict[str, int]] = {}

    for row_number, row in enumerate(raw_rows, start=1):
        prefix = f"[stage60] row {row_number} in {bridge_path}: "
        missing = [
            field for field in STAGE60_BRIDGE_REQUIRED_FIELDS if row.get(field) is None
        ]
        if missing:
            raise ValueError(f"{prefix}missing required fields: {missing}")

        row_id = row["id"]
        if row_id in seen_bridge_ids:
            raise ValueError(f"{prefix}duplicate bridge id: {row_id!r}")
        seen_bridge_ids.add(row_id)
        if row_id in existing_ids:
            raise ValueError(
                f"{prefix}bridge id {row_id!r} duplicates an existing clean train/dev id"
            )

        label = row["label"]
        if isinstance(label, bool) or label not in (0, 1, 2):
            raise ValueError(f"{prefix}label must be 0, 1, or 2; got {label!r}")
        final_label = v5.ID_TO_FINAL_LABEL[int(label)]
        row_final_label = row.get("final_label")
        if row_final_label is not None and row_final_label != final_label:
            raise ValueError(
                f"{prefix}label={label!r} does not match final_label={row_final_label!r}"
            )

        for field_name in (
            "id", "pair_id", "claim", "evidence", "final_label",
            "primary_failure_type", "intervention_type",
            "stage57_family", "stage57_bridge_family", "stage57_subtype",
            "stage57_generation_source", "stage57_leakage_policy",
        ):
            field_value = row.get(field_name)
            if isinstance(field_value, str):
                _stage60_check_forbidden_source(
                    field_value, f"row {row_number} field {field_name!r}"
                )

        normalized.append({
            "id": row_id,
            "pair_id": row.get("pair_id", row_id),
            "claim": row["claim"],
            "evidence": row["evidence"],
            "final_label": final_label,
            "frame_compatible_label": row.get("frame_compatible_label", 1),
            "predicate_covered_label": row.get("predicate_covered_label", 1),
            "sufficiency_label": row.get("sufficiency_label", 1),
            "polarity_label": row.get(
                "polarity_label", "NONE" if final_label == "NOT_ENTITLED" else final_label
            ),
            "primary_failure_type": row.get("primary_failure_type", "none"),
            "intervention_type": row.get("intervention_type", "stage57_bridge"),
            "stage57_family": row.get("stage57_family"),
            "stage57_bridge_family": row.get("stage57_bridge_family"),
            "stage57_subtype": row.get("stage57_subtype"),
            "stage57_generation_source": row.get("stage57_generation_source"),
            "stage57_leakage_policy": row.get("stage57_leakage_policy"),
        })

        label_counts[final_label] += 1
        family = row.get("stage57_bridge_family") or row.get("stage57_family") or "unknown"
        family_counts[family] = family_counts.get(family, 0) + 1
        family_label_counts.setdefault(
            family, {name: 0 for name in v5.ID_TO_FINAL_LABEL.values()}
        )
        family_label_counts[family][final_label] += 1

    return normalized, label_counts, family_counts, family_label_counts


# ---------------------------------------------------------------------------
# Stage69/Stage70: optional train-only Stage66 residual bridge integration.
#
# Same policy as the Stage57 bridge above: bridge rows are appended to the
# train split only, strictly after the clean main train/dev split has been
# created, and never enter dev / checkpoint selection.
#
# Difference from Stage57: Stage66 rows carry stage66_* metadata (no
# Stage57-specific metadata required), and row-level forbidden-source
# scanning is restricted to id/pair_id/claim/evidence only. In particular,
# stage66_leakage_policy is NOT scanned, because its expected value legitimately
# contains "no_vitaminc_text_or_labels_used_taxonomy_only" (a declaration that
# VitaminC was NOT used, not a leak).
# ---------------------------------------------------------------------------
STAGE66_BRIDGE_REQUIRED_FIELDS = ("id", "claim", "evidence", "label")


def load_stage66_bridge_train_rows(
    bridge_path: Path,
    existing_ids: set[str],
) -> tuple[list[dict], dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    """Load, validate, and normalize Stage66 residual bridge rows for train-only append.

    Mirrors load_stage57_bridge_train_rows for the Stage66 residual bridge dataset.
    Returns (normalized_records, label_counts, family_counts, family_label_counts).
    Raises ValueError/FileNotFoundError on any safety-check violation.
    """
    _stage60_check_forbidden_source(str(bridge_path), "--stage66-bridge-train-jsonl path")

    if not bridge_path.exists():
        raise FileNotFoundError(
            f"[stage66] --stage66-bridge-train-jsonl not found: {bridge_path}"
        )

    raw_rows: list[dict] = []
    with bridge_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"[stage66] invalid JSON on line {line_number} of {bridge_path}: {exc}"
                ) from exc

    if not raw_rows:
        raise ValueError(f"[stage66] --stage66-bridge-train-jsonl is empty: {bridge_path}")

    normalized: list[dict] = []
    seen_bridge_ids: set[str] = set()
    label_counts: dict[str, int] = {name: 0 for name in v5.ID_TO_FINAL_LABEL.values()}
    family_counts: dict[str, int] = {}
    family_label_counts: dict[str, dict[str, int]] = {}

    for row_number, row in enumerate(raw_rows, start=1):
        prefix = f"[stage66] row {row_number} in {bridge_path}: "
        missing = [
            field for field in STAGE66_BRIDGE_REQUIRED_FIELDS if row.get(field) is None
        ]
        if missing:
            raise ValueError(f"{prefix}missing required fields: {missing}")

        row_id = row["id"]
        if row_id in seen_bridge_ids:
            raise ValueError(f"{prefix}duplicate bridge id: {row_id!r}")
        seen_bridge_ids.add(row_id)
        if row_id in existing_ids:
            raise ValueError(
                f"{prefix}bridge id {row_id!r} duplicates an existing clean train/dev "
                "or already-appended bridge id"
            )

        label = row["label"]
        if isinstance(label, bool) or label not in (0, 1, 2):
            raise ValueError(f"{prefix}label must be 0, 1, or 2; got {label!r}")
        final_label = v5.ID_TO_FINAL_LABEL[int(label)]
        row_final_label = row.get("final_label")
        if row_final_label is not None and row_final_label != final_label:
            raise ValueError(
                f"{prefix}label={label!r} does not match final_label={row_final_label!r}"
            )

        # Row-level forbidden-source scanning is restricted to id/pair_id/claim/
        # evidence only. stage66_family/stage66_bridge_family/stage66_subtype/
        # stage66_target_error/stage66_generation_source/stage66_leakage_policy
        # are intentionally NOT scanned (see module docstring above).
        for field_name in ("id", "pair_id", "claim", "evidence"):
            field_value = row.get(field_name)
            if isinstance(field_value, str):
                _stage60_check_forbidden_source(
                    field_value, f"row {row_number} field {field_name!r}"
                )

        normalized.append({
            "id": row_id,
            "pair_id": row.get("pair_id", row_id),
            "claim": row["claim"],
            "evidence": row["evidence"],
            "final_label": final_label,
            "frame_compatible_label": row.get("frame_compatible_label", 1),
            "predicate_covered_label": row.get("predicate_covered_label", 1),
            "sufficiency_label": row.get("sufficiency_label", 1),
            "polarity_label": row.get(
                "polarity_label", "NONE" if final_label == "NOT_ENTITLED" else final_label
            ),
            "primary_failure_type": row.get("primary_failure_type", "none"),
            "intervention_type": row.get("intervention_type", "stage66_bridge"),
            "stage66_family": row.get("stage66_family"),
            "stage66_bridge_family": row.get("stage66_bridge_family"),
            "stage66_subtype": row.get("stage66_subtype"),
            "stage66_target_error": row.get("stage66_target_error"),
            "stage66_generation_source": row.get("stage66_generation_source"),
            "stage66_leakage_policy": row.get("stage66_leakage_policy"),
        })

        label_counts[final_label] += 1
        family = row.get("stage66_bridge_family") or row.get("stage66_family") or "unknown"
        family_counts[family] = family_counts.get(family, 0) + 1
        family_label_counts.setdefault(
            family, {name: 0 for name in v5.ID_TO_FINAL_LABEL.values()}
        )
        family_label_counts[family][final_label] += 1

    return normalized, label_counts, family_counts, family_label_counts


# ---------------------------------------------------------------------------
# Stage75C: optional train-only Stage75 targeted residual bridge integration.
#
# Same policy as the Stage57/Stage66 bridges above: bridge rows are appended to
# the train split only, strictly after the clean main train/dev split has been
# created (and after any Stage57/Stage66 bridge appends), and never enter
# dev / checkpoint selection.
#
# Difference from Stage57/Stage66: the Stage75B generator
# (scripts/write_stage75_targeted_residual_bridge.py) never emits an integer
# "label" field or a "pair_id" field. Rows carry "final_label" (one of
# SUPPORT/REFUTE/NOT_ENTITLED) and "polarity_label" directly, and deliberately
# omit "pair_id" so the bridge dataset itself can never be swept into the
# intervention-pairwise-loss grouping path that keys off pair_id. This loader
# therefore never requires a "pair_id" field on the raw Stage75 source rows and
# never writes one back to any file; the only place a "pair_id" key is set is
# on the in-memory normalized record passed to v5.encode_records (which
# requires the key on every record), defaulted to the row's own "id" exactly
# like load_stage57_bridge_train_rows / load_stage66_bridge_train_rows already
# do for their own bridge rows.
# ---------------------------------------------------------------------------
STAGE75_BRIDGE_REQUIRED_FIELDS = (
    "id", "claim", "evidence", "final_label",
    "frame_compatible_label", "predicate_covered_label", "sufficiency_label",
    "polarity_label",
)
STAGE75_EXPECTED_POLARITY_BY_FINAL_LABEL = {
    "SUPPORT": "SUPPORT",
    "REFUTE": "REFUTE",
    "NOT_ENTITLED": "NONE",
}


def load_stage75_bridge_train_rows(
    bridge_path: Path,
    existing_ids: set[str],
) -> tuple[list[dict], dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    """Load, validate, and normalize Stage75 targeted residual bridge rows for
    train-only append.

    Mirrors load_stage57_bridge_train_rows / load_stage66_bridge_train_rows for
    data/stage75_targeted_residual_bridge.jsonl. Returns (normalized_records,
    label_counts, family_counts, family_label_counts). Raises
    ValueError/FileNotFoundError on any safety-check violation.
    """
    _stage60_check_forbidden_source(str(bridge_path), "--stage75-bridge-train-jsonl path")

    if not bridge_path.exists():
        raise FileNotFoundError(
            f"[stage75] --stage75-bridge-train-jsonl not found: {bridge_path}"
        )

    raw_rows: list[dict] = []
    with bridge_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"[stage75] invalid JSON on line {line_number} of {bridge_path}: {exc}"
                ) from exc

    if not raw_rows:
        raise ValueError(f"[stage75] --stage75-bridge-train-jsonl is empty: {bridge_path}")

    normalized: list[dict] = []
    seen_bridge_ids: set[str] = set()
    label_counts: dict[str, int] = {name: 0 for name in v5.ID_TO_FINAL_LABEL.values()}
    family_counts: dict[str, int] = {}
    family_label_counts: dict[str, dict[str, int]] = {}

    for row_number, row in enumerate(raw_rows, start=1):
        prefix = f"[stage75] row {row_number} in {bridge_path}: "
        missing = [
            field for field in STAGE75_BRIDGE_REQUIRED_FIELDS if row.get(field) is None
        ]
        if missing:
            raise ValueError(f"{prefix}missing required fields: {missing}")

        row_id = row["id"]
        if row_id in seen_bridge_ids:
            raise ValueError(f"{prefix}duplicate bridge id: {row_id!r}")
        seen_bridge_ids.add(row_id)
        if row_id in existing_ids:
            raise ValueError(
                f"{prefix}bridge id {row_id!r} duplicates an existing clean train/dev "
                "or already-appended bridge id"
            )

        final_label = row["final_label"]
        if final_label not in v5.FINAL_LABEL_TO_ID:
            raise ValueError(
                f"{prefix}invalid final_label {final_label!r}; expected one of "
                f"{sorted(v5.FINAL_LABEL_TO_ID)}"
            )

        polarity_label = row["polarity_label"]
        if polarity_label not in v5.POLARITY_LABEL_TO_ID:
            raise ValueError(
                f"{prefix}invalid polarity_label {polarity_label!r}; expected one of "
                f"{sorted(v5.POLARITY_LABEL_TO_ID)}"
            )
        expected_polarity = STAGE75_EXPECTED_POLARITY_BY_FINAL_LABEL.get(final_label)
        if polarity_label != expected_polarity:
            raise ValueError(
                f"{prefix}polarity_label {polarity_label!r} does not match "
                f"final_label {final_label!r} (expected {expected_polarity!r})"
            )

        # Row-level forbidden-source scanning is restricted to id/claim/evidence
        # only, mirroring load_stage66_bridge_train_rows: bridge_family/
        # bridge_subtype/target_error_type/bridge_source/leakage_note are
        # intentionally NOT scanned because leakage_note legitimately contains
        # "vitaminc" as part of a documented non-use declaration, not a leak.
        for field_name in ("id", "claim", "evidence"):
            field_value = row.get(field_name)
            if isinstance(field_value, str):
                _stage60_check_forbidden_source(
                    field_value, f"row {row_number} field {field_name!r}"
                )

        normalized.append({
            "id": row_id,
            "pair_id": row.get("pair_id", row_id),
            "claim": row["claim"],
            "evidence": row["evidence"],
            "final_label": final_label,
            "frame_compatible_label": row["frame_compatible_label"],
            "predicate_covered_label": row["predicate_covered_label"],
            "sufficiency_label": row["sufficiency_label"],
            "polarity_label": polarity_label,
            "primary_failure_type": row.get("primary_failure_type", "none"),
            "intervention_type": row.get("intervention_type", "stage75_bridge"),
            "stage75_bridge_family": row.get("bridge_family"),
            "stage75_bridge_subtype": row.get("bridge_subtype"),
            "stage75_target_error_type": row.get("target_error_type"),
            "stage75_bridge_source": row.get("bridge_source"),
            "stage75_leakage_note": row.get("leakage_note"),
        })

        label_counts[final_label] += 1
        family = row.get("bridge_family") or "unknown"
        family_counts[family] = family_counts.get(family, 0) + 1
        family_label_counts.setdefault(
            family, {name: 0 for name in v5.ID_TO_FINAL_LABEL.values()}
        )
        family_label_counts[family][final_label] += 1

    return normalized, label_counts, family_counts, family_label_counts


# ---------------------------------------------------------------------------
# Stage80D: optional train-only Stage80A conservative Stage75v2 bridge
# integration.
#
# Same policy as the Stage57/Stage66/Stage75 bridges above: bridge rows are
# appended to the train split only, strictly after the clean main train/dev
# split has been created (and after any Stage57/Stage66/Stage75 bridge
# appends), and never enter dev / checkpoint selection.
#
# Row schema mirrors Stage75's exactly (scripts/generate_stage80a_conservative_
# stage75v2_bridge.py never emits an integer "label" field or a "pair_id"
# field; rows carry "final_label"/"polarity_label" directly), except the
# family/subtype key names are "family"/"family_subtype" instead of
# "bridge_family"/"bridge_subtype" and there is no "target_error_type" field.
# This loader therefore reuses STAGE75_BRIDGE_REQUIRED_FIELDS and
# STAGE75_EXPECTED_POLARITY_BY_FINAL_LABEL rather than redefining them.
# ---------------------------------------------------------------------------
def load_stage80a_bridge_train_rows(
    bridge_path: Path,
    existing_ids: set[str],
) -> tuple[list[dict], dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    """Load, validate, and normalize Stage80A conservative Stage75v2 bridge rows
    for train-only append.

    Mirrors load_stage75_bridge_train_rows for
    data/stage80a_conservative_stage75v2_bridge.jsonl. Returns
    (normalized_records, label_counts, family_counts, family_label_counts).
    Raises ValueError/FileNotFoundError on any safety-check violation.
    """
    _stage60_check_forbidden_source(str(bridge_path), "--stage80a-bridge-train-jsonl path")

    if not bridge_path.exists():
        raise FileNotFoundError(
            f"[stage80a] --stage80a-bridge-train-jsonl not found: {bridge_path}"
        )

    raw_rows: list[dict] = []
    with bridge_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"[stage80a] invalid JSON on line {line_number} of {bridge_path}: {exc}"
                ) from exc

    if not raw_rows:
        raise ValueError(f"[stage80a] --stage80a-bridge-train-jsonl is empty: {bridge_path}")

    normalized: list[dict] = []
    seen_bridge_ids: set[str] = set()
    label_counts: dict[str, int] = {name: 0 for name in v5.ID_TO_FINAL_LABEL.values()}
    family_counts: dict[str, int] = {}
    family_label_counts: dict[str, dict[str, int]] = {}

    for row_number, row in enumerate(raw_rows, start=1):
        prefix = f"[stage80a] row {row_number} in {bridge_path}: "
        missing = [
            field for field in STAGE75_BRIDGE_REQUIRED_FIELDS if row.get(field) is None
        ]
        if missing:
            raise ValueError(f"{prefix}missing required fields: {missing}")

        row_id = row["id"]
        if row_id in seen_bridge_ids:
            raise ValueError(f"{prefix}duplicate bridge id: {row_id!r}")
        seen_bridge_ids.add(row_id)
        if row_id in existing_ids:
            raise ValueError(
                f"{prefix}bridge id {row_id!r} duplicates an existing clean train/dev "
                "or already-appended bridge id"
            )

        final_label = row["final_label"]
        if final_label not in v5.FINAL_LABEL_TO_ID:
            raise ValueError(
                f"{prefix}invalid final_label {final_label!r}; expected one of "
                f"{sorted(v5.FINAL_LABEL_TO_ID)}"
            )

        polarity_label = row["polarity_label"]
        if polarity_label not in v5.POLARITY_LABEL_TO_ID:
            raise ValueError(
                f"{prefix}invalid polarity_label {polarity_label!r}; expected one of "
                f"{sorted(v5.POLARITY_LABEL_TO_ID)}"
            )
        expected_polarity = STAGE75_EXPECTED_POLARITY_BY_FINAL_LABEL.get(final_label)
        if polarity_label != expected_polarity:
            raise ValueError(
                f"{prefix}polarity_label {polarity_label!r} does not match "
                f"final_label {final_label!r} (expected {expected_polarity!r})"
            )

        # Row-level forbidden-source scanning is restricted to id/claim/evidence
        # only, mirroring load_stage75_bridge_train_rows: family/family_subtype/
        # bridge_source/leakage_note are intentionally NOT scanned because
        # leakage_note legitimately contains "vitaminc" as part of a documented
        # non-use declaration, not a leak, and bridge_source legitimately
        # contains this bridge's own official name (which references Stage75v2
        # as a naming convention, not as a used data source).
        for field_name in ("id", "claim", "evidence"):
            field_value = row.get(field_name)
            if isinstance(field_value, str):
                _stage60_check_forbidden_source(
                    field_value, f"row {row_number} field {field_name!r}"
                )

        normalized.append({
            "id": row_id,
            "pair_id": row.get("pair_id", row_id),
            "claim": row["claim"],
            "evidence": row["evidence"],
            "final_label": final_label,
            "frame_compatible_label": row["frame_compatible_label"],
            "predicate_covered_label": row["predicate_covered_label"],
            "sufficiency_label": row["sufficiency_label"],
            "polarity_label": polarity_label,
            "primary_failure_type": row.get("primary_failure_type", "none"),
            "intervention_type": row.get("intervention_type", "stage80a_bridge"),
            "stage80a_bridge_family": row.get("family"),
            "stage80a_bridge_subtype": row.get("family_subtype"),
            "stage80a_bridge_source": row.get("bridge_source"),
            "stage80a_leakage_note": row.get("leakage_note"),
        })

        label_counts[final_label] += 1
        family = row.get("family") or "unknown"
        family_counts[family] = family_counts.get(family, 0) + 1
        family_label_counts.setdefault(
            family, {name: 0 for name in v5.ID_TO_FINAL_LABEL.values()}
        )
        family_label_counts[family][final_label] += 1

    return normalized, label_counts, family_counts, family_label_counts


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.stage118_diagnostic_evidence_interface_sweep_list = (
            parse_stage118_diagnostic_evidence_interface_sweep(
                args.stage118_diagnostic_evidence_interface_sweep
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.stage118_diagnostic_evidence_interface_sweep_list:
        if args.stage118_diagnostic_jsonl is None:
            parser.error(
                "--stage118-diagnostic-evidence-interface-sweep requires "
                "--stage118-diagnostic-jsonl"
            )
        if args.stage118_diagnostic_sweep_output_dir is None:
            parser.error(
                "--stage118-diagnostic-evidence-interface-sweep requires "
                "--stage118-diagnostic-sweep-output-dir"
            )

    # ---------------------------------------------------------------------------
    # Stage48: optionally load the Stage47-validated frozen recovery config
    # ---------------------------------------------------------------------------
    if getattr(args, "use_stage47_selected_recovery_config", False):
        _stage47_support_w, _stage47_ne_w = load_stage47_selected_recovery_weights(
            args.stage47_recovery_config_path
        )
        if (
            args.stage45c_support_recovery_weight != 0.0
            or args.stage45c_entitled_ne_penalty_weight != 0.0
        ):
            print(
                "[stage48] Manual --stage45c-support-recovery-weight/"
                "--stage45c-entitled-ne-penalty-weight values "
                f"(support_w={args.stage45c_support_recovery_weight}, "
                f"ne_w={args.stage45c_entitled_ne_penalty_weight}) were overridden by "
                "the Stage47 frozen recovery config."
            )
        args.stage45c_support_recovery_weight = _stage47_support_w
        args.stage45c_entitled_ne_penalty_weight = _stage47_ne_w
        print(
            f"[stage48] Loaded Stage47 selected recovery config "
            f"'{STAGE47_SELECTED_CONFIG_NAME}' from {args.stage47_recovery_config_path}: "
            f"support_w={_stage47_support_w}, ne_w={_stage47_ne_w}."
        )

    # ---------------------------------------------------------------------------
    # Fail-fast: dummy backbone guard
    # Dummy backbone is valid ONLY for explicit smoke/plumbing diagnostics.
    # It has no text comprehension capacity; metrics it produces are not
    # claim-worthy. Require --allow-dummy-backbone to proceed.
    # ---------------------------------------------------------------------------
    #  Stage28-I-A: location boundary flag validation 
    _use_lb_head = getattr(args, "v7_use_location_boundary_head", False)
    _use_lb_loss = getattr(args, "v7_use_location_boundary_loss", False)
    _lb_cap_mode = getattr(args, "v7_location_boundary_cap_mode", "none")
    _lb_cap_gamma = getattr(args, "v7_location_boundary_cap_gamma", 1.0)
    if _use_lb_loss and not _use_lb_head:
        raise ValueError(
            "--v7-use-location-boundary-loss requires --v7-use-location-boundary-head. "
            "The location boundary head must be enabled to compute the auxiliary loss."
        )
    if _lb_cap_mode != "none" and not _use_lb_head:
        raise ValueError(
            f"--v7-location-boundary-cap-mode {_lb_cap_mode!r} requires "
            "--v7-use-location-boundary-head. "
            "The location boundary head must be enabled to apply the cap."
        )
    if _lb_cap_gamma <= 0:
        raise ValueError(
            f"--v7-location-boundary-cap-gamma must be > 0, got {_lb_cap_gamma!r}."
        )

    #  Stage30-C2: temporal safety flag validation 
    _use_ts_head = getattr(args, "v7_use_temporal_safety_head", False)
    _use_ts_loss = getattr(args, "v7_use_temporal_safety_loss", False)
    _ts_cap_mode = getattr(args, "v7_temporal_safety_cap_mode", "none")
    _ts_cap_gamma = getattr(args, "v7_temporal_safety_cap_gamma", 1.0)
    _ts_data = getattr(args, "v7_temporal_safety_data", None)
    if _use_ts_loss and not _use_ts_head:
        raise ValueError(
            "--v7-use-temporal-safety-loss requires --v7-use-temporal-safety-head. "
            "The temporal safety head must be enabled to compute the auxiliary loss."
        )
    if _use_ts_loss and _ts_data is None:
        raise ValueError(
            "--v7-use-temporal-safety-loss requires --v7-temporal-safety-data. "
            "Provide the temporal safety diagnostic JSONL via --v7-temporal-safety-data."
        )
    if _ts_cap_mode != "none" and not _use_ts_head:
        raise ValueError(
            f"--v7-temporal-safety-cap-mode {_ts_cap_mode!r} requires "
            "--v7-use-temporal-safety-head. "
            "The temporal safety head must be enabled to apply the cap."
        )
    if _ts_cap_gamma <= 0:
        raise ValueError(
            f"--v7-temporal-safety-cap-gamma must be > 0, got {_ts_cap_gamma!r}."
        )
    if _use_ts_loss and _ts_data is not None and not Path(_ts_data).exists():
        raise FileNotFoundError(
            f"[ts_head] --v7-temporal-safety-data not found: {_ts_data}"
        )
    if _use_ts_loss and _ts_data is not None:
        if Path(_ts_data).resolve() == Path(args.data).resolve():
            raise ValueError(
                "[ts_head] --v7-temporal-safety-data must not be the same as --data.\n"
                "Temporal safety records (including time_swap) must not be mixed "
                "into the main clean controlled train/eval classification data."
            )

    #  Stage30-D: temporal mismatch multihead flag validation 
    _use_tmm = getattr(args, "v7_use_temporal_mismatch_multihead", False)
    _use_tmm_loss = getattr(args, "v7_use_temporal_mismatch_multihead_loss", False)
    _tmm_cap_mode = getattr(args, "v7_temporal_mismatch_multihead_cap_mode", "none")
    _tmm_cap_gamma = getattr(args, "v7_temporal_mismatch_multihead_cap_gamma", 1.0)
    _tmm_data = getattr(args, "v7_temporal_mismatch_multihead_data", None)
    if _use_tmm_loss and not _use_tmm:
        raise ValueError(
            "--v7-use-temporal-mismatch-multihead-loss requires "
            "--v7-use-temporal-mismatch-multihead. "
            "The multihead must be enabled to compute the auxiliary loss."
        )
    if _use_tmm_loss and _tmm_data is None:
        raise ValueError(
            "--v7-use-temporal-mismatch-multihead-loss requires "
            "--v7-temporal-mismatch-multihead-data. "
            "Provide the temporal mismatch auxiliary JSONL via "
            "--v7-temporal-mismatch-multihead-data."
        )
    if _tmm_cap_mode != "none" and not _use_tmm:
        raise ValueError(
            f"--v7-temporal-mismatch-multihead-cap-mode {_tmm_cap_mode!r} requires "
            "--v7-use-temporal-mismatch-multihead. "
            "The multihead must be enabled to apply the cap."
        )
    if _tmm_cap_gamma <= 0:
        raise ValueError(
            f"--v7-temporal-mismatch-multihead-cap-gamma must be > 0, "
            f"got {_tmm_cap_gamma!r}."
        )
    if _tmm_cap_mode != "none" and _ts_cap_mode != "none":
        raise ValueError(
            "Stage30-C2 temporal safety cap (--v7-temporal-safety-cap-mode) and "
            "Stage30-D temporal mismatch multihead cap "
            "(--v7-temporal-mismatch-multihead-cap-mode) cannot both be active.\n"
            f"  v7_temporal_safety_cap_mode={_ts_cap_mode!r}\n"
            f"  v7_temporal_mismatch_multihead_cap_mode={_tmm_cap_mode!r}\n"
            "Set one to 'none'."
        )
    if _use_tmm_loss and _tmm_data is not None and not Path(_tmm_data).exists():
        raise FileNotFoundError(
            f"[tmm_head] --v7-temporal-mismatch-multihead-data not found: {_tmm_data}"
        )
    if _use_tmm_loss and _tmm_data is not None:
        if Path(_tmm_data).resolve() == Path(args.data).resolve():
            raise ValueError(
                "[tmm_head] --v7-temporal-mismatch-multihead-data must not be "
                "the same as --data.\n"
                "Temporal mismatch multihead records must not be mixed into the "
                "main clean controlled train/eval classification data."
            )

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

    if args.stage126_preflight_export_only:
        preflight_report = run_stage126_preflight_export_only(
            args=args, parser=parser, device=device
        )
        print(
            "[STAGE126 PREFLIGHT] "
            f"rows={preflight_report['n_preflight_rows']} "
            f"predictions={preflight_report['output_jsonl']} "
            f"report={preflight_report['report_json']}"
        )
        return 0

    records = v5.load_jsonl(args.data)
    if args.max_train_records is not None:
        records = records[: args.max_train_records]

    _stage45b_split_info: dict[str, Any] = {
        "stage45b_enabled": False,
        "stage45b_decision": "STAGE45B_INTERNAL_FAMILY_HOLDOUT_DISABLED",
        "stage45b_family_field_used": None,
        "stage45b_holdout_family": None,
        "stage45b_train_rows": None,
        "stage45b_holdout_rows": None,
        "stage45b_train_label_counts": None,
        "stage45b_holdout_label_counts": None,
        "stage45b_holdout_metrics": None,
        "stage45b_leakage_policy": (
            "Stage45-B family holdout is disabled; default internal train/dev split behavior is unchanged."
        ),
        "stage45b_recommendation": (
            "Enable --stage45-use-family-holdout only for internal controlled-family robustness diagnostics."
        ),
    }
    if args.stage45_use_family_holdout:
        if not args.stage45_holdout_family:
            raise ValueError(
                "--stage45-use-family-holdout requires --stage45-holdout-family."
            )
        train_records, dev_records, _stage45b_split_info = split_leave_family_out(
            records,
            holdout_family=args.stage45_holdout_family,
            family_field=args.stage45_family_field,
            min_holdout_size=args.stage45_min_holdout_size,
        )
    else:
        train_records, dev_records = v5.split_by_pair_id(
            records, dev_ratio=args.dev_ratio, seed=args.seed
        )

    # Stage71: train-source mask, captured immediately after the clean main
    # train/dev split and before any bridge rows are appended below. Used to
    # restrict intervention_pairwise_losses (which requires a full
    # intervention family per pair_id, including an intervention_type=="none"
    # original record) to clean main train rows only. Stage57/Stage66 bridge
    # rows are standalone examples that never satisfy that requirement, but
    # must still flow through CE/classification training normally, so they
    # are tracked here rather than removed from train_records.
    _clean_main_train_row_count = len(train_records)
    _train_source_labels: list[str] = ["clean_main"] * _clean_main_train_row_count

    # ---------------------------------------------------------------------------
    # Stage60: optional train-only Stage57 non-leaking external bridge append.
    # Must run strictly AFTER the clean main train/dev split above. Bridge rows
    # are only ever added to train_records; dev_records (checkpoint selection)
    # is never touched.
    # ---------------------------------------------------------------------------
    _stage60_bridge_info: dict[str, Any] = {
        "stage57_bridge_train_mode": args.stage57_bridge_train_mode,
        "stage57_bridge_train_jsonl": (
            str(args.stage57_bridge_train_jsonl)
            if args.stage57_bridge_train_jsonl is not None else None
        ),
        "stage57_bridge_train_enabled": False,
        "stage57_bridge_train_row_count": 0,
        "stage57_bridge_train_label_counts": None,
        "stage57_bridge_train_family_counts": None,
        "stage57_bridge_train_family_label_counts": None,
        "stage57_bridge_train_only": False,
        "stage57_bridge_appended_after_clean_split": False,
        "stage57_bridge_used_for_dev": False,
        "stage57_bridge_used_for_checkpoint_selection": False,
        "stage57_external_data_used_for_training": False,
        "stage57_external_metrics_used_for_threshold_tuning": False,
    }
    if args.stage57_bridge_train_mode == "append_train_only":
        if args.stage57_bridge_train_jsonl is None:
            raise ValueError(
                "--stage57-bridge-train-mode append_train_only requires "
                "--stage57-bridge-train-jsonl."
            )
        _stage60_bridge_path = Path(args.stage57_bridge_train_jsonl)
        _stage60_existing_ids = {r["id"] for r in train_records} | {
            r["id"] for r in dev_records
        }
        (
            _stage60_bridge_records,
            _stage60_bridge_label_counts,
            _stage60_bridge_family_counts,
            _stage60_bridge_family_label_counts,
        ) = load_stage57_bridge_train_rows(_stage60_bridge_path, _stage60_existing_ids)

        train_records = train_records + _stage60_bridge_records
        _train_source_labels = _train_source_labels + (
            ["stage57_bridge"] * len(_stage60_bridge_records)
        )

        _stage60_bridge_info.update({
            "stage57_bridge_train_enabled": True,
            "stage57_bridge_train_row_count": len(_stage60_bridge_records),
            "stage57_bridge_train_label_counts": _stage60_bridge_label_counts,
            "stage57_bridge_train_family_counts": _stage60_bridge_family_counts,
            "stage57_bridge_train_family_label_counts": _stage60_bridge_family_label_counts,
            "stage57_bridge_train_only": True,
            "stage57_bridge_appended_after_clean_split": True,
        })
        print(
            f"[stage60] appended Stage57 bridge train rows: "
            f"{len(_stage60_bridge_records)} from {_stage60_bridge_path}"
        )
        print(f"[stage60] bridge label counts: {_stage60_bridge_label_counts}")
        print(f"[stage60] bridge family counts: {_stage60_bridge_family_counts}")
    elif args.stage57_bridge_train_jsonl is not None:
        print(
            "[stage60] --stage57-bridge-train-jsonl was provided but "
            "--stage57-bridge-train-mode is 'none'; bridge data will NOT be used "
            "(default training/data split behavior is unchanged)."
        )

    # ---------------------------------------------------------------------------
    # Stage69/Stage70: optional train-only Stage66 residual bridge append.
    # Must run strictly AFTER the clean main train/dev split (and after any
    # Stage57 bridge append above). Bridge rows are only ever added to
    # train_records; dev_records (checkpoint selection) is never touched.
    # ---------------------------------------------------------------------------
    _stage66_bridge_info: dict[str, Any] = {
        "stage66_bridge_train_mode": args.stage66_bridge_train_mode,
        "stage66_bridge_train_jsonl": (
            str(args.stage66_bridge_train_jsonl)
            if args.stage66_bridge_train_jsonl is not None else None
        ),
        "stage66_bridge_train_enabled": False,
        "stage66_bridge_train_row_count": 0,
        "stage66_bridge_train_label_counts": None,
        "stage66_bridge_train_family_counts": None,
        "stage66_bridge_train_family_label_counts": None,
        "stage66_bridge_train_only": False,
        "stage66_bridge_appended_after_clean_split": False,
        "stage66_bridge_used_for_dev": False,
        "stage66_bridge_used_for_checkpoint_selection": False,
        "stage66_external_data_used_for_training": False,
        "stage66_external_metrics_used_for_threshold_tuning": False,
    }
    if args.stage66_bridge_train_mode == "append_train_only":
        if args.stage66_bridge_train_jsonl is None:
            raise ValueError(
                "--stage66-bridge-train-mode append_train_only requires "
                "--stage66-bridge-train-jsonl."
            )
        _stage66_bridge_path = Path(args.stage66_bridge_train_jsonl)
        _stage66_existing_ids = {r["id"] for r in train_records} | {
            r["id"] for r in dev_records
        }
        (
            _stage66_bridge_records,
            _stage66_bridge_label_counts,
            _stage66_bridge_family_counts,
            _stage66_bridge_family_label_counts,
        ) = load_stage66_bridge_train_rows(_stage66_bridge_path, _stage66_existing_ids)

        train_records = train_records + _stage66_bridge_records
        _train_source_labels = _train_source_labels + (
            ["stage66_bridge"] * len(_stage66_bridge_records)
        )

        _stage66_bridge_info.update({
            "stage66_bridge_train_enabled": True,
            "stage66_bridge_train_row_count": len(_stage66_bridge_records),
            "stage66_bridge_train_label_counts": _stage66_bridge_label_counts,
            "stage66_bridge_train_family_counts": _stage66_bridge_family_counts,
            "stage66_bridge_train_family_label_counts": _stage66_bridge_family_label_counts,
            "stage66_bridge_train_only": True,
            "stage66_bridge_appended_after_clean_split": True,
        })
        print(
            f"[stage66] appended Stage66 bridge train rows: "
            f"{len(_stage66_bridge_records)} from {_stage66_bridge_path}"
        )
        print(f"[stage66] bridge label counts: {_stage66_bridge_label_counts}")
        print(f"[stage66] bridge family counts: {_stage66_bridge_family_counts}")
    elif args.stage66_bridge_train_jsonl is not None:
        print(
            "[stage66] --stage66-bridge-train-jsonl was provided but "
            "--stage66-bridge-train-mode is 'none'; bridge data will NOT be used "
            "(default training/data split behavior is unchanged)."
        )

    # ---------------------------------------------------------------------------
    # Stage75C: optional train-only Stage75 targeted residual bridge append.
    # Must run strictly AFTER the clean main train/dev split (and after any
    # Stage57/Stage66 bridge appends above). Bridge rows are only ever added to
    # train_records; dev_records (checkpoint selection) is never touched.
    # ---------------------------------------------------------------------------
    _stage75_bridge_info: dict[str, Any] = {
        "stage75_bridge_train_mode": args.stage75_bridge_train_mode,
        "stage75_bridge_train_jsonl": (
            str(args.stage75_bridge_train_jsonl)
            if args.stage75_bridge_train_jsonl is not None else None
        ),
        "stage75_bridge_train_enabled": False,
        "stage75_bridge_train_row_count": 0,
        "stage75_bridge_train_label_counts": None,
        "stage75_bridge_train_family_counts": None,
        "stage75_bridge_train_family_label_counts": None,
        "stage75_bridge_train_only": False,
        "stage75_bridge_appended_after_clean_split": False,
        "stage75_bridge_used_for_dev": False,
        "stage75_bridge_used_for_checkpoint_selection": False,
        "stage75_external_data_used_for_training": False,
        "stage75_external_metrics_used_for_threshold_tuning": False,
    }
    if args.stage75_bridge_train_mode == "append_train_only":
        if args.stage75_bridge_train_jsonl is None:
            raise ValueError(
                "--stage75-bridge-train-mode append_train_only requires "
                "--stage75-bridge-train-jsonl."
            )
        _stage75_bridge_path = Path(args.stage75_bridge_train_jsonl)
        _stage75_existing_ids = {r["id"] for r in train_records} | {
            r["id"] for r in dev_records
        }
        (
            _stage75_bridge_records,
            _stage75_bridge_label_counts,
            _stage75_bridge_family_counts,
            _stage75_bridge_family_label_counts,
        ) = load_stage75_bridge_train_rows(_stage75_bridge_path, _stage75_existing_ids)

        train_records = train_records + _stage75_bridge_records
        _train_source_labels = _train_source_labels + (
            ["stage75_bridge"] * len(_stage75_bridge_records)
        )

        _stage75_bridge_info.update({
            "stage75_bridge_train_enabled": True,
            "stage75_bridge_train_row_count": len(_stage75_bridge_records),
            "stage75_bridge_train_label_counts": _stage75_bridge_label_counts,
            "stage75_bridge_train_family_counts": _stage75_bridge_family_counts,
            "stage75_bridge_train_family_label_counts": _stage75_bridge_family_label_counts,
            "stage75_bridge_train_only": True,
            "stage75_bridge_appended_after_clean_split": True,
        })
        print(
            f"[stage75] appended Stage75 bridge train rows: "
            f"{len(_stage75_bridge_records)} from {_stage75_bridge_path}"
        )
        print(f"[stage75] bridge label counts: {_stage75_bridge_label_counts}")
        print(f"[stage75] bridge family counts: {_stage75_bridge_family_counts}")
    elif args.stage75_bridge_train_jsonl is not None:
        print(
            "[stage75] --stage75-bridge-train-jsonl was provided but "
            "--stage75-bridge-train-mode is 'none'; bridge data will NOT be used "
            "(default training/data split behavior is unchanged)."
        )

    # ---------------------------------------------------------------------------
    # Stage80D: optional train-only Stage80A conservative Stage75v2 bridge
    # append. Must run strictly AFTER the clean main train/dev split (and after
    # any Stage57/Stage66/Stage75 bridge appends above). Bridge rows are only
    # ever added to train_records; dev_records (checkpoint selection) is never
    # touched. Stage75's full bridge is NOT used by this integration path --
    # this is a separate, independent bridge (data/stage80a_conservative_
    # stage75v2_bridge.jsonl) selected under --stage80a-bridge-train-jsonl.
    # ---------------------------------------------------------------------------
    _stage80a_bridge_info: dict[str, Any] = {
        "stage80a_bridge_train_mode": args.stage80a_bridge_train_mode,
        "stage80a_bridge_train_jsonl": (
            str(args.stage80a_bridge_train_jsonl)
            if args.stage80a_bridge_train_jsonl is not None else None
        ),
        "stage80a_bridge_train_enabled": False,
        "stage80a_bridge_train_row_count": 0,
        "stage80a_bridge_train_label_counts": None,
        "stage80a_bridge_train_family_counts": None,
        "stage80a_bridge_train_family_label_counts": None,
        "stage80a_bridge_train_only": False,
        "stage80a_bridge_appended_after_clean_split": False,
        "stage80a_bridge_used_for_dev": False,
        "stage80a_bridge_used_for_checkpoint_selection": False,
        "stage80a_external_data_used_for_training": False,
        "stage80a_external_metrics_used_for_threshold_tuning": False,
    }
    if args.stage80a_bridge_train_mode == "append_train_only":
        if args.stage80a_bridge_train_jsonl is None:
            raise ValueError(
                "--stage80a-bridge-train-mode append_train_only requires "
                "--stage80a-bridge-train-jsonl."
            )
        _stage80a_bridge_path = Path(args.stage80a_bridge_train_jsonl)
        _stage80a_existing_ids = {r["id"] for r in train_records} | {
            r["id"] for r in dev_records
        }
        (
            _stage80a_bridge_records,
            _stage80a_bridge_label_counts,
            _stage80a_bridge_family_counts,
            _stage80a_bridge_family_label_counts,
        ) = load_stage80a_bridge_train_rows(_stage80a_bridge_path, _stage80a_existing_ids)

        train_records = train_records + _stage80a_bridge_records
        _train_source_labels = _train_source_labels + (
            ["stage80a_bridge"] * len(_stage80a_bridge_records)
        )

        _stage80a_bridge_info.update({
            "stage80a_bridge_train_enabled": True,
            "stage80a_bridge_train_row_count": len(_stage80a_bridge_records),
            "stage80a_bridge_train_label_counts": _stage80a_bridge_label_counts,
            "stage80a_bridge_train_family_counts": _stage80a_bridge_family_counts,
            "stage80a_bridge_train_family_label_counts": _stage80a_bridge_family_label_counts,
            "stage80a_bridge_train_only": True,
            "stage80a_bridge_appended_after_clean_split": True,
        })
        print(
            f"[stage80a] appended Stage80A bridge train rows: "
            f"{len(_stage80a_bridge_records)} from {_stage80a_bridge_path}"
        )
        print(f"[stage80a] bridge label counts: {_stage80a_bridge_label_counts}")
        print(f"[stage80a] bridge family counts: {_stage80a_bridge_family_counts}")
    elif args.stage80a_bridge_train_jsonl is not None:
        print(
            "[stage80a] --stage80a-bridge-train-jsonl was provided but "
            "--stage80a-bridge-train-mode is 'none'; bridge data will NOT be used "
            "(default training/data split behavior is unchanged)."
        )

    # ---------------------------------------------------------------------------
    # Stage69/Stage70/Stage75C/Stage80D: report-field aliases (spec-named,
    # without "_train_") plus combined Stage57+Stage66+Stage75+Stage80A bridge
    # metadata. These are additive: the original stage57_bridge_train_* /
    # stage66_bridge_train_* / stage75_bridge_train_* / stage80a_bridge_train_*
    # fields above are preserved unchanged for backward compatibility.
    # ---------------------------------------------------------------------------
    _stage60_bridge_info.update({
        "stage57_bridge_enabled": _stage60_bridge_info["stage57_bridge_train_enabled"],
        "stage57_bridge_row_count": _stage60_bridge_info["stage57_bridge_train_row_count"],
        "stage57_bridge_label_counts": _stage60_bridge_info["stage57_bridge_train_label_counts"],
        "stage57_bridge_family_counts": _stage60_bridge_info["stage57_bridge_train_family_counts"],
        "stage57_used_for_dev": _stage60_bridge_info["stage57_bridge_used_for_dev"],
        "stage57_used_for_checkpoint_selection": (
            _stage60_bridge_info["stage57_bridge_used_for_checkpoint_selection"]
        ),
    })
    _stage66_bridge_info.update({
        "stage66_bridge_enabled": _stage66_bridge_info["stage66_bridge_train_enabled"],
        "stage66_bridge_row_count": _stage66_bridge_info["stage66_bridge_train_row_count"],
        "stage66_bridge_label_counts": _stage66_bridge_info["stage66_bridge_train_label_counts"],
        "stage66_bridge_family_counts": _stage66_bridge_info["stage66_bridge_train_family_counts"],
        "stage66_used_for_dev": _stage66_bridge_info["stage66_bridge_used_for_dev"],
        "stage66_used_for_checkpoint_selection": (
            _stage66_bridge_info["stage66_bridge_used_for_checkpoint_selection"]
        ),
    })
    _stage75_bridge_info.update({
        "stage75_bridge_enabled": _stage75_bridge_info["stage75_bridge_train_enabled"],
        "stage75_bridge_row_count": _stage75_bridge_info["stage75_bridge_train_row_count"],
        "stage75_bridge_label_counts": _stage75_bridge_info["stage75_bridge_train_label_counts"],
        "stage75_bridge_family_counts": _stage75_bridge_info["stage75_bridge_train_family_counts"],
        "stage75_used_for_dev": _stage75_bridge_info["stage75_bridge_used_for_dev"],
        "stage75_used_for_checkpoint_selection": (
            _stage75_bridge_info["stage75_bridge_used_for_checkpoint_selection"]
        ),
    })
    _stage80a_bridge_info.update({
        "stage80a_bridge_enabled": _stage80a_bridge_info["stage80a_bridge_train_enabled"],
        "stage80a_bridge_row_count": _stage80a_bridge_info["stage80a_bridge_train_row_count"],
        "stage80a_bridge_label_counts": _stage80a_bridge_info["stage80a_bridge_train_label_counts"],
        "stage80a_bridge_family_counts": _stage80a_bridge_info["stage80a_bridge_train_family_counts"],
        "stage80a_used_for_dev": _stage80a_bridge_info["stage80a_bridge_used_for_dev"],
        "stage80a_used_for_checkpoint_selection": (
            _stage80a_bridge_info["stage80a_bridge_used_for_checkpoint_selection"]
        ),
    })

    _bridge_sources_enabled = [
        _name for _name, _enabled in (
            ("stage57", _stage60_bridge_info["stage57_bridge_enabled"]),
            ("stage66", _stage66_bridge_info["stage66_bridge_enabled"]),
            ("stage75", _stage75_bridge_info["stage75_bridge_enabled"]),
            ("stage80a", _stage80a_bridge_info["stage80a_bridge_enabled"]),
        )
        if _enabled
    ]
    _combined_bridge_label_counts = {
        name: (
            (_stage60_bridge_info.get("stage57_bridge_label_counts") or {}).get(name, 0)
            + (_stage66_bridge_info.get("stage66_bridge_label_counts") or {}).get(name, 0)
            + (_stage75_bridge_info.get("stage75_bridge_label_counts") or {}).get(name, 0)
            + (_stage80a_bridge_info.get("stage80a_bridge_label_counts") or {}).get(name, 0)
        )
        for name in v5.ID_TO_FINAL_LABEL.values()
    }
    _combined_bridge_info: dict[str, Any] = {
        "combined_bridge_enabled": bool(_bridge_sources_enabled),
        "combined_bridge_row_count": (
            _stage60_bridge_info["stage57_bridge_row_count"]
            + _stage66_bridge_info["stage66_bridge_row_count"]
            + _stage75_bridge_info["stage75_bridge_row_count"]
            + _stage80a_bridge_info["stage80a_bridge_row_count"]
        ),
        "combined_bridge_label_counts": _combined_bridge_label_counts,
        "combined_bridge_train_only": bool(_bridge_sources_enabled),
        "bridge_sources_enabled": _bridge_sources_enabled,
        "clean_dev_for_checkpoint_selection": True,
        "external_data_used_for_training": False,
        "external_metrics_used_for_threshold_tuning": False,
        "time_swap_used": False,
    }

    # Stage71/Stage75C/Stage80D: Stage57/Stage66/Stage75/Stage80A bridge rows
    # have no intervention_type=="none" original record for their pair_id, so
    # intervention_pairwise_losses (which requires a full intervention family
    # per pair_id) must never see them. These counts are derived from the same
    # bridge-row-count bookkeeping as _combined_bridge_info above; the actual
    # exclusion is applied inside run_training_v6b using the
    # _train_source_labels mask built alongside the train/dev split and bridge
    # appends. Bridge rows are NOT removed from train_records/train_inputs --
    # they remain fully active for CE/label loss and all other per-row losses.
    assert len(_train_source_labels) == len(train_records), (
        "_train_source_labels must track train_records 1:1 after bridge appends"
    )

    # Stage123-A: optional experimental vNext evidence interface, applied only
    # after all train/dev membership decisions and bridge appends are complete.
    train_records = apply_vnext_evidence_interface_to_records(
        train_records, args.vnext_evidence_interface
    )
    dev_records = apply_vnext_evidence_interface_to_records(
        dev_records, args.vnext_evidence_interface
    )
    _pairwise_loss_stage57_excluded = _stage60_bridge_info["stage57_bridge_row_count"]
    _pairwise_loss_stage66_excluded = _stage66_bridge_info["stage66_bridge_row_count"]
    _pairwise_loss_stage75_excluded = _stage75_bridge_info["stage75_bridge_row_count"]
    _pairwise_loss_stage80a_excluded = _stage80a_bridge_info["stage80a_bridge_row_count"]
    _pairwise_loss_bridge_excluded = (
        _pairwise_loss_stage57_excluded
        + _pairwise_loss_stage66_excluded
        + _pairwise_loss_stage75_excluded
        + _pairwise_loss_stage80a_excluded
    )
    _combined_bridge_info.update({
        "bridge_rows_excluded_from_intervention_pairwise_loss": _pairwise_loss_bridge_excluded > 0,
        "stage57_excluded_from_intervention_pairwise_loss": _pairwise_loss_stage57_excluded > 0,
        "stage66_excluded_from_intervention_pairwise_loss": _pairwise_loss_stage66_excluded > 0,
        "stage75_excluded_from_intervention_pairwise_loss": _pairwise_loss_stage75_excluded > 0,
        "stage80a_excluded_from_intervention_pairwise_loss": _pairwise_loss_stage80a_excluded > 0,
        "intervention_pairwise_loss_source": (
            "clean_main_train_only" if _pairwise_loss_bridge_excluded > 0 else "full_train"
        ),
        "intervention_pairwise_loss_clean_main_row_count": _clean_main_train_row_count,
        # Stage80D2: final train row count after the clean main train/dev split
        # plus every append_train_only bridge (Stage57/Stage66/Stage75/Stage80A
        # as applicable). Computed dynamically from the already-finalized
        # train_records list -- never hardcoded -- so it always reflects
        # whichever bridges were actually enabled for this run.
        "final_train_row_count_expected": len(train_records),
        "intervention_pairwise_loss_bridge_row_count_excluded": _pairwise_loss_bridge_excluded,
        "intervention_pairwise_loss_stage57_row_count_excluded": _pairwise_loss_stage57_excluded,
        "intervention_pairwise_loss_stage66_row_count_excluded": _pairwise_loss_stage66_excluded,
        "intervention_pairwise_loss_stage75_row_count_excluded": _pairwise_loss_stage75_excluded,
        "intervention_pairwise_loss_stage80a_row_count_excluded": _pairwise_loss_stage80a_excluded,
    })

    ce_class_weights = compute_class_weights_v6b(train_records, args.class_weighting, device)
    label_counts: dict[str, int] = {name: 0 for name in v5.ID_TO_FINAL_LABEL.values()}
    for _record in train_records:
        label_counts[_record["final_label"]] += 1

    vocab: dict[str, int] | None = None
    tokenizer: Any | None = None

    if args.backbone == "dummy":
        vocab_records = (
            records
            if args.vnext_evidence_interface == "full_evidence"
            else train_records + dev_records
        )
        vocab = v5.build_vocab(vocab_records)
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
                v7_use_v6b_style_final_decision=args.v7_use_v6b_style_final_decision,
                v7_use_learnable_ne_alpha=args.v7_use_learnable_ne_alpha,
                v7_ne_alpha_init=args.v7_ne_alpha_init,
                v7_h1_entitlement_decision_signal=args.v7_h1_entitlement_decision_signal,
                v7_h1_entitlement_product_power=args.v7_h1_entitlement_product_power,
                v7_h1_hybrid_residual_beta=args.v7_h1_hybrid_residual_beta,
                v7_use_location_boundary_head=getattr(args, "v7_use_location_boundary_head", False),
                v7_location_boundary_cap_mode=getattr(args, "v7_location_boundary_cap_mode", "none"),
                v7_location_boundary_cap_gamma=getattr(args, "v7_location_boundary_cap_gamma", 1.0),
                v7_location_boundary_cap_detach=getattr(args, "v7_location_boundary_cap_detach", False),
                v7_use_temporal_safety_head=getattr(args, "v7_use_temporal_safety_head", False),
                v7_temporal_safety_cap_mode=getattr(args, "v7_temporal_safety_cap_mode", "none"),
                v7_temporal_safety_cap_gamma=getattr(args, "v7_temporal_safety_cap_gamma", 1.0),
                v7_temporal_safety_cap_detach=getattr(args, "v7_temporal_safety_cap_detach", False),
                v7_use_temporal_mismatch_multihead=getattr(
                    args, "v7_use_temporal_mismatch_multihead", False
                ),
                v7_temporal_mismatch_multihead_cap_mode=getattr(
                    args, "v7_temporal_mismatch_multihead_cap_mode", "none"
                ),
                v7_temporal_mismatch_multihead_cap_gamma=getattr(
                    args, "v7_temporal_mismatch_multihead_cap_gamma", 1.0
                ),
                v7_temporal_mismatch_multihead_cap_detach=getattr(
                    args, "v7_temporal_mismatch_multihead_cap_detach", False
                ),
                v7_temporal_mismatch_multihead_fusion=getattr(
                    args, "v7_temporal_mismatch_multihead_fusion", "frame_only"
                ),
                v7_use_temporal_preservation_head=getattr(
                    args, "v7_use_temporal_preservation_head", False
                ),
                v7_use_temporal_preservation_aware_cap=getattr(
                    args, "v7_use_temporal_preservation_aware_cap", False
                ),
                v7_temporal_preservation_cap_gamma=getattr(
                    args, "v7_temporal_preservation_cap_gamma", 1.0
                ),
                v7_temporal_preservation_cap_detach=getattr(
                    args, "v7_temporal_preservation_cap_detach", False
                ),
            )
        elif args.architecture == "vnext_minimal":
            model = build_vnext_mamba_model(
                args.model_name,
                freeze_encoder=args.freeze_encoder,
                freeze_a_log=args.freeze_a_log,
                vnext_router_mode=args.vnext_router_mode,
                vnext_enable_segmented_dual_pass=args.vnext_enable_segmented_dual_pass,
                vnext_segmented_context_role=args.vnext_segmented_context_role,
                vnext_context_risk_cap_alpha=args.vnext_context_risk_cap_alpha,
                vnext_context_risk_threshold=args.vnext_context_risk_threshold,
                vnext_context_risk_source=args.vnext_context_risk_source,
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

    attach_vnext_segmented_dual_pass_inputs(
        train_inputs,
        train_records,
        args=args,
        vocab=vocab,
        tokenizer=tokenizer,
        device=device,
    )
    attach_vnext_segmented_dual_pass_inputs(
        dev_inputs,
        dev_records,
        args=args,
        vocab=vocab,
        tokenizer=tokenizer,
        device=device,
    )

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
                v7_use_v6b_style_final_decision=args.v7_use_v6b_style_final_decision,
                v7_use_learnable_ne_alpha=args.v7_use_learnable_ne_alpha,
                v7_ne_alpha_init=args.v7_ne_alpha_init,
                v7_h1_entitlement_decision_signal=args.v7_h1_entitlement_decision_signal,
                v7_h1_entitlement_product_power=args.v7_h1_entitlement_product_power,
                v7_h1_hybrid_residual_beta=args.v7_h1_hybrid_residual_beta,
                v7_use_location_boundary_head=getattr(args, "v7_use_location_boundary_head", False),
                v7_location_boundary_cap_mode=getattr(args, "v7_location_boundary_cap_mode", "none"),
                v7_location_boundary_cap_gamma=getattr(args, "v7_location_boundary_cap_gamma", 1.0),
                v7_location_boundary_cap_detach=getattr(args, "v7_location_boundary_cap_detach", False),
                v7_use_temporal_safety_head=getattr(args, "v7_use_temporal_safety_head", False),
                v7_temporal_safety_cap_mode=getattr(args, "v7_temporal_safety_cap_mode", "none"),
                v7_temporal_safety_cap_gamma=getattr(args, "v7_temporal_safety_cap_gamma", 1.0),
                v7_temporal_safety_cap_detach=getattr(args, "v7_temporal_safety_cap_detach", False),
                v7_use_temporal_mismatch_multihead=getattr(
                    args, "v7_use_temporal_mismatch_multihead", False
                ),
                v7_temporal_mismatch_multihead_cap_mode=getattr(
                    args, "v7_temporal_mismatch_multihead_cap_mode", "none"
                ),
                v7_temporal_mismatch_multihead_cap_gamma=getattr(
                    args, "v7_temporal_mismatch_multihead_cap_gamma", 1.0
                ),
                v7_temporal_mismatch_multihead_cap_detach=getattr(
                    args, "v7_temporal_mismatch_multihead_cap_detach", False
                ),
                v7_temporal_mismatch_multihead_fusion=getattr(
                    args, "v7_temporal_mismatch_multihead_fusion", "frame_only"
                ),
                v7_use_temporal_preservation_head=getattr(
                    args, "v7_use_temporal_preservation_head", False
                ),
                v7_use_temporal_preservation_aware_cap=getattr(
                    args, "v7_use_temporal_preservation_aware_cap", False
                ),
                v7_temporal_preservation_cap_gamma=getattr(
                    args, "v7_temporal_preservation_cap_gamma", 1.0
                ),
                v7_temporal_preservation_cap_detach=getattr(
                    args, "v7_temporal_preservation_cap_detach", False
                ),
            )
        elif args.architecture == "vnext_minimal":
            model = build_vnext_model(
                len(vocab),
                max_length,
                vnext_router_mode=args.vnext_router_mode,
                vnext_enable_segmented_dual_pass=args.vnext_enable_segmented_dual_pass,
                vnext_segmented_context_role=args.vnext_segmented_context_role,
                vnext_context_risk_cap_alpha=args.vnext_context_risk_cap_alpha,
                vnext_context_risk_threshold=args.vnext_context_risk_threshold,
                vnext_context_risk_source=args.vnext_context_risk_source,
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
    if getattr(args, "v7_use_coverage_entailment_loss", False):
        if args.architecture != "v7_hierarchical":
            raise ValueError(
                "--v7-use-coverage-entailment-loss requires --architecture v7_hierarchical."
            )
        if not getattr(args, "v7_use_coverage_entailment_head", False):
            raise ValueError(
                "--v7-use-coverage-entailment-loss requires "
                "--v7-use-coverage-entailment-head."
            )
        if getattr(args, "v7_coverage_entailment_data", None) is None:
            raise ValueError(
                "--v7-use-coverage-entailment-loss requires "
                "--v7-coverage-entailment-data."
            )
    if getattr(args, "v7_use_coverage_entailment_head", False):
        if args.architecture != "v7_hierarchical":
            raise ValueError(
                "--v7-use-coverage-entailment-head requires --architecture v7_hierarchical."
            )
        install_stage31c_coverage_entailment_head(
            model,
            num_classes=getattr(args, "v7_coverage_entailment_num_classes", 3),
            input_mode=getattr(args, "v7_coverage_entailment_input_mode", "current"),
            detach_input=getattr(args, "v7_coverage_entailment_detach_input", False),
        )

    model = model.to(device)

    # Stage26-C TODO: add a one-shot contract check for v7 before the training loop.
    # Call validate_v7_output_contract(output) after the first model forward to catch
    # missing keys early.  Not wired here to avoid importing v7 code on v6B runs.
    # Example (in a test or a separate validation script):
    #   from contramamba.modeling_v7_hierarchical import validate_v7_output_contract
    #   with torch.no_grad(): out = model(**model_feature_inputs(dev_inputs))
    #   validate_v7_output_contract(out)

    if args.backbone == "mamba" and args.freeze_encoder and not _vnext_segmented_dual_pass_active(args):
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

    # Temporal diagnostic data loading ??separate dataset, never mixed into main train/dev.
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

    # Stage30-C2: temporal safety data loading ??separate dataset, never mixed into main train/dev.
    # Records contain none/paraphrase (temporal_safe=1) and time_swap (temporal_safe=0).
    # Stage15 OOD records must NOT be present here.
    # Filtering uses already-computed main train/dev pair_ids to prevent controlled-dev leakage.
    # v5.split_by_pair_id is NOT called (it requires 0 < dev_ratio < 1 and is not needed here).
    _ts_train_inputs: "dict[str, torch.Tensor] | None" = None
    _ts_train_labels: "torch.Tensor | None" = None
    _ts_train_mask: "torch.Tensor | None" = None
    # Provenance counts surfaced in summary JSON.
    _ts_meta_records_loaded: int = 0
    _ts_meta_records_used: int = 0
    _ts_meta_excluded_dev: int = 0
    _ts_meta_excluded_missing_pair_id: int = 0

    _ts_data_needed = (
        getattr(args, "v7_use_temporal_safety_loss", False)
        and getattr(args, "v7_temporal_safety_data", None) is not None
    )
    if _ts_data_needed:
        _ts_path = Path(args.v7_temporal_safety_data)
        _ts_all_records = load_temporal_safety_jsonl(_ts_path)
        _ts_meta_records_loaded = len(_ts_all_records)

        # Build pair_id sets from the already-computed main train/dev split.
        # train_records and dev_records are available at this point (line 3561).
        _main_train_pair_ids: set = {
            r["pair_id"] for r in train_records if r.get("pair_id") is not None
        }
        _main_dev_pair_ids: set = {
            r["pair_id"] for r in dev_records if r.get("pair_id") is not None
        }

        # Filter: include only records whose pair_id maps to main train.
        # Exclude records in main dev (controlled-dev leakage prevention).
        # Exclude records with no resolvable pair_id.
        _ts_train_records: list[dict] = []
        for _r in _ts_all_records:
            _pid = _r.get("pair_id") or _r.get("source_pair_id")
            if _pid is None:
                _ts_meta_excluded_missing_pair_id += 1
                continue
            if _pid in _main_dev_pair_ids:
                _ts_meta_excluded_dev += 1
                continue
            if _pid in _main_train_pair_ids:
                _ts_train_records.append(_r)
            # Records whose pair_id is in neither set are silently skipped
            # (should not occur with controlled-derived data; counted via len difference).

        _ts_meta_records_used = len(_ts_train_records)
        if _ts_meta_records_used == 0:
            print(
                f"[ts_head] WARNING: no temporal safety train records after pair_id filtering "
                f"(loaded={_ts_meta_records_loaded} "
                f"excluded_dev={_ts_meta_excluded_dev} "
                f"excluded_missing_pair_id={_ts_meta_excluded_missing_pair_id}). "
                "Loss will be zero."
            )
        else:
            _ts_train_labels, _ts_train_mask = encode_temporal_safety_labels(
                _ts_train_records, device
            )
            if args.backbone == "dummy":
                _ts_train_bundle = v5.encode_records(_ts_train_records, vocab)
            else:
                _ts_train_bundle = v5.encode_mamba_records(
                    _ts_train_records, tokenizer, args.max_length
                )
            _ts_train_inputs = v5.move_inputs(_ts_train_bundle["model_inputs"], device)
            _ts_seq = _ts_train_inputs["input_ids"].shape[1]
            if _ts_seq < max_length:
                _diff = max_length - _ts_seq
                for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _ts_train_inputs[_key] = F.pad(_ts_train_inputs[_key], (0, _diff), value=0)
            elif _ts_seq > max_length:
                for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _ts_train_inputs[_key] = _ts_train_inputs[_key][:, :max_length]
            if args.backbone == "mamba" and args.freeze_encoder:
                v5.cache_frozen_encoder_states(model, _ts_train_inputs)
            _ts_train_pos = int(_ts_train_labels.sum().item())
            _ts_train_neg = int(
                (_ts_train_mask.float() - _ts_train_labels * _ts_train_mask.float()).sum().item()
            )
            print(
                f"[ts_head] enabled weight={getattr(args, 'v7_temporal_safety_loss_weight', 0.0)}"
                f" loaded={_ts_meta_records_loaded}"
                f" used={_ts_meta_records_used}"
                f" excluded_dev={_ts_meta_excluded_dev}"
                f" excluded_missing_pair_id={_ts_meta_excluded_missing_pair_id}"
                f" ts_train_pos={_ts_train_pos}"
                f" ts_train_neg={_ts_train_neg}"
            )

    # Stage30-D: temporal mismatch multihead aux data loading ??separate from main train/dev.
    # Uses the same pair_id filtering as Stage30-C2 to exclude main dev records.
    _tmm_train_inputs: "dict[str, torch.Tensor] | None" = None
    _tmm_train_labels: "torch.Tensor | None" = None
    _tmm_train_mask: "torch.Tensor | None" = None
    _tmm_meta_records_loaded: int = 0
    _tmm_meta_records_used: int = 0
    _tmm_meta_excluded_dev: int = 0
    _tmm_meta_excluded_missing_pair_id: int = 0

    _tmm_data_needed = (
        getattr(args, "v7_use_temporal_mismatch_multihead_loss", False)
        and getattr(args, "v7_temporal_mismatch_multihead_data", None) is not None
    )
    if _tmm_data_needed:
        _tmm_path = Path(args.v7_temporal_mismatch_multihead_data)
        _tmm_all_records = load_temporal_safety_jsonl(_tmm_path)
        _tmm_meta_records_loaded = len(_tmm_all_records)

        _tmm_main_train_pair_ids: set = {
            r["pair_id"] for r in train_records if r.get("pair_id") is not None
        }
        _tmm_main_dev_pair_ids: set = {
            r["pair_id"] for r in dev_records if r.get("pair_id") is not None
        }

        _tmm_train_records: list[dict] = []
        for _r in _tmm_all_records:
            _pid = _r.get("pair_id") or _r.get("source_pair_id")
            if _pid is None:
                _tmm_meta_excluded_missing_pair_id += 1
                continue
            if _pid in _tmm_main_dev_pair_ids:
                _tmm_meta_excluded_dev += 1
                continue
            if _pid in _tmm_main_train_pair_ids:
                _tmm_train_records.append(_r)

        _tmm_meta_records_used = len(_tmm_train_records)
        if _tmm_meta_records_used == 0:
            print(
                f"[tmm_head] WARNING: no temporal mismatch multihead train records after "
                f"pair_id filtering "
                f"(loaded={_tmm_meta_records_loaded} "
                f"excluded_dev={_tmm_meta_excluded_dev} "
                f"excluded_missing_pair_id={_tmm_meta_excluded_missing_pair_id}). "
                "Loss will be zero."
            )
        else:
            _tmm_train_labels, _tmm_train_mask = encode_temporal_mismatch_multihead_labels(
                _tmm_train_records, device
            )
            if args.backbone == "dummy":
                _tmm_train_bundle = v5.encode_records(_tmm_train_records, vocab)
            else:
                _tmm_train_bundle = v5.encode_mamba_records(
                    _tmm_train_records, tokenizer, args.max_length
                )
            _tmm_train_inputs = v5.move_inputs(_tmm_train_bundle["model_inputs"], device)
            _tmm_seq = _tmm_train_inputs["input_ids"].shape[1]
            if _tmm_seq < max_length:
                _diff = max_length - _tmm_seq
                for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _tmm_train_inputs[_key] = F.pad(_tmm_train_inputs[_key], (0, _diff), value=0)
            elif _tmm_seq > max_length:
                for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _tmm_train_inputs[_key] = _tmm_train_inputs[_key][:, :max_length]
            if args.backbone == "mamba" and args.freeze_encoder:
                v5.cache_frozen_encoder_states(model, _tmm_train_inputs)
            _tmm_train_pos = int(_tmm_train_labels.sum().item())
            _tmm_train_neg = int(
                (_tmm_train_mask.float() - _tmm_train_labels * _tmm_train_mask.float()).sum().item()
            )
            print(
                f"[tmm_head] enabled "
                f"weight={getattr(args, 'v7_temporal_mismatch_multihead_loss_weight', 0.0)}"
                f" fusion={getattr(args, 'v7_temporal_mismatch_multihead_fusion', 'frame_only')}"
                f" loaded={_tmm_meta_records_loaded}"
                f" used={_tmm_meta_records_used}"
                f" excluded_dev={_tmm_meta_excluded_dev}"
                f" excluded_missing_pair_id={_tmm_meta_excluded_missing_pair_id}"
                f" tmm_train_pos={_tmm_train_pos}"
                f" tmm_train_neg={_tmm_train_neg}"
            )

    # Stage30-E: temporal preservation aux data loading ??separate from main train/dev.
    # Reuses encode_temporal_safety_labels: none/paraphrase ??label=1, time_swap ??label=0.
    # Same pair_id filtering as Stage30-C2/D to exclude main dev records.
    # Stage15 OOD records must NOT be present here.
    _tpres_train_inputs: "dict[str, torch.Tensor] | None" = None
    _tpres_train_labels: "torch.Tensor | None" = None
    _tpres_train_mask: "torch.Tensor | None" = None
    _tpres_meta_records_loaded: int = 0
    _tpres_meta_records_used: int = 0
    _tpres_meta_excluded_dev: int = 0
    _tpres_meta_excluded_missing_pair_id: int = 0

    _tpres_data_needed = (
        getattr(args, "v7_use_temporal_preservation_loss", False)
        and getattr(args, "v7_temporal_preservation_data", None) is not None
    )
    if _tpres_data_needed:
        _tpres_path = Path(args.v7_temporal_preservation_data)
        _tpres_all_records = load_temporal_safety_jsonl(_tpres_path)
        _tpres_meta_records_loaded = len(_tpres_all_records)

        _tpres_main_train_pair_ids: set = {
            r["pair_id"] for r in train_records if r.get("pair_id") is not None
        }
        _tpres_main_dev_pair_ids: set = {
            r["pair_id"] for r in dev_records if r.get("pair_id") is not None
        }

        _tpres_train_records: list[dict] = []
        for _r in _tpres_all_records:
            _pid = _r.get("pair_id") or _r.get("source_pair_id")
            if _pid is None:
                _tpres_meta_excluded_missing_pair_id += 1
                continue
            if _pid in _tpres_main_dev_pair_ids:
                _tpres_meta_excluded_dev += 1
                continue
            if _pid in _tpres_main_train_pair_ids:
                _tpres_train_records.append(_r)

        _tpres_meta_records_used = len(_tpres_train_records)
        if _tpres_meta_records_used == 0:
            print(
                f"[tpres_head] WARNING: no temporal preservation train records after "
                f"pair_id filtering "
                f"(loaded={_tpres_meta_records_loaded} "
                f"excluded_dev={_tpres_meta_excluded_dev} "
                f"excluded_missing_pair_id={_tpres_meta_excluded_missing_pair_id}). "
                "Loss will be zero."
            )
        else:
            # encode_temporal_safety_labels: safe=1 (none/paraphrase) = preserved=1; time_swap = preserved=0.
            _tpres_train_labels, _tpres_train_mask = encode_temporal_safety_labels(
                _tpres_train_records, device
            )
            if args.backbone == "dummy":
                _tpres_train_bundle = v5.encode_records(_tpres_train_records, vocab)
            else:
                _tpres_train_bundle = v5.encode_mamba_records(
                    _tpres_train_records, tokenizer, args.max_length
                )
            _tpres_train_inputs = v5.move_inputs(_tpres_train_bundle["model_inputs"], device)
            _tpres_seq = _tpres_train_inputs["input_ids"].shape[1]
            if _tpres_seq < max_length:
                _diff = max_length - _tpres_seq
                for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _tpres_train_inputs[_key] = F.pad(_tpres_train_inputs[_key], (0, _diff), value=0)
            elif _tpres_seq > max_length:
                for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _tpres_train_inputs[_key] = _tpres_train_inputs[_key][:, :max_length]
            if args.backbone == "mamba" and args.freeze_encoder:
                v5.cache_frozen_encoder_states(model, _tpres_train_inputs)
            _tpres_train_pos = int(_tpres_train_labels.sum().item())
            _tpres_train_neg = int(
                (
                    _tpres_train_mask.float()
                    - _tpres_train_labels * _tpres_train_mask.float()
                ).sum().item()
            )
            print(
                f"[tpres_head] enabled "
                f"weight={getattr(args, 'v7_temporal_preservation_loss_weight', 0.0)}"
                f" loaded={_tpres_meta_records_loaded}"
                f" used={_tpres_meta_records_used}"
                f" excluded_dev={_tpres_meta_excluded_dev}"
                f" excluded_missing_pair_id={_tpres_meta_excluded_missing_pair_id}"
                f" tpres_train_pos={_tpres_train_pos}"
                f" tpres_train_neg={_tpres_train_neg}"
            )

    # Stage31-C: coverage/entailment directional aux data loading.
    # Separate train/dev split from main controlled data. The Stage31-A/B evaluation
    # probe is explicitly rejected by load_stage31c_coverage_entailment_jsonl.
    _covent_train_inputs: "dict[str, torch.Tensor] | None" = None
    _covent_dev_inputs: "dict[str, torch.Tensor] | None" = None
    _covent_train_labels: "torch.Tensor | None" = None
    _covent_dev_labels: "torch.Tensor | None" = None
    _covent_meta_records_loaded: int = 0
    _covent_meta_train_records: int = 0
    _covent_meta_dev_records: int = 0
    _covent_class_weights: "torch.Tensor | None" = None

    _covent_data_needed = (
        getattr(args, "v7_use_coverage_entailment_loss", False)
        and getattr(args, "v7_coverage_entailment_data", None) is not None
    )
    if _covent_data_needed:
        _covent_path = Path(args.v7_coverage_entailment_data)
        _covent_all_records = load_stage31c_coverage_entailment_jsonl(_covent_path)
        _covent_train_records = [
            r for r in _covent_all_records if r.get("split") == "train"
        ]
        _covent_dev_records = [
            r for r in _covent_all_records if r.get("split") == "dev"
        ]
        _covent_meta_records_loaded = len(_covent_all_records)
        _covent_meta_train_records = len(_covent_train_records)
        _covent_meta_dev_records = len(_covent_dev_records)
        if not _covent_train_records:
            raise ValueError("[stage31c] no train records found in coverage-entailment aux file.")
        if not _covent_dev_records:
            raise ValueError("[stage31c] no dev records found in coverage-entailment aux file.")

        _covent_train_labels = encode_stage31c_coverage_entailment_labels(
            _covent_train_records,
            device,
            getattr(args, "v7_coverage_entailment_num_classes", 3),
        )
        _covent_dev_labels = encode_stage31c_coverage_entailment_labels(
            _covent_dev_records,
            device,
            getattr(args, "v7_coverage_entailment_num_classes", 3),
        )
        _covent_class_weights = parse_stage31c_class_weights(
            getattr(args, "v7_coverage_entailment_loss_class_weights", None),
            device,
            getattr(args, "v7_coverage_entailment_num_classes", 3),
        )

        if args.backbone == "dummy":
            _covent_train_bundle = v5.encode_records(_covent_train_records, vocab)
            _covent_dev_bundle = v5.encode_records(_covent_dev_records, vocab)
        else:
            _covent_train_bundle = v5.encode_mamba_records(
                _covent_train_records, tokenizer, args.max_length
            )
            _covent_dev_bundle = v5.encode_mamba_records(
                _covent_dev_records, tokenizer, args.max_length
            )
        _covent_train_inputs = v5.move_inputs(_covent_train_bundle["model_inputs"], device)
        _covent_dev_inputs = v5.move_inputs(_covent_dev_bundle["model_inputs"], device)
        for _covent_inputs in (_covent_train_inputs, _covent_dev_inputs):
            _covent_seq = _covent_inputs["input_ids"].shape[1]
            if _covent_seq < max_length:
                _diff = max_length - _covent_seq
                for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _covent_inputs[_key] = F.pad(_covent_inputs[_key], (0, _diff), value=0)
            elif _covent_seq > max_length:
                for _key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _covent_inputs[_key] = _covent_inputs[_key][:, :max_length]
        if args.backbone == "mamba" and args.freeze_encoder:
            v5.cache_frozen_encoder_states(model, _covent_train_inputs)
            v5.cache_frozen_encoder_states(model, _covent_dev_inputs)
        print(
            f"[stage31c_covent] enabled "
            f"weight={getattr(args, 'v7_coverage_entailment_loss_weight', 0.0)}"
            f" loaded={_covent_meta_records_loaded}"
            f" train={_covent_meta_train_records}"
            f" dev={_covent_meta_dev_records}"
            f" num_classes={getattr(args, 'v7_coverage_entailment_num_classes', 3)}"
            f" input_mode={getattr(args, 'v7_coverage_entailment_input_mode', 'current')}"
            f" detach_input={getattr(args, 'v7_coverage_entailment_detach_input', False)}"
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
        # Stage71: per-train-row source tags ("clean_main" / "stage57_bridge" /
        # "stage66_bridge"), aligned 1:1 with train_records/train_bundle/
        # train_inputs. When any bridge rows are present, intervention_pairwise_losses
        # is restricted to the "clean_main" rows only (bridge rows lack the
        # intervention_type=="none" original record required per pair_id).
        # None means no bridge rows were appended -- behavior is unchanged.
        train_source_labels=None,
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
        # Stage44-B: internal-only anti-collapse checkpoint selection (default off)
        stage44_use_anti_collapse_selection=False,
        stage44_min_support_recall=None,
        stage44_min_refute_recall=None,
        stage44_max_not_entitled_pred_rate=None,
        stage44_min_clean_dev_accuracy=None,
        stage44_selection_fallback="best_metric",
        stage44_use_prior_aware_ne_constraint=False,
        stage44_max_ne_gold_prior_delta=None,
        stage44_min_macro_f1=None,
        stage44_min_relative_macro_f1_of_best=None,
        stage44_min_support_precision=None,
        stage44_min_refute_precision=None,
        stage45b_split_info=None,
        # Stage45-C: internal-only SUPPORT entitlement recovery scaffold (default off)
        stage45c_enabled=False,
        stage45c_support_recovery_weight=0.0,
        stage45c_entitled_ne_penalty_weight=0.0,
        stage45c_target_label="SUPPORT",
        stage45c_entitled_labels=("SUPPORT", "REFUTE"),
        # Stage30-C2: temporal safety auxiliary BCE loss (separate dataset; v7 only)
        ts_train_inputs=None,
        ts_train_labels=None,
        ts_train_mask=None,
        ts_loss_weight=0.0,
        ts_loss_pos_weight=None,
        # Stage30-D: temporal mismatch multihead auxiliary BCE loss (separate dataset; v7 only)
        tmm_train_inputs=None,
        tmm_train_labels=None,
        tmm_train_mask=None,
        tmm_loss_weight=0.0,
        tmm_loss_pos_weight=None,
        # Stage30-E: temporal preservation head auxiliary BCE loss (separate dataset; v7 only)
        tpres_train_inputs=None,
        tpres_train_labels=None,
        tpres_train_mask=None,
        tpres_loss_weight=0.0,
        tpres_loss_pos_weight=None,
        # Stage31-C: coverage/entailment directional auxiliary CE loss
        covent_train_inputs=None,
        covent_dev_inputs=None,
        covent_train_labels=None,
        covent_dev_labels=None,
        covent_loss_weight=0.0,
        covent_class_weights=None,
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
        if stage44_use_anti_collapse_selection and (
            use_preservation_constrained_selection or use_td_constrained_selection
        ):
            raise ValueError(
                "--stage44-use-anti-collapse-selection cannot be combined with "
                "--use-preservation-constrained-selection or --use-td-constrained-selection. "
                "Enable only one post-epoch checkpoint selector per run."
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
        best_covent_metrics: dict[str, Any] = {}
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
        _stage45b_split_info = dict(stage45b_split_info or {})
        # Stage45-B2: diagnostics from v5.intervention_objective's internal-family-holdout
        # guard, refreshed each epoch when the ranking-objective branch runs.
        _stage45b2_intervention_diag: "dict[str, Any] | None" = None
        # Stage45-C: internal-only auxiliary loss diagnostics, refreshed each epoch.
        _stage45c_entitled_labels_tuple = tuple(
            str(name).strip() for name in stage45c_entitled_labels if str(name).strip()
        )
        _stage45c_diag: "dict[str, Any] | None" = None
        if stage45c_enabled:
            _stage45c_train_label_counts = stage45c_label_counts(
                train_inputs["final_labels"], v5.ID_TO_FINAL_LABEL
            )
        else:
            _stage45c_train_label_counts = {}

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

        # Stage44-B anti-collapse selection tracking (internal clean-dev only).
        _stage44_constraints: dict[str, Any] = {
            "min_support_recall": stage44_min_support_recall,
            "min_refute_recall": stage44_min_refute_recall,
            "max_not_entitled_pred_rate": stage44_max_not_entitled_pred_rate,
            "min_clean_dev_accuracy": stage44_min_clean_dev_accuracy,
            "use_prior_aware_ne_constraint": stage44_use_prior_aware_ne_constraint,
            "max_ne_gold_prior_delta": stage44_max_ne_gold_prior_delta,
            "min_macro_f1": stage44_min_macro_f1,
            "min_relative_macro_f1_of_best": stage44_min_relative_macro_f1_of_best,
            "min_support_precision": stage44_min_support_precision,
            "min_refute_precision": stage44_min_refute_precision,
        }
        _stage44_candidate_table: list[dict[str, Any]] = []
        _stage44_epoch: int = -1
        _stage44_score: float = float("-inf")
        _stage44_state: "dict[str, torch.Tensor] | None" = None
        _stage44_dev_metrics: "dict | None" = None
        _stage44_dev_interventions: "dict | None" = None
        _stage44_dev_pairwise_checks: "dict | None" = None
        _stage44_dev_predictions: "list | None" = None
        _stage44_pc_metrics: dict[str, Any] = {}
        _stage44_selected_metrics: dict[str, Any] | None = None
        _stage44_satisfying_count: int = 0
        _stage44_original_best_epoch: int | None = None
        _stage44_original_best_metrics: dict[str, Any] | None = None

        #  Audit ledger: loss accumulators (reporting only; no effect on training) 
        # Per-epoch raw (pre-weight) and weighted (actual contribution to total_loss) loss values.
        # Both lists grow one entry per epoch and are indexed by epoch-1 at ledger build time.
        _audit_per_epoch_raw: list[dict] = []
        _audit_per_epoch_weighted: list[dict] = []
        _audit_epoch_count: int = 0

        #  Stage26-F: v7 epoch diagnostic history 
        # Per-epoch dev metric snapshots stored for post-hoc diagnosis (e.g. label collapse
        # trajectory, channel prob trends).  Reporting only; no effect on training or selection.
        _v7_epoch_history: list[dict[str, Any]] = []

        # Stage26-F extended: captured v7 output tensors for best-epoch logit / per-gold summaries.
        # Updated inside the epoch loop whenever score > best_score.
        # Note: if TD/PCS constrained selection overrides best_epoch after the loop, the logit
        # summary may be from the unconstrained best epoch rather than the checkpointed one.
        # Collapse / recall fields are always from best_dev_metrics (correctly overridden).
        _best_dev_output_v7: "dict[str, Any] | None" = None

        # Stage71: precompute the clean-main-only view used for pairwise
        # intervention loss. Stage57/Stage66 bridge rows are appended to
        # train_records/train_inputs (so they still get CE/classification and
        # every other per-row loss below), but they are standalone examples
        # with no intervention_type=="none" original record for their
        # pair_id, so passing them into intervention_pairwise_losses raises
        # ValueError (pair_id has no original ('none') record). When
        # train_source_labels is None or contains no bridge rows, this is a
        # no-op and pairwise loss behavior is byte-for-byte unchanged.
        _pw_output_keys = (
            "frame_logit", "predicate_coverage_logit", "sufficiency_logit",
            "polarity_margin", "entitlement_prob", "logits",
        )
        _pw_clean_main_index_tensor: "torch.Tensor | None" = None
        _pw_pair_ids = train_bundle["pair_ids"]
        _pw_intervention_types = train_bundle["intervention_types"]
        _pw_final_labels = train_inputs["final_labels"]
        if train_source_labels is not None:
            _pw_clean_indices = [
                _i for _i, _src in enumerate(train_source_labels) if _src == "clean_main"
            ]
            if len(_pw_clean_indices) != len(train_source_labels):
                _pw_clean_main_index_tensor = torch.tensor(
                    _pw_clean_indices, dtype=torch.long, device=device
                )
                _pw_pair_ids = [train_bundle["pair_ids"][_i] for _i in _pw_clean_indices]
                _pw_intervention_types = [
                    train_bundle["intervention_types"][_i] for _i in _pw_clean_indices
                ]
                _pw_final_labels = train_inputs["final_labels"].index_select(
                    0, _pw_clean_main_index_tensor
                )

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            # CRITICAL: Pass flags to v6b model forward
            _ta_pen = ta_final_penalty_scale if use_temporal_residual_adapter and ta_final_penalty_scale > 0.0 else 0.0
            _tc_pen = tc_gated_penalty_scale if use_temporal_channel_gated_penalty and tc_gated_penalty_scale > 0.0 else 0.0
            output = model(
                **_vnext_model_feature_inputs(train_inputs),
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

                # Pairwise losses consume output["logits"] (final logits from v6b).
                # Stage71: when bridge rows are present, _pw_* is restricted to the
                # clean-main-row subset of this epoch's fresh `output` tensors (see
                # _pw_clean_main_index_tensor precomputed above the epoch loop);
                # otherwise _pw_* equals the original full-batch arguments unchanged.
                _pw_output = (
                    {
                        key: output[key].index_select(0, _pw_clean_main_index_tensor)
                        for key in _pw_output_keys
                    }
                    if _pw_clean_main_index_tensor is not None
                    else output
                )
                pairwise_losses = intervention_pairwise_losses(
                    _pw_output,
                    _pw_pair_ids,
                    _pw_intervention_types,
                    _pw_final_labels,
                    **loss_config,
                )
                active_intervention_loss = pairwise_losses["total"]
            else:
                active_intervention_loss = (
                    ranking_weight * v5.intervention_objective(output, train_records)
                )
                _stage45b2_intervention_diag = getattr(
                    v5.intervention_objective, "last_diagnostics", None
                )
            total_loss = losses["total"] + active_intervention_loss

            # Stage45-C: internal-only SUPPORT recovery / entitled-NE over-rejection
            # auxiliary terms. Computed strictly from the internal training split
            # (train_inputs["final_labels"], output["logits"]); no dev/holdout labels
            # or external data are read. Inactive (zero-valued, gradient-connected)
            # unless explicitly enabled with a positive weight.
            if stage45c_enabled and (
                stage45c_support_recovery_weight > 0.0
                or stage45c_entitled_ne_penalty_weight > 0.0
            ):
                _stage45c_terms = compute_support_recovery_terms(
                    output["logits"],
                    train_inputs["final_labels"],
                    label_to_id=v5.FINAL_LABEL_TO_ID,
                    target_label=stage45c_target_label,
                    entitled_labels=_stage45c_entitled_labels_tuple,
                )
                if stage45c_support_recovery_weight > 0.0:
                    total_loss = total_loss + (
                        stage45c_support_recovery_weight
                        * _stage45c_terms["support_recovery_loss"]
                    )
                if stage45c_entitled_ne_penalty_weight > 0.0:
                    total_loss = total_loss + (
                        stage45c_entitled_ne_penalty_weight
                        * _stage45c_terms["entitled_ne_penalty_loss"]
                    )
                _stage45c_diag = {
                    "support_recovery_loss_mean": _stage45c_terms[
                        "support_recovery_loss"
                    ].item(),
                    "entitled_ne_penalty_loss_mean": _stage45c_terms[
                        "entitled_ne_penalty_loss"
                    ].item(),
                    "support_recovery_active": _stage45c_terms["support_recovery_active"],
                    "entitled_ne_penalty_active": _stage45c_terms["entitled_ne_penalty_active"],
                }

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
                    **_vnext_model_feature_inputs(td_train_inputs),
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
                        **_vnext_model_feature_inputs(td_train_inputs),
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
                        **_vnext_model_feature_inputs(td_train_inputs),
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
                    **_vnext_model_feature_inputs(pc_pres_inputs),
                    temporal_mismatch_flags=_pc_zero_t,
                    predicate_mismatch_flags=_pc_zero_p,
                )
                _pc_frame_out = model(
                    **_vnext_model_feature_inputs(pc_frame_inputs),
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

            #  Stage26-G: v7 polarity margin auxiliary loss 
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

            #  Stage26-G: v7 entitlement BCE auxiliary loss 
            # BCE on v7_entitlement_logit with ground-truth entitled target derived from labels:
            #   entitled=1 for SUPPORT (2) and REFUTE (0) ??these require a polarity judgment
            #   entitled=0 for NOT_ENTITLED (1) ??evidence fails to entitle a polarity judgment
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
                    # SUPPORT=2 and REFUTE=0 ??entitled=1; NOT_ENTITLED=1 ??entitled=0
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

            #  Stage26-G: v7 entitled class-balanced CE 
            # Auxiliary CE over SUPPORT/REFUTE examples only, using v7_polarity_logits [B, 2].
            # Local labels: REFUTE (gold=0) ??local 0; SUPPORT (gold=2) ??local 1.
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
                        # REFUTE (gold=0) ??local 0; SUPPORT (gold=2) ??local 1
                        _ecb_local_labels = (_ecb_labels[_ecb_mask] == 2).long()
                        _v7_ecb_loss = F.cross_entropy(_ecb_pol_sub, _ecb_local_labels)
                        total_loss = total_loss + args.v7_entitled_class_balanced_ce_weight * _v7_ecb_loss

            #  Stage28-I-A: v7 location boundary BCE auxiliary loss 
            # Target: 0 for location_swap; 1 for none/paraphrase/polarity_flip.
            # All other intervention types are excluded from this loss.
            # Stage15/OOD is not used for this loss or target selection.
            # Zero masked records ??loss is zero; no crash.
            _v7_lb_loss = torch.tensor(0.0, device=device)
            if (
                args.architecture == "v7_hierarchical"
                and getattr(args, "v7_use_location_boundary_head", False)
                and getattr(args, "v7_use_location_boundary_loss", False)
                and getattr(args, "v7_location_boundary_loss_weight", 0.0) > 0.0
                and not args.v7_no_aux_losses
            ):
                _lb_logit = output.get("location_boundary_logit")
                if _lb_logit is not None:
                    _lb_indices: list[int] = []
                    _lb_targets: list[float] = []
                    for _lb_i, _lb_rec in enumerate(train_records):
                        _lb_itype = _s28e_normalize_intervention(_lb_rec)
                        if _lb_itype == "location_swap":
                            _lb_indices.append(_lb_i)
                            _lb_targets.append(0.0)
                        elif _lb_itype in ("none", "paraphrase", "polarity_flip"):
                            _lb_indices.append(_lb_i)
                            _lb_targets.append(1.0)
                    if _lb_indices:
                        _lb_idx_t = torch.tensor(_lb_indices, dtype=torch.long, device=device)
                        _lb_tgt_t = torch.tensor(
                            _lb_targets, dtype=_lb_logit.dtype, device=device
                        )
                        _lb_logit_sub = _lb_logit[_lb_idx_t]
                        _v7_lb_loss = F.binary_cross_entropy_with_logits(
                            _lb_logit_sub, _lb_tgt_t
                        )
                        total_loss = (
                            total_loss
                            + args.v7_location_boundary_loss_weight * _v7_lb_loss
                        )

            #  Stage30-C2: v7 temporal safety BCE auxiliary loss 
            # Separate forward pass on temporal safety diagnostic records.
            # Records are NOT in the main clean train/dev; no classification CE here.
            # Stage15/OOD is eval-only and is NEVER used here.
            _v7_ts_loss = torch.tensor(0.0, device=device)
            if (
                args.architecture == "v7_hierarchical"
                and getattr(args, "v7_use_temporal_safety_head", False)
                and getattr(args, "v7_use_temporal_safety_loss", False)
                and ts_loss_weight > 0.0
                and not args.v7_no_aux_losses
                and ts_train_inputs is not None
                and ts_train_labels is not None
                and ts_train_mask is not None
            ):
                _ts_n = ts_train_inputs["input_ids"].shape[0]
                _ts_zero_t = torch.zeros(_ts_n, dtype=torch.float32, device=device)
                _ts_zero_p = torch.zeros(_ts_n, dtype=torch.float32, device=device)
                _ts_train_out = model(
                    **_vnext_model_feature_inputs(ts_train_inputs),
                    temporal_mismatch_flags=_ts_zero_t,
                    predicate_mismatch_flags=_ts_zero_p,
                )
                _ts_logit = _ts_train_out.get("temporal_safety_logit")
                if _ts_logit is not None:
                    _active_ts = ts_train_mask.bool()
                    if torch.any(_active_ts):
                        if ts_loss_pos_weight is not None:
                            _ts_pos_w = torch.tensor(
                                ts_loss_pos_weight, dtype=_ts_logit.dtype, device=device
                            )
                            _v7_ts_loss = F.binary_cross_entropy_with_logits(
                                _ts_logit[_active_ts],
                                ts_train_labels[_active_ts],
                                pos_weight=_ts_pos_w,
                            )
                        else:
                            _v7_ts_loss = F.binary_cross_entropy_with_logits(
                                _ts_logit[_active_ts],
                                ts_train_labels[_active_ts],
                            )
                        total_loss = total_loss + ts_loss_weight * _v7_ts_loss

            #  Stage30-D: v7 temporal mismatch multihead BCE auxiliary loss 
            # Separate forward pass on temporal mismatch diagnostic records.
            # Records are NOT in the main clean train/dev; no classification CE here.
            # Stage15/OOD is eval-only and is NEVER used here.
            # Mean of the three per-head BCE losses for scale stability.
            _v7_tmm_loss = torch.tensor(0.0, device=device)
            if (
                args.architecture == "v7_hierarchical"
                and getattr(args, "v7_use_temporal_mismatch_multihead", False)
                and getattr(args, "v7_use_temporal_mismatch_multihead_loss", False)
                and tmm_loss_weight > 0.0
                and not args.v7_no_aux_losses
                and tmm_train_inputs is not None
                and tmm_train_labels is not None
                and tmm_train_mask is not None
            ):
                _tmm_n = tmm_train_inputs["input_ids"].shape[0]
                _tmm_zero_t = torch.zeros(_tmm_n, dtype=torch.float32, device=device)
                _tmm_zero_p = torch.zeros(_tmm_n, dtype=torch.float32, device=device)
                _tmm_train_out = model(
                    **_vnext_model_feature_inputs(tmm_train_inputs),
                    temporal_mismatch_flags=_tmm_zero_t,
                    predicate_mismatch_flags=_tmm_zero_p,
                )
                _active_tmm = tmm_train_mask.bool()
                if torch.any(_active_tmm):
                    _tmm_head_logit_keys = [
                        "temporal_frame_mismatch_logit",
                        "temporal_predicate_mismatch_logit",
                        "temporal_sufficiency_mismatch_logit",
                    ]
                    _tmm_per_head_losses: list[torch.Tensor] = []
                    for _hk in _tmm_head_logit_keys:
                        _h_logit = _tmm_train_out.get(_hk)
                        if _h_logit is not None:
                            if tmm_loss_pos_weight is not None:
                                _tmm_pw = torch.tensor(
                                    tmm_loss_pos_weight, dtype=_h_logit.dtype, device=device
                                )
                                _h_loss = F.binary_cross_entropy_with_logits(
                                    _h_logit[_active_tmm],
                                    tmm_train_labels[_active_tmm],
                                    pos_weight=_tmm_pw,
                                )
                            else:
                                _h_loss = F.binary_cross_entropy_with_logits(
                                    _h_logit[_active_tmm],
                                    tmm_train_labels[_active_tmm],
                                )
                            _tmm_per_head_losses.append(_h_loss)
                    if _tmm_per_head_losses:
                        # Mean for scale stability (3 heads ??3 scale without mean)
                        _v7_tmm_loss = sum(_tmm_per_head_losses) / len(_tmm_per_head_losses)
                        total_loss = total_loss + tmm_loss_weight * _v7_tmm_loss

            #  Stage30-E: v7 temporal preservation head BCE auxiliary loss 
            # Separate forward pass on temporal preservation diagnostic records.
            # Labels: 1 = preserved (none/paraphrase), 0 = not preserved (time_swap).
            # Stage15/OOD data must not be in the data file.
            _v7_tpres_loss = torch.tensor(0.0, device=device)
            if (
                args.architecture == "v7_hierarchical"
                and getattr(args, "v7_use_temporal_preservation_head", False)
                and getattr(args, "v7_use_temporal_preservation_loss", False)
                and tpres_loss_weight > 0.0
                and not args.v7_no_aux_losses
                and tpres_train_inputs is not None
                and tpres_train_labels is not None
                and tpres_train_mask is not None
            ):
                _tpres_n = tpres_train_inputs["input_ids"].shape[0]
                _tpres_zero_t = torch.zeros(_tpres_n, dtype=torch.float32, device=device)
                _tpres_zero_p = torch.zeros(_tpres_n, dtype=torch.float32, device=device)
                _tpres_train_out = model(
                    **_vnext_model_feature_inputs(tpres_train_inputs),
                    temporal_mismatch_flags=_tpres_zero_t,
                    predicate_mismatch_flags=_tpres_zero_p,
                )
                _tpres_logit = _tpres_train_out.get("temporal_preservation_logit")
                if _tpres_logit is not None:
                    _active_tpres = tpres_train_mask.bool()
                    if torch.any(_active_tpres):
                        if tpres_loss_pos_weight is not None:
                            _tpres_pw = torch.tensor(
                                tpres_loss_pos_weight, dtype=_tpres_logit.dtype, device=device
                            )
                            _v7_tpres_loss = F.binary_cross_entropy_with_logits(
                                _tpres_logit[_active_tpres],
                                tpres_train_labels[_active_tpres],
                                pos_weight=_tpres_pw,
                            )
                        else:
                            _v7_tpres_loss = F.binary_cross_entropy_with_logits(
                                _tpres_logit[_active_tpres],
                                tpres_train_labels[_active_tpres],
                            )
                        total_loss = total_loss + tpres_loss_weight * _v7_tpres_loss

            # Stage31-C: directional Coverage/Entailment diagnostic CE loss.
            # Separate aux batch; never changes output["logits"] and never uses the
            # Stage31-A/B evaluation probe.
            _v7_covent_loss = torch.tensor(0.0, device=device)
            if (
                args.architecture == "v7_hierarchical"
                and getattr(args, "v7_use_coverage_entailment_head", False)
                and getattr(args, "v7_use_coverage_entailment_loss", False)
                and covent_loss_weight > 0.0
                and not args.v7_no_aux_losses
                and covent_train_inputs is not None
                and covent_train_labels is not None
            ):
                _covent_n = covent_train_inputs["input_ids"].shape[0]
                _covent_zero_t = torch.zeros(_covent_n, dtype=torch.float32, device=device)
                _covent_zero_p = torch.zeros(_covent_n, dtype=torch.float32, device=device)
                _covent_train_out = model(
                    **_vnext_model_feature_inputs(covent_train_inputs),
                    temporal_mismatch_flags=_covent_zero_t,
                    predicate_mismatch_flags=_covent_zero_p,
                )
                _covent_logits = _covent_train_out.get("coverage_entailment_logits")
                if _covent_logits is not None:
                    _v7_covent_loss = F.cross_entropy(
                        _covent_logits,
                        covent_train_labels,
                        weight=covent_class_weights,
                    )
                    total_loss = total_loss + covent_loss_weight * _v7_covent_loss

            #  Audit ledger accumulation (reporting only; does not affect gradients) 
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
            _ep_v7_lb_raw = float(_v7_lb_loss.item()) if hasattr(_v7_lb_loss, "item") else 0.0
            _ep_v7_ts_raw = float(_v7_ts_loss.item()) if hasattr(_v7_ts_loss, "item") else 0.0
            _ep_v7_tmm_raw = float(_v7_tmm_loss.item()) if hasattr(_v7_tmm_loss, "item") else 0.0
            _ep_v7_tpres_raw = float(_v7_tpres_loss.item()) if hasattr(_v7_tpres_loss, "item") else 0.0
            _ep_v7_covent_raw = float(_v7_covent_loss.item()) if hasattr(_v7_covent_loss, "item") else 0.0
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
                # Stage28-I-A: location boundary loss (0.0 when disabled)
                "v7_location_boundary_loss": _ep_v7_lb_raw,
                # Stage30-C2: temporal safety loss (0.0 when disabled)
                "v7_temporal_safety_loss": _ep_v7_ts_raw,
                # Stage30-D: temporal mismatch multihead loss (0.0 when disabled)
                "v7_temporal_mismatch_multihead_loss": _ep_v7_tmm_raw,
                # Stage30-E: temporal preservation loss (0.0 when disabled)
                "v7_temporal_preservation_loss": _ep_v7_tpres_raw,
                # Stage31-C: coverage/entailment diagnostic loss (0.0 when disabled)
                "v7_coverage_entailment_loss": _ep_v7_covent_raw,
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
                # Stage28-I-A: location boundary loss weighted contribution
                "v7_location_boundary_loss": (
                    getattr(args, "v7_location_boundary_loss_weight", 0.0) * _ep_v7_lb_raw
                ),
                # Stage30-C2: temporal safety loss weighted contribution
                "v7_temporal_safety_loss": ts_loss_weight * _ep_v7_ts_raw,
                # Stage30-D: temporal mismatch multihead loss weighted contribution
                "v7_temporal_mismatch_multihead_loss": tmm_loss_weight * _ep_v7_tmm_raw,
                # Stage30-E: temporal preservation loss weighted contribution
                "v7_temporal_preservation_loss": tpres_loss_weight * _ep_v7_tpres_raw,
                # Stage31-C: coverage/entailment diagnostic loss weighted contribution
                "v7_coverage_entailment_loss": covent_loss_weight * _ep_v7_covent_raw,
                "total_loss": _ep_total,
            })
            _audit_epoch_count += 1
            #  end audit accumulation 

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
                        **_vnext_model_feature_inputs(td_train_inputs),
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
                        **_vnext_model_feature_inputs(td_dev_inputs),
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

            _covent_train_metrics: dict[str, Any] = {}
            _covent_dev_metrics: dict[str, Any] = {}
            if (
                covent_train_inputs is not None
                and covent_dev_inputs is not None
                and covent_train_labels is not None
                and covent_dev_labels is not None
            ):
                with torch.no_grad():
                    _covent_train_n = covent_train_inputs["input_ids"].shape[0]
                    _covent_train_zero_t = torch.zeros(
                        _covent_train_n, dtype=torch.float32, device=device
                    )
                    _covent_train_zero_p = torch.zeros(
                        _covent_train_n, dtype=torch.float32, device=device
                    )
                    _covent_train_eval_out = model(
                        **_vnext_model_feature_inputs(covent_train_inputs),
                        temporal_mismatch_flags=_covent_train_zero_t,
                        predicate_mismatch_flags=_covent_train_zero_p,
                    )
                    _covent_dev_n = covent_dev_inputs["input_ids"].shape[0]
                    _covent_dev_zero_t = torch.zeros(
                        _covent_dev_n, dtype=torch.float32, device=device
                    )
                    _covent_dev_zero_p = torch.zeros(
                        _covent_dev_n, dtype=torch.float32, device=device
                    )
                    _covent_dev_eval_out = model(
                        **_vnext_model_feature_inputs(covent_dev_inputs),
                        temporal_mismatch_flags=_covent_dev_zero_t,
                        predicate_mismatch_flags=_covent_dev_zero_p,
                    )
                _covent_train_metrics = compute_stage31c_coverage_entailment_metrics(
                    _covent_train_eval_out, covent_train_labels
                )
                _covent_dev_metrics = compute_stage31c_coverage_entailment_metrics(
                    _covent_dev_eval_out, covent_dev_labels
                )
                train_metrics["coverage_entailment_aux_train_accuracy"] = (
                    _covent_train_metrics.get("accuracy")
                )
                train_metrics["coverage_entailment_aux_train_macro_f1"] = (
                    _covent_train_metrics.get("macro_f1")
                )
                dev_metrics["coverage_entailment_aux_dev_accuracy"] = (
                    _covent_dev_metrics.get("accuracy")
                )
                dev_metrics["coverage_entailment_aux_dev_macro_f1"] = (
                    _covent_dev_metrics.get("macro_f1")
                )

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
                    **_vnext_model_feature_inputs(pc_pres_inputs),
                    temporal_mismatch_flags=_pc_zero_t,
                    predicate_mismatch_flags=_pc_zero_p,
                )
                _pc_frame_eval = model(
                    **_vnext_model_feature_inputs(pc_frame_inputs),
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
                best_dev_predictions = prediction_records_v6b(
                    dev_records,
                    dev_output,
                    stage32_owner_state_export=getattr(
                        args, "stage32_owner_state_export", False
                    ),
                    stage32_owner_state_shadow_mode=getattr(
                        args, "stage32_owner_state_shadow_mode", False
                    ),
                    stage32_coverage_owner_v2=getattr(
                        args, "stage32_coverage_owner_v2", False
                    ),
                    stage32_coverage_owner_v2_min_confidence=getattr(
                        args, "stage32_coverage_owner_v2_min_confidence", 0.50
                    ),
                    stage32_coverage_owner_v2_min_margin=getattr(
                        args, "stage32_coverage_owner_v2_min_margin", 0.05
                    ),
                    stage32_coverage_owner_v2_allow_abstain=getattr(
                        args, "stage32_coverage_owner_v2_allow_abstain", False
                    ),
                    stage33_structured_coverage_owner=getattr(
                        args, "stage33_use_structured_coverage_owner", False
                    ),
                    stage33_structured_coverage_owner_export=getattr(
                        args, "stage33_structured_coverage_owner_export", False
                    ),
                    stage33_structured_coverage_owner_shadow_mode=getattr(
                        args, "stage33_structured_coverage_owner_shadow_mode", False
                    ),
                    stage33_structured_coverage_preserve_can_support=getattr(
                        args, "stage33_structured_coverage_preserve_can_support", False
                    ),
                    stage33_structured_coverage_direct_support_rules=getattr(
                        args,
                        "stage33_structured_coverage_direct_support_rules",
                        _STAGE33_DEFAULT_DIRECT_SUPPORT_RULES,
                    ),
                    stage33_structured_coverage_disable_specific_general_direct_support=getattr(
                        args,
                        "stage33_structured_coverage_disable_specific_general_direct_support",
                        False,
                    ),
                    stage33_structured_coverage_weak_rules_to_residual=getattr(
                        args, "stage33_structured_coverage_weak_rules_to_residual", ""
                    ),
                    stage33_structured_coverage_conditional_fallback=getattr(
                        args, "stage33_structured_coverage_conditional_fallback", False
                    ),
                    stage33_structured_coverage_fallback_source=getattr(
                        args, "stage33_structured_coverage_fallback_source", "current_final"
                    ),
                    stage33_structured_coverage_enable_whole_part_rules=getattr(
                        args, "stage33_structured_coverage_enable_whole_part_rules", False
                    ),
                    stage33_structured_coverage_whole_part_direct_support=getattr(
                        args, "stage33_structured_coverage_whole_part_direct_support", False
                    ),
                    stage33_structured_coverage_whole_part_lexicon=getattr(
                        args, "stage33_structured_coverage_whole_part_lexicon", ""
                    ),
                    stage33_structured_coverage_whole_part_v2=getattr(
                        args, "stage33_structured_coverage_whole_part_v2", False
                    ),
                    stage33_structured_coverage_whole_part_v2_use_expanded_lexicon=getattr(
                        args,
                        "stage33_structured_coverage_whole_part_v2_use_expanded_lexicon",
                        False,
                    ),
                    stage33_structured_coverage_whole_part_v2_direct_support_policy=getattr(
                        args,
                        "stage33_structured_coverage_whole_part_v2_direct_support_policy",
                        "hard_core_required",
                    ),
                    stage33_whole_part_conditional_safe_overrides_hard_core=getattr(
                        args,
                        "stage33_whole_part_conditional_safe_overrides_hard_core",
                        False,
                    ),
                    stage36_support_safety_config=_stage36_config_from_args(args),
                    stage37_safe_support_recovery_config=_stage37_config_from_args(args),
                    args=args,
                )
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_pc_metrics = {
                    **_pc_eval_metrics,
                    "pair_contrastive_use_case": args.pair_contrastive_use_case,
                    "count_by_preservation_construction_type": _pc_pres_type_counts,
                    "count_by_frame_construction_type": _pc_frame_type_counts,
                }
                best_covent_metrics = {
                    "coverage_entailment_aux_train_accuracy": (
                        _covent_train_metrics.get("accuracy")
                    ),
                    "coverage_entailment_aux_train_macro_f1": (
                        _covent_train_metrics.get("macro_f1")
                    ),
                    "coverage_entailment_aux_dev_accuracy": (
                        _covent_dev_metrics.get("accuracy")
                    ),
                    "coverage_entailment_aux_dev_macro_f1": (
                        _covent_dev_metrics.get("macro_f1")
                    ),
                    "coverage_direction_confusion_matrix": (
                        _covent_dev_metrics.get("confusion_matrix")
                    ),
                    "coverage_direction_per_class": _covent_dev_metrics.get("per_class"),
                    "used_for_checkpoint_selection": False,
                }
                if capture_best_trainable_state:
                    best_trainable_state = v5.capture_trainable_state(model)
                # Stage26-F extended: snapshot v7 diagnostic tensors from the new best epoch
                if args.architecture == "v7_hierarchical":
                    _best_dev_output_v7 = _v7_capture_dev_output(dev_output)

            # Stage44-B internal-only anti-collapse checkpoint selection.
            # Uses only normal internal clean-dev metrics already computed above.
            if stage44_use_anti_collapse_selection and not smoke_mode:
                _stage44_row = _stage44b_clean_dev_selection_metrics(
                    epoch=epoch,
                    score=score,
                    dev_metrics=dev_metrics,
                    dev_inputs=dev_inputs,
                    constraints=_stage44_constraints,
                )
                _stage44_candidate_table.append(_stage44_row)
                if _stage44_row["constraints_satisfied"]:
                    _stage44_satisfying_count += 1
                    if score > _stage44_score:
                        _stage44_score = score
                        _stage44_epoch = epoch
                        _stage44_selected_metrics = _stage44_row
                        _stage44_dev_metrics = dev_metrics
                        _stage44_dev_interventions = intervention_diagnostics_v6b(
                            dev_records, dev_output
                        )
                        _stage44_dev_pairwise_checks = (
                            None if smoke_mode else v5.pairwise_checks(dev_records, dev_output)
                        )
                        _stage44_dev_predictions = prediction_records_v6b(
                            dev_records,
                            dev_output,
                            stage32_owner_state_export=getattr(
                                args, "stage32_owner_state_export", False
                            ),
                            stage32_owner_state_shadow_mode=getattr(
                                args, "stage32_owner_state_shadow_mode", False
                            ),
                            stage32_coverage_owner_v2=getattr(
                                args, "stage32_coverage_owner_v2", False
                            ),
                            stage32_coverage_owner_v2_min_confidence=getattr(
                                args, "stage32_coverage_owner_v2_min_confidence", 0.50
                            ),
                            stage32_coverage_owner_v2_min_margin=getattr(
                                args, "stage32_coverage_owner_v2_min_margin", 0.05
                            ),
                            stage32_coverage_owner_v2_allow_abstain=getattr(
                                args, "stage32_coverage_owner_v2_allow_abstain", False
                            ),
                            stage33_structured_coverage_owner=getattr(
                                args, "stage33_use_structured_coverage_owner", False
                            ),
                            stage33_structured_coverage_owner_export=getattr(
                                args, "stage33_structured_coverage_owner_export", False
                            ),
                            stage33_structured_coverage_owner_shadow_mode=getattr(
                                args, "stage33_structured_coverage_owner_shadow_mode", False
                            ),
                            stage33_structured_coverage_preserve_can_support=getattr(
                                args, "stage33_structured_coverage_preserve_can_support", False
                            ),
                            stage33_structured_coverage_direct_support_rules=getattr(
                                args,
                                "stage33_structured_coverage_direct_support_rules",
                                _STAGE33_DEFAULT_DIRECT_SUPPORT_RULES,
                            ),
                            stage33_structured_coverage_disable_specific_general_direct_support=getattr(
                                args,
                                "stage33_structured_coverage_disable_specific_general_direct_support",
                                False,
                            ),
                            stage33_structured_coverage_weak_rules_to_residual=getattr(
                                args, "stage33_structured_coverage_weak_rules_to_residual", ""
                            ),
                            stage33_structured_coverage_conditional_fallback=getattr(
                                args, "stage33_structured_coverage_conditional_fallback", False
                            ),
                            stage33_structured_coverage_fallback_source=getattr(
                                args, "stage33_structured_coverage_fallback_source", "current_final"
                            ),
                            stage33_structured_coverage_enable_whole_part_rules=getattr(
                                args, "stage33_structured_coverage_enable_whole_part_rules", False
                            ),
                            stage33_structured_coverage_whole_part_direct_support=getattr(
                                args, "stage33_structured_coverage_whole_part_direct_support", False
                            ),
                            stage33_structured_coverage_whole_part_lexicon=getattr(
                                args, "stage33_structured_coverage_whole_part_lexicon", ""
                            ),
                            stage33_structured_coverage_whole_part_v2=getattr(
                                args, "stage33_structured_coverage_whole_part_v2", False
                            ),
                            stage33_structured_coverage_whole_part_v2_use_expanded_lexicon=getattr(
                                args,
                                "stage33_structured_coverage_whole_part_v2_use_expanded_lexicon",
                                False,
                            ),
                            stage33_structured_coverage_whole_part_v2_direct_support_policy=getattr(
                                args,
                                "stage33_structured_coverage_whole_part_v2_direct_support_policy",
                                "hard_core_required",
                            ),
                            stage33_whole_part_conditional_safe_overrides_hard_core=getattr(
                                args,
                                "stage33_whole_part_conditional_safe_overrides_hard_core",
                                False,
                            ),
                            stage36_support_safety_config=_stage36_config_from_args(args),
                            stage37_safe_support_recovery_config=_stage37_config_from_args(args),
                            args=args,
                        )
                        _stage44_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                        _stage44_pc_metrics = {
                            **_pc_eval_metrics,
                            "pair_contrastive_use_case": args.pair_contrastive_use_case,
                            "count_by_preservation_construction_type": _pc_pres_type_counts,
                            "count_by_frame_construction_type": _pc_frame_type_counts,
                        }

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
                    _tc_dev_predictions = prediction_records_v6b(
                        dev_records,
                        dev_output,
                        stage32_owner_state_export=getattr(
                            args, "stage32_owner_state_export", False
                        ),
                        stage32_owner_state_shadow_mode=getattr(
                            args, "stage32_owner_state_shadow_mode", False
                        ),
                        stage32_coverage_owner_v2=getattr(
                            args, "stage32_coverage_owner_v2", False
                        ),
                        stage32_coverage_owner_v2_min_confidence=getattr(
                            args, "stage32_coverage_owner_v2_min_confidence", 0.50
                        ),
                        stage32_coverage_owner_v2_min_margin=getattr(
                            args, "stage32_coverage_owner_v2_min_margin", 0.05
                        ),
                        stage32_coverage_owner_v2_allow_abstain=getattr(
                            args, "stage32_coverage_owner_v2_allow_abstain", False
                        ),
                        stage33_structured_coverage_owner=getattr(
                            args, "stage33_use_structured_coverage_owner", False
                        ),
                        stage33_structured_coverage_owner_export=getattr(
                            args, "stage33_structured_coverage_owner_export", False
                        ),
                        stage33_structured_coverage_owner_shadow_mode=getattr(
                            args, "stage33_structured_coverage_owner_shadow_mode", False
                        ),
                        stage33_structured_coverage_preserve_can_support=getattr(
                            args, "stage33_structured_coverage_preserve_can_support", False
                        ),
                        stage33_structured_coverage_direct_support_rules=getattr(
                            args,
                            "stage33_structured_coverage_direct_support_rules",
                            _STAGE33_DEFAULT_DIRECT_SUPPORT_RULES,
                        ),
                        stage33_structured_coverage_disable_specific_general_direct_support=getattr(
                            args,
                            "stage33_structured_coverage_disable_specific_general_direct_support",
                            False,
                        ),
                        stage33_structured_coverage_weak_rules_to_residual=getattr(
                            args, "stage33_structured_coverage_weak_rules_to_residual", ""
                        ),
                        stage33_structured_coverage_conditional_fallback=getattr(
                            args, "stage33_structured_coverage_conditional_fallback", False
                        ),
                        stage33_structured_coverage_fallback_source=getattr(
                            args, "stage33_structured_coverage_fallback_source", "current_final"
                        ),
                        stage33_structured_coverage_enable_whole_part_rules=getattr(
                            args, "stage33_structured_coverage_enable_whole_part_rules", False
                        ),
                        stage33_structured_coverage_whole_part_direct_support=getattr(
                            args, "stage33_structured_coverage_whole_part_direct_support", False
                        ),
                        stage33_structured_coverage_whole_part_lexicon=getattr(
                            args, "stage33_structured_coverage_whole_part_lexicon", ""
                        ),
                        stage33_structured_coverage_whole_part_v2=getattr(
                            args, "stage33_structured_coverage_whole_part_v2", False
                        ),
                        stage33_structured_coverage_whole_part_v2_use_expanded_lexicon=getattr(
                            args,
                            "stage33_structured_coverage_whole_part_v2_use_expanded_lexicon",
                            False,
                        ),
                        stage33_structured_coverage_whole_part_v2_direct_support_policy=getattr(
                            args,
                            "stage33_structured_coverage_whole_part_v2_direct_support_policy",
                            "hard_core_required",
                        ),
                        stage33_whole_part_conditional_safe_overrides_hard_core=getattr(
                            args,
                            "stage33_whole_part_conditional_safe_overrides_hard_core",
                            False,
                        ),
                        stage36_support_safety_config=_stage36_config_from_args(args),
                        stage37_safe_support_recovery_config=_stage37_config_from_args(args),
                        args=args,
                    )
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
                        _pcs_dev_predictions = prediction_records_v6b(
                            dev_records,
                            dev_output,
                            stage32_owner_state_export=getattr(
                                args, "stage32_owner_state_export", False
                            ),
                            stage32_owner_state_shadow_mode=getattr(
                                args, "stage32_owner_state_shadow_mode", False
                            ),
                            stage32_coverage_owner_v2=getattr(
                                args, "stage32_coverage_owner_v2", False
                            ),
                            stage32_coverage_owner_v2_min_confidence=getattr(
                                args, "stage32_coverage_owner_v2_min_confidence", 0.50
                            ),
                            stage32_coverage_owner_v2_min_margin=getattr(
                                args, "stage32_coverage_owner_v2_min_margin", 0.05
                            ),
                            stage32_coverage_owner_v2_allow_abstain=getattr(
                                args, "stage32_coverage_owner_v2_allow_abstain", False
                            ),
                            stage33_structured_coverage_owner=getattr(
                                args, "stage33_use_structured_coverage_owner", False
                            ),
                            stage33_structured_coverage_owner_export=getattr(
                                args, "stage33_structured_coverage_owner_export", False
                            ),
                            stage33_structured_coverage_owner_shadow_mode=getattr(
                                args, "stage33_structured_coverage_owner_shadow_mode", False
                            ),
                            stage33_structured_coverage_preserve_can_support=getattr(
                                args, "stage33_structured_coverage_preserve_can_support", False
                            ),
                            stage33_structured_coverage_direct_support_rules=getattr(
                                args,
                                "stage33_structured_coverage_direct_support_rules",
                                _STAGE33_DEFAULT_DIRECT_SUPPORT_RULES,
                            ),
                            stage33_structured_coverage_disable_specific_general_direct_support=getattr(
                                args,
                                "stage33_structured_coverage_disable_specific_general_direct_support",
                                False,
                            ),
                            stage33_structured_coverage_weak_rules_to_residual=getattr(
                                args, "stage33_structured_coverage_weak_rules_to_residual", ""
                            ),
                            stage33_structured_coverage_conditional_fallback=getattr(
                                args, "stage33_structured_coverage_conditional_fallback", False
                            ),
                            stage33_structured_coverage_fallback_source=getattr(
                                args, "stage33_structured_coverage_fallback_source", "current_final"
                            ),
                            stage33_structured_coverage_enable_whole_part_rules=getattr(
                                args, "stage33_structured_coverage_enable_whole_part_rules", False
                            ),
                            stage33_structured_coverage_whole_part_direct_support=getattr(
                                args, "stage33_structured_coverage_whole_part_direct_support", False
                            ),
                            stage33_structured_coverage_whole_part_lexicon=getattr(
                                args, "stage33_structured_coverage_whole_part_lexicon", ""
                            ),
                            stage33_structured_coverage_whole_part_v2=getattr(
                                args, "stage33_structured_coverage_whole_part_v2", False
                            ),
                            stage33_structured_coverage_whole_part_v2_use_expanded_lexicon=getattr(
                                args,
                                "stage33_structured_coverage_whole_part_v2_use_expanded_lexicon",
                                False,
                            ),
                            stage33_structured_coverage_whole_part_v2_direct_support_policy=getattr(
                                args,
                                "stage33_structured_coverage_whole_part_v2_direct_support_policy",
                                "hard_core_required",
                            ),
                            stage33_whole_part_conditional_safe_overrides_hard_core=getattr(
                                args,
                                "stage33_whole_part_conditional_safe_overrides_hard_core",
                                False,
                            ),
                            stage36_support_safety_config=_stage36_config_from_args(args),
                            stage37_safe_support_recovery_config=_stage37_config_from_args(args),
                            args=args,
                        )
                        _pcs_state = {
                            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                        }
                        _pcs_pc_metrics = {
                            **_pc_eval_metrics,
                            "pair_contrastive_use_case": args.pair_contrastive_use_case,
                            "count_by_preservation_construction_type": _pc_pres_type_counts,
                            "count_by_frame_construction_type": _pc_frame_type_counts,
                        }

            #  Stage26-F: record per-epoch dev snapshot 
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
            #  end Stage26-F epoch snapshot 

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
                # At least one epoch satisfied all constraints ??override best_* with constrained
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
                # No epoch met all constraints ??fall back to unconstrained best
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
                # At least one epoch satisfied preservation constraints ??override best_*
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
                # No epoch met preservation constraints ??fall back to unconstrained best
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

        # Apply Stage44-B internal anti-collapse checkpoint selection after all epochs.
        _stage44_original_best_epoch = best_epoch
        _stage44_original_best_metrics = next(
            (
                row for row in _stage44_candidate_table
                if row.get("epoch") == _stage44_original_best_epoch
            ),
            None,
        )
        _stage44_prior_aware_enabled = bool(
            stage44_use_anti_collapse_selection
            and stage44_use_prior_aware_ne_constraint
        )
        _stage44_prior_aware_constraints = {
            "use_prior_aware_ne_constraint": bool(stage44_use_prior_aware_ne_constraint),
            "max_ne_gold_prior_delta": stage44_max_ne_gold_prior_delta,
            "min_macro_f1": stage44_min_macro_f1,
            "min_relative_macro_f1_of_best": stage44_min_relative_macro_f1_of_best,
            "min_support_precision": stage44_min_support_precision,
            "min_refute_precision": stage44_min_refute_precision,
        }
        _stage44_gold_label_rates = next(
            (
                row.get("gold_label_rates") for row in _stage44_candidate_table
                if row.get("gold_label_rates") is not None
            ),
            None,
        )
        _stage44_effective_epoch = _stage44_epoch
        _stage44_relative_macro_f1_satisfied = True
        _stage44_relative_macro_f1_threshold = None
        if (
            _stage44_prior_aware_enabled
            and stage44_min_relative_macro_f1_of_best is not None
            and _stage44_epoch >= 1
        ):
            _stage44_original_best_macro_f1 = (
                (_stage44_original_best_metrics or {}).get("macro_f1")
                or (best_dev_metrics or {}).get("final_macro_f1")
            )
            _stage44_selected_macro_f1 = (_stage44_selected_metrics or {}).get("macro_f1")
            if _stage44_original_best_macro_f1 is not None:
                _stage44_relative_macro_f1_threshold = (
                    _stage44_original_best_macro_f1
                    * stage44_min_relative_macro_f1_of_best
                )
            _stage44_relative_macro_f1_satisfied = (
                _stage44_selected_macro_f1 is not None
                and _stage44_relative_macro_f1_threshold is not None
                and _stage44_selected_macro_f1 >= _stage44_relative_macro_f1_threshold
            )
            if _stage44_selected_metrics is not None:
                _stage44_selected_metrics = dict(_stage44_selected_metrics)
                _stage44_checks = dict(_stage44_selected_metrics.get("constraint_checks") or {})
                _stage44_checks["min_relative_macro_f1_of_best"] = bool(
                    _stage44_relative_macro_f1_satisfied
                )
                _stage44_selected_metrics["constraint_checks"] = _stage44_checks
                _stage44_selected_metrics["constraints_satisfied"] = bool(
                    _stage44_selected_metrics.get("constraints_satisfied")
                    and _stage44_relative_macro_f1_satisfied
                )
                _stage44_selected_metrics["stage44b2_relative_macro_f1_threshold"] = (
                    _stage44_relative_macro_f1_threshold
                )
            if not _stage44_relative_macro_f1_satisfied:
                _stage44_effective_epoch = -1

        _stage44_selected_by_constraints = False
        _stage44_constraints_satisfied = False
        if not stage44_use_anti_collapse_selection:
            _stage44_decision = "STAGE44B_INTERNAL_ANTI_COLLAPSE_SELECTION_DISABLED"
            _stage44b2_decision = "STAGE44B2_PRIOR_AWARE_SELECTION_DISABLED"
            _stage44_recommendation = (
                "Stage44-B anti-collapse selection is disabled; default checkpoint selection behavior is unchanged."
            )
            _stage44b2_recommendation = (
                "Stage44-B2 prior-aware selection is disabled because Stage44-B selection is disabled."
            )
            _stage44_selected_metrics = _stage44_original_best_metrics
        elif _stage44_effective_epoch >= 1:
            _stage44_selected_by_constraints = True
            _stage44_constraints_satisfied = True
            best_epoch = _stage44_effective_epoch
            best_dev_metrics = _stage44_dev_metrics
            best_dev_interventions = _stage44_dev_interventions
            best_dev_pairwise_checks = _stage44_dev_pairwise_checks
            best_dev_predictions = _stage44_dev_predictions
            best_state = _stage44_state
            best_pc_metrics = _stage44_pc_metrics
            _stage44_decision = "STAGE44B_INTERNAL_ANTI_COLLAPSE_SELECTION_READY"
            _stage44b2_decision = (
                "STAGE44B2_PRIOR_AWARE_SELECTION_READY"
                if _stage44_prior_aware_enabled
                else "STAGE44B2_PRIOR_AWARE_SELECTION_DISABLED"
            )
            _stage44_recommendation = (
                "Selected the highest primary-metric internal clean-dev checkpoint satisfying Stage44-B anti-collapse constraints."
            )
            _stage44b2_recommendation = (
                "Selected the highest primary-metric internal clean-dev checkpoint satisfying enabled Stage44-B2 prior-aware constraints."
                if _stage44_prior_aware_enabled
                else "Stage44-B2 prior-aware selection is disabled; Stage44-B behavior is unchanged."
            )
            print(
                f"  [stage44b_sel] selected epoch={_stage44_effective_epoch} "
                f"score={_stage44_score:.4f} satisfying_epochs={_stage44_satisfying_count} "
                f"original_best_epoch={_stage44_original_best_epoch}"
            )
        elif stage44_selection_fallback == "fail_incomplete":
            _stage44_decision = "STAGE44B_INTERNAL_ANTI_COLLAPSE_SELECTION_INCOMPLETE"
            _stage44b2_decision = (
                "STAGE44B2_PRIOR_AWARE_SELECTION_INCOMPLETE"
                if _stage44_prior_aware_enabled
                else "STAGE44B2_PRIOR_AWARE_SELECTION_DISABLED"
            )
            _stage44_recommendation = (
                "No internal clean-dev epoch satisfied Stage44-B constraints; normal best-metric checkpoint is preserved, but Stage44-B should be treated as incomplete."
            )
            _stage44b2_recommendation = (
                "No internal clean-dev epoch satisfied the enabled Stage44-B2 prior-aware constraints; treat Stage44-B2 as incomplete."
                if _stage44_prior_aware_enabled
                else "Stage44-B2 prior-aware selection is disabled; Stage44-B fallback behavior is unchanged."
            )
            _stage44_selected_metrics = _stage44_original_best_metrics
            print(
                "  [stage44b_sel] WARNING: no eligible epoch found; "
                f"fallback=fail_incomplete preserves original best epoch={best_epoch}"
            )
        else:
            _stage44_decision = "STAGE44B_INTERNAL_ANTI_COLLAPSE_SELECTION_FALLBACK"
            _stage44b2_decision = (
                "STAGE44B2_PRIOR_AWARE_SELECTION_FALLBACK"
                if _stage44_prior_aware_enabled
                else "STAGE44B2_PRIOR_AWARE_SELECTION_DISABLED"
            )
            _stage44_recommendation = (
                "No internal clean-dev epoch satisfied Stage44-B constraints; selected the original best-metric checkpoint by fallback policy."
            )
            _stage44b2_recommendation = (
                "No internal clean-dev epoch satisfied the enabled Stage44-B2 prior-aware constraints; selected the original best-metric checkpoint by fallback policy."
                if _stage44_prior_aware_enabled
                else "Stage44-B2 prior-aware selection is disabled; Stage44-B fallback behavior is unchanged."
            )
            _stage44_selected_metrics = _stage44_original_best_metrics
            print(
                "  [stage44b_sel] WARNING: no eligible epoch found; "
                f"fallback=best_metric preserves original best epoch={best_epoch}"
            )

        if _stage44_selected_by_constraints and _stage44_selected_metrics is not None:
            _stage44_selected_metrics_final = _stage44_selected_metrics
        else:
            _stage44_selected_metrics_final = next(
                (row for row in _stage44_candidate_table if row.get("epoch") == best_epoch),
                _stage44_selected_metrics,
            )
        _stage44_selection_info: dict[str, Any] = {
            "stage44b_enabled": bool(stage44_use_anti_collapse_selection),
            "stage44b_decision": _stage44_decision,
            "stage44b_selection_mode": "internal_clean_dev_constraints",
            "stage44b_constraints": _stage44_constraints,
            "stage44b_selected_epoch": best_epoch if stage44_use_anti_collapse_selection else None,
            "stage44b_original_best_metric_epoch": (
                _stage44_original_best_epoch if stage44_use_anti_collapse_selection else None
            ),
            "stage44b_selected_by_constraints": _stage44_selected_by_constraints,
            "stage44b_constraints_satisfied": _stage44_constraints_satisfied,
            "stage44b_num_candidate_epochs": len(_stage44_candidate_table),
            "stage44b_num_constraint_satisfying_epochs": _stage44_satisfying_count,
            "stage44b_candidate_table": _stage44_candidate_table,
            "stage44b_selected_metrics": _stage44_selected_metrics_final,
            "stage44b_original_best_metric_metrics": _stage44_original_best_metrics,
            "stage44b_not_entitled_prediction_rate": (
                (_stage44_selected_metrics_final or {}).get("not_entitled_prediction_rate")
            ),
            "stage44b_support_recall": (
                (_stage44_selected_metrics_final or {}).get("support_recall")
            ),
            "stage44b_refute_recall": (
                (_stage44_selected_metrics_final or {}).get("refute_recall")
            ),
            "stage44b_leakage_policy": (
                "Stage44-B uses only the normal internal clean-dev split. It does not read or use Stage43-B1 external labels, metrics, examples, predictions, thresholds, calibration, checkpoint/model selection signals, loss design signals, or composer behavior changes."
            ),
            "stage44b_recommendation": _stage44_recommendation,
            "stage44b2_enabled": bool(_stage44_prior_aware_enabled),
            "stage44b2_decision": _stage44b2_decision,
            "stage44b2_prior_aware_enabled": bool(_stage44_prior_aware_enabled),
            "stage44b2_gold_label_rates": _stage44_gold_label_rates,
            "stage44b2_gold_not_entitled_rate": (
                (_stage44_gold_label_rates or {}).get("NOT_ENTITLED")
            ),
            "stage44b2_gold_support_rate": (
                (_stage44_gold_label_rates or {}).get("SUPPORT")
            ),
            "stage44b2_gold_refute_rate": (
                (_stage44_gold_label_rates or {}).get("REFUTE")
            ),
            "stage44b2_prior_aware_constraints": _stage44_prior_aware_constraints,
            "stage44b2_selected_epoch": (
                best_epoch if _stage44_prior_aware_enabled else None
            ),
            "stage44b2_original_best_metric_epoch": (
                _stage44_original_best_epoch if _stage44_prior_aware_enabled else None
            ),
            "stage44b2_selected_metrics": _stage44_selected_metrics_final,
            "stage44b2_original_best_metric_metrics": _stage44_original_best_metrics,
            "stage44b2_ne_pred_minus_gold_ne_rate": (
                (_stage44_selected_metrics_final or {}).get("stage44b2_ne_pred_minus_gold_ne_rate")
            ),
            "stage44b2_original_best_ne_pred_minus_gold_ne_rate": (
                (_stage44_original_best_metrics or {}).get("stage44b2_ne_pred_minus_gold_ne_rate")
            ),
            "stage44b2_reason_previous_fixed_cap_was_invalid": (
                "A fixed NOT_ENTITLED prediction cap below the internal clean-dev NOT_ENTITLED gold prior can force over-SUPPORT checkpoints; Stage44-B2 compares predicted NOT_ENTITLED rate against internal gold prior plus delta instead."
            ),
            "stage44b2_recommendation": _stage44b2_recommendation,
        }

        #  Build run-level audit ledger (reporting only; no model/loss/logit change)
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
                    "Baseline is NOT CE-only when this is enabled ??ranking loss is active by default."
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
                    "pairwise_losses['total']; internally weighted composite ??"
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
                    "pre-pair-projector slot states from FrameGate ??NOT frame_pair_repr. "
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
            # Stage26-G: v7 stabilization losses ??all off by default, v7_hierarchical only
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
                "target_derivation": "REFUTE->local_0, SUPPORT->local_1 (NOT_ENTITLED excluded)",
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
            # Stage30-D: temporal mismatch multihead loss
            "v7_temporal_mismatch_multihead_loss": {
                "enabled": (
                    args.architecture == "v7_hierarchical"
                    and getattr(args, "v7_use_temporal_mismatch_multihead", False)
                    and getattr(args, "v7_use_temporal_mismatch_multihead_loss", False)
                    and getattr(args, "v7_temporal_mismatch_multihead_loss_weight", 0.0) > 0.0
                    and not getattr(args, "v7_no_aux_losses", False)
                ),
                "weight": getattr(args, "v7_temporal_mismatch_multihead_loss_weight", 0.0),
                "pos_weight": getattr(
                    args, "v7_temporal_mismatch_multihead_loss_pos_weight", None
                ),
                "target": (
                    "temporal_frame_mismatch_logit, temporal_predicate_mismatch_logit, "
                    "temporal_sufficiency_mismatch_logit"
                ),
                "target_derivation": (
                    "temporal_mismatch=1 (time_swap/stage30_temporal_safe_label=0), "
                    "safe=0 (none/paraphrase/stage30_temporal_safe_label=1)"
                ),
                "fusion": getattr(args, "v7_temporal_mismatch_multihead_fusion", "frame_only"),
                "loss_combination": "mean of 3 per-head BCE losses",
                "stage15_used_for_selection_or_calibration": False,
                "external_probe_used": False,
                "ood_used": False,
                "raw_loss_key": "v7_temporal_mismatch_multihead_loss",
                "weighted_loss_key": "v7_temporal_mismatch_multihead_loss",
                "note": (
                    "Stage30-D: mean BCE over 3 heads (frame/predicate/sufficiency). "
                    "Separate aux dataset; pair_id filtered to main train only. "
                    "No Stage15. No external probe. No OOD."
                    if (
                        getattr(args, "v7_use_temporal_mismatch_multihead", False)
                        and getattr(args, "v7_use_temporal_mismatch_multihead_loss", False)
                        and getattr(args, "v7_temporal_mismatch_multihead_loss_weight", 0.0) > 0.0
                    ) else "disabled"
                ),
            },
            # Stage30-E: temporal preservation head loss
            "v7_temporal_preservation_loss": {
                "enabled": (
                    args.architecture == "v7_hierarchical"
                    and getattr(args, "v7_use_temporal_preservation_head", False)
                    and getattr(args, "v7_use_temporal_preservation_loss", False)
                    and getattr(args, "v7_temporal_preservation_loss_weight", 0.0) > 0.0
                    and not getattr(args, "v7_no_aux_losses", False)
                ),
                "weight": getattr(args, "v7_temporal_preservation_loss_weight", 0.0),
                "pos_weight": getattr(
                    args, "v7_temporal_preservation_loss_pos_weight", None
                ),
                "target": "temporal_preservation_logit",
                "target_derivation": (
                    "preserved=1 (none/paraphrase/stage30_temporal_safe_label=1), "
                    "not_preserved=0 (time_swap/stage30_temporal_safe_label=0)"
                ),
                "stage15_used_for_selection_or_calibration": False,
                "external_probe_used": False,
                "ood_used": False,
                "raw_loss_key": "v7_temporal_preservation_loss",
                "weighted_loss_key": "v7_temporal_preservation_loss",
                "note": (
                    "Stage30-E: BCE on temporal preservation head. "
                    "Separate aux dataset; pair_id filtered to main train only. "
                    "No Stage15. No external probe. No OOD."
                    if (
                        getattr(args, "v7_use_temporal_preservation_head", False)
                        and getattr(args, "v7_use_temporal_preservation_loss", False)
                        and getattr(args, "v7_temporal_preservation_loss_weight", 0.0) > 0.0
                    ) else "disabled"
                ),
            },
            "v7_coverage_entailment_loss": {
                "enabled": (
                    args.architecture == "v7_hierarchical"
                    and getattr(args, "v7_use_coverage_entailment_head", False)
                    and getattr(args, "v7_use_coverage_entailment_loss", False)
                    and getattr(args, "v7_coverage_entailment_loss_weight", 0.0) > 0.0
                    and not getattr(args, "v7_no_aux_losses", False)
                ),
                "weight": getattr(args, "v7_coverage_entailment_loss_weight", 0.0),
                "target": "coverage_entailment_logits",
                "target_derivation": "coverage_direction_id from Stage31-C auxiliary data",
                "classes": STAGE31C_COVERAGE_LABELS[
                    : getattr(args, "v7_coverage_entailment_num_classes", 3)
                ],
                "stage31_probe_used": False,
                "used_for_checkpoint_selection": False,
                "modifies_final_logits": False,
                "raw_loss_key": "v7_coverage_entailment_loss",
                "weighted_loss_key": "v7_coverage_entailment_loss",
                "note": (
                    "Stage31-C readout-only CE over separate auxiliary data. "
                    "It does not edit output['logits'], H1 composer, entitlement, caps, or NE."
                    if (
                        getattr(args, "v7_use_coverage_entailment_head", False)
                        and getattr(args, "v7_use_coverage_entailment_loss", False)
                        and getattr(args, "v7_coverage_entailment_loss_weight", 0.0) > 0.0
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
                    "Scale is a fixed hyperparameter ??not OOD-calibrated. Stage15 never used."
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
                    "AND PE signals non-entitlement. Scale is a fixed hyperparameter ??not "
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
                    "post-hoc global NE logit shift; selected on controlled data only ??not Stage15. "
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
                    "OOD-tuned global NE shift ??Stage15 is consulted to select the shift. "
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
                    "OOD-tuned selective NE shift ??Stage15 is consulted to select the shift. "
                    "NOT a valid final-model component. Diagnostic upper bound only."
                    if args.ood_selective_ne_shift_sweep is not None else "disabled"
                ),
            },
        }

        # Architectural logit components: learned parameters inside model.forward().
        # These modify final_logits as part of the model graph ??they are trained through
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
            "stage44b_anti_collapse_selection": {
                "enabled": bool(stage44_use_anti_collapse_selection),
                "constraints": _stage44_constraints if stage44_use_anti_collapse_selection else None,
                "fallback": stage44_selection_fallback,
                "stage43b1_used": False,
                "fallback_triggered": (
                    bool(stage44_use_anti_collapse_selection)
                    and not _stage44_selected_by_constraints
                ),
                "eligible_epoch_count": (
                    _stage44_satisfying_count if stage44_use_anti_collapse_selection else None
                ),
                "stage44b2_prior_aware": {
                    "enabled": bool(_stage44_prior_aware_enabled),
                    "constraints": _stage44_prior_aware_constraints,
                    "gold_label_rates": _stage44_gold_label_rates,
                    "decision": _stage44b2_decision,
                    "stage43b1_used": False,
                },
            },
        }

        # Audit warnings (reporting only; using weighted ratios as primary signal)
        _audit_warnings: list[str] = []
        if _aux_to_ce_w is not None and _aux_to_ce_w > 0.5:
            _audit_warnings.append(
                f"aux_to_ce_loss_ratio_weighted={_aux_to_ce_w:.3f} > 0.5: "
                "weighted auxiliary losses exceed 50% of CE ??total_loss is not CE-dominated"
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
                "are enabled ??this should have been caught at startup"
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

        #  Stage26-F extended: build v7 collapse and logit diagnostics 
        # No new forward passes.  All values derived from already-computed variables.
        # _best_dev_output_v7 : CPU tensors from the unconstrained-best epoch (may differ
        #                       from checkpointed epoch if TC/PCS selection applied).
        # dev_output           : last epoch's output (still in scope after the for-loop).
        # best_dev_metrics     : always reflects the actually selected epoch (after TC/PCS).
        _v7_ext_diagnostics: dict[str, Any] = {}
        if args.architecture == "v7_hierarchical":
            _v7_best_logit_summary = _v7_make_logit_summary(_best_dev_output_v7)

            # Final-epoch logit summary ??dev_output is the last iteration's output
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

            # Stage27-H2A: H1-path entitlement_for_decision stats (from best-dev epoch)
            _h1_efd_stats: "dict[str, float] | None" = None
            if (
                getattr(args, "v7_use_v6b_style_final_decision", False)
                and _v7_best_logit_summary is not None
                and "v7_h1_entitlement_for_decision" in _v7_best_logit_summary
            ):
                _h1_efd_stats = _v7_best_logit_summary["v7_h1_entitlement_for_decision"]

            # Component prob means from best-dev logit summary
            def _logit_mean(key: str) -> "float | None":
                if _v7_best_logit_summary is None:
                    return None
                sub = _v7_best_logit_summary.get(key)
                return sub["mean"] if isinstance(sub, dict) else None

            _v7_ext_diagnostics = {
                "v7_best_dev_logit_summary": _v7_best_logit_summary,
                "v7_final_epoch_logit_summary": _v7_final_logit_summary,
                "v7_best_dev_per_gold_label_summary": _v7_per_gold_summary,
                # Stage27-H2A: H1 entitlement_for_decision stats (None when H1 inactive)
                "v7_h1_entitlement_for_decision_mean": (
                    _h1_efd_stats["mean"] if _h1_efd_stats else None
                ),
                "v7_h1_entitlement_for_decision_std": (
                    _h1_efd_stats["std"] if _h1_efd_stats else None
                ),
                "v7_h1_entitlement_for_decision_min": (
                    _h1_efd_stats["min"] if _h1_efd_stats else None
                ),
                "v7_h1_entitlement_for_decision_max": (
                    _h1_efd_stats["max"] if _h1_efd_stats else None
                ),
                # Component probability means (from best-dev epoch; None when not v7)
                "frame_prob_mean": _logit_mean("v7_frame_prob"),
                "predicate_coverage_prob_mean": _logit_mean("v7_predicate_prob"),
                "sufficiency_prob_mean": _logit_mean("v7_sufficiency_prob"),
                "v7_entitlement_prob_mean": _logit_mean("v7_entitlement_prob"),
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
        #  end Stage26-F extended 

        _run_audit_ledger: dict[str, Any] = {
            "active_training_losses": _active_training_losses,
            "active_final_logit_modifiers": _active_final_logit_modifiers,
            "active_architectural_logit_components": _active_architectural_logit_components,
            "active_selection_rules": _active_selection_rules,
            # Loss component averages ??raw (pre-weight) and weighted (contribution to total_loss)
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
            # Ratios ??weighted is primary; raw is supplemental
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
            # Stage60: train-only Stage57 non-leaking external bridge provenance
            **_stage60_bridge_info,
            # Stage69/Stage70: train-only Stage66 residual bridge provenance
            **_stage66_bridge_info,
            # Stage75C: train-only Stage75 targeted residual bridge provenance
            **_stage75_bridge_info,
            # Stage80D: train-only Stage80A conservative Stage75v2 bridge +
            # combined Stage57+Stage66+Stage75+Stage80A bridge provenance
            **_stage80a_bridge_info,
            **_combined_bridge_info,
            # Stage26-A: v7 hierarchical architecture provenance (always False in Stage26-A)
            "stage15_used_for_v7_training": False,
            "stage15_used_for_v7_selection": False,
            "stage15_used_for_v7_aux_loss_targets": False,
            "time_swap_used_in_v7_main_clean_data": False,
        }

        if _stage45b_split_info.get("stage45b_enabled"):
            _stage45b_pred_counts = (best_dev_metrics or {}).get("prediction_distribution") or {}
            _stage45b_holdout_rows = _stage45b_split_info.get("stage45b_holdout_rows") or 0
            _stage45b_ne_rate = (
                _stage45b_pred_counts.get("NOT_ENTITLED", 0) / _stage45b_holdout_rows
                if _stage45b_holdout_rows else None
            )
            _stage45b_per_label = (best_dev_metrics or {}).get("per_label") or {}
            _stage45b_split_info["stage45b_holdout_metrics"] = {
                "accuracy": (best_dev_metrics or {}).get("final_accuracy"),
                "macro_f1": (best_dev_metrics or {}).get("final_macro_f1"),
                "per_label": _stage45b_per_label,
                "prediction_counts": _stage45b_pred_counts,
                "gold_counts": _stage45b_split_info.get("stage45b_holdout_label_counts"),
                "not_entitled_prediction_rate": _stage45b_ne_rate,
                "support_recall": _stage45b_per_label.get("SUPPORT", {}).get("recall"),
                "refute_recall": _stage45b_per_label.get("REFUTE", {}).get("recall"),
            }
            if _stage45b2_intervention_diag:
                _stage45b2_inactive = _stage45b2_intervention_diag.get(
                    "stage45b2_intervention_objective_effectively_inactive"
                )
                _stage45b_split_info["stage45b2_intervention_objective_guard_enabled"] = (
                    _stage45b2_intervention_diag.get(
                        "stage45b2_intervention_objective_guard_enabled", True
                    )
                )
                _stage45b_split_info[
                    "stage45b2_intervention_objective_missing_variant_counts"
                ] = _stage45b2_intervention_diag.get(
                    "stage45b2_intervention_objective_missing_variant_counts"
                )
                _stage45b_split_info[
                    "stage45b2_intervention_objective_skipped_group_count"
                ] = _stage45b2_intervention_diag.get(
                    "stage45b2_intervention_objective_skipped_group_count"
                )
                _stage45b_split_info[
                    "stage45b2_intervention_objective_active_group_count"
                ] = _stage45b2_intervention_diag.get(
                    "stage45b2_intervention_objective_active_group_count"
                )
                _stage45b_split_info[
                    "stage45b2_intervention_objective_effectively_inactive"
                ] = _stage45b2_inactive
                _stage45b_split_info["stage45b2_decision"] = (
                    "STAGE45B2_FAMILY_HOLDOUT_OBJECTIVE_GUARDED"
                )
                _stage45b_split_info["stage45b2_recommendation"] = (
                    (
                        "The intervention objective was entirely inactive for this split "
                        "because every group was missing a required variant; no ranking-"
                        "objective gradient signal was available for this run. Consider "
                        "--use-intervention-loss (structured pairwise objective) or a "
                        "different --stage45-family-field if ranking supervision is required."
                    )
                    if _stage45b2_inactive
                    else (
                        "Missing intervention-objective terms caused by the internal family "
                        "holdout were skipped, not fabricated. This is an internal robustness "
                        "diagnostic; treat resulting metrics as holdout-family-specific, not a "
                        "substitute for full-data training."
                    )
                )

            _stage45b3_diag = (best_dev_pairwise_checks or {}).get(
                "stage45b3_pairwise_diagnostics"
            )
            if _stage45b3_diag:
                _stage45b3_inactive = _stage45b3_diag.get(
                    "pairwise_checks_effectively_inactive"
                )
                _stage45b_split_info["stage45b3_pairwise_check_guard_enabled"] = (
                    _stage45b3_diag.get("stage45b3_pairwise_check_guard_enabled", True)
                )
                _stage45b_split_info["stage45b3_pairwise_missing_variant_counts"] = (
                    _stage45b3_diag.get("pairwise_missing_variant_counts")
                )
                _stage45b_split_info["stage45b3_pairwise_skipped_missing_none_count"] = (
                    _stage45b3_diag.get("pairwise_groups_skipped_missing_none")
                )
                _stage45b_split_info["stage45b3_pairwise_active_group_count"] = (
                    _stage45b3_diag.get("pairwise_active_group_count")
                )
                _stage45b_split_info["stage45b3_pairwise_checks_effectively_inactive"] = (
                    _stage45b3_inactive
                )
                _stage45b_split_info["stage45b3_decision"] = (
                    "STAGE45B3_FAMILY_HOLDOUT_PAIRWISE_CHECKS_GUARDED"
                )
                _stage45b_split_info["stage45b3_recommendation"] = (
                    (
                        "Pairwise preservation/degradation checks were entirely inactive for "
                        "this dev split because no group retained the 'none' anchor row (e.g. "
                        "a single-family holdout dev split). This is expected under Stage45 "
                        "internal family holdout and does not invalidate the holdout "
                        "classification metrics; treat pairwise diagnostics as undefined for "
                        "this run rather than failed."
                    )
                    if _stage45b3_inactive
                    else (
                        "Some pairwise checks were skipped because their required variant(s) "
                        "were missing from the dev split under internal family holdout. "
                        "Remaining checks were computed only from complete groups; no train "
                        "rows or external data were used to fill gaps."
                    )
                )

        _stage45c_report = build_stage45c_report(
            enabled=stage45c_enabled,
            support_recovery_weight=stage45c_support_recovery_weight,
            entitled_ne_penalty_weight=stage45c_entitled_ne_penalty_weight,
            target_label=stage45c_target_label,
            entitled_labels=_stage45c_entitled_labels_tuple,
            train_label_counts=_stage45c_train_label_counts,
            support_recovery_loss_mean=(
                (_stage45c_diag or {}).get("support_recovery_loss_mean")
            ),
            entitled_ne_penalty_loss_mean=(
                (_stage45c_diag or {}).get("entitled_ne_penalty_loss_mean")
            ),
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
            "best_stage31c_coverage_entailment_metrics": best_covent_metrics,
            **_stage45c_report,
            **_tc_selection_info,
            **_pcs_selection_info,
            **_stage44_selection_info,
            **_stage45b_split_info,
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
            train_source_labels=_train_source_labels,
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
            stage44_use_anti_collapse_selection=args.stage44_use_anti_collapse_selection,
            stage44_min_support_recall=args.stage44_min_support_recall,
            stage44_min_refute_recall=args.stage44_min_refute_recall,
            stage44_max_not_entitled_pred_rate=args.stage44_max_not_entitled_pred_rate,
            stage44_min_clean_dev_accuracy=args.stage44_min_clean_dev_accuracy,
            stage44_selection_fallback=args.stage44_selection_fallback,
            stage44_use_prior_aware_ne_constraint=args.stage44_use_prior_aware_ne_constraint,
            stage44_max_ne_gold_prior_delta=args.stage44_max_ne_gold_prior_delta,
            stage44_min_macro_f1=args.stage44_min_macro_f1,
            stage44_min_relative_macro_f1_of_best=args.stage44_min_relative_macro_f1_of_best,
            stage44_min_support_precision=args.stage44_min_support_precision,
            stage44_min_refute_precision=args.stage44_min_refute_precision,
            stage45b_split_info=_stage45b_split_info,
            stage45c_enabled=getattr(args, "stage45c_enable_support_recovery", False),
            stage45c_support_recovery_weight=getattr(
                args, "stage45c_support_recovery_weight", 0.0
            ),
            stage45c_entitled_ne_penalty_weight=getattr(
                args, "stage45c_entitled_ne_penalty_weight", 0.0
            ),
            stage45c_target_label=getattr(args, "stage45c_target_label", "SUPPORT"),
            stage45c_entitled_labels=tuple(
                name.strip()
                for name in getattr(
                    args, "stage45c_entitled_labels", "SUPPORT,REFUTE"
                ).split(",")
                if name.strip()
            ),
            ts_train_inputs=_ts_train_inputs if _ts_data_needed else None,
            ts_train_labels=_ts_train_labels if _ts_data_needed else None,
            ts_train_mask=_ts_train_mask if _ts_data_needed else None,
            ts_loss_weight=(
                getattr(args, "v7_temporal_safety_loss_weight", 0.0)
                if getattr(args, "v7_use_temporal_safety_loss", False) else 0.0
            ),
            ts_loss_pos_weight=getattr(args, "v7_temporal_safety_loss_pos_weight", None),
            tmm_train_inputs=_tmm_train_inputs if _tmm_data_needed else None,
            tmm_train_labels=_tmm_train_labels if _tmm_data_needed else None,
            tmm_train_mask=_tmm_train_mask if _tmm_data_needed else None,
            tmm_loss_weight=(
                getattr(args, "v7_temporal_mismatch_multihead_loss_weight", 0.0)
                if getattr(args, "v7_use_temporal_mismatch_multihead_loss", False) else 0.0
            ),
            tmm_loss_pos_weight=getattr(
                args, "v7_temporal_mismatch_multihead_loss_pos_weight", None
            ),
            tpres_train_inputs=_tpres_train_inputs if _tpres_data_needed else None,
            tpres_train_labels=_tpres_train_labels if _tpres_data_needed else None,
            tpres_train_mask=_tpres_train_mask if _tpres_data_needed else None,
            tpres_loss_weight=(
                getattr(args, "v7_temporal_preservation_loss_weight", 0.0)
                if getattr(args, "v7_use_temporal_preservation_loss", False) else 0.0
            ),
            tpres_loss_pos_weight=getattr(
                args, "v7_temporal_preservation_loss_pos_weight", None
            ),
            covent_train_inputs=_covent_train_inputs if _covent_data_needed else None,
            covent_dev_inputs=_covent_dev_inputs if _covent_data_needed else None,
            covent_train_labels=_covent_train_labels if _covent_data_needed else None,
            covent_dev_labels=_covent_dev_labels if _covent_data_needed else None,
            covent_loss_weight=(
                getattr(args, "v7_coverage_entailment_loss_weight", 0.0)
                if getattr(args, "v7_use_coverage_entailment_loss", False) else 0.0
            ),
            covent_class_weights=_covent_class_weights,
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
            "vnext_evidence_interface": args.vnext_evidence_interface,
            "vnext_evidence_interface_default_preserves_existing_behavior": (
                args.vnext_evidence_interface == "full_evidence"
            ),
            "vnext_enable_segmented_dual_pass": getattr(args, "vnext_enable_segmented_dual_pass", False),
            "vnext_segmented_context_role": getattr(args, "vnext_segmented_context_role", "diagnostic_only"),
            "vnext_context_risk_cap_alpha": getattr(args, "vnext_context_risk_cap_alpha", 0.0),
            "vnext_context_risk_threshold": getattr(args, "vnext_context_risk_threshold", 0.5),
            "vnext_context_risk_source": getattr(args, "vnext_context_risk_source", "context_not_entitled_prob"),
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
                    or (getattr(args, "v7_use_location_boundary_head", False)
                        and getattr(args, "v7_use_location_boundary_loss", False)
                        and getattr(args, "v7_location_boundary_loss_weight", 0.0) > 0.0)
                    or (getattr(args, "v7_use_temporal_safety_head", False)
                        and getattr(args, "v7_use_temporal_safety_loss", False)
                        and getattr(args, "v7_temporal_safety_loss_weight", 0.0) > 0.0)
                    or (getattr(args, "v7_use_temporal_mismatch_multihead", False)
                        and getattr(args, "v7_use_temporal_mismatch_multihead_loss", False)
                        and getattr(args, "v7_temporal_mismatch_multihead_loss_weight", 0.0) > 0.0)
                    or (getattr(args, "v7_use_coverage_entailment_head", False)
                        and getattr(args, "v7_use_coverage_entailment_loss", False)
                        and getattr(args, "v7_coverage_entailment_loss_weight", 0.0) > 0.0)
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
            # Stage26-H1 / Stage27-H2A: H1 bridge flags
            "v7_use_v6b_style_final_decision": getattr(
                args, "v7_use_v6b_style_final_decision", False
            ),
            "v7_use_learnable_ne_alpha": getattr(args, "v7_use_learnable_ne_alpha", False),
            "v7_ne_alpha_init": getattr(args, "v7_ne_alpha_init", 1.0),
            # Stage27-H2A: H1-path entitlement decision signal
            "v7_h1_entitlement_decision_signal": getattr(
                args, "v7_h1_entitlement_decision_signal", "learned"
            ),
            "v7_h1_entitlement_product_power": getattr(
                args, "v7_h1_entitlement_product_power", 1.0
            ),
            "v7_h1_hybrid_residual_beta": getattr(
                args, "v7_h1_hybrid_residual_beta", 0.25
            ),
            # Stage28-I-A: location boundary head / cap configuration
            "v7_use_location_boundary_head": getattr(
                args, "v7_use_location_boundary_head", False
            ),
            "v7_use_location_boundary_loss": getattr(
                args, "v7_use_location_boundary_loss", False
            ),
            "v7_location_boundary_loss_weight": getattr(
                args, "v7_location_boundary_loss_weight", 0.0
            ),
            "v7_location_boundary_cap_mode": getattr(
                args, "v7_location_boundary_cap_mode", "none"
            ),
            "v7_location_boundary_cap_gamma": getattr(
                args, "v7_location_boundary_cap_gamma", 1.0
            ),
            "v7_location_boundary_cap_detach": getattr(
                args, "v7_location_boundary_cap_detach", False
            ),
            "v7_location_boundary_target_mapping": (
                {"location_swap": 0, "none": 1, "paraphrase": 1, "polarity_flip": 1}
                if getattr(args, "v7_use_location_boundary_loss", False) else None
            ),
            "stage15_used_for_v7_location_boundary_loss": False,
            "ood_used_for_v7_location_boundary_loss": False,
            # Stage30-C2: temporal safety head / cap configuration
            "v7_use_temporal_safety_head": getattr(args, "v7_use_temporal_safety_head", False),
            "v7_use_temporal_safety_loss": getattr(args, "v7_use_temporal_safety_loss", False),
            "v7_temporal_safety_loss_weight": getattr(
                args, "v7_temporal_safety_loss_weight", 0.0
            ),
            "v7_temporal_safety_cap_mode": getattr(
                args, "v7_temporal_safety_cap_mode", "none"
            ),
            "v7_temporal_safety_cap_gamma": getattr(
                args, "v7_temporal_safety_cap_gamma", 1.0
            ),
            "v7_temporal_safety_cap_detach": getattr(
                args, "v7_temporal_safety_cap_detach", False
            ),
            "v7_temporal_safety_data": getattr(args, "v7_temporal_safety_data", None),
            "v7_temporal_safety_records_loaded": _ts_meta_records_loaded,
            "v7_temporal_safety_records_used_for_aux_train": _ts_meta_records_used,
            "v7_temporal_safety_records_excluded_main_dev_pair": _ts_meta_excluded_dev,
            "v7_temporal_safety_records_excluded_missing_pair_id": _ts_meta_excluded_missing_pair_id,
            "v7_temporal_safety_stage15_used": False,
            "v7_temporal_safety_external_probe_used": False,
            "stage15_used_for_v7_temporal_safety_loss": False,
            "ood_used_for_v7_temporal_safety_loss": False,
            # Stage30-D: temporal mismatch multihead configuration
            "v7_use_temporal_mismatch_multihead": getattr(
                args, "v7_use_temporal_mismatch_multihead", False
            ),
            "v7_use_temporal_mismatch_multihead_loss": getattr(
                args, "v7_use_temporal_mismatch_multihead_loss", False
            ),
            "v7_temporal_mismatch_multihead_loss_weight": getattr(
                args, "v7_temporal_mismatch_multihead_loss_weight", 0.0
            ),
            "v7_temporal_mismatch_multihead_cap_mode": getattr(
                args, "v7_temporal_mismatch_multihead_cap_mode", "none"
            ),
            "v7_temporal_mismatch_multihead_cap_gamma": getattr(
                args, "v7_temporal_mismatch_multihead_cap_gamma", 1.0
            ),
            "v7_temporal_mismatch_multihead_cap_detach": getattr(
                args, "v7_temporal_mismatch_multihead_cap_detach", False
            ),
            "v7_temporal_mismatch_multihead_fusion": getattr(
                args, "v7_temporal_mismatch_multihead_fusion", "frame_only"
            ),
            "v7_temporal_mismatch_multihead_data": getattr(
                args, "v7_temporal_mismatch_multihead_data", None
            ),
            "v7_temporal_mismatch_multihead_records_loaded": _tmm_meta_records_loaded,
            "v7_temporal_mismatch_multihead_records_used_for_aux_train": _tmm_meta_records_used,
            "v7_temporal_mismatch_multihead_records_excluded_main_dev_pair": _tmm_meta_excluded_dev,
            "v7_temporal_mismatch_multihead_records_excluded_missing_pair_id": (
                _tmm_meta_excluded_missing_pair_id
            ),
            "stage15_used_for_v7_temporal_mismatch_multihead_loss": False,
            "external_probe_used_for_v7_temporal_mismatch_multihead_loss": False,
            # Stage30-E: temporal preservation head and cap configuration
            "v7_use_temporal_preservation_head": getattr(
                args, "v7_use_temporal_preservation_head", False
            ),
            "v7_use_temporal_preservation_loss": getattr(
                args, "v7_use_temporal_preservation_loss", False
            ),
            "v7_temporal_preservation_loss_weight": getattr(
                args, "v7_temporal_preservation_loss_weight", 0.0
            ),
            "v7_use_temporal_preservation_aware_cap": getattr(
                args, "v7_use_temporal_preservation_aware_cap", False
            ),
            "v7_temporal_preservation_cap_gamma": getattr(
                args, "v7_temporal_preservation_cap_gamma", 1.0
            ),
            "v7_temporal_preservation_cap_detach": getattr(
                args, "v7_temporal_preservation_cap_detach", False
            ),
            "v7_temporal_preservation_data": getattr(
                args, "v7_temporal_preservation_data", None
            ),
            "v7_temporal_preservation_records_loaded": _tpres_meta_records_loaded,
            "v7_temporal_preservation_records_used_for_aux_train": _tpres_meta_records_used,
            "v7_temporal_preservation_records_excluded_main_dev_pair": _tpres_meta_excluded_dev,
            "v7_temporal_preservation_records_excluded_missing_pair_id": (
                _tpres_meta_excluded_missing_pair_id
            ),
            "stage15_used_for_v7_temporal_preservation_loss": False,
            "external_probe_used_for_v7_temporal_preservation_loss": False,
            # Stage31-C: coverage/entailment diagnostic owner configuration
            "v7_use_coverage_entailment_head": getattr(
                args, "v7_use_coverage_entailment_head", False
            ),
            "v7_use_coverage_entailment_loss": getattr(
                args, "v7_use_coverage_entailment_loss", False
            ),
            "v7_coverage_entailment_num_classes": getattr(
                args, "v7_coverage_entailment_num_classes", 3
            ),
            "v7_coverage_entailment_input_mode": getattr(
                args, "v7_coverage_entailment_input_mode", "current"
            ),
            "v7_coverage_entailment_loss_weight": getattr(
                args, "v7_coverage_entailment_loss_weight", 0.0
            ),
            "v7_coverage_entailment_data": getattr(
                args, "v7_coverage_entailment_data", None
            ),
            "v7_coverage_entailment_loss_class_weights": getattr(
                args, "v7_coverage_entailment_loss_class_weights", None
            ),
            "v7_coverage_entailment_detach_input": getattr(
                args, "v7_coverage_entailment_detach_input", False
            ),
            "v7_coverage_entailment_records_loaded": _covent_meta_records_loaded,
            "v7_coverage_entailment_train_records": _covent_meta_train_records,
            "v7_coverage_entailment_dev_records": _covent_meta_dev_records,
            "stage31_probe_used_for_v7_coverage_entailment_loss": False,
            "stage31_probe_used_for_checkpoint_selection": False,
            "stage31_probe_used_for_calibration": False,
            "stage31_probe_used_for_threshold_selection": False,
            "v7_coverage_entailment_modifies_final_logits": False,
            "v7_coverage_entailment_modifies_final_predictions": False,
            # Stage32-A: owner-state schema is shadow/export-only.
            "stage32_use_owner_state_schema": getattr(
                args, "stage32_use_owner_state_schema", False
            ),
            "stage32_use_owner_interfaces": getattr(
                args, "stage32_use_owner_interfaces", False
            ),
            "stage32_owner_state_export": getattr(
                args, "stage32_owner_state_export", False
            ),
            "stage32_owner_state_shadow_mode": getattr(
                args, "stage32_owner_state_shadow_mode", False
            ),
            "stage32_owner_state_modifies_final_logits": False,
            "stage32_owner_state_modifies_final_predictions": False,
            "stage32_owner_state_modifies_loss": False,
            "stage32_owner_state_used_for_checkpoint_selection": False,
            "stage32_coverage_owner_v2": getattr(
                args, "stage32_coverage_owner_v2", False
            ),
            "stage32_coverage_owner_v2_min_confidence": getattr(
                args, "stage32_coverage_owner_v2_min_confidence", 0.50
            ),
            "stage32_coverage_owner_v2_min_margin": getattr(
                args, "stage32_coverage_owner_v2_min_margin", 0.05
            ),
            "stage32_coverage_owner_v2_allow_abstain": getattr(
                args, "stage32_coverage_owner_v2_allow_abstain", False
            ),
            "stage32_coverage_owner_v2_modifies_final_logits": False,
            "stage32_coverage_owner_v2_modifies_final_predictions": False,
            "stage33_use_structured_coverage_owner": getattr(
                args, "stage33_use_structured_coverage_owner", False
            ),
            "stage33_structured_coverage_owner_export": getattr(
                args, "stage33_structured_coverage_owner_export", False
            ),
            "stage33_structured_coverage_owner_shadow_mode": getattr(
                args, "stage33_structured_coverage_owner_shadow_mode", False
            ),
            "stage33_structured_coverage_preserve_can_support": getattr(
                args, "stage33_structured_coverage_preserve_can_support", False
            ),
            "stage33_structured_coverage_direct_support_rules": getattr(
                args,
                "stage33_structured_coverage_direct_support_rules",
                _STAGE33_DEFAULT_DIRECT_SUPPORT_RULES,
            ),
            "stage33_structured_coverage_disable_specific_general_direct_support": getattr(
                args,
                "stage33_structured_coverage_disable_specific_general_direct_support",
                False,
            ),
            "stage33_structured_coverage_weak_rules_to_residual": getattr(
                args, "stage33_structured_coverage_weak_rules_to_residual", ""
            ),
            "stage33_structured_coverage_conditional_fallback": getattr(
                args, "stage33_structured_coverage_conditional_fallback", False
            ),
            "stage33_structured_coverage_fallback_source": getattr(
                args, "stage33_structured_coverage_fallback_source", "current_final"
            ),
            "stage33_structured_coverage_enable_whole_part_rules": getattr(
                args, "stage33_structured_coverage_enable_whole_part_rules", False
            ),
            "stage33_structured_coverage_whole_part_direct_support": getattr(
                args, "stage33_structured_coverage_whole_part_direct_support", False
            ),
            "stage33_structured_coverage_whole_part_lexicon": getattr(
                args, "stage33_structured_coverage_whole_part_lexicon", ""
            ),
            "stage33_structured_coverage_whole_part_v2": getattr(
                args, "stage33_structured_coverage_whole_part_v2", False
            ),
            "stage33_structured_coverage_whole_part_v2_use_expanded_lexicon": getattr(
                args,
                "stage33_structured_coverage_whole_part_v2_use_expanded_lexicon",
                False,
            ),
            "stage33_structured_coverage_whole_part_v2_direct_support_policy": getattr(
                args,
                "stage33_structured_coverage_whole_part_v2_direct_support_policy",
                "hard_core_required",
            ),
            "stage33_whole_part_conditional_safe_overrides_hard_core": getattr(
                args,
                "stage33_whole_part_conditional_safe_overrides_hard_core",
                False,
            ),
            "stage33_structured_coverage_modifies_final_logits": False,
            "stage33_structured_coverage_modifies_final_predictions": False,
            "stage36_use_support_safety_blockers": getattr(
                args, "stage36_use_support_safety_blockers", False
            ),
            "stage36_support_safety_export": getattr(
                args, "stage36_support_safety_export", False
            ),
            "stage36_support_safety_shadow_mode": getattr(
                args, "stage36_support_safety_shadow_mode", False
            ),
            "stage36_block_exception_scope": getattr(
                args, "stage36_block_exception_scope", False
            ),
            "stage36_block_not_all_existential": getattr(
                args, "stage36_block_not_all_existential", False
            ),
            "stage36_block_location_scope_mismatch": getattr(
                args, "stage36_block_location_scope_mismatch", False
            ),
            "stage36_block_temporal_scope_mismatch": getattr(
                args, "stage36_block_temporal_scope_mismatch", False
            ),
            "stage36_support_blocker_action": getattr(
                args, "stage36_support_blocker_action", "fallback_current_final"
            ),
            "stage36_support_safety_modifies_final_logits": False,
            "stage36_support_safety_modifies_final_predictions": False,
            "stage37_use_safe_support_recovery": getattr(
                args, "stage37_use_safe_support_recovery", False
            ),
            "stage37_safe_support_export": getattr(
                args, "stage37_safe_support_export", False
            ),
            "stage37_safe_support_shadow_mode": getattr(
                args, "stage37_safe_support_shadow_mode", False
            ),
            "stage37_recover_no_except_included_subset": getattr(
                args, "stage37_recover_no_except_included_subset", False
            ),
            "stage37_recover_coordination_universal_subset": getattr(
                args, "stage37_recover_coordination_universal_subset", False
            ),
            "stage37_recover_numeric_universal_subset": getattr(
                args, "stage37_recover_numeric_universal_subset", False
            ),
            "stage37_allow_recover_from_refute": getattr(
                args, "stage37_allow_recover_from_refute", False
            ),
            "stage37_safe_support_recovery_modifies_final_logits": False,
            "stage37_safe_support_recovery_modifies_final_predictions": False,
            "stage39_use_final_composer_opt_in": getattr(
                args, "stage39_use_final_composer_opt_in", False
            ),
            "stage39_final_composer_export": getattr(
                args, "stage39_final_composer_export", False
            ),
            "stage39_final_composer_policy": getattr(
                args, "stage39_final_composer_policy", "support_only"
            ),
            "stage39_final_composer_output_mode": getattr(
                args, "stage39_final_composer_output_mode", "export_only"
            ),
            "stage39_final_composer_source": getattr(
                args, "stage39_final_composer_source", "stage37_final_shadow_label"
            ),
            "stage39_disallow_refute_to_support": getattr(
                args, "stage39_disallow_refute_to_support", True
            ),
            "stage39_require_stage36_safety_clear": getattr(
                args, "stage39_require_stage36_safety_clear", True
            ),
            "stage39_require_stage37_not_from_refute": getattr(
                args, "stage39_require_stage37_not_from_refute", True
            ),
            "stage39_final_composer_modifies_final_logits": False,
            "stage39_final_composer_modifies_final_predictions_by_default": False,
            "v7_h1_entitlement_for_decision_source": (
                _V7_H1_DECISION_SIGNAL_SOURCE.get(
                    getattr(args, "v7_h1_entitlement_decision_signal", "learned"),
                    "learned",
                )
                if (
                    args.architecture == "v7_hierarchical"
                    and getattr(args, "v7_use_v6b_style_final_decision", False)
                )
                else None
            ),
            # Stage27-H2A reporting bug fix: v7_final_logit_composition now correctly
            # reports "v6b_style_softplus_multiplicative" when H1 is active.
            "v7_final_logit_composition": _resolve_v7_final_logit_composition(args),
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
            # Stage60: train-only Stage57 non-leaking external bridge provenance
            **_stage60_bridge_info,
            # Stage69/Stage70: train-only Stage66 residual bridge provenance
            **_stage66_bridge_info,
            # Stage75C: train-only Stage75 targeted residual bridge provenance
            **_stage75_bridge_info,
            # Stage80D: train-only Stage80A conservative Stage75v2 bridge +
            # combined Stage57+Stage66+Stage75+Stage80A bridge provenance
            **_stage80a_bridge_info,
            **_combined_bridge_info,
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

    if args.stage115_clean_dev_scalar_output_jsonl is not None:
        if len(reports) != 1:
            parser.error(
                "--stage115-clean-dev-scalar-output-jsonl requires a single non-sweep run"
            )
        run_name, _run_report = next(iter(reports.items()))
        if _ood_best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in _ood_best_state.items()})
        model.eval()
        _stage115_output_path = Path(args.stage115_clean_dev_scalar_output_jsonl)
        _stage115_output_path.parent.mkdir(parents=True, exist_ok=True)
        _stage115_prediction_rows = prediction_exports[run_name]
        write_jsonl(
            _stage115_output_path,
            _stage115_clean_dev_scalar_rows(_stage115_prediction_rows),
        )
        report.update(
            _stage115_clean_dev_scalar_report(
                _stage115_prediction_rows,
                _stage115_output_path,
            )
        )
    if args.stage118_diagnostic_jsonl is not None:
        if len(reports) != 1:
            parser.error("--stage118-diagnostic-jsonl requires a single non-sweep run")
        if _ood_best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in _ood_best_state.items()})
        model.eval()
        _stage118_sweep_interfaces = args.stage118_diagnostic_evidence_interface_sweep_list
        if _stage118_sweep_interfaces:
            if args.stage118_diagnostic_evidence_interface != "same_as_vnext":
                print(
                    "[STAGE118 GENERIC DIAGNOSTIC SWEEP] "
                    "--stage118-diagnostic-evidence-interface was also provided; "
                    "running only the deduplicated sweep interfaces. "
                    f"single_interface={args.stage118_diagnostic_evidence_interface} "
                    f"sweep_interfaces={_stage118_sweep_interfaces}"
                )
            _stage118_sweep_report = run_stage118_diagnostic_evidence_interface_sweep(
                model=model,
                input_jsonl=Path(args.stage118_diagnostic_jsonl),
                output_dir=Path(args.stage118_diagnostic_sweep_output_dir),
                diagnostic_name=args.stage118_diagnostic_name,
                batch_size=args.stage118_diagnostic_batch_size,
                sweep_interfaces=_stage118_sweep_interfaces,
                args=args,
                vocab=vocab,
                tokenizer=tokenizer,
                max_length=max_length,
                device=device,
            )
            report["stage118_generic_diagnostic_evidence_interface_sweep"] = _stage118_sweep_report
            print(
                f"[STAGE118 GENERIC DIAGNOSTIC SWEEP] name={args.stage118_diagnostic_name} "
                f"interfaces={_stage118_sweep_interfaces} "
                f"manifest={_stage118_sweep_report['manifest_path']}"
            )
            for _interface, _summary in _stage118_sweep_report["summaries"].items():
                print(
                    f"[STAGE118 GENERIC DIAGNOSTIC SWEEP] interface={_interface} "
                    f"valid={_summary['n_valid_rows']} "
                    f"skipped={_summary['n_skipped_rows']} "
                    f"acc={_summary['accuracy']:.4f} "
                    f"macro_f1={_summary['macro_f1']:.4f}"
                )
        else:
            if args.stage118_diagnostic_output_jsonl is None:
                parser.error(
                    "--stage118-diagnostic-jsonl requires --stage118-diagnostic-output-jsonl"
                )
            if args.stage118_diagnostic_summary_json is None:
                parser.error(
                    "--stage118-diagnostic-jsonl requires --stage118-diagnostic-summary-json"
                )
            _stage118_summary = run_stage118_generic_diagnostic_eval(
                model=model,
                input_jsonl=Path(args.stage118_diagnostic_jsonl),
                output_jsonl=Path(args.stage118_diagnostic_output_jsonl),
                summary_json=Path(args.stage118_diagnostic_summary_json),
                diagnostic_name=args.stage118_diagnostic_name,
                batch_size=args.stage118_diagnostic_batch_size,
                args=args,
                vocab=vocab,
                tokenizer=tokenizer,
                max_length=max_length,
                device=device,
            )
            report["stage118_generic_diagnostic_eval"] = _stage118_summary
            print(
                f"[STAGE118 GENERIC DIAGNOSTIC] name={args.stage118_diagnostic_name} "
                f"valid={_stage118_summary['n_valid_rows']} "
                f"skipped={_stage118_summary['n_skipped_rows']} "
                f"acc={_stage118_summary['accuracy']:.4f} "
                f"macro_f1={_stage118_summary['macro_f1']:.4f}"
            )
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
        # Stage28-E: enrich metadata when schema is stage28e_v1 (default)
        _pred_schema = getattr(args, "prediction_export_schema", "stage28e_v1")
        if _pred_schema == "stage28e_v1":
            metadata["prediction_export_schema_version"] = "stage28e_v1"
            metadata["output_source"] = "best_dev"
            metadata["label_space"] = {"0": "REFUTE", "1": "NOT_ENTITLED", "2": "SUPPORT"}
            metadata["config_summary"] = {
                "v7_h1_entitlement_decision_signal": getattr(
                    args, "v7_h1_entitlement_decision_signal", None
                ),
                "v7_h1_entitlement_product_power": getattr(
                    args, "v7_h1_entitlement_product_power", None
                ),
                "v7_h1_hybrid_residual_beta": getattr(
                    args, "v7_h1_hybrid_residual_beta", None
                ),
                "v7_use_v6b_style_final_decision": getattr(
                    args, "v7_use_v6b_style_final_decision", None
                ),
                "stage32_use_owner_state_schema": getattr(
                    args, "stage32_use_owner_state_schema", False
                ),
                "stage32_use_owner_interfaces": getattr(
                    args, "stage32_use_owner_interfaces", False
                ),
                "stage32_owner_state_export": getattr(
                    args, "stage32_owner_state_export", False
                ),
                "stage32_owner_state_shadow_mode": getattr(
                    args, "stage32_owner_state_shadow_mode", False
                ),
                "stage32_owner_state_modifies_final_logits": False,
                "stage32_owner_state_modifies_final_predictions": False,
                "stage32_coverage_owner_v2": getattr(
                    args, "stage32_coverage_owner_v2", False
                ),
                "stage32_coverage_owner_v2_min_confidence": getattr(
                    args, "stage32_coverage_owner_v2_min_confidence", 0.50
                ),
                "stage32_coverage_owner_v2_min_margin": getattr(
                    args, "stage32_coverage_owner_v2_min_margin", 0.05
                ),
                "stage32_coverage_owner_v2_allow_abstain": getattr(
                    args, "stage32_coverage_owner_v2_allow_abstain", False
                ),
                "stage33_use_structured_coverage_owner": getattr(
                    args, "stage33_use_structured_coverage_owner", False
                ),
                "stage33_structured_coverage_owner_export": getattr(
                    args, "stage33_structured_coverage_owner_export", False
                ),
                "stage33_structured_coverage_owner_shadow_mode": getattr(
                    args, "stage33_structured_coverage_owner_shadow_mode", False
                ),
                "stage33_structured_coverage_preserve_can_support": getattr(
                    args, "stage33_structured_coverage_preserve_can_support", False
                ),
                "stage33_structured_coverage_direct_support_rules": getattr(
                    args,
                    "stage33_structured_coverage_direct_support_rules",
                    _STAGE33_DEFAULT_DIRECT_SUPPORT_RULES,
                ),
                "stage33_structured_coverage_disable_specific_general_direct_support": getattr(
                    args,
                    "stage33_structured_coverage_disable_specific_general_direct_support",
                    False,
                ),
                "stage33_structured_coverage_weak_rules_to_residual": getattr(
                    args, "stage33_structured_coverage_weak_rules_to_residual", ""
                ),
                "stage33_structured_coverage_conditional_fallback": getattr(
                    args, "stage33_structured_coverage_conditional_fallback", False
                ),
                "stage33_structured_coverage_fallback_source": getattr(
                    args, "stage33_structured_coverage_fallback_source", "current_final"
                ),
                "stage33_structured_coverage_enable_whole_part_rules": getattr(
                    args, "stage33_structured_coverage_enable_whole_part_rules", False
                ),
                "stage33_structured_coverage_whole_part_direct_support": getattr(
                    args, "stage33_structured_coverage_whole_part_direct_support", False
                ),
                "stage33_structured_coverage_whole_part_lexicon": getattr(
                    args, "stage33_structured_coverage_whole_part_lexicon", ""
                ),
                "stage33_structured_coverage_whole_part_v2": getattr(
                    args, "stage33_structured_coverage_whole_part_v2", False
                ),
                "stage33_structured_coverage_whole_part_v2_use_expanded_lexicon": getattr(
                    args,
                    "stage33_structured_coverage_whole_part_v2_use_expanded_lexicon",
                    False,
                ),
                "stage33_structured_coverage_whole_part_v2_direct_support_policy": getattr(
                    args,
                    "stage33_structured_coverage_whole_part_v2_direct_support_policy",
                    "hard_core_required",
                ),
                "stage33_whole_part_conditional_safe_overrides_hard_core": getattr(
                    args,
                    "stage33_whole_part_conditional_safe_overrides_hard_core",
                    False,
                ),
                "stage33_structured_coverage_modifies_final_logits": False,
                "stage33_structured_coverage_modifies_final_predictions": False,
                "stage36_use_support_safety_blockers": getattr(
                    args, "stage36_use_support_safety_blockers", False
                ),
                "stage36_support_safety_export": getattr(
                    args, "stage36_support_safety_export", False
                ),
                "stage36_support_safety_shadow_mode": getattr(
                    args, "stage36_support_safety_shadow_mode", False
                ),
                "stage36_block_exception_scope": getattr(
                    args, "stage36_block_exception_scope", False
                ),
                "stage36_block_not_all_existential": getattr(
                    args, "stage36_block_not_all_existential", False
                ),
                "stage36_block_location_scope_mismatch": getattr(
                    args, "stage36_block_location_scope_mismatch", False
                ),
                "stage36_block_temporal_scope_mismatch": getattr(
                    args, "stage36_block_temporal_scope_mismatch", False
                ),
                "stage36_support_blocker_action": getattr(
                    args, "stage36_support_blocker_action", "fallback_current_final"
                ),
                "stage36_support_safety_modifies_final_logits": False,
                "stage36_support_safety_modifies_final_predictions": False,
                "stage37_use_safe_support_recovery": getattr(
                    args, "stage37_use_safe_support_recovery", False
                ),
                "stage37_safe_support_export": getattr(
                    args, "stage37_safe_support_export", False
                ),
                "stage37_safe_support_shadow_mode": getattr(
                    args, "stage37_safe_support_shadow_mode", False
                ),
                "stage37_recover_no_except_included_subset": getattr(
                    args, "stage37_recover_no_except_included_subset", False
                ),
                "stage37_recover_coordination_universal_subset": getattr(
                    args, "stage37_recover_coordination_universal_subset", False
                ),
                "stage37_recover_numeric_universal_subset": getattr(
                    args, "stage37_recover_numeric_universal_subset", False
                ),
                "stage37_allow_recover_from_refute": getattr(
                    args, "stage37_allow_recover_from_refute", False
                ),
                "stage37_safe_support_recovery_modifies_final_logits": False,
                "stage37_safe_support_recovery_modifies_final_predictions": False,
                "stage39_use_final_composer_opt_in": getattr(
                    args, "stage39_use_final_composer_opt_in", False
                ),
                "stage39_final_composer_export": getattr(
                    args, "stage39_final_composer_export", False
                ),
                "stage39_final_composer_policy": getattr(
                    args, "stage39_final_composer_policy", "support_only"
                ),
                "stage39_final_composer_output_mode": getattr(
                    args, "stage39_final_composer_output_mode", "export_only"
                ),
                "stage39_final_composer_source": getattr(
                    args,
                    "stage39_final_composer_source",
                    "stage37_final_shadow_label",
                ),
                "stage39_disallow_refute_to_support": getattr(
                    args, "stage39_disallow_refute_to_support", True
                ),
                "stage39_require_stage36_safety_clear": getattr(
                    args, "stage39_require_stage36_safety_clear", True
                ),
                "stage39_require_stage37_not_from_refute": getattr(
                    args, "stage39_require_stage37_not_from_refute", True
                ),
                "stage39_final_composer_modifies_final_logits": False,
                "stage39_final_composer_modifies_final_predictions_by_default": False,
                "freeze_encoder": getattr(args, "freeze_encoder", None),
                "freeze_a_log": getattr(args, "freeze_a_log", None),
                "max_length": getattr(args, "max_length", None),
                # Stage28-I-A: location boundary head / cap config
                "v7_use_location_boundary_head": getattr(
                    args, "v7_use_location_boundary_head", False
                ),
                "v7_use_location_boundary_loss": getattr(
                    args, "v7_use_location_boundary_loss", False
                ),
                "v7_location_boundary_loss_weight": getattr(
                    args, "v7_location_boundary_loss_weight", 0.0
                ),
                "v7_location_boundary_cap_mode": getattr(
                    args, "v7_location_boundary_cap_mode", "none"
                ),
                "v7_location_boundary_cap_gamma": getattr(
                    args, "v7_location_boundary_cap_gamma", 1.0
                ),
                "v7_location_boundary_cap_detach": getattr(
                    args, "v7_location_boundary_cap_detach", False
                ),
                # Stage30-C2: temporal safety head / cap config
                "v7_use_temporal_safety_head": getattr(
                    args, "v7_use_temporal_safety_head", False
                ),
                "v7_use_temporal_safety_loss": getattr(
                    args, "v7_use_temporal_safety_loss", False
                ),
                "v7_temporal_safety_loss_weight": getattr(
                    args, "v7_temporal_safety_loss_weight", 0.0
                ),
                "v7_temporal_safety_cap_mode": getattr(
                    args, "v7_temporal_safety_cap_mode", "none"
                ),
                "v7_temporal_safety_cap_gamma": getattr(
                    args, "v7_temporal_safety_cap_gamma", 1.0
                ),
                "v7_temporal_safety_cap_detach": getattr(
                    args, "v7_temporal_safety_cap_detach", False
                ),
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
            # TD constrained selection results ??lifted from run report to top level
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
            # Stage44-B internal anti-collapse selection results
            "stage44b_enabled",
            "stage44b_decision",
            "stage44b_selection_mode",
            "stage44b_constraints",
            "stage44b_selected_epoch",
            "stage44b_original_best_metric_epoch",
            "stage44b_selected_by_constraints",
            "stage44b_constraints_satisfied",
            "stage44b_num_candidate_epochs",
            "stage44b_num_constraint_satisfying_epochs",
            "stage44b_candidate_table",
            "stage44b_selected_metrics",
            "stage44b_original_best_metric_metrics",
            "stage44b_not_entitled_prediction_rate",
            "stage44b_support_recall",
            "stage44b_refute_recall",
            "stage44b_leakage_policy",
            "stage44b_recommendation",
            "stage44b2_enabled",
            "stage44b2_decision",
            "stage44b2_prior_aware_enabled",
            "stage44b2_gold_label_rates",
            "stage44b2_gold_not_entitled_rate",
            "stage44b2_gold_support_rate",
            "stage44b2_gold_refute_rate",
            "stage44b2_prior_aware_constraints",
            "stage44b2_selected_epoch",
            "stage44b2_original_best_metric_epoch",
            "stage44b2_selected_metrics",
            "stage44b2_original_best_metric_metrics",
            "stage44b2_ne_pred_minus_gold_ne_rate",
            "stage44b2_original_best_ne_pred_minus_gold_ne_rate",
            "stage44b2_reason_previous_fixed_cap_was_invalid",
            "stage44b2_recommendation",
            "stage45b_enabled",
            "stage45b_decision",
            "stage45b_family_field_used",
            "stage45b_holdout_family",
            "stage45b_train_rows",
            "stage45b_holdout_rows",
            "stage45b_train_label_counts",
            "stage45b_holdout_label_counts",
            "stage45b_holdout_metrics",
            "stage45b_leakage_policy",
            "stage45b_recommendation",
            "stage45b2_decision",
            "stage45b2_intervention_objective_guard_enabled",
            "stage45b2_intervention_objective_missing_variant_counts",
            "stage45b2_intervention_objective_skipped_group_count",
            "stage45b2_intervention_objective_active_group_count",
            "stage45b2_intervention_objective_effectively_inactive",
            "stage45b2_recommendation",
            "stage45b3_decision",
            "stage45b3_pairwise_check_guard_enabled",
            "stage45b3_pairwise_missing_variant_counts",
            "stage45b3_pairwise_skipped_missing_none_count",
            "stage45b3_pairwise_active_group_count",
            "stage45b3_pairwise_checks_effectively_inactive",
            "stage45b3_recommendation",
            "stage45c_enabled",
            "stage45c_support_recovery_weight",
            "stage45c_entitled_ne_penalty_weight",
            "stage45c_target_label",
            "stage45c_entitled_labels",
            "stage45c_train_support_count",
            "stage45c_train_refute_count",
            "stage45c_train_not_entitled_count",
            "stage45c_loss_terms_active",
            "stage45c_support_recovery_loss_mean",
            "stage45c_entitled_ne_penalty_loss_mean",
            "stage45c_leakage_policy",
            "stage45c_recommendation",
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
            # Stage60: train-only Stage57 non-leaking external bridge provenance
            "stage57_bridge_train_mode",
            "stage57_bridge_train_jsonl",
            "stage57_bridge_train_enabled",
            "stage57_bridge_train_row_count",
            "stage57_bridge_train_label_counts",
            "stage57_bridge_train_family_counts",
            "stage57_bridge_train_family_label_counts",
            "stage57_bridge_train_only",
            "stage57_bridge_appended_after_clean_split",
            "stage57_bridge_used_for_dev",
            "stage57_bridge_used_for_checkpoint_selection",
            "stage57_external_data_used_for_training",
            "stage57_external_metrics_used_for_threshold_tuning",
            "stage57_bridge_enabled",
            "stage57_bridge_row_count",
            "stage57_bridge_label_counts",
            "stage57_bridge_family_counts",
            "stage57_used_for_dev",
            "stage57_used_for_checkpoint_selection",
            # Stage69/Stage70: train-only Stage66 residual bridge provenance
            "stage66_bridge_train_mode",
            "stage66_bridge_train_jsonl",
            "stage66_bridge_train_enabled",
            "stage66_bridge_train_row_count",
            "stage66_bridge_train_label_counts",
            "stage66_bridge_train_family_counts",
            "stage66_bridge_train_family_label_counts",
            "stage66_bridge_train_only",
            "stage66_bridge_appended_after_clean_split",
            "stage66_bridge_used_for_dev",
            "stage66_bridge_used_for_checkpoint_selection",
            "stage66_external_data_used_for_training",
            "stage66_external_metrics_used_for_threshold_tuning",
            "stage66_bridge_enabled",
            "stage66_bridge_row_count",
            "stage66_bridge_label_counts",
            "stage66_bridge_family_counts",
            "stage66_used_for_dev",
            "stage66_used_for_checkpoint_selection",
            # Stage75C: train-only Stage75 targeted residual bridge provenance
            "stage75_bridge_train_mode",
            "stage75_bridge_train_jsonl",
            "stage75_bridge_train_enabled",
            "stage75_bridge_train_row_count",
            "stage75_bridge_train_label_counts",
            "stage75_bridge_train_family_counts",
            "stage75_bridge_train_family_label_counts",
            "stage75_bridge_train_only",
            "stage75_bridge_appended_after_clean_split",
            "stage75_bridge_used_for_dev",
            "stage75_bridge_used_for_checkpoint_selection",
            "stage75_external_data_used_for_training",
            "stage75_external_metrics_used_for_threshold_tuning",
            "stage75_bridge_enabled",
            "stage75_bridge_row_count",
            "stage75_bridge_label_counts",
            "stage75_bridge_family_counts",
            "stage75_used_for_dev",
            "stage75_used_for_checkpoint_selection",
            # Stage80D: train-only Stage80A conservative Stage75v2 bridge provenance
            "stage80a_bridge_train_mode",
            "stage80a_bridge_train_jsonl",
            "stage80a_bridge_train_enabled",
            "stage80a_bridge_train_row_count",
            "stage80a_bridge_train_label_counts",
            "stage80a_bridge_train_family_counts",
            "stage80a_bridge_train_family_label_counts",
            "stage80a_bridge_train_only",
            "stage80a_bridge_appended_after_clean_split",
            "stage80a_bridge_used_for_dev",
            "stage80a_bridge_used_for_checkpoint_selection",
            "stage80a_external_data_used_for_training",
            "stage80a_external_metrics_used_for_threshold_tuning",
            "stage80a_bridge_enabled",
            "stage80a_bridge_row_count",
            "stage80a_bridge_label_counts",
            "stage80a_bridge_family_counts",
            "stage80a_used_for_dev",
            "stage80a_used_for_checkpoint_selection",
            # Stage80D: combined Stage57+Stage66+Stage75+Stage80A bridge provenance
            "combined_bridge_enabled",
            "combined_bridge_row_count",
            "combined_bridge_label_counts",
            "combined_bridge_train_only",
            "bridge_sources_enabled",
            "clean_dev_for_checkpoint_selection",
            "external_data_used_for_training",
            "external_metrics_used_for_threshold_tuning",
            "time_swap_used",
            "bridge_rows_excluded_from_intervention_pairwise_loss",
            "stage57_excluded_from_intervention_pairwise_loss",
            "stage66_excluded_from_intervention_pairwise_loss",
            "stage75_excluded_from_intervention_pairwise_loss",
            "stage80a_excluded_from_intervention_pairwise_loss",
            "intervention_pairwise_loss_source",
            "intervention_pairwise_loss_clean_main_row_count",
            "intervention_pairwise_loss_bridge_row_count_excluded",
            "intervention_pairwise_loss_stage57_row_count_excluded",
            "intervention_pairwise_loss_stage66_row_count_excluded",
            "intervention_pairwise_loss_stage75_row_count_excluded",
            "intervention_pairwise_loss_stage80a_row_count_excluded",
            "final_train_row_count_expected",
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
    # G2: source=dev (default) ??same pool as before.
    # G3: source=train or train_dev ??larger pool without any OOD records.
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
                    **_vnext_model_feature_inputs(inputs),
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

        # Record sets derived from intervention_type only ??no Stage15 labels used
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

        # Joint (penalty, shift) sweep ??nested dict keyed by penalty then shift
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
                # would handle ??instead we check explicitly).
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
        # This is logging only ??no model behavior, loss, or logit change.
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
            "stage44_use_anti_collapse_selection": getattr(
                args, "stage44_use_anti_collapse_selection", False
            ),
            "stage44_min_support_recall": getattr(args, "stage44_min_support_recall", None),
            "stage44_min_refute_recall": getattr(args, "stage44_min_refute_recall", None),
            "stage44_max_not_entitled_pred_rate": getattr(
                args, "stage44_max_not_entitled_pred_rate", None
            ),
            "stage44_min_clean_dev_accuracy": getattr(
                args, "stage44_min_clean_dev_accuracy", None
            ),
            "stage44_selection_fallback": getattr(
                args, "stage44_selection_fallback", "best_metric"
            ),
            "stage44_use_prior_aware_ne_constraint": getattr(
                args, "stage44_use_prior_aware_ne_constraint", False
            ),
            "stage44_max_ne_gold_prior_delta": getattr(
                args, "stage44_max_ne_gold_prior_delta", None
            ),
            "stage44_min_macro_f1": getattr(args, "stage44_min_macro_f1", None),
            "stage44_min_relative_macro_f1_of_best": getattr(
                args, "stage44_min_relative_macro_f1_of_best", None
            ),
            "stage44_min_support_precision": getattr(
                args, "stage44_min_support_precision", None
            ),
            "stage44_min_refute_precision": getattr(
                args, "stage44_min_refute_precision", None
            ),
            "stage45_use_family_holdout": getattr(
                args, "stage45_use_family_holdout", False
            ),
            "stage45_family_field": getattr(args, "stage45_family_field", "auto"),
            "stage45_holdout_family": getattr(args, "stage45_holdout_family", None),
            "stage45_min_holdout_size": getattr(args, "stage45_min_holdout_size", 20),
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
                    **_vnext_model_feature_inputs(ood_inputs),
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
                    **_vnext_model_feature_inputs(ood_inputs),
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
        # to the main report ??Stage15 OOD labels never used for shift selection)
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
                    **_vnext_model_feature_inputs(ood_inputs),
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

    # ---------------------------------------------------------------------------
    # Stage29-B: external probe evaluation (eval-only)
    # Runs AFTER checkpoint selection and best-state loading.
    # External probe records are NEVER used for training, selection, or calibration.
    # Only runs when --external-eval-jsonl is provided; default behaviour unchanged.
    # ---------------------------------------------------------------------------
    _ext_jsonl_paths: list[str] = getattr(args, "external_eval_jsonl", []) or []
    if _ext_jsonl_paths:
        import os as _os

        _ext_names_raw: list[str] = getattr(args, "external_eval_name", []) or []
        _ext_output_dir: "str | None" = getattr(args, "external_output_dir", None)

        if _ext_names_raw and len(_ext_names_raw) != len(_ext_jsonl_paths):
            raise ValueError(
                f"--external-eval-name count ({len(_ext_names_raw)}) must match "
                f"--external-eval-jsonl count ({len(_ext_jsonl_paths)}). "
                "Either omit all names (auto-derived from file stem) or provide "
                "exactly one name per JSONL path."
            )

        _ext_names: list[str] = (
            _ext_names_raw
            if _ext_names_raw
            else [Path(p).stem for p in _ext_jsonl_paths]
        )

        # Load best checkpoint (same logic as OOD eval)
        if _ood_best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in _ood_best_state.items()})
        model.eval()
        _ext_eval_state = "best_dev" if _ood_best_state is not None else "final_epoch_fallback"
        _ext_eval_epoch = _ood_best_epoch

        if _ext_output_dir is not None:
            _os.makedirs(_ext_output_dir, exist_ok=True)

        _external_evals: dict[str, Any] = {}

        for _ext_path_str, _ext_name in zip(_ext_jsonl_paths, _ext_names):
            _ext_path = Path(_ext_path_str)
            print(
                f"[EXTERNAL PROBE S29B] name={_ext_name} path={_ext_path} "
                f"eval_state={_ext_eval_state} epoch={_ext_eval_epoch}"
            )

            # Load and normalize records. Stage73: records from an external
            # fact-verification schema (e.g. Stage43B1 / VitaminC, which uses
            # `label` instead of `final_label` and has no auxiliary labels) are
            # mapped onto the controlled v5/v6 schema here; records that already
            # use the controlled schema pass through unchanged aside from the
            # additive provenance metadata fields.
            _ext_records_raw = load_external_probe_jsonl(_ext_path)
            _ext_records, _ext_schema_report = normalize_external_factver_records(
                _ext_records_raw
            )

            # Encode (reuses same backbone/tokenizer as training)
            if args.backbone == "dummy":
                _ext_bundle = v5.encode_records(_ext_records, vocab)
            else:
                _ext_bundle = v5.encode_mamba_records(_ext_records, tokenizer, args.max_length)
            _ext_inputs = v5.move_inputs(_ext_bundle["model_inputs"], device)

            # Align sequence length to training max_length (pad or truncate)
            _ext_seq = _ext_inputs["input_ids"].shape[1]
            if _ext_seq < max_length:
                for _k in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _ext_inputs[_k] = F.pad(
                        _ext_inputs[_k], (0, max_length - _ext_seq), value=0
                    )
            elif _ext_seq > max_length:
                for _k in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                    _ext_inputs[_k] = _ext_inputs[_k][:, :max_length]

            # Eval-only forward pass ??no loss, no gradient
            _ext_result, _ext_pred_records = evaluate_external_probe(
                model=model,
                probe_records=_ext_records,
                probe_inputs=_ext_inputs,
                flag_source=args.flag_source,
                device=device,
                eval_name=_ext_name,
                eval_path=str(_ext_path),
                args=args,
            )
            _ext_result["eval_state"] = _ext_eval_state
            _ext_result["eval_epoch"] = _ext_eval_epoch

            # Stage73: external fact-verification schema normalization reporting
            # (requirement 9). Reflects only eval-time schema mapping; never
            # affects training, calibration, threshold selection, or checkpoint
            # selection.
            _ext_result.update(_ext_schema_report)
            _ext_result["external_eval_jsonl"] = str(_ext_path)
            _ext_result["external_eval_name"] = _ext_name

            # Write per-probe prediction JSON (optional)
            _ext_pred_out_str: "str | None" = None
            if _ext_output_dir is not None:
                # Keep as Path for write_predictions_json (which calls path.parent.mkdir)
                _ext_pred_out_path_obj: Path = (
                    Path(_ext_output_dir)
                    / f"external_probe_{_ext_name}_predictions.json"
                )
                # String copy for JSON-serialisable metadata / report
                _ext_pred_out_str = str(_ext_pred_out_path_obj)
                _ext_pred_metadata: dict[str, Any] = {
                    "external_eval_name": _ext_name,
                    "external_eval_path": str(_ext_path),
                    "prediction_export_schema_version": "stage28e_v1",
                    "label_space": {"0": "REFUTE", "1": "NOT_ENTITLED", "2": "SUPPORT"},
                    "n_records": len(_ext_records),
                    "eval_state": _ext_eval_state,
                    "eval_epoch": _ext_eval_epoch,
                    "stage29b_external_probe": True,
                    "stage15_used_for_training": False,
                    "stage15_used_for_checkpoint_selection": False,
                    "external_probe_used_for_checkpoint_selection": False,
                    "external_probe_used_for_calibration": False,
                    "external_probe_used_for_training": False,
                    "config_summary": _ext_result["config_summary"],
                    "external_schema_normalized": _ext_schema_report["external_schema_normalized"],
                    "external_schema_normalization_source": _ext_schema_report[
                        "external_schema_normalization_source"
                    ],
                    "external_schema_missing_final_label_fixed": _ext_schema_report[
                        "external_schema_missing_final_label_fixed"
                    ],
                    "external_schema_label_field_used": _ext_schema_report[
                        "external_schema_label_field_used"
                    ],
                    "external_schema_records_normalized": _ext_schema_report[
                        "external_schema_records_normalized"
                    ],
                    "external_schema_records_with_added_aux_labels": _ext_schema_report[
                        "external_schema_records_with_added_aux_labels"
                    ],
                }
                v5.write_predictions_json(
                    _ext_pred_out_path_obj, _ext_pred_metadata, _ext_pred_records
                )

            _ext_result["output_predictions_path"] = _ext_pred_out_str
            _external_evals[_ext_name] = _ext_result

            print(
                f"[EXTERNAL PROBE S29B] {_ext_name} "
                f"n={_ext_result['n_records']} "
                f"acc={_ext_result['final_accuracy']:.4f} "
                f"macro_f1={_ext_result['final_macro_f1']:.4f} "
                f"false_SUPPORT={_ext_result['false_SUPPORT_total']}"
            )

        report["external_evals"] = _external_evals


    # Stage43-C0: external fact-verification evaluation (eval-only)
    # Runs AFTER normal training, clean-dev checkpoint selection, and best-state
    # restoration. Never used for training, calibration, threshold selection,
    # checkpoint selection, loss design, or model/composer behavior changes.
    _stage43_paths: list[str] = getattr(args, "stage43_external_factver_jsonl", []) or []
    _stage43_enabled = bool(getattr(args, "enable_stage43_external_eval", False))
    if _stage43_paths and not _stage43_enabled:
        report["stage43_external_factver_eval_skipped"] = {
            "reason": "--stage43-external-factver-jsonl was supplied but --enable-stage43-external-eval was not set",
            "requested_paths": _stage43_paths,
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "used_for_threshold_selection": False,
        }
    if _stage43_enabled:
        if not _stage43_paths:
            raise ValueError(
                "--enable-stage43-external-eval requires at least one "
                "--stage43-external-factver-jsonl path"
            )
        if _ood_best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in _ood_best_state.items()})
        model.eval()
        _stage43_batch_size = getattr(args, "stage43_external_batch_size", None)
        if _stage43_batch_size is None:
            _stage43_batch_size = getattr(args, "batch_size", 8)
        _stage43_hook_result = run_stage43_external_factver_hook(
            model=model,
            jsonl_paths=_stage43_paths,
            output_dir=Path(getattr(args, "stage43_external_output_dir", "reports")),
            run_prefix=getattr(args, "stage43_external_run_prefix", "stage43c0"),
            max_rows=getattr(args, "stage43_external_max_rows", None),
            batch_size=int(_stage43_batch_size),
            args=args,
            vocab=vocab,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
        )
        report["stage43_external_factver_eval"] = _stage43_hook_result

    # Stage26-D: lift architecture metadata and dev metric aliases to root level.
    lift_report_aliases(report)

    print("\nFINAL_REPORT")
    if getattr(args, "stage44_selection_report_json", None) is not None:
        _stage44_report_keys = [
            "stage44b_enabled",
            "stage44b_decision",
            "stage44b_selection_mode",
            "stage44b_constraints",
            "stage44b_selected_epoch",
            "stage44b_original_best_metric_epoch",
            "stage44b_selected_by_constraints",
            "stage44b_constraints_satisfied",
            "stage44b_num_candidate_epochs",
            "stage44b_num_constraint_satisfying_epochs",
            "stage44b_candidate_table",
            "stage44b_selected_metrics",
            "stage44b_original_best_metric_metrics",
            "stage44b_not_entitled_prediction_rate",
            "stage44b_support_recall",
            "stage44b_refute_recall",
            "stage44b_leakage_policy",
            "stage44b_recommendation",
            "stage44b2_enabled",
            "stage44b2_decision",
            "stage44b2_prior_aware_enabled",
            "stage44b2_gold_label_rates",
            "stage44b2_gold_not_entitled_rate",
            "stage44b2_gold_support_rate",
            "stage44b2_gold_refute_rate",
            "stage44b2_prior_aware_constraints",
            "stage44b2_selected_epoch",
            "stage44b2_original_best_metric_epoch",
            "stage44b2_selected_metrics",
            "stage44b2_original_best_metric_metrics",
            "stage44b2_ne_pred_minus_gold_ne_rate",
            "stage44b2_original_best_ne_pred_minus_gold_ne_rate",
            "stage44b2_reason_previous_fixed_cap_was_invalid",
            "stage44b2_recommendation",
        ]
        _stage44_report = {
            key: report.get(key)
            for key in _stage44_report_keys
            if key in report
        }
        if "stage44b_decision" not in _stage44_report:
            _stage44_report["per_run_stage44b"] = {
                name: {
                    key: run_report.get(key)
                    for key in _stage44_report_keys
                    if key in run_report
                }
                for name, run_report in reports.items()
            }
        _stage44_report.update(
            {
                "stage44b_report_scope": "internal_clean_dev_only",
                "stage43b1_files_read": False,
                "stage43b1_used_for_threshold_selection": False,
                "stage43b1_used_for_checkpoint_selection": False,
                "stage43b1_used_for_calibration": False,
                "stage43b1_used_for_loss_design": False,
                "stage43b1_used_for_model_selection": False,
                "stage43b1_used_for_composer_behavior_changes": False,
            }
        )
        v5.write_report_json(_stage44_report, args.stage44_selection_report_json)

    if (
        getattr(args, "stage45_family_holdout_report_json", None) is not None
        or getattr(args, "stage45_family_holdout_report_md", None) is not None
    ):
        _stage45_report_keys = [
            "stage45b_enabled",
            "stage45b_decision",
            "stage45b_family_field_used",
            "stage45b_holdout_family",
            "stage45b_train_rows",
            "stage45b_holdout_rows",
            "stage45b_train_label_counts",
            "stage45b_holdout_label_counts",
            "stage45b_holdout_metrics",
            "stage45b_leakage_policy",
            "stage45b_recommendation",
            "stage45b2_decision",
            "stage45b2_intervention_objective_guard_enabled",
            "stage45b2_intervention_objective_missing_variant_counts",
            "stage45b2_intervention_objective_skipped_group_count",
            "stage45b2_intervention_objective_active_group_count",
            "stage45b2_intervention_objective_effectively_inactive",
            "stage45b2_recommendation",
            "stage45b3_decision",
            "stage45b3_pairwise_check_guard_enabled",
            "stage45b3_pairwise_missing_variant_counts",
            "stage45b3_pairwise_skipped_missing_none_count",
            "stage45b3_pairwise_active_group_count",
            "stage45b3_pairwise_checks_effectively_inactive",
            "stage45b3_recommendation",
        ]
        _stage45_report = {
            key: report.get(key)
            for key in _stage45_report_keys
            if key in report
        }
        if "stage45b_decision" not in _stage45_report:
            _stage45_report["per_run_stage45b"] = {
                name: {
                    key: run_report.get(key)
                    for key in _stage45_report_keys
                    if key in run_report
                }
                for name, run_report in reports.items()
            }
        _stage45_report.update(
            {
                "stage45b_report_scope": "internal_controlled_family_holdout_only",
                "stage43b1_files_read": False,
                "external_factver_files_read": False,
                "external_examples_used": False,
                "external_labels_or_metrics_used": False,
                "stage43b1_used_for_threshold_selection": False,
                "stage43b1_used_for_checkpoint_selection": False,
                "stage43b1_used_for_calibration": False,
                "stage43b1_used_for_loss_design": False,
                "stage43b1_used_for_model_selection": False,
                "stage43b1_used_for_composer_behavior_changes": False,
            }
        )
        if getattr(args, "stage45_family_holdout_report_json", None) is not None:
            v5.write_report_json(_stage45_report, args.stage45_family_holdout_report_json)
        if getattr(args, "stage45_family_holdout_report_md", None) is not None:
            _stage45_metrics = _stage45_report.get("stage45b_holdout_metrics") or {}
            _stage45_md_lines = [
                "# Stage45-B Internal Family Holdout Report",
                "",
                "## Decision",
                "",
                f"`{_stage45_report.get('stage45b_decision')}`",
                "",
                "## Split",
                "",
                f"- Enabled: {_stage45_report.get('stage45b_enabled')}",
                f"- Family field used: `{_stage45_report.get('stage45b_family_field_used')}`",
                f"- Holdout family: `{_stage45_report.get('stage45b_holdout_family')}`",
                f"- Train rows: {_stage45_report.get('stage45b_train_rows')}",
                f"- Holdout rows: {_stage45_report.get('stage45b_holdout_rows')}",
                "",
                "## Holdout Metrics",
                "",
                f"- Accuracy: {_stage45_metrics.get('accuracy')}",
                f"- Macro-F1: {_stage45_metrics.get('macro_f1')}",
                f"- NOT_ENTITLED prediction rate: {_stage45_metrics.get('not_entitled_prediction_rate')}",
                f"- SUPPORT recall: {_stage45_metrics.get('support_recall')}",
                f"- REFUTE recall: {_stage45_metrics.get('refute_recall')}",
                "",
                "## Stage45-B2 Intervention Objective Guard",
                "",
                f"`{_stage45_report.get('stage45b2_decision')}`",
                "",
                f"- Guard enabled: {_stage45_report.get('stage45b2_intervention_objective_guard_enabled')}",
                f"- Missing variant counts: {_stage45_report.get('stage45b2_intervention_objective_missing_variant_counts')}",
                f"- Skipped group count: {_stage45_report.get('stage45b2_intervention_objective_skipped_group_count')}",
                f"- Active group count: {_stage45_report.get('stage45b2_intervention_objective_active_group_count')}",
                f"- Effectively inactive: {_stage45_report.get('stage45b2_intervention_objective_effectively_inactive')}",
                "",
                str(_stage45_report.get("stage45b2_recommendation")),
                "",
                "## Stage45-B3 Pairwise Check Guard",
                "",
                f"`{_stage45_report.get('stage45b3_decision')}`",
                "",
                f"- Guard enabled: {_stage45_report.get('stage45b3_pairwise_check_guard_enabled')}",
                f"- Missing variant counts: {_stage45_report.get('stage45b3_pairwise_missing_variant_counts')}",
                f"- Groups skipped (missing none): {_stage45_report.get('stage45b3_pairwise_skipped_missing_none_count')}",
                f"- Active group count: {_stage45_report.get('stage45b3_pairwise_active_group_count')}",
                f"- Effectively inactive: {_stage45_report.get('stage45b3_pairwise_checks_effectively_inactive')}",
                "",
                str(_stage45_report.get("stage45b3_recommendation")),
                "",
                "## Leakage Policy",
                "",
                str(_stage45_report.get("stage45b_leakage_policy")),
                "",
                "## Recommendation",
                "",
                str(_stage45_report.get("stage45b_recommendation")),
                "",
            ]
            write_text(args.stage45_family_holdout_report_md, "\n".join(_stage45_md_lines))

    if (
        getattr(args, "stage45c_report_json", None) is not None
        or getattr(args, "stage45c_report_md", None) is not None
    ):
        _stage45c_report_keys = [
            "stage45c_enabled",
            "stage45c_support_recovery_weight",
            "stage45c_entitled_ne_penalty_weight",
            "stage45c_target_label",
            "stage45c_entitled_labels",
            "stage45c_train_support_count",
            "stage45c_train_refute_count",
            "stage45c_train_not_entitled_count",
            "stage45c_loss_terms_active",
            "stage45c_support_recovery_loss_mean",
            "stage45c_entitled_ne_penalty_loss_mean",
            "stage45c_leakage_policy",
            "stage45c_recommendation",
        ]
        _stage45c_standalone_report = {
            key: report.get(key)
            for key in _stage45c_report_keys
            if key in report
        }
        if "stage45c_enabled" not in _stage45c_standalone_report:
            _stage45c_standalone_report["per_run_stage45c"] = {
                name: {
                    key: run_report.get(key)
                    for key in _stage45c_report_keys
                    if key in run_report
                }
                for name, run_report in reports.items()
            }
        _stage45c_standalone_report.update(
            {
                "stage45c_report_scope": "internal_controlled_training_split_only",
                "stage43b1_files_read": False,
                "external_examples_used": False,
                "external_labels_or_metrics_used": False,
                "vitaminc_used": False,
                "climate_fever_used": False,
                "used_for_threshold_selection": False,
                "used_for_calibration": False,
                "used_for_checkpoint_selection": False,
                "used_dev_or_holdout_labels_in_loss": False,
            }
        )
        if getattr(args, "stage45c_report_json", None) is not None:
            v5.write_report_json(_stage45c_standalone_report, args.stage45c_report_json)
        if getattr(args, "stage45c_report_md", None) is not None:
            _stage45c_md_lines = [
                "# Stage45-C Internal SUPPORT Entitlement Recovery Report",
                "",
                "## Enabled",
                "",
                f"- Enabled: {_stage45c_standalone_report.get('stage45c_enabled')}",
                f"- Support recovery weight: {_stage45c_standalone_report.get('stage45c_support_recovery_weight')}",
                f"- Entitled NE penalty weight: {_stage45c_standalone_report.get('stage45c_entitled_ne_penalty_weight')}",
                f"- Target label: `{_stage45c_standalone_report.get('stage45c_target_label')}`",
                f"- Entitled labels: {_stage45c_standalone_report.get('stage45c_entitled_labels')}",
                "",
                "## Internal Training Label Counts",
                "",
                f"- SUPPORT: {_stage45c_standalone_report.get('stage45c_train_support_count')}",
                f"- REFUTE: {_stage45c_standalone_report.get('stage45c_train_refute_count')}",
                f"- NOT_ENTITLED: {_stage45c_standalone_report.get('stage45c_train_not_entitled_count')}",
                "",
                "## Loss Terms",
                "",
                f"- Active loss terms: {_stage45c_standalone_report.get('stage45c_loss_terms_active')}",
                f"- SUPPORT recovery loss mean: {_stage45c_standalone_report.get('stage45c_support_recovery_loss_mean')}",
                f"- Entitled NE penalty loss mean: {_stage45c_standalone_report.get('stage45c_entitled_ne_penalty_loss_mean')}",
                "",
                "## Leakage Policy",
                "",
                str(_stage45c_standalone_report.get("stage45c_leakage_policy")),
                "",
                "## Recommendation",
                "",
                str(_stage45c_standalone_report.get("stage45c_recommendation")),
                "",
            ]
            write_text(args.stage45c_report_md, "\n".join(_stage45c_md_lines))

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json is not None:
        v5.write_report_json(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
