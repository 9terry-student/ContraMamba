"""Train ContraMamba-v6B-minimal on controlled intervention data.

Minimal v6B wrapper: reuses v5 training infrastructure, adds temporal/predicate
comparator alphas with learnable scaling. No composer, no product_final_loss.
All CE/pairwise/intervention losses consume final calibrated logits.
"""

from __future__ import annotations

import argparse
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
from scripts import train_controlled_v5 as v5  # noqa: E402

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
        time_swap  → label=0 (temporal mismatch; unsafe)
        none, paraphrase → label=1 (temporally safe)
        all others → masked (excluded from loss)

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
        safe_label 0 → mismatch target = 1
        safe_label 1 → mismatch target = 0
    Fallback derivation from intervention_type:
        time_swap          → label=1 (temporal mismatch positive)
        none, paraphrase   → label=0 (temporal safe/control)
        all others         → masked (excluded from loss)

    Returns (labels, mask) both of shape [B].
    Stage15 OOD is never passed here.
    """
    labels: list[int] = []
    mask: list[int] = []
    for r in records:
        explicit = r.get("stage30_temporal_safe_label")
        if explicit is not None:
            # Invert: safe_label=0 means mismatch → target 1; safe_label=1 means safe → target 0
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


def _stage33_rule_strength(reason: str) -> str:
    if str(reason).startswith("weak_rule_forced_to_residual:"):
        reason = str(reason).split(":", 1)[1]
    return _STAGE33_RULE_STRENGTHS.get(str(reason), "unknown")


def build_stage33_structured_coverage_owner_state(
    record: dict[str, Any],
    *,
    enabled: bool,
    weak_rules_to_residual: set[str] | None = None,
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
    return {
        "enabled": bool(enabled),
        "label": label,
        "route": route,
        "reason": reason,
        "original_reason": original_reason,
        "rule_strength": rule_strength,
        "confidence": confidence,
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
    if route == "ENTAILMENT_PRESERVE":
        if direct_support_allowed:
            direct_block_reason = "allowed"
        elif not structured_preserve_can_support:
            direct_block_reason = "preserve_can_support_disabled"
        elif hard_core.get("pass") is not True:
            direct_block_reason = "hard_core_not_true"
        elif structured_strength != "high_precision":
            direct_block_reason = f"rule_strength:{structured_strength}"
        else:
            direct_block_reason = f"rule_not_allowed:{structured_reason}"
        structured_coverage["direct_support_allowed"] = direct_support_allowed
        structured_coverage["direct_support_block_reason"] = direct_block_reason

    if conditional_enabled:
        priority_trace.extend([
            "conditional_fallback:on",
            f"route:{route}",
            f"strength:{structured_strength}",
        ])
        if hard_core.get("pass") is False:
            shadow_label = current_final_label
            shadow_reason = "stage33_conditional_fallback_hard_core_block"
            conditional_action = "fallback_current_final"
            conditional_fallback_used = True
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
    scalars = {
        key: output[key].detach().cpu()
        for key in scalar_keys
        if key in output and output[key] is not None
    }

    # Stage28-E: v7 per-example diagnostic scalars (absent on v6b_minimal runs)
    v7_scalars = {
        key: output[key].detach().cpu()
        for key in _S28E_V7_SCALAR_KEYS
        if key in output and output[key] is not None
    }

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
            # ── Stable identity ───────────────────────────────────────────────
            "stable_id": stable_id,
            "source_id": source_id,
            "pair_id": pair_id,
            "example_index": index,
            # ── Text ─────────────────────────────────────────────────────────
            "claim": record.get("claim"),
            "evidence": record.get("evidence"),
            # ── Intervention ─────────────────────────────────────────────────
            "intervention": record.get("intervention_type"),
            "normalized_intervention": norm_intervention,
            "diagnostic_axis": diagnostic_axis,
            # ── Gold labels ───────────────────────────────────────────────────
            "gold_label_raw": gold_raw,
            "gold_label": gold_label,
            "gold_label_id": gold_label_id,
            # ── Predicted labels ──────────────────────────────────────────────
            "pred_label_id": pred_id,
            "pred_label": pred_label,
            "pred_label_raw": pred_label,
            "is_correct": is_correct,
            "is_false_support": is_false_support,
            "is_location_false_support": is_location_false_support,
            "is_role_false_support": is_role_false_support,
            # ── Final logits / probs (order: REFUTE=0, NOT_ENTITLED=1, SUPPORT=2) ──
            "final_logits": final_logits,
            "final_probs": final_probs_list,
            "refute_logit": refute_logit,
            "ne_logit": ne_logit,
            "support_logit": support_logit,
            "refute_prob": refute_prob,
            "ne_prob": ne_prob,
            "support_prob": support_prob,
            # ── Existing v6b diagnostic scalars ──────────────────────────────
            **{key: float(scalars[key][index]) for key in scalar_keys if key in scalars},
            # ── Gold auxiliary labels (when present in source record) ─────────
            **{key: record[key] for key in _S28E_AUX_LABEL_KEYS if key in record},
            # ── V7/H1 diagnostic scalars (absent on v6b_minimal runs) ─────────
            **{key: float(v7_scalars[key][index]) for key in _S28E_V7_SCALAR_KEYS
               if key in v7_scalars},
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
            # ── Legacy backward-compat fields ─────────────────────────────────
            **(
                flatten_stage32_owner_state(stage32_owner_state)
                if stage32_owner_state is not None
                else {}
            ),
            "id": record.get("id"),
            "intervention_type": record.get("intervention_type"),
            "gold_final_label": gold_raw,
            "pred_final_label": pred_label,
            # ── Shallow raw record snapshot ───────────────────────────────────
            "raw_record": {k: record[k] for k in _S28E_RAW_RECORD_KEYS if k in record},
        }
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

    # ── Stage28-I-A: independent location-boundary cap/head ──────────────────────────────────
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

    # ── Stage30-C2: independent temporal-safety cap/head ──────────────────────────────────────
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

    # ── Stage30-D: representation-decomposed temporal mismatch multihead ──────────────────────
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
            "Reads stage30_temporal_safe_label (0→mismatch=1, 1→safe=0) or "
            "falls back to intervention_type (time_swap→1, none/paraphrase→0). "
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

    # ── Stage30-E: temporal residual preservation-aware cap ──────────────────────────────────
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
            "none/paraphrase → preserved=1; time_swap → preserved=0. "
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

_S29B_PROBE_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "frame_compatible_label": 0,
    "predicate_covered_label": 0,
    "sufficiency_label": 0,
    "polarity_label": "NONE",
    "primary_failure_type": "none",
    "source_intervention_type": "",
}


def load_external_probe_jsonl(path: "Path") -> list[dict]:
    """Load external probe JSONL records tolerantly (no schema enforcement)."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _patch_probe_record(record: dict) -> dict:
    """Return a copy of record with conservative defaults for missing optional fields.

    Required for v5.encode_records / v5.encode_mamba_records which expect the full
    controlled-data schema.  Defaults are used only for tensor construction; they do
    not affect prediction or metric computation.
    """
    out = dict(record)
    for field, default in _S29B_PROBE_OPTIONAL_DEFAULTS.items():
        out.setdefault(field, default)
    return out


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
            **v5.model_feature_inputs(probe_inputs),
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ---------------------------------------------------------------------------
    # Fail-fast: dummy backbone guard
    # Dummy backbone is valid ONLY for explicit smoke/plumbing diagnostics.
    # It has no text comprehension capacity; metrics it produces are not
    # claim-worthy. Require --allow-dummy-backbone to proceed.
    # ---------------------------------------------------------------------------
    # ── Stage28-I-A: location boundary flag validation ────────────────────────────────────────
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

    # ── Stage30-C2: temporal safety flag validation ───────────────────────────────────────────
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

    # ── Stage30-D: temporal mismatch multihead flag validation ────────────────────────────────
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

    # Stage30-C2: temporal safety data loading — separate dataset, never mixed into main train/dev.
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

    # Stage30-D: temporal mismatch multihead aux data loading — separate from main train/dev.
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

    # Stage30-E: temporal preservation aux data loading — separate from main train/dev.
    # Reuses encode_temporal_safety_labels: none/paraphrase → label=1, time_swap → label=0.
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
            # encode_temporal_safety_labels: safe=1 (none/paraphrase) = preserved=1 ✓
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

            # ── Stage28-I-A: v7 location boundary BCE auxiliary loss ──────────────────────────
            # Target: 0 for location_swap; 1 for none/paraphrase/polarity_flip.
            # All other intervention types are excluded from this loss.
            # Stage15/OOD is not used for this loss or target selection.
            # Zero masked records → loss is zero; no crash.
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

            # ── Stage30-C2: v7 temporal safety BCE auxiliary loss ─────────────────────────────
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
                    **v5.model_feature_inputs(ts_train_inputs),
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

            # ── Stage30-D: v7 temporal mismatch multihead BCE auxiliary loss ──────────────────
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
                    **v5.model_feature_inputs(tmm_train_inputs),
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
                        # Mean for scale stability (3 heads → 3× scale without mean)
                        _v7_tmm_loss = sum(_tmm_per_head_losses) / len(_tmm_per_head_losses)
                        total_loss = total_loss + tmm_loss_weight * _v7_tmm_loss

            # ── Stage30-E: v7 temporal preservation head BCE auxiliary loss ────────────────────
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
                    **v5.model_feature_inputs(tpres_train_inputs),
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
                    **v5.model_feature_inputs(covent_train_inputs),
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
                        **v5.model_feature_inputs(covent_train_inputs),
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
                        **v5.model_feature_inputs(covent_dev_inputs),
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
            "best_stage31c_coverage_entailment_metrics": best_covent_metrics,
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
            "stage33_structured_coverage_modifies_final_logits": False,
            "stage33_structured_coverage_modifies_final_predictions": False,
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
                "stage33_structured_coverage_modifies_final_logits": False,
                "stage33_structured_coverage_modifies_final_predictions": False,
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

            # Load and patch records (fills in missing optional fields with safe defaults)
            _ext_records_raw = load_external_probe_jsonl(_ext_path)
            _ext_records = [_patch_probe_record(r) for r in _ext_records_raw]

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

            # Eval-only forward pass — no loss, no gradient
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

    # Stage26-D: lift architecture metadata and dev metric aliases to root level.
    lift_report_aliases(report)

    print("\nFINAL_REPORT")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json is not None:
        v5.write_report_json(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
