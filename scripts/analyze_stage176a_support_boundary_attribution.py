"""Stage176-A: clean-dev-only SUPPORT-boundary attribution audit.

This module is deliberately evaluation-only.  It validates two completed
Stage175 provenance/checkpoint pairs before constructing either model, reuses
the controlled pair split and Stage175 eligibility rules, and never creates an
optimizer or enters training mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import train_controlled_v5 as v5  # noqa: E402
from scripts import train_controlled_v6b_minimal as v6b  # noqa: E402
from scripts.stage175b_support_anchor import (  # noqa: E402
    build_train_support_anchor_index,
    support_margin,
)


STAGE = "Stage176-A"
COMPLETE = "STAGE176A_CLEAN_DEV_SUPPORT_BOUNDARY_ATTRIBUTION_COMPLETE"
BLOCKED = "STAGE176A_CLEAN_DEV_SUPPORT_BOUNDARY_ATTRIBUTION_BLOCKED"
LABELS = ("NOT_ENTITLED", "REFUTE", "SUPPORT")
CATEGORIES = (
    "eligible_support_paraphrase",
    "eligible_support_canonical_none",
    "other_gold_support",
    "gold_refute",
    "gold_not_entitled",
)
EXPECTED = {
    "total_pair_groups": 300,
    "train_pair_groups": 240,
    "dev_pair_groups": 60,
    "dev_rows": 720,
    "eligible_dev_support_anchor_pairs": 29,
}
OUTPUT_NAMES = {
    "json": "stage176a_support_boundary_attribution_report.json",
    "md": "stage176a_support_boundary_attribution_report.md",
    "rows": "stage176a_dev_row_transitions.csv",
    "categories": "stage176a_category_summary.csv",
    "interventions": "stage176a_intervention_summary.csv",
    "pairs": "stage176a_eligible_pair_drift.csv",
    "matrix": "stage176a_transition_matrix.csv",
}


class ValidationBlocked(ValueError):
    """Raised only for pre-forward provenance/checkpoint/data validation."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValidationBlocked(f"JSON root must be an object: {path}")
    return value


def _get(mapping: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _first(mapping: dict[str, Any], paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        value = _get(mapping, path)
        if value is not None:
            return value
    return default


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationBlocked(message)


def _same(left: Any, right: Any, name: str) -> None:
    _require(left is not None and right is not None, f"missing comparable {name}")
    _require(left == right, f"{name} mismatch: baseline={left!r}, treatment={right!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime(prov: dict[str, Any], key: str) -> Any:
    return _first(prov, (f"resolved_runtime_config.{key}", f"parsed_args.{key}"))


def _stage_config(prov: dict[str, Any], stage: str, key: str) -> Any:
    return _first(
        prov,
        (
            f"{stage}.{key}",
            f"configuration.{stage}.{key}",
            f"parsed_args.{stage.replace('_', '-')}-{key.replace('_', '-')}",
        ),
    )


def _arg(prov: dict[str, Any], name: str, default: Any = None) -> Any:
    return _first(prov, (f"parsed_args.{name}", f"resolved_runtime_config.{name}"), default)


def _data_record(prov: dict[str, Any]) -> dict[str, Any]:
    value = _get(prov, "data_provenance.main_data")
    _require(isinstance(value, dict), "provenance lacks data_provenance.main_data")
    return value


def _selection(prov: dict[str, Any], key: str) -> Any:
    return _first(prov, (f"training_selection_policy.{key}", f"finalization.{key}"))


def _resolve_checkpoint(
    provenance_path: Path, prov: dict[str, Any], explicit: Path | None
) -> Path:
    raw = explicit or _first(
        prov,
        (
            "finalization.selected_checkpoint_path",
            "finalization.selected_checkpoint.path",
            "selected_checkpoint_path",
            "selected_checkpoint.path",
        ),
    )
    _require(raw is not None, f"no saved selected checkpoint in {provenance_path}")
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((provenance_path.parent / path, ROOT / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValidationBlocked(f"selected checkpoint does not exist: {raw}")


def _load_checkpoint_header(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
        metadata = dict(payload.get("metadata") or {})
        for key in ("training_args", "architecture", "backbone", "label_mapping"):
            if key in payload and key not in metadata:
                metadata[key] = payload[key]
    elif isinstance(payload, dict) and all(isinstance(key, str) for key in payload):
        state = payload
    else:
        raise ValidationBlocked(f"unsupported checkpoint schema: {path}")
    _require(isinstance(state, dict) and bool(state), f"empty checkpoint state: {path}")
    _require(
        all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()),
        f"checkpoint state is not a tensor state_dict: {path}",
    )
    return state, metadata


def _selected_epoch(prov: dict[str, Any]) -> int:
    value = _get(prov, "finalization.selected_epoch")
    _require(isinstance(value, int), "provenance selected_epoch must be a scalar integer")
    return value


def _validate_checkpoint_metadata(
    path: Path, metadata: dict[str, Any], prov: dict[str, Any]
) -> None:
    _require(bool(metadata), f"checkpoint metadata required for hard validation: {path}")
    expected_epoch = _selected_epoch(prov)
    checkpoint_epoch = _first(metadata, ("selected_epoch", "best_epoch"))
    _require(
        checkpoint_epoch == expected_epoch,
        f"checkpoint selected epoch mismatch: provenance={expected_epoch}, checkpoint={checkpoint_epoch}",
    )
    for key in ("architecture", "backbone", "model_name", "seed"):
        saved = _first(metadata, (key, f"training_args.{key}"))
        expected = _runtime(prov, key)
        _require(saved == expected, f"checkpoint {key} mismatch: {saved!r} != {expected!r}")
    mapping = metadata.get("label_mapping")
    if mapping is not None:
        normalized = {int(key): value for key, value in mapping.items()}
        _require(normalized == v5.ID_TO_FINAL_LABEL, "checkpoint label mapping mismatch")


def _validate_provenances(
    baseline: dict[str, Any], treatment: dict[str, Any], data_path: Path
) -> dict[str, Any]:
    for role, prov in (("baseline", baseline), ("treatment", treatment)):
        _require(prov.get("status") == "completed", f"{role} status is not completed")
        _require(_runtime(prov, "architecture") == "v6b_minimal", f"{role} architecture must be v6b_minimal")
        _require(_runtime(prov, "backbone") == "mamba", f"{role} backbone must be mamba")
        _require(_runtime(prov, "model_name") == "state-spaces/mamba-130m-hf", f"{role} model name mismatch")
        _require(_selected_epoch(prov) == 20, f"{role} selected epoch must be 20")
        _require(_runtime(prov, "seed") == 174, f"{role} seed must be 174")
        _require(_arg(prov, "dev_ratio", 0.2) == 0.2, f"{role} dev_ratio must be 0.2")
        stage174 = prov.get("stage174c_clean_pairwise") or {}
        _require(stage174.get("mode") == "off", f"{role} Stage174-C mode must be off")
        _require(float(stage174.get("weight", 0.0)) == 0.0, f"{role} Stage174-C weight must be zero")
        policy = prov.get("training_selection_policy") or {}
        _require(policy.get("clean_dev_only_checkpoint_selection") is True, f"{role} selection was not clean-dev-only")
        _require(policy.get("external_evaluation_used_for_training") is False, f"{role} external data affected training")
        _require(policy.get("external_evaluation_used_for_calibration") is False, f"{role} external calibration was used")
        _require(policy.get("external_evaluation_used_for_threshold_selection") is False, f"{role} external threshold selection was used")
        _require(policy.get("external_evaluation_used_for_checkpoint_selection") is False, f"{role} external checkpoint selection was used")
        _require(policy.get("time_swap_included_in_main_classification_training") is False, f"{role} time_swap entered main training")
        _require(policy.get("final_ce_logits_source") in ('output["logits"]', "output['logits']"), f"{role} final CE source mismatch")
        _require(policy.get("loss_logits_used_for_final_classifier_ce") is False, f"{role} loss_logits was used")
        activity = _get(prov, "data_provenance.auxiliary_activity") or {}
        _require(activity.get("time_swap_active") is False, f"{role} time_swap auxiliary activity detected")
        _require(activity.get("external_evaluation_active") is False, f"{role} external evaluation activity detected")

    for key in ("seed", "architecture", "backbone", "model_name", "dev_ratio"):
        _same(_arg(baseline, key, _runtime(baseline, key)), _arg(treatment, key, _runtime(treatment, key)), key)
    base_data, treat_data = _data_record(baseline), _data_record(treatment)
    for key in ("path", "sha256", "row_count"):
        _same(base_data.get(key), treat_data.get(key), f"clean data {key}")
    actual_hash = _sha256(data_path)
    _require(base_data.get("sha256") == actual_hash, "--data SHA-256 does not match provenance")

    base175 = baseline.get("stage175b_support_anchor") or {}
    treat175 = treatment.get("stage175b_support_anchor") or {}
    _require(
        base175.get("mode") == "off" or float(base175.get("weight", 0.0)) == 0.0,
        "baseline Stage175-B must be off or zero-weight",
    )
    _require(treat175.get("mode") == "paraphrase_margin", "treatment Stage175-B mode mismatch")
    _require(math.isclose(float(treat175.get("weight", -1)), 0.05), "treatment Stage175-B weight mismatch")
    _require(math.isclose(float(treat175.get("tolerance", -1)), 0.10), "treatment Stage175-B tolerance mismatch")
    return {
        "status": "passed",
        "data_sha256": actual_hash,
        "same_seed_architecture_backbone_model_data_and_split": True,
        "clean_dev_only": True,
        "external_data_or_labels_used": False,
        "time_swap_used": False,
        "final_ce_source": 'output["logits"]',
        "loss_logits_used": False,
    }


def _record_training_args(prov: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    values = dict(prov.get("parsed_args") or {})
    values.update(metadata.get("training_args") or {})
    return values


def _construct_model(
    prov: dict[str, Any], metadata: dict[str, Any], state: dict[str, torch.Tensor], device: torch.device
) -> torch.nn.Module:
    config = _record_training_args(prov, metadata)
    keys = tuple(state)
    model = v6b.build_mamba_model(
        str(_runtime(prov, "model_name")),
        freeze_encoder=bool(config.get("freeze_encoder", True)),
        freeze_a_log=bool(config.get("freeze_a_log", True)),
        use_boundary_head=any(key.startswith("boundary_head.") for key in keys),
        use_frame_violation_head=any(key.startswith("frame_violation_head.") for key in keys),
        use_predicate_isolation_head=any(key.startswith("predicate_isolation_head.") for key in keys),
        use_preservation_entitlement_head=any(key.startswith("preservation_entitlement_head.") for key in keys),
        use_temporal_diagnostic_head=any(key.startswith("temporal_diagnostic_head.") for key in keys),
        use_temporal_residual_adapter=any(key.startswith("temporal_residual_adapter.") for key in keys),
        temporal_adapter_detach_input=bool(config.get("temporal_adapter_detach_input", True)),
        use_temporal_channel=any(key.startswith("temporal_channel_v1.") for key in keys),
        temporal_channel_detach_input=bool(config.get("temporal_channel_detach_input", True)),
        use_temporal_channel_loss=bool(config.get("use_temporal_channel_loss", False)),
        temporal_channel_loss_weight=float(config.get("temporal_channel_loss_weight", 0.0)),
        temporal_channel_loss_pos_weight=float(config.get("temporal_channel_loss_pos_weight", 1.0)),
        use_temporal_channel_gated_penalty=bool(config.get("use_temporal_channel_gated_penalty", False)),
        temporal_channel_gated_penalty_scale=float(config.get("temporal_channel_gated_penalty_scale", 0.0)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _forward(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    prov: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    config = prov.get("parsed_args") or {}
    temporal, predicate = v6b.extract_flags(
        records, str(config.get("flag_source", "controlled_heuristic")), device
    )
    with torch.no_grad():
        output = v6b._vnext_forward_maybe_batched(
            model,
            inputs,
            temporal_mismatch_flags=temporal,
            predicate_mismatch_flags=predicate,
            temporal_adapter_final_penalty_scale=float(config.get("temporal_adapter_final_penalty_scale", 0.0)),
            temporal_channel_gated_penalty_scale=float(config.get("temporal_channel_gated_penalty_scale", 0.0)),
            batch_size=batch_size,
            amp_enabled=False,
        )
    _require("logits" in output and "predictions" in output, "model output lacks logits/predictions")
    _require(torch.equal(output["predictions"], output["logits"].argmax(dim=-1)), "clean evaluator prediction semantics are not logits argmax")
    return output


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _trade(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["correctness_transition"] for row in rows)
    recovered = counts["incorrect_to_correct"]
    regressed = counts["correct_to_incorrect"]
    return {
        "recovered_errors": recovered,
        "regressed_errors": regressed,
        "unchanged_correct": counts["correct_to_correct"],
        "unchanged_incorrect": counts["incorrect_to_incorrect"],
        "net_correctness_change": recovered - regressed,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    base_correct = sum(row["baseline_predicted_label"] == row["gold_final_label"] for row in rows)
    treat_correct = sum(row["treatment_predicted_label"] == row["gold_final_label"] for row in rows)
    support_gold = [row for row in rows if row["gold_final_label"] == "SUPPORT"]
    deltas = [float(row["support_margin_delta"]) for row in rows]
    transitions = Counter(row["prediction_transition"] for row in rows)
    return {
        "row_count": count,
        "gold_label_count": dict(sorted(Counter(row["gold_final_label"] for row in rows).items())),
        "baseline_prediction_distribution": dict(sorted(Counter(row["baseline_predicted_label"] for row in rows).items())),
        "treatment_prediction_distribution": dict(sorted(Counter(row["treatment_predicted_label"] for row in rows).items())),
        "prediction_transition_counts": dict(sorted(transitions.items())),
        "baseline_accuracy": base_correct / count if count else None,
        "treatment_accuracy": treat_correct / count if count else None,
        "accuracy_delta": (treat_correct - base_correct) / count if count else None,
        "baseline_support_recall": (sum(row["baseline_predicted_label"] == "SUPPORT" for row in support_gold) / len(support_gold)) if support_gold else None,
        "treatment_support_recall": (sum(row["treatment_predicted_label"] == "SUPPORT" for row in support_gold) / len(support_gold)) if support_gold else None,
        "support_to_not_entitled": transitions["SUPPORT->NOT_ENTITLED"],
        "not_entitled_to_support": transitions["NOT_ENTITLED->SUPPORT"],
        "mean_baseline_support_margin": _mean([float(row["baseline_support_margin"]) for row in rows]),
        "median_baseline_support_margin": _median([float(row["baseline_support_margin"]) for row in rows]),
        "mean_treatment_support_margin": _mean([float(row["treatment_support_margin"]) for row in rows]),
        "median_treatment_support_margin": _median([float(row["treatment_support_margin"]) for row in rows]),
        "mean_support_margin_delta": _mean(deltas),
        "median_support_margin_delta": _median(deltas),
        "negative_margin_delta_row_count": sum(value < 0 for value in deltas),
        "positive_margin_delta_row_count": sum(value > 0 for value in deltas),
        "zero_margin_delta_row_count": sum(value == 0 for value in deltas),
        "correctness_tradeoff": _trade(rows),
    }


def _build_rows(
    dev_records: list[dict[str, Any]],
    baseline_output: dict[str, Any],
    treatment_output: dict[str, Any],
    eligible: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    label_to_id = v5.FINAL_LABEL_TO_ID
    _require(set(LABELS) == set(label_to_id), "unexpected FINAL_LABEL_TO_ID keys")
    base_logits = baseline_output["logits"].detach().float().cpu()
    treat_logits = treatment_output["logits"].detach().float().cpu()
    base_predictions = baseline_output["predictions"].detach().cpu()
    treat_predictions = treatment_output["predictions"].detach().cpu()
    base_margins = support_margin(base_logits, label_to_id)
    treat_margins = support_margin(treat_logits, label_to_id)
    eligible_pairs = set(eligible)
    result: list[dict[str, Any]] = []
    for index, record in enumerate(dev_records):
        gold = record["final_label"]
        intervention = record["intervention_type"]
        pair_id = str(record["pair_id"])
        if gold == "SUPPORT" and pair_id in eligible_pairs and intervention == "paraphrase":
            category = "eligible_support_paraphrase"
        elif gold == "SUPPORT" and pair_id in eligible_pairs and intervention == "none":
            category = "eligible_support_canonical_none"
        elif gold == "SUPPORT":
            category = "other_gold_support"
        elif gold == "REFUTE":
            category = "gold_refute"
        elif gold == "NOT_ENTITLED":
            category = "gold_not_entitled"
        else:
            raise RuntimeError(f"unassigned gold label at dev row {index}: {gold}")
        base_pred = v5.ID_TO_FINAL_LABEL[int(base_predictions[index])]
        treat_pred = v5.ID_TO_FINAL_LABEL[int(treat_predictions[index])]
        before_correct, after_correct = base_pred == gold, treat_pred == gold
        correctness = (
            ("correct" if before_correct else "incorrect")
            + "_to_"
            + ("correct" if after_correct else "incorrect")
        )
        item: dict[str, Any] = {
            "stable_row_index": index,
            "row_id": record.get("id", f"dev_row_{index}"),
            "pair_id": pair_id,
            "intervention_type": intervention,
            "category": category,
            "gold_final_label": gold,
            "gold_frame_label": record.get("frame_compatible_label"),
            "gold_predicate_label": record.get("predicate_covered_label"),
            "gold_sufficiency_label": record.get("sufficiency_label"),
            "polarity_label": record.get("polarity_label"),
            "baseline_predicted_label": base_pred,
            "treatment_predicted_label": treat_pred,
            "baseline_support_margin": float(base_margins[index]),
            "treatment_support_margin": float(treat_margins[index]),
            "support_margin_delta": float(treat_margins[index] - base_margins[index]),
            "baseline_max_logit": float(base_logits[index].max()),
            "treatment_max_logit": float(treat_logits[index].max()),
            "prediction_transition": f"{base_pred}->{treat_pred}",
            "correctness_transition": correctness,
        }
        for label in LABELS:
            key = label.lower()
            label_id = label_to_id[label]
            item[f"baseline_logit_{key}"] = float(base_logits[index, label_id])
            item[f"treatment_logit_{key}"] = float(treat_logits[index, label_id])
        result.append(item)
    _require(len(result) == len(dev_records), "row analysis lost dev rows")
    _require(all(row["category"] in CATEGORIES for row in result), "category overlap/unassigned row")
    return result


def _matrix(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix_rows: list[dict[str, Any]] = []
    nested: dict[str, dict[str, int]] = {}
    for before in LABELS:
        nested[before] = {}
        for after in LABELS:
            count = sum(
                row["baseline_predicted_label"] == before
                and row["treatment_predicted_label"] == after
                for row in rows
            )
            nested[before][after] = count
            matrix_rows.append({"baseline_prediction": before, "treatment_prediction": after, "row_count": count})
    base_support = sum(row["baseline_predicted_label"] == "SUPPORT" for row in rows)
    treat_support = sum(row["treatment_predicted_label"] == "SUPPORT" for row in rows)
    return matrix_rows, {
        "matrix": nested,
        "support_to_not_entitled": nested["SUPPORT"]["NOT_ENTITLED"],
        "support_to_refute": nested["SUPPORT"]["REFUTE"],
        "not_entitled_to_support": nested["NOT_ENTITLED"]["SUPPORT"],
        "refute_to_support": nested["REFUTE"]["SUPPORT"],
        "unchanged_support": nested["SUPPORT"]["SUPPORT"],
        "unchanged_not_entitled": nested["NOT_ENTITLED"]["NOT_ENTITLED"],
        "unchanged_refute": nested["REFUTE"]["REFUTE"],
        "baseline_support_predictions": base_support,
        "treatment_support_predictions": treat_support,
        "support_prediction_delta": treat_support - base_support,
        "net_support_decrease_is_39": base_support - treat_support == 39,
    }


def _intervention_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["intervention_type"]].append(row)
    _require(len(grouped) == 12, f"expected 12 intervention types, got {len(grouped)}")
    result: dict[str, dict[str, Any]] = {}
    for name, group in sorted(grouped.items()):
        summary = _summary(group)
        base = summary["baseline_prediction_distribution"].get("SUPPORT", 0)
        treat = summary["treatment_prediction_distribution"].get("SUPPORT", 0)
        base_ne = summary["baseline_prediction_distribution"].get("NOT_ENTITLED", 0)
        treat_ne = summary["treatment_prediction_distribution"].get("NOT_ENTITLED", 0)
        summary["support_prediction_delta"] = treat - base
        summary["not_entitled_prediction_delta"] = treat_ne - base_ne
        result[name] = summary
    return result


def _eligible_pair_drift(
    rows: list[dict[str, Any]], eligible: dict[str, dict[str, int]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["pair_id"] in eligible and row["intervention_type"] in ("none", "paraphrase"):
            by_pair[row["pair_id"]][row["intervention_type"]] = row
    _require(len(by_pair) == 29, f"expected 29 eligible pair drift groups, got {len(by_pair)}")
    output: list[dict[str, Any]] = []
    for pair_id, variants in sorted(by_pair.items()):
        _require(set(variants) == {"none", "paraphrase"}, f"eligible pair {pair_id} lacks canonical/paraphrase")
        canonical, paraphrase = variants["none"], variants["paraphrase"]
        bgap = paraphrase["baseline_support_margin"] - canonical["baseline_support_margin"]
        tgap = paraphrase["treatment_support_margin"] - canonical["treatment_support_margin"]
        output.append({
            "pair_id": pair_id,
            "baseline_canonical_margin": canonical["baseline_support_margin"],
            "treatment_canonical_margin": canonical["treatment_support_margin"],
            "canonical_margin_delta": canonical["support_margin_delta"],
            "baseline_paraphrase_margin": paraphrase["baseline_support_margin"],
            "treatment_paraphrase_margin": paraphrase["treatment_support_margin"],
            "paraphrase_margin_delta": paraphrase["support_margin_delta"],
            "baseline_paraphrase_minus_canonical_gap": bgap,
            "treatment_paraphrase_minus_canonical_gap": tgap,
            "gap_delta": tgap - bgap,
            "baseline_canonical_prediction": canonical["baseline_predicted_label"],
            "treatment_canonical_prediction": canonical["treatment_predicted_label"],
            "baseline_paraphrase_prediction": paraphrase["baseline_predicted_label"],
            "treatment_paraphrase_prediction": paraphrase["treatment_predicted_label"],
            "canonical_transition": canonical["prediction_transition"],
            "paraphrase_transition": paraphrase["prediction_transition"],
        })
    canonical_retained = lambda row: row["baseline_canonical_prediction"] == "SUPPORT" and row["treatment_canonical_prediction"] == "SUPPORT"
    paraphrase_retained = lambda row: row["baseline_paraphrase_prediction"] == "SUPPORT" and row["treatment_paraphrase_prediction"] == "SUPPORT"
    canonical_lost = lambda row: row["canonical_transition"] == "SUPPORT->NOT_ENTITLED"
    paraphrase_lost = lambda row: row["paraphrase_transition"] == "SUPPORT->NOT_ENTITLED"
    aggregate = {
        "pair_count": len(output),
        "canonical_mean_margin_drift": _mean([row["canonical_margin_delta"] for row in output]),
        "paraphrase_mean_margin_drift": _mean([row["paraphrase_margin_delta"] for row in output]),
        "mean_gap_delta": _mean([row["gap_delta"] for row in output]),
        "canonical_support_retention": sum(canonical_retained(row) for row in output),
        "paraphrase_support_retention": sum(paraphrase_retained(row) for row in output),
        "canonical_support_to_not_entitled": sum(canonical_lost(row) for row in output),
        "paraphrase_support_to_not_entitled": sum(paraphrase_lost(row) for row in output),
        "both_rows_lost_support": sum(canonical_lost(row) and paraphrase_lost(row) for row in output),
        "paraphrase_preserved_while_canonical_lost": sum(canonical_lost(row) and paraphrase_retained(row) for row in output),
        "canonical_preserved_while_paraphrase_lost": sum(canonical_retained(row) and paraphrase_lost(row) for row in output),
        "both_preserved": sum(canonical_retained(row) and paraphrase_retained(row) for row in output),
    }
    return output, aggregate


def _diagnosis(
    categories: dict[str, dict[str, Any]], interventions: dict[str, dict[str, Any]], pair_aggregate: dict[str, Any]
) -> dict[str, Any]:
    para = categories["eligible_support_paraphrase"]
    canonical = categories["eligible_support_canonical_none"]
    other = categories["other_gold_support"]
    negative_categories = [name for name, value in categories.items() if (value["mean_support_margin_delta"] or 0) < 0]
    ne_up_families = [name for name, value in interventions.items() if value["not_entitled_prediction_delta"] > 0]
    negative_families = [name for name, value in interventions.items() if (value["mean_support_margin_delta"] or 0) < 0]
    limitation = "Observational comparison of two selected clean-dev checkpoints; no causal threshold or calibrated boundary is introduced."
    return {
        "A_eligible_paraphrase_local_failure": {
            "supported": para["support_to_not_entitled"] > 0 or para["mean_support_margin_delta"] < 0,
            "supporting_counts": {"support_to_not_entitled": para["support_to_not_entitled"], "negative_margin_delta_rows": para["negative_margin_delta_row_count"]},
            "supporting_margin_statistics": {"mean_delta": para["mean_support_margin_delta"], "median_delta": para["median_support_margin_delta"]},
            "limitations": limitation,
        },
        "B_canonical_reference_drift": {
            "supported": canonical["support_to_not_entitled"] > 0 or canonical["mean_support_margin_delta"] < 0,
            "supporting_counts": {"support_to_not_entitled": canonical["support_to_not_entitled"], "pair_level_support_to_not_entitled": pair_aggregate["canonical_support_to_not_entitled"]},
            "supporting_margin_statistics": {"mean_delta": canonical["mean_support_margin_delta"], "pair_mean_drift": pair_aggregate["canonical_mean_margin_drift"]},
            "limitations": limitation,
        },
        "C_untargeted_support_degradation": {
            "supported": other["support_to_not_entitled"] > 0 or other["mean_support_margin_delta"] < 0,
            "supporting_counts": {"support_to_not_entitled": other["support_to_not_entitled"], "negative_margin_delta_rows": other["negative_margin_delta_row_count"]},
            "supporting_margin_statistics": {"mean_delta": other["mean_support_margin_delta"], "median_delta": other["median_support_margin_delta"]},
            "limitations": limitation,
        },
        "D_global_conservative_boundary_shift": {
            "supported": len(negative_categories) > 1 and len(negative_families) > 1 and len(ne_up_families) > 1,
            "supporting_counts": {"negative_margin_categories": negative_categories, "negative_margin_intervention_families": negative_families, "not_entitled_increase_families": ne_up_families},
            "supporting_margin_statistics": {"category_mean_deltas": {name: categories[name]["mean_support_margin_delta"] for name in negative_categories}},
            "limitations": limitation,
        },
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _named_rows(values: dict[str, dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    return [{key_name: name, **value} for name, value in values.items()]


def _render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall_transition"]
    categories = report["category_attribution"]
    pair = report["eligible_pair_drift"]["aggregate"]
    trade = report["correctness_tradeoff"]["overall"]
    lines = [
        "# Stage176-A clean-dev SUPPORT-boundary attribution",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        "## Scope and validation",
        "",
        "Diagnostic-only comparison of the two selected epoch-20 checkpoints on the deterministic seed-174 clean-dev split. No training, threshold tuning, calibration, checkpoint selection, external evaluation, external labels, or time-swap data is used.",
        "",
        f"Split: {report['split']['total_pair_groups']} total pairs, {report['split']['train_pair_groups']} train pairs, {report['split']['dev_pair_groups']} dev pairs, {report['split']['dev_rows']} dev rows, and {report['split']['eligible_dev_support_anchor_pairs']} eligible dev anchor pairs; overlap is {report['split']['train_dev_overlap']}.",
        "",
        "## Overall transition",
        "",
        f"Baseline/treatment SUPPORT predictions: {overall['baseline_support_predictions']}/{overall['treatment_support_predictions']} (delta {overall['support_prediction_delta']}). SUPPORT→NOT_ENTITLED: {overall['support_to_not_entitled']}; SUPPORT→REFUTE: {overall['support_to_refute']}; NOT_ENTITLED→SUPPORT: {overall['not_entitled_to_support']}; REFUTE→SUPPORT: {overall['refute_to_support']}.",
        "",
        "## Category attribution",
        "",
        "| Category | Rows | S→NE | Mean margin Δ | Accuracy Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CATEGORIES:
        value = categories[name]
        lines.append(f"| {name} | {value['row_count']} | {value['support_to_not_entitled']} | {value['mean_support_margin_delta']:.6f} | {value['accuracy_delta']:+.6f} |")
    lines.extend([
        "",
        "False-positive removal is split explicitly in the JSON report between gold NOT_ENTITLED and gold REFUTE rows; true SUPPORT losses are reported separately.",
        "",
        "## Eligible pair drift",
        "",
        f"Canonical/paraphrase mean margin drift: {pair['canonical_mean_margin_drift']:.6f}/{pair['paraphrase_mean_margin_drift']:.6f}. Canonical/paraphrase SUPPORT→NOT_ENTITLED: {pair['canonical_support_to_not_entitled']}/{pair['paraphrase_support_to_not_entitled']}. Both lost: {pair['both_rows_lost_support']}; both preserved: {pair['both_preserved']}.",
        "",
        "## Correctness tradeoff",
        "",
        f"Recovered errors: {trade['recovered_errors']}; regressed errors: {trade['regressed_errors']}; net correct rows: {trade['net_correctness_change']}.",
        "",
        "## Diagnosis",
        "",
    ])
    for name, value in report["diagnosis"].items():
        lines.append(f"- `{name}`: supported={str(value['supported']).lower()}; counts={json.dumps(value['supporting_counts'], ensure_ascii=False, sort_keys=True)}")
    lines.extend(["", "The diagnoses are descriptive checkpoint attribution, not causal claims.", ""])
    return "\n".join(lines)


def _write_blocked(output_dir: Path, error: Exception) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": STAGE,
        "decision": BLOCKED,
        "scope": {"diagnostic_only": True, "model_forward_executed": False},
        "input_validation": {"status": "blocked", "error": f"{type(error).__name__}: {error}"},
        "baseline": None,
        "treatment": None,
        "split": None,
        "overall_transition": None,
        "category_attribution": None,
        "intervention_attribution": None,
        "eligible_pair_drift": None,
        "correctness_tradeoff": None,
        "diagnosis": None,
        "limitations": ["Validation failed before model construction/forward; no attribution was computed."],
        "safety_policy": {"training": False, "optimizer": False, "backward": False, "external_evaluation": False, "time_swap": False},
    }
    (output_dir / OUTPUT_NAMES["json"]).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / OUTPUT_NAMES["md"]).write_text(f"# Stage176-A blocked\n\n**Decision:** `{BLOCKED}`\n\nValidation failed before model forward: `{type(error).__name__}: {error}`\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/controlled_v5_v3_without_time_swap.jsonl"))
    parser.add_argument("--baseline-provenance", type=Path, required=True)
    parser.add_argument("--treatment-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--treatment-checkpoint", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        _require(args.eval_batch_size > 0, "--eval-batch-size must be positive")
        _require(args.device == "cuda", "Stage176-A requires --device cuda")
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        data_path = args.data.resolve()
        _require(data_path.is_file(), f"data file does not exist: {data_path}")
        baseline_path = args.baseline_provenance.resolve()
        treatment_path = args.treatment_provenance.resolve()
        _require(baseline_path.is_file(), f"baseline provenance missing: {baseline_path}")
        _require(treatment_path.is_file(), f"treatment provenance missing: {treatment_path}")
        baseline_prov = _read_json(baseline_path)
        treatment_prov = _read_json(treatment_path)
        validation = _validate_provenances(baseline_prov, treatment_prov, data_path)
        baseline_checkpoint = _resolve_checkpoint(baseline_path, baseline_prov, args.baseline_checkpoint)
        treatment_checkpoint = _resolve_checkpoint(treatment_path, treatment_prov, args.treatment_checkpoint)
        _require(baseline_checkpoint != treatment_checkpoint, "baseline and treatment resolve to the same checkpoint")
        baseline_state, baseline_metadata = _load_checkpoint_header(baseline_checkpoint)
        treatment_state, treatment_metadata = _load_checkpoint_header(treatment_checkpoint)
        _validate_checkpoint_metadata(baseline_checkpoint, baseline_metadata, baseline_prov)
        _validate_checkpoint_metadata(treatment_checkpoint, treatment_metadata, treatment_prov)

        records = v5.load_jsonl(data_path)
        seed = int(_runtime(baseline_prov, "seed"))
        dev_ratio = float(_arg(baseline_prov, "dev_ratio", 0.2))
        train_records, dev_records = v5.split_by_pair_id(records, dev_ratio=dev_ratio, seed=seed)
        train_pairs = {str(row["pair_id"]) for row in train_records}
        dev_pairs = {str(row["pair_id"]) for row in dev_records}
        split = {
            "method": "scripts.build_controlled_v5.split_by_pair_id",
            "seed": seed,
            "dev_ratio": dev_ratio,
            "total_pair_groups": len(train_pairs | dev_pairs),
            "train_pair_groups": len(train_pairs),
            "dev_pair_groups": len(dev_pairs),
            "dev_rows": len(dev_records),
            "train_dev_overlap": len(train_pairs & dev_pairs),
        }
        for key in ("total_pair_groups", "train_pair_groups", "dev_pair_groups", "dev_rows"):
            _require(split[key] == EXPECTED[key], f"split {key}: expected {EXPECTED[key]}, got {split[key]}")
        _require(split["train_dev_overlap"] == 0, "train/dev pair overlap detected")
        eligible, eligible_validation = build_train_support_anchor_index(
            dev_records, [], expected_eligible_count=EXPECTED["eligible_dev_support_anchor_pairs"]
        )
        split["eligible_dev_support_anchor_pairs"] = len(eligible)
        split["eligible_validation"] = eligible_validation

        # All provenance, checkpoint metadata, data hash, split, and eligibility
        # checks above complete before either model is constructed or forwarded.
        from transformers import AutoTokenizer

        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(str(_runtime(baseline_prov, "model_name")))
        if tokenizer.pad_token_id is None:
            _require(tokenizer.eos_token_id is not None, "Mamba tokenizer has no pad/eos token")
            tokenizer.pad_token = tokenizer.eos_token
        max_length = int(_arg(baseline_prov, "max_length", baseline_metadata.get("max_length", 128)))
        _require(max_length == int(_arg(treatment_prov, "max_length", treatment_metadata.get("max_length", 128))), "max_length mismatch")
        bundle = v5.encode_mamba_records(dev_records, tokenizer, max_length)
        dev_inputs = v5.move_inputs(bundle["model_inputs"], device)

        baseline_model = _construct_model(baseline_prov, baseline_metadata, baseline_state, device)
        treatment_model = _construct_model(treatment_prov, treatment_metadata, treatment_state, device)
        baseline_output = _forward(baseline_model, dev_inputs, dev_records, baseline_prov, device, args.eval_batch_size)
        treatment_output = _forward(treatment_model, dev_inputs, dev_records, treatment_prov, device, args.eval_batch_size)

        rows = _build_rows(dev_records, baseline_output, treatment_output, eligible)
        matrix_rows, overall = _matrix(rows)
        categories = {name: _summary([row for row in rows if row["category"] == name]) for name in CATEGORIES}
        _require(categories["eligible_support_paraphrase"]["row_count"] == 29, "eligible paraphrase category count mismatch")
        _require(categories["eligible_support_canonical_none"]["row_count"] == 29, "eligible canonical category count mismatch")
        interventions = _intervention_summary(rows)
        pair_rows, pair_aggregate = _eligible_pair_drift(rows, eligible)
        correctness = {
            "overall": _trade(rows),
            "by_category": {name: value["correctness_tradeoff"] for name, value in categories.items()},
            "by_intervention": {name: value["correctness_tradeoff"] for name, value in interventions.items()},
            "gold_not_entitled_false_support_removed": sum(row["gold_final_label"] == "NOT_ENTITLED" and row["prediction_transition"] == "SUPPORT->NOT_ENTITLED" for row in rows),
            "gold_refute_false_support_removed": sum(row["gold_final_label"] == "REFUTE" and row["prediction_transition"] != "SUPPORT->SUPPORT" and row["baseline_predicted_label"] == "SUPPORT" and row["treatment_predicted_label"] != "SUPPORT" for row in rows),
            "gold_support_true_support_lost": sum(row["gold_final_label"] == "SUPPORT" and row["baseline_predicted_label"] == "SUPPORT" and row["treatment_predicted_label"] != "SUPPORT" for row in rows),
            "other_recovered_or_regressed_rows": [row["row_id"] for row in rows if row["correctness_transition"] in ("incorrect_to_correct", "correct_to_incorrect") and not (row["gold_final_label"] in ("NOT_ENTITLED", "SUPPORT") and row["baseline_predicted_label"] == "SUPPORT")],
        }
        diagnosis = _diagnosis(categories, interventions, pair_aggregate)
        validation.update({
            "baseline_checkpoint_metadata": "passed",
            "treatment_checkpoint_metadata": "passed",
            "validation_completed_before_model_construction_and_forward": True,
        })
        report = {
            "stage": STAGE,
            "decision": COMPLETE,
            "scope": {
                "data": str(data_path),
                "diagnostic_only": True,
                "prediction_source": 'output["predictions"] == output["logits"].argmax(dim=-1)',
                "support_margin_source": 'output["logits"]',
                "support_margin_definition": "support_logit - logsumexp(not_entitled_logit, refute_logit)",
                "eval_batch_size": args.eval_batch_size,
                "device": str(device),
            },
            "input_validation": validation,
            "baseline": {"provenance": str(baseline_path), "checkpoint": str(baseline_checkpoint), "selected_epoch": _selected_epoch(baseline_prov)},
            "treatment": {"provenance": str(treatment_path), "checkpoint": str(treatment_checkpoint), "selected_epoch": _selected_epoch(treatment_prov)},
            "split": split,
            "overall_transition": overall,
            "category_attribution": categories,
            "intervention_attribution": interventions,
            "eligible_pair_drift": {"aggregate": pair_aggregate, "rows": pair_rows},
            "correctness_tradeoff": correctness,
            "diagnosis": diagnosis,
            "limitations": [
                "Single-seed, two-checkpoint observational attribution on internal clean dev only.",
                "No threshold, calibration, causal intervention, external evaluation, or uncertainty estimate is introduced.",
                "A net SUPPORT count delta does not by itself identify every row transition; the 3x3 matrix is authoritative.",
            ],
            "safety_policy": {
                "clean_dev_only": True,
                "training": False,
                "optimizer_created": False,
                "backward": False,
                "threshold_tuning": False,
                "calibration": False,
                "checkpoint_selection": False,
                "external_evaluation": False,
                "external_labels": False,
                "time_swap": False,
                "loss_modified_or_implemented": False,
            },
        }
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / OUTPUT_NAMES["json"]).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / OUTPUT_NAMES["md"]).write_text(_render_markdown(report), encoding="utf-8")
        _write_csv(output_dir / OUTPUT_NAMES["rows"], rows)
        _write_csv(output_dir / OUTPUT_NAMES["categories"], _named_rows(categories, "category"))
        _write_csv(output_dir / OUTPUT_NAMES["interventions"], _named_rows(interventions, "intervention_type"))
        _write_csv(output_dir / OUTPUT_NAMES["pairs"], pair_rows)
        _write_csv(output_dir / OUTPUT_NAMES["matrix"], matrix_rows)
        print(json.dumps({"decision": COMPLETE, "output_dir": str(output_dir)}, sort_keys=True))
        return 0
    except ValidationBlocked as error:
        _write_blocked(args.output_dir.resolve(), error)
        print(json.dumps({"decision": BLOCKED, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
