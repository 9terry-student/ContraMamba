"""Stage177-E baseline-versus-frame-pairwise pilot attribution audit.

Evaluation only: validate both persisted seed-174 epoch-20 checkpoints before
model construction, compare their clean controlled outputs, and emit the fixed
Stage177-E attribution artifacts.  This module never trains, fits, calibrates,
searches a threshold, selects a checkpoint, or calls backward().
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import analyze_stage176a_support_boundary_attribution as stage176a  # noqa: E402
from scripts import analyze_stage176b_native_structural_separability as stage176b  # noqa: E402
from scripts import analyze_stage177a_frame_head_hard_subset as stage177a  # noqa: E402
from scripts import train_controlled_v5 as v5  # noqa: E402


STAGE = "Stage177-E"
CLEAN = "STAGE177E_FRAME_PAIRWISE_PILOT_CLEAN_BENEFIT_IDENTIFIED"
REPRESENTATION = "STAGE177E_FRAME_PAIRWISE_REPRESENTATION_CHANGE_WITHOUT_FINAL_BENEFIT"
REDUNDANT = "STAGE177E_FRAME_PAIRWISE_OBJECTIVE_REDUNDANT_PATH_CLOSED"
HARMFUL = "STAGE177E_FRAME_PAIRWISE_OBJECTIVE_HARMFUL_PATH_CLOSED"
BLOCKED = "STAGE177E_FRAME_PAIRWISE_PILOT_ATTRIBUTION_BLOCKED"
STAGE176A_COMPLETE = "STAGE176A_CLEAN_DEV_SUPPORT_BOUNDARY_ATTRIBUTION_COMPLETE"
STAGE177A_DECISION = "STAGE177A_FRAME_PAIRWISE_SIGNAL_PRESENT_ABSOLUTE_DISCRIMINATION_WEAK"
SCHEMA = "stage176a0_selected_checkpoint_v1"
MODEL_NAME = "state-spaces/mamba-130m-hf"
LABELS = ("NOT_ENTITLED", "REFUTE", "SUPPORT")
OUTPUTS = {
    "json": "stage177e_frame_pairwise_pilot_attribution_report.json",
    "md": "stage177e_frame_pairwise_pilot_attribution_report.md",
    "rows": "stage177e_dev_row_comparison.csv",
    "transitions": "stage177e_final_transition_matrix.csv",
    "frame": "stage177e_frame_metric_comparison.csv",
    "ranking": "stage177e_pair_ranking_comparison.csv",
    "pairs": "stage177e_pair_level_delta.csv",
    "family_pairs": "stage177e_family_pair_delta.csv",
    "cohorts": "stage177e_stage176_cohort_comparison.csv",
    "frame_errors": "stage177e_frame_error_transitions.csv",
    "parameter_summary": "stage177e_parameter_delta_summary.csv",
    "modules": "stage177e_module_parameter_delta.csv",
}


class AuditBlocked(ValueError):
    """A hard input, checkpoint, or semantic-contract failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditBlocked(message)


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _require_int(value: Any, *, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Required integer field is missing: {field_name}")
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a valid integer for {field_name}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer field {field_name}: {value!r}") from exc


def _optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, field_name=field_name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditBlocked(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except OSError as error:
        raise AuditBlocked(f"cannot read CSV {path}: {error}") from error
    _require(bool(rows), f"CSV is empty: {path}")
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def _median(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.median(items) if items else None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "percentile requires values")
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ci(values: list[float]) -> list[float] | None:
    return [_percentile(values, .025), _percentile(values, .975)] if values else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage(prov: dict[str, Any], name: str) -> dict[str, Any]:
    value = prov.get(name) or stage176a._get(prov, f"run_report.{name}", {}) or {}
    return value if isinstance(value, dict) else {}


def _resolve_provenance_integer_fields(run_name: str, provenance: dict[str, Any]) -> dict[str, int]:
    parsed_args = provenance.get("parsed_args") or {}
    finalization = provenance.get("finalization") or {}
    selected_checkpoint = finalization.get("selected_checkpoint") or {}
    _require(isinstance(parsed_args, dict), f"{run_name}.parsed_args must be an object")
    _require(isinstance(finalization, dict), f"{run_name}.finalization must be an object")
    _require(isinstance(selected_checkpoint, dict),
             f"{run_name}.finalization.selected_checkpoint must be an object")
    seed = _require_int(parsed_args.get("seed"), field_name=f"{run_name}.parsed_args.seed")
    epochs = _require_int(parsed_args.get("epochs"), field_name=f"{run_name}.parsed_args.epochs")
    selected_epoch_raw = _first_non_none(
        finalization.get("selected_epoch"),
        selected_checkpoint.get("selected_epoch"),
    )
    selected_epoch = _require_int(
        selected_epoch_raw,
        field_name=f"{run_name}.finalization.selected_epoch",
    )
    checkpoint_selected_epoch = _optional_int(
        selected_checkpoint.get("selected_epoch"),
        field_name=f"{run_name}.finalization.selected_checkpoint.selected_epoch",
    )
    if checkpoint_selected_epoch is not None:
        _require(checkpoint_selected_epoch == selected_epoch,
                 f"{run_name} finalization selected epoch fields disagree")
    return {"seed": seed, "epochs": epochs, "selected_epoch": selected_epoch}


def _validate_stage176_csv_identifiers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {"stable_row_index", "row_id", "pair_id", "intervention_type"}
    missing = sorted(required - set(rows[0]))
    _require(not missing, f"Stage176-A row-transition CSV missing columns: {missing}")
    for line_number, row in enumerate(rows, start=2):
        row["stable_row_index"] = _require_int(
            row.get("stable_row_index"),
            field_name=f"stage176a_row_transitions.line_{line_number}.stable_row_index",
        )
        for key in ("row_id", "pair_id", "intervention_type"):
            value = row.get(key)
            _require(value is not None and str(value) != "",
                     f"Stage176-A CSV identity is missing at line {line_number}: {key}")
            row[key] = str(value)
    return rows


def _validate_stage_reports(stage176a_report: dict[str, Any],
                            stage177a_report: dict[str, Any]) -> dict[str, Any]:
    decision176 = stage176a._first(stage176a_report, (
        "closure.attribution_completion_decision", "decision"))
    _require(decision176 == STAGE176A_COMPLETE, "Stage176-A completion decision mismatch")
    trade = stage176a_report.get("correctness_tradeoff") or {}
    beneficial_raw = stage176a._first(stage176a_report, (
        "correctness_tradeoff.gold_not_entitled_false_support_removed",
        "correctness_tradeoff.beneficial_correction.rows",
        "intervention_attribution.beneficial_correction.total"))
    harmful_raw = stage176a._first(stage176a_report, (
        "correctness_tradeoff.gold_support_true_support_lost",
        "correctness_tradeoff.harmful_regression.rows",
        "intervention_attribution.harmful_regression.total"))
    beneficial = _require_int(
        beneficial_raw,
        field_name="stage176a.required_beneficial_correction_count",
    )
    harmful = _require_int(
        harmful_raw,
        field_name="stage176a.required_harmful_regression_count",
    )
    _require(beneficial == 25 and harmful == 14,
             "Stage176-A beneficial/harmful cohorts must be 25/14")
    _require(stage177a_report.get("decision") == STAGE177A_DECISION,
             "Stage177-A decision mismatch")
    return {"status": "passed", "stage176a_decision": decision176,
            "beneficial_correction_rows": 25, "harmful_regression_rows": 14,
            "stage177a_decision": STAGE177A_DECISION,
            "stage176a_tradeoff_present": bool(trade)}


def _validate_provenance(role: str, prov: dict[str, Any], data_path: Path) -> dict[str, Any]:
    _require(prov.get("status") == "completed", f"{role} status must be completed")
    integers = _resolve_provenance_integer_fields(role, prov)
    _require(integers["seed"] == 174, f"{role} seed must be 174")
    _require(integers["epochs"] == 20, f"{role} epochs must be 20")
    _require(integers["selected_epoch"] == 20, f"{role} selected epoch must be 20")
    expected = {"architecture": "v6b_minimal", "backbone": "mamba",
                "model_name": MODEL_NAME}
    for key, value in expected.items():
        _require(stage176a._runtime(prov, key) == value, f"{role} {key} mismatch")
    record = stage176a._data_record(prov)
    actual_data_sha = _sha256(data_path)
    _require(record.get("sha256") == actual_data_sha, f"{role} data SHA-256 mismatch")
    data_row_count = _require_int(record.get("row_count"), field_name=f"{role}.data_provenance.main_data.row_count")
    _require(data_row_count == 3600, f"{role} data row count mismatch")
    recorded_path = record.get("resolved_path") or record.get("path")
    _require(recorded_path is not None and Path(str(recorded_path)).resolve() == data_path,
             f"{role} data path mismatch")
    _require(math.isclose(float(stage176a._arg(prov, "dev_ratio", -1)), .2),
             f"{role} dev_ratio must be 0.2")
    stage174 = _stage(prov, "stage174c_clean_pairwise")
    stage175 = _stage(prov, "stage175b_support_anchor")
    stage177 = _stage(prov, "stage177c_frame_pairwise")
    _require(stage174.get("mode") == "off" and float(stage174.get("weight", 0)) == 0,
             f"{role} Stage174-C must be off/0")
    _require(stage175.get("mode") == "off" and float(stage175.get("weight", 0)) == 0,
             f"{role} Stage175-B must be off/0")
    if role == "baseline":
        _require(stage177.get("mode", "off") == "off" and float(stage177.get("weight", 0)) == 0,
                 "baseline Stage177-C must be off/0")
    else:
        _require(stage177.get("mode") == "pair_softplus" and
                 math.isclose(float(stage177.get("weight", -1)), .05),
                 "pilot Stage177-C must be pair_softplus/0.05")
        for key, wanted in (("eligible_pair_count", 240), ("malformed_pair_count", 0)):
            actual = _require_int(
                stage176a._first(stage177, (f"topology.{key}", key)),
                field_name=f"pilot.stage177c_frame_pairwise.topology.{key}",
            )
            _require(actual == wanted, f"pilot Stage177-C {key} mismatch")
        raw = _require_int(
            stage176a._first(stage177, ("topology.raw_comparison_count", "raw_comparison_count")),
            field_name="pilot.stage177c_frame_pairwise.topology.raw_comparison_count",
        )
        _require(raw == 8640, "pilot Stage177-C raw comparisons must be 8640 per epoch")
        activities = stage177.get("run_activity") or {}
        epoch_metrics = [item for run in activities.values() if isinstance(run, dict)
                         for item in (run.get("epoch_metrics") or [])]
        _require(len(epoch_metrics) == 20, "pilot Stage177-C must expose 20 epoch diagnostics")
        _require(all(item.get("gradient_enabled") is True for item in epoch_metrics),
                 "pilot Stage177-C gradient must be enabled throughout")
        for epoch_index, item in enumerate(epoch_metrics, start=1):
            eligible = _require_int(
                item.get("eligible_pair_count"),
                field_name=f"pilot.stage177c.epoch_metrics[{epoch_index}].eligible_pair_count",
            )
            comparisons = _require_int(
                item.get("raw_comparison_count"),
                field_name=f"pilot.stage177c.epoch_metrics[{epoch_index}].raw_comparison_count",
            )
            malformed = _require_int(
                item.get("malformed_pair_count"),
                field_name=f"pilot.stage177c.epoch_metrics[{epoch_index}].malformed_pair_count",
            )
            _require((eligible, comparisons, malformed) == (240, 8640, 0),
                     f"pilot Stage177-C epoch {epoch_index} topology mismatch")
        nonfinite = _require_int(
            stage176a._first(stage177, ("aggregate_diagnostics.non_finite_count",)),
            field_name="pilot.stage177c_frame_pairwise.aggregate_diagnostics.non_finite_count",
        )
        direct = stage176a._first(stage177, ("final_classifier_logits_targeted",), None)
        extra = stage176a._first(stage177, ("extra_counterpart_forward_used",), None)
        _require(nonfinite == 0, "pilot Stage177-C non-finite count must be zero")
        _require(direct is False and extra is False,
                 "pilot Stage177-C target/forward contract mismatch")
    policy = prov.get("training_selection_policy") or {}
    _require(policy.get("final_ce_logits_source") in ('output["logits"]', "output['logits']"),
             f"{role} final CE source mismatch")
    _require(policy.get("loss_logits_used_for_final_classifier_ce") is False,
             f"{role} loss_logits final-CE use must be false")
    for key in ("external_evaluation_used_for_training", "external_evaluation_used_for_calibration",
                "external_evaluation_used_for_threshold_selection", "external_evaluation_used_for_checkpoint_selection",
                "time_swap_included_in_main_classification_training"):
        _require(policy.get(key) is False, f"{role} policy {key} must be false")
    return {"status": "passed", **expected, "seed": integers["seed"],
            "epochs": integers["epochs"], "selected_epoch": integers["selected_epoch"],
            "data_sha256": actual_data_sha, "stage174c_off": True,
            "stage175b_off": True, "stage177c": {"mode": stage177.get("mode", "off"),
            "weight": float(stage177.get("weight", 0))}, "external_or_time_swap": False}


def _load_and_validate_checkpoint(role: str, path: Path, prov: dict[str, Any],
                                  expected_stage177_mode: str, expected_weight: float
                                  ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    state, metadata, payload = stage176b._load_checkpoint(path)
    finalization = prov.get("finalization") or {}
    record = finalization.get("selected_checkpoint") or {}
    _require(isinstance(record, dict) and bool(record),
             f"{role}.finalization.selected_checkpoint is missing")
    integers = _resolve_provenance_integer_fields(role, prov)
    actual_sha = _sha256(path)
    _require(payload.get("schema_version") == SCHEMA and record.get("schema_version") == SCHEMA,
             f"{role} checkpoint schema mismatch")
    _require(record.get("sha256") == actual_sha, f"{role} checkpoint SHA-256 mismatch")
    _require(record.get("saved") is True and record.get("path"),
             f"{role} selected checkpoint was not persisted")
    metadata_epoch = _optional_int(
        metadata.get("selected_epoch"),
        field_name=f"{role}.checkpoint.metadata.selected_epoch",
    )
    if metadata_epoch is not None:
        _require(metadata_epoch == integers["selected_epoch"],
                 f"{role} checkpoint selected epoch mismatch")
    metadata_seed = _optional_int(
        metadata.get("seed"),
        field_name=f"{role}.checkpoint.metadata.seed",
    )
    if metadata_seed is not None:
        _require(metadata_seed == integers["seed"], f"{role} checkpoint seed mismatch")
    _require(metadata.get("main_data_sha256") == stage176a._data_record(prov).get("sha256"),
             f"{role} checkpoint data SHA-256 mismatch")
    for key, value in (("architecture", "v6b_minimal"),
                       ("backbone", "mamba"), ("model_name", MODEL_NAME)):
        _require(metadata.get(key) == value, f"{role} checkpoint {key} mismatch")
    _require(metadata.get("stage174c_clean_pairwise_mode") == "off" and
             float(metadata.get("stage174c_clean_pairwise_weight", -1)) == 0,
             f"{role} checkpoint Stage174-C mismatch")
    _require(metadata.get("stage175b_support_anchor_mode") == "off" and
             float(metadata.get("stage175b_support_anchor_weight", -1)) == 0,
             f"{role} checkpoint Stage175-B mismatch")
    stage177 = metadata.get("stage177c_frame_pairwise") or {}
    mode = stage177.get("mode", metadata.get("stage177c_frame_pairwise_mode", "off"))
    weight = float(stage177.get("weight", metadata.get("stage177c_frame_pairwise_weight", 0)))
    _require(mode == expected_stage177_mode and math.isclose(weight, expected_weight),
             f"{role} checkpoint Stage177-C mismatch")
    _require(metadata.get("final_ce_logits_source") in ('output["logits"]', "output['logits']"),
             f"{role} checkpoint CE source mismatch")
    _require(metadata.get("loss_logits_used_for_final_classifier_ce") is False,
             f"{role} checkpoint loss_logits use mismatch")
    _require(metadata.get("external_data_used") is False and
             metadata.get("external_labels_used") is False and metadata.get("time_swap_used") is False,
             f"{role} checkpoint external/time_swap contract mismatch")
    return state, metadata, {"status": "passed", "path": str(path), "sha256": actual_sha,
                             "schema_version": SCHEMA, "selected_epoch": integers["selected_epoch"],
                             "metadata_selected_epoch_present": metadata_epoch is not None,
                             "metadata_seed_present": metadata_seed is not None,
                             "stage177c_mode": mode, "stage177c_weight": weight}


def _validate_split(data_path: Path, prov: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = v5.load_jsonl(data_path)
    train, dev = v5.split_by_pair_id(records, dev_ratio=float(stage176a._arg(prov, "dev_ratio", .2)), seed=174)
    train_pairs = {str(row["pair_id"]) for row in train}
    dev_pairs = {str(row["pair_id"]) for row in dev}
    identities = [str(row.get("id")) for row in records]
    _require((len(records), len(train), len(dev)) == (3600, 2880, 720), "row split must be 3600/2880/720")
    _require((len(train_pairs), len(dev_pairs), len(train_pairs & dev_pairs)) == (240, 60, 0),
             "pair split must be 240/60 with zero overlap")
    _require(len(set(identities)) == len(identities), "stable row identities are not unique")
    return train, dev, {"status": "passed", "total_pairs": 300, "train_pairs": 240,
                        "dev_pairs": 60, "train_rows": 2880, "dev_rows": 720,
                        "pair_overlap": 0, "stable_row_identity_unique": True,
                        "ordering_source": "scripts.train_controlled_v5.split_by_pair_id"}


def _tensor(output: dict[str, Any], key: str, count: int) -> torch.Tensor:
    value = output.get(key)
    _require(torch.is_tensor(value), f"model output lacks tensor {key}")
    value = value.detach().float().cpu()
    _require(value.shape[0] == count and torch.isfinite(value).all().item(),
             f"model output {key} shape/finite check failed")
    return value


def _evaluate(records: list[dict[str, Any]], baseline_prov: dict[str, Any], pilot_prov: dict[str, Any],
              baseline_metadata: dict[str, Any], pilot_metadata: dict[str, Any],
              baseline_state: dict[str, torch.Tensor], pilot_state: dict[str, torch.Tensor],
              device: torch.device, batch_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        _require(tokenizer.eos_token_id is not None, "tokenizer has no pad/eos token")
        tokenizer.pad_token = tokenizer.eos_token
    baseline_max_length = _require_int(
        _first_non_none(
            (baseline_prov.get("parsed_args") or {}).get("max_length"),
            baseline_metadata.get("max_length"),
        ),
        field_name="baseline.parsed_args.max_length",
    )
    pilot_max_length = _require_int(
        _first_non_none(
            (pilot_prov.get("parsed_args") or {}).get("max_length"),
            pilot_metadata.get("max_length"),
        ),
        field_name="pilot.parsed_args.max_length",
    )
    _require(baseline_max_length == pilot_max_length, "baseline/pilot max_length mismatch")
    max_length = baseline_max_length
    bundle = v5.encode_mamba_records(records, tokenizer, max_length)
    inputs = v5.move_inputs(bundle["model_inputs"], device)
    baseline_model = stage176a._construct_model(baseline_prov, baseline_metadata, baseline_state, device)
    pilot_model = stage176a._construct_model(pilot_prov, pilot_metadata, pilot_state, device)
    _require(not baseline_model.training and not pilot_model.training, "models must remain in eval mode")
    baseline = stage176a._forward(baseline_model, inputs, records, baseline_prov, device, batch_size)
    pilot = stage176a._forward(pilot_model, inputs, records, pilot_prov, device, batch_size)
    retained_keys = ("logits", "predictions", "frame_logit", "frame_prob")
    for output in (baseline, pilot):
        for key in ("logits", "frame_logit", "frame_prob"):
            _tensor(output, key, len(records))
    # Retain only the native tensors required by this audit, on CPU.  The
    # trainer forward may expose large hidden diagnostics that must not remain
    # resident while the second split is evaluated.
    baseline_compact = {key: baseline[key].detach().cpu() for key in retained_keys}
    pilot_compact = {key: pilot[key].detach().cpu() for key in retained_keys}
    return baseline_compact, pilot_compact


def _margin(logits: torch.Tensor, target: int) -> torch.Tensor:
    other = [index for index in range(3) if index != target]
    return logits[:, target] - torch.logsumexp(logits[:, other], dim=-1)


def _entropy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = logits.softmax(dim=-1)
    return probabilities.max(dim=-1).values, -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)


def _build_dev_rows(records: list[dict[str, Any]], transitions: list[dict[str, Any]],
                    baseline: dict[str, Any], pilot: dict[str, Any]) -> list[dict[str, Any]]:
    transitions = stage176b._validate_row_alignment(transitions, records)
    count = len(records)
    b_logits, p_logits = _tensor(baseline, "logits", count), _tensor(pilot, "logits", count)
    b_frame, p_frame = _tensor(baseline, "frame_logit", count).reshape(-1), _tensor(pilot, "frame_logit", count).reshape(-1)
    b_prob, p_prob = _tensor(baseline, "frame_prob", count).reshape(-1), _tensor(pilot, "frame_prob", count).reshape(-1)
    b_pred, p_pred = b_logits.argmax(-1), p_logits.argmax(-1)
    b_max, b_entropy = _entropy(b_logits); p_max, p_entropy = _entropy(p_logits)
    margins = {side: {label: _margin(logits, v5.FINAL_LABEL_TO_ID[label]) for label in LABELS}
               for side, logits in (("baseline", b_logits), ("pilot", p_logits))}
    rows = []
    for index, (record, transition) in enumerate(zip(records, transitions)):
        gold = str(record["final_label"])
        bp, pp = v5.ID_TO_FINAL_LABEL[int(b_pred[index])], v5.ID_TO_FINAL_LABEL[int(p_pred[index])]
        cohort = stage176b._cohort(transition)
        row: dict[str, Any] = {
            "stable_row_index": index, "row_id": str(record["id"]),
            "pair_id": str(record["pair_id"]), "intervention_type": str(record["intervention_type"]),
            "gold_label": gold, "gold_frame_label": _require_int(
                record.get("frame_compatible_label"),
                field_name=f"dev_rows[{index}].frame_compatible_label",
            ),
            "stage176_cohort": cohort, "baseline_prediction": bp, "pilot_prediction": pp,
            "prediction_transition": f"{bp}->{pp}", "prediction_changed": bp != pp,
            "baseline_correct": bp == gold, "pilot_correct": pp == gold,
            "correctness_transition": f"{'correct' if bp == gold else 'incorrect'}_to_{'correct' if pp == gold else 'incorrect'}",
            "baseline_frame_logit": float(b_frame[index]), "pilot_frame_logit": float(p_frame[index]),
            "frame_logit_delta": float(p_frame[index] - b_frame[index]),
            "baseline_frame_prob": float(b_prob[index]), "pilot_frame_prob": float(p_prob[index]),
            "frame_prob_delta": float(p_prob[index] - b_prob[index]),
            "baseline_max_probability": float(b_max[index]), "pilot_max_probability": float(p_max[index]),
            "max_probability_delta": float(p_max[index] - b_max[index]),
            "baseline_entropy": float(b_entropy[index]), "pilot_entropy": float(p_entropy[index]),
            "entropy_delta": float(p_entropy[index] - b_entropy[index]),
        }
        for label in LABELS:
            slug = label.lower()
            label_id = v5.FINAL_LABEL_TO_ID[label]
            row[f"baseline_{slug}_logit"] = float(b_logits[index, label_id])
            row[f"pilot_{slug}_logit"] = float(p_logits[index, label_id])
            row[f"{slug}_logit_delta"] = float(p_logits[index, label_id] - b_logits[index, label_id])
            row[f"baseline_{slug}_margin"] = float(margins["baseline"][label][index])
            row[f"pilot_{slug}_margin"] = float(margins["pilot"][label][index])
            row[f"{slug}_margin_delta"] = float(margins["pilot"][label][index] - margins["baseline"][label][index])
        rows.append(row)
    return rows


def _classification(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    gold = [row["gold_label"] for row in rows]
    pred = [row[f"{side}_prediction"] for row in rows]
    per_class = {}
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(g == label for g in gold)}
    matrix = {g: {p: sum(gg == g and pp == p for gg, pp in zip(gold, pred)) for p in LABELS} for g in LABELS}
    return {"accuracy": sum(g == p for g, p in zip(gold, pred)) / len(rows),
            "macro_f1": statistics.fmean(item["f1"] for item in per_class.values()),
            "per_class": per_class, "prediction_counts": dict(sorted(Counter(pred).items())),
            "confusion_matrix_gold_by_prediction": matrix}


def _transition_analysis(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix = []
    for before in LABELS:
        for after in LABELS:
            group = [row for row in rows if row["baseline_prediction"] == before and row["pilot_prediction"] == after]
            matrix.append({"baseline_prediction": before, "pilot_prediction": after, "row_count": len(group),
                           "recovered_errors": sum(not r["baseline_correct"] and r["pilot_correct"] for r in group),
                           "introduced_errors": sum(r["baseline_correct"] and not r["pilot_correct"] for r in group)})
    recovered = sum(not row["baseline_correct"] and row["pilot_correct"] for row in rows)
    introduced = sum(row["baseline_correct"] and not row["pilot_correct"] for row in rows)
    changed = sum(row["prediction_changed"] for row in rows)
    def grouped(key: str) -> dict[str, Any]:
        return {name: dict(sorted(Counter(row["prediction_transition"] for row in rows if row[key] == name).items()))
                for name in sorted({str(row[key]) for row in rows})}
    return matrix, {"changed_rows": changed, "unchanged_rows": len(rows) - changed,
                    "recovered_errors": recovered, "introduced_errors": introduced,
                    "net_correctness_change": recovered - introduced,
                    "harmful_support_regressions": sum(
                        row["gold_label"] == "SUPPORT" and row["baseline_correct"] and not row["pilot_correct"]
                        for row in rows
                    ),
                    "aggregate_equal_but_row_permutation_checked": True,
                    "by_gold_label": grouped("gold_label"),
                    "by_intervention_family": grouped("intervention_type")}


def _drift_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    delta_fields = (
        [f"{label.lower()}_logit_delta" for label in LABELS]
        + [f"{label.lower()}_margin_delta" for label in LABELS]
        + ["max_probability_delta", "entropy_delta"]
    )
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {field: {"mean": _mean(float(r[field]) for r in group),
                        "median": _median(float(r[field]) for r in group)} for field in delta_fields}
    return {"overall": summarize(rows),
            "by_family": {name: summarize([r for r in rows if r["intervention_type"] == name]) for name in sorted({r["intervention_type"] for r in rows})},
            "by_gold_label": {name: summarize([r for r in rows if r["gold_label"] == name]) for name in LABELS},
            "by_correctness_transition": {name: summarize([r for r in rows if r["correctness_transition"] == name]) for name in sorted({r["correctness_transition"] for r in rows})},
            "context_only_not_a_stage177c_target": True}


def _frame_view(rows: list[dict[str, Any]], side: str, samples: int, seed: int,
                hard: bool = False) -> dict[str, Any]:
    converted = [{**row, f"{side}_frame_score": row[f"{side}_frame_logit"],
                  f"{side}_frame_prediction": int(float(row[f"{side}_frame_prob"]) >= .5)} for row in rows]
    if hard:
        return stage177a._hard_metrics(converted, side, samples, seed)
    metrics = stage177a._binary_metrics(converted, side)
    zero = [float(r[f"{side}_frame_logit"]) for r in converted if int(r["gold_frame_label"]) == 0]
    one = [float(r[f"{side}_frame_logit"]) for r in converted if int(r["gold_frame_label"]) == 1]
    auc = metrics["auroc"]
    metrics.update({"mean_incompatible": _mean(zero), "median_incompatible": _median(zero),
                    "mean_compatible": _mean(one), "median_compatible": _median(one),
                    "cliffs_delta": 2 * float(auc) - 1 if auc is not None else None,
                    "frame_accuracy": metrics["confusion"]["accuracy"],
                    "balanced_accuracy": metrics["confusion"]["balanced_accuracy"]})
    return metrics


def _pair_records(records: list[dict[str, Any]], baseline: dict[str, Any], pilot: dict[str, Any],
                  split: str) -> list[dict[str, Any]]:
    b = _tensor(baseline, "frame_logit", len(records)).reshape(-1).tolist()
    p = _tensor(pilot, "frame_logit", len(records)).reshape(-1).tolist()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record["pair_id"])].append({**record, "baseline": b[index], "pilot": p[index]})
    output = []
    for pair_id, group in sorted(groups.items()):
        compatible = [row for row in group if _require_int(
            row.get("frame_compatible_label"), field_name=f"{split}.{pair_id}.frame_compatible_label"
        ) == 1]
        incompatible = [row for row in group if _require_int(
            row.get("frame_compatible_label"), field_name=f"{split}.{pair_id}.frame_compatible_label"
        ) == 0]
        _require(len(compatible) == 6 and len(incompatible) == 6, f"malformed pair {pair_id}")
        item: dict[str, Any] = {"split": split, "pair_id": pair_id, "comparison_count": 36}
        for side in ("baseline", "pilot"):
            gaps = [float(pos[side]) - float(neg[side]) for pos in compatible for neg in incompatible]
            correct, tie = sum(gap > 0 for gap in gaps), sum(gap == 0 for gap in gaps)
            item[f"{side}_ranking_accuracy"] = correct / len(gaps)
            item[f"{side}_mean_gap"] = statistics.fmean(gaps)
            item[f"{side}_median_gap"] = statistics.median(gaps)
            item[f"{side}_correct_comparisons"] = correct
            item[f"{side}_ties"] = tie
            item[f"{side}_violations"] = len(gaps) - correct - tie
        item["ranking_delta"] = item["pilot_ranking_accuracy"] - item["baseline_ranking_accuracy"]
        item["mean_gap_delta"] = item["pilot_mean_gap"] - item["baseline_mean_gap"]
        output.append(item)
    return output


def _ranking_summary(pairs: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    def side_summary(side: str) -> dict[str, Any]:
        comparisons = sum(int(row["comparison_count"]) for row in pairs)
        correct = sum(int(row[f"{side}_correct_comparisons"]) for row in pairs)
        accuracy = correct / comparisons
        normalized = statistics.fmean(float(row[f"{side}_ranking_accuracy"]) for row in pairs)
        rng = random.Random(seed + (0 if side == "baseline" else 1))
        boot = []
        for _ in range(samples):
            sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            boot.append(statistics.fmean(float(row[f"{side}_ranking_accuracy"]) for row in sample))
        return {"comparison_ranking_accuracy": accuracy,
                "pair_normalized_ranking_accuracy": normalized,
                "mean_compatible_minus_incompatible_gap": statistics.fmean(float(row[f"{side}_mean_gap"]) for row in pairs),
                "median_compatible_minus_incompatible_gap": statistics.median(float(row[f"{side}_median_gap"]) for row in pairs),
                "fully_ordered_pair_count": sum(int(row[f"{side}_violations"]) == 0 and int(row[f"{side}_ties"]) == 0 for row in pairs),
                "partially_violated_pair_count": sum(int(row[f"{side}_correct_comparisons"]) > 0 and (int(row[f"{side}_violations"]) > 0 or int(row[f"{side}_ties"]) > 0) for row in pairs),
                "fully_reversed_pair_count": sum(int(row[f"{side}_correct_comparisons"]) == 0 for row in pairs),
                "pair_bootstrap_ranking_ci95": _ci(boot)}
    baseline, pilot = side_summary("baseline"), side_summary("pilot")
    base_gap = float(baseline["mean_compatible_minus_incompatible_gap"])
    return {"baseline": baseline, "pilot": pilot,
            "pilot_minus_baseline": {
                "ranking_delta": float(pilot["pair_normalized_ranking_accuracy"]) - float(baseline["pair_normalized_ranking_accuracy"]),
                "mean_gap_delta": float(pilot["mean_compatible_minus_incompatible_gap"]) - base_gap,
                "mean_gap_relative_increase": ((float(pilot["mean_compatible_minus_incompatible_gap"]) - base_gap) / abs(base_gap)) if base_gap else None,
                "improved_pairs": sum(float(row["ranking_delta"]) > 0 or float(row["mean_gap_delta"]) > 0 for row in pairs),
                "regressed_pairs": sum(float(row["ranking_delta"]) < 0 or float(row["mean_gap_delta"]) < 0 for row in pairs)}}


def _family_pair_delta(records: list[dict[str, Any]], outputs: tuple[dict[str, Any], dict[str, Any]], split: str) -> list[dict[str, Any]]:
    baseline, pilot = outputs
    b = _tensor(baseline, "frame_logit", len(records)).reshape(-1).tolist()
    p = _tensor(pilot, "frame_logit", len(records)).reshape(-1).tolist()
    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    by_pair: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records): by_pair[str(record["pair_id"])].append(index)
    for indexes in by_pair.values():
        pos = [i for i in indexes if _require_int(records[i].get("frame_compatible_label"),
                                                  field_name=f"{split}.rows[{i}].frame_compatible_label") == 1]
        neg = [i for i in indexes if _require_int(records[i].get("frame_compatible_label"),
                                                  field_name=f"{split}.rows[{i}].frame_compatible_label") == 0]
        for i in pos:
            for j in neg:
                groups[(str(records[i]["intervention_type"]), str(records[j]["intervention_type"]))].append((b[i] - b[j], p[i] - p[j]))
    return [{"split": split, "compatible_family": key[0], "incompatible_family": key[1],
             "comparison_count": len(values),
             "baseline_ranking_accuracy": _mean(a > 0 for a, _ in values),
             "pilot_ranking_accuracy": _mean(z > 0 for _, z in values),
             "ranking_delta": _mean(z > 0 for _, z in values) - _mean(a > 0 for a, _ in values),
             "baseline_mean_gap": _mean(a for a, _ in values), "pilot_mean_gap": _mean(z for _, z in values),
             "mean_gap_delta": _mean(z - a for a, z in values)} for key, values in sorted(groups.items())]


def _cohort_attribution(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hard = [row for row in rows if row["stage176_cohort"] in ("beneficial_correction", "harmful_regression")]
    _require(Counter(row["stage176_cohort"] for row in hard) == {"beneficial_correction": 25, "harmful_regression": 14},
             "Stage176 cohort alignment must be 25/14")
    _require(all(row["gold_label"] == "NOT_ENTITLED" and row["gold_frame_label"] == 0 for row in hard if row["stage176_cohort"] == "beneficial_correction"),
             "beneficial cohort gold contract mismatch")
    _require(all(row["gold_label"] == "SUPPORT" and row["gold_frame_label"] == 1 for row in hard if row["stage176_cohort"] == "harmful_regression"),
             "harmful cohort gold contract mismatch")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    csv_rows = []
    for row in hard:
        intended_frame = row["frame_logit_delta"] < 0 if row["stage176_cohort"] == "beneficial_correction" else row["frame_logit_delta"] > 0
        intended_margin = row["support_margin_delta"] < 0 if row["stage176_cohort"] == "beneficial_correction" else row["support_margin_delta"] > 0
        opponents = [candidate for candidate in by_pair[str(row["pair_id"])]
                     if int(candidate["gold_frame_label"]) != int(row["gold_frame_label"])]
        _require(bool(opponents), f"hard-cohort row {row['row_id']} has no opposite-frame same-pair rows")
        pair_ranking = {}
        for side in ("baseline", "pilot"):
            if int(row["gold_frame_label"]) == 1:
                correct = sum(float(row[f"{side}_frame_logit"]) > float(other[f"{side}_frame_logit"]) for other in opponents)
            else:
                correct = sum(float(other[f"{side}_frame_logit"]) > float(row[f"{side}_frame_logit"]) for other in opponents)
            pair_ranking[side] = correct / len(opponents)
        ranking_improved = pair_ranking["pilot"] > pair_ranking["baseline"]
        csv_rows.append({**row, "frame_moves_in_intended_direction": intended_frame,
                         "support_margin_moves_in_intended_direction": intended_margin,
                         "baseline_same_pair_row_ranking_accuracy": pair_ranking["baseline"],
                         "pilot_same_pair_row_ranking_accuracy": pair_ranking["pilot"],
                         "same_pair_row_ranking_delta": pair_ranking["pilot"] - pair_ranking["baseline"],
                         "frame_ranking_improves_without_final_prediction_change": ranking_improved and not row["prediction_changed"]})
    beneficial_new = sum(r["stage176_cohort"] == "beneficial_correction" and not r["baseline_correct"] and r["pilot_correct"] for r in hard)
    harmful_new = sum(r["stage176_cohort"] == "harmful_regression" and r["baseline_correct"] and not r["pilot_correct"] for r in hard)
    def summarize(cohort: str) -> dict[str, Any]:
        group = [r for r in csv_rows if r["stage176_cohort"] == cohort]
        return {"rows": len(group), "prediction_transitions": dict(sorted(Counter(r["prediction_transition"] for r in group).items())),
                "correctness_transitions": dict(sorted(Counter(r["correctness_transition"] for r in group).items())),
                "mean_support_margin_delta": _mean(float(r["support_margin_delta"]) for r in group),
                "mean_frame_logit_delta": _mean(float(r["frame_logit_delta"]) for r in group),
                "mean_frame_prob_delta": _mean(float(r["frame_prob_delta"]) for r in group),
                "by_family": {name: {"rows": sum(r["intervention_type"] == name for r in group),
                                      "prediction_transitions": dict(sorted(Counter(r["prediction_transition"] for r in group if r["intervention_type"] == name).items()))}
                              for name in sorted({r["intervention_type"] for r in group})}}
    return csv_rows, {"beneficial_correction": summarize("beneficial_correction"),
                      "harmful_regression": summarize("harmful_regression"),
                      "beneficial_rows_newly_corrected_by_pilot": beneficial_new,
                      "harmful_rows_newly_damaged_by_pilot": harmful_new,
                      "net_cohort_benefit": beneficial_new - harmful_new,
                      "rows_whose_frame_ranking_improves_without_final_prediction_change": sum(r["frame_ranking_improves_without_final_prediction_change"] for r in csv_rows),
                      "final_margin_moves_in_intended_direction": sum(r["support_margin_moves_in_intended_direction"] for r in csv_rows)}


def _frame_errors(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    for row in rows:
        gold = int(row["gold_frame_label"])
        def status(side: str) -> str:
            prediction = int(float(row[f"{side}_frame_prob"]) >= .5)
            if prediction == gold: return "correct"
            return "false_compatible" if gold == 0 else "false_incompatible"
        before, after = status("baseline"), status("pilot")
        output.append({"stable_row_index": row["stable_row_index"], "row_id": row["row_id"],
                       "pair_id": row["pair_id"], "intervention_type": row["intervention_type"],
                       "gold_frame_label": gold, "baseline_frame_status": before,
                       "pilot_frame_status": after, "frame_error_transition": f"{before}->{after}",
                       "hard_39": row["stage176_cohort"] in ("beneficial_correction", "harmful_regression"),
                       "final_prediction_transition": row["prediction_transition"],
                       "support_margin_delta": row["support_margin_delta"]})
    corrected = [r for r in output if r["baseline_frame_status"] != "correct" and r["pilot_frame_status"] == "correct"]
    introduced = [r for r in output if r["baseline_frame_status"] == "correct" and r["pilot_frame_status"] != "correct"]
    return output, {"baseline": dict(sorted(Counter(r["baseline_frame_status"] for r in output).items())),
                    "pilot": dict(sorted(Counter(r["pilot_frame_status"] for r in output).items())),
                    "corrected_frame_errors": len(corrected), "introduced_frame_errors": len(introduced),
                    "corrected_by_family": dict(sorted(Counter(r["intervention_type"] for r in corrected).items())),
                    "introduced_by_family": dict(sorted(Counter(r["intervention_type"] for r in introduced).items())),
                    "corrected_hard_39_overlap": sum(r["hard_39"] for r in corrected),
                    "introduced_hard_39_overlap": sum(r["hard_39"] for r in introduced),
                    "corrected_final_prediction_changed": sum(r["final_prediction_transition"].split("->")[0] != r["final_prediction_transition"].split("->")[1] for r in corrected),
                    "introduced_final_prediction_changed": sum(r["final_prediction_transition"].split("->")[0] != r["final_prediction_transition"].split("->")[1] for r in introduced),
                    "corrected_mean_support_margin_delta": _mean(float(r["support_margin_delta"]) for r in corrected),
                    "introduced_mean_support_margin_delta": _mean(float(r["support_margin_delta"]) for r in introduced)}


def _module(name: str) -> str:
    # Prefixes are the actual v6b_minimal module names; unmatched future or
    # optional modules intentionally remain `other` rather than being guessed.
    explicit = (("frame_head", ("frame_gate.", "frame_violation_head.")),
                ("final_classifier_head", ("decision_head.", "polarity_energy_head.")),
                ("predicate_head", ("predicate_coverage_head.", "predicate_isolation_head.")),
                ("sufficiency_head", ("sufficiency_gate.",)),
                ("entitlement_router", ("preservation_entitlement_head.", "boundary_head.")),
                ("encoder_backbone", ("mamba.",)))
    for category, prefixes in explicit:
        if any(name.startswith(prefix) for prefix in prefixes):
            return category
    return "other"


def _parameter_delta(baseline: dict[str, torch.Tensor], pilot: dict[str, torch.Tensor]
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _require(set(baseline) == set(pilot), "checkpoint state_dict parameter names differ")
    module_acc: dict[str, dict[str, float]] = defaultdict(lambda: {"tensor_count": 0, "changed": 0, "elements": 0, "delta_sq": 0.0, "base_sq": 0.0, "max_abs": 0.0})
    changed = unchanged = elements = 0; delta_sq = base_sq = max_abs = 0.0
    for name in sorted(baseline):
        left, right = baseline[name].detach().cpu(), pilot[name].detach().cpu()
        _require(left.shape == right.shape and left.dtype == right.dtype, f"parameter shape/dtype mismatch: {name}")
        delta = right.double() - left.double()
        is_changed = not torch.equal(left, right)
        norm_sq, reference_sq = float(delta.square().sum()), float(left.double().square().sum())
        local_max = float(delta.abs().max()) if delta.numel() else 0.0
        category = _module(name); acc = module_acc[category]
        acc["tensor_count"] += 1; acc["changed"] += int(is_changed); acc["elements"] += delta.numel()
        acc["delta_sq"] += norm_sq; acc["base_sq"] += reference_sq; acc["max_abs"] = max(acc["max_abs"], local_max)
        changed += int(is_changed); unchanged += int(not is_changed); elements += delta.numel()
        delta_sq += norm_sq; base_sq += reference_sq; max_abs = max(max_abs, local_max)
    modules = [{"module": name, "tensor_count": int(value["tensor_count"]),
                "changed_tensor_count": int(value["changed"]), "element_count": int(value["elements"]),
                "l2_delta": math.sqrt(value["delta_sq"]),
                "relative_l2_delta": math.sqrt(value["delta_sq"]) / math.sqrt(value["base_sq"]) if value["base_sq"] else None,
                "max_absolute_delta": value["max_abs"]} for name, value in sorted(module_acc.items())]
    summary = {"exact_shared_parameter_names": True, "tensor_count": changed + unchanged,
               "changed_tensor_count": changed, "unchanged_tensor_count": unchanged,
               "element_count": elements, "global_l2_delta": math.sqrt(delta_sq),
               "relative_l2_delta": math.sqrt(delta_sq) / math.sqrt(base_sq) if base_sq else None,
               "max_absolute_delta": max_abs,
               "interpretation": "Confirms checkpoint parameter change; it is not causal proof."}
    return [{"scope": "all_parameters", **summary}], modules, summary


def _diagnose(transitions: dict[str, Any], ranking: dict[str, Any], hard: dict[str, Any],
              cohorts: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    net = int(transitions["net_correctness_change"])
    cohort_net = int(cohorts["net_cohort_benefit"])
    harmful_support = int(transitions["harmful_support_regressions"])
    delta = float(ranking["pilot_minus_baseline"]["ranking_delta"])
    relative = ranking["pilot_minus_baseline"]["mean_gap_relative_increase"]
    relative = float(relative) if relative is not None else 0.0
    hard_delta = float(hard["pilot"]["auroc"] or 0) - float(hard["baseline"]["auroc"] or 0)
    if net < 0 or harmful_support >= 2 or delta <= -.01:
        decision, category = HARMFUL, "D_harmful_objective"
    elif (net >= 3 or cohort_net >= 3) and harmful_support <= 1 and delta >= 0:
        decision, category = CLEAN, "A_effective_clean_benefit"
    elif (delta >= .01 or relative >= .10) and -1 <= net <= 1 and cohort_net < 3 and harmful_support <= 1:
        decision, category = REPRESENTATION, "B_representation_only_change"
    elif net < 3 and cohort_net < 3 and delta < .01 and relative < .10 and hard_delta < .02:
        decision, category = REDUNDANT, "C_redundant_objective"
    else:
        decision, category = BLOCKED, "indeterminate_gate"
    evidence = {"classification": category, "baseline_direct_frame_supervision_present": True,
                "dev_pair_ranking_delta": delta, "mean_gap_relative_increase": relative,
                "hard_39_auroc_delta": hard_delta, "final_net_correctness": net,
                "stage176_cohort_net_change": cohort_net, "harmful_support_regressions": harmful_support}
    gate = {"decision": decision, "clean_benefit": {"net_correctness_at_least_3_or_cohort_net_at_least_3": net >= 3 or cohort_net >= 3, "harmful_support_regressions_at_most_1": harmful_support <= 1, "dev_pair_ranking_not_worse": delta >= 0},
            "representation_only": {"ranking_delta_at_least_0_01_or_gap_relative_at_least_0_10": delta >= .01 or relative >= .10, "net_correctness_between_minus1_plus1": -1 <= net <= 1, "cohort_net_less_than_3": cohort_net < 3},
            "redundant": {"net_correctness_less_than_3": net < 3, "cohort_net_less_than_3": cohort_net < 3, "ranking_delta_less_than_0_01": delta < .01, "gap_relative_less_than_0_10": relative < .10, "meaningful_hard39_improvement": hard_delta >= .02},
            "harmful": {"net_correctness_negative": net < 0, "harmful_support_regressions_at_least_2": harmful_support >= 2, "ranking_materially_decreases": delta <= -.01},
            "replication_or_weight_sweep_authorized": False}
    return decision, evidence, gate


def _render_markdown(report: dict[str, Any]) -> str:
    transition = report["row_transition_attribution"]
    ranking = report["pair_ranking_comparison"]["dev"]["pilot_minus_baseline"]
    cohort = report["stage176_cohort_attribution"]
    parameter = report["parameter_delta_audit"]
    return "\n".join([
        "# Stage177-E frame-pairwise pilot attribution audit", "",
        f"**Decision:** `{report['decision']}`", "", "## Final decisions", "",
        f"Changed rows: {transition['changed_rows']}; recovered errors: {transition['recovered_errors']}; introduced errors: {transition['introduced_errors']}; net correctness: {transition['net_correctness_change']}.", "",
        "## Frame head and same-pair ranking", "",
        f"Dev pair-ranking delta: {ranking['ranking_delta']:.6f}; mean-gap delta: {ranking['mean_gap_delta']:.6f}. Full-dev and fixed hard-39 native frame metrics are recorded in JSON and CSV.", "",
        "## Stage176 cohorts and frame errors", "",
        f"New beneficial corrections: {cohort['beneficial_rows_newly_corrected_by_pilot']}; new harmful damage: {cohort['harmful_rows_newly_damaged_by_pilot']}; cohort net: {cohort['net_cohort_benefit']}. Frame-error transitions and final-decision effects are emitted row by row.", "",
        "## Parameter delta", "",
        f"Changed tensors: {parameter['changed_tensor_count']}; unchanged tensors: {parameter['unchanged_tensor_count']}; global L2 delta: {parameter['global_l2_delta']:.6g}. This confirms parameter change, not causality.", "",
        "## Stage178 gate", "", f"`{report['stage178_gate']['decision']}`. No weight sweep, multi-seed run, external evaluation, threshold search, calibration, or training is authorized.", ""])


def _blocked(output_dir: Path, error: Exception, failure_stage: str,
             traceback_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = {
        "decision": BLOCKED,
        "error_type": type(error).__name__,
        "error": str(error),
        "failure_stage": failure_stage,
        "traceback": traceback_text,
    }
    report = {"stage": STAGE, **diagnostic, "scope": {"evaluation_only": True},
              "input_validation": {"status": "blocked", **diagnostic,
                                   "blocked_before_forward_when_input_validation_failed": failure_stage in {
                                       "argument_validation", "input_path_validation",
                                       "stage_report_validation", "stage176a_csv_validation",
                                       "provenance_validation", "checkpoint_validation",
                                       "deterministic_split_validation", "parameter_delta_audit",
                                   }},
              "checkpoint_contract": None, "aggregate_final_comparison": None,
              "row_transition_attribution": None, "final_logit_drift": None,
              "frame_head_comparison": None, "pair_ranking_comparison": None,
              "stage176_cohort_attribution": None, "frame_error_attribution": None,
              "parameter_delta_audit": None, "redundancy_diagnosis": None,
              "stage178_gate": {"decision": BLOCKED},
              "limitations": ["Validation failed; no attribution result is available."],
              "safety_policy": {"training": False, "optimizer": False, "backward": False,
                                "train_mode": False, "threshold_search": False,
                                "external_evaluation": False, "time_swap": False}}
    _write_json(output_dir / OUTPUTS["json"], report)
    (output_dir / OUTPUTS["md"]).write_text(
        f"# Stage177-E blocked\n\n**Decision:** `{BLOCKED}`\n\n"
        f"Failure stage: `{failure_stage}`\n\n"
        f"`{type(error).__name__}: {error}`\n\n```text\n{traceback_text}\n```\n",
        encoding="utf-8",
    )
    for key in OUTPUTS:
        if key not in ("json", "md"):
            _write_csv(output_dir / OUTPUTS[key], [])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/controlled_v5_v3_without_time_swap.jsonl"))
    parser.add_argument("--stage176a-report", type=Path, required=True)
    parser.add_argument("--stage176a-row-transitions", type=Path, required=True)
    parser.add_argument("--stage177a-report", type=Path, required=True)
    parser.add_argument("--baseline-provenance", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--pilot-provenance", type=Path, required=True)
    parser.add_argument("--pilot-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=177)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    current_stage = "argument_validation"
    try:
        _require(args.device == "cuda", "Stage177-E requires --device cuda")
        args.eval_batch_size = _require_int(args.eval_batch_size, field_name="cli.eval_batch_size")
        args.bootstrap_samples = _require_int(args.bootstrap_samples, field_name="cli.bootstrap_samples")
        args.bootstrap_seed = _require_int(args.bootstrap_seed, field_name="cli.bootstrap_seed")
        _require(args.eval_batch_size > 0 and args.bootstrap_samples > 0,
                 "eval batch size and bootstrap samples must be positive")
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        current_stage = "input_path_validation"
        names = ("data", "stage176a_report", "stage176a_row_transitions", "stage177a_report",
                 "baseline_provenance", "baseline_checkpoint", "pilot_provenance", "pilot_checkpoint")
        paths = {name: getattr(args, name).resolve() for name in names}
        for name, path in paths.items():
            _require(path.is_file(), f"{name} does not exist: {path}")
        _require(paths["baseline_checkpoint"] != paths["pilot_checkpoint"],
                 "baseline and pilot checkpoints resolve to the same path")

        current_stage = "stage_report_validation"
        stage_reports = _validate_stage_reports(_read_json(paths["stage176a_report"]),
                                                _read_json(paths["stage177a_report"]))
        current_stage = "stage176a_csv_validation"
        transitions = _validate_stage176_csv_identifiers(
            _read_csv(paths["stage176a_row_transitions"])
        )
        current_stage = "provenance_validation"
        baseline_prov, pilot_prov = _read_json(paths["baseline_provenance"]), _read_json(paths["pilot_provenance"])
        baseline_provenance = _validate_provenance("baseline", baseline_prov, paths["data"])
        pilot_provenance = _validate_provenance("pilot", pilot_prov, paths["data"])
        _require(baseline_provenance["data_sha256"] == pilot_provenance["data_sha256"], "provenance data hashes differ")
        current_stage = "checkpoint_validation"
        baseline_state, baseline_metadata, baseline_checkpoint = _load_and_validate_checkpoint(
            "baseline", paths["baseline_checkpoint"], baseline_prov, "off", 0.0)
        pilot_state, pilot_metadata, pilot_checkpoint = _load_and_validate_checkpoint(
            "pilot", paths["pilot_checkpoint"], pilot_prov, "pair_softplus", .05)
        _require(baseline_checkpoint["sha256"] != pilot_checkpoint["sha256"], "checkpoint SHA-256 values are identical")
        current_stage = "deterministic_split_validation"
        train_rows, dev_rows, split = _validate_split(paths["data"], baseline_prov)
        transitions = stage176b._validate_row_alignment(transitions, dev_rows)
        _require(Counter(stage176b._cohort(row) for row in transitions)["beneficial_correction"] == 25 and
                 Counter(stage176b._cohort(row) for row in transitions)["harmful_regression"] == 14,
                 "Stage176 transition CSV cohort counts mismatch")

        current_stage = "parameter_delta_audit"
        parameter_csv, module_csv, parameter_audit = _parameter_delta(baseline_state, pilot_state)
        input_validation = {"status": "passed", "stage_reports": stage_reports,
                            "baseline_provenance": baseline_provenance, "pilot_provenance": pilot_provenance,
                            "baseline_checkpoint": baseline_checkpoint, "pilot_checkpoint": pilot_checkpoint,
                            "split": split, "stage176_row_alignment": True,
                            "completed_before_model_construction_and_forward": True}

        device = torch.device("cuda")
        current_stage = "dev_model_forward"
        baseline_dev, pilot_dev = _evaluate(dev_rows, baseline_prov, pilot_prov, baseline_metadata,
                                            pilot_metadata, baseline_state, pilot_state, device, args.eval_batch_size)
        current_stage = "train_model_forward"
        baseline_train, pilot_train = _evaluate(train_rows, baseline_prov, pilot_prov, baseline_metadata,
                                                pilot_metadata, baseline_state, pilot_state, device, args.eval_batch_size)
        current_stage = "attribution_analysis"
        rows = _build_dev_rows(dev_rows, transitions, baseline_dev, pilot_dev)
        aggregate = {side: _classification(rows, side) for side in ("baseline", "pilot")}
        matrix_csv, transition_report = _transition_analysis(rows)
        aggregate["pilot_minus_baseline"] = {"accuracy": aggregate["pilot"]["accuracy"] - aggregate["baseline"]["accuracy"],
                                             "macro_f1": aggregate["pilot"]["macro_f1"] - aggregate["baseline"]["macro_f1"],
                                             "prediction_counts_equal": aggregate["pilot"]["prediction_counts"] == aggregate["baseline"]["prediction_counts"]}
        logit_drift = _drift_summary(rows)
        hard_rows = [row for row in rows if row["stage176_cohort"] in ("beneficial_correction", "harmful_regression")]
        _require(len(hard_rows) == 39, "hard subset must contain 39 rows")
        full_frame = {side: _frame_view(rows, side, args.bootstrap_samples, args.bootstrap_seed) for side in ("baseline", "pilot")}
        hard_frame = {side: _frame_view(hard_rows, side, args.bootstrap_samples, args.bootstrap_seed + 10, True) for side in ("baseline", "pilot")}
        frame_report = {"signal": {"logit": "output[\"frame_logit\"]", "probability": "output[\"frame_prob\"]"},
                        "full_dev": full_frame, "hard_39": hard_frame,
                        "pilot_minus_baseline": {"full_auroc": float(full_frame["pilot"]["auroc"] or 0) - float(full_frame["baseline"]["auroc"] or 0),
                                                 "hard_39_auroc": float(hard_frame["pilot"]["auroc"] or 0) - float(hard_frame["baseline"]["auroc"] or 0)}}
        frame_csv = [{"subset": subset, "view": side, **metrics[side]} for subset, metrics in (("full_dev", full_frame), ("hard_39", hard_frame)) for side in ("baseline", "pilot")]
        train_pairs = _pair_records(train_rows, baseline_train, pilot_train, "train")
        dev_pairs = _pair_records(dev_rows, baseline_dev, pilot_dev, "dev")
        ranking_report = {"train": _ranking_summary(train_pairs, args.bootstrap_samples, args.bootstrap_seed + 100),
                          "dev": _ranking_summary(dev_pairs, args.bootstrap_samples, args.bootstrap_seed + 200)}
        ranking_csv = [{"split": split_name, "view": side, **ranking_report[split_name][side]}
                       for split_name in ("train", "dev") for side in ("baseline", "pilot")]
        family_pair_csv = _family_pair_delta(train_rows, (baseline_train, pilot_train), "train") + _family_pair_delta(dev_rows, (baseline_dev, pilot_dev), "dev")
        cohort_csv, cohort_report = _cohort_attribution(rows)
        frame_error_csv, frame_error_report = _frame_errors(rows)
        decision, diagnosis, gate = _diagnose(transition_report, ranking_report["dev"], hard_frame, cohort_report)
        report = {
            "stage": STAGE, "decision": decision,
            "scope": {"data": str(paths["data"]), "seed": 174, "epochs": 20,
                      "selected_epoch": 20, "clean_controlled_data_only": True,
                      "evaluation_only": True, "single_seed": True,
                      "device": str(device), "eval_batch_size": args.eval_batch_size,
                      "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.bootstrap_seed},
            "input_validation": input_validation,
            "checkpoint_contract": {"schema": SCHEMA, "exact_parameter_name_match": True,
                                    "baseline_sha256": baseline_checkpoint["sha256"],
                                    "pilot_sha256": pilot_checkpoint["sha256"],
                                    "final_ce_source": 'output["logits"]'},
            "aggregate_final_comparison": aggregate,
            "row_transition_attribution": transition_report,
            "final_logit_drift": logit_drift,
            "frame_head_comparison": frame_report,
            "pair_ranking_comparison": ranking_report,
            "stage176_cohort_attribution": cohort_report,
            "frame_error_attribution": frame_error_report,
            "parameter_delta_audit": parameter_audit,
            "redundancy_diagnosis": diagnosis,
            "stage178_gate": gate,
            "limitations": ["Single-seed observational comparison of two selected checkpoints on controlled clean data.",
                            "Parameter deltas and output changes do not prove a causal mechanism.",
                            "Bootstrap intervals model row or pair resampling as stated, not training-seed uncertainty.",
                            "Final logits are used only for contextual attribution and were not a Stage177-C pairwise target."],
            "safety_policy": {"clean_controlled_data_only": True, "evaluation_only": True,
                              "training": False, "optimizer_created": False, "backward": False,
                              "train_mode_called": False, "threshold_search": False, "calibration": False,
                              "fitted_probe": False, "checkpoint_selection": False, "external_evaluation": False,
                              "external_labels": False, "time_swap": False, "trainer_modified": False,
                              "loss_modified": False, "weight_sweep": False, "multi_seed": False,
                              "final_logit_pairwise_objective": False}}
        current_stage = "output_serialization"
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / OUTPUTS["json"], report)
        (output_dir / OUTPUTS["md"]).write_text(_render_markdown(report), encoding="utf-8")
        _write_csv(output_dir / OUTPUTS["rows"], rows)
        _write_csv(output_dir / OUTPUTS["transitions"], matrix_csv)
        _write_csv(output_dir / OUTPUTS["frame"], frame_csv)
        _write_csv(output_dir / OUTPUTS["ranking"], ranking_csv)
        _write_csv(output_dir / OUTPUTS["pairs"], train_pairs + dev_pairs)
        _write_csv(output_dir / OUTPUTS["family_pairs"], family_pair_csv)
        _write_csv(output_dir / OUTPUTS["cohorts"], cohort_csv)
        _write_csv(output_dir / OUTPUTS["frame_errors"], frame_error_csv)
        _write_csv(output_dir / OUTPUTS["parameter_summary"], parameter_csv)
        _write_csv(output_dir / OUTPUTS["modules"], module_csv)
        print(json.dumps({"decision": decision, "output_dir": str(output_dir)}, sort_keys=True))
        return 0
    except (AuditBlocked, stage176a.ValidationBlocked, stage176b.AuditBlocked, OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        traceback_text = traceback.format_exc()
        diagnostic = {"decision": BLOCKED, "error_type": type(error).__name__,
                      "error": str(error), "failure_stage": current_stage,
                      "traceback": traceback_text}
        _blocked(output_dir, error, current_stage, traceback_text)
        print(json.dumps(diagnostic, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
