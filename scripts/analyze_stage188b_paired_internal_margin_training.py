#!/usr/bin/env python3
"""Analyze exported Stage188-B paired runs; never train or load a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DATA_SHA = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
SIDECAR_SHA = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
BLOCKED = "STAGE188B_PAIRED_INTERNAL_MARGIN_EXPERIMENT_BLOCKED"
NEGATIVE = "STAGE188B_SINGLE_SEED_MARGIN_NEGATIVE_OR_REGRESSIVE"
MIXED = "STAGE188B_SINGLE_SEED_MARGIN_MIXED_NO_REPLICATION_YET"
POSITIVE = "STAGE188B_SINGLE_SEED_MARGIN_POSITIVE_THREE_SEED_REPLICATION_CANDIDATE"
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
ALLOWED_ARG_DIFFS = {
    "compatible_positive_margin_weight",
    "compatible_positive_margin_logit",
    "controlled_integrity_sidecar_path",
    "expected_integrity_sidecar_semantic_sha256",
    "output_json",
    "output_predictions_json",
    "stage115_clean_dev_scalar_output_jsonl",
}
AGGREGATE_REQUIRED_KEYS = frozenset({
    "configured_weight",
    "configured_margin_logit",
    "compatible_positive_margin_eligible_count",
    "compatible_positive_margin_eligible_observation_count",
    "epoch_metrics",
    "score_source",
    "normalization",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage188a-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--intervention-run-dir", type=Path, required=True)
    parser.add_argument("--stage182b-dir", type=Path, required=True)
    parser.add_argument("--stage185a-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", default=DATA_SHA)
    parser.add_argument("--expected-sidecar-semantic-sha256", default=SIDECAR_SHA)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def find_unique(directory: Path, predicate) -> Path | None:
    matches = [path for path in sorted(directory.rglob("*")) if path.is_file() and predicate(path)]
    return matches[0] if len(matches) == 1 else None


def load_provenance(directory: Path) -> tuple[Path | None, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.rglob("*.json")):
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("parsed_args"), dict) and "source_provenance" in value:
            candidates.append((path, value))
    return candidates[0] if len(candidates) == 1 else (None, {})


def load_training_report(directory: Path) -> tuple[Path | None, dict[str, Any]]:
    preferred = [path for path in sorted(directory.rglob("*.json")) if "train" in path.name.lower() and "prediction" not in path.name.lower()]
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in preferred:
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and find_object_with_key(value, "best_dev_metrics") is not None:
            candidates.append((path, value))
    return candidates[0] if len(candidates) == 1 else (None, {})


def find_object_with_key(value: Any, key: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if key in value:
            return value
        for child in value.values():
            found = find_object_with_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_object_with_key(child, key)
            if found is not None:
                return found
    return None


def find_objects_with_keys(
    value: Any, required: frozenset[str], path: str = "$"
) -> list[tuple[str, dict[str, Any]]]:
    """Return every dict that directly contains all required keys, with its JSON path."""
    matches: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        if required.issubset(value):
            matches.append((path, value))
        for key, child in value.items():
            matches.extend(find_objects_with_keys(child, required, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(find_objects_with_keys(child, required, f"{path}[{index}]"))
    return matches


def extract_margin_aggregate(value: Any) -> tuple[dict[str, Any], str | None, int]:
    candidates = find_objects_with_keys(value, AGGREGATE_REQUIRED_KEYS)
    if len(candidates) != 1:
        return {}, None, len(candidates)
    path, aggregate = candidates[0]
    return aggregate, path, 1


def finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def argv_has_option(argv: Any, option: str) -> bool:
    if not isinstance(argv, list):
        return False
    flag = "--" + option.replace("_", "-")
    return any(isinstance(token, str) and (token == flag or token.startswith(flag + "=")) for token in argv)


def argv_option_value(argv: Any, option: str) -> Any:
    if not isinstance(argv, list):
        return None
    flag = "--" + option.replace("_", "-")
    for index, token in enumerate(argv):
        if token == flag:
            return argv[index + 1] if index + 1 < len(argv) else None
        if isinstance(token, str) and token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def aggregate_epoch_metrics(rows: Any) -> list[dict[str, Any]]:
    """Aggregate the trainer's per-batch epoch_metrics rows by their explicit epoch."""
    grouped: dict[Any, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return []
    for row in rows:
        if not isinstance(row, dict) or row.get("enabled") is not True or row.get("epoch") is None:
            continue
        epoch = row["epoch"]
        totals = grouped.setdefault(epoch, {
            "epoch": epoch, "eligible_count": 0, "active_count": 0,
            "hinge_loss_sum": 0.0, "eligible_frame_logit_sum": 0.0,
        })
        totals["eligible_count"] += int(row.get("eligible_count") or 0)
        totals["active_count"] += int(row.get("active_count") or 0)
        totals["hinge_loss_sum"] += float(row.get("hinge_loss_sum") or 0.0)
        totals["eligible_frame_logit_sum"] += float(row.get("eligible_frame_logit_sum") or 0.0)
    result: list[dict[str, Any]] = []
    for totals in grouped.values():
        eligible = totals["eligible_count"]
        totals["active_rate"] = totals["active_count"] / eligible if eligible else None
        totals["mean_eligible_frame_logit"] = (
            totals["eligible_frame_logit_sum"] / eligible if eligible else None
        )
        totals["compatible_positive_margin_loss_raw"] = (
            totals["hinge_loss_sum"] / eligible if eligible else None
        )
        result.append(totals)
    return result


def load_predictions(directory: Path) -> tuple[Path | None, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[tuple[Path, dict[str, Any], list[dict[str, Any]]]] = []
    for path in sorted(directory.rglob("*.json")):
        if "prediction" not in path.name.lower():
            continue
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("predictions"), list):
            rows = [row for row in value["predictions"] if isinstance(row, dict)]
            candidates.append((path, value.get("metadata") or {}, rows))
    return candidates[0] if len(candidates) == 1 else (None, {}, [])


def load_scalars(directory: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    candidates = [path for path in sorted(directory.rglob("clean_dev_scalars.jsonl")) if path.is_file()]
    if len(candidates) != 1:
        return None, []
    return candidates[0], read_jsonl(candidates[0])


def index_rows(rows: list[dict[str, Any]], artifact: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row_id(row)
        if identifier is None or identifier in result:
            raise ValueError(f"{artifact} row IDs are missing or duplicated")
        result[identifier] = row
    return result

def row_id(row: dict[str, Any]) -> str | None:
    for key in ("id", "row_id", "stable_id"):
        if row.get(key) is not None:
            return str(row[key])
    raw = row.get("raw_record")
    if isinstance(raw, dict) and raw.get("id") is not None:
        return str(raw["id"])
    return None


def gold(row: dict[str, Any]) -> str | None:
    for key in ("gold_final_label", "gold_label", "final_label"):
        if row.get(key) in LABELS:
            return str(row[key])
    raw = row.get("raw_record")
    return str(raw.get("final_label")) if isinstance(raw, dict) and raw.get("final_label") in LABELS else None


def prediction(row: dict[str, Any]) -> str | None:
    for key in ("pred_final_label", "pred_label", "prediction", "base_prediction"):
        if row.get(key) in LABELS:
            return str(row[key])
    return None


def native_frame_logit(row: dict[str, Any]) -> float | None:
    value = row.get("frame_logit")
    return float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def native_frame_prob(row: dict[str, Any]) -> float | None:
    value = row.get("frame_prob", row.get("frame_probability"))
    return float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def index_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row_id(row)
        if identifier is None or identifier in result:
            raise ValueError("prediction IDs are missing or duplicated")
        result[identifier] = row
    return result


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [(gold(row), prediction(row)) for row in rows]
    if any(g not in LABELS or p not in LABELS for g, p in pairs):
        raise ValueError("prediction rows lack canonical gold/predicted labels")
    count = len(pairs)
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in pairs:
        confusion[str(g)][str(p)] += 1
    recalls: dict[str, float | None] = {}
    f1s: list[float] = []
    for label in LABELS:
        tp = confusion[label][label]
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        fp = sum(confusion[g][label] for g in LABELS if g != label)
        recall = tp / (tp + fn) if tp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0
        recalls[label] = recall
        f1s.append(f1)
    false_ne = sum(1 for g, p in pairs if g in ("REFUTE", "SUPPORT") and p == "NOT_ENTITLED")
    false_entitlement = sum(1 for g, p in pairs if g == "NOT_ENTITLED" and p in ("REFUTE", "SUPPORT"))
    polarity = sum(1 for g, p in pairs if g in ("REFUTE", "SUPPORT") and p in ("REFUTE", "SUPPORT") and g != p)
    return {
        "row_count": count,
        "accuracy": sum(g == p for g, p in pairs) / count if count else None,
        "macro_f1": sum(f1s) / len(LABELS) if count else None,
        "support_recall": recalls["SUPPORT"],
        "refute_recall": recalls["REFUTE"],
        "not_entitled_recall": recalls["NOT_ENTITLED"],
        "prediction_counts": dict(Counter(str(p) for _, p in pairs)),
        "confusion_matrix": confusion,
        "false_not_entitled_total": false_ne,
        "false_entitlement_total": false_entitlement,
        "polarity_error_total": polarity,
    }


def value_at(mapping: dict[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def transition_category(g: str, before: str, after: str) -> str:
    before_ok, after_ok = before == g, after == g
    if before_ok and after_ok:
        return "stable_correct"
    if not before_ok and after_ok:
        return "beneficial_recovery"
    if before_ok and not after_ok:
        return "harmful_regression"
    if before == after:
        return "stable_wrong"
    return "wrong_to_different_wrong"


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    pairing_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    def check(name: str, required: Any, observed: Any, passed: bool | None, category: str = "runtime",
              reason: str = "", status: str | None = None) -> bool | None:
        gate_status = status or ("pass" if passed else "fail")
        gate_rows.append({"gate": name, "category": category, "required": json.dumps(required, sort_keys=True),
                          "observed": json.dumps(observed, sort_keys=True), "passed": passed,
                          "status": gate_status,
                          "blocking_reason": "" if gate_status != "fail" else reason})
        if category == "runtime" and gate_status == "fail":
            blockers.append(f"{name}: {reason}")
        return passed

    stage188a_report_path = args.stage188a_dir / "stage188a_paired_internal_margin_manifest_report.json"
    stage188a_report = read_json(stage188a_report_path) if stage188a_report_path.is_file() else {}
    baseline_manifest_path = args.stage188a_dir / "stage188a_baseline_manifest.json"
    baseline_manifest = read_json(baseline_manifest_path) if baseline_manifest_path.is_file() else {}
    check("stage188a_ready", "STAGE188A_PAIRED_INTERNAL_MARGIN_EXPERIMENT_SPEC_READY",
          stage188a_report.get("decision"), stage188a_report.get("decision") == "STAGE188A_PAIRED_INTERNAL_MARGIN_EXPERIMENT_SPEC_READY",
          reason="Stage188-A manifest gate was not ready")

    base_prov_path, base_prov = load_provenance(args.baseline_run_dir)
    int_prov_path, int_prov = load_provenance(args.intervention_run_dir)
    base_report_path, base_report_root = load_training_report(args.baseline_run_dir)
    int_report_path, int_report_root = load_training_report(args.intervention_run_dir)
    base_pred_path, base_meta, base_rows = load_predictions(args.baseline_run_dir)
    int_pred_path, int_meta, int_rows = load_predictions(args.intervention_run_dir)
    base_scalar_path, base_scalar_rows = load_scalars(args.baseline_run_dir)
    int_scalar_path, int_scalar_rows = load_scalars(args.intervention_run_dir)
    for arm, prov_path, report_path, pred_path, scalar_path in (
        ("baseline", base_prov_path, base_report_path, base_pred_path, base_scalar_path),
        ("intervention", int_prov_path, int_report_path, int_pred_path, int_scalar_path),
    ):
        check(f"{arm}_artifacts", "unique provenance/report/predictions/clean_dev_scalars",
              {"provenance": str(prov_path), "report": str(report_path), "predictions": str(pred_path),
               "scalars": str(scalar_path)},
              all(path is not None for path in (prov_path, report_path, pred_path, scalar_path)),
              reason="required exported artifact missing or ambiguous")

    base_run = find_object_with_key(base_report_root, "best_dev_metrics") or {}
    int_run = find_object_with_key(int_report_root, "best_dev_metrics") or {}
    base_margin, base_margin_path, base_margin_candidate_count = extract_margin_aggregate(base_report_root)
    int_margin, int_margin_path, int_margin_candidate_count = extract_margin_aggregate(int_report_root)
    check("baseline_unique_margin_aggregate", 1, base_margin_candidate_count,
          base_margin_candidate_count == 1, reason="completed-run margin aggregate is missing or ambiguous")
    check("intervention_unique_margin_aggregate", 1, int_margin_candidate_count,
          int_margin_candidate_count == 1, reason="completed-run margin aggregate is missing or ambiguous")

    base_args = base_prov.get("parsed_args") or {}
    int_args = int_prov.get("parsed_args") or {}
    base_commit = value_at(base_prov, "source_provenance.git_commit")
    int_commit = value_at(int_prov, "source_provenance.git_commit")
    manifest_commit = stage188a_report.get("current_git_commit")
    expected_trainer_sha = stage188a_report.get("trainer_sha256")
    manifest_trainer_path = Path(str(stage188a_report.get("trainer_path") or ""))
    trainer_path = args.repo_root / "scripts" / "train_controlled_v6b_minimal.py"
    observed_trainer_sha = sha256_file(trainer_path)
    runtime_trainer_names = [
        Path(str(provenance.get("training_script") or "")).name
        for provenance in (base_prov, int_prov)
    ]
    runtime_commit_matches_manifest = (
        bool(manifest_commit) and base_commit == int_commit == manifest_commit
    )
    trainer_identity_matches_manifest = (
        manifest_trainer_path.name == trainer_path.name
        and runtime_trainer_names == [trainer_path.name, trainer_path.name]
        and bool(expected_trainer_sha) and observed_trainer_sha == expected_trainer_sha
    )
    same_fields = ["data", "seed", "architecture", "backbone", "model_name", "device", "epochs", "lr", "head_lr",
                   "encoder_lr", "dev_ratio", "select_metric", "train_batch_size", "eval_batch_size",
                   "gradient_accumulation_steps", "fp16", "weighted_label_loss", "balanced_sampler",
                   "stage174c_clean_pairwise_mode", "stage174c_clean_pairwise_weight",
                   "stage175b_support_anchor_mode", "stage175b_support_anchor_weight",
                   "stage177c_frame_pairwise_mode", "stage177c_frame_pairwise_weight"]
    check("same_git_commit", "equal non-empty", [base_commit, int_commit], bool(base_commit) and base_commit == int_commit,
          reason="run commits differ or are absent")
    check("stage188a_runtime_git_commit", manifest_commit, [base_commit, int_commit],
          runtime_commit_matches_manifest,
          reason="runtime commit does not exactly match the Stage188-A manifest")
    check("stage188a_trainer_sha", expected_trainer_sha, observed_trainer_sha,
          trainer_identity_matches_manifest,
          reason="current trainer bytes do not exactly match the Stage188-A manifest")
    for field in same_fields:
        passed = base_args.get(field) == int_args.get(field)
        pairing_rows.append({"field": field, "baseline": json.dumps(base_args.get(field)),
                             "intervention": json.dumps(int_args.get(field)), "passed": passed,
                             "classification": "common"})
        check(f"paired_{field}", "equal", [base_args.get(field), int_args.get(field)], passed,
              reason="paired common configuration differs")
    all_arg_diffs = {key for key in set(base_args) | set(int_args) if base_args.get(key) != int_args.get(key)}
    forbidden_arg_diffs = sorted(all_arg_diffs - ALLOWED_ARG_DIFFS)
    check("exact_allowed_argument_diff", [], forbidden_arg_diffs, not forbidden_arg_diffs,
          reason="forbidden parsed-argument differences present")

    check("dataset_sha_pair", args.expected_dataset_sha256,
          [value_at(base_prov, "data_provenance.main_data.sha256"), value_at(int_prov, "data_provenance.main_data.sha256")],
          value_at(base_prov, "data_provenance.main_data.sha256") == args.expected_dataset_sha256
          and value_at(int_prov, "data_provenance.main_data.sha256") == args.expected_dataset_sha256,
          reason="authoritative dataset SHA missing or mismatched")
    base_runtime_margin = base_prov.get("compatible_positive_margin") or {}
    base_runtime_argv = base_prov.get("raw_sys_argv")
    base_manifest_argv = baseline_manifest.get("argv")
    base_weight = base_runtime_margin.get("weight", base_runtime_margin.get("configured_weight"))
    base_sidecar_evidence = {
        "manifest_weight": (baseline_manifest.get("arm_configuration") or {}).get("compatible_positive_margin_weight"),
        "manifest_argv_weight": argv_option_value(base_manifest_argv, "compatible_positive_margin_weight"),
        "runtime_argv_weight": argv_option_value(base_runtime_argv, "compatible_positive_margin_weight"),
        "runtime_parsed_weight": base_args.get("compatible_positive_margin_weight"),
        "manifest_sidecar_option_absent": isinstance(base_manifest_argv, list) and not argv_has_option(base_manifest_argv, "controlled_integrity_sidecar_path"),
        "manifest_expected_sha_option_absent": isinstance(base_manifest_argv, list) and not argv_has_option(base_manifest_argv, "expected_integrity_sidecar_semantic_sha256"),
        "runtime_sidecar_option_absent": isinstance(base_runtime_argv, list) and not argv_has_option(base_runtime_argv, "controlled_integrity_sidecar_path"),
        "runtime_expected_sha_option_absent": isinstance(base_runtime_argv, list) and not argv_has_option(base_runtime_argv, "expected_integrity_sidecar_semantic_sha256"),
        "runtime_enabled": base_runtime_margin.get("enabled"),
        "runtime_weight": base_weight,
        "runtime_sidecar_path": base_runtime_margin.get("sidecar_path"),
        "runtime_expected_sidecar_semantic_sha256": base_runtime_margin.get("expected_sidecar_semantic_sha256"),
        "aggregate_enabled": base_margin.get("enabled"),
        "aggregate_configured_weight": base_margin.get("configured_weight"),
        "aggregate_sidecar_accessed": (base_margin.get("sidecar_contract") or {}).get("sidecar_accessed"),
        "runtime_commit_matches_stage188a": runtime_commit_matches_manifest,
        "trainer_identity_matches_stage188a": trainer_identity_matches_manifest,
    }
    baseline_sidecar_not_accessed = (
        str(base_sidecar_evidence["manifest_weight"]) == "0.0"
        and str(base_sidecar_evidence["manifest_argv_weight"]) == "0.0"
        and str(base_sidecar_evidence["runtime_argv_weight"]) == "0.0"
        and base_args.get("compatible_positive_margin_weight") == 0.0
        and all(base_sidecar_evidence[key] for key in (
            "manifest_sidecar_option_absent", "manifest_expected_sha_option_absent",
            "runtime_sidecar_option_absent", "runtime_expected_sha_option_absent"))
        and base_runtime_margin.get("enabled") is False
        and base_weight == 0.0
        and base_runtime_margin.get("sidecar_path") is None
        and base_runtime_margin.get("expected_sidecar_semantic_sha256") is None
        and base_margin.get("enabled") is False
        and base_margin.get("configured_weight") == 0.0
        and (base_margin.get("sidecar_contract") or {}).get("sidecar_accessed") in (None, False)
        and runtime_commit_matches_manifest
        and trainer_identity_matches_manifest
    )
    check("baseline_margin_disabled", 0.0, base_sidecar_evidence,
          baseline_sidecar_not_accessed, reason="baseline default-off runtime contract is not proven")
    check("baseline_sidecar_not_accessed", True, base_sidecar_evidence,
          baseline_sidecar_not_accessed, reason="baseline sidecar access cannot be excluded")
    check("intervention_margin_weight", 0.05, int_args.get("compatible_positive_margin_weight"),
          int_args.get("compatible_positive_margin_weight") == 0.05 and int_margin.get("enabled") is True,
          reason="intervention margin is not active at weight 0.05")
    check("intervention_margin_logit", 0.0, int_args.get("compatible_positive_margin_logit"),
          int_args.get("compatible_positive_margin_logit") == 0.0, reason="intervention margin logit differs")
    observed_sidecar_sha = (int_margin.get("sidecar_contract") or {}).get("observed_sidecar_semantic_sha256")
    check("intervention_sidecar_sha", args.expected_sidecar_semantic_sha256, observed_sidecar_sha,
          observed_sidecar_sha == args.expected_sidecar_semantic_sha256, reason="intervention sidecar semantic SHA missing or mismatched")
    int_sidecar_contract = int_margin.get("sidecar_contract") or {}
    for name, expected in (("eligible_rows", 605), ("eligible_pairs", 121), ("eligible_families", 5)):
        check(f"intervention_sidecar_{name}", expected, int_sidecar_contract.get(name),
              int_sidecar_contract.get(name) == expected, reason=f"intervention sidecar {name} mismatch")
    for arm, provenance in (("baseline", base_prov), ("intervention", int_prov)):
        policy = provenance.get("training_selection_policy") or {}
        safe = all(policy.get(key) is False for key in (
            "external_evaluation_used_for_training", "external_evaluation_used_for_calibration",
            "external_evaluation_used_for_threshold_selection", "external_evaluation_used_for_checkpoint_selection"))
        check(f"{arm}_no_external_use", True, safe, safe, reason="external train/calibration/selection exclusion not proven")
        completed = value_at(provenance, "finalization.completed_epochs")
        expected_epochs = (provenance.get("parsed_args") or {}).get("epochs")
        check(f"{arm}_completed", expected_epochs, completed, completed == expected_epochs,
              reason="successful full epoch completion not proven")

    try:
        base_index = index_rows(base_rows, "baseline predictions")
        int_index = index_rows(int_rows, "intervention predictions")
        base_scalar_index = index_rows(base_scalar_rows, "baseline scalars")
        int_scalar_index = index_rows(int_scalar_rows, "intervention scalars")
    except ValueError as exc:
        blockers.append(str(exc))
        base_index, int_index, base_scalar_index, int_scalar_index = {}, {}, {}, {}
    paired_ids = set(base_index)
    exact_sets = (
        bool(paired_ids)
        and paired_ids == set(int_index)
        and paired_ids == set(base_scalar_index)
        and paired_ids == set(int_scalar_index)
    )
    check("identical_clean_dev_ids", "exact identical non-empty prediction/scalar sets for both arms",
          [len(base_index), len(int_index), len(base_scalar_index), len(int_scalar_index)], exact_sets,
          reason="prediction/scalar clean-dev row-ID sets differ")

    transitions: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    for identifier in sorted(paired_ids & set(int_index) & set(base_scalar_index) & set(int_scalar_index)):
        left, right = base_index[identifier], int_index[identifier]
        left_scalar, right_scalar = base_scalar_index[identifier], int_scalar_index[identifier]
        g, before, after = gold(left), prediction(left), prediction(right)
        identities_match = (
            g == gold(right) == gold(left_scalar) == gold(right_scalar)
            and before == prediction(left_scalar)
            and after == prediction(right_scalar)
        )
        if not identities_match:
            blockers.append(f"prediction/scalar gold or prediction identity mismatch for {identifier}")
            continue
        category = transition_category(str(g), str(before), str(after))
        transitions.append({"row_id": identifier, "gold": g, "baseline_prediction": before,
                            "intervention_prediction": after, "baseline_correct": before == g,
                            "intervention_correct": after == g, "transition_category": category,
                            "class_transition": f"{before}->{after}"})
        left_logit, right_logit = native_frame_logit(left_scalar), native_frame_logit(right_scalar)
        left_prob, right_prob = native_frame_prob(left_scalar), native_frame_prob(right_scalar)
        native_rows.append({"row_id": identifier, "gold": g, "baseline_prediction": before,
                            "intervention_prediction": after,
                            "frame_compatible_label": left.get("frame_compatible_label"),
                            "baseline_frame_logit": left_logit, "intervention_frame_logit": right_logit,
                            "frame_logit_delta": right_logit - left_logit if left_logit is not None and right_logit is not None else None,
                            "baseline_frame_prob": left_prob, "intervention_frame_prob": right_prob,
                            "score_source": 'direct scalar export from output["frame_logit"]' if left_logit is not None and right_logit is not None else "unavailable"})

    direct_frame_complete = bool(native_rows) and len(native_rows) == len(paired_ids) and all(
        row["frame_logit_delta"] is not None
        and row["baseline_frame_prob"] is not None
        and row["intervention_frame_prob"] is not None
        for row in native_rows
    )
    check("direct_native_frame_logits", "direct per-row output[frame_logit] for both arms", direct_frame_complete,
          direct_frame_complete, reason="row-level native frame_logit absent; final-classifier logits are never substituted")

    try:
        base_metrics, int_metrics = metric_summary(base_rows), metric_summary(int_rows)
    except ValueError as exc:
        blockers.append(str(exc))
        base_metrics, int_metrics = {}, {}
    metric_names = ["accuracy", "macro_f1", "support_recall", "refute_recall", "not_entitled_recall",
                    "false_not_entitled_total", "false_entitlement_total", "polarity_error_total"]
    metric_rows = [{"metric": name, "baseline": base_metrics.get(name), "intervention": int_metrics.get(name),
                    "absolute_delta": (int_metrics[name] - base_metrics[name]) if base_metrics.get(name) is not None and int_metrics.get(name) is not None else None}
                   for name in metric_names]
    for label in LABELS:
        metric_rows.append({"metric": f"prediction_count_{label}",
                            "baseline": (base_metrics.get("prediction_counts") or {}).get(label, 0),
                            "intervention": (int_metrics.get("prediction_counts") or {}).get(label, 0),
                            "absolute_delta": (int_metrics.get("prediction_counts") or {}).get(label, 0) - (base_metrics.get("prediction_counts") or {}).get(label, 0)})

    cohort_path = args.stage182b_dir / "stage182b_candidate_localization.csv"
    control_path = args.stage182b_dir / "stage182b_matched_control_pairs.csv"
    cohort_source = read_csv(cohort_path) if cohort_path.is_file() else []
    controls_source = read_csv(control_path) if control_path.is_file() else []
    compatible_ids = {row["row_id"] for row in cohort_source if row.get("native_error_direction") == "compatible_false_negative"}
    incompatible_ids = {row["row_id"] for row in cohort_source if row.get("native_error_direction") == "incompatible_false_positive"}
    control_ids = {row["control_row_id"] for row in controls_source if row.get("control_row_id")}
    check("stage182b_exact_topology", "13 compatible FN / 1 incompatible FP",
          [len(compatible_ids), len(incompatible_ids)], len(compatible_ids) == 13 and len(incompatible_ids) == 1,
          reason="Stage182-B cohort identity/topology mismatch")

    native_by_id = {row["row_id"]: row for row in native_rows}
    cohort_rows: list[dict[str, Any]] = []
    for cohort_name, identifiers in (("compatible_false_negative", compatible_ids),
                                     ("incompatible_false_positive", incompatible_ids),
                                     ("matched_controls", control_ids),
                                     ("clean_model_failure", compatible_ids | incompatible_ids)):
        selected = [native_by_id[item] for item in sorted(identifiers) if item in native_by_id]
        deltas = [float(row["frame_logit_delta"]) for row in selected if row.get("frame_logit_delta") is not None]
        corrected = sum(row["baseline_prediction"] != row["gold"] and row["intervention_prediction"] == row["gold"] for row in selected)
        harmed = sum(row["baseline_prediction"] == row["gold"] and row["intervention_prediction"] != row["gold"] for row in selected)
        cohort_rows.append({"cohort": cohort_name, "expected_rows": len(identifiers), "matched_rows": len(selected),
                            "mean_frame_logit_delta": statistics.fmean(deltas) if deltas else None,
                            "median_frame_logit_delta": statistics.median(deltas) if deltas else None,
                            "positive_delta_count": sum(delta > 0 for delta in deltas),
                            "corrected_final_predictions": corrected, "newly_harmed_predictions": harmed,
                            "diagnostic_status": "prior_evidence_selected_not_independent_evaluation"})
    compatible_row = next((row for row in cohort_rows if row["cohort"] == "compatible_false_negative"), {})
    incompatible_selected = [native_by_id[item] for item in incompatible_ids if item in native_by_id]
    compatible_selected = [native_by_id[item] for item in compatible_ids if item in native_by_id]
    compatible_fn_before = sum(row.get("baseline_frame_logit") is not None and row["baseline_frame_logit"] < 0 for row in compatible_selected)
    compatible_fn_after = sum(row.get("intervention_frame_logit") is not None and row["intervention_frame_logit"] < 0 for row in compatible_selected)
    incompatible_fp_before = sum(row["baseline_prediction"] != row["gold"] for row in incompatible_selected)
    incompatible_fp_after = sum(row["intervention_prediction"] != row["gold"] for row in incompatible_selected)
    incompatible_cohort_row = next(
        (row for row in cohort_rows if row["cohort"] == "incompatible_false_positive"), {}
    )
    incompatible_transition_counts_agree = (
        incompatible_fp_before - incompatible_fp_after
        == incompatible_cohort_row.get("corrected_final_predictions", 0)
        - incompatible_cohort_row.get("newly_harmed_predictions", 0)
    )
    check("incompatible_fp_transition_count_consistency", True,
          {"fp_before": incompatible_fp_before, "fp_after": incompatible_fp_after,
           "corrected_final_predictions": incompatible_cohort_row.get("corrected_final_predictions"),
           "newly_harmed_predictions": incompatible_cohort_row.get("newly_harmed_predictions")},
          bool(incompatible_selected) and incompatible_transition_counts_agree,
          reason="final-prediction FP counts disagree with corrected/harmed transition counts")

    mechanism_keys = ["configured_weight", "configured_margin_logit", "compatible_positive_margin_eligible_count",
                      "compatible_positive_margin_eligible_observation_count", "compatible_positive_margin_loss_raw",
                      "compatible_positive_margin_loss_weighted", "compatible_positive_margin_active_count",
                      "compatible_positive_margin_active_rate", "compatible_positive_margin_mean_eligible_frame_logit",
                      "zero_eligible_batch_count", "score_source", "normalization"]
    mechanism_rows = [{"metric": key, "baseline": base_margin.get(key), "intervention": int_margin.get(key)} for key in mechanism_keys]
    baseline_disabled_aggregate = (
        base_margin.get("enabled") is False
        and base_margin.get("configured_weight") == 0.0
        and base_margin.get("configured_margin_logit") == 0.0
        and base_margin.get("compatible_positive_margin_eligible_count") == 0
        and base_margin.get("compatible_positive_margin_eligible_observation_count") == 0
        and base_margin.get("compatible_positive_margin_loss_raw") in (None, 0, 0.0)
        and base_margin.get("compatible_positive_margin_loss_weighted") in (None, 0, 0.0)
    )
    check("baseline_disabled_aggregate_semantics", True,
          {key: base_margin.get(key) for key in mechanism_keys}, baseline_disabled_aggregate,
          reason="baseline aggregate does not have disabled/default-off semantics")
    check("intervention_aggregate_configuration", {"enabled": True, "weight": 0.05, "margin": 0.0},
          {"enabled": int_margin.get("enabled"), "weight": int_margin.get("configured_weight"),
           "margin": int_margin.get("configured_margin_logit")},
          int_margin.get("enabled") is True and int_margin.get("configured_weight") == 0.05
          and int_margin.get("configured_margin_logit") == 0.0,
          reason="intervention aggregate configuration mismatch")
    check("intervention_eligible_count", 605, int_margin.get("compatible_positive_margin_eligible_count"),
          int_margin.get("compatible_positive_margin_eligible_count") == 605, reason="eligible training cohort count mismatch")
    check("intervention_eligible_observations", "> 0", int_margin.get("compatible_positive_margin_eligible_observation_count"),
          finite_number(int_margin.get("compatible_positive_margin_eligible_observation_count"))
          and int_margin["compatible_positive_margin_eligible_observation_count"] > 0,
          reason="eligible observation count is missing or non-positive")
    for key in ("compatible_positive_margin_loss_raw", "compatible_positive_margin_loss_weighted",
                "compatible_positive_margin_mean_eligible_frame_logit"):
        check(f"intervention_finite_{key}", "finite", int_margin.get(key), finite_number(int_margin.get(key)),
              reason=f"{key} is absent or non-finite")
    active_count = int_margin.get("compatible_positive_margin_active_count")
    active_rate = int_margin.get("compatible_positive_margin_active_rate")
    zero_eligible_batches = int_margin.get("zero_eligible_batch_count")
    check("intervention_active_count", ">= 0", active_count,
          finite_number(active_count) and active_count >= 0, reason="active count is absent or negative")
    check("intervention_active_rate", "[0, 1]", active_rate,
          finite_number(active_rate) and 0 <= active_rate <= 1, reason="active rate is absent or outside [0, 1]")
    check("intervention_zero_eligible_batch_count", ">= 0", zero_eligible_batches,
          finite_number(zero_eligible_batches) and zero_eligible_batches >= 0,
          reason="zero-eligible batch count is absent or negative")
    check("intervention_score_source", 'output["frame_logit"]', int_margin.get("score_source"),
          int_margin.get("score_source") == 'output["frame_logit"]', reason="native score source mismatch")
    check("intervention_normalization", "eligible_row_mean", int_margin.get("normalization"),
          int_margin.get("normalization") == "eligible_row_mean", reason="normalization mismatch")
    check("intervention_checkpoint_selection_unchanged", True, int_margin.get("checkpoint_selection_unchanged"),
          int_margin.get("checkpoint_selection_unchanged") is True,
          reason="checkpoint-selection invariance is not proven")
    check("intervention_epoch_metrics", "non-empty", int_margin.get("epoch_metrics"),
          isinstance(int_margin.get("epoch_metrics"), list) and bool(int_margin["epoch_metrics"]),
          reason="intervention epoch mechanism metrics are absent")

    clean_guards = {
        "macro_f1_guard": int_metrics.get("macro_f1") is not None and int_metrics["macro_f1"] >= base_metrics["macro_f1"] - 0.01,
        "accuracy_guard": int_metrics.get("accuracy") is not None and int_metrics["accuracy"] >= base_metrics["accuracy"] - 0.01,
        "support_recall_guard": int_metrics.get("support_recall") is not None and int_metrics["support_recall"] >= base_metrics["support_recall"] - 0.02,
        "polarity_error_guard": int_metrics.get("polarity_error_total") is not None and int_metrics["polarity_error_total"] <= base_metrics["polarity_error_total"],
        "false_entitlement_guard": int_metrics.get("false_entitlement_total") is not None and int_metrics["false_entitlement_total"] <= base_metrics["false_entitlement_total"] + 2,
        "prediction_collapse_guard": all((int_metrics.get("prediction_counts") or {}).get(label, 0) > 0 for label in LABELS),
    } if base_metrics and int_metrics else {}
    epoch_metrics = aggregate_epoch_metrics(int_margin.get("epoch_metrics"))
    selected_epoch = int_run.get("best_epoch")
    first_epoch_metric = epoch_metrics[0] if epoch_metrics else None
    selected_epoch_metric = next(
        (row for row in epoch_metrics if row.get("epoch") == selected_epoch), None
    )
    final_epoch_metric = epoch_metrics[-1] if epoch_metrics else None
    comparison_epoch_metric = selected_epoch_metric or final_epoch_metric
    comparison_epoch_reference = "selected" if selected_epoch_metric is not None else "final"
    check("intervention_enabled_epoch_metrics", "non-empty enabled epoch aggregates", len(epoch_metrics),
          bool(epoch_metrics), reason="enabled epoch mechanism metrics are absent")
    check("intervention_selected_epoch_metric", selected_epoch,
          selected_epoch_metric.get("epoch") if selected_epoch_metric else None,
          selected_epoch_metric is not None,
          reason="actual selected/best epoch is absent from mechanism metrics")
    for label, row in (("first", first_epoch_metric), ("selected", selected_epoch_metric),
                       ("final", final_epoch_metric)):
        values = {
            "active_rate": row.get("active_rate") if row else None,
            "mean_eligible_frame_logit": row.get("mean_eligible_frame_logit") if row else None,
            "raw_margin_loss": row.get("compatible_positive_margin_loss_raw") if row else None,
            "active_count": row.get("active_count") if row else None,
        }
        check(f"intervention_{label}_epoch_mechanism_values", "finite", values,
              row is not None and all(finite_number(value) for value in values.values()),
              reason=f"{label} epoch mechanism values are absent or non-finite")

    def epoch_diagnostic(label: str, row: dict[str, Any] | None) -> dict[str, Any]:
        row = row or {}
        return {
            "reference": label,
            "epoch": row.get("epoch"),
            "active_rate": row.get("active_rate"),
            "mean_eligible_frame_logit": row.get("mean_eligible_frame_logit"),
            "raw_margin_loss": row.get("compatible_positive_margin_loss_raw"),
            "active_count": row.get("active_count"),
        }

    epoch_mechanism_diagnostics = {
        "selected_epoch_from_training_report": selected_epoch,
        "active_rate_comparison_reference": comparison_epoch_reference,
        "first": epoch_diagnostic("first", first_epoch_metric),
        "selected": epoch_diagnostic("selected", selected_epoch_metric),
        "final": epoch_diagnostic("final", final_epoch_metric),
    }
    for label in ("first", "selected", "final"):
        for metric, value in epoch_mechanism_diagnostics[label].items():
            if metric != "reference":
                mechanism_rows.append({"metric": f"epoch_{label}_{metric}", "baseline": None, "intervention": value})

    baseline_reference = None
    baseline_reference_available = False
    baseline_reference_reason = "baseline default-off contract forbids sidecar access"
    active_rate_decreases = (
        first_epoch_metric is not None and comparison_epoch_metric is not None
        and finite_number(first_epoch_metric.get("active_rate"))
        and finite_number(comparison_epoch_metric.get("active_rate"))
        and comparison_epoch_metric["active_rate"] < first_epoch_metric["active_rate"]
    )
    mechanism_guards: dict[str, bool | None] = {
        "eligible_train_mean_higher_than_baseline_reference": None,
        "active_rate_decreases": active_rate_decreases,
        "stage182_compatible_fn_median_delta_positive": compatible_row.get("median_frame_logit_delta") is not None and compatible_row["median_frame_logit_delta"] > 0,
        "stage182_at_least_9_positive": compatible_row.get("positive_delta_count", 0) >= 9,
        "incompatible_fp_not_increase": bool(incompatible_selected) and incompatible_fp_after <= incompatible_fp_before,
    }
    mechanism_gate_status = {
        name: ("not_evaluable_by_design" if passed is None else "pass" if passed else "fail")
        for name, passed in mechanism_guards.items()
    }
    for name, passed in clean_guards.items():
        check(name, True, passed, passed, category="clean_guardrail", reason="precommitted clean guardrail failed")
    for name, passed in mechanism_guards.items():
        status = mechanism_gate_status[name]
        check(name, True, passed, passed, category="mechanism",
              reason="precommitted mechanism-direction check failed", status=status)

    clear_regression = (
        (compatible_row.get("median_frame_logit_delta") is not None
         and compatible_row["median_frame_logit_delta"] <= 0)
        or (bool(incompatible_selected) and incompatible_fp_after > incompatible_fp_before)
    )
    if blockers:
        decision = BLOCKED
    elif clean_guards and not all(clean_guards.values()):
        decision = NEGATIVE
    elif clear_regression:
        decision = NEGATIVE
    elif (mechanism_guards and all(clean_guards.values())
          and all(status == "pass" for status in mechanism_gate_status.values())):
        decision = POSITIVE
    else:
        decision = MIXED

    confusion_rows: list[dict[str, Any]] = []
    for arm, metrics in (("baseline", base_metrics), ("intervention", int_metrics)):
        for gold_label in LABELS:
            for pred_label in LABELS:
                confusion_rows.append({"arm": arm, "gold": gold_label, "prediction": pred_label,
                                       "count": ((metrics.get("confusion_matrix") or {}).get(gold_label) or {}).get(pred_label)})
    summary = {
        "stage": "Stage188-B", "decision": decision,
        "replication_authorized": decision == POSITIVE,
        "authorized_next": "STAGE189_THREE_SEED_COMPATIBLE_POSITIVE_MARGIN_REPLICATION" if decision == POSITIVE else None,
        "single_seed_policy": "diagnostic_only_not_conclusive", "external_evaluation_authorized": False,
        "blocking_reasons": blockers, "baseline_metrics": base_metrics, "intervention_metrics": int_metrics,
        "clean_guardrails": clean_guards, "mechanism_direction": mechanism_guards,
        "mechanism_gate_status": mechanism_gate_status,
        "clear_regression": clear_regression,
        "margin_aggregate_selection": {
            "baseline": {"path": base_margin_path, "candidate_count": base_margin_candidate_count},
            "intervention": {"path": int_margin_path, "candidate_count": int_margin_candidate_count},
            "mechanism_extraction_source_path": int_margin_path,
            "required_keys": sorted(AGGREGATE_REQUIRED_KEYS),
        },
        "baseline_sidecar_non_access": {
            "proven": baseline_sidecar_not_accessed,
            "evidence": base_sidecar_evidence,
        },
        "baseline_eligible_reference": {
            "baseline_train_eligible_reference_available": baseline_reference_available,
            "value": baseline_reference,
            "reason": baseline_reference_reason,
            "gate_status": "not_evaluable_by_design",
        },
        "epoch_mechanism_diagnostics": epoch_mechanism_diagnostics,
        "prediction_transition_counts": dict(Counter(row["transition_category"] for row in transitions)),
        "class_transition_counts": dict(Counter(row["class_transition"] for row in transitions)),
        "stage182b": {"compatible_fn_before": compatible_fn_before, "compatible_fn_after": compatible_fn_after,
                      "incompatible_fp_before": incompatible_fp_before, "incompatible_fp_after": incompatible_fp_after,
                      "incompatible_fp_gate_status": mechanism_gate_status["incompatible_fp_not_increase"],
                      "incompatible_fp_count_source": "final class predictions",
                      "incompatible_transition_counts_agree": incompatible_transition_counts_agree,
                      "cohorts": cohort_rows, "independent_evaluation": False},
        "mechanism": int_margin,
        "artifact_paths": {"baseline_provenance": str(base_prov_path), "intervention_provenance": str(int_prov_path),
                           "baseline_report": str(base_report_path), "intervention_report": str(int_report_path),
                           "baseline_predictions": str(base_pred_path), "intervention_predictions": str(int_pred_path)},
    }
    write_json(output / "stage188b_paired_internal_margin_analysis_report.json", summary)
    write_csv(output / "stage188b_identity_pairing_audit.csv", ["field", "baseline", "intervention", "passed", "classification"], pairing_rows)
    write_csv(output / "stage188b_clean_dev_metrics.csv", ["metric", "baseline", "intervention", "absolute_delta"], metric_rows)
    write_csv(output / "stage188b_confusion_matrix.csv", ["arm", "gold", "prediction", "count"], confusion_rows)
    write_csv(output / "stage188b_prediction_transitions.csv",
              ["row_id", "gold", "baseline_prediction", "intervention_prediction", "baseline_correct", "intervention_correct", "transition_category", "class_transition"], transitions)
    write_csv(output / "stage188b_native_frame_diagnostics.csv",
              ["row_id", "gold", "baseline_prediction", "intervention_prediction", "frame_compatible_label", "baseline_frame_logit", "intervention_frame_logit", "frame_logit_delta", "baseline_frame_prob", "intervention_frame_prob", "score_source"], native_rows)
    write_csv(output / "stage188b_stage182b_cohort_diagnostics.csv",
              ["cohort", "expected_rows", "matched_rows", "mean_frame_logit_delta", "median_frame_logit_delta", "positive_delta_count", "corrected_final_predictions", "newly_harmed_predictions", "diagnostic_status"], cohort_rows)
    write_csv(output / "stage188b_training_mechanism_diagnostics.csv", ["metric", "baseline", "intervention"], mechanism_rows)
    write_csv(output / "stage188b_precommitted_gate.csv",
              ["gate", "category", "required", "observed", "passed", "status", "blocking_reason"], gate_rows)
    markdown = f"""# Stage188-B paired internal margin analysis

**Decision:** `{decision}`

This is a single-seed internal diagnostic, not conclusive evidence. The Stage182-B cohort was selected from prior evidence and is not independent evaluation.

## Clean-dev result

- Baseline accuracy / macro-F1 / SUPPORT recall: `{base_metrics.get('accuracy')}` / `{base_metrics.get('macro_f1')}` / `{base_metrics.get('support_recall')}`
- Intervention accuracy / macro-F1 / SUPPORT recall: `{int_metrics.get('accuracy')}` / `{int_metrics.get('macro_f1')}` / `{int_metrics.get('support_recall')}`
- Beneficial recoveries: `{sum(row['transition_category'] == 'beneficial_recovery' for row in transitions)}`
- Harmful regressions: `{sum(row['transition_category'] == 'harmful_regression' for row in transitions)}`

## Stage182-B diagnostic

- Compatible FN before / after: `{compatible_fn_before}` / `{compatible_fn_after}`
- Incompatible FP before / after: `{incompatible_fp_before}` / `{incompatible_fp_after}`
- Incompatible-FP source / gate: final class predictions / `{mechanism_gate_status['incompatible_fp_not_increase']}`
- Compatible-FN median frame-logit delta: `{compatible_row.get('median_frame_logit_delta')}`
- Compatible-FN positive deltas: `{compatible_row.get('positive_delta_count')}` of 13

## Training mechanism extraction

- Baseline / intervention aggregate paths: `{base_margin_path}` / `{int_margin_path}`
- Baseline sidecar non-access proven: `{baseline_sidecar_not_accessed}`
- Baseline eligible train reference: `not_evaluable_by_design` ({baseline_reference_reason})
- Active-rate comparison: first -> `{comparison_epoch_reference}`

## Blocking reasons

{chr(10).join('- ' + item for item in blockers) if blockers else '- None.'}

Only `{POSITIVE}` authorizes Stage189 three-seed replication. External evaluation remains unauthorized.
"""
    (output / "stage188b_paired_internal_margin_analysis_report.md").write_text(markdown, encoding="utf-8")
    return 2 if decision == BLOCKED else 0


if __name__ == "__main__":
    raise SystemExit(main())
