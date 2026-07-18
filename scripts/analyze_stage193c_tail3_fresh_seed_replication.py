#!/usr/bin/env python3
"""Analyze existing Stage193 fresh-seed trajectory exports without loading models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

TRAINER_BLOB_COMMIT = "e83d8af756fa84b7a91c14e0910ae388b07b5f02"
STAGE192_COMMIT = "a768d848256f88a7a1a15cc02a058f4d7d0a35f7"
STAGE191_COMMIT = "0872e66ccb05ae8a166f5cabf4e084272dc49500"
SIDECAR_SHA = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
A_READY = "STAGE193A_TAIL3_FRESH_SEED_MANIFEST_READY"
BLOCKED = "STAGE193C_TAIL3_FRESH_SEED_REPLICATION_BLOCKED"
REPLICATED = "STAGE193C_TAIL3_SMOOTHING_REPLICATED"
PARTIAL = "STAGE193C_TAIL3_SMOOTHING_PARTIAL_SIGNAL"
NOT_REPLICATED = "STAGE193C_TAIL3_SMOOTHING_NOT_REPLICATED"
STAGE192_CLOSED = "STAGE192A_NO_TRAJECTORY_STABLE_SELECTOR"
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
SEEDS = (177, 178, 179)
OLD_SEEDS = (174, 175, 176)
ARMS = ("baseline", "intervention")
RUNS = tuple(f"seed{seed}_{arm}" for seed in SEEDS for arm in ARMS)
EPOCHS = tuple(range(1, 21))
COMPARATORS = ("independent_selected", "tail2_mean_logits", "tail3_mean_logits")
METRICS = ("clean_ce", "accuracy", "macro_f1", "support_recall",
           "false_entitlement", "false_not_entitled", "polarity_error",
           "pred_REFUTE", "pred_NOT_ENTITLED", "pred_SUPPORT")
MATRIX_FIELDS = tuple(f"{a}_to_{b}" for a in LABELS for b in LABELS)
TRANSITION_FIELDS = (*MATRIX_FIELDS, "unchanged_rows", "changed_rows",
    "not_entitled_to_support", "support_to_not_entitled", "refute_involved_transitions",
    "exclusive_not_entitled_support_changed_rows",
    "exclusive_not_entitled_support_fraction_of_changed", "corrections", "regressions",
    "wrong_to_different_wrong")
A_OUTPUTS = {
    "stage193a_tail3_fresh_seed_manifest_report.json",
    "stage193a_tail3_fresh_seed_manifest_report.md", "stage193a_run_manifest.jsonl",
    "stage193a_run_command_matrix.csv", "stage193a_source_and_template_gate.csv",
    "stage193a_precommitted_gate.csv",
}
STAGE192_OUTPUTS = {
    "stage192a_trajectory_stable_selection_report.json",
    "stage192a_trajectory_stable_selection_report.md",
    "stage192a_stage191d_closure_gate.csv", "stage192a_selector_definition.csv",
    "stage192a_selector_choice_by_seed.csv", "stage192a_selected_arm_metrics.csv",
    "stage192a_selector_aggregate_metrics.csv", "stage192a_pair_delta_by_selector.csv",
    "stage192a_perturbation_grid.csv", "stage192a_perturbation_summary.csv",
    "stage192a_selected_pair_transition_summary.csv",
    "stage192a_selected_pair_transition_by_gold.csv",
    "stage192a_temporal_ensemble_comparator.csv", "stage192a_precommitted_gate.csv",
}
OUTPUTS = {
    "json": "stage193c_tail3_fresh_seed_replication_report.json",
    "md": "stage193c_tail3_fresh_seed_replication_report.md",
    "closure": "stage193c_stage192a_closure_gate.csv",
    "identity": "stage193c_run_identity_gate.csv",
    "epochs": "stage193c_epoch_metric_reconstruction.csv",
    "by_seed": "stage193c_comparator_metrics_by_seed.csv",
    "fresh": "stage193c_comparator_aggregate_fresh.csv",
    "pooled": "stage193c_comparator_aggregate_pooled.csv",
    "transitions": "stage193c_pair_transition_summary.csv",
    "transition_gold": "stage193c_pair_transition_by_gold.csv",
    "criteria": "stage193c_primary_criterion_gate.csv",
    "decision": "stage193c_precommitted_decision_gate.csv",
}
CSV_HEADERS = {
    "closure": ["gate", "required", "observed", "passed", "blocking_reason"],
    "identity": ["run", "gate", "required", "observed", "passed", "blocking_reason"],
    "epochs": ["run", "epoch", "prediction_export_path", "prediction_export_sha256",
        "row_count", "dev_position_exact", "gold_alignment_exact", "metrics_exact",
        "clean_ce_expected", "clean_ce_reconstructed", "passed", "blocking_reason"],
    "by_seed": ["comparator", "seed", "baseline_epoch_source", "intervention_epoch_source",
        *[f"baseline_{m}" for m in METRICS], *[f"intervention_{m}" for m in METRICS],
        *[f"delta_{m}" for m in METRICS]],
    "fresh": ["comparator", "seed_count", "mean_pair_clean_ce", "mean_pair_accuracy",
        "mean_pair_macro_f1", "mean_pair_support_recall", "support_delta_min",
        "support_delta_max", "support_delta_range", "false_entitlement_delta_min",
        "false_entitlement_delta_max", "false_entitlement_delta_range",
        "max_abs_refute_delta", "max_abs_polarity_delta"],
    "pooled": ["comparator", "seed_count", "descriptive_only", "no_statistical_significance_claim",
        "mean_pair_clean_ce", "mean_pair_accuracy", "mean_pair_macro_f1",
        "mean_pair_support_recall", "support_delta_min", "support_delta_max",
        "support_delta_range", "false_entitlement_delta_min",
        "false_entitlement_delta_max", "false_entitlement_delta_range",
        "max_abs_refute_delta", "max_abs_polarity_delta"],
    "transitions": ["comparator", "seed", *TRANSITION_FIELDS],
    "transition_gold": ["comparator", "seed", "gold_label", *TRANSITION_FIELDS],
    "criteria": ["criterion_group", "criterion", "required", "observed", "passed"],
    "decision": ["decision", "taxonomy_condition", "required", "observed", "passed"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--stage193a-dir", type=Path, required=True)
    p.add_argument("--stage193b-run-root", type=Path, required=True)
    p.add_argument("--stage192a-dir", type=Path, required=True)
    p.add_argument("--current-diagnostic-git-commit", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle: return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip(): raise ValueError(f"{path}:{number}: blank JSONL row")
            value = json.loads(line)
            if type(value) is not dict: raise ValueError(f"{path}:{number}: row is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle); return list(reader.fieldnames or []), list(reader)


def bool_csv(value: str) -> bool | None:
    if value in ("True", "true"): return True
    if value in ("False", "false"): return False
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def exact_int(value: Any) -> bool:
    return type(value) is int


def close(left: Any, right: Any) -> bool:
    return finite(left) and finite(right) and math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-7)


def csv_value(value: Any) -> Any:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value


def write_csv(path: Path, header: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore"); writer.writeheader()
        for row in rows: writer.writerow({key: csv_value(row.get(key)) for key in header})


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def git_call(repo: Path, arguments: list[str], *, binary: bool = False, dirty: bool = False) -> Any:
    result = subprocess.run(["git", *arguments], cwd=repo, check=False, capture_output=True, shell=False)
    if dirty:
        if result.returncode not in (0, 1): raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        return result.returncode
    if result.returncode != 0: raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict").strip()


def source_identity(repo: Path, commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", commit or "") is None:
        raise ValueError("current diagnostic commit must be lowercase hexadecimal length 40")
    files = {}
    for relative in ("reports/stage193a_tail3_fresh_seed_replication_spec.md",
                     "scripts/build_stage193a_tail3_fresh_seed_manifest.py",
                     "scripts/analyze_stage193c_tail3_fresh_seed_replication.py"):
        current = (repo / relative).read_bytes()
        blob = git_call(repo, ["show", f"{commit}:{relative}"], binary=True)
        files[relative] = {"current_sha256": hashlib.sha256(current).hexdigest(),
            "commit_blob_sha256": hashlib.sha256(blob).hexdigest(), "bytes_equal": current == blob,
            "unstaged_clean": git_call(repo, ["diff", "--quiet", "--", relative], dirty=True) == 0,
            "staged_clean": git_call(repo, ["diff", "--cached", "--quiet", "--", relative], dirty=True) == 0}
    trainer_relative = "scripts/train_controlled_v6b_minimal.py"
    trainer_current = (repo / trainer_relative).read_bytes()
    trainer_blob = git_call(repo, ["show", f"{TRAINER_BLOB_COMMIT}:{trainer_relative}"], binary=True)
    trainer_identity = {"path": str((repo / trainer_relative).resolve()),
        "blob_commit": TRAINER_BLOB_COMMIT,
        "current_sha256": hashlib.sha256(trainer_current).hexdigest(),
        "commit_blob_sha256": hashlib.sha256(trainer_blob).hexdigest(),
        "bytes_equal": trainer_current == trainer_blob}
    head = git_call(repo, ["rev-parse", "HEAD"])
    passed = (head == commit and trainer_identity["bytes_equal"] and
        all(x["bytes_equal"] and x["unstaged_clean"] and x["staged_clean"] for x in files.values()))
    return {"supplied_commit": commit, "repository_head": head, "files": files,
        "trainer": trainer_identity, "passed": passed}


def establish_safe_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    repo = args.repo_root.resolve()
    if not repo.is_dir(): raise ValueError("repo root is not a directory")
    reports = (repo / "reports").resolve()
    stage193a, run_root, stage192, output = (args.stage193a_dir.resolve(),
        args.stage193b_run_root.resolve(), args.stage192a_dir.resolve(), args.output_dir.resolve())
    if stage193a.parent != reports or not stage193a.name.startswith("stage193a_tail3_fresh_seed_manifest_") or not stage193a.is_dir():
        raise ValueError("Stage193-A directory is unsafe or absent")
    if run_root.parent != reports or not run_root.name.startswith("stage193b_tail3_fresh_seed_runs_") or not run_root.is_dir():
        raise ValueError("Stage193-B run root is unsafe or absent")
    if stage192.parent != reports or not stage192.name.startswith("stage192a_trajectory_stable_selection_") or not stage192.is_dir():
        raise ValueError("Stage192-A directory is unsafe or absent")
    if output.parent != reports or not output.name.startswith("stage193c_tail3_fresh_seed_replication_"):
        raise ValueError("Stage193-C output path is unsafe")
    if len({stage193a, run_root, stage192, output}) != 4: raise ValueError("input and output paths must differ")
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise ValueError("Stage193-C output exists and is nonempty")
    return repo, stage193a, run_root, stage192, output


def recursive_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if type(value) is dict:
        yield value
        for child in value.values(): yield from recursive_dicts(child)
    elif type(value) is list:
        for child in value: yield from recursive_dicts(child)

def option_map(argv: Any) -> dict[str, Any]:
    if type(argv) is not list or any(type(token) is not str for token in argv):
        raise ValueError("argv must be a string list")
    result: dict[str, Any] = {}; index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--") or "=" in token: raise ValueError(f"unsupported argv token {token!r}")
        key = token[2:].replace("-", "_")
        if key in result: raise ValueError(f"duplicate argv option {token}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            result[key] = argv[index + 1]; index += 2
        else:
            result[key] = True; index += 1
    return result


def exported_rows(path: Path, epoch: int, run: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != 720: raise ValueError(f"{run} epoch {epoch}: expected 720 rows")
    for position, row in enumerate(rows):
        if not exact_int(row.get("epoch")) or row["epoch"] != epoch or not exact_int(row.get("dev_position")) or row["dev_position"] != position:
            raise ValueError(f"{run} epoch {epoch}: epoch/dev_position mismatch")
        if row.get("gold_final_label") not in LABELS or row.get("predicted_final_label") not in LABELS:
            raise ValueError(f"{run} epoch {epoch}: noncanonical label")
        logits = row.get("final_logits")
        if type(logits) is not list or len(logits) != 3 or any(not finite(x) for x in logits):
            raise ValueError(f"{run} epoch {epoch}: invalid final_logits")
        if not finite(row.get("final_ce")): raise ValueError(f"{run} epoch {epoch}: invalid final_ce")
        best = max(range(3), key=lambda index: float(logits[index]))
        if row["predicted_final_label"] != LABELS[best]:
            raise ValueError(f"{run} epoch {epoch}: predicted label differs from canonical argmax")
    return rows


def metrics(rows: list[dict[str, Any]], torch: Any, *, float32_row_ce: bool) -> dict[str, Any]:
    if len(rows) != 720: raise ValueError("metric input must contain 720 rows")
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows: matrix[row["gold_final_label"]][row["predicted_final_label"]] += 1
    gold_counts = {gold: sum(matrix[gold].values()) for gold in LABELS}
    pred_counts = {pred: sum(matrix[gold][pred] for gold in LABELS) for pred in LABELS}
    if any(gold_counts[label] == 0 for label in LABELS): raise ValueError("canonical gold class absent")
    f1s = []
    for label in LABELS:
        tp = matrix[label][label]; predicted = pred_counts[label]; gold = gold_counts[label]
        precision = tp / predicted if predicted else 0.0; recall = tp / gold
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    ce_values = [float(row["final_ce"]) for row in rows]
    clean_ce = (torch.tensor(ce_values, dtype=torch.float32, device="cpu").mean().item()
                if float32_row_ce else sum(ce_values) / len(ce_values))
    return {"clean_ce": clean_ce, "accuracy": sum(matrix[x][x] for x in LABELS) / 720,
        "macro_f1": sum(f1s) / 3, "support_recall": matrix["SUPPORT"]["SUPPORT"] / gold_counts["SUPPORT"],
        "false_entitlement": matrix["NOT_ENTITLED"]["REFUTE"] + matrix["NOT_ENTITLED"]["SUPPORT"],
        "false_not_entitled": matrix["REFUTE"]["NOT_ENTITLED"] + matrix["SUPPORT"]["NOT_ENTITLED"],
        "polarity_error": matrix["REFUTE"]["SUPPORT"] + matrix["SUPPORT"]["REFUTE"],
        **{f"pred_{label}": pred_counts[label] for label in LABELS},
        "gold_counts": gold_counts, "confusion_matrix": matrix}


def trajectory_metrics_exact(trajectory: dict[str, Any], observed: dict[str, Any]) -> bool:
    float_fields = {"clean_dev_ce": "clean_ce", "clean_accuracy": "accuracy",
                    "clean_macro_f1": "macro_f1", "support_recall": "support_recall"}
    count_fields = {"false_entitlement_total": "false_entitlement",
                    "false_not_entitled_total": "false_not_entitled",
                    "polarity_error_total": "polarity_error"}
    prediction_counts = trajectory.get("normalized_prediction_counts")
    gold_counts = trajectory.get("gold_counts")
    matrix = trajectory.get("confusion_matrix_gold_by_prediction")
    strict_prediction_counts = (type(prediction_counts) is dict and set(prediction_counts) == set(LABELS) and
        all(exact_int(prediction_counts.get(label)) and
            prediction_counts.get(label) == observed[f"pred_{label}"] for label in LABELS))
    strict_gold_counts = (type(gold_counts) is dict and set(gold_counts) == set(LABELS) and
        all(exact_int(gold_counts.get(label)) and
            gold_counts.get(label) == observed["gold_counts"][label] for label in LABELS))
    strict_matrix = (type(matrix) is dict and set(matrix) == set(LABELS) and
        all(type(matrix.get(gold)) is dict and set(matrix[gold]) == set(LABELS) and
            all(exact_int(matrix[gold].get(pred)) and
                matrix[gold].get(pred) == observed["confusion_matrix"][gold][pred] for pred in LABELS)
            for gold in LABELS))
    return (exact_int(trajectory.get("dev_row_count")) and trajectory.get("dev_row_count") == 720 and
        trajectory.get("clean_dev_ce") == observed["clean_ce"] and
        all(close(trajectory.get(field), observed[key]) for field, key in float_fields.items() if field != "clean_dev_ce") and
        all(exact_int(trajectory.get(field)) and trajectory.get(field) == observed[key]
            for field, key in count_fields.items()) and
        strict_prediction_counts and strict_gold_counts and strict_matrix)

def ensemble_rows(epoch_rows: dict[int, list[dict[str, Any]]], epochs: tuple[int, ...]) -> list[dict[str, Any]]:
    output = []
    for position in range(720):
        originals = [epoch_rows[epoch][position] for epoch in epochs]
        gold = originals[0]["gold_final_label"]
        if any(row["gold_final_label"] != gold for row in originals): raise ValueError("ensemble gold alignment mismatch")
        logits = [sum(float(row["final_logits"][index]) for row in originals) / len(originals) for index in range(3)]
        best = max(range(3), key=lambda index: logits[index])
        maximum = max(logits); logsumexp = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
        output.append({"epoch": None, "dev_position": position, "gold_final_label": gold,
            "predicted_final_label": LABELS[best], "final_logits": logits,
            "final_ce": logsumexp - logits[LABELS.index(gold)]})
    return output


def transition(left: list[dict[str, Any]], right: list[dict[str, Any]], gold_filter: str | None = None) -> dict[str, Any]:
    matrix = {a: {b: 0 for b in LABELS} for a in LABELS}; corrections = regressions = wrong = 0
    for a, b in zip(left, right):
        if a["dev_position"] != b["dev_position"] or a["gold_final_label"] != b["gold_final_label"]:
            raise ValueError("transition alignment mismatch")
        gold = a["gold_final_label"]
        if gold_filter is not None and gold != gold_filter: continue
        before, after = a["predicted_final_label"], b["predicted_final_label"]
        matrix[before][after] += 1
        if before != after:
            if before != gold and after == gold: corrections += 1
            elif before == gold and after != gold: regressions += 1
            elif before != gold and after != gold: wrong += 1
    changed = sum(matrix[a][b] for a in LABELS for b in LABELS if a != b)
    exclusive = matrix["NOT_ENTITLED"]["SUPPORT"] + matrix["SUPPORT"]["NOT_ENTITLED"]
    return {**{f"{a}_to_{b}": matrix[a][b] for a in LABELS for b in LABELS},
        "unchanged_rows": sum(matrix[a][a] for a in LABELS), "changed_rows": changed,
        "not_entitled_to_support": matrix["NOT_ENTITLED"]["SUPPORT"],
        "support_to_not_entitled": matrix["SUPPORT"]["NOT_ENTITLED"],
        "refute_involved_transitions": sum(matrix[a][b] for a in LABELS for b in LABELS if a != b and "REFUTE" in (a, b)),
        "exclusive_not_entitled_support_changed_rows": exclusive,
        "exclusive_not_entitled_support_fraction_of_changed": exclusive / changed if changed else 0.0,
        "corrections": corrections, "regressions": regressions, "wrong_to_different_wrong": wrong}


def pair_row(comparator: str, seed: int, baseline: dict[str, Any], intervention: dict[str, Any], baseline_source: Any, intervention_source: Any) -> dict[str, Any]:
    return {"comparator": comparator, "seed": seed, "baseline_epoch_source": baseline_source,
        "intervention_epoch_source": intervention_source,
        **{f"baseline_{key}": baseline[key] for key in METRICS},
        **{f"intervention_{key}": intervention[key] for key in METRICS},
        **{f"delta_{key}": intervention[key] - baseline[key] for key in METRICS}}


def aggregate(comparator: str, rows: list[dict[str, Any]], *, pooled: bool) -> dict[str, Any]:
    support = [row["delta_pred_SUPPORT"] for row in rows]
    false_e = [row["delta_false_entitlement"] for row in rows]
    result = {"comparator": comparator, "seed_count": len(rows),
        "mean_pair_clean_ce": sum((r["baseline_clean_ce"] + r["intervention_clean_ce"]) / 2 for r in rows) / len(rows),
        "mean_pair_accuracy": sum((r["baseline_accuracy"] + r["intervention_accuracy"]) / 2 for r in rows) / len(rows),
        "mean_pair_macro_f1": sum((r["baseline_macro_f1"] + r["intervention_macro_f1"]) / 2 for r in rows) / len(rows),
        "mean_pair_support_recall": sum((r["baseline_support_recall"] + r["intervention_support_recall"]) / 2 for r in rows) / len(rows),
        "support_delta_min": min(support), "support_delta_max": max(support),
        "support_delta_range": max(support) - min(support),
        "false_entitlement_delta_min": min(false_e), "false_entitlement_delta_max": max(false_e),
        "false_entitlement_delta_range": max(false_e) - min(false_e),
        "max_abs_refute_delta": max(abs(r["delta_pred_REFUTE"]) for r in rows),
        "max_abs_polarity_delta": max(abs(r["delta_polarity_error"]) for r in rows)}
    if pooled: result.update({"descriptive_only": True, "no_statistical_significance_claim": True})
    return result


def validate_stage192(stage192: Path, closure_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    entries = {path.name for path in stage192.iterdir()}
    exact_outputs = entries == STAGE192_OUTPUTS and all((stage192 / name).is_file() for name in STAGE192_OUTPUTS)
    closure_rows.append({"gate": "exact_stage192a_fourteen_outputs", "required": sorted(STAGE192_OUTPUTS),
        "observed": sorted(entries), "passed": exact_outputs, "blocking_reason": "" if exact_outputs else "output set mismatch"})
    if not exact_outputs: raise ValueError("Stage192-A output set mismatch")
    report = read_json(stage192 / "stage192a_trajectory_stable_selection_report.json")
    required = {"stage": "Stage192-A", "decision": STAGE192_CLOSED, "runnable": True,
        "blocking_reasons": [], "diagnostic_only": True, "training_performed": False,
        "model_constructed": False, "external_data_used": False, "model_advancement_decision": False,
        "stage192b_training_authorized": False, "quality_preserving_selectors": [],
        "quality_tradeoff_selectors": [], "winning_selector": None,
        "current_diagnostic_git_commit": STAGE192_COMMIT}
    observed = {key: report.get(key) for key in required}
    closure = report.get("global_closure_results")
    passed = observed == required and type(closure) is dict and bool(closure) and all(value is True for value in closure.values())
    closure_rows.append({"gate": "stage192a_no_selector_closure", "required": required,
        "observed": observed, "passed": passed, "blocking_reason": "" if passed else "closure mismatch"})
    if not passed: raise ValueError("Stage192-A closure mismatch")

    header, arms = read_csv(stage192 / "stage192a_selected_arm_metrics.csv")
    expected_header = ["selector", "seed", "arm", "selected_epoch", "available",
        "clean_dev_ce", "clean_accuracy", "clean_macro_f1", "support_recall",
        "false_entitlement_total", "false_not_entitled_total", "polarity_error_total",
        "pred_REFUTE", "pred_NOT_ENTITLED", "pred_SUPPORT"]
    if header != expected_header: raise ValueError("Stage192-A selected-arm metric header mismatch")
    historical = [row for row in arms if row["selector"] == "historical_independent"]
    if len(historical) != 6: raise ValueError("Stage192-A historical independent rows are not exact")
    old: dict[str, list[dict[str, Any]]] = {name: [] for name in COMPARATORS}
    for seed in OLD_SEEDS:
        converted = {}
        for arm in ARMS:
            matches = [row for row in historical if int(row["seed"]) == seed and row["arm"] == arm and row["available"].lower() == "true"]
            if len(matches) != 1: raise ValueError("Stage192 historical arm identity mismatch")
            row = matches[0]
            converted[arm] = {"clean_ce": float(row["clean_dev_ce"]), "accuracy": float(row["clean_accuracy"]),
                "macro_f1": float(row["clean_macro_f1"]), "support_recall": float(row["support_recall"]),
                "false_entitlement": int(row["false_entitlement_total"]),
                "false_not_entitled": int(row["false_not_entitled_total"]),
                "polarity_error": int(row["polarity_error_total"]),
                **{f"pred_{label}": int(row[f"pred_{label}"]) for label in LABELS}}
        old["independent_selected"].append(pair_row("independent_selected", seed, converted["baseline"], converted["intervention"], "historical_selected", "historical_selected"))
    evidence = report.get("temporal_ensemble_evidence")
    if type(evidence) is not dict or set(evidence) != {"tail2_mean_logits", "tail3_mean_logits"}:
        raise ValueError("Stage192 temporal ensemble comparator names mismatch")
    source_names = {"clean_ce": "clean_dev_ce", "accuracy": "clean_accuracy", "macro_f1": "clean_macro_f1",
        "support_recall": "support_recall", "false_entitlement": "false_entitlement_total",
        "false_not_entitled": "false_not_entitled_total", "polarity_error": "polarity_error_total",
        "pred_REFUTE": "pred_REFUTE", "pred_NOT_ENTITLED": "pred_NOT_ENTITLED", "pred_SUPPORT": "pred_SUPPORT"}
    for comparator in ("tail2_mean_logits", "tail3_mean_logits"):
        rows = evidence[comparator]
        for seed in OLD_SEEDS:
            raw = rows.get(str(seed), rows.get(seed)) if type(rows) is dict else None
            if type(raw) is not dict or raw.get("comparator") != comparator or raw.get("seed") != seed:
                raise ValueError(f"Stage192 {comparator} seed row mismatch")
            baseline = {key: raw[f"baseline_{source}"] for key, source in source_names.items()}
            intervention = {key: raw[f"intervention_{source}"] for key, source in source_names.items()}
            old[comparator].append(pair_row(comparator, seed, baseline, intervention, raw.get("epochs"), raw.get("epochs")))
    return report, old

def load_stage193a(stage193a: Path, run_root: Path, current_commit: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries = {path.name for path in stage193a.iterdir()}
    if entries != A_OUTPUTS or any(not (stage193a / name).is_file() for name in A_OUTPUTS):
        raise ValueError("Stage193-A exact six-output set mismatch")
    report = read_json(stage193a / "stage193a_tail3_fresh_seed_manifest_report.json")
    required = {"stage": "Stage193-A", "decision": A_READY, "runnable": True,
        "blocking_reasons": [], "diagnostic_only": True,
        "exact_six_run_diagnostic_execution_authorized": True,
        "training_for_model_advancement_authorized": False, "model_advancement_decision": False,
        "subsequent_training_authorized": False, "external_data_used": False,
        "checkpoint_loaded": False, "model_loaded": False, "capsule_loaded": False,
        "statistical_significance_claim": False,
        "current_diagnostic_git_commit": current_commit,
        "stage193_runtime_repository_commit": current_commit,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT, "ordered_runs": list(RUNS),
        "run_manifest_count": 6, "expected_trajectory_rows_per_run": 20,
        "expected_prediction_exports_per_run": 20, "expected_prediction_rows_per_export": 720,
        "expected_state_capsules_per_run": 0, "canonical_labels": list(LABELS),
        "logits_source": 'output["logits"]'}
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("Stage193-A READY closure mismatch")
    report_integer_contract = {"run_manifest_count": 6,
        "expected_trajectory_rows_per_run": 20, "expected_prediction_exports_per_run": 20,
        "expected_prediction_rows_per_export": 720, "expected_state_capsules_per_run": 0}
    if any(not exact_int(report.get(key)) or report.get(key) != value
           for key, value in report_integer_contract.items()):
        raise ValueError("Stage193-A report strict integer contract mismatch")
    trainer_blob_sha256 = report.get("trainer_blob_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(trainer_blob_sha256 or "")) is None:
        raise ValueError("Stage193-A trainer blob SHA256 is invalid")
    frozen = report.get("frozen_source_identities") or {}
    frozen_required = {"trainer_blob_commit": TRAINER_BLOB_COMMIT,
        "stage192a_implementation_commit": STAGE192_COMMIT,
        "stage191b_replay_implementation_commit": STAGE191_COMMIT,
        "stage185_sidecar_semantic_sha256": SIDECAR_SHA}
    if any(frozen.get(key) != value for key, value in frozen_required.items()):
        raise ValueError("Stage193-A frozen source identities mismatch")
    source_identity_value = report.get("source_identity") or {}
    source_trainer = source_identity_value.get("trainer") or {}
    if (source_identity_value.get("passed") is not True or
            source_identity_value.get("supplied_commit") != current_commit or
            source_identity_value.get("repository_head") != current_commit or
            source_trainer.get("blob_commit") != TRAINER_BLOB_COMMIT or
            source_trainer.get("current_sha256") != trainer_blob_sha256 or
            source_trainer.get("commit_blob_sha256") != trainer_blob_sha256 or
            source_trainer.get("bytes_equal") is not True):
        raise ValueError("Stage193-A source identity did not preserve distinct runtime/blob identities")
    if Path(str(report.get("stage193b_run_root", ""))).resolve() != run_root:
        raise ValueError("Stage193-A run root differs from supplied Stage193-B root")
    gate_header = ["gate", "required", "observed", "passed", "blocking_reason"]
    for gate_name in ("stage193a_source_and_template_gate.csv", "stage193a_precommitted_gate.csv"):
        header, gate_rows = read_csv(stage193a / gate_name)
        if header != gate_header or not gate_rows or any(bool_csv(row["passed"]) is not True for row in gate_rows):
            raise ValueError(f"Stage193-A gate CSV closure mismatch: {gate_name}")
    matrix_header, matrix_rows = read_csv(stage193a / "stage193a_run_command_matrix.csv")
    required_matrix_header = ["run", "training_seed", "split_seed", "arm", "planned_run_directory",
        "planned_output_json_path", "planned_selected_checkpoint_path", "trainer_source_path",
        "trainer_blob_commit", "trainer_blob_sha256", "runtime_repository_commit", "command",
        "expected_trajectory_rows", "expected_prediction_exports",
        "expected_prediction_rows_per_export", "expected_state_capsules"]
    if matrix_header != required_matrix_header or [row.get("run") for row in matrix_rows] != list(RUNS):
        raise ValueError("Stage193-A command matrix schema/order mismatch")
    manifests = read_jsonl(stage193a / "stage193a_run_manifest.jsonl")
    if len(manifests) != 6 or [row.get("run") for row in manifests] != list(RUNS):
        raise ValueError("Stage193-A JSONL run order/cardinality mismatch")
    for row, matrix_row, run in zip(manifests, matrix_rows, RUNS):
        seed = int(run[4:7]); arm = run.split("_", 1)[1]; run_dir = (run_root / run).resolve()
        required_row = {"stage": "Stage193-A", "run": run, "training_seed": seed,
            "split_seed": 174, "arm": arm, "canonical_labels": list(LABELS),
            "trainer_blob_commit": TRAINER_BLOB_COMMIT,
            "trainer_blob_sha256": trainer_blob_sha256,
            "stage193_runtime_repository_commit": current_commit,
            "planned_run_directory": str(run_dir),
            "planned_output_json_path": str((run_dir / "training_report.json").resolve()),
            "planned_selected_checkpoint_path": str((run_dir / "selected_checkpoint.pt").resolve()),
            "expected_trajectory_contract_path": str((run_dir / "stage191_trajectory_contract.json").resolve()),
            "expected_trajectory_ledger_path": str((run_dir / "stage191_trajectory_epoch_metrics.jsonl").resolve()),
            "expected_prediction_export_paths": [str((run_dir / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl").resolve()) for epoch in EPOCHS],
            "expected_trajectory_rows": 20, "expected_prediction_exports": 20,
            "expected_prediction_rows_per_export": 720, "expected_state_capsules": 0,
            "logits_source": 'output["logits"]', "runnable": True, "diagnostic_only": True,
            "exact_six_run_diagnostic_execution_authorized": True,
            "training_for_model_advancement_authorized": False,
            "model_advancement_decision": False, "subsequent_training_authorized": False,
            "external_data_used": False}
        if any(row.get(key) != value for key, value in required_row.items()):
            raise ValueError(f"{run}: Stage193-A manifest row mismatch")
        manifest_integer_contract = {"training_seed": seed, "split_seed": 174,
            "expected_trajectory_rows": 20, "expected_prediction_exports": 20,
            "expected_prediction_rows_per_export": 720, "expected_state_capsules": 0}
        if any(not exact_int(row.get(key)) or row.get(key) != value
               for key, value in manifest_integer_contract.items()):
            raise ValueError(f"{run}: Stage193-A manifest strict integer contract mismatch")
        matrix_identity = {"trainer_source_path": source_trainer.get("path"),
            "trainer_blob_commit": TRAINER_BLOB_COMMIT,
            "trainer_blob_sha256": trainer_blob_sha256, "runtime_repository_commit": current_commit,
            "training_seed": str(seed), "split_seed": "174", "expected_trajectory_rows": "20",
            "expected_prediction_exports": "20", "expected_prediction_rows_per_export": "720",
            "expected_state_capsules": "0"}
        if any(matrix_row.get(key) != value for key, value in matrix_identity.items()):
            raise ValueError(f"{run}: Stage193-A command matrix identity/cardinality mismatch")
        options = option_map(row.get("argv"))
        argv_required = {"architecture": "v6b_minimal", "backbone": "mamba", "device": "cuda",
            "model_name": "state-spaces/mamba-130m-hf", "data": "data/controlled_v5_v3_without_time_swap.jsonl",
            "seed": str(seed), "split_seed": "174", "epochs": "20", "select_metric": "final_macro_f1",
            "selected_checkpoint_filename": "selected_checkpoint.pt",
            "stage193_tail3_fresh_seed_observability": True}
        if any(options.get(key) != value for key, value in argv_required.items()):
            raise ValueError(f"{run}: frozen manifest argv envelope mismatch")
        if "stage191_trajectory_replay_observability" in options or "stage191_save_trajectory_state_capsules" in options:
            raise ValueError(f"{run}: Stage191 observability/capsule argv present")
        output_required = {"output_json": run_dir / "training_report.json",
            "output_predictions_json": run_dir / "clean_dev_predictions.json",
            "stage115_clean_dev_scalar_output_jsonl": run_dir / "clean_dev_scalars.jsonl"}
        if any(Path(str(options.get(key, ""))).resolve() != path.resolve() for key, path in output_required.items()):
            raise ValueError(f"{run}: manifest output argv path mismatch")
        frozen_sidecar = Path(str(frozen.get("stage185_sidecar_path", ""))).resolve()
        if arm == "baseline":
            arm_ok = (options.get("compatible_positive_margin_weight") in ("0", "0.0") and
                      "controlled_integrity_sidecar_path" not in options and
                      "expected_integrity_sidecar_semantic_sha256" not in options)
        else:
            arm_ok = (options.get("compatible_positive_margin_weight") == "0.05" and
                      options.get("compatible_positive_margin_logit") == "0.0" and
                      Path(str(options.get("controlled_integrity_sidecar_path", ""))).resolve() == frozen_sidecar and
                      options.get("expected_integrity_sidecar_semantic_sha256") == SIDECAR_SHA)
        if not arm_ok: raise ValueError(f"{run}: manifest argv arm contract mismatch")
    return report, manifests

def load_fresh_runs(manifests: list[dict[str, Any]], run_root: Path, tables: dict[str, list[dict[str, Any]]],
                    torch: Any, runtime_commit: str, trainer_blob_sha256: str) -> dict[str, Any]:
    root_entries = {path.name for path in run_root.iterdir()}
    if root_entries != set(RUNS) or any(not (run_root / run).is_dir() for run in RUNS):
        raise ValueError("Stage193-B run root does not contain exactly the six frozen run directories")
    data: dict[str, Any] = {}; global_golds: list[str] | None = None
    for manifest in manifests:
        run = manifest["run"]; seed = manifest["training_seed"]; arm = manifest["arm"]
        run_dir = (run_root / run).resolve()
        def identity_gate(name: str, required: Any, observed: Any, passed: bool, reason: str) -> None:
            tables["identity"].append({"run": run, "gate": name, "required": required,
                "observed": observed, "passed": passed, "blocking_reason": "" if passed else reason})
            if not passed: raise ValueError(f"{run}: {reason}")
        identity_gate("run_directory", str(run_dir), str(run_dir), run_dir.is_dir(), "run directory absent")
        contract_path = Path(manifest["expected_trajectory_contract_path"]).resolve()
        ledger_path = Path(manifest["expected_trajectory_ledger_path"]).resolve()
        report_path = Path(manifest["planned_output_json_path"]).resolve()
        if not contract_path.is_file() or not ledger_path.is_file() or not report_path.is_file():
            raise ValueError(f"{run}: required report/contract/ledger absent")
        contract = read_json(contract_path)
        enabled = contract.get("enabled_flags")
        required_contract = {"observability_mode": "stage193_tail3_fresh_seed_replication",
            "authorized_training_seeds": list(SEEDS), "training_seed_authorized": True,
            "training_seed": seed, "split_seed": 174, "arm": arm, "epoch_count": 20,
            "expected_dev_rows": 720, "expected_state_capsules": 0,
            "canonical_logit_column_labels": list(LABELS), "logits_source": 'output["logits"]',
            "stage191_trajectory_observability_implementation_reused": True,
            "state_capsule_saving_enabled": False, "training_semantics_changed": False,
            "extra_forward_pass_performed": False, "loss_logits_used": False,
            "external_data_used": False, "trainer_source_commit": runtime_commit,
            "trainer_sha256": trainer_blob_sha256}
        contract_integer_contract = {"training_seed": seed, "split_seed": 174, "epoch_count": 20,
            "expected_dev_rows": 720, "expected_state_capsules": 0}
        authorized = contract.get("authorized_training_seeds")
        contract_integer_ok = (type(authorized) is list and authorized == list(SEEDS) and
            all(exact_int(value) for value in authorized) and
            all(exact_int(contract.get(key)) and contract.get(key) == value
                for key, value in contract_integer_contract.items()))
        contract_ok = (all(contract.get(key) == value for key, value in required_contract.items()) and
            contract_integer_ok)
        enabled_ok = enabled == {"stage191_trajectory_replay_observability": False,
            "stage191_save_trajectory_state_capsules": False,
            "stage193_tail3_fresh_seed_observability": True}
        identity_gate("trajectory_contract", required_contract,
            {key: contract.get(key) for key in required_contract}, contract_ok and enabled_ok,
            "Stage193 trajectory contract mismatch")

        training_report = read_json(report_path)
        if type(training_report) is not dict:
            raise ValueError(f"{run}: training report root is not an exact object")
        provenance_path = (run_dir / "run_provenance.json").resolve()
        if Path(str(training_report.get("run_provenance_json", ""))).resolve() != provenance_path or not provenance_path.is_file():
            raise ValueError(f"{run}: run provenance path mismatch")
        report_runs = training_report.get("runs")
        if (type(report_runs) is not dict or set(report_runs) != {"single"} or
                type(report_runs.get("single")) is not dict):
            raise ValueError(f"{run}: training report runs.single schema mismatch")
        report_selected_epoch = training_report["runs"]["single"]["best_epoch"]
        report_final_epoch = training_report["runs"]["single"]["final_epoch"]
        if (not exact_int(report_selected_epoch) or report_selected_epoch not in EPOCHS or
                not exact_int(report_final_epoch) or report_final_epoch != 20):
            raise ValueError(f"{run}: training report best/final epoch contract mismatch")
        provenance = read_json(provenance_path)
        raw_argv = provenance.get("raw_sys_argv")
        identity_gate("exact_invoked_argv", manifest["argv"], raw_argv,
                      raw_argv == manifest["argv"], "runtime argv differs from Stage193-A")
        source = provenance.get("source_provenance") or {}
        runtime_commit_ok = source.get("git_commit") == runtime_commit
        trainer_sha_ok = source.get("trainer_sha256") == trainer_blob_sha256
        identity_gate("runtime_repository_commit", runtime_commit, source.get("git_commit"),
                      runtime_commit_ok, "runtime repository commit mismatch")
        identity_gate("trainer_blob_sha256", trainer_blob_sha256, source.get("trainer_sha256"),
                      trainer_sha_ok, "trainer blob SHA256 mismatch")
        finalization = provenance.get("finalization") or {}
        if type(finalization) is not dict:
            raise ValueError(f"{run}: run provenance finalization is not an exact object")
        selected_epoch = finalization.get("selected_epoch")
        completed_epochs_ok = ("completed_epochs" not in finalization or
            (exact_int(finalization.get("completed_epochs")) and finalization.get("completed_epochs") == 20))
        if (not exact_int(selected_epoch) or selected_epoch not in EPOCHS or
                report_selected_epoch != selected_epoch or not completed_epochs_ok):
            raise ValueError(f"{run}: training-report/provenance selected epoch is not exact or aligned")
        split = provenance.get("split_seed_contract") or {}
        split_integer_contract = {"training_seed": seed, "configured_split_seed": 174,
            "resolved_split_seed": 174, "clean_main_dev_rows": 720}
        split_ok = (all(exact_int(split.get(key)) and split.get(key) == value
                        for key, value in split_integer_contract.items()) and
                    split.get("split_seed_explicit") is True)
        identity_gate("run_seed_split_arm", {"seed": seed, "split_seed": 174, "arm": arm},
            {"seed": split.get("training_seed"), "split_seed": split.get("resolved_split_seed"), "arm": contract.get("arm")},
            split_ok and contract.get("arm") == arm, "run seed/split/arm mismatch")
        activity = ((provenance.get("data_provenance") or {}).get("auxiliary_activity") or {})
        required_inactive = ("stage57_active", "stage66_active", "stage75_active", "stage80a_active",
                             "time_swap_active", "external_evaluation_active")
        external_ok = (contract.get("external_data_used") is False and type(activity) is dict and
                       all(activity.get(key) is False for key in required_inactive))
        identity_gate("no_external_or_auxiliary_data", True, activity, external_ok, "external/auxiliary activity present")
        capsules = list(run_dir.glob("stage191_trajectory_state_epoch_*.pt"))
        identity_gate("zero_state_capsules", 0, len(capsules), len(capsules) == 0, "state capsule files present")

        trajectory_rows = read_jsonl(ledger_path)
        if len(trajectory_rows) != 20 or [row.get("epoch") for row in trajectory_rows] != list(EPOCHS) or any(not exact_int(row.get("epoch")) for row in trajectory_rows):
            raise ValueError(f"{run}: trajectory ledger is not exact epochs 1 through 20")
        trajectory = {row["epoch"]: row for row in trajectory_rows}
        export_names = {path.name for path in run_dir.iterdir() if path.is_file() and path.name.startswith("stage191_dev_predictions_epoch_")}
        expected_names = {f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl" for epoch in EPOCHS}
        if export_names != expected_names: raise ValueError(f"{run}: enumerated prediction export set mismatch")
        epochs: dict[int, list[dict[str, Any]]] = {}; reconstructed = {}
        run_golds: list[str] | None = None
        for epoch, expected_path_string in zip(EPOCHS, manifest["expected_prediction_export_paths"]):
            prediction_path = Path(expected_path_string).resolve(); row = trajectory[epoch]
            sha = file_sha256(prediction_path) if prediction_path.is_file() else None
            path_hash_ok = (prediction_path == (run_dir / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl").resolve() and
                Path(str(row.get("prediction_export_path", ""))).resolve() == prediction_path and
                re.fullmatch(r"[0-9a-f]{64}", str(row.get("prediction_export_sha256", ""))) is not None and
                sha == row.get("prediction_export_sha256"))
            if not path_hash_ok: raise ValueError(f"{run} epoch {epoch}: prediction path/SHA mismatch")
            if (not exact_int(row.get("epoch")) or row.get("epoch") != epoch or
                    not exact_int(row.get("best_epoch_before")) or row.get("best_epoch_before") not in range(0, epoch) or
                    not exact_int(row.get("best_epoch_after")) or row.get("best_epoch_after") not in range(1, epoch + 1)):
                raise ValueError(f"{run} epoch {epoch}: ledger integer identity/cardinality mismatch")
            prediction_rows = exported_rows(prediction_path, epoch, run)
            observed = metrics(prediction_rows, torch, float32_row_ce=True)
            metrics_ok = trajectory_metrics_exact(row, observed)
            golds = [item["gold_final_label"] for item in prediction_rows]
            if run_golds is None: run_golds = golds
            gold_ok = golds == run_golds
            if not metrics_ok or not gold_ok: raise ValueError(f"{run} epoch {epoch}: metric/gold reconstruction mismatch")
            tables["epochs"].append({"run": run, "epoch": epoch, "prediction_export_path": str(prediction_path),
                "prediction_export_sha256": sha, "row_count": len(prediction_rows), "dev_position_exact": True,
                "gold_alignment_exact": gold_ok, "metrics_exact": metrics_ok,
                "clean_ce_expected": row.get("clean_dev_ce"), "clean_ce_reconstructed": observed["clean_ce"],
                "passed": True, "blocking_reason": ""})
            epochs[epoch] = prediction_rows; reconstructed[epoch] = observed
        if global_golds is None: global_golds = run_golds
        if run_golds != global_golds: raise ValueError(f"{run}: cross-run gold alignment mismatch")
        data[run] = {"seed": seed, "arm": arm, "selected_epoch": selected_epoch,
                     "epochs": epochs, "metrics": reconstructed}
    return data

def evaluate_fresh(data: dict[str, Any], tables: dict[str, list[dict[str, Any]]], torch: Any) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    by_comparator: dict[str, list[dict[str, Any]]] = {name: [] for name in COMPARATORS}
    epoch_sets = {"tail2_mean_logits": (19, 20), "tail3_mean_logits": (18, 19, 20)}
    for comparator in COMPARATORS:
        for seed in SEEDS:
            pair_rows = {}; pair_metrics = {}; sources = {}
            for arm in ARMS:
                run = f"seed{seed}_{arm}"; item = data[run]
                if comparator == "independent_selected":
                    selected = item["selected_epoch"]; rows = item["epochs"][selected]
                    values = item["metrics"][selected]; source = selected
                else:
                    source = epoch_sets[comparator]
                    rows = ensemble_rows(item["epochs"], source)
                    values = metrics(rows, torch, float32_row_ce=False)
                pair_rows[arm] = rows; pair_metrics[arm] = values; sources[arm] = source
            row = pair_row(comparator, seed, pair_metrics["baseline"], pair_metrics["intervention"],
                           sources["baseline"], sources["intervention"])
            tables["by_seed"].append(row); by_comparator[comparator].append(row)
            tables["transitions"].append({"comparator": comparator, "seed": seed,
                                          **transition(pair_rows["baseline"], pair_rows["intervention"])})
            for gold in LABELS:
                tables["transition_gold"].append({"comparator": comparator, "seed": seed,
                    "gold_label": gold, **transition(pair_rows["baseline"], pair_rows["intervention"], gold)})
    aggregates = {name: aggregate(name, rows, pooled=False) for name, rows in by_comparator.items()}
    tables["fresh"].extend(aggregates[name] for name in COMPARATORS)
    return by_comparator, aggregates


def add_criteria(fresh_rows: dict[str, list[dict[str, Any]]], fresh: dict[str, dict[str, Any]],
                 pooled: dict[str, dict[str, Any]], tables: dict[str, list[dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
    independent, tail3 = fresh["independent_selected"], fresh["tail3_mean_logits"]
    pooled_independent, pooled_tail3 = pooled["independent_selected"], pooled["tail3_mean_logits"]
    results: dict[str, bool] = {}
    def criterion(group: str, name: str, required: Any, observed: Any, passed: bool) -> None:
        tables["criteria"].append({"criterion_group": group, "criterion": name,
            "required": required, "observed": observed, "passed": passed}); results[name] = passed
    criterion("fresh_quality", "fresh_macro_f1_quality", independent["mean_pair_macro_f1"] - .015,
              tail3["mean_pair_macro_f1"], tail3["mean_pair_macro_f1"] >= independent["mean_pair_macro_f1"] - .015)
    criterion("fresh_quality", "fresh_accuracy_quality", independent["mean_pair_accuracy"] - .01,
              tail3["mean_pair_accuracy"], tail3["mean_pair_accuracy"] >= independent["mean_pair_accuracy"] - .01)
    criterion("fresh_quality", "fresh_refute_bound", 1, tail3["max_abs_refute_delta"], tail3["max_abs_refute_delta"] <= 1)
    criterion("fresh_quality", "fresh_polarity_bound", 1, tail3["max_abs_polarity_delta"], tail3["max_abs_polarity_delta"] <= 1)
    criterion("range", "fresh_support_range", .65 * independent["support_delta_range"],
              tail3["support_delta_range"], tail3["support_delta_range"] <= .65 * independent["support_delta_range"])
    criterion("range", "fresh_false_entitlement_range", .65 * independent["false_entitlement_delta_range"],
              tail3["false_entitlement_delta_range"], tail3["false_entitlement_delta_range"] <= .65 * independent["false_entitlement_delta_range"])
    seed_pass_count = sum(abs(row["delta_pred_SUPPORT"]) <= 40 and abs(row["delta_false_entitlement"]) <= 25
                          for row in fresh_rows["tail3_mean_logits"])
    criterion("fresh_seed_level", "fresh_seed_level_support", 2, seed_pass_count, seed_pass_count >= 2)
    criterion("pooled_quality", "pooled_macro_f1_quality", pooled_independent["mean_pair_macro_f1"] - .015,
              pooled_tail3["mean_pair_macro_f1"], pooled_tail3["mean_pair_macro_f1"] >= pooled_independent["mean_pair_macro_f1"] - .015)
    criterion("pooled_quality", "pooled_accuracy_quality", pooled_independent["mean_pair_accuracy"] - .01,
              pooled_tail3["mean_pair_accuracy"], pooled_tail3["mean_pair_accuracy"] >= pooled_independent["mean_pair_accuracy"] - .01)
    criterion("range", "pooled_support_range", .70 * pooled_independent["support_delta_range"],
              pooled_tail3["support_delta_range"], pooled_tail3["support_delta_range"] <= .70 * pooled_independent["support_delta_range"])
    criterion("range", "pooled_false_entitlement_range", .70 * pooled_independent["false_entitlement_delta_range"],
              pooled_tail3["false_entitlement_delta_range"], pooled_tail3["false_entitlement_delta_range"] <= .70 * pooled_independent["false_entitlement_delta_range"])
    criterion("pooled_quality", "pooled_refute_bound", 1, pooled_tail3["max_abs_refute_delta"], pooled_tail3["max_abs_refute_delta"] <= 1)
    criterion("pooled_quality", "pooled_polarity_bound", 1, pooled_tail3["max_abs_polarity_delta"], pooled_tail3["max_abs_polarity_delta"] <= 1)

    fresh_quality_names = ("fresh_macro_f1_quality", "fresh_accuracy_quality", "fresh_refute_bound", "fresh_polarity_bound")
    pooled_quality_names = ("pooled_macro_f1_quality", "pooled_accuracy_quality", "pooled_refute_bound", "pooled_polarity_bound")
    range_names = ("fresh_support_range", "fresh_false_entitlement_range", "pooled_support_range", "pooled_false_entitlement_range")
    all_quality = all(results[name] for name in (*fresh_quality_names, *pooled_quality_names))
    range_pass_count = sum(results[name] for name in range_names)
    positive = all_quality and range_pass_count == 4 and results["fresh_seed_level_support"]
    partial = all_quality and not positive and range_pass_count >= 3
    decision = REPLICATED if positive else PARTIAL if partial else NOT_REPLICATED
    conditions = {
        REPLICATED: positive, PARTIAL: partial, NOT_REPLICATED: not positive and not partial,
        BLOCKED: False,
    }
    taxonomy = {
        "all_fresh_and_pooled_quality_passed": all_quality,
        "range_criteria_pass_count": range_pass_count,
        "fresh_seed_level_passed": results["fresh_seed_level_support"],
        "positive_conjunction": positive, "partial_conjunction": partial,
        "selected_decision": decision,
    }
    for alternative in (BLOCKED, REPLICATED, PARTIAL, NOT_REPLICATED):
        tables["decision"].append({"decision": alternative,
            "taxonomy_condition": "fail-closed exception only" if alternative == BLOCKED else "precommitted integrity-passing conjunction",
            "required": alternative == decision, "observed": conditions[alternative],
            "passed": (alternative == decision) == conditions[alternative]})
    if not all(row["passed"] for row in tables["decision"]): raise RuntimeError("decision taxonomy closure failed")
    return decision, taxonomy


def analyze(args: argparse.Namespace, tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    import torch
    repo, stage193a, run_root, stage192, _ = establish_safe_paths(args)
    identity = source_identity(repo, args.current_diagnostic_git_commit)
    if not identity["passed"]: raise ValueError("Stage193-C diagnostic source identity failed")
    stage192_report, old_rows = validate_stage192(stage192, tables["closure"])
    stage193a_report, manifests = load_stage193a(stage193a, run_root, args.current_diagnostic_git_commit)
    runtime_commit = stage193a_report["stage193_runtime_repository_commit"]
    trainer_blob_sha256 = stage193a_report["trainer_blob_sha256"]
    if identity["trainer"]["current_sha256"] != trainer_blob_sha256:
        raise ValueError("current trainer SHA256 differs from frozen Stage193-A trainer blob SHA256")
    data = load_fresh_runs(manifests, run_root, tables, torch, runtime_commit, trainer_blob_sha256)
    fresh_rows, fresh_aggregates = evaluate_fresh(data, tables, torch)
    pooled_rows = {name: [*old_rows[name], *fresh_rows[name]] for name in COMPARATORS}
    pooled_aggregates = {name: aggregate(name, pooled_rows[name], pooled=True) for name in COMPARATORS}
    tables["pooled"].extend(pooled_aggregates[name] for name in COMPARATORS)
    decision, taxonomy = add_criteria(fresh_rows, fresh_aggregates, pooled_aggregates, tables)
    recommendation = ({REPLICATED: "Design one mechanistically interpretable EMA/SWA-style temporal-stability intervention, but do not authorize it.",
        PARTIAL: "Diagnose the exact fresh seed and transition direction responsible for failure before designing an intervention.",
        NOT_REPLICATED: "Close post-hoc temporal logit smoothing and design a distinct interpretable SUPPORT/NOT_ENTITLED boundary mechanism."})[decision]
    return {"stage": "Stage193-C", "decision": decision, "runnable": True, "blocking_reasons": [],
        "diagnostic_only": True, "checkpoint_loaded": False, "model_loaded": False,
        "capsule_loaded": False, "external_data_used": False,
        "statistical_significance_claim": False, "model_advancement_decision": False,
        "subsequent_training_authorized": False, "current_diagnostic_git_commit": args.current_diagnostic_git_commit,
        "diagnostic_source_identity": identity, "stage193a_directory": str(stage193a),
        "stage193b_run_root": str(run_root), "stage192a_directory": str(stage192),
        "stage193a_decision": stage193a_report["decision"], "stage192a_decision": stage192_report["decision"],
        "trainer_blob_commit": TRAINER_BLOB_COMMIT,
        "trainer_blob_sha256": trainer_blob_sha256,
        "stage193_runtime_repository_commit": runtime_commit,
        "frozen_source_identities": {"trainer_blob_commit": TRAINER_BLOB_COMMIT,
            "stage192a_implementation_commit": STAGE192_COMMIT,
            "stage191b_replay_implementation_commit": STAGE191_COMMIT,
            "stage185_sidecar_semantic_sha256": SIDECAR_SHA},
        "canonical_labels": list(LABELS), "comparators": list(COMPARATORS),
        "primary_candidate": "tail3_mean_logits", "tail2_descriptive_only": True,
        "fresh_aggregates": fresh_aggregates, "pooled_descriptive_aggregates": pooled_aggregates,
        "decision_taxonomy": taxonomy, "recommended_next_stage": recommendation,
        "interpretation_restrictions": ["diagnostic only", "no checkpoint/model/capsule loading",
            "no external data", "no statistical-significance claim", "no model advancement",
            "no subsequent training authorization"], "exception": None}


def blocked_report(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {"stage": "Stage193-C", "decision": BLOCKED, "runnable": False,
        "blocking_reasons": [f"{type(exc).__name__}: {exc}"], "diagnostic_only": True,
        "checkpoint_loaded": False, "model_loaded": False, "capsule_loaded": False,
        "external_data_used": False, "statistical_significance_claim": False,
        "model_advancement_decision": False, "subsequent_training_authorized": False,
        "current_diagnostic_git_commit": args.current_diagnostic_git_commit,
        "trainer_blob_commit": TRAINER_BLOB_COMMIT, "trainer_blob_sha256": None,
        "stage193_runtime_repository_commit": args.current_diagnostic_git_commit,
        "recommended_next_stage": "Resolve the frozen-input or analysis failure; do not authorize training.",
        "interpretation_restrictions": ["diagnostic only", "no checkpoint/model/capsule loading",
            "no external data", "no statistical-significance claim", "no model advancement",
            "no subsequent training authorization"],
        "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Stage193-C tail3 fresh-seed replication report", "", f"Decision: `{report['decision']}`", "",
        f"- Runnable: {str(report['runnable']).lower()}", "- Diagnostic only: true",
        "- Checkpoint/model/capsule loading: none",
        f"- Trainer blob commit: `{report.get('trainer_blob_commit')}`",
        f"- Trainer blob SHA256: `{report.get('trainer_blob_sha256')}`",
        f"- Stage193 runtime repository commit: `{report.get('stage193_runtime_repository_commit')}`",
        "- External data used: false",
        "- Statistical-significance claim: false", "- Model advancement: false",
        "- Subsequent training authorized: false", "", "## Interpretation", ""]
    if report["decision"] == REPLICATED: lines.append("The precommitted fresh and pooled tail3 replication conjunction passed.")
    elif report["decision"] == PARTIAL: lines.append("Quality passed and at least three range criteria passed, but the full positive conjunction did not.")
    elif report["decision"] == NOT_REPLICATED: lines.append("Artifact integrity passed, but neither the positive nor partial conjunction passed.")
    else: lines.append("Interpretation is blocked by a frozen-input or analysis failure.")
    lines += ["", "Pooled six-seed results are descriptive and are not a statistical-significance claim.", "",
              "## Recommended next stage", "", report["recommended_next_stage"], ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        *_, output = establish_safe_paths(args)
    except BaseException:
        traceback.print_exc(file=sys.stderr); return 2
    output.mkdir(parents=False, exist_ok=True)
    tables = {name: [] for name in CSV_HEADERS}
    try:
        report = analyze(args, tables)
    except BaseException as exc:
        report = blocked_report(args, exc)
        tables["decision"].append({"decision": BLOCKED, "taxonomy_condition": "fail-closed exception only",
            "required": True, "observed": {"type": type(exc).__name__, "message": str(exc)}, "passed": True})
    write_json(output / OUTPUTS["json"], report)
    (output / OUTPUTS["md"]).write_text(markdown(report), encoding="utf-8")
    for name, header in CSV_HEADERS.items(): write_csv(output / OUTPUTS[name], header, tables[name])
    return 0 if report["runnable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
