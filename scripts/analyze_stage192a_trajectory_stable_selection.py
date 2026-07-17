#!/usr/bin/env python3
"""Diagnose paired trajectory-stable checkpoint selection from frozen exports only."""
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


STAGE191B_COMMIT = "0872e66ccb05ae8a166f5cabf4e084272dc49500"
STAGE191D_COMMIT = "08ae49a79148ca448340c1948b5c9991b6919f04"
STAGE191B_DIRNAME = "stage191b_deterministic_replay_manifest_20260717_153524"
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
SEEDS = (174, 175, 176)
ARMS = ("baseline", "intervention")
RUNS = tuple(f"seed{seed}_{arm}" for seed in SEEDS for arm in ARMS)
EPOCHS = tuple(range(1, 21))
DOMAIN = tuple(range(15, 21))
HISTORICAL = {
    "seed174_baseline": 20, "seed174_intervention": 20,
    "seed175_baseline": 20, "seed175_intervention": 19,
    "seed176_baseline": 20, "seed176_intervention": 19,
}
SELECTORS = (
    "historical_independent", "sync_epoch19", "sync_epoch20", "sync_mean_macro",
    "sync_min_macro", "sync_mean_ce", "sync_stability_constrained",
)
COMPARATORS = {"tail2_mean_logits": (19, 20), "tail3_mean_logits": (18, 19, 20)}
BLOCKED = "STAGE192A_TRAJECTORY_SELECTION_DIAGNOSTIC_BLOCKED"
IDENTIFIED = "STAGE192A_TRAJECTORY_STABLE_PAIR_SELECTOR_IDENTIFIED"
TRADEOFF = "STAGE192A_STABILITY_GAIN_WITH_QUALITY_TRADEOFF_ONLY"
NONE = "STAGE192A_NO_TRAJECTORY_STABLE_SELECTOR"
CONFIRMED191D = "STAGE191D_LATE_SUPPORT_NE_PHASE_FLIP_CONFIRMED"
CE_CONTRACT = {"row_source": "ordered epoch export final_ce", "row_count": 720,
               "dtype": "torch.float32", "device": "cpu", "reduction": "mean",
               "comparison": "exact equality"}
METRICS = ("clean_dev_ce", "clean_accuracy", "clean_macro_f1", "support_recall",
           "false_entitlement_total", "false_not_entitled_total", "polarity_error_total",
           "pred_REFUTE", "pred_NOT_ENTITLED", "pred_SUPPORT")
MATRIX_FIELDS = tuple(f"{a}_to_{b}" for a in LABELS for b in LABELS)
TRANSITION_FIELDS = (*MATRIX_FIELDS, "unchanged_rows", "changed_rows", "not_entitled_to_support",
                     "support_to_not_entitled", "refute_involved_transitions",
                     "exclusive_not_entitled_support_changed_rows",
                     "exclusive_not_entitled_support_fraction_of_changed", "corrections",
                     "regressions", "wrong_to_different_wrong")

OUTPUTS = {
    "json": "stage192a_trajectory_stable_selection_report.json",
    "md": "stage192a_trajectory_stable_selection_report.md",
    "closure": "stage192a_stage191d_closure_gate.csv",
    "definitions": "stage192a_selector_definition.csv",
    "choices": "stage192a_selector_choice_by_seed.csv",
    "arms": "stage192a_selected_arm_metrics.csv",
    "aggregates": "stage192a_selector_aggregate_metrics.csv",
    "deltas": "stage192a_pair_delta_by_selector.csv",
    "grid": "stage192a_perturbation_grid.csv",
    "perturb": "stage192a_perturbation_summary.csv",
    "transitions": "stage192a_selected_pair_transition_summary.csv",
    "transition_gold": "stage192a_selected_pair_transition_by_gold.csv",
    "ensembles": "stage192a_temporal_ensemble_comparator.csv",
    "gates": "stage192a_precommitted_gate.csv",
}
STAGE191D_OUTPUTS = {
    "stage191d_trajectory_phase_flip_report.json", "stage191d_trajectory_phase_flip_report.md",
    "stage191d_stage191c_equivalence_gate.csv", "stage191d_run_epoch_metrics.csv",
    "stage191d_pair_delta_by_epoch.csv", "stage191d_late_sign_reversal.csv",
    "stage191d_within_run_transition_summary.csv", "stage191d_within_run_transition_by_gold.csv",
    "stage191d_paired_transition_summary.csv", "stage191d_paired_transition_by_gold.csv",
    "stage191d_late_instability_metrics.csv", "stage191d_state_step_summary.csv",
    "stage191d_parameter_group_step.csv", "stage191d_precommitted_gate.csv",
}

CSV_HEADERS = {
    "closure": ["gate", "required", "observed", "passed", "blocking_reason"],
    "definitions": ["selector", "reference", "winning_eligible", "synchronous", "definition", "tie_break"],
    "choices": ["selector", "seed", "available", "synchronous_selection", "baseline_epoch", "intervention_epoch", "selection_evidence"],
    "arms": ["selector", "seed", "arm", "selected_epoch", "available", *METRICS],
    "aggregates": ["selector", "available_all_seeds", "synchronous_all_seeds", "mean_clean_dev_ce", "mean_clean_accuracy", "mean_clean_macro_f1", "mean_support_recall", "total_false_entitlement", "mean_false_entitlement", "total_false_not_entitled", "mean_false_not_entitled", "total_polarity_error", "joint_phase_flip_seed_count", "mean_support_delta_range", "mean_false_entitlement_delta_range", "max_abs_refute_delta", "max_abs_polarity_delta", "quality_preserving", "quality_tradeoff"],
    "deltas": ["selector", "seed", "available", "baseline_epoch", "intervention_epoch", *[f"delta_{m}" for m in METRICS]],
    "grid": ["selector", "seed", "baseline_epoch", "intervention_epoch", "delta_pred_SUPPORT", "delta_pred_NOT_ENTITLED", "delta_pred_REFUTE", "delta_support_recall", "delta_false_entitlement_total", "delta_false_not_entitled_total", "delta_polarity_error_total", "pair_mean_macro_f1", "pair_mean_accuracy", "pair_mean_clean_dev_ce"],
    "perturb": ["selector", "seed", "available", "cell_count", "support_delta_min", "support_delta_max", "support_delta_range", "false_entitlement_delta_min", "false_entitlement_delta_max", "false_entitlement_delta_range", "support_nonzero_sign_set", "false_entitlement_nonzero_sign_set", "support_sign_flip_present", "false_entitlement_sign_flip_present", "joint_phase_flip_present", "max_abs_refute_delta", "max_abs_polarity_delta", "min_pair_mean_macro_f1", "max_pair_mean_clean_dev_ce"],
    "transitions": ["selector", "seed", "baseline_epoch", "intervention_epoch", *TRANSITION_FIELDS],
    "transition_gold": ["selector", "seed", "baseline_epoch", "intervention_epoch", "gold_label", *TRANSITION_FIELDS],
    "ensembles": ["comparator", "seed", "epochs", *[f"baseline_{m}" for m in METRICS], *[f"intervention_{m}" for m in METRICS], *[f"delta_{m}" for m in METRICS], *TRANSITION_FIELDS],
    "gates": ["selector", "gate", "required", "observed", "passed", "criterion_class"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage191b-dir", type=Path, required=True)
    parser.add_argument("--stage191d-dir", type=Path, required=True)
    parser.add_argument("--current-diagnostic-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def exact_int(value: Any) -> bool:
    return type(value) is int


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL line")
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError(f"{path}:{number}: row is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        if isinstance(value, set):
            value = sorted(value)
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in headers})


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def establish_safe_output(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    repo = args.repo_root.resolve()
    if not repo.is_dir():
        raise ValueError("repo root is not a directory")
    reports = (repo / "reports").resolve()
    stage191b = args.stage191b_dir.resolve()
    stage191d = args.stage191d_dir.resolve()
    if stage191b != (reports / STAGE191B_DIRNAME).resolve() or not stage191b.is_dir():
        raise ValueError("Stage191-B directory is not the exact frozen directory")
    if not stage191d.is_dir():
        raise ValueError("explicit Stage191-D directory is not an existing directory")
    output = args.output_dir.resolve()
    if output.parent != reports or not output.name.startswith("stage192a_trajectory_stable_selection_"):
        raise ValueError("output must be an immediate reports child with the Stage192-A prefix")
    for frozen in (stage191b, stage191d):
        if output == frozen or frozen in output.parents:
            raise ValueError("output is a frozen input or its descendant")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output exists and is not an empty directory")
    return repo, stage191b, stage191d, output


def git_call(repo: Path, arguments: list[str], *, binary: bool = False, dirty: bool = False) -> Any:
    result = subprocess.run(["git", *arguments], cwd=repo, check=False, capture_output=True, shell=False)
    if dirty:
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        return result.returncode
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict").strip()


def source_identity(repo: Path, commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", commit or "") is None:
        raise ValueError("current diagnostic commit must be lowercase 40-character hexadecimal")
    head = git_call(repo, ["rev-parse", "HEAD"])
    files = {}
    for relative in ("scripts/analyze_stage192a_trajectory_stable_selection.py",
                     "reports/stage192a_trajectory_stable_selection_spec.md"):
        current = (repo / relative).read_bytes()
        blob = git_call(repo, ["show", f"{commit}:{relative}"], binary=True)
        files[relative] = {
            "current_sha256": hashlib.sha256(current).hexdigest(),
            "commit_blob_sha256": hashlib.sha256(blob).hexdigest(),
            "bytes_equal": current == blob,
            "unstaged_clean": git_call(repo, ["diff", "--quiet", "--", relative], dirty=True) == 0,
            "staged_clean": git_call(repo, ["diff", "--cached", "--quiet", "--", relative], dirty=True) == 0,
        }
    passed = head == commit and all(x["bytes_equal"] and x["unstaged_clean"] and x["staged_clean"] for x in files.values())
    return {"supplied_commit": commit, "repository_head": head, "files": files, "passed": passed}


def bool_csv(value: str) -> bool | None:
    if value.strip().lower() == "true":
        return True
    if value.strip().lower() == "false":
        return False
    return None


def validate_stage191d(stage191d: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    entries = {path.name for path in stage191d.iterdir()}
    exact = entries == STAGE191D_OUTPUTS and all((stage191d / name).is_file() for name in STAGE191D_OUTPUTS)
    rows.append({"gate": "exact_authoritative_14_outputs", "required": sorted(STAGE191D_OUTPUTS),
                 "observed": sorted(entries), "passed": exact,
                 "blocking_reason": "Stage191-D directory does not contain exactly the authoritative 14 files" if not exact else ""})
    if not exact:
        raise ValueError("Stage191-D exact output set mismatch")
    report = read_json(stage191d / "stage191d_trajectory_phase_flip_report.json")
    required = {
        "decision": CONFIRMED191D, "runnable": True, "blocking_reasons": [],
        "stage191c_equivalence_passed": True, "current_diagnostic_git_commit": STAGE191D_COMMIT,
        "source_contract_passed": True, "diagnostic_only": True, "training_performed": False,
        "model_constructed": False, "model_advancement_decision": False, "external_data_used": False,
        "stage192_training_authorized": False, "clean_dev_ce_reduction_contract": CE_CONTRACT,
    }
    observed = {key: report.get(key) for key in required}
    contract_ok = observed == required
    rows.append({"gate": "stage191d_report_closure", "required": required, "observed": observed,
                 "passed": contract_ok, "blocking_reason": "Stage191-D report closure mismatch" if not contract_ok else ""})
    if not contract_ok:
        raise ValueError("Stage191-D report closure mismatch")
    for filename, gate in (("stage191d_stage191c_equivalence_gate.csv", "no_failed_stage191c_equivalence_gate"),
                           ("stage191d_precommitted_gate.csv", "no_failed_universally_required_precommitted_gate")):
        headers, data = read_csv(stage191d / filename)
        if "passed" not in headers:
            raise ValueError(f"{filename}: missing passed column")
        relevant = data
        if filename.endswith("precommitted_gate.csv"):
            relevant = [row for row in data if row.get("required") not in ("decision_alternative", "False", "false")]
        ok = bool(relevant) and all(bool_csv(row.get("passed", "")) is True for row in relevant)
        rows.append({"gate": gate, "required": True, "observed": {"row_count": len(relevant), "all_passed": ok},
                     "passed": ok, "blocking_reason": f"{filename} contains a failed universally required gate" if not ok else ""})
        if not ok:
            raise ValueError(f"{filename}: failed required gate")
    return report


def option_map(argv: Any) -> dict[str, Any]:
    if type(argv) is not list or any(type(token) is not str for token in argv):
        raise ValueError("argv must be a string list")
    result, index = {}, 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--") or "=" in token:
            raise ValueError(f"unsupported argv token {token!r}")
        key = token[2:].replace("-", "_")
        if key in result:
            raise ValueError(f"duplicate argv option {token}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            result[key], index = argv[index + 1], index + 2
        else:
            result[key], index = True, index + 1
    return result


def validate_internal_argv(argv: Any, label: str) -> None:
    options = option_map(argv)
    forbidden = ("ood_data", "output_ood_json", "output_ood_predictions_json", "external_data",
                 "external_output_dir", "external_eval_jsonl", "external_eval_name",
                 "stage43_external_factver_jsonl", "stage57_bridge_train_jsonl",
                 "stage66_bridge_train_jsonl", "stage75_bridge_train_jsonl", "stage80a_bridge_train_jsonl")
    flags = ("enable_external_eval", "enable_stage43_external_eval", "stage43_external_enable_shadow_export")
    modes = ("stage57_bridge_train_mode", "stage66_bridge_train_mode", "stage75_bridge_train_mode", "stage80a_bridge_train_mode")
    if any(k in options for k in forbidden + flags) or any(k in options and options[k] != "none" for k in modes):
        raise ValueError(f"{label}: external/OOD/bridge use is present")


def exported_rows(path: Path, epoch: int, run: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != 720:
        raise ValueError(f"{run} epoch {epoch}: expected 720 rows")
    for position, row in enumerate(rows):
        if not exact_int(row.get("epoch")) or row["epoch"] != epoch or not exact_int(row.get("dev_position")) or row["dev_position"] != position:
            raise ValueError(f"{run} epoch {epoch}: epoch/dev_position mismatch")
        if row.get("gold_final_label") not in LABELS or row.get("predicted_final_label") not in LABELS:
            raise ValueError(f"{run} epoch {epoch}: noncanonical label")
        logits = row.get("final_logits")
        if type(logits) is not list or len(logits) != 3 or any(not finite(x) for x in logits):
            raise ValueError(f"{run} epoch {epoch}: invalid final_logits")
        if not finite(row.get("final_ce")):
            raise ValueError(f"{run} epoch {epoch}: invalid final_ce")
    return rows


def metrics(rows: list[dict[str, Any]], torch: Any, *, ce_from_rows: bool = True) -> dict[str, Any]:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows:
        matrix[row["gold_final_label"]][row["predicted_final_label"]] += 1
    gold_counts = {gold: sum(matrix[gold].values()) for gold in LABELS}
    counts = {pred: sum(matrix[gold][pred] for gold in LABELS) for pred in LABELS}
    f1s = []
    for label in LABELS:
        tp, predicted, gold = matrix[label][label], counts[label], gold_counts[label]
        precision, recall = (tp / predicted if predicted else 0.0), (tp / gold if gold else 0.0)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    ce_values = [float(row["final_ce"]) for row in rows]
    ce = torch.tensor(ce_values, dtype=torch.float32, device="cpu").mean().item() if ce_from_rows else sum(ce_values) / len(ce_values)
    return {"clean_dev_ce": ce, "clean_accuracy": sum(matrix[x][x] for x in LABELS) / 720,
            "clean_macro_f1": sum(f1s) / 3, "support_recall": matrix["SUPPORT"]["SUPPORT"] / gold_counts["SUPPORT"],
            "false_entitlement_total": matrix["NOT_ENTITLED"]["REFUTE"] + matrix["NOT_ENTITLED"]["SUPPORT"],
            "false_not_entitled_total": matrix["REFUTE"]["NOT_ENTITLED"] + matrix["SUPPORT"]["NOT_ENTITLED"],
            "polarity_error_total": matrix["REFUTE"]["SUPPORT"] + matrix["SUPPORT"]["REFUTE"],
            **{f"pred_{label}": counts[label] for label in LABELS}}


def close(left: Any, right: Any) -> bool:
    return finite(left) and finite(right) and math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-7)


def load_replays(repo: Path, stage191b: Path, torch: Any) -> dict[str, Any]:
    main = read_json(stage191b / "stage191b_deterministic_replay_manifest_report.json")
    if main.get("decision") != "STAGE191B_DETERMINISTIC_REPLAY_MANIFEST_READY" or main.get("runnable") is not True or main.get("blocking_reasons") != [] or main.get("external_data_used") is not False:
        raise ValueError("Stage191-B main manifest closure mismatch")
    if (main.get("commit_identities") or {}).get("stage191b_replay_commit") != STAGE191B_COMMIT:
        raise ValueError("Stage191-B commit mismatch")
    data, gold_reference = {}, None
    for seed in SEEDS:
        for arm in ARMS:
            run = f"seed{seed}_{arm}"
            run_dir = (stage191b / run).resolve()
            manifest = read_json(stage191b / f"stage191b_{run}_replay_manifest.json")
            required = {"run": run, "seed": seed, "training_seed": seed, "split_seed": 174, "arm": arm,
                        "runnable": True, "blocking_reasons": [], "diagnostic_replay_only": True,
                        "training_for_model_advancement_authorized": False, "model_advancement_decision": False,
                        "external_data_used": False, "original_selected_epoch": HISTORICAL[run],
                        "expected_trajectory_rows": 20, "expected_prediction_rows_per_epoch": 720}
            if any(manifest.get(k) != v for k, v in required.items()) or any(not exact_int(manifest.get(k)) for k in ("seed", "training_seed", "split_seed", "original_selected_epoch")):
                raise ValueError(f"{run}: manifest identity/authorization mismatch")
            if (manifest.get("commit_identities") or {}).get("stage191b_replay_commit") != STAGE191B_COMMIT:
                raise ValueError(f"{run}: replay commit mismatch")
            if Path(str(manifest.get("replay_output_directory", ""))).resolve() != run_dir:
                raise ValueError(f"{run}: replay output path mismatch")
            validate_internal_argv(manifest.get("original_argv"), f"{run} original argv")
            validate_internal_argv(manifest.get("argv"), f"{run} replay argv")
            contract = read_json(run_dir / "stage191_trajectory_contract.json")
            if contract.get("trainer_source_commit") != STAGE191B_COMMIT or contract.get("training_seed") != seed or contract.get("split_seed") != 174 or contract.get("arm") != arm or contract.get("canonical_logit_column_labels") != list(LABELS) or contract.get("external_data_used") is not False or contract.get("training_semantics_changed") is not False or contract.get("extra_forward_pass_performed") is not False or contract.get("loss_logits_used") is not False:
                raise ValueError(f"{run}: trajectory contract mismatch")
            trajectory_rows = read_jsonl(run_dir / "stage191_trajectory_epoch_metrics.jsonl")
            if len(trajectory_rows) != 20 or [row.get("epoch") for row in trajectory_rows] != list(EPOCHS) or any(not exact_int(row.get("epoch")) for row in trajectory_rows):
                raise ValueError(f"{run}: trajectory is not exact epochs 1 through 20")
            trajectory = {row["epoch"]: row for row in trajectory_rows}
            export_names = {p.name for p in run_dir.iterdir() if p.is_file() and p.name.startswith("stage191_dev_predictions_epoch_")}
            expected_exports = {f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl" for epoch in EPOCHS}
            if export_names != expected_exports:
                raise ValueError(f"{run}: prediction export file set mismatch")
            predictions, reconstructed = {}, {}
            for epoch in EPOCHS:
                path = (run_dir / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl").resolve()
                tr = trajectory[epoch]
                if Path(str(tr.get("prediction_export_path", ""))).resolve() != path or not re.fullmatch(r"[0-9a-f]{64}", str(tr.get("prediction_export_sha256", ""))) or file_sha256(path) != tr.get("prediction_export_sha256"):
                    raise ValueError(f"{run} epoch {epoch}: prediction path/SHA mismatch")
                prediction_rows = exported_rows(path, epoch, run)
                observed = metrics(prediction_rows, torch)
                counts = tr.get("normalized_prediction_counts")
                if type(counts) is not dict or set(counts) != set(LABELS) or any(not exact_int(counts[x]) for x in LABELS):
                    raise ValueError(f"{run} epoch {epoch}: normalized counts schema mismatch")
                exact_fields = ("false_entitlement_total", "false_not_entitled_total", "polarity_error_total")
                if observed["clean_dev_ce"] != tr.get("clean_dev_ce") or any(observed[f"pred_{x}"] != counts[x] for x in LABELS) or any(not exact_int(tr.get(x)) or observed[x] != tr.get(x) for x in exact_fields) or any(not close(observed[x], tr.get(x)) for x in ("clean_accuracy", "clean_macro_f1", "support_recall")):
                    raise ValueError(f"{run} epoch {epoch}: reconstructed clean metrics mismatch")
                predictions[epoch], reconstructed[epoch] = prediction_rows, observed
            golds = [row["gold_final_label"] for row in predictions[1]]
            if any([row["gold_final_label"] for row in predictions[e]] != golds for e in EPOCHS):
                raise ValueError(f"{run}: gold labels change across epochs")
            if gold_reference is None:
                gold_reference = golds
            elif gold_reference != golds:
                raise ValueError(f"{run}: gold alignment differs across runs")
            training = read_json(run_dir / "training_report.json")
            single = (training.get("runs") or {}).get("single")
            if type(single) is not dict or single.get("best_epoch") != HISTORICAL[run] or single.get("final_epoch") != 20 or not exact_int(single.get("best_epoch")) or not exact_int(single.get("final_epoch")):
                raise ValueError(f"{run}: selected/final epoch mismatch")
            data[run] = {"seed": seed, "arm": arm, "trajectory": trajectory, "predictions": predictions, "metrics": reconstructed}
    return data


def sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def transition(previous: list[dict[str, Any]], nxt: list[dict[str, Any]], gold_filter: str | None = None) -> dict[str, Any]:
    matrix = {a: {b: 0 for b in LABELS} for a in LABELS}
    corrections = regressions = wrong = 0
    for position, (before, after) in enumerate(zip(previous, nxt)):
        if before.get("dev_position") != position or after.get("dev_position") != position or before.get("gold_final_label") != after.get("gold_final_label"):
            raise ValueError("transition alignment mismatch")
        gold = before["gold_final_label"]
        if gold_filter is not None and gold != gold_filter:
            continue
        old, new = before["predicted_final_label"], after["predicted_final_label"]
        matrix[old][new] += 1
        corrections += old != gold and new == gold
        regressions += old == gold and new != gold
        wrong += old != gold and new != gold and old != new
    total = sum(sum(x.values()) for x in matrix.values())
    unchanged = sum(matrix[x][x] for x in LABELS)
    changed = total - unchanged
    exclusive = matrix["NOT_ENTITLED"]["SUPPORT"] + matrix["SUPPORT"]["NOT_ENTITLED"]
    return {**{f"{a}_to_{b}": matrix[a][b] for a in LABELS for b in LABELS},
            "unchanged_rows": unchanged, "changed_rows": changed,
            "not_entitled_to_support": matrix["NOT_ENTITLED"]["SUPPORT"],
            "support_to_not_entitled": matrix["SUPPORT"]["NOT_ENTITLED"],
            "refute_involved_transitions": sum(matrix[a][b] for a in LABELS for b in LABELS if a != b and "REFUTE" in (a, b)),
            "exclusive_not_entitled_support_changed_rows": exclusive,
            "exclusive_not_entitled_support_fraction_of_changed": exclusive / changed if changed else None,
            "corrections": corrections, "regressions": regressions, "wrong_to_different_wrong": wrong}


def pair_values(data: dict[str, Any], seed: int, be: int, ie: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline = data[f"seed{seed}_baseline"]["metrics"][be]
    intervention = data[f"seed{seed}_intervention"]["metrics"][ie]
    delta = {f"delta_{metric}": intervention[metric] - baseline[metric] for metric in METRICS}
    return baseline, intervention, delta


def boundary_churn(data: dict[str, Any], seed: int, epochs: list[int]) -> float:
    count = 0
    for arm in ARMS:
        predictions = data[f"seed{seed}_{arm}"]["predictions"]
        for left, right in zip(epochs, epochs[1:]):
            for a, b in zip(predictions[left], predictions[right]):
                if {a["predicted_final_label"], b["predicted_final_label"]} == {"NOT_ENTITLED", "SUPPORT"}:
                    count += 1
    return count / 720


def choose_selectors(data: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> dict[str, dict[int, Any]]:
    definitions = {
        "historical_independent": (True, False, "frozen historical arm epochs", "none"),
        "sync_epoch19": (False, True, "epoch 19 for both arms", "none"),
        "sync_epoch20": (False, True, "epoch 20 for both arms", "none"),
        "sync_mean_macro": (False, True, "maximize pair mean macro-F1", "max min-arm macro-F1; min mean CE; later"),
        "sync_min_macro": (False, True, "maximize minimum-arm macro-F1", "max mean macro-F1; min mean CE; later"),
        "sync_mean_ce": (False, True, "minimize pair mean clean CE", "max mean macro-F1; max min-arm macro-F1; later"),
        "sync_stability_constrained": (False, True, "quality-eligible local paired stability tuple", "precommitted seven-key tuple"),
    }
    for name, (reference, synchronous, definition, tie) in definitions.items():
        tables["definitions"].append({"selector": name, "reference": reference, "winning_eligible": not reference,
                                      "synchronous": synchronous, "definition": definition, "tie_break": tie})
    choices: dict[str, dict[int, Any]] = {name: {} for name in SELECTORS}
    for seed in SEEDS:
        base_run, int_run = f"seed{seed}_baseline", f"seed{seed}_intervention"
        pairs = {}
        for epoch in DOMAIN:
            b, i = data[base_run]["metrics"][epoch], data[int_run]["metrics"][epoch]
            pairs[epoch] = {"mean_macro": (b["clean_macro_f1"] + i["clean_macro_f1"]) / 2,
                            "min_macro": min(b["clean_macro_f1"], i["clean_macro_f1"]),
                            "mean_ce": (b["clean_dev_ce"] + i["clean_dev_ce"]) / 2}
        selected = {
            "historical_independent": (HISTORICAL[base_run], HISTORICAL[int_run], {}),
            "sync_epoch19": (19, 19, {}), "sync_epoch20": (20, 20, {}),
            "sync_mean_macro": (max(DOMAIN, key=lambda e: (pairs[e]["mean_macro"], pairs[e]["min_macro"], -pairs[e]["mean_ce"], e)),) * 2 + ({},),
            "sync_min_macro": (max(DOMAIN, key=lambda e: (pairs[e]["min_macro"], pairs[e]["mean_macro"], -pairs[e]["mean_ce"], e)),) * 2 + ({},),
            "sync_mean_ce": (min(DOMAIN, key=lambda e: (pairs[e]["mean_ce"], -pairs[e]["mean_macro"], -pairs[e]["min_macro"], -e)),) * 2 + ({},),
        }
        best = max(pairs[e]["mean_macro"] for e in DOMAIN)
        historical_b = data[base_run]["metrics"][HISTORICAL[base_run]]["clean_macro_f1"]
        historical_i = data[int_run]["metrics"][HISTORICAL[int_run]]["clean_macro_f1"]
        eligible = [e for e in DOMAIN if pairs[e]["mean_macro"] >= best - .005 and data[base_run]["metrics"][e]["clean_macro_f1"] >= historical_b - .01 and data[int_run]["metrics"][e]["clean_macro_f1"] >= historical_i - .01]
        stability = {}
        for epoch in eligible:
            neighborhood = [e for e in (epoch - 1, epoch, epoch + 1) if e in DOMAIN]
            support, false_e = [], []
            for e in neighborhood:
                _, _, delta = pair_values(data, seed, e, e)
                support.append(delta["delta_pred_SUPPORT"])
                false_e.append(delta["delta_false_entitlement_total"])
            support_signs, false_signs = {sign(x) for x in support if x}, {sign(x) for x in false_e if x}
            stability[epoch] = (int(support_signs == {-1, 1}) + int(false_signs == {-1, 1}),
                                max(support) - min(support), max(false_e) - min(false_e),
                                boundary_churn(data, seed, neighborhood), -pairs[epoch]["mean_macro"],
                                pairs[epoch]["mean_ce"], -epoch)
        if eligible:
            epoch = min(eligible, key=lambda e: stability[e])
            selected["sync_stability_constrained"] = (epoch, epoch, {"eligible_epochs": eligible, "winning_tuple": stability[epoch]})
        else:
            selected["sync_stability_constrained"] = (None, None, {"eligible_epochs": []})
        for selector in SELECTORS:
            be, ie, evidence = selected[selector]
            available = be is not None and ie is not None
            choice = {"baseline_epoch": be, "intervention_epoch": ie, "available": available,
                      "synchronous_selection": available and be == ie, "selection_evidence": evidence}
            choices[selector][seed] = choice
            tables["choices"].append({"selector": selector, "seed": seed, **choice})
    return choices


def evaluate_selectors(data: dict[str, Any], choices: dict[str, dict[int, Any]], tables: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    pair_deltas: dict[tuple[str, int], dict[str, Any]] = {}
    for selector in SELECTORS:
        for seed in SEEDS:
            choice = choices[selector][seed]
            if not choice["available"]:
                tables["arms"].extend({"selector": selector, "seed": seed, "arm": arm, "selected_epoch": None, "available": False} for arm in ARMS)
                tables["deltas"].append({"selector": selector, "seed": seed, "available": False})
                tables["perturb"].append({"selector": selector, "seed": seed, "available": False, "cell_count": 0})
                summaries[selector, seed] = {"available": False}
                continue
            be, ie = choice["baseline_epoch"], choice["intervention_epoch"]
            baseline, intervention, delta = pair_values(data, seed, be, ie)
            for arm, epoch, values in (("baseline", be, baseline), ("intervention", ie, intervention)):
                tables["arms"].append({"selector": selector, "seed": seed, "arm": arm, "selected_epoch": epoch, "available": True, **values})
            delta_row = {"selector": selector, "seed": seed, "available": True, "baseline_epoch": be, "intervention_epoch": ie, **delta}
            tables["deltas"].append(delta_row)
            pair_deltas[selector, seed] = delta_row
            base_rows = data[f"seed{seed}_baseline"]["predictions"][be]
            int_rows = data[f"seed{seed}_intervention"]["predictions"][ie]
            tables["transitions"].append({"selector": selector, "seed": seed, "baseline_epoch": be, "intervention_epoch": ie, **transition(base_rows, int_rows)})
            for gold in LABELS:
                tables["transition_gold"].append({"selector": selector, "seed": seed, "baseline_epoch": be, "intervention_epoch": ie, "gold_label": gold, **transition(base_rows, int_rows, gold)})
            bes = [e for e in (be - 1, be, be + 1) if e in DOMAIN]
            ies = [e for e in (ie - 1, ie, ie + 1) if e in DOMAIN]
            cells = []
            for pb in bes:
                for pi in ies:
                    b, i, d = pair_values(data, seed, pb, pi)
                    cell = {"selector": selector, "seed": seed, "baseline_epoch": pb, "intervention_epoch": pi,
                            "delta_pred_SUPPORT": d["delta_pred_SUPPORT"], "delta_pred_NOT_ENTITLED": d["delta_pred_NOT_ENTITLED"],
                            "delta_pred_REFUTE": d["delta_pred_REFUTE"], "delta_support_recall": d["delta_support_recall"],
                            "delta_false_entitlement_total": d["delta_false_entitlement_total"],
                            "delta_false_not_entitled_total": d["delta_false_not_entitled_total"],
                            "delta_polarity_error_total": d["delta_polarity_error_total"],
                            "pair_mean_macro_f1": (b["clean_macro_f1"] + i["clean_macro_f1"]) / 2,
                            "pair_mean_accuracy": (b["clean_accuracy"] + i["clean_accuracy"]) / 2,
                            "pair_mean_clean_dev_ce": (b["clean_dev_ce"] + i["clean_dev_ce"]) / 2}
                    cells.append(cell); tables["grid"].append(cell)
            support = [x["delta_pred_SUPPORT"] for x in cells]
            false_e = [x["delta_false_entitlement_total"] for x in cells]
            ss, fs = {sign(x) for x in support if x}, {sign(x) for x in false_e if x}
            summary = {"selector": selector, "seed": seed, "available": True, "cell_count": len(cells),
                       "support_delta_min": min(support), "support_delta_max": max(support), "support_delta_range": max(support) - min(support),
                       "false_entitlement_delta_min": min(false_e), "false_entitlement_delta_max": max(false_e), "false_entitlement_delta_range": max(false_e) - min(false_e),
                       "support_nonzero_sign_set": sorted(ss), "false_entitlement_nonzero_sign_set": sorted(fs),
                       "support_sign_flip_present": ss == {-1, 1}, "false_entitlement_sign_flip_present": fs == {-1, 1},
                       "joint_phase_flip_present": ss == {-1, 1} and fs == {-1, 1},
                       "max_abs_refute_delta": max(abs(x["delta_pred_REFUTE"]) for x in cells),
                       "max_abs_polarity_delta": max(abs(x["delta_polarity_error_total"]) for x in cells),
                       "min_pair_mean_macro_f1": min(x["pair_mean_macro_f1"] for x in cells),
                       "max_pair_mean_clean_dev_ce": max(x["pair_mean_clean_dev_ce"] for x in cells)}
            tables["perturb"].append(summary); summaries[selector, seed] = summary
    aggregates = {}
    for selector in SELECTORS:
        available = all(choices[selector][s]["available"] for s in SEEDS)
        synchronous = available and all(choices[selector][s]["synchronous_selection"] for s in SEEDS)
        arm_rows = [row for row in tables["arms"] if row["selector"] == selector and row.get("available")]
        if available:
            agg = {"selector": selector, "available_all_seeds": True, "synchronous_all_seeds": synchronous,
                   "mean_clean_dev_ce": sum(r["clean_dev_ce"] for r in arm_rows) / 6,
                   "mean_clean_accuracy": sum(r["clean_accuracy"] for r in arm_rows) / 6,
                   "mean_clean_macro_f1": sum(r["clean_macro_f1"] for r in arm_rows) / 6,
                   "mean_support_recall": sum(r["support_recall"] for r in arm_rows) / 6,
                   "total_false_entitlement": sum(r["false_entitlement_total"] for r in arm_rows),
                   "mean_false_entitlement": sum(r["false_entitlement_total"] for r in arm_rows) / 6,
                   "total_false_not_entitled": sum(r["false_not_entitled_total"] for r in arm_rows),
                   "mean_false_not_entitled": sum(r["false_not_entitled_total"] for r in arm_rows) / 6,
                   "total_polarity_error": sum(r["polarity_error_total"] for r in arm_rows),
                   "joint_phase_flip_seed_count": sum(summaries[selector, s]["joint_phase_flip_present"] for s in SEEDS),
                   "mean_support_delta_range": sum(summaries[selector, s]["support_delta_range"] for s in SEEDS) / 3,
                   "mean_false_entitlement_delta_range": sum(summaries[selector, s]["false_entitlement_delta_range"] for s in SEEDS) / 3,
                   "max_abs_refute_delta": max(summaries[selector, s]["max_abs_refute_delta"] for s in SEEDS),
                   "max_abs_polarity_delta": max(summaries[selector, s]["max_abs_polarity_delta"] for s in SEEDS)}
        else:
            agg = {"selector": selector, "available_all_seeds": False, "synchronous_all_seeds": False}
        aggregates[selector] = agg
    reference = aggregates["historical_independent"]
    for selector in SELECTORS:
        agg = aggregates[selector]
        preserving_conditions = []
        if selector != "historical_independent":
            preserving_conditions = [
                ("available_all_seeds", True, agg.get("available_all_seeds")),
                ("synchronous_all_seeds", True, agg.get("synchronous_all_seeds")),
                ("mean_macro_f1_floor", reference["mean_clean_macro_f1"] - .005, agg.get("mean_clean_macro_f1")),
                ("mean_accuracy_floor", reference["mean_clean_accuracy"] - .01, agg.get("mean_clean_accuracy")),
                ("false_entitlement_ceiling", reference["total_false_entitlement"] + 30, agg.get("total_false_entitlement")),
                ("false_not_entitled_ceiling", reference["total_false_not_entitled"] + 30, agg.get("total_false_not_entitled")),
                ("polarity_error_ceiling", reference["total_polarity_error"] + 1, agg.get("total_polarity_error")),
                ("joint_phase_flip_seed_ceiling", 1, agg.get("joint_phase_flip_seed_count")),
                ("support_range_ratio_ceiling", .75 * reference["mean_support_delta_range"], agg.get("mean_support_delta_range")),
                ("false_entitlement_range_ratio_ceiling", .75 * reference["mean_false_entitlement_delta_range"], agg.get("mean_false_entitlement_delta_range")),
                ("base_refute_and_polarity_bounds_each_seed", True, all(abs(pair_deltas.get((selector, s), {}).get("delta_pred_REFUTE", 999)) <= 1 and abs(pair_deltas.get((selector, s), {}).get("delta_polarity_error_total", 999)) <= 1 for s in SEEDS)),
            ]
        passed = []
        for gate, required, observed in preserving_conditions:
            if gate in ("mean_macro_f1_floor", "mean_accuracy_floor"):
                ok = observed is not None and observed >= required
            elif gate in ("false_entitlement_ceiling", "false_not_entitled_ceiling", "polarity_error_ceiling", "joint_phase_flip_seed_ceiling", "support_range_ratio_ceiling", "false_entitlement_range_ratio_ceiling"):
                ok = observed is not None and observed <= required
            else:
                ok = observed is required
            passed.append(ok); tables["gates"].append({"selector": selector, "gate": gate, "required": required, "observed": observed, "passed": ok, "criterion_class": "quality_preserving"})
        quality = selector != "historical_independent" and bool(passed) and all(passed)
        trade_conditions = []
        if selector != "historical_independent":
            trade_conditions = [
                ("fails_quality_preserving", True, not quality, "eq"),
                ("available_and_synchronous", True, agg.get("available_all_seeds") and agg.get("synchronous_all_seeds"), "eq"),
                ("joint_phase_flip_seed_ceiling", 1, agg.get("joint_phase_flip_seed_count"), "le"),
                ("support_range_ratio_ceiling", .75 * reference["mean_support_delta_range"], agg.get("mean_support_delta_range"), "le"),
                ("false_entitlement_range_ratio_ceiling", .75 * reference["mean_false_entitlement_delta_range"], agg.get("mean_false_entitlement_delta_range"), "le"),
                ("mean_macro_f1_tradeoff_floor", reference["mean_clean_macro_f1"] - .015, agg.get("mean_clean_macro_f1"), "ge"),
                ("mean_accuracy_tradeoff_floor", reference["mean_clean_accuracy"] - .02, agg.get("mean_clean_accuracy"), "ge"),
            ]
        trade_pass = []
        for gate, required, observed, operation in trade_conditions:
            ok = observed is not None and ((observed is required) if operation == "eq" else (observed <= required if operation == "le" else observed >= required))
            trade_pass.append(ok); tables["gates"].append({"selector": selector, "gate": gate, "required": required, "observed": observed, "passed": ok, "criterion_class": "quality_tradeoff"})
        agg["quality_preserving"] = quality
        agg["quality_tradeoff"] = selector != "historical_independent" and bool(trade_pass) and all(trade_pass)
        tables["aggregates"].append(agg)
    return aggregates


def ensemble_rows(data: dict[str, Any], run: str, epochs: tuple[int, ...], torch: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = data[run]["predictions"]
    rows = []
    for position in range(720):
        originals = [source[e][position] for e in epochs]
        gold = originals[0]["gold_final_label"]
        if any(row["gold_final_label"] != gold for row in originals):
            raise ValueError("ensemble gold alignment mismatch")
        logits = [sum(float(row["final_logits"][j]) for row in originals) / len(originals) for j in range(3)]
        best = max(range(3), key=lambda j: logits[j])
        maximum = max(logits); logsumexp = maximum + math.log(sum(math.exp(x - maximum) for x in logits))
        rows.append({"dev_position": position, "gold_final_label": gold, "predicted_final_label": LABELS[best],
                     "final_logits": logits, "final_ce": logsumexp - logits[LABELS.index(gold)]})
    return rows, metrics(rows, torch, ce_from_rows=False)


def evaluate_ensembles(data: dict[str, Any], tables: dict[str, list[dict[str, Any]]], torch: Any) -> dict[str, Any]:
    evidence = {}
    for comparator, epochs in COMPARATORS.items():
        evidence[comparator] = {}
        for seed in SEEDS:
            b_rows, b = ensemble_rows(data, f"seed{seed}_baseline", epochs, torch)
            i_rows, i = ensemble_rows(data, f"seed{seed}_intervention", epochs, torch)
            delta = {f"delta_{m}": i[m] - b[m] for m in METRICS}
            row = {"comparator": comparator, "seed": seed, "epochs": list(epochs),
                   **{f"baseline_{m}": b[m] for m in METRICS}, **{f"intervention_{m}": i[m] for m in METRICS},
                   **delta, **transition(b_rows, i_rows)}
            tables["ensembles"].append(row); evidence[comparator][seed] = row
    return evidence


def analyze(args: argparse.Namespace, tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    import torch
    repo, stage191b, stage191d, _ = establish_safe_output(args)
    identity = source_identity(repo, args.current_diagnostic_git_commit)
    if not identity["passed"]:
        raise ValueError("Stage192-A source identity failed")
    stage191d_report = validate_stage191d(stage191d, tables["closure"])
    data = load_replays(repo, stage191b, torch)
    choices = choose_selectors(data, tables)
    aggregates = evaluate_selectors(data, choices, tables)
    ensemble_evidence = evaluate_ensembles(data, tables, torch)
    qualifiers = [name for name in SELECTORS if aggregates[name].get("quality_preserving")]
    tradeoffs = [name for name in SELECTORS if aggregates[name].get("quality_tradeoff")]
    winner = None
    if qualifiers:
        winner = min(qualifiers, key=lambda name: (aggregates[name]["joint_phase_flip_seed_count"],
                     aggregates[name]["mean_support_delta_range"], aggregates[name]["mean_false_entitlement_delta_range"],
                     -aggregates[name]["mean_clean_macro_f1"], -aggregates[name]["mean_clean_accuracy"],
                     aggregates[name]["mean_clean_dev_ce"], name))
        decision = IDENTIFIED
        recommendation = "Design Stage192-B fresh-seed validation for the frozen winning selector; do not authorize or execute it."
    elif tradeoffs:
        decision = TRADEOFF
        recommendation = "Refine the checkpoint-selection objective without training."
    else:
        decision = NONE
        recommendation = "Design trajectory-level optimization or regularization rather than further checkpoint-selection tuning."
    return {"stage": "Stage192-A", "decision": decision, "runnable": True, "blocking_reasons": [],
            "diagnostic_only": True, "training_performed": False, "model_constructed": False,
            "external_data_used": False, "model_advancement_decision": False, "stage192b_training_authorized": False,
            "current_diagnostic_git_commit": args.current_diagnostic_git_commit, "diagnostic_source_identity": identity,
            "stage191b_commit": STAGE191B_COMMIT, "stage191b_directory": str(stage191b),
            "stage191d_implementation_commit": STAGE191D_COMMIT, "stage191d_directory": str(stage191d),
            "stage191d_decision": stage191d_report["decision"], "clean_dev_ce_reduction_contract": CE_CONTRACT,
            "analysis_epoch_domain": list(DOMAIN), "canonical_labels": list(LABELS),
            "historical_selected_epochs": HISTORICAL, "selector_aggregates": aggregates,
            "quality_preserving_selectors": qualifiers, "quality_tradeoff_selectors": tradeoffs,
            "winning_selector": winner, "temporal_ensembles_descriptive_only": True,
            "temporal_ensemble_evidence": ensemble_evidence, "recommended_next_stage": recommendation,
            "interpretation_restrictions": ["no statistical significance", "no generalization beyond three frozen seeds",
                "no external or OOD performance", "no causal parameter-group claims", "no deployment validation",
                "no model advancement", "no Stage192-B training authorization"], "exception": None}


def blocked_report(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {"stage": "Stage192-A", "decision": BLOCKED, "runnable": False,
            "blocking_reasons": [f"{type(exc).__name__}: {exc}"], "diagnostic_only": True,
            "training_performed": False, "model_constructed": False, "external_data_used": False,
            "model_advancement_decision": False, "stage192b_training_authorized": False,
            "current_diagnostic_git_commit": args.current_diagnostic_git_commit,
            "stage191b_commit": STAGE191B_COMMIT, "stage191d_implementation_commit": STAGE191D_COMMIT,
            "clean_dev_ce_reduction_contract": CE_CONTRACT, "winning_selector": None,
            "recommended_next_stage": "Resolve the frozen-input or analysis failure; do not authorize training.",
            "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Stage192-A trajectory-stable selection report", "", f"Decision: `{report['decision']}`", "",
             f"- Runnable: {str(report['runnable']).lower()}", "- Diagnostic only: true", "- Training performed: false",
             "- Model constructed: false", "- External data used: false", "- Model advancement decision: false",
             "- Stage192-B training authorized: false", f"- Winning selector: {report.get('winning_selector') or 'none'}", "",
             "## Interpretation", ""]
    if report["decision"] == IDENTIFIED:
        lines.append("A precommitted synchronous selector met every frozen quality-preserving stability criterion across the three paired seeds.")
    elif report["decision"] == TRADEOFF:
        lines.append("No selector preserved the full clean-quality contract, although at least one synchronous selector met the precommitted tradeoff stability criteria.")
    elif report["decision"] == NONE:
        lines.append("The frozen trajectories do not identify a selector meeting either precommitted stability criterion.")
    else:
        lines.append("Interpretation is blocked; no checkpoint-selection conclusion is emitted.")
    lines += ["", "Temporal ensembles are descriptive only. This report makes no significance, broader-generalization, external-performance, causal, deployment, training-authorization, or model-advancement claim.", "", "## Recommended next stage", "", report["recommended_next_stage"], ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        _, _, _, output = establish_safe_output(args)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 2
    output.mkdir(parents=False, exist_ok=True)
    tables = {name: [] for name in CSV_HEADERS}
    try:
        report = analyze(args, tables)
    except BaseException as exc:
        report = blocked_report(args, exc)
        tables["gates"].append({"selector": "", "gate": "fail_closed_exception", "required": "no exception",
                                "observed": {"type": type(exc).__name__, "message": str(exc)}, "passed": False,
                                "criterion_class": "integrity"})
    write_json(output / OUTPUTS["json"], report)
    (output / OUTPUTS["md"]).write_text(markdown(report), encoding="utf-8")
    for name, headers in CSV_HEADERS.items():
        write_csv(output / OUTPUTS[name], headers, tables[name])
    return 0 if report["runnable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
