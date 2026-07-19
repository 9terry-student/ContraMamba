#!/usr/bin/env python3
"""Artifact-only Stage196-A persistent SUPPORT boundary localization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

STAGE195C_COMMIT = "b258328d4984160429217d18127b924ec0561415"
STAGE195C_DECISION = "STAGE195C_PARAMETER_SWA_MIXED_OR_INCONCLUSIVE"
BLOCKED = "STAGE196A_PERSISTENT_SUPPORT_BOUNDARY_LOCALIZATION_BLOCKED"
LOCAL = "STAGE196A_RECURRENT_LOCAL_CHANNEL_FAILURE"
AGGREGATION = "STAGE196A_RECURRENT_ENTITLEMENT_AGGREGATION_FAILURE"
COMPOSITION = "STAGE196A_RECURRENT_FINAL_COMPOSITION_BOUNDARY_FAILURE"
SEED_VARIANCE = "STAGE196A_SEED_SPECIFIC_BOUNDARY_VARIANCE"
MIXED = "STAGE196A_MIXED_PERSISTENT_MECHANISMS"
DECISIONS = (BLOCKED, LOCAL, AGGREGATION, COMPOSITION, SEED_VARIANCE, MIXED)
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
SEEDS = (180, 181, 182)
ARMS = ("baseline", "intervention")
RUNS = tuple(f"seed{s}_{a}" for s in SEEDS for a in ARMS)
EXPECTED_PERSISTENT = dict(zip(RUNS, (22, 14, 30, 20, 18, 23)))
ROW_COUNT, SUPPORT_COUNT, PERSISTENT_COUNT = 720, 89, 127
TRAINER_RUN = "single"
BUCKETS = ("MULTI_LOCAL_CHANNEL_FAILURE", "FRAME_ONLY_FAILURE",
           "PREDICATE_ONLY_FAILURE", "SUFFICIENCY_ONLY_FAILURE",
           "ENTITLEMENT_AGGREGATION_FAILURE", "FINAL_COMPOSITION_BOUNDARY_FAILURE")
LOCAL_BUCKETS = set(BUCKETS[:4])
SCALARS = ("frame_prob", "predicate_coverage_prob", "sufficiency_prob",
           "entitlement_prob", "polarity_margin", "frame_logit",
           "positive_energy", "negative_energy")
OUTPUTS = {
    "json": "stage196a_persistent_support_boundary_report.json",
    "md": "stage196a_persistent_support_boundary_report.md",
    "runs": "stage196a_run_population_summary.csv",
    "rows": "stage196a_persistent_row_localization.jsonl",
    "buckets": "stage196a_mechanism_bucket_summary.csv",
    "recurrence": "stage196a_cross_seed_recurrence.csv",
    "paired": "stage196a_paired_arm_transition.csv",
    "closure": "stage196a_source_closure.csv",
    "decision": "stage196a_precommitted_decision_gate.csv",
}
STAGE195C_FILES = {
    "stage195c_parameter_swa_causal_report.json", "stage195c_parameter_swa_causal_report.md",
    "stage195c_run_summary.csv", "stage195c_row_transition.jsonl",
    "stage195c_temporal_outlier_transition.csv", "stage195c_support_mechanism_summary.csv",
    "stage195c_paired_seed_arm_delta.csv", "stage195c_source_closure.csv",
    "stage195c_precommitted_decision_gate.csv",
}
STAGE195C_DECISION_ORDER = (
    "STAGE195C_PARAMETER_SWA_CAUSAL_ANALYSIS_BLOCKED",
    "STAGE195C_PARAMETER_SWA_REPLICATED_CAUSAL_HARM",
    "STAGE195C_PARAMETER_SWA_REPLICATED_TEMPORAL_CAUSAL_SUPPORT",
    "STAGE195C_PARAMETER_SWA_TEMPORAL_SUPPORT_WITH_BOUNDARY_TRADEOFF",
    "STAGE195C_PARAMETER_SWA_NO_TEMPORAL_CAUSAL_SUPPORT",
    "STAGE195C_PARAMETER_SWA_MIXED_OR_INCONCLUSIVE",
)
SOURCE_FILES = ("reports/stage196a_persistent_support_boundary_localization_spec.md",
                "scripts/analyze_stage196a_persistent_support_boundary.py")
GATE_HEADER = ["scope", "run", "gate", "required", "observed", "passed", "blocking_reason"]
DECISION_HEADER = ["decision", "taxonomy_condition", "required", "observed", "passed"]
RUN_HEADER = ["run", "seed", "arm", "gold_support_count", "persistent_stable_support_negative_count",
              "stable_correct_support_control_count", "temporal_support_consensus_outlier_count",
              "other_gold_support_count", "persistent_swa_support_rescue_count"]
BUCKET_HEADER = ["scope", "run", "arm", "mechanism_bucket", "count", "share"]
RECURRENCE_HEADER = ["arm", "dev_position", "stable_example_identity", "pair_id",
    "intervention_type", "persistent_seed_count", "stable_correct_seed_count",
    "recurrent_persistent_within_arm", "universal_persistent_within_arm",
    "mechanism_bucket_distribution", *[f"{s}_{k}" for s in SCALARS for k in ("mean", "min", "max")]]
PAIRED_HEADER = ["seed", "dev_position", "transition", "baseline_mechanism_bucket",
                 "intervention_mechanism_bucket", *[f"{s}_delta" for s in SCALARS]]
ROW_KEYS = ("stage", "run", "manifest_run_id", "trainer_run_name", "seed", "arm", "split_seed",
    "dev_position", "stable_example_identity", "pair_id", "intervention_type", "gold_label",
    "epoch18_prediction", "epoch19_prediction", "epoch20_prediction", "swa_prediction",
    *SCALARS, "frame_pass", "predicate_pass", "sufficiency_pass", "entitlement_pass",
    "all_local_channels_pass", "failed_local_channel_count", "mechanism_bucket",
    "epoch20_logits", "swa_logits", "epoch20_support_minus_ne_margin",
    "swa_support_minus_ne_margin", "recurrent_persistent_within_arm",
    "universal_persistent_within_arm", "persistent_seed_count_within_arm",
    "persistent_in_other_arm", "persistent_in_all_six_runs")
TRANSITION_KEYS = {"stage", "run", "seed", "arm", "split_seed", "dev_position", "gold_label",
    "epoch18_prediction", "epoch19_prediction", "epoch20_prediction", "mean_logit_prediction",
    "majority_available", "majority_prediction", "swa_prediction", "epoch20_correct",
    "mean_logit_correct", "majority_correct", "swa_correct", "temporal_consensus_outlier",
    "temporal_outlier_subtype", "swa_transition_type", "swa_aligns_majority",
    "swa_aligns_mean_logit", "target_support_consensus_outlier",
    "persistent_stable_support_negative", "epoch18_logits", "epoch19_logits", "epoch20_logits",
    "tail3_mean_logits", "swa_logits", "swa_vs_epoch20_l1", "swa_vs_epoch20_l2",
    "swa_vs_mean_logit_l1", "swa_vs_mean_logit_l2"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--stage195c-dir", required=True, type=Path)
    p.add_argument("--stage195b-run-root", required=True, type=Path)
    p.add_argument("--current-diagnostic-git-commit", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()

def exact_int(v: Any) -> bool: return type(v) is int
def finite(v: Any) -> bool: return type(v) in (int, float) and math.isfinite(float(v))
def ratio(n: int, d: int) -> float | None: return n / d if d else None

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f: return json.load(f)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip(): raise ValueError(f"{path}:{n}: blank JSONL row")
            row = json.loads(line)
            if type(row) is not dict: raise ValueError(f"{path}:{n}: row is not an object")
            out.append(row)
    return out

def read_csv(path: Path, header: list[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f); observed = list(r.fieldnames or []); rows = list(r)
    if header is not None and (observed != header or any(set(x) != set(header) for x in rows)):
        raise ValueError(f"{path}: exact CSV schema mismatch")
    return rows

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def git(repo: Path, args: list[str], *, binary: bool = False, dirty: bool = False) -> Any:
    r = subprocess.run(["git", *args], cwd=repo, check=False, capture_output=True, shell=False)
    if dirty:
        if r.returncode not in (0, 1): raise RuntimeError(r.stderr.decode(errors="replace"))
        return r.returncode
    if r.returncode: raise RuntimeError(r.stderr.decode(errors="replace"))
    return r.stdout if binary else r.stdout.decode("utf-8", errors="strict").strip()

def gate(rows: list[dict[str, Any]], scope: str, run: str, name: str,
         required: Any, observed: Any, passed: bool, reason: str) -> None:
    rows.append({"scope": scope, "run": run, "gate": name, "required": required,
                 "observed": observed, "passed": passed, "blocking_reason": "" if passed else reason})
    if not passed: raise ValueError(f"{run + ': ' if run else ''}{reason}")

def safe_paths(a: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    repo = a.repo_root.resolve(); reports = (repo / "reports").resolve()
    c = a.stage195c_dir.resolve(); runs = a.stage195b_run_root.resolve(); out = a.output_dir.resolve()
    if not repo.is_dir() or not reports.is_dir(): raise ValueError("repository/reports directory absent")
    if len({repo, c, runs, out}) != 4: raise ValueError("all supplied paths must be distinct")
    if c.parent != reports or not c.name.startswith("stage195c_parameter_swa_causal_analysis_") or not c.is_dir():
        raise ValueError("unsafe or absent Stage195-C directory")
    if runs.parent != reports or not runs.name.startswith("stage195b_tail3_parameter_swa_runs_") or not runs.is_dir():
        raise ValueError("unsafe or absent Stage195-B run root")
    if out.parent != reports or not out.name.startswith("stage196a_persistent_support_boundary_localization_"):
        raise ValueError("unsafe Stage196-A output path")
    if out.exists() and (not out.is_dir() or any(out.iterdir())): raise ValueError("output exists and is nonempty")
    return repo, c, runs, out

def validate_code(repo: Path, commit: str, closure: list[dict[str, Any]]) -> None:
    gate(closure, "source", "", "commit_format", "lowercase 40-hex", commit,
         re.fullmatch(r"[0-9a-f]{40}", commit or "") is not None, "invalid diagnostic commit")
    head = git(repo, ["rev-parse", "HEAD"])
    gate(closure, "source", "", "commit_equals_head", commit, head, head == commit, "commit differs from HEAD")
    for rel in SOURCE_FILES:
        current = (repo / rel).read_bytes(); blob = git(repo, ["show", f"{commit}:{rel}"], binary=True)
        clean_u = git(repo, ["diff", "--quiet", "--", rel], dirty=True) == 0
        clean_s = git(repo, ["diff", "--cached", "--quiet", "--", rel], dirty=True) == 0
        observed = {"bytes_equal": current == blob, "unstaged_clean": clean_u, "staged_clean": clean_s,
                    "sha256": hashlib.sha256(current).hexdigest()}
        gate(closure, "source", "", f"source_identity:{rel}",
             {"bytes_equal": True, "unstaged_clean": True, "staged_clean": True}, observed,
             current == blob and clean_u and clean_s, f"source identity failed: {rel}")

def logits(v: Any, context: str) -> list[float]:
    if type(v) is not list or len(v) != 3 or any(not finite(x) for x in v):
        raise ValueError(f"{context}: invalid finite length-three logits")
    return [float(x) for x in v]

def canonical(v: Sequence[float]) -> str:
    return LABELS[max(range(3), key=lambda i: float(v[i]))]

def validate_stage195c(path: Path, closure: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = {p.name for p in path.iterdir()}
    gate(closure, "stage195c", "", "exact_nine_file_closure", sorted(STAGE195C_FILES), sorted(names),
         names == STAGE195C_FILES and all((path / n).is_file() for n in names), "Stage195-C file closure mismatch")
    report = read_json(path / "stage195c_parameter_swa_causal_report.json")
    required = {"decision": STAGE195C_DECISION, "runnable": True, "blocking_reasons": [],
                "stage195c_runtime_repository_commit": STAGE195C_COMMIT,
                "ordered_runs": list(RUNS), "row_transition_count": 4320}
    gate(closure, "stage195c", "", "report_closure", required, {k: report.get(k) for k in required},
         all(report.get(k) == v for k, v in required.items()) and report.get("decision") != "STAGE195C_PARAMETER_SWA_CAUSAL_ANALYSIS_BLOCKED",
         "Stage195-C report closure mismatch")
    source = read_csv(path / "stage195c_source_closure.csv", GATE_HEADER)
    source_ok = bool(source) and all(x["passed"] == "True" and x["blocking_reason"] == "" for x in source)
    gate(closure, "stage195c", "", "all_source_gates_pass", True, source_ok, source_ok, "Stage195-C source gate failure")
    decisions = read_csv(path / "stage195c_precommitted_decision_gate.csv", DECISION_HEADER)
    if len(decisions) != len(STAGE195C_DECISION_ORDER) or tuple(
            row["decision"] for row in decisions) != STAGE195C_DECISION_ORDER:
        raise ValueError("Stage195-C decision row count/order mismatch")
    selected_rows = [row for row in decisions if row["required"] == "True"]
    if (len(selected_rows) != 1
            or sum(row["required"] == "False" for row in decisions) != 5):
        raise ValueError("Stage195-C required cardinality mismatch")
    selected = selected_rows[0]
    if selected["decision"] != STAGE195C_DECISION:
        raise ValueError("Stage195-C selected decision mismatch")
    if selected["decision"] != report["decision"]:
        raise ValueError("Stage195-C report/decision-gate mismatch")
    if any(row["passed"] != "True" for row in decisions):
        raise ValueError("Stage195-C non-passing decision row")
    observed_evidence: list[dict[str, Any]] = []
    for row in decisions:
        raw_observed = row["observed"]
        if raw_observed == "":
            raise ValueError("Stage195-C empty observed evidence")
        try:
            parsed_observed = json.loads(raw_observed)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Stage195-C invalid observed JSON") from exc
        if type(parsed_observed) is not dict:
            raise ValueError("Stage195-C observed evidence is not an object")
        observed_evidence.append(parsed_observed)
    selected_index = decisions.index(selected)
    if observed_evidence[selected_index].get("condition") is not True:
        raise ValueError("Stage195-C selected observed condition is not true")
    if any(evidence.get("condition") is not (row["required"] == "True")
           for row, evidence in zip(decisions, observed_evidence)):
        raise ValueError("Stage195-C observed condition/required mismatch")
    shared_evidence = [
        {key: value for key, value in evidence.items() if key != "condition"}
        for evidence in observed_evidence
    ]
    if any(evidence != shared_evidence[0] for evidence in shared_evidence[1:]):
        raise ValueError("Stage195-C observed evidence differs across decision rows")
    gate(closure, "stage195c", "", "precommitted_decision_gate", STAGE195C_DECISION,
         selected["decision"], True, "")
    rows = read_jsonl(path / "stage195c_row_transition.jsonl")
    if len(rows) != 4320: raise ValueError("Stage195-C transition cardinality mismatch")
    counts = Counter(); rescues = Counter(); per_run = Counter()
    for i, row in enumerate(rows):
        if set(row) != TRANSITION_KEYS: raise ValueError(f"Stage195-C row {i}: exact schema mismatch")
        run = row["run"]
        if run not in RUNS or row["seed"] != int(run[4:7]) or row["arm"] != run.split("_", 1)[1] or row["split_seed"] != 174:
            raise ValueError(f"Stage195-C row {i}: identity mismatch")
        pos = row["dev_position"]
        if not exact_int(pos) or not 0 <= pos < ROW_COUNT: raise ValueError("invalid Stage195-C position")
        per_run[run] += 1
        persistent = (row["gold_label"] == "SUPPORT" and all(row[f"epoch{e}_prediction"] == "NOT_ENTITLED" for e in (18,19,20)))
        if row["persistent_stable_support_negative"] is not persistent: raise ValueError("persistent flag recomputation mismatch")
        if persistent:
            counts[run] += 1
            if row["swa_prediction"] == "SUPPORT": rescues[run] += 1
        epoch20_logits = logits(row["epoch20_logits"], f"{run}:{pos}:epoch20")
        swa_logits = logits(row["swa_logits"], f"{run}:{pos}:swa")
        if (row["gold_label"] not in LABELS or row["epoch20_prediction"] != canonical(epoch20_logits)
                or row["swa_prediction"] != canonical(swa_logits)):
            raise ValueError(f"{run}:{pos}: canonical label/logit alignment mismatch")
    ok = list(per_run.keys()) == list(RUNS) and all(per_run[r] == 720 for r in RUNS)
    gate(closure, "stage195c", "", "six_run_order_and_cardinality", list(RUNS), list(per_run.keys()), ok,
         "Stage195-C run order/cardinality mismatch")
    gate(closure, "stage195c", "", "persistent_counts", EXPECTED_PERSISTENT, dict(counts),
         dict(counts) == EXPECTED_PERSISTENT and sum(counts.values()) == 127, "persistent count mismatch")
    gate(closure, "stage195c", "", "persistent_swa_rescue_zero", 0, dict(rescues),
         sum(rescues.values()) == 0 and all(rescues[r] == 0 for r in RUNS), "persistent SWA rescue mismatch")
    return rows

def nested_false(report: dict[str, Any], keys: Sequence[str]) -> bool:
    containers = [report, report.get("configuration", {})]
    return all(any(type(c) is dict and c.get(k) is False for c in containers) for k in keys)

def metadata_value(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in row and row[k] is not None: return row[k]
    return None

def validate_run_sources(root: Path, stage_rows: list[dict[str, Any]], closure: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    entries = [p.name for p in root.iterdir()]
    if set(entries) != set(RUNS) or any(not (root / r).is_dir() for r in RUNS): raise ValueError("Stage195-B exact six-run root mismatch")
    stage_by_run = {r: sorted((x for x in stage_rows if x["run"] == r), key=lambda x: x["dev_position"]) for r in RUNS}
    out = {}
    for run in RUNS:
        seed = int(run[4:7]); arm = run.split("_", 1)[1]; d = (root / run).resolve()
        needed = ("clean_dev_scalars.jsonl", "clean_dev_predictions.json", "training_report.json",
                  "stage191_dev_predictions_epoch_020.jsonl", "stage195_tail3_parameter_swa_contract.json")
        missing = [n for n in needed if not (d / n).is_file()]
        gate(closure, "run", run, "required_scalar_source_files", [], missing, not missing, "required source file missing")
        scalars = read_jsonl(d / needed[0]); pred_obj = read_json(d / needed[1]); training = read_json(d / needed[2])
        epoch20 = read_jsonl(d / needed[3]); contract = read_json(d / needed[4])
        if (type(pred_obj) is not dict or set(pred_obj) != {"metadata", "predictions"}
                or type(pred_obj["metadata"]) is not dict or type(pred_obj["predictions"]) is not list):
            raise ValueError(f"{run}: clean prediction top-level schema mismatch")
        preds = pred_obj["predictions"]
        if any(len(x) != ROW_COUNT for x in (scalars, preds, epoch20)): raise ValueError(f"{run}: 720-row source cardinality mismatch")
        if (contract.get("run") != TRAINER_RUN or contract.get("training_seed") != seed
                or contract.get("split_seed") != 174 or contract.get("arm") != arm
                or contract.get("external_data_used") is not False):
            raise ValueError(f"{run}: SWA contract identity mismatch")
        if training.get("training_seed") != seed or training.get("resolved_split_seed") != 174 or "single" not in training.get("runs", {}):
            raise ValueError(f"{run}: training report identity mismatch")
        if not nested_false(training, ("time_swap_used_in_main_clean_data", "time_swap_used_in_v7_main_clean_data")):
            raise ValueError(f"{run}: time_swap main-clean denial missing")
        forbidden_true = ("external_data_used", "external_data_used_for_training", "external_examples_used")
        if any(training.get(k) is True or training.get("configuration", {}).get(k) is True for k in forbidden_true):
            raise ValueError(f"{run}: external/OOD use recorded")
        scalar_path = training.get("stage115_clean_dev_scalar_output_jsonl")
        if scalar_path is None or Path(str(scalar_path)).resolve() != (d / needed[0]).resolve():
            raise ValueError(f"{run}: training report scalar-path/run-directory mismatch")
        if pred_obj["metadata"].get("seed") != seed:
            raise ValueError(f"{run}: clean prediction training-seed mismatch")
        joined = []
        for pos, (s, p, e, c) in enumerate(zip(scalars, preds, epoch20, stage_by_run[run])):
            scalar_base = {"id", "claim", "evidence", "gold_label", "prediction",
                           "frame_logit", "frame_prob", "score_source"}
            if not scalar_base.issubset(s) or "dev_position" in s or s.get("score_source") != 'direct output["frame_logit"]':
                raise ValueError(f"{run}:{pos}: Stage115 physical scalar schema mismatch")
            if set(e) != {"epoch", "dev_position", "source_row_id", "gold_final_label", "predicted_final_label", "final_logits", "final_ce", "frame_logit"}:
                raise ValueError(f"{run}:{pos}: Stage191 exact schema mismatch")
            if e["epoch"] != 20 or e["dev_position"] != pos or c["dev_position"] != pos:
                raise ValueError(f"{run}:{pos}: position mismatch")
            if e["gold_final_label"] != c["gold_label"] or e["predicted_final_label"] != c["epoch20_prediction"] or logits(e["final_logits"], "epoch20") != logits(c["epoch20_logits"], "stage195c"):
                raise ValueError(f"{run}:{pos}: epoch20 alignment mismatch")
            pid = metadata_value(p, ("id",)); sid = metadata_value(s, ("id",))
            if pid is None or sid is None or str(pid) != str(sid): raise ValueError(f"{run}:{pos}: scalar/prediction ID mismatch")
            gold = metadata_value(p, ("gold_label", "gold_final_label"))
            pred = metadata_value(p, ("pred_label", "pred_final_label", "prediction"))
            if gold != c["gold_label"] or pred != c["epoch20_prediction"] or s.get("gold_label") != gold or s.get("prediction") != pred:
                raise ValueError(f"{run}:{pos}: scalar prediction alignment mismatch")
            values = {}
            for key in SCALARS:
                value = s.get(key)
                if not finite(value): raise ValueError(f"{run}:{pos}: missing/nonfinite {key}")
                values[key] = float(value)
            if any(not 0.0 <= values[k] <= 1.0 for k in SCALARS[:4]): raise ValueError(f"{run}:{pos}: probability outside [0,1]")
            joined.append({**values, "stable_example_identity": metadata_value(p, ("stable_id", "id", "source_id")),
                "pair_id": metadata_value(p, ("pair_id",)),
                "intervention_type": metadata_value(p, ("intervention_type", "intervention", "normalized_intervention")),
                "source_row_id": e["source_row_id"], "gold_label": e["gold_final_label"]})
        out[run] = joined
        gate(closure, "run", run, "scalar_schema_identity_alignment", True, True, True, "")
    for pos in range(ROW_COUNT):
        identities = {(out[r][pos]["source_row_id"], out[r][pos]["stable_example_identity"],
                       out[r][pos]["pair_id"], out[r][pos]["intervention_type"],
                       out[r][pos]["gold_label"]) for r in RUNS}
        if len(identities) != 1:
            raise ValueError(f"dev position {pos}: cross-run source identity mismatch")
    gate(closure, "source", "", "cross_run_dev_order_identity", True, True, True, "")
    return out

def mechanism(v: dict[str, float]) -> tuple[str, dict[str, bool]]:
    passes = {"frame_pass": v["frame_prob"] >= .5, "predicate_pass": v["predicate_coverage_prob"] >= .5,
              "sufficiency_pass": v["sufficiency_prob"] >= .5, "entitlement_pass": v["entitlement_prob"] >= .5}
    failed = sum(not passes[k] for k in ("frame_pass", "predicate_pass", "sufficiency_pass"))
    if failed >= 2: bucket = BUCKETS[0]
    elif not passes["frame_pass"]: bucket = BUCKETS[1]
    elif not passes["predicate_pass"]: bucket = BUCKETS[2]
    elif not passes["sufficiency_pass"]: bucket = BUCKETS[3]
    elif not passes["entitlement_pass"]: bucket = BUCKETS[4]
    else: bucket = BUCKETS[5]
    passes.update({"all_local_channels_pass": failed == 0, "failed_local_channel_count": failed})
    return bucket, passes

def logit_diagnostics(v: Sequence[float]) -> dict[str, float | int]:
    order = sorted(range(3), key=lambda i: (-float(v[i]), i))
    return {"support_minus_ne_margin": float(v[2]) - float(v[1]),
            "support_minus_refute_margin": float(v[2]) - float(v[0]),
            "top1_top2_margin": float(v[order[0]]) - float(v[order[1]]),
            "not_entitled_logit_rank": order.index(1) + 1, "support_logit_rank": order.index(2) + 1}

def stats(values: Sequence[float]) -> dict[str, Any]:
    return {"count": len(values), "mean": math.fsum(values)/len(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}

def analyze(a: argparse.Namespace, tables: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repo, cdir, root, _ = safe_paths(a); validate_code(repo, a.current_diagnostic_git_commit, tables["closure"])
    source_rows = validate_stage195c(cdir, tables["closure"]); scalars = validate_run_sources(root, source_rows, tables["closure"])
    all_rows = []; population_rows = {}; bucket_counts = Counter()
    for run in RUNS:
        rows = sorted((x for x in source_rows if x["run"] == run), key=lambda x: x["dev_position"])
        seed, arm = int(run[4:7]), run.split("_", 1)[1]; populations = Counter(); population_rows[run] = {}
        for row, scalar in zip(rows, scalars[run]):
            if row["gold_label"] != "SUPPORT": continue
            if row["persistent_stable_support_negative"]: pop = "persistent_stable_support_negative"
            elif all(row[f"epoch{e}_prediction"] == "SUPPORT" for e in (18,19,20)): pop = "stable_correct_support_control"
            elif row["target_support_consensus_outlier"]: pop = "temporal_support_consensus_outlier"
            else: pop = "other_gold_support"
            populations[pop] += 1; population_rows[run][row["dev_position"]] = pop
            if pop != "persistent_stable_support_negative": continue
            bucket, passes = mechanism(scalar); bucket_counts[(run, bucket)] += 1
            e20 = logit_diagnostics(row["epoch20_logits"]); swa = logit_diagnostics(row["swa_logits"])
            item = {"stage": "Stage196-A", "run": run, "manifest_run_id": run, "trainer_run_name": TRAINER_RUN,
                "seed": seed, "arm": arm, "split_seed": 174, "dev_position": row["dev_position"],
                "stable_example_identity": scalar["stable_example_identity"], "pair_id": scalar["pair_id"],
                "intervention_type": scalar["intervention_type"], "gold_label": row["gold_label"],
                "epoch18_prediction": row["epoch18_prediction"], "epoch19_prediction": row["epoch19_prediction"],
                "epoch20_prediction": row["epoch20_prediction"], "swa_prediction": row["swa_prediction"],
                **{key: scalar[key] for key in SCALARS}, **passes,
                "mechanism_bucket": bucket, "epoch20_logits": row["epoch20_logits"], "swa_logits": row["swa_logits"],
                "epoch20_support_minus_ne_margin": e20["support_minus_ne_margin"],
                "swa_support_minus_ne_margin": swa["support_minus_ne_margin"]}
            item["_epoch20_diagnostics"] = e20; item["_swa_diagnostics"] = swa; all_rows.append(item)
        if sum(populations.values()) != SUPPORT_COUNT:
            raise ValueError(f"{run}: mutually exclusive gold SUPPORT population closure mismatch")
        tables["runs"].append({"run": run, "seed": seed, "arm": arm, "gold_support_count": sum(populations.values()),
            "persistent_stable_support_negative_count": populations["persistent_stable_support_negative"],
            "stable_correct_support_control_count": populations["stable_correct_support_control"],
            "temporal_support_consensus_outlier_count": populations["temporal_support_consensus_outlier"],
            "other_gold_support_count": populations["other_gold_support"], "persistent_swa_support_rescue_count": 0})
    if len(all_rows) != 127: raise ValueError("persistent localization must contain exactly 127 rows")
    by_run_pos = {(x["run"], x["dev_position"]): x for x in all_rows}; arm_positions = {}
    for arm in ARMS:
        positions = sorted({p for r in RUNS if r.endswith(arm) for p in population_rows[r]})
        arm_positions[arm] = {}
        for pos in positions:
            runs = [f"seed{s}_{arm}" for s in SEEDS]; ps = [r for r in runs if population_rows[r].get(pos) == "persistent_stable_support_negative"]
            cs = [r for r in runs if population_rows[r].get(pos) == "stable_correct_support_control"]
            samples = [scalars[r][pos] for r in runs]; buckets = Counter(by_run_pos[(r,pos)]["mechanism_bucket"] for r in ps)
            arm_positions[arm][pos] = len(ps)
            first = samples[0]; rec = {"arm": arm, "dev_position": pos, "stable_example_identity": first["stable_example_identity"],
                "pair_id": first["pair_id"], "intervention_type": first["intervention_type"], "persistent_seed_count": len(ps),
                "stable_correct_seed_count": len(cs), "recurrent_persistent_within_arm": len(ps)>=2,
                "universal_persistent_within_arm": len(ps)==3, "mechanism_bucket_distribution": dict(buckets)}
            for s in SCALARS:
                vals = [x[s] for x in samples]; rec.update({f"{s}_mean": math.fsum(vals)/3, f"{s}_min": min(vals), f"{s}_max": max(vals)})
            tables["recurrence"].append(rec)


    for x in all_rows:
        count = arm_positions[x["arm"]][x["dev_position"]]; other = "intervention" if x["arm"] == "baseline" else "baseline"
        x.update({"recurrent_persistent_within_arm": count >= 2, "universal_persistent_within_arm": count == 3,
                  "persistent_seed_count_within_arm": count,
                  "persistent_in_other_arm": arm_positions[other].get(x["dev_position"], 0) > 0,
                  "persistent_in_all_six_runs": arm_positions["baseline"].get(x["dev_position"],0)==3 and arm_positions["intervention"].get(x["dev_position"],0)==3})
    for seed in SEEDS:
        br, ir = f"seed{seed}_baseline", f"seed{seed}_intervention"
        for pos in range(ROW_COUNT):
            bp, ip = population_rows[br].get(pos), population_rows[ir].get(pos)
            if bp == "persistent_stable_support_negative" and ip == "persistent_stable_support_negative": t = "baseline_persistent_to_intervention_persistent"
            elif bp == "persistent_stable_support_negative" and ip == "stable_correct_support_control": t = "baseline_persistent_to_intervention_stable_correct"
            elif bp == "stable_correct_support_control" and ip == "persistent_stable_support_negative": t = "baseline_stable_correct_to_intervention_persistent"
            else: t = "neither_persistent"
            rec = {"seed": seed, "dev_position": pos, "transition": t,
                   "baseline_mechanism_bucket": by_run_pos.get((br,pos),{}).get("mechanism_bucket"),
                   "intervention_mechanism_bucket": by_run_pos.get((ir,pos),{}).get("mechanism_bucket")}
            rec.update({f"{s}_delta": scalars[ir][pos][s]-scalars[br][pos][s] for s in SCALARS}); tables["paired"].append(rec)
    for scope, arm, runs in [("arm", a, [r for r in RUNS if r.endswith(a)]) for a in ARMS] + [("pooled", "", list(RUNS))]:
        total = sum(bucket_counts[(r,b)] for r in runs for b in BUCKETS)
        for b in BUCKETS:
            n = sum(bucket_counts[(r,b)] for r in runs); tables["buckets"].append({"scope": scope, "run": "", "arm": arm, "mechanism_bucket": b, "count": n, "share": ratio(n,total)})
    for r in RUNS:
        for b in BUCKETS:
            n=bucket_counts[(r,b)]; tables["buckets"].append({"scope":"run","run":r,"arm":r.split("_",1)[1],"mechanism_bucket":b,"count":n,"share":ratio(n,EXPECTED_PERSISTENT[r])})
    pooled = Counter(x["mechanism_bucket"] for x in all_rows); recurring = {a: sum(n>=2 for n in arm_positions[a].values()) for a in ARMS}
    universal = {a: sum(n==3 for n in arm_positions[a].values()) for a in ARMS}; recurrence_exists = any(recurring.values())
    channel_fail = {"frame":sum(not x["frame_pass"] for x in all_rows), "predicate":sum(not x["predicate_pass"] for x in all_rows), "sufficiency":sum(not x["sufficiency_pass"] for x in all_rows)}
    cond = {LOCAL: recurrence_exists and sum(pooled[b] for b in LOCAL_BUCKETS)/127 >= 2/3 and max(channel_fail.values())/127 >= .5,
            AGGREGATION: recurrence_exists and pooled[BUCKETS[4]]/127 >= 2/3,
            COMPOSITION: recurrence_exists and pooled[BUCKETS[5]]/127 >= 2/3}
    unique_counts = [n for a in ARMS for n in arm_positions[a].values() if n]
    cond[SEED_VARIANCE] = bool(all_rows) and not recurrence_exists and sum(n==1 for n in unique_counts) > len(unique_counts)/2
    dominant = [d for d in (LOCAL,AGGREGATION,COMPOSITION) if cond[d]]
    decision = dominant[0] if len(dominant)==1 else SEED_VARIANCE if not dominant and cond[SEED_VARIANCE] else MIXED
    texts = {BLOCKED:"integrity failure",LOCAL:"recurrent local-channel failure thresholds",AGGREGATION:"recurrent entitlement aggregation threshold",COMPOSITION:"recurrent final-composition threshold",SEED_VARIANCE:"nonrecurrent strict-majority single-seed positions",MIXED:"all other integrity-passing results or simultaneous dominant conditions"}
    for d in DECISIONS:
        observed = decision == d; tables["decision"].append({"decision":d,"taxonomy_condition":texts[d],"required":observed,"observed":observed,"passed":True})
    pair_available = all(x["pair_id"] is not None and x["intervention_type"] is not None for r in RUNS for x in scalars[r])
    pair_analysis: dict[str,Any] = {"status":"not_available_from_frozen_source_schema"}
    if pair_available:
        pc=Counter()
        for run in RUNS:
            groups=defaultdict(dict)
            for pos,s in enumerate(scalars[run]):
                if source_rows[RUNS.index(run)*720+pos]["gold_label"]=="SUPPORT" and s["intervention_type"] in ("none","paraphrase"):
                    groups[str(s["pair_id"])][s["intervention_type"]]=population_rows[run].get(pos)
            for g in groups.values():
                if set(g)!={"none","paraphrase"}: continue
                n,p=g["none"],g["paraphrase"]
                if n==p=="persistent_stable_support_negative": pc["both_persistent"]+=1
                elif n=="persistent_stable_support_negative": pc["none_only_persistent"]+=1
                elif p=="persistent_stable_support_negative": pc["paraphrase_only_persistent"]+=1
                else: pc["neither_persistent"]+=1
                if n==p=="stable_correct_support_control": pc["both_stable_correct"]+=1
                if {n,p}=={"stable_correct_support_control","persistent_stable_support_negative"}: pc["one_stable_correct_one_persistent"]+=1
        pair_analysis={"status":"available","counts":dict(pc)}
    logit_summary={}
    for pop in ("persistent_stable_support_negative","stable_correct_support_control","temporal_support_consensus_outlier"):
        selected=[(r,row) for r in RUNS for row in (x for x in source_rows if x["run"]==r and population_rows[r].get(x["dev_position"])==pop)]
        logit_summary[pop]={}
        for predictor,key in (("epoch20","epoch20_logits"),("swa","swa_logits")):
            ds=[logit_diagnostics(row[key]) for _,row in selected]
            logit_summary[pop][predictor]={m:stats([float(x[m]) for x in ds]) for m in ds[0]} if ds else {}
    both_rec=sum(arm_positions["baseline"].get(p,0)>=2 and arm_positions["intervention"].get(p,0)>=2 for p in set(arm_positions["baseline"])|set(arm_positions["intervention"]))
    report={"stage":"Stage196-A","decision":decision,"runnable":True,"blocking_reasons":[],"artifact_only_analysis":True,
        "training_performed":False,"model_loaded":False,"tokenizer_loaded":False,"checkpoint_loaded":False,"state_capsule_loaded":False,"gpu_used":False,
        "stage196a_runtime_repository_commit":a.current_diagnostic_git_commit,"stage195c_runtime_repository_commit":STAGE195C_COMMIT,
        "ordered_runs":list(RUNS),"split_seed":174,"persistent_row_count":127,"native_channel_threshold":.5,
        "primary_estimands":{"pooled_mechanism_bucket_counts":dict(pooled),"recurrent_persistent_position_count_by_arm":recurring,
            "universal_persistent_position_count_by_arm":universal,"all_local_channels_pass_share":sum(x["all_local_channels_pass"] for x in all_rows)/127,
            "entitlement_aggregation_failure_share":pooled[BUCKETS[4]]/127,"final_composition_boundary_failure_share":pooled[BUCKETS[5]]/127,
            "dominant_local_failure_channel":max(channel_fail,key=channel_fail.get),"persistent_recurrence_overlap_between_arms":both_rec},
        "logit_margin_diagnostics":logit_summary,"pair_level_none_paraphrase_analysis":pair_analysis,
        "cross_arm_recurrence":{"baseline_recurring_only":recurring["baseline"]-both_rec,"intervention_recurring_only":recurring["intervention"]-both_rec,
            "recurring_in_both_arms":both_rec,"persistent_in_all_six_runs":sum(arm_positions["baseline"].get(p,0)==3 and arm_positions["intervention"].get(p,0)==3 for p in range(720))},
        "decision_taxonomy":{"selected_decision":decision,"conditions":cond},"new_architecture_selected":False,"new_loss_authorized":False,
        "new_intervention_implemented":False,"external_data_used":False,"calibration_authorized":False,
        "threshold_tuning_authorized":False,"stage196b_intervention_automatically_authorized":False,
        "production_advancement_authorized":False,"statistical_significance_claimed":False,"external_generalization_claimed":False,"exception":None}
    for x in all_rows: x.pop("_epoch20_diagnostics"); x.pop("_swa_diagnostics")
    if any(tuple(x) != ROW_KEYS for x in all_rows): raise ValueError("persistent JSONL exact key/order mismatch")
    return report, all_rows

def blocked_report(a: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {"stage":"Stage196-A","decision":BLOCKED,"runnable":False,"blocking_reasons":[f"{type(exc).__name__}: {exc}"],
        "artifact_only_analysis":True,"training_performed":False,"model_loaded":False,"tokenizer_loaded":False,"checkpoint_loaded":False,
        "state_capsule_loaded":False,"gpu_used":False,"stage196a_runtime_repository_commit":a.current_diagnostic_git_commit,
        "stage195c_runtime_repository_commit":STAGE195C_COMMIT,"ordered_runs":list(RUNS),"split_seed":174,"persistent_row_count":0,
        "native_channel_threshold":.5,"primary_estimands":{},"logit_margin_diagnostics":{},
        "pair_level_none_paraphrase_analysis":{"status":"not_available_from_frozen_source_schema"},"cross_arm_recurrence":{},
        "decision_taxonomy":{"selected_decision":BLOCKED},"new_architecture_selected":False,"new_loss_authorized":False,
        "new_intervention_implemented":False,"external_data_used":False,"calibration_authorized":False,
        "threshold_tuning_authorized":False,"stage196b_intervention_automatically_authorized":False,
        "production_advancement_authorized":False,"statistical_significance_claimed":False,"external_generalization_claimed":False,
        "exception":{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}}

def csv_value(v: Any) -> Any:
    return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(",",":")) if isinstance(v,(dict,list,tuple)) else v

def render_csv(header: list[str], rows: Iterable[dict[str, Any]]) -> str:
    s=io.StringIO(newline=""); w=csv.DictWriter(s,fieldnames=header,extrasaction="raise",lineterminator="\n"); w.writeheader()
    for row in rows:
        if set(row)!=set(header): raise ValueError(f"generated CSV schema mismatch: {set(row)^set(header)}")
        w.writerow({k:csv_value(row[k]) for k in header})
    return s.getvalue()

def render(report: dict[str, Any], rows: list[dict[str, Any]], tables: dict[str,list[dict[str,Any]]]) -> dict[str,str]:
    ready=report["decision"]!=BLOCKED
    if ready and len(rows)!=127: raise ValueError("READY persistent row count mismatch")
    if not ready and rows: raise ValueError("BLOCKED JSONL must be empty")
    md="\n".join(["# Stage196-A persistent SUPPORT boundary localization","",f"Decision: `{report['decision']}`","",
        f"- Runnable: {str(report['runnable']).lower()}","- Artifact-only diagnostic: true","- Training/model/tokenizer/checkpoint/state capsule/GPU use: false","",
        "This report localizes frozen persistent SUPPORT negatives only. It authorizes no intervention, calibration, threshold tuning, training, or production advancement.",""])
    out={OUTPUTS["json"]:json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+"\n",OUTPUTS["md"]:md,
         OUTPUTS["rows"]:"".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in rows)}
    for name,header in (("runs",RUN_HEADER),("buckets",BUCKET_HEADER),("recurrence",RECURRENCE_HEADER),("paired",PAIRED_HEADER),("closure",GATE_HEADER),("decision",DECISION_HEADER)):
        out[OUTPUTS[name]]=render_csv(header,tables[name])
    if set(out)!=set(OUTPUTS.values()): raise RuntimeError("exact nine-output closure mismatch")
    return out

def publish(output: Path, contents: dict[str,str]) -> None:
    temps={n:output/f".{n}.stage196a.tmp" for n in contents}; targets={n:output/n for n in contents}
    if any(p.exists() for p in [*temps.values(),*targets.values()]): raise FileExistsError("refusing overwrite")
    try:
        for n,v in contents.items():
            with temps[n].open("x",encoding="utf-8",newline="\n") as f: f.write(v); f.flush(); os.fsync(f.fileno())
        report_name=OUTPUTS["json"]
        for n in [x for x in contents if x!=report_name]+[report_name]: os.replace(temps[n],targets[n])
    except BaseException:
        for p in [*temps.values(),*targets.values()]:
            try: p.unlink(missing_ok=True)
            except OSError: pass
        raise

def main() -> int:
    a=parse_args()
    try: *_,out=safe_paths(a)
    except BaseException: traceback.print_exc(file=sys.stderr); return 2
    out.mkdir(parents=False,exist_ok=True); tables={k:[] for k in ("runs","buckets","recurrence","paired","closure","decision")}
    try:
        report,rows=analyze(a,tables); publish(out,render(report,rows,tables)); return 0
    except BaseException as exc:
        tables={k:[] for k in tables}; tables["closure"].append({"scope":"failure","run":"","gate":"fail_closed_exception","required":"no exception",
            "observed":{"type":type(exc).__name__,"message":str(exc)},"passed":False,"blocking_reason":f"{type(exc).__name__}: {exc}"})
        tables["decision"].append({"decision":BLOCKED,"taxonomy_condition":"integrity or calculation failure","required":True,
            "observed":{"type":type(exc).__name__,"message":str(exc)},"passed":True})
        try: publish(out,render(blocked_report(a,exc),[],tables))
        except BaseException: traceback.print_exc(file=sys.stderr); return 3
        return 2

if __name__ == "__main__": raise SystemExit(main())
