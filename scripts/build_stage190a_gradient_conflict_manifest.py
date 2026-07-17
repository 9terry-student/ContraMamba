#!/usr/bin/env python3
"""Build six fail-closed Stage190-B manifests. This script never loads a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SEEDS = (174, 175, 176)
ARMS = ("baseline", "intervention")
TRAINING_COMMIT = "bee2f5ad452d1d9f57b30f444d18835dbffdbecf"
TRAINER_SHA = "24b01c5799c762772fe1700204afae59f8566898f65e7f3eefa4ac57ac6f126f"
DATA_SHA = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
SIDECAR_SHA = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
READY189 = "STAGE189A_THREE_SEED_MARGIN_REPLICATION_AND_POSTHOC_REFERENCE_SPEC_READY"
NEGATIVE189 = "STAGE189D_THREE_SEED_MARGIN_NEGATIVE_OR_REGRESSIVE"
READY = "STAGE190A_GRADIENT_CONFLICT_DIAGNOSTIC_MANIFEST_READY"
BLOCKED = "STAGE190A_GRADIENT_CONFLICT_DIAGNOSTIC_MANIFEST_BLOCKED"
INVALIDATED_COMMIT = "21c733533317a5d5aff447a98cb4efeeaec4ee49"
STATUS_COUNTS = {"ELIGIBLE": 605, "INELIGIBLE": 716, "UNRESOLVED": 119}
DATA_REL = "data/controlled_v5_v3_without_time_swap.jsonl"
HELPER_REL = "scripts/export_external_scalars_from_checkpoint.py"
DIAGNOSTIC_REL = "scripts/run_stage190b_gradient_conflict_diagnostic.py"
DIAGNOSTIC_SOURCE_RELS = (
    "scripts/build_stage190a_gradient_conflict_manifest.py",
    "scripts/run_stage190b_gradient_conflict_diagnostic.py",
    "scripts/analyze_stage190c_gradient_conflict.py",
)
REQUIRED_FALSE_REPORT_FIELDS = (
    "external_data_used_for_training",
    "external_metrics_used_for_threshold_tuning",
    "stage57_external_data_used_for_training",
    "stage57_external_metrics_used_for_threshold_tuning",
    "stage66_external_data_used_for_training",
    "stage66_external_metrics_used_for_threshold_tuning",
    "stage75_external_data_used_for_training",
    "stage75_external_metrics_used_for_threshold_tuning",
    "stage80a_external_data_used_for_training",
    "stage80a_external_metrics_used_for_threshold_tuning",
    "stage15_used_for_training",
    "stage15_used_for_checkpoint_selection",
    "stage15_used_for_loss_selection",
    "stage15_used_for_final_logit_modifier_selection",
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--stage189a-dir", type=Path, required=True)
    p.add_argument("--stage189d-analysis-dir", type=Path, required=True)
    p.add_argument("--stage182b-dir", type=Path, required=True)
    p.add_argument("--stage185a-dir", type=Path, required=True)
    p.add_argument("--current-diagnostic-git-commit", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as h:
        return json.load(h)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as h:
        for number, line in enumerate(h, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            rows.append(value)
    return rows


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_bytes(repo: Path, *argv: str) -> bytes:
    result = subprocess.run(
        ["git", *argv], cwd=repo, check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(argv)} failed: {detail}")
    return result.stdout


def git_quiet(repo: Path, *argv: str) -> bool:
    result = subprocess.run(
        ["git", *argv], cwd=repo, check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in (0, 1):
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(argv)} failed: {detail}")
    return result.returncode == 0


def find_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        if key in value:
            found.append(value[key])
        for child in value.values():
            found.extend(find_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_values(child, key))
    return found




def resolved_optional_path(repo: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return (path if path.is_absolute() else repo / path).resolve()


def external_use_contract(parsed: dict[str, Any], training_report: Any) -> dict[str, Any]:
    parsed_expected = {
        "enable_stage43_external_eval": False,
        "stage43_external_enable_shadow_export": False,
        "stage43_external_factver_jsonl": [],
        "external_eval_jsonl": [],
        "ood_data": None,
        "ood_ablation_modes": None,
        "ood_unflagged_ne_shift_sweep": None,
        "ood_selective_ne_shift_sweep": None,
        "stage57_bridge_train_mode": "none",
        "stage57_bridge_train_jsonl": None,
        "stage66_bridge_train_mode": "none",
        "stage66_bridge_train_jsonl": None,
        "stage75_bridge_train_mode": "none",
        "stage75_bridge_train_jsonl": None,
        "stage80a_bridge_train_mode": "none",
        "stage80a_bridge_train_jsonl": None,
    }
    parsed_observed = {key: parsed.get(key) for key in parsed_expected}
    parsed_checks = {key: parsed_observed[key] == expected for key, expected in parsed_expected.items()}
    report_observed = {key: find_values(training_report, key) for key in REQUIRED_FALSE_REPORT_FIELDS}
    missing_required = [key for key, values in report_observed.items() if not values]
    non_false_required = [
        key for key, values in report_observed.items()
        if values and any(value is not False for value in values)
    ]
    return {
        "contract_version": "stage190a_explicit_external_use_v2",
        "required_false_report_fields": list(REQUIRED_FALSE_REPORT_FIELDS),
        "observed_required_false_report_field_values": report_observed,
        "missing_required_report_fields": missing_required,
        "non_false_required_report_fields": non_false_required,
        "parsed_expected": parsed_expected,
        "parsed_observed": parsed_observed,
        "parsed_checks": parsed_checks,
        "passed": (
            all(parsed_checks.values())
            and not missing_required
            and not non_false_required
        ),
    }


def arm_runtime_contract(*, arm: str, parsed: dict[str, Any], training_report: Any,
                         repo: Path, authoritative_sidecar: Path) -> dict[str, Any]:
    aggregate_schema_path = "$.runs.single.compatible_positive_margin"
    runs = training_report.get("runs") if isinstance(training_report, dict) else None
    single_run = runs.get("single") if isinstance(runs, dict) else None
    aggregate_present = (
        isinstance(single_run, dict)
        and "compatible_positive_margin" in single_run
    )
    aggregate_value = (
        single_run["compatible_positive_margin"]
        if aggregate_present else None
    )
    aggregate_is_dictionary = isinstance(aggregate_value, dict)
    aggregate = aggregate_value if aggregate_is_dictionary else {}
    sidecar_value = aggregate.get("sidecar_contract") if aggregate_is_dictionary else None
    sidecar = sidecar_value if isinstance(sidecar_value, dict) else {}
    observed = {
        "parsed_weight": parsed.get("compatible_positive_margin_weight"),
        "parsed_margin_logit": parsed.get("compatible_positive_margin_logit"),
        "parsed_sidecar_path": str(resolved_optional_path(repo, parsed.get("controlled_integrity_sidecar_path"))) if parsed.get("controlled_integrity_sidecar_path") else None,
        "parsed_expected_sidecar_sha": parsed.get("expected_integrity_sidecar_semantic_sha256"),
        "aggregate_enabled": aggregate.get("enabled"),
        "aggregate_configured_weight": aggregate.get("configured_weight"),
        "aggregate_configured_margin_logit": aggregate.get("configured_margin_logit"),
        "aggregate_sidecar_accessed": sidecar.get("sidecar_accessed"),
        "eligible_count": aggregate.get("compatible_positive_margin_eligible_count"),
        "eligible_observation_count": aggregate.get(
            "compatible_positive_margin_eligible_observation_count"
        ),
        "score_source": aggregate.get("score_source"),
        "normalization": aggregate.get("normalization"),
        "sidecar_contract": sidecar_value,
    }
    common = (
        aggregate_present
        and aggregate_is_dictionary
        and isinstance(sidecar_value, dict)
        and observed["parsed_margin_logit"] == 0.0
    )
    if arm == "baseline":
        checks = {
            "aggregate_present": aggregate_present,
            "aggregate_is_dictionary": aggregate_is_dictionary,
            "sidecar_contract_is_dictionary": isinstance(sidecar_value, dict),
            "common": common,
            "parsed_weight": observed["parsed_weight"] == 0.0,
            "parsed_sidecar_path": observed["parsed_sidecar_path"] is None,
            "parsed_expected_sidecar_sha": observed["parsed_expected_sidecar_sha"] is None,
            "aggregate_enabled": observed["aggregate_enabled"] is False,
            "aggregate_configured_weight": observed["aggregate_configured_weight"] == 0.0,
            "aggregate_configured_margin_logit": observed["aggregate_configured_margin_logit"] == 0.0,
            "aggregate_sidecar_accessed": observed["aggregate_sidecar_accessed"] is False,
            "eligible_count": observed["eligible_count"] == 0,
            "eligible_observation_count": observed["eligible_observation_count"] == 0,
            "score_source": observed["score_source"] == 'output["frame_logit"]',
            "normalization": observed["normalization"] == "eligible_row_mean",
        }
    else:
        checks = {
            "aggregate_present": aggregate_present,
            "aggregate_is_dictionary": aggregate_is_dictionary,
            "sidecar_contract_is_dictionary": isinstance(sidecar_value, dict),
            "common": common,
            "parsed_weight": observed["parsed_weight"] == 0.05,
            "parsed_sidecar_path": resolved_optional_path(repo, parsed.get("controlled_integrity_sidecar_path")) == authoritative_sidecar,
            "parsed_expected_sidecar_sha": observed["parsed_expected_sidecar_sha"] == SIDECAR_SHA,
            "aggregate_enabled": observed["aggregate_enabled"] is True,
            "aggregate_configured_weight": observed["aggregate_configured_weight"] == 0.05,
            "aggregate_configured_margin_logit": observed["aggregate_configured_margin_logit"] == 0.0,
            "aggregate_sidecar_accessed": observed["aggregate_sidecar_accessed"] is True,
            "eligible_count": observed["eligible_count"] == 605,
            "eligible_observation_count": observed["eligible_observation_count"] == 12100,
            "score_source": observed["score_source"] == 'output["frame_logit"]',
            "normalization": observed["normalization"] == "eligible_row_mean",
            "sidecar_semantic_sha": sidecar.get("observed_sidecar_semantic_sha256") == SIDECAR_SHA,
            "train_split_row_ids_exact": sidecar.get("train_split_row_ids_exact") is True,
            "dev_split_row_ids_exact": sidecar.get("dev_split_row_ids_exact") is True,
            "expected_frozen_split_exact_id_gate": sidecar.get("expected_frozen_split_exact_id_gate") is True,
            "aligned_train_rows": sidecar.get("aligned_train_rows") == 2880,
            "aligned_eligible_rows": sidecar.get("aligned_eligible_rows") == 605,
        }
    return {
        "arm": arm,
        "aggregate_schema_path": aggregate_schema_path,
        "aggregate_present": aggregate_present,
        "aggregate_is_dictionary": aggregate_is_dictionary,
        "exact_observed_aggregate": aggregate_value,
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
    }

def semantic_sidecar_sha(rows: list[dict[str, Any]]) -> str:
    semantic = [{k: row[k] for k in sorted(row) if k != "created_at"} for row in rows]
    payload = json.dumps(semantic, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str | None:
    value = row.get("row_id", row.get("id", row.get("stable_id")))
    return str(value) if value is not None else None


def unique_index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row_id(row)
        if key is None or key in result:
            raise ValueError(f"{name} has a missing or duplicate row ID")
        result[key] = row
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def resolve_run_dir(stage189a: Path, manifest: dict[str, Any], seed: int, arm: str) -> Path:
    candidates = []
    raw = manifest.get("run_directory")
    if raw:
        candidates.append(Path(raw))
    candidates.extend((stage189a / f"stage189b_seed{seed}_{arm}", stage189a.parent / f"stage189b_seed{seed}_{arm}"))
    existing = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and resolved not in existing:
            existing.append(resolved)
    if len(existing) != 1:
        raise ValueError(f"seed{seed} {arm}: expected one exact run directory, found {existing}")
    return existing[0]


def exact_file(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    return path.resolve()


def stage189c_files(run_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    reports = []
    for path in sorted(run_dir.glob("*.json")):
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("decision") in {
            "STAGE189C_POSTHOC_MARGIN_REFERENCE_EXPORTED", "STAGE189C_POSTHOC_MARGIN_REFERENCE_BLOCKED"
        }:
            reports.append((path.resolve(), value))
    if len(reports) != 1:
        raise ValueError(f"{run_dir}: expected exactly one Stage189-C report")
    report_path, report = reports[0]
    output_jsonl = Path(str(report.get("output_jsonl", "")))
    if not output_jsonl.is_absolute():
        output_jsonl = run_dir / output_jsonl
    output_jsonl = output_jsonl.resolve()
    if not output_jsonl.is_file():
        raise ValueError(f"Stage189-C JSONL missing: {output_jsonl}")
    return output_jsonl, report_path, report


def main() -> int:
    args = parse_args()
    repo, stage189a = args.repo_root.resolve(), args.stage189a_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    gates: list[dict[str, Any]] = []

    def gate(name: str, required: Any, observed: Any, passed: bool, reason: str) -> None:
        gates.append({"gate": name, "required": json.dumps(required, sort_keys=True),
                      "observed": json.dumps(observed, sort_keys=True), "passed": passed,
                      "blocking_reason": "" if passed else reason})
        if not passed:
            blockers.append(f"{name}: {reason}")

    diagnostic = (repo / DIAGNOSTIC_REL).resolve()
    helper = (repo / HELPER_REL).resolve()
    dataset = (repo / DATA_REL).resolve()
    diagnostic_sha = file_sha(diagnostic) if diagnostic.is_file() else None
    helper_sha = file_sha(helper) if helper.is_file() else None
    dataset_sha = file_sha(dataset) if dataset.is_file() else None
    diagnostic_source_rows: list[dict[str, Any]] = []
    actual_head: str | None = None
    try:
        actual_head = git_bytes(repo, "rev-parse", "HEAD").decode("ascii").strip()
        gate("diagnostic_head_identity", args.current_diagnostic_git_commit.strip(), actual_head,
             actual_head == args.current_diagnostic_git_commit.strip(),
             "actual repository HEAD differs from supplied diagnostic commit")
        for relative in DIAGNOSTIC_SOURCE_RELS:
            working_path = (repo / relative).resolve()
            working_sha = file_sha(working_path) if working_path.is_file() else None
            commit_bytes = git_bytes(repo, "show", f"{args.current_diagnostic_git_commit.strip()}:{relative}")
            commit_sha = bytes_sha(commit_bytes)
            unstaged_clean = git_quiet(repo, "diff", "--quiet", "--", relative)
            staged_clean = git_quiet(repo, "diff", "--cached", "--quiet", "--", relative)
            exact = working_sha == commit_sha and unstaged_clean and staged_clean
            diagnostic_source_rows.append({
                "artifact": "diagnostic_source", "path": str(working_path),
                "working_sha256": working_sha, "commit_blob_sha256": commit_sha,
                "head_commit": actual_head,
                "supplied_commit": args.current_diagnostic_git_commit.strip(),
                "head_matches": actual_head == args.current_diagnostic_git_commit.strip(),
                "working_matches_commit_blob": working_sha == commit_sha,
                "unstaged_diff_clean": unstaged_clean, "staged_diff_clean": staged_clean,
                "passed": exact,
            })
            gate(f"diagnostic_source_{Path(relative).stem}", True,
                 diagnostic_source_rows[-1], exact,
                 "diagnostic source working bytes or tracked diff differ from supplied commit blob")
    except Exception as exc:
        blockers.append(f"diagnostic repository identity audit failed: {type(exc).__name__}: {exc}")
    gate("diagnostic_commit_nonempty", "non-empty", args.current_diagnostic_git_commit.strip(),
         bool(args.current_diagnostic_git_commit.strip()), "diagnostic commit is empty")
    gate("diagnostic_script_present", True, diagnostic_sha, diagnostic_sha is not None, "Stage190-B script missing")
    gate("checkpoint_helper_present", True, helper_sha, helper_sha is not None, "checkpoint helper missing")
    gate("dataset_sha256", DATA_SHA, dataset_sha, dataset_sha == DATA_SHA, "dataset identity mismatch")

    try:
        manifest_report = read_json(stage189a / "stage189a_manifest_report.json")
    except Exception as exc:
        manifest_report = {}
        blockers.append(f"Stage189-A report unreadable: {exc}")
    gate("stage189a_ready", READY189, manifest_report.get("decision"),
         manifest_report.get("decision") == READY189, "Stage189-A is not READY")
    gate("stage189a_training_commit", TRAINING_COMMIT, manifest_report.get("current_git_commit"),
         manifest_report.get("current_git_commit") == TRAINING_COMMIT, "training commit mismatch")
    gate("stage189a_trainer_sha", TRAINER_SHA, manifest_report.get("trainer_sha256"),
         manifest_report.get("trainer_sha256") == TRAINER_SHA, "trainer SHA mismatch")
    gate("training_and_diagnostic_identity_separate", "not required equal",
         {"training": TRAINING_COMMIT, "diagnostic": args.current_diagnostic_git_commit.strip()}, True, "")

    closure_candidates = [args.stage189d_analysis_dir.resolve() / "stage189d_three_seed_analysis_report.json",
                          args.stage189d_analysis_dir.resolve() / "stage189d_three_seed_margin_negative_closure.json"]
    closure_path = next((p for p in closure_candidates if p.is_file()), None)
    try:
        closure = read_json(closure_path) if closure_path else {}
    except Exception as exc:
        closure = {}
        blockers.append(f"Stage189-D closure unreadable: {exc}")
    gate("stage189d_negative_closure", NEGATIVE189, closure.get("decision"),
         closure.get("decision") == NEGATIVE189, "Stage189-D decision mismatch")

    sidecar_path = args.stage185a_dir.resolve() / "stage185a_controlled_train_integrity_sidecar.jsonl"
    try:
        sidecar_rows = read_jsonl(sidecar_path)
        sidecar_index = unique_index(sidecar_rows, "Stage185-A sidecar")
        observed_sidecar_sha = semantic_sidecar_sha(sidecar_rows)
    except Exception as exc:
        sidecar_rows, sidecar_index, observed_sidecar_sha = [], {}, None
        blockers.append(f"Stage185-A sidecar unreadable: {exc}")
    train = [r for r in sidecar_rows if r.get("split") == "train"]
    dev = [r for r in sidecar_rows if r.get("split") == "dev"]
    compatible = [r for r in train if type(r.get("frame_compatible_label")) is int and r.get("frame_compatible_label") == 1]
    counts = Counter(r.get("integrity_status") for r in compatible)
    topology = {"total": len(sidecar_index), "train": len(train), "dev": len(dev),
                "train_compatible": len(compatible), **{k: counts.get(k, 0) for k in STATUS_COUNTS}}
    expected_topology = {"total": 3600, "train": 2880, "dev": 720, "train_compatible": 1440, **STATUS_COUNTS}
    gate("sidecar_semantic_sha256", SIDECAR_SHA, observed_sidecar_sha,
         observed_sidecar_sha == SIDECAR_SHA, "sidecar semantic identity mismatch")
    gate("sidecar_topology", expected_topology, topology, topology == expected_topology, "sidecar topology mismatch")

    manifests: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    checkpoint_shas: list[str] = []
    for seed in SEEDS:
        for arm in ARMS:
            local_blockers: list[str] = []
            arm_manifest_path = stage189a / f"stage189a_seed{seed}_{arm}_manifest.json"
            try:
                arm_manifest = read_json(arm_manifest_path)
                run_dir = resolve_run_dir(stage189a, arm_manifest, seed, arm)
                provenance_path = exact_file(run_dir, "run_provenance.json")
                checkpoint = exact_file(run_dir, "selected_checkpoint.pt")
                training_report = exact_file(run_dir, "training_report.json")
                training_report_value = read_json(training_report)
                predictions = exact_file(run_dir, "clean_dev_predictions.json")
                scalars = exact_file(run_dir, "clean_dev_scalars.jsonl")
                posthoc_jsonl, posthoc_report_path, posthoc_report = stage189c_files(run_dir)
                provenance = read_json(provenance_path)
                parsed = provenance.get("parsed_args") or {}
                source = provenance.get("source_provenance") or {}
                split = provenance.get("split_seed_contract") or {}
                run_data = (provenance.get("data_provenance") or {}).get("main_data") or {}
                posthoc_rows = read_jsonl(posthoc_jsonl)
                posthoc_counts = Counter(r.get("integrity_status") for r in posthoc_rows)
                posthoc_ids = unique_index(posthoc_rows, f"seed{seed} {arm} Stage189-C")
                checkpoint_sha = file_sha(checkpoint)
                provenance_checkpoint = ((provenance.get("finalization") or {}).get("selected_checkpoint") or {})
                row_identities = {json.dumps(row.get("checkpoint_identity"), sort_keys=True) for row in posthoc_rows}
                external_contract = external_use_contract(parsed, training_report_value)
                runtime_contract = arm_runtime_contract(
                    arm=arm, parsed=parsed, training_report=training_report_value,
                    repo=repo, authoritative_sidecar=sidecar_path.resolve(),
                )
                checks = {
                    "run_completed": provenance.get("status") == "completed",
                    "training_commit": source.get("git_commit") == TRAINING_COMMIT,
                    "trainer_sha": source.get("trainer_sha256") == TRAINER_SHA,
                    "dataset_sha": run_data.get("sha256") == DATA_SHA,
                    "training_seed": parsed.get("seed") == seed and split.get("training_seed") == seed,
                    "split_seed": parsed.get("split_seed") == 174 and split.get("resolved_split_seed") == 174,
                    "split_rows": split.get("clean_main_train_rows") == 2880 and split.get("clean_main_dev_rows") == 720,
                    "not_pre_split": source.get("git_commit") != INVALIDATED_COMMIT,
                    "stage189c_exported": posthoc_report.get("decision") == "STAGE189C_POSTHOC_MARGIN_REFERENCE_EXPORTED",
                    "stage189c_hash": posthoc_report.get("output_jsonl_sha256") == file_sha(posthoc_jsonl),
                    "checkpoint_provenance": provenance_checkpoint.get("sha256") == checkpoint_sha,
                    "stage189c_identity": posthoc_report.get("seed") == seed and posthoc_report.get("arm") == arm
                        and posthoc_report.get("training_seed") == seed and posthoc_report.get("split_seed") == 174
                        and posthoc_report.get("git_commit") == TRAINING_COMMIT and posthoc_report.get("trainer_sha256") == TRAINER_SHA
                        and posthoc_report.get("dataset_sha256") == DATA_SHA and posthoc_report.get("sidecar_semantic_sha256") == SIDECAR_SHA
                        and posthoc_report.get("checkpoint_sha256") == checkpoint_sha
                        and posthoc_report.get("checkpoint_helper_sha256") == helper_sha and len(row_identities) == 1,
                    "stage189c_topology": len(posthoc_ids) == 1440 and all(posthoc_counts.get(k, 0) == v for k, v in STATUS_COUNTS.items()),
                    "clean_artifacts_present": all(p.is_file() for p in (training_report, predictions, scalars)),
                    "external_use_contract": external_contract["passed"],
                    "arm_runtime_contract": runtime_contract["passed"],
                }
                local_blockers.extend(name for name, passed in checks.items() if not passed)
                checkpoint_shas.append(checkpoint_sha)
                argv = [
                    "--repo-root", str(repo), "--manifest", str((output / f"stage190a_seed{seed}_{arm}_manifest.json").resolve()),
                    "--stage182b-dir", str(args.stage182b_dir.resolve()), "--stage185a-dir", str(args.stage185a_dir.resolve()),
                    "--output-dir", str((output / f"stage190b_seed{seed}_{arm}").resolve()),
                ]
                stage182_candidate = (args.stage182b_dir.resolve() / "stage182b_candidate_localization.csv")
                stage182_controls = (args.stage182b_dir.resolve() / "stage182b_matched_control_pairs.csv")
                if not stage182_candidate.is_file() or not stage182_controls.is_file():
                    raise ValueError("required Stage182-B cohort artifacts are missing")
                artifact_hashes = {"run_provenance": file_sha(provenance_path), "training_report": file_sha(training_report),
                    "clean_dev_predictions": file_sha(predictions), "clean_dev_scalars": file_sha(scalars),
                    "stage189c_jsonl": file_sha(posthoc_jsonl), "stage189c_report": file_sha(posthoc_report_path),
                    "stage182b_candidate_localization": file_sha(stage182_candidate),
                    "stage182b_matched_control_pairs": file_sha(stage182_controls)}
                manifest = {"stage": "Stage190-A", "seed": seed, "arm": arm, "training_seed": seed,
                    "split_seed": 174, "training_git_commit": TRAINING_COMMIT,
                    "diagnostic_git_commit": args.current_diagnostic_git_commit.strip(), "trainer_sha256": TRAINER_SHA,
                    "diagnostic_script_path": str(diagnostic), "diagnostic_script_sha256": diagnostic_sha,
                    "diagnostic_source_identity": diagnostic_source_rows,
                    "checkpoint_helper_path": str(helper), "checkpoint_helper_sha256": helper_sha,
                    "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                    "run_directory": str(run_dir), "run_provenance_path": str(provenance_path),
                    "run_provenance_sha256": artifact_hashes["run_provenance"], "training_report_path": str(training_report),
                    "clean_dev_predictions_path": str(predictions), "clean_dev_scalars_path": str(scalars),
                    "stage189c_jsonl_path": str(posthoc_jsonl), "stage189c_jsonl_sha256": artifact_hashes["stage189c_jsonl"],
                    "stage189c_report_path": str(posthoc_report_path), "stage189c_report_sha256": artifact_hashes["stage189c_report"],
                    "stage182b_candidate_localization_path": str(stage182_candidate),
                    "stage182b_matched_control_pairs_path": str(stage182_controls),
                    "dataset_path": str(dataset), "dataset_sha256": DATA_SHA, "sidecar_path": str(sidecar_path.resolve()),
                    "sidecar_semantic_sha256": SIDECAR_SHA, "fixed_split_identity": {"split_seed": 174, "train": 2880, "dev": 720},
                    "external_use_contract": external_contract,
                    "arm_runtime_contract": runtime_contract,
                    "artifact_hashes": artifact_hashes, "argv": argv, "command_preview": shlex.join(["python", str(diagnostic), *argv]),
                    "runnable": False, "blocking_reasons": local_blockers}
            except Exception as exc:
                manifest = {"stage": "Stage190-A", "seed": seed, "arm": arm, "training_seed": seed,
                    "split_seed": 174, "training_git_commit": TRAINING_COMMIT,
                    "diagnostic_git_commit": args.current_diagnostic_git_commit.strip(), "trainer_sha256": TRAINER_SHA,
                    "diagnostic_script_sha256": diagnostic_sha, "checkpoint_helper_sha256": helper_sha,
                    "runnable": False, "argv": [], "blocking_reasons": [f"artifact inspection failed: {exc}"]}
                local_blockers = list(manifest["blocking_reasons"])
            if local_blockers:
                blockers.extend(f"seed{seed}_{arm}: {reason}" for reason in local_blockers)
            manifests.append(manifest)

    distinct = len(checkpoint_shas) == 6 and len(set(checkpoint_shas)) == 6
    gate("six_distinct_checkpoint_sha256", 6, len(set(checkpoint_shas)), distinct, "checkpoint hashes missing or not distinct")
    global_runnable = not blockers
    for manifest in manifests:
        manifest["runnable"] = global_runnable and not manifest.get("blocking_reasons")
        if not manifest["runnable"]:
            manifest["argv"] = []
            manifest["command_preview"] = None
        write_json(output / f"stage190a_seed{manifest['seed']}_{manifest['arm']}_manifest.json", manifest)
        matrix.append({"seed": manifest["seed"], "arm": manifest["arm"], "training_seed": manifest.get("training_seed"),
                       "split_seed": manifest.get("split_seed"), "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                       "external_use_contract_passed": (manifest.get("external_use_contract") or {}).get("passed"),
                       "arm_runtime_contract_passed": (manifest.get("arm_runtime_contract") or {}).get("passed"),
                       "runnable": manifest["runnable"], "blocking_reasons": " | ".join(manifest.get("blocking_reasons", []))})

    identity_rows = [
        {"artifact": "dataset", "path": str(dataset), "expected_sha256": DATA_SHA, "observed_sha256": dataset_sha, "passed": dataset_sha == DATA_SHA},
        {"artifact": "sidecar_semantic", "path": str(sidecar_path), "expected_sha256": SIDECAR_SHA, "observed_sha256": observed_sidecar_sha, "passed": observed_sidecar_sha == SIDECAR_SHA},
        {"artifact": "trainer", "path": "run provenance", "expected_sha256": TRAINER_SHA, "observed_sha256": manifest_report.get("trainer_sha256"), "passed": manifest_report.get("trainer_sha256") == TRAINER_SHA},
        {"artifact": "diagnostic_script", "path": str(diagnostic), "expected_sha256": "current bytes", "observed_sha256": diagnostic_sha, "passed": diagnostic_sha is not None},
        {"artifact": "checkpoint_helper", "path": str(helper), "expected_sha256": "current bytes", "observed_sha256": helper_sha, "passed": helper_sha is not None},
    ]
    identity_rows.extend(diagnostic_source_rows)
    decision = READY if global_runnable else BLOCKED
    report = {"stage": "Stage190-A", "decision": decision, "runnable": global_runnable,
              "training_performed": False, "checkpoint_loaded": False, "manifest_count": 6,
              "training_git_commit": TRAINING_COMMIT, "diagnostic_git_commit": args.current_diagnostic_git_commit.strip(),
              "actual_diagnostic_head": actual_head, "diagnostic_source_identity": diagnostic_source_rows,
              "trainer_sha256": TRAINER_SHA, "diagnostic_script_sha256": diagnostic_sha,
              "checkpoint_helper_sha256": helper_sha, "dataset_sha256": DATA_SHA,
              "sidecar_semantic_sha256": SIDECAR_SHA, "cohort_topology": expected_topology,
              "blocking_reasons": blockers}
    write_json(output / "stage190a_manifest_report.json", report)
    write_csv(output / "stage190a_gate.csv", ["gate", "required", "observed", "passed", "blocking_reason"], gates)
    write_csv(output / "six_checkpoint_matrix.csv", ["seed", "arm", "training_seed", "split_seed", "checkpoint_sha256", "external_use_contract_passed", "arm_runtime_contract_passed", "runnable", "blocking_reasons"], matrix)
    write_csv(output / "authoritative_input_identity.csv", ["artifact", "path", "expected_sha256", "observed_sha256", "working_sha256", "commit_blob_sha256", "head_commit", "supplied_commit", "head_matches", "working_matches_commit_blob", "unstaged_diff_clean", "staged_diff_clean", "passed"], identity_rows)
    md = f"""# Stage190-A gradient-conflict manifest report

**Decision:** `{decision}`

- Six selected checkpoints distinct: {distinct}
- Training commit: `{TRAINING_COMMIT}`
- Diagnostic commit: `{args.current_diagnostic_git_commit.strip()}` (separate identity; equality is not required)
- Dataset/sidecar topology: `2880/720`, train-compatible `1440`, `605/716/119`
- Training performed: no

## Blocking reasons

{chr(10).join('- ' + item for item in blockers) if blockers else '- None.'}
"""
    (output / "stage190a_manifest_report.md").write_text(md, encoding="utf-8")
    return 0 if global_runnable else 2


if __name__ == "__main__":
    raise SystemExit(main())
