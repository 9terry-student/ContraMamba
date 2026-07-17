#!/usr/bin/env python3
"""Build the Stage188-A paired manifests without executing either arm.

This program performs file/JSON inspection only.  It deliberately does not use
subprocess, import the trainer, import torch, load a checkpoint, or train.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


DATA_REL = "data/controlled_v5_v3_without_time_swap.jsonl"
DATA_SHA = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
SIDECAR_REL = (
    "reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/"
    "stage185a_controlled_train_integrity_sidecar.jsonl"
)
SIDECAR_SHA = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
SUCCESS = "STAGE188A_PAIRED_INTERNAL_MARGIN_EXPERIMENT_SPEC_READY"
BLOCKED = "STAGE188A_PAIRED_INTERNAL_MARGIN_EXPERIMENT_BLOCKED"
NEXT = "STAGE188B_PAIRED_INTERNAL_MARGIN_TRAINING"

ALLOWED_FIELDS = {
    "compatible_positive_margin_weight",
    "compatible_positive_margin_logit",
    "controlled_integrity_sidecar_path",
    "expected_integrity_sidecar_semantic_sha256",
    "output_json",
    "output_predictions_json",
    "stage115_clean_dev_scalar_output_jsonl",
    "run_directory",
    "run_name",
}

REQUIRED_PARSED = {
    "architecture": "v6b_minimal",
    "backbone": "mamba",
    "model_name": "state-spaces/mamba-130m-hf",
    "seed": 174,
    "epochs": 20,
    "device": "cuda",
    "select_metric": "final_macro_f1",
    "stage174c_clean_pairwise_mode": "off",
    "stage174c_clean_pairwise_weight": 0.0,
    "stage175b_support_anchor_mode": "off",
    "stage175b_support_anchor_weight": 0.0,
    "stage177c_frame_pairwise_mode": "off",
    "stage177c_frame_pairwise_weight": 0.0,
}

CSV_HEADERS = {
    "stage188a_authoritative_input_identity.csv":
        ["artifact", "path", "expected_identity", "observed_identity", "passed", "evidence"],
    "stage188a_baseline_lineage_resolution.csv":
        ["field", "expected", "observed", "source", "passed", "status"],
    "stage188a_common_configuration.csv":
        ["field", "value_json", "source"],
    "stage188a_allowed_configuration_diff.csv":
        ["field", "baseline_json", "intervention_json", "allowed", "reason"],
    "stage188a_forbidden_configuration_diff.csv":
        ["field", "baseline_json", "intervention_json", "forbidden", "reason"],
    "stage188a_execution_contract.csv":
        ["requirement", "baseline", "intervention", "failure_behavior"],
    "stage188a_analysis_contract.csv":
        ["section", "requirement", "classification"],
    "stage188a_stage188b_gate.csv":
        ["gate", "required", "observed", "passed", "blocking_reason"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage174d1-dir", type=Path, required=True)
    parser.add_argument("--stage186a-dir", type=Path, required=True)
    parser.add_argument("--stage187b-dir", type=Path, required=True)
    parser.add_argument("--trainer-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", default=DATA_SHA)
    parser.add_argument("--expected-sidecar-semantic-sha256", default=SIDECAR_SHA)
    return parser.parse_args()


def json_safe(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def semantic_sidecar_sha(path: Path) -> tuple[str, int]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"sidecar line {number} is not an object")
            rows.append({key: row[key] for key in sorted(row) if key != "created_at"})
    return hashlib.sha256(json_safe(rows).encode("utf-8")).hexdigest(), len(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def locate_json_with_decision(directory: Path, expected: str) -> tuple[Path | None, dict[str, Any] | None]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.rglob("*.json")) if directory.exists() else []:
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("decision") == expected:
            matches.append((path, payload))
    return matches[0] if len(matches) == 1 else (None, None)


def provenance_candidates(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.rglob("*.json")) if directory.exists() else []:
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("parsed_args"), dict) and isinstance(payload.get("raw_sys_argv"), list):
            found.append((path, payload))
    return found


def equivalent_provenance(candidates: list[tuple[Path, dict[str, Any]]]) -> bool:
    if not candidates:
        return False
    signatures = {
        json_safe({
            "parsed_args": item.get("parsed_args"),
            "resolved_runtime_config": item.get("resolved_runtime_config"),
            "training_selection_policy": item.get("training_selection_policy"),
            "data_provenance": item.get("data_provenance"),
            "raw_sys_argv": item.get("raw_sys_argv"),
            "git_commit": (item.get("source_provenance") or {}).get("git_commit"),
        })
        for _, item in candidates
    }
    return len(signatures) == 1


def set_option(argv: list[str], name: str, value: Any | None, *, flag: bool = False) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == name:
            index += 1
            if not flag and index < len(argv) and not argv[index].startswith("--"):
                index += 1
            continue
        if token.startswith(name + "="):
            index += 1
            continue
        result.append(token)
        index += 1
    if value is not None:
        result.append(name)
        if not flag:
            result.append(str(value))
    return result


def arm_argv(base: list[str], arm: str, output_dir: Path) -> list[str]:
    run_dir = output_dir / f"stage188b_{arm}"
    argv = list(base)
    settings = {
        "--output-json": run_dir / "training_report.json",
        "--output-predictions-json": run_dir / "clean_dev_predictions.json",
        "--stage115-clean-dev-scalar-output-jsonl": run_dir / "clean_dev_scalars.jsonl",
        "--compatible-positive-margin-weight": 0.0 if arm == "baseline" else 0.05,
        "--compatible-positive-margin-logit": 0.0,
        "--controlled-integrity-sidecar-path": None if arm == "baseline" else SIDECAR_REL,
        "--expected-integrity-sidecar-semantic-sha256": None if arm == "baseline" else SIDECAR_SHA,
    }
    for option, value in settings.items():
        argv = set_option(argv, option, value)
    return argv


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key in sorted(value):
        child = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value[key], dict):
            result.update(flatten(value[key], child))
        else:
            result[child] = value[key]
    return result


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    stage174 = args.stage174d1_dir.resolve()
    stage186 = args.stage186a_dir.resolve()
    stage187 = args.stage187b_dir.resolve()
    trainer = args.trainer_source.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    blockers: list[str] = []
    identity_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    def gate(name: str, required: Any, observed: Any, passed: bool, reason: str = "") -> None:
        gate_rows.append({"gate": name, "required": json_safe(required), "observed": json_safe(observed),
                          "passed": passed, "blocking_reason": "" if passed else reason})
        if not passed:
            blockers.append(f"{name}: {reason}")

    dataset = repo / DATA_REL
    sidecar = repo / SIDECAR_REL
    observed_data_sha = file_sha(dataset) if dataset.is_file() else None
    try:
        observed_sidecar_sha, sidecar_rows = semantic_sidecar_sha(sidecar) if sidecar.is_file() else (None, 0)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        observed_sidecar_sha, sidecar_rows = None, 0
        blockers.append(f"sidecar semantic SHA failed: {exc}")

    for artifact, path, expected, observed, evidence in (
        ("dataset", dataset, args.expected_dataset_sha256, observed_data_sha, "file SHA-256"),
        ("sidecar", sidecar, args.expected_sidecar_semantic_sha256, observed_sidecar_sha, "canonical rows excluding created_at"),
        ("trainer", trainer, "existing regular file", file_sha(trainer) if trainer.is_file() else None, "file SHA-256 recorded"),
    ):
        passed = observed is not None and (expected == "existing regular file" or observed == expected)
        identity_rows.append({"artifact": artifact, "path": str(path), "expected_identity": expected,
                              "observed_identity": observed, "passed": passed, "evidence": evidence})
        gate(f"authoritative_{artifact}_identity", expected, observed, passed, "missing or identity mismatch")
    gate("sidecar_row_count", 3600, sidecar_rows, sidecar_rows == 3600, "authoritative sidecar must contain 3,600 rows")

    _, stage186_report = locate_json_with_decision(stage186, "STAGE186A_FIXED_NO_SWEEP_COMPATIBLE_POSITIVE_MARGIN_SPEC_READY")
    _, stage187_report = locate_json_with_decision(stage187, "STAGE187B_DEFAULT_OFF_IMPLEMENTATION_RUNTIME_VALIDATION_PASSED")
    gate("stage186_decision", "STAGE186A_FIXED_NO_SWEEP_COMPATIBLE_POSITIVE_MARGIN_SPEC_READY",
         stage186_report.get("decision") if stage186_report else None, stage186_report is not None,
         "exactly one authoritative Stage186-A decision report is required")
    gate("stage187_decision", "STAGE187B_DEFAULT_OFF_IMPLEMENTATION_RUNTIME_VALIDATION_PASSED",
         stage187_report.get("decision") if stage187_report else None, stage187_report is not None,
         "exactly one authoritative Stage187-B decision report is required")
    if stage187_report:
        gate("stage187_checks", "14/14",
             f"{stage187_report.get('checks_passed')}/{stage187_report.get('checks_total')}",
             stage187_report.get("checks_passed") == 14 and stage187_report.get("checks_total") == 14,
             "Stage187-B must report 14/14")

    candidates = provenance_candidates(stage174)
    unambiguous = equivalent_provenance(candidates)
    gate("stage174d1_unambiguous_provenance", "one configuration or identical duplicates",
         [str(path) for path, _ in candidates], unambiguous,
         "no provenance found or plausible Stage174-D1 artifacts disagree")
    provenance = candidates[0][1] if unambiguous else {}
    provenance_path = candidates[0][0] if unambiguous else None
    parsed = deepcopy(provenance.get("parsed_args") or {})
    resolved = deepcopy(provenance.get("resolved_runtime_config") or {})
    selection = deepcopy(provenance.get("training_selection_policy") or {})
    raw_argv = list(provenance.get("raw_sys_argv") or [])
    git_commit = (provenance.get("source_provenance") or {}).get("git_commit")
    gate("stage174d1_git_commit", "non-empty commit", git_commit, isinstance(git_commit, str) and bool(git_commit),
         "authoritative Stage174-D1 provenance lacks git commit")
    gate("stage174d1_raw_argv", "non-empty argv array", raw_argv, bool(raw_argv),
         "authoritative Stage174-D1 provenance lacks raw argv")

    for field, expected in REQUIRED_PARSED.items():
        observed = parsed.get(field)
        passed = observed == expected
        lineage_rows.append({"field": field, "expected": json_safe(expected), "observed": json_safe(observed),
                             "source": str(provenance_path) if provenance_path else "unresolved",
                             "passed": passed, "status": "resolved" if passed else "blocked"})
        gate(f"baseline_{field}", expected, observed, passed, "Stage174-D1 lineage value missing or inconsistent")

    policy_checks = {
        "clean_dev_only_checkpoint_selection": True,
        "checkpoint_selection_metric": "final_macro_f1",
        "external_evaluation_used_for_training": False,
        "external_evaluation_used_for_calibration": False,
        "external_evaluation_used_for_threshold_selection": False,
        "external_evaluation_used_for_checkpoint_selection": False,
        "time_swap_included_in_main_classification_training": False,
        "loss_logits_used_for_final_classifier_ce": False,
    }
    for field, expected in policy_checks.items():
        observed = selection.get(field)
        passed = observed == expected
        lineage_rows.append({"field": f"training_selection_policy.{field}", "expected": json_safe(expected),
                             "observed": json_safe(observed), "source": str(provenance_path) if provenance_path else "unresolved",
                             "passed": passed, "status": "resolved" if passed else "blocked"})
        gate(f"policy_{field}", expected, observed, passed, "selection/safety policy missing or inconsistent")

    common = {
        "git_commit": git_commit,
        "trainer_source": str(trainer),
        "trainer_sha256": file_sha(trainer) if trainer.is_file() else None,
        "dataset_path": DATA_REL,
        "dataset_sha256": observed_data_sha,
        "parsed_args": parsed,
        "resolved_runtime_config": resolved,
        "training_selection_policy": selection,
    }
    baseline_config = deepcopy(common)
    intervention_config = deepcopy(common)
    for target, weight, sidecar_path, sidecar_sha in (
        (baseline_config, 0.0, None, None),
        (intervention_config, 0.05, SIDECAR_REL, SIDECAR_SHA),
    ):
        target["compatible_positive_margin_weight"] = weight
        target["compatible_positive_margin_logit"] = 0.0
        target["controlled_integrity_sidecar_path"] = sidecar_path
        target["expected_integrity_sidecar_semantic_sha256"] = sidecar_sha

    baseline_argv = arm_argv(raw_argv, "baseline", output)
    intervention_argv = arm_argv(raw_argv, "intervention", output)
    baseline_config["run_directory"] = str(output / "stage188b_baseline")
    baseline_config["run_name"] = "stage188b_baseline"
    intervention_config["run_directory"] = str(output / "stage188b_intervention")
    intervention_config["run_name"] = "stage188b_intervention"

    flat_base, flat_intervention = flatten(baseline_config), flatten(intervention_config)
    allowed_rows: list[dict[str, Any]] = []
    forbidden_rows: list[dict[str, Any]] = []
    for field in sorted(set(flat_base) | set(flat_intervention)):
        left, right = flat_base.get(field), flat_intervention.get(field)
        if left == right:
            continue
        leaf = field.rsplit(".", 1)[-1]
        row = {"field": field, "baseline_json": json_safe(left), "intervention_json": json_safe(right)}
        if leaf in ALLOWED_FIELDS:
            allowed_rows.append({**row, "allowed": True, "reason": "precommitted arm or descriptive output difference"})
        else:
            forbidden_rows.append({**row, "forbidden": True, "reason": "not in exact allowed-difference set"})
    gate("forbidden_configuration_diff", 0, len(forbidden_rows), len(forbidden_rows) == 0,
         "one or more non-authorized configuration fields differ")

    execution_rows = [
        {"requirement": "training invocation", "baseline": "manual Stage188-B only", "intervention": "manual Stage188-B only", "failure_behavior": "builder never executes"},
        {"requirement": "margin weight", "baseline": "0.0", "intervention": "0.05", "failure_behavior": "block"},
        {"requirement": "margin logit", "baseline": "0.0", "intervention": "0.0", "failure_behavior": "block"},
        {"requirement": "sidecar access", "baseline": "forbidden", "intervention": "authoritative Stage185 required", "failure_behavior": "block"},
        {"requirement": "external evaluation", "baseline": "forbidden", "intervention": "forbidden", "failure_behavior": "block"},
        {"requirement": "checkpoint selection", "baseline": "clean-dev final_macro_f1", "intervention": "clean-dev final_macro_f1", "failure_behavior": "block"},
    ]
    analysis_rows = [
        {"section": "identity", "requirement": "completion, commit, dataset, seed, config, exact allowed differences", "classification": "blocking"},
        {"section": "clean_dev", "requirement": "metrics, recalls, counts, confusion and error families", "classification": "guardrail"},
        {"section": "transitions", "requirement": "exact clean-dev row-ID prediction transition audit", "classification": "diagnostic"},
        {"section": "native_frame", "requirement": "native frame_logit/frame_prob only; never final logits", "classification": "blocking if required evidence absent"},
        {"section": "stage182b", "requirement": "13 compatible FN, 1 incompatible FP, controls and clean-model failures", "classification": "prior-selected diagnostic"},
        {"section": "mechanism", "requirement": "eligible observations/loss/active rate/logit/zero batches", "classification": "mechanism check"},
    ]

    decision = SUCCESS if not blockers else BLOCKED
    runnable = decision == SUCCESS
    manifests = {
        "baseline": {"stage": "Stage188-A", "arm": "baseline", "runnable": runnable,
                     "common_configuration": common, "arm_configuration": baseline_config,
                     "argv": baseline_argv if runnable else [],
                     "command_preview": shlex.join(["python", str(trainer), *baseline_argv]) if runnable else None},
        "intervention": {"stage": "Stage188-A", "arm": "intervention", "runnable": runnable,
                         "common_configuration": common, "arm_configuration": intervention_config,
                         "argv": intervention_argv if runnable else [],
                         "command_preview": shlex.join(["python", str(trainer), *intervention_argv]) if runnable else None},
    }
    report = {
        "stage": "Stage188-A", "decision": decision, "authorized_next": NEXT if runnable else None,
        "training_performed": False, "manifest_runnable": runnable, "blocking_reasons": blockers,
        "authoritative_identity": {"dataset_sha256": observed_data_sha, "sidecar_semantic_sha256": observed_sidecar_sha,
                                   "sidecar_rows": sidecar_rows, "trainer_sha256": common["trainer_sha256"]},
        "stage174d1": {"provenance_path": str(provenance_path) if provenance_path else None,
                       "candidate_paths": [str(path) for path, _ in candidates], "git_commit": git_commit},
        "allowed_difference_fields": sorted(ALLOWED_FIELDS), "forbidden_difference_count": len(forbidden_rows),
        "single_seed_policy": "diagnostic_only_not_conclusive", "unresolved_static_risks": blockers,
    }

    write_json(output / "stage188a_paired_internal_margin_manifest_report.json", report)
    write_json(output / "stage188a_baseline_manifest.json", manifests["baseline"])
    write_json(output / "stage188a_intervention_manifest.json", manifests["intervention"])
    common_rows = [{"field": key, "value_json": json_safe(value), "source": str(provenance_path) if provenance_path else "Stage188-A fixed"}
                   for key, value in sorted(flatten(common).items())]
    csv_payloads = {
        "stage188a_authoritative_input_identity.csv": identity_rows,
        "stage188a_baseline_lineage_resolution.csv": lineage_rows,
        "stage188a_common_configuration.csv": common_rows,
        "stage188a_allowed_configuration_diff.csv": allowed_rows,
        "stage188a_forbidden_configuration_diff.csv": forbidden_rows,
        "stage188a_execution_contract.csv": execution_rows,
        "stage188a_analysis_contract.csv": analysis_rows,
        "stage188a_stage188b_gate.csv": gate_rows,
    }
    for filename, headers in CSV_HEADERS.items():
        write_csv(output / filename, headers, csv_payloads.get(filename, []))
    markdown = f"""# Stage188-A paired internal margin manifest report

**Decision:** `{decision}`

- Training performed: no
- Runnable manifests: {'yes' if runnable else 'no'}
- Stage174-D1 provenance: `{provenance_path}`
- Dataset SHA-256: `{observed_data_sha}`
- Sidecar semantic SHA-256: `{observed_sidecar_sha}`
- Forbidden configuration differences: `{len(forbidden_rows)}`

## Blocking reasons

{chr(10).join('- ' + item for item in blockers) if blockers else '- None.'}

## Authorization

{'The validated manifests authorize only `' + NEXT + '`.' if runnable else 'Stage188-B remains unauthorized until every blocking gate passes.'}
"""
    (output / "stage188a_paired_internal_margin_manifest_report.md").write_text(markdown, encoding="utf-8")
    return 0 if runnable else 2


if __name__ == "__main__":
    raise SystemExit(main())
