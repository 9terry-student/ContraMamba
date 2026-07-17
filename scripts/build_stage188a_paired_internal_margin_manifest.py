#!/usr/bin/env python3
"""Build current-commit Stage188-A paired manifests without executing either arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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
HISTORICAL_REFERENCE_DECISION = "STAGE174D1F_CLEAN_LOCAL_PAIRWISE_OBJECTIVE_DIRECTION_CONFLICT_PATH_CLOSED"
HISTORICAL_RECOVERY_DECISION = "STAGE174D1_EXACT_HISTORICAL_RUN_NOT_RECOVERABLE"
BASELINE_DEFINITION = "current_commit_default_off_paired_baseline"
NON_BLOCKING_HISTORICAL_MISSING_PROVENANCE = (
    "exact argv",
    "historical git commit",
    "full resolved runtime config",
)
SUCCESS = "STAGE188A_PAIRED_INTERNAL_MARGIN_EXPERIMENT_SPEC_READY"
BLOCKED = "STAGE188A_PAIRED_INTERNAL_MARGIN_EXPERIMENT_BLOCKED"
NEXT = "STAGE188B_PAIRED_INTERNAL_MARGIN_TRAINING"

EXPECTED_HISTORICAL_SCOPE = {
    "architecture": "v6b_minimal",
    "backbone": "mamba",
    "seed": 174,
    "epochs": 20,
    "main_data": DATA_REL,
    "time_swap_excluded": True,
    "checkpoint_selection": "internal_clean_dev_only",
    "final_classifier_ce_source": 'output["logits"]',
    "loss_logits_used": False,
    "external_evaluation_run": False,
}

ALLOWED_ARG_DIFFS = {
    "compatible_positive_margin_weight",
    "controlled_integrity_sidecar_path",
    "expected_integrity_sidecar_semantic_sha256",
    "output_json",
    "output_predictions_json",
    "stage115_clean_dev_scalar_output_jsonl",
}

CSV_HEADERS = {
    "stage188a_authoritative_input_identity.csv":
        ["artifact", "path", "expected_identity", "observed_identity", "passed", "evidence"],
    "stage188a_baseline_lineage_resolution.csv":
        ["field", "expected", "observed", "source", "passed", "status"],
    "stage188a_common_configuration.csv": ["field", "value_json", "source"],
    "stage188a_allowed_configuration_diff.csv":
        ["field", "baseline_json", "intervention_json", "allowed", "reason"],
    "stage188a_forbidden_configuration_diff.csv":
        ["field", "baseline_json", "intervention_json", "forbidden", "reason"],
    "stage188a_execution_contract.csv":
        ["requirement", "baseline", "intervention", "failure_behavior"],
    "stage188a_analysis_contract.csv": ["section", "requirement", "classification"],
    "stage188a_stage188b_gate.csv":
        ["gate", "required", "observed", "passed", "blocking_reason"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--stage174d1-reference",
        type=Path,
        required=True,
        help=(
            "Stage188-A historical baseline recovery closure JSON; this is not "
            "original Stage174-D1F runtime provenance"
        ),
    )
    parser.add_argument("--stage186a-dir", type=Path, required=True)
    parser.add_argument("--stage187b-dir", type=Path, required=True)
    parser.add_argument("--trainer-source", type=Path, required=True)
    parser.add_argument("--current-git-commit", required=True)
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


def parser_options(repo: Path, trainer: Path) -> tuple[set[str], list[str]]:
    sources = [trainer]
    inherited = repo / "scripts" / "train_controlled_v5.py"
    if inherited.is_file():
        sources.append(inherited)
    found: set[str] = set()
    evidence: list[str] = []
    pattern = re.compile(r"add_argument\(\s*[\"'](--[a-z0-9-]+)[\"']", re.IGNORECASE)
    for source in sources:
        text = source.read_text(encoding="utf-8")
        options = set(pattern.findall(text))
        found.update(options)
        evidence.append(f"{source}:{len(options)} options")
    return found, evidence


def set_option(argv: list[str], option: str, value: Any | None) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            index += 2
            continue
        if token.startswith(option + "="):
            index += 1
            continue
        result.append(token)
        index += 1
    if value is not None:
        result.extend([option, str(value)])
    return result


def argv_map(argv: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional argv token: {token}")
        key = token[2:].replace("-", "_")
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            result[key] = True
            index += 1
        else:
            result[key] = argv[index + 1]
            index += 2
    return result


def common_argv() -> list[str]:
    return [
        "--data", DATA_REL,
        "--architecture", "v6b_minimal",
        "--backbone", "mamba",
        "--model-name", "state-spaces/mamba-130m-hf",
        "--device", "cuda",
        "--seed", "174",
        "--epochs", "20",
        "--select-metric", "final_macro_f1",
        "--stage174c-clean-pairwise-mode", "off",
        "--stage174c-clean-pairwise-weight", "0.0",
        "--stage175b-support-anchor-mode", "off",
        "--stage175b-support-anchor-weight", "0.0",
        "--stage177c-frame-pairwise-mode", "off",
        "--stage177c-frame-pairwise-weight", "0.0",
        "--compatible-positive-margin-logit", "0.0",
    ]


def arm_argv(common: list[str], arm: str, output_dir: Path) -> list[str]:
    run_dir = output_dir / f"stage188b_{arm}"
    settings: list[tuple[str, Any | None]] = [
        ("--compatible-positive-margin-weight", 0.0 if arm == "baseline" else 0.05),
        ("--controlled-integrity-sidecar-path", None if arm == "baseline" else SIDECAR_REL),
        ("--expected-integrity-sidecar-semantic-sha256", None if arm == "baseline" else SIDECAR_SHA),
        ("--output-json", run_dir / "training_report.json"),
        ("--output-predictions-json", run_dir / "clean_dev_predictions.json"),
        ("--stage115-clean-dev-scalar-output-jsonl", run_dir / "clean_dev_scalars.jsonl"),
    ]
    result = list(common)
    for option, value in settings:
        result = set_option(result, option, value)
    return result


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    reference_path = args.stage174d1_reference.resolve()
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

    try:
        if not reference_path.is_file():
            raise FileNotFoundError(f"recovery closure JSON does not exist: {reference_path}")
        loaded_reference = read_json(reference_path)
        if not isinstance(loaded_reference, dict):
            raise ValueError("recovery closure JSON root is not an object")
        reference = loaded_reference
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        reference = {}
        gate(
            "historical_recovery_closure_input",
            "existing well-formed Stage188-A recovery closure JSON object",
            str(reference_path),
            False,
            f"historical recovery closure unavailable or malformed: {exc}",
        )
    else:
        gate(
            "historical_recovery_closure_input",
            "existing well-formed Stage188-A recovery closure JSON object",
            str(reference_path),
            True,
        )
    gate("historical_reference_decision", HISTORICAL_REFERENCE_DECISION,
         reference.get("reference_decision"), reference.get("reference_decision") == HISTORICAL_REFERENCE_DECISION,
         "historical reference decision mismatch")
    gate("historical_recovery_decision", HISTORICAL_RECOVERY_DECISION,
         reference.get("decision"), reference.get("decision") == HISTORICAL_RECOVERY_DECISION,
         "historical recovery closure mismatch")
    gate("historical_reference_only", True, reference.get("historical_reference_only"),
         reference.get("historical_reference_only") is True, "Stage174-D1 must be reference-only")
    gate("exact_historical_run_recoverable", False, reference.get("exact_historical_run_recoverable"),
         reference.get("exact_historical_run_recoverable") is False, "exact historical recovery must remain false")
    gate("baseline_definition", BASELINE_DEFINITION, reference.get("baseline_definition"),
         reference.get("baseline_definition") == BASELINE_DEFINITION, "paired baseline definition mismatch")

    scope = reference.get("experimental_scope")
    scope_is_object = isinstance(scope, dict)
    gate(
        "historical_experimental_scope_object",
        "object",
        type(scope).__name__ if scope is not None else None,
        scope_is_object,
        "recovery closure experimental_scope must be an object",
    )
    if not scope_is_object:
        scope = {}
    for field, expected in EXPECTED_HISTORICAL_SCOPE.items():
        observed = scope.get(field)
        passed = str(observed).lower() == expected if field == "backbone" else observed == expected
        lineage_rows.append({"field": field, "expected": json_safe(expected), "observed": json_safe(observed),
                             "source": str(reference_path), "passed": passed,
                             "status": "historical_reference_only" if passed else "blocked"})
        gate(f"historical_scope_{field}", expected, observed, passed, "historical reference scope mismatch")

    current_commit = str(args.current_git_commit).strip()
    gate("current_git_commit", "non-empty caller-supplied commit", current_commit,
         bool(current_commit), "--current-git-commit must be non-empty")

    dataset = repo / DATA_REL
    sidecar = repo / SIDECAR_REL
    observed_data_sha = file_sha(dataset) if dataset.is_file() else None
    try:
        observed_sidecar_sha, sidecar_rows = semantic_sidecar_sha(sidecar) if sidecar.is_file() else (None, 0)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        observed_sidecar_sha, sidecar_rows = None, 0
        blockers.append(f"sidecar semantic SHA failed: {exc}")
    trainer_sha = file_sha(trainer) if trainer.is_file() else None
    for artifact, path, expected, observed, evidence in (
        ("dataset", dataset, args.expected_dataset_sha256, observed_data_sha, "file SHA-256"),
        ("sidecar", sidecar, args.expected_sidecar_semantic_sha256, observed_sidecar_sha, "semantic SHA-256"),
        ("trainer", trainer, "existing regular file", trainer_sha, "current trainer file SHA-256"),
    ):
        passed = observed is not None and (expected == "existing regular file" or observed == expected)
        identity_rows.append({"artifact": artifact, "path": str(path), "expected_identity": expected,
                              "observed_identity": observed, "passed": passed, "evidence": evidence})
        gate(f"authoritative_{artifact}_identity", expected, observed, passed, "missing or identity mismatch")
    gate("sidecar_row_count", 3600, sidecar_rows, sidecar_rows == 3600,
         "authoritative sidecar must contain 3,600 rows")

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

    common = common_argv()
    available_options, option_evidence = parser_options(repo, trainer) if trainer.is_file() else (set(), [])
    emitted_options = sorted({token for token in common if token.startswith("--")} | {
        "--compatible-positive-margin-weight", "--controlled-integrity-sidecar-path",
        "--expected-integrity-sidecar-semantic-sha256", "--output-json",
        "--output-predictions-json", "--stage115-clean-dev-scalar-output-jsonl",
    })
    missing_options = sorted(set(emitted_options) - available_options)
    gate("current_parser_options", [], missing_options, not missing_options,
         "one or more emitted options do not exist in the statically inspected current parser")

    baseline_argv = arm_argv(common, "baseline", output)
    intervention_argv = arm_argv(common, "intervention", output)
    base_map, intervention_map = argv_map(baseline_argv), argv_map(intervention_argv)
    allowed_rows: list[dict[str, Any]] = []
    forbidden_rows: list[dict[str, Any]] = []
    for field in sorted(set(base_map) | set(intervention_map)):
        left, right = base_map.get(field), intervention_map.get(field)
        if left == right:
            continue
        row = {"field": field, "baseline_json": json_safe(left), "intervention_json": json_safe(right)}
        if field in ALLOWED_ARG_DIFFS:
            allowed_rows.append({**row, "allowed": True, "reason": "precommitted arm/output argv difference"})
        else:
            forbidden_rows.append({**row, "forbidden": True, "reason": "not in exact allowed argv-difference set"})
    gate("forbidden_argv_difference", 0, len(forbidden_rows), not forbidden_rows,
         "one or more non-authorized argv fields differ")
    gate("margin_logit_equal", "0.0/0.0",
         [base_map.get("compatible_positive_margin_logit"), intervention_map.get("compatible_positive_margin_logit")],
         base_map.get("compatible_positive_margin_logit") == intervention_map.get("compatible_positive_margin_logit") == "0.0",
         "margin logit must be explicitly equal at 0.0")

    historical_status = {
        "historical_reference_only": True,
        "exact_historical_run_recoverable": False,
        "historical_recovery_decision": HISTORICAL_RECOVERY_DECISION,
        "historical_reference_decision": HISTORICAL_REFERENCE_DECISION,
        "missing_provenance_non_blocking": list(NON_BLOCKING_HISTORICAL_MISSING_PROVENANCE),
    }
    common_configuration = {
        "current_git_commit": current_commit,
        "trainer_path": str(trainer),
        "trainer_sha256": trainer_sha,
        "dataset_path": DATA_REL,
        "dataset_sha256": observed_data_sha,
        "sidecar_path": SIDECAR_REL,
        "sidecar_semantic_sha256": observed_sidecar_sha,
        "baseline_definition": BASELINE_DEFINITION,
        "historical_reference_status": historical_status,
        "common_argv": common,
        "omitted_parser_defaults_identity": {
            "current_git_commit": current_commit,
            "trainer_sha256": trainer_sha,
        },
    }

    for field, value in (
        ("historical_exact_run_recoverable", False),
        ("historical_reference_decision", HISTORICAL_REFERENCE_DECISION),
        ("historical_recovery_decision", HISTORICAL_RECOVERY_DECISION),
        ("historical_seed", 174), ("historical_epochs", 20),
        ("baseline_definition", BASELINE_DEFINITION),
        ("current_git_commit", current_commit), ("trainer_sha256", trainer_sha),
    ):
        lineage_rows.append({"field": field, "expected": json_safe(value), "observed": json_safe(value),
                             "source": "Stage188-A closure/current identity", "passed": True,
                             "status": "historical_reference_only" if field.startswith("historical_") else "current_paired_baseline"})

    execution_rows = [
        {"requirement": "training invocation", "baseline": "manual Stage188-B only", "intervention": "manual Stage188-B only", "failure_behavior": "builder never executes"},
        {"requirement": "current commit/trainer/common argv", "baseline": "identical", "intervention": "identical", "failure_behavior": "block"},
        {"requirement": "margin weight", "baseline": "0.0", "intervention": "0.05", "failure_behavior": "block"},
        {"requirement": "margin logit", "baseline": "0.0", "intervention": "0.0", "failure_behavior": "block"},
        {"requirement": "sidecar access", "baseline": "forbidden", "intervention": "authoritative Stage185 required", "failure_behavior": "block"},
        {"requirement": "external evaluation", "baseline": "forbidden", "intervention": "forbidden", "failure_behavior": "block"},
    ]
    analysis_rows = [
        {"section": "identity", "requirement": "same current commit, trainer SHA, common argv; exact allowed differences", "classification": "blocking"},
        {"section": "native_frame", "requirement": "direct finite scalar JSONL frame_logit; exact row-ID join", "classification": "blocking"},
        {"section": "stage182b", "requirement": "13 compatible FN, 1 incompatible FP, controls and clean-model failures", "classification": "prior-selected diagnostic"},
        {"section": "guardrails", "requirement": "retain precommitted clean and mechanism gates", "classification": "decision"},
    ]

    decision = SUCCESS if not blockers else BLOCKED
    runnable = decision == SUCCESS
    manifests: dict[str, Any] = {}
    for arm, arm_args, arm_map in (
        ("baseline", baseline_argv, base_map), ("intervention", intervention_argv, intervention_map)
    ):
        manifests[arm] = {
            "stage": "Stage188-A", "arm": arm, "runnable": runnable,
            "historical_reference_only": True,
            "baseline_definition": BASELINE_DEFINITION,
            "common_configuration": common_configuration,
            "arm_configuration": arm_map,
            "argv": arm_args if runnable else [],
            "command_preview": shlex.join(["python", str(trainer), *arm_args]) if runnable else None,
        }
    report = {
        "stage": "Stage188-A", "decision": decision, "authorized_next": NEXT if runnable else None,
        "training_performed": False, "manifest_runnable": runnable, "blocking_reasons": blockers,
        "historical_reference_only": True,
        "exact_historical_run_recoverable": False,
        "historical_recovery_decision": HISTORICAL_RECOVERY_DECISION,
        "historical_missing_provenance_non_blocking": list(
            NON_BLOCKING_HISTORICAL_MISSING_PROVENANCE
        ),
        "baseline_definition": BASELINE_DEFINITION,
        "current_git_commit": current_commit,
        "trainer_path": str(trainer), "trainer_sha256": trainer_sha,
        "common_argv": common,
        "parser_static_inspection": {"sources": option_evidence, "missing_emitted_options": missing_options},
        "authoritative_identity": {"dataset_sha256": observed_data_sha,
                                   "sidecar_semantic_sha256": observed_sidecar_sha, "sidecar_rows": sidecar_rows},
        "allowed_argv_difference_fields": sorted(ALLOWED_ARG_DIFFS),
        "forbidden_argv_difference_count": len(forbidden_rows),
        "single_seed_policy": "diagnostic_only_not_conclusive",
        "unresolved_static_risks": blockers,
    }

    write_json(output / "stage188a_paired_internal_margin_manifest_report.json", report)
    write_json(output / "stage188a_baseline_manifest.json", manifests["baseline"])
    write_json(output / "stage188a_intervention_manifest.json", manifests["intervention"])
    common_rows = [{"field": key, "value_json": json_safe(value), "source": "Stage188-A current-commit construction"}
                   for key, value in sorted(common_configuration.items())]
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
        write_csv(output / filename, headers, csv_payloads[filename])
    markdown = f"""# Stage188-A paired internal margin manifest report

**Decision:** `{decision}`

- Historical reference only: yes
- Exact historical run recoverable: no
- Historical recovery decision: `{HISTORICAL_RECOVERY_DECISION}`
- Baseline definition: `{BASELINE_DEFINITION}`
- Current Git commit: `{current_commit}`
- Trainer SHA-256: `{trainer_sha}`
- Training performed: no
- Runnable manifests: {'yes' if runnable else 'no'}
- Forbidden argv differences: `{len(forbidden_rows)}`

## Blocking reasons

{chr(10).join('- ' + item for item in blockers) if blockers else '- None.'}

## Authorization

{'The validated manifests authorize only `' + NEXT + '`.' if runnable else 'Stage188-B remains unauthorized until every blocking gate passes.'}
"""
    (output / "stage188a_paired_internal_margin_manifest_report.md").write_text(markdown, encoding="utf-8")
    return 0 if runnable else 2


if __name__ == "__main__":
    raise SystemExit(main())
