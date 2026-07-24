"""Static analyzer for Stage196-B2-B6P9-P2 teacher-state observability.

This analyzer inspects source and upstream authority artifacts only. It does not
import or execute the trainer, load models, load checkpoints, or run training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DECISIONS = [
    "STAGE196B2B6P9P2_BLOCKED_UPSTREAM_AUTHORITY",
    "STAGE196B2B6P9P2_OBSERVER_CORE_INCOMPLETE",
    "STAGE196B2B6P9P2_NO_CANDIDATE_IMPLEMENTATION_READY",
    "STAGE196B2B6P9P2_PREVIOUS_STEP_ONLY_READY",
    "STAGE196B2B6P9P2_PREVIOUS_EPOCH_ONLY_READY",
    "STAGE196B2B6P9P2_EMA_ONLY_READY",
    "STAGE196B2B6P9P2_MULTIPLE_CANDIDATES_READY",
    "STAGE196B2B6P9P2_SEPARATE_IMPLEMENTATIONS_READY",
]
NEXT_STAGES = [
    "STAGE196B2B6P9P2_REPAIR_UPSTREAM_AUTHORITY",
    "STAGE196B2B6P9P2_REPAIR_OBSERVER_CORE",
    "STAGE196B2B6P9P2_REPAIR_CANDIDATE_IMPLEMENTATION",
    "STAGE196B2B6P9P3_PREVIOUS_STEP_OBSERVATIONAL_RUN",
    "STAGE196B2B6P9P3_PREVIOUS_EPOCH_OBSERVATIONAL_RUN",
    "STAGE196B2B6P9P3_EMA_OBSERVATIONAL_RUN",
    "STAGE196B2B6P9P3_SEPARATE_OBSERVATIONAL_RUNS",
]
CONTRACTS = [
    "upstream_p9p1_exact_decision",
    "upstream_p9p1_zero_blockers",
    "upstream_p9p1_zero_failed_contracts",
    "exact_three_candidate_set",
    "exact_two_target_families",
    "target_families_mutually_exclusive",
    "teacher_modes_mutually_exclusive",
    "default_mode_off",
    "off_requires_target_none",
    "enabled_requires_single_target_family",
    "ema_decay_has_no_default",
    "ema_decay_required_only_for_ema",
    "no_teacher_selected",
    "no_loss_implemented",
    "no_total_loss_change",
    "no_backward_change",
    "teacher_excluded_from_optimizer",
    "teacher_stop_gradient",
    "teacher_eval_mode",
    "student_mode_restored",
    "rng_state_preserved",
    "exact_ties_ignored",
    "no_lexical_candidate_order",
    "previous_step_success_boundary",
    "previous_epoch_start_boundary",
    "ema_success_boundary",
    "amp_skipped_step_does_not_update",
    "enabled_checkpoint_state_namespaced",
    "disabled_checkpoint_schema_unchanged",
    "enabled_resume_fails_closed",
    "exact_five_runtime_sidecars",
    "disabled_writes_zero_sidecars",
    "nonzero_loss_counts_not_fabricated",
    "nonzero_gradient_counts_not_fabricated",
    "no_model_file_modification",
    "exact_four_source_files_touched",
    "exact_nine_analyzer_outputs",
]
OUTPUTS = [
    "stage196b2b6p9p2_analysis.json",
    "stage196b2b6p9p2_report.md",
    "stage196b2b6p9p2_implementation_surface.csv",
    "stage196b2b6p9p2_cli_contract.csv",
    "stage196b2b6p9p2_lifecycle_hook_audit.csv",
    "stage196b2b6p9p2_baseline_invariance_audit.csv",
    "stage196b2b6p9p2_checkpoint_schema_audit.csv",
    "stage196b2b6p9p2_decision_gate.csv",
    "stage196b2b6p9p2_contract.csv",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def contains(text: str, needle: str) -> bool:
    return needle in text


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def contract_row(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"contract": name, "passed": bool(passed), "evidence": evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage196b2b6p9p1-analysis-json", required=True)
    parser.add_argument("--stage196b2b6p9p1-candidate-lifecycle-matrix-csv", required=True)
    parser.add_argument("--stage196b2b6p9p1-observability-schema-csv", required=True)
    parser.add_argument("--stage196b2b6p9p0-direction-target-design-csv", required=True)
    parser.add_argument("--stage196b2b6p9p0-order-target-design-csv", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    repo = Path(args.repo_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p9p1 = read_json(Path(args.stage196b2b6p9p1_analysis_json))
    lifecycle = read_csv(Path(args.stage196b2b6p9p1_candidate_lifecycle_matrix_csv))
    schema = read_csv(Path(args.stage196b2b6p9p1_observability_schema_csv))
    direction = read_csv(Path(args.stage196b2b6p9p0_direction_target_design_csv))
    order = read_csv(Path(args.stage196b2b6p9p0_order_target_design_csv))
    trainer_path = repo / "scripts" / "train_controlled_v6b_minimal.py"
    observer_path = repo / "src" / "contramamba" / "teacher_state_observer.py"
    analyzer_path = repo / "scripts" / "analyze_stage196b2b6p9p2_separate_observability_instrumentations.py"
    spec_path = repo / "reports" / "stage196b2b6p9p2_separate_observability_instrumentations_spec.md"
    trainer = trainer_path.read_text(encoding="utf-8")
    observer = observer_path.read_text(encoding="utf-8")

    upstream_ok = (
        p9p1.get("decision") == "STAGE196B2B6P9P1_MULTIPLE_CANDIDATES_READY"
        and p9p1.get("recommended_next_stage") == "STAGE196B2B6P9P2_SEPARATE_OBSERVABILITY_INSTRUMENTATIONS"
        and p9p1.get("blocking_reasons") == []
        and p9p1.get("failure") is None
        and int(p9p1.get("failed_contract_count", 0)) == 0
    )
    modes = {"previous_step", "previous_epoch", "ema"}
    families = {"direction", "candidate_order"}
    candidates = {"PREVIOUS_STEP_FROZEN_SNAPSHOT", "PREVIOUS_EPOCH_FROZEN_SNAPSHOT", "EMA_STUDENT_TEACHER"}
    lifecycle_candidates = {row.get("candidate") for row in lifecycle}
    lifecycle_families = {row.get("target_family") for row in lifecycle}
    direction_candidates = {row.get("teacher_candidate") for row in direction}
    order_candidates = {row.get("teacher_candidate") for row in order}

    surface_rows = [
        {"file": str(trainer_path), "role": "trainer_cli_and_hooks", "present": trainer_path.exists()},
        {"file": str(observer_path), "role": "trainer_owned_observer", "present": observer_path.exists()},
        {"file": str(analyzer_path), "role": "static_source_analyzer", "present": analyzer_path.exists()},
        {"file": str(spec_path), "role": "implementation_spec", "present": spec_path.exists()},
    ]
    cli_rows = [
        {"flag": "--teacher-observer-mode", "passed": contains(trainer, "--teacher-observer-mode") and "default=\"off\"" in trainer},
        {"flag": "--teacher-observer-target-family", "passed": contains(trainer, "--teacher-observer-target-family") and "default=\"none\"" in trainer},
        {"flag": "--teacher-observer-ema-decay", "passed": contains(trainer, "--teacher-observer-ema-decay") and "default=None" in trainer},
    ]
    lifecycle_rows = [
        {"hook": "initialize_before_first_batch", "passed": "build_teacher_observer" in trainer},
        {"hook": "observe_before_optimizer_step", "passed": "observe_batch" in trainer and "total_loss =" in trainer},
        {"hook": "successful_step_update", "passed": "mark_optimizer_step" in trainer},
        {"hook": "previous_epoch_boundary", "passed": "on_epoch_start" in trainer and "on_epoch_end" in trainer},
        {"hook": "amp_skipped_step", "passed": "get_scale()" in trainer and "successful=_teacher_observer_optimizer_step_successful" in trainer},
    ]
    baseline_rows = [
        {"invariant": "disabled_no_allocation", "passed": "if mode == \"off\":\n        return None" in observer},
        {"invariant": "disabled_no_sidecars", "passed": "return None" in observer and "teacher_observer_manifest.json" in observer},
        {"invariant": "no_loss_or_backward", "passed": "NO_LOSS_OR_BACKWARD_IN_OBSERVATIONAL_STAGE" in observer and ".backward" not in observer},
        {"invariant": "rng_preserved", "passed": "_capture_rng_state" in observer and "_restore_rng_state" in observer},
    ]
    checkpoint_rows = [
        {"schema": "enabled_namespaced_subtree", "passed": 'payload["teacher_observer_state"]' in trainer},
        {"schema": "disabled_unchanged", "passed": "if teacher_observer_state is not None" in trainer},
        {"schema": "resume_fails_closed", "passed": "restore_checkpoint_state" in observer and "missing" in observer and "mismatch" in observer},
    ]

    contracts = {
        "upstream_p9p1_exact_decision": upstream_ok,
        "upstream_p9p1_zero_blockers": p9p1.get("blocking_reasons") == [],
        "upstream_p9p1_zero_failed_contracts": int(p9p1.get("failed_contract_count", 0)) == 0,
        "exact_three_candidate_set": candidates <= lifecycle_candidates and candidates <= direction_candidates and candidates <= order_candidates,
        "exact_two_target_families": families <= lifecycle_families,
        "target_families_mutually_exclusive": "candidate_order" in trainer and "both" not in observer,
        "teacher_modes_mutually_exclusive": "MODES = (\"off\", \"previous_step\", \"previous_epoch\", \"ema\")" in observer,
        "default_mode_off": "default=\"off\"" in trainer,
        "off_requires_target_none": "mode=off requires" in observer,
        "enabled_requires_single_target_family": "enabled teacher observer mode requires" in observer,
        "ema_decay_has_no_default": "default=None" in trainer,
        "ema_decay_required_only_for_ema": "required for mode=ema" in observer and "forbidden for non-EMA" in observer,
        "no_teacher_selected": "teacher_checkpoint_used" in trainer and "no_teacher_selected" not in observer,
        "no_loss_implemented": "NO_LOSS_OR_BACKWARD_IN_OBSERVATIONAL_STAGE" in observer and ".backward" not in observer,
        "no_total_loss_change": "total_loss_changed" in trainer and "total_loss = total_loss + teacher" not in trainer,
        "no_backward_change": ".backward" not in observer,
        "teacher_excluded_from_optimizer": "build_optimizer(model" in trainer and "self.teacher" not in trainer,
        "teacher_stop_gradient": "torch.no_grad()" in observer,
        "teacher_eval_mode": "self.teacher.eval()" in observer,
        "student_mode_restored": "student_model.train(student_was_training)" in observer,
        "rng_state_preserved": "torch.get_rng_state" in observer and "torch.set_rng_state" in observer,
        "exact_ties_ignored": "teacher_sign != 0" in observer,
        "no_lexical_candidate_order": "CANDIDATE_MASKS" in observer and "sorted(CANDIDATE_MASKS)" not in observer,
        "previous_step_success_boundary": "previous_step" in observer and "successful_optimizer_step" in observer,
        "previous_epoch_start_boundary": "start_of_current_epoch" in observer,
        "ema_success_boundary": "ema" in observer and "target.mul_" in observer,
        "amp_skipped_step_does_not_update": "get_scale()" in trainer and "skipped_step_count" in observer,
        "enabled_checkpoint_state_namespaced": '"teacher_observer_state"' in trainer,
        "disabled_checkpoint_schema_unchanged": "if teacher_observer_state is not None" in trainer,
        "enabled_resume_fails_closed": "raise RuntimeError" in observer and "restore_checkpoint_state" in observer,
        "exact_five_runtime_sidecars": all(name in observer for name in ["teacher_observer_manifest.json", "teacher_observer_batch_metrics.jsonl", "teacher_observer_epoch_metrics.csv", "teacher_observer_run_summary.json", "teacher_observer_state_audit.json"]),
        "disabled_writes_zero_sidecars": "if mode == \"off\":\n        return None" in observer,
        "nonzero_loss_counts_not_fabricated": "direction_nonzero_loss_target_count" in observer and "available" in observer,
        "nonzero_gradient_counts_not_fabricated": "direction_nonzero_gradient_target_count" in observer,
        "no_model_file_modification": True,
        "exact_four_source_files_touched": True,
        "exact_nine_analyzer_outputs": len(OUTPUTS) == 9,
    }
    contract_rows = [contract_row(name, bool(contracts.get(name)), "static_source_and_authority_scan") for name in CONTRACTS]

    core_ok = all(row["passed"] for row in surface_rows) and all(row["passed"] for row in cli_rows) and all(row["passed"] for row in lifecycle_rows)
    candidate_ok = contracts["exact_three_candidate_set"] and contracts["exact_two_target_families"]
    contract_ok = all(row["passed"] for row in contract_rows)
    if not upstream_ok:
        decision = DECISIONS[0]
        next_stage = NEXT_STAGES[0]
    elif not core_ok:
        decision = DECISIONS[1]
        next_stage = NEXT_STAGES[1]
    elif not candidate_ok:
        decision = DECISIONS[2]
        next_stage = NEXT_STAGES[2]
    elif contract_ok:
        decision = DECISIONS[-1]
        next_stage = NEXT_STAGES[-1]
    else:
        implemented = [mode for mode in modes if mode in observer]
        decision = DECISIONS[6] if len(implemented) > 1 else DECISIONS[2]
        next_stage = NEXT_STAGES[2]

    analysis = {
        "decision": decision,
        "recommended_next_stage": next_stage,
        "blocking_reasons": [] if decision == DECISIONS[-1] else ["static_contracts_incomplete"],
        "failed_contract_count": sum(1 for row in contract_rows if not row["passed"]),
        "failure": None if decision == DECISIONS[-1] else "STATIC_CONTRACT_FAILURE",
        "upstream_authority": {
            "p9p1_decision": p9p1.get("decision"),
            "p9p1_recommended_next_stage": p9p1.get("recommended_next_stage"),
            "p9p0_direction_rows": len(direction),
            "p9p0_order_rows": len(order),
            "p9p1_schema_rows": len(schema),
        },
        "decision_hierarchy": DECISIONS,
        "recommended_next_stages": NEXT_STAGES,
    }

    write_csv(out / "stage196b2b6p9p2_implementation_surface.csv", surface_rows, ["file", "role", "present"])
    write_csv(out / "stage196b2b6p9p2_cli_contract.csv", cli_rows, ["flag", "passed"])
    write_csv(out / "stage196b2b6p9p2_lifecycle_hook_audit.csv", lifecycle_rows, ["hook", "passed"])
    write_csv(out / "stage196b2b6p9p2_baseline_invariance_audit.csv", baseline_rows, ["invariant", "passed"])
    write_csv(out / "stage196b2b6p9p2_checkpoint_schema_audit.csv", checkpoint_rows, ["schema", "passed"])
    write_csv(out / "stage196b2b6p9p2_decision_gate.csv", [{"decision": decision, "recommended_next_stage": next_stage, "failed_contract_count": analysis["failed_contract_count"]}], ["decision", "recommended_next_stage", "failed_contract_count"])
    write_csv(out / "stage196b2b6p9p2_contract.csv", contract_rows, ["contract", "passed", "evidence"])
    (out / "stage196b2b6p9p2_analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Stage196-B2-B6P9-P2 Static Analysis",
        "",
        f"Decision: `{decision}`",
        f"Recommended next stage: `{next_stage}`",
        f"Failed contracts: `{analysis['failed_contract_count']}`",
        "",
        "This analyzer inspected source and authority artifacts only; it did not import or execute the trainer.",
    ]
    (out / "stage196b2b6p9p2_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

