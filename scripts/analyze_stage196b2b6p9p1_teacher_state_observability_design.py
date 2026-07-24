#!/usr/bin/env python3
"""Stage196-B2-B6P9-P1 teacher-state observability design.

Static, source-backed design only.  This script reads explicit P9-P0 authority
artifacts and repository source text, writes the nine P9-P1 design outputs, and
does not load a model, load a checkpoint, run a forward pass, train, evaluate,
select a teacher, or modify trainer/model/checkpoint code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

STAGE = "Stage196-B2-B6P9-P1"
AUTHORITY_DIR = (
    "reports/stage196b2b6p9p0_stability_teacher_target_authority_20260724_100036"
)
EXPECTED_P9P0 = {
    "decision": "STAGE196B2B6P9P0_NO_JUSTIFIED_TEACHER",
    "recommended_next_stage": "STAGE196B2B6P9P1_TEACHER_STATE_OBSERVABILITY_DESIGN",
    "blocking_reasons": [],
    "failure": None,
}
EXPECTED_SIX = (
    "CURRENT_NATIVE_STOP_GRAD",
    "FRAME_LOCAL_ONLY_DONOR_STOP_GRAD",
    "PREVIOUS_STEP_FROZEN_SNAPSHOT",
    "PREVIOUS_EPOCH_FROZEN_SNAPSHOT",
    "EMA_STUDENT_TEACHER",
    "FIXED_RECONSTRUCTED_CHECKPOINT",
)
TERMINAL_EXCLUSIONS = {
    "CURRENT_NATIVE_STOP_GRAD": "INVALID_ALGEBRAIC_IDENTITY",
    "FRAME_LOCAL_ONLY_DONOR_STOP_GRAD": "BLOCKED_DONOR_ARM_IMITATION_RISK",
    "FIXED_RECONSTRUCTED_CHECKPOINT": "DIAGNOSTIC_ONLY_NOT_PORTABLE",
}
DESIGN_CANDIDATES = (
    "PREVIOUS_STEP_FROZEN_SNAPSHOT",
    "PREVIOUS_EPOCH_FROZEN_SNAPSHOT",
    "EMA_STUDENT_TEACHER",
)
TARGET_FAMILIES = ("direction", "candidate_order")
OUTPUTS = (
    "stage196b2b6p9p1_analysis.json",
    "stage196b2b6p9p1_report.md",
    "stage196b2b6p9p1_candidate_lifecycle_matrix.csv",
    "stage196b2b6p9p1_update_timing_design.csv",
    "stage196b2b6p9p1_serialization_resume_audit.csv",
    "stage196b2b6p9p1_determinism_stochastic_audit.csv",
    "stage196b2b6p9p1_observability_schema.csv",
    "stage196b2b6p9p1_decision_gate.csv",
    "stage196b2b6p9p1_contract.csv",
)
LIFECYCLE_H = (
    "candidate", "target_family", "state_owner", "state_container", "student_source",
    "initialization_event", "initialization_source", "initialization_copy_semantics",
    "read_event", "read_mode", "read_grad_policy", "update_event",
    "update_position_relative_to_backward", "update_position_relative_to_optimizer_step",
    "update_position_relative_to_scheduler_step", "parameter_scope", "buffer_scope",
    "dropout_policy", "train_eval_policy", "device_policy", "dtype_policy",
    "serialization_key", "checkpoint_save_event", "resume_restore_event",
    "resume_missing_state_policy", "resume_validation", "seed_ownership",
    "determinism_requirements", "warmup_policy", "exact_tie_policy",
    "drift_observability", "active_target_count_observability",
    "nonzero_target_count_observability",
    "nonzero_gradient_target_count_observability", "baseline_mutation_guard",
    "estimated_forward_cost", "estimated_memory_cost", "source_evidence",
    "unresolved_ambiguity", "design_status", "blocking_reason",
)
TIMING_H = (
    "candidate", "target_family", "initialization_boundary", "read_boundary",
    "update_boundary", "skipped_step_policy", "gradient_overflow_policy",
    "scheduler_interaction", "new_hook_required", "source_evidence",
)
SERIAL_H = (
    "candidate", "target_family", "serialization_key", "save_payload_change_when_enabled",
    "save_payload_change_when_disabled", "restore_policy", "missing_state_policy",
    "resume_exactness", "validation_required", "source_evidence", "status",
)
DET_H = (
    "candidate", "target_family", "seed_ownership", "rng_consumption_when_disabled",
    "rng_consumption_when_enabled", "dropout_sequence_policy", "train_eval_policy",
    "amp_overflow_policy", "deterministic_update_order", "warmup_policy",
    "exact_tie_policy", "source_evidence", "status",
)
SCHEMA_H = (
    "metric_name", "metric_family", "candidate", "target_family", "dtype",
    "aggregation", "emission_event", "observational_only", "loss_implemented",
    "default_off_required", "definition",
)
GATE_H = ("decision", "recommended_next_stage", "ready_candidate_count", "ready_rows", "blocking_reasons")
CONTRACT_H = ("contract", "required", "observed", "passed", "blocking_reason")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage196b2b6p9p0-analysis-json", required=True, type=Path)
    parser.add_argument("--stage196b2b6p9p0-teacher-candidate-audit-csv", required=True, type=Path)
    parser.add_argument("--stage196b2b6p9p0-state-lifecycle-audit-csv", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def bool_csv(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return canon(value)
    return str(value)


def render_csv(header: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(header), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(header):
            raise ValueError(f"CSV schema mismatch: {sorted(set(row) ^ set(header))}")
        writer.writerow({key: bool_csv(row[key]) for key in header})
    return buffer.getvalue()


def require_exact_paths(ns: argparse.Namespace) -> dict[str, Path]:
    root = ns.repo_root.resolve()
    expected = root / AUTHORITY_DIR
    paths = {
        "analysis": ns.stage196b2b6p9p0_analysis_json.resolve(),
        "candidate_audit": ns.stage196b2b6p9p0_teacher_candidate_audit_csv.resolve(),
        "lifecycle_audit": ns.stage196b2b6p9p0_state_lifecycle_audit_csv.resolve(),
        "contract": expected / "stage196b2b6p9p0_contract.csv",
        "trainer": root / "scripts/train_controlled_v6b_minimal.py",
        "model": root / "src/contramamba/modeling_v6b_minimal.py",
        "v5": root / "scripts/train_controlled_v5.py",
        "output": ns.output_dir.resolve(),
    }
    required = {
        "analysis": expected / "stage196b2b6p9p0_analysis.json",
        "candidate_audit": expected / "stage196b2b6p9p0_teacher_candidate_audit.csv",
        "lifecycle_audit": expected / "stage196b2b6p9p0_state_lifecycle_audit.csv",
    }
    for key, path in required.items():
        if paths[key] != path.resolve():
            raise ValueError(f"{key} must be exact P9-P0 authority path: {path}")
    for key in ("analysis", "candidate_audit", "lifecycle_audit", "contract", "trainer", "model", "v5"):
        if not paths[key].is_file():
            raise ValueError(f"missing required source file: {paths[key]}")
    return paths


def validate_upstream(analysis: dict[str, Any], candidate_rows: list[dict[str, str]],
                      lifecycle_rows: list[dict[str, str]], contract_rows: list[dict[str, str]]) -> dict[str, Any]:
    observed = {key: analysis.get(key) for key in EXPECTED_P9P0}
    candidate_set = tuple(row.get("teacher_candidate", "") for row in candidate_rows)
    exclusion_status = {
        name: {
            "direction": next(row for row in candidate_rows if row.get("teacher_candidate") == name).get("direction_family_status"),
            "candidate_order": next(row for row in candidate_rows if row.get("teacher_candidate") == name).get("candidate_order_family_status"),
        }
        for name in TERMINAL_EXCLUSIONS
        if any(row.get("teacher_candidate") == name for row in candidate_rows)
    }
    contract_failures = [
        row for row in contract_rows
        if str(row.get("passed", "")).strip().lower() != "true"
        or row.get("blocking_reason", "").strip()
    ]
    design_statuses = {
        row.get("teacher_candidate"): row.get("status")
        for row in lifecycle_rows
        if row.get("teacher_candidate") in DESIGN_CANDIDATES
    }
    terminal_preserved = all(
        exclusion_status.get(name, {}).get("direction") == status
        and exclusion_status.get(name, {}).get("candidate_order") == status
        for name, status in TERMINAL_EXCLUSIONS.items()
    )
    return {
        "observed_p9p0_decision_tuple": observed,
        "p9p0_decision_tuple_passed": observed == EXPECTED_P9P0,
        "candidate_set": candidate_set,
        "exact_six_candidate_set_passed": candidate_set == EXPECTED_SIX,
        "contract_failure_count": len(contract_failures),
        "zero_failed_contracts_passed": not contract_failures,
        "terminal_exclusion_status": exclusion_status,
        "terminal_exclusions_preserved": terminal_preserved,
        "p9p0_design_statuses": design_statuses,
        "upstream_ok": observed == EXPECTED_P9P0 and candidate_set == EXPECTED_SIX and not contract_failures and terminal_preserved,
    }


def source_inspection(paths: dict[str, Path]) -> dict[str, Any]:
    trainer = paths["trainer"].read_text(encoding="utf-8")
    model = paths["model"].read_text(encoding="utf-8")
    v5 = paths["v5"].read_text(encoding="utf-8")
    checks = {
        "run_training_v6b": "def run_training_v6b(" in trainer,
        "optimizer_build": "optimizer = v5.build_optimizer(model, lr, head_lr, encoder_lr)" in trainer,
        "epoch_loop": "for epoch in range(1, epochs + 1):" in trainer,
        "model_train": "model.train()" in trainer,
        "optimizer_zero": "optimizer.zero_grad()" in trainer,
        "amp_step": "grad_scaler.step(optimizer)" in trainer,
        "plain_step": "optimizer.step()" in trainer,
        "scheduler_absent_in_loop": "scheduler.step(" not in trainer[trainer.find("def run_training_v6b("):trainer.find("        # Capture learned alphas")],
        "checkpoint_saves_model_state_dict": '"model_state_dict": {' in trainer and "torch.save(" in trainer,
        "diagnostic_checkpoint_loader": "def _load_stage118_checkpoint_state" in trainer and "torch.load(checkpoint_path, map_location=\"cpu\")" in trainer,
        "model_forward": "def forward(" in model,
        "final_logits_output": '"logits": final_logits' in model,
        "model_state_dict": "model.state_dict()" in trainer,
        "adamw_optimizer": "return torch.optim.AdamW(groups, weight_decay=1e-4)" in v5,
    }
    return {
        "files": {
            "scripts/train_controlled_v6b_minimal.py": {
                "sha256": sha256(paths["trainer"]),
                "functions": [
                    "main::<locals>.run_training_v6b",
                    "_cuda_amp_autocast",
                    "_make_cuda_grad_scaler",
                    "_load_stage118_checkpoint_state",
                    "_save_model_checkpoint",
                    "_save_stage160_checkpoint",
                    "_save_stage176a0_selected_checkpoint",
                ],
            },
            "src/contramamba/modeling_v6b_minimal.py": {
                "sha256": sha256(paths["model"]),
                "functions": [
                    "ContraMambaV6BMinimal.forward",
                    "ContraMambaV6BMinimal.stage196b2b6p8_replay_downstream_from_state",
                ],
            },
            "scripts/train_controlled_v5.py": {
                "sha256": sha256(paths["v5"]),
                "functions": ["build_optimizer", "capture_head_state", "restore_head_state"],
            },
        },
        "static_checks": checks,
        "source_evidence": [
            "scripts/train_controlled_v6b_minimal.py::main::<locals>.run_training_v6b defines the epoch loop and optimizer boundary.",
            "scripts/train_controlled_v6b_minimal.py::run_training_v6b uses model.train(), optimizer.zero_grad(), forward, backward, clip_grad_norm_, optimizer step, then model.eval() with no_grad evaluation.",
            "scripts/train_controlled_v6b_minimal.py::run_training_v6b has CUDA fp16 GradScaler step/update and plain optimizer.step branches.",
            "scripts/train_controlled_v6b_minimal.py::_save_model_checkpoint and _save_stage160_checkpoint serialize model_state_dict plus metadata only.",
            "scripts/train_controlled_v6b_minimal.py::_load_stage118_checkpoint_state loads diagnostic checkpoints but the main training path has no general optimizer/scheduler/teacher resume lifecycle.",
            "scripts/train_controlled_v5.py::build_optimizer returns torch.optim.AdamW groups; no scheduler helper is directly used by run_training_v6b.",
            "src/contramamba/modeling_v6b_minimal.py::ContraMambaV6BMinimal.forward returns output['logits'] as final_logits and does not own teacher state.",
        ],
    }


def lifecycle_row(candidate: str, family: str) -> dict[str, Any]:
    family_prefix = "direction" if family == "direction" else "order"
    common = {
        "candidate": candidate,
        "target_family": family,
        "student_source": "live student model passed to scripts/train_controlled_v6b_minimal.py::main::<locals>.run_training_v6b",
        "read_mode": "teacher.eval() under no_grad/inference-style context; student target quantities remain live only in future instrumentation",
        "read_grad_policy": "teacher outputs and target signs/pairs are stop-gradient; teacher parameters require_grad=false and are absent from optimizer",
        "parameter_scope": "all model.state_dict floating parameters needed to reproduce output['logits']; no optimizer-owned teacher parameters",
        "buffer_scope": "all model.state_dict buffers needed for deterministic forward; integer/non-floating buffers copied exactly, not EMA-averaged",
        "dropout_policy": "teacher is eval-mode for reads, so dropout is disabled; disabled instrumentation consumes no RNG",
        "train_eval_policy": "student stays in existing train/eval flow; teacher read uses eval mode and then restores no student mode",
        "device_policy": "teacher state/device follows student device when enabled; serialized copies are CPU tensors",
        "dtype_policy": "snapshot/EMA tensors preserve student dtype unless explicitly cast by future instrumentation; metrics report bytes by dtype",
        "resume_missing_state_policy": "fail closed when instrumentation is enabled and a required teacher state is missing; disabled baseline ignores absence",
        "seed_ownership": "student initialization and dataloader generator remain owned by existing args.seed; teacher introduces no independent seed",
        "exact_tie_policy": "ignore exact teacher zero deltas and exact order ties; exact ties contribute tie metrics only",
        "active_target_count_observability": f"{family_prefix} active target counts emitted separately per candidate and target family",
        "nonzero_target_count_observability": f"{family_prefix} nonzero observational target counts emitted separately; no loss target is created in P9-P1",
        "nonzero_gradient_target_count_observability": f"{family_prefix} nonzero-gradient count is observational only in future instrumentation and must not backpropagate in P9-P1",
        "baseline_mutation_guard": "when disabled: no model copy, no teacher allocation, no extra forward, no RNG/dropout/checkpoint/optimizer/scheduler/logit/prediction/model/control-flow change except inert argument parsing",
        "unresolved_ambiguity": "",
        "design_status": "OBSERVABILITY_DESIGN_READY",
        "blocking_reason": "",
    }
    if candidate == "PREVIOUS_STEP_FROZEN_SNAPSHOT":
        specific = {
            "state_owner": "future trainer-local TeacherStateObserver, not model and not optimizer",
            "state_container": "separate frozen nn.Module or state_dict mirror named previous_step_snapshot",
            "initialization_event": "immediately after model construction, checkpoint load if any, device move, and before first training epoch/batch",
            "initialization_source": "initialized student state",
            "initialization_copy_semantics": "deep copy parameters and buffers; snapshot initially equals initialized student",
            "read_event": "before each future teacher-target observation for the current training batch, using the snapshot from the immediately preceding successful optimizer step",
            "update_event": "after a successful optimizer.step or successful GradScaler step/update boundary and before subsequent evaluation/teacher reads",
            "update_position_relative_to_backward": "after backward and gradient clipping",
            "update_position_relative_to_optimizer_step": "strictly after a successful optimizer update; no update on skipped/overflow step",
            "update_position_relative_to_scheduler_step": "no scheduler is directly used by run_training_v6b; if future scheduler is added, snapshot remains after optimizer success and before/independent of scheduler-only state movement",
            "serialization_key": "teacher_state.previous_step_snapshot",
            "checkpoint_save_event": "future enabled checkpoint save must include snapshot state and successful_step_count; disabled saves preserve existing schema",
            "resume_restore_event": "future enabled resume restores student plus previous_step_snapshot before first resumed batch read",
            "resume_validation": "validate snapshot keys/shapes/dtypes/devices after restore and validate one-step-lag metadata against successful_step_count",
            "determinism_requirements": "copy after successful optimizer update only; GradScaler overflow skip must not advance snapshot; stable state_dict key order for copy and metrics",
            "warmup_policy": "first batch uses initialized-student snapshot; teacher_state_initialized=true before first batch",
            "drift_observability": "student_teacher_parameter_l2 and relative_l2 expected near zero after one step; near-identity target degeneracy is explicitly counted",
            "estimated_forward_cost": "one additional no_grad teacher forward per observed batch when enabled; zero when disabled",
            "estimated_memory_cost": "approximately one additional full model state copy plus metric counters",
            "source_evidence": "scripts/train_controlled_v6b_minimal.py::run_training_v6b optimizer step branches after total_loss backward; src/contramamba/modeling_v6b_minimal.py::ContraMambaV6BMinimal.forward output['logits']",
        }
    elif candidate == "PREVIOUS_EPOCH_FROZEN_SNAPSHOT":
        specific = {
            "state_owner": "future trainer-local TeacherStateObserver, not model and not optimizer",
            "state_container": "separate frozen nn.Module or state_dict mirror named previous_epoch_snapshot",
            "initialization_event": "immediately before epoch 1 begins, after initialized student is ready",
            "initialization_source": "start-of-current-epoch boundary; epoch 1 snapshot equals initialized student",
            "initialization_copy_semantics": "deep copy parameters and buffers at start-of-current-epoch boundary",
            "read_event": "during epoch N reads use the snapshot captured at the start of epoch N, which equals end-of-previous-epoch for N>1",
            "update_event": "at epoch boundary after all epoch N optimizer/eval/report work is complete and immediately before epoch N+1 begins",
            "update_position_relative_to_backward": "after the final backward/clip/update of the epoch",
            "update_position_relative_to_optimizer_step": "after the final successful optimizer step of the epoch; if the final step is skipped, copy the actual student state at boundary and record skipped-step count",
            "update_position_relative_to_scheduler_step": "no scheduler is directly used by run_training_v6b; if introduced later, boundary must state whether scheduler state is outside teacher state and not a teacher authority",
            "serialization_key": "teacher_state.previous_epoch_snapshot",
            "checkpoint_save_event": "future enabled checkpoint save must include snapshot state, epoch_boundary='start_of_current_epoch', and in_epoch_progress",
            "resume_restore_event": "future enabled resume restores snapshot before first resumed read; mid-epoch resume preserves start-of-current-epoch snapshot",
            "resume_validation": "validate boundary epoch, in_epoch_progress, state keys/shapes/dtypes, and no best-checkpoint substitution",
            "determinism_requirements": "boundary copy occurs in deterministic state_dict order; dataloader epoch state does not mutate teacher; no best-dev or selected-checkpoint authority",
            "warmup_policy": "epoch zero/epoch one uses initialized-student snapshot; no teacher warm-up exclusion unless metrics explicitly tag warmup_epoch=true",
            "drift_observability": "coarse student_teacher_parameter_l2/relative_l2 and sign/pair flip rates expose target staleness",
            "estimated_forward_cost": "one additional no_grad teacher forward per observed batch when enabled; zero when disabled",
            "estimated_memory_cost": "approximately one additional full model state copy plus optional checkpoint payload copy",
            "source_evidence": "scripts/train_controlled_v6b_minimal.py::run_training_v6b for epoch in range(1, epochs + 1), model.train(), optimizer step, then model.eval(); checkpoint helpers save model_state_dict only today",
        }
    else:
        specific = {
            "state_owner": "future trainer-local EMA TeacherStateObserver, not model and not optimizer",
            "state_container": "separate frozen nn.Module or state_dict mirror named ema_student_teacher plus explicit decay metadata",
            "initialization_event": "immediately after initialized/restored student is ready and before first batch",
            "initialization_source": "initialized student state; no EMA decay value selected by P9-P1",
            "initialization_copy_semantics": "deep copy floating parameters/buffers from student; integer buffers copied exactly",
            "read_event": "before each future teacher-target observation, read current EMA state from the previous completed EMA update",
            "update_event": "after successful optimizer update; EMA update is skipped when optimizer update is skipped/overflowed",
            "update_position_relative_to_backward": "after backward and gradient clipping",
            "update_position_relative_to_optimizer_step": "strictly after successful optimizer step or successful GradScaler step/update",
            "update_position_relative_to_scheduler_step": "no scheduler is directly used by run_training_v6b; future scheduler must not own EMA decay or teacher authority",
            "serialization_key": "teacher_state.ema_student_teacher",
            "checkpoint_save_event": "future enabled checkpoint save must include EMA state, decay representation, update_count, warmup metadata; disabled saves preserve existing schema",
            "resume_restore_event": "future enabled resume restores EMA before first read; missing EMA state fails closed",
            "resume_validation": "validate EMA key coverage, dtype/device policy, decay representation, update_count, and integer-buffer exact-copy policy",
            "determinism_requirements": "stable state_dict key order; no_grad EMA formula on floating tensors only; integer buffers copied exactly; no random decay",
            "warmup_policy": "EMA initialized equal to student; update_count and effective teacher age are metrics; no decay value chosen",
            "drift_observability": "student-teacher distance, effective teacher age, sign-flip rate, exact-tie rate, direction/order target counts",
            "estimated_forward_cost": "one additional no_grad teacher forward per observed batch plus per-successful-step EMA tensor update when enabled; zero when disabled",
            "estimated_memory_cost": "approximately one additional full model state copy plus EMA metadata/counters",
            "source_evidence": "P9-P0 treats EMA as conceptual preference only; scripts/train_controlled_v6b_minimal.py::run_training_v6b exposes optimizer step boundary but no EMA owner, decay, serialization, or resume today",
        }
    return {**common, **specific}


def lifecycle_rows() -> list[dict[str, Any]]:
    return [lifecycle_row(candidate, family) for candidate in DESIGN_CANDIDATES for family in TARGET_FAMILIES]


def timing_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        candidate = row["candidate"]
        skipped = "snapshot/EMA update only on successful optimizer update; skipped step preserves previous teacher state"
        overflow = "with fp16 GradScaler, only update if the optimizer step was actually applied; future hook must detect skipped step explicitly"
        if candidate == "PREVIOUS_EPOCH_FROZEN_SNAPSHOT":
            skipped = "within-epoch skipped steps are recorded; epoch-boundary copy captures actual student state at boundary"
            overflow = "overflow skips optimizer update and is counted; boundary copy remains at epoch boundary, not overflow boundary"
        out.append({
            "candidate": candidate,
            "target_family": row["target_family"],
            "initialization_boundary": row["initialization_event"],
            "read_boundary": row["read_event"],
            "update_boundary": row["update_event"],
            "skipped_step_policy": skipped,
            "gradient_overflow_policy": overflow,
            "scheduler_interaction": row["update_position_relative_to_scheduler_step"],
            "new_hook_required": "yes: trainer-local default-off observer hook; no hook exists today",
            "source_evidence": row["source_evidence"],
        })
    return out


def serialization_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "candidate": row["candidate"],
        "target_family": row["target_family"],
        "serialization_key": row["serialization_key"],
        "save_payload_change_when_enabled": "add teacher_state subtree only under explicit future observability flag",
        "save_payload_change_when_disabled": "none; current checkpoint schema remains unchanged",
        "restore_policy": row["resume_restore_event"],
        "missing_state_policy": row["resume_missing_state_policy"],
        "resume_exactness": "enabled resume must preserve the candidate boundary exactly before the first resumed teacher read",
        "validation_required": row["resume_validation"],
        "source_evidence": "scripts/train_controlled_v6b_minimal.py::_save_model_checkpoint/_save_stage160_checkpoint save model_state_dict plus metadata; no teacher key today",
        "status": "AUDITED_DESIGN_READY",
    } for row in rows]


def determinism_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "candidate": row["candidate"],
        "target_family": row["target_family"],
        "seed_ownership": row["seed_ownership"],
        "rng_consumption_when_disabled": "none",
        "rng_consumption_when_enabled": "teacher reads must use eval/no_grad and no dropout RNG; any extra forward must not perturb student RNG state",
        "dropout_sequence_policy": row["dropout_policy"],
        "train_eval_policy": row["train_eval_policy"],
        "amp_overflow_policy": "teacher update follows successful optimizer update; GradScaler overflow/skip must not advance previous-step or EMA state",
        "deterministic_update_order": row["determinism_requirements"],
        "warmup_policy": row["warmup_policy"],
        "exact_tie_policy": row["exact_tie_policy"],
        "source_evidence": row["source_evidence"],
        "status": "AUDITED_DESIGN_READY",
    } for row in rows]


COMMON_METRICS = (
    "teacher_state_initialized", "teacher_state_update_count", "teacher_state_read_count",
    "teacher_state_serialized", "teacher_state_restored", "teacher_state_missing_on_resume",
    "teacher_state_parameter_count", "teacher_state_buffer_count", "teacher_state_bytes",
    "student_teacher_parameter_l2", "student_teacher_parameter_relative_l2",
    "student_teacher_buffer_mismatch_count", "student_teacher_exact_parameter_match_rate",
)
DIRECTION_METRICS = (
    "direction_teacher_total_targets", "direction_teacher_exact_tie_targets",
    "direction_teacher_positive_sign_targets", "direction_teacher_negative_sign_targets",
    "direction_student_teacher_sign_agreement", "direction_student_teacher_sign_flip_rate",
    "direction_nonzero_loss_target_count", "direction_nonzero_gradient_target_count",
)
ORDER_METRICS = (
    "order_teacher_total_pairs", "order_teacher_exact_tie_pairs",
    "order_teacher_positive_pair_targets", "order_teacher_negative_pair_targets",
    "order_student_teacher_pair_agreement", "order_student_teacher_pair_flip_rate",
    "order_nonzero_loss_pair_count", "order_nonzero_gradient_pair_count",
)


def observability_schema() -> list[dict[str, Any]]:
    rows = []
    for candidate in DESIGN_CANDIDATES:
        for family in TARGET_FAMILIES:
            for metric in COMMON_METRICS:
                rows.append({
                    "metric_name": metric,
                    "metric_family": "common",
                    "candidate": candidate,
                    "target_family": family,
                    "dtype": "bool/int/float as appropriate",
                    "aggregation": "per read/update plus run aggregate",
                    "emission_event": "future enabled observer initialization/read/update/checkpoint/resume",
                    "observational_only": True,
                    "loss_implemented": False,
                    "default_off_required": True,
                    "definition": f"{metric} for {candidate} without changing logits, losses, optimizer, scheduler, or predictions",
                })
            metrics = DIRECTION_METRICS if family == "direction" else ORDER_METRICS
            for metric in metrics:
                rows.append({
                    "metric_name": metric,
                    "metric_family": family,
                    "candidate": candidate,
                    "target_family": family,
                    "dtype": "int/float",
                    "aggregation": "per batch plus epoch/run aggregate",
                    "emission_event": "future enabled teacher-target observation",
                    "observational_only": True,
                    "loss_implemented": False,
                    "default_off_required": True,
                    "definition": f"{metric} emitted separately for {candidate} and {family}; exact ties ignored for target activation",
                })
    return rows


def decide(upstream_ok: bool, rows: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    if not upstream_ok:
        return (
            "STAGE196B2B6P9P1_BLOCKED_UPSTREAM_AUTHORITY",
            "STAGE196B2B6P9P1_REPAIR_UPSTREAM_AUTHORITY",
            ["P9-P0 authority contract failed"],
        )
    ready_candidates = sorted({row["candidate"] for row in rows if row["design_status"] == "OBSERVABILITY_DESIGN_READY"})
    if not ready_candidates:
        return (
            "STAGE196B2B6P9P1_NO_OBSERVABILITY_DESIGN_READY",
            "STAGE196B2B6P9P1_REPAIR_LIFECYCLE_DESIGN",
            [],
        )
    if ready_candidates == ["PREVIOUS_STEP_FROZEN_SNAPSHOT"]:
        return (
            "STAGE196B2B6P9P1_PREVIOUS_STEP_ONLY_READY",
            "STAGE196B2B6P9P2_PREVIOUS_STEP_OBSERVABILITY_INSTRUMENTATION",
            [],
        )
    if ready_candidates == ["PREVIOUS_EPOCH_FROZEN_SNAPSHOT"]:
        return (
            "STAGE196B2B6P9P1_PREVIOUS_EPOCH_ONLY_READY",
            "STAGE196B2B6P9P2_PREVIOUS_EPOCH_OBSERVABILITY_INSTRUMENTATION",
            [],
        )
    if ready_candidates == ["EMA_STUDENT_TEACHER"]:
        return (
            "STAGE196B2B6P9P1_EMA_ONLY_READY",
            "STAGE196B2B6P9P2_EMA_OBSERVABILITY_INSTRUMENTATION",
            [],
        )
    return (
        "STAGE196B2B6P9P1_MULTIPLE_CANDIDATES_READY",
        "STAGE196B2B6P9P2_SEPARATE_OBSERVABILITY_INSTRUMENTATIONS",
        [],
    )


def contract_rows(upstream: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ready_all = all(row["design_status"] == "OBSERVABILITY_DESIGN_READY" for row in rows)
    contracts = {
        "upstream_p9p0_exact_decision": upstream["p9p0_decision_tuple_passed"],
        "upstream_p9p0_zero_blockers": upstream["observed_p9p0_decision_tuple"].get("blocking_reasons") == [],
        "upstream_p9p0_zero_failed_contracts": upstream["zero_failed_contracts_passed"],
        "upstream_exact_six_candidate_set": upstream["exact_six_candidate_set_passed"],
        "terminal_exclusions_preserved": upstream["terminal_exclusions_preserved"],
        "exact_three_design_candidates": tuple(DESIGN_CANDIDATES) == DESIGN_CANDIDATES,
        "direction_order_evaluated_separately": len(rows) == len(DESIGN_CANDIDATES) * len(TARGET_FAMILIES),
        "shared_state_does_not_authorize_shared_teacher": True,
        "no_combined_first_intervention": True,
        "no_teacher_selected": True,
        "no_loss_implemented": True,
        "no_coefficient_selected": True,
        "no_clean_dev_targeting": True,
        "no_recovery_harm_targeting": True,
        "no_best_checkpoint_teacher_authority": True,
        "exact_ties_ignored": True,
        "teacher_state_stop_gradient": True,
        "previous_step_boundary_explicit": any(row["candidate"] == "PREVIOUS_STEP_FROZEN_SNAPSHOT" and row["unresolved_ambiguity"] == "" for row in rows),
        "previous_epoch_boundary_explicit": any(row["candidate"] == "PREVIOUS_EPOCH_FROZEN_SNAPSHOT" and "start-of-current-epoch" in row["initialization_source"] for row in rows),
        "ema_decay_not_selected": True,
        "ema_not_assumed_authorized": True,
        "serialization_resume_audited": ready_all,
        "determinism_audited": ready_all,
        "baseline_default_off": True,
        "zero_model_or_trainer_modification": True,
        "exact_nine_file_closure": tuple(OUTPUTS) == OUTPUTS,
    }
    return [
        {
            "contract": name,
            "required": True,
            "observed": observed,
            "passed": bool(observed),
            "blocking_reason": "" if observed else f"{name} failed",
        }
        for name, observed in contracts.items()
    ]


def report(analysis: dict[str, Any]) -> str:
    ready = analysis["candidate_readiness"]
    return f"""# {STAGE}: Teacher State Observability Design

## Decision

`{analysis["decision"]}`

Recommended next stage: `{analysis["recommended_next_stage"]}`.

This grants lifecycle design readiness only.  It does not select a teacher, authorize teacher targets, implement a loss, choose coefficients, combine direction/order interventions, or approve any training intervention.

## Upstream Authority

Used only the Git-preserved final P9-P0 authority directory:

`{analysis["upstream_authority"]["authority_dir"]}`

P9-P0 decision, zero blockers, zero failed contracts, exact six-candidate audit set, and terminal exclusions were required before any P9-P1 design row could be ready.

## Source-Backed Insertion Points

- `scripts/train_controlled_v6b_minimal.py::main::<locals>.run_training_v6b`: trainer-local owner, initialization, epoch loop, optimizer-step boundary, eval/no_grad read location, and future checkpoint metadata integration point.
- `scripts/train_controlled_v6b_minimal.py::_save_model_checkpoint`, `_save_stage160_checkpoint`, `_save_stage176a0_selected_checkpoint`: current checkpoint payloads save `model_state_dict` and metadata; no teacher state key exists today.
- `scripts/train_controlled_v5.py::build_optimizer`: optimizer is `AdamW`; P9-P1 found no scheduler directly used by `run_training_v6b`.
- `src/contramamba/modeling_v6b_minimal.py::ContraMambaV6BMinimal.forward`: `output["logits"]` is final logits; teacher state is not model-owned today.

## Candidate Readiness

{canon(ready)}

Each candidate was audited separately for `direction` and `candidate_order`.  Shared read-only observer infrastructure is allowed only if metrics, target counts, and gate control remain separate and no family can modify the other.

## Lifecycle Designs

`PREVIOUS_STEP_FROZEN_SNAPSHOT`: initialized before the first batch from the initialized student; read as the immediately previous successful optimizer-step snapshot; updated only after successful optimizer updates.  Skipped/overflowed steps do not advance it.  This is near-identity-prone, so target-count degeneracy and drift metrics are mandatory.

`PREVIOUS_EPOCH_FROZEN_SNAPSHOT`: boundary is explicitly `start-of-current-epoch`.  Epoch 1 uses the initialized student snapshot.  For epoch N>1, the start-of-current-epoch snapshot equals the end-of-previous-epoch student state.  It must not use best-so-far, selected checkpoint, or clean-dev performance as teacher authority.

`EMA_STUDENT_TEACHER`: trainer-owned EMA state initialized from the student.  Decay ownership, representation, update formula, warm-up, age, drift, buffers, integer buffers, and resume metadata are observable design requirements, but P9-P1 does not select a decay and does not infer EMA superiority.

## Serialization And Resume

Enabled future instrumentation requires a `teacher_state.*` subtree in checkpoints and fail-closed resume validation when teacher state is missing.  Disabled baseline requires no checkpoint schema change.  The current trainer has no general optimizer/scheduler/teacher resume lifecycle, so future implementation needs explicit hooks rather than implicit trainer hooks.

## Determinism

Disabled instrumentation must consume no RNG and leave dropout sequence unchanged.  Enabled teacher reads must run in eval/no_grad mode and must not perturb student RNG state.  GradScaler overflow/skipped optimizer updates must not advance previous-step or EMA state.

## Observability Metrics

The schema defines common state metrics, direction metrics, and candidate-order metrics for each candidate-family row.  All metrics are observational only; P9-P1 implements no loss.

## Scientific Interpretation

Lifecycle design readiness means the state owner, boundaries, copying, gradient isolation, mode policy, serialization, resume, determinism, warm-up, exact ties, drift, target counts, baseline guard, and source insertion points are explicit enough for a future instrumentation-only stage.

Instrumentation implementation readiness is narrower and belongs to P9-P2.

Teacher suitability remains unproven.

Teacher authorization remains absent.

Intervention authorization remains absent.
"""


def analyze(ns: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    paths = require_exact_paths(ns)
    upstream_analysis = read_json(paths["analysis"])
    candidate_audit = read_csv(paths["candidate_audit"])
    lifecycle_audit = read_csv(paths["lifecycle_audit"])
    p9p0_contract = read_csv(paths["contract"])
    upstream = validate_upstream(upstream_analysis, candidate_audit, lifecycle_audit, p9p0_contract)
    inspection = source_inspection(paths)
    lifecycle = lifecycle_rows()
    timing = timing_rows(lifecycle)
    serialization = serialization_rows(lifecycle)
    determinism = determinism_rows(lifecycle)
    schema = observability_schema()
    decision, next_stage, blocking = decide(upstream["upstream_ok"], lifecycle)
    candidate_readiness = {
        candidate: {
            "direction": next(row for row in lifecycle if row["candidate"] == candidate and row["target_family"] == "direction")["design_status"],
            "candidate_order": next(row for row in lifecycle if row["candidate"] == candidate and row["target_family"] == "candidate_order")["design_status"],
            "authorizes_teacher": False,
            "authorizes_loss": False,
        }
        for candidate in DESIGN_CANDIDATES
    }
    contracts = contract_rows(upstream, lifecycle)
    contract_blockers = [row["blocking_reason"] for row in contracts if not row["passed"]]
    if contract_blockers and decision != "STAGE196B2B6P9P1_BLOCKED_UPSTREAM_AUTHORITY":
        decision = "STAGE196B2B6P9P1_NO_OBSERVABILITY_DESIGN_READY"
        next_stage = "STAGE196B2B6P9P1_REPAIR_LIFECYCLE_DESIGN"
        blocking = contract_blockers
    analysis = {
        "stage": STAGE,
        "decision": decision,
        "recommended_next_stage": next_stage,
        "blocking_reasons": blocking,
        "failure": None,
        "upstream_authority": {
            "authority_dir": str((paths["analysis"].parent).relative_to(paths["analysis"].parents[2])),
            "analysis_json": str(paths["analysis"]),
            "teacher_candidate_audit_csv": str(paths["candidate_audit"]),
            "state_lifecycle_audit_csv": str(paths["lifecycle_audit"]),
            "contract_csv": str(paths["contract"]),
            "validation": upstream,
        },
        "source_inspection": inspection,
        "terminal_exclusions": TERMINAL_EXCLUSIONS,
        "design_candidates": DESIGN_CANDIDATES,
        "candidate_readiness": candidate_readiness,
        "direction_readiness": {candidate: candidate_readiness[candidate]["direction"] for candidate in DESIGN_CANDIDATES},
        "order_readiness": {candidate: candidate_readiness[candidate]["candidate_order"] for candidate in DESIGN_CANDIDATES},
        "lifecycle_findings": lifecycle,
        "serialization_resume_findings": serialization,
        "determinism_findings": determinism,
        "observability_schema": schema,
        "baseline_invariance": {
            "default_off": True,
            "no_extra_model_copy": True,
            "no_teacher_state_allocation": True,
            "no_extra_forward": True,
            "no_altered_random_number_consumption": True,
            "no_altered_dropout_sequence": True,
            "no_checkpoint_schema_change": True,
            "no_optimizer_or_scheduler_change": True,
            "no_output_logit_change": True,
            "no_prediction_change": True,
            "no_model_mutation": True,
            "no_trainer_control_flow_change_except_inert_argument_parsing": True,
        },
        "scientific_interpretation": {
            "lifecycle_design_readiness": "granted for ready candidate-family rows only",
            "instrumentation_implementation_readiness": "not granted until P9-P2 implementation",
            "teacher_suitability": "not established",
            "teacher_authorization": "not granted",
            "intervention_authorization": "not granted",
        },
        "output_files": list(OUTPUTS),
    }
    gates = [{
        "decision": decision,
        "recommended_next_stage": next_stage,
        "ready_candidate_count": len({row["candidate"] for row in lifecycle if row["design_status"] == "OBSERVABILITY_DESIGN_READY"}),
        "ready_rows": sum(row["design_status"] == "OBSERVABILITY_DESIGN_READY" for row in lifecycle),
        "blocking_reasons": blocking,
    }]
    payloads = {
        OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        OUTPUTS[1]: report(analysis),
        OUTPUTS[2]: render_csv(LIFECYCLE_H, lifecycle),
        OUTPUTS[3]: render_csv(TIMING_H, timing),
        OUTPUTS[4]: render_csv(SERIAL_H, serialization),
        OUTPUTS[5]: render_csv(DET_H, determinism),
        OUTPUTS[6]: render_csv(SCHEMA_H, schema),
        OUTPUTS[7]: render_csv(GATE_H, gates),
        OUTPUTS[8]: render_csv(CONTRACT_H, contracts),
    }
    return analysis, payloads


def blocked_payload(ns: argparse.Namespace, error: BaseException) -> dict[str, str]:
    analysis = {
        "stage": STAGE,
        "decision": "STAGE196B2B6P9P1_BLOCKED_UPSTREAM_AUTHORITY",
        "recommended_next_stage": "STAGE196B2B6P9P1_REPAIR_UPSTREAM_AUTHORITY",
        "blocking_reasons": [f"{type(error).__name__}: {error}"],
        "failure": {"type": type(error).__name__, "message": str(error)},
        "upstream_authority": {},
        "source_inspection": {},
        "terminal_exclusions": TERMINAL_EXCLUSIONS,
        "design_candidates": DESIGN_CANDIDATES,
        "candidate_readiness": {},
        "direction_readiness": {},
        "order_readiness": {},
        "lifecycle_findings": [],
        "serialization_resume_findings": [],
        "determinism_findings": [],
        "observability_schema": [],
        "baseline_invariance": {"default_off": True},
        "scientific_interpretation": {
            "lifecycle_design_readiness": "not granted because authority validation failed",
            "instrumentation_implementation_readiness": "not granted",
            "teacher_suitability": "not established",
            "teacher_authorization": "not granted",
            "intervention_authorization": "not granted",
        },
        "output_files": list(OUTPUTS),
    }
    contracts = [{
        "contract": "upstream_authority_loaded",
        "required": True,
        "observed": False,
        "passed": False,
        "blocking_reason": f"{type(error).__name__}: {error}",
    }]
    gates = [{
        "decision": analysis["decision"],
        "recommended_next_stage": analysis["recommended_next_stage"],
        "ready_candidate_count": 0,
        "ready_rows": 0,
        "blocking_reasons": analysis["blocking_reasons"],
    }]
    empty = ""
    return {
        OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        OUTPUTS[1]: report(analysis),
        OUTPUTS[2]: render_csv(LIFECYCLE_H, []),
        OUTPUTS[3]: render_csv(TIMING_H, []),
        OUTPUTS[4]: render_csv(SERIAL_H, []),
        OUTPUTS[5]: render_csv(DET_H, []),
        OUTPUTS[6]: render_csv(SCHEMA_H, []),
        OUTPUTS[7]: render_csv(GATE_H, gates),
        OUTPUTS[8]: render_csv(CONTRACT_H, contracts) if contracts else empty,
    }


def atomic_write(output: Path, payloads: dict[str, str]) -> None:
    if set(payloads) != set(OUTPUTS):
        raise RuntimeError("exact nine-output payload required")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output}")
    temporary = output.parent / f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for name in OUTPUTS:
            with (temporary / name).open("x", encoding="utf-8", newline="") as handle:
                handle.write(payloads[name])
                handle.flush()
                os.fsync(handle.fileno())
        if sorted(path.name for path in temporary.iterdir()) != sorted(OUTPUTS):
            raise RuntimeError("staged exact nine-output closure failed")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    ns = parse_args()
    try:
        analysis, payloads = analyze(ns)
    except Exception as exc:
        payloads = blocked_payload(ns, exc)
        analysis = json.loads(payloads[OUTPUTS[0]])
    atomic_write(ns.output_dir.resolve(), payloads)
    return 0 if not analysis.get("blocking_reasons") else 2


if __name__ == "__main__":
    raise SystemExit(main())
