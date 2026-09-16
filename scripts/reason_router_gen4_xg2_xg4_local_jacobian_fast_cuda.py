from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import (
    reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1
    as phase1,
)
from scripts import (
    reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase2
    as phase2,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg2-xg4-local-jacobian"
IMPLEMENTATION_SCOPE_COMMIT = (
    "3a9dd04b3de398069eed3844dfed719e2ba36ca7"
)
PHASE1_ARTIFACT_FREEZE_COMMIT = (
    phase2.PHASE1_ARTIFACT_FREEZE_COMMIT
)
PHASE1_ARTIFACT_ROOT = phase2.PHASE1_ARTIFACT_ROOT

SOURCE_PAIR_COUNT = phase2.SOURCE_PAIR_COUNT
EPSILONS = (0.025, 0.05)
FORWARDS_PER_SIGNED_PROBE = 2
SIGNED_PROBES_PER_EPSILON = 2
FORWARDS_PER_PAIR_PER_EPSILON = (
    FORWARDS_PER_SIGNED_PROBE * SIGNED_PROBES_PER_EPSILON
)
FORWARDS_PER_PAIR = (
    FORWARDS_PER_PAIR_PER_EPSILON * len(EPSILONS)
)
SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY = (
    SOURCE_PAIR_COUNT * FORWARDS_PER_PAIR
)
BASELINE_FORWARD_BUDGET_THIS_RUN = 0

ITEM_SCHEMA = "gen4-k-xg2-xg4-local-jacobian-item-v1"
SUMMARY_SCHEMA = "gen4-k-xg2-xg4-local-jacobian-summary-v1"
MANIFEST_SCHEMA = "gen4-k-xg2-xg4-local-jacobian-manifest-v1"
RESULT_PASS = "PASS_XG2_XG4_LOCAL_JACOBIAN_OBSERVATION"

ITEM_FILE = "local_jacobian_items.jsonl"
SUMMARY_FILE = "local_jacobian_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

REUSED_PATHS = (
    "scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1.py",
    "scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase2.py",
    "scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_full_baseline.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runtime.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_core.py",
)


class LocalJacobianError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LocalJacobianError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LocalJacobianError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.call(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    require(
        _git_is_ancestor(
            IMPLEMENTATION_SCOPE_COMMIT,
            expected_head,
        ),
        "IMPLEMENTATION_SCOPE_NOT_ANCESTOR",
    )
    require(
        _git_is_ancestor(
            PHASE1_ARTIFACT_FREEZE_COMMIT,
            expected_head,
        ),
        "PHASE1_ARTIFACT_FREEZE_NOT_ANCESTOR",
    )

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            PHASE1_ARTIFACT_FREEZE_COMMIT,
            expected_head,
            "--",
            PHASE1_ARTIFACT_ROOT.as_posix(),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "FROZEN_PHASE1_ARTIFACT_TREE_DRIFT")

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            IMPLEMENTATION_SCOPE_COMMIT,
            expected_head,
            "--",
            *REUSED_PATHS,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REUSED_RUNTIME_DEPENDENCY_DRIFT")


def _expected_pairs(family: str) -> tuple[str, ...]:
    return phase1._expected_pairs(family)


def _epsilon_key(epsilon: float) -> str:
    if epsilon == EPSILONS[0]:
        return "0.025"
    if epsilon == EPSILONS[1]:
        return "0.05"
    raise LocalJacobianError(f"UNAUTHORIZED_EPSILON:{epsilon}")


def _unit_direction(
    plan: torch.Tensor,
    pair: str,
) -> tuple[torch.Tensor, float]:
    require(torch.is_tensor(plan), f"PLAN_NOT_TENSOR:{pair}")
    direction = (
        plan.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
        .clone()
    )
    require(
        bool(torch.isfinite(direction).all().item()),
        f"NONFINITE_PLAN:{pair}",
    )
    norm = float(
        torch.linalg.vector_norm(direction, ord=2).item()
    )
    require(math.isfinite(norm), f"NONFINITE_PLAN_NORM:{pair}")
    require(norm > 0.0, f"ZERO_PLAN_NORM:{pair}")

    unit = (direction / norm).contiguous()
    unit_norm = float(
        torch.linalg.vector_norm(unit, ord=2).item()
    )
    require(
        math.isfinite(unit_norm)
        and abs(unit_norm - 1.0) <= 1.0e-12,
        f"UNIT_DIRECTION_NORM:{pair}:{unit_norm}",
    )
    return unit, norm


def _validate_all_plan_norms(
    items: Sequence[Mapping[str, Any]],
    plans: torch.Tensor,
) -> None:
    require(
        len(items) == SOURCE_PAIR_COUNT,
        "PHASE1_ITEM_COUNT",
    )
    require(
        torch.is_tensor(plans)
        and int(plans.shape[0]) == SOURCE_PAIR_COUNT,
        "PHASE1_PLAN_COUNT",
    )
    for index, row in enumerate(items):
        pair = str(row["source_pair_id"])
        require(
            pair == _expected_pairs(str(row["family_key"]))[index],
            f"PHASE1_PAIR_ORDER:{index}",
        )
        _unit_direction(plans[index], pair)


def _input_row(
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    pair: str,
    cell: str,
) -> torch.Tensor:
    return phase2._input_row(
        encoded,
        row_index,
        pair,
        cell,
    )


def _run_signed_probe(
    family: str,
    baseline_item: Mapping[str, Any],
    unit_direction: torch.Tensor,
    *,
    epsilon: float,
    orientation: int,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    budget: Any,
) -> dict[str, Any]:
    require(family in {"xg2", "xg4"}, f"UNSUPPORTED_FAMILY:{family}")
    require(
        baseline_item["family_key"] == family,
        "BASELINE_ITEM_FAMILY",
    )
    require(
        epsilon in EPSILONS,
        f"UNAUTHORIZED_EPSILON:{epsilon}",
    )
    require(orientation in {-1, 1}, f"BAD_ORIENTATION:{orientation}")

    pair = str(baseline_item["source_pair_id"])
    unit_norm = float(
        torch.linalg.vector_norm(
            unit_direction.detach().cpu().to(torch.float64),
            ord=2,
        ).item()
    )
    require(
        math.isfinite(unit_norm)
        and abs(unit_norm - 1.0) <= 1.0e-12,
        f"UNIT_DIRECTION_NORM:{pair}:{unit_norm}",
    )

    delta_h = (
        unit_direction.detach()
        .cpu()
        .to(torch.float64)
        .mul(float(orientation) * 2.0 * float(epsilon))
        .contiguous()
        .clone()
    )
    require(
        bool(torch.isfinite(delta_h).all().item()),
        f"NONFINITE_PROBE_PLAN:{pair}",
    )

    parent = phase1.base.prevalence_eq.parent
    runtime = phase1.base.prevalence_eq.transport_runtime
    core = phase1.base.prevalence_eq.core
    cells = phase1._cells()
    anchors = phase1._anchors_for_pair(pair, events)

    for role, field in (
        ("tp", "target_plus_anchor"),
        ("tm", "target_minus_anchor"),
        ("rp", "reference_plus_anchor"),
        ("rm", "reference_minus_anchor"),
    ):
        require(
            anchors[role] == int(baseline_item[field]),
            f"FROZEN_ANCHOR_IDENTITY:{pair}:{role}",
        )

    captured: dict[str, Any] = {}
    for role, plus_branch in (("tp", True), ("tm", False)):
        captured[role] = parent.capture_branch(
            model,
            runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            input_ids=_input_row(
                encoded,
                row_index,
                pair,
                cells[role],
            ),
            anchor=anchors[role],
            budget=budget,
            capture_states=True,
            delta_h=delta_h,
            plus_branch=plus_branch,
        )

    audit = runtime.paired_intervention_audit(
        captured["tp"]["intervention_audit"],
        captured["tm"]["intervention_audit"],
        delta_h,
        plus_expected_token_index=(
            anchors["tp"] + core.TARGET_OFFSET
        ),
        minus_expected_token_index=(
            anchors["tm"] + core.TARGET_OFFSET
        ),
    )

    plus_pe = float(parent.path_efficiency(captured["tp"]))
    minus_pe = float(parent.path_efficiency(captured["tm"]))
    response = plus_pe - minus_pe

    values = (
        plus_pe,
        minus_pe,
        response,
        float(audit["midpoint_max_abs_residual"]),
        float(audit["pair_delta_max_abs_residual"]),
        float(audit["applied_correction_max_abs_residual"]),
        float(audit["runtime_correction_l2"]),
    )
    require(
        all(math.isfinite(value) for value in values),
        f"NONFINITE_SIGNED_PROBE:{pair}:{epsilon}:{orientation}",
    )

    intended_delta_l2 = 2.0 * float(epsilon)
    require(
        abs(float(audit["runtime_correction_l2"]) - intended_delta_l2)
        <= 1.0e-12,
        (
            "RUNTIME_CORRECTION_L2_MISMATCH:"
            f"{pair}:{epsilon}:{orientation}:"
            f"{audit['runtime_correction_l2']}"
        ),
    )

    return {
        "orientation": int(orientation),
        "delta_h_l2": intended_delta_l2,
        "plus_path_efficiency": plus_pe,
        "minus_path_efficiency": minus_pe,
        "F": response,
        "midpoint_max_abs_residual": float(
            audit["midpoint_max_abs_residual"]
        ),
        "pair_delta_max_abs_residual": float(
            audit["pair_delta_max_abs_residual"]
        ),
        "applied_correction_max_abs_residual": float(
            audit["applied_correction_max_abs_residual"]
        ),
        "runtime_correction_l2": float(
            audit["runtime_correction_l2"]
        ),
        "model_forward_count": FORWARDS_PER_SIGNED_PROBE,
    }


def _run_local_jacobian_pair(
    family: str,
    baseline_item: Mapping[str, Any],
    alignment_delta_h: torch.Tensor,
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    budget: Any,
) -> dict[str, Any]:
    pair = str(baseline_item["source_pair_id"])
    unit, original_plan_l2 = _unit_direction(
        alignment_delta_h,
        pair,
    )
    f0 = float(baseline_item["delta_baseline"])
    require(math.isfinite(f0), f"NONFINITE_F0:{pair}")

    probes: dict[str, Any] = {}

    for epsilon in EPSILONS:
        positive = _run_signed_probe(
            family,
            baseline_item,
            unit,
            epsilon=epsilon,
            orientation=1,
            model=model,
            runtime_ctx=runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            encoded=encoded,
            row_index=row_index,
            events=events,
            budget=budget,
        )
        negative = _run_signed_probe(
            family,
            baseline_item,
            unit,
            epsilon=epsilon,
            orientation=-1,
            model=model,
            runtime_ctx=runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            encoded=encoded,
            row_index=row_index,
            events=events,
            budget=budget,
        )

        f_plus = float(positive["F"])
        f_minus = float(negative["F"])
        j_value = (f_plus - f_minus) / (2.0 * epsilon)
        k_value = (
            f_plus + f_minus - 2.0 * f0
        ) / (epsilon * epsilon)

        require(
            all(
                math.isfinite(value)
                for value in (f_plus, f_minus, j_value, k_value)
            ),
            f"NONFINITE_LOCAL_RESPONSE:{pair}:{epsilon}",
        )

        probes[_epsilon_key(epsilon)] = {
            "epsilon": float(epsilon),
            "F_plus": f_plus,
            "F_minus": f_minus,
            "J": j_value,
            "K": k_value,
            "positive_probe": positive,
            "negative_probe": negative,
            "model_forward_count": FORWARDS_PER_PAIR_PER_EPSILON,
        }

    item = dict(baseline_item)
    item["phase1_schema_version"] = item["schema_version"]
    item["schema_version"] = ITEM_SCHEMA
    item["implementation_scope_commit"] = IMPLEMENTATION_SCOPE_COMMIT
    item["phase1_artifact_freeze_commit"] = PHASE1_ARTIFACT_FREEZE_COMMIT
    item["original_alignment_plan_l2"] = original_plan_l2
    item["unit_direction_l2"] = 1.0
    item["F0_delta_baseline"] = f0
    item["epsilons"] = list(EPSILONS)
    item["probes"] = probes
    item["baseline_model_forward_count_this_run"] = 0
    item["scientific_model_forward_count_this_run"] = FORWARDS_PER_PAIR

    return item


def _validate_items(
    family: str,
    items: Sequence[Mapping[str, Any]],
) -> None:
    require(family in {"xg2", "xg4"}, f"UNSUPPORTED_FAMILY:{family}")
    require(len(items) == SOURCE_PAIR_COUNT, "ITEM_COUNT")
    expected_pairs = _expected_pairs(family)

    for index, (expected_pair, raw) in enumerate(
        zip(expected_pairs, items, strict=True)
    ):
        row = dict(raw)
        require(row.get("schema_version") == ITEM_SCHEMA, f"ITEM_SCHEMA:{index}")
        require(
            row.get("phase1_schema_version") == phase1.ITEM_SCHEMA,
            f"PHASE1_SCHEMA:{index}",
        )
        require(row.get("family_key") == family, f"ITEM_FAMILY:{index}")
        require(row.get("source_pair_id") == expected_pair, f"PAIR_ORDER:{index}")
        require(row.get("alignment_plan_index") == index, f"PLAN_INDEX:{index}")
        require(
            row.get("implementation_scope_commit") == IMPLEMENTATION_SCOPE_COMMIT,
            f"SCOPE_COMMIT:{index}",
        )
        require(
            row.get("phase1_artifact_freeze_commit")
            == PHASE1_ARTIFACT_FREEZE_COMMIT,
            f"PHASE1_FREEZE:{index}",
        )
        require(row.get("epsilons") == list(EPSILONS), f"EPSILONS:{index}")
        require(
            row.get("baseline_model_forward_count_this_run") == 0,
            f"BASELINE_FORWARD_COUNT:{index}",
        )
        require(
            row.get("scientific_model_forward_count_this_run") == FORWARDS_PER_PAIR,
            f"SCIENTIFIC_FORWARD_COUNT:{index}",
        )

        original_l2 = float(row["original_alignment_plan_l2"])
        unit_l2 = float(row["unit_direction_l2"])
        f0 = float(row["F0_delta_baseline"])
        require(
            math.isfinite(original_l2) and original_l2 > 0.0,
            f"PLAN_L2:{index}",
        )
        require(
            math.isfinite(unit_l2) and abs(unit_l2 - 1.0) <= 1.0e-12,
            f"UNIT_L2:{index}",
        )
        require(math.isfinite(f0), f"F0:{index}")
        require(f0 == float(row["delta_baseline"]), f"F0_IDENTITY:{index}")

        probes = row.get("probes")
        require(isinstance(probes, dict), f"PROBES_OBJECT:{index}")
        require(
            set(probes) == {_epsilon_key(epsilon) for epsilon in EPSILONS},
            f"PROBE_KEYS:{index}",
        )

        for epsilon in EPSILONS:
            probe = probes[_epsilon_key(epsilon)]
            require(
                float(probe["epsilon"]) == epsilon,
                f"EPSILON_VALUE:{index}:{epsilon}",
            )
            require(
                int(probe["model_forward_count"])
                == FORWARDS_PER_PAIR_PER_EPSILON,
                f"EPSILON_FORWARD_COUNT:{index}:{epsilon}",
            )

            f_plus = float(probe["F_plus"])
            f_minus = float(probe["F_minus"])
            j_value = float(probe["J"])
            k_value = float(probe["K"])
            require(
                all(
                    math.isfinite(value)
                    for value in (f_plus, f_minus, j_value, k_value)
                ),
                f"NONFINITE_PROBE:{index}:{epsilon}",
            )
            require(
                j_value == (f_plus - f_minus) / (2.0 * epsilon),
                f"J_IDENTITY:{index}:{epsilon}",
            )
            require(
                k_value
                == (f_plus + f_minus - 2.0 * f0) / (epsilon * epsilon),
                f"K_IDENTITY:{index}:{epsilon}",
            )

            for name, orientation in (
                ("positive_probe", 1),
                ("negative_probe", -1),
            ):
                signed = probe[name]
                require(
                    int(signed["orientation"]) == orientation,
                    f"ORIENTATION:{index}:{epsilon}:{name}",
                )
                require(
                    int(signed["model_forward_count"])
                    == FORWARDS_PER_SIGNED_PROBE,
                    f"SIGNED_FORWARD_COUNT:{index}:{epsilon}:{name}",
                )
                runtime_l2 = float(signed["runtime_correction_l2"])
                require(
                    math.isfinite(runtime_l2)
                    and abs(runtime_l2 - 2.0 * epsilon) <= 1.0e-12,
                    f"RUNTIME_L2:{index}:{epsilon}:{name}",
                )
                for audit_field in (
                    "midpoint_max_abs_residual",
                    "pair_delta_max_abs_residual",
                    "applied_correction_max_abs_residual",
                ):
                    require(
                        math.isfinite(float(signed[audit_field])),
                        f"AUDIT_NONFINITE:{index}:{epsilon}:{name}:{audit_field}",
                    )


def _write_outputs(
    output_dir: Path,
    *,
    family: str,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")
    _validate_items(family, items)

    output_dir.mkdir(parents=True, exist_ok=False)
    payloads = {
        ITEM_FILE: phase2.jsonl_bytes(items),
        SUMMARY_FILE: phase2.canonical_json_bytes(summary),
    }
    hashes: dict[str, str] = {}
    for name, raw in payloads.items():
        path = output_dir / name
        path.write_bytes(raw)
        hashes[name] = phase2.sha256_bytes(raw)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "files": {
            name: {
                "sha256": digest,
                "bytes": int((output_dir / name).stat().st_size),
            }
            for name, digest in sorted(hashes.items())
        },
    }
    manifest_raw = phase2.canonical_json_bytes(manifest)
    (output_dir / MANIFEST_FILE).write_bytes(manifest_raw)
    hashes[MANIFEST_FILE] = phase2.sha256_bytes(manifest_raw)

    checksum_raw = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(hashes.items())
    ).encode("utf-8")
    (output_dir / CHECKSUM_FILE).write_bytes(checksum_raw)
    return hashes


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT_REQUIRED:{line_no}",
        )
        rows.append(value)
    return rows


def validate_local_jacobian_artifact(
    output_dir: Path,
    family: str,
) -> dict[str, Any]:
    require(family in {"xg2", "xg4"}, f"UNSUPPORTED_FAMILY:{family}")
    manifest_path = output_dir / MANIFEST_FILE
    checksum_path = output_dir / CHECKSUM_FILE
    require(manifest_path.is_file(), "MANIFEST_MISSING")
    require(checksum_path.is_file(), "CHECKSUM_MISSING")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8-sig")
    )
    require(
        manifest.get("schema_version") == MANIFEST_SCHEMA,
        "MANIFEST_SCHEMA",
    )
    files = manifest.get("files")
    require(isinstance(files, dict), "MANIFEST_FILES")
    require(
        set(files) == {ITEM_FILE, SUMMARY_FILE},
        "MANIFEST_FILE_SET",
    )

    observed_hashes: dict[str, str] = {}
    for name in (ITEM_FILE, SUMMARY_FILE):
        path = output_dir / name
        require(path.is_file(), f"ARTIFACT_MISSING:{name}")
        observed_sha = phase2.sha256_file(path)
        observed_bytes = int(path.stat().st_size)
        require(
            observed_sha == files[name]["sha256"],
            f"ARTIFACT_SHA256:{name}",
        )
        require(
            observed_bytes == int(files[name]["bytes"]),
            f"ARTIFACT_BYTES:{name}",
        )
        observed_hashes[name] = observed_sha

    observed_hashes[MANIFEST_FILE] = phase2.sha256_file(manifest_path)

    checksum_rows: dict[str, str] = {}
    for line in checksum_path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in checksum_rows, f"CHECKSUM_DUPLICATE:{name}")
        checksum_rows[name] = digest

    require(
        checksum_rows
        == {
            name: digest
            for name, digest in sorted(observed_hashes.items())
        },
        "CHECKSUM_CONTENT",
    )

    items = _load_jsonl(output_dir / ITEM_FILE)
    _validate_items(family, items)

    summary = json.loads(
        (output_dir / SUMMARY_FILE).read_text(encoding="utf-8-sig")
    )
    require(summary.get("schema_version") == SUMMARY_SCHEMA, "SUMMARY_SCHEMA")
    require(summary.get("result") == RESULT_PASS, "SUMMARY_RESULT")
    require(summary.get("family_key") == family, "SUMMARY_FAMILY")
    require(
        summary.get("source_pair_count") == SOURCE_PAIR_COUNT,
        "SUMMARY_PAIR_COUNT",
    )
    require(
        summary.get("implementation_scope_commit") == IMPLEMENTATION_SCOPE_COMMIT,
        "SUMMARY_SCOPE_COMMIT",
    )
    require(
        summary.get("phase1_artifact_freeze_commit")
        == PHASE1_ARTIFACT_FREEZE_COMMIT,
        "SUMMARY_PHASE1_FREEZE",
    )
    require(summary.get("epsilons") == list(EPSILONS), "SUMMARY_EPSILONS")
    require(
        summary.get("baseline_model_forward_count_this_run") == 0,
        "SUMMARY_BASELINE_FORWARD_COUNT",
    )
    require(
        summary.get("scientific_model_forward_count_this_run")
        == SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY,
        "SUMMARY_SCIENTIFIC_FORWARD_COUNT",
    )
    require(
        summary.get("primary_inference_executed") is False,
        "SUMMARY_PRIMARY_INFERENCE_BOUNDARY",
    )
    require(
        summary.get("scale_consistency_inference_executed") is False,
        "SUMMARY_SCALE_INFERENCE_BOUNDARY",
    )
    require(
        summary.get("training_executed") is False
        and summary.get("backward_executed") is False
        and summary.get("task_heads_executed") is False
        and summary.get("logits_read") is False,
        "SUMMARY_EXECUTION_BOUNDARY",
    )
    require(
        summary.get("scientific_conclusion") is None,
        "SUMMARY_CONCLUSION_BOUNDARY",
    )
    return {
        "summary": summary,
        "items": items,
        "manifest": manifest,
    }


def run_local_jacobian(
    *,
    family: str,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(family in {"xg2", "xg4"}, f"UNSUPPORTED_FAMILY:{family}")
    authenticate_repo(expected_head)
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")

    # All frozen artifact and plan-norm checks happen before runtime/model setup.
    phase1_loaded = phase2.load_phase1_artifact(family)
    phase1_summary = phase1_loaded["summary"]
    phase1_items = phase1_loaded["items"]
    phase1_plans = phase1_loaded["alignment_delta_h"]
    _validate_all_plan_norms(phase1_items, phase1_plans)

    phase1.base.prevalence_eq.backend.runtime_gate()

    with phase1.base.prevalence_eq.backend.parent_runtime_rebind():
        rows, encoded, event_rows = (
            phase1.base.fresh_eq.load_family_inputs(
                family,
                tokenizer_snapshot,
            )
        )
        pairs = phase1.base.fresh_eq._pair_order(family, rows)
        require(tuple(pairs) == _expected_pairs(family), "PAIR_ORDER")
        require(
            tuple(str(row["source_pair_id"]) for row in phase1_items)
            == tuple(pairs),
            "PHASE1_INPUT_PAIR_ORDER",
        )

        parent = phase1.base.prevalence_eq.parent
        events = parent.event_lookup(event_rows)
        parent.validate_transport_event_plan(pairs, events)
        row_index = parent.build_row_index(rows)
        trace_code, trace_line = (
            phase1.base.prevalence_eq.measurement
            ._resolve_and_validate_runtime_binding()
        )

        kernels = (
            phase1.base.prevalence_eq.kernel_compat
            .load_exact_fast_kernels()
        )
        with (
            phase1.base.prevalence_eq.kernel_compat
            .exact_transformers_kernel_loader(kernels)
        ) as constructor_kernel_calls:
            model, checkpoint_sha = (
                parent.load_representative_model_external(
                    model_snapshot=model_snapshot,
                    checkpoint_path=checkpoint_path,
                )
            )
            require(
                checkpoint_sha
                == (
                    phase1.base.prevalence_eq.extraction
                    .REPRESENTATIVE_CHECKPOINT_SHA256
                ),
                "CHECKPOINT_IDENTITY",
            )
            runtime_ctx = (
                phase1.base.prevalence_eq.transport_runtime
                .validate_runtime_components(model)
            )

        constructor_counts = Counter(constructor_kernel_calls)
        require(
            set(constructor_counts) == {"causal-conv1d", "mamba-ssm"},
            f"TRANSFORMERS_CONSTRUCTOR_KERNEL_NAMES:{dict(constructor_counts)}",
        )
        require(
            constructor_counts["causal-conv1d"] > 0
            and constructor_counts["causal-conv1d"]
            == constructor_counts["mamba-ssm"],
            f"TRANSFORMERS_CONSTRUCTOR_KERNEL_CALL_COUNT:{dict(constructor_counts)}",
        )
        (
            phase1.base.prevalence_eq.kernel_compat
            .validate_transformers_kernel_bindings(kernels)
        )

        model.to(torch.device("cuda:0"))
        model.eval()
        require(
            all(
                parameter.device.type == "cuda"
                for parameter in model.mamba.parameters()
            ),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = (
            phase1.base.prevalence_eq.backend
            ._make_fast_capture(kernels)
        )
        original_capture = parent.capture_branch
        budget = parent.ForwardBudget(
            SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY
        )
        items: list[dict[str, Any]] = []

        parent.capture_branch = fast_capture
        try:
            for index, pair in enumerate(pairs):
                require(
                    phase1_items[index]["source_pair_id"] == pair,
                    f"PAIR_IDENTITY:{index}",
                )
                item = _run_local_jacobian_pair(
                    family,
                    phase1_items[index],
                    phase1_plans[index],
                    model=model,
                    runtime_ctx=runtime_ctx,
                    trace_code=trace_code,
                    trace_line=trace_line,
                    encoded=encoded,
                    row_index=row_index,
                    events=events,
                    budget=budget,
                )
                items.append(item)

            budget.assert_exact()
            torch.cuda.synchronize()
        finally:
            parent.capture_branch = original_capture

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "family_key": family,
        "execution_head": expected_head,
        "implementation_scope_commit": IMPLEMENTATION_SCOPE_COMMIT,
        "phase1_artifact_freeze_commit": PHASE1_ARTIFACT_FREEZE_COMMIT,
        "phase1_artifact_path": PHASE1_ARTIFACT_ROOT.joinpath(family).as_posix(),
        "phase1_artifact_manifest_sha256": phase1_loaded[
            "artifact_manifest_sha256"
        ],
        "phase1_artifact_checksum_sha256": phase1_loaded[
            "artifact_checksum_sha256"
        ],
        "phase1_result": phase1_summary["result"],
        "source_pair_count": SOURCE_PAIR_COUNT,
        "pair_id_first": items[0]["source_pair_id"],
        "pair_id_last": items[-1]["source_pair_id"],
        "epsilons": list(EPSILONS),
        "unit_direction_from_frozen_phase1_plan": True,
        "phase2_response_used_for_selection": False,
        "baseline_model_forward_count_this_run": 0,
        "scientific_model_forward_count_this_run": (
            SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY
        ),
        "model_forwards_per_pair": FORWARDS_PER_PAIR,
        "model_forwards_per_pair_per_epsilon": (
            FORWARDS_PER_PAIR_PER_EPSILON
        ),
        "local_jacobian_observed": True,
        "local_curvature_observed": True,
        "primary_inference_executed": False,
        "scale_consistency_inference_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
        "scientific_conclusion_scope": "LOCAL_JACOBIAN_OBSERVATION_ONLY",
        "representative_checkpoint_sha256": checkpoint_sha,
    }

    _write_outputs(
        output_dir,
        family=family,
        items=items,
        summary=summary,
    )
    validated = validate_local_jacobian_artifact(output_dir, family)
    require(
        validated["summary"]["result"] == RESULT_PASS,
        "POSTWRITE_VALIDATION_RESULT",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prospective XG2/XG4 unit-direction local-Jacobian observation. "
            "Consumes frozen Phase-1 alignment_delta_h, normalizes each plan, "
            "and executes exactly epsilon=0.025 and 0.05 symmetric probes. "
            "No baseline model forward or inferential test is executed."
        )
    )
    parser.add_argument(
        "--family",
        choices=("xg2", "xg4"),
        required=True,
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_local_jacobian(
        family=args.family,
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT =", summary["result"])
    print("FAMILY =", summary["family_key"])
    print("SOURCE_PAIR_COUNT =", summary["source_pair_count"])
    print("EPSILONS =", summary["epsilons"])
    print(
        "BASELINE_MODEL_FORWARD_COUNT_THIS_RUN =",
        summary["baseline_model_forward_count_this_run"],
    )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN =",
        summary["scientific_model_forward_count_this_run"],
    )
    print("LOCAL_JACOBIAN_OBSERVED =", summary["local_jacobian_observed"])
    print("PRIMARY_INFERENCE = NOT EXECUTED")


if __name__ == "__main__":
    main()
