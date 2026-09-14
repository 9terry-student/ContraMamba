from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import reason_router_gen4_k_directional_alignment_transport_core as core
from scripts import reason_router_gen4_k_directional_alignment_transport_runner as parent
from scripts import reason_router_gen4_k_directional_alignment_transport_runtime as transport_runtime
from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as backend
from scripts import reason_router_gen4_large_correction_prospective_tokenizer_anchor_eligibility as eligibility
from scripts import reason_router_gen4_native_mamba_state_extraction as extraction
from scripts import reason_router_gen4_native_mamba_state_measurement as measurement
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-k-large-correction-prospective-validation"
PREREQUISITE_FREEZE_COMMIT = "3d7ca42005a1f5befae26acfa4e2fd3296768ce7"
ELIGIBILITY_IMPLEMENTATION_COMMIT = "2de39772dd641499ee6fef26fa57d9d0396f0e21"

ELIGIBILITY_DIR = Path(
    "reports/"
    "reason_router_gen4_large_correction_prospective_tokenizer_anchor_"
    "eligibility_2de3977"
)
ELIGIBILITY_ANCHOR_MANIFEST = ELIGIBILITY_DIR / "anchor_manifest.jsonl"
ELIGIBILITY_SUMMARY = ELIGIBILITY_DIR / "eligibility_summary.json"
ELIGIBILITY_ANCHOR_MANIFEST_SHA256 = (
    "c1e90ec0fdd41ee16a71edae23816cf37df320dcc93786201f66d04ed990a0e0"
)
ELIGIBILITY_SUMMARY_SHA256 = (
    "b617ad9bd56473a33963e4eb32387df29786f54fc022033a73a94d7c1471e3ae"
)

PROSPECTIVE_PAIR_ID = "generated_fact_301"
SOURCE_PAIR_COUNT = 300
FORWARDS_PER_BACKEND = 6
TOTAL_MODEL_FORWARDS = 12

# Reuse the already validated backend tolerances without relaxation.
STATE_ATOL = backend.STATE_ATOL
STATE_RTOL = backend.STATE_RTOL
GEOMETRY_ATOL = backend.GEOMETRY_ATOL
GEOMETRY_RTOL = backend.GEOMETRY_RTOL
PE_ATOL = backend.PE_ATOL

BRANCH_LABELS = (
    "baseline_tp",
    "baseline_tm",
    "baseline_rp",
    "baseline_rm",
    "alignment_tp",
    "alignment_tm",
)

EXACT_FIELDS = (
    "schema_version",
    "source_pair_id",
    "source_block",
    "target_residual_layer",
    "intervention_layer",
    "relative_coordinate",
    "target_plus_cell",
    "target_minus_cell",
    "reference_plus_cell",
    "reference_minus_cell",
    "anchor_name",
    "target_plus_anchor",
    "target_minus_anchor",
    "reference_plus_anchor",
    "reference_minus_anchor",
    "target_plus_intervention_token",
    "target_minus_intervention_token",
    "reference_plus_geometry_token",
    "reference_minus_geometry_token",
)

GEOMETRY_FIELDS = (
    "target_A",
    "target_B",
    "target_C",
    "reference_A",
    "reference_B",
    "reference_C",
    "alignment_shift_abs",
    "alignment_realized_A",
    "alignment_realized_B",
    "alignment_target_cosine",
    "alignment_realized_cosine",
    "alignment_cosine_abs_residual",
    "alignment_A_preservation_abs_residual",
    "alignment_B_preservation_abs_residual",
    "alignment_midpoint_max_abs_residual",
    "alignment_pair_delta_max_abs_residual",
    "alignment_applied_correction_max_abs_residual",
)

PE_FIELDS = (
    "baseline_plus_path_efficiency",
    "baseline_minus_path_efficiency",
    "alignment_plus_path_efficiency",
    "alignment_minus_path_efficiency",
    "delta_baseline",
    "delta_alignment",
    "R_ALIGN",
)

FROZEN_DEPENDENCY_PATHS = (
    "scripts/reason_router_gen4_k_directional_alignment_transport_core.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runner.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runtime.py",
    "scripts/reason_router_gen4_k_fast_cuda_one_pair_equivalence.py",
    "scripts/reason_router_gen4_large_correction_prospective_tokenizer_anchor_eligibility.py",
    "scripts/reason_router_gen4_native_mamba_state_extraction.py",
    "scripts/reason_router_gen4_native_mamba_state_measurement.py",
    "scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py",
    ELIGIBILITY_ANCHOR_MANIFEST.as_posix(),
    ELIGIBILITY_SUMMARY.as_posix(),
    eligibility.PROSPECTIVE_DATA_PATH.as_posix(),
    eligibility.PROSPECTIVE_MANIFEST_PATH.as_posix(),
)

ITEM_SCHEMA = "gen4-k-large-correction-prospective-equivalence-item-v1"
REPORT_SCHEMA = "gen4-k-large-correction-prospective-fast-cuda-one-pair-equivalence-v1"


class ProspectiveEquivalenceError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ProspectiveEquivalenceError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProspectiveEquivalenceError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            PREREQUISITE_FREEZE_COMMIT,
            expected_head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "PREREQUISITE_FREEZE_NOT_ANCESTOR")

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            PREREQUISITE_FREEZE_COMMIT,
            expected_head,
            "--",
            *FROZEN_DEPENDENCY_PATHS,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "FROZEN_DEPENDENCY_DRIFT")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT_REQUIRED:{path}:{line_no}")
        rows.append(value)
    return rows


def _pair_order(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        pair = str(row["source_pair_id"])
        if pair not in seen:
            seen.add(pair)
            order.append(pair)
    require(len(order) == SOURCE_PAIR_COUNT, "PAIR_COUNT")
    require(order[0] == PROSPECTIVE_PAIR_ID, "FIRST_PAIR_ID")
    require(order[-1] == "generated_fact_600", "LAST_PAIR_ID")
    return tuple(order)


def load_frozen_anchor_manifest() -> list[dict[str, Any]]:
    path = ROOT / ELIGIBILITY_ANCHOR_MANIFEST
    summary_path = ROOT / ELIGIBILITY_SUMMARY
    require(path.is_file(), "ELIGIBILITY_ANCHOR_MANIFEST_MISSING")
    require(summary_path.is_file(), "ELIGIBILITY_SUMMARY_MISSING")
    require(
        sha256_file(path) == ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
        "ELIGIBILITY_ANCHOR_MANIFEST_SHA256",
    )
    require(
        sha256_file(summary_path) == ELIGIBILITY_SUMMARY_SHA256,
        "ELIGIBILITY_SUMMARY_SHA256",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    require(
        summary.get("primary_complete_pair_prefix_feasibility") == "PASS_300_OF_300",
        "ELIGIBILITY_SUMMARY_NOT_PASS",
    )
    require(summary.get("complete_source_pair_count") == 300, "ELIGIBILITY_PAIR_COUNT")
    require(summary.get("required_anchor_row_count") == 1800, "ELIGIBILITY_ANCHOR_COUNT")
    require(summary.get("model_forward_count") == 0, "ELIGIBILITY_MODEL_FORWARD_BOUNDARY")
    require(summary.get("gpu_used") is False, "ELIGIBILITY_GPU_BOUNDARY")
    require(
        summary.get("anchor_manifest_sha256") == ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
        "ELIGIBILITY_SUMMARY_MANIFEST_IDENTITY",
    )

    rows = _read_jsonl(path)
    require(len(rows) == 1800, "ANCHOR_MANIFEST_ROW_COUNT")
    for index, row in enumerate(rows):
        require(
            row.get("schema_version") == eligibility.ANCHOR_MANIFEST_SCHEMA,
            f"ANCHOR_SCHEMA:{index}",
        )
        require(row.get("post4_eligible") is True, f"ANCHOR_NOT_ELIGIBLE:{index}")
        require(row.get("exclusion_code") is None, f"ANCHOR_EXCLUSION:{index}")
        require(
            type(row.get("absolute_anchor_token_index")) is int,
            f"ANCHOR_INDEX_TYPE:{index}",
        )
        require(
            type(row.get("terminal_index")) is int,
            f"TERMINAL_INDEX_TYPE:{index}",
        )
    return rows


def validate_prospective_population(
    rows: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    normalized = adapter.validate_gen4_rows(rows, require_canonical_shape=True)
    pairs = _pair_order(normalized)

    require(
        list(encoded["row_id"]) == [str(row["row_id"]) for row in normalized],
        "ENCODED_ROW_ORDER",
    )
    require(
        list(encoded["source_pair_id"])
        == [str(row["source_pair_id"]) for row in normalized],
        "ENCODED_PAIR_ORDER",
    )
    require(
        list(encoded["contrast_cell_id"])
        == [str(row["contrast_cell_id"]) for row in normalized],
        "ENCODED_CELL_ORDER",
    )

    input_ids = encoded["input_ids"]
    attention = encoded["attention_mask"]
    claim_mask = encoded["claim_mask"]
    evidence_mask = encoded["evidence_mask"]
    require(torch.is_tensor(input_ids), "INPUT_IDS_TENSOR")
    require(
        tuple(input_ids.shape) == (eligibility.EXPECTED_ROW_COUNT, adapter.MAX_LENGTH),
        "INPUT_IDS_SHAPE",
    )

    row_index = {str(row["row_id"]): i for i, row in enumerate(normalized)}
    require(len(row_index) == eligibility.EXPECTED_ROW_COUNT, "ROW_ID_CARDINALITY")

    events = parent.event_lookup(event_rows)
    parent.validate_transport_event_plan(pairs, events)

    for event in event_rows:
        row_id = str(event["row_id"])
        require(row_id in row_index, f"EVENT_ROW_ID:{row_id}")
        index = row_index[row_id]
        anchor = int(event["absolute_anchor_token_index"])
        terminal = int(event["terminal_index"])
        claim_count = int(event["claim_consumed_token_count"])
        evidence_count = int(event["evidence_consumed_token_count"])
        evidence_anchor = int(event["anchor_evidence_token_index"])

        require(anchor == claim_count + 1 + evidence_anchor, "ANCHOR_COORDINATE")
        require(terminal == claim_count + evidence_count, "TERMINAL_COORDINATE")
        require(anchor + 4 <= terminal - 1, "POST4_RULE")
        require(int(attention[index].sum().item()) == terminal + 1, "ATTENTION_TERMINAL")
        require(int(claim_mask[index].sum().item()) == claim_count, "CLAIM_MASK_COUNT")
        require(int(evidence_mask[index].sum().item()) == evidence_count, "EVIDENCE_MASK_COUNT")

    return pairs


def load_prospective_inputs(
    tokenizer_snapshot: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows, raw, _manifest = eligibility.load_frozen_prospective_rows(ROOT)
    _facts, regenerated, _materializer = (
        eligibility.reconstruct_validation_facts_and_verify_bytes(raw, root=ROOT)
    )
    require(regenerated == rows, "REGENERATED_ROW_OBJECT_DRIFT")

    tokenizer, _tokenizer_provenance = eligibility.load_canonical_analysis_tokenizer(
        tokenizer_snapshot
    )
    encoded = adapter.encode_gen4_rows(rows, tokenizer)
    event_rows = load_frozen_anchor_manifest()
    validate_prospective_population(rows, encoded, event_rows)
    return rows, encoded, event_rows


def _input_row(
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    pair: str,
    cell: str,
) -> torch.Tensor:
    key = (pair, cell)
    require(key in row_index, f"MISSING_INPUT_ROW:{key}")
    input_ids = encoded["input_ids"]
    require(torch.is_tensor(input_ids), "INPUT_IDS_NOT_TENSOR")
    index = int(row_index[key])
    return input_ids[index : index + 1].detach().cpu().contiguous()


def _prospective_reductions(
    baseline_plus: float,
    baseline_minus: float,
    alignment_plus: float,
    alignment_minus: float,
) -> dict[str, float]:
    values = (
        baseline_plus,
        baseline_minus,
        alignment_plus,
        alignment_minus,
    )
    require(all(math.isfinite(float(v)) for v in values), "NONFINITE_ENDPOINT")
    delta0 = float(baseline_plus) - float(baseline_minus)
    delta_a = float(alignment_plus) - float(alignment_minus)
    return {
        "delta_baseline": delta0,
        "delta_alignment": delta_a,
        "R_ALIGN": delta_a - delta0,
    }


def run_prospective_pair(
    pair: str,
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    budget: Any,
) -> dict[str, Any]:
    cells = {
        "tp": core.TARGET_PLUS_CELL,
        "tm": core.TARGET_MINUS_CELL,
        "rp": core.REFERENCE_PLUS_CELL,
        "rm": core.REFERENCE_MINUS_CELL,
    }
    anchors = {
        role: int(events[(pair, cell, core.ANCHOR_NAME)]["absolute_anchor_token_index"])
        for role, cell in cells.items()
    }

    baseline: dict[str, Any] = {}
    for role in ("tp", "tm", "rp", "rm"):
        baseline[role] = parent.capture_branch(
            model,
            runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            input_ids=_input_row(encoded, row_index, pair, cells[role]),
            anchor=anchors[role],
            budget=budget,
            capture_states=role in {"tp", "tm"},
        )

    target_geometry = core.reconstruct_pair_geometry(
        baseline["tp"]["geometry_branch"],
        baseline["tm"]["geometry_branch"],
        gamma=runtime_ctx["gamma"],
        w_hidden=runtime_ctx["w_hidden"],
        strong_mask=runtime_ctx["strong_mask"],
    )
    reference_geometry = core.reconstruct_pair_geometry(
        baseline["rp"]["geometry_branch"],
        baseline["rm"]["geometry_branch"],
        gamma=runtime_ctx["gamma"],
        w_hidden=runtime_ctx["w_hidden"],
        strong_mask=runtime_ctx["strong_mask"],
    )

    alignment_shift_abs = abs(
        float(reference_geometry["C"]) - float(target_geometry["C"])
    )
    require(math.isfinite(alignment_shift_abs), "NONFINITE_ALIGNMENT_SHIFT")

    alignment_norm_delta, align_core = core.alignment_delta(
        target_geometry["x"],
        target_geometry["y"],
        reference_geometry["C"],
    )
    alignment_delta_h = target_geometry["d"] * alignment_norm_delta

    alignment: dict[str, Any] = {}
    for role, plus_branch in (("tp", True), ("tm", False)):
        alignment[role] = parent.capture_branch(
            model,
            runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            input_ids=_input_row(encoded, row_index, pair, cells[role]),
            anchor=anchors[role],
            budget=budget,
            capture_states=True,
            delta_h=alignment_delta_h,
            plus_branch=plus_branch,
        )

    alignment_pair_audit = transport_runtime.paired_intervention_audit(
        alignment["tp"]["intervention_audit"],
        alignment["tm"]["intervention_audit"],
        alignment_delta_h,
        plus_expected_token_index=anchors["tp"] + core.TARGET_OFFSET,
        minus_expected_token_index=anchors["tm"] + core.TARGET_OFFSET,
    )

    baseline_plus = parent.path_efficiency(baseline["tp"])
    baseline_minus = parent.path_efficiency(baseline["tm"])
    alignment_plus = parent.path_efficiency(alignment["tp"])
    alignment_minus = parent.path_efficiency(alignment["tm"])
    reductions = _prospective_reductions(
        baseline_plus,
        baseline_minus,
        alignment_plus,
        alignment_minus,
    )

    align_cos_resid = abs(
        float(align_core["realized_C"]) - float(align_core["target_C"])
    )

    return {
        "schema_version": ITEM_SCHEMA,
        "source_pair_id": pair,
        "source_block": core.SOURCE_BLOCK,
        "target_residual_layer": core.TARGET_RESIDUAL_LAYER,
        "intervention_layer": core.INTERVENTION_LAYER,
        "relative_coordinate": core.TARGET_OFFSET,
        "target_plus_cell": core.TARGET_PLUS_CELL,
        "target_minus_cell": core.TARGET_MINUS_CELL,
        "reference_plus_cell": core.REFERENCE_PLUS_CELL,
        "reference_minus_cell": core.REFERENCE_MINUS_CELL,
        "anchor_name": core.ANCHOR_NAME,
        "target_plus_anchor": anchors["tp"],
        "target_minus_anchor": anchors["tm"],
        "reference_plus_anchor": anchors["rp"],
        "reference_minus_anchor": anchors["rm"],
        "target_plus_intervention_token": anchors["tp"] + core.TARGET_OFFSET,
        "target_minus_intervention_token": anchors["tm"] + core.TARGET_OFFSET,
        "reference_plus_geometry_token": anchors["rp"] + core.TARGET_OFFSET,
        "reference_minus_geometry_token": anchors["rm"] + core.TARGET_OFFSET,
        "target_A": float(target_geometry["A"]),
        "target_B": float(target_geometry["B"]),
        "target_C": float(target_geometry["C"]),
        "reference_A": float(reference_geometry["A"]),
        "reference_B": float(reference_geometry["B"]),
        "reference_C": float(reference_geometry["C"]),
        "alignment_shift_abs": alignment_shift_abs,
        "alignment_realized_A": float(align_core["realized_A"]),
        "alignment_realized_B": float(align_core["realized_B"]),
        "alignment_target_cosine": float(align_core["target_C"]),
        "alignment_realized_cosine": float(align_core["realized_C"]),
        "alignment_cosine_abs_residual": align_cos_resid,
        "alignment_A_preservation_abs_residual": float(
            align_core["A_preservation_abs_residual"]
        ),
        "alignment_B_preservation_abs_residual": float(
            align_core["B_preservation_abs_residual"]
        ),
        "alignment_midpoint_max_abs_residual": float(
            alignment_pair_audit["midpoint_max_abs_residual"]
        ),
        "alignment_pair_delta_max_abs_residual": float(
            alignment_pair_audit["pair_delta_max_abs_residual"]
        ),
        "alignment_applied_correction_max_abs_residual": float(
            alignment_pair_audit["applied_correction_max_abs_residual"]
        ),
        "baseline_plus_path_efficiency": float(baseline_plus),
        "baseline_minus_path_efficiency": float(baseline_minus),
        "alignment_plus_path_efficiency": float(alignment_plus),
        "alignment_minus_path_efficiency": float(alignment_minus),
        **reductions,
    }


def _five_state_window(states: Sequence[Any] | None, anchor: int):
    if states is None:
        return None
    require(anchor + 4 < len(states), "STATE_WINDOW_RANGE")
    return [
        np.ascontiguousarray(np.asarray(states[t], dtype=np.float32)).copy()
        for t in range(anchor, anchor + 5)
    ]


def _logging_capture(base_capture: Any, records: list[dict[str, Any]]):
    counter = {"value": 0}

    def wrapped(*args, **kwargs):
        index = counter["value"]
        require(index < len(BRANCH_LABELS), "CAPTURE_CALL_OVERFLOW")
        counter["value"] += 1
        result = base_capture(*args, **kwargs)
        anchor = int(result["anchor"])
        records.append({
            "label": BRANCH_LABELS[index],
            "anchor": anchor,
            "target_abs": int(result["target_abs"]),
            "states": _five_state_window(result["states"], anchor),
        })
        return result

    return wrapped


def _compare_scalar(
    observed: float,
    reference: float,
    *,
    atol: float,
    rtol: float = 0.0,
    label: str,
) -> float:
    a = float(observed)
    b = float(reference)
    require(math.isfinite(a) and math.isfinite(b), f"NONFINITE:{label}")
    diff = abs(a - b)
    limit = atol + rtol * abs(b)
    require(diff <= limit, f"EQUIVALENCE_FAILURE:{label}:{diff}:{limit}")
    return diff


def compare_pair(
    cpu_item: Mapping[str, Any],
    gpu_item: Mapping[str, Any],
    cpu_records: Sequence[Mapping[str, Any]],
    gpu_records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    require(set(cpu_item) == set(gpu_item), "ITEM_SCHEMA")

    for field in EXACT_FIELDS:
        require(cpu_item[field] == gpu_item[field], f"EXACT_FIELD:{field}")

    max_geometry = 0.0
    for field in GEOMETRY_FIELDS:
        max_geometry = max(
            max_geometry,
            _compare_scalar(
                gpu_item[field],
                cpu_item[field],
                atol=GEOMETRY_ATOL,
                rtol=GEOMETRY_RTOL,
                label=f"geometry:{field}",
            ),
        )

    max_pe = 0.0
    for field in PE_FIELDS:
        max_pe = max(
            max_pe,
            _compare_scalar(
                gpu_item[field],
                cpu_item[field],
                atol=PE_ATOL,
                label=f"pe:{field}",
            ),
        )

    require(
        len(cpu_records) == len(BRANCH_LABELS)
        and len(gpu_records) == len(BRANCH_LABELS),
        "CAPTURE_RECORD_COUNT",
    )

    max_state_abs = 0.0
    compared_windows = 0
    for cpu_record, gpu_record in zip(cpu_records, gpu_records, strict=True):
        require(cpu_record["label"] == gpu_record["label"], "CAPTURE_LABEL")
        require(cpu_record["anchor"] == gpu_record["anchor"], "CAPTURE_ANCHOR")
        require(cpu_record["target_abs"] == gpu_record["target_abs"], "CAPTURE_TARGET")
        cpu_states = cpu_record["states"]
        gpu_states = gpu_record["states"]
        require((cpu_states is None) == (gpu_states is None), "CAPTURE_STATE_PRESENCE")
        if cpu_states is None:
            continue
        require(len(cpu_states) == 5 and len(gpu_states) == 5, "STATE_WINDOW_WIDTH")
        compared_windows += 1
        for offset, (cpu_state, gpu_state) in enumerate(
            zip(cpu_states, gpu_states, strict=True)
        ):
            require(cpu_state.shape == gpu_state.shape, "STATE_SHAPE_MISMATCH")
            cpu64 = np.asarray(cpu_state, dtype=np.float64)
            gpu64 = np.asarray(gpu_state, dtype=np.float64)
            max_abs = float(np.max(np.abs(gpu64 - cpu64)))
            scale = float(np.max(np.abs(cpu64)))
            limit = STATE_ATOL + STATE_RTOL * scale
            require(
                max_abs <= limit,
                f"STATE_EQUIVALENCE_FAILURE:{cpu_record['label']}:{offset}:{max_abs}:{limit}",
            )
            max_state_abs = max(max_state_abs, max_abs)

    require(compared_windows == 4, "STATE_WINDOW_COUNT")
    return {
        "max_state_abs_diff": max_state_abs,
        "max_geometry_abs_diff": max_geometry,
        "max_pe_abs_diff": max_pe,
    }


def run_one_pair(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    backend.runtime_gate()
    require(not output.exists(), "OUTPUT_COLLISION")

    with backend.parent_runtime_rebind():
        rows, encoded, event_rows = load_prospective_inputs(tokenizer_snapshot)
        pairs = _pair_order(rows)
        require(pairs[0] == PROSPECTIVE_PAIR_ID, "PROSPECTIVE_PAIR_SELECTION")
        pair = PROSPECTIVE_PAIR_ID

        events = parent.event_lookup(event_rows)
        parent.validate_transport_event_plan(pairs, events)
        row_index = parent.build_row_index(rows)
        trace_code, trace_line = measurement._resolve_and_validate_runtime_binding()

        cpu_model, cpu_checkpoint_sha = parent.load_representative_model_external(
            model_snapshot=model_snapshot,
            checkpoint_path=checkpoint_path,
        )
        require(
            cpu_checkpoint_sha == extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
            "CPU_CHECKPOINT_IDENTITY",
        )
        cpu_ctx = transport_runtime.validate_runtime_components(cpu_model)

        gpu_model, gpu_checkpoint_sha = parent.load_representative_model_external(
            model_snapshot=model_snapshot,
            checkpoint_path=checkpoint_path,
        )
        require(gpu_checkpoint_sha == cpu_checkpoint_sha, "GPU_CHECKPOINT_IDENTITY")
        gpu_ctx = transport_runtime.validate_runtime_components(gpu_model)

        original_capture = parent.capture_branch

        cpu_records: list[dict[str, Any]] = []
        parent.capture_branch = _logging_capture(original_capture, cpu_records)
        cpu_budget = parent.ForwardBudget(FORWARDS_PER_BACKEND)
        try:
            cpu_item = run_prospective_pair(
                pair,
                model=cpu_model,
                runtime_ctx=cpu_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                encoded=encoded,
                row_index=row_index,
                events=events,
                budget=cpu_budget,
            )
        finally:
            parent.capture_branch = original_capture
        cpu_budget.assert_exact()

        kernels = backend.load_exact_fast_kernels()
        gpu_model.to(torch.device("cuda:0"))
        gpu_model.eval()
        require(
            all(parameter.device.type == "cuda" for parameter in gpu_model.mamba.parameters()),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = backend._make_fast_capture(kernels)
        gpu_records: list[dict[str, Any]] = []
        parent.capture_branch = _logging_capture(fast_capture, gpu_records)
        gpu_budget = parent.ForwardBudget(FORWARDS_PER_BACKEND)
        try:
            gpu_item = run_prospective_pair(
                pair,
                model=gpu_model,
                runtime_ctx=gpu_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                encoded=encoded,
                row_index=row_index,
                events=events,
                budget=gpu_budget,
            )
        finally:
            parent.capture_branch = original_capture
        gpu_budget.assert_exact()
        torch.cuda.synchronize()

    comparison = compare_pair(cpu_item, gpu_item, cpu_records, gpu_records)

    report = {
        "schema_version": REPORT_SCHEMA,
        "result": "PASS_PROSPECTIVE_FAST_CUDA_ONE_PAIR_EQUIVALENCE",
        "execution_head": expected_head,
        "prerequisite_freeze_commit": PREREQUISITE_FREEZE_COMMIT,
        "eligibility_implementation_commit": ELIGIBILITY_IMPLEMENTATION_COMMIT,
        "source_pair_id": pair,
        "source_pair_selection": "first_fixed_holdout_pair_outcome_blind",
        "cpu_model_forward_count": FORWARDS_PER_BACKEND,
        "gpu_model_forward_count": FORWARDS_PER_BACKEND,
        "total_model_forward_count": TOTAL_MODEL_FORWARDS,
        "scientific_budget_forward_count": 0,
        "scientific_conclusion": None,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "raw_vectors_persisted": False,
        "endpoint_values_persisted": False,
        "prospective_response_values_persisted": False,
        "prospective_holdout_rows_sha256": eligibility.EXPECTED_PROSPECTIVE_ROWS_SHA256,
        "eligibility_anchor_manifest_sha256": ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
        "representative_checkpoint_sha256": cpu_checkpoint_sha,
        "kernels_version": backend.KERNELS_VERSION,
        "mamba_revision": backend.MAMBA_REV,
        "mamba_binary_sha256": backend.MAMBA_BINARY_SHA256,
        "causal_conv_revision": backend.CONV_REV,
        "causal_conv_binary_sha256": backend.CONV_BINARY_SHA256,
        "build_variant": backend.BUILD_VARIANT,
        "python_version": backend.EXPECTED_RUNTIME["python"],
        "numpy_version": backend.EXPECTED_RUNTIME["numpy"],
        "torch_version": backend.EXPECTED_RUNTIME["torch"],
        "transformers_version": backend.EXPECTED_RUNTIME["transformers"],
        "cuda_runtime": backend.EXPECTED_CUDA_RUNTIME,
        "cuda_device": backend.EXPECTED_DEVICE_NAME,
        "cuda_capability": list(backend.EXPECTED_CAPABILITY),
        "state_atol": STATE_ATOL,
        "state_rtol": STATE_RTOL,
        "geometry_atol": GEOMETRY_ATOL,
        "geometry_rtol": GEOMETRY_RTOL,
        "pe_atol": PE_ATOL,
        **comparison,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded CPU-slow versus fast-CUDA equivalence gate for the frozen "
            "Gen4-K large-correction prospective holdout. This is not the "
            "1800-forward scientific execution."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_one_pair(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output=args.output,
    )
    print("RESULT =", report["result"])
    print("SOURCE_PAIR_ID =", report["source_pair_id"])
    print("CPU_MODEL_FORWARD_COUNT =", report["cpu_model_forward_count"])
    print("GPU_MODEL_FORWARD_COUNT =", report["gpu_model_forward_count"])
    print("TOTAL_MODEL_FORWARD_COUNT =", report["total_model_forward_count"])
    print("MAX_STATE_ABS_DIFF =", report["max_state_abs_diff"])
    print("MAX_GEOMETRY_ABS_DIFF =", report["max_geometry_abs_diff"])
    print("MAX_PE_ABS_DIFF =", report["max_pe_abs_diff"])
    print("SCIENTIFIC_CONCLUSION = NONE")


if __name__ == "__main__":
    main()
