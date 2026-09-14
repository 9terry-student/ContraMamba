from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import (
    reason_router_gen4_large_correction_prospective_fast_cuda_one_pair_equivalence
    as base_eq,
)
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as eligibility
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"
ELIGIBILITY_FREEZE_COMMIT = "ee7c2c10a0cbb4930b78eb0047ec0603b63e0d41"
BACKEND_EQUIVALENCE_FREEZE_COMMIT = "34a958fbc52300f0807b211b1d97f68f6fea7339"

ELIGIBILITY_DIR = Path(
    "reports/reason_router_gen4_xg1_tokenizer_anchor_eligibility_4a926a1"
)
ELIGIBILITY_ANCHOR_MANIFEST = ELIGIBILITY_DIR / "anchor_manifest.jsonl"
ELIGIBILITY_SUMMARY = ELIGIBILITY_DIR / "eligibility_summary.json"
ELIGIBILITY_ANCHOR_MANIFEST_SHA256 = (
    "ebce31f5f93f33da8eb238cdc4d66510ea7ade03bdbedac88ec8d6b82f9e739e"
)
ELIGIBILITY_SUMMARY_SHA256 = (
    "833d32e9bdeca2ba28e5b192b2211487242cb1e3b898bb8dae384ba678a1b407"
)

XG1_PAIR_ID = "xg1_fact_001"
SOURCE_PAIR_COUNT = 300
FORWARDS_PER_BACKEND = base_eq.FORWARDS_PER_BACKEND
TOTAL_MODEL_FORWARDS = base_eq.TOTAL_MODEL_FORWARDS

STATE_ATOL = base_eq.STATE_ATOL
STATE_RTOL = base_eq.STATE_RTOL
GEOMETRY_ATOL = base_eq.GEOMETRY_ATOL
GEOMETRY_RTOL = base_eq.GEOMETRY_RTOL
PE_ATOL = base_eq.PE_ATOL

BRANCH_LABELS = base_eq.BRANCH_LABELS
EXACT_FIELDS = base_eq.EXACT_FIELDS
GEOMETRY_FIELDS = base_eq.GEOMETRY_FIELDS
PE_FIELDS = base_eq.PE_FIELDS

backend = base_eq.backend
parent = base_eq.parent
transport_runtime = base_eq.transport_runtime
extraction = base_eq.extraction
measurement = base_eq.measurement

BACKEND_FROZEN_PATHS = (
    "scripts/reason_router_gen4_large_correction_prospective_fast_cuda_one_pair_equivalence.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_core.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runner.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runtime.py",
    "scripts/reason_router_gen4_k_fast_cuda_one_pair_equivalence.py",
    "scripts/reason_router_gen4_large_correction_prospective_tokenizer_anchor_eligibility.py",
    "scripts/reason_router_gen4_native_mamba_state_extraction.py",
    "scripts/reason_router_gen4_native_mamba_state_measurement.py",
    "scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py",
)

REPORT_SCHEMA = "gen4-k-xg1-fast-cuda-one-pair-equivalence-v1"


class XG1EquivalenceError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise XG1EquivalenceError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise XG1EquivalenceError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    # cm kaggle checks out the exact pinned commit detached.
    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for ancestor, label in (
        (ELIGIBILITY_FREEZE_COMMIT, "ELIGIBILITY_FREEZE"),
        (BACKEND_EQUIVALENCE_FREEZE_COMMIT, "BACKEND_EQUIVALENCE_FREEZE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, expected_head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_NOT_ANCESTOR")

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            BACKEND_EQUIVALENCE_FREEZE_COMMIT,
            expected_head,
            "--",
            *BACKEND_FROZEN_PATHS,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "FROZEN_BACKEND_DEPENDENCY_DRIFT")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
            f"JSONL_OBJECT_REQUIRED:{path}:{line_no}",
        )
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

    expected = tuple(
        f"xg1_fact_{index:03d}"
        for index in range(1, SOURCE_PAIR_COUNT + 1)
    )
    require(tuple(order) == expected, "XG1_PAIR_ORDER")
    return tuple(order)


def load_frozen_anchor_manifest() -> list[dict[str, Any]]:
    path = ROOT / ELIGIBILITY_ANCHOR_MANIFEST
    summary_path = ROOT / ELIGIBILITY_SUMMARY

    require(path.is_file(), "ELIGIBILITY_ANCHOR_MANIFEST_MISSING")
    require(summary_path.is_file(), "ELIGIBILITY_SUMMARY_MISSING")
    require(
        base_eq.sha256_file(path) == ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
        "ELIGIBILITY_ANCHOR_MANIFEST_SHA256",
    )
    require(
        base_eq.sha256_file(summary_path) == ELIGIBILITY_SUMMARY_SHA256,
        "ELIGIBILITY_SUMMARY_SHA256",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    require(
        summary.get("primary_complete_pair_prefix_feasibility")
        == "PASS_300_OF_300",
        "ELIGIBILITY_SUMMARY_NOT_PASS",
    )
    require(
        summary.get("complete_source_pair_count") == SOURCE_PAIR_COUNT,
        "ELIGIBILITY_PAIR_COUNT",
    )
    require(
        summary.get("required_anchor_row_count")
        == eligibility.EXPECTED_ANCHOR_ROWS,
        "ELIGIBILITY_ANCHOR_COUNT",
    )
    require(
        summary.get("model_forward_count") == 0,
        "ELIGIBILITY_MODEL_FORWARD_BOUNDARY",
    )
    require(
        summary.get("checkpoint_load_count") == 0,
        "ELIGIBILITY_CHECKPOINT_BOUNDARY",
    )
    require(
        summary.get("gpu_used") is False,
        "ELIGIBILITY_GPU_BOUNDARY",
    )
    require(
        summary.get("scientific_outcomes_observed") is False,
        "ELIGIBILITY_SCIENCE_BOUNDARY",
    )
    require(
        summary.get("anchor_manifest_sha256")
        == ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
        "ELIGIBILITY_SUMMARY_MANIFEST_IDENTITY",
    )

    rows = _read_jsonl(path)
    require(
        len(rows) == eligibility.EXPECTED_ANCHOR_ROWS,
        "ANCHOR_MANIFEST_ROW_COUNT",
    )

    for index, row in enumerate(rows):
        require(
            row.get("schema_version") == eligibility.ANCHOR_MANIFEST_SCHEMA,
            f"ANCHOR_SCHEMA:{index}",
        )
        require(
            row.get("post4_eligible") is True,
            f"ANCHOR_NOT_ELIGIBLE:{index}",
        )
        require(
            row.get("exclusion_code") is None,
            f"ANCHOR_EXCLUSION:{index}",
        )
        require(
            type(row.get("absolute_anchor_token_index")) is int,
            f"ANCHOR_INDEX_TYPE:{index}",
        )
        require(
            type(row.get("terminal_index")) is int,
            f"TERMINAL_INDEX_TYPE:{index}",
        )

    require(
        Counter(row["anchor_name"] for row in rows)
        == Counter({"A_IDENTITY": 1200, "A_NAME": 600}),
        "ANCHOR_NAME_COUNTS",
    )
    return rows


def validate_xg1_population(
    rows: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    normalized = adapter.validate_gen4_rows(
        rows,
        require_canonical_shape=True,
    )
    pairs = _pair_order(normalized)

    require(
        list(encoded["row_id"])
        == [str(row["row_id"]) for row in normalized],
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
        tuple(input_ids.shape)
        == (eligibility.EXPECTED_ROW_COUNT, adapter.MAX_LENGTH),
        "INPUT_IDS_SHAPE",
    )

    row_index = {
        str(row["row_id"]): index
        for index, row in enumerate(normalized)
    }
    require(
        len(row_index) == eligibility.EXPECTED_ROW_COUNT,
        "ROW_ID_CARDINALITY",
    )

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

        require(
            anchor == claim_count + 1 + evidence_anchor,
            "ANCHOR_COORDINATE",
        )
        require(
            terminal == claim_count + evidence_count,
            "TERMINAL_COORDINATE",
        )
        require(anchor + 4 <= terminal - 1, "POST4_RULE")
        require(
            int(attention[index].sum().item()) == terminal + 1,
            "ATTENTION_TERMINAL",
        )
        require(
            int(claim_mask[index].sum().item()) == claim_count,
            "CLAIM_MASK_COUNT",
        )
        require(
            int(evidence_mask[index].sum().item()) == evidence_count,
            "EVIDENCE_MASK_COUNT",
        )

    return pairs


def load_xg1_inputs(
    tokenizer_snapshot: str | Path | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    facts, rows, _manifest = eligibility.load_frozen_xg1_inputs(ROOT)
    require(len(facts) == SOURCE_PAIR_COUNT, "XG1_FACT_COUNT")

    tokenizer, _tokenizer_provenance = (
        eligibility.load_canonical_analysis_tokenizer(
            tokenizer_snapshot
        )
    )
    encoded = adapter.encode_gen4_rows(rows, tokenizer)

    event_rows = load_frozen_anchor_manifest()
    validate_xg1_population(rows, encoded, event_rows)
    return rows, encoded, event_rows


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
        rows, encoded, event_rows = load_xg1_inputs(tokenizer_snapshot)
        pairs = _pair_order(rows)
        require(pairs[0] == XG1_PAIR_ID, "XG1_PAIR_SELECTION")
        pair = XG1_PAIR_ID

        events = parent.event_lookup(event_rows)
        parent.validate_transport_event_plan(pairs, events)
        row_index = parent.build_row_index(rows)
        trace_code, trace_line = (
            measurement._resolve_and_validate_runtime_binding()
        )

        cpu_model, cpu_checkpoint_sha = (
            parent.load_representative_model_external(
                model_snapshot=model_snapshot,
                checkpoint_path=checkpoint_path,
            )
        )
        require(
            cpu_checkpoint_sha
            == extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
            "CPU_CHECKPOINT_IDENTITY",
        )
        cpu_ctx = transport_runtime.validate_runtime_components(cpu_model)

        gpu_model, gpu_checkpoint_sha = (
            parent.load_representative_model_external(
                model_snapshot=model_snapshot,
                checkpoint_path=checkpoint_path,
            )
        )
        require(
            gpu_checkpoint_sha == cpu_checkpoint_sha,
            "GPU_CHECKPOINT_IDENTITY",
        )
        gpu_ctx = transport_runtime.validate_runtime_components(gpu_model)

        original_capture = parent.capture_branch

        cpu_records: list[dict[str, Any]] = []
        parent.capture_branch = base_eq._logging_capture(
            original_capture,
            cpu_records,
        )
        cpu_budget = parent.ForwardBudget(FORWARDS_PER_BACKEND)
        try:
            cpu_item = base_eq.run_prospective_pair(
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
            all(
                parameter.device.type == "cuda"
                for parameter in gpu_model.mamba.parameters()
            ),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = backend._make_fast_capture(kernels)
        gpu_records: list[dict[str, Any]] = []
        parent.capture_branch = base_eq._logging_capture(
            fast_capture,
            gpu_records,
        )
        gpu_budget = parent.ForwardBudget(FORWARDS_PER_BACKEND)
        try:
            gpu_item = base_eq.run_prospective_pair(
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

    comparison = base_eq.compare_pair(
        cpu_item,
        gpu_item,
        cpu_records,
        gpu_records,
    )

    report = {
        "schema_version": REPORT_SCHEMA,
        "result": "PASS_XG1_FAST_CUDA_ONE_PAIR_EQUIVALENCE",
        "execution_head": expected_head,
        "eligibility_freeze_commit": ELIGIBILITY_FREEZE_COMMIT,
        "backend_equivalence_freeze_commit":
            BACKEND_EQUIVALENCE_FREEZE_COMMIT,
        "source_pair_id": pair,
        "source_pair_selection": "first_fixed_xg1_pair_outcome_blind",
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
        "xg1_response_values_persisted": False,
        "xg1_source_facts_sha256":
            eligibility.EXPECTED_SOURCE_FACTS_SHA256,
        "xg1_rows_sha256": eligibility.EXPECTED_ROWS_SHA256,
        "xg1_structural_manifest_sha256":
            eligibility.EXPECTED_STRUCTURAL_MANIFEST_SHA256,
        "eligibility_anchor_manifest_sha256":
            ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
        "eligibility_summary_sha256": ELIGIBILITY_SUMMARY_SHA256,
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
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded CPU-slow versus fast-CUDA equivalence gate for the "
            "frozen Gen4-K XG1 cross-generator cohort. This is not the "
            "1800-forward XG1 scientific execution."
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
    print(
        "CPU_MODEL_FORWARD_COUNT =",
        report["cpu_model_forward_count"],
    )
    print(
        "GPU_MODEL_FORWARD_COUNT =",
        report["gpu_model_forward_count"],
    )
    print(
        "TOTAL_MODEL_FORWARD_COUNT =",
        report["total_model_forward_count"],
    )
    print("MAX_STATE_ABS_DIFF =", report["max_state_abs_diff"])
    print(
        "MAX_GEOMETRY_ABS_DIFF =",
        report["max_geometry_abs_diff"],
    )
    print("MAX_PE_ABS_DIFF =", report["max_pe_abs_diff"])


if __name__ == "__main__":
    main()
