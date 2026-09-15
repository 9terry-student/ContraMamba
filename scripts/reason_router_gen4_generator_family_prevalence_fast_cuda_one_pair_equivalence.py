from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import reason_router_gen4_generator_family_prevalence_tokenizer_anchor_eligibility as eligibility
from scripts import reason_router_gen4_xg1_fast_cuda_one_pair_equivalence as xg1_eq
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"
ELIGIBILITY_FREEZE_COMMIT = "0af44566eaadc324a6aaf5cec19f3972c8e371ef"
ELIGIBILITY_EXECUTION_HEAD = "8df2a5e2cb72f8f4eed33c4996dfba594a1f9687"
BACKEND_EQUIVALENCE_FREEZE_COMMIT = xg1_eq.BACKEND_EQUIVALENCE_FREEZE_COMMIT

ELIGIBILITY_GATE_PATH = (
    "scripts/reason_router_gen4_generator_family_prevalence_tokenizer_anchor_eligibility.py"
)
ELIGIBILITY_GATE_BLOB = "29a97f343f372718ca56436c505893136cb505b6"
COHORT_BUILDER_PATH = "scripts/build_reason_router_gen4_generator_family_prevalence_cohorts.py"
COHORT_BUILDER_BLOB = "4505acc99db0627733592694da290a007d281421"

ELIGIBILITY_ROOT = Path(
    "reports/reason_router_gen4_generator_family_prevalence_tokenizer_anchor_eligibility_8df2a5e_r1"
)

FAMILY_CONFIG: dict[str, dict[str, str]] = {
    "xg2": {
        "pair_id": "xg2_fact_001",
        "anchor_sha256": "c1177819713bb850cb6cfa93fb76ff06b9feddbec9cac622fc015f969844135a",
        "summary_sha256": "ddc6158339345a9f37ba4a0edbff0b8040e51d39bdffe92c3a409a286ec08843",
        "source_sha256": "0d24fe924a9e30ba7341b515e6cd2bf1164eff441462451317ca5ec3beb33c5e",
        "rows_sha256": "c96ce74bb89dbd374386c27f3a7daa10b5c6630c22711074e0b9a44d59ab5cfa",
        "structural_manifest_sha256": "29afcee82a4f0251870b6a6ef8c8b626bc65acfc1a21b98b4c019a7b20de5704",
    },
    "xg4": {
        "pair_id": "xg4_fact_001",
        "anchor_sha256": "5d1d2c1921c464d82229e13189ff92dbec31c06d86f1948bd4fdba6d4dad06ef",
        "summary_sha256": "b7d9679bc68e79892dd4e6e6b4eb69af570aba6584493ff40b16a43db1312a55",
        "source_sha256": "23f81cb93a5deb7a18c98a6a2f4718b34f4bfd1713bd1fb294e037a714f71fb0",
        "rows_sha256": "9146404eefdbcf26e25b81c65496cbcc459d2eb85c90b5da44db35307de9a347",
        "structural_manifest_sha256": "bc7091577242a7610f5c7c87d8c9b1b8a4bb17ccb047eed48c48bddb9ebadc2a",
    },
}

SOURCE_PAIR_COUNT = 300
FORWARDS_PER_BACKEND = 4
TOTAL_MODEL_FORWARDS = 8
BASELINE_LABELS = (
    "baseline_tp",
    "baseline_tm",
    "baseline_rp",
    "baseline_rm",
)

STATE_ATOL = xg1_eq.STATE_ATOL
STATE_RTOL = xg1_eq.STATE_RTOL
GEOMETRY_ATOL = xg1_eq.GEOMETRY_ATOL
GEOMETRY_RTOL = xg1_eq.GEOMETRY_RTOL

core = xg1_eq.base_eq.core
parent = xg1_eq.parent
transport_runtime = xg1_eq.transport_runtime
backend = xg1_eq.backend
extraction = xg1_eq.extraction
measurement = xg1_eq.measurement

REPORT_SCHEMA = "gen4-generator-family-prevalence-baseline-fast-cuda-one-pair-equivalence-v1"
RESULT_PASS = "PASS_GENERATOR_FAMILY_PREVALENCE_BASELINE_FAST_CUDA_ONE_PAIR_EQUIVALENCE"

EXACT_FIELDS = (
    "schema_version",
    "family_key",
    "source_pair_id",
    "source_block",
    "target_residual_layer",
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
    "target_plus_geometry_token",
    "target_minus_geometry_token",
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
)


class PrevalenceBaselineEquivalenceError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PrevalenceBaselineEquivalenceError(message)


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrevalenceBaselineEquivalenceError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str, root: Path = ROOT) -> None:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")

    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git(root, "status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for ancestor, label in (
        (ELIGIBILITY_FREEZE_COMMIT, "ELIGIBILITY_FREEZE"),
        (BACKEND_EQUIVALENCE_FREEZE_COMMIT, "BACKEND_EQUIVALENCE_FREEZE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, expected_head],
            cwd=root,
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
            *xg1_eq.BACKEND_FROZEN_PATHS,
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "FROZEN_BACKEND_DEPENDENCY_DRIFT")

    for path, expected, label in (
        (ELIGIBILITY_GATE_PATH, ELIGIBILITY_GATE_BLOB, "ELIGIBILITY_GATE"),
        (COHORT_BUILDER_PATH, COHORT_BUILDER_BLOB, "COHORT_BUILDER"),
    ):
        observed = git(root, "rev-parse", f"HEAD:{path}")
        require(observed == expected, f"{label}_BLOB_DRIFT:{observed}")


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


def _pair_order(
    family: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        pair = str(row["source_pair_id"])
        if pair not in seen:
            seen.add(pair)
            order.append(pair)

    expected = tuple(
        f"{family}_fact_{index:03d}"
        for index in range(1, SOURCE_PAIR_COUNT + 1)
    )
    require(tuple(order) == expected, f"PAIR_ORDER:{family}")
    return tuple(order)


def _artifact_paths(family: str) -> tuple[Path, Path]:
    require(family in FAMILY_CONFIG, f"UNSUPPORTED_FAMILY:{family}")
    base = ELIGIBILITY_ROOT / family
    return (
        base / "anchor_manifest.jsonl",
        base / "eligibility_summary.json",
    )


def load_frozen_anchor_manifest(
    family: str,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    require(family in FAMILY_CONFIG, f"UNSUPPORTED_FAMILY:{family}")
    cfg = FAMILY_CONFIG[family]
    anchor_rel, summary_rel = _artifact_paths(family)
    anchor_path = root / anchor_rel
    summary_path = root / summary_rel

    require(anchor_path.is_file(), f"ELIGIBILITY_ANCHOR_MANIFEST_MISSING:{family}")
    require(summary_path.is_file(), f"ELIGIBILITY_SUMMARY_MISSING:{family}")
    require(
        xg1_eq.base_eq.sha256_file(anchor_path) == cfg["anchor_sha256"],
        f"ELIGIBILITY_ANCHOR_MANIFEST_SHA256:{family}",
    )
    require(
        xg1_eq.base_eq.sha256_file(summary_path) == cfg["summary_sha256"],
        f"ELIGIBILITY_SUMMARY_SHA256:{family}",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    require(summary.get("family_key") == family, f"ELIGIBILITY_FAMILY:{family}")
    require(
        summary.get("primary_complete_pair_prefix_feasibility")
        == "PASS_300_OF_300",
        f"ELIGIBILITY_SUMMARY_NOT_PASS:{family}",
    )
    require(
        summary.get("complete_source_pair_count") == SOURCE_PAIR_COUNT,
        f"ELIGIBILITY_PAIR_COUNT:{family}",
    )
    require(
        summary.get("required_anchor_row_count") == 1800,
        f"ELIGIBILITY_ANCHOR_COUNT:{family}",
    )
    require(
        summary.get("model_forward_count") == 0,
        f"ELIGIBILITY_MODEL_FORWARD_BOUNDARY:{family}",
    )
    require(
        summary.get("checkpoint_load_count") == 0,
        f"ELIGIBILITY_CHECKPOINT_BOUNDARY:{family}",
    )
    require(
        summary.get("gpu_used") is False,
        f"ELIGIBILITY_GPU_BOUNDARY:{family}",
    )
    require(
        summary.get("scientific_outcomes_observed") is False,
        f"ELIGIBILITY_SCIENCE_BOUNDARY:{family}",
    )
    require(
        summary.get("tokenizer_executed") is True,
        f"ELIGIBILITY_TOKENIZER_BOUNDARY:{family}",
    )
    require(
        summary.get("anchor_manifest_sha256") == cfg["anchor_sha256"],
        f"ELIGIBILITY_SUMMARY_MANIFEST_IDENTITY:{family}",
    )
    require(
        summary.get("provenance", {}).get("head") == ELIGIBILITY_EXECUTION_HEAD,
        f"ELIGIBILITY_EXECUTION_HEAD:{family}",
    )
    frozen = summary.get("frozen_input", {})
    require(
        frozen.get("source_facts_sha256") == cfg["source_sha256"],
        f"ELIGIBILITY_SOURCE_SHA:{family}",
    )
    require(
        frozen.get("rows_sha256") == cfg["rows_sha256"],
        f"ELIGIBILITY_ROWS_SHA:{family}",
    )
    require(
        frozen.get("structural_manifest_sha256")
        == cfg["structural_manifest_sha256"],
        f"ELIGIBILITY_STRUCTURAL_MANIFEST_SHA:{family}",
    )

    rows = _read_jsonl(anchor_path)
    require(len(rows) == 1800, f"ANCHOR_MANIFEST_ROW_COUNT:{family}")
    require(
        Counter(row["anchor_name"] for row in rows)
        == Counter({"A_IDENTITY": 1200, "A_NAME": 600}),
        f"ANCHOR_NAME_COUNTS:{family}",
    )

    for index, row in enumerate(rows):
        require(
            row.get("schema_version") == eligibility.ANCHOR_SCHEMA,
            f"ANCHOR_SCHEMA:{family}:{index}",
        )
        require(row.get("family_key") == family, f"ANCHOR_FAMILY:{family}:{index}")
        require(
            row.get("post4_eligible") is True,
            f"ANCHOR_NOT_ELIGIBLE:{family}:{index}",
        )
        require(
            row.get("exclusion_code") is None,
            f"ANCHOR_EXCLUSION:{family}:{index}",
        )
        require(
            type(row.get("absolute_anchor_token_index")) is int,
            f"ANCHOR_INDEX_TYPE:{family}:{index}",
        )
        require(
            type(row.get("anchor_evidence_token_index")) is int,
            f"EVIDENCE_ANCHOR_TYPE:{family}:{index}",
        )
        require(
            type(row.get("terminal_index")) is int,
            f"TERMINAL_INDEX_TYPE:{family}:{index}",
        )

    lookup = {
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["anchor_name"]),
        ): row
        for row in rows
    }
    require(len(lookup) == 1800, f"ANCHOR_EVENT_KEY_COUNT:{family}")
    for pair_index in range(1, SOURCE_PAIR_COUNT + 1):
        pair = f"{family}_fact_{pair_index:03d}"
        for cell in eligibility.TARGET_IDENTITY_NAME_CELLS:
            identity = lookup[(pair, cell, "A_IDENTITY")]
            name = lookup[(pair, cell, "A_NAME")]
            require(
                identity["absolute_anchor_token_index"]
                == name["absolute_anchor_token_index"],
                f"TARGET_IDENTITY_NAME_MISMATCH:{pair}:{cell}",
            )
    return rows


def validate_family_population(
    family: str,
    rows: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    normalized = adapter.validate_gen4_rows(
        rows,
        require_canonical_shape=True,
    )
    pairs = _pair_order(family, normalized)

    require(
        list(encoded["row_id"])
        == [str(row["row_id"]) for row in normalized],
        f"ENCODED_ROW_ORDER:{family}",
    )
    require(
        list(encoded["source_pair_id"])
        == [str(row["source_pair_id"]) for row in normalized],
        f"ENCODED_PAIR_ORDER:{family}",
    )
    require(
        list(encoded["contrast_cell_id"])
        == [str(row["contrast_cell_id"]) for row in normalized],
        f"ENCODED_CELL_ORDER:{family}",
    )

    input_ids = encoded["input_ids"]
    attention = encoded["attention_mask"]
    claim_mask = encoded["claim_mask"]
    evidence_mask = encoded["evidence_mask"]

    require(torch.is_tensor(input_ids), f"INPUT_IDS_TENSOR:{family}")
    require(
        tuple(input_ids.shape) == (1800, adapter.MAX_LENGTH),
        f"INPUT_IDS_SHAPE:{family}",
    )

    row_index = {
        str(row["row_id"]): index
        for index, row in enumerate(normalized)
    }
    require(len(row_index) == 1800, f"ROW_ID_CARDINALITY:{family}")

    events = parent.event_lookup(event_rows)
    parent.validate_transport_event_plan(pairs, events)

    for event in event_rows:
        row_id = str(event["row_id"])
        require(row_id in row_index, f"EVENT_ROW_ID:{family}:{row_id}")
        index = row_index[row_id]

        anchor = int(event["absolute_anchor_token_index"])
        evidence_anchor = int(event["anchor_evidence_token_index"])
        terminal = int(event["terminal_index"])
        claim_count = int(claim_mask[index].sum().item())
        evidence_count = int(evidence_mask[index].sum().item())

        require(
            anchor == claim_count + 1 + evidence_anchor,
            f"ANCHOR_COORDINATE:{family}:{row_id}",
        )
        require(
            terminal == claim_count + evidence_count,
            f"TERMINAL_COORDINATE:{family}:{row_id}",
        )
        require(
            anchor + 4 <= terminal - 1,
            f"POST4_RULE:{family}:{row_id}",
        )
        require(
            int(attention[index].sum().item()) == terminal + 1,
            f"ATTENTION_TERMINAL:{family}:{row_id}",
        )

    return pairs


def load_family_inputs(
    family: str,
    tokenizer_snapshot: str | Path | None,
    root: Path = ROOT,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    require(family in FAMILY_CONFIG, f"UNSUPPORTED_FAMILY:{family}")
    facts, rows, _manifest = eligibility.load_family(family, root)
    require(len(facts) == SOURCE_PAIR_COUNT, f"FACT_COUNT:{family}")

    tokenizer, _tokenizer_provenance = (
        xg1_eq.eligibility.load_canonical_analysis_tokenizer(
            tokenizer_snapshot
        )
    )
    encoded = adapter.encode_gen4_rows(rows, tokenizer)
    event_rows = load_frozen_anchor_manifest(family, root)
    validate_family_population(family, rows, encoded, event_rows)
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


def _cells() -> dict[str, str]:
    return {
        "tp": core.TARGET_PLUS_CELL,
        "tm": core.TARGET_MINUS_CELL,
        "rp": core.REFERENCE_PLUS_CELL,
        "rm": core.REFERENCE_MINUS_CELL,
    }


def _anchors_for_pair(
    pair: str,
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, int]:
    cells = _cells()
    return {
        role: int(
            events[(pair, cell, core.ANCHOR_NAME)][
                "absolute_anchor_token_index"
            ]
        )
        for role, cell in cells.items()
    }


def run_baseline_geometry_pair(
    family: str,
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
    cells = _cells()
    anchors = _anchors_for_pair(pair, events)

    baseline: dict[str, Any] = {}
    for role in ("tp", "tm", "rp", "rm"):
        baseline[role] = parent.capture_branch(
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
        float(reference_geometry["C"])
        - float(target_geometry["C"])
    )
    require(
        math.isfinite(alignment_shift_abs),
        "NONFINITE_ALIGNMENT_SHIFT",
    )

    return {
        "schema_version": "gen4-generator-family-prevalence-baseline-equivalence-item-v1",
        "family_key": family,
        "source_pair_id": pair,
        "source_block": core.SOURCE_BLOCK,
        "target_residual_layer": core.TARGET_RESIDUAL_LAYER,
        "relative_coordinate": core.TARGET_OFFSET,
        "target_plus_cell": cells["tp"],
        "target_minus_cell": cells["tm"],
        "reference_plus_cell": cells["rp"],
        "reference_minus_cell": cells["rm"],
        "anchor_name": core.ANCHOR_NAME,
        "target_plus_anchor": anchors["tp"],
        "target_minus_anchor": anchors["tm"],
        "reference_plus_anchor": anchors["rp"],
        "reference_minus_anchor": anchors["rm"],
        "target_plus_geometry_token": anchors["tp"] + core.TARGET_OFFSET,
        "target_minus_geometry_token": anchors["tm"] + core.TARGET_OFFSET,
        "reference_plus_geometry_token": anchors["rp"] + core.TARGET_OFFSET,
        "reference_minus_geometry_token": anchors["rm"] + core.TARGET_OFFSET,
        "target_A": float(target_geometry["A"]),
        "target_B": float(target_geometry["B"]),
        "target_C": float(target_geometry["C"]),
        "reference_A": float(reference_geometry["A"]),
        "reference_B": float(reference_geometry["B"]),
        "reference_C": float(reference_geometry["C"]),
        "alignment_shift_abs": alignment_shift_abs,
    }


def _five_state_window(
    states: Sequence[Any] | None,
    anchor: int,
) -> list[np.ndarray] | None:
    if states is None:
        return None
    require(anchor + 4 < len(states), "STATE_WINDOW_RANGE")
    return [
        np.ascontiguousarray(
            np.asarray(states[t], dtype=np.float32)
        ).copy()
        for t in range(anchor, anchor + 5)
    ]


def _logging_capture(
    base_capture: Any,
    records: list[dict[str, Any]],
):
    counter = {"value": 0}

    def wrapped(*args: Any, **kwargs: Any):
        index = counter["value"]
        require(index < len(BASELINE_LABELS), "CAPTURE_CALL_OVERFLOW")
        counter["value"] += 1

        result = base_capture(*args, **kwargs)
        anchor = int(result["anchor"])
        records.append(
            {
                "label": BASELINE_LABELS[index],
                "anchor": anchor,
                "target_abs": int(result["target_abs"]),
                "states": _five_state_window(
                    result["states"],
                    anchor,
                ),
            }
        )
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
    require(
        math.isfinite(a) and math.isfinite(b),
        f"NONFINITE:{label}",
    )
    diff = abs(a - b)
    limit = atol + rtol * abs(b)
    require(
        diff <= limit,
        f"EQUIVALENCE_FAILURE:{label}:{diff}:{limit}",
    )
    return diff


def compare_baseline_pair(
    cpu_item: Mapping[str, Any],
    gpu_item: Mapping[str, Any],
    cpu_records: Sequence[Mapping[str, Any]],
    gpu_records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    require(set(cpu_item) == set(gpu_item), "ITEM_SCHEMA")

    for field in EXACT_FIELDS:
        require(
            cpu_item[field] == gpu_item[field],
            f"EXACT_FIELD:{field}",
        )

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

    require(
        len(cpu_records) == len(BASELINE_LABELS)
        and len(gpu_records) == len(BASELINE_LABELS),
        "CAPTURE_RECORD_COUNT",
    )

    max_state_abs = 0.0
    compared_windows = 0
    for cpu_record, gpu_record in zip(
        cpu_records,
        gpu_records,
        strict=True,
    ):
        require(
            cpu_record["label"] == gpu_record["label"],
            "CAPTURE_LABEL",
        )
        require(
            cpu_record["anchor"] == gpu_record["anchor"],
            "CAPTURE_ANCHOR",
        )
        require(
            cpu_record["target_abs"] == gpu_record["target_abs"],
            "CAPTURE_TARGET",
        )

        cpu_states = cpu_record["states"]
        gpu_states = gpu_record["states"]
        require(
            cpu_states is not None and gpu_states is not None,
            "CAPTURE_STATE_PRESENCE",
        )
        require(
            len(cpu_states) == 5 and len(gpu_states) == 5,
            "STATE_WINDOW_WIDTH",
        )
        compared_windows += 1

        for offset, (cpu_state, gpu_state) in enumerate(
            zip(cpu_states, gpu_states, strict=True)
        ):
            require(
                cpu_state.shape == gpu_state.shape,
                "STATE_SHAPE_MISMATCH",
            )
            cpu64 = np.asarray(cpu_state, dtype=np.float64)
            gpu64 = np.asarray(gpu_state, dtype=np.float64)
            max_abs = float(np.max(np.abs(gpu64 - cpu64)))
            scale = float(np.max(np.abs(cpu64)))
            limit = STATE_ATOL + STATE_RTOL * scale
            require(
                max_abs <= limit,
                (
                    "STATE_EQUIVALENCE_FAILURE:"
                    f"{cpu_record['label']}:{offset}:"
                    f"{max_abs}:{limit}"
                ),
            )
            max_state_abs = max(max_state_abs, max_abs)

    require(
        compared_windows == len(BASELINE_LABELS),
        "STATE_WINDOW_COUNT",
    )

    return {
        "max_state_abs_diff": max_state_abs,
        "max_geometry_abs_diff": max_geometry,
    }


def run_one_pair(
    *,
    family: str,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output: Path,
) -> dict[str, Any]:
    require(family in FAMILY_CONFIG, f"UNSUPPORTED_FAMILY:{family}")
    authenticate_repo(expected_head)
    backend.runtime_gate()
    require(not output.exists(), "OUTPUT_COLLISION")

    cfg = FAMILY_CONFIG[family]
    pair = cfg["pair_id"]

    with backend.parent_runtime_rebind():
        rows, encoded, event_rows = load_family_inputs(
            family,
            tokenizer_snapshot,
        )
        pairs = _pair_order(family, rows)
        require(pairs[0] == pair, f"PAIR_SELECTION:{family}")

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
        cpu_ctx = transport_runtime.validate_runtime_components(
            cpu_model
        )

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
        gpu_ctx = transport_runtime.validate_runtime_components(
            gpu_model
        )

        original_capture = parent.capture_branch

        cpu_records: list[dict[str, Any]] = []
        parent.capture_branch = _logging_capture(
            original_capture,
            cpu_records,
        )
        cpu_budget = parent.ForwardBudget(FORWARDS_PER_BACKEND)
        try:
            cpu_item = run_baseline_geometry_pair(
                family,
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
        parent.capture_branch = _logging_capture(
            fast_capture,
            gpu_records,
        )
        gpu_budget = parent.ForwardBudget(FORWARDS_PER_BACKEND)
        try:
            gpu_item = run_baseline_geometry_pair(
                family,
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

    comparison = compare_baseline_pair(
        cpu_item,
        gpu_item,
        cpu_records,
        gpu_records,
    )

    anchor_rel, summary_rel = _artifact_paths(family)
    report = {
        "schema_version": REPORT_SCHEMA,
        "result": RESULT_PASS,
        "family_key": family,
        "execution_head": expected_head,
        "eligibility_freeze_commit": ELIGIBILITY_FREEZE_COMMIT,
        "backend_equivalence_freeze_commit":
            BACKEND_EQUIVALENCE_FREEZE_COMMIT,
        "source_pair_id": pair,
        "source_pair_selection":
            "first_fixed_family_pair_outcome_blind",
        "cpu_model_forward_count": FORWARDS_PER_BACKEND,
        "gpu_model_forward_count": FORWARDS_PER_BACKEND,
        "total_model_forward_count": TOTAL_MODEL_FORWARDS,
        "scientific_budget_forward_count": 0,
        "scientific_conclusion": None,
        "baseline_only": True,
        "alignment_intervention_executed": False,
        "magnitude_intervention_executed": False,
        "response_endpoints_computed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "raw_vectors_persisted": False,
        "geometry_values_persisted": False,
        "alignment_shift_value_persisted": False,
        "response_values_persisted": False,
        "family_source_facts_sha256": cfg["source_sha256"],
        "family_rows_sha256": cfg["rows_sha256"],
        "family_structural_manifest_sha256":
            cfg["structural_manifest_sha256"],
        "eligibility_anchor_manifest_path": anchor_rel.as_posix(),
        "eligibility_anchor_manifest_sha256":
            cfg["anchor_sha256"],
        "eligibility_summary_path": summary_rel.as_posix(),
        "eligibility_summary_sha256": cfg["summary_sha256"],
        "representative_checkpoint_sha256": cpu_checkpoint_sha,
        "kernels_version": backend.KERNELS_VERSION,
        "mamba_revision": backend.MAMBA_REV,
        "mamba_binary_sha256": backend.MAMBA_BINARY_SHA256,
        "causal_conv_revision": backend.CONV_REV,
        "causal_conv_binary_sha256":
            backend.CONV_BINARY_SHA256,
        "build_variant": backend.BUILD_VARIANT,
        "python_version": backend.EXPECTED_RUNTIME["python"],
        "numpy_version": backend.EXPECTED_RUNTIME["numpy"],
        "torch_version": backend.EXPECTED_RUNTIME["torch"],
        "transformers_version":
            backend.EXPECTED_RUNTIME["transformers"],
        "cuda_runtime": backend.EXPECTED_CUDA_RUNTIME,
        "cuda_device": backend.EXPECTED_DEVICE_NAME,
        "cuda_capability":
            list(backend.EXPECTED_CAPABILITY),
        "state_atol": STATE_ATOL,
        "state_rtol": STATE_RTOL,
        "geometry_atol": GEOMETRY_ATOL,
        "geometry_rtol": GEOMETRY_RTOL,
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
            "Baseline-only CPU-slow versus fast-CUDA equivalence gate "
            "for frozen Gen4 generator-family prevalence cohorts. "
            "Runs exactly four baseline forwards per backend and never "
            "executes alignment or magnitude intervention."
        )
    )
    parser.add_argument(
        "--family",
        choices=tuple(FAMILY_CONFIG),
        required=True,
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_one_pair(
        family=args.family,
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output=args.output,
    )

    print("FAMILY =", report["family_key"].upper())
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
    print(
        "MAX_STATE_ABS_DIFF =",
        report["max_state_abs_diff"],
    )
    print(
        "MAX_GEOMETRY_ABS_DIFF =",
        report["max_geometry_abs_diff"],
    )
    print("ALIGNMENT_INTERVENTION_EXECUTED = FALSE")
    print("RESPONSE_ENDPOINTS_COMPUTED = FALSE")
    print("SCIENTIFIC_CONCLUSION = NONE")


if __name__ == "__main__":
    main()
