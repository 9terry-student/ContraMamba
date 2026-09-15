from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen4_large_correction_prospective_fast_cuda_one_pair_equivalence as inherited
from scripts import reason_router_gen4_xg2_xg4_fresh_response_tokenizer_anchor_eligibility as eligibility
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as tokenizer_gate
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"

DESIGN_FREEZE_COMMIT = (
    "2074f52d39bf0fca6d63248016ce47c54bff5e06"
)
STRUCTURAL_FREEZE_COMMIT = (
    "4bda8dd4b46d56ffe9bb37cffc91f8506026ca69"
)
ELIGIBILITY_IMPLEMENTATION_COMMIT = (
    "18189ac9532250faec5d3c6642b70e5636b9a0b8"
)
ELIGIBILITY_FREEZE_COMMIT = (
    "309c82ebc9aa6da013633ee8171f73ac8a191ba7"
)

SOURCE_PAIR_COUNT = 300
FORWARDS_PER_BACKEND = 6
TOTAL_MODEL_FORWARDS = 12

STATE_ATOL = inherited.STATE_ATOL
STATE_RTOL = inherited.STATE_RTOL
GEOMETRY_ATOL = inherited.GEOMETRY_ATOL
GEOMETRY_RTOL = inherited.GEOMETRY_RTOL
PE_ATOL = inherited.PE_ATOL

require_inherited = inherited.require

ELIGIBILITY_ROOT = Path(
    "reports/"
    "reason_router_gen4_xg2_xg4_fresh_response_"
    "tokenizer_anchor_eligibility_18189ac_r1"
)

FAMILY_CONFIG: dict[str, dict[str, str]] = {
    "xg2": {
        "pair_id": "xg2_fact_301",
        "anchor_sha256":
            "c54d8e2cb440f684e7971043fafd9c65c6dfe28041ef1cf92b02e75665e8b217",
        "summary_sha256":
            "d05ee53f7a5044d02d7e22a19a76d710c1e1a316349e5434e09f9a3bbe1c5687",
        "source_sha256":
            "c02d2fea5a7f3c8b5243598ad5505bba3f276c099be7b8c428c06eb39ffc7141",
        "rows_sha256":
            "1ca4f1c79caf5719e8970bf889c75b0e59f8711e5c78a36f7578727020e23670",
        "structural_manifest_sha256":
            "7df94b0788a55c6a4927926f6b92ef7e90d66346c5342608f5716ddd517e47f9",
    },
    "xg4": {
        "pair_id": "xg4_fact_301",
        "anchor_sha256":
            "74240984ac319872c193a1eb5301ecb62ffd89e803bd68ade11c03d88436e509",
        "summary_sha256":
            "4d0cda9598d6de3b343019feb74f320ebd95ddf93d1eb557b5e0753aec160879",
        "source_sha256":
            "0666c9345505f993bce70d66b5b1f9b784edca782eecb13a750ecf827f14ba3a",
        "rows_sha256":
            "b407613cdcee15d193847130f64e6aa674e666f72d6d08a30b573005e5e9de7d",
        "structural_manifest_sha256":
            "ee8d9bc4ae0ca135c335cf9eaa6bd9025cd5354304d435f1e3bb703a8b11e80d",
    },
}

# Exact runtime/protocol surfaces inherited unchanged.
PINNED_BLOBS = {
    "scripts/reason_router_gen4_large_correction_prospective_fast_cuda_one_pair_equivalence.py":
        "582579f4c5de481bf6899ec3df900f5a56fb357c",
    "scripts/reason_router_gen4_k_fast_cuda_one_pair_equivalence.py":
        "4a8326c442b883510ba452f049c88f3163677fc5",
    "scripts/reason_router_gen4_k_directional_alignment_transport_core.py":
        "d98b2dcd3436433c04bb56ecc57dec4240abe820",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runner.py":
        "3677dd83950789e41417c3a1ffaf70b82d7003ad",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runtime.py":
        "989c4a8947560dcf35e9523373d09ba085a9431a",
    "scripts/reason_router_gen4_native_mamba_state_extraction.py":
        "7f51683ecbf60d63cdc0399d2b0e89f8d42dd0ab",
    "scripts/reason_router_gen4_native_mamba_state_measurement.py":
        "8c8d63cce182dfab66a632299e93cd5b25fc36af",
    "scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py":
        "00a81ce6ec4ada4d5c0bf36418347b222543d174",
    "scripts/reason_router_gen4_xg2_xg4_fresh_response_tokenizer_anchor_eligibility.py":
        "e90b6a06843700a452c2c42512552986c5dd8fae",
    "scripts/build_reason_router_gen4_xg2_xg4_fresh_response_holdouts.py":
        "911846cf0b3304caaab0399535bd9f3de7f8249e",
    "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py":
        "6c98ce022ca134e385db28851fd364dc6daff423",
}

REPORT_SCHEMA = (
    "gen4-xg2-xg4-fresh-response-full-fast-cuda-"
    "one-pair-equivalence-v1"
)
RESULT_PASS = (
    "PASS_XG2_XG4_FRESH_RESPONSE_FULL_FAST_CUDA_"
    "ONE_PAIR_EQUIVALENCE"
)


class FreshFullEquivalenceError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FreshFullEquivalenceError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshFullEquivalenceError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(
    expected_head: str,
    root: Path = ROOT,
) -> None:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")

    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH_MISMATCH:{branch}",
    )
    require(
        head == expected_head,
        f"HEAD_MISMATCH:{head}",
    )
    require(
        git(root, "status", "--porcelain") == "",
        "WORKTREE_NOT_CLEAN",
    )

    for ancestor, label in (
        (DESIGN_FREEZE_COMMIT, "DESIGN_FREEZE"),
        (STRUCTURAL_FREEZE_COMMIT, "STRUCTURAL_FREEZE"),
        (ELIGIBILITY_FREEZE_COMMIT, "ELIGIBILITY_FREEZE"),
    ):
        rc = subprocess.call(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                ancestor,
                expected_head,
            ],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(
            rc == 0,
            f"{label}_NOT_ANCESTOR",
        )

    for path, expected_blob in PINNED_BLOBS.items():
        observed = git(
            root,
            "rev-parse",
            f"HEAD:{path}",
        )
        require(
            observed == expected_blob,
            f"FROZEN_BLOB_DRIFT:{path}:{observed}",
        )


def _read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_no, line in enumerate(
        path.read_text(
            encoding="utf-8-sig"
        ).splitlines(),
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


def _artifact_paths(
    family: str,
) -> tuple[Path, Path]:
    require(
        family in FAMILY_CONFIG,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    base = ELIGIBILITY_ROOT / family

    return (
        base / "anchor_manifest.jsonl",
        base / "eligibility_summary.json",
    )


def load_frozen_anchor_manifest(
    family: str,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    require(
        family in FAMILY_CONFIG,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    cfg = FAMILY_CONFIG[family]

    anchor_rel, summary_rel = (
        _artifact_paths(family)
    )

    anchor_path = root / anchor_rel
    summary_path = root / summary_rel

    require(
        anchor_path.is_file(),
        f"ANCHOR_MANIFEST_MISSING:{family}",
    )
    require(
        summary_path.is_file(),
        f"ELIGIBILITY_SUMMARY_MISSING:{family}",
    )

    require(
        sha256_file(anchor_path)
        == cfg["anchor_sha256"],
        f"ANCHOR_MANIFEST_SHA256:{family}",
    )
    require(
        sha256_file(summary_path)
        == cfg["summary_sha256"],
        f"ELIGIBILITY_SUMMARY_SHA256:{family}",
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8-sig"
        )
    )

    require(
        summary.get("family_key") == family,
        f"ELIGIBILITY_FAMILY:{family}",
    )
    require(
        summary.get(
            "primary_complete_pair_prefix_feasibility"
        )
        == "PASS_300_OF_300",
        f"ELIGIBILITY_RESULT:{family}",
    )
    require(
        summary.get("complete_source_pair_count")
        == SOURCE_PAIR_COUNT,
        f"ELIGIBILITY_PAIR_COUNT:{family}",
    )
    require(
        summary.get("required_anchor_row_count")
        == 1800,
        f"ELIGIBILITY_ANCHOR_COUNT:{family}",
    )
    require(
        summary.get("model_forward_count") == 0,
        f"ELIGIBILITY_MODEL_FORWARD:{family}",
    )
    require(
        summary.get("checkpoint_load_count") == 0,
        f"ELIGIBILITY_CHECKPOINT_LOAD:{family}",
    )
    require(
        summary.get("gpu_used") is False,
        f"ELIGIBILITY_GPU:{family}",
    )
    require(
        summary.get("scientific_outcomes_observed")
        is False,
        f"ELIGIBILITY_SCIENCE:{family}",
    )
    require(
        summary.get("tokenizer_executed") is True,
        f"ELIGIBILITY_TOKENIZER:{family}",
    )
    require(
        summary.get(
            "anchor_manifest_sha256"
        )
        == cfg["anchor_sha256"],
        f"ELIGIBILITY_ANCHOR_IDENTITY:{family}",
    )
    require(
        summary.get(
            "provenance",
            {},
        ).get("head")
        == ELIGIBILITY_IMPLEMENTATION_COMMIT,
        f"ELIGIBILITY_EXECUTION_HEAD:{family}",
    )

    frozen = summary.get(
        "frozen_input",
        {},
    )

    require(
        frozen.get("source_facts_sha256")
        == cfg["source_sha256"],
        f"FRESH_SOURCE_SHA:{family}",
    )
    require(
        frozen.get("rows_sha256")
        == cfg["rows_sha256"],
        f"FRESH_ROWS_SHA:{family}",
    )
    require(
        frozen.get(
            "structural_manifest_sha256"
        )
        == cfg["structural_manifest_sha256"],
        f"FRESH_STRUCTURAL_MANIFEST_SHA:{family}",
    )
    require(
        frozen.get("pair_id_first")
        == f"{family}_fact_301",
        f"FRESH_FIRST_PAIR:{family}",
    )
    require(
        frozen.get("pair_id_last")
        == f"{family}_fact_600",
        f"FRESH_LAST_PAIR:{family}",
    )

    rows = _read_jsonl(
        anchor_path
    )

    require(
        len(rows) == 1800,
        f"ANCHOR_MANIFEST_ROW_COUNT:{family}",
    )

    require(
        Counter(
            row["anchor_name"]
            for row in rows
        )
        == Counter({
            "A_IDENTITY": 1200,
            "A_NAME": 600,
        }),
        f"ANCHOR_NAME_COUNTS:{family}",
    )

    for index, row in enumerate(rows):
        require(
            row.get("schema_version")
            == eligibility.ANCHOR_SCHEMA,
            f"ANCHOR_SCHEMA:{family}:{index}",
        )
        require(
            row.get("family_key") == family,
            f"ANCHOR_FAMILY:{family}:{index}",
        )
        require(
            row.get("post4_eligible") is True,
            f"ANCHOR_NOT_ELIGIBLE:{family}:{index}",
        )
        require(
            row.get("exclusion_code") is None,
            f"ANCHOR_EXCLUSION:{family}:{index}",
        )
        require(
            type(
                row.get(
                    "absolute_anchor_token_index"
                )
            ) is int,
            f"ANCHOR_INDEX_TYPE:{family}:{index}",
        )
        require(
            type(
                row.get(
                    "anchor_evidence_token_index"
                )
            ) is int,
            f"EVIDENCE_ANCHOR_TYPE:{family}:{index}",
        )
        require(
            type(
                row.get("terminal_index")
            ) is int,
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

    require(
        len(lookup) == 1800,
        f"ANCHOR_EVENT_KEY_COUNT:{family}",
    )

    for pair_index in range(
        301,
        601,
    ):
        pair = (
            f"{family}_fact_{pair_index:03d}"
        )

        for cell in (
            "C0_SHAM",
            "C2_NAME",
        ):
            identity = lookup[
                (
                    pair,
                    cell,
                    "A_IDENTITY",
                )
            ]
            name = lookup[
                (
                    pair,
                    cell,
                    "A_NAME",
                )
            ]

            require(
                identity[
                    "absolute_anchor_token_index"
                ]
                == name[
                    "absolute_anchor_token_index"
                ],
                (
                    "TARGET_IDENTITY_NAME_MISMATCH:"
                    f"{pair}:{cell}"
                ),
            )

    return rows


def _pair_order(
    family: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    order: list[str] = []
    seen: set[str] = set()

    for row in rows:
        pair = str(
            row["source_pair_id"]
        )

        if pair not in seen:
            seen.add(pair)
            order.append(pair)

    expected = tuple(
        f"{family}_fact_{index:03d}"
        for index in range(301, 601)
    )

    require(
        tuple(order) == expected,
        f"PAIR_ORDER:{family}",
    )

    return tuple(order)


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

    pairs = _pair_order(
        family,
        normalized,
    )

    require(
        list(encoded["row_id"])
        == [
            str(row["row_id"])
            for row in normalized
        ],
        f"ENCODED_ROW_ORDER:{family}",
    )
    require(
        list(encoded["source_pair_id"])
        == [
            str(row["source_pair_id"])
            for row in normalized
        ],
        f"ENCODED_PAIR_ORDER:{family}",
    )
    require(
        list(encoded["contrast_cell_id"])
        == [
            str(row["contrast_cell_id"])
            for row in normalized
        ],
        f"ENCODED_CELL_ORDER:{family}",
    )

    input_ids = encoded["input_ids"]
    attention = encoded[
        "attention_mask"
    ]
    claim_mask = encoded[
        "claim_mask"
    ]
    evidence_mask = encoded[
        "evidence_mask"
    ]

    require(
        torch.is_tensor(input_ids),
        f"INPUT_IDS_TENSOR:{family}",
    )
    require(
        tuple(input_ids.shape)
        == (1800, adapter.MAX_LENGTH),
        f"INPUT_IDS_SHAPE:{family}",
    )

    row_index = {
        str(row["row_id"]): index
        for index, row
        in enumerate(normalized)
    }

    require(
        len(row_index) == 1800,
        f"ROW_ID_CARDINALITY:{family}",
    )

    events = inherited.parent.event_lookup(
        event_rows
    )

    inherited.parent.validate_transport_event_plan(
        pairs,
        events,
    )

    for event in event_rows:
        row_id = str(
            event["row_id"]
        )

        require(
            row_id in row_index,
            f"EVENT_ROW_ID:{family}:{row_id}",
        )

        index = row_index[row_id]

        anchor = int(
            event[
                "absolute_anchor_token_index"
            ]
        )
        evidence_anchor = int(
            event[
                "anchor_evidence_token_index"
            ]
        )
        terminal = int(
            event["terminal_index"]
        )

        claim_count = int(
            claim_mask[index]
            .sum()
            .item()
        )
        evidence_count = int(
            evidence_mask[index]
            .sum()
            .item()
        )

        require(
            anchor
            == claim_count
            + 1
            + evidence_anchor,
            f"ANCHOR_COORDINATE:{family}:{row_id}",
        )
        require(
            terminal
            == claim_count
            + evidence_count,
            f"TERMINAL_COORDINATE:{family}:{row_id}",
        )
        require(
            anchor + 4
            <= terminal - 1,
            f"POST4_RULE:{family}:{row_id}",
        )
        require(
            int(
                attention[index]
                .sum()
                .item()
            )
            == terminal + 1,
            f"ATTENTION_TERMINAL:{family}:{row_id}",
        )

    return pairs


def load_family_inputs(
    family: str,
    tokenizer_snapshot:
        str | Path | None,
    root: Path = ROOT,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    require(
        family in FAMILY_CONFIG,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    facts, rows, _manifest = (
        eligibility.load_family(
            family,
            root,
        )
    )

    require(
        len(facts) == SOURCE_PAIR_COUNT,
        f"FACT_COUNT:{family}",
    )

    tokenizer, _provenance = (
        tokenizer_gate
        .load_canonical_analysis_tokenizer(
            tokenizer_snapshot
        )
    )

    encoded = (
        adapter.encode_gen4_rows(
            rows,
            tokenizer,
        )
    )

    event_rows = (
        load_frozen_anchor_manifest(
            family,
            root,
        )
    )

    validate_family_population(
        family,
        rows,
        encoded,
        event_rows,
    )

    return (
        rows,
        encoded,
        event_rows,
    )


def run_one_pair(
    *,
    family: str,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output: Path,
) -> dict[str, Any]:
    require(
        family in FAMILY_CONFIG,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    authenticate_repo(
        expected_head
    )

    inherited.backend.runtime_gate()

    require(
        not output.exists(),
        "OUTPUT_COLLISION",
    )

    cfg = FAMILY_CONFIG[family]
    pair = cfg["pair_id"]

    with inherited.backend.parent_runtime_rebind():
        rows, encoded, event_rows = (
            load_family_inputs(
                family,
                tokenizer_snapshot,
            )
        )

        pairs = _pair_order(
            family,
            rows,
        )

        require(
            pairs[0] == pair,
            f"PAIR_SELECTION:{family}",
        )

        events = (
            inherited.parent.event_lookup(
                event_rows
            )
        )

        inherited.parent.validate_transport_event_plan(
            pairs,
            events,
        )

        row_index = (
            inherited.parent
            .build_row_index(
                rows
            )
        )

        trace_code, trace_line = (
            inherited.measurement
            ._resolve_and_validate_runtime_binding()
        )

        cpu_model, cpu_checkpoint_sha = (
            inherited.parent
            .load_representative_model_external(
                model_snapshot=model_snapshot,
                checkpoint_path=checkpoint_path,
            )
        )

        require(
            cpu_checkpoint_sha
            == inherited.extraction
            .REPRESENTATIVE_CHECKPOINT_SHA256,
            "CPU_CHECKPOINT_IDENTITY",
        )

        cpu_ctx = (
            inherited.transport_runtime
            .validate_runtime_components(
                cpu_model
            )
        )

        gpu_model, gpu_checkpoint_sha = (
            inherited.parent
            .load_representative_model_external(
                model_snapshot=model_snapshot,
                checkpoint_path=checkpoint_path,
            )
        )

        require(
            gpu_checkpoint_sha
            == cpu_checkpoint_sha,
            "GPU_CHECKPOINT_IDENTITY",
        )

        gpu_ctx = (
            inherited.transport_runtime
            .validate_runtime_components(
                gpu_model
            )
        )

        original_capture = (
            inherited.parent.capture_branch
        )

        cpu_records: list[
            dict[str, Any]
        ] = []

        inherited.parent.capture_branch = (
            inherited._logging_capture(
                original_capture,
                cpu_records,
            )
        )

        cpu_budget = (
            inherited.parent.ForwardBudget(
                FORWARDS_PER_BACKEND
            )
        )

        try:
            cpu_item = (
                inherited.run_prospective_pair(
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
            )
        finally:
            inherited.parent.capture_branch = (
                original_capture
            )

        cpu_budget.assert_exact()

        kernels = (
            inherited.backend
            .load_exact_fast_kernels()
        )

        gpu_model.to(
            torch.device("cuda:0")
        )
        gpu_model.eval()

        require(
            all(
                parameter.device.type
                == "cuda"
                for parameter
                in gpu_model.mamba.parameters()
            ),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = (
            inherited.backend
            ._make_fast_capture(
                kernels
            )
        )

        gpu_records: list[
            dict[str, Any]
        ] = []

        inherited.parent.capture_branch = (
            inherited._logging_capture(
                fast_capture,
                gpu_records,
            )
        )

        gpu_budget = (
            inherited.parent.ForwardBudget(
                FORWARDS_PER_BACKEND
            )
        )

        try:
            gpu_item = (
                inherited.run_prospective_pair(
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
            )
        finally:
            inherited.parent.capture_branch = (
                original_capture
            )

        gpu_budget.assert_exact()

        torch.cuda.synchronize()

    comparison = inherited.compare_pair(
        cpu_item,
        gpu_item,
        cpu_records,
        gpu_records,
    )

    report = {
        "schema_version":
            REPORT_SCHEMA,
        "result":
            RESULT_PASS,
        "family_key":
            family,
        "execution_head":
            expected_head,
        "design_freeze_commit":
            DESIGN_FREEZE_COMMIT,
        "structural_freeze_commit":
            STRUCTURAL_FREEZE_COMMIT,
        "eligibility_implementation_commit":
            ELIGIBILITY_IMPLEMENTATION_COMMIT,
        "eligibility_freeze_commit":
            ELIGIBILITY_FREEZE_COMMIT,
        "source_pair_id":
            pair,
        "source_pair_selection":
            (
                "first_fixed_fresh_holdout_pair_"
                "outcome_blind"
            ),
        "cpu_model_forward_count":
            FORWARDS_PER_BACKEND,
        "gpu_model_forward_count":
            FORWARDS_PER_BACKEND,
        "total_model_forward_count":
            TOTAL_MODEL_FORWARDS,
        "scientific_budget_forward_count":
            0,
        "scientific_conclusion":
            None,
        "training_executed":
            False,
        "backward_executed":
            False,
        "task_heads_executed":
            False,
        "logits_read":
            False,
        "raw_vectors_persisted":
            False,
        "endpoint_values_persisted":
            False,
        "prospective_response_values_persisted":
            False,
        "fresh_source_sha256":
            cfg["source_sha256"],
        "fresh_rows_sha256":
            cfg["rows_sha256"],
        "fresh_structural_manifest_sha256":
            cfg[
                "structural_manifest_sha256"
            ],
        "eligibility_anchor_manifest_sha256":
            cfg["anchor_sha256"],
        "eligibility_summary_sha256":
            cfg["summary_sha256"],
        "representative_checkpoint_sha256":
            cpu_checkpoint_sha,
        "kernels_version":
            inherited.backend.KERNELS_VERSION,
        "mamba_revision":
            inherited.backend.MAMBA_REV,
        "mamba_binary_sha256":
            inherited.backend.MAMBA_BINARY_SHA256,
        "causal_conv_revision":
            inherited.backend.CONV_REV,
        "causal_conv_binary_sha256":
            inherited.backend.CONV_BINARY_SHA256,
        "build_variant":
            inherited.backend.BUILD_VARIANT,
        "python_version":
            inherited.backend
            .EXPECTED_RUNTIME["python"],
        "numpy_version":
            inherited.backend
            .EXPECTED_RUNTIME["numpy"],
        "torch_version":
            inherited.backend
            .EXPECTED_RUNTIME["torch"],
        "transformers_version":
            inherited.backend
            .EXPECTED_RUNTIME["transformers"],
        "cuda_runtime":
            inherited.backend
            .EXPECTED_CUDA_RUNTIME,
        "cuda_device":
            inherited.backend
            .EXPECTED_DEVICE_NAME,
        "cuda_capability":
            list(
                inherited.backend
                .EXPECTED_CAPABILITY
            ),
        "state_atol":
            STATE_ATOL,
        "state_rtol":
            STATE_RTOL,
        "geometry_atol":
            GEOMETRY_ATOL,
        "geometry_rtol":
            GEOMETRY_RTOL,
        "pe_atol":
            PE_ATOL,
        **comparison,
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "`n",
        encoding="utf-8",
        newline="`n",
    )

    return report


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded fresh XG2/XG4 pair-301 "
            "CPU6/GPU6 full-response equivalence. "
            "Diagnostic only; scientific budget is zero."
        )
    )

    parser.add_argument(
        "--family",
        choices=tuple(FAMILY_CONFIG),
        required=True,
    )
    parser.add_argument(
        "--expected-head",
        required=True,
    )
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


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)

    report = run_one_pair(
        family=args.family,
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output=args.output,
    )

    print(
        "FAMILY =",
        report["family_key"],
    )
    print(
        "RESULT =",
        report["result"],
    )
    print(
        "SOURCE_PAIR_ID =",
        report["source_pair_id"],
    )
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
    print(
        "MAX_PE_ABS_DIFF =",
        report["max_pe_abs_diff"],
    )
    print(
        "SCIENTIFIC_BUDGET_FORWARD_COUNT = 0"
    )
    print(
        "SCIENTIFIC_CONCLUSION = NONE"
    )


if __name__ == "__main__":
    main()
