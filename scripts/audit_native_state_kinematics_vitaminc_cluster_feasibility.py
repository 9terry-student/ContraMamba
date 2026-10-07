#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


AUTHORITY_COMMIT = "5f27d9f09ee236fcd4e05ff31bbb7753c5ac56fe"

EXPECTED_RAW_DATASET_SHA256 = (
    "452a24ec32302b4db6100c0f5897533d726f3d4408662b26bc3089b0b46d3191"
)
EXPECTED_FIXED219_SHA256 = (
    "1568905229d3f53a2c02bd9969df440a9b8ffde21bd335c2fe048191286f4859"
)
EXPECTED_PAIRS128_SHA256 = (
    "3e6ee542123589d53fcb657cd62aade824debe9c0737385f4f0558a554b6afa0"
)

EXPECTED = {
    "dataset_rows": 5000,
    "unique_case_ids": 1497,
    "complete_2x2_case_ids": 1010,
    "complete_2x2_postedit_suffix_ge4_case_ids": 681,
    "historical_decisive_error_rows": 128,
    "historical_unique_raw_error_rows": 93,
    "historical_unique_error_case_ids": 78,
    "historical_pair_rows": 128,
    "historical_same_predicted_class_pairs": 57,
    "historical_wrong_control_case_id_overlap": 4,
}

RAW_REQUIRED_COLUMNS = {
    "raw_idx",
    "unique_id",
    "case_id",
    "label",
    "claim",
    "evidence",
}

FIXED_REQUIRED_COLUMNS = {
    "source_seed",
    "raw_idx",
    "gold",
    "model_pred_original",
    "claim",
    "evidence",
    "full_source_pred",
    "full_source_conf",
    "full_still_ge_095",
    "full_gold_error",
}

PAIR_REQUIRED_COLUMNS = {
    "source_seed",
    "raw_idx",
    "model_pred_original",
    "full_source_pred",
    "selected_seed",
    "selected_raw_idx",
    "matched_raw_idx",
    "control_seed",
    "control_raw_idx",
    "control_pred",
    "control_correct",
    "wrong_claim_raw",
    "wrong_evidence_raw",
    "control_claim",
    "control_evidence",
}


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"MISSING_INPUT:{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV_HEADER_MISSING:{path}")
        return [dict(row) for row in reader]


def require_columns(
    rows: list[dict[str, str]],
    required: set[str],
    label: str,
) -> None:
    require(bool(rows), f"{label}_EMPTY")
    missing = sorted(required - set(rows[0]))
    require(not missing, f"{label}_MISSING_COLUMNS:{','.join(missing)}")


def as_int(value: Any, label: str) -> int:
    try:
        return int(str(value).strip())
    except Exception as exc:
        raise ContractError(f"INVALID_INT:{label}:{value}") from exc


def truthy01(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no", ""}:
        return False
    raise ContractError(f"INVALID_BOOLEAN:{value}")


def normalize_label(value: Any) -> str:
    text = str(value).strip().upper().replace("_", " ")
    aliases = {
        "SUPPORT": "SUPPORTS",
        "SUPPORTS": "SUPPORTS",
        "REFUTE": "REFUTES",
        "REFUTES": "REFUTES",
        "NEI": "NEI",
        "NOT ENOUGH INFO": "NEI",
        "NOT ENTITLED": "NEI",
    }
    return aliases.get(text, text)


def git_head(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    require(proc.returncode == 0, "GIT_HEAD_READ_FAILED")
    return proc.stdout.strip()


def whitespace_tokens(text: str) -> list[str]:
    return str(text).split()


def raw_edit_span(left: str, right: str) -> dict[str, int]:
    """
    Deterministic outcome-blind raw-whitespace edit span.

    The span is the minimal region remaining after removing the longest
    identical token prefix and suffix from the two evidence variants.
    """
    a = whitespace_tokens(left)
    b = whitespace_tokens(right)

    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1

    suffix = 0
    max_suffix = min(len(a) - prefix, len(b) - prefix)
    while suffix < max_suffix and a[len(a) - 1 - suffix] == b[len(b) - 1 - suffix]:
        suffix += 1

    a_end = len(a) - suffix
    b_end = len(b) - suffix

    return {
        "left_token_count": len(a),
        "right_token_count": len(b),
        "common_prefix_tokens": prefix,
        "common_suffix_tokens": suffix,
        "left_edit_start": prefix,
        "left_edit_end_exclusive": a_end,
        "right_edit_start": prefix,
        "right_edit_end_exclusive": b_end,
        "postedit_suffix_tokens": suffix,
    }


def is_complete_2x2(rows: list[dict[str, str]]) -> bool:
    if len(rows) != 4:
        return False

    claims = {row["claim"] for row in rows}
    evidences = {row["evidence"] for row in rows}
    if len(claims) != 2 or len(evidences) != 2:
        return False

    observed = {(row["claim"], row["evidence"]) for row in rows}
    expected = {(claim, evidence) for claim in claims for evidence in evidences}
    return observed == expected


def audit_raw_dataset(rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[int, dict[str, str]]]:
    require_columns(rows, RAW_REQUIRED_COLUMNS, "RAW_DATASET")

    raw_indices = [as_int(row["raw_idx"], "raw_idx") for row in rows]
    require(len(raw_indices) == len(set(raw_indices)), "RAW_IDX_NOT_UNIQUE")
    require(
        sorted(raw_indices) == list(range(len(rows))),
        "RAW_IDX_NOT_ZERO_BASED_CONTIGUOUS",
    )

    raw_by_idx = {as_int(row["raw_idx"], "raw_idx"): row for row in rows}

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        case_id = str(row["case_id"])
        require(case_id != "", "EMPTY_CASE_ID")
        groups[case_id].append(row)

    complete_case_ids: list[str] = []
    post4_case_ids: list[str] = []
    suffix_histogram: dict[int, int] = defaultdict(int)

    for case_id, case_rows in groups.items():
        if not is_complete_2x2(case_rows):
            continue

        complete_case_ids.append(case_id)
        evidences = sorted({row["evidence"] for row in case_rows})
        require(len(evidences) == 2, f"INTERNAL_EVIDENCE_CARDINALITY:{case_id}")

        span = raw_edit_span(evidences[0], evidences[1])
        suffix = span["postedit_suffix_tokens"]
        suffix_histogram[suffix] += 1

        if suffix >= 4:
            post4_case_ids.append(case_id)

    summary = {
        "dataset_rows": len(rows),
        "unique_case_ids": len(groups),
        "complete_2x2_case_ids": len(complete_case_ids),
        "complete_2x2_postedit_suffix_ge4_case_ids": len(post4_case_ids),
        "candidate_event_rule": (
            "longest_common_raw_whitespace_prefix_suffix_minimal_edit_span"
        ),
        "candidate_event_is_mamba_tau_e": False,
        "postedit_suffix_threshold_raw_whitespace_tokens": 4,
        "complete_case_ids_sha256": hashlib.sha256(
            "\n".join(sorted(complete_case_ids)).encode("utf-8")
        ).hexdigest(),
        "post4_candidate_case_ids_sha256": hashlib.sha256(
            "\n".join(sorted(post4_case_ids)).encode("utf-8")
        ).hexdigest(),
        "suffix_histogram": {
            str(key): suffix_histogram[key]
            for key in sorted(suffix_histogram)
        },
    }
    return summary, raw_by_idx


def assert_source_text(
    raw_by_idx: dict[int, dict[str, str]],
    raw_idx: int,
    claim: str,
    evidence: str,
    label: str,
) -> None:
    require(raw_idx in raw_by_idx, f"{label}_RAW_IDX_OUT_OF_RANGE:{raw_idx}")
    source = raw_by_idx[raw_idx]
    require(source["claim"] == claim, f"{label}_CLAIM_MISMATCH:{raw_idx}")
    require(source["evidence"] == evidence, f"{label}_EVIDENCE_MISMATCH:{raw_idx}")


def audit_historical(
    fixed_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    raw_by_idx: dict[int, dict[str, str]],
) -> dict[str, Any]:
    require_columns(fixed_rows, FIXED_REQUIRED_COLUMNS, "FIXED219")
    require_columns(pair_rows, PAIR_REQUIRED_COLUMNS, "PAIRS128")

    for row in fixed_rows:
        idx = as_int(row["raw_idx"], "fixed.raw_idx")
        assert_source_text(
            raw_by_idx,
            idx,
            row["claim"],
            row["evidence"],
            "FIXED",
        )
        require(truthy01(row["full_gold_error"]), f"FIXED_NOT_ERROR:{idx}")
        require(truthy01(row["full_still_ge_095"]), f"FIXED_NOT_HIGH_CONF:{idx}")

    decisive = [
        row
        for row in fixed_rows
        if normalize_label(row["model_pred_original"]) in {"SUPPORTS", "REFUTES"}
    ]

    decisive_raw_indices = {
        as_int(row["raw_idx"], "decisive.raw_idx")
        for row in decisive
    }
    decisive_case_ids = {
        raw_by_idx[idx]["case_id"]
        for idx in decisive_raw_indices
    }

    decisive_seed_raw = {
        (
            as_int(row["source_seed"], "decisive.source_seed"),
            as_int(row["raw_idx"], "decisive.raw_idx"),
        )
        for row in decisive
    }

    pair_seed_raw: set[tuple[int, int]] = set()
    wrong_case_ids: set[str] = set()
    control_case_ids: set[str] = set()
    same_predicted_class = 0
    paired_same_case_id = 0

    for row in pair_rows:
        source_seed = as_int(row["source_seed"], "pair.source_seed")
        selected_seed = as_int(row["selected_seed"], "pair.selected_seed")
        selected_raw_idx = as_int(row["selected_raw_idx"], "pair.selected_raw_idx")
        wrong_raw_idx = as_int(row["raw_idx"], "pair.raw_idx")
        control_seed = as_int(row["control_seed"], "pair.control_seed")
        control_raw_idx = as_int(row["control_raw_idx"], "pair.control_raw_idx")
        matched_raw_idx = as_int(row["matched_raw_idx"], "pair.matched_raw_idx")

        require(source_seed == selected_seed, "PAIR_SOURCE_SELECTED_SEED_MISMATCH")
        require(source_seed == control_seed, "PAIR_WRONG_CONTROL_SEED_MISMATCH")
        require(wrong_raw_idx == selected_raw_idx, "PAIR_SELECTED_RAW_IDX_MISMATCH")
        require(control_raw_idx == matched_raw_idx, "PAIR_CONTROL_RAW_IDX_MISMATCH")
        require(truthy01(row["control_correct"]), "PAIR_CONTROL_NOT_CORRECT")

        assert_source_text(
            raw_by_idx,
            wrong_raw_idx,
            row["wrong_claim_raw"],
            row["wrong_evidence_raw"],
            "PAIR_WRONG",
        )
        assert_source_text(
            raw_by_idx,
            control_raw_idx,
            row["control_claim"],
            row["control_evidence"],
            "PAIR_CONTROL",
        )

        wrong_case = raw_by_idx[wrong_raw_idx]["case_id"]
        control_case = raw_by_idx[control_raw_idx]["case_id"]
        wrong_case_ids.add(wrong_case)
        control_case_ids.add(control_case)

        pair_seed_raw.add((source_seed, wrong_raw_idx))

        wrong_pred = normalize_label(row["model_pred_original"])
        control_pred = normalize_label(row["control_pred"])
        if wrong_pred == control_pred:
            same_predicted_class += 1
        if wrong_case == control_case:
            paired_same_case_id += 1

    require(
        pair_seed_raw == decisive_seed_raw,
        "PAIRS128_NOT_EXACT_DECISIVE_FIXED_SUBSET",
    )

    overlap = wrong_case_ids & control_case_ids

    return {
        "historical_fixed_error_rows": len(fixed_rows),
        "historical_decisive_error_rows": len(decisive),
        "historical_unique_raw_error_rows": len(decisive_raw_indices),
        "historical_unique_error_case_ids": len(decisive_case_ids),
        "historical_pair_rows": len(pair_rows),
        "historical_same_predicted_class_pairs": same_predicted_class,
        "historical_wrong_control_case_id_overlap": len(overlap),
        "historical_paired_same_case_id_rows": paired_same_case_id,
        "historical_wrong_case_ids": len(wrong_case_ids),
        "historical_control_case_ids": len(control_case_ids),
        "wrong_control_overlap_case_ids_sha256": hashlib.sha256(
            "\n".join(sorted(overlap)).encode("utf-8")
        ).hexdigest(),
        "existing_pairs_confirmatory_eligible": False,
        "existing_pairs_status": "DIAGNOSTIC_ONLY_NOT_CONFIRMATORY",
    }


def enforce_expected(summary: dict[str, Any]) -> None:
    for key, expected in EXPECTED.items():
        actual = summary.get(key)
        require(
            actual == expected,
            f"EXPECTED_OBSERVATION_MISMATCH:{key}:expected={expected}:actual={actual}",
        )


def verify_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    require(
        actual == expected,
        f"{label}_SHA256_MISMATCH:expected={expected}:actual={actual}",
    )
    return actual


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_head:
        require(args.repo is not None, "--repo required with --expected-head")
        actual_head = git_head(args.repo)
        require(
            actual_head == args.expected_head,
            f"HEAD_MISMATCH:expected={args.expected_head}:actual={actual_head}",
        )

    hashes = {
        "raw_dataset_sha256": verify_hash(
            args.raw_dataset,
            EXPECTED_RAW_DATASET_SHA256,
            "RAW_DATASET",
        ),
        "fixed219_sha256": verify_hash(
            args.fixed_errors,
            EXPECTED_FIXED219_SHA256,
            "FIXED219",
        ),
        "pairs128_sha256": verify_hash(
            args.pairs,
            EXPECTED_PAIRS128_SHA256,
            "PAIRS128",
        ),
    }

    raw_rows = read_csv(args.raw_dataset)
    fixed_rows = read_csv(args.fixed_errors)
    pair_rows = read_csv(args.pairs)

    raw_summary, raw_by_idx = audit_raw_dataset(raw_rows)
    historical_summary = audit_historical(
        fixed_rows=fixed_rows,
        pair_rows=pair_rows,
        raw_by_idx=raw_by_idx,
    )

    summary: dict[str, Any] = {
        "schema_version": "NATIVE_Q1_VITAMINC_CLUSTER_STATIC_FEASIBILITY_V1",
        "status": "PASS_STATIC_REFORMULATION_FEASIBILITY",
        "authority_commit": AUTHORITY_COMMIT,
        "execution_head": args.expected_head,
        "hashes": hashes,
        **raw_summary,
        **historical_summary,
        "independent_sampling_unit": "case_id",
        "row_level_pseudoreplication_prohibited": True,
        "wrong_control_case_id_overlap_prohibited": True,
        "epistemicbert_role": "EXTERNAL_DIFFICULTY_AND_FEASIBILITY_DIAGNOSTIC_ONLY",
        "epistemicbert_labels_admitted_as_mamba_outcomes": False,
        "distilbert_token_coordinates_admitted_as_mamba_event_time": False,
        "model_forward_count": 0,
        "checkpoint_loaded": False,
        "native_state_accessed": False,
        "cuda_executed": False,
        "training_executed": False,
        "next_stage_on_pass": "MAMBA_PREDICTION_ONLY_COHORT_CENSUS_PREPARATION",
    }

    enforce_expected(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU/static Native Q1 VitaminC clustered-cohort feasibility audit. "
            "No model or native-state access."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-verify-only", action="store_true")
    mode.add_argument("--run-static-audit", action="store_true")

    parser.add_argument("--repo", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--fixed-errors", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_audit(args)

        if args.static_verify_only:
            require(args.output_json is None, "VERIFY_ONLY_MUST_NOT_WRITE_OUTPUT")
            print("NATIVE_Q1_VITAMINC_CLUSTER_STATIC_VERIFY_PASS")
            for key in (
                "dataset_rows",
                "unique_case_ids",
                "complete_2x2_case_ids",
                "complete_2x2_postedit_suffix_ge4_case_ids",
                "historical_decisive_error_rows",
                "historical_unique_raw_error_rows",
                "historical_unique_error_case_ids",
                "historical_pair_rows",
                "historical_same_predicted_class_pairs",
                "historical_wrong_control_case_id_overlap",
            ):
                print(f"{key.upper()}={summary[key]}")
            print("MODEL_FORWARD_COUNT=0")
            print("CHECKPOINT_LOADED=False")
            print("NATIVE_STATE_ACCESSED=False")
            print("CUDA_EXECUTED=False")
            print("TRAINING_EXECUTED=False")
            print("FILES_WRITTEN=0")
            return 0

        require(args.output_json is not None, "--output-json required for --run-static-audit")
        require(
            not args.output_json.exists(),
            f"OUTPUT_ALREADY_EXISTS:{args.output_json}",
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=False)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("NATIVE_Q1_VITAMINC_CLUSTER_STATIC_AUDIT_PASS")
        print(f"OUTPUT_JSON={args.output_json}")
        return 0

    except ContractError as exc:
        print(f"BLOCKED:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())