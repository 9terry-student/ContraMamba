#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (
    build_reason_router_gen4_averitec_gold_evidence_130m370m_token_gate
    as base,
)


EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "69256339cde93d77db44851782327a18bbdf965a"

DESIGN_ARTIFACT = Path(
    "reports/reason_router_gen4_causal_atlas_guided_steering_design.md"
)
DESIGN_ARTIFACT_GIT_BLOB = "0d35f9aebb37e6203bea40304d3a55ab2361b684"

BASE_GATE = Path(
    "scripts/build_reason_router_gen4_averitec_gold_evidence_130m370m_token_gate.py"
)
BASE_GATE_GIT_BLOB = "0f480164695b6bc4c4206575e1767926fbb71cf0"

UPSTREAM_REPO = "MichSchli/AVeriTeC"
UPSTREAM_COMMIT = "7c62d1ec8df3fb560d6efe2b85fa191135636f81"

TRAIN_PATH = "data/train.json"
TRAIN_BLOB_SHA1 = "0f190e115cf2ee23416e8a539c8d6ac043d7cc83"
TRAIN_BYTES = 10184813
TRAIN_RAW_URL = (
    "https://raw.githubusercontent.com/"
    f"{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/{TRAIN_PATH}"
)

DEV_PATH = "data/dev.json"
DEV_BLOB_SHA1 = "40974243267f395dc583d805d10f043812419249"
DEV_BYTES = 1785475
DEV_RAW_URL = (
    "https://raw.githubusercontent.com/"
    f"{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/{DEV_PATH}"
)

EXPECTED_TRAIN_ROWS = 3068
EXPECTED_TRAIN_LABEL_COUNTS = {
    "Supported": 849,
    "Refuted": 1742,
    "Not Enough Evidence": 282,
    "Conflicting Evidence/Cherrypicking": 195,
}
EXPECTED_COMPATIBLE_BEFORE_FRESHNESS = 2873

EXPECTED_EXCLUSION_COUNTS = {
    "incompatible_label": 195,
    "dev_claim_overlap": 6,
    "dev_url_overlap": 1,
    "later_within_train_duplicate_claim": 66,
}
EXPECTED_FINAL_COUNT = 2800
EXPECTED_FINAL_LABEL_COUNTS = {
    "Refuted": 1727,
    "Supported": 806,
    "Not Enough Evidence": 267,
}

COHORT_SCHEMA = "GEN4_AVERITEC_TRAIN_FRESH_STEERING_COHORT_V1"
MANIFEST_SCHEMA = "GEN4_AVERITEC_TRAIN_FRESH_STEERING_TOKEN_GATE_V1"
PASS_RESULT = "PASS_2800_OF_2800"
BLOCK_RESULT = "BLOCKED_STEERING_FRESH_TOKEN_GATE"

OUTPUT_DIR = Path(
    "data/reason_router_gen4_averitec_train_fresh_steering_boundary_v1"
)
COHORT_FILE = "fresh_compatible_cohort.jsonl"
MANIFEST_FILE = "token_gate_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

FORBIDDEN_RESPONSE_FIELDS = set(base.FORBIDDEN_RESPONSE_FIELDS) | {
    "native_prediction",
    "steering_prediction",
    "control_prediction",
    "native_correct",
    "steering_correct",
    "control_correct",
    "correction",
    "damage",
    "p_value",
    "margin_delta",
}


class SteeringCohortError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SteeringCohortError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SteeringCohortError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REQUIRED_ANCESTOR")

    pinned = {
        DESIGN_ARTIFACT.as_posix(): DESIGN_ARTIFACT_GIT_BLOB,
        BASE_GATE.as_posix(): BASE_GATE_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def validate_pinned_bytes(
    raw: bytes,
    *,
    expected_bytes: int,
    expected_blob_sha1: str,
    label: str,
) -> None:
    require(
        len(raw) == expected_bytes,
        f"{label}_BYTES:{len(raw)}",
    )
    require(
        git_blob_sha1(raw) == expected_blob_sha1,
        f"{label}_GIT_BLOB_SHA1",
    )


def _validate_example_structure(
    row: Mapping[str, Any],
    *,
    index: int,
    label_prefix: str,
) -> None:
    claim = row.get("claim")
    require(
        isinstance(claim, str) and bool(claim.strip()),
        f"{label_prefix}_CLAIM:{index}",
    )

    label = row.get("label")
    require(
        isinstance(label, str) and label in EXPECTED_TRAIN_LABEL_COUNTS,
        f"{label_prefix}_LABEL:{index}:{label}",
    )

    url = row.get("original_claim_url")
    require(
        url is None or isinstance(url, str),
        f"{label_prefix}_URL:{index}",
    )

    questions = row.get("questions")
    require(
        isinstance(questions, list) and len(questions) >= 1,
        f"{label_prefix}_QUESTIONS:{index}",
    )
    for qi, question_obj in enumerate(questions):
        require(
            isinstance(question_obj, dict),
            f"{label_prefix}_QUESTION_OBJECT:{index}:{qi}",
        )
        question = question_obj.get("question")
        require(
            isinstance(question, str) and bool(question.strip()),
            f"{label_prefix}_QUESTION_TEXT:{index}:{qi}",
        )
        answers = question_obj.get("answers")
        require(
            isinstance(answers, list) and len(answers) >= 1,
            f"{label_prefix}_ANSWERS:{index}:{qi}",
        )
        for ai, answer_obj in enumerate(answers):
            require(
                isinstance(answer_obj, dict),
                f"{label_prefix}_ANSWER_OBJECT:{index}:{qi}:{ai}",
            )
            answer = answer_obj.get("answer")
            require(
                isinstance(answer, str) and bool(answer.strip()),
                f"{label_prefix}_ANSWER_TEXT:{index}:{qi}:{ai}",
            )


def parse_train(raw: bytes) -> list[dict[str, Any]]:
    validate_pinned_bytes(
        raw,
        expected_bytes=TRAIN_BYTES,
        expected_blob_sha1=TRAIN_BLOB_SHA1,
        label="TRAIN",
    )
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, list), "TRAIN_TOPLEVEL_LIST")
    require(len(value) == EXPECTED_TRAIN_ROWS, "TRAIN_ROW_COUNT")
    require(
        all(isinstance(row, dict) for row in value),
        "TRAIN_OBJECT_ROWS",
    )

    counts = Counter(str(row.get("label")) for row in value)
    require(
        dict(counts) == EXPECTED_TRAIN_LABEL_COUNTS,
        f"TRAIN_LABEL_COUNTS:{dict(counts)}",
    )

    for index, row in enumerate(value):
        _validate_example_structure(
            row,
            index=index,
            label_prefix="TRAIN",
        )
    return value


def parse_dev(raw: bytes) -> list[dict[str, Any]]:
    validate_pinned_bytes(
        raw,
        expected_bytes=DEV_BYTES,
        expected_blob_sha1=DEV_BLOB_SHA1,
        label="DEV",
    )
    value = base.parse_source(raw)
    require(len(value) == 500, "DEV_ROW_COUNT")
    return value


def normalize_claim(text: str) -> str:
    require(isinstance(text, str), "NORMALIZE_CLAIM_TYPE")
    normalized = " ".join(text.strip().split()).lower()
    require(bool(normalized), "NORMALIZE_CLAIM_EMPTY")
    return normalized


def exact_nonempty_url(row: Mapping[str, Any]) -> str | None:
    value = row.get("original_claim_url")
    if value is None:
        return None
    require(isinstance(value, str), "URL_TYPE")
    return value if value != "" else None


def derive_fresh_examples(
    *,
    train: Sequence[Mapping[str, Any]],
    dev: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, Mapping[str, Any]]], dict[str, Any]]:
    require(len(train) == EXPECTED_TRAIN_ROWS, "DERIVE_TRAIN_N")
    require(len(dev) == 500, "DERIVE_DEV_N")

    dev_claims = {
        normalize_claim(str(row["claim"]))
        for row in dev
    }
    dev_urls = {
        url
        for row in dev
        if (url := exact_nonempty_url(row)) is not None
    }

    exclusions = Counter()
    seen_train_claims: set[str] = set()
    kept: list[tuple[int, Mapping[str, Any]]] = []

    compatible_before_freshness = 0

    for index, row in enumerate(train):
        label = str(row["label"])

        if label == base.EXCLUDED_LABEL:
            exclusions["incompatible_label"] += 1
            continue

        require(
            label in base.COMPATIBLE_LABELS,
            f"DERIVE_UNKNOWN_LABEL:{index}:{label}",
        )
        compatible_before_freshness += 1

        claim_key = normalize_claim(str(row["claim"]))
        if claim_key in dev_claims:
            exclusions["dev_claim_overlap"] += 1
            continue

        url = exact_nonempty_url(row)
        if url is not None and url in dev_urls:
            exclusions["dev_url_overlap"] += 1
            continue

        if claim_key in seen_train_claims:
            exclusions["later_within_train_duplicate_claim"] += 1
            continue

        seen_train_claims.add(claim_key)
        kept.append((index, row))

    require(
        compatible_before_freshness
        == EXPECTED_COMPATIBLE_BEFORE_FRESHNESS,
        f"COMPATIBLE_BEFORE_FRESHNESS:{compatible_before_freshness}",
    )
    require(
        dict(exclusions) == EXPECTED_EXCLUSION_COUNTS,
        f"EXCLUSION_COUNTS:{dict(exclusions)}",
    )
    require(
        len(kept) == EXPECTED_FINAL_COUNT,
        f"FINAL_COUNT:{len(kept)}",
    )

    label_counts = Counter(str(row["label"]) for _, row in kept)
    require(
        dict(label_counts) == EXPECTED_FINAL_LABEL_COUNTS,
        f"FINAL_LABEL_COUNTS:{dict(label_counts)}",
    )
    require(
        len({
            normalize_claim(str(row["claim"]))
            for _, row in kept
        }) == EXPECTED_FINAL_COUNT,
        "FINAL_NORMALIZED_CLAIM_UNIQUENESS",
    )

    kept_indices = [index for index, _ in kept]
    require(
        kept_indices == sorted(kept_indices),
        "FINAL_SOURCE_ORDER",
    )

    audit = {
        "normalization": (
            "strip; collapse each maximal whitespace run to one ASCII space; "
            "Unicode lowercase"
        ),
        "deduplication_order": [
            "exclude_conflicting_evidence_cherrypicking",
            "exclude_exact_normalized_claim_match_to_dev",
            "exclude_exact_nonempty_original_claim_url_match_to_dev",
            "retain_lowest_zero_based_train_index_per_remaining_normalized_claim",
        ],
        "compatible_before_freshness": compatible_before_freshness,
        "exclusion_counts": dict(exclusions),
        "final_count": len(kept),
        "final_label_counts": dict(label_counts),
    }
    return kept, audit


def gate_train_row(
    *,
    tokenizer: Any,
    source_index: int,
    example: Mapping[str, Any],
) -> dict[str, Any]:
    source_label = str(example["label"])
    require(
        source_label in base.LABEL_MAP,
        f"INCOMPATIBLE_LABEL:{source_label}",
    )
    mapped = base.LABEL_MAP[source_label]

    claim = str(example["claim"])
    evidence, question_count, answer_count = (
        base.serialize_gold_evidence(example)
    )

    claim_ids = base._token_ids(tokenizer, claim)
    evidence_ids = base._token_ids(tokenizer, evidence)
    consumed_claim = claim_ids[:base.CLAIM_BUDGET]
    consumed_evidence = evidence_ids[:base.EVIDENCE_BUDGET]

    boundary_index = len(consumed_claim)
    target_index = boundary_index + base.TARGET_OFFSET
    serialized_length = (
        len(consumed_claim)
        + 1
        + len(consumed_evidence)
    )

    gate_reasons: list[str] = []
    if not consumed_claim:
        gate_reasons.append("EMPTY_CONSUMED_CLAIM")
    if not consumed_evidence:
        gate_reasons.append("EMPTY_CONSUMED_EVIDENCE")
    if len(consumed_evidence) < 2:
        gate_reasons.append("FEWER_THAN_TWO_CONSUMED_EVIDENCE_TOKENS")
    if serialized_length > base.MAX_LENGTH:
        gate_reasons.append("SERIALIZED_LENGTH_EXCEEDS_128")
    if not (0 <= boundary_index < serialized_length):
        gate_reasons.append("BOUNDARY_OUTSIDE_ATTENDED_SEQUENCE")
    if not (0 <= target_index < serialized_length):
        gate_reasons.append("TARGET_OUTSIDE_ATTENDED_SEQUENCE")
    if target_index != len(consumed_claim) + 2:
        gate_reasons.append("TARGET_OFFSET_DRIFT")

    row = {
        "schema_version": COHORT_SCHEMA,
        "averitec_train_index": int(source_index),
        "example_id": f"averitec_train_{source_index:04d}",
        "claim": claim,
        "normalized_claim": normalize_claim(claim),
        "original_claim_url": exact_nonempty_url(example),
        "evidence": evidence,
        "source_label": source_label,
        "correct_label": str(mapped["label"]),
        "correct_label_id": int(mapped["label_id"]),
        "question_count": int(question_count),
        "answer_count": int(answer_count),
        "claim_raw_token_count": len(claim_ids),
        "claim_consumed_token_count": len(consumed_claim),
        "evidence_raw_token_count": len(evidence_ids),
        "evidence_consumed_token_count": len(consumed_evidence),
        "claim_truncated": len(claim_ids) > base.CLAIM_BUDGET,
        "evidence_truncated": len(evidence_ids) > base.EVIDENCE_BUDGET,
        "anchor_name": base.ANCHOR_NAME,
        "absolute_anchor_token_index": int(boundary_index),
        "target_intervention_token_index": int(target_index),
        "serialized_attended_length": int(serialized_length),
        "target_offset": base.TARGET_OFFSET,
        "token_gate_pass": not gate_reasons,
        "token_gate_reasons": gate_reasons,
    }
    require(
        not (FORBIDDEN_RESPONSE_FIELDS & set(row)),
        "RESPONSE_FIELD_PRESENT",
    )
    return row


def build_rows(
    *,
    fresh_examples: Sequence[tuple[int, Mapping[str, Any]]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    rows = [
        gate_train_row(
            tokenizer=tokenizer,
            source_index=index,
            example=example,
        )
        for index, example in fresh_examples
    ]

    require(len(rows) == EXPECTED_FINAL_COUNT, "BUILT_ROW_COUNT")
    require(
        len({row["example_id"] for row in rows})
        == EXPECTED_FINAL_COUNT,
        "EXAMPLE_ID_UNIQUENESS",
    )
    require(
        len({row["normalized_claim"] for row in rows})
        == EXPECTED_FINAL_COUNT,
        "NORMALIZED_CLAIM_UNIQUENESS",
    )
    require(
        [int(row["averitec_train_index"]) for row in rows]
        == sorted(int(row["averitec_train_index"]) for row in rows),
        "ROW_SOURCE_ORDER",
    )
    return rows


def build_manifest(
    *,
    train_raw: bytes,
    dev_raw: bytes,
    rows: Sequence[Mapping[str, Any]],
    freshness_audit: Mapping[str, Any],
    tokenizer_provenance: Mapping[str, Any],
    cohort_sha256: str,
) -> dict[str, Any]:
    require(len(rows) == EXPECTED_FINAL_COUNT, "MANIFEST_ROW_COUNT")

    pass_count = sum(bool(row["token_gate_pass"]) for row in rows)
    fail_count = len(rows) - pass_count
    failures = Counter(
        reason
        for row in rows
        for reason in row["token_gate_reasons"]
    )

    label_counts = Counter(str(row["source_label"]) for row in rows)
    require(
        dict(label_counts) == EXPECTED_FINAL_LABEL_COUNTS,
        f"MANIFEST_LABEL_COUNTS:{dict(label_counts)}",
    )

    result = (
        PASS_RESULT
        if pass_count == EXPECTED_FINAL_COUNT
        else BLOCK_RESULT
    )

    return {
        "schema_version": MANIFEST_SCHEMA,
        "result": result,
        "phase": "EXPERIMENT_3_CAUSAL_ATLAS_GUIDED_STEERING_FRESH_TOKEN_GATE",
        "design": {
            "artifact": DESIGN_ARTIFACT.as_posix(),
            "artifact_git_blob": DESIGN_ARTIFACT_GIT_BLOB,
        },
        "upstream": {
            "repository": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "train": {
                "path": TRAIN_PATH,
                "git_blob_sha1": TRAIN_BLOB_SHA1,
                "byte_count": len(train_raw),
                "sha256": sha256_bytes(train_raw),
            },
            "dev": {
                "path": DEV_PATH,
                "git_blob_sha1": DEV_BLOB_SHA1,
                "byte_count": len(dev_raw),
                "sha256": sha256_bytes(dev_raw),
            },
            "license": base.LICENSE,
            "paper_citation": base.PAPER_CITATION,
        },
        "source_train_count": EXPECTED_TRAIN_ROWS,
        "source_train_label_counts": dict(EXPECTED_TRAIN_LABEL_COUNTS),
        "compatible_labels": list(base.COMPATIBLE_LABELS),
        "excluded_label": base.EXCLUDED_LABEL,
        "freshness": dict(freshness_audit),
        "fresh_cohort_count": len(rows),
        "fresh_cohort_label_counts": dict(label_counts),
        "cohort_file_sha256": cohort_sha256,
        "gold_evidence_serialization": (
            "source-order repeated blocks: "
            "Question: <question>\\nAnswer: <answer>, joined by \\n\\n"
        ),
        "justification_used_as_model_input": False,
        "retrieval_executed": False,
        "question_generation_executed": False,
        "evidence_reordering_executed": False,
        "tokenizer": dict(tokenizer_provenance),
        "anchor_contract": {
            "anchor_name": base.ANCHOR_NAME,
            "anchor_definition": "EOS separator after consumed claim tokens",
            "target_offset": base.TARGET_OFFSET,
            "target_definition": "second consumed evidence token",
        },
        "token_gate": {
            "target_count": EXPECTED_FINAL_COUNT,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "failure_counts": dict(sorted(failures.items())),
        },
        "truncation": {
            "claim_truncated_count":
                sum(bool(row["claim_truncated"]) for row in rows),
            "evidence_truncated_count":
                sum(bool(row["evidence_truncated"]) for row in rows),
            "claim_raw_tokens":
                base.summarize_lengths(rows, "claim_raw_token_count"),
            "claim_consumed_tokens":
                base.summarize_lengths(rows, "claim_consumed_token_count"),
            "evidence_raw_tokens":
                base.summarize_lengths(rows, "evidence_raw_token_count"),
            "evidence_consumed_tokens":
                base.summarize_lengths(rows, "evidence_consumed_token_count"),
            "anchor_index":
                base.summarize_lengths(rows, "absolute_anchor_token_index"),
            "target_index":
                base.summarize_lengths(rows, "target_intervention_token_index"),
        },
        "planned_steering_execution": {
            "model_scale": "mamba370m",
            "conditions": [
                "native",
                "p3_mirror_steer",
                "p5_matched_control",
            ],
            "expected_row_count": EXPECTED_FINAL_COUNT,
            "expected_full_model_forward_count": 8400,
            "scientific_execution_authorized_by_this_artifact": False,
        },
        "model_checkpoint_loaded": False,
        "model_forward_count": 0,
        "cuda_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "scientific_inference_executed": False,
        "p_value_count_added": 0,
        "response_fields_present": False,
    }


def write_outputs(
    *,
    output_dir: Path,
    train_raw: bytes,
    dev_raw: bytes,
    rows: Sequence[Mapping[str, Any]],
    freshness_audit: Mapping[str, Any],
    tokenizer_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        not output_dir.exists(),
        f"OUTPUT_COLLISION:{output_dir}",
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    cohort_raw = jsonl_bytes(rows)
    cohort_sha = sha256_bytes(cohort_raw)

    manifest = build_manifest(
        train_raw=train_raw,
        dev_raw=dev_raw,
        rows=rows,
        freshness_audit=freshness_audit,
        tokenizer_provenance=tokenizer_provenance,
        cohort_sha256=cohort_sha,
    )
    manifest_raw = pretty_json_bytes(manifest)

    (output_dir / COHORT_FILE).write_bytes(cohort_raw)
    (output_dir / MANIFEST_FILE).write_bytes(manifest_raw)

    hashes = {
        COHORT_FILE: cohort_sha,
        MANIFEST_FILE: sha256_bytes(manifest_raw),
    }
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def materialize(
    *,
    output_dir: Path,
    expected_head: str,
    train_raw: bytes | None = None,
    dev_raw: bytes | None = None,
    tokenizer_snapshot: Path | None = None,
) -> dict[str, Any]:
    authenticate_repo(expected_head)

    if train_raw is None:
        train_raw = download(TRAIN_RAW_URL)
    if dev_raw is None:
        dev_raw = download(DEV_RAW_URL)

    train = parse_train(train_raw)
    dev = parse_dev(dev_raw)

    fresh_a, audit_a = derive_fresh_examples(
        train=train,
        dev=dev,
    )
    fresh_b, audit_b = derive_fresh_examples(
        train=train,
        dev=dev,
    )
    require(
        [index for index, _ in fresh_a]
        == [index for index, _ in fresh_b],
        "DETERMINISTIC_FRESHNESS_INDICES",
    )
    require(audit_a == audit_b, "DETERMINISTIC_FRESHNESS_AUDIT")

    if tokenizer_snapshot is None:
        tokenizer_snapshot = base.provision_tokenizer_snapshot()
    tokenizer, tokenizer_provenance = base.load_tokenizer(
        tokenizer_snapshot
    )

    rows_a = build_rows(
        fresh_examples=fresh_a,
        tokenizer=tokenizer,
    )
    rows_b = build_rows(
        fresh_examples=fresh_b,
        tokenizer=tokenizer,
    )
    require(
        jsonl_bytes(rows_a) == jsonl_bytes(rows_b),
        "DETERMINISTIC_TOKEN_GATE_REGENERATION",
    )

    manifest = write_outputs(
        output_dir=output_dir,
        train_raw=train_raw,
        dev_raw=dev_raw,
        rows=rows_a,
        freshness_audit=audit_a,
        tokenizer_provenance=tokenizer_provenance,
    )
    return manifest


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the pinned fresh deduplicated AVeriTeC train-derived "
            "2800-row steering cohort and execute the response-blind tokenizer gate. "
            "No checkpoint load, model forward, CUDA, or inference."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT_DIR)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        default=None,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    manifest = materialize(
        output_dir=args.output_dir,
        expected_head=str(args.expected_head),
        tokenizer_snapshot=args.tokenizer_snapshot,
    )

    print("RESULT=" + str(manifest["result"]))
    print("UPSTREAM_COMMIT=" + UPSTREAM_COMMIT)
    print("TRAIN_BLOB_SHA1=" + TRAIN_BLOB_SHA1)
    print("DEV_BLOB_SHA1=" + DEV_BLOB_SHA1)
    print("SOURCE_TRAIN_ROWS=3068")
    print("COMPATIBLE_BEFORE_FRESHNESS=2873")
    print("EXCLUDED_INCOMPATIBLE=195")
    print("EXCLUDED_DEV_CLAIM_OVERLAP=6")
    print("EXCLUDED_DEV_URL_OVERLAP=1")
    print("EXCLUDED_WITHIN_TRAIN_DUPLICATE=66")
    print("FRESH_COHORT_COUNT=" + str(manifest["fresh_cohort_count"]))
    print(
        "FRESH_LABEL_COUNTS="
        + json.dumps(
            manifest["fresh_cohort_label_counts"],
            sort_keys=True,
        )
    )
    print(
        "TOKEN_GATE_PASS_COUNT="
        + str(manifest["token_gate"]["pass_count"])
    )
    print(
        "TOKEN_GATE_FAIL_COUNT="
        + str(manifest["token_gate"]["fail_count"])
    )
    print("MODEL_CHECKPOINT_LOADED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("SCIENTIFIC_INFERENCE_EXECUTED=False")
    print("P_VALUE_COUNT_ADDED=0")

    return 0 if manifest["result"] == PASS_RESULT else 2


if __name__ == "__main__":
    raise SystemExit(main())
