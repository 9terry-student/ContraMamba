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


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
EXPECTED_HEAD = "de4a00efdf3008a66d6b3f24d46a273d4b74fb9f"

UPSTREAM_REPO = "MichSchli/AVeriTeC"
UPSTREAM_COMMIT = "7c62d1ec8df3fb560d6efe2b85fa191135636f81"
UPSTREAM_PATH = "data/dev.json"
UPSTREAM_BLOB_SHA1 = "40974243267f395dc583d805d10f043812419249"
UPSTREAM_BYTES = 1785475
UPSTREAM_RAW_URL = (
    "https://raw.githubusercontent.com/"
    f"{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/{UPSTREAM_PATH}"
)

EXPECTED_SOURCE_EXAMPLES = 500
EXPECTED_LABEL_COUNTS = {
    "Refuted": 305,
    "Supported": 122,
    "Not Enough Evidence": 35,
    "Conflicting Evidence/Cherrypicking": 38,
}
COMPATIBLE_LABELS = (
    "Refuted",
    "Not Enough Evidence",
    "Supported",
)
EXCLUDED_LABEL = "Conflicting Evidence/Cherrypicking"
EXPECTED_COMPATIBLE_COUNT = 462

LABEL_MAP = {
    "Refuted": {"label": "REFUTE", "label_id": 0},
    "Not Enough Evidence": {"label": "NOT_ENTITLED", "label_id": 1},
    "Supported": {"label": "SUPPORT", "label_id": 2},
}

HF_REPO = "state-spaces/mamba-370m-hf"
HF_REVISION = "589179554943157be31701edd8b4558889276674"
TOKENIZERS_VERSION = "0.22.2"
TOKENIZER_FILE_SHA256 = {
    "tokenizer.json":
        "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf",
    "tokenizer_config.json":
        "9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb",
    "special_tokens_map.json":
        "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
}

PRIMARY_SCALES = ("mamba130m", "mamba370m")
PRIMARY_P_VALUE_COUNT_PLANNED = 2
PRIMARY_MULTIPLICITY = "holm"

FROZEN_CAUSAL_OBJECTS = {
    "mamba130m": {
        "backbone": "state-spaces/mamba-130m-hf",
        "behavioral_checkpoint_sha256":
            "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f",
        "selected_plane": "P3",
        "control_plane": "P5",
        "intervention_layer": 17,
        "prior_behavioral_bridge_supported": True,
    },
    "mamba370m": {
        "backbone": "state-spaces/mamba-370m-hf",
        "behavioral_checkpoint_sha256":
            "9d8e3db22af4636938679aac6a8a97dd45344937d434fab29eac2ddc41a52a72",
        "selected_plane": "P3",
        "control_plane": "P5",
        "intervention_layer": 35,
        "prior_behavioral_bridge_supported": True,
    },
}

# The completed 130M behavioral bridge and the 370M bridge use the same
# active tokenizer byte identities and the same 63/1/64 encoding contract.
SHARED_TOKENIZER_BYTES_ACROSS_PRIMARY_SCALES = True

MAX_LENGTH = 128
CLAIM_BUDGET = 63
EVIDENCE_BUDGET = 64
EOS_TOKEN_ID = 0
ANCHOR_NAME = "A_CLAIM_EVIDENCE_BOUNDARY"
TARGET_OFFSET = 2

OUTPUT_DIR = Path(
    "data/reason_router_gen4_averitec_gold_evidence_130m370m_boundary_v1"
)
COHORT_FILE = "compatible_cohort.jsonl"
MANIFEST_FILE = "token_gate_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

COHORT_SCHEMA = "GEN4_AVERITEC_GOLD_EVIDENCE_130M370M_COMPATIBLE_COHORT_V1"
MANIFEST_SCHEMA = "GEN4_AVERITEC_GOLD_EVIDENCE_130M370M_BOUNDARY_TOKEN_GATE_V1"
PASS_RESULT = "PASS_462_OF_462"
BLOCK_RESULT = "BLOCKED_EXTERNAL_TOKEN_GATE"

FORBIDDEN_RESPONSE_FIELDS = {
    "logits",
    "prediction",
    "prediction_id",
    "q_authorized",
    "entitlement_prob",
    "D_EXT",
    "margin",
    "correct_class_logit_margin",
    "model_output",
    "response",
}

LICENSE = "CC BY-NC 4.0"
PAPER_CITATION = (
    "Michael Sejr Schlichtkrull, Zhijiang Guo, Andreas Vlachos. "
    "AVeriTeC: A Dataset for Real-world Claim Verification with Evidence "
    "from the Web. NeurIPS 2023 Datasets and Benchmarks Track."
)


class AVeriTeCGateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AVeriTeCGateError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AVeriTeCGateError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo() -> None:
    require(
        git("branch", "--show-current") == EXPECTED_BRANCH,
        "BRANCH_MISMATCH",
    )
    require(git("rev-parse", "HEAD") == EXPECTED_HEAD, "HEAD_MISMATCH")


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


def download_pinned_source(
    *,
    url: str = UPSTREAM_RAW_URL,
) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    validate_source_identity(raw)
    return raw


def validate_source_identity(raw: bytes) -> None:
    require(len(raw) == UPSTREAM_BYTES, f"SOURCE_BYTES:{len(raw)}")
    require(
        git_blob_sha1(raw) == UPSTREAM_BLOB_SHA1,
        "SOURCE_GIT_BLOB_SHA1",
    )


def parse_source(raw: bytes) -> list[dict[str, Any]]:
    validate_source_identity(raw)
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, list), "SOURCE_TOPLEVEL_LIST")
    require(len(value) == EXPECTED_SOURCE_EXAMPLES, "SOURCE_EXAMPLE_COUNT")
    require(
        all(isinstance(row, dict) for row in value),
        "SOURCE_OBJECT_ROWS",
    )

    counts = Counter(str(row.get("label")) for row in value)
    require(dict(counts) == EXPECTED_LABEL_COUNTS, f"LABEL_COUNTS:{dict(counts)}")

    for index, row in enumerate(value):
        require(
            isinstance(row.get("claim"), str) and bool(row["claim"].strip()),
            f"CLAIM:{index}",
        )
        questions = row.get("questions")
        require(
            isinstance(questions, list) and len(questions) >= 1,
            f"QUESTIONS:{index}",
        )
        for qi, q in enumerate(questions):
            require(
                isinstance(q, dict)
                and isinstance(q.get("question"), str)
                and bool(q["question"].strip()),
                f"QUESTION_TEXT:{index}:{qi}",
            )
            answers = q.get("answers")
            require(
                isinstance(answers, list) and len(answers) >= 1,
                f"ANSWERS:{index}:{qi}",
            )
            for ai, answer in enumerate(answers):
                require(
                    isinstance(answer, dict)
                    and isinstance(answer.get("answer"), str)
                    and bool(answer["answer"].strip()),
                    f"ANSWER_TEXT:{index}:{qi}:{ai}",
                )
    return value


def serialize_gold_evidence(example: Mapping[str, Any]) -> tuple[str, int, int]:
    blocks: list[str] = []
    question_count = 0
    answer_count = 0

    questions = example["questions"]
    require(isinstance(questions, list), "QUESTIONS_TYPE")
    for question_obj in questions:
        question = str(question_obj["question"])
        answers = question_obj["answers"]
        require(isinstance(answers, list), "ANSWERS_TYPE")
        question_count += 1
        for answer_obj in answers:
            answer = str(answer_obj["answer"])
            blocks.append(f"Question: {question}\nAnswer: {answer}")
            answer_count += 1

    require(question_count >= 1, "NO_QUESTIONS")
    require(answer_count >= 1, "NO_ANSWERS")
    evidence = "\n\n".join(blocks)
    require(bool(evidence.strip()), "EMPTY_EVIDENCE")
    return evidence, question_count, answer_count


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = [int(value) for value in encoded.ids]
    return ids


def gate_row(
    *,
    tokenizer: Any,
    source_index: int,
    example: Mapping[str, Any],
) -> dict[str, Any]:
    source_label = str(example["label"])
    require(source_label in LABEL_MAP, f"INCOMPATIBLE_LABEL:{source_label}")
    mapped = LABEL_MAP[source_label]

    claim = str(example["claim"])
    evidence, question_count, answer_count = serialize_gold_evidence(example)

    claim_ids = _token_ids(tokenizer, claim)
    evidence_ids = _token_ids(tokenizer, evidence)
    consumed_claim = claim_ids[:CLAIM_BUDGET]
    consumed_evidence = evidence_ids[:EVIDENCE_BUDGET]

    boundary_index = len(consumed_claim)
    target_index = boundary_index + TARGET_OFFSET
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
    if serialized_length > MAX_LENGTH:
        gate_reasons.append("SERIALIZED_LENGTH_EXCEEDS_128")
    if not (0 <= boundary_index < serialized_length):
        gate_reasons.append("BOUNDARY_OUTSIDE_ATTENDED_SEQUENCE")
    if not (0 <= target_index < serialized_length):
        gate_reasons.append("TARGET_OUTSIDE_ATTENDED_SEQUENCE")
    if target_index != len(consumed_claim) + 2:
        gate_reasons.append("TARGET_OFFSET_DRIFT")

    row = {
        "schema_version": COHORT_SCHEMA,
        "averitec_dev_index": int(source_index),
        "example_id": f"averitec_dev_{source_index:03d}",
        "claim": claim,
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
        "claim_truncated": len(claim_ids) > CLAIM_BUDGET,
        "evidence_truncated": len(evidence_ids) > EVIDENCE_BUDGET,
        "anchor_name": ANCHOR_NAME,
        "absolute_anchor_token_index": int(boundary_index),
        "target_intervention_token_index": int(target_index),
        "serialized_attended_length": int(serialized_length),
        "target_offset": TARGET_OFFSET,
        "token_gate_pass": not gate_reasons,
        "token_gate_reasons": gate_reasons,
    }
    require(not (FORBIDDEN_RESPONSE_FIELDS & set(row)), "RESPONSE_FIELD_PRESENT")
    return row


def compatible_examples(
    source: Sequence[Mapping[str, Any]],
) -> list[tuple[int, Mapping[str, Any]]]:
    compatible = [
        (index, row)
        for index, row in enumerate(source)
        if str(row["label"]) in COMPATIBLE_LABELS
    ]
    require(
        len(compatible) == EXPECTED_COMPATIBLE_COUNT,
        f"COMPATIBLE_COUNT:{len(compatible)}",
    )
    excluded = [
        row
        for row in source
        if str(row["label"]) == EXCLUDED_LABEL
    ]
    require(len(excluded) == EXPECTED_LABEL_COUNTS[EXCLUDED_LABEL], "EXCLUDED_COUNT")
    return compatible


def build_rows(
    *,
    source: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    rows = [
        gate_row(
            tokenizer=tokenizer,
            source_index=index,
            example=example,
        )
        for index, example in compatible_examples(source)
    ]
    require(len(rows) == EXPECTED_COMPATIBLE_COUNT, "BUILT_ROW_COUNT")
    require(
        len({row["example_id"] for row in rows}) == EXPECTED_COMPATIBLE_COUNT,
        "EXAMPLE_ID_UNIQUENESS",
    )
    return rows


def validate_tokenizer_files(snapshot: Path) -> dict[str, str]:
    require(snapshot.is_dir(), f"TOKENIZER_SNAPSHOT_MISSING:{snapshot}")
    observed: dict[str, str] = {}
    for filename, expected in TOKENIZER_FILE_SHA256.items():
        path = snapshot / filename
        require(path.is_file(), f"TOKENIZER_FILE_MISSING:{filename}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == expected, f"TOKENIZER_SHA256:{filename}:{digest}")
        observed[filename] = digest
    return observed


def provision_tokenizer_snapshot() -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise AVeriTeCGateError("HUGGINGFACE_HUB_REQUIRED") from exc

    path = Path(
        snapshot_download(
            repo_id=HF_REPO,
            revision=HF_REVISION,
            allow_patterns=sorted(TOKENIZER_FILE_SHA256),
        )
    )
    validate_tokenizer_files(path)
    return path


def load_tokenizer(snapshot: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import tokenizers
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise AVeriTeCGateError("TOKENIZERS_PACKAGE_REQUIRED") from exc

    require(
        tokenizers.__version__ == TOKENIZERS_VERSION,
        f"TOKENIZERS_VERSION:{tokenizers.__version__}",
    )
    observed_hashes = validate_tokenizer_files(snapshot)

    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()

    require(
        tokenizer.token_to_id("<|endoftext|>") == EOS_TOKEN_ID,
        "EOS_TOKEN_ID",
    )

    return tokenizer, {
        "repo": HF_REPO,
        "revision": HF_REVISION,
        "tokenizers_version": TOKENIZERS_VERSION,
        "file_sha256": observed_hashes,
        "eos_token_id": EOS_TOKEN_ID,
        "active_serialization": "claim[:63]+EOS(0)+evidence[:64]",
        "max_length": MAX_LENGTH,
        "claim_budget": CLAIM_BUDGET,
        "evidence_budget": EVIDENCE_BUDGET,
    }


def summarize_lengths(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [int(row[key]) for row in rows]
    require(values, f"EMPTY_LENGTHS:{key}")
    ordered = sorted(values)
    n = len(ordered)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / n,
        "median": (
            ordered[n // 2]
            if n % 2
            else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
        ),
    }


def build_manifest(
    *,
    source_raw: bytes,
    rows: Sequence[Mapping[str, Any]],
    tokenizer_provenance: Mapping[str, Any],
    cohort_sha256: str,
) -> dict[str, Any]:
    gate_pass_count = sum(bool(row["token_gate_pass"]) for row in rows)
    gate_fail_count = len(rows) - gate_pass_count
    failure_counts = Counter(
        reason
        for row in rows
        for reason in row["token_gate_reasons"]
    )
    mapped_counts = Counter(str(row["source_label"]) for row in rows)
    require(
        dict(mapped_counts)
        == {
            "Refuted": 305,
            "Supported": 122,
            "Not Enough Evidence": 35,
        },
        f"COMPATIBLE_LABEL_COUNTS:{dict(mapped_counts)}",
    )

    result = (
        PASS_RESULT
        if gate_pass_count == EXPECTED_COMPATIBLE_COUNT
        else BLOCK_RESULT
    )

    return {
        "schema_version": MANIFEST_SCHEMA,
        "result": result,
        "phase": "EXPERIMENT_2_AVERITEC_130M370M_RESPONSE_BLIND_TOKEN_GATE",
        "upstream": {
            "repository": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "path": UPSTREAM_PATH,
            "git_blob_sha1": UPSTREAM_BLOB_SHA1,
            "byte_count": len(source_raw),
            "sha256": sha256_bytes(source_raw),
            "license": LICENSE,
            "paper_citation": PAPER_CITATION,
        },
        "source_example_count": EXPECTED_SOURCE_EXAMPLES,
        "source_label_counts": EXPECTED_LABEL_COUNTS,
        "compatible_labels": list(COMPATIBLE_LABELS),
        "excluded_label": EXCLUDED_LABEL,
        "compatible_count": len(rows),
        "compatible_label_counts": dict(mapped_counts),
        "primary_external_transfer_family": {
            "scale_order": list(PRIMARY_SCALES),
            "primary_p_value_count_planned":
                PRIMARY_P_VALUE_COUNT_PLANNED,
            "multiplicity": PRIMARY_MULTIPLICITY,
            "mamba14b_in_family": False,
            "historical_behavioral_p_values_in_family": 0,
        },
        "frozen_causal_objects": FROZEN_CAUSAL_OBJECTS,
        "shared_tokenizer_bytes_across_primary_scales":
            SHARED_TOKENIZER_BYTES_ACROSS_PRIMARY_SCALES,
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
            "anchor_name": ANCHOR_NAME,
            "anchor_definition": "EOS separator after consumed claim tokens",
            "target_offset": TARGET_OFFSET,
            "target_definition": "second consumed evidence token",
            "semantic_identity_with_xg1_A_IDENTITY_claimed": False,
        },
        "token_gate": {
            "target_count": EXPECTED_COMPATIBLE_COUNT,
            "pass_count": gate_pass_count,
            "fail_count": gate_fail_count,
            "failure_counts": dict(sorted(failure_counts.items())),
        },
        "truncation": {
            "claim_truncated_count": sum(bool(row["claim_truncated"]) for row in rows),
            "evidence_truncated_count": sum(bool(row["evidence_truncated"]) for row in rows),
            "claim_raw_tokens": summarize_lengths(rows, "claim_raw_token_count"),
            "claim_consumed_tokens": summarize_lengths(rows, "claim_consumed_token_count"),
            "evidence_raw_tokens": summarize_lengths(rows, "evidence_raw_token_count"),
            "evidence_consumed_tokens": summarize_lengths(rows, "evidence_consumed_token_count"),
            "anchor_index": summarize_lengths(rows, "absolute_anchor_token_index"),
            "target_index": summarize_lengths(rows, "target_intervention_token_index"),
        },
        "model_checkpoint_loaded": False,
        "model_forward_count": 0,
        "cuda_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "scientific_inference_executed": False,
        "p_value_count_added": 0,
        "response_fields_present": False,
        "execution_authorized_by_this_artifact": False,
    }


def write_outputs(
    *,
    output_dir: Path,
    source_raw: bytes,
    rows: Sequence[Mapping[str, Any]],
    tokenizer_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    cohort_raw = jsonl_bytes(rows)
    cohort_sha = sha256_bytes(cohort_raw)
    manifest = build_manifest(
        source_raw=source_raw,
        rows=rows,
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
    source_raw: bytes | None = None,
    tokenizer_snapshot: Path | None = None,
) -> dict[str, Any]:
    authenticate_repo()

    if source_raw is None:
        source_raw = download_pinned_source()
    source = parse_source(source_raw)

    if tokenizer_snapshot is None:
        tokenizer_snapshot = provision_tokenizer_snapshot()
    tokenizer, tokenizer_provenance = load_tokenizer(tokenizer_snapshot)

    rows_a = build_rows(source=source, tokenizer=tokenizer)
    rows_b = build_rows(source=source, tokenizer=tokenizer)
    require(jsonl_bytes(rows_a) == jsonl_bytes(rows_b), "DETERMINISTIC_REGENERATION")

    manifest = write_outputs(
        output_dir=output_dir,
        source_raw=source_raw,
        rows=rows_a,
        tokenizer_provenance=tokenizer_provenance,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Provision pinned AVeriTeC dev, derive the frozen three-label "
            "gold-evidence cohort, and execute the shared response-blind "
            "130M/370M claim/evidence-boundary tokenizer gate. "
            "No model load or forward."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / OUTPUT_DIR,
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        default=None,
        help=(
            "Optional already-provisioned pinned Mamba-370M tokenizer snapshot. "
            "If omitted, only tokenizer files are downloaded from the exact "
            "frozen revision."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = materialize(
        output_dir=args.output_dir,
        tokenizer_snapshot=args.tokenizer_snapshot,
    )

    print("RESULT=" + str(manifest["result"]))
    print("UPSTREAM_COMMIT=" + UPSTREAM_COMMIT)
    print("UPSTREAM_BLOB_SHA1=" + UPSTREAM_BLOB_SHA1)
    print("UPSTREAM_SHA256=" + str(manifest["upstream"]["sha256"]))
    print("SOURCE_EXAMPLES=500")
    print("COMPATIBLE_COUNT=" + str(manifest["compatible_count"]))
    print("PRIMARY_SCALES=mamba130m,mamba370m")
    print("PRIMARY_P_VALUE_COUNT_PLANNED=2")
    print("PRIMARY_MULTIPLICITY=holm")
    print("MAMBA14B_IN_FAMILY=False")
    print("TOKEN_GATE_PASS_COUNT=" + str(manifest["token_gate"]["pass_count"]))
    print("TOKEN_GATE_FAIL_COUNT=" + str(manifest["token_gate"]["fail_count"]))
    print(
        "CLAIM_TRUNCATED_COUNT="
        + str(manifest["truncation"]["claim_truncated_count"])
    )
    print(
        "EVIDENCE_TRUNCATED_COUNT="
        + str(manifest["truncation"]["evidence_truncated_count"])
    )
    print("ANCHOR=A_CLAIM_EVIDENCE_BOUNDARY")
    print("TARGET_OFFSET=2")
    print("MODEL_CHECKPOINT_LOADED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("P_VALUE_COUNT_ADDED=0")

    return 0 if manifest["result"] == PASS_RESULT else 2


if __name__ == "__main__":
    raise SystemExit(main())
