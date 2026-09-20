#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

DESIGN = Path(
    "reports/reason_router_gen4_precursor_v2_dynamic_causal_susceptibility_stage_a_design.md"
)
DESIGN_GIT_BLOB = "d24011d698de1a53436a6bb997b2bb400881a718"
DESIGN_SHA256 = "fa820362e3ad7da014ef22c838b30189fb12ec775296dee5b1ff68c5b3461946"

SOURCE_DIR = Path(
    "data/reason_router_gen4_averitec_train_fresh_steering_boundary_v1"
)
SOURCE_COHORT = SOURCE_DIR / "fresh_compatible_cohort.jsonl"
SOURCE_MANIFEST = SOURCE_DIR / "token_gate_manifest.json"
SOURCE_SUMS = SOURCE_DIR / "SHA256SUMS.txt"

SOURCE_COHORT_GIT_BLOB = "f6c8edcfa37c10feb7dfa66ecd36648bd1cc2bf1"
SOURCE_MANIFEST_GIT_BLOB = "c31b20880eac0db2c3330a977fa7a5f66a788ddd"
SOURCE_SUMS_GIT_BLOB = "6a480e21178de09a7e24415413626a6576479693"

SOURCE_COHORT_SHA256 = "ce271c1b57b33e12399bc180f06d417ceef6e4e92ad9e5383e2d8187829a2810"
SOURCE_MANIFEST_SHA256 = "7fef7f8c9544f46fbdeb1184bee313259bd52154181c346c4ae6d367d08da738"

SOURCE_N = 2799
SOURCE_LABEL_COUNTS = {
    "Refuted": 1727,
    "Supported": 805,
    "Not Enough Evidence": 267,
}

TARGET_N = 800
TARGET_PER_SOURCE_LABEL = {
    "Refuted": 400,
    "Supported": 400,
}
TARGET_PER_CORRECT_LABEL = {
    "REFUTE": 400,
    "SUPPORT": 400,
}
SELECTION_NAMESPACE = "CONTRAMAMBA_PRECURSOR_V2_DCS_STAGE_A_V1"

COHORT_SCHEMA = "GEN4_PRECURSOR_V2_DCS_STAGE_A_COHORT_V1"
MANIFEST_SCHEMA = "GEN4_PRECURSOR_V2_DCS_STAGE_A_COHORT_MANIFEST_V1"
PASS_RESULT = "PASS_PRECURSOR_V2_DCS_STAGE_A_RESPONSE_FREE_COHORT"

OUTPUT_DIR = Path("data/reason_router_gen4_precursor_v2_dcs_stage_a_cohort_v1")
COHORT_FILE = "precursor_v2_stage_a_cohort.jsonl"
MANIFEST_FILE = "cohort_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

FORBIDDEN_RESPONSE_FIELDS = {
    "native_prediction",
    "steering_prediction",
    "control_prediction",
    "native_correct",
    "steering_correct",
    "control_correct",
    "correction",
    "damage",
    "p_value",
    "p_raw",
    "p_holm",
    "margin_delta",
    "native_margin",
    "steering_margin",
    "control_margin",
    "future_outcome",
    "supported_forced_decisive_commitment",
    "unsupported_forced_decisive_commitment",
    "primary_stage_a_group",
    "emitted_commitment",
    "emitted_label",
    "generated_tokens",
    "generation_text",
    "p3_component_l2",
    "p3_a",
    "p3_b",
    "chi_p3",
    "chi_p5",
    "susceptibility",
    "dynamic_causal_susceptibility",
}

REQUIRED_SOURCE_FIELDS = {
    "schema_version",
    "averitec_train_index",
    "example_id",
    "claim",
    "normalized_claim",
    "original_claim_url",
    "evidence",
    "source_label",
    "correct_label",
    "correct_label_id",
    "question_count",
    "answer_count",
    "claim_raw_token_count",
    "claim_consumed_token_count",
    "evidence_raw_token_count",
    "evidence_consumed_token_count",
    "claim_truncated",
    "evidence_truncated",
    "anchor_name",
    "absolute_anchor_token_index",
    "target_intervention_token_index",
    "serialized_attended_length",
    "target_offset",
    "token_gate_pass",
    "token_gate_reasons",
}

class CohortError(RuntimeError):
    pass

def require(ok: bool, message: str) -> None:
    if not ok:
        raise CohortError(message)

def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CohortError("GIT_FAILURE:" + " ".join(args)) from exc

def git_blob(path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "cat-file", "blob", f"HEAD:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CohortError(f"GIT_BLOB:{path}") from exc

def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

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

def selection_rank_key(example_id: str) -> str:
    require(isinstance(example_id, str) and bool(example_id), "EXAMPLE_ID")
    raw = (SELECTION_NAMESPACE + "|" + example_id).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def authenticate_repo(expected_head: str) -> None:
    require(git("branch", "--show-current") in ("", EXPECTED_BRANCH), "BRANCH")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")

    pinned = {
        DESIGN: DESIGN_GIT_BLOB,
        SOURCE_COHORT: SOURCE_COHORT_GIT_BLOB,
        SOURCE_MANIFEST: SOURCE_MANIFEST_GIT_BLOB,
        SOURCE_SUMS: SOURCE_SUMS_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path.as_posix()}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )

def validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    require(
        manifest["schema_version"]
        == "GEN4_AVERITEC_TRAIN_FRESH_STEERING_TOKEN_GATE_V1",
        "SOURCE_MANIFEST_SCHEMA",
    )
    require(manifest["result"] == "PASS_2799_OF_2799", "SOURCE_RESULT")
    require(manifest["fresh_cohort_count"] == SOURCE_N, "SOURCE_N")
    require(
        manifest["fresh_cohort_label_counts"] == SOURCE_LABEL_COUNTS,
        "SOURCE_LABEL_COUNTS",
    )
    require(
        manifest["cohort_file_sha256"] == SOURCE_COHORT_SHA256,
        "SOURCE_COHORT_SHA_IN_MANIFEST",
    )
    require(manifest["response_fields_present"] is False, "SOURCE_RESPONSE_FIELDS")
    require(manifest["model_checkpoint_loaded"] is False, "SOURCE_MODEL_LOAD")
    require(manifest["model_forward_count"] == 0, "SOURCE_MODEL_FORWARD")
    require(manifest["cuda_executed"] is False, "SOURCE_CUDA")
    require(manifest["training_executed"] is False, "SOURCE_TRAINING")
    require(manifest["backward_executed"] is False, "SOURCE_BACKWARD")
    require(
        manifest["scientific_inference_executed"] is False,
        "SOURCE_INFERENCE",
    )
    require(manifest["p_value_count_added"] == 0, "SOURCE_P_VALUE")

    freshness = manifest["freshness"]
    require(
        freshness["exclusion_counts"]
        == {
            "dev_claim_overlap": 6,
            "dev_url_overlap": 1,
            "empty_normalized_claim": 1,
            "incompatible_label": 195,
            "later_within_train_duplicate_claim": 66,
        },
        "SOURCE_FRESHNESS_EXCLUSIONS",
    )
    require(freshness["final_count"] == SOURCE_N, "SOURCE_FRESHNESS_N")
    require(
        freshness["final_label_counts"] == SOURCE_LABEL_COUNTS,
        "SOURCE_FRESHNESS_LABELS",
    )

    tok = manifest["tokenizer"]
    require(tok["repo"] == "state-spaces/mamba-370m-hf", "TOKENIZER_REPO")
    require(
        tok["revision"] == "589179554943157be31701edd8b4558889276674",
        "TOKENIZER_REVISION",
    )
    require(tok["tokenizers_version"] == "0.22.2", "TOKENIZERS_VERSION")
    require(
        tok["file_sha256"]
        == {
            "special_tokens_map.json": "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
            "tokenizer.json": "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf",
            "tokenizer_config.json": "9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb",
        },
        "TOKENIZER_BYTES",
    )

def load_source() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    design_raw = git_blob(DESIGN)
    cohort_raw = git_blob(SOURCE_COHORT)
    manifest_raw = git_blob(SOURCE_MANIFEST)
    sums_raw = git_blob(SOURCE_SUMS)

    require(sha256_bytes(design_raw) == DESIGN_SHA256, "DESIGN_SHA")
    require(sha256_bytes(cohort_raw) == SOURCE_COHORT_SHA256, "SOURCE_COHORT_SHA")
    require(
        sha256_bytes(manifest_raw) == SOURCE_MANIFEST_SHA256,
        "SOURCE_MANIFEST_SHA",
    )

    declared: dict[str, str] = {}
    for line in sums_raw.decode("utf-8").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            require(name not in declared, f"DUP_SUM:{name}")
            declared[name] = digest
    require(
        declared
        == {
            "fresh_compatible_cohort.jsonl": SOURCE_COHORT_SHA256,
            "token_gate_manifest.json": SOURCE_MANIFEST_SHA256,
        },
        "SOURCE_SUMS_CONTENT",
    )

    manifest = json.loads(manifest_raw.decode("utf-8"))
    validate_source_manifest(manifest)

    rows = [
        json.loads(line)
        for line in cohort_raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    require(len(rows) == SOURCE_N, "SOURCE_ROW_COUNT")
    return rows, manifest

def validate_source_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == SOURCE_N, "ROWS_N")
    require(
        len({str(row["example_id"]) for row in rows}) == SOURCE_N,
        "EXAMPLE_ID_UNIQUENESS",
    )
    require(
        len({str(row["normalized_claim"]) for row in rows}) == SOURCE_N,
        "NORMALIZED_CLAIM_UNIQUENESS",
    )

    counts = Counter(str(row["source_label"]) for row in rows)
    require(dict(counts) == SOURCE_LABEL_COUNTS, f"ROW_LABEL_COUNTS:{dict(counts)}")

    source_indices: list[int] = []
    for index, row in enumerate(rows):
        missing = REQUIRED_SOURCE_FIELDS - set(row)
        require(not missing, f"MISSING_FIELDS:{index}:{sorted(missing)}")
        require(
            row["schema_version"]
            == "GEN4_AVERITEC_TRAIN_FRESH_STEERING_COHORT_V1",
            f"SOURCE_ROW_SCHEMA:{index}",
        )
        require(
            not (FORBIDDEN_RESPONSE_FIELDS & set(row)),
            f"FORBIDDEN_RESPONSE_FIELD:{index}",
        )
        require(row["token_gate_pass"] is True, f"TOKEN_GATE:{index}")
        require(row["token_gate_reasons"] == [], f"TOKEN_GATE_REASON:{index}")
        require(
            row["anchor_name"] == "A_CLAIM_EVIDENCE_BOUNDARY",
            f"ANCHOR:{index}",
        )
        require(int(row["target_offset"]) == 2, f"TARGET_OFFSET:{index}")
        require(
            int(row["target_intervention_token_index"])
            == int(row["absolute_anchor_token_index"]) + 2,
            f"TARGET_INDEX:{index}",
        )
        require(bool(str(row["claim"])), f"CLAIM:{index}")
        require(bool(str(row["evidence"])), f"EVIDENCE:{index}")
        source_indices.append(int(row["averitec_train_index"]))

    require(source_indices == sorted(source_indices), "SOURCE_ORDER")

def select_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in rows
        if str(row["source_label"]) in TARGET_PER_SOURCE_LABEL
    ]
    selected: list[dict[str, Any]] = []
    for source_label, quota in TARGET_PER_SOURCE_LABEL.items():
        group = [row for row in eligible if row["source_label"] == source_label]
        ranked = sorted(
            (
                selection_rank_key(str(row["example_id"])),
                str(row["example_id"]),
                row,
            )
            for row in group
        )
        require(len(ranked) >= quota, f"QUOTA_AVAILABLE:{source_label}")
        for rank, (rank_key, _example_id, row) in enumerate(ranked[:quota]):
            out = dict(row)
            source_schema = str(out.pop("schema_version"))
            out["schema_version"] = COHORT_SCHEMA
            out["source_schema_version"] = source_schema
            out["precursor_v2_selection_namespace"] = SELECTION_NAMESPACE
            out["precursor_v2_rank_key_sha256"] = rank_key
            out["precursor_v2_rank_within_source_label_zero_based"] = rank
            out["precursor_v2_source_label_quota"] = quota
            out["fresh_for_precursor_v2_generation_response"] = True
            out["experiment3_response_accessed_for_selection"] = False
            out["old_precursor_response_accessed_for_selection"] = False
            require(
                not (FORBIDDEN_RESPONSE_FIELDS & set(out)),
                f"OUTPUT_RESPONSE_FIELD:{source_label}:{rank}",
            )
            selected.append(out)

    # Preserve deterministic upstream train order after response-blind quota selection.
    selected.sort(
        key=lambda row: (
            int(row["averitec_train_index"]),
            str(row["example_id"]),
        )
    )

    require(len(selected) == TARGET_N, "SELECTED_N")
    require(
        Counter(str(row["source_label"]) for row in selected)
        == Counter(TARGET_PER_SOURCE_LABEL),
        "SELECTED_SOURCE_LABELS",
    )
    require(
        Counter(str(row["correct_label"]) for row in selected)
        == Counter(TARGET_PER_CORRECT_LABEL),
        "SELECTED_CORRECT_LABELS",
    )
    require(
        len({str(row["example_id"]) for row in selected}) == TARGET_N,
        "SELECTED_ID_UNIQUENESS",
    )
    require(
        len({str(row["normalized_claim"]) for row in selected}) == TARGET_N,
        "SELECTED_CLAIM_UNIQUENESS",
    )
    return selected

def build_manifest(
    *,
    expected_head: str,
    rows: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    cohort_sha256: str,
) -> dict[str, Any]:
    source_counts = Counter(str(row["source_label"]) for row in rows)
    correct_counts = Counter(str(row["correct_label"]) for row in rows)
    require(dict(source_counts) == TARGET_PER_SOURCE_LABEL, "MANIFEST_SOURCE_COUNTS")
    require(dict(correct_counts) == TARGET_PER_CORRECT_LABEL, "MANIFEST_CORRECT_COUNTS")

    rank_keys = [str(row["precursor_v2_rank_key_sha256"]) for row in rows]
    require(len(set(rank_keys)) == TARGET_N, "RANK_KEY_UNIQUENESS")

    return {
        "schema_version": MANIFEST_SCHEMA,
        "result": PASS_RESULT,
        "phase": "PRECURSOR_V2_DCS_STAGE_A_RESPONSE_FREE_COHORT_FREEZE",
        "execution_head": expected_head,
        "design": {
            "path": DESIGN.as_posix(),
            "git_blob": DESIGN_GIT_BLOB,
            "sha256": DESIGN_SHA256,
        },
        "source": {
            "cohort_path": SOURCE_COHORT.as_posix(),
            "cohort_git_blob": SOURCE_COHORT_GIT_BLOB,
            "cohort_sha256": SOURCE_COHORT_SHA256,
            "manifest_path": SOURCE_MANIFEST.as_posix(),
            "manifest_git_blob": SOURCE_MANIFEST_GIT_BLOB,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "sums_path": SOURCE_SUMS.as_posix(),
            "sums_git_blob": SOURCE_SUMS_GIT_BLOB,
            "source_count": SOURCE_N,
            "source_label_counts": SOURCE_LABEL_COUNTS,
        },
        "selection": {
            "namespace": SELECTION_NAMESPACE,
            "rank_definition": (
                'SHA256("CONTRAMAMBA_PRECURSOR_V2_DCS_STAGE_A_V1|" + example_id)'
            ),
            "eligible_source_labels": ["Refuted", "Supported"],
            "excluded_source_label": "Not Enough Evidence",
            "quota_per_source_label": TARGET_PER_SOURCE_LABEL,
            "tie_break": "ascending example_id after ascending rank_key",
            "output_order": "ascending averitec_train_index then example_id",
            "response_fields_used": False,
            "experiment3_response_accessed": False,
            "old_precursor_response_accessed": False,
        },
        "cohort": {
            "count": TARGET_N,
            "source_label_counts": dict(source_counts),
            "correct_label_counts": dict(correct_counts),
            "cohort_file_sha256": cohort_sha256,
            "fresh_for_precursor_v2_generation_response": True,
            "all_token_gate_pass": all(bool(row["token_gate_pass"]) for row in rows),
        },
        "freshness": {
            "old_dev_precursor_overlap_claim": (
                "zero by inheritance from frozen source cohort full-dev normalized-claim "
                "and exact-nonempty-URL exclusion contract"
            ),
            "source_dev_claim_overlap_excluded_count":
                source_manifest["freshness"]["exclusion_counts"]["dev_claim_overlap"],
            "source_dev_url_overlap_excluded_count":
                source_manifest["freshness"]["exclusion_counts"]["dev_url_overlap"],
            "global_unseen_claimed": False,
            "train_source_previously_used_by_experiment3": True,
        },
        "tokenizer": dict(source_manifest["tokenizer"]),
        "response_fields_present": False,
        "model_checkpoint_loaded": False,
        "model_forward_count": 0,
        "cuda_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "scientific_inference_executed": False,
        "p_value_count_added": 0,
        "scientific_execution_authorized": False,
        "runner_implemented": False,
        "equivalence_executed": False,
    }

def write_outputs(
    *,
    output_dir: Path,
    expected_head: str,
    rows: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    cohort_raw = jsonl_bytes(rows)
    cohort_sha = sha256_bytes(cohort_raw)
    manifest = build_manifest(
        expected_head=expected_head,
        rows=rows,
        source_manifest=source_manifest,
        cohort_sha256=cohort_sha,
    )
    manifest_raw = pretty_json_bytes(manifest)

    (output_dir / COHORT_FILE).write_bytes(cohort_raw)
    (output_dir / MANIFEST_FILE).write_bytes(manifest_raw)

    hashes = {
        COHORT_FILE: cohort_sha,
        MANIFEST_FILE: sha256_bytes(manifest_raw),
    }
    (output_dir / SUMS_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )
    return manifest

def materialize(*, expected_head: str, output_dir: Path) -> dict[str, Any]:
    authenticate_repo(expected_head)
    source_rows, source_manifest = load_source()
    validate_source_rows(source_rows)

    selected_a = select_rows(source_rows)
    selected_b = select_rows(source_rows)
    require(
        jsonl_bytes(selected_a) == jsonl_bytes(selected_b),
        "DETERMINISTIC_REGENERATION",
    )

    return write_outputs(
        output_dir=output_dir,
        expected_head=expected_head,
        rows=selected_a,
        source_manifest=source_manifest,
    )

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the prospective response-free N=800 Precursor-v2 dynamic "
            "causal susceptibility Stage A cohort. No model load, CUDA, generation, "
            "scientific response, or inference."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT_DIR)
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = materialize(
        expected_head=str(args.expected_head),
        output_dir=args.output_dir,
    )
    print("RESULT=" + str(manifest["result"]))
    print("COHORT_COUNT=" + str(manifest["cohort"]["count"]))
    print("REFUTED_COUNT=" + str(manifest["cohort"]["source_label_counts"]["Refuted"]))
    print("SUPPORTED_COUNT=" + str(manifest["cohort"]["source_label_counts"]["Supported"]))
    print("NOT_ENOUGH_EVIDENCE_COUNT=0")
    print("SELECTION_NAMESPACE=" + SELECTION_NAMESPACE)
    print("FRESH_FOR_PRECURSOR_V2_GENERATION_RESPONSE=True")
    print("EXPERIMENT3_RESPONSE_ACCESSED=False")
    print("OLD_PRECURSOR_RESPONSE_ACCESSED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("P_VALUE_COUNT_ADDED=0")
    print("SCIENTIFIC_EXECUTION_AUTHORIZED=False")
    print("RUNNER_IMPLEMENTED=False")
    print("EQUIVALENCE_EXECUTED=False")
    print("COHORT_SHA256=" + str(manifest["cohort"]["cohort_file_sha256"]))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
