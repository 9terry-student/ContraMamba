#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    build_reason_router_gen4_mamba370m14b_behavioral_bridge_holdout
    as previous,
)

ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "3ee2d7600de649a711752b8293687047c3c5ec4e"

FIRST_PAIR = 5101
LAST_PAIR = 5400
PAIR_COUNT = 300
ROWS_PER_PAIR = 6
ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR

OUTPUT_DIR = Path(
    "data/reason_router_gen4_mamba14b_xg1_adjacent_site_specificity_v1"
)
SOURCE_FILE = previous.SOURCE_FILE
ROW_FILE = previous.ROW_FILE
MANIFEST_FILE = "structural_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

MANIFEST_SCHEMA = "GEN4_MAMBA14B_XG1_ADJACENT_SITE_SPECIFICITY_STRUCTURAL_V1"
RESULT = "PASS_MAMBA14B_XG1_ADJACENT_SITE_SPECIFICITY_5101_5400_STRUCTURAL"

PREVIOUS_DIR = previous.OUTPUT_DIR
PREVIOUS_FIRST = previous.FIRST_PAIR
PREVIOUS_LAST = previous.LAST_PAIR

CANONICAL_TRIPLET = (33, 34, 35)
ADJACENT_TRIPLET = (34, 35, 36)
SELECTED_CAUSAL_CANDIDATE = "P5"
EPSILON = 0.025
ANCHOR_NAME = "A_IDENTITY"
TARGET_OFFSET = 2

FORBIDDEN_OUTCOME_FIELDS = previous.FORBIDDEN_OUTCOME_FIELDS


class AdjacentSpecificityHoldoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AdjacentSpecificityHoldoutError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AdjacentSpecificityHoldoutError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def authenticate_repo() -> None:
    require(
        git("branch", "--show-current") == EXPECTED_BRANCH,
        "BRANCH_MISMATCH",
    )
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            REQUIRED_ANCESTOR,
            "HEAD",
        )
        == 0,
        "REQUIRED_ANCESTOR_MISSING",
    )

    frozen = (
        "scripts/build_reason_router_gen4_mamba14b_xg1_holdouts.py",
        "scripts/build_reason_router_gen4_mamba370m14b_behavioral_bridge_holdout.py",
        PREVIOUS_DIR.as_posix(),
        "reports/reason_router_gen4_one_shot_adjacent_site_specificity_design.md",
    )
    for path in frozen:
        require(
            git_rc(
                "diff",
                "--quiet",
                REQUIRED_ANCESTOR,
                "HEAD",
                "--",
                path,
            )
            == 0,
            f"FROZEN_DEPENDENCY_COMMIT_DRIFT:{path}",
        )
        require(
            git_rc("diff", "--quiet", "--", path) == 0,
            f"FROZEN_DEPENDENCY_WORKTREE_DRIFT:{path}",
        )
        require(
            git_rc("diff", "--cached", "--quiet", "--", path) == 0,
            f"FROZEN_DEPENDENCY_INDEX_DRIFT:{path}",
        )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return previous.jsonl_bytes(rows)


def canonical_manifest_bytes(value: Mapping[str, Any]) -> bytes:
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


def expected_pair_ids(
    first: int = FIRST_PAIR,
    last: int = LAST_PAIR,
) -> list[str]:
    return previous.expected_pair_ids(first, last)


def build_population(
    first: int = FIRST_PAIR,
    last: int = LAST_PAIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts, rows = previous.build_population(first, last)
    require(
        [str(row["pair_id"]) for row in facts]
        == expected_pair_ids(first, last),
        "PAIR_ID_SEQUENCE",
    )
    require(len(facts) == last - first + 1, "SOURCE_COUNT")
    require(len(rows) == len(facts) * ROWS_PER_PAIR, "ROW_COUNT")
    for fact in facts:
        require(
            not (FORBIDDEN_OUTCOME_FIELDS & set(fact)),
            f"SOURCE_OUTCOME_FIELD:{fact['pair_id']}",
        )
    for row in rows:
        require(
            not (FORBIDDEN_OUTCOME_FIELDS & set(row)),
            f"ROW_OUTCOME_FIELD:{row['row_id']}",
        )
    return facts, rows


def _parse_jsonl_bytes(raw: bytes, label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"{label}:OBJECT:{line_no}")
        out.append(value)
    return out


def prior_inventory_through_5100(
) -> tuple[set[str], set[str], set[str]]:
    pair_ids, claims, evidence = previous.prior_inventory_through_4800()

    source_rel = PREVIOUS_DIR / SOURCE_FILE
    rows_rel = PREVIOUS_DIR / ROW_FILE
    manifest_rel = PREVIOUS_DIR / previous.MANIFEST_FILE

    source_raw = previous.prior.m370.prior.git_blob_bytes(source_rel)
    rows_raw = previous.prior.m370.prior.git_blob_bytes(rows_rel)
    manifest_raw = previous.prior.m370.prior.git_blob_bytes(manifest_rel)

    manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    require(
        manifest["result"] == previous.RESULT,
        "PREVIOUS_MANIFEST_RESULT",
    )
    require(
        manifest["pair_id_first"] == "xg1_fact_4801"
        and manifest["pair_id_last"] == "xg1_fact_5100",
        "PREVIOUS_MANIFEST_RANGE",
    )
    require(
        manifest["source_file_sha256"] == sha256_bytes(source_raw),
        "PREVIOUS_SOURCE_SHA",
    )
    require(
        manifest["row_file_sha256"] == sha256_bytes(rows_raw),
        "PREVIOUS_ROWS_SHA",
    )

    regen_facts, regen_rows = build_population(
        PREVIOUS_FIRST,
        PREVIOUS_LAST,
    )
    require(
        jsonl_bytes(regen_facts) == source_raw,
        "PREVIOUS_SOURCE_REGENERATION",
    )
    require(
        jsonl_bytes(regen_rows) == rows_raw,
        "PREVIOUS_ROWS_REGENERATION",
    )

    frozen_facts = _parse_jsonl_bytes(source_raw, "PREVIOUS_SOURCE")
    frozen_rows = _parse_jsonl_bytes(rows_raw, "PREVIOUS_ROWS")

    new_pairs = {str(row["pair_id"]) for row in frozen_facts}
    new_claims = {str(row["claim"]) for row in frozen_rows}
    new_evidence = {str(row["evidence"]) for row in frozen_rows}

    require(not (pair_ids & new_pairs), "PREVIOUS_PAIR_OVERLAP")
    require(not (claims & new_claims), "PREVIOUS_CLAIM_OVERLAP")
    require(not (evidence & new_evidence), "PREVIOUS_EVIDENCE_OVERLAP")

    pair_ids |= new_pairs
    claims |= new_claims
    evidence |= new_evidence

    require(
        pair_ids == set(previous.expected_pair_ids(1, 5100)),
        "PRIOR_PAIR_COVERAGE_001_5100",
    )
    return pair_ids, claims, evidence


def build_payload() -> dict[str, Any]:
    prior_pairs, prior_claims, prior_evidence = (
        prior_inventory_through_5100()
    )

    facts_a, rows_a = build_population()
    facts_b, rows_b = build_population()

    require(
        jsonl_bytes(facts_a) == jsonl_bytes(facts_b),
        "SOURCE_REGENERATION",
    )
    require(
        jsonl_bytes(rows_a) == jsonl_bytes(rows_b),
        "ROW_REGENERATION",
    )
    previous.validate_semantic_labels(facts_a, rows_a)

    pairs = {str(row["pair_id"]) for row in facts_a}
    claims = {str(row["claim"]) for row in rows_a}
    evidence = {str(row["evidence"]) for row in rows_a}

    require(not (pairs & prior_pairs), "PAIR_OVERLAP_WITH_001_5100")
    require(not (claims & prior_claims), "CLAIM_OVERLAP_WITH_001_5100")
    require(
        not (evidence & prior_evidence),
        "EVIDENCE_OVERLAP_WITH_001_5100",
    )

    source_raw = jsonl_bytes(facts_a)
    row_raw = jsonl_bytes(rows_a)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT,
        "role": "one_shot_adjacent_site_specificity",
        "generator_family": previous.prior.m370.prior.base.GENERATOR_FAMILY,
        "deterministic_generator_semantics": True,
        "pair_id_first": "xg1_fact_5101",
        "pair_id_last": "xg1_fact_5400",
        "source_pair_count": PAIR_COUNT,
        "row_count": ROW_COUNT,
        "rows_per_pair": ROWS_PER_PAIR,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(row_raw),
        "prior_pair_range": "xg1_fact_001..xg1_fact_5100",
        "pair_id_overlap_with_prior": 0,
        "claim_overlap_with_prior": 0,
        "evidence_overlap_with_prior": 0,
        "canonical_triplet": list(CANONICAL_TRIPLET),
        "adjacent_triplet": list(ADJACENT_TRIPLET),
        "adjacent_shift": 1,
        "selected_causal_candidate_frozen": SELECTED_CAUSAL_CANDIDATE,
        "epsilon": EPSILON,
        "anchor_name": ANCHOR_NAME,
        "target_offset": TARGET_OFFSET,
        "labels_embedded_in_structural_rows": False,
        "response_fields_present": False,
        "endpoint_values_present": False,
        "selection_allowed": False,
        "layer_sweep_allowed": False,
        "second_adjacent_site_allowed": False,
        "token_sweep_allowed": False,
        "epsilon_sweep_allowed": False,
        "response_based_control_selection_allowed": False,
        "tokenizer_executed": False,
        "checkpoint_loaded": False,
        "model_executed": False,
        "cuda_executed": False,
    }
    return {
        "source": source_raw,
        "rows": row_raw,
        "manifest": manifest,
    }


def write_holdout(
    output_dir: Path = ROOT / OUTPUT_DIR,
) -> dict[str, Any]:
    authenticate_repo()
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")

    payload = build_payload()
    source_raw = bytes(payload["source"])
    row_raw = bytes(payload["rows"])
    manifest_raw = canonical_manifest_bytes(payload["manifest"])

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / SOURCE_FILE).write_bytes(source_raw)
    (output_dir / ROW_FILE).write_bytes(row_raw)
    (output_dir / MANIFEST_FILE).write_bytes(manifest_raw)

    hashes = {
        SOURCE_FILE: sha256_bytes(source_raw),
        ROW_FILE: sha256_bytes(row_raw),
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

    manifest = validate_written(output_dir)
    return {
        "result": RESULT,
        "manifest": manifest,
        "artifact_sha256": hashes,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return previous._read_jsonl(path)


def validate_written(output_dir: Path) -> dict[str, Any]:
    required = {
        SOURCE_FILE,
        ROW_FILE,
        MANIFEST_FILE,
        CHECKSUM_FILE,
    }
    require(output_dir.is_dir(), "OUTPUT_DIR_MISSING")
    require(
        {
            path.name
            for path in output_dir.iterdir()
            if path.is_file()
        }
        == required,
        "OUTPUT_FILE_SET",
    )

    source_raw = (output_dir / SOURCE_FILE).read_bytes()
    row_raw = (output_dir / ROW_FILE).read_bytes()
    manifest_raw = (output_dir / MANIFEST_FILE).read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))

    facts = _read_jsonl(output_dir / SOURCE_FILE)
    rows = _read_jsonl(output_dir / ROW_FILE)

    require(
        [str(row["pair_id"]) for row in facts]
        == expected_pair_ids(),
        "WRITTEN_PAIR_IDS",
    )
    previous.prior.m370.prior.base.validate_materialized_rows(
        rows,
        expected_pairs=PAIR_COUNT,
    )
    previous.validate_semantic_labels(facts, rows)

    require(manifest["schema_version"] == MANIFEST_SCHEMA, "MANIFEST_SCHEMA")
    require(manifest["result"] == RESULT, "MANIFEST_RESULT")
    require(
        manifest["pair_id_first"] == "xg1_fact_5101"
        and manifest["pair_id_last"] == "xg1_fact_5400",
        "MANIFEST_RANGE",
    )
    require(
        manifest["prior_pair_range"] == "xg1_fact_001..xg1_fact_5100",
        "MANIFEST_PRIOR_RANGE",
    )
    require(
        manifest["canonical_triplet"] == [33, 34, 35]
        and manifest["adjacent_triplet"] == [34, 35, 36],
        "MANIFEST_SITE_MAPPING",
    )
    require(
        manifest["selected_causal_candidate_frozen"] == "P5",
        "MANIFEST_SELECTED_CANDIDATE",
    )
    require(
        manifest["source_file_sha256"] == sha256_bytes(source_raw),
        "MANIFEST_SOURCE_SHA",
    )
    require(
        manifest["row_file_sha256"] == sha256_bytes(row_raw),
        "MANIFEST_ROWS_SHA",
    )

    for key in (
        "response_fields_present",
        "endpoint_values_present",
        "selection_allowed",
        "layer_sweep_allowed",
        "second_adjacent_site_allowed",
        "token_sweep_allowed",
        "epsilon_sweep_allowed",
        "response_based_control_selection_allowed",
        "tokenizer_executed",
        "checkpoint_loaded",
        "model_executed",
        "cuda_executed",
    ):
        require(manifest[key] is False, f"OUTCOME_BOUNDARY:{key}")

    expected_hashes = {
        SOURCE_FILE: sha256_bytes(source_raw),
        ROW_FILE: sha256_bytes(row_raw),
        MANIFEST_FILE: sha256_bytes(manifest_raw),
    }
    expected_sums = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(expected_hashes.items())
    )
    require(
        (output_dir / CHECKSUM_FILE).read_text(encoding="utf-8")
        == expected_sums,
        "CHECKSUMS",
    )
    return manifest


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the response-blind XG1 5101..5400 cohort for "
            "Mamba-1.4B one-shot adjacent-site specificity. Static only."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = write_holdout(args.output_dir)
    manifest = result["manifest"]
    print("RESULT=" + result["result"])
    print(
        "PAIR_RANGE="
        f"{manifest['pair_id_first']}..{manifest['pair_id_last']}"
    )
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("PAIR_OVERLAP_WITH_001_5100=0")
    print("CLAIM_OVERLAP_WITH_001_5100=0")
    print("EVIDENCE_OVERLAP_WITH_001_5100=0")
    print("CANONICAL_TRIPLET=33,34,35")
    print("ADJACENT_TRIPLET=34,35,36")
    print("SELECTED_CAUSAL_CANDIDATE=P5")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
