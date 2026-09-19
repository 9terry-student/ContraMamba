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

from scripts import build_reason_router_gen4_mamba14b_xg1_holdouts as prior


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "74ccb17ef168176f61b7c1946178dbadc7a1b685"

FIRST_PAIR = 4801
LAST_PAIR = 5100
PAIR_COUNT = 300
ROWS_PER_PAIR = 6
ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR

OUTPUT_DIR = Path(
    "data/reason_router_gen4_mamba370m14b_xg1_behavioral_bridge_v1"
)
SOURCE_FILE = prior.SOURCE_FILE
ROW_FILE = prior.ROW_FILE
MANIFEST_FILE = "structural_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
MANIFEST_SCHEMA = "GEN4_MAMBA370M14B_BEHAVIORAL_BRIDGE_XG1_STRUCTURAL_V1"
RESULT = "PASS_MAMBA370M14B_BEHAVIORAL_BRIDGE_XG1_4801_5100_STRUCTURAL"

FROZEN_TAIL_DIR = prior.RESIDUAL_DIR
FROZEN_TAIL_FIRST = prior.RESIDUAL_FIRST
FROZEN_TAIL_LAST = prior.RESIDUAL_LAST

LABEL_CONTRACT = {
    "C0_SHAM": {"label": "SUPPORT", "label_id": 2},
    "C2_NAME": {"label": "NOT_ENTITLED", "label_id": 1},
}
SCALE_ORDER = ("mamba370m", "mamba14b")
PRIMARY_FAMILY = {
    "test": "one_sample_student_t",
    "alternative": "greater",
    "family_alpha": 0.05,
    "multiplicity": "holm",
    "primary_p_value_count": 2,
    "n_per_scale": 300,
    "endpoint": "D_BEH=M_dominant_restored-M_dominant_control",
}

FORBIDDEN_OUTCOME_FIELDS = prior.FORBIDDEN_OUTCOME_FIELDS


class BehavioralHoldoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise BehavioralHoldoutError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BehavioralHoldoutError(
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
    for path in (
        "scripts/build_reason_router_gen4_xg1_cross_generator_cohort.py",
        "scripts/build_reason_router_gen4_seed181_behavioral_bridge_holdout.py",
        "scripts/build_reason_router_gen4_mamba370m_xg1_holdouts.py",
        "scripts/build_reason_router_gen4_mamba370m_xg1_residual_decomposition_holdout.py",
        "scripts/build_reason_router_gen4_mamba14b_xg1_holdouts.py",
        FROZEN_TAIL_DIR.as_posix(),
    ):
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
    return prior.m370.prior.jsonl_bytes(rows)


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
    return prior.expected_pair_ids(first, last)


def build_population(
    first: int = FIRST_PAIR,
    last: int = LAST_PAIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts, rows = prior.m370.build_population(first, last)
    require(
        [str(row["pair_id"]) for row in facts]
        == expected_pair_ids(first, last),
        "PAIR_ID_SEQUENCE",
    )
    require(
        len(facts) == last - first + 1,
        "SOURCE_COUNT",
    )
    require(
        len(rows) == len(facts) * ROWS_PER_PAIR,
        "ROW_COUNT",
    )
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


def validate_semantic_labels(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    facts_by_id = {str(row["pair_id"]): row for row in facts}
    by_key = {
        (str(row["source_pair_id"]), str(row["contrast_cell_id"])): row
        for row in rows
    }
    base = prior.m370.prior.base

    for pair_id, fact in facts_by_id.items():
        c0 = by_key[(pair_id, "C0_SHAM")]
        c2 = by_key[(pair_id, "C2_NAME")]

        require(
            c0["claim"] == c0["evidence"],
            f"C0_SUPPORT_SEMANTICS:{pair_id}",
        )
        require(
            c2["claim"] == c0["claim"],
            f"C2_CLAIM_IDENTITY:{pair_id}",
        )
        require(
            c2["evidence"]
            == base.render_statement(
                fact,
                name=str(fact["alternate_name"]),
            ),
            f"C2_NAME_SWAP:{pair_id}",
        )
        require(
            str(fact["name"]) != str(fact["alternate_name"]),
            f"C2_NAME_DIFFERENCE:{pair_id}",
        )


def verify_frozen_tail_regeneration() -> None:
    facts, rows = build_population(
        FROZEN_TAIL_FIRST,
        FROZEN_TAIL_LAST,
    )
    source_rel = FROZEN_TAIL_DIR / SOURCE_FILE
    row_rel = FROZEN_TAIL_DIR / ROW_FILE

    source_blob = prior.m370.prior.git_blob_bytes(source_rel)
    row_blob = prior.m370.prior.git_blob_bytes(row_rel)

    require(
        jsonl_bytes(facts) == source_blob,
        "FROZEN_TAIL_SOURCE_REGENERATION",
    )
    require(
        jsonl_bytes(rows) == row_blob,
        "FROZEN_TAIL_ROW_REGENERATION",
    )


def prior_inventory_through_4800(
) -> tuple[set[str], set[str], set[str]]:
    pair_ids: set[str] = set()
    claims: set[str] = set()
    evidence: set[str] = set()

    for first in range(1, 4801, 300):
        last = first + 299
        facts, rows = prior.m370.build_population(first, last)
        pairs = {str(row["pair_id"]) for row in facts}
        row_claims = {str(row["claim"]) for row in rows}
        row_evidence = {str(row["evidence"]) for row in rows}

        require(not (pair_ids & pairs), f"PRIOR_PAIR_OVERLAP:{first}")
        require(not (claims & row_claims), f"PRIOR_CLAIM_OVERLAP:{first}")
        require(not (evidence & row_evidence), f"PRIOR_EVIDENCE_OVERLAP:{first}")

        pair_ids |= pairs
        claims |= row_claims
        evidence |= row_evidence

    require(
        pair_ids == set(prior.expected_pair_ids(1, 4800)),
        "PRIOR_PAIR_COVERAGE_001_4800",
    )
    return pair_ids, claims, evidence


def build_payload() -> dict[str, Any]:
    verify_frozen_tail_regeneration()

    prior_pairs, prior_claims, prior_evidence = (
        prior_inventory_through_4800()
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

    validate_semantic_labels(facts_a, rows_a)

    pairs = {str(row["pair_id"]) for row in facts_a}
    claims = {str(row["claim"]) for row in rows_a}
    evidence = {str(row["evidence"]) for row in rows_a}

    require(not (pairs & prior_pairs), "PAIR_OVERLAP_WITH_001_4800")
    require(not (claims & prior_claims), "CLAIM_OVERLAP_WITH_001_4800")
    require(
        not (evidence & prior_evidence),
        "EVIDENCE_OVERLAP_WITH_001_4800",
    )

    source_raw = jsonl_bytes(facts_a)
    row_raw = jsonl_bytes(rows_a)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT,
        "role": "cross_scale_behavioral_bridge",
        "generator_family": prior.m370.prior.base.GENERATOR_FAMILY,
        "deterministic_generator_semantics": True,
        "pair_id_first": f"xg1_fact_{FIRST_PAIR:03d}",
        "pair_id_last": f"xg1_fact_{LAST_PAIR:03d}",
        "source_pair_count": PAIR_COUNT,
        "row_count": ROW_COUNT,
        "rows_per_pair": ROWS_PER_PAIR,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(row_raw),
        "prior_pair_range": "xg1_fact_001..xg1_fact_4800",
        "pair_id_overlap_with_prior": 0,
        "claim_overlap_with_prior": 0,
        "evidence_overlap_with_prior": 0,
        "shared_population_across_scales": True,
        "scale_order": list(SCALE_ORDER),
        "behavioral_target_cells": ["C0_SHAM", "C2_NAME"],
        "behavioral_label_contract": LABEL_CONTRACT,
        "labels_embedded_in_structural_rows": False,
        "conditions_planned": [
            "native",
            "dominant_neutralized",
            "dominant_restored",
            "dominant_control",
        ],
        "primary_family_planned": PRIMARY_FAMILY,
        "selection_allowed": False,
        "scale_specific_rescue_cohort_allowed": False,
        "response_fields_present": False,
        "endpoint_values_present": False,
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
    return prior.m370.prior.read_jsonl(path)


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
    prior.m370.prior.base.validate_materialized_rows(
        rows,
        expected_pairs=PAIR_COUNT,
    )
    validate_semantic_labels(facts, rows)

    require(manifest["schema_version"] == MANIFEST_SCHEMA, "MANIFEST_SCHEMA")
    require(manifest["result"] == RESULT, "MANIFEST_RESULT")
    require(
        manifest["pair_id_first"] == "xg1_fact_4801"
        and manifest["pair_id_last"] == "xg1_fact_5100",
        "MANIFEST_RANGE",
    )
    require(
        manifest["shared_population_across_scales"] is True,
        "MANIFEST_SHARED_POPULATION",
    )
    require(
        manifest["primary_family_planned"] == PRIMARY_FAMILY,
        "MANIFEST_PRIMARY_FAMILY",
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
        "tokenizer_executed",
        "checkpoint_loaded",
        "model_executed",
        "cuda_executed",
    ):
        require(manifest[key] is False, f"OUTCOME_BLINDNESS:{key}")

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
            "Materialize the shared response-blind XG1 4801..5100 "
            "behavioral-bridge population for Mamba-370M and Mamba-1.4B."
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
    print("SHARED_POPULATION_ACROSS_SCALES=True")
    print("PRIMARY_P_VALUE_COUNT_PLANNED=2")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
