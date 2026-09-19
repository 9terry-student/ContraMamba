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

from scripts import build_reason_router_gen4_mamba370m_xg1_holdouts as prior


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "bbff1a0f348ef8e2c00a845cde2d99678ad6f92f"

FIRST = 3601
LAST = 3900
PAIR_COUNT = 300
ROWS_PER_PAIR = 6
EXPECTED_ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR

OUTPUT_DIR = Path(
    "data/reason_router_gen4_mamba370m_xg1_residual_decomposition_v1"
)

SOURCE_FILE = prior.SOURCE_FILE
ROW_FILE = prior.ROW_FILE
MANIFEST_FILE = "structural_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

SCHEMA = "GEN4_MAMBA370M_XG1_RESIDUAL_DECOMPOSITION_HOLDOUT_V1"
RESULT = "PASS_MAMBA370M_XG1_RESIDUAL_DECOMPOSITION_3601_3900_STRUCTURAL"

DISCOVERY_DIR = prior.DISCOVERY_DIR
CONFIRMATION_DIR = prior.CONFIRMATION_DIR

DISCOVERY_SOURCE_SHA256 = (
    "246b5c262c6d018a1e6047ae61ed2f7b3041154d234ea2817dbf5f7da120bc80"
)
DISCOVERY_ROWS_SHA256 = (
    "831452947cdf8136376a20ebeca873f260449bcebcd0f6b83c4361dae3a90697"
)
DISCOVERY_MANIFEST_SHA256 = (
    "28e5e5aaf8c17a803d8db9c45b5747c90c7fec883a59bd189a875e60fc0d475f"
)

CONFIRMATION_SOURCE_SHA256 = (
    "5827ee4f8b60717348ad588c0d4f29772b90d02bd2d13e36599042fcf5a9d4b0"
)
CONFIRMATION_ROWS_SHA256 = (
    "14e216a0de7105ae7cf7863577f773cf72b4c2344ee31160cf0d1e2defa04be0"
)
CONFIRMATION_MANIFEST_SHA256 = (
    "80bb2817d5a2489f28603f433db703d91429606e93053fc9cd948b215b74597c"
)

FORBIDDEN_OUTCOME_FIELDS = prior.FORBIDDEN_OUTCOME_FIELDS

CONDITIONS = (
    "native",
    "p1_neutralized",
    "p2_neutralized",
    "p4_neutralized",
    "p5_neutralized",
    "residual_all_neutralized",
)

RESIDUAL_PLANES = ("P1", "P2", "P4", "P5")


class ResidualDecompositionHoldoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ResidualDecompositionHoldoutError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ResidualDecompositionHoldoutError(
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

    # The deterministic XG1 generator chain itself must remain unchanged
    # relative to the already-frozen 370M holdout implementation.
    for path in prior.FROZEN_DEPENDENCIES:
        require(
            git_rc(
                "diff",
                "--quiet",
                prior.FROZEN_PARENT_COMMIT,
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


def canonical_manifest_bytes(
    manifest: Mapping[str, Any],
) -> bytes:
    return (
        json.dumps(
            dict(manifest),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def expected_pair_ids(
    first: int = FIRST,
    last: int = LAST,
) -> list[str]:
    return prior.expected_pair_ids(first, last)


def _read_frozen_population(
    directory: Path,
    *,
    first: int,
    last: int,
    source_sha: str,
    rows_sha: str,
    manifest_sha: str,
    label: str,
) -> tuple[set[str], set[str], set[str]]:
    source = ROOT / directory / SOURCE_FILE
    rows = ROOT / directory / ROW_FILE
    manifest = ROOT / directory / MANIFEST_FILE

    require(source.is_file(), f"{label}:SOURCE_MISSING")
    require(rows.is_file(), f"{label}:ROWS_MISSING")
    require(manifest.is_file(), f"{label}:MANIFEST_MISSING")

    source_raw = source.read_bytes()
    rows_raw = rows.read_bytes()
    manifest_raw = manifest.read_bytes()

    require(
        sha256_bytes(source_raw) == source_sha,
        f"{label}:SOURCE_SHA",
    )
    require(
        sha256_bytes(rows_raw) == rows_sha,
        f"{label}:ROWS_SHA",
    )
    require(
        sha256_bytes(manifest_raw) == manifest_sha,
        f"{label}:MANIFEST_SHA",
    )

    return prior._inventory_from_serialized(
        source_raw,
        rows_raw,
        first=first,
        last=last,
        label=label,
    )


def prior_inventory_through_3600(
) -> tuple[set[str], set[str], set[str]]:
    pairs, claims, evidence = prior.prior_inventory_through_3000()

    d_pairs, d_claims, d_evidence = _read_frozen_population(
        DISCOVERY_DIR,
        first=3001,
        last=3300,
        source_sha=DISCOVERY_SOURCE_SHA256,
        rows_sha=DISCOVERY_ROWS_SHA256,
        manifest_sha=DISCOVERY_MANIFEST_SHA256,
        label="MAMBA370_DISCOVERY_3001_3300",
    )

    require(not (pairs & d_pairs), "DISCOVERY_PAIR_OVERLAP")
    require(not (claims & d_claims), "DISCOVERY_CLAIM_OVERLAP")
    require(not (evidence & d_evidence), "DISCOVERY_EVIDENCE_OVERLAP")

    pairs |= d_pairs
    claims |= d_claims
    evidence |= d_evidence

    c_pairs, c_claims, c_evidence = _read_frozen_population(
        CONFIRMATION_DIR,
        first=3301,
        last=3600,
        source_sha=CONFIRMATION_SOURCE_SHA256,
        rows_sha=CONFIRMATION_ROWS_SHA256,
        manifest_sha=CONFIRMATION_MANIFEST_SHA256,
        label="MAMBA370_CONFIRMATION_3301_3600",
    )

    require(not (pairs & c_pairs), "CONFIRMATION_PAIR_OVERLAP")
    require(not (claims & c_claims), "CONFIRMATION_CLAIM_OVERLAP")
    require(not (evidence & c_evidence), "CONFIRMATION_EVIDENCE_OVERLAP")

    pairs |= c_pairs
    claims |= c_claims
    evidence |= c_evidence

    require(
        pairs == set(prior.expected_pair_ids(1, 3600)),
        "PRIOR_PAIR_COVERAGE_001_3600",
    )

    return pairs, claims, evidence


def build_population(
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts, rows = prior.build_population(FIRST, LAST)

    require(
        [str(row["pair_id"]) for row in facts]
        == expected_pair_ids(),
        "PAIR_ID_SEQUENCE",
    )
    require(len(facts) == PAIR_COUNT, "SOURCE_COUNT")
    require(len(rows) == EXPECTED_ROW_COUNT, "ROW_COUNT")

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


def build_payload() -> dict[str, Any]:
    prior_pairs, prior_claims, prior_evidence = (
        prior_inventory_through_3600()
    )

    facts_a, rows_a = build_population()
    facts_b, rows_b = build_population()

    source_a = prior.prior.jsonl_bytes(facts_a)
    source_b = prior.prior.jsonl_bytes(facts_b)
    rows_a_raw = prior.prior.jsonl_bytes(rows_a)
    rows_b_raw = prior.prior.jsonl_bytes(rows_b)

    require(source_a == source_b, "SOURCE_REGENERATION")
    require(rows_a_raw == rows_b_raw, "ROW_REGENERATION")

    pair_ids, claims, evidence = prior._sets_for_rows(
        facts_a,
        rows_a,
    )

    require(not (pair_ids & prior_pairs), "PAIR_OVERLAP_WITH_PRIOR")
    require(not (claims & prior_claims), "CLAIM_OVERLAP_WITH_PRIOR")
    require(not (evidence & prior_evidence), "EVIDENCE_OVERLAP_WITH_PRIOR")

    require(pair_ids == set(expected_pair_ids()), "PAIR_ID_SET")
    require(len(claims) > 0, "CLAIMS_EMPTY")
    require(len(evidence) > 0, "EVIDENCE_EMPTY")

    manifest = {
        "schema_version": SCHEMA,
        "result": RESULT,
        "role": "exploratory_residual_decomposition",
        "generator_family": prior.prior.base.GENERATOR_FAMILY,
        "deterministic_generator_semantics": True,
        "pair_id_first": f"xg1_fact_{FIRST:03d}",
        "pair_id_last": f"xg1_fact_{LAST:03d}",
        "source_pair_count": PAIR_COUNT,
        "row_count": EXPECTED_ROW_COUNT,
        "rows_per_pair": ROWS_PER_PAIR,
        "source_file_sha256": sha256_bytes(source_a),
        "row_file_sha256": sha256_bytes(rows_a_raw),
        "prior_pair_range": "xg1_fact_001..xg1_fact_3600",
        "pair_id_overlap_with_prior": 0,
        "claim_overlap_with_prior": 0,
        "evidence_overlap_with_prior": 0,
        "labels_present": False,
        "response_fields_present": False,
        "endpoint_values_present": False,
        "tokenizer_executed": False,
        "checkpoint_loaded": False,
        "model_executed": False,
        "cuda_executed": False,
        "prospective_use":
            "same_pair_residual_sign_reversal_decomposition_only",
        "scientific_question":
            "Is the Mamba-370M residual sign reversal approximately an "
            "additive signed mixture of P1/P2/P4/P5 effects, or does a "
            "material aggregate interaction remain on the same pairs?",
        "selected_dominant_candidate_frozen": "P3",
        "residual_planes_frozen": list(RESIDUAL_PLANES),
        "planned_condition_order": list(CONDITIONS),
        "planned_pair_level_endpoints": {
            "S_P1": "Q_native-Q_p1_neutralized",
            "S_P2": "Q_native-Q_p2_neutralized",
            "S_P4": "Q_native-Q_p4_neutralized",
            "S_P5": "Q_native-Q_p5_neutralized",
            "S_RES": "Q_native-Q_residual_all_neutralized",
            "I_RES":
                "S_RES-(S_P1+S_P2+S_P4+S_P5)",
        },
        "formal_inference_allowed": False,
        "p_value_count": 0,
        "selection_allowed": False,
        "rescue_of_failed_cross_backbone_claim": False,
        "layer_sweep_allowed": False,
        "token_sweep_allowed": False,
        "epsilon_sweep_allowed": False,
    }

    return {
        "source": source_a,
        "rows": rows_a_raw,
        "manifest": manifest,
    }


def _write_payload(
    output_dir: Path,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    require(
        not output_dir.exists(),
        f"OUTPUT_COLLISION:{output_dir}",
    )

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

    return hashes


def validate_written(
    output_dir: Path,
) -> dict[str, Any]:
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
    rows_raw = (output_dir / ROW_FILE).read_bytes()
    manifest_raw = (output_dir / MANIFEST_FILE).read_bytes()

    facts = prior.prior.read_jsonl_bytes(
        source_raw,
        "WRITTEN_SOURCE",
    )
    rows = prior.prior.read_jsonl_bytes(
        rows_raw,
        "WRITTEN_ROWS",
    )

    require(
        [str(row["pair_id"]) for row in facts]
        == expected_pair_ids(),
        "WRITTEN_PAIR_IDS",
    )

    prior.prior.base.validate_materialized_rows(
        rows,
        expected_pairs=PAIR_COUNT,
    )

    manifest = json.loads(manifest_raw.decode("utf-8"))

    require(manifest["schema_version"] == SCHEMA, "WRITTEN_SCHEMA")
    require(manifest["result"] == RESULT, "WRITTEN_RESULT")
    require(
        manifest["source_file_sha256"] == sha256_bytes(source_raw),
        "WRITTEN_SOURCE_SHA",
    )
    require(
        manifest["row_file_sha256"] == sha256_bytes(rows_raw),
        "WRITTEN_ROWS_SHA",
    )
    require(
        manifest["planned_condition_order"] == list(CONDITIONS),
        "WRITTEN_CONDITION_ORDER",
    )
    require(
        manifest["formal_inference_allowed"] is False
        and manifest["p_value_count"] == 0
        and manifest["selection_allowed"] is False
        and manifest["rescue_of_failed_cross_backbone_claim"] is False,
        "WRITTEN_SCIENCE_BOUNDARY",
    )

    return manifest


def write_holdout(
    output_dir: Path = ROOT / OUTPUT_DIR,
) -> dict[str, Any]:
    authenticate_repo()

    require(
        not output_dir.exists(),
        f"OUTPUT_COLLISION:{output_dir}",
    )

    payload = build_payload()
    hashes = _write_payload(output_dir, payload)
    manifest = validate_written(output_dir)

    return {
        "result": RESULT,
        "manifest": manifest,
        "artifact_sha256": hashes,
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the prospective outcome-blind XG1 3601..3900 "
            "Mamba-370M residual sign-reversal decomposition holdout."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    result = write_holdout(args.output_dir)

    hashes = result["artifact_sha256"]

    print("RESULT=" + result["result"])
    print("PAIR_FIRST=xg1_fact_3601")
    print("PAIR_LAST=xg1_fact_3900")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("PRIOR_PAIR_OVERLAP=0")
    print("PRIOR_CLAIM_OVERLAP=0")
    print("PRIOR_EVIDENCE_OVERLAP=0")
    print("FORMAL_INFERENCE_ALLOWED=False")
    print("P_VALUE_COUNT=0")
    print("SELECTION_ALLOWED=False")
    print("RESCUE_OF_FAILED_CROSS_BACKBONE_CLAIM=False")
    print("SOURCE_SHA256=" + hashes[SOURCE_FILE])
    print("ROWS_SHA256=" + hashes[ROW_FILE])
    print("MANIFEST_SHA256=" + hashes[MANIFEST_FILE])


if __name__ == "__main__":
    main()
