#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Support both `python -m ...`/pytest imports and direct execution as
# `python scripts/<file>.py` from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    build_reason_router_gen4_seed181_behavioral_bridge_holdout
    as prior,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
FROZEN_PARENT_COMMIT = "f5711591407fc445a65e354314c0b594c609b9e1"

DISCOVERY_FIRST = 3001
DISCOVERY_LAST = 3300
CONFIRMATION_FIRST = 3301
CONFIRMATION_LAST = 3600
PAIR_COUNT = 300
ROWS_PER_PAIR = 6
EXPECTED_ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR

DISCOVERY_DIR = Path(
    "data/reason_router_gen4_mamba370m_xg1_discovery_v1"
)
CONFIRMATION_DIR = Path(
    "data/reason_router_gen4_mamba370m_xg1_confirmation_v1"
)

SOURCE_FILE = prior.SOURCE_FILE
ROW_FILE = prior.ROW_FILE
MANIFEST_FILE = "structural_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

MANIFEST_SCHEMA = (
    "GEN4_MAMBA370M_XG1_PROSPECTIVE_HOLDOUT_V1"
)

BEHAVIORAL_2701_3000_DIR = Path(
    "data/reason_router_gen4_xg1_behavioral_bridge_v1"
)
BEHAVIORAL_2701_3000_SOURCE_SHA256 = (
    "01b4989ee1d2b9273a2a6d2f044753dcc64de9300be5e1d8b5765c760e6d9fd4"
)
BEHAVIORAL_2701_3000_ROW_SHA256 = (
    "3cbdd208ea110c5576b24f5f2ee43e0d4892e35bc78275b7bda434f2deb20afa"
)

FROZEN_DEPENDENCIES = (
    "scripts/build_reason_router_gen4_xg1_cross_generator_cohort.py",
    "scripts/build_reason_router_gen4_seed181_behavioral_bridge_holdout.py",
    BEHAVIORAL_2701_3000_DIR.as_posix(),
)

FORBIDDEN_OUTCOME_FIELDS = frozenset(
    {
        "label",
        "labels",
        "final_label",
        "prediction",
        "predicted_label",
        "response",
        "responses",
        "logit",
        "logits",
        "probability",
        "probabilities",
        "endpoint",
        "endpoint_value",
        "q",
        "Q",
        "margin",
        "effect",
        "effect_size",
        "p_value",
        "pvalue",
    }
)


class Mamba370MXG1HoldoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Mamba370MXG1HoldoutError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Mamba370MXG1HoldoutError(
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
            FROZEN_PARENT_COMMIT,
            "HEAD",
        )
        == 0,
        "FROZEN_PARENT_NOT_ANCESTOR",
    )

    for path in FROZEN_DEPENDENCIES:
        require(
            git_rc(
                "diff",
                "--quiet",
                FROZEN_PARENT_COMMIT,
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
    first: int,
    last: int,
) -> list[str]:
    require(
        first >= 1 and last >= first,
        "PAIR_RANGE",
    )
    return [
        f"xg1_fact_{index:03d}"
        for index in range(first, last + 1)
    ]


def build_population(
    first: int,
    last: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts, rows = prior.build_population(first, last)

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


def _inventory_from_serialized(
    source_raw: bytes,
    row_raw: bytes,
    *,
    first: int,
    last: int,
    label: str,
) -> tuple[set[str], set[str], set[str]]:
    facts = prior.read_jsonl_bytes(
        source_raw,
        f"{label}:{SOURCE_FILE}",
    )
    rows = prior.read_jsonl_bytes(
        row_raw,
        f"{label}:{ROW_FILE}",
    )

    expected = expected_pair_ids(first, last)
    require(
        [str(row["pair_id"]) for row in facts] == expected,
        f"{label}:PAIR_IDS",
    )
    prior.base.validate_materialized_rows(
        rows,
        expected_pairs=len(expected),
    )

    return (
        set(expected),
        {str(row["claim"]) for row in rows},
        {str(row["evidence"]) for row in rows},
    )


def prior_inventory_through_3000(
) -> tuple[set[str], set[str], set[str]]:
    pair_ids, claims, evidence = prior.prior_inventory()

    source_rel = (
        BEHAVIORAL_2701_3000_DIR / SOURCE_FILE
    )
    row_rel = (
        BEHAVIORAL_2701_3000_DIR / ROW_FILE
    )
    source_raw = prior.git_blob_bytes(source_rel)
    row_raw = prior.git_blob_bytes(row_rel)

    require(
        sha256_bytes(source_raw)
        == BEHAVIORAL_2701_3000_SOURCE_SHA256,
        "BEHAVIORAL_2701_3000_SOURCE_SHA",
    )
    require(
        sha256_bytes(row_raw)
        == BEHAVIORAL_2701_3000_ROW_SHA256,
        "BEHAVIORAL_2701_3000_ROW_SHA",
    )

    p2, c2, e2 = _inventory_from_serialized(
        source_raw,
        row_raw,
        first=2701,
        last=3000,
        label="BEHAVIORAL_2701_3000",
    )

    require(not (pair_ids & p2), "PRIOR_PAIR_OVERLAP_2701_3000")
    require(not (claims & c2), "PRIOR_CLAIM_OVERLAP_2701_3000")
    require(not (evidence & e2), "PRIOR_EVIDENCE_OVERLAP_2701_3000")

    pair_ids |= p2
    claims |= c2
    evidence |= e2

    require(
        pair_ids == set(expected_pair_ids(1, 3000)),
        "PRIOR_PAIR_COVERAGE_001_3000",
    )
    return pair_ids, claims, evidence


def _sets_for_rows(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    return (
        {str(row["pair_id"]) for row in facts},
        {str(row["claim"]) for row in rows},
        {str(row["evidence"]) for row in rows},
    )


def _validate_fresh_against_inventory(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    prior_pairs: set[str],
    prior_claims: set[str],
    prior_evidence: set[str],
    label: str,
) -> tuple[set[str], set[str], set[str]]:
    pair_ids, claims, evidence = _sets_for_rows(
        facts,
        rows,
    )

    require(
        not (pair_ids & prior_pairs),
        f"{label}:PAIR_OVERLAP",
    )
    require(
        not (claims & prior_claims),
        f"{label}:CLAIM_OVERLAP",
    )
    require(
        not (evidence & prior_evidence),
        f"{label}:EVIDENCE_OVERLAP",
    )

    return pair_ids, claims, evidence


def build_manifest(
    *,
    role: str,
    first: int,
    last: int,
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    prior_last: int,
) -> dict[str, Any]:
    require(
        role in {"discovery", "confirmation"},
        "ROLE",
    )
    require(
        len(facts) == PAIR_COUNT,
        "MANIFEST_PAIR_COUNT",
    )
    require(
        len(rows) == EXPECTED_ROW_COUNT,
        "MANIFEST_ROW_COUNT",
    )

    source_raw = prior.jsonl_bytes(facts)
    row_raw = prior.jsonl_bytes(rows)

    return {
        "schema_version": MANIFEST_SCHEMA,
        "result": (
            "PASS_MAMBA370M_XG1_DISCOVERY_3001_3300_STRUCTURAL"
            if role == "discovery"
            else "PASS_MAMBA370M_XG1_CONFIRMATION_3301_3600_STRUCTURAL"
        ),
        "role": role,
        "generator_family": prior.base.GENERATOR_FAMILY,
        "deterministic_generator_semantics": True,
        "pair_id_first": f"xg1_fact_{first:03d}",
        "pair_id_last": f"xg1_fact_{last:03d}",
        "source_pair_count": PAIR_COUNT,
        "row_count": EXPECTED_ROW_COUNT,
        "rows_per_pair": ROWS_PER_PAIR,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(row_raw),
        "prior_pair_range": (
            f"xg1_fact_001..xg1_fact_{prior_last:03d}"
        ),
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
        "prospective_use": (
            "plane_selection_only"
            if role == "discovery"
            else "confirmatory_only_after_selection_freeze"
        ),
        "selection_response_access_allowed": (
            role == "discovery"
        ),
        "confirmation_response_access_allowed_before_selection_freeze": False,
        "rescue_policy": "none",
    }


def build_holdout_payloads() -> dict[str, dict[str, Any]]:
    prior_pairs, prior_claims, prior_evidence = (
        prior_inventory_through_3000()
    )

    d_facts_a, d_rows_a = build_population(
        DISCOVERY_FIRST,
        DISCOVERY_LAST,
    )
    d_facts_b, d_rows_b = build_population(
        DISCOVERY_FIRST,
        DISCOVERY_LAST,
    )
    require(
        prior.jsonl_bytes(d_facts_a)
        == prior.jsonl_bytes(d_facts_b),
        "DISCOVERY_SOURCE_REGENERATION",
    )
    require(
        prior.jsonl_bytes(d_rows_a)
        == prior.jsonl_bytes(d_rows_b),
        "DISCOVERY_ROW_REGENERATION",
    )

    d_pairs, d_claims, d_evidence = (
        _validate_fresh_against_inventory(
            d_facts_a,
            d_rows_a,
            prior_pairs=prior_pairs,
            prior_claims=prior_claims,
            prior_evidence=prior_evidence,
            label="DISCOVERY",
        )
    )

    c_facts_a, c_rows_a = build_population(
        CONFIRMATION_FIRST,
        CONFIRMATION_LAST,
    )
    c_facts_b, c_rows_b = build_population(
        CONFIRMATION_FIRST,
        CONFIRMATION_LAST,
    )
    require(
        prior.jsonl_bytes(c_facts_a)
        == prior.jsonl_bytes(c_facts_b),
        "CONFIRMATION_SOURCE_REGENERATION",
    )
    require(
        prior.jsonl_bytes(c_rows_a)
        == prior.jsonl_bytes(c_rows_b),
        "CONFIRMATION_ROW_REGENERATION",
    )

    c_pairs, c_claims, c_evidence = (
        _validate_fresh_against_inventory(
            c_facts_a,
            c_rows_a,
            prior_pairs=prior_pairs | d_pairs,
            prior_claims=prior_claims | d_claims,
            prior_evidence=prior_evidence | d_evidence,
            label="CONFIRMATION",
        )
    )

    require(
        len(d_pairs) == PAIR_COUNT
        and len(c_pairs) == PAIR_COUNT,
        "FRESH_PAIR_COUNT",
    )
    require(
        not (d_pairs & c_pairs),
        "DISCOVERY_CONFIRMATION_PAIR_OVERLAP",
    )
    require(
        not (d_claims & c_claims),
        "DISCOVERY_CONFIRMATION_CLAIM_OVERLAP",
    )
    require(
        not (d_evidence & c_evidence),
        "DISCOVERY_CONFIRMATION_EVIDENCE_OVERLAP",
    )

    d_source = prior.jsonl_bytes(d_facts_a)
    d_rows = prior.jsonl_bytes(d_rows_a)
    d_manifest = build_manifest(
        role="discovery",
        first=DISCOVERY_FIRST,
        last=DISCOVERY_LAST,
        facts=d_facts_a,
        rows=d_rows_a,
        prior_last=3000,
    )

    c_source = prior.jsonl_bytes(c_facts_a)
    c_rows = prior.jsonl_bytes(c_rows_a)
    c_manifest = build_manifest(
        role="confirmation",
        first=CONFIRMATION_FIRST,
        last=CONFIRMATION_LAST,
        facts=c_facts_a,
        rows=c_rows_a,
        prior_last=3300,
    )

    return {
        "discovery": {
            "source": d_source,
            "rows": d_rows,
            "manifest": d_manifest,
        },
        "confirmation": {
            "source": c_source,
            "rows": c_rows,
            "manifest": c_manifest,
        },
    }


def _write_one(
    output_dir: Path,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    require(
        not output_dir.exists(),
        f"OUTPUT_COLLISION:{output_dir}",
    )

    source_raw = bytes(payload["source"])
    row_raw = bytes(payload["rows"])
    manifest_raw = canonical_manifest_bytes(
        payload["manifest"]
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / SOURCE_FILE).write_bytes(source_raw)
    (output_dir / ROW_FILE).write_bytes(row_raw)
    (output_dir / MANIFEST_FILE).write_bytes(
        manifest_raw
    )

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
    *,
    expected_first: int,
    expected_last: int,
    expected_role: str,
) -> dict[str, Any]:
    required = {
        SOURCE_FILE,
        ROW_FILE,
        MANIFEST_FILE,
        CHECKSUM_FILE,
    }
    require(
        output_dir.is_dir(),
        f"OUTPUT_DIR_MISSING:{output_dir}",
    )
    require(
        {
            path.name
            for path in output_dir.iterdir()
            if path.is_file()
        }
        == required,
        f"OUTPUT_FILE_SET:{output_dir}",
    )

    source_raw = (output_dir / SOURCE_FILE).read_bytes()
    row_raw = (output_dir / ROW_FILE).read_bytes()
    manifest = json.loads(
        (output_dir / MANIFEST_FILE).read_text(
            encoding="utf-8"
        )
    )

    facts = prior.read_jsonl_bytes(
        source_raw,
        f"{output_dir}:{SOURCE_FILE}",
    )
    rows = prior.read_jsonl_bytes(
        row_raw,
        f"{output_dir}:{ROW_FILE}",
    )

    require(
        [str(row["pair_id"]) for row in facts]
        == expected_pair_ids(
            expected_first,
            expected_last,
        ),
        f"WRITTEN_PAIR_IDS:{output_dir}",
    )
    prior.base.validate_materialized_rows(
        rows,
        expected_pairs=PAIR_COUNT,
    )

    require(
        manifest["role"] == expected_role,
        f"WRITTEN_ROLE:{output_dir}",
    )
    require(
        manifest["source_file_sha256"]
        == sha256_bytes(source_raw),
        f"WRITTEN_SOURCE_SHA:{output_dir}",
    )
    require(
        manifest["row_file_sha256"]
        == sha256_bytes(row_raw),
        f"WRITTEN_ROW_SHA:{output_dir}",
    )
    require(
        manifest["labels_present"] is False
        and manifest["response_fields_present"] is False
        and manifest["endpoint_values_present"] is False
        and manifest["tokenizer_executed"] is False
        and manifest["checkpoint_loaded"] is False
        and manifest["model_executed"] is False
        and manifest["cuda_executed"] is False,
        f"WRITTEN_OUTCOME_BLINDNESS:{output_dir}",
    )
    return manifest


def write_holdouts(
    discovery_dir: Path = ROOT / DISCOVERY_DIR,
    confirmation_dir: Path = ROOT / CONFIRMATION_DIR,
) -> dict[str, Any]:
    authenticate_repo()

    require(
        not discovery_dir.exists(),
        f"OUTPUT_COLLISION:{discovery_dir}",
    )
    require(
        not confirmation_dir.exists(),
        f"OUTPUT_COLLISION:{confirmation_dir}",
    )

    payloads = build_holdout_payloads()

    d_hashes = _write_one(
        discovery_dir,
        payloads["discovery"],
    )
    c_hashes = _write_one(
        confirmation_dir,
        payloads["confirmation"],
    )

    d_manifest = validate_written(
        discovery_dir,
        expected_first=DISCOVERY_FIRST,
        expected_last=DISCOVERY_LAST,
        expected_role="discovery",
    )
    c_manifest = validate_written(
        confirmation_dir,
        expected_first=CONFIRMATION_FIRST,
        expected_last=CONFIRMATION_LAST,
        expected_role="confirmation",
    )

    return {
        "result": "PASS_MAMBA370M_XG1_PROSPECTIVE_HOLDOUTS",
        "discovery": {
            "manifest": d_manifest,
            "artifact_sha256": d_hashes,
        },
        "confirmation": {
            "manifest": c_manifest,
            "artifact_sha256": c_hashes,
        },
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize outcome-blind Mamba-370M XG1 "
            "discovery 3001..3300 and confirmation "
            "3301..3600 structural holdouts. Static only."
        )
    )
    parser.add_argument(
        "--discovery-output-dir",
        type=Path,
        default=ROOT / DISCOVERY_DIR,
    )
    parser.add_argument(
        "--confirmation-output-dir",
        type=Path,
        default=ROOT / CONFIRMATION_DIR,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    result = write_holdouts(
        discovery_dir=args.discovery_output_dir,
        confirmation_dir=args.confirmation_output_dir,
    )

    d = result["discovery"]["manifest"]
    c = result["confirmation"]["manifest"]

    print(
        "RESULT="
        "PASS_MAMBA370M_XG1_PROSPECTIVE_HOLDOUTS"
    )
    print(
        "DISCOVERY_RANGE="
        f"{d['pair_id_first']}..{d['pair_id_last']}"
    )
    print(
        "CONFIRMATION_RANGE="
        f"{c['pair_id_first']}..{c['pair_id_last']}"
    )
    print("DISCOVERY_SOURCE_PAIR_COUNT=300")
    print("CONFIRMATION_SOURCE_PAIR_COUNT=300")
    print("DISCOVERY_ROW_COUNT=1800")
    print("CONFIRMATION_ROW_COUNT=1800")
    print("PRIOR_PAIR_OVERLAP=0")
    print("DISCOVERY_CONFIRMATION_PAIR_OVERLAP=0")
    print("DISCOVERY_CONFIRMATION_CLAIM_OVERLAP=0")
    print("DISCOVERY_CONFIRMATION_EVIDENCE_OVERLAP=0")
    print("LABELS_PRESENT=False")
    print("RESPONSE_FIELDS_PRESENT=False")
    print("TOKENIZER_EXECUTED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
