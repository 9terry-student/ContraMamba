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
    build_reason_router_gen4_mamba14b_adjacent_site_specificity_holdout
    as previous,
)


ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

PLAN_FREEZE_COMMIT = "2111520366002d2f3ef412218d757968a7623887"
PLAN_PATH = (
    "reports/"
    "reason_router_gen4_low_displacement_behavioral_robustness_"
    "prospective_plan.md"
)
PLAN_BLOB = "86e9a324a8fec07a8e72761c4ef4a4a1a206a1dd"

PREVIOUS_BUILDER_PATH = (
    "scripts/"
    "build_reason_router_gen4_mamba14b_adjacent_site_specificity_holdout.py"
)
PREVIOUS_BUILDER_BLOB = "4401dba5d59b70647f9582eb29848c847fe5fd94"

FIRST_PAIR = 5401
LAST_PAIR = 5700
PAIR_COUNT = 300
ROWS_PER_PAIR = 6
ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR

OUTPUT_DIR = Path(
    "data/reason_router_gen4_mamba370m14b_low_displacement_xg1_v1"
)
SOURCE_FILE = previous.SOURCE_FILE
ROW_FILE = previous.ROW_FILE
MANIFEST_FILE = "structural_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

MANIFEST_SCHEMA = (
    "GEN4_MAMBA370M14B_LOW_DISPLACEMENT_XG1_STRUCTURAL_V1"
)
RESULT = (
    "PASS_MAMBA370M14B_LOW_DISPLACEMENT_XG1_5401_5700_STRUCTURAL"
)

PREVIOUS_DIR = previous.OUTPUT_DIR
PREVIOUS_FIRST = previous.FIRST_PAIR
PREVIOUS_LAST = previous.LAST_PAIR

TARGET_CELLS = ("C0_SHAM", "C2_NAME")
SCALES = ("mamba370m", "mamba14b")
BEHAVIORAL_ALPHAS = (0.5, 0.25)
PRIMARY_ALPHA = 0.25

SELECTED_PLANES = {
    "mamba370m": "P3",
    "mamba14b": "P5",
}
CONTROL_PLANES = {
    "mamba370m": "P5",
    "mamba14b": "P4",
}

SOURCE_BLOCK = 33
TARGET_RESIDUAL_LAYER = 34
INTERVENTION_LAYER = 35
ANCHOR_NAME = "A_IDENTITY"
TARGET_OFFSET = 2

FORBIDDEN_OUTCOME_FIELDS = previous.FORBIDDEN_OUTCOME_FIELDS


class LowDisplacementHoldoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LowDisplacementHoldoutError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LowDisplacementHoldoutError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def git_blob_bytes(path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LowDisplacementHoldoutError(
            f"GIT_BLOB_FAILURE:{path.as_posix()}"
        ) from exc


def authenticate_repo() -> None:
    require(
        git("branch", "--show-current") == EXPECTED_BRANCH,
        "BRANCH_MISMATCH",
    )
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            PLAN_FREEZE_COMMIT,
            "HEAD",
        )
        == 0,
        "PLAN_FREEZE_NOT_ANCESTOR",
    )
    require(
        git("rev-parse", f"HEAD:{PLAN_PATH}") == PLAN_BLOB,
        "PLAN_BLOB_DRIFT",
    )
    require(
        git("rev-parse", f"HEAD:{PREVIOUS_BUILDER_PATH}")
        == PREVIOUS_BUILDER_BLOB,
        "PREVIOUS_BUILDER_BLOB_DRIFT",
    )

    frozen_paths = (
        PLAN_PATH,
        PREVIOUS_BUILDER_PATH,
        PREVIOUS_DIR.as_posix(),
    )
    for path in frozen_paths:
        require(
            git_rc(
                "diff",
                "--quiet",
                PLAN_FREEZE_COMMIT,
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


def _parse_jsonl_bytes(
    raw: bytes,
    label: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        raw.decode("utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"{label}:OBJECT:{line_no}",
        )
        out.append(value)
    return out


def prior_inventory_through_5400(
) -> tuple[set[str], set[str], set[str]]:
    pair_ids, claims, evidence = previous.prior_inventory_through_5100()

    source_rel = PREVIOUS_DIR / SOURCE_FILE
    rows_rel = PREVIOUS_DIR / ROW_FILE
    manifest_rel = PREVIOUS_DIR / previous.MANIFEST_FILE

    source_raw = git_blob_bytes(source_rel)
    rows_raw = git_blob_bytes(rows_rel)
    manifest_raw = git_blob_bytes(manifest_rel)

    manifest = json.loads(manifest_raw.decode("utf-8-sig"))

    require(
        manifest["result"] == previous.RESULT,
        "PREVIOUS_MANIFEST_RESULT",
    )
    require(
        manifest["pair_id_first"] == "xg1_fact_5101"
        and manifest["pair_id_last"] == "xg1_fact_5400",
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

    frozen_facts = _parse_jsonl_bytes(
        source_raw,
        "PREVIOUS_SOURCE",
    )
    frozen_rows = _parse_jsonl_bytes(
        rows_raw,
        "PREVIOUS_ROWS",
    )

    new_pairs = {
        str(row["pair_id"])
        for row in frozen_facts
    }
    new_claims = {
        str(row["claim"])
        for row in frozen_rows
    }
    new_evidence = {
        str(row["evidence"])
        for row in frozen_rows
    }

    require(
        not (pair_ids & new_pairs),
        "PREVIOUS_PAIR_OVERLAP",
    )
    require(
        not (claims & new_claims),
        "PREVIOUS_CLAIM_OVERLAP",
    )
    require(
        not (evidence & new_evidence),
        "PREVIOUS_EVIDENCE_OVERLAP",
    )

    pair_ids |= new_pairs
    claims |= new_claims
    evidence |= new_evidence

    require(
        pair_ids
        == set(previous.expected_pair_ids(1, 5400)),
        "PRIOR_PAIR_COVERAGE_001_5400",
    )
    return pair_ids, claims, evidence


def _validate_semantic_labels(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    previous.previous.validate_semantic_labels(
        facts,
        rows,
    )


def build_payload() -> dict[str, Any]:
    prior_pairs, prior_claims, prior_evidence = (
        prior_inventory_through_5400()
    )

    facts_a, rows_a = build_population()
    facts_b, rows_b = build_population()

    source_raw = jsonl_bytes(facts_a)
    row_raw = jsonl_bytes(rows_a)

    require(
        source_raw == jsonl_bytes(facts_b),
        "SOURCE_REGENERATION",
    )
    require(
        row_raw == jsonl_bytes(rows_b),
        "ROW_REGENERATION",
    )
    _validate_semantic_labels(facts_a, rows_a)

    pairs = {
        str(row["pair_id"])
        for row in facts_a
    }
    claims = {
        str(row["claim"])
        for row in rows_a
    }
    evidence = {
        str(row["evidence"])
        for row in rows_a
    }

    require(
        not (pairs & prior_pairs),
        "PAIR_OVERLAP_WITH_001_5400",
    )
    require(
        not (claims & prior_claims),
        "CLAIM_OVERLAP_WITH_001_5400",
    )
    require(
        not (evidence & prior_evidence),
        "EVIDENCE_OVERLAP_WITH_001_5400",
    )

    require(
        len({str(row["pair_id"]) for row in facts_a})
        == PAIR_COUNT,
        "PAIR_ID_UNIQUENESS",
    )
    require(
        len(claims) == PAIR_COUNT,
        "CLAIM_CARDINALITY",
    )
    require(
        len(evidence) == ROW_COUNT,
        "EVIDENCE_CARDINALITY",
    )

    generator_family = {
        str(row["generator_family"])
        for row in facts_a
    }
    require(
        len(generator_family) == 1,
        "GENERATOR_FAMILY",
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT,
        "role": "low_displacement_behavioral_robustness",
        "plan_freeze_commit": PLAN_FREEZE_COMMIT,
        "generator_family": next(iter(generator_family)),
        "deterministic_generator_semantics": True,
        "pair_id_first": "xg1_fact_5401",
        "pair_id_last": "xg1_fact_5700",
        "source_pair_count": PAIR_COUNT,
        "row_count": ROW_COUNT,
        "rows_per_pair": ROWS_PER_PAIR,
        "target_cells": list(TARGET_CELLS),
        "scale_order": list(SCALES),
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(row_raw),
        "prior_pair_range": "xg1_fact_001..xg1_fact_5400",
        "pair_id_overlap_with_prior": 0,
        "claim_overlap_with_prior": 0,
        "evidence_overlap_with_prior": 0,
        "source_block": SOURCE_BLOCK,
        "target_residual_layer": TARGET_RESIDUAL_LAYER,
        "intervention_layer": INTERVENTION_LAYER,
        "anchor_name": ANCHOR_NAME,
        "target_offset": TARGET_OFFSET,
        "selected_planes_frozen": dict(SELECTED_PLANES),
        "control_planes_frozen": dict(CONTROL_PLANES),
        "behavioral_alphas": list(BEHAVIORAL_ALPHAS),
        "primary_alpha": PRIMARY_ALPHA,
        "labels_embedded_in_structural_rows": False,
        "response_fields_present": False,
        "endpoint_values_present": False,
        "selection_allowed": False,
        "cohort_replacement_allowed": False,
        "row_filtering_allowed": False,
        "alpha_sweep_allowed": False,
        "scale_specific_alpha_allowed": False,
        "layer_sweep_allowed": False,
        "token_sweep_allowed": False,
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return _parse_jsonl_bytes(
        path.read_bytes(),
        path.as_posix(),
    )


def validate_written(
    output_dir: Path,
) -> dict[str, Any]:
    required = {
        SOURCE_FILE,
        ROW_FILE,
        MANIFEST_FILE,
        CHECKSUM_FILE,
    }
    require(
        output_dir.is_dir(),
        "OUTPUT_DIR_MISSING",
    )
    require(
        {
            path.name
            for path in output_dir.iterdir()
            if path.is_file()
        }
        == required,
        "OUTPUT_FILE_SET",
    )

    source_raw = (
        output_dir / SOURCE_FILE
    ).read_bytes()
    row_raw = (
        output_dir / ROW_FILE
    ).read_bytes()
    manifest_raw = (
        output_dir / MANIFEST_FILE
    ).read_bytes()

    manifest = json.loads(
        manifest_raw.decode("utf-8-sig")
    )
    facts = _read_jsonl(
        output_dir / SOURCE_FILE
    )
    rows = _read_jsonl(
        output_dir / ROW_FILE
    )

    expected_facts, expected_rows = build_population()
    require(
        source_raw == jsonl_bytes(expected_facts),
        "WRITTEN_SOURCE_REGENERATION",
    )
    require(
        row_raw == jsonl_bytes(expected_rows),
        "WRITTEN_ROWS_REGENERATION",
    )
    _validate_semantic_labels(facts, rows)

    require(
        [str(row["pair_id"]) for row in facts]
        == expected_pair_ids(),
        "WRITTEN_PAIR_IDS",
    )

    require(
        manifest["schema_version"] == MANIFEST_SCHEMA,
        "MANIFEST_SCHEMA",
    )
    require(
        manifest["result"] == RESULT,
        "MANIFEST_RESULT",
    )
    require(
        manifest["plan_freeze_commit"]
        == PLAN_FREEZE_COMMIT,
        "MANIFEST_PLAN_COMMIT",
    )
    require(
        manifest["pair_id_first"] == "xg1_fact_5401"
        and manifest["pair_id_last"] == "xg1_fact_5700",
        "MANIFEST_RANGE",
    )
    require(
        manifest["prior_pair_range"]
        == "xg1_fact_001..xg1_fact_5400",
        "MANIFEST_PRIOR_RANGE",
    )
    require(
        manifest["source_pair_count"] == 300
        and manifest["row_count"] == 1800,
        "MANIFEST_COUNTS",
    )
    require(
        manifest["target_cells"] == ["C0_SHAM", "C2_NAME"],
        "MANIFEST_CELLS",
    )
    require(
        manifest["scale_order"]
        == ["mamba370m", "mamba14b"],
        "MANIFEST_SCALES",
    )
    require(
        manifest["selected_planes_frozen"]
        == SELECTED_PLANES,
        "MANIFEST_SELECTED",
    )
    require(
        manifest["control_planes_frozen"]
        == CONTROL_PLANES,
        "MANIFEST_CONTROL",
    )
    require(
        manifest["behavioral_alphas"]
        == [0.5, 0.25]
        and manifest["primary_alpha"] == 0.25,
        "MANIFEST_ALPHAS",
    )
    require(
        manifest["source_file_sha256"]
        == sha256_bytes(source_raw),
        "MANIFEST_SOURCE_SHA",
    )
    require(
        manifest["row_file_sha256"]
        == sha256_bytes(row_raw),
        "MANIFEST_ROWS_SHA",
    )

    for key in (
        "response_fields_present",
        "endpoint_values_present",
        "selection_allowed",
        "cohort_replacement_allowed",
        "row_filtering_allowed",
        "alpha_sweep_allowed",
        "scale_specific_alpha_allowed",
        "layer_sweep_allowed",
        "token_sweep_allowed",
        "response_based_control_selection_allowed",
        "tokenizer_executed",
        "checkpoint_loaded",
        "model_executed",
        "cuda_executed",
    ):
        require(
            manifest[key] is False,
            f"OUTCOME_BOUNDARY:{key}",
        )

    expected_hashes = {
        SOURCE_FILE: sha256_bytes(source_raw),
        ROW_FILE: sha256_bytes(row_raw),
        MANIFEST_FILE: sha256_bytes(manifest_raw),
    }
    expected_sums = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(
            expected_hashes.items()
        )
    )
    require(
        (
            output_dir / CHECKSUM_FILE
        ).read_text(encoding="utf-8")
        == expected_sums,
        "CHECKSUMS",
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
    source_raw = bytes(payload["source"])
    row_raw = bytes(payload["rows"])
    manifest_raw = canonical_manifest_bytes(
        payload["manifest"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )
    (
        output_dir / SOURCE_FILE
    ).write_bytes(source_raw)
    (
        output_dir / ROW_FILE
    ).write_bytes(row_raw)
    (
        output_dir / MANIFEST_FILE
    ).write_bytes(manifest_raw)

    hashes = {
        SOURCE_FILE: sha256_bytes(source_raw),
        ROW_FILE: sha256_bytes(row_raw),
        MANIFEST_FILE: sha256_bytes(manifest_raw),
    }
    (
        output_dir / CHECKSUM_FILE
    ).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(
                hashes.items()
            )
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


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the outcome-blind XG1 5401..5700 cohort "
            "for Gen4 370M/1.4B low-displacement behavioral "
            "robustness. Static only."
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
    result = write_holdout(
        args.output_dir
    )
    manifest = result["manifest"]

    print("RESULT=" + result["result"])
    print(
        "PAIR_RANGE="
        f"{manifest['pair_id_first']}.."
        f"{manifest['pair_id_last']}"
    )
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("PAIR_OVERLAP_WITH_001_5400=0")
    print("CLAIM_OVERLAP_WITH_001_5400=0")
    print("EVIDENCE_OVERLAP_WITH_001_5400=0")
    print("TARGET_CELLS=C0_SHAM,C2_NAME")
    print("SCALES=mamba370m,mamba14b")
    print("BEHAVIORAL_ALPHAS=0.5,0.25")
    print("PRIMARY_ALPHA=0.25")
    print("TOKENIZER_EXECUTED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
