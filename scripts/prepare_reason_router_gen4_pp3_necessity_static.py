from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import (
    build_reason_router_gen4_xg1_cross_generator_cohort as base
)
from scripts import (
    build_reason_router_gen4_xg1_fresh_specificity_cohort as fresh
)
from scripts import (
    reason_router_gen4_xg1_tokenizer_anchor_eligibility as legacy
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"

AUTHORITY_COMMIT = (
    "f4419f1e7efbeeb9e50e84b67accc56422580cf2"
)
DESIGN_PATH = Path(
    "reports/reason_router_gen4_pp3_necessity_design.md"
)
DESIGN_BLOB = (
    "0dacef464eb5802d121de2db0ab2bfcfd5281c98"
)

ORIGINAL_DIR = Path(
    "data/reason_router_gen4_xg1_cross_generator_v1"
)
FRESH_301_DIR = Path(
    "data/reason_router_gen4_xg1_fresh_specificity_v1"
)
OUTPUT_DATA_DIR = Path(
    "data/reason_router_gen4_xg1_necessity_v1"
)
OUTPUT_REPORT_DIR = Path(
    "reports/"
    "reason_router_gen4_pp3_necessity_static_preparation_f4419f1"
)

ORIGINAL_SOURCE_SHA = (
    "fccd6821eeb97194d5b898aca4911eaba"
    "71e893df7fe27c910aa37255a5695e0"
)
ORIGINAL_ROWS_SHA = (
    "6ea0484517e0ae7479ad7f3b0a74af4"
    "d75f7f7353d29586c597f2a9fee1e649f"
)

FRESH_301_SOURCE_SHA = (
    "aa5b8e3cfcbf19e71335ecdbea6593269"
    "25f8bea33312c2354de670fa7a15cf7"
)
FRESH_301_ROWS_SHA = (
    "3f28d8a75008d383855313a08168fef1"
    "a2b9b37257103636a7f2edb65ce76ad6"
)

PP3_DIR = Path(
    "reports/"
    "reason_router_gen4_pp3_xg1_external_transport_preparation_02e4c89"
)
PP5_DIR = Path(
    "reports/"
    "reason_router_gen4_pp3_pp5_fresh_xg1_specificity_preparation_0bc49ab"
)

PP3_PLUS_SHA = (
    "66ad0cd0f931b0aff88bfc8afc4e9ffa"
    "9e355d32f054d376acebb97b469a3cff"
)
PP3_MINUS_SHA = (
    "ea2c997de7dbaea4edad8b1f30cdac31"
    "b4a0c76db0f0df2983900f28a2529ce7"
)
PP5_PLUS_SHA = (
    "7eb8154a10f647a4b732f7a7b7e34087"
    "840a513b88177da06633d8f7b28a4df2"
)
PP5_MINUS_SHA = (
    "311a41c37b9586206ea9bfc9da390688"
    "d9a6bb5672a5f63bb589c9d019478855"
)

TOKENIZER_REVISION = (
    "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
)
TOKENIZER_HASHES = {
    "tokenizer.json":
        "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf",
    "tokenizer_config.json":
        "9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb",
    "special_tokens_map.json":
        "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
}

PAIR_FIRST = "xg1_fact_601"
PAIR_LAST = "xg1_fact_900"
PAIR_COUNT = 300
ROW_COUNT = 1800
VECTOR_DIM = 395
TOL = 1.0e-12


class StaticPreparationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StaticPreparationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticPreparationError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_show_bytes(
    path: Path,
    ref: str = "HEAD",
) -> bytes:
    try:
        return subprocess.check_output(
            [
                "git",
                "show",
                f"{ref}:{path.as_posix()}",
            ],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticPreparationError(
            f"GIT_SHOW_FAILURE:{ref}:{path.as_posix()}"
        ) from exc


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if line.strip():
            value = json.loads(line)
            require(
                isinstance(value, dict),
                f"JSONL_OBJECT:{path}",
            )
            rows.append(value)
    return rows


def authenticate_authority() -> dict[str, str]:
    branch = git("branch", "--show-current")
    require(
        branch == EXPECTED_BRANCH,
        f"BRANCH_MISMATCH:{branch}",
    )

    head = git("rev-parse", "HEAD")

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            AUTHORITY_COMMIT,
            head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "AUTHORITY_COMMIT_NOT_ANCESTOR",
    )

    design_blob = git(
        "rev-parse",
        f"HEAD:{DESIGN_PATH.as_posix()}",
    )
    require(
        design_blob == DESIGN_BLOB,
        f"DESIGN_BLOB_DRIFT:{design_blob}",
    )

    return {
        "branch": branch,
        "head": head,
        "authority_commit": AUTHORITY_COMMIT,
        "design_blob": design_blob,
    }


def prove_generator_identity() -> dict[str, Any]:
    original_source_path = (
        ROOT / ORIGINAL_DIR / base.SOURCE_FILE
    )
    original_rows_path = (
        ROOT / ORIGINAL_DIR / base.ROW_FILE
    )
    fresh_source_path = (
        ROOT / FRESH_301_DIR / base.SOURCE_FILE
    )
    fresh_rows_path = (
        ROOT / FRESH_301_DIR / base.ROW_FILE
    )

    require(
        sha256_bytes(git_show_bytes(ORIGINAL_DIR / base.SOURCE_FILE))
        == ORIGINAL_SOURCE_SHA,
        "ORIGINAL_SOURCE_SHA",
    )
    require(
        sha256_bytes(git_show_bytes(ORIGINAL_DIR / base.ROW_FILE))
        == ORIGINAL_ROWS_SHA,
        "ORIGINAL_ROWS_SHA",
    )
    require(
        sha256_bytes(git_show_bytes(FRESH_301_DIR / base.SOURCE_FILE))
        == FRESH_301_SOURCE_SHA,
        "FRESH_301_SOURCE_SHA",
    )
    require(
        sha256_bytes(git_show_bytes(FRESH_301_DIR / base.ROW_FILE))
        == FRESH_301_ROWS_SHA,
        "FRESH_301_ROWS_SHA",
    )

    facts_001 = fresh.build_source_facts(
        start=1,
        end=300,
    )
    rows_001 = fresh.materialize_facts(
        facts_001
    )

    facts_301 = fresh.build_source_facts(
        start=301,
        end=600,
    )
    rows_301 = fresh.materialize_facts(
        facts_301
    )

    require(
        base.jsonl_bytes(facts_001)
        == git_show_bytes(ORIGINAL_DIR / base.SOURCE_FILE),
        "REGENERATED_001_SOURCE_DRIFT",
    )
    require(
        base.jsonl_bytes(rows_001)
        == git_show_bytes(ORIGINAL_DIR / base.ROW_FILE),
        "REGENERATED_001_ROWS_DRIFT",
    )
    require(
        base.jsonl_bytes(facts_301)
        == git_show_bytes(FRESH_301_DIR / base.SOURCE_FILE),
        "REGENERATED_301_SOURCE_DRIFT",
    )
    require(
        base.jsonl_bytes(rows_301)
        == git_show_bytes(FRESH_301_DIR / base.ROW_FILE),
        "REGENERATED_301_ROWS_DRIFT",
    )

    return {
        "original_001_300_source_byte_identity": True,
        "original_001_300_row_byte_identity": True,
        "fresh_301_600_source_byte_identity": True,
        "fresh_301_600_row_byte_identity": True,
    }


def build_new_population() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    facts = fresh.build_source_facts(
        start=601,
        end=900,
    )
    rows = fresh.materialize_facts(facts)

    require(
        len(facts) == PAIR_COUNT,
        "NEW_FACT_COUNT",
    )
    require(
        len(rows) == ROW_COUNT,
        "NEW_ROW_COUNT",
    )

    expected_ids = [
        f"xg1_fact_{i:03d}"
        for i in range(601, 901)
    ]
    require(
        [str(x["pair_id"]) for x in facts]
        == expected_ids,
        "NEW_PAIR_ORDER",
    )

    old_rows = (
        read_jsonl(
            ROOT / ORIGINAL_DIR / base.ROW_FILE
        )
        + read_jsonl(
            ROOT / FRESH_301_DIR / base.ROW_FILE
        )
    )

    old_pair_ids = {
        str(row["source_pair_id"])
        for row in old_rows
    }
    new_pair_ids = {
        str(row["source_pair_id"])
        for row in rows
    }

    pair_overlap = (
        old_pair_ids & new_pair_ids
    )

    old_claims = {
        str(row["claim"])
        for row in old_rows
    }
    new_claims = {
        str(row["claim"])
        for row in rows
    }

    old_evidence = {
        str(row["evidence"])
        for row in old_rows
    }
    new_evidence = {
        str(row["evidence"])
        for row in rows
    }

    old_claim_evidence = {
        (
            str(row["claim"]),
            str(row["evidence"]),
        )
        for row in old_rows
    }
    new_claim_evidence = {
        (
            str(row["claim"]),
            str(row["evidence"]),
        )
        for row in rows
    }

    claim_overlap = (
        old_claims & new_claims
    )
    evidence_overlap = (
        old_evidence & new_evidence
    )
    pair_text_overlap = (
        old_claim_evidence
        & new_claim_evidence
    )

    require(
        not pair_overlap,
        f"PAIR_ID_OVERLAP:{len(pair_overlap)}",
    )
    require(
        not claim_overlap,
        f"CLAIM_OVERLAP:{len(claim_overlap)}",
    )
    require(
        not evidence_overlap,
        f"EVIDENCE_OVERLAP:{len(evidence_overlap)}",
    )
    require(
        not pair_text_overlap,
        (
            "CLAIM_EVIDENCE_ROW_OVERLAP:"
            f"{len(pair_text_overlap)}"
        ),
    )

    source_raw = base.jsonl_bytes(facts)
    rows_raw = base.jsonl_bytes(rows)

    return facts, rows, {
        "source_raw": source_raw,
        "rows_raw": rows_raw,
        "pair_id_overlap_count": 0,
        "claim_overlap_count": 0,
        "evidence_overlap_count": 0,
        "claim_evidence_row_overlap_count": 0,
    }


def load_vector(
    path: Path,
    expected_sha: str,
) -> np.ndarray:
    require(
        path.is_file(),
        f"VECTOR_MISSING:{path}",
    )
    observed = sha256_file(path)
    require(
        observed == expected_sha,
        (
            f"VECTOR_SHA:{path}:"
            f"{observed}"
        ),
    )

    vector = np.frombuffer(
        path.read_bytes(),
        dtype=np.dtype("<f8"),
    ).copy()

    require(
        vector.shape == (VECTOR_DIM,),
        f"VECTOR_SHAPE:{path}:{vector.shape}",
    )
    require(
        bool(np.isfinite(vector).all()),
        f"VECTOR_NONFINITE:{path}",
    )
    return vector


def geometry_audit() -> dict[str, Any]:
    p3p = load_vector(
        ROOT / PP3_DIR / "pp3_plus.f64le",
        PP3_PLUS_SHA,
    )
    p3m = load_vector(
        ROOT / PP3_DIR / "pp3_minus.f64le",
        PP3_MINUS_SHA,
    )
    p5p = load_vector(
        ROOT / PP5_DIR / "pp5_plus.f64le",
        PP5_PLUS_SHA,
    )
    p5m = load_vector(
        ROOT / PP5_DIR / "pp5_minus.f64le",
        PP5_MINUS_SHA,
    )

    b3 = np.stack([p3p, p3m], axis=1)
    b5 = np.stack([p5p, p5m], axis=1)
    eye = np.eye(2, dtype=np.float64)

    gram3 = b3.T @ b3
    gram5 = b5.T @ b5
    cross = b3.T @ b5

    gram3_residual = float(
        np.max(np.abs(gram3 - eye))
    )
    gram5_residual = float(
        np.max(np.abs(gram5 - eye))
    )
    cross_max = float(
        np.max(np.abs(cross))
    )

    require(
        gram3_residual <= TOL,
        f"PP3_GRAM_RESIDUAL:{gram3_residual}",
    )
    require(
        gram5_residual <= TOL,
        f"PP5_GRAM_RESIDUAL:{gram5_residual}",
    )
    require(
        cross_max <= TOL,
        f"PP3_PP5_CROSS_DOT:{cross_max}",
    )

    coefficient_probes = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [3.0, -4.0],
            [math.pi, math.e],
            [-0.125, 0.875],
        ],
        dtype=np.float64,
    )

    norm_diffs = []
    for coeff in coefficient_probes:
        d3 = b3 @ coeff
        d5 = b5 @ coeff
        norm_diffs.append(
            abs(
                float(np.linalg.norm(d3))
                - float(np.linalg.norm(d5))
            )
        )

    max_norm_diff = max(norm_diffs)

    require(
        max_norm_diff <= TOL,
        (
            "COEFFICIENT_TRANSFER_NORM:"
            f"{max_norm_diff}"
        ),
    )

    return {
        "schema_version":
            "GEN4_PP3_NECESSITY_GEOMETRY_STATIC_V1",
        "pp3_plus_sha256": PP3_PLUS_SHA,
        "pp3_minus_sha256": PP3_MINUS_SHA,
        "pp5_plus_sha256": PP5_PLUS_SHA,
        "pp5_minus_sha256": PP5_MINUS_SHA,
        "ambient_dim": VECTOR_DIM,
        "tolerance": TOL,
        "pp3_gram_max_abs_residual":
            gram3_residual,
        "pp5_gram_max_abs_residual":
            gram5_residual,
        "pp3_pp5_cross_plane_max_abs_dot":
            cross_max,
        "coefficient_transfer_probe_count":
            int(coefficient_probes.shape[0]),
        "coefficient_transfer_max_abs_l2_mismatch":
            max_norm_diff,
        "coefficient_transfer_norm_identity":
            (
                "same coefficients on two "
                "orthonormal two-vector bases"
            ),
        "result":
            "PASS_PP3_PP5_NECESSITY_GEOMETRY_STATIC",
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
    }


def tokenizer_eligibility(
    tokenizer_snapshot: Path,
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for name, expected in TOKENIZER_HASHES.items():
        path = tokenizer_snapshot / name
        require(
            path.is_file(),
            f"TOKENIZER_FILE_MISSING:{name}",
        )
        observed = sha256_file(path)
        require(
            observed == expected,
            (
                f"TOKENIZER_SHA:{name}:"
                f"{observed}"
            ),
        )

    tokenizer, provenance = (
        legacy.load_canonical_analysis_tokenizer(
            tokenizer_snapshot
        )
    )

    facts_by_id = {
        str(fact["pair_id"]): fact
        for fact in facts
    }
    require(
        len(facts_by_id) == PAIR_COUNT,
        "FACT_LOOKUP_COUNT",
    )

    anchor_rows: list[dict[str, Any]] = []

    for row in rows:
        pair_id = str(
            row["source_pair_id"]
        )
        produced = (
            legacy.analyze_required_anchors_for_row(
                row,
                facts_by_id[pair_id],
                tokenizer,
            )
        )
        for anchor in produced:
            value = dict(anchor)
            value["schema_version"] = (
                "GEN4_PP3_NECESSITY_XG1_"
                "TOKENIZER_ANCHOR_V1"
            )
            anchor_rows.append(value)

    require(
        len(anchor_rows) == ROW_COUNT,
        f"ANCHOR_ROW_COUNT:{len(anchor_rows)}",
    )

    anchor_counts = Counter(
        str(row["anchor_name"])
        for row in anchor_rows
    )
    require(
        dict(anchor_counts)
        == legacy.ANCHOR_EXPECTED_COUNTS,
        (
            "ANCHOR_COUNTS:"
            f"{dict(anchor_counts)}"
        ),
    )

    eligible_counts = Counter(
        str(row["anchor_name"])
        for row in anchor_rows
        if bool(row["post4_eligible"])
    )

    exclusions = Counter(
        str(row["exclusion_code"])
        for row in anchor_rows
        if row["exclusion_code"] is not None
    )

    pair_ok = {
        str(fact["pair_id"]): True
        for fact in facts
    }

    for row in anchor_rows:
        if not bool(row["post4_eligible"]):
            pair_ok[
                str(row["source_pair_id"])
            ] = False

    event_lookup = {
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["anchor_name"]),
        ): row
        for row in anchor_rows
    }
    require(
        len(event_lookup) == ROW_COUNT,
        "ANCHOR_EVENT_KEY_DUPLICATE",
    )

    mismatch_count = 0

    for pair_id in pair_ok:
        for cell_id in (
            legacy.TARGET_IDENTITY_NAME_CELLS
        ):
            identity = event_lookup[
                (
                    pair_id,
                    cell_id,
                    "A_IDENTITY",
                )
            ]
            name = event_lookup[
                (
                    pair_id,
                    cell_id,
                    "A_NAME",
                )
            ]

            if (
                identity[
                    "absolute_anchor_token_index"
                ]
                != name[
                    "absolute_anchor_token_index"
                ]
            ):
                pair_ok[pair_id] = False
                mismatch_count += 1

    complete_pairs = sum(pair_ok.values())

    require(
        complete_pairs == PAIR_COUNT,
        (
            "TOKENIZER_COMPLETE_PAIRS:"
            f"{complete_pairs}"
        ),
    )
    require(
        sum(eligible_counts.values())
        == ROW_COUNT,
        "ELIGIBLE_ANCHOR_TOTAL",
    )
    require(
        mismatch_count == 0,
        (
            "IDENTITY_NAME_MISMATCH:"
            f"{mismatch_count}"
        ),
    )
    require(
        not exclusions,
        f"ANCHOR_EXCLUSIONS:{dict(exclusions)}",
    )

    manifest_raw = b"".join(
        canonical_json_bytes(row)
        for row in anchor_rows
    )

    summary = {
        "schema_version":
            "GEN4_PP3_NECESSITY_XG1_TOKENIZER_ELIGIBILITY_SUMMARY_V1",
        "result": "PASS_300_OF_300",
        "authority_commit": AUTHORITY_COMMIT,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "source_pair_count": PAIR_COUNT,
        "anchor_row_count": ROW_COUNT,
        "complete_source_pair_count":
            complete_pairs,
        "eligible_anchor_counts": {
            key: int(
                eligible_counts.get(
                    key,
                    0,
                )
            )
            for key
            in legacy.ANCHOR_EXPECTED_COUNTS
        },
        "exclusion_counts": {},
        "target_identity_name_mismatch_count":
            mismatch_count,
        "active_serialization":
            "claim[:63]+EOS(0)+evidence[:64]",
        "post4_rule":
            "a+4 <= terminal_index-1",
        "tokenizer_revision":
            TOKENIZER_REVISION,
        "tokenizer": provenance,
        "tokenizer_hashes":
            dict(TOKENIZER_HASHES),
        "anchor_manifest_sha256":
            sha256_bytes(manifest_raw),
        "scientific_outcomes_observed": False,
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "training_executed": False,
        "evaluation_executed": False,
        "gpu_used": False,
    }

    return anchor_rows, summary


def write_outputs(
    provenance: Mapping[str, Any],
    identity: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
    geometry: Mapping[str, Any],
    anchor_rows: Sequence[Mapping[str, Any]],
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    data_dir = ROOT / OUTPUT_DATA_DIR
    report_dir = ROOT / OUTPUT_REPORT_DIR

    require(
        not data_dir.exists(),
        f"DATA_OUTPUT_COLLISION:{data_dir}",
    )
    require(
        not report_dir.exists(),
        f"REPORT_OUTPUT_COLLISION:{report_dir}",
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=False,
    )
    report_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    source_raw = population["source_raw"]
    rows_raw = population["rows_raw"]

    structural = {
        "schema_version":
            "GEN4_PP3_NECESSITY_XG1_STRUCTURAL_V1",
        "result":
            "PASS_XG1_601_900_NECESSITY_STRUCTURAL",
        "authority_commit": AUTHORITY_COMMIT,
        "generator_family":
            base.GENERATOR_FAMILY,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "source_pair_count": PAIR_COUNT,
        "row_count": ROW_COUNT,
        "rows_per_pair": 6,
        "source_file_sha256":
            sha256_bytes(source_raw),
        "row_file_sha256":
            sha256_bytes(rows_raw),
        **dict(identity),
        "pair_id_overlap_with_001_600":
            population[
                "pair_id_overlap_count"
            ],
        "claim_overlap_with_001_600":
            population[
                "claim_overlap_count"
            ],
        "evidence_overlap_with_001_600":
            population[
                "evidence_overlap_count"
            ],
        "claim_evidence_row_overlap_with_001_600":
            population[
                "claim_evidence_row_overlap_count"
            ],
        "deterministic_generator_semantics":
            True,
        "tokenizer_executed": False,
        "checkpoint_loaded": False,
        "model_executed": False,
        "cuda_executed": False,
    }

    data_payloads = {
        base.SOURCE_FILE: source_raw,
        base.ROW_FILE: rows_raw,
        "structural_manifest.json":
            pretty_json_bytes(structural),
    }

    data_hashes: dict[str, str] = {}

    for name, raw in data_payloads.items():
        path = data_dir / name
        path.write_bytes(raw)
        data_hashes[name] = sha256_bytes(raw)

    data_checksum = "".join(
        f"{digest}  {name}\n"
        for name, digest
        in sorted(data_hashes.items())
    ).encode("utf-8")

    (
        data_dir / "SHA256SUMS.txt"
    ).write_bytes(data_checksum)

    anchor_raw = b"".join(
        canonical_json_bytes(row)
        for row in anchor_rows
    )

    preparation = {
        "schema_version":
            "GEN4_PP3_NECESSITY_STATIC_PREPARATION_V1",
        "result":
            "PASS_PP3_NECESSITY_STATIC_PREPARATION",
        "authority": dict(provenance),
        "population": {
            "data_dir":
                OUTPUT_DATA_DIR.as_posix(),
            "source_pair_count":
                PAIR_COUNT,
            "row_count":
                ROW_COUNT,
            "pair_id_first":
                PAIR_FIRST,
            "pair_id_last":
                PAIR_LAST,
            "source_sha256":
                data_hashes[
                    base.SOURCE_FILE
                ],
            "rows_sha256":
                data_hashes[
                    base.ROW_FILE
                ],
            "structural_manifest_sha256":
                data_hashes[
                    "structural_manifest.json"
                ],
        },
        "geometry": {
            "result":
                geometry["result"],
            "pp3_plus_sha256":
                PP3_PLUS_SHA,
            "pp3_minus_sha256":
                PP3_MINUS_SHA,
            "pp5_plus_sha256":
                PP5_PLUS_SHA,
            "pp5_minus_sha256":
                PP5_MINUS_SHA,
            "pp3_gram_max_abs_residual":
                geometry[
                    "pp3_gram_max_abs_residual"
                ],
            "pp5_gram_max_abs_residual":
                geometry[
                    "pp5_gram_max_abs_residual"
                ],
            "pp3_pp5_cross_plane_max_abs_dot":
                geometry[
                    "pp3_pp5_cross_plane_max_abs_dot"
                ],
            "coefficient_transfer_max_abs_l2_mismatch":
                geometry[
                    "coefficient_transfer_max_abs_l2_mismatch"
                ],
        },
        "tokenizer_eligibility": {
            "result":
                eligibility["result"],
            "anchor_manifest_sha256":
                eligibility[
                    "anchor_manifest_sha256"
                ],
            "source_pair_count":
                eligibility[
                    "source_pair_count"
                ],
            "anchor_row_count":
                eligibility[
                    "anchor_row_count"
                ],
            "tokenizer_revision":
                TOKENIZER_REVISION,
        },
        "scientific_model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_outcomes_observed": False,
        "primary_inference_executed": False,
    }

    report_payloads = {
        "geometry_manifest.json":
            pretty_json_bytes(geometry),
        "tokenizer_anchor_manifest.jsonl":
            anchor_raw,
        "tokenizer_eligibility_summary.json":
            pretty_json_bytes(eligibility),
        "preparation_manifest.json":
            pretty_json_bytes(preparation),
    }

    report_hashes: dict[str, str] = {}

    for name, raw in report_payloads.items():
        path = report_dir / name
        path.write_bytes(raw)
        report_hashes[name] = sha256_bytes(raw)

    report_checksum = "".join(
        f"{digest}  {name}\n"
        for name, digest
        in sorted(report_hashes.items())
    ).encode("utf-8")

    (
        report_dir / "SHA256SUMS.txt"
    ).write_bytes(report_checksum)

    return {
        "data_hashes": data_hashes,
        "report_hashes": report_hashes,
        "structural": structural,
        "preparation": preparation,
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static preparation for frozen PP3 necessity design. "
            "Tokenizer-only eligibility is allowed; no model execution."
        )
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    require(
        not args.tokenizer_snapshot.resolve()
        .is_relative_to(ROOT.resolve()),
        "TOKENIZER_SNAPSHOT_MUST_BE_EXTERNAL_TO_REPO",
    )

    provenance = authenticate_authority()
    identity = prove_generator_identity()

    facts, rows, population = (
        build_new_population()
    )

    geometry = geometry_audit()

    anchor_rows, eligibility = (
        tokenizer_eligibility(
            args.tokenizer_snapshot,
            facts,
            rows,
        )
    )

    outputs = write_outputs(
        provenance,
        identity,
        facts,
        rows,
        population,
        geometry,
        anchor_rows,
        eligibility,
    )

    print(
        "RESULT="
        "PASS_PP3_NECESSITY_STATIC_PREPARATION"
    )
    print("PAIR_ID_FIRST=xg1_fact_601")
    print("PAIR_ID_LAST=xg1_fact_900")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("PAIR_ID_OVERLAP_WITH_001_600=0")
    print("CLAIM_OVERLAP_WITH_001_600=0")
    print("EVIDENCE_OVERLAP_WITH_001_600=0")
    print(
        "CLAIM_EVIDENCE_ROW_OVERLAP_WITH_001_600=0"
    )
    print(
        "PP3_GRAM_MAX_ABS_RESIDUAL="
        f"{geometry['pp3_gram_max_abs_residual']:.17g}"
    )
    print(
        "PP5_GRAM_MAX_ABS_RESIDUAL="
        f"{geometry['pp5_gram_max_abs_residual']:.17g}"
    )
    print(
        "PP3_PP5_CROSS_PLANE_MAX_ABS_DOT="
        f"{geometry['pp3_pp5_cross_plane_max_abs_dot']:.17g}"
    )
    print(
        "COEFFICIENT_TRANSFER_MAX_ABS_L2_MISMATCH="
        f"{geometry['coefficient_transfer_max_abs_l2_mismatch']:.17g}"
    )
    print(
        "TOKENIZER_ELIGIBILITY="
        f"{eligibility['result']}"
    )
    print(
        "ANCHOR_ROW_COUNT="
        f"{eligibility['anchor_row_count']}"
    )
    print(
        "ANCHOR_MANIFEST_SHA256="
        f"{eligibility['anchor_manifest_sha256']}"
    )
    print(
        "SOURCE_SHA256="
        f"{outputs['data_hashes'][base.SOURCE_FILE]}"
    )
    print(
        "ROWS_SHA256="
        f"{outputs['data_hashes'][base.ROW_FILE]}"
    )
    print(
        "STRUCTURAL_MANIFEST_SHA256="
        f"{outputs['data_hashes']['structural_manifest.json']}"
    )
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("CHECKPOINT_LOAD_COUNT=0")
    print("GPU_USED=False")
    print("SCIENTIFIC_OUTCOMES_OBSERVED=False")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
