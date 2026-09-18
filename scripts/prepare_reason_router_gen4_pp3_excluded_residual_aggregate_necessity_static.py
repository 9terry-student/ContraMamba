from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_ROOT_ENV = os.environ.get("CONTRAMAMBA_REPO_ROOT")
if not _ROOT_ENV:
    raise RuntimeError("CONTRAMAMBA_REPO_ROOT is required")
ROOT = Path(_ROOT_ENV).resolve()
sys.path.insert(0, str(ROOT))

from scripts import build_reason_router_gen4_xg1_cross_generator_cohort as base
from scripts import build_reason_router_gen4_xg1_fresh_specificity_cohort as fresh
from scripts import prepare_reason_router_gen4_pp3_necessity_static as prior
from scripts import prepare_reason_router_gen4_pp3_pp5_fresh_xg1_specificity as geom


EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"
EXPECTED_HEAD = "c4518d20f4417ca9f057fbd4940c28539e4ffb2c"

DESIGN_COMMIT = EXPECTED_HEAD
DESIGN_PATH = Path(
    "reports/reason_router_gen4_pp3_excluded_residual_aggregate_necessity_design.md"
)
DESIGN_BLOB = "bb9a25a195886bc3b3b6a7bbbfbb57067b23bee8"

PAIR_START = 1501
PAIR_END = 1800
PAIR_FIRST = "xg1_fact_1501"
PAIR_LAST = "xg1_fact_1800"
PAIR_COUNT = 300
ROW_COUNT = 1800

OUTPUT_DATA_DIR = Path(
    "data/reason_router_gen4_xg1_residual_aggregate_necessity_v1"
)
OUTPUT_REPORT_DIR = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_aggregate_necessity_"
    "static_preparation_c4518d2"
)
REPO_SCRIPT_PATH = Path(
    "scripts/"
    "prepare_reason_router_gen4_pp3_excluded_residual_aggregate_necessity_static.py"
)

OLD_COHORTS = (
    (
        1,
        300,
        Path("data/reason_router_gen4_xg1_cross_generator_v1"),
        "fccd6821eeb97194d5b898aca4911eaba71e893df7fe27c910aa37255a5695e0",
        "6ea0484517e0ae7479ad7f3b0a74af4d75f7f7353d29586c597f2a9fee1e649f",
    ),
    (
        301,
        600,
        Path("data/reason_router_gen4_xg1_fresh_specificity_v1"),
        "aa5b8e3cfcbf19e71335ecdbea659326925f8bea33312c2354de670fa7a15cf7",
        "3f28d8a75008d383855313a08168fef1a2b9b37257103636a7f2edb65ce76ad6",
    ),
    (
        601,
        900,
        Path("data/reason_router_gen4_xg1_necessity_v1"),
        "49bec37150630d31bb5f502f49ef23ffc9a75bb93e079127c8c430aae3da6abd",
        "e03534599c07201e371eb07938d8492d22a30de39dcbbb3c0c22300a4ff94224",
    ),
    (
        901,
        1200,
        Path("data/reason_router_gen4_xg1_restoration_sufficiency_v1"),
        "2c700452d818531c46a8ffd473eb6d64d8af29f3a9284da8371f5c9eb2610c21",
        "7ec2ea86f35562394244f6df6e8b098ea3ba8a9bf358868fc614f5029746241c",
    ),
    (
        1201,
        1500,
        Path("data/reason_router_gen4_xg1_residual_template_transport_v1"),
        "9b93eff2399b7f90fb3d63f28f834fb986efcac9d347038c4b265519062d71a6",
        "f60cc028e8276279499290976f18ba503fc7a35a4232c1d768b822075d87ec5e",
    ),
)

PP3_DIR = Path(
    "reports/reason_router_gen4_pp3_xg1_external_transport_preparation_02e4c89"
)
PP5_DIR = Path(
    "reports/reason_router_gen4_pp3_pp5_fresh_xg1_specificity_preparation_0bc49ab"
)

PP3_PLUS_SHA = (
    "66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff"
)
PP3_MINUS_SHA = (
    "ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7"
)
PP5_PLUS_SHA = (
    "7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2"
)
PP5_MINUS_SHA = (
    "311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855"
)

RESIDUAL_PLANE_NUMBERS = (1, 2, 4, 5)
ALL_PLANE_NUMBERS = (1, 2, 3, 4, 5)
VECTOR_DIM = 395
TOL = 1.0e-12

EXPECTED_S = (
    0.87061814189182785,
    0.94755022112376275,
    0.98692852916688512,
    0.99848952673382474,
    0.99986792842854511,
)


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


def git_show_bytes(path: Path, ref: str = "HEAD") -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path.as_posix()}"],
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


def read_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        raw.decode("utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{line_no}",
        )
        rows.append(value)
    return rows


def authenticate() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")

    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")
    require(head == EXPECTED_HEAD, f"HEAD_MISMATCH:{head}")
    require(status == "", "WORKTREE_NOT_CLEAN")

    blob = git("rev-parse", f"HEAD:{DESIGN_PATH.as_posix()}")
    require(blob == DESIGN_BLOB, f"DESIGN_BLOB_DRIFT:{blob}")

    return {
        "branch": branch,
        "head": head,
        "design_commit": DESIGN_COMMIT,
        "design_path": DESIGN_PATH.as_posix(),
        "design_blob": blob,
    }


def validate_old_populations() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_old_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []

    for start, end, directory, expected_source_sha, expected_rows_sha in OLD_COHORTS:
        source_path = directory / base.SOURCE_FILE
        rows_path = directory / base.ROW_FILE

        frozen_source = git_show_bytes(source_path)
        frozen_rows = git_show_bytes(rows_path)

        require(
            sha256_bytes(frozen_source) == expected_source_sha,
            f"OLD_SOURCE_SHA:{start}_{end}",
        )
        require(
            sha256_bytes(frozen_rows) == expected_rows_sha,
            f"OLD_ROWS_SHA:{start}_{end}",
        )

        facts = fresh.build_source_facts(start=start, end=end)
        rows = fresh.materialize_facts(facts)

        require(
            base.jsonl_bytes(facts) == frozen_source,
            f"OLD_SOURCE_REGEN_DRIFT:{start}_{end}",
        )
        require(
            base.jsonl_bytes(rows) == frozen_rows,
            f"OLD_ROWS_REGEN_DRIFT:{start}_{end}",
        )

        parsed_rows = read_jsonl_bytes(frozen_rows)
        require(
            len(parsed_rows) == 6 * (end - start + 1),
            f"OLD_ROW_COUNT:{start}_{end}",
        )
        all_old_rows.extend(parsed_rows)

        identities.append({
            "pair_first": f"xg1_fact_{start:03d}",
            "pair_last": f"xg1_fact_{end:03d}",
            "source_sha256": expected_source_sha,
            "rows_sha256": expected_rows_sha,
            "source_byte_regeneration_identity": True,
            "row_byte_regeneration_identity": True,
        })

    require(len(all_old_rows) == 9000, "OLD_TOTAL_ROW_COUNT")
    return all_old_rows, {
        "cohorts": identities,
        "old_pair_range": "xg1_fact_001..xg1_fact_1500",
        "old_source_pair_count": 1500,
        "old_row_count": 9000,
        "all_old_cohorts_byte_regeneration_identity": True,
    }


def build_new_population(
    old_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    facts = fresh.build_source_facts(start=PAIR_START, end=PAIR_END)
    rows = fresh.materialize_facts(facts)

    require(len(facts) == PAIR_COUNT, "NEW_FACT_COUNT")
    require(len(rows) == ROW_COUNT, "NEW_ROW_COUNT")

    expected_ids = [
        f"xg1_fact_{i:03d}"
        for i in range(PAIR_START, PAIR_END + 1)
    ]
    require(
        [str(x["pair_id"]) for x in facts] == expected_ids,
        "NEW_PAIR_ORDER",
    )

    old_pair_ids = {str(row["source_pair_id"]) for row in old_rows}
    new_pair_ids = {str(row["source_pair_id"]) for row in rows}
    old_claims = {str(row["claim"]) for row in old_rows}
    new_claims = {str(row["claim"]) for row in rows}
    old_evidence = {str(row["evidence"]) for row in old_rows}
    new_evidence = {str(row["evidence"]) for row in rows}
    old_claim_evidence = {
        (str(row["claim"]), str(row["evidence"]))
        for row in old_rows
    }
    new_claim_evidence = {
        (str(row["claim"]), str(row["evidence"]))
        for row in rows
    }

    require(not (old_pair_ids & new_pair_ids), "PAIR_ID_OVERLAP")
    require(not (old_claims & new_claims), "CLAIM_OVERLAP")
    require(not (old_evidence & new_evidence), "EVIDENCE_OVERLAP")
    require(
        not (old_claim_evidence & new_claim_evidence),
        "CLAIM_EVIDENCE_ROW_OVERLAP",
    )

    return facts, rows, {
        "source_raw": base.jsonl_bytes(facts),
        "rows_raw": base.jsonl_bytes(rows),
        "pair_id_overlap_with_001_1500": 0,
        "claim_overlap_with_001_1500": 0,
        "evidence_overlap_with_001_1500": 0,
        "claim_evidence_row_overlap_with_001_1500": 0,
    }


def reconstruct_geometry() -> dict[str, Any]:
    xg2_plan = geom.load_plan("xg2")
    xg4_plan = geom.load_plan("xg4")

    b2 = geom.fs.reconstruct_family_basis(
        "xg2",
        xg2_plan,
    )["basis"].detach().cpu().to(torch.float64).contiguous()
    b4 = geom.fs.reconstruct_family_basis(
        "xg4",
        xg4_plan,
    )["basis"].detach().cpu().to(torch.float64).contiguous()

    require(tuple(b2.shape) == (VECTOR_DIM, 5), "XG2_BASIS_SHAPE")
    require(tuple(b4.shape) == (VECTOR_DIM, 5), "XG4_BASIS_SHAPE")

    vectors: dict[int, dict[str, Any]] = {}

    for zero_index in range(5):
        plane = zero_index + 1
        a, b, c, s = geom.principal_pair(b2, b4, zero_index)
        plus, minus = geom.span_2d_eigh_raw(a, b, c, s)
        plus_raw = geom.raw_f64le(plus)
        minus_raw = geom.raw_f64le(minus)

        require(
            abs(s - EXPECTED_S[zero_index]) <= 2.0e-12,
            f"PLANE_EIGENVALUE:P{plane}:{s}",
        )

        vectors[plane] = {
            "plus": plus,
            "minus": minus,
            "plus_raw": plus_raw,
            "minus_raw": minus_raw,
            "plus_sha256": sha256_bytes(plus_raw),
            "minus_sha256": sha256_bytes(minus_raw),
            "metrics": geom.vector_metrics(plus, minus),
            "c": float(c),
            "s": float(s),
        }

    # Reproduce the already-frozen PP3 and PP5 bytes before freezing any new plane.
    checks = (
        (
            3,
            PP3_DIR / "pp3_plus.f64le",
            PP3_DIR / "pp3_minus.f64le",
            PP3_PLUS_SHA,
            PP3_MINUS_SHA,
        ),
        (
            5,
            PP5_DIR / "pp5_plus.f64le",
            PP5_DIR / "pp5_minus.f64le",
            PP5_PLUS_SHA,
            PP5_MINUS_SHA,
        ),
    )
    for plane, plus_rel, minus_rel, plus_sha, minus_sha in checks:
        plus_path = ROOT / plus_rel
        minus_path = ROOT / minus_rel
        require(plus_path.is_file(), f"FROZEN_PLUS_MISSING:P{plane}")
        require(minus_path.is_file(), f"FROZEN_MINUS_MISSING:P{plane}")
        require(sha256_file(plus_path) == plus_sha, f"FROZEN_PLUS_SHA:P{plane}")
        require(sha256_file(minus_path) == minus_sha, f"FROZEN_MINUS_SHA:P{plane}")
        require(
            vectors[plane]["plus_sha256"] == plus_sha
            and vectors[plane]["plus_raw"] == plus_path.read_bytes(),
            f"FROZEN_PLUS_REPRODUCTION:P{plane}",
        )
        require(
            vectors[plane]["minus_sha256"] == minus_sha
            and vectors[plane]["minus_raw"] == minus_path.read_bytes(),
            f"FROZEN_MINUS_REPRODUCTION:P{plane}",
        )

    columns: list[torch.Tensor] = []
    labels: list[str] = []
    for plane in ALL_PLANE_NUMBERS:
        columns.extend([vectors[plane]["plus"], vectors[plane]["minus"]])
        labels.extend([f"P{plane}+", f"P{plane}-"])

    matrix = torch.stack(columns, dim=1)
    gram = matrix.T @ matrix
    identity = torch.eye(10, dtype=torch.float64)
    gram_max = float(torch.max(torch.abs(gram - identity)).item())
    require(gram_max <= TOL, f"FULL_PRINCIPAL_GRAM:{gram_max}")

    pp3_cols = torch.stack(
        [vectors[3]["plus"], vectors[3]["minus"]],
        dim=1,
    )
    residual_cols = torch.stack(
        [
            vectors[p][sign]
            for p in RESIDUAL_PLANE_NUMBERS
            for sign in ("plus", "minus")
        ],
        dim=1,
    )
    residual_pp3_cross = residual_cols.T @ pp3_cols
    residual_pp3_cross_max = float(
        torch.max(torch.abs(residual_pp3_cross)).item()
    )
    require(
        residual_pp3_cross_max <= TOL,
        f"RESIDUAL_PP3_CROSS:{residual_pp3_cross_max}",
    )

    # Deterministic audit of the blockwise 90-degree control algebra.
    coefficient_probes = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0, -1.0, 2.0, 3.0, -4.0],
            [math.pi, math.e, -0.125, 0.875, 1.5, -2.5, 0.25, -0.75],
            [0.5, -0.5, 1.25, 2.25, -3.5, 1.0, 4.0, 2.0],
            [-7.0, 3.0, 2.0, -5.0, 0.125, 0.25, -0.5, 1.0],
        ],
        dtype=np.float64,
    )
    residual_matrix = np.column_stack([
        np.asarray(
            vectors[p][sign].detach().cpu().numpy(),
            dtype=np.float64,
        )
        for p in RESIDUAL_PLANE_NUMBERS
        for sign in ("plus", "minus")
    ])

    max_norm_mismatch = 0.0
    max_abs_dot = 0.0
    for coeff in coefficient_probes:
        rotated = coeff.copy()
        for offset in range(0, 8, 2):
            a = coeff[offset]
            b = coeff[offset + 1]
            rotated[offset] = -b
            rotated[offset + 1] = a

        treatment = -(residual_matrix @ coeff)
        control = -(residual_matrix @ rotated)

        max_norm_mismatch = max(
            max_norm_mismatch,
            abs(float(np.linalg.norm(treatment)) - float(np.linalg.norm(control))),
        )
        max_abs_dot = max(
            max_abs_dot,
            abs(float(np.dot(treatment, control))),
        )

    require(
        max_norm_mismatch <= TOL,
        f"QUARTER_TURN_NORM_MISMATCH:{max_norm_mismatch}",
    )
    require(
        max_abs_dot <= TOL,
        f"QUARTER_TURN_DOT:{max_abs_dot}",
    )

    plane_manifest: dict[str, Any] = {}
    for plane in ALL_PLANE_NUMBERS:
        item = vectors[plane]
        plane_manifest[f"P{plane}"] = {
            "scientific_principal_pair_number": plane,
            "plus_sha256": item["plus_sha256"],
            "minus_sha256": item["minus_sha256"],
            "c": item["c"],
            "s": item["s"],
            **item["metrics"],
            "existing_frozen_identity":
                plane in {3, 5},
        }

    return {
        "vectors": vectors,
        "manifest": {
            "schema_version":
                "GEN4_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_GEOMETRY_V1",
            "result":
                "PASS_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_GEOMETRY_STATIC",
            "algorithm": (
                "frozen XG2/XG4 Phase-1 bases; principal_pair SVD; "
                "2x2 torch.linalg.eigh; max-absolute ambient coordinate "
                "positive; raw little-endian float64"
            ),
            "ambient_dim": VECTOR_DIM,
            "residual_planes": ["P1", "P2", "P4", "P5"],
            "excluded_plane": "P3",
            "all_planes": plane_manifest,
            "full_principal_10_vector_gram_max_abs_residual": gram_max,
            "residual_to_pp3_max_abs_dot": residual_pp3_cross_max,
            "quarter_turn_control": {
                "coefficient_map": "(a,b)->(-b,a) independently per residual plane",
                "probe_count": int(coefficient_probes.shape[0]),
                "max_abs_l2_mismatch": max_norm_mismatch,
                "max_abs_treatment_control_dot": max_abs_dot,
                "norm_identity_proved_by_orthonormal_residual_basis": True,
                "orthogonality_identity_proved_by_blockwise_quarter_turn": True,
                "response_guided_weighting": False,
                "response_guided_plane_selection": False,
            },
            "pp3_frozen_plus_sha256": PP3_PLUS_SHA,
            "pp3_frozen_minus_sha256": PP3_MINUS_SHA,
            "pp3_frozen_bytes_reproduced_exactly": True,
            "pp5_frozen_plus_sha256": PP5_PLUS_SHA,
            "pp5_frozen_minus_sha256": PP5_MINUS_SHA,
            "pp5_frozen_bytes_reproduced_exactly": True,
            "model_forward_count": 0,
            "checkpoint_load_count": 0,
            "gpu_used": False,
            "scientific_outcomes_observed": False,
        },
    }


def tokenizer_eligibility(
    tokenizer_snapshot: Path,
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_authority = prior.AUTHORITY_COMMIT
    old_first = prior.PAIR_FIRST
    old_last = prior.PAIR_LAST
    try:
        prior.AUTHORITY_COMMIT = DESIGN_COMMIT
        prior.PAIR_FIRST = PAIR_FIRST
        prior.PAIR_LAST = PAIR_LAST
        anchor_rows, summary = prior.tokenizer_eligibility(
            tokenizer_snapshot,
            facts,
            rows,
        )
    finally:
        prior.AUTHORITY_COMMIT = old_authority
        prior.PAIR_FIRST = old_first
        prior.PAIR_LAST = old_last

    for row in anchor_rows:
        row["schema_version"] = (
            "GEN4_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_"
            "XG1_TOKENIZER_ANCHOR_V1"
        )

    anchor_raw = b"".join(canonical_json_bytes(row) for row in anchor_rows)
    summary = dict(summary)
    summary.update({
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_"
            "XG1_TOKENIZER_ELIGIBILITY_SUMMARY_V1",
        "authority_commit": DESIGN_COMMIT,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "anchor_manifest_sha256": sha256_bytes(anchor_raw),
    })

    require(summary["result"] == "PASS_300_OF_300", "TOKENIZER_RESULT")
    require(int(summary["source_pair_count"]) == PAIR_COUNT, "TOKENIZER_PAIR_COUNT")
    require(int(summary["anchor_row_count"]) == ROW_COUNT, "TOKENIZER_ANCHOR_COUNT")
    require(summary["model_forward_count"] == 0, "TOKENIZER_MODEL_FORWARD")
    require(summary["checkpoint_load_count"] == 0, "TOKENIZER_CHECKPOINT_LOAD")
    require(summary["gpu_used"] is False, "TOKENIZER_GPU")
    require(
        int(summary["complete_source_pair_count"]) == PAIR_COUNT,
        "TOKENIZER_COMPLETE_PAIR_COUNT",
    )

    return anchor_rows, summary


def write_outputs(
    provenance: Mapping[str, Any],
    old_identity: Mapping[str, Any],
    population: Mapping[str, Any],
    geometry: Mapping[str, Any],
    anchor_rows: Sequence[Mapping[str, Any]],
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    data_dir = ROOT / OUTPUT_DATA_DIR
    report_dir = ROOT / OUTPUT_REPORT_DIR
    repo_script = ROOT / REPO_SCRIPT_PATH

    require(not data_dir.exists(), f"DATA_OUTPUT_COLLISION:{data_dir}")
    require(not report_dir.exists(), f"REPORT_OUTPUT_COLLISION:{report_dir}")
    require(not repo_script.exists(), f"SCRIPT_OUTPUT_COLLISION:{repo_script}")

    data_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)

    source_raw = population["source_raw"]
    rows_raw = population["rows_raw"]

    structural = {
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_XG1_STRUCTURAL_V1",
        "result":
            "PASS_XG1_1501_1800_RESIDUAL_AGGREGATE_NECESSITY_STRUCTURAL",
        "design_commit": DESIGN_COMMIT,
        "generator_family": base.GENERATOR_FAMILY,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "source_pair_count": PAIR_COUNT,
        "row_count": ROW_COUNT,
        "rows_per_pair": 6,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(rows_raw),
        "prior_population_identity": dict(old_identity),
        "pair_id_overlap_with_001_1500": 0,
        "claim_overlap_with_001_1500": 0,
        "evidence_overlap_with_001_1500": 0,
        "claim_evidence_row_overlap_with_001_1500": 0,
        "deterministic_generator_semantics": True,
        "labels_present": False,
        "model_geometry_present": False,
        "endpoint_values_present": False,
        "response_fields_present": False,
        "tokenizer_executed": False,
        "checkpoint_loaded": False,
        "model_executed": False,
        "cuda_executed": False,
    }

    data_payloads = {
        base.SOURCE_FILE: source_raw,
        base.ROW_FILE: rows_raw,
        "structural_manifest.json": pretty_json_bytes(structural),
    }
    data_hashes: dict[str, str] = {}
    for name, raw in data_payloads.items():
        (data_dir / name).write_bytes(raw)
        data_hashes[name] = sha256_bytes(raw)

    (data_dir / "SHA256SUMS.txt").write_bytes(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(data_hashes.items())
        ).encode("utf-8")
    )

    # Freeze only the pre-specified residual-plane bytes; PP3 remains referenced
    # by its already-frozen exact-byte artifacts and is re-authenticated above.
    vector_hashes: dict[str, str] = {}
    for plane in RESIDUAL_PLANE_NUMBERS:
        for sign in ("plus", "minus"):
            name = f"p{plane}_{sign}.f64le"
            raw = geometry["vectors"][plane][f"{sign}_raw"]
            (report_dir / name).write_bytes(raw)
            vector_hashes[name] = sha256_bytes(raw)

    geometry_raw = pretty_json_bytes(geometry["manifest"])
    (report_dir / "geometry_manifest.json").write_bytes(geometry_raw)

    anchor_raw = b"".join(canonical_json_bytes(row) for row in anchor_rows)
    eligibility_raw = pretty_json_bytes(eligibility)

    preparation = {
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_STATIC_PREPARATION_V1",
        "result":
            "PASS_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_STATIC_PREPARATION",
        "design": dict(provenance),
        "population": {
            "data_dir": OUTPUT_DATA_DIR.as_posix(),
            "pair_id_first": PAIR_FIRST,
            "pair_id_last": PAIR_LAST,
            "source_pair_count": PAIR_COUNT,
            "row_count": ROW_COUNT,
            "source_sha256": data_hashes[base.SOURCE_FILE],
            "rows_sha256": data_hashes[base.ROW_FILE],
            "structural_manifest_sha256":
                data_hashes["structural_manifest.json"],
            "pair_id_overlap_with_001_1500": 0,
            "claim_overlap_with_001_1500": 0,
            "evidence_overlap_with_001_1500": 0,
            "claim_evidence_row_overlap_with_001_1500": 0,
        },
        "geometry": {
            "result": geometry["manifest"]["result"],
            "residual_planes": geometry["manifest"]["residual_planes"],
            "excluded_plane": "P3",
            "full_principal_10_vector_gram_max_abs_residual":
                geometry["manifest"][
                    "full_principal_10_vector_gram_max_abs_residual"
                ],
            "residual_to_pp3_max_abs_dot":
                geometry["manifest"]["residual_to_pp3_max_abs_dot"],
            "quarter_turn_control":
                geometry["manifest"]["quarter_turn_control"],
            "residual_vector_sha256": dict(sorted(vector_hashes.items())),
            "pp3_plus_sha256": PP3_PLUS_SHA,
            "pp3_minus_sha256": PP3_MINUS_SHA,
            "pp3_frozen_bytes_reproduced_exactly": True,
        },
        "tokenizer_eligibility": {
            "result": eligibility["result"],
            "source_pair_count": eligibility["source_pair_count"],
            "anchor_row_count": eligibility["anchor_row_count"],
            "anchor_manifest_sha256": sha256_bytes(anchor_raw),
            "tokenizer_revision": prior.TOKENIZER_REVISION,
        },
        "future_execution_contract": {
            "conditions": ["native", "residual_neutralized", "quarter_turn_control"],
            "pair_count": PAIR_COUNT,
            "directions_per_condition": 10,
            "forwards_per_direction": 4,
            "forwards_per_pair": 120,
            "scientific_model_forward_budget": 36000,
            "confirmatory_p_value_count": 1,
            "execution_authorized": False,
        },
        "scientific_model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_outcomes_observed": False,
        "primary_inference_executed": False,
    }

    report_payloads = {
        "geometry_manifest.json": geometry_raw,
        "tokenizer_anchor_manifest.jsonl": anchor_raw,
        "tokenizer_eligibility_summary.json": eligibility_raw,
        "preparation_manifest.json": pretty_json_bytes(preparation),
    }

    report_hashes: dict[str, str] = dict(vector_hashes)
    for name, raw in report_payloads.items():
        if name != "geometry_manifest.json":
            (report_dir / name).write_bytes(raw)
        report_hashes[name] = sha256_bytes(raw)

    (report_dir / "SHA256SUMS.txt").write_bytes(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(report_hashes.items())
        ).encode("utf-8")
    )

    # Preserve the exact external preparation code as the repository script.
    source_bytes = Path(__file__).read_bytes()
    repo_script.write_bytes(source_bytes)

    return {
        "data_hashes": data_hashes,
        "report_hashes": report_hashes,
        "script_sha256": sha256_bytes(source_bytes),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only static preparation for prospective PP3-excluded residual "
            "aggregate necessity on fresh XG1 1501..1800."
        )
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    args = parser.parse_args(argv)

    tokenizer_snapshot = args.tokenizer_snapshot.resolve()
    require(
        tokenizer_snapshot.is_dir(),
        f"TOKENIZER_SNAPSHOT_MISSING:{tokenizer_snapshot}",
    )
    require(
        not tokenizer_snapshot.is_relative_to(ROOT.resolve()),
        "TOKENIZER_SNAPSHOT_MUST_BE_EXTERNAL_TO_REPO",
    )

    provenance = authenticate()
    old_rows, old_identity = validate_old_populations()
    facts, rows, population = build_new_population(old_rows)
    geometry = reconstruct_geometry()
    anchor_rows, eligibility = tokenizer_eligibility(
        tokenizer_snapshot,
        facts,
        rows,
    )
    outputs = write_outputs(
        provenance,
        old_identity,
        population,
        geometry,
        anchor_rows,
        eligibility,
    )

    print("RESULT=PASS_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_STATIC_PREPARATION")
    print(f"PAIR_ID_FIRST={PAIR_FIRST}")
    print(f"PAIR_ID_LAST={PAIR_LAST}")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("OLD_XG1_001_1500_BYTE_REGENERATION_IDENTITY=True")
    print("PAIR_ID_OVERLAP_WITH_001_1500=0")
    print("CLAIM_OVERLAP_WITH_001_1500=0")
    print("EVIDENCE_OVERLAP_WITH_001_1500=0")
    print("CLAIM_EVIDENCE_ROW_OVERLAP_WITH_001_1500=0")
    print(
        "FULL_PRINCIPAL_10_VECTOR_GRAM_MAX_ABS_RESIDUAL="
        f"{geometry['manifest']['full_principal_10_vector_gram_max_abs_residual']:.17g}"
    )
    print(
        "RESIDUAL_TO_PP3_MAX_ABS_DOT="
        f"{geometry['manifest']['residual_to_pp3_max_abs_dot']:.17g}"
    )
    print(
        "QUARTER_TURN_MAX_ABS_L2_MISMATCH="
        f"{geometry['manifest']['quarter_turn_control']['max_abs_l2_mismatch']:.17g}"
    )
    print(
        "QUARTER_TURN_MAX_ABS_TREATMENT_CONTROL_DOT="
        f"{geometry['manifest']['quarter_turn_control']['max_abs_treatment_control_dot']:.17g}"
    )
    print("PP3_FROZEN_BYTES_REPRODUCED_EXACTLY=True")
    print("PP5_FROZEN_BYTES_REPRODUCED_EXACTLY=True")

    for plane in RESIDUAL_PLANE_NUMBERS:
        for sign in ("plus", "minus"):
            name = f"p{plane}_{sign}.f64le"
            print(
                f"{name.upper().replace('.', '_')}_SHA256="
                f"{outputs['report_hashes'][name]}"
            )

    print(f"TOKENIZER_ELIGIBILITY={eligibility['result']}")
    print(f"ANCHOR_ROW_COUNT={eligibility['anchor_row_count']}")
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
    print(
        "GEOMETRY_MANIFEST_SHA256="
        f"{outputs['report_hashes']['geometry_manifest.json']}"
    )
    print(
        "PREPARATION_MANIFEST_SHA256="
        f"{outputs['report_hashes']['preparation_manifest.json']}"
    )
    print(f"REPO_SCRIPT_SHA256={outputs['script_sha256']}")
    print("FUTURE_SCIENTIFIC_MODEL_FORWARD_BUDGET=36000")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("CHECKPOINT_LOAD_COUNT=0")
    print("GPU_USED=False")
    print("SCIENTIFIC_OUTCOMES_OBSERVED=False")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("EXECUTION_AUTHORIZED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
