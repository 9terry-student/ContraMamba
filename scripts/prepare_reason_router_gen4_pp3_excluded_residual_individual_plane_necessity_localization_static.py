from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT_ENV = os.environ.get("CONTRAMAMBA_REPO_ROOT")
if not _ROOT_ENV:
    raise RuntimeError("CONTRAMAMBA_REPO_ROOT is required")
ROOT = Path(_ROOT_ENV).resolve()
sys.path.insert(0, str(ROOT))

from scripts import (
    prepare_reason_router_gen4_pp3_excluded_residual_aggregate_necessity_static
    as aggregate
)

EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"
EXPECTED_HEAD = "d046a8e03e7522a72dfbd08cc9129b769cd5686a"
DESIGN_COMMIT = EXPECTED_HEAD
DESIGN_PATH = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_individual_plane_"
    "necessity_localization_design.md"
)
DESIGN_BLOB = "191e3a74647afb9a007a85bfdb7cefa8d79c9010"

PAIR_START = 1801
PAIR_END = 2100
PAIR_FIRST = "xg1_fact_1801"
PAIR_LAST = "xg1_fact_2100"
PAIR_COUNT = 300
ROW_COUNT = 1800
OLD_PAIR_COUNT = 1800
OLD_ROW_COUNT = 10800

RESIDUAL_PLANES = (1, 2, 4, 5)
TOL = 1.0e-12
FUTURE_FORWARDS = 108000
RAW_P_VALUE_COUNT = 4
FWER_ALPHA = 0.05
MULTIPLICITY = "Holm step-down across exactly P1,P2,P4,P5"

OUTPUT_DATA_DIR = Path(
    "data/reason_router_gen4_xg1_residual_individual_plane_necessity_v1"
)
OUTPUT_REPORT_DIR = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_individual_plane_"
    "necessity_localization_static_preparation_d046a8e"
)
REPO_SCRIPT_PATH = Path(
    "scripts/"
    "prepare_reason_router_gen4_pp3_excluded_residual_individual_plane_"
    "necessity_localization_static.py"
)

BASE = aggregate.base
FRESH = aggregate.fresh

OLD_COHORTS = (
    (
        1, 300,
        Path("data/reason_router_gen4_xg1_cross_generator_v1"),
        "fccd6821eeb97194d5b898aca4911eaba71e893df7fe27c910aa37255a5695e0",
        "6ea0484517e0ae7479ad7f3b0a74af4d75f7f7353d29586c597f2a9fee1e649f",
    ),
    (
        301, 600,
        Path("data/reason_router_gen4_xg1_fresh_specificity_v1"),
        "aa5b8e3cfcbf19e71335ecdbea659326925f8bea33312c2354de670fa7a15cf7",
        "3f28d8a75008d383855313a08168fef1a2b9b37257103636a7f2edb65ce76ad6",
    ),
    (
        601, 900,
        Path("data/reason_router_gen4_xg1_necessity_v1"),
        "49bec37150630d31bb5f502f49ef23ffc9a75bb93e079127c8c430aae3da6abd",
        "e03534599c07201e371eb07938d8492d22a30de39dcbbb3c0c22300a4ff94224",
    ),
    (
        901, 1200,
        Path("data/reason_router_gen4_xg1_restoration_sufficiency_v1"),
        "2c700452d818531c46a8ffd473eb6d64d8af29f3a9284da8371f5c9eb2610c21",
        "7ec2ea86f35562394244f6df6e8b098ea3ba8a9bf358868fc614f5029746241c",
    ),
    (
        1201, 1500,
        Path("data/reason_router_gen4_xg1_residual_template_transport_v1"),
        "9b93eff2399b7f90fb3d63f28f834fb986efcac9d347038c4b265519062d71a6",
        "f60cc028e8276279499290976f18ba503fc7a35a4232c1d768b822075d87ec5e",
    ),
    (
        1501, 1800,
        Path("data/reason_router_gen4_xg1_residual_aggregate_necessity_v1"),
        "b8b2186fb4bdf5d9781efb9ab6eb56eb8a3d51052618ab7cf7d257aa6e72df21",
        "16d9646cab9fa10d0241db2eeecf15164a04eb80740b873c2db6e8f02acab578",
    ),
)

FROZEN_RESIDUAL_DIR = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_aggregate_necessity_"
    "static_preparation_c4518d2"
)
FROZEN_RESIDUAL_SHA = {
    "p1_plus.f64le":
        "209da6bf007c0eadcd78db648835e6ee174d0290726acb61731007eff436aef1",
    "p1_minus.f64le":
        "b6470bbec5a586f34e87d6e87f32c7f1778af7a55622d60508d68c6679c1ab26",
    "p2_plus.f64le":
        "a0f48476f77e9e1876adc9919ba61a3c4e6ac894001f789216245d2ff3c945ea",
    "p2_minus.f64le":
        "b48683f584ab31e31d542fc9b20327d19c0019cca02a41c97db0418d2bd69a78",
    "p4_plus.f64le":
        "494f7b5de31673d53b368341d7960781767f948ed32afbae32b64b06a68f5cd8",
    "p4_minus.f64le":
        "5583cb0cb6ab6fe0a2abae926ed56e3dd800d524d0539acbc54ced9ca8dea079",
    "p5_plus.f64le":
        "7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2",
    "p5_minus.f64le":
        "311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855",
}


class StaticPreparationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StaticPreparationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
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


def git_show_bytes(path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticPreparationError(
            f"GIT_SHOW_FAILURE:{path.as_posix()}"
        ) from exc


def canonical(value: Mapping[str, Any]) -> bytes:
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


def pretty(value: Mapping[str, Any]) -> bytes:
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


def read_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        raw.decode("utf-8-sig").splitlines(), 1
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


def validate_old_populations():
    all_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []

    for start, end, directory, source_sha, rows_sha in OLD_COHORTS:
        source_path = directory / BASE.SOURCE_FILE
        rows_path = directory / BASE.ROW_FILE

        frozen_source = git_show_bytes(source_path)
        frozen_rows = git_show_bytes(rows_path)

        require(
            sha256_bytes(frozen_source) == source_sha,
            f"OLD_SOURCE_SHA:{start}_{end}",
        )
        require(
            sha256_bytes(frozen_rows) == rows_sha,
            f"OLD_ROWS_SHA:{start}_{end}",
        )

        facts = FRESH.build_source_facts(start=start, end=end)
        rows = FRESH.materialize_facts(facts)

        require(
            BASE.jsonl_bytes(facts) == frozen_source,
            f"OLD_SOURCE_REGEN_DRIFT:{start}_{end}",
        )
        require(
            BASE.jsonl_bytes(rows) == frozen_rows,
            f"OLD_ROWS_REGEN_DRIFT:{start}_{end}",
        )

        parsed = read_jsonl_bytes(frozen_rows)
        require(
            len(parsed) == 6 * (end - start + 1),
            f"OLD_ROW_COUNT:{start}_{end}",
        )
        all_rows.extend(parsed)

        identities.append({
            "pair_first": f"xg1_fact_{start:03d}",
            "pair_last": f"xg1_fact_{end:03d}",
            "source_sha256": source_sha,
            "rows_sha256": rows_sha,
            "source_byte_regeneration_identity": True,
            "row_byte_regeneration_identity": True,
        })

    require(len(all_rows) == OLD_ROW_COUNT, "OLD_TOTAL_ROW_COUNT")

    return all_rows, {
        "cohorts": identities,
        "old_pair_range": "xg1_fact_001..xg1_fact_1800",
        "old_source_pair_count": OLD_PAIR_COUNT,
        "old_row_count": OLD_ROW_COUNT,
        "all_old_cohorts_byte_regeneration_identity": True,
    }


def build_new_population(old_rows):
    facts = FRESH.build_source_facts(
        start=PAIR_START,
        end=PAIR_END,
    )
    rows = FRESH.materialize_facts(facts)

    require(len(facts) == PAIR_COUNT, "NEW_FACT_COUNT")
    require(len(rows) == ROW_COUNT, "NEW_ROW_COUNT")

    expected_ids = [
        f"xg1_fact_{index:03d}"
        for index in range(PAIR_START, PAIR_END + 1)
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

    old_pairs = {
        (str(row["claim"]), str(row["evidence"]))
        for row in old_rows
    }
    new_pairs = {
        (str(row["claim"]), str(row["evidence"]))
        for row in rows
    }

    require(not (old_pair_ids & new_pair_ids), "PAIR_ID_OVERLAP")
    require(not (old_claims & new_claims), "CLAIM_OVERLAP")
    require(not (old_evidence & new_evidence), "EVIDENCE_OVERLAP")
    require(not (old_pairs & new_pairs), "CLAIM_EVIDENCE_OVERLAP")

    return facts, rows, {
        "source_raw": BASE.jsonl_bytes(facts),
        "rows_raw": BASE.jsonl_bytes(rows),
        "pair_id_overlap_with_001_1800": 0,
        "claim_overlap_with_001_1800": 0,
        "evidence_overlap_with_001_1800": 0,
        "claim_evidence_row_overlap_with_001_1800": 0,
    }


def validate_geometry():
    reconstructed = aggregate.reconstruct_geometry()
    vectors = reconstructed["vectors"]

    frozen_sha: dict[str, str] = {}
    for plane in RESIDUAL_PLANES:
        for sign in ("plus", "minus"):
            name = f"p{plane}_{sign}.f64le"
            path = ROOT / FROZEN_RESIDUAL_DIR / name
            require(path.is_file(), f"FROZEN_VECTOR_MISSING:{name}")
            observed = sha256_file(path)
            expected = FROZEN_RESIDUAL_SHA[name]
            require(
                observed == expected,
                f"FROZEN_VECTOR_SHA:{name}:{observed}",
            )
            require(
                sha256_bytes(
                    reconstructed["vectors"][plane][f"{sign}_raw"]
                ) == expected,
                f"FROZEN_VECTOR_RECONSTRUCTION:{name}",
            )
            frozen_sha[name] = observed

    # Principal-pair geometry is already orthonormal; audit each plane's own
    # 90-degree matched control independently, as required by this design.
    coefficient_probes = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [math.pi, math.e],
            [-0.125, 0.875],
            [3.0, -4.0],
            [-7.0, 3.0],
        ],
        dtype=np.float64,
    )

    plane_controls: dict[str, Any] = {}
    global_max_norm_mismatch = 0.0
    global_max_abs_dot = 0.0
    global_max_other_plane_projection = 0.0
    global_max_pp3_projection = 0.0

    for plane in RESIDUAL_PLANES:
        plus = np.asarray(
            vectors[plane]["plus"].detach().cpu().numpy(),
            dtype=np.float64,
        )
        minus = np.asarray(
            vectors[plane]["minus"].detach().cpu().numpy(),
            dtype=np.float64,
        )

        local_max_norm_mismatch = 0.0
        local_max_abs_dot = 0.0
        local_max_other = 0.0
        local_max_pp3 = 0.0

        for a, b in coefficient_probes:
            native_component = a * plus + b * minus
            quarter_component = -b * plus + a * minus
            treatment = -native_component
            control = -quarter_component

            local_max_norm_mismatch = max(
                local_max_norm_mismatch,
                abs(
                    float(np.linalg.norm(treatment))
                    - float(np.linalg.norm(control))
                ),
            )
            local_max_abs_dot = max(
                local_max_abs_dot,
                abs(float(np.dot(treatment, control))),
            )

            for other in (1, 2, 3, 4, 5):
                if other == plane:
                    continue
                for sign in ("plus", "minus"):
                    vec = np.asarray(
                        vectors[other][sign]
                        .detach().cpu().numpy(),
                        dtype=np.float64,
                    )
                    projection = max(
                        abs(float(np.dot(treatment, vec))),
                        abs(float(np.dot(control, vec))),
                    )
                    if other == 3:
                        local_max_pp3 = max(local_max_pp3, projection)
                    else:
                        local_max_other = max(local_max_other, projection)

        require(
            local_max_norm_mismatch <= TOL,
            f"P{plane}:L2_MISMATCH:{local_max_norm_mismatch}",
        )
        require(
            local_max_abs_dot <= TOL,
            f"P{plane}:DOT:{local_max_abs_dot}",
        )
        require(
            local_max_other <= TOL,
            f"P{plane}:OTHER_PLANE_PROJECTION:{local_max_other}",
        )
        require(
            local_max_pp3 <= TOL,
            f"P{plane}:PP3_PROJECTION:{local_max_pp3}",
        )

        global_max_norm_mismatch = max(
            global_max_norm_mismatch,
            local_max_norm_mismatch,
        )
        global_max_abs_dot = max(
            global_max_abs_dot,
            local_max_abs_dot,
        )
        global_max_other_plane_projection = max(
            global_max_other_plane_projection,
            local_max_other,
        )
        global_max_pp3_projection = max(
            global_max_pp3_projection,
            local_max_pp3,
        )

        plane_controls[f"P{plane}"] = {
            "coefficient_map": "(a,b)->(-b,a)",
            "probe_count": int(coefficient_probes.shape[0]),
            "max_abs_l2_mismatch": local_max_norm_mismatch,
            "max_abs_treatment_control_dot": local_max_abs_dot,
            "max_abs_projection_onto_other_residual_planes":
                local_max_other,
            "max_abs_projection_onto_pp3":
                local_max_pp3,
            "equal_norm_identity": True,
            "orthogonal_control_identity": True,
            "response_guided_weighting": False,
            "response_guided_plane_selection": False,
        }

    manifest = {
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_"
            "LOCALIZATION_GEOMETRY_V1",
        "result":
            "PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_"
            "LOCALIZATION_GEOMETRY_STATIC",
        "design_commit": DESIGN_COMMIT,
        "residual_plane_order": ["P1", "P2", "P4", "P5"],
        "excluded_plane": "P3",
        "frozen_residual_vector_sha256": dict(sorted(frozen_sha.items())),
        "full_principal_10_vector_gram_max_abs_residual":
            reconstructed["manifest"][
                "full_principal_10_vector_gram_max_abs_residual"
            ],
        "residual_to_pp3_max_abs_dot":
            reconstructed["manifest"]["residual_to_pp3_max_abs_dot"],
        "per_plane_quarter_turn_control": plane_controls,
        "global_max_abs_l2_mismatch": global_max_norm_mismatch,
        "global_max_abs_treatment_control_dot": global_max_abs_dot,
        "global_max_abs_projection_onto_other_residual_planes":
            global_max_other_plane_projection,
        "global_max_abs_projection_onto_pp3":
            global_max_pp3_projection,
        "all_residual_vectors_reproduced_exactly": True,
        "scientific_model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_outcomes_observed": False,
        "primary_inference_executed": False,
    }
    return manifest


def tokenizer_eligibility(tokenizer_snapshot, facts, rows):
    # Reuse the previously validated tokenizer/anchor eligibility path while
    # replacing only the prospective authority and pair interval.
    old_design = aggregate.DESIGN_COMMIT
    old_first = aggregate.PAIR_FIRST
    old_last = aggregate.PAIR_LAST
    try:
        aggregate.DESIGN_COMMIT = DESIGN_COMMIT
        aggregate.PAIR_FIRST = PAIR_FIRST
        aggregate.PAIR_LAST = PAIR_LAST
        anchor_rows, summary = aggregate.tokenizer_eligibility(
            tokenizer_snapshot,
            facts,
            rows,
        )
    finally:
        aggregate.DESIGN_COMMIT = old_design
        aggregate.PAIR_FIRST = old_first
        aggregate.PAIR_LAST = old_last

    for row in anchor_rows:
        row["schema_version"] = (
            "GEN4_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_"
            "LOCALIZATION_XG1_TOKENIZER_ANCHOR_V1"
        )

    anchor_raw = b"".join(canonical(row) for row in anchor_rows)

    summary = dict(summary)
    summary.update({
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_"
            "LOCALIZATION_XG1_TOKENIZER_ELIGIBILITY_SUMMARY_V1",
        "authority_commit": DESIGN_COMMIT,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "anchor_manifest_sha256": sha256_bytes(anchor_raw),
    })

    require(summary["result"] == "PASS_300_OF_300", "TOKENIZER_RESULT")
    require(
        int(summary["source_pair_count"]) == PAIR_COUNT,
        "TOKENIZER_PAIR_COUNT",
    )
    require(
        int(summary["anchor_row_count"]) == ROW_COUNT,
        "TOKENIZER_ANCHOR_COUNT",
    )
    require(
        int(summary["complete_source_pair_count"]) == PAIR_COUNT,
        "TOKENIZER_COMPLETE_PAIR_COUNT",
    )
    require(summary["model_forward_count"] == 0, "TOKENIZER_FORWARD")
    require(summary["checkpoint_load_count"] == 0, "TOKENIZER_CHECKPOINT")
    require(summary["gpu_used"] is False, "TOKENIZER_GPU")

    return anchor_rows, summary


def write_outputs(
    provenance,
    old_identity,
    population,
    geometry,
    anchor_rows,
    eligibility,
):
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
            "GEN4_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_"
            "LOCALIZATION_XG1_STRUCTURAL_V1",
        "result":
            "PASS_XG1_1801_2100_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_"
            "LOCALIZATION_STRUCTURAL",
        "design_commit": DESIGN_COMMIT,
        "generator_family": BASE.GENERATOR_FAMILY,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "source_pair_count": PAIR_COUNT,
        "row_count": ROW_COUNT,
        "rows_per_pair": 6,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(rows_raw),
        "prior_population_identity": dict(old_identity),
        "pair_id_overlap_with_001_1800": 0,
        "claim_overlap_with_001_1800": 0,
        "evidence_overlap_with_001_1800": 0,
        "claim_evidence_row_overlap_with_001_1800": 0,
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
        BASE.SOURCE_FILE: source_raw,
        BASE.ROW_FILE: rows_raw,
        "structural_manifest.json": pretty(structural),
    }
    data_hashes: dict[str, str] = {}
    for name, raw in data_payloads.items():
        (data_dir / name).write_bytes(raw)
        data_hashes[name] = sha256_bytes(raw)

    (data_dir / "SHA256SUMS.txt").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(data_hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )

    geometry_raw = pretty(geometry)
    anchor_raw = b"".join(canonical(row) for row in anchor_rows)
    eligibility_raw = pretty(eligibility)

    preparation = {
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_"
            "LOCALIZATION_STATIC_PREPARATION_V1",
        "result":
            "PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_"
            "LOCALIZATION_STATIC_PREPARATION",
        "design": dict(provenance),
        "population": {
            "data_dir": OUTPUT_DATA_DIR.as_posix(),
            "pair_id_first": PAIR_FIRST,
            "pair_id_last": PAIR_LAST,
            "source_pair_count": PAIR_COUNT,
            "row_count": ROW_COUNT,
            "source_sha256": data_hashes[BASE.SOURCE_FILE],
            "rows_sha256": data_hashes[BASE.ROW_FILE],
            "structural_manifest_sha256":
                data_hashes["structural_manifest.json"],
            "pair_id_overlap_with_001_1800": 0,
            "claim_overlap_with_001_1800": 0,
            "evidence_overlap_with_001_1800": 0,
            "claim_evidence_row_overlap_with_001_1800": 0,
        },
        "geometry": {
            "result": geometry["result"],
            "residual_plane_order": geometry["residual_plane_order"],
            "excluded_plane": "P3",
            "frozen_residual_vector_sha256":
                geometry["frozen_residual_vector_sha256"],
            "all_residual_vectors_reproduced_exactly": True,
            "full_principal_10_vector_gram_max_abs_residual":
                geometry[
                    "full_principal_10_vector_gram_max_abs_residual"
                ],
            "residual_to_pp3_max_abs_dot":
                geometry["residual_to_pp3_max_abs_dot"],
            "per_plane_quarter_turn_control":
                geometry["per_plane_quarter_turn_control"],
        },
        "tokenizer_eligibility": {
            "result": eligibility["result"],
            "source_pair_count": eligibility["source_pair_count"],
            "anchor_row_count": eligibility["anchor_row_count"],
            "complete_source_pair_count":
                eligibility["complete_source_pair_count"],
            "anchor_manifest_sha256":
                eligibility["anchor_manifest_sha256"],
        },
        "future_execution_contract": {
            "condition_count": 9,
            "condition_order": [
                "native",
                "p1_neutralized",
                "p1_quarter_turn_control",
                "p2_neutralized",
                "p2_quarter_turn_control",
                "p4_neutralized",
                "p4_quarter_turn_control",
                "p5_neutralized",
                "p5_quarter_turn_control",
            ],
            "scientific_model_forward_budget": FUTURE_FORWARDS,
            "baseline_model_forward_budget": 0,
            "pair_count": PAIR_COUNT,
            "forwards_per_condition_per_pair": 40,
            "forwards_per_pair": 360,
            "raw_confirmatory_p_value_count": RAW_P_VALUE_COUNT,
            "multiplicity_method": MULTIPLICITY,
            "familywise_alpha": FWER_ALPHA,
            "execution_authorized": False,
        },
        "scientific_model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_outcomes_observed": False,
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
    }

    report_payloads = {
        "geometry_manifest.json": geometry_raw,
        "tokenizer_anchor_manifest.jsonl": anchor_raw,
        "tokenizer_eligibility_summary.json": eligibility_raw,
        "preparation_manifest.json": pretty(preparation),
    }

    report_hashes: dict[str, str] = {}
    for name, raw in report_payloads.items():
        (report_dir / name).write_bytes(raw)
        report_hashes[name] = sha256_bytes(raw)

    (report_dir / "SHA256SUMS.txt").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(report_hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )

    source_script = Path(__file__).resolve()
    shutil.copyfile(source_script, repo_script)
    script_sha = sha256_file(repo_script)

    return {
        "data_hashes": data_hashes,
        "report_hashes": report_hashes,
        "script_sha256": script_sha,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only static preparation for PP3-excluded residual "
            "individual-plane necessity localization on fresh XG1 1801..2100."
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
        not tokenizer_snapshot.is_relative_to(ROOT),
        "TOKENIZER_SNAPSHOT_MUST_BE_EXTERNAL_TO_REPO",
    )

    provenance = authenticate()
    old_rows, old_identity = validate_old_populations()
    facts, rows, population = build_new_population(old_rows)
    geometry = validate_geometry()
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

    print(
        "RESULT=PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_"
        "NECESSITY_LOCALIZATION_STATIC_PREPARATION"
    )
    print(f"PAIR_ID_FIRST={PAIR_FIRST}")
    print(f"PAIR_ID_LAST={PAIR_LAST}")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("OLD_XG1_001_1800_BYTE_REGENERATION_IDENTITY=True")
    print("PAIR_ID_OVERLAP_WITH_001_1800=0")
    print("CLAIM_OVERLAP_WITH_001_1800=0")
    print("EVIDENCE_OVERLAP_WITH_001_1800=0")
    print("CLAIM_EVIDENCE_ROW_OVERLAP_WITH_001_1800=0")
    print("ALL_RESIDUAL_VECTORS_REPRODUCED_EXACTLY=True")
    print(
        "FULL_PRINCIPAL_10_VECTOR_GRAM_MAX_ABS_RESIDUAL="
        f"{geometry['full_principal_10_vector_gram_max_abs_residual']:.17g}"
    )
    print(
        "RESIDUAL_TO_PP3_MAX_ABS_DOT="
        f"{geometry['residual_to_pp3_max_abs_dot']:.17g}"
    )
    print(
        "PER_PLANE_QUARTER_TURN_GLOBAL_MAX_ABS_L2_MISMATCH="
        f"{geometry['global_max_abs_l2_mismatch']:.17g}"
    )
    print(
        "PER_PLANE_QUARTER_TURN_GLOBAL_MAX_ABS_DOT="
        f"{geometry['global_max_abs_treatment_control_dot']:.17g}"
    )
    print(
        "PER_PLANE_GLOBAL_MAX_OTHER_RESIDUAL_PROJECTION="
        f"{geometry['global_max_abs_projection_onto_other_residual_planes']:.17g}"
    )
    print(
        "PER_PLANE_GLOBAL_MAX_PP3_PROJECTION="
        f"{geometry['global_max_abs_projection_onto_pp3']:.17g}"
    )
    print(f"TOKENIZER_ELIGIBILITY={eligibility['result']}")
    print(f"ANCHOR_ROW_COUNT={eligibility['anchor_row_count']}")
    print(
        "ANCHOR_MANIFEST_SHA256="
        f"{eligibility['anchor_manifest_sha256']}"
    )
    print(
        "SOURCE_SHA256="
        f"{outputs['data_hashes'][BASE.SOURCE_FILE]}"
    )
    print(
        "ROWS_SHA256="
        f"{outputs['data_hashes'][BASE.ROW_FILE]}"
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
        "TOKENIZER_ELIGIBILITY_SUMMARY_SHA256="
        f"{outputs['report_hashes']['tokenizer_eligibility_summary.json']}"
    )
    print(
        "PREPARATION_MANIFEST_SHA256="
        f"{outputs['report_hashes']['preparation_manifest.json']}"
    )
    print(f"REPO_SCRIPT_SHA256={outputs['script_sha256']}")
    print("FUTURE_CONDITION_COUNT=9")
    print("FUTURE_SCIENTIFIC_MODEL_FORWARD_BUDGET=108000")
    print("FUTURE_RAW_CONFIRMATORY_P_VALUE_COUNT=4")
    print("FUTURE_MULTIPLICITY=HOLM_FWER_0.05")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("CHECKPOINT_LOAD_COUNT=0")
    print("GPU_USED=False")
    print("SCIENTIFIC_OUTCOMES_OBSERVED=False")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("MULTIPLICITY_CORRECTION_EXECUTED=False")
    print("EXECUTION_AUTHORIZED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
