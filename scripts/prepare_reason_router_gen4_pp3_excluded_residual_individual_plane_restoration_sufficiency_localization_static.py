from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(os.environ["CONTRAMAMBA_REPO_ROOT"]).resolve()

from scripts import (
    prepare_reason_router_gen4_pp3_excluded_residual_individual_plane_necessity_localization_static
    as prior
)
from scripts import (
    reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda
    as holdout
)

DESIGN_COMMIT = "8c0ff6dbad77ed876fc1481b3b53c3fd47a27d3b"
DESIGN_BLOB = "6c1cd31ad852505eefc095011d184699c53a5e4b"
DESIGN_PATH = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_individual_plane_"
    "restoration_sufficiency_localization_design.md"
)

DATA_DIR = Path(
    "data/"
    "reason_router_gen4_xg1_residual_individual_plane_"
    "restoration_sufficiency_v1"
)

REPORT_DIR = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_individual_plane_"
    "restoration_sufficiency_localization_static_preparation_8c0ff6d"
)

SCRIPT_PATH = Path(
    "scripts/"
    "prepare_reason_router_gen4_pp3_excluded_residual_individual_plane_"
    "restoration_sufficiency_localization_static.py"
)

PAIR_START = 2101
PAIR_END = 2400
PAIR_FIRST = "xg1_fact_2101"
PAIR_LAST = "xg1_fact_2400"
PAIR_COUNT = 300
ROW_COUNT = 1800

TOL = 1.0e-12

PLAN_SHA = {
    "xg2": "b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c",
    "xg4": "792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f",
}

# --------------------------------------------------
# Rebind the previously validated static-prep machinery
# --------------------------------------------------

prior.EXPECTED_HEAD = DESIGN_COMMIT
prior.DESIGN_COMMIT = DESIGN_COMMIT
prior.DESIGN_PATH = DESIGN_PATH
prior.DESIGN_BLOB = DESIGN_BLOB

prior.PAIR_START = PAIR_START
prior.PAIR_END = PAIR_END
prior.PAIR_FIRST = PAIR_FIRST
prior.PAIR_LAST = PAIR_LAST
prior.PAIR_COUNT = PAIR_COUNT
prior.ROW_COUNT = ROW_COUNT

prior.OLD_PAIR_COUNT = 2100
prior.OLD_ROW_COUNT = 12600

prior.OUTPUT_DATA_DIR = DATA_DIR
prior.OUTPUT_REPORT_DIR = REPORT_DIR
prior.REPO_SCRIPT_PATH = SCRIPT_PATH

prior.OLD_COHORTS = prior.OLD_COHORTS + (
    (
        1801,
        2100,
        Path(
            "data/"
            "reason_router_gen4_xg1_residual_individual_plane_necessity_v1"
        ),
        "18381b4d31bf6b5b5975edf29dd23ad9a4cfbece23cf8c119cda4fb2539fcc9f",
        "f3b67944c05e0198f9f33a3b544ddee344a3741ceccb00fe2a74191f177b18b5",
    ),
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


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


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def endpoint_basis_audit() -> dict[str, Any]:
    # Validate only the frozen Phase1 alignment-plan artifacts required to
    # reconstruct the endpoint bases. Do not pull in the unrelated historical
    # baseline_items.jsonl validation chain.
    import torch

    family_basis = holdout.prior
    out: dict[str, Any] = {}

    for family in ("xg2", "xg4"):
        plan_path = family_basis._phase1_plan_path(family)

        require(
            plan_path.is_file(),
            f"PHASE1_PLAN_FILE_MISSING:{family}:{plan_path}",
        )

        observed = sha256_file(plan_path)
        require(
            observed == PLAN_SHA[family],
            f"BASIS_PLAN_SHA:{family}:{observed}",
        )

        plans = torch.load(
            plan_path,
            map_location="cpu",
            weights_only=True,
        )

        require(
            torch.is_tensor(plans),
            f"PLAN_NOT_TENSOR:{family}",
        )
        require(
            plans.ndim == 2 and int(plans.shape[0]) == 300,
            f"PLAN_SHAPE:{family}:{tuple(plans.shape)}",
        )
        require(
            bool(torch.isfinite(plans).all().item()),
            f"PLAN_NONFINITE:{family}",
        )

        basis_info = family_basis.reconstruct_family_basis(
            family,
            plans,
        )
        basis = basis_info["basis"]

        shape = tuple(int(x) for x in basis.shape)
        require(
            shape == (395, 5),
            f"BASIS_SHAPE:{family}:{shape}",
        )

        out[family] = {
            "plan_path": str(plan_path.relative_to(ROOT)).replace("\\", "/"),
            "plan_sha256": observed,
            "plan_shape": list(plans.shape),
            "basis_shape": list(shape),
            "minimum_selected_eigengap":
                float(basis_info["minimum_selected_eigengap"]),
            "orthonormality_max_abs_residual":
                float(basis_info["orthonormality_max_abs_residual"]),
            "sign_canonicalization":
                basis_info["sign_canonicalization"],
            "identity_verified": True,
        }

    return {
        "result": "PASS_FROZEN_XG2_XG4_ENDPOINT_BASIS_IDENTITY",
        "validation_scope":
            "exact frozen alignment_delta_h.pt SHA256 + deterministic basis reconstruction",
        "direction_order": [
            "xg2_0", "xg2_1", "xg2_2", "xg2_3", "xg2_4",
            "xg4_0", "xg4_1", "xg4_2", "xg4_3", "xg4_4",
        ],
        "epsilon": 0.025,
        "families": out,
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
    }


def restoration_geometry_audit() -> dict[str, Any]:
    base_geometry = prior.validate_geometry()
    reconstructed = prior.aggregate.reconstruct_geometry()
    raw_vectors = reconstructed["vectors"]

    vectors: dict[int, dict[str, np.ndarray]] = {}
    for plane in (1, 2, 3, 4, 5):
        vectors[plane] = {}
        for sign in ("plus", "minus"):
            vectors[plane][sign] = np.asarray(
                raw_vectors[plane][sign].detach().cpu().numpy(),
                dtype=np.float64,
            )

    probes = np.asarray(
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

    per_plane: dict[str, Any] = {}

    global_coeff = 0.0
    global_neutral = 0.0
    global_restore = 0.0
    global_replace = 0.0
    global_norm = 0.0
    global_replace_coord = 0.0
    global_pp3 = 0.0
    global_other = 0.0

    for target in (1, 2, 4, 5):
        tp = vectors[target]["plus"]
        tm = vectors[target]["minus"]

        metrics = {
            "coefficient_recovery_max_abs_residual": 0.0,
            "neutralized_target_projection_max_abs": 0.0,
            "restoration_equals_native_max_abs_residual": 0.0,
            "replacement_increment_max_abs_residual": 0.0,
            "restoration_addition_norm_max_abs_mismatch": 0.0,
            "replacement_coordinate_max_abs_residual": 0.0,
            "pp3_coordinate_drift_max_abs": 0.0,
            "other_residual_plane_coordinate_drift_max_abs": 0.0,
        }

        for i, (a, b) in enumerate(probes):
            h = np.zeros_like(tp)

            for plane in (1, 2, 3, 4, 5):
                if plane == target:
                    ca, cb = float(a), float(b)
                else:
                    ca = 0.071 * (i + 1) * (plane + 1)
                    cb = -0.053 * (i + 1) * (plane + 2)

                h = (
                    h
                    + ca * vectors[plane]["plus"]
                    + cb * vectors[plane]["minus"]
                )

            recovered = np.asarray(
                [float(h @ tp), float(h @ tm)],
                dtype=np.float64,
            )
            expected = np.asarray([a, b], dtype=np.float64)

            c = recovered[0] * tp + recovered[1] * tm
            r = -recovered[1] * tp + recovered[0] * tm

            background = h - c
            restored = background + c
            replacement = background + r

            metrics["coefficient_recovery_max_abs_residual"] = max(
                metrics["coefficient_recovery_max_abs_residual"],
                float(np.max(np.abs(recovered - expected))),
            )

            neutral_coords = np.asarray(
                [float(background @ tp), float(background @ tm)]
            )
            metrics["neutralized_target_projection_max_abs"] = max(
                metrics["neutralized_target_projection_max_abs"],
                float(np.max(np.abs(neutral_coords))),
            )

            metrics["restoration_equals_native_max_abs_residual"] = max(
                metrics["restoration_equals_native_max_abs_residual"],
                float(np.max(np.abs(restored - h))),
            )

            metrics["replacement_increment_max_abs_residual"] = max(
                metrics["replacement_increment_max_abs_residual"],
                float(np.max(np.abs((replacement - background) - r))),
            )

            metrics["restoration_addition_norm_max_abs_mismatch"] = max(
                metrics["restoration_addition_norm_max_abs_mismatch"],
                abs(
                    float(np.linalg.norm(restored - background))
                    - float(np.linalg.norm(replacement - background))
                ),
            )

            replacement_coords = np.asarray(
                [float(replacement @ tp), float(replacement @ tm)]
            )
            expected_replace = np.asarray(
                [-recovered[1], recovered[0]]
            )
            metrics["replacement_coordinate_max_abs_residual"] = max(
                metrics["replacement_coordinate_max_abs_residual"],
                float(
                    np.max(
                        np.abs(replacement_coords - expected_replace)
                    )
                ),
            )

            for sign in ("plus", "minus"):
                p3 = vectors[3][sign]
                native_coord = float(h @ p3)

                metrics["pp3_coordinate_drift_max_abs"] = max(
                    metrics["pp3_coordinate_drift_max_abs"],
                    abs(float(background @ p3) - native_coord),
                    abs(float(replacement @ p3) - native_coord),
                    abs(float(restored @ p3) - native_coord),
                )

            for other in (1, 2, 4, 5):
                if other == target:
                    continue
                for sign in ("plus", "minus"):
                    v = vectors[other][sign]
                    native_coord = float(h @ v)

                    metrics[
                        "other_residual_plane_coordinate_drift_max_abs"
                    ] = max(
                        metrics[
                            "other_residual_plane_coordinate_drift_max_abs"
                        ],
                        abs(float(background @ v) - native_coord),
                        abs(float(replacement @ v) - native_coord),
                        abs(float(restored @ v) - native_coord),
                    )

        for name, value in metrics.items():
            require(
                float(value) <= TOL,
                f"P{target}:{name}:{value}",
            )

        metrics.update(
            {
                "probe_count": int(probes.shape[0]),
                "native_component": "c_k=a_k*p_k+ + b_k*p_k-",
                "quarter_turn_component": "r_k=-b_k*p_k+ + a_k*p_k-",
                "neutralized_background": "B_k=h-c_k",
                "exact_restoration": "R_k=B_k+c_k=h",
                "matched_replacement": "C_k=B_k+r_k",
                "response_guided_selection": False,
                "response_guided_weighting": False,
            }
        )

        per_plane[f"P{target}"] = metrics

        global_coeff = max(
            global_coeff,
            metrics["coefficient_recovery_max_abs_residual"],
        )
        global_neutral = max(
            global_neutral,
            metrics["neutralized_target_projection_max_abs"],
        )
        global_restore = max(
            global_restore,
            metrics["restoration_equals_native_max_abs_residual"],
        )
        global_replace = max(
            global_replace,
            metrics["replacement_increment_max_abs_residual"],
        )
        global_norm = max(
            global_norm,
            metrics["restoration_addition_norm_max_abs_mismatch"],
        )
        global_replace_coord = max(
            global_replace_coord,
            metrics["replacement_coordinate_max_abs_residual"],
        )
        global_pp3 = max(
            global_pp3,
            metrics["pp3_coordinate_drift_max_abs"],
        )
        global_other = max(
            global_other,
            metrics[
                "other_residual_plane_coordinate_drift_max_abs"
            ],
        )

    controls = base_geometry.pop("per_plane_quarter_turn_control")

    base_geometry.update(
        {
            "schema_version":
                "GEN4_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_"
                "RESTORATION_SUFFICIENCY_LOCALIZATION_GEOMETRY_V1",
            "result":
                "PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_"
                "RESTORATION_SUFFICIENCY_LOCALIZATION_GEOMETRY_STATIC",
            "design_commit": DESIGN_COMMIT,
            "per_plane_quarter_turn_replacement_component": controls,
            "per_plane_restoration_audit": per_plane,
            "global_coefficient_recovery_max_abs_residual": global_coeff,
            "global_neutralized_target_projection_max_abs": global_neutral,
            "global_restoration_equals_native_max_abs_residual":
                global_restore,
            "global_replacement_increment_max_abs_residual":
                global_replace,
            "global_restoration_addition_norm_max_abs_mismatch":
                global_norm,
            "global_replacement_coordinate_max_abs_residual":
                global_replace_coord,
            "global_pp3_coordinate_drift_max_abs": global_pp3,
            "global_other_residual_plane_coordinate_drift_max_abs":
                global_other,
            "scientific_model_forward_count": 0,
            "checkpoint_load_count": 0,
            "gpu_used": False,
            "scientific_outcomes_observed": False,
            "primary_inference_executed": False,
        }
    )

    return base_geometry


def tokenizer_eligibility(
    snapshot: Path,
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
):
    anchors, summary = prior.tokenizer_eligibility(
        snapshot,
        facts,
        rows,
    )

    for row in anchors:
        row["schema_version"] = (
            "GEN4_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_"
            "RESTORATION_SUFFICIENCY_LOCALIZATION_XG1_TOKENIZER_ANCHOR_V1"
        )

    anchor_raw = b"".join(canonical(x) for x in anchors)

    summary = dict(summary)
    summary.update(
        {
            "schema_version":
                "GEN4_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_"
                "RESTORATION_SUFFICIENCY_LOCALIZATION_"
                "XG1_TOKENIZER_ELIGIBILITY_SUMMARY_V1",
            "authority_commit": DESIGN_COMMIT,
            "pair_id_first": PAIR_FIRST,
            "pair_id_last": PAIR_LAST,
            "anchor_manifest_sha256": sha256_bytes(anchor_raw),
        }
    )

    require(summary["result"] == "PASS_300_OF_300", "TOKENIZER_RESULT")
    require(int(summary["source_pair_count"]) == 300, "TOKENIZER_COUNT")
    require(int(summary["anchor_row_count"]) == 1800, "ANCHOR_COUNT")
    require(summary["model_forward_count"] == 0, "TOKENIZER_FORWARD")
    require(summary["checkpoint_load_count"] == 0, "TOKENIZER_CHECKPOINT")
    require(summary["gpu_used"] is False, "TOKENIZER_GPU")

    return anchors, summary


def write_outputs(
    provenance: Mapping[str, Any],
    old_identity: Mapping[str, Any],
    population: Mapping[str, Any],
    geometry: Mapping[str, Any],
    basis_identity: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:

    data_dir = ROOT / DATA_DIR
    report_dir = ROOT / REPORT_DIR
    script_path = ROOT / SCRIPT_PATH

    require(not data_dir.exists(), f"DATA_COLLISION:{data_dir}")
    require(not report_dir.exists(), f"REPORT_COLLISION:{report_dir}")
    require(not script_path.exists(), f"SCRIPT_COLLISION:{script_path}")

    data_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)

    source_raw = population["source_raw"]
    rows_raw = population["rows_raw"]

    structural = {
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_"
            "RESTORATION_SUFFICIENCY_LOCALIZATION_XG1_STRUCTURAL_V1",
        "result":
            "PASS_XG1_2101_2400_RESIDUAL_INDIVIDUAL_PLANE_"
            "RESTORATION_SUFFICIENCY_LOCALIZATION_STRUCTURAL",
        "design_commit": DESIGN_COMMIT,
        "generator_family": prior.BASE.GENERATOR_FAMILY,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "source_pair_count": 300,
        "row_count": 1800,
        "rows_per_pair": 6,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(rows_raw),
        "prior_population_identity": dict(old_identity),
        "pair_id_overlap_with_001_2100": 0,
        "claim_overlap_with_001_2100": 0,
        "evidence_overlap_with_001_2100": 0,
        "claim_evidence_row_overlap_with_001_2100": 0,
        "deterministic_generator_semantics": True,
        "labels_present": False,
        "endpoint_values_present": False,
        "response_fields_present": False,
        "model_executed": False,
        "checkpoint_loaded": False,
        "cuda_executed": False,
        "tokenizer_executed": False,
    }

    data_payloads = {
        prior.BASE.SOURCE_FILE: source_raw,
        prior.BASE.ROW_FILE: rows_raw,
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
    anchor_raw = b"".join(canonical(x) for x in anchors)
    eligibility_raw = pretty(eligibility)

    preparation = {
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_"
            "RESTORATION_SUFFICIENCY_LOCALIZATION_STATIC_PREPARATION_V1",
        "result":
            "PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_"
            "RESTORATION_SUFFICIENCY_LOCALIZATION_STATIC_PREPARATION",
        "design": dict(provenance),
        "population": {
            "data_dir": DATA_DIR.as_posix(),
            "pair_id_first": PAIR_FIRST,
            "pair_id_last": PAIR_LAST,
            "source_pair_count": 300,
            "row_count": 1800,
            "source_sha256": data_hashes[prior.BASE.SOURCE_FILE],
            "rows_sha256": data_hashes[prior.BASE.ROW_FILE],
            "structural_manifest_sha256":
                data_hashes["structural_manifest.json"],
            "pair_id_overlap_with_001_2100": 0,
            "claim_overlap_with_001_2100": 0,
            "evidence_overlap_with_001_2100": 0,
            "claim_evidence_row_overlap_with_001_2100": 0,
        },
        "endpoint_basis_identity": dict(basis_identity),
        "geometry": {
            "result": geometry["result"],
            "residual_plane_order": geometry["residual_plane_order"],
            "frozen_residual_vector_sha256":
                geometry["frozen_residual_vector_sha256"],
            "all_residual_vectors_reproduced_exactly": True,
            "full_principal_10_vector_gram_max_abs_residual":
                geometry[
                    "full_principal_10_vector_gram_max_abs_residual"
                ],
            "residual_to_pp3_max_abs_dot":
                geometry["residual_to_pp3_max_abs_dot"],
            "per_plane_restoration_audit":
                geometry["per_plane_restoration_audit"],
        },
        "tokenizer_eligibility": {
            "result": eligibility["result"],
            "source_pair_count": eligibility["source_pair_count"],
            "anchor_row_count": eligibility["anchor_row_count"],
            "complete_source_pair_count":
                eligibility["complete_source_pair_count"],
            "anchor_manifest_sha256":
                eligibility["anchor_manifest_sha256"],
            "tokenizer_revision":
                eligibility["tokenizer_revision"],
        },
        "future_execution_contract": {
            "condition_count": 9,
            "condition_order": [
                "native",
                "p1_neutralized",
                "p1_quarter_turn_replacement",
                "p2_neutralized",
                "p2_quarter_turn_replacement",
                "p4_neutralized",
                "p4_quarter_turn_replacement",
                "p5_neutralized",
                "p5_quarter_turn_replacement",
            ],
            "scientific_model_forward_budget": 108000,
            "baseline_model_forward_budget": 0,
            "pair_count": 300,
            "forwards_per_condition_per_pair": 40,
            "forwards_per_pair": 360,
            "raw_confirmatory_p_value_count": 4,
            "multiplicity_method":
                "Holm step-down across exactly P1,P2,P4,P5",
            "familywise_alpha": 0.05,
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

    shutil.copyfile(Path(__file__).resolve(), script_path)

    return {
        "data_hashes": data_hashes,
        "report_hashes": report_hashes,
        "script_sha256": sha256_file(script_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    args = parser.parse_args(argv)

    snapshot = args.tokenizer_snapshot.resolve()

    require(snapshot.is_dir(), f"TOKENIZER_SNAPSHOT_MISSING:{snapshot}")
    require(
        not snapshot.is_relative_to(ROOT),
        "TOKENIZER_SNAPSHOT_MUST_BE_EXTERNAL_TO_REPO",
    )

    provenance = prior.authenticate()

    old_rows, old_identity = prior.validate_old_populations()
    old_identity = dict(old_identity)
    old_identity.update(
        {
            "old_pair_range": "xg1_fact_001..xg1_fact_2100",
            "old_source_pair_count": 2100,
            "old_row_count": 12600,
            "all_old_cohorts_byte_regeneration_identity": True,
        }
    )

    facts, rows, population = prior.build_new_population(old_rows)

    basis_identity = endpoint_basis_audit()
    geometry = restoration_geometry_audit()

    anchors, eligibility = tokenizer_eligibility(
        snapshot,
        facts,
        rows,
    )

    outputs = write_outputs(
        provenance,
        old_identity,
        population,
        geometry,
        basis_identity,
        anchors,
        eligibility,
    )

    print(
        "RESULT="
        "PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_"
        "RESTORATION_SUFFICIENCY_LOCALIZATION_STATIC_PREPARATION"
    )
    print("PAIR_ID_FIRST=xg1_fact_2101")
    print("PAIR_ID_LAST=xg1_fact_2400")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("OLD_XG1_001_2100_BYTE_REGENERATION_IDENTITY=True")
    print("PAIR_ID_OVERLAP_WITH_001_2100=0")
    print("CLAIM_OVERLAP_WITH_001_2100=0")
    print("EVIDENCE_OVERLAP_WITH_001_2100=0")
    print("CLAIM_EVIDENCE_ROW_OVERLAP_WITH_001_2100=0")
    print("TOKENIZER_ELIGIBILITY=PASS_300_OF_300")
    print("ENDPOINT_BASIS_IDENTITY=PASS")
    print("ALL_RESIDUAL_VECTORS_REPRODUCED_EXACTLY=True")
    print(
        "RESTORATION_EQUALS_NATIVE_MAX_ABS_RESIDUAL="
        f"{geometry['global_restoration_equals_native_max_abs_residual']:.17g}"
    )
    print(
        "RESTORATION_ADDITION_NORM_MAX_ABS_MISMATCH="
        f"{geometry['global_restoration_addition_norm_max_abs_mismatch']:.17g}"
    )
    print(
        "PP3_COORDINATE_DRIFT_MAX_ABS="
        f"{geometry['global_pp3_coordinate_drift_max_abs']:.17g}"
    )
    print(
        "OTHER_RESIDUAL_PLANE_DRIFT_MAX_ABS="
        f"{geometry['global_other_residual_plane_coordinate_drift_max_abs']:.17g}"
    )
    print(
        "SOURCE_SHA256="
        f"{outputs['data_hashes'][prior.BASE.SOURCE_FILE]}"
    )
    print(
        "ROWS_SHA256="
        f"{outputs['data_hashes'][prior.BASE.ROW_FILE]}"
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
