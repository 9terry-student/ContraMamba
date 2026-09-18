from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(os.environ["CONTRAMAMBA_REPO_ROOT"]).resolve()

from scripts import (
    prepare_reason_router_gen4_pp3_excluded_residual_individual_plane_restoration_sufficiency_localization_static
    as prev
)

prior = prev.prior

DESIGN_COMMIT = "14d71b742488dd088ca22c416962522235d8d67b"
DESIGN_BLOB = "1c82c90b6286e9923a2df51c5679ae080b8a85a9"
DESIGN_PATH = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_aggregate_"
    "restoration_sufficiency_design.md"
)

DATA_DIR = Path(
    "data/"
    "reason_router_gen4_xg1_residual_aggregate_"
    "restoration_sufficiency_v1"
)

REPORT_DIR = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_aggregate_"
    "restoration_sufficiency_static_preparation_14d71b7"
)

SCRIPT_PATH = Path(
    "scripts/"
    "prepare_reason_router_gen4_pp3_excluded_residual_aggregate_"
    "restoration_sufficiency_static.py"
)

PAIR_START = 2401
PAIR_END = 2700
PAIR_FIRST = "xg1_fact_2401"
PAIR_LAST = "xg1_fact_2700"
PAIR_COUNT = 300
ROW_COUNT = 1800
OLD_PAIR_COUNT = 2400
OLD_ROW_COUNT = 14400

TOL = 1.0e-12

PLAN_SHA = {
    "xg2": "b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c",
    "xg4": "792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f",
}

# The imported predecessor has already extended `prior.OLD_COHORTS`
# through XG1 1801..2100. Append the validated 2101..2400 cohort exactly once.
require_last_end = prior.OLD_COHORTS[-1][1]
if require_last_end != 2100:
    raise RuntimeError(f"UNEXPECTED_PREDECESSOR_OLD_COHORT_END:{require_last_end}")

prior.OLD_COHORTS = prior.OLD_COHORTS + (
    (
        2101,
        2400,
        Path(
            "data/"
            "reason_router_gen4_xg1_residual_individual_plane_"
            "restoration_sufficiency_v1"
        ),
        "d2cc7276254bb96c32ea54af3c3fb768cbe005dcd7cb329512232df18629ea81",
        "56957588464df337ad65509e377b534e49c264ddaac766e3131633dd5e2ac087",
    ),
)

# Rebind the validated population/tokenizer/static-authentication machinery.
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
prior.OLD_PAIR_COUNT = OLD_PAIR_COUNT
prior.OLD_ROW_COUNT = OLD_ROW_COUNT

prior.OUTPUT_DATA_DIR = DATA_DIR
prior.OUTPUT_REPORT_DIR = REPORT_DIR
prior.REPO_SCRIPT_PATH = SCRIPT_PATH


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def endpoint_basis_audit() -> dict[str, Any]:
    # Reuse the already validated narrow Phase1 basis identity audit.
    result = prev.endpoint_basis_audit()
    require(
        result["result"] == "PASS_FROZEN_XG2_XG4_ENDPOINT_BASIS_IDENTITY",
        "ENDPOINT_BASIS_RESULT",
    )
    for family in ("xg2", "xg4"):
        require(
            result["families"][family]["plan_sha256"] == PLAN_SHA[family],
            f"ENDPOINT_PLAN_SHA:{family}",
        )
    return result


def aggregate_restoration_geometry_audit() -> dict[str, Any]:
    """
    CPU-only synthetic algebra audit of the frozen simultaneous residual
    restoration geometry. No model, checkpoint, tokenizer, or GPU is used.
    """
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

    residual_planes = (1, 2, 4, 5)

    # Deterministic coefficient probes. Every probe activates all four residual
    # planes simultaneously, so this is an aggregate rather than per-plane audit.
    seed_pairs = np.asarray(
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

    metrics = {
        "coefficient_recovery_max_abs_residual": 0.0,
        "aggregate_neutralized_residual_coordinate_max_abs": 0.0,
        "restoration_equals_native_max_abs_residual": 0.0,
        "replacement_increment_max_abs_residual": 0.0,
        "restoration_replacement_addition_norm_max_abs_mismatch": 0.0,
        "aggregate_native_replacement_dot_max_abs": 0.0,
        "replacement_coordinate_max_abs_residual": 0.0,
        "pp3_coordinate_drift_max_abs": 0.0,
    }

    for i in range(seed_pairs.shape[0]):
        h = np.zeros_like(vectors[1]["plus"])
        expected_coeff: dict[int, np.ndarray] = {}

        # Preserve a non-zero PP3 coordinate so PP3 preservation is actually tested.
        pp3_a = 0.071 * (i + 1)
        pp3_b = -0.053 * (i + 2)
        h = (
            h
            + pp3_a * vectors[3]["plus"]
            + pp3_b * vectors[3]["minus"]
        )

        for position, plane in enumerate(residual_planes):
            base = seed_pairs[(i + position) % seed_pairs.shape[0]]
            scale = 1.0 + 0.17 * position
            a = float(scale * base[0])
            b = float(scale * base[1])
            expected_coeff[plane] = np.asarray([a, b], dtype=np.float64)
            h = (
                h
                + a * vectors[plane]["plus"]
                + b * vectors[plane]["minus"]
            )

        recovered: dict[int, np.ndarray] = {}
        c_parts: list[np.ndarray] = []
        r_parts: list[np.ndarray] = []

        for plane in residual_planes:
            plus = vectors[plane]["plus"]
            minus = vectors[plane]["minus"]
            coeff = np.asarray(
                [float(h @ plus), float(h @ minus)],
                dtype=np.float64,
            )
            recovered[plane] = coeff

            metrics["coefficient_recovery_max_abs_residual"] = max(
                metrics["coefficient_recovery_max_abs_residual"],
                float(np.max(np.abs(coeff - expected_coeff[plane]))),
            )

            c_parts.append(coeff[0] * plus + coeff[1] * minus)
            r_parts.append(-coeff[1] * plus + coeff[0] * minus)

        c_r = np.sum(np.stack(c_parts, axis=0), axis=0)
        r_r = np.sum(np.stack(r_parts, axis=0), axis=0)

        background = h - c_r
        restored = background + c_r
        replacement = background + r_r

        # All residual coordinates must be neutralized simultaneously.
        for plane in residual_planes:
            for sign in ("plus", "minus"):
                v = vectors[plane][sign]
                metrics[
                    "aggregate_neutralized_residual_coordinate_max_abs"
                ] = max(
                    metrics[
                        "aggregate_neutralized_residual_coordinate_max_abs"
                    ],
                    abs(float(background @ v)),
                )

        metrics["restoration_equals_native_max_abs_residual"] = max(
            metrics["restoration_equals_native_max_abs_residual"],
            float(np.max(np.abs(restored - h))),
        )

        metrics["replacement_increment_max_abs_residual"] = max(
            metrics["replacement_increment_max_abs_residual"],
            float(np.max(np.abs((replacement - background) - r_r))),
        )

        metrics[
            "restoration_replacement_addition_norm_max_abs_mismatch"
        ] = max(
            metrics[
                "restoration_replacement_addition_norm_max_abs_mismatch"
            ],
            abs(
                float(np.linalg.norm(restored - background))
                - float(np.linalg.norm(replacement - background))
            ),
        )

        metrics["aggregate_native_replacement_dot_max_abs"] = max(
            metrics["aggregate_native_replacement_dot_max_abs"],
            abs(float(c_r @ r_r)),
        )

        # The simultaneous replacement must realize (-b_k, a_k) in every plane.
        for plane in residual_planes:
            plus = vectors[plane]["plus"]
            minus = vectors[plane]["minus"]
            actual = np.asarray(
                [float(replacement @ plus), float(replacement @ minus)],
                dtype=np.float64,
            )
            coeff = recovered[plane]
            expected = np.asarray(
                [-coeff[1], coeff[0]],
                dtype=np.float64,
            )
            metrics["replacement_coordinate_max_abs_residual"] = max(
                metrics["replacement_coordinate_max_abs_residual"],
                float(np.max(np.abs(actual - expected))),
            )

        # PP3 must remain untouched by background/restoration/replacement.
        for sign in ("plus", "minus"):
            p3 = vectors[3][sign]
            native_coord = float(h @ p3)
            metrics["pp3_coordinate_drift_max_abs"] = max(
                metrics["pp3_coordinate_drift_max_abs"],
                abs(float(background @ p3) - native_coord),
                abs(float(restored @ p3) - native_coord),
                abs(float(replacement @ p3) - native_coord),
            )

    for name, value in metrics.items():
        require(float(value) <= TOL, f"AGGREGATE_GEOMETRY:{name}:{value}")

    return {
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_"
            "SUFFICIENCY_GEOMETRY_V1",
        "result":
            "PASS_PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_"
            "SUFFICIENCY_GEOMETRY_STATIC",
        "design_commit": DESIGN_COMMIT,
        "residual_plane_order": ["P1", "P2", "P4", "P5"],
        "excluded_plane": "P3",
        "all_residual_vectors_reproduced_exactly":
            bool(base_geometry["all_residual_vectors_reproduced_exactly"]),
        "frozen_residual_vector_sha256":
            dict(base_geometry["frozen_residual_vector_sha256"]),
        "full_principal_10_vector_gram_max_abs_residual":
            float(base_geometry[
                "full_principal_10_vector_gram_max_abs_residual"
            ]),
        "residual_to_pp3_max_abs_dot":
            float(base_geometry["residual_to_pp3_max_abs_dot"]),
        "aggregate_restoration_audit": {
            **metrics,
            "probe_count": int(seed_pairs.shape[0]),
            "native_component":
                "c_R=sum_{k in {1,2,4,5}}(a_k*p_k+ + b_k*p_k-)",
            "quarter_turn_component":
                "r_R=sum_{k in {1,2,4,5}}(-b_k*p_k+ + a_k*p_k-)",
            "neutralized_background": "B=h-c_R",
            "exact_restoration": "R_native=B+c_R=h",
            "matched_replacement": "C=B+r_R",
            "equal_addition_norm_identity": True,
            "aggregate_orthogonal_replacement_identity": True,
            "response_guided_plane_selection": False,
            "response_guided_weighting": False,
            "residual_template_weighting": False,
        },
        "scientific_model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_outcomes_observed": False,
        "primary_inference_executed": False,
    }


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
            "GEN4_PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_"
            "SUFFICIENCY_XG1_TOKENIZER_ANCHOR_V1"
        )

    anchor_raw = b"".join(canonical(x) for x in anchors)

    summary = dict(summary)
    summary.update(
        {
            "schema_version":
                "GEN4_PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_"
                "SUFFICIENCY_XG1_TOKENIZER_ELIGIBILITY_SUMMARY_V1",
            "authority_commit": DESIGN_COMMIT,
            "pair_id_first": PAIR_FIRST,
            "pair_id_last": PAIR_LAST,
            "anchor_manifest_sha256": sha256_bytes(anchor_raw),
        }
    )

    require(summary["result"] == "PASS_300_OF_300", "TOKENIZER_RESULT")
    require(int(summary["source_pair_count"]) == 300, "TOKENIZER_PAIR_COUNT")
    require(int(summary["anchor_row_count"]) == 1800, "TOKENIZER_ANCHOR_COUNT")
    require(summary["model_forward_count"] == 0, "TOKENIZER_FORWARD_COUNT")
    require(summary["checkpoint_load_count"] == 0, "TOKENIZER_CHECKPOINT_COUNT")
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
    repo_script = ROOT / SCRIPT_PATH

    require(not data_dir.exists(), f"DATA_COLLISION:{data_dir}")
    require(not report_dir.exists(), f"REPORT_COLLISION:{report_dir}")
    require(not repo_script.exists(), f"SCRIPT_COLLISION:{repo_script}")

    data_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)

    source_raw = population["source_raw"]
    rows_raw = population["rows_raw"]

    structural = {
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_"
            "SUFFICIENCY_XG1_STRUCTURAL_V1",
        "result":
            "PASS_XG1_2401_2700_RESIDUAL_AGGREGATE_"
            "RESTORATION_SUFFICIENCY_STRUCTURAL",
        "design_commit": DESIGN_COMMIT,
        "generator_family": prior.BASE.GENERATOR_FAMILY,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "source_pair_count": PAIR_COUNT,
        "row_count": ROW_COUNT,
        "rows_per_pair": 6,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(rows_raw),
        "prior_population_identity": dict(old_identity),
        "pair_id_overlap_with_001_2400": 0,
        "claim_overlap_with_001_2400": 0,
        "evidence_overlap_with_001_2400": 0,
        "claim_evidence_row_overlap_with_001_2400": 0,
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
            "GEN4_PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_"
            "SUFFICIENCY_STATIC_PREPARATION_V1",
        "result":
            "PASS_PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_"
            "SUFFICIENCY_STATIC_PREPARATION",
        "design": dict(provenance),
        "population": {
            "data_dir": DATA_DIR.as_posix(),
            "pair_id_first": PAIR_FIRST,
            "pair_id_last": PAIR_LAST,
            "source_pair_count": PAIR_COUNT,
            "row_count": ROW_COUNT,
            "source_sha256": data_hashes[prior.BASE.SOURCE_FILE],
            "rows_sha256": data_hashes[prior.BASE.ROW_FILE],
            "structural_manifest_sha256":
                data_hashes["structural_manifest.json"],
            "pair_id_overlap_with_001_2400": 0,
            "claim_overlap_with_001_2400": 0,
            "evidence_overlap_with_001_2400": 0,
            "claim_evidence_row_overlap_with_001_2400": 0,
        },
        "endpoint_basis_identity": dict(basis_identity),
        "geometry": {
            "result": geometry["result"],
            "residual_plane_order": geometry["residual_plane_order"],
            "frozen_residual_vector_sha256":
                geometry["frozen_residual_vector_sha256"],
            "all_residual_vectors_reproduced_exactly":
                geometry["all_residual_vectors_reproduced_exactly"],
            "full_principal_10_vector_gram_max_abs_residual":
                geometry[
                    "full_principal_10_vector_gram_max_abs_residual"
                ],
            "residual_to_pp3_max_abs_dot":
                geometry["residual_to_pp3_max_abs_dot"],
            "aggregate_restoration_audit":
                geometry["aggregate_restoration_audit"],
        },
        "tokenizer_eligibility": {
            "result": eligibility["result"],
            "source_pair_count": eligibility["source_pair_count"],
            "complete_source_pair_count":
                eligibility["complete_source_pair_count"],
            "anchor_row_count": eligibility["anchor_row_count"],
            "tokenizer_revision": eligibility["tokenizer_revision"],
            "anchor_manifest_sha256":
                eligibility["anchor_manifest_sha256"],
        },
        "future_execution_contract": {
            "condition_order": [
                "native",
                "residual_neutralized",
                "residual_quarter_turn_replacement",
            ],
            "condition_count": 3,
            "forwards_per_condition_per_pair": 40,
            "forwards_per_pair": 120,
            "pair_count": 300,
            "scientific_model_forward_budget": 36000,
            "baseline_model_forward_budget": 0,
            "primary_endpoint":
                "D_RES_SUF=(Q0-Q_B)-(Q_C-Q_B)=Q0-Q_C",
            "raw_confirmatory_p_value_count": 1,
            "multiplicity_method": "none",
            "alpha": 0.05,
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

    source_bytes = Path(__file__).read_bytes()
    repo_script.parent.mkdir(parents=True, exist_ok=True)
    repo_script.write_bytes(source_bytes)

    return {
        "data_hashes": data_hashes,
        "report_hashes": report_hashes,
        "script_sha256": sha256_bytes(source_bytes),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "CPU-only static preparation for prospective PP3-excluded "
            "aggregate residual restoration sufficiency on fresh XG1 2401..2700."
        )
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

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
            "old_pair_range": "xg1_fact_001..xg1_fact_2400",
            "old_source_pair_count": OLD_PAIR_COUNT,
            "old_row_count": OLD_ROW_COUNT,
            "all_old_cohorts_byte_regeneration_identity": True,
        }
    )

    facts, rows, population = prior.build_new_population(old_rows)

    basis_identity = endpoint_basis_audit()
    geometry = aggregate_restoration_geometry_audit()

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

    audit = geometry["aggregate_restoration_audit"]

    print(
        "RESULT="
        "PASS_PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_"
        "SUFFICIENCY_STATIC_PREPARATION"
    )
    print("PAIR_ID_FIRST=xg1_fact_2401")
    print("PAIR_ID_LAST=xg1_fact_2700")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("OLD_XG1_001_2400_BYTE_REGENERATION_IDENTITY=True")
    print("PAIR_ID_OVERLAP_WITH_001_2400=0")
    print("CLAIM_OVERLAP_WITH_001_2400=0")
    print("EVIDENCE_OVERLAP_WITH_001_2400=0")
    print("CLAIM_EVIDENCE_ROW_OVERLAP_WITH_001_2400=0")
    print("TOKENIZER_ELIGIBILITY=PASS_300_OF_300")
    print("ENDPOINT_BASIS_IDENTITY=PASS")
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
        "AGGREGATE_NEUTRALIZED_RESIDUAL_COORDINATE_MAX_ABS="
        f"{audit['aggregate_neutralized_residual_coordinate_max_abs']:.17g}"
    )
    print(
        "RESTORATION_EQUALS_NATIVE_MAX_ABS_RESIDUAL="
        f"{audit['restoration_equals_native_max_abs_residual']:.17g}"
    )
    print(
        "RESTORATION_REPLACEMENT_ADDITION_NORM_MAX_ABS_MISMATCH="
        f"{audit['restoration_replacement_addition_norm_max_abs_mismatch']:.17g}"
    )
    print(
        "AGGREGATE_NATIVE_REPLACEMENT_DOT_MAX_ABS="
        f"{audit['aggregate_native_replacement_dot_max_abs']:.17g}"
    )
    print(
        "REPLACEMENT_COORDINATE_MAX_ABS_RESIDUAL="
        f"{audit['replacement_coordinate_max_abs_residual']:.17g}"
    )
    print(
        "PP3_COORDINATE_DRIFT_MAX_ABS="
        f"{audit['pp3_coordinate_drift_max_abs']:.17g}"
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
        "STRUCTURAL_MANIFEST_SHA256="
        f"{outputs['data_hashes']['structural_manifest.json']}"
    )
    print(
        "GEOMETRY_MANIFEST_SHA256="
        f"{outputs['report_hashes']['geometry_manifest.json']}"
    )
    print(
        "TOKENIZER_ANCHOR_MANIFEST_SHA256="
        f"{outputs['report_hashes']['tokenizer_anchor_manifest.jsonl']}"
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
    print("FUTURE_CONDITION_COUNT=3")
    print("FUTURE_SCIENTIFIC_MODEL_FORWARD_BUDGET=36000")
    print("FUTURE_RAW_CONFIRMATORY_P_VALUE_COUNT=1")
    print("FUTURE_MULTIPLICITY=NONE")
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
