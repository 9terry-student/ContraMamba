#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
RAW_FREEZE_COMMIT = "552236c171f3467cd6eb12f02aa89a3eefee2f45"
DESIGN_FREEZE_COMMIT = "7254f89c352de5e6c6594e43d01a6a49ea5aeeb4"

DESIGN_PATH = Path("reports/reason_router_gen4_small_epsilon_robustness_design.md")
DESIGN_BLOB = "b0e67c1d0aae2df08ae22d44d411b455db1c3cbd"

REFERENCE_RUN_NAME = (
    "g4k-finite-epsilon-five-plane-decomposition-xg1-2401-2700-"
    "e27dbbd-retry1"
)
REFERENCE_ROOT = Path(
    "reports/reason_router_gen4_finite_epsilon_five_plane_decomposition_runs"
) / REFERENCE_RUN_NAME
REFERENCE_ITEMS = (
    REFERENCE_ROOT / "finite_epsilon_principal_decomposition_items.jsonl"
)
REFERENCE_SUMMARY = (
    REFERENCE_ROOT / "finite_epsilon_principal_decomposition_summary.json"
)
REFERENCE_MANIFEST = REFERENCE_ROOT / "artifact_manifest.json"
REFERENCE_SUMS = REFERENCE_ROOT / "SHA256SUMS.txt"

REFERENCE_ITEMS_BLOB = "edd46066805082734f624a54736df8a6b82544e5"
REFERENCE_SUMMARY_BLOB = "996cc03c15d7cc8a04109325e35d6b0512e00ea2"
REFERENCE_MANIFEST_BLOB = "4967aca0c5634d136867e500548c757489053b6b"
REFERENCE_SUMS_BLOB = "9b19f40ac9843ccc74a72118e3ce4683bb79bea9"

REFERENCE_ITEMS_SHA256 = (
    "8db506d872ff81e72b08d44f9ff0af907cb1a65086c0071e3e365a75bd166e17"
)
REFERENCE_SUMMARY_SHA256 = (
    "7f9e01dd28ffb2d2a286b464aa4701c639e68f265e11de9eef67ca1e3c1e448e"
)
REFERENCE_MANIFEST_SHA256 = (
    "bcf6facd7d7a506b2dcbd1da4ce6eca67749eaa8e0e014bfb0ffac0467fa1d10"
)

RAW_RUN_NAME = (
    "g4k-small-epsilon-xg1-2401-2700-0125-00625-a2ec66b-retry1"
)
RAW_ROOT = Path(
    "reports/reason_router_gen4_small_epsilon_robustness_runs"
) / RAW_RUN_NAME
RAW_ITEMS = RAW_ROOT / "small_epsilon_robustness_items.jsonl"
RAW_SUMMARY = RAW_ROOT / "small_epsilon_robustness_summary.json"
RAW_MANIFEST = RAW_ROOT / "artifact_manifest.json"
RAW_SUMS = RAW_ROOT / "SHA256SUMS.txt"

RAW_ITEMS_BLOB = "9d1ff10a66436eaed2a372e7e1f4e861486f98ea"
RAW_SUMMARY_BLOB = "d2abe9c5fe93a668cf3d146968aa8d15218f8aa2"
RAW_MANIFEST_BLOB = "d2e81a091efbbb20fdb03e4fcb0932e22009c60f"
RAW_SUMS_BLOB = "058c022d0c28eee931c0a5e24689b0a93638fb59"

RAW_ITEMS_SHA256 = (
    "6d06a83fa91075690ed8e2acbe166ce51e3a29c6a698fb83986c815cc4fe8c15"
)
RAW_SUMMARY_SHA256 = (
    "769909692549196625e08cfc0d6a8ab04d2a0ccc63266c8580cd19e8ec310a7c"
)
RAW_MANIFEST_SHA256 = (
    "d28e08f7d67cac4c65e2a4d572517f1197f41c817b80f15e4b750e3bf72ccddc"
)
RAW_SUMS_SHA256 = (
    "658966c5e1bcebcb1471f0f80ab39acb8c5db184f10a41e88d8c82fbcbf5dfb9"
)

RAW_EXECUTION_HEAD = "a2ec66b49e92039104c875702102c8e1364dc0e9"
CHECKPOINT_SHA256 = (
    "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
)

REFERENCE_EPSILON = 0.025
NEW_EPSILONS = (0.0125, 0.00625)
ALL_EPSILONS = (0.025, 0.0125, 0.00625)

N = 300
PLANE_ORDER = ("P1", "P2", "P3", "P4", "P5")
DIRECTION_ORDER = (
    "P1_plus", "P1_minus",
    "P2_plus", "P2_minus",
    "P3_plus", "P3_minus",
    "P4_plus", "P4_minus",
    "P5_plus", "P5_minus",
)

POSITIVE_RESULT = "SMALL_EPSILON_P3_SPECTRAL_DOMINANCE_PRESERVED"
NEGATIVE_RESULT = "SMALL_EPSILON_P3_SPECTRAL_DOMINANCE_NOT_PRESERVED"

ANALYSIS_SCHEMA = "gen4-small-epsilon-robustness-analysis-v1"
ANALYSIS_FILE = "small_epsilon_robustness_analysis.json"
REPORT_FILE = "small_epsilon_robustness_analysis.md"
SUMS_FILE = "SHA256SUMS.txt"


class AnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AnalysisError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError("GIT_FAILURE:" + " ".join(args)) from exc


def git_blob_bytes(path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "cat-file", "blob", f"HEAD:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError(f"GIT_BLOB_FAILURE:{path}") from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical(dict(row)) for row in rows)


def read_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    require(all(isinstance(row, dict) for row in rows), "JSONL_OBJECTS")
    return rows


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(head == expected_head, f"HEAD:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for ancestor, name in (
        (DESIGN_FREEZE_COMMIT, "DESIGN"),
        (RAW_FREEZE_COMMIT, "RAW_FREEZE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{name}_NOT_ANCESTOR")

    frozen_blobs = {
        DESIGN_PATH: DESIGN_BLOB,
        REFERENCE_ITEMS: REFERENCE_ITEMS_BLOB,
        REFERENCE_SUMMARY: REFERENCE_SUMMARY_BLOB,
        REFERENCE_MANIFEST: REFERENCE_MANIFEST_BLOB,
        REFERENCE_SUMS: REFERENCE_SUMS_BLOB,
        RAW_ITEMS: RAW_ITEMS_BLOB,
        RAW_SUMMARY: RAW_SUMMARY_BLOB,
        RAW_MANIFEST: RAW_MANIFEST_BLOB,
        RAW_SUMS: RAW_SUMS_BLOB,
    }
    for path, expected_blob in frozen_blobs.items():
        observed = git("rev-parse", f"HEAD:{path.as_posix()}")
        require(observed == expected_blob, f"FROZEN_BLOB:{path}")


def validate_sha_sums(
    sums_raw: bytes,
    expected: Mapping[str, str],
    *,
    label: str,
) -> None:
    observed: dict[str, str] = {}
    for line in sums_raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in observed, f"{label}_DUPLICATE:{name}")
        observed[name] = digest
    require(observed == dict(expected), f"{label}_CONTENT")


def validate_and_load_frozen_inputs() -> dict[str, Any]:
    ref_items_raw = git_blob_bytes(REFERENCE_ITEMS)
    ref_summary_raw = git_blob_bytes(REFERENCE_SUMMARY)
    ref_manifest_raw = git_blob_bytes(REFERENCE_MANIFEST)
    ref_sums_raw = git_blob_bytes(REFERENCE_SUMS)

    raw_items_raw = git_blob_bytes(RAW_ITEMS)
    raw_summary_raw = git_blob_bytes(RAW_SUMMARY)
    raw_manifest_raw = git_blob_bytes(RAW_MANIFEST)
    raw_sums_raw = git_blob_bytes(RAW_SUMS)

    require(
        sha256_bytes(ref_items_raw) == REFERENCE_ITEMS_SHA256,
        "REFERENCE_ITEMS_SHA256",
    )
    require(
        sha256_bytes(ref_summary_raw) == REFERENCE_SUMMARY_SHA256,
        "REFERENCE_SUMMARY_SHA256",
    )
    require(
        sha256_bytes(ref_manifest_raw) == REFERENCE_MANIFEST_SHA256,
        "REFERENCE_MANIFEST_SHA256",
    )

    require(
        sha256_bytes(raw_items_raw) == RAW_ITEMS_SHA256,
        "RAW_ITEMS_SHA256",
    )
    require(
        sha256_bytes(raw_summary_raw) == RAW_SUMMARY_SHA256,
        "RAW_SUMMARY_SHA256",
    )
    require(
        sha256_bytes(raw_manifest_raw) == RAW_MANIFEST_SHA256,
        "RAW_MANIFEST_SHA256",
    )
    require(
        sha256_bytes(raw_sums_raw) == RAW_SUMS_SHA256,
        "RAW_SUMS_SHA256",
    )

    validate_sha_sums(
        ref_sums_raw,
        {
            "artifact_manifest.json": REFERENCE_MANIFEST_SHA256,
            "finite_epsilon_principal_decomposition_items.jsonl":
                REFERENCE_ITEMS_SHA256,
            "finite_epsilon_principal_decomposition_summary.json":
                REFERENCE_SUMMARY_SHA256,
        },
        label="REFERENCE_SUMS",
    )
    validate_sha_sums(
        raw_sums_raw,
        {
            "artifact_manifest.json": RAW_MANIFEST_SHA256,
            "small_epsilon_robustness_items.jsonl": RAW_ITEMS_SHA256,
            "small_epsilon_robustness_summary.json": RAW_SUMMARY_SHA256,
        },
        label="RAW_SUMS",
    )

    ref_items = read_jsonl_bytes(ref_items_raw)
    raw_items = read_jsonl_bytes(raw_items_raw)
    require(len(ref_items) == len(raw_items) == N, "ITEM_COUNT")

    ref_summary = json.loads(ref_summary_raw.decode("utf-8"))
    raw_summary = json.loads(raw_summary_raw.decode("utf-8"))
    ref_manifest = json.loads(ref_manifest_raw.decode("utf-8"))
    raw_manifest = json.loads(raw_manifest_raw.decode("utf-8"))

    require(ref_summary["epsilon"] == REFERENCE_EPSILON, "REFERENCE_EPSILON")
    require(ref_summary["source_pair_count"] == N, "REFERENCE_N")
    require(
        ref_summary["primary_inference_executed"] is False
        and ref_summary["scientific_conclusion"] is None,
        "REFERENCE_ANALYSIS_BOUNDARY",
    )

    require(
        raw_summary["schema_version"]
        == "gen4-small-epsilon-robustness-summary-v1",
        "RAW_SUMMARY_SCHEMA",
    )
    require(
        raw_summary["result"]
        == "PASS_SMALL_EPSILON_ROBUSTNESS_RAW_OBSERVATION",
        "RAW_RESULT",
    )
    require(raw_summary["execution_head"] == RAW_EXECUTION_HEAD, "RAW_HEAD")
    require(raw_summary["source_pair_count"] == N, "RAW_N")
    require(
        tuple(float(x) for x in raw_summary["new_epsilons"])
        == NEW_EPSILONS,
        "RAW_EPSILONS",
    )
    require(
        raw_summary["reference_items_sha256"] == REFERENCE_ITEMS_SHA256
        and raw_summary["reference_summary_sha256"] == REFERENCE_SUMMARY_SHA256
        and raw_summary["reference_manifest_sha256"] == REFERENCE_MANIFEST_SHA256,
        "RAW_REFERENCE_LINEAGE",
    )
    require(
        raw_summary["representative_checkpoint_sha256"] == CHECKPOINT_SHA256,
        "RAW_CHECKPOINT",
    )
    require(
        raw_summary["reference_scientific_model_forward_count_this_run"] == 0
        and raw_summary["scientific_model_forward_count_this_run"] == 24000
        and raw_summary["new_original_basis_scientific_model_forward_count"] == 0,
        "RAW_FORWARD_ACCOUNTING",
    )
    require(
        raw_summary["primary_inference_executed"] is False
        and raw_summary["p_value_count_added"] == 0
        and raw_summary["multiplicity_correction_executed"] is False
        and raw_summary["scientific_conclusion"] is None,
        "RAW_ANALYSIS_BOUNDARY",
    )

    require(
        ref_manifest["schema_version"]
        == "gen4-finite-epsilon-five-plane-decomposition-manifest-v1",
        "REFERENCE_MANIFEST_SCHEMA",
    )
    require(
        raw_manifest["schema_version"]
        == "gen4-small-epsilon-robustness-manifest-v1",
        "RAW_MANIFEST_SCHEMA",
    )

    for index, (ref_item, raw_item) in enumerate(
        zip(ref_items, raw_items, strict=True)
    ):
        pair = f"xg1_fact_{2401 + index}"
        require(
            ref_item["source_pair_id"] == raw_item["source_pair_id"] == pair,
            f"PAIR:{index}",
        )
        require(
            int(ref_item["pair_index"]) == int(raw_item["pair_index"]) == index,
            f"PAIR_INDEX:{index}",
        )
        require(
            ref_item["plane_order"] == raw_item["plane_order"] == list(PLANE_ORDER),
            f"PLANE_ORDER:{index}",
        )
        require(
            tuple(ref_item["principal_direction_order"])
            == tuple(raw_item["principal_direction_order"])
            == DIRECTION_ORDER,
            f"DIRECTION_ORDER:{index}",
        )
        require(
            float(ref_item["prior_native_q0"])
            == float(raw_item["prior_native_q0"]),
            f"Q0_JOIN:{index}",
        )
        require(
            float(ref_item["epsilon"]) == REFERENCE_EPSILON,
            f"REFERENCE_ITEM_EPS:{index}",
        )
        require(
            tuple(
                float(obs["epsilon"])
                for obs in raw_item["epsilon_observations"]
            ) == NEW_EPSILONS,
            f"RAW_ITEM_EPS:{index}",
        )
        require(
            len(ref_item["principal_direction_probes"]) == 10
            and all(
                len(obs["principal_direction_probes"]) == 10
                for obs in raw_item["epsilon_observations"]
            ),
            f"PROBE_COUNT:{index}",
        )

    return {
        "reference_items": ref_items,
        "raw_items": raw_items,
        "reference_summary": ref_summary,
        "raw_summary": raw_summary,
    }


def linear_quantile(values: Sequence[float], q: float) -> float | None:
    finite = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not finite:
        return None
    require(0.0 <= q <= 1.0, "QUANTILE_RANGE")
    if len(finite) == 1:
        return finite[0]
    h = (len(finite) - 1) * q
    lo = math.floor(h)
    hi = math.ceil(h)
    if lo == hi:
        return finite[lo]
    w = h - lo
    return finite[lo] * (1.0 - w) + finite[hi] * w


def mean(values: Sequence[float]) -> float:
    require(bool(values), "MEAN_EMPTY")
    return math.fsum(float(x) for x in values) / len(values)


def rms(values: Sequence[float]) -> float:
    require(bool(values), "RMS_EMPTY")
    return math.sqrt(
        math.fsum(float(x) * float(x) for x in values) / len(values)
    )


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    require(len(xs) == len(ys) and bool(xs), "PEARSON_LENGTH")
    mx = mean(xs)
    my = mean(ys)
    dx = [float(x) - mx for x in xs]
    dy = [float(y) - my for y in ys]
    sx2 = math.fsum(x * x for x in dx)
    sy2 = math.fsum(y * y for y in dy)
    if sx2 == 0.0 or sy2 == 0.0:
        return None
    return math.fsum(x * y for x, y in zip(dx, dy, strict=True)) / math.sqrt(
        sx2 * sy2
    )


def cosine(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    require(len(xs) == len(ys) and bool(xs), "COSINE_LENGTH")
    nx = math.sqrt(math.fsum(float(x) * float(x) for x in xs))
    ny = math.sqrt(math.fsum(float(y) * float(y) for y in ys))
    if nx == 0.0 or ny == 0.0:
        return None
    return math.fsum(
        float(x) * float(y)
        for x, y in zip(xs, ys, strict=True)
    ) / (nx * ny)


def sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def observation_from_reference(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "epsilon": REFERENCE_EPSILON,
        "prior_native_q0": float(item["prior_native_q0"]),
        "plane_contributions": {
            p: float(item["plane_contributions"][p])
            for p in PLANE_ORDER
        },
        "Q_principal": float(item["Q_principal"]),
        "reconstruction_residual": float(item["reconstruction_residual"]),
        "absolute_reconstruction_residual":
            float(item["absolute_reconstruction_residual"]),
        "absolute_relative_reconstruction_residual_to_Q0":
            item["absolute_relative_reconstruction_residual_to_Q0"],
        "principal_direction_probes": item["principal_direction_probes"],
    }


def observation_from_raw(
    item: Mapping[str, Any],
    epsilon: float,
) -> dict[str, Any]:
    matches = [
        obs
        for obs in item["epsilon_observations"]
        if float(obs["epsilon"]) == float(epsilon)
    ]
    require(len(matches) == 1, f"EPSILON_OBSERVATION:{epsilon}")
    obs = matches[0]
    return {
        "epsilon": float(epsilon),
        "prior_native_q0": float(item["prior_native_q0"]),
        "plane_contributions": {
            p: float(obs["plane_contributions"][p])
            for p in PLANE_ORDER
        },
        "Q_principal": float(obs["Q_principal"]),
        "reconstruction_residual": float(obs["reconstruction_residual"]),
        "absolute_reconstruction_residual":
            float(obs["absolute_reconstruction_residual"]),
        "absolute_relative_reconstruction_residual_to_Q0":
            obs["absolute_relative_reconstruction_residual_to_Q0"],
        "principal_direction_probes": obs["principal_direction_probes"],
    }


def epsilon_rows(
    reference_items: Sequence[Mapping[str, Any]],
    raw_items: Sequence[Mapping[str, Any]],
) -> dict[float, list[dict[str, Any]]]:
    return {
        REFERENCE_EPSILON: [
            observation_from_reference(item)
            for item in reference_items
        ],
        NEW_EPSILONS[0]: [
            observation_from_raw(item, NEW_EPSILONS[0])
            for item in raw_items
        ],
        NEW_EPSILONS[1]: [
            observation_from_raw(item, NEW_EPSILONS[1])
            for item in raw_items
        ],
    }


def spectral_profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    means = {
        plane: mean(
            [float(row["plane_contributions"][plane]) for row in rows]
        )
        for plane in PLANE_ORDER
    }
    ordered = sorted(
        PLANE_ORDER,
        key=lambda p: (-means[p], PLANE_ORDER.index(p)),
    )
    max_value = max(means.values())
    maxima = [p for p in PLANE_ORDER if means[p] == max_value]
    dominant = maxima[0] if len(maxima) == 1 else None
    return {
        "mean_plane_contributions": means,
        "mean_plane_contribution_signs": {
            p: sign(means[p]) for p in PLANE_ORDER
        },
        "rank_order_descending": ordered,
        "unique_spectral_dominant_candidate": dominant,
        "p3_minus_p5_mean_contribution":
            means["P3"] - means["P5"],
    }


def probe_numerator(
    probe: Mapping[str, Any],
    *,
    epsilon: float,
) -> float:
    f_plus = float(probe["F_plus"])
    f_minus = float(probe["F_minus"])
    numerator = f_plus - f_minus

    if "central_difference_numerator" in probe:
        require(
            float(probe["central_difference_numerator"]) == numerator,
            "PROBE_NUMERATOR_STORED_MISMATCH",
        )

    require(
        float(probe["J"]) == numerator / (2.0 * float(epsilon)),
        "PROBE_J_NUMERATOR_MISMATCH",
    )
    return numerator


def j_vectors(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[float]]:
    j_values: list[float] = []
    numerators: list[float] = []
    for row in rows:
        epsilon = float(row["epsilon"])
        probes = row["principal_direction_probes"]
        require(
            tuple(probe["direction_key"] for probe in probes)
            == DIRECTION_ORDER,
            "PROBE_ORDER_ANALYSIS",
        )
        for probe in probes:
            j_values.append(float(probe["J"]))
            numerators.append(
                probe_numerator(probe, epsilon=epsilon)
            )
    require(len(j_values) == N * len(DIRECTION_ORDER), "J_COUNT")
    return j_values, numerators


def distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    abs_values = [abs(float(x)) for x in values]
    finite = [x for x in abs_values if math.isfinite(x)]
    return {
        "nonfinite_count": len(abs_values) - len(finite),
        "exact_zero_count": sum(x == 0.0 for x in finite),
        "min": min(finite) if finite else None,
        "q05": linear_quantile(finite, 0.05),
        "median": linear_quantile(finite, 0.50),
        "q95": linear_quantile(finite, 0.95),
        "max": max(finite) if finite else None,
    }


def reconstruction_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    q0 = [float(row["prior_native_q0"]) for row in rows]
    qp = [float(row["Q_principal"]) for row in rows]
    residual = [float(row["reconstruction_residual"]) for row in rows]
    abs_residual = [
        float(row["absolute_reconstruction_residual"]) for row in rows
    ]
    abs_relative = [
        float(row["absolute_relative_reconstruction_residual_to_Q0"])
        for row in rows
        if row["absolute_relative_reconstruction_residual_to_Q0"] is not None
        and math.isfinite(
            float(row["absolute_relative_reconstruction_residual_to_Q0"])
        )
    ]
    q0_rms = rms(q0)
    q0_abs_mean = mean([abs(x) for x in q0])
    res_rmse = rms(residual)
    mean_abs_res = mean(abs_residual)

    return {
        "mean_Q0": mean(q0),
        "mean_Q_principal": mean(qp),
        "mean_residual": mean(residual),
        "mean_absolute_residual": mean_abs_res,
        "residual_RMSE": res_rmse,
        "Q0_RMS": q0_rms,
        "normalized_RMSE_over_RMS_Q0":
            (res_rmse / q0_rms) if q0_rms != 0.0 else None,
        "normalized_MAE_over_mean_abs_Q0":
            (mean_abs_res / q0_abs_mean) if q0_abs_mean != 0.0 else None,
        "pearson_Q0_Q_principal": pearson(q0, qp),
        "sign_agreement": mean([
            1.0 if sign(a) == sign(b) else 0.0
            for a, b in zip(q0, qp, strict=True)
        ]),
        "absolute_relative_residual_quantiles": {
            "q50": linear_quantile(abs_relative, 0.50),
            "q90": linear_quantile(abs_relative, 0.90),
            "q95": linear_quantile(abs_relative, 0.95),
            "q99": linear_quantile(abs_relative, 0.99),
        },
    }


def interscale_difference(
    a: Sequence[float],
    b: Sequence[float],
) -> dict[str, Any]:
    require(len(a) == len(b), "INTERSCALE_LENGTH")
    diffs = [
        float(x) - float(y)
        for x, y in zip(a, b, strict=True)
    ]
    abs_diffs = [abs(x) for x in diffs]
    return {
        "RMS_difference": rms(diffs),
        "mean_absolute_difference": mean(abs_diffs),
        "median_absolute_difference": linear_quantile(abs_diffs, 0.50),
    }


def analyze_inputs(
    reference_items: Sequence[Mapping[str, Any]],
    raw_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows_by_eps = epsilon_rows(reference_items, raw_items)

    profiles = {
        str(eps): spectral_profile(rows_by_eps[eps])
        for eps in ALL_EPSILONS
    }

    normalized_profiles: dict[str, list[float] | None] = {}
    raw_profile_vectors: dict[float, list[float]] = {}
    for eps in ALL_EPSILONS:
        vector = [
            float(profiles[str(eps)]["mean_plane_contributions"][p])
            for p in PLANE_ORDER
        ]
        raw_profile_vectors[eps] = vector
        norm = math.sqrt(math.fsum(x * x for x in vector))
        normalized_profiles[str(eps)] = (
            [x / norm for x in vector] if norm != 0.0 else None
        )

    profile_similarity = {
        "cosine_0.025_vs_0.0125":
            cosine(raw_profile_vectors[0.025], raw_profile_vectors[0.0125]),
        "cosine_0.025_vs_0.00625":
            cosine(raw_profile_vectors[0.025], raw_profile_vectors[0.00625]),
        "cosine_0.0125_vs_0.00625":
            cosine(raw_profile_vectors[0.0125], raw_profile_vectors[0.00625]),
    }

    js: dict[float, list[float]] = {}
    nums: dict[float, list[float]] = {}
    magnitude: dict[str, Any] = {}
    degeneracy: dict[str, Any] = {}
    reconstruction: dict[str, Any] = {}

    for eps in ALL_EPSILONS:
        j_values, numerator_values = j_vectors(rows_by_eps[eps])
        js[eps] = j_values
        nums[eps] = numerator_values
        finite_j = [x for x in j_values if math.isfinite(x)]
        require(len(finite_j) == len(j_values), f"NONFINITE_J:{eps}")
        magnitude[str(eps)] = {
            "RMS_J": rms(j_values),
        }
        degeneracy[str(eps)] = {
            "J_absolute_distribution": distribution_summary(j_values),
            "central_difference_numerator_absolute_distribution":
                distribution_summary(numerator_values),
        }
        reconstruction[str(eps)] = reconstruction_summary(rows_by_eps[eps])

    d1 = interscale_difference(js[0.0125], js[0.025])
    d2 = interscale_difference(js[0.00625], js[0.0125])
    ratio = (
        d2["RMS_difference"] / d1["RMS_difference"]
        if d1["RMS_difference"] != 0.0
        else None
    )
    convergence = {
        "RMS_J_by_epsilon": {
            str(eps): magnitude[str(eps)]["RMS_J"]
            for eps in ALL_EPSILONS
        },
        "0.0125_minus_0.025": d1,
        "0.00625_minus_0.0125": d2,
        "adjacent_RMS_difference_ratio_second_over_first": ratio,
    }

    gates: dict[str, Any] = {}
    passes = []
    for eps in NEW_EPSILONS:
        profile = profiles[str(eps)]
        dominant = profile["unique_spectral_dominant_candidate"]
        contrast = float(profile["p3_minus_p5_mean_contribution"])
        gate = {
            "epsilon": eps,
            "unique_spectral_dominant_candidate": dominant,
            "p3_unique_argmax": dominant == "P3",
            "p3_minus_p5_mean_contribution": contrast,
            "p3_minus_p5_positive": contrast > 0.0,
        }
        gate["pass"] = (
            gate["p3_unique_argmax"]
            and gate["p3_minus_p5_positive"]
        )
        gates[str(eps)] = gate
        passes.append(bool(gate["pass"]))

    preserved = all(passes)
    result = POSITIVE_RESULT if preserved else NEGATIVE_RESULT

    return {
        "result": result,
        "core_qualitative_robustness_rule": {
            "new_epsilon_gates": gates,
            "all_new_epsilons_pass": preserved,
            "rule": (
                "At both 0.0125 and 0.00625: unique spectral dominant "
                "candidate must be P3 and mean C(P3)-mean C(P5) must be > 0."
            ),
            "inferential_test_count": 0,
            "p_value_count": 0,
            "multiplicity_correction_count": 0,
            "epsilon_selection_count": 0,
        },
        "spectral_profiles": profiles,
        "normalized_mean_profile_vectors": normalized_profiles,
        "normalized_profile_similarity": profile_similarity,
        "finite_difference_magnitude_convergence": convergence,
        "reconstruction_stability": reconstruction,
        "numerical_degeneracy": degeneracy,
    }


def build_analysis(
    *,
    expected_head: str,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    values = analyze_inputs(
        loaded["reference_items"],
        loaded["raw_items"],
    )
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "result": values["result"],
        "analysis_head": expected_head,
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "input_provenance": {
            "reference_run_name": REFERENCE_RUN_NAME,
            "reference_items_git_blob": REFERENCE_ITEMS_BLOB,
            "reference_items_sha256": REFERENCE_ITEMS_SHA256,
            "reference_summary_git_blob": REFERENCE_SUMMARY_BLOB,
            "reference_summary_sha256": REFERENCE_SUMMARY_SHA256,
            "raw_run_name": RAW_RUN_NAME,
            "raw_execution_head": RAW_EXECUTION_HEAD,
            "raw_items_git_blob": RAW_ITEMS_BLOB,
            "raw_items_sha256": RAW_ITEMS_SHA256,
            "raw_summary_git_blob": RAW_SUMMARY_BLOB,
            "raw_summary_sha256": RAW_SUMMARY_SHA256,
            "raw_manifest_git_blob": RAW_MANIFEST_BLOB,
            "raw_manifest_sha256": RAW_MANIFEST_SHA256,
            "raw_sums_git_blob": RAW_SUMS_BLOB,
            "raw_sums_sha256": RAW_SUMS_SHA256,
            "representative_checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "population": {
            "family": "XG1",
            "pair_first": "xg1_fact_2401",
            "pair_last": "xg1_fact_2700",
            "N": N,
        },
        "epsilon_set": {
            "reference": REFERENCE_EPSILON,
            "new": list(NEW_EPSILONS),
            "reference_rerun": False,
        },
        **values,
        "analysis_execution": {
            "model_forward_count": 0,
            "cuda_executed": False,
            "training_executed": False,
            "backward_executed": False,
            "task_heads_executed": False,
            "logits_read": False,
            "inferential_test_count": 0,
            "p_value_count_added": 0,
            "multiplicity_correction_count": 0,
            "epsilon_selection_count": 0,
            "post_hoc_threshold_added": False,
            "rescue_performed": False,
        },
        "claim_boundary": {
            "allowed_if_positive": (
                "Across the preregistered smaller finite-difference scales "
                "0.0125 and 0.00625 on the frozen XG1 2401..2700 population, "
                "the P3 spectral dominant candidate and positive P3-minus-P5 "
                "mean contribution contrast were preserved."
            ),
            "not_established": [
                "causal additivity",
                "plane independence",
                "exact finite-epsilon identity",
                "optimal epsilon",
                "improved steering",
                "AVeriTeC utility",
                "new causal rank discovery",
            ],
        },
    }


def render_report(a: Mapping[str, Any]) -> str:
    lines = [
        "# Gen4 small-epsilon robustness analysis",
        "",
        "## Result",
        "",
        f"`{a['result']}`",
        "",
        "This analysis is descriptive only: 0 inferential tests, 0 p-values, "
        "0 multiplicity corrections, and 0 epsilon selections.",
        "",
        "## Frozen inputs",
        "",
        f"- analysis HEAD: `{a['analysis_head']}`",
        f"- raw freeze: `{a['raw_freeze_commit']}`",
        f"- reference epsilon: `{a['epsilon_set']['reference']}`",
        f"- new epsilons: `{a['epsilon_set']['new']}`",
        f"- population N: `{a['population']['N']}`",
        "",
        "## Mean signed spectral profiles",
        "",
    ]
    for eps in ALL_EPSILONS:
        p = a["spectral_profiles"][str(eps)]
        lines.append(f"### epsilon = {eps}")
        lines.append("")
        for plane in PLANE_ORDER:
            value = p["mean_plane_contributions"][plane]
            lines.append(f"- {plane}: `{value:.17g}`")
        lines.extend([
            f"- rank: `{p['rank_order_descending']}`",
            "- unique spectral dominant candidate: "
            f"`{p['unique_spectral_dominant_candidate']}`",
            "- P3 - P5 mean contribution: "
            f"`{p['p3_minus_p5_mean_contribution']:.17g}`",
            "",
        ])

    lines.extend([
        "## Core qualitative robustness gates",
        "",
    ])
    for eps in NEW_EPSILONS:
        g = a["core_qualitative_robustness_rule"][
            "new_epsilon_gates"
        ][str(eps)]
        lines.extend([
            f"### epsilon = {eps}",
            "",
            f"- P3 unique argmax: `{g['p3_unique_argmax']}`",
            f"- P3-P5 positive: `{g['p3_minus_p5_positive']}`",
            f"- gate pass: `{g['pass']}`",
            "",
        ])

    lines.extend([
        "## Normalized profile similarity",
        "",
    ])
    for key, value in a["normalized_profile_similarity"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend([
        "",
        "## Finite-difference magnitude convergence",
        "",
        "```json",
        json.dumps(
            a["finite_difference_magnitude_convergence"],
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Reconstruction stability",
        "",
        "```json",
        json.dumps(
            a["reconstruction_stability"],
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Numerical degeneracy diagnostics",
        "",
        "```json",
        json.dumps(
            a["numerical_degeneracy"],
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Interpretation boundary",
        "",
        "Finite-epsilon spectral contribution magnitude is descriptive and "
        "must not be interpreted as a new causal-plane ranking.",
        "",
        "This result does not establish causal additivity, plane independence, "
        "an exact finite-epsilon identity, an optimal epsilon, improved steering, "
        "AVeriTeC utility, or a new causal rank discovery.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    analysis: Mapping[str, Any],
) -> None:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    output_dir.mkdir(parents=True, exist_ok=False)

    analysis_path = output_dir / ANALYSIS_FILE
    report_path = output_dir / REPORT_FILE
    sums_path = output_dir / SUMS_FILE

    analysis_path.write_bytes(canonical(dict(analysis)))
    report_path.write_text(
        render_report(analysis),
        encoding="utf-8",
        newline="\n",
    )
    sums_path.write_text(
        f"{sha256_file(analysis_path)}  {ANALYSIS_FILE}\n"
        f"{sha256_file(report_path)}  {REPORT_FILE}\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze frozen epsilon=0.025 reference and frozen epsilon="
            "0.0125/0.00625 raw observations under the preregistered "
            "descriptive small-epsilon robustness design."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    authenticate_repo(args.expected_head)
    loaded = validate_and_load_frozen_inputs()
    analysis = build_analysis(
        expected_head=args.expected_head,
        loaded=loaded,
    )
    write_outputs(args.output_dir, analysis)

    print("RESULT=" + analysis["result"])
    print("ANALYSIS_HEAD=" + analysis["analysis_head"])
    print("N=300")
    print("EPSILONS=0.025,0.0125,0.00625")
    for eps in ALL_EPSILONS:
        p = analysis["spectral_profiles"][str(eps)]
        print(
            f"EPSILON_{eps}_DOMINANT="
            f"{p['unique_spectral_dominant_candidate']}"
        )
        print(
            f"EPSILON_{eps}_P3_MINUS_P5="
            f"{p['p3_minus_p5_mean_contribution']!r}"
        )
    print(
        "ALL_NEW_EPSILON_GATES_PASS="
        + str(
            analysis["core_qualitative_robustness_rule"][
                "all_new_epsilons_pass"
            ]
        )
    )
    print("INFERENTIAL_TEST_COUNT=0")
    print("P_VALUE_COUNT=0")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
