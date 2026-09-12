"""K0-RVG softplus operating-point audit.

Validated parent result:

    Z = dt_proj(time_step)
    T = softplus(Z)
    delta_T = G_sec * delta_Z

and:

    P_Z = mean(U) * mean(B) * delta_Z
    Q_T = G_sec * P_Z

The parent stage established that the corr k=1 -> k=2 Q_T rebound usually
occurs while ||P_Z|| decreases and effective softplus transmission rises.

This audit asks why effective transmission rises.

It separates:

1. broad channel-wise operating-point movement, measured without P_Z weighting;
2. redistribution of P_Z energy toward high-transmission channels, measured
   with exact per-channel P_Z^2 energy weights.

No intervention is performed and no raw vectors are persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


PARENT_FREEZE_COMMIT = "4ea1233c0dc45bddb6e48b9b7a3d7ff7d97b77b9"

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_time_step_softplus_secant_2700f2a_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "af0a8fd2c03a2d1632cc167019d289f9f3331b7cbe348d69f740d5baa919924b"
)

PARENT_RUNNER_REL = (
    "scripts/longterm_k0_rvg_time_step_softplus_secant_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "1f3539d7500bd0f8d933d680f5bc3ce2b055be82ab37550c0334c5673d4a690d"
)

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FORWARD_COUNT = 1344

PRIMARY_LAYER = 23
POST_HORIZON = 6
RELATIVE_COORDINATES = tuple(range(-1, 7))

EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"

QUESTION = (
    "Does corr k2 softplus transmission rise because channels broadly move "
    "to a higher-slope Z operating point, because P_Z energy reallocates "
    "toward already-high-gain channels, or both?"
)

COMPOSITION_RTOL = 1e-12
GAIN_TOL = 1e-6


class OperatingPointAuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise OperatingPointAuditError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def import_module(path: Path, name: str):
    require(path.is_file(), f"MODULE_MISSING:{path}")

    spec = importlib.util.spec_from_file_location(name, path)

    require(
        spec is not None and spec.loader is not None,
        f"MODULE_SPEC_FAILURE:{path}",
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def load_parent_runner(root: Path):
    path = root / PARENT_RUNNER_REL

    require(
        sha256_bytes(path.read_bytes()) == PARENT_RUNNER_SHA256,
        "PARENT_RUNNER_SHA256_MISMATCH",
    )

    return import_module(
        path,
        "k0_rvg_softplus_secant_parent",
    )


def build_plan(root: Path, parent: Any):
    discrete_b = parent.load_parent_runner(root)

    (
        write,
        carry,
        magnitude,
        base,
        plan,
        cohort,
        signatures,
    ) = parent.build_plan(
        root,
        discrete_b,
    )

    require(
        len(plan) == EXPECTED_PAIR_ROLE_COUNT,
        "PLAN_COUNT_MISMATCH",
    )

    require(
        len(cohort) == EXPECTED_COMMON_COUNT,
        "COMMON_COHORT_COUNT_MISMATCH",
    )

    return (
        discrete_b,
        write,
        carry,
        magnitude,
        base,
        plan,
        cohort,
        signatures,
    )


def authenticate_parent(root: Path, base: Any):
    repo = base.authenticate_repo(root)

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            PARENT_FREEZE_COMMIT,
            repo["head"],
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    require(
        rc == 0,
        "PARENT_FREEZE_NOT_ANCESTOR",
    )

    raw = base.git_bytes(
        root,
        f"{PARENT_FREEZE_COMMIT}:{PARENT_SUMMARY_REL}",
    )

    require(
        sha256_bytes(raw) == PARENT_SUMMARY_SHA256,
        "PARENT_SUMMARY_SHA256_MISMATCH",
    )

    current = root / PARENT_SUMMARY_REL

    require(
        current.is_file(),
        "PARENT_SUMMARY_MISSING",
    )

    require(
        current.read_bytes() == raw,
        "PARENT_SUMMARY_WORKTREE_DRIFT",
    )

    summary = json.loads(raw)

    require(
        summary.get("schema_version")
        == "k0-rvg-time-step-softplus-secant-summary-v1",
        "PARENT_SCHEMA_MISMATCH",
    )

    require(
        summary.get("item_count") == EXPECTED_ITEM_COUNT,
        "PARENT_ITEM_COUNT_MISMATCH",
    )

    require(
        summary.get("pair_role_count")
        == EXPECTED_PAIR_ROLE_COUNT,
        "PARENT_PAIR_ROLE_COUNT_MISMATCH",
    )

    require(
        summary.get("common_ddsssss_item_count")
        == EXPECTED_COMMON_COUNT,
        "PARENT_COMMON_COUNT_MISMATCH",
    )

    require(
        summary.get("primary_layer") == PRIMARY_LAYER,
        "PARENT_LAYER_MISMATCH",
    )

    require(
        summary.get("relative_coordinates")
        == list(RELATIVE_COORDINATES),
        "PARENT_COORDINATE_MISMATCH",
    )

    require(
        summary.get("execution_protocol")
        == EXECUTION_PROTOCOL,
        "PARENT_PROTOCOL_MISMATCH",
    )

    return repo, summary


def weighted_mean(values, weights) -> float:
    import torch

    total = torch.sum(weights)

    if float(total.item()) == 0.0:
        return 0.0

    return float(
        (
            torch.sum(values * weights)
            / total
        ).item()
    )


def weighted_rms(values, weights) -> float:
    import torch

    total = torch.sum(weights)

    if float(total.item()) == 0.0:
        return 0.0

    return float(
        torch.sqrt(
            torch.sum(values * values * weights)
            / total
        ).item()
    )


def weighted_median(values, weights) -> float:
    import torch

    total = float(torch.sum(weights).item())

    if total == 0.0:
        return 0.0

    order = torch.argsort(values)

    sorted_values = values[order]
    sorted_weights = weights[order]

    cumulative = torch.cumsum(
        sorted_weights,
        dim=0,
    )

    target = 0.5 * total

    index = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(
                target,
                dtype=cumulative.dtype,
                device=cumulative.device,
            ),
        ).item()
    )

    index = min(
        index,
        sorted_values.numel() - 1,
    )

    return float(
        sorted_values[index].item()
    )


def energy_fraction(mask, weights) -> float:
    import torch

    total = torch.sum(weights)

    if float(total.item()) == 0.0:
        return 0.0

    return float(
        (
            torch.sum(weights[mask])
            / total
        ).item()
    )


def metric_rows_for_pair(
    row: Mapping[str, Any],
    matched: Mapping[int, Mapping[str, Any]],
    swapped: Mapping[int, Mapping[str, Any]],
    cohort: frozenset[int],
    signature: str,
):
    import torch

    idx = int(row["local_template_index"])
    role = str(row["role"])
    anchor = int(row["anchor"])

    provisional = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k

        Um32 = matched[token]["U"]
        Zm32 = matched[token]["Z"]
        Tm32 = matched[token]["T"]
        Bm32 = matched[token]["B"]
        Wm32 = matched[token]["W"]

        Us32 = swapped[token]["U"]
        Zs32 = swapped[token]["Z"]
        Ts32 = swapped[token]["T"]
        Bs32 = swapped[token]["B"]
        Ws32 = swapped[token]["W"]

        u_equal = parent_torch_equal(Um32, Us32)
        z_equal = parent_torch_equal(Zm32, Zs32)
        t_equal = parent_torch_equal(Tm32, Ts32)
        b_equal = parent_torch_equal(Bm32, Bs32)
        w_equal = parent_torch_equal(Wm32, Ws32)

        if k == -1:
            require(
                (
                    u_equal
                    and z_equal
                    and t_equal
                    and b_equal
                    and w_equal
                ),
                f"K_MINUS_1_IDENTITY_FAILURE:{idx}:{role}",
            )

        Um = Um32.to(torch.float64)
        Zm = Zm32.to(torch.float64)
        Tm = Tm32.to(torch.float64)
        Bm = Bm32.to(torch.float64)
        Wm = Wm32.to(torch.float64)

        Us = Us32.to(torch.float64)
        Zs = Zs32.to(torch.float64)
        Ts = Ts32.to(torch.float64)
        Bs = Bs32.to(torch.float64)
        Ws = Ws32.to(torch.float64)

        mean_U = 0.5 * (Um + Us)
        mean_B = 0.5 * (Bm + Bs)

        delta_Z = Zm - Zs
        delta_T = Tm - Ts
        delta_W = Wm - Ws

        z_mid = 0.5 * (Zm + Zs)

        zero = delta_Z == 0
        nonzero = ~zero

        gain = torch.empty_like(delta_Z)

        gain[nonzero] = (
            delta_T[nonzero]
            / delta_Z[nonzero]
        )

        if bool(zero.any().item()):
            gain[zero] = torch.sigmoid(
                z_mid[zero]
            )

        require(
            bool(torch.isfinite(gain).all().item()),
            f"NONFINITE_GAIN:{idx}:{role}:{k}",
        )

        require(
            float(gain.min().item()) >= -GAIN_TOL,
            f"GAIN_BELOW_ZERO:{idx}:{role}:{k}",
        )

        require(
            float(gain.max().item()) <= 1.0 + GAIN_TOL,
            f"GAIN_ABOVE_ONE:{idx}:{role}:{k}",
        )

        midpoint_gain = torch.sigmoid(z_mid)

        M = (
            mean_U[:, :, None]
            * mean_B[:, None, :]
        )

        p_z = (
            M
            * delta_Z[:, :, None]
        )

        q_t = (
            M
            * delta_T[:, :, None]
        )

        reconstructed_q_t = (
            gain[:, :, None]
            * p_z
        )

        residual = (
            q_t
            - reconstructed_q_t
        )

        p_z_l2 = float(
            torch.linalg.vector_norm(p_z).item()
        )

        q_t_l2 = float(
            torch.linalg.vector_norm(q_t).item()
        )

        delta_w_l2 = float(
            torch.linalg.vector_norm(delta_W).item()
        )

        residual_l2 = float(
            torch.linalg.vector_norm(residual).item()
        )

        residual_relative = (
            residual_l2
            / max(q_t_l2, 1e-12)
        )

        require(
            residual_relative <= COMPOSITION_RTOL,
            (
                "GAIN_RECONSTRUCTION_FAILURE:"
                f"{idx}:{role}:{k}:{residual_relative}"
            ),
        )

        effective_gain = (
            q_t_l2 / p_z_l2
            if p_z_l2 > 0.0
            else 0.0
        )

        # Collapse state-size dimension into exact per-channel P_Z^2 energy.
        channel_energy = (
            torch.sum(
                p_z * p_z,
                dim=2,
            )
            .squeeze(0)
        )

        gain_1d = gain.squeeze(0)
        z_mid_1d = z_mid.squeeze(0)
        z_m_1d = Zm.squeeze(0)
        z_s_1d = Zs.squeeze(0)
        abs_delta_z_1d = torch.abs(
            delta_Z.squeeze(0)
        )
        midpoint_gain_1d = midpoint_gain.squeeze(0)

        gain_energy_rms = weighted_rms(
            gain_1d,
            channel_energy,
        )

        require(
            math.isclose(
                gain_energy_rms,
                effective_gain,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            (
                "GAIN_RMS_IDENTITY_FAILURE:"
                f"{idx}:{role}:{k}:"
                f"{gain_energy_rms}:{effective_gain}"
            ),
        )

        midpoint_gain_energy_rms = weighted_rms(
            midpoint_gain_1d,
            channel_energy,
        )

        secant_midpoint_gap_rms = weighted_rms(
            gain_1d - midpoint_gain_1d,
            channel_energy,
        )

        # Unweighted metrics describe broad channel movement.
        z_mid_unweighted_mean = float(
            torch.mean(z_mid_1d).item()
        )
        z_mid_unweighted_median = float(
            torch.median(z_mid_1d).item()
        )

        z_m_unweighted_mean = float(
            torch.mean(z_m_1d).item()
        )
        z_s_unweighted_mean = float(
            torch.mean(z_s_1d).item()
        )

        gain_unweighted_mean = float(
            torch.mean(gain_1d).item()
        )
        gain_unweighted_median = float(
            torch.median(gain_1d).item()
        )

        # P_Z^2-weighted metrics describe the operating points that
        # actually carry pre-softplus divergence energy.
        z_mid_energy_mean = weighted_mean(
            z_mid_1d,
            channel_energy,
        )
        z_mid_energy_median = weighted_median(
            z_mid_1d,
            channel_energy,
        )

        z_m_energy_mean = weighted_mean(
            z_m_1d,
            channel_energy,
        )
        z_s_energy_mean = weighted_mean(
            z_s_1d,
            channel_energy,
        )

        gain_energy_mean = weighted_mean(
            gain_1d,
            channel_energy,
        )
        gain_energy_median = weighted_median(
            gain_1d,
            channel_energy,
        )

        abs_delta_z_energy_mean = weighted_mean(
            abs_delta_z_1d,
            channel_energy,
        )

        energy_gain_ge_050 = energy_fraction(
            gain_1d >= 0.50,
            channel_energy,
        )
        energy_gain_ge_075 = energy_fraction(
            gain_1d >= 0.75,
            channel_energy,
        )
        energy_gain_ge_090 = energy_fraction(
            gain_1d >= 0.90,
            channel_energy,
        )

        energy_z_mid_ge_zero = energy_fraction(
            z_mid_1d >= 0.0,
            channel_energy,
        )

        values = (
            p_z_l2,
            q_t_l2,
            delta_w_l2,
            residual_relative,
            effective_gain,
            gain_energy_rms,
            midpoint_gain_energy_rms,
            secant_midpoint_gap_rms,
            z_mid_unweighted_mean,
            z_mid_unweighted_median,
            z_m_unweighted_mean,
            z_s_unweighted_mean,
            gain_unweighted_mean,
            gain_unweighted_median,
            z_mid_energy_mean,
            z_mid_energy_median,
            z_m_energy_mean,
            z_s_energy_mean,
            gain_energy_mean,
            gain_energy_median,
            abs_delta_z_energy_mean,
            energy_gain_ge_050,
            energy_gain_ge_075,
            energy_gain_ge_090,
            energy_z_mid_ge_zero,
        )

        require(
            all(math.isfinite(v) for v in values),
            f"NONFINITE_METRIC:{idx}:{role}:{k}",
        )

        provisional.append(
            {
                "schema_version":
                    "k0-rvg-softplus-operating-point-row-v1",
                "local_template_index":
                    idx,
                "stable_item_id":
                    row["stable_item_id"],
                "role":
                    role,
                "relative_coordinate":
                    k,
                "token_index":
                    token,
                "divergence_anchor_token_index":
                    anchor,
                "token_equality_signature_k0_to_k6":
                    signature,
                "in_common_ddsssss_cohort":
                    idx in cohort,
                "hidden_states_exact_equal":
                    u_equal,
                "pre_softplus_z_exact_equal":
                    z_equal,
                "discrete_time_step_exact_equal":
                    t_equal,
                "B_exact_equal":
                    b_equal,
                "write_exact_equal":
                    w_equal,
                "p_z_l2":
                    p_z_l2,
                "q_t_l2":
                    q_t_l2,
                "delta_write_l2":
                    delta_w_l2,
                "effective_softplus_gain":
                    effective_gain,
                "gain_energy_rms":
                    gain_energy_rms,
                "midpoint_gain_energy_rms":
                    midpoint_gain_energy_rms,
                "secant_midpoint_gap_rms":
                    secant_midpoint_gap_rms,
                "z_mid_unweighted_mean":
                    z_mid_unweighted_mean,
                "z_mid_unweighted_median":
                    z_mid_unweighted_median,
                "z_m_unweighted_mean":
                    z_m_unweighted_mean,
                "z_s_unweighted_mean":
                    z_s_unweighted_mean,
                "gain_unweighted_mean":
                    gain_unweighted_mean,
                "gain_unweighted_median":
                    gain_unweighted_median,
                "z_mid_pz_energy_mean":
                    z_mid_energy_mean,
                "z_mid_pz_energy_median":
                    z_mid_energy_median,
                "z_m_pz_energy_mean":
                    z_m_energy_mean,
                "z_s_pz_energy_mean":
                    z_s_energy_mean,
                "gain_pz_energy_mean":
                    gain_energy_mean,
                "gain_pz_energy_median":
                    gain_energy_median,
                "abs_delta_z_pz_energy_mean":
                    abs_delta_z_energy_mean,
                "pz_energy_fraction_gain_ge_050":
                    energy_gain_ge_050,
                "pz_energy_fraction_gain_ge_075":
                    energy_gain_ge_075,
                "pz_energy_fraction_gain_ge_090":
                    energy_gain_ge_090,
                "pz_energy_fraction_z_mid_ge_zero":
                    energy_z_mid_ge_zero,
                "composition_relative_residual":
                    residual_relative,
                "source_snapshot_dtype":
                    "torch.float32",
                "metric_accumulation_dtype":
                    "torch.float64",
            }
        )

    k0_w = next(
        float(r["delta_write_l2"])
        for r in provisional
        if int(r["relative_coordinate"]) == 0
    )

    require(
        k0_w > 0.0
        and math.isfinite(k0_w),
        f"K0_WRITE_MAGNITUDE_INVALID:{idx}:{role}",
    )

    for r in provisional:
        r["p_z_l2_over_k0_w"] = (
            float(r["p_z_l2"])
            / k0_w
        )

        r["q_t_l2_over_k0_w"] = (
            float(r["q_t_l2"])
            / k0_w
        )

    return provisional


def aggregate(values):
    vals = [float(v) for v in values]

    require(
        bool(vals),
        "EMPTY_AGGREGATE",
    )

    require(
        all(math.isfinite(v) for v in vals),
        "NONFINITE_AGGREGATE",
    )

    return {
        "count": len(vals),
        "mean": float(
            statistics.fmean(vals)
        ),
        "median": float(
            statistics.median(vals)
        ),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


SUMMARY_FIELDS = (
    "p_z_l2",
    "q_t_l2",
    "delta_write_l2",
    "p_z_l2_over_k0_w",
    "q_t_l2_over_k0_w",
    "effective_softplus_gain",
    "gain_energy_rms",
    "midpoint_gain_energy_rms",
    "secant_midpoint_gap_rms",
    "z_mid_unweighted_mean",
    "z_mid_unweighted_median",
    "z_m_unweighted_mean",
    "z_s_unweighted_mean",
    "gain_unweighted_mean",
    "gain_unweighted_median",
    "z_mid_pz_energy_mean",
    "z_mid_pz_energy_median",
    "z_m_pz_energy_mean",
    "z_s_pz_energy_mean",
    "gain_pz_energy_mean",
    "gain_pz_energy_median",
    "abs_delta_z_pz_energy_mean",
    "pz_energy_fraction_gain_ge_050",
    "pz_energy_fraction_gain_ge_075",
    "pz_energy_fraction_gain_ge_090",
    "pz_energy_fraction_z_mid_ge_zero",
    "composition_relative_residual",
)


def aggregate_trajectory(rows):
    out = {
        "corr": {},
        "ctrl": {},
    }

    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            bucket = [
                r
                for r in rows
                if r["role"] == role
                and int(r["relative_coordinate"]) == k
            ]

            require(
                bool(bucket),
                f"EMPTY_BUCKET:{role}:{k}",
            )

            out[role][str(k)] = {
                field: aggregate(
                    float(r[field])
                    for r in bucket
                )
                for field in SUMMARY_FIELDS
            }

    return out


def make_summary(rows, cohort):
    require(
        len(rows)
        == EXPECTED_PAIR_ROLE_COUNT
        * len(RELATIVE_COORDINATES),
        "ROW_COUNT_MISMATCH",
    )

    common_rows = [
        r
        for r in rows
        if bool(
            r["in_common_ddsssss_cohort"]
        )
    ]

    require(
        len(common_rows)
        == EXPECTED_COMMON_COUNT
        * 2
        * len(RELATIVE_COORDINATES),
        "COMMON_ROW_COUNT_MISMATCH",
    )

    return {
        "schema_version":
            "k0-rvg-softplus-operating-point-summary-v1",
        "scientific_question":
            QUESTION,
        "item_count":
            EXPECTED_ITEM_COUNT,
        "pair_role_count":
            EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count":
            len(cohort),
        "trajectory_row_count":
            len(rows),
        "primary_layer":
            PRIMARY_LAYER,
        "relative_coordinates":
            list(RELATIVE_COORDINATES),
        "execution_protocol":
            EXECUTION_PROTOCOL,
        "weighting_definition":
            "per-channel weight = sum_state(P_Z^2)",
        "full_336_trajectory":
            aggregate_trajectory(rows),
        "common_330_ddsssss_trajectory":
            aggregate_trajectory(common_rows),
        "max_composition_relative_residual":
            max(
                float(
                    r["composition_relative_residual"]
                )
                for r in rows
            ),
        "source_snapshot_dtype":
            "torch.float32",
        "metric_accumulation_dtype":
            "torch.float64",
        "raw_vectors_persisted":
            False,
        "tokenizer_invoked":
            False,
        "logits_read":
            False,
        "task_heads_executed":
            False,
        "training_executed":
            False,
        "causal_intervention_executed":
            False,
        "pca_probe_or_learned_geometry_executed":
            False,
    }


def validate_parent_reproduction(
    summary,
    parent_summary,
):
    current = summary[
        "common_330_ddsssss_trajectory"
    ]

    parent = parent_summary[
        "common_330_ddsssss_trajectory"
    ]

    fields = (
        "p_z_l2_over_k0_w",
        "q_t_l2_over_k0_w",
        "effective_softplus_gain",
    )

    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            for field in fields:
                got = float(
                    current[role][str(k)]
                    [field]["median"]
                )

                expected = float(
                    parent[role][str(k)]
                    [field]["median"]
                )

                require(
                    math.isclose(
                        got,
                        expected,
                        rel_tol=1e-13,
                        abs_tol=1e-13,
                    ),
                    (
                        "PARENT_REPRODUCTION_FAILURE:"
                        f"{role}:{k}:{field}:"
                        f"{got}:{expected}"
                    ),
                )


def json_bytes(obj):
    return (
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows):
    return b"".join(
        json_bytes(r)
        for r in rows
    )


def execute(
    root,
    parent,
    base,
    repo,
    parent_summary,
    plan,
    cohort,
    signatures,
    handoff_path,
    output_dir,
):
    final_dir = output_dir.resolve()
    partial_dir = Path(
        str(final_dir) + ".partial"
    )

    require(
        handoff_path.is_file(),
        f"HANDOFF_MISSING:{handoff_path}",
    )

    require(
        not final_dir.exists(),
        f"OUTPUT_DIR_EXISTS:{final_dir}",
    )

    require(
        not partial_dir.exists(),
        f"PARTIAL_OUTPUT_EXISTS:{partial_dir}",
    )

    (
        observer,
        k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        snapshot_info,
    ) = base.resolve_runtime(
        root,
        handoff_path,
    )

    require(
        binding.source_sha256
        == EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    rows = []
    forward_count = 0

    for n, row in enumerate(
        plan,
        start=1,
    ):
        idx = int(
            row["local_template_index"]
        )
        role = str(row["role"])
        anchor = int(row["anchor"])

        cutoff = (
            anchor
            + POST_HORIZON
            + 1
        )

        matched_prefix = tuple(
            row["matched_ids"][:cutoff]
        )

        swapped_prefix = tuple(
            row["swapped_ids"][:cutoff]
        )

        require(
            len(matched_prefix) == cutoff
            and len(swapped_prefix) == cutoff,
            f"PREFIX_LENGTH_FAILURE:{idx}:{role}",
        )

        matched = parent.capture_factors(
            base,
            model,
            binding,
            layer_map,
            matched_prefix,
            row["targets"],
        )
        forward_count += 1

        swapped = parent.capture_factors(
            base,
            model,
            binding,
            layer_map,
            swapped_prefix,
            row["targets"],
        )
        forward_count += 1

        rows.extend(
            metric_rows_for_pair(
                row,
                matched,
                swapped,
                cohort,
                signatures[(idx, role)],
            )
        )

        if (
            n % 16 == 0
            or n == len(plan)
        ):
            print(
                f"PROGRESS "
                f"pair_roles={n}/{len(plan)} "
                f"model_forwards={forward_count}",
                flush=True,
            )

    require(
        forward_count == EXPECTED_FORWARD_COUNT,
        "FORWARD_COUNT_MISMATCH",
    )

    summary = make_summary(
        rows,
        cohort,
    )

    validate_parent_reproduction(
        summary,
        parent_summary,
    )

    partial_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metrics_path = (
        partial_dir
        / "softplus_operating_point_metrics.jsonl"
    )

    summary_path = (
        partial_dir
        / "summary.json"
    )

    manifest_path = (
        partial_dir
        / "execution_manifest.json"
    )

    metrics_path.write_bytes(
        jsonl_bytes(rows)
    )

    summary_path.write_bytes(
        json_bytes(summary)
    )

    manifest = {
        "schema_version":
            "k0-rvg-softplus-operating-point-execution-manifest-v1",
        "runtime_git_head":
            repo["head"],
        "runtime_branch":
            repo["branch"],
        "parent_freeze_commit":
            PARENT_FREEZE_COMMIT,
        "scientific_question":
            QUESTION,
        "execution_protocol":
            EXECUTION_PROTOCOL,
        "weighting_definition":
            "per-channel weight = sum_state(P_Z^2)",
        "item_count":
            EXPECTED_ITEM_COUNT,
        "pair_role_count":
            EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count":
            EXPECTED_COMMON_COUNT,
        "model_forward_count":
            forward_count,
        "primary_layer":
            PRIMARY_LAYER,
        "relative_coordinates":
            list(RELATIVE_COORDINATES),
        "mamba_source_sha256":
            binding.source_sha256,
        "handoff_zip_sha256":
            handoff["zip_sha256"],
        "checkpoint_sha256":
            handoff["checkpoint_sha256"],
        "encoder_canonical_digest":
            encoder["canonical_digest"],
        "encoder_raw_concat_digest":
            encoder["raw_concat_digest"],
        "parent_softplus_summary_match":
            True,
        "source_snapshot_dtype":
            "torch.float32",
        "metric_accumulation_dtype":
            "torch.float64",
        "raw_vectors_persisted":
            False,
        "scientific_model_forward_executed":
            True,
        "scientific_operating_point_read":
            True,
        "tokenizer_invoked":
            False,
        "logits_read":
            False,
        "task_heads_executed":
            False,
        "training_executed":
            False,
        "causal_intervention_executed":
            False,
        "pca_probe_or_learned_geometry_executed":
            False,
        "runner_rel":
            Path(__file__)
            .resolve()
            .relative_to(root)
            .as_posix(),
        "runner_sha256":
            sha256_bytes(
                Path(__file__).read_bytes()
            ),
        "outputs": {
            "softplus_operating_point_metrics.jsonl":
                sha256_bytes(
                    metrics_path.read_bytes()
                ),
            "summary.json":
                sha256_bytes(
                    summary_path.read_bytes()
                ),
        },
    }

    manifest_path.write_bytes(
        json_bytes(manifest)
    )

    os.replace(
        partial_dir,
        final_dir,
    )

    common = summary[
        "common_330_ddsssss_trajectory"
    ]

    print(
        "PASS_SOFTPLUS_OPERATING_POINT_EXECUTION"
    )

    print(
        "output_dir =",
        final_dir,
    )

    print(
        "model_forward_count =",
        forward_count,
    )

    print(
        "parent_softplus_summary_match = True"
    )

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[role][str(k)]

            print(
                f"{role}_k{k}_median "
                f"Zmid_UW="
                f"{t['z_mid_unweighted_mean']['median']} "
                f"Zmid_EW="
                f"{t['z_mid_pz_energy_mean']['median']} "
                f"G_UW="
                f"{t['gain_unweighted_mean']['median']} "
                f"G_EW="
                f"{t['gain_pz_energy_mean']['median']} "
                f"G_eff="
                f"{t['effective_softplus_gain']['median']} "
                f"G_mid_rms="
                f"{t['midpoint_gain_energy_rms']['median']} "
                f"secant_mid_gap="
                f"{t['secant_midpoint_gap_rms']['median']} "
                f"E_g75="
                f"{t['pz_energy_fraction_gain_ge_075']['median']} "
                f"E_g90="
                f"{t['pz_energy_fraction_gain_ge_090']['median']}"
            )


def main():
    global parent_torch_equal

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--static-preflight",
        action="store_true",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    parser.add_argument(
        "--handoff",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    args = parser.parse_args()

    require(
        args.static_preflight
        ^ args.execute,
        "SELECT_EXACTLY_ONE_MODE",
    )

    root = Path.cwd().resolve()

    parent = load_parent_runner(root)

    parent_torch_equal = parent.torch_equal

    (
        discrete_b,
        write,
        carry,
        magnitude,
        base,
        plan,
        cohort,
        signatures,
    ) = build_plan(
        root,
        parent,
    )

    repo, parent_summary = authenticate_parent(
        root,
        base,
    )

    source = parent.validate_frozen_source_semantics()

    require(
        source["source_sha256"]
        == EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    print(
        "=== SOFTPLUS OPERATING-POINT AUDIT PLAN ==="
    )

    print(
        "branch =",
        repo["branch"],
    )

    print(
        "head =",
        repo["head"],
    )

    print(
        "parent_freeze_commit =",
        PARENT_FREEZE_COMMIT,
    )

    print(
        "item_count =",
        EXPECTED_ITEM_COUNT,
    )

    print(
        "pair_role_count =",
        len(plan),
    )

    print(
        "common_ddsssss_item_count =",
        len(cohort),
    )

    print(
        "primary_layer =",
        PRIMARY_LAYER,
    )

    print(
        "relative_coordinates =",
        list(RELATIVE_COORDINATES),
    )

    print(
        "scientific_question =",
        QUESTION,
    )

    print(
        "weighting_definition = "
        "per-channel weight = sum_state(P_Z^2)"
    )

    print(
        "mamba_source_sha256 =",
        source["source_sha256"],
    )

    print(
        "pre_softplus_trace_line =",
        source["pre_softplus_line"],
    )

    print(
        "parent_softplus_summary_preserved = True"
    )

    print(
        "raw_vectors_persisted = False"
    )

    print(
        "tokenizer_invoked = False"
    )

    print(
        "scientific_model_forward_executed = False"
    )

    if args.static_preflight:
        print(
            "PASS_SOFTPLUS_OPERATING_POINT_STATIC_PREFLIGHT"
        )
        return 0

    require(
        args.handoff is not None,
        "EXECUTE_REQUIRES_HANDOFF",
    )

    require(
        args.output_dir is not None,
        "EXECUTE_REQUIRES_OUTPUT_DIR",
    )

    execute(
        root,
        parent,
        base,
        repo,
        parent_summary,
        plan,
        cohort,
        signatures,
        args.handoff.resolve(),
        args.output_dir,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OperatingPointAuditError as exc:
        print(
            f"BLOCKED: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception as exc:
        print(
            f"BLOCKED_UNEXPECTED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise