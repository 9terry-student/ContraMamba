"""K0-RVG Ubar x delta-Z channel-selection audit.

Frozen parent result:

    P_Z[j,s] = mean(U_j) * mean(B_s) * delta_Z_j

so per-channel P_Z energy is:

    w_parent[j]
      = sum_s P_Z[j,s]^2
      = mean(U_j)^2 * delta_Z_j^2 * sum_s mean(B_s)^2

Within one pair/token, the B-norm factor is common to every intermediate
channel. Therefore the normalized parent channel distribution is exactly:

    p_product(j) proportional to mean(U_j)^2 * delta_Z_j^2

This audit compares three observed channel-weighting distributions:

    p_dz2(j)     proportional to delta_Z_j^2
    p_u2(j)      proportional to mean(U_j)^2
    p_product(j) proportional to mean(U_j)^2 * delta_Z_j^2

against the same frozen softplus secant gain and absolute Z midpoint.

The purpose is to determine whether the validated corr-k2 high-transmission
concentration is already present in delta-Z channel selection, is introduced
or sharpened by mean-U weighting, or depends on their multiplicative
alignment.

These are algebraic/observational reweighting diagnostics, not causal
interventions. No raw vectors are persisted.
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


PARENT_FREEZE_COMMIT = "e33e0edfdaad847a11fc4239f6252828f278030b"

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_softplus_operating_point_0a291ef_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "21ff7ef3a27d4635c2a15a8df24bfbdf63cbd1133699503982a606513a3ce718"
)

PARENT_RUNNER_REL = (
    "scripts/longterm_k0_rvg_softplus_operating_point_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "95bd7bac07341dc45be631109cc68bbef15b5da49e74d077d35fb22009f763d8"
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
    "Is corr-k2 high-softplus-transmission channel concentration already "
    "present in delta-Z squared channel selection, introduced or sharpened "
    "by mean-U squared reweighting, or dependent on their multiplicative "
    "alignment?"
)

GAIN_TOL = 1e-6
IDENTITY_TOL = 1e-12


class ChannelSelectionAuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ChannelSelectionAuditError(message)


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
        "k0_rvg_operating_point_parent",
    )


def build_plan(root: Path, parent: Any):
    secant = parent.load_parent_runner(root)

    (
        discrete_b,
        write,
        carry,
        magnitude,
        base,
        plan,
        cohort,
        signatures,
    ) = parent.build_plan(
        root,
        secant,
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
        secant,
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
        == "k0-rvg-softplus-operating-point-summary-v1",
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


def weighted_fraction(mask, weights) -> float:
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


def effective_channel_count(weights) -> float:
    import torch

    total = torch.sum(weights)

    if float(total.item()) == 0.0:
        return 0.0

    p = weights / total

    denom = torch.sum(p * p)

    return float(
        (1.0 / denom).item()
    )


def nonnegative_cosine(a, b) -> float:
    import torch

    denom = (
        torch.linalg.vector_norm(a)
        * torch.linalg.vector_norm(b)
    )

    if float(denom.item()) == 0.0:
        return 0.0

    return float(
        (
            torch.sum(a * b)
            / denom
        ).item()
    )


def multiplicative_enrichment(a, b) -> float:
    import torch

    ma = torch.mean(a)
    mb = torch.mean(b)

    denom = ma * mb

    if float(denom.item()) == 0.0:
        return 0.0

    return float(
        (
            torch.mean(a * b)
            / denom
        ).item()
    )


def distribution_metrics(
    gain,
    z_mid,
    weights,
):
    return {
        "z_mid_mean":
            weighted_mean(
                z_mid,
                weights,
            ),
        "gain_mean":
            weighted_mean(
                gain,
                weights,
            ),
        "gain_rms":
            weighted_rms(
                gain,
                weights,
            ),
        "fraction_gain_ge_075":
            weighted_fraction(
                gain >= 0.75,
                weights,
            ),
        "fraction_gain_ge_090":
            weighted_fraction(
                gain >= 0.90,
                weights,
            ),
        "fraction_z_mid_ge_zero":
            weighted_fraction(
                z_mid >= 0.0,
                weights,
            ),
        "effective_channel_count":
            effective_channel_count(
                weights,
            ),
    }


def metric_rows_for_pair(
    secant: Any,
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

    out = []

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

        u_equal = secant.torch_equal(
            Um32,
            Us32,
        )
        z_equal = secant.torch_equal(
            Zm32,
            Zs32,
        )
        t_equal = secant.torch_equal(
            Tm32,
            Ts32,
        )
        b_equal = secant.torch_equal(
            Bm32,
            Bs32,
        )
        w_equal = secant.torch_equal(
            Wm32,
            Ws32,
        )

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

        Us = Us32.to(torch.float64)
        Zs = Zs32.to(torch.float64)
        Ts = Ts32.to(torch.float64)
        Bs = Bs32.to(torch.float64)

        mean_U = 0.5 * (
            Um + Us
        )

        mean_B = 0.5 * (
            Bm + Bs
        )

        delta_Z = Zm - Zs
        delta_T = Tm - Ts
        z_mid = 0.5 * (
            Zm + Zs
        )

        zero = delta_Z == 0
        nonzero = ~zero

        gain = torch.empty_like(
            delta_Z
        )

        gain[nonzero] = (
            delta_T[nonzero]
            / delta_Z[nonzero]
        )

        if bool(zero.any().item()):
            gain[zero] = torch.sigmoid(
                z_mid[zero]
            )

        require(
            bool(
                torch.isfinite(
                    gain
                ).all().item()
            ),
            f"NONFINITE_GAIN:{idx}:{role}:{k}",
        )

        require(
            float(gain.min().item())
            >= -GAIN_TOL,
            f"GAIN_BELOW_ZERO:{idx}:{role}:{k}",
        )

        require(
            float(gain.max().item())
            <= 1.0 + GAIN_TOL,
            f"GAIN_ABOVE_ONE:{idx}:{role}:{k}",
        )

        u = mean_U.squeeze(0)
        dz = delta_Z.squeeze(0)
        g = gain.squeeze(0)
        zm = z_mid.squeeze(0)

        u2 = u * u
        dz2 = dz * dz
        product = u2 * dz2

        dz_metrics = distribution_metrics(
            g,
            zm,
            dz2,
        )

        u_metrics = distribution_metrics(
            g,
            zm,
            u2,
        )

        product_metrics = distribution_metrics(
            g,
            zm,
            product,
        )

        # Exact parent P_Z/Q_T effective gain identity.
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

        p_z_l2 = float(
            torch.linalg.vector_norm(
                p_z
            ).item()
        )

        q_t_l2 = float(
            torch.linalg.vector_norm(
                q_t
            ).item()
        )

        exact_parent_gain = (
            q_t_l2 / p_z_l2
            if p_z_l2 > 0.0
            else 0.0
        )

        require(
            math.isclose(
                product_metrics["gain_rms"],
                exact_parent_gain,
                rel_tol=IDENTITY_TOL,
                abs_tol=IDENTITY_TOL,
            ),
            (
                "PRODUCT_PARENT_GAIN_IDENTITY_FAILURE:"
                f"{idx}:{role}:{k}:"
                f"{product_metrics['gain_rms']}:"
                f"{exact_parent_gain}"
            ),
        )

        values = {
            "schema_version":
                "k0-rvg-uz-channel-selection-row-v1",
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
            "effective_softplus_gain":
                exact_parent_gain,
            "u2_dz2_cosine":
                nonnegative_cosine(
                    u2,
                    dz2,
                ),
            "u2_dz2_multiplicative_enrichment":
                multiplicative_enrichment(
                    u2,
                    dz2,
                ),
            "source_snapshot_dtype":
                "torch.float32",
            "metric_accumulation_dtype":
                "torch.float64",
        }

        for prefix, metrics in (
            ("dz2", dz_metrics),
            ("u2", u_metrics),
            ("product", product_metrics),
        ):
            for name, value in metrics.items():
                values[
                    f"{prefix}_{name}"
                ] = value

        # Directional reweighting contrasts.
        #
        # product - dz2:
        # observed change when U^2 reweights the observed delta-Z^2
        # channel distribution.
        #
        # product - u2:
        # observed change when delta-Z^2 reweights the observed U^2
        # channel distribution.
        values["u_reweight_gain_rms_delta"] = (
            product_metrics["gain_rms"]
            - dz_metrics["gain_rms"]
        )

        values["dz_reweight_gain_rms_delta"] = (
            product_metrics["gain_rms"]
            - u_metrics["gain_rms"]
        )

        values["joint_excess_gain_rms"] = (
            product_metrics["gain_rms"]
            - max(
                dz_metrics["gain_rms"],
                u_metrics["gain_rms"],
            )
        )

        values["u_reweight_gain090_fraction_delta"] = (
            product_metrics[
                "fraction_gain_ge_090"
            ]
            - dz_metrics[
                "fraction_gain_ge_090"
            ]
        )

        values["dz_reweight_gain090_fraction_delta"] = (
            product_metrics[
                "fraction_gain_ge_090"
            ]
            - u_metrics[
                "fraction_gain_ge_090"
            ]
        )

        values["joint_excess_gain090_fraction"] = (
            product_metrics[
                "fraction_gain_ge_090"
            ]
            - max(
                dz_metrics[
                    "fraction_gain_ge_090"
                ],
                u_metrics[
                    "fraction_gain_ge_090"
                ],
            )
        )

        require(
            all(
                math.isfinite(
                    float(v)
                )
                for key, v in values.items()
                if isinstance(
                    v,
                    (int, float),
                )
                and not isinstance(
                    v,
                    bool,
                )
            ),
            f"NONFINITE_METRIC:{idx}:{role}:{k}",
        )

        out.append(values)

    return out


SUMMARY_FIELDS = (
    "p_z_l2",
    "q_t_l2",
    "effective_softplus_gain",

    "dz2_z_mid_mean",
    "dz2_gain_mean",
    "dz2_gain_rms",
    "dz2_fraction_gain_ge_075",
    "dz2_fraction_gain_ge_090",
    "dz2_fraction_z_mid_ge_zero",
    "dz2_effective_channel_count",

    "u2_z_mid_mean",
    "u2_gain_mean",
    "u2_gain_rms",
    "u2_fraction_gain_ge_075",
    "u2_fraction_gain_ge_090",
    "u2_fraction_z_mid_ge_zero",
    "u2_effective_channel_count",

    "product_z_mid_mean",
    "product_gain_mean",
    "product_gain_rms",
    "product_fraction_gain_ge_075",
    "product_fraction_gain_ge_090",
    "product_fraction_z_mid_ge_zero",
    "product_effective_channel_count",

    "u2_dz2_cosine",
    "u2_dz2_multiplicative_enrichment",

    "u_reweight_gain_rms_delta",
    "dz_reweight_gain_rms_delta",
    "joint_excess_gain_rms",

    "u_reweight_gain090_fraction_delta",
    "dz_reweight_gain090_fraction_delta",
    "joint_excess_gain090_fraction",
)


def aggregate(values):
    vals = [
        float(v)
        for v in values
    ]

    require(
        bool(vals),
        "EMPTY_AGGREGATE",
    )

    require(
        all(
            math.isfinite(v)
            for v in vals
        ),
        "NONFINITE_AGGREGATE",
    )

    return {
        "count":
            len(vals),
        "mean":
            float(
                statistics.fmean(
                    vals
                )
            ),
        "median":
            float(
                statistics.median(
                    vals
                )
            ),
        "min":
            float(min(vals)),
        "max":
            float(max(vals)),
    }


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
                and int(
                    r[
                        "relative_coordinate"
                    ]
                ) == k
            ]

            require(
                bool(bucket),
                f"EMPTY_BUCKET:{role}:{k}",
            )

            out[role][str(k)] = {
                field:
                    aggregate(
                        float(r[field])
                        for r in bucket
                    )
                for field
                in SUMMARY_FIELDS
            }

    return out


def make_summary(
    rows,
    cohort,
):
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
            r[
                "in_common_ddsssss_cohort"
            ]
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
            "k0-rvg-uz-channel-selection-summary-v1",
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
            list(
                RELATIVE_COORDINATES
            ),
        "execution_protocol":
            EXECUTION_PROTOCOL,
        "channel_weight_definitions": {
            "dz2":
                "delta_Z_j^2",
            "u2":
                "mean(U_j)^2",
            "product":
                "mean(U_j)^2 * delta_Z_j^2",
        },
        "parent_distribution_identity":
            (
                "normalized P_Z channel energy "
                "equals normalized product weights"
            ),
        "full_336_trajectory":
            aggregate_trajectory(
                rows
            ),
        "common_330_ddsssss_trajectory":
            aggregate_trajectory(
                common_rows
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

    mapping = {
        "product_z_mid_mean":
            "z_mid_pz_energy_mean",
        "product_gain_mean":
            "gain_pz_energy_mean",
        "product_gain_rms":
            "effective_softplus_gain",
        "product_fraction_gain_ge_075":
            "pz_energy_fraction_gain_ge_075",
        "product_fraction_gain_ge_090":
            "pz_energy_fraction_gain_ge_090",
    }

    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            for current_field, parent_field in mapping.items():
                got = float(
                    current[role][str(k)]
                    [current_field]["median"]
                )

                expected = float(
                    parent[role][str(k)]
                    [parent_field]["median"]
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
                        f"{role}:{k}:"
                        f"{current_field}:"
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
    secant,
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
        str(final_dir)
        + ".partial"
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
            row[
                "local_template_index"
            ]
        )

        role = str(
            row["role"]
        )

        anchor = int(
            row["anchor"]
        )

        cutoff = (
            anchor
            + POST_HORIZON
            + 1
        )

        matched_prefix = tuple(
            row[
                "matched_ids"
            ][:cutoff]
        )

        swapped_prefix = tuple(
            row[
                "swapped_ids"
            ][:cutoff]
        )

        require(
            len(matched_prefix)
            == cutoff
            and len(swapped_prefix)
            == cutoff,
            (
                "PREFIX_LENGTH_FAILURE:"
                f"{idx}:{role}"
            ),
        )

        matched = secant.capture_factors(
            base,
            model,
            binding,
            layer_map,
            matched_prefix,
            row["targets"],
        )

        forward_count += 1

        swapped = secant.capture_factors(
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
                secant,
                row,
                matched,
                swapped,
                cohort,
                signatures[
                    (idx, role)
                ],
            )
        )

        if (
            n % 16 == 0
            or n == len(plan)
        ):
            print(
                "PROGRESS "
                f"pair_roles={n}/{len(plan)} "
                f"model_forwards={forward_count}",
                flush=True,
            )

    require(
        forward_count
        == EXPECTED_FORWARD_COUNT,
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
        / "uz_channel_selection_metrics.jsonl"
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
            "k0-rvg-uz-channel-selection-execution-manifest-v1",
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
        "channel_weight_definitions": {
            "dz2":
                "delta_Z_j^2",
            "u2":
                "mean(U_j)^2",
            "product":
                "mean(U_j)^2 * delta_Z_j^2",
        },
        "parent_distribution_identity":
            (
                "normalized P_Z channel energy "
                "equals normalized product weights"
            ),
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
            list(
                RELATIVE_COORDINATES
            ),
        "mamba_source_sha256":
            binding.source_sha256,
        "handoff_zip_sha256":
            handoff["zip_sha256"],
        "checkpoint_sha256":
            handoff["checkpoint_sha256"],
        "encoder_canonical_digest":
            encoder[
                "canonical_digest"
            ],
        "encoder_raw_concat_digest":
            encoder[
                "raw_concat_digest"
            ],
        "parent_operating_point_summary_match":
            True,
        "source_snapshot_dtype":
            "torch.float32",
        "metric_accumulation_dtype":
            "torch.float64",
        "raw_vectors_persisted":
            False,
        "scientific_model_forward_executed":
            True,
        "scientific_channel_selection_read":
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
                Path(__file__)
                .read_bytes()
            ),
        "outputs": {
            "uz_channel_selection_metrics.jsonl":
                sha256_bytes(
                    metrics_path
                    .read_bytes()
                ),
            "summary.json":
                sha256_bytes(
                    summary_path
                    .read_bytes()
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
        "PASS_UZ_CHANNEL_SELECTION_EXECUTION"
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
        "parent_operating_point_summary_match = True"
    )

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[
                role
            ][str(k)]

            print(
                f"{role}_k{k}_median "
                f"DZ_G="
                f"{t['dz2_gain_rms']['median']} "
                f"U_G="
                f"{t['u2_gain_rms']['median']} "
                f"PROD_G="
                f"{t['product_gain_rms']['median']} "
                f"U_reweight="
                f"{t['u_reweight_gain_rms_delta']['median']} "
                f"DZ_reweight="
                f"{t['dz_reweight_gain_rms_delta']['median']} "
                f"joint_excess="
                f"{t['joint_excess_gain_rms']['median']} "
                f"DZ_E90="
                f"{t['dz2_fraction_gain_ge_090']['median']} "
                f"U_E90="
                f"{t['u2_fraction_gain_ge_090']['median']} "
                f"PROD_E90="
                f"{t['product_fraction_gain_ge_090']['median']} "
                f"cos_U2_DZ2="
                f"{t['u2_dz2_cosine']['median']} "
                f"enrichment="
                f"{t['u2_dz2_multiplicative_enrichment']['median']}"
            )


def main():
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

    parent = load_parent_runner(
        root
    )

    (
        secant,
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

    source = secant.validate_frozen_source_semantics()

    require(
        source["source_sha256"]
        == EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    print(
        "=== UBAR x DELTA-Z CHANNEL-SELECTION AUDIT PLAN ==="
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
        list(
            RELATIVE_COORDINATES
        ),
    )

    print(
        "scientific_question =",
        QUESTION,
    )

    print(
        "dz2_weight = delta_Z_j^2"
    )

    print(
        "u2_weight = mean(U_j)^2"
    )

    print(
        "product_weight = mean(U_j)^2 * delta_Z_j^2"
    )

    print(
        "parent_distribution_identity = "
        "normalized P_Z channel energy equals normalized product weights"
    )

    print(
        "mamba_source_sha256 =",
        source["source_sha256"],
    )

    print(
        "pre_softplus_trace_line =",
        source[
            "pre_softplus_line"
        ],
    )

    print(
        "parent_operating_point_summary_preserved = True"
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
            "PASS_UZ_CHANNEL_SELECTION_STATIC_PREFLIGHT"
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
        secant,
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
        raise SystemExit(
            main()
        )
    except ChannelSelectionAuditError as exc:
        print(
            f"BLOCKED: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception as exc:
        print(
            "BLOCKED_UNEXPECTED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise