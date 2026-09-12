"""K0-RVG exact four-tap depthwise-convolution decomposition audit.

Authenticated layer-23 difference identity:

    Q_l(t) = K_(3-l) * delta_H_(t-l)
    delta_C_t = Q_0 + Q_1 + Q_2 + Q_3

where * is channelwise multiplication.

Because Conv1d bias is shared between matched and swapped branches, bias
cancels exactly in the difference.

Squared-norm identity:

    ||delta_C||^2
      = sum_l ||Q_l||^2
        + 2 sum_(l<m) <Q_l, Q_m>

The audit separates:

1. incoming lag-specific delta-H magnitude;
2. fixed-tap/channel-conditioned transfer into Q_l;
3. constructive/destructive vector addition among tap contributions.

Observational/algebraic only. No intervention, training, tokenizer execution,
logits, task heads, PCA, or learned probe.
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


PARENT_FREEZE_COMMIT = (
    "77276e1007e797d00dcf4f355c9a7c99d5f19af2"
)

PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_postconv_u_factorization_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "0f3e9527ff3d5a3c91e646547b1ab17e134e799130833100446bd976e4a4d052"
)

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_postconv_u_rf_factorization_fcc378e_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "76a6efde74276b3e26fdbc7e026c2443b3067bc85bb21e40bb0ce1436d0e88d6"
)

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FORWARD_COUNT = 1344

PRIMARY_LAYER = 23
INTERMEDIATE_SIZE = 1536
CONV_KERNEL_SIZE = 4

RELATIVE_COORDINATES = tuple(range(-1, 7))
POST_HORIZON = 6

EXECUTION_PROTOCOL = (
    "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"
)

CONV_RECON_REL_TOL = 2e-5

QUESTION = (
    "Is the layer-23 depthwise-convolution transfer pattern explained "
    "primarily by transport of delta-H across fixed causal taps of sharply "
    "different magnitude, by channel-conditioned transfer within those "
    "fixed taps, or by constructive/destructive interaction among the "
    "four tap contributions?"
)


class FourTapAuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FourTapAuditError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def import_module(path: Path, name: str):
    require(
        path.is_file(),
        f"MODULE_MISSING:{path}",
    )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"MODULE_SPEC_FAILURE:{path}",
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def load_parent_runner(root: Path):
    path = root / PARENT_RUNNER_REL

    require(
        sha256_bytes(path.read_bytes())
        == PARENT_RUNNER_SHA256,
        "PARENT_RUNNER_SHA256_MISMATCH",
    )

    return import_module(
        path,
        "k0_rvg_postconv_u_parent",
    )


def build_plan(root: Path, parent: Any):
    time_step = parent.load_parent_runner(
        root
    )

    (
        dt_projection,
        operating,
        secant,
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
        time_step,
    )

    require(
        len(plan)
        == EXPECTED_PAIR_ROLE_COUNT,
        "PLAN_COUNT_MISMATCH",
    )

    require(
        len(cohort)
        == EXPECTED_COMMON_COUNT,
        "COMMON_COHORT_COUNT_MISMATCH",
    )

    return (
        time_step,
        dt_projection,
        operating,
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


def authenticate_parent(
    root: Path,
    parent: Any,
    time_step: Any,
    base: Any,
):
    # Authenticate the complete upstream chain.
    parent.authenticate_parent(
        root,
        time_step,
        base,
    )

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
        sha256_bytes(raw)
        == PARENT_SUMMARY_SHA256,
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
        == "k0-rvg-postconv-u-factorization-summary-v1",
        "PARENT_SCHEMA_MISMATCH",
    )

    require(
        summary.get("item_count")
        == EXPECTED_ITEM_COUNT,
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

    return repo, summary


def resolve_kernel(
    parent: Any,
    model: Any,
    layer_map: Mapping[int, int],
):
    import torch

    mixer = parent.resolve_primary_mixer(
        model,
        layer_map,
    )

    weight = (
        mixer.conv1d.weight
        .detach()
        .cpu()
        .to(torch.float64)
        [:, 0, :]
        .contiguous()
    )

    require(
        tuple(weight.shape)
        == (
            INTERMEDIATE_SIZE,
            CONV_KERNEL_SIZE,
        ),
        "CONV_WEIGHT_SHAPE_MISMATCH",
    )

    rms = []

    for lag in range(CONV_KERNEL_SIZE):
        kernel_index = (
            CONV_KERNEL_SIZE
            - 1
            - lag
        )

        w = weight[:, kernel_index]

        rms.append(
            float(
                torch.sqrt(
                    torch.mean(w * w)
                ).item()
            )
        )

    # Frozen runtime structure observed during source authentication.
    require(
        rms[0] > 0.0,
        "LAG0_WEIGHT_RMS_NONPOSITIVE",
    )
    require(
        rms[1] > 0.0,
        "LAG1_WEIGHT_RMS_NONPOSITIVE",
    )
    require(
        rms[2] > 0.0,
        "LAG2_WEIGHT_RMS_NONPOSITIVE",
    )
    require(
        rms[3] == 0.0,
        "LAG3_WEIGHT_NOT_EXACT_ZERO",
    )

    return mixer, weight, tuple(rms)


def capture(
    parent: Any,
    dt_projection: Any,
    secant: Any,
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    mixer: Any,
    token_ids,
    targets,
):
    return parent.capture_hcu_factors(
        dt_projection,
        secant,
        base,
        model,
        binding,
        layer_map,
        mixer,
        token_ids,
        targets,
    )


def metric_rows_for_pair(
    row: Mapping[str, Any],
    matched: Mapping[int, Mapping[str, Any]],
    swapped: Mapping[int, Mapping[str, Any]],
    cohort: frozenset[int],
    signature: str,
    weight: Any,
    weight_rms: tuple[float, ...],
):
    import torch

    idx = int(
        row["local_template_index"]
    )

    role = str(
        row["role"]
    )

    anchor = int(
        row["anchor"]
    )

    output = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k

        hm = matched[token]["H_RF"].to(
            torch.float64
        )

        hs = swapped[token]["H_RF"].to(
            torch.float64
        )

        cm = matched[token]["C"].to(
            torch.float64
        )

        cs = swapped[token]["C"].to(
            torch.float64
        )

        require(
            tuple(hm.shape)
            == (
                CONV_KERNEL_SIZE,
                INTERMEDIATE_SIZE,
            ),
            f"HM_RF_SHAPE_MISMATCH:{idx}:{role}:{k}",
        )

        require(
            tuple(hs.shape)
            == (
                CONV_KERNEL_SIZE,
                INTERMEDIATE_SIZE,
            ),
            f"HS_RF_SHAPE_MISMATCH:{idx}:{role}:{k}",
        )

        dh = hm - hs
        dc = cm - cs

        q = []
        dh_norms = []
        q_norms = []
        tap_transfers = []
        normalized_transfers = []

        for lag in range(
            CONV_KERNEL_SIZE
        ):
            kernel_index = (
                CONV_KERNEL_SIZE
                - 1
                - lag
            )

            dhl = dh[lag, :]

            ql = (
                dhl
                * weight[:, kernel_index]
            )

            dhn = float(
                torch.linalg.vector_norm(
                    dhl
                ).item()
            )

            qn = float(
                torch.linalg.vector_norm(
                    ql
                ).item()
            )

            transfer = (
                qn / dhn
                if dhn > 0.0
                else 0.0
            )

            if (
                lag < 3
                and dhn > 0.0
            ):
                normalized = (
                    transfer
                    / weight_rms[lag]
                )
            else:
                normalized = 0.0

            q.append(ql)
            dh_norms.append(dhn)
            q_norms.append(qn)
            tap_transfers.append(
                transfer
            )
            normalized_transfers.append(
                normalized
            )

        # Exact structural consequence of zero lag-3 kernel.
        require(
            q_norms[3] == 0.0,
            f"LAG3_Q_NOT_ZERO:{idx}:{role}:{k}",
        )

        q_sum = (
            q[0]
            + q[1]
            + q[2]
            + q[3]
        )

        dc_l2 = float(
            torch.linalg.vector_norm(
                dc
            ).item()
        )

        recon_residual = (
            q_sum
            - dc.squeeze(0)
        )

        recon_rel = float(
            torch.linalg.vector_norm(
                recon_residual
            ).item()
            / max(
                dc_l2,
                1e-12,
            )
        )

        require(
            recon_rel
            <= CONV_RECON_REL_TOL,
            (
                "FOUR_TAP_RECONSTRUCTION_FAILURE:"
                f"{idx}:{role}:{k}:{recon_rel}"
            ),
        )

        rss_sq = float(
            sum(
                value * value
                for value in q_norms
            )
        )

        rss_l2 = math.sqrt(
            max(rss_sq, 0.0)
        )

        cross_terms = {}

        cross_total = 0.0

        for a in range(4):
            for b in range(a + 1, 4):
                value = float(
                    2.0
                    * torch.dot(
                        q[a],
                        q[b],
                    ).item()
                )

                cross_terms[
                    f"cross_{a}{b}"
                ] = value

                cross_total += value

        q_sum_l2 = float(
            torch.linalg.vector_norm(
                q_sum
            ).item()
        )

        squared_closure_error = abs(
            q_sum_l2 * q_sum_l2
            - (
                rss_sq
                + cross_total
            )
        )

        require(
            squared_closure_error
            <= 1e-12
            * max(
                q_sum_l2 * q_sum_l2,
                rss_sq,
                1.0,
            ),
            (
                "SQUARED_NORM_CLOSURE_FAILURE:"
                f"{idx}:{role}:{k}:"
                f"{squared_closure_error}"
            ),
        )

        addition_ratio = (
            dc_l2 / rss_l2
            if rss_l2 > 0.0
            else 0.0
        )

        cross_fraction = (
            cross_total / rss_sq
            if rss_sq > 0.0
            else 0.0
        )

        energy_fractions = [
            (
                qn * qn / rss_sq
                if rss_sq > 0.0
                else 0.0
            )
            for qn in q_norms
        ]

        dominant_q_lag = int(
            max(
                range(4),
                key=lambda lag:
                    q_norms[lag],
            )
        )

        values = {
            "schema_version":
                "k0-rvg-four-tap-convolution-row-v1",

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

            "delta_c_l2":
                dc_l2,

            "tap_rss_l2":
                rss_l2,

            "vector_addition_ratio":
                addition_ratio,

            "cross_total":
                cross_total,

            "cross_fraction_of_rss_sq":
                cross_fraction,

            "dominant_q_lag":
                dominant_q_lag,

            "four_tap_reconstruction_relative_residual":
                recon_rel,

            "squared_norm_closure_error":
                squared_closure_error,

            "source_snapshot_dtype":
                "torch.float32",

            "metric_accumulation_dtype":
                "torch.float64",
        }

        for lag in range(4):
            values[
                f"delta_h_lag{lag}_l2"
            ] = dh_norms[lag]

            values[
                f"q{lag}_l2"
            ] = q_norms[lag]

            values[
                f"q{lag}_tap_transfer"
            ] = tap_transfers[lag]

            values[
                f"q{lag}_energy_fraction"
            ] = energy_fractions[lag]

        # Weight-RMS-normalized transfer exists only for nonzero taps.
        for lag in range(3):
            values[
                f"q{lag}_transfer_over_weight_rms"
            ] = normalized_transfers[lag]

        values.update(
            cross_terms
        )

        require(
            all(
                math.isfinite(float(v))
                for v in values.values()
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
            ),
            f"NONFINITE_METRIC:{idx}:{role}:{k}",
        )

        output.append(values)

    return output


SUMMARY_FIELDS = (
    "delta_c_l2",
    "tap_rss_l2",
    "vector_addition_ratio",
    "cross_total",
    "cross_fraction_of_rss_sq",
    "four_tap_reconstruction_relative_residual",
    "squared_norm_closure_error",

    "delta_h_lag0_l2",
    "delta_h_lag1_l2",
    "delta_h_lag2_l2",
    "delta_h_lag3_l2",

    "q0_l2",
    "q1_l2",
    "q2_l2",
    "q3_l2",

    "q0_tap_transfer",
    "q1_tap_transfer",
    "q2_tap_transfer",
    "q3_tap_transfer",

    "q0_transfer_over_weight_rms",
    "q1_transfer_over_weight_rms",
    "q2_transfer_over_weight_rms",

    "q0_energy_fraction",
    "q1_energy_fraction",
    "q2_energy_fraction",
    "q3_energy_fraction",

    "cross_01",
    "cross_02",
    "cross_03",
    "cross_12",
    "cross_13",
    "cross_23",
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
                statistics.fmean(vals)
            ),

        "median":
            float(
                statistics.median(vals)
            ),

        "min":
            float(min(vals)),

        "max":
            float(max(vals)),
    }


def aggregate_trajectory(rows):
    result = {
        "corr": {},
        "ctrl": {},
    }

    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            bucket = [
                row
                for row in rows
                if row["role"] == role
                and int(
                    row[
                        "relative_coordinate"
                    ]
                ) == k
            ]

            require(
                bool(bucket),
                f"EMPTY_BUCKET:{role}:{k}",
            )

            result[role][str(k)] = {
                field: aggregate(
                    row[field]
                    for row in bucket
                )
                for field in SUMMARY_FIELDS
            }

    return result


def make_summary(
    rows,
    cohort,
    weight_rms,
):
    require(
        len(rows)
        == EXPECTED_PAIR_ROLE_COUNT
        * len(RELATIVE_COORDINATES),
        "ROW_COUNT_MISMATCH",
    )

    common_rows = [
        row
        for row in rows
        if bool(
            row[
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
            "k0-rvg-four-tap-convolution-summary-v1",

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

        "intermediate_size":
            INTERMEDIATE_SIZE,

        "conv_kernel_size":
            CONV_KERNEL_SIZE,

        "causal_lag_to_kernel_index":
            {
                "0": 3,
                "1": 2,
                "2": 1,
                "3": 0,
            },

        "kernel_weight_rms_by_causal_lag":
            {
                str(lag):
                    float(weight_rms[lag])
                for lag in range(4)
            },

        "lag3_kernel_exact_zero":
            True,

        "relative_coordinates":
            list(RELATIVE_COORDINATES),

        "execution_protocol":
            EXECUTION_PROTOCOL,

        "identity":
            (
                "delta_C_t = sum_l "
                "K_(3-l) elementwise delta_H_(t-l)"
            ),

        "full_336_trajectory":
            aggregate_trajectory(rows),

        "common_330_ddsssss_trajectory":
            aggregate_trajectory(
                common_rows
            ),

        "max_four_tap_reconstruction_relative_residual":
            max(
                float(
                    row[
                        "four_tap_reconstruction_relative_residual"
                    ]
                )
                for row in rows
            ),

        "max_squared_norm_closure_error":
            max(
                float(
                    row[
                        "squared_norm_closure_error"
                    ]
                )
                for row in rows
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
    for trajectory in (
        "full_336_trajectory",
        "common_330_ddsssss_trajectory",
    ):
        current = summary[trajectory]
        parent = parent_summary[trajectory]

        for role in ("corr", "ctrl"):
            for k in RELATIVE_COORDINATES:
                got = float(
                    current[role][str(k)]
                    ["delta_c_l2"]["median"]
                )

                expected = float(
                    parent[role][str(k)]
                    ["delta_c_l2"]["median"]
                )

                require(
                    math.isclose(
                        got,
                        expected,
                        rel_tol=1e-13,
                        abs_tol=1e-13,
                    ),
                    (
                        "PARENT_DELTA_C_REPRODUCTION_FAILURE:"
                        f"{trajectory}:{role}:{k}:"
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
        json_bytes(row)
        for row in rows
    )


def execute(
    root,
    parent,
    dt_projection,
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

    require(
        model.training is False,
        "MODEL_NOT_EVAL",
    )

    mixer, weight, weight_rms = (
        resolve_kernel(
            parent,
            model,
            layer_map,
        )
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
            row["matched_ids"][:cutoff]
        )

        swapped_prefix = tuple(
            row["swapped_ids"][:cutoff]
        )

        require(
            len(matched_prefix)
            == cutoff
            and len(swapped_prefix)
            == cutoff,
            f"PREFIX_LENGTH_FAILURE:{idx}:{role}",
        )

        matched = capture(
            parent,
            dt_projection,
            secant,
            base,
            model,
            binding,
            layer_map,
            mixer,
            matched_prefix,
            row["targets"],
        )

        forward_count += 1

        swapped = capture(
            parent,
            dt_projection,
            secant,
            base,
            model,
            binding,
            layer_map,
            mixer,
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
                signatures[
                    (idx, role)
                ],
                weight,
                weight_rms,
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
        weight_rms,
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
        / "four_tap_convolution_metrics.jsonl"
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
            "k0-rvg-four-tap-convolution-execution-manifest-v1",

        "runtime_git_head":
            repo["head"],

        "runtime_branch":
            repo["branch"],

        "parent_freeze_commit":
            PARENT_FREEZE_COMMIT,

        "scientific_question":
            QUESTION,

        "identity":
            (
                "delta_C_t = sum_l "
                "K_(3-l) elementwise delta_H_(t-l)"
            ),

        "causal_lag_to_kernel_index":
            {
                "0": 3,
                "1": 2,
                "2": 1,
                "3": 0,
            },

        "kernel_weight_rms_by_causal_lag":
            {
                str(lag):
                    float(weight_rms[lag])
                for lag in range(4)
            },

        "lag3_kernel_exact_zero":
            True,

        "execution_protocol":
            EXECUTION_PROTOCOL,

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

        "intermediate_size":
            INTERMEDIATE_SIZE,

        "conv_kernel_size":
            CONV_KERNEL_SIZE,

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

        "parent_delta_c_summary_match":
            True,

        "raw_vectors_persisted":
            False,

        "scientific_model_forward_executed":
            True,

        "scientific_preconv_receptive_field_read":
            True,

        "scientific_conv_preactivation_read":
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
            "four_tap_convolution_metrics.jsonl":
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
        "PASS_FOUR_TAP_CONVOLUTION_EXECUTION"
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
        "parent_delta_c_summary_match = True"
    )

    print(
        "kernel_weight_rms_by_causal_lag =",
        summary[
            "kernel_weight_rms_by_causal_lag"
        ],
    )

    print(
        "max_four_tap_reconstruction_relative_residual =",
        summary[
            "max_four_tap_reconstruction_relative_residual"
        ],
    )

    print(
        "max_squared_norm_closure_error =",
        summary[
            "max_squared_norm_closure_error"
        ],
    )

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[
                role
            ][str(k)]

            print(
                f"{role}_k{k}_median "
                f"DC={t['delta_c_l2']['median']} "
                f"Q0={t['q0_l2']['median']} "
                f"Q1={t['q1_l2']['median']} "
                f"Q2={t['q2_l2']['median']} "
                f"Q3={t['q3_l2']['median']} "
                f"RSS={t['tap_rss_l2']['median']} "
                f"ADD_RATIO={t['vector_addition_ratio']['median']} "
                f"CROSS_FRAC={t['cross_fraction_of_rss_sq']['median']}"
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
        time_step,
        dt_projection,
        operating,
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
        parent,
        time_step,
        base,
    )

    source = secant.validate_frozen_source_semantics()

    require(
        source["source_sha256"]
        == EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    print(
        "=== FOUR-TAP CONVOLUTION AUDIT PLAN ==="
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
        "scientific_question =",
        QUESTION,
    )

    print(
        "identity = "
        "delta_C_t = sum_l "
        "K_(3-l) elementwise delta_H_(t-l)"
    )

    print(
        "causal_lag_to_kernel_index = "
        "{0:3,1:2,2:1,3:0}"
    )

    print(
        "lag3_kernel_exact_zero_runtime_guard = True"
    )

    print(
        "mamba_source_sha256 =",
        source["source_sha256"],
    )

    print(
        "parent_delta_c_summary_preserved_by_design = True"
    )

    print(
        "raw_vectors_persisted = False"
    )

    print(
        "scientific_model_forward_executed = False"
    )

    print(
        "tokenizer_invoked = False"
    )

    if args.static_preflight:
        print(
            "PASS_FOUR_TAP_CONVOLUTION_STATIC_PREFLIGHT"
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
        dt_projection,
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
        raise SystemExit(main())

    except FourTapAuditError as exc:
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