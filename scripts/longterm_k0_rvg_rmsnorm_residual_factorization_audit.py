"""K0-RVG layer-23 RMSNorm residual factorization audit.

Authenticated source boundary:

    R -> RMSNorm -> X -> mixer

with exact runtime implementation:

    variance = mean(R^2)
    s = rsqrt(variance + eps)
    X = gamma * (s * R)

For matched/swapped branches:

    delta_X = Q_R + Q_s

where

    Q_R = gamma * s_bar * delta_R
    Q_s = gamma * R_bar * delta_s

and bars are symmetric branch averages.

Exact squared-norm decomposition:

    ||delta_X||^2
      = ||Q_R||^2
        + ||Q_s||^2
        + 2 <Q_R, Q_s>

Observational/algebraic only.
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
from typing import Any, Mapping, Sequence


PARENT_FREEZE_COMMIT = (
    "55f27f7444e4cb8d9ae898719ad68b58c3cdf44d"
)

PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_hidden_inproj_transfer_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "65810137bebb768c12bb11fd279a630d6a6a1f1a6116b3b12f00a0c286ea8d64"
)

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_hidden_inproj_transfer_8e9b5fd_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "d9bd8ea89ec9c7e561164a1cd6f4803d69943748c2249d769b748576b7ca953f"
)

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)

EXPECTED_RMSNORM_FORWARD_SHA256 = (
    "cdbbe12603e777d7097a001fcc5fede6684a0345fcc1c781ad5b0e25414d381f"
)

EXPECTED_BLOCK_FORWARD_SHA256 = (
    "0f808b4d539a496e1681c81d39799d7072fa5c6da381a07cfa19f35fe825b3e4"
)

EXPECTED_BACKBONE_FORWARD_SHA256 = (
    "3f332bd50e6ea4ff64468c8d748672f90ffc43912d22c9d25d02b3e87d1661a6"
)

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FORWARD_COUNT = 1344

PRIMARY_LAYER = 23
HIDDEN_SIZE = 768
RMS_EPS = 1e-5

RELATIVE_COORDINATES = tuple(range(-1, 7))
POST_HORIZON = 6

EXECUTION_PROTOCOL = (
    "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"
)

RMS_RECON_REL_TOL = 1e-6

# The symmetric Q_R + Q_s identity is evaluated in float64 from captured
# float32 runtime operands.  Its internal squared-norm closure can therefore
# be guarded at near-float64 precision.  Comparison against the observed
# runtime delta-X must separately allow the rounding introduced by the
# float32 RMSNorm execution.
OBSERVED_DECOMP_RECON_REL_TOL = 5e-6
ALGEBRAIC_CLOSURE_REL_TOL = 1e-12

QUESTION = (
    "Is the layer-23 normalized mixer-input delta-X pattern primarily "
    "inherited from the raw residual-stream delta-R vector, materially "
    "reshaped by branch-specific RMS scaling, or affected by interaction "
    "between the residual-vector and RMS-scale terms?"
)


class RMSNormAuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RMSNormAuditError(message)


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


def load_parent(root: Path):
    path = root / PARENT_RUNNER_REL

    require(
        sha256_bytes(path.read_bytes())
        == PARENT_RUNNER_SHA256,
        "PARENT_RUNNER_SHA256_MISMATCH",
    )

    return import_module(
        path,
        "k0_rvg_hidden_inproj_parent",
    )


def build_plan(root: Path, parent: Any):
    four_tap = parent.load_four_tap(
        root
    )

    (
        postconv,
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
    ) = parent.build_plan(
        root,
        four_tap,
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
        four_tap,
        postconv,
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
    four_tap: Any,
    postconv: Any,
    time_step: Any,
    base: Any,
):
    parent.authenticate_parent(
        root,
        four_tap,
        postconv,
        time_step,
        base,
    )

    repo = base.authenticate_repo(
        root
    )

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

    frozen = base.git_bytes(
        root,
        f"{PARENT_FREEZE_COMMIT}:{PARENT_SUMMARY_REL}",
    )

    require(
        sha256_bytes(frozen)
        == PARENT_SUMMARY_SHA256,
        "PARENT_SUMMARY_SHA256_MISMATCH",
    )

    current = root / PARENT_SUMMARY_REL

    require(
        current.is_file(),
        "PARENT_SUMMARY_MISSING",
    )

    require(
        current.read_bytes()
        == frozen,
        "PARENT_SUMMARY_WORKTREE_DRIFT",
    )

    summary = json.loads(
        frozen
    )

    require(
        summary.get("schema_version")
        == "k0-rvg-hidden-inproj-transfer-summary-v1",
        "PARENT_SCHEMA_MISMATCH",
    )

    return repo, summary


def resolve_layer23(
    parent: Any,
    postconv: Any,
    model: Any,
    layer_map: Mapping[int, int],
):
    import torch

    mixer, _ = parent.resolve_hidden_inproj(
        postconv,
        model,
        layer_map,
    )

    parents = []

    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child is mixer:
                parents.append(
                    (
                        module_name,
                        child_name,
                        module,
                    )
                )

    require(
        len(parents) == 1,
        "LAYER23_BLOCK_RESOLUTION_FAILURE",
    )

    block_name, child_name, block = parents[0]

    require(
        child_name == "mixer",
        "MIXER_CHILD_NAME_MISMATCH",
    )

    require(
        hasattr(block, "norm"),
        "BLOCK_NORM_MISSING",
    )

    norm = block.norm

    require(
        tuple(norm.weight.shape)
        == (HIDDEN_SIZE,),
        "RMS_WEIGHT_SHAPE_MISMATCH",
    )

    require(
        norm.weight.dtype
        == torch.float32,
        "RMS_WEIGHT_DTYPE_MISMATCH",
    )

    require(
        float(norm.variance_epsilon)
        == RMS_EPS,
        "RMS_EPS_MISMATCH",
    )

    require(
        bool(block.residual_in_fp32)
        is True,
        "RESIDUAL_IN_FP32_NOT_TRUE",
    )

    return block, norm, mixer


def capture_rx(
    parent: Any,
    four_tap: Any,
    postconv: Any,
    dt_projection: Any,
    secant: Any,
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    block: Any,
    norm: Any,
    mixer: Any,
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    import torch

    targets = tuple(
        int(i)
        for i in targets
    )

    r_records = {}
    x_records = {}

    counts = {
        "norm_pre": 0,
        "norm_post": 0,
    }

    def norm_pre_hook(module, args):
        counts["norm_pre"] += 1

        require(
            counts["norm_pre"] == 1,
            "DUPLICATE_NORM_PRE_HOOK",
        )

        require(
            len(args) == 1,
            "NORM_PRE_ARG_COUNT_MISMATCH",
        )

        r_full = secant.snapshot(
            args[0],
            "LAYER23_RMSNORM_INPUT_R_FULL",
        )

        require(
            tuple(r_full.shape[:1])
            == (1,),
            "R_BATCH_MISMATCH",
        )

        require(
            r_full.shape[-1]
            == HIDDEN_SIZE,
            "R_WIDTH_MISMATCH",
        )

        for i in targets:
            require(
                0 <= i < r_full.shape[1],
                f"R_TARGET_OUT_OF_RANGE:{i}",
            )

            r_records[i] = (
                r_full[:, i, :]
                .contiguous()
                .clone()
            )

    def norm_post_hook(module, args, output):
        counts["norm_post"] += 1

        require(
            counts["norm_post"] == 1,
            "DUPLICATE_NORM_POST_HOOK",
        )

        x_full = secant.snapshot(
            output,
            "LAYER23_RMSNORM_OUTPUT_X_FULL",
        )

        require(
            x_full.shape[-1]
            == HIDDEN_SIZE,
            "X_WIDTH_MISMATCH",
        )

        for i in targets:
            require(
                0 <= i < x_full.shape[1],
                f"X_TARGET_OUT_OF_RANGE:{i}",
            )

            x_records[i] = (
                x_full[:, i, :]
                .contiguous()
                .clone()
            )

    pre_handle = norm.register_forward_pre_hook(
        norm_pre_hook
    )

    post_handle = norm.register_forward_hook(
        norm_post_hook
    )

    try:
        records = parent.capture_xh(
            four_tap,
            postconv,
            dt_projection,
            secant,
            base,
            model,
            binding,
            layer_map,
            mixer,
            mixer.in_proj.weight[
                :1536, :
            ].detach().cpu().contiguous().clone(),
            token_ids,
            targets,
        )

    finally:
        pre_handle.remove()
        post_handle.remove()

    require(
        counts["norm_pre"] == 1
        and counts["norm_post"] == 1,
        "RMS_HOOK_COUNT_FAILURE",
    )

    require(
        set(r_records)
        == set(targets),
        "R_TARGET_SET_MISMATCH",
    )

    require(
        set(x_records)
        == set(targets),
        "X_TARGET_SET_MISMATCH",
    )

    require(
        set(records)
        == set(targets),
        "PARENT_TARGET_SET_MISMATCH",
    )

    gamma = (
        norm.weight.detach()
        .cpu()
        .contiguous()
    )

    for i in targets:
        r = r_records[i]
        x = x_records[i]

        require(
            tuple(r.shape)
            == (1, HIDDEN_SIZE),
            f"R_SHAPE_MISMATCH:{i}",
        )

        require(
            tuple(x.shape)
            == (1, HIDDEN_SIZE),
            f"X_SHAPE_MISMATCH:{i}",
        )

        require(
            "X" in records[i],
            f"PARENT_X_MISSING:{i}",
        )

        require(
            torch.equal(
                x,
                records[i]["X"],
            ),
            f"PARENT_X_EXACT_MISMATCH:{i}",
        )

        rf = r.to(torch.float32)

        variance = (
            rf.pow(2)
            .mean(
                dim=-1,
                keepdim=True,
            )
        )

        scale = torch.rsqrt(
            variance + RMS_EPS
        )

        reconstructed = (
            gamma
            * (
                rf
                * scale
            )
        )

        reconstructed = (
            reconstructed
            .to(x.dtype)
            .contiguous()
        )

        residual = (
            reconstructed
            - x
        )

        rel = float(
            torch.linalg.vector_norm(
                residual.to(torch.float64)
            ).item()
            / max(
                torch.linalg.vector_norm(
                    x.to(torch.float64)
                ).item(),
                1e-12,
            )
        )

        require(
            rel <= RMS_RECON_REL_TOL,
            (
                "RMS_RECONSTRUCTION_FAILURE:"
                f"{i}:{rel}"
            ),
        )

        records[i]["R"] = r
        records[i]["X"] = x
        records[i]["rms_scale"] = float(
            scale.item()
        )
        records[i][
            "rms_reconstruction_relative_residual"
        ] = rel

    return records


def metric_rows_for_pair(
    row: Mapping[str, Any],
    matched: Mapping[int, Mapping[str, Any]],
    swapped: Mapping[int, Mapping[str, Any]],
    cohort: frozenset[int],
    signature: str,
    gamma: Any,
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

    gamma64 = gamma.to(torch.float64)

    output = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k

        rm = matched[token]["R"].to(
            torch.float64
        )

        rs = swapped[token]["R"].to(
            torch.float64
        )

        xm = matched[token]["X"].to(
            torch.float64
        )

        xs = swapped[token]["X"].to(
            torch.float64
        )

        sm = float(
            matched[token]["rms_scale"]
        )

        ss = float(
            swapped[token]["rms_scale"]
        )

        dr = rm - rs
        dx = xm - xs

        rbar = 0.5 * (
            rm + rs
        )

        sbar = 0.5 * (
            sm + ss
        )

        ds = sm - ss

        q_r = (
            gamma64
            * (
                sbar * dr
            )
        )

        q_s = (
            gamma64
            * (
                rbar * ds
            )
        )

        q_sum = (
            q_r + q_s
        )

        residual = (
            q_sum - dx
        )

        dx_l2 = float(
            torch.linalg.vector_norm(
                dx
            ).item()
        )

        q_sum_l2 = float(
            torch.linalg.vector_norm(
                q_sum
            ).item()
        )

        dr_l2 = float(
            torch.linalg.vector_norm(
                dr
            ).item()
        )

        qr_l2 = float(
            torch.linalg.vector_norm(
                q_r
            ).item()
        )

        qs_l2 = float(
            torch.linalg.vector_norm(
                q_s
            ).item()
        )

        recon_rel = float(
            torch.linalg.vector_norm(
                residual
            ).item()
            / max(
                dx_l2,
                1e-12,
            )
        )

        require(
            recon_rel
            <= OBSERVED_DECOMP_RECON_REL_TOL,
            (
                "RMS_OBSERVED_FLOAT32_RECONSTRUCTION_FAILURE:"
                f"{idx}:{role}:{k}:{recon_rel}"
            ),
        )

        qr_sq = qr_l2 * qr_l2
        qs_sq = qs_l2 * qs_l2

        cross = float(
            2.0
            * torch.sum(
                q_r * q_s
            ).item()
        )

        rhs = (
            qr_sq
            + qs_sq
            + cross
        )

        # Strict algebraic closure is evaluated entirely on the float64
        # Q vectors.  Do not mix the observed float32 delta-X into this
        # identity guard.
        closure = abs(
            q_sum_l2 * q_sum_l2
            - rhs
        )

        require(
            closure
            <= ALGEBRAIC_CLOSURE_REL_TOL
            * max(
                q_sum_l2 * q_sum_l2,
                qr_sq + qs_sq,
                1.0,
            ),
            (
                "RMS_SQUARED_NORM_CLOSURE_FAILURE:"
                f"{idx}:{role}:{k}:{closure}"
            ),
        )

        rss_sq = qr_sq + qs_sq

        addition_ratio = (
            q_sum_l2
            / math.sqrt(rss_sq)
            if rss_sq > 0.0
            else 0.0
        )

        cross_fraction = (
            cross / rss_sq
            if rss_sq > 0.0
            else 0.0
        )

        if k == -1:
            require(
                dr_l2 == 0.0,
                f"K_MINUS_1_DR_NONZERO:{idx}:{role}",
            )

            require(
                dx_l2 == 0.0,
                f"K_MINUS_1_DX_NONZERO:{idx}:{role}",
            )

        values = {
            "schema_version":
                "k0-rvg-rmsnorm-residual-factorization-row-v1",

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

            "delta_r_l2":
                dr_l2,

            "delta_x_l2":
                dx_l2,

            "q_sum_l2":
                q_sum_l2,

            "matched_rms_scale":
                sm,

            "swapped_rms_scale":
                ss,

            "delta_rms_scale":
                ds,

            "q_r_l2":
                qr_l2,

            "q_s_l2":
                qs_l2,

            "q_r_energy_fraction":
                (
                    qr_sq / rss_sq
                    if rss_sq > 0.0
                    else 0.0
                ),

            "q_s_energy_fraction":
                (
                    qs_sq / rss_sq
                    if rss_sq > 0.0
                    else 0.0
                ),

            "cross_term":
                cross,

            "cross_fraction_of_rss_sq":
                cross_fraction,

            "vector_addition_ratio":
                addition_ratio,

            "squared_norm_closure_error":
                closure,

            "symmetric_decomposition_relative_residual":
                recon_rel,

            "rms_reconstruction_relative_residual":
                max(
                    float(
                        matched[token][
                            "rms_reconstruction_relative_residual"
                        ]
                    ),
                    float(
                        swapped[token][
                            "rms_reconstruction_relative_residual"
                        ]
                    ),
                ),

            "source_snapshot_dtype":
                "torch.float32",

            "metric_accumulation_dtype":
                "torch.float64",
        }

        require(
            all(
                math.isfinite(float(v))
                for v in values.values()
                if isinstance(
                    v,
                    (int, float),
                )
                and not isinstance(v, bool)
            ),
            f"NONFINITE_METRIC:{idx}:{role}:{k}",
        )

        output.append(values)

    return output


SUMMARY_FIELDS = (
    "delta_r_l2",
    "delta_x_l2",
    "q_sum_l2",
    "matched_rms_scale",
    "swapped_rms_scale",
    "delta_rms_scale",
    "q_r_l2",
    "q_s_l2",
    "q_r_energy_fraction",
    "q_s_energy_fraction",
    "cross_term",
    "cross_fraction_of_rss_sq",
    "vector_addition_ratio",
    "squared_norm_closure_error",
    "symmetric_decomposition_relative_residual",
    "rms_reconstruction_relative_residual",
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
                    row["relative_coordinate"]
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
            "k0-rvg-rmsnorm-residual-factorization-summary-v1",

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

        "hidden_size":
            HIDDEN_SIZE,

        "rms_epsilon":
            RMS_EPS,

        "relative_coordinates":
            list(RELATIVE_COORDINATES),

        "execution_protocol":
            EXECUTION_PROTOCOL,

        "identity":
            (
                "delta_X = "
                "gamma*(s_bar*delta_R + R_bar*delta_s)"
            ),

        "full_336_trajectory":
            aggregate_trajectory(rows),

        "common_330_ddsssss_trajectory":
            aggregate_trajectory(
                common_rows
            ),

        "max_symmetric_decomposition_relative_residual":
            max(
                float(
                    row[
                        "symmetric_decomposition_relative_residual"
                    ]
                )
                for row in rows
            ),

        "max_rms_reconstruction_relative_residual":
            max(
                float(
                    row[
                        "rms_reconstruction_relative_residual"
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
                    ["delta_x_l2"]["median"]
                )

                expected = float(
                    parent[role][str(k)]
                    ["delta_x_l2"]["median"]
                )

                require(
                    math.isclose(
                        got,
                        expected,
                        rel_tol=1e-13,
                        abs_tol=1e-13,
                    ),
                    (
                        "PARENT_DELTA_X_REPRODUCTION_FAILURE:"
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
    four_tap,
    postconv,
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

    block, norm, mixer = resolve_layer23(
        parent,
        postconv,
        model,
        layer_map,
    )

    gamma = (
        norm.weight.detach()
        .cpu()
        .contiguous()
        .clone()
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

        matched = capture_rx(
            parent,
            four_tap,
            postconv,
            dt_projection,
            secant,
            base,
            model,
            binding,
            layer_map,
            block,
            norm,
            mixer,
            matched_prefix,
            row["targets"],
        )

        forward_count += 1

        swapped = capture_rx(
            parent,
            four_tap,
            postconv,
            dt_projection,
            secant,
            base,
            model,
            binding,
            layer_map,
            block,
            norm,
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
                gamma,
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
        / "rmsnorm_residual_factorization_metrics.jsonl"
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
            "k0-rvg-rmsnorm-residual-factorization-execution-manifest-v1",

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
                "delta_X = "
                "gamma*(s_bar*delta_R + R_bar*delta_s)"
            ),

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

        "hidden_size":
            HIDDEN_SIZE,

        "rms_epsilon":
            RMS_EPS,

        "mamba_source_sha256":
            binding.source_sha256,

        "rmsnorm_forward_sha256":
            EXPECTED_RMSNORM_FORWARD_SHA256,

        "block_forward_sha256":
            EXPECTED_BLOCK_FORWARD_SHA256,

        "backbone_forward_sha256":
            EXPECTED_BACKBONE_FORWARD_SHA256,

        "handoff_zip_sha256":
            handoff["zip_sha256"],

        "checkpoint_sha256":
            handoff["checkpoint_sha256"],

        "encoder_canonical_digest":
            encoder["canonical_digest"],

        "encoder_raw_concat_digest":
            encoder["raw_concat_digest"],

        "parent_delta_x_summary_match":
            True,

        "raw_vectors_persisted":
            False,

        "scientific_model_forward_executed":
            True,

        "scientific_layer23_rms_input_r_read":
            True,

        "scientific_layer23_rms_output_x_read":
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
            "rmsnorm_residual_factorization_metrics.jsonl":
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
        "PASS_RMSNORM_RESIDUAL_FACTORIZATION_EXECUTION"
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
        "parent_delta_x_summary_match = True"
    )

    print(
        "max_symmetric_decomposition_relative_residual =",
        summary[
            "max_symmetric_decomposition_relative_residual"
        ],
    )

    print(
        "max_rms_reconstruction_relative_residual =",
        summary[
            "max_rms_reconstruction_relative_residual"
        ],
    )

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[
                role
            ][str(k)]

            print(
                f"{role}_k{k}_median "
                f"DR={t['delta_r_l2']['median']} "
                f"DX={t['delta_x_l2']['median']} "
                f"QR={t['q_r_l2']['median']} "
                f"QS={t['q_s_l2']['median']} "
                f"ADD={t['vector_addition_ratio']['median']} "
                f"CROSS={t['cross_fraction_of_rss_sq']['median']}"
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

    parent = load_parent(
        root
    )

    (
        four_tap,
        postconv,
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
        four_tap,
        postconv,
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
        "=== RMSNORM RESIDUAL FACTORIZATION AUDIT PLAN ==="
    )

    print("branch =", repo["branch"])
    print("head =", repo["head"])
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
        "rms_epsilon =",
        RMS_EPS,
    )
    print(
        "scientific_question =",
        QUESTION,
    )
    print(
        "identity = "
        "delta_X = gamma*(s_bar*delta_R + R_bar*delta_s)"
    )
    print(
        "RMSNorm source SHA =",
        EXPECTED_RMSNORM_FORWARD_SHA256,
    )
    print(
        "mamba_source_sha256 =",
        source["source_sha256"],
    )
    print(
        "parent_delta_x_summary_preserved_by_design = True"
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
            "PASS_RMSNORM_RESIDUAL_FACTORIZATION_STATIC_PREFLIGHT"
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
        four_tap,
        postconv,
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

    except RMSNormAuditError as exc:
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