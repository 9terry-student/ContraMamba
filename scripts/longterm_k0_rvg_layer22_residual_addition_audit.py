"""K0-RVG layer-22 residual-addition decomposition audit.

Authenticated block boundary:

    R22 -> RMSNorm22 -> Mixer22 -> Y22
    R23 = R22 + Y22

Therefore:

    delta_R23 = delta_R22 + delta_Y22

and:

    ||delta_R23||^2
      = ||delta_R22||^2
        + ||delta_Y22||^2
        + 2 <delta_R22, delta_Y22>

Observed runtime tensors are float32.  Strict algebraic closure is therefore
evaluated internally in float64, while comparison to observed delta-R23 uses
a separate float32 reconstruction tolerance.

Observational/algebraic only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence


PARENT_FREEZE_COMMIT = (
    "b29e05dd384cddddf8754a9e9925ff9753f49bdb"
)

PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_rmsnorm_residual_factorization_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "a4c5a1aea466df2e70b933050714c564ccf2d7db85f8bb9d36dc8c5e06937c4c"
)

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_rmsnorm_residual_factorization_f2f08a0_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "3f8bd3ed5fd7cf98f0f118a6ef8d685938270ed4780150eda69cd75de439aca8"
)

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
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

SOURCE_LAYER = 22
TARGET_LAYER = 23
HIDDEN_SIZE = 768

RELATIVE_COORDINATES = tuple(range(-1, 7))
POST_HORIZON = 6

EXECUTION_PROTOCOL = (
    "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"
)

BRANCH_RECON_REL_TOL = 1e-7
OBSERVED_DECOMP_RECON_REL_TOL = 5e-6
ALGEBRAIC_CLOSURE_REL_TOL = 1e-12

QUESTION = (
    "Is the layer-23 raw residual delta-R23 pattern already present in "
    "the incoming layer-22 residual delta-R22, added by the layer-22 "
    "mixer update delta-Y22, or materially reshaped by vector interaction "
    "between those two terms?"
)


class ResidualAdditionAuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ResidualAdditionAuditError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_sha(fn: Any) -> str:
    src = textwrap.dedent(
        inspect.getsource(fn)
    )

    return sha256_bytes(
        src.encode("utf-8")
    )


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

    module = importlib.util.module_from_spec(
        spec
    )

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
        "k0_rvg_rmsnorm_parent",
    )


def build_plan(root: Path, parent: Any):
    hidden_parent = parent.load_parent(
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
    ) = parent.build_plan(
        root,
        hidden_parent,
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
        hidden_parent,
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
    hidden_parent: Any,
    four_tap: Any,
    postconv: Any,
    time_step: Any,
    base: Any,
):
    parent.authenticate_parent(
        root,
        hidden_parent,
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
        == "k0-rvg-rmsnorm-residual-factorization-summary-v1",
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


def resolve_layers(
    parent: Any,
    hidden_parent: Any,
    postconv: Any,
    model: Any,
    layer_map: Mapping[int, int],
):
    import torch

    backbone = model.mamba

    require(
        len(backbone.layers) == 24,
        "BACKBONE_LAYER_COUNT_MISMATCH",
    )

    layer22 = backbone.layers[
        SOURCE_LAYER
    ]

    layer23 = backbone.layers[
        TARGET_LAYER
    ]

    require(
        type(layer22) is type(layer23),
        "BLOCK_CLASS_MISMATCH",
    )

    require(
        hasattr(layer22, "norm")
        and hasattr(layer22, "mixer"),
        "LAYER22_CHILDREN_MISSING",
    )

    require(
        bool(layer22.residual_in_fp32)
        is True,
        "LAYER22_RESIDUAL_IN_FP32_NOT_TRUE",
    )

    require(
        tuple(layer22.norm.weight.shape)
        == (HIDDEN_SIZE,),
        "LAYER22_NORM_WEIGHT_SHAPE_MISMATCH",
    )

    require(
        layer22.norm.weight.dtype
        == torch.float32,
        "LAYER22_NORM_WEIGHT_DTYPE_MISMATCH",
    )

    require(
        float(
            layer22.norm.variance_epsilon
        )
        == 1e-5,
        "LAYER22_RMS_EPS_MISMATCH",
    )

    require(
        int(layer22.mixer.hidden_size)
        == HIDDEN_SIZE,
        "LAYER22_MIXER_HIDDEN_SIZE_MISMATCH",
    )

    require(
        source_sha(
            type(layer22).forward
        )
        == EXPECTED_BLOCK_FORWARD_SHA256,
        "BLOCK_FORWARD_SHA256_MISMATCH",
    )

    require(
        source_sha(
            type(backbone).forward
        )
        == EXPECTED_BACKBONE_FORWARD_SHA256,
        "BACKBONE_FORWARD_SHA256_MISMATCH",
    )

    block23, norm23, mixer23 = (
        parent.resolve_layer23(
            hidden_parent,
            postconv,
            model,
            layer_map,
        )
    )

    require(
        block23 is layer23,
        "PARENT_LAYER23_RESOLUTION_MISMATCH",
    )

    return (
        backbone,
        layer22,
        layer23,
        norm23,
        mixer23,
    )


def capture_terms(
    parent: Any,
    hidden_parent: Any,
    four_tap: Any,
    postconv: Any,
    dt_projection: Any,
    secant: Any,
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    layer22: Any,
    layer23: Any,
    norm23: Any,
    mixer23: Any,
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    import torch

    targets = tuple(
        int(i)
        for i in targets
    )

    r22_records = {}
    y22_records = {}
    r23_records = {}

    counts = {
        "block22_pre": 0,
        "mixer22_post": 0,
        "block22_post": 0,
    }

    def block22_pre_hook(module, args):
        counts["block22_pre"] += 1

        require(
            counts["block22_pre"] == 1,
            "DUPLICATE_BLOCK22_PRE_HOOK",
        )

        require(
            len(args) >= 1,
            "BLOCK22_PRE_ARG_COUNT_MISMATCH",
        )

        full = secant.snapshot(
            args[0],
            "LAYER22_INPUT_R22_FULL",
        )

        require(
            len(full.shape) == 3,
            "R22_RANK_MISMATCH",
        )

        require(
            full.shape[0] == 1,
            "R22_BATCH_MISMATCH",
        )

        require(
            full.shape[-1]
            == HIDDEN_SIZE,
            "R22_WIDTH_MISMATCH",
        )

        for i in targets:
            require(
                0 <= i < full.shape[1],
                f"R22_TARGET_OUT_OF_RANGE:{i}",
            )

            r22_records[i] = (
                full[:, i, :]
                .contiguous()
                .clone()
            )

    def mixer22_post_hook(
        module,
        args,
        output,
    ):
        counts["mixer22_post"] += 1

        require(
            counts["mixer22_post"] == 1,
            "DUPLICATE_MIXER22_POST_HOOK",
        )

        full = secant.snapshot(
            output,
            "LAYER22_MIXER_OUTPUT_Y22_FULL",
        )

        require(
            len(full.shape) == 3,
            "Y22_RANK_MISMATCH",
        )

        require(
            full.shape[0] == 1,
            "Y22_BATCH_MISMATCH",
        )

        require(
            full.shape[-1]
            == HIDDEN_SIZE,
            "Y22_WIDTH_MISMATCH",
        )

        for i in targets:
            require(
                0 <= i < full.shape[1],
                f"Y22_TARGET_OUT_OF_RANGE:{i}",
            )

            y22_records[i] = (
                full[:, i, :]
                .contiguous()
                .clone()
            )

    def block22_post_hook(
        module,
        args,
        output,
    ):
        counts["block22_post"] += 1

        require(
            counts["block22_post"] == 1,
            "DUPLICATE_BLOCK22_POST_HOOK",
        )

        full = secant.snapshot(
            output,
            "LAYER22_OUTPUT_R23_FULL",
        )

        require(
            len(full.shape) == 3,
            "R23_RANK_MISMATCH",
        )

        require(
            full.shape[0] == 1,
            "R23_BATCH_MISMATCH",
        )

        require(
            full.shape[-1]
            == HIDDEN_SIZE,
            "R23_WIDTH_MISMATCH",
        )

        for i in targets:
            require(
                0 <= i < full.shape[1],
                f"R23_TARGET_OUT_OF_RANGE:{i}",
            )

            r23_records[i] = (
                full[:, i, :]
                .contiguous()
                .clone()
            )

    h1 = layer22.register_forward_pre_hook(
        block22_pre_hook
    )

    h2 = layer22.mixer.register_forward_hook(
        mixer22_post_hook
    )

    h3 = layer22.register_forward_hook(
        block22_post_hook
    )

    try:
        records = parent.capture_rx(
            hidden_parent,
            four_tap,
            postconv,
            dt_projection,
            secant,
            base,
            model,
            binding,
            layer_map,
            layer23,
            norm23,
            mixer23,
            token_ids,
            targets,
        )

    finally:
        h1.remove()
        h2.remove()
        h3.remove()

    require(
        counts["block22_pre"] == 1,
        "BLOCK22_PRE_HOOK_COUNT_FAILURE",
    )

    require(
        counts["mixer22_post"] == 1,
        "MIXER22_POST_HOOK_COUNT_FAILURE",
    )

    require(
        counts["block22_post"] == 1,
        "BLOCK22_POST_HOOK_COUNT_FAILURE",
    )

    require(
        set(r22_records)
        == set(targets),
        "R22_TARGET_SET_MISMATCH",
    )

    require(
        set(y22_records)
        == set(targets),
        "Y22_TARGET_SET_MISMATCH",
    )

    require(
        set(r23_records)
        == set(targets),
        "R23_TARGET_SET_MISMATCH",
    )

    require(
        set(records)
        == set(targets),
        "PARENT_TARGET_SET_MISMATCH",
    )

    for i in targets:
        r22 = r22_records[i]
        y22 = y22_records[i]
        r23 = r23_records[i]

        require(
            tuple(r22.shape)
            == (1, HIDDEN_SIZE),
            f"R22_SHAPE_MISMATCH:{i}",
        )

        require(
            tuple(y22.shape)
            == (1, HIDDEN_SIZE),
            f"Y22_SHAPE_MISMATCH:{i}",
        )

        require(
            tuple(r23.shape)
            == (1, HIDDEN_SIZE),
            f"R23_SHAPE_MISMATCH:{i}",
        )

        require(
            "R" in records[i],
            f"PARENT_R23_MISSING:{i}",
        )

        require(
            torch.equal(
                r23,
                records[i]["R"],
            ),
            f"PARENT_R23_EXACT_MISMATCH:{i}",
        )

        residual_used = (
            r22.to(torch.float32)
        )

        reconstructed = (
            residual_used
            + y22.to(torch.float32)
        )

        branch_residual = (
            reconstructed
            - r23.to(torch.float32)
        )

        branch_rel = float(
            torch.linalg.vector_norm(
                branch_residual.to(
                    torch.float64
                )
            ).item()
            / max(
                torch.linalg.vector_norm(
                    r23.to(torch.float64)
                ).item(),
                1e-12,
            )
        )

        require(
            branch_rel
            <= BRANCH_RECON_REL_TOL,
            (
                "BRANCH_R23_RECONSTRUCTION_FAILURE:"
                f"{i}:{branch_rel}"
            ),
        )

        records[i]["R22"] = r22
        records[i]["Y22"] = y22
        records[i]["R23"] = r23
        records[i][
            "branch_r23_reconstruction_relative_residual"
        ] = branch_rel

    return records


def metric_rows_for_pair(
    row: Mapping[str, Any],
    matched: Mapping[int, Mapping[str, Any]],
    swapped: Mapping[int, Mapping[str, Any]],
    cohort: frozenset[int],
    signature: str,
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

        r22m = (
            matched[token]["R22"]
            .to(torch.float64)
        )

        r22s = (
            swapped[token]["R22"]
            .to(torch.float64)
        )

        y22m = (
            matched[token]["Y22"]
            .to(torch.float64)
        )

        y22s = (
            swapped[token]["Y22"]
            .to(torch.float64)
        )

        r23m = (
            matched[token]["R23"]
            .to(torch.float64)
        )

        r23s = (
            swapped[token]["R23"]
            .to(torch.float64)
        )

        dr22 = r22m - r22s
        dy22 = y22m - y22s
        dr23 = r23m - r23s

        q_sum = dr22 + dy22

        dr22_l2 = float(
            torch.linalg.vector_norm(
                dr22
            ).item()
        )

        dy22_l2 = float(
            torch.linalg.vector_norm(
                dy22
            ).item()
        )

        dr23_l2 = float(
            torch.linalg.vector_norm(
                dr23
            ).item()
        )

        q_sum_l2 = float(
            torch.linalg.vector_norm(
                q_sum
            ).item()
        )

        observed_residual = (
            q_sum - dr23
        )

        observed_rel = float(
            torch.linalg.vector_norm(
                observed_residual
            ).item()
            / max(
                dr23_l2,
                1e-12,
            )
        )

        require(
            observed_rel
            <= OBSERVED_DECOMP_RECON_REL_TOL,
            (
                "OBSERVED_FLOAT32_RESIDUAL_ADD_RECON_FAILURE:"
                f"{idx}:{role}:{k}:{observed_rel}"
            ),
        )

        dr22_sq = dr22_l2 * dr22_l2
        dy22_sq = dy22_l2 * dy22_l2

        cross = float(
            2.0
            * torch.sum(
                dr22 * dy22
            ).item()
        )

        rhs = (
            dr22_sq
            + dy22_sq
            + cross
        )

        closure = abs(
            q_sum_l2 * q_sum_l2
            - rhs
        )

        require(
            closure
            <= ALGEBRAIC_CLOSURE_REL_TOL
            * max(
                q_sum_l2 * q_sum_l2,
                dr22_sq + dy22_sq,
                1.0,
            ),
            (
                "RESIDUAL_ADD_SQUARED_NORM_CLOSURE_FAILURE:"
                f"{idx}:{role}:{k}:{closure}"
            ),
        )

        rss_sq = (
            dr22_sq + dy22_sq
        )

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

        residual_add_transfer = (
            q_sum_l2 / dr22_l2
            if dr22_l2 > 0.0
            else 0.0
        )

        if k == -1:
            require(
                dr22_l2 == 0.0,
                f"K_MINUS_1_DR22_NONZERO:{idx}:{role}",
            )

            require(
                dy22_l2 == 0.0,
                f"K_MINUS_1_DY22_NONZERO:{idx}:{role}",
            )

            require(
                dr23_l2 == 0.0,
                f"K_MINUS_1_DR23_NONZERO:{idx}:{role}",
            )

        values = {
            "schema_version":
                "k0-rvg-layer22-residual-addition-row-v1",

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

            "delta_r22_l2":
                dr22_l2,

            "delta_y22_l2":
                dy22_l2,

            "delta_r23_l2":
                dr23_l2,

            "q_sum_l2":
                q_sum_l2,

            "r22_energy_fraction":
                (
                    dr22_sq / rss_sq
                    if rss_sq > 0.0
                    else 0.0
                ),

            "y22_energy_fraction":
                (
                    dy22_sq / rss_sq
                    if rss_sq > 0.0
                    else 0.0
                ),

            "cross_term":
                cross,

            "cross_fraction_of_rss_sq":
                cross_fraction,

            "vector_addition_ratio":
                addition_ratio,

            "residual_add_transfer":
                residual_add_transfer,

            "squared_norm_closure_error":
                closure,

            "observed_decomposition_relative_residual":
                observed_rel,

            "branch_r23_reconstruction_relative_residual":
                max(
                    float(
                        matched[token][
                            "branch_r23_reconstruction_relative_residual"
                        ]
                    ),
                    float(
                        swapped[token][
                            "branch_r23_reconstruction_relative_residual"
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
    "delta_r22_l2",
    "delta_y22_l2",
    "delta_r23_l2",
    "q_sum_l2",
    "r22_energy_fraction",
    "y22_energy_fraction",
    "cross_term",
    "cross_fraction_of_rss_sq",
    "vector_addition_ratio",
    "residual_add_transfer",
    "squared_norm_closure_error",
    "observed_decomposition_relative_residual",
    "branch_r23_reconstruction_relative_residual",
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
            "k0-rvg-layer22-residual-addition-summary-v1",

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

        "source_layer":
            SOURCE_LAYER,

        "target_layer":
            TARGET_LAYER,

        "hidden_size":
            HIDDEN_SIZE,

        "relative_coordinates":
            list(RELATIVE_COORDINATES),

        "execution_protocol":
            EXECUTION_PROTOCOL,

        "identity":
            (
                "delta_R23 = "
                "delta_R22 + delta_Y22"
            ),

        "full_336_trajectory":
            aggregate_trajectory(rows),

        "common_330_ddsssss_trajectory":
            aggregate_trajectory(
                common_rows
            ),

        "max_observed_decomposition_relative_residual":
            max(
                float(
                    row[
                        "observed_decomposition_relative_residual"
                    ]
                )
                for row in rows
            ),

        "max_branch_r23_reconstruction_relative_residual":
            max(
                float(
                    row[
                        "branch_r23_reconstruction_relative_residual"
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
                    ["delta_r23_l2"]["median"]
                )

                expected = float(
                    parent[role][str(k)]
                    ["delta_r_l2"]["median"]
                )

                require(
                    math.isclose(
                        got,
                        expected,
                        rel_tol=1e-13,
                        abs_tol=1e-13,
                    ),
                    (
                        "PARENT_DELTA_R23_REPRODUCTION_FAILURE:"
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
    hidden_parent,
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

    (
        backbone,
        layer22,
        layer23,
        norm23,
        mixer23,
    ) = resolve_layers(
        parent,
        hidden_parent,
        postconv,
        model,
        layer_map,
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

        require(
            len(matched_prefix)
            == cutoff
            and len(swapped_prefix)
            == cutoff,
            f"PREFIX_LENGTH_FAILURE:{idx}:{role}",
        )

        matched = capture_terms(
            parent,
            hidden_parent,
            four_tap,
            postconv,
            dt_projection,
            secant,
            base,
            model,
            binding,
            layer_map,
            layer22,
            layer23,
            norm23,
            mixer23,
            matched_prefix,
            row["targets"],
        )

        forward_count += 1

        swapped = capture_terms(
            parent,
            hidden_parent,
            four_tap,
            postconv,
            dt_projection,
            secant,
            base,
            model,
            binding,
            layer_map,
            layer22,
            layer23,
            norm23,
            mixer23,
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
        / "layer22_residual_addition_metrics.jsonl"
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
            "k0-rvg-layer22-residual-addition-execution-manifest-v1",

        "runtime_git_head":
            repo["head"],

        "runtime_branch":
            repo["branch"],

        "parent_freeze_commit":
            PARENT_FREEZE_COMMIT,

        "scientific_question":
            QUESTION,

        "identity":
            "delta_R23 = delta_R22 + delta_Y22",

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

        "source_layer":
            SOURCE_LAYER,

        "target_layer":
            TARGET_LAYER,

        "hidden_size":
            HIDDEN_SIZE,

        "block_forward_sha256":
            EXPECTED_BLOCK_FORWARD_SHA256,

        "backbone_forward_sha256":
            EXPECTED_BACKBONE_FORWARD_SHA256,

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

        "parent_delta_r23_summary_match":
            True,

        "raw_vectors_persisted":
            False,

        "scientific_model_forward_executed":
            True,

        "scientific_layer22_input_r22_read":
            True,

        "scientific_layer22_mixer_output_y22_read":
            True,

        "scientific_layer22_output_r23_read":
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
            "layer22_residual_addition_metrics.jsonl":
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
        "PASS_LAYER22_RESIDUAL_ADDITION_EXECUTION"
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
        "parent_delta_r23_summary_match = True"
    )

    print(
        "max_observed_decomposition_relative_residual =",
        summary[
            "max_observed_decomposition_relative_residual"
        ],
    )

    print(
        "max_branch_r23_reconstruction_relative_residual =",
        summary[
            "max_branch_r23_reconstruction_relative_residual"
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
                f"R22={t['delta_r22_l2']['median']} "
                f"Y22={t['delta_y22_l2']['median']} "
                f"R23={t['delta_r23_l2']['median']} "
                f"ADD={t['vector_addition_ratio']['median']} "
                f"CROSS={t['cross_fraction_of_rss_sq']['median']} "
                f"TRANSFER={t['residual_add_transfer']['median']}"
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
        hidden_parent,
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
        hidden_parent,
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
        "=== LAYER22 RESIDUAL ADDITION AUDIT PLAN ==="
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
        "source_layer =",
        SOURCE_LAYER,
    )

    print(
        "target_layer =",
        TARGET_LAYER,
    )

    print(
        "scientific_question =",
        QUESTION,
    )

    print(
        "identity = "
        "delta_R23 = delta_R22 + delta_Y22"
    )

    print(
        "squared_norm_identity = "
        "||delta_R23||^2 = "
        "||delta_R22||^2 + ||delta_Y22||^2 "
        "+ 2<delta_R22,delta_Y22>"
    )

    print(
        "block_forward_sha256 =",
        EXPECTED_BLOCK_FORWARD_SHA256,
    )

    print(
        "backbone_forward_sha256 =",
        EXPECTED_BACKBONE_FORWARD_SHA256,
    )

    print(
        "mamba_source_sha256 =",
        source["source_sha256"],
    )

    print(
        "observed_float32_reconstruction_rel_tol =",
        OBSERVED_DECOMP_RECON_REL_TOL,
    )

    print(
        "algebraic_closure_rel_tol =",
        ALGEBRAIC_CLOSURE_REL_TOL,
    )

    print(
        "parent_delta_r23_summary_preserved_by_design = True"
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
            "PASS_LAYER22_RESIDUAL_ADDITION_STATIC_PREFLIGHT"
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
        hidden_parent,
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

    except ResidualAdditionAuditError as exc:
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