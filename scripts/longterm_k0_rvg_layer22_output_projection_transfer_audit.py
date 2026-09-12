"""K0-RVG layer-22 output-projection transfer audit.

Authenticated slow-path boundary:

    V22 = gated SSM/skip output immediately before out_proj
    Y22 = W_O V22

where out_proj is bias-free Linear(1536 -> 768).

For matched/swapped branches:

    delta_Y22 = W_O delta_V22

and therefore:

    ||delta_Y22||
      = ||delta_V22|| * output_projection_transfer

with:

    output_projection_transfer
      = ||W_O delta_V22|| / ||delta_V22||

Observed runtime tensors are float32.  The source-level linear identity is
evaluated in float64 from captured operands, while comparison to observed
runtime delta-Y22 uses a separate float32 reconstruction tolerance.

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
    "e27e27fb5b4d6b3a5ac64973d8b9a566679f5775"
)

PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer22_residual_addition_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "ffeed8c4ce8b7edd7d62100d7f94ad8c558e2604c0a1a68cfc8ffc896ee906ce"
)

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_residual_addition_9cd5e30_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "8641a3a52168bd69f3c5625d11efae80bcf5cf5f13205d801959df50033831d4"
)

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)

EXPECTED_MIXER_FORWARD_SHA256 = (
    "63450e056b1ebbaef584f8025b9c26766f5b97a69cd0c579213f095970dab826"
)

EXPECTED_SLOW_FORWARD_SHA256 = (
    "283d2f21854a8f0804e01fe03c8d0ece29c86dc3128bdf6527a1d6b62d8cd7d9"
)

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FORWARD_COUNT = 1344

SOURCE_LAYER = 22
V_WIDTH = 1536
Y_WIDTH = 768

RELATIVE_COORDINATES = tuple(range(-1, 7))
POST_HORIZON = 6

EXECUTION_PROTOCOL = (
    "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"
)

FLOAT32_BRANCH_RECON_REL_TOL = 5e-6
OBSERVED_LINEAR_RECON_REL_TOL = 5e-6

QUESTION = (
    "Is the strong layer-22 mixer-update delta-Y22 pattern already present "
    "in the gated pre-output-projection delta-V22 representation, or is it "
    "materially produced or attenuated by the bias-free output-projection "
    "geometry?"
)


class OutputProjectionAuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise OutputProjectionAuditError(message)


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
        "k0_rvg_layer22_residual_parent",
    )


def build_plan(root: Path, parent: Any):
    rms_parent = parent.load_parent(
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
    ) = parent.build_plan(
        root,
        rms_parent,
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
        rms_parent,
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
    rms_parent: Any,
    hidden_parent: Any,
    four_tap: Any,
    postconv: Any,
    time_step: Any,
    base: Any,
):
    parent.authenticate_parent(
        root,
        rms_parent,
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
        == "k0-rvg-layer22-residual-addition-summary-v1",
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


def resolve_output_projection(
    parent: Any,
    rms_parent: Any,
    hidden_parent: Any,
    postconv: Any,
    model: Any,
    layer_map: Mapping[int, int],
):
    import torch

    (
        backbone,
        layer22,
        layer23,
        norm23,
        mixer23,
    ) = parent.resolve_layers(
        rms_parent,
        hidden_parent,
        postconv,
        model,
        layer_map,
    )

    mixer22 = layer22.mixer

    require(
        source_sha(
            type(mixer22).forward
        )
        == EXPECTED_MIXER_FORWARD_SHA256,
        "MIXER_FORWARD_SHA256_MISMATCH",
    )

    require(
        source_sha(
            type(mixer22).slow_forward
        )
        == EXPECTED_SLOW_FORWARD_SHA256,
        "SLOW_FORWARD_SHA256_MISMATCH",
    )

    require(
        bool(mixer22.use_mambapy)
        is False,
        "USE_MAMBAPY_NOT_FALSE",
    )

    require(
        hasattr(mixer22, "out_proj"),
        "OUT_PROJ_MISSING",
    )

    out_proj = mixer22.out_proj

    require(
        tuple(out_proj.weight.shape)
        == (Y_WIDTH, V_WIDTH),
        "OUT_PROJ_WEIGHT_SHAPE_MISMATCH",
    )

    require(
        out_proj.weight.dtype
        == torch.float32,
        "OUT_PROJ_WEIGHT_DTYPE_MISMATCH",
    )

    require(
        out_proj.bias is None,
        "OUT_PROJ_BIAS_PRESENT",
    )

    require(
        int(out_proj.in_features)
        == V_WIDTH,
        "OUT_PROJ_IN_FEATURES_MISMATCH",
    )

    require(
        int(out_proj.out_features)
        == Y_WIDTH,
        "OUT_PROJ_OUT_FEATURES_MISMATCH",
    )

    return (
        layer22,
        layer23,
        norm23,
        mixer23,
        mixer22,
        out_proj,
    )


def capture_vy(
    parent: Any,
    rms_parent: Any,
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
    mixer22: Any,
    out_proj: Any,
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    import torch
    import torch.nn.functional as F

    targets = tuple(
        int(i)
        for i in targets
    )

    v_records = {}
    out_records = {}

    counts = {
        "out_pre": 0,
        "out_post": 0,
    }

    def out_pre_hook(module, args):
        counts["out_pre"] += 1

        require(
            counts["out_pre"] == 1,
            "DUPLICATE_OUT_PROJ_PRE_HOOK",
        )

        require(
            len(args) == 1,
            "OUT_PROJ_PRE_ARG_COUNT_MISMATCH",
        )

        full = secant.snapshot(
            args[0],
            "LAYER22_PRE_OUT_PROJ_V22_FULL",
        )

        require(
            len(full.shape) == 3,
            "V22_RANK_MISMATCH",
        )

        require(
            full.shape[0] == 1,
            "V22_BATCH_MISMATCH",
        )

        require(
            full.shape[-1] == V_WIDTH,
            "V22_WIDTH_MISMATCH",
        )

        for i in targets:
            require(
                0 <= i < full.shape[1],
                f"V22_TARGET_OUT_OF_RANGE:{i}",
            )

            v_records[i] = (
                full[:, i, :]
                .contiguous()
                .clone()
            )

    def out_post_hook(
        module,
        args,
        output,
    ):
        counts["out_post"] += 1

        require(
            counts["out_post"] == 1,
            "DUPLICATE_OUT_PROJ_POST_HOOK",
        )

        full = secant.snapshot(
            output,
            "LAYER22_OUT_PROJ_Y22_FULL",
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
            full.shape[-1] == Y_WIDTH,
            "Y22_WIDTH_MISMATCH",
        )

        for i in targets:
            require(
                0 <= i < full.shape[1],
                f"Y22_TARGET_OUT_OF_RANGE:{i}",
            )

            out_records[i] = (
                full[:, i, :]
                .contiguous()
                .clone()
            )

    h_pre = out_proj.register_forward_pre_hook(
        out_pre_hook
    )

    h_post = out_proj.register_forward_hook(
        out_post_hook
    )

    try:
        records = parent.capture_terms(
            rms_parent,
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
            token_ids,
            targets,
        )

    finally:
        h_pre.remove()
        h_post.remove()

    require(
        counts["out_pre"] == 1
        and counts["out_post"] == 1,
        "OUT_PROJ_HOOK_COUNT_FAILURE",
    )

    require(
        set(v_records)
        == set(targets),
        "V22_TARGET_SET_MISMATCH",
    )

    require(
        set(out_records)
        == set(targets),
        "OUT_PROJ_Y22_TARGET_SET_MISMATCH",
    )

    require(
        set(records)
        == set(targets),
        "PARENT_TARGET_SET_MISMATCH",
    )

    weight = (
        out_proj.weight.detach()
        .cpu()
        .contiguous()
    )

    for i in targets:
        v = v_records[i]
        y = out_records[i]

        require(
            tuple(v.shape)
            == (1, V_WIDTH),
            f"V22_SHAPE_MISMATCH:{i}",
        )

        require(
            tuple(y.shape)
            == (1, Y_WIDTH),
            f"Y22_SHAPE_MISMATCH:{i}",
        )

        require(
            "Y22" in records[i],
            f"PARENT_Y22_MISSING:{i}",
        )

        require(
            torch.equal(
                y,
                records[i]["Y22"],
            ),
            f"PARENT_Y22_EXACT_MISMATCH:{i}",
        )

        reconstructed = F.linear(
            v.to(torch.float32),
            weight.to(torch.float32),
            None,
        )

        branch_rel = float(
            torch.linalg.vector_norm(
                (
                    reconstructed
                    - y.to(torch.float32)
                ).to(torch.float64)
            ).item()
            / max(
                torch.linalg.vector_norm(
                    y.to(torch.float64)
                ).item(),
                1e-12,
            )
        )

        require(
            branch_rel
            <= FLOAT32_BRANCH_RECON_REL_TOL,
            (
                "FLOAT32_OUT_PROJ_BRANCH_RECON_FAILURE:"
                f"{i}:{branch_rel}"
            ),
        )

        records[i]["V22"] = v
        records[i]["OUT_PROJ_Y22"] = y

        records[i][
            "float32_branch_reconstruction_relative_residual"
        ] = branch_rel

    return records


def metric_rows_for_pair(
    row: Mapping[str, Any],
    matched: Mapping[int, Mapping[str, Any]],
    swapped: Mapping[int, Mapping[str, Any]],
    weight: Any,
    cohort: frozenset[int],
    signature: str,
):
    import torch
    import torch.nn.functional as F

    idx = int(
        row["local_template_index"]
    )

    role = str(
        row["role"]
    )

    anchor = int(
        row["anchor"]
    )

    w64 = weight.to(
        torch.float64
    )

    output = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k

        vm = (
            matched[token]["V22"]
            .to(torch.float64)
        )

        vs = (
            swapped[token]["V22"]
            .to(torch.float64)
        )

        ym = (
            matched[token]["OUT_PROJ_Y22"]
            .to(torch.float64)
        )

        ys = (
            swapped[token]["OUT_PROJ_Y22"]
            .to(torch.float64)
        )

        dv = vm - vs
        dy = ym - ys

        projected_dv = F.linear(
            dv,
            w64,
            None,
        )

        dv_l2 = float(
            torch.linalg.vector_norm(
                dv
            ).item()
        )

        dy_l2 = float(
            torch.linalg.vector_norm(
                dy
            ).item()
        )

        projected_l2 = float(
            torch.linalg.vector_norm(
                projected_dv
            ).item()
        )

        recon_rel = float(
            torch.linalg.vector_norm(
                projected_dv - dy
            ).item()
            / max(
                dy_l2,
                1e-12,
            )
        )

        require(
            recon_rel
            <= OBSERVED_LINEAR_RECON_REL_TOL,
            (
                "OBSERVED_FLOAT32_OUT_PROJ_RECON_FAILURE:"
                f"{idx}:{role}:{k}:{recon_rel}"
            ),
        )

        transfer = (
            projected_l2 / dv_l2
            if dv_l2 > 0.0
            else 0.0
        )

        if dv_l2 > 0.0:
            require(
                math.isclose(
                    projected_l2,
                    dv_l2 * transfer,
                    rel_tol=1e-15,
                    abs_tol=1e-15,
                ),
                (
                    "OUTPUT_PROJECTION_SCALAR_IDENTITY_FAILURE:"
                    f"{idx}:{role}:{k}"
                ),
            )

        if k == -1:
            require(
                dv_l2 == 0.0,
                f"K_MINUS_1_DV22_NONZERO:{idx}:{role}",
            )

            require(
                dy_l2 == 0.0,
                f"K_MINUS_1_DY22_NONZERO:{idx}:{role}",
            )

            require(
                projected_l2 == 0.0,
                f"K_MINUS_1_PROJECTED_NONZERO:{idx}:{role}",
            )

        values = {
            "schema_version":
                "k0-rvg-layer22-output-projection-transfer-row-v1",

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

            "delta_v22_l2":
                dv_l2,

            "delta_y22_l2":
                dy_l2,

            "projected_delta_v22_l2":
                projected_l2,

            "output_projection_transfer":
                transfer,

            "observed_linear_reconstruction_relative_residual":
                recon_rel,

            "float32_branch_reconstruction_relative_residual":
                max(
                    float(
                        matched[token][
                            "float32_branch_reconstruction_relative_residual"
                        ]
                    ),
                    float(
                        swapped[token][
                            "float32_branch_reconstruction_relative_residual"
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
    "delta_v22_l2",
    "delta_y22_l2",
    "projected_delta_v22_l2",
    "output_projection_transfer",
    "observed_linear_reconstruction_relative_residual",
    "float32_branch_reconstruction_relative_residual",
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
            "k0-rvg-layer22-output-projection-transfer-summary-v1",

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

        "pre_projection_width":
            V_WIDTH,

        "output_width":
            Y_WIDTH,

        "relative_coordinates":
            list(RELATIVE_COORDINATES),

        "execution_protocol":
            EXECUTION_PROTOCOL,

        "identity":
            "delta_Y22 = W_O delta_V22",

        "full_336_trajectory":
            aggregate_trajectory(rows),

        "common_330_ddsssss_trajectory":
            aggregate_trajectory(
                common_rows
            ),

        "max_observed_linear_reconstruction_relative_residual":
            max(
                float(
                    row[
                        "observed_linear_reconstruction_relative_residual"
                    ]
                )
                for row in rows
            ),

        "max_float32_branch_reconstruction_relative_residual":
            max(
                float(
                    row[
                        "float32_branch_reconstruction_relative_residual"
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
                    ["delta_y22_l2"]["median"]
                )

                expected = float(
                    parent[role][str(k)]
                    ["delta_y22_l2"]["median"]
                )

                require(
                    math.isclose(
                        got,
                        expected,
                        rel_tol=1e-13,
                        abs_tol=1e-13,
                    ),
                    (
                        "PARENT_DELTA_Y22_REPRODUCTION_FAILURE:"
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
    rms_parent,
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
        layer22,
        layer23,
        norm23,
        mixer23,
        mixer22,
        out_proj,
    ) = resolve_output_projection(
        parent,
        rms_parent,
        hidden_parent,
        postconv,
        model,
        layer_map,
    )

    weight = (
        out_proj.weight.detach()
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

        require(
            len(matched_prefix)
            == cutoff
            and len(swapped_prefix)
            == cutoff,
            f"PREFIX_LENGTH_FAILURE:{idx}:{role}",
        )

        matched = capture_vy(
            parent,
            rms_parent,
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
            mixer22,
            out_proj,
            matched_prefix,
            row["targets"],
        )

        forward_count += 1

        swapped = capture_vy(
            parent,
            rms_parent,
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
            mixer22,
            out_proj,
            swapped_prefix,
            row["targets"],
        )

        forward_count += 1

        rows.extend(
            metric_rows_for_pair(
                row,
                matched,
                swapped,
                weight,
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
        / "layer22_output_projection_transfer_metrics.jsonl"
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
            "k0-rvg-layer22-output-projection-transfer-execution-manifest-v1",

        "runtime_git_head":
            repo["head"],

        "runtime_branch":
            repo["branch"],

        "parent_freeze_commit":
            PARENT_FREEZE_COMMIT,

        "scientific_question":
            QUESTION,

        "identity":
            "delta_Y22 = W_O delta_V22",

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

        "pre_projection_width":
            V_WIDTH,

        "output_width":
            Y_WIDTH,

        "mixer_forward_sha256":
            EXPECTED_MIXER_FORWARD_SHA256,

        "slow_forward_sha256":
            EXPECTED_SLOW_FORWARD_SHA256,

        "mamba_source_sha256":
            binding.source_sha256,

        "out_proj_weight_shape":
            [Y_WIDTH, V_WIDTH],

        "out_proj_bias_present":
            False,

        "handoff_zip_sha256":
            handoff["zip_sha256"],

        "checkpoint_sha256":
            handoff["checkpoint_sha256"],

        "encoder_canonical_digest":
            encoder["canonical_digest"],

        "encoder_raw_concat_digest":
            encoder["raw_concat_digest"],

        "parent_delta_y22_summary_match":
            True,

        "raw_vectors_persisted":
            False,

        "scientific_model_forward_executed":
            True,

        "scientific_pre_out_proj_v22_read":
            True,

        "scientific_y22_read":
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
            "layer22_output_projection_transfer_metrics.jsonl":
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
        "PASS_LAYER22_OUTPUT_PROJECTION_TRANSFER_EXECUTION"
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
        "parent_delta_y22_summary_match = True"
    )

    print(
        "max_observed_linear_reconstruction_relative_residual =",
        summary[
            "max_observed_linear_reconstruction_relative_residual"
        ],
    )

    print(
        "max_float32_branch_reconstruction_relative_residual =",
        summary[
            "max_float32_branch_reconstruction_relative_residual"
        ],
    )

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[
                role
            ][str(k)]

            print(
                f"{role}_k{k}_median "
                f"V22={t['delta_v22_l2']['median']} "
                f"Y22={t['delta_y22_l2']['median']} "
                f"PROJ={t['projected_delta_v22_l2']['median']} "
                f"TRANSFER={t['output_projection_transfer']['median']}"
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
        rms_parent,
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
        rms_parent,
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
        "=== LAYER22 OUTPUT PROJECTION TRANSFER AUDIT PLAN ==="
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
        "scientific_question =",
        QUESTION,
    )

    print(
        "identity = "
        "delta_Y22 = W_O delta_V22"
    )

    print(
        "mixer_forward_sha256 =",
        EXPECTED_MIXER_FORWARD_SHA256,
    )

    print(
        "slow_forward_sha256 =",
        EXPECTED_SLOW_FORWARD_SHA256,
    )

    print(
        "mamba_source_sha256 =",
        source["source_sha256"],
    )

    print(
        "out_proj_weight_shape =",
        (Y_WIDTH, V_WIDTH),
    )

    print(
        "out_proj_bias_present = False"
    )

    print(
        "float32_branch_reconstruction_rel_tol =",
        FLOAT32_BRANCH_RECON_REL_TOL,
    )

    print(
        "observed_linear_reconstruction_rel_tol =",
        OBSERVED_LINEAR_RECON_REL_TOL,
    )

    print(
        "parent_delta_y22_summary_preserved_by_design = True"
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
            "PASS_LAYER22_OUTPUT_PROJECTION_TRANSFER_STATIC_PREFLIGHT"
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
        rms_parent,
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

    except OutputProjectionAuditError as exc:
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