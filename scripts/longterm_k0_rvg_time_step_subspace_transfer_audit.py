"""K0-RVG x_proj time-step subspace transfer audit.

Exact frozen source identity at layer 23:

    r_t = W_r U_t
    delta_r = W_r delta_U

where:
    U_t   : post-convolution/activation Mamba hidden vector, width 1536
    W_r   : first 48 rows of bias-free x_proj, shape (48, 1536)
    r_t   : low-rank time-step vector, width 48

Magnitude factorization:

    ||delta_r||
      = ||delta_U||
        * (||W_r delta_U|| / ||delta_U||)

The audit separates upstream delta-U magnitude from total transfer through
the fixed x_proj time-step map.

This is observational/algebraic. No intervention, training, probe, PCA,
tokenizer execution, logits, or task heads.
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
    "a9f3f8023ec168c5f75ce8b3df4bdbee32eb7109"
)

PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_dt_projection_geometry_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "4182f7499215f51275333565f52b6cca42fef5e880cab6bccb515d516069217a"
)

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_dt_projection_geometry_705094d_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "eb407ed465fb149186ea4d70253dc47a4105553818a970ad4343aa6824310b22"
)

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FORWARD_COUNT = 1344

PRIMARY_LAYER = 23
TIME_STEP_RANK = 48
INTERMEDIATE_SIZE = 1536
SSM_STATE_SIZE = 16

RELATIVE_COORDINATES = tuple(range(-1, 7))
POST_HORIZON = 6

EXECUTION_PROTOCOL = (
    "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"
)

PROJECTION_REL_TOL = 2e-5

QUESTION = (
    "Is the corr-k2 low-rank delta-r magnitude already present as a "
    "larger upstream post-convolution delta-U magnitude, or does it arise "
    "from stronger total transfer of that delta-U direction through the "
    "fixed bias-free x_proj time-step map W_r?"
)


class TimeStepSubspaceTransferError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise TimeStepSubspaceTransferError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def import_module(path: Path, name: str):
    require(path.is_file(), f"MODULE_MISSING:{path}")

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
        "k0_rvg_dt_projection_geometry_parent",
    )


def build_plan(root: Path, parent: Any):
    upstream = parent.load_parent_runner(root)

    (
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
        upstream,
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
        upstream,
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
    base: Any,
):
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
        == "k0-rvg-dt-projection-geometry-summary-v1",
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

    require(
        summary.get("time_step_rank")
        == TIME_STEP_RANK,
        "PARENT_TIME_STEP_RANK_MISMATCH",
    )

    require(
        summary.get("primary_layer")
        == PRIMARY_LAYER,
        "PARENT_LAYER_MISMATCH",
    )

    return repo, summary


def resolve_time_step_map(
    secant: Any,
    model: Any,
    layer_map: Mapping[int, int],
):
    import torch

    mixers = [
        module
        for module in model.modules()
        if layer_map.get(id(module))
        == PRIMARY_LAYER
    ]

    require(
        len(mixers) == 1,
        "PRIMARY_MIXER_RESOLUTION_FAILURE",
    )

    mixer = mixers[0]

    require(
        int(mixer.intermediate_size)
        == INTERMEDIATE_SIZE,
        "INTERMEDIATE_SIZE_MISMATCH",
    )

    require(
        int(mixer.time_step_rank)
        == TIME_STEP_RANK,
        "TIME_STEP_RANK_MISMATCH",
    )

    require(
        int(mixer.ssm_state_size)
        == SSM_STATE_SIZE,
        "SSM_STATE_SIZE_MISMATCH",
    )

    x_proj = getattr(
        mixer,
        "x_proj",
        None,
    )

    require(
        isinstance(x_proj, torch.nn.Linear),
        "X_PROJ_NOT_LINEAR",
    )

    require(
        x_proj.bias is None,
        "X_PROJ_BIAS_PRESENT",
    )

    weight = secant.snapshot(
        x_proj.weight,
        "X_PROJ_WEIGHT",
    )

    require(
        tuple(weight.shape)
        == (
            TIME_STEP_RANK
            + 2 * SSM_STATE_SIZE,
            INTERMEDIATE_SIZE,
        ),
        "X_PROJ_WEIGHT_SHAPE_MISMATCH",
    )

    W_r = (
        weight[:TIME_STEP_RANK, :]
        .contiguous()
        .clone()
    )

    require(
        tuple(W_r.shape)
        == (
            TIME_STEP_RANK,
            INTERMEDIATE_SIZE,
        ),
        "W_R_SHAPE_MISMATCH",
    )

    return {
        "W_r": W_r,
        "x_proj_weight_shape":
            list(weight.shape),
        "W_r_shape":
            list(W_r.shape),
    }


def metric_rows_for_pair(
    parent: Any,
    secant: Any,
    geometry: Mapping[str, Any],
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

    W_r32 = geometry["W_r"]
    W_r = W_r32.to(torch.float64)

    out = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k

        Um32 = matched[token]["U"]
        Rm32 = matched[token]["R"]

        Us32 = swapped[token]["U"]
        Rs32 = swapped[token]["R"]

        require(
            tuple(Um32.shape)
            == (1, INTERMEDIATE_SIZE),
            f"U_SHAPE_MISMATCH:{idx}:{role}:{k}",
        )

        require(
            tuple(Rm32.shape)
            == (1, TIME_STEP_RANK),
            f"R_SHAPE_MISMATCH:{idx}:{role}:{k}",
        )

        if k == -1:
            require(
                secant.torch_equal(
                    Um32,
                    Us32,
                ),
                f"K_MINUS_1_U_IDENTITY_FAILURE:{idx}:{role}",
            )

            require(
                secant.torch_equal(
                    Rm32,
                    Rs32,
                ),
                f"K_MINUS_1_R_IDENTITY_FAILURE:{idx}:{role}",
            )

        Um = Um32.to(torch.float64)
        Us = Us32.to(torch.float64)

        Rm = Rm32.to(torch.float64)
        Rs = Rs32.to(torch.float64)

        delta_u = Um - Us
        delta_r = Rm - Rs

        projected_delta_r = torch.matmul(
            delta_u,
            W_r.transpose(0, 1),
        )

        residual = (
            delta_r
            - projected_delta_r
        )

        delta_u_l2 = float(
            torch.linalg.vector_norm(
                delta_u
            ).item()
        )

        delta_r_l2 = float(
            torch.linalg.vector_norm(
                delta_r
            ).item()
        )

        projected_delta_r_l2 = float(
            torch.linalg.vector_norm(
                projected_delta_r
            ).item()
        )

        residual_l2 = float(
            torch.linalg.vector_norm(
                residual
            ).item()
        )

        projection_relative_residual = (
            residual_l2
            / max(delta_r_l2, 1e-12)
        )

        if k == -1:
            require(
                delta_u_l2 == 0.0,
                f"K_MINUS_1_DELTA_U_NONZERO:{idx}:{role}",
            )

            require(
                delta_r_l2 == 0.0,
                f"K_MINUS_1_DELTA_R_NONZERO:{idx}:{role}",
            )

            transfer = 0.0

        else:
            require(
                delta_u_l2 > 0.0,
                f"ZERO_DELTA_U_AFTER_ANCHOR:{idx}:{role}:{k}",
            )

            require(
                delta_r_l2 > 0.0,
                f"ZERO_DELTA_R_AFTER_ANCHOR:{idx}:{role}:{k}",
            )

            require(
                projection_relative_residual
                <= PROJECTION_REL_TOL,
                (
                    "X_PROJ_TIME_STEP_RECONSTRUCTION_FAILURE:"
                    f"{idx}:{role}:{k}:"
                    f"{projection_relative_residual}"
                ),
            )

            transfer = (
                delta_r_l2
                / delta_u_l2
            )

        values = {
            "schema_version":
                "k0-rvg-time-step-subspace-transfer-row-v1",

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

            "delta_u_l2":
                delta_u_l2,

            "delta_r_l2":
                delta_r_l2,

            "projected_delta_r_l2":
                projected_delta_r_l2,

            "time_step_subspace_transfer":
                transfer,

            "projection_relative_residual":
                projection_relative_residual,

            "source_snapshot_dtype":
                "torch.float32",

            "metric_accumulation_dtype":
                "torch.float64",
        }

        require(
            all(
                math.isfinite(float(v))
                for v in values.values()
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
            ),
            f"NONFINITE_METRIC:{idx}:{role}:{k}",
        )

        out.append(values)

    return out


SUMMARY_FIELDS = (
    "delta_u_l2",
    "delta_r_l2",
    "projected_delta_r_l2",
    "time_step_subspace_transfer",
    "projection_relative_residual",
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
        "mean":
            float(statistics.fmean(vals)),
        "median":
            float(statistics.median(vals)),
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
                    r["relative_coordinate"]
                ) == k
            ]

            require(
                bool(bucket),
                f"EMPTY_BUCKET:{role}:{k}",
            )

            out[role][str(k)] = {
                field: aggregate(
                    r[field]
                    for r in bucket
                )
                for field in SUMMARY_FIELDS
            }

    return out


def make_summary(
    rows,
    cohort,
    geometry,
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
            "k0-rvg-time-step-subspace-transfer-summary-v1",

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

        "time_step_rank":
            TIME_STEP_RANK,

        "intermediate_size":
            INTERMEDIATE_SIZE,

        "x_proj_weight_shape":
            geometry[
                "x_proj_weight_shape"
            ],

        "W_r_shape":
            geometry["W_r_shape"],

        "relative_coordinates":
            list(RELATIVE_COORDINATES),

        "execution_protocol":
            EXECUTION_PROTOCOL,

        "exact_identity":
            "delta_r = W_r * delta_U",

        "magnitude_factorization":
            (
                "||delta_r|| = ||delta_U|| * "
                "(||W_r delta_U|| / ||delta_U||)"
            ),

        "full_336_trajectory":
            aggregate_trajectory(rows),

        "common_330_ddsssss_trajectory":
            aggregate_trajectory(
                common_rows
            ),

        "max_projection_relative_residual":
            max(
                float(
                    r[
                        "projection_relative_residual"
                    ]
                )
                for r in rows
                if int(
                    r[
                        "relative_coordinate"
                    ]
                ) != -1
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
                    ["delta_r_l2"]["median"]
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
                        "PARENT_DELTA_R_REPRODUCTION_FAILURE:"
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

    geometry = resolve_time_step_map(
        secant,
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

        matched = parent.capture_factors(
            secant,
            base,
            model,
            binding,
            layer_map,
            matched_prefix,
            row["targets"],
        )

        forward_count += 1

        swapped = parent.capture_factors(
            secant,
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
                parent,
                secant,
                geometry,
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
        geometry,
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
        / "time_step_subspace_transfer_metrics.jsonl"
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
            "k0-rvg-time-step-subspace-transfer-execution-manifest-v1",

        "runtime_git_head":
            repo["head"],

        "runtime_branch":
            repo["branch"],

        "parent_freeze_commit":
            PARENT_FREEZE_COMMIT,

        "scientific_question":
            QUESTION,

        "exact_identity":
            "delta_r = W_r * delta_U",

        "magnitude_factorization":
            (
                "||delta_r|| = ||delta_U|| * "
                "time_step_subspace_transfer"
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

        "intermediate_size":
            INTERMEDIATE_SIZE,

        "time_step_rank":
            TIME_STEP_RANK,

        "x_proj_weight_shape":
            geometry[
                "x_proj_weight_shape"
            ],

        "W_r_shape":
            geometry["W_r_shape"],

        "relative_coordinates":
            list(RELATIVE_COORDINATES),

        "mamba_source_sha256":
            binding.source_sha256,

        "handoff_zip_sha256":
            handoff["zip_sha256"],

        "checkpoint_sha256":
            handoff[
                "checkpoint_sha256"
            ],

        "encoder_canonical_digest":
            encoder[
                "canonical_digest"
            ],

        "encoder_raw_concat_digest":
            encoder[
                "raw_concat_digest"
            ],

        "parent_delta_r_summary_match":
            True,

        "raw_vectors_persisted":
            False,

        "scientific_model_forward_executed":
            True,

        "scientific_post_conv_U_read":
            True,

        "scientific_low_rank_time_step_read":
            True,

        "scientific_x_proj_time_step_map_read":
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
            "time_step_subspace_transfer_metrics.jsonl":
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
        "PASS_TIME_STEP_SUBSPACE_TRANSFER_EXECUTION"
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
        "parent_delta_r_summary_match = True"
    )

    print(
        "max_projection_relative_residual =",
        summary[
            "max_projection_relative_residual"
        ],
    )

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[
                role
            ][str(k)]

            print(
                f"{role}_k{k}_median "
                f"DU_L2="
                f"{t['delta_u_l2']['median']} "
                f"DR_L2="
                f"{t['delta_r_l2']['median']} "
                f"TRANSFER="
                f"{t['time_step_subspace_transfer']['median']}"
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
        upstream,
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
        base,
    )

    source = secant.validate_frozen_source_semantics()

    require(
        source["source_sha256"]
        == EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    require(
        source["pre_softplus_line"]
        == 318,
        "PRE_SOFTPLUS_LINE_MISMATCH",
    )

    print(
        "=== TIME-STEP SUBSPACE TRANSFER AUDIT PLAN ==="
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
        "exact_identity = "
        "delta_r = W_r * delta_U"
    )

    print(
        "magnitude_factorization = "
        "||delta_r|| = ||delta_U|| * "
        "(||W_r delta_U|| / ||delta_U||)"
    )

    print(
        "expected_W_r_shape =",
        (TIME_STEP_RANK, INTERMEDIATE_SIZE),
    )

    print(
        "mamba_source_sha256 =",
        source["source_sha256"],
    )

    print(
        "parent_delta_r_summary_preserved_by_design = True"
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
            "PASS_TIME_STEP_SUBSPACE_TRANSFER_STATIC_PREFLIGHT"
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
        raise SystemExit(main())

    except TimeStepSubspaceTransferError as exc:
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