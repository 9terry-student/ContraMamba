"""K0-RVG layer-23 hidden-branch in-projection transfer audit.

Exact authenticated boundary:

    H_t = W_H X_t

where:
- X_t is the layer-23 mixer input state, width 768;
- W_H is the first 1536 rows of the bias-free in_proj weight;
- H_t is the current-token pre-convolution hidden branch.

Difference identity:

    delta_H_t = W_H delta_X_t

Exact scalar magnitude factorization:

    ||delta_H_t||
      = ||delta_X_t||
        * (||delta_H_t|| / ||delta_X_t||)

The second factor is the direction-conditioned transfer through the fixed
hidden-branch in-projection.

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
    "5305afe5369f4952aa50e63c0fb381c30f577cd4"
)

PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_four_tap_convolution_decomposition_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "49cd2708bf7ba3ef9f6beb5987f845d585cb9f991e459939c86cb303e0ce5070"
)

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_four_tap_convolution_ef8f4c1_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "0ad1368107c07b7f0bc2b2a23bf4694da3ff68efaeac77690c19d39f3a038520"
)

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FORWARD_COUNT = 1344

PRIMARY_LAYER = 23
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 1536

RELATIVE_COORDINATES = tuple(range(-1, 7))
POST_HORIZON = 6

EXECUTION_PROTOCOL = (
    "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"
)

INPROJ_RECON_REL_TOL = 1e-6

QUESTION = (
    "Is the layer-23 current-token pre-convolution delta-H magnitude "
    "pattern already present in the upstream layer-input delta-X "
    "magnitude, or is it substantially reshaped by direction-conditioned "
    "transfer through the fixed bias-free hidden-branch in-projection?"
)


class HiddenInProjAuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise HiddenInProjAuditError(message)


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


def load_four_tap(root: Path):
    path = root / PARENT_RUNNER_REL

    require(
        sha256_bytes(path.read_bytes())
        == PARENT_RUNNER_SHA256,
        "PARENT_RUNNER_SHA256_MISMATCH",
    )

    return import_module(
        path,
        "k0_rvg_four_tap_parent",
    )


def build_plan(root: Path, four_tap: Any):
    postconv = four_tap.load_parent_runner(
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
    ) = four_tap.build_plan(
        root,
        postconv,
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
    four_tap: Any,
    postconv: Any,
    time_step: Any,
    base: Any,
):
    # Authenticate the full older chain first.
    four_tap.authenticate_parent(
        root,
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
        == "k0-rvg-four-tap-convolution-summary-v1",
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


def resolve_hidden_inproj(
    postconv: Any,
    model: Any,
    layer_map: Mapping[int, int],
):
    import torch

    mixer = postconv.resolve_primary_mixer(
        model,
        layer_map,
    )

    require(
        isinstance(
            mixer.in_proj,
            torch.nn.Linear,
        ),
        "IN_PROJ_NOT_LINEAR",
    )

    require(
        int(mixer.hidden_size)
        == HIDDEN_SIZE,
        "HIDDEN_SIZE_MISMATCH",
    )

    require(
        int(mixer.intermediate_size)
        == INTERMEDIATE_SIZE,
        "INTERMEDIATE_SIZE_MISMATCH",
    )

    require(
        tuple(
            mixer.in_proj.weight.shape
        )
        == (
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        ),
        "IN_PROJ_WEIGHT_SHAPE_MISMATCH",
    )

    require(
        mixer.in_proj.bias is None,
        "IN_PROJ_BIAS_PRESENT",
    )

    w_hidden = (
        mixer.in_proj.weight[
            :INTERMEDIATE_SIZE,
            :
        ]
        .detach()
        .cpu()
        .contiguous()
        .clone()
    )

    require(
        tuple(w_hidden.shape)
        == (
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        ),
        "HIDDEN_WEIGHT_SHAPE_MISMATCH",
    )

    return mixer, w_hidden


def capture_xh(
    four_tap: Any,
    postconv: Any,
    dt_projection: Any,
    secant: Any,
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    mixer: Any,
    w_hidden: Any,
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    import torch

    targets = tuple(
        int(i)
        for i in targets
    )

    x_records = {}

    hook_count = 0

    def inproj_pre_hook(module, args):
        nonlocal hook_count

        hook_count += 1

        require(
            hook_count == 1,
            "DUPLICATE_INPROJ_PRE_HOOK",
        )

        require(
            len(args) == 1,
            "INPROJ_PRE_HOOK_ARG_COUNT_MISMATCH",
        )

        x_full = secant.snapshot(
            args[0],
            "LAYER23_INPUT_X_FULL",
        )

        require(
            len(x_full.shape) == 3,
            "LAYER23_X_RANK_MISMATCH",
        )

        require(
            x_full.shape[0] == 1,
            "LAYER23_X_BATCH_MISMATCH",
        )

        require(
            x_full.shape[2]
            == HIDDEN_SIZE,
            "LAYER23_X_WIDTH_MISMATCH",
        )

        for i in targets:
            require(
                0 <= i < x_full.shape[1],
                f"LAYER23_X_TARGET_OUT_OF_RANGE:{i}",
            )

            x_records[i] = (
                x_full[:, i, :]
                .contiguous()
                .clone()
            )

    handle = (
        mixer.in_proj
        .register_forward_pre_hook(
            inproj_pre_hook
        )
    )

    try:
        records = four_tap.capture(
            postconv,
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

    finally:
        handle.remove()

    require(
        hook_count == 1,
        "INPROJ_PRE_HOOK_COUNT_FAILURE",
    )

    require(
        set(x_records)
        == set(targets),
        "LAYER23_X_TARGET_SET_MISMATCH",
    )

    require(
        set(records)
        == set(targets),
        "PARENT_TARGET_SET_MISMATCH",
    )

    for i in targets:
        x = x_records[i]

        require(
            tuple(x.shape)
            == (1, HIDDEN_SIZE),
            f"LAYER23_X_SHAPE_MISMATCH:{i}",
        )

        require(
            "H_RF" in records[i],
            f"PARENT_H_RF_MISSING:{i}",
        )

        h_rf = records[i]["H_RF"]

        require(
            tuple(h_rf.shape)
            == (4, INTERMEDIATE_SIZE),
            f"PARENT_H_RF_SHAPE_MISMATCH:{i}",
        )

        h_current = (
            h_rf[0, :]
            .unsqueeze(0)
            .contiguous()
        )

        with torch.inference_mode():
            reconstructed = (
                torch.nn.functional.linear(
                    x,
                    w_hidden,
                    bias=None,
                )
                .detach()
                .cpu()
                .contiguous()
            )

        residual = (
            reconstructed
            - h_current
        )

        rel = float(
            torch.linalg.vector_norm(
                residual.to(torch.float64)
            ).item()
            / max(
                torch.linalg.vector_norm(
                    h_current.to(torch.float64)
                ).item(),
                1e-12,
            )
        )

        require(
            rel <= INPROJ_RECON_REL_TOL,
            (
                "HIDDEN_INPROJ_RECONSTRUCTION_FAILURE:"
                f"{i}:{rel}"
            ),
        )

        records[i]["X"] = x

        records[i][
            "hidden_inproj_reconstruction_relative_residual"
        ] = rel

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

        xm = matched[token]["X"]
        xs = swapped[token]["X"]

        hm = (
            matched[token]["H_RF"][0, :]
            .unsqueeze(0)
        )

        hs = (
            swapped[token]["H_RF"][0, :]
            .unsqueeze(0)
        )

        require(
            tuple(xm.shape)
            == (1, HIDDEN_SIZE),
            f"XM_SHAPE_MISMATCH:{idx}:{role}:{k}",
        )

        require(
            tuple(xs.shape)
            == (1, HIDDEN_SIZE),
            f"XS_SHAPE_MISMATCH:{idx}:{role}:{k}",
        )

        require(
            tuple(hm.shape)
            == (1, INTERMEDIATE_SIZE),
            f"HM_SHAPE_MISMATCH:{idx}:{role}:{k}",
        )

        require(
            tuple(hs.shape)
            == (1, INTERMEDIATE_SIZE),
            f"HS_SHAPE_MISMATCH:{idx}:{role}:{k}",
        )

        if k == -1:
            require(
                torch.equal(xm, xs),
                f"K_MINUS_1_X_IDENTITY_FAILURE:{idx}:{role}",
            )

            require(
                torch.equal(hm, hs),
                f"K_MINUS_1_H_IDENTITY_FAILURE:{idx}:{role}",
            )

        dx = (
            xm.to(torch.float64)
            - xs.to(torch.float64)
        )

        dh = (
            hm.to(torch.float64)
            - hs.to(torch.float64)
        )

        dx_l2 = float(
            torch.linalg.vector_norm(
                dx
            ).item()
        )

        dh_l2 = float(
            torch.linalg.vector_norm(
                dh
            ).item()
        )

        recon_rel = max(
            float(
                matched[token][
                    "hidden_inproj_reconstruction_relative_residual"
                ]
            ),
            float(
                swapped[token][
                    "hidden_inproj_reconstruction_relative_residual"
                ]
            ),
        )

        if k == -1:
            require(
                dx_l2 == 0.0,
                f"K_MINUS_1_DELTA_X_NONZERO:{idx}:{role}",
            )

            require(
                dh_l2 == 0.0,
                f"K_MINUS_1_DELTA_H_NONZERO:{idx}:{role}",
            )

            transfer = 0.0

        else:
            require(
                dx_l2 > 0.0,
                f"ZERO_DELTA_X_AFTER_ANCHOR:{idx}:{role}:{k}",
            )

            require(
                dh_l2 > 0.0,
                f"ZERO_DELTA_H_AFTER_ANCHOR:{idx}:{role}:{k}",
            )

            transfer = (
                dh_l2
                / dx_l2
            )

            require(
                math.isclose(
                    dh_l2,
                    dx_l2 * transfer,
                    rel_tol=1e-13,
                    abs_tol=1e-13,
                ),
                (
                    "INPROJ_MAGNITUDE_FACTORIZATION_FAILURE:"
                    f"{idx}:{role}:{k}"
                ),
            )

        values = {
            "schema_version":
                "k0-rvg-hidden-inproj-transfer-row-v1",

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

            "delta_x_l2":
                dx_l2,

            "delta_h_l2":
                dh_l2,

            "hidden_inproj_transfer":
                transfer,

            "hidden_inproj_reconstruction_relative_residual":
                recon_rel,

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

        output.append(values)

    return output


SUMMARY_FIELDS = (
    "delta_x_l2",
    "delta_h_l2",
    "hidden_inproj_transfer",
    "hidden_inproj_reconstruction_relative_residual",
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

    post_rows = [
        row
        for row in rows
        if int(
            row["relative_coordinate"]
        ) != -1
    ]

    return {
        "schema_version":
            "k0-rvg-hidden-inproj-transfer-summary-v1",

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

        "intermediate_size":
            INTERMEDIATE_SIZE,

        "relative_coordinates":
            list(RELATIVE_COORDINATES),

        "execution_protocol":
            EXECUTION_PROTOCOL,

        "identity":
            "delta_H_t = W_H delta_X_t",

        "magnitude_factorization":
            (
                "||delta_H|| = ||delta_X|| * "
                "(||delta_H||/||delta_X||)"
            ),

        "full_336_trajectory":
            aggregate_trajectory(rows),

        "common_330_ddsssss_trajectory":
            aggregate_trajectory(
                common_rows
            ),

        "max_hidden_inproj_reconstruction_relative_residual":
            max(
                float(
                    row[
                        "hidden_inproj_reconstruction_relative_residual"
                    ]
                )
                for row in post_rows
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
                    ["delta_h_l2"]["median"]
                )

                expected = float(
                    parent[role][str(k)]
                    ["delta_h_lag0_l2"]["median"]
                )

                require(
                    math.isclose(
                        got,
                        expected,
                        rel_tol=1e-13,
                        abs_tol=1e-13,
                    ),
                    (
                        "PARENT_DELTA_H_REPRODUCTION_FAILURE:"
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

    require(
        model.training is False,
        "MODEL_NOT_EVAL",
    )

    mixer, w_hidden = resolve_hidden_inproj(
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

        matched = capture_xh(
            four_tap,
            postconv,
            dt_projection,
            secant,
            base,
            model,
            binding,
            layer_map,
            mixer,
            w_hidden,
            matched_prefix,
            row["targets"],
        )

        forward_count += 1

        swapped = capture_xh(
            four_tap,
            postconv,
            dt_projection,
            secant,
            base,
            model,
            binding,
            layer_map,
            mixer,
            w_hidden,
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
        / "hidden_inproj_transfer_metrics.jsonl"
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
            "k0-rvg-hidden-inproj-transfer-execution-manifest-v1",

        "runtime_git_head":
            repo["head"],

        "runtime_branch":
            repo["branch"],

        "parent_freeze_commit":
            PARENT_FREEZE_COMMIT,

        "scientific_question":
            QUESTION,

        "identity":
            "delta_H_t = W_H delta_X_t",

        "magnitude_factorization":
            (
                "||delta_H|| = ||delta_X|| * "
                "hidden_inproj_transfer"
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

        "intermediate_size":
            INTERMEDIATE_SIZE,

        "in_proj_weight_shape":
            list(
                mixer.in_proj.weight.shape
            ),

        "in_proj_bias_present":
            mixer.in_proj.bias is not None,

        "hidden_branch_weight_shape":
            list(
                w_hidden.shape
            ),

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

        "parent_delta_h_summary_match":
            True,

        "raw_vectors_persisted":
            False,

        "scientific_model_forward_executed":
            True,

        "scientific_layer23_input_x_read":
            True,

        "scientific_preconv_current_h_read":
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
            "hidden_inproj_transfer_metrics.jsonl":
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
        "PASS_HIDDEN_INPROJ_TRANSFER_EXECUTION"
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
        "parent_delta_h_summary_match = True"
    )

    print(
        "max_hidden_inproj_reconstruction_relative_residual =",
        summary[
            "max_hidden_inproj_reconstruction_relative_residual"
        ],
    )

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[
                role
            ][str(k)]

            print(
                f"{role}_k{k}_median "
                f"DX={t['delta_x_l2']['median']} "
                f"DH={t['delta_h_l2']['median']} "
                f"INPROJ_TR="
                f"{t['hidden_inproj_transfer']['median']}"
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

    four_tap = load_four_tap(
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
    ) = build_plan(
        root,
        four_tap,
    )

    repo, parent_summary = authenticate_parent(
        root,
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

    direct = (
        postconv
        .validate_direct_forward_contract(
            base
        )
    )

    print(
        "=== HIDDEN IN-PROJECTION AUDIT PLAN ==="
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
        "delta_H_t = W_H delta_X_t"
    )

    print(
        "magnitude_factorization = "
        "||delta_H|| = ||delta_X|| * "
        "hidden_inproj_transfer"
    )

    print(
        "hidden_branch_weight_shape = "
        "(1536, 768)"
    )

    print(
        "in_proj_bias_expected = False"
    )

    print(
        "attention_mask_passed =",
        direct["attention_mask_passed"],
    )

    print(
        "use_cache =",
        direct["use_cache"],
    )

    print(
        "mamba_source_sha256 =",
        source["source_sha256"],
    )

    print(
        "parent_delta_h_summary_preserved_by_design = True"
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
            "PASS_HIDDEN_INPROJ_TRANSFER_STATIC_PREFLIGHT"
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

    except HiddenInProjAuditError as exc:
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