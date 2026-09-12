"""K0-RVG post-convolution U factorization audit.

Authenticated layer-23 slow-path boundary:

    H = hidden branch after in_proj split
    C = depthwise causal conv(H), before SiLU
    U = SiLU(C)

Scientific direct forward uses:
    use_cache=False
    attention_mask=None

Therefore no cache-decoding or mask transformation lies between these
observed boundaries.

Magnitude factorization:

    ||delta_U_t||
      = ||delta_H_RF_t||
        * (||delta_C_t|| / ||delta_H_RF_t||)
        * (||delta_U_t|| / ||delta_C_t||)

where delta_H_RF_t is the concatenated four-token causal convolution
receptive-field difference:

    [delta_H_t, delta_H_(t-1), delta_H_(t-2), delta_H_(t-3)]

The audit separates:
- upstream pre-convolution receptive-field difference magnitude,
- exact depthwise-convolution receptive-field norm transfer,
- SiLU norm transfer.

The current-token delta-H magnitude is retained as a diagnostic but is not
used as the convolution-transfer denominator.

All diagnostics are observational/algebraic. No intervention, training,
tokenizer execution, logits, task heads, PCA, or learned probe.
"""

from __future__ import annotations

import argparse
import ast
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
    "2c54d9dbe62b3eb8581d6d582c497407212fe2f0"
)

PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_time_step_subspace_transfer_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "86dfe197c49b257fd786f99fcdd4ae2d873e644908325f15996718ea0821b0fc"
)

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_time_step_subspace_transfer_2d0f8c1_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "dfe618d961ba8fd7e0dcd16a75bf53cfb3e45878b2715e55fa656c1f6775d4cb"
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
ACTIVATION_REL_TOL = 1e-6

QUESTION = (
    "Is the corr-k2 post-convolution delta-U magnitude pattern already "
    "present in the four-token pre-convolution receptive-field delta-H "
    "magnitude, or is it reshaped across the depthwise causal convolution "
    "and SiLU stages?"
)


class PostConvFactorizationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PostConvFactorizationError(message)


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
        "k0_rvg_time_step_subspace_transfer_parent",
    )


def build_plan(root: Path, parent: Any):
    dt_projection = parent.load_parent_runner(root)

    (
        _upstream,
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
        dt_projection,
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
    base: Any,
):
    # Authenticate the upstream parent chain first.
    parent.authenticate_parent(
        root,
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
        == "k0-rvg-time-step-subspace-transfer-summary-v1",
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
        summary.get("primary_layer")
        == PRIMARY_LAYER,
        "PARENT_LAYER_MISMATCH",
    )

    require(
        summary.get("intermediate_size")
        == INTERMEDIATE_SIZE,
        "PARENT_INTERMEDIATE_SIZE_MISMATCH",
    )

    return repo, summary


def validate_direct_forward_contract(base: Any):
    source = textwrap.dedent(
        inspect.getsource(
            base.direct_backbone_forward
        )
    )

    tree = ast.parse(source)

    model_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mamba"
    ]

    require(
        len(model_calls) == 1,
        "DIRECT_BACKBONE_MAMBA_CALL_AMBIGUOUS",
    )

    call = model_calls[0]

    keywords = {
        kw.arg: kw.value
        for kw in call.keywords
        if kw.arg is not None
    }

    require(
        "attention_mask" not in keywords,
        "DIRECT_BACKBONE_ATTENTION_MASK_PRESENT",
    )

    require(
        "use_cache" in keywords,
        "DIRECT_BACKBONE_USE_CACHE_MISSING",
    )

    use_cache = keywords["use_cache"]

    require(
        isinstance(use_cache, ast.Constant)
        and use_cache.value is False,
        "DIRECT_BACKBONE_USE_CACHE_NOT_FALSE",
    )

    return {
        "attention_mask_passed": False,
        "use_cache": False,
    }


def resolve_primary_mixer(
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
        isinstance(
            mixer.conv1d,
            torch.nn.Conv1d,
        ),
        "CONV1D_NOT_CONV1D",
    )

    require(
        mixer.conv1d.in_channels
        == INTERMEDIATE_SIZE,
        "CONV1D_IN_CHANNEL_MISMATCH",
    )

    require(
        mixer.conv1d.out_channels
        == INTERMEDIATE_SIZE,
        "CONV1D_OUT_CHANNEL_MISMATCH",
    )

    require(
        mixer.conv1d.groups
        == INTERMEDIATE_SIZE,
        "CONV1D_NOT_DEPTHWISE",
    )

    require(
        tuple(mixer.conv1d.kernel_size)
        == (CONV_KERNEL_SIZE,),
        "CONV_KERNEL_SIZE_MISMATCH",
    )

    require(
        tuple(mixer.conv1d.padding)
        == (CONV_KERNEL_SIZE - 1,),
        "CONV_PADDING_MISMATCH",
    )

    require(
        str(mixer.activation).lower()
        == "silu",
        "ACTIVATION_NOT_SILU",
    )

    return mixer



def capture_hcu_factors(
    dt_projection: Any,
    secant: Any,
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    mixer: Any,
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    import torch

    targets = tuple(
        int(i)
        for i in targets
    )

    h_rf_records = {}
    c_records = {}

    kernel = secant.snapshot(
        mixer.conv1d.weight,
        "CONV_KERNEL",
    )

    require(
        tuple(kernel.shape)
        == (
            INTERMEDIATE_SIZE,
            1,
            CONV_KERNEL_SIZE,
        ),
        "CONV_KERNEL_SHAPE_MISMATCH",
    )

    kernel = (
        kernel[:, 0, :]
        .contiguous()
        .clone()
    )

    bias = (
        None
        if mixer.conv1d.bias is None
        else secant.snapshot(
            mixer.conv1d.bias,
            "CONV_BIAS",
        )
    )

    hook_counts = {
        "pre": 0,
        "post": 0,
    }

    def pre_hook(module, args):
        hook_counts["pre"] += 1

        require(
            hook_counts["pre"] == 1,
            "DUPLICATE_CONV_PRE_HOOK",
        )

        require(
            len(args) == 1,
            "CONV_PRE_HOOK_ARG_COUNT_MISMATCH",
        )

        h_full = secant.snapshot(
            args[0],
            "PRECONV_H_FULL",
        )

        require(
            len(h_full.shape) == 3,
            "PRECONV_H_RANK_MISMATCH",
        )

        require(
            h_full.shape[0] == 1,
            "PRECONV_H_BATCH_MISMATCH",
        )

        require(
            h_full.shape[1]
            == INTERMEDIATE_SIZE,
            "PRECONV_H_WIDTH_MISMATCH",
        )

        for i in targets:
            require(
                0 <= i < h_full.shape[2],
                f"PRECONV_TARGET_OUT_OF_RANGE:{i}",
            )

            lag_vectors = []

            for lag in range(
                CONV_KERNEL_SIZE
            ):
                pos = i - lag

                if pos >= 0:
                    vector = (
                        h_full[0, :, pos]
                        .contiguous()
                        .clone()
                    )
                else:
                    # Exact left zero-padding used by the causal Conv1d.
                    vector = torch.zeros(
                        INTERMEDIATE_SIZE,
                        dtype=h_full.dtype,
                        device=h_full.device,
                    )

                lag_vectors.append(
                    vector
                )

            h_rf_records[i] = (
                torch.stack(
                    lag_vectors,
                    dim=0,
                )
                .contiguous()
                .clone()
            )

    def post_hook(module, args, output):
        hook_counts["post"] += 1

        require(
            hook_counts["post"] == 1,
            "DUPLICATE_CONV_POST_HOOK",
        )

        c_full = secant.snapshot(
            output,
            "CONV_PREACT_FULL",
        )

        require(
            len(c_full.shape) == 3,
            "CONV_PREACT_RANK_MISMATCH",
        )

        require(
            c_full.shape[0] == 1,
            "CONV_PREACT_BATCH_MISMATCH",
        )

        require(
            c_full.shape[1]
            == INTERMEDIATE_SIZE,
            "CONV_PREACT_WIDTH_MISMATCH",
        )

        for i in targets:
            require(
                0 <= i < c_full.shape[2],
                f"CONV_TARGET_OUT_OF_RANGE:{i}",
            )

            c_records[i] = (
                c_full[:, :, i]
                .contiguous()
                .clone()
            )

    pre_handle = (
        mixer.conv1d
        .register_forward_pre_hook(
            pre_hook
        )
    )

    post_handle = (
        mixer.conv1d
        .register_forward_hook(
            post_hook
        )
    )

    try:
        records = dt_projection.capture_factors(
            secant,
            base,
            model,
            binding,
            layer_map,
            token_ids,
            targets,
        )

    finally:
        pre_handle.remove()
        post_handle.remove()

    require(
        hook_counts["pre"] == 1,
        "CONV_PRE_HOOK_COUNT_FAILURE",
    )

    require(
        hook_counts["post"] == 1,
        "CONV_POST_HOOK_COUNT_FAILURE",
    )

    require(
        set(h_rf_records)
        == set(targets),
        "PRECONV_RF_TARGET_SET_MISMATCH",
    )

    require(
        set(c_records)
        == set(targets),
        "CONV_PREACT_TARGET_SET_MISMATCH",
    )

    require(
        set(records)
        == set(targets),
        "PARENT_TARGET_SET_MISMATCH",
    )

    for i in targets:
        h_rf = h_rf_records[i]
        c = c_records[i]

        require(
            tuple(h_rf.shape)
            == (
                CONV_KERNEL_SIZE,
                INTERMEDIATE_SIZE,
            ),
            f"PRECONV_RF_SHAPE_MISMATCH:{i}",
        )

        records[i]["H_RF"] = h_rf
        records[i]["C"] = c

        # Conv1d is cross-correlation with left padding=K-1:
        # output[t] = sum_q w[q] * H[t-(K-1)+q].
        # With lag=0 denoting H[t], kernel index is K-1-lag.
        reconstructed_c = torch.zeros(
            (1, INTERMEDIATE_SIZE),
            dtype=h_rf.dtype,
            device=h_rf.device,
        )

        if bias is not None:
            reconstructed_c = (
                reconstructed_c
                + bias.unsqueeze(0)
            )

        for lag in range(
            CONV_KERNEL_SIZE
        ):
            kernel_index = (
                CONV_KERNEL_SIZE
                - 1
                - lag
            )

            reconstructed_c = (
                reconstructed_c
                + h_rf[lag, :].unsqueeze(0)
                * kernel[:, kernel_index].unsqueeze(0)
            )

        conv_residual = (
            reconstructed_c
            - c
        )

        conv_rel = float(
            torch.linalg.vector_norm(
                conv_residual.to(
                    torch.float64
                )
            ).item()
            / max(
                torch.linalg.vector_norm(
                    c.to(torch.float64)
                ).item(),
                1e-12,
            )
        )

        require(
            conv_rel
            <= CONV_RECON_REL_TOL,
            (
                "CONV_RECONSTRUCTION_FAILURE:"
                f"{i}:{conv_rel}"
            ),
        )

        records[i][
            "conv_reconstruction_relative_residual"
        ] = conv_rel

        require(
            "U" in records[i],
            f"PARENT_U_MISSING:{i}",
        )

        u = records[i]["U"]

        require(
            tuple(u.shape)
            == (1, INTERMEDIATE_SIZE),
            f"PARENT_U_SHAPE_MISMATCH:{i}",
        )

        with torch.inference_mode():
            reconstructed_u = (
                mixer.act(c)
                .detach()
                .cpu()
                .contiguous()
            )

        residual = (
            reconstructed_u
            - u
        )

        rel = float(
            torch.linalg.vector_norm(
                residual.to(torch.float64)
            ).item()
            / max(
                torch.linalg.vector_norm(
                    u.to(torch.float64)
                ).item(),
                1e-12,
            )
        )

        require(
            rel <= ACTIVATION_REL_TOL,
            (
                "ACTIVATION_RECONSTRUCTION_FAILURE:"
                f"{i}:{rel}"
            ),
        )

        records[i][
            "activation_reconstruction_relative_residual"
        ] = rel

    return records



def metric_rows_for_pair(
    secant: Any,
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

    out = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k

        hmrf32 = matched[token]["H_RF"]
        hsrf32 = swapped[token]["H_RF"]

        cm32 = matched[token]["C"]
        cs32 = swapped[token]["C"]

        um32 = matched[token]["U"]
        us32 = swapped[token]["U"]

        require(
            tuple(hmrf32.shape)
            == (
                CONV_KERNEL_SIZE,
                INTERMEDIATE_SIZE,
            ),
            f"HMRF_SHAPE_MISMATCH:{idx}:{role}:{k}",
        )

        require(
            tuple(hsrf32.shape)
            == (
                CONV_KERNEL_SIZE,
                INTERMEDIATE_SIZE,
            ),
            f"HSRF_SHAPE_MISMATCH:{idx}:{role}:{k}",
        )

        for name, tensor in (
            ("CM", cm32),
            ("CS", cs32),
            ("UM", um32),
            ("US", us32),
        ):
            require(
                tuple(tensor.shape)
                == (1, INTERMEDIATE_SIZE),
                f"{name}_SHAPE_MISMATCH:{idx}:{role}:{k}",
            )

        if k == -1:
            require(
                secant.torch_equal(
                    hmrf32,
                    hsrf32,
                ),
                f"K_MINUS_1_H_RF_IDENTITY_FAILURE:{idx}:{role}",
            )

            require(
                secant.torch_equal(
                    cm32,
                    cs32,
                ),
                f"K_MINUS_1_C_IDENTITY_FAILURE:{idx}:{role}",
            )

            require(
                secant.torch_equal(
                    um32,
                    us32,
                ),
                f"K_MINUS_1_U_IDENTITY_FAILURE:{idx}:{role}",
            )

        dh_rf = (
            hmrf32.to(torch.float64)
            - hsrf32.to(torch.float64)
        )

        dc = (
            cm32.to(torch.float64)
            - cs32.to(torch.float64)
        )

        du = (
            um32.to(torch.float64)
            - us32.to(torch.float64)
        )

        lag_l2 = [
            float(
                torch.linalg.vector_norm(
                    dh_rf[lag, :]
                ).item()
            )
            for lag in range(
                CONV_KERNEL_SIZE
            )
        ]

        dh_current_l2 = lag_l2[0]

        dh_rf_l2 = float(
            torch.linalg.vector_norm(
                dh_rf
            ).item()
        )

        dc_l2 = float(
            torch.linalg.vector_norm(
                dc
            ).item()
        )

        du_l2 = float(
            torch.linalg.vector_norm(
                du
            ).item()
        )

        conv_residual = max(
            float(
                matched[token][
                    "conv_reconstruction_relative_residual"
                ]
            ),
            float(
                swapped[token][
                    "conv_reconstruction_relative_residual"
                ]
            ),
        )

        activation_residual = max(
            float(
                matched[token][
                    "activation_reconstruction_relative_residual"
                ]
            ),
            float(
                swapped[token][
                    "activation_reconstruction_relative_residual"
                ]
            ),
        )

        if k == -1:
            require(
                dh_rf_l2 == 0.0,
                f"K_MINUS_1_DELTA_H_RF_NONZERO:{idx}:{role}",
            )

            require(
                dc_l2 == 0.0,
                f"K_MINUS_1_DELTA_C_NONZERO:{idx}:{role}",
            )

            require(
                du_l2 == 0.0,
                f"K_MINUS_1_DELTA_U_NONZERO:{idx}:{role}",
            )

            conv_rf_transfer = 0.0
            act_transfer = 0.0
            total_transfer = 0.0
            current_energy_fraction = 0.0

        else:
            require(
                dh_rf_l2 > 0.0,
                f"ZERO_DELTA_H_RF_AFTER_ANCHOR:{idx}:{role}:{k}",
            )

            require(
                dc_l2 > 0.0,
                f"ZERO_DELTA_C_AFTER_ANCHOR:{idx}:{role}:{k}",
            )

            require(
                du_l2 > 0.0,
                f"ZERO_DELTA_U_AFTER_ANCHOR:{idx}:{role}:{k}",
            )

            conv_rf_transfer = (
                dc_l2
                / dh_rf_l2
            )

            act_transfer = (
                du_l2
                / dc_l2
            )

            total_transfer = (
                du_l2
                / dh_rf_l2
            )

            current_energy_fraction = (
                dh_current_l2 ** 2
                / (dh_rf_l2 ** 2)
            )

            require(
                -1e-12
                <= current_energy_fraction
                <= 1.0 + 1e-12,
                (
                    "CURRENT_H_ENERGY_FRACTION_OUT_OF_RANGE:"
                    f"{idx}:{role}:{k}:"
                    f"{current_energy_fraction}"
                ),
            )

            require(
                math.isclose(
                    total_transfer,
                    conv_rf_transfer
                    * act_transfer,
                    rel_tol=1e-13,
                    abs_tol=1e-13,
                ),
                (
                    "STAGEWISE_TRANSFER_FACTORIZATION_FAILURE:"
                    f"{idx}:{role}:{k}"
                ),
            )

        values = {
            "schema_version":
                "k0-rvg-postconv-u-factorization-row-v1",

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

            "delta_h_current_l2":
                dh_current_l2,

            "delta_h_rf_l2":
                dh_rf_l2,

            "delta_h_lag0_l2":
                lag_l2[0],

            "delta_h_lag1_l2":
                lag_l2[1],

            "delta_h_lag2_l2":
                lag_l2[2],

            "delta_h_lag3_l2":
                lag_l2[3],

            "current_h_energy_fraction":
                current_energy_fraction,

            "delta_c_l2":
                dc_l2,

            "delta_u_l2":
                du_l2,

            "conv_rf_transfer":
                conv_rf_transfer,

            "activation_transfer":
                act_transfer,

            "total_rf_to_u_transfer":
                total_transfer,

            "conv_reconstruction_relative_residual":
                conv_residual,

            "activation_reconstruction_relative_residual":
                activation_residual,

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
    "delta_h_current_l2",
    "delta_h_rf_l2",
    "delta_h_lag0_l2",
    "delta_h_lag1_l2",
    "delta_h_lag2_l2",
    "delta_h_lag3_l2",
    "current_h_energy_fraction",
    "delta_c_l2",
    "delta_u_l2",
    "conv_rf_transfer",
    "activation_transfer",
    "total_rf_to_u_transfer",
    "conv_reconstruction_relative_residual",
    "activation_reconstruction_relative_residual",
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

    post_rows = [
        r
        for r in rows
        if int(
            r["relative_coordinate"]
        ) != -1
    ]

    return {
        "schema_version":
            "k0-rvg-postconv-u-factorization-summary-v1",

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

        "relative_coordinates":
            list(RELATIVE_COORDINATES),

        "execution_protocol":
            EXECUTION_PROTOCOL,

        "stage_identity":
            (
                "H_preconv_4token_receptive_field -> "
                "depthwise_conv -> C_preact -> SiLU -> U_postact"
            ),

        "magnitude_factorization":
            (
                "||delta_U|| = ||delta_H_RF|| * "
                "(||delta_C||/||delta_H_RF||) * "
                "(||delta_U||/||delta_C||)"
            ),

        "full_336_trajectory":
            aggregate_trajectory(rows),

        "common_330_ddsssss_trajectory":
            aggregate_trajectory(
                common_rows
            ),

        "max_conv_reconstruction_relative_residual":
            max(
                float(
                    r[
                        "conv_reconstruction_relative_residual"
                    ]
                )
                for r in post_rows
            ),

        "max_activation_reconstruction_relative_residual":
            max(
                float(
                    r[
                        "activation_reconstruction_relative_residual"
                    ]
                )
                for r in post_rows
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
                    ["delta_u_l2"]["median"]
                )

                expected = float(
                    parent[role][str(k)]
                    ["delta_u_l2"]["median"]
                )

                require(
                    math.isclose(
                        got,
                        expected,
                        rel_tol=1e-13,
                        abs_tol=1e-13,
                    ),
                    (
                        "PARENT_DELTA_U_REPRODUCTION_FAILURE:"
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

    mixer = resolve_primary_mixer(
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

        matched = capture_hcu_factors(
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

        swapped = capture_hcu_factors(
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
        / "postconv_u_factorization_metrics.jsonl"
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
            "k0-rvg-postconv-u-factorization-execution-manifest-v1",

        "runtime_git_head":
            repo["head"],

        "runtime_branch":
            repo["branch"],

        "parent_freeze_commit":
            PARENT_FREEZE_COMMIT,

        "scientific_question":
            QUESTION,

        "stage_identity":
            (
                "H_preconv_4token_receptive_field -> "
                "depthwise_conv -> C_preact -> SiLU -> U_postact"
            ),

        "magnitude_factorization":
            (
                "||delta_U|| = ||delta_H_RF|| * "
                "conv_rf_transfer * activation_transfer"
            ),

        "conv_receptive_field_size":
            CONV_KERNEL_SIZE,

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

        "parent_delta_u_summary_match":
            True,

        "attention_mask_passed":
            False,

        "use_cache":
            False,

        "raw_vectors_persisted":
            False,

        "scientific_model_forward_executed":
            True,

        "scientific_preconv_hidden_read":
            True,

        "scientific_preconv_receptive_field_read":
            True,

        "scientific_conv_preactivation_read":
            True,

        "scientific_postconv_U_read":
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
            "postconv_u_factorization_metrics.jsonl":
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
        "PASS_POSTCONV_U_FACTORIZATION_EXECUTION"
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
        "parent_delta_u_summary_match = True"
    )

    print(
        "max_conv_reconstruction_relative_residual =",
        summary[
            "max_conv_reconstruction_relative_residual"
        ],
    )

    print(
        "max_activation_reconstruction_relative_residual =",
        summary[
            "max_activation_reconstruction_relative_residual"
        ],
    )

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[
                role
            ][str(k)]

            print(
                f"{role}_k{k}_median "
                f"DH_CUR_L2="
                f"{t['delta_h_current_l2']['median']} "
                f"DH_RF_L2="
                f"{t['delta_h_rf_l2']['median']} "
                f"CUR_EFRAC="
                f"{t['current_h_energy_fraction']['median']} "
                f"DC_L2="
                f"{t['delta_c_l2']['median']} "
                f"DU_L2="
                f"{t['delta_u_l2']['median']} "
                f"CONV_RF_TR="
                f"{t['conv_rf_transfer']['median']} "
                f"ACT_TR="
                f"{t['activation_transfer']['median']} "
                f"TOTAL_RF_U_TR="
                f"{t['total_rf_to_u_transfer']['median']}"
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
        base,
    )

    source = secant.validate_frozen_source_semantics()

    require(
        source["source_sha256"]
        == EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    direct = validate_direct_forward_contract(
        base
    )

    print(
        "=== POSTCONV-U FACTORIZATION AUDIT PLAN ==="
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
        "stage_identity = "
        "H_preconv_4token_receptive_field -> "
        "depthwise_conv -> C_preact -> SiLU -> U_postact"
    )

    print(
        "magnitude_factorization = "
        "||delta_U|| = ||delta_H_RF|| * "
        "conv_rf_transfer * activation_transfer"
    )

    print(
        "attention_mask_passed =",
        direct[
            "attention_mask_passed"
        ],
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
        "parent_delta_u_summary_preserved_by_design = True"
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
            "PASS_POSTCONV_U_FACTORIZATION_STATIC_PREFLIGHT"
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

    except PostConvFactorizationError as exc:
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