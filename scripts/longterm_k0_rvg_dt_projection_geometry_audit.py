"""K0-RVG dt-projection row-geometry audit.

Frozen parent result:

    delta_Z = W_dt * delta_r

where delta_r is the matched-minus-swapped low-rank pre-dt-projection
time-step vector and the dt_proj bias cancels.

For dt_proj output row w_j:

    delta_Z_j^2
      = ||w_j||^2 * ||delta_r||^2 * cos(theta_j)^2

Therefore, for nonzero delta_r, the normalized projected delta-Z channel
distribution is exactly proportional to:

    row_norm2_j * direction_cos2_j

where:

    row_norm2_j      = ||w_j||^2
    direction_cos2_j = cos(theta_j)^2

This audit separates the fixed dt_proj row-norm geometry from the
sample/token-specific low-rank delta-r directional alignment.

The softplus gain is only defined on projected output channels. Accordingly,
the question is not whether low-rank coordinates themselves have softplus
gain, but whether delta-r preferentially aligns with dt_proj rows whose
current absolute-Z operating points have high softplus transmission.

These are observational/algebraic geometry diagnostics, not interventions.
No raw vectors are persisted.
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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence


PARENT_FREEZE_COMMIT = "634677160f18dfeed0c5de54a35542d649cd5a0a"

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_uz_channel_selection_d198276_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "743944cc424f8f08140e21bced8464e3f8b78f895a1e1528cd419e66d9623ee0"
)

PARENT_RUNNER_REL = (
    "scripts/longterm_k0_rvg_uz_channel_selection_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "c5ae455937a702d6a69df534000911b913227ccbf2de6eb9a583c2bdc2fdf333"
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
    "Does corr-k2 delta-Z high-softplus-transmission preselection arise "
    "primarily from fixed dt_proj row-norm geometry, from sample-specific "
    "low-rank delta-time-step directional alignment with high-gain rows, "
    "or from their multiplicative combination?"
)

GAIN_TOL = 1e-6
PROJECTION_REL_TOL = 2e-5
DISTRIBUTION_TOL = 5e-5


class DTProjectionGeometryError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise DTProjectionGeometryError(message)


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
        "k0_rvg_uz_channel_selection_parent",
    )


def build_plan(root: Path, parent: Any):
    operating = parent.load_parent_runner(root)

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
    ) = parent.build_plan(
        root,
        operating,
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
        == "k0-rvg-uz-channel-selection-summary-v1",
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


class LowRankTimeStepCollector:
    """Extend the frozen secant collector with pre-dt-projection time_step."""

    def __init__(
        self,
        secant: Any,
        binding: Any,
        layer_map: Mapping[int, int],
        target_indices: Sequence[int],
    ):
        self.secant = secant

        parent_cls = secant.SoftplusSecantCollector

        class Collector(parent_cls):
            def __init__(inner_self):
                super().__init__(
                    binding,
                    layer_map,
                    target_indices,
                )
                inner_self.r_records = None

            def _trace(inner_self, frame, event, arg):
                if (
                    frame.f_code is inner_self.binding.code
                    and event == "line"
                ):
                    mixer = frame.f_locals.get("self")
                    layer = inner_self.layer_map.get(id(mixer))

                    if (
                        layer == PRIMARY_LAYER
                        and frame.f_lineno
                        == secant.EXPECTED_PRE_SOFTPLUS_LINE
                    ):
                        require(
                            inner_self.r_records is not None,
                            "LOW_RANK_TRACE_NOT_ACTIVE",
                        )
                        require(
                            inner_self.r_records == {},
                            "DUPLICATE_LOW_RANK_CAPTURE",
                        )

                        r_full = frame.f_locals.get("time_step")

                        require(
                            r_full is not None,
                            "LOW_RANK_TIME_STEP_MISSING",
                        )

                        r_full = secant.snapshot(
                            r_full,
                            "LOW_RANK_TIME_STEP_FULL",
                        )

                        require(
                            len(r_full.shape) == 3,
                            "LOW_RANK_TIME_STEP_RANK_MISMATCH",
                        )
                        require(
                            r_full.shape[0] == 1,
                            "LOW_RANK_TIME_STEP_BATCH_MISMATCH",
                        )
                        require(
                            r_full.shape[2] > 0
                            and r_full.shape[2] < 1536,
                            "LOW_RANK_TIME_STEP_WIDTH_INVALID",
                        )

                        for i in inner_self.targets:
                            require(
                                i < r_full.shape[1],
                                f"LOW_RANK_TARGET_OUT_OF_RANGE:{i}",
                            )

                            inner_self.r_records[i] = (
                                r_full[:, i, :]
                                .contiguous()
                                .clone()
                            )

                return super(Collector, inner_self)._trace(
                    frame,
                    event,
                    arg,
                )

            @contextmanager
            def capture(inner_self):
                inner_self.r_records = {}

                with super(Collector, inner_self).capture():
                    yield inner_self

                require(
                    set(inner_self.r_records)
                    == set(inner_self.targets),
                    "LOW_RANK_TARGET_SET_MISMATCH",
                )

        self.collector = Collector()


def capture_factors(
    secant: Any,
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    wrapper = LowRankTimeStepCollector(
        secant,
        binding,
        layer_map,
        targets,
    )

    collector = wrapper.collector

    with collector.capture():
        base.direct_backbone_forward(
            model,
            token_ids,
        )

    require(
        collector.records is not None,
        "CAPTURE_RECORDS_MISSING",
    )

    require(
        collector.r_records is not None,
        "LOW_RANK_RECORDS_MISSING",
    )

    for i in collector.targets:
        require(
            i in collector.records
            and i in collector.r_records,
            f"LOW_RANK_RECORD_JOIN_FAILURE:{i}",
        )

        collector.records[i]["R"] = (
            collector.r_records[i]
        )

    return collector.records


def resolve_dt_proj_geometry(
    secant: Any,
    model: Any,
    layer_map: Mapping[int, int],
):
    import torch

    candidates = [
        module
        for module in model.modules()
        if layer_map.get(id(module)) == PRIMARY_LAYER
    ]

    require(
        len(candidates) == 1,
        "PRIMARY_MIXER_RESOLUTION_FAILURE",
    )

    mixer = candidates[0]
    dt_proj = getattr(mixer, "dt_proj", None)

    require(
        isinstance(dt_proj, torch.nn.Linear),
        "DT_PROJ_NOT_LINEAR",
    )

    weight = secant.snapshot(
        dt_proj.weight,
        "DT_PROJ_WEIGHT",
    )

    bias = secant.snapshot(
        dt_proj.bias,
        "DT_PROJ_BIAS",
    )

    require(
        len(weight.shape) == 2,
        "DT_PROJ_WEIGHT_RANK_MISMATCH",
    )
    require(
        weight.shape[0] == 1536,
        "DT_PROJ_OUTPUT_WIDTH_MISMATCH",
    )
    require(
        weight.shape[1] > 0
        and weight.shape[1] < 1536,
        "DT_PROJ_INPUT_WIDTH_INVALID",
    )
    require(
        tuple(bias.shape) == (1536,),
        "DT_PROJ_BIAS_SHAPE_MISMATCH",
    )

    row_norm2 = torch.sum(
        weight.to(torch.float64) ** 2,
        dim=1,
    )

    require(
        bool(torch.all(row_norm2 > 0).item()),
        "DT_PROJ_ZERO_ROW",
    )

    return {
        "weight": weight,
        "bias": bias,
        "row_norm2": row_norm2,
        "time_step_rank": int(weight.shape[1]),
    }


def distribution_metrics(
    parent: Any,
    gain,
    z_mid,
    weights,
):
    return parent.distribution_metrics(
        gain,
        z_mid,
        weights,
    )


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

    weight = geometry["weight"].to(torch.float64)
    row_norm2 = geometry["row_norm2"]

    out = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k

        Rm32 = matched[token]["R"]
        Zm32 = matched[token]["Z"]
        Tm32 = matched[token]["T"]

        Rs32 = swapped[token]["R"]
        Zs32 = swapped[token]["Z"]
        Ts32 = swapped[token]["T"]

        if k == -1:
            require(
                secant.torch_equal(Rm32, Rs32),
                f"K_MINUS_1_R_IDENTITY_FAILURE:{idx}:{role}",
            )
            require(
                secant.torch_equal(Zm32, Zs32),
                f"K_MINUS_1_Z_IDENTITY_FAILURE:{idx}:{role}",
            )

        Rm = Rm32.to(torch.float64)
        Rs = Rs32.to(torch.float64)

        Zm = Zm32.to(torch.float64)
        Zs = Zs32.to(torch.float64)

        Tm = Tm32.to(torch.float64)
        Ts = Ts32.to(torch.float64)

        require(
            Rm.shape[1] == geometry["time_step_rank"],
            f"LOW_RANK_WIDTH_MISMATCH:{idx}:{role}:{k}",
        )

        delta_r = Rm - Rs
        delta_z = Zm - Zs
        delta_t = Tm - Ts

        z_mid = 0.5 * (Zm + Zs)

        zero_z = delta_z == 0
        nonzero_z = ~zero_z

        gain = torch.empty_like(delta_z)

        gain[nonzero_z] = (
            delta_t[nonzero_z]
            / delta_z[nonzero_z]
        )

        if bool(zero_z.any().item()):
            gain[zero_z] = torch.sigmoid(
                z_mid[zero_z]
            )

        require(
            bool(torch.isfinite(gain).all().item()),
            f"NONFINITE_GAIN:{idx}:{role}:{k}",
        )

        require(
            float(gain.min().item()) >= -GAIN_TOL
            and float(gain.max().item()) <= 1.0 + GAIN_TOL,
            f"GAIN_RANGE_FAILURE:{idx}:{role}:{k}",
        )

        dr = delta_r.squeeze(0)
        dz = delta_z.squeeze(0)
        g = gain.squeeze(0)
        zm = z_mid.squeeze(0)

        dr_norm2 = torch.sum(dr * dr)

        projected = torch.matmul(
            weight,
            dr,
        )

        residual = projected - dz

        dz_norm = torch.linalg.vector_norm(dz)
        residual_norm = torch.linalg.vector_norm(residual)

        projection_relative_residual = float(
            (
                residual_norm
                / torch.clamp(dz_norm, min=1e-12)
            ).item()
        )

        if k == -1:
            require(
                float(dr_norm2.item()) == 0.0,
                f"K_MINUS_1_DELTA_R_NONZERO:{idx}:{role}",
            )
            require(
                float(dz_norm.item()) == 0.0,
                f"K_MINUS_1_DELTA_Z_NONZERO:{idx}:{role}",
            )

            direction_cos2 = torch.zeros_like(
                row_norm2
            )
            geometry_product = torch.zeros_like(
                row_norm2
            )

        else:
            require(
                float(dr_norm2.item()) > 0.0,
                f"ZERO_DELTA_R_AFTER_ANCHOR:{idx}:{role}:{k}",
            )

            require(
                projection_relative_residual
                <= PROJECTION_REL_TOL,
                (
                    "DT_PROJ_RECONSTRUCTION_FAILURE:"
                    f"{idx}:{role}:{k}:"
                    f"{projection_relative_residual}"
                ),
            )

            direction_cos2 = (
                projected * projected
                / (
                    row_norm2
                    * dr_norm2
                )
            )

            require(
                float(direction_cos2.min().item())
                >= -1e-12,
                f"NEGATIVE_DIRECTION_COS2:{idx}:{role}:{k}",
            )

            require(
                float(direction_cos2.max().item())
                <= 1.0 + 1e-8,
                f"DIRECTION_COS2_GT_ONE:{idx}:{role}:{k}",
            )

            direction_cos2 = torch.clamp(
                direction_cos2,
                min=0.0,
                max=1.0,
            )

            geometry_product = (
                row_norm2
                * direction_cos2
            )

            geometry_identity_error = float(
                torch.linalg.vector_norm(
                    (
                        geometry_product
                        * dr_norm2
                    )
                    - projected * projected
                ).item()
            )

            require(
                geometry_identity_error <= 1e-10,
                (
                    "GEOMETRY_PRODUCT_IDENTITY_FAILURE:"
                    f"{idx}:{role}:{k}:"
                    f"{geometry_identity_error}"
                ),
            )

        observed_dz2 = dz * dz

        row_metrics = distribution_metrics(
            parent,
            g,
            zm,
            row_norm2,
        )

        direction_metrics = distribution_metrics(
            parent,
            g,
            zm,
            direction_cos2,
        )

        product_metrics = distribution_metrics(
            parent,
            g,
            zm,
            geometry_product,
        )

        observed_metrics = distribution_metrics(
            parent,
            g,
            zm,
            observed_dz2,
        )

        if k != -1:
            require(
                abs(
                    product_metrics["gain_rms"]
                    - observed_metrics["gain_rms"]
                ) <= DISTRIBUTION_TOL,
                (
                    "PROJECTED_DISTRIBUTION_GAIN_MISMATCH:"
                    f"{idx}:{role}:{k}:"
                    f"{product_metrics['gain_rms']}:"
                    f"{observed_metrics['gain_rms']}"
                ),
            )

            require(
                abs(
                    product_metrics["fraction_gain_ge_090"]
                    - observed_metrics["fraction_gain_ge_090"]
                ) <= DISTRIBUTION_TOL,
                (
                    "PROJECTED_DISTRIBUTION_E90_MISMATCH:"
                    f"{idx}:{role}:{k}:"
                    f"{product_metrics['fraction_gain_ge_090']}:"
                    f"{observed_metrics['fraction_gain_ge_090']}"
                ),
            )

        values = {
            "schema_version":
                "k0-rvg-dt-projection-geometry-row-v1",
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
                float(torch.sqrt(dr_norm2).item()),

            "observed_delta_z_l2":
                float(dz_norm.item()),

            "projected_delta_z_l2":
                float(
                    torch.linalg.vector_norm(
                        projected
                    ).item()
                ),

            "projection_relative_residual":
                projection_relative_residual,

            "row_direction_cosine":
                parent.nonnegative_cosine(
                    row_norm2,
                    direction_cos2,
                ),

            "row_direction_multiplicative_enrichment":
                parent.multiplicative_enrichment(
                    row_norm2,
                    direction_cos2,
                ),

            "direction_reweight_gain_rms_delta":
                (
                    product_metrics["gain_rms"]
                    - row_metrics["gain_rms"]
                ),

            "row_norm_reweight_gain_rms_delta":
                (
                    product_metrics["gain_rms"]
                    - direction_metrics["gain_rms"]
                ),

            "joint_excess_gain_rms":
                (
                    product_metrics["gain_rms"]
                    - max(
                        row_metrics["gain_rms"],
                        direction_metrics["gain_rms"],
                    )
                ),

            "direction_reweight_gain090_fraction_delta":
                (
                    product_metrics[
                        "fraction_gain_ge_090"
                    ]
                    - row_metrics[
                        "fraction_gain_ge_090"
                    ]
                ),

            "row_norm_reweight_gain090_fraction_delta":
                (
                    product_metrics[
                        "fraction_gain_ge_090"
                    ]
                    - direction_metrics[
                        "fraction_gain_ge_090"
                    ]
                ),

            "joint_excess_gain090_fraction":
                (
                    product_metrics[
                        "fraction_gain_ge_090"
                    ]
                    - max(
                        row_metrics[
                            "fraction_gain_ge_090"
                        ],
                        direction_metrics[
                            "fraction_gain_ge_090"
                        ],
                    )
                ),

            "source_snapshot_dtype":
                "torch.float32",

            "metric_accumulation_dtype":
                "torch.float64",
        }

        for prefix, metrics in (
            ("row", row_metrics),
            ("direction", direction_metrics),
            ("product", product_metrics),
            ("observed_dz2", observed_metrics),
        ):
            for name, value in metrics.items():
                values[
                    f"{prefix}_{name}"
                ] = value

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
    "delta_r_l2",
    "observed_delta_z_l2",
    "projected_delta_z_l2",
    "projection_relative_residual",

    "row_gain_rms",
    "row_fraction_gain_ge_090",
    "row_z_mid_mean",
    "row_effective_channel_count",

    "direction_gain_rms",
    "direction_fraction_gain_ge_090",
    "direction_z_mid_mean",
    "direction_effective_channel_count",

    "product_gain_rms",
    "product_fraction_gain_ge_090",
    "product_z_mid_mean",
    "product_effective_channel_count",

    "observed_dz2_gain_rms",
    "observed_dz2_fraction_gain_ge_090",
    "observed_dz2_z_mid_mean",
    "observed_dz2_effective_channel_count",

    "row_direction_cosine",
    "row_direction_multiplicative_enrichment",

    "direction_reweight_gain_rms_delta",
    "row_norm_reweight_gain_rms_delta",
    "joint_excess_gain_rms",

    "direction_reweight_gain090_fraction_delta",
    "row_norm_reweight_gain090_fraction_delta",
    "joint_excess_gain090_fraction",
)


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
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
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
            "k0-rvg-dt-projection-geometry-summary-v1",

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
            geometry["time_step_rank"],

        "relative_coordinates":
            list(RELATIVE_COORDINATES),

        "execution_protocol":
            EXECUTION_PROTOCOL,

        "geometry_identity":
            (
                "delta_Z_j^2 = row_norm2_j * "
                "||delta_r||^2 * direction_cos2_j"
            ),

        "weight_definitions": {
            "row":
                "||dt_proj_row_j||^2",
            "direction":
                "cos(dt_proj_row_j, delta_r)^2",
            "product":
                "row_norm2_j * direction_cos2_j",
            "observed_dz2":
                "observed delta_Z_j^2",
        },

        "full_336_trajectory":
            aggregate_trajectory(rows),

        "common_330_ddsssss_trajectory":
            aggregate_trajectory(common_rows),

        "max_projection_relative_residual":
            max(
                float(
                    r[
                        "projection_relative_residual"
                    ]
                )
                for r in rows
                if int(
                    r["relative_coordinate"]
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
    current = summary[
        "common_330_ddsssss_trajectory"
    ]

    parent = parent_summary[
        "common_330_ddsssss_trajectory"
    ]

    mapping = {
        "observed_dz2_gain_rms":
            "dz2_gain_rms",
        "observed_dz2_fraction_gain_ge_090":
            "dz2_fraction_gain_ge_090",
        "observed_dz2_z_mid_mean":
            "dz2_z_mid_mean",
        "observed_dz2_effective_channel_count":
            "dz2_effective_channel_count",
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

    geometry = resolve_dt_proj_geometry(
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
            len(matched_prefix) == cutoff
            and len(swapped_prefix) == cutoff,
            f"PREFIX_LENGTH_FAILURE:{idx}:{role}",
        )

        matched = capture_factors(
            secant,
            base,
            model,
            binding,
            layer_map,
            matched_prefix,
            row["targets"],
        )

        forward_count += 1

        swapped = capture_factors(
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
                signatures[(idx, role)],
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
        / "dt_projection_geometry_metrics.jsonl"
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
            "k0-rvg-dt-projection-geometry-execution-manifest-v1",

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

        "geometry_identity":
            (
                "delta_Z_j^2 = row_norm2_j * "
                "||delta_r||^2 * direction_cos2_j"
            ),

        "time_step_rank":
            geometry["time_step_rank"],

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

        "parent_uz_summary_match":
            True,

        "source_snapshot_dtype":
            "torch.float32",

        "metric_accumulation_dtype":
            "torch.float64",

        "raw_vectors_persisted":
            False,

        "scientific_model_forward_executed":
            True,

        "scientific_low_rank_time_step_read":
            True,

        "scientific_dt_proj_geometry_read":
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
            "dt_projection_geometry_metrics.jsonl":
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
        "PASS_DT_PROJECTION_GEOMETRY_EXECUTION"
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
        "time_step_rank =",
        geometry["time_step_rank"],
    )

    print(
        "parent_uz_summary_match = True"
    )

    print(
        "max_projection_relative_residual =",
        summary[
            "max_projection_relative_residual"
        ],
    )

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[role][str(k)]

            print(
                f"{role}_k{k}_median "
                f"ROW_G="
                f"{t['row_gain_rms']['median']} "
                f"DIR_G="
                f"{t['direction_gain_rms']['median']} "
                f"PROD_G="
                f"{t['product_gain_rms']['median']} "
                f"OBS_DZ_G="
                f"{t['observed_dz2_gain_rms']['median']} "
                f"DIR_reweight="
                f"{t['direction_reweight_gain_rms_delta']['median']} "
                f"ROW_reweight="
                f"{t['row_norm_reweight_gain_rms_delta']['median']} "
                f"joint_excess="
                f"{t['joint_excess_gain_rms']['median']} "
                f"ROW_E90="
                f"{t['row_fraction_gain_ge_090']['median']} "
                f"DIR_E90="
                f"{t['direction_fraction_gain_ge_090']['median']} "
                f"PROD_E90="
                f"{t['product_fraction_gain_ge_090']['median']} "
                f"cos_ROW_DIR="
                f"{t['row_direction_cosine']['median']} "
                f"enrichment="
                f"{t['row_direction_multiplicative_enrichment']['median']}"
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

    print(
        "=== DT-PROJECTION GEOMETRY AUDIT PLAN ==="
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
        "geometry_identity = "
        "delta_Z_j^2 = ||w_j||^2 * ||delta_r||^2 * cos(theta_j)^2"
    )

    print(
        "row_weight = ||dt_proj_row_j||^2"
    )

    print(
        "direction_weight = cos(dt_proj_row_j, delta_r)^2"
    )

    print(
        "product_weight = row_weight * direction_weight"
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
        "low_rank_time_step_capture_line =",
        source["pre_softplus_line"],
    )

    print(
        "parent_uz_summary_preserved = True"
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
            "PASS_DT_PROJECTION_GEOMETRY_STATIC_PREFLIGHT"
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

    except DTProjectionGeometryError as exc:
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