"""K0-RVG layer-22 gate/content factorization audit.

Authenticated slow-path boundary:

    C22 = recurrent scan readout + D * hidden_states
    A22 = SiLU(gate)
    V22 = transpose(C22 * A22)

For matched/swapped branches, the exact symmetric product decomposition is:

    delta_V22 = Q_C + Q_A

    Q_C = Abar22 * delta_C22
    Q_A = Cbar22 * delta_A22

where bars are matched/swapped arithmetic means.

Observed runtime tensors are float32. Branch-level V22 = C22 * A22 replay
is validated in float32. The symmetric product identity is evaluated in
float64 from captured operands. Comparison of that float64 product delta to
the observed float32 delta-V22 is retained as a non-blocking numerical
diagnostic.

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
from pathlib import Path
from typing import Any, Mapping, Sequence


PARENT_FREEZE_COMMIT = (
    "f456e31f2730e0bee1633b30f7d7776bae7de05d"
)

PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer22_output_projection_transfer_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "2c2f2a228ae49a77b4ddaafd162a6b211de664796afe7b27f007dc32fc8dc2a1"
)

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_output_projection_transfer_20ebf51_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "7abeb6eec6d5060be57d53689a23f8e5d3b9bd35b40e234cab87a91b92efb536"
)

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FORWARD_COUNT = 1344

SOURCE_LAYER = 22
WIDTH = 1536

RELATIVE_COORDINATES = tuple(range(-1, 7))
POST_HORIZON = 6

EXECUTION_PROTOCOL = (
    "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"
)

FLOAT32_PRODUCT_RECON_REL_TOL = 5e-6
SYMMETRIC_PRODUCT_CLOSURE_REL_TOL = 1e-12

QUESTION = (
    "Is the strong layer-22 pre-output-projection delta-V22 role separation "
    "primarily associated with pre-gate content C22, activated-gate A22, "
    "or their vector interaction?"
)


class GateContentAuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateContentAuditError(message)


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
        "k0_rvg_layer22_output_projection_parent",
    )


def build_plan(root: Path, parent: Any):
    residual_parent = parent.load_parent(
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
    ) = parent.build_plan(
        root,
        residual_parent,
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
        residual_parent,
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
    residual_parent: Any,
    rms_parent: Any,
    hidden_parent: Any,
    four_tap: Any,
    postconv: Any,
    time_step: Any,
    base: Any,
):
    parent.authenticate_parent(
        root,
        residual_parent,
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
        == "k0-rvg-layer22-output-projection-transfer-summary-v1",
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


def capture_cav(
    parent: Any,
    residual_parent: Any,
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

    targets = tuple(
        int(i)
        for i in targets
    )

    c_records = {}
    a_records = {}

    gate_capture_count = 0

    slow_code = (
        type(mixer22)
        .slow_forward
        .__code__
    )

    def act_hook(
        module,
        args,
        output,
    ):
        nonlocal gate_capture_count

        require(
            len(args) == 1,
            "ACT_HOOK_ARG_COUNT_MISMATCH",
        )

        frame = inspect.currentframe()

        try:
            frame = (
                frame.f_back
                if frame is not None
                else None
            )

            slow_frame = None

            while frame is not None:
                if (
                    frame.f_code
                    is slow_code
                    and frame.f_locals.get("self")
                    is mixer22
                    and "scan_output"
                    in frame.f_locals
                    and "gate"
                    in frame.f_locals
                ):
                    slow_frame = frame
                    break

                frame = frame.f_back

            if slow_frame is None:
                return

            gate_capture_count += 1

            require(
                gate_capture_count == 1,
                "DUPLICATE_GATE_ACTIVATION_CAPTURE",
            )

            gate_input = args[0]
            gate_local = slow_frame.f_locals[
                "gate"
            ]

            require(
                gate_input is gate_local
                or torch.equal(
                    gate_input,
                    gate_local,
                ),
                "GATE_HOOK_INPUT_MISMATCH",
            )

            c_full = secant.snapshot(
                slow_frame.f_locals[
                    "scan_output"
                ],
                "LAYER22_PRE_GATE_CONTENT_C22_FULL",
            )

            a_full = secant.snapshot(
                output,
                "LAYER22_ACTIVATED_GATE_A22_FULL",
            )

            require(
                len(c_full.shape) == 3,
                "C22_RANK_MISMATCH",
            )

            require(
                len(a_full.shape) == 3,
                "A22_RANK_MISMATCH",
            )

            require(
                c_full.shape
                == a_full.shape,
                "C22_A22_SHAPE_MISMATCH",
            )

            require(
                c_full.shape[0] == 1,
                "C22_BATCH_MISMATCH",
            )

            require(
                c_full.shape[1] == WIDTH,
                "C22_WIDTH_MISMATCH",
            )

            for i in targets:
                require(
                    0 <= i
                    < c_full.shape[2],
                    f"C22_TARGET_OUT_OF_RANGE:{i}",
                )

                c_records[i] = (
                    c_full[
                        :,
                        :,
                        i,
                    ]
                    .contiguous()
                    .clone()
                )

                a_records[i] = (
                    a_full[
                        :,
                        :,
                        i,
                    ]
                    .contiguous()
                    .clone()
                )

        finally:
            del frame

    handle = (
        mixer22.act
        .register_forward_hook(
            act_hook
        )
    )

    try:
        records = parent.capture_vy(
            residual_parent,
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
            token_ids,
            targets,
        )

    finally:
        handle.remove()

    require(
        gate_capture_count == 1,
        "GATE_ACTIVATION_CAPTURE_COUNT_FAILURE",
    )

    require(
        set(c_records)
        == set(targets),
        "C22_TARGET_SET_MISMATCH",
    )

    require(
        set(a_records)
        == set(targets),
        "A22_TARGET_SET_MISMATCH",
    )

    require(
        set(records)
        == set(targets),
        "PARENT_TARGET_SET_MISMATCH",
    )

    for i in targets:
        c = c_records[i]
        a = a_records[i]
        v = records[i]["V22"]

        require(
            tuple(c.shape)
            == (1, WIDTH),
            f"C22_SHAPE_MISMATCH:{i}",
        )

        require(
            tuple(a.shape)
            == (1, WIDTH),
            f"A22_SHAPE_MISMATCH:{i}",
        )

        require(
            tuple(v.shape)
            == (1, WIDTH),
            f"V22_SHAPE_MISMATCH:{i}",
        )

        reconstructed = (
            c.to(torch.float32)
            * a.to(torch.float32)
        )

        residual = (
            reconstructed
            - v.to(torch.float32)
        )

        branch_rel = float(
            torch.linalg.vector_norm(
                residual.to(
                    torch.float64
                )
            ).item()
            / max(
                torch.linalg.vector_norm(
                    v.to(
                        torch.float64
                    )
                ).item(),
                1e-12,
            )
        )

        require(
            branch_rel
            <= FLOAT32_PRODUCT_RECON_REL_TOL,
            (
                "FLOAT32_GATE_CONTENT_PRODUCT_RECON_FAILURE:"
                f"{i}:{branch_rel}"
            ),
        )

        records[i]["C22"] = c
        records[i]["A22"] = a

        records[i][
            "float32_gate_content_product_reconstruction_relative_residual"
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

        cm = (
            matched[token]["C22"]
            .to(torch.float64)
        )

        cs = (
            swapped[token]["C22"]
            .to(torch.float64)
        )

        am = (
            matched[token]["A22"]
            .to(torch.float64)
        )

        a_s = (
            swapped[token]["A22"]
            .to(torch.float64)
        )

        vm = (
            matched[token]["V22"]
            .to(torch.float64)
        )

        vs = (
            swapped[token]["V22"]
            .to(torch.float64)
        )

        if k == -1:
            require(
                torch.equal(
                    matched[token]["C22"],
                    swapped[token]["C22"],
                ),
                f"K_MINUS_1_C22_IDENTITY_FAILURE:{idx}:{role}",
            )

            require(
                torch.equal(
                    matched[token]["A22"],
                    swapped[token]["A22"],
                ),
                f"K_MINUS_1_A22_IDENTITY_FAILURE:{idx}:{role}",
            )

            require(
                torch.equal(
                    matched[token]["V22"],
                    swapped[token]["V22"],
                ),
                f"K_MINUS_1_V22_IDENTITY_FAILURE:{idx}:{role}",
            )

        dc = cm - cs
        da = am - a_s
        dv = vm - vs

        cbar = (
            cm + cs
        ) / 2.0

        abar = (
            am + a_s
        ) / 2.0

        q_content = (
            abar * dc
        )

        q_gate = (
            cbar * da
        )

        qsum = (
            q_content
            + q_gate
        )

        product_delta = (
            cm * am
            - cs * a_s
        )

        dc_l2 = float(
            torch.linalg.vector_norm(
                dc
            ).item()
        )

        da_l2 = float(
            torch.linalg.vector_norm(
                da
            ).item()
        )

        dv_l2 = float(
            torch.linalg.vector_norm(
                dv
            ).item()
        )

        qc_l2 = float(
            torch.linalg.vector_norm(
                q_content
            ).item()
        )

        qg_l2 = float(
            torch.linalg.vector_norm(
                q_gate
            ).item()
        )

        qsum_l2 = float(
            torch.linalg.vector_norm(
                qsum
            ).item()
        )

        product_delta_l2 = float(
            torch.linalg.vector_norm(
                product_delta
            ).item()
        )

        closure_residual = float(
            torch.linalg.vector_norm(
                qsum
                - product_delta
            ).item()
        )

        closure_rel = float(
            closure_residual
            / max(
                product_delta_l2,
                1e-12,
            )
        )

        require(
            closure_rel
            <= SYMMETRIC_PRODUCT_CLOSURE_REL_TOL,
            (
                "SYMMETRIC_PRODUCT_CLOSURE_FAILURE:"
                f"{idx}:{role}:{k}:{closure_rel}"
            ),
        )

        observed_product_residual = float(
            torch.linalg.vector_norm(
                product_delta
                - dv
            ).item()
        )

        observed_delta_rel = float(
            observed_product_residual
            / max(
                dv_l2,
                1e-12,
            )
        )

        branch_scale = float(
            torch.linalg.vector_norm(
                vm
            ).item()
            + torch.linalg.vector_norm(
                vs
            ).item()
        )

        branch_scale_rel = float(
            observed_product_residual
            / max(
                branch_scale,
                1e-12,
            )
        )

        qc_sq = qc_l2 * qc_l2
        qg_sq = qg_l2 * qg_l2
        rss_sq = qc_sq + qg_sq
        rss_l2 = math.sqrt(
            rss_sq
        )

        dot = float(
            torch.sum(
                q_content
                * q_gate
            ).item()
        )

        normalized_cross = (
            2.0 * dot / rss_sq
            if rss_sq > 0.0
            else 0.0
        )

        content_energy_fraction = (
            qc_sq / rss_sq
            if rss_sq > 0.0
            else 0.0
        )

        gate_energy_fraction = (
            qg_sq / rss_sq
            if rss_sq > 0.0
            else 0.0
        )

        addition_factor = (
            qsum_l2 / rss_l2
            if rss_l2 > 0.0
            else 0.0
        )

        qsum_sq = (
            qsum_l2
            * qsum_l2
        )

        content_additive_share = (
            (qc_sq + dot)
            / qsum_sq
            if qsum_sq > 0.0
            else 0.0
        )

        gate_additive_share = (
            (qg_sq + dot)
            / qsum_sq
            if qsum_sq > 0.0
            else 0.0
        )

        if qsum_sq > 0.0:
            require(
                math.isclose(
                    content_additive_share
                    + gate_additive_share,
                    1.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                (
                    "ADDITIVE_SHARE_CLOSURE_FAILURE:"
                    f"{idx}:{role}:{k}"
                ),
            )

        if k == -1:
            for value, name in (
                (dc_l2, "DC22"),
                (da_l2, "DA22"),
                (dv_l2, "DV22"),
                (qc_l2, "QC"),
                (qg_l2, "QG"),
                (qsum_l2, "QSUM"),
                (
                    product_delta_l2,
                    "PRODUCT_DELTA",
                ),
            ):
                require(
                    value == 0.0,
                    (
                        f"K_MINUS_1_{name}_NONZERO:"
                        f"{idx}:{role}:{value}"
                    ),
                )

        values = {
            "schema_version":
                "k0-rvg-layer22-gate-content-factorization-row-v1",

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

            "delta_c22_l2":
                dc_l2,

            "delta_a22_l2":
                da_l2,

            "delta_v22_l2":
                dv_l2,

            "q_content_l2":
                qc_l2,

            "q_gate_l2":
                qg_l2,

            "qsum_l2":
                qsum_l2,

            "product_delta_v22_l2":
                product_delta_l2,

            "rss_l2":
                rss_l2,

            "content_energy_fraction":
                content_energy_fraction,

            "gate_energy_fraction":
                gate_energy_fraction,

            "normalized_cross":
                normalized_cross,

            "addition_factor":
                addition_factor,

            "content_additive_share":
                content_additive_share,

            "gate_additive_share":
                gate_additive_share,

            "symmetric_product_closure_relative_residual":
                closure_rel,

            "float64_product_delta_relative_to_observed_delta_v22":
                observed_delta_rel,

            "float64_product_delta_relative_to_branch_scale":
                branch_scale_rel,

            "float32_gate_content_product_reconstruction_relative_residual":
                max(
                    float(
                        matched[token][
                            "float32_gate_content_product_reconstruction_relative_residual"
                        ]
                    ),
                    float(
                        swapped[token][
                            "float32_gate_content_product_reconstruction_relative_residual"
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
                and not isinstance(
                    v,
                    bool,
                )
            ),
            f"NONFINITE_METRIC:{idx}:{role}:{k}",
        )

        output.append(
            values
        )

    return output


SUMMARY_FIELDS = (
    "delta_c22_l2",
    "delta_a22_l2",
    "delta_v22_l2",
    "q_content_l2",
    "q_gate_l2",
    "qsum_l2",
    "product_delta_v22_l2",
    "rss_l2",
    "content_energy_fraction",
    "gate_energy_fraction",
    "normalized_cross",
    "addition_factor",
    "content_additive_share",
    "gate_additive_share",
    "symmetric_product_closure_relative_residual",
    "float64_product_delta_relative_to_observed_delta_v22",
    "float64_product_delta_relative_to_branch_scale",
    "float32_gate_content_product_reconstruction_relative_residual",
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
            float(
                min(vals)
            ),

        "max":
            float(
                max(vals)
            ),
    }


def aggregate_trajectory(rows):
    result = {
        "corr": {},
        "ctrl": {},
    }

    for role in (
        "corr",
        "ctrl",
    ):
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
                field:
                    aggregate(
                        row[field]
                        for row
                        in bucket
                    )
                for field
                in SUMMARY_FIELDS
            }

    return result


def make_summary(
    rows,
    cohort,
):
    require(
        len(rows)
        == EXPECTED_PAIR_ROLE_COUNT
        * len(
            RELATIVE_COORDINATES
        ),
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
        * len(
            RELATIVE_COORDINATES
        ),
        "COMMON_ROW_COUNT_MISMATCH",
    )

    return {
        "schema_version":
            "k0-rvg-layer22-gate-content-factorization-summary-v1",

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

        "width":
            WIDTH,

        "relative_coordinates":
            list(
                RELATIVE_COORDINATES
            ),

        "execution_protocol":
            EXECUTION_PROTOCOL,

        "identity":
            (
                "delta_V22 = "
                "Abar22*delta_C22 + "
                "Cbar22*delta_A22"
            ),

        "full_336_trajectory":
            aggregate_trajectory(
                rows
            ),

        "common_330_ddsssss_trajectory":
            aggregate_trajectory(
                common_rows
            ),

        "max_symmetric_product_closure_relative_residual":
            max(
                float(
                    row[
                        "symmetric_product_closure_relative_residual"
                    ]
                )
                for row in rows
            ),

        "max_float64_product_delta_relative_to_observed_delta_v22":
            max(
                float(
                    row[
                        "float64_product_delta_relative_to_observed_delta_v22"
                    ]
                )
                for row in rows
            ),

        "max_float64_product_delta_relative_to_branch_scale":
            max(
                float(
                    row[
                        "float64_product_delta_relative_to_branch_scale"
                    ]
                )
                for row in rows
            ),

        "max_float32_gate_content_product_reconstruction_relative_residual":
            max(
                float(
                    row[
                        "float32_gate_content_product_reconstruction_relative_residual"
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
        current = summary[
            trajectory
        ]

        parent = parent_summary[
            trajectory
        ]

        for role in (
            "corr",
            "ctrl",
        ):
            for k in RELATIVE_COORDINATES:
                got = float(
                    current[
                        role
                    ][str(k)]
                    ["delta_v22_l2"]
                    ["median"]
                )

                expected = float(
                    parent[
                        role
                    ][str(k)]
                    ["delta_v22_l2"]
                    ["median"]
                )

                require(
                    math.isclose(
                        got,
                        expected,
                        rel_tol=1e-13,
                        abs_tol=1e-13,
                    ),
                    (
                        "PARENT_DELTA_V22_REPRODUCTION_FAILURE:"
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
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        )
        + "\n"
    ).encode(
        "utf-8"
    )


def jsonl_bytes(rows):
    return b"".join(
        json_bytes(row)
        for row in rows
    )


def execute(
    root,
    parent,
    residual_parent,
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
    import torch

    final_dir = (
        output_dir.resolve()
    )

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
        == parent.EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    (
        layer22,
        layer23,
        norm23,
        mixer23,
        mixer22,
        out_proj,
    ) = parent.resolve_output_projection(
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        model,
        layer_map,
    )

    require(
        tuple(
            mixer22.D.shape
        )
        == (WIDTH,),
        "D_SHAPE_MISMATCH",
    )

    probe = torch.tensor(
        [
            -3.0,
            -1.0,
            0.0,
            1.0,
            3.0,
        ],
        dtype=torch.float32,
    )

    with torch.inference_mode():
        actual = mixer22.act(
            probe.clone()
        )

        expected = torch.nn.functional.silu(
            probe.clone()
        )

    require(
        torch.equal(
            actual,
            expected,
        ),
        "ACTIVATION_NOT_EXACT_SILU",
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
            row[
                "role"
            ]
        )

        anchor = int(
            row[
                "anchor"
            ]
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
            len(
                matched_prefix
            ) == cutoff
            and len(
                swapped_prefix
            ) == cutoff,
            (
                "PREFIX_LENGTH_FAILURE:"
                f"{idx}:{role}"
            ),
        )

        matched = capture_cav(
            parent,
            residual_parent,
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
            row[
                "targets"
            ],
        )

        forward_count += 1

        swapped = capture_cav(
            parent,
            residual_parent,
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
            row[
                "targets"
            ],
        )

        forward_count += 1

        rows.extend(
            metric_rows_for_pair(
                row,
                matched,
                swapped,
                cohort,
                signatures[
                    (
                        idx,
                        role,
                    )
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
        / "layer22_gate_content_factorization_metrics.jsonl"
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
        jsonl_bytes(
            rows
        )
    )

    summary_path.write_bytes(
        json_bytes(
            summary
        )
    )

    manifest = {
        "schema_version":
            "k0-rvg-layer22-gate-content-factorization-execution-manifest-v1",

        "runtime_git_head":
            repo["head"],

        "runtime_branch":
            repo["branch"],

        "parent_freeze_commit":
            PARENT_FREEZE_COMMIT,

        "parent_summary_sha256":
            PARENT_SUMMARY_SHA256,

        "scientific_question":
            QUESTION,

        "identity":
            (
                "delta_V22 = "
                "Abar22*delta_C22 + "
                "Cbar22*delta_A22"
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

        "source_layer":
            SOURCE_LAYER,

        "width":
            WIDTH,

        "mixer_forward_sha256":
            parent.EXPECTED_MIXER_FORWARD_SHA256,

        "slow_forward_sha256":
            parent.EXPECTED_SLOW_FORWARD_SHA256,

        "mamba_source_sha256":
            binding.source_sha256,

        "activation":
            "SiLU",

        "activation_exact_synthetic_probe":
            True,

        "capture_method":
            (
                "mixer22.act forward hook with "
                "authenticated slow_forward frame inspection"
            ),

        "pre_gate_content_definition":
            (
                "scan_output after recurrent readout "
                "+ hidden_states * D and before gate multiplication"
            ),

        "activated_gate_definition":
            "mixer22.act(gate)",

        "float32_product_reconstruction_rel_tol":
            FLOAT32_PRODUCT_RECON_REL_TOL,

        "symmetric_product_closure_rel_tol":
            SYMMETRIC_PRODUCT_CLOSURE_REL_TOL,

        "float64_product_vs_observed_delta_validation_mode":
            "diagnostic_only",

        "handoff_zip_sha256":
            handoff[
                "zip_sha256"
            ],

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

        "parent_delta_v22_summary_match":
            True,

        "raw_vectors_persisted":
            False,

        "scientific_model_forward_executed":
            True,

        "scientific_c22_read":
            True,

        "scientific_a22_read":
            True,

        "scientific_v22_read":
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
            .relative_to(
                root
            )
            .as_posix(),

        "runner_sha256":
            sha256_bytes(
                Path(
                    __file__
                ).read_bytes()
            ),

        "outputs": {
            "layer22_gate_content_factorization_metrics.jsonl":
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
        json_bytes(
            manifest
        )
    )

    os.replace(
        partial_dir,
        final_dir,
    )

    common = summary[
        "common_330_ddsssss_trajectory"
    ]

    print(
        "PASS_LAYER22_GATE_CONTENT_FACTORIZATION_EXECUTION"
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
        "parent_delta_v22_summary_match = True"
    )

    print(
        "max_symmetric_product_closure_relative_residual =",
        summary[
            "max_symmetric_product_closure_relative_residual"
        ],
    )

    print(
        "max_float64_product_delta_relative_to_observed_delta_v22 =",
        summary[
            "max_float64_product_delta_relative_to_observed_delta_v22"
        ],
    )

    print(
        "max_float64_product_delta_relative_to_branch_scale =",
        summary[
            "max_float64_product_delta_relative_to_branch_scale"
        ],
    )

    print(
        "max_float32_gate_content_product_reconstruction_relative_residual =",
        summary[
            "max_float32_gate_content_product_reconstruction_relative_residual"
        ],
    )

    for role in (
        "corr",
        "ctrl",
    ):
        for k in (
            1,
            2,
            3,
        ):
            t = common[
                role
            ][str(k)]

            print(
                f"{role}_k{k}_median "
                f"C22={t['delta_c22_l2']['median']} "
                f"A22={t['delta_a22_l2']['median']} "
                f"V22={t['delta_v22_l2']['median']} "
                f"QC={t['q_content_l2']['median']} "
                f"QA={t['q_gate_l2']['median']} "
                f"CONTENT_E={t['content_energy_fraction']['median']} "
                f"CROSS={t['normalized_cross']['median']} "
                f"ADD={t['addition_factor']['median']}"
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
        residual_parent,
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
        residual_parent,
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
        == parent.EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    print(
        "=== LAYER22 GATE / CONTENT FACTORIZATION AUDIT PLAN ==="
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
        "parent_runner_sha256 =",
        PARENT_RUNNER_SHA256,
    )

    print(
        "parent_summary_sha256 =",
        PARENT_SUMMARY_SHA256,
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
        "delta_V22 = "
        "Abar22*delta_C22 + "
        "Cbar22*delta_A22"
    )

    print(
        "C22_definition = "
        "recurrent readout + hidden_states*D, "
        "before gate multiplication"
    )

    print(
        "A22_definition = SiLU(gate)"
    )

    print(
        "capture_method = "
        "act hook + authenticated slow_forward frame inspection"
    )

    print(
        "float32_product_reconstruction_rel_tol =",
        FLOAT32_PRODUCT_RECON_REL_TOL,
    )

    print(
        "symmetric_product_closure_rel_tol =",
        SYMMETRIC_PRODUCT_CLOSURE_REL_TOL,
    )

    print(
        "float64_product_vs_observed_delta_validation_mode = "
        "diagnostic_only"
    )

    print(
        "mixer_forward_sha256 =",
        parent.EXPECTED_MIXER_FORWARD_SHA256,
    )

    print(
        "slow_forward_sha256 =",
        parent.EXPECTED_SLOW_FORWARD_SHA256,
    )

    print(
        "mamba_source_sha256 =",
        source[
            "source_sha256"
        ],
    )

    print(
        "parent_delta_v22_summary_preserved_by_design = True"
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
            "PASS_LAYER22_GATE_CONTENT_FACTORIZATION_STATIC_PREFLIGHT"
        )
        return 0

    require(
        args.handoff
        is not None,
        "EXECUTE_REQUIRES_HANDOFF",
    )

    require(
        args.output_dir
        is not None,
        "EXECUTE_REQUIRES_OUTPUT_DIR",
    )

    execute(
        root,
        parent,
        residual_parent,
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
        raise SystemExit(
            main()
        )

    except GateContentAuditError as exc:
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
