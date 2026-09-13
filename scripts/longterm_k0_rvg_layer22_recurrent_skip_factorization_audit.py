"""K0-RVG layer-22 recurrent-readout / D-skip factorization audit.

Authenticated source boundary:

    C22 = R22 + K22

where:
- C22 is the observed pre-gate content captured by the frozen gate/content audit;
- K22 = hidden_states * D is reconstructed from the authenticated slow_forward
  frame using the exact source operands at the gate-call boundary;
- R22 := C22 - K22 is a source-defined recurrent-readout complement.

Important epistemic boundary:
R22 is NOT claimed to be a directly captured runtime tensor in this audit.
It is the exact complement of observed C22 after subtracting authenticated
D-skip K22.

For matched/swapped branches:

    delta_C22 = delta_R22 + delta_K22

and:

    ||delta_C22||^2
      = ||delta_R22||^2
      + ||delta_K22||^2
      + 2 <delta_R22, delta_K22>

This audit is observational/algebraic only.
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
    "bf15fc1b7fa48081930df3f3c0e070241413dabd"
)

PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer22_gate_content_factorization_audit.py"
)

PARENT_RUNNER_SHA256 = (
    "91505b337c0f208ee2606aace7caa4551d0e6d8666c208207e35f420cb089ec6"
)

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_gate_content_factorization_be292fc_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "cf1137b9b0fa92b497696ac81537ca883805218a38d17766544ff5badbb6e594"
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

ADDITIVE_CLOSURE_REL_TOL = 1e-12

QUESTION = (
    "Within the validated layer-22 pre-gate content C22 role separation, "
    "is the dominant factor the source-defined recurrent-readout complement "
    "R22, the direct D-skip K22 = hidden_states*D, or their vector addition "
    "geometry?"
)


class RecurrentSkipAuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RecurrentSkipAuditError(message)


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


def load_parent(root: Path):
    path = root / PARENT_RUNNER_REL
    require(
        sha256_bytes(path.read_bytes()) == PARENT_RUNNER_SHA256,
        "PARENT_RUNNER_SHA256_MISMATCH",
    )
    return import_module(
        path,
        "k0_rvg_layer22_gate_content_parent",
    )


def build_plan(root: Path, parent: Any):
    output_parent = parent.load_parent(root)

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
    ) = parent.build_plan(
        root,
        output_parent,
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
        output_parent,
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
    output_parent: Any,
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
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        four_tap,
        postconv,
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

    frozen = base.git_bytes(
        root,
        f"{PARENT_FREEZE_COMMIT}:{PARENT_SUMMARY_REL}",
    )

    require(
        sha256_bytes(frozen) == PARENT_SUMMARY_SHA256,
        "PARENT_SUMMARY_SHA256_MISMATCH",
    )

    current = root / PARENT_SUMMARY_REL
    require(
        current.is_file(),
        "PARENT_SUMMARY_MISSING",
    )
    require(
        current.read_bytes() == frozen,
        "PARENT_SUMMARY_WORKTREE_DRIFT",
    )

    summary = json.loads(frozen)

    require(
        summary.get("schema_version")
        == "k0-rvg-layer22-gate-content-factorization-summary-v1",
        "PARENT_SCHEMA_MISMATCH",
    )
    require(
        summary.get("item_count") == EXPECTED_ITEM_COUNT,
        "PARENT_ITEM_COUNT_MISMATCH",
    )
    require(
        summary.get("pair_role_count") == EXPECTED_PAIR_ROLE_COUNT,
        "PARENT_PAIR_ROLE_COUNT_MISMATCH",
    )
    require(
        summary.get("common_ddsssss_item_count") == EXPECTED_COMMON_COUNT,
        "PARENT_COMMON_COUNT_MISMATCH",
    )

    return repo, summary


def capture_ck(
    parent: Any,
    output_parent: Any,
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

    targets = tuple(int(i) for i in targets)
    k_records = {}
    skip_capture_count = 0

    slow_code = type(mixer22).slow_forward.__code__

    d_cpu = (
        mixer22.D
        .detach()
        .cpu()
        .contiguous()
        .clone()
    )

    require(
        tuple(d_cpu.shape) == (WIDTH,),
        "D_SHAPE_MISMATCH",
    )
    require(
        d_cpu.dtype == torch.float32,
        "D_DTYPE_MISMATCH",
    )

    def act_hook(module, args, output):
        nonlocal skip_capture_count

        require(
            len(args) == 1,
            "ACT_HOOK_ARG_COUNT_MISMATCH",
        )

        frame = inspect.currentframe()

        try:
            frame = frame.f_back if frame is not None else None
            slow_frame = None

            while frame is not None:
                if (
                    frame.f_code is slow_code
                    and frame.f_locals.get("self") is mixer22
                    and "scan_output" in frame.f_locals
                    and "gate" in frame.f_locals
                    and "hidden_states" in frame.f_locals
                ):
                    slow_frame = frame
                    break

                frame = frame.f_back

            if slow_frame is None:
                return

            gate_input = args[0]
            gate_local = slow_frame.f_locals["gate"]

            require(
                gate_input is gate_local
                or torch.equal(gate_input, gate_local),
                "GATE_HOOK_INPUT_MISMATCH",
            )

            skip_capture_count += 1

            require(
                skip_capture_count == 1,
                "DUPLICATE_D_SKIP_CAPTURE",
            )

            h_full = secant.snapshot(
                slow_frame.f_locals["hidden_states"],
                "LAYER22_D_SKIP_HIDDEN_STATES_FULL",
            )

            require(
                len(h_full.shape) == 3,
                "D_SKIP_HIDDEN_RANK_MISMATCH",
            )
            require(
                h_full.shape[0] == 1,
                "D_SKIP_HIDDEN_BATCH_MISMATCH",
            )
            require(
                h_full.shape[1] == WIDTH,
                "D_SKIP_HIDDEN_WIDTH_MISMATCH",
            )

            k_full = (
                h_full.to(torch.float32)
                * d_cpu[None, :, None]
            )

            require(
                tuple(k_full.shape)
                == tuple(h_full.shape),
                "D_SKIP_SHAPE_MISMATCH",
            )

            for i in targets:
                require(
                    0 <= i < k_full.shape[2],
                    f"D_SKIP_TARGET_OUT_OF_RANGE:{i}",
                )

                k_records[i] = (
                    k_full[:, :, i]
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
        records = parent.capture_cav(
            output_parent,
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
        skip_capture_count == 1,
        "D_SKIP_CAPTURE_COUNT_FAILURE",
    )
    require(
        set(k_records) == set(targets),
        "D_SKIP_TARGET_SET_MISMATCH",
    )
    require(
        set(records) == set(targets),
        "PARENT_TARGET_SET_MISMATCH",
    )

    for i in targets:
        c = records[i]["C22"]
        k = k_records[i]

        require(
            tuple(c.shape) == (1, WIDTH),
            f"C22_SHAPE_MISMATCH:{i}",
        )
        require(
            tuple(k.shape) == (1, WIDTH),
            f"K22_SHAPE_MISMATCH:{i}",
        )
        require(
            c.dtype == torch.float32,
            f"C22_DTYPE_MISMATCH:{i}",
        )
        require(
            k.dtype == torch.float32,
            f"K22_DTYPE_MISMATCH:{i}",
        )

        records[i]["K22"] = k

    return records


def metric_rows_for_pair(
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

    output = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k

        cm = matched[token]["C22"].to(torch.float64)
        cs = swapped[token]["C22"].to(torch.float64)

        km = matched[token]["K22"].to(torch.float64)
        ks = swapped[token]["K22"].to(torch.float64)

        # Source-defined recurrent-readout complements.
        # These are not direct runtime captures.
        rm = cm - km
        rs = cs - ks

        dc = cm - cs
        dk = km - ks
        dr = rm - rs

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
                    matched[token]["K22"],
                    swapped[token]["K22"],
                ),
                f"K_MINUS_1_K22_IDENTITY_FAILURE:{idx}:{role}",
            )

        reconstructed_dc = dr + dk

        closure_residual = float(
            torch.linalg.vector_norm(
                reconstructed_dc - dc
            ).item()
        )

        dc_l2 = float(
            torch.linalg.vector_norm(dc).item()
        )
        dr_l2 = float(
            torch.linalg.vector_norm(dr).item()
        )
        dk_l2 = float(
            torch.linalg.vector_norm(dk).item()
        )

        closure_rel = float(
            closure_residual
            / max(dc_l2, 1e-12)
        )

        require(
            closure_rel <= ADDITIVE_CLOSURE_REL_TOL,
            (
                "RECURRENT_SKIP_ADDITIVE_CLOSURE_FAILURE:"
                f"{idx}:{role}:{k}:{closure_rel}"
            ),
        )

        dr_sq = dr_l2 * dr_l2
        dk_sq = dk_l2 * dk_l2
        rss_sq = dr_sq + dk_sq
        rss_l2 = math.sqrt(rss_sq)

        dot = float(
            torch.sum(dr * dk).item()
        )

        normalized_cross = (
            2.0 * dot / rss_sq
            if rss_sq > 0.0
            else 0.0
        )

        recurrent_energy_fraction = (
            dr_sq / rss_sq
            if rss_sq > 0.0
            else 0.0
        )

        skip_energy_fraction = (
            dk_sq / rss_sq
            if rss_sq > 0.0
            else 0.0
        )

        addition_factor = (
            dc_l2 / rss_l2
            if rss_l2 > 0.0
            else 0.0
        )

        dc_sq = dc_l2 * dc_l2

        recurrent_additive_share = (
            (dr_sq + dot) / dc_sq
            if dc_sq > 0.0
            else 0.0
        )

        skip_additive_share = (
            (dk_sq + dot) / dc_sq
            if dc_sq > 0.0
            else 0.0
        )

        if dc_sq > 0.0:
            require(
                math.isclose(
                    recurrent_additive_share
                    + skip_additive_share,
                    1.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                (
                    "RECURRENT_SKIP_ADDITIVE_SHARE_CLOSURE_FAILURE:"
                    f"{idx}:{role}:{k}"
                ),
            )

        if k == -1:
            for value, name in (
                (dr_l2, "DR22_COMPLEMENT"),
                (dk_l2, "DK22"),
                (dc_l2, "DC22"),
                (rss_l2, "RSS22"),
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
                "k0-rvg-layer22-recurrent-skip-factorization-row-v1",
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
            "delta_r22_complement_l2":
                dr_l2,
            "delta_k22_l2":
                dk_l2,
            "delta_c22_l2":
                dc_l2,
            "rss_l2":
                rss_l2,
            "recurrent_energy_fraction":
                recurrent_energy_fraction,
            "skip_energy_fraction":
                skip_energy_fraction,
            "normalized_cross":
                normalized_cross,
            "addition_factor":
                addition_factor,
            "recurrent_additive_share":
                recurrent_additive_share,
            "skip_additive_share":
                skip_additive_share,
            "additive_closure_relative_residual":
                closure_rel,
            "r22_observation_mode":
                "source_defined_complement_C22_minus_K22",
            "k22_observation_mode":
                "reconstructed_from_observed_hidden_states_and_frozen_D",
            "c22_observation_mode":
                "direct_parent_runtime_capture",
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
    "delta_r22_complement_l2",
    "delta_k22_l2",
    "delta_c22_l2",
    "rss_l2",
    "recurrent_energy_fraction",
    "skip_energy_fraction",
    "normalized_cross",
    "addition_factor",
    "recurrent_additive_share",
    "skip_additive_share",
    "additive_closure_relative_residual",
)


def aggregate(values):
    vals = [float(v) for v in values]
    require(bool(vals), "EMPTY_AGGREGATE")
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
                and int(row["relative_coordinate"]) == k
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


def make_summary(rows, cohort):
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
            "k0-rvg-layer22-recurrent-skip-factorization-summary-v1",
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
            list(RELATIVE_COORDINATES),
        "execution_protocol":
            EXECUTION_PROTOCOL,
        "identity":
            (
                "delta_C22 = "
                "delta_R22_complement + "
                "delta_K22"
            ),
        "r22_definition":
            (
                "R22_complement := "
                "C22_observed - K22_reconstructed"
            ),
        "r22_direct_runtime_capture":
            False,
        "k22_definition":
            "K22 := hidden_states * D",
        "full_336_trajectory":
            aggregate_trajectory(rows),
        "common_330_ddsssss_trajectory":
            aggregate_trajectory(common_rows),
        "max_additive_closure_relative_residual":
            max(
                float(
                    row[
                        "additive_closure_relative_residual"
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
                    ["delta_c22_l2"]["median"]
                )
                expected = float(
                    parent[role][str(k)]
                    ["delta_c22_l2"]["median"]
                )

                require(
                    math.isclose(
                        got,
                        expected,
                        rel_tol=1e-13,
                        abs_tol=1e-13,
                    ),
                    (
                        "PARENT_DELTA_C22_REPRODUCTION_FAILURE:"
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
    output_parent,
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
        == output_parent.EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    (
        layer22,
        layer23,
        norm23,
        mixer23,
        mixer22,
        out_proj,
    ) = output_parent.resolve_output_projection(
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        model,
        layer_map,
    )

    require(
        tuple(mixer22.D.shape) == (WIDTH,),
        "D_SHAPE_MISMATCH",
    )
    require(
        mixer22.D.dtype == torch.float32,
        "D_DTYPE_MISMATCH",
    )

    rows = []
    forward_count = 0

    for n, row in enumerate(
        plan,
        start=1,
    ):
        idx = int(row["local_template_index"])
        role = str(row["role"])
        anchor = int(row["anchor"])

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

        matched = capture_ck(
            parent,
            output_parent,
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
            row["targets"],
        )
        forward_count += 1

        swapped = capture_ck(
            parent,
            output_parent,
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

        if n % 16 == 0 or n == len(plan):
            print(
                "PROGRESS "
                f"pair_roles={n}/{len(plan)} "
                f"model_forwards={forward_count}",
                flush=True,
            )

    require(
        forward_count == EXPECTED_FORWARD_COUNT,
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
        / "layer22_recurrent_skip_factorization_metrics.jsonl"
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
            "k0-rvg-layer22-recurrent-skip-factorization-execution-manifest-v1",
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
                "delta_C22 = "
                "delta_R22_complement + "
                "delta_K22"
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
            output_parent.EXPECTED_MIXER_FORWARD_SHA256,
        "slow_forward_sha256":
            output_parent.EXPECTED_SLOW_FORWARD_SHA256,
        "mamba_source_sha256":
            binding.source_sha256,
        "c22_observation_mode":
            "direct_parent_runtime_capture",
        "k22_definition":
            "hidden_states * D",
        "k22_observation_mode":
            (
                "reconstructed_from_authenticated_slow_forward_frame_operands"
            ),
        "r22_definition":
            "source_defined_complement_C22_minus_K22",
        "r22_direct_runtime_capture":
            False,
        "capture_method":
            (
                "mixer22.act forward hook restricted to authenticated "
                "slow_forward gate-call frame; capture hidden_states, "
                "reconstruct K22 from hidden_states*D, inherit direct C22 "
                "capture from frozen parent, define R22 complement as C22-K22"
            ),
        "additive_closure_rel_tol":
            ADDITIVE_CLOSURE_REL_TOL,
        "handoff_zip_sha256":
            handoff["zip_sha256"],
        "checkpoint_sha256":
            handoff["checkpoint_sha256"],
        "encoder_canonical_digest":
            encoder["canonical_digest"],
        "encoder_raw_concat_digest":
            encoder["raw_concat_digest"],
        "parent_delta_c22_summary_match":
            True,
        "raw_vectors_persisted":
            False,
        "scientific_model_forward_executed":
            True,
        "scientific_c22_read":
            True,
        "scientific_k22_reconstructed":
            True,
        "scientific_r22_direct_read":
            False,
        "scientific_r22_complement_computed":
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
            "layer22_recurrent_skip_factorization_metrics.jsonl":
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
        "PASS_LAYER22_RECURRENT_SKIP_FACTORIZATION_EXECUTION"
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
        "parent_delta_c22_summary_match = True"
    )
    print(
        "r22_direct_runtime_capture = False"
    )
    print(
        "r22_definition = source_defined_complement_C22_minus_K22"
    )
    print(
        "max_additive_closure_relative_residual =",
        summary[
            "max_additive_closure_relative_residual"
        ],
    )

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[
                role
            ][str(k)]

            print(
                f"{role}_k{k}_median "
                f"R22C={t['delta_r22_complement_l2']['median']} "
                f"K22={t['delta_k22_l2']['median']} "
                f"C22={t['delta_c22_l2']['median']} "
                f"REC_E={t['recurrent_energy_fraction']['median']} "
                f"SKIP_E={t['skip_energy_fraction']['median']} "
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
        args.static_preflight ^ args.execute,
        "SELECT_EXACTLY_ONE_MODE",
    )

    root = Path.cwd().resolve()

    parent = load_parent(root)

    (
        output_parent,
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
        output_parent,
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
        == output_parent.EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    print(
        "=== LAYER22 RECURRENT / D-SKIP FACTORIZATION AUDIT PLAN ==="
    )
    print("branch =", repo["branch"])
    print("head =", repo["head"])
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
        "delta_C22 = "
        "delta_R22_complement + "
        "delta_K22"
    )
    print(
        "C22_observation = direct frozen-parent runtime capture"
    )
    print(
        "K22_definition = hidden_states * D"
    )
    print(
        "K22_observation = reconstructed from authenticated "
        "slow_forward frame operands"
    )
    print(
        "R22_definition = C22 - K22"
    )
    print(
        "R22_direct_runtime_capture = False"
    )
    print(
        "additive_closure_rel_tol =",
        ADDITIVE_CLOSURE_REL_TOL,
    )
    print(
        "mixer_forward_sha256 =",
        output_parent.EXPECTED_MIXER_FORWARD_SHA256,
    )
    print(
        "slow_forward_sha256 =",
        output_parent.EXPECTED_SLOW_FORWARD_SHA256,
    )
    print(
        "mamba_source_sha256 =",
        source["source_sha256"],
    )
    print(
        "parent_delta_c22_summary_preserved_by_design = True"
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
            "PASS_LAYER22_RECURRENT_SKIP_FACTORIZATION_STATIC_PREFLIGHT"
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
        output_parent,
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

    except RecurrentSkipAuditError as exc:
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
