"""K0-RVG layer-22 recurrent-readout state / readout-C factorization audit.

Authenticated active sequential identity at layer 22:

    R_t[d] = sum_n S_t[d, n] * C_t[n]

where S_t is the post-update recurrent state consumed by the readout and C_t
is the runtime readout projection vector from the same slow_forward frame.

For matched/swapped branches, the exact symmetric bilinear decomposition is:

    delta_R_t = Q_state + Q_readout_C

    Q_state     := <C_bar_t, delta_S_t>
    Q_readout_C := <S_bar_t, delta_C_t>

The decomposition is evaluated in float64 from directly observed float32
runtime operands. A float32 operand reconstruction of the readout is compared
against the frozen parent R22 complement as a blocking numerical bridge, but
the parent complement is not retroactively relabeled as a direct runtime read.

This audit is observational/algebraic only. It does not tokenize, read logits,
execute task heads, train, intervene causally, or persist raw vectors.
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


PARENT_FREEZE_COMMIT = "f672fd6d9b15fa24465d39b07ca7c8c21178b99d"
PARENT_RUNNER_REL = "scripts/longterm_k0_rvg_layer22_recurrent_skip_factorization_audit.py"
PARENT_RUNNER_SHA256 = "fd7f02fa966af3359c479754f9d90e49df2e160aeca79ed43131b52f42c8d2fb"
PARENT_SUMMARY_REL = (
    "reports/longterm_k0_rvg_layer22_recurrent_skip_factorization_e8fea7a_v1/summary.json"
)
PARENT_SUMMARY_SHA256 = "220e01efe6358621f8bc3c33bdbad9b12a449c2fe0a52b2e75b9deab8e2414ba"

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FORWARD_COUNT = 1344
EXPECTED_LAYER_COUNT = 24
SOURCE_LAYER = 22
WIDTH = 1536
STATE_SIZE = 16
RELATIVE_COORDINATES = tuple(range(-1, 7))
POST_HORIZON = 6
EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"

ALGEBRAIC_CLOSURE_REL_TOL = 1e-12
FLOAT32_PARENT_R22_BRANCH_REL_TOL = 5e-6
PARENT_DELTA_MEDIAN_REL_TOL = 1e-13
PARENT_DELTA_MEDIAN_ABS_TOL = 1e-13

QUESTION = (
    "Within the validated layer-22 recurrent-readout-dominated C22 path, "
    "is recurrent-readout role separation primarily associated with the "
    "post-update state S_t, the readout projection C_t, or their vector interaction?"
)


class ReadoutFactorizationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ReadoutFactorizationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def import_module(path: Path, name: str):
    require(path.is_file(), f"MODULE_MISSING:{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"MODULE_SPEC_FAILURE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root).decode().strip()


def load_parent(root: Path):
    path = root / PARENT_RUNNER_REL
    require(path.is_file(), "PARENT_RUNNER_MISSING")
    require(sha256_bytes(path.read_bytes()) == PARENT_RUNNER_SHA256, "PARENT_RUNNER_SHA256_MISMATCH")
    return import_module(path, "k0_rvg_layer22_recurrent_skip_parent")


def build_plan(root: Path, parent: Any):
    gate_parent = parent.load_parent(root)
    values = parent.build_plan(root, gate_parent)
    require(len(values) == 18, "PARENT_BUILD_PLAN_ARITY_MISMATCH")
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
    ) = values
    require(len(plan) == EXPECTED_PAIR_ROLE_COUNT, "PLAN_COUNT_MISMATCH")
    require(len(cohort) == EXPECTED_COMMON_COUNT, "COMMON_COHORT_COUNT_MISMATCH")
    return (
        gate_parent,
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
    gate_parent: Any,
    output_parent: Any,
    residual_parent: Any,
    rms_parent: Any,
    hidden_parent: Any,
    four_tap: Any,
    postconv: Any,
    time_step: Any,
    base: Any,
):
    repo, _ = parent.authenticate_parent(
        root,
        gate_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        four_tap,
        postconv,
        time_step,
        base,
    )

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", PARENT_FREEZE_COMMIT, repo["head"]],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "PARENT_FREEZE_NOT_ANCESTOR")

    frozen = base.git_bytes(root, f"{PARENT_FREEZE_COMMIT}:{PARENT_SUMMARY_REL}")
    require(sha256_bytes(frozen) == PARENT_SUMMARY_SHA256, "PARENT_SUMMARY_SHA256_MISMATCH")
    current = root / PARENT_SUMMARY_REL
    require(current.is_file(), "PARENT_SUMMARY_MISSING")
    require(current.read_bytes() == frozen, "PARENT_SUMMARY_WORKTREE_DRIFT")

    summary = json.loads(frozen)
    require(
        summary.get("schema_version") == "k0-rvg-layer22-recurrent-skip-factorization-summary-v1",
        "PARENT_SCHEMA_MISMATCH",
    )
    require(summary.get("item_count") == EXPECTED_ITEM_COUNT, "PARENT_ITEM_COUNT_MISMATCH")
    require(summary.get("pair_role_count") == EXPECTED_PAIR_ROLE_COUNT, "PARENT_PAIR_ROLE_COUNT_MISMATCH")
    require(summary.get("common_ddsssss_item_count") == EXPECTED_COMMON_COUNT, "PARENT_COMMON_COUNT_MISMATCH")
    return repo, summary


def authenticate_observer_static(root: Path, base: Any, expected_source_sha256: str):
    observer_path = root / base.OBSERVER_REL
    observer = import_module(observer_path, "k0_rvg_readout_factorization_observer_static")
    require(hasattr(observer, "RawRecurrenceCollector"), "RAW_RECURRENCE_COLLECTOR_MISSING")
    require(hasattr(observer, "RecurrenceRecord"), "RECURRENCE_RECORD_MISSING")
    binding = observer.resolve_source_binding()
    require(binding.source_sha256 == expected_source_sha256, "OBSERVER_SOURCE_SHA256_MISMATCH")
    require(hasattr(binding, "readout_line"), "READOUT_LINE_BINDING_MISSING")
    require(hasattr(binding, "update_line"), "UPDATE_LINE_BINDING_MISSING")
    trace_src = inspect.getsource(observer.RawRecurrenceCollector._trace)
    require("s_post" in trace_src, "OBSERVER_S_POST_CAPTURE_MISSING")
    require("frame.f_locals.get(\"ssm_state\")" in trace_src, "OBSERVER_SSM_STATE_LOCAL_MISSING")
    require("self.records[key]" in trace_src, "OBSERVER_RECORD_WRITE_MISSING")
    require(hasattr(observer.RawRecurrenceCollector, "_coordinate"), "OBSERVER_COORDINATE_METHOD_MISSING")
    return observer, binding


def _cpu_clone(value: Any, role: str):
    import torch

    require(torch.is_tensor(value), f"{role}_NOT_TENSOR")
    out = value.detach().cpu().contiguous().clone()
    require(out.dtype == torch.float32, f"{role}_DTYPE_NOT_FLOAT32")
    require(out.device.type == "cpu", f"{role}_NOT_CPU")
    return out


def capture_readout_operands(
    parent: Any,
    gate_parent: Any,
    output_parent: Any,
    residual_parent: Any,
    rms_parent: Any,
    hidden_parent: Any,
    four_tap: Any,
    postconv: Any,
    dt_projection: Any,
    secant: Any,
    base: Any,
    observer: Any,
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

    class ReadoutOperandCollector(observer.RawRecurrenceCollector):
        def __init__(self, source_binding, layers, target_indices):
            super().__init__(source_binding, layers, target_indices)
            self._readout_binding = source_binding
            self.readout_c_records: dict[tuple[int, int], Any] = {}

        def _trace(self, frame: Any, event: str, arg: Any):
            super()._trace(frame, event, arg)

            if (
                frame.f_code is self._readout_binding.code
                and event == "line"
                and frame.f_lineno == self._readout_binding.readout_line
            ):
                key = self._coordinate(frame)
                if key is not None and key[0] == SOURCE_LAYER:
                    require(key not in self.readout_c_records, "DUPLICATE_READOUT_C_CAPTURE")
                    token_index = key[1]
                    c_full = frame.f_locals.get("C")
                    require(c_full is not None, "READOUT_C_LOCAL_MISSING")
                    c_full = _cpu_clone(c_full, "READOUT_C_FULL")
                    require(len(c_full.shape) == 3, "READOUT_C_RANK_MISMATCH")
                    require(c_full.shape[0] == 1, "READOUT_C_BATCH_MISMATCH")
                    require(c_full.shape[2] == STATE_SIZE, "READOUT_C_STATE_SIZE_MISMATCH")
                    require(0 <= token_index < c_full.shape[1], "READOUT_C_TOKEN_OUT_OF_RANGE")
                    self.readout_c_records[key] = c_full[:, token_index, :].contiguous().clone()

            return self._trace

    prior_trace = sys.gettrace()
    collector = ReadoutOperandCollector(binding, layer_map, targets)

    with collector.capture():
        records = parent.capture_ck(
            gate_parent,
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

    require(sys.gettrace() is prior_trace, "TRACE_RESTORATION_FAILURE")
    require(collector.records is not None, "RECURRENCE_RECORDS_MISSING")
    require(set(records) == set(targets), "PARENT_TARGET_SET_MISMATCH")

    branch_max_rel = 0.0

    for token in targets:
        key = (SOURCE_LAYER, token)
        require(key in collector.records, f"LAYER22_STATE_RECORD_MISSING:{token}")
        require(key in collector.readout_c_records, f"LAYER22_READOUT_C_MISSING:{token}")

        recurrence = collector.records[key]
        s_post = _cpu_clone(recurrence.s_post, "S_POST")
        c_readout = _cpu_clone(collector.readout_c_records[key], "C_READOUT")

        require(tuple(s_post.shape) == (1, WIDTH, STATE_SIZE), f"S_POST_SHAPE_MISMATCH:{token}")
        require(tuple(c_readout.shape) == (1, STATE_SIZE), f"C_READOUT_SHAPE_MISMATCH:{token}")

        r_operand_float32 = torch.matmul(
            s_post,
            c_readout.unsqueeze(-1),
        ).squeeze(-1)

        require(tuple(r_operand_float32.shape) == (1, WIDTH), f"R_OPERAND_SHAPE_MISMATCH:{token}")

        c22 = records[token]["C22"].to(torch.float64)
        k22 = records[token]["K22"].to(torch.float64)
        parent_r22 = c22 - k22

        residual = r_operand_float32.to(torch.float64) - parent_r22
        denom = max(
            float(torch.linalg.vector_norm(parent_r22).item()),
            float(torch.linalg.vector_norm(r_operand_float32.to(torch.float64)).item()),
            1e-12,
        )
        branch_rel = float(torch.linalg.vector_norm(residual).item() / denom)
        branch_max_rel = max(branch_max_rel, branch_rel)

        require(
            branch_rel <= FLOAT32_PARENT_R22_BRANCH_REL_TOL,
            f"READOUT_PARENT_R22_BRANCH_RECON_FAILURE:{token}:{branch_rel}",
        )

        records[token]["S22_POST"] = s_post
        records[token]["C22_READOUT"] = c_readout
        records[token]["R22_OPERAND_FLOAT32"] = r_operand_float32.contiguous().clone()
        records[token]["R22_PARENT_COMPLEMENT_FLOAT64"] = parent_r22.contiguous().clone()
        records[token]["readout_parent_branch_relative_residual"] = branch_rel

    return records, branch_max_rel


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

        sm32 = matched[token]["S22_POST"]
        ss32 = swapped[token]["S22_POST"]
        cm32 = matched[token]["C22_READOUT"]
        cs32 = swapped[token]["C22_READOUT"]

        if k == -1:
            require(torch.equal(sm32, ss32), f"K_MINUS_1_STATE_IDENTITY_FAILURE:{idx}:{role}")
            require(torch.equal(cm32, cs32), f"K_MINUS_1_READOUT_C_IDENTITY_FAILURE:{idx}:{role}")

        sm = sm32.to(torch.float64)
        ss = ss32.to(torch.float64)
        cm = cm32.to(torch.float64)
        cs = cs32.to(torch.float64)

        ds = sm - ss
        dc = cm - cs
        sbar = (sm + ss) * 0.5
        cbar = (cm + cs) * 0.5

        rm = torch.sum(sm * cm[:, None, :], dim=-1)
        rs = torch.sum(ss * cs[:, None, :], dim=-1)
        dr = rm - rs

        q_state = torch.sum(ds * cbar[:, None, :], dim=-1)
        q_readout_c = torch.sum(sbar * dc[:, None, :], dim=-1)
        qsum = q_state + q_readout_c

        closure_abs = float(torch.linalg.vector_norm(qsum - dr).item())
        dr_l2 = float(torch.linalg.vector_norm(dr).item())
        closure_rel = closure_abs / max(dr_l2, 1e-12)
        require(
            closure_rel <= ALGEBRAIC_CLOSURE_REL_TOL,
            f"READOUT_BILINEAR_CLOSURE_FAILURE:{idx}:{role}:{k}:{closure_rel}",
        )

        q_state_l2 = float(torch.linalg.vector_norm(q_state).item())
        q_readout_c_l2 = float(torch.linalg.vector_norm(q_readout_c).item())
        delta_s_l2 = float(torch.linalg.vector_norm(ds).item())
        delta_c_l2 = float(torch.linalg.vector_norm(dc).item())

        qs_sq = q_state_l2 * q_state_l2
        qc_sq = q_readout_c_l2 * q_readout_c_l2
        rss_sq = qs_sq + qc_sq
        rss_l2 = math.sqrt(rss_sq)
        dot = float(torch.sum(q_state * q_readout_c).item())
        normalized_cross = 2.0 * dot / rss_sq if rss_sq > 0.0 else 0.0
        state_energy_fraction = qs_sq / rss_sq if rss_sq > 0.0 else 0.0
        readout_c_energy_fraction = qc_sq / rss_sq if rss_sq > 0.0 else 0.0
        addition_factor = dr_l2 / rss_l2 if rss_l2 > 0.0 else 0.0

        dr_sq = dr_l2 * dr_l2
        state_additive_share = (qs_sq + dot) / dr_sq if dr_sq > 0.0 else 0.0
        readout_c_additive_share = (qc_sq + dot) / dr_sq if dr_sq > 0.0 else 0.0

        if dr_sq > 0.0:
            require(
                math.isclose(
                    state_additive_share + readout_c_additive_share,
                    1.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                f"READOUT_ADDITIVE_SHARE_CLOSURE_FAILURE:{idx}:{role}:{k}",
            )

        prm = matched[token]["R22_PARENT_COMPLEMENT_FLOAT64"]
        prs = swapped[token]["R22_PARENT_COMPLEMENT_FLOAT64"]
        parent_dr = prm - prs
        parent_dr_l2 = float(torch.linalg.vector_norm(parent_dr).item())
        operand_parent_delta_residual = float(torch.linalg.vector_norm(dr - parent_dr).item())
        operand_parent_delta_rel = operand_parent_delta_residual / max(parent_dr_l2, dr_l2, 1e-12)

        branch_rel_max = max(
            float(matched[token]["readout_parent_branch_relative_residual"]),
            float(swapped[token]["readout_parent_branch_relative_residual"]),
        )

        if k == -1:
            for value, name in (
                (dr_l2, "DR"),
                (q_state_l2, "Q_STATE"),
                (q_readout_c_l2, "Q_READOUT_C"),
                (rss_l2, "RSS"),
                (delta_s_l2, "DELTA_S"),
                (delta_c_l2, "DELTA_C"),
                (parent_dr_l2, "PARENT_DR"),
            ):
                require(value == 0.0, f"K_MINUS_1_{name}_NONZERO:{idx}:{role}:{value}")

        values = {
            "schema_version": "k0-rvg-layer22-recurrent-readout-factorization-row-v1",
            "local_template_index": idx,
            "stable_item_id": row["stable_item_id"],
            "role": role,
            "relative_coordinate": k,
            "token_index": token,
            "divergence_anchor_token_index": anchor,
            "token_equality_signature_k0_to_k6": signature,
            "in_common_ddsssss_cohort": idx in cohort,
            "delta_s22_post_l2": delta_s_l2,
            "delta_c22_readout_l2": delta_c_l2,
            "delta_r22_operand_l2": dr_l2,
            "q_state_l2": q_state_l2,
            "q_readout_c_l2": q_readout_c_l2,
            "rss_l2": rss_l2,
            "state_energy_fraction": state_energy_fraction,
            "readout_c_energy_fraction": readout_c_energy_fraction,
            "normalized_cross": normalized_cross,
            "addition_factor": addition_factor,
            "state_additive_share": state_additive_share,
            "readout_c_additive_share": readout_c_additive_share,
            "bilinear_closure_relative_residual": closure_rel,
            "parent_delta_r22_complement_l2": parent_dr_l2,
            "operand_vs_parent_delta_relative_residual": operand_parent_delta_rel,
            "max_branch_readout_vs_parent_relative_residual": branch_rel_max,
            "state_observation_mode": "direct_raw_recurrence_observer_s_post",
            "readout_c_observation_mode": "direct_slow_forward_readout_line_frame_local_C",
            "operand_algebra_dtype": "torch.float64",
            "runtime_operand_dtype": "torch.float32",
        }

        require(
            all(
                math.isfinite(float(v))
                for v in values.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ),
            f"NONFINITE_METRIC:{idx}:{role}:{k}",
        )
        output.append(values)

    return output


SUMMARY_FIELDS = (
    "delta_s22_post_l2",
    "delta_c22_readout_l2",
    "delta_r22_operand_l2",
    "q_state_l2",
    "q_readout_c_l2",
    "rss_l2",
    "state_energy_fraction",
    "readout_c_energy_fraction",
    "normalized_cross",
    "addition_factor",
    "state_additive_share",
    "readout_c_additive_share",
    "bilinear_closure_relative_residual",
    "parent_delta_r22_complement_l2",
    "operand_vs_parent_delta_relative_residual",
    "max_branch_readout_vs_parent_relative_residual",
)


def aggregate(values):
    vals = [float(v) for v in values]
    require(bool(vals), "EMPTY_AGGREGATE")
    require(all(math.isfinite(v) for v in vals), "NONFINITE_AGGREGATE")
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def aggregate_trajectory(rows):
    result = {"corr": {}, "ctrl": {}}
    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            bucket = [
                row
                for row in rows
                if row["role"] == role and int(row["relative_coordinate"]) == k
            ]
            require(bool(bucket), f"EMPTY_BUCKET:{role}:{k}")
            result[role][str(k)] = {
                field: aggregate(row[field] for row in bucket)
                for field in SUMMARY_FIELDS
            }
    return result


def make_summary(rows, cohort):
    require(
        len(rows) == EXPECTED_PAIR_ROLE_COUNT * len(RELATIVE_COORDINATES),
        "ROW_COUNT_MISMATCH",
    )
    common_rows = [row for row in rows if bool(row["in_common_ddsssss_cohort"])]
    require(
        len(common_rows) == EXPECTED_COMMON_COUNT * 2 * len(RELATIVE_COORDINATES),
        "COMMON_ROW_COUNT_MISMATCH",
    )
    return {
        "schema_version": "k0-rvg-layer22-recurrent-readout-factorization-summary-v1",
        "scientific_question": QUESTION,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count": len(cohort),
        "trajectory_row_count": len(rows),
        "source_layer": SOURCE_LAYER,
        "width": WIDTH,
        "state_size": STATE_SIZE,
        "relative_coordinates": list(RELATIVE_COORDINATES),
        "execution_protocol": EXECUTION_PROTOCOL,
        "identity": "delta_R22_operand = Q_state + Q_readout_C",
        "q_state_definition": "sum_n Cbar_t[n] * delta_S_t[d,n]",
        "q_readout_c_definition": "sum_n Sbar_t[d,n] * delta_C_t[n]",
        "full_336_trajectory": aggregate_trajectory(rows),
        "common_330_ddsssss_trajectory": aggregate_trajectory(common_rows),
        "max_bilinear_closure_relative_residual": max(
            float(row["bilinear_closure_relative_residual"]) for row in rows
        ),
        "max_operand_vs_parent_delta_relative_residual": max(
            float(row["operand_vs_parent_delta_relative_residual"]) for row in rows
        ),
        "max_branch_readout_vs_parent_relative_residual": max(
            float(row["max_branch_readout_vs_parent_relative_residual"]) for row in rows
        ),
        "raw_vectors_persisted": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_probe_or_learned_geometry_executed": False,
    }


def validate_parent_reproduction(summary, parent_summary):
    for trajectory in ("full_336_trajectory", "common_330_ddsssss_trajectory"):
        current = summary[trajectory]
        parent_traj = parent_summary[trajectory]
        for role in ("corr", "ctrl"):
            for k in RELATIVE_COORDINATES:
                got = float(current[role][str(k)]["parent_delta_r22_complement_l2"]["median"])
                expected = float(parent_traj[role][str(k)]["delta_r22_complement_l2"]["median"])
                require(
                    math.isclose(
                        got,
                        expected,
                        rel_tol=PARENT_DELTA_MEDIAN_REL_TOL,
                        abs_tol=PARENT_DELTA_MEDIAN_ABS_TOL,
                    ),
                    f"PARENT_DELTA_R22_REPRODUCTION_FAILURE:{trajectory}:{role}:{k}:{got}:{expected}",
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
    return b"".join(json_bytes(row) for row in rows)


def resolve_runtime_bundle(
    root,
    output_parent,
    residual_parent,
    rms_parent,
    hidden_parent,
    postconv,
    base,
    handoff_path,
):
    import torch

    (
        observer,
        k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        snapshot_info,
    ) = base.resolve_runtime(root, handoff_path)

    require(binding.source_sha256 == output_parent.EXPECTED_MAMBA_SOURCE_SHA256, "MAMBA_SOURCE_SHA256_MISMATCH")
    require(len(layer_map) == EXPECTED_LAYER_COUNT, "LAYER_MAP_COUNT_MISMATCH")
    require(SOURCE_LAYER in set(layer_map.values()), "SOURCE_LAYER_NOT_REGISTERED")

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

    require(bool(mixer22.use_mambapy) is False, "USE_MAMBAPY_NOT_FALSE")
    require(mixer22.training is False, "MIXER22_NOT_EVAL")
    require(tuple(mixer22.D.shape) == (WIDTH,), "D_SHAPE_MISMATCH")
    require(mixer22.D.dtype == torch.float32, "D_DTYPE_MISMATCH")

    return (
        observer,
        k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        snapshot_info,
        layer22,
        layer23,
        norm23,
        mixer23,
        mixer22,
        out_proj,
    )


def _prefixes(row):
    idx = int(row["local_template_index"])
    role = str(row["role"])
    anchor = int(row["anchor"])
    cutoff = anchor + POST_HORIZON + 1
    matched_prefix = tuple(row["matched_ids"][:cutoff])
    swapped_prefix = tuple(row["swapped_ids"][:cutoff])
    require(
        len(matched_prefix) == cutoff and len(swapped_prefix) == cutoff,
        f"PREFIX_LENGTH_FAILURE:{idx}:{role}",
    )
    return matched_prefix, swapped_prefix


def runtime_preflight(
    root,
    parent,
    gate_parent,
    output_parent,
    residual_parent,
    rms_parent,
    hidden_parent,
    four_tap,
    postconv,
    dt_projection,
    secant,
    base,
    plan,
    cohort,
    signatures,
    handoff_path,
):
    (
        observer,
        _k2s,
        model,
        binding,
        layer_map,
        _handoff,
        _encoder,
        _snapshot_info,
        layer22,
        layer23,
        norm23,
        mixer23,
        mixer22,
        out_proj,
    ) = resolve_runtime_bundle(
        root,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        handoff_path,
    )

    row = plan[0]
    idx = int(row["local_template_index"])
    role = str(row["role"])
    matched_prefix, swapped_prefix = _prefixes(row)

    matched, m_rel = capture_readout_operands(
        parent,
        gate_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        four_tap,
        postconv,
        dt_projection,
        secant,
        base,
        observer,
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
    swapped, s_rel = capture_readout_operands(
        parent,
        gate_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        four_tap,
        postconv,
        dt_projection,
        secant,
        base,
        observer,
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

    rows = metric_rows_for_pair(
        row,
        matched,
        swapped,
        cohort,
        signatures[(idx, role)],
    )

    print("PASS_LAYER22_RECURRENT_READOUT_RUNTIME_PREFLIGHT")
    print("pair_role =", idx, role)
    print("model_forward_count = 2")
    print("scientific_population_accessed = True")
    print("scientific_evidence_emitted = False")
    print("raw_vectors_persisted = False")
    print("max_branch_readout_vs_parent_relative_residual =", max(m_rel, s_rel))
    print(
        "max_bilinear_closure_relative_residual =",
        max(float(r["bilinear_closure_relative_residual"]) for r in rows),
    )
    print(
        "max_operand_vs_parent_delta_relative_residual =",
        max(float(r["operand_vs_parent_delta_relative_residual"]) for r in rows),
    )


def execute(
    root,
    parent,
    gate_parent,
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
    final_dir = output_dir.resolve()
    partial_dir = Path(str(final_dir) + ".partial")
    require(handoff_path.is_file(), f"HANDOFF_MISSING:{handoff_path}")
    require(not final_dir.exists(), f"OUTPUT_DIR_EXISTS:{final_dir}")
    require(not partial_dir.exists(), f"PARTIAL_OUTPUT_EXISTS:{partial_dir}")

    (
        observer,
        _k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        _snapshot_info,
        layer22,
        layer23,
        norm23,
        mixer23,
        mixer22,
        out_proj,
    ) = resolve_runtime_bundle(
        root,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        handoff_path,
    )

    rows = []
    forward_count = 0
    max_branch_rel = 0.0

    for n, row in enumerate(plan, start=1):
        idx = int(row["local_template_index"])
        role = str(row["role"])
        matched_prefix, swapped_prefix = _prefixes(row)

        matched, rel_m = capture_readout_operands(
            parent,
            gate_parent,
            output_parent,
            residual_parent,
            rms_parent,
            hidden_parent,
            four_tap,
            postconv,
            dt_projection,
            secant,
            base,
            observer,
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

        swapped, rel_s = capture_readout_operands(
            parent,
            gate_parent,
            output_parent,
            residual_parent,
            rms_parent,
            hidden_parent,
            four_tap,
            postconv,
            dt_projection,
            secant,
            base,
            observer,
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
        max_branch_rel = max(max_branch_rel, rel_m, rel_s)

        rows.extend(
            metric_rows_for_pair(
                row,
                matched,
                swapped,
                cohort,
                signatures[(idx, role)],
            )
        )

        if n % 16 == 0 or n == len(plan):
            print(
                f"PROGRESS pair_roles={n}/{len(plan)} model_forwards={forward_count}",
                flush=True,
            )

    require(forward_count == EXPECTED_FORWARD_COUNT, "FORWARD_COUNT_MISMATCH")

    summary = make_summary(rows, cohort)
    validate_parent_reproduction(summary, parent_summary)
    require(
        summary["max_branch_readout_vs_parent_relative_residual"] <= FLOAT32_PARENT_R22_BRANCH_REL_TOL,
        "SUMMARY_BRANCH_BRIDGE_TOLERANCE_FAILURE",
    )

    partial_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = partial_dir / "layer22_recurrent_readout_factorization_metrics.jsonl"
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "execution_manifest.json"

    metrics_path.write_bytes(jsonl_bytes(rows))
    summary_path.write_bytes(json_bytes(summary))

    manifest = {
        "schema_version": "k0-rvg-layer22-recurrent-readout-factorization-execution-manifest-v1",
        "runtime_git_head": repo["head"],
        "runtime_branch": repo["branch"],
        "parent_freeze_commit": PARENT_FREEZE_COMMIT,
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "parent_summary_sha256": PARENT_SUMMARY_SHA256,
        "scientific_question": QUESTION,
        "identity": "delta_R22_operand = Q_state + Q_readout_C",
        "execution_protocol": EXECUTION_PROTOCOL,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count": EXPECTED_COMMON_COUNT,
        "model_forward_count": forward_count,
        "source_layer": SOURCE_LAYER,
        "width": WIDTH,
        "state_size": STATE_SIZE,
        "mamba_source_sha256": binding.source_sha256,
        "slow_forward_sha256": output_parent.EXPECTED_SLOW_FORWARD_SHA256,
        "state_observation_mode": "existing_RawRecurrenceCollector_s_post",
        "readout_c_observation_mode": "same_forward_readout_line_frame_local_C",
        "capture_method": (
            "subclass existing RawRecurrenceCollector without changing recurrence semantics; "
            "reuse its exact s_post capture and additionally snapshot layer22 C[:,i,:] at the "
            "authenticated sequential readout line; parent hooks simultaneously preserve C22/K22"
        ),
        "float32_parent_r22_branch_rel_tol": FLOAT32_PARENT_R22_BRANCH_REL_TOL,
        "algebraic_closure_rel_tol": ALGEBRAIC_CLOSURE_REL_TOL,
        "handoff_zip_sha256": handoff["zip_sha256"],
        "checkpoint_sha256": handoff["checkpoint_sha256"],
        "encoder_canonical_digest": encoder["canonical_digest"],
        "encoder_raw_concat_digest": encoder["raw_concat_digest"],
        "parent_delta_r22_summary_match": True,
        "raw_vectors_persisted": False,
        "scientific_model_forward_executed": True,
        "scientific_s22_post_read": True,
        "scientific_readout_c22_read": True,
        "scientific_runtime_readout_operands_reconstructed": True,
        "scientific_parent_r22_complement_reused_as_bridge_only": True,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_probe_or_learned_geometry_executed": False,
        "runner_rel": Path(__file__).resolve().relative_to(root).as_posix(),
        "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "outputs": {
            "layer22_recurrent_readout_factorization_metrics.jsonl": sha256_bytes(metrics_path.read_bytes()),
            "summary.json": sha256_bytes(summary_path.read_bytes()),
        },
    }
    manifest_path.write_bytes(json_bytes(manifest))
    os.replace(partial_dir, final_dir)

    common = summary["common_330_ddsssss_trajectory"]
    print("PASS_LAYER22_RECURRENT_READOUT_FACTORIZATION_EXECUTION")
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print("parent_delta_r22_summary_match = True")
    print("max_branch_readout_vs_parent_relative_residual =", max_branch_rel)
    print("max_bilinear_closure_relative_residual =", summary["max_bilinear_closure_relative_residual"])
    print("max_operand_vs_parent_delta_relative_residual =", summary["max_operand_vs_parent_delta_relative_residual"])
    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[role][str(k)]
            print(
                f"{role}_k{k}_median "
                f"DR={t['delta_r22_operand_l2']['median']} "
                f"QS={t['q_state_l2']['median']} "
                f"QC={t['q_readout_c_l2']['median']} "
                f"STATE_E={t['state_energy_fraction']['median']} "
                f"C_E={t['readout_c_energy_fraction']['median']} "
                f"CROSS={t['normalized_cross']['median']} "
                f"ADD={t['addition_factor']['median']}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-preflight", action="store_true")
    parser.add_argument("--runtime-preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    require(
        sum(bool(v) for v in (args.static_preflight, args.runtime_preflight, args.execute)) == 1,
        "SELECT_EXACTLY_ONE_MODE",
    )

    root = Path.cwd().resolve()
    parent = load_parent(root)
    (
        gate_parent,
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
    ) = build_plan(root, parent)

    repo, parent_summary = authenticate_parent(
        root,
        parent,
        gate_parent,
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
    require(source["source_sha256"] == output_parent.EXPECTED_MAMBA_SOURCE_SHA256, "MAMBA_SOURCE_SHA256_MISMATCH")
    observer_static, observer_binding = authenticate_observer_static(
        root,
        base,
        output_parent.EXPECTED_MAMBA_SOURCE_SHA256,
    )

    print("=== LAYER22 RECURRENT READOUT FACTORIZATION AUDIT PLAN ===")
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print("parent_freeze_commit =", PARENT_FREEZE_COMMIT)
    print("parent_runner_sha256 =", PARENT_RUNNER_SHA256)
    print("parent_summary_sha256 =", PARENT_SUMMARY_SHA256)
    print("pair_role_count =", len(plan))
    print("common_ddsssss_item_count =", len(cohort))
    print("source_layer =", SOURCE_LAYER)
    print("state_size =", STATE_SIZE)
    print("scientific_question =", QUESTION)
    print("identity = delta_R22_operand = Q_state + Q_readout_C")
    print("Q_state = sum_n Cbar_t[n] * delta_S_t[d,n]")
    print("Q_readout_C = sum_n Sbar_t[d,n] * delta_C_t[n]")
    print("state_source = existing RawRecurrenceCollector.s_post")
    print("readout_C_source = authenticated slow_forward readout-line frame local C[:,i,:]")
    print("observer_readout_line =", observer_binding.readout_line)
    print("mamba_source_sha256 =", source["source_sha256"])
    print("parent_R22_complement = bridge/reproduction only")
    print("raw_vectors_persisted = False")
    print("tokenizer_invoked = False")
    print("training_executed = False")

    if args.static_preflight:
        print("scientific_model_forward_executed = False")
        print("PASS_LAYER22_RECURRENT_READOUT_FACTORIZATION_STATIC_PREFLIGHT")
        return 0

    require(args.handoff is not None, "RUNTIME_MODE_REQUIRES_HANDOFF")
    handoff_path = args.handoff.resolve()
    require(handoff_path.is_file(), f"HANDOFF_MISSING:{handoff_path}")

    if args.runtime_preflight:
        runtime_preflight(
            root,
            parent,
            gate_parent,
            output_parent,
            residual_parent,
            rms_parent,
            hidden_parent,
            four_tap,
            postconv,
            dt_projection,
            secant,
            base,
            plan,
            cohort,
            signatures,
            handoff_path,
        )
        return 0

    require(args.output_dir is not None, "EXECUTE_REQUIRES_OUTPUT_DIR")
    execute(
        root,
        parent,
        gate_parent,
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
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReadoutFactorizationError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"BLOCKED_UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
