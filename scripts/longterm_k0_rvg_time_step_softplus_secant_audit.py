"""K0-RVG selective time-step softplus secant audit.

Validated parent chain:

    Q_D = Q_T + Q_B

and the corr-specific k=2 discrete-B transient is dominated by Q_T.

Frozen slow-path semantics:

    Z = dt_proj(time_step)
    T = softplus(Z)

For each matched/swapped pair:

    delta_T = G_sec * delta_Z

where G_sec is the exact elementwise softplus secant gain.

The parent Q_T term is preserved as:

    M   = mean(U_m,U_s) * mean(B_m,B_s)
    P_Z = M * delta_Z
    Q_T = M * delta_T
        = G_sec * P_Z

This audit asks whether the validated Q_T transient is already present in the
pre-softplus projected drive P_Z, or whether changes in softplus operating-point
sensitivity materially shape it.

No causal intervention is performed.
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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence


PARENT_FREEZE_COMMIT = "ab798531b6b6349d961927a9255da244c5fa2dc0"

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_discrete_b_nested_decomposition_7dec742_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "b537e9b92aeb3070d0b9dadbd3835d2f070a60c73704a8e88fb0e4fb51056715"
)

PARENT_RUNNER_REL = (
    "scripts/longterm_k0_rvg_discrete_b_nested_decomposition.py"
)

PARENT_RUNNER_SHA256 = (
    "54080f6a508dfd841027a81f0646ac488237600c82f2a463ad5637644c88264a"
)

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)

EXPECTED_PRE_SOFTPLUS_LINE = 318
EXPECTED_UPDATE_LINE = 350

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FORWARD_COUNT = 1344

PRIMARY_LAYER = 23
POST_HORIZON = 6
RELATIVE_COORDINATES = tuple(range(-1, 7))

EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"

FACTORIZATION = (
    "T = softplus(Z); Z = dt_proj(time_step); "
    "delta_T = G_sec * delta_Z"
)

PARENT_PRESERVATION = (
    "Q_T = mean(U_m,U_s)*mean(B_m,B_s)*delta_T"
)

PRE_SOFTPLUS_DRIVE = (
    "P_Z = mean(U_m,U_s)*mean(B_m,B_s)*delta_Z"
)

COMPOSITION_RTOL = 1e-5
GAIN_TOL = 1e-6


class SoftplusSecantError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SoftplusSecantError(message)


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
        "k0_rvg_discrete_b_parent",
    )


def build_plan(root: Path, parent: Any):
    write = parent.load_write_runner(root)

    (
        carry,
        magnitude,
        base,
        plan,
        cohort,
        signatures,
    ) = parent.build_plan(root, write)

    require(
        len(plan) == EXPECTED_PAIR_ROLE_COUNT,
        "PLAN_COUNT_MISMATCH",
    )
    require(
        len(cohort) == EXPECTED_COMMON_COUNT,
        "COMMON_COHORT_COUNT_MISMATCH",
    )

    return write, carry, magnitude, base, plan, cohort, signatures


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

    require(rc == 0, "PARENT_FREEZE_NOT_ANCESTOR")

    raw = base.git_bytes(
        root,
        f"{PARENT_FREEZE_COMMIT}:{PARENT_SUMMARY_REL}",
    )

    require(
        sha256_bytes(raw) == PARENT_SUMMARY_SHA256,
        "PARENT_SUMMARY_SHA256_MISMATCH",
    )

    current = root / PARENT_SUMMARY_REL

    require(current.is_file(), "PARENT_SUMMARY_MISSING")
    require(
        current.read_bytes() == raw,
        "PARENT_SUMMARY_WORKTREE_DRIFT",
    )

    summary = json.loads(raw)

    require(
        summary.get("schema_version")
        == "k0-rvg-discrete-b-nested-decomposition-summary-v1",
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
        summary.get("execution_protocol") == EXECUTION_PROTOCOL,
        "PARENT_PROTOCOL_MISMATCH",
    )

    return repo, summary


def validate_frozen_source_semantics():
    import transformers
    import transformers.models.mamba.modeling_mamba as module

    require(
        transformers.__version__ == "5.12.1",
        "TRANSFORMERS_VERSION_MISMATCH",
    )

    fn = module.MambaMixer.slow_forward

    source_path = Path(
        inspect.getsourcefile(fn) or ""
    ).resolve()

    require(source_path.is_file(), "MAMBA_SOURCE_MISSING")

    raw = source_path.read_bytes()

    require(
        sha256_bytes(raw) == EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    tree = ast.parse(raw.decode("utf-8"))

    mixers = [
        n for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "MambaMixer"
    ]
    require(len(mixers) == 1, "MAMBA_MIXER_AMBIGUOUS")

    slow = [
        n for n in mixers[0].body
        if isinstance(n, ast.FunctionDef)
        and n.name == "slow_forward"
    ]
    require(len(slow) == 1, "SLOW_FORWARD_AMBIGUOUS")

    assignments = []

    for n in ast.walk(slow[0]):
        if not isinstance(n, ast.Assign):
            continue
        if len(n.targets) != 1:
            continue
        target = n.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == "discrete_time_step"
        ):
            assignments.append(n)

    assignments.sort(key=lambda n: n.lineno)

    require(
        len(assignments) == 2,
        "DISCRETE_TIME_STEP_ASSIGNMENT_COUNT_MISMATCH",
    )

    first, second = assignments

    require(
        int(second.lineno) == EXPECTED_PRE_SOFTPLUS_LINE,
        "PRE_SOFTPLUS_TRACE_LINE_MISMATCH",
    )

    first_names = {
        x.attr
        for x in ast.walk(first.value)
        if isinstance(x, ast.Attribute)
    }
    second_names = {
        x.attr
        for x in ast.walk(second.value)
        if isinstance(x, ast.Attribute)
    }

    require(
        "dt_proj" in first_names,
        "DT_PROJ_STRUCTURE_MISMATCH",
    )
    require(
        "softplus" in second_names,
        "SOFTPLUS_STRUCTURE_MISMATCH",
    )

    code_lines = {
        line
        for _, _, line in fn.__code__.co_lines()
        if line is not None
    }

    require(
        EXPECTED_PRE_SOFTPLUS_LINE in code_lines,
        "PRE_SOFTPLUS_LINE_NOT_EXECUTABLE",
    )
    require(
        EXPECTED_UPDATE_LINE in code_lines,
        "UPDATE_LINE_NOT_EXECUTABLE",
    )

    return {
        "source_path": str(source_path),
        "source_sha256": sha256_bytes(raw),
        "pre_softplus_line": EXPECTED_PRE_SOFTPLUS_LINE,
        "update_line": EXPECTED_UPDATE_LINE,
    }


def snapshot(value: Any, role: str):
    import torch

    require(
        isinstance(value, torch.Tensor),
        f"{role}_NOT_TENSOR",
    )
    require(
        bool(torch.isfinite(value).all().item()),
        f"{role}_NONFINITE",
    )

    out = value.detach().cpu().contiguous().clone()

    require(
        out.device.type == "cpu",
        f"{role}_NOT_CPU",
    )
    require(
        str(out.dtype) == "torch.float32",
        f"{role}_DTYPE_MISMATCH",
    )

    return out


def torch_equal(a: Any, b: Any) -> bool:
    import torch
    return bool(torch.equal(a, b))


class SoftplusSecantCollector:
    """Capture pre-softplus Z and parent U/T/B/W at layer 23."""

    def __init__(
        self,
        binding: Any,
        layer_map: Mapping[int, int],
        target_indices: Sequence[int],
    ):
        self.binding = binding
        self.layer_map = dict(layer_map)
        self.targets = tuple(sorted({int(v) for v in target_indices}))

        require(bool(self.targets), "EMPTY_TARGET_SET")

        self.z_records: dict[int, Any] | None = None
        self.records: dict[int, dict[str, Any]] | None = None
        self.prior_trace = None
        self.used = False

    def _trace(self, frame: Any, event: str, arg: Any):
        if frame.f_code is not self.binding.code or event != "line":
            return self._trace

        mixer = frame.f_locals.get("self")
        layer = self.layer_map.get(id(mixer))

        if layer != PRIMARY_LAYER:
            return self._trace

        if frame.f_lineno == EXPECTED_PRE_SOFTPLUS_LINE:
            require(
                self.z_records is not None,
                "TRACE_NOT_ACTIVE",
            )
            require(
                self.z_records == {},
                "DUPLICATE_PRE_SOFTPLUS_CAPTURE",
            )

            z_full = frame.f_locals.get("discrete_time_step")

            require(
                z_full is not None,
                "PRE_SOFTPLUS_Z_MISSING",
            )

            z_full = snapshot(
                z_full,
                "PRE_SOFTPLUS_Z_FULL",
            )

            require(
                len(z_full.shape) == 3,
                "PRE_SOFTPLUS_Z_RANK_MISMATCH",
            )
            require(
                z_full.shape[0] == 1,
                "PRE_SOFTPLUS_Z_BATCH_MISMATCH",
            )
            require(
                z_full.shape[2] == 1536,
                "PRE_SOFTPLUS_Z_WIDTH_MISMATCH",
            )

            for i in self.targets:
                require(
                    i < z_full.shape[1],
                    f"TARGET_OUT_OF_RANGE:{i}",
                )
                self.z_records[i] = (
                    z_full[:, i, :]
                    .contiguous()
                    .clone()
                )

            return self._trace

        if frame.f_lineno != EXPECTED_UPDATE_LINE:
            return self._trace

        require(
            self.records is not None
            and self.z_records is not None,
            "TRACE_NOT_ACTIVE",
        )

        i = frame.f_locals.get("i")

        require(
            type(i) is int and i >= 0,
            "AMBIGUOUS_TOKEN_INDEX",
        )

        if i not in self.targets:
            return self._trace

        require(
            i in self.z_records,
            f"Z_NOT_CAPTURED:{i}",
        )
        require(
            i not in self.records,
            f"DUPLICATE_UPDATE_CAPTURE:{i}",
        )

        hidden_states = frame.f_locals.get("hidden_states")
        discrete_time_step = frame.f_locals.get(
            "discrete_time_step"
        )
        B = frame.f_locals.get("B")
        deltaB_u = frame.f_locals.get("deltaB_u")

        require(hidden_states is not None, "U_MISSING")
        require(discrete_time_step is not None, "T_MISSING")
        require(B is not None, "B_MISSING")
        require(deltaB_u is not None, "W_MISSING")

        U = snapshot(
            hidden_states[:, :, i],
            "U",
        )
        T = snapshot(
            discrete_time_step[:, :, i],
            "T",
        )
        B_i = snapshot(
            B[:, i, :],
            "B",
        )
        W = snapshot(
            deltaB_u[:, :, i, :],
            "W",
        )
        Z = self.z_records[i]

        require(tuple(U.shape) == (1, 1536), "U_SHAPE_MISMATCH")
        require(tuple(T.shape) == (1, 1536), "T_SHAPE_MISMATCH")
        require(tuple(Z.shape) == (1, 1536), "Z_SHAPE_MISMATCH")
        require(tuple(B_i.shape) == (1, 16), "B_SHAPE_MISMATCH")
        require(
            tuple(W.shape) == (1, 1536, 16),
            "W_SHAPE_MISMATCH",
        )

        import torch

        reconstructed_T = torch.nn.functional.softplus(Z)

        require(
            torch_equal(reconstructed_T, T),
            "SOFTPLUS_EXACT_RECONSTRUCTION_FAILURE",
        )

        self.records[i] = {
            "U": U,
            "Z": Z,
            "T": T,
            "B": B_i,
            "W": W,
        }

        return self._trace

    @contextmanager
    def capture(self):
        require(not self.used, "TRACE_COLLECTOR_REUSE")

        self.z_records = {}
        self.records = {}
        self.prior_trace = sys.gettrace()

        sys.settrace(self._trace)

        completed = False

        try:
            yield self
            completed = True
        finally:
            sys.settrace(self.prior_trace)
            self.used = True

        if completed:
            target_set = set(self.targets)

            require(
                set(self.z_records) == target_set,
                "Z_TARGET_SET_MISMATCH",
            )
            require(
                set(self.records) == target_set,
                "RECORD_TARGET_SET_MISMATCH",
            )


def capture_factors(
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    collector = SoftplusSecantCollector(
        binding,
        layer_map,
        targets,
    )

    with collector.capture():
        base.direct_backbone_forward(
            model,
            token_ids,
        )

    require(
        collector.records is not None,
        "CAPTURE_RECORDS_MISSING",
    )

    return collector.records


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

    provisional = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k

        Um32 = matched[token]["U"]
        Zm32 = matched[token]["Z"]
        Tm32 = matched[token]["T"]
        Bm32 = matched[token]["B"]
        Wm32 = matched[token]["W"]

        Us32 = swapped[token]["U"]
        Zs32 = swapped[token]["Z"]
        Ts32 = swapped[token]["T"]
        Bs32 = swapped[token]["B"]
        Ws32 = swapped[token]["W"]

        u_equal = torch_equal(Um32, Us32)
        z_equal = torch_equal(Zm32, Zs32)
        t_equal = torch_equal(Tm32, Ts32)
        b_equal = torch_equal(Bm32, Bs32)
        w_equal = torch_equal(Wm32, Ws32)

        if k == -1:
            require(
                u_equal
                and z_equal
                and t_equal
                and b_equal
                and w_equal,
                f"K_MINUS_1_IDENTITY_FAILURE:{idx}:{role}",
            )

        Um = Um32.to(torch.float64)
        Zm = Zm32.to(torch.float64)
        Tm = Tm32.to(torch.float64)
        Bm = Bm32.to(torch.float64)
        Wm = Wm32.to(torch.float64)

        Us = Us32.to(torch.float64)
        Zs = Zs32.to(torch.float64)
        Ts = Ts32.to(torch.float64)
        Bs = Bs32.to(torch.float64)
        Ws = Ws32.to(torch.float64)

        mean_U = 0.5 * (Um + Us)
        mean_B = 0.5 * (Bm + Bs)

        delta_Z = Zm - Zs
        delta_T = Tm - Ts
        delta_W = Wm - Ws

        zero = delta_Z == 0
        nonzero = ~zero

        require(
            bool(torch.all(delta_T[zero] == 0).item()),
            f"ZERO_DELTA_Z_NONZERO_DELTA_T:{idx}:{role}:{k}",
        )

        gain = torch.empty_like(delta_Z)

        gain[nonzero] = (
            delta_T[nonzero]
            / delta_Z[nonzero]
        )

        if bool(zero.any().item()):
            gain[zero] = torch.sigmoid(
                0.5 * (Zm[zero] + Zs[zero])
            )

        require(
            bool(torch.isfinite(gain).all().item()),
            f"NONFINITE_SECANT_GAIN:{idx}:{role}:{k}",
        )

        gain_min = float(gain.min().item())
        gain_max = float(gain.max().item())

        require(
            gain_min >= -GAIN_TOL
            and gain_max <= 1.0 + GAIN_TOL,
            (
                f"SECANT_GAIN_RANGE_FAILURE:"
                f"{idx}:{role}:{k}:{gain_min}:{gain_max}"
            ),
        )

        M = (
            mean_U[:, :, None]
            * mean_B[:, None, :]
        )

        p_z = (
            M
            * delta_Z[:, :, None]
        )

        q_t = (
            M
            * delta_T[:, :, None]
        )

        reconstructed_q_t = (
            gain[:, :, None]
            * p_z
        )

        residual = q_t - reconstructed_q_t

        p_z_l2 = float(
            torch.linalg.vector_norm(p_z).item()
        )
        q_t_l2 = float(
            torch.linalg.vector_norm(q_t).item()
        )
        delta_z_l2 = float(
            torch.linalg.vector_norm(delta_Z).item()
        )
        delta_t_l2 = float(
            torch.linalg.vector_norm(delta_T).item()
        )
        delta_w_l2 = float(
            torch.linalg.vector_norm(delta_W).item()
        )
        residual_l2 = float(
            torch.linalg.vector_norm(residual).item()
        )

        residual_relative = (
            residual_l2
            / max(q_t_l2, 1e-12)
        )

        effective_gain = (
            q_t_l2 / p_z_l2
            if p_z_l2 > 0.0
            else 0.0
        )

        require(
            effective_gain
            <= 1.0 + GAIN_TOL,
            (
                f"EFFECTIVE_GAIN_GT_ONE:"
                f"{idx}:{role}:{k}:{effective_gain}"
            ),
        )

        require(
            residual_relative <= COMPOSITION_RTOL,
            (
                f"SECANT_RECONSTRUCTION_RESIDUAL_TOO_LARGE:"
                f"{idx}:{role}:{k}:{residual_relative}"
            ),
        )

        provisional.append(
            {
                "schema_version":
                    "k0-rvg-time-step-softplus-secant-row-v1",
                "local_template_index": idx,
                "stable_item_id": row["stable_item_id"],
                "role": role,
                "relative_coordinate": k,
                "token_index": token,
                "divergence_anchor_token_index": anchor,
                "token_equality_signature_k0_to_k6":
                    signature,
                "in_common_ddsssss_cohort":
                    idx in cohort,
                "hidden_states_exact_equal":
                    u_equal,
                "pre_softplus_z_exact_equal":
                    z_equal,
                "discrete_time_step_exact_equal":
                    t_equal,
                "B_exact_equal":
                    b_equal,
                "write_exact_equal":
                    w_equal,
                "delta_z_l2":
                    delta_z_l2,
                "delta_t_l2":
                    delta_t_l2,
                "p_z_l2":
                    p_z_l2,
                "q_t_l2":
                    q_t_l2,
                "delta_write_l2":
                    delta_w_l2,
                "effective_softplus_gain":
                    effective_gain,
                "secant_gain_min":
                    gain_min,
                "secant_gain_max":
                    gain_max,
                "composition_residual_l2":
                    residual_l2,
                "composition_relative_residual":
                    residual_relative,
                "source_snapshot_dtype":
                    "torch.float32",
                "metric_accumulation_dtype":
                    "torch.float64",
            }
        )

    k0_w = next(
        float(r["delta_write_l2"])
        for r in provisional
        if int(r["relative_coordinate"]) == 0
    )

    require(
        k0_w > 0.0
        and math.isfinite(k0_w),
        f"K0_WRITE_MAGNITUDE_INVALID:{idx}:{role}",
    )

    for r in provisional:
        r["p_z_l2_over_k0_w"] = (
            float(r["p_z_l2"]) / k0_w
        )
        r["q_t_l2_over_k0_w"] = (
            float(r["q_t_l2"]) / k0_w
        )

    return provisional


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
    fields = (
        "delta_z_l2",
        "delta_t_l2",
        "p_z_l2",
        "q_t_l2",
        "delta_write_l2",
        "p_z_l2_over_k0_w",
        "q_t_l2_over_k0_w",
        "effective_softplus_gain",
        "composition_relative_residual",
    )

    out = {"corr": {}, "ctrl": {}}

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
                for field in fields
            }

    return out


def make_summary(rows, cohort):
    require(
        len(rows)
        == EXPECTED_PAIR_ROLE_COUNT
        * len(RELATIVE_COORDINATES),
        "ROW_COUNT_MISMATCH",
    )

    common_rows = [
        r
        for r in rows
        if bool(r["in_common_ddsssss_cohort"])
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
            "k0-rvg-time-step-softplus-secant-summary-v1",
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
        "relative_coordinates":
            list(RELATIVE_COORDINATES),
        "execution_protocol":
            EXECUTION_PROTOCOL,
        "factorization":
            FACTORIZATION,
        "parent_preservation":
            PARENT_PRESERVATION,
        "pre_softplus_drive":
            PRE_SOFTPLUS_DRIVE,
        "full_336_trajectory":
            aggregate_trajectory(rows),
        "common_330_ddsssss_trajectory":
            aggregate_trajectory(common_rows),
        "max_composition_relative_residual":
            max(
                float(r["composition_relative_residual"])
                for r in rows
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


def validate_parent_q_t(summary, parent_summary):
    current = summary[
        "common_330_ddsssss_trajectory"
    ]
    parent = parent_summary[
        "common_330_ddsssss_trajectory"
    ]

    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            got = float(
                current[role][str(k)]
                ["q_t_l2_over_k0_w"]["median"]
            )
            expected = float(
                parent[role][str(k)]
                ["q_t_l2_over_k0_w"]["median"]
            )

            require(
                math.isclose(
                    got,
                    expected,
                    rel_tol=1e-13,
                    abs_tol=1e-13,
                ),
                (
                    "PARENT_Q_T_MEDIAN_MISMATCH:"
                    f"{role}:{k}:{got}:{expected}"
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
        binding.update_line == EXPECTED_UPDATE_LINE,
        "UPDATE_LINE_MISMATCH",
    )

    rows = []
    forward_count = 0

    for n, row in enumerate(plan, start=1):
        idx = int(row["local_template_index"])
        role = str(row["role"])
        anchor = int(row["anchor"])

        cutoff = anchor + POST_HORIZON + 1

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
            base,
            model,
            binding,
            layer_map,
            matched_prefix,
            row["targets"],
        )
        forward_count += 1

        swapped = capture_factors(
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
                row,
                matched,
                swapped,
                cohort,
                signatures[(idx, role)],
            )
        )

        if n % 16 == 0 or n == len(plan):
            print(
                f"PROGRESS pair_roles={n}/{len(plan)} "
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

    validate_parent_q_t(
        summary,
        parent_summary,
    )

    partial_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metrics_path = (
        partial_dir
        / "time_step_softplus_secant_metrics.jsonl"
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
            "k0-rvg-time-step-softplus-secant-execution-manifest-v1",
        "runtime_git_head":
            repo["head"],
        "runtime_branch":
            repo["branch"],
        "parent_freeze_commit":
            PARENT_FREEZE_COMMIT,
        "execution_protocol":
            EXECUTION_PROTOCOL,
        "factorization":
            FACTORIZATION,
        "parent_preservation":
            PARENT_PRESERVATION,
        "pre_softplus_drive":
            PRE_SOFTPLUS_DRIVE,
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
        "pre_softplus_trace_line":
            EXPECTED_PRE_SOFTPLUS_LINE,
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
        "parent_q_t_summary_match":
            True,
        "source_snapshot_dtype":
            "torch.float32",
        "metric_accumulation_dtype":
            "torch.float64",
        "raw_vectors_persisted":
            False,
        "scientific_model_forward_executed":
            True,
        "scientific_softplus_factor_read":
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
            "time_step_softplus_secant_metrics.jsonl":
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
        "PASS_TIME_STEP_SOFTPLUS_SECANT_EXECUTION"
    )
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print("parent_q_t_summary_match = True")

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[role][str(k)]

            print(
                f"{role}_k{k}_median "
                f"P_Z={t['p_z_l2_over_k0_w']['median']} "
                f"Q_T={t['q_t_l2_over_k0_w']['median']} "
                f"G_eff={t['effective_softplus_gain']['median']}"
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

    parent = load_parent_runner(root)

    (
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

    source = validate_frozen_source_semantics()

    print(
        "=== TIME-STEP SOFTPLUS SECANT AUDIT PLAN ==="
    )
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print(
        "parent_freeze_commit =",
        PARENT_FREEZE_COMMIT,
    )
    print("item_count =", EXPECTED_ITEM_COUNT)
    print("pair_role_count =", len(plan))
    print(
        "common_ddsssss_item_count =",
        len(cohort),
    )
    print("primary_layer =", PRIMARY_LAYER)
    print(
        "relative_coordinates =",
        list(RELATIVE_COORDINATES),
    )
    print(
        "execution_protocol =",
        EXECUTION_PROTOCOL,
    )
    print("factorization =", FACTORIZATION)
    print(
        "parent_preservation =",
        PARENT_PRESERVATION,
    )
    print(
        "pre_softplus_drive =",
        PRE_SOFTPLUS_DRIVE,
    )
    print(
        "mamba_source_sha256 =",
        source["source_sha256"],
    )
    print(
        "pre_softplus_trace_line =",
        source["pre_softplus_line"],
    )
    print("parent_q_t_preserved = True")
    print("tokenizer_invoked = False")
    print(
        "scientific_model_forward_executed = False"
    )

    if args.static_preflight:
        print(
            "PASS_TIME_STEP_SOFTPLUS_SECANT_STATIC_PREFLIGHT"
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
    except SoftplusSecantError as exc:
        print(
            f"BLOCKED: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception as exc:
        print(
            f"BLOCKED_UNEXPECTED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise