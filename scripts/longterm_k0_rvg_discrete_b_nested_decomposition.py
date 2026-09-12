"""K0-RVG nested decomposition of the validated discrete-B write contribution.

Parent write decomposition:

    delta_W = Q_U + Q_D
    Q_D = mean(U_m,U_s) * (D_m - D_s)

where:

    D = discrete_B
    U = conv-activated hidden_states

Frozen slow-path factorization:

    discrete_B = discrete_time_step * B

This runner preserves the parent Q_D term and decomposes it exactly as:

    Q_D = Q_T + Q_B

    Q_T = mean(U) * mean(B) * delta_T
    Q_B = mean(U) * mean(T) * delta_B

with broadcasting over intermediate/state dimensions.

Scope:
- frozen token IDs only,
- same divergence anchors and equal-length prefix protocol,
- CPU sequential Transformers 5.12.1 Mamba slow path,
- layer 23, k=-1..+6,
- common DDSSSSS 330-item cohort primary,
- no tokenizer, logits, task heads, training, intervention, PCA, or probes,
- raw vectors are not persisted.
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


PARENT_FREEZE_COMMIT = "076ffca0e2305442327f698cf012e9070e3bf69c"

PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_write_factor_decomposition_846b15d_v1/"
    "summary.json"
)

PARENT_SUMMARY_SHA256 = (
    "583fb225b62bab5c3e19010bd4c9869061c6d30edb5a71218d697bbb60c1c2c0"
)

WRITE_RUNNER_REL = (
    "scripts/longterm_k0_rvg_write_factor_decomposition.py"
)

WRITE_RUNNER_SHA256 = (
    "eadeacbae12b25cd30d077de9d15b359772b252a49bc26505c4b0c88679edf90"
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

FACTORIZATION = (
    "D = discrete_B = discrete_time_step * B"
)

DECOMPOSITION = (
    "Q_D = Q_T + Q_B; "
    "Q_D = mean(U_m,U_s)*(D_m-D_s); "
    "Q_T = mean(U_m,U_s)*mean(B_m,B_s)*(T_m-T_s); "
    "Q_B = mean(U_m,U_s)*mean(T_m,T_s)*(B_m-B_s)"
)

COMPOSITION_RTOL = 1e-5


class NestedDiscreteBError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise NestedDiscreteBError(message)


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


def load_write_runner(root: Path):
    path = root / WRITE_RUNNER_REL

    require(
        sha256_bytes(path.read_bytes()) == WRITE_RUNNER_SHA256,
        "WRITE_RUNNER_SHA256_MISMATCH",
    )

    return import_module(
        path,
        "k0_rvg_write_factor_parent",
    )


def build_plan(root: Path, write: Any):
    carry = write.load_parent_runner(root)

    magnitude, base, plan, cohort, signatures = write.build_plan(
        root,
        carry,
    )

    require(
        len(plan) == EXPECTED_PAIR_ROLE_COUNT,
        "PLAN_COUNT_MISMATCH",
    )
    require(
        len(cohort) == EXPECTED_COMMON_COUNT,
        "COMMON_COHORT_COUNT_MISMATCH",
    )

    return carry, magnitude, base, plan, cohort, signatures


def authenticate_parent(root: Path, base: Any):
    repo = base.authenticate_repo(root)

    ancestor = subprocess.call(
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
        ancestor == 0,
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
        == "k0-rvg-write-factor-decomposition-summary-v1",
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
    require(
        summary.get("factorization")
        == "W = discrete_B * hidden_states",
        "PARENT_FACTORIZATION_MISMATCH",
    )

    return repo, summary


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


class NestedFactorCollector:
    """Capture U, T, B, D, W at the frozen recurrence update line."""

    def __init__(
        self,
        binding: Any,
        layer_map: Mapping[int, int],
        target_indices: Sequence[int],
    ):
        self.binding = binding
        self.layer_map = dict(layer_map)
        self.targets = frozenset(int(v) for v in target_indices)

        self.records: dict[int, dict[str, Any]] | None = None
        self.prior_trace = None
        self.used = False

    def _trace(self, frame: Any, event: str, arg: Any):
        if (
            frame.f_code is not self.binding.code
            or event != "line"
            or frame.f_lineno != self.binding.update_line
        ):
            return self._trace

        mixer = frame.f_locals.get("self")
        layer = self.layer_map.get(id(mixer))

        if layer != PRIMARY_LAYER:
            return self._trace

        i = frame.f_locals.get("i")

        require(
            type(i) is int and i >= 0,
            "AMBIGUOUS_TOKEN_INDEX",
        )

        if i not in self.targets:
            return self._trace

        require(
            self.records is not None,
            "TRACE_NOT_ACTIVE",
        )
        require(
            i not in self.records,
            "DUPLICATE_TOKEN_CAPTURE",
        )

        hidden_states = frame.f_locals.get("hidden_states")
        discrete_time_step = frame.f_locals.get(
            "discrete_time_step"
        )
        B = frame.f_locals.get("B")
        discrete_B = frame.f_locals.get("discrete_B")
        deltaB_u = frame.f_locals.get("deltaB_u")

        require(
            hidden_states is not None,
            "HIDDEN_STATES_MISSING",
        )
        require(
            discrete_time_step is not None,
            "DISCRETE_TIME_STEP_MISSING",
        )
        require(
            B is not None,
            "B_MISSING",
        )
        require(
            discrete_B is not None,
            "DISCRETE_B_MISSING",
        )
        require(
            deltaB_u is not None,
            "DELTAB_U_MISSING",
        )

        U = snapshot(
            hidden_states[:, :, i],
            "HIDDEN_STATES",
        )
        T = snapshot(
            discrete_time_step[:, :, i],
            "DISCRETE_TIME_STEP",
        )
        B_i = snapshot(
            B[:, i, :],
            "B",
        )
        D = snapshot(
            discrete_B[:, :, i, :],
            "DISCRETE_B",
        )
        W = snapshot(
            deltaB_u[:, :, i, :],
            "WRITE",
        )

        require(
            tuple(U.shape) == (1, 1536),
            "U_SHAPE_MISMATCH",
        )
        require(
            tuple(T.shape) == (1, 1536),
            "T_SHAPE_MISMATCH",
        )
        require(
            tuple(B_i.shape) == (1, 16),
            "B_SHAPE_MISMATCH",
        )
        require(
            tuple(D.shape) == (1, 1536, 16),
            "D_SHAPE_MISMATCH",
        )
        require(
            tuple(W.shape) == (1, 1536, 16),
            "W_SHAPE_MISMATCH",
        )

        reconstructed_D = (
            T[:, :, None] * B_i[:, None, :]
        )

        require(
            torch_equal(reconstructed_D, D),
            "DISCRETE_B_EXACT_FACTORIZATION_FAILURE",
        )

        reconstructed_W = (
            D * U[:, :, None]
        )

        require(
            torch_equal(reconstructed_W, W),
            "WRITE_EXACT_FACTORIZATION_FAILURE",
        )

        self.records[i] = {
            "U": U,
            "T": T,
            "B": B_i,
            "D": D,
            "W": W,
        }

        return self._trace

    @contextmanager
    def capture(self):
        require(
            not self.used,
            "TRACE_COLLECTOR_REUSE",
        )

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
            require(
                set(self.records) == set(self.targets),
                "CAPTURE_COORDINATE_SET_MISMATCH",
            )


def capture_factors(
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    collector = NestedFactorCollector(
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
        Tm32 = matched[token]["T"]
        Bm32 = matched[token]["B"]
        Dm32 = matched[token]["D"]
        Wm32 = matched[token]["W"]

        Us32 = swapped[token]["U"]
        Ts32 = swapped[token]["T"]
        Bs32 = swapped[token]["B"]
        Ds32 = swapped[token]["D"]
        Ws32 = swapped[token]["W"]

        u_equal = torch_equal(Um32, Us32)
        t_equal = torch_equal(Tm32, Ts32)
        b_equal = torch_equal(Bm32, Bs32)
        d_equal = torch_equal(Dm32, Ds32)
        w_equal = torch_equal(Wm32, Ws32)

        if k == -1:
            require(
                (
                    u_equal
                    and t_equal
                    and b_equal
                    and d_equal
                    and w_equal
                ),
                f"K_MINUS_1_IDENTITY_FAILURE:{idx}:{role}",
            )

        Um = Um32.to(torch.float64)
        Tm = Tm32.to(torch.float64)
        Bm = Bm32.to(torch.float64)
        Dm = Dm32.to(torch.float64)
        Wm = Wm32.to(torch.float64)

        Us = Us32.to(torch.float64)
        Ts = Ts32.to(torch.float64)
        Bs = Bs32.to(torch.float64)
        Ds = Ds32.to(torch.float64)
        Ws = Ws32.to(torch.float64)

        mean_U = 0.5 * (Um + Us)
        mean_T = 0.5 * (Tm + Ts)
        mean_B = 0.5 * (Bm + Bs)

        delta_T = Tm - Ts
        delta_B = Bm - Bs
        delta_D = Dm - Ds
        delta_W = Wm - Ws

        # Parent discrete-B-side write contribution.
        q_d = (
            mean_U[:, :, None]
            * delta_D
        )

        # Exact nested symmetric decomposition.
        q_t = (
            mean_U[:, :, None]
            * delta_T[:, :, None]
            * mean_B[:, None, :]
        )

        q_b = (
            mean_U[:, :, None]
            * mean_T[:, :, None]
            * delta_B[:, None, :]
        )

        reconstructed = q_t + q_b
        residual = q_d - reconstructed

        q_d_l2 = float(
            torch.linalg.vector_norm(q_d).item()
        )
        q_t_l2 = float(
            torch.linalg.vector_norm(q_t).item()
        )
        q_b_l2 = float(
            torch.linalg.vector_norm(q_b).item()
        )
        delta_w_l2 = float(
            torch.linalg.vector_norm(delta_W).item()
        )

        interaction = float(
            2.0 * torch.sum(q_t * q_b).item()
        )

        residual_l2 = float(
            torch.linalg.vector_norm(residual).item()
        )

        residual_relative = (
            residual_l2
            / max(q_d_l2, 1e-12)
        )

        values = (
            q_d_l2,
            q_t_l2,
            q_b_l2,
            delta_w_l2,
            interaction,
            residual_l2,
            residual_relative,
        )

        require(
            all(math.isfinite(v) for v in values),
            f"NONFINITE_METRIC:{idx}:{role}:{k}",
        )

        require(
            residual_relative <= COMPOSITION_RTOL,
            (
                "NESTED_COMPOSITION_RESIDUAL_TOO_LARGE:"
                f"{idx}:{role}:{k}:{residual_relative}"
            ),
        )

        provisional.append(
            {
                "schema_version":
                    "k0-rvg-discrete-b-nested-decomposition-row-v1",
                "local_template_index": idx,
                "stable_item_id": row["stable_item_id"],
                "role": role,
                "relative_coordinate": k,
                "token_index": token,
                "divergence_anchor_token_index":
                    anchor,
                "token_equality_signature_k0_to_k6":
                    signature,
                "in_common_ddsssss_cohort":
                    idx in cohort,
                "hidden_states_exact_equal":
                    u_equal,
                "discrete_time_step_exact_equal":
                    t_equal,
                "B_exact_equal":
                    b_equal,
                "discrete_B_exact_equal":
                    d_equal,
                "write_exact_equal":
                    w_equal,
                "q_d_l2":
                    q_d_l2,
                "q_t_l2":
                    q_t_l2,
                "q_b_l2":
                    q_b_l2,
                "delta_write_l2":
                    delta_w_l2,
                "q_t_q_b_interaction":
                    interaction,
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

    k0_w_sq = k0_w * k0_w

    for r in provisional:
        r["q_d_l2_over_k0_w"] = (
            float(r["q_d_l2"]) / k0_w
        )
        r["q_t_l2_over_k0_w"] = (
            float(r["q_t_l2"]) / k0_w
        )
        r["q_b_l2_over_k0_w"] = (
            float(r["q_b_l2"]) / k0_w
        )
        r["interaction_over_k0_w_sq"] = (
            float(r["q_t_q_b_interaction"])
            / k0_w_sq
        )

    return provisional


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
    fields = (
        "q_d_l2",
        "q_t_l2",
        "q_b_l2",
        "delta_write_l2",
        "q_t_q_b_interaction",
        "q_d_l2_over_k0_w",
        "q_t_l2_over_k0_w",
        "q_b_l2_over_k0_w",
        "interaction_over_k0_w_sq",
        "composition_relative_residual",
    )

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

    km1 = [
        r
        for r in rows
        if int(r["relative_coordinate"]) == -1
    ]

    require(
        all(
            r["hidden_states_exact_equal"]
            and r["discrete_time_step_exact_equal"]
            and r["B_exact_equal"]
            and r["discrete_B_exact_equal"]
            and r["write_exact_equal"]
            for r in km1
        ),
        "K_MINUS_1_SUMMARY_IDENTITY_FAILURE",
    )

    return {
        "schema_version":
            "k0-rvg-discrete-b-nested-decomposition-summary-v1",
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
        "decomposition":
            DECOMPOSITION,
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


def validate_parent_q_d(
    summary: Mapping[str, Any],
    parent_summary: Mapping[str, Any],
):
    current = summary[
        "common_330_ddsssss_trajectory"
    ]
    parent = parent_summary[
        "common_330_ddsssss_trajectory"
    ]

    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            observed = float(
                current[role][str(k)]
                ["q_d_l2_over_k0_w"]["median"]
            )

            expected = float(
                parent[role][str(k)]
                ["q_d_l2_over_k0_w"]["median"]
            )

            require(
                math.isclose(
                    observed,
                    expected,
                    rel_tol=1e-13,
                    abs_tol=1e-13,
                ),
                (
                    "PARENT_Q_D_MEDIAN_MISMATCH:"
                    f"{role}:{k}:"
                    f"{observed}:{expected}"
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

    validate_parent_q_d(
        summary,
        parent_summary,
    )

    partial_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metrics_path = (
        partial_dir
        / "discrete_b_nested_metrics.jsonl"
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
            "k0-rvg-discrete-b-nested-decomposition-execution-manifest-v1",
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
        "decomposition":
            DECOMPOSITION,
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
        "source_snapshot_dtype":
            "torch.float32",
        "metric_accumulation_dtype":
            "torch.float64",
        "raw_vectors_persisted":
            False,
        "parent_q_d_summary_match":
            True,
        "scientific_model_forward_executed":
            True,
        "scientific_nested_factor_read":
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
            "discrete_b_nested_metrics.jsonl":
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
        "PASS_DISCRETE_B_NESTED_DECOMPOSITION_EXECUTION"
    )
    print("output_dir =", final_dir)
    print(
        "model_forward_count =",
        forward_count,
    )
    print("parent_q_d_summary_match = True")

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[role][str(k)]

            print(
                f"{role}_k{k}_median "
                f"Q_T={t['q_t_l2_over_k0_w']['median']} "
                f"Q_B={t['q_b_l2_over_k0_w']['median']} "
                f"Q_D={t['q_d_l2_over_k0_w']['median']} "
                f"interaction="
                f"{t['interaction_over_k0_w_sq']['median']}"
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

    write = load_write_runner(root)

    (
        carry,
        magnitude,
        base,
        plan,
        cohort,
        signatures,
    ) = build_plan(
        root,
        write,
    )

    repo, parent_summary = authenticate_parent(
        root,
        base,
    )

    print(
        "=== DISCRETE-B NESTED DECOMPOSITION PLAN ==="
    )
    print("branch =", repo["branch"])
    print("head =", repo["head"])
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
        "execution_protocol =",
        EXECUTION_PROTOCOL,
    )
    print(
        "factorization =",
        FACTORIZATION,
    )
    print(
        "decomposition =",
        DECOMPOSITION,
    )
    print(
        "expected_mamba_source_sha256 =",
        EXPECTED_MAMBA_SOURCE_SHA256,
    )
    print(
        "parent_q_d_preserved = True"
    )
    print(
        "tokenizer_invoked = False"
    )
    print(
        "scientific_model_forward_executed = False"
    )

    if args.static_preflight:
        print(
            "PASS_DISCRETE_B_NESTED_STATIC_PREFLIGHT"
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
    except NestedDiscreteBError as exc:
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