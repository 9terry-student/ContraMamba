"""K0-RVG layer-22 write-factor symmetric decomposition audit.

Validated parent result:
    layer-22 post-update state separation at k2 is write-dominant.

Authenticated slow-path write boundary:
    W32 = fl(D32 * U32[..., None])

where:
- D = discrete_B at the authenticated recurrent-update frame;
- U = conv-activated hidden_states at the same layer/token;
- W = deltaB_u, directly captured at the same frame.

The runtime float32 factorization is checked exactly branch-by-branch.  The
scientific symmetric bilinear decomposition is evaluated separately in
float64 from the captured float32 operands:

    W_operand64 = D64 * U64[..., None]
    delta_W_operand = Q_U + Q_D

    Q_U = mean(D_m, D_s) * (U_m - U_s)[..., None]
    Q_D = mean(U_m, U_s)[..., None] * (D_m - D_s)

Direct observed W remains a runtime bridge and is not conflated with the
float64 operand identity.  This audit is observational/algebraic only.  It
does not tokenize, read logits, execute task heads, train, intervene, run PCA,
or fit a learned probe.
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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

PARENT_FREEZE_COMMIT = "21b36fa4579bee53644fa7f3da84cdc947ddef5b"
PARENT_RUNNER_REL = "scripts/longterm_k0_rvg_layer22_carry_write_factorization_audit.py"
PARENT_SUMMARY_REL = (
    "reports/longterm_k0_rvg_layer22_carry_write_factorization_378ac88_v1/summary.json"
)
PARENT_MANIFEST_REL = (
    "reports/longterm_k0_rvg_layer22_carry_write_factorization_378ac88_v1/"
    "execution_manifest.json"
)
PARENT_METRICS_REL = (
    "reports/longterm_k0_rvg_layer22_carry_write_factorization_378ac88_v1/"
    "layer22_carry_write_factorization_metrics.jsonl"
)
PARENT_SUMMARY_SHA256 = "b17b8bc0a6aa679f0f18e0eb3f064d1d9e50b5bea135ecb026836f4fd28461f4"
PARENT_MANIFEST_SHA256 = "ee4ceb39839a26b00dfd46c450a22b5b34c21ca4451da51718bb1da807aeeb33"
PARENT_METRICS_SHA256 = "c9aa47f5cff0b8d4bd7ca81dfcef886e47fef02df9d6ca4e8e2dc2419367f2ad"
PARENT_EXECUTION_HEAD = "378ac88cf3413dc46c9c7fa153ef60785fd370a4"

REFERENCE_WRITE_RUNNER_REL = "scripts/longterm_k0_rvg_write_factor_decomposition.py"
REFERENCE_WRITE_RUNNER_BLOB = "c733dfd4f4150d372d6af17dd2fba9a843d157dc"

RUNNER_REL = "scripts/longterm_k0_rvg_layer22_write_factor_decomposition_audit.py"
K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}

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
OPERAND_VS_DIRECT_W_REL_TOL = 5e-6
PARENT_TRAJECTORY_REL_TOL = 1e-13
PARENT_TRAJECTORY_ABS_TOL = 1e-13

QUESTION = (
    "Within the validated layer-22 write term that carries the k2 post-update "
    "state role separation, is delta-W primarily associated with the current "
    "activation factor U, the selective coefficient D=discrete_B, or their "
    "bilinear vector interaction?"
)


class Layer22WriteFactorError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Layer22WriteFactorError(message)


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
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Layer22WriteFactorError(f"GIT_FAILURE:{' '.join(args)}") from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Layer22WriteFactorError(f"GIT_SHOW_FAILURE:{spec}") from exc


def _status_path(line: str) -> str:
    raw = line[3:] if len(line) >= 4 else ""
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip('"').replace("\\", "/")


def authenticate_repo(root: Path, runtime_mode: bool) -> dict[str, Any]:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, f"GIT_BRANCH_MISMATCH:{branch}")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", PARENT_FREEZE_COMMIT, head],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "PARENT_FREEZE_NOT_ANCESTOR")

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True
    ).splitlines()

    for line in status:
        path = _status_path(line)
        xy = line[:2]
        if path in K1_UNTRACKED:
            require(xy == "??", f"K1_STATE_CHANGED:{line}")
            continue
        if not runtime_mode and path == RUNNER_REL:
            require(xy == "??", f"STATIC_RUNNER_STATE_UNEXPECTED:{line}")
            continue
        raise Layer22WriteFactorError(f"UNEXPECTED_WORKTREE_CHANGE:{line}")

    runner = root / RUNNER_REL
    require(runner.is_file(), "RUNNER_FILE_MISSING")

    if runtime_mode:
        tracked = subprocess.call(
            ["git", "ls-files", "--error-unmatch", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(tracked == 0, "RUNTIME_REQUIRES_TRACKED_RUNNER")
        require(
            subprocess.call(
                ["git", "diff", "--quiet", "--", RUNNER_REL],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0,
            "RUNTIME_RUNNER_WORKTREE_DRIFT",
        )

    return {"branch": branch, "head": head, "status": status, "runtime_mode": runtime_mode}


def load_parent(root: Path):
    current = root / PARENT_RUNNER_REL
    require(current.is_file(), "PARENT_RUNNER_MISSING")
    frozen = git_bytes(root, f"{PARENT_FREEZE_COMMIT}:{PARENT_RUNNER_REL}")
    require(current.read_bytes() == frozen, "PARENT_RUNNER_WORKTREE_DRIFT")
    return import_module(current, "k0_rvg_layer22_carry_write_parent")


def authenticate_frozen_parent(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    summary_raw = git_bytes(root, f"{PARENT_FREEZE_COMMIT}:{PARENT_SUMMARY_REL}")
    manifest_raw = git_bytes(root, f"{PARENT_FREEZE_COMMIT}:{PARENT_MANIFEST_REL}")
    metrics_raw = git_bytes(root, f"{PARENT_FREEZE_COMMIT}:{PARENT_METRICS_REL}")

    require(sha256_bytes(summary_raw) == PARENT_SUMMARY_SHA256, "PARENT_SUMMARY_SHA256_MISMATCH")
    require(sha256_bytes(manifest_raw) == PARENT_MANIFEST_SHA256, "PARENT_MANIFEST_SHA256_MISMATCH")
    require(sha256_bytes(metrics_raw) == PARENT_METRICS_SHA256, "PARENT_METRICS_SHA256_MISMATCH")

    for rel, raw, label in (
        (PARENT_SUMMARY_REL, summary_raw, "SUMMARY"),
        (PARENT_MANIFEST_REL, manifest_raw, "MANIFEST"),
        (PARENT_METRICS_REL, metrics_raw, "METRICS"),
    ):
        path = root / rel
        require(path.is_file(), f"PARENT_{label}_MISSING")
        require(path.read_bytes() == raw, f"PARENT_{label}_WORKTREE_DRIFT")

    summary = json.loads(summary_raw)
    manifest = json.loads(manifest_raw)

    require(
        summary.get("schema_version") == "k0-rvg-layer22-carry-write-factorization-summary-v1",
        "PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("schema_version")
        == "k0-rvg-layer22-carry-write-factorization-execution-manifest-v1",
        "PARENT_MANIFEST_SCHEMA_MISMATCH",
    )
    require(summary.get("item_count") == EXPECTED_ITEM_COUNT, "PARENT_ITEM_COUNT_MISMATCH")
    require(summary.get("pair_role_count") == EXPECTED_PAIR_ROLE_COUNT, "PARENT_PAIR_ROLE_COUNT_MISMATCH")
    require(summary.get("common_ddsssss_item_count") == EXPECTED_COMMON_COUNT, "PARENT_COMMON_COUNT_MISMATCH")
    require(summary.get("source_layer") == SOURCE_LAYER, "PARENT_SOURCE_LAYER_MISMATCH")
    require(summary.get("state_size") == STATE_SIZE, "PARENT_STATE_SIZE_MISMATCH")
    require(summary.get("parent_delta_s22_post_trajectory_match") is True, "PARENT_STATE_REPRODUCTION_NOT_TRUE")
    require(manifest.get("runtime_git_head") == PARENT_EXECUTION_HEAD, "PARENT_EXECUTION_HEAD_MISMATCH")
    require(manifest.get("model_forward_count") == EXPECTED_FORWARD_COUNT, "PARENT_FORWARD_COUNT_MISMATCH")
    require(manifest.get("outputs", {}).get("summary.json") == PARENT_SUMMARY_SHA256, "PARENT_MANIFEST_SUMMARY_HASH_MISMATCH")
    require(
        manifest.get("outputs", {}).get("layer22_carry_write_factorization_metrics.jsonl")
        == PARENT_METRICS_SHA256,
        "PARENT_MANIFEST_METRICS_HASH_MISMATCH",
    )

    parent_runner_sha = sha256_bytes(git_bytes(root, f"{PARENT_FREEZE_COMMIT}:{PARENT_RUNNER_REL}"))
    return summary, manifest, parent_runner_sha


def authenticate_reference_write_factor(root: Path):
    blob = git(root, "rev-parse", f"{PARENT_FREEZE_COMMIT}:{REFERENCE_WRITE_RUNNER_REL}")
    require(blob == REFERENCE_WRITE_RUNNER_BLOB, "REFERENCE_WRITE_RUNNER_BLOB_MISMATCH")
    current_blob = git(root, "rev-parse", f"HEAD:{REFERENCE_WRITE_RUNNER_REL}")
    require(current_blob == REFERENCE_WRITE_RUNNER_BLOB, "REFERENCE_WRITE_RUNNER_HEAD_BLOB_MISMATCH")
    path = root / REFERENCE_WRITE_RUNNER_REL
    require(path.is_file(), "REFERENCE_WRITE_RUNNER_MISSING")
    require(
        subprocess.call(
            ["git", "diff", "--quiet", "--", REFERENCE_WRITE_RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0,
        "REFERENCE_WRITE_RUNNER_WORKTREE_DRIFT",
    )
    mod = import_module(path, "k0_rvg_layer23_write_factor_reference")
    require(hasattr(mod, "WriteFactorCollector"), "REFERENCE_COLLECTOR_MISSING")
    trace_src = inspect.getsource(mod.WriteFactorCollector._trace)
    for needle, label in (
        ('frame.f_locals.get("discrete_B")', "DISCRETE_B"),
        ('frame.f_locals.get("hidden_states")', "HIDDEN_STATES"),
        ('frame.f_locals.get("deltaB_u")', "DELTAB_U"),
        ('discrete_B[:, :, i, :]', "D_SLICE"),
        ('hidden_states[:, :, i]', "U_SLICE"),
        ('deltaB_u[:, :, i, :]', "W_SLICE"),
        ('reconstructed = D * U[:, :, None]', "PRODUCT_RECON"),
    ):
        require(needle in trace_src, f"REFERENCE_{label}_CONTRACT_MISSING")
    return mod


def build_plan(root: Path, parent: Any):
    recurrent_parent = parent.load_parent(root)
    values = parent.build_plan(root, recurrent_parent)
    require(len(values) == 19, "PARENT_BUILD_PLAN_ARITY_MISMATCH")
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
    ) = values
    require(len(plan) == EXPECTED_PAIR_ROLE_COUNT, "PLAN_COUNT_MISMATCH")
    require(len(cohort) == EXPECTED_COMMON_COUNT, "COMMON_COHORT_COUNT_MISMATCH")
    return (
        recurrent_parent,
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


def snapshot(value: Any, role: str, shape: tuple[int, ...]):
    import torch

    require(isinstance(value, torch.Tensor), f"{role}_NOT_TENSOR")
    out = value.detach().cpu().contiguous().clone()
    require(out.device.type == "cpu", f"{role}_NOT_CPU")
    require(out.dtype == torch.float32, f"{role}_DTYPE_MISMATCH")
    require(tuple(out.shape) == shape, f"{role}_SHAPE_MISMATCH:{tuple(out.shape)}")
    require(bool(torch.isfinite(out).all().item()), f"{role}_NONFINITE")
    return out


class Layer22WriteFactorCollector:
    """Single-use layer-22 observer for D, U, W at the recurrence update."""

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
        import torch

        if (
            frame.f_code is not self.binding.code
            or event != "line"
            or frame.f_lineno != self.binding.update_line
        ):
            return self._trace

        mixer = frame.f_locals.get("self")
        layer = self.layer_map.get(id(mixer))
        if layer != SOURCE_LAYER:
            return self._trace

        i = frame.f_locals.get("i")
        require(type(i) is int and i >= 0, "AMBIGUOUS_TOKEN_INDEX")
        if i not in self.targets:
            return self._trace

        require(self.records is not None, "TRACE_NOT_ACTIVE")
        require(i not in self.records, "DUPLICATE_TOKEN_CAPTURE")

        discrete_B = frame.f_locals.get("discrete_B")
        hidden_states = frame.f_locals.get("hidden_states")
        deltaB_u = frame.f_locals.get("deltaB_u")
        require(discrete_B is not None, "DISCRETE_B_MISSING")
        require(hidden_states is not None, "HIDDEN_STATES_MISSING")
        require(deltaB_u is not None, "DELTAB_U_MISSING")

        D = snapshot(discrete_B[:, :, i, :], "DISCRETE_B", (1, WIDTH, STATE_SIZE))
        U = snapshot(hidden_states[:, :, i], "HIDDEN_STATES", (1, WIDTH))
        W = snapshot(deltaB_u[:, :, i, :], "DELTAB_U", (1, WIDTH, STATE_SIZE))
        reconstructed = (D * U[:, :, None]).contiguous()
        require(torch.equal(reconstructed, W), "WRITE_EXACT_FLOAT32_FACTORIZATION_FAILURE")

        self.records[i] = {
            "D32": D,
            "U32": U,
            "W32": W,
            "branch_reconstruction_relative_residual": 0.0,
        }
        return self._trace

    @contextmanager
    def capture(self):
        require(not self.used, "TRACE_COLLECTOR_REUSE")
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
            require(set(self.records) == set(self.targets), "CAPTURE_COORDINATE_SET_MISMATCH")


def capture_factors(
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    collector = Layer22WriteFactorCollector(binding, layer_map, targets)
    prior_trace = sys.gettrace()
    with collector.capture():
        base.direct_backbone_forward(model, token_ids)
    require(sys.gettrace() is prior_trace, "TRACE_RESTORATION_FAILURE")
    require(collector.records is not None, "CAPTURE_RECORDS_MISSING")
    return collector.records


def _prefixes(parent: Any, recurrent_parent: Any, row: Mapping[str, Any]):
    matched, swapped = parent._prefixes(recurrent_parent, row)
    return tuple(matched), tuple(swapped)


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
        m = matched[token]
        s = swapped[token]

        Dm32, Um32, Wm32 = m["D32"], m["U32"], m["W32"]
        Ds32, Us32, Ws32 = s["D32"], s["U32"], s["W32"]

        d_equal = bool(torch.equal(Dm32, Ds32))
        u_equal = bool(torch.equal(Um32, Us32))
        w_equal = bool(torch.equal(Wm32, Ws32))
        if k == -1:
            require(d_equal and u_equal and w_equal, f"K_MINUS_1_IDENTITY_FAILURE:{idx}:{role}")

        Dm = Dm32.to(torch.float64)
        Ds = Ds32.to(torch.float64)
        Um = Um32.to(torch.float64)
        Us = Us32.to(torch.float64)
        Wm_direct = Wm32.to(torch.float64)
        Ws_direct = Ws32.to(torch.float64)

        Wm_operand = Dm * Um[:, :, None]
        Ws_operand = Ds * Us[:, :, None]
        delta_w_direct = Wm_direct - Ws_direct
        delta_w_operand = Wm_operand - Ws_operand

        delta_u = Um - Us
        delta_d = Dm - Ds
        mean_d = 0.5 * (Dm + Ds)
        mean_u = 0.5 * (Um + Us)
        q_u = mean_d * delta_u[:, :, None]
        q_d = mean_u[:, :, None] * delta_d
        qsum = q_u + q_d

        delta_w_l2 = float(torch.linalg.vector_norm(delta_w_direct).item())
        delta_w_operand_l2 = float(torch.linalg.vector_norm(delta_w_operand).item())
        q_u_l2 = float(torch.linalg.vector_norm(q_u).item())
        q_d_l2 = float(torch.linalg.vector_norm(q_d).item())

        closure_abs = float(torch.linalg.vector_norm(delta_w_operand - qsum).item())
        closure_rel = closure_abs / max(delta_w_operand_l2, 1e-12)
        require(
            closure_rel <= ALGEBRAIC_CLOSURE_REL_TOL,
            f"OPERAND_CLOSURE_FAILURE:{idx}:{role}:{k}:{closure_rel}",
        )

        bridge_abs = float(torch.linalg.vector_norm(delta_w_operand - delta_w_direct).item())
        bridge_rel = bridge_abs / max(delta_w_operand_l2, delta_w_l2, 1e-12)
        require(
            bridge_rel <= OPERAND_VS_DIRECT_W_REL_TOL,
            f"OPERAND_VS_DIRECT_W_FAILURE:{idx}:{role}:{k}:{bridge_rel}",
        )

        q_u_sq = q_u_l2 * q_u_l2
        q_d_sq = q_d_l2 * q_d_l2
        rss_sq = q_u_sq + q_d_sq
        rss_l2 = math.sqrt(rss_sq)
        dot = float(torch.sum(q_u * q_d).item())
        normalized_cross = 2.0 * dot / rss_sq if rss_sq > 0.0 else 0.0
        q_u_energy_fraction = q_u_sq / rss_sq if rss_sq > 0.0 else 0.0
        q_d_energy_fraction = q_d_sq / rss_sq if rss_sq > 0.0 else 0.0
        addition_factor = delta_w_operand_l2 / rss_l2 if rss_l2 > 0.0 else 0.0

        operand_sq = delta_w_operand_l2 * delta_w_operand_l2
        q_u_additive_share = (q_u_sq + dot) / operand_sq if operand_sq > 0.0 else 0.0
        q_d_additive_share = (q_d_sq + dot) / operand_sq if operand_sq > 0.0 else 0.0
        if operand_sq > 0.0:
            require(
                math.isclose(
                    q_u_additive_share + q_d_additive_share,
                    1.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                f"ADDITIVE_SHARE_CLOSURE_FAILURE:{idx}:{role}:{k}",
            )

        if k == -1:
            for value, label in (
                (delta_w_l2, "DELTA_W"),
                (delta_w_operand_l2, "DELTA_W_OPERAND"),
                (q_u_l2, "Q_U"),
                (q_d_l2, "Q_D"),
                (rss_l2, "RSS"),
            ):
                require(value == 0.0, f"K_MINUS_1_{label}_NONZERO:{idx}:{role}:{value}")

        values = {
            "schema_version": "k0-rvg-layer22-write-factor-decomposition-row-v1",
            "local_template_index": idx,
            "stable_item_id": row["stable_item_id"],
            "role": role,
            "relative_coordinate": k,
            "token_index": token,
            "divergence_anchor_token_index": anchor,
            "token_equality_signature_k0_to_k6": signature,
            "in_common_ddsssss_cohort": idx in cohort,
            "discrete_b_exact_equal": d_equal,
            "hidden_states_exact_equal": u_equal,
            "write_exact_equal": w_equal,
            "delta_w_l2": delta_w_l2,
            "delta_w_operand_l2": delta_w_operand_l2,
            "q_u_l2": q_u_l2,
            "q_d_l2": q_d_l2,
            "rss_l2": rss_l2,
            "q_u_energy_fraction": q_u_energy_fraction,
            "q_d_energy_fraction": q_d_energy_fraction,
            "normalized_cross": normalized_cross,
            "addition_factor": addition_factor,
            "q_u_additive_share": q_u_additive_share,
            "q_d_additive_share": q_d_additive_share,
            "operand_closure_relative_residual": closure_rel,
            "operand_vs_direct_w_relative_residual": bridge_rel,
            "branch_reconstruction_relative_residual": max(
                float(m["branch_reconstruction_relative_residual"]),
                float(s["branch_reconstruction_relative_residual"]),
            ),
            "direct_write_observation_mode": "deltaB_u_captured_at_authenticated_update_frame",
            "factor_observation_mode": "discrete_B_and_hidden_states_captured_at_same_update_frame",
            "runtime_operand_dtype": "torch.float32",
            "operand_algebra_dtype": "torch.float64",
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
    "delta_w_l2",
    "delta_w_operand_l2",
    "q_u_l2",
    "q_d_l2",
    "rss_l2",
    "q_u_energy_fraction",
    "q_d_energy_fraction",
    "normalized_cross",
    "addition_factor",
    "q_u_additive_share",
    "q_d_additive_share",
    "operand_closure_relative_residual",
    "operand_vs_direct_w_relative_residual",
    "branch_reconstruction_relative_residual",
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
                field: aggregate(row[field] for row in bucket) for field in SUMMARY_FIELDS
            }
    return result


def _aligned_common_k2(rows):
    corr = {
        int(r["local_template_index"]): r
        for r in rows
        if r["role"] == "corr"
        and int(r["relative_coordinate"]) == 2
        and bool(r["in_common_ddsssss_cohort"])
    }
    ctrl = {
        int(r["local_template_index"]): r
        for r in rows
        if r["role"] == "ctrl"
        and int(r["relative_coordinate"]) == 2
        and bool(r["in_common_ddsssss_cohort"])
    }
    require(set(corr) == set(ctrl), "COMMON_K2_ROLE_ALIGNMENT_MISMATCH")
    require(len(corr) == EXPECTED_COMMON_COUNT, "COMMON_K2_ALIGNMENT_COUNT_MISMATCH")
    return [(corr[i], ctrl[i]) for i in sorted(corr)]


def _paired_count(pairs, field: str) -> dict[str, int]:
    gt = sum(float(c[field]) > float(t[field]) for c, t in pairs)
    lt = sum(float(c[field]) < float(t[field]) for c, t in pairs)
    return {"count": len(pairs), "corr_gt_ctrl": gt, "corr_lt_ctrl": lt, "equal": len(pairs) - gt - lt}


def _dominance_count(rows, left: str, right: str) -> dict[str, int]:
    gt = sum(float(r[left]) > float(r[right]) for r in rows)
    lt = sum(float(r[left]) < float(r[right]) for r in rows)
    return {"count": len(rows), "left_gt_right": gt, "left_lt_right": lt, "equal": len(rows) - gt - lt}


def make_summary(rows, cohort):
    require(
        len(rows) == EXPECTED_PAIR_ROLE_COUNT * len(RELATIVE_COORDINATES),
        "ROW_COUNT_MISMATCH",
    )
    common_rows = [r for r in rows if bool(r["in_common_ddsssss_cohort"])]
    require(
        len(common_rows) == EXPECTED_COMMON_COUNT * 2 * len(RELATIVE_COORDINATES),
        "COMMON_ROW_COUNT_MISMATCH",
    )
    km1 = [r for r in rows if int(r["relative_coordinate"]) == -1]
    require(len(km1) == EXPECTED_PAIR_ROLE_COUNT, "K_MINUS_1_ROW_COUNT_MISMATCH")
    require(
        all(
            r["discrete_b_exact_equal"]
            and r["hidden_states_exact_equal"]
            and r["write_exact_equal"]
            and float(r["delta_w_l2"]) == 0.0
            and float(r["q_u_l2"]) == 0.0
            and float(r["q_d_l2"]) == 0.0
            for r in km1
        ),
        "K_MINUS_1_SUMMARY_IDENTITY_FAILURE",
    )

    common_k2 = [r for r in common_rows if int(r["relative_coordinate"]) == 2]
    common_k2_corr = [r for r in common_k2 if r["role"] == "corr"]
    common_k2_ctrl = [r for r in common_k2 if r["role"] == "ctrl"]
    pairs = _aligned_common_k2(rows)

    return {
        "schema_version": "k0-rvg-layer22-write-factor-decomposition-summary-v1",
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
        "runtime_identity": "W32 = fl(D32 * U32[..., None])",
        "operand_identity": "delta_W_operand64 = Q_U + Q_D",
        "q_u_definition": "Q_U = mean(D_m,D_s) * delta_U[...,None]",
        "q_d_definition": "Q_D = mean(U_m,U_s)[...,None] * delta_D",
        "direct_write_bridge": (
            "direct deltaB_u W is compared numerically to the float64 D*U operand "
            "difference; only the float64 operand identity is treated as exact algebra"
        ),
        "full_336_trajectory": aggregate_trajectory(rows),
        "common_330_ddsssss_trajectory": aggregate_trajectory(common_rows),
        "common_330_k2_role_counts": {
            field: _paired_count(pairs, field)
            for field in ("delta_w_l2", "q_u_l2", "q_d_l2")
        },
        "common_330_k2_corr_q_u_vs_q_d": _dominance_count(common_k2_corr, "q_u_l2", "q_d_l2"),
        "common_330_k2_ctrl_q_u_vs_q_d": _dominance_count(common_k2_ctrl, "q_u_l2", "q_d_l2"),
        "max_operand_closure_relative_residual": max(
            float(r["operand_closure_relative_residual"]) for r in rows
        ),
        "max_operand_vs_direct_w_relative_residual": max(
            float(r["operand_vs_direct_w_relative_residual"]) for r in rows
        ),
        "max_branch_reconstruction_relative_residual": max(
            float(r["branch_reconstruction_relative_residual"]) for r in rows
        ),
        "parent_delta_w_trajectory_match": False,
        "raw_vectors_persisted": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_probe_or_learned_geometry_executed": False,
    }


def validate_parent_reproduction(summary: dict[str, Any], parent_summary: Mapping[str, Any]) -> None:
    for trajectory in ("full_336_trajectory", "common_330_ddsssss_trajectory"):
        current = summary[trajectory]
        expected = parent_summary[trajectory]
        for role in ("corr", "ctrl"):
            for k in RELATIVE_COORDINATES:
                got_stats = current[role][str(k)]["delta_w_l2"]
                expected_stats = expected[role][str(k)]["delta_w_l2"]
                require(
                    int(got_stats["count"]) == int(expected_stats["count"]),
                    f"PARENT_W_COUNT_MISMATCH:{trajectory}:{role}:{k}",
                )
                for stat in ("mean", "median", "min", "max"):
                    got = float(got_stats[stat])
                    exp = float(expected_stats[stat])
                    require(
                        math.isclose(
                            got,
                            exp,
                            rel_tol=PARENT_TRAJECTORY_REL_TOL,
                            abs_tol=PARENT_TRAJECTORY_ABS_TOL,
                        ),
                        f"PARENT_W_TRAJECTORY_MISMATCH:{trajectory}:{role}:{k}:{stat}:{got}:{exp}",
                    )
    summary["parent_delta_w_trajectory_match"] = True


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
    return b"".join(json_bytes(r) for r in rows)


def resolve_runtime_bundle(
    root: Path,
    parent: Any,
    recurrent_parent: Any,
    output_parent: Any,
    residual_parent: Any,
    rms_parent: Any,
    hidden_parent: Any,
    postconv: Any,
    base: Any,
    handoff_path: Path,
):
    bundle = parent.resolve_runtime_bundle(
        root,
        recurrent_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        handoff_path,
    )
    require(len(bundle) == 14, "PARENT_RUNTIME_BUNDLE_ARITY_MISMATCH")
    (
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
    ) = bundle
    require(len(layer_map) == EXPECTED_LAYER_COUNT, "LAYER_MAP_COUNT_MISMATCH")
    require(SOURCE_LAYER in set(layer_map.values()), "SOURCE_LAYER_NOT_REGISTERED")
    require(bool(mixer22.use_mambapy) is False, "USE_MAMBAPY_NOT_FALSE")
    require(mixer22.training is False, "MIXER22_NOT_EVAL")
    return bundle


def runtime_preflight(
    root,
    parent,
    recurrent_parent,
    output_parent,
    residual_parent,
    rms_parent,
    hidden_parent,
    postconv,
    base,
    plan,
    cohort,
    signatures,
    handoff_path,
):
    (
        _observer,
        _k2s,
        model,
        binding,
        layer_map,
        _handoff,
        _encoder,
        _snapshot_info,
        _layer22,
        _layer23,
        _norm23,
        _mixer23,
        _mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
        recurrent_parent,
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
    matched_prefix, swapped_prefix = _prefixes(parent, recurrent_parent, row)
    matched = capture_factors(base, model, binding, layer_map, matched_prefix, row["targets"])
    swapped = capture_factors(base, model, binding, layer_map, swapped_prefix, row["targets"])
    rows = metric_rows_for_pair(row, matched, swapped, cohort, signatures[(idx, role)])

    print("PASS_LAYER22_WRITE_FACTOR_RUNTIME_PREFLIGHT")
    print("pair_role =", idx, role)
    print("model_forward_count = 2")
    print("scientific_population_accessed = True")
    print("scientific_evidence_emitted = False")
    print("raw_vectors_persisted = False")
    print(
        "max_branch_reconstruction_relative_residual =",
        max(float(r["branch_reconstruction_relative_residual"]) for r in rows),
    )
    print(
        "max_operand_closure_relative_residual =",
        max(float(r["operand_closure_relative_residual"]) for r in rows),
    )
    print(
        "max_operand_vs_direct_w_relative_residual =",
        max(float(r["operand_vs_direct_w_relative_residual"]) for r in rows),
    )


def execute(
    root,
    parent,
    recurrent_parent,
    output_parent,
    residual_parent,
    rms_parent,
    hidden_parent,
    postconv,
    base,
    repo,
    parent_summary,
    parent_runner_sha,
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
        _observer,
        _k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        _snapshot_info,
        _layer22,
        _layer23,
        _norm23,
        _mixer23,
        _mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
        recurrent_parent,
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
    for n, row in enumerate(plan, start=1):
        idx = int(row["local_template_index"])
        role = str(row["role"])
        matched_prefix, swapped_prefix = _prefixes(parent, recurrent_parent, row)
        matched = capture_factors(base, model, binding, layer_map, matched_prefix, row["targets"])
        forward_count += 1
        swapped = capture_factors(base, model, binding, layer_map, swapped_prefix, row["targets"])
        forward_count += 1
        rows.extend(
            metric_rows_for_pair(row, matched, swapped, cohort, signatures[(idx, role)])
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
        summary["max_operand_closure_relative_residual"] <= ALGEBRAIC_CLOSURE_REL_TOL,
        "SUMMARY_OPERAND_CLOSURE_TOLERANCE_FAILURE",
    )
    require(
        summary["max_operand_vs_direct_w_relative_residual"] <= OPERAND_VS_DIRECT_W_REL_TOL,
        "SUMMARY_OPERAND_VS_DIRECT_W_TOLERANCE_FAILURE",
    )
    require(
        summary["max_branch_reconstruction_relative_residual"] == 0.0,
        "SUMMARY_BRANCH_RECONSTRUCTION_NOT_EXACT",
    )

    partial_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = partial_dir / "layer22_write_factor_decomposition_metrics.jsonl"
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "execution_manifest.json"
    metrics_path.write_bytes(jsonl_bytes(rows))
    summary_path.write_bytes(json_bytes(summary))

    manifest = {
        "schema_version": "k0-rvg-layer22-write-factor-decomposition-execution-manifest-v1",
        "runtime_git_head": repo["head"],
        "runtime_branch": repo["branch"],
        "parent_freeze_commit": PARENT_FREEZE_COMMIT,
        "parent_runner_sha256": parent_runner_sha,
        "parent_summary_sha256": PARENT_SUMMARY_SHA256,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_metrics_sha256": PARENT_METRICS_SHA256,
        "reference_write_runner_blob": REFERENCE_WRITE_RUNNER_BLOB,
        "scientific_question": QUESTION,
        "runtime_identity": "W32 = fl(D32 * U32[..., None])",
        "operand_identity": "delta_W_operand64 = Q_U + Q_D",
        "execution_protocol": EXECUTION_PROTOCOL,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count": EXPECTED_COMMON_COUNT,
        "model_forward_count": forward_count,
        "source_layer": SOURCE_LAYER,
        "width": WIDTH,
        "state_size": STATE_SIZE,
        "mamba_source_sha256": binding.source_sha256,
        "float32_branch_reconstruction_exact": True,
        "operand_algebraic_closure_rel_tol": ALGEBRAIC_CLOSURE_REL_TOL,
        "operand_vs_direct_w_rel_tol": OPERAND_VS_DIRECT_W_REL_TOL,
        "handoff_zip_sha256": handoff["zip_sha256"],
        "checkpoint_sha256": handoff["checkpoint_sha256"],
        "encoder_canonical_digest": encoder["canonical_digest"],
        "encoder_raw_concat_digest": encoder["raw_concat_digest"],
        "parent_delta_w_trajectory_match": True,
        "capture_method": (
            "single trace collector captures discrete_B, hidden_states, and deltaB_u "
            "at the authenticated layer-22 recurrent-update frame; W32 == D32*U32 is "
            "checked with torch.equal before any scientific metric is computed"
        ),
        "raw_vectors_persisted": False,
        "scientific_model_forward_executed": True,
        "scientific_discrete_b_read": True,
        "scientific_hidden_states_read": True,
        "scientific_w_read": True,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_probe_or_learned_geometry_executed": False,
        "runner_rel": RUNNER_REL,
        "runner_sha256": sha256_bytes((root / RUNNER_REL).read_bytes()),
        "outputs": {
            "layer22_write_factor_decomposition_metrics.jsonl": sha256_bytes(metrics_path.read_bytes()),
            "summary.json": sha256_bytes(summary_path.read_bytes()),
        },
    }
    manifest_path.write_bytes(json_bytes(manifest))
    os.replace(partial_dir, final_dir)

    common = summary["common_330_ddsssss_trajectory"]
    print("PASS_LAYER22_WRITE_FACTOR_DECOMPOSITION_EXECUTION")
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print("parent_delta_w_trajectory_match = True")
    print(
        "max_branch_reconstruction_relative_residual =",
        summary["max_branch_reconstruction_relative_residual"],
    )
    print(
        "max_operand_closure_relative_residual =",
        summary["max_operand_closure_relative_residual"],
    )
    print(
        "max_operand_vs_direct_w_relative_residual =",
        summary["max_operand_vs_direct_w_relative_residual"],
    )
    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[role][str(k)]
            print(
                f"{role}_k{k}_median "
                f"W={t['delta_w_l2']['median']} "
                f"WOPERAND={t['delta_w_operand_l2']['median']} "
                f"Q_U={t['q_u_l2']['median']} "
                f"Q_D={t['q_d_l2']['median']} "
                f"Q_U_E={t['q_u_energy_fraction']['median']} "
                f"Q_D_E={t['q_d_energy_fraction']['median']} "
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
    runtime_mode = bool(args.runtime_preflight or args.execute)
    repo = authenticate_repo(root, runtime_mode=runtime_mode)
    parent = load_parent(root)
    parent_summary, _parent_manifest, parent_runner_sha = authenticate_frozen_parent(root)
    _reference = authenticate_reference_write_factor(root)

    (
        recurrent_parent,
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

    source = secant.validate_frozen_source_semantics()
    require(
        source["source_sha256"] == output_parent.EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )
    _observer, observer_binding = parent.authenticate_observer_static(
        root, base, output_parent.EXPECTED_MAMBA_SOURCE_SHA256
    )

    print("=== LAYER22 WRITE FACTOR DECOMPOSITION AUDIT PLAN ===")
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print("parent_freeze_commit =", PARENT_FREEZE_COMMIT)
    print("parent_runner_sha256 =", parent_runner_sha)
    print("parent_summary_sha256 =", PARENT_SUMMARY_SHA256)
    print("parent_manifest_sha256 =", PARENT_MANIFEST_SHA256)
    print("parent_metrics_sha256 =", PARENT_METRICS_SHA256)
    print("reference_write_runner_blob =", REFERENCE_WRITE_RUNNER_BLOB)
    print("pair_role_count =", len(plan))
    print("common_ddsssss_item_count =", len(cohort))
    print("source_layer =", SOURCE_LAYER)
    print("state_size =", STATE_SIZE)
    print("scientific_question =", QUESTION)
    print("runtime_identity = W32 = fl(D32 * U32[..., None])")
    print("operand_identity = delta_W_operand64 = Q_U + Q_D")
    print("observer_update_line =", observer_binding.update_line)
    print("observer_readout_line =", observer_binding.readout_line)
    print("mamba_source_sha256 =", source["source_sha256"])
    print("raw_vectors_persisted = False")
    print("tokenizer_invoked = False")
    print("training_executed = False")

    if args.static_preflight:
        print("scientific_model_forward_executed = False")
        print("PASS_LAYER22_WRITE_FACTOR_DECOMPOSITION_STATIC_PREFLIGHT")
        return 0

    require(args.handoff is not None, "RUNTIME_MODE_REQUIRES_HANDOFF")
    handoff_path = args.handoff.resolve()
    require(handoff_path.is_file(), f"HANDOFF_MISSING:{handoff_path}")

    if args.runtime_preflight:
        runtime_preflight(
            root,
            parent,
            recurrent_parent,
            output_parent,
            residual_parent,
            rms_parent,
            hidden_parent,
            postconv,
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
        recurrent_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        repo,
        parent_summary,
        parent_runner_sha,
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
    except Layer22WriteFactorError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"BLOCKED_UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
