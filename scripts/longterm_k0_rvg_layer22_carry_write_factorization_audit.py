"""K0-RVG layer-22 post-update state carry/write factorization audit.

Authenticated recurrence at layer 22:

    S_post32 = fl(G32 * S_prev32 + W32)

Runtime bridge:
    Carry32 = G32 * S_prev32
    S_recon32 = Carry32 + W32

Scientific additive decomposition is evaluated separately in float64 from the
captured float32 operands:

    Carry64 = Carry32.to(float64)
    W64 = W32.to(float64)
    S_operand64 = Carry64 + W64

For matched/swapped branches:

    delta_S_operand = delta_Carry + delta_W

and:

    ||delta_S_operand||^2
      = ||delta_Carry||^2
      + ||delta_W||^2
      + 2 <delta_Carry, delta_W>

Direct observer S_post remains a runtime bridge and is not conflated with the
float64 operand-sum identity. This audit is observational/algebraic only. It
does not tokenize, read logits, execute task heads, train, or intervene.
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


EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

PARENT_FREEZE_COMMIT = "f1f5c5a8bc67a8c2f34b34cdb6f86073891dbf2e"
PARENT_RUNNER_REL = (
    "scripts/longterm_k0_rvg_layer22_recurrent_readout_factorization_audit.py"
)
PARENT_RUNNER_SHA256 = (
    "b9a307b031f085f454203e7cd3c3bbe06b26552f6389d072d68f9996f6f182b0"
)
PARENT_SUMMARY_REL = (
    "reports/longterm_k0_rvg_layer22_recurrent_readout_factorization_d5243a9_v1/"
    "summary.json"
)
PARENT_SUMMARY_SHA256 = (
    "1654370d647e3a729fb9ec0a8c69254774176d24c4351cbbcc269535fd80d652"
)
PARENT_MANIFEST_REL = (
    "reports/longterm_k0_rvg_layer22_recurrent_readout_factorization_d5243a9_v1/"
    "execution_manifest.json"
)
PARENT_MANIFEST_SHA256 = (
    "2fb569b73b8fa3b95b9e612e168e5dd4dfa367cc8b2d68249693348404cf3912"
)

RUNNER_REL = "scripts/longterm_k0_rvg_layer22_carry_write_factorization_audit.py"
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
FLOAT32_BRANCH_RECON_REL_TOL = 5e-7
OPERAND_VS_DIRECT_STATE_REL_TOL = 5e-6
PARENT_TRAJECTORY_REL_TOL = 1e-13
PARENT_TRAJECTORY_ABS_TOL = 1e-13

QUESTION = (
    "Within the layer-22 post-update state term that dominates direct "
    "recurrent-readout role separation at k2, is the observed state difference "
    "primarily associated with carried prior state, current write, or their "
    "vector interaction?"
)


class CarryWriteAuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise CarryWriteAuditError(message)


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
        raise CarryWriteAuditError(f"GIT_FAILURE:{' '.join(args)}") from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CarryWriteAuditError(f"GIT_SHOW_FAILURE:{spec}") from exc


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
        raise CarryWriteAuditError(f"UNEXPECTED_WORKTREE_CHANGE:{line}")

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
        diff_rc = subprocess.call(
            ["git", "diff", "--quiet", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(diff_rc == 0, "RUNTIME_RUNNER_WORKTREE_DRIFT")

    return {
        "branch": branch,
        "head": head,
        "status": status,
        "runtime_mode": runtime_mode,
    }


def load_parent(root: Path):
    path = root / PARENT_RUNNER_REL
    require(path.is_file(), "PARENT_RUNNER_MISSING")
    current = path.read_bytes()
    require(
        sha256_bytes(current) == PARENT_RUNNER_SHA256,
        "PARENT_RUNNER_SHA256_MISMATCH",
    )
    frozen = git_bytes(root, f"{PARENT_FREEZE_COMMIT}:{PARENT_RUNNER_REL}")
    require(current == frozen, "PARENT_RUNNER_WORKTREE_DRIFT")
    require(
        sha256_bytes(frozen) == PARENT_RUNNER_SHA256,
        "PARENT_RUNNER_FROZEN_SHA256_MISMATCH",
    )
    return import_module(path, "k0_rvg_layer22_recurrent_readout_parent")


def authenticate_frozen_parent(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_frozen = git_bytes(root, f"{PARENT_FREEZE_COMMIT}:{PARENT_SUMMARY_REL}")
    manifest_frozen = git_bytes(root, f"{PARENT_FREEZE_COMMIT}:{PARENT_MANIFEST_REL}")
    require(
        sha256_bytes(summary_frozen) == PARENT_SUMMARY_SHA256,
        "PARENT_SUMMARY_SHA256_MISMATCH",
    )
    require(
        sha256_bytes(manifest_frozen) == PARENT_MANIFEST_SHA256,
        "PARENT_MANIFEST_SHA256_MISMATCH",
    )

    summary_path = root / PARENT_SUMMARY_REL
    manifest_path = root / PARENT_MANIFEST_REL
    require(summary_path.is_file(), "PARENT_SUMMARY_MISSING")
    require(manifest_path.is_file(), "PARENT_MANIFEST_MISSING")
    require(summary_path.read_bytes() == summary_frozen, "PARENT_SUMMARY_WORKTREE_DRIFT")
    require(manifest_path.read_bytes() == manifest_frozen, "PARENT_MANIFEST_WORKTREE_DRIFT")

    summary = json.loads(summary_frozen)
    manifest = json.loads(manifest_frozen)

    require(
        summary.get("schema_version")
        == "k0-rvg-layer22-recurrent-readout-factorization-summary-v1",
        "PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(summary.get("item_count") == EXPECTED_ITEM_COUNT, "PARENT_ITEM_COUNT_MISMATCH")
    require(
        summary.get("pair_role_count") == EXPECTED_PAIR_ROLE_COUNT,
        "PARENT_PAIR_ROLE_COUNT_MISMATCH",
    )
    require(
        summary.get("common_ddsssss_item_count") == EXPECTED_COMMON_COUNT,
        "PARENT_COMMON_COUNT_MISMATCH",
    )
    require(summary.get("source_layer") == SOURCE_LAYER, "PARENT_SOURCE_LAYER_MISMATCH")
    require(summary.get("state_size") == STATE_SIZE, "PARENT_STATE_SIZE_MISMATCH")
    require(
        manifest.get("runtime_git_head")
        == "d5243a907517fe4911b8df026647a7b15880626b",
        "PARENT_EXECUTION_HEAD_MISMATCH",
    )
    return summary, manifest


def build_plan(root: Path, parent: Any):
    skip_parent = parent.load_parent(root)
    values = parent.build_plan(root, skip_parent)
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


def authenticate_observer_static(root: Path, base: Any, expected_source_sha256: str):
    observer_path = root / base.OBSERVER_REL
    observer = import_module(observer_path, "k0_rvg_layer22_carry_write_observer")
    require(hasattr(observer, "RawRecurrenceCollector"), "RAW_RECURRENCE_COLLECTOR_MISSING")
    require(hasattr(observer, "RecurrenceRecord"), "RECURRENCE_RECORD_MISSING")
    binding = observer.resolve_source_binding()
    require(binding.source_sha256 == expected_source_sha256, "OBSERVER_SOURCE_SHA256_MISMATCH")
    require(binding.update_line == 350, "OBSERVER_UPDATE_LINE_MISMATCH")
    require(binding.readout_line == 351, "OBSERVER_READOUT_LINE_MISMATCH")

    trace_src = inspect.getsource(observer.RawRecurrenceCollector._trace)
    for needle, label in (
        ('frame.f_locals.get("ssm_state")', "SSM_STATE"),
        ('frame.f_locals.get("discrete_A")', "DISCRETE_A"),
        ('frame.f_locals.get("deltaB_u")', "DELTAB_U"),
        ("s_prev", "S_PREV"),
        ("g_meta", "G"),
        ("w_meta", "W"),
        ("s_post", "S_POST"),
        ("self.records[key]", "RECORD_WRITE"),
    ):
        require(needle in trace_src, f"OBSERVER_{label}_CAPTURE_CONTRACT_MISSING")
    return observer, binding


def _cpu_float32(value: Any, role: str):
    import torch

    require(torch.is_tensor(value), f"{role}_NOT_TENSOR")
    out = value.detach().cpu().contiguous().clone()
    require(out.dtype == torch.float32, f"{role}_DTYPE_NOT_FLOAT32")
    require(out.device.type == "cpu", f"{role}_NOT_CPU")
    require(bool(torch.isfinite(out).all().item()), f"{role}_NONFINITE")
    require(tuple(out.shape) == (1, WIDTH, STATE_SIZE), f"{role}_SHAPE_MISMATCH:{tuple(out.shape)}")
    return out


def resolve_runtime_bundle(
    root: Path,
    parent: Any,
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
    require(binding.source_sha256 == output_parent.EXPECTED_MAMBA_SOURCE_SHA256, "MAMBA_SOURCE_SHA256_MISMATCH")
    require(bool(mixer22.use_mambapy) is False, "USE_MAMBAPY_NOT_FALSE")
    require(mixer22.training is False, "MIXER22_NOT_EVAL")
    require(bool(mixer22.use_associative_scan) is True, "USE_ASSOCIATIVE_SCAN_NOT_TRUE")
    return bundle


def _prefixes(parent: Any, row: Mapping[str, Any]):
    matched, swapped = parent._prefixes(row)
    return tuple(matched), tuple(swapped)


def capture_carry_write(
    observer: Any,
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    import torch

    targets = tuple(int(v) for v in targets)
    collector = observer.RawRecurrenceCollector(binding, layer_map, targets)
    prior_trace = sys.gettrace()
    with collector.capture():
        base.direct_backbone_forward(model, token_ids)
    require(sys.gettrace() is prior_trace, "TRACE_RESTORATION_FAILURE")
    require(collector.records is not None, "RECURRENCE_RECORDS_MISSING")

    records: dict[int, dict[str, Any]] = {}
    max_branch_rel = 0.0

    for token in targets:
        key = (SOURCE_LAYER, token)
        require(key in collector.records, f"LAYER22_RECURRENCE_RECORD_MISSING:{token}")
        rec = collector.records[key]

        s_prev32 = _cpu_float32(rec.s_prev, "S_PREV")
        g32 = _cpu_float32(rec.g, "G")
        w32 = _cpu_float32(rec.w, "W")
        s_post32 = _cpu_float32(rec.s_post, "S_POST")

        carry32 = (g32 * s_prev32).contiguous()
        s_recon32 = (carry32 + w32).contiguous()

        residual64 = s_recon32.to(torch.float64) - s_post32.to(torch.float64)
        denom = max(
            float(torch.linalg.vector_norm(s_recon32.to(torch.float64)).item()),
            float(torch.linalg.vector_norm(s_post32.to(torch.float64)).item()),
            1e-12,
        )
        branch_rel = float(torch.linalg.vector_norm(residual64).item() / denom)
        require(
            branch_rel <= FLOAT32_BRANCH_RECON_REL_TOL,
            f"BRANCH_RECONSTRUCTION_FAILURE:{token}:{branch_rel}",
        )
        max_branch_rel = max(max_branch_rel, branch_rel)

        records[token] = {
            "S_PREV32": s_prev32,
            "G32": g32,
            "W32": w32,
            "S_POST32": s_post32,
            "CARRY32": carry32,
            "branch_reconstruction_relative_residual": branch_rel,
        }

    require(set(records) == set(targets), "TARGET_RECORD_SET_MISMATCH")
    return records, max_branch_rel


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

        sm32 = m["S_POST32"]
        ss32 = s["S_POST32"]
        wm32 = m["W32"]
        ws32 = s["W32"]
        cm32 = m["CARRY32"]
        cs32 = s["CARRY32"]

        if k == -1:
            require(torch.equal(sm32, ss32), f"K_MINUS_1_S_POST_IDENTITY_FAILURE:{idx}:{role}")
            require(torch.equal(wm32, ws32), f"K_MINUS_1_W_IDENTITY_FAILURE:{idx}:{role}")
            require(torch.equal(cm32, cs32), f"K_MINUS_1_CARRY_IDENTITY_FAILURE:{idx}:{role}")

        sm = sm32.to(torch.float64)
        ss = ss32.to(torch.float64)
        wm = wm32.to(torch.float64)
        ws = ws32.to(torch.float64)
        cm = cm32.to(torch.float64)
        cs = cs32.to(torch.float64)

        om = cm + wm
        os_ = cs + ws

        delta_s_post = sm - ss
        delta_carry = cm - cs
        delta_w = wm - ws
        delta_operand = om - os_
        qsum = delta_carry + delta_w

        delta_s_post_l2 = float(torch.linalg.vector_norm(delta_s_post).item())
        delta_carry_l2 = float(torch.linalg.vector_norm(delta_carry).item())
        delta_w_l2 = float(torch.linalg.vector_norm(delta_w).item())
        delta_operand_l2 = float(torch.linalg.vector_norm(delta_operand).item())

        closure_abs = float(torch.linalg.vector_norm(delta_operand - qsum).item())
        closure_rel = closure_abs / max(delta_operand_l2, 1e-12)
        require(
            closure_rel <= ALGEBRAIC_CLOSURE_REL_TOL,
            f"OPERAND_CLOSURE_FAILURE:{idx}:{role}:{k}:{closure_rel}",
        )

        operand_vs_direct_abs = float(
            torch.linalg.vector_norm(delta_operand - delta_s_post).item()
        )
        operand_vs_direct_rel = operand_vs_direct_abs / max(
            delta_operand_l2, delta_s_post_l2, 1e-12
        )
        require(
            operand_vs_direct_rel <= OPERAND_VS_DIRECT_STATE_REL_TOL,
            f"OPERAND_VS_DIRECT_STATE_FAILURE:{idx}:{role}:{k}:{operand_vs_direct_rel}",
        )

        carry_sq = delta_carry_l2 * delta_carry_l2
        write_sq = delta_w_l2 * delta_w_l2
        rss_sq = carry_sq + write_sq
        rss_l2 = math.sqrt(rss_sq)
        dot = float(torch.sum(delta_carry * delta_w).item())
        normalized_cross = 2.0 * dot / rss_sq if rss_sq > 0.0 else 0.0
        carry_energy_fraction = carry_sq / rss_sq if rss_sq > 0.0 else 0.0
        write_energy_fraction = write_sq / rss_sq if rss_sq > 0.0 else 0.0
        addition_factor = delta_operand_l2 / rss_l2 if rss_l2 > 0.0 else 0.0

        operand_sq = delta_operand_l2 * delta_operand_l2
        carry_additive_share = (carry_sq + dot) / operand_sq if operand_sq > 0.0 else 0.0
        write_additive_share = (write_sq + dot) / operand_sq if operand_sq > 0.0 else 0.0

        if operand_sq > 0.0:
            require(
                math.isclose(
                    carry_additive_share + write_additive_share,
                    1.0,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                f"ADDITIVE_SHARE_CLOSURE_FAILURE:{idx}:{role}:{k}",
            )

        branch_rel = max(
            float(m["branch_reconstruction_relative_residual"]),
            float(s["branch_reconstruction_relative_residual"]),
        )

        if k == -1:
            for value, name in (
                (delta_s_post_l2, "DELTA_S_POST"),
                (delta_operand_l2, "DELTA_S_OPERAND"),
                (delta_carry_l2, "DELTA_CARRY"),
                (delta_w_l2, "DELTA_W"),
                (rss_l2, "RSS"),
            ):
                require(value == 0.0, f"K_MINUS_1_{name}_NONZERO:{idx}:{role}:{value}")

        values = {
            "schema_version": "k0-rvg-layer22-carry-write-factorization-row-v1",
            "local_template_index": idx,
            "stable_item_id": row["stable_item_id"],
            "role": role,
            "relative_coordinate": k,
            "token_index": token,
            "divergence_anchor_token_index": anchor,
            "token_equality_signature_k0_to_k6": signature,
            "in_common_ddsssss_cohort": idx in cohort,
            "delta_s22_post_l2": delta_s_post_l2,
            "delta_s22_operand_l2": delta_operand_l2,
            "delta_carry_l2": delta_carry_l2,
            "delta_w_l2": delta_w_l2,
            "rss_l2": rss_l2,
            "carry_energy_fraction": carry_energy_fraction,
            "write_energy_fraction": write_energy_fraction,
            "normalized_cross": normalized_cross,
            "addition_factor": addition_factor,
            "carry_additive_share": carry_additive_share,
            "write_additive_share": write_additive_share,
            "operand_closure_relative_residual": closure_rel,
            "operand_vs_direct_state_relative_residual": operand_vs_direct_rel,
            "branch_reconstruction_relative_residual": branch_rel,
            "state_observation_mode": "direct_raw_recurrence_observer_s_post",
            "carry_observation_mode": "reconstructed_float32_G_times_S_prev_from_direct_observer_operands",
            "write_observation_mode": "direct_raw_recurrence_observer_w",
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
    "delta_s22_post_l2",
    "delta_s22_operand_l2",
    "delta_carry_l2",
    "delta_w_l2",
    "rss_l2",
    "carry_energy_fraction",
    "write_energy_fraction",
    "normalized_cross",
    "addition_factor",
    "carry_additive_share",
    "write_additive_share",
    "operand_closure_relative_residual",
    "operand_vs_direct_state_relative_residual",
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
                field: aggregate(row[field] for row in bucket)
                for field in SUMMARY_FIELDS
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
    eq = len(pairs) - gt - lt
    return {
        "count": len(pairs),
        "corr_gt_ctrl": gt,
        "corr_lt_ctrl": lt,
        "equal": eq,
    }


def _dominance_count(rows, left: str, right: str) -> dict[str, int]:
    gt = sum(float(r[left]) > float(r[right]) for r in rows)
    lt = sum(float(r[left]) < float(r[right]) for r in rows)
    eq = len(rows) - gt - lt
    return {"count": len(rows), "left_gt_right": gt, "left_lt_right": lt, "equal": eq}


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

    common_k2 = [
        r for r in common_rows if int(r["relative_coordinate"]) == 2
    ]
    common_k2_corr = [r for r in common_k2 if r["role"] == "corr"]
    common_k2_ctrl = [r for r in common_k2 if r["role"] == "ctrl"]
    pairs = _aligned_common_k2(rows)

    role_counts = {
        field: _paired_count(pairs, field)
        for field in ("delta_carry_l2", "delta_w_l2", "delta_s22_post_l2")
    }

    return {
        "schema_version": "k0-rvg-layer22-carry-write-factorization-summary-v1",
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
        "runtime_identity": "S_post32 = fl(G32 * S_prev32 + W32)",
        "operand_identity": "delta_S_operand64 = delta_Carry64 + delta_W64",
        "carry_definition": "Carry32 = G32 * S_prev32; Carry64 = Carry32.to(float64)",
        "write_definition": "W64 = captured_W32.to(float64)",
        "direct_state_bridge": (
            "direct observer s_post is compared numerically to the float32 runtime "
            "reconstruction and separately to the float64 operand-sum difference"
        ),
        "full_336_trajectory": aggregate_trajectory(rows),
        "common_330_ddsssss_trajectory": aggregate_trajectory(common_rows),
        "common_330_k2_role_counts": role_counts,
        "common_330_k2_corr_carry_vs_write": _dominance_count(
            common_k2_corr, "delta_carry_l2", "delta_w_l2"
        ),
        "common_330_k2_ctrl_carry_vs_write": _dominance_count(
            common_k2_ctrl, "delta_carry_l2", "delta_w_l2"
        ),
        "max_operand_closure_relative_residual": max(
            float(r["operand_closure_relative_residual"]) for r in rows
        ),
        "max_operand_vs_direct_state_relative_residual": max(
            float(r["operand_vs_direct_state_relative_residual"]) for r in rows
        ),
        "max_branch_reconstruction_relative_residual": max(
            float(r["branch_reconstruction_relative_residual"]) for r in rows
        ),
        "parent_delta_s22_post_trajectory_match": False,
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
                got_stats = current[role][str(k)]["delta_s22_post_l2"]
                expected_stats = expected[role][str(k)]["delta_s22_post_l2"]
                require(
                    int(got_stats["count"]) == int(expected_stats["count"]),
                    f"PARENT_STATE_COUNT_MISMATCH:{trajectory}:{role}:{k}",
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
                        f"PARENT_STATE_TRAJECTORY_MISMATCH:{trajectory}:{role}:{k}:{stat}:{got}:{exp}",
                    )
    summary["parent_delta_s22_post_trajectory_match"] = True


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


def runtime_preflight(
    root: Path,
    parent: Any,
    output_parent: Any,
    residual_parent: Any,
    rms_parent: Any,
    hidden_parent: Any,
    postconv: Any,
    base: Any,
    plan: Sequence[Mapping[str, Any]],
    cohort: frozenset[int],
    signatures: Mapping[tuple[int, str], str],
    handoff_path: Path,
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
        _layer22,
        _layer23,
        _norm23,
        _mixer23,
        _mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
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
    matched_prefix, swapped_prefix = _prefixes(parent, row)

    matched, m_rel = capture_carry_write(
        observer, base, model, binding, layer_map, matched_prefix, row["targets"]
    )
    swapped, s_rel = capture_carry_write(
        observer, base, model, binding, layer_map, swapped_prefix, row["targets"]
    )
    rows = metric_rows_for_pair(
        row, matched, swapped, cohort, signatures[(idx, role)]
    )

    print("PASS_LAYER22_CARRY_WRITE_RUNTIME_PREFLIGHT")
    print("pair_role =", idx, role)
    print("model_forward_count = 2")
    print("scientific_population_accessed = True")
    print("scientific_evidence_emitted = False")
    print("raw_vectors_persisted = False")
    print("max_branch_reconstruction_relative_residual =", max(m_rel, s_rel))
    print(
        "max_operand_closure_relative_residual =",
        max(float(r["operand_closure_relative_residual"]) for r in rows),
    )
    print(
        "max_operand_vs_direct_state_relative_residual =",
        max(float(r["operand_vs_direct_state_relative_residual"]) for r in rows),
    )


def execute(
    root: Path,
    parent: Any,
    output_parent: Any,
    residual_parent: Any,
    rms_parent: Any,
    hidden_parent: Any,
    postconv: Any,
    base: Any,
    repo: Mapping[str, Any],
    parent_summary: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    cohort: frozenset[int],
    signatures: Mapping[tuple[int, str], str],
    handoff_path: Path,
    output_dir: Path,
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
        _layer22,
        _layer23,
        _norm23,
        _mixer23,
        _mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
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
        matched_prefix, swapped_prefix = _prefixes(parent, row)

        matched, rel_m = capture_carry_write(
            observer, base, model, binding, layer_map, matched_prefix, row["targets"]
        )
        forward_count += 1
        swapped, rel_s = capture_carry_write(
            observer, base, model, binding, layer_map, swapped_prefix, row["targets"]
        )
        forward_count += 1
        max_branch_rel = max(max_branch_rel, rel_m, rel_s)

        rows.extend(
            metric_rows_for_pair(
                row, matched, swapped, cohort, signatures[(idx, role)]
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
        summary["max_operand_closure_relative_residual"] <= ALGEBRAIC_CLOSURE_REL_TOL,
        "SUMMARY_OPERAND_CLOSURE_TOLERANCE_FAILURE",
    )
    require(
        summary["max_operand_vs_direct_state_relative_residual"]
        <= OPERAND_VS_DIRECT_STATE_REL_TOL,
        "SUMMARY_OPERAND_VS_DIRECT_STATE_TOLERANCE_FAILURE",
    )
    require(
        summary["max_branch_reconstruction_relative_residual"]
        <= FLOAT32_BRANCH_RECON_REL_TOL,
        "SUMMARY_BRANCH_RECON_TOLERANCE_FAILURE",
    )

    partial_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = partial_dir / "layer22_carry_write_factorization_metrics.jsonl"
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "execution_manifest.json"

    metrics_path.write_bytes(jsonl_bytes(rows))
    summary_path.write_bytes(json_bytes(summary))

    manifest = {
        "schema_version": "k0-rvg-layer22-carry-write-factorization-execution-manifest-v1",
        "runtime_git_head": repo["head"],
        "runtime_branch": repo["branch"],
        "parent_freeze_commit": PARENT_FREEZE_COMMIT,
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "parent_summary_sha256": PARENT_SUMMARY_SHA256,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "scientific_question": QUESTION,
        "runtime_identity": "S_post32 = fl(G32 * S_prev32 + W32)",
        "operand_identity": "delta_S_operand64 = delta_Carry64 + delta_W64",
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
        "capture_method": (
            "existing RawRecurrenceCollector directly captures s_prev/g/w/s_post "
            "at authenticated layer/token coordinates; no additional nested trace "
            "instrumentation is introduced"
        ),
        "float32_branch_reconstruction_rel_tol": FLOAT32_BRANCH_RECON_REL_TOL,
        "operand_algebraic_closure_rel_tol": ALGEBRAIC_CLOSURE_REL_TOL,
        "operand_vs_direct_state_rel_tol": OPERAND_VS_DIRECT_STATE_REL_TOL,
        "handoff_zip_sha256": handoff["zip_sha256"],
        "checkpoint_sha256": handoff["checkpoint_sha256"],
        "encoder_canonical_digest": encoder["canonical_digest"],
        "encoder_raw_concat_digest": encoder["raw_concat_digest"],
        "parent_delta_s22_post_trajectory_match": True,
        "raw_vectors_persisted": False,
        "scientific_model_forward_executed": True,
        "scientific_s_prev_read": True,
        "scientific_g_read": True,
        "scientific_w_read": True,
        "scientific_s22_post_read": True,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_probe_or_learned_geometry_executed": False,
        "runner_rel": RUNNER_REL,
        "runner_sha256": sha256_bytes((root / RUNNER_REL).read_bytes()),
        "outputs": {
            "layer22_carry_write_factorization_metrics.jsonl": sha256_bytes(
                metrics_path.read_bytes()
            ),
            "summary.json": sha256_bytes(summary_path.read_bytes()),
        },
    }
    manifest_path.write_bytes(json_bytes(manifest))
    os.replace(partial_dir, final_dir)

    common = summary["common_330_ddsssss_trajectory"]
    print("PASS_LAYER22_CARRY_WRITE_FACTORIZATION_EXECUTION")
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print("parent_delta_s22_post_trajectory_match = True")
    print(
        "max_branch_reconstruction_relative_residual =",
        max_branch_rel,
    )
    print(
        "max_operand_closure_relative_residual =",
        summary["max_operand_closure_relative_residual"],
    )
    print(
        "max_operand_vs_direct_state_relative_residual =",
        summary["max_operand_vs_direct_state_relative_residual"],
    )
    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[role][str(k)]
            print(
                f"{role}_k{k}_median "
                f"SPOST={t['delta_s22_post_l2']['median']} "
                f"SOPERAND={t['delta_s22_operand_l2']['median']} "
                f"CARRY={t['delta_carry_l2']['median']} "
                f"W={t['delta_w_l2']['median']} "
                f"CARRY_E={t['carry_energy_fraction']['median']} "
                f"WRITE_E={t['write_energy_fraction']['median']} "
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
    parent_summary, _parent_manifest = authenticate_frozen_parent(root)

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

    source = secant.validate_frozen_source_semantics()
    require(
        source["source_sha256"] == output_parent.EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )
    _observer_static, observer_binding = authenticate_observer_static(
        root, base, output_parent.EXPECTED_MAMBA_SOURCE_SHA256
    )

    print("=== LAYER22 CARRY/WRITE FACTORIZATION AUDIT PLAN ===")
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print("parent_freeze_commit =", PARENT_FREEZE_COMMIT)
    print("parent_runner_sha256 =", PARENT_RUNNER_SHA256)
    print("parent_summary_sha256 =", PARENT_SUMMARY_SHA256)
    print("parent_manifest_sha256 =", PARENT_MANIFEST_SHA256)
    print("pair_role_count =", len(plan))
    print("common_ddsssss_item_count =", len(cohort))
    print("source_layer =", SOURCE_LAYER)
    print("state_size =", STATE_SIZE)
    print("scientific_question =", QUESTION)
    print("runtime_identity = S_post32 = fl(G32 * S_prev32 + W32)")
    print("operand_identity = delta_S_operand64 = delta_Carry64 + delta_W64")
    print("observer_update_line =", observer_binding.update_line)
    print("observer_readout_line =", observer_binding.readout_line)
    print("mamba_source_sha256 =", source["source_sha256"])
    print("nested_readout_instrumentation = False")
    print("raw_vectors_persisted = False")
    print("tokenizer_invoked = False")
    print("training_executed = False")

    if args.static_preflight:
        print("scientific_model_forward_executed = False")
        print("PASS_LAYER22_CARRY_WRITE_FACTORIZATION_STATIC_PREFLIGHT")
        return 0

    require(args.handoff is not None, "RUNTIME_MODE_REQUIRES_HANDOFF")
    handoff_path = args.handoff.resolve()
    require(handoff_path.is_file(), f"HANDOFF_MISSING:{handoff_path}")

    if args.runtime_preflight:
        runtime_preflight(
            root,
            parent,
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
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
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
    except CarryWriteAuditError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"BLOCKED_UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
