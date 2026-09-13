"""K0-RVG layer-22 lag-0 channel-conditioned transfer localization audit.

Frozen scientific question:
    Why does the fixed layer-22 lag-0 depthwise-convolution tap transfer the
    corr current-token hidden-difference direction more strongly than ctrl?

Exact norm-transfer identities:
    q_j = k_j * h_j
    t^2 = sum_j p_j k_j^2
    p_j = h_j^2 / sum_r h_r^2

With mu_k2 = mean_j(k_j^2):
    N^2 = t^2 / mu_k2
    a_j = k_j^2 / mu_k2 - 1
    N^2 - 1 = sum_j p_j a_j

For aligned corr/ctrl roles:
    delta_p_j = p_corr,j - p_ctrl,j
    c_j = a_j * delta_p_j
    N_corr^2 - N_ctrl^2 = sum_j c_j

The stage is observational/algebraic only. All 1536 channels are
preregistered. No learned geometry, intervention, tokenizer, logits, task
heads, training, or post-hoc channel/layer/lag/window search is performed.
Raw per-item channel vectors are never persisted.
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


EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

AUTHORITY_FREEZE_COMMIT = "c1d23d469acf37558c794d4b17c95fdb354c2c7f"
AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_static_design_candidate.md"
)
AUTHORITY_SHA256 = "f23d7604d57dd3c4c744c53a1ddb0f8b43623ad51bb0b13cd4875dc3c187abfd"
AUTHORITY_BLOB = "186ff5687e996ab8a51b5cf49786ac85ce8118b3"

PARENT_EVIDENCE_FREEZE_COMMIT = "6b0d282c22d0b1c8f8387d0853205c3d4226a392"
PARENT_IMPLEMENTATION_COMMIT = "3dd3791cab718421fd30ef119d44d3d2a4defde9"
PARENT_RUNTIME_HEAD = "a4c0a48411788cb5bb3ee32fa92e97f91d87de2d"
PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer22_four_tap_convolution_decomposition_audit.py"
)
PARENT_RUNNER_SHA256 = "bc48b1bcb222dbf75828ad61fb61c0d766a5024e2456d19c8ccee6ccdfcbe088"
PARENT_RUNNER_BLOB = "02d193ff9fa4ee8b30aae465fb451bcb96f87a59"

PARENT_RUN_DIR = (
    "reports/"
    "longterm_k0_rvg_layer22_four_tap_convolution_decomposition_3dd3791_v1"
)
PARENT_METRICS_REL = (
    PARENT_RUN_DIR
    + "/layer22_four_tap_convolution_decomposition_metrics.jsonl"
)
PARENT_METRICS_SHA256 = "65d845ecc50021a7dfb1d3b3cd4fd3431843c9d7c8318268f09df0945b4fcf76"
PARENT_SUMMARY_REL = PARENT_RUN_DIR + "/summary.json"
PARENT_SUMMARY_SHA256 = "620413f7f33a372a082a5ab0e1e938552dc6fa330836768b7e5b3e996f1a9872"
PARENT_MANIFEST_REL = PARENT_RUN_DIR + "/execution_manifest.json"
PARENT_MANIFEST_SHA256 = "77e2a06147637f7a233843ed2df2e2ad2f558071785c2ea06bd8c7726fda2e07"
PARENT_REPORT_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_four_tap_convolution_decomposition_"
    "validated_evidence_analysis_report_candidate.md"
)
PARENT_REPORT_SHA256 = "d63df7810ba30647fd6b245080e891aac3d9f84ea743f840012cc88fc2979cea"
PARENT_REPORT_BLOB = "f9b7cae9d867611809de66243174bf8ac7ed25d6"

RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_audit.py"
)
K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FULL_FORWARD_COUNT = 1344
EXPECTED_PREFLIGHT_FORWARD_COUNT = 4
EXPECTED_LAYER_COUNT = 24

SOURCE_LAYER = 22
TARGET_LAG = 0
TARGET_K = 2
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 1536
STATE_SIZE = 16
CONV_KERNEL_SIZE = 4
LAG0_KERNEL_INDEX = 3
EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"

EXPECTED_LAG0_KERNEL_RMS = 0.24383223809052498

KERNEL_RMS_REL_TOL = 1e-13
KERNEL_RMS_ABS_TOL = 1e-13
PARENT_METRIC_REL_TOL = 1e-13
PARENT_METRIC_ABS_TOL = 1e-13

ENERGY_NORMALIZATION_ABS_TOL = 5e-13
CENTERED_KERNEL_IDENTITY_ABS_TOL = 5e-12
CHANNEL_CONTRIBUTION_CLOSURE_ABS_TOL = 5e-12
STRONG_WEAK_ENERGY_CLOSURE_ABS_TOL = 5e-13
TRANSFER_BRIDGE_ABS_TOL = 5e-13
AGGREGATE_CONTRIBUTION_CLOSURE_ABS_TOL = 5e-12

QUESTION = (
    "Why does the fixed layer-22 lag-0 depthwise-convolution tap transfer "
    "the corr current-token hidden-difference direction more strongly than ctrl?"
)


class Layer22Lag0ChannelTransferError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Layer22Lag0ChannelTransferError(message)


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


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Layer22Lag0ChannelTransferError(
            f"GIT_FAILURE:{' '.join(args)}"
        ) from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", spec],
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Layer22Lag0ChannelTransferError(
            f"GIT_SHOW_FAILURE:{spec}"
        ) from exc


def _status_path(line: str) -> str:
    raw = line[3:] if len(line) >= 4 else ""
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip('"').replace("\\", "/")


def authenticate_repo(
    root: Path,
    runtime_mode: bool,
) -> dict[str, Any]:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")

    require(
        branch == EXPECTED_BRANCH,
        f"GIT_BRANCH_MISMATCH:{branch}",
    )

    for ancestor, label in (
        (AUTHORITY_FREEZE_COMMIT, "AUTHORITY"),
        (PARENT_EVIDENCE_FREEZE_COMMIT, "PARENT_EVIDENCE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(
            rc == 0,
            f"{label}_FREEZE_NOT_ANCESTOR",
        )

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
    ).splitlines()

    for line in status:
        path = _status_path(line)
        xy = line[:2]

        if path in K1_UNTRACKED:
            require(
                xy == "??",
                f"K1_STATE_CHANGED:{line}",
            )
            continue

        if not runtime_mode and path == RUNNER_REL:
            require(
                xy == "??",
                f"STATIC_RUNNER_STATE_UNEXPECTED:{line}",
            )
            continue

        raise Layer22Lag0ChannelTransferError(
            f"UNEXPECTED_WORKTREE_CHANGE:{line}"
        )

    runner = root / RUNNER_REL
    require(
        runner.is_file(),
        "RUNNER_FILE_MISSING",
    )

    if runtime_mode:
        tracked = subprocess.call(
            ["git", "ls-files", "--error-unmatch", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(
            tracked == 0,
            "RUNTIME_REQUIRES_TRACKED_RUNNER",
        )

        clean = subprocess.call(
            ["git", "diff", "--quiet", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(
            clean == 0,
            "RUNTIME_RUNNER_WORKTREE_DRIFT",
        )

    return {
        "branch": branch,
        "head": head,
        "status": status,
        "runtime_mode": runtime_mode,
    }


def authenticate_authority(root: Path) -> None:
    frozen = git_bytes(
        root,
        f"{AUTHORITY_FREEZE_COMMIT}:{AUTHORITY_REL}",
    )

    require(
        sha256_bytes(frozen) == AUTHORITY_SHA256,
        "AUTHORITY_SHA256_MISMATCH",
    )
    require(
        git(
            root,
            "rev-parse",
            f"{AUTHORITY_FREEZE_COMMIT}:{AUTHORITY_REL}",
        )
        == AUTHORITY_BLOB,
        "AUTHORITY_BLOB_MISMATCH",
    )

    current = root / AUTHORITY_REL
    require(
        current.is_file(),
        "AUTHORITY_FILE_MISSING",
    )
    require(
        current.read_bytes() == frozen,
        "AUTHORITY_WORKTREE_DRIFT",
    )


def load_parent(root: Path):
    path = root / PARENT_RUNNER_REL
    require(
        path.is_file(),
        "PARENT_RUNNER_MISSING",
    )

    frozen = git_bytes(
        root,
        f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{PARENT_RUNNER_REL}",
    )

    require(
        sha256_bytes(frozen) == PARENT_RUNNER_SHA256,
        "PARENT_RUNNER_SHA256_MISMATCH",
    )
    require(
        git(
            root,
            "rev-parse",
            f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{PARENT_RUNNER_REL}",
        )
        == PARENT_RUNNER_BLOB,
        "PARENT_RUNNER_BLOB_MISMATCH",
    )
    require(
        path.read_bytes() == frozen,
        "PARENT_RUNNER_WORKTREE_DRIFT",
    )

    return import_module(
        path,
        "k0_rvg_layer22_four_tap_parent",
    )


def _load_json(raw: bytes, label: str):
    try:
        return json.loads(raw)
    except Exception as exc:
        raise Layer22Lag0ChannelTransferError(
            f"{label}_JSON_PARSE_FAILURE"
        ) from exc


def authenticate_parent_evidence(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[tuple[int, str], dict[str, Any]],
]:
    specs = (
        (
            PARENT_METRICS_REL,
            PARENT_METRICS_SHA256,
            "METRICS",
        ),
        (
            PARENT_SUMMARY_REL,
            PARENT_SUMMARY_SHA256,
            "SUMMARY",
        ),
        (
            PARENT_MANIFEST_REL,
            PARENT_MANIFEST_SHA256,
            "MANIFEST",
        ),
        (
            PARENT_REPORT_REL,
            PARENT_REPORT_SHA256,
            "REPORT",
        ),
    )

    raws: dict[str, bytes] = {}

    for rel, expected_sha, label in specs:
        raw = git_bytes(
            root,
            f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{rel}",
        )
        require(
            sha256_bytes(raw) == expected_sha,
            f"PARENT_{label}_SHA256_MISMATCH",
        )

        current = root / rel
        require(
            current.is_file(),
            f"PARENT_{label}_MISSING",
        )
        require(
            current.read_bytes() == raw,
            f"PARENT_{label}_WORKTREE_DRIFT",
        )
        raws[label] = raw

    require(
        git(
            root,
            "rev-parse",
            f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{PARENT_REPORT_REL}",
        )
        == PARENT_REPORT_BLOB,
        "PARENT_REPORT_BLOB_MISMATCH",
    )

    summary = _load_json(
        raws["SUMMARY"],
        "PARENT_SUMMARY",
    )
    manifest = _load_json(
        raws["MANIFEST"],
        "PARENT_MANIFEST",
    )

    require(
        summary.get("schema_version")
        == "k0-rvg-layer22-four-tap-convolution-decomposition-summary-v1",
        "PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("schema_version")
        == "k0-rvg-layer22-four-tap-convolution-decomposition-execution-manifest-v1",
        "PARENT_MANIFEST_SCHEMA_MISMATCH",
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
        summary.get("common_ddsssss_item_count")
        == EXPECTED_COMMON_COUNT,
        "PARENT_COMMON_COUNT_MISMATCH",
    )
    require(
        summary.get("source_layer") == SOURCE_LAYER,
        "PARENT_SOURCE_LAYER_MISMATCH",
    )
    require(
        summary.get("parent_h_rf_trajectory_match") is True,
        "PARENT_H_RF_MATCH_NOT_TRUE",
    )
    require(
        summary.get("parent_delta_c_trajectory_match") is True,
        "PARENT_DELTA_C_MATCH_NOT_TRUE",
    )

    require(
        manifest.get("runtime_git_head") == PARENT_RUNTIME_HEAD,
        "PARENT_RUNTIME_HEAD_MISMATCH",
    )
    require(
        manifest.get("runner_sha256") == PARENT_RUNNER_SHA256,
        "PARENT_MANIFEST_RUNNER_SHA_MISMATCH",
    )
    require(
        manifest.get("model_forward_count")
        == EXPECTED_FULL_FORWARD_COUNT,
        "PARENT_FORWARD_COUNT_MISMATCH",
    )
    require(
        manifest.get("outputs", {}).get(
            "layer22_four_tap_convolution_decomposition_metrics.jsonl"
        )
        == PARENT_METRICS_SHA256,
        "PARENT_MANIFEST_METRICS_HASH_MISMATCH",
    )
    require(
        manifest.get("outputs", {}).get("summary.json")
        == PARENT_SUMMARY_SHA256,
        "PARENT_MANIFEST_SUMMARY_HASH_MISMATCH",
    )

    parent_rms = (
        manifest
        .get("kernel_weight_rms_by_causal_lag", {})
        .get("0")
    )
    require(
        parent_rms is not None,
        "PARENT_LAG0_KERNEL_RMS_MISSING",
    )
    require(
        math.isclose(
            float(parent_rms),
            EXPECTED_LAG0_KERNEL_RMS,
            rel_tol=KERNEL_RMS_REL_TOL,
            abs_tol=KERNEL_RMS_ABS_TOL,
        ),
        "PARENT_LAG0_KERNEL_RMS_MISMATCH",
    )

    parent_rows: dict[tuple[int, str], dict[str, Any]] = {}
    for line_no, line in enumerate(
        raws["METRICS"].splitlines(),
        start=1,
    ):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise Layer22Lag0ChannelTransferError(
                f"PARENT_METRICS_JSONL_PARSE_FAILURE:{line_no}"
            ) from exc

        if (
            bool(row.get("in_common_ddsssss_cohort"))
            and int(row.get("relative_coordinate")) == TARGET_K
        ):
            key = (
                int(row["local_template_index"]),
                str(row["role"]),
            )
            require(
                key not in parent_rows,
                f"PARENT_COMMON_K2_DUPLICATE:{key}",
            )
            parent_rows[key] = row

    require(
        len(parent_rows) == EXPECTED_COMMON_COUNT * 2,
        "PARENT_COMMON_K2_ROW_COUNT_MISMATCH",
    )

    return summary, manifest, parent_rows


def build_plan(
    root: Path,
    parent: Any,
):
    u_path_parent = parent.load_parent(root)
    values = parent.build_plan(
        root,
        u_path_parent,
    )

    require(
        len(values) == 22,
        "PARENT_BUILD_PLAN_ARITY_MISMATCH",
    )

    (
        write_parent,
        carry_parent,
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
    ) = values

    require(
        len(plan) == EXPECTED_PAIR_ROLE_COUNT,
        "PLAN_COUNT_MISMATCH",
    )
    require(
        len(cohort) == EXPECTED_COMMON_COUNT,
        "COMMON_COHORT_COUNT_MISMATCH",
    )

    return (
        u_path_parent,
        write_parent,
        carry_parent,
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


def resolve_runtime_bundle(
    root: Path,
    parent: Any,
    u_path_parent: Any,
    write_parent: Any,
    carry_parent: Any,
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
        u_path_parent,
        write_parent,
        carry_parent,
        recurrent_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        handoff_path,
    )

    require(
        len(bundle) == 14,
        "PARENT_RUNTIME_BUNDLE_ARITY_MISMATCH",
    )
    require(
        len(bundle[4]) == EXPECTED_LAYER_COUNT,
        "LAYER_MAP_COUNT_MISMATCH",
    )
    return bundle


def resolve_lag0_kernel(
    parent: Any,
    u_path_parent: Any,
    mixer: Any,
):
    import torch

    (
        w_hidden,
        kernel32,
        bias,
        kernel,
        weight_rms,
        exact_zero,
    ) = parent.resolve_kernel(
        u_path_parent,
        mixer,
    )

    require(
        tuple(kernel.shape)
        == (INTERMEDIATE_SIZE, CONV_KERNEL_SIZE),
        "KERNEL_SHAPE_MISMATCH",
    )
    require(
        len(weight_rms) == CONV_KERNEL_SIZE,
        "KERNEL_RMS_COUNT_MISMATCH",
    )
    require(
        exact_zero[TARGET_LAG] is False,
        "LAG0_KERNEL_UNEXPECTED_ZERO",
    )

    lag0 = (
        kernel[:, LAG0_KERNEL_INDEX]
        .to(torch.float64)
        .contiguous()
    )
    lag0_k2 = lag0 * lag0
    mu_k2 = float(
        torch.mean(lag0_k2).item()
    )
    rms = math.sqrt(mu_k2)

    require(
        math.isclose(
            rms,
            float(weight_rms[TARGET_LAG]),
            rel_tol=KERNEL_RMS_REL_TOL,
            abs_tol=KERNEL_RMS_ABS_TOL,
        ),
        "LAG0_RMS_PARENT_RUNTIME_BRIDGE_FAILURE",
    )
    require(
        math.isclose(
            rms,
            EXPECTED_LAG0_KERNEL_RMS,
            rel_tol=KERNEL_RMS_REL_TOL,
            abs_tol=KERNEL_RMS_ABS_TOL,
        ),
        "LAG0_RMS_FROZEN_PARENT_MISMATCH",
    )
    require(
        mu_k2 > 0.0 and math.isfinite(mu_k2),
        "MU_K2_INVALID",
    )

    normalized_k2 = lag0_k2 / mu_k2
    centered_strength = normalized_k2 - 1.0

    strong_mask = lag0_k2 > mu_k2
    weak_mask = lag0_k2 < mu_k2
    equal_mask = lag0_k2 == mu_k2

    require(
        int(
            strong_mask.sum().item()
            + weak_mask.sum().item()
            + equal_mask.sum().item()
        )
        == INTERMEDIATE_SIZE,
        "KERNEL_PARTITION_COUNT_MISMATCH",
    )

    # Deterministic descending k^2 order, channel index breaks exact ties.
    kernel_order = sorted(
        range(INTERMEDIATE_SIZE),
        key=lambda j: (
            -float(lag0_k2[j].item()),
            int(j),
        ),
    )

    return {
        "w_hidden": w_hidden,
        "kernel32": kernel32,
        "bias": bias,
        "kernel": kernel,
        "lag0": lag0,
        "lag0_k2": lag0_k2,
        "mu_k2": mu_k2,
        "rms": rms,
        "normalized_k2": normalized_k2,
        "centered_strength": centered_strength,
        "strong_mask": strong_mask,
        "weak_mask": weak_mask,
        "equal_mask": equal_mask,
        "kernel_order": tuple(kernel_order),
        "weight_rms": weight_rms,
        "exact_zero": exact_zero,
    }


def _prefixes(
    parent: Any,
    u_path_parent: Any,
    write_parent: Any,
    carry_parent: Any,
    recurrent_parent: Any,
    row: Mapping[str, Any],
):
    matched, swapped = parent._prefixes(
        u_path_parent,
        write_parent,
        carry_parent,
        recurrent_parent,
        row,
    )
    return tuple(matched), tuple(swapped)


def capture(
    parent: Any,
    u_path_parent: Any,
    write_parent: Any,
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    mixer: Any,
    kernel_info: Mapping[str, Any],
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    return parent.capture(
        u_path_parent,
        write_parent,
        base,
        model,
        binding,
        layer_map,
        mixer,
        kernel_info["w_hidden"],
        kernel_info["kernel32"],
        kernel_info["bias"],
        token_ids,
        targets,
    )


def _aggregate_nullable(values):
    vals = list(values)
    defined = [
        float(v)
        for v in vals
        if v is not None
    ]

    require(
        all(math.isfinite(v) for v in defined),
        "NONFINITE_AGGREGATE",
    )

    result = {
        "count": len(vals),
        "defined_count": len(defined),
        "undefined_count": len(vals) - len(defined),
        "mean": None,
        "median": None,
        "min": None,
        "max": None,
    }

    if defined:
        result.update(
            {
                "mean": float(
                    statistics.fmean(defined)
                ),
                "median": float(
                    statistics.median(defined)
                ),
                "min": float(min(defined)),
                "max": float(max(defined)),
            }
        )

    return result


def _paired_count(item_rows, corr_field: str, ctrl_field: str):
    gt = lt = eq = undefined = 0

    for row in item_rows:
        a = row[corr_field]
        b = row[ctrl_field]

        if a is None or b is None:
            undefined += 1
        elif float(a) > float(b):
            gt += 1
        elif float(a) < float(b):
            lt += 1
        else:
            eq += 1

    return {
        "count": len(item_rows),
        "defined_pair_count": len(item_rows) - undefined,
        "undefined_pair_count": undefined,
        "corr_gt_ctrl": gt,
        "corr_lt_ctrl": lt,
        "equal": eq,
    }


def _isclose_parent(got, expected) -> bool:
    if got is None or expected is None:
        return got is None and expected is None

    return math.isclose(
        float(got),
        float(expected),
        rel_tol=PARENT_METRIC_REL_TOL,
        abs_tol=PARENT_METRIC_ABS_TOL,
    )


def compute_role_exposure(
    h,
    kernel_info: Mapping[str, Any],
    label: str,
):
    import torch

    h = h.to(torch.float64).contiguous()

    require(
        tuple(h.shape) == (INTERMEDIATE_SIZE,),
        f"{label}_H_SHAPE_MISMATCH",
    )
    require(
        bool(torch.isfinite(h).all().item()),
        f"{label}_H_NONFINITE",
    )

    h_sq = h * h
    h_sq_sum = float(
        torch.sum(h_sq).item()
    )
    h_l2 = math.sqrt(max(h_sq_sum, 0.0))

    if h_sq_sum == 0.0:
        return {
            "defined": False,
            "h_l2": 0.0,
            "q_l2": 0.0,
            "t0": None,
            "n0": None,
            "n0_sq": None,
            "strong_energy_mass": None,
            "weak_energy_mass": None,
            "equal_energy_mass": None,
            "energy_normalization_abs_residual": 0.0,
            "centered_kernel_identity_abs_residual": 0.0,
            "strong_weak_energy_closure_abs_residual": 0.0,
            "transfer_bridge_abs_residual": 0.0,
            "p": None,
        }

    p = h_sq / h_sq_sum

    p_sum = float(
        torch.sum(p).item()
    )
    energy_residual = abs(p_sum - 1.0)
    require(
        energy_residual
        <= ENERGY_NORMALIZATION_ABS_TOL,
        f"{label}_ENERGY_NORMALIZATION_FAILURE:{energy_residual}",
    )

    q = h * kernel_info["lag0"]
    q_sq_sum = float(
        torch.sum(q * q).item()
    )
    q_l2 = math.sqrt(max(q_sq_sum, 0.0))

    t0 = q_l2 / h_l2
    n0 = t0 / float(kernel_info["rms"])
    n0_sq = n0 * n0

    weighted_k2 = float(
        torch.sum(
            p * kernel_info["lag0_k2"]
        ).item()
    )
    transfer_bridge_residual = abs(
        t0 * t0 - weighted_k2
    )
    require(
        transfer_bridge_residual
        <= TRANSFER_BRIDGE_ABS_TOL,
        f"{label}_TRANSFER_BRIDGE_FAILURE:{transfer_bridge_residual}",
    )

    centered_rhs = float(
        torch.sum(
            p
            * kernel_info["centered_strength"]
        ).item()
    )
    centered_residual = abs(
        (n0_sq - 1.0)
        - centered_rhs
    )
    require(
        centered_residual
        <= CENTERED_KERNEL_IDENTITY_ABS_TOL,
        f"{label}_CENTERED_KERNEL_IDENTITY_FAILURE:{centered_residual}",
    )

    strong_mass = float(
        torch.sum(
            p[kernel_info["strong_mask"]]
        ).item()
    )
    weak_mass = float(
        torch.sum(
            p[kernel_info["weak_mask"]]
        ).item()
    )
    equal_mass = float(
        torch.sum(
            p[kernel_info["equal_mask"]]
        ).item()
    )

    partition_residual = abs(
        strong_mass
        + weak_mass
        + equal_mass
        - 1.0
    )
    require(
        partition_residual
        <= STRONG_WEAK_ENERGY_CLOSURE_ABS_TOL,
        f"{label}_STRONG_WEAK_ENERGY_CLOSURE_FAILURE:{partition_residual}",
    )

    return {
        "defined": True,
        "h_l2": h_l2,
        "q_l2": q_l2,
        "t0": t0,
        "n0": n0,
        "n0_sq": n0_sq,
        "strong_energy_mass": strong_mass,
        "weak_energy_mass": weak_mass,
        "equal_energy_mass": equal_mass,
        "energy_normalization_abs_residual": energy_residual,
        "centered_kernel_identity_abs_residual": centered_residual,
        "strong_weak_energy_closure_abs_residual": partition_residual,
        "transfer_bridge_abs_residual": transfer_bridge_residual,
        "p": p,
    }


def role_record_from_capture(
    row: Mapping[str, Any],
    matched: Mapping[int, Mapping[str, Any]],
    swapped: Mapping[int, Mapping[str, Any]],
    cohort: frozenset[int],
    kernel_info: Mapping[str, Any],
    parent_rows: Mapping[tuple[int, str], Mapping[str, Any]],
):
    import torch

    idx = int(
        row["local_template_index"]
    )
    role = str(row["role"])

    require(
        idx in cohort,
        f"NONCOMMON_ROLE_RECORD_REQUEST:{idx}:{role}",
    )

    token = int(row["anchor"]) + TARGET_K
    require(
        token in matched and token in swapped,
        f"TARGET_TOKEN_MISSING:{idx}:{role}",
    )

    hm = matched[token]["H_RF32"]
    hs = swapped[token]["H_RF32"]

    require(
        tuple(hm.shape)
        == (CONV_KERNEL_SIZE, INTERMEDIATE_SIZE),
        f"MATCHED_H_RF_SHAPE_MISMATCH:{idx}:{role}",
    )
    require(
        tuple(hs.shape)
        == (CONV_KERNEL_SIZE, INTERMEDIATE_SIZE),
        f"SWAPPED_H_RF_SHAPE_MISMATCH:{idx}:{role}",
    )

    dh_lag0 = (
        hm[TARGET_LAG, :].to(torch.float64)
        - hs[TARGET_LAG, :].to(torch.float64)
    ).contiguous()

    exposure = compute_role_exposure(
        dh_lag0,
        kernel_info,
        f"{idx}:{role}",
    )

    parent_key = (idx, role)
    require(
        parent_key in parent_rows,
        f"PARENT_ROW_MISSING:{parent_key}",
    )
    parent_row = parent_rows[parent_key]

    bridges = {
        "delta_h_lag0_l2":
            exposure["h_l2"],
        "q0_l2":
            exposure["q_l2"],
        "q0_tap_transfer":
            exposure["t0"],
        "q0_transfer_over_weight_rms":
            exposure["n0"],
    }

    for field, got in bridges.items():
        expected = parent_row[field]
        require(
            _isclose_parent(
                got,
                expected,
            ),
            (
                f"PARENT_METRIC_REPRODUCTION_FAILURE:"
                f"{idx}:{role}:{field}:{got}:{expected}"
            ),
        )

    return {
        "idx": idx,
        "stable_item_id":
            row["stable_item_id"],
        "role": role,
        "token_index": token,
        "anchor": int(row["anchor"]),
        "signature":
            parent_row[
                "token_equality_signature_k0_to_k6"
            ],
        "h_l2": exposure["h_l2"],
        "q_l2": exposure["q_l2"],
        "t0": exposure["t0"],
        "n0": exposure["n0"],
        "n0_sq": exposure["n0_sq"],
        "strong_energy_mass":
            exposure["strong_energy_mass"],
        "weak_energy_mass":
            exposure["weak_energy_mass"],
        "equal_energy_mass":
            exposure["equal_energy_mass"],
        "energy_normalization_abs_residual":
            exposure[
                "energy_normalization_abs_residual"
            ],
        "centered_kernel_identity_abs_residual":
            exposure[
                "centered_kernel_identity_abs_residual"
            ],
        "strong_weak_energy_closure_abs_residual":
            exposure[
                "strong_weak_energy_closure_abs_residual"
            ],
        "transfer_bridge_abs_residual":
            exposure[
                "transfer_bridge_abs_residual"
            ],
        "p": exposure["p"],
    }


def pair_role_records(
    corr: Mapping[str, Any],
    ctrl: Mapping[str, Any],
    kernel_info: Mapping[str, Any],
):
    import torch

    require(
        corr["idx"] == ctrl["idx"],
        "PAIR_INDEX_MISMATCH",
    )
    require(
        corr["stable_item_id"]
        == ctrl["stable_item_id"],
        "PAIR_STABLE_ID_MISMATCH",
    )

    idx = int(corr["idx"])

    p_corr = corr["p"]
    p_ctrl = ctrl["p"]

    if p_corr is None or p_ctrl is None:
        require(
            p_corr is None
            and p_ctrl is None,
            f"PAIR_PARTIAL_UNDEFINED:{idx}",
        )
        delta_n0_sq = None
        channel_contribution_sum = None
        channel_closure_residual = None
        delta_p = None
        c = None
    else:
        delta_p = (
            p_corr - p_ctrl
        ).contiguous()
        c = (
            delta_p
            * kernel_info[
                "centered_strength"
            ]
        ).contiguous()

        delta_n0_sq = (
            float(corr["n0_sq"])
            - float(ctrl["n0_sq"])
        )
        channel_contribution_sum = float(
            torch.sum(c).item()
        )
        channel_closure_residual = abs(
            channel_contribution_sum
            - delta_n0_sq
        )

        require(
            channel_closure_residual
            <= CHANNEL_CONTRIBUTION_CLOSURE_ABS_TOL,
            (
                f"CHANNEL_CONTRIBUTION_CLOSURE_FAILURE:"
                f"{idx}:{channel_closure_residual}"
            ),
        )

        require(
            abs(
                float(
                    torch.sum(delta_p).item()
                )
            )
            <= ENERGY_NORMALIZATION_ABS_TOL,
            f"DELTA_P_SUM_FAILURE:{idx}",
        )

    def delta(a, b):
        if a is None or b is None:
            return None
        return float(a) - float(b)

    row = {
        "schema_version":
            "k0-rvg-layer22-lag0-channel-transfer-item-v1",
        "local_template_index": idx,
        "stable_item_id":
            corr["stable_item_id"],
        "relative_coordinate": TARGET_K,
        "source_layer": SOURCE_LAYER,
        "causal_lag": TARGET_LAG,
        "delta_h_lag0_l2_corr":
            corr["h_l2"],
        "delta_h_lag0_l2_ctrl":
            ctrl["h_l2"],
        "q0_l2_corr":
            corr["q_l2"],
        "q0_l2_ctrl":
            ctrl["q_l2"],
        "t0_corr":
            corr["t0"],
        "t0_ctrl":
            ctrl["t0"],
        "n0_corr":
            corr["n0"],
        "n0_ctrl":
            ctrl["n0"],
        "n0_sq_corr":
            corr["n0_sq"],
        "n0_sq_ctrl":
            ctrl["n0_sq"],
        "delta_n0_sq":
            delta_n0_sq,
        "strong_energy_mass_corr":
            corr["strong_energy_mass"],
        "strong_energy_mass_ctrl":
            ctrl["strong_energy_mass"],
        "delta_strong_energy_mass":
            delta(
                corr["strong_energy_mass"],
                ctrl["strong_energy_mass"],
            ),
        "weak_energy_mass_corr":
            corr["weak_energy_mass"],
        "weak_energy_mass_ctrl":
            ctrl["weak_energy_mass"],
        "delta_weak_energy_mass":
            delta(
                corr["weak_energy_mass"],
                ctrl["weak_energy_mass"],
            ),
        "equal_energy_mass_corr":
            corr["equal_energy_mass"],
        "equal_energy_mass_ctrl":
            ctrl["equal_energy_mass"],
        "channel_contribution_sum":
            channel_contribution_sum,
        "channel_contribution_closure_abs_residual":
            channel_closure_residual,
        "energy_normalization_abs_residual":
            max(
                corr[
                    "energy_normalization_abs_residual"
                ],
                ctrl[
                    "energy_normalization_abs_residual"
                ],
            ),
        "centered_kernel_identity_abs_residual":
            max(
                corr[
                    "centered_kernel_identity_abs_residual"
                ],
                ctrl[
                    "centered_kernel_identity_abs_residual"
                ],
            ),
        "strong_weak_energy_closure_abs_residual":
            max(
                corr[
                    "strong_weak_energy_closure_abs_residual"
                ],
                ctrl[
                    "strong_weak_energy_closure_abs_residual"
                ],
            ),
        "transfer_bridge_abs_residual":
            max(
                corr[
                    "transfer_bridge_abs_residual"
                ],
                ctrl[
                    "transfer_bridge_abs_residual"
                ],
            ),
        "raw_channel_vectors_persisted": False,
    }

    return row, p_corr, p_ctrl, delta_p, c


def build_channel_outputs(
    p_corr_matrix,
    p_ctrl_matrix,
    c_matrix,
    kernel_info: Mapping[str, Any],
):
    import torch

    require(
        p_corr_matrix.ndim == 2,
        "P_CORR_MATRIX_RANK_MISMATCH",
    )
    require(
        tuple(p_corr_matrix.shape)
        == (EXPECTED_COMMON_COUNT, INTERMEDIATE_SIZE),
        "P_CORR_MATRIX_SHAPE_MISMATCH",
    )
    require(
        tuple(p_ctrl_matrix.shape)
        == tuple(p_corr_matrix.shape),
        "P_CTRL_MATRIX_SHAPE_MISMATCH",
    )
    require(
        tuple(c_matrix.shape)
        == tuple(p_corr_matrix.shape),
        "C_MATRIX_SHAPE_MISMATCH",
    )

    mean_p_corr = torch.mean(
        p_corr_matrix,
        dim=0,
    )
    mean_p_ctrl = torch.mean(
        p_ctrl_matrix,
        dim=0,
    )
    mean_delta_p = (
        mean_p_corr - mean_p_ctrl
    )
    mean_c = torch.mean(
        c_matrix,
        dim=0,
    )
    # Use the same even-sample median convention as statistics.median:
    # average the two central order statistics for common-330.
    sorted_c, _ = torch.sort(
        c_matrix,
        dim=0,
    )
    lower = sorted_c[
        (EXPECTED_COMMON_COUNT // 2) - 1,
        :,
    ]
    upper = sorted_c[
        EXPECTED_COMMON_COUNT // 2,
        :,
    ]
    median_c = (
        lower + upper
    ) / 2.0

    positive_counts = torch.sum(
        c_matrix > 0.0,
        dim=0,
    )
    negative_counts = torch.sum(
        c_matrix < 0.0,
        dim=0,
    )
    zero_counts = torch.sum(
        c_matrix == 0.0,
        dim=0,
    )

    rank_by_channel = [0] * INTERMEDIATE_SIZE
    for rank, channel in enumerate(
        kernel_info["kernel_order"],
        start=1,
    ):
        rank_by_channel[channel] = rank

    channel_rows = []

    for j in range(INTERMEDIATE_SIZE):
        channel_rows.append(
            {
                "schema_version":
                    "k0-rvg-layer22-lag0-channel-transfer-channel-v1",
                "channel_index":
                    j,
                "kernel_weight":
                    float(
                        kernel_info["lag0"][j].item()
                    ),
                "kernel_weight_sq":
                    float(
                        kernel_info["lag0_k2"][j].item()
                    ),
                "kernel_weight_sq_over_mean":
                    float(
                        kernel_info[
                            "normalized_k2"
                        ][j].item()
                    ),
                "centered_kernel_strength":
                    float(
                        kernel_info[
                            "centered_strength"
                        ][j].item()
                    ),
                "kernel_magnitude_rank":
                    int(rank_by_channel[j]),
                "kernel_partition":
                    (
                        "strong"
                        if bool(
                            kernel_info[
                                "strong_mask"
                            ][j].item()
                        )
                        else (
                            "weak"
                            if bool(
                                kernel_info[
                                    "weak_mask"
                                ][j].item()
                            )
                            else "equal"
                        )
                    ),
                "mean_p_corr":
                    float(
                        mean_p_corr[j].item()
                    ),
                "mean_p_ctrl":
                    float(
                        mean_p_ctrl[j].item()
                    ),
                "mean_delta_p":
                    float(
                        mean_delta_p[j].item()
                    ),
                "mean_channel_contribution":
                    float(
                        mean_c[j].item()
                    ),
                "median_channel_contribution":
                    float(
                        median_c[j].item()
                    ),
                "channel_contribution_positive_count":
                    int(
                        positive_counts[j].item()
                    ),
                "channel_contribution_negative_count":
                    int(
                        negative_counts[j].item()
                    ),
                "channel_contribution_zero_count":
                    int(
                        zero_counts[j].item()
                    ),
            }
        )

    cumulative_rows = []
    cumulative_c = 0.0
    cumulative_p_corr = 0.0
    cumulative_p_ctrl = 0.0

    for rank, channel in enumerate(
        kernel_info["kernel_order"],
        start=1,
    ):
        cumulative_c += float(
            mean_c[channel].item()
        )
        cumulative_p_corr += float(
            mean_p_corr[channel].item()
        )
        cumulative_p_ctrl += float(
            mean_p_ctrl[channel].item()
        )

        cumulative_rows.append(
            {
                "schema_version":
                    "k0-rvg-layer22-lag0-kernel-rank-cumulative-v1",
                "kernel_magnitude_rank":
                    rank,
                "channel_index":
                    int(channel),
                "kernel_weight_sq":
                    float(
                        kernel_info[
                            "lag0_k2"
                        ][channel].item()
                    ),
                "cumulative_mean_channel_contribution":
                    cumulative_c,
                "cumulative_mean_p_corr":
                    cumulative_p_corr,
                "cumulative_mean_p_ctrl":
                    cumulative_p_ctrl,
            }
        )

    m_abs = float(
        torch.sum(
            torch.abs(mean_c)
        ).item()
    )
    m_net = float(
        torch.sum(mean_c).item()
    )
    denom = float(
        torch.sum(
            mean_c * mean_c
        ).item()
    )

    r_cancel = (
        m_net / m_abs
        if m_abs > 0.0
        else None
    )
    n_eff = (
        (m_abs * m_abs) / denom
        if denom > 0.0
        else None
    )

    strong_total = float(
        torch.sum(
            mean_c[
                kernel_info["strong_mask"]
            ]
        ).item()
    )
    weak_total = float(
        torch.sum(
            mean_c[
                kernel_info["weak_mask"]
            ]
        ).item()
    )
    equal_total = float(
        torch.sum(
            mean_c[
                kernel_info["equal_mask"]
            ]
        ).item()
    )

    aggregate = {
        "absolute_mean_contribution_mass":
            m_abs,
        "net_mean_contribution":
            m_net,
        "signed_cancellation_ratio":
            r_cancel,
        "effective_channel_count":
            n_eff,
        "positive_mean_contribution_channel_count":
            int(
                torch.sum(
                    mean_c > 0.0
                ).item()
            ),
        "negative_mean_contribution_channel_count":
            int(
                torch.sum(
                    mean_c < 0.0
                ).item()
            ),
        "zero_mean_contribution_channel_count":
            int(
                torch.sum(
                    mean_c == 0.0
                ).item()
            ),
        "strong_kernel_total_mean_contribution":
            strong_total,
        "weak_kernel_total_mean_contribution":
            weak_total,
        "equal_kernel_total_mean_contribution":
            equal_total,
    }

    return (
        channel_rows,
        cumulative_rows,
        aggregate,
    )


def make_summary(
    item_rows: Sequence[Mapping[str, Any]],
    channel_aggregate: Mapping[str, Any],
    kernel_info: Mapping[str, Any],
):
    require(
        len(item_rows) == EXPECTED_COMMON_COUNT,
        "ITEM_ROW_COUNT_MISMATCH",
    )

    role_fields = (
        "t0",
        "n0",
        "n0_sq",
        "strong_energy_mass",
        "weak_energy_mass",
        "equal_energy_mass",
    )

    role_summary = {
        "corr": {},
        "ctrl": {},
    }

    for role in ("corr", "ctrl"):
        for field in role_fields:
            role_summary[role][field] = (
                _aggregate_nullable(
                    row[f"{field}_{role}"]
                    for row in item_rows
                )
            )

    paired_counts = {
        field: _paired_count(
            item_rows,
            f"{field}_corr",
            f"{field}_ctrl",
        )
        for field in role_fields
    }

    delta_fields = (
        "delta_n0_sq",
        "delta_strong_energy_mass",
        "delta_weak_energy_mass",
        "channel_contribution_sum",
    )

    delta_summary = {
        field: _aggregate_nullable(
            row[field]
            for row in item_rows
        )
        for field in delta_fields
    }

    g_tap_values = []
    for row in item_rows:
        a = row["t0_corr"]
        b = row["t0_ctrl"]
        if (
            a is None
            or b is None
            or float(a) <= 0.0
            or float(b) <= 0.0
        ):
            g_tap_values.append(None)
        else:
            g_tap_values.append(
                math.log(
                    float(a) / float(b)
                )
            )

    mean_delta_n0_sq = delta_summary[
        "delta_n0_sq"
    ]["mean"]
    aggregate_residual = None

    if mean_delta_n0_sq is not None:
        aggregate_residual = abs(
            float(
                channel_aggregate[
                    "net_mean_contribution"
                ]
            )
            - float(mean_delta_n0_sq)
        )

        require(
            aggregate_residual
            <= AGGREGATE_CONTRIBUTION_CLOSURE_ABS_TOL,
            (
                "AGGREGATE_CONTRIBUTION_CLOSURE_FAILURE:"
                f"{aggregate_residual}"
            ),
        )

    return {
        "schema_version":
            "k0-rvg-layer22-lag0-channel-transfer-summary-v1",
        "scientific_question":
            QUESTION,
        "source_layer":
            SOURCE_LAYER,
        "causal_lag":
            TARGET_LAG,
        "relative_coordinate":
            TARGET_K,
        "common_ddsssss_item_count":
            EXPECTED_COMMON_COUNT,
        "intermediate_size":
            INTERMEDIATE_SIZE,
        "execution_protocol":
            EXECUTION_PROTOCOL,
        "lag0_kernel_rms":
            float(kernel_info["rms"]),
        "lag0_kernel_mean_sq":
            float(kernel_info["mu_k2"]),
        "strong_kernel_channel_count":
            int(
                kernel_info[
                    "strong_mask"
                ].sum().item()
            ),
        "weak_kernel_channel_count":
            int(
                kernel_info[
                    "weak_mask"
                ].sum().item()
            ),
        "equal_kernel_channel_count":
            int(
                kernel_info[
                    "equal_mask"
                ].sum().item()
            ),
        "role_summary":
            role_summary,
        "paired_counts":
            paired_counts,
        "delta_summary":
            delta_summary,
        "g_tap0_log_ratio":
            _aggregate_nullable(
                g_tap_values
            ),
        "channel_aggregate":
            dict(channel_aggregate),
        "max_energy_normalization_abs_residual":
            max(
                float(
                    row[
                        "energy_normalization_abs_residual"
                    ]
                )
                for row in item_rows
            ),
        "max_centered_kernel_identity_abs_residual":
            max(
                float(
                    row[
                        "centered_kernel_identity_abs_residual"
                    ]
                )
                for row in item_rows
            ),
        "max_channel_contribution_closure_abs_residual":
            max(
                float(
                    row[
                        "channel_contribution_closure_abs_residual"
                    ]
                )
                for row in item_rows
                if row[
                    "channel_contribution_closure_abs_residual"
                ]
                is not None
            ),
        "max_strong_weak_energy_closure_abs_residual":
            max(
                float(
                    row[
                        "strong_weak_energy_closure_abs_residual"
                    ]
                )
                for row in item_rows
            ),
        "max_transfer_bridge_abs_residual":
            max(
                float(
                    row[
                        "transfer_bridge_abs_residual"
                    ]
                )
                for row in item_rows
            ),
        "aggregate_contribution_closure_abs_residual":
            aggregate_residual,
        "parent_common_k2_lag0_metrics_match":
            True,
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
        "pca_svd_whitening_or_learned_geometry_executed":
            False,
        "posthoc_layer_lag_channel_item_or_window_search_executed":
            False,
    }


def json_bytes(obj) -> bytes:
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


def jsonl_bytes(rows) -> bytes:
    return b"".join(
        json_bytes(row)
        for row in rows
    )


def _first_common_pair_rows(
    plan: Sequence[Mapping[str, Any]],
    cohort: frozenset[int],
):
    by_idx: dict[int, dict[str, Mapping[str, Any]]] = {}

    for row in plan:
        idx = int(
            row["local_template_index"]
        )
        if idx not in cohort:
            continue

        role = str(row["role"])
        by_idx.setdefault(
            idx,
            {},
        )[role] = row

    for idx in sorted(by_idx):
        pair = by_idx[idx]
        if "corr" in pair and "ctrl" in pair:
            return pair["corr"], pair["ctrl"]

    raise Layer22Lag0ChannelTransferError(
        "NO_COMMON_PREFLIGHT_PAIR"
    )


def runtime_preflight(
    root: Path,
    parent: Any,
    u_path_parent: Any,
    write_parent: Any,
    carry_parent: Any,
    recurrent_parent: Any,
    output_parent: Any,
    residual_parent: Any,
    rms_parent: Any,
    hidden_parent: Any,
    postconv: Any,
    base: Any,
    plan,
    cohort,
    parent_rows,
    handoff_path: Path,
):
    import torch

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
        mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
        u_path_parent,
        write_parent,
        carry_parent,
        recurrent_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        handoff_path,
    )

    kernel_info = resolve_lag0_kernel(
        parent,
        u_path_parent,
        mixer22,
    )

    corr_row, ctrl_row = (
        _first_common_pair_rows(
            plan,
            cohort,
        )
    )

    role_records = {}
    forward_count = 0

    for row in (corr_row, ctrl_row):
        matched_prefix, swapped_prefix = (
            _prefixes(
                parent,
                u_path_parent,
                write_parent,
                carry_parent,
                recurrent_parent,
                row,
            )
        )

        matched = capture(
            parent,
            u_path_parent,
            write_parent,
            base,
            model,
            binding,
            layer_map,
            mixer22,
            kernel_info,
            matched_prefix,
            row["targets"],
        )
        forward_count += 1

        swapped = capture(
            parent,
            u_path_parent,
            write_parent,
            base,
            model,
            binding,
            layer_map,
            mixer22,
            kernel_info,
            swapped_prefix,
            row["targets"],
        )
        forward_count += 1

        rec = role_record_from_capture(
            row,
            matched,
            swapped,
            cohort,
            kernel_info,
            parent_rows,
        )
        role_records[
            rec["role"]
        ] = rec

    require(
        forward_count
        == EXPECTED_PREFLIGHT_FORWARD_COUNT,
        "PREFLIGHT_FORWARD_COUNT_MISMATCH",
    )
    require(
        set(role_records)
        == {"corr", "ctrl"},
        "PREFLIGHT_ROLE_SET_MISMATCH",
    )

    item_row, p_corr, p_ctrl, _dp, c = (
        pair_role_records(
            role_records["corr"],
            role_records["ctrl"],
            kernel_info,
        )
    )

    require(
        p_corr is not None
        and p_ctrl is not None
        and c is not None,
        "PREFLIGHT_TARGET_UNDEFINED",
    )

    print(
        "PASS_LAYER22_LAG0_CHANNEL_TRANSFER_RUNTIME_PREFLIGHT"
    )
    print(
        "local_template_index =",
        item_row["local_template_index"],
    )
    print(
        "model_forward_count =",
        forward_count,
    )
    print(
        "scientific_population_accessed = True"
    )
    print(
        "scientific_evidence_emitted = False"
    )
    print(
        "raw_vectors_persisted = False"
    )
    print(
        "lag0_kernel_rms =",
        kernel_info["rms"],
    )
    print(
        "lag0_kernel_mean_sq =",
        kernel_info["mu_k2"],
    )
    print(
        "strong_kernel_channel_count =",
        int(
            kernel_info[
                "strong_mask"
            ].sum().item()
        ),
    )
    print(
        "weak_kernel_channel_count =",
        int(
            kernel_info[
                "weak_mask"
            ].sum().item()
        ),
    )
    print(
        "equal_kernel_channel_count =",
        int(
            kernel_info[
                "equal_mask"
            ].sum().item()
        ),
    )
    print(
        "t0_corr =",
        item_row["t0_corr"],
    )
    print(
        "t0_ctrl =",
        item_row["t0_ctrl"],
    )
    print(
        "n0_sq_corr =",
        item_row["n0_sq_corr"],
    )
    print(
        "n0_sq_ctrl =",
        item_row["n0_sq_ctrl"],
    )
    print(
        "delta_n0_sq =",
        item_row["delta_n0_sq"],
    )
    print(
        "channel_contribution_sum =",
        item_row[
            "channel_contribution_sum"
        ],
    )
    print(
        "channel_contribution_closure_abs_residual =",
        item_row[
            "channel_contribution_closure_abs_residual"
        ],
    )
    print(
        "energy_normalization_abs_residual =",
        item_row[
            "energy_normalization_abs_residual"
        ],
    )
    print(
        "centered_kernel_identity_abs_residual =",
        item_row[
            "centered_kernel_identity_abs_residual"
        ],
    )
    print(
        "strong_weak_energy_closure_abs_residual =",
        item_row[
            "strong_weak_energy_closure_abs_residual"
        ],
    )
    print(
        "transfer_bridge_abs_residual =",
        item_row[
            "transfer_bridge_abs_residual"
        ],
    )


def execute(
    root: Path,
    parent: Any,
    u_path_parent: Any,
    write_parent: Any,
    carry_parent: Any,
    recurrent_parent: Any,
    output_parent: Any,
    residual_parent: Any,
    rms_parent: Any,
    hidden_parent: Any,
    postconv: Any,
    base: Any,
    repo: Mapping[str, Any],
    plan,
    cohort,
    parent_rows,
    handoff_path: Path,
    output_dir: Path,
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
        mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
        u_path_parent,
        write_parent,
        carry_parent,
        recurrent_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        handoff_path,
    )

    kernel_info = resolve_lag0_kernel(
        parent,
        u_path_parent,
        mixer22,
    )

    role_records: dict[
        tuple[int, str],
        dict[str, Any],
    ] = {}

    forward_count = 0

    for n, row in enumerate(
        plan,
        start=1,
    ):
        idx = int(
            row["local_template_index"]
        )
        role = str(row["role"])

        matched_prefix, swapped_prefix = (
            _prefixes(
                parent,
                u_path_parent,
                write_parent,
                carry_parent,
                recurrent_parent,
                row,
            )
        )

        matched = capture(
            parent,
            u_path_parent,
            write_parent,
            base,
            model,
            binding,
            layer_map,
            mixer22,
            kernel_info,
            matched_prefix,
            row["targets"],
        )
        forward_count += 1

        swapped = capture(
            parent,
            u_path_parent,
            write_parent,
            base,
            model,
            binding,
            layer_map,
            mixer22,
            kernel_info,
            swapped_prefix,
            row["targets"],
        )
        forward_count += 1

        if idx in cohort:
            rec = role_record_from_capture(
                row,
                matched,
                swapped,
                cohort,
                kernel_info,
                parent_rows,
            )
            key = (idx, role)
            require(
                key not in role_records,
                f"ROLE_RECORD_DUPLICATE:{key}",
            )
            role_records[key] = rec

        if (
            n % 16 == 0
            or n == len(plan)
        ):
            print(
                (
                    f"PROGRESS pair_roles={n}/{len(plan)} "
                    f"model_forwards={forward_count}"
                ),
                flush=True,
            )

    require(
        forward_count
        == EXPECTED_FULL_FORWARD_COUNT,
        "FORWARD_COUNT_MISMATCH",
    )
    require(
        len(role_records)
        == EXPECTED_COMMON_COUNT * 2,
        "COMMON_ROLE_RECORD_COUNT_MISMATCH",
    )

    item_rows = []
    p_corr_vectors = []
    p_ctrl_vectors = []
    c_vectors = []

    for idx in sorted(cohort):
        corr_key = (int(idx), "corr")
        ctrl_key = (int(idx), "ctrl")

        require(
            corr_key in role_records,
            f"CORR_ROLE_RECORD_MISSING:{idx}",
        )
        require(
            ctrl_key in role_records,
            f"CTRL_ROLE_RECORD_MISSING:{idx}",
        )

        (
            item_row,
            p_corr,
            p_ctrl,
            _delta_p,
            c,
        ) = pair_role_records(
            role_records[corr_key],
            role_records[ctrl_key],
            kernel_info,
        )

        require(
            p_corr is not None
            and p_ctrl is not None
            and c is not None,
            f"COMMON_TARGET_UNDEFINED:{idx}",
        )

        item_rows.append(
            item_row
        )
        p_corr_vectors.append(
            p_corr
        )
        p_ctrl_vectors.append(
            p_ctrl
        )
        c_vectors.append(
            c
        )

    require(
        len(item_rows)
        == EXPECTED_COMMON_COUNT,
        "ITEM_ROW_COUNT_MISMATCH",
    )

    p_corr_matrix = torch.stack(
        p_corr_vectors,
        dim=0,
    )
    p_ctrl_matrix = torch.stack(
        p_ctrl_vectors,
        dim=0,
    )
    c_matrix = torch.stack(
        c_vectors,
        dim=0,
    )

    (
        channel_rows,
        cumulative_rows,
        channel_aggregate,
    ) = build_channel_outputs(
        p_corr_matrix,
        p_ctrl_matrix,
        c_matrix,
        kernel_info,
    )

    summary = make_summary(
        item_rows,
        channel_aggregate,
        kernel_info,
    )

    partial_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    item_path = (
        partial_dir
        / "layer22_lag0_channel_transfer_item_metrics.jsonl"
    )
    channel_path = (
        partial_dir
        / "layer22_lag0_channel_transfer_channel_summary.jsonl"
    )
    cumulative_path = (
        partial_dir
        / "layer22_lag0_kernel_rank_cumulative.jsonl"
    )
    summary_path = (
        partial_dir
        / "summary.json"
    )
    manifest_path = (
        partial_dir
        / "execution_manifest.json"
    )

    item_path.write_bytes(
        jsonl_bytes(item_rows)
    )
    channel_path.write_bytes(
        jsonl_bytes(channel_rows)
    )
    cumulative_path.write_bytes(
        jsonl_bytes(cumulative_rows)
    )
    summary_path.write_bytes(
        json_bytes(summary)
    )

    manifest = {
        "schema_version":
            "k0-rvg-layer22-lag0-channel-transfer-execution-manifest-v1",
        "runtime_git_head":
            repo["head"],
        "runtime_branch":
            repo["branch"],
        "authority_freeze_commit":
            AUTHORITY_FREEZE_COMMIT,
        "authority_sha256":
            AUTHORITY_SHA256,
        "authority_blob":
            AUTHORITY_BLOB,
        "parent_evidence_freeze_commit":
            PARENT_EVIDENCE_FREEZE_COMMIT,
        "parent_implementation_commit":
            PARENT_IMPLEMENTATION_COMMIT,
        "parent_runtime_head":
            PARENT_RUNTIME_HEAD,
        "parent_runner_sha256":
            PARENT_RUNNER_SHA256,
        "parent_runner_blob":
            PARENT_RUNNER_BLOB,
        "parent_metrics_sha256":
            PARENT_METRICS_SHA256,
        "parent_summary_sha256":
            PARENT_SUMMARY_SHA256,
        "parent_manifest_sha256":
            PARENT_MANIFEST_SHA256,
        "parent_report_sha256":
            PARENT_REPORT_SHA256,
        "parent_report_blob":
            PARENT_REPORT_BLOB,
        "scientific_question":
            QUESTION,
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
        "causal_lag":
            TARGET_LAG,
        "relative_coordinate":
            TARGET_K,
        "hidden_size":
            HIDDEN_SIZE,
        "intermediate_size":
            INTERMEDIATE_SIZE,
        "state_size":
            STATE_SIZE,
        "conv_kernel_size":
            CONV_KERNEL_SIZE,
        "lag0_kernel_index":
            LAG0_KERNEL_INDEX,
        "lag0_kernel_rms":
            float(kernel_info["rms"]),
        "lag0_kernel_mean_sq":
            float(kernel_info["mu_k2"]),
        "strong_kernel_channel_count":
            int(
                kernel_info[
                    "strong_mask"
                ].sum().item()
            ),
        "weak_kernel_channel_count":
            int(
                kernel_info[
                    "weak_mask"
                ].sum().item()
            ),
        "equal_kernel_channel_count":
            int(
                kernel_info[
                    "equal_mask"
                ].sum().item()
            ),
        "mamba_source_sha256":
            binding.source_sha256,
        "kernel_rms_rel_tol":
            KERNEL_RMS_REL_TOL,
        "kernel_rms_abs_tol":
            KERNEL_RMS_ABS_TOL,
        "parent_metric_rel_tol":
            PARENT_METRIC_REL_TOL,
        "parent_metric_abs_tol":
            PARENT_METRIC_ABS_TOL,
        "energy_normalization_abs_tol":
            ENERGY_NORMALIZATION_ABS_TOL,
        "centered_kernel_identity_abs_tol":
            CENTERED_KERNEL_IDENTITY_ABS_TOL,
        "channel_contribution_closure_abs_tol":
            CHANNEL_CONTRIBUTION_CLOSURE_ABS_TOL,
        "strong_weak_energy_closure_abs_tol":
            STRONG_WEAK_ENERGY_CLOSURE_ABS_TOL,
        "transfer_bridge_abs_tol":
            TRANSFER_BRIDGE_ABS_TOL,
        "aggregate_contribution_closure_abs_tol":
            AGGREGATE_CONTRIBUTION_CLOSURE_ABS_TOL,
        "handoff_zip_sha256":
            handoff["zip_sha256"],
        "checkpoint_sha256":
            handoff["checkpoint_sha256"],
        "encoder_canonical_digest":
            encoder["canonical_digest"],
        "encoder_raw_concat_digest":
            encoder["raw_concat_digest"],
        "parent_common_k2_lag0_metrics_match":
            True,
        "capture_method": (
            "frozen parent layer-22 H_RF capture; "
            "only current-token lag-0 branch difference is used scientifically; "
            "authenticated fixed lag-0 kernel is applied algebraically in float64"
        ),
        "raw_vectors_persisted":
            False,
        "raw_per_item_channel_vectors_persisted":
            False,
        "scientific_model_forward_executed":
            True,
        "scientific_h_rf_read":
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
        "pca_svd_whitening_or_learned_geometry_executed":
            False,
        "posthoc_layer_lag_channel_item_or_window_search_executed":
            False,
        "runner_rel":
            RUNNER_REL,
        "runner_sha256":
            sha256_bytes(
                (root / RUNNER_REL)
                .read_bytes()
            ),
        "outputs": {
            item_path.name:
                sha256_bytes(
                    item_path.read_bytes()
                ),
            channel_path.name:
                sha256_bytes(
                    channel_path.read_bytes()
                ),
            cumulative_path.name:
                sha256_bytes(
                    cumulative_path.read_bytes()
                ),
            summary_path.name:
                sha256_bytes(
                    summary_path.read_bytes()
                ),
        },
    }

    manifest_path.write_bytes(
        json_bytes(manifest)
    )

    # Explicitly drop raw matrices before publishing the directory.
    del p_corr_matrix
    del p_ctrl_matrix
    del c_matrix
    del p_corr_vectors
    del p_ctrl_vectors
    del c_vectors
    del role_records

    os.replace(
        partial_dir,
        final_dir,
    )

    role_summary = summary[
        "role_summary"
    ]
    paired = summary[
        "paired_counts"
    ]
    delta_summary = summary[
        "delta_summary"
    ]
    ca = summary[
        "channel_aggregate"
    ]

    print(
        "PASS_LAYER22_LAG0_CHANNEL_TRANSFER_LOCALIZATION_EXECUTION"
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
        "parent_common_k2_lag0_metrics_match = True"
    )
    print(
        "lag0_kernel_rms =",
        summary["lag0_kernel_rms"],
    )
    print(
        "strong_kernel_channel_count =",
        summary[
            "strong_kernel_channel_count"
        ],
    )
    print(
        "weak_kernel_channel_count =",
        summary[
            "weak_kernel_channel_count"
        ],
    )
    print(
        "equal_kernel_channel_count =",
        summary[
            "equal_kernel_channel_count"
        ],
    )

    for role in ("corr", "ctrl"):
        print(
            (
                f"{role}_k2_lag0 "
                f"T_median={role_summary[role]['t0']['median']} "
                f"N_median={role_summary[role]['n0']['median']} "
                f"N2_median={role_summary[role]['n0_sq']['median']} "
                f"strong_energy_median="
                f"{role_summary[role]['strong_energy_mass']['median']} "
                f"strong_energy_mean="
                f"{role_summary[role]['strong_energy_mass']['mean']}"
            )
        )

    print(
        "k2_delta_n0_sq_median =",
        delta_summary[
            "delta_n0_sq"
        ]["median"],
    )
    print(
        "k2_strong_energy_corr_gt_ctrl =",
        paired[
            "strong_energy_mass"
        ]["corr_gt_ctrl"],
    )
    print(
        "k2_n0_sq_corr_gt_ctrl =",
        paired[
            "n0_sq"
        ]["corr_gt_ctrl"],
    )
    print(
        "g_tap0_median =",
        summary[
            "g_tap0_log_ratio"
        ]["median"],
    )
    print(
        "channel_M_abs =",
        ca[
            "absolute_mean_contribution_mass"
        ],
    )
    print(
        "channel_M_net =",
        ca[
            "net_mean_contribution"
        ],
    )
    print(
        "channel_R_cancel =",
        ca[
            "signed_cancellation_ratio"
        ],
    )
    print(
        "channel_N_eff =",
        ca[
            "effective_channel_count"
        ],
    )
    print(
        "strong_kernel_total_mean_contribution =",
        ca[
            "strong_kernel_total_mean_contribution"
        ],
    )
    print(
        "weak_kernel_total_mean_contribution =",
        ca[
            "weak_kernel_total_mean_contribution"
        ],
    )
    print(
        "max_channel_contribution_closure_abs_residual =",
        summary[
            "max_channel_contribution_closure_abs_residual"
        ],
    )
    print(
        "max_centered_kernel_identity_abs_residual =",
        summary[
            "max_centered_kernel_identity_abs_residual"
        ],
    )
    print(
        "max_transfer_bridge_abs_residual =",
        summary[
            "max_transfer_bridge_abs_residual"
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static-preflight",
        action="store_true",
    )
    parser.add_argument(
        "--runtime-preflight",
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
        sum(
            bool(v)
            for v in (
                args.static_preflight,
                args.runtime_preflight,
                args.execute,
            )
        )
        == 1,
        "SELECT_EXACTLY_ONE_MODE",
    )

    root = Path.cwd().resolve()
    runtime_mode = bool(
        args.runtime_preflight
        or args.execute
    )

    repo = authenticate_repo(
        root,
        runtime_mode=runtime_mode,
    )
    authenticate_authority(root)

    parent = load_parent(root)

    (
        parent_summary,
        parent_manifest,
        parent_rows,
    ) = authenticate_parent_evidence(
        root
    )

    (
        u_path_parent,
        write_parent,
        carry_parent,
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
    ) = build_plan(
        root,
        parent,
    )

    source = (
        secant
        .validate_frozen_source_semantics()
    )
    require(
        source["source_sha256"]
        == output_parent.EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    _observer, observer_binding = (
        carry_parent
        .authenticate_observer_static(
            root,
            base,
            output_parent.EXPECTED_MAMBA_SOURCE_SHA256,
        )
    )

    print(
        "=== LAYER22 LAG0 CHANNEL-CONDITIONED TRANSFER LOCALIZATION AUDIT PLAN ==="
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
        "authority_freeze_commit =",
        AUTHORITY_FREEZE_COMMIT,
    )
    print(
        "authority_sha256 =",
        AUTHORITY_SHA256,
    )
    print(
        "authority_blob =",
        AUTHORITY_BLOB,
    )
    print(
        "parent_evidence_freeze_commit =",
        PARENT_EVIDENCE_FREEZE_COMMIT,
    )
    print(
        "parent_runner_sha256 =",
        PARENT_RUNNER_SHA256,
    )
    print(
        "parent_metrics_sha256 =",
        PARENT_METRICS_SHA256,
    )
    print(
        "parent_summary_sha256 =",
        PARENT_SUMMARY_SHA256,
    )
    print(
        "parent_manifest_sha256 =",
        PARENT_MANIFEST_SHA256,
    )
    print(
        "parent_report_sha256 =",
        PARENT_REPORT_SHA256,
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
        "causal_lag =",
        TARGET_LAG,
    )
    print(
        "relative_coordinate =",
        TARGET_K,
    )
    print(
        "intermediate_size =",
        INTERMEDIATE_SIZE,
    )
    print(
        "scientific_question =",
        QUESTION,
    )
    print(
        "identity_t2 = sum_j p_j * k_j^2"
    )
    print(
        "identity_n2_minus_1 = sum_j p_j * (k_j^2/mu_k2 - 1)"
    )
    print(
        "identity_delta_n2 = sum_j c_j"
    )
    print(
        "expected_lag0_kernel_rms =",
        EXPECTED_LAG0_KERNEL_RMS,
    )
    print(
        "energy_normalization_abs_tol =",
        ENERGY_NORMALIZATION_ABS_TOL,
    )
    print(
        "centered_kernel_identity_abs_tol =",
        CENTERED_KERNEL_IDENTITY_ABS_TOL,
    )
    print(
        "channel_contribution_closure_abs_tol =",
        CHANNEL_CONTRIBUTION_CLOSURE_ABS_TOL,
    )
    print(
        "transfer_bridge_abs_tol =",
        TRANSFER_BRIDGE_ABS_TOL,
    )
    print(
        "observer_update_line =",
        observer_binding.update_line,
    )
    print(
        "observer_readout_line =",
        observer_binding.readout_line,
    )
    print(
        "mamba_source_sha256 =",
        source["source_sha256"],
    )
    print(
        "all_1536_channels_preregistered = True"
    )
    print(
        "raw_vectors_persisted = False"
    )
    print(
        "tokenizer_invoked = False"
    )
    print(
        "training_executed = False"
    )
    print(
        "causal_intervention_executed = False"
    )
    print(
        "posthoc_channel_layer_lag_window_search_executed = False"
    )

    if args.static_preflight:
        print(
            "scientific_model_forward_executed = False"
        )
        print(
            "PASS_LAYER22_LAG0_CHANNEL_TRANSFER_STATIC_PREFLIGHT"
        )
        return 0

    require(
        args.handoff is not None,
        "RUNTIME_MODE_REQUIRES_HANDOFF",
    )
    handoff_path = (
        args.handoff.resolve()
    )
    require(
        handoff_path.is_file(),
        f"HANDOFF_MISSING:{handoff_path}",
    )

    if args.runtime_preflight:
        runtime_preflight(
            root,
            parent,
            u_path_parent,
            write_parent,
            carry_parent,
            recurrent_parent,
            output_parent,
            residual_parent,
            rms_parent,
            hidden_parent,
            postconv,
            base,
            plan,
            cohort,
            parent_rows,
            handoff_path,
        )
        return 0

    require(
        args.output_dir is not None,
        "EXECUTE_REQUIRES_OUTPUT_DIR",
    )

    execute(
        root,
        parent,
        u_path_parent,
        write_parent,
        carry_parent,
        recurrent_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        postconv,
        base,
        repo,
        plan,
        cohort,
        parent_rows,
        handoff_path,
        args.output_dir,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Layer22Lag0ChannelTransferError as exc:
        print(
            f"BLOCKED: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception as exc:
        print(
            (
                "BLOCKED_UNEXPECTED: "
                f"{type(exc).__name__}: {exc}"
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
