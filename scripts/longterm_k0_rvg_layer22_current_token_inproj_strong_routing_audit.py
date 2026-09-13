"""K0-RVG layer-22 current-token in-projection strong-routing decomposition.

Frozen scientific question:
    How does the fixed bias-free layer-22 current-token hidden in-projection
    map corr and ctrl delta-X directions into the different delta-H channel
    energy distributions observed at the lag-0 convolution input?

Authenticated boundary:
    h = W_H x

For output row w_j:
    h_j = w_j^T x
    e_j = h_j^2 / ||x||^2
    r_j^2 = ||w_j||^2
    A_j(x) = cos^2(w_j, x)
    e_j = r_j^2 A_j(x)

For aligned corr/ctrl items:
    D_j = r_j^2 (A_corr,j - A_ctrl,j)
    sum_j D_j = T_H,corr^2 - T_H,ctrl^2

The downstream strong/weak H-channel partition is frozen by the preceding
lag-0 kernel evidence. Reconstruction validation preserves the frozen parent
branch-level gate and bridges the differenced algebra through the exact
branch-error difference identity. The analysis is observational/algebraic
only. Raw per-item channel vectors are never persisted.
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

AUTHORITY_FREEZE_COMMIT = "d4b3b2fb47c92d2e20b65764178ada4f82c2697b"
AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_current_token_inproj_strong_routing_static_design_candidate.md"
)
AUTHORITY_SHA256 = "e3da4a72418c2aae3379870796e718a5fe4b1e9c0e1e6b8fce172407f3f3bad3"
AUTHORITY_BLOB = "748ae3fc6fffa32e03461dc3ece7271ee49cbf4d"

CORRECTION_AUTHORITY_FREEZE_COMMIT = "3df1cfaac5af7a5551307c33380b5dc1f33fecbd"
CORRECTION_AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_current_token_inproj_strong_routing_"
    "reconstruction_gate_correction_static_design_candidate.md"
)
CORRECTION_AUTHORITY_SHA256 = "d379656ceeb0bbfa78b56bf0408f6eb279facba1c874c408e2c9fe4fae9a4def"
CORRECTION_AUTHORITY_BLOB = "30a1bb34b1f687a53776d2db22a509c5249806bb"
FAILED_IMPLEMENTATION_COMMIT = "cc539ff5a3b9d692fa946c0d8e470685ea8ac6e3"

PARENT_EVIDENCE_FREEZE_COMMIT = "bfa626261aba575ab7316877bc69ca6e5df38157"
PARENT_IMPLEMENTATION_COMMIT = "4c19d02d94600e47f39c15bebd839f5a4820a473"
PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_audit.py"
)
PARENT_RUNNER_SHA256 = "dd1e8ffb0516c4b8392389578815fa5c5c2f05ba826eb37055025b6eda6fae0c"
PARENT_RUNNER_BLOB = "70e14a48b9f2a644d091fb7ca8cfcd74dbb28aaf"

PARENT_RUN_DIR = (
    "reports/"
    "longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_4c19d02_v1"
)
PARENT_ITEM_REL = PARENT_RUN_DIR + "/layer22_lag0_channel_transfer_item_metrics.jsonl"
PARENT_ITEM_SHA256 = "81396fafb131b1efa5e62877d29ddd9c5b860e718fe94adb1f0c5ddbb8b92f81"
PARENT_CHANNEL_REL = PARENT_RUN_DIR + "/layer22_lag0_channel_transfer_channel_summary.jsonl"
PARENT_CHANNEL_SHA256 = "459af9290b110304160b04148ff46b511aca50ca8bae7849aa80dabe5461b451"
PARENT_CUMULATIVE_REL = PARENT_RUN_DIR + "/layer22_lag0_kernel_rank_cumulative.jsonl"
PARENT_CUMULATIVE_SHA256 = "ba838da6f12286a6dd7699275d5257a04366015c262f1a070b28f5ef5ace4c50"
PARENT_SUMMARY_REL = PARENT_RUN_DIR + "/summary.json"
PARENT_SUMMARY_SHA256 = "ca33ff9a00d2527ce480cbefd4244f7830bc6dbd32298fd6356030cc0ecfc3a3"
PARENT_MANIFEST_REL = PARENT_RUN_DIR + "/execution_manifest.json"
PARENT_MANIFEST_SHA256 = "5b304a571dfc17af26c21faa185bd7b9b9a74e3432ae1b80e085fa44b8e8a94c"
PARENT_REPORT_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_"
    "validated_evidence_analysis_report_candidate.md"
)
PARENT_REPORT_SHA256 = "39d6365ad628e27b96654989d91c71e9594b0cc624ef4290ae7e61d1792faa35"
PARENT_REPORT_BLOB = "89682e30aa20c3df4eedc9f23d4b7ca2e7fc5a57"

UPATH_EVIDENCE_FREEZE_COMMIT = "dabb9422dbf0e111319828cf027e8dc5d82fe326"
UPATH_METRICS_REL = (
    "reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1/"
    "layer22_u_path_source_localization_metrics.jsonl"
)
UPATH_METRICS_SHA256 = "735f66534774a8d760f758ba17427af2de0f817340729f3619dddaae2464ef37"
UPATH_SUMMARY_REL = (
    "reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1/summary.json"
)
UPATH_SUMMARY_SHA256 = "75c91fe3c0a047abfd0a74d37d12988dd7c34029c74768e7a86f56290995cb38"
UPATH_MANIFEST_REL = (
    "reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1/"
    "execution_manifest.json"
)
UPATH_MANIFEST_SHA256 = "a91721745e008c6e15a00ec4fc938d2d9de5bb81034fd7888c451219bc8c2c70"
UPATH_REPORT_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_u_path_source_localization_"
    "validated_evidence_analysis_report_candidate.md"
)
UPATH_REPORT_SHA256 = "b110225009ed63d7cc1ad31c9c8dbf1217051833cf9c37f991c371be751beb24"

RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer22_current_token_inproj_strong_routing_audit.py"
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
TARGET_K = 2
TARGET_LAG = 0
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 1536
CONV_KERNEL_SIZE = 4
EXPECTED_STRONG_COUNT = 240
EXPECTED_WEAK_COUNT = 1296
EXPECTED_EQUAL_COUNT = 0
EXPECTED_LAG0_KERNEL_RMS = 0.24383223809052498
EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"

PARENT_SCALAR_REL_TOL = 1e-13
PARENT_SCALAR_ABS_TOL = 1e-13
KERNEL_RMS_REL_TOL = 1e-13
KERNEL_RMS_ABS_TOL = 1e-13
PARENT_BRANCH_H_RECON_REL_TOL = 1e-6
ALGEBRAIC_ABS_TOL = 5e-12
ENERGY_ABS_TOL = 5e-12
PARENT_STRONG_ENERGY_ABS_TOL = 2e-6
ENRICHMENT_IDENTITY_ABS_TOL = 5e-12

QUESTION = (
    "How does the fixed bias-free layer-22 current-token hidden in-projection "
    "map corr and ctrl delta-X directions into the different delta-H channel "
    "energy distributions observed at the lag-0 convolution input?"
)


class CurrentTokenInProjRoutingError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise CurrentTokenInProjRoutingError(message)


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
        raise CurrentTokenInProjRoutingError(f"GIT_FAILURE:{' '.join(args)}") from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CurrentTokenInProjRoutingError(f"GIT_SHOW_FAILURE:{spec}") from exc


def _status_path(line: str) -> str:
    raw = line[3:] if len(line) >= 4 else ""
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip('"').replace("\\", "/")


def authenticate_repo(root: Path, runtime_mode: bool) -> dict[str, Any]:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, f"GIT_BRANCH_MISMATCH:{branch}")

    for ancestor, label in (
        (AUTHORITY_FREEZE_COMMIT, "AUTHORITY"),
        (CORRECTION_AUTHORITY_FREEZE_COMMIT, "CORRECTION_AUTHORITY"),
        (PARENT_EVIDENCE_FREEZE_COMMIT, "PARENT_EVIDENCE"),
        (UPATH_EVIDENCE_FREEZE_COMMIT, "UPATH_EVIDENCE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_FREEZE_NOT_ANCESTOR")

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
            require(
                xy in {" M", "M ", "MM"},
                f"STATIC_CORRECTION_RUNNER_STATE_UNEXPECTED:{line}",
            )
            continue
        raise CurrentTokenInProjRoutingError(f"UNEXPECTED_WORKTREE_CHANGE:{line}")

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
        clean = subprocess.call(
            ["git", "diff", "--quiet", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(clean == 0, "RUNTIME_RUNNER_WORKTREE_DRIFT")

    return {"branch": branch, "head": head, "status": status, "runtime_mode": runtime_mode}


def authenticate_authority(root: Path) -> None:
    frozen = git_bytes(root, f"{AUTHORITY_FREEZE_COMMIT}:{AUTHORITY_REL}")
    require(sha256_bytes(frozen) == AUTHORITY_SHA256, "AUTHORITY_SHA256_MISMATCH")
    require(
        git(root, "rev-parse", f"{AUTHORITY_FREEZE_COMMIT}:{AUTHORITY_REL}") == AUTHORITY_BLOB,
        "AUTHORITY_BLOB_MISMATCH",
    )
    current = root / AUTHORITY_REL
    require(current.is_file(), "AUTHORITY_FILE_MISSING")
    require(current.read_bytes() == frozen, "AUTHORITY_WORKTREE_DRIFT")

    correction = git_bytes(
        root,
        f"{CORRECTION_AUTHORITY_FREEZE_COMMIT}:{CORRECTION_AUTHORITY_REL}",
    )
    require(
        sha256_bytes(correction) == CORRECTION_AUTHORITY_SHA256,
        "CORRECTION_AUTHORITY_SHA256_MISMATCH",
    )
    require(
        git(
            root,
            "rev-parse",
            f"{CORRECTION_AUTHORITY_FREEZE_COMMIT}:{CORRECTION_AUTHORITY_REL}",
        )
        == CORRECTION_AUTHORITY_BLOB,
        "CORRECTION_AUTHORITY_BLOB_MISMATCH",
    )
    correction_current = root / CORRECTION_AUTHORITY_REL
    require(correction_current.is_file(), "CORRECTION_AUTHORITY_FILE_MISSING")
    require(
        correction_current.read_bytes() == correction,
        "CORRECTION_AUTHORITY_WORKTREE_DRIFT",
    )


def load_parent(root: Path):
    path = root / PARENT_RUNNER_REL
    require(path.is_file(), "PARENT_RUNNER_MISSING")
    frozen = git_bytes(root, f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{PARENT_RUNNER_REL}")
    require(sha256_bytes(frozen) == PARENT_RUNNER_SHA256, "PARENT_RUNNER_SHA256_MISMATCH")
    require(
        git(root, "rev-parse", f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{PARENT_RUNNER_REL}")
        == PARENT_RUNNER_BLOB,
        "PARENT_RUNNER_BLOB_MISMATCH",
    )
    require(path.read_bytes() == frozen, "PARENT_RUNNER_WORKTREE_DRIFT")
    return import_module(path, "k0_rvg_layer22_lag0_channel_parent")


def _load_json(raw: bytes, label: str):
    try:
        return json.loads(raw)
    except Exception as exc:
        raise CurrentTokenInProjRoutingError(f"{label}_JSON_PARSE_FAILURE") from exc


def _authenticate_frozen_file(
    root: Path,
    commit: str,
    rel: str,
    expected_sha: str,
    label: str,
) -> bytes:
    raw = git_bytes(root, f"{commit}:{rel}")
    require(sha256_bytes(raw) == expected_sha, f"{label}_SHA256_MISMATCH")
    current = root / rel
    require(current.is_file(), f"{label}_MISSING")
    require(current.read_bytes() == raw, f"{label}_WORKTREE_DRIFT")
    return raw


def authenticate_parent_evidence(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    item_raw = _authenticate_frozen_file(
        root, PARENT_EVIDENCE_FREEZE_COMMIT, PARENT_ITEM_REL, PARENT_ITEM_SHA256, "PARENT_ITEM"
    )
    _authenticate_frozen_file(
        root, PARENT_EVIDENCE_FREEZE_COMMIT, PARENT_CHANNEL_REL, PARENT_CHANNEL_SHA256,
        "PARENT_CHANNEL"
    )
    _authenticate_frozen_file(
        root, PARENT_EVIDENCE_FREEZE_COMMIT, PARENT_CUMULATIVE_REL, PARENT_CUMULATIVE_SHA256,
        "PARENT_CUMULATIVE"
    )
    summary_raw = _authenticate_frozen_file(
        root, PARENT_EVIDENCE_FREEZE_COMMIT, PARENT_SUMMARY_REL, PARENT_SUMMARY_SHA256,
        "PARENT_SUMMARY"
    )
    manifest_raw = _authenticate_frozen_file(
        root, PARENT_EVIDENCE_FREEZE_COMMIT, PARENT_MANIFEST_REL, PARENT_MANIFEST_SHA256,
        "PARENT_MANIFEST"
    )
    _authenticate_frozen_file(
        root, PARENT_EVIDENCE_FREEZE_COMMIT, PARENT_REPORT_REL, PARENT_REPORT_SHA256,
        "PARENT_REPORT"
    )

    require(
        git(root, "rev-parse", f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{PARENT_REPORT_REL}")
        == PARENT_REPORT_BLOB,
        "PARENT_REPORT_BLOB_MISMATCH",
    )

    summary = _load_json(summary_raw, "PARENT_SUMMARY")
    manifest = _load_json(manifest_raw, "PARENT_MANIFEST")

    require(
        summary.get("schema_version") == "k0-rvg-layer22-lag0-channel-transfer-summary-v1",
        "PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("schema_version")
        == "k0-rvg-layer22-lag0-channel-transfer-execution-manifest-v1",
        "PARENT_MANIFEST_SCHEMA_MISMATCH",
    )
    require(manifest.get("runtime_git_head") == PARENT_IMPLEMENTATION_COMMIT,
            "PARENT_RUNTIME_HEAD_MISMATCH")
    require(manifest.get("runner_sha256") == PARENT_RUNNER_SHA256,
            "PARENT_MANIFEST_RUNNER_SHA_MISMATCH")
    require(manifest.get("model_forward_count") == EXPECTED_FULL_FORWARD_COUNT,
            "PARENT_FORWARD_COUNT_MISMATCH")
    require(summary.get("common_ddsssss_item_count") == EXPECTED_COMMON_COUNT,
            "PARENT_COMMON_COUNT_MISMATCH")
    require(summary.get("strong_kernel_channel_count") == EXPECTED_STRONG_COUNT,
            "PARENT_STRONG_COUNT_MISMATCH")
    require(summary.get("weak_kernel_channel_count") == EXPECTED_WEAK_COUNT,
            "PARENT_WEAK_COUNT_MISMATCH")
    require(summary.get("equal_kernel_channel_count") == EXPECTED_EQUAL_COUNT,
            "PARENT_EQUAL_COUNT_MISMATCH")
    require(
        math.isclose(
            float(summary["lag0_kernel_rms"]),
            EXPECTED_LAG0_KERNEL_RMS,
            rel_tol=KERNEL_RMS_REL_TOL,
            abs_tol=KERNEL_RMS_ABS_TOL,
        ),
        "PARENT_KERNEL_RMS_MISMATCH",
    )

    items: dict[int, dict[str, Any]] = {}
    for line_no, line in enumerate(item_raw.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise CurrentTokenInProjRoutingError(
                f"PARENT_ITEM_JSONL_PARSE_FAILURE:{line_no}"
            ) from exc
        idx = int(row["local_template_index"])
        require(idx not in items, f"PARENT_ITEM_DUPLICATE:{idx}")
        items[idx] = row

    require(len(items) == EXPECTED_COMMON_COUNT, "PARENT_ITEM_COUNT_MISMATCH")
    return summary, manifest, items


def authenticate_upath_evidence(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    metrics_raw = _authenticate_frozen_file(
        root, UPATH_EVIDENCE_FREEZE_COMMIT, UPATH_METRICS_REL, UPATH_METRICS_SHA256,
        "UPATH_METRICS"
    )
    summary_raw = _authenticate_frozen_file(
        root, UPATH_EVIDENCE_FREEZE_COMMIT, UPATH_SUMMARY_REL, UPATH_SUMMARY_SHA256,
        "UPATH_SUMMARY"
    )
    manifest_raw = _authenticate_frozen_file(
        root, UPATH_EVIDENCE_FREEZE_COMMIT, UPATH_MANIFEST_REL, UPATH_MANIFEST_SHA256,
        "UPATH_MANIFEST"
    )
    _authenticate_frozen_file(
        root, UPATH_EVIDENCE_FREEZE_COMMIT, UPATH_REPORT_REL, UPATH_REPORT_SHA256,
        "UPATH_REPORT"
    )

    summary = _load_json(summary_raw, "UPATH_SUMMARY")
    manifest = _load_json(manifest_raw, "UPATH_MANIFEST")
    require(
        summary.get("schema_version") == "k0-rvg-layer22-u-path-source-localization-summary-v1",
        "UPATH_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("schema_version")
        == "k0-rvg-layer22-u-path-source-localization-execution-manifest-v1",
        "UPATH_MANIFEST_SCHEMA_MISMATCH",
    )
    require(summary.get("common_ddsssss_item_count") == EXPECTED_COMMON_COUNT,
            "UPATH_COMMON_COUNT_MISMATCH")
    require(summary.get("source_layer") == SOURCE_LAYER, "UPATH_SOURCE_LAYER_MISMATCH")

    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for line_no, line in enumerate(metrics_raw.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise CurrentTokenInProjRoutingError(
                f"UPATH_METRICS_JSONL_PARSE_FAILURE:{line_no}"
            ) from exc
        if (
            bool(row.get("in_common_ddsssss_cohort"))
            and int(row.get("relative_coordinate")) == TARGET_K
        ):
            key = (int(row["local_template_index"]), str(row["role"]))
            require(key not in rows, f"UPATH_COMMON_K2_DUPLICATE:{key}")
            rows[key] = row

    require(len(rows) == EXPECTED_COMMON_COUNT * 2, "UPATH_COMMON_K2_COUNT_MISMATCH")
    return summary, manifest, rows


def build_plan(root: Path, parent: Any):
    four_tap_parent = parent.load_parent(root)
    values = parent.build_plan(root, four_tap_parent)
    require(len(values) == 23, "PARENT_BUILD_PLAN_ARITY_MISMATCH")
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
    ) = values
    require(len(plan) == EXPECTED_PAIR_ROLE_COUNT, "PLAN_COUNT_MISMATCH")
    require(len(cohort) == EXPECTED_COMMON_COUNT, "COMMON_COHORT_COUNT_MISMATCH")
    return (
        four_tap_parent,
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
    four_tap_parent: Any,
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
        four_tap_parent,
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
    require(len(bundle) == 14, "PARENT_RUNTIME_BUNDLE_ARITY_MISMATCH")
    require(len(bundle[4]) == EXPECTED_LAYER_COUNT, "LAYER_MAP_COUNT_MISMATCH")
    return bundle


def resolve_operator(parent: Any, four_tap_parent: Any, u_path_parent: Any, mixer22: Any):
    import torch

    info = parent.resolve_lag0_kernel(four_tap_parent, u_path_parent, mixer22)
    require(tuple(info["w_hidden"].shape) == (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "W_H_SHAPE_MISMATCH")

    w = info["w_hidden"].to(torch.float64).contiguous()
    row_r2 = torch.sum(w * w, dim=1)
    zero_rows = row_r2 == 0.0

    require(int(info["strong_mask"].sum().item()) == EXPECTED_STRONG_COUNT,
            "STRONG_COUNT_RUNTIME_MISMATCH")
    require(int(info["weak_mask"].sum().item()) == EXPECTED_WEAK_COUNT,
            "WEAK_COUNT_RUNTIME_MISMATCH")
    require(int(info["equal_mask"].sum().item()) == EXPECTED_EQUAL_COUNT,
            "EQUAL_COUNT_RUNTIME_MISMATCH")
    require(
        math.isclose(
            float(info["rms"]),
            EXPECTED_LAG0_KERNEL_RMS,
            rel_tol=KERNEL_RMS_REL_TOL,
            abs_tol=KERNEL_RMS_ABS_TOL,
        ),
        "LAG0_KERNEL_RMS_RUNTIME_MISMATCH",
    )
    require(int(zero_rows.sum().item()) == 0, "ZERO_INPROJ_ROW_PRESENT")

    info = dict(info)
    info["w_hidden64"] = w
    info["row_r2"] = row_r2
    info["zero_rows"] = zero_rows
    return info


def _prefixes(
    parent: Any,
    four_tap_parent: Any,
    u_path_parent: Any,
    write_parent: Any,
    carry_parent: Any,
    recurrent_parent: Any,
    row: Mapping[str, Any],
):
    matched, swapped = parent._prefixes(
        four_tap_parent,
        u_path_parent,
        write_parent,
        carry_parent,
        recurrent_parent,
        row,
    )
    return tuple(matched), tuple(swapped)


def capture(
    parent: Any,
    four_tap_parent: Any,
    u_path_parent: Any,
    write_parent: Any,
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    mixer22: Any,
    operator: Mapping[str, Any],
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    return parent.capture(
        four_tap_parent,
        u_path_parent,
        write_parent,
        base,
        model,
        binding,
        layer_map,
        mixer22,
        operator,
        token_ids,
        targets,
    )


def _isclose_parent(got, expected) -> bool:
    if got is None or expected is None:
        return got is None and expected is None
    return math.isclose(
        float(got),
        float(expected),
        rel_tol=PARENT_SCALAR_REL_TOL,
        abs_tol=PARENT_SCALAR_ABS_TOL,
    )


def _median(values):
    vals = sorted(float(v) for v in values)
    n = len(vals)
    require(n > 0, "EMPTY_MEDIAN")
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _aggregate_nullable(values):
    vals = list(values)
    defined = [float(v) for v in vals if v is not None]
    require(all(math.isfinite(v) for v in defined), "NONFINITE_AGGREGATE")
    out = {
        "count": len(vals),
        "defined_count": len(defined),
        "undefined_count": len(vals) - len(defined),
        "mean": None,
        "median": None,
        "min": None,
        "max": None,
    }
    if defined:
        out.update(
            {
                "mean": float(statistics.fmean(defined)),
                "median": float(statistics.median(defined)),
                "min": float(min(defined)),
                "max": float(max(defined)),
            }
        )
    return out


def _paired_count(rows, corr_field: str, ctrl_field: str):
    gt = lt = eq = undefined = 0
    for row in rows:
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
        "count": len(rows),
        "defined_pair_count": len(rows) - undefined,
        "undefined_pair_count": undefined,
        "corr_gt_ctrl": gt,
        "corr_lt_ctrl": lt,
        "equal": eq,
    }


def compute_role_routing(
    x_matched,
    x_swapped,
    h_matched,
    h_swapped,
    operator: Mapping[str, Any],
    label: str,
):
    import torch

    x_matched = x_matched.to(torch.float64).contiguous()
    x_swapped = x_swapped.to(torch.float64).contiguous()
    h_matched = h_matched.to(torch.float64).contiguous()
    h_swapped = h_swapped.to(torch.float64).contiguous()

    require(tuple(x_matched.shape) == (HIDDEN_SIZE,), f"{label}_XM_SHAPE_MISMATCH")
    require(tuple(x_swapped.shape) == (HIDDEN_SIZE,), f"{label}_XS_SHAPE_MISMATCH")
    require(
        tuple(h_matched.shape) == (INTERMEDIATE_SIZE,),
        f"{label}_HM_SHAPE_MISMATCH",
    )
    require(
        tuple(h_swapped.shape) == (INTERMEDIATE_SIZE,),
        f"{label}_HS_SHAPE_MISMATCH",
    )
    for value, name in (
        (x_matched, "XM"),
        (x_swapped, "XS"),
        (h_matched, "HM"),
        (h_swapped, "HS"),
    ):
        require(bool(torch.isfinite(value).all().item()), f"{label}_{name}_NONFINITE")

    x = (x_matched - x_swapped).contiguous()
    h_observed = (h_matched - h_swapped).contiguous()

    x_sq = float(torch.dot(x, x).item())
    h_obs_sq = float(torch.dot(h_observed, h_observed).item())
    require(x_sq > 0.0, f"{label}_X_ZERO")
    require(h_obs_sq > 0.0, f"{label}_H_OBS_ZERO")

    w = operator["w_hidden64"]
    h_alg = torch.mv(w, x)
    h_alg_sq = float(torch.dot(h_alg, h_alg).item())
    require(h_alg_sq > 0.0, f"{label}_H_ALG_ZERO")

    # Correction authority:
    # parent capture preserves the frozen branch-level <=1e-6 gate.
    # The cancellation-sensitive difference-relative residual is diagnostic.
    h_matched_alg = torch.mv(w, x_matched)
    h_swapped_alg = torch.mv(w, x_swapped)
    err_matched = h_matched_alg - h_matched
    err_swapped = h_swapped_alg - h_swapped

    difference_error = h_alg - h_observed
    branch_error_difference = err_matched - err_swapped
    error_difference_identity_abs_residual = float(
        torch.max(torch.abs(difference_error - branch_error_difference)).item()
    )
    require(
        error_difference_identity_abs_residual <= ALGEBRAIC_ABS_TOL,
        f"{label}_ERROR_DIFFERENCE_IDENTITY_FAILURE:"
        f"{error_difference_identity_abs_residual}",
    )

    difference_reconstruction_abs_residual = float(
        torch.linalg.vector_norm(difference_error).item()
    )
    difference_reconstruction_relative_residual = (
        difference_reconstruction_abs_residual / math.sqrt(h_obs_sq)
    )

    matched_h_sq = float(torch.dot(h_matched, h_matched).item())
    swapped_h_sq = float(torch.dot(h_swapped, h_swapped).item())
    require(matched_h_sq > 0.0, f"{label}_HM_ZERO")
    require(swapped_h_sq > 0.0, f"{label}_HS_ZERO")
    matched_branch_reconstruction_relative_residual = float(
        torch.linalg.vector_norm(err_matched).item()
    ) / math.sqrt(matched_h_sq)
    swapped_branch_reconstruction_relative_residual = float(
        torch.linalg.vector_norm(err_swapped).item()
    ) / math.sqrt(swapped_h_sq)

    e = (h_alg * h_alg) / x_sq
    row_r2 = operator["row_r2"]
    alignment = e / row_r2
    require(bool(torch.isfinite(alignment).all().item()), f"{label}_ALIGNMENT_NONFINITE")

    dots = torch.mv(w, x)
    alignment_direct = (dots * dots) / (row_r2 * x_sq)
    factor_residual = float(torch.max(torch.abs(e - row_r2 * alignment_direct)).item())
    alignment_residual = float(torch.max(torch.abs(alignment - alignment_direct)).item())
    require(
        factor_residual <= ALGEBRAIC_ABS_TOL,
        f"{label}_ROW_GAIN_ALIGNMENT_FACTOR_FAILURE:{factor_residual}",
    )
    require(
        alignment_residual <= ALGEBRAIC_ABS_TOL,
        f"{label}_ALIGNMENT_FORMULA_FAILURE:{alignment_residual}",
    )

    t_h_sq = float(torch.sum(e).item())
    t_s_sq = float(torch.sum(e[operator["strong_mask"]]).item())
    t_w_sq = float(torch.sum(e[operator["weak_mask"]]).item())
    t_e_sq = float(torch.sum(e[operator["equal_mask"]]).item())

    total_closure = abs(t_h_sq - (t_s_sq + t_w_sq + t_e_sq))
    require(
        total_closure <= ALGEBRAIC_ABS_TOL,
        f"{label}_STRONG_WEAK_TOTAL_CLOSURE_FAILURE:{total_closure}",
    )

    t_h = math.sqrt(max(t_h_sq, 0.0))
    t_s = math.sqrt(max(t_s_sq, 0.0))
    t_w = math.sqrt(max(t_w_sq, 0.0))

    p_s = t_s_sq / t_h_sq
    p_w = t_w_sq / t_h_sq
    p_e = t_e_sq / t_h_sq
    energy_closure = abs(p_s + p_w + p_e - 1.0)
    require(
        energy_closure <= ENERGY_ABS_TOL,
        f"{label}_ENERGY_PARTITION_CLOSURE_FAILURE:{energy_closure}",
    )

    obs_energy = (h_observed * h_observed) / h_obs_sq
    p_s_observed = float(torch.sum(obs_energy[operator["strong_mask"]]).item())
    p_w_observed = float(torch.sum(obs_energy[operator["weak_mask"]]).item())
    p_e_observed = float(torch.sum(obs_energy[operator["equal_mask"]]).item())
    parent_energy_closure = abs(p_s_observed + p_w_observed + p_e_observed - 1.0)
    require(
        parent_energy_closure <= ENERGY_ABS_TOL,
        f"{label}_OBS_ENERGY_PARTITION_CLOSURE_FAILURE:{parent_energy_closure}",
    )

    p_s_bridge = abs(p_s - p_s_observed)
    require(
        p_s_bridge <= PARENT_STRONG_ENERGY_ABS_TOL,
        f"{label}_P_S_ALG_OBS_BRIDGE_FAILURE:{p_s_bridge}",
    )

    return {
        "x_l2": math.sqrt(x_sq),
        "h_observed_l2": math.sqrt(h_obs_sq),
        "h_algebraic_l2": math.sqrt(h_alg_sq),
        "difference_reconstruction_absolute_residual":
            difference_reconstruction_abs_residual,
        "difference_reconstruction_relative_residual":
            difference_reconstruction_relative_residual,
        "matched_branch_reconstruction_relative_residual":
            matched_branch_reconstruction_relative_residual,
        "swapped_branch_reconstruction_relative_residual":
            swapped_branch_reconstruction_relative_residual,
        "error_difference_identity_abs_residual":
            error_difference_identity_abs_residual,
        "row_gain_alignment_factor_abs_residual": factor_residual,
        "alignment_formula_abs_residual": alignment_residual,
        "strong_weak_total_closure_abs_residual": total_closure,
        "energy_partition_closure_abs_residual": energy_closure,
        "parent_energy_partition_closure_abs_residual": parent_energy_closure,
        "p_s_algebraic_observed_bridge_abs_residual": p_s_bridge,
        "t_h": t_h,
        "t_s": t_s,
        "t_w": t_w,
        "t_h_sq": t_h_sq,
        "t_s_sq": t_s_sq,
        "t_w_sq": t_w_sq,
        "p_s": p_s,
        "p_w": p_w,
        "p_s_observed": p_s_observed,
        "p_w_observed": p_w_observed,
        "e": e,
        "alignment": alignment_direct,
    }


def role_record_from_capture(
    row: Mapping[str, Any],
    matched: Mapping[int, Mapping[str, Any]],
    swapped: Mapping[int, Mapping[str, Any]],
    cohort: frozenset[int],
    operator: Mapping[str, Any],
    upath_rows: Mapping[tuple[int, str], Mapping[str, Any]],
    parent_items: Mapping[int, Mapping[str, Any]],
):
    import torch

    idx = int(row["local_template_index"])
    role = str(row["role"])
    require(idx in cohort, f"NONCOMMON_ROLE_RECORD_REQUEST:{idx}:{role}")

    token = int(row["anchor"]) + TARGET_K
    require(token in matched and token in swapped, f"TARGET_TOKEN_MISSING:{idx}:{role}")

    xm = matched[token]["X_RF32"]
    xs = swapped[token]["X_RF32"]
    hm = matched[token]["H_RF32"]
    hs = swapped[token]["H_RF32"]

    require(tuple(xm.shape) == (CONV_KERNEL_SIZE, HIDDEN_SIZE),
            f"XM_RF_SHAPE_MISMATCH:{idx}:{role}")
    require(tuple(xs.shape) == (CONV_KERNEL_SIZE, HIDDEN_SIZE),
            f"XS_RF_SHAPE_MISMATCH:{idx}:{role}")
    require(tuple(hm.shape) == (CONV_KERNEL_SIZE, INTERMEDIATE_SIZE),
            f"HM_RF_SHAPE_MISMATCH:{idx}:{role}")
    require(tuple(hs.shape) == (CONV_KERNEL_SIZE, INTERMEDIATE_SIZE),
            f"HS_RF_SHAPE_MISMATCH:{idx}:{role}")

    parent_branch_reconstruction_relative_residual = max(
        float(matched[token]["inproj_reconstruction_relative_residual"]),
        float(swapped[token]["inproj_reconstruction_relative_residual"]),
    )
    require(
        parent_branch_reconstruction_relative_residual
        <= PARENT_BRANCH_H_RECON_REL_TOL,
        f"PARENT_BRANCH_H_RECON_GATE_FAILURE:{idx}:{role}:"
        f"{parent_branch_reconstruction_relative_residual}",
    )

    routing = compute_role_routing(
        xm[TARGET_LAG, :],
        xs[TARGET_LAG, :],
        hm[TARGET_LAG, :],
        hs[TARGET_LAG, :],
        operator,
        f"{idx}:{role}",
    )
    routing["parent_branch_reconstruction_relative_residual"] = (
        parent_branch_reconstruction_relative_residual
    )

    upath_key = (idx, role)
    require(upath_key in upath_rows, f"UPATH_ROW_MISSING:{upath_key}")
    upath = upath_rows[upath_key]
    for field, got in (
        ("delta_x_current_l2", routing["x_l2"]),
        ("delta_h_current_l2", routing["h_observed_l2"]),
    ):
        expected = upath[field]
        require(
            _isclose_parent(got, expected),
            f"UPATH_CURRENT_REPRODUCTION_FAILURE:{idx}:{role}:{field}:{got}:{expected}",
        )

    require(idx in parent_items, f"PARENT_ITEM_MISSING:{idx}")
    parent_item = parent_items[idx]
    parent_ps = parent_item[f"strong_energy_mass_{role}"]
    parent_pw = parent_item[f"weak_energy_mass_{role}"]
    require(
        _isclose_parent(routing["p_s_observed"], parent_ps),
        f"PARENT_P_S_REPRODUCTION_FAILURE:{idx}:{role}:"
        f"{routing['p_s_observed']}:{parent_ps}",
    )
    require(
        _isclose_parent(routing["p_w_observed"], parent_pw),
        f"PARENT_P_W_REPRODUCTION_FAILURE:{idx}:{role}:"
        f"{routing['p_w_observed']}:{parent_pw}",
    )

    return {
        "idx": idx,
        "stable_item_id": row["stable_item_id"],
        "role": role,
        **routing,
    }


def _safe_log_ratio(a, b):
    if a is None or b is None:
        return None
    a = float(a)
    b = float(b)
    if a <= 0.0 or b <= 0.0:
        return None
    return math.log(a / b)


def pair_role_records(
    corr: Mapping[str, Any],
    ctrl: Mapping[str, Any],
    operator: Mapping[str, Any],
):
    import torch

    require(corr["idx"] == ctrl["idx"], "PAIR_INDEX_MISMATCH")
    require(corr["stable_item_id"] == ctrl["stable_item_id"], "PAIR_STABLE_ID_MISMATCH")
    idx = int(corr["idx"])

    a_corr = corr["alignment"]
    a_ctrl = ctrl["alignment"]
    e_corr = corr["e"]
    e_ctrl = ctrl["e"]
    d = operator["row_r2"] * (a_corr - a_ctrl)

    direct_d = e_corr - e_ctrl
    channel_identity_residual = float(torch.max(torch.abs(d - direct_d)).item())
    require(channel_identity_residual <= ALGEBRAIC_ABS_TOL,
            f"PAIR_CHANNEL_D_IDENTITY_FAILURE:{idx}:{channel_identity_residual}")

    sum_all = float(torch.sum(d).item())
    sum_strong = float(torch.sum(d[operator["strong_mask"]]).item())
    sum_weak = float(torch.sum(d[operator["weak_mask"]]).item())
    sum_equal = float(torch.sum(d[operator["equal_mask"]]).item())

    delta_h_sq = float(corr["t_h_sq"]) - float(ctrl["t_h_sq"])
    delta_s_sq = float(corr["t_s_sq"]) - float(ctrl["t_s_sq"])
    delta_w_sq = float(corr["t_w_sq"]) - float(ctrl["t_w_sq"])

    closure_all = abs(sum_all - delta_h_sq)
    closure_strong = abs(sum_strong - delta_s_sq)
    closure_weak = abs(sum_weak - delta_w_sq)
    closure_partition = abs(sum_all - (sum_strong + sum_weak + sum_equal))

    for value, label in (
        (closure_all, "ALL"),
        (closure_strong, "STRONG"),
        (closure_weak, "WEAK"),
        (closure_partition, "PARTITION"),
    ):
        require(value <= ALGEBRAIC_ABS_TOL,
                f"PAIR_D_CLOSURE_FAILURE:{idx}:{label}:{value}")

    g_h = _safe_log_ratio(corr["t_h"], ctrl["t_h"])
    g_s = _safe_log_ratio(corr["t_s"], ctrl["t_s"])
    g_w = _safe_log_ratio(corr["t_w"], ctrl["t_w"])
    log_ps = _safe_log_ratio(corr["p_s"], ctrl["p_s"])
    log_pw = _safe_log_ratio(corr["p_w"], ctrl["p_w"])

    strong_enrichment_residual = None
    if log_ps is not None and g_s is not None and g_h is not None:
        strong_enrichment_residual = abs(log_ps - 2.0 * (g_s - g_h))
        require(
            strong_enrichment_residual <= ENRICHMENT_IDENTITY_ABS_TOL,
            f"STRONG_ENRICHMENT_IDENTITY_FAILURE:{idx}:{strong_enrichment_residual}",
        )

    weak_enrichment_residual = None
    if log_pw is not None and g_w is not None and g_h is not None:
        weak_enrichment_residual = abs(log_pw - 2.0 * (g_w - g_h))
        require(
            weak_enrichment_residual <= ENRICHMENT_IDENTITY_ABS_TOL,
            f"WEAK_ENRICHMENT_IDENTITY_FAILURE:{idx}:{weak_enrichment_residual}",
        )

    def pref(role_rec, key):
        return role_rec[key]

    row = {
        "schema_version": "k0-rvg-layer22-current-token-inproj-routing-item-v1",
        "local_template_index": idx,
        "stable_item_id": corr["stable_item_id"],
        "source_layer": SOURCE_LAYER,
        "relative_coordinate": TARGET_K,
        "delta_x_current_l2_corr": pref(corr, "x_l2"),
        "delta_x_current_l2_ctrl": pref(ctrl, "x_l2"),
        "delta_h_current_l2_corr": pref(corr, "h_observed_l2"),
        "delta_h_current_l2_ctrl": pref(ctrl, "h_observed_l2"),
        "delta_h_algebraic_l2_corr": pref(corr, "h_algebraic_l2"),
        "delta_h_algebraic_l2_ctrl": pref(ctrl, "h_algebraic_l2"),
        "t_h_corr": pref(corr, "t_h"),
        "t_h_ctrl": pref(ctrl, "t_h"),
        "t_s_corr": pref(corr, "t_s"),
        "t_s_ctrl": pref(ctrl, "t_s"),
        "t_w_corr": pref(corr, "t_w"),
        "t_w_ctrl": pref(ctrl, "t_w"),
        "p_s_corr": pref(corr, "p_s"),
        "p_s_ctrl": pref(ctrl, "p_s"),
        "p_w_corr": pref(corr, "p_w"),
        "p_w_ctrl": pref(ctrl, "p_w"),
        "parent_p_s_corr": pref(corr, "p_s_observed"),
        "parent_p_s_ctrl": pref(ctrl, "p_s_observed"),
        "parent_p_w_corr": pref(corr, "p_w_observed"),
        "parent_p_w_ctrl": pref(ctrl, "p_w_observed"),
        "delta_t_h_sq": delta_h_sq,
        "delta_t_s_sq": delta_s_sq,
        "delta_t_w_sq": delta_w_sq,
        "sum_d_all": sum_all,
        "sum_d_strong": sum_strong,
        "sum_d_weak": sum_weak,
        "g_h": g_h,
        "g_s": g_s,
        "g_w": g_w,
        "log_p_s_ratio": log_ps,
        "log_p_w_ratio": log_pw,
        "channel_d_identity_max_abs_residual": channel_identity_residual,
        "all_channel_d_closure_abs_residual": closure_all,
        "strong_channel_d_closure_abs_residual": closure_strong,
        "weak_channel_d_closure_abs_residual": closure_weak,
        "partition_d_closure_abs_residual": closure_partition,
        "strong_enrichment_identity_abs_residual": strong_enrichment_residual,
        "weak_enrichment_identity_abs_residual": weak_enrichment_residual,
        "parent_branch_reconstruction_relative_residual": max(
            float(corr["parent_branch_reconstruction_relative_residual"]),
            float(ctrl["parent_branch_reconstruction_relative_residual"]),
        ),
        "difference_reconstruction_relative_residual": max(
            float(corr["difference_reconstruction_relative_residual"]),
            float(ctrl["difference_reconstruction_relative_residual"]),
        ),
        "difference_reconstruction_absolute_residual": max(
            float(corr["difference_reconstruction_absolute_residual"]),
            float(ctrl["difference_reconstruction_absolute_residual"]),
        ),
        "error_difference_identity_abs_residual": max(
            float(corr["error_difference_identity_abs_residual"]),
            float(ctrl["error_difference_identity_abs_residual"]),
        ),
        "row_gain_alignment_factor_abs_residual": max(
            float(corr["row_gain_alignment_factor_abs_residual"]),
            float(ctrl["row_gain_alignment_factor_abs_residual"]),
        ),
        "alignment_formula_abs_residual": max(
            float(corr["alignment_formula_abs_residual"]),
            float(ctrl["alignment_formula_abs_residual"]),
        ),
        "strong_weak_total_closure_abs_residual": max(
            float(corr["strong_weak_total_closure_abs_residual"]),
            float(ctrl["strong_weak_total_closure_abs_residual"]),
        ),
        "energy_partition_closure_abs_residual": max(
            float(corr["energy_partition_closure_abs_residual"]),
            float(ctrl["energy_partition_closure_abs_residual"]),
        ),
        "p_s_algebraic_observed_bridge_abs_residual": max(
            float(corr["p_s_algebraic_observed_bridge_abs_residual"]),
            float(ctrl["p_s_algebraic_observed_bridge_abs_residual"]),
        ),
        "raw_vectors_persisted": False,
    }

    return row, a_corr, a_ctrl, e_corr, e_ctrl, d


def _subset_concentration(mean_d, mask):
    import torch

    values = mean_d[mask]
    m_abs = float(torch.sum(torch.abs(values)).item())
    m_net = float(torch.sum(values).item())
    sq = float(torch.sum(values * values).item())
    return {
        "channel_count": int(values.numel()),
        "absolute_mean_contribution_mass": m_abs,
        "net_mean_contribution": m_net,
        "signed_cancellation_ratio": (m_net / m_abs if m_abs > 0.0 else None),
        "effective_channel_count": ((m_abs * m_abs) / sq if sq > 0.0 else None),
        "positive_mean_contribution_channel_count": int(torch.sum(values > 0.0).item()),
        "negative_mean_contribution_channel_count": int(torch.sum(values < 0.0).item()),
        "zero_mean_contribution_channel_count": int(torch.sum(values == 0.0).item()),
    }


def build_channel_outputs(
    alignment_corr_matrix,
    alignment_ctrl_matrix,
    e_corr_matrix,
    e_ctrl_matrix,
    d_matrix,
    operator: Mapping[str, Any],
):
    import torch

    expected_shape = (EXPECTED_COMMON_COUNT, INTERMEDIATE_SIZE)
    for matrix, label in (
        (alignment_corr_matrix, "ALIGN_CORR"),
        (alignment_ctrl_matrix, "ALIGN_CTRL"),
        (e_corr_matrix, "E_CORR"),
        (e_ctrl_matrix, "E_CTRL"),
        (d_matrix, "D"),
    ):
        require(tuple(matrix.shape) == expected_shape, f"{label}_MATRIX_SHAPE_MISMATCH")

    mean_a_corr = torch.mean(alignment_corr_matrix, dim=0)
    mean_a_ctrl = torch.mean(alignment_ctrl_matrix, dim=0)
    mean_delta_a = mean_a_corr - mean_a_ctrl
    mean_e_corr = torch.mean(e_corr_matrix, dim=0)
    mean_e_ctrl = torch.mean(e_ctrl_matrix, dim=0)
    mean_d = torch.mean(d_matrix, dim=0)

    sorted_d, _ = torch.sort(d_matrix, dim=0)
    lower = sorted_d[(EXPECTED_COMMON_COUNT // 2) - 1, :]
    upper = sorted_d[EXPECTED_COMMON_COUNT // 2, :]
    median_d = (lower + upper) / 2.0

    pos = torch.sum(d_matrix > 0.0, dim=0)
    neg = torch.sum(d_matrix < 0.0, dim=0)
    zero = torch.sum(d_matrix == 0.0, dim=0)

    channel_rows = []
    rank_by_kernel = [0] * INTERMEDIATE_SIZE
    for rank, channel in enumerate(operator["kernel_order"], start=1):
        rank_by_kernel[int(channel)] = rank

    for j in range(INTERMEDIATE_SIZE):
        channel_rows.append(
            {
                "schema_version": "k0-rvg-layer22-current-token-inproj-routing-channel-v1",
                "channel_index": j,
                "kernel_magnitude_rank": int(rank_by_kernel[j]),
                "downstream_lag0_kernel_weight": float(operator["lag0"][j].item()),
                "downstream_lag0_kernel_weight_sq": float(operator["lag0_k2"][j].item()),
                "downstream_partition": (
                    "strong" if bool(operator["strong_mask"][j].item())
                    else "weak" if bool(operator["weak_mask"][j].item())
                    else "equal"
                ),
                "inproj_row_gain_sq": float(operator["row_r2"][j].item()),
                "mean_alignment_corr": float(mean_a_corr[j].item()),
                "mean_alignment_ctrl": float(mean_a_ctrl[j].item()),
                "mean_delta_alignment": float(mean_delta_a[j].item()),
                "mean_transfer_energy_corr": float(mean_e_corr[j].item()),
                "mean_transfer_energy_ctrl": float(mean_e_ctrl[j].item()),
                "mean_channel_contribution_d": float(mean_d[j].item()),
                "median_channel_contribution_d": float(median_d[j].item()),
                "channel_contribution_positive_count": int(pos[j].item()),
                "channel_contribution_negative_count": int(neg[j].item()),
                "channel_contribution_zero_count": int(zero[j].item()),
            }
        )

    cumulative_rows = []
    cum_d = cum_a_corr = cum_a_ctrl = cum_e_corr = cum_e_ctrl = cum_r2 = 0.0
    for rank, channel in enumerate(operator["kernel_order"], start=1):
        ch = int(channel)
        cum_d += float(mean_d[ch].item())
        cum_a_corr += float(mean_a_corr[ch].item())
        cum_a_ctrl += float(mean_a_ctrl[ch].item())
        cum_e_corr += float(mean_e_corr[ch].item())
        cum_e_ctrl += float(mean_e_ctrl[ch].item())
        cum_r2 += float(operator["row_r2"][ch].item())
        cumulative_rows.append(
            {
                "schema_version":
                    "k0-rvg-layer22-current-token-inproj-kernel-rank-cumulative-v1",
                "kernel_magnitude_rank": rank,
                "channel_index": ch,
                "cumulative_mean_d": cum_d,
                "cumulative_mean_alignment_corr": cum_a_corr,
                "cumulative_mean_alignment_ctrl": cum_a_ctrl,
                "cumulative_mean_transfer_energy_corr": cum_e_corr,
                "cumulative_mean_transfer_energy_ctrl": cum_e_ctrl,
                "cumulative_inproj_row_gain_sq": cum_r2,
            }
        )

    aggregate = {
        "all_channels": _subset_concentration(
            mean_d, torch.ones_like(operator["strong_mask"], dtype=torch.bool)
        ),
        "strong_channels": _subset_concentration(mean_d, operator["strong_mask"]),
        "weak_channels": _subset_concentration(mean_d, operator["weak_mask"]),
    }

    return channel_rows, cumulative_rows, aggregate


def structural_row_gain_summary(operator: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    r2 = operator["row_r2"]
    strong = [float(v) for v in r2[operator["strong_mask"]].tolist()]
    weak = [float(v) for v in r2[operator["weak_mask"]].tolist()]
    require(len(strong) == EXPECTED_STRONG_COUNT, "ROW_GAIN_STRONG_COUNT_MISMATCH")
    require(len(weak) == EXPECTED_WEAK_COUNT, "ROW_GAIN_WEAK_COUNT_MISMATCH")

    strong_mean = float(statistics.fmean(strong))
    weak_mean = float(statistics.fmean(weak))
    strong_median = float(statistics.median(strong))
    weak_median = float(statistics.median(weak))

    return {
        "strong": {
            "count": len(strong),
            "mean_row_gain_sq": strong_mean,
            "median_row_gain_sq": strong_median,
            "min_row_gain_sq": min(strong),
            "max_row_gain_sq": max(strong),
        },
        "weak": {
            "count": len(weak),
            "mean_row_gain_sq": weak_mean,
            "median_row_gain_sq": weak_median,
            "min_row_gain_sq": min(weak),
            "max_row_gain_sq": max(weak),
        },
        "strong_over_weak_mean_row_gain_sq_ratio": strong_mean / weak_mean,
        "strong_over_weak_median_row_gain_sq_ratio": strong_median / weak_median,
    }


def make_summary(
    item_rows: Sequence[Mapping[str, Any]],
    operator: Mapping[str, Any],
    concentration: Mapping[str, Any],
):
    require(len(item_rows) == EXPECTED_COMMON_COUNT, "ITEM_ROW_COUNT_MISMATCH")

    role_fields = ("t_h", "t_s", "t_w", "p_s", "p_w")
    role_summary = {"corr": {}, "ctrl": {}}
    for role in ("corr", "ctrl"):
        for field in role_fields:
            role_summary[role][field] = _aggregate_nullable(
                row[f"{field}_{role}"] for row in item_rows
            )

    paired_counts = {
        field: _paired_count(item_rows, f"{field}_corr", f"{field}_ctrl")
        for field in role_fields
    }

    delta_fields = (
        "delta_t_h_sq",
        "delta_t_s_sq",
        "delta_t_w_sq",
        "sum_d_all",
        "sum_d_strong",
        "sum_d_weak",
        "g_h",
        "g_s",
        "g_w",
        "log_p_s_ratio",
        "log_p_w_ratio",
    )
    delta_summary = {
        field: _aggregate_nullable(row[field] for row in item_rows)
        for field in delta_fields
    }

    aggregate_d_residual = abs(
        float(delta_summary["delta_t_h_sq"]["mean"])
        - float(concentration["all_channels"]["net_mean_contribution"])
    )
    aggregate_strong_residual = abs(
        float(delta_summary["delta_t_s_sq"]["mean"])
        - float(concentration["strong_channels"]["net_mean_contribution"])
    )
    aggregate_weak_residual = abs(
        float(delta_summary["delta_t_w_sq"]["mean"])
        - float(concentration["weak_channels"]["net_mean_contribution"])
    )
    for residual, label in (
        (aggregate_d_residual, "ALL"),
        (aggregate_strong_residual, "STRONG"),
        (aggregate_weak_residual, "WEAK"),
    ):
        require(residual <= ALGEBRAIC_ABS_TOL,
                f"AGGREGATE_D_CLOSURE_FAILURE:{label}:{residual}")

    return {
        "schema_version": "k0-rvg-layer22-current-token-inproj-routing-summary-v1",
        "scientific_question": QUESTION,
        "source_layer": SOURCE_LAYER,
        "relative_coordinate": TARGET_K,
        "common_ddsssss_item_count": EXPECTED_COMMON_COUNT,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "execution_protocol": EXECUTION_PROTOCOL,
        "lag0_kernel_rms": float(operator["rms"]),
        "strong_kernel_channel_count": int(operator["strong_mask"].sum().item()),
        "weak_kernel_channel_count": int(operator["weak_mask"].sum().item()),
        "equal_kernel_channel_count": int(operator["equal_mask"].sum().item()),
        "structural_row_gain_summary": structural_row_gain_summary(operator),
        "role_summary": role_summary,
        "paired_counts": paired_counts,
        "paired_delta_and_enrichment_summary": delta_summary,
        "channel_concentration": concentration,
        "max_parent_branch_reconstruction_relative_residual": max(
            float(row["parent_branch_reconstruction_relative_residual"])
            for row in item_rows
        ),
        "max_difference_reconstruction_relative_residual": max(
            float(row["difference_reconstruction_relative_residual"])
            for row in item_rows
        ),
        "max_difference_reconstruction_absolute_residual": max(
            float(row["difference_reconstruction_absolute_residual"])
            for row in item_rows
        ),
        "max_error_difference_identity_abs_residual": max(
            float(row["error_difference_identity_abs_residual"])
            for row in item_rows
        ),
        "max_row_gain_alignment_factor_abs_residual": max(
            float(row["row_gain_alignment_factor_abs_residual"]) for row in item_rows
        ),
        "max_alignment_formula_abs_residual": max(
            float(row["alignment_formula_abs_residual"]) for row in item_rows
        ),
        "max_all_channel_d_closure_abs_residual": max(
            float(row["all_channel_d_closure_abs_residual"]) for row in item_rows
        ),
        "max_strong_channel_d_closure_abs_residual": max(
            float(row["strong_channel_d_closure_abs_residual"]) for row in item_rows
        ),
        "max_weak_channel_d_closure_abs_residual": max(
            float(row["weak_channel_d_closure_abs_residual"]) for row in item_rows
        ),
        "max_strong_weak_total_closure_abs_residual": max(
            float(row["strong_weak_total_closure_abs_residual"]) for row in item_rows
        ),
        "max_energy_partition_closure_abs_residual": max(
            float(row["energy_partition_closure_abs_residual"]) for row in item_rows
        ),
        "max_p_s_algebraic_observed_bridge_abs_residual": max(
            float(row["p_s_algebraic_observed_bridge_abs_residual"]) for row in item_rows
        ),
        "max_strong_enrichment_identity_abs_residual": max(
            float(row["strong_enrichment_identity_abs_residual"])
            for row in item_rows
            if row["strong_enrichment_identity_abs_residual"] is not None
        ),
        "max_weak_enrichment_identity_abs_residual": max(
            float(row["weak_enrichment_identity_abs_residual"])
            for row in item_rows
            if row["weak_enrichment_identity_abs_residual"] is not None
        ),
        "aggregate_all_d_closure_abs_residual": aggregate_d_residual,
        "aggregate_strong_d_closure_abs_residual": aggregate_strong_residual,
        "aggregate_weak_d_closure_abs_residual": aggregate_weak_residual,
        "parent_current_token_metrics_match": True,
        "parent_strong_energy_mass_match": True,
        "raw_vectors_persisted": False,
        "raw_per_item_channel_vectors_persisted": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_whitening_or_learned_geometry_executed": False,
        "posthoc_layer_lag_channel_item_or_window_search_executed": False,
    }


def json_bytes(obj) -> bytes:
    return (
        json.dumps(
            obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows) -> bytes:
    return b"".join(json_bytes(row) for row in rows)


def _first_common_pair_rows(plan, cohort):
    by_idx: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in plan:
        idx = int(row["local_template_index"])
        if idx not in cohort:
            continue
        by_idx.setdefault(idx, {})[str(row["role"])] = row
    for idx in sorted(by_idx):
        pair = by_idx[idx]
        if "corr" in pair and "ctrl" in pair:
            return pair["corr"], pair["ctrl"]
    raise CurrentTokenInProjRoutingError("NO_COMMON_PREFLIGHT_PAIR")


def _run_role_capture(
    row,
    parent,
    four_tap_parent,
    u_path_parent,
    write_parent,
    carry_parent,
    recurrent_parent,
    base,
    model,
    binding,
    layer_map,
    mixer22,
    operator,
):
    matched_prefix, swapped_prefix = _prefixes(
        parent,
        four_tap_parent,
        u_path_parent,
        write_parent,
        carry_parent,
        recurrent_parent,
        row,
    )
    matched = capture(
        parent,
        four_tap_parent,
        u_path_parent,
        write_parent,
        base,
        model,
        binding,
        layer_map,
        mixer22,
        operator,
        matched_prefix,
        row["targets"],
    )
    swapped = capture(
        parent,
        four_tap_parent,
        u_path_parent,
        write_parent,
        base,
        model,
        binding,
        layer_map,
        mixer22,
        operator,
        swapped_prefix,
        row["targets"],
    )
    return matched, swapped


def runtime_preflight(
    root,
    parent,
    four_tap_parent,
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
    upath_rows,
    parent_items,
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
        mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
        four_tap_parent,
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
    operator = resolve_operator(parent, four_tap_parent, u_path_parent, mixer22)

    corr_row, ctrl_row = _first_common_pair_rows(plan, cohort)
    role_records = {}
    forward_count = 0
    for row in (corr_row, ctrl_row):
        matched, swapped = _run_role_capture(
            row,
            parent,
            four_tap_parent,
            u_path_parent,
            write_parent,
            carry_parent,
            recurrent_parent,
            base,
            model,
            binding,
            layer_map,
            mixer22,
            operator,
        )
        forward_count += 2
        rec = role_record_from_capture(
            row, matched, swapped, cohort, operator, upath_rows, parent_items
        )
        role_records[rec["role"]] = rec

    require(forward_count == EXPECTED_PREFLIGHT_FORWARD_COUNT,
            "PREFLIGHT_FORWARD_COUNT_MISMATCH")
    require(set(role_records) == {"corr", "ctrl"}, "PREFLIGHT_ROLE_SET_MISMATCH")

    item, *_ = pair_role_records(
        role_records["corr"], role_records["ctrl"], operator
    )

    structural = structural_row_gain_summary(operator)

    print("PASS_LAYER22_CURRENT_TOKEN_INPROJ_STRONG_ROUTING_RUNTIME_PREFLIGHT")
    print("local_template_index =", item["local_template_index"])
    print("model_forward_count =", forward_count)
    print("scientific_population_accessed = True")
    print("scientific_evidence_emitted = False")
    print("raw_vectors_persisted = False")
    print("lag0_kernel_rms =", operator["rms"])
    print("strong_kernel_channel_count =", int(operator["strong_mask"].sum().item()))
    print("weak_kernel_channel_count =", int(operator["weak_mask"].sum().item()))
    print("strong_mean_row_gain_sq =", structural["strong"]["mean_row_gain_sq"])
    print("weak_mean_row_gain_sq =", structural["weak"]["mean_row_gain_sq"])
    print("t_h_corr =", item["t_h_corr"])
    print("t_h_ctrl =", item["t_h_ctrl"])
    print("t_s_corr =", item["t_s_corr"])
    print("t_s_ctrl =", item["t_s_ctrl"])
    print("t_w_corr =", item["t_w_corr"])
    print("t_w_ctrl =", item["t_w_ctrl"])
    print("p_s_corr =", item["p_s_corr"])
    print("p_s_ctrl =", item["p_s_ctrl"])
    print("parent_p_s_corr =", item["parent_p_s_corr"])
    print("parent_p_s_ctrl =", item["parent_p_s_ctrl"])
    print("g_h =", item["g_h"])
    print("g_s =", item["g_s"])
    print("g_w =", item["g_w"])
    print("sum_d_all =", item["sum_d_all"])
    print("sum_d_strong =", item["sum_d_strong"])
    print("sum_d_weak =", item["sum_d_weak"])
    print(
        "parent_branch_reconstruction_relative_residual =",
        item["parent_branch_reconstruction_relative_residual"],
    )
    print(
        "difference_reconstruction_relative_residual_diagnostic =",
        item["difference_reconstruction_relative_residual"],
    )
    print(
        "error_difference_identity_abs_residual =",
        item["error_difference_identity_abs_residual"],
    )
    print("all_channel_d_closure_abs_residual =", item["all_channel_d_closure_abs_residual"])
    print("strong_channel_d_closure_abs_residual =",
          item["strong_channel_d_closure_abs_residual"])
    print("weak_channel_d_closure_abs_residual =",
          item["weak_channel_d_closure_abs_residual"])
    print("strong_enrichment_identity_abs_residual =",
          item["strong_enrichment_identity_abs_residual"])
    print("weak_enrichment_identity_abs_residual =",
          item["weak_enrichment_identity_abs_residual"])


def execute(
    root,
    parent,
    four_tap_parent,
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
    upath_rows,
    parent_items,
    handoff_path,
    output_dir,
):
    import torch

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
        mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
        four_tap_parent,
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
    operator = resolve_operator(parent, four_tap_parent, u_path_parent, mixer22)

    role_records: dict[tuple[int, str], dict[str, Any]] = {}
    forward_count = 0

    for n, row in enumerate(plan, start=1):
        idx = int(row["local_template_index"])
        matched, swapped = _run_role_capture(
            row,
            parent,
            four_tap_parent,
            u_path_parent,
            write_parent,
            carry_parent,
            recurrent_parent,
            base,
            model,
            binding,
            layer_map,
            mixer22,
            operator,
        )
        forward_count += 2

        if idx in cohort:
            rec = role_record_from_capture(
                row, matched, swapped, cohort, operator, upath_rows, parent_items
            )
            key = (idx, str(row["role"]))
            require(key not in role_records, f"ROLE_RECORD_DUPLICATE:{key}")
            role_records[key] = rec

        if n % 16 == 0 or n == len(plan):
            print(
                f"PROGRESS pair_roles={n}/{len(plan)} model_forwards={forward_count}",
                flush=True,
            )

    require(forward_count == EXPECTED_FULL_FORWARD_COUNT, "FORWARD_COUNT_MISMATCH")
    require(len(role_records) == EXPECTED_COMMON_COUNT * 2,
            "COMMON_ROLE_RECORD_COUNT_MISMATCH")

    item_rows = []
    a_corr_vectors = []
    a_ctrl_vectors = []
    e_corr_vectors = []
    e_ctrl_vectors = []
    d_vectors = []

    for idx in sorted(cohort):
        corr = role_records[(int(idx), "corr")]
        ctrl = role_records[(int(idx), "ctrl")]
        item, a_corr, a_ctrl, e_corr, e_ctrl, d = pair_role_records(
            corr, ctrl, operator
        )
        item_rows.append(item)
        a_corr_vectors.append(a_corr)
        a_ctrl_vectors.append(a_ctrl)
        e_corr_vectors.append(e_corr)
        e_ctrl_vectors.append(e_ctrl)
        d_vectors.append(d)

    a_corr_matrix = torch.stack(a_corr_vectors, dim=0)
    a_ctrl_matrix = torch.stack(a_ctrl_vectors, dim=0)
    e_corr_matrix = torch.stack(e_corr_vectors, dim=0)
    e_ctrl_matrix = torch.stack(e_ctrl_vectors, dim=0)
    d_matrix = torch.stack(d_vectors, dim=0)

    channel_rows, cumulative_rows, concentration = build_channel_outputs(
        a_corr_matrix,
        a_ctrl_matrix,
        e_corr_matrix,
        e_ctrl_matrix,
        d_matrix,
        operator,
    )
    summary = make_summary(item_rows, operator, concentration)

    partial_dir.mkdir(parents=True, exist_ok=False)
    item_path = partial_dir / "layer22_current_token_inproj_routing_item_metrics.jsonl"
    channel_path = partial_dir / "layer22_current_token_inproj_routing_channel_summary.jsonl"
    cumulative_path = partial_dir / "layer22_current_token_inproj_kernel_rank_cumulative.jsonl"
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "execution_manifest.json"

    item_path.write_bytes(jsonl_bytes(item_rows))
    channel_path.write_bytes(jsonl_bytes(channel_rows))
    cumulative_path.write_bytes(jsonl_bytes(cumulative_rows))
    summary_path.write_bytes(json_bytes(summary))

    manifest = {
        "schema_version":
            "k0-rvg-layer22-current-token-inproj-routing-execution-manifest-v1",
        "runtime_git_head": repo["head"],
        "runtime_branch": repo["branch"],
        "authority_freeze_commit": AUTHORITY_FREEZE_COMMIT,
        "authority_sha256": AUTHORITY_SHA256,
        "authority_blob": AUTHORITY_BLOB,
        "correction_authority_freeze_commit": CORRECTION_AUTHORITY_FREEZE_COMMIT,
        "correction_authority_sha256": CORRECTION_AUTHORITY_SHA256,
        "correction_authority_blob": CORRECTION_AUTHORITY_BLOB,
        "failed_implementation_commit": FAILED_IMPLEMENTATION_COMMIT,
        "parent_evidence_freeze_commit": PARENT_EVIDENCE_FREEZE_COMMIT,
        "parent_implementation_commit": PARENT_IMPLEMENTATION_COMMIT,
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "parent_runner_blob": PARENT_RUNNER_BLOB,
        "parent_item_sha256": PARENT_ITEM_SHA256,
        "parent_channel_sha256": PARENT_CHANNEL_SHA256,
        "parent_cumulative_sha256": PARENT_CUMULATIVE_SHA256,
        "parent_summary_sha256": PARENT_SUMMARY_SHA256,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "upath_evidence_freeze_commit": UPATH_EVIDENCE_FREEZE_COMMIT,
        "upath_metrics_sha256": UPATH_METRICS_SHA256,
        "upath_summary_sha256": UPATH_SUMMARY_SHA256,
        "upath_manifest_sha256": UPATH_MANIFEST_SHA256,
        "scientific_question": QUESTION,
        "execution_protocol": EXECUTION_PROTOCOL,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count": EXPECTED_COMMON_COUNT,
        "model_forward_count": forward_count,
        "source_layer": SOURCE_LAYER,
        "relative_coordinate": TARGET_K,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "lag0_kernel_rms": float(operator["rms"]),
        "strong_kernel_channel_count": int(operator["strong_mask"].sum().item()),
        "weak_kernel_channel_count": int(operator["weak_mask"].sum().item()),
        "equal_kernel_channel_count": int(operator["equal_mask"].sum().item()),
        "mamba_source_sha256": binding.source_sha256,
        "parent_scalar_rel_tol": PARENT_SCALAR_REL_TOL,
        "parent_scalar_abs_tol": PARENT_SCALAR_ABS_TOL,
        "parent_branch_h_reconstruction_rel_tol": PARENT_BRANCH_H_RECON_REL_TOL,
        "difference_reconstruction_relative_residual_is_diagnostic_only": True,
        "error_difference_identity_abs_tol": ALGEBRAIC_ABS_TOL,
        "algebraic_abs_tol": ALGEBRAIC_ABS_TOL,
        "energy_abs_tol": ENERGY_ABS_TOL,
        "parent_strong_energy_abs_tol": PARENT_STRONG_ENERGY_ABS_TOL,
        "enrichment_identity_abs_tol": ENRICHMENT_IDENTITY_ABS_TOL,
        "handoff_zip_sha256": handoff["zip_sha256"],
        "checkpoint_sha256": handoff["checkpoint_sha256"],
        "encoder_canonical_digest": encoder["canonical_digest"],
        "encoder_raw_concat_digest": encoder["raw_concat_digest"],
        "parent_current_token_metrics_match": True,
        "parent_strong_energy_mass_match": True,
        "capture_method": (
            "frozen layer-22 U-path capture with parent branch-level "
            "in-projection reconstruction gate preserved; current-token "
            "X_RF32[0,:] and H_RF32[0,:] branches are differenced in float64; "
            "authenticated bias-free hidden-half W_H defines h_alg=W_H*delta_X; "
            "runtime observed delta_H is bridged by the exact branch-error "
            "difference identity"
        ),
        "raw_vectors_persisted": False,
        "raw_per_item_channel_vectors_persisted": False,
        "scientific_model_forward_executed": True,
        "scientific_x_current_read": True,
        "scientific_h_current_read": True,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_whitening_or_learned_geometry_executed": False,
        "posthoc_layer_lag_channel_item_or_window_search_executed": False,
        "runner_rel": RUNNER_REL,
        "runner_sha256": sha256_bytes((root / RUNNER_REL).read_bytes()),
        "outputs": {
            item_path.name: sha256_bytes(item_path.read_bytes()),
            channel_path.name: sha256_bytes(channel_path.read_bytes()),
            cumulative_path.name: sha256_bytes(cumulative_path.read_bytes()),
            summary_path.name: sha256_bytes(summary_path.read_bytes()),
        },
    }
    manifest_path.write_bytes(json_bytes(manifest))

    del role_records
    del a_corr_matrix, a_ctrl_matrix, e_corr_matrix, e_ctrl_matrix, d_matrix
    del a_corr_vectors, a_ctrl_vectors, e_corr_vectors, e_ctrl_vectors, d_vectors

    os.replace(partial_dir, final_dir)

    rs = summary["role_summary"]
    ds = summary["paired_delta_and_enrichment_summary"]
    pc = summary["paired_counts"]
    structural = summary["structural_row_gain_summary"]
    conc = summary["channel_concentration"]

    print("PASS_LAYER22_CURRENT_TOKEN_INPROJ_STRONG_ROUTING_EXECUTION")
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print("parent_current_token_metrics_match = True")
    print("parent_strong_energy_mass_match = True")
    print("lag0_kernel_rms =", summary["lag0_kernel_rms"])
    print("strong_kernel_channel_count =", summary["strong_kernel_channel_count"])
    print("weak_kernel_channel_count =", summary["weak_kernel_channel_count"])
    print("strong_mean_row_gain_sq =", structural["strong"]["mean_row_gain_sq"])
    print("weak_mean_row_gain_sq =", structural["weak"]["mean_row_gain_sq"])
    print("strong_over_weak_mean_row_gain_sq_ratio =",
          structural["strong_over_weak_mean_row_gain_sq_ratio"])

    for role in ("corr", "ctrl"):
        print(
            f"{role}_current_token "
            f"T_H_median={rs[role]['t_h']['median']} "
            f"T_S_median={rs[role]['t_s']['median']} "
            f"T_W_median={rs[role]['t_w']['median']} "
            f"P_S_median={rs[role]['p_s']['median']} "
            f"P_S_mean={rs[role]['p_s']['mean']}"
        )

    print("g_h_median =", ds["g_h"]["median"])
    print("g_s_median =", ds["g_s"]["median"])
    print("g_w_median =", ds["g_w"]["median"])
    print("log_p_s_ratio_median =", ds["log_p_s_ratio"]["median"])
    print("log_p_w_ratio_median =", ds["log_p_w_ratio"]["median"])
    print("p_s_corr_gt_ctrl =", pc["p_s"]["corr_gt_ctrl"])
    print("t_h_corr_gt_ctrl =", pc["t_h"]["corr_gt_ctrl"])
    print("t_s_corr_gt_ctrl =", pc["t_s"]["corr_gt_ctrl"])
    print("t_w_corr_gt_ctrl =", pc["t_w"]["corr_gt_ctrl"])
    print("mean_delta_t_h_sq =", ds["delta_t_h_sq"]["mean"])
    print("mean_delta_t_s_sq =", ds["delta_t_s_sq"]["mean"])
    print("mean_delta_t_w_sq =", ds["delta_t_w_sq"]["mean"])

    for name in ("all_channels", "strong_channels", "weak_channels"):
        x = conc[name]
        print(
            f"{name}_concentration "
            f"M_abs={x['absolute_mean_contribution_mass']} "
            f"M_net={x['net_mean_contribution']} "
            f"R_cancel={x['signed_cancellation_ratio']} "
            f"N_eff={x['effective_channel_count']}"
        )

    print(
        "max_parent_branch_reconstruction_relative_residual =",
        summary["max_parent_branch_reconstruction_relative_residual"],
    )
    print(
        "max_difference_reconstruction_relative_residual_diagnostic =",
        summary["max_difference_reconstruction_relative_residual"],
    )
    print(
        "max_error_difference_identity_abs_residual =",
        summary["max_error_difference_identity_abs_residual"],
    )
    print("max_all_channel_d_closure_abs_residual =",
          summary["max_all_channel_d_closure_abs_residual"])
    print("max_strong_channel_d_closure_abs_residual =",
          summary["max_strong_channel_d_closure_abs_residual"])
    print("max_weak_channel_d_closure_abs_residual =",
          summary["max_weak_channel_d_closure_abs_residual"])
    print("max_p_s_algebraic_observed_bridge_abs_residual =",
          summary["max_p_s_algebraic_observed_bridge_abs_residual"])


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
    authenticate_authority(root)
    parent = load_parent(root)
    parent_summary, parent_manifest, parent_items = authenticate_parent_evidence(root)
    upath_summary, upath_manifest, upath_rows = authenticate_upath_evidence(root)

    (
        four_tap_parent,
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
    ) = build_plan(root, parent)

    source = secant.validate_frozen_source_semantics()
    require(
        source["source_sha256"] == output_parent.EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )
    _observer, observer_binding = carry_parent.authenticate_observer_static(
        root, base, output_parent.EXPECTED_MAMBA_SOURCE_SHA256
    )

    print("=== LAYER22 CURRENT-TOKEN IN-PROJECTION STRONG-ROUTING AUDIT PLAN ===")
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print("authority_freeze_commit =", AUTHORITY_FREEZE_COMMIT)
    print("authority_sha256 =", AUTHORITY_SHA256)
    print("authority_blob =", AUTHORITY_BLOB)
    print("correction_authority_freeze_commit =", CORRECTION_AUTHORITY_FREEZE_COMMIT)
    print("correction_authority_sha256 =", CORRECTION_AUTHORITY_SHA256)
    print("correction_authority_blob =", CORRECTION_AUTHORITY_BLOB)
    print("failed_implementation_commit =", FAILED_IMPLEMENTATION_COMMIT)
    print("parent_evidence_freeze_commit =", PARENT_EVIDENCE_FREEZE_COMMIT)
    print("parent_runner_sha256 =", PARENT_RUNNER_SHA256)
    print("parent_item_sha256 =", PARENT_ITEM_SHA256)
    print("parent_summary_sha256 =", PARENT_SUMMARY_SHA256)
    print("parent_manifest_sha256 =", PARENT_MANIFEST_SHA256)
    print("upath_evidence_freeze_commit =", UPATH_EVIDENCE_FREEZE_COMMIT)
    print("upath_metrics_sha256 =", UPATH_METRICS_SHA256)
    print("pair_role_count =", len(plan))
    print("common_ddsssss_item_count =", len(cohort))
    print("source_layer =", SOURCE_LAYER)
    print("relative_coordinate =", TARGET_K)
    print("hidden_size =", HIDDEN_SIZE)
    print("intermediate_size =", INTERMEDIATE_SIZE)
    print("scientific_question =", QUESTION)
    print("identity_h = W_H x")
    print("identity_e_j = row_gain_sq_j * cos_sq_j")
    print("identity_delta_t_h_sq = sum_j D_j")
    print("strong_kernel_channel_count =", EXPECTED_STRONG_COUNT)
    print("weak_kernel_channel_count =", EXPECTED_WEAK_COUNT)
    print("expected_lag0_kernel_rms =", EXPECTED_LAG0_KERNEL_RMS)
    print("parent_branch_h_reconstruction_rel_tol =", PARENT_BRANCH_H_RECON_REL_TOL)
    print("difference_reconstruction_relative_residual_gate = diagnostic_only")
    print("error_difference_identity_abs_tol =", ALGEBRAIC_ABS_TOL)
    print("algebraic_abs_tol =", ALGEBRAIC_ABS_TOL)
    print("parent_strong_energy_abs_tol =", PARENT_STRONG_ENERGY_ABS_TOL)
    print("observer_update_line =", observer_binding.update_line)
    print("observer_readout_line =", observer_binding.readout_line)
    print("mamba_source_sha256 =", source["source_sha256"])
    print("all_1536_channels_preregistered = True")
    print("raw_vectors_persisted = False")
    print("tokenizer_invoked = False")
    print("training_executed = False")
    print("causal_intervention_executed = False")
    print("learned_geometry_executed = False")
    print("posthoc_search_executed = False")

    if args.static_preflight:
        print("scientific_model_forward_executed = False")
        print("PASS_LAYER22_CURRENT_TOKEN_INPROJ_STRONG_ROUTING_STATIC_PREFLIGHT")
        return 0

    require(args.handoff is not None, "RUNTIME_MODE_REQUIRES_HANDOFF")
    handoff_path = args.handoff.resolve()
    require(handoff_path.is_file(), f"HANDOFF_MISSING:{handoff_path}")

    if args.runtime_preflight:
        runtime_preflight(
            root,
            parent,
            four_tap_parent,
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
            upath_rows,
            parent_items,
            handoff_path,
        )
        return 0

    require(args.output_dir is not None, "EXECUTE_REQUIRES_OUTPUT_DIR")
    execute(
        root,
        parent,
        four_tap_parent,
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
        upath_rows,
        parent_items,
        handoff_path,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CurrentTokenInProjRoutingError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"BLOCKED_UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
