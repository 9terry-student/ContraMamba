"""K0-RVG layer-22 exact four-tap causal-convolution decomposition audit.

Frozen scientific question:
    Within the fixed layer-22 H_RF -> C causal-convolution boundary at
    common-330 k=2, which lag-specific incoming differences, fixed
    channel-conditioned tap transfers, and constructive/destructive vector
    interactions account for the corr-selective amplification?

Exact native difference identity:
    Q_l(t) = K[:, 3-l] * delta_H_(t-l)
    delta_C_t = Q_0 + Q_1 + Q_2 + Q_3

Squared-norm identity:
    ||delta_C||^2 ~= sum_l ||Q_l||^2 + 2 sum_(l<m) <Q_l, Q_m>

The direct delta-C term is observed at the frozen parent layer-22 pre-SiLU
boundary. Four-tap reconstruction is checked before scientific metrics are
accepted. The analysis is observational/algebraic only: no tokenizer, logits,
task heads, training, intervention, learned geometry, or post-hoc search.
Raw vectors are never persisted.
"""

from __future__ import annotations

import argparse
import collections
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

AUTHORITY_FREEZE_COMMIT = "7e9e4f5b4ebf0676568822d3aa855278c8c1f192"
AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_four_tap_convolution_decomposition_static_design_candidate.md"
)
AUTHORITY_SHA256 = "781b08bc5f2e2d812f10d6bf8f464fc38bbc8a13f738ad791c6f9c7edb870d6d"
AUTHORITY_BLOB = "42c51bccbf1f650f429faa0daf43330593ceb5eb"

PARENT_EVIDENCE_FREEZE_COMMIT = "dabb9422dbf0e111319828cf027e8dc5d82fe326"
PARENT_IMPLEMENTATION_COMMIT = "5f08eefd82195aed12052c625891608b43fe2f29"
PARENT_RUNNER_REL = "scripts/longterm_k0_rvg_layer22_u_path_source_localization_audit.py"
PARENT_RUNNER_SHA256 = "1cfafd365fff2dbb10fb2d3ccd610874eb849cc3ddb53a4424e6a3ba21dd8528"
PARENT_RUNNER_BLOB = "2fb0b90eb3bd75fc246b9d5eb9b9b380f10ccf19"

PARENT_RUN_DIR = "reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1"
PARENT_METRICS_REL = PARENT_RUN_DIR + "/layer22_u_path_source_localization_metrics.jsonl"
PARENT_METRICS_SHA256 = "735f66534774a8d760f758ba17427af2de0f817340729f3619dddaae2464ef37"
PARENT_SUMMARY_REL = PARENT_RUN_DIR + "/summary.json"
PARENT_SUMMARY_SHA256 = "75c91fe3c0a047abfd0a74d37d12988dd7c34029c74768e7a86f56290995cb38"
PARENT_MANIFEST_REL = PARENT_RUN_DIR + "/execution_manifest.json"
PARENT_MANIFEST_SHA256 = "a91721745e008c6e15a00ec4fc938d2d9de5bb81034fd7888c451219bc8c2c70"
PARENT_REPORT_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_u_path_source_localization_validated_evidence_analysis_report_candidate.md"
)
PARENT_REPORT_SHA256 = "b110225009ed63d7cc1ad31c9c8dbf1217051833cf9c37f991c371be751beb24"
PARENT_REPORT_BLOB = "e370ad275a7645bc3a77d4e089cfe69447f4e49d"

FOUR_TAP_PRECEDENT_REL = "scripts/longterm_k0_rvg_four_tap_convolution_decomposition_audit.py"
FOUR_TAP_PRECEDENT_SHA256 = "49cd2708bf7ba3ef9f6beb5987f845d585cb9f991e459939c86cb303e0ce5070"
FOUR_TAP_PRECEDENT_BLOB = "83928f80349aab50f0c077fb4b2d1f4349a08d41"

RUNNER_REL = "scripts/longterm_k0_rvg_layer22_four_tap_convolution_decomposition_audit.py"
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
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 1536
STATE_SIZE = 16
CONV_KERNEL_SIZE = 4
RELATIVE_COORDINATES = tuple(range(-1, 7))
EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"

FOUR_TAP_RECON_REL_TOL = 2e-5
SQUARED_NORM_CLOSURE_REL_TOL = 1e-12
INTERACTION_NORMALIZATION_ABS_TOL = 1e-12
ADDITION_FACTOR_IDENTITY_ABS_TOL = 5e-5
ENRICHMENT_IDENTITY_ABS_TOL = 1e-12
PARENT_TRAJECTORY_REL_TOL = 1e-13
PARENT_TRAJECTORY_ABS_TOL = 1e-13

QUESTION = (
    "Within the fixed layer-22 H_RF -> C causal-convolution boundary at "
    "common-330 k=2, which lag-specific incoming differences, fixed "
    "channel-conditioned tap transfers, and constructive/destructive vector "
    "interactions account for the corr-selective amplification?"
)


class Layer22FourTapError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Layer22FourTapError(message)


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
        raise Layer22FourTapError(f"GIT_FAILURE:{' '.join(args)}") from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Layer22FourTapError(f"GIT_SHOW_FAILURE:{spec}") from exc


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
        (PARENT_EVIDENCE_FREEZE_COMMIT, "PARENT_EVIDENCE"),
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
            require(xy == "??", f"STATIC_RUNNER_STATE_UNEXPECTED:{line}")
            continue
        raise Layer22FourTapError(f"UNEXPECTED_WORKTREE_CHANGE:{line}")

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


def authenticate_authority(root: Path) -> None:
    frozen = git_bytes(root, f"{AUTHORITY_FREEZE_COMMIT}:{AUTHORITY_REL}")
    require(sha256_bytes(frozen) == AUTHORITY_SHA256, "AUTHORITY_SHA256_MISMATCH")
    require(
        git(root, "rev-parse", f"{AUTHORITY_FREEZE_COMMIT}:{AUTHORITY_REL}")
        == AUTHORITY_BLOB,
        "AUTHORITY_BLOB_MISMATCH",
    )
    current = root / AUTHORITY_REL
    require(current.is_file(), "AUTHORITY_FILE_MISSING")
    require(current.read_bytes() == frozen, "AUTHORITY_WORKTREE_DRIFT")


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
    return import_module(path, "k0_rvg_layer22_u_path_parent")


def authenticate_parent_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    specs = (
        (PARENT_METRICS_REL, PARENT_METRICS_SHA256, "METRICS"),
        (PARENT_SUMMARY_REL, PARENT_SUMMARY_SHA256, "SUMMARY"),
        (PARENT_MANIFEST_REL, PARENT_MANIFEST_SHA256, "MANIFEST"),
        (PARENT_REPORT_REL, PARENT_REPORT_SHA256, "REPORT"),
    )
    raws: dict[str, bytes] = {}
    for rel, expected_sha, label in specs:
        raw = git_bytes(root, f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{rel}")
        require(sha256_bytes(raw) == expected_sha, f"PARENT_{label}_SHA256_MISMATCH")
        current = root / rel
        require(current.is_file(), f"PARENT_{label}_MISSING")
        require(current.read_bytes() == raw, f"PARENT_{label}_WORKTREE_DRIFT")
        raws[label] = raw

    require(
        git(root, "rev-parse", f"{PARENT_EVIDENCE_FREEZE_COMMIT}:{PARENT_REPORT_REL}")
        == PARENT_REPORT_BLOB,
        "PARENT_REPORT_BLOB_MISMATCH",
    )

    summary = json.loads(raws["SUMMARY"])
    manifest = json.loads(raws["MANIFEST"])
    require(
        summary.get("schema_version") == "k0-rvg-layer22-u-path-source-localization-summary-v1",
        "PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("schema_version")
        == "k0-rvg-layer22-u-path-source-localization-execution-manifest-v1",
        "PARENT_MANIFEST_SCHEMA_MISMATCH",
    )
    require(summary.get("item_count") == EXPECTED_ITEM_COUNT, "PARENT_ITEM_COUNT_MISMATCH")
    require(summary.get("pair_role_count") == EXPECTED_PAIR_ROLE_COUNT, "PARENT_PAIR_ROLE_COUNT_MISMATCH")
    require(summary.get("common_ddsssss_item_count") == EXPECTED_COMMON_COUNT, "PARENT_COMMON_COUNT_MISMATCH")
    require(summary.get("source_layer") == SOURCE_LAYER, "PARENT_SOURCE_LAYER_MISMATCH")
    require(summary.get("parent_delta_w_trajectory_match") is True, "PARENT_DELTA_W_MATCH_NOT_TRUE")
    require(manifest.get("runtime_git_head") == PARENT_IMPLEMENTATION_COMMIT, "PARENT_RUNTIME_HEAD_MISMATCH")
    require(manifest.get("runner_sha256") == PARENT_RUNNER_SHA256, "PARENT_MANIFEST_RUNNER_SHA_MISMATCH")
    require(manifest.get("model_forward_count") == EXPECTED_FORWARD_COUNT, "PARENT_FORWARD_COUNT_MISMATCH")
    require(manifest.get("outputs", {}).get("layer22_u_path_source_localization_metrics.jsonl") == PARENT_METRICS_SHA256, "PARENT_MANIFEST_METRICS_HASH_MISMATCH")
    require(manifest.get("outputs", {}).get("summary.json") == PARENT_SUMMARY_SHA256, "PARENT_MANIFEST_SUMMARY_HASH_MISMATCH")
    return summary, manifest


def authenticate_precedent(root: Path) -> str:
    path = root / FOUR_TAP_PRECEDENT_REL
    require(path.is_file(), "FOUR_TAP_PRECEDENT_MISSING")
    frozen = git_bytes(root, f"{AUTHORITY_FREEZE_COMMIT}:{FOUR_TAP_PRECEDENT_REL}")
    require(sha256_bytes(frozen) == FOUR_TAP_PRECEDENT_SHA256, "FOUR_TAP_PRECEDENT_SHA256_MISMATCH")
    blob = git(root, "rev-parse", f"{AUTHORITY_FREEZE_COMMIT}:{FOUR_TAP_PRECEDENT_REL}")
    require(blob == FOUR_TAP_PRECEDENT_BLOB, "FOUR_TAP_PRECEDENT_BLOB_MISMATCH")
    require(git(root, "rev-parse", f"HEAD:{FOUR_TAP_PRECEDENT_REL}") == FOUR_TAP_PRECEDENT_BLOB, "FOUR_TAP_PRECEDENT_HEAD_BLOB_MISMATCH")
    require(path.read_bytes() == frozen, "FOUR_TAP_PRECEDENT_WORKTREE_DRIFT")
    return blob


def build_plan(root: Path, parent: Any):
    write_parent = parent.load_parent(root)
    values = parent.build_plan(root, write_parent)
    require(len(values) == 21, "PARENT_BUILD_PLAN_ARITY_MISMATCH")
    (
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


def resolve_kernel(parent: Any, mixer: Any):
    import torch

    w_hidden, kernel32, bias = parent.resolve_path_operators(mixer)
    require(tuple(kernel32.shape) == (INTERMEDIATE_SIZE, CONV_KERNEL_SIZE), "KERNEL_SHAPE_MISMATCH")
    kernel = kernel32.to(torch.float64).contiguous()
    weight_rms = []
    exact_zero = []
    for lag in range(CONV_KERNEL_SIZE):
        kernel_index = CONV_KERNEL_SIZE - 1 - lag
        w = kernel[:, kernel_index]
        rms = float(torch.sqrt(torch.mean(w * w)).item())
        zero = bool(torch.count_nonzero(w).item() == 0)
        require(math.isfinite(rms) and rms >= 0.0, f"KERNEL_RMS_INVALID:{lag}")
        require((rms == 0.0) == zero, f"KERNEL_ZERO_RMS_INCONSISTENT:{lag}")
        weight_rms.append(rms)
        exact_zero.append(zero)
    return w_hidden, kernel32, bias, kernel, tuple(weight_rms), tuple(exact_zero)


def _prefixes(parent: Any, write_parent: Any, carry_parent: Any, recurrent_parent: Any, row):
    matched, swapped = parent._prefixes(write_parent, carry_parent, recurrent_parent, row)
    return tuple(matched), tuple(swapped)


def capture(
    parent: Any,
    write_parent: Any,
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    mixer: Any,
    w_hidden: Any,
    kernel32: Any,
    bias: Any,
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    return parent.capture_path(
        write_parent,
        base,
        model,
        binding,
        layer_map,
        mixer,
        w_hidden,
        kernel32,
        bias,
        token_ids,
        targets,
    )


def _safe_ratio(numerator: float, denominator: float):
    if denominator > 0.0:
        return numerator / denominator
    return None


def metric_rows_for_pair(
    row: Mapping[str, Any],
    matched: Mapping[int, Mapping[str, Any]],
    swapped: Mapping[int, Mapping[str, Any]],
    cohort: frozenset[int],
    signature: str,
    kernel: Any,
    weight_rms: tuple[float, ...],
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
        hm32 = m["H_RF32"]
        hs32 = s["H_RF32"]
        cm32 = m["C32"]
        cs32 = s["C32"]

        require(tuple(hm32.shape) == (CONV_KERNEL_SIZE, INTERMEDIATE_SIZE), f"HM_RF_SHAPE_MISMATCH:{idx}:{role}:{k}")
        require(tuple(hs32.shape) == (CONV_KERNEL_SIZE, INTERMEDIATE_SIZE), f"HS_RF_SHAPE_MISMATCH:{idx}:{role}:{k}")
        require(tuple(cm32.shape) == (1, INTERMEDIATE_SIZE), f"CM_SHAPE_MISMATCH:{idx}:{role}:{k}")
        require(tuple(cs32.shape) == (1, INTERMEDIATE_SIZE), f"CS_SHAPE_MISMATCH:{idx}:{role}:{k}")

        h_exact_equal = bool(torch.equal(hm32, hs32))
        c_exact_equal = bool(torch.equal(cm32, cs32))

        hm = hm32.to(torch.float64)
        hs = hs32.to(torch.float64)
        cm = cm32.to(torch.float64)
        cs = cs32.to(torch.float64)
        dh = hm - hs
        dc = cm - cs

        delta_h_rf_l2 = float(torch.linalg.vector_norm(dh).item())
        delta_c_l2 = float(torch.linalg.vector_norm(dc).item())

        q = []
        dh_norms = []
        q_norms = []
        transfers = []
        normalized_transfers = []
        for lag in range(CONV_KERNEL_SIZE):
            kernel_index = CONV_KERNEL_SIZE - 1 - lag
            dhl = dh[lag, :]
            ql = dhl * kernel[:, kernel_index]
            dhn = float(torch.linalg.vector_norm(dhl).item())
            qn = float(torch.linalg.vector_norm(ql).item())
            transfer = _safe_ratio(qn, dhn)
            normalized = None
            if transfer is not None and weight_rms[lag] > 0.0:
                normalized = transfer / weight_rms[lag]
            q.append(ql)
            dh_norms.append(dhn)
            q_norms.append(qn)
            transfers.append(transfer)
            normalized_transfers.append(normalized)

        q_sum = q[0] + q[1] + q[2] + q[3]
        q_sum_l2 = float(torch.linalg.vector_norm(q_sum).item())
        recon_abs = float(torch.linalg.vector_norm(q_sum - dc.squeeze(0)).item())
        recon_rel = recon_abs / max(delta_c_l2, 1e-12)
        require(recon_rel <= FOUR_TAP_RECON_REL_TOL, f"FOUR_TAP_RECONSTRUCTION_FAILURE:{idx}:{role}:{k}:{recon_rel}")

        rss_sq = float(sum(v * v for v in q_norms))
        rss_l2 = math.sqrt(max(rss_sq, 0.0))
        energy_fractions = [None] * CONV_KERNEL_SIZE
        if rss_sq > 0.0:
            energy_fractions = [(v * v) / rss_sq for v in q_norms]

        cross_terms: dict[str, float] = {}
        interaction_terms: dict[str, float | None] = {}
        cross_total = 0.0
        for a in range(CONV_KERNEL_SIZE):
            for b in range(a + 1, CONV_KERNEL_SIZE):
                cross = float(2.0 * torch.dot(q[a], q[b]).item())
                cross_terms[f"cross_{a}{b}"] = cross
                interaction_terms[f"i_{a}{b}"] = cross / rss_sq if rss_sq > 0.0 else None
                cross_total += cross

        q_sum_sq = q_sum_l2 * q_sum_l2
        closure_abs = abs(q_sum_sq - (rss_sq + cross_total))
        closure_rel = closure_abs / max(q_sum_sq, rss_sq, 1.0)
        require(closure_rel <= SQUARED_NORM_CLOSURE_REL_TOL, f"SQUARED_NORM_CLOSURE_FAILURE:{idx}:{role}:{k}:{closure_rel}")

        i_total = cross_total / rss_sq if rss_sq > 0.0 else None
        interaction_normalization_abs_residual = 0.0
        if rss_sq > 0.0:
            sum_i = sum(float(v) for v in interaction_terms.values() if v is not None)
            interaction_normalization_abs_residual = abs(float(i_total) - sum_i)
            require(interaction_normalization_abs_residual <= INTERACTION_NORMALIZATION_ABS_TOL, f"INTERACTION_NORMALIZATION_FAILURE:{idx}:{role}:{k}:{interaction_normalization_abs_residual}")

        addition_factor = delta_c_l2 / rss_l2 if rss_l2 > 0.0 else None
        addition_factor_identity_abs_residual = 0.0
        if addition_factor is not None and i_total is not None:
            addition_factor_identity_abs_residual = abs(addition_factor * addition_factor - (1.0 + i_total))
            require(addition_factor_identity_abs_residual <= ADDITION_FACTOR_IDENTITY_ABS_TOL, f"ADDITION_FACTOR_IDENTITY_FAILURE:{idx}:{role}:{k}:{addition_factor_identity_abs_residual}")

        dominant_q_lag = int(max(range(CONV_KERNEL_SIZE), key=lambda lag: (q_norms[lag], -lag)))

        if k == -1:
            require(h_exact_equal and c_exact_equal, f"K_MINUS_1_EXACT_IDENTITY_FAILURE:{idx}:{role}")
            require(delta_h_rf_l2 == 0.0 and delta_c_l2 == 0.0 and rss_l2 == 0.0, f"K_MINUS_1_NONZERO_PARENT_OR_RSS:{idx}:{role}")
            require(all(v == 0.0 for v in dh_norms), f"K_MINUS_1_DELTA_H_LAG_NONZERO:{idx}:{role}")
            require(all(v == 0.0 for v in q_norms), f"K_MINUS_1_Q_NONZERO:{idx}:{role}")
            require(all(v is None for v in transfers), f"K_MINUS_1_TRANSFER_DEFINED:{idx}:{role}")
            require(all(v is None for v in normalized_transfers), f"K_MINUS_1_NORMALIZED_TRANSFER_DEFINED:{idx}:{role}")
            require(all(v is None for v in energy_fractions), f"K_MINUS_1_ENERGY_FRACTION_DEFINED:{idx}:{role}")
            require(i_total is None and addition_factor is None, f"K_MINUS_1_INTERACTION_OR_ADD_DEFINED:{idx}:{role}")

        values: dict[str, Any] = {
            "schema_version": "k0-rvg-layer22-four-tap-convolution-decomposition-row-v1",
            "local_template_index": idx,
            "stable_item_id": row["stable_item_id"],
            "role": role,
            "relative_coordinate": k,
            "token_index": token,
            "divergence_anchor_token_index": anchor,
            "token_equality_signature_k0_to_k6": signature,
            "in_common_ddsssss_cohort": idx in cohort,
            "h_rf_exact_equal": h_exact_equal,
            "c_exact_equal": c_exact_equal,
            "delta_h_rf_l2": delta_h_rf_l2,
            "delta_c_l2": delta_c_l2,
            "tap_rss_l2": rss_l2,
            "vector_addition_factor": addition_factor,
            "cross_total": cross_total,
            "i_total": i_total,
            "dominant_q_lag": dominant_q_lag,
            "four_tap_reconstruction_relative_residual": recon_rel,
            "squared_norm_closure_relative_residual": closure_rel,
            "interaction_normalization_abs_residual": interaction_normalization_abs_residual,
            "addition_factor_identity_abs_residual": addition_factor_identity_abs_residual,
            "parent_conv_reconstruction_relative_residual": max(
                float(m["conv_reconstruction_relative_residual"]),
                float(s["conv_reconstruction_relative_residual"]),
            ),
            "source_snapshot_dtype": "torch.float32",
            "metric_accumulation_dtype": "torch.float64",
        }
        for lag in range(CONV_KERNEL_SIZE):
            values[f"delta_h_lag{lag}_l2"] = dh_norms[lag]
            values[f"q{lag}_l2"] = q_norms[lag]
            values[f"q{lag}_tap_transfer"] = transfers[lag]
            values[f"q{lag}_transfer_over_weight_rms"] = normalized_transfers[lag]
            values[f"q{lag}_energy_fraction"] = energy_fractions[lag]
        values.update(cross_terms)
        values.update(interaction_terms)

        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                require(math.isfinite(float(value)), f"NONFINITE_METRIC:{idx}:{role}:{k}:{key}")
        output.append(values)

    return output


LAG_FIELDS = tuple(
    field
    for lag in range(CONV_KERNEL_SIZE)
    for field in (
        f"delta_h_lag{lag}_l2",
        f"q{lag}_l2",
        f"q{lag}_tap_transfer",
        f"q{lag}_transfer_over_weight_rms",
        f"q{lag}_energy_fraction",
    )
)
PAIR_NAMES = tuple(f"{a}{b}" for a in range(CONV_KERNEL_SIZE) for b in range(a + 1, CONV_KERNEL_SIZE))
CROSS_FIELDS = tuple(f"cross_{name}" for name in PAIR_NAMES)
INTERACTION_FIELDS = tuple(f"i_{name}" for name in PAIR_NAMES)
SUMMARY_FIELDS = (
    "delta_h_rf_l2",
    "delta_c_l2",
    "tap_rss_l2",
    "vector_addition_factor",
    "cross_total",
    "i_total",
    "four_tap_reconstruction_relative_residual",
    "squared_norm_closure_relative_residual",
    "interaction_normalization_abs_residual",
    "addition_factor_identity_abs_residual",
    "parent_conv_reconstruction_relative_residual",
) + LAG_FIELDS + CROSS_FIELDS + INTERACTION_FIELDS


def aggregate_nullable(values):
    vals = list(values)
    defined = [float(v) for v in vals if v is not None]
    require(all(math.isfinite(v) for v in defined), "NONFINITE_AGGREGATE")
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
                "mean": float(statistics.fmean(defined)),
                "median": float(statistics.median(defined)),
                "min": float(min(defined)),
                "max": float(max(defined)),
            }
        )
    return result


def aggregate_trajectory(rows):
    result = {"corr": {}, "ctrl": {}}
    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            bucket = [r for r in rows if r["role"] == role and int(r["relative_coordinate"]) == k]
            require(bool(bucket), f"EMPTY_BUCKET:{role}:{k}")
            result[role][str(k)] = {
                field: aggregate_nullable(r[field] for r in bucket) for field in SUMMARY_FIELDS
            }
    return result


def _aligned_common_k2(rows):
    corr = {
        int(r["local_template_index"]): r
        for r in rows
        if r["role"] == "corr" and int(r["relative_coordinate"]) == 2 and bool(r["in_common_ddsssss_cohort"])
    }
    ctrl = {
        int(r["local_template_index"]): r
        for r in rows
        if r["role"] == "ctrl" and int(r["relative_coordinate"]) == 2 and bool(r["in_common_ddsssss_cohort"])
    }
    require(set(corr) == set(ctrl), "COMMON_K2_ROLE_ALIGNMENT_MISMATCH")
    require(len(corr) == EXPECTED_COMMON_COUNT, "COMMON_K2_ALIGNMENT_COUNT_MISMATCH")
    return [(corr[i], ctrl[i]) for i in sorted(corr)]


def _paired_count_nullable(pairs, field: str):
    gt = lt = eq = undefined = 0
    for corr, ctrl in pairs:
        a, b = corr[field], ctrl[field]
        if a is None or b is None:
            undefined += 1
        elif float(a) > float(b):
            gt += 1
        elif float(a) < float(b):
            lt += 1
        else:
            eq += 1
    return {
        "count": len(pairs),
        "defined_pair_count": len(pairs) - undefined,
        "undefined_pair_count": undefined,
        "corr_gt_ctrl": gt,
        "corr_lt_ctrl": lt,
        "equal": eq,
    }


def _log_ratio(a, b):
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if a <= 0.0 or b <= 0.0:
        return None
    return math.log(a / b)


def _role_medians(pairs, fields: Sequence[str]):
    result = {"corr": {}, "ctrl": {}}
    for role_idx, role in ((0, "corr"), (1, "ctrl")):
        for field in fields:
            result[role][field] = aggregate_nullable(pair[role_idx][field] for pair in pairs)
    return result


def common_k2_analysis(pairs):
    lag_analysis = {}
    max_enrichment_residual = 0.0

    for lag in range(CONV_KERNEL_SIZE):
        eh_values = []
        eq_values = []
        g_values = []
        for corr, ctrl in pairs:
            eh = _log_ratio(corr[f"delta_h_lag{lag}_l2"], ctrl[f"delta_h_lag{lag}_l2"])
            eq = _log_ratio(corr[f"q{lag}_l2"], ctrl[f"q{lag}_l2"])
            g = None if eh is None or eq is None else eq - eh
            alt = _log_ratio(corr[f"q{lag}_tap_transfer"], ctrl[f"q{lag}_tap_transfer"])
            if g is not None and alt is not None:
                residual = abs(g - alt)
                max_enrichment_residual = max(max_enrichment_residual, residual)
                require(residual <= ENRICHMENT_IDENTITY_ABS_TOL, f"LAG_ENRICHMENT_IDENTITY_FAILURE:{lag}:{residual}")
            eh_values.append(eh)
            eq_values.append(eq)
            g_values.append(g)
        lag_analysis[str(lag)] = {
            "e_h": aggregate_nullable(eh_values),
            "e_q": aggregate_nullable(eq_values),
            "g_tap": aggregate_nullable(g_values),
        }

    e_rss_values = []
    e_c_values = []
    g_add_values = []
    for corr, ctrl in pairs:
        e_rss = _log_ratio(corr["tap_rss_l2"], ctrl["tap_rss_l2"])
        e_c = _log_ratio(corr["delta_c_l2"], ctrl["delta_c_l2"])
        g_add = None if e_rss is None or e_c is None else e_c - e_rss
        alt = _log_ratio(corr["vector_addition_factor"], ctrl["vector_addition_factor"])
        if g_add is not None and alt is not None:
            residual = abs(g_add - alt)
            max_enrichment_residual = max(max_enrichment_residual, residual)
            require(residual <= ENRICHMENT_IDENTITY_ABS_TOL, f"ADD_ENRICHMENT_IDENTITY_FAILURE:{residual}")
        e_rss_values.append(e_rss)
        e_c_values.append(e_c)
        g_add_values.append(g_add)

    interaction = {}
    for name in PAIR_NAMES:
        field = f"i_{name}"
        deltas = []
        for corr, ctrl in pairs:
            a, b = corr[field], ctrl[field]
            deltas.append(None if a is None or b is None else float(a) - float(b))
        interaction[name] = {
            "paired_count": _paired_count_nullable(pairs, field),
            "delta_i": aggregate_nullable(deltas),
        }

    delta_i_total = []
    for corr, ctrl in pairs:
        a, b = corr["i_total"], ctrl["i_total"]
        delta_i_total.append(None if a is None or b is None else float(a) - float(b))

    return {
        "lag_enrichment": lag_analysis,
        "e_rss": aggregate_nullable(e_rss_values),
        "e_c": aggregate_nullable(e_c_values),
        "g_add": aggregate_nullable(g_add_values),
        "pairwise_interaction": interaction,
        "i_total": {
            "paired_count": _paired_count_nullable(pairs, "i_total"),
            "delta_i_total": aggregate_nullable(delta_i_total),
        },
        "max_enrichment_identity_abs_residual": max_enrichment_residual,
    }


def _dominant_counts(rows):
    result = {}
    for role in ("corr", "ctrl"):
        bucket = [r for r in rows if r["role"] == role and int(r["relative_coordinate"]) == 2 and bool(r["in_common_ddsssss_cohort"])]
        counts = collections.Counter(int(r["dominant_q_lag"]) for r in bucket)
        result[role] = {str(lag): int(counts.get(lag, 0)) for lag in range(CONV_KERNEL_SIZE)}
        require(sum(result[role].values()) == EXPECTED_COMMON_COUNT, f"DOMINANT_COUNT_MISMATCH:{role}")
    return result


def make_summary(rows, cohort, weight_rms, exact_zero):
    require(len(rows) == EXPECTED_PAIR_ROLE_COUNT * len(RELATIVE_COORDINATES), "ROW_COUNT_MISMATCH")
    common_rows = [r for r in rows if bool(r["in_common_ddsssss_cohort"])]
    require(len(common_rows) == EXPECTED_COMMON_COUNT * 2 * len(RELATIVE_COORDINATES), "COMMON_ROW_COUNT_MISMATCH")

    km1 = [r for r in rows if int(r["relative_coordinate"]) == -1]
    require(len(km1) == EXPECTED_PAIR_ROLE_COUNT, "K_MINUS_1_ROW_COUNT_MISMATCH")
    require(
        all(
            r["h_rf_exact_equal"]
            and r["c_exact_equal"]
            and float(r["delta_h_rf_l2"]) == 0.0
            and float(r["delta_c_l2"]) == 0.0
            and float(r["tap_rss_l2"]) == 0.0
            and r["vector_addition_factor"] is None
            and r["i_total"] is None
            for r in km1
        ),
        "K_MINUS_1_SUMMARY_IDENTITY_FAILURE",
    )

    pairs = _aligned_common_k2(rows)
    paired_fields = ["delta_c_l2", "tap_rss_l2", "vector_addition_factor", "i_total"]
    for lag in range(CONV_KERNEL_SIZE):
        paired_fields.extend(
            [
                f"delta_h_lag{lag}_l2",
                f"q{lag}_l2",
                f"q{lag}_tap_transfer",
                f"q{lag}_transfer_over_weight_rms",
                f"q{lag}_energy_fraction",
            ]
        )
    paired_fields.extend(INTERACTION_FIELDS)

    analysis = common_k2_analysis(pairs)
    return {
        "schema_version": "k0-rvg-layer22-four-tap-convolution-decomposition-summary-v1",
        "scientific_question": QUESTION,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count": len(cohort),
        "trajectory_row_count": len(rows),
        "source_layer": SOURCE_LAYER,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "state_size": STATE_SIZE,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "relative_coordinates": list(RELATIVE_COORDINATES),
        "execution_protocol": EXECUTION_PROTOCOL,
        "causal_lag_to_kernel_index": {str(lag): CONV_KERNEL_SIZE - 1 - lag for lag in range(CONV_KERNEL_SIZE)},
        "kernel_weight_rms_by_causal_lag": {str(lag): float(weight_rms[lag]) for lag in range(CONV_KERNEL_SIZE)},
        "kernel_exact_zero_by_causal_lag": {str(lag): bool(exact_zero[lag]) for lag in range(CONV_KERNEL_SIZE)},
        "identity": "delta_C_t = sum_l K[:,3-l] elementwise delta_H_(t-l)",
        "undefined_ratio_policy": "no epsilon; JSON null with explicit undefined count",
        "full_336_trajectory": aggregate_trajectory(rows),
        "common_330_ddsssss_trajectory": aggregate_trajectory(common_rows),
        "common_330_k2_role_medians": _role_medians(pairs, paired_fields),
        "common_330_k2_role_counts": {field: _paired_count_nullable(pairs, field) for field in paired_fields},
        "common_330_k2_dominant_q_lag_counts": _dominant_counts(rows),
        "common_330_k2_analysis": analysis,
        "max_four_tap_reconstruction_relative_residual": max(float(r["four_tap_reconstruction_relative_residual"]) for r in rows),
        "max_squared_norm_closure_relative_residual": max(float(r["squared_norm_closure_relative_residual"]) for r in rows),
        "max_interaction_normalization_abs_residual": max(float(r["interaction_normalization_abs_residual"]) for r in rows),
        "max_addition_factor_identity_abs_residual": max(float(r["addition_factor_identity_abs_residual"]) for r in rows),
        "max_parent_conv_reconstruction_relative_residual": max(float(r["parent_conv_reconstruction_relative_residual"]) for r in rows),
        "parent_h_rf_trajectory_match": False,
        "parent_delta_c_trajectory_match": False,
        "raw_vectors_persisted": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_whitening_or_learned_geometry_executed": False,
        "posthoc_layer_lag_channel_item_or_window_search_executed": False,
    }


def validate_parent_reproduction(summary: dict[str, Any], parent_summary: Mapping[str, Any]) -> None:
    for trajectory in ("full_336_trajectory", "common_330_ddsssss_trajectory"):
        current = summary[trajectory]
        expected = parent_summary[trajectory]
        for role in ("corr", "ctrl"):
            for k in RELATIVE_COORDINATES:
                for field in ("delta_h_rf_l2", "delta_c_l2"):
                    got = current[role][str(k)][field]
                    exp = expected[role][str(k)][field]
                    require(int(got["defined_count"]) == int(exp["defined_count"]), f"PARENT_COUNT_MISMATCH:{trajectory}:{role}:{k}:{field}")
                    require(int(got["undefined_count"]) == int(exp["undefined_count"]), f"PARENT_UNDEFINED_COUNT_MISMATCH:{trajectory}:{role}:{k}:{field}")
                    for stat in ("mean", "median", "min", "max"):
                        gv, ev = got[stat], exp[stat]
                        if gv is None or ev is None:
                            require(gv is None and ev is None, f"PARENT_NONE_MISMATCH:{trajectory}:{role}:{k}:{field}:{stat}")
                        else:
                            require(
                                math.isclose(float(gv), float(ev), rel_tol=PARENT_TRAJECTORY_REL_TOL, abs_tol=PARENT_TRAJECTORY_ABS_TOL),
                                f"PARENT_TRAJECTORY_MISMATCH:{trajectory}:{role}:{k}:{field}:{stat}:{gv}:{ev}",
                            )
    summary["parent_h_rf_trajectory_match"] = True
    summary["parent_delta_c_trajectory_match"] = True


def json_bytes(obj):
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows):
    return b"".join(json_bytes(row) for row in rows)


def runtime_preflight(
    root,
    parent,
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
        mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
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
    w_hidden, kernel32, bias, kernel, weight_rms, exact_zero = resolve_kernel(parent, mixer22)

    row = plan[0]
    idx = int(row["local_template_index"])
    role = str(row["role"])
    matched_prefix, swapped_prefix = _prefixes(parent, write_parent, carry_parent, recurrent_parent, row)
    matched = capture(parent, write_parent, base, model, binding, layer_map, mixer22, w_hidden, kernel32, bias, matched_prefix, row["targets"])
    swapped = capture(parent, write_parent, base, model, binding, layer_map, mixer22, w_hidden, kernel32, bias, swapped_prefix, row["targets"])
    rows = metric_rows_for_pair(row, matched, swapped, cohort, signatures[(idx, role)], kernel, weight_rms)

    print("PASS_LAYER22_FOUR_TAP_CONVOLUTION_RUNTIME_PREFLIGHT")
    print("pair_role =", idx, role)
    print("model_forward_count = 2")
    print("scientific_population_accessed = True")
    print("scientific_evidence_emitted = False")
    print("raw_vectors_persisted = False")
    for lag in range(CONV_KERNEL_SIZE):
        print(f"kernel_lag{lag}_weight_rms =", weight_rms[lag])
        print(f"kernel_lag{lag}_exact_zero =", exact_zero[lag])
    print("max_four_tap_reconstruction_relative_residual =", max(float(r["four_tap_reconstruction_relative_residual"]) for r in rows))
    print("max_squared_norm_closure_relative_residual =", max(float(r["squared_norm_closure_relative_residual"]) for r in rows))
    print("max_interaction_normalization_abs_residual =", max(float(r["interaction_normalization_abs_residual"]) for r in rows))
    print("max_addition_factor_identity_abs_residual =", max(float(r["addition_factor_identity_abs_residual"]) for r in rows))
    print("max_parent_conv_reconstruction_relative_residual =", max(float(r["parent_conv_reconstruction_relative_residual"]) for r in rows))


def execute(
    root,
    parent,
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
    parent_summary,
    precedent_blob,
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
        mixer22,
        _out_proj,
    ) = resolve_runtime_bundle(
        root,
        parent,
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
    w_hidden, kernel32, bias, kernel, weight_rms, exact_zero = resolve_kernel(parent, mixer22)

    rows = []
    forward_count = 0
    for n, row in enumerate(plan, start=1):
        idx = int(row["local_template_index"])
        role = str(row["role"])
        matched_prefix, swapped_prefix = _prefixes(parent, write_parent, carry_parent, recurrent_parent, row)
        matched = capture(parent, write_parent, base, model, binding, layer_map, mixer22, w_hidden, kernel32, bias, matched_prefix, row["targets"])
        forward_count += 1
        swapped = capture(parent, write_parent, base, model, binding, layer_map, mixer22, w_hidden, kernel32, bias, swapped_prefix, row["targets"])
        forward_count += 1
        rows.extend(metric_rows_for_pair(row, matched, swapped, cohort, signatures[(idx, role)], kernel, weight_rms))
        if n % 16 == 0 or n == len(plan):
            print(f"PROGRESS pair_roles={n}/{len(plan)} model_forwards={forward_count}", flush=True)

    require(forward_count == EXPECTED_FORWARD_COUNT, "FORWARD_COUNT_MISMATCH")
    summary = make_summary(rows, cohort, weight_rms, exact_zero)
    validate_parent_reproduction(summary, parent_summary)

    require(summary["max_four_tap_reconstruction_relative_residual"] <= FOUR_TAP_RECON_REL_TOL, "SUMMARY_FOUR_TAP_RECON_TOLERANCE_FAILURE")
    require(summary["max_squared_norm_closure_relative_residual"] <= SQUARED_NORM_CLOSURE_REL_TOL, "SUMMARY_SQUARED_NORM_CLOSURE_FAILURE")
    require(summary["max_interaction_normalization_abs_residual"] <= INTERACTION_NORMALIZATION_ABS_TOL, "SUMMARY_INTERACTION_NORMALIZATION_FAILURE")
    require(summary["max_addition_factor_identity_abs_residual"] <= ADDITION_FACTOR_IDENTITY_ABS_TOL, "SUMMARY_ADDITION_FACTOR_IDENTITY_FAILURE")
    require(summary["common_330_k2_analysis"]["max_enrichment_identity_abs_residual"] <= ENRICHMENT_IDENTITY_ABS_TOL, "SUMMARY_ENRICHMENT_IDENTITY_FAILURE")

    partial_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = partial_dir / "layer22_four_tap_convolution_decomposition_metrics.jsonl"
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "execution_manifest.json"
    metrics_path.write_bytes(jsonl_bytes(rows))
    summary_path.write_bytes(json_bytes(summary))

    manifest = {
        "schema_version": "k0-rvg-layer22-four-tap-convolution-decomposition-execution-manifest-v1",
        "runtime_git_head": repo["head"],
        "runtime_branch": repo["branch"],
        "authority_freeze_commit": AUTHORITY_FREEZE_COMMIT,
        "authority_sha256": AUTHORITY_SHA256,
        "authority_blob": AUTHORITY_BLOB,
        "parent_evidence_freeze_commit": PARENT_EVIDENCE_FREEZE_COMMIT,
        "parent_implementation_commit": PARENT_IMPLEMENTATION_COMMIT,
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "parent_runner_blob": PARENT_RUNNER_BLOB,
        "parent_metrics_sha256": PARENT_METRICS_SHA256,
        "parent_summary_sha256": PARENT_SUMMARY_SHA256,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "parent_report_blob": PARENT_REPORT_BLOB,
        "four_tap_precedent_blob": precedent_blob,
        "scientific_question": QUESTION,
        "execution_protocol": EXECUTION_PROTOCOL,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count": EXPECTED_COMMON_COUNT,
        "model_forward_count": forward_count,
        "source_layer": SOURCE_LAYER,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "state_size": STATE_SIZE,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "causal_lag_to_kernel_index": {str(lag): CONV_KERNEL_SIZE - 1 - lag for lag in range(CONV_KERNEL_SIZE)},
        "kernel_weight_rms_by_causal_lag": {str(lag): float(weight_rms[lag]) for lag in range(CONV_KERNEL_SIZE)},
        "kernel_exact_zero_by_causal_lag": {str(lag): bool(exact_zero[lag]) for lag in range(CONV_KERNEL_SIZE)},
        "mamba_source_sha256": binding.source_sha256,
        "four_tap_reconstruction_rel_tol": FOUR_TAP_RECON_REL_TOL,
        "squared_norm_closure_rel_tol": SQUARED_NORM_CLOSURE_REL_TOL,
        "interaction_normalization_abs_tol": INTERACTION_NORMALIZATION_ABS_TOL,
        "addition_factor_identity_abs_tol": ADDITION_FACTOR_IDENTITY_ABS_TOL,
        "enrichment_identity_abs_tol": ENRICHMENT_IDENTITY_ABS_TOL,
        "parent_trajectory_rel_tol": PARENT_TRAJECTORY_REL_TOL,
        "parent_trajectory_abs_tol": PARENT_TRAJECTORY_ABS_TOL,
        "undefined_ratio_policy": "no epsilon; JSON null",
        "handoff_zip_sha256": handoff["zip_sha256"],
        "checkpoint_sha256": handoff["checkpoint_sha256"],
        "encoder_canonical_digest": encoder["canonical_digest"],
        "encoder_raw_concat_digest": encoder["raw_concat_digest"],
        "parent_h_rf_trajectory_match": True,
        "parent_delta_c_trajectory_match": True,
        "capture_method": (
            "frozen parent layer-22 U-path capture observes H_RF and direct pre-SiLU C in the same forward; "
            "the authenticated layer-22 depthwise kernel is applied algebraically in float64 to branch differences"
        ),
        "raw_vectors_persisted": False,
        "scientific_model_forward_executed": True,
        "scientific_h_rf_read": True,
        "scientific_c_read": True,
        "scientific_u_read_for_parent_bridge": True,
        "scientific_w_read_for_parent_bridge": True,
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
            "layer22_four_tap_convolution_decomposition_metrics.jsonl": sha256_bytes(metrics_path.read_bytes()),
            "summary.json": sha256_bytes(summary_path.read_bytes()),
        },
    }
    manifest_path.write_bytes(json_bytes(manifest))
    os.replace(partial_dir, final_dir)

    med = summary["common_330_k2_role_medians"]
    analysis = summary["common_330_k2_analysis"]
    counts = summary["common_330_k2_role_counts"]
    print("PASS_LAYER22_FOUR_TAP_CONVOLUTION_DECOMPOSITION_EXECUTION")
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print("parent_h_rf_trajectory_match = True")
    print("parent_delta_c_trajectory_match = True")
    for lag in range(CONV_KERNEL_SIZE):
        print(f"kernel_lag{lag}_weight_rms =", weight_rms[lag])
        print(f"kernel_lag{lag}_exact_zero =", exact_zero[lag])
    print("max_four_tap_reconstruction_relative_residual =", summary["max_four_tap_reconstruction_relative_residual"])
    print("max_squared_norm_closure_relative_residual =", summary["max_squared_norm_closure_relative_residual"])
    print("max_interaction_normalization_abs_residual =", summary["max_interaction_normalization_abs_residual"])
    print("max_addition_factor_identity_abs_residual =", summary["max_addition_factor_identity_abs_residual"])

    for role in ("corr", "ctrl"):
        print(
            f"{role}_k2_median "
            f"C={med[role]['delta_c_l2']['median']} "
            f"RSS={med[role]['tap_rss_l2']['median']} "
            f"ADD={med[role]['vector_addition_factor']['median']} "
            f"I_TOTAL={med[role]['i_total']['median']}"
        )
        for lag in range(CONV_KERNEL_SIZE):
            print(
                f"{role}_k2_lag{lag}_median "
                f"DH={med[role][f'delta_h_lag{lag}_l2']['median']} "
                f"Q={med[role][f'q{lag}_l2']['median']} "
                f"T={med[role][f'q{lag}_tap_transfer']['median']} "
                f"N={med[role][f'q{lag}_transfer_over_weight_rms']['median']} "
                f"F={med[role][f'q{lag}_energy_fraction']['median']}"
            )

    for lag in range(CONV_KERNEL_SIZE):
        x = analysis["lag_enrichment"][str(lag)]
        print(
            f"k2_lag{lag}_enrichment_medians "
            f"E_H={x['e_h']['median']} E_Q={x['e_q']['median']} G_TAP={x['g_tap']['median']}"
        )
    print(
        "k2_crossfree_addition_enrichment_medians "
        f"E_RSS={analysis['e_rss']['median']} "
        f"E_C={analysis['e_c']['median']} "
        f"G_ADD={analysis['g_add']['median']}"
    )
    for name in PAIR_NAMES:
        info = analysis["pairwise_interaction"][name]
        print(
            f"k2_interaction_{name} "
            f"corr_median={med['corr'][f'i_{name}']['median']} "
            f"ctrl_median={med['ctrl'][f'i_{name}']['median']} "
            f"delta_median={info['delta_i']['median']} "
            f"corr_gt_ctrl={info['paired_count']['corr_gt_ctrl']}"
        )
    print("k2_dominant_q_lag_counts =", summary["common_330_k2_dominant_q_lag_counts"])
    print("k2_delta_c_corr_gt_ctrl =", counts["delta_c_l2"]["corr_gt_ctrl"])
    print("k2_rss_corr_gt_ctrl =", counts["tap_rss_l2"]["corr_gt_ctrl"])
    print("k2_addition_factor_corr_gt_ctrl =", counts["vector_addition_factor"]["corr_gt_ctrl"])


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
    parent_summary, _parent_manifest = authenticate_parent_evidence(root)
    precedent_blob = authenticate_precedent(root)

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
    ) = build_plan(root, parent)

    source = secant.validate_frozen_source_semantics()
    require(source["source_sha256"] == output_parent.EXPECTED_MAMBA_SOURCE_SHA256, "MAMBA_SOURCE_SHA256_MISMATCH")
    _observer, observer_binding = carry_parent.authenticate_observer_static(
        root, base, output_parent.EXPECTED_MAMBA_SOURCE_SHA256
    )

    print("=== LAYER22 FOUR-TAP CONVOLUTION DECOMPOSITION AUDIT PLAN ===")
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print("authority_freeze_commit =", AUTHORITY_FREEZE_COMMIT)
    print("authority_sha256 =", AUTHORITY_SHA256)
    print("authority_blob =", AUTHORITY_BLOB)
    print("parent_evidence_freeze_commit =", PARENT_EVIDENCE_FREEZE_COMMIT)
    print("parent_runner_sha256 =", PARENT_RUNNER_SHA256)
    print("parent_metrics_sha256 =", PARENT_METRICS_SHA256)
    print("parent_summary_sha256 =", PARENT_SUMMARY_SHA256)
    print("parent_manifest_sha256 =", PARENT_MANIFEST_SHA256)
    print("parent_report_sha256 =", PARENT_REPORT_SHA256)
    print("four_tap_precedent_blob =", precedent_blob)
    print("pair_role_count =", len(plan))
    print("common_ddsssss_item_count =", len(cohort))
    print("source_layer =", SOURCE_LAYER)
    print("intermediate_size =", INTERMEDIATE_SIZE)
    print("conv_kernel_size =", CONV_KERNEL_SIZE)
    print("scientific_question =", QUESTION)
    print("identity = delta_C_t = sum_l K[:,3-l] elementwise delta_H_(t-l)")
    print("four_tap_reconstruction_rel_tol =", FOUR_TAP_RECON_REL_TOL)
    print("squared_norm_closure_rel_tol =", SQUARED_NORM_CLOSURE_REL_TOL)
    print("interaction_normalization_abs_tol =", INTERACTION_NORMALIZATION_ABS_TOL)
    print("addition_factor_identity_abs_tol =", ADDITION_FACTOR_IDENTITY_ABS_TOL)
    print("enrichment_identity_abs_tol =", ENRICHMENT_IDENTITY_ABS_TOL)
    print("observer_update_line =", observer_binding.update_line)
    print("observer_readout_line =", observer_binding.readout_line)
    print("mamba_source_sha256 =", source["source_sha256"])
    print("raw_vectors_persisted = False")
    print("tokenizer_invoked = False")
    print("training_executed = False")
    print("causal_intervention_executed = False")
    print("posthoc_layer_lag_channel_item_or_window_search_executed = False")

    if args.static_preflight:
        print("scientific_model_forward_executed = False")
        print("PASS_LAYER22_FOUR_TAP_CONVOLUTION_STATIC_PREFLIGHT")
        return 0

    require(args.handoff is not None, "RUNTIME_MODE_REQUIRES_HANDOFF")
    handoff_path = args.handoff.resolve()
    require(handoff_path.is_file(), f"HANDOFF_MISSING:{handoff_path}")

    if args.runtime_preflight:
        runtime_preflight(
            root,
            parent,
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
            signatures,
            handoff_path,
        )
        return 0

    require(args.output_dir is not None, "EXECUTE_REQUIRES_OUTPUT_DIR")
    execute(
        root,
        parent,
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
        parent_summary,
        precedent_blob,
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
    except Layer22FourTapError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"BLOCKED_UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
