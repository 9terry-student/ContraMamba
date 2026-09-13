#!/usr/bin/env python3
"""K0-RVG frozen-strong channel-mass breadth/concentration static analyzer.

Deterministic model-free analysis over the frozen strong-240 channel artifact.
No model, checkpoint, handoff, tokenizer, logits, training, intervention,
learned geometry, new channel/item selection, or K1 work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

AUTHORITY_FREEZE_COMMIT = "86287c07546024315d5644de64d9e8d3cd206189"
AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_frozen_strong_channel_mass_breadth_concentration_"
    "static_design_candidate.md"
)
AUTHORITY_SHA256 = (
    "015867d8fd034391248a4f51aeea01c7f1abd62fecfef5142d635273f71677f1"
)
AUTHORITY_BLOB = "f05209580e6c2885b020c91996dab79482f78589"

IMMEDIATE_PARENT_EVIDENCE_FREEZE = "236c15acba1badfcd19ef5bf2a9c236a28f4f399"
IMMEDIATE_PARENT_IMPL = "8df86fa8d0302d9e97ac31e41190b3a613c76896"
IMMEDIATE_PARENT_REPORT_REL = (
    "reports/"
    "longterm_k0_rvg_frozen_strong_sign_contribution_itemwise_breadth_"
    "validated_evidence_analysis_report_candidate.md"
)
IMMEDIATE_PARENT_REPORT_SHA256 = (
    "6de60ce27f7a896a129cbec12208da2d719974f8752cb0403e99debceb64ff2c"
)
IMMEDIATE_PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_frozen_strong_sign_contribution_itemwise_breadth_"
    "8df86fa_v1/summary.json"
)
IMMEDIATE_PARENT_SUMMARY_SHA256 = (
    "4cbe22ea0e2355632347102fc47f72205b734ed0e5c0008869805fdadfa776bb"
)

CHANNEL_EVIDENCE_FREEZE = "431e8faa6e5c82a20d87f532b4ab960fcf641ec2"
CHANNEL_REL = (
    "reports/"
    "longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/"
    "layer20_y20_strong_interaction_geometry_channel_validation.jsonl"
)
CHANNEL_SHA256 = (
    "111bf87720c5755ef974c4a98a8a12e62c6728952c72b0a76e436e385e6e9942"
)

RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_frozen_strong_channel_mass_breadth_concentration_"
    "static_analysis.py"
)

ALLOWED_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
    "validate_frozen_strong_alignment_sign_contribution_artifacts.py",
    "validate_frozen_strong_sign_contribution_itemwise_breadth_artifacts.py",
}

EXPECTED_CHANNEL_COUNT = 240
EXPECTED_ITEM_COUNT = 330
MAJORITY_BOUNDARY = 120
EXPECTED_TOTAL = 0.760594723140676
ABS_TOL = 5e-12
REL_TOL = 1e-13

CHANNEL_SCHEMA = "k0-rvg-layer20-y20-strong-interaction-geometry-channel-validation-v1"
IMMEDIATE_PARENT_SUMMARY_SCHEMA = (
    "k0-rvg-frozen-strong-sign-contribution-itemwise-breadth-summary-v1"
)
ITEM_SCHEMA = "k0-rvg-frozen-strong-channel-mass-breadth-metric-v1"
SUMMARY_SCHEMA = "k0-rvg-frozen-strong-channel-mass-breadth-summary-v1"
MANIFEST_SCHEMA = (
    "k0-rvg-frozen-strong-channel-mass-breadth-static-analysis-manifest-v1"
)

TOP_COUNTS = (3, 12, 24, 60, 120)

QUESTION = (
    "Across the fixed strong-240 channel population, is the population-level "
    "strong interaction contrast distributed broadly across channels, or "
    "concentrated into a comparatively small effective set of channels?"
)


class AnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AnalysisError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError(f"GIT_FAILURE:{' '.join(args)}") from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError(f"GIT_SHOW_FAILURE:{spec}") from exc


def status_path(line: str) -> str:
    raw = line[3:] if len(line) >= 4 else ""
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip('"').replace("\\", "/")


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=ABS_TOL)


def finite(value: Any, label: str) -> float:
    x = float(value)
    require(math.isfinite(x), f"NONFINITE:{label}")
    return x


def authenticate_repo(
    root: Path,
    execute_mode: bool,
    output_dir: Path | None,
) -> dict[str, Any]:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")

    for ancestor, label in (
        (AUTHORITY_FREEZE_COMMIT, "AUTHORITY"),
        (IMMEDIATE_PARENT_EVIDENCE_FREEZE, "IMMEDIATE_PARENT"),
        (CHANNEL_EVIDENCE_FREEZE, "CHANNEL_EVIDENCE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_NOT_ANCESTOR")

    output_prefix = None
    if output_dir is not None:
        try:
            output_prefix = (
                output_dir.resolve()
                .relative_to(root.resolve())
                .as_posix()
                .rstrip("/")
            )
        except ValueError:
            output_prefix = None

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
    ).splitlines()

    for line in status:
        path = status_path(line)
        xy = line[:2]
        if path in ALLOWED_UNTRACKED:
            require(xy == "??", f"ALLOWED_FILE_STATE_CHANGED:{line}")
            continue
        if not execute_mode and path == RUNNER_REL:
            require(xy == "??", f"ANALYZER_STATE_UNEXPECTED:{line}")
            continue
        if (
            execute_mode
            and output_prefix
            and path.startswith(output_prefix + "/")
        ):
            raise AnalysisError(f"OUTPUT_ALREADY_PRESENT:{line}")
        raise AnalysisError(f"UNEXPECTED_WORKTREE_CHANGE:{line}")

    runner = root / RUNNER_REL
    require(runner.is_file(), "ANALYZER_FILE_MISSING")

    if execute_mode:
        tracked = subprocess.call(
            ["git", "ls-files", "--error-unmatch", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(tracked == 0, "EXECUTE_REQUIRES_TRACKED_ANALYZER")
        clean = subprocess.call(
            ["git", "diff", "--quiet", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(clean == 0, "ANALYZER_WORKTREE_DRIFT")

    return {"branch": branch, "head": head, "status": status}


def authenticate_file(
    root: Path,
    commit: str,
    rel: str,
    expected_sha: str,
    label: str,
    expected_blob: str | None = None,
) -> bytes:
    raw = git_bytes(root, f"{commit}:{rel}")
    require(sha256_bytes(raw) == expected_sha, f"{label}_SHA256_MISMATCH")
    if expected_blob is not None:
        require(
            git(root, "rev-parse", f"{commit}:{rel}") == expected_blob,
            f"{label}_BLOB_MISMATCH",
        )
    current = root / rel
    require(current.is_file(), f"{label}_WORKTREE_FILE_MISSING")
    require(current.read_bytes() == raw, f"{label}_WORKTREE_DRIFT")
    return raw


def load_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise AnalysisError(f"{label}_JSON_PARSE_FAILURE") from exc
    require(isinstance(obj, dict), f"{label}_NOT_OBJECT")
    return obj


def load_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:
            raise AnalysisError(f"{label}_JSONL_PARSE_FAILURE:{line_no}") from exc
        require(isinstance(obj, dict), f"{label}_ROW_NOT_OBJECT:{line_no}")
        rows.append(obj)
    return rows


def authenticate_inputs(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    authenticate_file(
        root,
        AUTHORITY_FREEZE_COMMIT,
        AUTHORITY_REL,
        AUTHORITY_SHA256,
        "AUTHORITY",
        AUTHORITY_BLOB,
    )
    authenticate_file(
        root,
        IMMEDIATE_PARENT_EVIDENCE_FREEZE,
        IMMEDIATE_PARENT_REPORT_REL,
        IMMEDIATE_PARENT_REPORT_SHA256,
        "IMMEDIATE_PARENT_REPORT",
    )
    parent_summary_raw = authenticate_file(
        root,
        IMMEDIATE_PARENT_EVIDENCE_FREEZE,
        IMMEDIATE_PARENT_SUMMARY_REL,
        IMMEDIATE_PARENT_SUMMARY_SHA256,
        "IMMEDIATE_PARENT_SUMMARY",
    )
    channel_raw = authenticate_file(
        root,
        CHANNEL_EVIDENCE_FREEZE,
        CHANNEL_REL,
        CHANNEL_SHA256,
        "CHANNEL_INPUT",
    )
    return load_jsonl(channel_raw, "CHANNEL_INPUT"), load_json(
        parent_summary_raw, "IMMEDIATE_PARENT_SUMMARY"
    )


def validate_immediate_parent_summary(summary: Mapping[str, Any]) -> None:
    require(
        summary.get("schema_version") == IMMEDIATE_PARENT_SUMMARY_SCHEMA,
        "IMMEDIATE_PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    primary = summary["primary_two_axis_classification"]
    require(
        primary["combined_classification"]
        == "A1/B1_BROAD_POSITIVE_TOTAL_WITH_MAJORITY_PARENT_CONCORDANCE",
        "IMMEDIATE_PARENT_CLASSIFICATION_MISMATCH",
    )
    require(
        summary["common_ddsssss_item_count"] == EXPECTED_ITEM_COUNT,
        "IMMEDIATE_PARENT_ITEM_COUNT_MISMATCH",
    )
    require(
        summary["strong_kernel_channel_count"] == EXPECTED_CHANNEL_COUNT,
        "IMMEDIATE_PARENT_STRONG_COUNT_MISMATCH",
    )
    require(summary["model_forward_count"] == 0, "IMMEDIATE_PARENT_FORWARD_NONZERO")


def sign_label(x: float) -> str:
    if x > 0.0:
        return "positive"
    if x < 0.0:
        return "negative"
    return "zero"


def rank_quartile(rank: int) -> str:
    require(1 <= rank <= EXPECTED_CHANNEL_COUNT, f"RANK_OUT_OF_RANGE:{rank}")
    if rank <= 60:
        return "Q1_RANK_1_60"
    if rank <= 120:
        return "Q2_RANK_61_120"
    if rank <= 180:
        return "Q3_RANK_121_180"
    return "Q4_RANK_181_240"


def gini(values: Sequence[float]) -> float:
    xs = sorted(float(x) for x in values)
    require(xs, "GINI_EMPTY")
    require(all(x >= 0.0 for x in xs), "GINI_NEGATIVE_INPUT")
    total = sum(xs)
    require(total > 0.0, "GINI_ZERO_TOTAL")
    n = len(xs)
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    g = (2.0 * weighted) / (n * total) - (n + 1.0) / n
    require(-1e-15 <= g <= 1.0 + 1e-15, f"GINI_OUT_OF_RANGE:{g}")
    return min(1.0, max(0.0, g))


def min_count_for_share(values_desc: Sequence[float], total: float, target: float) -> int:
    require(total > 0.0, "MASS_TOTAL_NONPOSITIVE")
    running = 0.0
    for i, x in enumerate(values_desc, 1):
        running += float(x)
        if running / total >= target:
            return i
    raise AnalysisError(f"TARGET_SHARE_NOT_REACHED:{target}")


def top_share(values_desc: Sequence[float], total: float, k: int) -> float:
    require(0 < k <= len(values_desc), f"TOP_K_INVALID:{k}")
    require(total > 0.0, "TOP_SHARE_TOTAL_NONPOSITIVE")
    return float(sum(values_desc[:k]) / total)


def validate_channels(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(
        len(rows) == EXPECTED_CHANNEL_COUNT,
        f"CHANNEL_COUNT_MISMATCH:{len(rows)}",
    )

    channel_ids: set[int] = set()
    ranks: set[int] = set()

    parsed: list[dict[str, Any]] = []
    max_formula_res = 0.0
    max_parent_res = 0.0
    max_persisted_parent_res = 0.0
    corr_zero_total = 0
    ctrl_zero_total = 0

    for row in rows:
        require(row.get("schema_version") == CHANNEL_SCHEMA, "CHANNEL_SCHEMA_MISMATCH")
        require(row.get("downstream_partition") == "strong", "PARTITION_MISMATCH")

        channel_index = int(row["channel_index"])
        kernel_rank = int(row["kernel_magnitude_rank"])
        require(channel_index not in channel_ids, f"DUPLICATE_CHANNEL_INDEX:{channel_index}")
        require(kernel_rank not in ranks, f"DUPLICATE_KERNEL_RANK:{kernel_rank}")
        channel_ids.add(channel_index)
        ranks.add(kernel_rank)

        corr = finite(row["corr_mean_signed_product"], f"CORR:{channel_index}")
        ctrl = finite(row["ctrl_mean_signed_product"], f"CTRL:{channel_index}")
        parent = finite(row["parent_mean_d_ry20"], f"PARENT:{channel_index}")
        recon = finite(row["reconstructed_mean_d_ry20"], f"RECON:{channel_index}")

        formula_res = abs(recon - (corr - ctrl))
        parent_res = abs(parent - recon)
        persisted_res = finite(
            row["parent_reproduction_abs_residual"],
            f"PERSISTED_PARENT_RES:{channel_index}",
        )
        require(formula_res <= ABS_TOL, f"CHANNEL_FORMULA_FAILURE:{channel_index}:{formula_res}")
        require(parent_res <= ABS_TOL, f"CHANNEL_PARENT_BRIDGE_FAILURE:{channel_index}:{parent_res}")
        require(
            persisted_res <= ABS_TOL,
            f"PERSISTED_PARENT_RESIDUAL_FAILURE:{channel_index}:{persisted_res}",
        )

        counts = {}
        for role in ("corr", "ctrl"):
            pos = int(row[f"{role}_positive_product_count"])
            neg = int(row[f"{role}_negative_product_count"])
            zero = int(row[f"{role}_zero_product_count"])
            for value, name in ((pos, "pos"), (neg, "neg"), (zero, "zero")):
                require(
                    0 <= value <= EXPECTED_ITEM_COUNT,
                    f"COUNT_RANGE_FAILURE:{channel_index}:{role}:{name}:{value}",
                )
            require(
                pos + neg + zero == EXPECTED_ITEM_COUNT,
                f"ROLE_COUNT_CLOSURE_FAILURE:{channel_index}:{role}",
            )
            counts[role] = (pos, neg, zero)

        corr_pos, _, corr_zero = counts["corr"]
        ctrl_pos, _, ctrl_zero = counts["ctrl"]
        q_corr = corr_pos / EXPECTED_ITEM_COUNT
        q_ctrl = ctrl_pos / EXPECTED_ITEM_COUNT
        dq = q_corr - q_ctrl

        corr_zero_total += corr_zero
        ctrl_zero_total += ctrl_zero
        max_formula_res = max(max_formula_res, formula_res)
        max_parent_res = max(max_parent_res, parent_res)
        max_persisted_parent_res = max(max_persisted_parent_res, persisted_res)

        parsed.append(
            {
                "channel_index": channel_index,
                "kernel_magnitude_rank": kernel_rank,
                "d": parent,
                "corr_mean_signed_product": corr,
                "ctrl_mean_signed_product": ctrl,
                "q_corr": q_corr,
                "q_ctrl": q_ctrl,
                "delta_q": dq,
                "formula_residual": formula_res,
                "parent_bridge_residual": parent_res,
                "persisted_parent_residual": persisted_res,
            }
        )

    require(
        ranks == set(range(1, EXPECTED_CHANNEL_COUNT + 1)),
        "KERNEL_RANKS_NOT_EXACT_PERMUTATION_1_TO_240",
    )

    total = sum(r["d"] for r in parsed)
    require(close(total, EXPECTED_TOTAL), f"POPULATION_TOTAL_MISMATCH:{total}")

    return parsed, {
        "population_total": total,
        "population_total_abs_residual": abs(total - EXPECTED_TOTAL),
        "max_formula_abs_residual": max_formula_res,
        "max_parent_bridge_abs_residual": max_parent_res,
        "max_persisted_parent_residual": max_persisted_parent_res,
        "corr_zero_product_count_total": corr_zero_total,
        "ctrl_zero_product_count_total": ctrl_zero_total,
    }


def build_outputs(
    parsed: Sequence[Mapping[str, Any]],
    bridges: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = [float(r["d"]) for r in parsed]
    positive = [max(x, 0.0) for x in d]
    absolute = [abs(x) for x in d]

    p_ch = sum(positive)
    n_ch = sum(min(x, 0.0) for x in d)
    g_ch = p_ch + n_ch
    a_ch = sum(absolute)

    require(p_ch > 0.0, "POSITIVE_MASS_NONPOSITIVE")
    require(a_ch > 0.0, "ABSOLUTE_MASS_NONPOSITIVE")
    require(close(g_ch, EXPECTED_TOTAL), f"SIGNED_MASS_TOTAL_MISMATCH:{g_ch}")

    weights_pos = [x / p_ch for x in positive]
    weights_abs = [x / a_ch for x in absolute]
    hhi_pos = sum(w * w for w in weights_pos)
    hhi_abs = sum(w * w for w in weights_abs)
    require(hhi_pos > 0.0 and hhi_abs > 0.0, "HHI_NONPOSITIVE")
    n_eff_pos = 1.0 / hhi_pos
    n_eff_abs = 1.0 / hhi_abs
    b_eff_pos = n_eff_pos / EXPECTED_CHANNEL_COUNT
    b_eff_abs = n_eff_abs / EXPECTED_CHANNEL_COUNT

    n_pos = sum(x > 0.0 for x in d)
    n_neg = sum(x < 0.0 for x in d)
    n_zero = EXPECTED_CHANNEL_COUNT - n_pos - n_neg
    require(n_pos + n_neg + n_zero == EXPECTED_CHANNEL_COUNT, "SIGN_COUNT_CLOSURE_FAILURE")

    axis_s = (
        "S1_POSITIVE_CHANNEL_MAJORITY"
        if n_pos > MAJORITY_BOUNDARY
        else "S2_NOT_POSITIVE_CHANNEL_MAJORITY"
    )
    axis_p = (
        "P1_BROAD_POSITIVE_SUPPORTING_MASS"
        if n_eff_pos > MAJORITY_BOUNDARY
        else "P2_CONCENTRATED_POSITIVE_SUPPORTING_MASS"
    )
    axis_c = (
        "C1_BROAD_ABSOLUTE_CHANNEL_MASS"
        if n_eff_abs > MAJORITY_BOUNDARY
        else "C2_CONCENTRATED_ABSOLUTE_CHANNEL_MASS"
    )
    primary = f"{axis_s.split('_', 1)[0]}/{axis_p.split('_', 1)[0]}"

    pos_desc = sorted(positive, reverse=True)
    abs_desc = sorted(absolute, reverse=True)

    top_pos = {str(k): top_share(pos_desc, p_ch, k) for k in TOP_COUNTS}
    top_abs = {str(k): top_share(abs_desc, a_ch, k) for k in TOP_COUNTS}

    dq_vals = [float(r["delta_q"]) for r in parsed]
    dq_counts = {
        "positive": sum(x > 0.0 for x in dq_vals),
        "negative": sum(x < 0.0 for x in dq_vals),
        "zero": sum(x == 0.0 for x in dq_vals),
    }

    sign_dq_crosstab = Counter()
    for r in parsed:
        sign_dq_crosstab[f"d_{sign_label(float(r['d']))}__dq_{sign_label(float(r['delta_q']))}"] += 1

    quartiles: dict[str, dict[str, float | int]] = {}
    for label in (
        "Q1_RANK_1_60",
        "Q2_RANK_61_120",
        "Q3_RANK_121_180",
        "Q4_RANK_181_240",
    ):
        qrows = [r for r in parsed if rank_quartile(int(r["kernel_magnitude_rank"])) == label]
        require(len(qrows) == 60, f"RANK_QUARTILE_COUNT_MISMATCH:{label}:{len(qrows)}")
        qd = [float(r["d"]) for r in qrows]
        qpos = sum(max(x, 0.0) for x in qd)
        qneg = sum(min(x, 0.0) for x in qd)
        qabs = sum(abs(x) for x in qd)
        quartiles[label] = {
            "channel_count": 60,
            "signed_sum": sum(qd),
            "positive_mass": qpos,
            "negative_mass": qneg,
            "absolute_mass": qabs,
            "positive_mass_fraction": qpos / p_ch,
            "absolute_mass_fraction": qabs / a_ch,
        }

    pos_ranks = [int(r["kernel_magnitude_rank"]) for r in parsed if float(r["d"]) > 0.0]
    neg_ranks = [int(r["kernel_magnitude_rank"]) for r in parsed if float(r["d"]) < 0.0]

    channel_rows: list[dict[str, Any]] = []
    for r, p, a, wp, wa in zip(parsed, positive, absolute, weights_pos, weights_abs):
        channel_rows.append(
            {
                "schema_version": ITEM_SCHEMA,
                "channel_index": int(r["channel_index"]),
                "kernel_magnitude_rank": int(r["kernel_magnitude_rank"]),
                "rank_quartile": rank_quartile(int(r["kernel_magnitude_rank"])),
                "d_parent_mean_d_ry20": float(r["d"]),
                "d_sign": sign_label(float(r["d"])),
                "positive_support_mass": p,
                "absolute_mass": a,
                "normalized_positive_mass_weight": wp,
                "normalized_absolute_mass_weight": wa,
                "corr_positive_product_occupancy": float(r["q_corr"]),
                "ctrl_positive_product_occupancy": float(r["q_ctrl"]),
                "delta_positive_product_occupancy": float(r["delta_q"]),
                "occupancy_shift_sign": sign_label(float(r["delta_q"])),
                "formula_abs_residual": float(r["formula_residual"]),
                "parent_bridge_abs_residual": float(r["parent_bridge_residual"]),
                "persisted_parent_reproduction_abs_residual": float(r["persisted_parent_residual"]),
                "new_model_execution": False,
                "new_channel_selection": False,
                "new_item_selection": False,
            }
        )
    channel_rows.sort(key=lambda x: x["channel_index"])

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "scientific_question": QUESTION,
        "analysis_type": "MODEL_FREE_FROZEN_ARTIFACT_STATIC_ANALYSIS",
        "strong_kernel_channel_count": EXPECTED_CHANNEL_COUNT,
        "common_ddsssss_item_count": EXPECTED_ITEM_COUNT,
        "majority_boundary": MAJORITY_BOUNDARY,

        "primary_classification": {
            "axis_S_sign_breadth": axis_s,
            "axis_P_positive_mass_breadth": axis_p,
            "combined_SP_classification": primary,
            "supporting_axis_C_absolute_mass_breadth": axis_c,
        },

        "channel_sign_breadth": {
            "positive": n_pos,
            "negative": n_neg,
            "zero": n_zero,
            "positive_fraction": n_pos / EXPECTED_CHANNEL_COUNT,
        },

        "signed_channel_mass": {
            "positive_supporting_mass_P_ch": p_ch,
            "negative_opposing_mass_N_ch": n_ch,
            "net_mass_G_ch": g_ch,
            "opposition_ratio_abs_N_over_P": abs(n_ch) / p_ch,
        },

        "absolute_mass": {
            "total_absolute_mass_A_ch": a_ch,
            "HHI_abs": hhi_abs,
            "N_eff_abs": n_eff_abs,
            "B_eff_abs": b_eff_abs,
            "Gini_abs": gini(absolute),
            "K_abs_50": min_count_for_share(abs_desc, a_ch, 0.50),
            "K_abs_80": min_count_for_share(abs_desc, a_ch, 0.80),
            "fixed_top_count_cumulative_shares": top_abs,
        },

        "positive_support_mass": {
            "HHI_pos": hhi_pos,
            "N_eff_pos": n_eff_pos,
            "B_eff_pos": b_eff_pos,
            "Gini_pos": gini(positive),
            "K_pos_50": min_count_for_share(pos_desc, p_ch, 0.50),
            "K_pos_80": min_count_for_share(pos_desc, p_ch, 0.80),
            "fixed_top_count_cumulative_shares": top_pos,
        },

        "occupancy_direction": {
            "delta_q_sign_counts": dq_counts,
            "d_sign_x_delta_q_sign_crosstab": dict(sorted(sign_dq_crosstab.items())),
            "corr_zero_product_count_total": bridges["corr_zero_product_count_total"],
            "ctrl_zero_product_count_total": bridges["ctrl_zero_product_count_total"],
        },

        "kernel_rank_diagnostics": {
            "positive_d_kernel_rank_median":
                float(statistics.median(pos_ranks)) if pos_ranks else None,
            "negative_d_kernel_rank_median":
                float(statistics.median(neg_ranks)) if neg_ranks else None,
            "fixed_rank_quartiles": quartiles,
        },

        "population_bridge": {
            "expected_total": EXPECTED_TOTAL,
            "reconstructed_total": bridges["population_total"],
            "population_total_abs_residual": bridges["population_total_abs_residual"],
            "max_formula_abs_residual": bridges["max_formula_abs_residual"],
            "max_parent_bridge_abs_residual": bridges["max_parent_bridge_abs_residual"],
            "max_persisted_parent_residual": bridges["max_persisted_parent_residual"],
        },

        "new_model_execution": False,
        "model_forward_count": 0,
        "checkpoint_loaded": False,
        "handoff_opened": False,
        "transformers_imported": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_or_learned_geometry_executed": False,
        "posthoc_subset_search_executed": False,
        "new_item_selection": False,
        "new_channel_selection": False,
        "raw_vectors_read_or_persisted": False,
    }
    return channel_rows, summary


def json_bytes(obj: Any) -> bytes:
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


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(json_bytes(row) for row in rows)


def print_plan(repo: Mapping[str, Any]) -> None:
    print("=== FROZEN-STRONG CHANNEL-MASS BREADTH / CONCENTRATION PLAN ===")
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print("authority_freeze_commit =", AUTHORITY_FREEZE_COMMIT)
    print("authority_sha256 =", AUTHORITY_SHA256)
    print("authority_blob =", AUTHORITY_BLOB)
    print("immediate_parent_evidence_freeze =", IMMEDIATE_PARENT_EVIDENCE_FREEZE)
    print("immediate_parent_report_sha256 =", IMMEDIATE_PARENT_REPORT_SHA256)
    print("immediate_parent_summary_sha256 =", IMMEDIATE_PARENT_SUMMARY_SHA256)
    print("channel_evidence_freeze =", CHANNEL_EVIDENCE_FREEZE)
    print("channel_input_sha256 =", CHANNEL_SHA256)
    print("strong_kernel_channel_count =", EXPECTED_CHANNEL_COUNT)
    print("common_ddsssss_item_count =", EXPECTED_ITEM_COUNT)
    print("majority_boundary =", MAJORITY_BOUNDARY)
    print("expected_population_total =", EXPECTED_TOTAL)
    print("scientific_question =", QUESTION)
    print("model_forward_count = 0")
    print("new_model_execution = False")
    print("checkpoint_loaded = False")
    print("handoff_opened = False")
    print("transformers_imported = False")
    print("tokenizer_invoked = False")
    print("training_executed = False")
    print("causal_intervention_executed = False")
    print("posthoc_subset_search_executed = False")


def execute(
    root: Path,
    repo: Mapping[str, Any],
    channel_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: Path,
) -> None:
    final_dir = output_dir.resolve()
    partial_dir = Path(str(final_dir) + ".partial")
    require(not final_dir.exists(), f"OUTPUT_DIR_EXISTS:{final_dir}")
    require(not partial_dir.exists(), f"PARTIAL_OUTPUT_EXISTS:{partial_dir}")

    partial_dir.mkdir(parents=True, exist_ok=False)
    channel_path = partial_dir / "frozen_strong_channel_mass_breadth_metrics.jsonl"
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "static_analysis_manifest.json"

    channel_path.write_bytes(jsonl_bytes(channel_rows))
    summary_path.write_bytes(json_bytes(summary))

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_git_head": repo["head"],
        "analysis_branch": repo["branch"],
        "analyzer_rel": RUNNER_REL,
        "analyzer_sha256": sha256_file(root / RUNNER_REL),
        "authority_freeze_commit": AUTHORITY_FREEZE_COMMIT,
        "authority_sha256": AUTHORITY_SHA256,
        "authority_blob": AUTHORITY_BLOB,
        "immediate_parent_evidence_freeze": IMMEDIATE_PARENT_EVIDENCE_FREEZE,
        "immediate_parent_implementation": IMMEDIATE_PARENT_IMPL,
        "immediate_parent_report_sha256": IMMEDIATE_PARENT_REPORT_SHA256,
        "immediate_parent_summary_sha256": IMMEDIATE_PARENT_SUMMARY_SHA256,
        "channel_evidence_freeze": CHANNEL_EVIDENCE_FREEZE,
        "channel_input_sha256": CHANNEL_SHA256,
        "scientific_question": QUESTION,
        "analysis_type": "MODEL_FREE_FROZEN_ARTIFACT_STATIC_ANALYSIS",
        "strong_kernel_channel_count": EXPECTED_CHANNEL_COUNT,
        "common_ddsssss_item_count": EXPECTED_ITEM_COUNT,
        "majority_boundary": MAJORITY_BOUNDARY,
        "absolute_tolerance": ABS_TOL,
        "relative_tolerance": REL_TOL,
        "new_model_execution": False,
        "model_forward_count": 0,
        "checkpoint_loaded": False,
        "handoff_opened": False,
        "transformers_imported": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_or_learned_geometry_executed": False,
        "posthoc_subset_search_executed": False,
        "new_item_selection": False,
        "new_channel_selection": False,
        "raw_vectors_read_or_persisted": False,
        "outputs": {
            channel_path.name: sha256_file(channel_path),
            summary_path.name: sha256_file(summary_path),
        },
    }
    manifest_path.write_bytes(json_bytes(manifest))
    os.replace(partial_dir, final_dir)

    p = summary["primary_classification"]
    sb = summary["channel_sign_breadth"]
    sm = summary["signed_channel_mass"]
    am = summary["absolute_mass"]
    pm = summary["positive_support_mass"]

    print("PASS_FROZEN_STRONG_CHANNEL_MASS_BREADTH_CONCENTRATION_STATIC_ANALYSIS")
    print("output_dir =", final_dir)
    print("model_forward_count = 0")
    print("channel_count =", EXPECTED_CHANNEL_COUNT)
    print("N_pos =", sb["positive"])
    print("N_neg =", sb["negative"])
    print("N_zero =", sb["zero"])
    print("F_pos =", sb["positive_fraction"])
    print("axis_S =", p["axis_S_sign_breadth"])
    print("axis_P =", p["axis_P_positive_mass_breadth"])
    print("axis_C =", p["supporting_axis_C_absolute_mass_breadth"])
    print("combined_SP =", p["combined_SP_classification"])
    print("P_ch =", sm["positive_supporting_mass_P_ch"])
    print("N_ch =", sm["negative_opposing_mass_N_ch"])
    print("G_ch =", sm["net_mass_G_ch"])
    print("opposition_ratio =", sm["opposition_ratio_abs_N_over_P"])
    print("HHI_pos =", pm["HHI_pos"])
    print("N_eff_pos =", pm["N_eff_pos"])
    print("B_eff_pos =", pm["B_eff_pos"])
    print("Gini_pos =", pm["Gini_pos"])
    print("K_pos_50 =", pm["K_pos_50"])
    print("K_pos_80 =", pm["K_pos_80"])
    print("top_pos_shares =", pm["fixed_top_count_cumulative_shares"])
    print("HHI_abs =", am["HHI_abs"])
    print("N_eff_abs =", am["N_eff_abs"])
    print("B_eff_abs =", am["B_eff_abs"])
    print("Gini_abs =", am["Gini_abs"])
    print("K_abs_50 =", am["K_abs_50"])
    print("K_abs_80 =", am["K_abs_80"])
    print("top_abs_shares =", am["fixed_top_count_cumulative_shares"])
    print("occupancy_direction =", summary["occupancy_direction"])
    print("kernel_rank_diagnostics =", summary["kernel_rank_diagnostics"])
    print("population_bridge =", summary["population_bridge"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    p.add_argument("--output-dir", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    require(
        not args.static_preflight or args.output_dir is None,
        "STATIC_PREFLIGHT_OUTPUT_DIR_FORBIDDEN",
    )
    require(
        not args.execute or args.output_dir is not None,
        "EXECUTE_OUTPUT_DIR_REQUIRED",
    )

    repo = authenticate_repo(
        root,
        execute_mode=bool(args.execute),
        output_dir=args.output_dir,
    )
    channel_input, immediate_parent_summary = authenticate_inputs(root)
    validate_immediate_parent_summary(immediate_parent_summary)
    parsed, bridges = validate_channels(channel_input)
    channel_rows, summary = build_outputs(parsed, bridges)

    print_plan(repo)

    if args.static_preflight:
        print("scientific_evidence_emitted = False")
        print("population_total =", bridges["population_total"])
        print("population_total_abs_residual =", bridges["population_total_abs_residual"])
        print("channel_count =", len(channel_rows))
        print("rank_permutation_1_to_240 = True")
        print("PASS_FROZEN_STRONG_CHANNEL_MASS_BREADTH_CONCENTRATION_STATIC_PREFLIGHT")
        return 0

    execute(root, repo, channel_rows, summary, args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print("BLOCKED:", exc, file=sys.stderr)
        raise SystemExit(2)
