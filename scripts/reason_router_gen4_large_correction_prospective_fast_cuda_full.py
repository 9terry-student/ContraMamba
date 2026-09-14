from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen4_k_directional_alignment_transport_core as core
from scripts import reason_router_gen4_k_directional_alignment_transport_runner as parent
from scripts import reason_router_gen4_k_directional_alignment_transport_runtime as transport_runtime
from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as backend
from scripts import reason_router_gen4_large_correction_prospective_fast_cuda_one_pair_equivalence as eq
from scripts import reason_router_gen4_native_mamba_state_extraction as extraction
from scripts import reason_router_gen4_native_mamba_state_measurement as measurement


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = eq.EXPECTED_BRANCH
GATE_FREEZE_COMMIT = "34a958fbc52300f0807b211b1d97f68f6fea7339"
GATE_EXECUTION_HEAD = "8b4a83314b4a69d8004a68c7ae4a848f6e4ef401"
GATE_ARTIFACT_REL = Path(
    "reports/"
    "reason_router_gen4_large_correction_prospective_fast_cuda_one_pair_"
    "equivalence_8b4a833_retry3/equivalence_report.json"
)
GATE_ARTIFACT_SHA256 = (
    "34392044015cbc9217bbb90535e499be64c13f7ed068d566b9bec2e4a2448061"
)
DESIGN_REL = Path(
    "reports/reason_router_gen4_large_correction_prospective_validation_design.md"
)

SOURCE_PAIR_COUNT = 300
BASELINE_FORWARDS_PER_PAIR = 4
ALIGNMENT_FORWARDS_PER_PAIR = 2
FORWARDS_PER_PAIR = BASELINE_FORWARDS_PER_PAIR + ALIGNMENT_FORWARDS_PER_PAIR
BASELINE_FORWARD_BUDGET = SOURCE_PAIR_COUNT * BASELINE_FORWARDS_PER_PAIR
ALIGNMENT_FORWARD_BUDGET = SOURCE_PAIR_COUNT * ALIGNMENT_FORWARDS_PER_PAIR
FULL_FORWARD_BUDGET = SOURCE_PAIR_COUNT * FORWARDS_PER_PAIR

ALIGNMENT_SHIFT_THRESHOLD = 0.11228626366380845
MIN_GROUP_SIZE = 30
FAMILY_ALPHA = 0.05

REGIME_LARGE = "LARGE"
REGIME_SMALL = "SMALL"

REGIME_ITEM_SCHEMA = "gen4-k-large-correction-prospective-regime-item-v1"
REGIME_FREEZE_SCHEMA = "gen4-k-large-correction-prospective-regime-freeze-v1"
ITEM_SCHEMA = "gen4-k-large-correction-prospective-full-item-v1"
SUMMARY_SCHEMA = "gen4-k-large-correction-prospective-full-summary-v1"
MANIFEST_SCHEMA = "gen4-k-large-correction-prospective-full-manifest-v1"

REGIME_FILE = "regime_manifest.jsonl"
REGIME_FREEZE_FILE = "regime_freeze.json"
ITEM_FILE = "item_metrics.jsonl"
SUMMARY_FILE = "summary.json"
MANIFEST_FILE = "manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

FROZEN_FULL_DEPENDENCY_PATHS = tuple(dict.fromkeys((
    *eq.FROZEN_DEPENDENCY_PATHS,
    "scripts/reason_router_gen4_large_correction_prospective_fast_cuda_one_pair_equivalence.py",
    DESIGN_REL.as_posix(),
    GATE_ARTIFACT_REL.as_posix(),
)))

_RESPONSE_FIELDS = frozenset({
    "alignment_plus_path_efficiency",
    "alignment_minus_path_efficiency",
    "delta_alignment",
    "R_ALIGN",
})


class ProspectiveFullError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ProspectiveFullError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProspectiveFullError("GIT_FAILURE:" + " ".join(args)) from exc


def validate_gate_artifact() -> dict[str, Any]:
    path = ROOT / GATE_ARTIFACT_REL
    require(path.is_file(), "GATE_ARTIFACT_MISSING")
    require(sha256_file(path) == GATE_ARTIFACT_SHA256, "GATE_ARTIFACT_SHA256")
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    require(
        report.get("result") == "PASS_PROSPECTIVE_FAST_CUDA_ONE_PAIR_EQUIVALENCE",
        "GATE_RESULT_NOT_PASS",
    )
    require(report.get("execution_head") == GATE_EXECUTION_HEAD, "GATE_EXECUTION_HEAD")
    require(report.get("source_pair_id") == "generated_fact_301", "GATE_PAIR_ID")
    require(report.get("cpu_model_forward_count") == 6, "GATE_CPU_FORWARD_COUNT")
    require(report.get("gpu_model_forward_count") == 6, "GATE_GPU_FORWARD_COUNT")
    require(report.get("total_model_forward_count") == 12, "GATE_TOTAL_FORWARD_COUNT")
    require(report.get("scientific_budget_forward_count") == 0, "GATE_SCIENTIFIC_BUDGET")
    require(report.get("scientific_conclusion") is None, "GATE_SCIENTIFIC_CONCLUSION")
    require(
        report.get("representative_checkpoint_sha256")
        == extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
        "GATE_CHECKPOINT_IDENTITY",
    )
    require(
        float(report["max_state_abs_diff"])
        <= float(report["state_atol"]) + float(report["state_rtol"]),
        "GATE_STATE_TOLERANCE",
    )
    require(
        float(report["max_geometry_abs_diff"])
        <= float(report["geometry_atol"]) + float(report["geometry_rtol"]),
        "GATE_GEOMETRY_TOLERANCE",
    )
    require(
        float(report["max_pe_abs_diff"]) <= float(report["pe_atol"]),
        "GATE_PE_TOLERANCE",
    )
    return report


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", GATE_FREEZE_COMMIT, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "GATE_FREEZE_NOT_ANCESTOR")

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            GATE_FREEZE_COMMIT,
            expected_head,
            "--",
            *FROZEN_FULL_DEPENDENCY_PATHS,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "FROZEN_FULL_DEPENDENCY_DRIFT")
    validate_gate_artifact()


def classify_regime(alignment_shift_abs: float) -> str:
    value = float(alignment_shift_abs)
    require(math.isfinite(value) and value >= 0.0, "BAD_ALIGNMENT_SHIFT")
    return REGIME_LARGE if value >= ALIGNMENT_SHIFT_THRESHOLD else REGIME_SMALL


def _cells() -> dict[str, str]:
    return {
        "tp": core.TARGET_PLUS_CELL,
        "tm": core.TARGET_MINUS_CELL,
        "rp": core.REFERENCE_PLUS_CELL,
        "rm": core.REFERENCE_MINUS_CELL,
    }


def _anchors_for_pair(
    pair: str,
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, int]:
    cells = _cells()
    return {
        role: int(events[(pair, cell, core.ANCHOR_NAME)]["absolute_anchor_token_index"])
        for role, cell in cells.items()
    }


def _run_baseline_pair(
    pair: str,
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    budget: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cells = _cells()
    anchors = _anchors_for_pair(pair, events)

    baseline: dict[str, Any] = {}
    for role in ("tp", "tm", "rp", "rm"):
        baseline[role] = parent.capture_branch(
            model,
            runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            input_ids=eq._input_row(encoded, row_index, pair, cells[role]),
            anchor=anchors[role],
            budget=budget,
            capture_states=role in {"tp", "tm"},
        )

    target_geometry = core.reconstruct_pair_geometry(
        baseline["tp"]["geometry_branch"],
        baseline["tm"]["geometry_branch"],
        gamma=runtime_ctx["gamma"],
        w_hidden=runtime_ctx["w_hidden"],
        strong_mask=runtime_ctx["strong_mask"],
    )
    reference_geometry = core.reconstruct_pair_geometry(
        baseline["rp"]["geometry_branch"],
        baseline["rm"]["geometry_branch"],
        gamma=runtime_ctx["gamma"],
        w_hidden=runtime_ctx["w_hidden"],
        strong_mask=runtime_ctx["strong_mask"],
    )

    alignment_shift_abs = abs(
        float(reference_geometry["C"]) - float(target_geometry["C"])
    )
    require(math.isfinite(alignment_shift_abs), "NONFINITE_ALIGNMENT_SHIFT")

    alignment_norm_delta, align_core = core.alignment_delta(
        target_geometry["x"],
        target_geometry["y"],
        reference_geometry["C"],
    )
    alignment_delta_h = (
        target_geometry["d"] * alignment_norm_delta
    ).detach().to("cuda:0").contiguous().clone()

    baseline_plus = parent.path_efficiency(baseline["tp"])
    baseline_minus = parent.path_efficiency(baseline["tm"])
    delta_baseline = float(baseline_plus) - float(baseline_minus)
    require(math.isfinite(delta_baseline), "NONFINITE_BASELINE_DELTA")

    public = {
        "schema_version": REGIME_ITEM_SCHEMA,
        "source_pair_id": pair,
        "source_block": core.SOURCE_BLOCK,
        "target_residual_layer": core.TARGET_RESIDUAL_LAYER,
        "intervention_layer": core.INTERVENTION_LAYER,
        "relative_coordinate": core.TARGET_OFFSET,
        "target_plus_cell": core.TARGET_PLUS_CELL,
        "target_minus_cell": core.TARGET_MINUS_CELL,
        "reference_plus_cell": core.REFERENCE_PLUS_CELL,
        "reference_minus_cell": core.REFERENCE_MINUS_CELL,
        "anchor_name": core.ANCHOR_NAME,
        "target_plus_anchor": anchors["tp"],
        "target_minus_anchor": anchors["tm"],
        "reference_plus_anchor": anchors["rp"],
        "reference_minus_anchor": anchors["rm"],
        "target_plus_intervention_token": anchors["tp"] + core.TARGET_OFFSET,
        "target_minus_intervention_token": anchors["tm"] + core.TARGET_OFFSET,
        "target_A": float(target_geometry["A"]),
        "target_B": float(target_geometry["B"]),
        "target_C": float(target_geometry["C"]),
        "reference_A": float(reference_geometry["A"]),
        "reference_B": float(reference_geometry["B"]),
        "reference_C": float(reference_geometry["C"]),
        "alignment_shift_abs": alignment_shift_abs,
        "threshold": ALIGNMENT_SHIFT_THRESHOLD,
        "regime": classify_regime(alignment_shift_abs),
        "baseline_plus_path_efficiency": float(baseline_plus),
        "baseline_minus_path_efficiency": float(baseline_minus),
        "delta_baseline": delta_baseline,
        "classification_frozen_before_alignment": True,
    }
    require(not (_RESPONSE_FIELDS & set(public)), "BASELINE_RESPONSE_FIELD_LEAK")

    runtime_plan = {
        "cells": cells,
        "anchors": anchors,
        "alignment_delta_h": alignment_delta_h,
        "alignment_realized_A": float(align_core["realized_A"]),
        "alignment_realized_B": float(align_core["realized_B"]),
        "alignment_target_cosine": float(align_core["target_C"]),
        "alignment_realized_cosine": float(align_core["realized_C"]),
        "alignment_cosine_abs_residual": abs(
            float(align_core["realized_C"]) - float(align_core["target_C"])
        ),
        "alignment_A_preservation_abs_residual": float(
            align_core["A_preservation_abs_residual"]
        ),
        "alignment_B_preservation_abs_residual": float(
            align_core["B_preservation_abs_residual"]
        ),
    }
    return public, runtime_plan


def freeze_regimes(
    pairs: Sequence[str],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    require(len(pairs) == SOURCE_PAIR_COUNT, "FREEZE_PAIR_COUNT")
    require(len(baseline_rows) == SOURCE_PAIR_COUNT, "FREEZE_BASELINE_ROW_COUNT")
    frozen: list[dict[str, Any]] = []
    counts = {REGIME_LARGE: 0, REGIME_SMALL: 0}

    for pair, raw in zip(pairs, baseline_rows, strict=True):
        require(raw.get("source_pair_id") == pair, f"FREEZE_PAIR_ORDER:{pair}")
        require(not (_RESPONSE_FIELDS & set(raw)), f"FREEZE_RESPONSE_FIELD_LEAK:{pair}")
        shift = float(raw["alignment_shift_abs"])
        regime = classify_regime(shift)
        require(raw.get("regime") == regime, f"FREEZE_REGIME_DRIFT:{pair}")
        row = {
            "schema_version": REGIME_ITEM_SCHEMA,
            "source_pair_id": pair,
            "target_C": float(raw["target_C"]),
            "reference_C": float(raw["reference_C"]),
            "alignment_shift_abs": shift,
            "threshold": ALIGNMENT_SHIFT_THRESHOLD,
            "regime": regime,
            "classification_frozen_before_alignment": True,
        }
        frozen.append(row)
        counts[regime] += 1

    require(sum(counts.values()) == SOURCE_PAIR_COUNT, "FREEZE_COUNT_TOTAL")
    return frozen, counts


def group_size_passes(counts: Mapping[str, int]) -> bool:
    return (
        int(counts.get(REGIME_LARGE, 0)) >= MIN_GROUP_SIZE
        and int(counts.get(REGIME_SMALL, 0)) >= MIN_GROUP_SIZE
    )


def _write_bytes(path: Path, raw: bytes) -> str:
    require(not path.exists(), f"OUTPUT_COLLISION:{path.name}")
    path.write_bytes(raw)
    return sha256_bytes(raw)


def _write_regime_freeze(
    output_dir: Path,
    regime_rows: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")
    output_dir.mkdir(parents=True, exist_ok=False)

    regime_raw = jsonl_bytes(regime_rows)
    regime_sha = _write_bytes(output_dir / REGIME_FILE, regime_raw)
    freeze = {
        "schema_version": REGIME_FREEZE_SCHEMA,
        "threshold": ALIGNMENT_SHIFT_THRESHOLD,
        "threshold_source": "frozen_discovery_q75",
        "source_pair_count": SOURCE_PAIR_COUNT,
        "n_LARGE": int(counts[REGIME_LARGE]),
        "n_SMALL": int(counts[REGIME_SMALL]),
        "minimum_group_size": MIN_GROUP_SIZE,
        "group_size_gate_pass": group_size_passes(counts),
        "baseline_model_forward_count": BASELINE_FORWARD_BUDGET,
        "alignment_model_forward_count_at_freeze": 0,
        "response_fields_observed_at_freeze": False,
        "regime_manifest_sha256": regime_sha,
    }
    freeze_sha = _write_bytes(
        output_dir / REGIME_FREEZE_FILE,
        canonical_json_bytes(freeze),
    )
    return {
        "regime_manifest_sha256": regime_sha,
        "regime_freeze_sha256": freeze_sha,
        "regime_freeze": freeze,
    }


def _run_alignment_pair(
    pair: str,
    *,
    baseline_row: Mapping[str, Any],
    runtime_plan: Mapping[str, Any],
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    budget: Any,
) -> dict[str, Any]:
    cells = runtime_plan["cells"]
    anchors = runtime_plan["anchors"]
    delta_h = runtime_plan["alignment_delta_h"]

    alignment: dict[str, Any] = {}
    for role, plus_branch in (("tp", True), ("tm", False)):
        alignment[role] = parent.capture_branch(
            model,
            runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            input_ids=eq._input_row(encoded, row_index, pair, cells[role]),
            anchor=int(anchors[role]),
            budget=budget,
            capture_states=True,
            delta_h=delta_h,
            plus_branch=plus_branch,
        )

    audit = transport_runtime.paired_intervention_audit(
        alignment["tp"]["intervention_audit"],
        alignment["tm"]["intervention_audit"],
        delta_h,
        plus_expected_token_index=int(anchors["tp"]) + core.TARGET_OFFSET,
        minus_expected_token_index=int(anchors["tm"]) + core.TARGET_OFFSET,
    )

    alignment_plus = parent.path_efficiency(alignment["tp"])
    alignment_minus = parent.path_efficiency(alignment["tm"])
    delta_alignment = float(alignment_plus) - float(alignment_minus)
    r_align = delta_alignment - float(baseline_row["delta_baseline"])
    require(
        all(math.isfinite(v) for v in (alignment_plus, alignment_minus, delta_alignment, r_align)),
        "NONFINITE_ALIGNMENT_RESPONSE",
    )

    return {
        "schema_version": ITEM_SCHEMA,
        **{k: v for k, v in baseline_row.items() if k != "schema_version"},
        "alignment_realized_A": float(runtime_plan["alignment_realized_A"]),
        "alignment_realized_B": float(runtime_plan["alignment_realized_B"]),
        "alignment_target_cosine": float(runtime_plan["alignment_target_cosine"]),
        "alignment_realized_cosine": float(runtime_plan["alignment_realized_cosine"]),
        "alignment_cosine_abs_residual": float(
            runtime_plan["alignment_cosine_abs_residual"]
        ),
        "alignment_A_preservation_abs_residual": float(
            runtime_plan["alignment_A_preservation_abs_residual"]
        ),
        "alignment_B_preservation_abs_residual": float(
            runtime_plan["alignment_B_preservation_abs_residual"]
        ),
        "alignment_midpoint_max_abs_residual": float(
            audit["midpoint_max_abs_residual"]
        ),
        "alignment_pair_delta_max_abs_residual": float(
            audit["pair_delta_max_abs_residual"]
        ),
        "alignment_applied_correction_max_abs_residual": float(
            audit["applied_correction_max_abs_residual"]
        ),
        "alignment_plus_path_efficiency": float(alignment_plus),
        "alignment_minus_path_efficiency": float(alignment_minus),
        "delta_alignment": delta_alignment,
        "R_ALIGN": r_align,
    }


def _regularized_beta(x: float, a: float, b: float) -> float:
    require(0.0 <= x <= 1.0, "BETA_X_RANGE")
    require(a > 0.0 and b > 0.0, "BETA_SHAPE")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    max_iter = 300
    eps = 3.0e-14
    fpmin = 1.0e-300

    def betacf(aa: float, bb: float, xx: float) -> float:
        qab = aa + bb
        qap = aa + 1.0
        qam = aa - 1.0
        c = 1.0
        d = 1.0 - qab * xx / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            coeff = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + coeff * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + coeff / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c

            coeff = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + coeff * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + coeff / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) <= eps:
                return h
        raise ProspectiveFullError("BETA_CONTINUED_FRACTION_DID_NOT_CONVERGE")

    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        value = bt * betacf(a, b, x) / a
    else:
        value = 1.0 - bt * betacf(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, value))


def student_t_cdf(t_statistic: float, degrees_of_freedom: float) -> float:
    t = float(t_statistic)
    df = float(degrees_of_freedom)
    require(math.isfinite(t), "NONFINITE_T_STATISTIC")
    require(math.isfinite(df) and df > 0.0, "BAD_T_DEGREES_OF_FREEDOM")
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    ibeta = _regularized_beta(x, df / 2.0, 0.5)
    cdf = 0.5 * ibeta if t < 0.0 else 1.0 - 0.5 * ibeta
    return min(1.0, max(0.0, cdf))


def one_sample_less(values: Sequence[float]) -> dict[str, float]:
    xs = [float(v) for v in values]
    require(len(xs) >= 2, "H1_SAMPLE_SIZE")
    require(all(math.isfinite(v) for v in xs), "H1_NONFINITE")
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    require(math.isfinite(sd) and sd > 0.0, "H1_ZERO_VARIANCE")
    t_stat = mean / (sd / math.sqrt(len(xs)))
    df = float(len(xs) - 1)
    p = student_t_cdf(t_stat, df)
    return {
        "t_statistic": float(t_stat),
        "degrees_of_freedom": df,
        "raw_p": float(p),
    }


def welch_less(large: Sequence[float], small: Sequence[float]) -> dict[str, float]:
    a = [float(v) for v in large]
    b = [float(v) for v in small]
    require(len(a) >= 2 and len(b) >= 2, "H2_SAMPLE_SIZE")
    require(all(math.isfinite(v) for v in (*a, *b)), "H2_NONFINITE")
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    term_a = var_a / len(a)
    term_b = var_b / len(b)
    denom2 = term_a + term_b
    require(math.isfinite(denom2) and denom2 > 0.0, "H2_ZERO_VARIANCE")
    t_stat = (mean_a - mean_b) / math.sqrt(denom2)
    df_denom = (term_a * term_a) / (len(a) - 1) + (term_b * term_b) / (len(b) - 1)
    require(math.isfinite(df_denom) and df_denom > 0.0, "H2_DF_DENOMINATOR")
    df = (denom2 * denom2) / df_denom
    p = student_t_cdf(t_stat, df)
    return {
        "t_statistic": float(t_stat),
        "degrees_of_freedom": float(df),
        "raw_p": float(p),
    }


def holm_two(tests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(tests) == 2, "HOLM_EXACTLY_TWO_TESTS")
    rows = [dict(test) for test in tests]
    names = [str(row["hypothesis"]) for row in rows]
    require(set(names) == {"H1", "H2"}, "HOLM_HYPOTHESIS_SET")
    for row in rows:
        p = float(row["raw_p"])
        require(math.isfinite(p) and 0.0 <= p <= 1.0, "HOLM_P_RANGE")

    order = sorted(range(2), key=lambda i: (float(rows[i]["raw_p"]), names[i]))
    running = 0.0
    for rank, index in enumerate(order):
        multiplicity = 2 - rank
        candidate = min(1.0, multiplicity * float(rows[index]["raw_p"]))
        running = max(running, candidate)
        rows[index]["holm_adjusted_p"] = running
        rows[index]["holm_reject"] = bool(running <= FAMILY_ALPHA)
        rows[index]["holm_rank"] = rank + 1
    return rows


def _describe(values: Sequence[float]) -> dict[str, Any]:
    xs = [float(v) for v in values]
    require(len(xs) >= 1, "DESCRIPTIVE_EMPTY")
    require(all(math.isfinite(v) for v in xs), "DESCRIPTIVE_NONFINITE")
    return {
        "n": len(xs),
        "mean": float(statistics.fmean(xs)),
        "median": float(statistics.median(xs)),
        "sample_sd": float(statistics.stdev(xs)) if len(xs) >= 2 else None,
        "negative_fraction": sum(v < 0.0 for v in xs) / len(xs),
        "positive_fraction": sum(v > 0.0 for v in xs) / len(xs),
        "zero_fraction": sum(v == 0.0 for v in xs) / len(xs),
    }


def analyze_responses(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(items) == SOURCE_PAIR_COUNT, "ANALYSIS_ITEM_COUNT")
    large = [float(row["R_ALIGN"]) for row in items if row["regime"] == REGIME_LARGE]
    small = [float(row["R_ALIGN"]) for row in items if row["regime"] == REGIME_SMALL]
    require(len(large) >= MIN_GROUP_SIZE, "ANALYSIS_LARGE_GROUP_SIZE")
    require(len(small) >= MIN_GROUP_SIZE, "ANALYSIS_SMALL_GROUP_SIZE")

    h1 = {
        "hypothesis": "H1",
        "alternative": "mean_R_ALIGN_LARGE_less_than_0",
        **one_sample_less(large),
        "direction_condition": statistics.fmean(large) < 0.0,
    }
    h2 = {
        "hypothesis": "H2",
        "alternative": "mean_R_ALIGN_LARGE_less_than_mean_R_ALIGN_SMALL",
        **welch_less(large, small),
        "direction_condition": statistics.fmean(large) < statistics.fmean(small),
    }
    tests = holm_two([h1, h2])
    support = all(
        bool(row["direction_condition"]) and bool(row["holm_reject"])
        for row in tests
    )
    return {
        "n_LARGE": len(large),
        "n_SMALL": len(small),
        "R_ALIGN_LARGE": _describe(large),
        "R_ALIGN_SMALL": _describe(small),
        "R_ALIGN_OVERALL": _describe(large + small),
        "inferential_family": {
            "family_alpha": FAMILY_ALPHA,
            "multiplicity": "Holm",
            "hypothesis_count": 2,
            "tests": tests,
        },
        "prospective_regime_test": (
            "SUPPORTED" if support else "NOT_ESTABLISHED"
        ),
        "scientific_conclusion": (
            "PROSPECTIVE_ADVERSE_REGIME_SUPPORTED"
            if support
            else "PROSPECTIVE_ADVERSE_REGIME_NOT_ESTABLISHED"
        ),
    }


def _write_final_artifacts(
    output_dir: Path,
    *,
    items: Sequence[Mapping[str, Any]] | None,
    summary: Mapping[str, Any],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in (REGIME_FILE, REGIME_FREEZE_FILE):
        path = output_dir / filename
        require(path.is_file(), f"MISSING_PREINTERVENTION_ARTIFACT:{filename}")
        hashes[filename] = sha256_file(path)

    if items is not None:
        raw = jsonl_bytes(items)
        hashes[ITEM_FILE] = _write_bytes(output_dir / ITEM_FILE, raw)

    hashes[SUMMARY_FILE] = _write_bytes(
        output_dir / SUMMARY_FILE,
        canonical_json_bytes(summary),
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "files": {
            name: {
                "sha256": digest,
                "bytes": int((output_dir / name).stat().st_size),
            }
            for name, digest in sorted(hashes.items())
        },
    }
    hashes[MANIFEST_FILE] = _write_bytes(
        output_dir / MANIFEST_FILE,
        canonical_json_bytes(manifest),
    )

    checksum_lines = [
        f"{digest}  {name}\n"
        for name, digest in sorted(hashes.items())
    ]
    _write_bytes(output_dir / CHECKSUM_FILE, "".join(checksum_lines).encode("utf-8"))
    return hashes


def run_full(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    backend.runtime_gate()
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")

    with backend.parent_runtime_rebind():
        rows, encoded, event_rows = eq.load_prospective_inputs(tokenizer_snapshot)
        pairs = eq._pair_order(rows)
        require(len(pairs) == SOURCE_PAIR_COUNT, "PAIR_COUNT")
        events = parent.event_lookup(event_rows)
        parent.validate_transport_event_plan(pairs, events)
        row_index = parent.build_row_index(rows)
        trace_code, trace_line = measurement._resolve_and_validate_runtime_binding()

        model, checkpoint_sha = parent.load_representative_model_external(
            model_snapshot=model_snapshot,
            checkpoint_path=checkpoint_path,
        )
        require(
            checkpoint_sha == extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
            "CHECKPOINT_IDENTITY",
        )
        runtime_ctx = transport_runtime.validate_runtime_components(model)
        kernels = backend.load_exact_fast_kernels()
        model.to(torch.device("cuda:0"))
        model.eval()
        require(
            all(parameter.device.type == "cuda" for parameter in model.mamba.parameters()),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = backend._make_fast_capture(kernels)
        original_capture = parent.capture_branch
        budget = parent.ForwardBudget(FULL_FORWARD_BUDGET)
        baseline_rows: list[dict[str, Any]] = []
        runtime_plans: dict[str, dict[str, Any]] = {}

        parent.capture_branch = fast_capture
        try:
            # Phase 1: all 300 pairs complete baseline geometry first. No
            # alignment response exists yet, so regime membership is outcome-blind.
            for pair in pairs:
                public, plan = _run_baseline_pair(
                    pair,
                    model=model,
                    runtime_ctx=runtime_ctx,
                    trace_code=trace_code,
                    trace_line=trace_line,
                    encoded=encoded,
                    row_index=row_index,
                    events=events,
                    budget=budget,
                )
                baseline_rows.append(public)
                runtime_plans[pair] = plan

            require(budget.used == BASELINE_FORWARD_BUDGET, "BASELINE_FORWARD_BUDGET")
            regime_rows, counts = freeze_regimes(pairs, baseline_rows)
            freeze_info = _write_regime_freeze(output_dir, regime_rows, counts)

            if not group_size_passes(counts):
                summary = {
                    "schema_version": SUMMARY_SCHEMA,
                    "result": "PASS_BASELINE_PHASE_BLOCKED_INSUFFICIENT_FIXED_GROUP_SIZE",
                    "execution_head": expected_head,
                    "gate_freeze_commit": GATE_FREEZE_COMMIT,
                    "gate_artifact_sha256": GATE_ARTIFACT_SHA256,
                    "prospective_holdout_rows_sha256": eq.eligibility.EXPECTED_PROSPECTIVE_ROWS_SHA256,
                    "eligibility_anchor_manifest_sha256": eq.ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
                    "representative_checkpoint_sha256": checkpoint_sha,
                    "threshold": ALIGNMENT_SHIFT_THRESHOLD,
                    "minimum_group_size": MIN_GROUP_SIZE,
                    "n_LARGE": int(counts[REGIME_LARGE]),
                    "n_SMALL": int(counts[REGIME_SMALL]),
                    "baseline_model_forward_count": BASELINE_FORWARD_BUDGET,
                    "alignment_model_forward_count": 0,
                    "total_model_forward_count": BASELINE_FORWARD_BUDGET,
                    "scientific_budget_forward_count": BASELINE_FORWARD_BUDGET,
                    "prospective_regime_test": "BLOCKED_INSUFFICIENT_FIXED_GROUP_SIZE",
                    "scientific_conclusion": None,
                    "regime_manifest_sha256": freeze_info["regime_manifest_sha256"],
                    "regime_freeze_sha256": freeze_info["regime_freeze_sha256"],
                    "training_executed": False,
                    "backward_executed": False,
                    "task_heads_executed": False,
                    "logits_read": False,
                    "response_dependent_exclusion": False,
                    "threshold_reestimated": False,
                }
                _write_final_artifacts(output_dir, items=None, summary=summary)
                torch.cuda.synchronize()
                return summary

            # Phase 2 begins only after the persisted regime freeze and group gate.
            items: list[dict[str, Any]] = []
            for pair, baseline_row in zip(pairs, baseline_rows, strict=True):
                items.append(
                    _run_alignment_pair(
                        pair,
                        baseline_row=baseline_row,
                        runtime_plan=runtime_plans[pair],
                        model=model,
                        runtime_ctx=runtime_ctx,
                        trace_code=trace_code,
                        trace_line=trace_line,
                        encoded=encoded,
                        row_index=row_index,
                        budget=budget,
                    )
                )
            budget.assert_exact()
            torch.cuda.synchronize()
        finally:
            parent.capture_branch = original_capture

    analysis = analyze_responses(items)
    item_raw = jsonl_bytes(items)
    item_sha = sha256_bytes(item_raw)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": "PASS_PROSPECTIVE_FULL_EXECUTION",
        "execution_head": expected_head,
        "gate_freeze_commit": GATE_FREEZE_COMMIT,
        "gate_artifact_sha256": GATE_ARTIFACT_SHA256,
        "prospective_holdout_rows_sha256": eq.eligibility.EXPECTED_PROSPECTIVE_ROWS_SHA256,
        "eligibility_anchor_manifest_sha256": eq.ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
        "representative_checkpoint_sha256": checkpoint_sha,
        "kernels_version": backend.KERNELS_VERSION,
        "mamba_revision": backend.MAMBA_REV,
        "mamba_binary_sha256": backend.MAMBA_BINARY_SHA256,
        "causal_conv_revision": backend.CONV_REV,
        "causal_conv_binary_sha256": backend.CONV_BINARY_SHA256,
        "build_variant": backend.BUILD_VARIANT,
        "python_version": backend.EXPECTED_RUNTIME["python"],
        "numpy_version": backend.EXPECTED_RUNTIME["numpy"],
        "torch_version": backend.EXPECTED_RUNTIME["torch"],
        "transformers_version": backend.EXPECTED_RUNTIME["transformers"],
        "cuda_runtime": backend.EXPECTED_CUDA_RUNTIME,
        "cuda_device": backend.EXPECTED_DEVICE_NAME,
        "cuda_capability": list(backend.EXPECTED_CAPABILITY),
        "threshold": ALIGNMENT_SHIFT_THRESHOLD,
        "minimum_group_size": MIN_GROUP_SIZE,
        "baseline_model_forward_count": BASELINE_FORWARD_BUDGET,
        "alignment_model_forward_count": ALIGNMENT_FORWARD_BUDGET,
        "total_model_forward_count": FULL_FORWARD_BUDGET,
        "scientific_budget_forward_count": FULL_FORWARD_BUDGET,
        "regime_manifest_sha256": freeze_info["regime_manifest_sha256"],
        "regime_freeze_sha256": freeze_info["regime_freeze_sha256"],
        "item_metrics_sha256": item_sha,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "response_dependent_exclusion": False,
        "threshold_reestimated": False,
        **analysis,
    }
    _write_final_artifacts(output_dir, items=items, summary=summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full 300-pair fast-CUDA prospective validation for the frozen "
            "Gen4-K large-correction regime design. Executes exactly 1800 "
            "scientific model forwards when the fixed group-size gate passes."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_full(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT =", summary["result"])
    print("N_LARGE =", summary["n_LARGE"])
    print("N_SMALL =", summary["n_SMALL"])
    print("TOTAL_MODEL_FORWARD_COUNT =", summary["total_model_forward_count"])
    print("PROSPECTIVE_REGIME_TEST =", summary["prospective_regime_test"])
    print("SCIENTIFIC_CONCLUSION =", summary["scientific_conclusion"])
    if "inferential_family" in summary:
        for test in summary["inferential_family"]["tests"]:
            print(
                test["hypothesis"],
                "RAW_P =", test["raw_p"],
                "HOLM_ADJUSTED_P =", test["holm_adjusted_p"],
                "HOLM_REJECT =", test["holm_reject"],
            )


if __name__ == "__main__":
    main()
