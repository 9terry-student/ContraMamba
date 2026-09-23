#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba1-five-scale-ladder-extension"

RAW_FREEZE_HEAD = "9bbea9cb03327d4dca86cdb148eec44a4dd27b34"
EXTENSION_PLAN_FREEZE = "e567338f1dcd99d61ed7465ec39d441566f71fa3"
PRIOR_FOUR_SCALE_ANALYSIS_FREEZE = "88a6d6c469d44071a070b494485efe56db4faa58"

RUN_NAME = (
    "g4k-mamba130m-vanillalm-readout-"
    "xg1-2701-3000-p3-p5-2gpu-cb679ae"
)

RAW_DIR = (
    ROOT
    / "reports"
    / "reason_router_gen4_mamba1_vanilla_lm_readout_runs"
    / RUN_NAME
)

RAW_ITEM = RAW_DIR / "vanilla_lm_readout_items.jsonl"
RAW_SUMMARY = RAW_DIR / "raw_vanilla_lm_readout_summary.json"
RAW_MANIFEST = RAW_DIR / "artifact_manifest.json"
RAW_SUMS = RAW_DIR / "SHA256SUMS.txt"

CONTRA_PAIR_FILE = (
    ROOT
    / "reports"
    / "reason_router_gen4_mamba130m_readout_alignment_analysis_v1"
    / "readout_alignment_pair_values.jsonl"
)

PRIOR_ANALYSIS_FILE = (
    ROOT
    / "reports"
    / "reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1"
    / "functional_control_analysis.json"
)

OUTPUT_DIR_REL = Path(
    "reports/"
    "reason_router_gen4_mamba130m_vanilla_lm_completeness_analysis_v1"
)

ANALYSIS_FILE = "mamba130m_completeness_analysis.json"
PAIR_FILE = "mamba130m_completeness_pair_values.jsonl"
REPORT_FILE = "mamba130m_completeness_analysis.md"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

RESULT = "PASS_MAMBA130M_VANILLA_LM_COMPLETENESS_STATIC_ANALYSIS"


class AnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AnalysisError(message)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), "BRANCH_MISMATCH")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for ancestor, label in (
        (RAW_FREEZE_HEAD, "RAW_FREEZE"),
        (EXTENSION_PLAN_FREEZE, "EXTENSION_PLAN"),
        (PRIOR_FOUR_SCALE_ANALYSIS_FREEZE, "PRIOR_ANALYSIS"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, expected_head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_NOT_ANCESTOR")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"MISSING:{path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def descriptive(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and a.size > 0, "DESCRIPTIVE_SHAPE")
    require(bool(np.isfinite(a).all()), "DESCRIPTIVE_NONFINITE")
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd_sample": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "min": float(a.min()),
        "max": float(a.max()),
        "fraction_positive": float(np.mean(a > 0.0)),
        "fraction_negative": float(np.mean(a < 0.0)),
        "fraction_zero": float(np.mean(a == 0.0)),
    }


def sign(value: float) -> str:
    require(math.isfinite(value), "SIGN_NONFINITE")
    if value > 0.0:
        return "+"
    if value < 0.0:
        return "-"
    return "0"


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 1 and a.size > 1, "PEARSON_SHAPE")
    require(bool(np.isfinite(a).all() and np.isfinite(b).all()), "PEARSON_NONFINITE")
    require(float(a.std()) > 0.0 and float(b.std()) > 0.0, "PEARSON_CONSTANT")
    return float(np.corrcoef(a, b)[0, 1])


def average_ranks(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and a[order[j]] == a[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def population_covariance(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 1, "COV_SHAPE")
    return float(np.mean((a - a.mean()) * (b - b.mean())))


def validate_raw_bundle() -> list[dict[str, Any]]:
    expected_names = {
        "SHA256SUMS.txt",
        "artifact_manifest.json",
        "raw_vanilla_lm_readout_summary.json",
        "vanilla_lm_readout_items.jsonl",
    }
    require(RAW_DIR.is_dir(), "RAW_DIR")
    require(
        {p.name for p in RAW_DIR.iterdir() if p.is_file()} == expected_names,
        "RAW_FILE_SET",
    )

    sums: dict[str, str] = {}
    for line in RAW_SUMS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            sums[name] = digest

    require(
        set(sums) == {
            "artifact_manifest.json",
            "raw_vanilla_lm_readout_summary.json",
            "vanilla_lm_readout_items.jsonl",
        },
        "RAW_SUM_SET",
    )
    for name, digest in sums.items():
        require(sha256_file(RAW_DIR / name) == digest, f"RAW_SUM:{name}")

    summary = json.loads(RAW_SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))

    for doc, label in ((summary, "SUMMARY"), (manifest, "MANIFEST")):
        require(
            doc["result"] == "PASS_MAMBA1_VANILLA_LM_FUNCTIONAL_READOUT_RAW",
            f"{label}_RESULT",
        )
        require(doc["scale"] == "mamba130m", f"{label}_SCALE")
        require(doc["pair_count"] == 300, f"{label}_PAIR_COUNT")
        require(doc["item_count"] == 600, f"{label}_ITEM_COUNT")
        require(doc["p_value_count"] == 0, f"{label}_PVALUE")
        require(doc["row_filter_performed"] is False, f"{label}_FILTER")
        require(doc["rescue_performed"] is False, f"{label}_RESCUE")
        require(doc["selection_reopened"] is False, f"{label}_SELECTION")

    require(
        manifest["mamba130_extension_plan_freeze_commit"] == EXTENSION_PLAN_FREEZE,
        "MANIFEST_EXTENSION_PLAN",
    )
    # The scientific run executed at cb679ae; the raw freeze commit is later.
    require(
        manifest["execution_head"] == "cb679aea7b9970923a5c8a0704b66c6a69cd8be8",
        "MANIFEST_EXECUTION_HEAD",
    )

    rows = read_jsonl(RAW_ITEM)
    require(len(rows) == 600, "RAW_ROW_COUNT")

    expected_pairs = tuple(f"xg1_fact_{i:04d}" for i in range(2701, 3001))
    expected_keys = {
        (pair, cell)
        for pair in expected_pairs
        for cell in ("C0_SHAM", "C2_NAME")
    }
    observed_keys = {
        (str(row["source_pair_id"]), str(row["contrast_cell_id"]))
        for row in rows
    }
    require(observed_keys == expected_keys, "RAW_ROW_COVERAGE")
    return rows


def analyze() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = validate_raw_bundle()
    prior = json.loads(PRIOR_ANALYSIS_FILE.read_text(encoding="utf-8"))
    contra_rows = read_jsonl(CONTRA_PAIR_FILE)

    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        by_pair.setdefault(pair, {})
        require(cell not in by_pair[pair], f"DUPLICATE:{pair}:{cell}")
        by_pair[pair][cell] = row

    pair_ids = sorted(by_pair)
    require(len(pair_ids) == 300, "PAIR_COUNT")

    pair_values: list[float] = []
    c0_values: list[float] = []
    c2_values: list[float] = []
    pair_rows: list[dict[str, Any]] = []

    contra_map = {
        str(row["source_pair_id"]): float(row["Delta_L"])
        for row in contra_rows
    }
    require(set(contra_map) == set(pair_ids), "CONTRA_PAIR_COVERAGE")

    contra_owned: list[float] = []
    contra_forward: list[float] = []

    for pair in pair_ids:
        c0 = float(by_pair[pair]["C0_SHAM"]["task_matched"]["Delta_L_LM"])
        c2 = float(by_pair[pair]["C2_NAME"]["task_matched"]["Delta_L_LM"])
        pair_mean = 0.5 * (c0 + c2)

        owned = contra_map[pair]
        forward = 2.0 * owned

        c0_values.append(c0)
        c2_values.append(c2)
        pair_values.append(pair_mean)
        contra_owned.append(owned)
        contra_forward.append(forward)

        pair_rows.append({
            "scale": "130M",
            "source_pair_id": pair,
            "LM_TASK_MATCHED": pair_mean,
            "LM_TASK_MATCHED_C0_SHAM": c0,
            "LM_TASK_MATCHED_C2_NAME": c2,
            "LM_COMMON_P3_P5": pair_mean,
            "ContraMamba_owned": owned,
            "ContraMamba_forward_equivalent": forward,
        })

    g = np.asarray([float(r["gradient_strong_l2"]) for r in rows], dtype=np.float64)
    n = np.asarray(
        [float(r["task_matched"]["selected_component_l2"]) for r in rows],
        dtype=np.float64,
    )
    h = np.asarray(
        [float(r["task_matched"]["cosine_gap"]) for r in rows],
        dtype=np.float64,
    )
    d = np.asarray(
        [float(r["task_matched"]["Delta_L_LM"]) for r in rows],
        dtype=np.float64,
    )
    reconstruction = d - g * n * h
    require(float(np.max(np.abs(reconstruction))) <= 1e-12, "DELTA_IDENTITY")

    m130 = {
        "task_matched_planes": ["P3", "P5"],
        "common_planes": ["P3", "P5"],
        "task_matched_pair": descriptive(pair_values),
        "by_cell": {
            "C0_SHAM": descriptive(c0_values),
            "C2_NAME": descriptive(c2_values),
        },
        "row_diagnostics": {
            "row_n": 600,
            "mean_cosine_gap": float(h.mean()),
            "corr_component_norm_cosine_gap": pearson(n, h),
            "mean_gradient_strong_l2": float(g.mean()),
            "mean_component_norm": float(n.mean()),
            "NORM_GAP": float(g.mean() * population_covariance(n, h)),
            "gradient_strong_l2": descriptive(g),
            "max_abs_delta_identity_error": float(np.max(np.abs(reconstruction))),
        },
        "contra_comparison": {
            "contra_owned": descriptive(contra_owned),
            "contra_forward_equivalent": descriptive(contra_forward),
            "pearson": pearson(pair_values, contra_forward),
            "spearman": spearman(pair_values, contra_forward),
            "pair_sign_agreement_fraction": float(
                np.mean(
                    np.sign(np.asarray(pair_values, dtype=np.float64))
                    == np.sign(np.asarray(contra_forward, dtype=np.float64))
                )
            ),
        },
    }

    old_order = ["370M", "790M", "1.4B", "2.8B"]
    old_lm = [
        sign(float(prior["scales"][scale]["task_matched"]["pair_endpoint"]["mean"]))
        for scale in old_order
    ]
    old_contra = [
        sign(
            float(
                prior["scales"][scale]["task_matched"]["contra_comparison"]
                ["contra_forward_equivalent"]["mean"]
            )
        )
        for scale in old_order
    ]

    m130_lm_sign = sign(float(m130["task_matched_pair"]["mean"]))
    m130_contra_sign = sign(
        float(m130["contra_comparison"]["contra_forward_equivalent"]["mean"])
    )

    analysis = {
        "result": RESULT,
        "raw_freeze_head": RAW_FREEZE_HEAD,
        "extension_plan_freeze": EXTENSION_PLAN_FREEZE,
        "prior_four_scale_analysis_freeze": PRIOR_FOUR_SCALE_ANALYSIS_FREEZE,
        "status": "POST_PRIMARY_DESCRIPTIVE_COMPLETENESS",
        "inferential_test_performed": False,
        "p_value_count": 0,
        "row_filter_performed": False,
        "rescue_performed": False,
        "selection_reopened": False,
        "mamba130m": m130,
        "extension_diagnostics": {
            "M130_TASK_MATCHED_SIGN_LM": m130_lm_sign,
            "M130_CONTRAMAMBA_SIGN": m130_contra_sign,
            "M130_SIGN_CONCORDANCE": m130_lm_sign == m130_contra_sign,
        },
        "five_scale_descriptive_completeness": {
            "scale_order": ["130M", *old_order],
            "vanilla_lm_task_matched_sign_vector": [m130_lm_sign, *old_lm],
            "contramamba_forward_equivalent_sign_vector": [
                m130_contra_sign,
                *old_contra,
            ],
            "aggregate_sign_opposite_at_all_five_sampled_scales": all(
                a != b and a != "0" and b != "0"
                for a, b in zip(
                    [m130_lm_sign, *old_lm],
                    [m130_contra_sign, *old_contra],
                    strict=True,
                )
            ),
            "prospective_status":
                "130M is post-primary completeness; 370M-2.8B remain the "
                "original prospective four-scale control.",
        },
        "interpretation": {
            "primary":
                "The post-primary 130M completeness extension is negative under "
                "the vanilla-LM TASK_MATCHED P3-P5 readout while the frozen "
                "ContraMamba 130M forward-equivalent comparator is positive.",
            "five_scale":
                "Combining the immutable original four-scale control with this "
                "extension yields descriptive sign vectors vanilla=-,-,-,+,+ "
                "versus ContraMamba=+,+,+,-,-.",
            "anti_inversion_boundary":
                "The 130M pair-level association is positive rather than a uniform "
                "negative mapping, so the five-scale aggregate sign opposition does "
                "not establish a simple rowwise inversion law.",
            "no_scaling_law": True,
            "no_threshold_claim": True,
            "no_plane_homology_claim": True,
            "no_new_p_value": True,
        },
    }

    return analysis, pair_rows


def report_markdown(analysis: Mapping[str, Any]) -> str:
    x = analysis["mamba130m"]
    comp = x["contra_comparison"]
    five = analysis["five_scale_descriptive_completeness"]

    lines = [
        "# Mamba-130M Vanilla-LM Functional-Control Completeness Extension",
        "",
        "## Status",
        "",
        "`POST_PRIMARY_DESCRIPTIVE_COMPLETENESS`",
        "",
        "The original four-scale vanilla-LM control remains immutable. This analysis "
        "adds the subsequently frozen 130M completeness extension only.",
        "",
        "No new p-value, row filtering, rescue, re-selection, monotonicity test, "
        "threshold estimate, or scaling-law fit is introduced.",
        "",
        "## 130M extension result",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Vanilla-LM TASK_MATCHED mean | {x['task_matched_pair']['mean']:+.12g} |",
        f"| Vanilla-LM median | {x['task_matched_pair']['median']:+.12g} |",
        f"| Vanilla-LM positive fraction | {x['task_matched_pair']['fraction_positive']:.6f} |",
        f"| C0_SHAM mean | {x['by_cell']['C0_SHAM']['mean']:+.12g} |",
        f"| C2_NAME mean | {x['by_cell']['C2_NAME']['mean']:+.12g} |",
        f"| ContraMamba forward-equivalent mean | {comp['contra_forward_equivalent']['mean']:+.12g} |",
        f"| Pair Pearson | {comp['pearson']:+.6f} |",
        f"| Pair Spearman | {comp['spearman']:+.6f} |",
        f"| Pair sign agreement | {comp['pair_sign_agreement_fraction']:.6f} |",
        "",
        "The 130M vanilla-LM mean is negative while the frozen ContraMamba "
        "forward-equivalent mean is positive, so `M130_SIGN_CONCORDANCE=False`.",
        "",
        "Both C0_SHAM and C2_NAME have negative aggregate means, so the 130M result "
        "is not localized to only one of the two frozen cells.",
        "",
        "## Geometry / functional diagnostics",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Mean cosine gap | {x['row_diagnostics']['mean_cosine_gap']:+.12g} |",
        f"| corr(component norm, cosine gap) | {x['row_diagnostics']['corr_component_norm_cosine_gap']:+.6f} |",
        f"| NORM_GAP | {x['row_diagnostics']['NORM_GAP']:+.12g} |",
        f"| Mean strong gradient norm | {x['row_diagnostics']['mean_gradient_strong_l2']:+.12g} |",
        f"| Max identity reconstruction error | {x['row_diagnostics']['max_abs_delta_identity_error']:.3e} |",
        "",
        "The mean cosine gap is negative. Component-norm/cosine-gap dependence is "
        "positive and NORM_GAP is positive, so the dependence correction partially "
        "opposes rather than creates the negative aggregate endpoint.",
        "",
        "## Five-scale descriptive completeness",
        "",
        "Scale order: `130M, 370M, 790M, 1.4B, 2.8B`.",
        "",
        "Vanilla-LM TASK_MATCHED sign vector: `-,-,-,+,+`.",
        "",
        "ContraMamba forward-equivalent sign vector: `+,+,+,-,-`.",
        "",
        f"`aggregate_sign_opposite_at_all_five_sampled_scales = "
        f"{five['aggregate_sign_opposite_at_all_five_sampled_scales']}`.",
        "",
        "This five-scale vector is descriptive completeness, not a retroactive "
        "replacement of the prospectively frozen four-scale control.",
        "",
        "## Interpretation boundary",
        "",
        "The additional 130M scale strengthens the descriptive observation that the "
        "pretrained LM objective does not reproduce the ContraMamba aggregate sign "
        "structure across the five sampled model sizes.",
        "",
        "It still does not support a simple rowwise inversion account: the 130M "
        "pair-level Pearson and Spearman associations are positive, and sign "
        "agreement is above one half.",
        "",
        "Do not claim a universal scaling law, continuous size threshold, semantic "
        "homology of plane labels, or a deterministic objective-to-objective inversion.",
        "",
        f"`RAW_FREEZE_HEAD = {RAW_FREEZE_HEAD}`",
        f"`EXTENSION_PLAN_FREEZE = {EXTENSION_PLAN_FREEZE}`",
        f"`PRIOR_FOUR_SCALE_ANALYSIS_FREEZE = {PRIOR_FOUR_SCALE_ANALYSIS_FREEZE}`",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    analysis: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
) -> None:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / ANALYSIS_FILE).write_bytes(
        (
            json.dumps(
                analysis,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )

    (output_dir / PAIR_FILE).write_bytes(
        b"".join(
            (
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            for row in pairs
        )
    )

    (output_dir / REPORT_FILE).write_bytes(
        report_markdown(analysis).encode("utf-8")
    )

    manifest = {
        "result": RESULT,
        "raw_freeze_head": RAW_FREEZE_HEAD,
        "extension_plan_freeze": EXTENSION_PLAN_FREEZE,
        "prior_four_scale_analysis_freeze": PRIOR_FOUR_SCALE_ANALYSIS_FREEZE,
        "status": "POST_PRIMARY_DESCRIPTIVE_COMPLETENESS",
        "pair_output_rows": len(pairs),
        "inferential_test_performed": False,
        "p_value_count": 0,
        "row_filter_performed": False,
        "rescue_performed": False,
        "selection_reopened": False,
        "output_files": [ANALYSIS_FILE, PAIR_FILE, REPORT_FILE],
    }

    (output_dir / MANIFEST_FILE).write_bytes(
        (
            json.dumps(
                manifest,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )

    names = (ANALYSIS_FILE, PAIR_FILE, REPORT_FILE, MANIFEST_FILE)
    (output_dir / SUMS_FILE).write_bytes(
        "".join(
            f"{sha256_file(output_dir / name)}  {name}\n"
            for name in sorted(names)
        ).encode("ascii")
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT_DIR_REL)
    args = parser.parse_args(argv)

    authenticate_repo(args.expected_head)
    analysis, pairs = analyze()
    write_outputs(args.output_dir, analysis, pairs)

    ext = analysis["extension_diagnostics"]
    five = analysis["five_scale_descriptive_completeness"]

    print("RESULT=" + RESULT)
    print("M130_TASK_MATCHED_SIGN_LM=" + ext["M130_TASK_MATCHED_SIGN_LM"])
    print("M130_CONTRAMAMBA_SIGN=" + ext["M130_CONTRAMAMBA_SIGN"])
    print("M130_SIGN_CONCORDANCE=" + str(ext["M130_SIGN_CONCORDANCE"]))
    print(
        "FIVE_SCALE_VANILLA_LM_SIGN_VECTOR="
        + ",".join(five["vanilla_lm_task_matched_sign_vector"])
    )
    print(
        "FIVE_SCALE_CONTRAMAMBA_SIGN_VECTOR="
        + ",".join(five["contramamba_forward_equivalent_sign_vector"])
    )
    print(
        "AGGREGATE_SIGN_OPPOSITE_ALL_FIVE="
        + str(five["aggregate_sign_opposite_at_all_five_sampled_scales"])
    )
    print("P_VALUE_COUNT=0")
    print("ROW_FILTER_PERFORMED=False")
    print("RESCUE_PERFORMED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
