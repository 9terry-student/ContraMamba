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
RAW_FREEZE_HEAD = "f78c56418902dc208adaef6c9a188378f9a7fa45"
PLAN_FREEZE_COMMIT = "1fe9a198a15c9cea0e5451d918cd949bc21bf7e0"

SCALE_ORDER = ("370M", "790M", "1.4B", "2.8B")
CELLS = ("C0_SHAM", "C2_NAME")
N = 300

VANILLA = {
    "370M": {
        "run": (
            "g4k-mamba370m-vanillalm-readout-xg1-4801-5100-"
            "p3-p5-2gpu-9c56906-retry7"
        ),
        "task": ("P3", "P5"),
        "pair_first": "xg1_fact_4801",
        "pair_last": "xg1_fact_5100",
    },
    "790M": {
        "run": (
            "g4k-mamba790m-vanillalm-readout-xg1-7501-7800-"
            "p2-p5-2gpu-e5f49b4"
        ),
        "task": ("P2", "P5"),
        "pair_first": "xg1_fact_7501",
        "pair_last": "xg1_fact_7800",
    },
    "1.4B": {
        "run": (
            "g4k-mamba14b-vanillalm-readout-xg1-4801-5100-"
            "p5-p4-2gpu-e7ee5fa"
        ),
        "task": ("P5", "P4"),
        "pair_first": "xg1_fact_4801",
        "pair_last": "xg1_fact_5100",
    },
    "2.8B": {
        "run": (
            "g4k-mamba28b-vanillalm-readout-xg1-6601-6900-"
            "p3-p5-2gpu-c3baf3d"
        ),
        "task": ("P3", "P5"),
        "pair_first": "xg1_fact_6601",
        "pair_last": "xg1_fact_6900",
    },
}

VANILLA_ROOT = (
    ROOT
    / "reports"
    / "reason_router_gen4_mamba1_vanilla_lm_readout_runs"
)

CONTRA_370_14 = (
    ROOT
    / "reports"
    / "reason_router_gen4_mamba370m14b_readout_alignment_analysis_v1"
    / "readout_alignment_pair_values.jsonl"
)
CONTRA_790 = (
    ROOT
    / "reports"
    / "reason_router_gen4_mamba790m_readout_alignment_analysis_v1"
    / "readout_alignment_pair_values.jsonl"
)
CONTRA_28 = (
    ROOT
    / "reports"
    / "reason_router_gen4_mamba28b_readout_alignment_analysis_v1"
    / "readout_alignment_pair_values.jsonl"
)

OUTPUT_DIR_REL = Path(
    "reports/reason_router_gen4_mamba1_vanilla_lm_functional_control_analysis_v1"
)
ANALYSIS_FILE = "functional_control_analysis.json"
PAIR_FILE = "functional_control_pair_values.jsonl"
REPORT_FILE = "functional_control_analysis.md"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

RESULT = "PASS_MAMBA1_VANILLA_LM_FUNCTIONAL_CONTROL_STATIC_ANALYSIS"


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
    require(
        git("branch", "--show-current") in ("", EXPECTED_BRANCH),
        "BRANCH_MISMATCH",
    )
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", PLAN_FREEZE_COMMIT, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "PLAN_FREEZE_NOT_ANCESTOR")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", RAW_FREEZE_HEAD, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "RAW_FREEZE_NOT_ANCESTOR")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"MISSING:{path}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL_OBJECT:{path}")
            out.append(value)
    return out


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


def mean_sign(x: float) -> str:
    require(math.isfinite(x), "MEAN_SIGN_NONFINITE")
    if x > 0.0:
        return "+"
    if x < 0.0:
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


def validate_raw_bundle(scale: str) -> list[dict[str, Any]]:
    spec = VANILLA[scale]
    root = VANILLA_ROOT / spec["run"]
    expected_names = {
        "SHA256SUMS.txt",
        "artifact_manifest.json",
        "raw_vanilla_lm_readout_summary.json",
        "vanilla_lm_readout_items.jsonl",
    }
    require(root.is_dir(), f"RAW_DIR:{scale}")
    require(
        {p.name for p in root.iterdir() if p.is_file()} == expected_names,
        f"RAW_FILE_SET:{scale}",
    )

    sums = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            sums[name] = digest
    require(
        set(sums) == {
            "artifact_manifest.json",
            "raw_vanilla_lm_readout_summary.json",
            "vanilla_lm_readout_items.jsonl",
        },
        f"RAW_SUMS_SET:{scale}",
    )
    for name, digest in sums.items():
        require(sha256_file(root / name) == digest, f"RAW_SUM:{scale}:{name}")

    summary = json.loads(
        (root / "raw_vanilla_lm_readout_summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "artifact_manifest.json").read_text(encoding="utf-8")
    )

    for doc, label in ((summary, "SUMMARY"), (manifest, "MANIFEST")):
        require(
            doc["result"] == "PASS_MAMBA1_VANILLA_LM_FUNCTIONAL_READOUT_RAW",
            f"{label}_RESULT:{scale}",
        )
        require(doc["pair_count"] == 300, f"{label}_PAIR_COUNT:{scale}")
        require(doc["item_count"] == 600, f"{label}_ITEM_COUNT:{scale}")
        require(doc["p_value_count"] == 0, f"{label}_PVALUE:{scale}")
        require(doc["selection_reopened"] is False, f"{label}_SELECTION:{scale}")
        require(doc["row_filter_performed"] is False, f"{label}_FILTER:{scale}")
        require(doc["rescue_performed"] is False, f"{label}_RESCUE:{scale}")

    require(summary["task_matched_selected_plane"] == spec["task"][0], f"TASK_SEL:{scale}")
    require(summary["task_matched_control_plane"] == spec["task"][1], f"TASK_CTL:{scale}")
    require(summary["common_selected_plane"] == "P3", f"COMMON_SEL:{scale}")
    require(summary["common_control_plane"] == "P5", f"COMMON_CTL:{scale}")

    rows = read_jsonl(root / "vanilla_lm_readout_items.jsonl")
    require(len(rows) == 600, f"ROW_COUNT:{scale}")

    expected_pairs = tuple(
        f"xg1_fact_{i}"
        for i in range(
            int(spec["pair_first"].split("_")[-1]),
            int(spec["pair_last"].split("_")[-1]) + 1,
        )
    )
    keys = {(str(r["source_pair_id"]), str(r["contrast_cell_id"])) for r in rows}
    expected_keys = {(p, c) for p in expected_pairs for c in CELLS}
    require(keys == expected_keys, f"ROW_COVERAGE:{scale}")
    return rows


def contra_pair_map(scale: str) -> dict[str, float]:
    if scale in ("370M", "1.4B"):
        rows = read_jsonl(CONTRA_370_14)
        field = "Delta_L_370M" if scale == "370M" else "Delta_L_1.4B"
        return {str(r["source_pair_id"]): 2.0 * float(r[field]) for r in rows}
    if scale == "790M":
        rows = read_jsonl(CONTRA_790)
    else:
        rows = read_jsonl(CONTRA_28)
    return {
        str(r["source_pair_id"]): float(r["Delta_L_forward_equivalent"])
        for r in rows
    }


def analyze_contrast(
    rows: Sequence[Mapping[str, Any]],
    contrast_key: str,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        by_pair.setdefault(pair, {})
        require(cell not in by_pair[pair], f"DUPLICATE:{pair}:{cell}")
        by_pair[pair][cell] = row

    pair_values: dict[str, dict[str, float]] = {}
    for pair in sorted(by_pair):
        require(set(by_pair[pair]) == set(CELLS), f"PAIR_CELLS:{pair}")
        c0 = float(by_pair[pair]["C0_SHAM"][contrast_key]["Delta_L_LM"])
        c2 = float(by_pair[pair]["C2_NAME"][contrast_key]["Delta_L_LM"])
        pair_values[pair] = {
            "C0_SHAM": c0,
            "C2_NAME": c2,
            "pair_mean": 0.5 * (c0 + c2),
        }

    pair = [pair_values[p]["pair_mean"] for p in sorted(pair_values)]
    c0 = [pair_values[p]["C0_SHAM"] for p in sorted(pair_values)]
    c2 = [pair_values[p]["C2_NAME"] for p in sorted(pair_values)]

    g = np.asarray([float(r["gradient_strong_l2"]) for r in rows], dtype=np.float64)
    n = np.asarray(
        [float(r[contrast_key]["selected_component_l2"]) for r in rows],
        dtype=np.float64,
    )
    h = np.asarray(
        [float(r[contrast_key]["cosine_gap"]) for r in rows],
        dtype=np.float64,
    )
    d = np.asarray(
        [float(r[contrast_key]["Delta_L_LM"]) for r in rows],
        dtype=np.float64,
    )
    require(bool(np.isfinite(g).all() and np.isfinite(n).all() and np.isfinite(h).all()), "ROW_NONFINITE")

    reconstruction = d - g * n * h
    require(float(np.max(np.abs(reconstruction))) <= 1e-12, "DELTA_IDENTITY")

    result = {
        "pair_endpoint": descriptive(pair),
        "by_cell": {
            "C0_SHAM": descriptive(c0),
            "C2_NAME": descriptive(c2),
        },
        "row_diagnostics": {
            "row_n": int(len(rows)),
            "mean_cosine_gap": float(h.mean()),
            "corr_component_norm_cosine_gap": pearson(n, h),
            "mean_gradient_strong_l2": float(g.mean()),
            "mean_component_norm": float(n.mean()),
            "NORM_GAP": float(g.mean() * population_covariance(n, h)),
            "gradient_strong_l2": descriptive(g),
            "max_abs_delta_identity_error": float(np.max(np.abs(reconstruction))),
        },
    }
    return result, pair_values


def analyze_all() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    analysis: dict[str, Any] = {
        "result": RESULT,
        "raw_freeze_head": RAW_FREEZE_HEAD,
        "plan_freeze_commit": PLAN_FREEZE_COMMIT,
        "inferential_test_performed": False,
        "p_value_count": 0,
        "row_filter_performed": False,
        "rescue_performed": False,
        "selection_reopened": False,
        "scales": {},
    }
    pair_output: list[dict[str, Any]] = []

    for scale in SCALE_ORDER:
        rows = validate_raw_bundle(scale)
        task, task_pairs = analyze_contrast(rows, "task_matched")
        common, common_pairs = analyze_contrast(rows, "common_p3_p5")
        contra = contra_pair_map(scale)

        pair_ids = sorted(task_pairs)
        require(set(pair_ids) == set(contra), f"CONTRA_PAIR_COVERAGE:{scale}")

        lm = [task_pairs[p]["pair_mean"] for p in pair_ids]
        cm = [contra[p] for p in pair_ids]
        sign_agreement = float(
            np.mean(
                np.sign(np.asarray(lm, dtype=np.float64))
                == np.sign(np.asarray(cm, dtype=np.float64))
            )
        )

        task["contra_comparison"] = {
            "contra_forward_equivalent": descriptive(cm),
            "pearson": pearson(lm, cm),
            "spearman": spearman(lm, cm),
            "pair_sign_agreement_fraction": sign_agreement,
        }

        analysis["scales"][scale] = {
            "task_matched_planes": list(VANILLA[scale]["task"]),
            "common_planes": ["P3", "P5"],
            "task_matched": task,
            "common_p3_p5": common,
        }

        for pair in pair_ids:
            pair_output.append({
                "scale": scale,
                "source_pair_id": pair,
                "LM_TASK_MATCHED": task_pairs[pair]["pair_mean"],
                "LM_TASK_MATCHED_C0_SHAM": task_pairs[pair]["C0_SHAM"],
                "LM_TASK_MATCHED_C2_NAME": task_pairs[pair]["C2_NAME"],
                "LM_COMMON_P3_P5": common_pairs[pair]["pair_mean"],
                "LM_COMMON_P3_P5_C0_SHAM": common_pairs[pair]["C0_SHAM"],
                "LM_COMMON_P3_P5_C2_NAME": common_pairs[pair]["C2_NAME"],
                "ContraMamba_forward_equivalent": contra[pair],
            })

    lm_vector = [
        mean_sign(
            float(
                analysis["scales"][scale]["task_matched"]["pair_endpoint"]["mean"]
            )
        )
        for scale in SCALE_ORDER
    ]
    contra_vector = ["+", "+", "-", "-"]

    analysis["pattern_diagnostics"] = {
        "scale_order": list(SCALE_ORDER),
        "TASK_MATCHED_SIGN_VECTOR_LM": lm_vector,
        "FROZEN_CONTRAMAMBA_SIGN_VECTOR": contra_vector,
        "FULL_TASK_MATCHED_SIGN_CONCORDANCE": lm_vector == contra_vector,
        "POSITIVE_SCALE_SIGN_PRESERVATION": lm_vector[:2] == contra_vector[:2],
        "NEGATIVE_SCALE_SIGN_PRESERVATION": lm_vector[2:] == contra_vector[2:],
        "aggregate_sign_opposite_at_all_four_sampled_scales": all(
            a != b and a != "0" and b != "0"
            for a, b in zip(lm_vector, contra_vector, strict=True)
        ),
    }

    analysis["interpretation"] = {
        "primary":
            "The frozen ContraMamba four-scale TASK_MATCHED sign structure is not "
            "preserved by the pretrained vanilla-Mamba LM objective.",
        "supported_scope":
            "The existing native-Mamba geometry findings remain valid, while the "
            "ContraMamba Delta-L reversal is better interpreted as objective- or "
            "task-functional coupling to the frozen native geometry rather than "
            "as the same functional sign reversal already present under the "
            "pretrained LM objective.",
        "anti_inversion_boundary":
            "Aggregate signs are opposite at all four sampled scales, but pair-level "
            "Pearson/Spearman associations are not uniformly negative; therefore this "
            "does not establish a simple rowwise inversion map between objectives.",
        "no_scaling_law": True,
        "no_threshold_claim": True,
        "no_plane_homology_claim": True,
        "no_new_p_value": True,
    }

    return analysis, pair_output


def report_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# ContraMamba Gen4 Vanilla-Mamba LM Functional Control",
        "",
        "## Status",
        "",
        "`STATIC_PROSPECTIVE_FUNCTIONAL_CONTROL_ANALYSIS`",
        "",
        "This analysis opens the four prospectively frozen vanilla-LM raw bundles "
        "only after all four were independently validated and frozen.",
        "",
        "No training, model execution, intervention forward, row filtering, rescue, "
        "re-selection, or p-value is introduced here.",
        "",
        "## Primary result",
        "",
        "The frozen ContraMamba four-scale TASK_MATCHED sign vector is `+,+,-,-`.",
        "",
        "The pretrained vanilla-Mamba LM TASK_MATCHED sign vector is `-,-,+,+`.",
        "",
        "Therefore `FULL_TASK_MATCHED_SIGN_CONCORDANCE = False`.",
        "",
        "| Scale | Vanilla LM mean | Contra forward-equivalent mean | Pearson | Spearman | Pair sign agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for scale in SCALE_ORDER:
        task = analysis["scales"][scale]["task_matched"]
        lines.append(
            "| {scale} | {lm:+.12g} | {cm:+.12g} | {p:+.4f} | {s:+.4f} | {a:.4f} |".format(
                scale=scale,
                lm=task["pair_endpoint"]["mean"],
                cm=task["contra_comparison"]["contra_forward_equivalent"]["mean"],
                p=task["contra_comparison"]["pearson"],
                s=task["contra_comparison"]["spearman"],
                a=task["contra_comparison"]["pair_sign_agreement_fraction"],
            )
        )

    lines.extend([
        "",
        "## Cell localization",
        "",
        "| Scale | C0 mean | C2 mean |",
        "|---|---:|---:|",
    ])
    for scale in SCALE_ORDER:
        task = analysis["scales"][scale]["task_matched"]
        lines.append(
            "| {scale} | {c0:+.12g} | {c2:+.12g} |".format(
                scale=scale,
                c0=task["by_cell"]["C0_SHAM"]["mean"],
                c2=task["by_cell"]["C2_NAME"]["mean"],
            )
        )

    lines.extend([
        "",
        "Both cells share the aggregate TASK_MATCHED sign at every sampled scale; "
        "the four-scale pattern is therefore not localized to only C0_SHAM or C2_NAME.",
        "",
        "## Geometry / functional diagnostics",
        "",
        "| Scale | mean cosine gap | corr(component norm, cosine gap) | NORM_GAP |",
        "|---|---:|---:|---:|",
    ])
    for scale in SCALE_ORDER:
        d = analysis["scales"][scale]["task_matched"]["row_diagnostics"]
        lines.append(
            "| {scale} | {h:+.12g} | {c:+.4f} | {n:+.12g} |".format(
                scale=scale,
                h=d["mean_cosine_gap"],
                c=d["corr_component_norm_cosine_gap"],
                n=d["NORM_GAP"],
            )
        )

    lines.extend([
        "",
        "At 370M, the vanilla-LM mean cosine gap is positive while the mean endpoint "
        "is negative, with adverse component-norm/gap dependence. At 790M both the "
        "mean cosine gap and endpoint are negative. At 1.4B and 2.8B the TASK_MATCHED "
        "mean cosine gap and endpoint are positive.",
        "",
        "## COMMON_P3_P5 sensitivity",
        "",
        "| Scale | COMMON_P3_P5 mean | Sign |",
        "|---|---:|:---:|",
    ])
    for scale in SCALE_ORDER:
        v = analysis["scales"][scale]["common_p3_p5"]["pair_endpoint"]["mean"]
        lines.append(f"| {scale} | {v:+.12g} | {mean_sign(float(v))} |")

    lines.extend([
        "",
        "The common contrast gives `-,-,-,+`. The 1.4B result is therefore "
        "contrast-specific: TASK_MATCHED P5-P4 is positive while common P3-P5 is "
        "negative. This sensitivity does not alter the primary comparison, whose "
        "TASK_MATCHED contrasts were frozen prospectively.",
        "",
        "## Scientific interpretation",
        "",
        "The vanilla-LM control does not reproduce the ContraMamba `+,+,-,-` "
        "functional sign structure. The native Mamba-side geometry evidence remains "
        "unchanged, but the ContraMamba Delta-L reversal should not be described as "
        "the same functional reversal already intrinsic to the pretrained LM objective.",
        "",
        "Instead, the combined evidence supports an objective-conditioned functional "
        "coupling account: scale reorganizes the pretrained native-state substrate, "
        "and the downstream task gradient reads that substrate differently from the "
        "pretrained next-token LM gradient.",
        "",
        "The exact opposite aggregate signs at all four sampled scales must not be "
        "promoted to a simple inversion law. Pair-level Pearson/Spearman associations "
        "are not uniformly negative, so there is no rowwise one-to-one sign inversion.",
        "",
        "## Paper-facing boundary",
        "",
        "Appropriate claim: objective-dependent functional readout / repurposing of "
        "scale-reorganized native Mamba geometry.",
        "",
        "Do not claim: universal vanilla-Mamba sign reversal, continuous scaling law, "
        "parameter-count threshold, semantic homology of same-numbered planes, or "
        "simple Contra-vs-LM inversion.",
        "",
        f"`RAW_FREEZE_HEAD = {RAW_FREEZE_HEAD}`",
        f"`PLAN_FREEZE_COMMIT = {PLAN_FREEZE_COMMIT}`",
        "",
    ])
    return "\n".join(lines)


def write_outputs(output_dir: Path, analysis: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]]) -> None:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / ANALYSIS_FILE).write_bytes(
        (json.dumps(analysis, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    )
    (output_dir / PAIR_FILE).write_bytes(
        b"".join(
            (json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            for row in pairs
        )
    )
    (output_dir / REPORT_FILE).write_bytes(report_markdown(analysis).encode("utf-8"))

    manifest = {
        "result": RESULT,
        "raw_freeze_head": RAW_FREEZE_HEAD,
        "plan_freeze_commit": PLAN_FREEZE_COMMIT,
        "scale_order": list(SCALE_ORDER),
        "pair_count_per_scale": N,
        "pair_output_rows": len(pairs),
        "inferential_test_performed": False,
        "p_value_count": 0,
        "row_filter_performed": False,
        "rescue_performed": False,
        "selection_reopened": False,
        "output_files": [ANALYSIS_FILE, PAIR_FILE, REPORT_FILE],
    }
    (output_dir / MANIFEST_FILE).write_bytes(
        (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
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
    analysis, pairs = analyze_all()
    write_outputs(args.output_dir, analysis, pairs)

    pattern = analysis["pattern_diagnostics"]
    print("RESULT=" + RESULT)
    print("TASK_MATCHED_SIGN_VECTOR_LM=" + ",".join(pattern["TASK_MATCHED_SIGN_VECTOR_LM"]))
    print("FROZEN_CONTRAMAMBA_SIGN_VECTOR=" + ",".join(pattern["FROZEN_CONTRAMAMBA_SIGN_VECTOR"]))
    print("FULL_TASK_MATCHED_SIGN_CONCORDANCE=" + str(pattern["FULL_TASK_MATCHED_SIGN_CONCORDANCE"]))
    print("AGGREGATE_SIGN_OPPOSITE_ALL_FOUR=" + str(pattern["aggregate_sign_opposite_at_all_four_sampled_scales"]))
    print("P_VALUE_COUNT=0")
    print("ROW_FILTER_PERFORMED=False")
    print("RESCUE_PERFORMED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
