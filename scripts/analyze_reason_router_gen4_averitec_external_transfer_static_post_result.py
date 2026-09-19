#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    analyze_reason_router_gen4_averitec_130m370m_external_transfer
    as primary_analysis,
)

EXPECTED_EXECUTION_HEAD = "c5b470f0601c0d287218bf39cefc0e681b3f81fd"
EXPECTED_RECOVERY_RUN = (
    "g4k-averitec-external-transfer-130m370m-dev462-"
    "c5b470f-postcheck-recovery2"
)
EXPECTED_ANALYSIS_SHA256 = (
    "0c8e5fff026fbc243f7344d9e2af57cd00a6597ce7c468f0fd8b604dbe7d2a7f"
)
SCALES = ("mamba130m", "mamba370m")
CONDITIONS = (
    "native",
    "dominant_neutralized",
    "dominant_control",
)
RESULT = "PASS_AVERITEC_EXTERNAL_TRANSFER_STATIC_POST_RESULT_ANALYSIS"


class StaticAnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StaticAnalysisError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(value: Any, label: str) -> float:
    out = float(value)
    require(math.isfinite(out), f"NONFINITE:{label}")
    return out


def descriptive(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and array.size > 0, "DESCRIPTIVE_EMPTY")
    require(bool(np.isfinite(array).all()), "DESCRIPTIVE_NONFINITE")
    sorted_values = np.sort(array)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "q25": float(np.quantile(sorted_values, 0.25)),
        "q75": float(np.quantile(sorted_values, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "fraction_positive": float(np.mean(array > 0.0)),
        "fraction_negative": float(np.mean(array < 0.0)),
        "fraction_zero": float(np.mean(array == 0.0)),
    }


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    require(xa.shape == ya.shape and xa.ndim == 1, "PEARSON_SHAPE")
    if xa.size < 2:
        return None
    sx = float(np.std(xa))
    sy = float(np.std(ya))
    if sx == 0.0 or sy == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def read_primary(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis_path = run_dir / "external_transfer_analysis.json"
    recovery_path = run_dir / "recovery_manifest.json"
    require(analysis_path.is_file(), "PRIMARY_ANALYSIS_MISSING")
    require(recovery_path.is_file(), "RECOVERY_MANIFEST_MISSING")

    require(
        sha256_file(analysis_path) == EXPECTED_ANALYSIS_SHA256,
        "PRIMARY_ANALYSIS_SHA256",
    )

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))

    require(
        analysis["result"]
        == "PASS_AVERITEC_GOLD_EVIDENCE_EXTERNAL_TRANSFER_ANALYSIS",
        "PRIMARY_RESULT",
    )
    require(analysis["N_per_scale"] == 462, "PRIMARY_N")
    require(analysis["primary_scale_order"] == list(SCALES), "PRIMARY_SCALES")
    require(analysis["primary_p_value_count"] == 2, "PRIMARY_P_COUNT")
    require(
        analysis["scientific_conclusion"]
        == "CROSS_SCALE_AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_"
        "THROUGH_370M_SUPPORTED",
        "PRIMARY_CONCLUSION",
    )
    require(analysis["selection_reopened"] is False, "PRIMARY_SELECTION")
    require(analysis["rescue_performed"] is False, "PRIMARY_RESCUE")
    require(analysis["additional_primary_p_values"] is False, "PRIMARY_EXTRA_P")
    require(analysis["mamba14b_in_family"] is False, "PRIMARY_14B")

    require(
        recovery["result"]
        == "PASS_AVERITEC_EXTERNAL_TRANSFER_POSTCHECK_RECOVERY",
        "RECOVERY_RESULT",
    )
    require(
        recovery["execution_head"] == EXPECTED_EXECUTION_HEAD,
        "RECOVERY_HEAD",
    )
    require(
        recovery["recovery_analysis_sha256"] == EXPECTED_ANALYSIS_SHA256,
        "RECOVERY_ANALYSIS_SHA",
    )
    require(recovery["analysis_byte_identity"] is True, "RECOVERY_BYTE_IDENTITY")
    require(
        recovery["raw_artifacts_copied_byte_identically"] is True,
        "RECOVERY_RAW_IDENTITY",
    )
    require(
        recovery["recovery_scientific_full_model_forward_count_added"] == 0,
        "RECOVERY_FORWARD_COUNT",
    )
    require(recovery["additional_primary_p_values"] is False, "RECOVERY_EXTRA_P")
    return analysis, recovery


def load_scale_rows(
    run_dir: Path,
    *,
    scale: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = primary_analysis.validate_scale(
        run_dir / scale,
        scale=scale,
    )
    require(summary["execution_head"] == EXPECTED_EXECUTION_HEAD, "SCALE_HEAD")
    return rows, summary


def build_item_records(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], Mapping[str, Any]] = {
        (str(row["example_id"]), str(row["condition"])): row
        for row in rows
    }
    item_ids = sorted({str(row["example_id"]) for row in rows})
    require(len(item_ids) == 462, "ITEM_N")

    records: list[dict[str, Any]] = []
    for item_id in item_ids:
        native = by[(item_id, "native")]
        neutral = by[(item_id, "dominant_neutralized")]
        control = by[(item_id, "dominant_control")]

        m_native = finite_float(native["correct_class_logit_margin"], "M_NATIVE")
        m_neutral = finite_float(
            neutral["correct_class_logit_margin"],
            "M_NEUTRAL",
        )
        m_control = finite_float(
            control["correct_class_logit_margin"],
            "M_CONTROL",
        )
        a_sel = m_native - m_neutral
        b_ctrl = m_control - m_neutral
        d_ext = m_native - m_control
        require(abs((a_sel - b_ctrl) - d_ext) <= 1.0e-12, "DECOMPOSITION")

        pred_native = int(native["prediction_id"])
        pred_neutral = int(neutral["prediction_id"])
        pred_control = int(control["prediction_id"])
        correct_id = int(native["correct_label_id"])
        native_correct = bool(native["is_correct"])
        neutral_correct = bool(neutral["is_correct"])
        control_correct = bool(control["is_correct"])

        if pred_native == pred_control:
            control_flip_class = "same_prediction"
        elif native_correct and not control_correct:
            control_flip_class = "native_correct_control_wrong"
        elif (not native_correct) and control_correct:
            control_flip_class = "native_wrong_control_correct"
        else:
            control_flip_class = "both_wrong_different_prediction"

        if pred_native == pred_neutral:
            neutral_flip_class = "same_prediction"
        elif native_correct and not neutral_correct:
            neutral_flip_class = "native_correct_neutral_wrong"
        elif (not native_correct) and neutral_correct:
            neutral_flip_class = "native_wrong_neutral_correct"
        else:
            neutral_flip_class = "both_wrong_different_prediction"

        records.append({
            "example_id": item_id,
            "averitec_dev_index": int(native["averitec_dev_index"]),
            "source_label": str(native["source_label"]),
            "correct_label": str(native["correct_label"]),
            "M_native": m_native,
            "M_neutralized": m_neutral,
            "M_control": m_control,
            "A_sel": a_sel,
            "B_ctrl": b_ctrl,
            "D_EXT": d_ext,
            "delta_q_native_minus_control":
                finite_float(native["q_authorized"], "Q_NATIVE")
                - finite_float(control["q_authorized"], "Q_CONTROL"),
            "native_correct": native_correct,
            "neutral_correct": neutral_correct,
            "control_correct": control_correct,
            "prediction_flip_native_vs_control":
                pred_native != pred_control,
            "prediction_flip_native_vs_neutral":
                pred_native != pred_neutral,
            "control_flip_class": control_flip_class,
            "neutral_flip_class": neutral_flip_class,
        })
    return records


def grouped_descriptive(
    records: Sequence[Mapping[str, Any]],
    *,
    key: str,
    value: str = "D_EXT",
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(float(record[value]))
    return {
        group: descriptive(values)
        for group, values in sorted(grouped.items())
    }


def scale_static(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    d = [float(r["D_EXT"]) for r in records]
    a = [float(r["A_sel"]) for r in records]
    b = [float(r["B_ctrl"]) for r in records]
    native_margin = [float(r["M_native"]) for r in records]
    q_delta = [float(r["delta_q_native_minus_control"]) for r in records]

    return {
        "overall": {
            "D_EXT": descriptive(d),
            "A_sel": descriptive(a),
            "B_ctrl": descriptive(b),
        },
        "by_source_label": grouped_descriptive(
            records,
            key="source_label",
        ),
        "by_native_correct": grouped_descriptive(
            records,
            key="native_correct",
        ),
        "by_control_flip": grouped_descriptive(
            records,
            key="prediction_flip_native_vs_control",
        ),
        "by_control_flip_class": grouped_descriptive(
            records,
            key="control_flip_class",
        ),
        "by_neutral_flip": grouped_descriptive(
            records,
            key="prediction_flip_native_vs_neutral",
        ),
        "by_neutral_flip_class": grouped_descriptive(
            records,
            key="neutral_flip_class",
        ),
        "prediction_counts": {
            "native_correct": int(sum(bool(r["native_correct"]) for r in records)),
            "neutral_correct": int(sum(bool(r["neutral_correct"]) for r in records)),
            "control_correct": int(sum(bool(r["control_correct"]) for r in records)),
            "native_vs_control_flip": int(
                sum(bool(r["prediction_flip_native_vs_control"]) for r in records)
            ),
            "native_vs_neutral_flip": int(
                sum(bool(r["prediction_flip_native_vs_neutral"]) for r in records)
            ),
            "control_flip_class": dict(sorted(Counter(
                str(r["control_flip_class"]) for r in records
            ).items())),
            "neutral_flip_class": dict(sorted(Counter(
                str(r["neutral_flip_class"]) for r in records
            ).items())),
        },
        "descriptive_correlations": {
            "native_margin_vs_D_EXT_pearson": pearson(native_margin, d),
            "q_delta_vs_D_EXT_pearson": pearson(q_delta, d),
            "A_sel_vs_D_EXT_pearson": pearson(a, d),
            "B_ctrl_vs_D_EXT_pearson": pearson(b, d),
        },
    }


def cross_scale(
    records130: Sequence[Mapping[str, Any]],
    records370: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by130 = {str(r["example_id"]): r for r in records130}
    by370 = {str(r["example_id"]): r for r in records370}
    require(set(by130) == set(by370), "CROSS_SCALE_ITEM_SET")

    ids = sorted(by130)
    d130 = np.asarray([float(by130[i]["D_EXT"]) for i in ids])
    d370 = np.asarray([float(by370[i]["D_EXT"]) for i in ids])
    sign130 = np.sign(d130)
    sign370 = np.sign(d370)

    both_positive = (d130 > 0.0) & (d370 > 0.0)
    both_negative = (d130 < 0.0) & (d370 < 0.0)
    opposite = sign130 * sign370 < 0.0
    same_sign_nonzero = sign130 == sign370

    label_pairs: dict[str, dict[str, Any]] = {}
    for label in sorted({str(by130[i]["source_label"]) for i in ids}):
        label_ids = [i for i in ids if str(by130[i]["source_label"]) == label]
        x = [float(by130[i]["D_EXT"]) for i in label_ids]
        y = [float(by370[i]["D_EXT"]) for i in label_ids]
        label_pairs[label] = {
            "n": len(label_ids),
            "mean_D_130m": float(np.mean(x)),
            "mean_D_370m": float(np.mean(y)),
            "pearson": pearson(x, y),
            "both_positive_fraction": float(np.mean(
                (np.asarray(x) > 0.0) & (np.asarray(y) > 0.0)
            )),
        }

    return {
        "n": len(ids),
        "D_EXT_pearson": pearson(d130.tolist(), d370.tolist()),
        "D_EXT_sign_agreement_fraction": float(np.mean(same_sign_nonzero)),
        "both_positive_fraction": float(np.mean(both_positive)),
        "both_negative_fraction": float(np.mean(both_negative)),
        "opposite_sign_fraction": float(np.mean(opposite)),
        "mean_D_130m_minus_370m": float(np.mean(d130 - d370)),
        "by_source_label": label_pairs,
    }


def interpretation_flags(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    s130 = result["scale_descriptives"]["mamba130m"]
    s370 = result["scale_descriptives"]["mamba370m"]

    return {
        "primary_supported_at_both_scales": True,
        "effect_size_context": "small standardized effects at both scales",
        "label_heterogeneity_present": (
            len({
                math.copysign(1.0, float(v["mean"]))
                if float(v["mean"]) != 0.0 else 0.0
                for v in s130["by_source_label"].values()
            }) > 1
            or len({
                math.copysign(1.0, float(v["mean"]))
                if float(v["mean"]) != 0.0 else 0.0
                for v in s370["by_source_label"].values()
            }) > 1
        ),
        "benchmark_accuracy_claim_supported": False,
        "uniform_label_transfer_claim_supported": False,
        "unique_mediation_claim_supported": False,
        "subset_promotion_performed": False,
        "new_inferential_test_performed": False,
    }


def markdown(result: Mapping[str, Any]) -> str:
    p = result["primary_context"]
    s130 = result["scale_descriptives"]["mamba130m"]
    s370 = result["scale_descriptives"]["mamba370m"]
    cross = result["cross_scale"]

    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        if isinstance(x, bool):
            return str(x)
        if isinstance(x, int):
            return str(x)
        return format(float(x), ".8g")

    lines = [
        "# AVeriTeC External Transfer — Static Post-Result Analysis",
        "",
        "## Status",
        "",
        f"`{RESULT}`",
        "",
        "This analysis is descriptive only. It adds zero model forwards and zero new p-values.",
        "",
        "## Frozen primary context",
        "",
        "| Scale | mean D_EXT | Holm p | Cohen dz | supported |",
        "|---|---:|---:|---:|:---:|",
    ]
    for scale in SCALES:
        q = p[scale]
        lines.append(
            f"| {scale} | {fmt(q['mean_D_EXT'])} | {fmt(q['p_holm'])} | "
            f"{fmt(q['cohen_dz'])} | {q['supported_holm']} |"
        )

    lines += [
        "",
        "## Descriptive decomposition",
        "",
        "| Scale | mean A_sel | mean B_ctrl | mean D_EXT | native→control flips |",
        "|---|---:|---:|---:|---:|",
    ]
    for scale, s in (("mamba130m", s130), ("mamba370m", s370)):
        lines.append(
            f"| {scale} | {fmt(s['overall']['A_sel']['mean'])} | "
            f"{fmt(s['overall']['B_ctrl']['mean'])} | "
            f"{fmt(s['overall']['D_EXT']['mean'])} | "
            f"{s['prediction_counts']['native_vs_control_flip']} |"
        )

    lines += [
        "",
        "## Source-label heterogeneity",
        "",
        "| Scale | Source label | n | mean D_EXT | fraction positive |",
        "|---|---|---:|---:|---:|",
    ]
    for scale, s in (("mamba130m", s130), ("mamba370m", s370)):
        for label, d in s["by_source_label"].items():
            lines.append(
                f"| {scale} | {label} | {d['n']} | {fmt(d['mean'])} | "
                f"{fmt(d['fraction_positive'])} |"
            )

    lines += [
        "",
        "## Descriptive correlations",
        "",
        "| Scale | native margin vs D | q-delta vs D | A_sel vs D | B_ctrl vs D |",
        "|---|---:|---:|---:|---:|",
    ]
    for scale, s in (("mamba130m", s130), ("mamba370m", s370)):
        c = s["descriptive_correlations"]
        lines.append(
            f"| {scale} | {fmt(c['native_margin_vs_D_EXT_pearson'])} | "
            f"{fmt(c['q_delta_vs_D_EXT_pearson'])} | "
            f"{fmt(c['A_sel_vs_D_EXT_pearson'])} | "
            f"{fmt(c['B_ctrl_vs_D_EXT_pearson'])} |"
        )

    lines += [
        "",
        "## Cross-scale item alignment",
        "",
        f"- D_EXT Pearson: `{fmt(cross['D_EXT_pearson'])}`",
        f"- sign agreement fraction: `{fmt(cross['D_EXT_sign_agreement_fraction'])}`",
        f"- both-positive fraction: `{fmt(cross['both_positive_fraction'])}`",
        f"- opposite-sign fraction: `{fmt(cross['opposite_sign_fraction'])}`",
        "",
        "## Interpretation boundary",
        "",
        "- The preregistered primary endpoint is supported at both 130M and 370M.",
        "- This static analysis does not add or alter the primary family.",
        "- Source-label breakdowns are descriptive and are not promoted to new inferential claims.",
        "- The result supports external transfer in correct-class margin, not benchmark accuracy improvement.",
        "- It does not establish uniform transfer across labels or a unique causal mediator.",
        "",
    ]
    return "\n".join(lines)


def analyze(run_dir: Path) -> dict[str, Any]:
    require(run_dir.name == EXPECTED_RECOVERY_RUN, "RUN_NAME")
    primary, recovery = read_primary(run_dir)

    all_records: dict[str, list[dict[str, Any]]] = {}
    scale_results: dict[str, Any] = {}
    for scale in SCALES:
        rows, _summary = load_scale_rows(run_dir, scale=scale)
        records = build_item_records(rows)
        all_records[scale] = records
        scale_results[scale] = scale_static(records)

    result: dict[str, Any] = {
        "schema_version":
            "gen4-averitec-external-transfer-static-post-result-analysis-v1",
        "result": RESULT,
        "source_recovery_run": EXPECTED_RECOVERY_RUN,
        "execution_head": EXPECTED_EXECUTION_HEAD,
        "source_analysis_sha256": EXPECTED_ANALYSIS_SHA256,
        "source_primary_scientific_conclusion":
            primary["scientific_conclusion"],
        "primary_context": {
            scale: {
                "mean_D_EXT":
                    float(primary["scale_results"][scale]["primary"]["mean"]),
                "p_holm":
                    float(primary["scale_results"][scale]["primary"]["p_holm"]),
                "cohen_dz":
                    float(primary["scale_results"][scale]["primary"]["cohen_dz"]),
                "supported_holm":
                    bool(primary["scale_results"][scale]["primary"]["supported_holm"]),
            }
            for scale in SCALES
        },
        "scale_descriptives": scale_results,
        "cross_scale": cross_scale(
            all_records["mamba130m"],
            all_records["mamba370m"],
        ),
        "new_model_forward_count": 0,
        "new_p_value_count": 0,
        "primary_family_reopened": False,
        "selection_reopened": False,
        "rescue_performed": False,
        "mamba14b_added": False,
        "subset_promotion_performed": False,
        "interpretation_flags": {},
    }
    result["interpretation_flags"] = interpretation_flags(result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require(not args.output_json.exists(), "OUTPUT_JSON_COLLISION")
    require(not args.output_md.exists(), "OUTPUT_MD_COLLISION")
    result = analyze(args.run_dir)

    args.output_json.write_text(
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.output_md.write_text(
        markdown(result),
        encoding="utf-8",
        newline="\n",
    )

    print("RESULT=" + result["result"])
    print("NEW_MODEL_FORWARD_COUNT=0")
    print("NEW_P_VALUE_COUNT=0")
    print("PRIMARY_FAMILY_REOPENED=False")
    print("SELECTION_REOPENED=False")
    print("RESCUE_PERFORMED=False")

    for scale in SCALES:
        s = result["scale_descriptives"][scale]
        print(scale.upper() + "_MEAN_D_EXT=" + format(
            float(s["overall"]["D_EXT"]["mean"]), ".17g"
        ))
        print(scale.upper() + "_MEAN_A_SEL=" + format(
            float(s["overall"]["A_sel"]["mean"]), ".17g"
        ))
        print(scale.upper() + "_MEAN_B_CTRL=" + format(
            float(s["overall"]["B_ctrl"]["mean"]), ".17g"
        ))
        print(scale.upper() + "_NATIVE_CONTROL_FLIPS=" + str(
            s["prediction_counts"]["native_vs_control_flip"]
        ))
        print(scale.upper() + "_NATIVE_MARGIN_D_PEARSON=" + json.dumps(
            s["descriptive_correlations"]["native_margin_vs_D_EXT_pearson"]
        ))
        print(scale.upper() + "_Q_D_PEARSON=" + json.dumps(
            s["descriptive_correlations"]["q_delta_vs_D_EXT_pearson"]
        ))
        print(scale.upper() + "_BY_LABEL=" + json.dumps(
            {
                k: {
                    "n": v["n"],
                    "mean": v["mean"],
                    "fraction_positive": v["fraction_positive"],
                }
                for k, v in s["by_source_label"].items()
            },
            sort_keys=True,
        ))

    c = result["cross_scale"]
    print("CROSS_SCALE_D_PEARSON=" + json.dumps(c["D_EXT_pearson"]))
    print(
        "CROSS_SCALE_SIGN_AGREEMENT="
        + format(float(c["D_EXT_sign_agreement_fraction"]), ".17g")
    )
    print(
        "CROSS_SCALE_BOTH_POSITIVE="
        + format(float(c["both_positive_fraction"]), ".17g")
    )
    print(
        "CROSS_SCALE_OPPOSITE_SIGN="
        + format(float(c["opposite_sign_fraction"]), ".17g")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
