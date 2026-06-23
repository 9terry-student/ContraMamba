"""Analyze Stage15 slot-sensitivity predictions.

This script merges Stage15 probe rows with one or more prediction JSON files and
summarizes where false entitlement appears, which auxiliary signals move, and
whether temporal erasure changes model behavior.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE = ROOT / "data" / "stage15_slot_sensitivity_probe.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results"
ENTITLED = {"REFUTE", "SUPPORT"}
LABEL_ORDER = ("REFUTE", "NOT_ENTITLED", "SUPPORT")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def expand_prediction_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_predictions(paths: Sequence[Path]) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        predictions = payload.get("predictions")
        if not isinstance(predictions, list):
            raise ValueError(f"prediction file has no predictions list: {path}")
        seed = payload.get("metadata", {}).get("seed")
        for pred in predictions:
            row = dict(pred)
            row["_prediction_file"] = str(path)
            row["_seed"] = seed
            loaded.append(row)
    return loaded


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def support_prob(row: dict[str, Any]) -> float | None:
    probs = row.get("final_probs")
    if isinstance(probs, list) and len(probs) >= 3:
        return as_float(probs[2])
    return None


def merge_probe_predictions(
    probe_rows: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    probe_by_id = {row["id"]: row for row in probe_rows}
    merged: list[dict[str, Any]] = []
    missing = 0
    for pred in predictions:
        probe = probe_by_id.get(pred.get("id"))
        if probe is None:
            missing += 1
            continue
        row = dict(probe)
        row.update(
            {
                "pred_final_label": pred.get("pred_final_label"),
                "gold_final_label": pred.get("gold_final_label")
                or probe.get("final_label")
                or probe.get("label"),
                "final_probs": pred.get("final_probs"),
                "support_prob": support_prob(pred),
                "entitlement_prob": pred.get("entitlement_prob"),
                "frame_prob": pred.get("frame_prob"),
                "predicate_coverage_prob": pred.get("predicate_coverage_prob"),
                "sufficiency_prob": pred.get("sufficiency_prob"),
                "polarity_margin": pred.get("polarity_margin"),
                "_prediction_file": pred.get("_prediction_file"),
                "_seed": pred.get("_seed"),
            }
        )
        merged.append(row)
    if not merged:
        raise ValueError(
            f"no predictions matched probe ids; unmatched prediction rows: {missing}"
        )
    return merged


def prediction_distribution(rows: Sequence[dict[str, Any]]) -> str:
    counts = Counter(row.get("pred_final_label") for row in rows)
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def summarize_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(
        1
        for row in rows
        if row.get("pred_final_label")
        == (row.get("gold_final_label") or row.get("final_label") or row.get("label"))
    )
    false_entitled = [
        row
        for row in rows
        if (row.get("gold_final_label") or row.get("final_label") or row.get("label"))
        == "NOT_ENTITLED"
        and row.get("pred_final_label") in ENTITLED
    ]
    return {
        "stage15_probe_type": rows[0].get("stage15_probe_type") if rows else "",
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "false_entitled_count": len(false_entitled),
        "false_entitled_rate": len(false_entitled) / n if n else 0.0,
        "mean_entitlement_prob": mean(
            x for row in rows if (x := as_float(row.get("entitlement_prob"))) is not None
        ),
        "mean_frame_prob": mean(
            x for row in rows if (x := as_float(row.get("frame_prob"))) is not None
        ),
        "mean_predicate_coverage_prob": mean(
            x
            for row in rows
            if (x := as_float(row.get("predicate_coverage_prob"))) is not None
        ),
        "mean_sufficiency_prob": mean(
            x for row in rows if (x := as_float(row.get("sufficiency_prob"))) is not None
        ),
        "mean_polarity_margin": mean(
            x for row in rows if (x := as_float(row.get("polarity_margin"))) is not None
        ),
        "prediction_distribution": prediction_distribution(rows),
    }


def group_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row.get("stage15_probe_type", "")].append(row)
    return [summarize_group(by_type[key]) for key in sorted(by_type)]


def temporal_pairs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_file_source: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("_prediction_file")), str(row.get("stage15_source_id")))
        by_file_source[key][str(row.get("stage15_probe_type"))] = row

    pairs: list[dict[str, Any]] = []
    for (prediction_file, source_id), variants in sorted(by_file_source.items()):
        mismatch = variants.get("temporal_mismatch")
        erased = variants.get("temporal_erased")
        if mismatch is None or erased is None:
            continue
        mismatch_support = as_float(mismatch.get("support_prob"))
        erased_support = as_float(erased.get("support_prob"))
        pairs.append(
            {
                "prediction_file": prediction_file,
                "seed": mismatch.get("_seed"),
                "stage15_source_id": source_id,
                "mismatch_pred": mismatch.get("pred_final_label"),
                "erased_pred": erased.get("pred_final_label"),
                "mismatch_support_prob": mismatch_support,
                "erased_support_prob": erased_support,
                "support_prob_delta_erased_minus_mismatch": (
                    erased_support - mismatch_support
                    if erased_support is not None and mismatch_support is not None
                    else ""
                ),
                "mismatch_frame_prob": mismatch.get("frame_prob"),
                "erased_frame_prob": erased.get("frame_prob"),
                "mismatch_entitlement_prob": mismatch.get("entitlement_prob"),
                "erased_entitlement_prob": erased.get("entitlement_prob"),
            }
        )
    return pairs


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(metrics: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Stage15 Slot Sensitivity Summary",
        "",
        "Stage15 is a diagnostic probe, not a new model or full OOD benchmark.",
        "",
        "## Group metrics",
        "",
        "| probe type | n | accuracy | false-entitled | false-entitled rate | mean entitlement | mean frame | mean predicate | mean sufficiency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    metric_by_type = {row["stage15_probe_type"]: row for row in metrics}
    for key in sorted(metric_by_type):
        row = metric_by_type[key]
        lines.append(
            "| {stage15_probe_type} | {n} | {accuracy:.3f} | {false_entitled_count} | "
            "{false_entitled_rate:.3f} | {mean_entitlement_prob:.3f} | "
            "{mean_frame_prob:.3f} | {mean_predicate_coverage_prob:.3f} | "
            "{mean_sufficiency_prob:.3f} |".format(**row)
        )

    lines.extend(["", "## Temporal erasure sensitivity", ""])
    if pairs:
        deltas = [
            as_float(row.get("support_prob_delta_erased_minus_mismatch"))
            for row in pairs
        ]
        deltas = [x for x in deltas if x is not None]
        lines.append(
            f"Temporal mismatch/erasure paired rows: {len(pairs)}. "
            f"Mean SUPPORT-probability delta, erased minus mismatch: {mean(deltas):.3f}."
        )
        lines.append(
            "If this delta is small while temporal mismatch remains over-entitled, "
            "the model is not using the temporal slot strongly."
        )
    else:
        lines.append("No temporal mismatch/erasure pairs were available.")

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "- If `temporal_mismatch` and `temporal_erased` have similar SUPPORT probabilities, treat this as temporal slot insensitivity.",
            "- If `frame_location_mismatch` or `frame_role_mismatch` retain high `frame_prob`, treat this as frame-head slot insensitivity.",
            "- If `predicate_mismatch` has low `predicate_coverage_prob` but still predicts SUPPORT/REFUTE, treat this as final aggregation ignoring an auxiliary predicate warning.",
            "- These diagnostics identify failure anatomy; they do not establish broad OOD robustness.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--preds", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    probe_rows = load_jsonl(args.probe)
    prediction_paths = expand_prediction_paths(args.preds)
    predictions = load_predictions(prediction_paths)
    merged = merge_probe_predictions(probe_rows, predictions)
    metrics = group_metrics(merged)
    pairs = temporal_pairs(merged)

    output_dir = args.output_dir
    write_csv(output_dir / "stage15_slot_sensitivity_group_metrics.csv", metrics)
    write_csv(output_dir / "stage15_slot_sensitivity_examples.csv", merged)
    write_csv(output_dir / "stage15_temporal_erasure_pairs.csv", pairs)
    (output_dir / "stage15_slot_sensitivity_summary.md").write_text(
        markdown_summary(metrics, pairs),
        encoding="utf-8",
    )

    print("STAGE15_SLOT_SENSITIVITY_ANALYSIS")
    print(f"probe_rows\t{len(probe_rows)}")
    print(f"prediction_files\t{len(prediction_paths)}")
    print(f"merged_rows\t{len(merged)}")
    print(f"group_metrics\t{output_dir / 'stage15_slot_sensitivity_group_metrics.csv'}")
    print(f"examples\t{output_dir / 'stage15_slot_sensitivity_examples.csv'}")
    print(f"temporal_pairs\t{output_dir / 'stage15_temporal_erasure_pairs.csv'}")
    print(f"summary\t{output_dir / 'stage15_slot_sensitivity_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

