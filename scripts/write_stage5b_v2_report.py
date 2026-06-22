"""Write the Stage 5B v2 grouped ablation comparison report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

from scripts.aggregate_controlled_results import aggregate_groups


CLASSIFICATION = (
    ("accuracy", "final_accuracy"),
    ("macro-F1", "final_macro_f1"),
    ("frame", "frame_accuracy"),
    ("predicate", "predicate_accuracy"),
    ("polarity", "polarity_accuracy_entitled"),
    ("sufficiency", "sufficiency_accuracy"),
)
PAIRWISE = (
    ("paraphrase", "paraphrase_preserved"),
    ("predicate-pair", "predicate_disentangled"),
    ("polarity-flip", "polarity_flip_preserved_and_reversed"),
    ("deletion", "deletion_sufficiency_lower"),
    ("truncation", "truncation_sufficiency_lower"),
    ("entity", "entity_frame_lower"),
    ("event", "event_frame_lower"),
)
KEY_CONTRAST = (
    ("macro-F1", "final_macro_f1"),
    ("polarity accuracy", "polarity_accuracy_entitled"),
    ("polarity-flip", "polarity_flip_preserved_and_reversed"),
    ("predicate-pair", "predicate_disentangled"),
)


def formatted(group: dict[str, Any], metric: str) -> str:
    aggregate = group["aggregate"][metric]
    return f"{aggregate['mean']:.3f} ± {aggregate['std']:.3f}"


def summary_table(
    groups: dict[str, Any], columns: Sequence[tuple[str, str]]
) -> str:
    headers = ["config", *[label for label, _ in columns]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name, group in groups.items():
        values = [name, *[formatted(group, metric) for _, metric in columns]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _best_config(groups: dict[str, Any], metric: str) -> tuple[str, float]:
    name = max(groups, key=lambda item: groups[item]["aggregate"][metric]["mean"])
    return name, groups[name]["aggregate"][metric]["mean"]


def interpretation(groups: dict[str, Any]) -> str:
    classifier, classifier_score = _best_config(groups, "final_macro_f1")
    predicate, predicate_score = _best_config(groups, "predicate_disentangled")
    polarity, polarity_score = _best_config(
        groups, "polarity_flip_preserved_and_reversed"
    )
    diverges = classifier != predicate or classifier != polarity
    lines = [
        f"- Best final-label classifier: `{classifier}` "
        f"(macro-F1 {classifier_score:.3f}).",
        f"- Best predicate-pair consistency: `{predicate}` ({predicate_score:.3f}).",
        f"- Best polarity-flip consistency: `{polarity}` ({polarity_score:.3f}).",
        "- Final-label classification and intervention consistency "
        + ("diverge across configurations." if diverges else "select the same configuration."),
    ]

    reference_name = "v2_full4e" if "v2_full4e" in groups else predicate
    if "v2_no_predicate_contrast" in groups:
        reference = groups[reference_name]["aggregate"]["predicate_disentangled"]["mean"]
        ablated = groups["v2_no_predicate_contrast"]["aggregate"][
            "predicate_disentangled"
        ]["mean"]
        outcome = "weakens" if ablated < reference else "does not weaken"
        lines.append(
            f"- Removing predicate contrast {outcome} predicate behavior "
            f"({ablated:.3f} vs {reference:.3f})."
        )
    if "v2_no_polarity_flip" in groups:
        polarity_reference_name = (
            "v2_full4e" if "v2_full4e" in groups else polarity
        )
        reference = groups[polarity_reference_name]["aggregate"][
            "polarity_flip_preserved_and_reversed"
        ]["mean"]
        ablated = groups["v2_no_polarity_flip"]["aggregate"][
            "polarity_flip_preserved_and_reversed"
        ]["mean"]
        collapsed = reference > 0 and ablated < 0.5 * reference
        lines.append(
            "- Removing polarity-flip loss "
            + ("collapses" if collapsed else "does not collapse")
            + f" polarity-flip behavior ({ablated:.3f} vs {reference:.3f})."
        )
    return "\n".join(lines)


def markdown_report(result: dict[str, Any]) -> str:
    groups = result["groups"]
    return "\n\n".join(
        [
            "## CLASSIFICATION_SUMMARY\n\n" + summary_table(groups, CLASSIFICATION),
            "## PAIRWISE_CONSISTENCY_SUMMARY\n\n" + summary_table(groups, PAIRWISE),
            "## KEY_CONTRAST_SUMMARY\n\n" + summary_table(groups, KEY_CONTRAST),
            "## INTERPRETATION\n\n" + interpretation(groups),
        ]
    ) + "\n"


def csv_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sections = (
        ("classification", CLASSIFICATION),
        ("pairwise", PAIRWISE),
        ("key_contrast", KEY_CONTRAST),
    )
    for section, columns in sections:
        for config, group in result["groups"].items():
            for label, metric in columns:
                values = group["aggregate"][metric]
                rows.append(
                    {
                        "section": section,
                        "config": config,
                        "metric": label,
                        "mean": values["mean"],
                        "std": values["std"],
                        "formatted": formatted(group, metric),
                    }
                )
    return rows


def write_report(
    result: dict[str, Any], output_md: Path, output_csv: Path
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown_report(result), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("section", "config", "metric", "mean", "std", "formatted"),
        )
        writer.writeheader()
        writer.writerows(csv_rows(result))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/stage5b_v2_ablation_comparison.md"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/stage5b_v2_ablation_comparison.csv"),
    )
    args = parser.parse_args(argv)
    result = aggregate_groups(args.group)
    write_report(result, args.output_md, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

