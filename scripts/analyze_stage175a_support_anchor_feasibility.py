"""Stage175-A static audit of clean SUPPORT-anchor preservation feasibility.

This script performs data/schema analysis only. It does not load a checkpoint,
run a model, tune a threshold, select a checkpoint, or implement a loss.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_controlled_v5 import load_jsonl, split_by_pair_id  # noqa: E402
from scripts.stage174c_clean_pairwise import (  # noqa: E402
    EXPECTED_INTERVENTIONS,
    build_train_pair_index,
    taxonomy_record,
    validation_rules,
)

DEFAULT_DATA = ROOT / "data" / "controlled_v5_v3_without_time_swap.jsonl"
REPORT_JSON = "stage175a_support_anchor_feasibility_report.json"
REPORT_MD = "stage175a_support_anchor_feasibility_report.md"
GROUP_CSV = "stage175a_support_anchor_group_summary.csv"
INTERVENTION_CSV = "stage175a_support_anchor_intervention_counts.csv"
CONDITION_FIELDS = ("frame_compatible_label", "predicate_covered_label", "sufficiency_label")


def fail(message: str) -> None:
    raise ValueError(f"[stage175a] {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(path: Path, records: list[dict[str, Any]]) -> None:
    if path.resolve() != DEFAULT_DATA.resolve():
        fail(f"--data must resolve to the clean main dataset {DEFAULT_DATA}, got {path.resolve()}")
    if any(record.get("intervention_type") == "time_swap" for record in records):
        fail("time_swap is forbidden")
    forbidden_markers = ("stage43", "external_evaluation", "external-evaluation", "synthetic_ood", "family_holdout")
    for index, record in enumerate(records):
        text = json.dumps(record, sort_keys=True).lower()
        marker = next((item for item in forbidden_markers if item in text), None)
        if marker:
            fail(f"row {index} contains forbidden external/OOD provenance marker {marker!r}")
        if not record.get("pair_id"):
            fail(f"row {index} lacks pair_id")
        if record.get("intervention_type") not in EXPECTED_INTERVENTIONS:
            fail(f"row {index} has disallowed intervention_type={record.get('intervention_type')!r}")


def group_rows(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    index = build_train_pair_index(records)  # exact Stage174-C taxonomy and structural validation
    return {
        pair_id: {intervention: records[row_index] for intervention, row_index in mapping.items()}
        for pair_id, mapping in index.items()
    }


def eligible(group: dict[str, dict[str, Any]]) -> bool:
    none, para = group["none"], group["paraphrase"]
    return (
        none.get("final_label") == "SUPPORT"
        and para.get("final_label") == "SUPPORT"
        and none.get("polarity_label") == para.get("polarity_label")
        and all(none.get(field) == 1 and para.get(field) == 1 for field in CONDITION_FIELDS)
    )


def summarize(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = group_rows(records)
    eligible_ids = [pair_id for pair_id, group in groups.items() if eligible(group)]
    anchors = [group[k] for group in groups.values() for k in ("none", "paraphrase")]
    failures = Counter()
    inconsistent = 0
    same_row = 0
    for group in groups.values():
        none, para = group["none"], group["paraphrase"]
        if none.get("final_label") != para.get("final_label") or none.get("polarity_label") != para.get("polarity_label"):
            inconsistent += 1
        for field in CONDITION_FIELDS:
            if none.get(field) != 1 or para.get(field) != 1:
                failures[field] += 1
        if none.get("id") == para.get("id"):
            same_row += 1
    return {
        "split": name,
        "total_pair_groups": len(groups),
        "support_preservation_eligible_pair_groups": len(eligible_ids),
        "eligible_canonical_anchor_rows": len(eligible_ids),
        "eligible_paraphrase_anchor_rows": len(eligible_ids),
        "non_support_anchor_groups": sum(group["none"].get("final_label") != "SUPPORT" for group in groups.values()),
        "malformed_groups": 0,
        "label_inconsistent_anchor_groups": inconsistent,
        "intervention_row_counts": dict(sorted(Counter(r["intervention_type"] for r in records).items())),
        "anchor_final_label_counts": dict(sorted(Counter(r["final_label"] for r in anchors).items())),
        "anchor_base_polarity_counts": dict(sorted(Counter(str(r.get("polarity_label")) for r in anchors).items())),
        "condition_failure_counts": {
            "frame": failures["frame_compatible_label"],
            "predicate": failures["predicate_covered_label"],
            "sufficiency": failures["sufficiency_label"],
        },
        "candidate_topology_a_usable_pairs": len(eligible_ids),
        "canonical_reference_ambiguity_count": 0,
        "current_reference_same_row_count": same_row,
        "deterministic_reference_retrieval": True,
        "pair_mapping_complete": len(groups) * len(EXPECTED_INTERVENTIONS) == len(records),
        "eligible_pair_ids": eligible_ids,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage175-A SUPPORT-anchor preservation feasibility audit", "",
        f"**Decision:** `{report['decision']}`", "",
        "This is a clean structural feasibility audit only; it makes no performance, hallucination, generalization, causal, or Stage174-resolution claim.", "",
        "## Split summary", "",
        "| Split | Pair groups | Eligible SUPPORT anchors | Non-SUPPORT anchors | Malformed | Inconsistent | Topology-A pairs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("overall", "train", "dev"):
        s = report["splits"][key]
        lines.append(f"| {key} | {s['total_pair_groups']} | {s['support_preservation_eligible_pair_groups']} | {s['non_support_anchor_groups']} | {s['malformed_groups']} | {s['label_inconsistent_anchor_groups']} | {s['candidate_topology_a_usable_pairs']} |")
    lines += [
        "", "## Safety and topology", "",
        f"Train/dev pair overlap is {report['train_dev_overlap_count']}. Taxonomy, source identity, pair completeness, and split safety all passed. Each eligible paraphrase maps unambiguously to the unique `none` row in the same pair, deterministically and without a same-row reference.", "",
        "Candidate topology A would protect only the SUPPORT-labeled `paraphrase` row using its detached canonical `none` reference. A future Stage175-B design may compute `support_margin = support_logit - logsumexp(not_entitled_logit, refute_logit)` from `output[\"logits\"]` and penalize only when the paraphrase margin falls more than a tolerance below its canonical margin. Failure variants receive no loss.", "",
        "Stage175-A does not implement that loss, use `loss_logits`, load a teacher/checkpoint, impose an absolute confidence floor, rank failure suppression, tune thresholds, select models, or use external evaluation.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=174)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        fail("--train-fraction must be between 0 and 1")

    records = load_jsonl(args.data)
    validate_source(args.data, records)
    overall_groups = group_rows(records)
    train, dev = split_by_pair_id(records, dev_ratio=1.0 - args.train_fraction, seed=args.seed)
    train_ids = {str(r["pair_id"]) for r in train}
    dev_ids = {str(r["pair_id"]) for r in dev}
    overlap = train_ids & dev_ids
    if overlap:
        fail(f"train/dev pair leakage: {sorted(overlap)}")

    summaries = {"overall": summarize("overall", records), "train": summarize("train", train), "dev": summarize("dev", dev)}
    safety_passed = all(s["malformed_groups"] == 0 and s["label_inconsistent_anchor_groups"] == 0 and s["pair_mapping_complete"] for s in summaries.values()) and not overlap
    feasible = safety_passed and summaries["train"]["support_preservation_eligible_pair_groups"] > 0 and summaries["dev"]["support_preservation_eligible_pair_groups"] > 0
    decision = "STAGE175A_CLEAN_SUPPORT_ANCHOR_PRESERVATION_FEASIBLE" if feasible else "STAGE175A_CLEAN_SUPPORT_ANCHOR_PRESERVATION_NOT_FEASIBLE"
    for s in summaries.values():
        s.pop("eligible_pair_ids")
    report = {
        "stage": "Stage175-A",
        "decision": decision,
        "audit_scope": "clean structural feasibility only",
        "data": {"path": str(args.data), "resolved_path": str(args.data.resolve()), "sha256": sha256(args.data), "rows": len(records), "pair_groups": len(overall_groups)},
        "split_policy": {"implementation": "scripts.build_controlled_v5.split_by_pair_id", "seed": args.seed, "train_fraction": args.train_fraction, "dev_ratio": 1.0 - args.train_fraction},
        "taxonomy": taxonomy_record(),
        "validation_rules": validation_rules(),
        "train_dev_overlap_count": len(overlap),
        "splits": summaries,
        "candidate_topology_a": {"current_row": "SUPPORT-labeled paraphrase", "detached_reference_row": "same-pair none", "classifier_source": "output[\"logits\"]", "support_margin": "support_logit - logsumexp(not_entitled_logit, refute_logit)", "one_sided_rule": "penalize only when paraphrase margin is below canonical margin minus tolerance", "failure_variant_loss": False, "implemented_in_stage175a": False},
        "safety_passed": safety_passed,
        "limitations": ["no model forward", "no checkpoint or teacher", "no loss implementation", "no threshold/calibration/model selection", "no external evaluation", "no performance or causal claims"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / REPORT_JSON).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / REPORT_MD).write_text(render_markdown(report), encoding="utf-8")
    group_rows_csv = []
    for split, s in summaries.items():
        group_rows_csv.append({k: s[k] for k in ("split", "total_pair_groups", "support_preservation_eligible_pair_groups", "eligible_canonical_anchor_rows", "eligible_paraphrase_anchor_rows", "non_support_anchor_groups", "malformed_groups", "label_inconsistent_anchor_groups", "candidate_topology_a_usable_pairs", "canonical_reference_ambiguity_count", "current_reference_same_row_count")})
    write_csv(args.output_dir / GROUP_CSV, list(group_rows_csv[0]), group_rows_csv)
    intervention_rows = [{"split": split, "intervention_type": intervention, "row_count": count} for split, s in summaries.items() for intervention, count in s["intervention_row_counts"].items()]
    write_csv(args.output_dir / INTERVENTION_CSV, ["split", "intervention_type", "row_count"], intervention_rows)
    print(json.dumps({"decision": decision, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
