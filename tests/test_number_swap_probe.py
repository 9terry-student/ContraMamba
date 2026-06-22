from __future__ import annotations

import re
from pathlib import Path

from scripts.create_number_swap_probe import (
    NUMBER_SWAP_FINAL_LABEL,
    build_number_swap_probe,
)
from scripts.evaluate_number_swap_probe import write_outputs


def test_number_swap_changes_only_numeric_quantity() -> None:
    records = build_number_swap_probe(3)
    swaps = [record for record in records if record["intervention_type"] == "number_swap"]
    for record in swaps:
        claim_numbers = re.findall(r"\d+", record["claim"])
        evidence_numbers = re.findall(r"\d+", record["evidence"])
        assert len(claim_numbers) == len(evidence_numbers) == 1
        assert claim_numbers != evidence_numbers
        assert re.sub(r"\d+", "<NUMBER>", record["claim"]) == re.sub(
            r"\d+", "<NUMBER>", record["evidence"]
        )


def test_pair_id_grouping_and_ontology_are_preserved() -> None:
    records = build_number_swap_probe(4)
    pair_ids = {record["pair_id"] for record in records}
    assert len(pair_ids) == 4
    for pair_id in pair_ids:
        group = [record for record in records if record["pair_id"] == pair_id]
        assert len(group) == 2
        assert {record["intervention_type"] for record in group} == {"none", "number_swap"}
        swap = next(record for record in group if record["intervention_type"] == "number_swap")
        assert swap["final_label"] == NUMBER_SWAP_FINAL_LABEL == "NOT_ENTITLED"
        assert swap["frame_compatible_label"] == 0
        assert swap["predicate_covered_label"] == 1
        assert swap["sufficiency_label"] == 1


def test_probe_markdown_contains_required_decision_and_warnings(tmp_path: Path) -> None:
    number = {
        "probe": "number_swap", "n": 2, "gold_NOT_ENTITLED": 2, "gold_SUPPORT": 0,
        "gold_REFUTE": 0, "pred_SUPPORT": 2, "pred_REFUTE": 0,
        "pred_NOT_ENTITLED": 0, "classifier_error": 2, "false_entitled": 2,
        "false_entitled_rate": 1.0, "mean_frame_prob": 0.9,
        "mean_predicate_coverage_prob": 0.9, "mean_sufficiency_prob": 0.9,
        "mean_entitlement_prob": 0.9, "mean_polarity_margin": 1.0,
    }
    time = {**number, "probe": "time_swap"}
    csv_path, md_path = tmp_path / "probe.csv", tmp_path / "probe.md"
    write_outputs(csv_path, md_path, [number, time])
    markdown = md_path.read_text(encoding="utf-8")
    assert "Temporal-specific vs same-type-substitution decision" in markdown
    assert "same_type_low_surface_change_slot_value_failure" in markdown
    assert "Do not generalize the time-swap result into broad presence-vs-match blindness" in markdown
    assert "Do not infer shared gate mechanisms from global correlation alone" in markdown
