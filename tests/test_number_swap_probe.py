from __future__ import annotations

import re
import json
from pathlib import Path

import torch

from scripts.create_number_swap_probe import (
    NUMBER_SWAP_FINAL_LABEL,
    build_number_swap_probe,
)
from scripts.build_controlled_v5 import build_seed_records
from scripts.evaluate_number_swap_probe import main as evaluate_main, write_outputs
from scripts.train_and_export_stage10a_number_swap import (
    export_probe_predictions,
    export_stage10a_predictions,
    load_number_swap_probe,
)


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


def _synthetic_output(size: int) -> dict[str, torch.Tensor]:
    logits = torch.tensor([[0.0, -1.0, 2.0]]).repeat(size, 1)
    return {
        "logits": logits,
        "predictions": logits.argmax(dim=-1),
        "frame_prob": torch.full((size,), 0.9),
        "predicate_coverage_prob": torch.full((size,), 0.8),
        "sufficiency_prob": torch.full((size,), 0.7),
        "entitlement_prob": torch.full((size,), 0.504),
        "polarity_margin": torch.full((size,), 1.0),
    }


def test_probe_prediction_export_schema_and_intervention_coverage(tmp_path: Path) -> None:
    records = build_number_swap_probe(2)
    path = tmp_path / "number-preds.json"
    export_probe_predictions(path, records, _synthetic_output(len(records)), {"seed": 1})
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "id", "pair_id", "claim", "evidence", "intervention_type",
        "gold_final_label", "pred_final_label", "final_probs", "frame_prob",
        "predicate_coverage_prob", "sufficiency_prob", "entitlement_prob",
        "polarity_margin",
    }
    assert payload["metadata"] == {"seed": 1}
    assert all(set(row) == expected for row in payload["predictions"])
    assert {row["intervention_type"] for row in payload["predictions"]} == {
        "none", "number_swap"
    }


def test_probe_loader_rejects_ontology_changes(tmp_path: Path) -> None:
    records = build_number_swap_probe(1)
    records[1]["final_label"] = "REFUTE"
    path = tmp_path / "bad-probe.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    try:
        load_number_swap_probe(path)
    except ValueError as error:
        assert "ontology" in str(error)
    else:
        raise AssertionError("changed number_swap ontology was accepted")


def test_same_checkpoint_export_contains_time_swap_and_matching_schema(
    tmp_path: Path,
) -> None:
    probe_records = build_number_swap_probe(1)
    controlled_records = build_seed_records()[:13]
    number_path = tmp_path / "number.json"
    matched_path = tmp_path / "matched-controlled.json"
    metadata = {"seed": 1, "checkpoint_id": "same-best-checkpoint"}
    export_stage10a_predictions(
        number_path,
        matched_path,
        probe_records,
        _synthetic_output(len(probe_records)),
        controlled_records,
        _synthetic_output(len(controlled_records)),
        metadata,
    )
    number_payload = json.loads(number_path.read_text(encoding="utf-8"))
    matched_payload = json.loads(matched_path.read_text(encoding="utf-8"))
    assert number_payload["metadata"]["checkpoint_id"] == "same-best-checkpoint"
    assert matched_payload["metadata"]["checkpoint_id"] == "same-best-checkpoint"
    assert any(
        row["intervention_type"] == "time_swap"
        for row in matched_payload["predictions"]
    )
    assert set(number_payload["predictions"][0]) == set(
        matched_payload["predictions"][0]
    )


def test_evaluator_compares_exported_number_and_matched_time_predictions(
    tmp_path: Path,
) -> None:
    number_records = build_number_swap_probe(2)
    number_path = tmp_path / "number.json"
    export_probe_predictions(
        number_path, number_records, _synthetic_output(len(number_records)), {"seed": 1}
    )
    time_records = []
    for index in range(2):
        base = dict(number_records[index * 2 + 1])
        base["id"] = f"time-{index}"
        base["intervention_type"] = "time_swap"
        time_records.append(base)
    time_path = tmp_path / "time.json"
    export_probe_predictions(
        time_path, time_records, _synthetic_output(len(time_records)), {"seed": 1}
    )
    csv_path, md_path = tmp_path / "comparison.csv", tmp_path / "comparison.md"
    assert evaluate_main([
        "--number-preds", str(number_path), "--time-preds", str(time_path),
        "--output-csv", str(csv_path), "--output-md", str(md_path),
    ]) == 0
    assert csv_path.exists() and md_path.exists()
