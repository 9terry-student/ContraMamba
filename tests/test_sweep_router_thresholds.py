from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.sweep_router_thresholds import main, sweep_thresholds


INTERVENTIONS = (
    "none", "paraphrase", "entity_swap", "event_swap", "time_swap",
    "location_swap", "role_swap", "title_name_swap", "predicate_swap",
    "evidence_deletion", "evidence_truncation", "irrelevant_evidence",
    "polarity_flip",
)


def _payload(kind: str) -> dict:
    rows = []
    for intervention in INTERVENTIONS:
        gold = {"none": "SUPPORT", "paraphrase": "SUPPORT", "polarity_flip": "REFUTE"}.get(
            intervention, "NOT_ENTITLED"
        )
        classifier_label = "REFUTE" if intervention == "polarity_flip" else "SUPPORT"
        auditor_label = gold
        gate = 0.55 if intervention in {"none", "paraphrase", "polarity_flip"} else 0.2
        rows.append(
            {
                "id": f"p1-{intervention}", "pair_id": "p1",
                "intervention_type": intervention, "claim": "claim", "evidence": "evidence",
                "gold_final_label": gold,
                "pred_final_label": classifier_label if kind == "classifier" else auditor_label,
                "frame_prob": gate, "predicate_coverage_prob": gate,
                "sufficiency_prob": gate, "entitlement_prob": gate,
                "polarity_margin": -1.0 if intervention == "polarity_flip" else 1.0,
            }
        )
    return {"metadata": {"kind": kind}, "predictions": rows}


def test_threshold_sweep_applies_cutoff_and_writes_outputs(tmp_path: Path) -> None:
    classifier = _payload("classifier")
    balanced = _payload("balanced")
    strict = _payload("strict")
    swept = sweep_thresholds(classifier, balanced, strict, (0.5, 0.6))
    balanced_rows = {
        row["threshold"]: row
        for row in swept
        if row["system"] == "conservative_balanced_router"
    }
    assert balanced_rows[0.5]["entitled_output_count"] == 3
    assert balanced_rows[0.6]["entitled_output_count"] == 0
    assert balanced_rows[0.5]["polarity_flip_output_ok"] == 1.0
    assert balanced_rows[0.6]["polarity_flip_output_ok"] == 0.0

    inputs = {}
    for kind, payload in (("classifier", classifier), ("balanced", balanced), ("strict", strict)):
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        inputs[kind] = path
    output_csv = tmp_path / "sweep.csv"
    output_md = tmp_path / "sweep.md"
    assert main([
        "--classifier-preds", str(inputs["classifier"]),
        "--balanced-preds", str(inputs["balanced"]),
        "--strict-preds", str(inputs["strict"]),
        "--seed", "7", "--output-csv", str(output_csv), "--output-md", str(output_md),
    ]) == 0
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert "threshold" in rows[0]
    assert {float(row["threshold"]) for row in rows} == {0.3, 0.4, 0.5, 0.6, 0.7}
    assert "Stage 6A Router Threshold Sweep" in output_md.read_text(encoding="utf-8")
