from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_controlled_v5 import build_controlled_records
from scripts.evaluate_router_ensemble import (
    build_system_predictions,
    evaluate_router_systems,
    main as router_main,
    merge_prediction_files,
    pairwise_prediction_checks,
)
from scripts.train_controlled_v5 import main as train_main


TEST_DIR = Path(__file__).resolve().parent
INTERVENTIONS = (
    "none",
    "paraphrase",
    "entity_swap",
    "event_swap",
    "time_swap",
    "location_swap",
    "role_swap",
    "title_name_swap",
    "predicate_swap",
    "evidence_deletion",
    "evidence_truncation",
    "irrelevant_evidence",
    "polarity_flip",
)


def _prediction_payload(kind: str) -> dict:
    predictions = []
    for intervention in INTERVENTIONS:
        gold = {
            "none": "SUPPORT",
            "paraphrase": "SUPPORT",
            "polarity_flip": "REFUTE",
        }.get(intervention, "NOT_ENTITLED")
        classifier_prediction = {
            "polarity_flip": "REFUTE",
            "paraphrase": "SUPPORT",
        }.get(intervention, "SUPPORT")
        auditor_prediction = gold
        passes = intervention in {"none", "paraphrase", "polarity_flip"}
        gate = 0.9 if passes else 0.1
        predictions.append(
            {
                "id": f"pair-1-{intervention}",
                "pair_id": "pair-1",
                "intervention_type": intervention,
                "claim": "A claim.",
                "evidence": "Some evidence.",
                "gold_final_label": gold,
                "pred_final_label": (
                    classifier_prediction if kind == "classifier" else auditor_prediction
                ),
                "final_probs": [0.05, 0.05, 0.9],
                "frame_prob": gate,
                "predicate_coverage_prob": gate,
                "sufficiency_prob": gate,
                "entitlement_prob": gate,
                "polarity_margin": -1.0 if intervention == "polarity_flip" else 1.0,
            }
        )
    return {"metadata": {"kind": kind}, "predictions": predictions}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_router_downgrades_failed_audits_and_keeps_passed_audits() -> None:
    classifier = _prediction_payload("classifier")
    balanced = _prediction_payload("balanced")
    strict = _prediction_payload("strict")
    merged = merge_prediction_files(classifier, balanced, strict)
    systems = build_system_predictions(merged)

    assert systems["conservative_balanced_router"]["pair-1-none"] == "SUPPORT"
    assert (
        systems["conservative_balanced_router"]["pair-1-predicate_swap"]
        == "NOT_ENTITLED"
    )
    assert systems["dual_auditor_router"]["pair-1-entity_swap"] == "NOT_ENTITLED"


def test_pairwise_polarity_flip_consistency_uses_final_predictions() -> None:
    payload = _prediction_payload("classifier")
    labels = {row["id"]: row["pred_final_label"] for row in payload["predictions"]}
    checks = pairwise_prediction_checks(payload["predictions"], labels)

    assert checks["polarity_flip_preserved_and_reversed"] == {
        "passed": 1,
        "total": 1,
        "pass_rate": 1.0,
    }


def test_router_cli_writes_markdown_and_csv() -> None:
    paths = {
        name: TEST_DIR / f"._router_{name}.json"
        for name in ("classifier", "balanced", "strict")
    }
    output_md = TEST_DIR / "._router_report.md"
    output_csv = TEST_DIR / "._router_report.csv"
    try:
        for name, path in paths.items():
            _write_json(path, _prediction_payload(name))
        assert (
            router_main(
                [
                    "--classifier-preds",
                    str(paths["classifier"]),
                    "--balanced-preds",
                    str(paths["balanced"]),
                    "--strict-preds",
                    str(paths["strict"]),
                    "--output-md",
                    str(output_md),
                    "--output-csv",
                    str(output_csv),
                ]
            )
            == 0
        )
        markdown = output_md.read_text(encoding="utf-8")
        for section in (
            "ROUTER_CLASSIFICATION_SUMMARY",
            "ROUTER_PAIRWISE_SUMMARY",
            "ROUTER_KEY_CONTRAST",
            "INTERPRETATION",
        ):
            assert f"## {section}" in markdown
        with output_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert set(rows[0]) == {"section", "system", "metric", "value"}
        results = evaluate_router_systems(
            _prediction_payload("classifier"),
            _prediction_payload("balanced"),
            _prediction_payload("strict"),
        )
        assert results["conservative_balanced_router"]["final_accuracy"] == 1.0
    finally:
        for path in (*paths.values(), output_md, output_csv):
            path.unlink(missing_ok=True)


def test_training_exports_best_epoch_predictions_only_when_requested() -> None:
    data_path = TEST_DIR / "._router_controlled.jsonl"
    output_path = TEST_DIR / "._router_predictions.json"
    omitted_path = TEST_DIR / "._router_not_written.json"
    records = build_controlled_records(2)
    try:
        data_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        assert (
            train_main(
                [
                    "--data",
                    str(data_path),
                    "--backbone",
                    "dummy",
                    "--epochs",
                    "1",
                    "--dev-ratio",
                    "0.5",
                    "--device",
                    "cpu",
                    "--output-predictions-json",
                    str(output_path),
                ]
            )
            == 0
        )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["metadata"]["best_epoch"] == 1
        assert payload["metadata"]["data_path"] == str(data_path)
        assert len(payload["predictions"]) == 13
        expected = {
            "id",
            "pair_id",
            "intervention_type",
            "claim",
            "evidence",
            "gold_final_label",
            "pred_final_label",
            "final_probs",
            "frame_prob",
            "predicate_coverage_prob",
            "sufficiency_prob",
            "entitlement_prob",
            "polarity_margin",
        }
        assert expected <= set(payload["predictions"][0])
        assert not omitted_path.exists()
    finally:
        for path in (data_path, output_path, omitted_path):
            path.unlink(missing_ok=True)
