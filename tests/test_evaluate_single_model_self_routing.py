from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.evaluate_single_model_self_routing import evaluate_self_routing, main


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERVENTIONS = (
    "none", "paraphrase", "entity_swap", "event_swap", "time_swap",
    "location_swap", "role_swap", "title_name_swap", "predicate_swap",
    "evidence_deletion", "evidence_truncation", "irrelevant_evidence", "polarity_flip",
)


def _payload(kind: str) -> dict:
    predictions = []
    for intervention in INTERVENTIONS:
        gold = {"none": "SUPPORT", "paraphrase": "SUPPORT", "polarity_flip": "REFUTE"}.get(
            intervention, "NOT_ENTITLED"
        )
        label = gold if kind != "classifier" else (
            "REFUTE" if intervention == "polarity_flip" else "SUPPORT"
        )
        gate = 0.55 if intervention in {"none", "paraphrase", "polarity_flip"} else 0.2
        predictions.append(
            {
                "id": f"p1-{intervention}", "pair_id": "p1",
                "intervention_type": intervention, "claim": "claim", "evidence": "evidence",
                "gold_final_label": gold, "pred_final_label": label,
                "frame_prob": gate, "predicate_coverage_prob": gate,
                "sufficiency_prob": gate, "entitlement_prob": gate,
                "polarity_margin": -1.0 if intervention == "polarity_flip" else 1.0,
            }
        )
    return {"metadata": {"kind": kind}, "predictions": predictions}


def _metric(rows: list[dict], threshold: float, system: str, metric: str) -> float:
    return float(next(row["value"] for row in rows if row["threshold"] == threshold
                      and row["system"] == system and row["metric"] == metric))


def test_raw_is_stable_and_self_routing_applies_threshold(tmp_path: Path) -> None:
    payloads = {kind: _payload(kind) for kind in ("classifier", "balanced", "strict")}
    rows = evaluate_self_routing(
        payloads["classifier"], payloads["balanced"], payloads["strict"], 1, (0.5, 0.6)
    )
    assert _metric(rows, 0.5, "raw_classifier_only", "final_accuracy") == _metric(
        rows, 0.6, "raw_classifier_only", "final_accuracy"
    )
    assert _metric(rows, 0.5, "self_routed_classifier", "entitled_output_count") == 3
    assert _metric(rows, 0.6, "self_routed_classifier", "entitled_output_count") == 0

    paths = {}
    for kind, payload in payloads.items():
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[kind] = path
    output_csv = tmp_path / "self.csv"
    output_md = tmp_path / "self.md"
    assert main([
        "--classifier-preds", str(paths["classifier"]),
        "--balanced-preds", str(paths["balanced"]),
        "--strict-preds", str(paths["strict"]), "--seed", "1",
        "--thresholds", "0.5", "0.6", "--output-csv", str(output_csv),
        "--output-md", str(output_md),
    ]) == 0
    with output_csv.open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert set(exported[0]) == {"seed", "threshold", "system", "metric", "value"}
    assert {float(row["threshold"]) for row in exported} == {0.5, 0.6}
    assert "# Stage 6B Single-Model Self-Routing" in output_md.read_text(encoding="utf-8")


def test_self_routing_cli_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_single_model_self_routing.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--classifier-preds" in completed.stdout
