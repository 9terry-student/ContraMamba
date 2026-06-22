from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.evaluate_hybrid_router_search import build_hybrid_systems, main


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERVENTIONS = (
    "none", "paraphrase", "entity_swap", "event_swap", "time_swap", "location_swap",
    "role_swap", "title_name_swap", "predicate_swap", "evidence_deletion",
    "evidence_truncation", "irrelevant_evidence", "polarity_flip",
)


def _row(example_id: str, label: str, gate: float) -> dict:
    return {"id": example_id, "pair_id": "p", "intervention_type": "none",
            "claim": "claim", "evidence": "evidence", "gold_final_label": label,
            "pred_final_label": label, "frame_prob": gate,
            "predicate_coverage_prob": gate, "sufficiency_prob": gate,
            "entitlement_prob": gate, "polarity_margin": 1.0 if label == "SUPPORT" else -1.0}


def _item(example_id: str, c: tuple[str, float], b: tuple[str, float], s: tuple[str, float]) -> dict:
    return {"classifier": _row(example_id, *c), "balanced": _row(example_id, *b),
            "strict": _row(example_id, *s)}


def test_hybrid_router_rules() -> None:
    items = [
        _item("low-balanced", ("SUPPORT", 0.9), ("SUPPORT", 0.4), ("SUPPORT", 0.9)),
        _item("override", ("NOT_ENTITLED", 0.9), ("SUPPORT", 0.8), ("NOT_ENTITLED", 0.9)),
        _item("veto", ("SUPPORT", 0.9), ("SUPPORT", 0.9), ("SUPPORT", 0.4)),
        _item("majority", ("SUPPORT", 0.6), ("SUPPORT", 0.8), ("NOT_ENTITLED", 0.9)),
        _item("tie", ("SUPPORT", 0.9), ("REFUTE", 0.9), ("NOT_ENTITLED", 0.9)),
        _item("invalid-majority", ("SUPPORT", 0.4), ("SUPPORT", 0.4), ("NOT_ENTITLED", 0.9)),
        _item("cautious", ("NOT_ENTITLED", 0.9), ("REFUTE", 0.8), ("REFUTE", 0.8)),
        _item("cautious-disagree", ("NOT_ENTITLED", 0.9), ("REFUTE", 0.8), ("SUPPORT", 0.8)),
    ]
    labels, _ = build_hybrid_systems(items, threshold=0.5, high_threshold=0.7)
    assert labels["conservative_balanced_router"]["low-balanced"] == "NOT_ENTITLED"
    assert labels["balanced_override_router"]["override"] == "SUPPORT"
    assert labels["strict_veto_balanced_router"]["veto"] == "NOT_ENTITLED"
    assert labels["majority_gate_verified_router"]["majority"] == "SUPPORT"
    assert labels["majority_gate_verified_router"]["tie"] == "NOT_ENTITLED"
    assert labels["majority_gate_verified_router"]["invalid-majority"] == "NOT_ENTITLED"
    assert labels["cautious_promotion_router"]["cautious"] == "REFUTE"
    assert labels["cautious_promotion_router"]["cautious-disagree"] == "NOT_ENTITLED"

    blocked, _ = build_hybrid_systems(
        [_item("blocked", ("NOT_ENTITLED", 0.9), ("SUPPORT", 0.65), ("SUPPORT", 0.9))],
        threshold=0.5, high_threshold=0.7,
    )
    assert blocked["balanced_override_router"]["blocked"] == "NOT_ENTITLED"


def _payload(kind: str) -> dict:
    rows = []
    for intervention in INTERVENTIONS:
        gold = {"none": "SUPPORT", "paraphrase": "SUPPORT", "polarity_flip": "REFUTE"}.get(
            intervention, "NOT_ENTITLED"
        )
        rows.append({**_row(f"p-{intervention}", gold, 0.8), "intervention_type": intervention})
    return {"metadata": {"kind": kind}, "predictions": rows}


def test_hybrid_cli_writes_long_outputs_and_runs_directly(tmp_path: Path) -> None:
    paths = {}
    for kind in ("classifier", "balanced", "strict"):
        path = tmp_path / f"{kind}.json"; path.write_text(json.dumps(_payload(kind)), encoding="utf-8")
        paths[kind] = path
    output_csv, output_md = tmp_path / "hybrid.csv", tmp_path / "hybrid.md"
    assert main(["--classifier-preds", str(paths["classifier"]), "--balanced-preds", str(paths["balanced"]),
                 "--strict-preds", str(paths["strict"]), "--seed", "1", "--thresholds", "0.5",
                 "--high-threshold", "0.7", "--output-csv", str(output_csv), "--output-md", str(output_md)]) == 0
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert set(rows[0]) == {"seed", "threshold", "system", "metric", "value"}
    assert {float(row["threshold"]) for row in rows} == {0.5}
    assert "# Stage 6C Hybrid Expert Router Search" in output_md.read_text(encoding="utf-8")
    completed = subprocess.run([sys.executable, "scripts/evaluate_hybrid_router_search.py", "--help"],
                               cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
