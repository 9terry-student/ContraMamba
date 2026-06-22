from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.write_stage6a_threshold_sweep_aggregate import main


def _write_seed(path: Path, macro_f1: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("threshold", "system", "final_macro_f1", "note"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "threshold": 0.5,
                "system": "conservative_balanced_router",
                "final_macro_f1": macro_f1,
                "note": "nonnumeric value",
            }
        )


def test_threshold_aggregate_writes_mean_std_and_ignores_text(tmp_path: Path) -> None:
    seed1 = tmp_path / "seed1.csv"
    seed2 = tmp_path / "seed2.csv"
    output_csv = tmp_path / "aggregate.csv"
    output_md = tmp_path / "aggregate.md"
    _write_seed(seed1, 0.8)
    _write_seed(seed2, 0.9)

    assert main([
        "--input", str(seed1), "--input", str(seed2),
        "--output-csv", str(output_csv), "--output-md", str(output_md),
    ]) == 0
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["threshold", "system", "metric", "mean", "std", "n", "formatted"]
    assert rows[0]["metric"] == "final_macro_f1"
    assert float(rows[0]["mean"]) == pytest.approx(0.85)
    assert float(rows[0]["std"]) == pytest.approx(0.1 / 2**0.5)
    assert int(rows[0]["n"]) == 2
    assert rows[0]["formatted"] == "0.850 +/- 0.071"
    assert "note" not in {row["metric"] for row in rows}
    markdown = output_md.read_text(encoding="utf-8")
    assert "## THRESHOLD 0.5" in markdown
    assert "+/-" in markdown
    assert "±" not in markdown
