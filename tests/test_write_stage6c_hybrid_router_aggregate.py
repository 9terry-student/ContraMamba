from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.write_stage6c_hybrid_router_aggregate import main


REPO_ROOT = Path(__file__).resolve().parents[1]


def _seed(path: Path, seed: int, value: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "threshold", "system", "metric", "value"))
        writer.writeheader()
        for system in ("conservative_balanced_router", "cautious_promotion_router"):
            writer.writerow({"seed": seed, "threshold": 0.5, "system": system,
                             "metric": "final_macro_f1", "value": value})
        writer.writerow({"seed": seed, "threshold": 0.5, "system": "cautious_promotion_router",
                         "metric": "text", "value": "ignored"})


def test_hybrid_aggregate_sample_std_markdown_and_direct_cli(tmp_path: Path) -> None:
    seed1, seed2 = tmp_path / "s1.csv", tmp_path / "s2.csv"
    output_csv, output_md = tmp_path / "aggregate.csv", tmp_path / "aggregate.md"
    _seed(seed1, 1, 0.8); _seed(seed2, 2, 0.9)
    assert main(["--input", str(seed1), "--input", str(seed2), "--output-csv", str(output_csv),
                 "--output-md", str(output_md)]) == 0
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    first = rows[0]
    assert list(first) == ["threshold", "system", "metric", "mean", "std", "n", "formatted"]
    assert float(first["mean"]) == pytest.approx(0.85)
    assert float(first["std"]) == pytest.approx(0.1 / 2**0.5)
    assert "## THRESHOLD 0.5" in output_md.read_text(encoding="utf-8")
    completed = subprocess.run([sys.executable, "scripts/write_stage6c_hybrid_router_aggregate.py", "--help"],
                               cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
