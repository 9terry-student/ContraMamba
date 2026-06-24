"""Stage21-F0: OOD tradeoff summary from Stage21-E3 best-dev artifacts."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MEAN_CSV = ROOT / "results" / "stage21_e3_bestdev_v5_vs_v6b_ood_3seed_mean.csv"
OUT_TABLE = ROOT / "results" / "stage21_f0_ood_tradeoff_table.csv"
OUT_NOTES = ROOT / "results" / "stage21_f0_ood_tradeoff_notes.md"

GROUPS = [
    "__overall__",
    "temporal_mismatch",
    "predicate_mismatch",
    "surface_control",
    "temporal_erased",
    "frame_location_mismatch",
    "frame_role_mismatch",
    "sufficiency_control",
]

METRICS = [
    "final_accuracy",
    "final_macro_f1",
    "false_entitled_rate",
    "false_not_entitled_rate",
]


def _interpret(metric: str, delta: float) -> str:
    if metric in ("final_accuracy", "final_macro_f1"):
        if delta > 0.05:
            return "v6B improves"
        if delta < -0.05:
            return "v6B worsens"
        return "similar"
    if metric == "false_entitled_rate":
        if delta < -0.05:
            return "v6B reduces false entitlement"
        if delta > 0.05:
            return "v6B worsens false entitlement"
        return "similar"
    if metric == "false_not_entitled_rate":
        if delta < -0.05:
            return "v6B reduces over-rejection"
        if delta > 0.05:
            return "v6B worsens over-rejection"
        return "similar"
    return "n/a"


def _load(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    data: dict[tuple[str, str], dict[str, str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            data[(row["run"], row["group"])] = row
    return data


def _fmt(raw: str) -> str:
    if not raw:
        return ""
    try:
        return f"{float(raw):.4f}"
    except ValueError:
        return raw


def main() -> None:
    data = _load(MEAN_CSV)

    table_rows: list[dict[str, str]] = []
    for group in GROUPS:
        v5 = data.get(("v5_bestdev", group), {})
        v6b = data.get(("v6b_bestdev", group), {})
        for metric in METRICS:
            v5_raw = v5.get(metric, "")
            v6b_raw = v6b.get(metric, "")
            if not v5_raw and not v6b_raw:
                continue
            try:
                v5_val: float | None = float(v5_raw) if v5_raw else None
            except ValueError:
                v5_val = None
            try:
                v6b_val: float | None = float(v6b_raw) if v6b_raw else None
            except ValueError:
                v6b_val = None
            if v5_val is not None and v6b_val is not None:
                delta: float | None = v6b_val - v5_val
                interp = _interpret(metric, delta)
                delta_str = f"{delta:+.4f}"
            else:
                delta = None
                interp = "n/a"
                delta_str = ""
            table_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "v5_bestdev": _fmt(v5_raw),
                    "v6b_bestdev": _fmt(v6b_raw),
                    "delta_v6b_minus_v5": delta_str,
                    "interpretation": interp,
                }
            )

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TABLE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "metric",
                "v5_bestdev",
                "v6b_bestdev",
                "delta_v6b_minus_v5",
                "interpretation",
            ],
        )
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Wrote {OUT_TABLE}")

    notes = """\
# Stage21-F0 OOD Tradeoff Summary

## Source
Stage21-E3 best-dev checkpoints (3 seeds, mean across seeds).

## Key Findings

### v6B strengths — targeted guard signal
- **temporal_mismatch false-entitled rate**: v5=0.230 → v6B=0.000
  (reduced to zero across all 3 seeds)
- **predicate_mismatch false-entitled rate**: v5=0.203 → v6B=0.000
  (reduced to zero across all 3 seeds)
- **Overall OOD accuracy**: v5=0.559 → v6B=0.636 (+0.077)
- **Overall OOD macro-F1**: v5=0.279 → v6B=0.324 (+0.045)

### v6B failures — SUPPORT control over-rejection
- **surface_control false-not-entitled rate**: v5=0.797 → v6B=0.697
  (improvement of 0.100 but still severe)
- **temporal_erased false-not-entitled rate**: v5=0.830 → v6B=0.787
  (improvement of 0.043 but still severe)

### v6B regressions — frame mismatch
- **frame_location_mismatch false-entitled rate**: v5=0.250 → v6B=0.333 (-0.083, v6B worsens)
- **frame_role_mismatch false-entitled rate**: v5=0.200 → v6B=0.350 (-0.150, v6B worsens)

## Conclusion
v6B is a targeted temporal/predicate guard, not a complete selective OOD solution.
The temporal and predicate comparators eliminate false entitlement on their target probe
types but the model still severely over-rejects SUPPORT controls (surface_control,
temporal_erased) and regresses on frame mismatch detection.

## Next Stage
Stage21-F1: comparator ablation (current / no-flags / temporal-only / predicate-only)
with full Stage15 OOD evaluation to isolate which comparator drives each effect.
"""
    OUT_NOTES.write_text(notes, encoding="utf-8")
    print(f"Wrote {OUT_NOTES}")


if __name__ == "__main__":
    main()
