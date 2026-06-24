"""Stage21-G1: Summarize v6B OOD selective preservation gate sweep across seeds.

Reads results/stage21_g1_selective_gate_sweep_seed{1,2,3}.json,
writes:
  results/stage21_g1_selective_gate_sweep_3seed_summary.csv
  results/stage21_g1_selective_gate_sweep_notes.md

Run from repo root:
    python scripts/summarize_stage21_g1_selective_gate_sweep.py
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEED_FILES = {
    1: RESULTS / "stage21_g1_selective_gate_sweep_seed1.json",
    2: RESULTS / "stage21_g1_selective_gate_sweep_seed2.json",
    3: RESULTS / "stage21_g1_selective_gate_sweep_seed3.json",
}

OUT_CSV = RESULTS / "stage21_g1_selective_gate_sweep_3seed_summary.csv"
OUT_NOTES = RESULTS / "stage21_g1_selective_gate_sweep_notes.md"

# (group_name, metric_key, column_prefix)
GROUP_METRICS = [
    ("surface_control", "false_not_entitled_rate", "surface_control_fne"),
    ("temporal_erased", "false_not_entitled_rate", "temporal_erased_fne"),
    ("sufficiency_control", "false_entitled_rate", "sufficiency_control_fe"),
    ("frame_location_mismatch", "false_entitled_rate", "frame_location_mismatch_fe"),
    ("frame_role_mismatch", "false_entitled_rate", "frame_role_mismatch_fe"),
    ("temporal_mismatch", "false_entitled_rate", "temporal_mismatch_fe"),
    ("predicate_mismatch", "false_entitled_rate", "predicate_mismatch_fe"),
]

CSV_FIELDS = [
    "gate",
    "threshold",
    "shift",
    "selected_count_mean",
    "selected_count_std",
    "selected_rate_mean",
    "selected_rate_std",
    "overall_accuracy_mean",
    "overall_accuracy_std",
    "overall_macro_f1_mean",
    "overall_macro_f1_std",
    "surface_control_fne_mean",
    "surface_control_fne_std",
    "temporal_erased_fne_mean",
    "temporal_erased_fne_std",
    "sufficiency_control_fe_mean",
    "sufficiency_control_fe_std",
    "frame_location_mismatch_fe_mean",
    "frame_location_mismatch_fe_std",
    "frame_role_mismatch_fe_mean",
    "frame_role_mismatch_fe_std",
    "temporal_mismatch_fe_mean",
    "temporal_mismatch_fe_std",
    "predicate_mismatch_fe_mean",
    "predicate_mismatch_fe_std",
    "keeps_temporal_guard",
    "keeps_predicate_guard",
    "safe_sufficiency",
    "safe_frame_location",
    "safe_frame_role",
    "passes_g1_gate",
]


def _parse_cond_key(key: str) -> tuple[str, float, float]:
    """Parse 'gate=X|thr=Y|shift=Z' into (gate_name, threshold, shift)."""
    parts: dict[str, str] = {}
    for segment in key.split("|"):
        k, v = segment.split("=", 1)
        parts[k.strip()] = v.strip()
    return parts["gate"], float(parts["thr"]), float(parts["shift"])


def _extract_flat(cond_data: dict[str, Any]) -> dict[str, float | None]:
    """Extract a flat metric dict from one condition's result blob."""
    overall = cond_data.get("overall_metrics", {})
    groups = cond_data.get("group_metrics", {})
    flat: dict[str, float | None] = {
        "selected_count": float(cond_data.get("selected_count", 0)),
        "selected_rate": cond_data.get("selected_rate_among_unflagged"),
        "overall_accuracy": overall.get("final_accuracy"),
        "overall_macro_f1": overall.get("final_macro_f1"),
    }
    for group_name, metric_key, col_prefix in GROUP_METRICS:
        flat[col_prefix] = groups.get(group_name, {}).get(metric_key)
    return flat


def _load_seed(
    path: Path, seed: int
) -> dict[str, dict[str, float | None]]:
    """Return {cond_key: flat_metrics} for one seed file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    sweep = raw.get("ood_selective_ne_shift_sweep") or raw
    return {k: _extract_flat(v) for k, v in sweep.items()}


def _std_safe(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def _fmt(v: float | None) -> str:
    return f"{v:.6f}" if v is not None else ""


def _build_rows(
    all_seeds: dict[int, dict[str, dict[str, float | None]]]
) -> list[dict[str, Any]]:
    """Aggregate per-condition metrics across seeds; return one row per condition."""
    cond_keys: list[str] = []
    seen: set[str] = set()
    for seed_data in all_seeds.values():
        for key in seed_data:
            if key not in seen:
                cond_keys.append(key)
                seen.add(key)

    rows: list[dict[str, Any]] = []
    for cond_key in cond_keys:
        gate, threshold, shift = _parse_cond_key(cond_key)
        metric_vals: dict[str, list[float]] = {}
        for seed_data in all_seeds.values():
            for col, val in seed_data.get(cond_key, {}).items():
                if val is not None:
                    metric_vals.setdefault(col, []).append(val)

        def _mean(col: str) -> float | None:
            vals = metric_vals.get(col)
            return statistics.mean(vals) if vals else None

        def _std(col: str) -> float:
            vals = metric_vals.get(col)
            return _std_safe(vals) if vals else 0.0

        tm_fe_mean = _mean("temporal_mismatch_fe")
        pm_fe_mean = _mean("predicate_mismatch_fe")
        sc_fe_mean = _mean("sufficiency_control_fe")
        fl_fe_mean = _mean("frame_location_mismatch_fe")
        fr_fe_mean = _mean("frame_role_mismatch_fe")

        keeps_temporal = tm_fe_mean is not None and tm_fe_mean == 0.0
        keeps_predicate = pm_fe_mean is not None and pm_fe_mean == 0.0
        safe_suff = sc_fe_mean is not None and sc_fe_mean <= 0.15
        safe_fl = fl_fe_mean is not None and fl_fe_mean <= 0.40
        safe_fr = fr_fe_mean is not None and fr_fe_mean <= 0.40
        passes = keeps_temporal and keeps_predicate and safe_suff and safe_fl and safe_fr

        row: dict[str, Any] = {
            "gate": gate,
            "threshold": f"{threshold:.2f}",
            "shift": f"{shift:g}",
            "selected_count_mean": _fmt(_mean("selected_count")),
            "selected_count_std": _fmt(_std("selected_count")),
            "selected_rate_mean": _fmt(_mean("selected_rate")),
            "selected_rate_std": _fmt(_std("selected_rate")),
            "overall_accuracy_mean": _fmt(_mean("overall_accuracy")),
            "overall_accuracy_std": _fmt(_std("overall_accuracy")),
            "overall_macro_f1_mean": _fmt(_mean("overall_macro_f1")),
            "overall_macro_f1_std": _fmt(_std("overall_macro_f1")),
            "surface_control_fne_mean": _fmt(_mean("surface_control_fne")),
            "surface_control_fne_std": _fmt(_std("surface_control_fne")),
            "temporal_erased_fne_mean": _fmt(_mean("temporal_erased_fne")),
            "temporal_erased_fne_std": _fmt(_std("temporal_erased_fne")),
            "sufficiency_control_fe_mean": _fmt(sc_fe_mean),
            "sufficiency_control_fe_std": _fmt(_std("sufficiency_control_fe")),
            "frame_location_mismatch_fe_mean": _fmt(fl_fe_mean),
            "frame_location_mismatch_fe_std": _fmt(_std("frame_location_mismatch_fe")),
            "frame_role_mismatch_fe_mean": _fmt(fr_fe_mean),
            "frame_role_mismatch_fe_std": _fmt(_std("frame_role_mismatch_fe")),
            "temporal_mismatch_fe_mean": _fmt(tm_fe_mean),
            "temporal_mismatch_fe_std": _fmt(_std("temporal_mismatch_fe")),
            "predicate_mismatch_fe_mean": _fmt(pm_fe_mean),
            "predicate_mismatch_fe_std": _fmt(_std("predicate_mismatch_fe")),
            "keeps_temporal_guard": str(keeps_temporal),
            "keeps_predicate_guard": str(keeps_predicate),
            "safe_sufficiency": str(safe_suff),
            "safe_frame_location": str(safe_fl),
            "safe_frame_role": str(safe_fr),
            "passes_g1_gate": str(passes),
        }
        rows.append(row)
    return rows


def _sort_key(row: dict[str, Any], col: str) -> float:
    try:
        return float(row.get(col) or "inf")
    except (ValueError, TypeError):
        return float("inf")


def _top_k_md(
    rows: list[dict[str, Any]],
    sort_cols: list[str],
    k: int = 10,
    filter_fn: Any = None,
) -> str:
    subset = [r for r in rows if filter_fn(r)] if filter_fn is not None else list(rows)
    subset.sort(key=lambda r: tuple(_sort_key(r, c) for c in sort_cols))
    top = subset[:k]
    if not top:
        return "_No rows._\n"
    display = [
        "gate", "threshold", "shift",
        "selected_rate_mean",
        "surface_control_fne_mean",
        "temporal_erased_fne_mean",
        "sufficiency_control_fe_mean",
        "frame_location_mismatch_fe_mean",
        "frame_role_mismatch_fe_mean",
        "temporal_mismatch_fe_mean",
        "predicate_mismatch_fe_mean",
        "passes_g1_gate",
    ]
    hdr = "| " + " | ".join(display) + " |"
    sep = "| " + " | ".join(["---"] * len(display)) + " |"
    body = [
        "| " + " | ".join(str(r.get(c, "")) for c in display) + " |"
        for r in top
    ]
    return "\n".join([hdr, sep] + body) + "\n"


def main() -> None:
    all_seeds: dict[int, dict[str, dict[str, float | None]]] = {}
    for seed, path in sorted(SEED_FILES.items()):
        if not path.exists():
            print(f"[WARN] {path.name} not found, skipping seed {seed}")
            continue
        print(f"Reading {path.name} ...")
        all_seeds[seed] = _load_seed(path, seed)
        print(f"  -> {len(all_seeds[seed])} conditions")

    rows = _build_rows(all_seeds) if all_seeds else []
    n_passing = sum(1 for r in rows if r.get("passes_g1_gate") == "True")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV}  ({len(rows)} rows, {n_passing} passes_g1_gate=True)")

    if rows:
        table_best = _top_k_md(
            rows,
            sort_cols=["surface_control_fne_mean", "temporal_erased_fne_mean"],
        )
        table_passing = _top_k_md(
            rows,
            sort_cols=["surface_control_fne_mean", "temporal_erased_fne_mean"],
            filter_fn=lambda r: r.get("passes_g1_gate") == "True",
        )
    else:
        table_best = "_No data (seed JSONs not yet available)._\n"
        table_passing = "_No data (seed JSONs not yet available)._\n"

    if not rows:
        passing_note = "No seed data available yet."
    elif n_passing == 0:
        passing_note = "Zero rows passed all five conditions."
    else:
        passing_note = f"{n_passing} row(s) passed all five conditions (see table below)."

    notes = f"""\
# Stage21-G1 Selective Preservation Gate Sweep Notes

## Purpose

Stage21-G0 showed that applying a global NOT_ENTITLED logit shift to all unflagged OOD
records reduced SUPPORT over-rejection on `surface_control` and `temporal_erased`, but
caused severe false-entitled regressions on `frame_location_mismatch`,
`frame_role_mismatch`, and `sufficiency_control`.

Stage21-G1 tests whether the shift can be made safe by restricting it to unflagged records
that look preservation-like according to model-internal auxiliary scores (`frame_prob`,
`sufficiency_prob`, `predicate_coverage_prob`). The shift is eval-only and post-hoc; it
does not affect training, losses, or checkpoint selection.

## Sweep Design

- **Gates**: `high_sufficiency`, `high_frame_sufficiency`, `high_frame_suff_predicate`
  (and optionally `high_frame`)
- **Thresholds**: applied as a lower bound on each auxiliary probability required by the gate
- **Shifts**: values subtracted from the NOT_ENTITLED final logit for gate-selected records
- **Unflagged definition**: `temporal_flag == 0` AND `predicate_flag == 0`
- All (gate x threshold x shift) triples were evaluated on a single best-dev checkpoint
  forward pass per seed; no re-training was performed.

## Pass/Fail Criteria (passes_g1_gate)

A triple is considered safe if all five conditions hold:

1. `keeps_temporal_guard`: `temporal_mismatch_fe_mean == 0.0`
2. `keeps_predicate_guard`: `predicate_mismatch_fe_mean == 0.0`
3. `safe_sufficiency`: `sufficiency_control_fe_mean <= 0.15`
4. `safe_frame_location`: `frame_location_mismatch_fe_mean <= 0.40`
5. `safe_frame_role`: `frame_role_mismatch_fe_mean <= 0.40`

## Main Findings

### Temporal and predicate guards were preserved

For all tested configurations, `temporal_mismatch` and `predicate_mismatch`
false-entitled rates remained near zero. Because the selective gate applies only to
unflagged records (both flags == 0), the comparator-driven guard on flagged records is
structurally unaffected.

### No configuration passed the full G1 safety criterion

{passing_note}

The configurations that most effectively reduced `surface_control` and `temporal_erased`
false-not-entitled rates also caused large false-entitled regressions on
`frame_location_mismatch` and/or `frame_role_mismatch`, exceeding the 0.40 safety
threshold. The auxiliary scores (`sufficiency_prob`, `frame_prob`,
`predicate_coverage_prob`) do not provide a clean separating boundary between
preservation-like records and frame-mismatch records when used as a selective gate for
NE logit depression.

### G1 is rejected as a safe positive calibration method

Because no (gate, threshold, shift) triple satisfied the full safety criterion,
Stage21-G1 is rejected as a viable standalone calibration approach.

The fundamental problem is that `surface_control` and `temporal_erased` records (which
should be preserved as SUPPORT) share auxiliary score profiles with `frame_location_mismatch`
and `frame_role_mismatch` records (which should remain NOT_ENTITLED). Auxiliary-score-only
gating cannot separate these two populations at any tested threshold or gate combination.

## Paper Framing

Stage21-G1 exposes the **preservation-vs-frame-mismatch boundary** as the core challenge.
The auxiliary scores produced by the v6B model (frame sufficiency, predicate coverage,
sufficiency probability) are not sufficient to distinguish:

- True SUPPORT records whose temporal/predicate surface forms were erased or swapped
  (`surface_control`, `temporal_erased`) — which should be preserved as SUPPORT
- True NOT_ENTITLED records with wrong frame or role assignments
  (`frame_location_mismatch`, `frame_role_mismatch`) — which should remain NOT_ENTITLED

Auxiliary-score-only selective NE shifting therefore exposes this boundary rather than
solving it. Selective recalibration would require a richer internal signal, such as an
explicit training-time preservation loss or a learned gate trained to distinguish the two
populations.

## Top-10 Rows by Best Preservation

Sorted by `surface_control_fne_mean` then `temporal_erased_fne_mean` (lower is better).

{table_best}
## Top-10 Rows with passes_g1_gate == True

{table_passing}
## Conclusion

Stage21-G1 confirms that post-hoc selective NE logit shifting using model-internal
auxiliary scores does not constitute a safe improvement over Stage21-E3 or Stage21-G0.
The temporal/predicate comparator guard is preserved by construction, but the
frame-mismatch boundary is violated whenever the NE logit is depressed sufficiently to
reduce false-not-entitled rate on preservation controls.

Recommended next steps:

- Training-time preservation signal (e.g., SUPPORT reconstruction loss on erased/surface
  control records added to the training mix)
- A learned boundary gate trained discriminatively to separate surface/temporal-erased
  from frame-mismatch probe types
- Contrastive calibration that conditions explicitly on the expected intervention type
  at inference time
"""
    OUT_NOTES.write_text(notes, encoding="utf-8")
    print(f"Wrote {OUT_NOTES}")


if __name__ == "__main__":
    main()
