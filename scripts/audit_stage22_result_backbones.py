"""Audit Stage22 result JSON files for backbone type (dummy vs real).

Scans result JSONs produced by train_controlled_v6b_minimal.py and earlier
training scripts. Extracts backbone, freeze_encoder, model_name, OOD metric
presence, and pair-contrastive fields. Classifies each file and outputs a CSV
and Markdown summary so engineers can identify which runs need real-backbone reruns.

Classification rules
--------------------
implementation_only_ok
    No OOD or performance metrics present, or file is clearly a data/audit artifact
    (e.g. a _preds.json prediction export, a generated summary JSONL, or a
    dataset file). Dummy backbone is fine for these — they are plumbing validation.

needs_real_backbone_rerun
    backbone == "dummy" AND at least one OOD or performance metric is present.
    Metrics in these files are not claim-worthy and the experiment should be
    rerun with a real Mamba backbone.

claim_candidate
    backbone is NOT "dummy" (e.g. "mamba", a real model_name) AND OOD/performance
    metrics are present. These may be used as model evidence pending review.

unknown_backbone_review_needed
    OOD/performance metrics are present but backbone cannot be determined from the
    file. Manual review required.

Usage
-----
    python scripts/audit_stage22_result_backbones.py \\
        --results-dir results \\
        --glob "stage22*.json" \\
        --output-csv results/stage22_backbone_audit.csv \\
        --output-md  results/stage22_backbone_audit.md
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Keys that signal OOD / performance metric presence
# ---------------------------------------------------------------------------

_OOD_METRIC_KEYS = frozenset({
    "ood_metrics", "ood_results", "ood_eval", "stage15_results",
    "group_metrics", "ood_group_metrics",
})

_DEV_METRIC_KEYS = frozenset({
    "best_dev_metrics", "dev_metrics", "best_dev_interventions",
    "best_dev_pairwise_checks",
})

# Keys in a configuration dict that identify the backbone
_BACKBONE_CONFIG_KEYS = ("backbone", "backbone_name", "model_name")
_DUMMY_BACKBONE_VALUES = frozenset({"dummy", "dummy_backbone", "dummy-backbone"})

# Keys that signal pair-contrastive was configured
_PC_KEYS = (
    "use_pair_contrastive_frame_loss",
    "pair_contrastive_frame_data",
    "pair_contrastive_frame_loss_weight",
    "pair_contrastive_use_case",
    "pair_contrastive_valid_count",
)

# Filename patterns that are clearly data artifacts requiring no rerun
_SKIP_STEM_SUFFIXES = ("_preds", "_preds_export", "_summary", "_audit")
_SKIP_STEM_PREFIXES = ("stage22a4_pair_contrastive", "stage22a4d", "controlled_v5")


# ---------------------------------------------------------------------------
# JSON traversal helpers
# ---------------------------------------------------------------------------

def _flat_search(obj: Any, target_keys: frozenset[str]) -> dict[str, Any]:
    """Recursively find any key from target_keys in nested dicts; return first value."""
    found: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in target_keys:
                found[k] = v
            if isinstance(v, (dict, list)):
                deeper = _flat_search(v, target_keys - found.keys())
                found.update(deeper)
                if found.keys() >= target_keys:
                    break
    elif isinstance(obj, list):
        for item in obj:
            deeper = _flat_search(item, target_keys - found.keys())
            found.update(deeper)
            if found.keys() >= target_keys:
                break
    return found


def _collect_configuration(data: dict[str, Any]) -> dict[str, Any]:
    """Collect all configuration / loss_config dicts from the JSON, merged."""
    cfg: dict[str, Any] = {}
    # top-level configuration
    top_cfg = data.get("configuration")
    if isinstance(top_cfg, dict):
        cfg.update(top_cfg)
    # runs.*.configuration and runs.*.loss_config
    runs = data.get("runs")
    if isinstance(runs, dict):
        for run_val in runs.values():
            if isinstance(run_val, dict):
                run_cfg = run_val.get("configuration")
                if isinstance(run_cfg, dict):
                    cfg.update(run_cfg)
                loss_cfg = run_val.get("loss_config")
                if isinstance(loss_cfg, dict):
                    cfg.update(loss_cfg)
    return cfg


def _has_ood_metrics(data: dict[str, Any]) -> bool:
    for key in _OOD_METRIC_KEYS:
        if _flat_search(data, frozenset({key})):
            return True
    return False


def _has_dev_metrics(data: dict[str, Any]) -> bool:
    for key in _DEV_METRIC_KEYS:
        if _flat_search(data, frozenset({key})):
            return True
    return False


# ---------------------------------------------------------------------------
# Single-file record extraction
# ---------------------------------------------------------------------------

def _stage_guess(stem: str) -> str:
    """Return best stage guess from filename stem."""
    import re
    m = re.match(r"(stage\d+[a-z_]*)", stem)
    return m.group(1) if m else "unknown"


def _seed_guess(stem: str) -> str:
    """Return seed string from filename, or empty."""
    import re
    m = re.search(r"seed(\d+)", stem)
    return m.group(1) if m else ""


def _is_data_artifact(path: Path) -> bool:
    stem = path.stem.lower()
    if any(stem.endswith(suf) for suf in _SKIP_STEM_SUFFIXES):
        return True
    if any(stem.startswith(pfx) for pfx in _SKIP_STEM_PREFIXES):
        return True
    # metadata-only pattern: file has only 'metadata' and 'predictions' top-level keys
    return False


def audit_file(path: Path) -> dict[str, str]:
    """Return a flat audit row dict for a single JSON file."""
    row: dict[str, str] = {
        "file_path": str(path),
        "stage": _stage_guess(path.stem),
        "seed": _seed_guess(path.stem),
        "backbone": "",
        "freeze_encoder": "",
        "model_name": "",
        "use_pair_contrastive": "",
        "pair_contrastive_use_case": "",
        "pair_contrastive_valid_count": "",
        "has_ood_metrics": "false",
        "has_dev_metrics": "false",
        "ood_groups_detected": "",
        "classification": "unknown_backbone_review_needed",
        "note": "",
    }

    # Load JSON
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception as exc:
        row["classification"] = "parse_error"
        row["note"] = f"JSON parse error: {exc}"
        return row

    if not isinstance(data, dict):
        row["classification"] = "implementation_only_ok"
        row["note"] = "top-level is not a dict (list/primitive)"
        return row

    top_keys = set(data.keys())

    # Data-artifact quick exit (prediction exports, summaries)
    if _is_data_artifact(path) or top_keys <= {"metadata", "predictions", "records"}:
        row["classification"] = "implementation_only_ok"
        row["note"] = "prediction export or metadata-only file; no rerun needed"
        return row

    # Collect merged configuration
    cfg = _collect_configuration(data)

    # Extract backbone
    for bk in _BACKBONE_CONFIG_KEYS:
        val = cfg.get(bk)
        if val is not None:
            if bk == "backbone":
                row["backbone"] = str(val)
            elif bk == "model_name":
                row["model_name"] = str(val)
            else:
                row["backbone"] = row["backbone"] or str(val)

    # If no top-level backbone from config, try a flat search over whole file
    if not row["backbone"]:
        found = _flat_search(data, frozenset({"backbone"}))
        if "backbone" in found:
            row["backbone"] = str(found["backbone"])

    if not row["model_name"]:
        found = _flat_search(data, frozenset({"model_name"}))
        if "model_name" in found:
            row["model_name"] = str(found["model_name"])

    row["freeze_encoder"] = str(cfg.get("freeze_encoder", ""))

    # Pair contrastive fields
    pc_used = cfg.get("use_pair_contrastive_frame_loss")
    row["use_pair_contrastive"] = str(pc_used) if pc_used is not None else ""
    row["pair_contrastive_use_case"] = str(cfg.get("pair_contrastive_use_case", ""))
    row["pair_contrastive_valid_count"] = str(cfg.get("pair_contrastive_valid_count", ""))

    # OOD / dev metric presence
    has_ood = _has_ood_metrics(data)
    has_dev = _has_dev_metrics(data)
    row["has_ood_metrics"] = "true" if has_ood else "false"
    row["has_dev_metrics"] = "true" if has_dev else "false"

    # OOD group names (look for group_metrics or stage15_results keys)
    ood_groups: list[str] = []
    for key in ("group_metrics", "ood_group_metrics", "stage15_results", "ood_results"):
        found = _flat_search(data, frozenset({key}))
        if key in found and isinstance(found[key], dict):
            ood_groups.extend(found[key].keys())
    if ood_groups:
        row["ood_groups_detected"] = "|".join(sorted(set(ood_groups)))

    has_metrics = has_ood or has_dev

    # Classify
    backbone_val = row["backbone"].strip().lower()
    is_dummy = backbone_val in _DUMMY_BACKBONE_VALUES

    if not has_metrics:
        row["classification"] = "implementation_only_ok"
        row["note"] = "no OOD or dev metrics found; plumbing/audit file only"
    elif is_dummy:
        row["classification"] = "needs_real_backbone_rerun"
        row["note"] = (
            "backbone=dummy with performance metrics; "
            "results are not claim-worthy; rerun with real Mamba backbone"
        )
    elif backbone_val and backbone_val not in _DUMMY_BACKBONE_VALUES:
        row["classification"] = "claim_candidate"
        row["note"] = f"non-dummy backbone ({row['backbone']}); metrics may be used as evidence pending review"
    else:
        row["classification"] = "unknown_backbone_review_needed"
        row["note"] = "performance metrics present but backbone field missing; manual review required"

    return row


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------

def scan_results(
    results_dir: Path,
    glob_pattern: str,
    *,
    include_nested: bool = True,
) -> list[dict[str, str]]:
    """Scan all matching JSON files and return audit rows."""
    if include_nested:
        all_json = list(results_dir.rglob("*.json"))
    else:
        all_json = list(results_dir.glob("*.json"))

    matched = [
        p for p in all_json
        if fnmatch.fnmatch(p.name, glob_pattern)
    ]
    matched.sort(key=lambda p: (p.parent, p.stem))

    rows: list[dict[str, str]] = []
    for path in matched:
        rows.append(audit_file(path))
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "file_path", "stage", "seed", "backbone", "freeze_encoder", "model_name",
    "use_pair_contrastive", "pair_contrastive_use_case", "pair_contrastive_valid_count",
    "has_ood_metrics", "has_dev_metrics", "ood_groups_detected",
    "classification", "note",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    from collections import Counter

    by_backbone: Counter[str] = Counter(r["backbone"] or "(unknown)" for r in rows)
    by_class: Counter[str] = Counter(r["classification"] for r in rows)

    needs_rerun = [r for r in rows if r["classification"] == "needs_real_backbone_rerun"]
    unknown_bk = [r for r in rows if r["classification"] == "unknown_backbone_review_needed"]
    claim_candidates = [r for r in rows if r["classification"] == "claim_candidate"]

    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"{'#' * level} {text}\n")

    def p(text: str) -> None:
        lines.append(text + "\n")

    def table(headers: list[str], rows_: list[list[str]]) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row_ in rows_:
            lines.append("| " + " | ".join(str(c) for c in row_) + " |")
        lines.append("")

    h(1, "Stage22 result backbone audit")
    p(f"**Results dir:** `{args.results_dir}`  |  **Glob:** `{args.glob}`")
    p(f"**Total files scanned:** {len(rows)}")
    p("")

    h(2, "Important note on dummy-backbone results")
    p(
        "Results produced with `backbone=dummy` are valid for plumbing and smoke "
        "validation (confirming that loss paths, CLI wiring, and metric reporting "
        "work correctly). They are **not claim-worthy** as model performance evidence "
        "because the dummy backbone has no representation capacity and cannot learn "
        "meaningful features from text."
    )
    p(
        "Dataset generation scripts, audit scripts, and summary JSONL files "
        "do not need rerun — they are data artifacts, not experiment results."
    )
    p("")

    h(2, "Count by detected backbone")
    table(
        ["backbone", "count"],
        [[bk, str(cnt)] for bk, cnt in sorted(by_backbone.items(), key=lambda x: -x[1])],
    )

    h(2, "Count by classification")
    table(
        ["classification", "count", "description"],
        [
            ["needs_real_backbone_rerun", str(by_class.get("needs_real_backbone_rerun", 0)),
             "backbone=dummy with metrics; not claim-worthy; must rerun"],
            ["claim_candidate", str(by_class.get("claim_candidate", 0)),
             "non-dummy backbone with metrics; may be used as evidence"],
            ["unknown_backbone_review_needed", str(by_class.get("unknown_backbone_review_needed", 0)),
             "metrics present but backbone unknown; manual review required"],
            ["implementation_only_ok", str(by_class.get("implementation_only_ok", 0)),
             "no performance metrics or data artifact; no rerun needed"],
            ["parse_error", str(by_class.get("parse_error", 0)),
             "JSON could not be parsed"],
        ],
    )

    h(2, "Files needing real-backbone rerun")
    if needs_rerun:
        table(
            ["file", "stage", "backbone", "use_pair_contrastive", "pair_use_case"],
            [
                [
                    Path(r["file_path"]).name,
                    r["stage"], r["backbone"],
                    r["use_pair_contrastive"],
                    r["pair_contrastive_use_case"],
                ]
                for r in needs_rerun
            ],
        )
    else:
        p("_None detected._")

    h(2, "Files with unknown backbone but performance metrics")
    if unknown_bk:
        table(
            ["file", "stage", "has_ood_metrics", "has_dev_metrics"],
            [
                [
                    Path(r["file_path"]).name,
                    r["stage"],
                    r["has_ood_metrics"],
                    r["has_dev_metrics"],
                ]
                for r in unknown_bk
            ],
        )
    else:
        p("_None detected._")

    h(2, "Claim candidates (non-dummy backbone with metrics)")
    if claim_candidates:
        table(
            ["file", "stage", "backbone", "model_name"],
            [
                [
                    Path(r["file_path"]).name,
                    r["stage"], r["backbone"], r["model_name"],
                ]
                for r in claim_candidates
            ],
        )
    else:
        p("_None detected._")

    h(2, "Recommended next step")
    p(
        "1. For each file in **needs_real_backbone_rerun**: rerun the corresponding "
        "Kaggle experiment with the real Mamba backbone. The dummy result can be "
        "kept as a plumbing reference but must not be cited as model performance."
    )
    p(
        "2. For each file in **unknown_backbone_review_needed**: open the file and "
        "manually confirm whether a real backbone was used. If yes, reclassify as "
        "claim_candidate; if no, rerun."
    )
    p(
        "3. **implementation_only_ok** files require no action. "
        "Dataset JSONL, audit CSVs, and prediction export files are not affected "
        "by backbone type."
    )
    p(
        "4. **parse_error** files should be inspected for corruption or incomplete writes."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Audit Stage22 result JSON files to identify dummy-backbone vs "
            "real-backbone runs. Outputs a CSV and Markdown summary."
        )
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory to scan for result JSON files (default: results).",
    )
    p.add_argument(
        "--glob",
        default="stage22*.json",
        help="Glob pattern for JSON filenames (default: stage22*.json).",
    )
    p.add_argument(
        "--output-csv",
        required=True,
        help="Path to write the output CSV audit table.",
    )
    p.add_argument(
        "--output-md",
        required=True,
        help="Path to write the output Markdown summary.",
    )
    p.add_argument(
        "--include-nested",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recursively scan subdirectories under results-dir (default: true). "
            "Use --no-include-nested to restrict to the top-level directory only."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: results-dir not found: {results_dir}", file=sys.stderr)
        return 1

    print(
        f"Scanning {results_dir!s} (glob={args.glob!r}"
        f" nested={args.include_nested}) ..."
    )
    rows = scan_results(results_dir, args.glob, include_nested=args.include_nested)
    print(f"  {len(rows)} files matched")

    # Summary counts
    from collections import Counter
    by_class: Counter[str] = Counter(r["classification"] for r in rows)
    for cls, cnt in sorted(by_class.items()):
        print(f"  {cls}: {cnt}")

    out_csv = Path(args.output_csv)
    write_csv(out_csv, rows)
    print(f"CSV written: {out_csv}")

    out_md = Path(args.output_md)
    write_md(out_md, rows, args)
    print(f"Markdown written: {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
