"""Global backbone audit for all result JSON files across all stages.

Scans result JSONs produced by all training scripts in this repo. Extracts
backbone, model_name, freeze_encoder, device, data paths, OOD/dev metric
presence, and pair-contrastive fields. Classifies each file and optionally
infers backbone for OOD-companion JSONs from their paired main-seed file.

Classification rules
--------------------
implementation_only_ok
    No OOD or performance metrics present, or file is clearly a data/audit
    artifact (prediction export, dataset summary, audit output). Dummy backbone
    is acceptable — these are plumbing or data artifacts.

needs_real_backbone_rerun
    Explicit or inferred backbone is "dummy" AND performance/OOD metrics are
    present. Metrics are NOT claim-worthy. Rerun with real Mamba backbone.

claim_candidate
    Explicit or inferred backbone is non-dummy (e.g. "mamba") AND
    performance/OOD metrics are present. May be cited as model evidence
    pending review.

unknown_backbone_review_needed
    Performance/OOD metrics are present but backbone cannot be determined
    (no explicit value, no usable inference). Manual review required.

parse_error
    JSON could not be parsed.

OOD companion inference
-----------------------
Files matching "*_ood_seed*.json" or "*_ood.json" patterns often lack the
"configuration" block that holds backbone info, because they are companion
outputs to a paired main-seed file. When --infer-ood-companion-backbone is
enabled (default), the script looks for the most likely paired main file by
removing "_ood" from the stem and checking whether that file exists in the
same directory. If found and its backbone is extractable, the companion file
records inferred_backbone and inferred_from, and classification uses the
inferred value.

Usage
-----
    python scripts/audit_result_backbones.py \\
        --results-dir results \\
        --glob "*.json" \\
        --output-csv results/global_backbone_audit.csv \\
        --output-md  results/global_backbone_audit.md
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Keys that signal OOD / performance metric presence
# ---------------------------------------------------------------------------

_OOD_METRIC_KEYS = frozenset({
    "ood_metrics", "ood_results", "ood_eval", "stage15_results",
    "group_metrics", "ood_group_metrics", "ood_group_metrics_by_probe_type",
    # Branch-specific keys from train_controlled_v6b_minimal.py
    "ood_selective_ne_shift_sweep", "ood_unflagged_ne_shift_sweep", "ood_ablation",
})

_DEV_METRIC_KEYS = frozenset({
    "best_dev_metrics", "dev_metrics", "best_dev_interventions",
    "best_dev_pairwise_checks", "train_dev_report",
})

# Keys in any config/metadata dict that identify backbone
_BACKBONE_KEYS = ("backbone", "backbone_name", "model_name")
_DUMMY_BACKBONE_VALUES = frozenset({"dummy", "dummy_backbone", "dummy-backbone"})

# Keys that signal pair-contrastive was used
_PC_KEYS = (
    "use_pair_contrastive_frame_loss",
    "pair_contrastive_frame_data",
    "pair_contrastive_use_case",
    "pair_contrastive_valid_count",
    "pair_contrastive_frame_loss_weight",
    "pair_contrastive_frame_margin",
)

# Filename suffix/prefix patterns indicating data artifacts (no rerun needed)
_SKIP_STEM_SUFFIXES = ("_preds", "_preds_export", "_summary", "_audit")
_SKIP_STEM_PREFIXES = (
    "stage22a4_pair_contrastive", "stage22a4d", "controlled_v5",
    "global_backbone_audit", "stage22_backbone_audit",
)


# ---------------------------------------------------------------------------
# JSON traversal helpers
# ---------------------------------------------------------------------------

def _flat_search(obj: Any, keys: frozenset[str]) -> dict[str, Any]:
    """Return first occurrence of each key found anywhere in the nested structure."""
    found: dict[str, Any] = {}
    remaining = keys - found.keys()
    if not remaining:
        return found
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in remaining:
                found[k] = v
                remaining = keys - found.keys()
            if remaining and isinstance(v, (dict, list)):
                deeper = _flat_search(v, remaining)
                found.update(deeper)
                remaining = keys - found.keys()
            if not remaining:
                break
    elif isinstance(obj, list):
        for item in obj:
            if not remaining:
                break
            deeper = _flat_search(item, remaining)
            found.update(deeper)
            remaining = keys - found.keys()
    return found


def _collect_config(data: dict[str, Any]) -> dict[str, Any]:
    """Merge all configuration sources in the JSON into one flat dict.

    Sources (applied in order; later sources fill only keys already missing):
      1. data["configuration"]              -- stage12 / train_controlled_v6b_minimal
      2. data["runs"][*]["configuration"]   -- per-run configs
      3. data["runs"][*]["loss_config"]     -- per-run loss configs
      4. data["metadata"]                   -- stage13–stage16 and companions
      5. data["ood_provenance"]             -- OOD-only JSONs (stage22 A4+)
    """
    cfg: dict[str, Any] = {}

    def _merge(src: Any) -> None:
        if isinstance(src, dict):
            for k, v in src.items():
                cfg.setdefault(k, v)

    _merge(data.get("configuration"))

    runs = data.get("runs")
    if isinstance(runs, dict):
        for run_val in runs.values():
            if isinstance(run_val, dict):
                _merge(run_val.get("configuration"))
                _merge(run_val.get("loss_config"))

    _merge(data.get("metadata"))
    _merge(data.get("ood_provenance"))
    return cfg


# ---------------------------------------------------------------------------
# OOD companion inference
# ---------------------------------------------------------------------------

def _ood_companion_candidates(stem: str) -> list[str]:
    """Generate candidate main-file stems from an OOD companion stem.

    Handles patterns like:
      foo_ood_seed1      -> foo_seed1
      foo_mini_ood_seed1 -> foo_mini_seed1
      foo_ood            -> foo
    """
    candidates: list[str] = []
    # Replace _ood_ with _ (covers the interior position _ood_seed / _ood_mini)
    if "_ood_" in stem:
        c = stem.replace("_ood_", "_", 1)
        if c != stem:
            candidates.append(c)
    # Trailing _ood
    if stem.endswith("_ood"):
        candidates.append(stem[:-4])
    return candidates


def _extract_backbone_from_file(path: Path) -> str:
    """Load a JSON file and return its backbone string, or '' if undetectable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    cfg = _collect_config(data)
    for key in _BACKBONE_KEYS:
        val = cfg.get(key)
        if val is not None:
            return str(val)
    # Flat search fallback
    found = _flat_search(data, frozenset({"backbone"}))
    return str(found["backbone"]) if "backbone" in found else ""


def _infer_companion_backbone(
    path: Path, all_paths_by_name: dict[str, Path]
) -> tuple[str, str]:
    """Try to infer backbone for an OOD companion file.

    Returns (inferred_backbone, inferred_from_filename).
    """
    stem = path.stem
    for candidate_stem in _ood_companion_candidates(stem):
        candidate_name = candidate_stem + ".json"
        candidate_path = all_paths_by_name.get(candidate_name)
        if candidate_path is not None:
            bk = _extract_backbone_from_file(candidate_path)
            if bk:
                return bk, candidate_path.name
    return "", ""


def _looks_like_ood_companion(stem: str) -> bool:
    """True when the filename looks like an OOD companion output."""
    return "_ood_" in stem or stem.endswith("_ood")


# ---------------------------------------------------------------------------
# Stage / seed guess
# ---------------------------------------------------------------------------

def _stage_guess(stem: str) -> str:
    m = re.match(r"(stage[0-9]+[a-z0-9]*)", stem)
    return m.group(1) if m else "unknown"


def _seed_guess(stem: str) -> str:
    m = re.search(r"seed(\d+)", stem)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Data-artifact detection
# ---------------------------------------------------------------------------

def _is_data_artifact(path: Path, data: dict[str, Any]) -> bool:
    stem = path.stem.lower()
    if any(stem.endswith(s) for s in _SKIP_STEM_SUFFIXES):
        return True
    if any(stem.startswith(s) for s in _SKIP_STEM_PREFIXES):
        return True
    # Files with only metadata + predictions / records
    if set(data.keys()) <= {"metadata", "predictions", "records"}:
        # But allow if metadata has backbone — it may be a real eval summary
        meta = data.get("metadata", {})
        if not isinstance(meta, dict) or not any(k in meta for k in _BACKBONE_KEYS):
            return True
    return False


# ---------------------------------------------------------------------------
# Single-file audit
# ---------------------------------------------------------------------------

def audit_file(
    path: Path,
    all_paths_by_name: dict[str, Path],
    *,
    infer_ood_companion: bool,
) -> dict[str, str]:
    row: dict[str, str] = {
        "file_path": str(path),
        "stage": _stage_guess(path.stem),
        "seed": _seed_guess(path.stem),
        "backbone": "",
        "inferred_backbone": "",
        "inferred_from": "",
        "effective_backbone": "",
        "freeze_encoder": "",
        "model_name": "",
        "device": "",
        "data_path": "",
        "ood_data_path": "",
        "use_pair_contrastive": "",
        "pair_contrastive_use_case": "",
        "pair_contrastive_valid_count": "",
        "has_ood_metrics": "false",
        "has_dev_metrics": "false",
        "has_group_metrics": "false",
        "classification": "unknown_backbone_review_needed",
        "note": "",
    }

    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception as exc:
        row["classification"] = "parse_error"
        row["note"] = f"JSON parse error: {exc}"
        return row

    if not isinstance(data, dict):
        row["classification"] = "implementation_only_ok"
        row["note"] = "top-level is not a dict"
        return row

    # Data-artifact short-circuit
    if _is_data_artifact(path, data):
        row["classification"] = "implementation_only_ok"
        row["note"] = "prediction export or metadata-only artifact; no rerun needed"
        return row

    # Collect merged config
    cfg = _collect_config(data)

    # Extract backbone from config
    for key in _BACKBONE_KEYS:
        val = cfg.get(key)
        if val is not None:
            if key == "backbone":
                row["backbone"] = str(val)
            elif key == "model_name":
                row["model_name"] = row["model_name"] or str(val)
            else:
                row["backbone"] = row["backbone"] or str(val)

    # Flat-search fallback for backbone
    if not row["backbone"]:
        found = _flat_search(data, frozenset({"backbone"}))
        if "backbone" in found:
            row["backbone"] = str(found["backbone"])

    if not row["model_name"]:
        found = _flat_search(data, frozenset({"model_name"}))
        if "model_name" in found:
            row["model_name"] = str(found["model_name"])

    row["freeze_encoder"] = str(cfg.get("freeze_encoder", ""))
    row["device"] = str(cfg.get("device", ""))

    # Data paths
    for data_key in ("data", "train_data", "train"):
        val = cfg.get(data_key)
        if val:
            row["data_path"] = str(val)
            break
    for ood_key in ("ood_data", "ood_data_path", "ood"):
        val = cfg.get(ood_key)
        if val:
            row["ood_data_path"] = str(val)
            break

    # Pair-contrastive fields
    pc_used = cfg.get("use_pair_contrastive_frame_loss")
    row["use_pair_contrastive"] = str(pc_used) if pc_used is not None else ""
    row["pair_contrastive_use_case"] = str(cfg.get("pair_contrastive_use_case", ""))
    row["pair_contrastive_valid_count"] = str(cfg.get("pair_contrastive_valid_count", ""))

    # Metric presence
    has_ood = any(_flat_search(data, frozenset({k})) for k in _OOD_METRIC_KEYS)
    has_dev = any(_flat_search(data, frozenset({k})) for k in _DEV_METRIC_KEYS)
    has_grp = bool(_flat_search(data, frozenset({"group_metrics", "ood_group_metrics"})))
    row["has_ood_metrics"] = "true" if has_ood else "false"
    row["has_dev_metrics"] = "true" if has_dev else "false"
    row["has_group_metrics"] = "true" if has_grp else "false"

    has_metrics = has_ood or has_dev

    # OOD companion inference
    if infer_ood_companion and not row["backbone"] and _looks_like_ood_companion(path.stem):
        inferred_bk, inferred_from = _infer_companion_backbone(path, all_paths_by_name)
        row["inferred_backbone"] = inferred_bk
        row["inferred_from"] = inferred_from

    # Effective backbone: explicit wins, then inferred
    eff_bk = row["backbone"] or row["inferred_backbone"]
    row["effective_backbone"] = eff_bk
    is_dummy = eff_bk.lower() in _DUMMY_BACKBONE_VALUES
    has_explicit_bk = bool(row["backbone"])

    # Classify
    if not has_metrics:
        row["classification"] = "implementation_only_ok"
        row["note"] = "no OOD or dev metrics; plumbing/audit file"
    elif is_dummy:
        inferred_tag = "" if has_explicit_bk else " (inferred)"
        row["classification"] = "needs_real_backbone_rerun"
        row["note"] = (
            f"backbone=dummy{inferred_tag}; metrics present but not claim-worthy; "
            "rerun with real Mamba backbone"
        )
    elif eff_bk and not is_dummy:
        inferred_tag = "" if has_explicit_bk else f" (inferred from {row['inferred_from']})"
        row["classification"] = "claim_candidate"
        row["note"] = f"non-dummy backbone ({eff_bk}){inferred_tag}; metrics may be cited pending review"
    else:
        row["classification"] = "unknown_backbone_review_needed"
        row["note"] = "metrics present but backbone not determinable; manual review required"

    return row


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------

def scan_results(
    results_dir: Path,
    glob_pattern: str,
    *,
    include_nested: bool,
    infer_ood_companion: bool,
) -> list[dict[str, str]]:
    if include_nested:
        all_json = list(results_dir.rglob("*.json"))
    else:
        all_json = list(results_dir.glob("*.json"))

    matched = sorted(
        (p for p in all_json if fnmatch.fnmatch(p.name, glob_pattern)),
        key=lambda p: (p.parent, p.stem),
    )

    # Build name→path index for companion inference
    all_paths_by_name: dict[str, Path] = {p.name: p for p in all_json}

    return [
        audit_file(path, all_paths_by_name, infer_ood_companion=infer_ood_companion)
        for path in matched
    ]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "file_path", "stage", "seed",
    "backbone", "inferred_backbone", "inferred_from", "effective_backbone",
    "freeze_encoder", "model_name", "device",
    "data_path", "ood_data_path",
    "use_pair_contrastive", "pair_contrastive_use_case", "pair_contrastive_valid_count",
    "has_ood_metrics", "has_dev_metrics", "has_group_metrics",
    "classification", "note",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _group_by_stage(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[r["stage"]].append(r)
    return dict(sorted(groups.items()))


def write_md(path: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    from collections import Counter

    by_explicit_bk: Counter[str] = Counter(r["backbone"] or "(unknown)" for r in rows)
    by_inferred_bk: Counter[str] = Counter(
        r["inferred_backbone"] for r in rows if r["inferred_backbone"]
    )
    by_class: Counter[str] = Counter(r["classification"] for r in rows)

    needs_rerun = [r for r in rows if r["classification"] == "needs_real_backbone_rerun"]
    unknown_bk = [r for r in rows if r["classification"] == "unknown_backbone_review_needed"]
    claim_cands = [r for r in rows if r["classification"] == "claim_candidate"]

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

    h(1, "Global result backbone audit")
    p(f"**Results dir:** `{args.results_dir}`  |  **Glob:** `{args.glob}`")
    p(
        f"**OOD companion inference:** "
        f"{'enabled' if args.infer_ood_companion_backbone else 'disabled'}"
    )
    p(f"**Total files scanned:** {len(rows)}")
    p("")

    h(2, "Important note on dummy-backbone results")
    p(
        "Results produced with `backbone=dummy` are valid for **plumbing and smoke "
        "validation** (confirming that loss paths, CLI wiring, encoding, and metric "
        "reporting work correctly). They are **not claim-worthy** as model performance "
        "evidence because the dummy backbone has no text comprehension capacity."
    )
    p(
        "Dataset generation scripts, audit scripts, summary JSONs, and prediction "
        "export files do not require rerun regardless of when they were produced."
    )
    p("")

    h(2, "Count by explicit backbone")
    table(
        ["backbone", "files"],
        [[bk, str(cnt)] for bk, cnt in sorted(by_explicit_bk.items(), key=lambda x: -x[1])],
    )

    if by_inferred_bk:
        h(2, "Count by inferred backbone (OOD companion inference)")
        table(
            ["inferred_backbone", "files"],
            [[bk, str(cnt)] for bk, cnt in sorted(by_inferred_bk.items(), key=lambda x: -x[1])],
        )

    h(2, "Count by classification")
    table(
        ["classification", "count", "description"],
        [
            ["needs_real_backbone_rerun",
             str(by_class.get("needs_real_backbone_rerun", 0)),
             "backbone=dummy with metrics; not claim-worthy; must rerun"],
            ["claim_candidate",
             str(by_class.get("claim_candidate", 0)),
             "non-dummy backbone with metrics; may be cited as evidence"],
            ["unknown_backbone_review_needed",
             str(by_class.get("unknown_backbone_review_needed", 0)),
             "metrics present but backbone unknown; manual review"],
            ["implementation_only_ok",
             str(by_class.get("implementation_only_ok", 0)),
             "no performance metrics or data artifact; no rerun needed"],
            ["parse_error",
             str(by_class.get("parse_error", 0)),
             "JSON could not be parsed"],
        ],
    )

    h(2, "Files needing real-backbone rerun (by stage)")
    if needs_rerun:
        for stage, grp in _group_by_stage(needs_rerun).items():
            h(3, f"Stage: `{stage}`")
            table(
                ["file", "backbone", "effective_backbone", "inferred_from",
                 "has_ood", "has_dev", "pair_use_case"],
                [
                    [
                        Path(r["file_path"]).name,
                        r["backbone"], r["effective_backbone"], r["inferred_from"],
                        r["has_ood_metrics"], r["has_dev_metrics"],
                        r["pair_contrastive_use_case"],
                    ]
                    for r in grp
                ],
            )
    else:
        p("_None detected._")

    h(2, "Files with unknown backbone but performance metrics (by stage)")
    if unknown_bk:
        for stage, grp in _group_by_stage(unknown_bk).items():
            h(3, f"Stage: `{stage}`")
            table(
                ["file", "has_ood", "has_dev", "has_group"],
                [
                    [
                        Path(r["file_path"]).name,
                        r["has_ood_metrics"], r["has_dev_metrics"], r["has_group_metrics"],
                    ]
                    for r in grp
                ],
            )
    else:
        p("_None detected._")

    h(2, "Claim candidates — non-dummy backbone with metrics (by stage)")
    if claim_cands:
        for stage, grp in _group_by_stage(claim_cands).items():
            h(3, f"Stage: `{stage}`")
            table(
                ["file", "backbone", "model_name", "inferred_from"],
                [
                    [
                        Path(r["file_path"]).name,
                        r["backbone"], r["model_name"], r["inferred_from"],
                    ]
                    for r in grp
                ],
            )
    else:
        p("_None detected. All metric-bearing files used dummy backbone or have unknown backbone._")

    h(2, "Recommended next steps")
    p(
        "1. For each file in **needs_real_backbone_rerun**: rerun the corresponding "
        "Kaggle experiment with a real Mamba backbone. "
        "Keep dummy results as plumbing references but do not cite as performance evidence."
    )
    p(
        "2. For each file in **unknown_backbone_review_needed**: open the file and "
        "confirm manually whether a real backbone was used. "
        "If yes, the file is a claim candidate. If dummy, it needs rerun."
    )
    p(
        "3. **implementation_only_ok** files need no action regardless of backbone type."
    )
    p(
        "4. **parse_error** files should be inspected for incomplete writes or corruption."
    )
    p(
        "5. If **claim_candidate** count is 0 across all stages, no experiment in this "
        "repo has produced claim-worthy OOD or dev metrics with a real backbone yet."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Global backbone audit: scan all result JSON files across all stages "
            "to identify dummy-backbone vs real-backbone runs."
        )
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory to scan for result JSON files (default: results).",
    )
    p.add_argument(
        "--glob",
        default="*.json",
        help="Glob pattern matched against filenames (default: *.json).",
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
        help="Recursively scan subdirectories (default: true).",
    )
    p.add_argument(
        "--infer-ood-companion-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Infer backbone for OOD companion JSONs (files with _ood_ or _ood suffix) "
            "by looking up the paired main-seed file in the same directory (default: true). "
            "Use --no-infer-ood-companion-backbone to disable."
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
        f"Scanning {results_dir!s}"
        f" (glob={args.glob!r}"
        f" nested={args.include_nested}"
        f" infer_companion={args.infer_ood_companion_backbone}) ..."
    )
    rows = scan_results(
        results_dir,
        args.glob,
        include_nested=args.include_nested,
        infer_ood_companion=args.infer_ood_companion_backbone,
    )
    print(f"  {len(rows)} files matched")

    from collections import Counter
    by_class: Counter[str] = Counter(r["classification"] for r in rows)
    for cls in ("needs_real_backbone_rerun", "claim_candidate",
                "unknown_backbone_review_needed", "implementation_only_ok", "parse_error"):
        cnt = by_class.get(cls, 0)
        if cnt:
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
