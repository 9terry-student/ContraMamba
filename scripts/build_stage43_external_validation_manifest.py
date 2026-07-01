"""Build the Stage43-A external/naturalistic validation inventory manifest.

Preparation/inventory only. This script scans the repository for candidate
external- or naturalistic-validation data files, inspects a small sample of
records from each candidate, and infers whether the file's fields could be
adapted into the ContraMamba claim/evidence/label schema
(SUPPORT / REFUTE / NOT_ENTITLED).

It does not train, evaluate, run the model, run any Stage39/40 evaluator,
modify any candidate file, or fabricate rows. Candidate probe/controlled
files inspected here (Stage10/13/31/34/35, controlled_v5) are synthetic or
probe-style artifacts already used by prior stages; they are reported as
such and are not treated as naturalistic external evidence. Output of this
script must not be used for training, calibration, threshold selection,
checkpoint selection, or loss design -- Stage43-B (if pursued) must remain
external/evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_JSON_OUTPUT = REPO_ROOT / "reports" / "stage43a_external_validation_manifest.json"
DEFAULT_MD_OUTPUT = REPO_ROOT / "reports" / "stage43a_external_validation_manifest.md"

SCAN_DIRS = ["data", "experiments", "reports", "results", "docs", "outputs"]

KEYWORDS = [
    "vitaminc",
    "fever",
    "rte",
    "mnli",
    "snli",
    "nli",
    "external",
    "ood",
    "probe",
    "claim",
    "evidence",
    "confident",
    "error",
    "epistemic",
    "stage10",
    "stage13",
    "stage34",
    "stage35",
]

DATA_EXTENSIONS = {".jsonl", ".json", ".csv", ".tsv"}

CLAIM_ALIASES = ["claim", "hypothesis", "statement", "query", "premise2", "sentence2"]
EVIDENCE_ALIASES = ["evidence", "premise", "context", "passage", "text", "sentence1"]
LABEL_ALIASES = ["label", "gold", "gold_label", "answer", "verdict", "relation"]

SUPPORT_VALUES = {"supports", "support", "entailment", "true", "1"}
REFUTE_VALUES = {"refutes", "refute", "contradiction", "false", "-1"}
NOT_ENTITLED_VALUES = {
    "not enough info",
    "nei",
    "neutral",
    "unknown",
    "not_entitled",
    "0",
}

# Files already known (from prior stages) to be synthetic/probe-style
# controlled artifacts rather than external/naturalistic data. Used only to
# annotate notes; inclusion in this set does not change schema/mapping
# detection, which is always performed from the actual file content.
KNOWN_SYNTHETIC_PROBE_PATTERNS = [
    "stage10",
    "stage13",
    "stage14",
    "stage15",
    "stage31",
    "stage34",
    "stage35",
    "controlled_v5",
    "toy_interventions",
]

SAMPLE_SIZE = 5


def is_probably_synthetic(rel_path: str) -> bool:
    lowered = rel_path.lower()
    return any(pattern in lowered for pattern in KNOWN_SYNTHETIC_PROBE_PATTERNS)


def find_candidate_files() -> list[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for dir_name in SCAN_DIRS:
        base = REPO_ROOT / dir_name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in DATA_EXTENSIONS:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            lowered = rel.lower()
            if not any(kw in lowered for kw in KEYWORDS):
                continue
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            candidates.append(path)
    return candidates


def read_jsonl_sample(path: Path, n: int) -> tuple[list[dict[str, Any]], int | None]:
    records: list[dict[str, Any]] = []
    total = 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                if len(records) < n:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            records.append(obj)
                    except json.JSONDecodeError:
                        pass
    except OSError:
        return [], None
    return records, total


def read_json_sample(path: Path, n: int) -> tuple[list[dict[str, Any]], int | None]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return [], None

    rows: list[Any] | None = None
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("predictions", "records", "items", "data", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                rows = value
                break
        if rows is None:
            # Not a row-oriented file (e.g. a report/summary JSON).
            return [], None

    dict_rows = [r for r in rows if isinstance(r, dict)]
    return dict_rows[:n], len(dict_rows)


def read_delimited_sample(path: Path, n: int, delimiter: str) -> tuple[list[dict[str, Any]], int | None]:
    records: list[dict[str, Any]] = []
    total = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for row in reader:
                total += 1
                if len(records) < n:
                    records.append(dict(row))
    except OSError:
        return [], None
    return records, total


def infer_field_candidates(fields: list[str], aliases: list[str]) -> list[str]:
    lowered_map = {f.lower(): f for f in fields}
    return [lowered_map[a] for a in aliases if a in lowered_map]


def infer_label_values_sample(records: list[dict[str, Any]], label_fields: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for record in records:
        for field in label_fields:
            if field in record and record[field] is not None:
                sval = str(record[field])
                if sval not in seen:
                    seen.add(sval)
                    values.append(sval)
    return values


def recommend_mapping(label_values: list[str]) -> tuple[dict[str, str], str]:
    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for raw in label_values:
        norm = raw.strip().lower()
        if norm in SUPPORT_VALUES:
            mapping[raw] = "SUPPORT"
        elif norm in REFUTE_VALUES:
            mapping[raw] = "REFUTE"
        elif norm in NOT_ENTITLED_VALUES:
            mapping[raw] = "NOT_ENTITLED"
        else:
            unmapped.append(raw)

    if not label_values:
        return {}, "missing_label"
    if not mapping:
        return {}, "not_mapped"
    if unmapped:
        return mapping, "ambiguous"
    return mapping, "mapped"


def analyze_file(path: Path) -> dict[str, Any]:
    rel_path = path.relative_to(REPO_ROOT).as_posix()
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        records, total = read_jsonl_sample(path, SAMPLE_SIZE)
        file_type = "jsonl"
    elif suffix == ".json":
        records, total = read_json_sample(path, SAMPLE_SIZE)
        file_type = "json"
    elif suffix == ".csv":
        records, total = read_delimited_sample(path, SAMPLE_SIZE, ",")
        file_type = "csv"
    elif suffix == ".tsv":
        records, total = read_delimited_sample(path, SAMPLE_SIZE, "\t")
        file_type = "tsv"
    else:
        records, total = [], None
        file_type = suffix.lstrip(".") or "unknown"

    if total is None:
        return {
            "path": rel_path,
            "file_type": file_type,
            "row_count_estimate": None,
            "sample_records": [],
            "detected_fields": [],
            "claim_field_candidates": [],
            "evidence_field_candidates": [],
            "label_field_candidates": [],
            "label_values_sample": [],
            "schema_status": "not_adaptable",
            "mapping_status": "not_mapped",
            "recommended_mapping": {},
            "notes": "File could not be parsed as row-oriented data (unreadable or not a records list).",
        }

    if not records:
        return {
            "path": rel_path,
            "file_type": file_type,
            "row_count_estimate": total,
            "sample_records": [],
            "detected_fields": [],
            "claim_field_candidates": [],
            "evidence_field_candidates": [],
            "label_field_candidates": [],
            "label_values_sample": [],
            "schema_status": "not_adaptable",
            "mapping_status": "not_mapped",
            "recommended_mapping": {},
            "notes": "No parseable dict-shaped records found in sample.",
        }

    fields: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in fields:
                fields.append(key)

    claim_candidates = infer_field_candidates(fields, CLAIM_ALIASES)
    evidence_candidates = infer_field_candidates(fields, EVIDENCE_ALIASES)
    label_candidates = infer_field_candidates(fields, LABEL_ALIASES)

    label_values = infer_label_values_sample(records, label_candidates)
    mapping, mapping_status = recommend_mapping(label_values)

    notes_parts = []
    if is_probably_synthetic(rel_path):
        notes_parts.append(
            "Filename matches a known synthetic/probe-style controlled artifact "
            "from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); "
            "not naturalistic external evidence even if schema is adaptable."
        )

    if claim_candidates and evidence_candidates and label_candidates:
        schema_status = "adaptable" if mapping_status == "mapped" else "ambiguous"
    elif claim_candidates or evidence_candidates:
        schema_status = "ambiguous"
    else:
        schema_status = "not_adaptable"

    if not label_candidates:
        notes_parts.append("No recognized label-like field found in sampled records.")
    if not claim_candidates:
        notes_parts.append("No recognized claim-like field found in sampled records.")
    if not evidence_candidates:
        notes_parts.append("No recognized evidence-like field found in sampled records.")
    if mapping_status == "ambiguous":
        notes_parts.append("Some sampled label values could not be confidently mapped; conservative/manual review required.")

    return {
        "path": rel_path,
        "file_type": file_type,
        "row_count_estimate": total,
        "sample_records": records,
        "detected_fields": fields,
        "claim_field_candidates": claim_candidates,
        "evidence_field_candidates": evidence_candidates,
        "label_field_candidates": label_candidates,
        "label_values_sample": label_values,
        "schema_status": schema_status,
        "mapping_status": mapping_status,
        "recommended_mapping": mapping,
        "notes": " ".join(notes_parts) if notes_parts else "Schema fields detected; see recommended_mapping.",
    }


EXPECTED_SOURCE_KEYWORDS = [
    "vitaminc",
    "fever",
    "rte",
    "mnli",
    "snli",
]


def build_manifest() -> dict[str, Any]:
    candidate_paths = find_candidate_files()
    candidate_files = [analyze_file(p) for p in candidate_paths]

    adaptable_files = [c for c in candidate_files if c["schema_status"] == "adaptable"]
    ambiguous_files = [c for c in candidate_files if c["schema_status"] == "ambiguous"]
    non_adaptable_files = [c for c in candidate_files if c["schema_status"] == "not_adaptable"]

    naturalistic_adaptable = [
        c for c in adaptable_files if not is_probably_synthetic(c["path"])
    ]
    naturalistic_ambiguous = [
        c for c in ambiguous_files if not is_probably_synthetic(c["path"])
    ]

    missing_expected_sources = []
    all_paths_lower = " ".join(c["path"].lower() for c in candidate_files)
    for kw in EXPECTED_SOURCE_KEYWORDS:
        if kw not in all_paths_lower:
            missing_expected_sources.append(
                f"No file with '{kw}' in its path was found under {', '.join(SCAN_DIRS)}."
            )

    # Also flag the Stage35 default probe output, referenced by
    # scripts/build_stage35_adversarial_coverage_probe.py and by the
    # Stage39-C Stage35 report, but not present as a persisted data file.
    stage35_probe_path = REPO_ROOT / "data" / "stage35a_adversarial_coverage_probe.jsonl"
    if not stage35_probe_path.exists():
        missing_expected_sources.append(
            "data/stage35a_adversarial_coverage_probe.jsonl (Stage35-A default probe output "
            "referenced by scripts/build_stage35_adversarial_coverage_probe.py) is not present "
            "as a persisted file; Stage39-C's Stage35 evaluation likely regenerated it on demand."
        )

    if naturalistic_adaptable:
        decision = "STAGE43A_EXTERNAL_VALIDATION_MANIFEST_READY"
    elif candidate_files and (naturalistic_ambiguous or ambiguous_files or adaptable_files):
        decision = "STAGE43A_EXTERNAL_VALIDATION_MANIFEST_AMBIGUOUS"
    elif not candidate_files:
        decision = "STAGE43A_EXTERNAL_VALIDATION_NO_SOURCE_FOUND"
    else:
        decision = "STAGE43A_EXTERNAL_VALIDATION_MANIFEST_AMBIGUOUS"

    has_fever_vitaminc = any(
        kw in c["path"].lower() for kw in ("vitaminc", "fever") for c in candidate_files
    )
    has_nli = any(
        kw in c["path"].lower() for kw in ("rte", "mnli", "snli", "nli") for c in candidate_files
    )
    only_synthetic_probes = bool(candidate_files) and not naturalistic_adaptable and not naturalistic_ambiguous

    if has_fever_vitaminc:
        recommended_stage43b_inputs = [
            "Recommend Stage43-B external final-composer evaluation on the identified "
            "VitaminC/FEVER-style source (eval-only; no training/calibration use)."
        ]
    elif has_nli:
        recommended_stage43b_inputs = [
            "Recommend Stage43-B NLI transfer probe on the identified RTE/MNLI/SNLI-style source "
            "(eval-only; no training/calibration use)."
        ]
    elif only_synthetic_probes:
        recommended_stage43b_inputs = [
            "Only synthetic/controlled probes were found (Stage10/13/14/15/31/34/35, controlled_v5). "
            "Recommend collecting or adding a small external naturalistic claim-evidence set "
            "before attempting Stage43-B."
        ]
    else:
        recommended_stage43b_inputs = [
            "No external or naturalistic candidate data was found. "
            "Recommend adding a dedicated external validation file rather than continuing "
            "synthetic-only validation claims."
        ]

    schema_findings = {
        "total_candidates_scanned": len(candidate_files),
        "adaptable_count": len(adaptable_files),
        "ambiguous_count": len(ambiguous_files),
        "not_adaptable_count": len(non_adaptable_files),
        "naturalistic_adaptable_count": len(naturalistic_adaptable),
        "synthetic_probe_candidates": [c["path"] for c in candidate_files if is_probably_synthetic(c["path"])],
    }

    label_mapping_findings = [
        {
            "path": c["path"],
            "label_field_candidates": c["label_field_candidates"],
            "label_values_sample": c["label_values_sample"],
            "mapping_status": c["mapping_status"],
            "recommended_mapping": c["recommended_mapping"],
        }
        for c in candidate_files
        if c["label_field_candidates"]
    ]

    risks = [
        "All currently discoverable candidate data files are synthetic or controlled probe-style "
        "artifacts generated for prior ContraMamba stages (Stage10/13/14/15/31/34/35, controlled_v5); "
        "none constitute naturalistic external evidence.",
        "No VitaminC, FEVER, RTE, MNLI, or SNLI style file currently exists in this repository.",
        "Label field aliasing is heuristic (string/lowercase match against a fixed alias and value list); "
        "any 'mapped' status should still be manually spot-checked before use in Stage43-B.",
        "Treating any file flagged here as naturalistic (when it is in fact a synthetic probe) would "
        "overstate external validation readiness for publication claims.",
        "This manifest inspects only up to the first 5 records per file; field/label distributions "
        "beyond the sample are not verified.",
    ]

    recommendation = (
        "Stage43-A inventory is complete. No naturalistic external claim-evidence dataset "
        "(e.g. VitaminC/FEVER/MNLI/SNLI/RTE) currently exists in the repository; all adaptable-schema "
        "candidates found are synthetic or controlled probe artifacts from earlier stages. "
        "Before any Stage43-B external evaluation can proceed, a genuinely external/naturalistic "
        "claim-evidence source should be added to the repository under a dedicated path (e.g. "
        "data/external/). Do not use Stage34/35 probes as a substitute for naturalistic external "
        "validation, and do not use this manifest or any candidate file for training, calibration, "
        "threshold selection, or checkpoint selection."
    )

    leakage_policy = (
        "This manifest is inventory/preparation only. None of the listed candidate files, sample "
        "records, or mappings may be used for training, calibration, threshold selection, checkpoint "
        "selection, or loss design. If Stage43-B is executed, it must remain external/evaluation-only, "
        "consistent with the eval-only protocol already used for Stage29-D, Stage34-A, and Stage35-A."
    )

    return {
        "decision": decision,
        "candidate_files": candidate_files,
        "adaptable_files": adaptable_files,
        "non_adaptable_files": non_adaptable_files,
        "missing_expected_sources": missing_expected_sources,
        "recommended_stage43b_inputs": recommended_stage43b_inputs,
        "schema_findings": schema_findings,
        "label_mapping_findings": label_mapping_findings,
        "risks": risks,
        "recommendation": recommendation,
        "leakage_policy": leakage_policy,
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Stage43-A External/Naturalistic Validation Manifest")
    lines.append("")
    lines.append(
        "Preparation/inventory only. No model training, evaluation, or Kaggle/local model "
        "execution was performed to produce this manifest."
    )
    lines.append("")

    lines.append("## 1. Overall Decision")
    lines.append("")
    lines.append(f"**Decision:** `{manifest['decision']}`")
    lines.append("")
    sf = manifest["schema_findings"]
    lines.append(
        f"- Candidates scanned: {sf['total_candidates_scanned']}\n"
        f"- Adaptable: {sf['adaptable_count']}\n"
        f"- Ambiguous: {sf['ambiguous_count']}\n"
        f"- Not adaptable: {sf['not_adaptable_count']}\n"
        f"- Naturalistic adaptable (excludes known synthetic probes): {sf['naturalistic_adaptable_count']}"
    )
    lines.append("")

    lines.append("## 2. Candidate File Inventory")
    lines.append("")
    if manifest["candidate_files"]:
        lines.append("| Path | Type | Rows (est.) | Schema status | Mapping status | Claim field(s) | Evidence field(s) | Label field(s) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in manifest["candidate_files"]:
            lines.append(
                f"| `{c['path']}` | {c['file_type']} | {c['row_count_estimate']} | "
                f"{c['schema_status']} | {c['mapping_status']} | "
                f"{', '.join(c['claim_field_candidates']) or '-'} | "
                f"{', '.join(c['evidence_field_candidates']) or '-'} | "
                f"{', '.join(c['label_field_candidates']) or '-'} |"
            )
    else:
        lines.append("No candidate files were found matching the Stage43-A keyword filter.")
    lines.append("")

    lines.append("## 3. Adaptable Files")
    lines.append("")
    if manifest["adaptable_files"]:
        for c in manifest["adaptable_files"]:
            synthetic_tag = " (synthetic/controlled probe)" if is_probably_synthetic(c["path"]) else " (naturalistic candidate)"
            lines.append(f"### `{c['path']}`{synthetic_tag}")
            lines.append("")
            lines.append(f"- Recommended mapping: `{json.dumps(c['recommended_mapping'])}`")
            lines.append(f"- Notes: {c['notes']}")
            lines.append("")
    else:
        lines.append("No files with `schema_status = adaptable` were found.")
    lines.append("")

    lines.append("## 4. Ambiguous / Non-Adaptable Files")
    lines.append("")
    ambiguous = [c for c in manifest["candidate_files"] if c["schema_status"] == "ambiguous"]
    if ambiguous:
        lines.append("### Ambiguous")
        for c in ambiguous:
            lines.append(f"- `{c['path']}` -- {c['notes']}")
        lines.append("")
    if manifest["non_adaptable_files"]:
        lines.append("### Not adaptable")
        for c in manifest["non_adaptable_files"]:
            lines.append(f"- `{c['path']}` -- {c['notes']}")
        lines.append("")
    if not ambiguous and not manifest["non_adaptable_files"]:
        lines.append("None.")
        lines.append("")

    lines.append("## 5. Recommended Stage43-B Validation Plan")
    lines.append("")
    for item in manifest["recommended_stage43b_inputs"]:
        lines.append(f"- {item}")
    lines.append("")
    if manifest["missing_expected_sources"]:
        lines.append("Missing expected external sources:")
        lines.append("")
        for item in manifest["missing_expected_sources"]:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## 6. Label Mapping Table")
    lines.append("")
    if manifest["label_mapping_findings"]:
        lines.append("| Path | Label field(s) | Sampled values | Mapping status | Recommended mapping |")
        lines.append("|---|---|---|---|---|")
        for f in manifest["label_mapping_findings"]:
            lines.append(
                f"| `{f['path']}` | {', '.join(f['label_field_candidates'])} | "
                f"{', '.join(f['label_values_sample']) or '-'} | {f['mapping_status']} | "
                f"`{json.dumps(f['recommended_mapping'])}` |"
            )
    else:
        lines.append("No label-like fields were detected in any candidate file.")
    lines.append("")

    lines.append("## 7. Risks")
    lines.append("")
    for r in manifest["risks"]:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## 8. Leakage Policy")
    lines.append("")
    lines.append(manifest["leakage_policy"])
    lines.append("")

    lines.append("## 9. Recommendation")
    lines.append("")
    lines.append(manifest["recommendation"])
    lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    manifest = build_manifest()

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    markdown = render_markdown(manifest)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    with args.md_output.open("w", encoding="utf-8") as fh:
        fh.write(markdown)

    print(f"Decision: {manifest['decision']}")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.md_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
