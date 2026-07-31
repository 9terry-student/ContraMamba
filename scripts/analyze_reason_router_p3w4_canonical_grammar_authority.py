from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import random
import re
import importlib.util
import inspect
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "reason_router_p3w4_canonical_grammar_authority_audit_v3"
STATIC_DECISION = "P3W4_IMPLEMENTATION_READY_FOR_STATIC_REVIEW"
P3W3_SCHEMA_VERSION = "reason_router_p3w3_polarity_authority_audit_v3"
P3W3_STATUS = "P3W3_AUDIT_EXECUTION_COMPLETE_PENDING_RESULT_REVIEW"
P3W3_DECISION = "P3W3_MIXED_REMEDIATION_REQUIRED"
EXPECTED_F1_PAIRS = 121
EXPECTED_F2_PAIRS = 119
EXPECTED_REFUTE_ROWS = 359
EXPECTED_AFFECTED_PAIRS = 240
EXPECTED_GENERATOR_DEFECT_REFUTE = 240
EXPECTED_INTEGRITY_SOURCE_REQUIRED_REFUTE = 119
EXPECTED_ELIGIBLE_SUPPORT = 242
EXPECTED_ELIGIBLE_REFUTE = 0
EXPECTED_AFFECTED_MEMBER_ROWS = 478
EXPECTED_F1_AFFECTED_MEMBERS = 121
EXPECTED_F2_AFFECTED_MEMBERS = 357
MINIMUM_REFUTE_READY_COUNT = 50

FINAL_LABELS = {"SUPPORT", "REFUTE", "NOT_ENTITLED"}
STATUS_FIELDS = (
    "schema_status",
    "dataset_source_status",
    "grammar_status",
    "canonical_status",
    "intervention_contract_status",
    "polarity_contamination_status",
    "time_swap_status",
)
TEXT_FIELD_HINTS = (
    "claim",
    "evidence",
    "context",
    "premise",
    "hypothesis",
    "subject",
    "predicate",
    "object",
    "canonical",
    "generated",
    "template",
    "surface",
    "realization",
)
GRAMMAR_CODES = {
    "DID_NOT_INFLECTED_PREDICATE",
    "GRAMMAR_TEMPLATE_FAIL",
    "CANONICAL_ROW_KNOWN_GENERATOR_DEFECT",
}
NEGATION_MARKERS = {"not", "no", "never", "didn't", "doesn't", "isn't", "wasn't", "cannot", "can't"}
AUXILIARIES = {
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
}
REVIEW_COLUMNS = [
    "pair_id",
    "canonical_none_row_id",
    "paraphrase_row_id",
    "polarity_flip_row_id",
    "canonical_final_label",
    "paraphrase_final_label",
    "polarity_flip_final_label",
    "canonical_claim",
    "paraphrase_claim",
    "polarity_flip_claim",
    "canonical_evidence",
    "paraphrase_evidence",
    "polarity_flip_evidence",
    "canonical_grammar_status",
    "paraphrase_grammar_status",
    "polarity_flip_grammar_status",
    "canonical_reason_codes",
    "paraphrase_reason_codes",
    "polarity_flip_reason_codes",
    "canonical_claim_text_diff_summary",
    "paraphrase_claim_text_diff_summary",
    "polarity_flip_claim_text_diff_summary",
    "canonical_evidence_text_diff_summary",
    "paraphrase_evidence_text_diff_summary",
    "polarity_flip_evidence_text_diff_summary",
    "automatic_root_cause_class",
    "automatic_evidence",
    "human_canonical_semantics",
    "human_paraphrase_semantics",
    "human_polarity_flip_semantics",
    "human_grammar_validity",
    "human_authority_decision",
    "human_notes",
]

REFUTE_JSONL_REQUIRED_FIELDS = {
    "row_id",
    "pair_id",
    "intervention_type",
    "final_label",
    "canonical_row_id",
    "canonical_counterpart_row_id",
    "canonical_counterpart_final_label",
    "canonical_counterpart_eligibility",
    "ordered_exclusion_codes",
    "generator_evidence_class",
    "generator_source_sha256",
    "integrity_builder_sha256",
}
VALIDATOR_METADATA_FIELDS = (
    "validator_source_path",
    "validator_function",
    "validator_source_sha256",
    "validator_authority_source",
    "validator_authority_function",
    "validator_definition_kind",
    "validator_call_site_function",
    "validator_call_site_lineno",
    "validator_call_site_reachable_from_run",
    "validator_call_site_authorized",
    "validator_call_chain_verified",
    "validator_authorized_call_sites",
    "validator_authorized_call_site_count",
    "validator_callable_source_path",
    "validator_signature",
)

VALIDATOR_SUMMARY_FIELDS = (
    "validator_source_path",
    "validator_function",
    "validator_source_sha256",
    "validator_authority_source",
    "validator_authority_function",
    "validator_definition_kind",
    "validator_call_site_function",
    "validator_call_site_lineno",
    "validator_call_site_reachable_from_run",
    "validator_call_site_authorized",
    "validator_authorized_call_sites",
    "validator_authorized_call_site_count",
    "validator_callable_source_path",
    "validator_signature",
    "validator_call_chain_verified",
)

AUTHORIZED_STAGE185_GRAMMAR_CALL_SITES = {"build_sidecar"}
EXECUTION_ISOLATION = {
    "model_loaded": False,
    "tokenizer_loaded": False,
    "cuda_required": False,
    "forward_executed": False,
    "backward_executed": False,
    "optimizer_step_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "external_api_used": False,
    "LLM_used": False,
    "production_behavior_modified": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL line is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON file is not an object: {path}")
    return value


def semantic_sidecar_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = [{key: row[key] for key in sorted(row) if key != "created_at"} for row in rows]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_by_pair(rows: list[dict[str, Any]], seed: int, ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    require(0.0 < ratio < 1.0, "dev ratio must be strictly between 0 and 1")
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    random.Random(seed).shuffle(pair_ids)
    count = min(len(pair_ids) - 1, max(1, round(len(pair_ids) * ratio)))
    dev_ids = set(pair_ids[:count])
    train = [row for row in rows if str(row["pair_id"]) not in dev_ids]
    dev = [row for row in rows if str(row["pair_id"]) in dev_ids]
    return train, dev, dev_ids


def ordered_train_identity_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record.get('id', '')}\t{record.get('pair_id', '')}\t{record.get('final_label', '')}\n".encode("utf-8")
        )
    return digest.hexdigest()


def exact_map(rows: list[dict[str, Any]], key: str, error: str = "duplicate row ID") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        require(bool(value), f"missing {key}")
        require(value not in result, error)
        result[value] = row
    return result




def git_output(args: list[str], root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(exc.stderr.strip() or exc.stdout.strip() or "git command failed") from exc
    return completed.stdout.strip()


def verify_git_gates(root: Path, execution_commit: str) -> dict[str, Any]:
    head = git_output(["rev-parse", "HEAD"], root)
    require(head == execution_commit, "current execution commit mismatch")
    dirty = git_output(["status", "--short", "--untracked-files=no"], root)
    require(dirty == "", "dirty tracked tree")
    return {"git_head": head, "tracked_tree_clean": True}


def relative_to_root(path: Path, root: Path) -> str:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        return str(resolved.relative_to(root_resolved)).replace("\\", "/")
    except ValueError as exc:
        raise ValueError("external-repository artifact path rejected") from exc


def verify_artifact_sha(path: Path, expected_sha256: str) -> None:
    require(file_sha256(path) == expected_sha256, "P3-W3 artifact SHA mismatch")


def verify_tracked_artifact(path: Path, root: Path, expected_sha256: str) -> None:
    rel = relative_to_root(path, root)
    require(path.exists(), "P3-W3 authority artifact missing")
    verify_artifact_sha(path, expected_sha256)
    try:
        git_output(["ls-files", "--error-unmatch", rel], root)
    except ValueError as exc:
        raise ValueError("P3W3_ARTIFACT_NOT_GIT_TRACKED") from exc
    try:
        git_output(["cat-file", "-e", f"HEAD:{rel}"], root)
    except ValueError as exc:
        raise ValueError("P3W3_ARTIFACT_NOT_HEAD_ADDRESSABLE") from exc


def git_blob_identity(root: Path, commit: str, rel_path: str) -> str:
    try:
        return git_output(["rev-parse", f"{commit}:{rel_path}"], root)
    except ValueError as exc:
        raise ValueError("P3W4_SOURCE_BLOB_IDENTITY_MISMATCH") from exc


def verify_source_blob_identities(root: Path, p3w3_commit: str) -> dict[str, Any]:
    specs = {
        "generator_source_blob_identity": "scripts/build_controlled_v5.py",
        "integrity_builder_source_blob_identity": "scripts/build_stage185a_controlled_train_integrity_sidecar.py",
        "stage182_regression_oracle_blob_identity": "scripts/analyze_stage182a_controlled_intervention_integrity.py",
    }
    result: dict[str, Any] = {}
    for key, rel_path in specs.items():
        p3w3_blob = git_blob_identity(root, p3w3_commit, rel_path)
        head_blob = git_blob_identity(root, "HEAD", rel_path)
        require(p3w3_blob == head_blob, "P3W4_SOURCE_BLOB_IDENTITY_MISMATCH")
        result[key] = {"path": rel_path, "p3w3_blob": p3w3_blob, "head_blob": head_blob, "matches": True}
    return result


def verify_resolved_validator_blob_identity(root: Path, p3w3_commit: str, validator_record: dict[str, Any]) -> dict[str, Any]:
    validate_grammar_validator_record(validator_record)
    require(validator_record.get("validator_call_chain_verified") is True, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    rel_path = validator_record["validator_authority_source"]
    p3w3_blob = git_blob_identity(root, p3w3_commit, rel_path)
    head_blob = git_blob_identity(root, "HEAD", rel_path)
    require(p3w3_blob == head_blob, "P3W4_GRAMMAR_VALIDATOR_BLOB_IDENTITY_MISMATCH")
    return {"path": rel_path, "p3w3_blob": p3w3_blob, "head_blob": head_blob, "matches": True}

def load_intervention_authority(root: Path) -> dict[str, Any]:
    path = root / "scripts" / "build_controlled_v5.py"
    require(path.exists(), "missing intervention authority source")
    text = path.read_text(encoding="utf-8")
    require("INTERVENTION_TYPES" in text, "missing intervention authority symbol")
    spec = importlib.util.spec_from_file_location("p3w4_build_controlled_v5", path)
    require(spec is not None and spec.loader is not None, "cannot load intervention authority")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = getattr(module, "INTERVENTION_TYPES", None)
    try:
        canonical_values = frozenset(str(value) for value in values)
    except TypeError as exc:
        raise ValueError("missing intervention authority symbol") from exc
    require(bool(canonical_values), "missing intervention authority symbol")
    return {"source": "scripts/build_controlled_v5.py::INTERVENTION_TYPES", "values": canonical_values, "module": module}


def load_fact_authority(root: Path, all_pair_ids: set[str], affected_pair_ids: set[str]) -> dict[str, dict[str, Any]]:
    authority = load_intervention_authority(root)
    module = authority["module"]
    require(hasattr(module, "fact_templates_for_count"), "missing fact authority")
    facts = module.fact_templates_for_count(len(all_pair_ids))
    require(isinstance(facts, list), "missing fact authority")
    by_pair = exact_map(facts, "pair_id", "duplicate fact pair ID")
    require(set(by_pair) == all_pair_ids, "complete fact authority identity mismatch")
    require(affected_pair_ids.issubset(all_pair_ids), "missing affected pair ID")
    return {pair_id: by_pair[pair_id] for pair_id in sorted(affected_pair_ids)}


def verify_singleton_source_sha(root: Path, p3w3_rows: list[dict[str, Any]], field: str, rel_path: str, error: str) -> str:
    values = {str(row.get(field, "")) for row in p3w3_rows}
    require(len(values) == 1 and "" not in values, error)
    source_sha = file_sha256(root / rel_path)
    require(next(iter(values)) == source_sha, error)
    return source_sha


def verify_generator_source_authority(root: Path, p3w3_rows: list[dict[str, Any]]) -> str:
    return verify_singleton_source_sha(root, p3w3_rows, "generator_source_sha256", "scripts/build_controlled_v5.py", "P3W4_GENERATOR_SOURCE_AUTHORITY_MISMATCH")


def verify_integrity_builder_authority(root: Path, p3w3_rows: list[dict[str, Any]]) -> str:
    return verify_singleton_source_sha(root, p3w3_rows, "integrity_builder_sha256", "scripts/build_stage185a_controlled_train_integrity_sidecar.py", "P3W4_INTEGRITY_BUILDER_AUTHORITY_MISMATCH")

def validate_source_rows(rows: list[dict[str, Any]], intervention_types: set[str]) -> None:
    seen: set[str] = set()
    by_pair_intervention: set[tuple[str, str]] = set()
    for row in rows:
        row_id = str(row.get("id", ""))
        require(row_id and row_id not in seen, "duplicate row ID")
        seen.add(row_id)
        for field in ("id", "pair_id", "claim", "evidence", "final_label", "polarity_label", "primary_failure_type", "intervention_type"):
            require(field in row and isinstance(row[field], str) and row[field].strip(), f"missing textual fields needed for grammar audit: {row_id}:{field}")
        for field in ("frame_compatible_label", "predicate_covered_label", "sufficiency_label"):
            require(type(row.get(field)) is int and row[field] in (0, 1), f"unknown label binary field: {row_id}:{field}")
        require(row["final_label"] in FINAL_LABELS, f"unknown labels or intervention types: {row_id}:final_label")
        require(row["intervention_type"] in intervention_types, f"unknown labels or intervention types: {row_id}:intervention_type")
        key = (str(row["pair_id"]), str(row["intervention_type"]))
        require(key not in by_pair_intervention, "P3W4_DUPLICATE_PAIR_INTERVENTION_MEMBER")
        by_pair_intervention.add(key)


def validate_split_contract(
    rows: list[dict[str, Any]],
    split_seed: int,
    dev_ratio: float,
    expected_train_row_count: int,
    expected_train_identity_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    train, dev, dev_ids = split_by_pair(rows, split_seed, dev_ratio)
    train_pairs = {str(row["pair_id"]) for row in train}
    dev_pairs = {str(row["pair_id"]) for row in dev}
    require(not (train_pairs & dev_pairs), "train/dev pair leakage")
    require(len(train) == expected_train_row_count, "split/count/identity mismatch")
    observed = ordered_train_identity_hash(train)
    require(observed == expected_train_identity_hash, "split/count/identity mismatch")
    return train, dev, dev_ids


def validate_sidecar(rows: list[dict[str, Any]], sidecar_rows: list[dict[str, Any]], expected_data_sha: str) -> dict[str, dict[str, Any]]:
    source_by_id = exact_map(rows, "id", "duplicate source id")
    sidecar_by_id = exact_map(sidecar_rows, "row_id", "duplicate sidecar row ID")
    require(set(source_by_id) == set(sidecar_by_id), "sidecar/source identity mismatch")
    for row_id, source in source_by_id.items():
        sidecar = sidecar_by_id[row_id]
        require(sidecar.get("pair_id") == source.get("pair_id"), "sidecar/source identity mismatch")
        require(sidecar.get("source_dataset_sha256") == expected_data_sha, "sidecar/source identity mismatch")
        require(sidecar.get("source_dataset_path") == "data/controlled_v5_v3_without_time_swap.jsonl", "sidecar/source identity mismatch")
        require(isinstance(sidecar.get("reason_codes"), list), f"sidecar reason_codes missing: {row_id}")
        for field in STATUS_FIELDS:
            require(field in sidecar, f"sidecar/source identity mismatch: missing {field}")
    return sidecar_by_id


def exclusion_codes(row: dict[str, Any]) -> list[str]:
    for field in ("ordered_exclusion_codes", "p2_reason_exclusion_codes", "exclusion_codes", "exclusions", "p2_exclusion_codes"):
        value = row.get(field)
        if isinstance(value, list):
            return [str(item) for item in value]
    value = row.get("exclusion")
    if isinstance(value, str) and value:
        return [value]
    return []


def row_id_of(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("id") or "")


def canonical_row_id_of(row: dict[str, Any], sidecar: dict[str, Any] | None = None) -> str:
    if sidecar is not None:
        return str(sidecar.get("canonical_row_id", ""))
    return str(row.get("canonical_row_id", ""))


def text_fields(row: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in row.items():
        lowered = str(key).lower()
        if isinstance(value, str) and any(hint in lowered for hint in TEXT_FIELD_HINTS):
            result[str(key)] = value
    return result


def tokenise(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text)


def normalise_token(token: str) -> str:
    return token.lower()


def changed_spans(a_tokens: list[str], b_tokens: list[str]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(a=[normalise_token(t) for t in a_tokens], b=[normalise_token(t) for t in b_tokens], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            spans.append(
                {
                    "op": tag,
                    "from_tokens": a_tokens[i1:i2],
                    "to_tokens": b_tokens[j1:j2],
                    "normalized_from": [normalise_token(t) for t in a_tokens[i1:i2]],
                    "normalized_to": [normalise_token(t) for t in b_tokens[j1:j2]],
                }
            )
    return spans


def _only_punctuation(spans: list[dict[str, Any]]) -> bool:
    changed = [token for span in spans for token in span["from_tokens"] + span["to_tokens"]]
    return bool(changed) and all(re.fullmatch(r"[^\w\s]", token) for token in changed)


def _only_whitespace(a: str, b: str) -> bool:
    return a != b and re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b)


def _looks_inflection_pair(a: str, b: str) -> bool:
    x, y = a.lower(), b.lower()
    pairs = {(x, y), (y, x)}
    if x.rstrip("e") == y.rstrip("e"):
        return True
    return any(
        left == right + suffix or right == left + suffix
        for left, right in pairs
        for suffix in ("s", "ed", "d", "ing")
    )


def text_diagnostics(canonical_text: str, transformed_text: str) -> dict[str, Any]:
    a_tokens = tokenise(canonical_text)
    b_tokens = tokenise(transformed_text)
    spans = changed_spans(a_tokens, b_tokens)
    a_norm = [normalise_token(t) for t in a_tokens if re.search(r"\w", t)]
    b_norm = [normalise_token(t) for t in b_tokens if re.search(r"\w", t)]
    a_neg = [t for t in a_norm if t in NEGATION_MARKERS]
    b_neg = [t for t in b_norm if t in NEGATION_MARKERS]
    aux_a = [t for t in a_norm if t in AUXILIARIES]
    aux_b = [t for t in b_norm if t in AUXILIARIES]
    inserted = sum(len(span["to_tokens"]) for span in spans if span["op"] == "insert")
    deleted = sum(len(span["from_tokens"]) for span in spans if span["op"] == "delete")
    replacements = [span for span in spans if span["op"] == "replace"]
    inflection_changes = [
        span for span in replacements
        if len(span["normalized_from"]) == 1 and len(span["normalized_to"]) == 1
        and _looks_inflection_pair(span["normalized_from"][0], span["normalized_to"][0])
    ]
    negation_changed = a_neg != b_neg
    only_negation = bool(spans) and all(
        set(span["normalized_from"] + span["normalized_to"]).issubset(NEGATION_MARKERS | AUXILIARIES)
        for span in spans
    ) and negation_changed
    only_inflection = bool(spans) and len(inflection_changes) == len(spans)
    if canonical_text == transformed_text:
        pattern = "no text difference"
    elif _only_whitespace(canonical_text, transformed_text):
        pattern = "whitespace-only differences"
    elif _only_punctuation(spans):
        pattern = "punctuation-only differences"
    elif only_negation:
        pattern = "negation-only difference"
    elif only_inflection:
        pattern = "predicate-inflection-only difference"
    elif negation_changed and any(inflection_changes):
        pattern = "negation plus predicate difference"
    elif deleted and not inserted and not replacements:
        pattern = "token deletion"
    elif inserted and not deleted and not replacements:
        pattern = "token insertion"
    elif len(spans) == 1 and replacements:
        pattern = "lexical substitution"
    elif len(spans) > 1:
        pattern = "multiple changes"
    else:
        pattern = "unclassifiable"
    return {
        "canonical_text": canonical_text,
        "transformed_text": transformed_text,
        "token_level_changed_span": spans,
        "normalized_changed_span": [
            {"op": span["op"], "from": span["normalized_from"], "to": span["normalized_to"]} for span in spans
        ],
        "negation_markers_added": sorted(set(b_neg) - set(a_neg)),
        "negation_markers_removed": sorted(set(a_neg) - set(b_neg)),
        "auxiliary_verb_changes": {"from": aux_a, "to": aux_b},
        "predicate_inflection_changes": inflection_changes,
        "subject_verb_agreement_analysis_available": False,
        "tense_analysis_available": False,
        "duplicate_or_missing_tokens": {"inserted": inserted, "deleted": deleted},
        "punctuation_only_differences": _only_punctuation(spans),
        "whitespace_only_differences": _only_whitespace(canonical_text, transformed_text),
        "pattern": pattern,
        "summary": pattern,
    }


def resolve_grammar_rule_provenance(root: Path) -> dict[str, dict[str, Any]]:
    targets = [
        root / "scripts" / "analyze_stage182a_controlled_intervention_integrity.py",
        root / "scripts" / "build_stage185a_controlled_train_integrity_sidecar.py",
    ]
    provenance: dict[str, dict[str, Any]] = {}
    for path in targets:
        require(path.exists(), f"missing provenance source: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        for code in GRAMMAR_CODES:
            for index, line in enumerate(lines):
                if code not in line:
                    continue
                function = "<module>"
                for cursor in range(index, -1, -1):
                    match = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", lines[cursor])
                    if match:
                        function = match.group(1)
                        break
                provenance.setdefault(
                    code,
                    {
                        "source_file": str(path.relative_to(root)).replace("\\", "/"),
                        "function": function,
                        "condition": "",
                        "input_fields": [],
                        "expected_transformation": "",
                        "failure_output": code,
                        "rule_kind": "heuristic",
                    },
                )
    require(GRAMMAR_CODES.issubset(provenance), "missing provenance source")
    provenance["DID_NOT_INFLECTED_PREDICATE"].update(
        {
            "condition": "evidence text matches did not followed by fact predicate or alternate_predicate",
            "input_fields": ["row.evidence", "fact.predicate", "fact.alternate_predicate"],
            "expected_transformation": "negative polarity must use did not plus an uninflected predicate form",
            "rule_kind": "syntactic/lexical heuristic",
        }
    )
    provenance["GRAMMAR_TEMPLATE_FAIL"].update(
        {
            "condition": "grammar_anomaly(row, fact) is true in the Stage185 sidecar builder",
            "input_fields": ["row.evidence", "fact.predicate", "fact.alternate_predicate"],
            "expected_transformation": "row must satisfy deterministic template grammar contract",
            "rule_kind": "syntactic template contract",
        }
    )
    provenance["CANONICAL_ROW_KNOWN_GENERATOR_DEFECT"].update(
        {
            "condition": "canonical_defect[pair_id] is true and derivative row_id is not the canonical anchor",
            "input_fields": ["row.id", "anchor.id", "canonical_defect[pair_id]"],
            "expected_transformation": "derivative integrity remains unresolved when canonical source is a known generator defect",
            "rule_kind": "provenance/integrity heuristic",
        }
    )
    return provenance




def load_module_from_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, "missing provenance source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_validator_signature(validator: Any) -> inspect.Signature:
    signature = inspect.signature(validator)
    try:
        signature.bind(object(), object())
    except TypeError as exc:
        raise ValueError("wrong callable signature") from exc
    return signature


def validate_grammar_validator_record(record: dict[str, Any]) -> None:
    require(isinstance(record, dict), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    for field in VALIDATOR_METADATA_FIELDS:
        require(field in record, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    if not record.get("validator_call_chain_verified"):
        expected = empty_validator_record()
        for key, value in expected.items():
            require(record.get(key) == value, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        return
    require(callable(record.get("validator")), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(isinstance(record.get("validator_source_path"), str) and bool(record["validator_source_path"]), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(record.get("validator_function") == "grammar_anomaly", "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(isinstance(record.get("validator_source_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", record["validator_source_sha256"]), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(isinstance(record.get("validator_authority_source"), str) and bool(record["validator_authority_source"]), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(record.get("validator_authority_function") == "grammar_anomaly", "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(record.get("validator_definition_kind") in {"stage185_local", "stage182_imported"}, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(isinstance(record.get("validator_call_site_function"), str) and bool(record["validator_call_site_function"]), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(isinstance(record.get("validator_call_site_lineno"), int) and record["validator_call_site_lineno"] > 0, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(isinstance(record.get("validator_call_site_reachable_from_run"), bool), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(record.get("validator_call_site_authorized") is True, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    inventory = record.get("validator_authorized_call_sites")
    require(isinstance(inventory, list), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(record.get("validator_authorized_call_site_count") == len(inventory) and record["validator_authorized_call_site_count"] > 0, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    identities: list[tuple[str, str, int, int]] = []
    for entry in inventory:
        require(isinstance(entry, dict), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        for field in ("function_name", "scope_path", "lineno", "col_offset", "reachable_from_run_or_main", "authorized_sidecar_construction_function", "nested_scope", "class_scope", "module_level", "definition_time_expression", "scope_kind", "context_path", "authorized"):
            require(field in entry, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(entry.get("function_name") == "build_sidecar", "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(entry.get("scope_path") == ["function:build_sidecar"], "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(isinstance(entry.get("lineno"), int) and entry["lineno"] > 0, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(isinstance(entry.get("col_offset"), int) and entry["col_offset"] >= 0, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(isinstance(entry.get("reachable_from_run_or_main"), bool), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(entry.get("authorized_sidecar_construction_function") is True, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(entry.get("nested_scope") is False, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(entry.get("class_scope") is False, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(entry.get("module_level") is False, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(entry.get("definition_time_expression") is False, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(entry.get("scope_kind") == "sync_function_body", "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(isinstance(entry.get("context_path"), list) and all(isinstance(item, str) for item in entry["context_path"]), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        require(entry.get("authorized") is True, "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
        identities.append(("/".join(entry["scope_path"]), "/".join(entry["context_path"]), entry["lineno"], entry["col_offset"]))
    require(len(set(identities)) == len(identities), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(identities == sorted(identities), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    first = inventory[0]
    require(record.get("validator_call_site_function") == first["function_name"], "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(record.get("validator_call_site_lineno") == first["lineno"], "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(record.get("validator_call_site_reachable_from_run") == first["reachable_from_run_or_main"], "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(record.get("validator_callable_source_path") == record.get("validator_authority_source"), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    require(isinstance(record.get("validator_signature"), str) and bool(record["validator_signature"]), "P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED")
    validate_validator_signature(record["validator"])


def callable_source_relative_path(validator: Any, root: Path) -> str | None:
    try:
        source = inspect.getsourcefile(validator)
    except TypeError:
        return None
    if source is None:
        return None
    try:
        return str(Path(source).resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return None


def walk_function_body_without_nested(function: ast.FunctionDef) -> list[ast.AST]:
    result: list[ast.AST] = []
    stack = list(function.body)
    while stack:
        node = stack.pop()
        result.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return result


def function_calls_name(function: ast.FunctionDef, name: str) -> bool:
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name for node in walk_function_body_without_nested(function))


def stage185_call_graph(functions: dict[str, ast.FunctionDef]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {name: set() for name in functions}
    for name, function in functions.items():
        for node in walk_function_body_without_nested(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
                graph[name].add(node.func.id)
    return graph


def reachable_from_run_or_main(functions: dict[str, ast.FunctionDef], target: str) -> bool:
    graph = stage185_call_graph(functions)
    stack = [name for name in ("run", "main") if name in graph]
    seen: set[str] = set()
    while stack:
        name = stack.pop()
        if name == target:
            return True
        if name in seen:
            continue
        seen.add(name)
        stack.extend(sorted(graph.get(name, set()) - seen))
    return False


def collect_grammar_anomaly_call_sites(stage185_text: str) -> dict[str, Any]:
    tree = ast.parse(stage185_text)
    top_functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    local_defined = "grammar_anomaly" in top_functions
    imported_stage182 = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "analyze_stage182a_controlled_intervention_integrity"
        and any(alias.name == "grammar_anomaly" for alias in node.names)
        for node in tree.body
    )
    call_sites: list[dict[str, Any]] = []

    class GrammarCallCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope_path: list[str] = []
            self.context_path: list[str] = []
            self.function_stack: list[str] = []
            self.class_depth = 0
            self.scope_kind = "module"
            self.definition_time_expression = False

        def with_context(self, scope_kind: str, definition_time_expression: bool, context_label: str | None, visit_nodes: list[ast.AST]) -> None:
            old_scope_kind = self.scope_kind
            old_definition_time_expression = self.definition_time_expression
            if context_label is not None:
                self.context_path.append(context_label)
            self.scope_kind = scope_kind
            self.definition_time_expression = definition_time_expression
            for item in visit_nodes:
                self.visit(item)
            self.scope_kind = old_scope_kind
            self.definition_time_expression = old_definition_time_expression
            if context_label is not None:
                self.context_path.pop()

        def annotation_nodes(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> list[ast.AST]:
            result: list[ast.AST] = []
            args = node.args
            for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
                if arg.annotation is not None:
                    result.append(arg.annotation)
            if args.vararg is not None and args.vararg.annotation is not None:
                result.append(args.vararg.annotation)
            if args.kwarg is not None and args.kwarg.annotation is not None:
                result.append(args.kwarg.annotation)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
                result.append(node.returns)
            return result

        def function_definition_nodes(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> list[ast.AST]:
            result: list[ast.AST] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.extend(node.decorator_list)
                result.extend(getattr(node, "type_params", []))
            result.extend(default for default in node.args.defaults if default is not None)
            result.extend(default for default in node.args.kw_defaults if default is not None)
            result.extend(self.annotation_nodes(node))
            return result

        def record_call(self, node: ast.Call) -> None:
            function_name = self.function_stack[-1] if self.function_stack else None
            scope_path = list(self.scope_path)
            context_path = list(self.context_path)
            module_level = self.scope_kind == "module"
            nested_scope = len(self.function_stack) > 1 or any(part == "lambda" for part in scope_path)
            class_scope = self.class_depth > 0
            authorized = bool(
                function_name == "build_sidecar"
                and scope_path == ["function:build_sidecar"]
                and self.scope_kind == "sync_function_body"
                and self.definition_time_expression is False
                and not nested_scope
                and not class_scope
                and not module_level
            )
            call_sites.append({
                "function_name": function_name,
                "scope_path": scope_path,
                "lineno": getattr(node, "lineno", None),
                "col_offset": getattr(node, "col_offset", None),
                "reachable_from_run_or_main": bool(function_name and reachable_from_run_or_main(top_functions, function_name)),
                "authorized_sidecar_construction_function": function_name in AUTHORIZED_STAGE185_GRAMMAR_CALL_SITES,
                "nested_scope": nested_scope,
                "class_scope": class_scope,
                "module_level": module_level,
                "definition_time_expression": self.definition_time_expression,
                "scope_kind": self.scope_kind,
                "context_path": context_path,
                "authorized": authorized,
            })

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "grammar_anomaly":
                self.record_call(node)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.with_context("function_definition_expression", True, f"function_def:{node.name}", self.function_definition_nodes(node))
            self.scope_path.append(f"function:{node.name}")
            self.context_path.append(f"function_body:{node.name}")
            self.function_stack.append(node.name)
            old_scope_kind = self.scope_kind
            old_definition_time_expression = self.definition_time_expression
            self.scope_kind = "sync_function_body"
            self.definition_time_expression = False
            for child in node.body:
                self.visit(child)
            self.scope_kind = old_scope_kind
            self.definition_time_expression = old_definition_time_expression
            self.function_stack.pop()
            self.context_path.pop()
            self.scope_path.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.with_context("function_definition_expression", True, f"function_def:{node.name}", self.function_definition_nodes(node))
            self.scope_path.append(f"function:{node.name}")
            self.context_path.append(f"async_function_body:{node.name}")
            self.function_stack.append(node.name)
            old_scope_kind = self.scope_kind
            old_definition_time_expression = self.definition_time_expression
            self.scope_kind = "async_function_body"
            self.definition_time_expression = False
            for child in node.body:
                self.visit(child)
            self.scope_kind = old_scope_kind
            self.definition_time_expression = old_definition_time_expression
            self.function_stack.pop()
            self.context_path.pop()
            self.scope_path.pop()

        def visit_Lambda(self, node: ast.Lambda) -> None:
            self.with_context("lambda_definition_expression", True, "lambda_def", self.function_definition_nodes(node))
            self.scope_path.append("lambda")
            self.context_path.append("lambda_body")
            old_scope_kind = self.scope_kind
            old_definition_time_expression = self.definition_time_expression
            self.scope_kind = "lambda_body"
            self.definition_time_expression = False
            self.visit(node.body)
            self.scope_kind = old_scope_kind
            self.definition_time_expression = old_definition_time_expression
            self.context_path.pop()
            self.scope_path.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            definition_nodes = list(node.decorator_list) + list(node.bases) + [keyword.value for keyword in node.keywords]
            definition_nodes.extend(getattr(node, "type_params", []))
            self.with_context("class_definition_expression", True, f"class_def:{node.name}", definition_nodes)
            self.scope_path.append(f"class:{node.name}")
            self.context_path.append(f"class_body:{node.name}")
            self.class_depth += 1
            old_scope_kind = self.scope_kind
            old_definition_time_expression = self.definition_time_expression
            self.scope_kind = "class_body"
            self.definition_time_expression = False
            for child in node.body:
                self.visit(child)
            self.scope_kind = old_scope_kind
            self.definition_time_expression = old_definition_time_expression
            self.class_depth -= 1
            self.context_path.pop()
            self.scope_path.pop()

    GrammarCallCollector().visit(tree)
    call_sites.sort(key=lambda site: ("/".join(site["scope_path"]), "/".join(site["context_path"]), site["lineno"] or -1, site["col_offset"] or -1))
    return {"local_defined": local_defined, "imported_stage182": imported_stage182, "call_sites": call_sites}

def stage185_validator_ast_authority(stage185_text: str) -> dict[str, Any]:
    inventory = collect_grammar_anomaly_call_sites(stage185_text)
    local_defined = inventory["local_defined"]
    imported_stage182 = inventory["imported_stage182"]
    if local_defined and imported_stage182:
        raise ValueError("P3W4_GRAMMAR_VALIDATOR_SOURCE_AMBIGUOUS")
    call_sites = inventory["call_sites"]
    if any(not site["authorized"] for site in call_sites):
        raise ValueError("P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED")
    if not call_sites:
        return {
            "local_defined": local_defined,
            "imported_stage182": imported_stage182,
            "call_site_function": None,
            "call_site_lineno": None,
            "call_site_reachable_from_run": False,
            "call_site_authorized": False,
            "authorized_call_sites": [],
            "authorized_call_site_count": 0,
        }
    first = call_sites[0]
    return {
        "local_defined": local_defined,
        "imported_stage182": imported_stage182,
        "call_site_function": first["function_name"],
        "call_site_lineno": first["lineno"],
        "call_site_reachable_from_run": first["reachable_from_run_or_main"],
        "call_site_authorized": True,
        "authorized_call_sites": call_sites,
        "authorized_call_site_count": len(call_sites),
    }

def empty_validator_record() -> dict[str, Any]:
    return {
        "validator": None,
        "validator_source_path": None,
        "validator_function": None,
        "validator_source_sha256": None,
        "validator_authority_source": None,
        "validator_authority_function": None,
        "validator_definition_kind": None,
        "validator_call_site_function": None,
        "validator_call_site_lineno": None,
        "validator_call_site_reachable_from_run": False,
        "validator_call_site_authorized": False,
        "validator_call_chain_verified": False,
        "validator_authorized_call_sites": [],
        "validator_authorized_call_site_count": 0,
        "validator_callable_source_path": None,
        "validator_signature": None,
    }


def load_production_grammar_validator(root: Path) -> dict[str, Any]:
    stage185_path = root / "scripts" / "build_stage185a_controlled_train_integrity_sidecar.py"
    stage182_path = root / "scripts" / "analyze_stage182a_controlled_intervention_integrity.py"
    require(stage185_path.exists() and stage182_path.exists(), "missing provenance source")
    stage185_text = stage185_path.read_text(encoding="utf-8")
    ast_authority = stage185_validator_ast_authority(stage185_text)
    if ast_authority["call_site_function"] is None:
        return empty_validator_record()
    stage185_module = load_module_from_path(stage185_path, "p3w4_stage185_builder")
    authority_source = None
    definition_kind = None
    validator = None
    if ast_authority["local_defined"]:
        validator = getattr(stage185_module, "grammar_anomaly", None)
        authority_source = "scripts/build_stage185a_controlled_train_integrity_sidecar.py"
        definition_kind = "stage185_local"
    elif ast_authority["imported_stage182"]:
        stage182_module = load_module_from_path(stage182_path, "p3w4_stage182_grammar")
        validator = getattr(stage182_module, "grammar_anomaly", None)
        authority_source = "scripts/analyze_stage182a_controlled_intervention_integrity.py"
        definition_kind = "stage182_imported"
    else:
        return empty_validator_record()
    if not callable(validator):
        return empty_validator_record()
    callable_source = callable_source_relative_path(validator, root)
    require(callable_source == authority_source, "missing provenance source")
    signature = validate_validator_signature(validator)
    source_path = root / authority_source
    return {
        "validator": validator,
        "validator_source_path": authority_source,
        "validator_function": "grammar_anomaly",
        "validator_source_sha256": file_sha256(source_path),
        "validator_authority_source": authority_source,
        "validator_authority_function": "grammar_anomaly",
        "validator_definition_kind": definition_kind,
        "validator_call_site_function": ast_authority["call_site_function"],
        "validator_call_site_lineno": ast_authority["call_site_lineno"],
        "validator_call_site_reachable_from_run": ast_authority["call_site_reachable_from_run"],
        "validator_call_site_authorized": ast_authority["call_site_authorized"],
        "validator_call_chain_verified": True,
        "validator_authorized_call_sites": ast_authority["authorized_call_sites"],
        "validator_authorized_call_site_count": ast_authority["authorized_call_site_count"],
        "validator_callable_source_path": callable_source,
        "validator_signature": str(signature),
    }

def reproduce_grammar_rule(member: dict[str, Any], fact: dict[str, Any] | None, grammar_validator: dict[str, Any] | None, generator_source_sha_matches: bool) -> dict[str, Any]:
    sidecar_codes = set(member.get("sidecar", {}).get("reason_codes", []))
    claimed = {"DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"}.issubset(sidecar_codes)
    base = {
        "validator_source_path": None,
        "validator_function": None,
        "validator_source_sha256": None,
        "validator_authority_source": None,
        "validator_authority_function": None,
        "validator_definition_kind": None,
        "validator_call_site_function": None,
        "validator_call_site_lineno": None,
        "validator_call_site_reachable_from_run": False,
        "validator_call_site_authorized": False,
        "validator_call_chain_verified": False,
        "validator_authorized_call_sites": [],
        "validator_authorized_call_site_count": 0,
        "validator_callable_source_path": None,
        "validator_signature": None,
        "sidecar_rule_claimed_failure": claimed,
        "production_rule_reproduction_result": False,
        "fact_pair_id": None if fact is None else fact.get("pair_id"),
        "fact_predicate": None if fact is None else fact.get("predicate"),
        "fact_alternate_predicate": None if fact is None else fact.get("alternate_predicate"),
        "matched_surface_span": None,
        "generator_source_sha_matches": generator_source_sha_matches,
        "exact_row_evidence_supplied": bool(member.get("source_row", {}).get("evidence")),
    }
    if fact is None or grammar_validator is None or not grammar_validator.get("validator_call_chain_verified"):
        return base
    validate_grammar_validator_record(grammar_validator)
    row = member["source_row"]
    try:
        result = bool(grammar_validator["validator"](row, fact))
    except TypeError as exc:
        raise ValueError("P3W4_GRAMMAR_VALIDATOR_INVOCATION_FAILED") from exc
    base.update({
        "validator_source_path": grammar_validator["validator_source_path"],
        "validator_function": grammar_validator["validator_function"],
        "validator_source_sha256": grammar_validator["validator_source_sha256"],
        "validator_authority_source": grammar_validator["validator_authority_source"],
        "validator_authority_function": grammar_validator["validator_authority_function"],
        "validator_definition_kind": grammar_validator["validator_definition_kind"],
        "validator_call_site_function": grammar_validator["validator_call_site_function"],
        "validator_call_site_lineno": grammar_validator["validator_call_site_lineno"],
        "validator_call_site_reachable_from_run": grammar_validator["validator_call_site_reachable_from_run"],
        "validator_call_site_authorized": grammar_validator["validator_call_site_authorized"],
        "validator_call_chain_verified": grammar_validator["validator_call_chain_verified"],
        "validator_authorized_call_sites": grammar_validator["validator_authorized_call_sites"],
        "validator_authorized_call_site_count": grammar_validator["validator_authorized_call_site_count"],
        "validator_callable_source_path": grammar_validator["validator_callable_source_path"],
        "validator_signature": grammar_validator["validator_signature"],
        "production_rule_reproduction_result": result,
    })
    evidence = str(row.get("evidence", ""))
    for key in ("predicate", "alternate_predicate"):
        predicate = str(fact.get(key, ""))
        if not predicate:
            continue
        match = re.search(rf"\bdid\s+not\s+({re.escape(predicate)})\b", evidence, re.IGNORECASE)
        if match:
            base["matched_surface_span"] = match.group(0)
            break
    return base

def intervention_of(row: dict[str, Any]) -> str:
    value = row.get("intervention_type")
    require(isinstance(value, str) and bool(value), "P3W4_MISSING_INTERVENTION_TYPE")
    return value


def final_label_of(row: dict[str, Any]) -> str:
    value = row.get("final_label")
    require(isinstance(value, str) and value in FINAL_LABELS, "P3W4_INVALID_FINAL_LABEL")
    return value


def canonical_counterpart_of(row: dict[str, Any]) -> str:
    value = row.get("canonical_counterpart_row_id")
    require(isinstance(value, str) and bool(value), "P3W4_MISSING_CANONICAL_COUNTERPART")
    return value


def require_reason_codes(sidecar: dict[str, Any], required: set[str], error: str) -> None:
    require(required.issubset(set(sidecar.get("reason_codes", []))), error)


def require_f1_sidecar_contract(sidecar: dict[str, Any]) -> None:
    require(sidecar.get("grammar_status") == "FAIL", "P3W4_F1_SIDECAR_CONTRACT_MISMATCH")
    require(sidecar.get("canonical_status") == "PASS", "P3W4_F1_SIDECAR_CONTRACT_MISMATCH")
    require_reason_codes(sidecar, {"DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"}, "P3W4_F1_SIDECAR_CONTRACT_MISMATCH")


def require_f2_canonical_sidecar_contract(sidecar: dict[str, Any]) -> None:
    require(sidecar.get("grammar_status") == "FAIL", "P3W4_F2_CANONICAL_SIDECAR_CONTRACT_MISMATCH")
    require(sidecar.get("canonical_status") == "PASS", "P3W4_F2_CANONICAL_SIDECAR_CONTRACT_MISMATCH")
    require_reason_codes(sidecar, {"DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"}, "P3W4_F2_CANONICAL_SIDECAR_CONTRACT_MISMATCH")


def require_f2_paraphrase_sidecar_contract(sidecar: dict[str, Any]) -> None:
    require(sidecar.get("canonical_status") == "UNRESOLVED", "P3W4_F2_PARAPHRASE_SIDECAR_CONTRACT_MISMATCH")
    require_reason_codes(sidecar, {"CANONICAL_ROW_KNOWN_GENERATOR_DEFECT"}, "P3W4_F2_PARAPHRASE_SIDECAR_CONTRACT_MISMATCH")

def p3w3_row_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row_id_of(row)
        require(row_id, "P3-W3 row count mismatch")
        require(row_id not in result, "duplicate REFUTE row ID")
        result[row_id] = row
    return result


def validate_refute_jsonl_row_fields(refute_rows: list[dict[str, Any]]) -> None:
    for row in refute_rows:
        row_id = row_id_of(row)
        missing = sorted(field for field in REFUTE_JSONL_REQUIRED_FIELDS if field not in row or row[field] in (None, ""))
        require(not missing, f"missing REFUTE JSONL authority field: {row_id}:{missing}")
        require(isinstance(row.get("ordered_exclusion_codes"), list) and row["ordered_exclusion_codes"], f"missing REFUTE JSONL authority field: {row_id}:ordered_exclusion_codes")
        require(type(row.get("canonical_counterpart_eligibility")) is bool, f"missing REFUTE JSONL authority field: {row_id}:canonical_counterpart_eligibility")

def validate_refute_jsonl_partition_contract(refute_rows: list[dict[str, Any]]) -> Counter:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_counts = Counter(str(row["generator_evidence_class"]) for row in refute_rows)
    for row in refute_rows:
        by_pair[str(row["pair_id"])].append(row)
        require(row["canonical_row_id"] == row["canonical_counterpart_row_id"], "P3-W3 exact REFUTE partition mismatch")
    consumed: set[str] = set()
    for pair_id, rows in by_pair.items():
        if len(rows) == 1:
            row = rows[0]
            require(row["intervention_type"] == "polarity_flip", "P3-W3 exact REFUTE partition mismatch")
            require(row["final_label"] == "REFUTE", "P3-W3 exact REFUTE partition mismatch")
            require(row["ordered_exclusion_codes"] == ["P2_GENERATOR_STATUS_DEFECT"], "P3-W3 exact REFUTE partition mismatch")
            require(row["generator_evidence_class"] == "INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE", "P3-W3 exact REFUTE partition mismatch")
            require(row["canonical_counterpart_final_label"] == "SUPPORT", "P3-W3 exact REFUTE partition mismatch")
            require(row["canonical_counterpart_eligibility"] is True, "P3-W3 exact REFUTE partition mismatch")
        elif len(rows) == 2:
            by_intervention = exact_map(rows, "intervention_type", "P3-W3 exact REFUTE partition mismatch")
            require(set(by_intervention) == {"none", "paraphrase"}, "P3-W3 exact REFUTE partition mismatch")
            none = by_intervention["none"]
            paraphrase = by_intervention["paraphrase"]
            require(none["final_label"] == "REFUTE", "P3-W3 exact REFUTE partition mismatch")
            require(none["ordered_exclusion_codes"] == ["P2_GENERATOR_STATUS_DEFECT"], "P3-W3 exact REFUTE partition mismatch")
            require(none["generator_evidence_class"] == "INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE", "P3-W3 exact REFUTE partition mismatch")
            require(none["canonical_counterpart_final_label"] == "REFUTE", "P3-W3 exact REFUTE partition mismatch")
            require(none["canonical_counterpart_eligibility"] is False, "P3-W3 exact REFUTE partition mismatch")
            require(paraphrase["final_label"] == "REFUTE", "P3-W3 exact REFUTE partition mismatch")
            require(paraphrase["ordered_exclusion_codes"] == ["P2_INTEGRITY_SOURCE_REQUIRED"], "P3-W3 exact REFUTE partition mismatch")
            require(paraphrase["generator_evidence_class"] == "AMBIGUOUS_INTEGRITY_EVIDENCE", "P3-W3 exact REFUTE partition mismatch")
            require(paraphrase["canonical_counterpart_final_label"] == "REFUTE", "P3-W3 exact REFUTE partition mismatch")
            require(paraphrase["canonical_counterpart_eligibility"] is False, "P3-W3 exact REFUTE partition mismatch")
        else:
            raise ValueError("P3-W3 exact REFUTE partition mismatch")
        consumed.update(row_id_of(row) for row in rows)
    require(len(consumed) == len(refute_rows), "unconsumed REFUTE row")
    return evidence_counts

def validate_p3w3_artifacts(summary: dict[str, Any], refute_rows: list[dict[str, Any]], expected_execution_commit: str) -> dict[str, Any]:
    require(summary.get("schema_version") == P3W3_SCHEMA_VERSION, "wrong schema/status/decision")
    require(summary.get("status") == P3W3_STATUS, "wrong schema/status/decision")
    require(summary.get("decision") == P3W3_DECISION, "wrong schema/status/decision")
    require("execution_commit" in summary, "missing P3-W3 summary execution_commit")
    require(str(summary["execution_commit"]) == expected_execution_commit, "P3-W3 summary commit mismatch")
    require(summary.get("refute_row_count_exported") == EXPECTED_REFUTE_ROWS, "wrong evidence counts")
    plc = summary.get("pair_level_canonical_comparison", {})
    require(plc.get("refute_row_count") == EXPECTED_REFUTE_ROWS, "wrong evidence counts")
    require(plc.get("unique_refute_pair_count") == EXPECTED_AFFECTED_PAIRS, "wrong evidence counts")
    require(plc.get("multi_refute_row_pair_count") == EXPECTED_F2_PAIRS, "wrong evidence counts")
    flo = summary.get("final_label_overview", {})
    require(flo.get("eligible_REFUTE_polarity_targets") == EXPECTED_ELIGIBLE_REFUTE, "wrong evidence counts")
    require(flo.get("eligible_SUPPORT_polarity_targets") == EXPECTED_ELIGIBLE_SUPPORT, "wrong evidence counts")
    evidence = summary.get("generator_evidence_class_counts", {})
    require(evidence.get("AMBIGUOUS_INTEGRITY_EVIDENCE") == EXPECTED_INTEGRITY_SOURCE_REQUIRED_REFUTE, "wrong evidence counts")
    require(evidence.get("INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE") == EXPECTED_GENERATOR_DEFECT_REFUTE, "wrong evidence counts")
    interp = summary.get("sidecar_semantic_interpretation_audit", {})
    require(interp.get("generator_evidence_proof_contract_available") is False, "wrong evidence counts")
    c5 = summary.get("counterfactual_eligibility_results", {}).get("C5", {})
    require(c5.get("newly_admitted_row_count") == 0, "wrong evidence counts")
    u8 = summary.get("candidate_universe_counts", {}).get("U8_final_polarity_applicable_rows", {})
    require(u8.get("REFUTE") == EXPECTED_ELIGIBLE_REFUTE, "wrong evidence counts")
    require(u8.get("SUPPORT") == EXPECTED_ELIGIBLE_SUPPORT, "wrong evidence counts")
    u8_row_ids = u8.get("row_ids")
    require(isinstance(u8_row_ids, list), "wrong evidence counts")
    require(len(u8_row_ids) == EXPECTED_ELIGIBLE_SUPPORT, "wrong evidence counts")
    require(len(set(u8_row_ids)) == EXPECTED_ELIGIBLE_SUPPORT, "wrong evidence counts")
    require(summary.get("A1_A3_released") is False, "wrong evidence counts")
    require(summary.get("polarity_supervision_released") is False, "wrong evidence counts")
    require(len(refute_rows) == EXPECTED_REFUTE_ROWS, "P3-W3 row count mismatch")
    validate_refute_jsonl_row_fields(refute_rows)
    p3w3_row_index(refute_rows)
    evidence_counts_from_rows = validate_refute_jsonl_partition_contract(refute_rows)
    require(set(evidence_counts_from_rows) == {"INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE", "AMBIGUOUS_INTEGRITY_EVIDENCE"}, "wrong evidence counts")
    require(evidence_counts_from_rows["INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE"] == evidence.get("INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE"), "wrong evidence counts")
    require(evidence_counts_from_rows["AMBIGUOUS_INTEGRITY_EVIDENCE"] == evidence.get("AMBIGUOUS_INTEGRITY_EVIDENCE"), "wrong evidence counts")
    refute_row_ids = {row_id_of(row) for row in refute_rows}
    require(not (refute_row_ids & set(u8_row_ids)), "wrong evidence counts")
    counts = Counter(str(row.get("pair_id", "")) for row in refute_rows)
    require(Counter(counts.values()) == Counter({1: EXPECTED_F1_PAIRS, 2: EXPECTED_F2_PAIRS}), "P3-W3 exact REFUTE partition mismatch")
    return {"u8_support_row_ids": set(u8["row_ids"])}

def build_pair_member_map(train_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in train_rows:
        pair_id = str(row["pair_id"])
        intervention = str(row["intervention_type"])
        require(intervention not in by_pair[pair_id], "P3W4_DUPLICATE_PAIR_INTERVENTION_MEMBER")
        by_pair[pair_id][intervention] = row
    return by_pair


def require_lineage(row: dict[str, Any], sidecar: dict[str, Any], canonical_id: str, p3row: dict[str, Any] | None = None) -> None:
    require(sidecar.get("canonical_row_id") == canonical_id, "P3W4_CANONICAL_LINEAGE_INCONSISTENCY")
    require(sidecar.get("pair_id") == row.get("pair_id"), "pair_id consistency")
    if p3row is not None:
        counterpart = canonical_counterpart_of(p3row)
        require(counterpart == canonical_id, "P3-W3 canonical_counterpart_row_id inconsistency")
        require(str(p3row.get("pair_id", "")) == str(row.get("pair_id", "")), "pair_id consistency")


def reconstruct_families(
    train_rows: list[dict[str, Any]],
    sidecar_by_id: dict[str, dict[str, Any]],
    p3w3_rows: list[dict[str, Any]],
    p3w3_authority: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_row = exact_map(train_rows, "id", "duplicate source id")
    by_pair = build_pair_member_map(train_rows)
    p3_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in p3w3_rows:
        p3_by_pair[str(row.get("pair_id", ""))].append(row)
    u8_support = set((p3w3_authority or {}).get("u8_support_row_ids", []))
    require(u8_support, "F1 canonical U8 eligibility authority missing")
    f1: dict[str, dict[str, Any]] = {}
    f2: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()
    for pair_id, exported in p3_by_pair.items():
        require(len(exported) in (1, 2), "P3-W3 exact REFUTE partition mismatch")
        members = by_pair.get(pair_id, {})
        none = members.get("none")
        polarity = members.get("polarity_flip")
        require(none is not None, "P3W4_MISSING_PAIR_MEMBER")
        require(polarity is not None, "P3W4_MISSING_PAIR_MEMBER")
        canonical_id = str(none["id"])
        require_lineage(none, sidecar_by_id[str(none["id"])], canonical_id)
        require_lineage(polarity, sidecar_by_id[str(polarity["id"])], canonical_id)
        paraphrase = members.get("paraphrase")
        if paraphrase is not None:
            require_lineage(paraphrase, sidecar_by_id[str(paraphrase["id"])], canonical_id)
        if len(exported) == 1:
            p3 = exported[0]
            row = by_row.get(row_id_of(p3))
            require(row is not None and row is polarity, "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(intervention_of(p3) == "polarity_flip", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(exclusion_codes(p3) == ["P2_GENERATOR_STATUS_DEFECT"], "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(final_label_of(p3) == "REFUTE", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(p3.get("generator_evidence_class") == "INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(p3.get("canonical_counterpart_final_label") == "SUPPORT", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(p3.get("canonical_counterpart_eligibility") is True, "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(none["final_label"] == "SUPPORT", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(canonical_id in u8_support, "F1 canonical U8 eligibility mismatch")
            require_f1_sidecar_contract(sidecar_by_id[str(polarity["id"])])
            require_lineage(polarity, sidecar_by_id[str(polarity["id"])], canonical_id, p3)
            consumed.add(row_id_of(p3))
            family = {"canonical": none, "polarity_flip": polarity, "p3w3_rows": exported}
            if paraphrase is not None:
                family["paraphrase"] = paraphrase
            f1[pair_id] = family
        else:
            require(paraphrase is not None, "P3W4_MISSING_PAIR_MEMBER")
            none_p3 = next((row for row in exported if row_id_of(row) == canonical_id), None)
            para_p3 = next((row for row in exported if row_id_of(row) == paraphrase["id"]), None)
            require(none_p3 is not None and para_p3 is not None, "P3-W3 exact REFUTE partition mismatch")
            require(exclusion_codes(none_p3) == ["P2_GENERATOR_STATUS_DEFECT"], "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(exclusion_codes(para_p3) == ["P2_INTEGRITY_SOURCE_REQUIRED"], "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(intervention_of(none_p3) == "none", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(final_label_of(none_p3) == "REFUTE" and none["final_label"] == "REFUTE", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(none_p3.get("generator_evidence_class") == "INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(none_p3.get("canonical_counterpart_final_label") == "REFUTE", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(none_p3.get("canonical_counterpart_eligibility") is False, "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(intervention_of(para_p3) == "paraphrase", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(final_label_of(para_p3) == "REFUTE" and paraphrase["final_label"] == "REFUTE", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(para_p3.get("generator_evidence_class") == "AMBIGUOUS_INTEGRITY_EVIDENCE", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(para_p3.get("canonical_counterpart_final_label") == "REFUTE", "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require(para_p3.get("canonical_counterpart_eligibility") is False, "P3W4_UNEXPECTED_INTERVENTION_COMPOSITION")
            require_f2_canonical_sidecar_contract(sidecar_by_id[canonical_id])
            require_f2_paraphrase_sidecar_contract(sidecar_by_id[str(paraphrase["id"])])
            require_lineage(none, sidecar_by_id[canonical_id], canonical_id, none_p3)
            require_lineage(paraphrase, sidecar_by_id[str(paraphrase["id"])], canonical_id, para_p3)
            consumed.update({row_id_of(none_p3), row_id_of(para_p3)})
            f2[pair_id] = {"canonical": none, "paraphrase": paraphrase, "polarity_flip": polarity, "p3w3_rows": exported}
    require(len(f1) == EXPECTED_F1_PAIRS, "P3-W3 pair-family count mismatch")
    require(len(f2) == EXPECTED_F2_PAIRS, "P3-W3 pair-family count mismatch")
    require(not (set(f1) & set(f2)), "F1 and F2 pair sets are not disjoint")
    require(len(set(f1) | set(f2)) == EXPECTED_AFFECTED_PAIRS, "P3-W3 pair-family count mismatch")
    require(consumed == {row_id_of(row) for row in p3w3_rows}, "unconsumed REFUTE row")
    return f1, f2
def classify_pair(family_name: str, family: dict[str, Any], sidecar_by_id: dict[str, dict[str, Any]], grammar_proof: dict[str, Any] | None = None) -> tuple[str, str]:
    if family_name == "F1":
        proof = grammar_proof or {}
        if (proof.get("sidecar_rule_claimed_failure") and proof.get("generator_source_sha_matches") and proof.get("validator_call_chain_verified") and proof.get("validator_source_path") and proof.get("fact_pair_id") and proof.get("production_rule_reproduction_result") and proof.get("exact_row_evidence_supplied")):
            return "F1_TRUE_POLARITY_GENERATION_DEFECT", "production grammar validator reproduced claimed generator defect"
        return "F1_AMBIGUOUS_REQUIRES_REVIEW", "production grammar-rule proof unavailable or unreproduced"
    canonical = family["canonical"]
    paraphrase = family["paraphrase"]
    canonical_sc = sidecar_by_id[str(canonical["id"])]
    paraphrase_sc = sidecar_by_id[str(paraphrase["id"])]
    if (
        canonical_sc.get("grammar_status") == "FAIL"
        and paraphrase_sc.get("canonical_status") == "UNRESOLVED"
        and "CANONICAL_ROW_KNOWN_GENERATOR_DEFECT" in set(paraphrase_sc.get("reason_codes", []))
    ):
        return "F2_CANONICAL_DEFECT_PROPAGATED_TO_DERIVATIVES", "canonical grammar failure propagates to paraphrase canonical_status UNRESOLVED"
    return "F2_AMBIGUOUS_REQUIRES_REVIEW", "deterministic recoverability proof unavailable"

def compact_reason_codes(sidecar: dict[str, Any]) -> str:
    return json.dumps(list(sidecar.get("reason_codes", [])), ensure_ascii=False, separators=(",", ":"))


def pair_record(
    pair_id: str,
    family_name: str,
    family: dict[str, Any],
    sidecar_by_id: dict[str, dict[str, Any]],
    fact_by_pair: dict[str, dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
    grammar_validator: dict[str, Any] | None = None,
    generator_source_sha_matches: bool = False,
) -> dict[str, Any]:
    canonical = family["canonical"]
    rows = {name: value for name, value in family.items() if isinstance(value, dict) and "id" in value}
    p3w3_by_row = {row_id_of(row): row for row in family.get("p3w3_rows", [])}
    members: dict[str, Any] = {}
    for name, row in rows.items():
        sidecar = sidecar_by_id[str(row["id"])]
        member = {
            "source_row": row,
            "sidecar": sidecar,
            "p3w3_exclusion_codes": exclusion_codes(p3w3_by_row.get(str(row["id"]), {})),
            "text_fields": text_fields(row),
            "claim_diagnostics_vs_canonical_claim": text_diagnostics(str(canonical["claim"]), str(row["claim"])),
            "evidence_diagnostics_vs_canonical_evidence": text_diagnostics(str(canonical["evidence"]), str(row["evidence"])),
        }
        member["grammar_rule_reproduction"] = reproduce_grammar_rule(member, (fact_by_pair or {}).get(pair_id), grammar_validator, generator_source_sha_matches)
        members[name] = member
    proof = members.get("polarity_flip", {}).get("grammar_rule_reproduction") if family_name == "F1" else members.get("canonical", {}).get("grammar_rule_reproduction")
    root_class, evidence = classify_pair(family_name, family, sidecar_by_id, proof)
    return {
        "pair_id": pair_id,
        "family": family_name,
        "members": members,
        "automatic_root_cause_class": root_class,
        "automatic_evidence": evidence,
        "remediation_state": "MANUAL_REVIEW_REQUIRED" if family_name == "F2" else ("REGENERATION_REQUIRED" if root_class == "F1_TRUE_POLARITY_GENERATION_DEFECT" else "MANUAL_REVIEW_REQUIRED"),
    }

def aggregate_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    families = Counter(record["family"] for record in records)
    composition: dict[str, Any] = {}
    text_patterns = Counter()
    reason_combos = Counter()
    propagation = Counter()
    for record in records:
        family = record["family"]
        composition.setdefault(family, {"intervention_type_x_final_label": Counter(), "intervention_type_x_grammar_status": Counter(), "intervention_type_x_exclusion_code": Counter()})
        for member in record["members"].values():
            row = member["source_row"]
            sidecar = member["sidecar"]
            comp = composition[family]
            comp["intervention_type_x_final_label"][f"{row['intervention_type']}|{row['final_label']}"] += 1
            comp["intervention_type_x_grammar_status"][f"{row['intervention_type']}|{sidecar['grammar_status']}"] += 1
            for code in member.get("p3w3_exclusion_codes", []):
                comp["intervention_type_x_exclusion_code"][f"{row['intervention_type']}|{code}"] += 1
            combo = tuple(sorted(sidecar.get("reason_codes", [])))
            reason_combos[combo] += 1
            text_patterns[f"claim:{member['claim_diagnostics_vs_canonical_claim']['pattern']}"] += 1
            text_patterns[f"evidence:{member['evidence_diagnostics_vs_canonical_evidence']['pattern']}"] += 1
        if record["family"] == "F2":
            canonical = record["members"]["canonical"]["sidecar"]
            paraphrase = record["members"]["paraphrase"]["sidecar"]
            others = []
            for name, member in sorted(record["members"].items()):
                if name not in {"canonical", "paraphrase"} and member["sidecar"].get("integrity_status") != "ELIGIBLE":
                    others.append(f"{name}:{member['sidecar'].get('integrity_status')}:{member['sidecar'].get('canonical_status')}")
            key = (
                f"canonical:{canonical.get('grammar_status')}:{','.join(canonical.get('reason_codes', []))}"
                f" -> paraphrase:{paraphrase.get('canonical_status')}:{','.join(paraphrase.get('reason_codes', []))}"
                f" -> other:{';'.join(others) if others else 'none'}"
            )
            propagation[key] += 1
    serial_comp = {
        family: {name: dict(sorted(counter.items())) for name, counter in values.items()}
        for family, values in composition.items()
    }
    return {
        "family_counts": {
            "F1_pair_count": int(families["F1"]),
            "F2_pair_count": int(families["F2"]),
            "affected_unique_pair_count": len(records),
            "p3w3_exported_REFUTE_row_count": EXPECTED_REFUTE_ROWS,
            "p3w4_affected_member_row_count": EXPECTED_AFFECTED_MEMBER_ROWS,
            "F1_polarity_flip_members": EXPECTED_F1_AFFECTED_MEMBERS,
            "F2_complete_triple_members": EXPECTED_F2_AFFECTED_MEMBERS,
        },
        "intervention_composition": serial_comp,
        "reason_code_combinations": {"|".join(combo): count for combo, count in sorted(reason_combos.items())},
        "text_difference_patterns": dict(sorted(text_patterns.items())),
        "f2_propagation_patterns": dict(sorted(propagation.items())),
    }


def affected_member_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["family"] == "F1":
            rows.append(record["members"]["polarity_flip"]["source_row"])
        else:
            for name in ("canonical", "paraphrase", "polarity_flip"):
                rows.append(record["members"][name]["source_row"])
    return rows


def scenario_from_sets(
    action_review_ids: set[str],
    action_regen_ids: set[str],
    action_exclude_ids: set[str],
    current_retained_ids: set[str],
    conditional_rows: list[dict[str, Any]],
    universe_ids: set[str],
    preserves: bool | str,
    derivative_outcome_unresolved: bool = False,
) -> dict[str, Any]:
    require(not (action_review_ids & action_regen_ids or action_review_ids & action_exclude_ids or action_review_ids & current_retained_ids or action_regen_ids & action_exclude_ids or action_regen_ids & current_retained_ids or action_exclude_ids & current_retained_ids), "scenario row-id conservation failed")
    require(action_review_ids | action_regen_ids | action_exclude_ids | current_retained_ids == universe_ids, "scenario row-id conservation failed")
    conditional_ids = {str(row["id"]) for row in conditional_rows}
    labels = Counter(str(row["final_label"]) for row in conditional_rows)
    support_contribution = labels["SUPPORT"]
    refute_contribution = labels["REFUTE"]
    total_support = EXPECTED_ELIGIBLE_SUPPORT + support_contribution
    total_refute = EXPECTED_ELIGIBLE_REFUTE + refute_contribution
    return {
        "baseline_eligible_SUPPORT": EXPECTED_ELIGIBLE_SUPPORT,
        "baseline_eligible_REFUTE": EXPECTED_ELIGIBLE_REFUTE,
        "conditional_SUPPORT_contribution": support_contribution,
        "conditional_REFUTE_contribution": refute_contribution,
        "potential_total_SUPPORT": total_support,
        "potential_total_REFUTE": total_refute,
        "rows_requiring_human_review": len(action_review_ids),
        "rows_requiring_regeneration": len(action_regen_ids),
        "rows_remaining_excluded": len(action_exclude_ids),
        "action_review_row_ids": sorted(action_review_ids),
        "action_regenerate_row_ids": sorted(action_regen_ids),
        "action_exclude_row_ids": sorted(action_exclude_ids),
        "current_retained_row_ids": sorted(current_retained_ids),
        "conditional_potential_admitted_row_ids": sorted(conditional_ids),
        "derivative_outcome_unresolved": derivative_outcome_unresolved,
        "preserves_pair_canonical_authority": preserves,
        "polarity_minimum_50_could_be_met": total_refute >= MINIMUM_REFUTE_READY_COUNT,
        "diagnostic_only": True,
    }


def scenario_diagnostics(pair_records: list[dict[str, Any]]) -> dict[str, Any]:
    affected = affected_member_rows(pair_records)
    universe_ids = {str(row["id"]) for row in affected}
    f1_pol = [record["members"]["polarity_flip"]["source_row"] for record in pair_records if record["family"] == "F1"]
    f2_none = [record["members"]["canonical"]["source_row"] for record in pair_records if record["family"] == "F2"]
    f2_triples = [member["source_row"] for record in pair_records if record["family"] == "F2" for member in record["members"].values()]
    f1_ids = {str(row["id"]) for row in f1_pol}
    f2_none_ids = {str(row["id"]) for row in f2_none}
    f2_triple_ids = {str(row["id"]) for row in f2_triples}
    r3 = scenario_from_sets(set(), f2_none_ids, universe_ids - f2_none_ids, set(), [], universe_ids, True, derivative_outcome_unresolved=True)
    r3.update({
        "confirmed_SUPPORT_contribution": 0,
        "confirmed_REFUTE_contribution": 0,
        "conditional_REFUTE_contribution_if_label_contract_preserved": len([row for row in f2_none if row.get("final_label") == "REFUTE"]),
        "conditional_SUPPORT_contribution_if_label_contract_preserved": len([row for row in f2_none if row.get("final_label") == "SUPPORT"]),
        "conditional_REFUTE_contribution": 0,
        "conditional_SUPPORT_contribution": 0,
        "potential_total_SUPPORT": "unresolved",
        "potential_total_REFUTE": "unresolved",
        "polarity_minimum_50_could_be_met": "unresolved",
    })
    r4 = scenario_from_sets(set(), f1_ids | f2_triple_ids, universe_ids - (f1_ids | f2_triple_ids), set(), f1_pol + f2_triples, universe_ids, True)
    r4.update({
        "confirmed_SUPPORT_contribution": 0,
        "confirmed_REFUTE_contribution": 0,
        "conditional_SUPPORT_contribution_if_label_contract_preserved": r4["conditional_SUPPORT_contribution"],
        "conditional_REFUTE_contribution_if_label_contract_preserved": r4["conditional_REFUTE_contribution"],
        "regeneration_result_required": True,
    })
    return {
        "R0": scenario_from_sets(set(), set(), universe_ids, set(), [], universe_ids, True),
        "R1": scenario_from_sets(f2_none_ids, set(), universe_ids - f2_none_ids, set(), f2_none, universe_ids, False),
        "R2": scenario_from_sets(f2_triple_ids, set(), universe_ids - f2_triple_ids, set(), f2_triples, universe_ids, False),
        "R3": r3,
        "R4": r4,
        "R5": {
            "baseline_eligible_SUPPORT": EXPECTED_ELIGIBLE_SUPPORT,
            "baseline_eligible_REFUTE": EXPECTED_ELIGIBLE_REFUTE,
            "conditional_SUPPORT_contribution": 0,
            "conditional_REFUTE_contribution": "unknown",
            "potential_total_SUPPORT": EXPECTED_ELIGIBLE_SUPPORT,
            "potential_total_REFUTE": "unknown",
            "minimum_required_new_REFUTE_rows": MINIMUM_REFUTE_READY_COUNT,
            "actually_available_new_REFUTE_rows": "unknown",
            "polarity_minimum_50_could_be_met": "unresolved until authority exists",
            "rows_requiring_human_review": 0,
            "rows_requiring_regeneration": 0,
            "rows_remaining_excluded": len(universe_ids),
            "action_review_row_ids": [],
            "action_regenerate_row_ids": [],
            "action_exclude_row_ids": sorted(universe_ids),
            "current_retained_row_ids": [],
            "conditional_potential_admitted_row_ids": [],
            "preserves_pair_canonical_authority": False,
            "diagnostic_only": True,
        },
    }

def refute_row_count(record: dict[str, Any]) -> int:
    return sum(1 for member in record["members"].values() if member["source_row"].get("final_label") == "REFUTE")


def records_with_remediation(pair_records: list[dict[str, Any]], state: str) -> list[dict[str, Any]]:
    return [record for record in pair_records if record.get("remediation_state") == state]


def potential_authority_yield(pair_records: list[dict[str, Any]]) -> dict[str, Any]:
    auto_records = records_with_remediation(pair_records, "TEXTUALLY_RECOVERABLE")
    manual_records = records_with_remediation(pair_records, "MANUAL_REVIEW_REQUIRED")
    regen_records = records_with_remediation(pair_records, "REGENERATION_REQUIRED")
    conflict_records = records_with_remediation(pair_records, "SEMANTIC_CONFLICT")
    scenarios = scenario_diagnostics(pair_records)
    return {
        "automatic_recoverable_pair_count": len(auto_records),
        "automatic_recoverable_REFUTE_row_count": sum(refute_row_count(record) for record in auto_records),
        "manual_review_pair_count": len(manual_records),
        "manual_review_REFUTE_row_count": sum(refute_row_count(record) for record in manual_records),
        "regeneration_pair_count": len(regen_records),
        "regeneration_REFUTE_row_count": sum(refute_row_count(record) for record in regen_records),
        "semantic_conflict_pair_count": len(conflict_records),
        "semantic_conflict_REFUTE_row_count": sum(refute_row_count(record) for record in conflict_records),
        "potential_polarity_class_count_after_each_scenario": scenarios,
        "minimum_50_count_readiness_by_scenario": {name: value.get("polarity_minimum_50_could_be_met") for name, value in scenarios.items()},
    }

def validate_output_namespace(output_json: Path, output_pair_jsonl: Path, output_review_csv: Path) -> Path:
    parents = {output_json.parent, output_pair_jsonl.parent, output_review_csv.parent}
    require(len(parents) == 1, "output namespace mismatch")
    parent = output_json.parent
    require(not parent.exists(), "output namespace already exists")
    require(not output_json.exists() and not output_pair_jsonl.exists() and not output_review_csv.exists(), "partial output artifact already exists")
    return parent


def write_outputs_atomically(output_json: Path, output_pair_jsonl: Path, output_review_csv: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    final_parent = validate_output_namespace(output_json, output_pair_jsonl, output_review_csv)
    base = final_parent.parent if str(final_parent.parent) else Path(".")
    tmp_parent = Path(tempfile.mkdtemp(prefix=final_parent.name + ".tmp.", dir=str(base)))
    try:
        tmp_json = tmp_parent / output_json.name
        tmp_pair = tmp_parent / output_pair_jsonl.name
        tmp_csv = tmp_parent / output_review_csv.name
        tmp_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_pair_jsonl(tmp_pair, records)
        write_review_csv(tmp_csv, records)
        require(tmp_json.exists() and tmp_pair.exists() and tmp_csv.exists(), "atomic output staging failed")
        tmp_parent.rename(final_parent)
    except Exception:
        shutil.rmtree(tmp_parent, ignore_errors=True)
        raise

def write_pair_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for record in records:
            if record["family"] != "F2":
                continue
            members = record["members"]
            canonical = members["canonical"]["source_row"]
            paraphrase = members["paraphrase"]["source_row"]
            polarity = members["polarity_flip"]["source_row"]
            writer.writerow({
                "pair_id": record["pair_id"],
                "canonical_none_row_id": canonical["id"],
                "paraphrase_row_id": paraphrase["id"],
                "polarity_flip_row_id": polarity["id"],
                "canonical_final_label": canonical["final_label"],
                "paraphrase_final_label": paraphrase["final_label"],
                "polarity_flip_final_label": polarity["final_label"],
                "canonical_claim": canonical["claim"],
                "paraphrase_claim": paraphrase["claim"],
                "polarity_flip_claim": polarity["claim"],
                "canonical_evidence": canonical["evidence"],
                "paraphrase_evidence": paraphrase["evidence"],
                "polarity_flip_evidence": polarity["evidence"],
                "canonical_grammar_status": members["canonical"]["sidecar"]["grammar_status"],
                "paraphrase_grammar_status": members["paraphrase"]["sidecar"]["grammar_status"],
                "polarity_flip_grammar_status": members["polarity_flip"]["sidecar"]["grammar_status"],
                "canonical_reason_codes": compact_reason_codes(members["canonical"]["sidecar"]),
                "paraphrase_reason_codes": compact_reason_codes(members["paraphrase"]["sidecar"]),
                "polarity_flip_reason_codes": compact_reason_codes(members["polarity_flip"]["sidecar"]),
                "canonical_claim_text_diff_summary": members["canonical"]["claim_diagnostics_vs_canonical_claim"]["summary"],
                "paraphrase_claim_text_diff_summary": members["paraphrase"]["claim_diagnostics_vs_canonical_claim"]["summary"],
                "polarity_flip_claim_text_diff_summary": members["polarity_flip"]["claim_diagnostics_vs_canonical_claim"]["summary"],
                "canonical_evidence_text_diff_summary": members["canonical"]["evidence_diagnostics_vs_canonical_evidence"]["summary"],
                "paraphrase_evidence_text_diff_summary": members["paraphrase"]["evidence_diagnostics_vs_canonical_evidence"]["summary"],
                "polarity_flip_evidence_text_diff_summary": members["polarity_flip"]["evidence_diagnostics_vs_canonical_evidence"]["summary"],
                "automatic_root_cause_class": record["automatic_root_cause_class"],
                "automatic_evidence": record["automatic_evidence"],
                "human_canonical_semantics": "",
                "human_paraphrase_semantics": "",
                "human_polarity_flip_semantics": "",
                "human_grammar_validity": "",
                "human_authority_decision": "",
                "human_notes": "",
            })

def partition_decision_pair_ids(pair_records: list[dict[str, Any]]) -> dict[str, list[str]]:
    blocking = [record["pair_id"] for record in pair_records if record.get("remediation_state") in {"MANUAL_REVIEW_REQUIRED", "SEMANTIC_CONFLICT"}]
    supporting = [record["pair_id"] for record in pair_records if record.get("remediation_state") in {"TEXTUALLY_RECOVERABLE", "REGENERATION_REQUIRED"}]
    return {"blocking": sorted(blocking), "supporting": sorted(supporting)}


REMEDIATION_STATE_ORDER = (
    "TEXTUALLY_RECOVERABLE",
    "REGENERATION_REQUIRED",
    "MANUAL_REVIEW_REQUIRED",
    "SEMANTIC_CONFLICT",
)


def remediation_states(records: list[dict[str, Any]]) -> list[str]:
    observed = {str(record.get("remediation_state")) for record in records if record.get("remediation_state")}
    return [state for state in REMEDIATION_STATE_ORDER if state in observed]


def f2_actions_from_states(states: list[str]) -> list[str] | str:
    mapping = {
        "TEXTUALLY_RECOVERABLE": "textual review/recovery candidate",
        "REGENERATION_REQUIRED": "regeneration",
        "MANUAL_REVIEW_REQUIRED": "manual textual/semantic review",
        "SEMANTIC_CONFLICT": "semantic conflict resolution",
    }
    actions = [mapping[state] for state in states if state in mapping]
    if len(actions) == 1:
        return actions[0]
    return actions or "none"


def provisional_decision(pair_records: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    f1 = [record for record in pair_records if record["family"] == "F1"]
    f2 = [record for record in pair_records if record["family"] == "F2"]
    f1_states = remediation_states(f1)
    f2_states = remediation_states(f2)
    f1_regen = [record for record in f1 if record.get("remediation_state") == "REGENERATION_REQUIRED"]
    f1_manual = [record for record in f1 if record.get("remediation_state") == "MANUAL_REVIEW_REQUIRED"]
    f2_textual = [record for record in f2 if record.get("remediation_state") == "TEXTUALLY_RECOVERABLE"]
    f2_regen = [record for record in f2 if record.get("remediation_state") == "REGENERATION_REQUIRED"]
    f2_manual = [record for record in f2 if record.get("remediation_state") == "MANUAL_REVIEW_REQUIRED"]
    f2_conflict = [record for record in f2 if record.get("remediation_state") == "SEMANTIC_CONFLICT"]
    criteria = {
        "F1_regeneration_required_count": len(f1_regen),
        "F1_manual_review_required_count": len(f1_manual),
        "F2_textually_recoverable_count": len(f2_textual),
        "F2_regeneration_required_count": len(f2_regen),
        "F2_manual_review_required_count": len(f2_manual),
        "F2_semantic_conflict_count": len(f2_conflict),
        "F1_remediation_states": f1_states,
        "F2_remediation_states": f2_states,
        "F1_remediation_mixed": len(f1_states) > 1,
        "F2_remediation_mixed": len(f2_states) > 1,
        "F1_action": f2_actions_from_states(f1_states) if len(f1_states) > 1 else ("regeneration" if "REGENERATION_REQUIRED" in f1_states else ("manual review" if "MANUAL_REVIEW_REQUIRED" in f1_states else "none")),
        "F2_action": f2_actions_from_states(f2_states),
        "F2_regeneration_approved": f2_states == ["REGENERATION_REQUIRED"],
    }
    separate_pairs = [
        ({"REGENERATION_REQUIRED"}, {"TEXTUALLY_RECOVERABLE"}),
        ({"REGENERATION_REQUIRED"}, {"MANUAL_REVIEW_REQUIRED"}),
        ({"REGENERATION_REQUIRED"}, {"SEMANTIC_CONFLICT"}),
        ({"MANUAL_REVIEW_REQUIRED"}, {"REGENERATION_REQUIRED"}),
        ({"MANUAL_REVIEW_REQUIRED"}, {"TEXTUALLY_RECOVERABLE"}),
    ]
    f1_set = set(f1_states)
    f2_set = set(f2_states)
    if any(left <= f1_set and right <= f2_set for left, right in separate_pairs):
        return "P3W4_F1_AND_F2_REQUIRE_SEPARATE_REMEDIATION", criteria
    if f2_set == {"REGENERATION_REQUIRED"}:
        return "P3W4_F2_CANONICAL_ROWS_REQUIRE_REGENERATION", criteria
    if f2_set == {"TEXTUALLY_RECOVERABLE"} and not f1_set:
        return "P3W4_EXISTING_F2_AUTHORITY_RECOVERABLE_BY_TEXTUAL_REVIEW", criteria
    return "P3W4_AUDIT_BLOCKED", criteria

def grammar_validator_summary_authority(record: dict[str, Any]) -> dict[str, Any]:
    validate_grammar_validator_record(record)
    return {field: record.get(field) for field in VALIDATOR_SUMMARY_FIELDS}


def validate_pair_validator_metadata(pair_records: list[dict[str, Any]], authority: dict[str, Any]) -> None:
    for record in pair_records:
        for member in record.get("members", {}).values():
            proof = member.get("grammar_rule_reproduction", {})
            observed = {field: proof.get(field) for field in VALIDATOR_SUMMARY_FIELDS}
            require(observed == authority, "P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH")


def validate_runtime_grammar_validator_authority(authority: dict[str, Any], blob_identity: dict[str, Any], pair_records: list[dict[str, Any]]) -> None:
    require(authority.get("validator_source_path") == authority.get("validator_authority_source"), "P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH")
    require(blob_identity.get("path") == authority.get("validator_authority_source"), "P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH")
    require(authority.get("validator_source_sha256"), "P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH")
    require(authority.get("validator_definition_kind"), "P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH")
    require(authority.get("validator_call_site_authorized") is True, "P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH")
    require(isinstance(authority.get("validator_authorized_call_sites"), list), "P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH")
    require(authority.get("validator_authorized_call_site_count") == len(authority["validator_authorized_call_sites"]), "P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH")
    require(authority.get("validator_call_chain_verified") is True, "P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH")
    require(authority.get("validator_signature"), "P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH")
    validate_pair_validator_metadata(pair_records, authority)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3-W4 canonical grammar authority audit")
    parser.add_argument("--data", required=True)
    parser.add_argument("--controlled-integrity-sidecar-path", required=True)
    parser.add_argument("--p3w3-summary-json", required=True)
    parser.add_argument("--p3w3-refute-jsonl", required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-sidecar-semantic-sha256", required=True)
    parser.add_argument("--expected-p3w3-summary-sha256", required=True)
    parser.add_argument("--expected-p3w3-refute-jsonl-sha256", required=True)
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--dev-ratio", required=True, type=float)
    parser.add_argument("--expected-train-row-count", required=True, type=int)
    parser.add_argument("--expected-train-row-identity-hash", required=True)
    parser.add_argument("--expected-p3w3-execution-commit", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-pair-jsonl", required=True)
    parser.add_argument("--output-review-csv", required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    sidecar_path = Path(args.controlled_integrity_sidecar_path)
    p3w3_summary_path = Path(args.p3w3_summary_json)
    p3w3_refute_path = Path(args.p3w3_refute_jsonl)
    output_json = Path(args.output_json)
    output_pair_jsonl = Path(args.output_pair_jsonl)
    output_review_csv = Path(args.output_review_csv)
    validate_output_namespace(output_json, output_pair_jsonl, output_review_csv)
    require(file_sha256(data_path) == args.expected_data_sha256, "dataset identity mismatch")
    verify_tracked_artifact(p3w3_summary_path, root, args.expected_p3w3_summary_sha256)
    verify_tracked_artifact(p3w3_refute_path, root, args.expected_p3w3_refute_jsonl_sha256)
    source_rows = load_jsonl(data_path)
    sidecar_rows = load_jsonl(sidecar_path)
    require(semantic_sidecar_sha256(sidecar_rows) == args.expected_sidecar_semantic_sha256, "sidecar semantic identity mismatch")
    intervention_authority = load_intervention_authority(root)
    validate_source_rows(source_rows, intervention_authority["values"])
    train_rows, dev_rows, _dev_ids = validate_split_contract(
        source_rows,
        args.split_seed,
        args.dev_ratio,
        args.expected_train_row_count,
        args.expected_train_row_identity_hash,
    )
    sidecar_by_id = validate_sidecar(source_rows, sidecar_rows, args.expected_data_sha256)
    for row in train_rows:
        require(sidecar_by_id[str(row["id"])].get("split") == "train", "sidecar/source identity mismatch")
    for row in dev_rows:
        require(sidecar_by_id[str(row["id"])].get("split") == "dev", "sidecar/source identity mismatch")
    git_gates = verify_git_gates(root, args.execution_commit)
    source_blob_identities = verify_source_blob_identities(root, args.expected_p3w3_execution_commit)
    p3w3_summary = load_json(p3w3_summary_path)
    p3w3_refute_rows = load_jsonl(p3w3_refute_path)
    p3w3_authority = validate_p3w3_artifacts(p3w3_summary, p3w3_refute_rows, args.expected_p3w3_execution_commit)
    f1, f2 = reconstruct_families(train_rows, sidecar_by_id, p3w3_refute_rows, p3w3_authority)
    affected_pair_ids = set(f1) | set(f2)
    all_pair_ids = {str(row["pair_id"]) for row in source_rows}
    generator_source_sha = verify_generator_source_authority(root, p3w3_refute_rows)
    integrity_builder_sha = verify_integrity_builder_authority(root, p3w3_refute_rows)
    fact_by_pair = load_fact_authority(root, all_pair_ids, affected_pair_ids)
    provenance = resolve_grammar_rule_provenance(root)
    grammar_validator = load_production_grammar_validator(root)
    grammar_validator_source_blob_identity = verify_resolved_validator_blob_identity(root, args.expected_p3w3_execution_commit, grammar_validator)
    pair_records = [
        *(pair_record(pair_id, "F1", family, sidecar_by_id, fact_by_pair, provenance, grammar_validator, True) for pair_id, family in sorted(f1.items())),
        *(pair_record(pair_id, "F2", family, sidecar_by_id, fact_by_pair, provenance, grammar_validator, True) for pair_id, family in sorted(f2.items())),
    ]
    grammar_validator_authority = grammar_validator_summary_authority(grammar_validator)
    validate_runtime_grammar_validator_authority(grammar_validator_authority, grammar_validator_source_blob_identity, pair_records)
    decision, criteria = provisional_decision(pair_records)
    decision_pair_ids = partition_decision_pair_ids(pair_records)
    blocking_ids = decision_pair_ids["blocking"]
    supporting_ids = decision_pair_ids["supporting"]
    aggregates = aggregate_pairs(pair_records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "P3W4_AUDIT_EXECUTION_COMPLETE_PENDING_RESULT_REVIEW",
        "decision": decision,
        "static_implementation_decision": STATIC_DECISION,
        "corrected_interpretation": (
            "The 119 ambiguous REFUTE rows are paraphrase derivatives of 119 defective canonical REFUTE rows, "
            "not an independent unresolved polarity cohort."
        ),
        "authority": {
            "F1_pairs": len(f1),
            "F2_pairs": len(f2),
            "affected_pairs": len(pair_records),
            "p3w3_exported_REFUTE_row_count": EXPECTED_REFUTE_ROWS,
            "p3w4_affected_member_row_count": EXPECTED_AFFECTED_MEMBER_ROWS,
            "F1_polarity_flip_members": EXPECTED_F1_AFFECTED_MEMBERS,
            "F2_complete_triple_members": EXPECTED_F2_AFFECTED_MEMBERS,
            "generator_defect_REFUTE": EXPECTED_GENERATOR_DEFECT_REFUTE,
            "integrity_source_required_REFUTE": EXPECTED_INTEGRITY_SOURCE_REQUIRED_REFUTE,
        },
        "source_text_fields_discovered": ["claim", "evidence"],
        "intervention_type_authority": {"source": intervention_authority["source"], "values": sorted(intervention_authority["values"])},
        "grammar_rule_provenance": provenance,
        "grammar_validator_authority": grammar_validator_authority,
        "generator_source_sha256": generator_source_sha,
        "integrity_builder_sha256": integrity_builder_sha,
        "source_blob_identities": source_blob_identities,
        "generator_source_blob_identity": source_blob_identities["generator_source_blob_identity"],
        "grammar_validator_source_blob_identity": grammar_validator_source_blob_identity,
        "stage182_regression_oracle_blob_identity": source_blob_identities["stage182_regression_oracle_blob_identity"],
        "integrity_builder_source_blob_identity": source_blob_identities["integrity_builder_source_blob_identity"],
        "pair_level_root_cause_classes": dict(Counter(record["automatic_root_cause_class"] for record in pair_records)),
        "manual_review_csv_schema": REVIEW_COLUMNS,
        "aggregates": aggregates,
        "potential_authority_yield": potential_authority_yield(pair_records),
        "counterfactual_scenarios": scenario_diagnostics(pair_records),
        "decision_criteria_audit": criteria,
        "decision_supporting_pair_ids": supporting_ids,
        "decision_blocking_pair_ids": blocking_ids,
        "audit_execution_completed": True,
        "result_static_review_completed": False,
        "human_review_required": True,
        "production_repair_approved": False,
        "polarity_supervision_released": False,
        "execution_isolation": EXECUTION_ISOLATION,
        "git_gates": git_gates,
        "production_behavior_modified": False,
        "polarity_local_training_ready": False,
        "A1_A3_released": False,
        "remaining_blockers": [
            "P2_POLARITY_LOCAL_SUPERVISION_NOT_TRAINING_READY",
            "P3W4_RESULT_STATIC_REVIEW_PENDING",
        ],
    }
    write_outputs_atomically(output_json, output_pair_jsonl, output_review_csv, summary, pair_records)
    return summary


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
