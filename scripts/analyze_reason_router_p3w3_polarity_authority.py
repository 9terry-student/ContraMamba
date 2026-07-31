from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_controlled_v5 as v5  # noqa: E402
from scripts import train_controlled_v6b_minimal as trainer  # noqa: E402


SCHEMA_VERSION = "reason_router_p3w3_polarity_authority_audit_v3"
EXPECTED_SPLIT_SEED = 174
EXPECTED_DEV_RATIO = 0.2
EXPECTED_TRAIN_ROWS = 2880
EXPECTED_TRAIN_IDENTITY = "cbce1775ddc73f2fbad024ded6a314d15e2eb1988ef107fa72a5eacbdd836784"
EXPECTED_DATA_SHA256 = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
EXPECTED_SIDECAR_SEMANTIC_SHA256 = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
INTERVENTION_TYPE_AUTHORITY = "scripts/build_controlled_v5.py::INTERVENTION_TYPES"
AUTHORIZED_INTERVENTION_TYPES = frozenset(v5.INTERVENTION_TYPES)

EXCLUSION_CODES = (
    "P2_SIDECAR_MISSING",
    "P2_SPLIT_MISMATCH",
    "P2_CANONICAL_ROW_ID_MISMATCH",
    "P2_SIDECAR_SOURCE_BINARY_MISMATCH",
    "P2_POLARITY_INTERVENTION_CONTRACT_FAIL",
    "P2_INTEGRITY_SOURCE_REQUIRED",
    "P2_GENERATOR_STATUS_DEFECT",
    "P2_PRIMARY_REASON_AXIS_CONFLICT",
    "P2_FAILURE_FINAL_LABEL_MISMATCH",
    "P2_AUTHORIZED_FINAL_LABEL_MISMATCH",
    "P2_POLARITY_TARGET_FINAL_MISMATCH",
)

INDEPENDENT_DEFECT_CODES = {
    "P2_SIDECAR_SOURCE_BINARY_MISMATCH",
    "P2_PRIMARY_REASON_AXIS_CONFLICT",
    "P2_FAILURE_FINAL_LABEL_MISMATCH",
    "P2_AUTHORIZED_FINAL_LABEL_MISMATCH",
    "P2_POLARITY_TARGET_FINAL_MISMATCH",
}


PRODUCTION_INDEPENDENT_EXCLUSION_CODES = set(INDEPENDENT_DEFECT_CODES)
PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY = "PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY"
INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE = "INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE"
AMBIGUOUS_INTEGRITY_EVIDENCE = "AMBIGUOUS_INTEGRITY_EVIDENCE"
CLEAN_GENERATOR_EVIDENCE = "CLEAN"
GENERATOR_EVIDENCE_PROOF_CONTRACT = {
    "available": False,
    "source": None,
    "clause": None,
    "accepted_reason_codes": [],
    "required_preserved_axes": [],
}
COUNTERFACTUAL_IGNORES = {
    "C0": set(),
    "C1": {"P2_GENERATOR_STATUS_DEFECT"},
    "C2": {"P2_INTEGRITY_SOURCE_REQUIRED"},
    "C3": {"P2_POLARITY_INTERVENTION_CONTRACT_FAIL"},
    "C4": {"P2_GENERATOR_STATUS_DEFECT", "P2_POLARITY_INTERVENTION_CONTRACT_FAIL"},
}

EXECUTION_ISOLATION_AUDIT = {
    "model_loaded": False,
    "encoder_loaded": False,
    "tokenizer_loaded": False,
    "cuda_required": False,
    "forward_executed": False,
    "backward_executed": False,
    "optimizer_step_executed": False,
    "scheduler_step_executed": False,
    "training_executed": False,
    "dev_evaluation_executed": False,
    "checkpoint_loaded": False,
    "checkpoint_written": False,
    "prediction_artifact_used": False,
    "A0_artifact_used": False,
    "EMA_artifact_used": False,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl_raw(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} is not an object: {path}")
            rows.append(row)
    return rows


def semantic_sidecar_sha256(rows: list[dict[str, Any]]) -> str:
    return trainer._stage187_semantic_sidecar_sha256(rows)


def ordered_train_identity_hash(records: list[dict[str, Any]]) -> str:
    return trainer._p3w1_ordered_train_identity(records)["ordered_train_row_identity_hash"]


def normalize_label(value: Any) -> str:
    label = trainer._s28e_normalize_label(value)
    if label not in {"REFUTE", "SUPPORT", "NOT_ENTITLED"}:
        raise ValueError(f"P2_UNKNOWN_FINAL_LABEL: value={value!r}")
    return str(label)


def exact_binary(row: dict[str, Any], field: str, row_id: str, prefix: str = "P2") -> int:
    value = row.get(field)
    if isinstance(value, bool) or value not in (0, 1):
        raise ValueError(f"{prefix}_EXACT_BINARY_VALIDATION_FAILED: row_id={row_id} field={field} value={value!r}")
    return int(value)


def canonical_polarity(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return {0: "NONE", 1: "REFUTE", 2: "SUPPORT"}.get(value, "UNKNOWN")
    return str(value).strip().upper()


def _inc(counter: Counter, *parts: Any) -> None:
    counter["|".join(str(part) for part in parts)] += 1


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def load_reason_sidecar(
    *,
    data_path: Path,
    source_records: list[dict[str, Any]],
    sidecar_path: Path,
    expected_data_sha256: str,
    expected_semantic_sha256: str,
    enforce_authoritative_paths: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if _file_sha256(data_path) != expected_data_sha256:
        raise ValueError("P2_METADATA_SOURCE_MISMATCH: dataset SHA mismatch")
    rows = _load_jsonl_raw(sidecar_path)
    observed_semantic = semantic_sidecar_sha256(rows)
    if observed_semantic != expected_semantic_sha256:
        raise ValueError("P2 sidecar semantic SHA mismatch")
    if enforce_authoritative_paths:
        trainer._p2_load_reason_integrity_sidecar(
            data_path=data_path,
            source_records=source_records,
            sidecar_path=sidecar_path,
            expected_semantic_sha256=expected_semantic_sha256,
        )
    source_ids = [str(row.get("id", "")) for row in source_records]
    sidecar_ids = [str(row.get("row_id", "")) for row in rows]
    if source_ids != sidecar_ids or len(set(sidecar_ids)) != len(sidecar_ids):
        raise ValueError("P2 sidecar row_id order is not authoritative")
    for line_number, row in enumerate(rows, 1):
        missing = [field for field in trainer.P2_SIDE_CAR_REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"P2 sidecar line {line_number} missing fields: {missing}")
        if row.get("source_dataset_sha256") != expected_data_sha256:
            raise ValueError(f"P2 sidecar line {line_number} source_dataset_sha256 mismatch")
        if not isinstance(row.get("reason_codes"), list):
            raise ValueError(f"P2 sidecar line {line_number} reason_codes must be a list")
    return {str(row["row_id"]): row for row in rows}, {
        "path": str(sidecar_path.resolve()),
        "schema_fields": sorted(rows[0].keys()) if rows else [],
        "expected_semantic_sha256": expected_semantic_sha256,
        "observed_semantic_sha256": observed_semantic,
        "row_id_join": "exact_one_to_one_source_order",
        "production_functions": {
            "sidecar_semantic_sha256": "scripts/train_controlled_v6b_minimal.py::_stage187_semantic_sidecar_sha256",
            "sidecar_loader": "scripts/train_controlled_v6b_minimal.py::_p2_load_reason_integrity_sidecar",
            "production_row_authority": "scripts/train_controlled_v6b_minimal.py::_p2_prepare_reason_supervision_train_only",
            "generator_status_normalizer": "scripts/train_controlled_v6b_minimal.py::_p2_normalized_generator_status",
            "canonical_lineage": "scripts/train_controlled_v6b_minimal.py::_p2_resolve_canonical_lineage_for_split",
        },
        "intervention_type_authority": INTERVENTION_TYPE_AUTHORITY,
        "intervention_contract_pass_field": "intervention_contract_status",
        "generator_integrity_status_fields": list(trainer.P2_GENERATOR_COMPONENT_STATUS_FIELDS),
        "diagnostic_provenance_fields": [
            "reason_codes",
            "audit_changed_axes",
            "audit_expected_axes",
            "audit_preserved_axes",
            "generator_source_path",
            "generator_source_sha256",
            "integrity_builder_sha256",
            "stage182a_report_sha256",
            "stage184a_report_sha256",
        ],
        "generator_integrity_status_normalization": {
            "CLEAN": "all authoritative component status fields are PASS",
            "DEFECT": "at least one authoritative component status field is non-PASS",
            "UNRESOLVED": "one or more authoritative component status fields are missing or unsupported",
        },
        "generator_evidence_proof_contract_available": False,
        "generator_evidence_proof_contract_source": None,
        "semantic_interpretation_note": (
            "Static repository review found sidecar diagnostic provenance fields but no explicit "
            "repository-backed polarity-only mismatch proof contract; field names alone are not treated as proof."
        ),
    }


def validate_split_contract(
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    expected_train_count: int,
    expected_train_identity_hash: str,
) -> str:
    all_ids = [str(row.get("id", "")) for row in train_records + dev_records]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("P2_DUPLICATE_ROW_ID")
    train_pairs = {str(row.get("pair_id", "")) for row in train_records if str(row.get("pair_id", ""))}
    dev_pairs = {str(row.get("pair_id", "")) for row in dev_records if str(row.get("pair_id", ""))}
    leakage = sorted(train_pairs & dev_pairs)
    if leakage:
        raise ValueError(f"P2_TRAIN_DEV_PAIR_ID_LEAKAGE: pair_ids={leakage[:10]}")
    if len(train_records) != expected_train_count:
        raise ValueError("P3W3_TRAIN_ROW_COUNT_MISMATCH")
    observed = ordered_train_identity_hash(train_records)
    if observed != expected_train_identity_hash:
        raise ValueError("P3W3_ORDERED_TRAIN_IDENTITY_MISMATCH")
    return observed


def _validate_source_metadata(record: dict[str, Any]) -> tuple[int, int, int, str, str, str]:
    row_id = str(record.get("id", ""))
    missing = [field for field in trainer.P2_SOURCE_REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"P2_REQUIRED_SOURCE_METADATA_MISSING: row_id={row_id} missing={missing}")
    frame = exact_binary(record, "frame_compatible_label", row_id)
    predicate = exact_binary(record, "predicate_covered_label", row_id)
    sufficiency = exact_binary(record, "sufficiency_label", row_id)
    raw_primary = str(record.get("primary_failure_type", "")).strip().lower()
    if raw_primary not in trainer.P2_PRIMARY_FAILURE_TYPES:
        raise ValueError(f"P2_UNKNOWN_PRIMARY_FAILURE_TYPE: row_id={row_id} value={raw_primary!r}")
    final_label = normalize_label(record.get("final_label"))
    intervention_type = str(record.get("intervention_type", "")).strip().lower()
    if intervention_type not in AUTHORIZED_INTERVENTION_TYPES:
        raise ValueError(
            "P2_UNKNOWN_INTERVENTION_TYPE: "
            f"row_id={row_id} value={intervention_type!r}"
        )
    return frame, predicate, sufficiency, raw_primary, final_label, intervention_type



def validate_train_sidecar_identity_and_lineage(train_records: list[dict[str, Any]], sidecar_by_id: dict[str, dict[str, Any]]) -> dict[str, str]:
    missing_sidecar_ids = [
        str(record.get("id", ""))
        for record in train_records
        if str(record.get("id", "")) not in sidecar_by_id
    ]
    if missing_sidecar_ids:
        raise ValueError(f"P2_SIDECAR_MISSING: row_ids={missing_sidecar_ids[:10]}")
    for record in train_records:
        row_id = str(record.get("id", ""))
        pair_id = str(record.get("pair_id", ""))
        sidecar = sidecar_by_id[row_id]
        if sidecar.get("row_id") != row_id or sidecar.get("pair_id") != pair_id:
            raise ValueError(f"P3W3_SIDECAR_ROW_OR_PAIR_IDENTITY_MISMATCH: row_id={row_id}")
        if sidecar.get("split") != "train":
            raise ValueError(f"P3W3_SIDECAR_SPLIT_MISMATCH: row_id={row_id}")
        exact_binary(sidecar, "frame_compatible_label", row_id, prefix="P2_SIDECAR")
    return trainer._p2_resolve_canonical_lineage_for_split(records=train_records, sidecar_by_id=sidecar_by_id, split="train")


def apply_production_reason_supervision(train_records: list[dict[str, Any]], sidecar_by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    annotated = copy.deepcopy(train_records)
    supervision_audit = trainer._p2_prepare_reason_supervision_train_only(
        train_records=annotated,
        train_inputs={},
        train_source_labels=["clean_main"] * len(annotated),
        sidecar_by_id=sidecar_by_id,
        require_min_counts=False,
        min_train_count=50,
        device=torch.device("cpu"),
    )
    return annotated, supervision_audit, {
        "production_helper": "scripts/train_controlled_v6b_minimal.py::_p2_prepare_reason_supervision_train_only",
        "device": "cpu",
        "require_min_counts": False,
        "authoritative_record_fields": [
            "p2_primary_reason",
            "p2_primary_reason_target_4",
            "p2_reason_supervision_eligible",
            "p2_reason_exclusion_codes",
            "p2_frame_applicable",
            "p2_predicate_applicable",
            "p2_sufficiency_applicable",
            "p2_polarity_applicable",
            "p2_polarity_target_2",
            "intervention_contract_pass",
            "generator_integrity_status",
        ],
    }


def _identity_valid(state: dict[str, Any]) -> bool:
    return bool(
        state["sidecar_present"]
        and state["sidecar_row_id_matches"]
        and state["sidecar_pair_id_matches"]
        and state["sidecar_split"] == "train"
        and state["canonical_lineage_valid"]
    )


def resolve_generator_evidence_proof_authority() -> dict[str, Any]:
    return dict(GENERATOR_EVIDENCE_PROOF_CONTRACT)


def classify_generator_evidence(state: dict[str, Any], proof_authority: dict[str, Any] | None = None) -> tuple[str, list[str], dict[str, Any]]:
    proof = dict(proof_authority or resolve_generator_evidence_proof_authority())
    proof_meta = {
        "proof_contract_available": bool(proof.get("available")),
        "proof_contract_source": proof.get("source"),
        "proof_contract_clause": proof.get("clause"),
    }
    codes = set(state["ordered_exclusion_codes"])
    fields = state["raw_generator_integrity_fields"]
    component_statuses = {
        field: fields.get(field)
        for field in trainer.P2_GENERATOR_COMPONENT_STATUS_FIELDS
    }
    clean_statuses = set(trainer.P2_GENERATOR_CLEAN_STATUSES)
    supported_statuses = clean_statuses | {"FAIL"}
    unsupported_or_missing = [
        field for field, value in component_statuses.items()
        if type(value) is not str or value not in supported_statuses
    ]
    if unsupported_or_missing:
        return AMBIGUOUS_INTEGRITY_EVIDENCE, ["missing or unsupported generator component status provenance: " + ",".join(unsupported_or_missing)], proof_meta
    if state["normalized_generator_integrity_status"] == "CLEAN" and not codes:
        return CLEAN_GENERATOR_EVIDENCE, ["production helper marked row eligible and generator components normalize CLEAN"], proof_meta
    reasons: list[str] = []
    if any(code in codes for code in PRODUCTION_INDEPENDENT_EXCLUSION_CODES):
        reasons.append("production independent axis/label/source exclusion is present")
    independent_fail_fields = [
        field for field in (
            "schema_status",
            "dataset_source_status",
            "grammar_status",
            "canonical_status",
            "polarity_contamination_status",
            "time_swap_status",
        )
        if component_statuses.get(field) == "FAIL"
    ]
    if independent_fail_fields:
        reasons.append("independent generator component failure fields: " + ",".join(independent_fail_fields))
    if reasons:
        return INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE, reasons, proof_meta
    if not _identity_valid(state):
        return AMBIGUOUS_INTEGRITY_EVIDENCE, ["source, sidecar, split, or canonical identity is not fully valid"], proof_meta
    if state["final_label"] != "REFUTE" or state["polarity_label"] != "REFUTE" or state["intervention_type"] != "polarity_flip":
        return AMBIGUOUS_INTEGRITY_EVIDENCE, ["row is not an intended REFUTE polarity_flip target"], proof_meta
    if state["raw_primary_failure_type"] != "polarity":
        return AMBIGUOUS_INTEGRITY_EVIDENCE, ["primary_failure_type does not assert intended polarity intervention"], proof_meta
    if component_statuses.get("intervention_contract_status") == "FAIL" and not proof_meta["proof_contract_available"]:
        return AMBIGUOUS_INTEGRITY_EVIDENCE, ["repository-backed polarity-only mismatch proof contract is unavailable"], proof_meta
    if not proof_meta["proof_contract_available"]:
        return AMBIGUOUS_INTEGRITY_EVIDENCE, ["repository-backed polarity-only mismatch proof contract is unavailable"], proof_meta
    changed = set(state["audit_changed_axes"])
    expected = set(state["audit_expected_axes"])
    preserved = set(state["audit_preserved_axes"])
    required_preserved = set(proof.get("required_preserved_axes") or [])
    accepted_reason_codes = set(proof.get("accepted_reason_codes") or [])
    if changed != {"polarity"} or expected != {"polarity"}:
        return AMBIGUOUS_INTEGRITY_EVIDENCE, ["audit axis provenance does not prove polarity is the only expected changed axis"], proof_meta
    if required_preserved and not required_preserved.issubset(preserved):
        return AMBIGUOUS_INTEGRITY_EVIDENCE, ["audit preserved axes do not satisfy repository-backed proof authority"], proof_meta
    reason_codes = set(state["sidecar_reason_codes"])
    if not accepted_reason_codes or not reason_codes or not reason_codes.issubset(accepted_reason_codes):
        return AMBIGUOUS_INTEGRITY_EVIDENCE, ["reason_codes do not match repository-backed polarity-only proof authority"], proof_meta
    nonpass_fields = {field for field, value in component_statuses.items() if value not in clean_statuses}
    allowed_nonpass_fields = set(proof.get("allowed_nonpass_fields") or [])
    if nonpass_fields and not nonpass_fields.issubset(allowed_nonpass_fields):
        return AMBIGUOUS_INTEGRITY_EVIDENCE, ["non-PASS component fields exceed repository-backed proof authority"], proof_meta
    return PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY, ["repository-backed proof authority proves expected polarity-only intervention mismatch"], proof_meta


def _build_row_state(
    record: dict[str, Any],
    *,
    split: str,
    source_label: str,
    sidecar_by_id: dict[str, dict[str, Any]],
    expected_canonical_by_pair: dict[str, str],
) -> dict[str, Any]:
    if split != "train" or source_label != "clean_main":
        raise ValueError("P3W3_ANALYZER_ONLY_SUPPORTS_AUTHORITATIVE_CLEAN_TRAIN_ROWS")
    row_id = str(record.get("id", ""))
    pair_id = str(record.get("pair_id", ""))
    frame, predicate, sufficiency, raw_primary, final_label, intervention_type = _validate_source_metadata(record)
    sidecar = sidecar_by_id.get(row_id)
    sidecar_present = sidecar is not None
    raw_generator_fields = {field: sidecar.get(field) for field in trainer.P2_GENERATOR_COMPONENT_STATUS_FIELDS} if sidecar else {}
    canonical_row_id = sidecar.get("canonical_row_id") if sidecar else None
    state = {
        "record": record,
        "row_id": row_id,
        "pair_id": pair_id,
        "canonical_row_id": canonical_row_id,
        "sidecar_row_id_matches": bool(sidecar and sidecar.get("row_id") == row_id),
        "sidecar_pair_id_matches": bool(sidecar and sidecar.get("pair_id") == pair_id),
        "canonical_lineage_valid": bool(sidecar and canonical_row_id == expected_canonical_by_pair.get(pair_id)),
        "intervention_type": intervention_type,
        "raw_primary_failure_type": raw_primary,
        "derived_primary_reason": record["p2_primary_reason"],
        "expected_primary_reason": trainer._p2_expected_primary_from_record(record),
        "frame_compatible_label": frame,
        "predicate_covered_label": predicate,
        "sufficiency_label": sufficiency,
        "final_label": final_label,
        "polarity_label": canonical_polarity(record.get("polarity_label")),
        "sidecar_present": sidecar_present,
        "sidecar_split": sidecar.get("split") if sidecar else None,
        "intervention_contract_status": sidecar.get("intervention_contract_status") if sidecar else None,
        "intervention_contract_pass": bool(record["intervention_contract_pass"]),
        "normalized_generator_integrity_status": record["generator_integrity_status"],
        "raw_generator_integrity_fields": raw_generator_fields,
        "sidecar_reason_codes": list(sidecar.get("reason_codes", [])) if sidecar else [],
        "audit_changed_axes": list(sidecar.get("audit_changed_axes", [])) if sidecar else [],
        "audit_expected_axes": list(sidecar.get("audit_expected_axes", [])) if sidecar else [],
        "audit_preserved_axes": list(sidecar.get("audit_preserved_axes", [])) if sidecar else [],
        "generator_source_path": sidecar.get("generator_source_path") if sidecar else None,
        "generator_source_sha256": sidecar.get("generator_source_sha256") if sidecar else None,
        "integrity_builder_sha256": sidecar.get("integrity_builder_sha256") if sidecar else None,
        "stage182a_report_sha256": sidecar.get("stage182a_report_sha256") if sidecar else None,
        "stage184a_report_sha256": sidecar.get("stage184a_report_sha256") if sidecar else None,
        "p2_reason_supervision_eligible": bool(record["p2_reason_supervision_eligible"]),
        "p2_primary_reason_target_4": int(record["p2_primary_reason_target_4"]),
        "p2_frame_applicable": bool(record["p2_frame_applicable"]),
        "p2_predicate_applicable": bool(record["p2_predicate_applicable"]),
        "p2_sufficiency_applicable": bool(record["p2_sufficiency_applicable"]),
        "p2_polarity_applicable": bool(record["p2_polarity_applicable"]),
        "p2_polarity_target_2": int(record["p2_polarity_target_2"]),
        "ordered_exclusion_codes": list(record["p2_reason_exclusion_codes"]),
    }
    state["exclusion_code_combination_key"] = "+".join(state["ordered_exclusion_codes"]) if state["ordered_exclusion_codes"] else "NONE"
    state["axis_authorized_directional"] = frame == 1 and predicate == 1 and sufficiency == 1 and final_label in trainer.P2_DIRECTIONAL_LABELS
    state["raw_polarity_candidate_contract"] = raw_primary in {"none", "polarity"} or intervention_type in {"none", "polarity_flip"}
    evidence_class, evidence_reasons, proof_meta = classify_generator_evidence(state)
    state["generator_evidence_class"] = evidence_class
    state["generator_evidence_reasons"] = evidence_reasons
    state.update(proof_meta)
    return state

def build_row_states(
    *,
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]] | None,
    sidecar_by_id: dict[str, dict[str, Any]],
    train_source_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    if train_source_labels is not None and train_source_labels != ["clean_main"] * len(train_records):
        raise ValueError("P3W3_TRAIN_SOURCE_LABELS_MUST_BE_CLEAN_MAIN")
    for record in train_records:
        _validate_source_metadata(record)
    if dev_records is not None:
        validate_split_contract(train_records, dev_records, len(train_records), ordered_train_identity_hash(train_records))
    expected_canonical_by_pair = validate_train_sidecar_identity_and_lineage(train_records, sidecar_by_id)
    annotated, _supervision_audit, _authority = apply_production_reason_supervision(train_records, sidecar_by_id)
    return [
        _build_row_state(
            record,
            split="train",
            source_label="clean_main",
            sidecar_by_id=sidecar_by_id,
            expected_canonical_by_pair=expected_canonical_by_pair,
        )
        for record in annotated
    ]

def _row_id_set(rows: list[dict[str, Any]]) -> set[str]:
    return {state["row_id"] for state in rows}


def _stage_counts(states: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    u1 = [s for s in states if s["final_label"] in {"SUPPORT", "REFUTE"}]
    u2 = [s for s in u1 if s["axis_authorized_directional"]]
    u3 = [s for s in u2 if s["raw_polarity_candidate_contract"]]
    u4 = [s for s in u3 if _identity_valid(s)]
    u5 = [s for s in u4 if "P2_POLARITY_INTERVENTION_CONTRACT_FAIL" not in s["ordered_exclusion_codes"]]
    u6 = [s for s in u5 if s["normalized_generator_integrity_status"] == "CLEAN"]
    u7 = [s for s in u6 if s["p2_reason_supervision_eligible"] and s["derived_primary_reason"] == "AUTHORIZED"]
    u8 = [s for s in u7 if s["p2_polarity_applicable"] and s["p2_polarity_target_2"] in {0, 1}]
    ordered = [
        ("U1_directional_final_label_rows", u1),
        ("U2_axis_authorized_directional_rows", u2),
        ("U3_raw_polarity_candidates", u3),
        ("U4_sidecar_identity_lineage_valid_polarity_candidates", u4),
        ("U5_contract_valid_polarity_candidates", u5),
        ("U6_integrity_valid_polarity_candidates", u6),
        ("U7_fully_p2_eligible_primary_authorized_rows", u7),
        ("U8_final_polarity_applicable_rows", u8),
    ]
    previous_ids: set[str] | None = None
    previous_counts: dict[str, int] | None = None
    result: dict[str, dict[str, Any]] = {}
    for name, rows in ordered:
        ids = _row_id_set(rows)
        counts = {
            "SUPPORT": sum(1 for s in rows if s["final_label"] == "SUPPORT"),
            "REFUTE": sum(1 for s in rows if s["final_label"] == "REFUTE"),
        }
        if previous_ids is not None and not ids.issubset(previous_ids):
            raise ValueError(f"P3W3_UNIVERSE_NOT_NESTED: {name}")
        if previous_counts is not None:
            for label in ("SUPPORT", "REFUTE"):
                if counts[label] > previous_counts[label]:
                    raise ValueError(f"P3W3_NEGATIVE_ATTRITION_FORBIDDEN: {name} {label}")
        result[name] = {**counts, "row_ids": [s["row_id"] for s in rows]}
        previous_ids = ids
        previous_counts = counts
    return result

def _attrition_funnel(stages: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_support: int | None = None
    prev_refute: int | None = None
    for name, payload in stages.items():
        support = int(payload["SUPPORT"])
        refute = int(payload["REFUTE"])
        support_attrition = None if prev_support is None else prev_support - support
        refute_attrition = None if prev_refute is None else prev_refute - refute
        if (support_attrition is not None and support_attrition < 0) or (refute_attrition is not None and refute_attrition < 0):
            raise ValueError(f"P3W3_NEGATIVE_ATTRITION_FORBIDDEN: {name}")
        rows.append({
            "stage": name,
            "support_rows_remaining": support,
            "refute_rows_remaining": refute,
            "support_attrition_from_previous_stage": support_attrition,
            "refute_attrition_from_previous_stage": refute_attrition,
        })
        prev_support = support
        prev_refute = refute
    return rows


def _binary_counts(states: list[dict[str, Any]]) -> dict[str, dict[int, int]]:
    cohorts = {"frame": [], "predicate": [], "sufficiency": [], "polarity": []}
    for state in states:
        if state["p2_reason_supervision_eligible"]:
            cohorts["frame"].append(state["frame_compatible_label"])
            if state["frame_compatible_label"] == 1:
                cohorts["predicate"].append(state["predicate_covered_label"])
            if state["frame_compatible_label"] == 1 and state["predicate_covered_label"] == 1:
                cohorts["sufficiency"].append(state["sufficiency_label"])
            if state["p2_polarity_applicable"]:
                cohorts["polarity"].append(1 if state["final_label"] == "SUPPORT" else 0)
    return {name: {0: values.count(0), 1: values.count(1)} for name, values in cohorts.items()}


def _c5_admissible(state: dict[str, Any]) -> bool:
    return bool(
        state["final_label"] == "REFUTE"
        and state["polarity_label"] == "REFUTE"
        and state["intervention_type"] == "polarity_flip"
        and state["axis_authorized_directional"]
        and _identity_valid(state)
        and not any(code in state["ordered_exclusion_codes"] for code in PRODUCTION_INDEPENDENT_EXCLUSION_CODES)
        and state["generator_evidence_class"] == PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY
    )


def _counterfactual_result(states: list[dict[str, Any]], name: str, ignored: set[str]) -> dict[str, Any]:
    eligible_states: list[dict[str, Any]] = []
    newly: list[dict[str, Any]] = []
    for state in states:
        if name == "C5":
            eligible = not state["ordered_exclusion_codes"] or _c5_admissible(state)
        else:
            residual = [code for code in state["ordered_exclusion_codes"] if code not in ignored]
            eligible = not residual
        if eligible and state["axis_authorized_directional"]:
            eligible_states.append(state)
            if state["ordered_exclusion_codes"]:
                newly.append(state)
    combo = Counter(s["exclusion_code_combination_key"] for s in newly)
    support = sum(1 for s in eligible_states if s["final_label"] == "SUPPORT")
    refute = sum(1 for s in eligible_states if s["final_label"] == "REFUTE")
    return {
        "eligible_support_count": support,
        "eligible_refute_count": refute,
        "polarity_binary_readiness": support > 0 and refute > 0,
        "newly_admitted_row_count": len(newly),
        "newly_admitted_exact_exclusion_combinations": _counter_dict(combo),
        "newly_admitted_rows": [
            {
                "row_id": s["row_id"],
                "generator_evidence_class": s["generator_evidence_class"],
                "generator_evidence_reasons": s["generator_evidence_reasons"],
                "ordered_exclusion_codes": s["ordered_exclusion_codes"],
                "proof_contract_available": s["proof_contract_available"],
                "proof_contract_source": s["proof_contract_source"],
                "proof_contract_clause": s["proof_contract_clause"],
            }
            for s in newly
        ],
        "ignored_exclusion_codes": sorted(ignored),
    }

def counterfactual_results(states: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results = {
        name: _counterfactual_result(states, name, ignored)
        for name, ignored in COUNTERFACTUAL_IGNORES.items()
    }
    results["C5"] = _counterfactual_result(states, "C5", set())
    return results


def pair_level_canonical_comparison(states: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {s["row_id"]: s for s in states}
    refutes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        if state["final_label"] == "REFUTE":
            refutes_by_pair[state["pair_id"]].append(state)
    counters = Counter()
    counters["refute_row_count"] = sum(len(rows) for rows in refutes_by_pair.values())
    counters["unique_refute_pair_count"] = len(refutes_by_pair)
    counters["multi_refute_row_pair_count"] = sum(1 for rows in refutes_by_pair.values() if len(rows) > 1)
    for _pair_id, rows in refutes_by_pair.items():
        canonical_ids = {str(row["canonical_row_id"]) for row in rows}
        canonical_rows = [by_id.get(row_id) for row_id in canonical_ids]
        if len(canonical_ids) != 1 or any(row is None for row in canonical_rows):
            counters["pairs_with_canonical_lineage_inconsistency"] += 1
            continue
        canonical = canonical_rows[0]
        if canonical["final_label"] == "SUPPORT":
            counters["pairs_with_canonical_SUPPORT_row"] += 1
        if canonical["final_label"] == "SUPPORT" and canonical["p2_reason_supervision_eligible"]:
            counters["pairs_with_eligible_canonical_SUPPORT_row"] += 1
        if canonical["final_label"] == "SUPPORT" and canonical["generator_evidence_class"] == CLEAN_GENERATOR_EVIDENCE:
            counters["pairs_with_generator_evidence_clean_canonical_SUPPORT_row"] += 1
        if any(not row["p2_reason_supervision_eligible"] for row in rows) and canonical["p2_reason_supervision_eligible"]:
            counters["pairs_where_REFUTE_is_ineligible_but_canonical_SUPPORT_is_eligible"] += 1
        if any(not row["p2_reason_supervision_eligible"] for row in rows) and not canonical["p2_reason_supervision_eligible"]:
            counters["pairs_where_both_rows_are_ineligible"] += 1
        if any(row["intervention_contract_status"] != canonical["intervention_contract_status"] for row in rows):
            counters["pairs_with_intervention_contract_disagreement"] += 1
        if any(row["generator_evidence_class"] == INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE for row in rows):
            counters["pairs_with_independent_REFUTE_defect_evidence"] += 1
        if any(row["generator_evidence_class"] == AMBIGUOUS_INTEGRITY_EVIDENCE for row in rows):
            counters["pairs_with_ambiguous_REFUTE_integrity_evidence"] += 1
        if any(row["generator_evidence_class"] == PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY for row in rows):
            counters["pairs_with_proven_expected_polarity_mismatch_only"] += 1
    for key in (
        "refute_row_count",
        "unique_refute_pair_count",
        "multi_refute_row_pair_count",
        "pairs_with_canonical_SUPPORT_row",
        "pairs_with_eligible_canonical_SUPPORT_row",
        "pairs_with_generator_evidence_clean_canonical_SUPPORT_row",
        "pairs_where_REFUTE_is_ineligible_but_canonical_SUPPORT_is_eligible",
        "pairs_where_both_rows_are_ineligible",
        "pairs_with_canonical_lineage_inconsistency",
        "pairs_with_intervention_contract_disagreement",
        "pairs_with_independent_REFUTE_defect_evidence",
        "pairs_with_ambiguous_REFUTE_integrity_evidence",
        "pairs_with_proven_expected_polarity_mismatch_only",
    ):
        counters.setdefault(key, 0)
    return _counter_dict(counters)


def decide(states: list[dict[str, Any]], counterfactuals: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    axis_authorized_refutes = [
        s for s in states
        if s["final_label"] == "REFUTE" and s["axis_authorized_directional"]
    ]
    eligible_refute_count = sum(1 for s in axis_authorized_refutes if s["p2_polarity_applicable"] and s["p2_polarity_target_2"] == 0)
    missing_refutes = [s for s in axis_authorized_refutes if not (s["p2_polarity_applicable"] and s["p2_polarity_target_2"] == 0)]
    criteria_base = {
        "axis_authorized_refute_count": len(axis_authorized_refutes),
        "eligible_refute_count": eligible_refute_count,
        "missing_axis_authorized_refute_count": len(missing_refutes),
    }
    if not axis_authorized_refutes:
        return "P3W3_NEW_REFUTE_AUTHORITY_REQUIRED", {**criteria_base, "reason": "no axis-authorized REFUTE rows exist"}
    if not missing_refutes and eligible_refute_count > 0:
        return "P3W3_AUDIT_BLOCKED", {**criteria_base, "reason": "polarity supervision blocker is not reproduced"}
    classes = Counter(s["generator_evidence_class"] for s in missing_refutes)
    if classes.get(CLEAN_GENERATOR_EVIDENCE, 0):
        return "P3W3_AUDIT_BLOCKED", {
            **criteria_base,
            "generator_evidence_class_counts": _counter_dict(classes),
            "required_remediation_classes": [],
            "mixed_remediation_required": False,
            "reason": "clean missing REFUTE state is inconsistent with production eligibility",
        }
    remediation_by_evidence = {
        PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY: "NARROW_CONTRACT_REPAIR",
        AMBIGUOUS_INTEGRITY_EVIDENCE: "INTEGRITY_REANNOTATION",
        INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE: "NEW_REFUTE_AUTHORITY",
    }
    remediation_classes = sorted({remediation_by_evidence.get(s["generator_evidence_class"], "UNKNOWN_REMEDIATION") for s in missing_refutes})
    criteria = {
        **criteria_base,
        "generator_evidence_class_counts": _counter_dict(classes),
        "required_remediation_classes": remediation_classes,
        "mixed_remediation_required": len(remediation_classes) > 1,
    }
    if len(remediation_classes) > 1:
        return "P3W3_MIXED_REMEDIATION_REQUIRED", criteria
    if remediation_classes == ["NARROW_CONTRACT_REPAIR"]:
        return "P3W3_EXISTING_AUTHORITY_RECOVERABLE_BY_NARROW_CONTRACT_REPAIR", criteria
    if remediation_classes == ["INTEGRITY_REANNOTATION"]:
        return "P3W3_EXISTING_AUTHORITY_REQUIRES_INTEGRITY_REANNOTATION", criteria
    if remediation_classes == ["NEW_REFUTE_AUTHORITY"]:
        return "P3W3_NEW_REFUTE_AUTHORITY_REQUIRED", criteria
    return "P3W3_AUDIT_BLOCKED", {**criteria, "reason": "unknown remediation class in missing REFUTE subset"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def row_reconstruction_counts(states: list[dict[str, Any]]) -> dict[str, Any]:
    primary_counts = Counter()
    exclusion_counts = Counter()
    for state in states:
        primary_counts[state["derived_primary_reason"]] += int(state["p2_reason_supervision_eligible"])
        for code in state["ordered_exclusion_codes"]:
            exclusion_counts[code] += 1
    return {
        "train_reason_counts": {name: int(primary_counts.get(name, 0)) for name in trainer.P2_REASON_CLASS_ORDER},
        "target_class_counts": {"train_applicable_binary": _binary_counts(states)},
        "train_exclusion_counts": _counter_dict(exclusion_counts),
    }


def verify_production_audit_reconstruction(states: list[dict[str, Any]], production_supervision_audit: dict[str, Any] | None) -> dict[str, Any]:
    row_counts = row_reconstruction_counts(states)
    if production_supervision_audit is None:
        return {"checked": False, "row_derived": row_counts}
    expected = {
        "train_reason_counts": production_supervision_audit.get("train_reason_counts", {}),
        "target_class_counts": {"train_applicable_binary": (production_supervision_audit.get("target_class_counts") or {}).get("train_applicable_binary", {})},
        "train_exclusion_counts": production_supervision_audit.get("train_exclusion_counts", {}),
    }
    if _jsonable(row_counts) != _jsonable(expected):
        raise ValueError("P3W3_PRODUCTION_AUDIT_RECONSTRUCTION_MISMATCH")
    return {"checked": True, "row_derived": row_counts, "production": expected}


def analyze_states(states: list[dict[str, Any]], *, execution_commit: str, dataset: dict[str, Any], sidecar: dict[str, Any], split: dict[str, Any], production_authority: dict[str, Any] | None = None, production_supervision_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    stages = _stage_counts(states)
    exclusion_marginals = Counter()
    combinations = Counter()
    final_intervention = Counter()
    final_primary = Counter()
    refute_intervention_primary = Counter()
    generator_status = Counter()
    raw_generator_fields = Counter()
    contract_status = Counter()
    evidence_classes = Counter()
    evidence_class_by_combo = Counter()
    refute_records: list[dict[str, Any]] = []
    for state in states:
        _inc(final_intervention, state["final_label"], state["intervention_type"])
        _inc(final_primary, state["final_label"], state["raw_primary_failure_type"])
        if state["final_label"] == "REFUTE":
            _inc(refute_intervention_primary, state["intervention_type"], state["raw_primary_failure_type"])
            generator_status[state["normalized_generator_integrity_status"]] += 1
            raw_key = json.dumps(state["raw_generator_integrity_fields"], sort_keys=True, separators=(",", ":"))
            raw_generator_fields[raw_key] += 1
            contract_status[state["intervention_contract_status"]] += 1
            evidence_classes[state["generator_evidence_class"]] += 1
            _inc(evidence_class_by_combo, state["exclusion_code_combination_key"], state["generator_evidence_class"])
            for code in state["ordered_exclusion_codes"]:
                exclusion_marginals[code] += 1
            combinations[state["exclusion_code_combination_key"]] += 1
            refute_records.append(refute_export_record(state, states))
    for code in EXCLUSION_CODES:
        exclusion_marginals.setdefault(code, 0)
    reconstruction = verify_production_audit_reconstruction(states, production_supervision_audit)
    authoritative_counts = reconstruction.get("production") or reconstruction["row_derived"]
    cfs = counterfactual_results(states)
    decision, criteria = decide(states, cfs)
    eligible_support = sum(1 for s in states if s["p2_polarity_applicable"] and s["p2_polarity_target_2"] == 1)
    eligible_refute = sum(1 for s in states if s["p2_polarity_applicable"] and s["p2_polarity_target_2"] == 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "P3W3_AUDIT_EXECUTION_COMPLETE_PENDING_RESULT_REVIEW",
        "audit_execution_completed": True,
        "result_static_review_completed": False,
        "production_behavior_modified": False,
        "polarity_supervision_released": False,
        "A1_A3_released": False,
        "decision": decision,
        "execution_commit": execution_commit,
        "dataset": dataset,
        "sidecar": sidecar,
        "sidecar_semantic_interpretation_audit": sidecar,
        "production_helper_authority": production_authority or {},
        "production_supervision_audit": production_supervision_audit or {},
        "row_reconstruction_consistency": reconstruction,
        "split_seed": split["split_seed"],
        "dev_ratio": split["dev_ratio"],
        "train_dev_counts": {"train": split["train_rows"], "dev": split["dev_rows"]},
        "ordered_train_identity": split["ordered_train_identity"],
        "execution_isolation_audit": dict(EXECUTION_ISOLATION_AUDIT),
        "candidate_universe_counts": {
            "U0_all_train_rows": {"count": len(states), "row_ids": [s["row_id"] for s in states]},
            **stages,
        },
        "attrition_funnel": _attrition_funnel(stages),
        "primary_reason_counts": authoritative_counts["train_reason_counts"],
        "local_binary_cohort_counts": authoritative_counts["target_class_counts"]["train_applicable_binary"],
        "row_derived_primary_reason_counts": reconstruction["row_derived"]["train_reason_counts"],
        "row_derived_local_binary_cohort_counts": reconstruction["row_derived"]["target_class_counts"]["train_applicable_binary"],
        "row_derived_train_exclusion_counts": reconstruction["row_derived"]["train_exclusion_counts"],
        "final_label_overview": {
            "all_train_SUPPORT": sum(1 for s in states if s["final_label"] == "SUPPORT"),
            "all_train_REFUTE": sum(1 for s in states if s["final_label"] == "REFUTE"),
            "axis_authorized_SUPPORT": stages["U2_axis_authorized_directional_rows"]["SUPPORT"],
            "axis_authorized_REFUTE": stages["U2_axis_authorized_directional_rows"]["REFUTE"],
            "eligible_SUPPORT_polarity_targets": eligible_support,
            "eligible_REFUTE_polarity_targets": eligible_refute,
        },
        "SUPPORT_eligibility_funnel": {stage: payload["SUPPORT"] for stage, payload in stages.items()},
        "REFUTE_eligibility_funnel": {stage: payload["REFUTE"] for stage, payload in stages.items()},
        "REFUTE_intervention_cross_tabs": {
            "final_label_x_intervention_type": _counter_dict(final_intervention),
            "final_label_x_primary_failure_type": _counter_dict(final_primary),
            "REFUTE_x_intervention_type_x_primary_failure_type": _counter_dict(refute_intervention_primary),
        },
        "REFUTE_generator_status_cross_tabs": {
            "normalized_generator_status": _counter_dict(generator_status),
            "raw_generator_status_fields": _counter_dict(raw_generator_fields),
            "intervention_contract_status": _counter_dict(contract_status),
        },
        "REFUTE_exclusion_marginals": _counter_dict(exclusion_marginals),
        "REFUTE_exact_exclusion_combinations": _counter_dict(combinations),
        "generator_evidence_class_counts": _counter_dict(evidence_classes),
        "generator_evidence_class_by_exact_exclusion_combination": _counter_dict(evidence_class_by_combo),
        "pair_level_canonical_comparison": pair_level_canonical_comparison(states),
        "counterfactual_eligibility_results": cfs,
        "polarity_readiness_under_each_counterfactual": {
            name: result["polarity_binary_readiness"] for name, result in cfs.items()
        },
        "decision_criteria_audit": criteria,
        "remaining_blockers": [
            "P2_POLARITY_LOCAL_SUPERVISION_NOT_TRAINING_READY",
            "P3W3_RESULT_STATIC_REVIEW_PENDING",
        ],
        "forbidden_claims_not_made": [
            "P3W3_AUDIT_PASS",
            "P2_POLARITY_SUPERVISION_RESOLVED",
            "P2_INTEGRITY_GATE_DEFECTIVE",
            "P2_REFUTE_ROWS_RECOVERABLE",
            "NEW_REFUTE_DATA_REQUIRED",
            "A1_READY",
            "A2_READY",
            "A3_READY",
            "P3_PASS",
        ],
        "refute_row_count_exported": len(refute_records),
        "refute_rows": refute_records,
    }


def refute_export_record(state: dict[str, Any], states: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {s["row_id"]: s for s in states}
    canonical = by_id.get(str(state["canonical_row_id"]))
    return {
        "row_id": state["row_id"],
        "pair_id": state["pair_id"],
        "canonical_row_id": state["canonical_row_id"],
        "intervention_type": state["intervention_type"],
        "raw_primary_failure_type": state["raw_primary_failure_type"],
        "derived_primary_reason": state["derived_primary_reason"],
        "expected_primary_reason": state["expected_primary_reason"],
        "frame_compatible_label": state["frame_compatible_label"],
        "predicate_covered_label": state["predicate_covered_label"],
        "sufficiency_label": state["sufficiency_label"],
        "final_label": state["final_label"],
        "polarity_label": state["polarity_label"],
        "sidecar_present": state["sidecar_present"],
        "sidecar_split": state["sidecar_split"],
        "intervention_contract_status": state["intervention_contract_status"],
        "normalized_generator_integrity_status": state["normalized_generator_integrity_status"],
        "raw_generator_integrity_fields": state["raw_generator_integrity_fields"],
        "sidecar_reason_codes": state["sidecar_reason_codes"],
        "audit_changed_axes": state["audit_changed_axes"],
        "audit_expected_axes": state["audit_expected_axes"],
        "audit_preserved_axes": state["audit_preserved_axes"],
        "generator_source_path": state["generator_source_path"],
        "generator_source_sha256": state["generator_source_sha256"],
        "integrity_builder_sha256": state["integrity_builder_sha256"],
        "stage182a_report_sha256": state["stage182a_report_sha256"],
        "stage184a_report_sha256": state["stage184a_report_sha256"],
        "generator_evidence_class": state["generator_evidence_class"],
        "generator_evidence_reasons": state["generator_evidence_reasons"],
        "proof_contract_available": state["proof_contract_available"],
        "proof_contract_source": state["proof_contract_source"],
        "proof_contract_clause": state["proof_contract_clause"],
        "p2_reason_supervision_eligible": state["p2_reason_supervision_eligible"],
        "p2_primary_reason_target_4": state["p2_primary_reason_target_4"],
        "p2_polarity_applicable": state["p2_polarity_applicable"],
        "p2_polarity_target_2": state["p2_polarity_target_2"],
        "ordered_exclusion_codes": state["ordered_exclusion_codes"],
        "exclusion_code_combination_key": state["exclusion_code_combination_key"],
        "canonical_counterpart_row_id": None if canonical is None else canonical["row_id"],
        "canonical_counterpart_final_label": None if canonical is None else canonical["final_label"],
        "canonical_counterpart_eligibility": None if canonical is None else canonical["p2_reason_supervision_eligible"],
    }


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def validate_git_authority(execution_commit: str) -> None:
    head = _git_output(["git", "rev-parse", "HEAD"])
    if head != execution_commit:
        raise ValueError("P3W3_EXECUTION_COMMIT_MISMATCH")
    dirty = _git_output(["git", "status", "--porcelain", "--untracked-files=no"])
    if dirty:
        raise ValueError("P3W3_DIRTY_TRACKED_WORKING_TREE")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--controlled-integrity-sidecar-path", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-integrity-sidecar-semantic-sha256", required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--dev-ratio", type=float, required=True)
    parser.add_argument("--expected-train-row-count", type=int, required=True)
    parser.add_argument("--expected-train-row-identity-hash", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-refute-jsonl", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.split_seed != EXPECTED_SPLIT_SEED:
        raise ValueError("P3W3_SPLIT_SEED_MISMATCH")
    if float(args.dev_ratio) != EXPECTED_DEV_RATIO:
        raise ValueError("P3W3_DEV_RATIO_MISMATCH")
    if args.expected_train_row_count != EXPECTED_TRAIN_ROWS:
        raise ValueError("P3W3_EXPECTED_TRAIN_ROW_COUNT_MISMATCH")
    if args.expected_train_row_identity_hash != EXPECTED_TRAIN_IDENTITY:
        raise ValueError("P3W3_EXPECTED_TRAIN_IDENTITY_HASH_MISMATCH")
    if args.expected_data_sha256 != EXPECTED_DATA_SHA256:
        raise ValueError("P3W3_EXPECTED_DATA_SHA256_NOT_AUTHORITATIVE")
    if args.expected_integrity_sidecar_semantic_sha256 != EXPECTED_SIDECAR_SEMANTIC_SHA256:
        raise ValueError("P3W3_EXPECTED_SIDECAR_SHA256_NOT_AUTHORITATIVE")
    validate_git_authority(args.execution_commit)

    data_path = args.data if args.data.is_absolute() else ROOT / args.data
    sidecar_path = (
        args.controlled_integrity_sidecar_path
        if args.controlled_integrity_sidecar_path.is_absolute()
        else ROOT / args.controlled_integrity_sidecar_path
    )
    records = v5.load_jsonl(data_path)
    sidecar_by_id, sidecar_audit = load_reason_sidecar(
        data_path=data_path,
        source_records=records,
        sidecar_path=sidecar_path,
        expected_data_sha256=args.expected_data_sha256,
        expected_semantic_sha256=args.expected_integrity_sidecar_semantic_sha256,
        enforce_authoritative_paths=True,
    )
    train_records, dev_records = v5.split_by_pair_id(records, dev_ratio=args.dev_ratio, seed=args.split_seed)
    identity = validate_split_contract(
        train_records,
        dev_records,
        args.expected_train_row_count,
        args.expected_train_row_identity_hash,
    )
    expected_train_ids = {row_id for row_id, row in sidecar_by_id.items() if row.get("split") == "train"}
    expected_dev_ids = {row_id for row_id, row in sidecar_by_id.items() if row.get("split") == "dev"}
    if {str(row.get("id", "")) for row in train_records} != expected_train_ids:
        raise ValueError("P3W3_TRAIN_SPLIT_SIDECAR_IDENTITY_MISMATCH")
    if {str(row.get("id", "")) for row in dev_records} != expected_dev_ids:
        raise ValueError("P3W3_DEV_SPLIT_SIDECAR_IDENTITY_MISMATCH")
    expected_canonical_by_pair = validate_train_sidecar_identity_and_lineage(train_records, sidecar_by_id)
    production_annotated, production_supervision_audit, production_authority = apply_production_reason_supervision(train_records, sidecar_by_id)
    states = [
        _build_row_state(record, split="train", source_label="clean_main", sidecar_by_id=sidecar_by_id, expected_canonical_by_pair=expected_canonical_by_pair)
        for record in production_annotated
    ]
    summary = analyze_states(
        states,
        execution_commit=args.execution_commit,
        dataset={"path": str(data_path.resolve()), "sha256": args.expected_data_sha256},
        sidecar=sidecar_audit,
        split={
            "split_seed": args.split_seed,
            "dev_ratio": args.dev_ratio,
            "train_rows": len(train_records),
            "dev_rows": len(dev_records),
            "ordered_train_identity": identity,
        },
        production_authority=production_authority,
        production_supervision_audit=production_supervision_audit,
    )
    refute_rows = summary.pop("refute_rows")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_refute_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_refute_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in refute_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
