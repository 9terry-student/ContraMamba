from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.train_controlled_v6b_minimal as trainer


def _source_row(row_id: str, pair_id: str, *, frame: int = 1, predicate: int = 1, sufficiency: int = 1) -> dict:
    return {
        "id": row_id,
        "pair_id": pair_id,
        "claim": f"claim {row_id}",
        "evidence": f"evidence {row_id}",
        "final_label": "SUPPORT" if frame and predicate and sufficiency else "NOT_ENTITLED",
        "frame_compatible_label": frame,
        "predicate_covered_label": predicate,
        "sufficiency_label": sufficiency,
        "polarity_label": "SUPPORT" if frame and predicate and sufficiency else "NONE",
        "primary_failure_type": (
            "frame" if not frame else "predicate" if not predicate else "sufficiency" if not sufficiency else "none"
        ),
        "intervention_type": "none" if row_id.endswith("__none") else "predicate_swap",
    }


def _sidecar_row(
    source: dict,
    *,
    split: str,
    integrity_status: str,
    p2_eligible: bool,
    positive_margin: bool,
    reason_codes: list[str] | None = None,
) -> dict:
    return {
        "namespace": "P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR",
        "schema_version": trainer._P4X_SIDECAR_SCHEMA_VERSION,
        "source_order_index": 0,
        "row_id": source["id"],
        "split": split,
        "pair_id": source["pair_id"],
        "canonical_row_id": source["id"] if source["intervention_type"] == "none" else f"{source['pair_id']}__none",
        "canonical_status": "PASS",
        "intervention_contract_status": "PASS",
        "integrity_status": integrity_status,
        "schema_status": "PASS",
        "dataset_source_status": "PASS",
        "grammar_status": "PASS",
        "polarity_contamination_status": "PASS",
        "time_swap_status": "PASS",
        "reason_codes": [] if reason_codes is None else reason_codes,
        "source_dataset_path": trainer._STAGE187_AUTHORITATIVE_DATA.as_posix(),
        "source_dataset_sha256": trainer._STAGE187_DATASET_SHA256,
        "source_dataset_semantic_sha256": trainer._P4X_SOURCE_DATASET_SEMANTIC_SHA256,
        "frame_compatible_label": source["frame_compatible_label"],
        "intervention_type": source["intervention_type"],
        "eligible_for_positive_margin": positive_margin,
        "p2_reason_supervision_eligible": p2_eligible,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _install_p4x_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, mutate=None) -> tuple[Path, Path, Path, list[dict], list[dict]]:
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    data = tmp_path / "source.jsonl"
    sidecar = canonical_dir / "sidecar.jsonl"
    provenance_path = canonical_dir / "provenance.json"
    source_rows = [
        _source_row("p1__none", "p1"),
        _source_row("p2__none", "p2", frame=0),
        _source_row("p3__none", "p3", predicate=0),
    ]
    monkeypatch.setattr(trainer, "_STAGE187_EXPECTED_SIDECAR_ROWS", 3)
    monkeypatch.setattr(trainer, "_STAGE187_AUTHORITATIVE_DATA", data)
    monkeypatch.setattr(trainer, "_P4X_CANONICAL_DIR", canonical_dir)
    monkeypatch.setattr(trainer, "_STAGE187_AUTHORITATIVE_SIDECAR", sidecar)
    monkeypatch.setattr(trainer, "_P4X_CANONICAL_PROVENANCE", provenance_path)
    source_semantic = trainer._p4x_source_dataset_semantic_sha256(source_rows)
    monkeypatch.setattr(trainer, "_P4X_SOURCE_DATASET_SEMANTIC_SHA256", source_semantic)
    sidecar_rows = [
        _sidecar_row(source_rows[0], split="train", integrity_status="ELIGIBLE", p2_eligible=True, positive_margin=True),
        _sidecar_row(source_rows[1], split="train", integrity_status="INELIGIBLE", p2_eligible=False, positive_margin=False, reason_codes=["P2_GENERATOR_STATUS_DEFECT"]),
        _sidecar_row(source_rows[2], split="dev", integrity_status="UNRESOLVED", p2_eligible=False, positive_margin=False, reason_codes=["P2_INTEGRITY_SOURCE_REQUIRED"]),
    ]
    if mutate is not None:
        mutate(source_rows, sidecar_rows)
    _write_jsonl(data, source_rows)
    monkeypatch.setattr(trainer, "_STAGE187_DATASET_SHA256", trainer._stage187_file_sha256(data))
    for row in sidecar_rows:
        row["source_dataset_sha256"] = trainer._STAGE187_DATASET_SHA256
        row["source_dataset_semantic_sha256"] = trainer._P4X_SOURCE_DATASET_SEMANTIC_SHA256
    _write_jsonl(sidecar, sidecar_rows)
    sidecar_semantic = trainer._stage187_semantic_sidecar_sha256(sidecar_rows)
    monkeypatch.setattr(trainer, "_STAGE187_SIDECAR_SEMANTIC_SHA256", sidecar_semantic)
    monkeypatch.setattr(trainer, "_P4X_SIDECAR_PHYSICAL_SHA256", trainer._stage187_file_sha256(sidecar))
    monkeypatch.setattr(trainer, "_P4X_EXPECTED_REASON_ELIGIBLE_ROWS", 1)
    monkeypatch.setattr(trainer, "_P4X_EXPECTED_REASON_INELIGIBLE_ROWS", 2)
    monkeypatch.setattr(trainer, "_P4X_EXPECTED_INTEGRITY_COUNTS", {"ELIGIBLE": 1, "INELIGIBLE": 1, "UNRESOLVED": 1})
    monkeypatch.setattr(trainer, "_STAGE187_EXPECTED_ELIGIBLE_ROWS", 1)
    monkeypatch.setattr(trainer, "_P4X_EXPECTED_POSITIVE_MARGIN_INELIGIBLE_ROWS", 2)
    provenance = {
        "schema_version": trainer._P4X_PROVENANCE_SCHEMA_VERSION,
        "sidecar_schema_version": trainer._P4X_SIDECAR_SCHEMA_VERSION,
        "p4l_authority_commit": trainer._P4X_P4L_AUTHORITY_COMMIT,
        "builder_source_commit": trainer._P4X_BUILDER_SOURCE_COMMIT,
        "row_count": trainer._STAGE187_EXPECTED_SIDECAR_ROWS,
        "source_dataset_path": data.as_posix(),
        "source_dataset_sha256": trainer._STAGE187_DATASET_SHA256,
        "source_dataset_semantic_sha256": trainer._P4X_SOURCE_DATASET_SEMANTIC_SHA256,
        "sidecar_path": sidecar.as_posix(),
        "sidecar_physical_sha256": trainer._P4X_SIDECAR_PHYSICAL_SHA256,
        "sidecar_semantic_sha256": trainer._STAGE187_SIDECAR_SEMANTIC_SHA256,
        "provenance_path": provenance_path.as_posix(),
        "training_admission_released": False,
        "implementation_authorized": True,
        "artifact_materialization_authorized_by_p4l": False,
        "a0_execution_authorized": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "kaggle_authorized": False,
        "gpu_authorized": False,
    }
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(trainer, "_P4X_PROVENANCE_PHYSICAL_SHA256", trainer._stage187_file_sha256(provenance_path))
    return data, sidecar, provenance_path, source_rows, sidecar_rows


def _load(data: Path, sidecar: Path, source_rows: list[dict]) -> tuple[dict[str, dict], dict]:
    return trainer._p2_load_reason_integrity_sidecar(
        data_path=data,
        source_records=source_rows,
        sidecar_path=sidecar,
        expected_semantic_sha256=trainer._STAGE187_SIDECAR_SEMANTIC_SHA256,
    )


def test_p4x_canonical_binding_success_metadata_and_join(monkeypatch, tmp_path):
    data, sidecar, _provenance, source_rows, _sidecar_rows = _install_p4x_fixture(monkeypatch, tmp_path)
    by_id, audit = _load(data, sidecar, source_rows)
    assert list(by_id) == [row["id"] for row in source_rows]
    assert audit["source"] == "P4-L canonical current-lineage effective integrity sidecar"
    assert audit["stable_join_key"] == {"source": "id", "sidecar": "row_id"}
    assert audit["count_reconciliation"]["p2_reason_supervision_eligible_true"] == 1
    metadata = trainer._p2_checkpoint_metadata_from_args(
        type("Args", (), {
            "reason_router_arm": "A3",
            "expected_integrity_sidecar_semantic_sha256": trainer._STAGE187_SIDECAR_SEMANTIC_SHA256,
        })()
    )
    assert metadata["integrity_sidecar_source"].startswith("P4-L canonical")
    assert metadata["data_semantic_sha256"] == trainer._P4X_SOURCE_DATASET_SEMANTIC_SHA256


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("sidecar_physical", "P4X_SIDECAR_PHYSICAL_SHA_MISMATCH"),
        ("provenance_physical", "P4X_PROVENANCE_PHYSICAL_SHA_MISMATCH"),
        ("source_physical", "P4X_SOURCE_PHYSICAL_SHA_MISMATCH"),
        ("sidecar_semantic", "P4X_SIDECAR_SEMANTIC_SHA_MISMATCH"),
        ("source_semantic", "P4X_SOURCE_SEMANTIC_SHA_MISMATCH"),
    ],
)
def test_p4x_hash_mismatches_fail_closed(monkeypatch, tmp_path, field, expected_error):
    data, sidecar, _provenance, source_rows, _sidecar_rows = _install_p4x_fixture(monkeypatch, tmp_path)
    if field == "sidecar_physical":
        monkeypatch.setattr(trainer, "_P4X_SIDECAR_PHYSICAL_SHA256", "0" * 64)
    elif field == "provenance_physical":
        monkeypatch.setattr(trainer, "_P4X_PROVENANCE_PHYSICAL_SHA256", "0" * 64)
    elif field == "source_physical":
        monkeypatch.setattr(trainer, "_STAGE187_DATASET_SHA256", "0" * 64)
    elif field == "sidecar_semantic":
        monkeypatch.setattr(trainer, "_STAGE187_SIDECAR_SEMANTIC_SHA256", "0" * 64)
    else:
        monkeypatch.setattr(trainer, "_P4X_SOURCE_DATASET_SEMANTIC_SHA256", "0" * 64)
    with pytest.raises((ValueError, FileNotFoundError), match=expected_error):
        _load(data, sidecar, source_rows)


@pytest.mark.parametrize(
    ("remove", "expected_error"),
    [("sidecar", "missing sidecar"), ("provenance", "missing provenance")],
)
def test_p4x_missing_artifacts_fail_closed(monkeypatch, tmp_path, remove, expected_error):
    data, sidecar, provenance, source_rows, _sidecar_rows = _install_p4x_fixture(monkeypatch, tmp_path)
    (sidecar if remove == "sidecar" else provenance).unlink()
    with pytest.raises(FileNotFoundError, match=expected_error):
        _load(data, sidecar, source_rows)


def test_p4x_wrong_canonical_path_fails_closed(monkeypatch, tmp_path):
    data, sidecar, _provenance, source_rows, _sidecar_rows = _install_p4x_fixture(monkeypatch, tmp_path)
    wrong = tmp_path / "wrong.jsonl"
    wrong.write_bytes(sidecar.read_bytes())
    with pytest.raises(ValueError, match="wrong canonical sidecar path"):
        _load(data, wrong, source_rows)


def test_p4x_symlink_rejection_where_supported(monkeypatch, tmp_path):
    data, sidecar, _provenance, source_rows, _sidecar_rows = _install_p4x_fixture(monkeypatch, tmp_path)
    link = tmp_path / "canonical-link.jsonl"
    try:
        link.symlink_to(sidecar)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unsupported here: {exc}")
    monkeypatch.setattr(trainer, "_STAGE187_AUTHORITATIVE_SIDECAR", link)
    with pytest.raises(ValueError, match="contains a symlink"):
        _load(data, link, source_rows)


@pytest.mark.parametrize(
    ("mutate_provenance", "expected_error"),
    [
        (lambda p: p.update(schema_version="bad"), "schema_version"),
        (lambda p: p.update(sidecar_schema_version="bad"), "sidecar_schema_version"),
        (lambda p: p.update(training_authorized=True), "AUTHORITY_OVERCLAIM"),
        (lambda p: p.update(kaggle_authorized="false"), "BOOLEAN_MALFORMED"),
    ],
)
def test_p4x_provenance_schema_and_authority_fail_closed(monkeypatch, tmp_path, mutate_provenance, expected_error):
    data, sidecar, provenance_path, source_rows, _sidecar_rows = _install_p4x_fixture(monkeypatch, tmp_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    mutate_provenance(provenance)
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(trainer, "_P4X_PROVENANCE_PHYSICAL_SHA256", trainer._stage187_file_sha256(provenance_path))
    with pytest.raises(ValueError, match=expected_error):
        _load(data, sidecar, source_rows)


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (lambda s, c: s.__setitem__(1, {**s[1], "id": s[0]["id"]}), "P4X_DUPLICATE_SOURCE_ID"),
        (lambda s, c: c.__setitem__(1, {**c[1], "row_id": c[0]["row_id"]}), "P4X_DUPLICATE_SIDECAR_ROW_ID"),
        (lambda s, c: c.__setitem__(1, {**c[1], "row_id": "missing"}), "P4X_STABLE_JOIN_KEYSET_MISMATCH"),
        (lambda s, c: c.reverse(), "P4X_SOURCE_ORDER_DEFENSE_FAILED"),
    ],
)
def test_p4x_join_failures(monkeypatch, tmp_path, mutate, expected_error):
    data, sidecar, _provenance, source_rows, _sidecar_rows = _install_p4x_fixture(monkeypatch, tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match=expected_error):
        _load(data, sidecar, source_rows)


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (lambda s, c: c[0].pop("row_id"), "SIDECAR_ROW_ID_INVALID"),
        (lambda s, c: c[0].update(schema_version="bad"), "SIDECAR_SCHEMA_VERSION_MISMATCH"),
        (lambda s, c: c[0].update(frame_compatible_label=True), "SIDECAR_EXACT_BINARY_INVALID"),
        (lambda s, c: c[0].update(p2_reason_supervision_eligible=1), "BOOLEAN_MALFORMED"),
        (lambda s, c: c[0].update(reason_codes=["Z", "A"]), "REASON_CODES_NOT_SORTED_UNIQUE"),
        (lambda s, c: c[0].update(integrity_status="BAD"), "INTEGRITY_STATUS_INVALID"),
    ],
)
def test_p4x_malformed_fields_fail_closed(monkeypatch, tmp_path, mutate, expected_error):
    data, sidecar, _provenance, source_rows, _sidecar_rows = _install_p4x_fixture(monkeypatch, tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match=expected_error):
        _load(data, sidecar, source_rows)


def test_p4x_source_exact_binary_field_typing_fail_closed(monkeypatch, tmp_path):
    data, sidecar, _provenance, source_rows, _sidecar_rows = _install_p4x_fixture(monkeypatch, tmp_path)
    source_rows[0]["frame_compatible_label"] = True
    with pytest.raises(ValueError, match="SOURCE_EXACT_BINARY_INVALID"):
        _load(data, sidecar, source_rows)


def test_p4x_reason_loader_and_positive_margin_share_canonical_gate(monkeypatch, tmp_path):
    data, sidecar, _provenance, source_rows, _sidecar_rows = _install_p4x_fixture(monkeypatch, tmp_path)
    eligibility, split_by_id, audit = trainer._stage187_load_integrity_sidecar(
        data_path=data,
        source_records=source_rows,
        sidecar_path=sidecar,
        expected_semantic_sha256=trainer._STAGE187_SIDECAR_SEMANTIC_SHA256,
    )
    assert eligibility == {"p1__none": True, "p2__none": False, "p3__none": False}
    assert split_by_id == {"p1__none": "train", "p2__none": "train", "p3__none": "dev"}
    assert audit["eligible_rows"] == 1
    assert audit["fail_closed_pretraining"] is True
