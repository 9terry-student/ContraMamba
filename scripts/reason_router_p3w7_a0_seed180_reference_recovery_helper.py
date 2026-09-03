#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


SEED = 180
SPLIT_SEED = 174
SPLIT_POLICY = "fixed_explicit_split_seed"
DEV_RATIO = 0.2
RUN_ID = "p3_seed180_A0"
AUDIT_ID = "p3_seed180_A0_REFERENCE_AUDIT"
SOURCE_EXECUTION_COMMIT = "2737c3c6116ae3766b469801f990e2c45ba9a55e"
RECOVERY_AUTHORITY_COMMIT = "233ed0be080e1d30dd47de2e66136475ec2ede76"
ORIGINAL_AUTHORIZED_WRAPPER_SHA256 = (
    "dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e"
)
RECOVERY_SCHEMA = "contramamba-seed180-a0-provenance-recovery-v1"
EXPECTED_ZIP_SHA256 = "6bbd0e89a5858d7c68b1eecc1cf44911cc415c7411670a94605c56c08e955966"
EXPECTED_RECOVERY_MANIFEST_SHA256 = (
    "69c2202b5cf4eb543a4bfb07a5602bae567dc84216c8929620c3a6b725e879ed"
)
DESTINATION_REL = Path("reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0")
DATASET_REL = Path(
    "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458"
    "/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
)
SIDECAR_REL = Path(
    "reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458"
    "/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"
)
EXPECTED_DATASET_SHA256 = "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
EXPECTED_SIDECAR_SEMANTIC_SHA256 = (
    "0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08"
)
P2_IMPLEMENTATION_TESTED_COMMIT = SOURCE_EXECUTION_COMMIT
VALID_LABELS = {"REFUTE", "NOT_ENTITLED", "SUPPORT"}
LABEL_NORMALIZE = {
    "REFUTE": "REFUTE",
    "REFUTES": "REFUTE",
    "CONTRADICT": "REFUTE",
    "CONTRADICTION": "REFUTE",
    "NOT_ENTITLED": "NOT_ENTITLED",
    "NOT-ENTITLED": "NOT_ENTITLED",
    "NE": "NOT_ENTITLED",
    "UNKNOWN": "NOT_ENTITLED",
    "NEI": "NOT_ENTITLED",
    "SUPPORT": "SUPPORT",
    "SUPPORTS": "SUPPORT",
    "ENTAILMENT": "SUPPORT",
}

SOURCE_ARTIFACTS = [
    {
        "name": "training_report.json",
        "zip": "files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json",
        "size": 306114,
        "sha256": "71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508",
    },
    {
        "name": "clean_dev_predictions.json",
        "zip": "files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json",
        "size": 4838225,
        "sha256": "92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2",
    },
    {
        "name": "training_report_predictions.jsonl",
        "zip": "files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl",
        "size": 3934123,
        "sha256": "e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef",
    },
    {
        "name": "selected_checkpoint.pt",
        "zip": "files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/selected_checkpoint.pt",
        "size": 518269815,
        "sha256": "dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da",
    },
    {
        "name": "run_provenance.json",
        "zip": "files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/run_provenance.json",
        "size": 68429,
        "sha256": "4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b",
    },
]
ARTIFACT_BY_NAME = {item["name"]: item for item in SOURCE_ARTIFACTS}
EXPECTED_ZIP_ENTRIES = ("recovery_manifest.json", *(item["zip"] for item in SOURCE_ARTIFACTS))


class Blocker(RuntimeError):
    pass


def block(message: str) -> None:
    raise Blocker(f"REFERENCE_RECOVERY_BLOCKED: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        block(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json_loads(text: str) -> Any:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                block(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=hook)
    except Blocker:
        raise
    except json.JSONDecodeError as exc:
        block(f"malformed JSON: {exc}")


def strict_json_file(path: Path) -> Any:
    return strict_json_loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def canonical_semantic_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def run_git(root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        block(f"git command failed: {' '.join(args)}: {exc}")
    return result.stdout.strip()


def repo_root() -> Path:
    return Path(run_git(Path.cwd(), ["rev-parse", "--show-toplevel"])).resolve()


def require_runtime_authority(root: Path, expected_commit: str) -> str:
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_commit or "") is not None,
        "recovery execution authority commit must be lowercase 40-hex",
    )
    obj_type = run_git(root, ["cat-file", "-t", expected_commit])
    require(obj_type == "commit", "recovery execution authority object is not a commit")
    head = run_git(root, ["rev-parse", "HEAD"])
    require(head == expected_commit, "current HEAD does not match recovery execution authority commit")
    return expected_commit


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return LABEL_NORMALIZE.get(value.strip().upper())
    return LABEL_NORMALIZE.get(str(value).strip().upper())


def row_identity(record: dict[str, Any], item: dict[str, Any] | None = None) -> str:
    for source in (item or {}, record):
        for key in ("stable_id", "row_id", "source_id", "id"):
            value = source.get(key)
            if value is not None and str(value) != "":
                return str(value)
    return ""


def pair_identity(record: dict[str, Any], item: dict[str, Any] | None = None) -> str:
    for source in (item or {}, record):
        value = source.get("pair_id")
        if value is not None and str(value) != "":
            return str(value)
    return ""


def load_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = strict_json_loads(line)
            require(isinstance(row, dict), f"JSONL row {line_number} must be object")
            rows.append(row)
    require(rows, f"JSONL file has no rows: {path}")
    return rows


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl_strict(path)
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        require(isinstance(row.get("id"), str) and row["id"], f"dataset row {index} missing id")
        require(isinstance(row.get("pair_id"), str) and row["pair_id"], f"dataset row {index} missing pair_id")
        require(row["id"] not in seen, f"dataset duplicate id: {row['id']}")
        seen.add(row["id"])
    return rows


def split_dev_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_ids = sorted({record["pair_id"] for record in records})
    require(len(pair_ids) >= 2, "at least two pair IDs required")
    random.Random(SPLIT_SEED).shuffle(pair_ids)
    dev_count = min(len(pair_ids) - 1, max(1, round(len(pair_ids) * DEV_RATIO)))
    dev_pairs = set(pair_ids[:dev_count])
    return [record for record in records if record["pair_id"] in dev_pairs]


def identity_hash(rows: list[tuple[str, str, str]]) -> str:
    digest = hashlib.sha256()
    for row_id, pair_id, gold in rows:
        digest.update(f"{row_id}\t{pair_id}\t{gold}\n".encode("utf-8"))
    return digest.hexdigest()


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = strict_json_loads(handle.read())
            require(isinstance(payload, list), "prediction JSON must be list or JSONL rows")
            rows = payload
        else:
            rows = [strict_json_loads(line) for line in handle if line.strip()]
    require(all(isinstance(row, dict) for row in rows), "prediction rows must be objects")
    return rows


def semantic_sidecar_sha256(path: Path) -> str:
    rows = load_jsonl_strict(path)
    canonical = [
        {key: row[key] for key in sorted(row) if key != "created_at"}
        for row in rows
    ]
    return hashlib.sha256(canonical_semantic_bytes(canonical)).hexdigest()


def validate_dev_identity(dataset_path: Path, prediction_path: Path) -> dict[str, Any]:
    dev_records = split_dev_records(load_dataset(dataset_path))
    predictions = load_prediction_rows(prediction_path)
    require(len(dev_records) == len(predictions), "authoritative dev count != prediction count")

    source_rows: list[tuple[str, str, str]] = []
    source_ids: set[str] = set()
    for record in dev_records:
        row_id = row_identity(record)
        pair_id = pair_identity(record)
        gold = normalize_label(record.get("gold_label") or record.get("gold_final_label") or record.get("final_label"))
        require(row_id, "source missing row identity")
        require(row_id not in source_ids, f"source duplicate row identity: {row_id}")
        require(pair_id, "source missing pair identity")
        require(gold in VALID_LABELS, f"source invalid gold label: {gold}")
        source_ids.add(row_id)
        source_rows.append((row_id, pair_id, gold))

    prediction_by_row: dict[str, tuple[str, str, str]] = {}
    pred_values: list[str] = []
    gold_values: list[str] = []
    for item in predictions:
        row_id = row_identity({}, item)
        pair_id = pair_identity({}, item)
        gold = normalize_label(item.get("gold_label") or item.get("gold_final_label") or item.get("final_label"))
        pred = normalize_label(item.get("pred_label") or item.get("prediction") or item.get("pred_final_label"))
        require(row_id, "prediction missing row identity")
        require(row_id not in prediction_by_row, f"prediction duplicate row identity: {row_id}")
        require(pair_id, "prediction missing pair identity")
        require(gold in VALID_LABELS, f"prediction invalid gold label: {gold}")
        require(pred in VALID_LABELS, f"prediction invalid class: {pred}")
        prediction_by_row[row_id] = (row_id, pair_id, gold)
        pred_values.append(pred)

    require(set(prediction_by_row) == source_ids, "source/prediction row-ID sets differ")
    joined_rows: list[tuple[str, str, str]] = []
    for row_id, pair_id, gold in source_rows:
        pred_row = prediction_by_row[row_id]
        require(pred_row[1] == pair_id, f"pair mismatch: {row_id}")
        require(pred_row[2] == gold, f"gold mismatch: {row_id}")
        joined_rows.append(pred_row)
        gold_values.append(gold)

    authoritative_hash = identity_hash(source_rows)
    joined_hash = identity_hash(joined_rows)
    require(authoritative_hash == joined_hash, "identity hash mismatch")
    return {
        "authoritative_dev_row_count": len(source_rows),
        "authoritative_dev_row_identity_hash": authoritative_hash,
        "prediction_joined_dev_row_identity_hash": joined_hash,
        "gold_counts": dict(Counter(gold_values)),
        "prediction_counts": dict(Counter(pred_values)),
        "a0_false_entitlement_count": sum(
            gold == "NOT_ENTITLED" and pred in {"REFUTE", "SUPPORT"}
            for gold, pred in zip(gold_values, pred_values)
        ),
        "a0_stable_true_support_count": sum(
            gold == "SUPPORT" and pred == "SUPPORT"
            for gold, pred in zip(gold_values, pred_values)
        ),
        "row_count": len(predictions),
        "unique_row_id_count": len(set(prediction_by_row)),
        "unique_row_pair_count": len({row[1] for row in joined_rows}),
    }


def validate_zip_member_path(name: str, is_dir: bool, mode: int, flag_bits: int) -> str:
    require("\x00" not in name, "ZIP NUL path rejected")
    require("\\" not in name, "ZIP backslash path rejected")
    require(not name.startswith("/"), "ZIP absolute path rejected")
    require(re.match(r"^[A-Za-z]:", name) is None, "ZIP drive path rejected")
    require(not name.endswith("/") and not is_dir, "ZIP directory member rejected")
    pure = PurePosixPath(name)
    require(str(pure) == name, "ZIP malformed/noncanonical path rejected")
    require("." not in pure.parts and ".." not in pure.parts, "ZIP dot/traversal path rejected")
    kind = stat.S_IFMT(mode)
    require(kind in {0, stat.S_IFREG}, "ZIP symlink or special-file member rejected")
    require(flag_bits & 0x1 == 0, "ZIP encrypted member rejected")
    return name


def validate_zip_member_name(info: zipfile.ZipInfo) -> str:
    return validate_zip_member_path(
        info.filename,
        info.is_dir(),
        (info.external_attr >> 16) & 0xFFFF,
        info.flag_bits,
    )


def validate_zip_structure(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = zf.infolist()
    names = [validate_zip_member_name(info) for info in infos]
    require(len(names) == len(set(names)), "ZIP duplicate member rejected")
    lowered = [name.lower() for name in names]
    require(len(lowered) == len(set(lowered)), "ZIP duplicate logical member rejected")
    require(set(names) == set(EXPECTED_ZIP_ENTRIES), "ZIP member set mismatch")
    by_name = dict(zip(names, infos))
    for item in SOURCE_ARTIFACTS:
        require(by_name[item["zip"]].file_size == item["size"], f"ZIP size mismatch: {item['name']}")
    return by_name


def read_zip_text_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    with zf.open(info, "r") as handle:
        try:
            return handle.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            block(f"ZIP text entry is not UTF-8: {info.filename}: {exc}")


def sha256_zip_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with zf.open(info, "r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def validate_manifest(payload: Any) -> None:
    require(isinstance(payload, dict), "manifest root must be object")
    expected = {
        "schema": RECOVERY_SCHEMA,
        "seed": SEED,
        "attempt_disposition": "CONSUMED",
        "execution_status": "completed",
        "original_authorized_wrapper_sha256": ORIGINAL_AUTHORIZED_WRAPPER_SHA256,
        "original_execution_commit": SOURCE_EXECUTION_COMMIT,
        "recovery_authority_commit": RECOVERY_AUTHORITY_COMMIT,
        "source_run_provenance_sha256": ARTIFACT_BY_NAME["run_provenance.json"]["sha256"],
        "source_trainer_git_commit": SOURCE_EXECUTION_COMMIT,
        "standard_cm_wrapper_provenance": "missing/incomplete",
        "scientific_conclusion": "NOT_ESTABLISHED",
    }
    for key, value in expected.items():
        require(payload.get(key) == value, f"manifest binding mismatch: {key}")
    table = payload.get("artifact_files")
    require(isinstance(table, list), "manifest artifact table missing")
    expected_table = {
        item["zip"].removeprefix("files/"): {"size": item["size"], "sha256": item["sha256"]}
        for item in SOURCE_ARTIFACTS
    }
    seen: set[str] = set()
    for row in table:
        require(isinstance(row, dict), "manifest artifact row must be object")
        path = row.get("path")
        require(isinstance(path, str) and path in expected_table and path not in seen, "manifest artifact path mismatch")
        seen.add(path)
        require(row.get("size") == expected_table[path]["size"], f"manifest size mismatch: {path}")
        require(row.get("sha256") == expected_table[path]["sha256"], f"manifest sha mismatch: {path}")
    require(seen == set(expected_table), "manifest artifact table incomplete")


def validate_run_provenance(payload: Any) -> None:
    require(isinstance(payload, dict), "run_provenance root must be object")
    source = payload.get("source_provenance", {})
    require(isinstance(source, dict), "source_provenance must be object")
    require(source.get("git_commit") == SOURCE_EXECUTION_COMMIT, "run provenance execution commit mismatch")
    require(source.get("git_is_dirty") is False, "run provenance dirty flag mismatch")
    parsed = payload.get("parsed_args", {})
    if isinstance(parsed, dict):
        require(parsed.get("seed") in {SEED, str(SEED)}, "run provenance seed mismatch")
        require(parsed.get("split_seed") in {SPLIT_SEED, str(SPLIT_SEED)}, "run provenance split seed mismatch")
        require(parsed.get("reason_router_arm") in {None, "A0"}, "run provenance arm mismatch")
    finalization = payload.get("finalization", {})
    require(isinstance(finalization, dict), "finalization must be object")
    selected = finalization.get("selected_checkpoint", {})
    require(isinstance(selected, dict), "selected checkpoint provenance must be object")
    require(selected.get("sha256") == ARTIFACT_BY_NAME["selected_checkpoint.pt"]["sha256"], "checkpoint provenance sha mismatch")


def validate_zip_source(zip_path: Path) -> dict[str, Any]:
    require(zip_path.exists() and zip_path.is_file() and not zip_path.is_symlink(), "ZIP path is missing or unsafe")
    require(sha256_file(zip_path) == EXPECTED_ZIP_SHA256, "ZIP SHA256 mismatch")
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = validate_zip_structure(zf)
        manifest_text = read_zip_text_entry(zf, infos["recovery_manifest.json"])
        require(sha256_bytes(manifest_text.encode("utf-8")) == EXPECTED_RECOVERY_MANIFEST_SHA256, "recovery_manifest SHA mismatch")
        manifest = strict_json_loads(manifest_text)
        validate_manifest(manifest)
        for item in SOURCE_ARTIFACTS:
            observed_size, observed_sha = sha256_zip_entry(zf, infos[item["zip"]])
            require(observed_size == item["size"], f"artifact size mismatch: {item['name']}")
            require(observed_sha == item["sha256"], f"artifact SHA mismatch: {item['name']}")
        run_provenance = read_zip_text_entry(zf, infos[ARTIFACT_BY_NAME["run_provenance.json"]["zip"]])
        validate_run_provenance(strict_json_loads(run_provenance))
    return manifest


def prevalidate_environment(root: Path, destination: Path, dataset_path: Path, sidecar_path: Path) -> None:
    require(sha256_file(dataset_path) == EXPECTED_DATASET_SHA256, "dataset SHA mismatch")
    require(semantic_sidecar_sha256(sidecar_path) == EXPECTED_SIDECAR_SEMANTIC_SHA256, "sidecar semantic SHA mismatch")
    resolved_destination = destination.resolve()
    require(root.resolve() in (resolved_destination, *resolved_destination.parents), "destination escaped repository")
    if destination.exists():
        require(destination.is_dir() and not destination.is_symlink(), "destination path is unsafe")
    require(not (destination / "A0_REFERENCE_AUDIT.json").exists(), "A0_REFERENCE_AUDIT.json already exists")
    for item in SOURCE_ARTIFACTS:
        target = destination / item["name"]
        if target.exists():
            require(target.is_file() and not target.is_symlink(), f"destination unsafe collision: {item['name']}")
            require(sha256_file(target) == item["sha256"], f"destination nonidentical collision: {item['name']}")


def write_staged_zip_entry(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    staging: Path,
    item: dict[str, Any],
) -> Path:
    path = staging / item["name"]
    digest = hashlib.sha256()
    size = 0
    with path.open("xb") as handle:
        with zf.open(info, "r") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
                handle.write(chunk)
    require(size == item["size"], f"staged size mismatch before write: {item['name']}")
    require(digest.hexdigest() == item["sha256"], f"staged SHA mismatch before write: {item['name']}")
    require(path.stat().st_size == item["size"], f"staged size mismatch after write: {item['name']}")
    require(sha256_file(path) == item["sha256"], f"staged SHA mismatch after write: {item['name']}")
    return path


def publish_exclusive(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except FileExistsError:
        block(f"destination appeared before publish: {target.name}")
    except OSError as exc:
        block(f"atomic exclusive hardlink publication unavailable: {exc}")


def selected_epoch_from_report(report: dict[str, Any]) -> tuple[Any, str | None]:
    single = report.get("runs", {}).get("single", {}) if isinstance(report.get("runs"), dict) else {}
    if isinstance(single, dict) and single.get("best_epoch") is not None:
        return single.get("best_epoch"), "runs.single.best_epoch"
    if report.get("best_epoch") is not None:
        return report.get("best_epoch"), "best_epoch"
    return None, None


def build_audit(
    root: Path,
    destination: Path,
    zip_path: Path,
    dataset_path: Path,
    sidecar_path: Path,
) -> dict[str, Any]:
    report_path = destination / "training_report.json"
    prediction_path = destination / "training_report_predictions.jsonl"
    checkpoint_path = destination / "selected_checkpoint.pt"
    require(
        sha256_file(report_path) == ARTIFACT_BY_NAME["training_report.json"]["sha256"],
        "report reread SHA mismatch",
    )
    require(
        sha256_file(prediction_path) == ARTIFACT_BY_NAME["training_report_predictions.jsonl"]["sha256"],
        "prediction reread SHA mismatch",
    )
    require(
        sha256_file(checkpoint_path) == ARTIFACT_BY_NAME["selected_checkpoint.pt"]["sha256"],
        "checkpoint reread SHA mismatch",
    )
    report = strict_json_file(report_path)
    require(isinstance(report, dict), "training_report root must be object")
    epoch, epoch_source = selected_epoch_from_report(report)
    identity = validate_dev_identity(dataset_path, prediction_path)
    audit = {
        "audit_id": AUDIT_ID,
        "run_id": RUN_ID,
        "seed": SEED,
        "status": "PASS",
        "errors": [],
        "execution_commit": SOURCE_EXECUTION_COMMIT,
        "p2_implementation_tested_commit": P2_IMPLEMENTATION_TESTED_COMMIT,
        "output_dir": str(destination.relative_to(root).as_posix()),
        "reference_prediction_path": str(prediction_path.relative_to(root).as_posix()),
        "prediction_sha256": ARTIFACT_BY_NAME["training_report_predictions.jsonl"]["sha256"],
        "selected_checkpoint_path": str(checkpoint_path.relative_to(root).as_posix()),
        "selected_checkpoint_sha256": ARTIFACT_BY_NAME["selected_checkpoint.pt"]["sha256"],
        "report_path": str(report_path.relative_to(root).as_posix()),
        "report_sha256": ARTIFACT_BY_NAME["training_report.json"]["sha256"],
        "selected_epoch": epoch,
        "selected_epoch_source": epoch_source,
        "data_path": str(dataset_path.relative_to(root).as_posix()),
        "dataset_sha256_expected": EXPECTED_DATASET_SHA256,
        "dataset_sha256_observed": sha256_file(dataset_path),
        "sidecar_path": str(sidecar_path.relative_to(root).as_posix()),
        "sidecar_semantic_sha256_expected": EXPECTED_SIDECAR_SEMANTIC_SHA256,
        "sidecar_semantic_sha256_observed": semantic_sidecar_sha256(sidecar_path),
        "split_seed": SPLIT_SEED,
        "split_policy": SPLIT_POLICY,
        "dev_ratio": DEV_RATIO,
        "source_execution_commit": SOURCE_EXECUTION_COMMIT,
        "recovery_authority_commit": RECOVERY_AUTHORITY_COMMIT,
        "retained_zip_path": str(zip_path),
        "retained_zip_sha256": EXPECTED_ZIP_SHA256,
        "recovery_manifest_sha256": EXPECTED_RECOVERY_MANIFEST_SHA256,
        "standard_cm_wrapper_provenance": "INCOMPLETE",
        "provenance_disposition": "RECOVERY_BRIDGE_WITH_HISTORICAL_STANDARD_CM_WRAPPER_PROVENANCE_INCOMPLETE",
        "recovery_reference_status": "RECOVERY_REFERENCE_AUDIT_PASS",
    }
    audit.update(identity)
    required = [
        "selected_epoch",
        "selected_epoch_source",
        "prediction_sha256",
        "selected_checkpoint_sha256",
        "report_sha256",
        "authoritative_dev_row_identity_hash",
        "prediction_joined_dev_row_identity_hash",
    ]
    for field in required:
        if audit.get(field) is None:
            audit["status"] = "P3_A0_REFERENCE_AUDIT_FAILED"
            audit["errors"].append(f"NULL_REQUIRED_FIELD:{field}")
    require(audit["status"] == "PASS", "audit status would not be PASS")
    return audit


def validate_persisted_audit(payload: Any, expected_audit: dict[str, Any]) -> None:
    require(isinstance(payload, dict), "persisted audit root must be object")
    require(payload.get("status") == "PASS", "persisted audit status is not PASS")
    require(payload == expected_audit, "persisted audit content mismatch")


def validate_published_reference(
    root: Path,
    destination: Path,
    dataset_path: Path,
    sidecar_path: Path,
    expected_audit: dict[str, Any],
) -> None:
    for item in SOURCE_ARTIFACTS:
        target = destination / item["name"]
        require(target.exists() and target.is_file() and not target.is_symlink(), f"published artifact missing: {item['name']}")
        require(sha256_file(target) == item["sha256"], f"published artifact SHA mismatch: {item['name']}")
    persisted = strict_json_file(destination / "A0_REFERENCE_AUDIT.json")
    validate_persisted_audit(persisted, expected_audit)
    require(
        persisted["dataset_sha256_observed"] == sha256_file(dataset_path),
        "persisted dataset identity drift",
    )
    require(
        persisted["sidecar_semantic_sha256_observed"] == semantic_sidecar_sha256(sidecar_path),
        "persisted sidecar identity drift",
    )
    require(
        persisted["prediction_joined_dev_row_identity_hash"]
        == validate_dev_identity(dataset_path, destination / "training_report_predictions.jsonl")[
            "prediction_joined_dev_row_identity_hash"
        ],
        "persisted prediction identity drift",
    )


def materialize_reference(args: argparse.Namespace) -> Path:
    root = repo_root()
    require_runtime_authority(root, args.expected_recovery_execution_authority_commit)
    zip_path = Path(args.zip).expanduser().resolve()
    validate_zip_source(zip_path)
    destination = root / DESTINATION_REL
    dataset_path = root / DATASET_REL
    sidecar_path = root / SIDECAR_REL
    prevalidate_environment(root, destination, dataset_path, sidecar_path)
    destination.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".seed180_reference_recovery_staging.{os.getpid()}"
    try:
        staging.mkdir(exist_ok=False)
    except FileExistsError:
        block("helper staging path already exists")

    try:
        staged: dict[str, Path] = {}
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = validate_zip_structure(zf)
            for item in SOURCE_ARTIFACTS:
                target = destination / item["name"]
                if not target.exists():
                    staged[item["name"]] = write_staged_zip_entry(zf, infos[item["zip"]], staging, item)
        for name, staged_path in staged.items():
            publish_exclusive(staged_path, destination / name)
        for item in SOURCE_ARTIFACTS:
            target = destination / item["name"]
            require(sha256_file(target) == item["sha256"], f"published artifact SHA mismatch: {item['name']}")
        audit = build_audit(root, destination, zip_path, dataset_path, sidecar_path)
        audit_tmp = staging / "A0_REFERENCE_AUDIT.json"
        with audit_tmp.open("xb") as handle:
            handle.write(canonical_json_bytes(audit))
        validate_persisted_audit(strict_json_file(audit_tmp), audit)
        publish_exclusive(audit_tmp, destination / "A0_REFERENCE_AUDIT.json")
        validate_published_reference(root, destination, dataset_path, sidecar_path, audit)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination / "A0_REFERENCE_AUDIT.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3-W7 A0 seed180 retained-artifact reference recovery helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize-reference")
    materialize.add_argument("--zip", required=True)
    materialize.add_argument("--expected-recovery-execution-authority-commit", required=True)
    materialize.set_defaults(func=materialize_reference)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.func(args)
    except Blocker as exc:
        print(str(exc), file=sys.stderr)
        return 64
    print(str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
