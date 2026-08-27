import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from scripts import kaggle_cache_integrity as kci


@pytest.fixture
def tmp_path():
    """Use a task-owned writable fixture root instead of the blocked host temp root."""
    root = Path(__file__).parents[1] / ".pre_urp_kaggle_cache_test_tmp"
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            root.rmdir()
        except OSError:
            pass


def make_cache(tmp_path: Path, *, revisions=("rev-a",), with_ref=False):
    hub = tmp_path / "hub"
    family = hub / "models--state-spaces--mamba-130m-hf"
    blob = family / "blobs"
    snapshots = family / "snapshots"
    blob.mkdir(parents=True)
    snapshots.mkdir()
    for revision in revisions:
        snap = snapshots / revision
        snap.mkdir()
        payload = b"shared-bytes"
        content = blob / "blob-shared"
        content.write_bytes(payload)
        (snap / "config.json").write_bytes(b'{"hidden_size": 16}')
        link = snap / "weights.bin"
        try:
            link.symlink_to(content)
        except (OSError, NotImplementedError):
            link.write_bytes(payload)
    if with_ref:
        refs = family / "refs"
        refs.mkdir()
        (refs / "main").write_text(revisions[0], encoding="utf-8")
    return hub, family


def make_manifest(tmp_path, *, revisions=("rev-a",), with_ref=True):
    hub, family = make_cache(tmp_path, revisions=revisions, with_ref=with_ref)
    manifest_path = tmp_path / "manifest.json"
    payload = kci.create_manifest(hub, "state-spaces/mamba-130m-hf", manifest_path)
    return hub, family, manifest_path, payload


def test_create_and_verify_deterministic_with_shared_content(tmp_path):
    hub, _, path, payload = make_manifest(tmp_path)
    assert kci.verify_manifest(hub, path) == payload
    assert payload["summary"]["logical_file_count"] == 2
    assert payload["summary"]["unique_content_object_count"] == 2
    assert payload["summary"]["logical_total_bytes"] == sum(x["size_bytes"] for x in payload["files"])
    second = tmp_path / "second.json"
    payload2 = kci.create_manifest(hub, "state-spaces/mamba-130m-hf", second, revision="rev-a")
    assert payload["files"] == payload2["files"]
    assert path.read_bytes() == path.read_bytes()


def test_corruption_missing_and_extra_fail(tmp_path):
    hub, family, path, _ = make_manifest(tmp_path)
    (family / "snapshots" / "rev-a" / "config.json").unlink()
    with pytest.raises(kci.CacheIntegrityError):
        kci.verify_manifest(hub, path)
    (family / "snapshots" / "rev-a" / "config.json").write_bytes(b"corrupt")
    with pytest.raises(kci.CacheIntegrityError):
        kci.verify_manifest(hub, path)
    (family / "snapshots" / "rev-a" / "extra.txt").write_bytes(b"x")
    with pytest.raises(kci.CacheIntegrityError):
        kci.verify_manifest(hub, path)


def test_manifest_rejects_drive_qualified_record(tmp_path):
    hub, _, path, payload = make_manifest(tmp_path)
    payload["files"][0]["logical_path"] = "C:/escape"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(kci.CacheIntegrityError):
        kci.verify_manifest(hub, path)


def test_wrong_revision_and_ambiguous_selection_fail(tmp_path):
    hub, family = make_cache(tmp_path, revisions=("rev-a", "rev-b"))
    with pytest.raises(kci.CacheIntegrityError):
        kci.create_manifest(hub, "state-spaces/mamba-130m-hf", tmp_path / "m.json")
    with pytest.raises(kci.CacheIntegrityError):
        kci.create_manifest(hub, "state-spaces/mamba-130m-hf", tmp_path / "m.json", revision="missing")
    refs = family / "refs"
    refs.mkdir()
    (refs / "main").write_text("rev-b", encoding="utf-8")
    assert kci.create_manifest(hub, "state-spaces/mamba-130m-hf", tmp_path / "m.json")["revision"] == "rev-b"


def test_broken_link_and_escape_fail(tmp_path):
    hub, family = make_cache(tmp_path)
    snap = family / "snapshots" / "rev-a"
    link = snap / "broken.bin"
    try:
        link.symlink_to(family / "blobs" / "missing")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(kci.CacheIntegrityError):
        kci.create_manifest(hub, "state-spaces/mamba-130m-hf", tmp_path / "m.json")
    link.unlink()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link.symlink_to(outside)
    with pytest.raises(kci.CacheIntegrityError):
        kci.create_manifest(hub, "state-spaces/mamba-130m-hf", tmp_path / "m.json")


def test_manifest_structure_errors_and_summary_tamper_fail(tmp_path):
    hub, _, path, payload = make_manifest(tmp_path)
    cases = [
        {},
        {**payload, "marker": "wrong"},
        {**payload, "evidence_boundary": "SCIENTIFIC"},
        {**payload, "schema_version": 99},
        {**payload, "summary": {**payload["summary"], "logical_total_bytes": 0}},
        {**payload, "files": payload["files"] + [payload["files"][0]]},
    ]
    for candidate in cases:
        path.write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises(kci.CacheIntegrityError):
            kci.verify_manifest(hub, path)


def test_create_refuses_overwrite_and_is_read_only(tmp_path):
    hub, family = make_cache(tmp_path)
    path = tmp_path / "manifest.json"
    kci.create_manifest(hub, "state-spaces/mamba-130m-hf", path)
    before = {p: p.read_bytes() for p in family.rglob("*") if p.is_file() and not p.is_symlink()}
    with pytest.raises(kci.CacheIntegrityError):
        kci.create_manifest(hub, "state-spaces/mamba-130m-hf", path)
    after = {p: p.read_bytes() for p in family.rglob("*") if p.is_file() and not p.is_symlink()}
    assert before == after


def test_direct_cli_and_malformed_manifest(tmp_path):
    hub, _, path, _ = make_manifest(tmp_path)
    env = os.environ.copy()
    command = [sys.executable, "scripts/kaggle_cache_integrity.py", "verify", "--hub-cache", str(hub), "--manifest", str(path)]
    result = subprocess.run(command, cwd=Path(__file__).parents[1], text=True, capture_output=True, env=env)
    assert result.returncode == 0
    assert "CACHE_INTEGRITY: PASS" in result.stdout
    path.write_text("{", encoding="utf-8")
    result = subprocess.run(command, cwd=Path(__file__).parents[1], text=True, capture_output=True, env=env)
    assert result.returncode != 0
