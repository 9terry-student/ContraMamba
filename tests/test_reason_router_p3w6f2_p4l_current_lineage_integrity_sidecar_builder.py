import errno
import json
import os
import random
import subprocess
from pathlib import Path

import pytest

from scripts import build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar as builder


class FakeRenameAt2:
    def __init__(self, *, result=0, err=0, action=None):
        self.result = result
        self.err = err
        self.action = action
        self.calls = []

    def __call__(self, olddirfd, oldpath, newdirfd, newpath, flags):
        self.calls.append((olddirfd, oldpath, newdirfd, newpath, flags))
        if self.action is not None:
            self.action(os.fsdecode(oldpath), os.fsdecode(newpath))
        if self.result != 0:
            builder.ctypes.set_errno(self.err)
        return self.result


class FakeLibc:
    def __init__(self, renameat2):
        self.renameat2 = renameat2


def payloads():
    return {
        builder.SIDECAR_NAME: b'{"row_id":"attempt"}\n',
        builder.PROVENANCE_NAME: b'{"attempt":true}\n',
    }


def force_linux(monkeypatch):
    monkeypatch.setattr(builder, "running_on_windows", lambda: False)
    monkeypatch.setattr(builder, "running_on_linux", lambda: True)


def force_windows(monkeypatch):
    monkeypatch.setattr(builder, "running_on_windows", lambda: True)
    monkeypatch.setattr(builder, "running_on_linux", lambda: False)


def row(row_id, pair_id, intervention_type="none", frame=1, predicate=1, sufficiency=1):
    return {
        "id": row_id,
        "pair_id": pair_id,
        "claim": "claim",
        "evidence": "evidence",
        "final_label": "SUPPORT" if frame and predicate and sufficiency else "NOT_ENTITLED",
        "frame_compatible_label": frame,
        "predicate_covered_label": predicate,
        "sufficiency_label": sufficiency,
        "polarity_label": "SUPPORT" if frame and predicate and sufficiency else "NONE",
        "primary_failure_type": (
            "frame" if not frame else "predicate" if not predicate else "sufficiency" if not sufficiency else "none"
        ),
        "intervention_type": intervention_type,
    }


def test_deterministic_split_is_pair_level_and_seeded():
    rows = [row(f"p{i}__none", f"p{i}") for i in range(10)]
    observed = builder.deterministic_pair_split(rows, seed=174, dev_ratio=0.2)
    pair_ids = sorted({item["pair_id"] for item in rows})
    shuffled = list(pair_ids)
    random.Random(174).shuffle(shuffled)
    expected_dev = set(shuffled[:2])
    assert observed == {pair_id: "dev" if pair_id in expected_dev else "train" for pair_id in pair_ids}


def test_lineage_modes_are_explicit_and_historical_default_is_preserved():
    historical = builder.lineage_config()
    revised = builder.lineage_config(builder.REVISED_LINEAGE_MODE)
    assert historical.mode == builder.HISTORICAL_LINEAGE_MODE
    assert historical.split_seed == 174
    assert historical.dev_ratio == 0.2
    assert historical.sidecar_name == builder.SIDECAR_NAME
    assert revised.split_seed == 8192
    assert revised.dev_ratio == 0.2
    assert revised.p4l_authority_commit == "ff181f565cefa0a28280c084246862286daf1f2d"
    assert revised.split_authority_commit == "b4fbb5666d796161f95ae23612ce2448c25063ee"
    assert revised.sidecar_name == "p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl"
    assert revised.provenance_name == "p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json"
    with pytest.raises(builder.BuildBlocked, match="LINEAGE_MODE_UNSUPPORTED"):
        builder.lineage_config("seed8192")


def test_revised_output_path_is_distinct_and_binds_full_commits(tmp_path):
    commit = "a" * 40
    historical = builder.canonical_output_dir(tmp_path, commit)
    revised = builder.canonical_output_dir(tmp_path, commit, builder.lineage_config(builder.REVISED_LINEAGE_MODE))
    assert historical.name == f"reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_{commit}"
    assert revised.name == f"reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_{commit}"
    assert revised != historical


def test_revised_split_identities_and_row_identities_match_authority():
    repo_root = Path(__file__).resolve().parents[1]
    config = builder.lineage_config(builder.REVISED_LINEAGE_MODE)
    source = repo_root / builder.SOURCE_DATASET_PATH
    rows = builder.validate_source_dataset(repo_root, source, config)
    split = builder.deterministic_pair_split(rows, seed=config.split_seed, dev_ratio=config.dev_ratio)
    builder.validate_split_identities(rows, split, config)
    assert sum(value == "train" for value in split.values()) == 240
    assert sum(value == "dev" for value in split.values()) == 60
    assert sum(split[row["pair_id"]] == "train" for row in rows) == 2880
    assert sum(split[row["pair_id"]] == "dev" for row in rows) == 720


def test_revised_dataset_accepts_git_lf_identity_and_consumed_working_tree_semantics(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    relative = "frozen/source.jsonl"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes((repo_root / builder.SOURCE_DATASET_PATH).read_bytes())
    monkeypatch.setattr(builder, "SOURCE_DATASET_PATH", relative)
    monkeypatch.setattr(builder, "tracked_git_blob_sha256", lambda root, observed: builder.SOURCE_DATASET_SHA256)

    rows = builder.validate_source_dataset(tmp_path, path, builder.lineage_config(builder.REVISED_LINEAGE_MODE))

    assert len(rows) == builder.EXPECTED_ROW_COUNT


def test_revised_dataset_rejects_semantically_modified_working_tree_despite_valid_git_lf_blob(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    relative = "frozen/source.jsonl"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes((repo_root / builder.SOURCE_DATASET_PATH).read_bytes())
    monkeypatch.setattr(builder, "SOURCE_DATASET_PATH", relative)
    monkeypatch.setattr(builder, "tracked_git_blob_sha256", lambda root, observed: builder.SOURCE_DATASET_SHA256)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    first_row = json.loads(lines[0])
    first_row["claim"] = f"mutated {first_row['claim']}"
    path.write_text(json.dumps(first_row) + "\n" + "".join(lines[1:]), encoding="utf-8", newline="\n")

    with pytest.raises(builder.BuildBlocked, match="SOURCE_DATASET_SEMANTIC_SHA_MISMATCH"):
        builder.validate_source_dataset(tmp_path, path, builder.lineage_config(builder.REVISED_LINEAGE_MODE))


def git(repo_root, *args):
    return subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True)


def frozen_git_repo(tmp_path, relative="frozen/input.jsonl", payload=b'{"source":"head"}\n', checkout_crlf=False):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "test")
    if checkout_crlf:
        git(tmp_path, "config", "core.autocrlf", "true")
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    git(tmp_path, "add", relative)
    git(tmp_path, "commit", "-m", "frozen input")
    return path, builder.sha256_bytes(payload)


def test_revised_canonical_reader_consumes_clean_crlf_checkout_blob_bytes(tmp_path):
    relative = "frozen/input.jsonl"
    path, expected = frozen_git_repo(tmp_path, relative, checkout_crlf=True)
    path.write_bytes(b'{"source":"head"}\r\n')

    assert path.read_bytes() == b'{"source":"head"}\r\n'
    assert subprocess.run(["git", "diff", "--quiet", "--", relative], cwd=tmp_path).returncode == 0
    canonical = builder.canonical_frozen_head_bytes(tmp_path, relative, expected)

    assert canonical == b'{"source":"head"}\n'
    assert builder.read_jsonl_bytes(canonical, relative) == [{"source": "head"}]


@pytest.mark.parametrize(
    ("relative", "staged", "reason"),
    [
        ("frozen/p4b_rows.jsonl", False, "FROZEN_INPUT_WORKTREE_DIRTY"),
        ("frozen/p4b_rows.jsonl", True, "FROZEN_INPUT_INDEX_DIRTY"),
        ("frozen/p4b_summary.json", False, "FROZEN_INPUT_WORKTREE_DIRTY"),
        ("frozen/p4b_provenance.json", False, "FROZEN_INPUT_WORKTREE_DIRTY"),
        ("frozen/stage185_source.py", False, "FROZEN_INPUT_WORKTREE_DIRTY"),
    ],
)
def test_revised_canonical_reader_fails_closed_for_dirty_bridge_paths(tmp_path, relative, staged, reason):
    path, expected = frozen_git_repo(tmp_path, relative)
    path.write_bytes(b"modified\n")
    if staged:
        git(tmp_path, "add", relative)

    with pytest.raises(builder.BuildBlocked, match=f"{reason}:{relative}"):
        builder.canonical_frozen_head_bytes(tmp_path, relative, expected)


def test_revised_canonical_reader_rejects_head_blob_hash_mismatch(tmp_path):
    _, expected = frozen_git_repo(tmp_path)
    with pytest.raises(builder.BuildBlocked, match="FROZEN_INPUT_GIT_BLOB_SHA256_MISMATCH"):
        builder.canonical_frozen_head_bytes(tmp_path, "frozen/input.jsonl", "0" * 64)


def test_real_revised_build_completes_in_memory_without_canonical_output():
    repo_root = Path(__file__).resolve().parents[1]
    config = builder.lineage_config(builder.REVISED_LINEAGE_MODE)
    output_dir = builder.canonical_output_dir(repo_root, "a" * 40, config)
    assert not output_dir.exists()

    rows, provenance, sidecar_payload, provenance_payload = builder.build_sidecar_artifacts(
        repo_root=repo_root,
        builder_commit="a" * 40,
        created_at="2026-09-07T00:00:00Z",
        lineage_mode=builder.REVISED_LINEAGE_MODE,
    )

    assert len(rows) == 3600
    assert provenance["p4l_authority_commit"] == config.p4l_authority_commit
    assert provenance["split_authority_commit"] == config.split_authority_commit
    assert sidecar_payload and provenance_payload
    assert not output_dir.exists()


def test_revised_provenance_binds_lineage_without_future_artifact_hashes(tmp_path):
    config = builder.lineage_config(builder.REVISED_LINEAGE_MODE)
    provenance = builder.build_provenance(
        builder_commit="b" * 40,
        builder_source_sha256="c" * 64,
        output_dir=Path("reports/revised"),
        sidecar_physical_sha256="d" * 64,
        sidecar_semantic_sha256="e" * 64,
        config=config,
    )
    assert provenance["lineage_mode"] == builder.REVISED_LINEAGE_MODE
    assert provenance["split_rule"]["shuffle_seed"] == 8192
    assert provenance["split_rule"]["dev_ratio"] == 0.2
    assert provenance["provenance_physical_sha256_self_certified"] is False
    assert provenance["builder_source_commit"] == "b" * 40
    assert "revised_p4l_provenance_physical_sha256" not in provenance


def test_canonical_mapping_requires_unique_same_pair_none_self_anchor():
    rows = [
        row("p1__none", "p1", "none"),
        row("p1__paraphrase", "p1", "paraphrase"),
        row("p2__none", "p2", "none"),
    ]
    split = {"p1": "train", "p2": "dev"}
    canonical = builder.canonical_row_ids(rows, split)
    assert canonical["p1__none"] == "p1__none"
    assert canonical["p1__paraphrase"] == "p1__none"
    builder.validate_canonical_lineage(rows, split, canonical)

    duplicate_none = rows + [row("p1__none2", "p1", "none")]
    with pytest.raises(builder.BuildBlocked):
        builder.canonical_row_ids(duplicate_none, split)


def test_semantic_sidecar_hash_excludes_created_at_only():
    rows_a = [{"row_id": "a", "created_at": "one", "status": "PASS", "hash_field": "kept"}]
    rows_b = [{"row_id": "a", "created_at": "two", "status": "PASS", "hash_field": "kept"}]
    rows_c = [{"row_id": "a", "created_at": "one", "status": "FAIL", "hash_field": "kept"}]
    assert builder.semantic_sidecar_sha256(rows_a) == builder.semantic_sidecar_sha256(rows_b)
    assert builder.semantic_sidecar_sha256(rows_a) != builder.semantic_sidecar_sha256(rows_c)


def test_exact_binary_rejects_bool_and_non_binary_integer():
    assert builder.exact_binary({"x": 1}, "x", "r1") == 1
    with pytest.raises(builder.BuildBlocked):
        builder.exact_binary({"x": True}, "x", "r1")
    with pytest.raises(builder.BuildBlocked):
        builder.exact_binary({"x": 2}, "x", "r1")


def test_duplicate_row_id_rejected():
    rows = [row("dup", "p1"), row("dup", "p2")]
    with pytest.raises(builder.BuildBlocked):
        builder.validate_source_rows(rows)


def test_fail_closed_invalid_source_identity(monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row("p1__none", "p1")) + "\n", encoding="utf-8")
    monkeypatch.setattr(builder, "SOURCE_DATASET_PATH", source.name)
    monkeypatch.setattr(builder, "SOURCE_DATASET_SHA256", "wrong")
    with pytest.raises(builder.BuildBlocked):
        builder.validate_source_dataset(tmp_path, source)


def test_reason_derivation_order_and_expected_primary():
    assert builder.primary_reason_from_axes(0, 0, 0) == "FRAME"
    assert builder.primary_reason_from_axes(1, 0, 0) == "PREDICATE"
    assert builder.primary_reason_from_axes(1, 1, 0) == "SUFFICIENCY"
    assert builder.primary_reason_from_axes(1, 1, 1) == "AUTHORIZED"
    assert builder.expected_primary_from_record({"primary_failure_type": "polarity"}) == "AUTHORIZED"


def test_positive_margin_eligibility_contract():
    sidecar = {
        "integrity_status": "ELIGIBLE",
        "split": "train",
        "frame_compatible_label": 1,
        "time_swap_status": "PASS",
        "dataset_source_status": "PASS",
    }
    assert builder.positive_margin_eligible(sidecar) is True
    assert builder.positive_margin_eligible({**sidecar, "split": "dev"}) is False
    assert builder.positive_margin_eligible({**sidecar, "frame_compatible_label": 0}) is False


def test_compact_jsonl_serialization_lf_no_bom_and_final_newline():
    payload = builder.compact_jsonl_bytes([{"b": 2, "a": 1}])
    assert payload == b'{"a":1,"b":2}\n'
    assert payload.endswith(b"\n")
    assert not payload.startswith(b"\xef\xbb\xbf")


def test_linux_finalize_publishes_whole_directory_with_renameat2_noreplace(monkeypatch, tmp_path):
    force_linux(monkeypatch)
    output_dir = tmp_path / "canonical_output"

    def publish(oldpath, newpath):
        os.rename(oldpath, newpath)

    fake_renameat2 = FakeRenameAt2(action=publish)
    monkeypatch.setattr(builder, "load_libc", lambda: FakeLibc(fake_renameat2))

    assert builder.finalize_payloads_atomic(output_dir, payloads()) == "PUBLISHED"

    assert sorted(item.name for item in output_dir.iterdir()) == sorted(builder.EXPECTED_OUTPUT_NAMES)
    assert (output_dir / builder.SIDECAR_NAME).read_bytes() == b'{"row_id":"attempt"}\n'
    assert (output_dir / builder.PROVENANCE_NAME).read_bytes() == b'{"attempt":true}\n'
    assert not list(tmp_path.glob(".canonical_output.p4l-staging-*"))
    assert len(fake_renameat2.calls) == 1
    olddirfd, oldpath, newdirfd, newpath, flags = fake_renameat2.calls[0]
    assert olddirfd == builder.AT_FDCWD
    assert newdirfd == builder.AT_FDCWD
    assert flags == builder.RENAME_NOREPLACE
    assert os.fsdecode(oldpath).startswith(str(tmp_path / ".canonical_output.p4l-staging-"))
    assert newpath == os.fsencode(output_dir)


def test_linux_renameat2_signature_uses_abi_constants_and_filesystem_bytes(monkeypatch, tmp_path):
    force_linux(monkeypatch)
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "canonical_output"
    staging_dir.mkdir()
    fake_renameat2 = FakeRenameAt2(action=lambda oldpath, newpath: os.rename(oldpath, newpath))
    monkeypatch.setattr(builder, "load_libc", lambda: FakeLibc(fake_renameat2))

    builder.atomic_publish_directory_noreplace(staging_dir, output_dir)

    assert fake_renameat2.calls == [
        (
            builder.AT_FDCWD,
            os.fsencode(staging_dir),
            builder.AT_FDCWD,
            os.fsencode(output_dir),
            builder.RENAME_NOREPLACE,
        )
    ]


def test_linux_race_collision_at_renameat2_boundary_preserves_foreign_destination(monkeypatch, tmp_path):
    force_linux(monkeypatch)
    output_dir = tmp_path / "canonical_output"

    def create_foreign_destination(_oldpath, newpath):
        destination = tmp_path / Path(newpath).name
        destination.mkdir()
        (destination / "foreign.txt").write_text("do not touch\n", encoding="utf-8")

    fake_renameat2 = FakeRenameAt2(result=-1, err=errno.EEXIST, action=create_foreign_destination)
    monkeypatch.setattr(builder, "load_libc", lambda: FakeLibc(fake_renameat2))

    with pytest.raises(builder.BuildBlocked) as excinfo:
        builder.finalize_payloads_atomic(output_dir, payloads())

    assert str(excinfo.value) == "P4L_OUTPUT_PATH_PREEXISTING"
    assert output_dir.is_dir()
    assert sorted(item.name for item in output_dir.iterdir()) == ["foreign.txt"]
    assert (output_dir / "foreign.txt").read_text(encoding="utf-8") == "do not touch\n"
    assert not (output_dir / builder.SIDECAR_NAME).exists()
    assert not (output_dir / builder.PROVENANCE_NAME).exists()
    assert not list(tmp_path.glob(".canonical_output.p4l-staging-*"))


def test_linux_unsupported_renameat2_symbol_fails_closed_without_fallback(monkeypatch, tmp_path):
    force_linux(monkeypatch)
    output_dir = tmp_path / "canonical_output"

    class LibcWithoutRenameAt2:
        pass

    monkeypatch.setattr(builder, "load_libc", lambda: LibcWithoutRenameAt2())

    with pytest.raises(builder.BuildBlocked) as excinfo:
        builder.finalize_payloads_atomic(output_dir, payloads())

    assert str(excinfo.value) == "P4L_ATOMIC_DIRECTORY_NOREPLACE_UNSUPPORTED"
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".canonical_output.p4l-staging-*"))


def test_linux_unsupported_renameat2_errno_fails_closed_without_fallback(monkeypatch, tmp_path):
    force_linux(monkeypatch)
    output_dir = tmp_path / "canonical_output"
    fake_renameat2 = FakeRenameAt2(result=-1, err=errno.ENOSYS)
    monkeypatch.setattr(builder, "load_libc", lambda: FakeLibc(fake_renameat2))

    with pytest.raises(builder.BuildBlocked) as excinfo:
        builder.finalize_payloads_atomic(output_dir, payloads())

    assert str(excinfo.value) == "P4L_ATOMIC_DIRECTORY_NOREPLACE_UNSUPPORTED"
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".canonical_output.p4l-staging-*"))


def test_windows_publication_branch_uses_existing_directory_rename(monkeypatch, tmp_path):
    force_windows(monkeypatch)
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "canonical_output"
    staging_dir.mkdir()
    (staging_dir / builder.SIDECAR_NAME).write_bytes(b'{"row_id":"attempt"}\n')

    builder.atomic_publish_directory_noreplace(staging_dir, output_dir)

    assert output_dir.is_dir()
    assert not staging_dir.exists()
    assert (output_dir / builder.SIDECAR_NAME).read_bytes() == b'{"row_id":"attempt"}\n'


def test_windows_publication_branch_preserves_fileexists_mapping(monkeypatch, tmp_path):
    force_windows(monkeypatch)
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "canonical_output"
    staging_dir.mkdir()
    output_dir.mkdir()

    def raise_file_exists(self, target):
        assert self == staging_dir
        assert target == output_dir
        raise FileExistsError("exists")

    monkeypatch.setattr(builder.Path, "rename", raise_file_exists)

    with pytest.raises(builder.BuildBlocked) as excinfo:
        builder.atomic_publish_directory_noreplace(staging_dir, output_dir)

    assert str(excinfo.value) == "P4L_OUTPUT_PATH_PREEXISTING"
    assert staging_dir.is_dir()
    assert output_dir.is_dir()


def test_other_platforms_fail_closed_without_generic_rename(monkeypatch, tmp_path):
    monkeypatch.setattr(builder, "running_on_windows", lambda: False)
    monkeypatch.setattr(builder, "running_on_linux", lambda: False)
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "canonical_output"
    staging_dir.mkdir()

    with pytest.raises(builder.BuildBlocked) as excinfo:
        builder.atomic_publish_directory_noreplace(staging_dir, output_dir)

    assert str(excinfo.value) == "P4L_ATOMIC_DIRECTORY_NOREPLACE_UNSUPPORTED"
    assert staging_dir.is_dir()
    assert not output_dir.exists()


def test_finalize_fails_closed_when_output_dir_preexists_with_unrelated_contents(tmp_path):
    output_dir = tmp_path / "canonical_output"
    unrelated_dir = output_dir / "unrelated_dir"
    unrelated_file = unrelated_dir / "user_data.txt"
    unrelated_dir.mkdir(parents=True)
    unrelated_file.write_text("do not touch\n", encoding="utf-8")
    payloads = {
        builder.SIDECAR_NAME: b'{"row_id":"attempt"}\n',
        builder.PROVENANCE_NAME: b'{"attempt":true}\n',
    }

    with pytest.raises(builder.BuildBlocked) as excinfo:
        builder.finalize_payloads_atomic(output_dir, payloads)

    assert str(excinfo.value) == "P4L_OUTPUT_PATH_PREEXISTING"
    assert unrelated_dir.is_dir()
    assert unrelated_file.read_text(encoding="utf-8") == "do not touch\n"
    assert not (output_dir / builder.SIDECAR_NAME).exists()
    assert not (output_dir / builder.PROVENANCE_NAME).exists()
    assert not list(tmp_path.glob(".canonical_output.p4l-backup-*"))


def test_finalize_fails_closed_when_output_dir_preexists_empty(tmp_path):
    output_dir = tmp_path / "canonical_output"
    output_dir.mkdir()
    payloads = {
        builder.SIDECAR_NAME: b'{"row_id":"attempt"}\n',
        builder.PROVENANCE_NAME: b'{"attempt":true}\n',
    }

    with pytest.raises(builder.BuildBlocked) as excinfo:
        builder.finalize_payloads_atomic(output_dir, payloads)

    assert str(excinfo.value) == "P4L_OUTPUT_PATH_PREEXISTING"
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_finalize_fails_closed_when_output_path_preexists_as_file(tmp_path):
    output_path = tmp_path / "canonical_output"
    output_path.write_text("existing file\n", encoding="utf-8")
    payloads = {
        builder.SIDECAR_NAME: b'{"row_id":"attempt"}\n',
        builder.PROVENANCE_NAME: b'{"attempt":true}\n',
    }

    with pytest.raises(builder.BuildBlocked) as excinfo:
        builder.finalize_payloads_atomic(output_path, payloads)

    assert str(excinfo.value) == "P4L_OUTPUT_PATH_PREEXISTING"
    assert output_path.read_text(encoding="utf-8") == "existing file\n"


def test_finalize_fails_closed_when_output_path_preexists_as_broken_symlink(tmp_path):
    output_path = tmp_path / "canonical_output"
    try:
        os.symlink("missing-target", output_path, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    payloads = {
        builder.SIDECAR_NAME: b'{"row_id":"attempt"}\n',
        builder.PROVENANCE_NAME: b'{"attempt":true}\n',
    }

    with pytest.raises(builder.BuildBlocked) as excinfo:
        builder.finalize_payloads_atomic(output_path, payloads)

    assert str(excinfo.value) == "P4L_OUTPUT_PATH_PREEXISTING"
    assert output_path.is_symlink()
    assert not output_path.exists()
