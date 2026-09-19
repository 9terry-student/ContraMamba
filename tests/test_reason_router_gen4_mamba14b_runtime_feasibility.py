from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import (
    reason_router_gen4_mamba14b_runtime_feasibility
    as subject,
)


def test_frozen_protocol_constants():
    assert subject.MODEL_REPO == "state-spaces/mamba-1.4b-hf"
    assert subject.DISCOVERY_REVISION == "main"
    assert subject.EXPECTED_CONFIG == {
        "hidden_size": 2048,
        "intermediate_size": 4096,
        "num_hidden_layers": 48,
        "state_size": 16,
        "conv_kernel": 4,
        "vocab_size": 50280,
    }
    assert subject.SEQ_LEN == 128
    assert subject.PROBE_BATCH_SIZES == (1, 2, 4, 8, 16, 32)
    assert subject.ARM == "G3-GROUP-D-HALF"


def test_validate_config_dict_pass_and_fail():
    subject.validate_config_dict(dict(subject.EXPECTED_CONFIG))

    bad = dict(subject.EXPECTED_CONFIG)
    bad["hidden_size"] = 1024

    with pytest.raises(
        subject.FeasibilityError,
        match="CONFIG_MISMATCH:hidden_size",
    ):
        subject.validate_config_dict(bad)


def test_snapshot_identity_records_resolved_revision_and_hashes(tmp_path):
    for name in subject.SNAPSHOT_FILES:
        (tmp_path / name).write_bytes(
            (name + "\n").encode("utf-8")
        )

    config = dict(subject.EXPECTED_CONFIG)
    (tmp_path / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()

    for name in subject.SNAPSHOT_FILES:
        source = tmp_path / name
        target = snapshot / name
        target.write_bytes(source.read_bytes())

    (snapshot / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    identity = subject.snapshot_identity(snapshot)

    assert identity["resolved_revision"] == revision
    assert identity["requested_revision"] == "main"
    assert identity["config"] == subject.EXPECTED_CONFIG

    expected = hashlib.sha256(
        (snapshot / "tokenizer.json").read_bytes()
    ).hexdigest()
    assert (
        identity["files"]["tokenizer.json"]["sha256"]
        == expected
    )


def test_snapshot_identity_rejects_non_sha_directory(tmp_path):
    snapshot = tmp_path / "main"
    snapshot.mkdir()

    for name in subject.SNAPSHOT_FILES:
        (snapshot / name).write_bytes(b"x")

    (snapshot / "config.json").write_text(
        json.dumps(subject.EXPECTED_CONFIG),
        encoding="utf-8",
    )

    with pytest.raises(
        subject.FeasibilityError,
        match="RESOLVED_REVISION_NOT_SHA",
    ):
        subject.snapshot_identity(snapshot)


def test_result_boundary_is_non_scientific():
    assert "RAW_OBSERVATION" in subject.RESULT_PASS
    assert subject.GPU_COUNT == 2
    assert subject.RESULT_PASS.startswith("PASS_MAMBA14B_RUNTIME_FEASIBILITY")
