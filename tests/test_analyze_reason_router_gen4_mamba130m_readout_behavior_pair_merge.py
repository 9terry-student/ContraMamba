from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import (
    analyze_reason_router_gen4_mamba130m_readout_behavior_pair_merge
    as subject,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sums(root: Path, names: list[str]) -> None:
    (root / subject.SUMS_FILE).write_text(
        "".join(f"{_sha(root / name)}  {name}\n" for name in sorted(names)),
        encoding="utf-8",
        newline="\n",
    )


def test_sign_categories_preserve_zero() -> None:
    assert subject.sign_category(1.0, 2.0) == "both_positive"
    assert subject.sign_category(-1.0, -2.0) == "both_negative"
    assert subject.sign_category(1.0, -2.0) == "delta_positive_beh_negative"
    assert subject.sign_category(-1.0, 2.0) == "delta_negative_beh_positive"
    assert subject.sign_category(0.0, 0.0) == "both_zero"
    assert subject.sign_category(0.0, 2.0) == "delta_zero_beh_positive"
    assert subject.sign_category(-1.0, 0.0) == "delta_negative_beh_zero"


def test_spearman_coefficient_has_no_p_value() -> None:
    x = np.linspace(-2.0, 2.0, subject.N)
    y = x**3
    value = subject.spearman_without_p(x, y)
    assert value == pytest.approx(1.0)
    out = subject.summarize(x, y)
    assert "spearman_Delta_L_vs_D_BEH" in out
    assert not any("p_value" in key.lower() for key in out)


def test_summary_high_association_and_residual() -> None:
    x = np.linspace(-0.01, 0.02, subject.N)
    y = 2.0 * x + 0.003
    out = subject.summarize(x, y)
    assert out["pearson_Delta_L_vs_D_BEH"] == pytest.approx(1.0)
    assert out["spearman_Delta_L_vs_D_BEH"] == pytest.approx(1.0)
    assert out["residual_D_BEH_minus_Delta_L"]["mean"] == pytest.approx(
        float(np.mean(y - x))
    )


def test_readout_identity_constants_are_exact() -> None:
    assert subject.READOUT_FREEZE_COMMIT == (
        "8a2043a261c2ff7a01ff3c24855aca8640e1ca87"
    )
    assert subject.EXPECTED_READOUT_SHA256[subject.READOUT_PAIR_FILE] == (
        "8012e07d2a53b90d9d271ca4e4aa93af26f78c2fdfeaf0d149e38ff602c232b4"
    )
    assert subject.PLAN_SHA256 == (
        "a8a2cb2af1ed35b82f404a713c15ebf176cd645f653c6eacf73722b6e34058b3"
    )


def test_behavior_identity_constants_are_exact() -> None:
    assert subject.EXPECTED_BEHAVIOR_SHA256[0][subject.BEHAVIOR_ROW_FILE] == (
        "f9e61c1979c5b3f4ce4b9c6b3f15829bbd014efe7674a38b19f276f948672dc8"
    )
    assert subject.EXPECTED_BEHAVIOR_SHA256[1][subject.BEHAVIOR_ROW_FILE] == (
        "b3e6873db21a892f21e4ac451d84491193e88ccaa917cb3cc30f16477f8b7fbe"
    )
    assert subject.EXPECTED_CHECKPOINT_SHA256 == (
        "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f"
    )


def test_output_contract_has_zero_inference() -> None:
    assert subject.RESULT == (
        "PASS_GEN4_MAMBA130M_READOUT_BEHAVIOR_PAIR_LEVEL_DESCRIPTIVE_MERGE"
    )
    assert subject.N == 300
    assert subject.EXPECTED_PAIRS[0] == "xg1_fact_2701"
    assert subject.EXPECTED_PAIRS[-1] == "xg1_fact_3000"

def test_git_blob_checksum_mode_survives_crlf_worktree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "historical"
    root.mkdir()

    row_blob = b'{"x":1}\n{"x":2}\n'
    summary_blob = b'{"result":"ok"}\n'
    expected = {
        subject.BEHAVIOR_ROW_FILE:
            hashlib.sha256(row_blob).hexdigest(),
        subject.BEHAVIOR_SUMMARY_FILE:
            hashlib.sha256(summary_blob).hexdigest(),
    }
    sums_blob = "".join(
        f"{digest}  {name}\n"
        for name, digest in expected.items()
    ).encode("utf-8")

    # Simulate Windows checkout conversion: local bytes differ from frozen bytes.
    (root / subject.BEHAVIOR_ROW_FILE).write_bytes(
        row_blob.replace(b"\n", b"\r\n")
    )
    (root / subject.BEHAVIOR_SUMMARY_FILE).write_bytes(
        summary_blob.replace(b"\n", b"\r\n")
    )
    (root / subject.SUMS_FILE).write_bytes(
        sums_blob.replace(b"\n", b"\r\n")
    )

    blob_map = {
        (root / subject.BEHAVIOR_ROW_FILE).resolve(): row_blob,
        (root / subject.BEHAVIOR_SUMMARY_FILE).resolve(): summary_blob,
        (root / subject.SUMS_FILE).resolve(): sums_blob,
    }

    monkeypatch.setattr(
        subject,
        "git_blob_bytes",
        lambda path: blob_map[path.resolve()],
    )

    subject.validate_exact_sums(
        root,
        expected=expected,
        git_blob=True,
    )
    rows = subject.jsonl_rows(
        root / subject.BEHAVIOR_ROW_FILE,
        git_blob=True,
    )
    assert rows == [{"x": 1}, {"x": 2}]
