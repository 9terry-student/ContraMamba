from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "longterm_k3c_native_recurrence_contribution_execution.py"
)
spec = importlib.util.spec_from_file_location("k3c_exec", MODULE_PATH)
assert spec is not None and spec.loader is not None
k3ce = importlib.util.module_from_spec(spec)
sys.modules["k3c_exec"] = k3ce
spec.loader.exec_module(k3ce)


def test_frozen_replay_identity():
    assert k3ce.K3C_REPLAY_IMPLEMENTATION_COMMIT == (
        "e91116b4b5837a16de1d6eabfa2503be5bfe1d3d"
    )
    assert k3ce.K3C_REPLAY_SHA256 == (
        "ab07c6052a07af354043e0aa38bf24fd73c2fde53fccec125c25f7db2718f233"
    )
    assert k3ce.K3C_REPLAY_GIT_BLOB == (
        "633ca87b365f50ac27dbec5a8595dbdad42a7dfe"
    )


def test_frozen_prereg_identity():
    assert k3ce.K3C_PREREG_COMMIT == (
        "b85272d88d0bb57db45fdc963d313714529e7975"
    )
    assert k3ce.K3C_PREREG_SHA256 == (
        "0a9de28237e107ce3a62d4fa9e0bb7f230d2e0289e019d3864edf491dd271436"
    )


def test_population_identities():
    assert k3ce.GENERATED_SOURCE_CANONICAL_SHA256 == (
        "33bff5a0b657d1ceb38ae9c651e1cadfc8308286398cc1b8c4245c47f1c42000"
    )
    assert k3ce.CANDIDATE_POOL_CANONICAL_SHA256 == (
        "9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e"
    )
    assert k3ce.RECIPROCAL_MAPPING_CANONICAL_SHA256 == (
        "4fbc0f6642db3b2c3cca148fdc03cdc738dd6e8718cffb3fcdee73cd5b7f9acc"
    )


def test_condition_set_exact():
    assert k3ce.CONDITIONS == (
        "BASE", "W_EQ", "H_EQ", "W_SEED_H_CARRY"
    )


def test_primary_order_exact():
    assert k3ce.PRIMARY_TEST_ORDER == (
        "R_DOM", "R_CARRY",
        "D_DOM", "D_CARRY",
        "DISP_DOM", "DISP_CARRY",
        "P_DOM", "P_CARRY",
    )


def test_authority_marker_parser():
    text = """
`K3C_EXECUTION_AUTHORITY_SCHEMA=k3c-scientific-execution-authority-v1`
`K3C_SCIENTIFIC_RECURRENT_STATE_EXECUTION_AUTHORIZED=YES`
`K3C_ONE_SCIENTIFIC_EXECUTION=YES`
`K3C_EXECUTION_IMPLEMENTATION_COMMIT=abc`
`K3C_EXECUTION_RUNNER_SHA256=runner`
`K3C_EXECUTION_TEST_SHA256=test`
`K3C_REPLAY_IMPLEMENTATION_COMMIT=replay`
`K3C_REPLAY_MODULE_SHA256=replay_sha`
`K3C_PREREG_COMMIT=prereg`
`K3C_PREREG_SHA256=prereg_sha`
"""
    out = k3ce.parse_authority_markers(text)
    assert out["K3C_EXECUTION_IMPLEMENTATION_COMMIT"] == "abc"
    assert out["K3C_ONE_SCIENTIFIC_EXECUTION"] == "YES"


def test_authority_marker_duplicate_rejected():
    text = """
K3C_EXECUTION_AUTHORITY_SCHEMA=x
K3C_EXECUTION_AUTHORITY_SCHEMA=x
K3C_SCIENTIFIC_RECURRENT_STATE_EXECUTION_AUTHORIZED=YES
K3C_ONE_SCIENTIFIC_EXECUTION=YES
K3C_EXECUTION_IMPLEMENTATION_COMMIT=a
K3C_EXECUTION_RUNNER_SHA256=b
K3C_EXECUTION_TEST_SHA256=c
K3C_REPLAY_IMPLEMENTATION_COMMIT=d
K3C_REPLAY_MODULE_SHA256=e
K3C_PREREG_COMMIT=f
K3C_PREREG_SHA256=g
"""
    with pytest.raises(k3ce.ContractError):
        k3ce.parse_authority_markers(text)


def test_authority_marker_missing_rejected():
    with pytest.raises(k3ce.ContractError):
        k3ce.parse_authority_markers(
            "K3C_EXECUTION_AUTHORITY_SCHEMA=x\n"
        )


def test_output_target_exact_name(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = tmp_path / "k3c-contribution-abcdef123456"
    out = k3ce.output_target(
        repo,
        target,
        "abcdef1234567890",
    )
    assert out == target.resolve()


def test_output_target_wrong_name_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(k3ce.ContractError):
        k3ce.output_target(
            repo,
            tmp_path / "wrong",
            "abcdef1234567890",
        )


def test_output_target_inside_repo_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(k3ce.ContractError):
        k3ce.output_target(
            repo,
            repo / "k3c-contribution-abcdef123456",
            "abcdef1234567890",
        )


def test_output_target_existing_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = tmp_path / "k3c-contribution-abcdef123456"
    target.mkdir()
    with pytest.raises(k3ce.ContractError):
        k3ce.output_target(repo, target, "abcdef1234567890")


def _candidates():
    return [
        {"stable_item_id": f"id-{i:03d}"}
        for i in range(300)
    ]


def test_reciprocal_mapping_shape_with_patched_sha(monkeypatch):
    rows = [
        {
            "block_index": b,
            "item_a_stable_id": f"id-{2*b:03d}",
            "item_b_stable_id": f"id-{2*b+1:03d}",
            "item_a_donor_index": 2*b+1,
            "item_b_donor_index": 2*b,
        }
        for b in range(150)
    ]
    wanted = k3ce.sha256_bytes(k3ce.canonical_json(rows))
    monkeypatch.setattr(
        k3ce,
        "RECIPROCAL_MAPPING_CANONICAL_SHA256",
        wanted,
    )
    out = k3ce.reciprocal_mapping(_candidates())
    assert len(out) == 150
    assert out[0]["item_a_donor_index"] == 1
    assert out[0]["item_b_donor_index"] == 0


def _fake_item(index, base):
    conditions = {}
    for condition, shift in (
        ("BASE", 0.0),
        ("W_EQ", -1.0),
        ("H_EQ", -0.25),
        ("W_SEED_H_CARRY", -0.5),
    ):
        conditions[condition] = {
            "X_pair_specificity": {
                "R": base + shift,
                "D": base + shift,
                "DISP": -(base + shift),
                "P": -(base + shift),
            }
        }
    return {
        "block_index": index // 2,
        "stable_item_id": f"id-{index:03d}",
        "conditions": conditions,
    }


class _FakeK3C:
    EXPECTED_DIRECTION = {"R": 1, "D": 1, "DISP": -1, "P": -1}

    @classmethod
    def aligned_signal(cls, metric, value):
        if value is None:
            return None
        return cls.EXPECTED_DIRECTION[metric] * float(value)

    @staticmethod
    def mechanism_values(z_base, z_w_eq, z_h_eq, z_carry):
        if None in (z_base, z_w_eq, z_h_eq):
            return {
                "ATT_W": None,
                "ATT_H": None,
                "DOM": None,
                "CARRY": z_carry,
            }
        att_w = z_base - z_w_eq
        att_h = z_base - z_h_eq
        return {
            "ATT_W": att_w,
            "ATT_H": att_h,
            "DOM": att_w - att_h,
            "CARRY": z_carry,
        }


def test_build_block_rows():
    items = [
        _fake_item(i, 2.0 + (i % 2))
        for i in range(300)
    ]
    rows = k3ce.build_block_rows(_FakeK3C, items)
    assert len(rows) == 150
    first = rows[0]
    assert first["B_BASE_R"] == pytest.approx(2.5)
    assert first["Z_BASE_DISP"] == pytest.approx(2.5)
    assert first["DOM_R"] == pytest.approx(0.75)
    assert first["CARRY_R"] == pytest.approx(2.0)


def test_block_rows_reject_bad_count():
    with pytest.raises(k3ce.ContractError):
        k3ce.build_block_rows(_FakeK3C, [])


def test_parser_requires_output_dir():
    parser = k3ce.parser()
    actions = {a.dest for a in parser._actions}
    assert "output_dir" in actions
    assert "seed180_handoff" in actions
    assert "hf_revision" in actions


def test_output_schemas_are_k3c_specific():
    assert k3ce.ITEM_SCHEMA == "k3c-contribution-item-v1"
    assert k3ce.BLOCK_SCHEMA == "k3c-contribution-block-v1"
    assert k3ce.INTEGRITY_SCHEMA == "k3c-contribution-integrity-v1"
    assert k3ce.MANIFEST_SCHEMA == "k3c-contribution-manifest-v1"


def test_scientific_artifact_conditions_exclude_wh_eq():
    assert "WH_EQ" not in k3ce.CONDITIONS
    assert "W_SEED_H_CARRY" in k3ce.CONDITIONS


def test_runtime_is_cpu_sequential_identity():
    assert k3ce.TRANSFORMERS_VERSION == "5.12.1"
    assert k3ce.PRIMARY_LAYER == 23
    assert k3ce.W == 8
    assert k3ce.MAMBA_SOURCE_SHA256 == (
        "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
    )
