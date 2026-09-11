from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "longterm_k3_selective_ssm_retention_write_causal_execution.py"
)
spec = importlib.util.spec_from_file_location("k3_exec", MODULE_PATH)
assert spec is not None and spec.loader is not None
k3e = importlib.util.module_from_spec(spec)
sys.modules["k3_exec"] = k3e
spec.loader.exec_module(k3e)

REPLAY_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "longterm_k3_selective_ssm_retention_write_causal_decomposition.py"
)
rspec = importlib.util.spec_from_file_location("k3_replay_for_exec_test", REPLAY_PATH)
assert rspec is not None and rspec.loader is not None
k3 = importlib.util.module_from_spec(rspec)
sys.modules["k3_replay_for_exec_test"] = k3
rspec.loader.exec_module(k3)


def test_frozen_replay_dependency():
    assert k3e.K3_REPLAY_IMPLEMENTATION_COMMIT == "a2f33796903466680091ccbede9f6728022e678f"
    assert k3e.K3_REPLAY_SHA256 == "116e9daae8a4d62b19d8ab4e7a4d4171df8ad7ae99b08953fc1f2ba49563a60e"
    assert k3e.K3_REPLAY_GIT_BLOB == "24278583af0132f4ec5cf8610d2cb85972c1bd51"


def test_frozen_prereg_dependency():
    assert k3e.K3_PREREG_COMMIT == "20032bb53d77416bb7eb25411eb4f77b008648b4"
    assert k3e.K3_PREREG_SHA256 == "7561f4188921b645eba3d7108bcd007b7c223389b5e1abc512cc8d61af976c84"


def test_frozen_k2r_population_and_baseline():
    assert k3e.K2R_CANDIDATE_SHA256 == "00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4"
    assert k3e.K2R_BLOCK_SHA256 == "e4982e8a57e15863227d17f080e7a7a699fb14100387fc36c7c47ab68962c8de"


def test_frozen_geometry():
    assert k3e.PRIMARY_LAYER == 23
    assert k3e.W == 8
    assert k3e.N_ITEMS == 300
    assert k3e.N_BLOCKS == 150
    assert k3e.METRICS == ("R", "D", "DISP", "P")


def test_primary_test_order_exact():
    assert k3e.PRIMARY_TEST_ORDER == (
        "R_ATT", "R_SEL", "D_ATT", "D_SEL",
        "DISP_ATT", "DISP_SEL", "P_ATT", "P_SEL",
    )


def _authority_text(**overrides):
    values = {
        "K3_EXECUTION_AUTHORITY_SCHEMA": k3e.AUTHORITY_SCHEMA,
        "K3_SCIENTIFIC_INTERVENTION_EXECUTION_AUTHORIZED": "YES",
        "K3_ONE_SCIENTIFIC_EXECUTION": "YES",
        "K3_EXECUTION_IMPLEMENTATION_COMMIT": "a" * 40,
        "K3_EXECUTION_RUNNER_SHA256": "b" * 64,
        "K3_EXECUTION_TEST_SHA256": "c" * 64,
        "K3_REPLAY_IMPLEMENTATION_COMMIT": k3e.K3_REPLAY_IMPLEMENTATION_COMMIT,
        "K3_REPLAY_MODULE_SHA256": k3e.K3_REPLAY_SHA256,
        "K3_PREREG_COMMIT": k3e.K3_PREREG_COMMIT,
    }
    values.update(overrides)
    return "\n".join(f"`{k} = {v}`" for k, v in values.items()) + "\n"


def test_authority_marker_parser_exact_set():
    out = k3e.parse_authority_markers(_authority_text())
    assert out["K3_EXECUTION_AUTHORITY_SCHEMA"] == k3e.AUTHORITY_SCHEMA
    assert out["K3_SCIENTIFIC_INTERVENTION_EXECUTION_AUTHORIZED"] == "YES"
    assert len(out) == 9


def test_authority_marker_parser_rejects_duplicate():
    text = _authority_text() + f"`K3_ONE_SCIENTIFIC_EXECUTION = YES`\n"
    with pytest.raises(k3e.ContractError):
        k3e.parse_authority_markers(text)


def test_authority_marker_parser_rejects_missing():
    text = _authority_text().replace("`K3_ONE_SCIENTIFIC_EXECUTION = YES`\n", "")
    with pytest.raises(k3e.ContractError):
        k3e.parse_authority_markers(text)


def test_output_name_is_authority_bound(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    commit = "1234567890abcdef" + "0" * 24
    good = tmp_path / f"k3-retention-write-{commit[:12]}"
    assert k3e.output_target(repo, good, commit) == good.resolve()


def test_output_name_mismatch_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(k3e.ContractError):
        k3e.output_target(repo, tmp_path / "wrong", "1" * 40)


def _fake_item(index: int, base):
    conditions = {}
    for condition, offset in (("BASE", 0.0), ("W_EQ", 1.0), ("G_EQ", 2.0)):
        conditions[condition] = {
            "X_pair_specificity": {
                "R": base[0] + offset,
                "D": base[1] + offset,
                "DISP": base[2] + offset,
                "P": base[3] + offset,
            }
        }
    return {
        "item_index": index,
        "block_index": index // 2,
        "stable_item_id": f"id-{index}",
        "conditions": conditions,
    }


def test_block_rows_preserve_half_sum_and_alignment():
    items = []
    for i in range(k3e.N_ITEMS):
        items.append(_fake_item(i, (1.0, 2.0, -3.0, -4.0)))
    rows = k3e.build_block_rows(k3, items)
    assert len(rows) == 150
    row = rows[0]
    assert row["B_BASE_R"] == 1.0
    assert row["B_BASE_DISP"] == -3.0
    assert row["Z_BASE_DISP"] == 3.0


def test_block_att_selectivity_uses_frozen_dominance():
    items = []
    for i in range(k3e.N_ITEMS):
        row = _fake_item(i, (10.0, 10.0, -10.0, -10.0))
        # Local: W_EQ weakens more. Net: G_EQ weakens more after sign alignment.
        row["conditions"]["W_EQ"]["X_pair_specificity"].update({"R": 2.0, "D": 2.0, "DISP": -8.0, "P": -8.0})
        row["conditions"]["G_EQ"]["X_pair_specificity"].update({"R": 8.0, "D": 8.0, "DISP": -2.0, "P": -2.0})
        items.append(row)
    block = k3e.build_block_rows(k3, items)[0]
    assert block["ATT_DOM_R"] == 8.0
    assert block["ATT_OTHER_R"] == 2.0
    assert block["SEL_R"] == 6.0
    assert block["ATT_DOM_P"] == 8.0
    assert block["ATT_OTHER_P"] == 2.0
    assert block["SEL_P"] == 6.0


def _archived_row(block_index=0, values=None):
    values = values or {"R": 1.0, "D": 2.0, "DISP": -3.0, "P": -4.0}
    return {
        "block_index": block_index,
        "block_id": f"k2r-block-{block_index:03d}",
        "item_a_stable_id": f"a-{block_index}",
        "item_b_stable_id": f"b-{block_index}",
        **{f"B_{m}": values[m] for m in k3e.METRICS},
    }


def _new_row(block_index=0, values=None):
    values = values or {"R": 1.0, "D": 2.0, "DISP": -3.0, "P": -4.0}
    return {
        "block_index": block_index,
        "block_id": f"k2r-block-{block_index:03d}",
        "item_a_stable_id": f"a-{block_index}",
        "item_b_stable_id": f"b-{block_index}",
        **{f"B_BASE_{m}": values[m] for m in k3e.METRICS},
    }


def test_baseline_reproduction_requires_exact_values():
    new = [_new_row(i) for i in range(k3e.N_BLOCKS)]
    old = [_archived_row(i) for i in range(k3e.N_BLOCKS)]
    out = k3e.verify_archived_baseline(new, old)
    assert out["status"] == "PASS_EXACT"
    assert out["metric_value_comparisons"] == 600


def test_baseline_reproduction_rejects_tolerance_style_difference():
    new = [_new_row(i) for i in range(k3e.N_BLOCKS)]
    old = [_archived_row(i) for i in range(k3e.N_BLOCKS)]
    old[3]["B_R"] += 1e-15
    with pytest.raises(k3e.ContractError):
        k3e.verify_archived_baseline(new, old)


def test_primary_from_blocks_maps_att_and_sel():
    rows = []
    for i in range(k3e.N_BLOCKS):
        row = {"block_index": i}
        for metric in k3e.METRICS:
            row[f"ATT_DOM_{metric}"] = 1.0
            row[f"SEL_{metric}"] = 1.0
        rows.append(row)
    result = k3e.primary_from_blocks(k3, rows)
    assert result["full_support"] is True
    assert result["scientific_verdict"] == k3.SUCCESS_VERDICT
    assert result["schema_version"] == k3e.PRIMARY_SCHEMA


def test_primary_from_blocks_contradiction_propagates():
    rows = []
    for i in range(k3e.N_BLOCKS):
        row = {"block_index": i}
        for metric in k3e.METRICS:
            row[f"ATT_DOM_{metric}"] = 1.0
            row[f"SEL_{metric}"] = 1.0
        row["SEL_D"] = -1.0
        rows.append(row)
    result = k3e.primary_from_blocks(k3, rows)
    assert result["full_support"] is False
    assert result["scientific_verdict"] == k3.CONTRADICTION_VERDICT


def test_canonical_jsonl_lf_and_roundtrip():
    raw = k3e.canonical_jsonl([{"b": 2, "a": 1}, {"x": "y"}])
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    rows = k3e.parse_jsonl_bytes(raw)
    assert rows == [{"a": 1, "b": 2}, {"x": "y"}]


def test_parser_exposes_only_scientific_required_args():
    p = k3e.parser()
    actions = {a.dest for a in p._actions}
    assert {"seed180_handoff", "hf_revision", "output_dir"} <= actions
    assert "replay_preflight" not in actions
    assert "layer" not in actions
    assert "window" not in actions


def test_no_configurable_scientific_geometry_in_parser():
    p = k3e.parser()
    options = {opt for a in p._actions for opt in a.option_strings}
    forbidden = {"--layer", "--window", "--metric", "--intervention-onset", "--alpha", "--threshold"}
    assert not (options & forbidden)


def test_manifest_schema_is_frozen():
    assert k3e.MANIFEST_SCHEMA == "k3-causal-manifest-v1"
    assert k3e.ITEM_SCHEMA == "k3-causal-item-v1"
    assert k3e.BLOCK_SCHEMA == "k3-causal-block-v1"


def test_runtime_recurrence_identity_frozen():
    assert k3e.TRANSFORMERS_VERSION == "5.12.1"
    assert k3e.MAMBA_SOURCE_SHA256 == "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
    assert k3e.HF_REVISION == "5708daa364c50b880e7bd92eab456e0d34492ee9"


def test_authority_file_path_is_single_fixed_path():
    assert k3e.AUTHORITY_REL == "reports/longterm_k3_selective_ssm_retention_write_causal_execution_authority_spec_candidate.md"


def test_historical_dirty_contract_only_k1():
    assert k3e.HISTORICAL_K1_UNTRACKED == {
        "scripts/longterm_k1_native_state_kinematics.py",
        "tests/test_longterm_k1_native_state_kinematics.py",
    }
