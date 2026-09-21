from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path


holdout_stub = types.ModuleType(
    "scripts.build_reason_router_gen4_factor2_small_alpha_holdout"
)
bridge_stub = types.ModuleType(
    "scripts.reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda"
)
sys.modules[
    "scripts.build_reason_router_gen4_factor2_small_alpha_holdout"
] = holdout_stub
sys.modules[
    "scripts.reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda"
] = bridge_stub

from scripts import (  # noqa: E402
    reason_router_gen4_factor2_small_alpha_runtime_inputs
    as m,
)


def test_protocol_constants_are_exact() -> None:
    assert m.PAIR_FIRST == 5701
    assert m.PAIR_LAST == 6000
    assert m.PAIR_COUNT == 300
    assert m.PAIR_IDS == tuple(
        f"xg1_fact_{i}" for i in range(5701, 6001)
    )
    assert m.TARGET_CELLS == ("C0_SHAM", "C2_NAME")
    assert m.ROWS_PER_SCALE == 600
    assert m.BEHAVIORAL_ALPHAS == (
        0.25,
        0.125,
        0.0625,
        0.03125,
    )


def test_frozen_structural_and_token_gate_hashes_are_exact() -> None:
    assert m.STRUCTURAL_SOURCE_SHA256 == (
        "05026973b2ec61847c85d6aab800eada130aad9e9c7edb14a4d8d88f41544c4e"
    )
    assert m.STRUCTURAL_ROW_SHA256 == (
        "08da7b3b1d9d92f189b6481abd0b889aeac908519eeae804b8574328ce497f51"
    )
    assert m.TOKEN_GATE_CROSS_SCALE_SHA256 == (
        "81790aa7fe8c6dc35022485cb70c9300ea28cfe299f2d8822673e947abed2d9d"
    )


def test_runtime_input_path_reuses_frozen_scale_spec() -> None:
    source = inspect.getsource(m.scale_spec)
    assert "bridge.scale_spec" in source


def test_population_contract_is_fresh_and_prospective() -> None:
    source = inspect.getsource(m.load_population)
    assert "xg1_fact_5701" in source
    assert "xg1_fact_6000" in source
    assert '"K(alpha)"' in source
    assert '"negative_alpha_arm_allowed"' in source


def test_input_state_revalidates_anchor_contract() -> None:
    source = inspect.getsource(m.build_input_state)
    assert "validate_token_gate" in source
    assert "geom.load_tokenizer" in source
    assert "adapter.encode_gen4_rows" in source
    assert "analyze_required_anchors_for_row" in source
    assert '"A_IDENTITY"' in source
    assert '"A_NAME"' in source
    assert "IDENTITY_NAME_INDEX_MISMATCH" in source


def test_runtime_helper_contains_no_model_or_behavior_execution() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8").lower()
    forbidden = (
        "historical_forward",
        ".forward(",
        ".backward(",
        "torch.cuda",
        "correct_class_logit_margin",
        "d_beh",
        "pearson",
        "spearman",
        "rmse",
        "ttest",
    )
    for token in forbidden:
        assert token not in source
