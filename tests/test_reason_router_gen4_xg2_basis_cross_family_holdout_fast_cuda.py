from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda
    as run,
)


def _signed_probe(
    orientation: int,
    f_value: float,
) -> dict[str, object]:
    return {
        "orientation": orientation,
        "delta_h_l2":
            2.0 * run.EPSILON,
        "plus_path_efficiency":
            f_value + 1.0,
        "minus_path_efficiency":
            1.0,
        "F": f_value,
        "midpoint_max_abs_residual":
            0.0,
        "pair_delta_max_abs_residual":
            0.0,
        (
            "applied_correction_"
            "max_abs_residual"
        ):
            0.0,
        "runtime_correction_l2":
            2.0 * run.EPSILON,
        "model_forward_count": 2,
    }


def _direction_probe(
    basis_family: str,
    basis_index: int,
    j_value: float,
) -> dict[str, object]:
    f_plus = (
        j_value * run.EPSILON
    )
    f_minus = (
        -j_value * run.EPSILON
    )

    return {
        "basis_family":
            basis_family,
        "basis_index":
            basis_index,
        "epsilon":
            run.EPSILON,
        "F_plus":
            f_plus,
        "F_minus":
            f_minus,
        "J":
            j_value,
        "J_squared":
            j_value * j_value,
        "positive_probe":
            _signed_probe(
                1,
                f_plus,
            ),
        "negative_probe":
            _signed_probe(
                -1,
                f_minus,
            ),
        "model_forward_count":
            run.FORWARDS_PER_DIRECTION,
    }


def _seed(
    family: str,
    index: int = 0,
) -> dict[str, object]:
    return {
        "schema_version":
            run.PROBE_SEED_SCHEMA,
        "family_key":
            family,
        "source_pair_id":
            run._expected_pairs(
                family
            )[index],
        "holdout_pair_index":
            index,
        "target_plus_anchor":
            10,
        "target_minus_anchor":
            11,
        "reference_plus_anchor":
            12,
        "reference_minus_anchor":
            13,
    }


def test_frozen_contract_and_budget():
    assert run.EXPECTED_BRANCH == (
        "gen4-k-xg2-basis-holdout"
    )
    assert run.SCOPE_FREEZE_COMMIT == (
        "c5679b8a22f103956a46cbc665354b4f9cd7063a"
    )
    assert (
        run.PREPARATION_FREEZE_COMMIT
        == "69b830b16d5b97745957c049cbc9eca14c5e5209"
    )
    assert run.SUBSPACE_DIM == 5
    assert run.EPSILON == 0.025
    assert run.FORWARDS_PER_DIRECTION == 4
    assert run.FORWARDS_PER_PAIR == 40
    assert (
        run.SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY
        == 12000
    )
    assert (
        run.BASELINE_FORWARD_BUDGET_THIS_RUN
        == 0
    )


@pytest.mark.parametrize(
    "family",
    ["xg2", "xg4"],
)
def test_exact_601_900_pair_range(
    family,
):
    pairs = run._expected_pairs(
        family
    )
    assert len(pairs) == 300
    assert pairs[0] == (
        f"{family}_fact_601"
    )
    assert pairs[-1] == (
        f"{family}_fact_900"
    )


@pytest.mark.parametrize(
    "family",
    ["xg2", "xg4"],
)
def test_frozen_holdout_and_anchor_artifacts(
    family,
):
    facts, rows, manifest = (
        run.eligibility.load_family(
            family
        )
    )

    assert len(facts) == 300
    assert len(rows) == 1800
    assert facts[0]["pair_id"] == (
        f"{family}_fact_601"
    )
    assert facts[-1]["pair_id"] == (
        f"{family}_fact_900"
    )
    assert (
        manifest["family_key"]
        == family
    )

    anchors = (
        run.load_frozen_anchor_manifest(
            family
        )
    )
    assert len(anchors) == 1800


@pytest.mark.parametrize(
    "source_family",
    ["xg2", "xg4"],
)
def test_endpoint_is_fixed_xg2_minus_xg4(
    monkeypatch,
    source_family,
):
    calls = []

    def fake_direction(
        _family,
        _seed,
        _direction,
        *,
        basis_family,
        basis_index,
        **_kwargs,
    ):
        calls.append(
            (
                basis_family,
                basis_index,
            )
        )

        if basis_family == "xg2":
            j_value = float(
                basis_index + 1
            )
        else:
            j_value = 0.5 * float(
                basis_index + 1
            )

        return _direction_probe(
            basis_family,
            basis_index,
            j_value,
        )

    monkeypatch.setattr(
        run.prior,
        "_run_direction_j",
        fake_direction,
    )

    basis = torch.eye(
        6,
        dtype=torch.float64,
    )[:, :5]

    result = run._run_pair(
        source_family,
        _seed(source_family),
        xg2_basis=basis,
        xg4_basis=basis,
        model=object(),
        runtime_ctx={},
        trace_code=object(),
        trace_line=1,
        encoded={},
        row_index={},
        events={},
        budget=object(),
    )

    assert calls == [
        ("xg2", 0),
        ("xg2", 1),
        ("xg2", 2),
        ("xg2", 3),
        ("xg2", 4),
        ("xg4", 0),
        ("xg4", 1),
        ("xg4", 2),
        ("xg4", 3),
        ("xg4", 4),
    ]

    expected_xg2 = sum(
        float(j * j)
        for j in range(1, 6)
    ) / 5.0

    expected_xg4 = sum(
        float((0.5 * j) ** 2)
        for j in range(1, 6)
    ) / 5.0

    assert result["E_XG2"] == (
        pytest.approx(
            expected_xg2
        )
    )
    assert result["E_XG4"] == (
        pytest.approx(
            expected_xg4
        )
    )
    assert result["Q"] == (
        pytest.approx(
            expected_xg2
            - expected_xg4
        )
    )
    assert (
        result[
            "scientific_model_forward_count_this_run"
        ]
        == 40
    )


@pytest.mark.parametrize(
    "source_family",
    ["xg2", "xg4"],
)
def test_negative_q_is_preserved_without_rescue(
    monkeypatch,
    source_family,
):
    def fake_direction(
        _family,
        _seed,
        _direction,
        *,
        basis_family,
        basis_index,
        **_kwargs,
    ):
        scale = (
            0.25
            if basis_family == "xg2"
            else 1.0
        )

        return _direction_probe(
            basis_family,
            basis_index,
            scale
            * float(
                basis_index + 1
            ),
        )

    monkeypatch.setattr(
        run.prior,
        "_run_direction_j",
        fake_direction,
    )

    basis = torch.eye(
        6,
        dtype=torch.float64,
    )[:, :5]

    result = run._run_pair(
        source_family,
        _seed(source_family),
        xg2_basis=basis,
        xg4_basis=basis,
        model=object(),
        runtime_ctx={},
        trace_code=object(),
        trace_line=1,
        encoded={},
        row_index={},
        events={},
        budget=object(),
    )

    assert result["Q"] < 0.0
    assert (
        result["Q"]
        == result["E_XG2"]
        - result["E_XG4"]
    )


def test_probe_seed_is_anchor_only(
    monkeypatch,
):
    monkeypatch.setattr(
        run.phase1,
        "_anchors_for_pair",
        lambda _pair, _events: {
            "tp": 101,
            "tm": 102,
            "rp": 103,
            "rm": 104,
        },
    )

    seed = run._probe_seed(
        "xg2",
        0,
        "xg2_fact_601",
        {},
    )

    assert seed == {
        "schema_version":
            run.PROBE_SEED_SCHEMA,
        "family_key":
            "xg2",
        "source_pair_id":
            "xg2_fact_601",
        "holdout_pair_index":
            0,
        "target_plus_anchor":
            101,
        "target_minus_anchor":
            102,
        "reference_plus_anchor":
            103,
        "reference_minus_anchor":
            104,
    }

    assert "delta_baseline" not in seed
    assert (
        "baseline_plus_path_efficiency"
        not in seed
    )
    assert (
        "baseline_minus_path_efficiency"
        not in seed
    )


def test_source_contains_no_inferential_execution():
    source = inspect.getsource(
        run
    )

    for token in (
        "import scipy",
        "from scipy",
        "ttest_",
        "multipletests",
        ".backward(",
        ".train(",
    ):
        assert token not in source

    assert (
        '"primary_inference_executed":\n            False'
        in source
    )
    assert (
        '"holm_correction_executed":\n            False'
        in source
    )
    assert (
        '"scientific_conclusion":\n            None'
        in source
    )


def test_parse_args_requires_execution_identity():
    args = run.parse_args([
        "--family",
        "xg2",
        "--expected-head",
        "abc",
        "--model-snapshot",
        "m",
        "--tokenizer-snapshot",
        "t",
        "--checkpoint",
        "c",
        "--output-dir",
        "o",
    ])

    assert args.family == "xg2"
    assert args.expected_head == "abc"
    assert args.output_dir == Path("o")
