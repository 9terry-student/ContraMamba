from __future__ import annotations

import inspect
import math

from scripts import (
    analyze_reason_router_gen4_mamba370m14b_stagewise_coupling_localization
    as analysis,
)
from scripts import (
    reason_router_gen4_mamba370m14b_stagewise_coupling_localization_fast_cuda
    as runner,
)


def fake_summary(values):
    return {
        "n": 300,
        "mean": values,
        "median": values,
        "fraction_positive": float(values > 0),
        "fraction_negative": float(values < 0),
        "min": values,
        "max": values,
    }


def fake_stage(d, a, b, c2=None):
    c2 = d if c2 is None else c2
    return {
        "pair_margin": {
            "A_sel": fake_summary(a),
            "B_ctrl": fake_summary(b),
            "D": fake_summary(d),
        },
        "cell_margin": {
            "C0_SHAM": {
                "A_sel": fake_summary(a),
                "B_ctrl": fake_summary(b),
                "D": fake_summary(d),
            },
            "C2_NAME": {
                "A_sel": fake_summary(a),
                "B_ctrl": fake_summary(b),
                "D": fake_summary(c2),
            },
        },
    }


def test_decomposition_identity() -> None:
    got = analysis.decompose_three(2.0, 1.0, 1.4)
    assert math.isclose(got["A_sel_native_minus_neutralized"], 1.0)
    assert math.isclose(got["B_ctrl_control_minus_neutralized"], 0.4)
    assert math.isclose(got["D_native_minus_control"], 0.6)


def test_first_persistent_localization() -> None:
    stages = {}
    for i, name in enumerate(runner.STAGE_ORDER):
        d = 0.1 if i < 5 else -0.1
        stages[name] = fake_stage(d=d, a=d, b=-d, c2=d)
    loc = analysis.localize("mamba14b", stages)
    assert loc["first_persistent_D_negative"] == runner.STAGE_ORDER[5]
    assert loc["first_persistent_A_sel_negative"] == runner.STAGE_ORDER[5]
    assert loc["first_persistent_B_ctrl_positive"] == runner.STAGE_ORDER[5]


def test_cross_scale_persistent_opposition() -> None:
    s370 = {
        name: fake_stage(d=(0.0 if i == 0 else 0.1), a=0.1, b=-0.1, c2=(0.0 if i == 0 else 0.2))
        for i, name in enumerate(runner.STAGE_ORDER)
    }
    s14 = {
        name: fake_stage(d=(0.0 if i < 4 else -0.1), a=-0.1, b=0.1, c2=(0.0 if i < 4 else -0.2))
        for i, name in enumerate(runner.STAGE_ORDER)
    }
    got = analysis.cross_scale_localization(s370, s14)
    assert got["first_persistent_pair_D_opposition"] == runner.STAGE_ORDER[4]
    assert got["first_persistent_C2_D_opposition"] == runner.STAGE_ORDER[4]


def test_analyzer_adds_no_inferential_p_value() -> None:
    source = inspect.getsource(analysis).lower()
    assert "scipy" not in source
    assert "ttest" not in source
    assert '"p_value_count_added": 0' in source
