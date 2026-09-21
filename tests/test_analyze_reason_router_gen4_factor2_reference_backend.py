from __future__ import annotations

import inspect
import numpy as np

from scripts import analyze_reason_router_gen4_factor2_reference_backend as analyzer


def test_origin_slope_1x_2x() -> None:
    x = [1.0, -2.0, 3.0]
    assert analyzer.origin_slope(x, x) == 1.0
    assert analyzer.origin_slope(x, [2.0, -4.0, 6.0]) == 2.0


def test_vector_relation_distinguishes_1x_2x() -> None:
    x = np.linspace(-1.0, 1.0, 8)
    one = analyzer.vector_relation(x, x)
    two = analyzer.vector_relation(x, 2.0 * x)
    assert np.isclose(one["origin_slope"], 1.0)
    assert np.isclose(one["rmse_vs_1x"], 0.0)
    assert np.isclose(two["origin_slope"], 2.0)
    assert np.isclose(two["rmse_vs_2x"], 0.0)


def test_analyzer_is_static_descriptive_only() -> None:
    source = inspect.getsource(analyzer).lower()
    assert "torch" not in source
    assert "transformers" not in source
    assert "scipy" not in source
    assert "ttest" not in source
    assert '"scientific_conclusion": none' in source
    assert '"p_value_count_added": 0' in source
