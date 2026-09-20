from __future__ import annotations
import inspect
from scripts import analyze_reason_router_gen4_averitec_370m_fixed_mirror_steering_failure_anatomy as s
from scripts import analyze_reason_router_gen4_pre_emission_forced_decisive_signed_p3_failure_anatomy as p

def test_steering_pins_frozen_raw():
    assert s.RAW_ROWS_SHA256 == "912fb91bdfc5e778e4d2f99cb9a8f7a5bdb8a4a1389b39de1d47947eb9058d0c"
    assert s.N == 2799 and s.RAW_ROW_COUNT == 8397

def test_steering_no_inference_or_execution():
    src=inspect.getsource(s).lower()
    for token in ("scipy","ttest","wilcoxon","mannwhitney","permutation_test","import torch","cuda"):
        assert token not in src
    assert '"no_new_p_values": true' in src
    assert '"no_model_execution": true' in src
    assert '"no_magnitude_search": true' in src

def test_steering_lambda_is_descriptive_only():
    src=inspect.getsource(s)
    assert "lambda_star_linearized" in src
    assert "not an executed steering coefficient" in src
    assert "LAMBDA_LT_ONE_WITHOUT_FLIP" in src

def test_precursor_pins_frozen_raw():
    assert p.RAW_ROWS_SHA256 == "5cc15d2626b350108c2c532ab8502d87aacb0d5253f82905ab83dc438aff5fcd"
    assert p.OFFSETS == (-4,-3,-2,-1)

def test_precursor_reports_all_signed_coordinates_symmetrically():
    src=inspect.getsource(p)
    for token in ('"a":desc(', '"b":desc(', '"l2":desc(', '"theta_radians":desc(', '"delta_a"', '"delta_b"', '"delta_l2"', '"delta_theta_wrapped"'):
        assert token in src

def test_precursor_no_inference_or_feature_selection():
    src=inspect.getsource(p).lower()
    for token in ("scipy","ttest","wilcoxon","mannwhitney","permutation_test","import torch","cuda"):
        assert token not in src
    assert '"new_p_value_count":0' in src
    assert '"feature_selection_performed":false' in src
    assert '"offset_selection_performed":false' in src
    assert '"coordinate_selection_performed":false' in src
    assert '"precursor_rescue_claimed":false' in src
