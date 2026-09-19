from __future__ import annotations

import inspect
import math
import subprocess
import sys
from pathlib import Path

from scripts import analyze_reason_router_gen4_small_epsilon_robustness as m


def test_frozen_constants() -> None:
    assert m.RAW_FREEZE_COMMIT == (
        "552236c171f3467cd6eb12f02aa89a3eefee2f45"
    )
    assert m.DESIGN_BLOB == "b0e67c1d0aae2df08ae22d44d411b455db1c3cbd"
    assert m.RAW_ITEMS_BLOB == "9d1ff10a66436eaed2a372e7e1f4e861486f98ea"
    assert m.RAW_SUMMARY_BLOB == "d2abe9c5fe93a668cf3d146968aa8d15218f8aa2"
    assert m.RAW_MANIFEST_BLOB == "d2e81a091efbbb20fdb03e4fcb0932e22009c60f"
    assert m.RAW_SUMS_BLOB == "058c022d0c28eee931c0a5e24689b0a93638fb59"
    assert m.RAW_ITEMS_SHA256 == (
        "6d06a83fa91075690ed8e2acbe166ce51e3a29c6a698fb83986c815cc4fe8c15"
    )
    assert m.REFERENCE_EPSILON == 0.025
    assert m.NEW_EPSILONS == (0.0125, 0.00625)
    assert m.ALL_EPSILONS == (0.025, 0.0125, 0.00625)


def test_frozen_inputs_validate_without_analysis() -> None:
    loaded = m.validate_and_load_frozen_inputs()
    assert len(loaded["reference_items"]) == 300
    assert len(loaded["raw_items"]) == 300
    assert loaded["raw_summary"]["primary_inference_executed"] is False
    assert loaded["raw_summary"]["p_value_count_added"] == 0
    assert loaded["raw_summary"]["scientific_conclusion"] is None

    reference_probe = loaded["reference_items"][0][
        "principal_direction_probes"
    ][0]
    assert "central_difference_numerator" not in reference_probe
    assert m.probe_numerator(
        reference_probe,
        epsilon=m.REFERENCE_EPSILON,
    ) == float(reference_probe["F_plus"]) - float(reference_probe["F_minus"])

    new_probe = loaded["raw_items"][0]["epsilon_observations"][0][
        "principal_direction_probes"
    ][0]
    assert "central_difference_numerator" in new_probe
    assert m.probe_numerator(
        new_probe,
        epsilon=m.NEW_EPSILONS[0],
    ) == float(new_probe["central_difference_numerator"])


def test_linear_quantile() -> None:
    xs = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert m.linear_quantile(xs, 0.0) == 0.0
    assert m.linear_quantile(xs, 0.5) == 20.0
    assert m.linear_quantile(xs, 1.0) == 40.0
    assert m.linear_quantile(xs, 0.25) == 10.0


def test_pearson_and_cosine() -> None:
    assert math.isclose(m.pearson([1, 2, 3], [2, 4, 6]), 1.0)
    assert math.isclose(m.cosine([1, 0], [1, 0]), 1.0)
    assert math.isclose(m.cosine([1, 0], [0, 1]), 0.0)


def _probe(
    key: str,
    j: float,
    *,
    epsilon: float,
    store_numerator: bool,
) -> dict:
    numerator = 2.0 * float(epsilon) * float(j)
    f_plus = numerator
    f_minus = 0.0
    realized_j = (f_plus - f_minus) / (2.0 * float(epsilon))
    probe = {
        "direction_key": key,
        "F_plus": f_plus,
        "F_minus": f_minus,
        "J": realized_j,
    }
    if store_numerator:
        probe["central_difference_numerator"] = numerator
    return probe


def _observation(
    eps: float,
    *,
    p3: float,
    p5: float,
    q0: float,
    qp: float,
) -> dict:
    contrib = {
        "P1": 1.0,
        "P2": 2.0,
        "P3": p3,
        "P4": 0.5,
        "P5": p5,
    }
    residual = q0 - qp
    return {
        "epsilon": eps,
        "prior_native_q0": q0,
        "plane_contributions": contrib,
        "Q_principal": qp,
        "reconstruction_residual": residual,
        "absolute_reconstruction_residual": abs(residual),
        "absolute_relative_reconstruction_residual_to_Q0":
            abs(residual / q0) if q0 != 0 else None,
        "principal_direction_probes": [
            _probe(
                key,
                float(i + 1),
                epsilon=eps,
                store_numerator=(eps != m.REFERENCE_EPSILON),
            )
            for i, key in enumerate(m.DIRECTION_ORDER)
        ],
    }


def _synthetic_inputs(*, bad_small: bool = False):
    refs = []
    raws = []
    for i in range(300):
        q0 = 10.0 + i / 100.0
        ref_obs = _observation(
            0.025, p3=5.0, p5=4.0, q0=q0, qp=q0 - 0.01
        )
        refs.append({
            "prior_native_q0": q0,
            "plane_contributions": ref_obs["plane_contributions"],
            "Q_principal": ref_obs["Q_principal"],
            "reconstruction_residual": ref_obs["reconstruction_residual"],
            "absolute_reconstruction_residual":
                ref_obs["absolute_reconstruction_residual"],
            "absolute_relative_reconstruction_residual_to_Q0":
                ref_obs["absolute_relative_reconstruction_residual_to_Q0"],
            "principal_direction_probes":
                ref_obs["principal_direction_probes"],
        })

        p3_1, p5_1 = (3.0, 4.0) if bad_small else (5.0, 4.0)
        obs1 = _observation(
            0.0125, p3=p3_1, p5=p5_1, q0=q0, qp=q0 - 0.008
        )
        obs2 = _observation(
            0.00625, p3=5.1, p5=4.1, q0=q0, qp=q0 - 0.006
        )
        raws.append({
            "prior_native_q0": q0,
            "epsilon_observations": [obs1, obs2],
        })
    return refs, raws


def test_synthetic_positive_rule() -> None:
    refs, raws = _synthetic_inputs()
    out = m.analyze_inputs(refs, raws)
    assert out["result"] == m.POSITIVE_RESULT
    assert out["core_qualitative_robustness_rule"][
        "all_new_epsilons_pass"
    ] is True
    assert out["core_qualitative_robustness_rule"]["p_value_count"] == 0


def test_synthetic_negative_rule() -> None:
    refs, raws = _synthetic_inputs(bad_small=True)
    out = m.analyze_inputs(refs, raws)
    assert out["result"] == m.NEGATIVE_RESULT
    gate = out["core_qualitative_robustness_rule"][
        "new_epsilon_gates"
    ]["0.0125"]
    assert gate["pass"] is False


def test_degeneracy_summary() -> None:
    out = m.distribution_summary([0.0, -1.0, 2.0, float("inf")])
    assert out["nonfinite_count"] == 1
    assert out["exact_zero_count"] == 1
    assert out["min"] == 0.0
    assert out["max"] == 2.0


def test_historical_reference_probe_reconstructs_numerator() -> None:
    epsilon = 0.025
    f_plus = 3.5
    f_minus = 1.5
    probe = {
        "direction_key": "P1_plus",
        "F_plus": f_plus,
        "F_minus": f_minus,
        "J": (f_plus - f_minus) / (2.0 * epsilon),
    }
    assert "central_difference_numerator" not in probe
    assert m.probe_numerator(probe, epsilon=epsilon) == 2.0


def test_new_probe_stored_numerator_is_checked() -> None:
    epsilon = 0.0125
    f_plus = 2.25
    f_minus = 1.75
    numerator = f_plus - f_minus
    probe = {
        "direction_key": "P1_plus",
        "F_plus": f_plus,
        "F_minus": f_minus,
        "central_difference_numerator": numerator,
        "J": numerator / (2.0 * epsilon),
    }
    assert m.probe_numerator(probe, epsilon=epsilon) == numerator


def test_no_inference_threshold_or_epsilon_selection() -> None:
    source = inspect.getsource(m).lower()
    forbidden = (
        "import scipy",
        "binomtest",
        "ttest",
        "multipletests",
        "holm",
        "alpha =",
        "p_value =",
        "epsilon_search",
        "best_epsilon",
        "selected_epsilon",
        "cosine_threshold",
        "support_threshold",
    )
    for token in forbidden:
        assert token not in source

    assert '"p_value_count_added": 0' in source
    assert '"inferential_test_count": 0' in source
    assert '"epsilon_selection_count": 0' in source


def test_direct_cli_help() -> None:
    repo_root = Path(m.__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "analyze_reason_router_gen4_small_epsilon_robustness.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--expected-head" in completed.stdout
    assert "--output-dir" in completed.stdout
