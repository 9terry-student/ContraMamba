from __future__ import annotations

import inspect
import math

from scripts import (
    reason_router_gen4_small_epsilon_robustness_fast_cuda
    as m,
)


def test_protocol_and_forward_budget_contract() -> None:
    m.validate_protocol()
    m.validate_shards()

    assert m.REQUIRED_ANCESTOR == (
        "7254f89c352de5e6c6594e43d01a6a49ea5aeeb4"
    )
    assert m.REFERENCE_EPSILON == 0.025
    assert m.NEW_EPSILONS == (0.0125, 0.00625)
    assert m.F_DIRECTION == 4
    assert m.F_PAIR_PER_EPSILON == 40
    assert m.F_PAIR == 80
    assert m.F_TOTAL_PER_EPSILON == 12000
    assert m.F_TOTAL == 24000
    assert sum(s["forward_budget"] for s in m.SHARDS) == 24000
    assert m.SHARDS[0]["forward_budget"] == 12000
    assert m.SHARDS[1]["forward_budget"] == 12000


def test_design_and_reference_identity_constants() -> None:
    assert m.DESIGN_ARTIFACT_GIT_BLOB == (
        "b0e67c1d0aae2df08ae22d44d411b455db1c3cbd"
    )
    assert m.REFERENCE_ITEMS_GIT_BLOB == (
        "edd46066805082734f624a54736df8a6b82544e5"
    )
    assert m.REFERENCE_SUMMARY_GIT_BLOB == (
        "996cc03c15d7cc8a04109325e35d6b0512e00ea2"
    )
    assert m.REFERENCE_MANIFEST_GIT_BLOB == (
        "4967aca0c5634d136867e500548c757489053b6b"
    )
    assert m.REFERENCE_ITEMS_SHA256 == (
        "8db506d872ff81e72b08d44f9ff0af907cb1a65086c0071e3e365a75bd166e17"
    )
    assert m.REFERENCE_SUMMARY_SHA256 == (
        "7f9e01dd28ffb2d2a286b464aa4701c639e68f265e11de9eef67ca1e3c1e448e"
    )


def test_frozen_reference_artifact_validates() -> None:
    validated = m.validate_reference_artifact()
    assert len(validated["items"]) == 300
    summary = validated["summary"]
    assert summary["epsilon"] == 0.025
    assert summary["scientific_model_forward_count_this_run"] == 12000
    assert summary["primary_inference_executed"] is False
    assert summary["scientific_conclusion"] is None


def test_direction_order_is_unchanged() -> None:
    assert m.PLANE_ORDER == ("P1", "P2", "P3", "P4", "P5")
    assert m.DIRECTION_ORDER == (
        "P1_plus", "P1_minus",
        "P2_plus", "P2_minus",
        "P3_plus", "P3_minus",
        "P4_plus", "P4_minus",
        "P5_plus", "P5_minus",
    )


def test_central_difference_uses_explicit_epsilon() -> None:
    f_plus = 1.25
    f_minus = 0.75

    assert m.central_difference(f_plus, f_minus, 0.0125) == 20.0
    assert m.central_difference(f_plus, f_minus, 0.00625) == 40.0


def test_decomposition_formula_is_reference_formula() -> None:
    j = {
        key: float(index + 1) / 10.0
        for index, key in enumerate(m.DIRECTION_ORDER)
    }
    eigenvalues = (0.2, 0.3, 0.4, 0.5, 0.6)
    q0 = 0.125

    observed = m.reference.decomposition_from_j(j, eigenvalues, q0)

    expected = {}
    for index, plane in enumerate(m.PLANE_ORDER):
        jp = j[f"{plane}_plus"]
        jm = j[f"{plane}_minus"]
        expected[plane] = (
            eigenvalues[index] * (jp * jp - jm * jm) / 5.0
        )

    assert observed["plane_contributions"] == expected
    assert observed["Q_principal"] == math.fsum(expected.values())


def test_principal_geometry_is_unchanged() -> None:
    planes, eigenvalues = m.reference.principal_geometry()
    assert len(eigenvalues) == 5
    for actual, expected in zip(
        eigenvalues,
        m.EXPECTED_EIGENVALUES,
        strict=True,
    ):
        assert abs(actual - expected) <= 2.0e-12

    assert set(planes) >= {
        "p1_plus", "p1_minus",
        "p2_plus", "p2_minus",
        "pp3_plus", "pp3_minus",
        "p4_plus", "p4_minus",
        "p5_plus", "p5_minus",
    }


def test_no_module_global_epsilon_mutation() -> None:
    source = inspect.getsource(m)
    assert "reference.base.EPS =" not in source
    assert "reference.EPS =" not in source
    assert "epsilon=epsilon" in source
    assert "* float(epsilon)" in source


def test_raw_runner_has_no_inference_or_epsilon_selection() -> None:
    source = inspect.getsource(m).lower()

    forbidden = (
        "import scipy",
        "binomtest",
        "ttest",
        "multipletests",
        "holm",
        "alpha =",
        "support_threshold",
        "epsilon_grid",
        "epsilon_search",
        "best_epsilon",
        "selected_epsilon",
    )
    for token in forbidden:
        assert token not in source

    assert 'p_value_count_added": 0' in source
    assert 'primary_inference_executed": false' in source


def test_direct_cli_help() -> None:
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(m.__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "reason_router_gen4_small_epsilon_robustness_fast_cuda.py"
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
    assert "--model-snapshot" in completed.stdout
    assert "--tokenizer-snapshot" in completed.stdout
    assert "--checkpoint" in completed.stdout
    assert "--output-dir" in completed.stdout
