from __future__ import annotations

import numpy as np

from scripts import (
    audit_reason_router_gen4_mamba370m14b_low_displacement
    as subject,
)


def test_native_nearest_neighbor_excludes_self() -> None:
    x = np.arange(600.0).reshape(300, 2)
    d, idx = subject.pairwise_native_distances(x)
    assert np.all(np.isfinite(d))
    assert np.all(idx != np.arange(300))
    assert np.all(d > 0.0)


def test_zero_delta_has_zero_relative_and_unit_nn_ratio() -> None:
    native = np.arange(1200.0).reshape(300, 4) / 100.0
    strong = native[:, :2]
    delta = np.zeros_like(native)
    out, vectors = subject.audit_alpha(
        native,
        strong,
        delta,
        np.array([0, 1], dtype=np.int64),
        alpha=0.25,
    )
    assert out["R_rel_full"]["max"] == 0.0
    assert out["R_rel_strong"]["max"] == 0.0
    assert abs(out["R_NN_full"]["mean"] - 1.0) < 1e-12
    assert abs(out["R_NN_strong"]["mean"] - 1.0) < 1e-12
    assert np.all(vectors["R_rel_full"] == 0.0)


def test_relative_displacement_scales_exactly_for_binary_alphas() -> None:
    rng = np.random.default_rng(0)
    native = rng.normal(size=(300, 8))
    strong_indices = np.array([0, 2, 4, 6], dtype=np.int64)
    strong = native[:, strong_indices]
    delta = rng.normal(size=(300, 8))

    _, v1 = subject.audit_alpha(
        native, strong, delta, strong_indices, alpha=1.0
    )
    _, v05 = subject.audit_alpha(
        native, strong, delta, strong_indices, alpha=0.5
    )
    _, v025 = subject.audit_alpha(
        native, strong, delta, strong_indices, alpha=0.25
    )

    for metric in ("R_rel_full", "R_rel_strong"):
        assert np.allclose(v05[metric], 0.5 * v1[metric], rtol=1e-12, atol=1e-12)
        assert np.allclose(v025[metric], 0.25 * v1[metric], rtol=1e-12, atol=1e-12)


def test_audit_has_zero_inference_contract() -> None:
    assert subject.ALPHAS == (1.0, 0.5, 0.25)
    assert subject.RESULT == "PASS_GEN4_LOW_DISPLACEMENT_STATIC_DISPLACEMENT_AUDIT"
