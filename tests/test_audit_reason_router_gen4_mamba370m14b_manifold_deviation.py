from __future__ import annotations

import numpy as np
from scripts import audit_reason_router_gen4_mamba370m14b_manifold_deviation as subject

def test_native_nearest_neighbor_excludes_self() -> None:
    x = np.arange(600.0).reshape(300, 2)
    d, idx = subject.pairwise_native_distances(x)
    assert np.all(np.isfinite(d))
    assert np.all(idx != np.arange(300))
    assert np.all(d > 0)

def test_zero_delta_gives_unit_nn_ratio_and_zero_relative_displacement() -> None:
    native = np.arange(1200.0).reshape(300, 4) / 100.0
    strong = native[:, :2]
    delta = np.zeros_like(native)
    out = subject.audit_condition(
        native,
        strong,
        delta,
        np.array([0, 1], dtype=np.int64),
    )
    assert out["R_rel_full"]["max"] == 0.0
    assert out["R_rel_strong"]["max"] == 0.0
    assert abs(out["R_NN_full"]["mean"] - 1.0) < 1e-12
    assert abs(out["R_NN_strong"]["mean"] - 1.0) < 1e-12
