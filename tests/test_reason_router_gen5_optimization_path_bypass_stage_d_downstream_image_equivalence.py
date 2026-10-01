from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reason_router_gen5_optimization_path_bypass_stage_d_downstream_image_equivalence.py"
)
SPEC = importlib.util.spec_from_file_location("stage_d", SCRIPT)
assert SPEC and SPEC.loader
sd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sd)


def test_gpu_queue_exact_matrix():
    sd.validate_gpu_queues()
    flat = tuple(sd.GPU_QUEUES[0]) + tuple(sd.GPU_QUEUES[1])
    assert len(flat) == 9
    assert set(flat) == set(sd.CELLS)
    assert len(sd.GPU_QUEUES[0]) == 5
    assert len(sd.GPU_QUEUES[1]) == 4


def test_final_b_basis_is_orthonormal_and_basis_invariant():
    g = torch.Generator().manual_seed(7)
    b = torch.randn(sd.STATE_WIDTH, 2, generator=g)
    q, audit = sd.orthonormal_basis_from_b(b)
    assert audit["effective_rank"] == 2
    assert torch.allclose(
        q.T @ q,
        torch.eye(2, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )

    rot, _ = torch.linalg.qr(
        torch.tensor([[2.0, 1.0], [-1.0, 2.0]], dtype=torch.float64)
    )
    q2, _ = sd.orthonormal_basis_from_b(b.to(torch.float64) @ rot)
    geo = sd._surface_geometry_one(q, q2)
    assert geo["rank2_interpretation_valid"]
    assert geo["affinity"] == pytest.approx(1.0, abs=1e-10)


def test_surface_geometry_orthogonal_planes():
    left = torch.zeros(8, 2, dtype=torch.float64)
    right = torch.zeros(8, 2, dtype=torch.float64)
    left[0, 0] = 1.0
    left[1, 1] = 1.0
    right[2, 0] = 1.0
    right[3, 1] = 1.0
    geo = sd._surface_geometry_one(left, right)
    assert geo["left_effective_rank"] == 2
    assert geo["right_effective_rank"] == 2
    assert geo["rank2_interpretation_valid"]
    assert geo["affinity"] == pytest.approx(0.0, abs=1e-12)
    assert geo["principal_angles_deg"] == pytest.approx([90.0, 90.0], abs=1e-10)


def test_rank_collapse_invalidates_rank2_interpretation():
    left = torch.zeros(8, 2, dtype=torch.float64)
    right = torch.zeros(8, 2, dtype=torch.float64)
    left[0, :] = torch.tensor([1.0, 2.0], dtype=torch.float64)
    right[0, 0] = 1.0
    right[1, 1] = 1.0
    geo = sd._surface_geometry_one(left, right)
    assert geo["left_effective_rank"] == 1
    assert geo["right_effective_rank"] == 2
    assert not geo["rank2_interpretation_valid"]
    assert len(geo["principal_cosines"]) == 1


def test_batch_geometry_matches_identity():
    g = torch.Generator().manual_seed(11)
    c0 = torch.randn(3, 10, generator=g)
    c1 = torch.randn(3, 10, generator=g)
    rows = sd.surface_geometry_batch([c0, c1], [c0, c1])
    assert len(rows) == 3
    for row in rows:
        assert row["affinity"] == pytest.approx(1.0, abs=1e-10)


def test_scale_consistency_exact_match():
    g = torch.Generator().manual_seed(13)
    c0 = torch.randn(4, 12, generator=g)
    c1 = torch.randn(4, 12, generator=g)
    rows = sd.scale_consistency_batch([c0, c1], [c0.clone(), c1.clone()])
    for row in rows:
        assert row["relative_frobenius_difference"] == pytest.approx(0.0, abs=1e-12)
        assert row["frobenius_cosine"] == pytest.approx(1.0, abs=1e-12)


def test_probe_constants_and_surfaces_are_frozen():
    assert sd.PRIMARY_RADIUS == 0.025
    assert sd.AUDIT_RADIUS == 0.05
    assert sd.AUDIT_ROWS == 32
    assert sd.EXPECTED_STRESSOR_ROWS == 480
    assert sd.PROBE_STREAM_ROWS == 16
    assert sd.SURFACES == (
        "final_backbone_hidden",
        "task_repr",
        "decision_primitives",
    )


def test_probe_radius_contract_accepts_signed_nonzero_radius():
    # The runtime central difference uses +r and -r; the low-level contribution
    # contract therefore validates magnitude rather than sign.
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'abs(float(radius)) > 0.0' in source


def test_baseline_trajectory_label_is_dev_eval_not_train_step0():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "PHASE3A_ZERO_CORRECTION_DEV_EVAL" in source
    assert "PHASE3A_STEP0_ZERO_CORRECTION" not in source
