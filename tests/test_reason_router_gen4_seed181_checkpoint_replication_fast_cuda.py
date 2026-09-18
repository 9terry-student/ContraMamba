from __future__ import annotations

import math

import torch

from scripts import reason_router_gen4_seed181_checkpoint_replication_fast_cuda as mod


def test_checkpoint_identity_is_seed181_same_arm() -> None:
    assert mod.SEED == 181
    assert mod.ARM == "G3-GROUP-D-HALF"
    assert mod.CHECKPOINT_SHA256 == (
        "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f"
    )
    assert mod.TOTAL_FORWARD_BUDGET == 50400


def test_projector_modes_exact_spectrum_on_synthetic_principal_pairs() -> None:
    d = 12
    k = 5
    b2 = torch.zeros((d, k), dtype=torch.float64)
    b4 = torch.zeros((d, k), dtype=torch.float64)
    angles = [0.2, 0.35, 0.5, 0.7, 0.9]
    for i, theta in enumerate(angles):
        b2[2 * i, i] = 1.0
        b4[2 * i, i] = math.cos(theta)
        b4[2 * i + 1, i] = math.sin(theta)

    result = mod.projector_modes(b2, b4)
    observed = list(result["positive_eigenvalues"])
    assert result["principal_mode_gram_max_abs_residual"] < 1e-10
    assert result["projector_contrast_reconstruction_max_abs_residual"] < 1e-10
    assert result["minimum_positive_eigengap"] > mod.EIG_TOL
    expected = sorted([math.sin(x) for x in angles])
    # torch.linalg.svd sorts singular values descending, hence theta ascending here.
    assert len(observed) == 5
    for left, right in zip(observed, expected, strict=True):
        assert abs(left - right) < 1.0e-10

    planes = result["planes"]
    z = torch.stack(
        [planes[f"P{i}_plus"] for i in range(1, 6)]
        + [planes[f"P{i}_minus"] for i in range(1, 6)],
        dim=1,
    )
    assert torch.max(torch.abs(z.T @ z - torch.eye(10, dtype=torch.float64))) < 1e-10


def test_homolog_rule_is_response_blind_projector_overlap(monkeypatch) -> None:
    d = 8
    planes = {}
    for i in range(1, 5):
        planes[f"P{i}_plus"] = torch.eye(d, dtype=torch.float64)[:, 2 * (i - 1)]
        planes[f"P{i}_minus"] = torch.eye(d, dtype=torch.float64)[:, 2 * (i - 1) + 1]
    # P5 reuses a rotated copy inside P4 only for this tiny matching-unit test.
    planes["P5_plus"] = (planes["P4_plus"] + planes["P4_minus"]) / math.sqrt(2.0)
    planes["P5_minus"] = (planes["P4_plus"] - planes["P4_minus"]) / math.sqrt(2.0)

    frozen = {
        "pp3_plus": planes["P2_plus"].clone(),
        "pp3_minus": planes["P2_minus"].clone(),
        "pp5_plus": planes["P4_plus"].clone(),
        "pp5_minus": planes["P4_minus"].clone(),
    }
    monkeypatch.setattr(mod.restoration, "load_planes", lambda: frozen)
    eigenvalues = [0.2, 0.3, 0.4, 0.9, 0.8]
    result = mod.match_homolog(planes, eigenvalues)
    assert result["homolog_plane_index"] == 2
    assert result["control_plane_index"] == 4
    assert abs(result["homolog_overlap"] - 2.0) < 1e-12


def test_seed181_binding_restores_historical_identity() -> None:
    extraction = mod.restoration.holdout.phase1.base.prevalence_eq.extraction
    before = (
        extraction.REPRESENTATIVE_SEED,
        extraction.REPRESENTATIVE_ARM,
        extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
    )
    with mod.seed181_binding():
        assert extraction.REPRESENTATIVE_SEED == 181
        assert extraction.REPRESENTATIVE_ARM == "G3-GROUP-D-HALF"
        assert extraction.REPRESENTATIVE_CHECKPOINT_SHA256 == mod.CHECKPOINT_SHA256
    after = (
        extraction.REPRESENTATIVE_SEED,
        extraction.REPRESENTATIVE_ARM,
        extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
    )
    assert after == before


def test_authenticate_repo_accepts_detached_exact_commit(monkeypatch) -> None:
    expected_head = "0123456789abcdef0123456789abcdef01234567"

    values = {
        ("branch", "--show-current"): "",
        ("rev-parse", "HEAD"): expected_head,
        ("status", "--porcelain"): "",
    }

    def fake_git(*args: str) -> str:
        return values[args]

    monkeypatch.setattr(mod, "git", fake_git)
    monkeypatch.setattr(
        mod.subprocess,
        "call",
        lambda *args, **kwargs: 0,
    )

    mod.authenticate_repo(expected_head)


def test_replication_restoration_endpoint_accepts_float_roundoff() -> None:
    q_b = 2.205616652956685e-07
    q_r3 = -1.536534738677208e-07
    q_r5 = -1.0487738235156595e-07

    expanded = (q_r3 - q_b) - (q_r5 - q_b)
    direct = q_r3 - q_r5

    # This realistic-scale example is mathematically identical but
    # intentionally not bit-exact under IEEE-754 evaluation order.
    assert expanded != direct

    row = mod.replication_restoration_endpoint(
        q_b,
        q_r3,
        q_r5,
    )

    assert row["S3"] == q_r3 - q_b
    assert row["S5"] == q_r5 - q_b
    assert row["D_SUF"] == expanded

    algebra_ulp = max(
        math.ulp(value)
        for value in (
            q_b,
            q_r3,
            q_r5,
            row["S3"],
            row["S5"],
            row["D_SUF"],
            direct,
        )
    )

    assert (
        abs(row["D_SUF"] - direct)
        <= mod.FLOAT_IDENTITY_ULPS * algebra_ulp
    )

    historical = mod.restoration.endpoint

    with mod.replication_restoration_endpoint_binding():
        assert (
            mod.restoration.endpoint
            is mod.replication_restoration_endpoint
        )
        rebound = mod.restoration.endpoint(
            q_b,
            q_r3,
            q_r5,
        )
        assert rebound["D_SUF"] == expanded

    assert mod.restoration.endpoint is historical
