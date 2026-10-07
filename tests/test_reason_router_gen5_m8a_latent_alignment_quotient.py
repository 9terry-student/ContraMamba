from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_reason_router_gen5_m8a_latent_alignment_quotient.py"
)
SPEC = importlib.util.spec_from_file_location("m8a_mod", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def random_full_rank_a(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    a = torch.randn((2, 768), generator=generator, dtype=torch.float64)
    assert torch.linalg.matrix_rank(a) == 2
    return a


def test_ordered_pair_orientation_recovers_known_transport():
    donor = random_full_rank_a(1)
    true_t = torch.tensor([[1.2, -0.3], [0.4, 0.8]], dtype=torch.float64)
    recipient = true_t @ donor
    fitted = mod.fit_gl_transport(recipient, donor)
    assert torch.allclose(fitted, true_t, atol=1e-11, rtol=1e-11)
    assert torch.allclose(fitted @ donor, recipient, atol=1e-11, rtol=1e-11)


def test_least_squares_is_exact_for_coordinate_reparameterization():
    donor = random_full_rank_a(2)
    true_t = torch.tensor([[0.8, 0.2], [-0.1, 1.1]], dtype=torch.float64)
    recipient = true_t @ donor
    fitted = mod.fit_gl_transport(recipient, donor)
    residual = mod.normalized_matrix_residual(recipient, fitted @ donor)
    assert residual < 1e-11


def test_orthogonal_procrustes_recovers_orthogonal_transport():
    donor = random_full_rank_a(3)
    q = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    recipient = q @ donor
    fitted = mod.fit_orthogonal_transport(recipient, donor)
    assert torch.allclose(fitted.T @ fitted, torch.eye(2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(fitted @ donor, recipient, atol=1e-11, rtol=1e-11)


def test_gram_weighted_residual_matches_materialized_action():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4)
    x = torch.randn((25, 768), generator=generator, dtype=torch.float64)
    gram = x.T @ x
    left = torch.randn((2, 768), generator=generator, dtype=torch.float64)
    right = torch.randn((2, 768), generator=generator, dtype=torch.float64)
    observed = mod.normalized_matrix_residual(left, right, gram)
    lx = x @ left.T
    rx = x @ right.T
    expected = torch.linalg.norm(lx - rx) / torch.sqrt(
        0.5 * (torch.linalg.norm(lx) ** 2 + torch.linalg.norm(rx) ** 2)
    )
    assert observed == pytest.approx(float(expected.item()), rel=1e-11, abs=1e-11)


def test_b_transport_direction_is_b_times_inverse_transport():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(5)
    donor_a = random_full_rank_a(5)
    t = torch.tensor([[1.3, 0.2], [-0.4, 0.9]], dtype=torch.float64)
    recipient_a = t @ donor_a
    donor_b = torch.randn((24576, 2), generator=generator, dtype=torch.float64)
    aligned_b, meta = mod.transport_b_to_recipient(donor_b, t)
    assert meta["transport_full_rank"] is True
    assert torch.allclose(
        aligned_b @ recipient_a,
        donor_b @ donor_a,
        atol=1e-10,
        rtol=1e-10,
    )


def test_inverse_and_pseudoinverse_agree_for_well_conditioned_transport():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(6)
    b = torch.randn((24576, 2), generator=generator, dtype=torch.float64)
    t = torch.tensor([[1.0, 0.2], [-0.1, 0.9]], dtype=torch.float64)
    _aligned, meta = mod.transport_b_to_recipient(b, t)
    assert meta["transport_full_rank"] is True
    assert meta["inverse_pinv_relative_difference"] is not None
    assert meta["inverse_pinv_relative_difference"] < 1e-10


def test_rank_deficient_transport_uses_pinv_without_direct_inverse_claim():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    b = torch.randn((24576, 2), generator=generator, dtype=torch.float64)
    t = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    aligned, meta = mod.transport_b_to_recipient(b, t)
    assert aligned.shape == b.shape
    assert meta["transport_full_rank"] is False
    assert meta["inverse_pinv_relative_difference"] is None


def test_operator_residual_low_rank_identity_matches_materialization():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(8)
    # Use a reduced B width for the algebraic helper; it does not enforce width.
    b1 = torch.randn((17, 2), generator=generator, dtype=torch.float64)
    b2 = torch.randn((17, 2), generator=generator, dtype=torch.float64)
    a1 = torch.randn((2, 768), generator=generator, dtype=torch.float64)
    a2 = torch.randn((2, 768), generator=generator, dtype=torch.float64)
    observed = mod.operator_normalized_residual(b1, a1, b2, a2)
    o1 = b1 @ a1
    o2 = b2 @ a2
    expected = torch.linalg.norm(o1 - o2) / torch.sqrt(
        0.5 * (torch.linalg.norm(o1) ** 2 + torch.linalg.norm(o2) ** 2)
    )
    assert observed == pytest.approx(float(expected.item()), rel=1e-11, abs=1e-11)


def test_operator_task_residual_matches_explicit_task_action():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9)
    x = torch.randn((19, 768), generator=generator, dtype=torch.float64)
    gram = x.T @ x
    b1 = torch.randn((13, 2), generator=generator, dtype=torch.float64)
    b2 = torch.randn((13, 2), generator=generator, dtype=torch.float64)
    a1 = torch.randn((2, 768), generator=generator, dtype=torch.float64)
    a2 = torch.randn((2, 768), generator=generator, dtype=torch.float64)
    observed = mod.operator_normalized_residual(b1, a1, b2, a2, gram)
    y1 = x @ a1.T @ b1.T
    y2 = x @ a2.T @ b2.T
    expected = torch.linalg.norm(y1 - y2) / torch.sqrt(
        0.5 * (torch.linalg.norm(y1) ** 2 + torch.linalg.norm(y2) ** 2)
    )
    assert observed == pytest.approx(float(expected.item()), rel=1e-10, abs=1e-10)


def test_source_hash_verification_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "source.bin"
    source.write_bytes(b"wrong")
    monkeypatch.setattr(mod, "ROOT", root)
    with pytest.raises(mod.M8AError, match="M8A_TEST_SHA"):
        mod._verify_source_file(Path("source.bin"), "0" * 64, "TEST")


def test_transport_fit_has_no_m7_output_input_or_reference():
    signature = inspect.signature(mod.fit_gl_transport)
    assert list(signature.parameters) == ["a_rec", "a_don"]
    source = inspect.getsource(mod.fit_gl_transport).lower()
    assert "logit" not in source
    assert "interaction" not in source
    assert "affinity" not in source
    assert "m7" not in source


def test_task_gram_extractor_authenticates_shape_count_and_metadata():
    gram = torch.eye(768, dtype=torch.float64)
    payload = {
        "schema_version": "GEN5_TASK_REACHABLE_STATE_GRAM_V1",
        "gram": gram,
        "valid_token_count": 60094,
        "dev_rows": 840,
        "dev_encoding_sha256": mod.DEV_ENCODING_SHA256,
        "dev_order_sha256": mod.DEV_ORDER_SHA256,
        "execution_commit": mod.TASK_QUOTIENT_EXECUTION_COMMIT,
        "arm": "G5-C0",
        "pressure": "P0",
    }
    extracted, meta = mod.extract_task_gram(payload)
    assert torch.equal(extracted, gram)
    assert meta["valid_token_count"] == 60094


def test_task_gram_extractor_fails_closed_on_wrong_token_count():
    payload = {
        "schema_version": "GEN5_TASK_REACHABLE_STATE_GRAM_V1",
        "gram": torch.eye(768, dtype=torch.float64),
        "valid_token_count": 1,
    }
    with pytest.raises(mod.M8AError, match="M8A_TASK_GRAM_TOKEN_COUNT"):
        mod.extract_task_gram(payload)


def test_structural_predicates_do_not_encode_scientific_cutoff():
    source = inspect.getsource(mod.summarize_structural_predicates)
    assert "0.1" not in source
    assert "0.2" not in source
    assert "0.5" not in source
    assert "threshold" not in source.lower()
    assert "DEFER_TO_VALIDATED_STATIC_EVIDENCE_INTERPRETATION" in source


def test_static_verify_mode_forbids_output_and_freeze_args():
    args = mod.argparse.Namespace(
        static_verify_only=True,
        run_static_audit=False,
        expected_head="x" * 40,
        implementation_freeze_commit=None,
        output_root=None,
    )
    mod.validate_args(args)

    args.output_root = Path("bad")
    with pytest.raises(mod.M8AError, match="M8A_STATIC_VERIFY_OUTPUT_FORBIDDEN"):
        mod.validate_args(args)


def test_run_static_audit_requires_pinned_freeze_head():
    args = mod.argparse.Namespace(
        static_verify_only=False,
        run_static_audit=True,
        expected_head="a" * 40,
        implementation_freeze_commit="b" * 40,
        output_root=Path("out"),
    )
    with pytest.raises(mod.M8AError, match="M8A_IMPLEMENTATION_FREEZE_HEAD_MISMATCH"):
        mod.validate_args(args)


def test_implementation_contains_no_forward_cuda_autograd_training_calls():
    source = MODULE_PATH.read_text(encoding="utf-8")
    forbidden = [
        ".backward(",
        "torch.optim",
        "optimizer.step(",
        "torch.cuda",
        "autograd.grad",
        ".forward(",
    ]
    for token in forbidden:
        assert token not in source
