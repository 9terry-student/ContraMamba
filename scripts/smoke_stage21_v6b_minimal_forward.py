"""Smoke test for Stage21-A v6B-minimal forward pass and logit modulation.

Verifies:
1. Model import and instantiation work
2. Forward pass executes on synthetic batch
3. Alpha parameters are learnable (require_grad=True)
4. Temporal/predicate modulation changes final logits in expected direction
5. Predictions derived from final logits
6. No loss bypass keys (loss_logits, pairwise_logits)
7. final_logits_used flag is set
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contramamba import ContraMambaV5  # noqa: E402


def build_dummy_backbone(vocab_size: int = 256, hidden_size: int = 48, max_length: int = 64) -> nn.Module:
    """Lightweight dummy backbone matching train_controlled_v5 pattern."""

    class DummyBackbone(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int, max_length: int):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.token_embedding.weight.requires_grad_(False)
            self.position_embedding = nn.Embedding(max_length, hidden_size)
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )

        def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            positions = positions.unsqueeze(0).expand_as(input_ids)
            states = self.token_embedding(input_ids) + self.position_embedding(positions)
            return SimpleNamespace(last_hidden_state=self.encoder(states))

    return DummyBackbone(vocab_size, hidden_size, max_length)


def test_import():
    """Test that v6b_minimal can be imported."""
    print("TEST 1: Import v6b_minimal...")
    try:
        from contramamba.modeling_v6b_minimal import ContraMambaV6BMinimal  # noqa: E402, F401
        print("  [OK] Import successful")
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_instantiation():
    """Test that model can be instantiated."""
    print("TEST 2: Instantiate v6b_minimal model...")
    try:
        from contramamba.modeling_v6b_minimal import ContraMambaV6BMinimal  # noqa: E402

        backbone = build_dummy_backbone(vocab_size=256, hidden_size=48, max_length=64)
        model = ContraMambaV6BMinimal(
            backbone=backbone,
            frame_size=32,
            predicate_size=32,
            sufficiency_size=32,
            energy_size=24,
            dropout=0.0,
            decision_mode="explicit_product",
            use_temporal_comparator=True,
            use_predicate_comparator=True,
            alpha_temporal_init=1.25,
            alpha_predicate_init=1.25,
        )
        print("  [OK] Instantiation successful")
        return True, model, backbone
    except Exception as e:
        print(f"  [FAIL] Instantiation failed: {e}")
        return False, None, None


def test_forward_pass(model: nn.Module, backbone: nn.Module):
    """Test forward pass on synthetic batch."""
    print("TEST 3: Forward pass on synthetic batch...")
    try:
        batch_size = 4
        seq_len = 32

        # Synthetic inputs
        input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        claim_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        evidence_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

        # Set first 10 tokens to claim, rest to evidence
        claim_mask[:, :10] = True
        evidence_mask[:, 10:] = True

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            claim_mask=claim_mask,
            evidence_mask=evidence_mask,
        )

        assert "logits" in output, "output missing 'logits' key"
        assert "base_logits" in output, "output missing 'base_logits' key"
        assert "predictions" in output, "output missing 'predictions' key"
        assert output["logits"].shape == (batch_size, 3), f"wrong logits shape: {output['logits'].shape}"
        print(f"  [OK] Forward pass successful, logits shape={output['logits'].shape}")
        return True, output
    except Exception as e:
        print(f"  [FAIL] Forward pass failed: {e}")
        return False, None


def test_alpha_parameters(model: nn.Module):
    """Test that alpha parameters are learnable."""
    print("TEST 4: Alpha parameters are learnable...")
    try:
        assert model.alpha_temporal_raw is not None, "alpha_temporal_raw not initialized"
        assert model.alpha_predicate_raw is not None, "alpha_predicate_raw not initialized"
        assert model.alpha_temporal_raw.requires_grad, "alpha_temporal_raw not trainable"
        assert model.alpha_predicate_raw.requires_grad, "alpha_predicate_raw not trainable"

        alpha_t = model.alpha_temporal()
        alpha_p = model.alpha_predicate()
        assert isinstance(alpha_t, torch.Tensor), "alpha_temporal not a tensor"
        assert isinstance(alpha_p, torch.Tensor), "alpha_predicate not a tensor"

        # Check initialization near 1.25
        assert 1.0 < float(alpha_t) < 1.5, f"alpha_temporal={alpha_t} not near 1.25"
        assert 1.0 < float(alpha_p) < 1.5, f"alpha_predicate={alpha_p} not near 1.25"

        print(
            f"  [OK] Alphas initialized correctly: "
            f"alpha_temporal={float(alpha_t):.4f}, alpha_predicate={float(alpha_p):.4f}"
        )
        return True
    except Exception as e:
        print(f"  [FAIL] Alpha parameter test failed: {e}")
        return False


def test_logit_modulation(model: nn.Module, backbone: nn.Module):
    """Test that temporal/predicate modulation changes logits correctly."""
    print("TEST 5: Logit modulation with flags...")
    try:
        batch_size = 2
        seq_len = 32

        input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        claim_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        evidence_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        claim_mask[:, :10] = True
        evidence_mask[:, 10:] = True

        # Test temporal modulation
        temporal_flags = torch.tensor([1, 0], dtype=torch.long)
        output_temporal = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            claim_mask=claim_mask,
            evidence_mask=evidence_mask,
            temporal_mismatch_flags=temporal_flags,
        )

        alpha_t = float(model.alpha_temporal())

        # Check that modulation was applied to index 0 (first example, temporal flag active)
        base_logits = output_temporal["base_logits"]
        final_logits = output_temporal["logits"]

        # First example (flag=1): support should decrease, not_entitled should increase
        diff_support = final_logits[0, 0] - base_logits[0, 0]
        diff_not_entitled = final_logits[0, 1] - base_logits[0, 1]
        diff_refute = final_logits[0, 2] - base_logits[0, 2]

        assert abs(float(diff_support) - (-alpha_t)) < 1e-5, f"support not decreased by alpha: {diff_support}"
        assert abs(float(diff_not_entitled) - alpha_t) < 1e-5, f"not_entitled not increased by alpha: {diff_not_entitled}"
        assert abs(float(diff_refute) - (-alpha_t)) < 1e-5, f"refute not decreased by alpha: {diff_refute}"

        # Second example (flag=0): logits should be unchanged
        assert torch.allclose(
            final_logits[1], base_logits[1], atol=1e-6
        ), "logits changed when flag=0"

        assert output_temporal["temporal_flag_count"] == 1, "temporal_flag_count incorrect"
        assert output_temporal["final_logits_used"] is True, "final_logits_used not set"

        print(f"  [OK] Temporal modulation correct: support-={alpha_t:.4f}, not_entitled+={alpha_t:.4f}")

        # Test predicate modulation
        predicate_flags = torch.tensor([1, 0], dtype=torch.long)
        output_predicate = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            claim_mask=claim_mask,
            evidence_mask=evidence_mask,
            predicate_mismatch_flags=predicate_flags,
        )

        alpha_p = float(model.alpha_predicate())
        base_logits = output_predicate["base_logits"]
        final_logits = output_predicate["logits"]

        diff_support = final_logits[0, 0] - base_logits[0, 0]
        diff_not_entitled = final_logits[0, 1] - base_logits[0, 1]

        assert abs(float(diff_support) - (-alpha_p)) < 1e-5, f"predicate: support not decreased: {diff_support}"
        assert abs(float(diff_not_entitled) - alpha_p) < 1e-5, f"predicate: not_entitled not increased: {diff_not_entitled}"
        assert output_predicate["predicate_flag_count"] == 1, "predicate_flag_count incorrect"

        print(f"  [OK] Predicate modulation correct: support-={alpha_p:.4f}, not_entitled+={alpha_p:.4f}")
        return True
    except Exception as e:
        print(f"  [FAIL] Logit modulation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_predictions_from_final_logits(model: nn.Module, backbone: nn.Module):
    """Test that predictions are derived from final logits."""
    print("TEST 6: Predictions derived from final logits...")
    try:
        batch_size = 4
        seq_len = 32

        input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        claim_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        evidence_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        claim_mask[:, :10] = True
        evidence_mask[:, 10:] = True

        temporal_flags = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            claim_mask=claim_mask,
            evidence_mask=evidence_mask,
            temporal_mismatch_flags=temporal_flags,
        )

        # Verify predictions = argmax of final logits
        expected_predictions = output["logits"].argmax(dim=-1)
        actual_predictions = output["predictions"]

        assert torch.all(expected_predictions == actual_predictions), (
            f"predictions mismatch: expected {expected_predictions}, got {actual_predictions}"
        )
        print(f"  [OK] Predictions correctly derived from final logits")
        return True
    except Exception as e:
        print(f"  [FAIL] Predictions test failed: {e}")
        return False


def test_no_bypass_keys(output: dict[str, Any]):
    """Test that output dict has no loss bypass keys."""
    print("TEST 7: No loss bypass keys...")
    try:
        forbidden_keys = ["loss_logits", "pairwise_logits"]
        for key in forbidden_keys:
            assert key not in output, f"forbidden key '{key}' found in output"
        print("  [OK] No bypass keys (loss_logits, pairwise_logits)")
        return True
    except Exception as e:
        print(f"  [FAIL] Bypass key test failed: {e}")
        return False


def main() -> int:
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("STAGE21-A SMOKE TEST: v6B-minimal forward pass and logit modulation")
    print("=" * 70 + "\n")

    results: dict[str, bool] = {}

    results["import"] = test_import()
    success, model, backbone = test_instantiation()
    results["instantiation"] = success

    if success and model is not None and backbone is not None:
        success, output = test_forward_pass(model, backbone)
        results["forward_pass"] = success

        if success and output is not None:
            results["alpha_parameters"] = test_alpha_parameters(model)
            results["logit_modulation"] = test_logit_modulation(model, backbone)
            results["predictions"] = test_predictions_from_final_logits(model, backbone)
            results["no_bypass_keys"] = test_no_bypass_keys(output)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    for name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")

    if passed == total:
        print("\n[OK] ALL SMOKE TESTS PASSED")
        return 0
    else:
        print(f"\n[FAIL] {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
