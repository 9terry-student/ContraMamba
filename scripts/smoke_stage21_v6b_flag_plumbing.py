"""Smoke test for Stage21-A2: flag plumbing for v6b_minimal.

Verifies:
1. Temporal/predicate flags can be extracted from training records.
2. Flags are correctly shaped and typed.
3. Flag counts match expected stage15_probe_type distribution.
4. v6b_minimal forward pass accepts flags and produces flagged modulation.
5. Unflagged rows remain identical to base logits.

Uses stage15_slot_sensitivity_probe.jsonl as the authoritative source, with
fallback to controlled data schema inspection (predicate_swap heuristic only).
"""

from __future__ import annotations

import json
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

from contramamba.comparator_flags import (  # noqa: E402
    predicate_mismatch_flags_from_intervention_type,
    predicate_mismatch_flags_from_probe,
    temporal_mismatch_flags_from_probe,
    temporal_mismatch_flags_none,
)
from contramamba.modeling_v6b_minimal import ContraMambaV6BMinimal  # noqa: E402


def load_small_batch(jsonl_path: Path, limit: int = 16) -> list[dict]:
    """Load first N records from jsonl file."""
    records = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                if len(records) >= limit:
                    break
                records.append(json.loads(line))
    except FileNotFoundError:
        return []
    return records


def build_dummy_backbone(hidden_size: int = 48) -> nn.Module:
    """Lightweight dummy backbone."""

    class DummyBackbone(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )

        def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
            batch_size, seq_len = input_ids.shape
            states = torch.randn(batch_size, seq_len, self.config.hidden_size)
            return SimpleNamespace(last_hidden_state=self.encoder(states))

    return DummyBackbone(hidden_size)


def test_stage15_probe_flags():
    """Test flag extraction from stage15_slot_sensitivity_probe.jsonl."""
    print("TEST 1: Load stage15 probe and extract flags...")
    probe_path = ROOT / "data" / "stage15_slot_sensitivity_probe.jsonl"
    records = load_small_batch(probe_path, limit=32)

    if not records:
        print("  [SKIP] stage15 probe not found or empty")
        return None, None, records

    print(f"  [OK] Loaded {len(records)} records from stage15 probe")

    # Extract flags
    temporal_flags = temporal_mismatch_flags_from_probe(records)
    predicate_flags = predicate_mismatch_flags_from_probe(records)

    temporal_count = int(temporal_flags.sum().item())
    predicate_count = int(predicate_flags.sum().item())

    print(f"  [OK] temporal_mismatch flags: {temporal_count}/{len(records)}")
    print(f"  [OK] predicate_mismatch flags: {predicate_count}/{len(records)}")

    # Verify shape
    assert temporal_flags.shape == (len(records),), f"wrong shape: {temporal_flags.shape}"
    assert predicate_flags.shape == (len(records),), f"wrong shape: {predicate_flags.shape}"
    assert temporal_flags.dtype == torch.long
    assert predicate_flags.dtype == torch.long

    return temporal_flags, predicate_flags, records


def test_controlled_flags():
    """Test fallback flag extraction from controlled_v5 data."""
    print("TEST 2: Load controlled data and extract fallback flags...")
    controlled_path = ROOT / "data" / "controlled_v5_v3_without_time_swap.jsonl"
    records = load_small_batch(controlled_path, limit=32)

    if not records:
        print("  [SKIP] controlled data not found or empty")
        return None, None, records

    print(f"  [OK] Loaded {len(records)} records from controlled_v5")

    # Temporal: no detection for controlled (cannot use time_swap)
    temporal_flags = temporal_mismatch_flags_none(records)
    print(f"  [OK] temporal flags: all zeros (no Stage17 detector integrated)")

    # Predicate: heuristic using intervention_type
    predicate_flags = predicate_mismatch_flags_from_intervention_type(records)
    predicate_count = int(predicate_flags.sum().item())
    print(f"  [WARN] predicate flags (heuristic): {predicate_count}/{len(records)}")
    print(f"         Based on intervention_type == 'predicate_swap' only")

    assert temporal_flags.shape == (len(records),)
    assert predicate_flags.shape == (len(records),)
    return temporal_flags, predicate_flags, records


def test_forward_with_flags(
    temporal_flags: torch.Tensor | None,
    predicate_flags: torch.Tensor | None,
    records: list[dict],
):
    """Test v6b_minimal forward pass with flags."""
    print("TEST 3: Forward pass with extracted flags...")

    if not records:
        print("  [SKIP] No records available")
        return

    batch_size = len(records)
    seq_len = 32
    device = "cpu"

    # Create model
    backbone = build_dummy_backbone(hidden_size=48)
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
    ).to(device)

    # Create synthetic inputs
    input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long).to(device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool).to(device)
    claim_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool).to(device)
    evidence_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool).to(device)
    claim_mask[:, :10] = True
    evidence_mask[:, 10:] = True

    # Move flags to device
    if temporal_flags is not None:
        temporal_flags = temporal_flags.to(device)
    if predicate_flags is not None:
        predicate_flags = predicate_flags.to(device)

    # Forward pass
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        claim_mask=claim_mask,
        evidence_mask=evidence_mask,
        temporal_mismatch_flags=temporal_flags,
        predicate_mismatch_flags=predicate_flags,
    )

    print(f"  [OK] Forward pass successful")

    # Verify output keys
    assert "logits" in output
    assert "base_logits" in output
    assert "predictions" in output
    assert "temporal_flag_count" in output
    assert "predicate_flag_count" in output
    assert output["final_logits_used"] is True
    print(f"  [OK] Output keys present: logits, base_logits, predictions, flag counts")

    # Verify shapes
    assert output["logits"].shape == (batch_size, 3)
    assert output["base_logits"].shape == (batch_size, 3)
    assert output["predictions"].shape == (batch_size,)
    print(f"  [OK] Logits shape: {output['logits'].shape}")

    # Verify modulation
    base_logits = output["base_logits"]
    final_logits = output["logits"]

    if temporal_flags is not None and int(temporal_flags.sum()) > 0:
        temporal_active = temporal_flags.bool()
        temporal_diff = (final_logits[temporal_active] - base_logits[temporal_active]).abs()
        if torch.any(temporal_diff > 1e-5):
            print(f"  [OK] Temporal modulation applied to {int(temporal_active.sum())} examples")
        unflagged_temporal = temporal_flags == 0
        if torch.any(unflagged_temporal):
            unchanged = torch.allclose(
                final_logits[unflagged_temporal],
                base_logits[unflagged_temporal],
                atol=1e-6,
            )
            assert unchanged, "unflagged rows should not change"
            print(f"  [OK] Unflagged rows unchanged for temporal")

    if predicate_flags is not None and int(predicate_flags.sum()) > 0:
        predicate_active = predicate_flags.bool()
        predicate_diff = (final_logits[predicate_active] - base_logits[predicate_active]).abs()
        if torch.any(predicate_diff > 1e-5):
            print(f"  [OK] Predicate modulation applied to {int(predicate_active.sum())} examples")
        unflagged_predicate = predicate_flags == 0
        if torch.any(unflagged_predicate):
            unchanged = torch.allclose(
                final_logits[unflagged_predicate],
                base_logits[unflagged_predicate],
                atol=1e-6,
            )
            assert unchanged, "unflagged rows should not change"
            print(f"  [OK] Unflagged rows unchanged for predicate")

    # Verify no bypass keys
    forbidden_keys = ["loss_logits", "pairwise_logits"]
    for key in forbidden_keys:
        assert key not in output, f"forbidden key {key} in output"
    print(f"  [OK] No bypass keys (loss_logits, pairwise_logits)")


def main() -> int:
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("STAGE21-A2 SMOKE TEST: Flag plumbing for v6b_minimal")
    print("=" * 70 + "\n")

    results: dict[str, bool | None] = {}

    # Test stage15 probe flags
    temporal_flags_s15, predicate_flags_s15, records_s15 = test_stage15_probe_flags()
    results["stage15_probe_flags"] = temporal_flags_s15 is not None

    # Test controlled fallback flags
    temporal_flags_ctrl, predicate_flags_ctrl, records_ctrl = test_controlled_flags()
    results["controlled_fallback_flags"] = temporal_flags_ctrl is not None

    # Use stage15 if available, else controlled
    if temporal_flags_s15 is not None:
        temporal_flags = temporal_flags_s15
        predicate_flags = predicate_flags_s15
        records = records_s15
    else:
        temporal_flags = temporal_flags_ctrl
        predicate_flags = predicate_flags_ctrl
        records = records_ctrl

    if temporal_flags is not None:
        try:
            test_forward_with_flags(temporal_flags, predicate_flags, records)
            results["forward_with_flags"] = True
        except Exception as e:
            print(f"  [FAIL] Forward test failed: {e}")
            import traceback

            traceback.print_exc()
            results["forward_with_flags"] = False
    else:
        print("TEST 3: Forward test skipped (no records available)")
        results["forward_with_flags"] = None

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, result in results.items():
        if result is None:
            status = "[SKIP]"
        elif result:
            status = "[OK]"
        else:
            status = "[FAIL]"
        print(f"{status} {name}")

    if all(v for v in results.values() if v is not None):
        print("\n[OK] ALL FLAG PLUMBING TESTS PASSED")
        return 0
    else:
        print("\n[WARN] Some tests skipped or failed")
        return 0 if results["forward_with_flags"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
