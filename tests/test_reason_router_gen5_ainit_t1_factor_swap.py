from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from scripts import audit_reason_router_gen5_ainit_t1_factor_swap as mod


def test_factor_swap_state_counts_and_partition():
    assert len(mod.ALL_HYBRIDS) == 27
    assert len(mod.CROSS_HYBRIDS) == 18
    assert sum(1 for state in mod.ALL_HYBRIDS if state[0] == state[1]) == 9
    assert mod.WORKER_RANGES == {0: (0, 420), 1: (420, 840)}
    assert len(mod.worker_batches(0)) == 14
    assert len(mod.worker_batches(1)) == 14
    assert mod.worker_batches(0)[0] == (0, 32)
    assert mod.worker_batches(0)[-1] == (416, 420)
    assert mod.worker_batches(1)[0] == (420, 452)
    assert mod.worker_batches(1)[-1] == (836, 840)


def test_factor_swap_planned_forward_counts():
    plan = mod.planned_counts()
    assert plan["row_batch_size"] == 32
    assert plan["all_states"] == 27
    assert plan["new_cross_states"] == 18
    assert plan["full_matched_anchor_recompute"] == 0
    assert plan["batches_per_worker"] == {"0": 14, "1": 14}
    assert plan["cross_forward_calls_per_worker"] == {"0": 252, "1": 252}
    assert plan["cross_forward_calls_total"] == 504


@pytest.mark.parametrize("axis", ["recipient_A", "donor_A", "donor_R"])
def test_factor_axis_pair_count(axis):
    pairs = mod.factor_axis_pairs(axis)
    assert len(pairs) == 27
    assert len(set(pairs)) == 27


def test_cross_hybrids_are_exactly_cross_a():
    assert all(a_rec != a_don for a_rec, a_don, _r in mod.CROSS_HYBRIDS)
    expected = {
        mod.hybrid_name(*state)
        for state in mod.ALL_HYBRIDS
        if state[0] != state[1]
    }
    assert expected == {mod.hybrid_name(*state) for state in mod.CROSS_HYBRIDS}


def test_affinity_sign_convention():
    recipient = torch.tensor([[0.0, 0.0]])
    donor = torch.tensor([[10.0, 0.0]])
    near_recipient = torch.tensor([[1.0, 0.0]])
    near_donor = torch.tensor([[9.0, 0.0]])
    rec = mod._coordinate_affinity(near_recipient, recipient, donor)
    don = mod._coordinate_affinity(near_donor, recipient, donor)
    assert rec["affinity_mean"] > 0
    assert don["affinity_mean"] < 0


def test_segment_diagnostic():
    recipient = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    donor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    hybrid = torch.tensor([[0.5, 0.0], [2.0, 0.0]])
    row = mod._segment_diagnostic(hybrid, recipient, donor)
    assert row["outside_segment_count"] == 1
    assert row["degenerate_reference_count"] == 0


def test_chunk_round_trip(tmp_path):
    args = mod.argparse.Namespace(
        expected_head="x" * 40,
        implementation_freeze_commit="x" * 40,
    )
    path = tmp_path / "chunk.pt"
    logits = torch.zeros((18, 7, 3), dtype=torch.float32)
    mod._write_chunk(
        path,
        args=args,
        worker_id=0,
        start=0,
        stop=7,
        logits=logits,
    )
    payload = mod._load_valid_chunk(
        path,
        args=args,
        worker_id=0,
        start=0,
        stop=7,
    )
    assert torch.equal(payload["logits"], logits)
    assert payload["training_executed"] is False
    assert payload["backward_executed"] is False


def test_worker_common_context_is_outside_hybrid_forward_loop():
    source = inspect.getsource(mod.run_worker)
    context_pos = source.index("_phase_b_prepare_common_context")
    cross_loop_pos = source.index(
        "for index, (a_rec, a_don, r_don) in enumerate(CROSS_HYBRIDS)"
    )
    assert context_pos < cross_loop_pos
    assert source.count("_phase_b_prepare_common_context") == 1


def test_worker_uses_recipient_latent_cache():
    source = inspect.getsource(mod.run_worker)
    assert "_recipient_latents" in source
    assert "latents[a_rec]" in source


def test_merge_only_has_no_runtime_model_or_cuda_calls():
    source = inspect.getsource(mod.merge_only)
    assert "_runtime_bundle" not in source
    assert "_phase_b_prepare_worker_runtime" not in source
    assert "torch.cuda" not in source


def test_no_training_or_backward_calls_in_implementation():
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "optimizer.step(" not in source


def test_static_mode_forbids_runtime_args():
    args = mod.argparse.Namespace(
        static_verify_only=True,
        cuda_preflight_only=False,
        run_factor_swap=False,
        factor_swap_worker=False,
        merge_only=False,
        expected_head="x",
        implementation_freeze_commit=None,
        model_snapshot=None,
        tokenizer_snapshot=None,
        checkpoint=None,
        output_root=None,
        worker_id=None,
    )
    mod.validate_args(args)


def test_merge_mode_forbids_model_runtime_inputs():
    args = mod.argparse.Namespace(
        static_verify_only=False,
        cuda_preflight_only=False,
        run_factor_swap=False,
        factor_swap_worker=False,
        merge_only=True,
        expected_head="x",
        implementation_freeze_commit="x",
        model_snapshot=Path("bad"),
        tokenizer_snapshot=None,
        checkpoint=None,
        output_root=Path("out"),
        worker_id=None,
    )
    with pytest.raises(mod.FactorSwapError):
        mod.validate_args(args)
