"""Bounded non-scientific verifier for Gen5 Phase 2 WRITE22 implementation."""

from __future__ import annotations

import hashlib
import json

import torch

from contramamba.gen5_phase2_state_update_ownership import (
    CorrectionShape,
    StateWriteCorrection,
    explicit_total_recurrence,
    phase2_final_three_way_ce,
)


TEST_SHAPE = CorrectionShape(
    hidden_size=4,
    intermediate_size=3,
    state_size=2,
    rank=2,
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def main() -> None:
    torch.manual_seed(20260930)

    eye = torch.eye(TEST_SHAPE.state_width, dtype=torch.float64)
    r22 = eye[:, :2].clone()
    c22 = eye[:, 2:4].clone()

    corrections = {
        arm: StateWriteCorrection(
            arm=arm,
            r22=r22,
            c22=c22,
            seed=5201,
            shape=TEST_SHAPE,
            strict_frozen_dimensions=False,
        )
        for arm in ("G5-C0", "G5-C1", "G5-M1")
    }

    # Gate: deterministic initialization + exact zero output.
    a_hashes = {arm: tensor_sha(m.A_theta.weight) for arm, m in corrections.items()}
    b_hashes = {arm: tensor_sha(m.B_theta.weight) for arm, m in corrections.items()}
    require(len(set(a_hashes.values())) == 1, "A_INIT_NOT_IDENTICAL")
    require(len(set(b_hashes.values())) == 1, "B_INIT_NOT_IDENTICAL")

    x = torch.randn(2, 5, TEST_SHAPE.hidden_size)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
        dtype=torch.bool,
    )
    for arm, module in corrections.items():
        require(torch.count_nonzero(module(x, attention_mask=mask)) == 0, f"ZERO_OUTPUT:{arm}")

    # Gate: nonzero projector semantics + autograd.
    for module in corrections.values():
        with torch.no_grad():
            module.B_theta.weight.normal_(0.0, 0.02)

    raw, m1_eff = corrections["G5-M1"](
        x,
        attention_mask=mask,
        return_preproject=True,
    )
    _, c1_eff = corrections["G5-C1"](
        x,
        attention_mask=mask,
        return_preproject=True,
    )
    c0_eff = corrections["G5-C0"](x, attention_mask=mask)

    require(
        torch.max(torch.abs(m1_eff.reshape(-1, 6).double() @ r22)).item() < 1e-5,
        "M1_R22_RESIDUAL",
    )
    require(
        torch.max(torch.abs(c1_eff.reshape(-1, 6).double() @ c22)).item() < 1e-5,
        "C1_C22_RESIDUAL",
    )
    require(
        torch.allclose(
            c0_eff[mask],
            corrections["G5-C0"].B_theta(
                corrections["G5-C0"].A_theta(x)
            )[mask],
        ),
        "C0_CHANGED",
    )

    loss = m1_eff.square().mean()
    loss.backward()
    require(corrections["G5-M1"].A_theta.weight.grad is not None, "A_GRAD_MISSING")
    require(corrections["G5-M1"].B_theta.weight.grad is not None, "B_GRAD_MISSING")
    require(corrections["G5-M1"].R22.grad is None, "R22_GRAD_PRESENT")
    require(corrections["G5-M1"].C22.grad is None, "C22_GRAD_PRESENT")

    # Gate: recurrence superposition.
    batch, intermediate, seq, state = 2, 3, 5, 2
    discrete_a = torch.sigmoid(
        torch.randn(batch, intermediate, seq, state)
    )
    native_write = torch.randn(batch, seq, intermediate, state)
    correction_write = torch.randn(batch, seq, intermediate, state)

    native, correction, direct = explicit_total_recurrence(
        discrete_a,
        native_write,
        correction_write,
        active_mask=mask,
    )
    max_residual = float(torch.max(torch.abs(native + correction - direct)).item())
    require(max_residual <= 1e-6, f"SUPERPOSITION_RESIDUAL:{max_residual}")

    # Gate: objective helper is plain CE.
    logits = torch.randn(8, 3, requires_grad=True)
    labels = torch.randint(0, 3, (8,))
    ce = phase2_final_three_way_ce(logits, labels)
    ref = torch.nn.functional.cross_entropy(logits, labels)
    require(torch.equal(ce, ref), "CE_OBJECTIVE_DRIFT")

    report = {
        "result": "PASS_GEN5_PHASE2_WRITE22_BOUNDED_IMPLEMENTATION_VERIFICATION",
        "code_correctness": "PASS",
        "scientific_execution": False,
        "training_executed": False,
        "task_evaluation_executed": False,
        "fresh_xg1_loaded": False,
        "scientific_p_value_count": 0,
        "initialization_identical_across_arms": True,
        "projector_semantics": "PASS",
        "autograd_to_correction": "PASS",
        "basis_gradients_absent": True,
        "recurrence_superposition_max_abs_residual": max_residual,
        "objective": "FINAL_3WAY_CROSS_ENTROPY_ONLY",
        "next_stage": "INDEPENDENT_FORWARD_BACKWARD_IMPLEMENTATION_REVIEW",
    }
    print(json.dumps(report, sort_keys=True))
    print("CODE_CORRECTNESS=PASS")
    print("SCIENTIFIC_EXECUTION=False")
    print("TRAINING_EXECUTED=False")
    print("SCIENTIFIC_P_VALUE_COUNT=0")


if __name__ == "__main__":
    main()
