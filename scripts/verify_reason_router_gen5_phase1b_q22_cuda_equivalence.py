from __future__ import annotations

import ast
from pathlib import Path

import torch

from scripts import reason_router_gen5_phase1b_q22_cuda_equivalence as impl

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py"
AUTHORITY = ROOT / "reports/reason_router_gen5_phase1b_q22_cuda_backend_equivalence_gate_authority_spec_candidate.md"


class VerificationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise VerificationError(message)


def synthetic_bases():
    r22 = torch.zeros((impl.STATE_WIDTH, 2), dtype=torch.float64)
    c22 = torch.zeros((impl.STATE_WIDTH, 2), dtype=torch.float64)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0
    return r22, c22


def fake_update(state, u, delta, a_matrix, b, c, d, gate, bias, dt_softplus=True):
    del delta, a_matrix, c, d, gate, bias, dt_softplus
    state.add_(u.unsqueeze(-1) * b.unsqueeze(1))
    return torch.zeros_like(u)


def fake_scan(u, delta, a_matrix, b, c, d, gate, bias, delta_softplus=True, return_last_state=True):
    del delta_softplus, return_last_state
    state = torch.zeros((u.shape[0], u.shape[1], b.shape[1]), dtype=u.dtype, device=u.device)
    for token in range(u.shape[-1]):
        fake_update(state, u[..., token], delta[..., token], a_matrix, b[..., token], c[..., token], d, gate[..., token], bias)
    return torch.zeros_like(u), state


def synthetic_replay_selfcheck() -> None:
    r22, c22 = synthetic_bases()
    seq = 8
    anchor = 2
    u = torch.zeros((1, 1536, seq), dtype=torch.float32)
    for token in range(seq):
        u[0, 0, token] = 1.0 + token
        u[0, 1, token] = 2.0 + token
        u[0, 2, token] = 3.0 + token
        u[0, 3, token] = 4.0 + token
    delta = torch.ones_like(u)
    a = torch.zeros((1536, 16), dtype=torch.float32)
    b = torch.ones((1, 16, seq), dtype=torch.float32)
    c = torch.ones((1, 16, seq), dtype=torch.float32)
    d = torch.zeros((1536,), dtype=torch.float32)
    gate = torch.ones_like(u)
    bias = torch.zeros((1536,), dtype=torch.float32)
    captured = (u, delta, a, b, c, d, gate, bias)

    native = impl.replay_q22_window(captured, anchor=anchor, condition="native22", r22=r22, c22=c22, kernel_scan=fake_scan, kernel_update=fake_update)
    r = impl.replay_q22_window(captured, anchor=anchor, condition="r22_neutralized", r22=r22, c22=c22, kernel_scan=fake_scan, kernel_update=fake_update)
    cctrl = impl.replay_q22_window(captured, anchor=anchor, condition="c22_coefficient_control", r22=r22, c22=c22, kernel_scan=fake_scan, kernel_update=fake_update)

    require(torch.allclose(r["actual_write"].reshape(-1)[:2], torch.zeros(2)), "R22_REMOVE")
    require(torch.allclose(cctrl["actual_write"].reshape(-1)[:2], native["native_write"].reshape(-1)[:2]), "C22_PRESERVE_R")
    require(abs(float(r["plan"]["r_correction_l2"]) - float(cctrl["plan"]["c_correction_l2"])) <= 1e-12, "MATCHED_NORM")
    require(0.0 <= native["pe22"] <= 1.0, "PE22_RANGE")


def main() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    authority = AUTHORITY.read_text(encoding="utf-8")
    require("FROZEN_ON_COMMIT" in authority, "AUTHORITY_NOT_FROZEN")

    require(impl.AUTHORITY_COMMIT == "9770a2f0588353b0708b760e7fcce6083f4c7e06", "AUTHORITY_COMMIT")
    require(impl.PAIR_ID == "xg1_fact_7801", "PAIR_ID")
    require(impl.FORWARDS_PER_BACKEND == 120, "FORWARDS_PER_BACKEND")
    require(impl.TOTAL_EQUIVALENCE_MODEL_FORWARDS == 240, "TOTAL_FORWARD_BUDGET")
    require(impl.J22_ATOL == 0.008, "J22_ATOL")

    require("reason_router_gen5_phase1b_xg1_necessity_confirmation_v1" not in source, "NECESSITY_COHORT_ACCESS")
    require("reason_router_gen5_phase1b_xg1_restoration_confirmation_v1" not in source, "RESTORATION_COHORT_ACCESS")
    require("construction.load_inputs(" in source, "CONSTRUCTION_INPUT_BINDING")
    require("force_layer22_slow" in source, "SLOW_REFERENCE_MISSING")
    require("selective_scan_fn" in source, "FAST_SCAN_CAPTURE_MISSING")
    require("selective_state_update" in source, "FAST_UPDATE_REPLAY_MISSING")
    require("zero_u = torch.zeros_like" in source, "DECAY_ONLY_RECOVERY_MISSING")
    require("state_native - state_decay" in source, "WRITE_RECOVERY_MISSING")
    require("necessity.intervention_vectors" in source, "FROZEN_WRITE_RULE_NOT_REUSED")
    require("necessity.q22_path_efficiency" in source, "FROZEN_Q22_READOUT_NOT_REUSED")

    attrs = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
    require("backward" not in attrs, "BACKWARD_PATH")

    r22, c22, geometry = impl.necessity.load_owner_bases()
    require(tuple(r22.shape) == (24576, 2), "R22_SHAPE")
    require(tuple(c22.shape) == (24576, 2), "C22_SHAPE")
    require(geometry["r22_c22_cross_max_abs"] <= 1e-10, "BASIS_GEOMETRY")
    impl.construction.validate_static_inputs()
    pp3_path = "scripts/reason_router_gen4_pp3_necessity_fast_cuda.py"
    pp3_blob = impl.git("rev-parse", f"HEAD:{pp3_path}")
    require(
        pp3_blob == "26ca67ad8603799a849c151a39728368227326df",
        f"PP3_SOURCE_BLOB:{pp3_blob}",
    )
    require(impl.necessity.pp3.DIM == 395, "PP3_DIM")
    require(impl.necessity.pp3.K == 5, "PP3_K")
    require(set(impl.necessity.pp3.PLAN_SHA) == {"xg2", "xg4"}, "PP3_PLAN_FAMILIES")
    for family in ("xg2", "xg4"):
        plan_sha = str(impl.necessity.pp3.PLAN_SHA[family])
        require(
            len(plan_sha) == 64
            and all(ch in "0123456789abcdef" for ch in plan_sha),
            f"PP3_PLAN_SHA:{family}:{plan_sha}",
        )
    load_bases_source = (
        ROOT / pp3_path
    ).read_text(encoding="utf-8")
    require(
        "loaded=holdout._load_frozen_bases(); out={}" in load_bases_source,
        "PP3_FROZEN_BASIS_LOADER_BINDING",
    )
    require(
        'require(loaded[fam]["plan_sha256"]==PLAN_SHA[fam]' in load_bases_source,
        "PP3_PLAN_SHA_BINDING",
    )

    synthetic_replay_selfcheck()
    require(impl.j2_error_bound(2.0) == 2.0 * 2.0 * 0.008 + 0.008 ** 2, "J2_BOUND")

    print("RESULT=PASS_READY_FOR_GEN5_PHASE1B_Q22_CUDA_EQUIVALENCE_EXECUTION")
    print("AUTHORITY_COMMIT=" + impl.AUTHORITY_COMMIT)
    print("EQUIVALENCE_PAIR=xg1_fact_7801")
    print("REFERENCE_BACKEND=CUDA_LAYER22_SLOW_FORWARD")
    print("ACCELERATED_BACKEND=CUDA_SCAN_CAPTURE_PLUS_STATE_REPLAY")
    print("TOTAL_EQUIVALENCE_MODEL_FORWARDS=240")
    print("CPU_MODEL_FORWARD_COUNT=0")
    print("NECESSITY_CONFIRMATION_POPULATION_ACCESS=False")
    print("RESTORATION_CONFIRMATION_POPULATION_ACCESS=False")
    print("MODEL_LOADED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_SCIENTIFIC_EXECUTION=False")


if __name__ == "__main__":
    main()
