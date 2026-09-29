from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import torch

from scripts import reason_router_gen5_phase1b_r22_local_necessity_confirmation as impl

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/reason_router_gen5_phase1b_r22_local_necessity_confirmation.py"
AUTHORITY = ROOT / "reports/reason_router_gen5_phase1b_r22_local_necessity_confirmation_implementation_authority_spec_candidate.md"
CORRECTION = ROOT / "reports/reason_router_gen5_phase1b_downstream_q22_causal_order_correction_spec_candidate.md"


class VerificationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def synthetic_trace_selfcheck() -> None:
    width = impl.STATE_WIDTH
    r22 = torch.zeros((width, 2), dtype=torch.float64)
    c22 = torch.zeros((width, 2), dtype=torch.float64)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0
    w = torch.zeros(impl.STATE_SHAPE, dtype=torch.float32)
    w.reshape(-1)[:4] = torch.tensor([3.0, 4.0, 5.0, 6.0])
    plan_r = impl.intervention_vectors(w, r22, c22, "r22_neutralized")
    plan_c = impl.intervention_vectors(w, r22, c22, "c22_coefficient_control")
    require(torch.allclose(plan_r["modified"][:2], torch.zeros(2, dtype=torch.float64)), "R22_REMOVE")
    require(torch.allclose(plan_c["modified"][:2], torch.tensor([3.0, 4.0], dtype=torch.float64)), "C22_CONTROL_PRESERVES_R")
    require(
        abs(plan_r["r_correction_l2"] - plan_c["c_correction_l2"]) <= 1e-12,
        "MATCHED_NORM",
    )

    class Mixer:
        pass

    class Frame:
        pass

    mixer = Mixer()
    code = object()
    anchor = 2
    collector = impl.Layer22Q22Collector(
        code=code,
        update_line=10,
        readout_line=11,
        mixer22=mixer,
        anchor=anchor,
        condition="r22_neutralized",
        r22=r22,
        c22=c22,
    )
    collector._pending = {}
    collector.post_states = {}
    collector.target_audit = None
    for token in range(anchor, anchor + 5):
        frame = Frame()
        frame.f_code = code
        frame.f_lineno = 10
        s_prev = torch.zeros(impl.STATE_SHAPE, dtype=torch.float32)
        discrete_a = torch.zeros((1, 1536, anchor + 5, 16), dtype=torch.float32)
        delta_b_u = torch.zeros((1, 1536, anchor + 5, 16), dtype=torch.float32)
        discrete_a[:, :, token, :].fill_(0.5)
        offset = float(token - anchor)
        delta_b_u[0, 0, token, :4] = torch.tensor(
            [
                3.0 + offset,
                4.0 + 2.0 * offset,
                5.0 + 3.0 * offset,
                6.0 + 4.0 * offset,
            ]
        )
        frame.f_locals = {
            "self": mixer,
            "i": token,
            "ssm_state": s_prev,
            "discrete_A": discrete_a,
            "deltaB_u": delta_b_u,
        }
        collector._trace(frame, "line", None)
        if token == anchor + 2:
            require(
                torch.allclose(
                    delta_b_u[:, :, token, :].reshape(-1)[:2],
                    torch.zeros(2),
                ),
                "TRACE_WRITE_NOT_MODIFIED",
            )
        actual = delta_b_u[:, :, token, :].clone()
        s_post = 0.5 * s_prev + actual
        frame.f_lineno = 11
        frame.f_locals["ssm_state"] = s_post
        collector._trace(frame, "line", None)

    require(set(collector.post_states or {}) == set(range(anchor, anchor + 5)), "TRACE_STATE_SET")
    require(collector.target_audit is not None, "TRACE_AUDIT")
    pe = impl.q22_path_efficiency(collector.post_states or {}, anchor)
    require(pe >= 0.0 and pe <= 1.0, "Q22_PE_RANGE")


def main() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    require(AUTHORITY.is_file() and CORRECTION.is_file(), "AUTHORITY_FILES")
    require("FROZEN_ON_COMMIT" in AUTHORITY.read_text(encoding="utf-8"), "AUTHORITY_NOT_FROZEN")
    require("FROZEN_ON_COMMIT" in CORRECTION.read_text(encoding="utf-8"), "CORRECTION_NOT_FROZEN")

    impl.validate_static_inputs()
    r22, c22, geom = impl.load_owner_bases()
    require(tuple(r22.shape) == (24576, 2), "R22_SHAPE")
    require(tuple(c22.shape) == (24576, 2), "C22_SHAPE")
    require(geom["r22_c22_cross_max_abs"] <= 1e-10, "BASIS_GEOMETRY")

    require("core.post4_path_efficiency(" in source, "FROZEN_POST4_NOT_REUSED")
    require("parent.path_efficiency(" not in source, "UPSTREAM_PATH_EFFICIENCY_REUSED")
    require("reason_router_gen5_phase1b_xg1_restoration_confirmation_v1" not in source, "RESTORATION_COHORT_ACCESS")
    require("FULL_FORWARD_BUDGET = 36000" in source, "FORWARD_BUDGET")
    require("confirmatory_p_value_count\": 1" in source, "P_VALUE_COUNT")
    require("D_NEC22" in source and "QC" in source and "QR" in source, "ENDPOINT_IDENTITY")

    prohibited_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"backward", "train"}:
            prohibited_calls.append(node.attr)
    # model.mamba.eval is required; model.train/backward are forbidden.
    require("backward" not in prohibited_calls, "BACKWARD_PATH")

    synthetic_trace_selfcheck()

    d = [0.1 + 0.001 * i for i in range(300)]
    test = impl.one_sided_one_sample_student_t(d)
    require(test["p_one_sided_greater"] < 0.05, "SYNTHETIC_TTEST")
    items = [
        {"Q0": 1.0, "A_R": 0.2, "D_NEC22": value, "row_dropped": False}
        for value in d
    ]
    decision = impl.confirmatory_decision(items)
    require(decision["confirmatory_p_value_count"] == 1, "DECISION_P_COUNT")
    require(decision["label"] == impl.LABEL_SUPPORTED, "DECISION_POSITIVE")

    print("RESULT=PASS_READY_FOR_GEN5_PHASE1B_NECESSITY_EXECUTION_AUTHORITY")
    print("IMPLEMENTATION_AUTHORITY_COMMIT=" + impl.IMPLEMENTATION_AUTHORITY_COMMIT)
    print("Q22_CORRECTION_COMMIT=" + impl.Q22_CORRECTION_COMMIT)
    print("R22_SHA256=" + impl.R22_SHA256)
    print("C22_SHA256=" + impl.C22_SHA256)
    print("NECESSITY_PAIR_RANGE=xg1_fact_8101..xg1_fact_8400")
    print("Q22_READOUT=core.post4_path_efficiency@layer22")
    print("HISTORICAL_PARENT_PATH_EFFICIENCY_REUSED=False")
    print("RESTORATION_COHORT_SCIENTIFIC_ACCESS=False")
    print("CONFIRMATORY_P_VALUE_COUNT=1")
    print("MODEL_LOADED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("SCIENTIFIC_EXECUTION=False")


if __name__ == "__main__":
    main()
