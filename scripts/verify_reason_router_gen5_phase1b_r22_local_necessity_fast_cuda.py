from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from scripts import reason_router_gen5_phase1b_r22_local_necessity_fast_cuda as impl

ROOT = Path(__file__).resolve().parents[1]


class VerificationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise VerificationError(message)


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"HEAD:{path}"],
        cwd=ROOT,
        stderr=subprocess.STDOUT,
    )


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def frozen_identity_checks() -> None:
    require(
        subprocess.call(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                impl.AUTHORITY_COMMIT,
                "HEAD",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0,
        "AUTHORITY_NOT_ANCESTOR",
    )
    for path, expected_blob in impl.FROZEN_BLOBS.items():
        observed = git_text("rev-parse", f"HEAD:{path}")
        require(
            observed == expected_blob,
            f"FROZEN_BLOB:{path}:{observed}",
        )

    eq_root = impl.CUDA_EQ_ARTIFACT_ROOT.as_posix()
    require(
        sha256(git_bytes(f"{eq_root}/q22_cuda_equivalence_summary.json"))
        == impl.CUDA_EQ_SUMMARY_SHA256,
        "EQ_SUMMARY_SHA256",
    )
    require(
        sha256(git_bytes(f"{eq_root}/q22_cuda_equivalence_items.jsonl"))
        == impl.CUDA_EQ_ITEMS_SHA256,
        "EQ_ITEMS_SHA256",
    )
    require(
        sha256(git_bytes(f"{eq_root}/artifact_manifest.json"))
        == impl.CUDA_EQ_MANIFEST_SHA256,
        "EQ_MANIFEST_SHA256",
    )

    summary = json.loads(
        git_bytes(
            f"{eq_root}/q22_cuda_equivalence_summary.json"
        ).decode("utf-8")
    )
    require(
        summary["result"]
        == "PASS_GEN5_PHASE1B_Q22_CUDA_BACKEND_EQUIVALENCE",
        "EQ_RESULT",
    )
    require(
        summary["execution_head"]
        == impl.CUDA_EQ_IMPLEMENTATION_FREEZE_COMMIT,
        "EQ_EXECUTION_HEAD",
    )
    require(
        summary["scientific_p_value_count"] == 0,
        "EQ_P_VALUE_COUNT",
    )
    require(
        summary["necessity_confirmation_population_loaded"] is False,
        "EQ_NECESSITY_FIREWALL",
    )


def static_population_and_basis_checks() -> None:
    for name, expected in impl.necessity.STATIC_INPUT_SHA256.items():
        path = (
            impl.necessity.DATA_ROOT / name
        ).as_posix()
        require(
            sha256(git_bytes(path)) == expected,
            f"NECESSITY_STATIC_INPUT:{name}",
        )

    r_raw = git_bytes(impl.necessity.R22_PATH.as_posix())
    c_raw = git_bytes(impl.necessity.C22_PATH.as_posix())
    require(sha256(r_raw) == impl.necessity.R22_SHA256, "R22_SHA256")
    require(sha256(c_raw) == impl.necessity.C22_SHA256, "C22_SHA256")

    r22 = impl.necessity._basis_from_bytes(r_raw, "R22_VERIFY")
    c22 = impl.necessity._basis_from_bytes(c_raw, "C22_VERIFY")
    eye = torch.eye(impl.necessity.RANK, dtype=torch.float64)
    require(
        float(torch.max(torch.abs(r22.T @ r22 - eye)).item())
        <= impl.necessity.BASIS_ATOL,
        "R22_GEOMETRY",
    )
    require(
        float(torch.max(torch.abs(c22.T @ c22 - eye)).item())
        <= impl.necessity.BASIS_ATOL,
        "C22_GEOMETRY",
    )
    require(
        float(torch.max(torch.abs(r22.T @ c22)).item())
        <= impl.necessity.BASIS_ATOL,
        "R22_C22_CROSS",
    )

    pp3_path = "scripts/reason_router_gen4_pp3_necessity_fast_cuda.py"
    require(
        git_text("rev-parse", f"HEAD:{pp3_path}")
        == "26ca67ad8603799a849c151a39728368227326df",
        "PP3_SOURCE_BLOB",
    )
    require(impl.necessity.pp3.DIM == 395, "PP3_DIM")
    require(impl.necessity.pp3.K == 5, "PP3_K")
    require(
        set(impl.necessity.pp3.PLAN_SHA) == {"xg2", "xg4"},
        "PP3_PLAN_FAMILIES",
    )


def synthetic_branch(
    *,
    condition: str,
    r22: torch.Tensor,
    c22: torch.Tensor,
    anchor: int = 2,
):
    native = torch.zeros(impl.necessity.STATE_SHAPE, dtype=torch.float32)
    native.reshape(-1)[0] = 1.25
    native.reshape(-1)[1] = -0.75
    native.reshape(-1)[2] = 0.5
    plan = impl.necessity.intervention_vectors(
        native,
        r22,
        c22,
        condition,
    )
    actual = (
        plan["modified"]
        .reshape(impl.necessity.STATE_SHAPE)
        .to(torch.float32)
        .contiguous()
    )
    states = {}
    for token in range(anchor, anchor + 5):
        state = torch.zeros(
            impl.necessity.STATE_SHAPE,
            dtype=torch.float32,
        )
        state.reshape(-1)[:4] = torch.tensor(
            [1.0 + token, 2.0 + token, 3.0 + token, 4.0 + token]
        )
        if token == anchor + impl.necessity.core.TARGET_OFFSET:
            state = state + actual
        states[token] = state
    return {
        "native_write": native,
        "actual_write": actual,
        "plan": plan,
        "post_states": states,
        "pe22": float(impl.necessity.q22_path_efficiency(states, anchor)),
        "layer17_probe_norm": 2.0 * impl.necessity.EPSILON,
        "target_token_index":
            anchor + impl.necessity.core.TARGET_OFFSET,
    }


def synthetic_semantic_checks() -> None:
    r22 = torch.zeros(
        (impl.necessity.STATE_WIDTH, impl.necessity.RANK),
        dtype=torch.float64,
    )
    c22 = torch.zeros_like(r22)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0

    records = {}
    for condition in impl.necessity.CONDITIONS:
        audit, state = impl.audit_accelerated_branch(
            synthetic_branch(
                condition=condition,
                r22=r22,
                c22=c22,
            ),
            condition=condition,
            plus_branch=True,
            r22=r22,
            c22=c22,
        )
        records[condition] = [
            ("xg2_0", "positive_probe", "tp", audit, state)
        ] * 40

    diagnostics = impl.validate_condition_matching(records)
    require(diagnostics["coordinate_count"] == 40, "SYNTH_COORDINATES")
    require(
        diagnostics["max_r22_coefficient_difference"] <= 1e-12,
        "SYNTH_COEFFICIENT_MATCH",
    )

    ep = impl.necessity.endpoint(1.2, 0.7, 1.0)
    require(abs(ep["D_NEC22"] - 0.3) <= 1e-12, "SYNTH_ENDPOINT")


def source_boundary_checks() -> None:
    path = ROOT / (
        "scripts/"
        "reason_router_gen5_phase1b_r22_local_necessity_fast_cuda.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    require(
        "cuda_eq.run_backend_pair(" in source,
        "QUALIFIED_BACKEND_PAIR_CALL",
    )
    require(
        "branch_runner=cuda_eq.run_accelerated_branch" in source,
        "QUALIFIED_BACKEND_BRANCH_CALL",
    )
    require(
        "necessity.endpoint(" in source,
        "FROZEN_ENDPOINT_CALL",
    )
    require(
        "necessity.confirmatory_decision(items)" in source,
        "FROZEN_DECISION_CALL",
    )
    require(
        "necessity.write_outputs(" in source,
        "FROZEN_SERIALIZATION_CALL",
    )
    require(
        "def one_sided_one_sample_student_t" not in source,
        "DUPLICATED_CONFIRMATORY_TEST",
    )
    require(
        "reason_router_gen5_phase1b_xg1_restoration_confirmation" not in source,
        "RESTORATION_POPULATION_ACCESS",
    )
    require(
        "FULL_CUDA_MODEL_FORWARD_BUDGET = necessity.FULL_FORWARD_BUDGET"
        in source,
        "FORWARD_BUDGET_BINDING",
    )

    attrs = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    ]
    require("backward" not in attrs, "BACKWARD_PATH")


def main() -> None:
    frozen_identity_checks()
    static_population_and_basis_checks()
    synthetic_semantic_checks()
    source_boundary_checks()

    require(
        impl.FULL_CUDA_MODEL_FORWARD_BUDGET == 36000,
        "FULL_FORWARD_BUDGET",
    )
    require(
        impl.CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET == 0,
        "CPU_FORWARD_BUDGET",
    )
    require(
        impl.necessity.PAIR_FIRST == 8101
        and impl.necessity.PAIR_LAST == 8400
        and impl.necessity.PAIR_COUNT == 300,
        "PAIR_RANGE",
    )

    print(
        "RESULT="
        "PASS_READY_FOR_GEN5_PHASE1B_FULL_CUDA_NECESSITY_EXECUTION"
    )
    print("AUTHORITY_COMMIT=" + impl.AUTHORITY_COMMIT)
    print(
        "QUALIFIED_CUDA_EQ_ARTIFACT_FREEZE_COMMIT="
        + impl.CUDA_EQ_ARTIFACT_FREEZE_COMMIT
    )
    print("NECESSITY_PAIR_RANGE=xg1_fact_8101..xg1_fact_8400")
    print("FULL_CUDA_MODEL_FORWARD_BUDGET=36000")
    print("CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET=0")
    print("CONFIRMATORY_P_VALUE_COUNT_BEFORE_EXECUTION=0")
    print("MODEL_LOADED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_SCIENTIFIC_EXECUTION=False")
    print("RESTORATION_CONFIRMATION_POPULATION_ACCESS=False")


if __name__ == "__main__":
    main()
