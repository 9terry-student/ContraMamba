from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen5_phase1b_q22_cuda_equivalence as cuda_eq
from scripts import reason_router_gen5_phase1b_r22_local_necessity_confirmation as necessity

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

AUTHORITY_COMMIT = "244e206024f949e1ca6430ff2f14a8dc037236f9"
CUDA_EQ_ARTIFACT_FREEZE_COMMIT = "96f8a9a8385d71175db6c0d52a86f16c5ea75040"
CUDA_EQ_IMPLEMENTATION_FREEZE_COMMIT = "5075d862c69ca406b12c30f9ca438c3203aef2a2"
NECESSITY_IMPLEMENTATION_FREEZE_COMMIT = "130a3474cf83d1ba6080561afbf886614dde4786"

AUTHORITY_PATH = (
    "reports/"
    "reason_router_gen5_phase1b_r22_local_necessity_full_cuda_execution_"
    "authority_spec_candidate.md"
)
CUDA_EQ_ARTIFACT_ROOT = Path(
    "reports/reason_router_gen5_phase1b_q22_cuda_equivalence_5075d86_r3"
)
CUDA_EQ_SUMMARY_SHA256 = (
    "fde0ca28dbc22e5ab953bcd6dc68c46ba32be9c25cf5f587cfd5f608ae7ff528"
)
CUDA_EQ_ITEMS_SHA256 = (
    "24f25de8bb2c87b45b2a4e6f8e3969195b821596b422cdaa6ab15cb43542dd84"
)
CUDA_EQ_MANIFEST_SHA256 = (
    "a0c6944e35246ac81f57a5b642a9eb20921a2798ba51de03e8c77e472701cff2"
)

FROZEN_BLOBS = {
    AUTHORITY_PATH: "b9ea1aea54aa61131bca148225e568d0dd9f2fff",
    "scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py":
        "421d30f00cf71690ed41c983ccf0540808e1de1c",
    "scripts/reason_router_gen5_phase1b_r22_local_necessity_confirmation.py":
        "50be219720b2809969b4f3caac25b2c25598a81b",
    (
        "reports/reason_router_gen5_phase1b_q22_cuda_equivalence_5075d86_r3/"
        "q22_cuda_equivalence_summary.json"
    ): "35327232f3bad0bb25b3a958d9c0b5587a71c032",
    (
        "reports/reason_router_gen5_phase1b_q22_cuda_equivalence_5075d86_r3/"
        "q22_cuda_equivalence_items.jsonl"
    ): "4143a314d5d7f68f4ccf06303d56122fbfcdb2d7",
    (
        "reports/reason_router_gen5_phase1b_q22_cuda_equivalence_5075d86_r3/"
        "artifact_manifest.json"
    ): "e614ea05270055db15b02505a4e2555f0f8062c0",
}

QUALIFIED_BACKEND = "FROZEN_MAMBA_SSM_KERNEL_CAPTURE_PLUS_LAYER22_STATE_REPLAY"
FULL_CUDA_MODEL_FORWARD_BUDGET = necessity.FULL_FORWARD_BUDGET
CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET = 0


class FastCudaNecessityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FastCudaNecessityError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FastCudaNecessityError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for ancestor, label in (
        (AUTHORITY_COMMIT, "FULL_CUDA_AUTHORITY"),
        (CUDA_EQ_ARTIFACT_FREEZE_COMMIT, "CUDA_EQ_ARTIFACT_FREEZE"),
        (CUDA_EQ_IMPLEMENTATION_FREEZE_COMMIT, "CUDA_EQ_IMPLEMENTATION_FREEZE"),
        (NECESSITY_IMPLEMENTATION_FREEZE_COMMIT, "NECESSITY_IMPLEMENTATION_FREEZE"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", ancestor, expected_head) == 0,
            f"{label}_NOT_ANCESTOR",
        )

    for path, expected_blob in FROZEN_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(
            observed == expected_blob,
            f"FROZEN_BLOB_DRIFT:{path}:{observed}",
        )

    # Reuse the qualified backend's own frozen-source ancestry checks.
    cuda_eq.authenticate_repo(expected_head)


def validate_qualified_backend_evidence() -> dict[str, Any]:
    root = ROOT / CUDA_EQ_ARTIFACT_ROOT
    summary_path = root / "q22_cuda_equivalence_summary.json"
    items_path = root / "q22_cuda_equivalence_items.jsonl"
    manifest_path = root / "artifact_manifest.json"

    require(
        sha256_file(summary_path) == CUDA_EQ_SUMMARY_SHA256,
        "CUDA_EQ_SUMMARY_SHA256",
    )
    require(
        sha256_file(items_path) == CUDA_EQ_ITEMS_SHA256,
        "CUDA_EQ_ITEMS_SHA256",
    )
    require(
        sha256_file(manifest_path) == CUDA_EQ_MANIFEST_SHA256,
        "CUDA_EQ_MANIFEST_SHA256",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    require(
        summary.get("result")
        == "PASS_GEN5_PHASE1B_Q22_CUDA_BACKEND_EQUIVALENCE",
        "CUDA_EQ_RESULT",
    )
    require(
        summary.get("execution_head")
        == CUDA_EQ_IMPLEMENTATION_FREEZE_COMMIT,
        "CUDA_EQ_EXECUTION_HEAD",
    )
    require(
        summary.get("accelerated_backend")
        == "frozen_cuda_scan_capture_plus_layer22_state_replay",
        "CUDA_EQ_BACKEND",
    )
    require(
        int(summary.get("total_cuda_model_forward_count", -1)) == 240,
        "CUDA_EQ_FORWARD_COUNT",
    )
    require(
        int(summary.get("cpu_model_forward_count", -1)) == 0,
        "CUDA_EQ_CPU_FORWARD_COUNT",
    )
    require(
        summary.get("necessity_confirmation_population_loaded") is False,
        "CUDA_EQ_NECESSITY_POPULATION_FIREWALL",
    )
    require(
        summary.get("restoration_confirmation_population_loaded") is False,
        "CUDA_EQ_RESTORATION_POPULATION_FIREWALL",
    )
    require(
        int(summary.get("scientific_p_value_count", -1)) == 0,
        "CUDA_EQ_P_VALUE_COUNT",
    )
    require(
        summary.get("scientific_conclusion") is None,
        "CUDA_EQ_SCIENTIFIC_CONCLUSION",
    )
    require(
        summary.get("model_parameter_signature_before")
        == summary.get("model_parameter_signature_after"),
        "CUDA_EQ_PARAMETER_MUTATION",
    )
    require(
        manifest.get("result")
        == "PASS_GEN5_PHASE1B_Q22_CUDA_BACKEND_EQUIVALENCE",
        "CUDA_EQ_MANIFEST_RESULT",
    )
    return summary


def _tensor_hash(tensor: torch.Tensor) -> str:
    return necessity.tensor_sha256(
        tensor.detach().cpu().to(torch.float32).contiguous()
    )


def audit_accelerated_branch(
    branch: Mapping[str, Any],
    *,
    condition: str,
    plus_branch: bool,
    r22: torch.Tensor,
    c22: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    require(condition in necessity.CONDITIONS, f"CONDITION:{condition}")
    native = (
        branch["native_write"]
        .detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
    )
    actual = (
        branch["actual_write"]
        .detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
    )
    require(
        tuple(native.shape) == necessity.STATE_SHAPE,
        f"NATIVE_WRITE_SHAPE:{tuple(native.shape)}",
    )
    require(
        tuple(actual.shape) == necessity.STATE_SHAPE,
        f"ACTUAL_WRITE_SHAPE:{tuple(actual.shape)}",
    )

    plan = branch["plan"]
    target = int(branch["target_token_index"])
    probe_norm = float(branch["layer17_probe_norm"])
    require(
        abs(probe_norm - 2.0 * necessity.EPSILON) <= 1e-12,
        f"LAYER17_PROBE_NORM:{probe_norm}",
    )

    native64 = native.to(torch.float64).reshape(-1)
    actual64 = actual.to(torch.float64).reshape(-1)
    applied = actual64 - native64
    intended = plan["applied"].detach().cpu().to(torch.float64).reshape(-1)
    applied_residual = float(torch.max(torch.abs(applied - intended)).item())
    require(
        applied_residual <= necessity.MATCH_TOL,
        f"WRITE_APPLIED_RESIDUAL:{applied_residual}",
    )

    if condition == "native22":
        require(
            float(torch.max(torch.abs(applied)).item()) <= necessity.MATCH_TOL,
            "NATIVE22_WRITE_CHANGED",
        )

    a = plan["a"].detach().cpu().to(torch.float64).contiguous()
    require(tuple(a.shape) == (necessity.RANK,), "R22_COEFFICIENT_SHAPE")
    r_after = float(torch.linalg.vector_norm(r22.T @ actual64).item())
    c_after = float(torch.linalg.vector_norm(c22.T @ actual64).item())

    post_states = branch["post_states"]
    require(
        set(post_states) == set(range(target - necessity.core.TARGET_OFFSET, target - necessity.core.TARGET_OFFSET + 5)),
        "POST_STATE_COORDINATES",
    )
    target_state = (
        post_states[target]
        .detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
        .clone()
    )
    require(
        tuple(target_state.shape) == necessity.STATE_SHAPE,
        "TARGET_POST_STATE_SHAPE",
    )
    require(
        bool(torch.isfinite(target_state).all().item()),
        "TARGET_POST_STATE_NONFINITE",
    )

    audit = {
        "condition": condition,
        "target_token_index": target,
        "plus_branch": bool(plus_branch),
        "layer17_probe_l2": probe_norm,
        "native_write_sha256": _tensor_hash(native),
        "modified_write_sha256": _tensor_hash(actual),
        "target_post_state_sha256": _tensor_hash(target_state),
        "r22_coefficients": [float(v) for v in a.tolist()],
        "native_r22_projection_l2": float(torch.linalg.vector_norm(a).item()),
        "native_c22_projection_l2": float(
            torch.linalg.vector_norm(c22.T @ native64).item()
        ),
        "post_r22_projection_l2": r_after,
        "post_c22_projection_l2": c_after,
        "r22_candidate_correction_l2": float(plan["r_correction_l2"]),
        "c22_candidate_correction_l2": float(plan["c_correction_l2"]),
        "coefficient_transfer_norm_equality_residual": float(
            plan["norm_equality_residual"]
        ),
        "coefficient_transfer_norm_equality_tolerance": float(
            plan["norm_equality_tolerance"]
        ),
        "actual_write_change_l2": float(torch.linalg.vector_norm(applied).item()),
        "applied_correction_max_abs_residual": applied_residual,
        "qualified_cuda_backend": QUALIFIED_BACKEND,
    }
    return audit, target_state


def _public_signed_probe(
    raw: Mapping[str, Any],
    *,
    condition: str,
    r22: torch.Tensor,
    c22: torch.Tensor,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any], torch.Tensor]]]:
    orientation = int(raw["orientation"])
    require(orientation in (-1, 1), f"ORIENTATION:{orientation}")
    branch_audits: dict[str, Any] = {}
    records: list[tuple[str, dict[str, Any], torch.Tensor]] = []

    for role, plus_branch in (("tp", True), ("tm", False)):
        audit, target_state = audit_accelerated_branch(
            raw["_branches"][role],
            condition=condition,
            plus_branch=plus_branch,
            r22=r22,
            c22=c22,
        )
        branch_audits[role] = audit
        records.append((role, audit, target_state))

    f22 = float(raw["F22"])
    tp_pe = float(raw["_branches"]["tp"]["pe22"])
    tm_pe = float(raw["_branches"]["tm"]["pe22"])
    require(
        math.isfinite(f22) and math.isfinite(tp_pe) and math.isfinite(tm_pe),
        "SIGNED_NONFINITE",
    )
    require(abs((tp_pe - tm_pe) - f22) <= 1e-12, "F22_IDENTITY")

    return (
        {
            "condition": condition,
            "orientation": orientation,
            "F22": f22,
            "tp_pe22": tp_pe,
            "tm_pe22": tm_pe,
            "branch_audits": branch_audits,
            "model_forward_count": necessity.FORWARDS_PER_SIGNED,
        },
        records,
    )


def public_condition_from_backend(
    raw: Mapping[str, Any],
    *,
    r22: torch.Tensor,
    c22: torch.Tensor,
) -> tuple[
    dict[str, Any],
    list[tuple[str, str, str, dict[str, Any], torch.Tensor]],
]:
    condition = str(raw["condition"])
    require(condition in necessity.CONDITIONS, f"CONDITION:{condition}")
    raw_probes = raw["direction_probes"]
    require(len(raw_probes) == 2 * necessity.K, "DIRECTION_COUNT")

    public_probes: list[dict[str, Any]] = []
    records: list[
        tuple[str, str, str, dict[str, Any], torch.Tensor]
    ] = []

    for raw_direction in raw_probes:
        direction_key = str(raw_direction["direction_key"])
        positive, positive_records = _public_signed_probe(
            raw_direction["_positive"],
            condition=condition,
            r22=r22,
            c22=c22,
        )
        negative, negative_records = _public_signed_probe(
            raw_direction["_negative"],
            condition=condition,
            r22=r22,
            c22=c22,
        )
        j22 = float(raw_direction["J22"])
        j2 = float(raw_direction["J22_squared"])
        require(
            math.isfinite(j22)
            and math.isfinite(j2)
            and abs(j22 * j22 - j2) <= 1e-12 * max(abs(j2), 1.0),
            f"J22_IDENTITY:{direction_key}",
        )

        public_probes.append(
            {
                "direction_key": direction_key,
                "basis_family": str(raw_direction["basis_family"]),
                "basis_index": int(raw_direction["basis_index"]),
                "F22_plus": float(raw_direction["F22_plus"]),
                "F22_minus": float(raw_direction["F22_minus"]),
                "J22": j22,
                "J22_squared": j2,
                "positive_probe": positive,
                "negative_probe": negative,
                "model_forward_count": necessity.FORWARDS_PER_DIRECTION,
            }
        )

        for probe_name, signed_records in (
            ("positive_probe", positive_records),
            ("negative_probe", negative_records),
        ):
            for role, audit, state in signed_records:
                records.append(
                    (direction_key, probe_name, role, audit, state)
                )

    require(
        [row["direction_key"] for row in public_probes]
        == list(necessity.DIRECTIONS),
        "DIRECTION_ORDER",
    )
    e2 = sum(float(row["J22_squared"]) for row in public_probes[: necessity.K]) / necessity.K
    e4 = sum(float(row["J22_squared"]) for row in public_probes[necessity.K :]) / necessity.K
    q22 = e2 - e4
    require(
        abs(e2 - float(raw["E_XG2_22"])) <= 1e-12 * max(abs(e2), 1.0),
        f"E_XG2_IDENTITY:{condition}",
    )
    require(
        abs(e4 - float(raw["E_XG4_22"])) <= 1e-12 * max(abs(e4), 1.0),
        f"E_XG4_IDENTITY:{condition}",
    )
    require(
        abs(q22 - float(raw["Q22"])) <= 1e-12 * max(abs(q22), 1.0),
        f"Q22_IDENTITY:{condition}",
    )
    require(
        all(math.isfinite(v) for v in (e2, e4, q22)),
        f"Q22_NONFINITE:{condition}",
    )

    return (
        {
            "condition": condition,
            "direction_order": list(necessity.DIRECTIONS),
            "direction_probes": public_probes,
            "E_XG2_22": float(e2),
            "E_XG4_22": float(e4),
            "Q22": float(q22),
            "scientific_model_forward_count": necessity.FORWARDS_PER_CONDITION,
        },
        records,
    )


def validate_condition_matching(
    records_by_condition: Mapping[
        str,
        Sequence[tuple[str, str, str, Mapping[str, Any], torch.Tensor]],
    ]
) -> dict[str, Any]:
    native = list(records_by_condition["native22"])
    r_rows = list(records_by_condition["r22_neutralized"])
    c_rows = list(records_by_condition["c22_coefficient_control"])
    require(
        len(native) == len(r_rows) == len(c_rows) == 40,
        "COORDINATE_COUNT",
    )

    post_r_changes: list[float] = []
    post_c_changes: list[float] = []
    r_write_reductions: list[float] = []
    c_write_reductions: list[float] = []
    max_coeff_difference = 0.0
    max_norm_equality_residual = 0.0
    max_applied_residual = 0.0

    for n, r, c in zip(native, r_rows, c_rows, strict=True):
        require(n[:3] == r[:3] == c[:3], "COORDINATE_KEY")
        na, ns = n[3], n[4]
        ra, rs = r[3], r[4]
        ca, cs = c[3], c[4]

        require(
            na["native_write_sha256"]
            == ra["native_write_sha256"]
            == ca["native_write_sha256"],
            "NATIVE_WRITE_IDENTITY_ACROSS_CONDITIONS",
        )
        require(
            na["target_token_index"]
            == ra["target_token_index"]
            == ca["target_token_index"],
            "TARGET_TOKEN_IDENTITY",
        )
        require(
            bool(na["plus_branch"])
            == bool(ra["plus_branch"])
            == bool(ca["plus_branch"]),
            "LAYER17_BRANCH_SIGN_IDENTITY",
        )
        require(
            abs(float(na["layer17_probe_l2"]) - float(ra["layer17_probe_l2"]))
            <= 1e-12
            and abs(
                float(na["layer17_probe_l2"]) - float(ca["layer17_probe_l2"])
            )
            <= 1e-12,
            "LAYER17_PROBE_NORM_IDENTITY",
        )

        for x, y in zip(
            ra["r22_coefficients"],
            ca["r22_coefficients"],
            strict=True,
        ):
            diff = abs(float(x) - float(y))
            max_coeff_difference = max(max_coeff_difference, diff)
            require(diff <= necessity.MATCH_TOL, "R22_COEFFICIENT_MATCH")

        max_norm_equality_residual = max(
            max_norm_equality_residual,
            float(ra["coefficient_transfer_norm_equality_residual"]),
            float(ca["coefficient_transfer_norm_equality_residual"]),
        )
        max_applied_residual = max(
            max_applied_residual,
            float(ra["applied_correction_max_abs_residual"]),
            float(ca["applied_correction_max_abs_residual"]),
        )

        post_r_changes.append(
            float(torch.linalg.vector_norm(rs.to(torch.float64) - ns.to(torch.float64)).item())
        )
        post_c_changes.append(
            float(torch.linalg.vector_norm(cs.to(torch.float64) - ns.to(torch.float64)).item())
        )
        r_write_reductions.append(
            float(na["native_r22_projection_l2"])
            - float(ra["post_r22_projection_l2"])
        )
        c_write_reductions.append(
            float(na["native_r22_projection_l2"])
            - float(ca["post_r22_projection_l2"])
        )

    return {
        "coordinate_count": 40,
        "mean_r22_write_projection_reduction":
            sum(r_write_reductions) / 40.0,
        "mean_c22_write_projection_reduction":
            sum(c_write_reductions) / 40.0,
        "mean_r22_target_post_state_change_l2":
            sum(post_r_changes) / 40.0,
        "mean_c22_target_post_state_change_l2":
            sum(post_c_changes) / 40.0,
        "r22_write_reduction_positive":
            (sum(r_write_reductions) / 40.0) > 0.0,
        "r22_vs_c22_post_state_change_delta_positive":
            (sum(post_r_changes) - sum(post_c_changes)) > 0.0,
        "max_r22_coefficient_difference": max_coeff_difference,
        "max_coefficient_transfer_norm_equality_residual":
            max_norm_equality_residual,
        "max_applied_correction_residual": max_applied_residual,
        "qualified_cuda_backend": QUALIFIED_BACKEND,
    }


def item_from_backend(
    seed: Mapping[str, Any],
    raw_conditions: Sequence[Mapping[str, Any]],
    *,
    r22: torch.Tensor,
    c22: torch.Tensor,
) -> dict[str, Any]:
    require(
        [str(row["condition"]) for row in raw_conditions]
        == list(necessity.CONDITIONS),
        "CONDITION_ORDER",
    )
    public_conditions: list[dict[str, Any]] = []
    records_by_condition: dict[str, Any] = {}

    for raw in raw_conditions:
        public, records = public_condition_from_backend(
            raw,
            r22=r22,
            c22=c22,
        )
        public_conditions.append(public)
        records_by_condition[public["condition"]] = records

    by = {row["condition"]: row for row in public_conditions}
    ep = necessity.endpoint(
        float(by["native22"]["Q22"]),
        float(by["r22_neutralized"]["Q22"]),
        float(by["c22_coefficient_control"]["Q22"]),
    )
    diagnostics = validate_condition_matching(records_by_condition)

    return {
        **dict(seed),
        "schema_version": necessity.ITEM_SCHEMA,
        "implementation_authority_commit": necessity.IMPLEMENTATION_AUTHORITY_COMMIT,
        "full_cuda_execution_authority_commit": AUTHORITY_COMMIT,
        "qualified_cuda_equivalence_artifact_freeze_commit":
            CUDA_EQ_ARTIFACT_FREEZE_COMMIT,
        "q22_correction_commit": necessity.Q22_CORRECTION_COMMIT,
        "r22_c22_freeze_commit": necessity.R22_C22_FREEZE_COMMIT,
        "qualified_backend": QUALIFIED_BACKEND,
        "condition_order": list(necessity.CONDITIONS),
        "direction_order": list(necessity.DIRECTIONS),
        "conditions": public_conditions,
        **ep,
        "native_diagnostics": diagnostics,
        "scientific_model_forward_count_this_run":
            necessity.FORWARDS_PER_PAIR,
        "cpu_scientific_model_forward_count_this_run": 0,
        "row_dropped": False,
    }


def run_full_cuda_necessity(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    equivalence_summary = validate_qualified_backend_evidence()
    cuda_eq.backend.runtime_gate()
    necessity.validate_static_inputs()
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    r22, c22, basis_geometry = necessity.load_owner_bases()
    q_bases = necessity.load_q_bases()

    with cuda_eq.backend.parent_runtime_rebind():
        (
            rows,
            encoded,
            events,
            row_index,
            tokenizer_provenance,
        ) = necessity.load_inputs(tokenizer_snapshot)
        del rows

        kernels = cuda_eq.kernel_compat.load_exact_fast_kernels()
        with cuda_eq.kernel_compat.exact_transformers_kernel_loader(
            kernels
        ) as constructor_calls:
            model, checkpoint_sha = (
                cuda_eq.parent.load_representative_model_external(
                    model_snapshot=model_snapshot,
                    checkpoint_path=checkpoint_path,
                )
            )
            require(
                checkpoint_sha == necessity.CHECKPOINT_SHA256,
                "CHECKPOINT_SHA256",
            )
            runtime_ctx = cuda_eq.transport_runtime.validate_runtime_components(
                model
            )

        counts = Counter(constructor_calls)
        require(
            set(counts) == {"causal-conv1d", "mamba-ssm"},
            f"CONSTRUCTOR_KERNEL_NAMES:{dict(counts)}",
        )
        require(
            counts["causal-conv1d"] > 0
            and counts["causal-conv1d"] == counts["mamba-ssm"],
            f"CONSTRUCTOR_KERNEL_COUNTS:{dict(counts)}",
        )
        cuda_eq.kernel_compat.validate_transformers_kernel_bindings(kernels)

        model.to(torch.device("cuda:0"))
        model.eval()
        require(
            all(
                parameter.device.type == "cuda"
                for parameter in model.mamba.parameters()
            ),
            "MODEL_NOT_CUDA",
        )
        before_signature = cuda_eq.model_parameter_signature(model)

        budget = cuda_eq.parent.ForwardBudget(
            FULL_CUDA_MODEL_FORWARD_BUDGET
        )
        items: list[dict[str, Any]] = []

        for index, pair in enumerate(necessity.expected_pairs()):
            seed = necessity.probe_seed(index, pair, events)
            raw_conditions = cuda_eq.run_backend_pair(
                seed,
                q_bases=q_bases,
                branch_runner=cuda_eq.run_accelerated_branch,
                model=model,
                runtime_ctx=runtime_ctx,
                kernels=kernels,
                encoded=encoded,
                row_index=row_index,
                events=events,
                r22=r22,
                c22=c22,
                budget=budget,
            )
            item = item_from_backend(
                seed,
                raw_conditions,
                r22=r22,
                c22=c22,
            )
            items.append(item)
            del raw_conditions
            if (index + 1) % 10 == 0:
                gc.collect()
                print(
                    f"GEN5_NECESSITY_CUDA_PROGRESS={index + 1}/"
                    f"{necessity.PAIR_COUNT}",
                    flush=True,
                )

        budget.assert_exact()
        torch.cuda.synchronize()
        after_signature = cuda_eq.model_parameter_signature(model)
        require(
            before_signature == after_signature,
            "MODEL_PARAMETER_MUTATION",
        )

    require(len(items) == necessity.PAIR_COUNT, "ITEM_COUNT")
    require(
        all(row.get("row_dropped") is False for row in items),
        "ROW_DROPPING",
    )

    # This is the sole confirmatory p-value and frozen decision implementation.
    decision = necessity.confirmatory_decision(items)

    summary = {
        "schema_version": necessity.SUMMARY_SCHEMA,
        "result": necessity.RESULT_PASS,
        "execution_head": expected_head,
        "implementation_authority_commit":
            necessity.IMPLEMENTATION_AUTHORITY_COMMIT,
        "full_cuda_execution_authority_commit": AUTHORITY_COMMIT,
        "qualified_cuda_equivalence_artifact_freeze_commit":
            CUDA_EQ_ARTIFACT_FREEZE_COMMIT,
        "qualified_cuda_equivalence_execution_head":
            CUDA_EQ_IMPLEMENTATION_FREEZE_COMMIT,
        "qualified_cuda_equivalence_summary_sha256":
            CUDA_EQ_SUMMARY_SHA256,
        "qualified_backend": QUALIFIED_BACKEND,
        "qualified_backend_source_blob":
            FROZEN_BLOBS[
                "scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py"
            ],
        "q22_correction_commit": necessity.Q22_CORRECTION_COMMIT,
        "r22_c22_freeze_commit": necessity.R22_C22_FREEZE_COMMIT,
        "source_pair_count": necessity.PAIR_COUNT,
        "pair_id_first": "xg1_fact_8101",
        "pair_id_last": "xg1_fact_8400",
        "condition_order": list(necessity.CONDITIONS),
        "direction_order": list(necessity.DIRECTIONS),
        "epsilon": necessity.EPSILON,
        "subspace_dim": necessity.K,
        "forwards_per_condition": necessity.FORWARDS_PER_CONDITION,
        "forwards_per_pair": necessity.FORWARDS_PER_PAIR,
        "scientific_model_forward_count_this_run":
            FULL_CUDA_MODEL_FORWARD_BUDGET,
        "cuda_scientific_model_forward_count_this_run":
            FULL_CUDA_MODEL_FORWARD_BUDGET,
        "cpu_scientific_model_forward_count_this_run": 0,
        "checkpoint_load_count_this_run": 1,
        "representative_checkpoint_sha256": checkpoint_sha,
        "native_backbone_signature_sha256":
            necessity.NATIVE_BACKBONE_SIGNATURE_SHA256,
        "target_layer": necessity.LAYER22,
        "target_object": "WRITE22=recovered_by_qualified_cuda_state_replay",
        "post_state_object": "POST_STATE22=qualified_cuda_replayed_state",
        "q22_definition": "E_XG2_22-E_XG4_22",
        "q22_readout_function": "core.post4_path_efficiency",
        "q22_state_layer": 22,
        "basis_geometry": basis_geometry,
        "r22_basis_sha256": necessity.R22_SHA256,
        "c22_basis_sha256": necessity.C22_SHA256,
        "model_parameter_signature_before": before_signature,
        "model_parameter_signature_after": after_signature,
        "tokenizer": tokenizer_provenance,
        "runtime": dict(cuda_eq.backend.EXPECTED_RUNTIME),
        "cuda_runtime": cuda_eq.backend.EXPECTED_CUDA_RUNTIME,
        "cuda_device": cuda_eq.backend.EXPECTED_DEVICE_NAME,
        "cuda_capability": list(cuda_eq.backend.EXPECTED_CAPABILITY),
        "kernels_version": cuda_eq.backend.KERNELS_VERSION,
        "mamba_kernel_revision": cuda_eq.backend.MAMBA_REV,
        "causal_conv1d_kernel_revision": cuda_eq.backend.CONV_REV,
        "mamba_binary_sha256": cuda_eq.backend.MAMBA_BINARY_SHA256,
        "causal_conv1d_binary_sha256": cuda_eq.backend.CONV_BINARY_SHA256,
        "equivalence_observed_maxima":
            equivalence_summary["observed_maxima"],
        "decision": decision,
        "confirmatory_p_value_count": 1,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "cuda_executed": True,
        "cpu_scientific_forward_executed": False,
        "row_dropping_executed": False,
        "raw_native_vectors_persisted": False,
        "raw_post_state_vectors_persisted": False,
        "restoration_executed": False,
        "ownership_implementation_executed": False,
        "scientific_conclusion": decision["label"],
        "next_stage": (
            "RESTORATION_CONFIRMATION_IMPLEMENTATION_ONLY_IF_NECESSITY_SUPPORTED"
            if decision["label"] == necessity.LABEL_SUPPORTED
            else "STOP_PHASE1B_BRIDGE_NOT_ESTABLISHED"
        ),
    }

    necessity.write_outputs(
        output_dir,
        items=items,
        summary=summary,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gen5 Phase1B full 300-pair R22 local-necessity confirmation "
            "using the frozen qualified Q22 CUDA state-replay backend."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_full_cuda_necessity(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    decision = summary["decision"]
    print("RESULT=" + summary["result"])
    print("PAIR_ID_FIRST=" + summary["pair_id_first"])
    print("PAIR_ID_LAST=" + summary["pair_id_last"])
    print(
        "CUDA_SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(summary["cuda_scientific_model_forward_count_this_run"])
    )
    print("CPU_SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN=0")
    print("CONFIRMATORY_P_VALUE_COUNT=1")
    print("MEAN_Q0=" + repr(decision["mean_Q0"]))
    print("MEAN_A_R=" + repr(decision["mean_A_R"]))
    print("MEAN_D_NEC22=" + repr(decision["mean_D_NEC22"]))
    print(
        "P_ONE_SIDED_GREATER="
        + repr(decision["confirmatory_test"]["p_one_sided_greater"])
    )
    print("SCIENTIFIC_CONCLUSION=" + summary["scientific_conclusion"])
    print("NEXT_STAGE=" + summary["next_stage"])


if __name__ == "__main__":
    main()
