from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

AUTHORITY_COMMIT = "d6cb3a3882fab5dee13ccbf3e53ef62439b62e7a"
AUTHORITY_PATH = (
    "reports/reason_router_gen5_phase1b_r22_restoration_2gpu_topology_"
    "equivalence_implementation_authority_spec_candidate.md"
)
AUTHORITY_BLOB = "7db6f1adc630d47709833e810cc04546f9a491ff"

FILES = (
    "scripts/reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence.py",
    "scripts/verify_reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence.py",
    "tests/test_reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence.py",
)

PARENT_BLOBS = {
    "scripts/reason_router_gen5_phase1b_r22_restoration_confirmation.py":
        "786f186ab89ffc44bb5373b70df7a1041165b7a5",
    "scripts/reason_router_gen5_phase1b_r22_restoration_fast_cuda_2gpu.py":
        "ed73904fdf9a555e11d30e3b9068ac12941225cd",
    "scripts/verify_reason_router_gen5_phase1b_r22_restoration_confirmation.py":
        "e50888814d0b489ffb37f0e97c89a102e4709f41",
    "tests/test_reason_router_gen5_phase1b_r22_restoration_confirmation.py":
        "55e38b2c31f76e85ab99e7aa2c27900a336f541d",
    "scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py":
        "421d30f00cf71690ed41c983ccf0540808e1de1c",
}

STATIC = {
    "data/reason_router_gen5_phase1b_xg1_construction_v1/SHA256SUMS.txt":
        "c84902de093eab2ae27f157ed814041d2c26a4a5c272cf0ad7f807ccdbb40855",
    "data/reason_router_gen5_phase1b_xg1_construction_v1/structured_source_facts.jsonl":
        "3f8eac771794e1d022bee9f669f315c3ac05feeaa9c2195095b8a892b4269ea1",
    "data/reason_router_gen5_phase1b_xg1_construction_v1/synthetic_reason_router_six_cell.jsonl":
        "5a82508e54cd6097aeee5afe10d7c416357302701d558cfdc8740382168feca4",
    "data/reason_router_gen5_phase1b_xg1_construction_v1/structural_manifest.json":
        "d68d812ca43a6c651d284f0aa2889f87adb7e030c599af37cb49deaa9d6d8105",
    "data/reason_router_gen5_phase1b_xg1_construction_v1/tokenizer_anchor_manifest.jsonl":
        "1dbdd3e072245072f197d87443cc8cf81cbbcd2e61437c99b26e0d82815e33b4",
    "data/reason_router_gen5_phase1b_xg1_construction_v1/tokenizer_eligibility_summary.json":
        "7e304fb1d6f2ecb9ed6fb87911623c47d0b6ae9c2eb9bf9b8f45f70ae1fcb06a",
}


def req(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def git_blob_sha256(rel: str) -> str:
    raw = subprocess.check_output(
        ["git", "show", f"HEAD:{rel}"],
        cwd=ROOT,
        stderr=subprocess.STDOUT,
    )
    return hashlib.sha256(raw).hexdigest()


def literals(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return out


def main() -> None:
    req(git("branch", "--show-current") in {"", EXPECTED_BRANCH}, "BRANCH")
    req(
        git_rc("merge-base", "--is-ancestor", AUTHORITY_COMMIT, "HEAD") == 0,
        "AUTHORITY_ANCESTRY",
    )
    req(git("rev-parse", f"HEAD:{AUTHORITY_PATH}") == AUTHORITY_BLOB,
        "AUTHORITY_BLOB")

    for rel, expected in PARENT_BLOBS.items():
        req(git("rev-parse", f"HEAD:{rel}") == expected, f"PARENT_BLOB:{rel}")

    for rel, expected in STATIC.items():
        req((ROOT / rel).is_file(), f"STATIC_MISSING:{rel}")
        req(git_rc("diff", "--quiet", "--", rel) == 0, f"STATIC_WORKTREE_DRIFT:{rel}")
        req(git_blob_sha256(rel) == expected, f"STATIC_SHA:{rel}")

    for rel in FILES:
        p = ROOT / rel
        req(p.is_file(), f"IMPLEMENTATION_FILE_MISSING:{rel}")
        ast.parse(p.read_text(encoding="utf-8"))

    gate_path = ROOT / FILES[0]
    values = literals(gate_path)

    req(values.get("GATE_PAIR_FIRST") == 7801, "GATE_PAIR_FIRST")
    req(values.get("GATE_PAIR_LAST") == 7804, "GATE_PAIR_LAST")
    req(values.get("GATE_PAIR_COUNT") == 4, "GATE_PAIR_COUNT")
    req(values.get("REFERENCE_FORWARD_BUDGET") == 640, "REFERENCE_BUDGET")
    req(values.get("CANDIDATE0_FORWARD_BUDGET") == 320, "CANDIDATE0_BUDGET")
    req(values.get("CANDIDATE1_FORWARD_BUDGET") == 320, "CANDIDATE1_BUDGET")
    req(values.get("CANDIDATE_TOTAL_FORWARD_BUDGET") == 640, "CANDIDATE_TOTAL")
    req(values.get("TOTAL_GATE_FORWARD_BUDGET") == 1280, "TOTAL_GATE_BUDGET")
    req(values.get("CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET") == 0, "CPU_BUDGET")
    req(values.get("FLOAT_ATOL") == 1e-9, "FLOAT_ATOL")
    req(values.get("FLOAT_RTOL") == 1e-7, "FLOAT_RTOL")

    source = gate_path.read_text(encoding="utf-8")
    required = (
        'env["CUDA_VISIBLE_DEVICES"] = str(WORKER_SPECS[role]["physical_gpu"])',
        'construction.load_inputs(args.tokenizer_snapshot)',
        'torch.cuda.device_count() == 2',
        'torch.cuda.device_count() == 1',
        '"confirmatory_p_value_count": 0',
        '"scientific_conclusion": None',
        'restoration_confirmation_population_loaded": False',
        'reference_launch = _launch_worker(',
        'candidate0_launch = _launch_worker(',
        'candidate1_launch = _launch_worker(',
    )
    for needle in required:
        req(needle in source, f"GATE_CONTRACT:{needle}")

    forbidden = (
        "restoration.load_inputs(",
        "restoration.confirmatory_decision(",
        "one_sided_one_sample_student_t(",
        "xg1_fact_8401",
        "xg1_fact_8700",
        "optimizer.step(",
        ".backward(",
        "model.train(",
    )
    for needle in forbidden:
        req(needle not in source, f"FORBIDDEN_GATE_SURFACE:{needle}")

    # The gate intentionally emits this zero-count marker in both execution
    # surfaces: once from a worker and once from the coordinator.
    req(source.count("CONFIRMATORY_P_VALUE_COUNT=0") == 2,
        "PRINT_PVALUE_ZERO_COUNT")

    print(
        "RESULT=PASS_READY_FOR_GEN5_PHASE1B_"
        "RESTORATION_2GPU_TOPOLOGY_GATE_EXECUTION_AUTHORITY"
    )
    print("AUTHORITY_COMMIT=" + AUTHORITY_COMMIT)
    print("IMPLEMENTATION_FILES=3")
    print("GATE_PAIRS=xg1_fact_7801..xg1_fact_7804")
    print("REFERENCE_FORWARD_BUDGET=640")
    print("CANDIDATE_GPU0_FORWARD_BUDGET=320")
    print("CANDIDATE_GPU1_FORWARD_BUDGET=320")
    print("TOTAL_GATE_FORWARD_BUDGET=1280")
    print("CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET=0")
    print("FLOAT_ATOL=1e-9")
    print("FLOAT_RTOL=1e-7")
    print("MODEL_LOADED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_SCIENTIFIC_EXECUTION=False")
    print("SCIENTIFIC_P_VALUE_COUNT=0")


if __name__ == "__main__":
    main()
