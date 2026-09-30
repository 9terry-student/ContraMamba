from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
AUTHORITY_COMMIT = "e0715c83a35a500c5ed9d1125f2af08e39afcd33"
AUTHORITY_PATH = (
    "reports/reason_router_gen5_phase1b_r22_restoration_confirmation_"
    "implementation_authority_spec_candidate.md"
)
AUTHORITY_BLOB = "d17d0439ee83c7418aec9266a4b56e3a48915f56"

FILES = (
    "scripts/reason_router_gen5_phase1b_r22_restoration_confirmation.py",
    "scripts/reason_router_gen5_phase1b_r22_restoration_fast_cuda_2gpu.py",
    "scripts/verify_reason_router_gen5_phase1b_r22_restoration_confirmation.py",
    "tests/test_reason_router_gen5_phase1b_r22_restoration_confirmation.py",
)

STATIC = {
    "data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1/SHA256SUMS.txt":
        "ca8dd49b71235e3c876168b46ec97b7071fddabeb1125fd2d833901f8c5c40dd",
    "data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1/structured_source_facts.jsonl":
        "a2681a1fe7a76ffa7ba42c08bbe9809bc8e751d282e4909fb84ee65c49bb0245",
    "data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1/synthetic_reason_router_six_cell.jsonl":
        "8d601c44a25f7733d3613e2ce361fe1add467ae84bd7db7ce41c511802590de2",
    "data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1/structural_manifest.json":
        "a443fc4a04f05bb7ce1630b6b6a56d08126d86ffcc6d236d941e0af1a31dfeb4",
    "data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1/tokenizer_anchor_manifest.jsonl":
        "c1529631d7c82d88815a3d402858aa3d439ecea6860a523c6844c1ef38008a7d",
    "data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1/tokenizer_eligibility_summary.json":
        "22618f92edbba8fe138b3f8aa7c6ed66430bc15a8871547044af66989c0594f1",
    "reports/reason_router_gen5_phase1b_r22_local_necessity_full_cuda_437187c_retry2/r22_local_necessity_summary.json":
        "053405f6770a67e7f312fa5a192783bdfdc8bbde332d85491367d670054b6d0d",
    "reports/reason_router_gen5_phase1b_r22_local_necessity_full_cuda_437187c_retry2/r22_local_necessity_items.jsonl":
        "a62e4fd8030a86c06f93268dcc862525bb188ed6b1276e39665e362bec557ea9",
    "reports/reason_router_gen5_phase1b_r22_local_necessity_full_cuda_437187c_retry2/artifact_manifest.json":
        "5f55cd3cc4cf898a62a91fe93e908631971827c4b9e4ae163d128b6dac173dd9",
}

SOURCE_BLOBS = {
    "scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py":
        "421d30f00cf71690ed41c983ccf0540808e1de1c",
    "scripts/reason_router_gen4_pp3_necessity_fast_cuda.py":
        "26ca67ad8603799a849c151a39728368227326df",
    "scripts/reason_router_gen5_phase1b_r22_local_necessity_confirmation.py":
        "50be219720b2809969b4f3caac25b2c25598a81b",
}


def req(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


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


def literal_assignments(path: Path) -> dict[str, object]:
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
    req(git("rev-parse", "HEAD") != "", "HEAD")
    req(
        git_rc("merge-base", "--is-ancestor", AUTHORITY_COMMIT, "HEAD") == 0,
        "AUTHORITY_ANCESTRY",
    )
    req(git("rev-parse", f"HEAD:{AUTHORITY_PATH}") == AUTHORITY_BLOB, "AUTHORITY_BLOB")

    # Authenticate canonical Git bytes, not checkout EOL representation.
    # Windows core.autocrlf may materialize tracked text as CRLF while the
    # repository blob remains the exact frozen LF byte sequence.
    for rel, expected in STATIC.items():
        p = ROOT / rel
        req(p.is_file(), f"MISSING:{rel}")
        req(git_rc("diff", "--quiet", "--", rel) == 0, f"WORKTREE_DRIFT:{rel}")
        req(git_blob_sha256(rel) == expected, f"CANONICAL_SHA:{rel}")

    for rel, expected in SOURCE_BLOBS.items():
        req(git("rev-parse", f"HEAD:{rel}") == expected, f"SOURCE_BLOB:{rel}")

    for rel in FILES:
        p = ROOT / rel
        req(p.is_file(), f"IMPLEMENTATION_FILE_MISSING:{rel}")
        ast.parse(p.read_text(encoding="utf-8"))

    conf = literal_assignments(ROOT / FILES[0])
    req(conf.get("PAIR_FIRST") == 8401 and conf.get("PAIR_LAST") == 8700, "PAIR_RANGE")
    req(conf.get("PAIR_COUNT") == 300, "PAIR_COUNT")
    req(conf.get("GPU0_FIRST") == 8401 and conf.get("GPU0_LAST") == 8550, "GPU0_RANGE")
    req(conf.get("GPU1_FIRST") == 8551 and conf.get("GPU1_LAST") == 8700, "GPU1_RANGE")
    req(conf.get("SHARD_PAIR_COUNT") == 150, "SHARD_PAIR_COUNT")
    req(conf.get("FORWARDS_PER_PAIR") == 160, "FORWARDS_PER_PAIR")
    req(conf.get("FULL_MODEL_FORWARD_BUDGET") == 48000, "FULL_FORWARD_BUDGET")
    req(conf.get("SHARD_MODEL_FORWARD_BUDGET") == 24000, "SHARD_FORWARD_BUDGET")
    req(conf.get("CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET") == 0, "CPU_FORWARD_BUDGET")

    runner_text = (ROOT / FILES[1]).read_text(encoding="utf-8")
    for required in (
        'env["CUDA_VISIBLE_DEVICES"] = str(shard_id)',
        'torch.cuda.device_count() == 2',
        'torch.cuda.device_count() == 1',
        'restoration.confirmatory_decision(items)',
        'confirmatory_p_value_count": 0',
        '"CONFIRMATORY_P_VALUE_COUNT=0"',
        'WORKER_FORWARD_BUDGET',
    ):
        req(required in runner_text, f"RUNNER_CONTRACT:{required}")

    req(
        runner_text.count("restoration.confirmatory_decision(items)") == 1,
        "CONFIRMATORY_DECISION_CALL_COUNT",
    )

    forbidden = ("optimizer.step(", ".backward(", "loss.backward(", "model.train(")
    for needle in forbidden:
        req(needle not in runner_text, f"FORBIDDEN_SURFACE:{needle}")

    structural = json.loads(
        (
            ROOT
            / "data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1/structural_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    req(structural["role"] == "restoration_confirmation", "STRUCTURAL_ROLE")
    req(structural["primary_p_value_count"] == 1, "STRUCTURAL_P_COUNT")
    req(structural["necessity_responses_access_allowed"] is False, "NECESSITY_FIREWALL")

    print("RESULT=PASS_READY_FOR_GEN5_PHASE1B_RESTORATION_EXECUTION_AUTHORITY")
    print("AUTHORITY_COMMIT=" + AUTHORITY_COMMIT)
    print("IMPLEMENTATION_FILES=4")
    print("PAIR_RANGE=xg1_fact_8401..xg1_fact_8700")
    print("GPU0_RANGE=xg1_fact_8401..xg1_fact_8550")
    print("GPU1_RANGE=xg1_fact_8551..xg1_fact_8700")
    print("FULL_MODEL_FORWARD_BUDGET=48000")
    print("PER_SHARD_MODEL_FORWARD_BUDGET=24000")
    print("CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET=0")
    print("MODEL_LOADED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_SCIENTIFIC_EXECUTION=False")
    print("SCIENTIFIC_P_VALUE_COUNT=0")


if __name__ == "__main__":
    main()
