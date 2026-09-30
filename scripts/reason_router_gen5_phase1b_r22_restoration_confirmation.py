from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen4_pp3_necessity_fast_cuda as pp3
from scripts import reason_router_gen5_phase1b_r22_local_necessity_confirmation as necessity

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

IMPLEMENTATION_AUTHORITY_COMMIT = "e0715c83a35a500c5ed9d1125f2af08e39afcd33"
IMPLEMENTATION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_phase1b_r22_restoration_confirmation_"
    "implementation_authority_spec_candidate.md"
)
IMPLEMENTATION_AUTHORITY_BLOB = "d17d0439ee83c7418aec9266a4b56e3a48915f56"

PHASE1B_DESIGN_COMMIT = "c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8"
NECESSITY_EVIDENCE_FREEZE_COMMIT = "4f7dd3a9e0ca2606a7aee8e204e416f32adc884e"
R22_C22_FREEZE_COMMIT = "1d3542013934870aa9181d1bbaf565ff4724112c"
CUDA_EQ_ARTIFACT_FREEZE_COMMIT = "96f8a9a8385d71175db6c0d52a86f16c5ea75040"

NECESSITY_ROOT = Path(
    "reports/reason_router_gen5_phase1b_r22_local_necessity_full_cuda_437187c_retry2"
)
NECESSITY_FILE_SHA256 = {
    "r22_local_necessity_summary.json":
        "053405f6770a67e7f312fa5a192783bdfdc8bbde332d85491367d670054b6d0d",
    "r22_local_necessity_items.jsonl":
        "a62e4fd8030a86c06f93268dcc862525bb188ed6b1276e39665e362bec557ea9",
    "artifact_manifest.json":
        "5f55cd3cc4cf898a62a91fe93e908631971827c4b9e4ae163d128b6dac173dd9",
}
NECESSITY_RESULT = "PASS_GEN5_PHASE1B_R22_LOCAL_NECESSITY_CONFIRMATION"
NECESSITY_DECISION = "GEN5_R22_LOCAL_NECESSITY_OVER_MATCHED_C22_CONTROL_SUPPORTED"

DATA_ROOT = Path("data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1")
SOURCE_FILE = "structured_source_facts.jsonl"
ROWS_FILE = "synthetic_reason_router_six_cell.jsonl"
STRUCTURAL_FILE = "structural_manifest.json"
ANCHOR_FILE = "tokenizer_anchor_manifest.jsonl"
ELIGIBILITY_FILE = "tokenizer_eligibility_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

STATIC_INPUT_SHA256 = {
    SOURCE_FILE: "a2681a1fe7a76ffa7ba42c08bbe9809bc8e751d282e4909fb84ee65c49bb0245",
    ROWS_FILE: "8d601c44a25f7733d3613e2ce361fe1add467ae84bd7db7ce41c511802590de2",
    STRUCTURAL_FILE: "a443fc4a04f05bb7ce1630b6b6a56d08126d86ffcc6d236d941e0af1a31dfeb4",
    ANCHOR_FILE: "c1529631d7c82d88815a3d402858aa3d439ecea6860a523c6844c1ef38008a7d",
    ELIGIBILITY_FILE: "22618f92edbba8fe138b3f8aa7c6ed66430bc15a8871547044af66989c0594f1",
}
CHECKSUMS_SHA256 = "ca8dd49b71235e3c876168b46ec97b7071fddabeb1125fd2d833901f8c5c40dd"

PAIR_FIRST = 8401
PAIR_LAST = 8700
PAIR_COUNT = 300
ROWS_PER_PAIR = 6
ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR

GPU0_FIRST = 8401
GPU0_LAST = 8550
GPU1_FIRST = 8551
GPU1_LAST = 8700
SHARD_PAIR_COUNT = 150

RANK = necessity.RANK
STATE_WIDTH = necessity.STATE_WIDTH
STATE_SHAPE = necessity.STATE_SHAPE
LAYER22 = necessity.LAYER22
K = necessity.K
EPSILON = necessity.EPSILON
MATCH_TOL = necessity.MATCH_TOL
RECURRENCE_REL_TOL = necessity.RECURRENCE_REL_TOL
DIRECTIONS = necessity.DIRECTIONS
BRANCHES = necessity.BRANCHES

FORWARDS_PER_DONOR_CONDITION = 40
FORWARDS_PER_RESTORATION_CONDITION = 40
FORWARDS_PER_PAIR = 160
FULL_MODEL_FORWARD_BUDGET = 48000
SHARD_MODEL_FORWARD_BUDGET = 24000
CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET = 0

ITEM_FILE = "r22_restoration_items.jsonl"
SUMMARY_FILE = "r22_restoration_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
ITEM_SCHEMA = "gen5-phase1b-r22-restoration-item-v1"
SUMMARY_SCHEMA = "gen5-phase1b-r22-restoration-summary-v1"
MANIFEST_SCHEMA = "gen5-phase1b-r22-restoration-manifest-v1"
RESULT_PASS = "PASS_GEN5_PHASE1B_R22_RESTORATION_CONFIRMATION"
LABEL_SUPPORTED = "GEN5_R22_RESTORATION_OVER_MATCHED_C22_REPLACEMENT_SUPPORTED"
LABEL_NOT_ESTABLISHED = "GEN5_R22_RESTORATION_NOT_ESTABLISHED"


class RestorationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RestorationError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RestorationError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False
        ) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        out.append(value)
    return out


def expected_pairs() -> tuple[str, ...]:
    return tuple(f"xg1_fact_{i}" for i in range(PAIR_FIRST, PAIR_LAST + 1))


def shard_pairs(shard_id: int) -> tuple[str, ...]:
    require(shard_id in (0, 1), f"SHARD_ID:{shard_id}")
    lo, hi = (
        (GPU0_FIRST, GPU0_LAST) if shard_id == 0
        else (GPU1_FIRST, GPU1_LAST)
    )
    rows = tuple(f"xg1_fact_{i}" for i in range(lo, hi + 1))
    require(len(rows) == SHARD_PAIR_COUNT, f"SHARD_PAIR_COUNT:{shard_id}")
    return rows


def merge_shards(
    shard0: Sequence[Mapping[str, Any]],
    shard1: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    require(
        [str(x["source_pair_id"]) for x in shard0] == list(shard_pairs(0)),
        "SHARD0_ORDER",
    )
    require(
        [str(x["source_pair_id"]) for x in shard1] == list(shard_pairs(1)),
        "SHARD1_ORDER",
    )
    merged = [dict(x) for x in shard0] + [dict(x) for x in shard1]
    require(
        [str(x["source_pair_id"]) for x in merged] == list(expected_pairs()),
        "MERGED_ORDER",
    )
    require(len({str(x["source_pair_id"]) for x in merged}) == PAIR_COUNT, "PAIR_DUPLICATE")
    require(
        sum(int(x["scientific_model_forward_count_this_run"]) for x in merged)
        == FULL_MODEL_FORWARD_BUDGET,
        "GLOBAL_FORWARD_BUDGET",
    )
    return merged


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    for commit, label in (
        (IMPLEMENTATION_AUTHORITY_COMMIT, "IMPLEMENTATION_AUTHORITY"),
        (PHASE1B_DESIGN_COMMIT, "PHASE1B_DESIGN"),
        (NECESSITY_EVIDENCE_FREEZE_COMMIT, "NECESSITY_EVIDENCE"),
        (R22_C22_FREEZE_COMMIT, "R22_C22_FREEZE"),
        (CUDA_EQ_ARTIFACT_FREEZE_COMMIT, "CUDA_EQ_ARTIFACT_FREEZE"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", commit, expected_head) == 0,
            f"{label}_NOT_ANCESTOR",
        )
    observed = git("rev-parse", f"HEAD:{IMPLEMENTATION_AUTHORITY_PATH}")
    require(
        observed == IMPLEMENTATION_AUTHORITY_BLOB,
        f"AUTHORITY_BLOB_DRIFT:{observed}",
    )


def validate_necessity_prerequisite() -> dict[str, Any]:
    for name, expected in NECESSITY_FILE_SHA256.items():
        path = ROOT / NECESSITY_ROOT / name
        require(path.is_file(), f"NECESSITY_FILE_MISSING:{name}")
        require(sha256_file(path) == expected, f"NECESSITY_SHA256:{name}")
    summary = json.loads(
        (ROOT / NECESSITY_ROOT / "r22_local_necessity_summary.json")
        .read_text(encoding="utf-8-sig")
    )
    require(summary.get("result") == NECESSITY_RESULT, "NECESSITY_RESULT")
    require(summary.get("scientific_conclusion") == NECESSITY_DECISION, "NECESSITY_DECISION")
    require(int(summary.get("confirmatory_p_value_count", -1)) == 1, "NECESSITY_P_COUNT")
    require(int(summary.get("cuda_scientific_model_forward_count_this_run", -1)) == 36000,
            "NECESSITY_FORWARD_COUNT")
    return summary


def validate_static_inputs() -> None:
    sums = ROOT / DATA_ROOT / CHECKSUM_FILE
    require(sums.is_file() and sha256_file(sums) == CHECKSUMS_SHA256, "CHECKSUMS_SHA256")
    for name, expected in STATIC_INPUT_SHA256.items():
        path = ROOT / DATA_ROOT / name
        require(path.is_file(), f"STATIC_INPUT_MISSING:{name}")
        require(sha256_file(path) == expected, f"STATIC_INPUT_SHA256:{name}")

    structural = json.loads(
        (ROOT / DATA_ROOT / STRUCTURAL_FILE).read_text(encoding="utf-8-sig")
    )
    require(structural.get("result") == "PASS_GEN5_PHASE1B_XG1_STRUCTURAL_HOLDOUT",
            "STRUCTURAL_RESULT")
    require(structural.get("role") == "restoration_confirmation", "STRUCTURAL_ROLE")
    require(structural.get("pair_id_first") == "xg1_fact_8401", "STRUCTURAL_FIRST")
    require(structural.get("pair_id_last") == "xg1_fact_8700", "STRUCTURAL_LAST")
    require(structural.get("source_pair_count") == PAIR_COUNT, "STRUCTURAL_PAIR_COUNT")
    require(structural.get("row_count") == ROW_COUNT, "STRUCTURAL_ROW_COUNT")
    require(structural.get("primary_p_value_count") == 1, "STRUCTURAL_P_VALUE_COUNT")
    require(
        structural.get("primary_endpoint")
        == "D_SUF22=Q_R22_RESTORED-Q_C22_REPLACEMENT",
        "STRUCTURAL_ENDPOINT",
    )
    for key in (
        "labels_present", "response_fields_present", "endpoint_values_present",
        "tokenizer_executed", "checkpoint_loaded", "model_executed",
        "cuda_executed", "training_executed", "backward_executed",
        "R22_construction_allowed", "C22_construction_allowed",
        "construction_responses_access_allowed", "necessity_responses_access_allowed",
        "cohort_replacement_allowed", "row_filtering_allowed",
    ):
        require(structural.get(key) is False, f"STRUCTURAL_BOUNDARY:{key}")

    eligibility = json.loads(
        (ROOT / DATA_ROOT / ELIGIBILITY_FILE).read_text(encoding="utf-8-sig")
    )
    require(
        eligibility.get("result") == "PASS_GEN5_PHASE1B_TOKENIZER_ANCHOR_ELIGIBILITY",
        "ELIGIBILITY_RESULT",
    )
    require(eligibility.get("role") == "restoration_confirmation", "ELIGIBILITY_ROLE")
    require(eligibility.get("eligible_anchor_row_count") == ROW_COUNT, "ELIGIBILITY_COUNT")
    require(
        eligibility.get("identity_name_coordinate_mismatch_count") == 0,
        "ELIGIBILITY_COORDINATE_MISMATCH",
    )


def load_inputs(tokenizer_snapshot: str | Path | None):
    validate_static_inputs()
    rows = necessity.adapter.validate_gen4_rows(
        read_jsonl(ROOT / DATA_ROOT / ROWS_FILE),
        require_canonical_shape=True,
    )
    pairs = []
    seen: set[str] = set()
    for row in rows:
        pair = str(row["source_pair_id"])
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    require(tuple(pairs) == expected_pairs(), "PAIR_ORDER")
    require(len(rows) == ROW_COUNT, "ROW_COUNT")

    tokenizer, tokenizer_provenance = (
        necessity.tokenizer_gate.load_canonical_analysis_tokenizer(tokenizer_snapshot)
    )
    encoded = necessity.adapter.encode_gen4_rows(rows, tokenizer)
    event_rows = read_jsonl(ROOT / DATA_ROOT / ANCHOR_FILE)
    require(len(event_rows) == ROW_COUNT, "ANCHOR_ROW_COUNT")
    events = necessity.parent.event_lookup(event_rows)
    necessity.parent.validate_transport_event_plan(tuple(pairs), events)
    row_index = necessity.parent.build_row_index(rows)
    require(
        list(encoded["source_pair_id"]) == [str(row["source_pair_id"]) for row in rows],
        "ENCODED_PAIR_ORDER",
    )
    return rows, encoded, events, row_index, tokenizer_provenance


def load_owner_bases():
    return necessity.load_owner_bases()


def load_q_bases():
    return necessity.load_q_bases()


def load_pp3_planes():
    planes = pp3.load_planes()
    require(set(planes) == {"pp3_plus", "pp3_minus", "pp5_plus", "pp5_minus"},
            "PP3_PLANE_SET")
    return planes


def restoration_vectors(
    background_write: torch.Tensor,
    donor_write: torch.Tensor,
    r22: torch.Tensor,
    c22: torch.Tensor,
    condition: str,
) -> dict[str, Any]:
    require(condition in {"B", "RR", "RC"}, f"CONDITION:{condition}")
    background = background_write.detach().cpu().to(torch.float64).reshape(-1).contiguous()
    donor = donor_write.detach().cpu().to(torch.float64).reshape(-1).contiguous()
    require(background.numel() == donor.numel() == STATE_WIDTH, "WRITE_WIDTH")
    require(bool(torch.isfinite(background).all().item()), "BACKGROUND_NONFINITE")
    require(bool(torch.isfinite(donor).all().item()), "DONOR_NONFINITE")

    a_native = (r22.T @ donor).contiguous()
    r_add = (r22 @ a_native).contiguous()
    c_add = (c22 @ a_native).contiguous()
    r_norm = float(torch.linalg.vector_norm(r_add).item())
    c_norm = float(torch.linalg.vector_norm(c_add).item())
    residual = abs(r_norm - c_norm)
    tolerance = max(1e-12, 1e-10 * max(r_norm, c_norm, 1.0))
    require(residual <= tolerance, f"MATCHED_ADDITION_NORM:{residual}:{tolerance}")

    if condition == "B":
        applied = torch.zeros_like(background)
    elif condition == "RR":
        applied = r_add
    else:
        applied = c_add

    return {
        "background": background,
        "donor": donor,
        "a_native": a_native,
        "r_add": r_add,
        "c_add": c_add,
        "applied": applied.contiguous(),
        "modified": (background + applied).contiguous(),
        "r_add_l2": r_norm,
        "c_add_l2": c_norm,
        "matched_addition_norm_residual": residual,
        "matched_addition_norm_tolerance": tolerance,
    }


def endpoint(q_b: float, q_rr: float, q_rc: float) -> dict[str, float]:
    q_b = float(q_b)
    q_rr = float(q_rr)
    q_rc = float(q_rc)
    require(all(math.isfinite(x) for x in (q_b, q_rr, q_rc)), "ENDPOINT_NONFINITE")
    s_r = q_rr - q_b
    s_c = q_rc - q_b
    d = q_rr - q_rc
    require(abs((s_r - s_c) - d) <= 1e-12, "D_SUF22_IDENTITY")
    return {
        "Q_B": q_b,
        "Q_RR": q_rr,
        "Q_RC": q_rc,
        "S_R": s_r,
        "S_C": s_c,
        "D_SUF22": d,
    }


def confirmatory_decision(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(items) == PAIR_COUNT, "DECISION_ITEM_COUNT")
    require(
        [str(row["source_pair_id"]) for row in items] == list(expected_pairs()),
        "DECISION_PAIR_ORDER",
    )
    require(all(row.get("row_dropped") is False for row in items), "ROW_DROPPING")
    q_rr = [float(row["Q_RR"]) for row in items]
    s_r = [float(row["S_R"]) for row in items]
    d = [float(row["D_SUF22"]) for row in items]
    test = necessity.one_sided_one_sample_student_t(d)
    mean_q_rr = sum(q_rr) / PAIR_COUNT
    mean_s_r = sum(s_r) / PAIR_COUNT
    mean_d = sum(d) / PAIR_COUNT
    supported = (
        mean_q_rr > 0.0
        and mean_s_r > 0.0
        and mean_d > 0.0
        and float(test["p_one_sided_greater"]) < 0.05
    )
    return {
        "mean_Q_RR": mean_q_rr,
        "mean_S_R": mean_s_r,
        "mean_D_SUF22": mean_d,
        "confirmatory_test": {
            "test": "one_sided_one_sample_student_t",
            "alternative": "greater_than_zero",
            "endpoint": "D_SUF22",
            **test,
        },
        "confirmatory_p_value_count": 1,
        "label": LABEL_SUPPORTED if supported else LABEL_NOT_ESTABLISHED,
    }


def checksums_bytes(files: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{sha256_bytes(raw)}  {name}\n" for name, raw in sorted(files.items())
    ).encode("utf-8")


def write_outputs(
    output_dir: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".staging-",
        dir=str(output_dir.parent),
    ) as tmp:
        staging = Path(tmp)
        primary = {
            ITEM_FILE: jsonl_bytes(items),
            SUMMARY_FILE: canonical_json_bytes(summary),
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "result": summary["result"],
            "confirmatory_p_value_count": 1,
            "raw_native_vectors_persisted": False,
            "raw_post_state_vectors_persisted": False,
            "files": {
                name: {"bytes": len(raw), "sha256": sha256_bytes(raw)}
                for name, raw in primary.items()
            },
        }
        all_files = {
            **primary,
            MANIFEST_FILE: canonical_json_bytes(manifest),
        }
        all_files[CHECKSUM_FILE] = checksums_bytes(all_files)
        require(
            set(all_files)
            == {ITEM_FILE, SUMMARY_FILE, MANIFEST_FILE, CHECKSUM_FILE},
            "ARTIFACT_BOUNDARY",
        )
        for name, raw in all_files.items():
            (staging / name).write_bytes(raw)
        staging.rename(output_dir)
