from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen5_phase1b_q22_cuda_equivalence as cuda_eq
from scripts import reason_router_gen5_phase1b_r22_c22_construction as construction
from scripts import reason_router_gen5_phase1b_r22_restoration_confirmation as restoration
from scripts import reason_router_gen5_phase1b_r22_restoration_fast_cuda_2gpu as restoration_cuda

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

IMPLEMENTATION_AUTHORITY_COMMIT = "d6cb3a3882fab5dee13ccbf3e53ef62439b62e7a"
IMPLEMENTATION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_phase1b_r22_restoration_2gpu_topology_"
    "equivalence_implementation_authority_spec_candidate.md"
)
IMPLEMENTATION_AUTHORITY_BLOB = "7db6f1adc630d47709833e810cc04546f9a491ff"

RESTORATION_IMPLEMENTATION_FREEZE_COMMIT = (
    "d8d87aa8891ae1f2bef16ed3f1b174e58f87b978"
)
FROZEN_BLOBS = {
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

CONSTRUCTION_DATA_ROOT = Path("data/reason_router_gen5_phase1b_xg1_construction_v1")
CONSTRUCTION_STATIC_SHA256 = {
    "SHA256SUMS.txt":
        "c84902de093eab2ae27f157ed814041d2c26a4a5c272cf0ad7f807ccdbb40855",
    "structured_source_facts.jsonl":
        "3f8eac771794e1d022bee9f669f315c3ac05feeaa9c2195095b8a892b4269ea1",
    "synthetic_reason_router_six_cell.jsonl":
        "5a82508e54cd6097aeee5afe10d7c416357302701d558cfdc8740382168feca4",
    "structural_manifest.json":
        "d68d812ca43a6c651d284f0aa2889f87adb7e030c599af37cb49deaa9d6d8105",
    "tokenizer_anchor_manifest.jsonl":
        "1dbdd3e072245072f197d87443cc8cf81cbbcd2e61437c99b26e0d82815e33b4",
    "tokenizer_eligibility_summary.json":
        "7e304fb1d6f2ecb9ed6fb87911623c47d0b6ae9c2eb9bf9b8f45f70ae1fcb06a",
}

GATE_PAIR_FIRST = 7801
GATE_PAIR_LAST = 7804
GATE_PAIR_COUNT = 4
REFERENCE_PAIRS = tuple(f"xg1_fact_{i}" for i in range(7801, 7805))
CANDIDATE0_PAIRS = tuple(f"xg1_fact_{i}" for i in range(7801, 7803))
CANDIDATE1_PAIRS = tuple(f"xg1_fact_{i}" for i in range(7803, 7805))

FORWARDS_PER_PAIR = 160
REFERENCE_FORWARD_BUDGET = 640
CANDIDATE0_FORWARD_BUDGET = 320
CANDIDATE1_FORWARD_BUDGET = 320
CANDIDATE_TOTAL_FORWARD_BUDGET = 640
TOTAL_GATE_FORWARD_BUDGET = 1280
CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET = 0

FLOAT_ATOL = 1e-9
FLOAT_RTOL = 1e-7

ITEM_FILE = "topology_equivalence_items.jsonl"
SUMMARY_FILE = "topology_equivalence_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
ITEM_SCHEMA = "gen5-phase1b-r22-restoration-2gpu-topology-equivalence-item-v1"
SUMMARY_SCHEMA = "gen5-phase1b-r22-restoration-2gpu-topology-equivalence-summary-v1"
MANIFEST_SCHEMA = "gen5-phase1b-r22-restoration-2gpu-topology-equivalence-manifest-v1"
RESULT_PASS = "PASS_GEN5_PHASE1B_RESTORATION_2GPU_TOPOLOGY_EQUIVALENCE"

WORKER_SPECS = {
    "reference": {
        "physical_gpu": 0,
        "pairs": REFERENCE_PAIRS,
        "forward_budget": REFERENCE_FORWARD_BUDGET,
    },
    "candidate0": {
        "physical_gpu": 0,
        "pairs": CANDIDATE0_PAIRS,
        "forward_budget": CANDIDATE0_FORWARD_BUDGET,
    },
    "candidate1": {
        "physical_gpu": 1,
        "pairs": CANDIDATE1_PAIRS,
        "forward_budget": CANDIDATE1_FORWARD_BUDGET,
    },
}


class TopologyEquivalenceError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise TopologyEquivalenceError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TopologyEquivalenceError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        rows.append(value)
    return rows


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    for commit, label in (
        (IMPLEMENTATION_AUTHORITY_COMMIT, "GATE_IMPLEMENTATION_AUTHORITY"),
        (RESTORATION_IMPLEMENTATION_FREEZE_COMMIT, "RESTORATION_IMPLEMENTATION"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", commit, expected_head) == 0,
            f"{label}_NOT_ANCESTOR",
        )
    require(
        git("rev-parse", f"HEAD:{IMPLEMENTATION_AUTHORITY_PATH}")
        == IMPLEMENTATION_AUTHORITY_BLOB,
        "AUTHORITY_BLOB_DRIFT",
    )
    for path, expected_blob in FROZEN_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected_blob, f"FROZEN_BLOB_DRIFT:{path}:{observed}")
    restoration_cuda.authenticate_repo(expected_head)


def validate_construction_inputs() -> None:
    for name, expected in CONSTRUCTION_STATIC_SHA256.items():
        path = ROOT / CONSTRUCTION_DATA_ROOT / name
        require(path.is_file(), f"CONSTRUCTION_FILE_MISSING:{name}")
        require(sha256_file(path) == expected, f"CONSTRUCTION_SHA256:{name}")
    structural = json.loads(
        (ROOT / CONSTRUCTION_DATA_ROOT / "structural_manifest.json")
        .read_text(encoding="utf-8-sig")
    )
    require(structural.get("role") == "construction", "CONSTRUCTION_ROLE")
    require(
        structural.get("scientific_confirmation_allowed") is False,
        "CONSTRUCTION_CONFIRMATION_BOUNDARY",
    )
    require(
        structural.get("confirmation_response_access_allowed") is False,
        "CONSTRUCTION_CONFIRMATION_RESPONSE_FIREWALL",
    )
    require(
        structural.get("pair_id_first") == "xg1_fact_7801"
        and structural.get("pair_id_last") == "xg1_fact_8100",
        "CONSTRUCTION_PAIR_RANGE",
    )


def gate_seed(
    construction_index: int,
    pair: str,
    events: Mapping[Any, Any],
) -> dict[str, Any]:
    require(0 <= construction_index < GATE_PAIR_COUNT, "GATE_PAIR_INDEX")
    require(pair == REFERENCE_PAIRS[construction_index], "GATE_PAIR_ID")
    anchors = restoration.necessity.phase1._anchors_for_pair(pair, events)
    return {
        "family_key": "xg1",
        "source_pair_id": pair,
        "pair_index": int(construction_index),
        "target_plus_anchor": int(anchors["tp"]),
        "target_minus_anchor": int(anchors["tm"]),
        "reference_plus_anchor": int(anchors["rp"]),
        "reference_minus_anchor": int(anchors["rm"]),
    }


def _global_gate_index(pair: str) -> int:
    require(pair in REFERENCE_PAIRS, f"PAIR_NOT_IN_GATE:{pair}")
    return REFERENCE_PAIRS.index(pair)


def run_worker(args: argparse.Namespace) -> None:
    require(args.worker_role in WORKER_SPECS, "WORKER_ROLE")
    spec = WORKER_SPECS[args.worker_role]
    authenticate_repo(args.expected_head)
    validate_construction_inputs()
    require(
        str(restoration.DATA_ROOT)
        == "data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1",
        "RESTORATION_DATA_ROOT_IDENTITY",
    )

    cuda_eq.backend.runtime_gate()
    require(torch.cuda.device_count() == 1, f"VISIBLE_GPU_COUNT:{torch.cuda.device_count()}")
    require(torch.cuda.get_device_name(0) == "Tesla T4", "VISIBLE_GPU_DEVICE")
    capability = torch.cuda.get_device_capability(0)
    require(tuple(capability) == (7, 5), f"CUDA_CAPABILITY:{capability}")
    require(not args.worker_output.exists(), "WORKER_OUTPUT_COLLISION")
    require(not args.worker_meta.exists(), "WORKER_META_COLLISION")

    r22, c22, basis_geometry = restoration.load_owner_bases()
    q_bases = restoration.load_q_bases()
    pp3_planes = restoration.load_pp3_planes()

    with cuda_eq.backend.parent_runtime_rebind():
        (
            rows,
            encoded,
            events,
            row_index,
            tokenizer_provenance,
        ) = construction.load_inputs(args.tokenizer_snapshot)
        del rows

        kernels = cuda_eq.kernel_compat.load_exact_fast_kernels()
        with cuda_eq.kernel_compat.exact_transformers_kernel_loader(
            kernels
        ) as constructor_calls:
            model, checkpoint_sha = cuda_eq.parent.load_representative_model_external(
                model_snapshot=args.model_snapshot,
                checkpoint_path=args.checkpoint,
            )
            require(
                checkpoint_sha == restoration.necessity.CHECKPOINT_SHA256,
                "CHECKPOINT_SHA256",
            )
            runtime_ctx = cuda_eq.transport_runtime.validate_runtime_components(model)

        counts = Counter(constructor_calls)
        require(
            set(counts) == {"causal-conv1d", "mamba-ssm"}
            and counts["causal-conv1d"] > 0
            and counts["causal-conv1d"] == counts["mamba-ssm"],
            f"KERNEL_CONSTRUCTOR_COUNTS:{dict(counts)}",
        )
        cuda_eq.kernel_compat.validate_transformers_kernel_bindings(kernels)

        model.to(torch.device("cuda:0"))
        model.eval()
        before_signature = cuda_eq.model_parameter_signature(model)
        budget = cuda_eq.parent.ForwardBudget(int(spec["forward_budget"]))
        items: list[dict[str, Any]] = []

        for pair in spec["pairs"]:
            index = _global_gate_index(pair)
            seed = gate_seed(index, pair, events)
            item = restoration_cuda.run_pair(
                seed,
                q_bases=q_bases,
                model=model,
                runtime_ctx=runtime_ctx,
                kernels=kernels,
                encoded=encoded,
                row_index=row_index,
                events=events,
                pp3_planes=pp3_planes,
                r22=r22,
                c22=c22,
                budget=budget,
            )
            require(
                int(item["scientific_model_forward_count_this_run"])
                == FORWARDS_PER_PAIR,
                "PAIR_FORWARD_COUNT",
            )
            items.append(item)

        budget.assert_exact()
        torch.cuda.synchronize()
        after_signature = cuda_eq.model_parameter_signature(model)
        require(before_signature == after_signature, "MODEL_PARAMETER_MUTATION")

    expected_pairs = tuple(spec["pairs"])
    require(
        tuple(str(item["source_pair_id"]) for item in items) == expected_pairs,
        "WORKER_PAIR_ORDER",
    )
    require(
        sum(int(item["scientific_model_forward_count_this_run"]) for item in items)
        == int(spec["forward_budget"]),
        "WORKER_FORWARD_ACCOUNTING",
    )
    require(all(item.get("row_dropped") is False for item in items), "ROW_DROPPING")

    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_bytes(jsonl_bytes(items))
    args.worker_meta.write_bytes(
        canonical_json_bytes(
            {
                "worker_role": args.worker_role,
                "physical_gpu": int(spec["physical_gpu"]),
                "visible_gpu_count": 1,
                "logical_device": "cuda:0",
                "device_name": "Tesla T4",
                "compute_capability": [7, 5],
                "pair_first": expected_pairs[0],
                "pair_last": expected_pairs[-1],
                "pair_count": len(expected_pairs),
                "scientific_model_forward_count": int(spec["forward_budget"]),
                "cpu_scientific_model_forward_count": 0,
                "confirmatory_p_value_count": 0,
                "scientific_conclusion": None,
                "checkpoint_load_count": 1,
                "checkpoint_sha256": checkpoint_sha,
                "model_parameter_signature_before": before_signature,
                "model_parameter_signature_after": after_signature,
                "basis_geometry": basis_geometry,
                "tokenizer": tokenizer_provenance,
            }
        )
    )
    print(f"RESULT=PASS_GATE_WORKER_{args.worker_role.upper()}", flush=True)
    print(f"MODEL_FORWARD_COUNT={spec['forward_budget']}", flush=True)
    print("CONFIRMATORY_P_VALUE_COUNT=0", flush=True)
    print("SCIENTIFIC_CONCLUSION=None", flush=True)


def _skip_equivalence_key(key: str) -> bool:
    return key.lower().endswith("sha256")


def _compare_leaf(
    reference: Any,
    candidate: Any,
    *,
    path: str,
    stats: dict[str, Any],
) -> None:
    if isinstance(reference, bool) or isinstance(candidate, bool):
        require(type(reference) is type(candidate) is bool, f"TYPE:{path}")
        require(reference == candidate, f"BOOL_MISMATCH:{path}")
        stats["exact_leaf_count"] += 1
        return

    if isinstance(reference, int) or isinstance(candidate, int):
        require(
            type(reference) is type(candidate) is int,
            f"TYPE:{path}:{type(reference)}:{type(candidate)}",
        )
        require(reference == candidate, f"INT_MISMATCH:{path}:{reference}:{candidate}")
        stats["exact_leaf_count"] += 1
        return

    if isinstance(reference, float) or isinstance(candidate, float):
        require(
            isinstance(reference, (int, float))
            and not isinstance(reference, bool)
            and isinstance(candidate, (int, float))
            and not isinstance(candidate, bool),
            f"TYPE:{path}",
        )
        ref = float(reference)
        cand = float(candidate)
        require(math.isfinite(ref) and math.isfinite(cand), f"NONFINITE:{path}")
        diff = abs(ref - cand)
        bound = FLOAT_ATOL + FLOAT_RTOL * max(abs(ref), abs(cand), 1.0)
        ratio = diff / bound if bound > 0.0 else (0.0 if diff == 0.0 else math.inf)
        require(diff <= bound, f"FLOAT_MISMATCH:{path}:{diff}:{bound}")
        stats["float_leaf_count"] += 1
        stats["max_abs_diff"] = max(float(stats["max_abs_diff"]), diff)
        stats["max_bound_usage_ratio"] = max(
            float(stats["max_bound_usage_ratio"]), ratio
        )
        if ratio >= float(stats["worst_bound_usage_ratio"]):
            stats["worst_bound_usage_ratio"] = ratio
            stats["worst_float_path"] = path
            stats["worst_float_abs_diff"] = diff
            stats["worst_float_bound"] = bound
        return

    require(type(reference) is type(candidate), f"TYPE:{path}")
    require(reference == candidate, f"EXACT_MISMATCH:{path}:{reference!r}:{candidate!r}")
    stats["exact_leaf_count"] += 1


def compare_values(
    reference: Any,
    candidate: Any,
    *,
    path: str,
    stats: dict[str, Any],
) -> None:
    if isinstance(reference, Mapping):
        require(isinstance(candidate, Mapping), f"DICT_TYPE:{path}")
        require(set(reference) == set(candidate), f"DICT_KEYS:{path}")
        for key in sorted(reference):
            child = f"{path}.{key}" if path else str(key)
            if _skip_equivalence_key(str(key)):
                stats["skipped_sha256_leaf_count"] += 1
                continue
            compare_values(
                reference[key],
                candidate[key],
                path=child,
                stats=stats,
            )
        return

    if isinstance(reference, (list, tuple)):
        require(isinstance(candidate, type(reference)), f"SEQUENCE_TYPE:{path}")
        require(len(reference) == len(candidate), f"SEQUENCE_LEN:{path}")
        for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
            compare_values(
                left, right, path=f"{path}[{index}]", stats=stats
            )
        return

    _compare_leaf(reference, candidate, path=path, stats=stats)


def compare_pair(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        reference.get("source_pair_id") == candidate.get("source_pair_id"),
        "PAIR_IDENTITY",
    )
    stats: dict[str, Any] = {
        "exact_leaf_count": 0,
        "float_leaf_count": 0,
        "skipped_sha256_leaf_count": 0,
        "max_abs_diff": 0.0,
        "max_bound_usage_ratio": 0.0,
        "worst_bound_usage_ratio": -1.0,
        "worst_float_path": None,
        "worst_float_abs_diff": 0.0,
        "worst_float_bound": 0.0,
    }
    compare_values(reference, candidate, path="", stats=stats)
    return {
        "schema_version": ITEM_SCHEMA,
        "source_pair_id": str(reference["source_pair_id"]),
        "pair_index": int(reference["pair_index"]),
        "exact_discrete_equivalence_pass": True,
        "floating_equivalence_pass": True,
        "exact_leaf_count": int(stats["exact_leaf_count"]),
        "float_leaf_count": int(stats["float_leaf_count"]),
        "skipped_tensor_sha256_leaf_count":
            int(stats["skipped_sha256_leaf_count"]),
        "float_atol": FLOAT_ATOL,
        "float_rtol": FLOAT_RTOL,
        "max_float_abs_diff": float(stats["max_abs_diff"]),
        "max_float_bound_usage_ratio": float(stats["max_bound_usage_ratio"]),
        "worst_float_path": stats["worst_float_path"],
        "worst_float_abs_diff": float(stats["worst_float_abs_diff"]),
        "worst_float_bound": float(stats["worst_float_bound"]),
        "equivalence_pass": True,
    }


def merge_candidate(
    candidate0: Sequence[Mapping[str, Any]],
    candidate1: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    require(
        tuple(str(x["source_pair_id"]) for x in candidate0) == CANDIDATE0_PAIRS,
        "CANDIDATE0_ORDER",
    )
    require(
        tuple(str(x["source_pair_id"]) for x in candidate1) == CANDIDATE1_PAIRS,
        "CANDIDATE1_ORDER",
    )
    merged = [dict(x) for x in candidate0] + [dict(x) for x in candidate1]
    require(
        tuple(str(x["source_pair_id"]) for x in merged) == REFERENCE_PAIRS,
        "CANDIDATE_MERGE_ORDER",
    )
    require(len({str(x["source_pair_id"]) for x in merged}) == GATE_PAIR_COUNT,
            "CANDIDATE_DUPLICATE_PAIR")
    require(
        sum(int(x["scientific_model_forward_count_this_run"]) for x in merged)
        == CANDIDATE_TOTAL_FORWARD_BUDGET,
        "CANDIDATE_TOTAL_FORWARD_BUDGET",
    )
    return merged


def _launch_worker(
    *,
    role: str,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint: Path,
    tmpdir: Path,
) -> tuple[subprocess.Popen[Any], Path, Path]:
    require(role in WORKER_SPECS, f"ROLE:{role}")
    output = tmpdir / f"{role}.jsonl"
    meta = tmpdir / f"{role}.meta.json"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence",
        "--worker",
        "--worker-role", role,
        "--expected-head", expected_head,
        "--model-snapshot", str(model_snapshot),
        "--tokenizer-snapshot", str(tokenizer_snapshot),
        "--checkpoint", str(checkpoint),
        "--worker-output", str(output),
        "--worker-meta", str(meta),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(WORKER_SPECS[role]["physical_gpu"])
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    return proc, output, meta


def _wait_worker(
    launched: tuple[subprocess.Popen[Any], Path, Path],
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proc, output, meta_path = launched
    rc = proc.wait()
    require(rc == 0, f"WORKER_EXIT:{role}:{rc}")
    require(output.is_file(), f"WORKER_OUTPUT_MISSING:{role}")
    require(meta_path.is_file(), f"WORKER_META_MISSING:{role}")
    items = read_jsonl(output)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    require(meta.get("worker_role") == role, f"WORKER_META_ROLE:{role}")
    require(int(meta.get("confirmatory_p_value_count", -1)) == 0,
            f"WORKER_P_VALUE:{role}")
    require(meta.get("scientific_conclusion") is None, f"WORKER_CONCLUSION:{role}")
    require(
        meta.get("model_parameter_signature_before")
        == meta.get("model_parameter_signature_after"),
        f"WORKER_PARAMETER_MUTATION:{role}",
    )
    require(
        int(meta.get("scientific_model_forward_count", -1))
        == int(WORKER_SPECS[role]["forward_budget"]),
        f"WORKER_BUDGET:{role}",
    )
    return items, meta


def checksums_bytes(files: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{sha256_bytes(raw)}  {name}\n"
        for name, raw in sorted(files.items())
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
    ) as temp:
        staging = Path(temp)
        primary = {
            ITEM_FILE: jsonl_bytes(items),
            SUMMARY_FILE: canonical_json_bytes(summary),
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "result": summary["result"],
            "confirmatory_p_value_count": 0,
            "scientific_conclusion": None,
            "raw_native_vectors_persisted": False,
            "raw_post_state_vectors_persisted": False,
            "files": {
                name: {
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                }
                for name, raw in primary.items()
            },
        }
        files = {
            **primary,
            MANIFEST_FILE: canonical_json_bytes(manifest),
        }
        files[CHECKSUM_FILE] = checksums_bytes(files)
        require(
            set(files)
            == {ITEM_FILE, SUMMARY_FILE, MANIFEST_FILE, CHECKSUM_FILE},
            "ARTIFACT_BOUNDARY",
        )
        for name, raw in files.items():
            (staging / name).write_bytes(raw)
        staging.rename(output_dir)


def run_coordinator(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head)
    validate_construction_inputs()
    require(not args.output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() == 2, f"EXACT_TWO_GPU_REQUIRED:{torch.cuda.device_count()}")
    require(
        [torch.cuda.get_device_name(i) for i in range(2)]
        == ["Tesla T4", "Tesla T4"],
        "EXACT_TWO_T4_REQUIRED",
    )
    require(
        [tuple(torch.cuda.get_device_capability(i)) for i in range(2)]
        == [(7, 5), (7, 5)],
        "EXACT_TWO_CAPABILITY_REQUIRED",
    )

    with tempfile.TemporaryDirectory(prefix="gen5-r22-restoration-topology-gate-") as temp:
        tmpdir = Path(temp)

        reference_launch = _launch_worker(
            role="reference",
            expected_head=args.expected_head,
            model_snapshot=args.model_snapshot,
            tokenizer_snapshot=args.tokenizer_snapshot,
            checkpoint=args.checkpoint,
            tmpdir=tmpdir,
        )
        reference, reference_meta = _wait_worker(
            reference_launch, "reference"
        )

        candidate0_launch = _launch_worker(
            role="candidate0",
            expected_head=args.expected_head,
            model_snapshot=args.model_snapshot,
            tokenizer_snapshot=args.tokenizer_snapshot,
            checkpoint=args.checkpoint,
            tmpdir=tmpdir,
        )
        candidate1_launch = _launch_worker(
            role="candidate1",
            expected_head=args.expected_head,
            model_snapshot=args.model_snapshot,
            tokenizer_snapshot=args.tokenizer_snapshot,
            checkpoint=args.checkpoint,
            tmpdir=tmpdir,
        )
        candidate0, candidate0_meta = _wait_worker(
            candidate0_launch, "candidate0"
        )
        candidate1, candidate1_meta = _wait_worker(
            candidate1_launch, "candidate1"
        )

        require(
            tuple(str(x["source_pair_id"]) for x in reference) == REFERENCE_PAIRS,
            "REFERENCE_ORDER",
        )
        require(
            sum(int(x["scientific_model_forward_count_this_run"]) for x in reference)
            == REFERENCE_FORWARD_BUDGET,
            "REFERENCE_FORWARD_ACCOUNTING",
        )
        candidate = merge_candidate(candidate0, candidate1)

        comparison_items = [
            compare_pair(ref, cand)
            for ref, cand in zip(reference, candidate, strict=True)
        ]
        require(
            all(item["equivalence_pass"] is True for item in comparison_items),
            "PAIR_EQUIVALENCE",
        )

        worker_metas = [
            reference_meta,
            candidate0_meta,
            candidate1_meta,
        ]
        require(
            all(int(meta["cpu_scientific_model_forward_count"]) == 0
                for meta in worker_metas),
            "CPU_SCIENTIFIC_FORWARD_COUNT",
        )
        require(
            all(int(meta["confirmatory_p_value_count"]) == 0
                for meta in worker_metas),
            "WORKER_CONFIRMATORY_P_VALUE_COUNT",
        )
        total_forward_count = sum(
            int(meta["scientific_model_forward_count"])
            for meta in worker_metas
        )
        require(total_forward_count == TOTAL_GATE_FORWARD_BUDGET,
                "TOTAL_GATE_FORWARD_BUDGET")

        max_abs = max(float(x["max_float_abs_diff"]) for x in comparison_items)
        max_ratio = max(
            float(x["max_float_bound_usage_ratio"])
            for x in comparison_items
        )

        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "result": RESULT_PASS,
            "execution_head": args.expected_head,
            "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
            "restoration_implementation_freeze_commit":
                RESTORATION_IMPLEMENTATION_FREEZE_COMMIT,
            "gate_pair_count": GATE_PAIR_COUNT,
            "pair_id_first": "xg1_fact_7801",
            "pair_id_last": "xg1_fact_7804",
            "reference_topology": {
                "physical_gpu": 0,
                "visible_gpu_count": 1,
                "logical_device": "cuda:0",
                "pair_count": 4,
                "forward_budget": REFERENCE_FORWARD_BUDGET,
            },
            "candidate_topology": {
                "physical_gpu_count": 2,
                "worker0_physical_gpu": 0,
                "worker1_physical_gpu": 1,
                "worker0_pairs": list(CANDIDATE0_PAIRS),
                "worker1_pairs": list(CANDIDATE1_PAIRS),
                "worker0_forward_budget": CANDIDATE0_FORWARD_BUDGET,
                "worker1_forward_budget": CANDIDATE1_FORWARD_BUDGET,
                "total_forward_budget": CANDIDATE_TOTAL_FORWARD_BUDGET,
                "canonical_merge_order": list(REFERENCE_PAIRS),
            },
            "total_gate_model_forward_count": TOTAL_GATE_FORWARD_BUDGET,
            "cuda_scientific_model_forward_count_this_run":
                TOTAL_GATE_FORWARD_BUDGET,
            "cpu_scientific_model_forward_count_this_run": 0,
            "checkpoint_load_count_this_run": 3,
            "confirmatory_p_value_count": 0,
            "scientific_conclusion": None,
            "float_atol": FLOAT_ATOL,
            "float_rtol": FLOAT_RTOL,
            "max_float_abs_diff": max_abs,
            "max_float_bound_usage_ratio": max_ratio,
            "all_discrete_equivalence_pass": True,
            "all_floating_equivalence_pass": True,
            "restoration_confirmation_population_loaded": False,
            "training_executed": False,
            "backward_executed": False,
            "row_dropping_executed": False,
            "raw_native_vectors_persisted": False,
            "raw_post_state_vectors_persisted": False,
            "qualified_backend":
                restoration_cuda.QUALIFIED_BACKEND,
            "representative_checkpoint_sha256":
                restoration.necessity.CHECKPOINT_SHA256,
            "r22_basis_sha256": restoration.necessity.R22_SHA256,
            "c22_basis_sha256": restoration.necessity.C22_SHA256,
            "next_stage":
                "FULL_RESTORATION_CONFIRMATION_EXECUTION_AUTHORITY_ONLY_AFTER_IMPORT_VALIDATION",
        }
        write_outputs(args.output_dir, items=comparison_items, summary=summary)

    print("RESULT=" + RESULT_PASS)
    print("PAIR_ID_FIRST=xg1_fact_7801")
    print("PAIR_ID_LAST=xg1_fact_7804")
    print("REFERENCE_FORWARD_COUNT=640")
    print("CANDIDATE_GPU0_FORWARD_COUNT=320")
    print("CANDIDATE_GPU1_FORWARD_COUNT=320")
    print("TOTAL_GATE_MODEL_FORWARD_COUNT=1280")
    print("CPU_SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN=0")
    print("CONFIRMATORY_P_VALUE_COUNT=0")
    print("SCIENTIFIC_CONCLUSION=None")
    print("MAX_FLOAT_ABS_DIFF=" + repr(max_abs))
    print("MAX_FLOAT_BOUND_USAGE_RATIO=" + repr(max_ratio))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--worker-role",
        choices=tuple(WORKER_SPECS),
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-meta", type=Path)
    args = parser.parse_args(argv)

    if args.worker:
        require(args.worker_role in WORKER_SPECS, "WORKER_ROLE_REQUIRED")
        require(
            args.worker_output is not None and args.worker_meta is not None,
            "WORKER_OUTPUTS_REQUIRED",
        )
        require(args.output_dir is None, "WORKER_OUTPUT_DIR_FORBIDDEN")
    else:
        require(args.worker_role is None, "COORDINATOR_WORKER_ROLE_FORBIDDEN")
        require(args.output_dir is not None, "OUTPUT_DIR_REQUIRED")
        require(
            args.worker_output is None and args.worker_meta is None,
            "COORDINATOR_WORKER_OUTPUT_FORBIDDEN",
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker:
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
