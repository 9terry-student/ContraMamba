#!/usr/bin/env python3
"""Gen5 Phase 3A contention runner.

Implementation authority:
    ccda16d6c7ad321588ec6f5321457e0afb7a1d9e

Modes:
  --static-verify-only
      CPU/read-only authentication. No checkpoint/model/CUDA/forward.
  --cuda-preflight-only
      One-pressure runtime verification. No optimizer step/training.
      Execution requires a later explicit authority.
  --run-cell
      One frozen Phase-3A C0 cell. Execution requires a later explicit
      authority. Presence of this mode is implementation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from contramamba.gen5_phase2_state_update_ownership import (  # noqa: E402
    correction_optimizer_parameters,
    correction_parameter_audit,
    install_phase2_layer22_wrapper,
    load_frozen_owner_bases,
    parent_parameter_fingerprint,
    phase2_active_mask,
    phase2_final_three_way_ce,
)
from contramamba.gen5_phase3_causal_role_contention import (  # noqa: E402
    PRESSURES,
    batch_stressor_hook,
    contention_fractions,
    derive_strong_partition,
    load_frozen_planes,
)

EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
IMPLEMENTATION_AUTHORITY_COMMIT = "ccda16d6c7ad321588ec6f5321457e0afb7a1d9e"
IMPLEMENTATION_AUTHORITY_PATH = (
    "reports/reason_router_gen5_phase3a_contention_stressor_"
    "implementation_authority_spec_candidate.md"
)
IMPLEMENTATION_AUTHORITY_BLOB = "6fb3b68798ffbbf4f8a79d0cf3155de15a5a49c3"

STATIC_FREEZE_COMMIT = "654992ab9e2f9b77fd270ee6bf889c7dcddedd51"
STATIC_TREE_SHA256 = "f244b4b614a6632bb3c7a9dd98c3213e357e8212cc445cad85a062e9d01b6053"
STATIC_EXECUTION_HEAD = "c77f4adb98a949391b44e498488778e722d65eee"

FROZEN_BLOBS = {
    IMPLEMENTATION_AUTHORITY_PATH: IMPLEMENTATION_AUTHORITY_BLOB,
    "src/contramamba/gen5_phase2_state_update_ownership.py":
        "a5c18f5b5d597d9830c3c44a9299af8233677f37",
    "scripts/train_reason_router_gen5_phase2_state_update_ownership.py":
        "8f711776c6b8cab90fbdafcda166780f3295e4ae",
    "scripts/reason_router_gen4_pp3_necessity_fast_cuda.py":
        "26ca67ad8603799a849c151a39728368227326df",
    "scripts/reason_router_gen4_k_directional_alignment_transport_core.py":
        "d98b2dcd3436433c04bb56ecc57dec4240abe820",
}

AUTHORIZED_IMPLEMENTATION_PATHS = frozenset({
    "src/contramamba/gen5_phase3_causal_role_contention.py",
    "scripts/train_reason_router_gen5_phase3a_contention.py",
    "tests/test_reason_router_gen5_phase3_causal_role_contention.py",
    "tests/test_train_reason_router_gen5_phase3a_contention.py",
})

TRAIN_ROOT = Path("data/reason_router_gen5_phase3_xg1_contention_training_v1")
ASSAY_ROOT = Path("data/reason_router_gen5_phase3_xg1_ownership_interaction_assay_v1")
STATIC_REPORT_ROOT = Path("reports/reason_router_gen5_phase3_static_preparation_c77f4ad_v1")

STATIC_FILES = (
    "data/reason_router_gen5_phase3_xg1_contention_training_v1/SHA256SUMS.txt",
    "data/reason_router_gen5_phase3_xg1_contention_training_v1/pair_split_manifest.json",
    "data/reason_router_gen5_phase3_xg1_contention_training_v1/phase3_seven_cell_labeled_training.jsonl",
    "data/reason_router_gen5_phase3_xg1_contention_training_v1/stressor_target_manifest.jsonl",
    "data/reason_router_gen5_phase3_xg1_contention_training_v1/structural_manifest.json",
    "data/reason_router_gen5_phase3_xg1_contention_training_v1/structured_source_facts.jsonl",
    "data/reason_router_gen5_phase3_xg1_contention_training_v1/synthetic_reason_router_six_cell.jsonl",
    "data/reason_router_gen5_phase3_xg1_contention_training_v1/tokenizer_anchor_manifest.jsonl",
    "data/reason_router_gen5_phase3_xg1_contention_training_v1/tokenizer_eligibility_summary.json",
    "data/reason_router_gen5_phase3_xg1_ownership_interaction_assay_v1/SHA256SUMS.txt",
    "data/reason_router_gen5_phase3_xg1_ownership_interaction_assay_v1/structural_manifest.json",
    "data/reason_router_gen5_phase3_xg1_ownership_interaction_assay_v1/structured_source_facts.jsonl",
    "data/reason_router_gen5_phase3_xg1_ownership_interaction_assay_v1/synthetic_reason_router_six_cell.jsonl",
    "data/reason_router_gen5_phase3_xg1_ownership_interaction_assay_v1/tokenizer_anchor_manifest.jsonl",
    "data/reason_router_gen5_phase3_xg1_ownership_interaction_assay_v1/tokenizer_eligibility_summary.json",
    "reports/reason_router_gen5_phase3_static_preparation_c77f4ad_v1/SHA256SUMS.txt",
    "reports/reason_router_gen5_phase3_static_preparation_c77f4ad_v1/static_preparation_summary.json",
)

TRAIN_ROWS = 3360
DEV_ROWS = 840
TRAIN_PAIRS = 480
DEV_PAIRS = 120
SPLIT_SEED = 16384
MAX_LENGTH = 128
CLAIM_BUDGET = 63
EVIDENCE_BUDGET = 64
EOS_TOKEN_ID = 0
PAD_TOKEN_ID = 0
BACKBONE_STREAM_ROWS = 240

TRAINING_SEEDS = (6201, 6202, 6203)
TRAINING_PRESSURES = ("P0", "PR", "PC")
ARM = "G5-C0"

EPOCHS = 20
TOTAL_OPTIMIZER_STEPS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP_NORM = 5.0

PARENT_CHECKPOINT_SHA256 = "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
R22_SHA256 = "a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214"
C22_SHA256 = "c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4"

LABEL_TO_ID = {"REFUTE": 0, "NOT_ENTITLED": 1, "SUPPORT": 2}
STRESSOR_CELLS = {"C0_SHAM", "C1_TITLE", "C2_NAME", "C5_TITLE_NAME"}


class Phase3ARunnerError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Phase3ARunnerError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase3ARunnerError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _status_paths() -> set[str]:
    raw = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    result = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        require(len(line) >= 4, f"MALFORMED_STATUS:{line!r}")
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        result.add(path)
    return result


def validate_checkout_identity(
    branch: str,
    head: str,
    expected_head: str,
) -> str:
    """Accept the local research branch or Kaggle's detached exact HEAD."""
    require(head == expected_head, f"HEAD:{head}")

    if branch == EXPECTED_BRANCH:
        return "attached_expected_branch"

    if branch == "":
        return "detached_exact_head"

    raise Phase3ARunnerError(f"BRANCH:{branch}")


def authenticate_repo(expected_head: str, *, allow_opening_worktree: bool = False) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    checkout_mode = validate_checkout_identity(
        branch,
        head,
        expected_head,
    )

    for ancestor, label in (
        (STATIC_FREEZE_COMMIT, "STATIC_FREEZE"),
        (IMPLEMENTATION_AUTHORITY_COMMIT, "IMPLEMENTATION_AUTHORITY"),
    ):
        require(git_rc("merge-base", "--is-ancestor", ancestor, head) == 0, f"{label}_NOT_ANCESTOR")

    status = _status_paths()
    if allow_opening_worktree:
        require(status <= AUTHORIZED_IMPLEMENTATION_PATHS, f"UNAUTHORIZED_WORKTREE_PATHS:{sorted(status)}")
    else:
        require(not status, f"WORKTREE_NOT_CLEAN:{sorted(status)}")

    for path, expected_blob in FROZEN_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected_blob, f"FROZEN_BLOB:{path}:{observed}")

    if head != IMPLEMENTATION_AUTHORITY_COMMIT:
        changed = {
            line.strip().replace("\\", "/")
            for line in git("diff", "--name-only", f"{IMPLEMENTATION_AUTHORITY_COMMIT}..{head}").splitlines()
            if line.strip()
        }
        require(changed <= AUTHORIZED_IMPLEMENTATION_PATHS, f"POST_AUTHORITY_SCOPE:{sorted(changed)}")

    return {
        "branch": branch,
        "checkout_mode": checkout_mode,
        "head": head,
        "static_freeze_commit": STATIC_FREEZE_COMMIT,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
    }


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_number}")
                rows.append(value)
    return rows


def git_blob_bytes(path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{path}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase3ARunnerError(f"GIT_BLOB_BYTES:{path}") from exc


def authenticate_static_tree() -> dict[str, Any]:
    rows = []
    for rel in STATIC_FILES:
        require((ROOT / rel).is_file(), f"STATIC_FILE_MISSING:{rel}")
        raw = git_blob_bytes(rel)
        rows.append((rel, len(raw), sha256_bytes(raw)))
    rows.sort(key=lambda x: x[0])
    canonical = "".join(
        f"{rel}\t{size}\t{digest}\n"
        for rel, size, digest in rows
    ).encode("utf-8")
    observed = sha256_bytes(canonical)
    require(observed == STATIC_TREE_SHA256, f"STATIC_TREE_SHA256:{observed}")
    return {
        "file_count": len(rows),
        "canonical_bytes": len(canonical),
        "tree_sha256": observed,
    }


def validate_static_artifacts() -> dict[str, Any]:
    tree = authenticate_static_tree()

    view = read_jsonl(ROOT / TRAIN_ROOT / "phase3_seven_cell_labeled_training.jsonl")
    split = json.loads(
        (ROOT / TRAIN_ROOT / "pair_split_manifest.json").read_text(encoding="utf-8-sig")
    )
    targets = read_jsonl(ROOT / TRAIN_ROOT / "stressor_target_manifest.jsonl")
    summary = json.loads(
        (ROOT / STATIC_REPORT_ROOT / "static_preparation_summary.json").read_text(encoding="utf-8-sig")
    )

    require(len(view) == 4200, f"LABELED_VIEW_ROWS:{len(view)}")
    require(
        Counter(str(row["final_label"]) for row in view)
        == Counter({"REFUTE": 600, "NOT_ENTITLED": 3000, "SUPPORT": 600}),
        "LABEL_COUNTS",
    )
    require(split["split_seed"] == SPLIT_SEED, "SPLIT_SEED")
    require(split["train_pair_count"] == TRAIN_PAIRS, "TRAIN_PAIR_COUNT")
    require(split["dev_pair_count"] == DEV_PAIRS, "DEV_PAIR_COUNT")
    require(split["train_row_count"] == TRAIN_ROWS, "TRAIN_ROW_COUNT")
    require(split["dev_row_count"] == DEV_ROWS, "DEV_ROW_COUNT")
    require(len(targets) == 2400, f"STRESSOR_TARGET_COUNT:{len(targets)}")
    require(
        Counter(str(row["contrast_cell_id"]) for row in targets)
        == Counter({cell: 600 for cell in STRESSOR_CELLS}),
        "STRESSOR_TARGET_CELLS",
    )

    target_map = {}
    for row in targets:
        key = (str(row["source_pair_id"]), str(row["contrast_cell_id"]))
        require(key not in target_map, f"STRESSOR_TARGET_DUPLICATE:{key}")
        require(row["anchor_name"] == "A_IDENTITY", f"ANCHOR_NAME:{key}")
        require(int(row["target_offset"]) == 2, f"TARGET_OFFSET:{key}")
        require(int(row["intervention_layer"]) == 17, f"LAYER:{key}")
        require(bool(row["post4_eligible"]), f"POST4:{key}")
        target_map[key] = int(row["intervention_token_index"])

    train_pair_set = set(str(x) for x in split["train_pair_ids"])
    dev_pair_set = set(str(x) for x in split["dev_pair_ids"])
    require(not (train_pair_set & dev_pair_set), "PAIR_SPLIT_OVERLAP")

    train_rows = [row for row in view if str(row["pair_id"]) in train_pair_set]
    dev_rows = [row for row in view if str(row["pair_id"]) in dev_pair_set]
    require(len(train_rows) == TRAIN_ROWS, "FILTERED_TRAIN_ROWS")
    require(len(dev_rows) == DEV_ROWS, "FILTERED_DEV_ROWS")

    for row in view:
        label = str(row["final_label"])
        require(int(row["final_label_id"]) == LABEL_TO_ID[label], "LABEL_ID")
        cell = str(row["contrast_cell_id"])
        expected_active = cell in STRESSOR_CELLS
        require(bool(row["stressor_domain"]) == expected_active, "STRESSOR_DOMAIN")
        key = (str(row["source_pair_id"]), cell)
        require((key in target_map) == expected_active, f"STRESSOR_TARGET_MEMBERSHIP:{key}")

    require(summary["result"] == "PASS_GEN5_PHASE3_STATIC_PREPARATION", "STATIC_SUMMARY_RESULT")
    require(summary["execution_head"] == STATIC_EXECUTION_HEAD, "STATIC_EXECUTION_HEAD")
    require(summary["scientific_model_forward_count"] == 0, "STATIC_MODEL_FORWARD")
    require(summary["checkpoint_load_count"] == 0, "STATIC_CHECKPOINT_LOAD")
    require(summary["cuda_executed"] is False, "STATIC_CUDA")
    require(summary["training_executed"] is False, "STATIC_TRAINING")
    require(summary["p_value_count"] == 0, "STATIC_P_VALUE")

    return {
        "tree": tree,
        "view": view,
        "split": split,
        "target_map": target_map,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "summary": summary,
    }


def row_order_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    value = [
        {
            "id": str(row["id"]),
            "pair_id": str(row["pair_id"]),
            "contrast_cell_id": str(row["contrast_cell_id"]),
            "final_label": str(row["final_label"]),
            "final_label_id": int(row["final_label_id"]),
            "stressor_domain": bool(row["stressor_domain"]),
        }
        for row in rows
    ]
    return sha256_bytes(canonical_json_bytes(value))


def _encode_rows_with_active_tokenizer(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    target_map: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    input_ids = torch.full((len(rows), MAX_LENGTH), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros(len(rows), MAX_LENGTH, dtype=torch.bool)
    claim_mask = torch.zeros_like(attention_mask)
    evidence_mask = torch.zeros_like(attention_mask)
    labels = torch.empty(len(rows), dtype=torch.long)
    stressor_active = torch.zeros(len(rows), dtype=torch.bool)
    target_indices = torch.full((len(rows),), -1, dtype=torch.long)

    row_ids, pair_ids, cells = [], [], []

    for index, row in enumerate(rows):
        claim = [
            int(x)
            for x in tokenizer.encode(str(row["claim"]), add_special_tokens=False).ids
        ][:CLAIM_BUDGET]
        evidence = [
            int(x)
            for x in tokenizer.encode(str(row["evidence"]), add_special_tokens=False).ids
        ][:EVIDENCE_BUDGET]
        require(bool(claim), f"EMPTY_CLAIM:{row['id']}")
        require(bool(evidence), f"EMPTY_EVIDENCE:{row['id']}")

        combined = claim + [EOS_TOKEN_ID] + evidence
        require(len(combined) <= MAX_LENGTH, f"SERIALIZED_LENGTH:{row['id']}")
        input_ids[index, :len(combined)] = torch.tensor(combined, dtype=torch.long)
        attention_mask[index, :len(combined)] = True
        claim_mask[index, :len(claim)] = True
        evidence_start = len(claim) + 1
        evidence_mask[index, evidence_start:evidence_start + len(evidence)] = True
        labels[index] = int(row["final_label_id"])

        cell = str(row["contrast_cell_id"])
        pair = str(row["source_pair_id"])
        active = bool(row["stressor_domain"])
        stressor_active[index] = active
        if active:
            target = int(target_map[(pair, cell)])
            require(target < len(combined), f"TARGET_OUTSIDE_ACTIVE_SEQUENCE:{row['id']}")
            target_indices[index] = target

        row_ids.append(str(row["row_id"]))
        pair_ids.append(pair)
        cells.append(cell)

    return {
        "model_inputs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "claim_mask": claim_mask,
            "evidence_mask": evidence_mask,
            "final_labels": labels,
        },
        "stressor_active": stressor_active,
        "target_indices": target_indices,
        "row_ids": row_ids,
        "pair_ids": pair_ids,
        "contrast_cell_ids": cells,
    }


def encoded_bundle_sha256(bundle: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in ("row_ids", "pair_ids", "contrast_cell_ids"):
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(canonical_json_bytes(list(bundle[key])))
        h.update(b"\n")
    for key in ("stressor_active", "target_indices"):
        tensor = bundle[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(b"\0")
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(b"\0")
        h.update(tensor.numpy().tobytes())
        h.update(b"\n")
    for key in sorted(bundle["model_inputs"]):
        tensor = bundle["model_inputs"][key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(b"\0")
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(b"\0")
        h.update(tensor.numpy().tobytes())
        h.update(b"\n")
    return h.hexdigest()


def load_runtime_encoding(static: Mapping[str, Any], tokenizer_snapshot: Path | None) -> dict[str, Any]:
    from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as gate
    tokenizer, provenance = gate.load_canonical_analysis_tokenizer(tokenizer_snapshot)
    train = _encode_rows_with_active_tokenizer(static["train_rows"], tokenizer, static["target_map"])
    dev = _encode_rows_with_active_tokenizer(static["dev_rows"], tokenizer, static["target_map"])
    require(train["model_inputs"]["input_ids"].shape == (TRAIN_ROWS, MAX_LENGTH), "TRAIN_ENCODING_SHAPE")
    require(dev["model_inputs"]["input_ids"].shape == (DEV_ROWS, MAX_LENGTH), "DEV_ENCODING_SHAPE")
    return {
        "tokenizer_provenance": provenance,
        "train_bundle": train,
        "dev_bundle": dev,
        "train_encoding_sha256": encoded_bundle_sha256(train),
        "dev_encoding_sha256": encoded_bundle_sha256(dev),
    }


def validate_checkpoint(path: Path) -> str:
    require(path.is_file(), f"CHECKPOINT_MISSING:{path}")
    observed = sha256_file(path)
    require(observed == PARENT_CHECKPOINT_SHA256, f"CHECKPOINT_SHA256:{observed}")
    return observed


def _prepare_runtime_model(
    *,
    snapshot: Path,
    checkpoint_path: Path,
    seed: int,
) -> tuple[torch.nn.Module, Any, dict[str, Any], torch.Tensor, dict[str, torch.Tensor]]:
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train

    require(seed in TRAINING_SEEDS, f"SEED:{seed}")
    runtime, kernel_compat, _backend = p2train.validate_cuda_runtime()
    kernels = kernel_compat.load_exact_fast_kernels()
    r22, c22, _basis_geometry = load_frozen_owner_bases(ROOT)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, constructor_counts = p2train._load_parent_model(
        snapshot=snapshot,
        checkpoint=checkpoint_path,
        device=torch.device("cuda:0"),
        kernel_compat=kernel_compat,
        kernels=kernels,
    )
    parent_before = parent_parameter_fingerprint(model)
    wrapper = install_phase2_layer22_wrapper(
        model,
        arm=ARM,
        r22=r22,
        c22=c22,
        seed=seed,
    )
    require(parent_parameter_fingerprint(model) == parent_before, "PARENT_INSTALL_MUTATION")
    require(int(torch.count_nonzero(wrapper.correction.B_theta.weight).item()) == 0, "B_NOT_ZERO_INITIALIZED")

    mixer17 = model.mamba.layers[17].mixer
    partition = derive_strong_partition(mixer17.conv1d.weight, require_frozen_identity=True)
    planes, plane_geometry = load_frozen_planes(ROOT)

    meta = {
        "runtime": runtime,
        "constructor_counts": constructor_counts,
        "parent_before": parent_before,
        "partition": {
            "mu_k2": partition["mu_k2"],
            "strong_count": int(partition["strong"].numel()),
            "strong_index_sha256": partition["strong_index_sha256"],
        },
        "plane_geometry": plane_geometry,
    }
    return model, wrapper, meta, partition["strong_mask"], planes


def _feature_batch_to_device(
    bundle: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = bundle["model_inputs"]
    features = {
        key: inputs[key].to(device)
        for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask")
    }
    return (
        features,
        inputs["final_labels"].to(device),
        bundle["stressor_active"].to(device),
        bundle["target_indices"].to(device),
    )


def _historical_forward_from_hidden(
    model: torch.nn.Module,
    features: Mapping[str, torch.Tensor],
    hidden_states: torch.Tensor,
) -> Mapping[str, Any]:
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    return p2train._historical_forward_from_hidden(model, features, hidden_states)


def _streamed_backbone_hidden(
    model: torch.nn.Module,
    features: Mapping[str, torch.Tensor],
    stressor_active: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    pressure: str,
    strong_mask: torch.Tensor,
    planes: Mapping[str, torch.Tensor],
    stream_rows: int = BACKBONE_STREAM_ROWS,
) -> tuple[torch.Tensor, int]:
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train

    require(pressure in PRESSURES, f"PRESSURE:{pressure}")
    input_ids = features["input_ids"]
    attention_mask = features["attention_mask"]
    require(tuple(attention_mask.shape) == tuple(input_ids.shape), "ATTENTION_SHAPE")
    row_count = int(input_ids.shape[0])
    require(tuple(stressor_active.shape) == (row_count,), "STRESSOR_ACTIVE_SHAPE")
    require(tuple(target_indices.shape) == (row_count,), "TARGET_INDICES_SHAPE")

    rng_before = p2train._capture_rng_state()
    chunks = []
    mixer17 = model.mamba.layers[17].mixer

    for start in range(0, row_count, stream_rows):
        stop = min(start + stream_rows, row_count)
        chunk_active = stressor_active[start:stop]
        chunk_targets = target_indices[start:stop]

        def chunk_forward(chunk_input_ids: torch.Tensor, chunk_attention_mask: torch.Tensor) -> torch.Tensor:
            with phase2_active_mask(model, chunk_attention_mask):
                with batch_stressor_hook(
                    mixer17.in_proj,
                    pressure=pressure,
                    strong_mask=strong_mask,
                    active_rows=chunk_active,
                    target_indices=chunk_targets,
                    planes=planes,
                ):
                    result = model.mamba(input_ids=chunk_input_ids)
            return result.last_hidden_state

        chunk_hidden = torch_checkpoint(
            chunk_forward,
            input_ids[start:stop],
            attention_mask[start:stop],
            use_reentrant=False,
            preserve_rng_state=True,
        )
        chunks.append(chunk_hidden)

    rng_after = p2train._capture_rng_state()
    p2train._require_rng_state_equal(rng_before, rng_after, f"PHASE3A_STREAM_{pressure}")
    hidden = torch.cat(chunks, dim=0)
    require(int(hidden.shape[0]) == row_count, "STREAM_ROW_COUNT")
    return hidden, len(chunks)


def _streamed_forward(
    model: torch.nn.Module,
    features: Mapping[str, torch.Tensor],
    stressor_active: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    pressure: str,
    strong_mask: torch.Tensor,
    planes: Mapping[str, torch.Tensor],
) -> tuple[Mapping[str, Any], int]:
    hidden, chunks = _streamed_backbone_hidden(
        model,
        features,
        stressor_active,
        target_indices,
        pressure=pressure,
        strong_mask=strong_mask,
        planes=planes,
    )
    return _historical_forward_from_hidden(model, features, hidden), chunks


def run_static_verify(args: argparse.Namespace) -> dict[str, Any]:
    provenance = authenticate_repo(args.expected_head, allow_opening_worktree=args.allow_opening_worktree)
    static = validate_static_artifacts()
    planes, plane_geometry = load_frozen_planes(ROOT)
    del planes
    r22, c22, basis_geometry = load_frozen_owner_bases(ROOT)
    del r22, c22

    report = {
        "schema_version": "GEN5_PHASE3A_STATIC_VERIFY_V1",
        "result": "PASS_GEN5_PHASE3A_STATIC_VERIFY",
        "execution_head": args.expected_head,
        "repository": provenance,
        "static_tree": static["tree"],
        "train_order_sha256": row_order_sha256(static["train_rows"]),
        "dev_order_sha256": row_order_sha256(static["dev_rows"]),
        "plane_geometry": plane_geometry,
        "basis_geometry": basis_geometry,
        "train_rows": TRAIN_ROWS,
        "dev_rows": DEV_ROWS,
        "stressor_target_count": len(static["target_map"]),
        "checkpoint_loaded": False,
        "model_instantiated": False,
        "model_forward_count": 0,
        "cuda_executed": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "training_executed": False,
        "task_evaluation_executed": False,
        "scientific_p_value_count": 0,
    }
    print("RESULT=PASS_GEN5_PHASE3A_STATIC_VERIFY")
    print(f"STATIC_TREE_SHA256={static['tree']['tree_sha256']}")
    print(f"TRAIN_ORDER_SHA256={report['train_order_sha256']}")
    print(f"DEV_ORDER_SHA256={report['dev_order_sha256']}")
    print(f"STRESSOR_TARGET_COUNT={report['stressor_target_count']}")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("P_VALUE_COUNT=0")
    return report


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"OUTPUT_COLLISION:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def run_cuda_preflight(args: argparse.Namespace) -> dict[str, Any]:
    authenticate_repo(args.expected_head)
    static = validate_static_artifacts()
    require(args.pressure in TRAINING_PRESSURES, f"PRESSURE:{args.pressure}")
    require(args.seed in TRAINING_SEEDS, f"SEED:{args.seed}")
    require(args.checkpoint is not None, "CHECKPOINT_REQUIRED")
    require(args.preflight_output is not None, "PREFLIGHT_OUTPUT_REQUIRED")

    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    snapshot = p2train.resolve_exact_snapshot(args.model_snapshot)
    checkpoint_path = Path(args.checkpoint)
    validate_checkpoint(checkpoint_path)
    encoded = load_runtime_encoding(static, args.tokenizer_snapshot)

    model, wrapper, runtime_meta, strong_mask, planes = _prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
    )
    features, labels, active, targets = _feature_batch_to_device(encoded["train_bundle"], torch.device("cuda:0"))

    model.train()
    model.mamba.config.use_cache = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model.zero_grad(set_to_none=True)

    output, chunks = _streamed_forward(
        model,
        features,
        active,
        targets,
        pressure=args.pressure,
        strong_mask=strong_mask,
        planes=planes,
    )
    logits = output["logits"]
    require(tuple(logits.shape) == (TRAIN_ROWS, 3), "PREFLIGHT_LOGIT_SHAPE")
    require(bool(torch.isfinite(logits).all().item()), "PREFLIGHT_LOGIT_FINITE")
    loss = phase2_final_three_way_ce(logits, labels)
    require(bool(torch.isfinite(loss).item()), "PREFLIGHT_LOSS_FINITE")
    loss.backward()

    a_grad = wrapper.correction.A_theta.weight.grad
    b_grad = wrapper.correction.B_theta.weight.grad
    require(a_grad is not None and b_grad is not None, "CORRECTION_GRAD_MISSING")
    require(bool(torch.isfinite(a_grad).all().item()), "A_GRAD_NONFINITE")
    require(bool(torch.isfinite(b_grad).all().item()), "B_GRAD_NONFINITE")
    parent_grads = [
        name
        for name, parameter in model.named_parameters()
        if ".correction." not in name and parameter.grad is not None
    ]
    require(not parent_grads, f"PARENT_GRADIENT:{parent_grads[:5]}")

    report = {
        "schema_version": "GEN5_PHASE3A_CUDA_PREFLIGHT_V1",
        "result": "PASS_GEN5_PHASE3A_CUDA_PREFLIGHT",
        "execution_head": args.expected_head,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "pressure": args.pressure,
        "seed": args.seed,
        "train_rows": TRAIN_ROWS,
        "stream_rows": BACKBONE_STREAM_ROWS,
        "stream_chunk_count": chunks,
        "runtime": runtime_meta,
        "train_order_sha256": row_order_sha256(static["train_rows"]),
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "loss": float(loss.detach().cpu().item()),
        "backward_executed": True,
        "optimizer_step_executed": False,
        "training_executed": False,
        "task_evaluation_executed": False,
        "scientific_p_value_count": 0,
    }
    _write_json_once(Path(args.preflight_output), report)
    print("RESULT=PASS_GEN5_PHASE3A_CUDA_PREFLIGHT")
    print(f"PRESSURE={args.pressure}")
    print(f"SEED={args.seed}")
    print(f"STREAM_CHUNKS={chunks}")
    print("BACKWARD_EXECUTED=True")
    print("OPTIMIZER_STEP_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    return report


def _checkpoint_payload(*, wrapper: Any, args: argparse.Namespace, seed: int, pressure: str) -> dict[str, Any]:
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    a = wrapper.correction.A_theta.weight.detach().cpu().contiguous()
    b = wrapper.correction.B_theta.weight.detach().cpu().contiguous()
    return {
        "schema_version": "GEN5_PHASE3A_FINAL_CORRECTION_V1",
        "execution_commit": args.expected_head,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "static_freeze_commit": STATIC_FREEZE_COMMIT,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "r22_sha256": R22_SHA256,
        "c22_sha256": C22_SHA256,
        "seed": seed,
        "arm": ARM,
        "pressure": pressure,
        "state_dict": {"A_theta.weight": a, "B_theta.weight": b},
        "tensor_sha256": {
            "A_theta.weight": p2train.tensor_sha256(a),
            "B_theta.weight": p2train.tensor_sha256(b),
        },
    }


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    authenticate_repo(args.expected_head)
    static = validate_static_artifacts()
    require(args.pressure in TRAINING_PRESSURES, f"PRESSURE:{args.pressure}")
    require(args.seed in TRAINING_SEEDS, f"SEED:{args.seed}")
    require(args.checkpoint is not None, "CHECKPOINT_REQUIRED")
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")

    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    snapshot = p2train.resolve_exact_snapshot(args.model_snapshot)
    checkpoint_path = Path(args.checkpoint)
    validate_checkpoint(checkpoint_path)
    encoded = load_runtime_encoding(static, args.tokenizer_snapshot)

    cell_dir = Path(args.output_root) / f"seed{args.seed}" / args.pressure
    require(not cell_dir.exists(), f"CELL_OUTPUT_COLLISION:{cell_dir}")
    cell_dir.mkdir(parents=True, exist_ok=False)

    model, wrapper, runtime_meta, strong_mask, planes = _prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
    )
    parent_before = runtime_meta["parent_before"]

    features, labels, active, targets = _feature_batch_to_device(encoded["train_bundle"], torch.device("cuda:0"))
    optimizer_parameters = correction_optimizer_parameters(model)
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()
    model.mamba.config.use_cache = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    losses = []
    grad_norms = []

    for step in range(TOTAL_OPTIMIZER_STEPS):
        optimizer.zero_grad(set_to_none=True)
        output, chunks = _streamed_forward(
            model,
            features,
            active,
            targets,
            pressure=args.pressure,
            strong_mask=strong_mask,
            planes=planes,
        )
        require(chunks == TRAIN_ROWS // BACKBONE_STREAM_ROWS, f"CHUNKS:{chunks}")
        logits = output["logits"]
        require(tuple(logits.shape) == (TRAIN_ROWS, 3), "TRAIN_LOGIT_SHAPE")
        require(bool(torch.isfinite(logits).all().item()), "TRAIN_LOGIT_NONFINITE")
        loss = phase2_final_three_way_ce(logits, labels)
        require(bool(torch.isfinite(loss).item()), f"LOSS_NONFINITE:{step}")
        loss.backward()

        a_grad = wrapper.correction.A_theta.weight.grad
        b_grad = wrapper.correction.B_theta.weight.grad
        require(a_grad is not None and b_grad is not None, f"GRAD_MISSING:{step}")
        parent_grads = [
            name
            for name, parameter in model.named_parameters()
            if ".correction." not in name and parameter.grad is not None
        ]
        require(not parent_grads, f"PARENT_GRADIENT:{parent_grads[:5]}")

        clipped = torch.nn.utils.clip_grad_norm_(optimizer_parameters, GRADIENT_CLIP_NORM)
        require(bool(torch.isfinite(clipped).item()), "GRAD_NORM_NONFINITE")
        optimizer.step()
        require(
            bool(torch.isfinite(wrapper.correction.A_theta.weight).all().item())
            and bool(torch.isfinite(wrapper.correction.B_theta.weight).all().item()),
            "CORRECTION_PARAMETER_NONFINITE",
        )
        losses.append(float(loss.detach().cpu().item()))
        grad_norms.append(float(clipped.detach().cpu().item()))
        del output, logits, loss
        torch.cuda.synchronize()

    require(len(losses) == TOTAL_OPTIMIZER_STEPS, "LOSS_COUNT")
    require(parent_parameter_fingerprint(model) == parent_before, "PARENT_MUTATION")

    r22, c22, _ = load_frozen_owner_bases(ROOT)
    geometry = contention_fractions(
        wrapper.correction.A_theta.weight,
        wrapper.correction.B_theta.weight,
        r22,
        c22,
    )

    payload = _checkpoint_payload(
        wrapper=wrapper,
        args=args,
        seed=args.seed,
        pressure=args.pressure,
    )
    checkpoint_out = cell_dir / "final_correction.pt"
    torch.save(payload, checkpoint_out)

    audit = correction_parameter_audit(model)
    report = {
        "schema_version": "GEN5_PHASE3A_TRAINING_REPORT_V1",
        "result": "PASS_GEN5_PHASE3A_TRAINING_CELL",
        "execution_commit": args.expected_head,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "static_freeze_commit": STATIC_FREEZE_COMMIT,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "seed": args.seed,
        "arm": ARM,
        "pressure": args.pressure,
        "train_rows": TRAIN_ROWS,
        "dev_rows": DEV_ROWS,
        "split_seed": SPLIT_SEED,
        "train_order_sha256": row_order_sha256(static["train_rows"]),
        "dev_order_sha256": row_order_sha256(static["dev_rows"]),
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "dev_encoding_sha256": encoded["dev_encoding_sha256"],
        "training_losses": losses,
        "step0_loss": losses[0],
        "last_preupdate_loss": losses[-1],
        "loss_decreased_from_step0": bool(losses[-1] < losses[0]),
        "gradient_norms_before_clip": grad_norms,
        "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": None,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "checkpoint_selection": "FINAL_FIXED_STEP_ONLY",
        "objective": "FINAL_3WAY_CROSS_ENTROPY_ONLY",
        "contention_geometry": geometry,
        "trainable_tensor_names": audit["trainable_names"],
        "trainable_tensor_count": audit["trainable_tensor_count"],
        "trainable_numel": audit["trainable_numel"],
        "parent_signature_before": parent_before,
        "parent_signature_after": parent_parameter_fingerprint(model),
        "final_correction_file_sha256": sha256_file(checkpoint_out),
        "task_evaluation_executed": False,
        "confirmatory_assay_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    (cell_dir / "training_report.json").write_bytes(canonical_json_bytes(report) + b"\n")

    provenance = {
        "schema_version": "GEN5_PHASE3A_TRAINING_PROVENANCE_V1",
        "status": "PASS",
        "execution_commit": args.expected_head,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "static_tree_sha256": STATIC_TREE_SHA256,
        "seed": args.seed,
        "pressure": args.pressure,
        "arm": ARM,
        "training_report_sha256": sha256_file(cell_dir / "training_report.json"),
        "final_correction_file_sha256": sha256_file(checkpoint_out),
        "training_executed": True,
        "task_evaluation_executed": False,
        "confirmatory_assay_loaded": False,
        "scientific_p_value_count": 0,
    }
    (cell_dir / "run_provenance.json").write_bytes(canonical_json_bytes(provenance) + b"\n")

    print(
        f"CELL_PASS seed={args.seed} pressure={args.pressure} "
        f"step0_loss={losses[0]:.9g} last_loss={losses[-1]:.9g} "
        f"F_R={geometry['F_R']:.9g} F_C={geometry['F_C']:.9g}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-verify-only", action="store_true")
    modes.add_argument("--cuda-preflight-only", action="store_true")
    modes.add_argument("--run-cell", action="store_true")
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--allow-opening-worktree", action="store_true")
    parser.add_argument("--pressure", choices=TRAINING_PRESSURES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def validate_mode_args(args: argparse.Namespace) -> None:
    if args.static_verify_only:
        require(args.pressure is None, "STATIC_PRESSURE_FORBIDDEN")
        require(args.seed is None, "STATIC_SEED_FORBIDDEN")
        require(args.checkpoint is None, "STATIC_CHECKPOINT_FORBIDDEN")
        require(args.preflight_output is None, "STATIC_PREFLIGHT_OUTPUT_FORBIDDEN")
        require(args.output_root is None, "STATIC_OUTPUT_ROOT_FORBIDDEN")
        require(args.model_snapshot is None, "STATIC_MODEL_SNAPSHOT_FORBIDDEN")
        require(args.tokenizer_snapshot is None, "STATIC_TOKENIZER_SNAPSHOT_FORBIDDEN")
        return

    require(not args.allow_opening_worktree, "RUNTIME_OPENING_WORKTREE_FORBIDDEN")
    require(args.pressure in TRAINING_PRESSURES, "RUNTIME_PRESSURE_REQUIRED")
    require(args.seed in TRAINING_SEEDS, "RUNTIME_SEED_REQUIRED")
    require(args.checkpoint is not None, "RUNTIME_CHECKPOINT_REQUIRED")

    if args.cuda_preflight_only:
        require(args.preflight_output is not None, "PREFLIGHT_OUTPUT_REQUIRED")
        require(args.output_root is None, "PREFLIGHT_OUTPUT_ROOT_FORBIDDEN")
    elif args.run_cell:
        require(args.output_root is not None, "RUN_OUTPUT_ROOT_REQUIRED")
        require(args.preflight_output is None, "RUN_PREFLIGHT_OUTPUT_FORBIDDEN")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_mode_args(args)
    if args.static_verify_only:
        run_static_verify(args)
    elif args.cuda_preflight_only:
        run_cuda_preflight(args)
    else:
        run_cell(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
