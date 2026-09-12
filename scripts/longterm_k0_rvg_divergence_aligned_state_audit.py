"""Divergence-aligned layer-23 exact-state audit.

Scientific scope:
- frozen active token IDs only; no tokenizer invocation,
- per-role matched-vs-swapped first token difference is the anchor,
- common full-population window k=-1..+6,
- layer 23 S_prev/G/W/S_post exact SHA256 comparison,
- no logits, no heads, no geometry, no training, no intervention.

The scientific execution path performs direct Mamba-backbone forwards using
frozen input_ids. The existing frozen raw-recurrence observer is reused
without modification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


FROZEN_COMMIT = "75f34389faf6b83de4fb87758be0beeec8f7f2b1"
EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

ACTIVE_REL = (
    "reports/"
    "longterm_k0_rvg_post_p2_active_encoding_token_coordinate_equivalence_ebdecf9_v1/"
    "active_encoding_token_coordinate_equivalence.json"
)

OBSERVER_REL = "scripts/longterm_k0_rvg_raw_recurrence_observer.py"
K2S_REL = "scripts/longterm_k2s_pair_specific_event_dynamics.py"

EXPECTED_OBSERVER_BLOB = "2722b6ee0e3cd04c8167924d41e5fb801e3945e5"
EXPECTED_K2S_BLOB = "3a651fb508669bdcf72441a4869b863d6eee6c1f"

ACTIVE_SCHEMA = "k0-rvg-post-p2-active-encoding-token-coordinate-equivalence-v1"

PRIMARY_LAYER = 23
POST_HORIZON = 6
RELATIVE_COORDINATES = tuple(range(-1, POST_HORIZON + 1))
EXPECTED_ITEM_COUNT = 336
EXPECTED_LAYER_COUNT = 24
HASH_FIELDS = ("S_prev", "G", "W", "S_post")

ROLE_NAMES = {
    "corr": ("matched_corr", "swapped_corr"),
    "ctrl": ("matched_ctrl", "swapped_ctrl"),
}


class AuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AuditError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError(f"GIT_FAILURE:{' '.join(args)}") from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", spec],
            cwd=root,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError(f"GIT_SHOW_FAILURE:{spec}") from exc


def repo_root() -> Path:
    here = Path.cwd().resolve()
    top = Path(git(here, "rev-parse", "--show-toplevel")).resolve()
    require(top == here, f"RUN_FROM_REPO_ROOT_REQUIRED:{top}")
    return top


def authenticate_repo(root: Path) -> dict[str, Any]:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")

    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")

    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, head],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "FROZEN_COMMIT_NOT_ANCESTOR")

    # Tracked modifications are forbidden. Existing unrelated untracked work
    # is intentionally not touched or rejected.
    require(
        git(root, "diff", "--name-only") == "",
        "TRACKED_UNSTAGED_CHANGES_PRESENT",
    )
    require(
        git(root, "diff", "--cached", "--name-only") == "",
        "TRACKED_STAGED_CHANGES_PRESENT",
    )

    observer_blob = git(root, "rev-parse", f"{FROZEN_COMMIT}:{OBSERVER_REL}")
    k2s_blob = git(root, "rev-parse", f"{FROZEN_COMMIT}:{K2S_REL}")

    require(observer_blob == EXPECTED_OBSERVER_BLOB, "OBSERVER_FROZEN_BLOB_MISMATCH")
    require(k2s_blob == EXPECTED_K2S_BLOB, "K2S_FROZEN_BLOB_MISMATCH")

    observer_path = root / OBSERVER_REL
    k2s_path = root / K2S_REL

    require(observer_path.is_file(), "OBSERVER_FILE_MISSING")
    require(k2s_path.is_file(), "K2S_FILE_MISSING")

    require(
        observer_path.read_bytes()
        == git_bytes(root, f"{FROZEN_COMMIT}:{OBSERVER_REL}"),
        "OBSERVER_WORKTREE_DRIFT",
    )
    require(
        k2s_path.read_bytes()
        == git_bytes(root, f"{FROZEN_COMMIT}:{K2S_REL}"),
        "K2S_WORKTREE_DRIFT",
    )

    return {
        "branch": branch,
        "head": head,
        "frozen_commit": FROZEN_COMMIT,
        "observer_blob": observer_blob,
        "k2s_blob": k2s_blob,
        "active_artifact_blob": git(
            root, "rev-parse", f"{FROZEN_COMMIT}:{ACTIVE_REL}"
        ),
    }


def load_active_artifact(root: Path) -> dict[str, Any]:
    raw = git_bytes(root, f"{FROZEN_COMMIT}:{ACTIVE_REL}")
    obj = json.loads(raw)

    require(
        obj.get("schema_version") == ACTIVE_SCHEMA,
        "ACTIVE_SCHEMA_MISMATCH",
    )
    require(obj.get("item_count") == EXPECTED_ITEM_COUNT, "ACTIVE_ITEM_COUNT_FIELD")
    require(
        isinstance(obj.get("items"), list)
        and len(obj["items"]) == EXPECTED_ITEM_COUNT,
        "ACTIVE_ITEMS_INVALID",
    )

    return obj


def first_difference(a: Sequence[int], b: Sequence[int]) -> int | None:
    n = min(len(a), len(b))

    for i in range(n):
        if a[i] != b[i]:
            return i

    if len(a) != len(b):
        return n

    return None


def build_static_plan(active: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = active["items"]

    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    plan: list[dict[str, Any]] = []

    for item in items:
        idx = item.get("local_template_index")
        stable_id = item.get("stable_item_id")

        require(type(idx) is int, "LOCAL_TEMPLATE_INDEX_INVALID")
        require(idx not in seen_indices, f"DUPLICATE_INDEX:{idx}")
        seen_indices.add(idx)

        require(isinstance(stable_id, str) and stable_id, f"STABLE_ID_INVALID:{idx}")
        require(stable_id not in seen_ids, f"DUPLICATE_STABLE_ID:{idx}")
        seen_ids.add(stable_id)

        require(
            item.get("status") == "EXACT_CONTRACT_MATCH",
            f"ACTIVE_ITEM_STATUS_NOT_EXACT:{idx}",
        )

        sequences = item.get("sequences")
        require(isinstance(sequences, Mapping), f"SEQUENCES_MISSING:{idx}")

        for role, (matched_name, swapped_name) in ROLE_NAMES.items():
            matched = sequences.get(matched_name)
            swapped = sequences.get(swapped_name)

            require(isinstance(matched, Mapping), f"MATCHED_SEQUENCE_MISSING:{idx}:{role}")
            require(isinstance(swapped, Mapping), f"SWAPPED_SEQUENCE_MISSING:{idx}:{role}")

            m_ids = matched.get("token_ids")
            s_ids = swapped.get("token_ids")

            require(
                isinstance(m_ids, list)
                and m_ids
                and all(type(v) is int for v in m_ids),
                f"MATCHED_TOKEN_IDS_INVALID:{idx}:{role}",
            )
            require(
                isinstance(s_ids, list)
                and s_ids
                and all(type(v) is int for v in s_ids),
                f"SWAPPED_TOKEN_IDS_INVALID:{idx}:{role}",
            )

            fd = first_difference(m_ids, s_ids)

            require(fd is not None, f"FULL_BRANCH_IDENTICAL:{idx}:{role}")
            require(fd < min(len(m_ids), len(s_ids)), f"STRICT_PREFIX_DIFFERENCE:{idx}:{role}")
            require(fd >= 1, f"DIVERGENCE_AT_SEQUENCE_START:{idx}:{role}")

            common_post = min(len(m_ids), len(s_ids)) - 1 - fd
            require(
                common_post >= POST_HORIZON,
                f"INSUFFICIENT_POST_SUPPORT:{idx}:{role}:{common_post}",
            )

            require(
                m_ids[fd - 1] == s_ids[fd - 1],
                f"PREANCHOR_TOKEN_NOT_EQUAL:{idx}:{role}",
            )
            require(
                m_ids[fd] != s_ids[fd],
                f"ANCHOR_TOKEN_NOT_DIFFERENT:{idx}:{role}",
            )

            targets = tuple(fd + k for k in RELATIVE_COORDINATES)

            require(min(targets) >= 0, f"NEGATIVE_TARGET:{idx}:{role}")
            require(
                max(targets) < len(m_ids) and max(targets) < len(s_ids),
                f"TARGET_OUT_OF_RANGE:{idx}:{role}",
            )

            plan.append(
                {
                    "local_template_index": idx,
                    "stable_item_id": stable_id,
                    "role": role,
                    "matched_branch": matched_name,
                    "swapped_branch": swapped_name,
                    "anchor": fd,
                    "targets": targets,
                    "matched_ids": tuple(m_ids),
                    "swapped_ids": tuple(s_ids),
                    "matched_length": len(m_ids),
                    "swapped_length": len(s_ids),
                    "common_post_horizon": common_post,
                }
            )

    require(len(seen_indices) == EXPECTED_ITEM_COUNT, "UNIQUE_INDEX_COUNT_MISMATCH")
    require(len(seen_ids) == EXPECTED_ITEM_COUNT, "UNIQUE_STABLE_ID_COUNT_MISMATCH")
    require(len(plan) == EXPECTED_ITEM_COUNT * 2, "PLAN_ROW_COUNT_MISMATCH")

    return sorted(
        plan,
        key=lambda row: (
            row["local_template_index"],
            row["role"],
        ),
    )


def static_summary(plan: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    role_anchor_hist: dict[str, Counter[int]] = {
        "corr": Counter(),
        "ctrl": Counter(),
    }
    support_hist: dict[str, Counter[int]] = {
        "corr": Counter(),
        "ctrl": Counter(),
    }

    for row in plan:
        role = str(row["role"])
        role_anchor_hist[role][int(row["anchor"])] += 1
        support_hist[role][int(row["common_post_horizon"])] += 1

    return {
        "schema_version": "k0-rvg-divergence-aligned-static-plan-v1",
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": len(plan),
        "primary_layer": PRIMARY_LAYER,
        "relative_coordinates": list(RELATIVE_COORDINATES),
        "tokenizer_invoked": False,
        "corr_anchor_absolute_histogram": dict(sorted(role_anchor_hist["corr"].items())),
        "ctrl_anchor_absolute_histogram": dict(sorted(role_anchor_hist["ctrl"].items())),
        "corr_common_post_horizon_histogram": dict(sorted(support_hist["corr"].items())),
        "ctrl_common_post_horizon_histogram": dict(sorted(support_hist["ctrl"].items())),
        "minimum_common_post_horizon": min(
            int(row["common_post_horizon"]) for row in plan
        ),
    }


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"MODULE_SPEC_FAILURE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_runtime(root: Path, handoff_path: Path):
    observer = import_module(
        root / OBSERVER_REL,
        "k0_rvg_divergence_observer",
    )
    k2s = import_module(
        root / K2S_REL,
        "k0_rvg_divergence_k2s",
    )

    handoff = k2s.audit_handoff(handoff_path)
    checkpoint = k2s.load_authenticated_checkpoint(handoff)
    encoder = k2s.encoder_fingerprint(checkpoint["model_state_dict"])

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=k2s.HF_MODEL,
            revision=k2s.HF_REVISION,
            allow_patterns=["config.json"],
            local_files_only=True,
        )
    )

    require(snapshot.is_dir(), "HF_LOCAL_SNAPSHOT_MISSING")
    require(
        (snapshot / "config.json").is_file(),
        "HF_LOCAL_CONFIG_MISSING",
    )

    model = k2s.build_a0_model(root, snapshot, checkpoint)
    model.eval()

    binding = observer.resolve_source_binding()
    layer_map = observer.registered_mamba_layers(model)

    if hasattr(k2s, "registered_mamba_layers"):
        require(
            layer_map == k2s.registered_mamba_layers(model),
            "LAYER_MAP_DRIFT",
        )

    import torch

    params = list(model.parameters())
    require(bool(params), "MODEL_HAS_NO_PARAMETERS")
    require(
        {p.device.type for p in params} == {"cpu"},
        "MODEL_NOT_CPU",
    )

    return observer, k2s, model, binding, layer_map, handoff, encoder, snapshot


def direct_backbone_forward(model: Any, token_ids: Sequence[int]) -> None:
    import torch

    input_ids = torch.tensor(
        list(token_ids),
        dtype=torch.long,
    ).unsqueeze(0)

    require(input_ids.device.type == "cpu", "INPUT_IDS_NOT_CPU")

    with torch.inference_mode():
        # Deliberately bypass all ContraMamba task heads and their masks.
        # Only native Mamba recurrence is in scope.
        model.mamba(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )


def capture_branch_hashes(
    observer: Any,
    model: Any,
    layer_map: Mapping[int, int],
    binding: Any,
    token_ids: Sequence[int],
    targets: Sequence[int],
) -> dict[int, dict[str, str]]:
    target_set = tuple(int(v) for v in targets)

    collector = observer.RawRecurrenceCollector(
        binding,
        layer_map,
        target_set,
    )

    with collector.capture():
        direct_backbone_forward(model, token_ids)

    require(collector.records is not None, "CAPTURE_RECORDS_MISSING")

    expected_keys = {
        (layer, token)
        for layer in range(EXPECTED_LAYER_COUNT)
        for token in target_set
    }

    require(
        set(collector.records) == expected_keys,
        "CAPTURE_COORDINATE_SET_MISMATCH",
    )

    out: dict[int, dict[str, str]] = {}

    for token in target_set:
        record = collector.records[(PRIMARY_LAYER, token)]

        require(
            record.layer_index == PRIMARY_LAYER,
            "PRIMARY_LAYER_RECORD_MISMATCH",
        )
        require(
            record.token_index == token,
            "PRIMARY_TOKEN_INDEX_MISMATCH",
        )

        recurrence = observer.validate_recurrence_record(record)

        require(
            recurrence.get("recurrence_exact") == "PASS_EXACT",
            "RECURRENCE_EXACT_FAILURE",
        )

        out[token] = observer.record_hashes(record)

        require(
            set(out[token]) == set(HASH_FIELDS),
            "HASH_FIELD_SET_MISMATCH",
        )

    return out


def comparison_rows_for_pair(
    plan_row: Mapping[str, Any],
    matched_hashes: Mapping[int, Mapping[str, str]],
    swapped_hashes: Mapping[int, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    idx = int(plan_row["local_template_index"])
    role = str(plan_row["role"])
    anchor = int(plan_row["anchor"])
    m_ids = plan_row["matched_ids"]
    s_ids = plan_row["swapped_ids"]

    rows: list[dict[str, Any]] = []
    first_state_difference: int | None = None
    first_difference_fields: list[str] | None = None

    for k in RELATIVE_COORDINATES:
        token = anchor + k
        mh = matched_hashes[token]
        sh = swapped_hashes[token]

        differing_fields = [
            field
            for field in HASH_FIELDS
            if mh[field] != sh[field]
        ]

        if differing_fields and first_state_difference is None:
            first_state_difference = k
            first_difference_fields = list(differing_fields)

        rows.append(
            {
                "schema_version": "k0-rvg-divergence-aligned-state-hash-comparison-v1",
                "local_template_index": idx,
                "stable_item_id": plan_row["stable_item_id"],
                "role": role,
                "matched_branch": plan_row["matched_branch"],
                "swapped_branch": plan_row["swapped_branch"],
                "divergence_anchor_token_index": anchor,
                "relative_coordinate": k,
                "token_index": token,
                "matched_token_id": int(m_ids[token]),
                "swapped_token_id": int(s_ids[token]),
                "token_equal": bool(m_ids[token] == s_ids[token]),
                "matched_hashes": dict(mh),
                "swapped_hashes": dict(sh),
                "differing_hash_fields": differing_fields,
                "all_four_hashes_equal": not differing_fields,
            }
        )

    pair_summary = {
        "schema_version": "k0-rvg-divergence-aligned-pair-summary-v1",
        "local_template_index": idx,
        "stable_item_id": plan_row["stable_item_id"],
        "role": role,
        "matched_branch": plan_row["matched_branch"],
        "swapped_branch": plan_row["swapped_branch"],
        "matched_length": int(plan_row["matched_length"]),
        "swapped_length": int(plan_row["swapped_length"]),
        "divergence_anchor_token_index": anchor,
        "common_post_horizon": int(plan_row["common_post_horizon"]),
        "first_state_hash_difference_relative_coordinate": first_state_difference,
        "first_state_hash_difference_fields": first_difference_fields,
        "state_hash_identity_through_k_plus_6": first_state_difference is None,
    }

    return rows, pair_summary


def make_scientific_summary(
    comparison_rows: Sequence[Mapping[str, Any]],
    pair_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(
        len(comparison_rows)
        == EXPECTED_ITEM_COUNT * 2 * len(RELATIVE_COORDINATES),
        "COMPARISON_ROW_COUNT_MISMATCH",
    )
    require(
        len(pair_summaries) == EXPECTED_ITEM_COUNT * 2,
        "PAIR_SUMMARY_COUNT_MISMATCH",
    )

    first_hist: dict[str, Counter[str]] = {
        "corr": Counter(),
        "ctrl": Counter(),
    }

    for row in pair_summaries:
        role = str(row["role"])
        first = row["first_state_hash_difference_relative_coordinate"]
        label = (
            "IDENTITY_THROUGH_K_PLUS_6"
            if first is None
            else str(first)
        )
        first_hist[role][label] += 1

    field_coord: dict[str, Counter[int]] = {
        field: Counter() for field in HASH_FIELDS
    }

    for row in comparison_rows:
        k = int(row["relative_coordinate"])
        for field in row["differing_hash_fields"]:
            field_coord[field][k] += 1

    token_contract_k_minus_1 = sum(
        bool(row["token_equal"])
        for row in comparison_rows
        if row["relative_coordinate"] == -1
    )

    token_contract_k_zero_different = sum(
        not bool(row["token_equal"])
        for row in comparison_rows
        if row["relative_coordinate"] == 0
    )

    identity_pairs = sum(
        bool(row["state_hash_identity_through_k_plus_6"])
        for row in pair_summaries
    )

    by_item: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_summaries:
        by_item[int(row["local_template_index"])].append(row)

    both_identity_items = sum(
        len(rows) == 2
        and all(bool(r["state_hash_identity_through_k_plus_6"]) for r in rows)
        for rows in by_item.values()
    )

    return {
        "schema_version": "k0-rvg-divergence-aligned-layer23-state-audit-summary-v1",
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": len(pair_summaries),
        "comparison_row_count": len(comparison_rows),
        "primary_layer": PRIMARY_LAYER,
        "relative_coordinates": list(RELATIVE_COORDINATES),
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "token_contract": {
            "k_minus_1_equal_pair_count": token_contract_k_minus_1,
            "k_zero_different_pair_count": token_contract_k_zero_different,
            "expected_pair_count": EXPECTED_ITEM_COUNT * 2,
        },
        "first_state_hash_difference_histogram": {
            "corr": dict(sorted(first_hist["corr"].items())),
            "ctrl": dict(sorted(first_hist["ctrl"].items())),
        },
        "hash_field_difference_coordinate_histogram": {
            field: dict(sorted(counter.items()))
            for field, counter in field_coord.items()
        },
        "pair_role_state_identity_through_k_plus_6_count": identity_pairs,
        "item_both_roles_state_identity_through_k_plus_6_count": both_identity_items,
        "item_163": [
            dict(row)
            for row in pair_summaries
            if row["local_template_index"] == 163
        ],
    }


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def json_bytes(obj: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(obj),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def scientific_execute(
    root: Path,
    repo_info: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    handoff_path: Path,
    output_dir: Path,
) -> None:
    require(handoff_path.is_file(), f"HANDOFF_MISSING:{handoff_path}")

    final_dir = output_dir.resolve()
    partial_dir = Path(str(final_dir) + ".partial")

    require(not final_dir.exists(), f"OUTPUT_DIR_EXISTS:{final_dir}")
    require(not partial_dir.exists(), f"PARTIAL_OUTPUT_DIR_EXISTS:{partial_dir}")

    (
        observer,
        k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        snapshot,
    ) = resolve_runtime(root, handoff_path)

    comparison_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    forward_count = 0

    for n, row in enumerate(plan, start=1):
        matched_hashes = capture_branch_hashes(
            observer,
            model,
            layer_map,
            binding,
            row["matched_ids"],
            row["targets"],
        )
        forward_count += 1

        swapped_hashes = capture_branch_hashes(
            observer,
            model,
            layer_map,
            binding,
            row["swapped_ids"],
            row["targets"],
        )
        forward_count += 1

        rows, pair_summary = comparison_rows_for_pair(
            row,
            matched_hashes,
            swapped_hashes,
        )

        comparison_rows.extend(rows)
        pair_summaries.append(pair_summary)

        if n % 16 == 0 or n == len(plan):
            print(
                f"PROGRESS pair_roles={n}/{len(plan)} "
                f"model_forwards={forward_count}",
                flush=True,
            )

    require(forward_count == EXPECTED_ITEM_COUNT * 4, "FORWARD_COUNT_MISMATCH")

    summary = make_scientific_summary(
        comparison_rows,
        pair_summaries,
    )

    partial_dir.mkdir(parents=True, exist_ok=False)

    comparisons_path = partial_dir / "state_hash_comparisons.jsonl"
    pairs_path = partial_dir / "pair_summaries.jsonl"
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "execution_manifest.json"

    comparisons_path.write_bytes(jsonl_bytes(comparison_rows))
    pairs_path.write_bytes(jsonl_bytes(pair_summaries))
    summary_path.write_bytes(json_bytes(summary))

    manifest = {
        "schema_version": "k0-rvg-divergence-aligned-layer23-execution-manifest-v1",
        "runtime_git_head": repo_info["head"],
        "runtime_branch": repo_info["branch"],
        "frozen_token_evidence_commit": FROZEN_COMMIT,
        "active_token_artifact_rel": ACTIVE_REL,
        "active_token_artifact_blob": repo_info["active_artifact_blob"],
        "observer_rel": OBSERVER_REL,
        "observer_blob": repo_info["observer_blob"],
        "k2s_rel": K2S_REL,
        "k2s_blob": repo_info["k2s_blob"],
        "runner_rel": Path(__file__).resolve().relative_to(root).as_posix(),
        "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "handoff_zip_path": str(handoff_path.resolve()),
        "handoff_zip_sha256": handoff["zip_sha256"],
        "checkpoint_sha256": handoff["checkpoint_sha256"],
        "encoder_canonical_digest": encoder["canonical_digest"],
        "encoder_raw_concat_digest": encoder["raw_concat_digest"],
        "hf_model": k2s.HF_MODEL,
        "hf_revision": k2s.HF_REVISION,
        "hf_snapshot_path": str(snapshot.resolve()),
        "hf_local_files_only": True,
        "primary_layer": PRIMARY_LAYER,
        "relative_coordinates": list(RELATIVE_COORDINATES),
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_ITEM_COUNT * 2,
        "model_forward_count": forward_count,
        "scientific_model_forward_executed": True,
        "scientific_recurrent_state_read": True,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "outputs": {
            "state_hash_comparisons.jsonl": sha256_bytes(comparisons_path.read_bytes()),
            "pair_summaries.jsonl": sha256_bytes(pairs_path.read_bytes()),
            "summary.json": sha256_bytes(summary_path.read_bytes()),
        },
    }

    manifest_path.write_bytes(json_bytes(manifest))

    os.replace(partial_dir, final_dir)

    print("PASS_DIVERGENCE_ALIGNED_LAYER23_EXECUTION")
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print(
        "pair_role_state_identity_through_k_plus_6_count =",
        summary["pair_role_state_identity_through_k_plus_6_count"],
    )
    print(
        "item_both_roles_state_identity_through_k_plus_6_count =",
        summary["item_both_roles_state_identity_through_k_plus_6_count"],
    )
    print(
        "first_state_hash_difference_histogram =",
        summary["first_state_hash_difference_histogram"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--static-preflight",
        action="store_true",
        help="Authenticate frozen inputs and construct the 336x2 divergence-aligned plan only.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the full 336-item divergence-aligned layer-23 state audit.",
    )
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args()

    require(
        args.static_preflight ^ args.execute,
        "SELECT_EXACTLY_ONE_MODE",
    )

    root = repo_root()
    repo_info = authenticate_repo(root)
    active = load_active_artifact(root)
    plan = build_static_plan(active)
    static = static_summary(plan)

    print("=== DIVERGENCE-ALIGNED STATIC PLAN ===")
    print("branch =", repo_info["branch"])
    print("head =", repo_info["head"])
    print("frozen_commit =", FROZEN_COMMIT)
    print("item_count =", static["item_count"])
    print("pair_role_count =", static["pair_role_count"])
    print("primary_layer =", PRIMARY_LAYER)
    print("relative_coordinates =", list(RELATIVE_COORDINATES))
    print("minimum_common_post_horizon =", static["minimum_common_post_horizon"])
    print("tokenizer_invoked = False")

    if args.static_preflight:
        print("PASS_DIVERGENCE_ALIGNED_STATIC_PREFLIGHT")
        return 0

    require(args.handoff is not None, "EXECUTE_REQUIRES_HANDOFF")
    require(args.output_dir is not None, "EXECUTE_REQUIRES_OUTPUT_DIR")

    scientific_execute(
        root,
        repo_info,
        plan,
        args.handoff,
        args.output_dir,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)