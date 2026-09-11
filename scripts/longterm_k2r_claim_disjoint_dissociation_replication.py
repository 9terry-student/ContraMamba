"""K2R claim-disjoint local-vs-net native-state replication.

Fail-closed confirmatory successor to K2S.  K2R owns only the prospective
claim-disjoint population, reciprocal matched/swapped construction, DISP
endpoint, and four-endpoint replication statistics.  Native Mamba recurrent-
state instrumentation is imported from the exact frozen K2S implementation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Frozen scientific / provenance constants
# ---------------------------------------------------------------------------
K2R_PREREG_AUTHORITY_COMMIT = "54b1a9a2188e3e678e8378b43faa5d527aca1457"
K2R_PREREG_REL = "reports/longterm_k2r_claim_disjoint_local_vs_net_replication_prereg_candidate.md"
K2R_PREREG_SHA256 = "8cc8f10be0dac663267412e8221b87a851cb08ee9bf78fb6c8034ed0b1824dbc"
K2R_GENERATOR_AUTHORITY_COMMIT = "624ea02130eddce416a92239a97d7728bd8aa5b1"
K2S_IMPLEMENTATION_COMMIT = "d6d901cb3f1ab636d4db8b6cbea0bdf1e0581631"
K2S_RESULT_ARCHIVE_COMMIT = "6069213234793286e658948a2e6f3b4f1105543d"
K2S_SECONDARY_INTERPRETATION_COMMIT = "624ea02130eddce416a92239a97d7728bd8aa5b1"
K2W_CLOSURE_COMMIT = "b94f81b411bcbc74e32ad0ad6564b8978016eec0"
A0_COMMIT = "55debe94f0d19d16a334395e8561901fed6b52fa"

GENERATOR_REL = "scripts/build_controlled_v5.py"
GENERATOR_SHA256 = "4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c"
GENERATOR_GIT_BLOB = "baee23a9f71333125f4a8735c2c92d20cab7eb4f"
K2S_DEPENDENCY_REL = "scripts/longterm_k2s_pair_specific_event_dynamics.py"
K2S_DEPENDENCY_SHA256 = "f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8"
K2S_DEPENDENCY_GIT_BLOB = "3a651fb508669bdcf72441a4869b863d6eee6c1f"

OLD_POPULATION_REL = (
    "reports/longterm_k2w_fixed_window_phase_a_c7c7a0c218bb/"
    "candidate_pool.jsonl"
)
OLD_POPULATION_SHA256 = "abf693d3267cc4e3dd27a8127d2948b36fdf8ba24e135a643215f0f31a26d808"

GLOBAL_TEMPLATE_START = 300
GLOBAL_TEMPLATE_STOP = 600
GENERATED_SOURCE_ROWS = 3900
GENERATED_SOURCE_CANONICAL_SHA256 = "e2bf70d8d25a7dca4c590d8ccb72db74308edbcd4ef8158f0482a8730d3978a2"
CANDIDATE_POOL_CANONICAL_SHA256 = "00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4"

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"
HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}
K2R_UNTRACKED = {
    "scripts/longterm_k2r_claim_disjoint_dissociation_replication.py",
    "tests/test_longterm_k2r_claim_disjoint_dissociation_replication.py",
}

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"

W = 8
EPSILON = 1e-12
PRIMARY_LAYER = 23
N_LAYERS = 24
N_ITEMS = 300
N_BLOCKS = 150
PRIMARY_ORDER = ("R", "D", "DISP", "P")
EXPECTED_DIRECTION = {"R": 1, "D": 1, "DISP": -1, "P": -1}

MANIFEST_SCHEMA = "k2r-claim-disjoint-replication-manifest-v1"
ITEM_SCHEMA = "k2r-item-metrics-v1"
LAYER_SCHEMA = "k2r-layer-metrics-v1"
BLOCK_SCHEMA = "k2r-block-metrics-v1"
PRIMARY_SCHEMA = "k2r-primary-stats-v1"
CANDIDATE_SCHEMA = "k2r-independent-population-v1"

REPLICATED_VERDICT = "LOCAL_VS_NET_TRAJECTORY_DISSOCIATION_REPLICATED"
NOT_FULLY_REPLICATED_VERDICT = "LOCAL_VS_NET_DISSOCIATION_NOT_FULLY_REPLICATED"
CONTRADICTION_VERDICT = "LOCAL_VS_NET_DISSOCIATION_DIRECTIONAL_CONTRADICTION"
POPULATION_PROVENANCE_FAILURE = "POPULATION_PROVENANCE_FAILURE"
REPLICATION_POPULATION_OVERLAP_FAILURE = "REPLICATION_POPULATION_OVERLAP_FAILURE"

A0_MODEL_REL = "src/contramamba/modeling_v6b_minimal.py"
A0_HEADS_REL = "src/contramamba/heads"


class ContractError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_line(value: Any) -> bytes:
    return canonical_json(value) + b"\n"


def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_line(dict(row)) for row in rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"git failure: {' '.join(args)}") from exc


def _git_bytes(root: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(["git", *args], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"git failure: {' '.join(args)}") from exc


def parse_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    require(not raw.startswith(b"\xef\xbb\xbf"), "JSONL_BOM_FORBIDDEN")
    require(b"\r" not in raw, "JSONL_CR_FORBIDDEN")
    require(raw.endswith(b"\n"), "JSONL_FINAL_LF_REQUIRED")
    lines = raw[:-1].split(b"\n")
    require(bool(lines) and all(lines), "JSONL_BLANK_LINE_FORBIDDEN")
    out: list[dict[str, Any]] = []
    try:
        for line in lines:
            value = json.loads(line.decode("utf-8", "strict"))
            require(isinstance(value, dict), "JSONL_ROW_NOT_OBJECT")
            out.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("JSONL_DECODE_FAILURE") from exc
    return out


# ---------------------------------------------------------------------------
# Exact dependency / Git authority binding
# ---------------------------------------------------------------------------
def git_provenance(root: Path, instrumentation_preflight: bool = False) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")
    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", K2R_PREREG_AUTHORITY_COMMIT, head],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "K2R_PREREG_NOT_ANCESTOR")

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True
    ).splitlines()
    allowed = set(HISTORICAL_K1_UNTRACKED)
    if instrumentation_preflight:
        allowed |= K2R_UNTRACKED
    for line in status:
        path = line[3:].replace("\\", "/") if len(line) >= 4 else ""
        require(line[:2] == "??" and path in allowed, "GIT_DIRTY_CONTRACT_MISMATCH")

    prereg = root / K2R_PREREG_REL
    generator = root / GENERATOR_REL
    k2s_path = root / K2S_DEPENDENCY_REL
    require(prereg.is_file() and file_sha256(prereg) == K2R_PREREG_SHA256, "K2R_PREREG_BYTE_MISMATCH")
    require(generator.is_file() and file_sha256(generator) == GENERATOR_SHA256, "K2R_GENERATOR_BYTE_MISMATCH")
    require(k2s_path.is_file() and file_sha256(k2s_path) == K2S_DEPENDENCY_SHA256, "K2S_DEPENDENCY_BYTE_MISMATCH")

    generator_authority_blob = _git(root, "rev-parse", f"{K2R_GENERATOR_AUTHORITY_COMMIT}:{GENERATOR_REL}")
    generator_runtime_blob = _git(root, "rev-parse", f"{head}:{GENERATOR_REL}")
    k2s_runtime_blob = _git(root, "rev-parse", f"{head}:{K2S_DEPENDENCY_REL}")
    require(generator_authority_blob == GENERATOR_GIT_BLOB, "K2R_GENERATOR_AUTHORITY_BLOB_MISMATCH")
    require(generator_runtime_blob == GENERATOR_GIT_BLOB, "K2R_GENERATOR_RUNTIME_BLOB_DRIFT")
    require(k2s_runtime_blob == K2S_DEPENDENCY_GIT_BLOB, "K2S_DEPENDENCY_RUNTIME_BLOB_DRIFT")

    current_model_blob = _git(root, "rev-parse", f"{head}:{A0_MODEL_REL}")
    a0_model_blob = _git(root, "rev-parse", f"{A0_COMMIT}:{A0_MODEL_REL}")
    current_heads_tree = _git(root, "rev-parse", f"{head}:{A0_HEADS_REL}")
    a0_heads_tree = _git(root, "rev-parse", f"{A0_COMMIT}:{A0_HEADS_REL}")
    require(current_model_blob == a0_model_blob, "A0_MODEL_SOURCE_DRIFT")
    require(current_heads_tree == a0_heads_tree, "A0_HEADS_SOURCE_DRIFT")

    return {
        "runtime_branch": branch,
        "runtime_git_head": head,
        "runtime_dirty_contract": status,
        "k2r_prereg_sha256": K2R_PREREG_SHA256,
        "generator_sha256": GENERATOR_SHA256,
        "generator_authority_commit": K2R_GENERATOR_AUTHORITY_COMMIT,
        "generator_git_blob": generator_runtime_blob,
        "k2s_dependency_sha256": K2S_DEPENDENCY_SHA256,
        "k2s_dependency_git_blob": k2s_runtime_blob,
        "a0_model_blob_sha": current_model_blob,
        "a0_heads_tree_sha": current_heads_tree,
    }


def load_frozen_dependencies(root: Path) -> tuple[Any, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    generator = importlib.import_module("scripts.build_controlled_v5")
    k2s = importlib.import_module("scripts.longterm_k2s_pair_specific_event_dynamics")
    require(Path(generator.__file__).resolve() == (root / GENERATOR_REL).resolve(), "GENERATOR_IMPORT_PATH_MISMATCH")
    require(Path(k2s.__file__).resolve() == (root / K2S_DEPENDENCY_REL).resolve(), "K2S_IMPORT_PATH_MISMATCH")
    require(file_sha256(Path(generator.__file__)) == GENERATOR_SHA256, "GENERATOR_IMPORT_BYTE_MISMATCH")
    require(file_sha256(Path(k2s.__file__)) == K2S_DEPENDENCY_SHA256, "K2S_IMPORT_BYTE_MISMATCH")
    require(k2s.W == W and k2s.PRIMARY_LAYER == PRIMARY_LAYER and k2s.N_LAYERS == N_LAYERS, "K2S_MEASUREMENT_GEOMETRY_DRIFT")
    require(k2s.HF_MODEL == HF_MODEL and k2s.HF_REVISION == HF_REVISION, "K2S_HF_IDENTITY_DRIFT")
    return generator, k2s


# ---------------------------------------------------------------------------
# Claim-disjoint population materialization
# ---------------------------------------------------------------------------
def _claim_text_from_prefix(prefix: str) -> str:
    marker = "\nEvidence:"
    require(marker in prefix, "PREFIX_CLAIM_MARKER_MISSING")
    return prefix.split(marker, 1)[0]


def materialize_population(root: Path, generator: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    templates_all = generator.fact_templates_for_count(GLOBAL_TEMPLATE_STOP)
    templates = [dict(row) for row in templates_all[GLOBAL_TEMPLATE_START:GLOBAL_TEMPLATE_STOP]]
    require(len(templates) == N_ITEMS, POPULATION_PROVENANCE_FAILURE)
    expected_pair_ids = [f"generated_fact_{i:03d}" for i in range(301, 601)]
    pair_ids = [str(row.get("pair_id", "")) for row in templates]
    require(pair_ids == expected_pair_ids, POPULATION_PROVENANCE_FAILURE)
    require(len(set(pair_ids)) == N_ITEMS, POPULATION_PROVENANCE_FAILURE)

    source_rows = generator._build_records(templates)
    require(len(source_rows) == GENERATED_SOURCE_ROWS, POPULATION_PROVENANCE_FAILURE)
    source_bytes = canonical_jsonl(source_rows)
    require(sha256_bytes(source_bytes) == GENERATED_SOURCE_CANONICAL_SHA256, POPULATION_PROVENANCE_FAILURE)

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in source_rows:
        groups.setdefault(str(row.get("pair_id", "")), []).append(row)
    require(set(groups) == set(pair_ids), POPULATION_PROVENANCE_FAILURE)

    candidate_rows: list[dict[str, Any]] = []
    correction_sources: Counter[str] = Counter()
    for pair_id in sorted(groups):
        rows = groups[pair_id]
        trunc = [r for r in rows if r.get("intervention_type") == "evidence_truncation"]
        entity = [r for r in rows if r.get("intervention_type") == "entity_swap"]
        refute_polarity = [
            r for r in rows
            if r.get("intervention_type") == "polarity_flip" and r.get("final_label") == "REFUTE"
        ]
        refute_none = [
            r for r in rows
            if r.get("intervention_type") == "none" and r.get("final_label") == "REFUTE"
        ]
        require(len(trunc) == 1 and len(entity) == 1, POPULATION_PROVENANCE_FAILURE)
        if len(refute_polarity) == 1:
            correction = refute_polarity[0]
            correction_source = "polarity_flip"
        elif len(refute_none) == 1:
            correction = refute_none[0]
            correction_source = "none"
        else:
            raise ContractError(POPULATION_PROVENANCE_FAILURE)

        t, e, q = trunc[0], entity[0], correction
        require(t.get("claim") == e.get("claim") == q.get("claim"), POPULATION_PROVENANCE_FAILURE)
        require(
            t.get("final_label") == "NOT_ENTITLED"
            and t.get("primary_failure_type") == "sufficiency"
            and t.get("sufficiency_label") == 0,
            POPULATION_PROVENANCE_FAILURE,
        )
        require(
            e.get("final_label") == "NOT_ENTITLED"
            and e.get("primary_failure_type") == "frame"
            and e.get("polarity_label") == "NONE",
            POPULATION_PROVENANCE_FAILURE,
        )
        require(
            q.get("final_label") == "REFUTE" and q.get("polarity_label") == "REFUTE",
            POPULATION_PROVENANCE_FAILURE,
        )

        prefix = (
            "Claim: " + str(t["claim"])
            + "\nEvidence: " + str(t["evidence"])
            + "\nAdditional evidence:\n"
        )
        recipe = {
            "schema_version": CANDIDATE_SCHEMA,
            "generator_sha256": GENERATOR_SHA256,
            "global_template_start": GLOBAL_TEMPLATE_START,
            "global_template_stop": GLOBAL_TEMPLATE_STOP,
            "pair_id": pair_id,
            "truncation_source_id": str(t["id"]),
            "correction_source_id": str(q["id"]),
            "correction_source_intervention": correction_source,
            "control_source_id": str(e["id"]),
            "prefix_text": prefix,
            "correction_text": str(q["evidence"]),
            "control_text": str(e["evidence"]),
        }
        candidate_rows.append(
            {
                **recipe,
                "stable_item_id": "k2r-v1:" + sha256_bytes(canonical_json(recipe)),
                "base_claim_sha256": sha256_bytes(str(t["claim"]).encode("utf-8")),
            }
        )
        correction_sources[correction_source] += 1

    candidate_rows.sort(key=lambda row: row["stable_item_id"])
    require(len(candidate_rows) == N_ITEMS, POPULATION_PROVENANCE_FAILURE)
    require(len({r["stable_item_id"] for r in candidate_rows}) == N_ITEMS, POPULATION_PROVENANCE_FAILURE)
    require(len({r["base_claim_sha256"] for r in candidate_rows}) == N_ITEMS, POPULATION_PROVENANCE_FAILURE)
    require(
        correction_sources == Counter({"none": 150, "polarity_flip": 150}),
        POPULATION_PROVENANCE_FAILURE,
    )
    candidate_bytes = canonical_jsonl(candidate_rows)
    require(sha256_bytes(candidate_bytes) == CANDIDATE_POOL_CANONICAL_SHA256, POPULATION_PROVENANCE_FAILURE)

    old_raw = _git_bytes(root, "show", f"{K2W_CLOSURE_COMMIT}:{OLD_POPULATION_REL}")
    require(sha256_bytes(old_raw) == OLD_POPULATION_SHA256, "OLD_POPULATION_SHA256_MISMATCH")
    old_rows = parse_jsonl_bytes(old_raw)
    old_pair_ids = {str(r["pair_id"]) for r in old_rows}
    old_claim_sha = {str(r["base_claim_sha256"]) for r in old_rows}
    old_claim_text = {_claim_text_from_prefix(str(r["prefix_text"])) for r in old_rows}
    new_pair_ids = {str(r["pair_id"]) for r in candidate_rows}
    new_claim_sha = {str(r["base_claim_sha256"]) for r in candidate_rows}
    new_claim_text = {_claim_text_from_prefix(str(r["prefix_text"])) for r in candidate_rows}
    overlap = {
        "old_new_pair_id_overlap": len(old_pair_ids & new_pair_ids),
        "old_new_claim_sha_overlap": len(old_claim_sha & new_claim_sha),
        "old_new_claim_text_overlap": len(old_claim_text & new_claim_text),
    }
    require(all(value == 0 for value in overlap.values()), REPLICATION_POPULATION_OVERLAP_FAILURE)

    audit = {
        "generator_authority_commit": K2R_GENERATOR_AUTHORITY_COMMIT,
        "generator_path": GENERATOR_REL,
        "generator_sha256": GENERATOR_SHA256,
        "generator_git_blob": GENERATOR_GIT_BLOB,
        "global_template_start": GLOBAL_TEMPLATE_START,
        "global_template_stop": GLOBAL_TEMPLATE_STOP,
        "first_pair_id": expected_pair_ids[0],
        "last_pair_id": expected_pair_ids[-1],
        "generated_source_rows": len(source_rows),
        "generated_source_canonical_sha256": sha256_bytes(source_bytes),
        "candidate_pool_count": len(candidate_rows),
        "candidate_pool_canonical_sha256": sha256_bytes(candidate_bytes),
        "correction_source_counts": dict(sorted(correction_sources.items())),
        **overlap,
    }
    return [dict(r) for r in source_rows], candidate_rows, audit


def donor_index(index: int) -> int:
    require(type(index) is int and 0 <= index < N_ITEMS, "DONOR_INDEX_INVALID")
    return index ^ 1


def reciprocal_blocks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(rows) == N_ITEMS, "RECIPROCAL_POPULATION_COUNT")
    blocks: list[dict[str, Any]] = []
    for block_index in range(N_BLOCKS):
        a, b = 2 * block_index, 2 * block_index + 1
        ra, rb = rows[a], rows[b]
        require(donor_index(a) == b and donor_index(b) == a, "RECIPROCAL_MAPPING_BROKEN")
        require(ra["stable_item_id"] != rb["stable_item_id"], "RECIPROCAL_STABLE_ID_COLLISION")
        require(ra["base_claim_sha256"] != rb["base_claim_sha256"], "RECIPROCAL_BASE_CLAIM_COLLISION")
        blocks.append(
            {
                "block_index": block_index,
                "block_id": f"k2r-block-{block_index:03d}",
                "item_a_index": a,
                "item_b_index": b,
                "item_a_stable_id": ra["stable_item_id"],
                "item_b_stable_id": rb["stable_item_id"],
            }
        )
    return blocks


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"] if isinstance(out, Mapping) else out.input_ids
    return list(ids)


def first_divergence_after_prefix(corr: Sequence[int], ctrl: Sequence[int], prefix_len: int) -> int | None:
    stop = min(len(corr), len(ctrl), prefix_len + W)
    return next((i for i in range(prefix_len, stop) if corr[i] != ctrl[i]), None)


def build_input_contracts(rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(getattr(tokenizer, "is_fast", False), "TOKENIZER_MUST_BE_FAST")
    require(len(rows) == N_ITEMS, "INPUT_CONTRACT_COUNT")
    contracts: list[dict[str, Any]] = []
    matched_d: Counter[str] = Counter()
    swapped_d: Counter[str] = Counter()
    availability: dict[str, list[int]] = {
        "matched_corr": [], "matched_ctrl": [], "swapped_corr": [], "swapped_ctrl": []
    }
    for i, row in enumerate(rows):
        donor = rows[donor_index(i)]
        prefix = str(row["prefix_text"])
        p_ids = _token_ids(tokenizer, prefix)
        require(len(p_ids) >= 2, "PREFIX_TOO_SHORT_FOR_PRESTATE")
        p = len(p_ids) - 1
        texts = {
            "matched_corr": prefix + str(row["correction_text"]),
            "matched_ctrl": prefix + str(row["control_text"]),
            "swapped_corr": prefix + str(donor["correction_text"]),
            "swapped_ctrl": prefix + str(donor["control_text"]),
        }
        arrays: dict[str, list[int]] = {}
        for name, text in texts.items():
            ids = _token_ids(tokenizer, text)
            require(ids[: len(p_ids)] == p_ids, "INVALID_EXACT_PREFIX")
            require(len(ids) - len(p_ids) >= W, "INVALID_WINDOW8")
            arrays[name] = ids
            availability[name].append(len(ids) - len(p_ids))
        md = first_divergence_after_prefix(arrays["matched_corr"], arrays["matched_ctrl"], len(p_ids))
        sd = first_divergence_after_prefix(arrays["swapped_corr"], arrays["swapped_ctrl"], len(p_ids))
        require(md is not None, "MATCHED_NO_DIVERGENCE_WITHIN_W8")
        require(sd is not None, "SWAPPED_NO_DIVERGENCE_WITHIN_W8")
        matched_d[str(md - p)] += 1
        swapped_d[str(sd - p)] += 1
        contracts.append(
            {
                "item_index": i,
                "block_index": i // 2,
                "block_id": f"k2r-block-{i // 2:03d}",
                "stable_item_id": row["stable_item_id"],
                "pair_id": row["pair_id"],
                "base_claim_sha256": row["base_claim_sha256"],
                "correction_source_intervention": row["correction_source_intervention"],
                "donor_index": donor_index(i),
                "donor_stable_item_id": donor["stable_item_id"],
                "donor_base_claim_sha256": donor["base_claim_sha256"],
                "prefix_text": prefix,
                "correction_text": row["correction_text"],
                "control_text": row["control_text"],
                "donor_correction_text": donor["correction_text"],
                "donor_control_text": donor["control_text"],
                "p": p,
                "prefix_token_ids": p_ids,
                "branch_texts": texts,
                "branch_token_ids": arrays,
                "matched_d_minus_p": md - p,
                "swapped_d_minus_p": sd - p,
            }
        )

    require(
        sorted(str(r["correction_text"]) for r in rows)
        == sorted(str(rows[donor_index(i)]["correction_text"]) for i in range(N_ITEMS)),
        "CORRECTION_MARGINAL_NOT_PRESERVED",
    )
    require(
        sorted(str(r["control_text"]) for r in rows)
        == sorted(str(rows[donor_index(i)]["control_text"]) for i in range(N_ITEMS)),
        "CONTROL_MARGINAL_NOT_PRESERVED",
    )
    summary = {
        "N_total": N_ITEMS,
        "N_blocks": N_BLOCKS,
        "N_matched_valid": N_ITEMS,
        "N_swapped_valid": N_ITEMS,
        "matched_d_minus_p": dict(sorted(matched_d.items(), key=lambda kv: int(kv[0]))),
        "swapped_d_minus_p": dict(sorted(swapped_d.items(), key=lambda kv: int(kv[0]))),
        "matched_corr_available_minmax": [min(availability["matched_corr"]), max(availability["matched_corr"])],
        "matched_ctrl_available_minmax": [min(availability["matched_ctrl"]), max(availability["matched_ctrl"])],
        "swapped_corr_available_minmax": [min(availability["swapped_corr"]), max(availability["swapped_corr"])],
        "swapped_ctrl_available_minmax": [min(availability["swapped_ctrl"]), max(availability["swapped_ctrl"])],
        "prefix_marginal_preserved": True,
        "correction_marginal_preserved": True,
        "control_marginal_preserved": True,
    }
    expected = {
        "N_total": 300,
        "N_blocks": 150,
        "N_matched_valid": 300,
        "N_swapped_valid": 300,
        "matched_d_minus_p": {"2": 150, "3": 150},
        "swapped_d_minus_p": {"2": 150, "3": 150},
        "matched_corr_available_minmax": [23, 28],
        "matched_ctrl_available_minmax": [20, 27],
        "swapped_corr_available_minmax": [23, 28],
        "swapped_ctrl_available_minmax": [20, 27],
        "prefix_marginal_preserved": True,
        "correction_marginal_preserved": True,
        "control_marginal_preserved": True,
    }
    require(summary == expected, "K2R_FROZEN_TOKEN_FEASIBILITY_MISMATCH")
    return contracts, summary


# ---------------------------------------------------------------------------
# Four-endpoint kinematics and block statistics
# ---------------------------------------------------------------------------
METRIC_FIELD = {
    "R": "R_mean_speed",
    "D": "D_mean_turn",
    "DISP": "displacement",
    "P": "P_efficiency",
}


def recipient_layer_result(branches: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = {"matched_corr", "matched_ctrl", "swapped_corr", "swapped_ctrl"}
    require(set(branches) == required, "RECIPIENT_BRANCH_SET_INVALID")
    prefix_p = {branches[name]["state_p_sha256"] for name in required}
    prefix_pm1 = {branches[name]["state_p_minus_1_sha256"] for name in required}
    prefix_sequences = {branches[name]["prefix_state_sequence_sha256"] for name in required}
    prefix_counts = {branches[name]["prefix_state_token_count"] for name in required}
    require(
        len(prefix_p) == 1
        and len(prefix_pm1) == 1
        and len(prefix_sequences) == 1
        and len(prefix_counts) == 1,
        "SCIENTIFIC_PREFIX_STATE_IDENTITY_FAILURE",
    )

    def delta(field: str, a: str, b: str) -> float | None:
        x, y = branches[a][field], branches[b][field]
        if x is None or y is None:
            return None
        value = float(x - y)
        require(math.isfinite(value), "RECIPIENT_DELTA_NONFINITE")
        return value

    deltas: dict[str, float | None] = {}
    x_values: dict[str, float | None] = {}
    for metric in PRIMARY_ORDER:
        field = METRIC_FIELD[metric]
        matched = delta(field, "matched_corr", "matched_ctrl")
        swapped = delta(field, "swapped_corr", "swapped_ctrl")
        deltas[f"{metric}_matched"] = matched
        deltas[f"{metric}_swapped"] = swapped
        if matched is None or swapped is None:
            x_values[metric] = None
        else:
            x = abs(matched) - abs(swapped)
            require(math.isfinite(x), "RECIPIENT_X_NONFINITE")
            x_values[metric] = float(x)
    for metric in ("R", "DISP", "P"):
        require(x_values[metric] is not None, f"{metric}_PAIR_SPECIFICITY_UNDEFINED")

    return {
        "branches": {name: dict(branches[name]) for name in sorted(branches)},
        "signed_contrasts": deltas,
        "X_pair_specificity": x_values,
        "common_state_p_sha256": next(iter(prefix_p)),
        "common_state_p_minus_1_sha256": next(iter(prefix_pm1)),
        "common_prefix_state_sequence_sha256": next(iter(prefix_sequences)),
        "common_prefix_state_token_count": next(iter(prefix_counts)),
    }


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    require(positive >= 0 and negative >= 0, "SIGN_COUNTS_INVALID")
    n = positive + negative
    if n == 0:
        return 1.0
    tail = min(positive, negative)
    numerator = sum(math.comb(n, k) for k in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2**n))


def holm_adjust(raw: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    require(set(raw) == set(PRIMARY_ORDER), "HOLM_PRIMARY_SET_MISMATCH")
    tie = {name: i for i, name in enumerate(PRIMARY_ORDER)}
    ordered = sorted(PRIMARY_ORDER, key=lambda name: (float(raw[name]), tie[name]))
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (m - rank) * float(raw[name]))
        running = max(running, candidate)
        adjusted[name] = min(1.0, running)
    return {
        name: {
            "raw_p": float(raw[name]),
            "holm_adjusted_p": float(adjusted[name]),
            "holm_reject": bool(adjusted[name] <= 0.05),
        }
        for name in PRIMARY_ORDER
    }


def summarize_endpoint(values: Sequence[float | None]) -> dict[str, Any]:
    undefined = sum(value is None for value in values)
    valid = [float(value) for value in values if value is not None]
    require(all(math.isfinite(value) for value in valid), "BLOCK_VALUE_NONFINITE")
    positive = sum(value > 0.0 for value in valid)
    negative = sum(value < 0.0 for value in valid)
    zero = sum(value == 0.0 for value in valid)
    n_valid = len(valid)
    n_eff = positive + negative
    floor_pass = n_valid >= 120 and n_eff >= 30
    raw_p = exact_two_sided_sign_p(positive, negative) if floor_pass else 1.0
    effect = None if n_eff == 0 else (positive - negative) / n_eff
    return {
        "n_valid": n_valid,
        "n_eff": n_eff,
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "undefined_count": undefined,
        "promotion_floor_pass": floor_pass,
        "raw_p": raw_p,
        "rank_biserial_sign_effect": effect,
    }


def primary_statistics(block_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(block_rows) == N_BLOCKS, "BLOCK_COUNT_MISMATCH")
    endpoints = {
        metric: summarize_endpoint([row[f"B_{metric}"] for row in block_rows])
        for metric in PRIMARY_ORDER
    }
    holm = holm_adjust({metric: endpoints[metric]["raw_p"] for metric in PRIMARY_ORDER})
    direction_matches: list[str] = []
    contradictions: list[str] = []
    for metric in PRIMARY_ORDER:
        endpoints[metric].update(holm[metric])
        effect = endpoints[metric]["rank_biserial_sign_effect"]
        expected = EXPECTED_DIRECTION[metric]
        endpoints[metric]["expected_direction"] = "positive" if expected > 0 else "negative"
        match = bool(
            endpoints[metric]["promotion_floor_pass"]
            and endpoints[metric]["holm_reject"]
            and effect is not None
            and effect * expected > 0
        )
        contradiction = bool(
            endpoints[metric]["promotion_floor_pass"]
            and endpoints[metric]["holm_reject"]
            and effect is not None
            and effect * expected < 0
        )
        endpoints[metric]["direction_match"] = match
        endpoints[metric]["directional_contradiction"] = contradiction
        if match:
            direction_matches.append(metric)
        if contradiction:
            contradictions.append(metric)

    full = direction_matches == list(PRIMARY_ORDER)
    if full:
        verdict = REPLICATED_VERDICT
    elif contradictions:
        verdict = CONTRADICTION_VERDICT
    else:
        verdict = NOT_FULLY_REPLICATED_VERDICT
    return {
        "schema_version": PRIMARY_SCHEMA,
        "primary_order": list(PRIMARY_ORDER),
        "expected_direction": dict(EXPECTED_DIRECTION),
        "holm_m": 4,
        "holm_alpha": 0.05,
        "holm_tie_order": list(PRIMARY_ORDER),
        "endpoints": endpoints,
        "direction_matched_endpoints": direction_matches,
        "directional_contradiction_endpoints": contradictions,
        "full_replication": full,
        "scientific_verdict": verdict,
    }


def block_rows_from_items(item_primary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(item_primary) == N_ITEMS, "ITEM_PRIMARY_COUNT_MISMATCH")
    rows: list[dict[str, Any]] = []
    for block_index in range(N_BLOCKS):
        a, b = item_primary[2 * block_index], item_primary[2 * block_index + 1]
        require(a["block_index"] == block_index and b["block_index"] == block_index, "BLOCK_ITEM_ALIGNMENT_FAILURE")
        row: dict[str, Any] = {
            "schema_version": BLOCK_SCHEMA,
            "block_index": block_index,
            "block_id": f"k2r-block-{block_index:03d}",
            "item_a_stable_id": a["stable_item_id"],
            "item_b_stable_id": b["stable_item_id"],
        }
        for metric in PRIMARY_ORDER:
            xa, xb = a[f"X_{metric}"], b[f"X_{metric}"]
            if xa is None or xb is None:
                row[f"B_{metric}"] = None
            else:
                value = (float(xa) + float(xb)) / 2.0
                require(math.isfinite(value), "BLOCK_METRIC_NONFINITE")
                row[f"B_{metric}"] = value
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Scientific execution and artifacts
# ---------------------------------------------------------------------------
def _output_dir(root: Path, supplied: Path) -> Path:
    target = supplied.resolve()
    repo = root.resolve()
    require(target != repo and repo not in target.parents, "OUTPUT_DIRECTORY_MUST_BE_OUTSIDE_REPO")
    require(not target.exists(), "OUTPUT_DIRECTORY_ALREADY_EXISTS")
    return target


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def _runtime_versions() -> dict[str, str]:
    import torch
    import transformers
    import tokenizers
    import huggingface_hub
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
    }


def execute_scientific(
    *,
    root: Path,
    k2s: Any,
    model: Any,
    source_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    population_audit: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
    binding: Any,
    output_dir: Path,
    runtime: Mapping[str, Any],
    hf: Mapping[str, Any],
    handoff: Mapping[str, Any],
    encoder_fp: Mapping[str, Any],
    preflight: Mapping[str, Any],
    feasibility: Mapping[str, Any],
) -> dict[str, Any]:
    layer_map = k2s.registered_mamba_layers(model)
    item_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    item_primary: list[dict[str, Any]] = []

    for i, contract in enumerate(contracts):
        branch_results: dict[str, dict[int, dict[str, Any]]] = {}
        for branch_name in ("matched_corr", "matched_ctrl", "swapped_corr", "swapped_ctrl"):
            branch_results[branch_name] = k2s.run_native_branch(
                model,
                contract["branch_token_ids"][branch_name],
                int(contract["p"]),
                binding,
                layer_map,
            )
        layer_result_for_item: dict[int, dict[str, Any]] = {}
        for layer in range(N_LAYERS):
            branches = {name: branch_results[name][layer] for name in branch_results}
            result = recipient_layer_result(branches)
            layer_result_for_item[layer] = result
            layer_rows.append(
                {
                    "schema_version": LAYER_SCHEMA,
                    "item_index": i,
                    "block_index": contract["block_index"],
                    "block_id": contract["block_id"],
                    "stable_item_id": contract["stable_item_id"],
                    "donor_stable_item_id": contract["donor_stable_item_id"],
                    "correction_source_intervention": contract["correction_source_intervention"],
                    "layer_index": layer,
                    **result,
                }
            )
        primary = layer_result_for_item[PRIMARY_LAYER]
        item_rows.append(
            {
                "schema_version": ITEM_SCHEMA,
                **dict(contract),
                "primary_layer": PRIMARY_LAYER,
                "primary_layer_result": primary,
            }
        )
        item_primary.append(
            {
                "item_index": i,
                "block_index": contract["block_index"],
                "stable_item_id": contract["stable_item_id"],
                **{f"X_{metric}": primary["X_pair_specificity"][metric] for metric in PRIMARY_ORDER},
            }
        )
        del branch_results, layer_result_for_item
        if (i + 1) % 10 == 0 or i == 0:
            print(f"K2R_PROGRESS items={i + 1}/{N_ITEMS}", flush=True)
        if (i + 1) % 25 == 0:
            gc.collect()

    blocks = block_rows_from_items(item_primary)
    primary = primary_statistics(blocks)
    output_dir.mkdir(parents=True, exist_ok=False)

    report = (
        "# K2R Claim-Disjoint Local-vs-Net Dissociation Replication Result\n\n"
        f"Scientific verdict: `{primary['scientific_verdict']}`\n\n"
        f"Runtime HEAD: `{runtime['runtime_git_head']}`\n\n"
        "Scope: same-generator, claim-disjoint prospective replication.\n\n"
        "Primary inference uses 150 reciprocal blocks at frozen layer 23/W=8, "
        "with R+, D+, DISP-, P- required simultaneously under Holm m=4.\n\n"
        "This result does not by itself authorize K3 causal intervention or external-distribution generalization.\n"
    ).encode("utf-8")

    blobs: dict[str, bytes] = {
        "generated_source.jsonl": canonical_jsonl(source_rows),
        "candidate_pool.jsonl": canonical_jsonl(candidate_rows),
        "item_metrics.jsonl": canonical_jsonl(item_rows),
        "secondary_layer_metrics.jsonl": canonical_jsonl(layer_rows),
        "block_metrics.jsonl": canonical_jsonl(blocks),
        "primary_stats.json": canonical_json_line(primary),
        "report.md": report,
    }
    require(sha256_bytes(blobs["generated_source.jsonl"]) == GENERATED_SOURCE_CANONICAL_SHA256, POPULATION_PROVENANCE_FAILURE)
    require(sha256_bytes(blobs["candidate_pool.jsonl"]) == CANDIDATE_POOL_CANONICAL_SHA256, POPULATION_PROVENANCE_FAILURE)
    for name, data in blobs.items():
        _write(output_dir / name, data)

    artifact_hashes = {name: sha256_bytes(data) for name, data in blobs.items()}
    test_path = root / "tests" / "test_longterm_k2r_claim_disjoint_dissociation_replication.py"
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "k2r_prereg_authority_commit": K2R_PREREG_AUTHORITY_COMMIT,
        "k2r_prereg_sha256": K2R_PREREG_SHA256,
        "k2s_result_archive_commit": K2S_RESULT_ARCHIVE_COMMIT,
        "k2s_secondary_interpretation_commit": K2S_SECONDARY_INTERPRETATION_COMMIT,
        "implementation_commit": runtime["runtime_git_head"],
        "runtime_git_head": runtime["runtime_git_head"],
        "runtime_branch": runtime["runtime_branch"],
        "runtime_dirty_contract": runtime["runtime_dirty_contract"],
        "script_sha256": file_sha256(Path(__file__)),
        "test_sha256": file_sha256(test_path) if test_path.is_file() else None,
        "dependency_binding": {
            "generator_authority_commit": K2R_GENERATOR_AUTHORITY_COMMIT,
            "generator_path": GENERATOR_REL,
            "generator_sha256": GENERATOR_SHA256,
            "generator_git_blob": GENERATOR_GIT_BLOB,
            "k2s_instrumentation_dependency_commit": K2S_IMPLEMENTATION_COMMIT,
            "k2s_instrumentation_dependency_path": K2S_DEPENDENCY_REL,
            "k2s_instrumentation_dependency_sha256": K2S_DEPENDENCY_SHA256,
            "k2s_instrumentation_dependency_git_blob": K2S_DEPENDENCY_GIT_BLOB,
        },
        "population": dict(population_audit),
        "reciprocal_mapping": [
            {
                "item_index": row["item_index"],
                "stable_item_id": row["stable_item_id"],
                "donor_index": row["donor_index"],
                "donor_stable_item_id": row["donor_stable_item_id"],
                "block_id": row["block_id"],
            }
            for row in contracts
        ],
        "scientific_input_binding": {
            "generated_source_artifact": "generated_source.jsonl",
            "generated_source_sha256": artifact_hashes["generated_source.jsonl"],
            "candidate_pool_artifact": "candidate_pool.jsonl",
            "candidate_pool_sha256": artifact_hashes["candidate_pool.jsonl"],
            "item_metrics_artifact": "item_metrics.jsonl",
            "item_metrics_sha256": artifact_hashes["item_metrics.jsonl"],
            "contains_exact_texts_and_token_arrays": True,
        },
        "feasibility": dict(feasibility),
        "event_anchor": "prefix_end_p",
        "window_W": W,
        "epsilon": EPSILON,
        "primary_layer": PRIMARY_LAYER,
        "primary_metrics": list(PRIMARY_ORDER),
        "expected_direction": dict(EXPECTED_DIRECTION),
        "hf": {key: value for key, value in hf.items() if key not in {"config", "tokenizer"}},
        "handoff": dict(handoff),
        "encoder_fingerprint": dict(encoder_fp),
        "instrumentation_preflight": dict(preflight),
        "instrumentation": {
            "source_runner": K2S_DEPENDENCY_REL,
            "source_runner_sha256": K2S_DEPENDENCY_SHA256,
            "capture_source_qualname": binding.qualname,
            "capture_source_path": str(binding.source_path),
            "capture_source_sha256": binding.source_sha256,
            "capture_source_bytes": binding.source_bytes,
            "recurrence_update_line": binding.recurrence_update_line,
            "capture_line": binding.capture_line,
            "capture_state_source": "native_selective_ssm_recurrent_state",
            "capture_state_timing": "post_consumption_s_t",
            "capture_method": "cpython_line_trace_local_ssm_state_clone",
            "raw_states_serialized": False,
            "all_24_layers_captured_and_reduced": True,
        },
        "artifact_sha256": artifact_hashes,
        "artifact_hash_binding_note": "all non-self scientific artifacts are bound here; manifest.json is bound by SHA256SUMS.txt",
        "required_artifacts": [
            "generated_source.jsonl",
            "candidate_pool.jsonl",
            "item_metrics.jsonl",
            "secondary_layer_metrics.jsonl",
            "block_metrics.jsonl",
            "primary_stats.json",
            "manifest.json",
            "report.md",
            "SHA256SUMS.txt",
        ],
        "primary_statistics": primary,
        "exact_command": sys.argv,
        **_runtime_versions(),
    }
    manifest_bytes = canonical_json_line(manifest)
    _write(output_dir / "manifest.json", manifest_bytes)

    sums_targets = [
        "generated_source.jsonl",
        "candidate_pool.jsonl",
        "item_metrics.jsonl",
        "secondary_layer_metrics.jsonl",
        "block_metrics.jsonl",
        "primary_stats.json",
        "manifest.json",
        "report.md",
    ]
    sums = "".join(f"{file_sha256(output_dir / name)}  {name}\n" for name in sums_targets).encode("utf-8")
    _write(output_dir / "SHA256SUMS.txt", sums)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="K2R claim-disjoint local-vs-net native Mamba replication")
    p.add_argument("--seed180-handoff", required=True)
    p.add_argument("--hf-revision", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--instrumentation-preflight", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    runtime = git_provenance(root, instrumentation_preflight=args.instrumentation_preflight)
    generator, k2s = load_frozen_dependencies(root)

    source_rows, candidate_rows, population_audit = materialize_population(root, generator)
    blocks = reciprocal_blocks(candidate_rows)
    require(len(blocks) == N_BLOCKS, "RECIPROCAL_BLOCK_COUNT_MISMATCH")

    snapshot, hf = k2s.resolve_hf_snapshot(args.hf_revision)
    contracts, feasibility = build_input_contracts(candidate_rows, hf["tokenizer"])

    handoff = k2s.audit_handoff(Path(args.seed180_handoff))
    checkpoint = k2s.load_authenticated_checkpoint(handoff)
    encoder_fp = k2s.encoder_fingerprint(checkpoint["model_state_dict"])
    model = k2s.build_a0_model(root, snapshot, checkpoint)
    del checkpoint
    gc.collect()

    binding = k2s.resolve_capture_binding()
    preflight = k2s.run_instrumentation_preflight(model, hf["tokenizer"], binding)
    handoff_full = {
        **handoff,
        "encoder": encoder_fp,
        "strict_load": "PASS",
        "metadata_training_args": "PASS",
    }

    if args.instrumentation_preflight:
        print(
            canonical_json(
                {
                    "instrumentation_preflight": "PASS",
                    "population": population_audit,
                    "feasibility": feasibility,
                    "handoff": handoff_full,
                    "instrumentation": preflight,
                    "runtime": runtime,
                    "dependency_binding": {
                        "generator_sha256": GENERATOR_SHA256,
                        "generator_git_blob": GENERATOR_GIT_BLOB,
                        "k2s_dependency_sha256": K2S_DEPENDENCY_SHA256,
                        "k2s_dependency_git_blob": K2S_DEPENDENCY_GIT_BLOB,
                    },
                }
            ).decode("utf-8")
        )
        return 0

    require(args.output_dir is not None, "SCIENTIFIC_OUTPUT_DIR_REQUIRED")
    target = _output_dir(root, Path(args.output_dir))
    manifest = execute_scientific(
        root=root,
        k2s=k2s,
        model=model,
        source_rows=source_rows,
        candidate_rows=candidate_rows,
        population_audit=population_audit,
        contracts=contracts,
        binding=binding,
        output_dir=target,
        runtime=runtime,
        hf=hf,
        handoff=handoff_full,
        encoder_fp=encoder_fp,
        preflight=preflight,
        feasibility=feasibility,
    )
    print(
        canonical_json(
            {
                "scientific_execution": "PASS",
                "output_dir": str(target),
                "scientific_verdict": manifest["primary_statistics"]["scientific_verdict"],
                "primary_statistics": manifest["primary_statistics"],
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
