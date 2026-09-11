"""K0-RVG-P0 state-blind population/token provisioning.

This module is intentionally model-free. It may use only:
- the frozen controlled-data generator,
- archived prior candidate pools,
- the frozen tokenizer,
- deterministic serialization/hashing/provenance utilities.

It must never load a checkpoint, construct a Mamba model, execute logits,
or read recurrent states.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUTHORITY_COMMIT = "ec4230c735be4355531bd46c50ef0571693cb641"
PARENT_PREREG_COMMIT = "cdad87acf664cd61e48406f9d4568b6ab206da24"
PARENT_PREREG_SHA256 = "b2d4ec941c55b3a25ff3c30653784c73fbbc34fbaf31b2fb6493f09971a8c386"

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

GENERATOR_REL = "scripts/build_controlled_v5.py"
GENERATOR_SHA256 = "4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c"
GENERATOR_GIT_BLOB = "baee23a9f71333125f4a8735c2c92d20cab7eb4f"
EXPLICIT_TEMPLATE_COUNT = 30
PHASE_PERIOD = 168

FRESH_START = 1236
FRESH_STOP = 1572
FRESH_N = 336
SOURCE_ROW_COUNT = 4368
FIRST_PAIR_ID = "generated_fact_1237"
LAST_PAIR_ID = "generated_fact_1572"

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"
TOKENIZER_CLASS_PREFIX = "GPTNeoXTokenizer"

CANDIDATE_SCHEMA = "k0-rvg-p0-candidate-v1"
TOKEN_SCHEMA = "k0-rvg-p0-token-contract-v1"
MAPPING_SCHEMA = "k0-rvg-p0-phase-map-v1"
MANIFEST_SCHEMA = "k0-rvg-p0-provisioning-manifest-v1"

SCRIPT_REL = "scripts/longterm_k0_rvg_p0_state_blind_provisioning.py"
TEST_REL = "tests/test_longterm_k0_rvg_p0_state_blind_provisioning.py"

HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}
IMPLEMENTATION_UNTRACKED = {SCRIPT_REL, TEST_REL}

PRIOR_POOLS = {
    "k2w": (
        "reports/longterm_k2w_fixed_window_phase_a_c7c7a0c218bb/candidate_pool.jsonl",
        "abf693d3267cc4e3dd27a8127d2948b36fdf8ba24e135a643215f0f31a26d808",
    ),
    "k2r_k3": (
        "reports/longterm_k2r_claim_disjoint_replication_52bd363_v1/candidate_pool.jsonl",
        "00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4",
    ),
    "k3c": (
        "reports/longterm_k3c_contribution_db75edfbf34b_v1/candidate_pool.jsonl",
        "9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e",
    ),
    "k3t": (
        "reports/longterm_k3t_r_transportability_8c96b476da28_v1/candidate_pool.jsonl",
        "d95d245e358ff497ea50b95e4f54d1192be64d09fec2c06538fc15f75b09ef70",
    ),
}

SIGNATURE_FIELDS = (
    "title",
    "name",
    "alternate_title",
    "alternate_name",
    "role",
    "alternate_role",
    "predicate",
    "alternate_predicate",
    "time",
    "alternate_time",
    "location",
    "alternate_location",
)

ARTIFACT_NAMES = (
    "generated_source.jsonl",
    "candidate_pool.jsonl",
    "phase_pair_mapping.json",
    "token_contracts.jsonl",
    "provisioning_manifest.json",
)

FORBIDDEN_SOURCE_TOKENS = (
    "longterm_k0_rvg_raw_recurrence_observer",
    "load_authenticated_checkpoint",
    "build_a0_model",
    "_full_model_forward",
    "_logits",
    "TraceCollector",
    "RawRecurrenceCollector",
    "torch",
)


class ContractError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(obj: Any, *, final_lf: bool = False) -> bytes:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    payload = b"".join(canonical_json_bytes(dict(row), final_lf=True) for row in rows)
    require(payload.endswith(b"\n"), "JSONL_FINAL_LF_MISSING")
    require(not payload.startswith(b"\xef\xbb\xbf"), "JSONL_BOM_FORBIDDEN")
    require(b"\r" not in payload, "JSONL_CR_FORBIDDEN")
    return payload


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"GIT_FAILURE:{' '.join(args)}") from exc


def repo_contract(root: Path) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")
    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", AUTHORITY_COMMIT, head],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "P0_AUTHORITY_NOT_ANCESTOR")

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True
    ).splitlines()
    allowed = HISTORICAL_K1_UNTRACKED | IMPLEMENTATION_UNTRACKED
    for line in status:
        path = line[3:].replace("\\", "/") if len(line) >= 4 else ""
        require(line[:2] == "??" and path in allowed, "GIT_DIRTY_CONTRACT_MISMATCH")

    return {
        "runtime_branch": branch,
        "runtime_git_head": head,
        "p0_authority_commit": AUTHORITY_COMMIT,
        "p0_authority_is_ancestor": True,
        "runtime_dirty_contract": status,
    }


def load_generator(root: Path):
    path = root / GENERATOR_REL
    require(path.is_file(), "GENERATOR_MISSING")
    require(file_sha256(path) == GENERATOR_SHA256, "GENERATOR_SHA256_MISMATCH")
    require(
        _git(root, "rev-parse", f"HEAD:{GENERATOR_REL}") == GENERATOR_GIT_BLOB,
        "GENERATOR_GIT_BLOB_MISMATCH",
    )
    spec = importlib.util.spec_from_file_location("k0_rvg_p0_generator", path)
    require(spec is not None and spec.loader is not None, "GENERATOR_IMPORT_SPEC_FAILURE")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(len(module.FACT_TEMPLATES) == EXPLICIT_TEMPLATE_COUNT, "EXPLICIT_TEMPLATE_COUNT_MISMATCH")
    return module


def claim_from_prefix(prefix_text: str) -> str:
    require(prefix_text.startswith("Claim: "), "PRIOR_PREFIX_CLAIM_MISSING")
    line = prefix_text.split("\n", 1)[0]
    claim = line[len("Claim: "):]
    require(bool(claim), "PRIOR_PREFIX_EMPTY_CLAIM")
    return claim


def claim_sha256(claim: str) -> str:
    return sha256_bytes(claim.encode("utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"JSONL_PARSE_FAILURE:{path}:{lineno}") from exc
        require(isinstance(obj, dict), f"JSONL_OBJECT_REQUIRED:{path}:{lineno}")
        rows.append(obj)
    return rows


def authenticate_prior_pools(root: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for name, (rel, wanted_sha) in PRIOR_POOLS.items():
        path = root / rel
        require(path.is_file(), f"PRIOR_POOL_MISSING:{name}")
        require(file_sha256(path) == wanted_sha, f"PRIOR_POOL_SHA_MISMATCH:{name}")
        out[name] = load_jsonl(path)
        require(bool(out[name]), f"PRIOR_POOL_EMPTY:{name}")
    return out


def build_source_and_candidates(generator: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_templates = generator.fact_templates_for_count(FRESH_STOP)
    require(len(all_templates) == FRESH_STOP, "GENERATED_TEMPLATE_COUNT_MISMATCH")
    templates = [dict(x) for x in all_templates[FRESH_START:FRESH_STOP]]
    require(len(templates) == FRESH_N, "FRESH_TEMPLATE_SLICE_COUNT_MISMATCH")
    require(templates[0]["pair_id"] == FIRST_PAIR_ID, "FIRST_PAIR_ID_MISMATCH")
    require(templates[-1]["pair_id"] == LAST_PAIR_ID, "LAST_PAIR_ID_MISMATCH")

    source_rows = [dict(x) for x in generator._build_records(templates)]
    require(len(source_rows) == SOURCE_ROW_COUNT, "SOURCE_ROW_COUNT_MISMATCH")

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)

    candidates: list[dict[str, Any]] = []
    for local_index, fact in enumerate(templates):
        pair_id = str(fact["pair_id"])
        rows = by_pair.get(pair_id, [])
        require(len(rows) == 13, f"SOURCE_PAIR_ROW_COUNT_MISMATCH:{pair_id}")

        trunc = [
            r for r in rows
            if r["intervention_type"] == "evidence_truncation"
            and r["final_label"] == "NOT_ENTITLED"
            and r["primary_failure_type"] == "sufficiency"
            and int(r["sufficiency_label"]) == 0
        ]
        control = [
            r for r in rows
            if r["intervention_type"] == "entity_swap"
            and r["final_label"] == "NOT_ENTITLED"
            and r["primary_failure_type"] == "frame"
            and r["polarity_label"] == "NONE"
        ]
        correction = [
            r for r in rows
            if r["intervention_type"] in {"polarity_flip", "none"}
            and r["final_label"] == "REFUTE"
            and r["polarity_label"] == "REFUTE"
        ]
        require(len(trunc) == 1, f"TRUNCATION_MULTIPLICITY_FAILURE:{pair_id}")
        require(len(control) == 1, f"CONTROL_MULTIPLICITY_FAILURE:{pair_id}")
        require(len(correction) == 1, f"CORRECTION_MULTIPLICITY_FAILURE:{pair_id}")

        claims = {str(trunc[0]["claim"]), str(control[0]["claim"]), str(correction[0]["claim"])}
        require(len(claims) == 1, f"CLAIM_CONSISTENCY_FAILURE:{pair_id}")
        claim = next(iter(claims))

        global_index = FRESH_START + local_index
        phase = (global_index - EXPLICIT_TEMPLATE_COUNT) % PHASE_PERIOD
        cycle = local_index // PHASE_PERIOD
        prefix_text = (
            f"Claim: {claim}\n"
            f"Evidence: {trunc[0]['evidence']}\n"
            "Additional evidence:\n"
        )
        recipe = {
            "schema_version": CANDIDATE_SCHEMA,
            "generator_sha256": GENERATOR_SHA256,
            "global_template_index": global_index,
            "local_template_index": local_index,
            "generator_phase_class": phase,
            "cycle_in_slice": cycle,
            "pair_id": pair_id,
            "truncation_source_id": str(trunc[0]["id"]),
            "correction_source_id": str(correction[0]["id"]),
            "correction_source_intervention": str(correction[0]["intervention_type"]),
            "control_source_id": str(control[0]["id"]),
            "prefix_text": prefix_text,
            "correction_text": str(correction[0]["evidence"]),
            "control_text": str(control[0]["evidence"]),
            "base_claim_sha256": claim_sha256(claim),
        }
        stable_hash = sha256_bytes(canonical_json_bytes(recipe))
        candidate = {**recipe, "stable_item_id": "k0-rvg-p0-v1:" + stable_hash}
        candidates.append(candidate)

    require(len(candidates) == FRESH_N, "CANDIDATE_COUNT_MISMATCH")
    return source_rows, candidates, templates


def validate_phase_structure(
    candidates: Sequence[Mapping[str, Any]],
    templates: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    require(len(candidates) == FRESH_N and len(templates) == FRESH_N, "PHASE_INPUT_COUNT_MISMATCH")
    phase_counts = Counter(int(c["generator_phase_class"]) for c in candidates)
    require(set(phase_counts) == set(range(PHASE_PERIOD)), "PHASE_CLASS_SET_MISMATCH")
    require(set(phase_counts.values()) == {2}, "PHASE_CLASS_CARDINALITY_MISMATCH")

    correction_counts = Counter(str(c["correction_source_intervention"]) for c in candidates)
    require(correction_counts == Counter({"polarity_flip": 168, "none": 168}), "CORRECTION_SOURCE_BALANCE_MISMATCH")

    for p in range(PHASE_PERIOD):
        a = candidates[p]
        b = candidates[p + PHASE_PERIOD]
        expected_phase = (FRESH_START + p - EXPLICIT_TEMPLATE_COUNT) % PHASE_PERIOD
        require(
            int(a["generator_phase_class"]) == expected_phase,
            f"PHASE_A_INDEX_MISMATCH:{p}",
        )
        require(
            int(b["generator_phase_class"]) == expected_phase,
            f"PHASE_B_INDEX_MISMATCH:{p}",
        )
        sig_a = tuple(str(templates[p][field]) for field in SIGNATURE_FIELDS)
        sig_b = tuple(str(templates[p + PHASE_PERIOD][field]) for field in SIGNATURE_FIELDS)
        require(sig_a == sig_b, f"PHASE_SIGNATURE_MISMATCH:{p}")
        require(str(a["pair_id"]) != str(b["pair_id"]), f"PHASE_PAIR_ID_COLLISION:{p}")
        require(
            {str(a["correction_source_intervention"]), str(b["correction_source_intervention"])}
            == {"polarity_flip", "none"},
            f"PHASE_CORRECTION_PAIR_MISMATCH:{p}",
        )
        require(a["base_claim_sha256"] != b["base_claim_sha256"], f"PHASE_CLAIM_HASH_COLLISION:{p}")
    return dict(correction_counts)


def phase_mapping(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for p in range(PHASE_PERIOD):
        a = candidates[p]
        b = candidates[p + PHASE_PERIOD]
        blocks.append({
            "block_index": p,
            "phase_class": int(a["generator_phase_class"]),
            "item_a_local_index": p,
            "item_b_local_index": p + PHASE_PERIOD,
            "item_a_pair_id": a["pair_id"],
            "item_b_pair_id": b["pair_id"],
            "item_a_stable_id": a["stable_item_id"],
            "item_b_stable_id": b["stable_item_id"],
            "item_a_correction_source_intervention": a["correction_source_intervention"],
            "item_b_correction_source_intervention": b["correction_source_intervention"],
        })
    return {"schema_version": MAPPING_SCHEMA, "blocks": blocks}


def overlap_audit(
    candidates: Sequence[Mapping[str, Any]],
    prior_pools: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    fresh_pair = {str(c["pair_id"]) for c in candidates}
    fresh_sha = {str(c["base_claim_sha256"]) for c in candidates}
    fresh_claim = {claim_from_prefix(str(c["prefix_text"])) for c in candidates}
    require(len(fresh_pair) == FRESH_N, "FRESH_PAIR_ID_UNIQUENESS_FAILURE")
    require(len(fresh_sha) == FRESH_N, "FRESH_CLAIM_SHA_UNIQUENESS_FAILURE")
    require(len(fresh_claim) == FRESH_N, "FRESH_CLAIM_TEXT_UNIQUENESS_FAILURE")

    result: dict[str, int] = {}
    for pool_name, rows in prior_pools.items():
        pair_ids = {str(r["pair_id"]) for r in rows}
        claim_shas = {
            str(r.get("base_claim_sha256") or claim_sha256(claim_from_prefix(str(r["prefix_text"]))))
            for r in rows
        }
        claims = {claim_from_prefix(str(r["prefix_text"])) for r in rows}
        counts = {
            f"{pool_name}.pair_id": len(fresh_pair & pair_ids),
            f"{pool_name}.claim_text": len(fresh_claim & claims),
            f"{pool_name}.claim_sha256": len(fresh_sha & claim_shas),
        }
        for key, count in counts.items():
            require(count == 0, f"PRIOR_POOL_OVERLAP:{key}:{count}")
        result.update(counts)
    require(len(result) == 12, "OVERLAP_COUNT_CARDINALITY_MISMATCH")
    return result


def load_tokenizer():
    import transformers
    from transformers import AutoTokenizer

    require(transformers.__version__ == TRANSFORMERS_VERSION, "TRANSFORMERS_VERSION_MISMATCH")
    tok = AutoTokenizer.from_pretrained(
        HF_MODEL,
        revision=HF_REVISION,
        use_fast=True,
    )
    cls_name = type(tok).__name__
    require(cls_name.startswith(TOKENIZER_CLASS_PREFIX), "TOKENIZER_CLASS_MISMATCH")
    require(bool(getattr(tok, "is_fast", False)), "TOKENIZER_NOT_FAST")
    return tok, cls_name


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    ids = encoded["input_ids"]
    require(isinstance(ids, list) and all(type(v) is int for v in ids), "TOKEN_IDS_INVALID")
    return ids


def divergence_anchor(corr: Sequence[int], ctrl: Sequence[int], prefix_len: int) -> int:
    upper = min(len(corr), len(ctrl), prefix_len + 8)
    for k in range(prefix_len, upper):
        if corr[k] != ctrl[k]:
            return k
    raise ContractError("DIVERGENCE_NOT_WITHIN_FIRST_8_CONTINUATION_TOKENS")


def token_contracts(
    candidates: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int], int]:
    rows: list[dict[str, Any]] = []
    matched_hist: Counter[int] = Counter()
    swapped_hist: Counter[int] = Counter()
    min_available = 10**9

    for i, item in enumerate(candidates):
        mate_i = i + PHASE_PERIOD if i < PHASE_PERIOD else i - PHASE_PERIOD
        mate = candidates[mate_i]
        prefix = str(item["prefix_text"])
        prefix_ids = token_ids(tokenizer, prefix)
        require(len(prefix_ids) >= 1, f"PREFIX_TOKEN_EMPTY:{i}")
        prefix_sha = sha256_bytes(canonical_json_bytes(prefix_ids))

        branches = {
            "matched_corr": prefix + str(item["correction_text"]),
            "matched_ctrl": prefix + str(item["control_text"]),
            "swapped_corr": prefix + str(mate["correction_text"]),
            "swapped_ctrl": prefix + str(mate["control_text"]),
        }
        branch_ids = {name: token_ids(tokenizer, text) for name, text in branches.items()}
        for name, ids in branch_ids.items():
            require(ids[:len(prefix_ids)] == prefix_ids, f"PREFIX_TOKEN_IDENTITY_FAILURE:{i}:{name}")

        m_te = divergence_anchor(branch_ids["matched_corr"], branch_ids["matched_ctrl"], len(prefix_ids))
        s_te = divergence_anchor(branch_ids["swapped_corr"], branch_ids["swapped_ctrl"], len(prefix_ids))
        require(m_te >= 1 and s_te >= 1, f"INCOMING_VELOCITY_UNAVAILABLE:{i}")
        m_corr_avail = len(branch_ids["matched_corr"]) - m_te
        m_ctrl_avail = len(branch_ids["matched_ctrl"]) - m_te
        s_corr_avail = len(branch_ids["swapped_corr"]) - s_te
        s_ctrl_avail = len(branch_ids["swapped_ctrl"]) - s_te
        require(min(m_corr_avail, m_ctrl_avail) >= 8, f"MATCHED_W8_UNAVAILABLE:{i}")
        require(min(s_corr_avail, s_ctrl_avail) >= 8, f"SWAPPED_W8_UNAVAILABLE:{i}")
        m_off = m_te - len(prefix_ids)
        s_off = s_te - len(prefix_ids)
        require(0 <= m_off <= 7, f"MATCHED_DIVERGENCE_OFFSET_INVALID:{i}")
        require(0 <= s_off <= 7, f"SWAPPED_DIVERGENCE_OFFSET_INVALID:{i}")
        matched_hist[m_off] += 1
        swapped_hist[s_off] += 1
        min_available = min(min_available, m_corr_avail, m_ctrl_avail, s_corr_avail, s_ctrl_avail)

        rows.append({
            "schema_version": TOKEN_SCHEMA,
            "stable_item_id": item["stable_item_id"],
            "pair_id": item["pair_id"],
            "local_template_index": i,
            "phase_block_index": i % PHASE_PERIOD,
            "prefix_token_count": len(prefix_ids),
            "prefix_token_sha256": prefix_sha,
            "matched_correction_token_count": len(branch_ids["matched_corr"]),
            "matched_control_token_count": len(branch_ids["matched_ctrl"]),
            "matched_divergence_anchor": m_te,
            "matched_divergence_offset_from_prefix": m_off,
            "matched_w8_available": True,
            "swapped_correction_token_count": len(branch_ids["swapped_corr"]),
            "swapped_control_token_count": len(branch_ids["swapped_ctrl"]),
            "swapped_divergence_anchor": s_te,
            "swapped_divergence_offset_from_prefix": s_off,
            "swapped_w8_available": True,
            "phase_mate_stable_id": mate["stable_item_id"],
            "phase_mate_pair_id": mate["pair_id"],
        })
    require(len(rows) == FRESH_N, "TOKEN_CONTRACT_COUNT_MISMATCH")
    return rows, dict(sorted(matched_hist.items())), dict(sorted(swapped_hist.items())), int(min_available)


def static_no_scientific_access_check() -> None:
    path = Path(__file__).resolve()
    text = path.read_text("utf-8")
    tree = ast.parse(text)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    require(not any("longterm_k0_rvg_raw_recurrence_observer" in x for x in imports), "OBSERVER_IMPORT_FORBIDDEN")
    require(not any(x == "torch" or x.startswith("torch.") for x in imports), "TORCH_IMPORT_FORBIDDEN")
    # Forbidden call/import tokens may appear in the policy constant itself; scan the AST calls/imports
    # rather than raw source for those.
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in {"torch", "longterm_k0_rvg_raw_recurrence_observer"}:
            continue
        require(token not in called, f"FORBIDDEN_SCIENTIFIC_CALL:{token}")


@dataclass(frozen=True)
class ProvisionResult:
    artifacts: dict[str, bytes]
    manifest: dict[str, Any]


def materialize(root: Path) -> ProvisionResult:
    static_no_scientific_access_check()
    provenance = repo_contract(root)
    generator = load_generator(root)
    prior_pools = authenticate_prior_pools(root)
    source_rows, candidates, templates = build_source_and_candidates(generator)
    correction_counts = validate_phase_structure(candidates, templates)
    overlaps = overlap_audit(candidates, prior_pools)
    tokenizer, tokenizer_class = load_tokenizer()
    token_rows, matched_hist, swapped_hist, min_available = token_contracts(candidates, tokenizer)
    mapping = phase_mapping(candidates)

    artifacts: dict[str, bytes] = {
        "generated_source.jsonl": canonical_jsonl_bytes(source_rows),
        "candidate_pool.jsonl": canonical_jsonl_bytes(candidates),
        "phase_pair_mapping.json": canonical_json_bytes(mapping, final_lf=False),
        "token_contracts.jsonl": canonical_jsonl_bytes(token_rows),
    }
    hashes = {name: sha256_bytes(raw) for name, raw in artifacts.items()}

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "runtime_git_head": provenance["runtime_git_head"],
        "p0_authority_commit": AUTHORITY_COMMIT,
        "p0_authority_is_ancestor": True,
        "parent_preregistration_commit": PARENT_PREREG_COMMIT,
        "parent_preregistration_sha256": PARENT_PREREG_SHA256,
        "generator": {
            "path": GENERATOR_REL,
            "sha256": GENERATOR_SHA256,
            "git_blob": GENERATOR_GIT_BLOB,
            "explicit_template_count": EXPLICIT_TEMPLATE_COUNT,
        },
        "tokenizer": {
            "model_id": HF_MODEL,
            "revision": HF_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "class": tokenizer_class,
            "is_fast": True,
            "add_special_tokens": False,
        },
        "fresh_template_range": [FRESH_START, FRESH_STOP],
        "first_pair_id": FIRST_PAIR_ID,
        "last_pair_id": LAST_PAIR_ID,
        "item_count": len(candidates),
        "phase_block_count": PHASE_PERIOD,
        "source_row_count": len(source_rows),
        "correction_source_counts": correction_counts,
        "prior_overlap_counts": overlaps,
        "matched_divergence_offset_histogram": {str(k): v for k, v in matched_hist.items()},
        "swapped_divergence_offset_histogram": {str(k): v for k, v in swapped_hist.items()},
        "minimum_branch_post_divergence_availability": min_available,
        "artifact_sha256": hashes,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "model_forward_executed": False,
        "logits_read": False,
        "recurrent_state_read": False,
        "observer_imported": False,
        "scientific_endpoint_computed": False,
        "status": "STATE_BLIND_INPUT_CONTRACT_VALID",
    }
    artifacts["provisioning_manifest.json"] = canonical_json_bytes(manifest, final_lf=True)
    return ProvisionResult(artifacts=artifacts, manifest=manifest)


def write_artifacts(result: ProvisionResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    require(set(result.artifacts) == set(ARTIFACT_NAMES), "ARTIFACT_SET_MISMATCH")
    hashes: dict[str, str] = {}
    for name in ARTIFACT_NAMES:
        raw = result.artifacts[name]
        path = output_dir / name
        path.write_bytes(raw)
        hashes[name] = sha256_bytes(raw)
    return hashes


def deterministic_repeat_check(root: Path, first: ProvisionResult) -> None:
    second = materialize(root)
    require(set(first.artifacts) == set(second.artifacts), "REPEAT_ARTIFACT_SET_MISMATCH")
    for name in ARTIFACT_NAMES:
        require(first.artifacts[name] == second.artifacts[name], f"REPEAT_BYTE_IDENTITY_FAILURE:{name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K0-RVG-P0 state-blind provisioning only.")
    parser.add_argument("--provision", action="store_true", help="Materialize the frozen state-blind input contract.")
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory for the five state-blind artifacts.")
    parser.add_argument("--validate-repeat", action="store_true", help="Re-materialize and require byte-identical artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.provision:
        parser.error("only --provision is implemented")
    if args.output_dir is None:
        parser.error("--output-dir is required")
    root = Path(__file__).resolve().parents[1]
    result = materialize(root)
    if args.validate_repeat:
        deterministic_repeat_check(root, result)
    hashes = write_artifacts(result, args.output_dir.resolve())
    summary = {
        "status": result.manifest["status"],
        "output_dir": str(args.output_dir.resolve()),
        "artifact_sha256": hashes,
        "item_count": result.manifest["item_count"],
        "source_row_count": result.manifest["source_row_count"],
        "phase_block_count": result.manifest["phase_block_count"],
        "correction_source_counts": result.manifest["correction_source_counts"],
        "prior_overlap_counts": result.manifest["prior_overlap_counts"],
        "matched_divergence_offset_histogram": result.manifest["matched_divergence_offset_histogram"],
        "swapped_divergence_offset_histogram": result.manifest["swapped_divergence_offset_histogram"],
        "minimum_branch_post_divergence_availability": result.manifest["minimum_branch_post_divergence_availability"],
        "model_loaded": False,
        "checkpoint_loaded": False,
        "model_forward_executed": False,
        "logits_read": False,
        "recurrent_state_read": False,
        "observer_imported": False,
        "scientific_endpoint_computed": False,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
