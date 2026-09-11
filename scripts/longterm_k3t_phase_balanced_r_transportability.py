"""K3T phase-balanced R transportability.

Bounded implementation under the frozen K3T preregistration.

This module intentionally separates three boundaries:

1. state-blind population/tokenizer validation;
2. synthetic non-study instrumentation validation;
3. scientific population recurrent-state execution.

The third boundary is fail-closed unless a separate one-file K3T scientific
execution-authority commit is the exact runtime HEAD.  The frozen K3T
preregistration alone does not authorize scientific execution.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

# Direct CLI execution sets sys.path[0] to the scripts directory.  Bind the
# repository root explicitly so frozen sibling modules under scripts.* resolve
# identically under direct execution and import-based tests.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Frozen preregistration / dependency bindings
# ---------------------------------------------------------------------------

K3T_PREREG_COMMIT = "58cbcd316c7714ddc8c041c2a2ec4376e79a4bd0"
K3T_PREREG_REL = "reports/longterm_k3t_phase_balanced_r_transportability_prereg_candidate.md"
K3T_PREREG_SHA256 = "57b5bd2375fbacb7ef5e22260f3f1ecf71dba839f383f7726c03544e33a45cc5"

R_AUDIT_COMMIT = "aeb2960cd6cbe7ab78c4c0e42e350a6699070d79"
R_AUDIT_REL = "reports/longterm_k3c_r_cross_population_transportability_audit_report_candidate.md"
R_AUDIT_SHA256 = "cf0b7f933d2f3299b7d3e4650337a8950dfa82ef95734e0978b9a2385b61f437"

GENERATOR_REL = "scripts/build_controlled_v5.py"
GENERATOR_SHA256 = "4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c"
GENERATOR_GIT_BLOB = "baee23a9f71333125f4a8735c2c92d20cab7eb4f"

K2S_HELPER_REL = "scripts/longterm_k2s_pair_specific_event_dynamics.py"
K2S_HELPER_SHA256 = "f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8"
K2S_HELPER_GIT_BLOB = "3a651fb508669bdcf72441a4869b863d6eee6c1f"

PRIOR_POOLS = {
    "K2W": (
        "reports/longterm_k2w_fixed_window_phase_a_c7c7a0c218bb/candidate_pool.jsonl",
        "abf693d3267cc4e3dd27a8127d2948b36fdf8ba24e135a643215f0f31a26d808",
    ),
    "K2R_K3": (
        "reports/longterm_k2r_claim_disjoint_replication_52bd363_v1/candidate_pool.jsonl",
        "00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4",
    ),
    "K3C": (
        "reports/longterm_k3c_contribution_db75edfbf34b_v1/candidate_pool.jsonl",
        "9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e",
    ),
}

AUTHORITY_REL = "reports/longterm_k3t_phase_balanced_r_transportability_execution_authority_spec_candidate.md"
AUTHORITY_SCHEMA = "k3t-scientific-execution-authority-v1"

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"
HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}
K3T_IMPLEMENTATION_UNTRACKED = {
    "scripts/longterm_k3t_phase_balanced_r_transportability.py",
    "tests/test_longterm_k3t_phase_balanced_r_transportability.py",
}

# ---------------------------------------------------------------------------
# Frozen K3T population / measurement constants
# ---------------------------------------------------------------------------

GLOBAL_TEMPLATE_START = 900
GLOBAL_TEMPLATE_STOP = 1236
FIRST_PAIR_ID = "generated_fact_901"
LAST_PAIR_ID = "generated_fact_1236"

N_ITEMS = 336
N_BLOCKS = 168
PHASE_PERIOD = 168
W = 8
PRIMARY_LAYER = 23

CANDIDATE_SCHEMA = "k3t-phase-balanced-population-v1"
STABLE_ID_PREFIX = "k3t-v1:"

GENERATED_SOURCE_ROWS = 4368
GENERATED_SOURCE_SHA256 = "fb699bcc99e00b8c437fd49c204345933d61c69215af853746bc4639545c7400"
CANDIDATE_POOL_SHA256 = "d95d245e358ff497ea50b95e4f54d1192be64d09fec2c06538fc15f75b09ef70"
RECIPROCAL_MAPPING_SHA256 = "1c458054540ac0d38857cb3f55595e0b286428225ef11ed22059981dfd1e28ad"

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"
TOKENIZER_CLASS = "GPTNeoXTokenizer"

EXPECTED_ZIP_SHA256 = "96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861"
EXPECTED_CHECKPOINT_SHA256 = "4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c"
ENCODER_CANONICAL_SHA256 = "48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"
ENCODER_RAW_SHA256 = "968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"
ENCODER_TENSOR_COUNT = 242
ENCODER_NUMEL = 129_135_360
ENCODER_RAW_BYTES = 516_541_440

RECURRENCE_SOURCE_SHA256 = "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"

POSITIVE_VERDICT = "R_PHASE_BALANCED_TRANSPORTABILITY_SIGNAL_REPLICATED"
NEGATIVE_VERDICT = "R_PHASE_BALANCED_TRANSPORTABILITY_SIGNAL_CONTRADICTED"
NULL_VERDICT = "R_PHASE_BALANCED_TRANSPORTABILITY_NOT_ESTABLISHED"

ITEM_SCHEMA = "k3t-r-transportability-item-v1"
BLOCK_SCHEMA = "k3t-r-transportability-block-v1"
PRIMARY_SCHEMA = "k3t-r-transportability-primary-v1"
INTEGRITY_SCHEMA = "k3t-r-transportability-integrity-v1"
MANIFEST_SCHEMA = "k3t-r-transportability-manifest-v1"

LEXICAL_FIELDS = (
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


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def parse_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    require(not raw.startswith(b"\xef\xbb\xbf"), "JSONL_BOM_FORBIDDEN")
    require(b"\r" not in raw, "JSONL_CR_FORBIDDEN")
    require(raw.endswith(b"\n"), "JSONL_FINAL_LF_REQUIRED")
    body = raw[:-1]
    require(bool(body), "JSONL_EMPTY")
    rows: list[dict[str, Any]] = []
    for line in body.split(b"\n"):
        require(bool(line), "JSONL_BLANK_LINE_FORBIDDEN")
        try:
            value = json.loads(line.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("JSONL_DECODE_FAILURE") from exc
        require(isinstance(value, dict), "JSONL_ROW_NOT_OBJECT")
        rows.append(value)
    return rows


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("GIT_FAILURE:" + " ".join(args)) from exc


def _git_bytes(root: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(["git", *args], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("GIT_FAILURE:" + " ".join(args)) from exc


def _status_lines(root: Path) -> list[str]:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain=v1"],
            cwd=root,
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("GIT_STATUS_FAILURE") from exc


def _status_path(line: str) -> str:
    return line[3:].replace("\\", "/") if len(line) >= 4 else ""


def _require_file_identity(
    root: Path,
    rel: str,
    expected_sha256: str,
    expected_blob: str | None = None,
    ref: str = "HEAD",
) -> None:
    path = root / rel
    require(path.is_file(), "DEPENDENCY_MISSING:" + rel)
    require(file_sha256(path) == expected_sha256, "DEPENDENCY_SHA256_MISMATCH:" + rel)
    if expected_blob is not None:
        actual_blob = _git(root, "rev-parse", f"{ref}:{rel}")
        require(actual_blob == expected_blob, "DEPENDENCY_GIT_BLOB_MISMATCH:" + rel)


def _require_prereg_and_dependencies(root: Path) -> None:
    branch = _git(root, "branch", "--show-current")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")

    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", K3T_PREREG_COMMIT, "HEAD"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "K3T_PREREG_NOT_ANCESTOR")

    prereg_raw = _git_bytes(root, "show", f"{K3T_PREREG_COMMIT}:{K3T_PREREG_REL}")
    require(sha256_bytes(prereg_raw) == K3T_PREREG_SHA256, "K3T_PREREG_SHA256_MISMATCH")

    audit_raw = _git_bytes(root, "show", f"{R_AUDIT_COMMIT}:{R_AUDIT_REL}")
    require(sha256_bytes(audit_raw) == R_AUDIT_SHA256, "R_AUDIT_SHA256_MISMATCH")

    _require_file_identity(root, GENERATOR_REL, GENERATOR_SHA256, GENERATOR_GIT_BLOB)
    _require_file_identity(root, K2S_HELPER_REL, K2S_HELPER_SHA256, K2S_HELPER_GIT_BLOB)

    for label, (rel, expected_sha) in PRIOR_POOLS.items():
        _require_file_identity(root, rel, expected_sha)


def git_provenance(root: Path, implementation_preflight: bool = False) -> dict[str, Any]:
    _require_prereg_and_dependencies(root)

    status = _status_lines(root)
    allowed = set(HISTORICAL_K1_UNTRACKED)
    if implementation_preflight:
        allowed |= K3T_IMPLEMENTATION_UNTRACKED

    for line in status:
        path = _status_path(line)
        require(
            line[:2] == "??" and path in allowed,
            "GIT_DIRTY_CONTRACT_MISMATCH:" + line,
        )

    return {
        "runtime_branch": _git(root, "branch", "--show-current"),
        "runtime_git_head": _git(root, "rev-parse", "HEAD"),
        "runtime_dirty_contract": status,
        "k3t_prereg_commit": K3T_PREREG_COMMIT,
        "k3t_prereg_sha256": K3T_PREREG_SHA256,
        "generator_git_blob": GENERATOR_GIT_BLOB,
        "generator_sha256": GENERATOR_SHA256,
        "k2s_helper_git_blob": K2S_HELPER_GIT_BLOB,
        "k2s_helper_sha256": K2S_HELPER_SHA256,
    }


def _load_k2s_helper() -> Any:
    return importlib.import_module("scripts.longterm_k2s_pair_specific_event_dynamics")


def _load_generator() -> Any:
    return importlib.import_module("scripts.build_controlled_v5")


def claim_text_from_prefix(prefix: str) -> str:
    marker = "\nEvidence:"
    require(marker in prefix, "PREFIX_CLAIM_MARKER_MISSING")
    return prefix.split(marker, 1)[0]


def phase_class(global_template_index: int, fixed_seed_count: int = 30) -> int:
    require(global_template_index >= fixed_seed_count, "GLOBAL_TEMPLATE_BEFORE_GENERATED_REGION")
    return (global_template_index - fixed_seed_count) % PHASE_PERIOD


def _validate_phase_signatures(
    templates: Sequence[Mapping[str, Any]],
    fixed_seed_count: int,
) -> None:
    require(len(templates) == N_ITEMS, "TEMPLATE_COUNT_MISMATCH")
    counts: Counter[int] = Counter()
    signatures: dict[int, tuple[str, ...]] = {}

    for local_index, row in enumerate(templates):
        global_index = GLOBAL_TEMPLATE_START + local_index
        phase = phase_class(global_index, fixed_seed_count)
        signature = tuple(str(row[field]) for field in LEXICAL_FIELDS)
        counts[phase] += 1
        if phase in signatures:
            require(signatures[phase] == signature, f"PHASE_SIGNATURE_DRIFT:{phase}")
        else:
            signatures[phase] = signature

    require(len(counts) == PHASE_PERIOD, "PHASE_CLASS_COUNT_MISMATCH")
    require(set(counts.values()) == {2}, "PHASE_COUNT_NOT_EXACTLY_TWO")
    require(len(set(signatures.values())) == PHASE_PERIOD, "PHASE_SIGNATURE_COLLISION")

    for i in range(PHASE_PERIOD):
        left = tuple(str(templates[i][field]) for field in LEXICAL_FIELDS)
        right = tuple(str(templates[i + PHASE_PERIOD][field]) for field in LEXICAL_FIELDS)
        require(left == right, f"TWO_CYCLE_LEXICAL_MISMATCH:{i}")


def _candidate_rows_from_source(
    templates: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    fixed_seed_count: int,
) -> list[dict[str, Any]]:
    pair_ids = [str(row["pair_id"]) for row in templates]
    expected_pair_ids = [f"generated_fact_{i:03d}" for i in range(901, 1237)]
    require(pair_ids == expected_pair_ids, "PAIR_ID_SEQUENCE_MISMATCH")
    require(pair_ids[0] == FIRST_PAIR_ID and pair_ids[-1] == LAST_PAIR_ID, "PAIR_ID_BOUNDARY_MISMATCH")

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        groups[str(row["pair_id"])].append(row)
    require(set(groups) == set(pair_ids), "SOURCE_PAIR_SET_MISMATCH")

    pair_to_local = {pair_id: i for i, pair_id in enumerate(pair_ids)}
    correction_sources: Counter[str] = Counter()
    phase_source: Counter[tuple[int, str]] = Counter()
    candidates: list[dict[str, Any]] = []

    for pair_id in sorted(groups):
        rows = groups[pair_id]
        trunc = [r for r in rows if r.get("intervention_type") == "evidence_truncation"]
        entity = [r for r in rows if r.get("intervention_type") == "entity_swap"]
        refute_polarity = [
            r for r in rows
            if r.get("intervention_type") == "polarity_flip"
            and r.get("final_label") == "REFUTE"
        ]
        refute_none = [
            r for r in rows
            if r.get("intervention_type") == "none"
            and r.get("final_label") == "REFUTE"
        ]

        require(len(trunc) == 1, "TRUNCATION_MULTIPLICITY:" + pair_id)
        require(len(entity) == 1, "CONTROL_MULTIPLICITY:" + pair_id)

        if len(refute_polarity) == 1:
            correction = refute_polarity[0]
            correction_source = "polarity_flip"
        elif len(refute_none) == 1:
            correction = refute_none[0]
            correction_source = "none"
        else:
            raise ContractError("CORRECTION_MULTIPLICITY:" + pair_id)

        t = trunc[0]
        e = entity[0]
        q = correction

        require(t["claim"] == e["claim"] == q["claim"], "CLAIM_IDENTITY_MISMATCH:" + pair_id)
        require(
            t.get("final_label") == "NOT_ENTITLED"
            and t.get("primary_failure_type") == "sufficiency"
            and t.get("sufficiency_label") == 0,
            "TRUNCATION_CONTRACT_MISMATCH:" + pair_id,
        )
        require(
            e.get("final_label") == "NOT_ENTITLED"
            and e.get("primary_failure_type") == "frame"
            and e.get("polarity_label") == "NONE",
            "CONTROL_CONTRACT_MISMATCH:" + pair_id,
        )
        require(
            q.get("final_label") == "REFUTE"
            and q.get("polarity_label") == "REFUTE",
            "CORRECTION_CONTRACT_MISMATCH:" + pair_id,
        )

        local_index = pair_to_local[pair_id]
        global_index = GLOBAL_TEMPLATE_START + local_index
        phase = phase_class(global_index, fixed_seed_count)
        cycle = local_index // PHASE_PERIOD

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
            "global_template_index": global_index,
            "generator_phase_period": PHASE_PERIOD,
            "generator_phase_class": phase,
            "cycle_in_slice": cycle,
            "pair_id": pair_id,
            "truncation_source_id": str(t["id"]),
            "correction_source_id": str(q["id"]),
            "correction_source_intervention": correction_source,
            "control_source_id": str(e["id"]),
            "prefix_text": prefix,
            "correction_text": str(q["evidence"]),
            "control_text": str(e["evidence"]),
        }

        candidate = {
            **recipe,
            "stable_item_id": STABLE_ID_PREFIX + sha256_bytes(canonical_json(recipe)),
            "base_claim_sha256": sha256_bytes(str(t["claim"]).encode("utf-8")),
        }
        candidates.append(candidate)
        correction_sources[correction_source] += 1
        phase_source[(phase, correction_source)] += 1

    require(
        correction_sources == Counter({"polarity_flip": 168, "none": 168}),
        "CORRECTION_SOURCE_BALANCE_FAILURE",
    )
    for phase in range(PHASE_PERIOD):
        require(
            phase_source[(phase, "polarity_flip")] == 1
            and phase_source[(phase, "none")] == 1,
            f"PHASE_SOURCE_BALANCE_FAILURE:{phase}",
        )

    candidates.sort(key=lambda row: row["stable_item_id"])
    require(len(candidates) == N_ITEMS, "CANDIDATE_COUNT_MISMATCH")
    require(len({r["stable_item_id"] for r in candidates}) == N_ITEMS, "STABLE_ID_DUPLICATE")
    require(len({r["base_claim_sha256"] for r in candidates}) == N_ITEMS, "CLAIM_SHA_DUPLICATE")
    return candidates


def _overlap_against_prior(root: Path, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    new_sets = {
        "pair_id": {str(r["pair_id"]) for r in candidates},
        "claim_sha": {str(r["base_claim_sha256"]) for r in candidates},
        "claim_text": {claim_text_from_prefix(str(r["prefix_text"])) for r in candidates},
    }
    result: dict[str, Any] = {}

    for label, (rel, expected_sha) in PRIOR_POOLS.items():
        raw = (root / rel).read_bytes()
        require(sha256_bytes(raw) == expected_sha, f"{label}_POOL_SHA_MISMATCH")
        old = parse_jsonl_bytes(raw)
        old_sets = {
            "pair_id": {str(r["pair_id"]) for r in old},
            "claim_sha": {str(r["base_claim_sha256"]) for r in old},
            "claim_text": {claim_text_from_prefix(str(r["prefix_text"])) for r in old},
        }
        overlap = {
            key: len(new_sets[key] & old_sets[key])
            for key in ("pair_id", "claim_sha", "claim_text")
        }
        require(all(value == 0 for value in overlap.values()), f"{label}_OVERLAP_FAILURE")
        result[label] = overlap
    return result


def reciprocal_mapping(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(candidates) == N_ITEMS, "RECIPROCAL_CANDIDATE_COUNT_MISMATCH")
    rows: list[dict[str, Any]] = []
    for block_index in range(N_BLOCKS):
        a = 2 * block_index
        b = a + 1
        require((a ^ 1) == b and (b ^ 1) == a, "RECIPROCAL_XOR_FAILURE")
        rows.append({
            "block_index": block_index,
            "item_a_stable_id": str(candidates[a]["stable_item_id"]),
            "item_b_stable_id": str(candidates[b]["stable_item_id"]),
            "item_a_donor_index": a ^ 1,
            "item_b_donor_index": b ^ 1,
        })
    return rows


def materialize_population(root: Path) -> dict[str, Any]:
    _require_prereg_and_dependencies(root)
    generator = _load_generator()

    fixed_seed_count = len(generator.FACT_TEMPLATES)
    require(fixed_seed_count == 30, "FACT_TEMPLATE_COUNT_DRIFT")

    all_templates = generator.fact_templates_for_count(GLOBAL_TEMPLATE_STOP)
    templates = [dict(row) for row in all_templates[GLOBAL_TEMPLATE_START:GLOBAL_TEMPLATE_STOP]]
    require(len(templates) == N_ITEMS, "TEMPLATE_COUNT_MISMATCH")

    _validate_phase_signatures(templates, fixed_seed_count)

    source_rows = generator._build_records(templates)
    require(len(source_rows) == GENERATED_SOURCE_ROWS, "GENERATED_SOURCE_ROW_COUNT_MISMATCH")
    source_raw = canonical_jsonl(source_rows)
    require(sha256_bytes(source_raw) == GENERATED_SOURCE_SHA256, "GENERATED_SOURCE_SHA256_MISMATCH")

    candidates = _candidate_rows_from_source(templates, source_rows, fixed_seed_count)
    candidate_raw = canonical_jsonl(candidates)
    require(sha256_bytes(candidate_raw) == CANDIDATE_POOL_SHA256, "CANDIDATE_POOL_SHA256_MISMATCH")

    mapping = reciprocal_mapping(candidates)
    mapping_sha = sha256_bytes(canonical_json(mapping))
    require(mapping_sha == RECIPROCAL_MAPPING_SHA256, "RECIPROCAL_MAPPING_SHA256_MISMATCH")

    overlap = _overlap_against_prior(root, candidates)

    phase_counts = Counter(int(r["generator_phase_class"]) for r in candidates)
    phase_source = Counter(
        (int(r["generator_phase_class"]), str(r["correction_source_intervention"]))
        for r in candidates
    )
    require(len(phase_counts) == PHASE_PERIOD, "CANDIDATE_PHASE_CLASS_COUNT_MISMATCH")
    require(set(phase_counts.values()) == {2}, "CANDIDATE_PHASE_COUNT_MISMATCH")
    for phase in range(PHASE_PERIOD):
        require(
            phase_source[(phase, "polarity_flip")] == 1
            and phase_source[(phase, "none")] == 1,
            f"CANDIDATE_PHASE_SOURCE_BALANCE_FAILURE:{phase}",
        )

    return {
        "source_rows": source_rows,
        "source_raw": source_raw,
        "candidates": candidates,
        "candidate_raw": candidate_raw,
        "mapping": mapping,
        "mapping_sha256": mapping_sha,
        "prior_overlap": overlap,
        "phase_class_count": len(phase_counts),
        "phase_count_unique_values": sorted(set(phase_counts.values())),
        "correction_source_counts": dict(
            sorted(Counter(str(r["correction_source_intervention"]) for r in candidates).items())
        ),
    }


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"] if isinstance(out, Mapping) else out.input_ids
    return list(ids)


def _first_divergence(corr: Sequence[int], ctrl: Sequence[int], prefix_len: int) -> int:
    stop = min(len(corr), len(ctrl), prefix_len + W)
    value = next((i for i in range(prefix_len, stop) if corr[i] != ctrl[i]), None)
    require(value is not None, "PAIR_DIVERGENCE_MISSING_WITHIN_W")
    return int(value)


def build_input_contracts(
    candidates: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(len(candidates) == N_ITEMS, "INPUT_CANDIDATE_COUNT_MISMATCH")
    require(getattr(tokenizer, "is_fast", False), "TOKENIZER_MUST_BE_FAST")
    require(type(tokenizer).__name__ == TOKENIZER_CLASS, "TOKENIZER_CLASS_MISMATCH")

    contracts: list[dict[str, Any]] = []
    matched_d: Counter[str] = Counter()
    swapped_d: Counter[str] = Counter()
    availability = {
        "matched_corr": [],
        "matched_ctrl": [],
        "swapped_corr": [],
        "swapped_ctrl": [],
    }

    corr_texts = [str(row["correction_text"]) for row in candidates]
    ctrl_texts = [str(row["control_text"]) for row in candidates]
    swapped_corr_texts = [str(candidates[i ^ 1]["correction_text"]) for i in range(N_ITEMS)]
    swapped_ctrl_texts = [str(candidates[i ^ 1]["control_text"]) for i in range(N_ITEMS)]
    require(sorted(corr_texts) == sorted(swapped_corr_texts), "CORRECTION_MARGINAL_FAILURE")
    require(sorted(ctrl_texts) == sorted(swapped_ctrl_texts), "CONTROL_MARGINAL_FAILURE")

    for i, row in enumerate(candidates):
        donor = candidates[i ^ 1]
        prefix = str(row["prefix_text"])
        prefix_ids = _token_ids(tokenizer, prefix)
        require(len(prefix_ids) >= 2, "PREFIX_TOO_SHORT")
        p = len(prefix_ids) - 1

        branch_texts = {
            "matched_corr": prefix + str(row["correction_text"]),
            "matched_ctrl": prefix + str(row["control_text"]),
            "swapped_corr": prefix + str(donor["correction_text"]),
            "swapped_ctrl": prefix + str(donor["control_text"]),
        }
        branch_token_ids: dict[str, list[int]] = {}
        for name, text in branch_texts.items():
            ids = _token_ids(tokenizer, text)
            require(ids[:len(prefix_ids)] == prefix_ids, "EXACT_PREFIX_TOKEN_FAILURE")
            continuation = len(ids) - len(prefix_ids)
            require(continuation >= W, "WINDOW8_FEASIBILITY_FAILURE")
            branch_token_ids[name] = ids
            availability[name].append(continuation)

        md = _first_divergence(
            branch_token_ids["matched_corr"],
            branch_token_ids["matched_ctrl"],
            len(prefix_ids),
        )
        sd = _first_divergence(
            branch_token_ids["swapped_corr"],
            branch_token_ids["swapped_ctrl"],
            len(prefix_ids),
        )
        matched_d[str(md - p)] += 1
        swapped_d[str(sd - p)] += 1

        contracts.append({
            "item_index": i,
            "block_index": i // 2,
            "block_id": f"k3t-block-{i // 2:03d}",
            "stable_item_id": str(row["stable_item_id"]),
            "pair_id": str(row["pair_id"]),
            "base_claim_sha256": str(row["base_claim_sha256"]),
            "global_template_index": int(row["global_template_index"]),
            "generator_phase_class": int(row["generator_phase_class"]),
            "cycle_in_slice": int(row["cycle_in_slice"]),
            "correction_source_intervention": str(row["correction_source_intervention"]),
            "donor_index": i ^ 1,
            "donor_stable_item_id": str(donor["stable_item_id"]),
            "p": p,
            "prefix_token_ids": prefix_ids,
            "branch_token_ids": branch_token_ids,
            "matched_d_minus_p": md - p,
            "swapped_d_minus_p": sd - p,
        })

    summary = {
        "N_total": N_ITEMS,
        "N_blocks": N_BLOCKS,
        "N_matched_valid": N_ITEMS,
        "N_swapped_valid": N_ITEMS,
        "matched_d_minus_p": dict(sorted(matched_d.items())),
        "swapped_d_minus_p": dict(sorted(swapped_d.items())),
        "matched_corr_available_minmax": [
            min(availability["matched_corr"]),
            max(availability["matched_corr"]),
        ],
        "matched_ctrl_available_minmax": [
            min(availability["matched_ctrl"]),
            max(availability["matched_ctrl"]),
        ],
        "swapped_corr_available_minmax": [
            min(availability["swapped_corr"]),
            max(availability["swapped_corr"]),
        ],
        "swapped_ctrl_available_minmax": [
            min(availability["swapped_ctrl"]),
            max(availability["swapped_ctrl"]),
        ],
        "prefix_marginal_preserved": True,
        "correction_marginal_preserved": True,
        "control_marginal_preserved": True,
    }

    expected = {
        "N_total": 336,
        "N_blocks": 168,
        "N_matched_valid": 336,
        "N_swapped_valid": 336,
        "matched_d_minus_p": {"2": 168, "3": 168},
        "swapped_d_minus_p": {"2": 168, "3": 168},
        "matched_corr_available_minmax": [23, 28],
        "matched_ctrl_available_minmax": [20, 27],
        "swapped_corr_available_minmax": [23, 28],
        "swapped_ctrl_available_minmax": [20, 27],
        "prefix_marginal_preserved": True,
        "correction_marginal_preserved": True,
        "control_marginal_preserved": True,
    }
    require(summary == expected, "K3T_FROZEN_TOKEN_FEASIBILITY_MISMATCH")
    return contracts, summary


def resolve_state_blind(root: Path, hf_revision: str) -> dict[str, Any]:
    require(hf_revision == HF_REVISION, "HF_REVISION_MISMATCH")
    population = materialize_population(root)
    k2s = _load_k2s_helper()
    snapshot, hf = k2s.resolve_hf_snapshot(hf_revision)
    require(hf["hf_model_id"] == HF_MODEL, "HF_MODEL_MISMATCH")
    require(hf["resolved_hf_revision"] == HF_REVISION, "HF_RESOLVED_REVISION_MISMATCH")
    require(hf["transformers_version"] == TRANSFORMERS_VERSION, "TRANSFORMERS_VERSION_MISMATCH")
    require(hf["tokenizer_class"] == TOKENIZER_CLASS, "TOKENIZER_CLASS_MISMATCH")
    require(bool(hf["tokenizer_is_fast"]), "TOKENIZER_NOT_FAST")
    contracts, feasibility = build_input_contracts(population["candidates"], hf["tokenizer"])
    return {
        **population,
        "contracts": contracts,
        "feasibility": feasibility,
        "snapshot": snapshot,
        "hf": hf,
    }


def _validate_encoder_fp(fp: Mapping[str, Any]) -> None:
    require(fp["canonical_digest"] == ENCODER_CANONICAL_SHA256, "ENCODER_CANONICAL_MISMATCH")
    require(fp["raw_concat_digest"] == ENCODER_RAW_SHA256, "ENCODER_RAW_MISMATCH")
    require(fp["tensor_count"] == ENCODER_TENSOR_COUNT, "ENCODER_TENSOR_COUNT_MISMATCH")
    require(fp["total_numel"] == ENCODER_NUMEL, "ENCODER_NUMEL_MISMATCH")
    require(fp["total_raw_bytes"] == ENCODER_RAW_BYTES, "ENCODER_RAW_BYTES_MISMATCH")
    require(fp["dtypes"] == ["torch.float32"], "ENCODER_DTYPE_MISMATCH")


def run_synthetic_preflight(
    root: Path,
    seed180_handoff: Path,
    hf_revision: str,
    implementation_preflight: bool = True,
) -> dict[str, Any]:
    """Synthetic-only model validation. Never feeds K3T population text to the model."""
    runtime = git_provenance(root, implementation_preflight=implementation_preflight)
    require(hf_revision == HF_REVISION, "HF_REVISION_MISMATCH")

    k2s = _load_k2s_helper()
    snapshot, hf = k2s.resolve_hf_snapshot(hf_revision)
    require(hf["transformers_version"] == TRANSFORMERS_VERSION, "TRANSFORMERS_VERSION_MISMATCH")
    require(hf["tokenizer_class"] == TOKENIZER_CLASS, "TOKENIZER_CLASS_MISMATCH")

    handoff = k2s.audit_handoff(seed180_handoff)
    require(handoff["zip_sha256"] == EXPECTED_ZIP_SHA256, "HANDOFF_ZIP_SHA_MISMATCH")
    require(handoff["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "CHECKPOINT_SHA_MISMATCH")

    checkpoint = k2s.load_authenticated_checkpoint(handoff)
    fp = k2s.encoder_fingerprint(checkpoint["model_state_dict"])
    _validate_encoder_fp(fp)

    model = k2s.build_a0_model(root, snapshot, checkpoint)
    del checkpoint
    gc.collect()

    binding = k2s.resolve_capture_binding()
    require(binding.source_sha256 == RECURRENCE_SOURCE_SHA256, "RECURRENCE_SOURCE_SHA_MISMATCH")
    require(binding.qualname == "MambaMixer.slow_forward", "RECURRENCE_QUALNAME_MISMATCH")

    preflight = k2s.run_instrumentation_preflight(model, hf["tokenizer"], binding)
    require(preflight["status"] == "PASS_INSTRUMENTATION_PREFLIGHT", "SYNTHETIC_PREFLIGHT_FAILURE")
    require(preflight["trace_logit_noninterference"] == "PASS_EXACT", "TRACE_NONINTERFERENCE_FAILURE")
    require(preflight["fresh_state_isolation"] == "PASS_EXACT", "FRESH_STATE_FAILURE")
    require(
        preflight["causal_prefix_state_identity"]
        == "PASS_EXACT_ALL_PREFIX_TOKENS_ALL_LAYERS",
        "SYNTHETIC_PREFIX_IDENTITY_FAILURE",
    )
    require(preflight["device"] == "cpu", "SYNTHETIC_DEVICE_MISMATCH")

    return {
        "runtime": runtime,
        "handoff": handoff,
        "encoder": fp,
        "hf": {k: v for k, v in hf.items() if k not in {"config", "tokenizer"}},
        "synthetic_preflight": preflight,
        "scientific_population_model_forward": False,
        "scientific_population_recurrent_state_read": False,
    }


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    require(type(positive) is int and type(negative) is int, "SIGN_COUNT_TYPE")
    require(positive >= 0 and negative >= 0, "SIGN_COUNT_NEGATIVE")
    n = positive + negative
    if n == 0:
        return 1.0
    tail = min(positive, negative)
    numerator = sum(math.comb(n, k) for k in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2 ** n))


def primary_statistics(block_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(block_rows) == N_BLOCKS, "PRIMARY_BLOCK_COUNT_MISMATCH")
    values: list[float] = []
    for row in block_rows:
        require("B_R" in row, "PRIMARY_B_R_MISSING")
        value = row["B_R"]
        require(value is not None, "PRIMARY_R_UNDEFINED_INTEGRITY_FAILURE")
        value = float(value)
        require(math.isfinite(value), "PRIMARY_R_NONFINITE")
        values.append(value)

    positive = sum(v > 0 for v in values)
    negative = sum(v < 0 for v in values)
    zero = sum(v == 0 for v in values)
    n_eff = positive + negative
    floor_pass = n_eff >= 30
    raw_p = exact_two_sided_sign_p(positive, negative) if floor_pass else 1.0
    effect = None if n_eff == 0 else (positive - negative) / n_eff

    if floor_pass and raw_p <= 0.05 and effect is not None and effect > 0:
        verdict = POSITIVE_VERDICT
    elif floor_pass and raw_p <= 0.05 and effect is not None and effect < 0:
        verdict = NEGATIVE_VERDICT
    else:
        verdict = NULL_VERDICT

    return {
        "schema_version": PRIMARY_SCHEMA,
        "endpoint": "R",
        "expected_direction": "positive",
        "n_valid": N_BLOCKS,
        "n_eff": n_eff,
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "undefined_count": 0,
        "promotion_floor_pass": floor_pass,
        "raw_p": raw_p,
        "adjusted_p": raw_p,
        "family_m": 1,
        "alpha": 0.05,
        "rank_biserial_sign_effect": effect,
        "scientific_verdict": verdict,
    }


def block_rows_from_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(items) == N_ITEMS, "ITEM_COUNT_FOR_BLOCKS_MISMATCH")
    rows: list[dict[str, Any]] = []
    for block_index in range(N_BLOCKS):
        a = items[2 * block_index]
        b = items[2 * block_index + 1]
        require(
            int(a["block_index"]) == block_index and int(b["block_index"]) == block_index,
            "BLOCK_ITEM_ALIGNMENT_FAILURE",
        )
        xa = a["X_R"]
        xb = b["X_R"]
        require(xa is not None and xb is not None, "ITEM_X_R_UNDEFINED")
        value = (float(xa) + float(xb)) / 2.0
        require(math.isfinite(value), "BLOCK_R_NONFINITE")
        rows.append({
            "schema_version": BLOCK_SCHEMA,
            "block_index": block_index,
            "block_id": f"k3t-block-{block_index:03d}",
            "item_a_stable_id": str(a["stable_item_id"]),
            "item_b_stable_id": str(b["stable_item_id"]),
            "B_R": value,
        })
    return rows


def parse_authority_markers(text: str) -> dict[str, str]:
    expected_keys = {
        "K3T_EXECUTION_AUTHORITY_SCHEMA",
        "K3T_SCIENTIFIC_RECURRENT_STATE_EXECUTION_AUTHORIZED",
        "K3T_ONE_SCIENTIFIC_EXECUTION",
        "K3T_EXECUTION_IMPLEMENTATION_COMMIT",
        "K3T_EXECUTION_RUNNER_SHA256",
        "K3T_EXECUTION_TEST_SHA256",
        "K3T_PREREG_COMMIT",
        "K3T_PREREG_SHA256",
    }
    found: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        key = left.strip().strip("`")
        if key not in expected_keys:
            continue
        require(key not in found, "AUTHORITY_DUPLICATE_MARKER:" + key)
        found[key] = right.strip().strip("`").strip()
    require(set(found) == expected_keys, "AUTHORITY_MARKER_SET_MISMATCH")
    return found


def require_scientific_authority(root: Path) -> dict[str, Any]:
    """Fail closed before any scientific-population model forward."""
    _require_prereg_and_dependencies(root)

    authority_path = root / AUTHORITY_REL
    require(authority_path.is_file(), "K3T_EXECUTION_AUTHORITY_MISSING")
    raw = authority_path.read_bytes()
    require(not raw.startswith(b"\xef\xbb\xbf"), "AUTHORITY_BOM_FORBIDDEN")
    require(b"\r" not in raw, "AUTHORITY_CR_FORBIDDEN")
    text = raw.decode("utf-8", "strict")
    markers = parse_authority_markers(text)

    require(markers["K3T_EXECUTION_AUTHORITY_SCHEMA"] == AUTHORITY_SCHEMA, "AUTHORITY_SCHEMA_MISMATCH")
    require(
        markers["K3T_SCIENTIFIC_RECURRENT_STATE_EXECUTION_AUTHORIZED"] == "YES",
        "AUTHORITY_EXECUTION_NOT_AUTHORIZED",
    )
    require(markers["K3T_ONE_SCIENTIFIC_EXECUTION"] == "YES", "AUTHORITY_ONE_RUN_MISSING")
    require(markers["K3T_PREREG_COMMIT"] == K3T_PREREG_COMMIT, "AUTHORITY_PREREG_COMMIT_MISMATCH")
    require(markers["K3T_PREREG_SHA256"] == K3T_PREREG_SHA256, "AUTHORITY_PREREG_SHA_MISMATCH")

    head = _git(root, "rev-parse", "HEAD")
    parent = _git(root, "rev-parse", "HEAD^")
    addition_commit = _git(root, "log", "--diff-filter=A", "--format=%H", "--", AUTHORITY_REL)
    require(addition_commit == head, "AUTHORITY_ADDITION_COMMIT_AMBIGUOUS:" + AUTHORITY_REL)

    changed = _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    require(changed == [AUTHORITY_REL], "AUTHORITY_COMMIT_SCOPE_MISMATCH")

    implementation_commit = markers["K3T_EXECUTION_IMPLEMENTATION_COMMIT"]
    require(parent == implementation_commit, "AUTHORITY_PARENT_IMPLEMENTATION_MISMATCH")

    script_rel = "scripts/longterm_k3t_phase_balanced_r_transportability.py"
    test_rel = "tests/test_longterm_k3t_phase_balanced_r_transportability.py"
    script_sha = file_sha256(root / script_rel)
    test_sha = file_sha256(root / test_rel)
    require(
        markers["K3T_EXECUTION_RUNNER_SHA256"] == script_sha,
        "AUTHORITY_RUNNER_SHA_MISMATCH",
    )
    require(
        markers["K3T_EXECUTION_TEST_SHA256"] == test_sha,
        "AUTHORITY_TEST_SHA_MISMATCH",
    )

    status = _status_lines(root)
    for line in status:
        path = _status_path(line)
        require(
            line[:2] == "??" and path in HISTORICAL_K1_UNTRACKED,
            "SCIENTIFIC_RUNTIME_DIRTY_CONTRACT_MISMATCH:" + line,
        )

    return {
        "runtime_git_head": head,
        "implementation_commit": implementation_commit,
        "authority_sha256": sha256_bytes(raw),
        "authority_markers": markers,
        "runtime_dirty_contract": status,
        "script_sha256": script_sha,
        "test_sha256": test_sha,
    }


def _output_dir(root: Path, supplied: Path) -> Path:
    target = supplied.resolve()
    repo = root.resolve()
    require(target != repo and repo not in target.parents, "OUTPUT_DIRECTORY_MUST_BE_OUTSIDE_REPO")
    require(not target.exists(), "OUTPUT_DIRECTORY_ALREADY_EXISTS")
    expected_prefix = "k3t-r-transportability-"
    require(target.name.startswith(expected_prefix), "OUTPUT_DIRECTORY_BASENAME_INVALID")
    return target


def _branch_reference_metrics(branch: Mapping[str, Any]) -> dict[str, Any]:
    """Reference diagnostics are serialized but never enter K3T inference."""
    return {
        "R_mean_speed": branch["R_mean_speed"],
        "D_mean_turn": branch["D_mean_turn"],
        "P_efficiency": branch["P_efficiency"],
        "path_length": branch["path_length"],
        "displacement": branch["displacement"],
        "state_window_sha256": branch["state_window_sha256"],
        "state_p_minus_1_sha256": branch["state_p_minus_1_sha256"],
        "state_p_sha256": branch["state_p_sha256"],
        "prefix_state_sequence_sha256": branch["prefix_state_sequence_sha256"],
        "prefix_state_token_count": branch["prefix_state_token_count"],
    }


def execute_scientific(
    root: Path,
    seed180_handoff: Path,
    hf_revision: str,
    output_dir: Path,
) -> dict[str, Any]:
    """One scientific execution. Caller must not retry after state read begins."""
    authority = require_scientific_authority(root)
    out = _output_dir(root, output_dir)

    # State-blind population/tokenizer reconstruction happens only after
    # authority has passed.  No model forward has occurred yet.
    state_blind = resolve_state_blind(root, hf_revision)
    k2s = _load_k2s_helper()

    # Synthetic non-study model validation.
    handoff = k2s.audit_handoff(seed180_handoff)
    require(handoff["zip_sha256"] == EXPECTED_ZIP_SHA256, "HANDOFF_ZIP_SHA_MISMATCH")
    checkpoint = k2s.load_authenticated_checkpoint(handoff)
    fp = k2s.encoder_fingerprint(checkpoint["model_state_dict"])
    _validate_encoder_fp(fp)
    model = k2s.build_a0_model(root, state_blind["snapshot"], checkpoint)
    del checkpoint
    gc.collect()

    binding = k2s.resolve_capture_binding()
    require(binding.source_sha256 == RECURRENCE_SOURCE_SHA256, "RECURRENCE_SOURCE_SHA_MISMATCH")
    preflight = k2s.run_instrumentation_preflight(model, state_blind["hf"]["tokenizer"], binding)
    require(preflight["status"] == "PASS_INSTRUMENTATION_PREFLIGHT", "SYNTHETIC_PREFLIGHT_FAILURE")

    layer_map = k2s.registered_mamba_layers(model)
    item_rows: list[dict[str, Any]] = []
    item_primary: list[dict[str, Any]] = []
    prefix_identity_count = 0

    # This loop is the scientific-population recurrent-state boundary.
    for i, contract in enumerate(state_blind["contracts"]):
        branch_results: dict[str, dict[int, dict[str, Any]]] = {}
        for branch_name in ("matched_corr", "matched_ctrl", "swapped_corr", "swapped_ctrl"):
            branch_results[branch_name] = k2s.run_native_branch(
                model,
                contract["branch_token_ids"][branch_name],
                int(contract["p"]),
                binding,
                layer_map,
            )

        branches = {
            name: branch_results[name][PRIMARY_LAYER]
            for name in branch_results
        }
        recipient = k2s.recipient_layer_result(branches)

        x_r = recipient["X_pair_specificity"]["R"]
        require(x_r is not None and math.isfinite(float(x_r)), "SCIENTIFIC_X_R_INVALID")
        prefix_identity_count += 1

        item_row = {
            "schema_version": ITEM_SCHEMA,
            "item_index": i,
            "block_index": int(contract["block_index"]),
            "block_id": str(contract["block_id"]),
            "stable_item_id": str(contract["stable_item_id"]),
            "donor_index": int(contract["donor_index"]),
            "donor_stable_item_id": str(contract["donor_stable_item_id"]),
            "pair_id": str(contract["pair_id"]),
            "base_claim_sha256": str(contract["base_claim_sha256"]),
            "global_template_index": int(contract["global_template_index"]),
            "generator_phase_class": int(contract["generator_phase_class"]),
            "cycle_in_slice": int(contract["cycle_in_slice"]),
            "correction_source_intervention": str(contract["correction_source_intervention"]),
            "p": int(contract["p"]),
            "matched_d_minus_p": int(contract["matched_d_minus_p"]),
            "swapped_d_minus_p": int(contract["swapped_d_minus_p"]),
            "primary_layer": PRIMARY_LAYER,
            "signed_contrasts_R": {
                "matched": recipient["signed_contrasts"]["R_matched"],
                "swapped": recipient["signed_contrasts"]["R_swapped"],
            },
            "X_R": float(x_r),
            "common_state_p_sha256": recipient["common_state_p_sha256"],
            "common_state_p_minus_1_sha256": recipient["common_state_p_minus_1_sha256"],
            "common_prefix_state_sequence_sha256": recipient["common_prefix_state_sequence_sha256"],
            "reference_diagnostics_only": {
                name: _branch_reference_metrics(branches[name])
                for name in sorted(branches)
            },
        }
        item_rows.append(item_row)
        item_primary.append({
            "block_index": int(contract["block_index"]),
            "stable_item_id": str(contract["stable_item_id"]),
            "X_R": float(x_r),
        })

        del branch_results, branches, recipient
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == N_ITEMS:
            print(f"K3T_PROGRESS items={i + 1}/{N_ITEMS}", flush=True)
        if (i + 1) % 24 == 0:
            gc.collect()

    require(prefix_identity_count == N_ITEMS, "SCIENTIFIC_PREFIX_IDENTITY_COUNT_MISMATCH")

    blocks = block_rows_from_items(item_primary)
    primary = primary_statistics(blocks)

    integrity = {
        "schema_version": INTEGRITY_SCHEMA,
        "candidate_count": N_ITEMS,
        "block_count": N_BLOCKS,
        "phase_class_count": PHASE_PERIOD,
        "phase_count_each": 2,
        "phase_source_balance": "PASS_EXACT",
        "prior_overlap": state_blind["prior_overlap"],
        "tokenizer_feasibility": state_blind["feasibility"],
        "synthetic_instrumentation_preflight": preflight,
        "scientific_prefix_state_identity_item_count": prefix_identity_count,
        "scientific_layer": PRIMARY_LAYER,
        "scientific_device": "cpu",
        "state_timing": "post_consumption_s_t",
        "reference_diagnostics_inference_authorized": False,
    }

    out.mkdir(parents=True, exist_ok=False)

    report = (
        "# K3T Phase-Balanced R Transportability Result\n\n"
        f"Scientific verdict: `{primary['scientific_verdict']}`\n\n"
        f"Runtime authority HEAD: `{authority['runtime_git_head']}`\n\n"
        "Primary inference uses only layer-23 R over 168 reciprocal blocks, "
        "with a two-sided exact sign test and family size m=1.\n\n"
        "D, displacement, P, and path length are reference diagnostics only "
        "and cannot modify the K3T verdict.\n\n"
        "A positive result does not erase the frozen K3C R non-replication "
        "and does not by itself establish unrestricted cross-claim transportability.\n"
    ).encode("utf-8")

    blobs: dict[str, bytes] = {
        "generated_source.jsonl": state_blind["source_raw"],
        "candidate_pool.jsonl": state_blind["candidate_raw"],
        "reciprocal_mapping.json": canonical_json_line(state_blind["mapping"]),
        "item_metrics.jsonl": canonical_jsonl(item_rows),
        "block_metrics.jsonl": canonical_jsonl(blocks),
        "primary_stats.json": canonical_json_line(primary),
        "integrity.json": canonical_json_line(integrity),
        "report.md": report,
    }
    for name, raw in blobs.items():
        (out / name).write_bytes(raw)

    artifact_sha = {name: sha256_bytes(raw) for name, raw in blobs.items()}

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "authority": authority,
        "prereg": {
            "commit": K3T_PREREG_COMMIT,
            "path": K3T_PREREG_REL,
            "sha256": K3T_PREREG_SHA256,
        },
        "generator": {
            "path": GENERATOR_REL,
            "sha256": GENERATOR_SHA256,
            "git_blob": GENERATOR_GIT_BLOB,
        },
        "k2s_helper": {
            "path": K2S_HELPER_REL,
            "sha256": K2S_HELPER_SHA256,
            "git_blob": K2S_HELPER_GIT_BLOB,
        },
        "population": {
            "global_template_start": GLOBAL_TEMPLATE_START,
            "global_template_stop": GLOBAL_TEMPLATE_STOP,
            "first_pair_id": FIRST_PAIR_ID,
            "last_pair_id": LAST_PAIR_ID,
            "n_items": N_ITEMS,
            "n_blocks": N_BLOCKS,
            "phase_period": PHASE_PERIOD,
            "phase_count_each": 2,
            "phase_source_balance": "PASS_EXACT",
            "generated_source_sha256": GENERATED_SOURCE_SHA256,
            "candidate_pool_sha256": CANDIDATE_POOL_SHA256,
            "reciprocal_mapping_canonical_sha256": RECIPROCAL_MAPPING_SHA256,
            "prior_overlap": state_blind["prior_overlap"],
            "feasibility": state_blind["feasibility"],
        },
        "hf": {
            key: value
            for key, value in state_blind["hf"].items()
            if key not in {"config", "tokenizer"}
        },
        "handoff": handoff,
        "encoder_fingerprint": fp,
        "instrumentation_preflight": preflight,
        "measurement": {
            "primary_layer": PRIMARY_LAYER,
            "window_W": W,
            "primary_endpoint": "R",
            "norm": "raw_frobenius",
            "recipient_pair_specificity": "abs_delta_matched_minus_abs_delta_swapped",
            "block_statistic": "mean_two_reciprocal_X_R",
            "test": "two_sided_exact_sign",
            "family_m": 1,
            "alpha": 0.05,
            "reference_diagnostics_only": ["D", "displacement", "P", "path_length"],
        },
        "primary_statistics": primary,
        "scientific_verdict": primary["scientific_verdict"],
        "artifact_sha256": artifact_sha,
    }
    manifest_raw = canonical_json_line(manifest)
    (out / "manifest.json").write_bytes(manifest_raw)

    all_hashes = {**artifact_sha, "manifest.json": sha256_bytes(manifest_raw)}
    sums = "".join(f"{all_hashes[name]}  {name}\n" for name in sorted(all_hashes)).encode("utf-8")
    (out / "SHA256SUMS.txt").write_bytes(sums)

    return {
        "scientific_execution": "PASS",
        "scientific_verdict": primary["scientific_verdict"],
        "primary_statistics": primary,
        "output_dir": str(out),
        "artifact_sha256": {
            **all_hashes,
            "SHA256SUMS.txt": sha256_bytes(sums),
        },
    }


def _state_blind_summary(root: Path, hf_revision: str) -> dict[str, Any]:
    runtime = git_provenance(root, implementation_preflight=True)
    result = resolve_state_blind(root, hf_revision)
    return {
        "runtime": runtime,
        "execution_boundary": {
            "k3t_design_frozen": True,
            "k3t_implementation_validation_authorized": True,
            "k3t_scientific_recurrent_state_execution_authorized": False,
            "scientific_population_model_forward": False,
            "scientific_population_recurrent_state_read": False,
        },
        "population": {
            "global_template_start": GLOBAL_TEMPLATE_START,
            "global_template_stop": GLOBAL_TEMPLATE_STOP,
            "first_pair_id": FIRST_PAIR_ID,
            "last_pair_id": LAST_PAIR_ID,
            "N_items": N_ITEMS,
            "N_blocks": N_BLOCKS,
            "phase_period": PHASE_PERIOD,
            "phase_class_count": result["phase_class_count"],
            "phase_count_unique_values": result["phase_count_unique_values"],
            "correction_source_counts": result["correction_source_counts"],
            "generated_source_rows": len(result["source_rows"]),
            "generated_source_sha256": sha256_bytes(result["source_raw"]),
            "candidate_pool_sha256": sha256_bytes(result["candidate_raw"]),
            "reciprocal_mapping_sha256": result["mapping_sha256"],
            "prior_overlap": result["prior_overlap"],
            "feasibility": result["feasibility"],
        },
        "hf": {
            key: value
            for key, value in result["hf"].items()
            if key not in {"config", "tokenizer"}
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="K3T phase-balanced R transportability bounded runner"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--state-blind-preflight", action="store_true")
    mode.add_argument("--synthetic-preflight", action="store_true")
    mode.add_argument("--scientific", action="store_true")
    parser.add_argument("--seed180-handoff")
    parser.add_argument("--hf-revision", default=HF_REVISION)
    parser.add_argument("--output-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]

    if args.state_blind_preflight:
        result = _state_blind_summary(root, args.hf_revision)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0

    if args.synthetic_preflight:
        require(bool(args.seed180_handoff), "SEED180_HANDOFF_REQUIRED")
        result = run_synthetic_preflight(
            root,
            Path(args.seed180_handoff),
            args.hf_revision,
            implementation_preflight=True,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0

    require(args.scientific, "MODE_UNREACHABLE")
    require(bool(args.seed180_handoff), "SEED180_HANDOFF_REQUIRED")
    require(bool(args.output_dir), "OUTPUT_DIR_REQUIRED")

    # Authority gate is intentionally first inside the scientific mode.
    # It must fail before state-blind materialization, model construction, or
    # scientific-population model forward if no authority exists.
    require_scientific_authority(root)

    result = execute_scientific(
        root,
        Path(args.seed180_handoff),
        args.hf_revision,
        Path(args.output_dir),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print("K3T_CONTRACT_FAILURE:" + str(exc), file=sys.stderr)
        raise SystemExit(2)
