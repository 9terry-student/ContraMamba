"""K3C native recurrence contribution decomposition.

Implementation authority: frozen K3C preregistration at commit
b85272d88d0bb57db45fdc963d313714529e7975.

This module implements the prospectively frozen K3C population contract,
contribution-level replay interventions, confirmatory statistics, and
synthetic/replay preflight. The only CLI action authorized in this phase is
--replay-preflight. Scientific recurrent-state execution on the K3C population
is intentionally unreachable from this implementation-phase CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

K3C_PREREG_AUTHORITY_COMMIT = "b85272d88d0bb57db45fdc963d313714529e7975"
K3C_PREREG_REL = (
    "reports/longterm_k3c_native_recurrence_contribution_decomposition_"
    "prereg_candidate.md"
)
K3C_PREREG_SHA256 = (
    "0a9de28237e107ce3a62d4fa9e0bb7f230d2e0289e019d3864edf491dd271436"
)

K3_SUCCESSOR_AUTHORITY_COMMIT = "a8101ce9baf340f35c0c6b135f7312681805693c"
K3_SUCCESSOR_REL = (
    "reports/longterm_k3_successor_write_injection_retained_carry_"
    "hypothesis_report_candidate.md"
)
K3_SUCCESSOR_SHA256 = (
    "0040d3d3573faad4dfacb0b617b4626b5e1143a9aa946bd4d270fdf80c44523e"
)

GENERATOR_REL = "scripts/build_controlled_v5.py"
GENERATOR_SHA256 = (
    "4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c"
)
GENERATOR_GIT_BLOB = "baee23a9f71333125f4a8735c2c92d20cab7eb4f"

K2S_RUNNER_REL = "scripts/longterm_k2s_pair_specific_event_dynamics.py"
K2S_RUNNER_SHA256 = (
    "f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8"
)
K2S_RUNNER_GIT_BLOB = "3a651fb508669bdcf72441a4869b863d6eee6c1f"

K3_REPLAY_REL = (
    "scripts/longterm_k3_selective_ssm_retention_write_"
    "causal_decomposition.py"
)
K3_REPLAY_SHA256 = (
    "116e9daae8a4d62b19d8ab4e7a4d4171df8ad7ae99b08953fc1f2ba49563a60e"
)
K3_REPLAY_GIT_BLOB = "24278583af0132f4ec5cf8610d2cb85972c1bd51"

K2W_POOL_REL = (
    "reports/longterm_k2w_fixed_window_phase_a_c7c7a0c218bb/"
    "candidate_pool.jsonl"
)
K2W_POOL_SHA256 = (
    "abf693d3267cc4e3dd27a8127d2948b36fdf8ba24e135a643215f0f31a26d808"
)
K2R_POOL_REL = (
    "reports/longterm_k2r_claim_disjoint_replication_52bd363_v1/"
    "candidate_pool.jsonl"
)
K2R_POOL_SHA256 = (
    "00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4"
)

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"
HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}
K3C_UNTRACKED = {
    "scripts/longterm_k3c_native_recurrence_contribution_decomposition.py",
    "tests/test_longterm_k3c_native_recurrence_contribution_decomposition.py",
}

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"
MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)

EXPECTED_ZIP_SHA256 = (
    "96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c"
)
COMMON_ENCODER_CANONICAL_SHA256 = (
    "48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"
)
COMMON_ENCODER_RAW_CONCAT_SHA256 = (
    "968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"
)

GLOBAL_TEMPLATE_START = 600
GLOBAL_TEMPLATE_STOP = 900
N_ITEMS = 300
N_BLOCKS = 150
GENERATED_SOURCE_ROWS = 3900
GENERATED_SOURCE_CANONICAL_SHA256 = (
    "33bff5a0b657d1ceb38ae9c651e1cadfc8308286398cc1b8c4245c47f1c42000"
)
CANDIDATE_SCHEMA = "k3c-independent-population-v1"
STABLE_ID_PREFIX = "k3c-v1:"
CANDIDATE_POOL_CANONICAL_SHA256 = (
    "9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e"
)
RECIPROCAL_MAPPING_CANONICAL_SHA256 = (
    "4fbc0f6642db3b2c3cca148fdc03cdc738dd6e8718cffb3fcdee73cd5b7f9acc"
)

PRIMARY_LAYER = 23
N_LAYERS = 24
W = 8
EPSILON = 1e-12
METRICS = ("R", "D", "DISP", "P")
EXPECTED_DIRECTION = {"R": 1, "D": 1, "DISP": -1, "P": -1}
BASE_ORDER = ("R", "D", "DISP", "P")
PRIMARY_TEST_ORDER = (
    "R_DOM", "R_CARRY",
    "D_DOM", "D_CARRY",
    "DISP_DOM", "DISP_CARRY",
    "P_DOM", "P_CARRY",
)

SUCCESS_VERDICT = "LAYER23_WRITE_INJECTION_WITH_RETAINED_CARRY_CAUSALLY_SUPPORTED"
CONTRADICTION_VERDICT = "LAYER23_WRITE_INJECTION_WITH_RETAINED_CARRY_CONTRADICTED"
NOT_ESTABLISHED_VERDICT = "LAYER23_WRITE_INJECTION_WITH_RETAINED_CARRY_NOT_ESTABLISHED"
BASE_FAILURE_VERDICT = "INCONCLUSIVE_DUE_TO_BASELINE_REPLICATION_FAILURE"


class ContractError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json(dict(row)) + b"\n" for row in rows)


def parse_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    require(not raw.startswith(b"\xef\xbb\xbf"), "JSONL_BOM_FORBIDDEN")
    require(b"\r" not in raw, "JSONL_CR_FORBIDDEN")
    require(raw.endswith(b"\n"), "JSONL_FINAL_LF_REQUIRED")
    lines = raw[:-1].split(b"\n")
    require(bool(lines) and all(lines), "JSONL_BLANK_LINE_FORBIDDEN")
    out: list[dict[str, Any]] = []
    for line in lines:
        value = json.loads(line.decode("utf-8", "strict"))
        require(isinstance(value, dict), "JSONL_ROW_NOT_OBJECT")
        out.append(value)
    return out


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"GIT_FAILURE:{' '.join(args)}") from exc


def git_provenance(root: Path, replay_preflight: bool) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")
    ancestor = subprocess.call(
        [
            "git", "merge-base", "--is-ancestor",
            K3C_PREREG_AUTHORITY_COMMIT, head,
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "K3C_PREREG_NOT_ANCESTOR")

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
    ).splitlines()
    allowed = set(HISTORICAL_K1_UNTRACKED)
    if replay_preflight:
        allowed |= K3C_UNTRACKED
    for line in status:
        path = line[3:].replace("\\", "/") if len(line) >= 4 else ""
        require(
            line[:2] == "??" and path in allowed,
            "GIT_DIRTY_CONTRACT_MISMATCH",
        )

    exact_files = {
        K3C_PREREG_REL: K3C_PREREG_SHA256,
        K3_SUCCESSOR_REL: K3_SUCCESSOR_SHA256,
        GENERATOR_REL: GENERATOR_SHA256,
        K2S_RUNNER_REL: K2S_RUNNER_SHA256,
        K3_REPLAY_REL: K3_REPLAY_SHA256,
        K2W_POOL_REL: K2W_POOL_SHA256,
        K2R_POOL_REL: K2R_POOL_SHA256,
    }
    for rel, expected in exact_files.items():
        path = root / rel
        require(path.is_file(), f"REQUIRED_FILE_MISSING:{rel}")
        require(
            file_sha256(path) == expected,
            f"REQUIRED_FILE_SHA_MISMATCH:{rel}",
        )

    require(
        _git(root, "rev-parse", f"{head}:{GENERATOR_REL}") == GENERATOR_GIT_BLOB,
        "GENERATOR_RUNTIME_BLOB_DRIFT",
    )
    require(
        _git(root, "rev-parse", f"{head}:{K2S_RUNNER_REL}") == K2S_RUNNER_GIT_BLOB,
        "K2S_RUNTIME_BLOB_DRIFT",
    )
    require(
        _git(root, "rev-parse", f"{head}:{K3_REPLAY_REL}") == K3_REPLAY_GIT_BLOB,
        "K3_REPLAY_RUNTIME_BLOB_DRIFT",
    )

    return {
        "runtime_branch": branch,
        "runtime_git_head": head,
        "runtime_dirty_contract": status,
        "k3c_prereg_sha256": K3C_PREREG_SHA256,
        "k3_successor_sha256": K3_SUCCESSOR_SHA256,
        "generator_sha256": GENERATOR_SHA256,
        "generator_git_blob": GENERATOR_GIT_BLOB,
        "k2s_runner_sha256": K2S_RUNNER_SHA256,
        "k2s_runner_git_blob": K2S_RUNNER_GIT_BLOB,
        "k3_replay_sha256": K3_REPLAY_SHA256,
        "k3_replay_git_blob": K3_REPLAY_GIT_BLOB,
    }


def load_frozen_dependencies(root: Path) -> tuple[Any, Any, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    generator = importlib.import_module("scripts.build_controlled_v5")
    k2s = importlib.import_module(
        "scripts.longterm_k2s_pair_specific_event_dynamics"
    )
    k3 = importlib.import_module(
        "scripts.longterm_k3_selective_ssm_retention_write_"
        "causal_decomposition"
    )

    bindings = (
        (generator, GENERATOR_REL, GENERATOR_SHA256),
        (k2s, K2S_RUNNER_REL, K2S_RUNNER_SHA256),
        (k3, K3_REPLAY_REL, K3_REPLAY_SHA256),
    )
    for module, rel, expected in bindings:
        require(
            Path(module.__file__).resolve() == (root / rel).resolve(),
            f"IMPORT_PATH_MISMATCH:{rel}",
        )
        require(
            file_sha256(Path(module.__file__)) == expected,
            f"IMPORT_BYTE_MISMATCH:{rel}",
        )

    require(k2s.PRIMARY_LAYER == PRIMARY_LAYER, "K2S_LAYER_DRIFT")
    require(k2s.W == W, "K2S_WINDOW_DRIFT")
    require(k3.PRIMARY_LAYER == PRIMARY_LAYER, "K3_LAYER_DRIFT")
    require(k3.W == W, "K3_WINDOW_DRIFT")
    require(k3.MAMBA_SOURCE_SHA256 == MAMBA_SOURCE_SHA256, "K3_SOURCE_DRIFT")
    return generator, k2s, k3


def _claim_text_from_prefix(prefix: str) -> str:
    marker = "\nEvidence:"
    require(marker in prefix, "PREFIX_CLAIM_MARKER_MISSING")
    return prefix.split(marker, 1)[0]


def materialize_population(
    root: Path,
    generator: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    templates_all = generator.fact_templates_for_count(GLOBAL_TEMPLATE_STOP)
    templates = [
        dict(row)
        for row in templates_all[GLOBAL_TEMPLATE_START:GLOBAL_TEMPLATE_STOP]
    ]
    require(len(templates) == N_ITEMS, "K3C_TEMPLATE_COUNT")

    expected_pair_ids = [
        f"generated_fact_{i:03d}"
        for i in range(601, 901)
    ]
    pair_ids = [str(row.get("pair_id", "")) for row in templates]
    require(pair_ids == expected_pair_ids, "K3C_PAIR_ID_SEQUENCE")
    require(len(set(pair_ids)) == N_ITEMS, "K3C_PAIR_ID_DUPLICATE")

    source_rows = generator._build_records(templates)
    require(len(source_rows) == GENERATED_SOURCE_ROWS, "K3C_SOURCE_ROW_COUNT")
    source_bytes = canonical_jsonl(source_rows)
    require(
        sha256_bytes(source_bytes) == GENERATED_SOURCE_CANONICAL_SHA256,
        "K3C_GENERATED_SOURCE_SHA_MISMATCH",
    )

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in source_rows:
        groups.setdefault(str(row.get("pair_id", "")), []).append(row)
    require(set(groups) == set(pair_ids), "K3C_GROUP_SET")

    candidates: list[dict[str, Any]] = []
    correction_sources: Counter[str] = Counter()

    for pair_id in sorted(groups):
        rows = groups[pair_id]
        trunc = [
            r for r in rows
            if r.get("intervention_type") == "evidence_truncation"
        ]
        entity = [
            r for r in rows
            if r.get("intervention_type") == "entity_swap"
        ]
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
        require(
            len(trunc) == 1 and len(entity) == 1,
            "K3C_TRUNC_OR_CONTROL_MULTIPLICITY",
        )

        if len(refute_polarity) == 1:
            correction = refute_polarity[0]
            correction_source = "polarity_flip"
        elif len(refute_none) == 1:
            correction = refute_none[0]
            correction_source = "none"
        else:
            raise ContractError("K3C_CORRECTION_MULTIPLICITY")

        t, e, q = trunc[0], entity[0], correction
        require(
            t.get("claim") == e.get("claim") == q.get("claim"),
            "K3C_CLAIM_IDENTITY",
        )
        require(
            t.get("final_label") == "NOT_ENTITLED"
            and t.get("primary_failure_type") == "sufficiency"
            and t.get("sufficiency_label") == 0,
            "K3C_TRUNCATION_CONTRACT",
        )
        require(
            e.get("final_label") == "NOT_ENTITLED"
            and e.get("primary_failure_type") == "frame"
            and e.get("polarity_label") == "NONE",
            "K3C_CONTROL_CONTRACT",
        )
        require(
            q.get("final_label") == "REFUTE"
            and q.get("polarity_label") == "REFUTE",
            "K3C_CORRECTION_CONTRACT",
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
        candidates.append(
            {
                **recipe,
                "stable_item_id":
                    STABLE_ID_PREFIX + sha256_bytes(canonical_json(recipe)),
                "base_claim_sha256":
                    sha256_bytes(str(t["claim"]).encode("utf-8")),
            }
        )
        correction_sources[correction_source] += 1

    candidates.sort(key=lambda row: row["stable_item_id"])
    require(len(candidates) == N_ITEMS, "K3C_CANDIDATE_COUNT")
    require(
        len({r["stable_item_id"] for r in candidates}) == N_ITEMS,
        "K3C_STABLE_ID_DUPLICATE",
    )
    require(
        len({r["base_claim_sha256"] for r in candidates}) == N_ITEMS,
        "K3C_CLAIM_SHA_DUPLICATE",
    )
    require(
        correction_sources == Counter({"none": 150, "polarity_flip": 150}),
        "K3C_CORRECTION_SOURCE_BALANCE",
    )

    candidate_bytes = canonical_jsonl(candidates)
    require(
        sha256_bytes(candidate_bytes) == CANDIDATE_POOL_CANONICAL_SHA256,
        "K3C_CANDIDATE_SHA_MISMATCH",
    )

    new_sets = {
        "pair_id": {str(r["pair_id"]) for r in candidates},
        "claim_sha": {str(r["base_claim_sha256"]) for r in candidates},
        "claim_text": {
            _claim_text_from_prefix(str(r["prefix_text"]))
            for r in candidates
        },
    }
    overlap: dict[str, dict[str, int]] = {}
    for label, rel, expected_sha in (
        ("K2W", K2W_POOL_REL, K2W_POOL_SHA256),
        ("K2R_K3", K2R_POOL_REL, K2R_POOL_SHA256),
    ):
        raw = (root / rel).read_bytes()
        require(sha256_bytes(raw) == expected_sha, f"{label}_POOL_SHA_MISMATCH")
        old = parse_jsonl_bytes(raw)
        old_sets = {
            "pair_id": {str(r["pair_id"]) for r in old},
            "claim_sha": {str(r["base_claim_sha256"]) for r in old},
            "claim_text": {
                _claim_text_from_prefix(str(r["prefix_text"]))
                for r in old
            },
        }
        overlap[label] = {
            key: len(old_sets[key] & new_sets[key])
            for key in ("pair_id", "claim_sha", "claim_text")
        }
    require(
        all(v == 0 for group in overlap.values() for v in group.values()),
        "K3C_PRIOR_POPULATION_OVERLAP",
    )

    mapping = []
    for block_index in range(N_BLOCKS):
        a = 2 * block_index
        b = a + 1
        require((a ^ 1) == b and (b ^ 1) == a, "K3C_RECIPROCAL_INDEX")
        mapping.append(
            {
                "block_index": block_index,
                "item_a_stable_id": candidates[a]["stable_item_id"],
                "item_b_stable_id": candidates[b]["stable_item_id"],
                "item_a_donor_index": a ^ 1,
                "item_b_donor_index": b ^ 1,
            }
        )
    mapping_sha = sha256_bytes(canonical_json(mapping))
    require(
        mapping_sha == RECIPROCAL_MAPPING_CANONICAL_SHA256,
        "K3C_RECIPROCAL_MAPPING_SHA_MISMATCH",
    )

    audit = {
        "generator_sha256": GENERATOR_SHA256,
        "generator_git_blob": GENERATOR_GIT_BLOB,
        "global_template_start": GLOBAL_TEMPLATE_START,
        "global_template_stop": GLOBAL_TEMPLATE_STOP,
        "first_pair_id": expected_pair_ids[0],
        "last_pair_id": expected_pair_ids[-1],
        "generated_source_rows": len(source_rows),
        "generated_source_canonical_sha256": sha256_bytes(source_bytes),
        "candidate_pool_count": len(candidates),
        "candidate_pool_canonical_sha256": sha256_bytes(candidate_bytes),
        "reciprocal_block_count": N_BLOCKS,
        "reciprocal_mapping_canonical_sha256": mapping_sha,
        "correction_source_counts": dict(sorted(correction_sources.items())),
        "prior_overlap": overlap,
        "scientific_outcome_values_read": False,
        "scientific_population_recurrent_state_read": False,
    }
    return [dict(r) for r in source_rows], candidates, audit


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"] if isinstance(out, Mapping) else out.input_ids
    return list(ids)


def first_divergence_after_prefix(
    corr: Sequence[int],
    ctrl: Sequence[int],
    prefix_len: int,
) -> int:
    stop = min(len(corr), len(ctrl), prefix_len + W)
    value = next(
        (i for i in range(prefix_len, stop) if corr[i] != ctrl[i]),
        None,
    )
    require(value is not None, "K3C_PAIR_DIVERGENCE_MISSING")
    return int(value)


def build_input_contracts(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(getattr(tokenizer, "is_fast", False), "TOKENIZER_MUST_BE_FAST")
    require(len(rows) == N_ITEMS, "K3C_INPUT_CONTRACT_COUNT")
    contracts: list[dict[str, Any]] = []
    matched_d: Counter[str] = Counter()
    swapped_d: Counter[str] = Counter()
    availability: dict[str, list[int]] = {
        "matched_corr": [],
        "matched_ctrl": [],
        "swapped_corr": [],
        "swapped_ctrl": [],
    }

    for i, row in enumerate(rows):
        donor = rows[i ^ 1]
        prefix = str(row["prefix_text"])
        p_ids = _token_ids(tokenizer, prefix)
        require(len(p_ids) >= 2, "K3C_PREFIX_TOO_SHORT")
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
            require(ids[:len(p_ids)] == p_ids, "K3C_EXACT_PREFIX_FAILURE")
            continuation = len(ids) - len(p_ids)
            require(continuation >= W, "K3C_WINDOW8_FAILURE")
            arrays[name] = ids
            availability[name].append(continuation)

        md = first_divergence_after_prefix(
            arrays["matched_corr"],
            arrays["matched_ctrl"],
            len(p_ids),
        )
        sd = first_divergence_after_prefix(
            arrays["swapped_corr"],
            arrays["swapped_ctrl"],
            len(p_ids),
        )
        matched_d[str(md - p)] += 1
        swapped_d[str(sd - p)] += 1

        contracts.append(
            {
                "item_index": i,
                "block_index": i // 2,
                "stable_item_id": row["stable_item_id"],
                "donor_index": i ^ 1,
                "donor_stable_item_id": donor["stable_item_id"],
                "p": p,
                "matched_d_minus_p": md - p,
                "swapped_d_minus_p": sd - p,
            }
        )

    require(
        sorted(str(r["correction_text"]) for r in rows)
        == sorted(str(rows[i ^ 1]["correction_text"]) for i in range(N_ITEMS)),
        "K3C_CORRECTION_MARGINAL_NOT_PRESERVED",
    )
    require(
        sorted(str(r["control_text"]) for r in rows)
        == sorted(str(rows[i ^ 1]["control_text"]) for i in range(N_ITEMS)),
        "K3C_CONTROL_MARGINAL_NOT_PRESERVED",
    )

    summary = {
        "N_total": N_ITEMS,
        "N_blocks": N_BLOCKS,
        "N_matched_valid": N_ITEMS,
        "N_swapped_valid": N_ITEMS,
        "matched_d_minus_p":
            dict(sorted(matched_d.items(), key=lambda kv: int(kv[0]))),
        "swapped_d_minus_p":
            dict(sorted(swapped_d.items(), key=lambda kv: int(kv[0]))),
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
    require(summary == expected, "K3C_FROZEN_TOKEN_FEASIBILITY_MISMATCH")
    return contracts, summary


def arithmetic_midpoint(a: Any, b: Any) -> Any:
    return 0.5 * (a + b)


def retained_contribution(g: Any, state: Any) -> Any:
    return g * state


def structural_replay(
    initial_state: Any,
    G: Mapping[int, Any],
    W_terms: Mapping[int, Any],
    start: int,
    end: int,
) -> dict[int, Any]:
    require(start <= end, "REPLAY_RANGE_INVALID")
    require(set(range(start, end + 1)) <= set(G), "REPLAY_G_INCOMPLETE")
    require(
        set(range(start, end + 1)) <= set(W_terms),
        "REPLAY_W_INCOMPLETE",
    )
    current = initial_state.detach().cpu().contiguous().clone()
    out: dict[int, Any] = {}
    for t in range(start, end + 1):
        current = retained_contribution(G[t], current) + W_terms[t]
        out[t] = current.detach().cpu().contiguous().clone()
    return out


def replay_pair_contribution(
    corr: Mapping[str, Any],
    ctrl: Mapping[str, Any],
    start: int,
    end: int,
    condition: str,
) -> tuple[dict[int, Any], dict[int, Any]]:
    import torch

    require(
        condition in {"BASE", "W_EQ", "H_EQ", "W_SEED_H_CARRY", "WH_EQ"},
        "K3C_INTERVENTION_CONDITION_INVALID",
    )
    require(
        start - 1 in corr["post_state"]
        and start - 1 in ctrl["post_state"],
        "K3C_PAIR_INITIAL_STATE_MISSING",
    )
    require(
        torch.equal(
            corr["post_state"][start - 1],
            ctrl["post_state"][start - 1],
        ),
        "K3C_PAIR_INITIAL_STATE_NOT_EQUAL",
    )
    needed = set(range(start, end + 1))
    for branch in (corr, ctrl):
        require(needed <= set(branch["G"]), "K3C_PAIR_G_INCOMPLETE")
        require(needed <= set(branch["W"]), "K3C_PAIR_W_INCOMPLETE")

    c_state = corr["post_state"][start - 1].detach().cpu().contiguous().clone()
    n_state = ctrl["post_state"][start - 1].detach().cpu().contiguous().clone()
    c_out: dict[int, Any] = {}
    n_out: dict[int, Any] = {}

    for t in range(start, end + 1):
        c_h = retained_contribution(corr["G"][t], c_state)
        n_h = retained_contribution(ctrl["G"][t], n_state)
        hbar = arithmetic_midpoint(c_h, n_h)
        wbar = arithmetic_midpoint(corr["W"][t], ctrl["W"][t])

        if condition == "BASE":
            c_next = c_h + corr["W"][t]
            n_next = n_h + ctrl["W"][t]
        elif condition == "W_EQ":
            c_next = c_h + wbar
            n_next = n_h + wbar
        elif condition == "H_EQ":
            c_next = hbar + corr["W"][t]
            n_next = hbar + ctrl["W"][t]
        elif condition == "W_SEED_H_CARRY":
            if t == start:
                c_next = hbar + corr["W"][t]
                n_next = hbar + ctrl["W"][t]
            else:
                c_next = c_h + wbar
                n_next = n_h + wbar
        else:  # WH_EQ
            c_next = hbar + wbar
            n_next = hbar + wbar

        c_state = c_next.detach().cpu().contiguous().clone()
        n_state = n_next.detach().cpu().contiguous().clone()
        c_out[t] = c_state
        n_out[t] = n_state

    return c_out, n_out


def combine_natural_and_replay(
    natural_post: Mapping[int, Any],
    replayed: Mapping[int, Any],
    start: int,
    p: int,
) -> dict[int, Any]:
    required = range(p - 1, p + W + 1)
    out: dict[int, Any] = {}
    for t in required:
        if t < start:
            require(t in natural_post, "K3C_NATURAL_PREINTERVENTION_MISSING")
            out[t] = natural_post[t]
        else:
            require(t in replayed, "K3C_REPLAYED_STATE_MISSING")
            out[t] = replayed[t]
    return out


def x_pair_specificity(
    branch_metrics_by_name: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | None]:
    required = {
        "matched_corr",
        "matched_ctrl",
        "swapped_corr",
        "swapped_ctrl",
    }
    require(set(branch_metrics_by_name) == required, "K3C_BRANCH_SET_INVALID")
    fields = {
        "R": "R_mean_speed",
        "D": "D_mean_turn",
        "DISP": "displacement",
        "P": "P_efficiency",
    }
    out: dict[str, float | None] = {}
    for metric, field in fields.items():
        mc = branch_metrics_by_name["matched_corr"][field]
        mn = branch_metrics_by_name["matched_ctrl"][field]
        sc = branch_metrics_by_name["swapped_corr"][field]
        sn = branch_metrics_by_name["swapped_ctrl"][field]
        if None in (mc, mn, sc, sn):
            out[metric] = None
        else:
            dm = float(mc - mn)
            ds = float(sc - sn)
            value = abs(dm) - abs(ds)
            require(math.isfinite(value), "K3C_X_NONFINITE")
            out[metric] = value
    return out


def aligned_signal(metric: str, block_value: float | None) -> float | None:
    require(metric in METRICS, "K3C_METRIC_INVALID")
    return (
        None
        if block_value is None
        else EXPECTED_DIRECTION[metric] * float(block_value)
    )


def mechanism_values(
    z_base: float | None,
    z_w_eq: float | None,
    z_h_eq: float | None,
    z_carry: float | None,
) -> dict[str, float | None]:
    if None in (z_base, z_w_eq, z_h_eq):
        dom = None
        att_w = None
        att_h = None
    else:
        att_w = float(z_base - z_w_eq)
        att_h = float(z_base - z_h_eq)
        dom = float(att_w - att_h)
    carry = None if z_carry is None else float(z_carry)
    return {
        "ATT_W": att_w,
        "ATT_H": att_h,
        "DOM": dom,
        "CARRY": carry,
    }


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    require(positive >= 0 and negative >= 0, "SIGN_COUNTS_INVALID")
    n = positive + negative
    if n == 0:
        return 1.0
    tail = min(positive, negative)
    numerator = sum(math.comb(n, k) for k in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2 ** n))


def summarize_test(values: Sequence[float | None]) -> dict[str, Any]:
    undefined = sum(v is None for v in values)
    valid = [float(v) for v in values if v is not None]
    require(
        all(math.isfinite(v) for v in valid),
        "K3C_PRIMARY_VALUE_NONFINITE",
    )
    pos = sum(v > 0.0 for v in valid)
    neg = sum(v < 0.0 for v in valid)
    zero = sum(v == 0.0 for v in valid)
    n_eff = pos + neg
    floor = len(valid) >= 120 and n_eff >= 30
    raw_p = exact_two_sided_sign_p(pos, neg) if floor else 1.0
    effect = None if n_eff == 0 else (pos - neg) / n_eff
    return {
        "n_valid": len(valid),
        "n_eff": n_eff,
        "positive_count": pos,
        "negative_count": neg,
        "zero_count": zero,
        "undefined_count": undefined,
        "promotion_floor_pass": floor,
        "raw_p": raw_p,
        "rank_biserial_sign_effect": effect,
    }


def holm_adjust(
    raw: Mapping[str, float],
    order: Sequence[str],
) -> dict[str, dict[str, Any]]:
    require(set(raw) == set(order), "K3C_HOLM_SET_MISMATCH")
    tie = {name: i for i, name in enumerate(order)}
    ordered = sorted(order, key=lambda name: (float(raw[name]), tie[name]))
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
        for name in order
    }


def base_replication_statistics(
    values: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    require(set(values) == set(BASE_ORDER), "K3C_BASE_TEST_SET_MISMATCH")
    endpoints = {
        metric: summarize_test(values[metric])
        for metric in BASE_ORDER
    }
    holm = holm_adjust(
        {metric: endpoints[metric]["raw_p"] for metric in BASE_ORDER},
        BASE_ORDER,
    )
    matches: list[str] = []
    contradictions: list[str] = []
    for metric in BASE_ORDER:
        endpoints[metric].update(holm[metric])
        effect = endpoints[metric]["rank_biserial_sign_effect"]
        expected = EXPECTED_DIRECTION[metric]
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
        endpoints[metric]["expected_direction"] = (
            "positive" if expected > 0 else "negative"
        )
        endpoints[metric]["direction_match"] = match
        endpoints[metric]["directional_contradiction"] = contradiction
        if match:
            matches.append(metric)
        if contradiction:
            contradictions.append(metric)

    if matches == list(BASE_ORDER):
        verdict = "PASS"
    elif contradictions:
        verdict = "CONTRADICTED"
    else:
        verdict = "NOT_ESTABLISHED"

    return {
        "primary_order": list(BASE_ORDER),
        "holm_m": 4,
        "holm_alpha": 0.05,
        "endpoints": endpoints,
        "direction_matched_endpoints": matches,
        "directional_contradiction_endpoints": contradictions,
        "base_replication_verdict": verdict,
    }


def mechanism_statistics(
    values: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    require(
        set(values) == set(PRIMARY_TEST_ORDER),
        "K3C_MECHANISM_TEST_SET_MISMATCH",
    )
    tests = {
        name: summarize_test(values[name])
        for name in PRIMARY_TEST_ORDER
    }
    holm = holm_adjust(
        {name: tests[name]["raw_p"] for name in PRIMARY_TEST_ORDER},
        PRIMARY_TEST_ORDER,
    )
    matches: list[str] = []
    contradictions: list[str] = []
    for name in PRIMARY_TEST_ORDER:
        tests[name].update(holm[name])
        effect = tests[name]["rank_biserial_sign_effect"]
        match = bool(
            tests[name]["promotion_floor_pass"]
            and tests[name]["holm_reject"]
            and effect is not None
            and effect > 0
        )
        contradiction = bool(
            tests[name]["promotion_floor_pass"]
            and tests[name]["holm_reject"]
            and effect is not None
            and effect < 0
        )
        tests[name]["direction_match"] = match
        tests[name]["directional_contradiction"] = contradiction
        if match:
            matches.append(name)
        if contradiction:
            contradictions.append(name)

    full = matches == list(PRIMARY_TEST_ORDER)
    if full:
        verdict = SUCCESS_VERDICT
    elif contradictions:
        verdict = CONTRADICTION_VERDICT
    else:
        verdict = NOT_ESTABLISHED_VERDICT

    return {
        "primary_test_order": list(PRIMARY_TEST_ORDER),
        "holm_m": 8,
        "holm_alpha": 0.05,
        "tests": tests,
        "direction_matched_tests": matches,
        "directional_contradiction_tests": contradictions,
        "full_support": full,
        "mechanism_verdict": verdict,
    }


def k3c_scientific_statistics(
    base_values: Mapping[str, Sequence[float | None]],
    mechanism_test_values: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    base = base_replication_statistics(base_values)
    mechanism = mechanism_statistics(mechanism_test_values)
    if base["base_replication_verdict"] != "PASS":
        scientific_verdict = BASE_FAILURE_VERDICT
        mechanism_authorized_for_promotion = False
    else:
        scientific_verdict = mechanism["mechanism_verdict"]
        mechanism_authorized_for_promotion = True
    return {
        "base_replication": base,
        "mechanism": mechanism,
        "mechanism_authorized_for_promotion":
            mechanism_authorized_for_promotion,
        "scientific_verdict": scientific_verdict,
    }


def _synthetic_pair_terms() -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    corr = {
        "G": {
            2: torch.tensor([[[0.5]]], dtype=torch.float32),
            3: torch.tensor([[[0.5]]], dtype=torch.float32),
            4: torch.tensor([[[0.5]]], dtype=torch.float32),
        },
        "W": {
            2: torch.tensor([[[2.0]]], dtype=torch.float32),
            3: torch.tensor([[[2.0]]], dtype=torch.float32),
            4: torch.tensor([[[2.0]]], dtype=torch.float32),
        },
        "post_state": {
            1: torch.tensor([[[0.0]]], dtype=torch.float32),
        },
    }
    ctrl = {
        "G": {
            2: torch.tensor([[[0.25]]], dtype=torch.float32),
            3: torch.tensor([[[0.25]]], dtype=torch.float32),
            4: torch.tensor([[[0.25]]], dtype=torch.float32),
        },
        "W": {
            2: torch.tensor([[[0.0]]], dtype=torch.float32),
            3: torch.tensor([[[0.0]]], dtype=torch.float32),
            4: torch.tensor([[[0.0]]], dtype=torch.float32),
        },
        "post_state": {
            1: torch.tensor([[[0.0]]], dtype=torch.float32),
        },
    }
    return corr, ctrl


def synthetic_known_term_source_carry_control() -> dict[str, Any]:
    import torch
    corr, ctrl = _synthetic_pair_terms()

    seed_c, seed_n = replay_pair_contribution(
        corr, ctrl, 2, 4, "W_SEED_H_CARRY"
    )
    require(
        torch.equal(seed_c[2], torch.tensor([[[2.0]]]))
        and torch.equal(seed_n[2], torch.tensor([[[0.0]]])),
        "K3C_KNOWN_SEED_STEP_FAILURE",
    )
    require(
        not torch.equal(seed_c[3], seed_n[3]),
        "K3C_KNOWN_CARRY_STEP_FAILURE",
    )
    # Later W is equalized to 1.0, so residual divergence after d is H-carried.
    require(
        torch.equal(seed_c[3], torch.tensor([[[2.0]]]))
        and torch.equal(seed_n[3], torch.tensor([[[1.0]]])),
        "K3C_KNOWN_CARRY_VALUE_FAILURE",
    )

    h_c, h_n = replay_pair_contribution(corr, ctrl, 2, 4, "H_EQ")
    # H equalization makes the retained contribution identical each step;
    # branch differences therefore equal the branch-specific W difference.
    require(
        torch.equal(h_c[2] - h_n[2], torch.tensor([[[2.0]]])),
        "K3C_KNOWN_H_EQ_FAILURE",
    )

    wh_c, wh_n = replay_pair_contribution(corr, ctrl, 2, 4, "WH_EQ")
    for t in range(2, 5):
        require(
            torch.equal(wh_c[t], wh_n[t]),
            "K3C_KNOWN_WH_EQ_COLLAPSE_FAILURE",
        )

    return {
        "known_term_source_carry_replay": "PASS_EXACT",
        "known_term_h_eq": "PASS_EXACT",
        "known_term_wh_eq_collapse": "PASS_EXACT",
    }


def run_replay_preflight(
    root: Path,
    k2s: Any,
    k3: Any,
    model: Any,
    tokenizer: Any,
    component_binding: Any,
) -> dict[str, Any]:
    import torch

    prefix = (
        "Claim: synthetic k3 blorp\n"
        "Evidence: synthetic k3 snarp\n"
        "Additional evidence:\n"
    )
    corr = (
        " synthetic corrective alpha beta gamma delta epsilon "
        "zeta eta theta iota kappa."
    )
    ctrl = (
        " synthetic control lambda mu nu xi omicron pi rho sigma "
        "tau upsilon."
    )
    prefix_ids = _token_ids(tokenizer, prefix)
    corr_ids = _token_ids(tokenizer, prefix + corr)
    ctrl_ids = _token_ids(tokenizer, prefix + ctrl)
    require(
        corr_ids[:len(prefix_ids)] == prefix_ids
        == ctrl_ids[:len(prefix_ids)],
        "K3C_SYNTHETIC_PREFIX_TOKEN_MISMATCH",
    )
    p = len(prefix_ids) - 1
    require(p >= 1, "K3C_SYNTHETIC_PREFIX_TOO_SHORT")
    require(
        len(corr_ids) >= p + W + 1
        and len(ctrl_ids) >= p + W + 1,
        "K3C_SYNTHETIC_WINDOW_TOO_SHORT",
    )
    d = k3.first_divergence(
        corr_ids,
        ctrl_ids,
        len(prefix_ids),
        p + W,
    )
    end = p + W
    component_indices = range(d, end + 1)
    state_indices = range(p - 1, end + 1)

    # Noninterference on synthetic text only.
    bundle = k2s.task_mask_bundle(tokenizer, prefix + corr)
    baseline = k2s._full_model_forward(model, bundle)
    baseline_logits = k2s._logits(baseline).detach().cpu().clone()
    mixer = model.mamba.layers[PRIMARY_LAYER].mixer
    trace = k3.ComponentCollector(
        component_binding,
        mixer,
        component_indices,
        state_indices,
    )
    with trace.capture():
        traced = k2s._full_model_forward(model, bundle)
    traced_logits = k2s._logits(traced).detach().cpu()
    require(
        torch.equal(baseline_logits, traced_logits),
        "K3C_COMPONENT_TRACE_NONINTERFERENCE_FAILURE",
    )

    corr_cap = k3._branch_capture(
        model,
        corr_ids,
        component_binding,
        component_indices,
        state_indices,
    )
    ctrl_cap = k3._branch_capture(
        model,
        ctrl_ids,
        component_binding,
        component_indices,
        state_indices,
    )
    require(
        torch.equal(
            corr_cap["post_state"][d - 1],
            ctrl_cap["post_state"][d - 1],
        ),
        "K3C_SYNTHETIC_D_MINUS_1_STATE_MISMATCH",
    )

    # Natural H identity and natural replay exactness.
    for branch in (corr_cap, ctrl_cap):
        for t in component_indices:
            require(
                torch.equal(
                    branch["pre_state"][t],
                    branch["post_state"][t - 1],
                ),
                "K3C_PRE_POST_COORDINATE_MISMATCH",
            )
            natural_h = retained_contribution(
                branch["G"][t],
                branch["pre_state"][t],
            )
            expected_post = natural_h + branch["W"][t]
            require(
                torch.equal(expected_post, branch["post_state"][t]),
                "K3C_NATURAL_H_PLUS_W_IDENTITY_FAILURE",
            )

        natural = structural_replay(
            branch["post_state"][d - 1],
            branch["G"],
            branch["W"],
            d,
            end,
        )
        for t in range(d, end + 1):
            require(
                torch.equal(natural[t], branch["post_state"][t]),
                "K3C_NATURAL_STRUCTURAL_REPLAY_FAILURE",
            )

        sham_g = {t: branch["G"][t].clone() for t in component_indices}
        sham_w = {t: branch["W"][t].clone() for t in component_indices}
        sham = structural_replay(
            branch["post_state"][d - 1],
            sham_g,
            sham_w,
            d,
            end,
        )
        for t in range(d, end + 1):
            require(
                torch.equal(sham[t], branch["post_state"][t]),
                "K3C_SHAM_REPLAY_FAILURE",
            )

    # All contribution interventions must be executable/finite.
    condition_outputs: dict[str, tuple[dict[int, Any], dict[int, Any]]] = {}
    for condition in ("W_EQ", "H_EQ", "W_SEED_H_CARRY", "WH_EQ"):
        a, b = replay_pair_contribution(
            corr_cap,
            ctrl_cap,
            d,
            end,
            condition,
        )
        condition_outputs[condition] = (a, b)
        for t in range(d, end + 1):
            require(
                bool(torch.isfinite(a[t]).all().item()),
                f"K3C_{condition}_CORR_NONFINITE",
            )
            require(
                bool(torch.isfinite(b[t]).all().item()),
                f"K3C_{condition}_CTRL_NONFINITE",
            )

    wh_corr, wh_ctrl = condition_outputs["WH_EQ"]
    for t in range(d, end + 1):
        require(
            torch.equal(wh_corr[t], wh_ctrl[t]),
            "K3C_WH_EQ_PAIR_STATE_COLLAPSE_FAILURE",
        )

    # At d, W_SEED_H_CARRY has equal retained contribution by construction.
    seed_corr, seed_ctrl = condition_outputs["W_SEED_H_CARRY"]
    c_h_d = retained_contribution(
        corr_cap["G"][d],
        corr_cap["post_state"][d - 1],
    )
    n_h_d = retained_contribution(
        ctrl_cap["G"][d],
        ctrl_cap["post_state"][d - 1],
    )
    hbar_d = arithmetic_midpoint(c_h_d, n_h_d)
    require(
        torch.equal(
            seed_corr[d],
            hbar_d + corr_cap["W"][d],
        )
        and torch.equal(
            seed_ctrl[d],
            hbar_d + ctrl_cap["W"][d],
        ),
        "K3C_W_SEED_D_SEMANTICS_FAILURE",
    )

    known = synthetic_known_term_source_carry_control()

    return {
        "status": "PASS_K3C_REPLAY_PREFLIGHT",
        "scientific_population_recurrent_state_read": False,
        "scientific_population_intervention_executed": False,
        "component_capture_noninterference": "PASS_EXACT",
        "natural_h_plus_w_identity": "PASS_EXACT",
        "natural_structural_replay": "PASS_EXACT",
        "sham_replay": "PASS_EXACT",
        "w_eq_synthetic_semantics": "PASS",
        "h_eq_dynamic_semantics": "PASS",
        "w_seed_h_carry_synthetic_semantics": "PASS",
        "wh_eq_pair_state_collapse": "PASS_EXACT",
        **known,
        "capture_layer": PRIMARY_LAYER,
        "synthetic_prefix_token_count": len(prefix_ids),
        "synthetic_corr_token_count": len(corr_ids),
        "synthetic_ctrl_token_count": len(ctrl_ids),
        "synthetic_p": p,
        "synthetic_d": d,
        "synthetic_d_minus_p": d - p,
        "synthetic_replay_end": end,
        "component_dtype": "torch.float32",
        "device": "cpu",
        "recurrence_source": {
            "qualname": component_binding.qualname,
            "path": str(component_binding.source_path),
            "sha256": component_binding.source_sha256,
            "bytes": component_binding.source_bytes,
            "discrete_A_line": component_binding.discrete_A_line,
            "deltaB_u_line": component_binding.deltaB_u_line,
            "recurrence_update_line":
                component_binding.recurrence_update_line,
            "post_update_line": component_binding.post_update_line,
        },
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="K3C native recurrence contribution decomposition"
    )
    p.add_argument("--seed180-handoff", required=True)
    p.add_argument("--hf-revision", required=True)
    p.add_argument("--replay-preflight", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    require(
        args.replay_preflight,
        "K3C_SCIENTIFIC_EXECUTION_NOT_AUTHORIZED_IN_IMPLEMENTATION_PHASE",
    )
    root = Path(__file__).resolve().parents[1]
    runtime = git_provenance(root, replay_preflight=True)
    generator, k2s, k3 = load_frozen_dependencies(root)

    # State-blind prospective population reproduction. No model forward here.
    source_rows, candidates, population = materialize_population(root, generator)

    snapshot, hf = k2s.resolve_hf_snapshot(args.hf_revision)
    require(
        hf["hf_model_id"] == HF_MODEL
        and hf["resolved_hf_revision"] == HF_REVISION,
        "K3C_HF_IDENTITY_MISMATCH",
    )
    require(
        hf["transformers_version"] == TRANSFORMERS_VERSION,
        "K3C_TRANSFORMERS_VERSION_MISMATCH",
    )
    input_contracts, feasibility = build_input_contracts(
        candidates,
        hf["tokenizer"],
    )

    # After population tokenization only, load the frozen model for SYNTHETIC
    # replay preflight. No candidate branch is forwarded through the model.
    handoff = k2s.audit_handoff(Path(args.seed180_handoff))
    require(
        handoff["zip_sha256"] == EXPECTED_ZIP_SHA256,
        "K3C_HANDOFF_ZIP_SHA_MISMATCH",
    )
    require(
        handoff["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "K3C_CHECKPOINT_SHA_MISMATCH",
    )
    checkpoint = k2s.load_authenticated_checkpoint(handoff)
    encoder = k2s.encoder_fingerprint(checkpoint["model_state_dict"])
    require(
        encoder["canonical_digest"] == COMMON_ENCODER_CANONICAL_SHA256,
        "K3C_ENCODER_CANONICAL_SHA_MISMATCH",
    )
    require(
        encoder["raw_concat_digest"] == COMMON_ENCODER_RAW_CONCAT_SHA256,
        "K3C_ENCODER_RAW_SHA_MISMATCH",
    )

    model = k2s.build_a0_model(root, snapshot, checkpoint)
    model.eval()
    first_parameter = next(model.parameters())
    require(
        first_parameter.device.type == "cpu",
        "K3C_CPU_REQUIRED",
    )
    binding = k3.resolve_component_binding(k2s)
    preflight = run_replay_preflight(
        root,
        k2s,
        k3,
        model,
        hf["tokenizer"],
        binding,
    )

    result = {
        "k3c_replay_preflight": "PASS",
        "runtime": runtime,
        "population": {
            **population,
            "input_contract_count": len(input_contracts),
            "tokenizer_feasibility": feasibility,
            "generated_source_materialized_in_memory_only": True,
            "candidate_pool_materialized_in_memory_only": True,
        },
        "handoff": {
            **handoff,
            "encoder": encoder,
            "strict_load": "PASS",
        },
        "hf": {
            key: value
            for key, value in hf.items()
            if key not in {"config", "tokenizer"}
        },
        "dependency_binding": {
            "generator_sha256": GENERATOR_SHA256,
            "generator_git_blob": GENERATOR_GIT_BLOB,
            "k2s_runner_sha256": K2S_RUNNER_SHA256,
            "k2s_runner_git_blob": K2S_RUNNER_GIT_BLOB,
            "k3_replay_sha256": K3_REPLAY_SHA256,
            "k3_replay_git_blob": K3_REPLAY_GIT_BLOB,
        },
        "replay": preflight,
        "execution_boundary": {
            "k3c_design_frozen": True,
            "k3c_implementation_validation_authorized": True,
            "k3c_scientific_recurrent_state_execution_authorized": False,
            "scientific_cli_reachable": False,
        },
    }
    print(canonical_json(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
