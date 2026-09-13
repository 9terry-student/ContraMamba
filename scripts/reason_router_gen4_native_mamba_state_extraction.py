from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import reason_router_gen4_native_mamba_state_measurement as measurement  # noqa: E402
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter  # noqa: E402
from scripts import reason_router_gen4_six_cell_tier2_scientific_inference as r5  # noqa: E402

PHASE_D_RUNNER_IMPLEMENTATION_AUTHORITY = "66f56152e97445429eb5b329e0fc849e7cc81492"
SOURCE_ROLE_VALIDATOR_CORRECTION_AUTHORITY = "5ee0826c629978a72240bb88f732aacf6f357c4b"
PHASE_C_MEASUREMENT_IMPLEMENTATION_COMMIT = "e3c870e7f24e183b0046b568e1de3b71446c182d"
PHASE_C_IMPLEMENTATION_AUTHORITY = "480ff74aebf5ef942aa9f47fa78c06612a8f97a4"
PHASE_AB_FEASIBILITY_FREEZE = "26fd55803acd05febefc8bd031f2fc23c17b0ef4"
MECHANISTIC_BRIDGE_SPECIFICATION = "a2617aa037d1a9834003535b62ac81770a5b96aa"

RUNNER_PATH = Path("scripts/reason_router_gen4_native_mamba_state_extraction.py")
EXECUTION_AUTHORITY_PATH = Path(
    "reports/reason_router_gen4_native_mamba_state_bridge_"
    "phase_d_extraction_execution_authority_correction_spec_candidate.md"
)
EXECUTION_AUTHORITY_ID = "GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_D_EXTRACTION_EXECUTION_AUTHORITY"
MEASUREMENT_PATH = Path("scripts/reason_router_gen4_native_mamba_state_measurement.py")
ADAPTER_PATH = Path("scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py")
R5_PATH = Path("scripts/reason_router_gen4_six_cell_tier2_scientific_inference.py")
HISTORICAL_MODEL_PATH = Path("src/contramamba/modeling_v6b_minimal_gen3_grouped_snapshot.py")

MEASUREMENT_SHA256 = "e31d82af229c79b7de3d55f3419a1ae1cc9da0e1c0af5bfd84ba4cec85a36a82"
ADAPTER_SHA256 = "83177c351f82a781586c63bd8d4ef1b40e759b5a94858dc1837d65502cbff6e5"
R5_SHA256 = "468a758a7d20d048c75a0ca7e298b73a65f538527df55d3ecad3c7ff1760cf4d"
HISTORICAL_MODEL_SHA256 = "8c365bfa857157d91f363358d5db3abaab425dec3e0d7c62683b4207a589b6a5"

CANONICAL_GEN4_ARTIFACT = adapter.CANONICAL_GEN4_ARTIFACT
CANONICAL_GEN4_ARTIFACT_SHA256 = adapter.CANONICAL_GEN4_ARTIFACT_SHA256
CANONICAL_GEN4_ARTIFACT_BYTES = adapter.CANONICAL_GEN4_ARTIFACT_BYTES
CANONICAL_GEN4_ROWS = 1800
SOURCE_PAIR_COUNT = 300
CANONICAL_CELLS = tuple(adapter.CANONICAL_CELLS)

EVENT_MANIFEST = Path(
    "reports/reason_router_gen4_six_cell_native_mamba_state_bridge_feasibility_audit_a2617aa/"
    "event_anchor_prefix_manifest_candidate.jsonl"
)
EVENT_MANIFEST_SHA256 = "70c84c68b36751bb7c7145b33ccb71ab91bc8ee9e6cc5f2c7a0d4e925f36581f"
EVENT_MANIFEST_BYTES = 2268260
EVENT_SCHEMA = "gen4_event_anchor_prefix_manifest_v1"
EVENT_KEYS = {
    "absolute_anchor_token_index",
    "anchor_name",
    "anchor_token_id",
    "claim_active_token_count",
    "claim_raw_token_count",
    "contrast_cell_id",
    "evidence_active_token_count",
    "evidence_anchor_token_index",
    "evidence_raw_token_count",
    "generator_span_end_char_exclusive",
    "generator_span_field",
    "generator_span_start_char",
    "post4_eligible",
    "post4_end_index",
    "post4_rule",
    "row_id",
    "schema_version",
    "source_pair_id",
    "terminal_index",
}
EVENT_TYPES = {
    "absolute_anchor_token_index": int,
    "anchor_name": str,
    "anchor_token_id": int,
    "claim_active_token_count": int,
    "claim_raw_token_count": int,
    "contrast_cell_id": str,
    "evidence_active_token_count": int,
    "evidence_anchor_token_index": int,
    "evidence_raw_token_count": int,
    "generator_span_end_char_exclusive": int,
    "generator_span_field": str,
    "generator_span_start_char": int,
    "post4_eligible": bool,
    "post4_end_index": int,
    "post4_rule": str,
    "row_id": str,
    "schema_version": str,
    "source_pair_id": str,
    "terminal_index": int,
}
ANCHOR_ORDER = ("A_TITLE", "A_NAME", "A_ROLE", "A_PREDICATE", "A_IDENTITY")
REQUIRED_ANCHOR_CELLS = {
    "A_TITLE": ("C0_SHAM", "C1_TITLE"),
    "A_NAME": ("C0_SHAM", "C2_NAME"),
    "A_ROLE": ("C0_SHAM", "C3_ROLE"),
    "A_PREDICATE": ("C0_SHAM", "C4_PREDICATE"),
    "A_IDENTITY": ("C0_SHAM", "C1_TITLE", "C2_NAME", "C5_TITLE_NAME"),
}
REQUIRED_ANCHOR_CELL_ROWS = 3600
POST4_RULE = "a+4<=terminal_index-1"

REPRESENTATIVE_SEED = 180
REPRESENTATIVE_ARM = "G3-GROUP-D-HALF"
REPRESENTATIVE_CHECKPOINT = Path(
    "reports/reason_router_gen3_grouped_factorial_runs/seed180/"
    "G3-GROUP-D-HALF/selected_checkpoint.pt"
)
REPRESENTATIVE_CHECKPOINT_SHA256 = "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"

MODEL_CONFIG_SHA256 = r5.MODEL_CONFIG_SHA256
MODEL_CONFIG_BYTES = r5.MODEL_CONFIG_BYTES
TOKENIZER_REVISION_REFERENCE = adapter.TOKENIZER_REVISION_REFERENCE
R2_ENCODED_COORDINATE_SHA256 = r5.R2_ENCODED_COORDINATE_SHA256

PRIMARY_LAYER = measurement.PRIMARY_LAYER_INDEX
STATE_VECTOR_SIZE = measurement.FLATTENED_STATE_SIZE
MAX_MODEL_SEQUENCE_LENGTH = adapter.MAX_LENGTH
SCIENTIFIC_FORWARD_BATCH_SIZE = 1
EXPECTED_BACKBONE_FORWARD_COUNT = 1800
MODEL_REPLICATION_COUNT = 1

SUPPORT_ROW_SCHEMA = "gen4_native_mamba_support_state_row_v1"
ENDPOINT_ROW_SCHEMA = "gen4_native_mamba_kinematic_endpoint_row_v1"
MANIFEST_SCHEMA = "gen4_native_mamba_state_extraction_manifest_v1"
REQUIRED_ARTIFACTS = (
    "manifest.json",
    "support_state_rows.jsonl",
    "support_states.npy",
    "kinematic_endpoints.jsonl",
    "SHA256SUMS.txt",
)

BLOCKED_EXECUTION_AUTHORITY = "BLOCKED_PHASE_D_EXECUTION_AUTHORITY"
BLOCKED_NONFINITE_MEASUREMENT = "BLOCKED_NONFINITE_MEASUREMENT"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ExtractionContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExtractionContractError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_show_bytes(revision: str, path: str | Path) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{revision}:{Path(path).as_posix()}"], cwd=ROOT
    )


def git_commit_exists(revision: str) -> bool:
    if not _HEX40.fullmatch(str(revision)):
        return False
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def canonical_git_sha256(revision: str, path: str | Path) -> str:
    return sha256_bytes(git_show_bytes(revision, path))


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def git_status_porcelain() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)


def validate_execution_authority_document(
    execution_authority_commit: str,
    *,
    runner_implementation_commit: str,
    expected_runner_sha256: str,
) -> None:
    try:
        raw = git_show_bytes(execution_authority_commit, EXECUTION_AUTHORITY_PATH)
    except subprocess.CalledProcessError as exc:
        raise ExtractionContractError(BLOCKED_EXECUTION_AUTHORITY) from exc
    text = raw.decode("utf-8")
    required = (
        f"AUTHORITY_ID =\n{EXECUTION_AUTHORITY_ID}",
        "SCIENTIFIC_EXECUTION_ALLOWED =\nYES_BOUNDED",
        f"PHASE_D_RUNNER_IMPLEMENTATION_COMMIT =\n{runner_implementation_commit}",
        f"PHASE_D_RUNNER_SHA256 =\n{expected_runner_sha256}",
    )
    for clause in required:
        require(clause in text, BLOCKED_EXECUTION_AUTHORITY)


def validate_frozen_dependency_identities(revision: str) -> None:
    expected = {
        MEASUREMENT_PATH: MEASUREMENT_SHA256,
        ADAPTER_PATH: ADAPTER_SHA256,
        R5_PATH: R5_SHA256,
        HISTORICAL_MODEL_PATH: HISTORICAL_MODEL_SHA256,
    }
    for path, digest in expected.items():
        observed = canonical_git_sha256(revision, path)
        require(observed == digest, f"frozen dependency identity mismatch: {path}")


def validate_execution_binding(
    *,
    expected_execution_head: str,
    runner_implementation_commit: str,
    expected_runner_sha256: str,
    execution_authority_commit: str,
) -> str:
    observed_head = git_head()
    require(observed_head == expected_execution_head, "execution HEAD mismatch")
    require(execution_authority_commit == expected_execution_head, BLOCKED_EXECUTION_AUTHORITY)
    require(execution_authority_commit != PHASE_D_RUNNER_IMPLEMENTATION_AUTHORITY, BLOCKED_EXECUTION_AUTHORITY)
    require(git_commit_exists(execution_authority_commit), BLOCKED_EXECUTION_AUTHORITY)
    require(_HEX40.fullmatch(runner_implementation_commit) is not None, "runner commit format")
    require(git_commit_exists(runner_implementation_commit), "runner implementation commit missing")
    require(re.fullmatch(r"[0-9a-f]{64}", expected_runner_sha256) is not None, "runner SHA256 format")
    require(git_is_ancestor(runner_implementation_commit, execution_authority_commit), BLOCKED_EXECUTION_AUTHORITY)
    validate_execution_authority_document(
        execution_authority_commit,
        runner_implementation_commit=runner_implementation_commit,
        expected_runner_sha256=expected_runner_sha256,
    )
    require(git_status_porcelain() == "", "repository dirty before scientific execution")
    runner_bytes = git_show_bytes(runner_implementation_commit, RUNNER_PATH)
    require(sha256_bytes(runner_bytes) == expected_runner_sha256, "runner identity mismatch")
    require(Path(ROOT / RUNNER_PATH).read_bytes().replace(b"\r\n", b"\n") == runner_bytes, "runner worktree mismatch")
    validate_frozen_dependency_identities(observed_head)
    return observed_head


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"non-object JSONL row: {line_no}")
        rows.append(value)
    return rows


def load_event_manifest(path: str | Path = EVENT_MANIFEST) -> list[dict[str, Any]]:
    path = Path(path)
    raw = path.read_bytes()
    require(len(raw) == EVENT_MANIFEST_BYTES, "event manifest byte-count mismatch")
    require(sha256_bytes(raw) == EVENT_MANIFEST_SHA256, "event manifest SHA256 mismatch")
    rows = read_jsonl(path)
    validate_event_manifest(rows)
    return rows


def validate_event_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == REQUIRED_ANCHOR_CELL_ROWS, "event manifest row-count mismatch")
    combos: Counter[tuple[str, str]] = Counter()
    identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        require(set(raw) == EVENT_KEYS, f"event manifest schema mismatch row={index}")
        for key, expected_type in EVENT_TYPES.items():
            require(type(raw[key]) is expected_type, f"event field type mismatch: {key}")
        require(raw["schema_version"] == EVENT_SCHEMA, "event schema version mismatch")
        anchor = raw["anchor_name"]
        cell = raw["contrast_cell_id"]
        require(anchor in REQUIRED_ANCHOR_CELLS, "unexpected anchor")
        require(cell in REQUIRED_ANCHOR_CELLS[anchor], "unexpected anchor/cell combination")
        require(type(raw["absolute_anchor_token_index"]) is int, "anchor index type")
        require(type(raw["terminal_index"]) is int, "terminal index type")
        require(type(raw["post4_end_index"]) is int, "post4 end type")
        require(raw["post4_eligible"] is True, "post4 eligibility")
        require(raw["post4_rule"] == POST4_RULE, "post4 rule")
        anchor_index = int(raw["absolute_anchor_token_index"])
        terminal = int(raw["terminal_index"])
        require(anchor_index >= 1, "anchor requires s_(a-1)")
        require(int(raw["post4_end_index"]) == anchor_index + 4, "post4 end mismatch")
        require(
            anchor_index
            == int(raw["claim_active_token_count"]) + 1 + int(raw["evidence_anchor_token_index"]),
            "absolute/evidence anchor coordinate mismatch",
        )
        require(
            terminal
            == int(raw["claim_active_token_count"]) + int(raw["evidence_active_token_count"]),
            "active terminal coordinate mismatch",
        )
        require(anchor_index + 4 <= terminal - 1, "active-terminal prefix rule")
        require(0 <= terminal < MAX_MODEL_SEQUENCE_LENGTH, "active terminal range")
        identity = (str(raw["row_id"]), str(anchor))
        require(identity not in identities, "duplicate row/anchor event")
        identities.add(identity)
        combos[(str(anchor), str(cell))] += 1
    expected = {
        (anchor, cell): SOURCE_PAIR_COUNT
        for anchor in ANCHOR_ORDER
        for cell in REQUIRED_ANCHOR_CELLS[anchor]
    }
    require(dict(combos) == expected, "event anchor/cell counts mismatch")
    require(len({str(row["source_pair_id"]) for row in rows}) == SOURCE_PAIR_COUNT, "event source-pair count mismatch")


def validate_canonical_population(
    rows: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
) -> None:
    normalized = adapter.validate_gen4_rows(rows, require_canonical_shape=True)
    require(len(normalized) == CANONICAL_GEN4_ROWS, "canonical row count mismatch")
    require(len(encoded["row_id"]) == CANONICAL_GEN4_ROWS, "encoded row count mismatch")
    require(list(encoded["row_id"]) == [row["row_id"] for row in normalized], "encoded row order mismatch")
    require(list(encoded["source_pair_id"]) == [row["source_pair_id"] for row in normalized], "encoded pair order mismatch")
    require(list(encoded["contrast_cell_id"]) == [row["contrast_cell_id"] for row in normalized], "encoded cell order mismatch")
    require(adapter.encoded_coordinate_sha256(encoded) == R2_ENCODED_COORDINATE_SHA256, "encoded coordinate identity mismatch")
    input_ids = encoded["input_ids"]
    attention = encoded["attention_mask"]
    claim_mask = encoded["claim_mask"]
    evidence_mask = encoded["evidence_mask"]
    require(isinstance(input_ids, torch.Tensor), "encoded input_ids tensor")
    require(isinstance(attention, torch.Tensor), "encoded attention tensor")
    row_index = {str(row["row_id"]): index for index, row in enumerate(normalized)}
    for event in event_rows:
        index = row_index[str(event["row_id"])]
        anchor = int(event["absolute_anchor_token_index"])
        terminal = int(event["terminal_index"])
        require(int(input_ids[index, anchor].item()) == int(event["anchor_token_id"]), "anchor token-id mismatch")
        require(int(attention[index].sum().item()) == terminal + 1, "active terminal/attention mismatch")
        require(int(claim_mask[index].sum().item()) == int(event["claim_active_token_count"]), "claim active-count mismatch")
        require(int(evidence_mask[index].sum().item()) == int(event["evidence_active_token_count"]), "evidence active-count mismatch")
    event_ids = {str(row["row_id"]) for row in event_rows}
    require(event_ids == {str(row["row_id"]) for row in normalized}, "event/canonical row population mismatch")


def event_rows_by_row_id(event_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in event_rows:
        result[str(raw["row_id"])].append(dict(raw))
    return dict(result)


def support_indices_for_events(events: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    support: set[int] = set()
    for row in events:
        anchor = int(row["absolute_anchor_token_index"])
        terminal = int(row["terminal_index"])
        require(anchor >= 1, "anchor requires s_(a-1)")
        require(anchor + 4 <= terminal - 1, "active-terminal prefix rule")
        support.update(range(anchor - 1, anchor + 5))
    return tuple(sorted(support))


def canonical_pair_order(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    seen: set[str] = set()
    order: list[str] = []
    for row in rows:
        pair = str(row["source_pair_id"])
        if pair not in seen:
            seen.add(pair)
            order.append(pair)
    require(len(order) == SOURCE_PAIR_COUNT, "canonical pair order count")
    return tuple(order)


def canonical_row_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        key = (str(raw["source_pair_id"]), str(raw["contrast_cell_id"]))
        require(key not in mapping, "duplicate canonical pair/cell")
        mapping[key] = dict(raw)
    return mapping


def ordered_events(
    canonical_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs = canonical_pair_order(canonical_rows)
    by_key = {
        (str(row["source_pair_id"]), str(row["anchor_name"]), str(row["contrast_cell_id"])): dict(row)
        for row in event_rows
    }
    expected_size = sum(len(REQUIRED_ANCHOR_CELLS[a]) for a in ANCHOR_ORDER) * len(pairs)
    require(len(by_key) == expected_size, "event identity cardinality")
    result: list[dict[str, Any]] = []
    for pair in pairs:
        for anchor in ANCHOR_ORDER:
            for cell in CANONICAL_CELLS:
                if cell not in REQUIRED_ANCHOR_CELLS[anchor]:
                    continue
                key = (pair, anchor, cell)
                require(key in by_key, f"missing event {key}")
                result.append(by_key[key])
    require(len(result) == REQUIRED_ANCHOR_CELL_ROWS, "ordered event count")
    return result


def build_support_plan(
    canonical_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], int]]:
    by_row = event_rows_by_row_id(event_rows)
    plan: list[dict[str, Any]] = []
    index_map: dict[tuple[str, int], int] = {}
    vector_index = 0
    for row in canonical_rows:
        row_id = str(row["row_id"])
        require(row_id in by_row, "missing row event plan")
        terminals = {int(event["terminal_index"]) for event in by_row[row_id]}
        require(len(terminals) == 1, "row active-terminal inconsistency")
        for token_index in support_indices_for_events(by_row[row_id]):
            key = (row_id, token_index)
            require(key not in index_map, "duplicate support coordinate")
            index_map[key] = vector_index
            plan.append({
                "schema_version": SUPPORT_ROW_SCHEMA,
                "source_pair_id": str(row["source_pair_id"]),
                "row_id": row_id,
                "contrast_cell_id": str(row["contrast_cell_id"]),
                "absolute_token_index": token_index,
                "active_terminal_index": int(by_row[row_id][0]["terminal_index"]),
                "layer_index": PRIMARY_LAYER,
                "state_source": measurement.STATE_SOURCE,
                "state_timing": measurement.STATE_TIMING,
                "tensor_shape": list(measurement.PER_EXAMPLE_STATE_SHAPE),
                "flattened_size": STATE_VECTOR_SIZE,
                "vector_index": vector_index,
            })
            vector_index += 1
    return plan, index_map


def validate_support_plan(plan: Sequence[Mapping[str, Any]]) -> None:
    for expected_index, row in enumerate(plan):
        require(row["vector_index"] == expected_index, "support vector index ordering")
        require(row["layer_index"] == PRIMARY_LAYER, "support layer")
        require(row["flattened_size"] == STATE_VECTOR_SIZE, "support flattened size")
        require(row["tensor_shape"] == list(measurement.PER_EXAMPLE_STATE_SHAPE), "support tensor shape")


def endpoint_from_states(states: Sequence[Any], event: Mapping[str, Any]) -> dict[str, float]:
    anchor = int(event["absolute_anchor_token_index"])
    terminal = int(event["terminal_index"])
    require(anchor + 4 <= terminal - 1, "active-terminal prefix rule")
    try:
        result = measurement.post4_kinematics(states, anchor)
    except measurement.ContractError:
        raise
    except Exception as exc:
        raise ExtractionContractError("kinematic computation failed") from exc
    for key in ("POST4_SPEED", "POST4_TURNING", "POST4_PATH_EFFICIENCY"):
        value = result[key]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ExtractionContractError(BLOCKED_NONFINITE_MEASUREMENT)
    return {key: float(result[key]) for key in result}


def endpoint_row(
    event: Mapping[str, Any],
    support_index_map: Mapping[tuple[str, int], int],
    result: Mapping[str, float],
) -> dict[str, Any]:
    row_id = str(event["row_id"])
    anchor = int(event["absolute_anchor_token_index"])
    support_tokens = list(range(anchor - 1, anchor + 5))
    indices = []
    for token in support_tokens:
        key = (row_id, token)
        require(key in support_index_map, "endpoint support reference missing")
        indices.append(int(support_index_map[key]))
    return {
        "schema_version": ENDPOINT_ROW_SCHEMA,
        "source_pair_id": str(event["source_pair_id"]),
        "row_id": row_id,
        "contrast_cell_id": str(event["contrast_cell_id"]),
        "anchor_name": str(event["anchor_name"]),
        "anchor_token_index": anchor,
        "active_terminal_index": int(event["terminal_index"]),
        "layer_index": PRIMARY_LAYER,
        "support_absolute_token_indices": support_tokens,
        "support_vector_indices": indices,
        "POST4_SPEED": float(result["POST4_SPEED"]),
        "POST4_TURNING": float(result["POST4_TURNING"]),
        "POST4_PATH_EFFICIENCY": float(result["POST4_PATH_EFFICIENCY"]),
    }


def validate_endpoint_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == REQUIRED_ANCHOR_CELL_ROWS, "endpoint row count")
    forbidden = (
        "delta_", "interaction", "p_value", "pvalue", "confidence", "effect_size",
        "holm", "multiplicity", "title_minus_name", "support_verdict",
    )
    for row in rows:
        lower_keys = [str(key).lower() for key in row]
        require(not any(any(token in key for token in forbidden) for key in lower_keys), "statistical field prohibited")
        require(row["layer_index"] == PRIMARY_LAYER, "endpoint layer")
        support = list(row["support_absolute_token_indices"])
        anchor = int(row["anchor_token_index"])
        require(support == list(range(anchor - 1, anchor + 5)), "endpoint support window")
        require(len(row["support_vector_indices"]) == 6, "endpoint support refs")
        for key in ("POST4_SPEED", "POST4_TURNING", "POST4_PATH_EFFICIENCY"):
            require(math.isfinite(float(row[key])), BLOCKED_NONFINITE_MEASUREMENT)


def reconstruct_endpoint_from_support(
    endpoint: Mapping[str, Any],
    support_states: np.ndarray,
) -> dict[str, float]:
    vectors = [np.asarray(support_states[int(index)], dtype=np.float32) for index in endpoint["support_vector_indices"]]
    # Local six-state window corresponds to global [a-1, ..., a+4]. Compute the
    # same frozen definitions directly; this validator does not relax blockers.
    transitions = [np.ascontiguousarray(vectors[i] - vectors[i - 1], dtype=np.float32) for i in range(1, 6)]
    speed_values = [float(np.linalg.norm(transitions[i])) for i in range(1, 5)]
    speed = float(np.mean(np.asarray(speed_values, dtype=np.float32)))
    turning_values = [measurement.transition_turning(transitions[i], transitions[i - 1]) for i in range(1, 5)]
    turning = float(np.mean(np.asarray(turning_values, dtype=np.float32)))
    denominator = float(np.sum(np.asarray(speed_values, dtype=np.float32)))
    if denominator == 0.0:
        raise measurement.ContractError(measurement.BLOCKED_ZERO_PATH)
    displacement = float(np.linalg.norm(np.ascontiguousarray(vectors[5] - vectors[1], dtype=np.float32)))
    path_efficiency = displacement / denominator
    result = {
        "POST4_SPEED": speed,
        "POST4_TURNING": turning,
        "POST4_PATH_EFFICIENCY": float(path_efficiency),
    }
    require(all(math.isfinite(value) for value in result.values()), BLOCKED_NONFINITE_MEASUREMENT)
    return result


def exact_float32_equal(left: float, right: float) -> bool:
    return np.asarray(left, dtype=np.float32).tobytes() == np.asarray(right, dtype=np.float32).tobytes()


def validate_endpoint_reconstruction(endpoint_rows: Sequence[Mapping[str, Any]], support_states: np.ndarray) -> None:
    for row in endpoint_rows:
        reconstructed = reconstruct_endpoint_from_support(row, support_states)
        for key, value in reconstructed.items():
            require(exact_float32_equal(value, float(row[key])), f"endpoint reconstruction mismatch: {key}")


def canonical_npy_array(path: str | Path) -> np.ndarray:
    value = np.load(Path(path), allow_pickle=False, mmap_mode="r")
    require(value.ndim == 2, "support_states rank")
    require(value.shape[1] == STATE_VECTOR_SIZE, "support_states vector size")
    require(value.dtype == np.dtype("<f4"), "support_states dtype")
    require(value.flags.c_contiguous, "support_states contiguity")
    require(np.isfinite(value).all(), "support_states nonfinite")
    return value


def write_support_states(path: Path, vectors: Sequence[np.ndarray]) -> None:
    array = np.empty((len(vectors), STATE_VECTOR_SIZE), dtype="<f4")
    for index, vector in enumerate(vectors):
        item = np.asarray(vector, dtype="<f4")
        require(item.shape == (STATE_VECTOR_SIZE,), "support vector shape")
        require(np.isfinite(item).all(), "support vector nonfinite")
        array[index] = item
    np.save(path, array, allow_pickle=False)


def artifact_checksums(directory: Path) -> list[tuple[str, str]]:
    result = []
    for name in REQUIRED_ARTIFACTS[:-1]:
        path = directory / name
        require(path.is_file(), f"missing artifact: {name}")
        result.append((name, sha256_file(path)))
    return result


def checksum_bytes(checksums: Sequence[tuple[str, str]]) -> bytes:
    return "".join(f"{digest}  {name}\n" for name, digest in checksums).encode("utf-8")


def validate_checksum_file(directory: Path) -> None:
    expected = checksum_bytes(artifact_checksums(directory))
    observed = (directory / "SHA256SUMS.txt").read_bytes()
    require(observed == expected, "checksum file mismatch")


def validate_bundle(directory: Path) -> None:
    require(directory.is_dir(), "bundle directory missing")
    require(tuple(sorted(path.name for path in directory.iterdir())) == tuple(sorted(REQUIRED_ARTIFACTS)), "artifact set mismatch")
    support_rows = read_jsonl(directory / "support_state_rows.jsonl")
    endpoints = read_jsonl(directory / "kinematic_endpoints.jsonl")
    states = canonical_npy_array(directory / "support_states.npy")
    require(len(support_rows) == states.shape[0], "support metadata/vector count mismatch")
    validate_support_plan(support_rows)
    validate_endpoint_rows(endpoints)
    validate_endpoint_reconstruction(endpoints, states)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema_version"] == MANIFEST_SCHEMA, "manifest schema")
    require(manifest["support_state_vector_count"] == states.shape[0], "manifest support count")
    require(manifest["endpoint_row_count"] == REQUIRED_ANCHOR_CELL_ROWS, "manifest endpoint count")
    require(manifest["model_replication_count"] == MODEL_REPLICATION_COUNT, "manifest replication count")
    require(manifest["model_forward_mode"] == "backbone_only", "manifest model forward mode")
    require(manifest["downstream_forward_count"] == 0, "manifest downstream forward")
    require(manifest["training"] is False and manifest["backward"] is False and manifest["statistical_testing"] is False, "manifest scientific boundary")
    require(manifest["scientific_conclusion"] == "NONE", "manifest scientific conclusion")
    validate_checksum_file(directory)


def transactional_publish(
    final_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    support_rows: Sequence[Mapping[str, Any]],
    support_vectors: Sequence[np.ndarray],
    endpoint_rows: Sequence[Mapping[str, Any]],
) -> None:
    final_dir = Path(final_dir)
    staging = final_dir.with_name(final_dir.name + ".staging")
    require(not final_dir.exists(), "output collision")
    require(not staging.exists(), "staging collision")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "manifest.json").write_bytes(canonical_json_bytes(dict(manifest)))
        (staging / "support_state_rows.jsonl").write_bytes(jsonl_bytes(support_rows))
        write_support_states(staging / "support_states.npy", support_vectors)
        (staging / "kinematic_endpoints.jsonl").write_bytes(jsonl_bytes(endpoint_rows))
        checksums = artifact_checksums(staging)
        (staging / "SHA256SUMS.txt").write_bytes(checksum_bytes(checksums))
        validate_bundle(staging)
        staging.replace(final_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_scientific_inputs(tokenizer_snapshot: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    event_rows = load_event_manifest()
    rows = adapter.load_canonical_gen4_artifact()
    tokenizer = adapter.load_canonical_tokenizer(tokenizer_snapshot)
    encoded = adapter.encode_gen4_rows(rows, tokenizer)
    validate_canonical_population(rows, encoded, event_rows)
    return rows, encoded, event_rows


def load_representative_model(model_snapshot: str | Path) -> tuple[Any, str]:
    require(adapter.expected_checkpoint_sha256(REPRESENTATIVE_SEED, REPRESENTATIVE_ARM) == REPRESENTATIVE_CHECKPOINT_SHA256, "representative registry mismatch")
    require(r5.expected_checkpoint_path(REPRESENTATIVE_SEED, REPRESENTATIVE_ARM).resolve() == (ROOT / REPRESENTATIVE_CHECKPOINT).resolve(), "representative path mismatch")
    return r5.load_evaluator_model(
        seed=REPRESENTATIVE_SEED,
        arm=REPRESENTATIVE_ARM,
        model_snapshot=model_snapshot,
        device=torch.device("cpu"),
    )


def extract_row_states(model: Any, input_ids: torch.Tensor) -> list[Any]:
    require(tuple(input_ids.shape) == (1, MAX_MODEL_SEQUENCE_LENGTH), "scientific batch/sequence shape")
    require(input_ids.device.type == "cpu", "input device")
    layers = getattr(model.mamba, "layers", None)
    require(layers is not None and len(layers) == measurement.NATIVE_MAMBA_LAYER_COUNT, "Mamba layer count")
    mixer = layers[PRIMARY_LAYER].mixer
    observer = measurement.NativeStateObserver({id(mixer): {"layer_index": PRIMARY_LAYER}}, enabled=True)
    model.mamba.eval()
    with observer.capture():
        with torch.inference_mode():
            _ = model.mamba(input_ids=input_ids)
    require(observer.snapshots is not None, "observer produced no snapshots")
    return measurement.validate_capture_coordinates(observer.snapshots, token_count=MAX_MODEL_SEQUENCE_LENGTH)


def run_scientific_extraction(
    *,
    output_dir: str | Path,
    expected_execution_head: str,
    runner_implementation_commit: str,
    expected_runner_sha256: str,
    execution_authority_commit: str,
    model_snapshot: str | Path | None = None,
    tokenizer_snapshot: str | Path | None = None,
    runtime_gate: Callable[[], None] = measurement.runtime_gate,
    input_loader: Callable[[str | Path | None], tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]] = load_scientific_inputs,
    model_loader: Callable[[str | Path], tuple[Any, str]] = load_representative_model,
    row_extractor: Callable[[Any, torch.Tensor], list[Any]] = extract_row_states,
) -> dict[str, Any]:
    head = validate_execution_binding(
        expected_execution_head=expected_execution_head,
        runner_implementation_commit=runner_implementation_commit,
        expected_runner_sha256=expected_runner_sha256,
        execution_authority_commit=execution_authority_commit,
    )
    final_dir = Path(output_dir)
    staging = final_dir.with_name(final_dir.name + ".staging")
    require(not final_dir.exists(), "output collision")
    require(not staging.exists(), "staging collision")

    # Must occur before checkpoint deserialization/model construction/forward.
    runtime_gate()
    rows, encoded, event_rows = input_loader(tokenizer_snapshot)
    support_plan, support_index_map = build_support_plan(rows, event_rows)
    validate_support_plan(support_plan)
    events_in_order = ordered_events(rows, event_rows)
    if model_snapshot is None:
        model_snapshot = r5.canonical_model_snapshot_dir()
    model, checkpoint_sha = model_loader(model_snapshot)
    require(checkpoint_sha == REPRESENTATIVE_CHECKPOINT_SHA256, "representative checkpoint identity")
    require(not any(parameter.requires_grad for parameter in model.mamba.parameters()), "Mamba unexpectedly trainable")
    event_by_row = event_rows_by_row_id(event_rows)
    endpoint_results: dict[tuple[str, str], dict[str, float]] = {}
    forward_count = 0
    input_ids = encoded["input_ids"]
    require(isinstance(input_ids, torch.Tensor), "encoded input_ids tensor")
    require(tuple(input_ids.shape) == (CANONICAL_GEN4_ROWS, MAX_MODEL_SEQUENCE_LENGTH), "encoded input shape")

    staging.mkdir(parents=True, exist_ok=False)
    support_path = staging / "support_states.npy"
    support_mm = None
    try:
        support_mm = np.lib.format.open_memmap(
            support_path,
            mode="w+",
            dtype="<f4",
            shape=(len(support_plan), STATE_VECTOR_SIZE),
        )
        for row_index, row in enumerate(rows):
            row_id = str(row["row_id"])
            states = row_extractor(model, input_ids[row_index : row_index + 1].to("cpu"))
            forward_count += 1
            require(len(states) == MAX_MODEL_SEQUENCE_LENGTH, "captured state count")
            # Kinematics receives full sequence vectors so the frozen Phase C prefix
            # validator stays intact. Only [a-1,a+4] contributes numerically.
            full_vectors = [measurement.flatten_scientific_state(state) for state in states]
            for token_index in support_indices_for_events(event_by_row[row_id]):
                vector_index = support_index_map[(row_id, token_index)]
                support_mm[vector_index] = np.asarray(full_vectors[token_index], dtype="<f4")
            for event in event_by_row[row_id]:
                result = endpoint_from_states(full_vectors, event)
                endpoint_results[(row_id, str(event["anchor_name"]))] = result
        require(forward_count == EXPECTED_BACKBONE_FORWARD_COUNT, "scientific forward count")
        support_mm.flush()
        del support_mm
        support_mm = None

        endpoints: list[dict[str, Any]] = []
        for event in events_in_order:
            key = (str(event["row_id"]), str(event["anchor_name"]))
            require(key in endpoint_results, "missing endpoint result")
            endpoints.append(endpoint_row(event, support_index_map, endpoint_results[key]))
        validate_endpoint_rows(endpoints)

        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "phase_d_execution_authority_commit": execution_authority_commit,
            "phase_d_runner_implementation_commit": runner_implementation_commit,
            "runner_path": RUNNER_PATH.as_posix(),
            "runner_sha256": expected_runner_sha256,
            "runner_bytes": len(git_show_bytes(runner_implementation_commit, RUNNER_PATH)),
            "phase_d_runner_implementation_authority": PHASE_D_RUNNER_IMPLEMENTATION_AUTHORITY,
            "source_role_validator_correction_authority": SOURCE_ROLE_VALIDATOR_CORRECTION_AUTHORITY,
            "phase_c_measurement_implementation_commit": PHASE_C_MEASUREMENT_IMPLEMENTATION_COMMIT,
            "phase_c_measurement_path": MEASUREMENT_PATH.as_posix(),
            "phase_c_measurement_sha256": MEASUREMENT_SHA256,
            "phase_c_implementation_authority": PHASE_C_IMPLEMENTATION_AUTHORITY,
            "phase_ab_feasibility_freeze": PHASE_AB_FEASIBILITY_FREEZE,
            "mechanistic_bridge_specification": MECHANISTIC_BRIDGE_SPECIFICATION,
            "execution_head": head,
            "runtime_versions": {
                "python": ".".join(map(str, sys.version_info[:3])),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
            },
            "mamba_source_sha256": measurement.MAMBA_SHA256,
            "mamba_source_bytes": measurement.MAMBA_BYTES,
            "cache_source_sha256": measurement.CACHE_SHA256,
            "cache_source_bytes": measurement.CACHE_BYTES,
            "representative_checkpoint": REPRESENTATIVE_CHECKPOINT.as_posix(),
            "representative_checkpoint_sha256": checkpoint_sha,
            "representative_seed": REPRESENTATIVE_SEED,
            "representative_arm": REPRESENTATIVE_ARM,
            "model_config_revision": TOKENIZER_REVISION_REFERENCE,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "model_config_bytes": MODEL_CONFIG_BYTES,
            "tokenizer_revision_reference": TOKENIZER_REVISION_REFERENCE,
            "tokenizer_file_sha256": dict(adapter.TOKENIZER_FILE_SHA256),
            "canonical_gen4_artifact": CANONICAL_GEN4_ARTIFACT.as_posix(),
            "canonical_gen4_artifact_sha256": CANONICAL_GEN4_ARTIFACT_SHA256,
            "canonical_gen4_artifact_bytes": CANONICAL_GEN4_ARTIFACT_BYTES,
            "event_anchor_manifest": EVENT_MANIFEST.as_posix(),
            "event_anchor_manifest_sha256": EVENT_MANIFEST_SHA256,
            "event_anchor_manifest_bytes": EVENT_MANIFEST_BYTES,
            "encoded_coordinate_sha256": R2_ENCODED_COORDINATE_SHA256,
            "source_pair_count": SOURCE_PAIR_COUNT,
            "canonical_row_count": CANONICAL_GEN4_ROWS,
            "cell_count": len(CANONICAL_CELLS),
            "support_state_vector_count": len(support_plan),
            "endpoint_row_count": len(endpoints),
            "layer_index": PRIMARY_LAYER,
            "scientific_forward_count": forward_count,
            "model_replication_count": MODEL_REPLICATION_COUNT,
            "model_forward_mode": "backbone_only",
            "downstream_forward_count": 0,
            "training": False,
            "backward": False,
            "statistical_testing": False,
            "scientific_conclusion": "NONE",
            "blocker": None,
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        (staging / "support_state_rows.jsonl").write_bytes(jsonl_bytes(support_plan))
        (staging / "kinematic_endpoints.jsonl").write_bytes(jsonl_bytes(endpoints))
        checksums = artifact_checksums(staging)
        (staging / "SHA256SUMS.txt").write_bytes(checksum_bytes(checksums))
        validate_bundle(staging)
        staging.replace(final_dir)
        return manifest
    except Exception:
        if support_mm is not None:
            try:
                support_mm.flush()
            except Exception:
                pass
            del support_mm
        if staging.exists():
            shutil.rmtree(staging)
        raise

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--runner-implementation-commit", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--execution-authority-commit", required=True)
    parser.add_argument("--model-snapshot", default=None)
    parser.add_argument("--tokenizer-snapshot", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_scientific_extraction(
        output_dir=args.output_dir,
        expected_execution_head=args.expected_execution_head,
        runner_implementation_commit=args.runner_implementation_commit,
        expected_runner_sha256=args.expected_runner_sha256,
        execution_authority_commit=args.execution_authority_commit,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
    )


if __name__ == "__main__":
    main()
