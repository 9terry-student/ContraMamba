"""Fail-closed future Gen4 NAME Q1/Q3 extraction implementation."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts import reason_router_gen4_native_mamba_state_q1_q3_measurement as measurement


IMPLEMENTATION_AUTHORITY_COMMIT = "c395e634448a3b37c30053d89abef37c5a269afe"
INPUT_COORDINATE_CORRECTION_COMMIT = "b9d26005b2e475d9e2645c507eedcbd0c79bbac3"
FEASIBILITY_PROVENANCE_COMMIT = "c7d841a920a0a6d075f7f3804da9212ce4706673"
SCIENTIFIC_SPECIFICATION_COMMIT = "01801ad1617b2ebc3ffa859ba440636d4755a55c"
DEPTH_INDEX_SPECIFICATION_COMMIT = "ecbf6ba720c0e173ac7089a00bb5b783aa16fa6c"
PHASE_F_RESULT_FREEZE_COMMIT = "ab3428e7be08af26fa1fdafd1483a34e48fbcf8c"

PRIMARY_MEASUREMENT_SHA256 = (
    "7729424f03058b86b4f120dc0e6da573"
    "d6c996b0877858f2d6d38aa94dac268c"
)
R2_ENCODED_COORDINATE_SHA256 = (
    "d3cf61e55bb04e6dbebe44be433597c"
    "ce958cb15989412dcf6efc15a329c576a"
)
STRUCTURAL_SHA256 = (
    "b9c54604863ed15c237fa17c7890a20"
    "f3f5ec062a7429638b39e8b468be050a7"
)
STRUCTURAL_BYTES = 1465573
EVENT_SHA256 = (
    "70c84c68b36751bb7c7145b33ccb71ab"
    "91bc8ee9e6cc5f2c7a0d4e925f36581f"
)
EVENT_BYTES = 2268260

TOKENIZER_REVISION_REFERENCE = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
TOKENIZER_FILE_SHA256 = {
    "tokenizer.json": "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf",
    "tokenizer_config.json": "9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb",
    "special_tokens_map.json": "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
}

EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "torch": "2.10.0+cpu",
    "transformers": "5.0.0",
}
MAMBA_SOURCE_SHA256 = "4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83"
MAMBA_SOURCE_BYTES = 39500
CACHE_SOURCE_SHA256 = "6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc"
CACHE_SOURCE_BYTES = 60432

REPRESENTATIVE_SEED = 180
REPRESENTATIVE_ARM = "G3-GROUP-D-HALF"
REPRESENTATIVE_CHECKPOINT = (
    "reports/reason_router_gen3_grouped_factorial_runs/"
    "seed180/G3-GROUP-D-HALF/selected_checkpoint.pt"
)
REPRESENTATIVE_CHECKPOINT_SHA256 = "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"

LAYERS = (5, 17)
CELLS = ("C0_SHAM", "C2_NAME")
SOURCE_PAIRS = 300
ROWS = 600
FULL_COORDINATE_ROWS = 1800
MODEL_SEQUENCE_LENGTH = 128
EXPECTED_BACKBONE_FORWARD_COUNT = 600
SUPPORT_ROWS = 7200
ENDPOINT_ROWS = 1200
SUPPORT_VECTOR_SIZE = 24576

REQUIRED_ARTIFACTS = (
    "manifest.json",
    "support_state_rows.jsonl",
    "support_states.npy",
    "kinematic_endpoints.jsonl",
    "SHA256SUMS.txt",
)

ENCODED_KEYS = (
    "row_id",
    "source_pair_id",
    "contrast_cell_id",
    "input_ids",
    "attention_mask",
    "claim_mask",
    "evidence_mask",
)

SUPPORT_ROW_KEYS = {
    "source_pair_id",
    "row_id",
    "contrast_cell_id",
    "semantic_anchor",
    "anchor_token_index",
    "layer_index",
    "support_token_index",
    "tensor_row_index",
}

ENDPOINT_ROW_KEYS = {
    "source_pair_id",
    "row_id",
    "contrast_cell_id",
    "semantic_anchor",
    "anchor_token_index",
    "layer_index",
    "support_absolute_token_indices",
    "support_tensor_row_indices",
    "POST4_SPEED",
    "POST4_TURNING",
    "POST4_PATH_EFFICIENCY",
}

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ExtractionContractError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ExtractionContractError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def read_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"non-object JSONL row: {line_no}")
        rows.append(value)
    return rows


def _to_list(value: Any) -> Any:
    detached = value.detach() if hasattr(value, "detach") else value
    cpu = detached.cpu() if hasattr(detached, "cpu") else detached
    if hasattr(cpu, "tolist"):
        return cpu.tolist()
    return np.asarray(cpu).tolist()


def validate_encoded_coordinate(
    encoded: Mapping[str, Any],
    *,
    expected_rows: int = FULL_COORDINATE_ROWS,
) -> None:
    require(isinstance(encoded, Mapping), "encoded coordinate mapping")
    require(set(encoded) == set(ENCODED_KEYS), "encoded coordinate schema")

    for key in ("row_id", "source_pair_id", "contrast_cell_id"):
        require(len(encoded[key]) == expected_rows, f"encoded coordinate length: {key}")

    for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
        value = encoded[key]
        shape = tuple(getattr(value, "shape", np.asarray(value).shape))
        require(
            shape == (expected_rows, MODEL_SEQUENCE_LENGTH),
            f"encoded coordinate shape: {key}",
        )

    row_ids = [str(value) for value in encoded["row_id"]]
    pair_ids = [str(value) for value in encoded["source_pair_id"]]
    cells = [str(value) for value in encoded["contrast_cell_id"]]
    require(len(set(row_ids)) == expected_rows, "encoded duplicate row_id")
    require(
        len(set(zip(pair_ids, cells))) == expected_rows,
        "encoded duplicate pair/cell",
    )


def encoded_coordinate_sha256(encoded: Mapping[str, Any]) -> str:
    """Exact R2 aggregate identity over row ids, input ids, and all masks."""
    validate_encoded_coordinate(encoded)
    serializable: list[dict[str, Any]] = []

    for index in range(len(encoded["row_id"])):
        serializable.append({
            "row_id": str(encoded["row_id"][index]),
            "source_pair_id": str(encoded["source_pair_id"][index]),
            "contrast_cell_id": str(encoded["contrast_cell_id"][index]),
            "input_ids": _to_list(encoded["input_ids"][index]),
            "attention_mask": _to_list(encoded["attention_mask"][index]),
            "claim_mask": _to_list(encoded["claim_mask"][index]),
            "evidence_mask": _to_list(encoded["evidence_mask"][index]),
        })

    return sha256_bytes(canonical_json_bytes(serializable))


def validate_structural_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == FULL_COORDINATE_ROWS, "full 1800 structural rows")
    identities: set[tuple[str, str]] = set()
    cell_counts: Counter[str] = Counter()
    source_pairs: set[str] = set()

    for index, row in enumerate(rows):
        require(isinstance(row, Mapping), f"structural row mapping: {index}")
        for key in ("row_id", "source_pair_id", "contrast_cell_id"):
            require(key in row and str(row[key]), f"structural identity: {key}")

        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        identity = (pair, cell)
        require(identity not in identities, "structural duplicate pair/cell")
        identities.add(identity)
        source_pairs.add(pair)
        cell_counts[cell] += 1

    require(len(source_pairs) == SOURCE_PAIRS, "structural source-pair count")
    require(
        cell_counts == Counter({
            "C0_SHAM": 300,
            "C1_TITLE": 300,
            "C2_NAME": 300,
            "C3_ROLE": 300,
            "C4_PREDICATE": 300,
            "C5_TITLE_NAME": 300,
        }),
        "structural cell counts",
    )


def authenticate_structural_artifact(path: str | Path) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    require(len(raw) == STRUCTURAL_BYTES, "structural artifact byte-count")
    require(sha256_bytes(raw) == STRUCTURAL_SHA256, "structural artifact identity mismatch")
    rows = read_jsonl_bytes(raw)
    validate_structural_rows(rows)
    return rows


def authenticate_event_manifest(path: str | Path) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    require(len(raw) == EVENT_BYTES, "event manifest byte-count")
    require(sha256_bytes(raw) == EVENT_SHA256, "event manifest identity mismatch")
    return read_jsonl_bytes(raw)


def authenticate_tokenizer_snapshot(snapshot_dir: str | Path) -> dict[str, str]:
    directory = Path(snapshot_dir)
    require(directory.name == TOKENIZER_REVISION_REFERENCE, "tokenizer revision")
    observed: dict[str, str] = {}

    for filename, expected in TOKENIZER_FILE_SHA256.items():
        path = directory / filename
        require(path.is_file(), f"tokenizer file missing: {filename}")
        digest = sha256_file(path)
        require(digest == expected, f"tokenizer file identity: {filename}")
        observed[filename] = digest

    return observed


def validate_encoded_row_identity(
    rows: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
) -> None:
    validate_structural_rows(rows)
    validate_encoded_coordinate(encoded)

    for index, row in enumerate(rows):
        require(str(encoded["row_id"][index]) == str(row["row_id"]), "encoded row_id ordering")
        require(
            str(encoded["source_pair_id"][index]) == str(row["source_pair_id"]),
            "encoded source_pair_id ordering",
        )
        require(
            str(encoded["contrast_cell_id"][index]) == str(row["contrast_cell_id"]),
            "encoded contrast_cell_id ordering",
        )


def reconstruct_then_select(
    rows: Sequence[Mapping[str, Any]],
    reconstruct: Callable[[], Mapping[str, Any]],
    *,
    expected_sha256: str = R2_ENCODED_COORDINATE_SHA256,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """Validate the complete frozen coordinate before any Q1/Q3 subset work."""
    validate_structural_rows(rows)
    encoded = reconstruct()
    validate_encoded_coordinate(encoded)
    require(encoded_coordinate_sha256(encoded) == expected_sha256, "encoded coordinate identity mismatch")
    validate_encoded_row_identity(rows, encoded)

    selected = [row for row in rows if row["contrast_cell_id"] in CELLS]
    validate_subset(selected)
    return selected, encoded


def prepare_after_coordinate_gate(
    rows: Sequence[Mapping[str, Any]],
    reconstruct: Callable[[], Mapping[str, Any]],
    model_constructor: Callable[[], Any],
    *,
    expected_sha256: str = R2_ENCODED_COORDINATE_SHA256,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], Any]:
    selected, encoded = reconstruct_then_select(
        rows,
        reconstruct,
        expected_sha256=expected_sha256,
    )
    model = model_constructor()
    return selected, encoded, model


def validate_subset(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == ROWS, "Q1/Q3 subset row count")
    pairs: dict[str, set[str]] = {}
    row_ids: set[str] = set()

    for row in rows:
        pair = str(row.get("source_pair_id"))
        row_id = str(row.get("row_id"))
        cell = str(row.get("contrast_cell_id"))
        require(cell in CELLS, "Q1/Q3 cell set")
        require(row_id not in row_ids, "duplicate Q1/Q3 row_id")
        row_ids.add(row_id)
        pairs.setdefault(pair, set()).add(cell)

    require(
        len(pairs) == SOURCE_PAIRS and all(cells == set(CELLS) for cells in pairs.values()),
        "Q1/Q3 pair completeness",
    )


def bind_events(
    selected: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    validate_subset(selected)
    selected_by_id = {str(row["row_id"]): row for row in selected}
    found: dict[str, Mapping[str, Any]] = {}

    for event in events:
        row_id = str(event.get("row_id"))
        if row_id not in selected_by_id or event.get("anchor_name") != "A_NAME":
            continue

        row = selected_by_id[row_id]
        require(row_id not in found, "duplicate A_NAME event")
        require(str(event.get("source_pair_id")) == str(row["source_pair_id"]), "A_NAME source-pair binding")
        require(str(event.get("contrast_cell_id")) == str(row["contrast_cell_id"]), "A_NAME contrast-cell binding")

        anchor = event.get("absolute_anchor_token_index")
        terminal = event.get("terminal_index")
        require(type(anchor) is int and anchor >= 1, "A_NAME anchor coordinate")
        require(type(terminal) is int, "A_NAME terminal coordinate")
        require(event.get("post4_eligible") is True, "A_NAME post4 eligibility")
        require(anchor + 4 <= terminal - 1, "A_NAME active-terminal prefix rule")
        found[row_id] = event

    require(set(found) == set(selected_by_id), "missing/incorrect A_NAME event binding")
    return found


def build_support_metadata(
    selected: Sequence[Mapping[str, Any]],
    events: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_subset(selected)
    require(set(measurement.ALLOWED_SECONDARY_LAYERS) == set(LAYERS), "layer policy")
    result: list[dict[str, Any]] = []

    for row in selected:
        row_id = str(row["row_id"])
        require(row_id in events, "missing bound A_NAME event")
        anchor = events[row_id]["absolute_anchor_token_index"]

        for layer_index in LAYERS:
            for token_index in range(anchor - 1, anchor + 5):
                result.append({
                    "source_pair_id": str(row["source_pair_id"]),
                    "row_id": row_id,
                    "contrast_cell_id": str(row["contrast_cell_id"]),
                    "semantic_anchor": "A_NAME",
                    "anchor_token_index": anchor,
                    "layer_index": layer_index,
                    "support_token_index": token_index,
                    "tensor_row_index": len(result),
                })

    require(len(result) == SUPPORT_ROWS, "support-state count")
    validate_support_rows(result)
    return result


def validate_support_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == SUPPORT_ROWS, "support-state count")
    for index, row in enumerate(rows):
        require(set(row) == SUPPORT_ROW_KEYS, "support-state schema")
        require(row["semantic_anchor"] == "A_NAME", "support-state anchor")
        require(row["layer_index"] in LAYERS, "support-state layer")
        require(row["tensor_row_index"] == index, "support-state tensor ordering")
        require(
            row["support_token_index"] in range(row["anchor_token_index"] - 1, row["anchor_token_index"] + 5),
            "support-state window",
        )


def support_index_map(
    support_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int], int]:
    validate_support_rows(support_rows)
    result: dict[tuple[str, int, int], int] = {}
    for row in support_rows:
        key = (str(row["row_id"]), int(row["layer_index"]), int(row["support_token_index"]))
        require(key not in result, "duplicate support coordinate")
        result[key] = int(row["tensor_row_index"])
    return result


def build_endpoint_row(
    row: Mapping[str, Any],
    event: Mapping[str, Any],
    layer_index: int,
    result: Mapping[str, float],
    support_map: Mapping[tuple[str, int, int], int],
) -> dict[str, Any]:
    require(layer_index in LAYERS, "endpoint layer")
    anchor = event["absolute_anchor_token_index"]
    token_indices = list(range(anchor - 1, anchor + 5))
    row_id = str(row["row_id"])
    vector_indices = [support_map[(row_id, layer_index, token)] for token in token_indices]

    endpoint = {
        "source_pair_id": str(row["source_pair_id"]),
        "row_id": row_id,
        "contrast_cell_id": str(row["contrast_cell_id"]),
        "semantic_anchor": "A_NAME",
        "anchor_token_index": anchor,
        "layer_index": layer_index,
        "support_absolute_token_indices": token_indices,
        "support_tensor_row_indices": vector_indices,
        "POST4_SPEED": float(result["POST4_SPEED"]),
        "POST4_TURNING": float(result["POST4_TURNING"]),
        "POST4_PATH_EFFICIENCY": float(result["POST4_PATH_EFFICIENCY"]),
    }
    for key in ("POST4_SPEED", "POST4_TURNING", "POST4_PATH_EFFICIENCY"):
        require(np.isfinite(endpoint[key]), "nonfinite endpoint")
    return endpoint


def validate_endpoint_rows(
    rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
) -> None:
    require(len(rows) == ENDPOINT_ROWS, "endpoint row count")
    support_map = support_index_map(support_rows)
    identities: set[tuple[str, int]] = set()

    for row in rows:
        require(set(row) == ENDPOINT_ROW_KEYS, "endpoint schema")
        require(row["semantic_anchor"] == "A_NAME", "endpoint anchor")
        require(row["layer_index"] in LAYERS, "endpoint layer")
        identity = (str(row["row_id"]), int(row["layer_index"]))
        require(identity not in identities, "duplicate endpoint row")
        identities.add(identity)

        anchor = int(row["anchor_token_index"])
        expected_tokens = list(range(anchor - 1, anchor + 5))
        require(row["support_absolute_token_indices"] == expected_tokens, "endpoint support tokens")
        expected_rows = [
            support_map[(str(row["row_id"]), int(row["layer_index"]), token)]
            for token in expected_tokens
        ]
        require(row["support_tensor_row_indices"] == expected_rows, "endpoint support references")
        for key in ("POST4_SPEED", "POST4_TURNING", "POST4_PATH_EFFICIENCY"):
            require(np.isfinite(float(row[key])), "nonfinite endpoint")


def validate_execution_plan(
    selected: Sequence[Mapping[str, Any]],
    events: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_subset(selected)
    require(set(measurement.ALLOWED_SECONDARY_LAYERS) == set(LAYERS), "layer policy")
    return build_support_metadata(selected, events)


def execute_dual_layer_forward_plan(
    selected: Sequence[Mapping[str, Any]],
    events: Mapping[str, Mapping[str, Any]],
    encoded: Mapping[str, Any],
    forward_once: Callable[[Any], Mapping[int, Sequence[Any]]],
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], int]:
    """Future policy: 600 callbacks total, each callback returns both layers."""
    validate_subset(selected)
    validate_encoded_coordinate(encoded)
    support_rows = build_support_metadata(selected, events)
    support_map = support_index_map(support_rows)
    encoded_index = {str(row_id): index for index, row_id in enumerate(encoded["row_id"])}

    state_vectors: list[np.ndarray] = []
    endpoint_rows: list[dict[str, Any]] = []
    forward_count = 0

    for row in selected:
        row_id = str(row["row_id"])
        require(row_id in encoded_index, "selected row missing from encoded coordinate")
        layer_states = forward_once(encoded["input_ids"][encoded_index[row_id]])
        forward_count += 1
        require(set(layer_states) == set(LAYERS), "dual-layer forward result")
        event = events[row_id]
        anchor = event["absolute_anchor_token_index"]

        for layer_index in LAYERS:
            states = layer_states[layer_index]
            require(len(states) == MODEL_SEQUENCE_LENGTH, "captured token count")
            for token_index in range(anchor - 1, anchor + 5):
                vector = measurement.flatten_scientific_state(states[token_index])
                state_vectors.append(vector)
            endpoint_result = measurement.endpoint_from_same_layer(states, anchor, layer_index)
            endpoint_rows.append(
                build_endpoint_row(row, event, layer_index, endpoint_result, support_map)
            )

    require(forward_count == EXPECTED_BACKBONE_FORWARD_COUNT, "backbone forward count")
    require(len(state_vectors) == SUPPORT_ROWS, "support-state count")
    require(len(endpoint_rows) == ENDPOINT_ROWS, "endpoint row count")
    vectors = np.asarray(state_vectors, dtype=np.float32)
    return support_rows, vectors, endpoint_rows, forward_count


def authenticated_representative_checkpoint_load(
    path: str | Path,
    loader: Callable[[Path], Any],
) -> tuple[str, Any]:
    checkpoint = Path(path)
    normalized = checkpoint.as_posix()
    require(normalized.endswith(REPRESENTATIVE_CHECKPOINT), "representative checkpoint path")
    require(checkpoint.is_file(), "representative checkpoint missing")
    digest = sha256_file(checkpoint)
    require(digest == REPRESENTATIVE_CHECKPOINT_SHA256, "checkpoint identity before deserialization")
    payload = loader(checkpoint)
    return digest, payload


def build_manifest(
    *,
    implementation_commit: str,
    measurement_implementation_sha256: str,
    extraction_runner_sha256: str,
    runtime_versions: Mapping[str, str],
    peer_artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    require(_HEX40.fullmatch(implementation_commit) is not None, "implementation commit format")
    require(_HEX64.fullmatch(measurement_implementation_sha256) is not None, "measurement implementation SHA256 format")
    require(_HEX64.fullmatch(extraction_runner_sha256) is not None, "extraction runner SHA256 format")
    require(dict(runtime_versions) == EXPECTED_RUNTIME_VERSIONS, "runtime version contract")

    expected_peer_names = {
        "support_state_rows.jsonl",
        "support_states.npy",
        "kinematic_endpoints.jsonl",
    }
    require(set(peer_artifact_hashes) == expected_peer_names, "peer artifact hash schema")
    for digest in peer_artifact_hashes.values():
        require(_HEX64.fullmatch(str(digest)) is not None, "peer artifact SHA256 format")

    return {
        "schema_version": "gen4_name_q1_q3_native_mamba_state_extraction_manifest_v1",
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "input_coordinate_correction_commit": INPUT_COORDINATE_CORRECTION_COMMIT,
        "feasibility_provenance_commit": FEASIBILITY_PROVENANCE_COMMIT,
        "scientific_specification_commit": SCIENTIFIC_SPECIFICATION_COMMIT,
        "depth_index_specification_commit": DEPTH_INDEX_SPECIFICATION_COMMIT,
        "phase_f_result_freeze_commit": PHASE_F_RESULT_FREEZE_COMMIT,
        "implementation_commit": implementation_commit,
        "measurement_implementation_sha256": measurement_implementation_sha256,
        "extraction_runner_sha256": extraction_runner_sha256,
        "frozen_primary_measurement_sha256": PRIMARY_MEASUREMENT_SHA256,
        "structural_artifact_sha256": STRUCTURAL_SHA256,
        "event_manifest_sha256": EVENT_SHA256,
        "r2_encoded_coordinate_sha256": R2_ENCODED_COORDINATE_SHA256,
        "tokenizer_revision": TOKENIZER_REVISION_REFERENCE,
        "tokenizer_file_sha256": dict(TOKENIZER_FILE_SHA256),
        "runtime_versions": dict(runtime_versions),
        "mamba_source_sha256": MAMBA_SOURCE_SHA256,
        "mamba_source_bytes": MAMBA_SOURCE_BYTES,
        "cache_source_sha256": CACHE_SOURCE_SHA256,
        "cache_source_bytes": CACHE_SOURCE_BYTES,
        "representative_seed": REPRESENTATIVE_SEED,
        "representative_arm": REPRESENTATIVE_ARM,
        "representative_checkpoint": REPRESENTATIVE_CHECKPOINT,
        "representative_checkpoint_sha256": REPRESENTATIVE_CHECKPOINT_SHA256,
        "layer_set": list(LAYERS),
        "semantic_anchor": "A_NAME",
        "source_pair_count": SOURCE_PAIRS,
        "model_input_row_count": ROWS,
        "backbone_forward_count": EXPECTED_BACKBONE_FORWARD_COUNT,
        "support_state_row_count": SUPPORT_ROWS,
        "endpoint_row_count": ENDPOINT_ROWS,
        "peer_artifact_hashes": dict(peer_artifact_hashes),
    }


def write_future_bundle(
    output_dir: str | Path,
    *,
    support_rows: Sequence[Mapping[str, Any]],
    states: np.ndarray,
    endpoints: Sequence[Mapping[str, Any]],
    implementation_commit: str,
    measurement_implementation_sha256: str,
    extraction_runner_sha256: str,
    runtime_versions: Mapping[str, str],
) -> None:
    directory = Path(output_dir)
    staging = directory.with_name(directory.name + ".staging")
    require(not directory.exists(), "output collision")
    require(not staging.exists(), "staging collision")

    validate_support_rows(support_rows)
    validate_endpoint_rows(endpoints, support_rows)
    require(states.shape == (SUPPORT_ROWS, SUPPORT_VECTOR_SIZE), "support tensor shape")
    require(states.dtype == np.dtype("float32"), "support tensor dtype")
    require(np.isfinite(states).all(), "support tensor nonfinite")

    staging.mkdir(parents=True)
    try:
        support_path = staging / "support_state_rows.jsonl"
        states_path = staging / "support_states.npy"
        endpoint_path = staging / "kinematic_endpoints.jsonl"
        support_path.write_bytes(jsonl_bytes(support_rows))
        np.save(states_path, states, allow_pickle=False)
        endpoint_path.write_bytes(jsonl_bytes(endpoints))

        peer_hashes = {
            "support_state_rows.jsonl": sha256_file(support_path),
            "support_states.npy": sha256_file(states_path),
            "kinematic_endpoints.jsonl": sha256_file(endpoint_path),
        }
        manifest = build_manifest(
            implementation_commit=implementation_commit,
            measurement_implementation_sha256=measurement_implementation_sha256,
            extraction_runner_sha256=extraction_runner_sha256,
            runtime_versions=runtime_versions,
            peer_artifact_hashes=peer_hashes,
        )
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))

        checksum_names = (
            "manifest.json",
            "support_state_rows.jsonl",
            "support_states.npy",
            "kinematic_endpoints.jsonl",
        )
        checksums = b"".join(
            f"{sha256_file(staging / name)}  {name}\n".encode("utf-8")
            for name in checksum_names
        )
        (staging / "SHA256SUMS.txt").write_bytes(checksums)
        require(
            {path.name for path in staging.iterdir()} == set(REQUIRED_ARTIFACTS),
            "output artifact set",
        )
        staging.replace(directory)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
