#!/usr/bin/env python3
"""O0a native Mamba hidden-state proxy trajectory observer.

This is an observer-only future execution entry point.  It loads the native
``MambaModel`` backbone and matching tokenizer, performs inference-only prefix
forwards, and writes descriptive artifacts over Hugging Face layer hidden
states.  These are native pretrained Mamba hidden-state proxies, not the
selective SSM recurrent state, selective-scan state matrices, or direct
A/B/C/Delta dynamics.  It does not train, generate, or instantiate any
ContraMamba or reason-router component.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPERIMENT_NAME = "O0a — Native Mamba Entitlement-Sensitive Hidden-State Proxy Dynamics Screening"
SCHEMA_VERSION = "longterm_o0a_native_mamba_state_dynamics_v1"
MODEL_ID = "state-spaces/mamba-130m-hf"
MODEL_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TOKENIZER_ID = MODEL_ID
TOKENIZER_REVISION = MODEL_REVISION
MODEL_NAME = MODEL_ID
TOKENIZER_NAME = TOKENIZER_ID
AUTHORIZED_DEVICE = "cpu"
AUTHORIZED_DTYPE_NAME = "float32"
PREFIX_SCHEDULE = (0.00, 0.25, 0.50, 0.75, 1.00)
SERIALIZATION_TEMPLATE = "Claim: <claim>\nEvidence: <evidence-prefix>"
HIDDEN_STATE_PROXY_BOUNDARY = (
    "O0a measures Hugging Face MambaModel layer hidden states: native pretrained Mamba hidden-state proxies. "
    "They are not the selective SSM recurrent state, internal selective-scan state matrices, or direct "
    "A/B/C/Delta dynamics. Hugging Face cache_params and deeper selective-SSM recurrent-state instrumentation "
    "are out of scope for O0a and require separate O0b/O1 authority."
)
NORMALIZED_L2_COSINE_REDUNDANCY = (
    "For unit-normalized vectors, D_l2 squared equals 2 * D_cos; terminal normalized-L2 distance and terminal "
    "cosine distance are algebraically redundant coordinates and must not be counted as independent evidence."
)
PARAPHRASE_CONTROL_BOUNDARY = (
    "Toy paraphrase rows may alter both claim wording and evidence wording, so paraphrase is a whole-pair "
    "semantic-invariance / surface-form control, not a pure evidence-only paraphrase control. At the 0% evidence "
    "prefix, paraphrase distance can be nonzero because the claim itself may be paraphrased."
)
TRAJECTORY_SUMMARY_BOUNDARY = (
    "trajectory_summary_difference is retained only as an unweighted descriptive convenience diagnostic over "
    "heterogeneous scalar summaries; it has no independent inferential weight and is not a primary scientific score."
)
REQUIRED_INTERVENTIONS = frozenset(
    {
        "none",
        "paraphrase",
        "entity_swap",
        "predicate_swap",
        "evidence_deletion",
        "polarity_flip",
    }
)
ZERO_PREFIX_EQUALITY_INTERVENTIONS = frozenset(
    {"entity_swap", "predicate_swap", "evidence_deletion", "polarity_flip"}
)
REQUIRED_DATASET_FIELDS = frozenset(
    {
        "id",
        "pair_id",
        "claim",
        "evidence",
        "intervention_type",
        "primary_failure_type",
        "final_label",
    }
)
REQUIRED_ARTIFACTS = (
    "manifest.json",
    "observations.jsonl",
    "terminal_hidden_states.npz",
    "paired_distances.jsonl",
    "summary.json",
    "report.md",
    "SHA256SUMS.txt",
)

OBSERVATION_KEYS = (
    "schema_version",
    "row_id",
    "pair_id",
    "intervention_type",
    "primary_failure_type",
    "final_label",
    "requested_prefix_fraction",
    "requested_prefix_fractions",
    "actual_evidence_prefix_token_count",
    "full_evidence_token_count",
    "actual_fraction",
    "serialization_template",
    "serialized_input_text",
    "serialized_input_utf8_sha256",
    "input_token_ids",
    "vector_index",
    "layers",
)
LAYER_OBSERVATION_KEYS = (
    "layer_index",
    "layer_role",
    "state_source",
    "terminal_hidden_state_npz_index",
    "terminal_hidden_norm",
    "last_step_delta",
    "terminal_consecutive_state_cosine",
    "terminal_acceleration",
    "evidence_region_mean_consecutive_state_delta",
    "evidence_region_max_consecutive_state_delta",
)
PAIRED_DISTANCE_KEYS = (
    "schema_version",
    "pair_id",
    "row_id",
    "reference_row_id",
    "intervention_type",
    "requested_prefix_fraction",
    "actual_evidence_prefix_token_count",
    "reference_actual_evidence_prefix_token_count",
    "layer_index",
    "layer_role",
    "terminal_normalized_l2_distance",
    "terminal_cosine_distance",
    "transition_magnitude_difference",
    "trajectory_summary_difference",
)
TRAJECTORY_SUMMARY_METRICS = (
    "terminal_hidden_norm",
    "last_step_delta",
    "terminal_consecutive_state_cosine",
    "terminal_acceleration",
    "evidence_region_mean_consecutive_state_delta",
    "evidence_region_max_consecutive_state_delta",
)
MAMBA_CONFIG_FLAG_NAMES = (
    "model_type",
    "hidden_size",
    "state_size",
    "num_hidden_layers",
    "conv_kernel",
    "expand",
    "intermediate_size",
    "time_step_rank",
    "use_bias",
    "use_conv_bias",
    "hidden_act",
    "rms_norm",
    "residual_in_fp32",
    "layer_norm_epsilon",
    "initializer_range",
    "rescale_prenorm_residual",
)


class ContractError(RuntimeError):
    """Raised when an O0a fail-closed contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_head(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip().lower()
    require(len(head) == 40 and all(c in "0123456789abcdef" for c in head), "invalid repository HEAD")
    return head


def validate_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    require(
        len(normalized) == 64 and all(c in "0123456789abcdef" for c in normalized),
        f"{label} must be a 64-character SHA256",
    )
    return normalized


def validate_commit_sha(value: str, label: str) -> str:
    normalized = value.strip().lower()
    require(
        len(normalized) == 40 and all(c in "0123456789abcdef" for c in normalized),
        f"{label} must be a full 40-character commit SHA",
    )
    return normalized


def ensure_output_directory_available(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")


def assert_unique_keys(keys: Iterable[Any], label: str) -> None:
    seen: set[Any] = set()
    for key in keys:
        if key in seen:
            raise ContractError(f"{label} collision: {key!r}")
        seen.add(key)


def requested_token_count(fraction: float, full_token_count: int) -> int:
    require(full_token_count >= 0, "full token count must be non-negative")
    require(0.0 <= fraction <= 1.0, "prefix fraction must be in [0, 1]")
    if fraction == 0.0 or full_token_count == 0:
        return 0
    return min(full_token_count, int(math.ceil(fraction * full_token_count)))


def build_prefix_token_schedule(
    full_token_count: int,
    fractions: Sequence[float] = PREFIX_SCHEDULE,
) -> list[dict[str, Any]]:
    """Map requested fractions to unique token counts in deterministic order."""

    require(tuple(fractions) == tuple(sorted(fractions)), "prefix fractions must be sorted")
    grouped: dict[int, list[float]] = {}
    for fraction in fractions:
        normalized = float(fraction)
        token_count = requested_token_count(normalized, full_token_count)
        grouped.setdefault(token_count, []).append(normalized)
    schedule: list[dict[str, Any]] = []
    for token_count, requested_fractions in grouped.items():
        schedule.append(
            {
                "requested_prefix_fraction": requested_fractions[0],
                "requested_prefix_fractions": requested_fractions,
                "actual_evidence_prefix_token_count": token_count,
                "full_evidence_token_count": full_token_count,
                "actual_fraction": (token_count / full_token_count) if full_token_count else 0.0,
            }
        )
    return schedule


def _tokenize_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    require(isinstance(token_ids, (list, tuple)), "tokenizer.encode must return a token sequence")
    result = [int(token_id) for token_id in token_ids]
    require(all(token_id >= 0 for token_id in result), "token ids must be non-negative")
    return result


def construct_token_prefixes(tokenizer: Any, claim: str, evidence: str) -> dict[str, Any]:
    """Construct exact serialization prefixes by slicing the evidence token suffix."""

    require(isinstance(claim, str) and isinstance(evidence, str), "claim and evidence must be strings")
    marker_text = f"Claim: {claim}\nEvidence:"
    full_text = f"Claim: {claim}\nEvidence: {evidence}"
    marker_ids = _tokenize_without_special_tokens(tokenizer, marker_text)
    full_ids = _tokenize_without_special_tokens(tokenizer, full_text)
    require(marker_ids, "fixed claim/evidence marker tokenization is empty")
    require(
        len(full_ids) >= len(marker_ids) and full_ids[: len(marker_ids)] == marker_ids,
        "serialization/token-prefix invariant violation: fixed marker tokens are not a prefix of full serialization",
    )
    evidence_suffix_ids = full_ids[len(marker_ids) :]
    if evidence:
        require(evidence_suffix_ids, "non-empty evidence produced an empty evidence-token suffix")

    prefixes: list[dict[str, Any]] = []
    for schedule_entry in build_prefix_token_schedule(len(evidence_suffix_ids)):
        token_count = schedule_entry["actual_evidence_prefix_token_count"]
        input_ids = marker_ids + evidence_suffix_ids[:token_count]
        if token_count == 0:
            serialized_input_text = marker_text
        else:
            serialized_input_text = tokenizer.decode(
                input_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        prefixes.append(
            {
                **schedule_entry,
                "input_ids": input_ids,
                "serialized_input_text": serialized_input_text,
            }
        )

    require(prefixes[0]["input_ids"] == marker_ids, "0% prefix must contain marker tokens only")
    require(prefixes[-1]["input_ids"] == full_ids, "100% prefix must equal full serialized tokens")
    assert_unique_keys(
        (entry["actual_evidence_prefix_token_count"] for entry in prefixes),
        "unique evidence-prefix token count",
    )
    return {
        "marker_text": marker_text,
        "full_text": full_text,
        "marker_token_ids": marker_ids,
        "full_token_ids": full_ids,
        "evidence_suffix_token_ids": evidence_suffix_ids,
        "prefixes": prefixes,
    }


def load_and_validate_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(f"invalid JSON at dataset line {line_number}: {error}") from error
            require(isinstance(row, dict), f"dataset line {line_number} is not an object")
            missing = REQUIRED_DATASET_FIELDS - row.keys()
            require(not missing, f"dataset line {line_number} missing fields: {sorted(missing)}")
            for field in REQUIRED_DATASET_FIELDS:
                require(isinstance(row[field], str), f"dataset line {line_number} field {field} must be a string")
                require(bool(row[field]), f"dataset line {line_number} field {field} must be non-empty")
            rows.append(row)
    require(rows, "dataset contains no rows")
    assert_unique_keys((row["id"] for row in rows), "dataset id")

    present_interventions = {row["intervention_type"] for row in rows}
    missing_interventions = REQUIRED_INTERVENTIONS - present_interventions
    require(not missing_interventions, f"missing required intervention families: {sorted(missing_interventions)}")

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    for pair_id, pair_rows in by_pair.items():
        references = [row for row in pair_rows if row["intervention_type"] == "none"]
        require(len(references) == 1, f"pair_id {pair_id!r} must have exactly one none reference")
        assert_unique_keys(
            (row["intervention_type"] for row in pair_rows),
            f"pair/intervention for {pair_id}",
        )
    return rows


def select_none_references(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    references: dict[str, Mapping[str, Any]] = {}
    for pair_id, pair_rows in by_pair.items():
        none_rows = [row for row in pair_rows if row["intervention_type"] == "none"]
        require(len(none_rows) == 1, f"pair_id {pair_id!r} must have exactly one none reference")
        references[pair_id] = none_rows[0]
    return references


def safe_cosine_similarity(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-12) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    require(left64.shape == right64.shape, "cosine vectors must have identical shapes")
    require(np.isfinite(left64).all() and np.isfinite(right64).all(), "cosine vectors must be finite")
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    if left_norm <= epsilon and right_norm <= epsilon:
        return 1.0
    if left_norm <= epsilon or right_norm <= epsilon:
        return 0.0
    cosine = float(np.dot(left64.ravel(), right64.ravel()) / (left_norm * right_norm))
    return float(np.clip(cosine, -1.0, 1.0))


def normalized_l2_distance(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-12) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    require(left64.shape == right64.shape, "distance vectors must have identical shapes")
    require(np.isfinite(left64).all() and np.isfinite(right64).all(), "distance vectors must be finite")
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    if left_norm <= epsilon and right_norm <= epsilon:
        return 0.0
    left_unit = left64 / left_norm if left_norm > epsilon else np.zeros_like(left64)
    right_unit = right64 / right_norm if right_norm > epsilon else np.zeros_like(right64)
    return float(np.linalg.norm(left_unit - right_unit))


def compute_layer_observation(hidden_states: np.ndarray, evidence_start_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    hidden = np.asarray(hidden_states, dtype=np.float32)
    require(hidden.ndim == 2, "layer hidden states must have shape [tokens, hidden_size]")
    require(hidden.shape[0] >= 1 and hidden.shape[1] >= 1, "layer hidden states must be non-empty")
    require(0 < evidence_start_index <= hidden.shape[0], "invalid evidence start index")
    require(np.isfinite(hidden).all(), "non-finite native hidden state")

    terminal = np.array(hidden[-1], dtype=np.float32, copy=True)
    last_step_delta: float | None = None
    terminal_cosine: float | None = None
    terminal_acceleration: float | None = None
    if hidden.shape[0] >= 2:
        last_step_delta = float(np.linalg.norm(hidden[-1] - hidden[-2]))
        terminal_cosine = safe_cosine_similarity(hidden[-1], hidden[-2])
    if hidden.shape[0] >= 3:
        latest_transition = hidden[-1] - hidden[-2]
        previous_transition = hidden[-2] - hidden[-3]
        terminal_acceleration = float(np.linalg.norm(latest_transition - previous_transition))

    evidence_count = hidden.shape[0] - evidence_start_index
    evidence_mean_delta: float | None = None
    evidence_max_delta: float | None = None
    if evidence_count > 0:
        evidence_transitions = hidden[evidence_start_index:] - hidden[evidence_start_index - 1 : -1]
        transition_norms = np.linalg.norm(evidence_transitions, axis=1)
        evidence_mean_delta = float(np.mean(transition_norms))
        evidence_max_delta = float(np.max(transition_norms))

    metrics = {
        "terminal_hidden_norm": float(np.linalg.norm(terminal)),
        "last_step_delta": last_step_delta,
        "terminal_consecutive_state_cosine": terminal_cosine,
        "terminal_acceleration": terminal_acceleration,
        "evidence_region_mean_consecutive_state_delta": evidence_mean_delta,
        "evidence_region_max_consecutive_state_delta": evidence_max_delta,
    }
    assert_finite_metrics(metrics, "layer observation")
    return terminal, metrics


def assert_finite_metrics(metrics: Mapping[str, Any], label: str) -> None:
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, (float, int, np.floating, np.integer)):
            require(math.isfinite(float(value)), f"non-finite metric {label}.{key}")


def trajectory_summary_distance(current: Mapping[str, Any], reference: Mapping[str, Any]) -> float:
    differences: list[float] = []
    for metric in TRAJECTORY_SUMMARY_METRICS:
        current_value = current.get(metric)
        reference_value = reference.get(metric)
        if current_value is None or reference_value is None:
            continue
        differences.append(float(current_value) - float(reference_value))
    require(differences, "trajectory summaries have no jointly available metrics")
    result = float(np.linalg.norm(np.asarray(differences, dtype=np.float64)))
    require(math.isfinite(result), "non-finite trajectory-summary difference")
    return result


def compute_paired_distances(
    current_vector: np.ndarray,
    reference_vector: np.ndarray,
    current_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
) -> dict[str, float]:
    current_transition = current_metrics.get("last_step_delta")
    reference_transition = reference_metrics.get("last_step_delta")
    require(current_transition is not None and reference_transition is not None, "last-step delta unavailable")
    result = {
        "terminal_normalized_l2_distance": normalized_l2_distance(current_vector, reference_vector),
        "terminal_cosine_distance": 1.0 - safe_cosine_similarity(current_vector, reference_vector),
        "transition_magnitude_difference": abs(float(current_transition) - float(reference_transition)),
        "trajectory_summary_difference": trajectory_summary_distance(current_metrics, reference_metrics),
    }
    assert_finite_metrics(result, "paired distance")
    return result


def assert_zero_prefix_identical(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    claims_identical: bool,
    intervention_type: str,
    tolerance: float,
) -> None:
    if not claims_identical or intervention_type not in ZERO_PREFIX_EQUALITY_INTERVENTIONS:
        return
    require(reference["actual_evidence_prefix_token_count"] == 0, "reference is not a 0% prefix")
    require(candidate["actual_evidence_prefix_token_count"] == 0, "candidate is not a 0% prefix")
    require(
        reference["serialized_input_text"].encode("utf-8")
        == candidate["serialized_input_text"].encode("utf-8"),
        f"0% serialized bytes differ for intervention {intervention_type}",
    )
    require(
        tuple(reference["input_ids"]) == tuple(candidate["input_ids"]),
        f"0% input tokens differ for intervention {intervention_type}",
    )
    reference_vectors = np.asarray(reference["terminal_vectors"], dtype=np.float32)
    candidate_vectors = np.asarray(candidate["terminal_vectors"], dtype=np.float32)
    require(reference_vectors.shape == candidate_vectors.shape, "0% terminal-state shape mismatch")
    require(
        np.allclose(reference_vectors, candidate_vectors, rtol=0.0, atol=tolerance),
        f"0% native hidden states differ for intervention {intervention_type}",
    )


def assert_observer_source_safety(script_path: Path) -> None:
    """Fail closed if a future edit adds prohibited execution calls."""

    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    forbidden_call_attributes = {"backward", "generate", "train", "step", "save_pretrained"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            require(
                not any(name == "src.contramamba" or name.startswith("contramamba") for name in names),
                "ContraMamba/downstream imports are prohibited",
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            require(
                node.func.attr not in forbidden_call_attributes,
                f"prohibited observer call present: {node.func.attr}",
            )
        if isinstance(node, ast.Attribute):
            require(node.attr not in {"compile", "optim", "optimizer"}, "optimizer/compile path is prohibited")


def _resolve_runtime(device_name: str, dtype_name: str) -> tuple[Any, Any, str]:
    import torch

    require(device_name == AUTHORIZED_DEVICE, f"O0a authority binds device={AUTHORIZED_DEVICE}; observed {device_name!r}")
    require(
        dtype_name == AUTHORIZED_DTYPE_NAME,
        f"O0a authority binds dtype={AUTHORIZED_DTYPE_NAME}; observed {dtype_name!r}",
    )
    dtype_by_name = {
        "float32": torch.float32,
    }
    dtype = dtype_by_name[dtype_name]
    return torch, dtype, AUTHORIZED_DEVICE


def load_native_model_and_tokenizer(device_name: str, dtype_name: str) -> tuple[Any, Any, Any, str]:
    """The sole native model/tokenizer loading path used by an authorized run."""

    torch, dtype, resolved_device = _resolve_runtime(device_name, dtype_name)
    from transformers import AutoTokenizer, MambaModel

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    model = MambaModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=dtype)
    model.to(resolved_device)
    model.eval()
    model.requires_grad_(False)
    require(not model.training, "native Mamba must be in eval mode")
    require(not any(parameter.requires_grad for parameter in model.parameters()), "native Mamba parameters must be frozen")
    return torch, tokenizer, model, resolved_device


def configure_determinism(torch: Any) -> dict[str, Any]:
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "model_eval": True,
        "torch_inference_mode": True,
        "parameters_require_grad": False,
        "optimizer": False,
        "backward": False,
        "generation": False,
    }


def extract_mamba_config_flags(config: Any) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for name in MAMBA_CONFIG_FLAG_NAMES:
        if hasattr(config, name):
            value = getattr(config, name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                flags[name] = value
    return flags


def optional_resolved_hf_commit_hash(*objects: Any) -> str | None:
    for obj in objects:
        config = getattr(obj, "config", obj)
        value = getattr(config, "_commit_hash", None)
        if not isinstance(value, str):
            continue
        candidate = value.strip().lower()
        if len(candidate) == 40 and all(char in "0123456789abcdef" for char in candidate):
            return candidate
    return None


def _native_hidden_layers(outputs: Any, torch: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    hidden_states = getattr(outputs, "hidden_states", None)
    require(hidden_states is not None and len(hidden_states) > 0, "native model did not expose hidden_states")
    layers = list(hidden_states)
    sources = [f"hidden_states[{index}]" for index in range(len(layers))]
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    require(last_hidden_state is not None, "native model did not expose last_hidden_state")
    if layers[-1].shape != last_hidden_state.shape or not torch.equal(layers[-1], last_hidden_state):
        layers.append(last_hidden_state)
        sources.append("last_hidden_state")

    descriptors: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        if index == 0:
            role = "embedding_or_initial_hidden_state"
        elif index == len(sources) - 1:
            role = "output_hidden_state"
        else:
            role = "intermediate_hidden_state"
        descriptors.append({"layer_index": index, "layer_role": role, "state_source": source})
    return layers, descriptors


def run_native_forward(torch: Any, model: Any, input_ids: Sequence[int], device: str) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    token_tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        require(not torch.is_grad_enabled(), "gradient mode unexpectedly enabled inside inference_mode")
        outputs = model(
            input_ids=token_tensor,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    native_layers, descriptors = _native_hidden_layers(outputs, torch)
    arrays: list[np.ndarray] = []
    for tensor in native_layers:
        require(tensor.ndim == 3 and tensor.shape[0] == 1, "native hidden state must have shape [1, tokens, hidden]")
        array = tensor[0].detach().to(device="cpu", dtype=torch.float32).numpy()
        require(np.isfinite(array).all(), "non-finite native hidden-state tensor")
        arrays.append(array)
    return arrays, descriptors


def _fraction_to_count(tokenization: Mapping[str, Any]) -> dict[float, int]:
    result: dict[float, int] = {}
    for prefix in tokenization["prefixes"]:
        for fraction in prefix["requested_prefix_fractions"]:
            require(float(fraction) not in result, "requested prefix fraction collision")
            result[float(fraction)] = int(prefix["actual_evidence_prefix_token_count"])
    require(tuple(result) == PREFIX_SCHEDULE, "prefix schedule metadata mismatch")
    return result


def collect_observations(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    torch: Any,
    model: Any,
    device: str,
    deterministic_tolerance: float,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    terminal_vector_rows: list[np.ndarray] = []
    state_index: dict[tuple[str, int], dict[str, Any]] = {}
    tokenizations: dict[str, dict[str, Any]] = {}
    layer_descriptors: list[dict[str, Any]] | None = None

    for row in rows:
        row_id = str(row["id"])
        tokenization = construct_token_prefixes(tokenizer, str(row["claim"]), str(row["evidence"]))
        tokenizations[row_id] = tokenization
        evidence_start_index = len(tokenization["marker_token_ids"])
        for prefix in tokenization["prefixes"]:
            token_count = int(prefix["actual_evidence_prefix_token_count"])
            hidden_layers, current_descriptors = run_native_forward(torch, model, prefix["input_ids"], device)
            if layer_descriptors is None:
                layer_descriptors = current_descriptors
            require(current_descriptors == layer_descriptors, "native hidden-state layer layout changed across forwards")

            terminal_vectors: list[np.ndarray] = []
            layer_records: list[dict[str, Any]] = []
            vector_index = len(terminal_vector_rows)
            for descriptor, hidden_layer in zip(layer_descriptors, hidden_layers, strict=True):
                terminal, metrics = compute_layer_observation(hidden_layer, evidence_start_index)
                terminal_vectors.append(terminal)
                layer_index = int(descriptor["layer_index"])
                layer_record = {
                    **descriptor,
                    "terminal_hidden_state_npz_index": [vector_index, layer_index],
                    **metrics,
                }
                require(tuple(layer_record) == LAYER_OBSERVATION_KEYS, "layer observation schema drift")
                layer_records.append(layer_record)

            stacked_vectors = np.stack(terminal_vectors).astype(np.float32, copy=False)
            require(np.isfinite(stacked_vectors).all(), "non-finite terminal vector")
            terminal_vector_rows.append(stacked_vectors)
            serialized_text = str(prefix["serialized_input_text"])
            observation = {
                "schema_version": SCHEMA_VERSION,
                "row_id": row_id,
                "pair_id": str(row["pair_id"]),
                "intervention_type": str(row["intervention_type"]),
                "primary_failure_type": str(row["primary_failure_type"]),
                "final_label": str(row["final_label"]),
                "requested_prefix_fraction": float(prefix["requested_prefix_fraction"]),
                "requested_prefix_fractions": [float(value) for value in prefix["requested_prefix_fractions"]],
                "actual_evidence_prefix_token_count": token_count,
                "full_evidence_token_count": int(prefix["full_evidence_token_count"]),
                "actual_fraction": float(prefix["actual_fraction"]),
                "serialization_template": SERIALIZATION_TEMPLATE,
                "serialized_input_text": serialized_text,
                "serialized_input_utf8_sha256": sha256_bytes(serialized_text.encode("utf-8")),
                "input_token_ids": [int(value) for value in prefix["input_ids"]],
                "vector_index": vector_index,
                "layers": layer_records,
            }
            require(tuple(observation) == OBSERVATION_KEYS, "observation schema drift")
            observations.append(observation)
            state_key = (row_id, token_count)
            require(state_key not in state_index, f"row/prefix state collision: {state_key!r}")
            state_index[state_key] = {
                "actual_evidence_prefix_token_count": token_count,
                "serialized_input_text": serialized_text,
                "input_ids": list(prefix["input_ids"]),
                "terminal_vectors": stacked_vectors,
                "layer_records": layer_records,
            }

    require(layer_descriptors is not None, "no native hidden-state layers collected")
    assert_unique_keys(state_index.keys(), "row/prefix state")
    vectors = np.stack(terminal_vector_rows).astype(np.float32, copy=False)

    references = select_none_references(rows)
    row_by_id = {str(row["id"]): row for row in rows}
    for row in rows:
        intervention_type = str(row["intervention_type"])
        if intervention_type not in ZERO_PREFIX_EQUALITY_INTERVENTIONS:
            continue
        reference_row = references[str(row["pair_id"])]
        if str(row["claim"]) != str(reference_row["claim"]):
            continue
        reference_state = state_index[(str(reference_row["id"]), 0)]
        candidate_state = state_index[(str(row["id"]), 0)]
        assert_zero_prefix_identical(
            reference_state,
            candidate_state,
            claims_identical=True,
            intervention_type=intervention_type,
            tolerance=deterministic_tolerance,
        )

    context = {
        "state_index": state_index,
        "tokenizations": tokenizations,
        "row_by_id": row_by_id,
        "references": references,
    }
    return observations, vectors, context, layer_descriptors


def build_paired_distances(
    rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    layer_descriptors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    state_index = context["state_index"]
    tokenizations = context["tokenizations"]
    references = context["references"]
    records: list[dict[str, Any]] = []

    for row in rows:
        if row["intervention_type"] == "none":
            continue
        row_id = str(row["id"])
        pair_id = str(row["pair_id"])
        reference_row = references[pair_id]
        reference_id = str(reference_row["id"])
        row_counts = _fraction_to_count(tokenizations[row_id])
        reference_counts = _fraction_to_count(tokenizations[reference_id])
        for fraction in PREFIX_SCHEDULE:
            token_count = row_counts[fraction]
            reference_token_count = reference_counts[fraction]
            current_state = state_index[(row_id, token_count)]
            reference_state = state_index[(reference_id, reference_token_count)]
            for descriptor in layer_descriptors:
                layer_index = int(descriptor["layer_index"])
                distances = compute_paired_distances(
                    current_state["terminal_vectors"][layer_index],
                    reference_state["terminal_vectors"][layer_index],
                    current_state["layer_records"][layer_index],
                    reference_state["layer_records"][layer_index],
                )
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "pair_id": pair_id,
                    "row_id": row_id,
                    "reference_row_id": reference_id,
                    "intervention_type": str(row["intervention_type"]),
                    "requested_prefix_fraction": float(fraction),
                    "actual_evidence_prefix_token_count": token_count,
                    "reference_actual_evidence_prefix_token_count": reference_token_count,
                    "layer_index": layer_index,
                    "layer_role": str(descriptor["layer_role"]),
                    **distances,
                }
                require(tuple(record) == PAIRED_DISTANCE_KEYS, "paired-distance schema drift")
                records.append(record)
    assert_unique_keys(
        ((
            record["pair_id"],
            record["intervention_type"],
            record["requested_prefix_fraction"],
            record["layer_index"],
        ) for record in records),
        "paired-distance key",
    )
    return records


def descriptive_stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0 and np.isfinite(array).all(), "descriptive values must be non-empty and finite")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def build_summary(
    observations: Sequence[Mapping[str, Any]],
    paired_distances: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observation_groups: dict[tuple[str, float, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for observation in observations:
        for fraction in observation["requested_prefix_fractions"]:
            for layer in observation["layers"]:
                key = (
                    str(observation["intervention_type"]),
                    float(fraction),
                    int(layer["layer_index"]),
                    str(layer["layer_role"]),
                )
                for metric in TRAJECTORY_SUMMARY_METRICS:
                    value = layer[metric]
                    if value is not None:
                        observation_groups[key][metric].append(float(value))

    observation_summaries: list[dict[str, Any]] = []
    for (intervention, fraction, layer_index, layer_role), metrics in sorted(observation_groups.items()):
        observation_summaries.append(
            {
                "intervention_type": intervention,
                "requested_prefix_fraction": fraction,
                "layer_index": layer_index,
                "layer_role": layer_role,
                "metrics": {name: descriptive_stats(values) for name, values in sorted(metrics.items())},
            }
        )

    paired_groups: dict[tuple[str, float, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    paired_lookup: dict[tuple[str, str, float, int], Mapping[str, Any]] = {}
    for record in paired_distances:
        group_key = (
            str(record["intervention_type"]),
            float(record["requested_prefix_fraction"]),
            int(record["layer_index"]),
            str(record["layer_role"]),
        )
        paired_groups[group_key].append(record)
        lookup_key = (
            str(record["pair_id"]),
            str(record["intervention_type"]),
            float(record["requested_prefix_fraction"]),
            int(record["layer_index"]),
        )
        require(lookup_key not in paired_lookup, f"paired ranking lookup collision: {lookup_key!r}")
        paired_lookup[lookup_key] = record

    distance_names = (
        "terminal_normalized_l2_distance",
        "terminal_cosine_distance",
        "transition_magnitude_difference",
        "trajectory_summary_difference",
    )
    paired_summaries: list[dict[str, Any]] = []
    for (intervention, fraction, layer_index, layer_role), group_records in sorted(paired_groups.items()):
        ranking_count = 0
        ranking_denominator = 0
        for record in group_records:
            paraphrase = paired_lookup.get((str(record["pair_id"]), "paraphrase", fraction, layer_index))
            if paraphrase is None:
                continue
            ranking_denominator += 1
            if float(record["terminal_normalized_l2_distance"]) > float(
                paraphrase["terminal_normalized_l2_distance"]
            ):
                ranking_count += 1
        paired_summaries.append(
            {
                "intervention_type": intervention,
                "requested_prefix_fraction": fraction,
                "layer_index": layer_index,
                "layer_role": layer_role,
                "pair_count": len(group_records),
                "metrics": {
                    name: descriptive_stats([float(record[name]) for record in group_records])
                    for name in distance_names
                },
                "paired_ranking_diagnostic": {
                    "distance": "terminal_normalized_l2_distance",
                    "comparison": "distance(intervention, none) > distance(paraphrase, none)",
                    "count": ranking_count,
                    "denominator": ranking_denominator,
                },
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "status": "DESCRIPTIVE_SCREENING_ONLY",
        "scientific_claim_boundary": (
            "Execution success is not scientific evidence of a hallucination precursor; O0a observes only "
            "Hugging Face MambaModel layer hidden-state proxy trajectories as a prerequisite."
        ),
        "hidden_state_proxy_boundary": HIDDEN_STATE_PROXY_BOUNDARY,
        "normalized_l2_cosine_redundancy": NORMALIZED_L2_COSINE_REDUNDANCY,
        "paraphrase_control_boundary": PARAPHRASE_CONTROL_BOUNDARY,
        "trajectory_summary_boundary": TRAJECTORY_SUMMARY_BOUNDARY,
        "hard_pass_threshold": None,
        "observation_groups": observation_summaries,
        "paired_distance_groups": paired_summaries,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")))
            handle.write("\n")


def _write_report(path: Path, summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    lines = [
        f"# {EXPERIMENT_NAME}",
        "",
        "Status: DESCRIPTIVE SCREENING OUTPUT / NOT EXECUTION AUTHORITY",
        "",
        "O0a does not observe an unauthorized generated commitment and is not a hallucination-detector experiment. ",
        "Execution success is not scientific evidence of a hallucination precursor. These artifacts describe only ",
        "Hugging Face MambaModel layer hidden-state proxy distances for controlled token prefixes.",
        "",
        HIDDEN_STATE_PROXY_BOUNDARY,
        "",
        NORMALIZED_L2_COSINE_REDUNDANCY,
        "",
        PARAPHRASE_CONTROL_BOUNDARY,
        "",
        TRAJECTORY_SUMMARY_BOUNDARY,
        "",
        f"Repository HEAD: `{manifest['repository_head']}`  ",
        f"Dataset SHA256: `{manifest['dataset_sha256']}`  ",
        f"Model/tokenizer: `{manifest['model_id']}` @ `{manifest['model_revision']}`",
        f"Runtime: `{manifest['device']}` / `{manifest['dtype']}`",
        "",
        "No learned classifier, threshold tuning, best-layer selection, training, or generation was performed.",
        "`polarity_flip` is an authorized REFUTE semantic-sensitivity control; it is not a non-entitlement case.",
        "",
        "## Descriptive paired distances",
        "",
        "The table reports mean terminal normalized-L2 distance and the number of pair IDs exceeding the ",
        "whole-pair paraphrase-control distance. It is a screening diagnostic without a hard PASS threshold.",
        "",
        "| Intervention | Prefix | Layer | Mean D_l2 | Exceeds paraphrase |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in summary["paired_distance_groups"]:
        diagnostic = group["paired_ranking_diagnostic"]
        lines.append(
            f"| {group['intervention_type']} | {group['requested_prefix_fraction']:.2f} | "
            f"{group['layer_index']} | "
            f"{group['metrics']['terminal_normalized_l2_distance']['mean']:.8g} | "
            f"{diagnostic['count']}/{diagnostic['denominator']} |"
        )
    lines.extend(
        [
            "",
            "A negative or surface-form-dominated result would provide little support for a useful early native ",
            "precursor in this formulation, but would not falsify the overall ContraMamba program.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _relative_dataset_path(dataset_path: Path, repository_root: Path) -> str:
    try:
        return dataset_path.relative_to(repository_root).as_posix()
    except ValueError:
        return str(dataset_path)


def write_artifacts(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    vectors: np.ndarray,
    paired_distances: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "manifest.json", manifest)
    _write_jsonl(output_dir / "observations.jsonl", observations)
    np.savez_compressed(
        output_dir / "terminal_hidden_states.npz",
        schema_version=np.asarray(SCHEMA_VERSION),
        terminal_hidden_states=np.asarray(vectors, dtype=np.float32),
        row_ids=np.asarray([record["row_id"] for record in observations]),
        pair_ids=np.asarray([record["pair_id"] for record in observations]),
        intervention_types=np.asarray([record["intervention_type"] for record in observations]),
        actual_evidence_prefix_token_counts=np.asarray(
            [record["actual_evidence_prefix_token_count"] for record in observations], dtype=np.int64
        ),
        requested_prefix_fractions_json=np.asarray(
            [json.dumps(record["requested_prefix_fractions"], separators=(",", ":")) for record in observations]
        ),
    )
    _write_jsonl(output_dir / "paired_distances.jsonl", paired_distances)
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary, manifest)

    checksum_paths = sorted(path for path in output_dir.iterdir() if path.name != "SHA256SUMS.txt")
    expected_without_checksum = set(REQUIRED_ARTIFACTS) - {"SHA256SUMS.txt"}
    require({path.name for path in checksum_paths} == expected_without_checksum, "required artifact set mismatch")
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in checksum_paths]
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="ascii", newline="\n")
    require({path.name for path in output_dir.iterdir()} == set(REQUIRED_ARTIFACTS), "final artifact set mismatch")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--dataset", type=Path, default=Path("data/toy_interventions_v5.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authority-repository-head", required=True)
    parser.add_argument("--authority-dataset-sha256", required=True)
    parser.add_argument("--device", default=AUTHORIZED_DEVICE, help="O0a authority-bound value: cpu")
    parser.add_argument("--dtype", default=AUTHORIZED_DTYPE_NAME, help="O0a authority-bound value: float32")
    parser.add_argument("--deterministic-tolerance", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[1]
    dataset_path = args.dataset if args.dataset.is_absolute() else repository_root / args.dataset
    dataset_path = dataset_path.resolve()
    output_dir = args.output_dir.resolve()

    assert_observer_source_safety(script_path)
    require(args.deterministic_tolerance >= 0.0, "deterministic tolerance must be non-negative")
    ensure_output_directory_available(output_dir)
    expected_head = validate_commit_sha(args.authority_repository_head, "authority repository HEAD")
    actual_head = repository_head(repository_root)
    require(actual_head == expected_head, f"repository HEAD mismatch: expected {expected_head}, observed {actual_head}")

    require(dataset_path.is_file(), f"dataset does not exist: {dataset_path}")
    expected_dataset_sha256 = validate_sha256(args.authority_dataset_sha256, "authority dataset SHA256")
    actual_dataset_sha256 = sha256_file(dataset_path)
    require(
        actual_dataset_sha256 == expected_dataset_sha256,
        f"dataset SHA256 mismatch: expected {expected_dataset_sha256}, observed {actual_dataset_sha256}",
    )
    rows = load_and_validate_dataset(dataset_path)

    torch, tokenizer, model, resolved_device = load_native_model_and_tokenizer(args.device, args.dtype)
    deterministic_settings = configure_determinism(torch)
    observations, vectors, context, layer_descriptors = collect_observations(
        rows,
        tokenizer,
        torch,
        model,
        resolved_device,
        args.deterministic_tolerance,
    )
    paired_distances = build_paired_distances(rows, context, layer_descriptors)
    summary = build_summary(observations, paired_distances)

    import transformers

    parameter = next(model.parameters())
    require(str(parameter.device) == AUTHORIZED_DEVICE, f"O0a execution device drifted to {parameter.device}")
    require(parameter.dtype == torch.float32, f"O0a execution dtype drifted to {parameter.dtype}")
    model_config = getattr(model, "config", None)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "status": "DESCRIPTIVE_SCREENING_ONLY",
        "repository_head": actual_head,
        "authority_repository_head": expected_head,
        "script_path": script_path.relative_to(repository_root).as_posix(),
        "script_sha256": sha256_file(script_path),
        "dataset_path": _relative_dataset_path(dataset_path, repository_root),
        "dataset_sha256": actual_dataset_sha256,
        "authority_dataset_sha256": expected_dataset_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "model_name": MODEL_NAME,
        "tokenizer_name": TOKENIZER_NAME,
        "resolved_huggingface_commit_hash": optional_resolved_hf_commit_hash(model, tokenizer),
        "resolved_huggingface_commit_hash_note": (
            "Recorded when exposed by the loaded object; exact O0a provenance is bound by explicit model/tokenizer "
            "ID and immutable revision, not by private/internal fields alone."
        ),
        "native_model_class": "transformers.MambaModel",
        "measurement_target": "Hugging Face MambaModel layer hidden states",
        "measurement_target_name": "native pretrained Mamba hidden-state proxies",
        "out_of_scope_state_instrumentation": (
            "Hugging Face cache_params and deeper selective-SSM recurrent-state instrumentation are out of scope for "
            "O0a and require separate O0b/O1 authority."
        ),
        "mamba_config_flags": extract_mamba_config_flags(model_config),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": AUTHORIZED_DEVICE,
        "dtype": AUTHORIZED_DTYPE_NAME,
        "torch_parameter_device": str(parameter.device),
        "torch_parameter_dtype": str(parameter.dtype),
        "prefix_schedule": list(PREFIX_SCHEDULE),
        "nonzero_prefix_token_rule": "ceil(requested_fraction * full_evidence_token_count)",
        "duplicate_prefix_token_count_rule": "one forward per unique token count; all requested fractions retained",
        "serialization_template": SERIALIZATION_TEMPLATE,
        "tokenization": {"add_special_tokens": False, "prefix_space_owned_by_evidence_suffix": True},
        "row_count": len(rows),
        "unique_pair_id_count": len({str(row["pair_id"]) for row in rows}),
        "observation_count": len(observations),
        "paired_distance_count": len(paired_distances),
        "native_hidden_state_layers": layer_descriptors,
        "deterministic_settings": deterministic_settings,
        "zero_prefix_hidden_state_tolerance": args.deterministic_tolerance,
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "scientific_claim_boundary": (
            "O0a is a mechanistic screening experiment over hidden-state proxy trajectories only. It does not support "
            "population estimates, significance claims, generalization claims, hallucination-prediction claims, or "
            "scientific hallucination-precursor evidence by execution success alone."
        ),
        "hidden_state_proxy_boundary": HIDDEN_STATE_PROXY_BOUNDARY,
        "normalized_l2_cosine_redundancy": NORMALIZED_L2_COSINE_REDUNDANCY,
        "paraphrase_control_boundary": PARAPHRASE_CONTROL_BOUNDARY,
        "trajectory_summary_boundary": TRAJECTORY_SUMMARY_BOUNDARY,
        "urp_relationship": "NONE; O0a is separate and consumes no URP authority, checkpoint, artifact, or conclusion.",
        "training": False,
        "generation": False,
        "kaggle_required": False,
    }
    ensure_output_directory_available(output_dir)
    write_artifacts(
        output_dir,
        manifest=manifest,
        observations=observations,
        vectors=vectors,
        paired_distances=paired_distances,
        summary=summary,
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
