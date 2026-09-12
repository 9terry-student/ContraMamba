from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter  # noqa: E402


R5_AUTHORITY_COMMIT = "6b0ab6e62fc670191f3921d86fae8daf27d43caf"
R5_CORRECTION_COMMIT = "8f335e88ea905b7e1acd7b164e01ec6750144bf4"
R4_RESULT_FREEZE_COMMIT = "ffa889d184ad4236689a690384d5268665f5bd87"
R3_IMPLEMENTATION_COMMIT = "d62a424c8d13e582ffde7e8d2f8b7e0f43b610b2"
HISTORICAL_SOURCE_COMMIT = "3e0e9a435068c552abf20f3a74e0c3eccca344a3"
STRUCTURAL_ARTIFACT_COMMIT = "5b79d8585b20cf6fa4cfe52bbdbdce52374653aa"
STATISTICAL_SPECIFICATION_COMMIT = "4dc5bacd10a254b5ecd339ac1fe78bad9def5c47"

R5_AUTHORITY_SHA256 = (
    "f865fa6307597b562ae18b812219bf8233c4116b53f60ee2ad057a15c217a2aa"
)
R5_CORRECTION_SHA256 = (
    "8f090ca16ae750eca207e85190addfd9453ad67bb9b4d6eacf730752c467173f"
)

MODEL_CONFIG_SHA256 = (
    "784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a"
)
MODEL_CONFIG_BYTES = 895

EXPECTED_PYTHON = "3.12.13"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_TRANSFORMERS = "5.0.0"
EXPECTED_CUDA = "12.8"
EXPECTED_GPU_NAME = "Tesla T4"

ENCODER_CACHE_BATCH_SIZE = 8
DOWNSTREAM_BATCH_SIZE = 720
ROWS_PER_EVALUATOR = 1800
ENCODER_CACHE_CHUNKS_PER_EVALUATOR = 225
DOWNSTREAM_BATCH_PARTITION = (720, 720, 360)
EXPECTED_SCIENTIFIC_MAMBA_CACHE_FORWARD_CALLS = 4050
EXPECTED_SCIENTIFIC_DOWNSTREAM_FORWARD_CALLS = 54
EXPECTED_SCIENTIFIC_ROWS = 32400

R2_ENCODED_COORDINATE_SHA256 = (
    "d3cf61e55bb04e6dbebe44be433597cce958cb15989412dcf6efc15a329c576a"
)

SCIENTIFIC_ROW_SCHEMA = "gen4_r5_tier2_scientific_evaluator_row_v1"
SYNTHETIC_SUMMARY_SCHEMA = "gen4_r5_synthetic_preflight_v1"
SCIENTIFIC_SUMMARY_SCHEMA = "gen4_r5_scientific_inference_summary_v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def require_head(expected_head: str) -> str:
    observed = git_head()
    if observed != expected_head:
        raise RuntimeError(
            f"HEAD mismatch: expected={expected_head} observed={observed}"
        )
    return observed


def fixed_slices(total: int, batch_size: int) -> tuple[tuple[int, int], ...]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return tuple(
        (start, min(start + batch_size, total))
        for start in range(0, total, batch_size)
    )


def encoder_cache_slices(total: int) -> tuple[tuple[int, int], ...]:
    return fixed_slices(total, ENCODER_CACHE_BATCH_SIZE)


def downstream_slices(total: int) -> tuple[tuple[int, int], ...]:
    return fixed_slices(total, DOWNSTREAM_BATCH_SIZE)


def expected_evaluators() -> tuple[tuple[int, str], ...]:
    return tuple(
        (seed, arm)
        for seed in adapter.SEEDS
        for arm in adapter.ARMS
    )


def expected_checkpoint_path(seed: int, arm: str) -> Path:
    adapter.expected_checkpoint_sha256(seed, arm)
    return (
        ROOT
        / "reports"
        / "reason_router_gen3_grouped_factorial_runs"
        / f"seed{seed}"
        / arm
        / "selected_checkpoint.pt"
    )


def canonical_model_snapshot_dir(
    home: str | Path | None = None,
) -> Path:
    return adapter.canonical_tokenizer_snapshot_dir(home)


def authenticate_model_config(snapshot_dir: str | Path) -> Path:
    snapshot_dir = Path(snapshot_dir)

    if snapshot_dir.name != adapter.TOKENIZER_REVISION_REFERENCE:
        raise ValueError("model snapshot revision directory mismatch")

    config_path = snapshot_dir / "config.json"

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    raw = config_path.read_bytes()

    if len(raw) != MODEL_CONFIG_BYTES:
        raise ValueError("model config byte-count mismatch")

    observed = sha256_bytes(raw)

    if observed != MODEL_CONFIG_SHA256:
        raise ValueError(
            "model config SHA256 mismatch: "
            f"expected={MODEL_CONFIG_SHA256} observed={observed}"
        )

    return config_path


def build_local_mamba_backbone(snapshot_dir: str | Path) -> torch.nn.Module:
    config_path = authenticate_model_config(snapshot_dir)

    from transformers import MambaConfig, MambaModel

    config_data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    config = MambaConfig.from_dict(config_data)
    config.use_mamba_kernels = True

    return MambaModel(config)


def runtime_identity() -> dict[str, Any]:
    import transformers

    mamba_ssm_spec = importlib.util.find_spec("mamba_ssm")

    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime_reported_by_pytorch": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": (
            int(torch.cuda.device_count())
            if torch.cuda.is_available()
            else 0
        ),
        "gpu0_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "mamba_ssm_version": None if mamba_ssm_spec is None else "PRESENT",
        "default_dtype": str(torch.get_default_dtype()),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
    }


def validate_runtime_identity(runtime: Mapping[str, Any]) -> None:
    expected = {
        "python_version": EXPECTED_PYTHON,
        "torch_version": EXPECTED_TORCH,
        "transformers_version": EXPECTED_TRANSFORMERS,
        "cuda_runtime_reported_by_pytorch": EXPECTED_CUDA,
        "cuda_available": True,
        "gpu0_name": EXPECTED_GPU_NAME,
        "mamba_ssm_version": None,
        "default_dtype": "torch.float32",
        "autocast_enabled": False,
    }

    for key, value in expected.items():
        if runtime.get(key) != value:
            raise RuntimeError(
                f"runtime identity mismatch for {key}: "
                f"expected={value!r} observed={runtime.get(key)!r}"
            )

    if int(runtime.get("cuda_device_count", 0)) < 1:
        raise RuntimeError("at least one CUDA device is required")


def tokenizer_identity() -> dict[str, Any]:
    return {
        "family": "A",
        "revision_reference": adapter.TOKENIZER_REVISION_REFERENCE,
        "file_sha256": dict(adapter.TOKENIZER_FILE_SHA256),
    }


def load_and_validate_canonical_inputs(
    tokenizer_snapshot: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = adapter.load_canonical_gen4_artifact()
    tokenizer = adapter.load_canonical_tokenizer(tokenizer_snapshot)
    encoded = adapter.encode_gen4_rows(rows, tokenizer)

    coordinate_sha = adapter.encoded_coordinate_sha256(encoded)

    if coordinate_sha != R2_ENCODED_COORDINATE_SHA256:
        raise RuntimeError(
            "R2 encoded-coordinate identity mismatch: "
            f"expected={R2_ENCODED_COORDINATE_SHA256} observed={coordinate_sha}"
        )

    return rows, encoded


def _feature_slice(
    encoded: Mapping[str, Any],
    start: int,
    end: int,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}

    for key in (
        "input_ids",
        "attention_mask",
        "claim_mask",
        "evidence_mask",
    ):
        value = encoded[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"encoded {key} must be tensor")
        result[key] = value[start:end].to(device)

    return result


def cache_encoder_hidden_states(
    model: torch.nn.Module,
    encoded: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    if any(parameter.requires_grad for parameter in model.mamba.parameters()):
        raise ValueError("encoder cache requires fully frozen Mamba parameters")

    input_ids = encoded["input_ids"]

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("encoded input_ids must be tensor")

    model.mamba.eval()
    chunks: list[torch.Tensor] = []
    call_count = 0

    with torch.inference_mode():
        for start, end in encoder_cache_slices(input_ids.shape[0]):
            ids = input_ids[start:end].to(device)
            output = model.mamba(input_ids=ids)
            hidden = output.last_hidden_state

            if not isinstance(hidden, torch.Tensor):
                raise TypeError("Mamba last_hidden_state must be tensor")

            if hidden.dtype != torch.float32:
                raise RuntimeError(
                    f"encoder hidden dtype drift: {hidden.dtype}"
                )

            if not torch.isfinite(hidden).all():
                raise RuntimeError("non-finite encoder hidden state")

            chunks.append(hidden)
            call_count += 1

    if not chunks:
        raise RuntimeError("encoder cache produced no chunks")

    combined = torch.cat(chunks, dim=0)

    if combined.shape[0] != input_ids.shape[0]:
        raise RuntimeError("encoder cache row-count mismatch")

    return combined, call_count


def cached_downstream_forward(
    model: torch.nn.Module,
    feature_batch: Mapping[str, torch.Tensor],
    encoder_hidden_states: torch.Tensor,
    *,
    arm: str,
) -> Mapping[str, Any]:
    return model(
        input_ids=feature_batch["input_ids"],
        attention_mask=feature_batch["attention_mask"],
        claim_mask=feature_batch["claim_mask"],
        evidence_mask=feature_batch["evidence_mask"],
        encoder_hidden_states=encoder_hidden_states,
        decision_mode=adapter.DECISION_MODE,
        gradient_ownership_mode=adapter.GRADIENT_OWNERSHIP_MODE,
        edge_gradient_lambdas=adapter.expected_edge_gradient_lambdas(arm),
        return_q_diagnostics=True,
    )


def _require_finite_output(output: Mapping[str, Any]) -> None:
    for key in ("q_authorized", "entitlement_prob", "logits"):
        value = output.get(key)

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model output {key} must be tensor")

        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite model output: {key}")


def _decorate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    identity = tokenizer_identity()
    decorated: list[dict[str, Any]] = []

    for raw in rows:
        row = dict(raw)
        seed = int(row["seed"])
        arm = str(row["arm"])

        row.update({
            "schema_version": SCIENTIFIC_ROW_SCHEMA,
            "structural_artifact_commit": STRUCTURAL_ARTIFACT_COMMIT,
            "statistical_specification_commit": STATISTICAL_SPECIFICATION_COMMIT,
            "recovery_implementation_commit": R3_IMPLEMENTATION_COMMIT,
            "r4_result_freeze_commit": R4_RESULT_FREEZE_COMMIT,
            "r5_execution_authority_commit": R5_AUTHORITY_COMMIT,
            "r5_batch_cache_correction_commit": R5_CORRECTION_COMMIT,
            "historical_evaluator_source_commit": HISTORICAL_SOURCE_COMMIT,
            "evaluator_seed": seed,
            "evaluator_arm": arm,
            "tokenizer_identity": identity,
        })

        for key in (
            "q_authorized",
            "entitlement_prob",
            "refute_logit",
            "ne_logit",
            "support_logit",
            "support_vs_best_nonsupport_logit_margin",
        ):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(
                    f"non-finite or non-numeric scientific output {key}"
                )

        decorated.append(row)

    return decorated


def run_cached_downstream_batches(
    model: torch.nn.Module,
    encoded: Mapping[str, Any],
    metadata_rows: Sequence[Mapping[str, Any]],
    encoder_hidden_states: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
    arm: str,
    checkpoint_sha256: str,
) -> tuple[list[dict[str, Any]], int]:
    if len(metadata_rows) != len(encoded["row_id"]):
        raise ValueError("metadata/encoded row-count mismatch")

    if encoder_hidden_states.shape[0] != len(metadata_rows):
        raise ValueError("encoder cache/metadata row-count mismatch")

    serialized: list[dict[str, Any]] = []
    call_count = 0

    with torch.inference_mode():
        for start, end in downstream_slices(len(metadata_rows)):
            feature_batch = _feature_slice(
                encoded,
                start,
                end,
                device=device,
            )

            hidden_batch = encoder_hidden_states[start:end]

            output = cached_downstream_forward(
                model,
                feature_batch,
                hidden_batch,
                arm=arm,
            )

            _require_finite_output(output)

            batch_rows = adapter.serialize_model_outputs(
                output,
                metadata_rows[start:end],
                seed=seed,
                arm=arm,
                checkpoint_sha256=checkpoint_sha256,
            )

            serialized.extend(_decorate_rows(batch_rows))
            call_count += 1

    return serialized, call_count


def _torch_checkpoint_loader(path: Path) -> Any:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


def load_evaluator_model(
    *,
    seed: int,
    arm: str,
    model_snapshot: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    checkpoint_path = expected_checkpoint_path(seed, arm)

    observed_sha, payload = adapter.authenticated_checkpoint_load(
        checkpoint_path,
        seed=seed,
        arm=arm,
        loader=_torch_checkpoint_loader,
    )

    backbone = build_local_mamba_backbone(model_snapshot)

    model = adapter.build_historical_model_from_backbone(
        backbone=backbone,
        arm=arm,
    )

    adapter.strict_load_state_dict(model, payload)

    model.to(device)
    model.eval()

    if any(parameter.requires_grad for parameter in model.mamba.parameters()):
        raise RuntimeError("Mamba encoder unexpectedly trainable")

    return model, observed_sha


def _synthetic_encoded(rows: int = 16) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if rows <= 0:
        raise ValueError("synthetic row count must be positive")

    input_ids = torch.zeros(
        (rows, adapter.MAX_LENGTH),
        dtype=torch.long,
    )
    attention = torch.zeros(
        (rows, adapter.MAX_LENGTH),
        dtype=torch.bool,
    )
    claim_mask = torch.zeros_like(attention)
    evidence_mask = torch.zeros_like(attention)

    for index in range(rows):
        claim_ids = torch.arange(1, 33, dtype=torch.long) + index
        evidence_ids = torch.arange(101, 133, dtype=torch.long) + index

        input_ids[index, :32] = claim_ids
        input_ids[index, 32] = adapter.EOS_TOKEN_ID
        input_ids[index, 33:65] = evidence_ids

        attention[index, :65] = True
        claim_mask[index, :32] = True
        evidence_mask[index, 33:65] = True

    metadata = [
        {
            "row_id": f"synthetic-{index:03d}",
            "source_pair_id": f"synthetic-pair-{index:03d}",
            "contrast_cell_id": "SYNTHETIC_NON_GEN4",
            "claim": f"synthetic claim {index}",
            "evidence": f"synthetic evidence {index}",
        }
        for index in range(rows)
    ]

    encoded = {
        "row_id": [row["row_id"] for row in metadata],
        "source_pair_id": [row["source_pair_id"] for row in metadata],
        "contrast_cell_id": [row["contrast_cell_id"] for row in metadata],
        "input_ids": input_ids,
        "attention_mask": attention,
        "claim_mask": claim_mask,
        "evidence_mask": evidence_mask,
    }

    return metadata, encoded


def _extract_preflight_tensors(
    output: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_finite_output(output)

    return (
        output["q_authorized"].detach().cpu(),
        output["entitlement_prob"].detach().cpu(),
        output["logits"].detach().cpu(),
    )


def synthetic_forward_preflight(
    *,
    expected_head: str,
    model_snapshot: str | Path,
) -> dict[str, Any]:
    head = require_head(expected_head)
    runtime = runtime_identity()
    validate_runtime_identity(runtime)

    device = torch.device("cuda:0")
    seed = 180
    arm = "G3-GROUP-D-HALF"

    model, checkpoint_sha = load_evaluator_model(
        seed=seed,
        arm=arm,
        model_snapshot=model_snapshot,
        device=device,
    )

    metadata, encoded = _synthetic_encoded()

    mamba_calls = 0
    downstream_calls = 0
    repeated_outputs = []

    def _mamba_hook(_module, _inputs, _output):
        nonlocal mamba_calls
        mamba_calls += 1

    handle = model.mamba.register_forward_hook(_mamba_hook)

    try:
        for _repeat in range(2):
            hidden, cache_calls = cache_encoder_hidden_states(
                model,
                encoded,
                device=device,
            )

            if cache_calls != 2:
                raise RuntimeError(
                    f"synthetic encoder cache call-count drift: {cache_calls}"
                )

            before = mamba_calls

            feature_batch = _feature_slice(
                encoded,
                0,
                len(metadata),
                device=device,
            )

            with torch.inference_mode():
                output = cached_downstream_forward(
                    model,
                    feature_batch,
                    hidden,
                    arm=arm,
                )

            downstream_calls += 1

            if mamba_calls != before:
                raise RuntimeError(
                    "cached downstream unexpectedly re-executed Mamba"
                )

            repeated_outputs.append(
                _extract_preflight_tensors(output)
            )
    finally:
        handle.remove()

    if mamba_calls != 4:
        raise RuntimeError(
            f"synthetic total Mamba call-count drift: {mamba_calls}"
        )

    if downstream_calls != 2:
        raise RuntimeError(
            f"synthetic downstream call-count drift: {downstream_calls}"
        )

    first = repeated_outputs[0]
    second = repeated_outputs[1]

    if any(
        left.shape != right.shape
        for left, right in zip(first, second)
    ):
        raise RuntimeError("synthetic repeated output shape mismatch")

    if any(
        not torch.equal(left, right)
        for left, right in zip(first, second)
    ):
        raise RuntimeError("synthetic repeated outputs are not deterministic")

    return {
        "schema_version": SYNTHETIC_SUMMARY_SCHEMA,
        "result": "PASS",
        "scientific_evidence": False,
        "head_commit": head,
        "r5_execution_authority_commit": R5_AUTHORITY_COMMIT,
        "r5_batch_cache_correction_commit": R5_CORRECTION_COMMIT,
        "runtime": runtime,
        "device": "cuda:0",
        "dtype": "float32",
        "autocast": False,
        "seed": seed,
        "arm": arm,
        "checkpoint_sha256": checkpoint_sha,
        "synthetic_row_count": len(metadata),
        "encoder_cache_batch_size": ENCODER_CACHE_BATCH_SIZE,
        "encoder_cache_forward_calls": mamba_calls,
        "downstream_forward_calls": downstream_calls,
        "cached_downstream_reexecuted_mamba": False,
        "deterministic_repeat": True,
        "finite_outputs": True,
        "training": False,
        "backward": False,
        "statistical_testing": False,
    }


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


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def scientific_jsonl_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    return b"".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def write_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, int]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw = scientific_jsonl_bytes(rows)
    path.write_bytes(raw)

    return sha256_bytes(raw), len(raw)


def validate_preflight_summary(
    path: str | Path,
    *,
    expected_head: str,
) -> dict[str, Any]:
    path = Path(path)
    summary = json.loads(path.read_text(encoding="utf-8"))

    required = {
        "schema_version": SYNTHETIC_SUMMARY_SCHEMA,
        "result": "PASS",
        "scientific_evidence": False,
        "head_commit": expected_head,
        "r5_execution_authority_commit": R5_AUTHORITY_COMMIT,
        "r5_batch_cache_correction_commit": R5_CORRECTION_COMMIT,
        "encoder_cache_batch_size": ENCODER_CACHE_BATCH_SIZE,
        "cached_downstream_reexecuted_mamba": False,
        "deterministic_repeat": True,
        "finite_outputs": True,
        "training": False,
        "backward": False,
        "statistical_testing": False,
    }

    for key, value in required.items():
        if summary.get(key) != value:
            raise RuntimeError(
                f"synthetic preflight summary mismatch for {key}"
            )

    validate_runtime_identity(summary["runtime"])

    return summary


def run_one_evaluator(
    *,
    seed: int,
    arm: str,
    rows: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
    model_snapshot: str | Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], str, int, int]:
    model, checkpoint_sha = load_evaluator_model(
        seed=seed,
        arm=arm,
        model_snapshot=model_snapshot,
        device=device,
    )

    hidden, cache_calls = cache_encoder_hidden_states(
        model,
        encoded,
        device=device,
    )

    output_rows, downstream_calls = run_cached_downstream_batches(
        model,
        encoded,
        rows,
        hidden,
        device=device,
        seed=seed,
        arm=arm,
        checkpoint_sha256=checkpoint_sha,
    )

    del hidden
    del model
    torch.cuda.empty_cache()

    return (
        output_rows,
        checkpoint_sha,
        cache_calls,
        downstream_calls,
    )


def run_scientific_matrix(
    *,
    expected_head: str,
    preflight_summary_path: str | Path,
    model_snapshot: str | Path,
    tokenizer_snapshot: str | Path | None,
    output_jsonl: str | Path,
    output_summary: str | Path,
) -> dict[str, Any]:
    head = require_head(expected_head)

    runtime = runtime_identity()
    validate_runtime_identity(runtime)

    preflight = validate_preflight_summary(
        preflight_summary_path,
        expected_head=head,
    )

    rows, encoded = load_and_validate_canonical_inputs(
        tokenizer_snapshot
    )

    if len(rows) != ROWS_PER_EVALUATOR:
        raise RuntimeError("canonical row-count drift")

    if len(encoder_cache_slices(len(rows))) != ENCODER_CACHE_CHUNKS_PER_EVALUATOR:
        raise RuntimeError("encoder cache partition drift")

    if tuple(
        end - start
        for start, end in downstream_slices(len(rows))
    ) != DOWNSTREAM_BATCH_PARTITION:
        raise RuntimeError("downstream partition drift")

    device = torch.device("cuda:0")
    all_rows: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []
    cache_forward_count = 0
    downstream_forward_count = 0

    for seed, arm in expected_evaluators():
        evaluator_rows, checkpoint_sha, cache_calls, downstream_calls = (
            run_one_evaluator(
                seed=seed,
                arm=arm,
                rows=rows,
                encoded=encoded,
                model_snapshot=model_snapshot,
                device=device,
            )
        )

        if cache_calls != ENCODER_CACHE_CHUNKS_PER_EVALUATOR:
            raise RuntimeError(
                f"encoder cache count drift: seed={seed} arm={arm}"
            )

        if downstream_calls != len(DOWNSTREAM_BATCH_PARTITION):
            raise RuntimeError(
                f"downstream count drift: seed={seed} arm={arm}"
            )

        all_rows.extend(evaluator_rows)

        checkpoint_manifest.append({
            "seed": seed,
            "arm": arm,
            "checkpoint_sha256": checkpoint_sha,
        })

        cache_forward_count += cache_calls
        downstream_forward_count += downstream_calls

    if cache_forward_count != EXPECTED_SCIENTIFIC_MAMBA_CACHE_FORWARD_CALLS:
        raise RuntimeError("scientific Mamba cache forward-count drift")

    if downstream_forward_count != EXPECTED_SCIENTIFIC_DOWNSTREAM_FORWARD_CALLS:
        raise RuntimeError("scientific downstream forward-count drift")

    adapter.validate_complete_matrix(
        all_rows,
        rows,
        expected_evaluators(),
    )

    if len(all_rows) != EXPECTED_SCIENTIFIC_ROWS:
        raise RuntimeError("scientific output row-count drift")

    output_sha, output_bytes = write_jsonl(
        output_jsonl,
        all_rows,
    )

    preflight_sha = sha256_file(preflight_summary_path)

    summary = {
        "schema_version": SCIENTIFIC_SUMMARY_SCHEMA,
        "result": "PASS_EXECUTION_MATRIX_PRODUCED",
        "head_commit": head,
        "r5_execution_authority_commit": R5_AUTHORITY_COMMIT,
        "r5_execution_authority_sha256": R5_AUTHORITY_SHA256,
        "r5_batch_cache_correction_commit": R5_CORRECTION_COMMIT,
        "r5_batch_cache_correction_sha256": R5_CORRECTION_SHA256,
        "r4_result_freeze_commit": R4_RESULT_FREEZE_COMMIT,
        "r3_implementation_commit": R3_IMPLEMENTATION_COMMIT,
        "historical_evaluator_source_commit": HISTORICAL_SOURCE_COMMIT,
        "runtime": runtime,
        "device": "cuda:0",
        "dtype": "float32",
        "autocast": False,
        "effective_mamba_backend": "transformers_5.0.0_no_mamba_ssm",
        "encoder_cache_batch_size": ENCODER_CACHE_BATCH_SIZE,
        "downstream_batch_size": DOWNSTREAM_BATCH_SIZE,
        "downstream_batch_partition": list(DOWNSTREAM_BATCH_PARTITION),
        "canonical_input_sha256": adapter.CANONICAL_GEN4_ARTIFACT_SHA256,
        "encoded_coordinate_sha256": R2_ENCODED_COORDINATE_SHA256,
        "tokenizer_identity": tokenizer_identity(),
        "checkpoint_manifest": checkpoint_manifest,
        "scientific_row_count": len(all_rows),
        "unique_key_count": len({
            adapter.output_key(row)
            for row in all_rows
        }),
        "complete_matrix_validation": "PASS",
        "output_jsonl_sha256": output_sha,
        "output_jsonl_bytes": output_bytes,
        "mamba_cache_forward_count": cache_forward_count,
        "downstream_model_forward_count": downstream_forward_count,
        "synthetic_preflight_summary_sha256": preflight_sha,
        "synthetic_preflight_result": preflight["result"],
        "training": False,
        "backward": False,
        "optimizer": False,
        "statistical_testing": False,
        "scientific_statistical_conclusion": "NOT_ESTABLISHED",
    }

    write_json(output_summary, summary)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gen4 R5 scientific-inference execution harness. "
            "Execution authority is commit- and runtime-bound."
        )
    )

    subparsers = parser.add_subparsers(
        dest="operation",
        required=True,
    )

    preflight = subparsers.add_parser(
        "synthetic-preflight",
    )
    preflight.add_argument(
        "--expected-head",
        required=True,
    )
    preflight.add_argument(
        "--model-snapshot",
        default=None,
    )
    preflight.add_argument(
        "--output-summary",
        required=True,
    )

    run = subparsers.add_parser(
        "run-scientific",
    )
    run.add_argument(
        "--expected-head",
        required=True,
    )
    run.add_argument(
        "--model-snapshot",
        default=None,
    )
    run.add_argument(
        "--tokenizer-snapshot",
        default=None,
    )
    run.add_argument(
        "--preflight-summary",
        required=True,
    )
    run.add_argument(
        "--output-jsonl",
        required=True,
    )
    run.add_argument(
        "--output-summary",
        required=True,
    )

    return parser


def _resolve_model_snapshot(value: str | None) -> Path:
    if value is None:
        return canonical_model_snapshot_dir()
    return Path(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.operation == "synthetic-preflight":
        summary = synthetic_forward_preflight(
            expected_head=args.expected_head,
            model_snapshot=_resolve_model_snapshot(
                args.model_snapshot
            ),
        )
        write_json(args.output_summary, summary)
        print(json.dumps(summary, sort_keys=True, indent=2))
        return 0

    if args.operation == "run-scientific":
        summary = run_scientific_matrix(
            expected_head=args.expected_head,
            preflight_summary_path=args.preflight_summary,
            model_snapshot=_resolve_model_snapshot(
                args.model_snapshot
            ),
            tokenizer_snapshot=args.tokenizer_snapshot,
            output_jsonl=args.output_jsonl,
            output_summary=args.output_summary,
        )
        print(json.dumps(summary, sort_keys=True, indent=2))
        return 0

    raise RuntimeError(
        f"unsupported operation: {args.operation}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
