#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import (
    reason_router_gen4_mamba370m_geometry_prepare_fast_cuda as geom,
)


DESIGN_FREEZE_COMMIT = "8879e80db019384eef37e63e1095e9f79366d60f"
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

HF_REPO = geom.HF_REPO
HF_REVISION = geom.HF_REVISION
EXPECTED_BACKBONE_CANONICAL_SHA256 = geom.MAMBA_STATE_CANONICAL_SHA256
EXPECTED_BACKBONE_STATE_KEY_COUNT = 482
EXPECTED_HIDDEN_SIZE = geom.HIDDEN_SIZE
EXPECTED_LAYER_COUNT = geom.LAYER_COUNT
EXPECTED_STATE_SIZE = geom.STATE_SIZE
EXPECTED_INTERMEDIATE_SIZE = geom.INTERMEDIATE_SIZE
EXPECTED_VOCAB_SIZE = 50280

SELECTED_PLANE = "P3"
CONTROL_PLANE = "P5"
EARLY_BLOCK = 35
LATE_BLOCK = 47

CLASS_ORDER = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
DECISIVE_CLASSES = ("REFUTE", "SUPPORT")

# These exact surfaces are frozen before any generation response is inspected.
# They deliberately share a long lexical prefix so t*-4..t*-1 can exist
# before the class identity becomes unique.
COMMITMENT_SURFACES: dict[str, str] = {
    "REFUTE": "Based on the evidence, the verdict is REFUTE.",
    "NOT_ENTITLED": "Based on the evidence, the verdict is NOT_ENTITLED.",
    "SUPPORT": "Based on the evidence, the verdict is SUPPORT.",
}

MIN_PRECOMMIT_GENERATED_TOKENS = 4
RESULT_PASS = "PASS_PRE_EMISSION_CAUSAL_LM_IDENTITY_AND_GRAMMAR_GATE"
SCHEMA_VERSION = "gen4-pre-emission-causal-lm-identity-grammar-gate-v1"


class PrecursorGateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PrecursorGateError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def _encoding_ids(value: Any) -> list[int]:
    if hasattr(value, "ids"):
        raw = value.ids
    elif isinstance(value, Mapping) and "input_ids" in value:
        raw = value["input_ids"]
    else:
        raw = value

    require(
        isinstance(raw, Sequence)
        and not isinstance(raw, (str, bytes, bytearray)),
        "TOKENIZER_ENCODING_TYPE",
    )
    ids = [int(x) for x in raw]
    require(ids, "EMPTY_COMMITMENT_ENCODING")
    require(all(x >= 0 for x in ids), "NEGATIVE_TOKEN_ID")
    return ids


def tokenize_surface(tokenizer: Any, surface: str) -> list[int]:
    require(
        isinstance(surface, str) and surface and surface == surface.strip(),
        "COMMITMENT_SURFACE",
    )
    encoded = tokenizer.encode(surface, add_special_tokens=False)
    return _encoding_ids(encoded)


def longest_common_prefix_length(
    sequences: Sequence[Sequence[int]],
) -> int:
    require(len(sequences) >= 2, "COMMON_PREFIX_SEQUENCE_COUNT")
    width = min(len(x) for x in sequences)
    count = 0
    for index in range(width):
        token = int(sequences[0][index])
        if all(int(seq[index]) == token for seq in sequences[1:]):
            count += 1
        else:
            break
    return count


def is_complete_prefix(
    left: Sequence[int],
    right: Sequence[int],
) -> bool:
    return (
        len(left) <= len(right)
        and list(left) == list(right[: len(left)])
    )


def unique_commitment_index(
    target: Sequence[int],
    all_sequences: Mapping[str, Sequence[int]],
) -> int:
    for index in range(len(target)):
        prefix = list(target[: index + 1])
        compatible = [
            label
            for label, sequence in all_sequences.items()
            if len(sequence) >= len(prefix)
            and list(sequence[: len(prefix)]) == prefix
        ]
        if len(compatible) == 1:
            return index
    raise PrecursorGateError("COMMITMENT_NEVER_UNIQUE")


def allowed_next_tokens(
    prefix: Sequence[int],
    all_sequences: Mapping[str, Sequence[int]],
) -> tuple[int, ...]:
    prefix_list = [int(x) for x in prefix]
    out: set[int] = set()
    compatible = 0

    for sequence in all_sequences.values():
        seq = [int(x) for x in sequence]
        if len(prefix_list) > len(seq):
            continue
        if seq[: len(prefix_list)] != prefix_list:
            continue
        compatible += 1
        if len(prefix_list) < len(seq):
            out.add(seq[len(prefix_list)])

    require(compatible > 0, "GRAMMAR_PREFIX_NOT_RECOGNIZED")
    return tuple(sorted(out))


def validate_token_sequences(
    token_ids: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    require(tuple(token_ids) == CLASS_ORDER, "CLASS_ORDER")
    require(
        all(token_ids[label] for label in CLASS_ORDER),
        "EMPTY_CLASS_ENCODING",
    )

    normalized = {
        label: [int(x) for x in token_ids[label]]
        for label in CLASS_ORDER
    }

    require(
        len({tuple(normalized[label]) for label in CLASS_ORDER})
        == len(CLASS_ORDER),
        "DUPLICATE_CLASS_ENCODING",
    )

    for left_label in CLASS_ORDER:
        for right_label in CLASS_ORDER:
            if left_label == right_label:
                continue
            require(
                not is_complete_prefix(
                    normalized[left_label],
                    normalized[right_label],
                ),
                f"COMPLETE_PREFIX:{left_label}:{right_label}",
            )

    common_prefix = longest_common_prefix_length(
        [normalized[label] for label in CLASS_ORDER]
    )
    require(
        common_prefix >= MIN_PRECOMMIT_GENERATED_TOKENS,
        f"COMMON_PREFIX_TOO_SHORT:{common_prefix}",
    )

    unique_indices = {
        label: unique_commitment_index(
            normalized[label],
            normalized,
        )
        for label in CLASS_ORDER
    }

    require(
        min(unique_indices.values())
        >= MIN_PRECOMMIT_GENERATED_TOKENS,
        f"UNIQUE_COMMITMENT_TOO_EARLY:{unique_indices}",
    )
    require(
        len(set(unique_indices.values())) == 1,
        f"NONDETERMINISTIC_COMMITMENT_POSITION:{unique_indices}",
    )

    t_star = {
        label: unique_indices[label]
        for label in DECISIVE_CLASSES
    }
    require(
        len(set(t_star.values())) == 1,
        f"DECISIVE_TSTAR_MISMATCH:{t_star}",
    )

    common_tokens = normalized[CLASS_ORDER[0]][:common_prefix]
    grammar_core = {
        "class_order": list(CLASS_ORDER),
        "token_ids": normalized,
        "longest_common_prefix_length": common_prefix,
        "longest_common_prefix_token_ids": common_tokens,
        "unique_commitment_token_index_zero_based": unique_indices,
        "decisive_t_star_zero_based": t_star,
        "minimum_precommit_generated_tokens": (
            MIN_PRECOMMIT_GENERATED_TOKENS
        ),
    }
    grammar_core["grammar_sha256"] = sha256_bytes(
        canonical_json_bytes(grammar_core)
    )
    return grammar_core


def build_grammar_manifest(
    tokenizer: Any,
    *,
    tokenizer_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    token_ids = {
        label: tokenize_surface(
            tokenizer,
            COMMITMENT_SURFACES[label],
        )
        for label in CLASS_ORDER
    }
    grammar = validate_token_sequences(token_ids)
    grammar["surfaces"] = {
        label: COMMITMENT_SURFACES[label]
        for label in CLASS_ORDER
    }
    grammar["tokenizer"] = dict(tokenizer_provenance)
    return grammar


def validate_config_dict(config: Mapping[str, Any]) -> dict[str, Any]:
    architectures = config.get("architectures")
    require(
        architectures == ["MambaForCausalLM"],
        f"CONFIG_ARCHITECTURES:{architectures}",
    )
    require(
        int(config.get("hidden_size", -1)) == EXPECTED_HIDDEN_SIZE,
        "CONFIG_HIDDEN_SIZE",
    )
    require(
        int(config.get("num_hidden_layers", -1))
        == EXPECTED_LAYER_COUNT,
        "CONFIG_LAYER_COUNT",
    )
    require(
        int(config.get("state_size", -1)) == EXPECTED_STATE_SIZE,
        "CONFIG_STATE_SIZE",
    )
    require(
        int(config.get("intermediate_size", -1))
        == EXPECTED_INTERMEDIATE_SIZE,
        "CONFIG_INTERMEDIATE_SIZE",
    )
    require(
        int(config.get("vocab_size", -1)) == EXPECTED_VOCAB_SIZE,
        "CONFIG_VOCAB_SIZE",
    )
    require(
        int(config.get("eos_token_id", -1)) == geom.EOS_TOKEN_ID,
        "CONFIG_EOS_TOKEN_ID",
    )
    require(
        int(config.get("pad_token_id", -1))
        == geom.EFFECTIVE_PAD_TOKEN_ID,
        "CONFIG_PAD_TOKEN_ID",
    )
    return {
        "architectures": list(architectures),
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "num_hidden_layers": EXPECTED_LAYER_COUNT,
        "state_size": EXPECTED_STATE_SIZE,
        "intermediate_size": EXPECTED_INTERMEDIATE_SIZE,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "eos_token_id": geom.EOS_TOKEN_ID,
        "pad_token_id": geom.EFFECTIVE_PAD_TOKEN_ID,
    }


def validate_snapshot_and_config(snapshot: Path) -> dict[str, Any]:
    snapshot_identity = geom.validate_snapshot(snapshot)
    config = json.loads(
        (snapshot / "config.json").read_text(encoding="utf-8-sig")
    )
    return {
        "snapshot_files": snapshot_identity,
        "config": validate_config_dict(config),
    }


def validate_loading_info(
    loading_info: Mapping[str, Any],
) -> dict[str, Any]:
    fields = (
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    )
    for field in fields:
        value = loading_info.get(field, [])
        require(not value, f"CAUSAL_LM_LOADING_INFO:{field}:{value}")
    return {field: [] for field in fields}


def tensor_sha256(tensor: Any) -> str:
    import torch

    require(torch.is_tensor(tensor), "TENSOR_SHA_TYPE")
    value = tensor.detach().cpu().contiguous()
    array = value.numpy()
    return sha256_bytes(array.tobytes(order="C"))


def validate_loaded_causal_lm(
    model: Any,
    *,
    loading_info: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    clean_loading = validate_loading_info(loading_info)

    backbone = getattr(model, "backbone", None)
    require(backbone is not None, "CAUSAL_LM_BACKBONE_MISSING")

    state = backbone.state_dict()
    require(
        len(state) == EXPECTED_BACKBONE_STATE_KEY_COUNT,
        f"BACKBONE_STATE_KEY_COUNT:{len(state)}",
    )
    canonical_state = {
        f"mamba.{key}": value.detach().cpu().clone()
        for key, value in state.items()
    }
    backbone_sha = geom.canonical_state_sha256(canonical_state)
    require(
        backbone_sha == EXPECTED_BACKBONE_CANONICAL_SHA256,
        f"BACKBONE_CANONICAL_SHA256:{backbone_sha}",
    )

    lm_head = getattr(model, "lm_head", None)
    require(lm_head is not None, "LM_HEAD_MISSING")
    weight = getattr(lm_head, "weight", None)
    require(torch.is_tensor(weight), "LM_HEAD_WEIGHT_MISSING")
    require(
        tuple(weight.shape)
        == (EXPECTED_VOCAB_SIZE, EXPECTED_HIDDEN_SIZE),
        f"LM_HEAD_SHAPE:{tuple(weight.shape)}",
    )

    require(model.training is False, "CAUSAL_LM_NOT_EVAL")
    require(
        not any(parameter.requires_grad for parameter in model.parameters()),
        "CAUSAL_LM_REQUIRES_GRAD",
    )

    return {
        "loading_info": clean_loading,
        "backbone_state_key_count": len(state),
        "backbone_canonical_sha256": backbone_sha,
        "expected_backbone_canonical_sha256":
            EXPECTED_BACKBONE_CANONICAL_SHA256,
        "lm_head_shape": list(weight.shape),
        "lm_head_weight_sha256": tensor_sha256(weight),
        "model_training": bool(model.training),
        "any_parameter_requires_grad": False,
    }


def load_causal_lm_identity(
    snapshot: Path,
    *,
    factory: Any | None = None,
    config_factory: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    if factory is None or config_factory is None:
        from transformers import MambaConfig, MambaForCausalLM

        if factory is None:
            factory = MambaForCausalLM
        if config_factory is None:
            config_factory = MambaConfig

    validate_snapshot_and_config(snapshot)

    config = config_factory.from_pretrained(
        snapshot,
        local_files_only=True,
    )

    # This stage performs no model forward. Disable the optional fast-kernel
    # execution path so identity validation remains CPU-only and independent
    # of CUDA kernel availability. This does not alter any loaded parameter.
    if hasattr(config, "use_mamba_kernels"):
        config.use_mamba_kernels = False

    loaded = factory.from_pretrained(
        snapshot,
        config=config,
        local_files_only=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    require(
        isinstance(loaded, tuple) and len(loaded) == 2,
        "CAUSAL_LM_LOADING_RETURN",
    )
    model, loading_info = loaded
    require(isinstance(loading_info, Mapping), "LOADING_INFO_TYPE")

    model.to("cpu")
    model.eval()
    model.requires_grad_(False)

    identity = validate_loaded_causal_lm(
        model,
        loading_info=loading_info,
    )
    identity["identity_load_device"] = "cpu"
    identity["use_mamba_kernels_for_identity_load"] = False
    identity["model_forward_count"] = 0
    identity["generation_call_count"] = 0
    return model, identity


def build_gate_manifest(
    *,
    snapshot: Path,
    check_causal_lm: bool,
) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    snapshot_and_config = validate_snapshot_and_config(snapshot)
    tokenizer, tokenizer_provenance = geom.load_tokenizer(snapshot)
    grammar = build_grammar_manifest(
        tokenizer,
        tokenizer_provenance=tokenizer_provenance,
    )

    causal_lm_identity: dict[str, Any]
    if check_causal_lm:
        model, causal_lm_identity = load_causal_lm_identity(snapshot)
        del model
        causal_lm_identity["status"] = "PASS"
    else:
        causal_lm_identity = {
            "status": "NOT_RUN",
            "model_forward_count": 0,
            "generation_call_count": 0,
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT_PASS,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "expected_branch": EXPECTED_BRANCH,
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "snapshot": snapshot_and_config,
        "causal_lm_identity": causal_lm_identity,
        "frozen_causal_objects": {
            "selected_plane": SELECTED_PLANE,
            "control_plane": CONTROL_PLANE,
            "early_block": EARLY_BLOCK,
            "late_block": LATE_BLOCK,
        },
        "commitment_grammar": grammar,
        "scientific_generation_executed": False,
        "scientific_model_forward_count": 0,
        "cuda_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "p_value_count_added": 0,
        "generation_response_inspected": False,
    }
    manifest["manifest_sha256"] = sha256_bytes(
        canonical_json_bytes(manifest)
    )
    return manifest


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only fail-closed gate for the 370M causal-LM identity "
            "and finite pre-emission commitment grammar. No generation."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Exact local HF snapshot directory for the frozen 370M revision.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON manifest path.",
    )
    parser.add_argument(
        "--check-causal-lm",
        action="store_true",
        help=(
            "Also instantiate MambaForCausalLM on CPU and verify that its "
            "backbone canonical hash exactly matches the frozen 370M backbone."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_gate_manifest(
        snapshot=args.snapshot,
        check_causal_lm=bool(args.check_causal_lm),
    )
    write_manifest(args.output, manifest)

    print(f"RESULT={manifest['result']}")
    print(f"HF_REVISION={manifest['hf_revision']}")
    print(
        "GRAMMAR_SHA256="
        + str(manifest["commitment_grammar"]["grammar_sha256"])
    )
    print(
        "COMMON_PREFIX_TOKENS="
        + str(
            manifest["commitment_grammar"][
                "longest_common_prefix_length"
            ]
        )
    )
    print(
        "DECISIVE_TSTAR="
        + json.dumps(
            manifest["commitment_grammar"][
                "decisive_t_star_zero_based"
            ],
            sort_keys=True,
        )
    )
    print(
        "CAUSAL_LM_IDENTITY="
        + str(manifest["causal_lm_identity"]["status"])
    )
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("GENERATION_RESPONSE_INSPECTED=False")
    print("CUDA_EXECUTED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
