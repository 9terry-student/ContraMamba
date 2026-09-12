from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


HISTORICAL_SOURCE_COMMIT = "3e0e9a435068c552abf20f3a74e0c3eccca344a3"
R3_AUTHORITY_COMMIT = "b5bd5491ee0c77d2b407f07b68ccea398ab65da8"

HISTORICAL_MODEL_SHA256 = (
    "8c365bfa857157d91f363358d5db3abaab425dec3e0d7c62683b4207a589b6a5"
)
HISTORICAL_HEADS_TREE = "68d26855aa511fcd41d6f395ae5f87177a162678"

MODEL_NAME = "state-spaces/mamba-130m-hf"
ARCHITECTURE = "v6b_minimal"
DECISION_MODE = "explicit_product"
GRADIENT_OWNERSHIP_MODE = "edge_specific"

HISTORICAL_FRAME_SIZE = 128
HISTORICAL_PREDICATE_SIZE = 128
HISTORICAL_SUFFICIENCY_SIZE = 128
HISTORICAL_ENERGY_SIZE = 64
HISTORICAL_DROPOUT = 0.1
HISTORICAL_FREEZE_ENCODER = True
HISTORICAL_FREEZE_A_LOG = True
HISTORICAL_REASON_ROUTER_EPSILON = 1e-8
HISTORICAL_USE_TEMPORAL_COMPARATOR = False
HISTORICAL_USE_PREDICATE_COMPARATOR = False
HISTORICAL_ALPHA_TEMPORAL_INIT = 1.25
HISTORICAL_ALPHA_PREDICATE_INIT = 1.25
HISTORICAL_GRADIENT_OWNERSHIP_LAMBDA = None

MAX_LENGTH = 128
CLAIM_BUDGET = 63
EVIDENCE_BUDGET = 64
EOS_TOKEN_ID = 0
EFFECTIVE_PAD_TOKEN_ID = 0

EXTERNAL_CLASS_ORDER = (
    "REFUTE",
    "NOT_ENTITLED",
    "SUPPORT",
)

EDGE_GRADIENT_LAMBDA_KEYS = (
    "F_TO_P",
    "F_TO_S",
    "P_TO_S",
    "F_TO_Q",
    "P_TO_Q",
    "S_TO_Q",
    "F_TO_D",
    "P_TO_D",
    "S_TO_D",
    "Q_TO_D",
)

CANONICAL_GEN4_ARTIFACT = Path(
    "reports/"
    "reason_router_gen4_six_cell_masked_slot_substitution_materialization_"
    "fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/"
    "gen4_six_cell_masked_slot_substitution.jsonl"
)

CANONICAL_GEN4_ARTIFACT_SHA256 = (
    "b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7"
)
CANONICAL_GEN4_ARTIFACT_BYTES = 1465573
CANONICAL_GEN4_ROWS = 1800
CANONICAL_GEN4_SOURCE_PAIRS = 300

CANONICAL_CELLS = (
    "C0_SHAM",
    "C1_TITLE",
    "C2_NAME",
    "C3_ROLE",
    "C4_PREDICATE",
    "C5_TITLE_NAME",
)

TOKENIZER_REVISION_REFERENCE = (
    "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
)

TOKENIZER_FILE_SHA256 = {
    "tokenizer.json":
        "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf",
    "tokenizer_config.json":
        "9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb",
    "special_tokens_map.json":
        "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
}

SEEDS = (180, 181, 182)

ARMS = (
    "G3-GROUP-D-HALF",
    "G3-GROUP-Q-D-HALF",
    "G3-GROUP-Q-HALF",
    "G3-GROUP-U-D-HALF",
    "G3-GROUP-U-HALF",
    "G3-GROUP-U-Q-HALF",
)

EDGE_GRADIENT_LAMBDAS_BY_ARM = {
    "G3-GROUP-D-HALF": {
        "F_TO_P": 1.0,
        "F_TO_S": 1.0,
        "P_TO_S": 1.0,
        "F_TO_Q": 1.0,
        "P_TO_Q": 1.0,
        "S_TO_Q": 1.0,
        "F_TO_D": 0.5,
        "P_TO_D": 0.5,
        "S_TO_D": 0.5,
        "Q_TO_D": 0.5,
    },
    "G3-GROUP-Q-D-HALF": {
        "F_TO_P": 1.0,
        "F_TO_S": 1.0,
        "P_TO_S": 1.0,
        "F_TO_Q": 0.5,
        "P_TO_Q": 0.5,
        "S_TO_Q": 0.5,
        "F_TO_D": 0.5,
        "P_TO_D": 0.5,
        "S_TO_D": 0.5,
        "Q_TO_D": 0.5,
    },
    "G3-GROUP-Q-HALF": {
        "F_TO_P": 1.0,
        "F_TO_S": 1.0,
        "P_TO_S": 1.0,
        "F_TO_Q": 0.5,
        "P_TO_Q": 0.5,
        "S_TO_Q": 0.5,
        "F_TO_D": 1.0,
        "P_TO_D": 1.0,
        "S_TO_D": 1.0,
        "Q_TO_D": 1.0,
    },
    "G3-GROUP-U-D-HALF": {
        "F_TO_P": 0.5,
        "F_TO_S": 0.5,
        "P_TO_S": 0.5,
        "F_TO_Q": 1.0,
        "P_TO_Q": 1.0,
        "S_TO_Q": 1.0,
        "F_TO_D": 0.5,
        "P_TO_D": 0.5,
        "S_TO_D": 0.5,
        "Q_TO_D": 0.5,
    },
    "G3-GROUP-U-HALF": {
        "F_TO_P": 0.5,
        "F_TO_S": 0.5,
        "P_TO_S": 0.5,
        "F_TO_Q": 1.0,
        "P_TO_Q": 1.0,
        "S_TO_Q": 1.0,
        "F_TO_D": 1.0,
        "P_TO_D": 1.0,
        "S_TO_D": 1.0,
        "Q_TO_D": 1.0,
    },
    "G3-GROUP-U-Q-HALF": {
        "F_TO_P": 0.5,
        "F_TO_S": 0.5,
        "P_TO_S": 0.5,
        "F_TO_Q": 0.5,
        "P_TO_Q": 0.5,
        "S_TO_Q": 0.5,
        "F_TO_D": 1.0,
        "P_TO_D": 1.0,
        "S_TO_D": 1.0,
        "Q_TO_D": 1.0,
    },
}

CHECKPOINT_SHA256 = {
    (180, "G3-GROUP-D-HALF"):
        "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f",
    (180, "G3-GROUP-Q-D-HALF"):
        "2e51f64702a3ebf21d5d8e8aa84745b62b3faa01112b5b8f10525ba6435dbc8c",
    (180, "G3-GROUP-Q-HALF"):
        "eb349aefca6d992df42f6239e7cf642d560755c0b1819397dba1d746b33bd8e3",
    (180, "G3-GROUP-U-D-HALF"):
        "08654abb9c1ec67d42fa1b3464f19298f21ff79b866fb0cf8b7a97d59a45ff86",
    (180, "G3-GROUP-U-HALF"):
        "a8cd296136816f806394ca98d6433bfa560f5691ab37e661347c2db838966708",
    (180, "G3-GROUP-U-Q-HALF"):
        "0701ce934ae3ef34cd9f9d229c9321599b4ca150db8dabc3c8a740668b8f0aad",

    (181, "G3-GROUP-D-HALF"):
        "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f",
    (181, "G3-GROUP-Q-D-HALF"):
        "390b4fe3266d8eddebe74d9732321d1f96e2a7095ecae67b6155a2d535b655ba",
    (181, "G3-GROUP-Q-HALF"):
        "3b5044fddb7f542c9e06a318a5a81a731d94475f7f67b7e5c5a7787ab3af0ba6",
    (181, "G3-GROUP-U-D-HALF"):
        "7adffc577e00b9a9150bca28ed83b35eb5574458f71d5bc276ebd8f557b00e4d",
    (181, "G3-GROUP-U-HALF"):
        "e2a9fd1ca6e50856b2349fc5bc915c54e6a71848aaa8c59aaa1f8c8647e89699",
    (181, "G3-GROUP-U-Q-HALF"):
        "1be3be2ddd13762d36c69ef16ccbdd0ee4bd5ad732eff46e7a66cab703c2db50",

    (182, "G3-GROUP-D-HALF"):
        "f9db48a3b3b9fdc6df4e2bb2086d11fd80fd595e6096c0095d1992f6c7d777f2",
    (182, "G3-GROUP-Q-D-HALF"):
        "cb1f4812d11643089bb87064c436b2e890554435254c961e5ed3f766b61b412b",
    (182, "G3-GROUP-Q-HALF"):
        "67d0cbf855b24a291f55ce87425dcd4d77b5f7a59fb119c57c261c6378a4342e",
    (182, "G3-GROUP-U-D-HALF"):
        "f1d84bab31f9080f0f3cfc6d0ee49cdc2743ad7c8c3a620bee7f576ca32ebef1",
    (182, "G3-GROUP-U-HALF"):
        "47b43899119a0a450e0b5cf8134ade521d8cea9ca568110b32223de6109ef5a4",
    (182, "G3-GROUP-U-Q-HALF"):
        "129be6e930b5f7e6737ad671a9646c867d150e8751907bb1abfd3fe64570669f",
}


REQUIRED_ROW_FIELDS = (
    "row_id",
    "source_pair_id",
    "contrast_cell_id",
    "claim",
    "evidence",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def expected_checkpoint_sha256(seed: int, arm: str) -> str:
    if seed not in SEEDS:
        raise ValueError(f"unknown frozen evaluator seed: {seed}")
    if arm not in ARMS:
        raise ValueError(f"unknown frozen evaluator arm: {arm}")
    try:
        return CHECKPOINT_SHA256[(seed, arm)]
    except KeyError as exc:
        raise ValueError(
            f"missing frozen checkpoint registry entry: seed={seed} arm={arm}"
        ) from exc


def expected_edge_gradient_lambdas(
    arm: str,
) -> dict[str, float]:
    if arm not in ARMS:
        raise ValueError(
            f"unknown frozen evaluator arm: {arm}"
        )

    return validate_edge_gradient_lambdas(
        dict(EDGE_GRADIENT_LAMBDAS_BY_ARM[arm])
    )


def historical_model_constructor_kwargs() -> dict[str, Any]:
    """Return the recovered historical grouped production constructor contract."""
    return {
        "frame_size": HISTORICAL_FRAME_SIZE,
        "predicate_size": HISTORICAL_PREDICATE_SIZE,
        "sufficiency_size": HISTORICAL_SUFFICIENCY_SIZE,
        "energy_size": HISTORICAL_ENERGY_SIZE,
        "dropout": HISTORICAL_DROPOUT,
        "freeze_a_log": HISTORICAL_FREEZE_A_LOG,
        "decision_mode": DECISION_MODE,
        "reason_router_epsilon":
            HISTORICAL_REASON_ROUTER_EPSILON,
        "use_temporal_comparator":
            HISTORICAL_USE_TEMPORAL_COMPARATOR,
        "use_predicate_comparator":
            HISTORICAL_USE_PREDICATE_COMPARATOR,
        "alpha_temporal_init":
            HISTORICAL_ALPHA_TEMPORAL_INIT,
        "alpha_predicate_init":
            HISTORICAL_ALPHA_PREDICATE_INIT,
    }


def authenticate_checkpoint(
    path: str | Path,
    *,
    seed: int,
    arm: str,
) -> str:
    path = Path(path)
    expected = expected_checkpoint_sha256(seed, arm)

    if not path.is_file():
        raise FileNotFoundError(path)

    observed = sha256_file(path)

    if observed != expected:
        raise ValueError(
            "checkpoint SHA256 mismatch before deserialization: "
            f"seed={seed} arm={arm} expected={expected} observed={observed}"
        )

    return observed


def authenticated_checkpoint_load(
    path: str | Path,
    *,
    seed: int,
    arm: str,
    loader: Callable[[Path], Any],
) -> tuple[str, Any]:
    path = Path(path)

    observed = authenticate_checkpoint(
        path,
        seed=seed,
        arm=arm,
    )

    # The callback is intentionally reached only after exact byte authentication.
    payload = loader(path)

    return observed, payload


def validate_edge_gradient_lambdas(
    values: Mapping[str, float],
) -> dict[str, float]:
    if set(values) != set(EDGE_GRADIENT_LAMBDA_KEYS):
        missing = sorted(set(EDGE_GRADIENT_LAMBDA_KEYS) - set(values))
        extra = sorted(set(values) - set(EDGE_GRADIENT_LAMBDA_KEYS))
        raise ValueError(
            "edge-gradient map must contain exactly frozen keys; "
            f"missing={missing} extra={extra}"
        )

    result: dict[str, float] = {}

    for key in EDGE_GRADIENT_LAMBDA_KEYS:
        value = float(values[key])

        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"invalid edge-gradient lambda {key}={value}"
            )

        result[key] = value

    return result


def validate_gen4_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_canonical_shape: bool = False,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"row {index} is not a mapping")

        missing = [key for key in REQUIRED_ROW_FIELDS if key not in raw]

        if missing:
            raise ValueError(
                f"row {index} missing required fields: {missing}"
            )

        if not isinstance(raw["claim"], str):
            raise ValueError(f"row {index} claim must be str")

        if not isinstance(raw["evidence"], str):
            raise ValueError(f"row {index} evidence must be str")

        row = dict(raw)

        for key in ("row_id", "source_pair_id", "contrast_cell_id"):
            if row[key] is None or str(row[key]) == "":
                raise ValueError(
                    f"row {index} invalid identity field: {key}"
                )
            row[key] = str(row[key])

        normalized.append(row)

    row_ids = [row["row_id"] for row in normalized]

    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate row_id detected")

    pair_cell = [
        (row["source_pair_id"], row["contrast_cell_id"])
        for row in normalized
    ]

    if len(set(pair_cell)) != len(pair_cell):
        raise ValueError(
            "duplicate source_pair_id/contrast_cell_id detected"
        )

    if require_canonical_shape:
        if len(normalized) != CANONICAL_GEN4_ROWS:
            raise ValueError(
                f"canonical Gen4 row count must be {CANONICAL_GEN4_ROWS}"
            )

        pair_count = len({
            row["source_pair_id"]
            for row in normalized
        })

        if pair_count != CANONICAL_GEN4_SOURCE_PAIRS:
            raise ValueError(
                "canonical Gen4 source-pair count mismatch"
            )

        counts = Counter(
            row["contrast_cell_id"]
            for row in normalized
        )

        expected = {
            cell: 300
            for cell in CANONICAL_CELLS
        }

        if dict(counts) != expected:
            raise ValueError(
                f"canonical contrast-cell counts mismatch: {dict(counts)}"
            )

    return normalized


def load_canonical_gen4_artifact(
    path: str | Path = CANONICAL_GEN4_ARTIFACT,
) -> list[dict[str, Any]]:
    path = Path(path)

    raw = path.read_bytes()

    observed_sha = sha256_bytes(raw)

    if observed_sha != CANONICAL_GEN4_ARTIFACT_SHA256:
        raise ValueError(
            "canonical Gen4 artifact SHA256 mismatch: "
            f"{observed_sha}"
        )

    if len(raw) != CANONICAL_GEN4_ARTIFACT_BYTES:
        raise ValueError(
            "canonical Gen4 artifact byte-count mismatch"
        )

    rows: list[dict[str, Any]] = []

    for line_no, line in enumerate(
        raw.decode("utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        value = json.loads(line)

        if not isinstance(value, dict):
            raise ValueError(
                f"canonical JSONL row {line_no} is not an object"
            )

        rows.append(value)

    return validate_gen4_rows(
        rows,
        require_canonical_shape=True,
    )


def canonical_tokenizer_snapshot_dir(
    home: str | Path | None = None,
) -> Path:
    root = Path(home) if home is not None else Path.home()

    return (
        root
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--state-spaces--mamba-130m-hf"
        / "snapshots"
        / TOKENIZER_REVISION_REFERENCE
    )


def authenticate_canonical_tokenizer(
    snapshot_dir: str | Path,
) -> dict[str, str]:
    snapshot_dir = Path(snapshot_dir)

    if snapshot_dir.name != TOKENIZER_REVISION_REFERENCE:
        raise ValueError(
            "tokenizer snapshot revision directory mismatch"
        )

    observed: dict[str, str] = {}

    for filename, expected in TOKENIZER_FILE_SHA256.items():
        path = snapshot_dir / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        digest = sha256_file(path)

        if digest != expected:
            raise ValueError(
                f"tokenizer file SHA256 mismatch: "
                f"{filename} expected={expected} observed={digest}"
            )

        observed[filename] = digest

    return observed


def _token_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value

    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, str):
            return content

    return None


def load_canonical_tokenizer(
    snapshot_dir: str | Path | None = None,
) -> Any:
    if snapshot_dir is None:
        snapshot_dir = canonical_tokenizer_snapshot_dir()

    snapshot_dir = Path(snapshot_dir)

    authenticate_canonical_tokenizer(snapshot_dir)

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "tokenizers package is required"
        ) from exc

    config = json.loads(
        (snapshot_dir / "tokenizer_config.json").read_text(
            encoding="utf-8-sig"
        )
    )

    special = json.loads(
        (snapshot_dir / "special_tokens_map.json").read_text(
            encoding="utf-8-sig"
        )
    )

    eos_candidates = [
        _token_content(special.get("eos_token")),
        _token_content(config.get("eos_token")),
    ]
    eos_candidates = [
        value
        for value in eos_candidates
        if value is not None
    ]

    if not eos_candidates or len(set(eos_candidates)) != 1:
        raise ValueError(
            "canonical EOS token string is not uniquely recoverable"
        )

    tokenizer = Tokenizer.from_file(
        str(snapshot_dir / "tokenizer.json")
    )

    tokenizer.no_padding()
    tokenizer.no_truncation()

    eos_id = tokenizer.token_to_id(eos_candidates[0])

    if eos_id != EOS_TOKEN_ID:
        raise ValueError(
            f"canonical EOS token id mismatch: {eos_id}"
        )

    pad_candidates = [
        _token_content(special.get("pad_token")),
        _token_content(config.get("pad_token")),
    ]
    pad_candidates = [
        value
        for value in pad_candidates
        if value is not None
    ]

    if len(set(pad_candidates)) > 1:
        raise ValueError(
            "canonical PAD token declarations disagree"
        )

    if pad_candidates:
        pad_id = tokenizer.token_to_id(pad_candidates[0])
    else:
        pad_id = eos_id

    if pad_id != EFFECTIVE_PAD_TOKEN_ID:
        raise ValueError(
            f"effective PAD token id mismatch: {pad_id}"
        )

    return tokenizer


def construct_feature_tensors(
    claim_raw_ids: Sequence[int],
    evidence_raw_ids: Sequence[int],
) -> dict[str, torch.Tensor]:
    claim_ids = [int(value) for value in claim_raw_ids[:CLAIM_BUDGET]]
    evidence_ids = [
        int(value)
        for value in evidence_raw_ids[:EVIDENCE_BUDGET]
    ]

    if not claim_ids:
        raise ValueError("empty claim token span")

    if not evidence_ids:
        raise ValueError("empty evidence token span")

    sequence = claim_ids + [EOS_TOKEN_ID] + evidence_ids

    if len(sequence) > MAX_LENGTH:
        raise AssertionError(
            "63/1/64 construction exceeded 128 tokens"
        )

    padding = MAX_LENGTH - len(sequence)

    input_ids = torch.tensor(
        sequence + ([EFFECTIVE_PAD_TOKEN_ID] * padding),
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        ([True] * len(sequence)) + ([False] * padding),
        dtype=torch.bool,
    )

    claim_mask = torch.tensor(
        ([True] * len(claim_ids))
        + [False]
        + ([False] * len(evidence_ids))
        + ([False] * padding),
        dtype=torch.bool,
    )

    evidence_mask = torch.tensor(
        ([False] * len(claim_ids))
        + [False]
        + ([True] * len(evidence_ids))
        + ([False] * padding),
        dtype=torch.bool,
    )

    for tensor in (
        input_ids,
        attention_mask,
        claim_mask,
        evidence_mask,
    ):
        if tensor.shape != (MAX_LENGTH,):
            raise AssertionError(
                f"feature tensor shape drift: {tensor.shape}"
            )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "claim_mask": claim_mask,
        "evidence_mask": evidence_mask,
    }


def encode_gen4_row(
    row: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    validated = validate_gen4_rows([row])[0]

    claim_ids = tokenizer.encode(
        validated["claim"],
        add_special_tokens=False,
    ).ids

    evidence_ids = tokenizer.encode(
        validated["evidence"],
        add_special_tokens=False,
    ).ids

    features = construct_feature_tensors(
        claim_ids,
        evidence_ids,
    )

    return {
        "row_id": validated["row_id"],
        "source_pair_id": validated["source_pair_id"],
        "contrast_cell_id": validated["contrast_cell_id"],
        **features,
    }


def encode_gen4_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    validated = validate_gen4_rows(rows)

    encoded = [
        encode_gen4_row(row, tokenizer)
        for row in validated
    ]

    return {
        "row_id": [row["row_id"] for row in encoded],
        "source_pair_id": [
            row["source_pair_id"]
            for row in encoded
        ],
        "contrast_cell_id": [
            row["contrast_cell_id"]
            for row in encoded
        ],
        "input_ids": torch.stack([
            row["input_ids"]
            for row in encoded
        ]),
        "attention_mask": torch.stack([
            row["attention_mask"]
            for row in encoded
        ]),
        "claim_mask": torch.stack([
            row["claim_mask"]
            for row in encoded
        ]),
        "evidence_mask": torch.stack([
            row["evidence_mask"]
            for row in encoded
        ]),
    }


def encoded_coordinate_sha256(
    encoded: Mapping[str, Any],
) -> str:
    row_count = len(encoded["row_id"])

    serializable = []

    for index in range(row_count):
        serializable.append({
            "row_id": encoded["row_id"][index],
            "source_pair_id":
                encoded["source_pair_id"][index],
            "contrast_cell_id":
                encoded["contrast_cell_id"][index],
            "input_ids":
                encoded["input_ids"][index].tolist(),
            "attention_mask":
                encoded["attention_mask"][index].tolist(),
            "claim_mask":
                encoded["claim_mask"][index].tolist(),
            "evidence_mask":
                encoded["evidence_mask"][index].tolist(),
        })

    return sha256_bytes(canonical_json_bytes(serializable))


def build_historical_model_from_backbone(
    *,
    backbone: torch.nn.Module,
    arm: str,
    hidden_size: int | None = None,
) -> torch.nn.Module:
    # Lazy historical import: merely importing this adapter never constructs
    # or downloads a model.
    from contramamba.modeling_v6b_minimal_gen3_grouped_snapshot import (
        ContraMambaV6BMinimal,
    )

    constructor_kwargs = historical_model_constructor_kwargs()
    edge_map = expected_edge_gradient_lambdas(arm)

    model = ContraMambaV6BMinimal(
        model_name=MODEL_NAME,
        backbone=backbone,
        hidden_size=hidden_size,
        **constructor_kwargs,
    )

    if HISTORICAL_FREEZE_ENCODER:
        for parameter in model.mamba.parameters():
            parameter.requires_grad = False

    model.gradient_ownership_mode = GRADIENT_OWNERSHIP_MODE
    model.gradient_ownership_lambda = (
        HISTORICAL_GRADIENT_OWNERSHIP_LAMBDA
    )
    model.edge_gradient_lambdas = edge_map
    model.return_q_diagnostics = True

    return model


def historical_forward(
    model: torch.nn.Module,
    feature_batch: Mapping[str, torch.Tensor],
    *,
    arm: str,
) -> Mapping[str, Any]:
    edge_map = expected_edge_gradient_lambdas(arm)

    return model(
        input_ids=feature_batch["input_ids"],
        attention_mask=feature_batch["attention_mask"],
        claim_mask=feature_batch["claim_mask"],
        evidence_mask=feature_batch["evidence_mask"],
        decision_mode=DECISION_MODE,
        gradient_ownership_mode=GRADIENT_OWNERSHIP_MODE,
        edge_gradient_lambdas=edge_map,
        return_q_diagnostics=True,
    )


def strict_load_state_dict(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
) -> Any:
    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
    else:
        state = payload

    if not isinstance(state, Mapping):
        raise ValueError(
            "checkpoint payload does not contain a state dictionary"
        )

    return model.load_state_dict(
        state,
        strict=True,
    )


def serialize_model_outputs(
    output: Mapping[str, Any],
    metadata_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    arm: str,
    checkpoint_sha256: str,
) -> list[dict[str, Any]]:
    expected_sha = expected_checkpoint_sha256(seed, arm)

    if checkpoint_sha256 != expected_sha:
        raise ValueError(
            "serializer checkpoint SHA does not match frozen registry"
        )

    rows = validate_gen4_rows(metadata_rows)

    required = (
        "q_authorized",
        "entitlement_prob",
        "logits",
    )

    for key in required:
        if key not in output:
            raise ValueError(
                f"model output missing required key: {key}"
            )

    q_authorized = output["q_authorized"]
    entitlement_prob = output["entitlement_prob"]
    logits = output["logits"]

    if not isinstance(q_authorized, torch.Tensor):
        raise TypeError("q_authorized must be tensor")

    if not isinstance(entitlement_prob, torch.Tensor):
        raise TypeError("entitlement_prob must be tensor")

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be tensor")

    batch = len(rows)

    if q_authorized.shape != (batch,):
        raise ValueError("q_authorized shape mismatch")

    if entitlement_prob.shape != (batch,):
        raise ValueError("entitlement_prob shape mismatch")

    if logits.shape != (batch, 3):
        raise ValueError("final logits shape mismatch")

    prediction_ids = logits.argmax(dim=-1)

    if "predictions" in output:
        historical_predictions = output["predictions"]

        if (
            not isinstance(historical_predictions, torch.Tensor)
            or historical_predictions.shape != (batch,)
            or not torch.equal(
                historical_predictions.detach().cpu(),
                prediction_ids.detach().cpu(),
            )
        ):
            raise ValueError(
                "historical prediction tensor disagrees with final logits"
            )

    q_cpu = q_authorized.detach().cpu()
    entitlement_cpu = entitlement_prob.detach().cpu()
    logits_cpu = logits.detach().cpu()
    prediction_cpu = prediction_ids.detach().cpu()

    serialized: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        refute_logit = float(logits_cpu[index, 0].item())
        ne_logit = float(logits_cpu[index, 1].item())
        support_logit = float(logits_cpu[index, 2].item())

        prediction_id = int(prediction_cpu[index].item())

        serialized.append({
            "seed": seed,
            "arm": arm,
            "checkpoint_sha256": checkpoint_sha256,
            "row_id": row["row_id"],
            "source_pair_id": row["source_pair_id"],
            "contrast_cell_id": row["contrast_cell_id"],
            "q_authorized":
                float(q_cpu[index].item()),
            "entitlement_prob":
                float(entitlement_cpu[index].item()),
            "refute_logit": refute_logit,
            "ne_logit": ne_logit,
            "support_logit": support_logit,
            "final_logits": [
                refute_logit,
                ne_logit,
                support_logit,
            ],
            "prediction_id": prediction_id,
            "prediction":
                EXTERNAL_CLASS_ORDER[prediction_id],
            "support_vs_best_nonsupport_logit_margin":
                support_logit
                - max(refute_logit, ne_logit),
        })

    return serialized


def output_key(
    row: Mapping[str, Any],
) -> tuple[int, str, str]:
    return (
        int(row["seed"]),
        str(row["arm"]),
        str(row["row_id"]),
    )


def validate_complete_matrix(
    output_rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
    evaluator_keys: Sequence[tuple[int, str]],
) -> None:
    expected_metadata = validate_gen4_rows(expected_rows)

    evaluator_set = {
        (int(seed), str(arm))
        for seed, arm in evaluator_keys
    }

    if len(evaluator_set) != len(evaluator_keys):
        raise ValueError("duplicate evaluator key")

    row_meta = {
        row["row_id"]: (
            row["source_pair_id"],
            row["contrast_cell_id"],
        )
        for row in expected_metadata
    }

    expected = {
        (seed, arm, row_id)
        for seed, arm in evaluator_set
        for row_id in row_meta
    }

    observed: set[tuple[int, str, str]] = set()

    pair_counts: Counter[str] = Counter()

    for raw in output_rows:
        key = output_key(raw)

        if key in observed:
            raise ValueError(
                f"duplicate evaluator-row key: {key}"
            )

        seed, arm, row_id = key

        if (seed, arm) not in evaluator_set:
            raise ValueError(
                f"unexpected evaluator key: {(seed, arm)}"
            )

        if row_id not in row_meta:
            raise ValueError(
                f"unexpected row_id: {row_id}"
            )

        expected_pair, expected_cell = row_meta[row_id]

        if str(raw.get("source_pair_id")) != expected_pair:
            raise ValueError(
                f"source_pair_id mismatch for row_id={row_id}"
            )

        if str(raw.get("contrast_cell_id")) != expected_cell:
            raise ValueError(
                f"contrast_cell_id mismatch for row_id={row_id}"
            )

        observed.add(key)
        pair_counts[expected_pair] += 1

    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)

        raise ValueError(
            "incomplete evaluator matrix: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    expected_pair_rows = Counter(
        row["source_pair_id"]
        for row in expected_metadata
    )

    for pair_id, row_count in expected_pair_rows.items():
        expected_count = row_count * len(evaluator_set)

        if pair_counts[pair_id] != expected_count:
            raise ValueError(
                f"pair matrix count mismatch: "
                f"pair={pair_id} "
                f"expected={expected_count} "
                f"observed={pair_counts[pair_id]}"
            )


def build_provenance_record(
    *,
    adapter_source_commit: str,
    python_version: str,
    torch_version: str,
    transformers_version: str,
    device: str,
    dtype: str,
    batch_size: int,
    seed: int,
    arm: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    expected_sha = expected_checkpoint_sha256(seed, arm)

    if checkpoint_sha256 != expected_sha:
        raise ValueError(
            "provenance checkpoint SHA mismatch"
        )

    return {
        "adapter_source_commit": adapter_source_commit,
        "historical_source_commit": HISTORICAL_SOURCE_COMMIT,
        "historical_snapshot_sha256": HISTORICAL_MODEL_SHA256,
        "historical_heads_tree": HISTORICAL_HEADS_TREE,
        "canonical_gen4_artifact_sha256":
            CANONICAL_GEN4_ARTIFACT_SHA256,
        "canonical_tokenizer_revision_reference":
            TOKENIZER_REVISION_REFERENCE,
        "canonical_tokenizer_file_sha256":
            dict(TOKENIZER_FILE_SHA256),
        "historical_wrapper_runtime_equivalence":
            "UNRESOLVED",
        "python_version": python_version,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "device": device,
        "dtype": dtype,
        "batch_size": int(batch_size),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": int(seed),
        "arm": arm,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gen4 six-cell Tier-2 dedicated inference adapter. "
            "Authority remains phase-specific."
        )
    )

    subparsers = parser.add_subparsers(
        dest="operation",
        required=True,
    )

    validate_parser = subparsers.add_parser(
        "validate-input",
        help=(
            "Authenticate and encode canonical Gen4 input only. "
            "No model or checkpoint access."
        ),
    )

    validate_parser.add_argument(
        "--artifact",
        default=str(CANONICAL_GEN4_ARTIFACT),
    )

    validate_parser.add_argument(
        "--tokenizer-snapshot",
        default=None,
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.operation == "validate-input":
        rows = load_canonical_gen4_artifact(
            args.artifact
        )

        tokenizer = load_canonical_tokenizer(
            args.tokenizer_snapshot
        )

        encoded = encode_gen4_rows(
            rows,
            tokenizer,
        )

        result = {
            "operation": "validate-input",
            "row_count": len(rows),
            "coordinate_sha256":
                encoded_coordinate_sha256(encoded),
            "model_instantiation": "NOT_PERFORMED",
            "checkpoint_load": "NOT_PERFORMED",
            "model_forward": "NOT_PERFORMED",
        }

        print(
            json.dumps(
                result,
                sort_keys=True,
                indent=2,
            )
        )

        return 0

    raise RuntimeError(
        f"unsupported operation: {args.operation}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
