#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


AUTHORITY_COMMIT = "7b12527289924c9979fe19231bb8e7e756c0234e"

DATASET_SHA256 = (
    "452a24ec32302b4db6100c0f5897533d726f3d4408662b26bc3089b0b46d3191"
)

CHECKPOINT_SHA256 = (
    "4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c"
)

TOKENIZER_REVISION = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"

TOKENIZER_JSON_SHA256 = (
    "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf"
)

KERNELS_VERSION = "0.10.2"

KERNEL_BUILD_VARIANT = (
    "torch210-cxx11-cu128-x86_64-linux"
)

MAMBA_KERNEL_BINARY_SHA256 = (
    "dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587"
)

CAUSAL_CONV_KERNEL_BINARY_SHA256 = (
    "6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6"
)

KERNEL_TRANSPORT_IDENTITY_STATUS = (
    "EXACT_FROZEN_BINARY_SHA256_MATCH"
)

MODEL_NAME = "state-spaces/mamba-130m-hf"
ARCHITECTURE = "v6b_minimal"
BACKBONE = "mamba"

TRAINING_SEED = 180
SPLIT_SEED = 8192

EXPECTED_ROWS = 5000
EXPECTED_CASE_IDS = 1497

MAX_LENGTH = 128
CLAIM_BUDGET = 63
EVIDENCE_BUDGET = 64

CONFIDENCE_THRESHOLD = 0.5

CLASS_ORDER = (
    "REFUTE",
    "NOT_ENTITLED",
    "SUPPORT",
)

DECISIVE_LABELS = {
    "REFUTE",
    "SUPPORT",
}

RAW_REQUIRED_COLUMNS = {
    "raw_idx",
    "unique_id",
    "case_id",
    "label",
    "claim",
    "evidence",
}

SERIALIZED_ROW_FIELDS = {
    "raw_idx",
    "unique_id",
    "case_id",
    "gold_label",
    "pred_label",
    "final_logits",
    "final_probs",
    "confidence",
    "correct",
    "decisive_prediction",
    "confident",
    "confident_decisive_correct",
    "confident_decisive_wrong",
    "input_token_length",
    "claim_token_length",
    "evidence_token_length",
    "claim_truncated",
    "evidence_truncated",
}

FORBIDDEN_SERIALIZED_KEY_FRAGMENTS = (
    "native_state",
    "recurrent_state",
    "hidden_state",
    "hidden_states",
    "ssm_state",
    "cache_params",
    "trajectory",
    "post4_speed",
    "post4_turning",
    "post4_path_efficiency",
)


class CensusError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CensusError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def git_head(repo: Path) -> str:
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    require(
        proc.returncode == 0,
        "GIT_HEAD_READ_FAILED",
    )

    return proc.stdout.strip()


def git_is_clean(repo: Path) -> bool:
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    require(
        proc.returncode == 0,
        "GIT_STATUS_READ_FAILED",
    )

    return proc.stdout.strip() == ""


def canonicalize_vitaminc_label(value: Any) -> str:
    key = (
        str(value)
        .strip()
        .upper()
        .replace("_", " ")
        .replace("-", " ")
    )

    key = " ".join(key.split())

    mapping = {
        "SUPPORT": "SUPPORT",
        "SUPPORTS": "SUPPORT",
        "REFUTE": "REFUTE",
        "REFUTES": "REFUTE",
        "NEI": "NOT_ENTITLED",
        "NOT ENOUGH INFO": "NOT_ENTITLED",
        "NOT ENTITLED": "NOT_ENTITLED",
    }

    require(
        key in mapping,
        f"UNMAPPED_VITAMINC_LABEL:{value!r}",
    )

    return mapping[key]


def read_dataset(path: Path) -> list[dict[str, str]]:
    require(
        path.is_file(),
        f"DATASET_MISSING:{path}",
    )

    require(
        sha256_file(path) == DATASET_SHA256,
        "DATASET_SHA256_MISMATCH",
    )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        require(
            reader.fieldnames is not None,
            "DATASET_HEADER_MISSING",
        )

        missing = sorted(
            RAW_REQUIRED_COLUMNS
            - set(reader.fieldnames)
        )

        require(
            not missing,
            "DATASET_MISSING_COLUMNS:"
            + ",".join(missing),
        )

        rows = [
            dict(row)
            for row in reader
        ]

    require(
        len(rows) == EXPECTED_ROWS,
        f"DATASET_ROW_COUNT:{len(rows)}",
    )

    raw_indices = [
        int(row["raw_idx"])
        for row in rows
    ]

    require(
        len(raw_indices)
        == len(set(raw_indices)),
        "RAW_IDX_NOT_UNIQUE",
    )

    require(
        sorted(raw_indices)
        == list(range(EXPECTED_ROWS)),
        "RAW_IDX_NOT_CONTIGUOUS",
    )

    unique_ids = [
        row["unique_id"]
        for row in rows
    ]

    require(
        all(unique_ids),
        "EMPTY_UNIQUE_ID",
    )

    require(
        len(unique_ids)
        == len(set(unique_ids)),
        "UNIQUE_ID_NOT_UNIQUE",
    )

    case_ids = {
        row["case_id"]
        for row in rows
    }

    require(
        "" not in case_ids,
        "EMPTY_CASE_ID",
    )

    require(
        len(case_ids)
        == EXPECTED_CASE_IDS,
        f"CASE_ID_COUNT:{len(case_ids)}",
    )

    for row in rows:
        require(
            bool(row["claim"].strip()),
            f"EMPTY_CLAIM:{row['raw_idx']}",
        )

        require(
            bool(row["evidence"].strip()),
            f"EMPTY_EVIDENCE:{row['raw_idx']}",
        )

        canonicalize_vitaminc_label(
            row["label"]
        )

    return rows


def external_flags(
    row_count: int,
) -> tuple[list[int], list[int]]:
    require(
        row_count >= 0,
        "NEGATIVE_ROW_COUNT",
    )

    # Frozen external identity:
    # intervention_type = stage43b1_external_factver
    #
    # controlled_heuristic semantics:
    # temporal -> all zero
    # predicate -> 1 only for predicate_swap
    #
    # Therefore both are exactly zero here.

    return (
        [0] * row_count,
        [0] * row_count,
    )


def maximum_case_level_pair_capacity(
    wrong_rows: Sequence[dict[str, Any]],
    correct_control_rows: Sequence[dict[str, Any]],
) -> int:
    """Maximum predicted-class-compatible matching over independent case IDs.

    Each wrong case_id and control case_id may be used at most once.
    An edge exists iff the two cases have at least one eligible row with
    the same decisive predicted class.
    """

    wrong_labels: dict[str, set[str]] = {}
    control_labels: dict[str, set[str]] = {}

    for row in wrong_rows:
        label = str(row["pred_label"])
        require(
            label in DECISIVE_LABELS,
            f"NONDECISIVE_WRONG_ROW:{label}",
        )
        wrong_labels.setdefault(
            str(row["case_id"]),
            set(),
        ).add(label)

    for row in correct_control_rows:
        label = str(row["pred_label"])
        require(
            label in DECISIVE_LABELS,
            f"NONDECISIVE_CONTROL_ROW:{label}",
        )
        control_labels.setdefault(
            str(row["case_id"]),
            set(),
        ).add(label)

    overlap = (
        set(wrong_labels)
        & set(control_labels)
    )

    require(
        not overlap,
        "WRONG_CONTROL_CASE_OVERLAP:"
        + ",".join(sorted(overlap)),
    )

    adjacency: dict[str, list[str]] = {
        wrong_case: sorted(
            control_case
            for control_case, control_case_labels
            in control_labels.items()
            if wrong_case_labels
            & control_case_labels
        )
        for wrong_case, wrong_case_labels
        in wrong_labels.items()
    }

    matched_control_to_wrong: dict[str, str] = {}

    def augment(
        wrong_case: str,
        seen_controls: set[str],
    ) -> bool:
        for control_case in adjacency[wrong_case]:
            if control_case in seen_controls:
                continue

            seen_controls.add(control_case)

            incumbent = matched_control_to_wrong.get(
                control_case
            )

            if (
                incumbent is None
                or augment(
                    incumbent,
                    seen_controls,
                )
            ):
                matched_control_to_wrong[
                    control_case
                ] = wrong_case
                return True

        return False

    matched = 0

    for wrong_case in sorted(
        adjacency,
        key=lambda case_id: (
            len(adjacency[case_id]),
            case_id,
        ),
    ):
        if augment(
            wrong_case,
            set(),
        ):
            matched += 1

    return matched


def encode_one(
    tokenizer: Any,
    row: dict[str, str],
) -> dict[str, Any]:
    claim_full = tokenizer.encode(
        row["claim"],
        add_special_tokens=False,
        truncation=False,
    )

    evidence_full = tokenizer.encode(
        row["evidence"],
        add_special_tokens=False,
        truncation=False,
    )

    require(
        bool(claim_full),
        f"EMPTY_CLAIM_TOKENIZATION:{row['raw_idx']}",
    )

    require(
        bool(evidence_full),
        f"EMPTY_EVIDENCE_TOKENIZATION:{row['raw_idx']}",
    )

    claim_ids = tokenizer.encode(
        row["claim"],
        add_special_tokens=False,
        truncation=True,
        max_length=CLAIM_BUDGET,
    )

    evidence_ids = tokenizer.encode(
        row["evidence"],
        add_special_tokens=False,
        truncation=True,
        max_length=EVIDENCE_BUDGET,
    )

    separator_id = tokenizer.eos_token_id

    if separator_id is None:
        separator_id = tokenizer.pad_token_id

    require(
        separator_id is not None,
        "TOKENIZER_NO_EOS_OR_PAD",
    )

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else separator_id
    )

    ids = (
        list(claim_ids)
        + [int(separator_id)]
        + list(evidence_ids)
    )

    require(
        len(ids) <= MAX_LENGTH,
        f"ENCODED_TOO_LONG:{row['raw_idx']}:{len(ids)}",
    )

    evidence_start = len(claim_ids) + 1
    padding = MAX_LENGTH - len(ids)

    return {
        "input_ids":
            ids
            + [int(pad_id)] * padding,

        "attention_mask":
            [1] * len(ids)
            + [0] * padding,

        "claim_mask":
            [1] * len(claim_ids)
            + [0] * (
                MAX_LENGTH
                - len(claim_ids)
            ),

        "evidence_mask":
            [0] * evidence_start
            + [1] * len(evidence_ids)
            + [0] * (
                MAX_LENGTH
                - evidence_start
                - len(evidence_ids)
            ),

        "input_token_length":
            len(ids),

        "claim_token_length":
            len(claim_ids),

        "evidence_token_length":
            len(evidence_ids),

        "claim_truncated":
            len(claim_full)
            > CLAIM_BUDGET,

        "evidence_truncated":
            len(evidence_full)
            > EVIDENCE_BUDGET,
    }


def _lookup(
    metadata: dict[str, Any],
    paths: Sequence[Sequence[str]],
) -> Any:
    for path in paths:
        current: Any = metadata
        found = True

        for key in path:
            if (
                not isinstance(current, dict)
                or key not in current
            ):
                found = False
                break

            current = current[key]

        if (
            found
            and current is not None
        ):
            return current

    return None


def validate_checkpoint_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    require(
        isinstance(metadata, dict),
        "CHECKPOINT_METADATA_MISSING",
    )

    observed = {
        "architecture":
            _lookup(
                metadata,
                (
                    ("architecture",),
                    ("training_args", "architecture"),
                ),
            ),

        "backbone":
            _lookup(
                metadata,
                (
                    ("backbone",),
                    ("training_args", "backbone"),
                ),
            ),

        "model_name":
            _lookup(
                metadata,
                (
                    ("model_name",),
                    ("training_args", "model_name"),
                ),
            ),

        "training_seed":
            _lookup(
                metadata,
                (
                    ("training_seed",),
                    ("seed",),
                    ("training_args", "seed"),
                ),
            ),

        "split_seed":
            _lookup(
                metadata,
                (
                    ("resolved_split_seed",),
                    (
                        "split_identity",
                        "resolved_split_seed",
                    ),
                    (
                        "training_args",
                        "resolved_split_seed",
                    ),
                    (
                        "training_args",
                        "split_seed",
                    ),
                ),
            ),

        "reason_router_arm":
            _lookup(
                metadata,
                (
                    ("reason_router_arm",),
                    (
                        "training_args",
                        "reason_router_arm",
                    ),
                ),
            ),

        "reason_router_mode":
            _lookup(
                metadata,
                (
                    ("reason_router_mode",),
                    ("reason_router_composer",),
                    (
                        "training_args",
                        "resolved_reason_router_mode",
                    ),
                    (
                        "training_args",
                        "reason_router_mode",
                    ),
                ),
            ),

        "gradient_ownership_mode":
            _lookup(
                metadata,
                (
                    ("gradient_ownership_mode",),
                    (
                        "training_args",
                        "resolved_gradient_ownership_mode",
                    ),
                    (
                        "training_args",
                        "gradient_ownership_mode",
                    ),
                ),
            ),

        "max_length":
            _lookup(
                metadata,
                (
                    ("max_length",),
                    (
                        "training_args",
                        "max_length",
                    ),
                ),
            ),

        "freeze_encoder":
            _lookup(
                metadata,
                (
                    (
                        "training_args",
                        "freeze_encoder",
                    ),
                    ("freeze_encoder",),
                ),
            ),
    }

    expected = {
        "architecture":
            ARCHITECTURE,

        "backbone":
            BACKBONE,

        "model_name":
            MODEL_NAME,

        "training_seed":
            TRAINING_SEED,

        "split_seed":
            SPLIT_SEED,

        "reason_router_arm":
            "A0",

        "reason_router_mode":
            "explicit_product",

        "gradient_ownership_mode":
            "joint",

        "max_length":
            MAX_LENGTH,

        "freeze_encoder":
            True,
    }

    for key, expected_value in expected.items():
        actual = observed[key]

        require(
            actual is not None,
            f"CHECKPOINT_METADATA_FIELD_MISSING:{key}",
        )

        if key in {
            "training_seed",
            "split_seed",
            "max_length",
        }:
            actual = int(actual)

        if key == "freeze_encoder":
            require(
                type(actual) is bool,
                "CHECKPOINT_FREEZE_ENCODER_NOT_BOOL",
            )

        require(
            actual == expected_value,
            (
                "CHECKPOINT_METADATA_MISMATCH:"
                f"{key}:"
                f"expected={expected_value!r}:"
                f"actual={actual!r}"
            ),
        )

    return observed


def load_checkpoint_payload(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        path.is_file(),
        f"CHECKPOINT_MISSING:{path}",
    )

    require(
        sha256_file(path)
        == CHECKPOINT_SHA256,
        "CHECKPOINT_SHA256_MISMATCH",
    )

    import torch

    payload = torch.load(
        path,
        map_location="cpu",
    )

    require(
        isinstance(payload, dict),
        "CHECKPOINT_PAYLOAD_NOT_DICT",
    )

    state = payload.get(
        "model_state_dict"
    )

    metadata = payload.get(
        "metadata"
    )

    require(
        isinstance(state, dict)
        and bool(state),
        "CHECKPOINT_STATE_MISSING",
    )

    require(
        isinstance(metadata, dict),
        "CHECKPOINT_METADATA_MISSING",
    )

    validate_checkpoint_metadata(
        metadata
    )

    require(
        "alpha_temporal_raw"
        not in state,
        "UNEXPECTED_TEMPORAL_COMPARATOR_PARAMETER",
    )

    require(
        "alpha_predicate_raw"
        not in state,
        "UNEXPECTED_PREDICATE_COMPARATOR_PARAMETER",
    )

    return state, metadata


def load_tokenizer(
    snapshot: Path,
) -> Any:
    tokenizer_json = (
        snapshot
        / "tokenizer.json"
    )

    require(
        snapshot.is_dir(),
        f"TOKENIZER_SNAPSHOT_MISSING:{snapshot}",
    )

    require(
        tokenizer_json.is_file(),
        "TOKENIZER_JSON_MISSING",
    )

    require(
        sha256_file(tokenizer_json)
        == TOKENIZER_JSON_SHA256,
        "TOKENIZER_JSON_SHA256_MISMATCH",
    )

    from transformers import AutoTokenizer

    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(snapshot),
            local_files_only=True,
        )
    )

    if tokenizer.pad_token_id is None:
        require(
            tokenizer.eos_token_id
            is not None,
            "TOKENIZER_NO_EOS_FOR_PAD_NORMALIZATION",
        )

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    require(
        tokenizer.eos_token_id
        is not None,
        "TOKENIZER_EOS_MISSING",
    )

    return tokenizer


def build_model_from_local_config(
    snapshot: Path,
    device: Any,
) -> Any:
    from transformers import (
        MambaConfig,
        MambaModel,
    )

    from contramamba.modeling_v6b_minimal import (
        ContraMambaV6BMinimal,
    )

    config = (
        MambaConfig.from_pretrained(
            str(snapshot),
            local_files_only=True,
        )
    )

    config.use_mamba_kernels = bool(
        str(device).startswith("cuda")
    )

    backbone = MambaModel(
        config
    )

    model = ContraMambaV6BMinimal(
        backbone=backbone,
        hidden_size=int(
            config.hidden_size
        ),
        frame_size=128,
        predicate_size=128,
        sufficiency_size=128,
        energy_size=64,
        dropout=0.1,
        freeze_a_log=True,
        return_token_diagnostics=False,
        decision_mode="explicit_product",
        reason_router_epsilon=1e-8,
        use_temporal_comparator=False,
        use_predicate_comparator=False,
    )

    return model


def validate_kernel_constructor_calls(
    calls: Sequence[str],
    layer_count: int,
) -> dict[str, int]:
    require(
        layer_count > 0,
        "KERNEL_LAYER_COUNT_NONPOSITIVE",
    )

    observed = dict(
        sorted(
            Counter(
                calls
            ).items()
        )
    )

    expected = {
        "causal-conv1d":
            layer_count,

        "mamba-ssm":
            layer_count,
    }

    require(
        observed == expected,
        (
            "KERNEL_CONSTRUCTOR_COUNTS:"
            + json.dumps(
                observed,
                sort_keys=True,
            )
        ),
    )

    return observed


def exact_kernel_binary_identity(
    module: Any,
    *,
    label: str,
    expected_sha256: str,
) -> dict[str, Any]:
    module_file_raw = getattr(
        module,
        "__file__",
        None,
    )

    require(
        module_file_raw
        is not None,
        f"{label}_MODULE_FILE_MISSING",
    )

    module_file = Path(
        module_file_raw
    )

    variant_roots = [
        parent
        for parent
        in module_file.parents
        if parent.name
        == KERNEL_BUILD_VARIANT
    ]

    require(
        len(
            variant_roots
        )
        == 1,
        (
            f"{label}_BUILD_VARIANT_PATH:"
            f"{module_file}"
        ),
    )

    variant_root = (
        variant_roots[0]
    )

    exact_binaries = [
        candidate
        for candidate
        in sorted(
            variant_root.rglob(
                "*.so"
            )
        )
        if sha256_file(
            candidate
        )
        == expected_sha256
    ]

    require(
        len(
            exact_binaries
        )
        == 1,
        (
            f"{label}_EXACT_BINARY_COUNT:"
            f"{len(exact_binaries)}"
        ),
    )

    binary = (
        exact_binaries[0]
    )

    return {
        "module_file":
            str(
                module_file
            ),

        "binary_path":
            str(
                binary
            ),

        "binary_bytes":
            int(
                binary.stat().st_size
            ),

        "binary_sha256":
            expected_sha256,
    }


def load_exact_kernel_runtime():
    import importlib.metadata

    from scripts import (
        reason_router_gen4_generator_family_prevalence_kernel_compat
        as kernel_compat
    )

    installed_version = (
        importlib.metadata.version(
            "kernels"
        )
    )

    require(
        installed_version
        == KERNELS_VERSION,
        (
            "KERNELS_VERSION:"
            f"expected={KERNELS_VERSION}:"
            f"observed={installed_version}"
        ),
    )

    require(
        kernel_compat.KERNELS_VERSION
        == KERNELS_VERSION,
        "KERNEL_COMPAT_VERSION_DRIFT",
    )

    require(
        kernel_compat.BUILD_VARIANT
        == KERNEL_BUILD_VARIANT,
        "KERNEL_BUILD_VARIANT_DRIFT",
    )

    require(
        kernel_compat.MAMBA_SPEC.binary_sha256
        == MAMBA_KERNEL_BINARY_SHA256,
        "MAMBA_KERNEL_SHA_CONSTANT_DRIFT",
    )

    require(
        kernel_compat.CONV_SPEC.binary_sha256
        == CAUSAL_CONV_KERNEL_BINARY_SHA256,
        "CAUSAL_CONV_KERNEL_SHA_CONSTANT_DRIFT",
    )

    kernels = (
        kernel_compat.load_exact_fast_kernels()
    )

    require(
        kernels[
            "transport_identity_status"
        ]
        == KERNEL_TRANSPORT_IDENTITY_STATUS,
        "KERNEL_TRANSPORT_IDENTITY",
    )

    mamba_identity = (
        exact_kernel_binary_identity(
            kernels[
                "mamba"
            ],
            label="MAMBA",
            expected_sha256=(
                MAMBA_KERNEL_BINARY_SHA256
            ),
        )
    )

    conv_identity = (
        exact_kernel_binary_identity(
            kernels[
                "conv"
            ],
            label="CAUSAL_CONV",
            expected_sha256=(
                CAUSAL_CONV_KERNEL_BINARY_SHA256
            ),
        )
    )

    runtime_identity = {
        "kernel_package_name":
            "kernels",

        "kernel_package_version":
            installed_version,

        "build_variant":
            KERNEL_BUILD_VARIANT,

        "transport_identity_status":
            kernels[
                "transport_identity_status"
            ],

        "mamba_ssm": {
            "scientific_revision":
                kernel_compat.MAMBA_SPEC.scientific_revision,

            "transport_revision":
                kernels[
                    "mamba_transport_revision"
                ],

            "transport_repo_type":
                kernels[
                    "mamba_transport_repo_type"
                ],

            "transport_source":
                kernels[
                    "mamba_transport_source"
                ],

            **mamba_identity,
        },

        "causal_conv1d": {
            "scientific_revision":
                kernel_compat.CONV_SPEC.scientific_revision,

            "transport_revision":
                kernels[
                    "causal_conv_transport_revision"
                ],

            "transport_repo_type":
                kernels[
                    "causal_conv_transport_repo_type"
                ],

            "transport_source":
                kernels[
                    "causal_conv_transport_source"
                ],

            **conv_identity,
        },
    }

    return (
        kernel_compat,
        kernels,
        runtime_identity,
    )


def make_prediction_row(
    source: dict[str, str],
    encoded: dict[str, Any],
    logits: Sequence[float],
    probs: Sequence[float],
) -> dict[str, Any]:
    require(
        len(logits) == 3,
        "LOGIT_CLASS_DIMENSION_NOT_THREE",
    )

    require(
        len(probs) == 3,
        "PROB_CLASS_DIMENSION_NOT_THREE",
    )

    pred_id = max(
        range(3),
        key=lambda idx: float(
            probs[idx]
        ),
    )

    pred_label = (
        CLASS_ORDER[pred_id]
    )

    gold_label = (
        canonicalize_vitaminc_label(
            source["label"]
        )
    )

    confidence = float(
        probs[pred_id]
    )

    correct = (
        pred_label
        == gold_label
    )

    decisive = (
        pred_label
        in DECISIVE_LABELS
    )

    confident = (
        confidence
        >= CONFIDENCE_THRESHOLD
    )

    row = {
        "raw_idx":
            int(
                source["raw_idx"]
            ),

        "unique_id":
            source["unique_id"],

        "case_id":
            source["case_id"],

        "gold_label":
            gold_label,

        "pred_label":
            pred_label,

        "final_logits":
            [
                float(value)
                for value in logits
            ],

        "final_probs":
            [
                float(value)
                for value in probs
            ],

        "confidence":
            confidence,

        "correct":
            bool(correct),

        "decisive_prediction":
            bool(decisive),

        "confident":
            bool(confident),

        "confident_decisive_correct":
            bool(
                confident
                and decisive
                and correct
            ),

        "confident_decisive_wrong":
            bool(
                confident
                and decisive
                and not correct
            ),

        "input_token_length":
            int(
                encoded[
                    "input_token_length"
                ]
            ),

        "claim_token_length":
            int(
                encoded[
                    "claim_token_length"
                ]
            ),

        "evidence_token_length":
            int(
                encoded[
                    "evidence_token_length"
                ]
            ),

        "claim_truncated":
            bool(
                encoded[
                    "claim_truncated"
                ]
            ),

        "evidence_truncated":
            bool(
                encoded[
                    "evidence_truncated"
                ]
            ),
    }

    require(
        set(row)
        == SERIALIZED_ROW_FIELDS,
        "SERIALIZED_ROW_SCHEMA_MISMATCH",
    )

    lower_keys = [
        key.lower()
        for key in row
    ]

    for fragment in (
        FORBIDDEN_SERIALIZED_KEY_FRAGMENTS
    ):
        require(
            not any(
                fragment in key
                for key in lower_keys
            ),
            (
                "FORBIDDEN_SERIALIZED_KEY_FRAGMENT:"
                + fragment
            ),
        )

    return row


def build_summary(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    require(
        len(rows)
        == EXPECTED_ROWS,
        f"PREDICTION_ROW_COUNT:{len(rows)}",
    )

    raw_indices = {
        int(row["raw_idx"])
        for row in rows
    }

    require(
        len(raw_indices)
        == EXPECTED_ROWS,
        "PREDICTION_RAW_IDX_NOT_UNIQUE",
    )

    case_ids = {
        str(row["case_id"])
        for row in rows
    }

    require(
        len(case_ids)
        == EXPECTED_CASE_IDS,
        (
            "PREDICTION_CASE_ID_COUNT:"
            f"{len(case_ids)}"
        ),
    )

    prediction_distribution = Counter(
        row["pred_label"]
        for row in rows
    )

    correctness = Counter(
        (
            "CORRECT"
            if row["correct"]
            else "WRONG"
        )
        for row in rows
    )

    decisive_rows = [
        row
        for row in rows
        if row[
            "decisive_prediction"
        ]
    ]

    confident_decisive_rows = [
        row
        for row in rows
        if (
            row["confident"]
            and row[
                "decisive_prediction"
            ]
        )
    ]

    wrong_rows = [
        row
        for row in rows
        if row[
            "confident_decisive_wrong"
        ]
    ]

    correct_rows = [
        row
        for row in rows
        if row[
            "confident_decisive_correct"
        ]
    ]

    wrong_case_ids = {
        row["case_id"]
        for row in wrong_rows
    }

    correct_control_rows = [
        row
        for row in correct_rows
        if row["case_id"]
        not in wrong_case_ids
    ]

    correct_control_case_ids = {
        row["case_id"]
        for row
        in correct_control_rows
    }

    by_predicted_class = {}
    class_capacities = {}

    for label in (
        "REFUTE",
        "SUPPORT",
    ):
        class_wrong_case_ids = {
            row["case_id"]
            for row in wrong_rows
            if row["pred_label"]
            == label
        }

        class_correct_case_ids = {
            row["case_id"]
            for row
            in correct_control_rows
            if row["pred_label"]
            == label
        }

        capacity = min(
            len(
                class_wrong_case_ids
            ),
            len(
                class_correct_case_ids
            ),
        )

        class_capacities[
            label
        ] = capacity

        by_predicted_class[
            label
        ] = {
            "confident_decisive_wrong_rows":
                sum(
                    1
                    for row in wrong_rows
                    if row["pred_label"]
                    == label
                ),

            "confident_decisive_wrong_case_ids":
                len(
                    class_wrong_case_ids
                ),

            "eligible_correct_control_rows":
                sum(
                    1
                    for row
                    in correct_control_rows
                    if row["pred_label"]
                    == label
                ),

            "eligible_correct_control_case_ids":
                len(
                    class_correct_case_ids
                ),

            "case_level_pair_capacity_upper_bound":
                capacity,
        }

    per_case_counts = Counter(
        row["case_id"]
        for row in rows
    )

    multiplicity_histogram = Counter(
        per_case_counts.values()
    )

    token_lengths = [
        int(
            row[
                "input_token_length"
            ]
        )
        for row in rows
    ]

    total_case_level_pair_capacity = (
        maximum_case_level_pair_capacity(
            wrong_rows,
            correct_control_rows,
        )
    )

    return {
        "schema_version":
            "NATIVE_Q1_VITAMINC_MAMBA_PREDICTION_CENSUS_V1",

        "status":
            "PREDICTION_CENSUS_COMPLETE",

        "authority_commit":
            AUTHORITY_COMMIT,

        "dataset_row_count":
            len(rows),

        "unique_case_id_count":
            len(case_ids),

        "prediction_distribution": {
            label:
                int(
                    prediction_distribution.get(
                        label,
                        0,
                    )
                )
            for label in CLASS_ORDER
        },

        "correct_wrong_counts": {
            key:
                int(
                    correctness.get(
                        key,
                        0,
                    )
                )
            for key in (
                "CORRECT",
                "WRONG",
            )
        },

        "decisive_counts": {
            "rows":
                len(
                    decisive_rows
                ),

            "case_ids":
                len(
                    {
                        row["case_id"]
                        for row
                        in decisive_rows
                    }
                ),
        },

        "confident_decisive_counts": {
            "rows":
                len(
                    confident_decisive_rows
                ),

            "case_ids":
                len(
                    {
                        row["case_id"]
                        for row
                        in confident_decisive_rows
                    }
                ),
        },

        "confident_decisive_correct_rows":
            len(
                correct_rows
            ),

        "confident_decisive_wrong_rows":
            len(
                wrong_rows
            ),

        "confident_decisive_correct_case_ids":
            len(
                {
                    row["case_id"]
                    for row
                    in correct_rows
                }
            ),

        "confident_decisive_wrong_case_ids":
            len(
                wrong_case_ids
            ),

        "counts_by_predicted_class":
            by_predicted_class,

        "row_multiplicity_per_case": {
            str(key):
                int(value)
            for key, value
            in sorted(
                multiplicity_histogram.items()
            )
        },

        "wrong_case_id_set":
            sorted(
                wrong_case_ids
            ),

        "correct_control_case_id_set_excluding_all_wrong_cases":
            sorted(
                correct_control_case_ids
            ),

        "predicted_class_case_level_pair_capacity":
            class_capacities,

        "total_case_level_pair_capacity":
            total_case_level_pair_capacity,

        "total_case_level_pair_capacity_upper_bound_without_predicted_class_constraint":
            min(
                len(
                    wrong_case_ids
                ),
                len(
                    correct_control_case_ids
                ),
            ),

        "pair_capacity_respects_predicted_class":
            True,

        "pair_capacity_respects_case_id_uniqueness":
            True,

        "pair_capacity_is_final_matching":
            False,

        "input_length_distribution": {
            "min":
                min(
                    token_lengths
                ),

            "max":
                max(
                    token_lengths
                ),

            "mean":
                (
                    sum(
                        token_lengths
                    )
                    / len(
                        token_lengths
                    )
                ),
        },

        "truncation_counts": {
            "claim":
                sum(
                    bool(
                        row[
                            "claim_truncated"
                        ]
                    )
                    for row in rows
                ),

            "evidence":
                sum(
                    bool(
                        row[
                            "evidence_truncated"
                        ]
                    )
                    for row in rows
                ),

            "either":
                sum(
                    bool(
                        row[
                            "claim_truncated"
                        ]
                        or row[
                            "evidence_truncated"
                        ]
                    )
                    for row in rows
                ),
        },

        "confidence_statistic":
            "PREDICTED_CLASS_FINAL_PROBABILITY",

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "independent_sampling_unit":
            "case_id",

        "auxiliary_gold_labels_forwarded":
            False,

        "native_state_accessed":
            False,

        "hidden_state_exported":
            False,

        "training_executed":
            False,

        "optimizer_created":
            False,

        "gradient_computation_enabled":
            False,

        "final_matching_algorithm_frozen":
            False,
    }


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: Iterable[
        dict[str, Any]
    ],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch
    import transformers

    require(
        git_head(args.repo)
        == args.expected_head,
        "HEAD_MISMATCH",
    )

    require(
        args.allow_post_authority_implementation_head,
        (
            "POST_AUTHORITY_IMPLEMENTATION_HEAD"
            "_NOT_EXPLICITLY_ENABLED"
        ),
    )

    require(
        git_is_clean(args.repo),
        "WORKTREE_NOT_CLEAN",
    )

    require(
        not args.output_root.exists(),
        (
            "OUTPUT_COLLISION:"
            f"{args.output_root}"
        ),
    )

    require(
        str(args.device).startswith(
            "cuda"
        ),
        "CENSUS_EXECUTION_REQUIRES_CUDA",
    )

    require(
        torch.cuda.is_available(),
        "CUDA_NOT_AVAILABLE",
    )

    source_rows = read_dataset(
        args.dataset_csv
    )

    tokenizer = load_tokenizer(
        args.tokenizer_snapshot
    )

    state, checkpoint_metadata = (
        load_checkpoint_payload(
            args.checkpoint
        )
    )

    (
        kernel_compat,
        kernels,
        kernel_runtime,
    ) = load_exact_kernel_runtime()

    with (
        kernel_compat
        .exact_transformers_kernel_loader(
            kernels
        )
    ) as kernel_constructor_calls:
        model = (
            build_model_from_local_config(
                args.tokenizer_snapshot,
                args.device,
            )
        )

    kernel_runtime[
        "constructor_counts"
    ] = validate_kernel_constructor_calls(
        kernel_constructor_calls,
        len(
            model.mamba.layers
        ),
    )

    kernel_compat.validate_transformers_kernel_bindings(
        kernels
    )

    kernel_runtime[
        "transformers_kernel_bindings_validated"
    ] = True

    incompatible = (
        model.load_state_dict(
            state,
            strict=True,
        )
    )

    require(
        not incompatible.missing_keys,
        (
            "STRICT_LOAD_MISSING_KEYS:"
            + ",".join(
                incompatible.missing_keys
            )
        ),
    )

    require(
        not incompatible.unexpected_keys,
        (
            "STRICT_LOAD_UNEXPECTED_KEYS:"
            + ",".join(
                incompatible.unexpected_keys
            )
        ),
    )

    model.to(
        args.device
    )

    model.eval()

    temporal_list, predicate_list = (
        external_flags(
            len(
                source_rows
            )
        )
    )

    require(
        not any(
            temporal_list
        ),
        "NONZERO_TEMPORAL_EXTERNAL_FLAG",
    )

    require(
        not any(
            predicate_list
        ),
        "NONZERO_PREDICATE_EXTERNAL_FLAG",
    )

    prediction_rows = []

    with torch.no_grad():
        for start in range(
            0,
            len(source_rows),
            args.batch_size,
        ):
            batch_source = (
                source_rows[
                    start:
                    start
                    + args.batch_size
                ]
            )

            encoded = [
                encode_one(
                    tokenizer,
                    row,
                )
                for row
                in batch_source
            ]

            input_ids = torch.tensor(
                [
                    item[
                        "input_ids"
                    ]
                    for item
                    in encoded
                ],
                dtype=torch.long,
                device=args.device,
            )

            attention_mask = torch.tensor(
                [
                    item[
                        "attention_mask"
                    ]
                    for item
                    in encoded
                ],
                dtype=torch.bool,
                device=args.device,
            )

            claim_mask = torch.tensor(
                [
                    item[
                        "claim_mask"
                    ]
                    for item
                    in encoded
                ],
                dtype=torch.bool,
                device=args.device,
            )

            evidence_mask = torch.tensor(
                [
                    item[
                        "evidence_mask"
                    ]
                    for item
                    in encoded
                ],
                dtype=torch.bool,
                device=args.device,
            )

            batch_size = len(
                batch_source
            )

            temporal_flags = (
                torch.zeros(
                    batch_size,
                    dtype=torch.long,
                    device=args.device,
                )
            )

            predicate_flags = (
                torch.zeros(
                    batch_size,
                    dtype=torch.long,
                    device=args.device,
                )
            )

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                claim_mask=claim_mask,
                evidence_mask=evidence_mask,
                temporal_mismatch_flags=temporal_flags,
                predicate_mismatch_flags=predicate_flags,
            )

            logits = output.get(
                "logits"
            )

            predictions = output.get(
                "predictions"
            )

            require(
                logits is not None,
                "MODEL_FINAL_LOGITS_MISSING",
            )

            require(
                predictions is not None,
                "MODEL_FINAL_PREDICTIONS_MISSING",
            )

            require(
                tuple(
                    logits.shape
                )
                == (
                    batch_size,
                    3,
                ),
                (
                    "LOGIT_SHAPE:"
                    f"{tuple(logits.shape)}"
                ),
            )

            require(
                torch.equal(
                    predictions,
                    logits.argmax(
                        dim=-1
                    ),
                ),
                (
                    "PREDICTION_NOT_ARGMAX_"
                    "FINAL_LOGITS"
                ),
            )

            probs = torch.softmax(
                logits.float(),
                dim=-1,
            )

            logits_cpu = (
                logits
                .detach()
                .float()
                .cpu()
                .tolist()
            )

            probs_cpu = (
                probs
                .detach()
                .cpu()
                .tolist()
            )

            for (
                source,
                encoding,
                row_logits,
                row_probs,
            ) in zip(
                batch_source,
                encoded,
                logits_cpu,
                probs_cpu,
            ):
                prediction_rows.append(
                    make_prediction_row(
                        source,
                        encoding,
                        row_logits,
                        row_probs,
                    )
                )

    summary = build_summary(
        prediction_rows
    )

    summary.update(
        {
            "execution_head":
                args.expected_head,

            "dataset_sha256":
                DATASET_SHA256,

            "checkpoint_sha256":
                CHECKPOINT_SHA256,

            "tokenizer_revision":
                TOKENIZER_REVISION,

            "tokenizer_json_sha256":
                TOKENIZER_JSON_SHA256,

            "checkpoint_metadata_validation":
                validate_checkpoint_metadata(
                    checkpoint_metadata
                ),

            "device":
                str(
                    args.device
                ),

            "batch_size":
                int(
                    args.batch_size
                ),

            "model_forward_rows":
                len(
                    prediction_rows
                ),
        }
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    rows_path = (
        args.output_root
        / "prediction_rows.jsonl"
    )

    summary_path = (
        args.output_root
        / "prediction_census_summary.json"
    )

    provenance_path = (
        args.output_root
        / "run_provenance.json"
    )

    write_jsonl(
        rows_path,
        prediction_rows,
    )

    write_json(
        summary_path,
        summary,
    )

    provenance = {
        "schema_version":
            "NATIVE_Q1_VITAMINC_MAMBA_PREDICTION_CENSUS_PROVENANCE_V2",

        "authority_commit":
            AUTHORITY_COMMIT,

        "execution_head":
            args.expected_head,

        "dataset_path":
            str(
                args.dataset_csv
            ),

        "dataset_sha256":
            DATASET_SHA256,

        "checkpoint_path":
            str(
                args.checkpoint
            ),

        "checkpoint_sha256":
            CHECKPOINT_SHA256,

        "tokenizer_snapshot":
            str(
                args.tokenizer_snapshot
            ),

        "tokenizer_revision":
            TOKENIZER_REVISION,

        "tokenizer_json_sha256":
            TOKENIZER_JSON_SHA256,

        "python_version":
            sys.version,

        "torch_version":
            torch.__version__,

        "transformers_version":
            transformers.__version__,

        "torch_cuda_version":
            torch.version.cuda,

        "cuda_device_name":
            torch.cuda.get_device_name(
                torch.cuda.current_device()
            ),

        "cuda_device_capability":
            list(
                torch.cuda.get_device_capability(
                    torch.cuda.current_device()
                )
            ),

        "kernel_runtime":
            kernel_runtime,

        "device":
            str(
                args.device
            ),

        "cuda_available":
            bool(
                torch.cuda.is_available()
            ),

        "prediction_rows_sha256":
            sha256_file(
                rows_path
            ),

        "prediction_census_summary_sha256":
            sha256_file(
                summary_path
            ),

        "auxiliary_gold_labels_forwarded":
            False,

        "native_state_accessed":
            False,

        "hidden_state_exported":
            False,

        "training_executed":
            False,

        "optimizer_created":
            False,

        "gradient_computation_enabled":
            False,
    }

    write_json(
        provenance_path,
        provenance,
    )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Native Q1 VitaminC "
            "frozen-Mamba prediction-only "
            "cohort census"
        )
    )

    parser.add_argument(
        "--static-contract-verify",
        action="store_true",
    )

    parser.add_argument(
        "--repo",
        type=Path,
    )

    parser.add_argument(
        "--expected-head",
    )

    parser.add_argument(
        "--allow-post-authority-implementation-head",
        action="store_true",
    )

    parser.add_argument(
        "--dataset-csv",
        type=Path,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
    )

    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    return parser


def static_contract() -> dict[str, Any]:
    return {
        "authority_commit":
            AUTHORITY_COMMIT,

        "dataset_sha256":
            DATASET_SHA256,

        "checkpoint_sha256":
            CHECKPOINT_SHA256,

        "tokenizer_revision":
            TOKENIZER_REVISION,

        "tokenizer_json_sha256":
            TOKENIZER_JSON_SHA256,

        "model_name":
            MODEL_NAME,

        "architecture":
            ARCHITECTURE,

        "training_seed":
            TRAINING_SEED,

        "split_seed":
            SPLIT_SEED,

        "max_length":
            MAX_LENGTH,

        "claim_budget":
            CLAIM_BUDGET,

        "evidence_budget":
            EVIDENCE_BUDGET,

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "external_temporal_flags":
            "ALL_ZERO",

        "external_predicate_flags":
            "ALL_ZERO",

        "kernel_runtime_contract": {
            "kernels_version":
                KERNELS_VERSION,

            "build_variant":
                KERNEL_BUILD_VARIANT,

            "transport_identity_status":
                KERNEL_TRANSPORT_IDENTITY_STATUS,

            "mamba_binary_sha256":
                MAMBA_KERNEL_BINARY_SHA256,

            "causal_conv_binary_sha256":
                CAUSAL_CONV_KERNEL_BINARY_SHA256,
        },

        "auxiliary_gold_labels_forwarded":
            False,

        "native_state_access_allowed":
            False,

        "training_allowed":
            False,
    }


def main(
    argv: list[str] | None = None,
) -> int:
    args = (
        build_parser()
        .parse_args(
            argv
        )
    )

    if args.static_contract_verify:
        print(
            "NATIVE_Q1_MAMBA_"
            "PREDICTION_CENSUS_"
            "STATIC_CONTRACT_PASS"
        )

        print(
            json.dumps(
                static_contract(),
                indent=2,
                sort_keys=True,
            )
        )

        return 0

    required = (
        "repo",
        "expected_head",
        "dataset_csv",
        "checkpoint",
        "tokenizer_snapshot",
        "output_root",
    )

    missing = [
        name
        for name in required
        if getattr(
            args,
            name,
        ) is None
    ]

    if missing:
        print(
            "BLOCKED:"
            "MISSING_ARGUMENTS:"
            + ",".join(
                missing
            ),
            file=sys.stderr,
        )

        return 2

    try:
        require(
            args.batch_size > 0,
            "BATCH_SIZE_MUST_BE_POSITIVE",
        )

        summary = run(
            args
        )

    except CensusError as exc:
        print(
            f"BLOCKED:{exc}",
            file=sys.stderr,
        )

        return 2

    print(
        "NATIVE_Q1_VITAMINC_"
        "MAMBA_PREDICTION_CENSUS_PASS"
    )

    print(
        json.dumps(
            {
                "dataset_row_count":
                    summary[
                        "dataset_row_count"
                    ],

                "unique_case_id_count":
                    summary[
                        "unique_case_id_count"
                    ],

                "confident_decisive_wrong_rows":
                    summary[
                        "confident_decisive_wrong_rows"
                    ],

                "confident_decisive_wrong_case_ids":
                    summary[
                        "confident_decisive_wrong_case_ids"
                    ],

                "total_case_level_pair_capacity":
                    summary[
                        "total_case_level_pair_capacity"
                    ],
            },
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )