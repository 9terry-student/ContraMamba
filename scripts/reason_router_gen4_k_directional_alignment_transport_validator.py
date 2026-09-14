#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-directional-alignment-transport"

RUNNER_FREEZE = (
    "2bfcd7b4243832f38e32abf389229d7601b180b2"
)
CORE_FREEZE = (
    "3ced19dfcf011ae7b300242723eb1f10b3a3437f"
)
IMPLEMENTATION_PARENT = (
    "ac4f682acaeea5a00eafdedcd5ff097b61d3a3c0"
)
GEN4_PARENT = (
    "a738b4c169d30b0a7e21563c3aba2c29c481c456"
)
K_CAUSAL_PARENT = (
    "5f5f4d6a80085ad8baf43445475d1c3049535c22"
)

RUNNER_REL = (
    "scripts/"
    "reason_router_gen4_k_directional_alignment_transport_runner.py"
)
CORE_REL = (
    "scripts/"
    "reason_router_gen4_k_directional_alignment_transport_core.py"
)
RUNTIME_REL = (
    "scripts/"
    "reason_router_gen4_k_directional_alignment_transport_runtime.py"
)

RUNNER_BLOB = "3677dd83950789e41417c3a1ffaf70b82d7003ad"
CORE_BLOB = "d98b2dcd3436433c04bb56ecc57dec4240abe820"
RUNTIME_BLOB = "989c4a8947560dcf35e9523373d09ba085a9431a"

CHECKPOINT_SHA256 = (
    "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
)

FROZEN_ENDPOINT_REL = (
    "reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_"
    "scientific_extraction_v1/kinematic_endpoints.jsonl"
)
FROZEN_ENDPOINT_SHA256 = (
    "b47e32496f73f5493e8737b040b30bc4a23dd8cd22454a85244fa812273b6605"
)
EVENT_MANIFEST_REL = (
    "reports/reason_router_gen4_six_cell_native_mamba_state_bridge_"
    "feasibility_audit_a2617aa/"
    "event_anchor_prefix_manifest_candidate.jsonl"
)
EVENT_MANIFEST_SHA256 = (
    "70c84c68b36751bb7c7145b33ccb71ab91bc8ee9e6cc5f2c7a0d4e925f36581f"
)
MAMBA_SOURCE_SHA256 = (
    "4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83"
)

STRONG_INDEX_SHA256 = (
    "6950bb6c6cc777375f5e4ce18f22fd3272d7b80c5aff0726c25a6ec77d5813ce"
)

FROZEN_BASELINE_MEAN = -0.013998957453394283

SOURCE_PAIR_COUNT = 300
PREFLIGHT_PAIR_COUNT = 2
FORWARDS_PER_PAIR = 8
FULL_FORWARD_BUDGET = 2400
PREFLIGHT_FORWARD_BUDGET = 16

SOURCE_BLOCK = 15
TARGET_RESIDUAL_LAYER = 16
INTERVENTION_LAYER = 17
RELATIVE_COORDINATE = 2

TARGET_PLUS = "C2_NAME"
TARGET_MINUS = "C0_SHAM"
REFERENCE_PLUS = "C5_TITLE_NAME"
REFERENCE_MINUS = "C1_TITLE"
ANCHOR_NAME = "A_IDENTITY"

STRONG_COUNT = 395
WEAK_COUNT = 1141
EQUAL_COUNT = 0

BASELINE_REPRO_TOL = 1e-12
VECTOR_TOL = 5e-12
MIDPOINT_TOL = 5e-6
CAST_TOL = 5e-6
RECOMPUTE_TOL = 1e-12
FAMILY_ALPHA = 0.05

ITEM_SCHEMA = "gen4-k-directional-alignment-transport-item-v1"
SUMMARY_SCHEMA = "gen4-k-directional-alignment-transport-summary-v1"
MANIFEST_SCHEMA = "gen4-k-directional-alignment-transport-manifest-v1"
PREFLIGHT_SCHEMA = "gen4-k-directional-alignment-transport-preflight-v1"

ITEM_FILE = "item_metrics.jsonl"
SUMMARY_FILE = "summary.json"
MANIFEST_FILE = "manifest.json"
PREFLIGHT_FILE = "preflight.json"
CHECKSUM_FILE = "SHA256SUMS.txt"


class ValidationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValidationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_text(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_bytes(revision: str, path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{revision}:{path}"],
            cwd=ROOT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError(
            f"GIT_SHOW_FAILURE:{revision}:{path}"
        ) from exc


def git_blob(revision: str, path: str) -> str:
    return git_text(
        "rev-parse",
        f"{revision}:{path}",
    )


def git_is_ancestor(
    ancestor: str,
    descendant: str,
) -> bool:
    return (
        subprocess.call(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def paths_unchanged(
    base: str,
    head: str,
    paths: Sequence[str],
) -> bool:
    return (
        subprocess.call(
            [
                "git",
                "diff",
                "--quiet",
                base,
                head,
                "--",
                *paths,
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def json_object(
    path: Path,
    label: str,
) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValidationError(
            f"{label}_JSON_FAILURE"
        ) from exc

    require(
        isinstance(value, dict),
        f"{label}_NOT_OBJECT",
    )
    return value


def jsonl_objects(
    path: Path,
    label: str,
) -> list[dict[str, Any]]:
    rows = []

    for line_no, raw in enumerate(
        path.read_bytes().splitlines(),
        start=1,
    ):
        require(
            bool(raw),
            f"{label}_BLANK_LINE:{line_no}",
        )
        try:
            value = json.loads(raw)
        except Exception as exc:
            raise ValidationError(
                f"{label}_JSON_FAILURE:{line_no}"
            ) from exc

        require(
            isinstance(value, dict),
            f"{label}_ROW_NOT_OBJECT:{line_no}",
        )
        rows.append(value)

    return rows


def finite_number(
    value: Any,
    label: str,
) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool),
        f"{label}_NOT_NUMBER",
    )

    out = float(value)

    require(
        math.isfinite(out),
        f"{label}_NONFINITE",
    )
    return out


def close(
    observed: Any,
    expected: Any,
    label: str,
    tol: float = RECOMPUTE_TOL,
) -> None:
    a = finite_number(observed, label + ":OBS")
    b = finite_number(expected, label + ":EXP")

    require(
        abs(a - b) <= tol,
        (
            f"{label}_MISMATCH:"
            f"{a}:{b}:{abs(a-b)}"
        ),
    )


def reject_private_payload(
    value: Any,
    path: str = "root",
) -> None:
    forbidden = (
        "raw_vector",
        "activation_vector",
        "checkpoint_bytes",
        "token_text",
        "final_logits",
    )

    if isinstance(value, Mapping):
        for key, item in value.items():
            field = path + "." + str(key)
            lower = str(key).lower()

            if lower == "logits_read":
                require(
                    item is False,
                    "LOGITS_READ_NOT_FALSE:" + field,
                )
            else:
                require(
                    not any(
                        token in lower
                        for token in forbidden
                    ),
                    "FORBIDDEN_PUBLIC_FIELD:" + field,
                )

            reject_private_payload(
                item,
                field,
            )

    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_private_payload(
                item,
                f"{path}[{index}]",
            )


def validate_checksums(
    directory: Path,
    data_files: Sequence[str],
) -> None:
    observed = (
        directory / CHECKSUM_FILE
    ).read_bytes()

    expected = "".join(
        (
            f"{sha256_file(directory / name)}"
            f"  {name}\n"
        )
        for name in sorted(data_files)
    ).encode("utf-8")

    require(
        observed == expected,
        "CHECKSUM_FILE_MISMATCH",
    )


def validate_frozen_code_identity(
    execution_head: str,
) -> dict[str, str]:
    require(
        git_is_ancestor(
            RUNNER_FREEZE,
            execution_head,
        ),
        "RUNNER_FREEZE_NOT_ANCESTOR",
    )
    require(
        git_is_ancestor(
            CORE_FREEZE,
            execution_head,
        ),
        "CORE_FREEZE_NOT_ANCESTOR",
    )

    frozen = {
        RUNNER_REL: (
            RUNNER_FREEZE,
            RUNNER_BLOB,
        ),
        CORE_REL: (
            CORE_FREEZE,
            CORE_BLOB,
        ),
        RUNTIME_REL: (
            RUNNER_FREEZE,
            RUNTIME_BLOB,
        ),
    }

    for path, (
        freeze,
        expected_blob,
    ) in frozen.items():
        require(
            git_blob(
                freeze,
                path,
            )
            == expected_blob,
            f"FROZEN_BLOB_MISMATCH:{path}",
        )
        require(
            paths_unchanged(
                freeze,
                execution_head,
                (path,),
            ),
            f"FROZEN_CODE_CHANGED_AFTER_FREEZE:{path}",
        )

    return {
        path:
            sha256_bytes(
                git_bytes(
                    freeze,
                    path,
                )
            )
        for path, (
            freeze,
            _,
        ) in frozen.items()
    }


MANIFEST_FIELDS = frozenset({
    "schema_version",
    "mode",
    "runtime_branch",
    "runtime_git_head",
    "implementation_parent",
    "gen4_parent",
    "k_causal_parent",
    "runner_rel",
    "runner_sha256",
    "core_rel",
    "core_sha256",
    "runtime_rel",
    "runtime_sha256",
    "checkpoint_sha256",
    "checkpoint_source_mode",
    "frozen_endpoint_rel",
    "frozen_endpoint_sha256",
    "event_manifest_rel",
    "event_manifest_sha256",
    "mamba_source_sha256",
    "source_pair_count",
    "model_forward_count",
    "forwards_per_pair",
    "source_block",
    "target_residual_layer",
    "intervention_layer",
    "relative_coordinate",
    "target_pair",
    "reference_pair",
    "anchor_name",
    "strong_count",
    "weak_count",
    "equal_count",
    "strong_index_sha256",
    "training_executed",
    "backward_executed",
    "task_heads_executed",
    "logits_read",
    "raw_vectors_persisted",
    "tokenizer_invoked",
    "causal_intervention_executed",
    "statistical_testing",
})


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_execution_head: str,
    mode: str,
    code_sha: Mapping[str, str],
) -> None:
    require(
        set(manifest) == MANIFEST_FIELDS,
        "MANIFEST_FIELDS",
    )
    require(
        manifest["schema_version"]
        == MANIFEST_SCHEMA,
        "MANIFEST_SCHEMA",
    )
    require(
        manifest["mode"] == mode,
        "MANIFEST_MODE",
    )
    require(
        manifest["runtime_branch"]
        == EXPECTED_BRANCH,
        "MANIFEST_BRANCH",
    )
    require(
        manifest["runtime_git_head"]
        == expected_execution_head,
        "MANIFEST_HEAD",
    )
    require(
        manifest["implementation_parent"]
        == IMPLEMENTATION_PARENT,
        "MANIFEST_IMPLEMENTATION_PARENT",
    )
    require(
        manifest["gen4_parent"]
        == GEN4_PARENT,
        "MANIFEST_GEN4_PARENT",
    )
    require(
        manifest["k_causal_parent"]
        == K_CAUSAL_PARENT,
        "MANIFEST_K_PARENT",
    )

    require(
        manifest["runner_rel"]
        == RUNNER_REL,
        "MANIFEST_RUNNER_REL",
    )
    require(
        manifest["core_rel"]
        == CORE_REL,
        "MANIFEST_CORE_REL",
    )
    require(
        manifest["runtime_rel"]
        == RUNTIME_REL,
        "MANIFEST_RUNTIME_REL",
    )

    require(
        manifest["runner_sha256"]
        == code_sha[RUNNER_REL],
        "MANIFEST_RUNNER_SHA",
    )
    require(
        manifest["core_sha256"]
        == code_sha[CORE_REL],
        "MANIFEST_CORE_SHA",
    )
    require(
        manifest["runtime_sha256"]
        == code_sha[RUNTIME_REL],
        "MANIFEST_RUNTIME_SHA",
    )

    require(
        manifest["checkpoint_sha256"]
        == CHECKPOINT_SHA256,
        "MANIFEST_CHECKPOINT_SHA",
    )
    require(
        manifest["checkpoint_source_mode"]
        == "external_exact_sha256",
        "MANIFEST_CHECKPOINT_MODE",
    )
    require(
        manifest["frozen_endpoint_rel"]
        == FROZEN_ENDPOINT_REL,
        "MANIFEST_ENDPOINT_REL",
    )
    require(
        manifest["frozen_endpoint_sha256"]
        == FROZEN_ENDPOINT_SHA256,
        "MANIFEST_ENDPOINT_SHA",
    )
    require(
        manifest["event_manifest_rel"]
        == EVENT_MANIFEST_REL,
        "MANIFEST_EVENT_REL",
    )
    require(
        manifest["event_manifest_sha256"]
        == EVENT_MANIFEST_SHA256,
        "MANIFEST_EVENT_SHA",
    )
    require(
        manifest["mamba_source_sha256"]
        == MAMBA_SOURCE_SHA256,
        "MANIFEST_MAMBA_SHA",
    )

    expected_pairs = (
        PREFLIGHT_PAIR_COUNT
        if mode == "preflight"
        else SOURCE_PAIR_COUNT
    )
    expected_forwards = (
        PREFLIGHT_FORWARD_BUDGET
        if mode == "preflight"
        else FULL_FORWARD_BUDGET
    )

    require(
        manifest["source_pair_count"]
        == expected_pairs,
        "MANIFEST_PAIR_COUNT",
    )
    require(
        manifest["model_forward_count"]
        == expected_forwards,
        "MANIFEST_FORWARD_COUNT",
    )
    require(
        manifest["forwards_per_pair"]
        == FORWARDS_PER_PAIR,
        "MANIFEST_FORWARDS_PER_PAIR",
    )

    require(
        manifest["source_block"]
        == SOURCE_BLOCK,
        "MANIFEST_SOURCE_BLOCK",
    )
    require(
        manifest["target_residual_layer"]
        == TARGET_RESIDUAL_LAYER,
        "MANIFEST_TARGET_LAYER",
    )
    require(
        manifest["intervention_layer"]
        == INTERVENTION_LAYER,
        "MANIFEST_INTERVENTION_LAYER",
    )
    require(
        manifest["relative_coordinate"]
        == RELATIVE_COORDINATE,
        "MANIFEST_RELATIVE_COORDINATE",
    )

    require(
        manifest["target_pair"]
        == [TARGET_PLUS, TARGET_MINUS],
        "MANIFEST_TARGET_PAIR",
    )
    require(
        manifest["reference_pair"]
        == [REFERENCE_PLUS, REFERENCE_MINUS],
        "MANIFEST_REFERENCE_PAIR",
    )
    require(
        manifest["anchor_name"]
        == ANCHOR_NAME,
        "MANIFEST_ANCHOR",
    )

    require(
        manifest["strong_count"]
        == STRONG_COUNT,
        "MANIFEST_STRONG_COUNT",
    )
    require(
        manifest["weak_count"]
        == WEAK_COUNT,
        "MANIFEST_WEAK_COUNT",
    )
    require(
        manifest["equal_count"]
        == EQUAL_COUNT,
        "MANIFEST_EQUAL_COUNT",
    )
    require(
        manifest["strong_index_sha256"]
        == STRONG_INDEX_SHA256,
        "MANIFEST_STRONG_SHA",
    )

    for flag in (
        "training_executed",
        "backward_executed",
        "task_heads_executed",
        "logits_read",
        "raw_vectors_persisted",
    ):
        require(
            manifest[flag] is False,
            f"MANIFEST_NEGATIVE_FLAG:{flag}",
        )

    require(
        manifest["tokenizer_invoked"]
        is True,
        "MANIFEST_TOKENIZER_FLAG",
    )
    require(
        manifest["causal_intervention_executed"]
        is True,
        "MANIFEST_CAUSAL_FLAG",
    )
    require(
        manifest["statistical_testing"]
        is (mode == "full"),
        "MANIFEST_STATISTICAL_FLAG",
    )


PREFLIGHT_FIELDS = frozenset({
    "schema_version",
    "pair_count",
    "model_forward_count",
    "max_baseline_reproduction_abs_residual",
    "max_alignment_cosine_abs_residual",
    "max_magnitude_cosine_abs_residual",
    "max_alignment_A_preservation_abs_residual",
    "max_alignment_B_preservation_abs_residual",
    "max_magnitude_A_target_abs_residual",
    "max_magnitude_B_target_abs_residual",
    "max_alignment_midpoint_abs_residual",
    "max_magnitude_midpoint_abs_residual",
    "max_alignment_pair_delta_abs_residual",
    "max_magnitude_pair_delta_abs_residual",
    "max_alignment_applied_correction_abs_residual",
    "max_magnitude_applied_correction_abs_residual",
    "scientific_endpoint_values_serialized",
    "inferential_statistics_executed",
    "result",
})


def validate_preflight_public(
    value: Mapping[str, Any],
) -> None:
    require(
        set(value) == PREFLIGHT_FIELDS,
        "PREFLIGHT_FIELDS",
    )
    require(
        value["schema_version"]
        == PREFLIGHT_SCHEMA,
        "PREFLIGHT_SCHEMA",
    )
    require(
        value["pair_count"]
        == PREFLIGHT_PAIR_COUNT,
        "PREFLIGHT_PAIR_COUNT",
    )
    require(
        value["model_forward_count"]
        == PREFLIGHT_FORWARD_BUDGET,
        "PREFLIGHT_FORWARD_COUNT",
    )

    tol_fields = {
        "max_baseline_reproduction_abs_residual":
            BASELINE_REPRO_TOL,
        "max_alignment_cosine_abs_residual":
            VECTOR_TOL,
        "max_magnitude_cosine_abs_residual":
            VECTOR_TOL,
        "max_alignment_A_preservation_abs_residual":
            VECTOR_TOL,
        "max_alignment_B_preservation_abs_residual":
            VECTOR_TOL,
        "max_magnitude_A_target_abs_residual":
            VECTOR_TOL,
        "max_magnitude_B_target_abs_residual":
            VECTOR_TOL,
        "max_alignment_midpoint_abs_residual":
            MIDPOINT_TOL,
        "max_magnitude_midpoint_abs_residual":
            MIDPOINT_TOL,
        "max_alignment_pair_delta_abs_residual":
            CAST_TOL,
        "max_magnitude_pair_delta_abs_residual":
            CAST_TOL,
        "max_alignment_applied_correction_abs_residual":
            CAST_TOL,
        "max_magnitude_applied_correction_abs_residual":
            CAST_TOL,
    }

    for field, tolerance in tol_fields.items():
        observed = finite_number(
            value[field],
            field,
        )
        require(
            0.0 <= observed <= tolerance,
            f"PREFLIGHT_TOLERANCE:{field}:{observed}",
        )

    require(
        value["scientific_endpoint_values_serialized"]
        is False,
        "PREFLIGHT_SCIENTIFIC_VALUES",
    )
    require(
        value["inferential_statistics_executed"]
        is False,
        "PREFLIGHT_INFERENCE",
    )
    require(
        value["result"]
        == "PASS_BOUNDED_PREFLIGHT",
        "PREFLIGHT_RESULT",
    )


ITEM_FIELDS = frozenset({
    "schema_version",
    "source_pair_id",
    "source_block",
    "target_residual_layer",
    "intervention_layer",
    "relative_coordinate",
    "target_plus_cell",
    "target_minus_cell",
    "reference_plus_cell",
    "reference_minus_cell",
    "anchor_name",
    "target_plus_anchor",
    "target_minus_anchor",
    "reference_plus_anchor",
    "reference_minus_anchor",
    "target_plus_intervention_token",
    "target_minus_intervention_token",
    "reference_plus_geometry_token",
    "reference_minus_geometry_token",
    "target_A",
    "target_B",
    "target_C",
    "reference_A",
    "reference_B",
    "reference_C",
    "alignment_realized_A",
    "alignment_realized_B",
    "alignment_target_cosine",
    "alignment_realized_cosine",
    "alignment_cosine_abs_residual",
    "alignment_A_preservation_abs_residual",
    "alignment_B_preservation_abs_residual",
    "alignment_midpoint_max_abs_residual",
    "alignment_pair_delta_max_abs_residual",
    "alignment_applied_correction_max_abs_residual",
    "magnitude_target_A",
    "magnitude_target_B",
    "magnitude_realized_A",
    "magnitude_realized_B",
    "magnitude_baseline_cosine",
    "magnitude_realized_cosine",
    "magnitude_cosine_abs_residual",
    "magnitude_A_target_abs_residual",
    "magnitude_B_target_abs_residual",
    "magnitude_midpoint_max_abs_residual",
    "magnitude_pair_delta_max_abs_residual",
    "magnitude_applied_correction_max_abs_residual",
    "baseline_plus_path_efficiency",
    "baseline_minus_path_efficiency",
    "frozen_plus_path_efficiency",
    "frozen_minus_path_efficiency",
    "baseline_plus_reproduction_abs_residual",
    "baseline_minus_reproduction_abs_residual",
    "alignment_plus_path_efficiency",
    "alignment_minus_path_efficiency",
    "magnitude_plus_path_efficiency",
    "magnitude_minus_path_efficiency",
    "delta_baseline",
    "delta_alignment",
    "delta_magnitude",
    "R_ALIGN",
    "R_MAG",
    "ALIGNMENT_SPECIFICITY",
})


def load_frozen_endpoint_map(
) -> dict[tuple[str, str], float]:
    path = ROOT / FROZEN_ENDPOINT_REL

    require(
        path.is_file(),
        "FROZEN_ENDPOINT_MISSING",
    )
    require(
        sha256_file(path)
        == FROZEN_ENDPOINT_SHA256,
        "FROZEN_ENDPOINT_HASH",
    )

    result = {}

    for row in jsonl_objects(
        path,
        "FROZEN_ENDPOINT",
    ):
        if int(row["layer_index"]) != 17:
            continue
        if str(row["contrast_cell_id"]) not in {
            TARGET_PLUS,
            TARGET_MINUS,
        }:
            continue

        require(
            str(row["semantic_anchor"])
            == "A_NAME",
            "FROZEN_ENDPOINT_ANCHOR",
        )

        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(
            key not in result,
            f"FROZEN_ENDPOINT_DUPLICATE:{key}",
        )
        result[key] = finite_number(
            row["POST4_PATH_EFFICIENCY"],
            "FROZEN_PE",
        )

    require(
        len(result)
        == 2 * SOURCE_PAIR_COUNT,
        "FROZEN_ENDPOINT_COUNT",
    )

    return result


def validate_item(
    row: Mapping[str, Any],
    frozen: Mapping[
        tuple[str, str],
        float,
    ],
) -> None:
    require(
        set(row) == ITEM_FIELDS,
        "ITEM_FIELDS",
    )
    require(
        row["schema_version"]
        == ITEM_SCHEMA,
        "ITEM_SCHEMA",
    )

    pair = str(row["source_pair_id"])
    require(bool(pair), "ITEM_PAIR_ID")

    require(
        row["source_block"]
        == SOURCE_BLOCK,
        "ITEM_SOURCE_BLOCK",
    )
    require(
        row["target_residual_layer"]
        == TARGET_RESIDUAL_LAYER,
        "ITEM_TARGET_LAYER",
    )
    require(
        row["intervention_layer"]
        == INTERVENTION_LAYER,
        "ITEM_INTERVENTION_LAYER",
    )
    require(
        row["relative_coordinate"]
        == RELATIVE_COORDINATE,
        "ITEM_RELATIVE_COORDINATE",
    )

    require(
        row["target_plus_cell"]
        == TARGET_PLUS,
        "ITEM_TARGET_PLUS",
    )
    require(
        row["target_minus_cell"]
        == TARGET_MINUS,
        "ITEM_TARGET_MINUS",
    )
    require(
        row["reference_plus_cell"]
        == REFERENCE_PLUS,
        "ITEM_REFERENCE_PLUS",
    )
    require(
        row["reference_minus_cell"]
        == REFERENCE_MINUS,
        "ITEM_REFERENCE_MINUS",
    )
    require(
        row["anchor_name"]
        == ANCHOR_NAME,
        "ITEM_ANCHOR",
    )

    anchor_fields = (
        "target_plus_anchor",
        "target_minus_anchor",
        "reference_plus_anchor",
        "reference_minus_anchor",
    )
    for field in anchor_fields:
        require(
            type(row[field]) is int
            and row[field] >= 0,
            f"ITEM_ANCHOR_RANGE:{field}",
        )

    require(
        row["target_plus_intervention_token"]
        == row["target_plus_anchor"]
        + RELATIVE_COORDINATE,
        "ITEM_TARGET_PLUS_TOKEN",
    )
    require(
        row["target_minus_intervention_token"]
        == row["target_minus_anchor"]
        + RELATIVE_COORDINATE,
        "ITEM_TARGET_MINUS_TOKEN",
    )
    require(
        row["reference_plus_geometry_token"]
        == row["reference_plus_anchor"]
        + RELATIVE_COORDINATE,
        "ITEM_REFERENCE_PLUS_TOKEN",
    )
    require(
        row["reference_minus_geometry_token"]
        == row["reference_minus_anchor"]
        + RELATIVE_COORDINATE,
        "ITEM_REFERENCE_MINUS_TOKEN",
    )

    for field in (
        "target_A",
        "target_B",
        "reference_A",
        "reference_B",
        "alignment_realized_A",
        "alignment_realized_B",
        "magnitude_target_A",
        "magnitude_target_B",
        "magnitude_realized_A",
        "magnitude_realized_B",
    ):
        require(
            finite_number(row[field], field) > 0.0,
            f"ITEM_POSITIVE:{field}",
        )

    for field in (
        "target_C",
        "reference_C",
        "alignment_target_cosine",
        "alignment_realized_cosine",
        "magnitude_baseline_cosine",
        "magnitude_realized_cosine",
    ):
        value = finite_number(row[field], field)
        require(
            -1.0 - VECTOR_TOL
            <= value
            <= 1.0 + VECTOR_TOL,
            f"ITEM_COSINE_RANGE:{field}",
        )

    close(
        row["alignment_target_cosine"],
        row["reference_C"],
        "ALIGNMENT_TARGET_C",
    )
    close(
        row["alignment_A_preservation_abs_residual"],
        abs(
            float(row["alignment_realized_A"])
            - float(row["target_A"])
        ),
        "ALIGNMENT_A_RESIDUAL",
    )
    close(
        row["alignment_B_preservation_abs_residual"],
        abs(
            float(row["alignment_realized_B"])
            - float(row["target_B"])
        ),
        "ALIGNMENT_B_RESIDUAL",
    )
    close(
        row["alignment_cosine_abs_residual"],
        abs(
            float(row["alignment_realized_cosine"])
            - float(row["alignment_target_cosine"])
        ),
        "ALIGNMENT_C_RESIDUAL",
    )

    close(
        row["magnitude_target_A"],
        row["reference_A"],
        "MAGNITUDE_TARGET_A",
    )
    close(
        row["magnitude_target_B"],
        row["reference_B"],
        "MAGNITUDE_TARGET_B",
    )
    close(
        row["magnitude_baseline_cosine"],
        row["target_C"],
        "MAGNITUDE_BASELINE_C",
    )
    close(
        row["magnitude_A_target_abs_residual"],
        abs(
            float(row["magnitude_realized_A"])
            - float(row["magnitude_target_A"])
        ),
        "MAGNITUDE_A_RESIDUAL",
    )
    close(
        row["magnitude_B_target_abs_residual"],
        abs(
            float(row["magnitude_realized_B"])
            - float(row["magnitude_target_B"])
        ),
        "MAGNITUDE_B_RESIDUAL",
    )
    close(
        row["magnitude_cosine_abs_residual"],
        abs(
            float(row["magnitude_realized_cosine"])
            - float(row["magnitude_baseline_cosine"])
        ),
        "MAGNITUDE_C_RESIDUAL",
    )

    frozen_plus = frozen[(pair, TARGET_PLUS)]
    frozen_minus = frozen[(pair, TARGET_MINUS)]

    close(
        row["frozen_plus_path_efficiency"],
        frozen_plus,
        "FROZEN_PLUS",
    )
    close(
        row["frozen_minus_path_efficiency"],
        frozen_minus,
        "FROZEN_MINUS",
    )

    close(
        row["baseline_plus_reproduction_abs_residual"],
        abs(
            float(row["baseline_plus_path_efficiency"])
            - frozen_plus
        ),
        "BASELINE_PLUS_REPRO",
    )
    close(
        row["baseline_minus_reproduction_abs_residual"],
        abs(
            float(row["baseline_minus_path_efficiency"])
            - frozen_minus
        ),
        "BASELINE_MINUS_REPRO",
    )

    bplus = finite_number(
        row["baseline_plus_path_efficiency"],
        "BPLUS",
    )
    bminus = finite_number(
        row["baseline_minus_path_efficiency"],
        "BMINUS",
    )
    aplus = finite_number(
        row["alignment_plus_path_efficiency"],
        "APLUS",
    )
    aminus = finite_number(
        row["alignment_minus_path_efficiency"],
        "AMINUS",
    )
    mplus = finite_number(
        row["magnitude_plus_path_efficiency"],
        "MPLUS",
    )
    mminus = finite_number(
        row["magnitude_minus_path_efficiency"],
        "MMINUS",
    )

    delta0 = bplus - bminus
    delta_a = aplus - aminus
    delta_m = mplus - mminus
    r_align = delta_a - delta0
    r_mag = delta_m - delta0
    specificity = r_align - r_mag

    close(
        row["delta_baseline"],
        delta0,
        "DELTA_BASELINE",
    )
    close(
        row["delta_alignment"],
        delta_a,
        "DELTA_ALIGNMENT",
    )
    close(
        row["delta_magnitude"],
        delta_m,
        "DELTA_MAGNITUDE",
    )
    close(
        row["R_ALIGN"],
        r_align,
        "R_ALIGN",
    )
    close(
        row["R_MAG"],
        r_mag,
        "R_MAG",
    )
    close(
        row["ALIGNMENT_SPECIFICITY"],
        specificity,
        "ALIGNMENT_SPECIFICITY",
    )

    tolerance_fields = {
        "baseline_plus_reproduction_abs_residual":
            BASELINE_REPRO_TOL,
        "baseline_minus_reproduction_abs_residual":
            BASELINE_REPRO_TOL,
        "alignment_cosine_abs_residual":
            VECTOR_TOL,
        "alignment_A_preservation_abs_residual":
            VECTOR_TOL,
        "alignment_B_preservation_abs_residual":
            VECTOR_TOL,
        "magnitude_cosine_abs_residual":
            VECTOR_TOL,
        "magnitude_A_target_abs_residual":
            VECTOR_TOL,
        "magnitude_B_target_abs_residual":
            VECTOR_TOL,
        "alignment_midpoint_max_abs_residual":
            MIDPOINT_TOL,
        "magnitude_midpoint_max_abs_residual":
            MIDPOINT_TOL,
        "alignment_pair_delta_max_abs_residual":
            CAST_TOL,
        "magnitude_pair_delta_max_abs_residual":
            CAST_TOL,
        "alignment_applied_correction_max_abs_residual":
            CAST_TOL,
        "magnitude_applied_correction_max_abs_residual":
            CAST_TOL,
    }

    for field, tolerance in tolerance_fields.items():
        value = finite_number(row[field], field)
        require(
            0.0 <= value <= tolerance,
            f"ITEM_TOLERANCE:{field}:{value}",
        )


def _beta_continued_fraction(
    a: float,
    b: float,
    x: float,
) -> float:
    max_iterations = 300
    epsilon = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap

    if abs(d) < fpmin:
        d = fpmin

    d = 1.0 / d
    h = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = (
            m
            * (b - m)
            * x
            / ((qam + m2) * (a + m2))
        )

        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin

        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin

        d = 1.0 / d
        h *= d * c

        aa = (
            -(a + m)
            * (qab + m)
            * x
            / ((a + m2) * (qap + m2))
        )

        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin

        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin

        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) <= epsilon:
            return h

    raise ValidationError(
        "BETA_CONTINUED_FRACTION_DID_NOT_CONVERGE"
    )


def regularized_incomplete_beta(
    a: float,
    b: float,
    x: float,
) -> float:
    require(
        a > 0.0 and b > 0.0,
        "BETA_PARAMETER",
    )
    require(
        0.0 <= x <= 1.0,
        "BETA_X",
    )

    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)

    if x < (a + 1.0) / (a + b + 2.0):
        result = (
            bt
            * _beta_continued_fraction(
                a,
                b,
                x,
            )
            / a
        )
    else:
        result = (
            1.0
            - bt
            * _beta_continued_fraction(
                b,
                a,
                1.0 - x,
            )
            / b
        )

    return min(
        1.0,
        max(0.0, result),
    )


def student_t_two_sided_p(
    t_statistic: float,
    df: int,
) -> float:
    require(df > 0, "T_DF")
    require(
        not math.isnan(t_statistic),
        "T_NAN",
    )

    if math.isinf(t_statistic):
        return 0.0

    t_abs = abs(float(t_statistic))

    if t_abs == 0.0:
        return 1.0

    x = df / (
        df + t_abs * t_abs
    )

    return regularized_incomplete_beta(
        df / 2.0,
        0.5,
        x,
    )


def one_sided_greater_t(
    values: Sequence[float],
) -> dict[str, float | int]:
    vals = [
        finite_number(v, "T_VALUE")
        for v in values
    ]

    require(
        len(vals) == SOURCE_PAIR_COUNT,
        "T_SAMPLE_SIZE",
    )

    n = len(vals)
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals)

    require(sd > 0.0, "T_ZERO_VARIANCE")

    se = sd / math.sqrt(n)
    t = mean / se
    df = n - 1

    two_sided = student_t_two_sided_p(
        t,
        df,
    )

    raw_p = (
        0.5 * two_sided
        if t >= 0.0
        else 1.0 - 0.5 * two_sided
    )

    return {
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "standard_error": se,
        "df": df,
        "t_statistic": t,
        "raw_p_value":
            min(
                1.0,
                max(0.0, raw_p),
            ),
        "d_z": mean / sd,
    }


def holm_two(
    p1: float,
    p2: float,
) -> tuple[float, float]:
    values = [
        finite_number(p1, "P1"),
        finite_number(p2, "P2"),
    ]

    require(
        all(0.0 <= p <= 1.0 for p in values),
        "HOLM_RANGE",
    )

    order = sorted(
        range(2),
        key=lambda index: (
            values[index],
            index,
        ),
    )

    adjusted = [0.0, 0.0]
    running = 0.0

    for rank, index in enumerate(order):
        candidate = min(
            1.0,
            (2 - rank) * values[index],
        )
        running = max(
            running,
            candidate,
        )
        adjusted[index] = running

    return adjusted[0], adjusted[1]


SUMMARY_FIELDS = frozenset({
    "schema_version",
    "population_size",
    "model_forward_count",
    "baseline_delta_name_path_efficiency_mean",
    "frozen_baseline_delta_name_path_efficiency_mean",
    "max_baseline_reproduction_abs_residual",
    "max_alignment_cosine_abs_residual",
    "max_alignment_A_preservation_abs_residual",
    "max_alignment_B_preservation_abs_residual",
    "max_magnitude_cosine_abs_residual",
    "max_magnitude_A_target_abs_residual",
    "max_magnitude_B_target_abs_residual",
    "max_alignment_midpoint_abs_residual",
    "max_magnitude_midpoint_abs_residual",
    "max_alignment_pair_delta_abs_residual",
    "max_magnitude_pair_delta_abs_residual",
    "max_alignment_applied_correction_abs_residual",
    "max_magnitude_applied_correction_abs_residual",
    "all_mandatory_manipulation_checks_pass",
    "hypothesis_family",
    "outcome",
})

HYPOTHESIS_FIELDS = frozenset({
    "id",
    "n",
    "mean",
    "sample_sd",
    "standard_error",
    "df",
    "t_statistic",
    "raw_p_value",
    "d_z",
    "holm_adjusted_p_value",
    "reject_holm_alpha_0_05",
})


def maximum(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> float:
    return max(
        finite_number(
            row[field],
            field,
        )
        for row in rows
    )


def validate_summary(
    summary: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> None:
    require(
        set(summary) == SUMMARY_FIELDS,
        "SUMMARY_FIELDS",
    )
    require(
        summary["schema_version"]
        == SUMMARY_SCHEMA,
        "SUMMARY_SCHEMA",
    )
    require(
        summary["population_size"]
        == SOURCE_PAIR_COUNT,
        "SUMMARY_POPULATION",
    )
    require(
        summary["model_forward_count"]
        == FULL_FORWARD_BUDGET,
        "SUMMARY_FORWARD_COUNT",
    )

    baseline = [
        float(row["delta_baseline"])
        for row in items
    ]
    r_align = [
        float(row["R_ALIGN"])
        for row in items
    ]
    specificity = [
        float(row["ALIGNMENT_SPECIFICITY"])
        for row in items
    ]

    baseline_mean = statistics.fmean(
        baseline
    )

    close(
        baseline_mean,
        FROZEN_BASELINE_MEAN,
        "FROZEN_BASELINE_MEAN",
    )
    close(
        summary[
            "baseline_delta_name_path_efficiency_mean"
        ],
        baseline_mean,
        "SUMMARY_BASELINE_MEAN",
    )
    close(
        summary[
            "frozen_baseline_delta_name_path_efficiency_mean"
        ],
        FROZEN_BASELINE_MEAN,
        "SUMMARY_FROZEN_MEAN",
    )

    maxima = {
        "max_baseline_reproduction_abs_residual":
            max(
                maximum(
                    items,
                    "baseline_plus_reproduction_abs_residual",
                ),
                maximum(
                    items,
                    "baseline_minus_reproduction_abs_residual",
                ),
            ),
        "max_alignment_cosine_abs_residual":
            maximum(
                items,
                "alignment_cosine_abs_residual",
            ),
        "max_alignment_A_preservation_abs_residual":
            maximum(
                items,
                "alignment_A_preservation_abs_residual",
            ),
        "max_alignment_B_preservation_abs_residual":
            maximum(
                items,
                "alignment_B_preservation_abs_residual",
            ),
        "max_magnitude_cosine_abs_residual":
            maximum(
                items,
                "magnitude_cosine_abs_residual",
            ),
        "max_magnitude_A_target_abs_residual":
            maximum(
                items,
                "magnitude_A_target_abs_residual",
            ),
        "max_magnitude_B_target_abs_residual":
            maximum(
                items,
                "magnitude_B_target_abs_residual",
            ),
        "max_alignment_midpoint_abs_residual":
            maximum(
                items,
                "alignment_midpoint_max_abs_residual",
            ),
        "max_magnitude_midpoint_abs_residual":
            maximum(
                items,
                "magnitude_midpoint_max_abs_residual",
            ),
        "max_alignment_pair_delta_abs_residual":
            maximum(
                items,
                "alignment_pair_delta_max_abs_residual",
            ),
        "max_magnitude_pair_delta_abs_residual":
            maximum(
                items,
                "magnitude_pair_delta_max_abs_residual",
            ),
        "max_alignment_applied_correction_abs_residual":
            maximum(
                items,
                "alignment_applied_correction_max_abs_residual",
            ),
        "max_magnitude_applied_correction_abs_residual":
            maximum(
                items,
                "magnitude_applied_correction_max_abs_residual",
            ),
    }

    for field, expected in maxima.items():
        close(
            summary[field],
            expected,
            "SUMMARY_" + field,
        )

    manipulation_pass = (
        maxima[
            "max_baseline_reproduction_abs_residual"
        ]
        <= BASELINE_REPRO_TOL
        and maxima[
            "max_alignment_cosine_abs_residual"
        ]
        <= VECTOR_TOL
        and maxima[
            "max_alignment_A_preservation_abs_residual"
        ]
        <= VECTOR_TOL
        and maxima[
            "max_alignment_B_preservation_abs_residual"
        ]
        <= VECTOR_TOL
        and maxima[
            "max_magnitude_cosine_abs_residual"
        ]
        <= VECTOR_TOL
        and maxima[
            "max_magnitude_A_target_abs_residual"
        ]
        <= VECTOR_TOL
        and maxima[
            "max_magnitude_B_target_abs_residual"
        ]
        <= VECTOR_TOL
        and maxima[
            "max_alignment_midpoint_abs_residual"
        ]
        <= MIDPOINT_TOL
        and maxima[
            "max_magnitude_midpoint_abs_residual"
        ]
        <= MIDPOINT_TOL
        and maxima[
            "max_alignment_pair_delta_abs_residual"
        ]
        <= CAST_TOL
        and maxima[
            "max_magnitude_pair_delta_abs_residual"
        ]
        <= CAST_TOL
        and maxima[
            "max_alignment_applied_correction_abs_residual"
        ]
        <= CAST_TOL
        and maxima[
            "max_magnitude_applied_correction_abs_residual"
        ]
        <= CAST_TOL
    )

    require(
        summary[
            "all_mandatory_manipulation_checks_pass"
        ]
        is manipulation_pass,
        "SUMMARY_MANIPULATION_PASS",
    )

    h1 = one_sided_greater_t(
        r_align
    )
    h2 = one_sided_greater_t(
        specificity
    )
    h1_adj, h2_adj = holm_two(
        h1["raw_p_value"],
        h2["raw_p_value"],
    )

    family = summary["hypothesis_family"]

    require(
        isinstance(family, list)
        and len(family) == 2,
        "SUMMARY_HYPOTHESIS_COUNT",
    )

    expected = (
        (
            "H1_R_ALIGN_GT_ZERO",
            h1,
            h1_adj,
        ),
        (
            "H2_ALIGNMENT_SPECIFICITY_GT_ZERO",
            h2,
            h2_adj,
        ),
    )

    rejects = []

    for observed, (
        identifier,
        recomputed,
        adjusted,
    ) in zip(
        family,
        expected,
        strict=True,
    ):
        require(
            isinstance(observed, Mapping)
            and set(observed)
            == HYPOTHESIS_FIELDS,
            "HYPOTHESIS_FIELDS",
        )
        require(
            observed["id"] == identifier,
            "HYPOTHESIS_ID",
        )

        for field in (
            "n",
            "df",
        ):
            require(
                observed[field]
                == recomputed[field],
                f"HYPOTHESIS_{field}",
            )

        for field in (
            "mean",
            "sample_sd",
            "standard_error",
            "t_statistic",
            "raw_p_value",
            "d_z",
        ):
            close(
                observed[field],
                recomputed[field],
                "HYPOTHESIS_" + field,
            )

        close(
            observed[
                "holm_adjusted_p_value"
            ],
            adjusted,
            "HYPOTHESIS_HOLM",
        )

        reject = (
            adjusted < FAMILY_ALPHA
            and float(
                recomputed["mean"]
            ) > 0.0
        )

        require(
            observed[
                "reject_holm_alpha_0_05"
            ]
            is reject,
            "HYPOTHESIS_REJECT",
        )
        rejects.append(reject)

    expected_outcome = (
        "SUPPORTED_DIRECTIONAL_ALIGNMENT_CAUSAL_TRANSPORT"
        if (
            manipulation_pass
            and all(rejects)
        )
        else
        "DIRECTIONAL_ALIGNMENT_CAUSAL_TRANSPORT_NOT_ESTABLISHED"
    )

    require(
        summary["outcome"]
        == expected_outcome,
        "SUMMARY_OUTCOME",
    )


def validate_preflight_bundle(
    directory: Path,
    *,
    expected_execution_head: str,
) -> dict[str, Any]:
    expected_files = {
        MANIFEST_FILE,
        PREFLIGHT_FILE,
        CHECKSUM_FILE,
    }

    require(
        {
            path.name
            for path in directory.iterdir()
        }
        == expected_files,
        "PREFLIGHT_FILE_SET",
    )

    validate_checksums(
        directory,
        (
            MANIFEST_FILE,
            PREFLIGHT_FILE,
        ),
    )

    manifest = json_object(
        directory / MANIFEST_FILE,
        "MANIFEST",
    )
    preflight = json_object(
        directory / PREFLIGHT_FILE,
        "PREFLIGHT",
    )

    reject_private_payload(manifest)
    reject_private_payload(preflight)

    code_sha = (
        validate_frozen_code_identity(
            expected_execution_head
        )
    )

    validate_manifest(
        manifest,
        expected_execution_head=
            expected_execution_head,
        mode="preflight",
        code_sha=code_sha,
    )
    validate_preflight_public(
        preflight
    )

    return {
        "mode": "preflight",
        "result":
            "PASS_INDEPENDENT_PREFLIGHT_VALIDATION",
        "model_forward_count":
            PREFLIGHT_FORWARD_BUDGET,
        "scientific_conclusion":
            "NONE",
    }


def validate_full_bundle(
    directory: Path,
    *,
    expected_execution_head: str,
) -> dict[str, Any]:
    expected_files = {
        MANIFEST_FILE,
        ITEM_FILE,
        SUMMARY_FILE,
        CHECKSUM_FILE,
    }

    require(
        {
            path.name
            for path in directory.iterdir()
        }
        == expected_files,
        "FULL_FILE_SET",
    )

    validate_checksums(
        directory,
        (
            MANIFEST_FILE,
            ITEM_FILE,
            SUMMARY_FILE,
        ),
    )

    manifest = json_object(
        directory / MANIFEST_FILE,
        "MANIFEST",
    )
    items = jsonl_objects(
        directory / ITEM_FILE,
        "ITEMS",
    )
    summary = json_object(
        directory / SUMMARY_FILE,
        "SUMMARY",
    )

    reject_private_payload(manifest)
    reject_private_payload(items)
    reject_private_payload(summary)

    code_sha = (
        validate_frozen_code_identity(
            expected_execution_head
        )
    )

    validate_manifest(
        manifest,
        expected_execution_head=
            expected_execution_head,
        mode="full",
        code_sha=code_sha,
    )

    require(
        len(items) == SOURCE_PAIR_COUNT,
        "FULL_ITEM_COUNT",
    )

    frozen = load_frozen_endpoint_map()
    seen = set()

    for row in items:
        pair = str(
            row.get(
                "source_pair_id",
                "",
            )
        )
        require(
            pair not in seen,
            f"DUPLICATE_ITEM_PAIR:{pair}",
        )
        seen.add(pair)
        validate_item(
            row,
            frozen,
        )

    frozen_pairs = {
        pair
        for pair, _cell
        in frozen
    }

    require(
        seen == frozen_pairs,
        "FULL_PAIR_POPULATION",
    )

    validate_summary(
        summary,
        items,
    )

    return {
        "mode": "full",
        "result":
            "PASS_INDEPENDENT_FULL_VALIDATION",
        "model_forward_count":
            FULL_FORWARD_BUDGET,
        "outcome":
            summary["outcome"],
    }


def validate_bundle(
    directory: Path,
    *,
    expected_execution_head: str,
) -> dict[str, Any]:
    require(
        directory.is_dir(),
        "BUNDLE_DIRECTORY_MISSING",
    )

    require(
        git_text(
            "branch",
            "--show-current",
        )
        == EXPECTED_BRANCH,
        "VALIDATOR_BRANCH",
    )
    require(
        git_text(
            "rev-parse",
            "HEAD",
        )
        == expected_execution_head,
        "VALIDATOR_HEAD",
    )

    manifest = json_object(
        directory / MANIFEST_FILE,
        "MANIFEST_PROBE",
    )

    mode = manifest.get("mode")

    if mode == "preflight":
        return validate_preflight_bundle(
            directory,
            expected_execution_head=
                expected_execution_head,
        )

    if mode == "full":
        return validate_full_bundle(
            directory,
            expected_execution_head=
                expected_execution_head,
        )

    raise ValidationError(
        f"UNKNOWN_MODE:{mode}"
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bundle-dir",
        required=True,
    )
    parser.add_argument(
        "--expected-execution-head",
        required=True,
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)

    result = validate_bundle(
        Path(args.bundle_dir),
        expected_execution_head=
            args.expected_execution_head,
    )

    print(
        "RESULT =",
        result["result"],
    )
    print(
        "MODE =",
        result["mode"],
    )
    print(
        "MODEL_FORWARD_COUNT =",
        result["model_forward_count"],
    )

    if result["mode"] == "preflight":
        print(
            "SCIENTIFIC_CONCLUSION = NONE"
        )
    else:
        print(
            "OUTCOME =",
            result["outcome"],
        )


if __name__ == "__main__":
    main()
