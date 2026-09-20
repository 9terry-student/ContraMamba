#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (
    build_reason_router_gen4_precursor_v2_dcs_stage_a_cohort
    as cohort_builder,
)
from scripts import (
    reason_router_gen4_averitec_370m_fixed_mirror_steering_fast_cuda
    as steering,
)
from scripts import (
    reason_router_gen4_pre_emission_forced_decisive_stage_a_generation_fast_cuda
    as forced,
)
from scripts import (
    reason_router_gen4_pre_emission_stage_a_generation_fast_cuda
    as base,
)


EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "b05d1394ed97110894f4f3c093d4d08cfe0e60d5"

DESIGN_ARTIFACT = Path(
    "reports/reason_router_gen4_precursor_v2_dynamic_causal_susceptibility_stage_a_design.md"
)
DESIGN_GIT_BLOB = "d24011d698de1a53436a6bb997b2bb400881a718"

FORWARD_CORRECTION = Path(
    "reports/reason_router_gen4_precursor_v2_dcs_forward_accounting_correction.md"
)
FORWARD_CORRECTION_GIT_BLOB = "a37393140dc05a19889805206b5ec3070575dbe1"

COHORT_ROOT = Path("data/reason_router_gen4_precursor_v2_dcs_stage_a_cohort_v1")
COHORT_FILE = COHORT_ROOT / "precursor_v2_stage_a_cohort.jsonl"
COHORT_MANIFEST = COHORT_ROOT / "cohort_manifest.json"
COHORT_SUMS = COHORT_ROOT / "SHA256SUMS.txt"

COHORT_GIT_BLOB = "842e66434af5e142f9b8b0abff1a57b986884e4f"
COHORT_MANIFEST_GIT_BLOB = "7b86a94add40f3f89f2c32f2a37b67c4a7c22b6f"
COHORT_SUMS_GIT_BLOB = "885a0816a689391e53562f2ec66b3db37e05b3bf"

COHORT_SHA256 = "d12bb49dfeeebe7cf6eaeeabb5125717f979f85df5782141f0efea8346cab4a9"
COHORT_MANIFEST_SHA256 = "81fa809d28172cb3cfcaa35c2ee659a2ab2664a83608342671647b2e0e0c05f4"
COHORT_SUMS_SHA256 = "2f4f9c33c2189059361ceb72fab695bfd9b31a42403deeb65f4bb0aa77904b1e"

COHORT_BUILDER = Path(
    "scripts/build_reason_router_gen4_precursor_v2_dcs_stage_a_cohort.py"
)
COHORT_BUILDER_GIT_BLOB = "5b79b9ad835c6785120c0cb9056c107f8ff439ba"

STEERING_RUNNER = Path(
    "scripts/reason_router_gen4_averitec_370m_fixed_mirror_steering_fast_cuda.py"
)
STEERING_RUNNER_GIT_BLOB = "9e724927bcc700608d20898fa1b3f1595fb42bb0"

FORCED_RUNNER = Path(
    "scripts/reason_router_gen4_pre_emission_forced_decisive_stage_a_generation_fast_cuda.py"
)
FORCED_RUNNER_GIT_BLOB = "99fb5a7b43d35d667d3dda646adea0e37392e647"

BASE_GENERATOR = Path(
    "scripts/reason_router_gen4_pre_emission_stage_a_generation_fast_cuda.py"
)
BASE_GENERATOR_GIT_BLOB = "aa6725bdc1cbfa43547e89293f3b7911a4cdd6ed"

N = 800
SOURCE_LABEL_COUNTS = {"Refuted": 400, "Supported": 400}
CORRECT_LABEL_COUNTS = {"REFUTE": 400, "SUPPORT": 400}

SELECTED_PLANE = "P3"
CONTROL_PLANE = "P5"
EARLY_BLOCK = 35
LATE_BLOCK = 47
DIM = 650

EPSILON = 0.025
OBSERVATION_OFFSETS = (-4, -3, -2, -1)
OBSERVATION_PREFIX_LENGTHS = (5, 6, 7, 8)
PREFIX_LENGTH_TO_OFFSET = dict(
    zip(OBSERVATION_PREFIX_LENGTHS, OBSERVATION_OFFSETS)
)

FORCED_CLASS_ORDER = ("REFUTE", "SUPPORT")
NATIVE_FORWARDS_PER_ROW = 12
PLANES = (SELECTED_PLANE, CONTROL_PLANE)
BASIS_NAMES = ("plus", "minus")
EPSILON_SIGNS = (-1, 1)
PROBES_PER_OFFSET = (
    len(PLANES) * len(BASIS_NAMES) * len(EPSILON_SIGNS)
)
PROBES_PER_ROW = PROBES_PER_OFFSET * len(OBSERVATION_OFFSETS)
TOTAL_FORWARDS_PER_ROW = NATIVE_FORWARDS_PER_ROW + PROBES_PER_ROW
SCIENTIFIC_FORWARD_BUDGET = N * TOTAL_FORWARDS_PER_ROW

SHARDS = (
    {
        "shard_id": 0,
        "physical_device": 0,
        "start": 0,
        "end": 400,
        "example_count": 400,
        "forward_budget": 17600,
    },
    {
        "shard_id": 1,
        "physical_device": 1,
        "start": 400,
        "end": 800,
        "example_count": 400,
        "forward_budget": 17600,
    },
)
SHARD_COUNT = len(SHARDS)

ROW_FILE = "precursor_v2_dcs_stage_a_raw_rows.jsonl"
SUMMARY_FILE = "execution_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ROW_SCHEMA = "gen4-precursor-v2-dcs-stage-a-raw-row-v1"
SHARD_SCHEMA = "gen4-precursor-v2-dcs-stage-a-shard-v1"
SUMMARY_SCHEMA = "gen4-precursor-v2-dcs-stage-a-raw-summary-v1"
RESULT_PASS = "PASS_PRECURSOR_V2_DCS_STAGE_A_RAW"

PRIMARY_INFERENCE_EXECUTED = False
P_VALUE_COUNT_ADDED = 0


class PrecursorV2DCSError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PrecursorV2DCSError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrecursorV2DCSError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"JSONL_MISSING:{path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{path}:{line_no}",
        )
        rows.append(value)
    return rows


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REQUIRED_ANCESTOR")

    pinned = {
        DESIGN_ARTIFACT: DESIGN_GIT_BLOB,
        FORWARD_CORRECTION: FORWARD_CORRECTION_GIT_BLOB,
        COHORT_FILE: COHORT_GIT_BLOB,
        COHORT_MANIFEST: COHORT_MANIFEST_GIT_BLOB,
        COHORT_SUMS: COHORT_SUMS_GIT_BLOB,
        COHORT_BUILDER: COHORT_BUILDER_GIT_BLOB,
        STEERING_RUNNER: STEERING_RUNNER_GIT_BLOB,
        FORCED_RUNNER: FORCED_RUNNER_GIT_BLOB,
        BASE_GENERATOR: BASE_GENERATOR_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path.as_posix()}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_protocol() -> None:
    require(N == 800, "N")
    require(SOURCE_LABEL_COUNTS == {"Refuted": 400, "Supported": 400}, "SOURCE_LABEL_COUNTS")
    require(CORRECT_LABEL_COUNTS == {"REFUTE": 400, "SUPPORT": 400}, "CORRECT_LABEL_COUNTS")
    require(SELECTED_PLANE == "P3", "SELECTED_PLANE")
    require(CONTROL_PLANE == "P5", "CONTROL_PLANE")
    require(EARLY_BLOCK == 35, "EARLY_BLOCK")
    require(LATE_BLOCK == 47, "LATE_BLOCK")
    require(DIM == 650, "DIM")
    require(EPSILON == 0.025, "EPSILON")
    require(OBSERVATION_OFFSETS == (-4, -3, -2, -1), "OFFSETS")
    require(
        OBSERVATION_PREFIX_LENGTHS == (5, 6, 7, 8),
        "PREFIX_LENGTHS",
    )
    require(PROBES_PER_OFFSET == 8, "PROBES_PER_OFFSET")
    require(PROBES_PER_ROW == 32, "PROBES_PER_ROW")
    require(NATIVE_FORWARDS_PER_ROW == 12, "NATIVE_FORWARDS_PER_ROW")
    require(TOTAL_FORWARDS_PER_ROW == 44, "TOTAL_FORWARDS_PER_ROW")
    require(SCIENTIFIC_FORWARD_BUDGET == 35200, "SCIENTIFIC_FORWARD_BUDGET")
    require(SHARD_COUNT == 2, "SHARD_COUNT")

    covered: list[int] = []
    forward_budget = 0
    for expected_id, shard in enumerate(SHARDS):
        require(int(shard["shard_id"]) == expected_id, "SHARD_ID")
        require(int(shard["physical_device"]) == expected_id, "SHARD_DEVICE")
        start = int(shard["start"])
        end = int(shard["end"])
        count = int(shard["example_count"])
        require(end - start == count, f"SHARD_COUNT:{expected_id}")
        require(
            int(shard["forward_budget"]) == count * TOTAL_FORWARDS_PER_ROW,
            f"SHARD_FORWARD_BUDGET:{expected_id}",
        )
        covered.extend(range(start, end))
        forward_budget += int(shard["forward_budget"])
    require(covered == list(range(N)), "SHARD_COVERAGE")
    require(forward_budget == SCIENTIFIC_FORWARD_BUDGET, "SHARD_FORWARD_SUM")

    require(base.SELECTED_PLANE == SELECTED_PLANE, "BASE_SELECTED_PLANE")
    require(base.CONTROL_PLANE == CONTROL_PLANE, "BASE_CONTROL_PLANE")
    require(base.EARLY_BLOCK == EARLY_BLOCK, "BASE_EARLY_BLOCK")
    require(base.LATE_BLOCK == LATE_BLOCK, "BASE_LATE_BLOCK")
    require(base.DIM == DIM, "BASE_DIM")
    require(forced.FORCED_CLASS_ORDER == FORCED_CLASS_ORDER, "FORCED_CLASS_ORDER")
    require(
        forced.OBSERVATION_OFFSETS == OBSERVATION_OFFSETS,
        "FORCED_OFFSETS",
    )
    require(
        forced.OBSERVATION_GENERATED_PREFIX_LENGTHS
        == OBSERVATION_PREFIX_LENGTHS,
        "FORCED_PREFIX_LENGTHS",
    )


def validate_frozen_cohort() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(sha256_file(ROOT / COHORT_FILE) == COHORT_SHA256, "COHORT_SHA256")
    require(
        sha256_file(ROOT / COHORT_MANIFEST) == COHORT_MANIFEST_SHA256,
        "COHORT_MANIFEST_SHA256",
    )
    require(
        sha256_file(ROOT / COHORT_SUMS) == COHORT_SUMS_SHA256,
        "COHORT_SUMS_SHA256",
    )

    declared: dict[str, str] = {}
    for line in (ROOT / COHORT_SUMS).read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in declared, f"SUM_DUPLICATE:{name}")
        declared[name] = digest
    require(
        declared
        == {
            "cohort_manifest.json": COHORT_MANIFEST_SHA256,
            "precursor_v2_stage_a_cohort.jsonl": COHORT_SHA256,
        },
        "COHORT_SUMS_CONTENT",
    )

    manifest = json.loads(
        (ROOT / COHORT_MANIFEST).read_text(encoding="utf-8")
    )
    require(
        manifest["schema_version"]
        == "GEN4_PRECURSOR_V2_DCS_STAGE_A_COHORT_MANIFEST_V1",
        "MANIFEST_SCHEMA",
    )
    require(
        manifest["result"]
        == "PASS_PRECURSOR_V2_DCS_STAGE_A_RESPONSE_FREE_COHORT",
        "MANIFEST_RESULT",
    )
    require(manifest["cohort"]["count"] == N, "MANIFEST_N")
    require(
        manifest["cohort"]["source_label_counts"] == SOURCE_LABEL_COUNTS,
        "MANIFEST_SOURCE_LABELS",
    )
    require(
        manifest["cohort"]["correct_label_counts"] == CORRECT_LABEL_COUNTS,
        "MANIFEST_CORRECT_LABELS",
    )
    require(
        manifest["cohort"]["cohort_file_sha256"] == COHORT_SHA256,
        "MANIFEST_COHORT_SHA",
    )
    require(manifest["response_fields_present"] is False, "MANIFEST_RESPONSE_FIELDS")
    require(manifest["model_checkpoint_loaded"] is False, "MANIFEST_MODEL_LOAD")
    require(manifest["model_forward_count"] == 0, "MANIFEST_FORWARD_COUNT")
    require(manifest["cuda_executed"] is False, "MANIFEST_CUDA")
    require(manifest["training_executed"] is False, "MANIFEST_TRAINING")
    require(manifest["backward_executed"] is False, "MANIFEST_BACKWARD")
    require(
        manifest["scientific_inference_executed"] is False,
        "MANIFEST_INFERENCE",
    )
    require(manifest["p_value_count_added"] == 0, "MANIFEST_P_VALUE")
    require(
        manifest["scientific_execution_authorized"] is False,
        "MANIFEST_EXECUTION_AUTHORIZATION",
    )
    require(manifest["runner_implemented"] is False, "MANIFEST_RUNNER_PRESTATE")
    require(manifest["equivalence_executed"] is False, "MANIFEST_EQUIVALENCE_PRESTATE")
    require(
        manifest["selection"]["namespace"]
        == cohort_builder.SELECTION_NAMESPACE,
        "MANIFEST_NAMESPACE",
    )
    require(
        manifest["selection"]["response_fields_used"] is False,
        "MANIFEST_SELECTION_RESPONSE",
    )
    require(
        manifest["selection"]["experiment3_response_accessed"] is False,
        "MANIFEST_EXPERIMENT3_RESPONSE",
    )
    require(
        manifest["selection"]["old_precursor_response_accessed"] is False,
        "MANIFEST_OLD_RESPONSE",
    )

    rows = read_jsonl(ROOT / COHORT_FILE)
    require(len(rows) == N, "COHORT_N")
    require(
        len({str(row["example_id"]) for row in rows}) == N,
        "COHORT_ID_UNIQUENESS",
    )
    require(
        [int(row["averitec_train_index"]) for row in rows]
        == sorted(int(row["averitec_train_index"]) for row in rows),
        "COHORT_ORDER",
    )
    require(
        dict(Counter(str(row["source_label"]) for row in rows))
        == SOURCE_LABEL_COUNTS,
        "COHORT_SOURCE_LABELS",
    )
    require(
        dict(Counter(str(row["correct_label"]) for row in rows))
        == CORRECT_LABEL_COUNTS,
        "COHORT_CORRECT_LABELS",
    )

    forbidden = set(cohort_builder.FORBIDDEN_RESPONSE_FIELDS)
    for index, row in enumerate(rows):
        require(
            row["schema_version"]
            == "GEN4_PRECURSOR_V2_DCS_STAGE_A_COHORT_V1",
            f"COHORT_ROW_SCHEMA:{index}",
        )
        require(
            not (forbidden & set(row)),
            f"COHORT_RESPONSE_FIELD:{index}",
        )
        require(row["token_gate_pass"] is True, f"COHORT_GATE:{index}")
        require(row["token_gate_reasons"] == [], f"COHORT_GATE_REASONS:{index}")
        require(
            str(row["correct_label"]) in FORCED_CLASS_ORDER,
            f"COHORT_GOLD:{index}",
        )
        require(
            str(row["source_label"]) in SOURCE_LABEL_COUNTS,
            f"COHORT_SOURCE:{index}",
        )
        require(
            row["fresh_for_precursor_v2_generation_response"] is True,
            f"COHORT_FRESHNESS:{index}",
        )
        require(
            row["experiment3_response_accessed_for_selection"] is False,
            f"COHORT_EXP3_SELECTION:{index}",
        )
        require(
            row["old_precursor_response_accessed_for_selection"] is False,
            f"COHORT_OLD_SELECTION:{index}",
        )
        require(bool(str(row["claim"])), f"COHORT_CLAIM:{index}")
        require(bool(str(row["evidence"])), f"COHORT_EVIDENCE:{index}")
    return rows, manifest


def model_rows(
    cohort: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out = [
        {
            "row_id": str(row["example_id"]),
            "source_pair_id": str(row["example_id"]),
            "contrast_cell_id": "PRECURSOR_V2_DCS",
            "claim": str(row["claim"]),
            "evidence": str(row["evidence"]),
        }
        for row in cohort
    ]
    require(len(out) == N, "MODEL_ROWS_N")
    return out


def validate_encoding(
    *,
    cohort: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
) -> None:
    for key in (
        "input_ids",
        "attention_mask",
        "claim_mask",
        "evidence_mask",
    ):
        require(key in encoded, f"ENCODED_KEY:{key}")
        require(torch.is_tensor(encoded[key]), f"ENCODED_TENSOR:{key}")
        require(
            tuple(encoded[key].shape) == (N, 128),
            f"ENCODED_SHAPE:{key}:{tuple(encoded[key].shape)}",
        )

    for index, row in enumerate(cohort):
        attention = encoded["attention_mask"][index]
        claim = encoded["claim_mask"][index]
        evidence = encoded["evidence_mask"][index]

        attended = int(attention.sum().item())
        claim_count = int(claim.sum().item())
        evidence_count = int(evidence.sum().item())

        require(
            attended == int(row["serialized_attended_length"]),
            f"ENCODED_ATTENDED:{index}",
        )
        require(
            claim_count == int(row["claim_consumed_token_count"]),
            f"ENCODED_CLAIM:{index}",
        )
        require(
            evidence_count == int(row["evidence_consumed_token_count"]),
            f"ENCODED_EVIDENCE:{index}",
        )
        require(
            int(row["absolute_anchor_token_index"]) == claim_count,
            f"ENCODED_ANCHOR:{index}",
        )
        require(
            int(row["target_intervention_token_index"])
            == claim_count + int(row["target_offset"]),
            f"ENCODED_TARGET:{index}",
        )
        require(
            bool(torch.all(attention[:attended] == 1).item()),
            f"ENCODED_PREFIX_ATTENTION:{index}",
        )
        if attended < attention.numel():
            require(
                bool(torch.all(attention[attended:] == 0).item()),
                f"ENCODED_SUFFIX_ATTENTION:{index}",
            )


def build_input(
    *,
    snapshot: Path,
    cohort: Sequence[Mapping[str, Any]],
    cohort_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    spec = steering.bridge.scale_spec("mamba370m")
    spec["geom"].validate_snapshot(snapshot)
    tokenizer, tokenizer_provenance = spec["geom"].load_tokenizer(snapshot)
    rows = model_rows(cohort)
    encoded = spec["adapter"].encode_gen4_rows(rows, tokenizer)
    validate_encoding(cohort=cohort, encoded=encoded)

    expected_tok = cohort_manifest["tokenizer"]
    require(
        tokenizer_provenance["repo"] == expected_tok["repo"],
        "TOKENIZER_REPO",
    )
    require(
        tokenizer_provenance["revision"] == expected_tok["revision"],
        "TOKENIZER_REVISION",
    )
    require(
        tokenizer_provenance["file_sha256"]
        == expected_tok["file_sha256"],
        "TOKENIZER_FILE_SHA256",
    )
    return rows, encoded, tokenizer_provenance


def extract_prompt_ids(
    encoded: Mapping[str, Any],
    *,
    index: int,
    expected_attended: int,
) -> torch.Tensor:
    input_ids = encoded["input_ids"][index].detach().cpu().to(torch.long)
    attention = encoded["attention_mask"][index].detach().cpu().to(torch.long)
    attended = int(attention.sum().item())
    require(attended == expected_attended, f"PROMPT_ATTENDED:{index}")
    require(attended > 0, f"PROMPT_EMPTY:{index}")
    require(
        bool(torch.all(attention[:attended] == 1).item()),
        f"PROMPT_PREFIX_MASK:{index}",
    )
    if attended < attention.numel():
        require(
            bool(torch.all(attention[attended:] == 0).item()),
            f"PROMPT_SUFFIX_MASK:{index}",
        )
    return input_ids[:attended].contiguous()


def gold_aligned_margin(
    gold_label: str,
    branch_logits: Mapping[str, float],
) -> float:
    require(gold_label in FORCED_CLASS_ORDER, f"GOLD_LABEL:{gold_label}")
    require(
        tuple(branch_logits) == FORCED_CLASS_ORDER,
        "BRANCH_LOGIT_ORDER",
    )
    refute = float(branch_logits["REFUTE"])
    support = float(branch_logits["SUPPORT"])
    require(
        math.isfinite(refute) and math.isfinite(support),
        "BRANCH_LOGIT_NONFINITE",
    )
    value = (
        support - refute
        if gold_label == "SUPPORT"
        else refute - support
    )
    require(math.isfinite(value), "GOLD_MARGIN_NONFINITE")
    return value


def perturbation_hook(
    output: torch.Tensor,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    basis: torch.Tensor,
    epsilon_sign: int,
    epsilon: float,
    intermediate_size: int,
    dim: int,
    cast_tol: float,
    audit: dict[str, Any],
) -> torch.Tensor:
    require(epsilon_sign in (-1, 1), "PERTURB_SIGN")
    require(epsilon > 0.0, "PERTURB_EPSILON")
    require(
        torch.is_tensor(output)
        and output.ndim == 3
        and output.shape[0] == 1
        and output.shape[-1] == 2 * intermediate_size,
        "PERTURB_INPROJ_SHAPE",
    )
    require(0 <= token_index < output.shape[1], "PERTURB_TOKEN_INDEX")

    before = output.detach().clone()
    mask = strong_mask.detach().cpu().bool().contiguous()
    require(
        mask.numel() == intermediate_size
        and int(mask.sum().item()) == dim,
        "PERTURB_STRONG_MASK",
    )
    basis64 = basis.detach().cpu().to(torch.float64).contiguous()
    require(tuple(basis64.shape) == (dim,), "PERTURB_BASIS_SHAPE")
    require(bool(torch.isfinite(basis64).all().item()), "PERTURB_BASIS_FINITE")

    delta64 = (float(epsilon_sign) * float(epsilon) * basis64).contiguous()
    out = output.clone()
    mask_device = mask.to(out.device)
    intended = delta64.to(device=out.device, dtype=out.dtype)
    out[
        0,
        token_index,
        :intermediate_size,
    ][mask_device] += intended

    require(
        torch.equal(
            out[:, :, intermediate_size:],
            before[:, :, intermediate_size:],
        ),
        "PERTURB_GATE_CHANGED",
    )
    require(
        torch.equal(
            out[:, :, :intermediate_size][:, :, ~mask_device],
            before[:, :, :intermediate_size][:, :, ~mask_device],
        ),
        "PERTURB_NONSTRONG_CHANGED",
    )
    if token_index:
        require(
            torch.equal(
                out[:, :token_index, :],
                before[:, :token_index, :],
            ),
            "PERTURB_EARLIER_TOKEN_CHANGED",
        )
    if token_index + 1 < out.shape[1]:
        require(
            torch.equal(
                out[:, token_index + 1:, :],
                before[:, token_index + 1:, :],
            ),
            "PERTURB_LATER_TOKEN_CHANGED",
        )

    applied = (
        out[0, token_index, :intermediate_size][mask_device]
        - before[0, token_index, :intermediate_size][mask_device]
    ).detach().cpu().to(torch.float64)
    intended64 = intended.detach().cpu().to(torch.float64)
    residual = float(torch.max(torch.abs(applied - intended64)).item())
    require(residual <= cast_tol, f"PERTURB_APPLIED_RESIDUAL:{residual}")

    audit.clear()
    audit.update({
        "token_index": int(token_index),
        "epsilon": float(epsilon),
        "epsilon_sign": int(epsilon_sign),
        "basis_l2": float(torch.linalg.vector_norm(basis64).item()),
        "intended_delta_l2": float(
            torch.linalg.vector_norm(delta64).item()
        ),
        "applied_delta_l2": float(
            torch.linalg.vector_norm(applied).item()
        ),
        "applied_correction_max_abs_residual": residual,
        "strong_dim": int(dim),
        "gate_changed": False,
        "nonstrong_changed": False,
        "other_token_changed": False,
    })
    return out


def probe_forward(
    *,
    model: Any,
    runtime: Mapping[str, Any],
    frozen: Mapping[str, Any],
    input_ids: torch.Tensor,
    gold_label: str,
    branch_tokens: Mapping[str, int],
    plane: str,
    basis_name: str,
    epsilon_sign: int,
    cast_tol: float,
) -> dict[str, Any]:
    require(plane in PLANES, f"PROBE_PLANE:{plane}")
    require(basis_name in BASIS_NAMES, f"PROBE_BASIS:{basis_name}")
    require(epsilon_sign in EPSILON_SIGNS, f"PROBE_SIGN:{epsilon_sign}")
    require(
        input_ids.ndim == 2 and input_ids.shape[0] == 1,
        "PROBE_INPUT_SHAPE",
    )
    require(input_ids.dtype == torch.long, "PROBE_INPUT_DTYPE")

    token_index = int(input_ids.shape[1]) - 1
    require(token_index >= 0, "PROBE_EMPTY_INPUT")
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    audit: dict[str, Any] = {}
    capture: dict[str, Any] = {}

    basis = frozen["planes"][plane][basis_name]
    intermediate_size = int(
        runtime["strong_mask"].detach().cpu().numel()
    )
    require(intermediate_size > DIM, "PROBE_INTERMEDIATE_SIZE")

    def early_hook(_module, _args, output):
        return perturbation_hook(
            output,
            token_index=token_index,
            strong_mask=runtime["strong_mask"],
            basis=basis,
            epsilon_sign=epsilon_sign,
            epsilon=EPSILON,
            intermediate_size=intermediate_size,
            dim=DIM,
            cast_tol=cast_tol,
            audit=audit,
        )

    def late_hook(_module, _args, output):
        value = base._hook_tensor(output, "POST_BLOCK_47")
        capture["post_block_47"] = value.detach().clone()
        return None

    early_handle = runtime["early_in_proj"].register_forward_hook(early_hook)
    late_handle = runtime["late_layer"].register_forward_hook(late_hook)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
    finally:
        late_handle.remove()
        early_handle.remove()

    require(
        set(capture) == {"post_block_47"},
        "PROBE_CAPTURE_KEYS",
    )
    logits = output.logits
    require(
        torch.is_tensor(logits)
        and tuple(logits.shape[:2]) == (1, 1),
        f"PROBE_LOGITS_SHAPE:{tuple(logits.shape)}",
    )
    next_logits = logits[0, -1].detach()

    manual_logits = base.late_readout_from_post_block_47(
        capture["post_block_47"],
        final_norm=runtime["final_norm"],
        lm_head=runtime["lm_head"],
    )
    require(
        manual_logits.shape == next_logits.shape,
        "PROBE_MANUAL_LOGIT_SHAPE",
    )
    repro = float(
        torch.max(torch.abs(manual_logits - next_logits))
        .detach().cpu().item()
    )
    require(
        repro <= base.LATE_READOUT_REPRO_ATOL,
        f"PROBE_LATE_READOUT_REPRO:{repro}",
    )

    branch_logits = {
        label: float(
            manual_logits[int(branch_tokens[label])]
            .detach().cpu().item()
        )
        for label in FORCED_CLASS_ORDER
    }
    margin = gold_aligned_margin(gold_label, branch_logits)
    return {
        "plane": plane,
        "basis": basis_name,
        "epsilon_sign": int(epsilon_sign),
        "epsilon": EPSILON,
        "branch_start_logits": branch_logits,
        "gold_aligned_decisive_margin": margin,
        "late_readout_reproduction_max_abs_error": repro,
        "intervention_audit": dict(audit),
        "scientific_full_model_forward_count": 1,
    }


def summarize_offset_from_probe_values(
    *,
    probes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(len(probes) == PROBES_PER_OFFSET, "OFFSET_PROBE_COUNT")
    by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for probe in probes:
        key = (
            str(probe["plane"]),
            str(probe["basis"]),
            int(probe["epsilon_sign"]),
        )
        require(key not in by_key, f"OFFSET_PROBE_DUP:{key}")
        require(key[0] in PLANES, f"OFFSET_PLANE:{key[0]}")
        require(key[1] in BASIS_NAMES, f"OFFSET_BASIS:{key[1]}")
        require(key[2] in EPSILON_SIGNS, f"OFFSET_SIGN:{key[2]}")
        by_key[key] = probe

    expected_keys = {
        (plane, basis, sign)
        for plane in PLANES
        for basis in BASIS_NAMES
        for sign in EPSILON_SIGNS
    }
    require(set(by_key) == expected_keys, "OFFSET_PROBE_KEYS")

    derivatives: dict[str, dict[str, float]] = {}
    chi: dict[str, float] = {}
    for plane in PLANES:
        derivatives[plane] = {}
        squares = 0.0
        for basis in BASIS_NAMES:
            f_plus = float(
                by_key[(plane, basis, 1)][
                    "gold_aligned_decisive_margin"
                ]
            )
            f_minus = float(
                by_key[(plane, basis, -1)][
                    "gold_aligned_decisive_margin"
                ]
            )
            require(
                math.isfinite(f_plus) and math.isfinite(f_minus),
                "OFFSET_MARGIN_NONFINITE",
            )
            derivative = (f_plus - f_minus) / (2.0 * EPSILON)
            require(math.isfinite(derivative), "OFFSET_DERIVATIVE_NONFINITE")
            derivatives[plane][basis] = derivative
            squares += derivative * derivative
        chi_value = math.sqrt(squares)
        require(math.isfinite(chi_value), "OFFSET_CHI_NONFINITE")
        chi[plane] = chi_value

    d_value = chi[SELECTED_PLANE] - chi[CONTROL_PLANE]
    require(math.isfinite(d_value), "OFFSET_D_NONFINITE")
    return {
        "derivatives": derivatives,
        "chi_p3": chi[SELECTED_PLANE],
        "chi_p5": chi[CONTROL_PLANE],
        "d_t": d_value,
    }


def measure_offset(
    *,
    model: Any,
    runtime: Mapping[str, Any],
    frozen: Mapping[str, Any],
    input_ids: torch.Tensor,
    gold_label: str,
    branch_tokens: Mapping[str, int],
    relative_offset: int,
    generated_prefix_length: int,
    cast_tol: float,
) -> dict[str, Any]:
    require(relative_offset in OBSERVATION_OFFSETS, "MEASURE_OFFSET")
    require(
        PREFIX_LENGTH_TO_OFFSET[generated_prefix_length]
        == relative_offset,
        "MEASURE_PREFIX_OFFSET",
    )

    probes: list[dict[str, Any]] = []
    for plane in PLANES:
        for basis_name in BASIS_NAMES:
            for epsilon_sign in EPSILON_SIGNS:
                probes.append(
                    probe_forward(
                        model=model,
                        runtime=runtime,
                        frozen=frozen,
                        input_ids=input_ids,
                        gold_label=gold_label,
                        branch_tokens=branch_tokens,
                        plane=plane,
                        basis_name=basis_name,
                        epsilon_sign=epsilon_sign,
                        cast_tol=cast_tol,
                    )
                )

    summary = summarize_offset_from_probe_values(probes=probes)
    return {
        "relative_offset": int(relative_offset),
        "generated_prefix_length": int(generated_prefix_length),
        "last_emitted_token_index_zero_based":
            int(generated_prefix_length) - 1,
        "current_next_token_is_commitment_branch":
            int(generated_prefix_length) == forced.COMMON_PREFIX_LENGTH,
        "probes": probes,
        **summary,
        "probe_forward_count": len(probes),
    }


def run_one(
    *,
    model: Any,
    device: torch.device,
    runtime: Mapping[str, Any],
    frozen: Mapping[str, Any],
    prompt_ids: torch.Tensor,
    cohort_row: Mapping[str, Any],
    token_ids: Mapping[str, Sequence[int]],
    branch_tokens: Mapping[str, int],
    cast_tol: float,
) -> dict[str, Any]:
    gold = str(cohort_row["correct_label"])
    require(gold in FORCED_CLASS_ORDER, f"ROW_GOLD:{gold}")

    prompt = prompt_ids.detach().cpu().to(torch.long).contiguous()
    generated: list[int] = []
    measurements: list[dict[str, Any]] = []
    native_forward_count = 0
    probe_forward_count = 0
    branch_choice_allowed_logits: dict[str, float] | None = None

    while True:
        complete = (
            forced.completed_label(generated, token_ids)
            if generated
            else None
        )
        if complete is not None:
            emitted = complete
            break

        require(
            len(generated) < forced.MAX_GENERATED_TOKENS,
            "GENERATION_MAX_TOKENS",
        )
        allowed = base.identity_gate.allowed_next_tokens(
            generated,
            token_ids,
        )
        require(allowed, "GENERATION_ALLOWED_EMPTY")

        combined = torch.cat(
            [
                prompt,
                torch.tensor(generated, dtype=torch.long),
            ]
        ).unsqueeze(0).to(device)

        if len(generated) in PREFIX_LENGTH_TO_OFFSET:
            measurement = measure_offset(
                model=model,
                runtime=runtime,
                frozen=frozen,
                input_ids=combined,
                gold_label=gold,
                branch_tokens=branch_tokens,
                relative_offset=PREFIX_LENGTH_TO_OFFSET[len(generated)],
                generated_prefix_length=len(generated),
                cast_tol=cast_tol,
            )
            measurements.append(measurement)
            probe_forward_count += int(measurement["probe_forward_count"])

        next_logits, observation = forced.forward_next_token(
            model=model,
            runtime=runtime,
            frozen=frozen,
            input_ids=combined,
            branch_tokens=branch_tokens,
            observe=False,
        )
        require(observation is None, "NATIVE_OBSERVATION_UNEXPECTED")
        native_forward_count += 1

        next_token, allowed_logits = base.greedy_allowed_token(
            next_logits,
            allowed,
        )

        if len(generated) == forced.COMMON_PREFIX_LENGTH:
            require(
                set(int(x) for x in allowed)
                == set(branch_tokens.values()),
                "BRANCH_ALLOWED_SET",
            )
            branch_choice_allowed_logits = {
                label: float(
                    allowed_logits[str(branch_tokens[label])]
                )
                for label in FORCED_CLASS_ORDER
            }

        generated.append(int(next_token))
        forced.compatible_labels(generated, token_ids)

    require(emitted in FORCED_CLASS_ORDER, "EMITTED_LABEL")
    require(
        generated == [int(x) for x in token_ids[emitted]],
        "EMITTED_TOKEN_SEQUENCE",
    )
    require(
        native_forward_count == NATIVE_FORWARDS_PER_ROW,
        f"NATIVE_FORWARD_COUNT:{native_forward_count}",
    )
    require(
        probe_forward_count == PROBES_PER_ROW,
        f"PROBE_FORWARD_COUNT:{probe_forward_count}",
    )
    require(
        [int(x["relative_offset"]) for x in measurements]
        == list(OBSERVATION_OFFSETS),
        "MEASUREMENT_OFFSET_COVERAGE",
    )
    require(
        [int(x["generated_prefix_length"]) for x in measurements]
        == list(OBSERVATION_PREFIX_LENGTHS),
        "MEASUREMENT_PREFIX_COVERAGE",
    )
    require(
        branch_choice_allowed_logits is not None,
        "BRANCH_CHOICE_LOGITS",
    )

    z_i = sum(float(x["d_t"]) for x in measurements) / 4.0
    require(math.isfinite(z_i), "Z_NONFINITE")
    supported = emitted == gold
    total_forwards = native_forward_count + probe_forward_count
    require(
        total_forwards == TOTAL_FORWARDS_PER_ROW,
        f"ROW_FORWARD_COUNT:{total_forwards}",
    )

    return {
        "schema_version": ROW_SCHEMA,
        "averitec_train_index": int(cohort_row["averitec_train_index"]),
        "example_id": str(cohort_row["example_id"]),
        "source_label": str(cohort_row["source_label"]),
        "correct_label": gold,
        "correct_label_id": int(cohort_row["correct_label_id"]),
        "prompt_attended_token_count": int(prompt.numel()),
        "generated_token_ids": list(generated),
        "generated_token_count": len(generated),
        "generated_commitment_surface": forced.FORCED_SURFACES[emitted],
        "emitted_commitment_label": emitted,
        "branch_decision_token_index_zero_based":
            forced.DECISIVE_TSTAR_ZERO_BASED,
        "branch_choice_allowed_logits": branch_choice_allowed_logits,
        "supported_forced_decisive_commitment": bool(supported),
        "unsupported_forced_decisive_commitment": bool(not supported),
        "primary_stage_a_group":
            "supported" if supported else "unsupported",
        "outcome_category":
            "supported_forced_decisive"
            if supported
            else "unsupported_forced_decisive",
        "dynamic_causal_susceptibility_measurements": measurements,
        "z_i": z_i,
        "native_generation_forward_count": native_forward_count,
        "susceptibility_probe_forward_count": probe_forward_count,
        "scientific_full_model_forward_count": total_forwards,
        "generate_api_call_count": 0,
        "manual_constrained_decoding": True,
        "forced_decisive_two_class_grammar": True,
        "not_entitled_available": False,
        "sampling": False,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "beam_search": False,
        "repetition_penalty": None,
        "logit_bias": None,
        "primary_inference_executed": False,
        "p_value_count_added": 0,
    }


def shard_worker(
    *,
    shard: Mapping[str, int],
    snapshot: str,
    temp_output: str,
) -> None:
    try:
        physical_device = int(shard["physical_device"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        snapshot_path = Path(snapshot)

        cohort, cohort_manifest = validate_frozen_cohort()
        _model_rows, encoded, tokenizer_provenance = build_input(
            snapshot=snapshot_path,
            cohort=cohort,
            cohort_manifest=cohort_manifest,
        )

        gate_artifact = base.load_gate_artifact()
        grammar_manifest = forced.forced_grammar_manifest(gate_artifact)
        token_ids = forced.forced_token_ids(grammar_manifest)
        branch_tokens = forced.branch_token_by_label(token_ids)

        model, device, runtime, frozen, model_identity = (
            base.load_generation_model(
                snapshot=snapshot_path,
                physical_device=physical_device,
            )
        )

        require(
            int(runtime["strong_mask"].detach().cpu().sum().item()) == DIM,
            "RUNTIME_STRONG_DIM",
        )
        require(
            SELECTED_PLANE in frozen["planes"]
            and CONTROL_PLANE in frozen["planes"],
            "RUNTIME_PLANES",
        )

        spec = steering.bridge.scale_spec("mamba370m")
        cast_tol = float(
            spec["confirmation"].transport_runtime.RUNTIME_CAST_TOL
        )
        require(
            math.isfinite(cast_tol) and cast_tol > 0.0,
            "RUNTIME_CAST_TOL",
        )

        rows: list[dict[str, Any]] = []
        for index in range(
            int(shard["start"]),
            int(shard["end"]),
        ):
            cohort_row = cohort[index]
            prompt = extract_prompt_ids(
                encoded,
                index=index,
                expected_attended=int(
                    cohort_row["serialized_attended_length"]
                ),
            )
            rows.append(
                run_one(
                    model=model,
                    device=device,
                    runtime=runtime,
                    frozen=frozen,
                    prompt_ids=prompt,
                    cohort_row=cohort_row,
                    token_ids=token_ids,
                    branch_tokens=branch_tokens,
                    cast_tol=cast_tol,
                )
            )

        torch.cuda.synchronize(device)
        require(
            len(rows) == int(shard["example_count"]),
            "SHARD_ROW_COUNT",
        )
        require(
            sum(
                int(row["scientific_full_model_forward_count"])
                for row in rows
            )
            == int(shard["forward_budget"]),
            "SHARD_FORWARD_COUNT",
        )

        payload = {
            "schema_version": SHARD_SCHEMA,
            "shard": dict(shard),
            "model_identity": dict(model_identity),
            "tokenizer": dict(tokenizer_provenance),
            "forced_grammar_sha256":
                grammar_manifest["grammar_sha256"],
            "runtime_cast_tolerance": cast_tol,
            "rows": rows,
        }
        Path(temp_output).write_bytes(canonical_json_bytes(payload))

    except BaseException:
        traceback.print_exc()
        raise


def launch_shards(
    *,
    snapshot: Path,
    temp_dir: Path,
) -> list[Path]:
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    outputs: list[Path] = []

    for shard in SHARDS:
        output = temp_dir / f"shard{shard['shard_id']}.json"
        outputs.append(output)
        process = ctx.Process(
            target=shard_worker,
            kwargs={
                "shard": shard,
                "snapshot": str(snapshot),
                "temp_output": str(output),
            },
            name=f"precursor-v2-dcs-stage-a-shard-{shard['shard_id']}",
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        require(
            process.exitcode == 0,
            f"SHARD_EXIT:{process.name}:{process.exitcode}",
        )

    require(
        all(path.is_file() for path in outputs),
        "SHARD_OUTPUT_MISSING",
    )
    return outputs


def merge_shards(
    *,
    shard_paths: Sequence[Path],
    cohort: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(len(shard_paths) == SHARD_COUNT, "MERGE_SHARD_COUNT")

    rows: list[dict[str, Any]] = []
    shard_meta: list[dict[str, Any]] = []
    model_identity_ref: Mapping[str, Any] | None = None
    tokenizer_ref: Mapping[str, Any] | None = None

    for shard_id, path in enumerate(shard_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload["schema_version"] == SHARD_SCHEMA, "MERGE_SCHEMA")
        require(
            int(payload["shard"]["shard_id"]) == shard_id,
            "MERGE_SHARD_ID",
        )
        require(payload["shard"] == SHARDS[shard_id], "MERGE_SHARD_SPEC")
        require(
            payload["forced_grammar_sha256"]
            == forced.FORCED_GRAMMAR_SHA256,
            "MERGE_GRAMMAR_SHA",
        )

        identity = payload["model_identity"]
        tokenizer = payload["tokenizer"]
        if model_identity_ref is None:
            model_identity_ref = identity
            tokenizer_ref = tokenizer
        else:
            for key in (
                "backbone_canonical_sha256",
                "lm_head_weight_sha256",
                "input_embedding_weight_sha256",
            ):
                require(
                    identity[key] == model_identity_ref[key],
                    f"MERGE_MODEL_IDENTITY:{key}",
                )
            require(
                tokenizer == tokenizer_ref,
                "MERGE_TOKENIZER_IDENTITY",
            )

        shard_rows = payload["rows"]
        require(
            len(shard_rows) == int(SHARDS[shard_id]["example_count"]),
            "MERGE_SHARD_ROWS",
        )
        rows.extend(shard_rows)
        shard_meta.append({
            "shard": dict(payload["shard"]),
            "model_identity": dict(identity),
            "tokenizer": dict(tokenizer),
            "runtime_cast_tolerance":
                float(payload["runtime_cast_tolerance"]),
            "row_count": len(shard_rows),
        })

    require(len(rows) == N, "MERGE_N")
    expected_ids = [str(row["example_id"]) for row in cohort]
    observed_ids = [str(row["example_id"]) for row in rows]
    require(observed_ids == expected_ids, "MERGE_ROW_ORDER")
    require(len(set(observed_ids)) == N, "MERGE_ROW_UNIQUENESS")
    return rows, shard_meta


def validate_raw_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(len(rows) == N, "RAW_N")
    group_counts = Counter(
        str(row["primary_stage_a_group"])
        for row in rows
    )
    emitted_counts = Counter(
        str(row["emitted_commitment_label"])
        for row in rows
    )

    for row in rows:
        require(row["schema_version"] == ROW_SCHEMA, "RAW_SCHEMA")
        require(
            row["emitted_commitment_label"] in FORCED_CLASS_ORDER,
            "RAW_EMITTED",
        )
        require(
            row["primary_stage_a_group"]
            in ("supported", "unsupported"),
            "RAW_GROUP",
        )
        require(
            bool(row["supported_forced_decisive_commitment"])
            != bool(row["unsupported_forced_decisive_commitment"]),
            "RAW_OUTCOME_XOR",
        )
        require(
            len(row["dynamic_causal_susceptibility_measurements"]) == 4,
            "RAW_MEASUREMENT_COUNT",
        )
        require(
            [
                int(x["relative_offset"])
                for x in row[
                    "dynamic_causal_susceptibility_measurements"
                ]
            ]
            == list(OBSERVATION_OFFSETS),
            "RAW_OFFSET_COVERAGE",
        )
        require(math.isfinite(float(row["z_i"])), "RAW_Z_FINITE")
        require(
            int(row["native_generation_forward_count"])
            == NATIVE_FORWARDS_PER_ROW,
            "RAW_NATIVE_FORWARDS",
        )
        require(
            int(row["susceptibility_probe_forward_count"])
            == PROBES_PER_ROW,
            "RAW_PROBE_FORWARDS",
        )
        require(
            int(row["scientific_full_model_forward_count"])
            == TOTAL_FORWARDS_PER_ROW,
            "RAW_TOTAL_FORWARDS",
        )
        require(row["generate_api_call_count"] == 0, "RAW_GENERATE_API")
        require(
            row["manual_constrained_decoding"] is True,
            "RAW_MANUAL_DECODE",
        )
        require(
            row["primary_inference_executed"] is False,
            "RAW_INFERENCE",
        )
        require(row["p_value_count_added"] == 0, "RAW_P_VALUE")

    total_forwards = sum(
        int(row["scientific_full_model_forward_count"])
        for row in rows
    )
    require(
        total_forwards == SCIENTIFIC_FORWARD_BUDGET,
        f"RAW_FORWARD_BUDGET:{total_forwards}",
    )
    return {
        "supported_unsupported_group_counts":
            dict(sorted(group_counts.items())),
        "emitted_commitment_counts":
            dict(sorted(emitted_counts.items())),
        "scientific_full_model_forward_count_this_run":
            total_forwards,
        "native_generation_forward_count_this_run":
            N * NATIVE_FORWARDS_PER_ROW,
        "susceptibility_probe_forward_count_this_run":
            N * PROBES_PER_ROW,
    }


def write_output(
    *,
    output_dir: Path,
    expected_head: str,
    rows: Sequence[Mapping[str, Any]],
    shard_meta: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(
        not output_dir.exists(),
        f"OUTPUT_COLLISION:{output_dir}",
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_summary = validate_raw_rows(rows)
    row_raw = jsonl_bytes(rows)
    row_sha = sha256_bytes(row_raw)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "experiment":
            "PRECURSOR_V2_DYNAMIC_CAUSAL_SUSCEPTIBILITY_STAGE_A",
        "model_scale": "mamba370m",
        "hf_repo": base.geom.HF_REPO,
        "hf_revision": base.geom.HF_REVISION,
        "selected_plane": SELECTED_PLANE,
        "response_blind_control_plane": CONTROL_PLANE,
        "intervention_block": EARLY_BLOCK,
        "readout_block": LATE_BLOCK,
        "epsilon": EPSILON,
        "observation_offsets": list(OBSERVATION_OFFSETS),
        "observation_generated_prefix_lengths":
            list(OBSERVATION_PREFIX_LENGTHS),
        "endpoint":
            "Z_i=mean(D_t over t*-4,t*-3,t*-2,t*-1)",
        "forward_accounting": {
            "probes_per_offset": PROBES_PER_OFFSET,
            "probes_per_row": PROBES_PER_ROW,
            "native_forwards_per_row": NATIVE_FORWARDS_PER_ROW,
            "total_forwards_per_row": TOTAL_FORWARDS_PER_ROW,
            "cohort_count": N,
            "scientific_forward_budget":
                SCIENTIFIC_FORWARD_BUDGET,
        },
        "cohort": {
            "path": COHORT_FILE.as_posix(),
            "git_blob": COHORT_GIT_BLOB,
            "sha256": COHORT_SHA256,
            "count": N,
            "source_label_counts": SOURCE_LABEL_COUNTS,
            "correct_label_counts": CORRECT_LABEL_COUNTS,
        },
        "design": {
            "path": DESIGN_ARTIFACT.as_posix(),
            "git_blob": DESIGN_GIT_BLOB,
        },
        "forward_accounting_correction": {
            "path": FORWARD_CORRECTION.as_posix(),
            "git_blob": FORWARD_CORRECTION_GIT_BLOB,
        },
        "dependencies": {
            "cohort_builder_git_blob": COHORT_BUILDER_GIT_BLOB,
            "steering_runner_git_blob": STEERING_RUNNER_GIT_BLOB,
            "forced_runner_git_blob": FORCED_RUNNER_GIT_BLOB,
            "base_generator_git_blob": BASE_GENERATOR_GIT_BLOB,
        },
        "forced_grammar_sha256": forced.FORCED_GRAMMAR_SHA256,
        "row_file": ROW_FILE,
        "row_file_sha256": row_sha,
        "shards": [dict(x) for x in shard_meta],
        **raw_summary,
        "scientific_inference_executed": False,
        "primary_test_executed": False,
        "p_value_count_added": 0,
        "training_executed": False,
        "backward_executed": False,
        "generation_api_used": False,
    }

    summary_raw = pretty_json_bytes(summary)
    (output_dir / ROW_FILE).write_bytes(row_raw)
    (output_dir / SUMMARY_FILE).write_bytes(summary_raw)

    hashes = {
        ROW_FILE: row_sha,
        SUMMARY_FILE: sha256_bytes(summary_raw),
    }
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def execute(
    *,
    expected_head: str,
    snapshot: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    validate_protocol()
    cohort, _manifest = validate_frozen_cohort()

    require(snapshot.is_dir(), f"SNAPSHOT_DIR:{snapshot}")
    with tempfile.TemporaryDirectory(
        prefix="precursor-v2-dcs-stage-a-"
    ) as tmp:
        shard_paths = launch_shards(
            snapshot=snapshot,
            temp_dir=Path(tmp),
        )
        rows, shard_meta = merge_shards(
            shard_paths=shard_paths,
            cohort=cohort,
        )

    return write_output(
        output_dir=output_dir,
        expected_head=expected_head,
        rows=rows,
        shard_meta=shard_meta,
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Precursor-v2 Dynamic Causal Susceptibility "
            "Stage A raw CUDA workload. Produces raw per-row susceptibility "
            "and future forced-decisive outcome artifacts only. It performs "
            "no primary statistical inference and adds no p-values."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = execute(
        expected_head=str(args.expected_head),
        snapshot=args.snapshot,
        output_dir=args.output_dir,
    )
    print("RESULT=" + str(summary["result"]))
    print("COHORT_COUNT=" + str(N))
    print("PROBES_PER_OFFSET=" + str(PROBES_PER_OFFSET))
    print("PROBES_PER_ROW=" + str(PROBES_PER_ROW))
    print("NATIVE_FORWARDS_PER_ROW=" + str(NATIVE_FORWARDS_PER_ROW))
    print("TOTAL_FORWARDS_PER_ROW=" + str(TOTAL_FORWARDS_PER_ROW))
    print(
        "SCIENTIFIC_FORWARD_BUDGET="
        + str(SCIENTIFIC_FORWARD_BUDGET)
    )
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("P_VALUE_COUNT_ADDED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
