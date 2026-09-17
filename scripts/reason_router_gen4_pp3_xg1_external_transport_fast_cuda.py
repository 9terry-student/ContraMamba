from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import (
    reason_router_gen4_family_subspace_sensitivity_fast_cuda as prior,
)
from scripts import (
    reason_router_gen4_xg1_fast_cuda_one_pair_equivalence as xg1_eq,
)
from scripts import (
    reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda as holdout,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"
SCOPE_FREEZE_COMMIT = "02e4c897d65f8f6e90b855054793594866545bf2"
PREPARATION_FREEZE_COMMIT = "30a1dcbb1be7dc9b3a834b0b539b5d29016e55ed"

SCOPE_PATH = "reports/reason_router_gen4_pp3_xg1_external_transport_scope.md"
SCOPE_BLOB = "7f5d297f3f02225d7935c372db9c36caa6ad60ab"

PREPARATION_ROOT = Path(
    "reports/reason_router_gen4_pp3_xg1_external_transport_preparation_02e4c89"
)
PREPARATION_MANIFEST_FILE = "preparation_manifest.json"
PREPARATION_CHECKSUM_FILE = "SHA256SUMS.txt"
PP3_PLUS_FILE = "pp3_plus.f64le"
PP3_MINUS_FILE = "pp3_minus.f64le"

PREPARATION_MANIFEST_SHA256 = (
    "f44c7c0596fb2019556a496a247929a9b18f4f1a05e5eb59fc83eca6d33fc342"
)
PP3_PLUS_SHA256 = (
    "66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff"
)
PP3_MINUS_SHA256 = (
    "ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7"
)

FROZEN_GIT_BLOBS = {
    SCOPE_PATH: SCOPE_BLOB,
    (
        PREPARATION_ROOT / PREPARATION_CHECKSUM_FILE
    ).as_posix(): "2900a94aa5f44c95f4088553d1229e2063046ca5",
    (
        PREPARATION_ROOT / PP3_PLUS_FILE
    ).as_posix(): "703bda8006f5881fdaf6d7c0dbf47852fb9dff76",
    (
        PREPARATION_ROOT / PP3_MINUS_FILE
    ).as_posix(): "c551f71a10a4e8280401090564699746f9b11933",
    (
        PREPARATION_ROOT / PREPARATION_MANIFEST_FILE
    ).as_posix(): "70174914c3ec0610f77f7ef9566d3db144e412ac",
    (
        "scripts/reason_router_gen4_family_subspace_sensitivity_fast_cuda.py"
    ): "03f3bf1482913bce30bfdb665ab223a67e6e4159",
    (
        "scripts/reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda.py"
    ): "2d35e5ed936fd37f4ecfc063e060290304c6bf10",
    (
        "scripts/reason_router_gen4_xg1_fast_cuda_one_pair_equivalence.py"
    ): "3a1b71ca32a397347e9fe352d99802bd80cd7760",
    (
        "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py"
    ): "6c98ce022ca134e385db28851fd364dc6daff423",
    (
        "scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py"
    ): "00a81ce6ec4ada4d5c0bf36418347b222543d174",
}

RUNTIME_REUSED_PATHS = tuple(
    dict.fromkeys(
        (
            "scripts/reason_router_gen4_xg1_fast_cuda_one_pair_equivalence.py",
            "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py",
            "scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py",
            "scripts/reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda.py",
            "scripts/reason_router_gen4_family_subspace_sensitivity_fast_cuda.py",
            *xg1_eq.BACKEND_FROZEN_PATHS,
            *prior.REUSED_PATHS,
        )
    )
)

SOURCE_PAIR_COUNT = 300
AMBIENT_DIM = 395
EPSILON = 0.025
S3 = 0.98692852916688512
VECTOR_NORM_TOL = 1.0e-12
VECTOR_DOT_TOL = 1.0e-12

FORWARDS_PER_SIGNED_PROBE = 2
SIGNED_PROBES_PER_DIRECTION = 2
FORWARDS_PER_DIRECTION = (
    FORWARDS_PER_SIGNED_PROBE * SIGNED_PROBES_PER_DIRECTION
)
DIRECTIONS_PER_PAIR = 2
FORWARDS_PER_PAIR = FORWARDS_PER_DIRECTION * DIRECTIONS_PER_PAIR
SCIENTIFIC_FORWARD_BUDGET = SOURCE_PAIR_COUNT * FORWARDS_PER_PAIR
BASELINE_FORWARD_BUDGET_THIS_RUN = 0

PROBE_SEED_SCHEMA = "gen4-pp3-xg1-external-transport-probe-seed-v1"
DIRECTION_PROBE_SCHEMA = "gen4-pp3-xg1-external-transport-direction-probe-v1"
ITEM_SCHEMA = "gen4-pp3-xg1-external-transport-item-v1"
SUMMARY_SCHEMA = "gen4-pp3-xg1-external-transport-summary-v1"
MANIFEST_SCHEMA = "gen4-pp3-xg1-external-transport-manifest-v1"
RESULT_PASS = "PASS_PP3_XG1_EXTERNAL_TRANSPORT_OBSERVATION"

ITEM_FILE = "pp3_xg1_external_transport_items.jsonl"
SUMMARY_FILE = "pp3_xg1_external_transport_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"


class PP3XG1ExternalTransportError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PP3XG1ExternalTransportError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PP3XG1ExternalTransportError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def _paths_unchanged(base: str, head: str, paths: Sequence[str]) -> bool:
    return (
        subprocess.call(
            ["git", "diff", "--quiet", base, head, "--", *paths],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for ancestor, label in (
        (SCOPE_FREEZE_COMMIT, "SCOPE_FREEZE"),
        (PREPARATION_FREEZE_COMMIT, "PREPARATION_FREEZE"),
    ):
        require(
            _git_is_ancestor(ancestor, expected_head),
            f"{label}_NOT_ANCESTOR",
        )

    for path, expected_blob in FROZEN_GIT_BLOBS.items():
        observed_blob = git("rev-parse", f"HEAD:{path}")
        require(
            observed_blob == expected_blob,
            f"FROZEN_GIT_BLOB_DRIFT:{path}:{observed_blob}",
        )

    require(
        _paths_unchanged(
            PREPARATION_FREEZE_COMMIT,
            expected_head,
            RUNTIME_REUSED_PATHS,
        ),
        "RUNTIME_REUSED_PATH_DRIFT",
    )


def _expected_pairs() -> tuple[str, ...]:
    return tuple(
        f"xg1_fact_{index:03d}"
        for index in range(1, SOURCE_PAIR_COUNT + 1)
    )


def _read_checksum_file(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in rows, f"CHECKSUM_DUPLICATE:{name}")
        rows[name] = digest
    return rows


def _vector_from_raw(raw: bytes, label: str) -> torch.Tensor:
    require(
        len(raw) == AMBIENT_DIM * 8,
        f"VECTOR_BYTE_LENGTH:{label}:{len(raw)}",
    )
    values = struct.unpack(f"<{AMBIENT_DIM}d", raw)
    vector = torch.tensor(values, dtype=torch.float64).contiguous()
    require(
        tuple(vector.shape) == (AMBIENT_DIM,),
        f"VECTOR_SHAPE:{label}:{tuple(vector.shape)}",
    )
    require(
        bool(torch.isfinite(vector).all().item()),
        f"VECTOR_NONFINITE:{label}",
    )
    return vector


def load_pp3_vectors(root: Path = ROOT) -> dict[str, Any]:
    prep = root / PREPARATION_ROOT
    manifest_path = prep / PREPARATION_MANIFEST_FILE
    checksum_path = prep / PREPARATION_CHECKSUM_FILE
    plus_path = prep / PP3_PLUS_FILE
    minus_path = prep / PP3_MINUS_FILE

    for path in (manifest_path, checksum_path, plus_path, minus_path):
        require(path.is_file(), f"PREPARATION_FILE_MISSING:{path}")

    require(
        sha256_file(manifest_path) == PREPARATION_MANIFEST_SHA256,
        "PREPARATION_MANIFEST_SHA256",
    )
    require(sha256_file(plus_path) == PP3_PLUS_SHA256, "PP3_PLUS_SHA256")
    require(sha256_file(minus_path) == PP3_MINUS_SHA256, "PP3_MINUS_SHA256")

    checksums = _read_checksum_file(checksum_path)
    require(
        checksums
        == {
            PP3_MINUS_FILE: PP3_MINUS_SHA256,
            PP3_PLUS_FILE: PP3_PLUS_SHA256,
            PREPARATION_MANIFEST_FILE: PREPARATION_MANIFEST_SHA256,
        },
        "PREPARATION_CHECKSUM_CONTENT",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    require(
        manifest.get("schema_version")
        == "GEN4_PP3_XG1_EXTERNAL_TRANSPORT_PREPARATION_V1",
        "PREPARATION_SCHEMA",
    )
    require(
        manifest.get("result") == "PASS_PP3_XG1_OUTCOME_BLIND_PREPARATION",
        "PREPARATION_RESULT",
    )
    require(
        manifest.get("branch") == EXPECTED_BRANCH,
        "PREPARATION_BRANCH",
    )
    require(
        manifest.get("scope_freeze_commit") == SCOPE_FREEZE_COMMIT,
        "PREPARATION_SCOPE_FREEZE",
    )
    require(manifest.get("scope_git_blob") == SCOPE_BLOB, "PREPARATION_SCOPE_BLOB")

    xg1 = manifest.get("xg1")
    require(isinstance(xg1, dict), "PREPARATION_XG1")
    require(xg1.get("source_pair_count") == SOURCE_PAIR_COUNT, "PREPARATION_XG1_COUNT")
    require(xg1.get("pair_id_first") == "xg1_fact_001", "PREPARATION_XG1_FIRST")
    require(xg1.get("pair_id_last") == "xg1_fact_300", "PREPARATION_XG1_LAST")
    require(
        xg1.get("source_facts_sha256")
        == xg1_eq.eligibility.EXPECTED_SOURCE_FACTS_SHA256,
        "PREPARATION_XG1_SOURCE_SHA256",
    )
    require(
        xg1.get("rows_sha256") == xg1_eq.eligibility.EXPECTED_ROWS_SHA256,
        "PREPARATION_XG1_ROWS_SHA256",
    )
    require(
        xg1.get("structural_manifest_sha256")
        == xg1_eq.eligibility.EXPECTED_STRUCTURAL_MANIFEST_SHA256,
        "PREPARATION_XG1_STRUCTURAL_SHA256",
    )
    require(
        xg1.get("anchor_manifest_sha256")
        == xg1_eq.ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
        "PREPARATION_XG1_ANCHOR_SHA256",
    )
    require(
        xg1.get("eligibility_summary_sha256")
        == xg1_eq.ELIGIBILITY_SUMMARY_SHA256,
        "PREPARATION_XG1_ELIGIBILITY_SHA256",
    )

    science = manifest.get("scientific_execution")
    require(isinstance(science, dict), "PREPARATION_SCIENCE_BOUNDARY")
    require(science.get("model_forward_count") == 0, "PREPARATION_MODEL_FORWARD")
    require(science.get("baseline_model_forward_count") == 0, "PREPARATION_BASELINE_FORWARD")
    require(science.get("checkpoint_load_count") == 0, "PREPARATION_CHECKPOINT_LOAD")
    require(science.get("cuda_executed") is False, "PREPARATION_CUDA")
    require(science.get("training_executed") is False, "PREPARATION_TRAINING")
    require(science.get("backward_executed") is False, "PREPARATION_BACKWARD")
    require(
        science.get("scientific_outcomes_observed") is False,
        "PREPARATION_OUTCOME_BOUNDARY",
    )
    require(
        science.get("primary_inference_executed") is False,
        "PREPARATION_INFERENCE_BOUNDARY",
    )

    pp3 = manifest.get("pp3")
    require(isinstance(pp3, dict), "PREPARATION_PP3")
    require(pp3.get("ambient_dim") == AMBIENT_DIM, "PP3_AMBIENT_DIM")
    require(pp3.get("scientific_principal_pair_number") == 3, "PP3_NUMBER")
    require(pp3.get("zero_based_principal_pair_index") == 2, "PP3_ZERO_INDEX")
    require(
        pp3.get("serialization") == "raw little-endian IEEE754 float64, 395 scalars",
        "PP3_SERIALIZATION",
    )
    require(float(pp3["s3"]) == S3, "PP3_S3")
    require(pp3.get("plus_file") == PP3_PLUS_FILE, "PP3_PLUS_FILE")
    require(pp3.get("minus_file") == PP3_MINUS_FILE, "PP3_MINUS_FILE")
    require(pp3.get("plus_sha256") == PP3_PLUS_SHA256, "PP3_PLUS_MANIFEST_SHA256")
    require(pp3.get("minus_sha256") == PP3_MINUS_SHA256, "PP3_MINUS_MANIFEST_SHA256")

    plus = _vector_from_raw(plus_path.read_bytes(), "pp3_plus")
    minus = _vector_from_raw(minus_path.read_bytes(), "pp3_minus")

    plus_norm = float(torch.linalg.vector_norm(plus, ord=2).item())
    minus_norm = float(torch.linalg.vector_norm(minus, ord=2).item())
    dot = float(torch.dot(plus, minus).item())
    require(
        math.isfinite(plus_norm) and abs(plus_norm - 1.0) <= VECTOR_NORM_TOL,
        f"PP3_PLUS_NORM:{plus_norm}",
    )
    require(
        math.isfinite(minus_norm) and abs(minus_norm - 1.0) <= VECTOR_NORM_TOL,
        f"PP3_MINUS_NORM:{minus_norm}",
    )
    require(
        math.isfinite(dot) and abs(dot) <= VECTOR_DOT_TOL,
        f"PP3_ORTHOGONALITY:{dot}",
    )

    plus_pivot = int(torch.argmax(torch.abs(plus)).item())
    minus_pivot = int(torch.argmax(torch.abs(minus)).item())
    require(plus_pivot == 267, f"PP3_PLUS_PIVOT:{plus_pivot}")
    require(minus_pivot == 23, f"PP3_MINUS_PIVOT:{minus_pivot}")
    require(float(plus[plus_pivot].item()) > 0.0, "PP3_PLUS_SIGN")
    require(float(minus[minus_pivot].item()) > 0.0, "PP3_MINUS_SIGN")

    return {
        "pp3_plus": plus,
        "pp3_minus": minus,
        "plus_norm": plus_norm,
        "minus_norm": minus_norm,
        "plus_minus_dot": dot,
        "manifest": manifest,
    }


def _probe_seed(
    index: int,
    pair: str,
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    require(pair == _expected_pairs()[index], f"PROBE_SEED_PAIR:{index}:{pair}")
    anchors = holdout.phase1._anchors_for_pair(pair, events)
    return {
        "schema_version": PROBE_SEED_SCHEMA,
        "family_key": "xg1",
        "source_pair_id": pair,
        "pair_index": index,
        "pair_ordinal": index + 1,
        "target_plus_anchor": int(anchors["tp"]),
        "target_minus_anchor": int(anchors["tm"]),
        "reference_plus_anchor": int(anchors["rp"]),
        "reference_minus_anchor": int(anchors["rm"]),
    }


def _input_row(
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    pair: str,
    cell: str,
) -> torch.Tensor:
    return holdout.phase2._input_row(encoded, row_index, pair, cell)


def _run_signed_probe(
    seed: Mapping[str, Any],
    unit_direction: torch.Tensor,
    *,
    orientation: int,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    budget: Any,
) -> dict[str, Any]:
    require(seed.get("family_key") == "xg1", "PROBE_FAMILY")
    require(orientation in {-1, 1}, f"BAD_ORIENTATION:{orientation}")

    pair = str(seed["source_pair_id"])
    direction = (
        unit_direction.detach().cpu().to(torch.float64).contiguous().clone()
    )
    require(tuple(direction.shape) == (AMBIENT_DIM,), "PROBE_DIRECTION_SHAPE")
    direction_norm = float(torch.linalg.vector_norm(direction, ord=2).item())
    require(
        math.isfinite(direction_norm)
        and abs(direction_norm - 1.0) <= VECTOR_NORM_TOL,
        f"PROBE_DIRECTION_NORM:{pair}:{direction_norm}",
    )

    delta_h = (
        direction.mul(float(orientation) * 2.0 * EPSILON)
        .contiguous()
        .clone()
    )
    require(bool(torch.isfinite(delta_h).all().item()), f"NONFINITE_PROBE_PLAN:{pair}")

    runtime = holdout.phase1.base.prevalence_eq
    parent = runtime.parent
    transport_runtime = runtime.transport_runtime
    core = runtime.core
    cells = holdout.phase1._cells()
    anchors = holdout.phase1._anchors_for_pair(pair, events)

    for role, field in (
        ("tp", "target_plus_anchor"),
        ("tm", "target_minus_anchor"),
        ("rp", "reference_plus_anchor"),
        ("rm", "reference_minus_anchor"),
    ):
        require(
            anchors[role] == int(seed[field]),
            f"FROZEN_ANCHOR_IDENTITY:{pair}:{role}",
        )

    captured: dict[str, Any] = {}
    for role, plus_branch in (("tp", True), ("tm", False)):
        captured[role] = parent.capture_branch(
            model,
            runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            input_ids=_input_row(
                encoded,
                row_index,
                pair,
                cells[role],
            ),
            anchor=anchors[role],
            budget=budget,
            capture_states=True,
            delta_h=delta_h,
            plus_branch=plus_branch,
        )

    audit = transport_runtime.paired_intervention_audit(
        captured["tp"]["intervention_audit"],
        captured["tm"]["intervention_audit"],
        delta_h,
        plus_expected_token_index=anchors["tp"] + core.TARGET_OFFSET,
        minus_expected_token_index=anchors["tm"] + core.TARGET_OFFSET,
    )

    plus_pe = float(parent.path_efficiency(captured["tp"]))
    minus_pe = float(parent.path_efficiency(captured["tm"]))
    response = plus_pe - minus_pe

    values = (
        plus_pe,
        minus_pe,
        response,
        float(audit["midpoint_max_abs_residual"]),
        float(audit["pair_delta_max_abs_residual"]),
        float(audit["applied_correction_max_abs_residual"]),
        float(audit["runtime_correction_l2"]),
    )
    require(
        all(math.isfinite(value) for value in values),
        f"NONFINITE_SIGNED_PROBE:{pair}:{orientation}",
    )

    intended_delta_l2 = 2.0 * EPSILON
    require(
        abs(float(audit["runtime_correction_l2"]) - intended_delta_l2)
        <= 1.0e-12,
        f"RUNTIME_CORRECTION_L2_MISMATCH:{pair}:{orientation}",
    )

    return {
        "orientation": int(orientation),
        "delta_h_l2": intended_delta_l2,
        "plus_path_efficiency": plus_pe,
        "minus_path_efficiency": minus_pe,
        "F": response,
        "midpoint_max_abs_residual": float(audit["midpoint_max_abs_residual"]),
        "pair_delta_max_abs_residual": float(audit["pair_delta_max_abs_residual"]),
        "applied_correction_max_abs_residual": float(
            audit["applied_correction_max_abs_residual"]
        ),
        "runtime_correction_l2": float(audit["runtime_correction_l2"]),
        "model_forward_count": FORWARDS_PER_SIGNED_PROBE,
    }


def _run_pp3_direction_j(
    seed: Mapping[str, Any],
    unit_direction: torch.Tensor,
    *,
    direction_key: str,
    vector_sha256: str,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    budget: Any,
) -> dict[str, Any]:
    require(direction_key in {"pp3_plus", "pp3_minus"}, f"BAD_DIRECTION:{direction_key}")
    expected_sha = PP3_PLUS_SHA256 if direction_key == "pp3_plus" else PP3_MINUS_SHA256
    require(vector_sha256 == expected_sha, f"DIRECTION_SHA256:{direction_key}")

    direction = unit_direction.detach().cpu().to(torch.float64).contiguous().clone()
    require(tuple(direction.shape) == (AMBIENT_DIM,), f"DIRECTION_SHAPE:{direction_key}")
    norm = float(torch.linalg.vector_norm(direction, ord=2).item())
    require(
        math.isfinite(norm) and abs(norm - 1.0) <= VECTOR_NORM_TOL,
        f"DIRECTION_NORM:{direction_key}:{norm}",
    )

    positive = _run_signed_probe(
        seed,
        direction,
        orientation=1,
        model=model,
        runtime_ctx=runtime_ctx,
        trace_code=trace_code,
        trace_line=trace_line,
        encoded=encoded,
        row_index=row_index,
        events=events,
        budget=budget,
    )
    negative = _run_signed_probe(
        seed,
        direction,
        orientation=-1,
        model=model,
        runtime_ctx=runtime_ctx,
        trace_code=trace_code,
        trace_line=trace_line,
        encoded=encoded,
        row_index=row_index,
        events=events,
        budget=budget,
    )

    f_plus = float(positive["F"])
    f_minus = float(negative["F"])
    j_value = (f_plus - f_minus) / (2.0 * EPSILON)
    j_squared = j_value * j_value
    require(
        all(math.isfinite(value) for value in (f_plus, f_minus, j_value, j_squared)),
        f"NONFINITE_DIRECTION_RESPONSE:{direction_key}",
    )

    return {
        "schema_version": DIRECTION_PROBE_SCHEMA,
        "direction_key": direction_key,
        "vector_sha256": vector_sha256,
        "epsilon": EPSILON,
        "F_plus": f_plus,
        "F_minus": f_minus,
        "J": j_value,
        "J_squared": j_squared,
        "positive_probe": positive,
        "negative_probe": negative,
        "model_forward_count": FORWARDS_PER_DIRECTION,
    }


def _run_pair(
    seed: Mapping[str, Any],
    *,
    pp3_plus: torch.Tensor,
    pp3_minus: torch.Tensor,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    budget: Any,
) -> dict[str, Any]:
    plus = _run_pp3_direction_j(
        seed,
        pp3_plus,
        direction_key="pp3_plus",
        vector_sha256=PP3_PLUS_SHA256,
        model=model,
        runtime_ctx=runtime_ctx,
        trace_code=trace_code,
        trace_line=trace_line,
        encoded=encoded,
        row_index=row_index,
        events=events,
        budget=budget,
    )
    minus = _run_pp3_direction_j(
        seed,
        pp3_minus,
        direction_key="pp3_minus",
        vector_sha256=PP3_MINUS_SHA256,
        model=model,
        runtime_ctx=runtime_ctx,
        trace_code=trace_code,
        trace_line=trace_line,
        encoded=encoded,
        row_index=row_index,
        events=events,
        budget=budget,
    )

    j_plus = float(plus["J"])
    j_minus = float(minus["J"])
    j_plus_squared = float(plus["J_squared"])
    j_minus_squared = float(minus["J_squared"])
    c_pp3 = (S3 / 5.0) * (j_plus_squared - j_minus_squared)
    require(
        all(
            math.isfinite(value)
            for value in (
                j_plus,
                j_minus,
                j_plus_squared,
                j_minus_squared,
                c_pp3,
            )
        ),
        f"NONFINITE_PAIR_ENDPOINT:{seed['source_pair_id']}",
    )

    item = dict(seed)
    item["probe_seed_schema_version"] = item["schema_version"]
    item["schema_version"] = ITEM_SCHEMA
    item["scope_freeze_commit"] = SCOPE_FREEZE_COMMIT
    item["preparation_freeze_commit"] = PREPARATION_FREEZE_COMMIT
    item["epsilon"] = EPSILON
    item["s3"] = S3
    item["pp3_plus_sha256"] = PP3_PLUS_SHA256
    item["pp3_minus_sha256"] = PP3_MINUS_SHA256
    item["direction_order"] = ["pp3_plus", "pp3_minus"]
    item["pp3_plus_probe"] = plus
    item["pp3_minus_probe"] = minus
    item["J_PP3_PLUS"] = j_plus
    item["J_PP3_PLUS_squared"] = j_plus_squared
    item["J_PP3_MINUS"] = j_minus
    item["J_PP3_MINUS_squared"] = j_minus_squared
    item["C_PP3"] = c_pp3
    item["baseline_model_forward_count_this_run"] = 0
    item["scientific_model_forward_count_this_run"] = FORWARDS_PER_PAIR
    return item


def _validate_signed_probe(probe: Mapping[str, Any], orientation: int) -> None:
    require(int(probe["orientation"]) == orientation, "SIGNED_ORIENTATION")
    require(float(probe["delta_h_l2"]) == 2.0 * EPSILON, "SIGNED_DELTA_L2")
    require(
        int(probe["model_forward_count"]) == FORWARDS_PER_SIGNED_PROBE,
        "SIGNED_FORWARD_COUNT",
    )
    values = (
        float(probe["plus_path_efficiency"]),
        float(probe["minus_path_efficiency"]),
        float(probe["F"]),
        float(probe["midpoint_max_abs_residual"]),
        float(probe["pair_delta_max_abs_residual"]),
        float(probe["applied_correction_max_abs_residual"]),
        float(probe["runtime_correction_l2"]),
    )
    require(all(math.isfinite(value) for value in values), "SIGNED_NONFINITE")
    require(
        float(probe["F"])
        == float(probe["plus_path_efficiency"])
        - float(probe["minus_path_efficiency"]),
        "SIGNED_F_IDENTITY",
    )
    require(
        abs(float(probe["runtime_correction_l2"]) - 2.0 * EPSILON) <= 1.0e-12,
        "SIGNED_RUNTIME_L2",
    )


def _validate_direction_probe(
    probe: Mapping[str, Any],
    *,
    direction_key: str,
    vector_sha256: str,
) -> None:
    require(probe.get("schema_version") == DIRECTION_PROBE_SCHEMA, "DIRECTION_SCHEMA")
    require(probe.get("direction_key") == direction_key, "DIRECTION_KEY")
    require(probe.get("vector_sha256") == vector_sha256, "DIRECTION_VECTOR_SHA256")
    require(float(probe["epsilon"]) == EPSILON, "DIRECTION_EPSILON")
    require(
        int(probe["model_forward_count"]) == FORWARDS_PER_DIRECTION,
        "DIRECTION_FORWARD_COUNT",
    )
    _validate_signed_probe(probe["positive_probe"], 1)
    _validate_signed_probe(probe["negative_probe"], -1)

    f_plus = float(probe["F_plus"])
    f_minus = float(probe["F_minus"])
    j_value = float(probe["J"])
    j_squared = float(probe["J_squared"])
    require(
        all(math.isfinite(value) for value in (f_plus, f_minus, j_value, j_squared)),
        "DIRECTION_NONFINITE",
    )
    require(
        f_plus == float(probe["positive_probe"]["F"]),
        "DIRECTION_F_PLUS_IDENTITY",
    )
    require(
        f_minus == float(probe["negative_probe"]["F"]),
        "DIRECTION_F_MINUS_IDENTITY",
    )
    require(j_value == (f_plus - f_minus) / (2.0 * EPSILON), "DIRECTION_J_IDENTITY")
    require(j_squared == j_value * j_value, "DIRECTION_J_SQUARED_IDENTITY")


def _validate_items(items: Sequence[Mapping[str, Any]]) -> None:
    require(len(items) == SOURCE_PAIR_COUNT, "ITEM_COUNT")
    expected_pairs = _expected_pairs()

    for index, (expected_pair, raw) in enumerate(
        zip(expected_pairs, items, strict=True)
    ):
        row = dict(raw)
        require(row.get("schema_version") == ITEM_SCHEMA, f"ITEM_SCHEMA:{index}")
        require(
            row.get("probe_seed_schema_version") == PROBE_SEED_SCHEMA,
            f"PROBE_SEED_SCHEMA:{index}",
        )
        require(row.get("family_key") == "xg1", f"ITEM_FAMILY:{index}")
        require(row.get("source_pair_id") == expected_pair, f"PAIR_ORDER:{index}")
        require(row.get("pair_index") == index, f"PAIR_INDEX:{index}")
        require(row.get("pair_ordinal") == index + 1, f"PAIR_ORDINAL:{index}")
        require(
            row.get("scope_freeze_commit") == SCOPE_FREEZE_COMMIT,
            f"SCOPE_FREEZE:{index}",
        )
        require(
            row.get("preparation_freeze_commit") == PREPARATION_FREEZE_COMMIT,
            f"PREPARATION_FREEZE:{index}",
        )
        require(float(row["epsilon"]) == EPSILON, f"EPSILON:{index}")
        require(float(row["s3"]) == S3, f"S3:{index}")
        require(row.get("pp3_plus_sha256") == PP3_PLUS_SHA256, f"PLUS_SHA:{index}")
        require(row.get("pp3_minus_sha256") == PP3_MINUS_SHA256, f"MINUS_SHA:{index}")
        require(
            row.get("direction_order") == ["pp3_plus", "pp3_minus"],
            f"DIRECTION_ORDER:{index}",
        )
        require(
            row.get("baseline_model_forward_count_this_run") == 0,
            f"BASELINE_FORWARD_COUNT:{index}",
        )
        require(
            row.get("scientific_model_forward_count_this_run") == FORWARDS_PER_PAIR,
            f"SCIENTIFIC_FORWARD_COUNT:{index}",
        )

        plus = row["pp3_plus_probe"]
        minus = row["pp3_minus_probe"]
        _validate_direction_probe(
            plus,
            direction_key="pp3_plus",
            vector_sha256=PP3_PLUS_SHA256,
        )
        _validate_direction_probe(
            minus,
            direction_key="pp3_minus",
            vector_sha256=PP3_MINUS_SHA256,
        )

        j_plus = float(row["J_PP3_PLUS"])
        j_minus = float(row["J_PP3_MINUS"])
        j_plus_squared = float(row["J_PP3_PLUS_squared"])
        j_minus_squared = float(row["J_PP3_MINUS_squared"])
        c_pp3 = float(row["C_PP3"])
        require(
            all(
                math.isfinite(value)
                for value in (
                    j_plus,
                    j_minus,
                    j_plus_squared,
                    j_minus_squared,
                    c_pp3,
                )
            ),
            f"ITEM_NONFINITE:{index}",
        )
        require(j_plus == float(plus["J"]), f"PLUS_J_IDENTITY:{index}")
        require(j_minus == float(minus["J"]), f"MINUS_J_IDENTITY:{index}")
        require(
            j_plus_squared == j_plus * j_plus == float(plus["J_squared"]),
            f"PLUS_J_SQUARED_IDENTITY:{index}",
        )
        require(
            j_minus_squared == j_minus * j_minus == float(minus["J_squared"]),
            f"MINUS_J_SQUARED_IDENTITY:{index}",
        )
        require(
            c_pp3 == (S3 / 5.0) * (j_plus_squared - j_minus_squared),
            f"C_PP3_IDENTITY:{index}",
        )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _write_outputs(
    output_dir: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")
    _validate_items(items)
    output_dir.mkdir(parents=True, exist_ok=False)

    payloads = {
        ITEM_FILE: _jsonl_bytes(items),
        SUMMARY_FILE: _canonical_json_bytes(summary),
    }
    hashes: dict[str, str] = {}
    for name, raw in payloads.items():
        path = output_dir / name
        path.write_bytes(raw)
        hashes[name] = sha256_bytes(raw)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "files": {
            name: {
                "sha256": digest,
                "bytes": int((output_dir / name).stat().st_size),
            }
            for name, digest in sorted(hashes.items())
        },
    }
    manifest_raw = _canonical_json_bytes(manifest)
    (output_dir / MANIFEST_FILE).write_bytes(manifest_raw)
    hashes[MANIFEST_FILE] = sha256_bytes(manifest_raw)

    checksum_raw = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(hashes.items())
    ).encode("utf-8")
    (output_dir / CHECKSUM_FILE).write_bytes(checksum_raw)
    return hashes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT_REQUIRED:{path}:{line_no}")
        rows.append(value)
    return rows


def validate_artifact(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_FILE
    checksum_path = output_dir / CHECKSUM_FILE
    require(manifest_path.is_file(), "MANIFEST_MISSING")
    require(checksum_path.is_file(), "CHECKSUM_MISSING")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    require(manifest.get("schema_version") == MANIFEST_SCHEMA, "MANIFEST_SCHEMA")
    files = manifest.get("files")
    require(isinstance(files, dict), "MANIFEST_FILES")
    require(set(files) == {ITEM_FILE, SUMMARY_FILE}, "MANIFEST_FILE_SET")

    observed_hashes: dict[str, str] = {}
    for name in (ITEM_FILE, SUMMARY_FILE):
        path = output_dir / name
        require(path.is_file(), f"ARTIFACT_MISSING:{name}")
        observed_sha = sha256_file(path)
        require(observed_sha == files[name]["sha256"], f"ARTIFACT_SHA256:{name}")
        require(
            int(path.stat().st_size) == int(files[name]["bytes"]),
            f"ARTIFACT_BYTES:{name}",
        )
        observed_hashes[name] = observed_sha
    observed_hashes[MANIFEST_FILE] = sha256_file(manifest_path)

    checksum_rows = _read_checksum_file(checksum_path)
    require(
        checksum_rows
        == {name: digest for name, digest in sorted(observed_hashes.items())},
        "CHECKSUM_CONTENT",
    )

    items = _read_jsonl(output_dir / ITEM_FILE)
    _validate_items(items)

    summary = json.loads((output_dir / SUMMARY_FILE).read_text(encoding="utf-8-sig"))
    require(summary.get("schema_version") == SUMMARY_SCHEMA, "SUMMARY_SCHEMA")
    require(summary.get("result") == RESULT_PASS, "SUMMARY_RESULT")
    require(summary.get("execution_head"), "SUMMARY_EXECUTION_HEAD")
    require(summary.get("scope_freeze_commit") == SCOPE_FREEZE_COMMIT, "SUMMARY_SCOPE_FREEZE")
    require(
        summary.get("preparation_freeze_commit") == PREPARATION_FREEZE_COMMIT,
        "SUMMARY_PREPARATION_FREEZE",
    )
    require(summary.get("source_pair_count") == SOURCE_PAIR_COUNT, "SUMMARY_PAIR_COUNT")
    require(summary.get("pair_id_first") == "xg1_fact_001", "SUMMARY_FIRST_PAIR")
    require(summary.get("pair_id_last") == "xg1_fact_300", "SUMMARY_LAST_PAIR")
    require(float(summary["epsilon"]) == EPSILON, "SUMMARY_EPSILON")
    require(float(summary["s3"]) == S3, "SUMMARY_S3")
    require(summary.get("pp3_plus_sha256") == PP3_PLUS_SHA256, "SUMMARY_PLUS_SHA")
    require(summary.get("pp3_minus_sha256") == PP3_MINUS_SHA256, "SUMMARY_MINUS_SHA")
    require(
        summary.get("model_forwards_per_direction") == FORWARDS_PER_DIRECTION,
        "SUMMARY_FORWARD_DIRECTION",
    )
    require(summary.get("model_forwards_per_pair") == FORWARDS_PER_PAIR, "SUMMARY_FORWARD_PAIR")
    require(
        summary.get("scientific_model_forward_count_this_run") == SCIENTIFIC_FORWARD_BUDGET,
        "SUMMARY_SCIENTIFIC_FORWARD_COUNT",
    )
    require(
        summary.get("baseline_model_forward_count_this_run") == 0,
        "SUMMARY_BASELINE_FORWARD_COUNT",
    )
    require(summary.get("finite_value_audit") == "PASS", "SUMMARY_FINITE_AUDIT")
    require(summary.get("primary_endpoint_C_PP3_observed") is True, "SUMMARY_ENDPOINT_OBSERVED")
    require(summary.get("signed_pp3_plus_observed") is True, "SUMMARY_SIGNED_OBSERVED")
    require(
        summary.get("primary_inference_executed") is False,
        "SUMMARY_INFERENCE_BOUNDARY",
    )
    require(
        summary.get("multiplicity_correction_executed") is False,
        "SUMMARY_MULTIPLICITY_BOUNDARY",
    )
    require(
        summary.get("training_executed") is False
        and summary.get("backward_executed") is False
        and summary.get("task_heads_executed") is False
        and summary.get("logits_read") is False,
        "SUMMARY_EXECUTION_BOUNDARY",
    )
    require(summary.get("scientific_conclusion") is None, "SUMMARY_CONCLUSION_BOUNDARY")
    return {"summary": summary, "items": items, "manifest": manifest}


def run_observation(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")

    vectors = load_pp3_vectors()
    pp3_plus = vectors["pp3_plus"]
    pp3_minus = vectors["pp3_minus"]

    runtime = holdout.phase1.base.prevalence_eq
    runtime.backend.runtime_gate()

    with runtime.backend.parent_runtime_rebind():
        rows, encoded, event_rows = xg1_eq.load_xg1_inputs(tokenizer_snapshot)
        pairs = xg1_eq._pair_order(rows)
        require(pairs == _expected_pairs(), "XG1_PAIR_POPULATION")

        parent = runtime.parent
        events = parent.event_lookup(event_rows)
        parent.validate_transport_event_plan(pairs, events)
        row_index = parent.build_row_index(rows)
        trace_code, trace_line = runtime.measurement._resolve_and_validate_runtime_binding()

        kernels = runtime.kernel_compat.load_exact_fast_kernels()
        with runtime.kernel_compat.exact_transformers_kernel_loader(
            kernels
        ) as constructor_kernel_calls:
            model, checkpoint_sha = parent.load_representative_model_external(
                model_snapshot=model_snapshot,
                checkpoint_path=checkpoint_path,
            )
            require(
                checkpoint_sha == runtime.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
                "CHECKPOINT_IDENTITY",
            )
            runtime_ctx = runtime.transport_runtime.validate_runtime_components(model)

        constructor_counts = Counter(constructor_kernel_calls)
        require(
            set(constructor_counts) == {"causal-conv1d", "mamba-ssm"},
            f"TRANSFORMERS_CONSTRUCTOR_KERNEL_NAMES:{dict(constructor_counts)}",
        )
        require(
            constructor_counts["causal-conv1d"] > 0
            and constructor_counts["causal-conv1d"]
            == constructor_counts["mamba-ssm"],
            f"TRANSFORMERS_CONSTRUCTOR_KERNEL_CALL_COUNT:{dict(constructor_counts)}",
        )
        runtime.kernel_compat.validate_transformers_kernel_bindings(kernels)

        model.to(torch.device("cuda:0"))
        model.eval()
        require(
            all(parameter.device.type == "cuda" for parameter in model.mamba.parameters()),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = runtime.backend._make_fast_capture(kernels)
        original_capture = parent.capture_branch
        budget = parent.ForwardBudget(SCIENTIFIC_FORWARD_BUDGET)
        items: list[dict[str, Any]] = []
        parent.capture_branch = fast_capture
        try:
            for index, pair in enumerate(pairs):
                seed = _probe_seed(index, pair, events)
                items.append(
                    _run_pair(
                        seed,
                        pp3_plus=pp3_plus,
                        pp3_minus=pp3_minus,
                        model=model,
                        runtime_ctx=runtime_ctx,
                        trace_code=trace_code,
                        trace_line=trace_line,
                        encoded=encoded,
                        row_index=row_index,
                        events=events,
                        budget=budget,
                    )
                )
            budget.assert_exact()
            torch.cuda.synchronize()
        finally:
            parent.capture_branch = original_capture

    _validate_items(items)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "scope_freeze_commit": SCOPE_FREEZE_COMMIT,
        "preparation_freeze_commit": PREPARATION_FREEZE_COMMIT,
        "source_pair_count": SOURCE_PAIR_COUNT,
        "pair_id_first": items[0]["source_pair_id"],
        "pair_id_last": items[-1]["source_pair_id"],
        "epsilon": EPSILON,
        "s3": S3,
        "pp3_plus_sha256": PP3_PLUS_SHA256,
        "pp3_minus_sha256": PP3_MINUS_SHA256,
        "pp3_ambient_dim": AMBIENT_DIM,
        "direction_order": ["pp3_plus", "pp3_minus"],
        "model_forwards_per_direction": FORWARDS_PER_DIRECTION,
        "model_forwards_per_pair": FORWARDS_PER_PAIR,
        "scientific_model_forward_count_this_run": SCIENTIFIC_FORWARD_BUDGET,
        "baseline_model_forward_count_this_run": BASELINE_FORWARD_BUDGET_THIS_RUN,
        "primary_endpoint_definition": "C_PP3=(s3/5)*(J_PP3_PLUS^2-J_PP3_MINUS^2)",
        "primary_endpoint_C_PP3_observed": True,
        "signed_pp3_plus_observed": True,
        "finite_value_audit": "PASS",
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
        "representative_checkpoint_sha256": checkpoint_sha,
    }

    _write_outputs(output_dir, items=items, summary=summary)
    validated = validate_artifact(output_dir)
    require(validated["summary"]["result"] == RESULT_PASS, "POSTWRITE_VALIDATION_RESULT")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen PP3+/PP3- observation on the exact 300-pair XG1 external "
            "generator population. Produces raw J and C_PP3 observations only."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_observation(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT =", summary["result"])
    print("PAIR_ID_FIRST =", summary["pair_id_first"])
    print("PAIR_ID_LAST =", summary["pair_id_last"])
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT =",
        summary["scientific_model_forward_count_this_run"],
    )
    print(
        "BASELINE_MODEL_FORWARD_COUNT =",
        summary["baseline_model_forward_count_this_run"],
    )
    print("PRIMARY_INFERENCE_EXECUTED =", summary["primary_inference_executed"])
    print("SCIENTIFIC_CONCLUSION =", summary["scientific_conclusion"])


if __name__ == "__main__":
    main()
