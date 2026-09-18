from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import struct
import subprocess
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import (
    reason_router_gen4_pp3_excluded_residual_aggregate_necessity_fast_cuda
    as aggregate
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"

DESIGN_COMMIT = "d046a8e03e7522a72dfbd08cc9129b769cd5686a"
STATIC_COMMIT = "ab9a6ebbc95bec20e8682b9538365f53167f108e"
AUTHORITY_COMMIT = "ac13d93785719345fc0362298c9988bf040e694d"

AUTHORITY_PATH = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_individual_plane_"
    "necessity_localization_implementation_authority.md"
)
AUTHORITY_BLOB = "97da88df71d348e1f2fd33d6b44dbf531e4dee00"

DATA_ROOT = Path(
    "data/reason_router_gen4_xg1_residual_individual_plane_necessity_v1"
)
STATIC_ROOT = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_individual_plane_"
    "necessity_localization_static_preparation_d046a8e"
)
FROZEN_RESIDUAL_ROOT = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_aggregate_necessity_"
    "static_preparation_c4518d2"
)
PP3_ROOT = Path(
    "reports/reason_router_gen4_pp3_xg1_external_transport_preparation_02e4c89"
)

SOURCE_SHA = "18381b4d31bf6b5b5975edf29dd23ad9a4cfbece23cf8c119cda4fb2539fcc9f"
ROWS_SHA = "f3b67944c05e0198f9f33a3b544ddee344a3741ceccb00fe2a74191f177b18b5"
STRUCT_SHA = "5cacd0639ce8dbb635bc2f895b81612482219859ff4e36a0768617350fd9e51d"
ANCHOR_SHA = "26b8762065063cc266286a00b394ddfc7088688643a8a4fed5b5b9825da22789"
ELIG_SHA = "25f95dff5d8929be216ea2b9961a76136ae7506fc6b4afb35670ec95354b35c7"
GEOMETRY_SHA = "dead284a5bb8a21becc10d8542018a7de3f1cf2af6870f2cfda2990104d215ec"
PREPARATION_SHA = "d1ba40bd13382b512fa26d5f446b16581a11fe925764bfc9fd970fe7278107b0"

RESIDUAL_FILES = {
    "p1_plus": (
        FROZEN_RESIDUAL_ROOT / "p1_plus.f64le",
        "209da6bf007c0eadcd78db648835e6ee174d0290726acb61731007eff436aef1",
    ),
    "p1_minus": (
        FROZEN_RESIDUAL_ROOT / "p1_minus.f64le",
        "b6470bbec5a586f34e87d6e87f32c7f1778af7a55622d60508d68c6679c1ab26",
    ),
    "p2_plus": (
        FROZEN_RESIDUAL_ROOT / "p2_plus.f64le",
        "a0f48476f77e9e1876adc9919ba61a3c4e6ac894001f789216245d2ff3c945ea",
    ),
    "p2_minus": (
        FROZEN_RESIDUAL_ROOT / "p2_minus.f64le",
        "b48683f584ab31e31d542fc9b20327d19c0019cca02a41c97db0418d2bd69a78",
    ),
    "p4_plus": (
        FROZEN_RESIDUAL_ROOT / "p4_plus.f64le",
        "494f7b5de31673d53b368341d7960781767f948ed32afbae32b64b06a68f5cd8",
    ),
    "p4_minus": (
        FROZEN_RESIDUAL_ROOT / "p4_minus.f64le",
        "5583cb0cb6ab6fe0a2abae926ed56e3dd800d524d0539acbc54ced9ca8dea079",
    ),
    "p5_plus": (
        FROZEN_RESIDUAL_ROOT / "p5_plus.f64le",
        "7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2",
    ),
    "p5_minus": (
        FROZEN_RESIDUAL_ROOT / "p5_minus.f64le",
        "311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855",
    ),
}
PP3_FILES = {
    "pp3_plus": (
        PP3_ROOT / "pp3_plus.f64le",
        "66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff",
    ),
    "pp3_minus": (
        PP3_ROOT / "pp3_minus.f64le",
        "ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7",
    ),
}

PLAN_SHA = {
    "xg2": "b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c",
    "xg4": "792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f",
}

RESIDUAL_PLANES = ("P1", "P2", "P4", "P5")
CONDITIONS = (
    "native",
    "p1_neutralized",
    "p1_quarter_turn_control",
    "p2_neutralized",
    "p2_quarter_turn_control",
    "p4_neutralized",
    "p4_quarter_turn_control",
    "p5_neutralized",
    "p5_quarter_turn_control",
)

N = 300
ROWS = 1800
DIM = 395
K = 5
EPS = 0.025
TOL = 1.0e-12

DIRECTIONS = tuple(
    [f"xg2_{i}" for i in range(K)]
    + [f"xg4_{i}" for i in range(K)]
)

F_SIGNED = 2
F_DIR = 4
F_COND = 40
F_PAIR = 360
F_TOTAL = 108000

GPU_COUNT = 2
SHARDS = (
    {
        "shard_id": 0,
        "gpu_id": 0,
        "start_index": 0,
        "end_index": 150,
        "pair_first": "xg1_fact_1801",
        "pair_last": "xg1_fact_1950",
        "pair_count": 150,
        "forward_budget": 54000,
    },
    {
        "shard_id": 1,
        "gpu_id": 1,
        "start_index": 150,
        "end_index": 300,
        "pair_first": "xg1_fact_1951",
        "pair_last": "xg1_fact_2100",
        "pair_count": 150,
        "forward_budget": 54000,
    },
)

ITEM_FILE = "pp3_excluded_residual_individual_plane_necessity_items.jsonl"
SUMMARY_FILE = "pp3_excluded_residual_individual_plane_necessity_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen4-pp3-excluded-residual-individual-plane-necessity-item-v1"
SUMMARY_SCHEMA = "gen4-pp3-excluded-residual-individual-plane-necessity-summary-v1"
MANIFEST_SCHEMA = "gen4-pp3-excluded-residual-individual-plane-necessity-manifest-v1"
RESULT_PASS = (
    "PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_"
    "LOCALIZATION_RAW_OBSERVATION"
)
PLANNED_MULTIPLICITY = "Holm step-down across exactly P1,P2,P4,P5"
PLANNED_RAW_P_VALUE_COUNT = 4
PLANNED_FWER_ALPHA = 0.05

transport = aggregate.transport
holdout = aggregate.holdout
tokenizer_gate = aggregate.tokenizer_gate
adapter = aggregate.adapter


class ResidualIndividualPlaneNecessityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ResidualIndividualPlaneNecessityError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ResidualIndividualPlaneNecessityError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for commit, label in (
        (DESIGN_COMMIT, "DESIGN"),
        (STATIC_COMMIT, "STATIC"),
        (AUTHORITY_COMMIT, "AUTHORITY"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_NOT_ANCESTOR")

    require(
        git("rev-parse", f"HEAD:{AUTHORITY_PATH.as_posix()}")
        == AUTHORITY_BLOB,
        "AUTHORITY_BLOB_DRIFT",
    )


def expected_pairs() -> tuple[str, ...]:
    return tuple(
        f"xg1_fact_{index:03d}"
        for index in range(1801, 2101)
    )


def neutral_condition(plane: str) -> str:
    return f"{plane.lower()}_neutralized"


def control_condition(plane: str) -> str:
    return f"{plane.lower()}_quarter_turn_control"


def parse_condition(condition: str) -> tuple[str | None, str]:
    require(condition in CONDITIONS, f"CONDITION:{condition}")
    if condition == "native":
        return None, "native"
    for plane in RESIDUAL_PLANES:
        if condition == neutral_condition(plane):
            return plane, "neutralized"
        if condition == control_condition(plane):
            return plane, "quarter_turn_control"
    raise ResidualIndividualPlaneNecessityError(
        f"CONDITION_PARSE:{condition}"
    )


def validate_shards() -> None:
    require(len(SHARDS) == GPU_COUNT == 2, "SHARD_COUNT")
    covered: list[int] = []
    for expected_id, shard in enumerate(SHARDS):
        require(shard["shard_id"] == expected_id, "SHARD_ID")
        require(shard["gpu_id"] == expected_id, "SHARD_GPU")
        require(
            shard["end_index"] - shard["start_index"]
            == shard["pair_count"],
            "SHARD_PAIR_COUNT",
        )
        require(
            shard["pair_count"] * F_PAIR
            == shard["forward_budget"],
            "SHARD_FORWARD_BUDGET",
        )
        pair_slice = expected_pairs()[
            shard["start_index"]:shard["end_index"]
        ]
        require(
            len(pair_slice) == shard["pair_count"]
            and pair_slice[0] == shard["pair_first"]
            and pair_slice[-1] == shard["pair_last"],
            "SHARD_PAIR_RANGE",
        )
        covered.extend(
            range(shard["start_index"], shard["end_index"])
        )
    require(covered == list(range(N)), "SHARD_COVERAGE")
    require(
        sum(int(s["forward_budget"]) for s in SHARDS)
        == F_TOTAL,
        "TOTAL_FORWARD_BUDGET",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{path}:{line_no}",
        )
        out.append(value)
    return out


def canonical(value: Mapping[str, Any]) -> bytes:
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


def jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical(row) for row in rows)


def validate_static_inputs() -> dict[str, Any]:
    checks = {
        DATA_ROOT / "structured_source_facts.jsonl": SOURCE_SHA,
        DATA_ROOT / "synthetic_reason_router_six_cell.jsonl": ROWS_SHA,
        DATA_ROOT / "structural_manifest.json": STRUCT_SHA,
        STATIC_ROOT / "tokenizer_anchor_manifest.jsonl": ANCHOR_SHA,
        STATIC_ROOT / "tokenizer_eligibility_summary.json": ELIG_SHA,
        STATIC_ROOT / "geometry_manifest.json": GEOMETRY_SHA,
        STATIC_ROOT / "preparation_manifest.json": PREPARATION_SHA,
    }
    for path, expected in checks.items():
        full = ROOT / path
        require(full.is_file(), f"MISSING:{path}")
        require(sha256_file(full) == expected, f"SHA:{path}")

    structural = json.loads(
        (ROOT / DATA_ROOT / "structural_manifest.json")
        .read_text(encoding="utf-8-sig")
    )
    require(
        structural["pair_id_first"] == "xg1_fact_1801"
        and structural["pair_id_last"] == "xg1_fact_2100"
        and structural["source_pair_count"] == N
        and structural["row_count"] == ROWS
        and structural["pair_id_overlap_with_001_1800"] == 0
        and structural["claim_overlap_with_001_1800"] == 0
        and structural["evidence_overlap_with_001_1800"] == 0
        and structural["claim_evidence_row_overlap_with_001_1800"] == 0
        and structural["model_executed"] is False
        and structural["checkpoint_loaded"] is False
        and structural["cuda_executed"] is False,
        "STRUCTURAL_BOUNDARY",
    )

    preparation = json.loads(
        (ROOT / STATIC_ROOT / "preparation_manifest.json")
        .read_text(encoding="utf-8-sig")
    )
    require(
        preparation["result"]
        == (
            "PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_"
            "LOCALIZATION_STATIC_PREPARATION"
        ),
        "PREPARATION_RESULT",
    )
    require(
        preparation["scientific_model_forward_count"] == 0
        and preparation["checkpoint_load_count"] == 0
        and preparation["gpu_used"] is False
        and preparation["scientific_outcomes_observed"] is False
        and preparation["primary_inference_executed"] is False
        and preparation["multiplicity_correction_executed"] is False,
        "PREPARATION_BOUNDARY",
    )
    contract = preparation["future_execution_contract"]
    require(
        contract["condition_count"] == len(CONDITIONS)
        and contract["condition_order"] == list(CONDITIONS)
        and contract["scientific_model_forward_budget"] == F_TOTAL
        and contract["baseline_model_forward_budget"] == 0
        and contract["pair_count"] == N
        and contract["forwards_per_condition_per_pair"] == F_COND
        and contract["forwards_per_pair"] == F_PAIR
        and contract["raw_confirmatory_p_value_count"]
        == PLANNED_RAW_P_VALUE_COUNT
        and contract["multiplicity_method"] == PLANNED_MULTIPLICITY
        and float(contract["familywise_alpha"]) == PLANNED_FWER_ALPHA
        and contract["execution_authorized"] is False,
        "PREPARATION_EXECUTION_CONTRACT",
    )

    eligibility = json.loads(
        (ROOT / STATIC_ROOT / "tokenizer_eligibility_summary.json")
        .read_text(encoding="utf-8-sig")
    )
    require(
        eligibility["result"] == "PASS_300_OF_300"
        and eligibility["source_pair_count"] == N
        and eligibility["complete_source_pair_count"] == N
        and eligibility["anchor_row_count"] == ROWS
        and eligibility["pair_id_first"] == "xg1_fact_1801"
        and eligibility["pair_id_last"] == "xg1_fact_2100"
        and eligibility["model_forward_count"] == 0
        and eligibility["checkpoint_load_count"] == 0
        and eligibility["gpu_used"] is False,
        "ELIGIBILITY_BOUNDARY",
    )

    geometry = json.loads(
        (ROOT / STATIC_ROOT / "geometry_manifest.json")
        .read_text(encoding="utf-8-sig")
    )
    require(
        geometry["result"]
        == (
            "PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_"
            "LOCALIZATION_GEOMETRY_STATIC"
        )
        and geometry["residual_plane_order"] == list(RESIDUAL_PLANES)
        and geometry["excluded_plane"] == "P3"
        and geometry["all_residual_vectors_reproduced_exactly"] is True
        and float(
            geometry["full_principal_10_vector_gram_max_abs_residual"]
        ) <= TOL
        and float(geometry["residual_to_pp3_max_abs_dot"]) <= TOL
        and geometry["scientific_model_forward_count"] == 0
        and geometry["checkpoint_load_count"] == 0
        and geometry["gpu_used"] is False
        and geometry["scientific_outcomes_observed"] is False
        and geometry["primary_inference_executed"] is False,
        "GEOMETRY_BOUNDARY",
    )
    for plane in RESIDUAL_PLANES:
        value = geometry["per_plane_quarter_turn_control"][plane]
        require(
            float(value["max_abs_l2_mismatch"]) <= TOL
            and abs(float(value["max_abs_treatment_control_dot"])) <= TOL
            and abs(float(
                value["max_abs_projection_onto_other_residual_planes"]
            )) <= TOL
            and abs(float(value["max_abs_projection_onto_pp3"])) <= TOL
            and value["equal_norm_identity"] is True
            and value["orthogonal_control_identity"] is True
            and value["response_guided_weighting"] is False
            and value["response_guided_plane_selection"] is False,
            f"GEOMETRY_PLANE:{plane}",
        )

    return {
        "structural": structural,
        "preparation": preparation,
        "eligibility": eligibility,
        "geometry": geometry,
    }


def vector(raw: bytes, label: str) -> torch.Tensor:
    require(len(raw) == DIM * 8, f"VECTOR_BYTES:{label}")
    value = torch.tensor(
        struct.unpack(f"<{DIM}d", raw),
        dtype=torch.float64,
    )
    require(
        bool(torch.isfinite(value).all().item()),
        f"VECTOR_FINITE:{label}",
    )
    require(
        abs(float(torch.linalg.vector_norm(value)) - 1.0) <= TOL,
        f"VECTOR_NORM:{label}",
    )
    return value.contiguous()


def load_planes() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, (relative, expected_sha) in {
        **RESIDUAL_FILES,
        **PP3_FILES,
    }.items():
        path = ROOT / relative
        require(
            path.is_file() and sha256_file(path) == expected_sha,
            f"PLANE:{name}",
        )
        out[name] = vector(path.read_bytes(), name)

    residual_order = [
        "p1_plus", "p1_minus",
        "p2_plus", "p2_minus",
        "p4_plus", "p4_minus",
        "p5_plus", "p5_minus",
    ]
    residual = torch.stack(
        [out[name] for name in residual_order],
        dim=1,
    )
    require(
        float(torch.max(torch.abs(
            residual.T @ residual
            - torch.eye(8, dtype=torch.float64)
        ))) <= TOL,
        "RESIDUAL_GRAM",
    )
    pp3 = torch.stack(
        [out["pp3_plus"], out["pp3_minus"]],
        dim=1,
    )
    require(
        float(torch.max(torch.abs(residual.T @ pp3))) <= TOL,
        "RESIDUAL_PP3_ORTH",
    )
    require(
        abs(float(torch.dot(
            out["pp3_plus"], out["pp3_minus"]
        ))) <= TOL,
        "PP3_ORTH",
    )
    return out


def load_bases() -> dict[str, torch.Tensor]:
    loaded = holdout._load_frozen_bases()
    out: dict[str, torch.Tensor] = {}
    for family in ("xg2", "xg4"):
        require(
            loaded[family]["plan_sha256"] == PLAN_SHA[family],
            f"PLAN_SHA:{family}",
        )
        basis = (
            loaded[family]["basis"]["basis"]
            .detach().cpu().to(torch.float64).contiguous()
        )
        require(tuple(basis.shape) == (DIM, K), f"BASIS_SHAPE:{family}")
        require(
            float(torch.max(torch.abs(
                basis.T @ basis
                - torch.eye(K, dtype=torch.float64)
            ))) <= TOL,
            f"BASIS_GRAM:{family}",
        )
        out[family] = basis
    return out


def pair_order(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    seen: set[str] = set()
    order: list[str] = []
    for row in rows:
        pair = str(row["source_pair_id"])
        if pair not in seen:
            seen.add(pair)
            order.append(pair)
    require(tuple(order) == expected_pairs(), "PAIR_ORDER")
    return tuple(order)


def load_inputs(tokenizer_snapshot: str | Path | None):
    validate_static_inputs()
    rows = adapter.validate_gen4_rows(
        read_jsonl(
            ROOT / DATA_ROOT / "synthetic_reason_router_six_cell.jsonl"
        ),
        require_canonical_shape=True,
    )
    pairs = pair_order(rows)

    tokenizer, _ = tokenizer_gate.load_canonical_analysis_tokenizer(
        tokenizer_snapshot
    )
    encoded = adapter.encode_gen4_rows(rows, tokenizer)

    event_rows = read_jsonl(
        ROOT / STATIC_ROOT / "tokenizer_anchor_manifest.jsonl"
    )
    require(len(rows) == ROWS and len(event_rows) == ROWS, "ROW_COUNTS")
    require(
        Counter(str(row["anchor_name"]) for row in event_rows)
        == Counter({"A_IDENTITY": 1200, "A_NAME": 600}),
        "ANCHOR_COUNTS",
    )

    runtime = holdout.phase1.base.prevalence_eq
    events = runtime.parent.event_lookup(event_rows)
    runtime.parent.validate_transport_event_plan(pairs, events)
    require(
        list(encoded["source_pair_id"])
        == [str(row["source_pair_id"]) for row in rows],
        "ENCODED_PAIR_ORDER",
    )
    return rows, encoded, event_rows


def all_plane_components(
    h: torch.Tensor,
    planes: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    h = h.detach().cpu().to(torch.float64).contiguous()
    require(tuple(h.shape) == (DIM,), "H_SHAPE")

    coefficients: dict[str, list[float]] = {}
    per_plane: dict[str, dict[str, Any]] = {}

    for plane in RESIDUAL_PLANES:
        key = plane.lower()
        plus = planes[f"{key}_plus"]
        minus = planes[f"{key}_minus"]
        a = float(torch.dot(h, plus))
        b = float(torch.dot(h, minus))
        component = (a * plus + b * minus).contiguous()
        quarter = (-b * plus + a * minus).contiguous()
        delta_n = (-component).contiguous()
        delta_c = (-quarter).contiguous()
        l2_n = float(torch.linalg.vector_norm(delta_n))
        l2_c = float(torch.linalg.vector_norm(delta_c))
        mismatch = abs(l2_n - l2_c)
        dot = float(torch.dot(delta_n, delta_c))
        require(mismatch <= TOL, f"CORRECTION_L2_MISMATCH:{plane}:{mismatch}")
        require(abs(dot) <= TOL, f"CORRECTION_DOT:{plane}:{dot}")
        coefficients[plane] = [a, b]
        per_plane[plane] = {
            "component": component,
            "quarter_component": quarter,
            "delta_n": delta_n,
            "delta_c": delta_c,
            "treatment_correction_l2": l2_n,
            "control_correction_l2": l2_c,
            "treatment_control_l2_mismatch": mismatch,
            "treatment_control_dot": dot,
        }

    return {
        "h": h,
        "coefficients": coefficients,
        "per_plane": per_plane,
        "pp3_native_coefficients": [
            float(torch.dot(h, planes["pp3_plus"])),
            float(torch.dot(h, planes["pp3_minus"])),
        ],
    }


def condition_correction(
    h: torch.Tensor,
    condition: str,
    planes: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    selected_plane, kind = parse_condition(condition)
    info = all_plane_components(h, planes)

    if selected_plane is None:
        correction = torch.zeros(DIM, dtype=torch.float64)
        selected = None
    else:
        selected = info["per_plane"][selected_plane]
        correction = (
            selected["delta_n"]
            if kind == "neutralized"
            else selected["delta_c"]
        )

    post = (info["h"] + correction).contiguous()

    pp3_post = [
        float(torch.dot(post, planes["pp3_plus"])),
        float(torch.dot(post, planes["pp3_minus"])),
    ]
    pp3_drift = max(
        abs(a - b)
        for a, b in zip(
            pp3_post, info["pp3_native_coefficients"], strict=True
        )
    )
    require(pp3_drift <= TOL, f"PP3_DRIFT:{condition}:{pp3_drift}")

    residual_post: dict[str, list[float]] = {}
    for plane in RESIDUAL_PLANES:
        key = plane.lower()
        residual_post[plane] = [
            float(torch.dot(post, planes[f"{key}_plus"])),
            float(torch.dot(post, planes[f"{key}_minus"])),
        ]

    other_drift = 0.0
    for plane in RESIDUAL_PLANES:
        if plane == selected_plane:
            continue
        native_pair = info["coefficients"][plane]
        post_pair = residual_post[plane]
        other_drift = max(
            other_drift,
            abs(float(post_pair[0]) - float(native_pair[0])),
            abs(float(post_pair[1]) - float(native_pair[1])),
        )
    require(
        other_drift <= TOL,
        f"OTHER_PLANE_DRIFT:{condition}:{other_drift}",
    )

    target_neutral = None
    if kind == "neutralized":
        assert selected_plane is not None
        target_neutral = max(
            abs(value) for value in residual_post[selected_plane]
        )
        require(
            target_neutral <= TOL,
            f"TARGET_NEUTRALIZATION:{selected_plane}:{target_neutral}",
        )

    if selected is None:
        treatment_l2 = None
        control_l2 = None
        mismatch = None
        dot = None
    else:
        treatment_l2 = selected["treatment_correction_l2"]
        control_l2 = selected["control_correction_l2"]
        mismatch = selected["treatment_control_l2_mismatch"]
        dot = selected["treatment_control_dot"]

    return {
        "h": info["h"],
        "d": correction.contiguous(),
        "selected_plane": selected_plane,
        "intervention_kind": kind,
        "native_residual_coefficients": info["coefficients"],
        "treatment_correction_l2": treatment_l2,
        "control_correction_l2": control_l2,
        "treatment_control_l2_mismatch": mismatch,
        "treatment_control_dot": dot,
        "condition_correction_l2":
            float(torch.linalg.vector_norm(correction)),
        "pp3_native_coefficients": info["pp3_native_coefficients"],
        "pp3_post_coefficients": pp3_post,
        "pp3_coefficient_drift_max_abs": pp3_drift,
        "residual_post_condition_projections": residual_post,
        "other_residual_plane_drift_max_abs": other_drift,
        "target_plane_neutralization_max_abs_projection": target_neutral,
    }


def apply_hook(
    output: torch.Tensor,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    condition: str,
    planes: Mapping[str, torch.Tensor],
    direction: torch.Tensor,
    orientation: int,
    branch_sign: int,
    audit: dict[str, Any],
) -> torch.Tensor:
    runtime = holdout.phase1.base.prevalence_eq
    core = runtime.core

    require(
        output.ndim == 3
        and output.shape[0] == 1
        and output.shape[-1] == 2 * core.INTERMEDIATE_SIZE,
        "INPROJ_SHAPE",
    )
    require(orientation in {-1, 1} and branch_sign in {-1, 1}, "SIGN")

    mask = strong_mask.detach().cpu().bool().contiguous()
    require(
        mask.numel() == core.INTERMEDIATE_SIZE
        and int(mask.sum()) == DIM,
        "MASK",
    )

    before = output.detach().clone()
    mask_device = mask.to(before.device)
    h = (
        before[0, token_index, : core.INTERMEDIATE_SIZE][mask_device]
        .detach().cpu().to(torch.float64).contiguous()
    )
    condition_info = condition_correction(h, condition, planes)

    vector_value = (
        direction.detach().cpu().to(torch.float64).contiguous()
    )
    require(
        tuple(vector_value.shape) == (DIM,)
        and abs(float(torch.linalg.vector_norm(vector_value)) - 1.0) <= TOL,
        "DIRECTION",
    )

    probe = vector_value * (
        float(branch_sign) * float(orientation) * EPS
    )
    total = (condition_info["d"] + probe).contiguous()

    out = output.clone()
    intended = total.to(device=out.device, dtype=out.dtype)
    out[0, token_index, : core.INTERMEDIATE_SIZE][mask_device] += intended

    require(
        torch.equal(
            out[:, :, core.INTERMEDIATE_SIZE:],
            before[:, :, core.INTERMEDIATE_SIZE:],
        ),
        "GATE_CHANGED",
    )
    require(
        torch.equal(
            out[:, :, : core.INTERMEDIATE_SIZE][:, :, ~mask_device],
            before[:, :, : core.INTERMEDIATE_SIZE][:, :, ~mask_device],
        ),
        "NONSTRONG_CHANGED",
    )
    if token_index:
        require(
            torch.equal(
                out[:, :token_index, :],
                before[:, :token_index, :],
            ),
            "EARLIER_CHANGED",
        )
    if token_index + 1 < out.shape[1]:
        require(
            torch.equal(
                out[:, token_index + 1:, :],
                before[:, token_index + 1:, :],
            ),
            "LATER_CHANGED",
        )

    applied = (
        out[0, token_index, : core.INTERMEDIATE_SIZE][mask_device]
        - before[0, token_index, : core.INTERMEDIATE_SIZE][mask_device]
    ).detach().cpu().to(torch.float64)
    residual = float(torch.max(torch.abs(
        applied - intended.detach().cpu().to(torch.float64)
    )))
    require(
        residual <= runtime.transport_runtime.RUNTIME_CAST_TOL,
        f"APPLIED_RESIDUAL:{residual}",
    )

    audit.clear()
    audit.update({
        "condition": condition,
        "selected_plane": condition_info["selected_plane"],
        "intervention_kind": condition_info["intervention_kind"],
        "token_index": int(token_index),
        "orientation": int(orientation),
        "branch_sign": int(branch_sign),
        "coefficient_source":
            "branch_local_native_residual_coordinates",
        "native_residual_coefficients":
            condition_info["native_residual_coefficients"],
        "treatment_correction_l2":
            condition_info["treatment_correction_l2"],
        "control_correction_l2":
            condition_info["control_correction_l2"],
        "treatment_control_l2_mismatch":
            condition_info["treatment_control_l2_mismatch"],
        "treatment_control_dot":
            condition_info["treatment_control_dot"],
        "condition_correction_l2":
            condition_info["condition_correction_l2"],
        "pp3_native_coefficients":
            condition_info["pp3_native_coefficients"],
        "pp3_post_coefficients":
            condition_info["pp3_post_coefficients"],
        "pp3_coefficient_drift_max_abs":
            condition_info["pp3_coefficient_drift_max_abs"],
        "residual_post_condition_projections":
            condition_info["residual_post_condition_projections"],
        "other_residual_plane_drift_max_abs":
            condition_info["other_residual_plane_drift_max_abs"],
        "target_plane_neutralization_max_abs_projection":
            condition_info[
                "target_plane_neutralization_max_abs_projection"
            ],
        "probe_correction_l2":
            float(torch.linalg.vector_norm(probe)),
        "applied_correction_max_abs_residual": residual,
    })
    return out


def install_hook(mixer17: Any, **kwargs):
    def hook(_module, _args, output):
        return apply_hook(output, **kwargs)
    return mixer17.in_proj.register_forward_hook(hook)


def probe_seed(
    index: int,
    pair: str,
    events,
) -> dict[str, Any]:
    require(pair == expected_pairs()[index], f"PAIR:{index}")
    anchors = holdout.phase1._anchors_for_pair(pair, events)
    return {
        "family_key": "xg1",
        "source_pair_id": pair,
        "pair_index": index,
        "target_plus_anchor": int(anchors["tp"]),
        "target_minus_anchor": int(anchors["tm"]),
        "reference_plus_anchor": int(anchors["rp"]),
        "reference_minus_anchor": int(anchors["rm"]),
    }


def input_row(encoded, row_index, pair, cell):
    return holdout.phase2._input_row(
        encoded, row_index, pair, cell
    )


def run_signed(
    seed,
    direction,
    *,
    condition,
    orientation,
    planes,
    model,
    runtime_ctx,
    trace_code,
    trace_line,
    encoded,
    row_index,
    events,
    budget,
):
    runtime = holdout.phase1.base.prevalence_eq
    parent = runtime.parent
    core = runtime.core

    pair = str(seed["source_pair_id"])
    cells = holdout.phase1._cells()
    anchors = holdout.phase1._anchors_for_pair(pair, events)

    captured = {}
    audits = {}
    for role, branch_sign in (("tp", 1), ("tm", -1)):
        audit: dict[str, Any] = {}
        target = anchors[role] + core.TARGET_OFFSET
        handle = install_hook(
            runtime_ctx["mixer17"],
            token_index=target,
            strong_mask=runtime_ctx["strong_mask"],
            condition=condition,
            planes=planes,
            direction=direction,
            orientation=orientation,
            branch_sign=branch_sign,
            audit=audit,
        )
        try:
            captured[role] = parent.capture_branch(
                model,
                runtime_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                input_ids=input_row(
                    encoded, row_index, pair, cells[role]
                ),
                anchor=anchors[role],
                budget=budget,
                capture_states=True,
            )
        finally:
            handle.remove()

        require(
            bool(audit)
            and captured[role]["intervention_audit"] is None,
            "HOOK_AUDIT",
        )
        audits[role] = dict(audit)

    plus_efficiency = float(parent.path_efficiency(captured["tp"]))
    minus_efficiency = float(parent.path_efficiency(captured["tm"]))
    return {
        "condition": condition,
        "orientation": int(orientation),
        "F": plus_efficiency - minus_efficiency,
        "plus_path_efficiency": plus_efficiency,
        "minus_path_efficiency": minus_efficiency,
        "branch_audits": audits,
        "model_forward_count": F_SIGNED,
    }


def run_direction(
    seed,
    direction,
    *,
    condition,
    family,
    index,
    **kwargs,
):
    positive = run_signed(
        seed,
        direction,
        condition=condition,
        orientation=1,
        **kwargs,
    )
    negative = run_signed(
        seed,
        direction,
        condition=condition,
        orientation=-1,
        **kwargs,
    )
    f_plus = float(positive["F"])
    f_minus = float(negative["F"])
    j_value = (f_plus - f_minus) / (2.0 * EPS)
    return {
        "direction_key": f"{family}_{index}",
        "basis_family": family,
        "basis_index": index,
        "F_plus": f_plus,
        "F_minus": f_minus,
        "J": j_value,
        "J_squared": j_value * j_value,
        "positive_probe": positive,
        "negative_probe": negative,
        "model_forward_count": F_DIR,
    }


def run_condition(
    seed,
    *,
    condition,
    bases,
    **kwargs,
):
    probes = []
    for family in ("xg2", "xg4"):
        for index in range(K):
            probes.append(
                run_direction(
                    seed,
                    bases[family][:, index],
                    condition=condition,
                    family=family,
                    index=index,
                    **kwargs,
                )
            )
    require(
        [p["direction_key"] for p in probes] == list(DIRECTIONS),
        "DIRECTION_ORDER",
    )
    e_xg2 = math.fsum(
        float(p["J_squared"]) for p in probes[:K]
    ) / K
    e_xg4 = math.fsum(
        float(p["J_squared"]) for p in probes[K:]
    ) / K
    return {
        "condition": condition,
        "direction_order": list(DIRECTIONS),
        "direction_probes": probes,
        "E_XG2": e_xg2,
        "E_XG4": e_xg4,
        "Q": e_xg2 - e_xg4,
        "scientific_model_forward_count": F_COND,
    }


def plane_endpoint(
    q0: float,
    qn: float,
    qc: float,
) -> dict[str, float]:
    return {
        "QN": qn,
        "QC": qc,
        "A_N": q0 - qn,
        "A_C": q0 - qc,
        "D": qc - qn,
    }


def endpoints_from_conditions(
    by_condition: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    q0 = float(by_condition["native"]["Q"])
    plane_endpoints: dict[str, dict[str, float]] = {}
    for plane in RESIDUAL_PLANES:
        plane_endpoints[plane] = plane_endpoint(
            q0,
            float(by_condition[neutral_condition(plane)]["Q"]),
            float(by_condition[control_condition(plane)]["Q"]),
        )
    return {
        "Q0": q0,
        "plane_endpoints": plane_endpoints,
    }


def validate_endpoints(row: Mapping[str, Any]) -> None:
    q0 = float(row["Q0"])
    require(
        list(row["plane_endpoints"].keys()) == list(RESIDUAL_PLANES),
        "ENDPOINT_PLANE_ORDER",
    )
    for plane in RESIDUAL_PLANES:
        value = row["plane_endpoints"][plane]
        expected = plane_endpoint(
            q0,
            float(value["QN"]),
            float(value["QC"]),
        )
        require(
            set(value.keys()) == {"QN", "QC", "A_N", "A_C", "D"},
            f"ENDPOINT_KEYS:{plane}",
        )
        for key, expected_value in expected.items():
            require(
                float(value[key]) == expected_value,
                f"ENDPOINT:{plane}:{key}",
            )


def iter_audits(condition):
    for direction in condition["direction_probes"]:
        for probe_name in ("positive_probe", "negative_probe"):
            for role in ("tp", "tm"):
                yield (
                    direction["direction_key"],
                    probe_name,
                    role,
                    direction[probe_name]["branch_audits"][role],
                )


def _flatten_coefficients(
    value: Mapping[str, Sequence[float]],
) -> list[float]:
    require(
        list(value.keys()) == list(RESIDUAL_PLANES),
        "COEFFICIENT_PLANE_ORDER",
    )
    out: list[float] = []
    for plane in RESIDUAL_PLANES:
        pair = value[plane]
        require(len(pair) == 2, "COEFFICIENT_PAIR")
        out.extend([float(pair[0]), float(pair[1])])
    return out


def validate_matching(item) -> None:
    runtime = holdout.phase1.base.prevalence_eq
    cast_tol = runtime.transport_runtime.RUNTIME_CAST_TOL
    by_condition = {
        condition["condition"]: condition
        for condition in item["conditions"]
    }
    groups = [
        list(iter_audits(by_condition[name]))
        for name in CONDITIONS
    ]
    require(
        all(len(group) == 40 for group in groups),
        "MATCH_COUNT",
    )

    for matched in zip(*groups, strict=True):
        require(
            all(entry[:3] == matched[0][:3] for entry in matched),
            "MATCH_KEY",
        )
        coefficient_rows = [
            _flatten_coefficients(
                entry[3]["native_residual_coefficients"]
            )
            for entry in matched
        ]
        for column in range(8):
            values = [row[column] for row in coefficient_rows]
            require(
                max(values) - min(values) <= cast_tol,
                f"MATCH:COEFFICIENT:{column}",
            )


def run_pair(
    seed,
    *,
    bases,
    **kwargs,
):
    conditions = [
        run_condition(
            seed,
            condition=condition,
            bases=bases,
            **kwargs,
        )
        for condition in CONDITIONS
    ]
    require(
        [c["condition"] for c in conditions] == list(CONDITIONS),
        "CONDITION_ORDER",
    )
    by_condition = {
        c["condition"]: c for c in conditions
    }
    values = endpoints_from_conditions(by_condition)

    item = {
        **seed,
        "schema_version": ITEM_SCHEMA,
        "implementation_authority_commit": AUTHORITY_COMMIT,
        "static_preparation_freeze_commit": STATIC_COMMIT,
        "design_commit": DESIGN_COMMIT,
        "epsilon": EPS,
        "residual_plane_order": list(RESIDUAL_PLANES),
        "condition_order": list(CONDITIONS),
        "direction_order": list(DIRECTIONS),
        "conditions": conditions,
        **values,
        "baseline_model_forward_count_this_run": 0,
        "scientific_model_forward_count_this_run": F_PAIR,
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "scientific_conclusion": None,
    }
    validate_matching(item)
    validate_endpoints(item)
    return item


def validate_item(
    item: Mapping[str, Any],
    expected_pair: str,
    index: int,
) -> None:
    require(
        item["schema_version"] == ITEM_SCHEMA
        and item["source_pair_id"] == expected_pair
        and int(item["pair_index"]) == index
        and item["family_key"] == "xg1",
        f"ITEM_ID:{index}",
    )
    require(
        item["implementation_authority_commit"] == AUTHORITY_COMMIT
        and item["static_preparation_freeze_commit"] == STATIC_COMMIT
        and item["design_commit"] == DESIGN_COMMIT,
        f"ITEM_PROVENANCE:{index}",
    )
    require(float(item["epsilon"]) == EPS, f"ITEM_EPSILON:{index}")
    require(
        item["residual_plane_order"] == list(RESIDUAL_PLANES)
        and item["condition_order"] == list(CONDITIONS)
        and item["direction_order"] == list(DIRECTIONS),
        f"ITEM_ORDER:{index}",
    )
    require(
        item["scientific_model_forward_count_this_run"] == F_PAIR
        and item["baseline_model_forward_count_this_run"] == 0,
        f"ITEM_BUDGET:{index}",
    )
    require(
        item["primary_inference_executed"] is False
        and item["multiplicity_correction_executed"] is False
        and item["scientific_conclusion"] is None,
        f"ITEM_BOUNDARY:{index}",
    )

    conditions = item["conditions"]
    require(
        [c["condition"] for c in conditions] == list(CONDITIONS),
        f"COND_ORDER:{index}",
    )
    cast_tol = (
        holdout.phase1.base.prevalence_eq
        .transport_runtime.RUNTIME_CAST_TOL
    )

    for condition in conditions:
        condition_name = condition["condition"]
        selected_plane, kind = parse_condition(condition_name)
        require(
            condition["direction_order"] == list(DIRECTIONS)
            and condition["scientific_model_forward_count"] == F_COND,
            "COND_META",
        )
        probes = condition["direction_probes"]
        require(len(probes) == 10, "PROBE_COUNT")

        for key, probe in zip(DIRECTIONS, probes, strict=True):
            require(
                probe["direction_key"] == key
                and probe["model_forward_count"] == F_DIR,
                "DIR_META",
            )
            f_plus = float(probe["F_plus"])
            f_minus = float(probe["F_minus"])
            j_value = float(probe["J"])
            require(
                j_value == (f_plus - f_minus) / (2.0 * EPS)
                and float(probe["J_squared"]) == j_value * j_value,
                "J_ID",
            )

            for orientation, probe_name in (
                (1, "positive_probe"),
                (-1, "negative_probe"),
            ):
                signed = probe[probe_name]
                require(
                    signed["orientation"] == orientation
                    and signed["model_forward_count"] == F_SIGNED
                    and float(signed["F"])
                    == float(signed["plus_path_efficiency"])
                    - float(signed["minus_path_efficiency"]),
                    "SIGNED_META",
                )
                for role, branch_sign in (("tp", 1), ("tm", -1)):
                    audit = signed["branch_audits"][role]
                    require(
                        audit["condition"] == condition_name
                        and audit["selected_plane"] == selected_plane
                        and audit["intervention_kind"] == kind
                        and audit["orientation"] == orientation
                        and audit["branch_sign"] == branch_sign,
                        "AUDIT_COORDINATE",
                    )
                    require(
                        audit["coefficient_source"]
                        == "branch_local_native_residual_coordinates",
                        "AUDIT_SOURCE",
                    )
                    _flatten_coefficients(
                        audit["native_residual_coefficients"]
                    )
                    require(
                        float(audit["pp3_coefficient_drift_max_abs"]) <= TOL
                        and float(
                            audit["other_residual_plane_drift_max_abs"]
                        ) <= TOL,
                        "AUDIT_GEOMETRY_DRIFT",
                    )
                    require(
                        abs(float(audit["probe_correction_l2"]) - EPS)
                        <= TOL,
                        "AUDIT_PROBE_L2",
                    )
                    require(
                        float(audit["applied_correction_max_abs_residual"])
                        <= cast_tol,
                        "AUDIT_CAST_RESIDUAL",
                    )

                    if selected_plane is None:
                        require(
                            float(audit["condition_correction_l2"]) == 0.0
                            and audit["treatment_correction_l2"] is None
                            and audit["control_correction_l2"] is None
                            and audit["treatment_control_l2_mismatch"] is None
                            and audit["treatment_control_dot"] is None
                            and audit[
                                "target_plane_neutralization_max_abs_projection"
                            ] is None,
                            "NATIVE_AUDIT",
                        )
                    else:
                        for name in (
                            "treatment_correction_l2",
                            "control_correction_l2",
                            "treatment_control_l2_mismatch",
                            "treatment_control_dot",
                            "condition_correction_l2",
                        ):
                            require(
                                math.isfinite(float(audit[name])),
                                f"AUDIT_FINITE:{name}",
                            )
                        require(
                            float(audit["treatment_control_l2_mismatch"])
                            <= TOL
                            and abs(float(
                                audit["treatment_control_dot"]
                            )) <= TOL,
                            "AUDIT_MATCHED_CONTROL",
                        )
                        if kind == "neutralized":
                            neutral = audit[
                                "target_plane_neutralization_max_abs_projection"
                            ]
                            require(
                                neutral is not None
                                and float(neutral) <= TOL,
                                "TARGET_NEUTRALIZATION_AUDIT",
                            )
                        else:
                            require(
                                audit[
                                    "target_plane_neutralization_max_abs_projection"
                                ] is None,
                                "TARGET_NEUTRALIZATION_NULL",
                            )

        j2 = [float(p["J"]) for p in probes[:K]]
        j4 = [float(p["J"]) for p in probes[K:]]
        e2 = math.fsum(x * x for x in j2) / K
        e4 = math.fsum(x * x for x in j4) / K
        require(
            float(condition["E_XG2"]) == e2
            and float(condition["E_XG4"]) == e4
            and float(condition["Q"]) == e2 - e4,
            "Q_ID",
        )

    validate_matching(item)
    validate_endpoints(item)


def validate_items(
    items: Sequence[Mapping[str, Any]],
) -> None:
    require(len(items) == N, "ITEM_COUNT")
    for index, (pair, item) in enumerate(
        zip(expected_pairs(), items, strict=True)
    ):
        validate_item(item, pair, index)


def validate_shard_items(
    items: Sequence[Mapping[str, Any]],
    shard: Mapping[str, Any],
) -> None:
    require(len(items) == shard["pair_count"], "SHARD_ITEM_COUNT")
    for local_index, item in enumerate(items):
        global_index = int(shard["start_index"]) + local_index
        validate_item(
            item,
            expected_pairs()[global_index],
            global_index,
        )
    require(
        math.fsum(
            float(item["scientific_model_forward_count_this_run"])
            for item in items
        ) == shard["forward_budget"],
        "SHARD_ITEM_FORWARD_SUM",
    )


def merge_shards(
    shard_payloads: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(len(shard_payloads) == GPU_COUNT, "MERGE_SHARD_COUNT")
    by_id = {
        int(payload["shard_id"]): payload
        for payload in shard_payloads
    }
    require(set(by_id) == {0, 1}, "MERGE_SHARD_IDS")

    merged: list[dict[str, Any]] = []
    shard_meta: list[dict[str, Any]] = []
    checkpoint_shas: set[str] = set()

    for shard in SHARDS:
        payload = by_id[int(shard["shard_id"])]
        items = payload["items"]
        validate_shard_items(items, shard)
        require(
            payload["gpu_id"] == shard["gpu_id"]
            and payload["pair_first"] == shard["pair_first"]
            and payload["pair_last"] == shard["pair_last"]
            and payload["scientific_model_forward_count_this_run"]
            == shard["forward_budget"],
            "MERGE_SHARD_META",
        )
        checkpoint_shas.add(str(payload["checkpoint_sha256"]))
        merged.extend(items)
        shard_meta.append({
            "shard_id": int(payload["shard_id"]),
            "gpu_id": int(payload["gpu_id"]),
            "device_name": str(payload["device_name"]),
            "pair_first": str(payload["pair_first"]),
            "pair_last": str(payload["pair_last"]),
            "pair_count": int(payload["pair_count"]),
            "scientific_model_forward_count_this_run":
                int(payload["scientific_model_forward_count_this_run"]),
            "checkpoint_sha256": str(payload["checkpoint_sha256"]),
        })

    require(len(checkpoint_shas) == 1, "SHARD_CHECKPOINT_MISMATCH")
    validate_items(merged)
    return merged, shard_meta


def write_shard_payload(
    temp_dir: Path,
    shard_id: int,
    payload: Mapping[str, Any],
) -> None:
    (temp_dir / f"shard_{shard_id}.json").write_bytes(
        canonical(payload)
    )


def read_shard_payload(
    temp_dir: Path,
    shard_id: int,
) -> dict[str, Any]:
    path = temp_dir / f"shard_{shard_id}.json"
    require(path.is_file(), f"SHARD_PAYLOAD_MISSING:{shard_id}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(
        isinstance(value, dict),
        f"SHARD_PAYLOAD_OBJECT:{shard_id}",
    )
    return value


def runtime_gate_for_device(
    runtime: Any,
    gpu_id: int,
) -> None:
    transport.runtime_gate_for_device(runtime, gpu_id)


def make_fast_capture_for_device(
    runtime: Any,
    kernels: Mapping[str, Any],
    device: torch.device,
):
    return transport.make_fast_capture_for_device(
        runtime, kernels, device
    )


def worker_run(
    *,
    shard: Mapping[str, Any],
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    temp_dir: Path,
) -> None:
    shard_id = int(shard["shard_id"])
    error_path = temp_dir / f"shard_{shard_id}.error.txt"
    try:
        gpu_id = int(shard["gpu_id"])
        require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
        require(
            torch.cuda.device_count() >= GPU_COUNT,
            "CUDA_DEVICE_COUNT",
        )
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        authenticate_repo(expected_head)
        validate_shards()
        validate_static_inputs()
        planes = load_planes()
        bases = load_bases()

        runtime = holdout.phase1.base.prevalence_eq
        runtime_gate_for_device(runtime, gpu_id)

        with runtime.backend.parent_runtime_rebind():
            rows, encoded, event_rows = load_inputs(
                tokenizer_snapshot
            )
            pairs = pair_order(rows)
            parent = runtime.parent
            events = parent.event_lookup(event_rows)
            row_index = parent.build_row_index(rows)

            trace_code, trace_line = (
                runtime.measurement
                ._resolve_and_validate_runtime_binding()
            )
            kernels = runtime.kernel_compat.load_exact_fast_kernels()

            with (
                runtime.kernel_compat
                .exact_transformers_kernel_loader(kernels)
            ) as calls:
                model, checkpoint_sha = (
                    parent.load_representative_model_external(
                        model_snapshot=model_snapshot,
                        checkpoint_path=checkpoint_path,
                    )
                )
                require(
                    checkpoint_sha
                    == runtime.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
                    "CHECKPOINT",
                )
                runtime_ctx = (
                    runtime.transport_runtime
                    .validate_runtime_components(model)
                )

            counts = Counter(calls)
            require(
                set(counts) == {"causal-conv1d", "mamba-ssm"}
                and counts["causal-conv1d"] > 0
                and counts["causal-conv1d"] == counts["mamba-ssm"],
                "KERNEL_CONSTRUCTOR",
            )
            runtime.kernel_compat.validate_transformers_kernel_bindings(
                kernels
            )

            model.to(device)
            model.eval()

            fast_capture = make_fast_capture_for_device(
                runtime, kernels, device
            )
            original_capture = parent.capture_branch
            budget = parent.ForwardBudget(int(shard["forward_budget"]))
            items: list[dict[str, Any]] = []
            parent.capture_branch = fast_capture
            try:
                for global_index in range(
                    int(shard["start_index"]),
                    int(shard["end_index"]),
                ):
                    pair = pairs[global_index]
                    items.append(
                        run_pair(
                            probe_seed(global_index, pair, events),
                            bases=bases,
                            planes=planes,
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
                torch.cuda.synchronize(device)
            finally:
                parent.capture_branch = original_capture

        validate_shard_items(items, shard)
        payload = {
            "shard_id": shard_id,
            "gpu_id": gpu_id,
            "device_name": torch.cuda.get_device_name(gpu_id),
            "pair_first": items[0]["source_pair_id"],
            "pair_last": items[-1]["source_pair_id"],
            "pair_count": len(items),
            "scientific_model_forward_count_this_run":
                int(shard["forward_budget"]),
            "checkpoint_sha256": checkpoint_sha,
            "items": items,
        }
        write_shard_payload(temp_dir, shard_id, payload)
    except BaseException:
        error_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def worker_entry(
    shard: Mapping[str, Any],
    expected_head: str,
    model_snapshot: str,
    tokenizer_snapshot: str,
    checkpoint_path: str,
    temp_dir: str,
) -> None:
    worker_run(
        shard=shard,
        expected_head=expected_head,
        model_snapshot=Path(model_snapshot),
        tokenizer_snapshot=Path(tokenizer_snapshot),
        checkpoint_path=Path(checkpoint_path),
        temp_dir=Path(temp_dir),
    )


def write_outputs(
    out: Path,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    require(not out.exists(), "OUTPUT_COLLISION")
    validate_items(items)
    out.mkdir(parents=True, exist_ok=False)

    payloads = {
        ITEM_FILE: jsonl(items),
        SUMMARY_FILE: canonical(summary),
    }
    hashes: dict[str, str] = {}
    for name, raw in payloads.items():
        (out / name).write_bytes(raw)
        hashes[name] = sha256_bytes(raw)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "files": {
            name: {
                "sha256": digest,
                "bytes": int((out / name).stat().st_size),
            }
            for name, digest in sorted(hashes.items())
        },
    }
    manifest_raw = canonical(manifest)
    (out / MANIFEST_FILE).write_bytes(manifest_raw)
    hashes[MANIFEST_FILE] = sha256_bytes(manifest_raw)

    (out / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )


def validate_artifact(
    out: Path,
) -> dict[str, Any]:
    manifest = json.loads(
        (out / MANIFEST_FILE).read_text(encoding="utf-8-sig")
    )
    require(
        manifest["schema_version"] == MANIFEST_SCHEMA,
        "MANIFEST_SCHEMA",
    )

    hashes: dict[str, str] = {}
    for name in (ITEM_FILE, SUMMARY_FILE):
        path = out / name
        require(path.is_file(), f"FILE_MISSING:{name}")
        digest = sha256_file(path)
        require(
            digest == manifest["files"][name]["sha256"]
            and path.stat().st_size == manifest["files"][name]["bytes"],
            f"FILE:{name}",
        )
        hashes[name] = digest
    hashes[MANIFEST_FILE] = sha256_file(out / MANIFEST_FILE)

    observed: dict[str, str] = {}
    for line in (
        out / CHECKSUM_FILE
    ).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in observed, f"CHECKSUM_DUPLICATE:{name}")
        observed[name] = digest
    require(
        observed == {
            name: digest
            for name, digest in sorted(hashes.items())
        },
        "CHECKSUMS",
    )

    items = read_jsonl(out / ITEM_FILE)
    validate_items(items)

    summary = json.loads(
        (out / SUMMARY_FILE).read_text(encoding="utf-8-sig")
    )
    require(
        summary["schema_version"] == SUMMARY_SCHEMA
        and summary["result"] == RESULT_PASS,
        "SUMMARY_RESULT",
    )
    require(
        summary["source_pair_count"] == N
        and summary["pair_id_first"] == "xg1_fact_1801"
        and summary["pair_id_last"] == "xg1_fact_2100",
        "SUMMARY_POPULATION",
    )
    require(
        bool(summary["execution_head"])
        and summary["design_commit"] == DESIGN_COMMIT
        and summary["static_preparation_freeze_commit"] == STATIC_COMMIT
        and summary["implementation_authority_commit"] == AUTHORITY_COMMIT,
        "SUMMARY_PROVENANCE",
    )
    require(float(summary["epsilon"]) == EPS, "SUMMARY_EPSILON")
    require(
        summary["residual_plane_order"] == list(RESIDUAL_PLANES)
        and summary["condition_order"] == list(CONDITIONS)
        and summary["direction_order"] == list(DIRECTIONS),
        "SUMMARY_ORDER",
    )
    require(
        summary["model_forwards_per_direction"] == F_DIR
        and summary["model_forwards_per_condition"] == F_COND
        and summary["model_forwards_per_pair"] == F_PAIR
        and summary["scientific_model_forward_count_this_run"] == F_TOTAL
        and summary["baseline_model_forward_count_this_run"] == 0,
        "SUMMARY_BUDGET",
    )
    require(
        summary["gpu_count"] == GPU_COUNT
        and len(summary["shards"]) == GPU_COUNT,
        "SUMMARY_GPU_TOPOLOGY",
    )
    for shard, expected in zip(
        summary["shards"], SHARDS, strict=True
    ):
        require(
            shard["shard_id"] == expected["shard_id"]
            and shard["gpu_id"] == expected["gpu_id"]
            and shard["pair_first"] == expected["pair_first"]
            and shard["pair_last"] == expected["pair_last"]
            and shard["pair_count"] == expected["pair_count"]
            and shard["scientific_model_forward_count_this_run"]
            == expected["forward_budget"],
            "SUMMARY_SHARD",
        )
    require(
        len({
            shard["checkpoint_sha256"]
            for shard in summary["shards"]
        }) == 1
        and summary["representative_checkpoint_sha256"]
        == summary["shards"][0]["checkpoint_sha256"],
        "SUMMARY_CHECKPOINT",
    )
    require(
        summary["representative_checkpoint_sha256"]
        == holdout.phase1.base.prevalence_eq
        .extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
        "SUMMARY_FROZEN_CHECKPOINT",
    )
    require(
        summary["primary_endpoint_definition"]
        == "for each k in {P1,P2,P4,P5}: D_k=QC_k-QN_k",
        "SUMMARY_ENDPOINT",
    )
    require(
        summary["planned_raw_confirmatory_p_value_count"]
        == PLANNED_RAW_P_VALUE_COUNT
        and summary["planned_multiplicity_method"]
        == PLANNED_MULTIPLICITY
        and float(summary["planned_familywise_alpha"])
        == PLANNED_FWER_ALPHA,
        "SUMMARY_PLANNED_INFERENCE",
    )
    for key in (
        "primary_inference_executed",
        "multiplicity_correction_executed",
        "training_executed",
        "backward_executed",
        "task_heads_executed",
        "logits_read",
    ):
        require(summary[key] is False, f"BOUNDARY:{key}")
    require(summary["scientific_conclusion"] is None, "CONCLUSION")
    return {
        "items": items,
        "summary": summary,
        "manifest": manifest,
    }


def run_observation(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    validate_shards()
    validate_static_inputs()
    load_planes()
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(
        torch.cuda.device_count() >= GPU_COUNT,
        "CUDA_DEVICE_COUNT",
    )

    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(
        prefix="gen4_residual_individual_plane_necessity_"
    ) as temp_name:
        temp_dir = Path(temp_name)
        processes = []
        for shard in SHARDS:
            process = ctx.Process(
                target=worker_entry,
                args=(
                    dict(shard),
                    expected_head,
                    str(model_snapshot),
                    str(tokenizer_snapshot),
                    str(checkpoint_path),
                    str(temp_dir),
                ),
                name=(
                    "gen4-residual-individual-plane-"
                    f"gpu{shard['gpu_id']}"
                ),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        failures = []
        for shard, process in zip(
            SHARDS, processes, strict=True
        ):
            if process.exitcode != 0:
                error_path = (
                    temp_dir
                    / f"shard_{int(shard['shard_id'])}.error.txt"
                )
                detail = (
                    error_path.read_text(encoding="utf-8")
                    if error_path.is_file()
                    else f"exitcode={process.exitcode}"
                )
                failures.append(
                    f"SHARD_{shard['shard_id']}_FAILED:\n{detail}"
                )
        require(not failures, "\n".join(failures))

        payloads = [
            read_shard_payload(
                temp_dir,
                int(shard["shard_id"]),
            )
            for shard in SHARDS
        ]
        items, shard_meta = merge_shards(payloads)

    checkpoint_sha = shard_meta[0]["checkpoint_sha256"]
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "design_commit": DESIGN_COMMIT,
        "static_preparation_freeze_commit": STATIC_COMMIT,
        "implementation_authority_commit": AUTHORITY_COMMIT,
        "source_pair_count": N,
        "pair_id_first": items[0]["source_pair_id"],
        "pair_id_last": items[-1]["source_pair_id"],
        "epsilon": EPS,
        "residual_plane_order": list(RESIDUAL_PLANES),
        "condition_order": list(CONDITIONS),
        "direction_order": list(DIRECTIONS),
        "model_forwards_per_direction": F_DIR,
        "model_forwards_per_condition": F_COND,
        "model_forwards_per_pair": F_PAIR,
        "scientific_model_forward_count_this_run": F_TOTAL,
        "baseline_model_forward_count_this_run": 0,
        "gpu_count": GPU_COUNT,
        "parallelization": "independent_pair_shards_spawn",
        "shards": shard_meta,
        "primary_endpoint_definition":
            "for each k in {P1,P2,P4,P5}: D_k=QC_k-QN_k",
        "planned_raw_confirmatory_p_value_count":
            PLANNED_RAW_P_VALUE_COUNT,
        "planned_multiplicity_method": PLANNED_MULTIPLICITY,
        "planned_familywise_alpha": PLANNED_FWER_ALPHA,
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
        "representative_checkpoint_sha256": checkpoint_sha,
    }

    write_outputs(output_dir, items, summary)
    validate_artifact(output_dir)
    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen 2-GPU raw observation for PP3-excluded residual "
            "individual-plane necessity localization; no statistical inference."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    summary = run_observation(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT=" + summary["result"])
    print("GPU_COUNT=" + str(summary["gpu_count"]))
    for shard in summary["shards"]:
        print(
            f"SHARD_{shard['shard_id']}_GPU="
            f"{shard['gpu_id']}:"
            f"{shard['pair_first']}..{shard['pair_last']}:"
            "FORWARDS="
            f"{shard['scientific_model_forward_count_this_run']}"
        )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(summary["scientific_model_forward_count_this_run"])
    )
    print("BASELINE_MODEL_FORWARD_COUNT_THIS_RUN=0")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("MULTIPLICITY_CORRECTION_EXECUTED=False")
    print("SCIENTIFIC_CONCLUSION=None")


if __name__ == "__main__":
    main()
