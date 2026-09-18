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

from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as tokenizer_gate
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter
from scripts import reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda as holdout


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"

AUTHORITY_COMMIT = "cae566ed458e5c6f93c86ce03b029950035652dc"
STATIC_COMMIT = "0f907574f1ca25ec12e35573a83e1b499ed1b53b"
AUTHORITY_PATH = (
    "reports/reason_router_gen4_pp3_restoration_sufficiency_"
    "implementation_authority.md"
)
AUTHORITY_BLOB = "090b68503c3f10327065bc275f7b6daffdccfc59"

DATA_ROOT = Path("data/reason_router_gen4_xg1_restoration_sufficiency_v1")
STATIC_ROOT = Path(
    "reports/"
    "reason_router_gen4_pp3_restoration_sufficiency_static_preparation_854bcd5"
)

SOURCE_SHA = "2c700452d818531c46a8ffd473eb6d64d8af29f3a9284da8371f5c9eb2610c21"
ROWS_SHA = "7ec2ea86f35562394244f6df6e8b098ea3ba8a9bf358868fc614f5029746241c"
STRUCT_SHA = "a1c6957d7d48fb93b53f7b93bb5378caf0bbdc0adb96c438f6aa56a779ae9922"
ANCHOR_SHA = "dc4f2cd4ca2806249467407c7c980411d8fa02051418f9a8b625c1b1c4756253"
ELIG_SHA = "0fb8da67687f223b6c72ea7bc946542e6543bb85c050163056696a8e945e5a89"
GEOMETRY_SHA = "4fdb7778738db109138199ab2e9725cfb7db3cb2f173a88b96a045c595c17da7"
PREPARATION_SHA = "19fce1d109b8ab2d8e7faddd9ea5b2f5571e0ea5f1c2c30927a4d7a10e0e061c"

PP3_ROOT = Path(
    "reports/reason_router_gen4_pp3_xg1_external_transport_preparation_02e4c89"
)
PP5_ROOT = Path(
    "reports/"
    "reason_router_gen4_pp3_pp5_fresh_xg1_specificity_preparation_0bc49ab"
)
PLANE_FILES = {
    "pp3_plus": (
        PP3_ROOT / "pp3_plus.f64le",
        "66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff",
    ),
    "pp3_minus": (
        PP3_ROOT / "pp3_minus.f64le",
        "ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7",
    ),
    "pp5_plus": (
        PP5_ROOT / "pp5_plus.f64le",
        "7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2",
    ),
    "pp5_minus": (
        PP5_ROOT / "pp5_minus.f64le",
        "311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855",
    ),
}

PLAN_SHA = {
    "xg2": "b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c",
    "xg4": "792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f",
}

N = 300
ROWS = 1800
DIM = 395
K = 5
EPS = 0.025
CONDITIONS = ("pp3_neutralized", "pp3_restored", "pp5_replacement")
DIRECTIONS = tuple(
    [f"xg2_{i}" for i in range(K)] + [f"xg4_{i}" for i in range(K)]
)
F_SIGNED = 2
F_DIR = 4
F_COND = 40
F_PAIR = 120
F_TOTAL = 36000
TOL = 1e-12

ITEM_FILE = "pp3_restoration_sufficiency_items.jsonl"
SUMMARY_FILE = "pp3_restoration_sufficiency_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen4-pp3-restoration-sufficiency-item-v1"
SUMMARY_SCHEMA = "gen4-pp3-restoration-sufficiency-summary-v1"
MANIFEST_SCHEMA = "gen4-pp3-restoration-sufficiency-manifest-v1"
RESULT_PASS = "PASS_PP3_RESTORATION_SUFFICIENCY_RAW_OBSERVATION"


class PP3RestorationSufficiencyError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PP3RestorationSufficiencyError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise PP3RestorationSufficiencyError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for commit, label in (
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

    observed_blob = git("rev-parse", f"HEAD:{AUTHORITY_PATH}")
    require(
        observed_blob == AUTHORITY_BLOB,
        f"AUTHORITY_BLOB_DRIFT:{observed_blob}",
    )


def expected_pairs() -> tuple[str, ...]:
    return tuple(f"xg1_fact_{i:03d}" for i in range(901, 1201))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{number}")
        rows.append(value)
    return rows


def validate_static_inputs() -> None:
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

    eligibility = json.loads(
        (ROOT / STATIC_ROOT / "tokenizer_eligibility_summary.json")
        .read_text(encoding="utf-8-sig")
    )
    require(
        eligibility["result"] == "PASS_300_OF_300"
        and eligibility["model_forward_count"] == 0
        and eligibility["checkpoint_load_count"] == 0
        and eligibility["gpu_used"] is False,
        "ELIGIBILITY_BOUNDARY",
    )

    preparation = json.loads(
        (ROOT / STATIC_ROOT / "preparation_manifest.json")
        .read_text(encoding="utf-8-sig")
    )
    require(
        preparation["result"]
        == "PASS_PP3_RESTORATION_SUFFICIENCY_STATIC_PREPARATION",
        "PREPARATION_RESULT",
    )
    require(
        preparation["scientific_model_forward_count"] == 0
        and preparation["checkpoint_load_count"] == 0
        and preparation["gpu_used"] is False
        and preparation["scientific_outcomes_observed"] is False
        and preparation["primary_inference_executed"] is False,
        "PREPARATION_BOUNDARY",
    )


def vector(raw: bytes, label: str) -> torch.Tensor:
    require(len(raw) == DIM * 8, f"VECTOR_BYTES:{label}")
    value = torch.tensor(
        struct.unpack(f"<{DIM}d", raw),
        dtype=torch.float64,
    )
    require(bool(torch.isfinite(value).all()), f"VECTOR_FINITE:{label}")
    require(
        abs(float(torch.linalg.vector_norm(value)) - 1.0) <= TOL,
        f"VECTOR_NORM:{label}",
    )
    return value.contiguous()


def load_planes() -> dict[str, torch.Tensor]:
    planes: dict[str, torch.Tensor] = {}
    for name, (relative, expected_sha) in PLANE_FILES.items():
        path = ROOT / relative
        require(
            path.is_file() and sha256_file(path) == expected_sha,
            f"PLANE:{name}",
        )
        planes[name] = vector(path.read_bytes(), name)

    require(
        abs(float(torch.dot(planes["pp3_plus"], planes["pp3_minus"])))
        <= TOL,
        "PP3_ORTH",
    )
    require(
        abs(float(torch.dot(planes["pp5_plus"], planes["pp5_minus"])))
        <= TOL,
        "PP5_ORTH",
    )
    pp3 = torch.stack(
        [planes["pp3_plus"], planes["pp3_minus"]],
        dim=1,
    )
    pp5 = torch.stack(
        [planes["pp5_plus"], planes["pp5_minus"]],
        dim=1,
    )
    cross = pp3.T @ pp5
    require(
        float(torch.max(torch.abs(cross))) <= TOL,
        "CROSS_ORTH",
    )
    return planes


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
            .detach()
            .cpu()
            .to(torch.float64)
            .contiguous()
        )
        require(tuple(basis.shape) == (DIM, K), f"BASIS_SHAPE:{family}")
        out[family] = basis
    return out


def pair_order(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
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
        read_jsonl(ROOT / DATA_ROOT / "synthetic_reason_router_six_cell.jsonl"),
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
    require(
        len(rows) == ROWS and len(event_rows) == ROWS,
        "ROW_COUNTS",
    )
    require(
        Counter(str(row["anchor_name"]) for row in event_rows)
        == Counter({"A_IDENTITY": 1200, "A_NAME": 600}),
        "ANCHOR_COUNTS",
    )

    runtime = holdout.phase1.base.prevalence_eq
    parent = runtime.parent
    events = parent.event_lookup(event_rows)
    parent.validate_transport_event_plan(pairs, events)

    require(
        list(encoded["source_pair_id"])
        == [str(row["source_pair_id"]) for row in rows],
        "ENCODED_PAIR_ORDER",
    )
    return rows, encoded, event_rows


def condition_correction(
    h: torch.Tensor,
    condition: str,
    planes: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    require(condition in CONDITIONS, f"CONDITION:{condition}")

    h = h.detach().cpu().to(torch.float64).contiguous()
    require(tuple(h.shape) == (DIM,), "H_SHAPE")

    a = float(torch.dot(h, planes["pp3_plus"]))
    b = float(torch.dot(h, planes["pp3_minus"]))

    c3 = (
        a * planes["pp3_plus"]
        + b * planes["pp3_minus"]
    ).contiguous()
    c5 = (
        a * planes["pp5_plus"]
        + b * planes["pp5_minus"]
    ).contiguous()

    c3_l2 = float(torch.linalg.vector_norm(c3))
    c5_l2 = float(torch.linalg.vector_norm(c5))
    addition_norm_mismatch = abs(c3_l2 - c5_l2)
    require(
        addition_norm_mismatch <= TOL,
        f"RESTORATION_ADDITION_NORM:{addition_norm_mismatch}",
    )

    if condition == "pp3_neutralized":
        correction = -c3
    elif condition == "pp3_restored":
        correction = torch.zeros_like(h)
    else:
        correction = -c3 + c5

    correction = correction.contiguous()
    post = h + correction

    pp3_res_plus = float(torch.dot(post, planes["pp3_plus"]))
    pp3_res_minus = float(torch.dot(post, planes["pp3_minus"]))

    neutralized_residual = None
    r3_residual = None
    r5_residual = None

    if condition == "pp3_neutralized":
        expected = h - c3
        neutralized_residual = float(
            torch.max(torch.abs(post - expected))
        )
        require(
            abs(pp3_res_plus) <= TOL
            and abs(pp3_res_minus) <= TOL
            and neutralized_residual <= TOL,
            "PP3_NEUTRALIZATION_IDENTITY",
        )
    elif condition == "pp3_restored":
        r3_residual = float(torch.max(torch.abs(post - h)))
        require(r3_residual <= TOL, f"R3_NATIVE_IDENTITY:{r3_residual}")
    else:
        expected = h - c3 + c5
        r5_residual = float(torch.max(torch.abs(post - expected)))
        require(r5_residual <= TOL, f"R5_CONSTRUCTION:{r5_residual}")

    return {
        "a": a,
        "b": b,
        "c3": c3,
        "c5": c5,
        "c3_l2": c3_l2,
        "c5_l2": c5_l2,
        "restoration_addition_norm_mismatch": addition_norm_mismatch,
        "d": correction,
        "direct_final_state_correction_l2": float(
            torch.linalg.vector_norm(correction)
        ),
        "pp3_post_condition_residual_plus": pp3_res_plus,
        "pp3_post_condition_residual_minus": pp3_res_minus,
        "neutralized_construction_max_abs_residual":
            neutralized_residual,
        "r3_native_state_max_abs_residual": r3_residual,
        "r5_construction_max_abs_residual": r5_residual,
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
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )

    condition_info = condition_correction(h, condition, planes)

    vector_value = (
        direction.detach().cpu().to(torch.float64).contiguous()
    )
    require(
        tuple(vector_value.shape) == (DIM,)
        and abs(float(torch.linalg.vector_norm(vector_value)) - 1.0)
        <= TOL,
        "DIRECTION",
    )

    probe = vector_value * (
        float(branch_sign) * float(orientation) * EPS
    )
    total = (condition_info["d"] + probe).contiguous()

    out = output.clone()
    intended = total.to(device=out.device, dtype=out.dtype)
    out[
        0,
        token_index,
        : core.INTERMEDIATE_SIZE,
    ][mask_device] += intended

    require(
        torch.equal(
            out[:, :, core.INTERMEDIATE_SIZE :],
            before[:, :, core.INTERMEDIATE_SIZE :],
        ),
        "GATE_CHANGED",
    )

    nonstrong = ~mask_device
    require(
        torch.equal(
            out[:, :, : core.INTERMEDIATE_SIZE][:, :, nonstrong],
            before[:, :, : core.INTERMEDIATE_SIZE][:, :, nonstrong],
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
                out[:, token_index + 1 :, :],
                before[:, token_index + 1 :, :],
            ),
            "LATER_CHANGED",
        )

    applied = (
        out[
            0,
            token_index,
            : core.INTERMEDIATE_SIZE,
        ][mask_device]
        - before[
            0,
            token_index,
            : core.INTERMEDIATE_SIZE,
        ][mask_device]
    ).detach().cpu().to(torch.float64)

    residual = float(
        torch.max(
            torch.abs(
                applied
                - intended.detach().cpu().to(torch.float64)
            )
        )
    )
    require(
        residual <= runtime.transport_runtime.RUNTIME_CAST_TOL,
        f"APPLIED_RESIDUAL:{residual}",
    )

    audit.clear()
    audit.update(
        {
            "condition": condition,
            "token_index": int(token_index),
            "orientation": int(orientation),
            "branch_sign": int(branch_sign),
            "coefficient_source": "native_pp3_coordinates",
            "native_pp3_a": condition_info["a"],
            "native_pp3_b": condition_info["b"],
            "pp3_component_l2": condition_info["c3_l2"],
            "pp5_component_l2": condition_info["c5_l2"],
            "restoration_addition_norm_mismatch":
                condition_info["restoration_addition_norm_mismatch"],
            "direct_final_state_correction_l2":
                condition_info["direct_final_state_correction_l2"],
            "pp3_post_condition_residual_plus":
                condition_info["pp3_post_condition_residual_plus"],
            "pp3_post_condition_residual_minus":
                condition_info["pp3_post_condition_residual_minus"],
            "neutralized_construction_max_abs_residual":
                condition_info[
                    "neutralized_construction_max_abs_residual"
                ],
            "r3_native_state_max_abs_residual":
                condition_info["r3_native_state_max_abs_residual"],
            "r5_construction_max_abs_residual":
                condition_info["r5_construction_max_abs_residual"],
            "probe_correction_l2":
                float(torch.linalg.vector_norm(probe)),
            "applied_correction_max_abs_residual": residual,
        }
    )
    return out


def install_hook(mixer17: Any, **kwargs):
    def hook(_module, _args, output):
        return apply_hook(output, **kwargs)

    return mixer17.in_proj.register_forward_hook(hook)


def probe_seed(index: int, pair: str, events):
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
        encoded,
        row_index,
        pair,
        cell,
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
                    encoded,
                    row_index,
                    pair,
                    cells[role],
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
        "orientation": orientation,
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
    j_value = (f_plus - f_minus) / (2 * EPS)

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


def run_condition(seed, *, condition, bases, **kwargs):
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
        [probe["direction_key"] for probe in probes]
        == list(DIRECTIONS),
        "DIRECTION_ORDER",
    )

    e_xg2 = sum(
        float(probe["J_squared"])
        for probe in probes[:K]
    ) / K
    e_xg4 = sum(
        float(probe["J_squared"])
        for probe in probes[K:]
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


def endpoint(
    q_b: float,
    q_r3: float,
    q_r5: float,
) -> dict[str, float]:
    s3 = q_r3 - q_b
    s5 = q_r5 - q_b
    d_suf = s3 - s5
    require(
        d_suf == q_r3 - q_r5,
        "D_SUF_INTERNAL",
    )
    return {
        "Q_B": q_b,
        "Q_R3": q_r3,
        "Q_R5": q_r5,
        "S3": s3,
        "S5": s5,
        "D_SUF": d_suf,
    }


def validate_endpoint(row: Mapping[str, Any]) -> None:
    expected = endpoint(
        float(row["Q_B"]),
        float(row["Q_R3"]),
        float(row["Q_R5"]),
    )
    for key, value in expected.items():
        require(float(row[key]) == value, f"ENDPOINT:{key}")


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


def validate_matching(item) -> None:
    runtime = holdout.phase1.base.prevalence_eq
    by_condition = {
        condition["condition"]: condition
        for condition in item["conditions"]
    }

    neutralized = list(
        iter_audits(by_condition["pp3_neutralized"])
    )
    restored = list(
        iter_audits(by_condition["pp3_restored"])
    )
    replacement = list(
        iter_audits(by_condition["pp5_replacement"])
    )

    require(
        len(neutralized) == len(restored) == len(replacement) == 40,
        "MATCH_COUNT",
    )

    for left, middle, right in zip(
        neutralized,
        restored,
        replacement,
        strict=True,
    ):
        require(
            left[:3] == middle[:3] == right[:3],
            "MATCH_KEY",
        )

        audits = (left[3], middle[3], right[3])
        for key in (
            "native_pp3_a",
            "native_pp3_b",
            "pp3_component_l2",
            "pp5_component_l2",
        ):
            values = [float(audit[key]) for audit in audits]
            require(
                max(values) - min(values)
                <= runtime.transport_runtime.RUNTIME_CAST_TOL,
                f"MATCH:{key}",
            )

        for audit in audits:
            require(
                float(audit["restoration_addition_norm_mismatch"])
                <= TOL,
                "MATCH:RESTORATION_ADDITION_NORM",
            )


def run_pair(seed, *, bases, **kwargs):
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
        [condition["condition"] for condition in conditions]
        == list(CONDITIONS),
        "CONDITION_ORDER",
    )

    by_condition = {
        condition["condition"]: condition
        for condition in conditions
    }
    endpoint_values = endpoint(
        float(by_condition["pp3_neutralized"]["Q"]),
        float(by_condition["pp3_restored"]["Q"]),
        float(by_condition["pp5_replacement"]["Q"]),
    )

    item = {
        **seed,
        "schema_version": ITEM_SCHEMA,
        "implementation_authority_commit": AUTHORITY_COMMIT,
        "static_preparation_freeze_commit": STATIC_COMMIT,
        "epsilon": EPS,
        "condition_order": list(CONDITIONS),
        "direction_order": list(DIRECTIONS),
        "conditions": conditions,
        **endpoint_values,
        "baseline_model_forward_count_this_run": 0,
        "scientific_model_forward_count_this_run": F_PAIR,
    }

    validate_matching(item)
    validate_endpoint(item)
    return item


def validate_item(
    item: Mapping[str, Any],
    expected_pair: str,
    index: int,
) -> None:
    require(
        item["schema_version"] == ITEM_SCHEMA
        and item["source_pair_id"] == expected_pair
        and item["pair_index"] == index,
        f"ITEM_ID:{index}",
    )
    require(item.get("family_key") == "xg1", f"ITEM_FAMILY:{index}")
    require(
        item.get("implementation_authority_commit")
        == AUTHORITY_COMMIT,
        f"ITEM_AUTHORITY:{index}",
    )
    require(
        item.get("static_preparation_freeze_commit")
        == STATIC_COMMIT,
        f"ITEM_STATIC_FREEZE:{index}",
    )
    require(float(item["epsilon"]) == EPS, f"ITEM_EPSILON:{index}")
    require(
        item["condition_order"] == list(CONDITIONS)
        and item["direction_order"] == list(DIRECTIONS),
        f"ITEM_ORDER:{index}",
    )
    require(
        item["scientific_model_forward_count_this_run"] == F_PAIR
        and item["baseline_model_forward_count_this_run"] == 0,
        f"ITEM_BUDGET:{index}",
    )

    conditions = item["conditions"]
    require(
        [condition["condition"] for condition in conditions]
        == list(CONDITIONS),
        f"COND_ORDER:{index}",
    )

    runtime = holdout.phase1.base.prevalence_eq
    cast_tol = runtime.transport_runtime.RUNTIME_CAST_TOL

    for condition in conditions:
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

            for orientation, probe_name in (
                (1, "positive_probe"),
                (-1, "negative_probe"),
            ):
                signed = probe[probe_name]
                require(
                    signed["orientation"] == orientation
                    and signed["model_forward_count"] == F_SIGNED,
                    "SIGNED_META",
                )
                require(
                    float(signed["F"])
                    == float(signed["plus_path_efficiency"])
                    - float(signed["minus_path_efficiency"]),
                    "F_ID",
                )

                for role, branch_sign in (("tp", 1), ("tm", -1)):
                    audit = signed["branch_audits"][role]
                    require(
                        audit["condition"] == condition["condition"]
                        and audit["orientation"] == orientation
                        and audit["branch_sign"] == branch_sign,
                        "AUDIT_COORDINATE",
                    )
                    require(
                        audit["coefficient_source"]
                        == "native_pp3_coordinates",
                        "AUDIT_SOURCE",
                    )

                    finite_keys = (
                        "native_pp3_a",
                        "native_pp3_b",
                        "pp3_component_l2",
                        "pp5_component_l2",
                        "restoration_addition_norm_mismatch",
                        "direct_final_state_correction_l2",
                        "pp3_post_condition_residual_plus",
                        "pp3_post_condition_residual_minus",
                        "probe_correction_l2",
                        "applied_correction_max_abs_residual",
                    )
                    require(
                        all(
                            math.isfinite(float(audit[k]))
                            for k in finite_keys
                        ),
                        "AUDIT_FINITE",
                    )
                    require(
                        abs(
                            float(audit["probe_correction_l2"])
                            - EPS
                        )
                        <= TOL,
                        "AUDIT_PROBE_L2",
                    )
                    require(
                        float(
                            audit[
                                "restoration_addition_norm_mismatch"
                            ]
                        )
                        <= TOL,
                        "AUDIT_RESTORATION_NORM",
                    )
                    require(
                        float(
                            audit[
                                "applied_correction_max_abs_residual"
                            ]
                        )
                        <= cast_tol,
                        "AUDIT_RESID",
                    )

                    if condition["condition"] == "pp3_neutralized":
                        require(
                            abs(
                                float(
                                    audit[
                                        "pp3_post_condition_residual_plus"
                                    ]
                                )
                            )
                            <= TOL
                            and abs(
                                float(
                                    audit[
                                        "pp3_post_condition_residual_minus"
                                    ]
                                )
                            )
                            <= TOL,
                            "NEUTRAL_AUDIT",
                        )
                        require(
                            audit[
                                "neutralized_construction_max_abs_residual"
                            ]
                            is not None
                            and float(
                                audit[
                                    "neutralized_construction_max_abs_residual"
                                ]
                            )
                            <= TOL,
                            "NEUTRAL_CONSTRUCTION",
                        )
                    elif condition["condition"] == "pp3_restored":
                        require(
                            float(
                                audit[
                                    "direct_final_state_correction_l2"
                                ]
                            )
                            == 0.0,
                            "R3_DIRECT_CORRECTION",
                        )
                        require(
                            audit["r3_native_state_max_abs_residual"]
                            is not None
                            and float(
                                audit[
                                    "r3_native_state_max_abs_residual"
                                ]
                            )
                            <= TOL,
                            "R3_IDENTITY",
                        )
                    else:
                        require(
                            audit["r5_construction_max_abs_residual"]
                            is not None
                            and float(
                                audit[
                                    "r5_construction_max_abs_residual"
                                ]
                            )
                            <= TOL,
                            "R5_IDENTITY",
                        )

            f_plus = float(probe["F_plus"])
            f_minus = float(probe["F_minus"])
            j_value = float(probe["J"])
            require(
                j_value == (f_plus - f_minus) / (2 * EPS)
                and float(probe["J_squared"])
                == j_value * j_value,
                "J_ID",
            )

        e_xg2 = sum(
            float(probe["J_squared"])
            for probe in probes[:K]
        ) / K
        e_xg4 = sum(
            float(probe["J_squared"])
            for probe in probes[K:]
        ) / K
        require(
            float(condition["E_XG2"]) == e_xg2
            and float(condition["E_XG4"]) == e_xg4
            and float(condition["Q"]) == e_xg2 - e_xg4,
            "Q_ID",
        )

    validate_matching(item)
    validate_endpoint(item)


def validate_items(items) -> None:
    require(len(items) == N, "ITEM_COUNT")
    for index, (pair, item) in enumerate(
        zip(expected_pairs(), items, strict=True)
    ):
        validate_item(item, pair, index)


def canonical(value) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def jsonl(rows) -> bytes:
    return b"".join(canonical(row) for row in rows)


def write_outputs(out: Path, items, summary) -> None:
    require(not out.exists(), "OUTPUT_COLLISION")
    validate_items(items)
    out.mkdir(parents=True)

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
                "bytes": (out / name).stat().st_size,
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


def validate_artifact(out: Path):
    manifest = json.loads(
        (out / MANIFEST_FILE).read_text(encoding="utf-8-sig")
    )
    require(
        manifest["schema_version"] == MANIFEST_SCHEMA,
        "MANIFEST",
    )

    hashes: dict[str, str] = {}
    for name in (ITEM_FILE, SUMMARY_FILE):
        path = out / name
        digest = sha256_file(path)
        require(
            digest == manifest["files"][name]["sha256"]
            and path.stat().st_size
            == manifest["files"][name]["bytes"],
            f"FILE:{name}",
        )
        hashes[name] = digest

    hashes[MANIFEST_FILE] = sha256_file(out / MANIFEST_FILE)

    observed: dict[str, str] = {}
    for line in (
        out / CHECKSUM_FILE
    ).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            observed[name] = digest

    require(
        observed
        == {
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
        "SUMMARY",
    )
    require(
        summary["source_pair_count"] == N
        and summary["pair_id_first"] == "xg1_fact_901"
        and summary["pair_id_last"] == "xg1_fact_1200",
        "SUMMARY_POP",
    )
    require(
        bool(summary.get("execution_head")),
        "SUMMARY_EXECUTION_HEAD",
    )
    require(
        summary.get("implementation_authority_commit")
        == AUTHORITY_COMMIT,
        "SUMMARY_AUTHORITY",
    )
    require(
        summary.get("static_preparation_freeze_commit")
        == STATIC_COMMIT,
        "SUMMARY_STATIC_FREEZE",
    )
    require(float(summary["epsilon"]) == EPS, "SUMMARY_EPSILON")
    require(
        summary.get("condition_order") == list(CONDITIONS),
        "SUMMARY_CONDITION_ORDER",
    )
    require(
        summary.get("direction_order") == list(DIRECTIONS),
        "SUMMARY_DIRECTION_ORDER",
    )
    require(
        summary.get("model_forwards_per_direction") == F_DIR,
        "SUMMARY_FORWARD_DIRECTION",
    )
    require(
        summary.get("model_forwards_per_condition") == F_COND,
        "SUMMARY_FORWARD_CONDITION",
    )
    require(
        summary.get("model_forwards_per_pair") == F_PAIR,
        "SUMMARY_FORWARD_PAIR",
    )
    require(
        summary.get("representative_checkpoint_sha256")
        == holdout.phase1.base.prevalence_eq.extraction
        .REPRESENTATIVE_CHECKPOINT_SHA256,
        "SUMMARY_CHECKPOINT",
    )
    require(
        summary["scientific_model_forward_count_this_run"]
        == F_TOTAL
        and summary["baseline_model_forward_count_this_run"] == 0,
        "SUMMARY_BUDGET",
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
    require(
        summary["primary_endpoint_definition"]
        == "D_SUF=(Q_R3-Q_B)-(Q_R5-Q_B)=Q_R3-Q_R5",
        "SUMMARY_ENDPOINT",
    )

    return {
        "items": items,
        "summary": summary,
        "manifest": manifest,
    }


def run_observation(
    *,
    expected_head,
    model_snapshot,
    tokenizer_snapshot,
    checkpoint_path,
    output_dir,
):
    authenticate_repo(expected_head)
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    planes = load_planes()
    bases = load_bases()

    runtime = holdout.phase1.base.prevalence_eq
    runtime.backend.runtime_gate()

    with runtime.backend.parent_runtime_rebind():
        rows, encoded, event_rows = load_inputs(tokenizer_snapshot)
        pairs = pair_order(rows)

        parent = runtime.parent
        events = parent.event_lookup(event_rows)
        row_index = parent.build_row_index(rows)

        trace_code, trace_line = (
            runtime.measurement._resolve_and_validate_runtime_binding()
        )
        kernels = runtime.kernel_compat.load_exact_fast_kernels()

        with runtime.kernel_compat.exact_transformers_kernel_loader(
            kernels
        ) as calls:
            model, checkpoint_sha = (
                parent.load_representative_model_external(
                    model_snapshot=model_snapshot,
                    checkpoint_path=checkpoint_path,
                )
            )
            require(
                checkpoint_sha
                == runtime.extraction
                .REPRESENTATIVE_CHECKPOINT_SHA256,
                "CHECKPOINT",
            )
            runtime_ctx = (
                runtime.transport_runtime.validate_runtime_components(
                    model
                )
            )

        constructor_counts = Counter(calls)
        require(
            set(constructor_counts)
            == {"causal-conv1d", "mamba-ssm"}
            and constructor_counts["causal-conv1d"] > 0
            and constructor_counts["causal-conv1d"]
            == constructor_counts["mamba-ssm"],
            "KERNEL_CONSTRUCTOR",
        )

        runtime.kernel_compat.validate_transformers_kernel_bindings(
            kernels
        )

        model.to(torch.device("cuda:0"))
        model.eval()

        fast_capture = runtime.backend._make_fast_capture(kernels)
        original_capture = parent.capture_branch
        budget = parent.ForwardBudget(F_TOTAL)
        items = []
        parent.capture_branch = fast_capture

        try:
            for index, pair in enumerate(pairs):
                items.append(
                    run_pair(
                        probe_seed(index, pair, events),
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
            torch.cuda.synchronize()
        finally:
            parent.capture_branch = original_capture

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "implementation_authority_commit": AUTHORITY_COMMIT,
        "static_preparation_freeze_commit": STATIC_COMMIT,
        "source_pair_count": N,
        "pair_id_first": items[0]["source_pair_id"],
        "pair_id_last": items[-1]["source_pair_id"],
        "epsilon": EPS,
        "condition_order": list(CONDITIONS),
        "direction_order": list(DIRECTIONS),
        "model_forwards_per_direction": F_DIR,
        "model_forwards_per_condition": F_COND,
        "model_forwards_per_pair": F_PAIR,
        "scientific_model_forward_count_this_run": F_TOTAL,
        "baseline_model_forward_count_this_run": 0,
        "primary_endpoint_definition":
            "D_SUF=(Q_R3-Q_B)-(Q_R5-Q_B)=Q_R3-Q_R5",
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
            "Frozen PP3 restoration-sufficiency raw observation; "
            "no statistical inference."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
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
    print("RESULT=" + summary["result"])
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(summary["scientific_model_forward_count_this_run"])
    )
    print("BASELINE_MODEL_FORWARD_COUNT_THIS_RUN=0")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("SCIENTIFIC_CONCLUSION=None")


if __name__ == "__main__":
    main()
