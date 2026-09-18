from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import shutil
import subprocess
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as tokenizer_gate
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter
from scripts import reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda as holdout


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"

DESIGN_COMMIT = "d3cc008fad221862e6fe9718b67b6ba0c87d0368"
STATIC_COMMIT = "6d65de694f112b051366d8ccf7e4617caafba1b6"
DESIGN_PATH = Path(
    "reports/reason_router_gen4_pp3_excluded_residual_template_transport_design.md"
)
DESIGN_BLOB = "8b99238182027191fa9d5f3014774ce7b2e9a342"

DATA_ROOT = Path("data/reason_router_gen4_xg1_residual_template_transport_v1")
STATIC_ROOT = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_template_transport_"
    "static_preparation_d3cc008"
)

SOURCE_SHA = "9b93eff2399b7f90fb3d63f28f834fb986efcac9d347038c4b265519062d71a6"
ROWS_SHA = "f60cc028e8276279499290976f18ba503fc7a35a4232c1d768b822075d87ec5e"
STRUCT_SHA = "29a94032474bc92e277247fd5d05f404901b647e1ffab961115c198b7f3b5f61"
ANCHOR_SHA = "74f6a16fe9bdf6da43eda1e4ab8bd8893c2eec86d5ead57394726a0d911eda60"
ELIG_SHA = "70d4e7c804cf0bb5ff4e73a575bb8993ea80cfe7ad3387a4eadf7baef5b0b6fe"
TEMPLATE_MANIFEST_SHA = (
    "8bfec35f48775d9ee91a4151ddf08dbf01aaa062bc9ebe983fd83c90fc857a7d"
)
PREPARATION_SHA = (
    "b54fcf107ed8fef1cf688a2652696d219c856939d58c547d1460dca5d994f717"
)

PLAN_SHA = {
    "xg2": "b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c",
    "xg4": "792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f",
}

EXPECTED_EIGENVALUES = (
    0.87061814189182785,
    0.94755022112376275,
    0.98692852916688512,
    0.99848952673382474,
    0.99986792842854511,
)
RESIDUAL_PLANES = ("P1", "P2", "P4", "P5")
RESIDUAL_INDICES = (0, 1, 3, 4)

XG2_UNIT_TEMPLATE = (
    0.4007549782451175,
    0.5213470856800944,
    -0.022043162722820656,
    0.7530649126348957,
)
XG4_UNIT_TEMPLATE = (
    0.14700867790791192,
    -0.17145767505443896,
    0.9138296650882098,
    0.337499714799086,
)
TEMPLATE_COSINE = 0.2035409975407082

N = 300
ROWS = 1800
DIM = 395
K = 5
EPS = 0.025

DIRECTIONS = tuple(
    [f"xg2_{i}" for i in range(K)]
    + [f"xg4_{i}" for i in range(K)]
)
F_SIGNED = 2
F_DIR = 4
F_PAIR = 40
F_TOTAL = 12000

GPU_COUNT = 2
SHARDS = (
    {
        "shard_id": 0,
        "gpu_id": 0,
        "start_index": 0,
        "end_index": 150,
        "pair_first": "xg1_fact_1201",
        "pair_last": "xg1_fact_1350",
        "pair_count": 150,
        "forward_budget": 6000,
    },
    {
        "shard_id": 1,
        "gpu_id": 1,
        "start_index": 150,
        "end_index": 300,
        "pair_first": "xg1_fact_1351",
        "pair_last": "xg1_fact_1500",
        "pair_count": 150,
        "forward_budget": 6000,
    },
)

TOL = 1.0e-12
Q_RECON_TOL = 2.0e-18

ITEM_FILE = "pp3_excluded_residual_template_transport_items.jsonl"
SUMMARY_FILE = "pp3_excluded_residual_template_transport_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen4-pp3-excluded-residual-template-transport-item-v1"
SUMMARY_SCHEMA = "gen4-pp3-excluded-residual-template-transport-summary-v1"
MANIFEST_SCHEMA = "gen4-pp3-excluded-residual-template-transport-manifest-v1"
RESULT_PASS = "PASS_PP3_EXCLUDED_RESIDUAL_TEMPLATE_TRANSPORT_RAW_OBSERVATION"


class ResidualTemplateTransportError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ResidualTemplateTransportError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
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
        raise ResidualTemplateTransportError(
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
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_NOT_ANCESTOR")

    blob = git("rev-parse", f"HEAD:{DESIGN_PATH.as_posix()}")
    require(blob == DESIGN_BLOB, f"DESIGN_BLOB_DRIFT:{blob}")


def expected_pairs() -> tuple[str, ...]:
    return tuple(f"xg1_fact_{i:03d}" for i in range(1201, 1501))


def validate_shards() -> None:
    require(len(SHARDS) == GPU_COUNT == 2, "SHARD_COUNT")
    covered: list[int] = []
    for expected_id, shard in enumerate(SHARDS):
        require(shard["shard_id"] == expected_id, "SHARD_ID")
        require(shard["gpu_id"] == expected_id, "SHARD_GPU")
        require(
            shard["end_index"] - shard["start_index"] == shard["pair_count"],
            "SHARD_PAIR_COUNT",
        )
        require(
            shard["pair_count"] * F_PAIR == shard["forward_budget"],
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
        covered.extend(range(shard["start_index"], shard["end_index"]))

    require(covered == list(range(N)), "SHARD_COVERAGE")
    require(
        sum(int(s["forward_budget"]) for s in SHARDS) == F_TOTAL,
        "TOTAL_FORWARD_BUDGET",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        rows.append(value)
    return rows


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
        STATIC_ROOT / "template_manifest.json": TEMPLATE_MANIFEST_SHA,
        STATIC_ROOT / "preparation_manifest.json": PREPARATION_SHA,
    }
    for path, expected in checks.items():
        full = ROOT / path
        require(full.is_file(), f"MISSING:{path}")
        require(sha256_file(full) == expected, f"SHA:{path}")

    preparation = json.loads(
        (ROOT / STATIC_ROOT / "preparation_manifest.json")
        .read_text(encoding="utf-8-sig")
    )
    require(
        preparation["result"]
        == "PASS_PP3_EXCLUDED_RESIDUAL_TEMPLATE_TRANSPORT_STATIC_PREPARATION",
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
    contract = preparation["future_execution_contract"]
    require(
        contract["scientific_model_forward_budget"] == F_TOTAL
        and contract["planned_gpu_count"] == GPU_COUNT
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
        and eligibility["pair_id_first"] == "xg1_fact_1201"
        and eligibility["pair_id_last"] == "xg1_fact_1500"
        and eligibility["model_forward_count"] == 0
        and eligibility["checkpoint_load_count"] == 0
        and eligibility["gpu_used"] is False,
        "ELIGIBILITY_BOUNDARY",
    )

    template = json.loads(
        (ROOT / STATIC_ROOT / "template_manifest.json")
        .read_text(encoding="utf-8-sig")
    )
    require(
        template["result"] == "PASS_FROZEN_RESIDUAL_TEMPLATES"
        and template["residual_plane_order"] == list(RESIDUAL_PLANES)
        and template["xg2_unit_template"] == list(XG2_UNIT_TEMPLATE)
        and template["xg4_unit_template"] == list(XG4_UNIT_TEMPLATE)
        and float(template["template_cosine"]) == TEMPLATE_COSINE
        and template["xg1_outcome_used_for_template_construction"] is False,
        "TEMPLATE_BOUNDARY",
    )
    return {
        "preparation": preparation,
        "eligibility": eligibility,
        "template": template,
    }


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
        gram = basis.T @ basis
        require(
            float(
                torch.max(
                    torch.abs(
                        gram - torch.eye(K, dtype=torch.float64)
                    )
                )
            )
            <= TOL,
            f"BASIS_ORTHONORMAL:{family}",
        )
        out[family] = basis
    return out


def build_principal_geometry(
    bases: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor | list[float]]:
    b2 = bases["xg2"]
    b4 = bases["xg4"]
    union = torch.cat([b2, b4], dim=1)
    union_gram = union.T @ union
    require(
        int(torch.linalg.matrix_rank(union_gram)) == 2 * K,
        "UNION_RANK",
    )

    contrast = b2 @ b2.T - b4 @ b4.T
    evals, evecs = torch.linalg.eigh(contrast)
    pos_idx = [i for i, x in enumerate(evals.tolist()) if x > 1e-10]
    neg_idx = [i for i, x in enumerate(evals.tolist()) if x < -1e-10]
    require(len(pos_idx) == K and len(neg_idx) == K, "CONTRAST_RANK")

    pos_idx = sorted(pos_idx, key=lambda i: float(evals[i]))
    neg_idx = sorted(neg_idx, key=lambda i: abs(float(evals[i])))

    lam_pos = [float(evals[i]) for i in pos_idx]
    lam_neg = [abs(float(evals[i])) for i in neg_idx]
    for i, expected in enumerate(EXPECTED_EIGENVALUES):
        require(
            abs(lam_pos[i] - expected) <= 2e-12
            and abs(lam_neg[i] - expected) <= 2e-12
            and abs(lam_pos[i] - lam_neg[i]) <= 2e-12,
            f"EIGENVALUE:P{i+1}",
        )

    return {
        "union": union.contiguous(),
        "union_gram": union_gram.contiguous(),
        "plus": torch.stack([evecs[:, i] for i in pos_idx], dim=1)
        .contiguous(),
        "minus": torch.stack([evecs[:, i] for i in neg_idx], dim=1)
        .contiguous(),
        "lambda": lam_pos,
    }


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
    events = runtime.parent.event_lookup(event_rows)
    runtime.parent.validate_transport_event_plan(pairs, events)

    require(
        list(encoded["source_pair_id"])
        == [str(row["source_pair_id"]) for row in rows],
        "ENCODED_PAIR_ORDER",
    )
    return rows, encoded, event_rows


def apply_probe_hook(
    output: torch.Tensor,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
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
    require(orientation in {-1, 1}, "ORIENTATION")
    require(branch_sign in {-1, 1}, "BRANCH_SIGN")

    mask = strong_mask.detach().cpu().bool().contiguous()
    require(
        mask.numel() == core.INTERMEDIATE_SIZE
        and int(mask.sum()) == DIM,
        "MASK",
    )

    vector = direction.detach().cpu().to(torch.float64).contiguous()
    require(
        tuple(vector.shape) == (DIM,)
        and abs(float(torch.linalg.vector_norm(vector)) - 1.0) <= TOL,
        "DIRECTION",
    )

    probe = vector * (
        float(branch_sign) * float(orientation) * EPS
    )

    before = output.detach().clone()
    out = output.clone()
    mask_device = mask.to(out.device)
    intended = probe.to(device=out.device, dtype=out.dtype)

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
    audit.update({
        "token_index": int(token_index),
        "orientation": int(orientation),
        "branch_sign": int(branch_sign),
        "probe_correction_l2": float(torch.linalg.vector_norm(probe)),
        "applied_correction_max_abs_residual": residual,
        "native_state_direct_correction_applied": False,
    })
    return out


def install_hook(mixer17: Any, **kwargs):
    def hook(_module, _args, output):
        return apply_probe_hook(output, **kwargs)
    return mixer17.in_proj.register_forward_hook(hook)


def input_row(encoded, row_index, pair, cell):
    return holdout.phase2._input_row(
        encoded,
        row_index,
        pair,
        cell,
    )


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


def run_signed(
    seed,
    direction,
    *,
    orientation,
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
    family,
    index,
    **kwargs,
):
    positive = run_signed(
        seed,
        direction,
        orientation=1,
        **kwargs,
    )
    negative = run_signed(
        seed,
        direction,
        orientation=-1,
        **kwargs,
    )
    f_plus = float(positive["F"])
    f_minus = float(negative["F"])
    j_value = (f_plus - f_minus) / (2.0 * EPS)

    return {
        "direction_key": f"{family}_{index}",
        "basis_family": family,
        "basis_index": int(index),
        "F_plus": f_plus,
        "F_minus": f_minus,
        "J": j_value,
        "J_squared": j_value * j_value,
        "positive_probe": positive,
        "negative_probe": negative,
        "model_forward_count": F_DIR,
    }


def decompose_endpoint(
    probes: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        [str(p["direction_key"]) for p in probes] == list(DIRECTIONS),
        "DECOMP_DIRECTION_ORDER",
    )
    j2 = [float(p["J"]) for p in probes[:K]]
    j4 = [float(p["J"]) for p in probes[K:]]
    y = torch.tensor(j2 + j4, dtype=torch.float64)

    union = geometry["union"]
    union_gram = geometry["union_gram"]
    g_union = union @ torch.linalg.solve(union_gram, y)

    plus = geometry["plus"]
    minus = geometry["minus"]
    lam = torch.tensor(geometry["lambda"], dtype=torch.float64)

    z_plus = plus.T @ g_union
    z_minus = minus.T @ g_union
    plane_net = (
        lam * (z_plus.square() - z_minus.square()) / K
    ).tolist()

    e_xg2 = math.fsum(x * x for x in j2) / K
    e_xg4 = math.fsum(x * x for x in j4) / K
    q = e_xg2 - e_xg4
    q_reconstructed = math.fsum(float(x) for x in plane_net)
    q_residual = q_reconstructed - q
    require(
        abs(q_residual) <= Q_RECON_TOL,
        f"Q_RECONSTRUCTION:{q_residual}",
    )

    residual = [float(plane_net[i]) for i in RESIDUAL_INDICES]
    residual_norm = math.sqrt(math.fsum(x * x for x in residual))
    require(
        residual_norm > 0.0 and math.isfinite(residual_norm),
        "ZERO_OR_NONFINITE_RESIDUAL_NORM",
    )

    c_xg2 = (
        math.fsum(
            x * t
            for x, t in zip(
                residual,
                XG2_UNIT_TEMPLATE,
                strict=True,
            )
        )
        / residual_norm
    )
    c_xg4 = (
        math.fsum(
            x * t
            for x, t in zip(
                residual,
                XG4_UNIT_TEMPLATE,
                strict=True,
            )
        )
        / residual_norm
    )
    d_template = c_xg2 - c_xg4

    require(
        all(
            math.isfinite(x)
            for x in (
                e_xg2,
                e_xg4,
                q,
                q_reconstructed,
                q_residual,
                residual_norm,
                c_xg2,
                c_xg4,
                d_template,
            )
        ),
        "ENDPOINT_NONFINITE",
    )

    return {
        "E_XG2": e_xg2,
        "E_XG4": e_xg4,
        "Q": q,
        "principal_plane_net": {
            f"P{i+1}": float(plane_net[i])
            for i in range(K)
        },
        "Q_reconstructed": q_reconstructed,
        "Q_reconstruction_residual": q_residual,
        "residual_plane_order": list(RESIDUAL_PLANES),
        "residual_net_vector": residual,
        "residual_norm": residual_norm,
        "C_XG2": c_xg2,
        "C_XG4": c_xg4,
        "D_TEMPLATE": d_template,
    }


def run_pair(seed, *, bases, geometry, **kwargs):
    probes = []
    for family in ("xg2", "xg4"):
        for index in range(K):
            probes.append(
                run_direction(
                    seed,
                    bases[family][:, index],
                    family=family,
                    index=index,
                    **kwargs,
                )
            )
    require(
        [p["direction_key"] for p in probes] == list(DIRECTIONS),
        "DIRECTION_ORDER",
    )

    endpoint = decompose_endpoint(probes, geometry)
    item = {
        **seed,
        "schema_version": ITEM_SCHEMA,
        "design_commit": DESIGN_COMMIT,
        "static_preparation_freeze_commit": STATIC_COMMIT,
        "epsilon": EPS,
        "direction_order": list(DIRECTIONS),
        "direction_probes": probes,
        **endpoint,
        "baseline_model_forward_count_this_run": 0,
        "scientific_model_forward_count_this_run": F_PAIR,
        "primary_inference_executed": False,
        "scientific_conclusion": None,
    }
    validate_item(
        item,
        expected_pairs()[int(seed["pair_index"])],
        int(seed["pair_index"]),
    )
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
        item["design_commit"] == DESIGN_COMMIT
        and item["static_preparation_freeze_commit"] == STATIC_COMMIT,
        f"ITEM_PROVENANCE:{index}",
    )
    require(float(item["epsilon"]) == EPS, f"ITEM_EPSILON:{index}")
    require(
        item["direction_order"] == list(DIRECTIONS),
        f"ITEM_DIRECTION_ORDER:{index}",
    )
    require(
        item["scientific_model_forward_count_this_run"] == F_PAIR
        and item["baseline_model_forward_count_this_run"] == 0,
        f"ITEM_BUDGET:{index}",
    )
    require(
        item["primary_inference_executed"] is False
        and item["scientific_conclusion"] is None,
        f"ITEM_BOUNDARY:{index}",
    )

    probes = item["direction_probes"]
    require(len(probes) == 10, f"PROBE_COUNT:{index}")
    runtime = holdout.phase1.base.prevalence_eq
    cast_tol = runtime.transport_runtime.RUNTIME_CAST_TOL

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

        for orientation, name in (
            (1, "positive_probe"),
            (-1, "negative_probe"),
        ):
            signed = probe[name]
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
                    audit["orientation"] == orientation
                    and audit["branch_sign"] == branch_sign
                    and audit["native_state_direct_correction_applied"]
                    is False,
                    "AUDIT_COORDINATE",
                )
                require(
                    abs(float(audit["probe_correction_l2"]) - EPS)
                    <= TOL,
                    "AUDIT_PROBE_L2",
                )
                require(
                    float(audit["applied_correction_max_abs_residual"])
                    <= cast_tol,
                    "AUDIT_RESIDUAL",
                )

    j2 = [float(p["J"]) for p in probes[:K]]
    j4 = [float(p["J"]) for p in probes[K:]]
    e2 = math.fsum(x * x for x in j2) / K
    e4 = math.fsum(x * x for x in j4) / K
    require(
        float(item["E_XG2"]) == e2
        and float(item["E_XG4"]) == e4
        and float(item["Q"]) == e2 - e4,
        "Q_ID",
    )

    plane = item["principal_plane_net"]
    require(
        list(plane.keys()) == [f"P{i+1}" for i in range(K)],
        "PLANE_KEYS",
    )
    q_reconstructed = math.fsum(float(plane[f"P{i+1}"]) for i in range(K))
    require(
        float(item["Q_reconstructed"]) == q_reconstructed
        and float(item["Q_reconstruction_residual"])
        == q_reconstructed - float(item["Q"])
        and abs(float(item["Q_reconstruction_residual"]))
        <= Q_RECON_TOL,
        "Q_RECON_ID",
    )

    expected_residual = [
        float(plane["P1"]),
        float(plane["P2"]),
        float(plane["P4"]),
        float(plane["P5"]),
    ]
    require(
        item["residual_plane_order"] == list(RESIDUAL_PLANES)
        and [float(x) for x in item["residual_net_vector"]]
        == expected_residual,
        "RESIDUAL_VECTOR",
    )
    norm_value = math.sqrt(math.fsum(x * x for x in expected_residual))
    require(
        norm_value > 0.0
        and float(item["residual_norm"]) == norm_value,
        "RESIDUAL_NORM",
    )
    c2 = (
        math.fsum(
            x * t
            for x, t in zip(
                expected_residual,
                XG2_UNIT_TEMPLATE,
                strict=True,
            )
        )
        / norm_value
    )
    c4 = (
        math.fsum(
            x * t
            for x, t in zip(
                expected_residual,
                XG4_UNIT_TEMPLATE,
                strict=True,
            )
        )
        / norm_value
    )
    require(
        float(item["C_XG2"]) == c2
        and float(item["C_XG4"]) == c4
        and float(item["D_TEMPLATE"]) == c2 - c4,
        "TEMPLATE_ENDPOINT",
    )


def validate_items(items: Sequence[Mapping[str, Any]]) -> None:
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
        )
        == shard["forward_budget"],
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
    path = temp_dir / f"shard_{shard_id}.json"
    path.write_bytes(canonical(payload))


def read_shard_payload(temp_dir: Path, shard_id: int) -> dict[str, Any]:
    path = temp_dir / f"shard_{shard_id}.json"
    require(path.is_file(), f"SHARD_PAYLOAD_MISSING:{shard_id}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"SHARD_PAYLOAD_OBJECT:{shard_id}")
    return value



def runtime_gate_for_device(runtime: Any, gpu_id: int) -> None:
    backend = runtime.backend
    import transformers

    observed = {
        "python": backend.platform.python_version(),
        "numpy": backend.np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    require(
        observed == backend.EXPECTED_RUNTIME,
        f"RUNTIME_MISMATCH:{observed}",
    )
    require(
        runtime.kernel_compat._kernel_package_version()
        == backend.KERNELS_VERSION,
        "KERNELS_VERSION",
    )
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= GPU_COUNT, "CUDA_DEVICE_COUNT")
    require(torch.version.cuda == backend.EXPECTED_CUDA_RUNTIME, "CUDA_RUNTIME")
    require(0 <= gpu_id < torch.cuda.device_count(), "GPU_ID_RANGE")
    torch.cuda.set_device(gpu_id)
    require(
        torch.cuda.get_device_name(gpu_id) == backend.EXPECTED_DEVICE_NAME,
        f"CUDA_DEVICE_NAME:{gpu_id}",
    )
    require(
        tuple(torch.cuda.get_device_capability(gpu_id))
        == backend.EXPECTED_CAPABILITY,
        f"CUDA_CAPABILITY:{gpu_id}",
    )

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_fast_capture_for_device(
    runtime: Any,
    kernels: Mapping[str, Any],
    device: torch.device,
):
    import transformers.models.mamba.modeling_mamba as mm

    backend = runtime.backend
    core = runtime.core
    parent = runtime.parent
    extraction = runtime.extraction
    transport_runtime = runtime.transport_runtime
    kernel_scan = kernels["selective_scan_fn"]
    kernel_update = kernels["selective_state_update"]

    def capture_branch(
        model: Any,
        runtime_ctx: Mapping[str, Any],
        *,
        trace_code: Any,
        trace_line: int,
        input_ids: torch.Tensor,
        anchor: int,
        budget: Any,
        capture_states: bool,
        delta_h: Any | None = None,
        plus_branch: bool | None = None,
    ) -> dict[str, Any]:
        del trace_code, trace_line

        target_abs = int(anchor) + core.TARGET_OFFSET
        require(
            tuple(input_ids.shape)
            == (1, extraction.MAX_MODEL_SEQUENCE_LENGTH),
            "INPUT_SHAPE",
        )
        input_ids = input_ids.detach().to(device).contiguous()

        layer15 = runtime_ctx["layer15"]
        norm17 = runtime_ctx["norm17"]
        mixer17 = runtime_ctx["mixer17"]

        holders: dict[str, Any] = {}
        counts = {"r15": 0, "y15": 0, "r17": 0, "x17": 0}

        def take(full: Any, label: str) -> torch.Tensor:
            value = parent._finite_tensor(full, label)
            require(
                0 <= target_abs < value.shape[1],
                f"{label}_TARGET_RANGE",
            )
            return value[0, target_abs, :].contiguous().clone()

        def layer15_pre(_module, args):
            counts["r15"] += 1
            require(
                counts["r15"] == 1 and len(args) >= 1,
                "R15_HOOK",
            )
            holders["R"] = take(args[0], "R15_FULL")

        def layer15_post(_module, _args, output):
            counts["y15"] += 1
            require(counts["y15"] == 1, "Y15_HOOK")
            holders["Y"] = take(output, "Y15_FULL")

        def norm17_pre(_module, args):
            counts["r17"] += 1
            require(
                counts["r17"] == 1 and len(args) == 1,
                "R17_HOOK",
            )
            holders["R17"] = take(args[0], "R17_FULL")

        def norm17_post(_module, _args, output):
            counts["x17"] += 1
            require(counts["x17"] == 1, "X17_HOOK")
            holders["X"] = take(output, "X17_FULL")

        handles = [
            layer15.register_forward_pre_hook(layer15_pre),
            layer15.mixer.register_forward_hook(layer15_post),
            norm17.register_forward_pre_hook(norm17_pre),
            norm17.register_forward_hook(norm17_post),
        ]

        intervention_audit = None
        if delta_h is not None:
            require(
                plus_branch is not None,
                "INTERVENTION_BRANCH_REQUIRED",
            )
            intervention_audit = {}
            handles.append(
                transport_runtime.install_inproj_hook(
                    mixer17,
                    token_index=target_abs,
                    strong_mask=runtime_ctx["strong_mask"],
                    delta_h=delta_h,
                    plus_branch=bool(plus_branch),
                    audit=intervention_audit,
                )
            )

        active = {"value": False}
        captured: list[tuple[torch.Tensor, ...]] = []

        def scan_wrapper(*args, **kwargs):
            if active["value"]:
                require(len(captured) == 0, "LAYER17_SCAN_DUPLICATE")
                require(len(args) >= 8, "SCAN_ARG_COUNT")
                captured.append(
                    tuple(v.detach().clone() for v in args[:8])
                )
            return kernel_scan(*args, **kwargs)

        original_scan = mm.selective_scan_fn
        original_cuda = mixer17.cuda_kernels_forward

        def cuda_wrapper(
            _self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        ):
            require(not active["value"], "LAYER17_ACTIVE_REENTRY")
            active["value"] = True
            try:
                return original_cuda(
                    hidden_states,
                    cache_params,
                    cache_position,
                    attention_mask,
                )
            finally:
                active["value"] = False

        mm.selective_scan_fn = scan_wrapper
        mixer17.cuda_kernels_forward = backend.types.MethodType(
            cuda_wrapper,
            mixer17,
        )

        budget.consume()
        try:
            model.mamba.eval()
            with torch.inference_mode():
                _ = model.mamba(input_ids=input_ids)
            torch.cuda.synchronize(device)
        finally:
            for handle in reversed(handles):
                handle.remove()
            mm.selective_scan_fn = original_scan
            if "cuda_kernels_forward" in mixer17.__dict__:
                del mixer17.__dict__["cuda_kernels_forward"]

        require(
            counts == {"r15": 1, "y15": 1, "r17": 1, "x17": 1},
            "HOOK_COUNT_FAILURE",
        )
        require(
            set(holders) == {"R", "Y", "R17", "X"},
            "HOOK_CAPTURE_MISSING",
        )
        require(len(captured) == 1, "LAYER17_SCAN_CAPTURE_COUNT")

        r17 = holders["R17"].to(torch.float64)
        eps = float(norm17.variance_epsilon)
        scale = float(torch.rsqrt(r17.pow(2).mean() + eps).item())
        require(math.isfinite(scale) and scale > 0.0, "RMS_SCALE")

        states = None
        if capture_states:
            (
                u,
                delta,
                a_matrix,
                b_scan,
                c_scan,
                d_vector,
                gate,
                delta_bias,
            ) = captured[0]

            prefix_end = int(anchor) + 1
            require(prefix_end + 4 <= u.shape[-1], "FAST_POST4_RANGE")

            _, state = kernel_scan(
                u[..., :prefix_end].contiguous(),
                delta[..., :prefix_end].contiguous(),
                a_matrix,
                b_scan[..., :prefix_end].contiguous(),
                c_scan[..., :prefix_end].contiguous(),
                d_vector,
                gate[..., :prefix_end].contiguous(),
                delta_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            window = [backend._flatten_state(state)]

            for token in range(prefix_end, prefix_end + 4):
                _ = kernel_update(
                    state,
                    u[..., token],
                    delta[..., token],
                    a_matrix,
                    b_scan[..., token],
                    c_scan[..., token],
                    d_vector,
                    gate[..., token],
                    delta_bias,
                    dt_softplus=True,
                )
                window.append(backend._flatten_state(state))

            filler = window[0]
            token_count = int(input_ids.shape[1])
            states = [filler.copy() for _ in range(token_count)]
            for offset, vector in enumerate(window):
                states[int(anchor) + offset] = vector

        if delta_h is not None:
            require(
                intervention_audit is not None and bool(intervention_audit),
                "INTERVENTION_HOOK_NOT_OBSERVED",
            )
            require(
                intervention_audit["token_index"] == target_abs,
                "INTERVENTION_TOKEN_AUDIT",
            )

        return {
            "geometry_branch": {
                "R": holders["R"],
                "Y": holders["Y"],
                "X": holders["X"],
                "rms_scale": scale,
            },
            "states": states,
            "intervention_audit": intervention_audit,
            "anchor": int(anchor),
            "target_abs": target_abs,
        }

    return capture_branch

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
        require(torch.cuda.device_count() >= GPU_COUNT, "CUDA_DEVICE_COUNT")
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        authenticate_repo(expected_head)
        validate_shards()
        validate_static_inputs()

        bases = load_bases()
        geometry = build_principal_geometry(bases)

        runtime = holdout.phase1.base.prevalence_eq
        runtime_gate_for_device(runtime, gpu_id)

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
                    == runtime.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
                    "CHECKPOINT",
                )
                runtime_ctx = (
                    runtime.transport_runtime.validate_runtime_components(
                        model
                    )
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
                            geometry=geometry,
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


def validate_artifact(out: Path) -> dict[str, Any]:
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
        if line.strip():
            digest, name = line.split("  ", 1)
            require(name not in observed, f"CHECKSUM_DUPLICATE:{name}")
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
        "SUMMARY_RESULT",
    )
    require(
        summary["source_pair_count"] == N
        and summary["pair_id_first"] == "xg1_fact_1201"
        and summary["pair_id_last"] == "xg1_fact_1500",
        "SUMMARY_POPULATION",
    )
    require(
        bool(summary["execution_head"])
        and summary["design_commit"] == DESIGN_COMMIT
        and summary["static_preparation_freeze_commit"] == STATIC_COMMIT,
        "SUMMARY_PROVENANCE",
    )
    require(float(summary["epsilon"]) == EPS, "SUMMARY_EPSILON")
    require(
        summary["direction_order"] == list(DIRECTIONS)
        and summary["residual_plane_order"] == list(RESIDUAL_PLANES),
        "SUMMARY_ORDER",
    )
    require(
        summary["model_forwards_per_direction"] == F_DIR
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
    for shard, expected in zip(summary["shards"], SHARDS, strict=True):
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
        len({s["checkpoint_sha256"] for s in summary["shards"]}) == 1
        and summary["representative_checkpoint_sha256"]
        == summary["shards"][0]["checkpoint_sha256"],
        "SUMMARY_CHECKPOINT",
    )
    require(
        summary["representative_checkpoint_sha256"]
        == holdout.phase1.base.prevalence_eq.extraction
        .REPRESENTATIVE_CHECKPOINT_SHA256,
        "SUMMARY_FROZEN_CHECKPOINT",
    )
    require(
        summary["primary_endpoint_definition"]
        == "D_TEMPLATE=C_XG2-C_XG4",
        "SUMMARY_ENDPOINT",
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
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= GPU_COUNT, "CUDA_DEVICE_COUNT")

    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(
        prefix="gen4_residual_template_transport_"
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
                name=f"gen4-residual-template-gpu{shard['gpu_id']}",
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        failures = []
        for shard, process in zip(SHARDS, processes, strict=True):
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
            read_shard_payload(temp_dir, int(shard["shard_id"]))
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
        "source_pair_count": N,
        "pair_id_first": items[0]["source_pair_id"],
        "pair_id_last": items[-1]["source_pair_id"],
        "epsilon": EPS,
        "direction_order": list(DIRECTIONS),
        "residual_plane_order": list(RESIDUAL_PLANES),
        "model_forwards_per_direction": F_DIR,
        "model_forwards_per_pair": F_PAIR,
        "scientific_model_forward_count_this_run": F_TOTAL,
        "baseline_model_forward_count_this_run": 0,
        "gpu_count": GPU_COUNT,
        "parallelization": "independent_pair_shards_spawn",
        "shards": shard_meta,
        "primary_endpoint_definition": "D_TEMPLATE=C_XG2-C_XG4",
        "template_cosine": TEMPLATE_COSINE,
        "xg2_unit_template": list(XG2_UNIT_TEMPLATE),
        "xg4_unit_template": list(XG4_UNIT_TEMPLATE),
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
            "template transport; no statistical inference."
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
    print("RESULT=" + summary["result"])
    print("GPU_COUNT=" + str(summary["gpu_count"]))
    for shard in summary["shards"]:
        print(
            f"SHARD_{shard['shard_id']}_GPU={shard['gpu_id']}:"
            f"{shard['pair_first']}..{shard['pair_last']}:"
            f"FORWARDS={shard['scientific_model_forward_count_this_run']}"
        )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(summary["scientific_model_forward_count_this_run"])
    )
    print("BASELINE_MODEL_FORWARD_COUNT_THIS_RUN=0")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("SCIENTIFIC_CONCLUSION=None")


if __name__ == "__main__":
    main()
