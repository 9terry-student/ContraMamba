#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
AUTHORITY_COMMIT = "e1a091e313a605868dd790968d2fa80ca0bec038"
AUTHORITY_PATH = (
    "reports/reason_router_gen5_m8a_latent_alignment_quotient_"
    "static_audit_authority_spec_candidate.md"
)
SOURCE_HEAD = "286435b586688889f513ea77190ed65a0ba3a2b1"

FACTOR_SEEDS = (6201, 6202, 6203)
DEV_ROWS = 840
VALID_TOKEN_COUNT = 60094
HIDDEN_WIDTH = 768
LATENT_WIDTH = 2
WRITE_WIDTH = 24576

PHASE_A_TRAJECTORY_PATH = Path(
    "reports/reason_router_gen5_ainit_temporal_birth_replay_runs/"
    "gen5-ainit-temporal-birth-phase-a-numerical-auth-d940e19-r1/"
    "temporal_birth_trajectory.pt"
)
PHASE_A_TRAJECTORY_SHA256 = (
    "0f7cd4248faa92223597e0816597b59e426f08829dadd366f9603f56a8de809e"
)

M7_ROOT = Path(
    "reports/reason_router_gen5_ainit_t1_factor_swap_runs/"
    "gen5-m7-t1-factor-swap-c95194f-r1"
)
M7_LOGITS_PATH = M7_ROOT / "factor_swap_logits.pt"
M7_LOGITS_SHA256 = (
    "d5e53f6841a968d9f397294680ef7938e1821858d42e920f3a95d199b1e58262"
)
M7_SUMMARY_PATH = M7_ROOT / "factor_swap_summary.json"
M7_SUMMARY_SHA256 = (
    "421af32a25b2d3bcf75a208cddb090010ae50fb5c0ff46804e67d677056a9cf8"
)
M7B_REPORT_PATH = Path(
    "reports/reason_router_gen5_ainit_t1_factor_swap_"
    "interaction_decomposition_report_candidate.md"
)

TASK_QUOTIENT_ROOT = Path(
    "reports/reason_router_gen5_task_reachable_operator_quotient_runs/"
    "gen5-task-reachable-operator-quotient-c731270-r1"
)
TASK_GRAM_PATH = TASK_QUOTIENT_ROOT / "task_state_gram.pt"
TASK_GRAM_SHA256 = (
    "649cd0208e40a1e0c60d3b4955eaf74131752a9ddf405da502c008fd673edaac"
)
TASK_QUOTIENT_EXECUTION_COMMIT = "c731270221c4e0e131fb68173f40bf0ad8a2bfdd"
DEV_ENCODING_SHA256 = (
    "e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51"
)
DEV_ORDER_SHA256 = (
    "b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25"
)

ALLOWED_IMPLEMENTATION_PATHS = frozenset(
    {
        "scripts/audit_reason_router_gen5_m8a_latent_alignment_quotient.py",
        "tests/test_reason_router_gen5_m8a_latent_alignment_quotient.py",
    }
)

OUTPUT_FILENAMES = (
    "m8a_latent_alignment_quotient_summary.json",
    "m8a_ordered_pair_metrics.jsonl",
    "run_provenance.json",
    "artifact_manifest.json",
)

FLOAT_DTYPE = torch.float64
EPS64 = float(torch.finfo(FLOAT_DTYPE).eps)


class M8AError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M8AError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_bytes(data)
    os.replace(temp, path)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="strict",
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise M8AError("M8A_GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    try:
        return subprocess.call(
            ["git", *args],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise M8AError("M8A_GIT_FAILURE:" + " ".join(args)) from exc


def status_paths() -> set[str]:
    try:
        raw = subprocess.check_output(
            ["git", "status", "--porcelain=v1"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="strict",
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise M8AError("M8A_GIT_STATUS_FAILURE") from exc

    paths: set[str] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        require(len(line) >= 4, f"M8A_MALFORMED_STATUS:{line!r}")
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path.replace("\\", "/"))
    return paths


def cell_name(a_seed: int, r_seed: int) -> str:
    return f"A{a_seed}-R{r_seed}"


def hybrid_name(a_rec: int, a_don: int, r_don: int) -> str:
    return f"AREC{a_rec}-ADON{a_don}-R{r_don}"


def expected_cells() -> tuple[str, ...]:
    return tuple(
        cell_name(a_seed, r_seed)
        for a_seed in FACTOR_SEEDS
        for r_seed in FACTOR_SEEDS
    )


def expected_hybrids() -> tuple[str, ...]:
    return tuple(
        hybrid_name(a_rec, a_don, r_don)
        for a_rec in FACTOR_SEEDS
        for a_don in FACTOR_SEEDS
        for r_don in FACTOR_SEEDS
    )


def ordered_a_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(
        (recipient, donor)
        for recipient in FACTOR_SEEDS
        for donor in FACTOR_SEEDS
        if recipient != donor
    )


def _machine_rank_tolerance(matrix: torch.Tensor) -> float:
    matrix = matrix.to(FLOAT_DTYPE)
    s = torch.linalg.svdvals(matrix)
    if s.numel() == 0:
        return 0.0
    return float(max(matrix.shape) * EPS64 * float(s.max().item()))


def numerical_rank(matrix: torch.Tensor) -> tuple[int, float, torch.Tensor]:
    matrix = matrix.to(FLOAT_DTYPE)
    s = torch.linalg.svdvals(matrix)
    tol = _machine_rank_tolerance(matrix)
    rank = int(torch.count_nonzero(s > tol).item())
    return rank, tol, s


def _pinv_rtol(matrix: torch.Tensor) -> float:
    return float(max(matrix.shape) * EPS64)


def fit_gl_transport(a_rec: torch.Tensor, a_don: torch.Tensor) -> torch.Tensor:
    """Fit donor->recipient 2x2 transport from representation matrices only.

    Solves A_rec ~= T @ A_don in minimum-norm least squares form.
    """
    a_rec = a_rec.to(FLOAT_DTYPE)
    a_don = a_don.to(FLOAT_DTYPE)
    require(tuple(a_rec.shape) == (LATENT_WIDTH, HIDDEN_WIDTH), "M8A_AREC_SHAPE")
    require(tuple(a_don.shape) == (LATENT_WIDTH, HIDDEN_WIDTH), "M8A_ADON_SHAPE")
    solution_t = torch.linalg.lstsq(a_don.T, a_rec.T).solution
    transport = solution_t.T.contiguous()
    require(tuple(transport.shape) == (LATENT_WIDTH, LATENT_WIDTH), "M8A_T_SHAPE")
    require(torch.isfinite(transport).all().item(), "M8A_T_NONFINITE")
    return transport


def fit_orthogonal_transport(
    a_rec: torch.Tensor,
    a_don: torch.Tensor,
) -> torch.Tensor:
    a_rec = a_rec.to(FLOAT_DTYPE)
    a_don = a_don.to(FLOAT_DTYPE)
    cross = a_rec @ a_don.T
    u, _s, vh = torch.linalg.svd(cross, full_matrices=False)
    q = u @ vh
    require(tuple(q.shape) == (LATENT_WIDTH, LATENT_WIDTH), "M8A_Q_SHAPE")
    require(
        torch.allclose(q.T @ q, torch.eye(LATENT_WIDTH, dtype=FLOAT_DTYPE), atol=1e-12, rtol=1e-12),
        "M8A_Q_NOT_ORTHOGONAL",
    )
    return q


def matrix_weighted_sqnorm(
    matrix: torch.Tensor,
    gram: torch.Tensor | None = None,
) -> torch.Tensor:
    matrix = matrix.to(FLOAT_DTYPE)
    if gram is None:
        return torch.sum(matrix * matrix)
    gram = gram.to(FLOAT_DTYPE)
    require(tuple(gram.shape) == (HIDDEN_WIDTH, HIDDEN_WIDTH), "M8A_GRAM_SHAPE")
    value = torch.sum((matrix @ gram) * matrix)
    # Numerical Gram noise may yield a tiny negative value.
    tolerance = max(1.0, float(torch.linalg.norm(matrix).item()) ** 2) * 1e-12
    require(float(value.item()) >= -tolerance, f"M8A_NEGATIVE_WEIGHTED_NORM:{value.item()}")
    return torch.clamp(value, min=0.0)


def normalized_matrix_residual(
    left: torch.Tensor,
    right: torch.Tensor,
    gram: torch.Tensor | None = None,
) -> float:
    left = left.to(FLOAT_DTYPE)
    right = right.to(FLOAT_DTYPE)
    num = matrix_weighted_sqnorm(left - right, gram)
    den_sq = 0.5 * (
        matrix_weighted_sqnorm(left, gram)
        + matrix_weighted_sqnorm(right, gram)
    )
    den = torch.sqrt(torch.clamp(den_sq, min=0.0))
    require(float(den.item()) > 0.0, "M8A_ZERO_MATRIX_RESIDUAL_DENOM")
    return float((torch.sqrt(num) / den).item())


def row_space_geometry(
    a_left: torch.Tensor,
    a_right: torch.Tensor,
) -> dict[str, Any]:
    a_left = a_left.to(FLOAT_DTYPE)
    a_right = a_right.to(FLOAT_DTYPE)
    rank_left, tol_left, s_left = numerical_rank(a_left)
    rank_right, tol_right, s_right = numerical_rank(a_right)
    require(rank_left == LATENT_WIDTH, f"M8A_LEFT_A_RANK:{rank_left}")
    require(rank_right == LATENT_WIDTH, f"M8A_RIGHT_A_RANK:{rank_right}")

    _u_l, _s_l, vh_l = torch.linalg.svd(a_left, full_matrices=False)
    _u_r, _s_r, vh_r = torch.linalg.svd(a_right, full_matrices=False)
    q_left = vh_l[:LATENT_WIDTH]
    q_right = vh_r[:LATENT_WIDTH]
    cosines = torch.linalg.svdvals(q_left @ q_right.T)
    cosines = torch.clamp(cosines, 0.0, 1.0)
    angles = torch.acos(cosines)
    return {
        "left_rank": rank_left,
        "right_rank": rank_right,
        "left_rank_tolerance": tol_left,
        "right_rank_tolerance": tol_right,
        "left_singular_values": [float(x) for x in s_left],
        "right_singular_values": [float(x) for x in s_right],
        "principal_cosines": [float(x) for x in cosines],
        "principal_angles_radians": [float(x) for x in angles],
        "row_space_affinity_mean_sq_cosine": float(torch.mean(cosines.square()).item()),
    }


def transport_diagnostics(transport: torch.Tensor) -> dict[str, Any]:
    transport = transport.to(FLOAT_DTYPE)
    rank, tol, s = numerical_rank(transport)
    determinant = float(torch.linalg.det(transport).item())
    full_rank = rank == LATENT_WIDTH
    condition = math.inf
    if full_rank:
        condition = float((s.max() / s.min()).item())
    return {
        "matrix": [[float(x) for x in row] for row in transport],
        "singular_values": [float(x) for x in s],
        "rank": rank,
        "rank_tolerance": tol,
        "determinant": determinant,
        "condition_number": condition,
        "full_rank": full_rank,
        "pinv_rtol": _pinv_rtol(transport),
    }


def transport_b_to_recipient(
    b_don: torch.Tensor,
    transport_donor_to_recipient: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    b_don = b_don.to(FLOAT_DTYPE)
    transport = transport_donor_to_recipient.to(FLOAT_DTYPE)
    require(tuple(b_don.shape) == (WRITE_WIDTH, LATENT_WIDTH), "M8A_B_SHAPE")
    require(tuple(transport.shape) == (LATENT_WIDTH, LATENT_WIDTH), "M8A_T_B_SHAPE")
    diagnostics = transport_diagnostics(transport)
    pinv = torch.linalg.pinv(transport, rtol=_pinv_rtol(transport))
    aligned = b_don @ pinv

    inverse_agreement = None
    if diagnostics["full_rank"]:
        inverse = torch.linalg.inv(transport)
        denom = max(float(torch.linalg.norm(pinv).item()), EPS64)
        inverse_agreement = float(torch.linalg.norm(inverse - pinv).item() / denom)
        # A full-rank 2x2 matrix under the same numerical rank criterion should
        # have an inverse consistent with the Moore-Penrose pseudoinverse.
        require(
            inverse_agreement <= 1e-10,
            f"M8A_INVERSE_PINV_DISAGREEMENT:{inverse_agreement}",
        )

    return aligned, {
        "transport_full_rank": diagnostics["full_rank"],
        "transport_condition_number": diagnostics["condition_number"],
        "inverse_pinv_relative_difference": inverse_agreement,
    }


def operator_inner_product(
    b_left: torch.Tensor,
    a_left: torch.Tensor,
    b_right: torch.Tensor,
    a_right: torch.Tensor,
    gram: torch.Tensor | None = None,
) -> torch.Tensor:
    b_left = b_left.to(FLOAT_DTYPE)
    b_right = b_right.to(FLOAT_DTYPE)
    a_left = a_left.to(FLOAT_DTYPE)
    a_right = a_right.to(FLOAT_DTYPE)
    b_cross = b_left.T @ b_right
    if gram is None:
        a_cross = a_left @ a_right.T
    else:
        gram = gram.to(FLOAT_DTYPE)
        require(tuple(gram.shape) == (HIDDEN_WIDTH, HIDDEN_WIDTH), "M8A_OPERATOR_GRAM_SHAPE")
        a_cross = a_left @ gram @ a_right.T
    return torch.sum(b_cross * a_cross)


def operator_normalized_residual(
    b_left: torch.Tensor,
    a_left: torch.Tensor,
    b_right: torch.Tensor,
    a_right: torch.Tensor,
    gram: torch.Tensor | None = None,
) -> float:
    n_left = operator_inner_product(b_left, a_left, b_left, a_left, gram)
    n_right = operator_inner_product(b_right, a_right, b_right, a_right, gram)
    cross = operator_inner_product(b_left, a_left, b_right, a_right, gram)
    diff_sq = torch.clamp(n_left + n_right - 2.0 * cross, min=0.0)
    den_sq = 0.5 * (n_left + n_right)
    require(float(den_sq.item()) > 0.0, "M8A_OPERATOR_ZERO_DENOM")
    return float((torch.sqrt(diff_sq) / torch.sqrt(den_sq)).item())


def _walk_mapping(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            yield path, child
            yield from _walk_mapping(child, path)


def _find_scalar_by_keys(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> Any | None:
    wanted = set(keys)
    matches = []
    for path, value in _walk_mapping(payload):
        if path.rsplit(".", 1)[-1] in wanted and not isinstance(value, Mapping):
            matches.append((path, value))
    if not matches:
        return None
    # If repeated, require semantic agreement.
    first = matches[0][1]
    for _path, value in matches[1:]:
        if isinstance(first, torch.Tensor) or isinstance(value, torch.Tensor):
            continue
        require(value == first, f"M8A_METADATA_CONFLICT:{matches}")
    return first


def extract_task_gram(payload: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    require(isinstance(payload, Mapping), "M8A_TASK_GRAM_PAYLOAD_NOT_MAPPING")
    schema = payload.get("schema_version")
    require(isinstance(schema, str) and schema, "M8A_TASK_GRAM_SCHEMA_MISSING")

    gram_candidates: list[tuple[str, torch.Tensor]] = []
    for path, value in _walk_mapping(payload):
        if isinstance(value, torch.Tensor) and tuple(value.shape) == (
            HIDDEN_WIDTH,
            HIDDEN_WIDTH,
        ):
            gram_candidates.append((path, value))
    require(
        len(gram_candidates) == 1,
        "M8A_TASK_GRAM_TENSOR_COUNT:"
        + ",".join(path for path, _value in gram_candidates),
    )
    gram_path, gram = gram_candidates[0]
    gram = gram.detach().cpu().to(FLOAT_DTYPE).contiguous()
    require(torch.isfinite(gram).all().item(), "M8A_TASK_GRAM_NONFINITE")
    symmetry_error = float(torch.max(torch.abs(gram - gram.T)).item())
    require(symmetry_error <= 1e-8, f"M8A_TASK_GRAM_ASYMMETRY:{symmetry_error}")
    gram = 0.5 * (gram + gram.T)

    token_count = _find_scalar_by_keys(
        payload,
        ("valid_token_count", "valid_tokens", "token_count"),
    )
    require(token_count is not None, "M8A_TASK_GRAM_TOKEN_COUNT_MISSING")
    if isinstance(token_count, torch.Tensor):
        require(token_count.numel() == 1, "M8A_TASK_GRAM_TOKEN_COUNT_TENSOR")
        token_count = int(token_count.item())
    token_count = int(token_count)
    require(token_count == VALID_TOKEN_COUNT, f"M8A_TASK_GRAM_TOKEN_COUNT:{token_count}")

    dev_rows = _find_scalar_by_keys(payload, ("dev_rows", "row_count"))
    if dev_rows is not None:
        if isinstance(dev_rows, torch.Tensor):
            dev_rows = int(dev_rows.item())
        require(int(dev_rows) == DEV_ROWS, f"M8A_TASK_GRAM_DEV_ROWS:{dev_rows}")

    encoding = _find_scalar_by_keys(payload, ("dev_encoding_sha256",))
    if encoding is not None:
        require(str(encoding) == DEV_ENCODING_SHA256, "M8A_TASK_GRAM_ENCODING_SHA")

    order = _find_scalar_by_keys(payload, ("dev_order_sha256",))
    if order is not None:
        require(str(order) == DEV_ORDER_SHA256, "M8A_TASK_GRAM_ORDER_SHA")

    execution_commit = _find_scalar_by_keys(
        payload,
        ("execution_commit", "execution_head"),
    )
    if execution_commit is not None:
        require(
            str(execution_commit) == TASK_QUOTIENT_EXECUTION_COMMIT,
            f"M8A_TASK_GRAM_EXECUTION_COMMIT:{execution_commit}",
        )

    arm = _find_scalar_by_keys(payload, ("arm",))
    if arm is not None:
        require(str(arm) == "G5-C0", f"M8A_TASK_GRAM_ARM:{arm}")

    pressure = _find_scalar_by_keys(payload, ("pressure",))
    if pressure is not None:
        require(str(pressure) == "P0", f"M8A_TASK_GRAM_PRESSURE:{pressure}")

    eigvals = torch.linalg.eigvalsh(gram)
    negative_tol = max(1.0, float(torch.max(torch.abs(eigvals)).item())) * 1e-10
    require(
        float(eigvals.min().item()) >= -negative_tol,
        f"M8A_TASK_GRAM_NOT_PSD:{eigvals.min().item()}",
    )

    return gram, {
        "schema_version": schema,
        "gram_tensor_path": gram_path,
        "valid_token_count": token_count,
        "symmetry_max_abs": symmetry_error,
        "min_eigenvalue": float(eigvals.min().item()),
        "max_eigenvalue": float(eigvals.max().item()),
        "dev_rows_metadata_present": dev_rows is not None,
        "dev_encoding_metadata_present": encoding is not None,
        "dev_order_metadata_present": order is not None,
        "execution_commit_metadata_present": execution_commit is not None,
    }


def _load_trajectory_a0_b1(path: Path) -> tuple[dict[int, torch.Tensor], dict[tuple[int, int], torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(
        isinstance(payload, Mapping)
        and payload.get("schema_version") == "GEN5_AINIT_TEMPORAL_BIRTH_PHASE_A_TRAJECTORY_V2",
        "M8A_TRAJECTORY_SCHEMA",
    )
    require(tuple(payload.get("factor_seeds", ())) == FACTOR_SEEDS, "M8A_TRAJECTORY_SEEDS")
    cells = payload.get("cells")
    require(isinstance(cells, Mapping), "M8A_TRAJECTORY_CELLS")
    require(set(cells) == set(expected_cells()), "M8A_TRAJECTORY_CELL_SET")

    a0: dict[int, torch.Tensor] = {}
    b1: dict[tuple[int, int], torch.Tensor] = {}

    for a_seed in FACTOR_SEEDS:
        reference_a: torch.Tensor | None = None
        for r_seed in FACTOR_SEEDS:
            cell = cells[cell_name(a_seed, r_seed)]
            snapshots = cell["snapshots"]
            snap0 = snapshots[0]
            snap1 = snapshots[1]
            require(int(snap0["t"]) == 0, "M8A_TRAJECTORY_T0")
            require(int(snap1["t"]) == 1, "M8A_TRAJECTORY_T1")
            a = snap0["A_theta.weight"].detach().cpu().to(FLOAT_DTYPE).contiguous()
            b0 = snap0["B_theta.weight"].detach().cpu()
            b = snap1["B_theta.weight"].detach().cpu().to(FLOAT_DTYPE).contiguous()
            require(tuple(a.shape) == (LATENT_WIDTH, HIDDEN_WIDTH), "M8A_A0_SHAPE")
            require(tuple(b0.shape) == (WRITE_WIDTH, LATENT_WIDTH), "M8A_B0_SHAPE")
            require(tuple(b.shape) == (WRITE_WIDTH, LATENT_WIDTH), "M8A_B1_SHAPE")
            require(int(torch.count_nonzero(b0).item()) == 0, "M8A_B0_NONZERO")
            require(torch.isfinite(a).all().item(), "M8A_A0_NONFINITE")
            require(torch.isfinite(b).all().item(), "M8A_B1_NONFINITE")
            if reference_a is None:
                reference_a = a
            else:
                require(torch.equal(reference_a, a), f"M8A_A0_R_DRIFT:{a_seed}:{r_seed}")
            b1[(a_seed, r_seed)] = b
        require(reference_a is not None, f"M8A_A0_MISSING:{a_seed}")
        a0[a_seed] = reference_a

    return a0, b1, {
        "schema_version": payload["schema_version"],
        "factor_seeds": list(FACTOR_SEEDS),
        "cell_count": len(cells),
    }


def _load_m7_logits(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(
        isinstance(payload, Mapping)
        and payload.get("schema_version") == "GEN5_M7_FACTOR_SWAP_LOGITS_V1",
        "M8A_M7_LOGITS_SCHEMA",
    )
    require(list(payload.get("state_order", ())) == list(expected_hybrids()), "M8A_M7_STATE_ORDER")
    logits = payload["logits"]
    margins = payload["two_margins"]
    predictions = payload["predictions"]
    require(tuple(logits.shape) == (27, DEV_ROWS, 3), "M8A_M7_LOGITS_SHAPE")
    require(tuple(margins.shape) == (27, DEV_ROWS, 2), "M8A_M7_MARGIN_SHAPE")
    require(tuple(predictions.shape) == (27, DEV_ROWS), "M8A_M7_PRED_SHAPE")
    require(torch.isfinite(logits).all().item(), "M8A_M7_LOGITS_NONFINITE")
    require(torch.isfinite(margins).all().item(), "M8A_M7_MARGIN_NONFINITE")
    return {
        "state_order": list(payload["state_order"]),
        "two_margins": margins.detach().cpu().to(FLOAT_DTYPE).contiguous(),
    }


def _authenticate_m7_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == "GEN5_M7_FACTOR_SWAP_SUMMARY_V1", "M8A_M7_SUMMARY_SCHEMA")
    require(payload.get("result") == "PASS_GEN5_M7_FACTOR_SWAP", "M8A_M7_SUMMARY_RESULT")
    require(int(payload.get("dev_rows", -1)) == DEV_ROWS, "M8A_M7_SUMMARY_ROWS")
    require(int(payload.get("all_state_count", -1)) == 27, "M8A_M7_SUMMARY_STATES")
    require(payload.get("training_executed") is False, "M8A_M7_SUMMARY_TRAINING")
    require(payload.get("backward_executed") is False, "M8A_M7_SUMMARY_BACKWARD")
    require(payload.get("optimizer_constructed") is False, "M8A_M7_SUMMARY_OPTIMIZER")
    require(payload.get("confirmatory_9601_9900_loaded") is False, "M8A_M7_SUMMARY_CONFIRMATORY")
    return {
        "schema_version": payload["schema_version"],
        "result": payload["result"],
        "execution_head": payload.get("execution_head"),
    }


def _verify_source_file(path: Path, expected_sha: str, label: str) -> None:
    full = ROOT / path
    require(full.is_file(), f"M8A_{label}_MISSING:{path}")
    actual = sha256_file(full)
    require(actual == expected_sha, f"M8A_{label}_SHA:{actual}")


def authenticate_repo(
    expected_head: str,
    *,
    allow_implementation_worktree: bool,
) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"M8A_BRANCH:{branch}")
    require(head == expected_head, f"M8A_HEAD:{head}")
    require(
        git_rc("merge-base", "--is-ancestor", AUTHORITY_COMMIT, expected_head) == 0,
        "M8A_AUTHORITY_NOT_ANCESTOR",
    )
    require(
        git_rc("merge-base", "--is-ancestor", SOURCE_HEAD, expected_head) == 0,
        "M8A_SOURCE_NOT_ANCESTOR",
    )
    frozen_blob = git("rev-parse", f"{AUTHORITY_COMMIT}:{AUTHORITY_PATH}")
    live_blob = git("rev-parse", f"HEAD:{AUTHORITY_PATH}")
    require(frozen_blob == live_blob, "M8A_AUTHORITY_BLOB_DRIFT")
    source_m7b_blob = git("rev-parse", f"{SOURCE_HEAD}:{M7B_REPORT_PATH.as_posix()}")
    live_m7b_blob = git("rev-parse", f"HEAD:{M7B_REPORT_PATH.as_posix()}")
    require(source_m7b_blob == live_m7b_blob, "M8A_M7B_REPORT_BLOB_DRIFT")

    observed = status_paths()
    if allow_implementation_worktree:
        require(
            observed <= ALLOWED_IMPLEMENTATION_PATHS,
            f"M8A_IMPLEMENTATION_SCOPE:{sorted(observed)}",
        )
    else:
        require(not observed, f"M8A_WORKTREE_NOT_CLEAN:{sorted(observed)}")


def load_static_sources() -> dict[str, Any]:
    _verify_source_file(PHASE_A_TRAJECTORY_PATH, PHASE_A_TRAJECTORY_SHA256, "TRAJECTORY")
    _verify_source_file(M7_LOGITS_PATH, M7_LOGITS_SHA256, "M7_LOGITS")
    _verify_source_file(M7_SUMMARY_PATH, M7_SUMMARY_SHA256, "M7_SUMMARY")
    _verify_source_file(TASK_GRAM_PATH, TASK_GRAM_SHA256, "TASK_GRAM")

    a0, b1, trajectory_meta = _load_trajectory_a0_b1(ROOT / PHASE_A_TRAJECTORY_PATH)
    m7 = _load_m7_logits(ROOT / M7_LOGITS_PATH)
    m7_summary_meta = _authenticate_m7_summary(ROOT / M7_SUMMARY_PATH)
    task_payload = torch.load(ROOT / TASK_GRAM_PATH, map_location="cpu", weights_only=True)
    gram, gram_meta = extract_task_gram(task_payload)

    return {
        "a0": a0,
        "b1": b1,
        "m7": m7,
        "gram": gram,
        "trajectory_meta": trajectory_meta,
        "m7_summary_meta": m7_summary_meta,
        "gram_meta": gram_meta,
    }


def _state_index(state_order: Sequence[str]) -> dict[tuple[int, int, int], int]:
    result: dict[tuple[int, int, int], int] = {}
    for a_rec in FACTOR_SEEDS:
        for a_don in FACTOR_SEEDS:
            for r_don in FACTOR_SEEDS:
                name = hybrid_name(a_rec, a_don, r_don)
                require(name in state_order, f"M8A_M7_STATE_MISSING:{name}")
                result[(a_rec, a_don, r_don)] = state_order.index(name)
    return result


def m7_ordered_affinity(
    m7: Mapping[str, Any],
    recipient: int,
    donor: int,
) -> float:
    require(recipient != donor, "M8A_AFFINITY_DIAGONAL")
    margins = m7["two_margins"]
    index = _state_index(m7["state_order"])
    affinities = []
    eps = 1e-12
    for r_don in FACTOR_SEEDS:
        cross = margins[index[(recipient, donor, r_don)]]
        rec = margins[index[(recipient, recipient, r_don)]]
        don = margins[index[(donor, donor, r_don)]]
        d_rec = torch.linalg.vector_norm(cross - rec, dim=-1)
        d_don = torch.linalg.vector_norm(cross - don, dim=-1)
        affinities.append((d_don - d_rec) / (d_don + d_rec + eps))
    return float(torch.cat(affinities).mean().item())


def m7b_interaction_strengths(m7: Mapping[str, Any]) -> dict[tuple[int, int], float]:
    margins = m7["two_margins"]
    index = _state_index(m7["state_order"])
    y = torch.empty((3, 3, 3, DEV_ROWS, 2), dtype=FLOAT_DTYPE)
    for i, a_rec in enumerate(FACTOR_SEEDS):
        for j, a_don in enumerate(FACTOR_SEEDS):
            for k, r_don in enumerate(FACTOR_SEEDS):
                y[i, j, k] = margins[index[(a_rec, a_don, r_don)]]

    grand = y.mean(dim=(0, 1, 2), keepdim=True)
    rec_main = y.mean(dim=(1, 2), keepdim=True) - grand
    don_main = y.mean(dim=(0, 2), keepdim=True) - grand
    rec_don_mean = y.mean(dim=2, keepdim=True)
    interaction = rec_don_mean - grand - rec_main - don_main

    result: dict[tuple[int, int], float] = {}
    for i, a_rec in enumerate(FACTOR_SEEDS):
        for j, a_don in enumerate(FACTOR_SEEDS):
            row_norm = torch.linalg.vector_norm(interaction[i, j, 0], dim=-1)
            result[(a_rec, a_don)] = float(row_norm.mean().item())
    return result


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        mean_rank = 0.5 * ((pos + 1) + end)
        for k in range(pos, end):
            ranks[order[k]] = mean_rank
        pos = end
    return ranks


def descriptive_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    require(len(left) == len(right), "M8A_SPEARMAN_LENGTH")
    if len(left) < 2:
        return None
    x = torch.tensor(_rankdata(left), dtype=FLOAT_DTYPE)
    y = torch.tensor(_rankdata(right), dtype=FLOAT_DTYPE)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) == 0.0:
        return None
    return float(torch.dot(x, y).item() / denom.item())


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def compute_ordered_pair_metrics(
    a0: Mapping[int, torch.Tensor],
    b1: Mapping[tuple[int, int], torch.Tensor],
    gram: torch.Tensor,
    m7: Mapping[str, Any],
) -> list[dict[str, Any]]:
    interaction_strength = m7b_interaction_strengths(m7)
    rows: list[dict[str, Any]] = []

    for recipient, donor in ordered_a_pairs():
        a_rec = a0[recipient]
        a_don = a0[donor]
        geometry = row_space_geometry(a_rec, a_don)
        t_gl = fit_gl_transport(a_rec, a_don)
        t_diag = transport_diagnostics(t_gl)
        t_orth = fit_orthogonal_transport(a_rec, a_don)

        transported_a = t_gl @ a_don
        raw_a_ambient = normalized_matrix_residual(a_rec, a_don)
        aligned_a_ambient = normalized_matrix_residual(a_rec, transported_a)
        raw_a_task = normalized_matrix_residual(a_rec, a_don, gram)
        aligned_a_task = normalized_matrix_residual(a_rec, transported_a, gram)
        orth_a_ambient = normalized_matrix_residual(a_rec, t_orth @ a_don)
        orth_a_task = normalized_matrix_residual(a_rec, t_orth @ a_don, gram)

        per_rng = []
        for r_don in FACTOR_SEEDS:
            b_don = b1[(donor, r_don)]
            aligned_b, b_meta = transport_b_to_recipient(b_don, t_gl)

            raw_ambient = operator_normalized_residual(
                b_don, a_rec, b_don, a_don, None
            )
            aligned_ambient = operator_normalized_residual(
                aligned_b, a_rec, b_don, a_don, None
            )
            raw_task = operator_normalized_residual(
                b_don, a_rec, b_don, a_don, gram
            )
            aligned_task = operator_normalized_residual(
                aligned_b, a_rec, b_don, a_don, gram
            )
            per_rng.append(
                {
                    "donor_training_rng": r_don,
                    "E_O_raw_ambient": raw_ambient,
                    "E_O_aligned_ambient": aligned_ambient,
                    "E_O_raw_task": raw_task,
                    "E_O_aligned_task": aligned_task,
                    "S_O_ambient": _safe_ratio(aligned_ambient, raw_ambient),
                    "S_O_task": _safe_ratio(aligned_task, raw_task),
                    **b_meta,
                }
            )

        def mean_key(key: str) -> float:
            return float(sum(float(row[key]) for row in per_rng) / len(per_rng))

        row = {
            "recipient_a_seed": recipient,
            "donor_a_seed": donor,
            "row_space_geometry": geometry,
            "transport_gl": t_diag,
            "transport_orthogonal_matrix": [
                [float(x) for x in line] for line in t_orth
            ],
            "E_A_raw_ambient": raw_a_ambient,
            "E_A_aligned_ambient": aligned_a_ambient,
            "S_A_ambient": _safe_ratio(aligned_a_ambient, raw_a_ambient),
            "E_A_raw_task": raw_a_task,
            "E_A_aligned_task": aligned_a_task,
            "S_A_task": _safe_ratio(aligned_a_task, raw_a_task),
            "E_A_orthogonal_ambient": orth_a_ambient,
            "E_A_orthogonal_task": orth_a_task,
            "E_O_raw_ambient": mean_key("E_O_raw_ambient"),
            "E_O_aligned_ambient": mean_key("E_O_aligned_ambient"),
            "E_O_raw_task": mean_key("E_O_raw_task"),
            "E_O_aligned_task": mean_key("E_O_aligned_task"),
            "S_O_ambient": mean_key("S_O_ambient"),
            "S_O_task": mean_key("S_O_task"),
            "frozen_m7_ordered_mean_affinity": m7_ordered_affinity(
                m7, recipient, donor
            ),
            "frozen_m7b_recipient_donor_interaction_strength": interaction_strength[
                (recipient, donor)
            ],
            "per_donor_rng": per_rng,
        }
        rows.append(row)

    require(len(rows) == 6, f"M8A_ORDERED_PAIR_COUNT:{len(rows)}")
    return rows


def summarize_structural_predicates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def all_lt(left: str, right: str) -> bool:
        return all(float(row[left]) < float(row[right]) for row in rows)

    predicates = {
        "all_transport_full_rank": all(
            bool(row["transport_gl"]["full_rank"]) for row in rows
        ),
        "A_ambient_reduced_all_ordered_pairs": all_lt(
            "E_A_aligned_ambient", "E_A_raw_ambient"
        ),
        "A_task_reduced_all_ordered_pairs": all_lt(
            "E_A_aligned_task", "E_A_raw_task"
        ),
        "operator_ambient_reduced_all_ordered_pairs": all_lt(
            "E_O_aligned_ambient", "E_O_raw_ambient"
        ),
        "operator_task_reduced_all_ordered_pairs": all_lt(
            "E_O_aligned_task", "E_O_raw_task"
        ),
        "A_aligned_task_below_aligned_ambient_all_ordered_pairs": all(
            float(row["E_A_aligned_task"]) < float(row["E_A_aligned_ambient"])
            for row in rows
        ),
        "operator_aligned_task_below_aligned_ambient_all_ordered_pairs": all(
            float(row["E_O_aligned_task"]) < float(row["E_O_aligned_ambient"])
            for row in rows
        ),
    }

    interaction = [
        float(row["frozen_m7b_recipient_donor_interaction_strength"])
        for row in rows
    ]
    aligned_ambient = [float(row["E_O_aligned_ambient"]) for row in rows]
    aligned_task = [float(row["E_O_aligned_task"]) for row in rows]
    affinity = [float(row["frozen_m7_ordered_mean_affinity"]) for row in rows]

    predicates["descriptive_spearman_interaction_vs_aligned_ambient_operator_residual"] = (
        descriptive_spearman(interaction, aligned_ambient)
    )
    predicates["descriptive_spearman_interaction_vs_aligned_task_operator_residual"] = (
        descriptive_spearman(interaction, aligned_task)
    )
    predicates["descriptive_spearman_affinity_vs_aligned_task_operator_residual"] = (
        descriptive_spearman(affinity, aligned_task)
    )

    # These are prospective structural predicates for later evidence interpretation.
    predicates["case_A_structural_support"] = bool(
        predicates["all_transport_full_rank"]
        and predicates["A_ambient_reduced_all_ordered_pairs"]
        and predicates["operator_ambient_reduced_all_ordered_pairs"]
        and predicates["operator_task_reduced_all_ordered_pairs"]
    )
    predicates["case_B_structural_support"] = bool(
        (not predicates["operator_ambient_reduced_all_ordered_pairs"])
        and (
            predicates[
                "A_aligned_task_below_aligned_ambient_all_ordered_pairs"
            ]
            or predicates[
                "operator_aligned_task_below_aligned_ambient_all_ordered_pairs"
            ]
        )
    )
    predicates["scientific_classification"] = "DEFER_TO_VALIDATED_STATIC_EVIDENCE_INTERPRETATION"
    return predicates


def task_gram_spectrum(gram: torch.Tensor) -> dict[str, Any]:
    eigvals = torch.linalg.eigvalsh(gram.to(FLOAT_DTYPE))
    eigvals = torch.clamp(eigvals, min=0.0)
    descending = torch.flip(eigvals, dims=(0,))
    total = float(descending.sum().item())
    require(total > 0.0, "M8A_GRAM_ZERO_TRACE")
    topks = (1, 2, 4, 8, 16, 32, 64, 128)
    cumulative = {
        str(k): float(descending[:k].sum().item() / total) for k in topks
    }
    tr2 = float(torch.sum(descending.square()).item())
    pr = (total * total) / tr2 if tr2 > 0.0 else None
    return {
        "trace": total,
        "cumulative_energy_fraction": cumulative,
        "participation_ratio_effective_dimension": pr,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return None
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    data = b"".join(canonical_json_bytes(_json_safe(dict(row))) for row in rows)
    atomic_bytes(path, data)


def _artifact_manifest(output_root: Path) -> dict[str, Any]:
    files = {}
    for name in OUTPUT_FILENAMES:
        if name == "artifact_manifest.json":
            continue
        path = output_root / name
        require(path.is_file(), f"M8A_OUTPUT_MISSING_FOR_MANIFEST:{name}")
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "GEN5_M8A_STATIC_AUDIT_ARTIFACT_MANIFEST_V1",
        "files": files,
    }


def run_static_verify(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=True)
    sources = load_static_sources()
    require(len(sources["a0"]) == 3, "M8A_VERIFY_A_COUNT")
    require(len(sources["b1"]) == 9, "M8A_VERIFY_B_COUNT")
    require(tuple(sources["gram"].shape) == (HIDDEN_WIDTH, HIDDEN_WIDTH), "M8A_VERIFY_GRAM_SHAPE")
    print("GEN5_M8A_STATIC_VERIFY_PASS")
    print(f"HEAD={args.expected_head}")
    print(f"AUTHORITY_COMMIT={AUTHORITY_COMMIT}")
    print(f"TRAJECTORY_SCHEMA={sources['trajectory_meta']['schema_version']}")
    print(f"M7_SCHEMA={sources['m7_summary_meta']['schema_version']}")
    print(f"TASK_GRAM_SCHEMA={sources['gram_meta']['schema_version']}")
    print(f"TASK_GRAM_TENSOR_PATH={sources['gram_meta']['gram_tensor_path']}")
    print(f"VALID_TOKEN_COUNT={sources['gram_meta']['valid_token_count']}")
    print("UNIQUE_A0=3")
    print("B1_CELLS=9")
    print("ORDERED_OFFDIAGONAL_A_PAIRS=6")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("AUTOGRAD_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("TRAINING_EXECUTED=False")
    print("FILES_WRITTEN=0")


def run_static_audit(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    output_root = Path(args.output_root)
    require(not output_root.exists(), f"M8A_OUTPUT_ALREADY_EXISTS:{output_root}")
    output_root.mkdir(parents=True)

    sources = load_static_sources()
    rows = compute_ordered_pair_metrics(
        sources["a0"],
        sources["b1"],
        sources["gram"],
        sources["m7"],
    )
    predicates = summarize_structural_predicates(rows)

    pair_path = output_root / "m8a_ordered_pair_metrics.jsonl"
    _write_jsonl_atomic(pair_path, rows)

    summary = {
        "schema_version": "GEN5_M8A_LATENT_ALIGNMENT_QUOTIENT_SUMMARY_V1",
        "status": "PASS_STATIC_M8A_AUDIT",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "authority_commit": AUTHORITY_COMMIT,
        "source_head": SOURCE_HEAD,
        "source_identities": {
            "phase_a_trajectory_sha256": PHASE_A_TRAJECTORY_SHA256,
            "m7_logits_sha256": M7_LOGITS_SHA256,
            "m7_summary_sha256": M7_SUMMARY_SHA256,
            "task_state_gram_sha256": TASK_GRAM_SHA256,
        },
        "task_gram_metadata": sources["gram_meta"],
        "task_gram_spectrum": task_gram_spectrum(sources["gram"]),
        "ordered_pair_count": len(rows),
        "structural_predicates": predicates,
        "anti_circularity": {
            "transport_fit_inputs": ["A_rec", "A_don"],
            "m7_outputs_used_for_transport_fitting": False,
            "m7_outputs_role": "HELD_OUT_DESCRIPTIVE_VALIDATION_TARGET_ONLY",
        },
        "scientific_p_value_count": 0,
        "model_forward_count": 0,
        "cuda_executed": False,
        "autograd_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "training_executed": False,
        "checkpoint_mutation": False,
        "confirmatory_9601_9900_loaded": False,
        "scientific_classification": "DEFER_TO_VALIDATED_STATIC_EVIDENCE_INTERPRETATION",
    }
    summary_path = output_root / "m8a_latent_alignment_quotient_summary.json"
    atomic_bytes(summary_path, canonical_json_bytes(_json_safe(summary)))

    provenance = {
        "schema_version": "GEN5_M8A_STATIC_AUDIT_PROVENANCE_V1",
        "status": "PASS",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "authority_commit": AUTHORITY_COMMIT,
        "phase_a_trajectory_sha256": PHASE_A_TRAJECTORY_SHA256,
        "m7_logits_sha256": M7_LOGITS_SHA256,
        "m7_summary_sha256": M7_SUMMARY_SHA256,
        "task_state_gram_sha256": TASK_GRAM_SHA256,
        "summary_sha256": sha256_file(summary_path),
        "pair_metrics_sha256": sha256_file(pair_path),
        "model_forward_count": 0,
        "cuda_executed": False,
        "autograd_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "training_executed": False,
        "checkpoint_mutation": False,
        "confirmatory_9601_9900_loaded": False,
    }
    provenance_path = output_root / "run_provenance.json"
    atomic_bytes(provenance_path, canonical_json_bytes(provenance))

    manifest_path = output_root / "artifact_manifest.json"
    atomic_bytes(
        manifest_path,
        canonical_json_bytes(_artifact_manifest(output_root)),
    )

    print("GEN5_M8A_STATIC_AUDIT_PASS")
    print(f"HEAD={args.expected_head}")
    print(f"ORDERED_PAIR_COUNT={len(rows)}")
    print(
        "CASE_A_STRUCTURAL_SUPPORT="
        f"{predicates['case_A_structural_support']}"
    )
    print(
        "CASE_B_STRUCTURAL_SUPPORT="
        f"{predicates['case_B_structural_support']}"
    )
    print("SCIENTIFIC_CLASSIFICATION=DEFER_TO_VALIDATED_STATIC_EVIDENCE_INTERPRETATION")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("AUTOGRAD_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("TRAINING_EXECUTED=False")
    print(f"SUMMARY={summary_path}")
    print(f"PAIR_METRICS={pair_path}")
    print(f"PROVENANCE={provenance_path}")
    print(f"MANIFEST={manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-verify-only", action="store_true")
    modes.add_argument("--run-static-audit", action="store_true")
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--implementation-freeze-commit")
    parser.add_argument("--output-root", type=Path)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.static_verify_only:
        require(
            args.implementation_freeze_commit is None,
            "M8A_STATIC_VERIFY_FREEZE_FORBIDDEN",
        )
        require(args.output_root is None, "M8A_STATIC_VERIFY_OUTPUT_FORBIDDEN")
        return

    require(args.run_static_audit, "M8A_MODE")
    require(
        args.implementation_freeze_commit is not None,
        "M8A_IMPLEMENTATION_FREEZE_REQUIRED",
    )
    require(
        args.implementation_freeze_commit == args.expected_head,
        "M8A_IMPLEMENTATION_FREEZE_HEAD_MISMATCH",
    )
    require(args.output_root is not None, "M8A_OUTPUT_ROOT_REQUIRED")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.static_verify_only:
        run_static_verify(args)
        return
    if args.run_static_audit:
        run_static_audit(args)
        return
    raise M8AError("M8A_UNREACHABLE_MODE")


if __name__ == "__main__":
    main()
