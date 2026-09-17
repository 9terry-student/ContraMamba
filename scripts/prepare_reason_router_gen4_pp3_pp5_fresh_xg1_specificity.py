from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import (
    build_reason_router_gen4_xg1_fresh_specificity_cohort
    as fresh
)
from scripts import (
    reason_router_gen4_family_subspace_sensitivity_fast_cuda
    as fs
)


ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "GEN4_PP3_PP5_FRESH_XG1_SPECIFICITY_PREPARATION_V1"

PLAN_SHA = {
    "xg2": "b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c",
    "xg4": "792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f",
}

PP3_PREPARATION_REL = Path(
    "reports/reason_router_gen4_pp3_xg1_external_transport_preparation_02e4c89"
)
PP3_PLUS_SHA = (
    "66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff"
)
PP3_MINUS_SHA = (
    "ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7"
)

C3 = 0.16115855024319592
S3 = 0.98692852916688512
C5 = 0.016251944499344834
S5 = 0.99986792842854511

PP3_ZERO_BASED_INDEX = 2
PP5_ZERO_BASED_INDEX = 4

PP5_PLUS_FILE = "pp5_plus.f64le"
PP5_MINUS_FILE = "pp5_minus.f64le"
MANIFEST_FILE = "preparation_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"


class GeometryPreparationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GeometryPreparationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
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


def load_plan(family: str) -> torch.Tensor:
    require(family in PLAN_SHA, f"FAMILY:{family}")

    path = (
        ROOT
        / fs.PHASE1_ARTIFACT_ROOT
        / family
        / fs.phase1.PLAN_FILE
    )
    require(path.is_file(), f"PLAN_MISSING:{family}")

    observed = sha256_file(path)
    require(
        observed == PLAN_SHA[family],
        f"PLAN_SHA256:{family}:{observed}",
    )

    try:
        value = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        value = torch.load(
            path,
            map_location="cpu",
        )

    require(
        torch.is_tensor(value),
        f"PLAN_NOT_TENSOR:{family}",
    )
    return value


def sign_only(vector: torch.Tensor) -> torch.Tensor:
    out = (
        vector.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
        .clone()
    )

    require(
        out.ndim == 1 and int(out.numel()) == 395,
        f"VECTOR_SHAPE:{tuple(out.shape)}",
    )
    require(
        bool(torch.isfinite(out).all().item()),
        "VECTOR_NONFINITE",
    )

    pivot = int(
        torch.argmax(torch.abs(out)).item()
    )
    pivot_value = float(out[pivot].item())

    require(
        math.isfinite(pivot_value)
        and pivot_value != 0.0,
        "SIGN_PIVOT",
    )

    if pivot_value < 0.0:
        out.mul_(-1.0)

    return out.contiguous()


def raw_f64le(vector: torch.Tensor) -> bytes:
    array = np.asarray(
        vector.detach().cpu().numpy(),
        dtype=np.dtype("<f8"),
    )
    return array.tobytes(order="C")


def principal_pair(
    b2: torch.Tensor,
    b4: torch.Tensor,
    zero_based_index: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    u, singular, vh = torch.linalg.svd(
        b2.T @ b4,
        full_matrices=False,
    )

    a = (
        b2 @ u[:, zero_based_index]
    ).contiguous()
    b = (
        b4 @ vh.T[:, zero_based_index]
    ).contiguous()

    c = float(
        singular[zero_based_index].item()
    )
    s = math.sqrt(max(0.0, 1.0 - c * c))

    return a, b, c, s


def span_2d_eigh_raw(
    a: torch.Tensor,
    b: torch.Tensor,
    c: float,
    s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    require(s > 0.0, "DEGENERATE_PRINCIPAL_PLANE")

    e1 = a
    e2 = (b - c * a) / s

    matrix = torch.tensor(
        [
            [s * s, -c * s],
            [-c * s, -s * s],
        ],
        dtype=torch.float64,
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(
        matrix
    )

    require(
        bool(torch.isfinite(eigenvalues).all().item()),
        "PLANE_EIGENVALUES_NONFINITE",
    )
    require(
        bool(torch.isfinite(eigenvectors).all().item()),
        "PLANE_EIGENVECTORS_NONFINITE",
    )

    minus = (
        e1 * eigenvectors[0, 0]
        + e2 * eigenvectors[1, 0]
    )
    plus = (
        e1 * eigenvectors[0, 1]
        + e2 * eigenvectors[1, 1]
    )

    return sign_only(plus), sign_only(minus)


def vector_metrics(
    plus: torch.Tensor,
    minus: torch.Tensor,
) -> dict[str, Any]:
    plus_norm = float(
        torch.linalg.vector_norm(plus).item()
    )
    minus_norm = float(
        torch.linalg.vector_norm(minus).item()
    )
    dot = float(torch.dot(plus, minus).item())

    require(
        abs(plus_norm - 1.0) <= 1.0e-12,
        f"PLUS_NORM:{plus_norm}",
    )
    require(
        abs(minus_norm - 1.0) <= 1.0e-12,
        f"MINUS_NORM:{minus_norm}",
    )
    require(
        abs(dot) <= 1.0e-12,
        f"PLUS_MINUS_DOT:{dot}",
    )

    plus_pivot = int(
        torch.argmax(torch.abs(plus)).item()
    )
    minus_pivot = int(
        torch.argmax(torch.abs(minus)).item()
    )

    return {
        "plus_norm": plus_norm,
        "minus_norm": minus_norm,
        "plus_minus_dot": dot,
        "plus_sign_pivot_index": plus_pivot,
        "plus_sign_pivot_value": float(
            plus[plus_pivot].item()
        ),
        "minus_sign_pivot_index": minus_pivot,
        "minus_sign_pivot_value": float(
            minus[minus_pivot].item()
        ),
    }


def reconstruct_geometry() -> dict[str, Any]:
    fresh.authenticate_static_authority()

    xg2_plan = load_plan("xg2")
    xg4_plan = load_plan("xg4")

    b2 = fs.reconstruct_family_basis(
        "xg2",
        xg2_plan,
    )["basis"]
    b4 = fs.reconstruct_family_basis(
        "xg4",
        xg4_plan,
    )["basis"]

    a3, b3, c3, s3 = principal_pair(
        b2,
        b4,
        PP3_ZERO_BASED_INDEX,
    )
    require(c3 == C3, f"C3:{c3}")
    require(s3 == S3, f"S3:{s3}")

    pp3_plus, pp3_minus = span_2d_eigh_raw(
        a3,
        b3,
        c3,
        s3,
    )

    pp3_plus_raw = raw_f64le(pp3_plus)
    pp3_minus_raw = raw_f64le(pp3_minus)

    require(
        sha256_bytes(pp3_plus_raw) == PP3_PLUS_SHA,
        "PP3_PLUS_RECONSTRUCTION_SHA",
    )
    require(
        sha256_bytes(pp3_minus_raw) == PP3_MINUS_SHA,
        "PP3_MINUS_RECONSTRUCTION_SHA",
    )

    frozen_plus_path = (
        ROOT
        / PP3_PREPARATION_REL
        / "pp3_plus.f64le"
    )
    frozen_minus_path = (
        ROOT
        / PP3_PREPARATION_REL
        / "pp3_minus.f64le"
    )

    require(
        frozen_plus_path.is_file(),
        "FROZEN_PP3_PLUS_MISSING",
    )
    require(
        frozen_minus_path.is_file(),
        "FROZEN_PP3_MINUS_MISSING",
    )

    require(
        sha256_file(frozen_plus_path) == PP3_PLUS_SHA,
        "FROZEN_PP3_PLUS_SHA",
    )
    require(
        sha256_file(frozen_minus_path) == PP3_MINUS_SHA,
        "FROZEN_PP3_MINUS_SHA",
    )

    require(
        frozen_plus_path.read_bytes() == pp3_plus_raw,
        "PP3_PLUS_BYTE_REPRODUCTION",
    )
    require(
        frozen_minus_path.read_bytes() == pp3_minus_raw,
        "PP3_MINUS_BYTE_REPRODUCTION",
    )

    a5, b5, c5, s5 = principal_pair(
        b2,
        b4,
        PP5_ZERO_BASED_INDEX,
    )
    require(c5 == C5, f"C5:{c5}")
    require(s5 == S5, f"S5:{s5}")

    pp5_plus, pp5_minus = span_2d_eigh_raw(
        a5,
        b5,
        c5,
        s5,
    )

    pp5_metrics = vector_metrics(
        pp5_plus,
        pp5_minus,
    )

    pp5_plus_raw = raw_f64le(pp5_plus)
    pp5_minus_raw = raw_f64le(pp5_minus)

    return {
        "pp3_plus": pp3_plus,
        "pp3_minus": pp3_minus,
        "pp3_plus_sha256": PP3_PLUS_SHA,
        "pp3_minus_sha256": PP3_MINUS_SHA,
        "pp5_plus": pp5_plus,
        "pp5_minus": pp5_minus,
        "pp5_plus_raw": pp5_plus_raw,
        "pp5_minus_raw": pp5_minus_raw,
        "pp5_plus_sha256": sha256_bytes(pp5_plus_raw),
        "pp5_minus_sha256": sha256_bytes(pp5_minus_raw),
        "pp5_metrics": pp5_metrics,
        "c3": c3,
        "s3": s3,
        "c5": c5,
        "s5": s5,
    }


def write_preparation(
    output_dir: Path,
) -> dict[str, Any]:
    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    geometry = reconstruct_geometry()

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    plus_path = output_dir / PP5_PLUS_FILE
    minus_path = output_dir / PP5_MINUS_FILE

    plus_path.write_bytes(
        geometry["pp5_plus_raw"]
    )
    minus_path.write_bytes(
        geometry["pp5_minus_raw"]
    )

    metrics = geometry["pp5_metrics"]

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "result": "PASS_PP3_PP5_FRESH_XG1_OUTCOME_BLIND_PREPARATION",
        "design_freeze_commit": fresh.DESIGN_FREEZE_COMMIT,
        "design_git_blob": fresh.DESIGN_GIT_BLOB,
        "algorithm": (
            "principal-plane span 2x2 torch.linalg.eigh; "
            "raw eigenvectors; maximum-absolute ambient "
            "coordinate positive; no post-eigh renormalization"
        ),
        "ambient_dim": 395,
        "frozen_phase1": {
            "xg2_alignment_delta_h_sha256": PLAN_SHA["xg2"],
            "xg4_alignment_delta_h_sha256": PLAN_SHA["xg4"],
        },
        "pp3_reproduction": {
            "scientific_principal_pair_number": 3,
            "zero_based_principal_pair_index": 2,
            "c3": geometry["c3"],
            "s3": geometry["s3"],
            "plus_sha256": geometry["pp3_plus_sha256"],
            "minus_sha256": geometry["pp3_minus_sha256"],
            "frozen_bytes_reproduced_exactly": True,
        },
        "pp5": {
            "scientific_principal_pair_number": 5,
            "zero_based_principal_pair_index": 4,
            "c5": geometry["c5"],
            "s5": geometry["s5"],
            "theta_deg": math.degrees(
                math.acos(geometry["c5"])
            ),
            "positive_eigenvalue": geometry["s5"],
            "negative_eigenvalue": -geometry["s5"],
            "plus_file": PP5_PLUS_FILE,
            "minus_file": PP5_MINUS_FILE,
            "plus_sha256": geometry["pp5_plus_sha256"],
            "minus_sha256": geometry["pp5_minus_sha256"],
            "plus_norm": metrics["plus_norm"],
            "minus_norm": metrics["minus_norm"],
            "plus_minus_dot": metrics["plus_minus_dot"],
            "plus_sign_pivot_index": (
                metrics["plus_sign_pivot_index"]
            ),
            "plus_sign_pivot_value": (
                metrics["plus_sign_pivot_value"]
            ),
            "minus_sign_pivot_index": (
                metrics["minus_sign_pivot_index"]
            ),
            "minus_sign_pivot_value": (
                metrics["minus_sign_pivot_value"]
            ),
            "canonical_sign_rule": (
                "maximum-absolute ambient coordinate must be positive"
            ),
            "serialization": (
                "raw little-endian IEEE754 float64, 395 scalars"
            ),
        },
        "scientific_execution": {
            "tokenizer_executed": False,
            "checkpoint_load_count": 0,
            "model_forward_count": 0,
            "baseline_model_forward_count": 0,
            "cuda_executed": False,
            "training_executed": False,
            "backward_executed": False,
            "primary_inference_executed": False,
            "scientific_outcomes_observed": False,
        },
    }

    manifest_raw = canonical_json_bytes(manifest)
    manifest_path = output_dir / MANIFEST_FILE
    manifest_path.write_bytes(manifest_raw)

    checksums = {
        PP5_PLUS_FILE: geometry["pp5_plus_sha256"],
        PP5_MINUS_FILE: geometry["pp5_minus_sha256"],
        MANIFEST_FILE: sha256_bytes(manifest_raw),
    }

    checksum_raw = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(checksums.items())
    ).encode("utf-8")
    (output_dir / CHECKSUM_FILE).write_bytes(
        checksum_raw
    )

    return {
        **manifest,
        "manifest_sha256": sha256_bytes(manifest_raw),
        "checksum_sha256": sha256_bytes(checksum_raw),
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Statically reproduce frozen PP3 bytes and materialize "
            "response-blind PP5 specificity-control geometry."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = write_preparation(args.output_dir)

    pp5 = report["pp5"]

    print(
        "RESULT=PASS_PP3_PP5_FRESH_XG1_OUTCOME_BLIND_PREPARATION"
    )
    print("ALGORITHM=SPAN_2D_EIGH_RAW_SIGN_ONLY")
    print("PP3_FROZEN_BYTES_REPRODUCED_EXACTLY=True")
    print(f"C5={pp5['c5']:.17g}")
    print(f"S5={pp5['s5']:.17g}")
    print(
        "PP5_PLUS_SHA256=",
        pp5["plus_sha256"],
        sep="",
    )
    print(
        "PP5_MINUS_SHA256=",
        pp5["minus_sha256"],
        sep="",
    )
    print(
        "PP5_PLUS_NORM=",
        format(pp5["plus_norm"], ".17g"),
        sep="",
    )
    print(
        "PP5_MINUS_NORM=",
        format(pp5["minus_norm"], ".17g"),
        sep="",
    )
    print(
        "PP5_PLUS_MINUS_DOT=",
        format(pp5["plus_minus_dot"], ".17g"),
        sep="",
    )
    print(
        "PP5_PLUS_PIVOT=",
        pp5["plus_sign_pivot_index"],
        sep="",
    )
    print(
        "PP5_MINUS_PIVOT=",
        pp5["minus_sign_pivot_index"],
        sep="",
    )
    print(
        "MANIFEST_SHA256=",
        report["manifest_sha256"],
        sep="",
    )
    print("TOKENIZER_EXECUTED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")
    print("SCIENTIFIC_OUTCOMES_OBSERVED=False")


if __name__ == "__main__":
    main()
