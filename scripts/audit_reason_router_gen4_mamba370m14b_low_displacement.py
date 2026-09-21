#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
RAW_FREEZE_COMMIT = "320b340d4b580beb2919daa0024cbf3c04ed65c1"
RAW_RUN_NAME = "g4k-lowdisp-behavioral-537e89b-r1-2t4"

RAW_DIR = (
    ROOT
    / "reports/reason_router_gen4_mamba370m14b_low_displacement_state_capture_raw_runs"
    / RAW_RUN_NAME
)
STATE_FILE = "low_displacement_states.npz"
INDEX_FILE = "capture_index.jsonl"
RAW_MANIFEST_FILE = "artifact_manifest.json"
RAW_SUMS_FILE = "SHA256SUMS.txt"

EXPECTED_RAW_SHA256 = {
    RAW_MANIFEST_FILE:
        "25849d983201a76dc1cfd160a266dde3a04e8abb6fb96bbc42a5a3a93108bbaa",
    INDEX_FILE:
        "32710d1eb46e79cf21d6570f48bc21ac0711827bde9f0efb56b100f7dbb2995f",
    STATE_FILE:
        "163bd4d90f5b5e649301eb3ee4d24d871e9452910b3bcfd82b74316374462580",
}

SCALES = ("mamba370m", "mamba14b")
CELLS = ("C0_SHAM", "C2_NAME")
PAIR_IDS = tuple(f"xg1_fact_{i}" for i in range(5401, 5701))
ALPHAS = (1.0, 0.5, 0.25)
N = 300

AUDIT_FILE = "low_displacement_audit.json"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

RESULT = "PASS_GEN4_LOW_DISPLACEMENT_STATIC_DISPLACEMENT_AUDIT"
REL_SCALING_RTOL = 1.0e-12
REL_SCALING_ATOL = 1.0e-12


class LowDisplacementAuditError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LowDisplacementAuditError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LowDisplacementAuditError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH_MISMATCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", RAW_FREEZE_COMMIT, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "RAW_FREEZE_NOT_ANCESTOR")


def validate_raw_bundle(raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_files = {STATE_FILE, INDEX_FILE, RAW_MANIFEST_FILE, RAW_SUMS_FILE}
    require(raw_dir.is_dir(), "RAW_DIR_MISSING")
    require(
        {p.name for p in raw_dir.iterdir() if p.is_file()} == expected_files,
        "RAW_FILE_SET",
    )
    for name, digest in EXPECTED_RAW_SHA256.items():
        require(sha256_file(raw_dir / name) == digest, f"RAW_SHA256:{name}")

    sums: dict[str, str] = {}
    for line in (raw_dir / RAW_SUMS_FILE).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        sums[name] = digest
    require(sums == EXPECTED_RAW_SHA256, "RAW_SUMS_CONTENT")

    manifest = json.loads(
        (raw_dir / RAW_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    require(
        manifest["result"] == "PASS_GEN4_LOW_DISPLACEMENT_STATE_CAPTURE_RAW",
        "RAW_RESULT",
    )
    require(
        manifest["execution_head"] == "537e89b4266ffef1b788832b5e00de411e5ff785",
        "RAW_EXECUTION_HEAD",
    )
    require(manifest["population"] == "xg1_fact_5401..xg1_fact_5700", "RAW_POPULATION")
    require(manifest["state_row_count"] == 1200, "RAW_STATE_ROWS")
    require(manifest["behavioral_alphas"] == [0.5, 0.25], "RAW_ALPHAS")
    require(
        manifest["alpha1_geometric_reference_permitted_static_only"] is True,
        "RAW_ALPHA1_GEOMETRIC",
    )
    require(
        manifest["alpha1_behavioral_forward_executed"] is False,
        "RAW_ALPHA1_BEHAVIOR",
    )
    require(manifest["manifold_metric_computed"] is False, "RAW_METRIC_BOUNDARY")
    require(manifest["p_value_count_executed"] == 0, "RAW_PVALUES")

    index = [
        json.loads(line)
        for line in (raw_dir / INDEX_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(index) == 1200, "INDEX_COUNT")
    return index, manifest


def descriptive(x: np.ndarray) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    require(a.ndim == 1 and a.size > 0 and np.isfinite(a).all(), "DESC")
    return {
        "mean": float(a.mean()),
        "sd_population": float(a.std(ddof=0)),
        "min": float(a.min()),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "max": float(a.max()),
    }


def pairwise_native_distances(
    native: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(native, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] == N, "NATIVE_SHAPE")
    gram = x @ x.T
    sq = np.sum(x * x, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    np.fill_diagonal(d2, np.inf)
    idx = np.argmin(d2, axis=1)
    d = np.sqrt(d2[np.arange(N), idx])
    require(np.isfinite(d).all(), "NATIVE_DISTANCE_NONFINITE")
    return d, idx


def intervention_to_native_distances(
    intervened: np.ndarray,
    native: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(intervened, dtype=np.float64)
    x = np.asarray(native, dtype=np.float64)
    require(y.shape == x.shape and x.shape[0] == N, "INTERVENED_SHAPE")
    d2 = np.maximum(
        np.sum(y * y, axis=1)[:, None]
        + np.sum(x * x, axis=1)[None, :]
        - 2.0 * (y @ x.T),
        0.0,
    )
    np.fill_diagonal(d2, np.inf)
    idx = np.argmin(d2, axis=1)
    d = np.sqrt(d2[np.arange(N), idx])
    require(np.isfinite(d).all(), "INTERVENED_DISTANCE_NONFINITE")
    return d, idx


def audit_alpha(
    native_full: np.ndarray,
    native_strong: np.ndarray,
    delta_full_unscaled: np.ndarray,
    strong_indices: np.ndarray,
    *,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    require(alpha in ALPHAS, f"ALPHA:{alpha}")
    nf = np.asarray(native_full, dtype=np.float64)
    ns = np.asarray(native_strong, dtype=np.float64)
    du = np.asarray(delta_full_unscaled, dtype=np.float64)
    require(nf.shape == du.shape and nf.shape[0] == N, "FULL_SHAPE")
    require(ns.shape[0] == N, "STRONG_ROWS")
    ds_u = du[:, strong_indices]
    require(ds_u.shape == ns.shape, "STRONG_DELTA_SHAPE")

    df = alpha * du
    ds = alpha * ds_u

    nf_norm = np.linalg.norm(nf, axis=1)
    ns_norm = np.linalg.norm(ns, axis=1)
    r_rel_full = np.linalg.norm(df, axis=1) / np.maximum(nf_norm, 1e-12)
    r_rel_strong = np.linalg.norm(ds, axis=1) / np.maximum(ns_norm, 1e-12)

    d_native_full, nn_native_full = pairwise_native_distances(nf)
    d_native_strong, nn_native_strong = pairwise_native_distances(ns)
    d_int_full, nn_int_full = intervention_to_native_distances(nf + df, nf)
    d_int_strong, nn_int_strong = intervention_to_native_distances(ns + ds, ns)

    r_nn_full = d_int_full / np.maximum(d_native_full, 1e-12)
    r_nn_strong = d_int_strong / np.maximum(d_native_strong, 1e-12)

    nonfinite_count = int(
        np.size(nf) - np.isfinite(nf).sum()
        + np.size(ns) - np.isfinite(ns).sum()
        + np.size(du) - np.isfinite(du).sum()
    )
    require(nonfinite_count == 0, "NONFINITE_INPUT")

    result = {
        "alpha": alpha,
        "R_rel_full": descriptive(r_rel_full),
        "R_rel_strong": descriptive(r_rel_strong),
        "R_NN_full": {
            **descriptive(r_nn_full),
            "fraction_gt_1": float(np.mean(r_nn_full > 1.0)),
            "fraction_gt_2": float(np.mean(r_nn_full > 2.0)),
        },
        "R_NN_strong": {
            **descriptive(r_nn_strong),
            "fraction_gt_1": float(np.mean(r_nn_strong > 1.0)),
            "fraction_gt_2": float(np.mean(r_nn_strong > 2.0)),
        },
        "nearest_native_identity_change_rate_full":
            float(np.mean(nn_native_full != nn_int_full)),
        "nearest_native_identity_change_rate_strong":
            float(np.mean(nn_native_strong != nn_int_strong)),
        "zero_native_full_norm_count": int(np.sum(nf_norm <= 1e-12)),
        "zero_native_strong_norm_count": int(np.sum(ns_norm <= 1e-12)),
        "nonfinite_input_count": nonfinite_count,
    }
    return result, {
        "R_rel_full": r_rel_full,
        "R_rel_strong": r_rel_strong,
    }


def audit(raw_dir: Path) -> dict[str, Any]:
    index, raw_manifest = validate_raw_bundle(raw_dir)

    by_scale_rows: dict[str, list[dict[str, Any]]] = {}
    for scale in SCALES:
        rows = [r for r in index if str(r["scale"]) == scale]
        rows.sort(key=lambda r: int(r["scale_row_index"]))
        require(len(rows) == 600, f"SCALE_INDEX_COUNT:{scale}")
        by_scale_rows[scale] = rows

    result: dict[str, Any] = {
        "schema_version":
            "gen4-mamba370m14b-low-displacement-static-displacement-audit-v1",
        "result": RESULT,
        "raw_execution_head": raw_manifest["execution_head"],
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "population": "xg1_fact_5401..xg1_fact_5700",
        "scales": {},
        "alphas": list(ALPHAS),
        "primary_p_value_count": 0,
        "inference_performed": False,
        "binary_manifold_classification_performed": False,
        "model_forward_count_this_analysis": 0,
        "backward_count_this_analysis": 0,
        "training_executed": False,
    }

    with np.load(raw_dir / STATE_FILE, allow_pickle=False) as data:
        for scale in SCALES:
            rows_scale = by_scale_rows[scale]
            nf_all = np.asarray(data[f"{scale}__h_native_full"])
            ns_all = np.asarray(data[f"{scale}__h_native_strong"])
            du_all = np.asarray(data[f"{scale}__delta_control_full"])
            strong_indices = np.asarray(
                data[f"{scale}__strong_indices"],
                dtype=np.int64,
            )

            require(nf_all.shape[0] == 600, f"FULL_ROWS:{scale}")
            require(ns_all.shape[0] == 600, f"STRONG_ROWS:{scale}")
            require(du_all.shape == nf_all.shape, f"DELTA_SHAPE:{scale}")
            strong_dim = int(rows_scale[0]["strong_dim"])
            require(strong_indices.shape == (strong_dim,), f"STRONG_INDEX_SHAPE:{scale}")
            require(np.all(strong_indices[:-1] < strong_indices[1:]), f"STRONG_INDEX_ORDER:{scale}")
            require(
                int(strong_indices[0]) >= 0
                and int(strong_indices[-1]) < nf_all.shape[1],
                f"STRONG_INDEX_RANGE:{scale}",
            )
            require(
                np.array_equal(nf_all[:, strong_indices], ns_all),
                f"STRONG_GATHER_IDENTITY:{scale}",
            )

            scale_out: dict[str, Any] = {}
            for cell in CELLS:
                positions = [
                    int(r["scale_row_index"])
                    for r in rows_scale
                    if str(r["contrast_cell_id"]) == cell
                ]
                require(len(positions) == N, f"CELL_COUNT:{scale}:{cell}")
                pair_order = [
                    str(rows_scale[pos]["source_pair_id"])
                    for pos in positions
                ]
                require(pair_order == list(PAIR_IDS), f"PAIR_ORDER:{scale}:{cell}")
                pos = np.asarray(positions, dtype=np.int64)
                nf = nf_all[pos]
                ns = ns_all[pos]
                du = du_all[pos]

                alpha_results: dict[str, Any] = {}
                rel_vectors: dict[float, dict[str, np.ndarray]] = {}
                for alpha in ALPHAS:
                    ares, vecs = audit_alpha(
                        nf, ns, du, strong_indices, alpha=alpha
                    )
                    tag = "1_0" if alpha == 1.0 else ("0_5" if alpha == 0.5 else "0_25")
                    alpha_results[f"alpha_{tag}"] = ares
                    rel_vectors[alpha] = vecs

                for metric in ("R_rel_full", "R_rel_strong"):
                    base = rel_vectors[1.0][metric]
                    for alpha in (0.5, 0.25):
                        require(
                            np.allclose(
                                rel_vectors[alpha][metric],
                                alpha * base,
                                rtol=REL_SCALING_RTOL,
                                atol=REL_SCALING_ATOL,
                            ),
                            f"R_REL_SCALING:{scale}:{cell}:{metric}:{alpha}",
                        )

                alpha_results["r_rel_scaling_identity"] = {
                    "alpha_0_5_equals_0_5_times_alpha_1": True,
                    "alpha_0_25_equals_0_25_times_alpha_1": True,
                    "rtol": REL_SCALING_RTOL,
                    "atol": REL_SCALING_ATOL,
                }
                scale_out[cell] = alpha_results
            result["scales"][scale] = scale_out
    return result


def write_audit(
    *,
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    result = audit(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / AUDIT_FILE).write_bytes(pretty_json_bytes(result))
    manifest = {
        "schema_version":
            "gen4-mamba370m14b-low-displacement-displacement-audit-manifest-v1",
        "result": RESULT,
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "raw_state_sha256": EXPECTED_RAW_SHA256[STATE_FILE],
        "audit_sha256": sha256_file(output_dir / AUDIT_FILE),
        "alphas": list(ALPHAS),
        "primary_p_value_count": 0,
        "inference_performed": False,
        "binary_manifold_classification_performed": False,
        "r_rel_scaling_identity_pass": True,
        "additional_model_forward_count": 0,
        "additional_backward_count": 0,
        "training_executed": False,
    }
    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(manifest))

    names = (AUDIT_FILE, MANIFEST_FILE)
    (output_dir / SUMS_FILE).write_text(
        "".join(
            f"{sha256_file(output_dir / name)}  {name}\n"
            for name in sorted(names)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only low-displacement audit using frozen native states and "
            "unscaled control corrections. Computes Study-C R_rel and R_NN "
            "for alpha=1.0,0.5,0.25 with zero p-values."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    authenticate_repo(args.expected_head)
    result = write_audit(raw_dir=args.raw_dir, output_dir=args.output_dir)
    print("RESULT=" + result["result"])
    print("ALPHAS=1.0,0.5,0.25")
    print("R_REL_SCALING_IDENTITY=PASS")
    print("PRIMARY_P_VALUE_COUNT=0")
    print("INFERENCE_PERFORMED=False")
    print("BINARY_MANIFOLD_CLASSIFICATION_PERFORMED=False")
    print("MODEL_FORWARD_COUNT_THIS_ANALYSIS=0")
    print("BACKWARD_COUNT_THIS_ANALYSIS=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
