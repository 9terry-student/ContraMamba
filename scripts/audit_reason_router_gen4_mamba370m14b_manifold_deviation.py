#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCALES = ("mamba370m", "mamba14b")
CELLS = ("C0_SHAM", "C2_NAME")
CONDITIONS = ("dominant_neutralized", "dominant_control")
STATE_FILE = "manifold_capture_states.npz"
INDEX_FILE = "manifold_capture_index.jsonl"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

class ManifoldAuditError(RuntimeError):
    pass

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ManifoldAuditError(message)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_sums(root: Path) -> None:
    seen = set()
    for line in (root / SUMS_FILE).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(sha256_file(root / name) == digest, f"SUM:{name}")
        seen.add(name)
    require(seen == {STATE_FILE, INDEX_FILE, MANIFEST_FILE}, f"SUM_SET:{seen}")

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

def pairwise_native_distances(native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(native, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] == 300, "NATIVE_SHAPE")
    gram = x @ x.T
    sq = np.sum(x * x, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    np.fill_diagonal(d2, np.inf)
    idx = np.argmin(d2, axis=1)
    d = np.sqrt(d2[np.arange(x.shape[0]), idx])
    require(np.isfinite(d).all(), "NATIVE_DISTANCE_NONFINITE")
    return d, idx

def intervention_to_native_distances(
    intervened: np.ndarray,
    native: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(intervened, dtype=np.float64)
    x = np.asarray(native, dtype=np.float64)
    require(y.shape == x.shape and x.shape[0] == 300, "INTERVENED_SHAPE")
    d2 = np.maximum(
        np.sum(y * y, axis=1)[:, None]
        + np.sum(x * x, axis=1)[None, :]
        - 2.0 * (y @ x.T),
        0.0,
    )
    np.fill_diagonal(d2, np.inf)
    idx = np.argmin(d2, axis=1)
    d = np.sqrt(d2[np.arange(x.shape[0]), idx])
    require(np.isfinite(d).all(), "INTERVENED_DISTANCE_NONFINITE")
    return d, idx

def audit_condition(
    native_full: np.ndarray,
    native_strong: np.ndarray,
    delta_full: np.ndarray,
    strong_indices: np.ndarray,
) -> dict[str, Any]:
    nf = np.asarray(native_full, dtype=np.float64)
    ns = np.asarray(native_strong, dtype=np.float64)
    df = np.asarray(delta_full, dtype=np.float64)
    require(nf.shape[0] == ns.shape[0] == df.shape[0] == 300, "AUDIT_N")
    require(nf.shape == df.shape, "DELTA_FULL_SHAPE")
    ds = df[:, strong_indices]
    require(ds.shape == ns.shape, "DELTA_STRONG_SHAPE")

    r_rel_full = np.linalg.norm(df, axis=1) / np.maximum(np.linalg.norm(nf, axis=1), 1e-12)
    r_rel_strong = np.linalg.norm(ds, axis=1) / np.maximum(np.linalg.norm(ns, axis=1), 1e-12)

    d_native_full, nn_native_full = pairwise_native_distances(nf)
    d_native_strong, nn_native_strong = pairwise_native_distances(ns)
    d_int_full, nn_int_full = intervention_to_native_distances(nf + df, nf)
    d_int_strong, nn_int_strong = intervention_to_native_distances(ns + ds, ns)

    r_nn_full = d_int_full / np.maximum(d_native_full, 1e-12)
    r_nn_strong = d_int_strong / np.maximum(d_native_strong, 1e-12)

    return {
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
        "nearest_neighbor_identity_change_rate_full":
            float(np.mean(nn_native_full != nn_int_full)),
        "nearest_neighbor_identity_change_rate_strong":
            float(np.mean(nn_native_strong != nn_int_strong)),
        "zero_native_full_norm_count":
            int(np.sum(np.linalg.norm(nf, axis=1) <= 1e-12)),
        "zero_native_strong_norm_count":
            int(np.sum(np.linalg.norm(ns, axis=1) <= 1e-12)),
    }

def audit(capture_dir: Path) -> dict[str, Any]:
    validate_sums(capture_dir)
    manifest = json.loads((capture_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    require(manifest["manifold_metric_computed"] is False, "RAW_METRIC_BOUNDARY")
    index = [
        json.loads(line)
        for line in (capture_dir / INDEX_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(index) == 1200, "INDEX_COUNT")

    by_scale_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in index:
        by_scale_cell.setdefault(
            (str(row["scale"]), str(row["contrast_cell_id"])),
            [],
        ).append(row)

    result: dict[str, Any] = {
        "result": "PASS_GEN4_INTERVENTION_MANIFOLD_DEVIATION_STATIC_AUDIT",
        "primary_p_value_count": 0,
        "inference_performed": False,
        "scales": {},
    }

    with np.load(capture_dir / STATE_FILE, allow_pickle=False) as data:
        for scale in SCALES:
            rows_scale = [
                row for row in index if str(row["scale"]) == scale
            ]
            rows_scale.sort(key=lambda r: int(r["scale_row_index"]))
            strong_dim = int(rows_scale[0]["strong_dim"])
            full_key = f"{scale}__h_native_full"
            strong_key = f"{scale}__h_native_strong"
            neutral_key = f"{scale}__delta_neutralized_full"
            control_key = f"{scale}__delta_control_full"
            nf_all = data[full_key]
            ns_all = data[strong_key]
            dn_all = data[neutral_key]
            dc_all = data[control_key]
            require(nf_all.shape[0] == ns_all.shape[0] == 600, f"SCALE_ROWS:{scale}")
            require(ns_all.shape[1] == strong_dim, f"STRONG_DIM:{scale}")

            strong_indices = np.asarray(
                data[f"{scale}__strong_indices"],
                dtype=np.int64,
            )
            require(
                strong_indices.shape == (strong_dim,),
                f"STRONG_INDEX_SHAPE:{scale}",
            )
            require(
                np.all(strong_indices[:-1] < strong_indices[1:]),
                f"STRONG_INDEX_ORDER:{scale}",
            )
            require(
                int(strong_indices[0]) >= 0
                and int(strong_indices[-1]) < nf_all.shape[1],
                f"STRONG_INDEX_RANGE:{scale}",
            )
            require(
                np.array_equal(nf_all[:, strong_indices], ns_all),
                f"STRONG_GATHER_IDENTITY:{scale}",
            )

            scale_result: dict[str, Any] = {}
            for cell in CELLS:
                row_positions = [
                    int(row["scale_row_index"])
                    for row in rows_scale
                    if str(row["contrast_cell_id"]) == cell
                ]
                require(len(row_positions) == 300, f"CELL_N:{scale}:{cell}")
                pos = np.asarray(row_positions, dtype=np.int64)
                # Preserve pair order xg1_fact_4801..5100.
                pair_order = [
                    str(rows_scale[i]["source_pair_id"])
                    for i in row_positions
                ]
                require(
                    pair_order == [f"xg1_fact_{i}" for i in range(4801, 5101)],
                    f"PAIR_ORDER:{scale}:{cell}",
                )
                nf = nf_all[pos]
                ns = ns_all[pos]
                cell_result = {
                    "dominant_neutralized": audit_condition(
                        nf, ns, dn_all[pos], strong_indices
                    ),
                    "dominant_control": audit_condition(
                        nf, ns, dc_all[pos], strong_indices
                    ),
                    "dominant_restored": {
                        "R_rel_full": 0.0,
                        "R_rel_strong": 0.0,
                        "R_NN_full": 1.0,
                        "R_NN_strong": 1.0,
                        "sanity_only": True,
                    },
                }
                scale_result[cell] = cell_result
            result["scales"][scale] = scale_result
    return result

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    require(not args.output_dir.exists(), "OUTPUT_COLLISION")
    result = audit(args.capture_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = (
        json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()
    (args.output_dir / "manifold_deviation_audit.json").write_bytes(report)
    manifest = {
        "result": result["result"],
        "primary_p_value_count": 0,
        "inference_performed": False,
        "rescue_performed": False,
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    names = ("manifold_deviation_audit.json", "artifact_manifest.json")
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(args.output_dir/name)}  {name}\n" for name in sorted(names)),
        encoding="utf-8",
        newline="\n",
    )
    print("RESULT=" + result["result"])
    print("PRIMARY_P_VALUE_COUNT=0")
    print("INFERENCE_PERFORMED=False")
    print("RESCUE_PERFORMED=False")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
