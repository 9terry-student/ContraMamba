from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PROTOCOL = {
    "schema_version": "iclr2027-pair-resampling-robustness-protocol-v1",
    "source_head": "ef1806bef9e4a7f55e1f7e77f7a0f08592fb4abe",
    "bootstrap_replicates": 10000,
    "bootstrap_ci_quantiles": [0.025, 0.975],
    "bootstrap_seed": 20260924,
    "split_half_repeats": 1000,
    "split_half_size": 150,
    "split_half_seed": 20260925,
    "geometry_families": ["xg2", "xg4"],
    "geometry_primary_metric": "row-normalized-centered-linear-cka",
    "geometry_resampling_unit": "source_pair_id",
    "geometry_joint_scale_resampling": True,
    "split_half_metrics": ["pearson", "spearman", "mad", "rmse"],
    "objective_resampling_unit": "source_pair_id",
    "task_quantity": "Delta_L_forward_equivalent",
    "lm_quantity": "TASK_MATCHED pair mean over C0_SHAM and C2_NAME",
    "interval_method": "percentile",
    "permutation_control": False,
    "interpretation": "pair-resampling stability / sampling robustness; not a noise ceiling or independent-population replication",
}

SCALE_ORDER = ("130M", "370M", "790M", "1.4B", "2.8B")
FAMILIES = ("xg2", "xg4")
N = 300

GEOMETRY_PROVENANCE = (
    "paper/iclr2027/frozen_sources/coordinate_free_geometry_v1/provenance_manifest.json"
)
GEOMETRY_RESULT = (
    "paper/iclr2027/frozen_sources/coordinate_free_geometry_v1/coordinate_free_geometry_result.json"
)

TASK_SOURCES = {
    "130M": {
        "path": "reports/reason_router_gen4_mamba130m_readout_alignment_analysis_v1/readout_alignment_pair_values.jsonl",
        "field": "Delta_L",
        "multiplier": 2.0,
        "expected_first": "xg1_fact_2701",
        "expected_last": "xg1_fact_3000",
        "expected_mean": 0.002240732073064253,
    },
    "370M": {
        "path": "reports/reason_router_gen4_mamba370m14b_readout_alignment_analysis_v1/readout_alignment_pair_values.jsonl",
        "field": "Delta_L_370M",
        "multiplier": 2.0,
        "expected_first": "xg1_fact_4801",
        "expected_last": "xg1_fact_5100",
        "expected_mean": 0.001017912703770835,
    },
    "790M": {
        "path": "reports/reason_router_gen4_mamba790m_readout_alignment_analysis_v1/readout_alignment_pair_values.jsonl",
        "field": "Delta_L_forward_equivalent",
        "multiplier": 1.0,
        "expected_first": "xg1_fact_7501",
        "expected_last": "xg1_fact_7800",
        "expected_mean": 0.004553422899712989,
    },
    "1.4B": {
        "path": "reports/reason_router_gen4_mamba370m14b_readout_alignment_analysis_v1/readout_alignment_pair_values.jsonl",
        "field": "Delta_L_1.4B",
        "multiplier": 2.0,
        "expected_first": "xg1_fact_4801",
        "expected_last": "xg1_fact_5100",
        "expected_mean": -0.002390895011772448,
    },
    "2.8B": {
        "path": "reports/reason_router_gen4_mamba28b_readout_alignment_analysis_v1/readout_alignment_pair_values.jsonl",
        "field": "Delta_L_forward_equivalent",
        "multiplier": 1.0,
        "expected_first": "xg1_fact_6601",
        "expected_last": "xg1_fact_6900",
        "expected_mean": -0.0001483499070046979,
    },
}

LM_SOURCES = {
    "130M": {
        "path": "reports/reason_router_gen4_mamba1_vanilla_lm_readout_runs/g4k-mamba130m-vanillalm-readout-xg1-2701-3000-p3-p5-2gpu-cb679ae/vanilla_lm_readout_items.jsonl",
        "expected_first": "xg1_fact_2701",
        "expected_last": "xg1_fact_3000",
        "expected_mean": -0.05641685,
    },
    "370M": {
        "path": "reports/reason_router_gen4_mamba1_vanilla_lm_readout_runs/g4k-mamba370m-vanillalm-readout-xg1-4801-5100-p3-p5-2gpu-9c56906-retry7/vanilla_lm_readout_items.jsonl",
        "expected_first": "xg1_fact_4801",
        "expected_last": "xg1_fact_5100",
        "expected_mean": -0.00197532,
    },
    "790M": {
        "path": "reports/reason_router_gen4_mamba1_vanilla_lm_readout_runs/g4k-mamba790m-vanillalm-readout-xg1-7501-7800-p2-p5-2gpu-e5f49b4/vanilla_lm_readout_items.jsonl",
        "expected_first": "xg1_fact_7501",
        "expected_last": "xg1_fact_7800",
        "expected_mean": -0.01596436,
    },
    "1.4B": {
        "path": "reports/reason_router_gen4_mamba1_vanilla_lm_readout_runs/g4k-mamba14b-vanillalm-readout-xg1-4801-5100-p5-p4-2gpu-e7ee5fa/vanilla_lm_readout_items.jsonl",
        "expected_first": "xg1_fact_4801",
        "expected_last": "xg1_fact_5100",
        "expected_mean": 0.00783318,
    },
    "2.8B": {
        "path": "reports/reason_router_gen4_mamba1_vanilla_lm_readout_runs/g4k-mamba28b-vanillalm-readout-xg1-6601-6900-p3-p5-2gpu-c3baf3d/vanilla_lm_readout_items.jsonl",
        "expected_first": "xg1_fact_6601",
        "expected_last": "xg1_fact_6900",
        "expected_mean": 0.01644840,
    },
}


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def git_bytes(ref: str, path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError(f"GIT_SHOW_FAILURE:{ref}:{path}") from exc


def git_text(ref: str, path: str) -> str:
    return git_bytes(ref, path).decode("utf-8-sig")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError("GIT_HEAD_FAILURE") from exc


def git_blob_sha(ref: str, path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", f"{ref}:{path}"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError(f"GIT_BLOB_SHA_FAILURE:{ref}:{path}") from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_bytes(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8-sig"))


def read_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in raw.decode("utf-8-sig").splitlines():
        if line.strip():
            item = json.loads(line)
            require(isinstance(item, dict), "JSONL_NON_OBJECT")
            rows.append(item)
    return rows


def verify_protocol(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"MISSING_PROTOCOL:{path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(value == EXPECTED_PROTOCOL, "PROTOCOL_MISMATCH")
    return value


def load_tensor_from_git(ref: str, path: str) -> torch.Tensor:
    raw = git_bytes(ref, path)
    try:
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(io.BytesIO(raw), map_location="cpu")
    require(torch.is_tensor(value), f"NOT_TENSOR:{path}")
    return value.detach().cpu().to(torch.float64).contiguous()


def row_normalized_gram(tensor: torch.Tensor) -> np.ndarray:
    require(tensor.ndim == 2 and tensor.shape[0] == N, "GEOMETRY_SHAPE")
    require(bool(torch.isfinite(tensor).all().item()), "GEOMETRY_NONFINITE")
    norms = torch.linalg.vector_norm(tensor, ord=2, dim=1)
    require(bool(torch.all(norms > 0).item()), "GEOMETRY_ZERO_ROW_NORM")
    normalized = tensor / norms[:, None]
    gram = (normalized @ normalized.T).numpy()
    require(np.isfinite(gram).all(), "GRAM_NONFINITE")
    return np.asarray(gram, dtype=np.float64)


def centered_inner_full(k: np.ndarray, l: np.ndarray) -> float:
    kc = k - k.mean(axis=1, keepdims=True) - k.mean(axis=0, keepdims=True) + k.mean()
    lc = l - l.mean(axis=1, keepdims=True) - l.mean(axis=0, keepdims=True) + l.mean()
    return float(np.sum(kc * lc))


def full_cka(k: np.ndarray, l: np.ndarray) -> float:
    num = centered_inner_full(k, l)
    den = math.sqrt(centered_inner_full(k, k) * centered_inner_full(l, l))
    require(den > 0.0, "FULL_CKA_ZERO_DENOM")
    value = num / den
    require(math.isfinite(value), "FULL_CKA_NONFINITE")
    return value


PAIR_INDEX = tuple((i, j) for i in range(5) for j in range(i + 1, 5))
ALL_INDEX = tuple((i, j) for i in range(5) for j in range(i, 5))


def weighted_cka_vectors(
    grams: list[np.ndarray],
    weights: np.ndarray,
    sample_n: int,
) -> np.ndarray:
    """
    CKA for resampled/weighted copies of the same N source pairs.

    weights[r, i] is the multiplicity of source pair i in replicate r.
    The formula is algebraically identical to constructing each expanded
    sample Gram matrix and applying H K H, but avoids materializing
    replicate x N x N arrays.
    """
    require(weights.ndim == 2 and weights.shape[1] == N, "WEIGHT_SHAPE")
    require(np.all(weights >= 0), "NEGATIVE_WEIGHT")
    require(np.all(weights.sum(axis=1) == sample_n), "WEIGHT_SUM")

    w = np.asarray(weights, dtype=np.float64)
    kw = [w @ k for k in grams]
    q = [np.sum(w * item, axis=1) for item in kw]

    centered_ip: dict[tuple[int, int], np.ndarray] = {}
    n = float(sample_n)
    for i, j in ALL_INDEX:
        had = grams[i] * grams[j]
        first = np.sum((w @ had) * w, axis=1)
        middle = np.sum(w * kw[i] * kw[j], axis=1)
        value = first - (2.0 / n) * middle + (q[i] * q[j]) / (n * n)
        centered_ip[(i, j)] = value

    result = np.empty((w.shape[0], 10), dtype=np.float64)
    for col, (i, j) in enumerate(PAIR_INDEX):
        denom = np.sqrt(centered_ip[(i, i)] * centered_ip[(j, j)])
        require(np.all(denom > 0.0), f"WEIGHTED_CKA_ZERO_DENOM:{i}:{j}")
        result[:, col] = centered_ip[(i, j)] / denom

    require(np.isfinite(result).all(), "WEIGHTED_CKA_NONFINITE")
    return result


def multinomial_weights(rng: np.random.Generator, replicates: int) -> np.ndarray:
    return rng.multinomial(N, np.full(N, 1.0 / N), size=replicates).astype(
        np.int16, copy=False
    )


def balanced_split_weights(
    rng: np.random.Generator, repeats: int
) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((repeats, N), dtype=np.int8)
    for r in range(repeats):
        chosen = rng.choice(N, size=N // 2, replace=False)
        a[r, chosen] = 1
    b = 1 - a
    return a, b


def pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    da = a - a.mean()
    db = b - b.mean()
    den = np.linalg.norm(da) * np.linalg.norm(db)
    require(float(den) > 0.0, "PEARSON_ZERO_DENOM")
    return float(np.dot(da, db) / den)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def split_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    pearson = pearson_1d(a, b)
    spearman = pearson_1d(rankdata_average(a), rankdata_average(b))
    diff = a - b
    mad = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return pearson, spearman, mad, rmse


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sd_sample": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "q025": quantile(values, 0.025),
        "q975": quantile(values, 0.975),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def load_geometry(
    source_head: str,
) -> tuple[dict[str, list[np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    prov_raw = git_bytes(source_head, GEOMETRY_PROVENANCE)
    frozen_raw = git_bytes(source_head, GEOMETRY_RESULT)
    prov = read_json_bytes(prov_raw)
    frozen = read_json_bytes(frozen_raw)

    require(prov["N"] == N, "GEOMETRY_PROV_N")
    require(tuple(prov["scale_order"]) == SCALE_ORDER, "GEOMETRY_PROV_SCALE_ORDER")
    require(tuple(prov["families"]) == FAMILIES, "GEOMETRY_PROV_FAMILY_ORDER")

    by_key = {
        (str(entry["family"]), str(entry["scale"])): entry
        for entry in prov["artifacts"]
    }
    grams: dict[str, list[np.ndarray]] = {family: [] for family in FAMILIES}
    consumed: list[dict[str, Any]] = []

    for family in FAMILIES:
        for scale in SCALE_ORDER:
            entry = by_key[(family, scale)]
            tensor_path = str(entry["tensor_path"])
            raw = git_bytes(source_head, tensor_path)
            actual_sha = sha256_bytes(raw)
            require(
                actual_sha == str(entry["tensor_sha256"]),
                f"GEOMETRY_TENSOR_SHA:{family}:{scale}",
            )
            tensor = load_tensor_from_git(source_head, tensor_path)
            require(
                list(tensor.shape) == list(entry["tensor_shape"]),
                f"GEOMETRY_TENSOR_SHAPE:{family}:{scale}",
            )
            grams[family].append(row_normalized_gram(tensor))
            consumed.append(
                {
                    "kind": "geometry_tensor",
                    "family": family,
                    "scale": scale,
                    "path": tensor_path,
                    "git_blob_sha1": git_blob_sha(source_head, tensor_path),
                    "sha256": actual_sha,
                    "shape": list(tensor.shape),
                    "historical_provenance": bool(entry["historical_provenance"]),
                }
            )

    # Recompute the full-sample CKA before any resampling and require exact
    # agreement (within floating tolerance) with the frozen v1 result.
    for family in FAMILIES:
        expected_matrix = np.asarray(
            frozen["families"][family]["cka_matrix"], dtype=np.float64
        )
        for i, j in PAIR_INDEX:
            observed = full_cka(grams[family][i], grams[family][j])
            expected = float(expected_matrix[i, j])
            require(
                abs(observed - expected) <= 2e-12,
                f"FROZEN_CKA_MISMATCH:{family}:{SCALE_ORDER[i]}:{SCALE_ORDER[j]}:"
                f"{observed}:{expected}",
            )

    metadata = {
        "geometry_provenance_path": GEOMETRY_PROVENANCE,
        "geometry_provenance_git_blob_sha1": git_blob_sha(
            source_head, GEOMETRY_PROVENANCE
        ),
        "geometry_provenance_sha256": sha256_bytes(prov_raw),
        "geometry_result_path": GEOMETRY_RESULT,
        "geometry_result_git_blob_sha1": git_blob_sha(source_head, GEOMETRY_RESULT),
        "geometry_result_sha256": sha256_bytes(frozen_raw),
    }
    return grams, consumed, metadata


def load_task_series(
    source_head: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    series: dict[str, np.ndarray] = {}
    consumed: list[dict[str, Any]] = []

    cache: dict[str, tuple[bytes, list[dict[str, Any]]]] = {}
    for scale in SCALE_ORDER:
        cfg = TASK_SOURCES[scale]
        path = str(cfg["path"])
        if path not in cache:
            raw = git_bytes(source_head, path)
            cache[path] = (raw, read_jsonl_bytes(raw))
        raw, rows = cache[path]
        require(len(rows) == N, f"TASK_ROW_COUNT:{scale}:{len(rows)}")
        ids = [str(row["source_pair_id"]) for row in rows]
        require(ids[0] == cfg["expected_first"], f"TASK_FIRST_ID:{scale}")
        require(ids[-1] == cfg["expected_last"], f"TASK_LAST_ID:{scale}")
        require(len(set(ids)) == N, f"TASK_DUPLICATE_ID:{scale}")

        values = np.asarray(
            [float(row[str(cfg["field"])]) * float(cfg["multiplier"]) for row in rows],
            dtype=np.float64,
        )
        require(np.isfinite(values).all(), f"TASK_NONFINITE:{scale}")
        observed_mean = float(values.mean())
        require(
            abs(observed_mean - float(cfg["expected_mean"])) <= 2e-12,
            f"TASK_POINT_MEAN_MISMATCH:{scale}:{observed_mean}",
        )
        series[scale] = values

        if not any(item["path"] == path for item in consumed):
            consumed.append(
                {
                    "kind": "task_pair_values",
                    "path": path,
                    "git_blob_sha1": git_blob_sha(source_head, path),
                    "sha256": sha256_bytes(raw),
                }
            )

    return series, consumed


def load_lm_series(
    source_head: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    series: dict[str, np.ndarray] = {}
    consumed: list[dict[str, Any]] = []

    for scale in SCALE_ORDER:
        cfg = LM_SOURCES[scale]
        path = str(cfg["path"])
        raw = git_bytes(source_head, path)
        rows = read_jsonl_bytes(raw)
        require(len(rows) == 2 * N, f"LM_ITEM_COUNT:{scale}:{len(rows)}")

        grouped: dict[str, dict[str, float]] = defaultdict(dict)
        order: list[str] = []
        seen = set()
        for row in rows:
            pair_id = str(row["source_pair_id"])
            cell = str(row["contrast_cell_id"])
            require(cell in {"C0_SHAM", "C2_NAME"}, f"LM_CELL:{scale}:{cell}")
            require(cell not in grouped[pair_id], f"LM_DUPLICATE_CELL:{scale}:{pair_id}")
            grouped[pair_id][cell] = float(row["task_matched"]["Delta_L_LM"])
            if pair_id not in seen:
                seen.add(pair_id)
                order.append(pair_id)

        require(len(order) == N, f"LM_PAIR_COUNT:{scale}:{len(order)}")
        require(order[0] == cfg["expected_first"], f"LM_FIRST_ID:{scale}")
        require(order[-1] == cfg["expected_last"], f"LM_LAST_ID:{scale}")

        values_list: list[float] = []
        for pair_id in order:
            cells = grouped[pair_id]
            require(
                set(cells) == {"C0_SHAM", "C2_NAME"},
                f"LM_MISSING_CELL:{scale}:{pair_id}",
            )
            values_list.append((cells["C0_SHAM"] + cells["C2_NAME"]) / 2.0)

        values = np.asarray(values_list, dtype=np.float64)
        require(np.isfinite(values).all(), f"LM_NONFINITE:{scale}")
        observed_mean = float(values.mean())
        # Manuscript values are printed to 8 decimal places, so this check is
        # intentionally against the printed value rather than a hidden constant.
        require(
            abs(observed_mean - float(cfg["expected_mean"])) <= 5e-9,
            f"LM_POINT_MEAN_MISMATCH:{scale}:{observed_mean}",
        )
        series[scale] = values
        consumed.append(
            {
                "kind": "lm_item_values",
                "scale": scale,
                "path": path,
                "git_blob_sha1": git_blob_sha(source_head, path),
                "sha256": sha256_bytes(raw),
            }
        )

    return series, consumed


def objective_bootstrap(
    task: dict[str, np.ndarray],
    lm: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    b = int(protocol["bootstrap_replicates"])
    seed = int(protocol["bootstrap_seed"])
    qlo, qhi = [float(x) for x in protocol["bootstrap_ci_quantiles"]]

    rows: list[dict[str, Any]] = []
    replicates: dict[str, np.ndarray] = {}

    for scale_index, scale in enumerate(SCALE_ORDER):
        rng = np.random.default_rng(seed + scale_index)
        indices = rng.integers(0, N, size=(b, N), endpoint=False)

        for objective, values in (("task", task[scale]), ("lm", lm[scale])):
            boot = values[indices].mean(axis=1)
            replicates[f"{scale}:{objective}"] = boot
            point = float(values.mean())
            low = quantile(boot, qlo)
            high = quantile(boot, qhi)
            rows.append(
                {
                    "scale": scale,
                    "objective": objective,
                    "N_pairs": N,
                    "point_mean": point,
                    "bootstrap_mean": float(boot.mean()),
                    "bootstrap_sd": float(boot.std(ddof=1)),
                    "ci_low": low,
                    "ci_high": high,
                    "ci_contains_zero": bool(low <= 0.0 <= high),
                    "bootstrap_positive_fraction": float(np.mean(boot > 0.0)),
                    "bootstrap_negative_fraction": float(np.mean(boot < 0.0)),
                    "bootstrap_zero_fraction": float(np.mean(boot == 0.0)),
                }
            )

    return rows, replicates


def geometry_bootstrap_and_split(
    grams: dict[str, list[np.ndarray]],
    frozen_result: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    b = int(protocol["bootstrap_replicates"])
    bootstrap_seed = int(protocol["bootstrap_seed"])
    split_repeats = int(protocol["split_half_repeats"])
    split_seed = int(protocol["split_half_seed"])
    half_n = int(protocol["split_half_size"])
    qlo, qhi = [float(x) for x in protocol["bootstrap_ci_quantiles"]]

    bootstrap_rows: list[dict[str, Any]] = []
    split_replicates: list[dict[str, Any]] = []
    split_summary: list[dict[str, Any]] = []

    for family_index, family in enumerate(FAMILIES):
        rng_boot = np.random.default_rng(bootstrap_seed + 100 + family_index)
        weights = multinomial_weights(rng_boot, b)
        boot_vectors = weighted_cka_vectors(grams[family], weights, N)

        frozen_matrix = np.asarray(
            frozen_result["families"][family]["cka_matrix"], dtype=np.float64
        )
        for col, (i, j) in enumerate(PAIR_INDEX):
            values = boot_vectors[:, col]
            low = quantile(values, qlo)
            high = quantile(values, qhi)
            bootstrap_rows.append(
                {
                    "family": family,
                    "scale_a": SCALE_ORDER[i],
                    "scale_b": SCALE_ORDER[j],
                    "N_pairs": N,
                    "point_cka": float(frozen_matrix[i, j]),
                    "bootstrap_mean": float(values.mean()),
                    "bootstrap_sd": float(values.std(ddof=1)),
                    "ci_low": low,
                    "ci_high": high,
                }
            )

        rng_split = np.random.default_rng(split_seed + family_index)
        wa, wb = balanced_split_weights(rng_split, split_repeats)
        va = weighted_cka_vectors(grams[family], wa, half_n)
        vb = weighted_cka_vectors(grams[family], wb, half_n)

        metrics = {
            "pearson": np.empty(split_repeats, dtype=np.float64),
            "spearman": np.empty(split_repeats, dtype=np.float64),
            "mad": np.empty(split_repeats, dtype=np.float64),
            "rmse": np.empty(split_repeats, dtype=np.float64),
        }
        for r in range(split_repeats):
            p, s, mad, rmse = split_metrics(va[r], vb[r])
            metrics["pearson"][r] = p
            metrics["spearman"][r] = s
            metrics["mad"][r] = mad
            metrics["rmse"][r] = rmse
            split_replicates.append(
                {
                    "family": family,
                    "replicate": r,
                    "pearson": p,
                    "spearman": s,
                    "mad": mad,
                    "rmse": rmse,
                }
            )

        for metric_name, values in metrics.items():
            info = summarize(values)
            split_summary.append(
                {
                    "family": family,
                    "metric": metric_name,
                    "repeats": split_repeats,
                    **info,
                }
            )

    return bootstrap_rows, split_replicates, split_summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"EMPTY_CSV:{path.name}")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(x: float) -> str:
    return f"{x:.8g}"


def build_report(
    protocol: dict[str, Any],
    objective_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    split_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# ICLR Pair-Resampling Robustness Analysis",
        "",
        "## Status",
        "",
        "`CPU_ONLY_FROZEN_ARTIFACT_RESAMPLING_RESULT`",
        "",
        f"Source HEAD: `{protocol['source_head']}`",
        "",
        "No model execution, tokenizer execution, training, evaluation, forward pass, "
        "backward pass, representation collection, row filtering, or response-guided "
        "selection occurred.",
        "",
        f"Bootstrap protocol: B={protocol['bootstrap_replicates']}, percentile 95% "
        f"pair-resampling intervals, fixed seed={protocol['bootstrap_seed']}.",
        "",
        f"Split-half protocol: {protocol['split_half_repeats']} repeated balanced "
        f"150/150 splits, fixed seed={protocol['split_half_seed']}.",
        "",
        "These intervals quantify pair-resampling stability on the frozen cohorts. "
        "They are not a noise ceiling and not an independent-population replication.",
        "",
        "## Objective mean pair-resampling",
        "",
        "| Scale | Objective | Point mean | 95% interval | Contains 0 | P(boot > 0) |",
        "|---|---|---:|---:|:---:|---:|",
    ]
    for scale in SCALE_ORDER:
        for objective in ("task", "lm"):
            row = next(
                item
                for item in objective_rows
                if item["scale"] == scale and item["objective"] == objective
            )
            lines.append(
                f"| {scale} | {objective} | {fmt_float(row['point_mean'])} | "
                f"[{fmt_float(row['ci_low'])}, {fmt_float(row['ci_high'])}] | "
                f"{'yes' if row['ci_contains_zero'] else 'no'} | "
                f"{row['bootstrap_positive_fraction']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Geometry joint pair-resampling",
            "",
            "The same bootstrap multiplicities are applied jointly to all five scales "
            "within a generator family. XG2 and XG4 remain separate.",
            "",
            "| Family | Scale pair | Point CKA | 95% interval |",
            "|---|---|---:|---:|",
        ]
    )
    for row in geometry_rows:
        lines.append(
            f"| {str(row['family']).upper()} | {row['scale_a']}--{row['scale_b']} | "
            f"{row['point_cka']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] |"
        )

    lines.extend(
        [
            "",
            "## Repeated balanced split-half stability",
            "",
            "Each half produces the fixed 10-dimensional off-diagonal cross-scale CKA "
            "vector. Metrics compare the two vectors within each split.",
            "",
            "| Family | Metric | Median | 2.5% | 97.5% |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in split_summary:
        lines.append(
            f"| {str(row['family']).upper()} | {row['metric']} | "
            f"{row['median']:.6f} | {row['q025']:.6f} | {row['q975']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Geometry intervals are sampling-robustness summaries for the existing "
            "300 controlled pairs, not model-remeasurement uncertainty.",
            "- Objective intervals are descriptive pair-resampling uncertainty. They "
            "do not retroactively create a prospective five-scale inferential family.",
            "- Point-estimate sign vectors remain point estimates; a cell whose "
            "resampling interval contains zero is not individually sign-stable under "
            "this resampling analysis.",
            "- No permutation control is included in this frozen stage.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = verify_protocol(args.protocol.resolve())
    source_head = str(protocol["source_head"])
    current_head = git_head()
    require(current_head == source_head, f"HEAD_MISMATCH:{current_head}:{source_head}")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    require(ROOT.resolve() in output_dir.parents, "OUTPUT_OUTSIDE_REPO")
    require(not output_dir.exists(), f"OUTPUT_EXISTS:{output_dir}")

    # Deterministic CPU-only numerical path.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    grams, geometry_consumed, geometry_meta = load_geometry(source_head)
    frozen_result = read_json_bytes(git_bytes(source_head, GEOMETRY_RESULT))
    task, task_consumed = load_task_series(source_head)
    lm, lm_consumed = load_lm_series(source_head)

    objective_rows, _objective_replicates = objective_bootstrap(task, lm, protocol)
    geometry_rows, split_replicates, split_summary = geometry_bootstrap_and_split(
        grams, frozen_result, protocol
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    provenance = {
        "schema_version": "iclr2027-pair-resampling-robustness-provenance-v1",
        "source_head": source_head,
        "current_head_at_execution": current_head,
        "protocol_path": args.protocol.resolve().relative_to(ROOT.resolve()).as_posix()
        if ROOT.resolve() in args.protocol.resolve().parents
        else str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "geometry_metadata": geometry_meta,
        "consumed_artifacts": geometry_consumed + task_consumed + lm_consumed,
        "scientific_model_forward_count": 0,
        "tokenizer_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "backward_executed": False,
        "gpu_required": False,
    }

    result = {
        "schema_version": "iclr2027-pair-resampling-robustness-result-v1",
        "protocol": protocol,
        "objective_bootstrap": objective_rows,
        "geometry_cka_bootstrap": geometry_rows,
        "split_half_summary": split_summary,
    }

    paths = {
        "result": output_dir / "analysis_result.json",
        "provenance": output_dir / "provenance_manifest.json",
        "objective": output_dir / "objective_bootstrap.csv",
        "geometry": output_dir / "geometry_cka_bootstrap.csv",
        "split_replicates": output_dir / "geometry_split_half_replicates.csv",
        "split_summary": output_dir / "geometry_split_half_summary.csv",
        "report": output_dir / "analysis_report.md",
    }

    paths["result"].write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(paths["objective"], objective_rows)
    write_csv(paths["geometry"], geometry_rows)
    write_csv(paths["split_replicates"], split_replicates)
    write_csv(paths["split_summary"], split_summary)
    paths["report"].write_text(
        build_report(protocol, objective_rows, geometry_rows, split_summary),
        encoding="utf-8",
    )

    sums = output_dir / "SHA256SUMS.txt"
    sums.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in sorted(paths.values(), key=lambda p: p.name)
        ),
        encoding="ascii",
    )

    print("RESULT=PASS_CPU_ONLY_PAIR_RESAMPLING")
    print(f"SOURCE_HEAD={source_head}")
    print("MODEL_FORWARD_COUNT=0")
    print("TOKENIZER_EXECUTED=NO")
    print("BACKWARD_EXECUTED=NO")
    print("GPU_REQUIRED=NO")
    print(f"OUTPUT_DIR={output_dir}")
    print("")
    print(paths["report"].read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
