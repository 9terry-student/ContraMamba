#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

N = 300
SCALES = ("mamba370m", "mamba14b")
CELLS = ("C0_SHAM", "C2_NAME")
ALPHA = 0.05

ITEM_FILE = "readout_alignment_items.jsonl"
SUMMARY_FILE = "raw_readout_alignment_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

RESULT_SUPPORTED = "READOUT_ALIGNMENT_SCALE_SIGN_REVERSAL_SUPPORTED"
RESULT_SEPARATED = "READOUT_ALIGNMENT_CROSS_SCALE_SEPARATION_WITHOUT_SIGN_REVERSAL"
RESULT_NOT = "READOUT_ALIGNMENT_SCALE_SIGN_REVERSAL_NOT_ESTABLISHED"

class AnalysisError(RuntimeError):
    pass

def require(ok: bool, message: str) -> None:
    if not ok:
        raise AnalysisError(message)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_sums(root: Path) -> None:
    path = root / SUMS_FILE
    require(path.is_file(), "SUMS_MISSING")
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require((root / name).is_file(), f"SUM_TARGET:{name}")
        require(sha256_file(root / name) == digest, f"SUM:{name}")
        seen.add(name)
    require(seen == {ITEM_FILE, SUMMARY_FILE, MANIFEST_FILE}, f"SUM_SET:{seen}")

def read_items(root: Path) -> list[dict[str, Any]]:
    validate_sums(root)
    summary = json.loads((root / SUMMARY_FILE).read_text(encoding="utf-8"))
    manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    require(summary["inferential_test_performed"] is False, "RAW_INFERENCE_BOUNDARY")
    require(manifest["primary_p_value_count_executed"] == 0, "RAW_P_COUNT")
    rows = [
        json.loads(line)
        for line in (root / ITEM_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(rows) == 1200, "ROW_COUNT")
    return rows

def pair_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["scale"]),
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(key not in by_key, f"DUPLICATE:{key}")
        by_key[key] = row

    out: dict[str, list[dict[str, Any]]] = {}
    expected_pairs = [f"xg1_fact_{i}" for i in range(4801, 5101)]
    for scale in SCALES:
        records = []
        for pair in expected_pairs:
            values = [
                float(by_key[(scale, pair, cell)]["Delta_L_row"])
                for cell in CELLS
            ]
            require(all(math.isfinite(x) for x in values), f"DELTA_NONFINITE:{scale}:{pair}")
            records.append({
                "source_pair_id": pair,
                "Delta_L": float(np.mean(np.asarray(values, dtype=np.float64))),
                "Delta_L_C0_SHAM": values[0],
                "Delta_L_C2_NAME": values[1],
            })
        out[scale] = records
    return out

def primary_inference(
    delta370: Sequence[float],
    delta14: Sequence[float],
) -> dict[str, Any]:
    a = np.asarray(delta370, dtype=np.float64)
    b = np.asarray(delta14, dtype=np.float64)
    require(a.shape == (N,) and b.shape == (N,), "PRIMARY_SHAPE")
    require(bool(np.isfinite(a).all()) and bool(np.isfinite(b).all()), "PRIMARY_NONFINITE")
    r = a - b
    sd = float(np.std(r, ddof=1))
    require(sd > 0.0 and math.isfinite(sd), "PRIMARY_SD")
    test = stats.ttest_1samp(r, 0.0, alternative="greater")
    p = float(test.pvalue)
    t = float(test.statistic)
    require(math.isfinite(t) and 0.0 <= p <= 1.0, "PRIMARY_TEST")
    mean370 = float(a.mean())
    mean14 = float(b.mean())
    mean_r = float(r.mean())
    primary_pass = bool(p < ALPHA and mean_r > 0.0)
    sign370 = mean370 > 0.0
    sign14 = mean14 < 0.0
    supported = bool(primary_pass and sign370 and sign14)
    if supported:
        label = RESULT_SUPPORTED
    elif primary_pass:
        label = RESULT_SEPARATED
    else:
        label = RESULT_NOT
    return {
        "result": label,
        "primary_endpoint": "R=Delta_L_370M-Delta_L_1.4B",
        "primary_test": {
            "test": "paired_one_sample_student_t",
            "alternative": "greater",
            "n": N,
            "df": N - 1,
            "alpha": ALPHA,
            "p_value_count": 1,
            "mean_R": mean_r,
            "sd_R": sd,
            "t_statistic": t,
            "p_value": p,
        },
        "sign_gates": {
            "mean_Delta_L_370M": mean370,
            "mean_Delta_L_1.4B": mean14,
            "gate_370M_positive": sign370,
            "gate_1.4B_negative": sign14,
        },
        "sign_reversal_supported": supported,
    }

def descriptive(x: Sequence[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    require(bool(np.isfinite(a).all()), "DESC_NONFINITE")
    return {
        "mean": float(a.mean()),
        "sd_population": float(a.std(ddof=0)),
        "min": float(a.min()),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "max": float(a.max()),
        "positive_fraction": float(np.mean(a > 0.0)),
    }

def analyze(raw_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_items(raw_dir)
    pairs = pair_values(rows)
    d370 = [float(r["Delta_L"]) for r in pairs["mamba370m"]]
    d14 = [float(r["Delta_L"]) for r in pairs["mamba14b"]]
    result = primary_inference(d370, d14)
    result["descriptive"] = {
        "mamba370m_Delta_L": descriptive(d370),
        "mamba14b_Delta_L": descriptive(d14),
        "pairwise_cross_scale_pearson": (
            None
            if float(np.std(d370)) == 0.0 or float(np.std(d14)) == 0.0
            else float(np.corrcoef(np.asarray(d370), np.asarray(d14))[0, 1])
        ),
    }
    pair_rows = []
    for r370, r14 in zip(pairs["mamba370m"], pairs["mamba14b"], strict=True):
        require(r370["source_pair_id"] == r14["source_pair_id"], "PAIR_ALIGNMENT")
        pair_rows.append({
            "source_pair_id": r370["source_pair_id"],
            "Delta_L_370M": r370["Delta_L"],
            "Delta_L_1.4B": r14["Delta_L"],
            "R": float(r370["Delta_L"] - r14["Delta_L"]),
        })
    return result, pair_rows

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    require(not args.output_dir.exists(), "OUTPUT_COLLISION")
    result, pair_rows = analyze(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_raw = (
        json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()
    pair_raw = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
        for row in pair_rows
    )
    (args.output_dir / "readout_alignment_analysis.json").write_bytes(result_raw)
    (args.output_dir / "readout_alignment_pair_values.jsonl").write_bytes(pair_raw)
    manifest = {
        "result": result["result"],
        "primary_p_value_count": 1,
        "rescue_performed": False,
        "row_filter_performed": False,
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    names = (
        "readout_alignment_analysis.json",
        "readout_alignment_pair_values.jsonl",
        "artifact_manifest.json",
    )
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(args.output_dir/name)}  {name}\n" for name in sorted(names)),
        encoding="utf-8",
        newline="\n",
    )
    print("RESULT=" + result["result"])
    print("PRIMARY_P_VALUE_COUNT=1")
    print("SIGN_REVERSAL_SUPPORTED=" + str(result["sign_reversal_supported"]))
    print("RESCUE_PERFORMED=False")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
