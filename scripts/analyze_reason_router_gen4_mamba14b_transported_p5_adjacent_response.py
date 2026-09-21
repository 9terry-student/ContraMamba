#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
RAW_FREEZE_COMMIT = "1f1cb3e"

RAW_ROOT = Path(
    "reports/reason_router_gen4_mamba14b_transported_p5_adjacent_response_raw_runs/"
    "g4k-mamba14b-transported-p5-adjacent-response-5aae216-2t4"
)
RAW_ITEMS = RAW_ROOT / "transported_adjacent_response_items.jsonl"
RAW_SUMMARY = RAW_ROOT / "raw_response_summary.json"
RAW_MANIFEST = RAW_ROOT / "artifact_manifest.json"
RAW_SUMS = RAW_ROOT / "SHA256SUMS.txt"

HIST_ROOT = Path(
    "reports/reason_router_gen4_mamba14b_adjacent_site_specificity_raw_runs/"
    "g4k-mamba14b-adjacent-specificity-raw-xg1-5101-5400-2gpu-d8327f9"
)
HIST_ITEMS = HIST_ROOT / "paired_specificity_items.jsonl"
HIST_SUMMARY = HIST_ROOT / "raw_response_summary.json"

PINNED_GIT_BLOBS = {
    str(RAW_SUMS).replace("\\", "/"):
        "e867a831f21d72fd00beef0909ca1cdc92b97f26",
    str(RAW_MANIFEST).replace("\\", "/"):
        "315788fe070f50b1b53fcd6c54bf8e6b90ebf7b5",
    str(RAW_SUMMARY).replace("\\", "/"):
        "b431f8acdc021eb16a6ac7fae18e86a220661a90",
    str(RAW_ITEMS).replace("\\", "/"):
        "041de6a0667c685ebd558707993fd47ccf35642a",
    str(HIST_ITEMS).replace("\\", "/"):
        "64da91ba82a17982feab0e7c8bd3bfba87bade00",
    str(HIST_SUMMARY).replace("\\", "/"):
        "f1f1e764c3cd5c412d13be67555004b10795b649",
}

N = 300
ALPHA = 0.05
PRIMARY_ENDPOINT = "G=D_TRANSPORT-D_ADJ"
PRIMARY_TEST = "paired_one_sample_student_t_greater"
PRIMARY_P_VALUE_COUNT = 1
SIGN_GATE = "mean(D_TRANSPORT)>0"

PAIR_FILE = "pair_level_contrasts.jsonl"
ANALYSIS_FILE = "primary_analysis.json"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

RESULT_PASS = "PASS_MAMBA14B_TRANSPORTED_P5_ADJACENT_RESPONSE_STATIC_ANALYSIS"
CONCLUSION_RESTORED = (
    "TRANSPORTED_P5_POSITIVE_ADJACENT_RESPONSE_RESTORATION_SUPPORTED"
)
CONCLUSION_SHIFT_ONLY = (
    "TRANSPORTED_P5_RELATIVE_SHIFT_SUPPORTED_POSITIVE_RESTORATION_NOT_ESTABLISHED"
)
CONCLUSION_NOT = "TRANSPORTED_P5_RESPONSE_SHIFT_NOT_ESTABLISHED"


class StaticAnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StaticAnalysisError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticAnalysisError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
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


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"MISSING:{path}")
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
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


def validate_repo() -> str:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    head = git("rev-parse", "HEAD")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", RAW_FREEZE_COMMIT, head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "RAW_FREEZE_ANCESTOR_MISSING")

    for path, expected_blob in PINNED_GIT_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(
            observed == expected_blob,
            f"GIT_BLOB:{path}:{observed}",
        )
    return head


def validate_raw_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    raw_summary = json.loads(RAW_SUMMARY.read_text(encoding="utf-8"))
    raw_manifest = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    hist_summary = json.loads(HIST_SUMMARY.read_text(encoding="utf-8"))

    require(
        raw_summary["result"]
        == "PASS_MAMBA14B_TRANSPORTED_P5_ADJACENT_RESPONSE_RAW",
        "RAW_RESULT",
    )
    require(
        raw_manifest["result"]
        == "PASS_MAMBA14B_TRANSPORTED_P5_ADJACENT_RESPONSE_RAW",
        "RAW_MANIFEST_RESULT",
    )
    require(
        raw_summary["population"]["pair_count"] == N,
        "RAW_PAIR_COUNT",
    )
    require(
        raw_summary["population"]["pair_first"] == "xg1_fact_5101"
        and raw_summary["population"]["pair_last"] == "xg1_fact_5400",
        "RAW_PAIR_RANGE",
    )
    require(
        raw_summary["planned_static_analysis"]["primary_endpoint"]
        == PRIMARY_ENDPOINT,
        "RAW_PRIMARY_ENDPOINT",
    )
    require(
        raw_summary["planned_static_analysis"]["primary_test"]
        == PRIMARY_TEST,
        "RAW_PRIMARY_TEST",
    )
    require(
        raw_summary["planned_static_analysis"]["primary_p_value_count"]
        == PRIMARY_P_VALUE_COUNT,
        "RAW_PRIMARY_P_COUNT",
    )
    require(
        raw_summary["planned_static_analysis"]["sign_gate"] == SIGN_GATE,
        "RAW_SIGN_GATE",
    )
    require(
        raw_summary["planned_static_analysis"]["D_CAN_role"]
        == "descriptive_only",
        "RAW_D_CAN_ROLE",
    )
    boundary = raw_summary["boundary"]
    require(boundary["inferential_test_performed"] is False, "RAW_INFERENCE")
    require(boundary["historical_D_ADJ_accessed_during_execution"] is False, "RAW_D_ADJ_ACCESS")
    require(boundary["historical_D_CAN_accessed_during_execution"] is False, "RAW_D_CAN_ACCESS")
    require(boundary["row_subset_selection_executed"] is False, "RAW_ROW_SUBSET")
    require(boundary["epsilon_sweep_executed"] is False, "RAW_EPS_SWEEP")
    require(boundary["layer_sweep_executed"] is False, "RAW_LAYER_SWEEP")
    require(boundary["token_sweep_executed"] is False, "RAW_TOKEN_SWEEP")
    require(boundary["training_executed"] is False, "RAW_TRAINING")

    require(
        raw_manifest["historical_items_git_blob"]
        == PINNED_GIT_BLOBS[str(HIST_ITEMS).replace("\\", "/")],
        "HIST_ITEMS_PIN",
    )
    require(
        raw_manifest["historical_summary_git_blob"]
        == PINNED_GIT_BLOBS[str(HIST_SUMMARY).replace("\\", "/")],
        "HIST_SUMMARY_PIN",
    )
    require(
        hist_summary["result"]
        == "PASS_MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_RAW_RESPONSE",
        "HIST_RESULT",
    )
    require(
        hist_summary["pair_count"] == N
        and hist_summary["pair_first"] == "xg1_fact_5101"
        and hist_summary["pair_last"] == "xg1_fact_5400",
        "HIST_POPULATION",
    )
    return raw_summary, hist_summary


def descriptive(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    require(
        array.shape == (N,) and bool(np.isfinite(array).all()),
        "DESCRIPTIVE_VECTOR",
    )
    return {
        "mean": float(array.mean()),
        "sd_population": float(array.std(ddof=0)),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def analyze() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    execution_head = validate_repo()
    raw_summary, hist_summary = validate_raw_bundle()

    transported = read_jsonl(RAW_ITEMS)
    historical = read_jsonl(HIST_ITEMS)
    require(len(transported) == N and len(historical) == N, "ITEM_COUNT")

    expected_pairs = [f"xg1_fact_{index}" for index in range(5101, 5401)]
    require(
        [str(row["source_pair_id"]) for row in transported]
        == expected_pairs,
        "TRANSPORT_PAIR_ORDER",
    )
    require(
        [str(row["source_pair_id"]) for row in historical]
        == expected_pairs,
        "HIST_PAIR_ORDER",
    )

    pair_rows: list[dict[str, Any]] = []
    d_transport: list[float] = []
    d_adj: list[float] = []
    d_can: list[float] = []
    g_values: list[float] = []

    for index, (new, old) in enumerate(
        zip(transported, historical, strict=True)
    ):
        pair = expected_pairs[index]
        require(
            str(new["source_pair_id"]) == pair
            and str(old["source_pair_id"]) == pair,
            f"PAIR_ALIGNMENT:{pair}",
        )
        require(
            int(new["pair_index"]) == index
            and int(old["pair_index"]) == index,
            f"PAIR_INDEX:{pair}",
        )
        dt = float(new["D_TRANSPORT"])
        da = float(old["D_ADJ"])
        dc = float(old["D_CAN"])
        g = dt - da
        require(
            all(math.isfinite(v) for v in (dt, da, dc, g)),
            f"NONFINITE:{pair}",
        )
        d_transport.append(dt)
        d_adj.append(da)
        d_can.append(dc)
        g_values.append(g)
        pair_rows.append({
            "source_pair_id": pair,
            "pair_index": index,
            "D_TRANSPORT": dt,
            "D_ADJ": da,
            "D_CAN_descriptive_only": dc,
            "G": g,
        })

    g_array = np.asarray(g_values, dtype=np.float64)
    sd = float(g_array.std(ddof=1))
    require(sd > 0.0 and math.isfinite(sd), "PRIMARY_SD")
    test = stats.ttest_1samp(
        g_array,
        popmean=0.0,
        alternative="greater",
    )
    t_stat = float(test.statistic)
    p_value = float(test.pvalue)
    require(
        math.isfinite(t_stat) and 0.0 <= p_value <= 1.0,
        "PRIMARY_TEST_NONFINITE",
    )

    mean_g = float(g_array.mean())
    mean_transport = float(np.mean(np.asarray(d_transport, dtype=np.float64)))
    primary_contrast_pass = bool(mean_g > 0.0 and p_value < ALPHA)
    sign_gate_pass = bool(mean_transport > 0.0)

    if primary_contrast_pass and sign_gate_pass:
        scientific_conclusion = CONCLUSION_RESTORED
    elif primary_contrast_pass:
        scientific_conclusion = CONCLUSION_SHIFT_ONLY
    else:
        scientific_conclusion = CONCLUSION_NOT

    analysis = {
        "result": RESULT_PASS,
        "execution_head": execution_head,
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "population": {
            "pair_first": expected_pairs[0],
            "pair_last": expected_pairs[-1],
            "pair_count": N,
        },
        "primary": {
            "endpoint": PRIMARY_ENDPOINT,
            "test": "one_sample_student_t_on_paired_difference_G",
            "alternative": "greater",
            "alpha": ALPHA,
            "p_value_count": PRIMARY_P_VALUE_COUNT,
            "n": N,
            "df": N - 1,
            "mean_G": mean_g,
            "sd_G_sample": sd,
            "t_statistic": t_stat,
            "p_value": p_value,
            "fraction_G_positive": float(np.mean(g_array > 0.0)),
            "contrast_gate_mean_G_positive": bool(mean_g > 0.0),
            "primary_contrast_pass": primary_contrast_pass,
        },
        "mandatory_sign_gate": {
            "definition": SIGN_GATE,
            "mean_D_TRANSPORT": mean_transport,
            "pass": sign_gate_pass,
        },
        "descriptive_only": {
            "D_TRANSPORT": descriptive(d_transport),
            "historical_D_ADJ": descriptive(d_adj),
            "historical_D_CAN": descriptive(d_can),
            "G": descriptive(g_values),
        },
        "scientific_conclusion": scientific_conclusion,
        "interpretation": (
            "The transported-P5 response is significantly shifted upward relative "
            "to the same-pair frozen historical adjacent-P5 response, but its mean "
            "remains negative; therefore positive adjacent-response restoration is "
            "not established."
            if scientific_conclusion == CONCLUSION_SHIFT_ONLY
            else (
                "The transported-P5 response is significantly shifted upward relative "
                "to historical adjacent P5 and its mean is positive."
                if scientific_conclusion == CONCLUSION_RESTORED
                else
                "The prospectively specified transported-P5 response shift is not established."
            )
        ),
        "boundary": {
            "primary_p_value_count": 1,
            "D_CAN_inferential_role": "none_descriptive_only",
            "row_filter_performed": False,
            "rescue_performed": False,
            "training_executed": False,
            "gpu_execution_required": False,
            "raw_summary_claim_boundary_preserved": True,
            "historical_summary_execution_head":
                hist_summary["execution_head"],
            "raw_execution_head": raw_summary["execution_head"],
        },
    }
    return analysis, pair_rows


def write_bundle(output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    analysis, pair_rows = analyze()
    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / PAIR_FILE).write_bytes(jsonl_bytes(pair_rows))
    (output_dir / ANALYSIS_FILE).write_bytes(pretty_json_bytes(analysis))

    manifest = {
        "result": RESULT_PASS,
        "scientific_conclusion": analysis["scientific_conclusion"],
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_test": PRIMARY_TEST,
        "primary_p_value_count": 1,
        "sign_gate": SIGN_GATE,
        "raw_items_git_blob":
            PINNED_GIT_BLOBS[str(RAW_ITEMS).replace("\\", "/")],
        "historical_items_git_blob":
            PINNED_GIT_BLOBS[str(HIST_ITEMS).replace("\\", "/")],
        "pair_file_sha256": sha256_file(output_dir / PAIR_FILE),
        "analysis_file_sha256": sha256_file(output_dir / ANALYSIS_FILE),
        "rescue_performed": False,
        "training_executed": False,
    }
    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(manifest))

    names = (PAIR_FILE, ANALYSIS_FILE, MANIFEST_FILE)
    (output_dir / SUMS_FILE).write_text(
        "".join(
            f"{sha256_file(output_dir / name)}  {name}\n"
            for name in sorted(names)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return analysis


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static Study-A analysis for transported canonical P5 adjacent response. "
            "Computes exactly one primary p-value on G=D_TRANSPORT-D_ADJ."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    analysis = write_bundle(args.output_dir)
    primary = analysis["primary"]
    sign_gate = analysis["mandatory_sign_gate"]

    print("RESULT=" + analysis["result"])
    print("SCIENTIFIC_CONCLUSION=" + analysis["scientific_conclusion"])
    print("PAIR_COUNT=300")
    print("MEAN_D_TRANSPORT=" + repr(sign_gate["mean_D_TRANSPORT"]))
    print("MEAN_G=" + repr(primary["mean_G"]))
    print("T_STATISTIC=" + repr(primary["t_statistic"]))
    print("PRIMARY_P_VALUE=" + repr(primary["p_value"]))
    print("PRIMARY_P_VALUE_COUNT=1")
    print("PRIMARY_CONTRAST_PASS=" + str(primary["primary_contrast_pass"]))
    print("SIGN_GATE_PASS=" + str(sign_gate["pass"]))
    print("D_CAN_ROLE=descriptive_only")
    print("RESCUE_PERFORMED=False")
    print("GPU_EXECUTION_REQUIRED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
