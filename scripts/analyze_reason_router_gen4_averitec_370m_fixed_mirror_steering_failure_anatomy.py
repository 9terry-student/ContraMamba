#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

RAW_ROOT = Path(
    "reports/reason_router_gen4_averitec_370m_fixed_mirror_steering_runs/"
    "g4k-averitec370-fixed-mirror-steering-raw-2799-25a626d"
)
RAW_ROWS = RAW_ROOT / "steering_raw_rows.jsonl"
RAW_SUMMARY = RAW_ROOT / "execution_summary.json"
RAW_SUMS = RAW_ROOT / "SHA256SUMS.txt"
PRIOR_ANALYSIS = Path(
    "reports/reason_router_gen4_averitec_370m_fixed_mirror_steering_analysis_328ae53/"
    "steering_analysis.json"
)

RAW_ROWS_BLOB = "52e07a7c93f96eb6d10f7b87e38a0aee7fb6ac2d"
RAW_SUMMARY_BLOB = "b35b4ed1aa5b53e896abd45fb8f22cb273c6660f"
RAW_SUMS_BLOB = "6c5d1825378f0207c0bd560e968b9c9360cb093d"
PRIOR_ANALYSIS_BLOB = "a3b078063b4dd83fa54270fabc89f3d17832a2b6"

RAW_ROWS_SHA256 = "912fb91bdfc5e778e4d2f99cb9a8f7a5bdb8a4a1389b39de1d47947eb9058d0c"
RAW_SUMMARY_SHA256 = "ad9497956b8db7b9d3930e13437d7d6daf5c33e3b800999e0d3ee2787cca4f7e"
RAW_SUMS_SHA256 = "372d243e5bb6acc1f0e4ad1309fb446f474e813acbdedb9466fbd56940614465"

N = 2799
RAW_ROW_COUNT = 8397
CONDITIONS = ("native", "p3_mirror_steer", "p5_matched_control")
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")

SCHEMA = "gen4-averitec-370m-fixed-mirror-steering-failure-anatomy-v1"
RESULT = "PASS_AVERITEC_370M_FIXED_MIRROR_STEERING_FAILURE_ANATOMY"
JSON_FILE = "steering_failure_anatomy.json"
REPORT_FILE = "steering_failure_anatomy.md"
SUMS_FILE = "SHA256SUMS.txt"

class AnatomyError(RuntimeError):
    pass

def req(ok: bool, msg: str) -> None:
    if not ok:
        raise AnatomyError(msg)

def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnatomyError("GIT_FAILURE:" + " ".join(args)) from exc

def blob(path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "cat-file", "blob", f"HEAD:{path.as_posix()}"],
            cwd=ROOT, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnatomyError(f"GIT_BLOB:{path}") from exc

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def finite(x: Any, label: str) -> float:
    y = float(x)
    req(math.isfinite(y), f"NONFINITE:{label}")
    return y

def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(float(x) for x in values)
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return v[lo]
    w = pos - lo
    return v[lo] + w * (v[hi] - v[lo])

def desc(values: Sequence[float]) -> dict[str, Any]:
    v = [finite(x, "DESC") for x in values]
    req(bool(v), "DESC_EMPTY")
    return {
        "n": len(v),
        "mean": statistics.fmean(v),
        "median": quantile(v, .5),
        "q10": quantile(v, .1),
        "q25": quantile(v, .25),
        "q75": quantile(v, .75),
        "q90": quantile(v, .9),
        "q95": quantile(v, .95),
        "min": min(v),
        "max": max(v),
        "fraction_positive": sum(x > 0 for x in v) / len(v),
        "fraction_negative": sum(x < 0 for x in v) / len(v),
        "fraction_zero": sum(x == 0 for x in v) / len(v),
    }

def authenticate(expected_head: str) -> None:
    req(git("branch","--show-current") in ("", EXPECTED_BRANCH), "BRANCH")
    req(git("rev-parse","HEAD") == expected_head, "HEAD")
    req(git("status","--porcelain") == "", "WORKTREE")
    for p, expected in {
        RAW_ROWS: RAW_ROWS_BLOB,
        RAW_SUMMARY: RAW_SUMMARY_BLOB,
        RAW_SUMS: RAW_SUMS_BLOB,
        PRIOR_ANALYSIS: PRIOR_ANALYSIS_BLOB,
    }.items():
        req(git("rev-parse", f"HEAD:{p.as_posix()}") == expected, f"BLOB:{p}")

def load() -> tuple[list[dict[str,Any]], dict[str,Any], dict[str,Any]]:
    rr, sr, xr, pr = blob(RAW_ROWS), blob(RAW_SUMMARY), blob(RAW_SUMS), blob(PRIOR_ANALYSIS)
    req(sha256_bytes(rr) == RAW_ROWS_SHA256, "ROWS_SHA")
    req(sha256_bytes(sr) == RAW_SUMMARY_SHA256, "SUMMARY_SHA")
    req(sha256_bytes(xr) == RAW_SUMS_SHA256, "SUMS_SHA")

    declared = {}
    for line in xr.decode().splitlines():
        if line.strip():
            d,n = line.split("  ",1)
            declared[n] = d
    req(declared == {
        "execution_summary.json": RAW_SUMMARY_SHA256,
        "steering_raw_rows.jsonl": RAW_ROWS_SHA256,
    }, "SUMS_CONTENT")

    rows = [json.loads(line) for line in rr.decode().splitlines() if line.strip()]
    summary = json.loads(sr)
    prior = json.loads(pr)

    req(len(rows) == RAW_ROW_COUNT, "ROW_COUNT")
    req(summary["fresh_cohort_count"] == N, "N")
    req(summary["conditions"] == list(CONDITIONS), "CONDITIONS")
    req(summary["p_value_count_added"] == 0, "RAW_P")
    req(summary["rescue_performed"] is False, "RAW_RESCUE")
    req(summary["magnitude_search_performed"] is False, "RAW_MAG_SEARCH")
    req(prior["result"] == "AVERITEC_370M_FIXED_MIRROR_P3_STEERING_NOT_ESTABLISHED", "PRIOR_RESULT")
    req(prior["primary_endpoint"]["C_corrections"] == 0, "CORR")
    req(prior["primary_endpoint"]["D_damages"] == 0, "DMG")

    for i,row in enumerate(rows):
        logits = [finite(x, f"logit:{i}") for x in row["final_logits"]]
        req(len(logits) == 3, f"LOGIT_N:{i}")
        gold = int(row["correct_label_id"])
        margin = logits[gold] - max(logits[j] for j in range(3) if j != gold)
        req(math.isclose(finite(row["correct_class_logit_margin"], f"m:{i}"),
                         margin, rel_tol=0, abs_tol=1e-12), f"MARGIN:{i}")

    for i in range(0,len(rows),3):
        g = rows[i:i+3]
        req(tuple(str(x["condition"]) for x in g) == CONDITIONS, f"ORDER:{i}")
        req(len({str(x["example_id"]) for x in g}) == 1, f"ID:{i}")
    return rows, summary, prior

def bucket_lambda(x: float) -> str:
    if x <= 2: return "(1,2]"
    if x <= 5: return "(2,5]"
    if x <= 10: return "(5,10]"
    if x <= 100: return "(10,100]"
    return ">100"

def analyze(rows: Sequence[Mapping[str,Any]], head: str) -> dict[str,Any]:
    native_errors = []
    all_examples = []
    by_label: dict[str,list[dict[str,Any]]] = {k:[] for k in LABELS}

    for i in range(0,len(rows),3):
        native, steer, control = rows[i:i+3]
        mn = finite(native["correct_class_logit_margin"],"mn")
        ms = finite(steer["correct_class_logit_margin"],"ms")
        mc = finite(control["correct_class_logit_margin"],"mc")
        ds = ms - mn
        dc = mn - mc
        audit = steer["intervention_audit"]
        req(isinstance(audit, dict), "STEER_AUDIT")
        rec = {
            "example_id": str(native["example_id"]),
            "gold": str(native["correct_label"]),
            "native_correct": bool(native["is_correct"]),
            "m_native": mn,
            "m_steer": ms,
            "m_control": mc,
            "delta_steer": ds,
            "delta_control_symmetric": dc,
            "mirror_asymmetry": ds - dc,
            "p3_a": finite(audit["native_selected_a"],"p3a"),
            "p3_b": finite(audit["native_selected_b"],"p3b"),
            "p3_theta": math.atan2(
                finite(audit["native_selected_b"],"p3b2"),
                finite(audit["native_selected_a"],"p3a2"),
            ),
            "correction_l2": finite(audit["correction_l2"],"corr_l2"),
        }
        all_examples.append(rec)
        if not rec["native_correct"]:
            req(mn <= 0.0, "ERROR_MARGIN_POSITIVE")
            deficit = -mn
            rec["native_margin_deficit"] = deficit
            if ds > 0 and deficit > 0:
                lam = deficit / ds
                rec["lambda_star_linearized"] = lam
                rec["fraction_deficit_closed"] = ds / deficit
                req(ms <= 0.0, "UNEXPECTED_ACTUAL_CORRECTION")
                req(lam >= 1.0 - 1e-12, "LAMBDA_LT_ONE_WITHOUT_FLIP")
            else:
                rec["lambda_star_linearized"] = None
                rec["fraction_deficit_closed"] = None
            native_errors.append(rec)
            by_label[rec["gold"]].append(rec)

    req(len(native_errors) == 2468, "NATIVE_ERROR_N")

    helpful = [r for r in native_errors if r["delta_steer"] > 0]
    nonhelp = [r for r in native_errors if r["delta_steer"] <= 0]
    lambdas = [r["lambda_star_linearized"] for r in helpful if r["lambda_star_linearized"] is not None]
    closure = [r["fraction_deficit_closed"] for r in helpful if r["fraction_deficit_closed"] is not None]

    lambda_buckets = {k:0 for k in ("(1,2]","(2,5]","(5,10]","(10,100]",">100")}
    for x in lambdas:
        lambda_buckets[bucket_lambda(float(x))] += 1

    label_summary = {}
    for label, rs in by_label.items():
        hs = [r for r in rs if r["delta_steer"] > 0]
        ls = [r["lambda_star_linearized"] for r in hs if r["lambda_star_linearized"] is not None]
        label_summary[label] = {
            "n_native_errors": len(rs),
            "helpful_direction_count": len(hs),
            "helpful_direction_fraction": (len(hs)/len(rs)) if rs else None,
            "delta_steer": desc([r["delta_steer"] for r in rs]) if rs else None,
            "lambda_star_linearized_helpful": desc(ls) if ls else None,
        }

    preferred = sum(
        r["m_steer"] > r["m_native"] > r["m_control"]
        for r in all_examples
    )

    return {
        "schema_version": SCHEMA,
        "result": RESULT,
        "analysis_head": head,
        "source_result": "AVERITEC_370M_FIXED_MIRROR_P3_STEERING_NOT_ESTABLISHED",
        "population": {
            "N": N,
            "native_error_count": len(native_errors),
            "native_correct_count": N-len(native_errors),
        },
        "continuous_displacement": {
            "all_examples_delta_steer": desc([r["delta_steer"] for r in all_examples]),
            "all_examples_native_minus_control": desc([r["delta_control_symmetric"] for r in all_examples]),
            "all_examples_mirror_asymmetry": desc([r["mirror_asymmetry"] for r in all_examples]),
            "preferred_Msteer_gt_Mnative_gt_Mcontrol_count": preferred,
            "preferred_fraction": preferred / N,
        },
        "native_error_failure_decomposition": {
            "helpful_direction_count": len(helpful),
            "helpful_direction_fraction": len(helpful)/len(native_errors),
            "nonhelpful_or_zero_direction_count": len(nonhelp),
            "nonhelpful_or_zero_direction_fraction": len(nonhelp)/len(native_errors),
            "delta_steer_all_native_errors": desc([r["delta_steer"] for r in native_errors]),
            "delta_steer_helpful_only": desc([r["delta_steer"] for r in helpful]) if helpful else None,
            "native_margin_deficit": desc([r["native_margin_deficit"] for r in native_errors]),
            "lambda_star_linearized_helpful_only": desc(lambdas) if lambdas else None,
            "fraction_deficit_closed_helpful_only": desc(closure) if closure else None,
            "lambda_star_buckets_helpful_only": lambda_buckets,
            "actual_prediction_flip_count": 0,
        },
        "label_descriptive": label_summary,
        "p3_native_coordinate_descriptive_on_native_errors": {
            "a": desc([r["p3_a"] for r in native_errors]),
            "b": desc([r["p3_b"] for r in native_errors]),
            "theta_radians": desc([r["p3_theta"] for r in native_errors]),
            "correction_l2": desc([r["correction_l2"] for r in native_errors]),
        },
        "interpretation_boundary": {
            "lambda_star_definition":
                "For native errors with delta_steer>0 only: -m_native/(m_steer-m_native).",
            "lambda_star_is":
                "A descriptive local-linear extrapolation of the already observed unit mirror displacement; it is not an executed steering coefficient and does not establish what a scaled intervention would do.",
            "no_new_p_values": True,
            "no_model_execution": True,
            "no_rescue": True,
            "no_magnitude_search": True,
            "no_feature_selection": True,
        },
    }

def render(a: Mapping[str,Any]) -> str:
    d = a["native_error_failure_decomposition"]
    c = a["continuous_displacement"]
    l = d["lambda_star_linearized_helpful_only"]
    lines = [
        "# Experiment 3 — Static Steering Failure Anatomy",
        "",
        f"Result: `{a['result']}`",
        "",
        "This is descriptive anatomy of the frozen failed steering experiment. It adds no p-values and performs no model execution.",
        "",
        "## Native-error decomposition",
        "",
        f"- Native errors: `{a['population']['native_error_count']}`.",
        f"- Helpful margin direction (`M_steer > M_native`): `{d['helpful_direction_count']}` / `{a['population']['native_error_count']}` = `{d['helpful_direction_fraction']}`.",
        f"- Non-helpful or zero direction: `{d['nonhelpful_or_zero_direction_count']}` / `{a['population']['native_error_count']}` = `{d['nonhelpful_or_zero_direction_fraction']}`.",
        "- Actual prediction flips among native errors: `0`.",
        "",
        "## Continuous mirror displacement",
        "",
        f"- Mean `M_steer-M_native` over all examples: `{c['all_examples_delta_steer']['mean']}`.",
        f"- Mean `M_native-M_control` over all examples: `{c['all_examples_native_minus_control']['mean']}`.",
        f"- Mean mirror asymmetry: `{c['all_examples_mirror_asymmetry']['mean']}`.",
        f"- `M_steer > M_native > M_control`: `{c['preferred_fraction']}` of all examples.",
        "",
        "## Local-linear leverage diagnostic",
        "",
    ]
    if l:
        lines += [
            f"- Helpful-direction median `lambda*`: `{l['median']}`.",
            f"- Q25/Q75: `[{l['q25']}, {l['q75']}]`.",
            f"- Q90/Q95: `[{l['q90']}, {l['q95']}]`.",
            f"- Range: `[{l['min']}, {l['max']}]`.",
        ]
    lines += [
        "",
        "`lambda*` is not an executed coefficient. It is only the local-linear amount of the observed unit displacement that would be required to close the native correct-class margin deficit if the measured displacement scaled linearly.",
        "",
        "## Interpretation boundary",
        "",
        "This anatomy separates two descriptive failure modes: itemwise direction can fail to improve the correct-class margin, and even when the direction is favorable the observed unit displacement can be too small to reach the decision boundary. It does not rescue Experiment 3 and does not authorize a magnitude sweep.",
        "",
    ]
    return "\n".join(lines)

def write(out: Path, a: Mapping[str,Any]) -> None:
    req(not out.exists(), "OUTPUT_COLLISION")
    out.mkdir(parents=True)
    jp, rp, sp = out/JSON_FILE, out/REPORT_FILE, out/SUMS_FILE
    jp.write_text(json.dumps(a,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    rp.write_text(render(a),encoding="utf-8",newline="\n")
    hashes={JSON_FILE:sha256_file(jp),REPORT_FILE:sha256_file(rp)}
    sp.write_text("".join(f"{d}  {n}\n" for n,d in sorted(hashes.items())),encoding="utf-8",newline="\n")

def main(argv: Sequence[str] | None=None) -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args(argv)
    authenticate(args.expected_head)
    rows,_,_=load()
    a=analyze(rows,args.expected_head)
    write(args.output_dir,a)
    d=a["native_error_failure_decomposition"]
    print("RESULT="+a["result"])
    print("NATIVE_ERROR_COUNT="+str(a["population"]["native_error_count"]))
    print("HELPFUL_DIRECTION_COUNT="+str(d["helpful_direction_count"]))
    print("HELPFUL_DIRECTION_FRACTION="+format(d["helpful_direction_fraction"],".17g"))
    print("NONHELPFUL_DIRECTION_COUNT="+str(d["nonhelpful_or_zero_direction_count"]))
    if d["lambda_star_linearized_helpful_only"]:
        print("LAMBDA_STAR_MEDIAN="+format(d["lambda_star_linearized_helpful_only"]["median"],".17g"))
        print("LAMBDA_STAR_Q25="+format(d["lambda_star_linearized_helpful_only"]["q25"],".17g"))
        print("LAMBDA_STAR_Q75="+format(d["lambda_star_linearized_helpful_only"]["q75"],".17g"))
    print("NEW_P_VALUE_COUNT=0")
    print("MODEL_FORWARD_COUNT=0")
    print("RESCUE_PERFORMED=False")

if __name__=="__main__":
    main()
