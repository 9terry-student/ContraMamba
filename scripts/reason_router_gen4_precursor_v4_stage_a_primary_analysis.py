#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "2adc5442c8a642e17b6156f075e2d766e0e8a756"

DESIGN = Path(
    "reports/reason_router_gen4_precursor_v4_analytic_vjp_susceptibility_design_candidate.md"
)
DESIGN_GIT_BLOB = "4f2e04de11f668e2a592971ad7952eb1863bd538"

RAW_ROOT = Path(
    "reports/reason_router_gen4_precursor_v4_analytic_stage_a_runs/"
    "g4k-precursorv4-analytic-stagea-n800-71b3417-gpu2"
)
RAW_SUMS = RAW_ROOT / "SHA256SUMS.txt"
RAW_SUMMARY = RAW_ROOT / "execution_summary.json"
RAW_ROWS = RAW_ROOT / "precursor_v4_analytic_stage_a_raw_rows.jsonl"

RAW_SUMS_GIT_BLOB = "1455295d65830fbcc69911c41bd79774ca23be3e"
RAW_SUMMARY_GIT_BLOB = "063c51d7d926dc51c4cfb9b5f05ddba8525215ac"
RAW_ROWS_GIT_BLOB = "3c890ed8496fb495dc7804731615889b8bb69f55"

RAW_SUMMARY_SHA256 = (
    "eb159f8cb736aeef8cbbb15cd7af406665460266f33e5011a100c4eb7e3b270d"
)
RAW_ROWS_SHA256 = (
    "ee323cdf154b50914c1c02dc1ea34594993d375b7700045f6213b2c6f7d9e726"
)

N = 800
GROUP_ORDER = ("unsupported", "supported")
MIN_GROUP_N = 30
ALPHA = 0.05
PRIMARY_TEST = "two_sided_welch_independent_samples_t"
PRIMARY_ENDPOINT = "z_i"
P_VALUE_COUNT = 1

REPORT_FILE = "precursor_v4_stage_a_primary_analysis.json"
SUMS_FILE = "SHA256SUMS.txt"
REPORT_SCHEMA = "gen4-precursor-v4-stage-a-primary-analysis-v1"
RESULT_PASS = "PASS_PRECURSOR_V4_STAGE_A_PRIMARY_ANALYSIS"
CONCLUSION_OBSERVED = (
    "LOCAL_DIFFERENTIAL_SUSCEPTIBILITY_TEMPORAL_PRECEDENCE_OBSERVED"
)
CONCLUSION_NOT_SUPPORTED = (
    "LOCAL_DIFFERENTIAL_SUSCEPTIBILITY_TEMPORAL_PRECEDENCE_NOT_SUPPORTED"
)
CONCLUSION_NOT_ESTIMABLE = "PRECURSOR_V4_STAGE_A_NOT_ESTIMABLE"


class PrecursorV4AnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PrecursorV4AnalysisError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrecursorV4AnalysisError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            REQUIRED_ANCESTOR,
            expected_head,
        )
        == 0,
        "RAW_FREEZE_NOT_ANCESTOR",
    )

    pinned = {
        DESIGN: DESIGN_GIT_BLOB,
        RAW_SUMS: RAW_SUMS_GIT_BLOB,
        RAW_SUMMARY: RAW_SUMMARY_GIT_BLOB,
        RAW_ROWS: RAW_ROWS_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path.as_posix()}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_protocol() -> None:
    require(N == 800, "N")
    require(GROUP_ORDER == ("unsupported", "supported"), "GROUP_ORDER")
    require(MIN_GROUP_N == 30, "MIN_GROUP_N")
    require(ALPHA == 0.05, "ALPHA")
    require(
        PRIMARY_TEST == "two_sided_welch_independent_samples_t",
        "PRIMARY_TEST",
    )
    require(PRIMARY_ENDPOINT == "z_i", "PRIMARY_ENDPOINT")
    require(P_VALUE_COUNT == 1, "P_VALUE_COUNT")


def validate_raw_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sums_path = ROOT / RAW_SUMS
    summary_path = ROOT / RAW_SUMMARY
    rows_path = ROOT / RAW_ROWS

    require(sums_path.is_file(), "RAW_SUMS_MISSING")
    require(summary_path.is_file(), "RAW_SUMMARY_MISSING")
    require(rows_path.is_file(), "RAW_ROWS_MISSING")
    require(
        sha256_file(summary_path) == RAW_SUMMARY_SHA256,
        "RAW_SUMMARY_SHA256",
    )
    require(
        sha256_file(rows_path) == RAW_ROWS_SHA256,
        "RAW_ROWS_SHA256",
    )

    manifest: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        manifest[name] = digest

    require(
        manifest
        == {
            "execution_summary.json": RAW_SUMMARY_SHA256,
            "precursor_v4_analytic_stage_a_raw_rows.jsonl":
                RAW_ROWS_SHA256,
        },
        "RAW_SUMS_CONTENT",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(
        summary["result"] == "PASS_PRECURSOR_V4_ANALYTIC_STAGE_A_RAW",
        "RAW_RESULT",
    )
    require(summary["cohort"]["count"] == N, "RAW_N")
    require(
        summary["scientific_inference_executed"] is False,
        "RAW_PRIOR_INFERENCE",
    )
    require(
        summary["primary_test_executed"] is False,
        "RAW_PRIOR_TEST",
    )
    require(summary["p_value_count_added"] == 0, "RAW_PRIOR_P_VALUE")
    require(summary["scientific_conclusion"] is None, "RAW_PRIOR_CONCLUSION")
    require(
        summary["endpoint"]
        == "Z_i=mean(D_t over t*-4,t*-3,t*-2,t*-1)",
        "RAW_ENDPOINT",
    )

    rows: list[dict[str, Any]] = []
    with rows_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f"RAW_ROW_OBJECT:{line_no}")
            rows.append(row)

    require(len(rows) == N, "RAW_ROW_COUNT")
    require(
        len({str(row["example_id"]) for row in rows}) == N,
        "RAW_ROW_IDS",
    )
    return summary, rows


def extract_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    groups = {name: [] for name in GROUP_ORDER}
    for index, row in enumerate(rows):
        group = str(row["primary_stage_a_group"])
        require(group in groups, f"GROUP:{index}:{group}")
        value = float(row[PRIMARY_ENDPOINT])
        require(math.isfinite(value), f"Z_NONFINITE:{index}")
        require(row["primary_inference_executed"] is False, f"ROW_INFERENCE:{index}")
        require(int(row["p_value_count_added"]) == 0, f"ROW_PVALUE:{index}")
        groups[group].append(value)

    require(
        sum(len(values) for values in groups.values()) == len(rows),
        "GROUP_TOTAL",
    )
    return groups


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 400
    epsilon = 3.0e-14
    floor = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    h = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = (
            m * (b - m) * x
            / ((qam + m2) * (a + m2))
        )
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        h *= d * c

        aa = (
            -(a + m) * (qab + m) * x
            / ((a + m2) * (qap + m2))
        )
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < epsilon:
            require(math.isfinite(h), "BETA_CF_NONFINITE")
            return h

    raise PrecursorV4AnalysisError("BETA_CF_NO_CONVERGENCE")


def regularized_incomplete_beta(
    a: float,
    b: float,
    x: float,
) -> float:
    require(a > 0.0 and b > 0.0, "BETA_PARAMETERS")
    require(0.0 <= x <= 1.0, "BETA_X")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)

    if x < (a + 1.0) / (a + b + 2.0):
        value = (
            front
            * _beta_continued_fraction(a, b, x)
            / a
        )
    else:
        value = 1.0 - (
            front
            * _beta_continued_fraction(b, a, 1.0 - x)
            / b
        )

    require(math.isfinite(value), "BETA_NONFINITE")
    require(-1.0e-14 <= value <= 1.0 + 1.0e-14, "BETA_RANGE")
    return min(1.0, max(0.0, value))


def student_t_two_sided_p_value(t_statistic: float, df: float) -> float:
    require(math.isfinite(t_statistic), "T_NONFINITE")
    require(math.isfinite(df) and df > 0.0, "DF")
    x = df / (df + t_statistic * t_statistic)
    value = regularized_incomplete_beta(df / 2.0, 0.5, x)
    require(0.0 <= value <= 1.0, "P_VALUE_RANGE")
    return value


def welch_test(
    first: Sequence[float],
    second: Sequence[float],
) -> dict[str, float | int]:
    require(len(first) >= 2, "FIRST_N")
    require(len(second) >= 2, "SECOND_N")
    require(all(math.isfinite(float(x)) for x in first), "FIRST_FINITE")
    require(all(math.isfinite(float(x)) for x in second), "SECOND_FINITE")

    n1 = len(first)
    n2 = len(second)
    mean1 = statistics.fmean(first)
    mean2 = statistics.fmean(second)
    var1 = statistics.variance(first)
    var2 = statistics.variance(second)
    require(var1 >= 0.0 and var2 >= 0.0, "VARIANCE")

    a = var1 / n1
    b = var2 / n2
    standard_error = math.sqrt(a + b)
    require(
        math.isfinite(standard_error) and standard_error > 0.0,
        "STANDARD_ERROR",
    )

    t_statistic = (mean1 - mean2) / standard_error
    numerator = (a + b) ** 2
    denominator = (
        (a * a) / (n1 - 1)
        + (b * b) / (n2 - 1)
    )
    require(denominator > 0.0, "DF_DENOMINATOR")
    df = numerator / denominator
    p_value = student_t_two_sided_p_value(t_statistic, df)

    return {
        "first_n": n1,
        "second_n": n2,
        "first_mean": mean1,
        "second_mean": mean2,
        "first_sample_sd": math.sqrt(var1),
        "second_sample_sd": math.sqrt(var2),
        "mean_difference_first_minus_second": mean1 - mean2,
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "degrees_of_freedom": df,
        "p_value_two_sided": p_value,
    }


def analyze(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups = extract_groups(rows)
    unsupported = groups["unsupported"]
    supported = groups["supported"]

    if (
        len(unsupported) < MIN_GROUP_N
        or len(supported) < MIN_GROUP_N
    ):
        return {
            "estimable": False,
            "minimum_group_n": MIN_GROUP_N,
            "group_counts": {
                "unsupported": len(unsupported),
                "supported": len(supported),
            },
            "primary_test_executed": False,
            "p_value_count_added": 0,
            "scientific_conclusion": CONCLUSION_NOT_ESTIMABLE,
        }

    test = welch_test(unsupported, supported)
    p_value = float(test["p_value_two_sided"])
    significant = p_value < ALPHA
    conclusion = (
        CONCLUSION_OBSERVED
        if significant
        else CONCLUSION_NOT_SUPPORTED
    )

    return {
        "estimable": True,
        "minimum_group_n": MIN_GROUP_N,
        "group_counts": {
            "unsupported": len(unsupported),
            "supported": len(supported),
        },
        "primary_test": {
            "name": PRIMARY_TEST,
            "endpoint": PRIMARY_ENDPOINT,
            "group_order": list(GROUP_ORDER),
            "alternative": "two_sided",
            "equal_variance_assumed": False,
            "alpha": ALPHA,
            **test,
            "significant": significant,
        },
        "primary_test_executed": True,
        "p_value_count_added": P_VALUE_COUNT,
        "scientific_conclusion": conclusion,
    }


def write_output(
    *,
    output_dir: Path,
    expected_head: str,
    raw_summary: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "schema_version": REPORT_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "design": {
            "path": DESIGN.as_posix(),
            "git_blob": DESIGN_GIT_BLOB,
        },
        "raw_stage_a_input": {
            "root": RAW_ROOT.as_posix(),
            "sums_git_blob": RAW_SUMS_GIT_BLOB,
            "summary_git_blob": RAW_SUMMARY_GIT_BLOB,
            "rows_git_blob": RAW_ROWS_GIT_BLOB,
            "summary_sha256": RAW_SUMMARY_SHA256,
            "rows_sha256": RAW_ROWS_SHA256,
            "raw_result": raw_summary["result"],
            "raw_execution_head": raw_summary["execution_head"],
        },
        "analysis_boundary": {
            "model_forward_count": 0,
            "local_vjp_count": 0,
            "training_executed": False,
            "parameter_update_executed": False,
            "new_model_response_generated": False,
            "layer_plane_offset_or_subgroup_selection_performed": False,
            "alternative_aggregation_performed": False,
        },
        "scientific_inference_executed":
            bool(analysis["primary_test_executed"]),
        **dict(analysis),
    }

    raw = pretty_json_bytes(report)
    report_path = output_dir / REPORT_FILE
    report_path.write_bytes(raw)
    report_sha = sha256_bytes(raw)
    (output_dir / SUMS_FILE).write_text(
        f"{report_sha}  {REPORT_FILE}\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def execute(
    *,
    expected_head: str,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    validate_protocol()
    raw_summary, rows = validate_raw_inputs()
    analysis = analyze(rows)
    return write_output(
        output_dir=output_dir,
        expected_head=expected_head,
        raw_summary=raw_summary,
        analysis=analysis,
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single frozen Precursor-v4 Stage A primary inference: "
            "one two-sided Welch independent-samples t-test on item-level z_i, "
            "unsupported versus supported. No model execution occurs."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = execute(
        expected_head=str(args.expected_head),
        output_dir=args.output_dir,
    )
    print("RESULT=" + str(report["result"]))
    print(
        "GROUP_COUNTS="
        + json.dumps(report["group_counts"], sort_keys=True)
    )
    print(
        "PRIMARY_TEST_EXECUTED="
        + str(report["primary_test_executed"])
    )
    print(
        "P_VALUE_COUNT_ADDED="
        + str(report["p_value_count_added"])
    )
    if report["primary_test_executed"]:
        primary = report["primary_test"]
        print("T_STATISTIC=" + repr(primary["t_statistic"]))
        print("DEGREES_OF_FREEDOM=" + repr(primary["degrees_of_freedom"]))
        print("P_VALUE_TWO_SIDED=" + repr(primary["p_value_two_sided"]))
        print("ALPHA=" + repr(primary["alpha"]))
        print("SIGNIFICANT=" + str(primary["significant"]))
    print(
        "SCIENTIFIC_CONCLUSION="
        + str(report["scientific_conclusion"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
