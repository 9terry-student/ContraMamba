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

EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "880eab834c442054642773935a88fa60a31287c3"

DESIGN_ARTIFACT = Path(
    "reports/reason_router_gen4_pre_emission_forced_decisive_stage_a_design.md"
)
DESIGN_ARTIFACT_GIT_BLOB = "b4f9c3ebf2e43a556fbfeb43c5146b4a04f4c620"

RAW_ROOT = Path(
    "reports/reason_router_gen4_pre_emission_forced_decisive_stage_a_generation_runs/"
    "g4k-preemission-forceddecisive-stagea-raw-370m-0d3a094-gpu1"
)
RAW_SUMS = RAW_ROOT / "SHA256SUMS.txt"
RAW_SUMMARY = RAW_ROOT / "execution_summary.json"
RAW_ROWS = RAW_ROOT / "forced_decisive_stage_a_generation_rows.jsonl"

RAW_SUMS_GIT_BLOB = "a7c41fb72d08bc8f5083f45d96bf39aca693bd64"
RAW_SUMMARY_GIT_BLOB = "725e5454a824adc18814d3a20931bb2ab65209f7"
RAW_ROWS_GIT_BLOB = "1691f8e9582f786760df19bee778f247d53b238b"

RAW_SUMMARY_SHA256 = (
    "0dc283213dda376b9684ee34e093de3ad6e09a425ea4fa4884390fa8b14555c5"
)
RAW_ROWS_SHA256 = (
    "5cc15d2626b350108c2c532ab8502d87aacb0d5253f82905ab83dc438aff5fcd"
)
PARTITION_SHA256 = (
    "871fb5c1e2c62f247c284ceae409ef9ec19acc75f63ffc832829c2664db46311"
)
FORCED_GRAMMAR_SHA256 = (
    "62c9c53871f68fcc0f57d38c96b75d2391ee6ddc473a4325a522f91c9249bd00"
)

N = 462
PRIMARY_PARTITION = "confirmatory"
GROUP_ORDER = ("unsupported", "supported")
OFFSETS = (-4, -3, -2, -1)
SIGNAL = "p3_component_l2"
TEST = "two_sided_welch_t"
MULTIPLICITY = "holm"
ALPHA = 0.05
P_VALUE_COUNT = 4

EXPECTED_CONFIRMATORY_GROUP_COUNTS = {
    "supported": 62,
    "unsupported": 169,
}

ANALYSIS_SCHEMA = (
    "gen4-pre-emission-forced-decisive-stage-a-static-analysis-v1"
)
RESULT_SUPPORTED = (
    "FORCED_DECISIVE_PRE_EMISSION_TEMPORAL_PRECEDENCE_OBSERVED"
)
RESULT_NOT_SUPPORTED = (
    "FORCED_DECISIVE_PRE_EMISSION_TEMPORAL_PRECEDENCE_NOT_SUPPORTED"
)

JSON_FILE = "stage_a_analysis.json"
REPORT_FILE = "stage_a_analysis_report.md"
CHECKSUM_FILE = "SHA256SUMS.txt"


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


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REQUIRED_ANCESTOR")

    pinned = {
        DESIGN_ARTIFACT.as_posix(): DESIGN_ARTIFACT_GIT_BLOB,
        RAW_SUMS.as_posix(): RAW_SUMS_GIT_BLOB,
        RAW_SUMMARY.as_posix(): RAW_SUMMARY_GIT_BLOB,
        RAW_ROWS.as_posix(): RAW_ROWS_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_protocol_constants() -> None:
    require(PRIMARY_PARTITION == "confirmatory", "PRIMARY_PARTITION")
    require(GROUP_ORDER == ("unsupported", "supported"), "GROUP_ORDER")
    require(OFFSETS == (-4, -3, -2, -1), "OFFSETS")
    require(SIGNAL == "p3_component_l2", "SIGNAL")
    require(TEST == "two_sided_welch_t", "TEST")
    require(MULTIPLICITY == "holm", "MULTIPLICITY")
    require(P_VALUE_COUNT == 4, "P_VALUE_COUNT")
    require(ALPHA == 0.05, "ALPHA")


def validate_raw_artifact() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for path in (RAW_SUMS, RAW_SUMMARY, RAW_ROWS):
        require((ROOT / path).is_file(), f"RAW_MISSING:{path}")

    require(
        sha256_file(ROOT / RAW_SUMMARY) == RAW_SUMMARY_SHA256,
        "RAW_SUMMARY_SHA256",
    )
    require(
        sha256_file(ROOT / RAW_ROWS) == RAW_ROWS_SHA256,
        "RAW_ROWS_SHA256",
    )

    declared: dict[str, str] = {}
    for line in (ROOT / RAW_SUMS).read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in declared, f"RAW_DUPLICATE_CHECKSUM:{name}")
        declared[name] = digest

    require(
        declared
        == {
            "execution_summary.json": RAW_SUMMARY_SHA256,
            "forced_decisive_stage_a_generation_rows.jsonl":
                RAW_ROWS_SHA256,
        },
        f"RAW_CHECKSUM_SET:{declared}",
    )

    summary = json.loads(
        (ROOT / RAW_SUMMARY).read_text(encoding="utf-8")
    )
    require(
        summary["result"]
        == "PASS_PRE_EMISSION_FORCED_DECISIVE_STAGE_A_RAW_GENERATION",
        "RAW_RESULT",
    )
    require(
        summary["execution_head"]
        == "0d3a094cd6f1a97ce70a19d01d234cf7795c6b1f",
        "RAW_EXECUTION_HEAD",
    )
    require(
        summary["forced_decisive_grammar"]["grammar_sha256"]
        == FORCED_GRAMMAR_SHA256,
        "RAW_GRAMMAR_SHA",
    )
    require(
        summary["partition"]["partition_sha256"]
        == PARTITION_SHA256,
        "RAW_PARTITION_SHA",
    )
    require(
        summary["confirmatory_forced_decisive_group_counts"]
        == EXPECTED_CONFIRMATORY_GROUP_COUNTS,
        "RAW_CONFIRMATORY_GROUP_COUNTS",
    )
    require(
        summary["planned_stage_a_inference_not_executed"]
        == {
            "alpha": 0.05,
            "estimability_rule":
                "both_confirmatory_forced_decisive_groups_must_have_positive_count_before_any_p_value",
            "multiplicity": "holm",
            "planned_p_value_count": 4,
            "primary_comparison":
                "unsupported_forced_decisive_vs_supported_forced_decisive",
            "primary_partition": "confirmatory",
            "primary_signal": "p3_component_l2",
            "relative_offsets": [-4, -3, -2, -1],
            "support_rule":
                "at_least_one_holm_adjusted_p_below_alpha_at_a_preregistered_pre_emission_offset",
            "test": "two_sided_welch_t",
        },
        "RAW_PLANNED_FAMILY",
    )
    require(
        summary["primary_inference_executed"] is False,
        "RAW_INFERENCE_ALREADY_EXECUTED",
    )
    require(
        summary["p_value_count_added"] == 0,
        "RAW_P_VALUE_COUNT",
    )
    require(
        summary["selection_reopened"] is False,
        "RAW_SELECTION_REOPENED",
    )
    require(
        summary["layer_scan_executed"] is False,
        "RAW_LAYER_SCAN",
    )
    require(
        summary["rescue_performed"] is False,
        "RAW_RESCUE",
    )
    require(
        summary["scientific_conclusion"] is None,
        "RAW_CONCLUSION_ALREADY_SET",
    )

    rows = [
        json.loads(line)
        for line in (ROOT / RAW_ROWS).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    require(len(rows) == N, f"RAW_ROW_COUNT:{len(rows)}")
    require(
        len({str(row["example_id"]) for row in rows}) == N,
        "RAW_DUPLICATE_IDS",
    )
    return summary, rows


def collect_primary_values(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, list[float]]]:
    out = {
        offset: {
            "unsupported": [],
            "supported": [],
        }
        for offset in OFFSETS
    }

    confirmatory_count = 0
    group_counts = {
        "unsupported": 0,
        "supported": 0,
    }

    for row in rows:
        if row["partition"] != PRIMARY_PARTITION:
            continue

        confirmatory_count += 1
        group = str(row["primary_stage_a_group"])
        require(group in GROUP_ORDER, f"PRIMARY_GROUP:{group}")
        group_counts[group] += 1

        observations = {
            int(obs["relative_offset"]): obs
            for obs in row["observations"]
        }
        require(
            tuple(sorted(observations)) == OFFSETS,
            f"PRIMARY_OFFSETS:{sorted(observations)}",
        )

        for offset in OFFSETS:
            value = float(observations[offset][SIGNAL])
            require(
                math.isfinite(value),
                f"PRIMARY_NONFINITE:{offset}:{row['example_id']}",
            )
            out[offset][group].append(value)

    require(
        confirmatory_count == 231,
        f"CONFIRMATORY_COUNT:{confirmatory_count}",
    )
    require(
        group_counts == {
            "unsupported": 169,
            "supported": 62,
        },
        f"PRIMARY_GROUP_COUNTS:{group_counts}",
    )

    for offset in OFFSETS:
        require(
            len(out[offset]["unsupported"]) == 169,
            f"OFFSET_UNSUPPORTED_N:{offset}",
        )
        require(
            len(out[offset]["supported"]) == 62,
            f"OFFSET_SUPPORTED_N:{offset}",
        )

    return out


def _beta_continued_fraction(
    a: float,
    b: float,
    x: float,
) -> float:
    max_iterations = 256
    epsilon = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m

        aa = (
            m
            * (b - m)
            * x
            / ((qam + m2) * (a + m2))
        )
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = (
            -(a + m)
            * (qab + m)
            * x
            / ((a + m2) * (qap + m2))
        )
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) <= epsilon:
            return h

    raise StaticAnalysisError("BETA_CONTINUED_FRACTION_DID_NOT_CONVERGE")


def regularized_incomplete_beta(
    x: float,
    a: float,
    b: float,
) -> float:
    require(a > 0.0 and b > 0.0, "BETA_PARAMETERS")
    require(0.0 <= x <= 1.0, "BETA_X")

    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)

    if x < (a + 1.0) / (a + b + 2.0):
        value = (
            bt
            * _beta_continued_fraction(a, b, x)
            / a
        )
    else:
        value = 1.0 - (
            bt
            * _beta_continued_fraction(b, a, 1.0 - x)
            / b
        )

    require(
        math.isfinite(value),
        "BETA_NONFINITE",
    )
    require(
        -1.0e-14 <= value <= 1.0 + 1.0e-14,
        f"BETA_RANGE:{value}",
    )
    return min(1.0, max(0.0, value))


def student_t_two_sided_p(
    t_statistic: float,
    degrees_of_freedom: float,
) -> float:
    require(
        math.isfinite(t_statistic),
        "T_STATISTIC_NONFINITE",
    )
    require(
        math.isfinite(degrees_of_freedom)
        and degrees_of_freedom > 0.0,
        "T_DF",
    )

    if t_statistic == 0.0:
        return 1.0

    x = (
        degrees_of_freedom
        / (
            degrees_of_freedom
            + t_statistic * t_statistic
        )
    )
    value = regularized_incomplete_beta(
        x,
        degrees_of_freedom / 2.0,
        0.5,
    )
    require(
        0.0 <= value <= 1.0,
        f"T_P_RANGE:{value}",
    )
    return value


def welch_t_test(
    unsupported: Sequence[float],
    supported: Sequence[float],
) -> dict[str, float | int | str]:
    x = [float(v) for v in unsupported]
    y = [float(v) for v in supported]

    require(len(x) >= 2, "WELCH_UNSUPPORTED_N")
    require(len(y) >= 2, "WELCH_SUPPORTED_N")
    require(
        all(math.isfinite(v) for v in x + y),
        "WELCH_NONFINITE",
    )

    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    var_x = statistics.variance(x)
    var_y = statistics.variance(y)

    require(
        var_x >= 0.0 and var_y >= 0.0,
        "WELCH_VARIANCE",
    )

    term_x = var_x / len(x)
    term_y = var_y / len(y)
    standard_error_squared = term_x + term_y
    require(
        standard_error_squared > 0.0
        and math.isfinite(standard_error_squared),
        "WELCH_ZERO_OR_NONFINITE_STANDARD_ERROR",
    )

    standard_error = math.sqrt(standard_error_squared)
    difference = mean_x - mean_y
    t_statistic = difference / standard_error

    denominator = (
        (term_x * term_x) / (len(x) - 1)
        + (term_y * term_y) / (len(y) - 1)
    )
    require(
        denominator > 0.0
        and math.isfinite(denominator),
        "WELCH_DF_DENOMINATOR",
    )

    degrees_of_freedom = (
        standard_error_squared
        * standard_error_squared
        / denominator
    )
    p_value = student_t_two_sided_p(
        t_statistic,
        degrees_of_freedom,
    )

    direction = (
        "unsupported_greater"
        if difference > 0.0
        else "unsupported_less"
        if difference < 0.0
        else "equal_means"
    )

    return {
        "n_unsupported": len(x),
        "n_supported": len(y),
        "mean_unsupported": mean_x,
        "mean_supported": mean_y,
        "mean_difference_unsupported_minus_supported":
            difference,
        "variance_unsupported": var_x,
        "variance_supported": var_y,
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "raw_p_value": p_value,
        "direction": direction,
    }


def holm_adjust(
    p_values: Sequence[float],
) -> list[float]:
    values = [float(p) for p in p_values]
    require(
        len(values) == P_VALUE_COUNT,
        f"HOLM_FAMILY_SIZE:{len(values)}",
    )
    require(
        all(
            math.isfinite(p) and 0.0 <= p <= 1.0
            for p in values
        ),
        "HOLM_P_VALUE_RANGE",
    )

    ordered = sorted(
        enumerate(values),
        key=lambda pair: (pair[1], pair[0]),
    )

    adjusted = [0.0] * len(values)
    running_max = 0.0

    for rank, (original_index, p_value) in enumerate(
        ordered,
        start=1,
    ):
        multiplier = len(values) - rank + 1
        candidate = min(1.0, multiplier * p_value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(1.0, running_max)

    return adjusted


def analyze(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = collect_primary_values(rows)

    tests: list[dict[str, Any]] = []
    for offset in OFFSETS:
        result = welch_t_test(
            values[offset]["unsupported"],
            values[offset]["supported"],
        )
        tests.append({
            "relative_offset": offset,
            **result,
        })

    require(
        len(tests) == P_VALUE_COUNT,
        "ANALYSIS_TEST_COUNT",
    )

    adjusted = holm_adjust(
        [float(row["raw_p_value"]) for row in tests]
    )

    for row, adjusted_p in zip(tests, adjusted):
        row["holm_adjusted_p_value"] = adjusted_p
        row["holm_significant_at_alpha_0_05"] = (
            adjusted_p < ALPHA
        )

    significant_offsets = [
        int(row["relative_offset"])
        for row in tests
        if row["holm_significant_at_alpha_0_05"]
    ]

    conclusion = (
        RESULT_SUPPORTED
        if significant_offsets
        else RESULT_NOT_SUPPORTED
    )

    return {
        "schema_version": ANALYSIS_SCHEMA,
        "result": conclusion,
        "raw_evidence_freeze_commit": REQUIRED_ANCESTOR,
        "raw_evidence": {
            "root": RAW_ROOT.as_posix(),
            "summary_sha256": RAW_SUMMARY_SHA256,
            "rows_sha256": RAW_ROWS_SHA256,
            "partition_sha256": PARTITION_SHA256,
            "forced_grammar_sha256": FORCED_GRAMMAR_SHA256,
        },
        "confirmatory_group_counts":
            dict(EXPECTED_CONFIRMATORY_GROUP_COUNTS),
        "primary_family": {
            "partition": PRIMARY_PARTITION,
            "comparison":
                "unsupported_forced_decisive_vs_supported_forced_decisive",
            "orientation":
                "unsupported_minus_supported",
            "signal": SIGNAL,
            "relative_offsets": list(OFFSETS),
            "test": TEST,
            "multiplicity": MULTIPLICITY,
            "p_value_count": P_VALUE_COUNT,
            "alpha": ALPHA,
            "support_rule":
                "at_least_one_holm_adjusted_p_below_alpha_at_a_preregistered_pre_emission_offset",
        },
        "tests": tests,
        "significant_offsets": significant_offsets,
        "primary_inference_executed": True,
        "p_value_count_added": P_VALUE_COUNT,
        "training_executed": False,
        "backward_executed": False,
        "selection_reopened": False,
        "layer_scan_executed": False,
        "rescue_performed": False,
        "stage_b_executed": False,
        "stage_c_executed": False,
        "claim_scope":
            "forced_decisive_two_class_finite_grammar_only",
    }


def render_report(
    analysis: Mapping[str, Any],
) -> str:
    lines = [
        "# ContraMamba Gen4 — Forced-Decisive Stage A Static Analysis",
        "",
        "## Status",
        "",
        "This report applies exactly the preregistered confirmatory Stage A family to the frozen forced-decisive raw artifact.",
        "",
        f"Result: `{analysis['result']}`",
        "",
        "Scope: forced-decisive two-class finite grammar only.",
        "",
        "## Frozen input",
        "",
        f"- Raw evidence freeze commit: `{REQUIRED_ANCESTOR}`",
        f"- Rows SHA256: `{RAW_ROWS_SHA256}`",
        f"- Summary SHA256: `{RAW_SUMMARY_SHA256}`",
        f"- Partition SHA256: `{PARTITION_SHA256}`",
        f"- Forced grammar SHA256: `{FORCED_GRAMMAR_SHA256}`",
        "- Confirmatory unsupported: 169",
        "- Confirmatory supported: 62",
        "",
        "## Preregistered primary family",
        "",
        "- Signal: `p3_component_l2`",
        "- Comparison orientation: unsupported minus supported",
        "- Offsets: `t*-4`, `t*-3`, `t*-2`, `t*-1`",
        "- Test: two-sided Welch t-test",
        "- Multiplicity: Holm across exactly 4 p-values",
        "- Familywise alpha: 0.05",
        "",
        "## Results",
        "",
        "| Offset | N unsupported | N supported | Mean unsupported | Mean supported | Difference U-S | Welch t | df | Raw p | Holm p | Significant |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    for row in analysis["tests"]:
        lines.append(
            "| {offset} | {nu} | {ns} | {mu:.12g} | {ms:.12g} | {diff:.12g} | {t:.12g} | {df:.12g} | {p:.12g} | {hp:.12g} | {sig} |".format(
                offset=row["relative_offset"],
                nu=row["n_unsupported"],
                ns=row["n_supported"],
                mu=row["mean_unsupported"],
                ms=row["mean_supported"],
                diff=row[
                    "mean_difference_unsupported_minus_supported"
                ],
                t=row["t_statistic"],
                df=row["degrees_of_freedom"],
                p=row["raw_p_value"],
                hp=row["holm_adjusted_p_value"],
                sig=(
                    "YES"
                    if row["holm_significant_at_alpha_0_05"]
                    else "NO"
                ),
            )
        )

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
    ])

    if analysis["result"] == RESULT_SUPPORTED:
        lines.extend([
            "At least one preregistered pre-emission offset passed Holm-adjusted `p < 0.05`.",
            "",
            "The bounded Stage A label is:",
            "",
            f"`{RESULT_SUPPORTED}`",
            "",
            "This supports temporal separation under the prospectively frozen forced-decisive two-class grammar only.",
        ])
    else:
        lines.extend([
            "No preregistered pre-emission offset passed Holm-adjusted `p < 0.05`.",
            "",
            "The bounded Stage A label is:",
            "",
            f"`{RESULT_NOT_SUPPORTED}`",
            "",
            "This does not support temporal separation under the prospectively frozen forced-decisive two-class grammar.",
        ])

    lines.extend([
        "",
        "This report does not establish spontaneous hallucination prediction, behavior with abstention available, free-form generation generalization, prospective Stage B prediction, or causal Stage C prevention.",
        "",
        "No training, backward pass, selection reopening, layer scan, rescue, Stage B, or Stage C was executed.",
        "",
    ])
    return "\n".join(lines)


def write_output(
    output_dir: Path,
    analysis: Mapping[str, Any],
) -> None:
    require(
        not output_dir.exists(),
        f"OUTPUT_COLLISION:{output_dir}",
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    json_raw = pretty_json_bytes(dict(analysis))
    report_raw = render_report(analysis).encode("utf-8")

    (output_dir / JSON_FILE).write_bytes(json_raw)
    (output_dir / REPORT_FILE).write_bytes(report_raw)

    hashes = {
        JSON_FILE: sha256_bytes(json_raw),
        REPORT_FILE: sha256_bytes(report_raw),
    }
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered CPU/static confirmatory Stage A "
            "Welch×4 + Holm analysis on the frozen forced-decisive raw artifact."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    validate_protocol_constants()
    authenticate_repo(str(args.expected_head))
    _summary, rows = validate_raw_artifact()

    analysis = analyze(rows)
    write_output(args.output_dir, analysis)

    print("RESULT=" + str(analysis["result"]))
    print("PRIMARY_INFERENCE_EXECUTED=True")
    print("P_VALUE_COUNT_ADDED=4")
    print(
        "SIGNIFICANT_OFFSETS="
        + json.dumps(analysis["significant_offsets"])
    )
    for row in analysis["tests"]:
        print(
            "OFFSET={offset} T={t:.17g} DF={df:.17g} "
            "RAW_P={p:.17g} HOLM_P={hp:.17g} "
            "MEAN_U={mu:.17g} MEAN_S={ms:.17g}".format(
                offset=row["relative_offset"],
                t=row["t_statistic"],
                df=row["degrees_of_freedom"],
                p=row["raw_p_value"],
                hp=row["holm_adjusted_p_value"],
                mu=row["mean_unsupported"],
                ms=row["mean_supported"],
            )
        )
    print("TRAINING_EXECUTED=False")
    print("STAGE_B_EXECUTED=False")
    print("STAGE_C_EXECUTED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
