from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STATISTICAL_SPECIFICATION_COMMIT = (
    "4dc5bacd10a254b5ecd339ac1fe78bad9def5c47"
)
R5_ARTIFACT_FREEZE_COMMIT = (
    "a3b5bcf2ded8dc0e86e859bbba12b5601a2fdea0"
)
R6_IMPLEMENTATION_AUTHORITY_COMMIT = (
    "5ae6d7cfd11f9d2b64617a145c7c248becd97aad"
)

CANONICAL_R5_ROWS_SHA256 = (
    "e3157cb5e4e878e4fbe99914a689e576"
    "48568d18b4ae71162e3ba1278093204e"
)
CANONICAL_R5_ROWS_BYTES = 53_510_707

CANONICAL_R5_SUMMARY_SHA256 = (
    "e6887d86c1ac2242ce5c2d74c379912"
    "fd81ac441d71b23a13d0db37412d2905b"
)
CANONICAL_R5_SUMMARY_BYTES = 4_635

PRIMARY_PAIR_COUNT = 300
EVALUATOR_COUNT = 18
CELL_COUNT = 6
ROWS_PER_PAIR = EVALUATOR_COUNT * CELL_COUNT
EXPECTED_ROW_COUNT = PRIMARY_PAIR_COUNT * ROWS_PER_PAIR

FAMILYWISE_ALPHA = 0.05

CELLS = (
    "C0_SHAM",
    "C1_TITLE",
    "C2_NAME",
    "C3_ROLE",
    "C4_PREDICATE",
    "C5_TITLE_NAME",
)

ARMS = (
    "G3-GROUP-D-HALF",
    "G3-GROUP-Q-D-HALF",
    "G3-GROUP-Q-HALF",
    "G3-GROUP-U-D-HALF",
    "G3-GROUP-U-HALF",
    "G3-GROUP-U-Q-HALF",
)

SEEDS = (180, 181, 182)

EXPECTED_EVALUATORS = tuple(
    (seed, arm)
    for seed in SEEDS
    for arm in ARMS
)

ESTIMANDS = (
    "delta_title",
    "delta_name",
    "delta_role",
    "delta_predicate",
    "interaction_title_name",
    "title_minus_name",
)

SECONDARY_METRICS = (
    "entitlement_prob",
    "support_vs_best_nonsupport_logit_margin",
)

DZ_ZERO_VARIANCE_SENTINEL = "UNDEFINED_ZERO_VARIANCE"
POSITIVE_INFINITY_SENTINEL = "POSITIVE_INFINITY"
NEGATIVE_INFINITY_SENTINEL = "NEGATIVE_INFINITY"

PRIMARY_SCHEMA = "gen4_r6_primary_confirmatory_results_v1"
SECONDARY_SCHEMA = "gen4_r6_secondary_descriptive_results_v1"
SUMMARY_SCHEMA = "gen4_r6_statistical_analysis_summary_v1"


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(16 * 1024 * 1024),
            b"",
        ):
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


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} must be numeric")

    result = float(value)

    if not math.isfinite(result):
        raise RuntimeError(f"{field} must be finite")

    return result


# ---------------------------------------------------------------------------
# Pure-Python Student-t distribution
# ---------------------------------------------------------------------------

def _beta_continued_fraction(
    a: float,
    b: float,
    x: float,
) -> float:
    max_iterations = 300
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

    raise RuntimeError("incomplete-beta continued fraction did not converge")


def regularized_incomplete_beta(
    a: float,
    b: float,
    x: float,
) -> float:
    if not (a > 0.0 and b > 0.0):
        raise ValueError("beta parameters must be positive")

    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0, 1]")

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
        result = (
            bt
            * _beta_continued_fraction(a, b, x)
            / a
        )
    else:
        result = 1.0 - (
            bt
            * _beta_continued_fraction(b, a, 1.0 - x)
            / b
        )

    return min(1.0, max(0.0, result))


def student_t_two_sided_p(
    t_statistic: float,
    df: int,
) -> float:
    if df <= 0:
        raise ValueError("df must be positive")

    if math.isnan(t_statistic):
        raise ValueError("t statistic must not be NaN")

    if math.isinf(t_statistic):
        return 0.0

    t_abs = abs(float(t_statistic))

    if t_abs == 0.0:
        return 1.0

    x = df / (df + t_abs * t_abs)

    return regularized_incomplete_beta(
        df / 2.0,
        0.5,
        x,
    )


def student_t_critical(
    df: int,
    confidence: float = 0.95,
) -> float:
    if df <= 0:
        raise ValueError("df must be positive")

    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")

    target_two_sided_p = 1.0 - confidence

    low = 0.0
    high = 1.0

    while (
        student_t_two_sided_p(high, df)
        > target_two_sided_p
    ):
        high *= 2.0

        if high > 1.0e8:
            raise RuntimeError("failed to bracket Student-t critical value")

    for _ in range(120):
        mid = (low + high) / 2.0
        p = student_t_two_sided_p(mid, df)

        if p > target_two_sided_p:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# Matrix validation
# ---------------------------------------------------------------------------

def _evaluator_coordinate(
    row: Mapping[str, Any],
) -> tuple[int, str]:
    return (
        int(row["evaluator_seed"]),
        str(row["evaluator_arm"]),
    )


def validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = PRIMARY_PAIR_COUNT,
    expected_evaluators: Sequence[
        tuple[int, str]
    ] = EXPECTED_EVALUATORS,
) -> None:
    evaluator_set = set(expected_evaluators)

    if len(evaluator_set) != EVALUATOR_COUNT:
        raise RuntimeError("expected evaluator set is not exact-18")

    expected_rows = (
        expected_pair_count
        * CELL_COUNT
        * len(evaluator_set)
    )

    if len(records) != expected_rows:
        raise RuntimeError(
            "matrix row count mismatch: "
            f"expected={expected_rows} "
            f"observed={len(records)}"
        )

    required_fields = {
        "evaluator_seed",
        "evaluator_arm",
        "row_id",
        "source_pair_id",
        "contrast_cell_id",
        "q_authorized",
        "entitlement_prob",
        "support_vs_best_nonsupport_logit_margin",
    }

    keys: set[tuple[int, str, str]] = set()
    observed_evaluators: set[tuple[int, str]] = set()

    by_row_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for row in records:
        missing = required_fields - set(row)

        if missing:
            raise RuntimeError(
                f"required fields missing: {sorted(missing)}"
            )

        evaluator = _evaluator_coordinate(row)

        if evaluator not in evaluator_set:
            raise RuntimeError(
                f"unexpected evaluator coordinate: {evaluator}"
            )

        observed_evaluators.add(evaluator)

        row_id = str(row["row_id"])
        pair_id = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])

        if cell not in CELLS:
            raise RuntimeError(f"unexpected contrast cell: {cell}")

        key = (
            evaluator[0],
            evaluator[1],
            row_id,
        )

        if key in keys:
            raise RuntimeError(
                f"duplicate composite key: {key}"
            )

        keys.add(key)

        _finite_number(
            row["q_authorized"],
            field="q_authorized",
        )

        for metric in SECONDARY_METRICS:
            _finite_number(
                row[metric],
                field=metric,
            )

        by_row_id[row_id].append(row)
        by_pair[pair_id].append(row)

    if observed_evaluators != evaluator_set:
        raise RuntimeError(
            "required evaluator coordinate set is incomplete"
        )

    if len(keys) != expected_rows:
        raise RuntimeError("unique composite-key count mismatch")

    expected_per_row_id = len(evaluator_set)

    for row_id, group in by_row_id.items():
        if len(group) != expected_per_row_id:
            raise RuntimeError(
                f"row_id multiplicity mismatch: {row_id}"
            )

        coords = {
            _evaluator_coordinate(row)
            for row in group
        }

        if coords != evaluator_set:
            raise RuntimeError(
                f"row_id evaluator set mismatch: {row_id}"
            )

        identities = {
            (
                str(row["source_pair_id"]),
                str(row["contrast_cell_id"]),
            )
            for row in group
        }

        if len(identities) != 1:
            raise RuntimeError(
                f"row_id structural identity drift: {row_id}"
            )

    if len(by_pair) != expected_pair_count:
        raise RuntimeError(
            "source-pair count mismatch: "
            f"expected={expected_pair_count} "
            f"observed={len(by_pair)}"
        )

    expected_pair_rows = CELL_COUNT * len(evaluator_set)

    for pair_id, group in by_pair.items():
        if len(group) != expected_pair_rows:
            raise RuntimeError(
                f"source-pair multiplicity mismatch: {pair_id}"
            )

        by_cell: dict[
            str,
            list[Mapping[str, Any]],
        ] = defaultdict(list)

        for row in group:
            by_cell[str(row["contrast_cell_id"])].append(row)

        if set(by_cell) != set(CELLS):
            raise RuntimeError(
                f"incomplete six-cell block: {pair_id}"
            )

        for cell in CELLS:
            cell_rows = by_cell[cell]

            if len(cell_rows) != len(evaluator_set):
                raise RuntimeError(
                    f"cell multiplicity mismatch: {pair_id}/{cell}"
                )

            coords = {
                _evaluator_coordinate(row)
                for row in cell_rows
            }

            if coords != evaluator_set:
                raise RuntimeError(
                    f"cell evaluator set mismatch: {pair_id}/{cell}"
                )

            row_ids = {
                str(row["row_id"])
                for row in cell_rows
            }

            if len(row_ids) != 1:
                raise RuntimeError(
                    f"cell row_id identity mismatch: {pair_id}/{cell}"
                )


def evaluator_averaged_cells(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = PRIMARY_PAIR_COUNT,
) -> dict[str, dict[str, dict[str, float]]]:
    validate_records(
        records,
        expected_pair_count=expected_pair_count,
    )

    grouped: dict[
        tuple[str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for row in records:
        grouped[
            (
                str(row["source_pair_id"]),
                str(row["contrast_cell_id"]),
            )
        ].append(row)

    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    metrics = (
        "q_authorized",
        *SECONDARY_METRICS,
    )

    for (pair_id, cell), rows in grouped.items():
        if len(rows) != EVALUATOR_COUNT:
            raise RuntimeError(
                "evaluator averaging requires exactly 18 rows"
            )

        result[pair_id][cell] = {
            metric: statistics.fmean(
                _finite_number(
                    row[metric],
                    field=metric,
                )
                for row in rows
            )
            for metric in metrics
        }

    return {
        pair_id: dict(cells)
        for pair_id, cells in result.items()
    }


def pair_contrasts(
    averaged: Mapping[
        str,
        Mapping[str, Mapping[str, float]],
    ],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}

    for pair_id in sorted(averaged):
        cells = averaged[pair_id]

        if set(cells) != set(CELLS):
            raise RuntimeError(
                f"incomplete averaged six-cell block: {pair_id}"
            )

        y0 = float(cells["C0_SHAM"]["q_authorized"])
        y1 = float(cells["C1_TITLE"]["q_authorized"])
        y2 = float(cells["C2_NAME"]["q_authorized"])
        y3 = float(cells["C3_ROLE"]["q_authorized"])
        y4 = float(cells["C4_PREDICATE"]["q_authorized"])
        y5 = float(cells["C5_TITLE_NAME"]["q_authorized"])

        delta_title = y1 - y0
        delta_name = y2 - y0

        result[pair_id] = {
            "delta_title": delta_title,
            "delta_name": delta_name,
            "delta_role": y3 - y0,
            "delta_predicate": y4 - y0,
            "interaction_title_name": (
                y5 - y1 - y2 + y0
            ),
            "title_minus_name": (
                delta_title - delta_name
            ),
        }

    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _descriptive_summary(
    values: Sequence[float],
) -> dict[str, float | int]:
    if not values:
        raise ValueError("values must be non-empty")

    finite = [
        _finite_number(value, field="value")
        for value in values
    ]

    n = len(finite)
    mean = statistics.fmean(finite)

    if n == 1:
        sd = 0.0
    else:
        sd = statistics.stdev(finite)

    return {
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "standard_error": (
            sd / math.sqrt(n)
            if n > 0
            else 0.0
        ),
        "median": statistics.median(finite),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def one_sample_t_statistics(
    values: Sequence[float],
    *,
    expected_n: int = PRIMARY_PAIR_COUNT,
) -> dict[str, Any]:
    if len(values) != expected_n:
        raise RuntimeError(
            f"primary N mismatch: expected={expected_n} "
            f"observed={len(values)}"
        )

    summary = _descriptive_summary(values)

    n = int(summary["n"])
    mean = float(summary["mean"])
    sd = float(summary["sample_sd"])
    se = float(summary["standard_error"])
    df = n - 1

    if df != expected_n - 1:
        raise RuntimeError("degrees-of-freedom drift")

    t_critical = student_t_critical(
        df,
        confidence=0.95,
    )

    ci_low = mean - t_critical * se
    ci_high = mean + t_critical * se

    if sd == 0.0:
        if mean == 0.0:
            t_statistic: float | str = 0.0
            raw_p = 1.0
        elif mean > 0.0:
            t_statistic = POSITIVE_INFINITY_SENTINEL
            raw_p = 0.0
        else:
            t_statistic = NEGATIVE_INFINITY_SENTINEL
            raw_p = 0.0

        dz: float | str = DZ_ZERO_VARIANCE_SENTINEL
    else:
        t_numeric = mean / se
        t_statistic = t_numeric
        raw_p = student_t_two_sided_p(
            t_numeric,
            df,
        )
        dz = mean / sd

    return {
        **summary,
        "df": df,
        "t_statistic": t_statistic,
        "raw_p_value": raw_p,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "d_z": dz,
    }


def holm_bonferroni(
    raw_p_values: Sequence[float],
    *,
    alpha: float = FAMILYWISE_ALPHA,
) -> list[dict[str, Any]]:
    if len(raw_p_values) != 6:
        raise RuntimeError(
            "Holm family must contain exactly six p-values"
        )

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    indexed: list[tuple[int, float]] = []

    for index, p_value in enumerate(raw_p_values):
        p = _finite_number(
            p_value,
            field="raw_p_value",
        )

        if p < 0.0 or p > 1.0:
            raise RuntimeError("p-value outside [0, 1]")

        indexed.append((index, p))

    ordered = sorted(
        indexed,
        key=lambda item: (item[1], item[0]),
    )

    m = len(ordered)
    adjusted_by_index: dict[int, float] = {}
    running_adjusted = 0.0

    for rank_zero_based, (original_index, p) in enumerate(ordered):
        multiplier = m - rank_zero_based
        adjusted = min(
            1.0,
            multiplier * p,
        )

        running_adjusted = max(
            running_adjusted,
            adjusted,
        )

        adjusted_by_index[original_index] = running_adjusted

    return [
        {
            "raw_p_value": float(raw_p_values[index]),
            "holm_adjusted_p_value": adjusted_by_index[index],
            "reject_holm_alpha_0_05": (
                adjusted_by_index[index] < alpha
            ),
        }
        for index in range(m)
    ]


def analyze_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = PRIMARY_PAIR_COUNT,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    averaged = evaluator_averaged_cells(
        records,
        expected_pair_count=expected_pair_count,
    )

    contrasts = pair_contrasts(averaged)

    if len(contrasts) != expected_pair_count:
        raise RuntimeError("primary pair-count drift")

    primary: list[dict[str, Any]] = []

    for estimand in ESTIMANDS:
        values = [
            contrasts[pair_id][estimand]
            for pair_id in sorted(contrasts)
        ]

        stats = one_sample_t_statistics(
            values,
            expected_n=expected_pair_count,
        )

        primary.append({
            "schema_version": PRIMARY_SCHEMA,
            "estimand": estimand,
            **stats,
        })

    holm = holm_bonferroni(
        [
            float(row["raw_p_value"])
            for row in primary
        ]
    )

    for row, correction in zip(primary, holm):
        row.update(correction)

    secondary: list[dict[str, Any]] = []

    for metric in SECONDARY_METRICS:
        for cell in CELLS:
            values = [
                float(
                    averaged[pair_id][cell][metric]
                )
                for pair_id in sorted(averaged)
            ]

            summary = _descriptive_summary(values)

            if int(summary["n"]) != expected_pair_count:
                raise RuntimeError(
                    "secondary pair-count drift"
                )

            secondary.append({
                "schema_version": SECONDARY_SCHEMA,
                "metric": metric,
                "contrast_cell_id": cell,
                **summary,
            })

    return primary, secondary


# ---------------------------------------------------------------------------
# Deterministic output
# ---------------------------------------------------------------------------

PRIMARY_FIELDS = (
    "schema_version",
    "estimand",
    "n",
    "mean",
    "sample_sd",
    "standard_error",
    "median",
    "minimum",
    "maximum",
    "df",
    "t_statistic",
    "raw_p_value",
    "holm_adjusted_p_value",
    "reject_holm_alpha_0_05",
    "ci95_low",
    "ci95_high",
    "d_z",
)

SECONDARY_FIELDS = (
    "schema_version",
    "metric",
    "contrast_cell_id",
    "n",
    "mean",
    "sample_sd",
    "standard_error",
    "median",
    "minimum",
    "maximum",
)


def _csv_text(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> str:
    from io import StringIO

    buffer = StringIO(newline="")

    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        lineterminator="\n",
        extrasaction="raise",
    )

    writer.writeheader()

    for row in rows:
        writer.writerow({
            field: row[field]
            for field in fields
        })

    return buffer.getvalue()


def render_report(
    summary: Mapping[str, Any],
    primary: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# ContraMamba Gen4 R6 Statistical Analysis Report - Candidate",
        "",
        "## Provenance",
        "",
        f"- Statistical specification: `{STATISTICAL_SPECIFICATION_COMMIT}`",
        f"- R5 artifact freeze: `{R5_ARTIFACT_FREEZE_COMMIT}`",
        f"- R6 implementation authority: `{R6_IMPLEMENTATION_AUTHORITY_COMMIT}`",
        f"- Input SHA256: `{summary['input_jsonl_sha256']}`",
        f"- Input rows: `{summary['row_count']}`",
        f"- Source pairs: `{summary['pair_count']}`",
        f"- Evaluators: `{summary['evaluator_count']}`",
        "",
        "## Confirmatory family",
        "",
        "| Estimand | Mean | t | raw p | Holm p | Reject | d_z |",
        "|---|---:|---:|---:|---:|:---:|---:|",
    ]

    for row in primary:
        lines.append(
            "| {estimand} | {mean} | {t} | {raw_p} | {holm_p} | "
            "{reject} | {dz} |".format(
                estimand=row["estimand"],
                mean=row["mean"],
                t=row["t_statistic"],
                raw_p=row["raw_p_value"],
                holm_p=row["holm_adjusted_p_value"],
                reject=row["reject_holm_alpha_0_05"],
                dz=row["d_z"],
            )
        )

    lines.extend([
        "",
        "## Scope limitation",
        "",
        "These tests concern only the frozen 300 source-pair population under "
        "the fixed prespecified 18-evaluator population.",
        "",
        "They do not establish arbitrary-model generalization, native-Mamba "
        "state causality, training benefit, or task-performance improvement.",
        "",
    ])

    return "\n".join(lines)


def write_analysis_outputs(
    output_dir: str | Path,
    *,
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    output_dir = Path(output_dir)

    if output_dir.exists():
        raise RuntimeError(
            f"output directory already exists: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    primary_path = (
        output_dir
        / "r6_primary_confirmatory_results.csv"
    )
    secondary_path = (
        output_dir
        / "r6_secondary_descriptive_results.csv"
    )
    summary_path = (
        output_dir
        / "r6_statistical_analysis_summary.json"
    )
    report_path = (
        output_dir
        / "r6_statistical_analysis_report_candidate.md"
    )

    primary_path.write_text(
        _csv_text(primary, PRIMARY_FIELDS),
        encoding="utf-8",
        newline="\n",
    )

    secondary_path.write_text(
        _csv_text(secondary, SECONDARY_FIELDS),
        encoding="utf-8",
        newline="\n",
    )

    summary_path.write_bytes(
        canonical_json_bytes(dict(summary))
    )

    report_path.write_text(
        render_report(
            summary,
            primary,
        ),
        encoding="utf-8",
        newline="\n",
    )

    return {
        path.name: sha256_file(path)
        for path in (
            primary_path,
            secondary_path,
            summary_path,
            report_path,
        )
    }


# ---------------------------------------------------------------------------
# Canonical execution path -- implemented now, but execution requires
# a later frozen R6 execution authority.
# ---------------------------------------------------------------------------

def _validate_exact_file(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> Path:
    path = Path(path)

    if not path.is_file():
        raise RuntimeError(f"required file missing: {path}")

    if path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"byte-count mismatch for {path}"
        )

    observed = sha256_file(path)

    if observed != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: {observed}"
        )

    return path


def validate_r5_summary(
    path: str | Path,
) -> dict[str, Any]:
    path = _validate_exact_file(
        path,
        expected_sha256=CANONICAL_R5_SUMMARY_SHA256,
        expected_bytes=CANONICAL_R5_SUMMARY_BYTES,
    )

    summary = json.loads(
        path.read_text(encoding="utf-8")
    )

    required = {
        "result": "PASS_EXECUTION_MATRIX_PRODUCED",
        "scientific_row_count": EXPECTED_ROW_COUNT,
        "unique_key_count": EXPECTED_ROW_COUNT,
        "complete_matrix_validation": "PASS",
        "statistical_testing": False,
        "output_jsonl_sha256": CANONICAL_R5_ROWS_SHA256,
        "output_jsonl_bytes": CANONICAL_R5_ROWS_BYTES,
    }

    for key, expected in required.items():
        if summary.get(key) != expected:
            raise RuntimeError(
                f"R5 summary mismatch for {key}"
            )

    return summary


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(
                    f"blank JSONL record at line {line_number}"
                )

            value = json.loads(line)

            if not isinstance(value, dict):
                raise RuntimeError(
                    f"non-object JSONL row at line {line_number}"
                )

            rows.append(value)

    return rows


def run_canonical(
    *,
    input_jsonl: str | Path,
    input_summary: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    input_jsonl = _validate_exact_file(
        input_jsonl,
        expected_sha256=CANONICAL_R5_ROWS_SHA256,
        expected_bytes=CANONICAL_R5_ROWS_BYTES,
    )

    validate_r5_summary(input_summary)

    rows = load_jsonl(input_jsonl)

    primary, secondary = analyze_records(
        rows,
        expected_pair_count=PRIMARY_PAIR_COUNT,
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": "PASS_STATISTICAL_ANALYSIS_PRODUCED",
        "statistical_specification_commit":
            STATISTICAL_SPECIFICATION_COMMIT,
        "r5_artifact_freeze_commit":
            R5_ARTIFACT_FREEZE_COMMIT,
        "r6_implementation_authority_commit":
            R6_IMPLEMENTATION_AUTHORITY_COMMIT,
        "input_jsonl_path": str(input_jsonl),
        "input_jsonl_sha256": CANONICAL_R5_ROWS_SHA256,
        "input_jsonl_bytes": CANONICAL_R5_ROWS_BYTES,
        "row_count": len(rows),
        "unique_key_count": len({
            (
                int(row["evaluator_seed"]),
                str(row["evaluator_arm"]),
                str(row["row_id"]),
            )
            for row in rows
        }),
        "pair_count": PRIMARY_PAIR_COUNT,
        "evaluator_count": EVALUATOR_COUNT,
        "confirmatory_hypothesis_count": 6,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "multiplicity_method": "HOLM_BONFERRONI",
        "primary_inferential_unit": "SOURCE_PAIR",
        "primary": list(primary),
        "secondary_descriptive": list(secondary),
        "statistical_testing": True,
        "model_inference": False,
        "training": False,
        "backward": False,
        "scientific_scope": (
            "Frozen 300 source pairs under the fixed prespecified "
            "18-evaluator population only."
        ),
    }

    output_hashes = write_analysis_outputs(
        output_dir,
        primary=primary,
        secondary=secondary,
        summary=summary,
    )

    return {
        **summary,
        "output_sha256": output_hashes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ContraMamba Gen4 R6 statistical-analysis harness. "
            "Canonical execution requires a later frozen execution authority."
        )
    )

    subparsers = parser.add_subparsers(
        dest="operation",
        required=True,
    )

    run = subparsers.add_parser(
        "run-canonical"
    )

    run.add_argument(
        "--input-jsonl",
        required=True,
    )
    run.add_argument(
        "--input-summary",
        required=True,
    )
    run.add_argument(
        "--output-dir",
        required=True,
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.operation == "run-canonical":
        result = run_canonical(
            input_jsonl=args.input_jsonl,
            input_summary=args.input_summary,
            output_dir=args.output_dir,
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )

        return 0

    raise RuntimeError(
        f"unexpected operation: {args.operation}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
