from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence

SCIENTIFIC_SPECIFICATION_COMMIT = "b3e0ade126622f244b557e1db07296c622bd7202"
IMPLEMENTATION_AUTHORITY_COMMIT = "724c28b528b0f182bc0c79cf5ee0b3adfca76ec2"
REFERENCE_Q1_Q3_IMPLEMENTATION_SHA256 = (
    "dc50ac3552182067d5dc76e4f3cecf2f674b24d045fa08352afa84c640266cf4"
)

CANONICAL_PHASE_F_PAIR_PATH = (
    "reports/reason_router_gen4_native_mamba_state_phase_f_"
    "statistical_analysis_e917e4c_v1/phase_f_pair_level_contrasts.csv"
)
CANONICAL_PHASE_F_PAIR_SHA256 = (
    "abb7e837395138d095e285e40ec8782ea863061087b58844bec6fad5d4ef5e73"
)
CANONICAL_PHASE_F_PAIR_BYTES = 542_251
CANONICAL_PHASE_F_TOTAL_ROWS = 4_500
CANONICAL_PHASE_F_DELTA_NAME_ROWS = 900

CANONICAL_Q1_Q3_PAIR_PATH = (
    "reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_"
    "statistical_analysis_retry1_v1/name_q1_q3_pair_level_contrasts.csv"
)
CANONICAL_Q1_Q3_PAIR_SHA256 = (
    "c0c917560b5a37c5df82ad4e203a87441370c129622afc668b77712b98698f82"
)
CANONICAL_Q1_Q3_PAIR_BYTES = 196_542
CANONICAL_Q1_Q3_TOTAL_ROWS = 1_800

SOURCE_PAIR_COUNT = 300
MIDPOINT_LAYER_INDEX = 11
SECONDARY_LAYERS = (5, 17)
LAYER_CONTRASTS = (("5_MINUS_11", 5), ("17_MINUS_11", 17))
ENDPOINTS = (
    "POST4_SPEED",
    "POST4_TURNING",
    "POST4_PATH_EFFICIENCY",
)
ESTIMAND = "DELTA_NAME"
PHASE_F_ESTIMANDS = (
    "DELTA_TITLE",
    "DELTA_NAME",
    "DELTA_ROLE",
    "DELTA_PREDICATE",
    "INTERACTION_TITLE_NAME",
)
HYPOTHESIS_COUNT = 6
FAMILYWISE_ALPHA = 0.05

PHASE_F_INPUT_SCHEMA = "gen4_native_mamba_phase_f_pair_level_contrasts_v1"
Q1_Q3_INPUT_SCHEMA = "gen4_name_q1_q3_pair_level_contrasts_v1"
PAIR_LEVEL_SCHEMA = "gen4_name_direct_cross_layer_pair_level_differences_v1"
CONFIRMATORY_SCHEMA = "gen4_name_direct_cross_layer_confirmatory_results_v1"
MANIFEST_SCHEMA = "gen4_name_direct_cross_layer_statistical_analysis_manifest_v1"

SUPPORTED = "PRESPECIFIED_DIRECT_CROSS_LAYER_DIFFERENCE_SUPPORTED"
NOT_ESTABLISHED = "PRESPECIFIED_DIRECT_CROSS_LAYER_DIFFERENCE_NOT_ESTABLISHED"
FAMILY_SUPPORTED = (
    "DIRECT_CROSS_LAYER_NAME_DIFFERENCE_SUPPORTED_FOR_AT_LEAST_ONE_"
    "PRESPECIFIED_LAYER_PAIR_ENDPOINT"
)
FAMILY_NOT_ESTABLISHED = (
    "DIRECT_CROSS_LAYER_NAME_DIFFERENCE_NOT_ESTABLISHED_WITHIN_THE_"
    "PRESPECIFIED_FAMILY"
)

PHASE_F_FIELDS = (
    "schema_version",
    "source_pair_id",
    "endpoint",
    "estimand",
    "contrast_value",
)
Q1_Q3_FIELDS = (
    "schema_version",
    "source_pair_id",
    "layer_index",
    "endpoint",
    "estimand",
    "contrast_value",
)
PAIR_LEVEL_FIELDS = (
    "schema_version",
    "source_pair_id",
    "layer_contrast",
    "secondary_layer_index",
    "midpoint_layer_index",
    "endpoint",
    "estimand",
    "secondary_contrast_value",
    "midpoint_contrast_value",
    "cross_layer_difference",
)
CONFIRMATORY_FIELDS = (
    "schema_version",
    "layer_contrast",
    "secondary_layer_index",
    "midpoint_layer_index",
    "endpoint",
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
    "decision",
)
OUTPUT_FILENAMES = (
    "name_direct_cross_layer_pair_level_differences.csv",
    "name_direct_cross_layer_confirmatory_results.csv",
    "name_direct_cross_layer_statistical_analysis_manifest.json",
    "name_direct_cross_layer_statistical_analysis_report_candidate.md",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
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


def csv_bytes(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue().encode("utf-8")


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be numeric and non-boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} must be numeric and non-boolean") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{field} must be finite")
    return result


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
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
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
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


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not (a > 0.0 and b > 0.0):
        raise ValueError("beta parameters must be positive")
    if not (0.0 <= x <= 1.0):
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
        result = bt * _beta_continued_fraction(a, b, x) / a
    else:
        result = 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, result))


def student_t_two_sided_p(t_statistic: float, df: int) -> float:
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
    return regularized_incomplete_beta(df / 2.0, 0.5, x)


def student_t_critical(df: int, confidence: float = 0.95) -> float:
    if df <= 0:
        raise ValueError("df must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    target_p = 1.0 - confidence
    low = 0.0
    high = 1.0
    while student_t_two_sided_p(high, df) > target_p:
        high *= 2.0
        if high > 1.0e8:
            raise RuntimeError("failed to bracket Student-t critical value")
    for _ in range(120):
        mid = (low + high) / 2.0
        if student_t_two_sided_p(mid, df) > target_p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def descriptive_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("values must be non-empty")
    finite = [_finite_number(value, field="value") for value in values]
    n = len(finite)
    mean = statistics.fmean(finite)
    sd = statistics.stdev(finite) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "standard_error": sd / math.sqrt(n),
        "median": statistics.median(finite),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def one_sample_t_statistics(
    values: Sequence[float],
    *,
    expected_n: int = SOURCE_PAIR_COUNT,
) -> dict[str, float | int]:
    if len(values) != expected_n:
        raise RuntimeError(
            f"cross-layer N mismatch: expected={expected_n} observed={len(values)}"
        )
    summary = descriptive_summary(values)
    n = int(summary["n"])
    mean = float(summary["mean"])
    sd = float(summary["sample_sd"])
    se = float(summary["standard_error"])
    df = n - 1
    if df != expected_n - 1:
        raise RuntimeError("degrees-of-freedom drift")
    if sd == 0.0 or se == 0.0:
        raise RuntimeError("zero-variance cross-layer difference distribution")
    t_statistic = mean / se
    raw_p = student_t_two_sided_p(t_statistic, df)
    t_critical = student_t_critical(df, confidence=0.95)
    return {
        **summary,
        "df": df,
        "t_statistic": t_statistic,
        "raw_p_value": raw_p,
        "ci95_low": mean - t_critical * se,
        "ci95_high": mean + t_critical * se,
        "d_z": mean / sd,
    }


def holm_bonferroni(
    raw_p_values: Sequence[float],
    *,
    alpha: float = FAMILYWISE_ALPHA,
) -> list[dict[str, Any]]:
    if len(raw_p_values) != HYPOTHESIS_COUNT:
        raise RuntimeError(
            f"Holm family must contain exactly {HYPOTHESIS_COUNT} p-values"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    indexed: list[tuple[int, float]] = []
    for index, value in enumerate(raw_p_values):
        p = _finite_number(value, field="raw_p_value")
        if not (0.0 <= p <= 1.0):
            raise RuntimeError("p-value outside [0, 1]")
        indexed.append((index, p))
    ordered = sorted(indexed, key=lambda item: (item[1], item[0]))
    m = len(ordered)
    adjusted_by_index: dict[int, float] = {}
    running = 0.0
    for rank, (original_index, p) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        adjusted_by_index[original_index] = running
    return [
        {
            "raw_p_value": float(raw_p_values[index]),
            "holm_adjusted_p_value": adjusted_by_index[index],
            "reject_holm_alpha_0_05": adjusted_by_index[index] < alpha,
        }
        for index in range(m)
    ]


def _require_exact_fields(row: Mapping[str, Any], fields: Sequence[str], *, label: str) -> None:
    observed = tuple(row.keys())
    if observed != tuple(fields):
        raise RuntimeError(
            f"{label} field contract mismatch: expected={list(fields)} observed={list(observed)}"
        )


def validate_phase_f_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = SOURCE_PAIR_COUNT,
) -> tuple[dict[tuple[str, str], float], list[str]]:
    expected_rows = expected_pair_count * len(ENDPOINTS) * len(PHASE_F_ESTIMANDS)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Phase F row count mismatch: expected={expected_rows} observed={len(rows)}"
        )
    by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    lookup: dict[tuple[str, str, str], float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"non-object Phase F row: {index}")
        _require_exact_fields(row, PHASE_F_FIELDS, label="Phase F")
        if str(row["schema_version"]) != PHASE_F_INPUT_SCHEMA:
            raise RuntimeError("unexpected Phase F schema_version")
        pair_id = str(row["source_pair_id"])
        endpoint = str(row["endpoint"])
        estimand = str(row["estimand"])
        if not pair_id:
            raise RuntimeError("empty Phase F source_pair_id")
        if endpoint not in ENDPOINTS:
            raise RuntimeError(f"unexpected Phase F endpoint: {endpoint}")
        if estimand not in PHASE_F_ESTIMANDS:
            raise RuntimeError(f"unexpected Phase F estimand: {estimand}")
        value = _finite_number(row["contrast_value"], field="Phase F contrast_value")
        key = (pair_id, endpoint, estimand)
        if key in lookup:
            raise RuntimeError(f"duplicate Phase F structural key: {key}")
        lookup[key] = value
        by_cell[(endpoint, estimand)].append(pair_id)

    expected_cells = {(endpoint, estimand) for endpoint in ENDPOINTS for estimand in PHASE_F_ESTIMANDS}
    if set(by_cell) != expected_cells:
        raise RuntimeError("Phase F structural cell set mismatch")
    reference_ids: list[str] | None = None
    for endpoint in ENDPOINTS:
        for estimand in PHASE_F_ESTIMANDS:
            ids = by_cell[(endpoint, estimand)]
            if len(ids) != expected_pair_count or len(set(ids)) != expected_pair_count:
                raise RuntimeError(f"Phase F pair population mismatch: {endpoint}/{estimand}")
            if ids != sorted(ids):
                raise RuntimeError(f"Phase F pair order not lexicographic: {endpoint}/{estimand}")
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise RuntimeError(f"Phase F source-pair alignment mismatch: {endpoint}/{estimand}")
    assert reference_ids is not None
    delta_lookup = {
        (pair_id, endpoint): lookup[(pair_id, endpoint, ESTIMAND)]
        for pair_id in reference_ids
        for endpoint in ENDPOINTS
    }
    if len(delta_lookup) != expected_pair_count * len(ENDPOINTS):
        raise RuntimeError("Phase F DELTA_NAME row cardinality mismatch")
    return delta_lookup, list(reference_ids)


def validate_q1_q3_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = SOURCE_PAIR_COUNT,
) -> tuple[dict[tuple[int, str, str], float], list[str]]:
    expected_rows = expected_pair_count * len(SECONDARY_LAYERS) * len(ENDPOINTS)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Q1/Q3 row count mismatch: expected={expected_rows} observed={len(rows)}"
        )
    by_cell: dict[tuple[int, str], list[str]] = defaultdict(list)
    lookup: dict[tuple[int, str, str], float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"non-object Q1/Q3 row: {index}")
        _require_exact_fields(row, Q1_Q3_FIELDS, label="Q1/Q3")
        if str(row["schema_version"]) != Q1_Q3_INPUT_SCHEMA:
            raise RuntimeError("unexpected Q1/Q3 schema_version")
        pair_id = str(row["source_pair_id"])
        if not pair_id:
            raise RuntimeError("empty Q1/Q3 source_pair_id")
        try:
            layer = int(row["layer_index"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("unexpected Q1/Q3 layer_index") from exc
        endpoint = str(row["endpoint"])
        estimand = str(row["estimand"])
        if layer not in SECONDARY_LAYERS:
            raise RuntimeError(f"unexpected Q1/Q3 layer_index: {layer}")
        if endpoint not in ENDPOINTS:
            raise RuntimeError(f"unexpected Q1/Q3 endpoint: {endpoint}")
        if estimand != ESTIMAND:
            raise RuntimeError(f"unexpected Q1/Q3 estimand: {estimand}")
        value = _finite_number(row["contrast_value"], field="Q1/Q3 contrast_value")
        key = (layer, pair_id, endpoint)
        if key in lookup:
            raise RuntimeError(f"duplicate Q1/Q3 structural key: {key}")
        lookup[key] = value
        by_cell[(layer, endpoint)].append(pair_id)

    expected_cells = {(layer, endpoint) for layer in SECONDARY_LAYERS for endpoint in ENDPOINTS}
    if set(by_cell) != expected_cells:
        raise RuntimeError("Q1/Q3 structural cell set mismatch")
    reference_ids: list[str] | None = None
    for layer in SECONDARY_LAYERS:
        for endpoint in ENDPOINTS:
            ids = by_cell[(layer, endpoint)]
            if len(ids) != expected_pair_count or len(set(ids)) != expected_pair_count:
                raise RuntimeError(f"Q1/Q3 pair population mismatch: {layer}/{endpoint}")
            if ids != sorted(ids):
                raise RuntimeError(f"Q1/Q3 pair order not lexicographic: {layer}/{endpoint}")
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise RuntimeError(f"Q1/Q3 source-pair alignment mismatch: {layer}/{endpoint}")
    assert reference_ids is not None
    return lookup, list(reference_ids)


def construct_pair_level_differences(
    phase_f_rows: Sequence[Mapping[str, Any]],
    q1_q3_rows: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = SOURCE_PAIR_COUNT,
) -> list[dict[str, Any]]:
    midpoint_lookup, midpoint_ids = validate_phase_f_rows(
        phase_f_rows, expected_pair_count=expected_pair_count
    )
    secondary_lookup, secondary_ids = validate_q1_q3_rows(
        q1_q3_rows, expected_pair_count=expected_pair_count
    )
    if midpoint_ids != secondary_ids:
        raise RuntimeError("cross-input source-pair population mismatch")

    output: list[dict[str, Any]] = []
    for layer_contrast, secondary_layer in LAYER_CONTRASTS:
        for endpoint in ENDPOINTS:
            for pair_id in midpoint_ids:
                secondary = secondary_lookup[(secondary_layer, pair_id, endpoint)]
                midpoint = midpoint_lookup[(pair_id, endpoint)]
                output.append(
                    {
                        "schema_version": PAIR_LEVEL_SCHEMA,
                        "source_pair_id": pair_id,
                        "layer_contrast": layer_contrast,
                        "secondary_layer_index": secondary_layer,
                        "midpoint_layer_index": MIDPOINT_LAYER_INDEX,
                        "endpoint": endpoint,
                        "estimand": ESTIMAND,
                        "secondary_contrast_value": secondary,
                        "midpoint_contrast_value": midpoint,
                        "cross_layer_difference": float(secondary - midpoint),
                    }
                )
    expected_rows = expected_pair_count * HYPOTHESIS_COUNT
    if len(output) != expected_rows:
        raise RuntimeError("pair-level cross-layer output cardinality mismatch")
    return output


def family_decision(confirmatory: Sequence[Mapping[str, Any]]) -> str:
    if len(confirmatory) != HYPOTHESIS_COUNT:
        raise RuntimeError("confirmatory result row count drift")
    return (
        FAMILY_SUPPORTED
        if any(bool(row["reject_holm_alpha_0_05"]) for row in confirmatory)
        else FAMILY_NOT_ESTABLISHED
    )


def analyze_rows(
    phase_f_rows: Sequence[Mapping[str, Any]],
    q1_q3_rows: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = SOURCE_PAIR_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows = construct_pair_level_differences(
        phase_f_rows,
        q1_q3_rows,
        expected_pair_count=expected_pair_count,
    )
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row["layer_contrast"]), str(row["endpoint"]))].append(
            float(row["cross_layer_difference"])
        )

    confirmatory: list[dict[str, Any]] = []
    for layer_contrast, secondary_layer in LAYER_CONTRASTS:
        for endpoint in ENDPOINTS:
            stats = one_sample_t_statistics(
                grouped[(layer_contrast, endpoint)],
                expected_n=expected_pair_count,
            )
            confirmatory.append(
                {
                    "schema_version": CONFIRMATORY_SCHEMA,
                    "layer_contrast": layer_contrast,
                    "secondary_layer_index": secondary_layer,
                    "midpoint_layer_index": MIDPOINT_LAYER_INDEX,
                    "endpoint": endpoint,
                    "estimand": ESTIMAND,
                    **stats,
                }
            )

    correction = holm_bonferroni(
        [float(row["raw_p_value"]) for row in confirmatory]
    )
    for row, adjusted in zip(confirmatory, correction):
        row.update(adjusted)
        row["decision"] = (
            SUPPORTED
            if bool(row["reject_holm_alpha_0_05"])
            else NOT_ESTABLISHED
        )
    if len(confirmatory) != HYPOTHESIS_COUNT:
        raise RuntimeError("confirmatory result row count drift")
    return pair_rows, confirmatory


def render_report(
    *,
    implementation_commit: str,
    script_sha256: str,
    execution_authority_commit: str,
    confirmatory: Sequence[Mapping[str, Any]],
) -> bytes:
    family = family_decision(confirmatory)
    lines = [
        "# ContraMamba Gen4 Native Mamba State Bridge",
        "# NAME Direct Cross-Layer Paired-Difference Statistical Analysis Report - Candidate",
        "",
        "## Provenance",
        "",
        f"- Scientific specification: `{SCIENTIFIC_SPECIFICATION_COMMIT}`",
        f"- Implementation authority: `{IMPLEMENTATION_AUTHORITY_COMMIT}`",
        f"- Statistical execution authority: `{execution_authority_commit}`",
        f"- Implementation commit: `{implementation_commit}`",
        f"- Script SHA256: `{script_sha256}`",
        f"- Phase F input: `{CANONICAL_PHASE_F_PAIR_PATH}`",
        f"- Phase F input SHA256: `{CANONICAL_PHASE_F_PAIR_SHA256}`",
        f"- Q1/Q3 input: `{CANONICAL_Q1_Q3_PAIR_PATH}`",
        f"- Q1/Q3 input SHA256: `{CANONICAL_Q1_Q3_PAIR_SHA256}`",
        f"- Source pairs: `{SOURCE_PAIR_COUNT}`",
        "",
        "## Direct cross-layer confirmatory family",
        "",
        "| Layer contrast | Endpoint | Estimand | Mean | t | raw p | Holm p | Reject | d_z |",
        "|---|---|---|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in confirmatory:
        lines.append(
            "| {contrast} | {endpoint} | {estimand} | {mean} | {t} | {raw_p} | "
            "{holm_p} | {reject} | {dz} |".format(
                contrast=row["layer_contrast"],
                endpoint=row["endpoint"],
                estimand=row["estimand"],
                mean=row["mean"],
                t=row["t_statistic"],
                raw_p=row["raw_p_value"],
                holm_p=row["holm_adjusted_p_value"],
                reject=row["reject_holm_alpha_0_05"],
                dz=row["d_z"],
            )
        )
    lines.extend(
        [
            "",
            "## Family decision",
            "",
            family,
            "",
            "## Inference boundary",
            "",
            "The Holm correction controls only this frozen six-member adaptive",
            "direct cross-layer family. Overall adaptive-program FWER across Phase F,",
            "the Q1/Q3 follow-up, and this phase is not claimed.",
            "",
            "A supported member establishes only the exact prespecified between-layer",
            "difference for that layer pair and NAME kinematic endpoint.",
            "",
            "The broad unqualified claim that NAME is depth-selective is prohibited.",
            "No causal mediation, necessity, sufficiency, or state-to-output causation",
            "claim follows from this analysis.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def build_output_bytes(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    confirmatory: Sequence[Mapping[str, Any]],
    implementation_commit: str,
    script_sha256: str,
    execution_authority_commit: str,
) -> dict[str, bytes]:
    if len(pair_rows) != SOURCE_PAIR_COUNT * HYPOTHESIS_COUNT:
        raise RuntimeError("pair-level output cardinality mismatch")
    if len(confirmatory) != HYPOTHESIS_COUNT:
        raise RuntimeError("confirmatory output cardinality mismatch")

    pair_data = csv_bytes(pair_rows, PAIR_LEVEL_FIELDS)
    confirmatory_data = csv_bytes(confirmatory, CONFIRMATORY_FIELDS)
    report_data = render_report(
        implementation_commit=implementation_commit,
        script_sha256=script_sha256,
        execution_authority_commit=execution_authority_commit,
        confirmatory=confirmatory,
    )
    peer_sha256 = {
        "name_direct_cross_layer_pair_level_differences.csv": sha256_bytes(pair_data),
        "name_direct_cross_layer_confirmatory_results.csv": sha256_bytes(confirmatory_data),
        "name_direct_cross_layer_statistical_analysis_report_candidate.md": sha256_bytes(report_data),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": "PASS_STATISTICAL_ANALYSIS_PRODUCED",
        "scientific_specification_commit": SCIENTIFIC_SPECIFICATION_COMMIT,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "statistical_execution_authority_commit": execution_authority_commit,
        "implementation_commit": implementation_commit,
        "script_sha256": script_sha256,
        "phase_f_input_path": CANONICAL_PHASE_F_PAIR_PATH,
        "phase_f_input_sha256": CANONICAL_PHASE_F_PAIR_SHA256,
        "phase_f_input_bytes": CANONICAL_PHASE_F_PAIR_BYTES,
        "phase_f_input_rows": CANONICAL_PHASE_F_TOTAL_ROWS,
        "q1_q3_input_path": CANONICAL_Q1_Q3_PAIR_PATH,
        "q1_q3_input_sha256": CANONICAL_Q1_Q3_PAIR_SHA256,
        "q1_q3_input_bytes": CANONICAL_Q1_Q3_PAIR_BYTES,
        "q1_q3_input_rows": CANONICAL_Q1_Q3_TOTAL_ROWS,
        "source_pair_count": SOURCE_PAIR_COUNT,
        "layer_contrast_order": [name for name, _ in LAYER_CONTRASTS],
        "secondary_layer_order": list(SECONDARY_LAYERS),
        "midpoint_layer_index": MIDPOINT_LAYER_INDEX,
        "endpoint_order": list(ENDPOINTS),
        "structural_estimand": ESTIMAND,
        "hypothesis_count": HYPOTHESIS_COUNT,
        "n_per_hypothesis": SOURCE_PAIR_COUNT,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "multiplicity_method": "HOLM_BONFERRONI",
        "numeric_analysis_dtype": "FLOAT64",
        "primary_inferential_unit": "SOURCE_PAIR",
        "family_decision": family_decision(confirmatory),
        "overall_adaptive_program_fwer_claimed": False,
        "direct_cross_layer_difference_test": True,
        "peer_artifact_sha256": peer_sha256,
        "statistical_testing": True,
        "model_inference": False,
        "training": False,
        "backward": False,
        "scientific_scope": (
            "Frozen 300 source pairs, DELTA_NAME, direct 5_MINUS_11 and 17_MINUS_11 "
            "comparisons, and three prespecified local native-state kinematic endpoints only."
        ),
    }
    if "manifest_sha256" in manifest:
        raise RuntimeError("manifest self SHA256 is forbidden")
    return {
        "name_direct_cross_layer_pair_level_differences.csv": pair_data,
        "name_direct_cross_layer_confirmatory_results.csv": confirmatory_data,
        "name_direct_cross_layer_statistical_analysis_manifest.json": canonical_json_bytes(manifest),
        "name_direct_cross_layer_statistical_analysis_report_candidate.md": report_data,
    }


def write_output_bytes(
    output_dir: str | Path,
    outputs: Mapping[str, bytes],
) -> dict[str, str]:
    if tuple(outputs) != OUTPUT_FILENAMES:
        raise RuntimeError("output filename/order contract mismatch")
    final_dir = Path(output_dir)
    staging_dir = final_dir.with_name(final_dir.name + ".staging")
    if final_dir.exists():
        raise RuntimeError(f"output directory already exists: {final_dir}")
    if staging_dir.exists():
        raise RuntimeError(f"output staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        result: dict[str, str] = {}
        for name in OUTPUT_FILENAMES:
            data = outputs[name]
            path = staging_dir / name
            path.write_bytes(data)
            result[name] = sha256_bytes(data)
        staging_dir.rename(final_dir)
        return result
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


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
        raise RuntimeError(f"byte-count mismatch for {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA256 mismatch for {path}: {observed}")
    return path


def _validate_requested_logical_path(path: str | Path, expected: str) -> None:
    observed = str(path).replace("\\", "/")
    if observed != expected:
        raise RuntimeError(
            f"canonical logical path mismatch: expected={expected} observed={observed}"
        )


def load_pair_csv(path: str | Path, *, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("pair-level CSV must be UTF-8") from exc
    reader = csv.DictReader(StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != tuple(expected_fields):
        raise RuntimeError("CSV header field contract mismatch")
    rows = list(reader)
    for row in rows:
        if None in row:
            raise RuntimeError("CSV row has unexpected extra fields")
    return rows


def run_canonical(
    *,
    phase_f_pair_csv: str | Path,
    q1_q3_pair_csv: str | Path,
    output_dir: str | Path,
    implementation_commit: str,
    script_sha256: str,
    execution_authority_commit: str,
) -> dict[str, Any]:
    _validate_requested_logical_path(phase_f_pair_csv, CANONICAL_PHASE_F_PAIR_PATH)
    _validate_requested_logical_path(q1_q3_pair_csv, CANONICAL_Q1_Q3_PAIR_PATH)
    phase_path = _validate_exact_file(
        phase_f_pair_csv,
        expected_sha256=CANONICAL_PHASE_F_PAIR_SHA256,
        expected_bytes=CANONICAL_PHASE_F_PAIR_BYTES,
    )
    q_path = _validate_exact_file(
        q1_q3_pair_csv,
        expected_sha256=CANONICAL_Q1_Q3_PAIR_SHA256,
        expected_bytes=CANONICAL_Q1_Q3_PAIR_BYTES,
    )
    phase_rows = load_pair_csv(phase_path, expected_fields=PHASE_F_FIELDS)
    q_rows = load_pair_csv(q_path, expected_fields=Q1_Q3_FIELDS)
    if len(phase_rows) != CANONICAL_PHASE_F_TOTAL_ROWS:
        raise RuntimeError("canonical Phase F row-count mismatch")
    if len(q_rows) != CANONICAL_Q1_Q3_TOTAL_ROWS:
        raise RuntimeError("canonical Q1/Q3 row-count mismatch")

    pair_rows, confirmatory = analyze_rows(phase_rows, q_rows)
    outputs = build_output_bytes(
        pair_rows=pair_rows,
        confirmatory=confirmatory,
        implementation_commit=implementation_commit,
        script_sha256=script_sha256,
        execution_authority_commit=execution_authority_commit,
    )
    output_sha256 = write_output_bytes(output_dir, outputs)
    return {
        "result": "PASS_STATISTICAL_ANALYSIS_PRODUCED",
        "output_sha256": output_sha256,
        "source_pair_count": SOURCE_PAIR_COUNT,
        "hypothesis_count": HYPOTHESIS_COUNT,
        "family_decision": family_decision(confirmatory),
        "statistical_testing": True,
        "direct_cross_layer_difference_test": True,
        "scientific_conclusion": "BOUNDED_BY_FROZEN_DECISION_RULES",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ContraMamba Gen4 NAME direct cross-layer paired-difference statistical analysis. "
            "Canonical execution requires a separately frozen execution authority."
        )
    )
    sub = parser.add_subparsers(dest="operation", required=True)
    run = sub.add_parser("run-canonical")
    run.add_argument("--phase-f-pair-csv", required=True)
    run.add_argument("--q1-q3-pair-csv", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--implementation-commit", required=True)
    run.add_argument("--script-sha256", required=True)
    run.add_argument("--execution-authority-commit", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.operation == "run-canonical":
        result = run_canonical(
            phase_f_pair_csv=args.phase_f_pair_csv,
            q1_q3_pair_csv=args.q1_q3_pair_csv,
            output_dir=args.output_dir,
            implementation_commit=args.implementation_commit,
            script_sha256=args.script_sha256,
            execution_authority_commit=args.execution_authority_commit,
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
    raise RuntimeError("unknown operation")


if __name__ == "__main__":
    raise SystemExit(main())
