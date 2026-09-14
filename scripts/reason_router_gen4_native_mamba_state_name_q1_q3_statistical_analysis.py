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

STATISTICAL_SPECIFICATION_COMMIT = "2030de52922c251caeafceb36dd5f2f57b5b39c7"
SCIENTIFIC_SPECIFICATION_COMMIT = "01801ad1617b2ebc3ffa859ba440636d4755a55c"
IMPLEMENTATION_AUTHORITY_COMMIT = "66a35cb469a24c881830fbf69169fb6efcd6cc0c"
RELEASE_BINDING_FREEZE_COMMIT = "a0ce07ad828c5ee0e6a4c36d8f9a6e972090590a"
PHASE_F_REFERENCE_IMPLEMENTATION_COMMIT = "e917e4c4fe0c94aa4ef5be33f2a69e6c64732189"

CANONICAL_INPUT_PATH = (
    "reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_"
    "scientific_extraction_v1/kinematic_endpoints.jsonl"
)
CANONICAL_INPUT_SHA256 = (
    "b47e32496f73f5493e8737b040b30bc4a23dd8cd22454a85244fa812273b6605"
)
CANONICAL_INPUT_BYTES = 513_396
CANONICAL_INPUT_ROWS = 1_200

SOURCE_PAIR_COUNT = 300
ROWS_PER_PAIR = 4
LAYERS = (5, 17)
ENDPOINTS = (
    "POST4_SPEED",
    "POST4_TURNING",
    "POST4_PATH_EFFICIENCY",
)
ESTIMAND = "DELTA_NAME"
SECONDARY_HYPOTHESIS_COUNT = 6
FAMILYWISE_ALPHA = 0.05
EXPECTED_IDENTITIES = frozenset(
    (layer, cell)
    for layer in LAYERS
    for cell in ("C0_SHAM", "C2_NAME")
)

PAIR_SCHEMA = "gen4_name_q1_q3_pair_level_contrasts_v1"
SECONDARY_SCHEMA = "gen4_name_q1_q3_secondary_confirmatory_results_v1"
MANIFEST_SCHEMA = "gen4_name_q1_q3_statistical_analysis_manifest_v1"

SUPPORTED = "PRESPECIFIED_SECONDARY_LAYER_LOCAL_NAME_KINEMATIC_CORRELATE_SUPPORTED"
NOT_ESTABLISHED = (
    "PRESPECIFIED_SECONDARY_LAYER_LOCAL_NAME_KINEMATIC_CORRELATE_NOT_ESTABLISHED"
)
FAMILY_SUPPORTED = "SECONDARY_LAYER_NAME_LOCALIZATION_SUPPORTED"
FAMILY_NOT_ESTABLISHED = "SECONDARY_LAYER_NAME_LOCALIZATION_NOT_ESTABLISHED"

PAIR_FIELDS = (
    "schema_version",
    "source_pair_id",
    "layer_index",
    "endpoint",
    "estimand",
    "contrast_value",
)
SECONDARY_FIELDS = (
    "schema_version",
    "layer_index",
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
    "name_q1_q3_pair_level_contrasts.csv",
    "name_q1_q3_secondary_confirmatory_results.csv",
    "name_q1_q3_statistical_analysis_manifest.json",
    "name_q1_q3_statistical_analysis_report_candidate.md",
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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} must be numeric and non-boolean")
    result = float(value)
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


def validate_endpoint_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = SOURCE_PAIR_COUNT,
) -> None:
    expected_rows = expected_pair_count * ROWS_PER_PAIR
    if len(records) != expected_rows:
        raise RuntimeError(
            f"endpoint row count mismatch: expected={expected_rows} observed={len(records)}"
        )

    identities: set[tuple[str, int, str]] = set()
    row_layer_ids: set[tuple[str, int]] = set()
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    required = {
        "source_pair_id",
        "row_id",
        "contrast_cell_id",
        "semantic_anchor",
        "layer_index",
        *ENDPOINTS,
    }

    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"non-object endpoint row: {index}")

        missing = required - set(row)
        if missing:
            raise RuntimeError(f"required fields missing: {sorted(missing)}")

        pair_id = str(row["source_pair_id"])
        row_id = str(row["row_id"])
        cell = str(row["contrast_cell_id"])
        anchor = str(row["semantic_anchor"])
        layer = row["layer_index"]

        if not pair_id:
            raise RuntimeError("empty source_pair_id")
        if not row_id:
            raise RuntimeError("empty row_id")

        if anchor != "A_NAME":
            raise RuntimeError(f"unexpected semantic_anchor: {anchor}")
        if type(layer) is not int or layer not in LAYERS:
            raise RuntimeError(f"unexpected layer_index: {layer}")

        row_layer_identity = (row_id, layer)
        if row_layer_identity in row_layer_ids:
            raise RuntimeError(
                f"duplicate row/layer identity: {row_layer_identity}"
            )
        row_layer_ids.add(row_layer_identity)

        if cell not in {"C0_SHAM", "C2_NAME"}:
            raise RuntimeError(f"unexpected contrast_cell_id: {cell}")

        identity = (pair_id, layer, cell)
        if identity in identities:
            raise RuntimeError(f"duplicate pair/layer/cell identity: {identity}")
        identities.add(identity)

        for endpoint in ENDPOINTS:
            _finite_number(row[endpoint], field=endpoint)

        by_pair[pair_id].append(row)

    if len(by_pair) != expected_pair_count:
        raise RuntimeError(
            f"source-pair count mismatch: expected={expected_pair_count} "
            f"observed={len(by_pair)}"
        )

    for pair_id, rows in by_pair.items():
        if len(rows) != ROWS_PER_PAIR:
            raise RuntimeError(f"source-pair row count mismatch: {pair_id}")

        observed = {
            (int(row["layer_index"]), str(row["contrast_cell_id"]))
            for row in rows
        }
        if observed != EXPECTED_IDENTITIES:
            missing = sorted(EXPECTED_IDENTITIES - observed)
            extra = sorted(observed - EXPECTED_IDENTITIES)
            raise RuntimeError(
                f"source-pair endpoint matrix mismatch: {pair_id}; "
                f"missing={missing}; extra={extra}"
            )


def _record_lookup(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    return {
        (
            str(row["source_pair_id"]),
            int(row["layer_index"]),
            str(row["contrast_cell_id"]),
        ): row
        for row in records
    }


def construct_pair_level_contrasts(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = SOURCE_PAIR_COUNT,
) -> list[dict[str, Any]]:
    validate_endpoint_records(records, expected_pair_count=expected_pair_count)
    lookup = _record_lookup(records)
    pair_ids = sorted({str(row["source_pair_id"]) for row in records})

    output: list[dict[str, Any]] = []
    for layer in LAYERS:
        for endpoint in ENDPOINTS:
            for pair_id in pair_ids:
                treatment = _finite_number(
                    lookup[(pair_id, layer, "C2_NAME")][endpoint],
                    field=endpoint,
                )
                sham = _finite_number(
                    lookup[(pair_id, layer, "C0_SHAM")][endpoint],
                    field=endpoint,
                )
                output.append(
                    {
                        "schema_version": PAIR_SCHEMA,
                        "source_pair_id": pair_id,
                        "layer_index": layer,
                        "endpoint": endpoint,
                        "estimand": ESTIMAND,
                        "contrast_value": float(treatment - sham),
                    }
                )

    expected = expected_pair_count * SECONDARY_HYPOTHESIS_COUNT
    if len(output) != expected:
        raise RuntimeError(
            f"pair-level contrast row count mismatch: expected={expected} "
            f"observed={len(output)}"
        )
    return output


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
            f"secondary N mismatch: expected={expected_n} observed={len(values)}"
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
        raise RuntimeError("zero-variance secondary contrast distribution")

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
    if len(raw_p_values) != SECONDARY_HYPOTHESIS_COUNT:
        raise RuntimeError(
            f"Holm family must contain exactly {SECONDARY_HYPOTHESIS_COUNT} p-values"
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


def family_decision(secondary: Sequence[Mapping[str, Any]]) -> str:
    if len(secondary) != SECONDARY_HYPOTHESIS_COUNT:
        raise RuntimeError("secondary result row count drift")
    return (
        FAMILY_SUPPORTED
        if any(bool(row["reject_holm_alpha_0_05"]) for row in secondary)
        else FAMILY_NOT_ESTABLISHED
    )


def analyze_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = SOURCE_PAIR_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows = construct_pair_level_contrasts(
        records,
        expected_pair_count=expected_pair_count,
    )

    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in pair_rows:
        grouped[(int(row["layer_index"]), str(row["endpoint"]))].append(
            float(row["contrast_value"])
        )

    secondary: list[dict[str, Any]] = []
    for layer in LAYERS:
        for endpoint in ENDPOINTS:
            values = grouped[(layer, endpoint)]
            stats = one_sample_t_statistics(
                values,
                expected_n=expected_pair_count,
            )
            secondary.append(
                {
                    "schema_version": SECONDARY_SCHEMA,
                    "layer_index": layer,
                    "endpoint": endpoint,
                    "estimand": ESTIMAND,
                    **stats,
                }
            )

    correction = holm_bonferroni(
        [float(row["raw_p_value"]) for row in secondary]
    )
    for row, adjusted in zip(secondary, correction):
        row.update(adjusted)
        row["decision"] = (
            SUPPORTED
            if bool(row["reject_holm_alpha_0_05"])
            else NOT_ESTABLISHED
        )

    if len(secondary) != SECONDARY_HYPOTHESIS_COUNT:
        raise RuntimeError("secondary result row count drift")

    return pair_rows, secondary


def render_report(
    *,
    input_path: str,
    input_sha256: str,
    implementation_commit: str,
    script_sha256: str,
    execution_authority_commit: str,
    secondary: Sequence[Mapping[str, Any]],
) -> bytes:
    family = family_decision(secondary)
    lines = [
        "# ContraMamba Gen4 Native Mamba State Bridge",
        "# NAME Q1/Q3 Secondary Six-Family Statistical Analysis Report - Candidate",
        "",
        "## Provenance",
        "",
        f"- Statistical specification: `{STATISTICAL_SPECIFICATION_COMMIT}`",
        f"- Scientific specification: `{SCIENTIFIC_SPECIFICATION_COMMIT}`",
        f"- Implementation authority: `{IMPLEMENTATION_AUTHORITY_COMMIT}`",
        f"- Extraction release binding: `{RELEASE_BINDING_FREEZE_COMMIT}`",
        f"- Statistical execution authority: `{execution_authority_commit}`",
        f"- Implementation commit: `{implementation_commit}`",
        f"- Script SHA256: `{script_sha256}`",
        f"- Input: `{input_path}`",
        f"- Input SHA256: `{input_sha256}`",
        f"- Source pairs: `{SOURCE_PAIR_COUNT}`",
        "",
        "## Secondary confirmatory family",
        "",
        "| Layer | Endpoint | Estimand | Mean | t | raw p | Holm p | Reject | d_z |",
        "|---:|---|---|---:|---:|---:|---:|:---:|---:|",
    ]

    for row in secondary:
        lines.append(
            "| {layer} | {endpoint} | {estimand} | {mean} | {t} | {raw_p} | "
            "{holm_p} | {reject} | {dz} |".format(
                layer=row["layer_index"],
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
            "## Scope limitation",
            "",
            "These results concern only the frozen 300 source-pair Gen4 population,",
            "DELTA_NAME, the architecture-defined secondary layers 5 and 17, and",
            "the three prespecified local native-state kinematic endpoints.",
            "",
            "A supported member is a prespecified secondary-layer local NAME",
            "kinematic correlate. It does not establish causal mediation, necessity,",
            "sufficiency, state-to-output causation, significant depth selectivity,",
            "arbitrary-model generalization, or arbitrary-dataset generalization.",
            "",
            "No direct layer-to-layer difference test is part of this family.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def build_output_bytes(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    input_path: str,
    input_sha256: str,
    input_bytes: int,
    input_rows: int,
    implementation_commit: str,
    script_sha256: str,
    execution_authority_commit: str,
) -> dict[str, bytes]:
    if len(pair_rows) != SOURCE_PAIR_COUNT * SECONDARY_HYPOTHESIS_COUNT:
        raise RuntimeError("pair-level output cardinality mismatch")
    if len(secondary) != SECONDARY_HYPOTHESIS_COUNT:
        raise RuntimeError("secondary output cardinality mismatch")

    pair_data = csv_bytes(pair_rows, PAIR_FIELDS)
    secondary_data = csv_bytes(secondary, SECONDARY_FIELDS)
    report_data = render_report(
        input_path=input_path,
        input_sha256=input_sha256,
        implementation_commit=implementation_commit,
        script_sha256=script_sha256,
        execution_authority_commit=execution_authority_commit,
        secondary=secondary,
    )

    peer_sha256 = {
        "name_q1_q3_pair_level_contrasts.csv": sha256_bytes(pair_data),
        "name_q1_q3_secondary_confirmatory_results.csv": sha256_bytes(
            secondary_data
        ),
        "name_q1_q3_statistical_analysis_report_candidate.md": sha256_bytes(
            report_data
        ),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": "PASS_STATISTICAL_ANALYSIS_PRODUCED",
        "statistical_specification_commit": STATISTICAL_SPECIFICATION_COMMIT,
        "scientific_specification_commit": SCIENTIFIC_SPECIFICATION_COMMIT,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "extraction_release_binding_freeze_commit": RELEASE_BINDING_FREEZE_COMMIT,
        "statistical_execution_authority_commit": execution_authority_commit,
        "implementation_commit": implementation_commit,
        "script_sha256": script_sha256,
        "input_path": input_path,
        "input_sha256": input_sha256,
        "input_bytes": int(input_bytes),
        "input_rows": int(input_rows),
        "source_pair_count": SOURCE_PAIR_COUNT,
        "rows_per_source_pair": ROWS_PER_PAIR,
        "layer_order": list(LAYERS),
        "endpoint_order": list(ENDPOINTS),
        "structural_estimand": ESTIMAND,
        "hypothesis_count": SECONDARY_HYPOTHESIS_COUNT,
        "n_per_hypothesis": SOURCE_PAIR_COUNT,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "multiplicity_method": "HOLM_BONFERRONI",
        "numeric_analysis_dtype": "FLOAT64",
        "primary_inferential_unit": "SOURCE_PAIR",
        "family_decision": family_decision(secondary),
        "overall_adaptive_program_fwer_claimed": False,
        "cross_layer_difference_test": False,
        "peer_artifact_sha256": peer_sha256,
        "statistical_testing": True,
        "model_inference": False,
        "training": False,
        "backward": False,
        "scientific_scope": (
            "Frozen 300 source pairs, DELTA_NAME, layers 5 and 17, and three "
            "prespecified local native-state kinematic endpoints only."
        ),
    }
    if "manifest_sha256" in manifest:
        raise RuntimeError("manifest self SHA256 is forbidden")

    return {
        "name_q1_q3_pair_level_contrasts.csv": pair_data,
        "name_q1_q3_secondary_confirmatory_results.csv": secondary_data,
        "name_q1_q3_statistical_analysis_manifest.json": canonical_json_bytes(
            manifest
        ),
        "name_q1_q3_statistical_analysis_report_candidate.md": report_data,
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


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank JSONL record at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL row at line {line_number}")
            rows.append(value)
    return rows


def run_canonical(
    *,
    input_jsonl: str | Path,
    output_dir: str | Path,
    implementation_commit: str,
    script_sha256: str,
    execution_authority_commit: str,
) -> dict[str, Any]:
    input_path = _validate_exact_file(
        input_jsonl,
        expected_sha256=CANONICAL_INPUT_SHA256,
        expected_bytes=CANONICAL_INPUT_BYTES,
    )
    rows = load_jsonl(input_path)
    if len(rows) != CANONICAL_INPUT_ROWS:
        raise RuntimeError("canonical input row-count mismatch")

    pair_rows, secondary = analyze_records(rows)
    outputs = build_output_bytes(
        pair_rows=pair_rows,
        secondary=secondary,
        input_path=CANONICAL_INPUT_PATH,
        input_sha256=CANONICAL_INPUT_SHA256,
        input_bytes=CANONICAL_INPUT_BYTES,
        input_rows=CANONICAL_INPUT_ROWS,
        implementation_commit=implementation_commit,
        script_sha256=script_sha256,
        execution_authority_commit=execution_authority_commit,
    )
    output_sha256 = write_output_bytes(output_dir, outputs)
    return {
        "result": "PASS_STATISTICAL_ANALYSIS_PRODUCED",
        "output_sha256": output_sha256,
        "source_pair_count": SOURCE_PAIR_COUNT,
        "hypothesis_count": SECONDARY_HYPOTHESIS_COUNT,
        "family_decision": family_decision(secondary),
        "statistical_testing": True,
        "scientific_conclusion": "BOUNDED_BY_FROZEN_DECISION_RULES",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ContraMamba Gen4 NAME Q1/Q3 secondary-six statistical analysis. "
            "Canonical execution requires a separately frozen execution authority."
        )
    )
    sub = parser.add_subparsers(dest="operation", required=True)
    run = sub.add_parser("run-canonical")
    run.add_argument("--input-jsonl", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--implementation-commit", required=True)
    run.add_argument("--script-sha256", required=True)
    run.add_argument("--execution-authority-commit", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.operation == "run-canonical":
        result = run_canonical(
            input_jsonl=args.input_jsonl,
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
