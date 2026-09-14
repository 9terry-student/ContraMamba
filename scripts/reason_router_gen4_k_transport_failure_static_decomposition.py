#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-transport-failure-static-decomposition"
CLOSURE_COMMIT = "ad375c88385eef507e151f95717e951919a6a3fe"
CLOSURE_REPORT_REL = (
    "reports/"
    "reason_router_gen4_k_directional_alignment_transport_"
    "backend_invariant_closure_report_candidate.md"
)

CPU_REFERENCE_HEAD = "8496ece911e0d461f0abbdf1a0fa619f8a2f22ab"
CPU_REFERENCE_ZIP_SHA256 = (
    "25a9e6a862f7c1ad7c272d85cb5000ba8542cca2c8976b785021e5e4159ebcaf"
)
EXPECTED_OUTCOME = "DIRECTIONAL_ALIGNMENT_CAUSAL_TRANSPORT_NOT_ESTABLISHED"
SOURCE_PAIR_COUNT = 300
MODEL_FORWARD_COUNT = 2400

SCHEMA = "gen4-k-transport-failure-static-decomposition-v1"

BUNDLE_FILES = frozenset(
    {
        "manifest.json",
        "item_metrics.jsonl",
        "summary.json",
        "SHA256SUMS.txt",
    }
)

RESPONSES = (
    "R_ALIGN",
    "R_MAG",
    "ALIGNMENT_SPECIFICITY",
)

FEATURES = (
    "baseline_delta",
    "baseline_severity_abs",
    "target_A",
    "target_B",
    "target_C",
    "reference_A",
    "reference_B",
    "reference_C",
    "alignment_shift",
    "alignment_shift_abs",
    "magnitude_log_ratio_A",
    "magnitude_log_ratio_B",
    "magnitude_shift_norm",
)

QUARTILE_FEATURES = (
    "baseline_delta",
    "baseline_severity_abs",
    "target_C",
    "alignment_shift",
    "alignment_shift_abs",
    "magnitude_shift_norm",
)


class StaticDecompositionError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StaticDecompositionError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticDecompositionError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo() -> dict[str, str]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    require(
        branch == EXPECTED_BRANCH,
        f"BRANCH_MISMATCH:{branch}",
    )
    require(
        git("status", "--porcelain") == "",
        "WORKTREE_NOT_CLEAN",
    )

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            CLOSURE_COMMIT,
            head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "CLOSURE_COMMIT_NOT_ANCESTOR",
    )

    require(
        (
            ROOT / CLOSURE_REPORT_REL
        ).is_file(),
        "CLOSURE_REPORT_MISSING",
    )

    return {
        "analysis_branch": branch,
        "analysis_head": head,
        "closure_commit": CLOSURE_COMMIT,
        "closure_report_rel": CLOSURE_REPORT_REL,
    }


def _parse_checksums(text: str) -> dict[str, str]:
    result: dict[str, str] = {}

    for line in text.splitlines():
        if not line.strip():
            continue

        parts = line.split("  ", 1)
        require(
            len(parts) == 2,
            "MALFORMED_CHECKSUM_LINE",
        )
        digest, name = parts
        require(
            len(digest) == 64,
            "MALFORMED_CHECKSUM_DIGEST",
        )
        require(
            name not in result,
            f"DUPLICATE_CHECKSUM:{name}",
        )
        result[name] = digest

    return result


def load_cpu_bundle(path: Path) -> dict[str, bytes]:
    require(
        path.is_file(),
        f"CPU_ZIP_MISSING:{path}",
    )
    require(
        sha256_file(path) == CPU_REFERENCE_ZIP_SHA256,
        "CPU_REFERENCE_ZIP_SHA256_MISMATCH",
    )

    with zipfile.ZipFile(path, "r") as archive:
        regular = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
        ]

        manifest_names = [
            name
            for name in regular
            if PurePosixPath(name).name == "manifest.json"
        ]

        roots: list[PurePosixPath] = []
        regular_set = set(regular)

        for manifest_name in manifest_names:
            parent = PurePosixPath(manifest_name).parent
            members = {
                (
                    (parent / filename).as_posix()
                    if parent != PurePosixPath(".")
                    else filename
                )
                for filename in BUNDLE_FILES
            }
            if members <= regular_set:
                roots.append(parent)

        require(
            len(roots) == 1,
            f"CPU_ZIP_BUNDLE_ROOT_COUNT:{len(roots)}",
        )
        root = roots[0]

        root_files = {
            PurePosixPath(name).name
            for name in regular
            if PurePosixPath(name).parent == root
        }
        require(
            root_files == BUNDLE_FILES,
            f"CPU_ZIP_FILE_SET:{sorted(root_files)}",
        )

        selected = {}
        for filename in BUNDLE_FILES:
            member = (
                (root / filename).as_posix()
                if root != PurePosixPath(".")
                else filename
            )
            selected[filename] = archive.read(member)

    expected = _parse_checksums(
        selected["SHA256SUMS.txt"].decode("utf-8")
    )
    require(
        set(expected)
        == {
            "manifest.json",
            "item_metrics.jsonl",
            "summary.json",
        },
        "INTERNAL_CHECKSUM_FILE_SET",
    )

    for filename, digest in expected.items():
        observed = hashlib.sha256(
            selected[filename]
        ).hexdigest()
        require(
            observed == digest,
            f"INTERNAL_CHECKSUM_MISMATCH:{filename}",
        )

    return selected


def decode_bundle(
    files: Mapping[str, bytes],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(
        files["manifest.json"].decode("utf-8")
    )
    summary = json.loads(
        files["summary.json"].decode("utf-8")
    )
    items = [
        json.loads(line)
        for line in files[
            "item_metrics.jsonl"
        ].decode("utf-8").splitlines()
        if line.strip()
    ]

    require(
        isinstance(manifest, dict),
        "MANIFEST_NOT_OBJECT",
    )
    require(
        isinstance(summary, dict),
        "SUMMARY_NOT_OBJECT",
    )
    require(
        all(isinstance(row, dict) for row in items),
        "ITEM_NOT_OBJECT",
    )

    return manifest, items, summary


def validate_cpu_reference(
    manifest: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    require(
        manifest["mode"] == "full",
        "CPU_MODE",
    )
    require(
        manifest["runtime_git_head"] == CPU_REFERENCE_HEAD,
        "CPU_REFERENCE_HEAD",
    )
    require(
        int(manifest["source_pair_count"])
        == SOURCE_PAIR_COUNT,
        "CPU_PAIR_COUNT",
    )
    require(
        int(manifest["model_forward_count"])
        == MODEL_FORWARD_COUNT,
        "CPU_FORWARD_COUNT",
    )
    require(
        len(items) == SOURCE_PAIR_COUNT,
        "CPU_ITEM_COUNT",
    )
    require(
        summary["outcome"] == EXPECTED_OUTCOME,
        "CPU_OUTCOME",
    )
    require(
        summary[
            "all_mandatory_manipulation_checks_pass"
        ]
        is True,
        "CPU_MANIPULATION_CHECKS",
    )

    require(
        manifest["training_executed"] is False
        and manifest["backward_executed"] is False
        and manifest["task_heads_executed"] is False
        and manifest["logits_read"] is False
        and manifest["raw_vectors_persisted"] is False,
        "CPU_SAFETY_FLAGS",
    )

    pair_ids = [
        str(row["source_pair_id"])
        for row in items
    ]
    require(
        len(set(pair_ids)) == SOURCE_PAIR_COUNT,
        "CPU_PAIR_UNIQUENESS",
    )

    required_fields = {
        "source_pair_id",
        "target_A",
        "target_B",
        "target_C",
        "reference_A",
        "reference_B",
        "reference_C",
        "delta_baseline",
        "R_ALIGN",
        "R_MAG",
        "ALIGNMENT_SPECIFICITY",
    }
    for index, row in enumerate(items):
        require(
            required_fields <= set(row),
            f"CPU_ITEM_SCHEMA:{index}",
        )


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    require(
        math.isfinite(result),
        f"NONFINITE:{label}",
    )
    return result


def derive_features(
    row: Mapping[str, Any],
) -> dict[str, float]:
    target_a = finite_float(
        row["target_A"], "target_A"
    )
    target_b = finite_float(
        row["target_B"], "target_B"
    )
    target_c = finite_float(
        row["target_C"], "target_C"
    )
    reference_a = finite_float(
        row["reference_A"], "reference_A"
    )
    reference_b = finite_float(
        row["reference_B"], "reference_B"
    )
    reference_c = finite_float(
        row["reference_C"], "reference_C"
    )
    baseline_delta = finite_float(
        row["delta_baseline"], "delta_baseline"
    )

    require(
        target_a > 0.0
        and target_b > 0.0
        and reference_a > 0.0
        and reference_b > 0.0,
        "NONPOSITIVE_MAGNITUDE",
    )

    shift = reference_c - target_c
    log_ratio_a = math.log(
        reference_a / target_a
    )
    log_ratio_b = math.log(
        reference_b / target_b
    )

    return {
        "baseline_delta": baseline_delta,
        "baseline_severity_abs":
            abs(baseline_delta),
        "target_A": target_a,
        "target_B": target_b,
        "target_C": target_c,
        "reference_A": reference_a,
        "reference_B": reference_b,
        "reference_C": reference_c,
        "alignment_shift": shift,
        "alignment_shift_abs": abs(shift),
        "magnitude_log_ratio_A":
            log_ratio_a,
        "magnitude_log_ratio_B":
            log_ratio_b,
        "magnitude_shift_norm":
            math.hypot(
                log_ratio_a,
                log_ratio_b,
            ),
    }


def derive_responses(
    row: Mapping[str, Any],
) -> dict[str, float]:
    return {
        name: finite_float(
            row[name],
            name,
        )
        for name in RESPONSES
    }


def quantile_sorted(
    values: Sequence[float],
    probability: float,
) -> float:
    require(
        len(values) > 0,
        "EMPTY_QUANTILE",
    )
    require(
        0.0 <= probability <= 1.0,
        "QUANTILE_PROBABILITY",
    )

    ordered = sorted(
        float(value)
        for value in values
    )
    position = (
        (len(ordered) - 1)
        * probability
    )
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def summarize_values(
    values: Sequence[float],
) -> dict[str, Any]:
    finite = [
        finite_float(v, "summary_value")
        for v in values
    ]
    require(
        len(finite) > 0,
        "EMPTY_SUMMARY",
    )

    positive = sum(v > 0.0 for v in finite)
    negative = sum(v < 0.0 for v in finite)
    zero = len(finite) - positive - negative

    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "sample_sd": (
            statistics.stdev(finite)
            if len(finite) >= 2
            else 0.0
        ),
        "min": min(finite),
        "q10": quantile_sorted(finite, 0.10),
        "q25": quantile_sorted(finite, 0.25),
        "q50": quantile_sorted(finite, 0.50),
        "q75": quantile_sorted(finite, 0.75),
        "q90": quantile_sorted(finite, 0.90),
        "max": max(finite),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "positive_fraction":
            positive / len(finite),
        "negative_fraction":
            negative / len(finite),
        "zero_fraction":
            zero / len(finite),
    }


def pearson(
    xs: Sequence[float],
    ys: Sequence[float],
) -> float:
    require(
        len(xs) == len(ys) and len(xs) >= 2,
        "PEARSON_SHAPE",
    )

    x = [
        finite_float(v, "pearson_x")
        for v in xs
    ]
    y = [
        finite_float(v, "pearson_y")
        for v in ys
    ]

    mx = statistics.fmean(x)
    my = statistics.fmean(y)

    dx = [v - mx for v in x]
    dy = [v - my for v in y]

    sx = math.sqrt(
        sum(v * v for v in dx)
    )
    sy = math.sqrt(
        sum(v * v for v in dy)
    )

    require(
        sx > 0.0 and sy > 0.0,
        "PEARSON_ZERO_VARIANCE",
    )

    result = (
        sum(a * b for a, b in zip(dx, dy))
        / (sx * sy)
    )
    return max(
        -1.0,
        min(1.0, result),
    )


def average_ranks(
    values: Sequence[float],
) -> list[float]:
    pairs = sorted(
        enumerate(
            finite_float(v, "rank_value")
            for v in values
        ),
        key=lambda item: (
            item[1],
            item[0],
        ),
    )

    result = [0.0] * len(pairs)
    i = 0

    while i < len(pairs):
        j = i + 1
        value = pairs[i][1]

        while (
            j < len(pairs)
            and pairs[j][1] == value
        ):
            j += 1

        average_rank = (
            (i + 1) + j
        ) / 2.0

        for k in range(i, j):
            result[
                pairs[k][0]
            ] = average_rank

        i = j

    return result


def spearman(
    xs: Sequence[float],
    ys: Sequence[float],
) -> float:
    return pearson(
        average_ranks(xs),
        average_ranks(ys),
    )


def stable_quartiles(
    pair_ids: Sequence[str],
    feature_values: Sequence[float],
) -> list[list[int]]:
    require(
        len(pair_ids)
        == len(feature_values)
        == SOURCE_PAIR_COUNT,
        "QUARTILE_INPUT_SHAPE",
    )

    ordered = sorted(
        range(SOURCE_PAIR_COUNT),
        key=lambda index: (
            finite_float(
                feature_values[index],
                "quartile_feature",
            ),
            str(pair_ids[index]),
        ),
    )

    require(
        SOURCE_PAIR_COUNT % 4 == 0,
        "QUARTILE_POPULATION_DIVISIBILITY",
    )
    width = SOURCE_PAIR_COUNT // 4

    result = [
        ordered[
            quartile * width:
            (quartile + 1) * width
        ]
        for quartile in range(4)
    ]
    require(
        all(len(group) == width for group in result),
        "QUARTILE_WIDTH",
    )
    return result


def correlation_table(
    feature_columns: Mapping[str, Sequence[float]],
    response_columns: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for response_name in RESPONSES:
        result[response_name] = {}

        for feature_name in FEATURES:
            result[response_name][
                feature_name
            ] = {
                "pearson": pearson(
                    feature_columns[
                        feature_name
                    ],
                    response_columns[
                        response_name
                    ],
                ),
                "spearman": spearman(
                    feature_columns[
                        feature_name
                    ],
                    response_columns[
                        response_name
                    ],
                ),
            }

    return result


def response_response_correlations(
    response_columns: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    pairs = (
        ("R_ALIGN", "R_MAG"),
        (
            "R_ALIGN",
            "ALIGNMENT_SPECIFICITY",
        ),
        (
            "R_MAG",
            "ALIGNMENT_SPECIFICITY",
        ),
    )

    result = {}
    for left, right in pairs:
        key = f"{left}__vs__{right}"
        result[key] = {
            "pearson": pearson(
                response_columns[left],
                response_columns[right],
            ),
            "spearman": spearman(
                response_columns[left],
                response_columns[right],
            ),
        }
    return result


def quartile_diagnostics(
    *,
    pair_ids: Sequence[str],
    feature_columns: Mapping[str, Sequence[float]],
    response_columns: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for feature_name in QUARTILE_FEATURES:
        values = feature_columns[
            feature_name
        ]
        groups = stable_quartiles(
            pair_ids,
            values,
        )

        feature_result = {}
        for q_index, indices in enumerate(
            groups,
            start=1,
        ):
            label = f"Q{q_index}"
            group_values = [
                values[index]
                for index in indices
            ]
            group_pair_ids = [
                str(pair_ids[index])
                for index in indices
            ]

            feature_result[label] = {
                "n": len(indices),
                "feature_min":
                    min(group_values),
                "feature_median":
                    statistics.median(
                        group_values
                    ),
                "feature_max":
                    max(group_values),
                "pair_ids":
                    group_pair_ids,
                "responses": {
                    response_name:
                        summarize_values(
                            [
                                response_columns[
                                    response_name
                                ][index]
                                for index in indices
                            ]
                        )
                    for response_name
                    in RESPONSES
                },
            }

        result[feature_name] = feature_result

    return result


def build_report(
    *,
    provenance: Mapping[str, str],
    cpu_zip: Path,
    manifest: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    pair_ids = [
        str(row["source_pair_id"])
        for row in items
    ]
    feature_rows = [
        derive_features(row)
        for row in items
    ]
    response_rows = [
        derive_responses(row)
        for row in items
    ]

    feature_columns = {
        name: [
            row[name]
            for row in feature_rows
        ]
        for name in FEATURES
    }
    response_columns = {
        name: [
            row[name]
            for row in response_rows
        ]
        for name in RESPONSES
    }

    # Reconfirm that our itemwise reductions reproduce the
    # already-frozen CPU summary. This is an integrity check, not
    # a new hypothesis test.
    frozen_hypothesis = {
        str(row["id"]): row
        for row in summary[
            "hypothesis_family"
        ]
    }
    require(
        set(frozen_hypothesis)
        == {
            "H1_R_ALIGN_GT_ZERO",
            "H2_ALIGNMENT_SPECIFICITY_GT_ZERO",
        },
        "FROZEN_HYPOTHESIS_FAMILY",
    )

    align_mean = statistics.fmean(
        response_columns["R_ALIGN"]
    )
    specificity_mean = statistics.fmean(
        response_columns[
            "ALIGNMENT_SPECIFICITY"
        ]
    )

    require(
        abs(
            align_mean
            - float(
                frozen_hypothesis[
                    "H1_R_ALIGN_GT_ZERO"
                ]["mean"]
            )
        )
        <= 1e-15,
        "R_ALIGN_MEAN_REPRODUCTION",
    )
    require(
        abs(
            specificity_mean
            - float(
                frozen_hypothesis[
                    "H2_ALIGNMENT_SPECIFICITY_GT_ZERO"
                ]["mean"]
            )
        )
        <= 1e-15,
        "SPECIFICITY_MEAN_REPRODUCTION",
    )

    return {
        "schema_version": SCHEMA,
        "analysis_phase":
            "POSTHOC_STATIC_DIAGNOSTIC",
        "analysis_scope":
            "TRANSPORT_FAILURE_DECOMPOSITION_ONLY",
        "exploratory_posthoc": True,
        "new_inferential_statistics_executed":
            False,
        "new_causal_claim_authorized":
            False,
        "training_executed": False,
        "model_forward_count": 0,
        "tokenizer_invoked": False,
        **dict(provenance),
        "cpu_reference_zip": {
            "path": str(cpu_zip),
            "sha256":
                CPU_REFERENCE_ZIP_SHA256,
            "execution_head":
                CPU_REFERENCE_HEAD,
            "source_pair_count":
                SOURCE_PAIR_COUNT,
            "model_forward_count":
                MODEL_FORWARD_COUNT,
            "outcome":
                EXPECTED_OUTCOME,
        },
        "frozen_summary_reproduction": {
            "R_ALIGN_mean": align_mean,
            "ALIGNMENT_SPECIFICITY_mean":
                specificity_mean,
            "result": "PASS",
        },
        "feature_definitions": {
            "baseline_delta":
                "delta_baseline = PE(C2)-PE(C0)",
            "baseline_severity_abs":
                "abs(delta_baseline)",
            "target_A":
                "pre-intervention target geometry A",
            "target_B":
                "pre-intervention target geometry B",
            "target_C":
                "pre-intervention target cosine C",
            "reference_A":
                "pre-intervention reference geometry A",
            "reference_B":
                "pre-intervention reference geometry B",
            "reference_C":
                "pre-intervention reference cosine C",
            "alignment_shift":
                "reference_C - target_C",
            "alignment_shift_abs":
                "abs(reference_C - target_C)",
            "magnitude_log_ratio_A":
                "log(reference_A / target_A)",
            "magnitude_log_ratio_B":
                "log(reference_B / target_B)",
            "magnitude_shift_norm":
                "hypot(log(reference_A/target_A), log(reference_B/target_B))",
        },
        "quartile_rule": {
            "type":
                "stable_equal_count_rank_quartiles",
            "population": SOURCE_PAIR_COUNT,
            "items_per_quartile":
                SOURCE_PAIR_COUNT // 4,
            "tie_break":
                "source_pair_id lexical order",
            "features":
                list(QUARTILE_FEATURES),
            "purpose":
                "descriptive regime localization only; not a new significance family",
        },
        "response_summary": {
            name: summarize_values(
                response_columns[name]
            )
            for name in RESPONSES
        },
        "feature_summary": {
            name: summarize_values(
                feature_columns[name]
            )
            for name in FEATURES
        },
        "feature_response_correlations":
            correlation_table(
                feature_columns,
                response_columns,
            ),
        "response_response_correlations":
            response_response_correlations(
                response_columns
            ),
        "quartile_diagnostics":
            quartile_diagnostics(
                pair_ids=pair_ids,
                feature_columns=
                    feature_columns,
                response_columns=
                    response_columns,
            ),
        "claim_boundary": [
            "This report is descriptive/post-hoc and cannot rescue the closed bridge hypothesis.",
            "No layer, endpoint, offset, channel, checkpoint, or intervention sweep is performed.",
            "Positive subgroups, if present, are hypotheses for a new study rather than evidence of transport.",
            "The closed backend-invariant negative result remains unchanged.",
        ],
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-zip",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    output = Path(args.output)
    cpu_zip = Path(args.cpu_zip)

    require(
        not output.exists(),
        f"OUTPUT_COLLISION:{output}",
    )

    provenance = authenticate_repo()
    files = load_cpu_bundle(cpu_zip)
    manifest, items, summary = (
        decode_bundle(files)
    )
    validate_cpu_reference(
        manifest,
        items,
        summary,
    )

    report = build_report(
        provenance=provenance,
        cpu_zip=cpu_zip,
        manifest=manifest,
        items=items,
        summary=summary,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        "RESULT = PASS_TRANSPORT_FAILURE_STATIC_DECOMPOSITION"
    )
    print(
        "SOURCE_PAIR_COUNT =",
        SOURCE_PAIR_COUNT,
    )
    print(
        "R_ALIGN_MEAN =",
        report[
            "response_summary"
        ]["R_ALIGN"]["mean"],
    )
    print(
        "R_ALIGN_POSITIVE_FRACTION =",
        report[
            "response_summary"
        ]["R_ALIGN"][
            "positive_fraction"
        ],
    )
    print(
        "SPECIFICITY_MEAN =",
        report[
            "response_summary"
        ][
            "ALIGNMENT_SPECIFICITY"
        ]["mean"],
    )
    print(
        "SPECIFICITY_POSITIVE_FRACTION =",
        report[
            "response_summary"
        ][
            "ALIGNMENT_SPECIFICITY"
        ][
            "positive_fraction"
        ],
    )
    print(
        "MODEL_FORWARD_COUNT = 0"
    )
    print(
        "NEW_INFERENTIAL_STATISTICS = NONE"
    )


if __name__ == "__main__":
    main()
