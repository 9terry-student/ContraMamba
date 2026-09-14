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
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-transport-failure-static-decomposition"
STATIC_FREEZE_COMMIT = "4d68f06d5bfaf9bc680e625000062e36b29c0b30"
STATIC_REPORT_REL = (
    "reports/"
    "reason_router_gen4_k_transport_failure_static_decomposition_v1.json"
)
STATIC_REPORT_SHA256 = (
    "10adb86cf8b86b14230d657a19420c02a9bc0d19bebfff9ff3374fee879fd459"
)

CPU_REFERENCE_HEAD = "8496ece911e0d461f0abbdf1a0fa619f8a2f22ab"
CPU_REFERENCE_ZIP_SHA256 = (
    "25a9e6a862f7c1ad7c272d85cb5000ba8542cca2c8976b785021e5e4159ebcaf"
)
EXPECTED_OUTCOME = "DIRECTIONAL_ALIGNMENT_CAUSAL_TRANSPORT_NOT_ESTABLISHED"
SOURCE_PAIR_COUNT = 300
MODEL_FORWARD_COUNT = 2400

SCHEMA = "gen4-k-transport-failure-tail-influence-v1"

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

REMOVAL_COUNTS = (1, 5, 10, 20, 30, 40, 50, 60, 75)
TAIL_COUNTS = (10, 20, 30, 60)
TRIM_FRACTIONS = (0.05, 0.10, 0.20)

QUARTILE_FEATURES = (
    "alignment_shift_abs",
    "baseline_delta",
    "baseline_severity_abs",
    "target_C",
    "magnitude_shift_norm",
)


class TailInfluenceError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise TailInfluenceError(message)


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
        raise TailInfluenceError(
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
            STATIC_FREEZE_COMMIT,
            head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "STATIC_FREEZE_NOT_ANCESTOR",
    )

    static_path = ROOT / STATIC_REPORT_REL
    require(
        static_path.is_file(),
        "STATIC_REPORT_MISSING",
    )
    require(
        sha256_file(static_path)
        == STATIC_REPORT_SHA256,
        "STATIC_REPORT_SHA256_MISMATCH",
    )

    return {
        "analysis_branch": branch,
        "analysis_head": head,
        "static_freeze_commit":
            STATIC_FREEZE_COMMIT,
        "static_report_rel":
            STATIC_REPORT_REL,
        "static_report_sha256":
            STATIC_REPORT_SHA256,
    }


def _parse_checksums(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
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
            name not in out,
            f"DUPLICATE_CHECKSUM:{name}",
        )
        out[name] = digest
    return out


def load_cpu_bundle(path: Path) -> dict[str, bytes]:
    require(
        path.is_file(),
        f"CPU_ZIP_MISSING:{path}",
    )
    require(
        sha256_file(path)
        == CPU_REFERENCE_ZIP_SHA256,
        "CPU_REFERENCE_ZIP_SHA256_MISMATCH",
    )

    with zipfile.ZipFile(path, "r") as archive:
        regular = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
        ]
        regular_set = set(regular)

        roots = []
        for name in regular:
            p = PurePosixPath(name)
            if p.name != "manifest.json":
                continue
            parent = p.parent
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
        isinstance(manifest, dict)
        and isinstance(summary, dict),
        "BUNDLE_JSON_SCHEMA",
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
        manifest["runtime_git_head"]
        == CPU_REFERENCE_HEAD,
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

    pair_ids = [
        str(row["source_pair_id"])
        for row in items
    ]
    require(
        len(set(pair_ids)) == SOURCE_PAIR_COUNT,
        "CPU_PAIR_UNIQUENESS",
    )


def finite_float(value: Any, label: str) -> float:
    out = float(value)
    require(
        math.isfinite(out),
        f"NONFINITE:{label}",
    )
    return out


def derive_features(
    row: Mapping[str, Any],
) -> dict[str, float]:
    target_a = finite_float(row["target_A"], "target_A")
    target_b = finite_float(row["target_B"], "target_B")
    target_c = finite_float(row["target_C"], "target_C")
    reference_a = finite_float(row["reference_A"], "reference_A")
    reference_b = finite_float(row["reference_B"], "reference_B")
    reference_c = finite_float(row["reference_C"], "reference_C")
    baseline_delta = finite_float(
        row["delta_baseline"],
        "delta_baseline",
    )

    require(
        target_a > 0.0
        and target_b > 0.0
        and reference_a > 0.0
        and reference_b > 0.0,
        "NONPOSITIVE_MAGNITUDE",
    )

    log_a = math.log(reference_a / target_a)
    log_b = math.log(reference_b / target_b)
    shift = reference_c - target_c

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
        "magnitude_log_ratio_A": log_a,
        "magnitude_log_ratio_B": log_b,
        "magnitude_shift_norm":
            math.hypot(log_a, log_b),
    }


def response_columns(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    return {
        name: [
            finite_float(
                row[name],
                f"{name}:{i}",
            )
            for i, row in enumerate(items)
        ]
        for name in RESPONSES
    }


def feature_columns(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    rows = [
        derive_features(row)
        for row in items
    ]
    return {
        name: [
            row[name]
            for row in rows
        ]
        for name in FEATURES
    }


def contribution_mass(
    values: Sequence[float],
) -> dict[str, Any]:
    positive_sum = sum(
        value
        for value in values
        if value > 0.0
    )
    negative_sum = sum(
        value
        for value in values
        if value < 0.0
    )
    absolute_total = (
        positive_sum + abs(negative_sum)
    )

    require(
        absolute_total > 0.0,
        "ZERO_CONTRIBUTION_MASS",
    )

    return {
        "positive_sum": positive_sum,
        "negative_sum": negative_sum,
        "net_sum": sum(values),
        "positive_absolute_mass_fraction":
            positive_sum / absolute_total,
        "negative_absolute_mass_fraction":
            abs(negative_sum) / absolute_total,
        "negative_to_positive_mass_ratio":
            (
                abs(negative_sum) / positive_sum
                if positive_sum > 0.0
                else None
            ),
    }


def removal_curve(
    values: Sequence[float],
    pair_ids: Sequence[str],
) -> list[dict[str, Any]]:
    ordered = sorted(
        range(len(values)),
        key=lambda i: (
            values[i],
            pair_ids[i],
        ),
    )

    base_mean = statistics.fmean(values)
    result = []

    for count in REMOVAL_COUNTS:
        removed = ordered[:count]
        removed_set = set(removed)
        kept = [
            value
            for i, value in enumerate(values)
            if i not in removed_set
        ]
        result.append(
            {
                "removed_worst_count": count,
                "remaining_n": len(kept),
                "remaining_mean":
                    statistics.fmean(kept),
                "mean_shift_from_full":
                    statistics.fmean(kept)
                    - base_mean,
                "removed_sum":
                    sum(values[i] for i in removed),
                "removed_pair_ids":
                    [pair_ids[i] for i in removed],
            }
        )

    return result


def symmetric_trimmed_means(
    values: Sequence[float],
) -> list[dict[str, Any]]:
    ordered = sorted(float(v) for v in values)
    result = []

    for fraction in TRIM_FRACTIONS:
        count = int(
            math.floor(
                len(ordered) * fraction
            )
        )
        require(
            count > 0
            and 2 * count < len(ordered),
            "TRIM_COUNT",
        )
        kept = ordered[
            count:
            len(ordered) - count
        ]
        result.append(
            {
                "trim_fraction_each_tail":
                    fraction,
                "trim_count_each_tail":
                    count,
                "remaining_n":
                    len(kept),
                "trimmed_mean":
                    statistics.fmean(kept),
            }
        )

    return result


def stable_quartile_labels(
    pair_ids: Sequence[str],
    values: Sequence[float],
) -> list[int]:
    require(
        len(pair_ids)
        == len(values)
        == SOURCE_PAIR_COUNT,
        "QUARTILE_SHAPE",
    )
    ordered = sorted(
        range(SOURCE_PAIR_COUNT),
        key=lambda i: (
            values[i],
            pair_ids[i],
        ),
    )
    require(
        SOURCE_PAIR_COUNT % 4 == 0,
        "QUARTILE_DIVISIBILITY",
    )
    width = SOURCE_PAIR_COUNT // 4
    labels = [0] * SOURCE_PAIR_COUNT

    for q in range(4):
        for index in ordered[
            q * width:
            (q + 1) * width
        ]:
            labels[index] = q + 1

    require(
        set(labels) == {1, 2, 3, 4},
        "QUARTILE_LABELS",
    )
    return labels


def tail_overlap(
    *,
    response: Sequence[float],
    pair_ids: Sequence[str],
    features: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    ordered = sorted(
        range(SOURCE_PAIR_COUNT),
        key=lambda i: (
            response[i],
            pair_ids[i],
        ),
    )

    quartile_labels = {
        name: stable_quartile_labels(
            pair_ids,
            features[name],
        )
        for name in QUARTILE_FEATURES
    }

    result: dict[str, Any] = {}

    for tail_count in TAIL_COUNTS:
        tail = ordered[:tail_count]
        entry = {
            "tail_count": tail_count,
            "tail_fraction":
                tail_count / SOURCE_PAIR_COUNT,
            "response_min":
                min(response[i] for i in tail),
            "response_max":
                max(response[i] for i in tail),
            "response_mean":
                statistics.fmean(
                    response[i] for i in tail
                ),
            "pair_ids":
                [pair_ids[i] for i in tail],
            "quartile_enrichment": {},
        }

        for feature_name in QUARTILE_FEATURES:
            labels = quartile_labels[
                feature_name
            ]
            counts = {
                f"Q{q}":
                    sum(
                        labels[i] == q
                        for i in tail
                    )
                for q in range(1, 5)
            }
            entry[
                "quartile_enrichment"
            ][feature_name] = {
                "counts": counts,
                "fractions": {
                    key:
                        value / tail_count
                    for key, value
                    in counts.items()
                },
                "Q4_enrichment_ratio_vs_population":
                    (
                        counts["Q4"]
                        / tail_count
                    )
                    / 0.25,
            }

        result[
            f"bottom_{tail_count}"
        ] = entry

    return result


def group_feature_summary(
    indices: Sequence[int],
    features: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    require(
        len(indices) > 0,
        "EMPTY_GROUP",
    )
    return {
        name: {
            "mean": statistics.fmean(
                features[name][i]
                for i in indices
            ),
            "median": statistics.median(
                features[name][i]
                for i in indices
            ),
        }
        for name in FEATURES
    }


def tail_geometry_contrast(
    *,
    response: Sequence[float],
    pair_ids: Sequence[str],
    features: Mapping[str, Sequence[float]],
    count: int = 30,
) -> dict[str, Any]:
    ordered = sorted(
        range(SOURCE_PAIR_COUNT),
        key=lambda i: (
            response[i],
            pair_ids[i],
        ),
    )
    bottom = ordered[:count]
    top = ordered[-count:]
    middle = ordered[count:-count]

    population_sd = {
        name: statistics.stdev(
            features[name]
        )
        for name in FEATURES
    }

    bottom_summary = group_feature_summary(
        bottom,
        features,
    )
    top_summary = group_feature_summary(
        top,
        features,
    )
    middle_summary = group_feature_summary(
        middle,
        features,
    )

    standardized = {}
    for name in FEATURES:
        sd = population_sd[name]
        require(
            sd > 0.0,
            f"ZERO_FEATURE_SD:{name}",
        )
        standardized[name] = {
            "bottom_minus_population_sd_units":
                (
                    bottom_summary[name]["mean"]
                    - statistics.fmean(
                        features[name]
                    )
                )
                / sd,
            "bottom_minus_top_sd_units":
                (
                    bottom_summary[name]["mean"]
                    - top_summary[name]["mean"]
                )
                / sd,
        }

    return {
        "count_per_tail": count,
        "bottom_pair_ids":
            [pair_ids[i] for i in bottom],
        "top_pair_ids":
            [pair_ids[i] for i in top],
        "bottom_feature_summary":
            bottom_summary,
        "middle_feature_summary":
            middle_summary,
        "top_feature_summary":
            top_summary,
        "standardized_feature_contrasts":
            standardized,
    }


def specificity_tail_decomposition(
    *,
    pair_ids: Sequence[str],
    responses: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    specificity = responses[
        "ALIGNMENT_SPECIFICITY"
    ]
    ordered = sorted(
        range(SOURCE_PAIR_COUNT),
        key=lambda i: (
            specificity[i],
            pair_ids[i],
        ),
    )

    result = {}

    for count in TAIL_COUNTS:
        bottom = ordered[:count]
        mean_align = statistics.fmean(
            responses["R_ALIGN"][i]
            for i in bottom
        )
        mean_mag = statistics.fmean(
            responses["R_MAG"][i]
            for i in bottom
        )
        mean_spec = statistics.fmean(
            specificity[i]
            for i in bottom
        )

        require(
            abs(
                mean_spec
                - (mean_align - mean_mag)
            )
            <= 1e-15,
            "SPECIFICITY_IDENTITY",
        )

        categories = {
            "R_ALIGN_negative": sum(
                responses["R_ALIGN"][i] < 0.0
                for i in bottom
            ),
            "R_MAG_positive": sum(
                responses["R_MAG"][i] > 0.0
                for i in bottom
            ),
            "both_align_negative_and_mag_positive":
                sum(
                    responses["R_ALIGN"][i] < 0.0
                    and responses["R_MAG"][i] > 0.0
                    for i in bottom
                ),
        }

        result[f"bottom_{count}"] = {
            "n": count,
            "mean_R_ALIGN": mean_align,
            "mean_R_MAG": mean_mag,
            "mean_specificity": mean_spec,
            "category_counts": categories,
            "pair_ids":
                [pair_ids[i] for i in bottom],
        }

    return result


def build_report(
    *,
    provenance: Mapping[str, str],
    cpu_zip: Path,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    pair_ids = [
        str(row["source_pair_id"])
        for row in items
    ]
    responses = response_columns(items)
    features = feature_columns(items)

    frozen = {
        str(row["id"]): row
        for row in summary[
            "hypothesis_family"
        ]
    }
    require(
        abs(
            statistics.fmean(
                responses["R_ALIGN"]
            )
            - float(
                frozen[
                    "H1_R_ALIGN_GT_ZERO"
                ]["mean"]
            )
        )
        <= 1e-15,
        "R_ALIGN_MEAN_REPRO",
    )
    require(
        abs(
            statistics.fmean(
                responses[
                    "ALIGNMENT_SPECIFICITY"
                ]
            )
            - float(
                frozen[
                    "H2_ALIGNMENT_SPECIFICITY_GT_ZERO"
                ]["mean"]
            )
        )
        <= 1e-15,
        "SPECIFICITY_MEAN_REPRO",
    )

    response_analysis = {}
    for name in RESPONSES:
        values = responses[name]
        response_analysis[name] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "positive_count":
                sum(v > 0.0 for v in values),
            "negative_count":
                sum(v < 0.0 for v in values),
            "contribution_mass":
                contribution_mass(values),
            "remove_worst_curve":
                removal_curve(
                    values,
                    pair_ids,
                ),
            "symmetric_trimmed_means":
                symmetric_trimmed_means(
                    values
                ),
            "bottom_tail_overlap":
                tail_overlap(
                    response=values,
                    pair_ids=pair_ids,
                    features=features,
                ),
            "bottom30_vs_top30_geometry":
                tail_geometry_contrast(
                    response=values,
                    pair_ids=pair_ids,
                    features=features,
                    count=30,
                ),
        }

    return {
        "schema_version": SCHEMA,
        "analysis_phase":
            "POSTHOC_STATIC_DIAGNOSTIC",
        "analysis_scope":
            "NEGATIVE_TAIL_INFLUENCE_LOCALIZATION",
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
        "fixed_diagnostic_parameters": {
            "remove_worst_counts":
                list(REMOVAL_COUNTS),
            "bottom_tail_counts":
                list(TAIL_COUNTS),
            "symmetric_trim_fractions":
                list(TRIM_FRACTIONS),
            "quartile_features":
                list(QUARTILE_FEATURES),
            "quartile_size": 75,
            "quartile_tie_break":
                "source_pair_id lexical order",
        },
        "response_analysis":
            response_analysis,
        "specificity_negative_tail_decomposition":
            specificity_tail_decomposition(
                pair_ids=pair_ids,
                responses=responses,
            ),
        "claim_boundary": [
            "This is descriptive/post-hoc influence localization and does not reopen the closed bridge.",
            "No new hypothesis test, p-value, layer scan, endpoint scan, offset scan, channel scan, checkpoint scan, or intervention sweep is executed.",
            "Tail removal is an influence diagnostic, not a revised estimator for the closed hypothesis.",
            "Any localized regime is a candidate for a separately designed future study, not evidence of transport in the closed bridge.",
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
    cpu_zip = Path(args.cpu_zip)
    output = Path(args.output)

    require(
        not output.exists(),
        f"OUTPUT_COLLISION:{output}",
    )

    provenance = authenticate_repo()
    files = load_cpu_bundle(cpu_zip)
    manifest, items, summary = decode_bundle(files)
    validate_cpu_reference(
        manifest,
        items,
        summary,
    )

    report = build_report(
        provenance=provenance,
        cpu_zip=cpu_zip,
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

    align = report[
        "response_analysis"
    ]["R_ALIGN"]
    spec = report[
        "response_analysis"
    ]["ALIGNMENT_SPECIFICITY"]

    print(
        "RESULT = PASS_TRANSPORT_FAILURE_TAIL_INFLUENCE"
    )
    print(
        "R_ALIGN_MEAN =",
        align["mean"],
    )
    print(
        "R_ALIGN_NEGATIVE_MASS_FRACTION =",
        align[
            "contribution_mass"
        ][
            "negative_absolute_mass_fraction"
        ],
    )
    print(
        "R_ALIGN_MEAN_AFTER_WORST10_REMOVED =",
        next(
            row["remaining_mean"]
            for row in align[
                "remove_worst_curve"
            ]
            if row[
                "removed_worst_count"
            ] == 10
        ),
    )
    print(
        "R_ALIGN_MEAN_AFTER_WORST30_REMOVED =",
        next(
            row["remaining_mean"]
            for row in align[
                "remove_worst_curve"
            ]
            if row[
                "removed_worst_count"
            ] == 30
        ),
    )
    print(
        "R_ALIGN_BOTTOM30_ALIGNMENT_SHIFT_ABS_Q4_ENRICHMENT =",
        align[
            "bottom_tail_overlap"
        ]["bottom_30"][
            "quartile_enrichment"
        ]["alignment_shift_abs"][
            "Q4_enrichment_ratio_vs_population"
        ],
    )
    print(
        "SPECIFICITY_NEGATIVE_MASS_FRACTION =",
        spec[
            "contribution_mass"
        ][
            "negative_absolute_mass_fraction"
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
