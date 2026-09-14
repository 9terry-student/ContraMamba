#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


CPU_REFERENCE_HEAD = "8496ece911e0d461f0abbdf1a0fa619f8a2f22ab"
CPU_REFERENCE_ZIP_SHA256 = (
    "25a9e6a862f7c1ad7c272d85cb5000ba8542cca2c8976b785021e5e4159ebcaf"
)
CUDA_BRANCH = "gen4-k-fast-cuda-equivalence"

SOURCE_PAIR_COUNT = 300
MODEL_FORWARD_COUNT = 2400

STATE_ATOL = 1e-4
STATE_RTOL = 1e-4
GEOMETRY_ATOL = 1e-4
GEOMETRY_RTOL = 1e-4
PE_ATOL = 1e-4

VECTOR_TOL = 5e-12
RUNTIME_CAST_TOL = 5e-6
MIDPOINT_TOL = 5e-6
CPU_BASELINE_TOL = 1e-12
CUDA_BASELINE_TOL = PE_ATOL

NEGATIVE_OUTCOME = (
    "DIRECTIONAL_ALIGNMENT_CAUSAL_TRANSPORT_NOT_ESTABLISHED"
)

FILES = frozenset(
    {
        "manifest.json",
        "item_metrics.jsonl",
        "summary.json",
        "SHA256SUMS.txt",
    }
)

EXACT_FIELDS = (
    "schema_version",
    "source_pair_id",
    "source_block",
    "target_residual_layer",
    "intervention_layer",
    "relative_coordinate",
    "target_plus_cell",
    "target_minus_cell",
    "reference_plus_cell",
    "reference_minus_cell",
    "anchor_name",
    "target_plus_anchor",
    "target_minus_anchor",
    "reference_plus_anchor",
    "reference_minus_anchor",
    "target_plus_intervention_token",
    "target_minus_intervention_token",
    "reference_plus_geometry_token",
    "reference_minus_geometry_token",
    "frozen_plus_path_efficiency",
    "frozen_minus_path_efficiency",
)

GEOMETRY_FIELDS = (
    "target_A",
    "target_B",
    "target_C",
    "reference_A",
    "reference_B",
    "reference_C",
    "alignment_realized_A",
    "alignment_realized_B",
    "alignment_target_cosine",
    "alignment_realized_cosine",
    "magnitude_target_A",
    "magnitude_target_B",
    "magnitude_realized_A",
    "magnitude_realized_B",
    "magnitude_baseline_cosine",
    "magnitude_realized_cosine",
)

PE_FIELDS = (
    "baseline_plus_path_efficiency",
    "baseline_minus_path_efficiency",
    "alignment_plus_path_efficiency",
    "alignment_minus_path_efficiency",
    "magnitude_plus_path_efficiency",
    "magnitude_minus_path_efficiency",
    "delta_baseline",
    "delta_alignment",
    "delta_magnitude",
    "R_ALIGN",
    "R_MAG",
    "ALIGNMENT_SPECIFICITY",
)

BASELINE_RESIDUAL_FIELDS = (
    "baseline_plus_reproduction_abs_residual",
    "baseline_minus_reproduction_abs_residual",
)

VECTOR_RESIDUAL_FIELDS = (
    "alignment_cosine_abs_residual",
    "alignment_A_preservation_abs_residual",
    "alignment_B_preservation_abs_residual",
    "magnitude_cosine_abs_residual",
    "magnitude_A_target_abs_residual",
    "magnitude_B_target_abs_residual",
)

MIDPOINT_RESIDUAL_FIELDS = (
    "alignment_midpoint_max_abs_residual",
    "magnitude_midpoint_max_abs_residual",
)

CAST_RESIDUAL_FIELDS = (
    "alignment_pair_delta_max_abs_residual",
    "magnitude_pair_delta_max_abs_residual",
    "alignment_applied_correction_max_abs_residual",
    "magnitude_applied_correction_max_abs_residual",
)


class CompareError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise CompareError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1 << 20),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def _parse_checksums(text: str) -> dict[str, str]:
    result: dict[str, str] = {}

    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = raw.split("  ", 1)
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
            "DUPLICATE_CHECKSUM_NAME",
        )
        result[name] = digest

    return result


class Bundle:
    def __init__(
        self,
        *,
        files: Mapping[str, bytes],
        source_label: str,
    ):
        require(
            set(files) == FILES,
            f"BUNDLE_FILE_SET:{source_label}:{sorted(files)}",
        )
        self.files = dict(files)
        self.source_label = source_label
        self._verify_checksums()

    def _verify_checksums(self) -> None:
        expected = _parse_checksums(
            self.files[
                "SHA256SUMS.txt"
            ].decode("utf-8")
        )

        require(
            set(expected)
            == {
                "manifest.json",
                "item_metrics.jsonl",
                "summary.json",
            },
            f"CHECKSUM_FILE_SET:{self.source_label}",
        )

        for name, digest in expected.items():
            observed = hashlib.sha256(
                self.files[name]
            ).hexdigest()
            require(
                observed == digest,
                f"CHECKSUM_MISMATCH:{self.source_label}:{name}",
            )

    def json(self, name: str) -> dict[str, Any]:
        value = json.loads(
            self.files[name].decode("utf-8")
        )
        require(
            isinstance(value, dict),
            f"JSON_NOT_OBJECT:{self.source_label}:{name}",
        )
        return value

    def jsonl(
        self,
        name: str,
    ) -> list[dict[str, Any]]:
        rows = []
        for raw in self.files[name].decode(
            "utf-8"
        ).splitlines():
            if not raw.strip():
                continue
            value = json.loads(raw)
            require(
                isinstance(value, dict),
                f"JSONL_NOT_OBJECT:{self.source_label}:{name}",
            )
            rows.append(value)
        return rows


def _bundle_from_directory(path: Path) -> Bundle:
    candidates = []

    if path.is_dir():
        if all(
            (path / name).is_file()
            for name in FILES
        ):
            candidates.append(path)

        for child in path.iterdir():
            if (
                child.is_dir()
                and all(
                    (child / name).is_file()
                    for name in FILES
                )
            ):
                candidates.append(child)

    require(
        len(candidates) == 1,
        f"DIRECTORY_BUNDLE_ROOT:{path}:{len(candidates)}",
    )

    root = candidates[0]
    files = {
        name: (root / name).read_bytes()
        for name in FILES
    }
    return Bundle(
        files=files,
        source_label=str(path),
    )


def _bundle_from_zip(path: Path) -> Bundle:
    with zipfile.ZipFile(path, "r") as archive:
        regular = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
        ]

        manifest_names = [
            name
            for name in regular
            if PurePosixPath(name).name
            == "manifest.json"
        ]

        roots = []
        for manifest_name in manifest_names:
            parent = PurePosixPath(
                manifest_name
            ).parent
            names = {
                (
                    (parent / filename).as_posix()
                    if parent != PurePosixPath(".")
                    else filename
                )
                for filename in FILES
            }
            if names <= set(regular):
                roots.append(parent)

        require(
            len(roots) == 1,
            f"ZIP_BUNDLE_ROOT:{path}:{len(roots)}",
        )
        root = roots[0]

        selected: dict[str, bytes] = {}
        for filename in FILES:
            member = (
                (root / filename).as_posix()
                if root != PurePosixPath(".")
                else filename
            )
            selected[filename] = archive.read(
                member
            )

        root_regular = {
            PurePosixPath(name).name
            for name in regular
            if PurePosixPath(name).parent == root
        }
        require(
            root_regular == FILES,
            f"ZIP_ROOT_FILE_SET:{path}:{sorted(root_regular)}",
        )

    return Bundle(
        files=selected,
        source_label=str(path),
    )


def load_bundle(path: Path) -> Bundle:
    require(
        path.exists(),
        f"BUNDLE_NOT_FOUND:{path}",
    )
    if path.is_dir():
        return _bundle_from_directory(path)
    require(
        path.suffix.lower() == ".zip",
        f"UNSUPPORTED_BUNDLE_SOURCE:{path}",
    )
    return _bundle_from_zip(path)


def _finite(value: Any, label: str) -> float:
    result = float(value)
    require(
        math.isfinite(result),
        f"NONFINITE:{label}",
    )
    return result


def _close(
    observed: Any,
    reference: Any,
    *,
    atol: float,
    rtol: float = 0.0,
    label: str,
) -> float:
    a = _finite(observed, label + ":observed")
    b = _finite(reference, label + ":reference")
    diff = abs(a - b)
    limit = atol + rtol * abs(b)
    require(
        diff <= limit,
        f"NUMERIC_MISMATCH:{label}:{diff}:{limit}",
    )
    return diff


def _bound(
    value: Any,
    *,
    limit: float,
    label: str,
) -> None:
    number = _finite(value, label)
    require(
        0.0 <= number <= limit,
        f"RESIDUAL_BOUND:{label}:{number}:{limit}",
    )


def compare_item(
    cpu: Mapping[str, Any],
    cuda: Mapping[str, Any],
) -> dict[str, float]:
    require(
        set(cpu) == set(cuda),
        "ITEM_SCHEMA_MISMATCH",
    )

    pair = str(cpu["source_pair_id"])
    require(
        pair == str(cuda["source_pair_id"]),
        "PAIR_ID_MISMATCH",
    )

    for field in EXACT_FIELDS:
        require(
            cpu[field] == cuda[field],
            f"EXACT_ITEM_MISMATCH:{pair}:{field}",
        )

    max_geometry = 0.0
    for field in GEOMETRY_FIELDS:
        max_geometry = max(
            max_geometry,
            _close(
                cuda[field],
                cpu[field],
                atol=GEOMETRY_ATOL,
                rtol=GEOMETRY_RTOL,
                label=f"{pair}:{field}",
            ),
        )

    max_pe = 0.0
    for field in PE_FIELDS:
        max_pe = max(
            max_pe,
            _close(
                cuda[field],
                cpu[field],
                atol=PE_ATOL,
                label=f"{pair}:{field}",
            ),
        )

    for field in BASELINE_RESIDUAL_FIELDS:
        _bound(
            cpu[field],
            limit=CPU_BASELINE_TOL,
            label=f"cpu:{pair}:{field}",
        )
        _bound(
            cuda[field],
            limit=CUDA_BASELINE_TOL,
            label=f"cuda:{pair}:{field}",
        )

    for field in VECTOR_RESIDUAL_FIELDS:
        _bound(
            cpu[field],
            limit=VECTOR_TOL,
            label=f"cpu:{pair}:{field}",
        )
        _bound(
            cuda[field],
            limit=VECTOR_TOL,
            label=f"cuda:{pair}:{field}",
        )

    for field in MIDPOINT_RESIDUAL_FIELDS:
        _bound(
            cpu[field],
            limit=MIDPOINT_TOL,
            label=f"cpu:{pair}:{field}",
        )
        _bound(
            cuda[field],
            limit=MIDPOINT_TOL,
            label=f"cuda:{pair}:{field}",
        )

    for field in CAST_RESIDUAL_FIELDS:
        _bound(
            cpu[field],
            limit=RUNTIME_CAST_TOL,
            label=f"cpu:{pair}:{field}",
        )
        _bound(
            cuda[field],
            limit=RUNTIME_CAST_TOL,
            label=f"cuda:{pair}:{field}",
        )

    return {
        "max_geometry_abs_diff":
            max_geometry,
        "max_pe_abs_diff":
            max_pe,
    }


def _validate_cpu(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    require(
        manifest["mode"] == "full",
        "CPU_MODE",
    )
    require(
        manifest["runtime_git_head"]
        == CPU_REFERENCE_HEAD,
        "CPU_HEAD",
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
        manifest["training_executed"] is False
        and manifest["backward_executed"] is False
        and manifest["task_heads_executed"] is False
        and manifest["logits_read"] is False
        and manifest["raw_vectors_persisted"] is False,
        "CPU_SAFETY_FLAGS",
    )
    require(
        summary[
            "all_mandatory_manipulation_checks_pass"
        ]
        is True,
        "CPU_MANIPULATION",
    )
    require(
        summary["outcome"]
        == NEGATIVE_OUTCOME,
        "CPU_CANONICAL_OUTCOME",
    )


def _validate_cuda(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    expected_head: str,
) -> None:
    require(
        manifest["mode"]
        == "full_backend_equivalence",
        "CUDA_MODE",
    )
    require(
        manifest["runtime_branch"]
        == CUDA_BRANCH,
        "CUDA_BRANCH",
    )
    require(
        manifest["runtime_git_head"]
        == expected_head,
        "CUDA_HEAD",
    )
    require(
        manifest["cpu_reference_head"]
        == CPU_REFERENCE_HEAD,
        "CUDA_CPU_HEAD_BINDING",
    )
    require(
        manifest["cpu_reference_zip_sha256"]
        == CPU_REFERENCE_ZIP_SHA256,
        "CUDA_CPU_ZIP_BINDING",
    )
    require(
        int(manifest["source_pair_count"])
        == SOURCE_PAIR_COUNT,
        "CUDA_PAIR_COUNT",
    )
    require(
        int(manifest["model_forward_count"])
        == MODEL_FORWARD_COUNT,
        "CUDA_FORWARD_COUNT",
    )
    require(
        manifest["device"] == "cuda:0"
        and manifest["backend_equivalence_only"] is True
        and manifest["scientific_authority"] is False,
        "CUDA_BACKEND_SCOPE",
    )
    require(
        manifest["training_executed"] is False
        and manifest["backward_executed"] is False
        and manifest["task_heads_executed"] is False
        and manifest["logits_read"] is False
        and manifest["raw_vectors_persisted"] is False,
        "CUDA_SAFETY_FLAGS",
    )
    require(
        summary[
            "all_mandatory_manipulation_checks_pass"
        ]
        is True,
        "CUDA_MANIPULATION",
    )
    require(
        summary["backend_equivalence_only"] is True
        and summary["scientific_authority"] is False,
        "CUDA_SUMMARY_SCOPE",
    )


def compare_summaries(
    cpu: Mapping[str, Any],
    cuda: Mapping[str, Any],
) -> dict[str, float]:
    require(
        int(cpu["population_size"])
        == SOURCE_PAIR_COUNT
        and int(cuda["population_size"])
        == SOURCE_PAIR_COUNT,
        "SUMMARY_POPULATION",
    )

    baseline_diff = _close(
        cuda[
            "baseline_delta_name_path_efficiency_mean"
        ],
        cpu[
            "baseline_delta_name_path_efficiency_mean"
        ],
        atol=PE_ATOL,
        label="summary:baseline_mean",
    )

    cpu_family = cpu["hypothesis_family"]
    cuda_family = cuda["hypothesis_family"]
    require(
        isinstance(cpu_family, list)
        and isinstance(cuda_family, list)
        and len(cpu_family) == 2
        and len(cuda_family) == 2,
        "HYPOTHESIS_FAMILY_SHAPE",
    )

    max_hypothesis_mean_diff = 0.0

    for a, b in zip(
        cpu_family,
        cuda_family,
        strict=True,
    ):
        require(
            a["id"] == b["id"],
            "HYPOTHESIS_ID",
        )
        require(
            int(a["n"]) == SOURCE_PAIR_COUNT
            and int(b["n"]) == SOURCE_PAIR_COUNT,
            "HYPOTHESIS_N",
        )
        max_hypothesis_mean_diff = max(
            max_hypothesis_mean_diff,
            _close(
                b["mean"],
                a["mean"],
                atol=PE_ATOL,
                label=f"hypothesis:{a['id']}:mean",
            ),
        )
        require(
            bool(
                a[
                    "reject_holm_alpha_0_05"
                ]
            )
            is bool(
                b[
                    "reject_holm_alpha_0_05"
                ]
            ),
            f"HOLM_DECISION:{a['id']}",
        )

    require(
        cpu["outcome"] == cuda["outcome"],
        "OUTCOME_MISMATCH",
    )

    return {
        "baseline_mean_abs_diff":
            baseline_diff,
        "max_hypothesis_mean_abs_diff":
            max_hypothesis_mean_diff,
    }


def compare_bundles(
    *,
    cpu_path: Path,
    cuda_path: Path,
    expected_cuda_head: str,
    report_path: Path,
) -> dict[str, Any]:
    require(
        sha256_file(cpu_path)
        == CPU_REFERENCE_ZIP_SHA256,
        "CPU_REFERENCE_ZIP_SHA256",
    )

    cpu = load_bundle(cpu_path)
    cuda = load_bundle(cuda_path)

    cpu_manifest = cpu.json("manifest.json")
    cuda_manifest = cuda.json("manifest.json")
    cpu_summary = cpu.json("summary.json")
    cuda_summary = cuda.json("summary.json")

    _validate_cpu(
        cpu_manifest,
        cpu_summary,
    )
    _validate_cuda(
        cuda_manifest,
        cuda_summary,
        expected_cuda_head,
    )

    provenance_fields = (
        "checkpoint_sha256",
        "frozen_endpoint_sha256",
        "event_manifest_sha256",
        "mamba_source_sha256",
        "source_block",
        "target_residual_layer",
        "intervention_layer",
        "relative_coordinate",
        "target_pair",
        "reference_pair",
        "anchor_name",
        "strong_count",
        "weak_count",
        "equal_count",
        "strong_index_sha256",
    )

    for field in provenance_fields:
        require(
            cpu_manifest[field]
            == cuda_manifest[field],
            f"PROVENANCE_MISMATCH:{field}",
        )

    cpu_items = cpu.jsonl(
        "item_metrics.jsonl"
    )
    cuda_items = cuda.jsonl(
        "item_metrics.jsonl"
    )

    require(
        len(cpu_items)
        == SOURCE_PAIR_COUNT
        and len(cuda_items)
        == SOURCE_PAIR_COUNT,
        "ITEM_COUNT",
    )

    cpu_order = [
        str(row["source_pair_id"])
        for row in cpu_items
    ]
    cuda_order = [
        str(row["source_pair_id"])
        for row in cuda_items
    ]
    require(
        cpu_order == cuda_order,
        "PAIR_ORDER",
    )
    require(
        len(set(cpu_order))
        == SOURCE_PAIR_COUNT,
        "PAIR_UNIQUENESS",
    )

    max_geometry = 0.0
    max_pe = 0.0

    for cpu_item, cuda_item in zip(
        cpu_items,
        cuda_items,
        strict=True,
    ):
        result = compare_item(
            cpu_item,
            cuda_item,
        )
        max_geometry = max(
            max_geometry,
            result[
                "max_geometry_abs_diff"
            ],
        )
        max_pe = max(
            max_pe,
            result[
                "max_pe_abs_diff"
            ],
        )

    summary_comparison = (
        compare_summaries(
            cpu_summary,
            cuda_summary,
        )
    )

    require(
        cpu_summary["outcome"]
        == NEGATIVE_OUTCOME,
        "CPU_OUTCOME_CHANGED",
    )
    require(
        cuda_summary["outcome"]
        == NEGATIVE_OUTCOME,
        "CUDA_OUTCOME_CHANGED",
    )

    report = {
        "schema_version":
            "gen4-k-fast-cuda-full-equivalence-report-v1",
        "result":
            "PASS_FAST_CUDA_FULL_BACKEND_EQUIVALENCE",
        "cpu_reference_zip_sha256":
            CPU_REFERENCE_ZIP_SHA256,
        "cuda_bundle_sha256":
            sha256_file(cuda_path),
        "cpu_reference_head":
            CPU_REFERENCE_HEAD,
        "cuda_execution_head":
            expected_cuda_head,
        "source_pair_count":
            SOURCE_PAIR_COUNT,
        "cpu_model_forward_count":
            MODEL_FORWARD_COUNT,
        "cuda_model_forward_count":
            MODEL_FORWARD_COUNT,
        "geometry_atol":
            GEOMETRY_ATOL,
        "geometry_rtol":
            GEOMETRY_RTOL,
        "pe_atol":
            PE_ATOL,
        "max_item_geometry_abs_diff":
            max_geometry,
        "max_item_pe_abs_diff":
            max_pe,
        **summary_comparison,
        "cpu_outcome":
            cpu_summary["outcome"],
        "cuda_outcome":
            cuda_summary["outcome"],
        "holm_decisions_match":
            True,
        "backend_invariant_negative_result":
            True,
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
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

    return report


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-zip",
        required=True,
    )
    parser.add_argument(
        "--cuda-zip",
        required=True,
    )
    parser.add_argument(
        "--expected-cuda-head",
        required=True,
    )
    parser.add_argument(
        "--report",
        required=True,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)

    report = compare_bundles(
        cpu_path=Path(args.cpu_zip),
        cuda_path=Path(args.cuda_zip),
        expected_cuda_head=
            args.expected_cuda_head,
        report_path=Path(args.report),
    )

    print("RESULT =", report["result"])
    print(
        "SOURCE_PAIR_COUNT =",
        report["source_pair_count"],
    )
    print(
        "MAX_ITEM_GEOMETRY_ABS_DIFF =",
        report["max_item_geometry_abs_diff"],
    )
    print(
        "MAX_ITEM_PE_ABS_DIFF =",
        report["max_item_pe_abs_diff"],
    )
    print(
        "CPU_OUTCOME =",
        report["cpu_outcome"],
    )
    print(
        "CUDA_OUTCOME =",
        report["cuda_outcome"],
    )


if __name__ == "__main__":
    main()
