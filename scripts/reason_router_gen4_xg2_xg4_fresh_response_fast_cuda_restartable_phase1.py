from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import (
    reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_full_baseline
    as base,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"

BASELINE_ARTIFACT_FREEZE_COMMIT = (
    "40d21032b09560e246b16001c22837e46d5d3c70"
)
CORRECTION_FREEZE_COMMIT = (
    "bc8b0e2ca128eb81271761c9b3f5ed7b0a940455"
)
IMPLEMENTATION_SCOPE_COMMIT = (
    "917e505cd1156ee870e979300b1bc912f0aa43c2"
)

BASELINE_RUNNER_REL = (
    "scripts/"
    "reason_router_gen4_xg2_xg4_fresh_response_"
    "fast_cuda_full_baseline.py"
)
BASELINE_RUNNER_BLOB = (
    "7452cd4d7941b36c5e794d699ba8e2e0495a6f60"
)

SOURCE_PAIR_COUNT = base.SOURCE_PAIR_COUNT
BASELINE_FORWARDS_PER_PAIR = base.BASELINE_FORWARDS_PER_PAIR
FULL_BASELINE_FORWARD_BUDGET = (
    base.FULL_BASELINE_FORWARD_BUDGET
)

ALIGNMENT_SHIFT_THRESHOLD = base.ALIGNMENT_SHIFT_THRESHOLD
MIN_GROUP_SIZE = base.MIN_GROUP_SIZE
REGIME_LARGE = base.REGIME_LARGE
REGIME_SMALL = base.REGIME_SMALL

FROZEN_REGIME_COUNTS = {
    "xg2": {
        REGIME_LARGE: 92,
        REGIME_SMALL: 208,
    },
    "xg4": {
        REGIME_LARGE: 57,
        REGIME_SMALL: 243,
    },
}

ITEM_SCHEMA = (
    "gen4-xg2-xg4-fresh-response-"
    "restartable-phase1-item-v1"
)
SUMMARY_SCHEMA = (
    "gen4-xg2-xg4-fresh-response-"
    "restartable-phase1-summary-v1"
)
MANIFEST_SCHEMA = (
    "gen4-xg2-xg4-fresh-response-"
    "restartable-phase1-manifest-v1"
)

RESULT_PASS = (
    "PASS_XG2_XG4_FRESH_RESPONSE_"
    "RESTARTABLE_PHASE1_FREEZE"
)

ITEM_FILE = "baseline_items.jsonl"
PLAN_FILE = "alignment_delta_h.pt"
SUMMARY_FILE = "regime_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

RESPONSE_FIELDS = frozenset(
    {
        "alignment_plus_path_efficiency",
        "alignment_minus_path_efficiency",
        "delta_alignment",
        "R_ALIGN",
    }
)


class RestartablePhase1Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RestartablePhase1Error(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1 << 20),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
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


def jsonl_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    return b"".join(
        canonical_json_bytes(dict(row))
        for row in rows
    )


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise RestartablePhase1Error(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def _git_is_ancestor(
    ancestor: str,
    descendant: str,
) -> bool:
    return (
        subprocess.call(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def authenticate_repo(
    expected_head: str,
) -> None:
    base.authenticate_repo(expected_head)

    branch = git("branch", "--show-current")
    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH_MISMATCH:{branch}",
    )

    for ancestor, label in (
        (
            BASELINE_ARTIFACT_FREEZE_COMMIT,
            "BASELINE_ARTIFACT_FREEZE",
        ),
        (
            CORRECTION_FREEZE_COMMIT,
            "CORRECTION_FREEZE",
        ),
        (
            IMPLEMENTATION_SCOPE_COMMIT,
            "IMPLEMENTATION_SCOPE",
        ),
    ):
        require(
            _git_is_ancestor(
                ancestor,
                expected_head,
            ),
            f"{label}_NOT_ANCESTOR",
        )

    observed_blob = git(
        "rev-parse",
        f"HEAD:{BASELINE_RUNNER_REL}",
    )
    require(
        observed_blob == BASELINE_RUNNER_BLOB,
        (
            "FROZEN_BASELINE_RUNNER_DRIFT:"
            f"{observed_blob}"
        ),
    )


def _expected_pairs(
    family: str,
) -> tuple[str, ...]:
    require(
        family in FROZEN_REGIME_COUNTS,
        f"UNSUPPORTED_FAMILY:{family}",
    )
    return tuple(
        f"{family}_fact_{index:03d}"
        for index in range(301, 601)
    )


def _cells() -> dict[str, str]:
    core = base.prevalence_eq.core
    return {
        "tp": core.TARGET_PLUS_CELL,
        "tm": core.TARGET_MINUS_CELL,
        "rp": core.REFERENCE_PLUS_CELL,
        "rm": core.REFERENCE_MINUS_CELL,
    }


def _anchors_for_pair(
    pair: str,
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
) -> dict[str, int]:
    core = base.prevalence_eq.core
    cells = _cells()

    return {
        role: int(
            events[
                (
                    pair,
                    cell,
                    core.ANCHOR_NAME,
                )
            ]["absolute_anchor_token_index"]
        )
        for role, cell in cells.items()
    }


def _run_restartable_baseline_pair(
    family: str,
    pair: str,
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    budget: Any,
) -> tuple[dict[str, Any], torch.Tensor]:
    core = base.prevalence_eq.core
    parent = base.prevalence_eq.parent

    cells = _cells()
    anchors = _anchors_for_pair(
        pair,
        events,
    )

    baseline: dict[str, Any] = {}

    for role in (
        "tp",
        "tm",
        "rp",
        "rm",
    ):
        baseline[role] = parent.capture_branch(
            model,
            runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            input_ids=base.prevalence_eq._input_row(
                encoded,
                row_index,
                pair,
                cells[role],
            ),
            anchor=anchors[role],
            budget=budget,
            capture_states=True,
        )

    target_geometry = core.reconstruct_pair_geometry(
        baseline["tp"]["geometry_branch"],
        baseline["tm"]["geometry_branch"],
        gamma=runtime_ctx["gamma"],
        w_hidden=runtime_ctx["w_hidden"],
        strong_mask=runtime_ctx["strong_mask"],
    )

    reference_geometry = (
        core.reconstruct_pair_geometry(
            baseline["rp"]["geometry_branch"],
            baseline["rm"]["geometry_branch"],
            gamma=runtime_ctx["gamma"],
            w_hidden=runtime_ctx["w_hidden"],
            strong_mask=runtime_ctx["strong_mask"],
        )
    )

    alignment_shift_abs = abs(
        float(reference_geometry["C"])
        - float(target_geometry["C"])
    )
    require(
        math.isfinite(alignment_shift_abs),
        "NONFINITE_ALIGNMENT_SHIFT",
    )

    alignment_norm_delta, align_core = (
        core.alignment_delta(
            target_geometry["x"],
            target_geometry["y"],
            reference_geometry["C"],
        )
    )

    alignment_delta_h = (
        target_geometry["d"]
        * alignment_norm_delta
    )
    alignment_delta_h = (
        alignment_delta_h.detach()
        .cpu()
        .contiguous()
        .clone()
    )

    require(
        torch.is_tensor(alignment_delta_h),
        "ALIGNMENT_DELTA_NOT_TENSOR",
    )
    require(
        bool(
            torch.isfinite(
                alignment_delta_h
            ).all().item()
        ),
        "NONFINITE_ALIGNMENT_DELTA",
    )

    baseline_plus = parent.path_efficiency(
        baseline["tp"]
    )
    baseline_minus = parent.path_efficiency(
        baseline["tm"]
    )

    delta_baseline = (
        float(baseline_plus)
        - float(baseline_minus)
    )

    require(
        all(
            math.isfinite(value)
            for value in (
                float(baseline_plus),
                float(baseline_minus),
                float(delta_baseline),
            )
        ),
        "NONFINITE_BASELINE_PE",
    )

    item = {
        "schema_version": ITEM_SCHEMA,
        "family_key": family,
        "source_pair_id": pair,
        "source_block": core.SOURCE_BLOCK,
        "target_residual_layer": (
            core.TARGET_RESIDUAL_LAYER
        ),
        "relative_coordinate": (
            core.TARGET_OFFSET
        ),
        "target_plus_cell": cells["tp"],
        "target_minus_cell": cells["tm"],
        "reference_plus_cell": cells["rp"],
        "reference_minus_cell": cells["rm"],
        "anchor_name": core.ANCHOR_NAME,
        "target_plus_anchor": anchors["tp"],
        "target_minus_anchor": anchors["tm"],
        "reference_plus_anchor": anchors["rp"],
        "reference_minus_anchor": anchors["rm"],
        "target_plus_geometry_token": (
            anchors["tp"]
            + core.TARGET_OFFSET
        ),
        "target_minus_geometry_token": (
            anchors["tm"]
            + core.TARGET_OFFSET
        ),
        "reference_plus_geometry_token": (
            anchors["rp"]
            + core.TARGET_OFFSET
        ),
        "reference_minus_geometry_token": (
            anchors["rm"]
            + core.TARGET_OFFSET
        ),
        "target_A": float(
            target_geometry["A"]
        ),
        "target_B": float(
            target_geometry["B"]
        ),
        "target_C": float(
            target_geometry["C"]
        ),
        "reference_A": float(
            reference_geometry["A"]
        ),
        "reference_B": float(
            reference_geometry["B"]
        ),
        "reference_C": float(
            reference_geometry["C"]
        ),
        "alignment_shift_abs": (
            alignment_shift_abs
        ),
        "threshold": (
            ALIGNMENT_SHIFT_THRESHOLD
        ),
        "regime": base.classify_regime(
            alignment_shift_abs
        ),
        "classification_from_baseline_only": True,
        "baseline_plus_path_efficiency": (
            float(baseline_plus)
        ),
        "baseline_minus_path_efficiency": (
            float(baseline_minus)
        ),
        "delta_baseline": (
            float(delta_baseline)
        ),
        "alignment_delta_h_dtype": (
            str(alignment_delta_h.dtype)
            .removeprefix("torch.")
        ),
        "alignment_delta_h_shape": list(
            alignment_delta_h.shape
        ),
        "alignment_realized_A": float(
            align_core["realized_A"]
        ),
        "alignment_realized_B": float(
            align_core["realized_B"]
        ),
        "alignment_target_cosine": float(
            align_core["target_C"]
        ),
        "alignment_realized_cosine": float(
            align_core["realized_C"]
        ),
        "alignment_cosine_abs_residual": abs(
            float(align_core["realized_C"])
            - float(align_core["target_C"])
        ),
        "alignment_A_preservation_abs_residual": (
            float(
                align_core[
                    "A_preservation_abs_residual"
                ]
            )
        ),
        "alignment_B_preservation_abs_residual": (
            float(
                align_core[
                    "B_preservation_abs_residual"
                ]
            )
        ),
    }

    require(
        not (
            RESPONSE_FIELDS
            & set(item)
        ),
        "BASELINE_RESPONSE_FIELD_LEAK",
    )

    require(
        item["delta_baseline"]
        == (
            item[
                "baseline_plus_path_efficiency"
            ]
            - item[
                "baseline_minus_path_efficiency"
            ]
        ),
        "DELTA_BASELINE_IDENTITY",
    )

    return item, alignment_delta_h


def _validate_frozen_regime_counts(
    family: str,
    items: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    require(
        family in FROZEN_REGIME_COUNTS,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    counts = Counter(
        str(row["regime"])
        for row in items
    )

    observed = {
        REGIME_LARGE: int(
            counts[REGIME_LARGE]
        ),
        REGIME_SMALL: int(
            counts[REGIME_SMALL]
        ),
    }

    require(
        observed
        == FROZEN_REGIME_COUNTS[family],
        (
            "FROZEN_REGIME_COUNT_MISMATCH:"
            f"{family}:{observed}"
        ),
    )

    return observed


def _validate_item_rows(
    family: str,
    items: Sequence[Mapping[str, Any]],
) -> None:
    require(
        len(items) == SOURCE_PAIR_COUNT,
        "ITEM_COUNT",
    )

    expected_pairs = _expected_pairs(family)

    for index, (
        expected_pair,
        raw,
    ) in enumerate(
        zip(
            expected_pairs,
            items,
            strict=True,
        )
    ):
        row = dict(raw)

        require(
            row.get("schema_version")
            == ITEM_SCHEMA,
            f"ITEM_SCHEMA:{index}",
        )
        require(
            row.get("family_key")
            == family,
            f"ITEM_FAMILY:{index}",
        )
        require(
            row.get("source_pair_id")
            == expected_pair,
            f"PAIR_ORDER:{index}",
        )
        require(
            row.get(
                "alignment_plan_index"
            )
            == index,
            f"PLAN_INDEX:{index}",
        )
        require(
            float(row["threshold"])
            == ALIGNMENT_SHIFT_THRESHOLD,
            f"THRESHOLD:{index}",
        )
        require(
            not (
                RESPONSE_FIELDS
                & set(row)
            ),
            (
                "RESPONSE_FIELD_LEAK:"
                f"{expected_pair}"
            ),
        )

        plus = float(
            row[
                "baseline_plus_path_efficiency"
            ]
        )
        minus = float(
            row[
                "baseline_minus_path_efficiency"
            ]
        )
        delta = float(
            row["delta_baseline"]
        )

        require(
            all(
                math.isfinite(value)
                for value in (
                    plus,
                    minus,
                    delta,
                )
            ),
            (
                "NONFINITE_BASELINE_PE:"
                f"{expected_pair}"
            ),
        )
        require(
            delta == plus - minus,
            (
                "DELTA_BASELINE_IDENTITY:"
                f"{expected_pair}"
            ),
        )

        dtype = row.get(
            "alignment_delta_h_dtype"
        )
        shape = row.get(
            "alignment_delta_h_shape"
        )

        require(
            isinstance(dtype, str)
            and bool(dtype),
            (
                "PLAN_DTYPE:"
                f"{expected_pair}"
            ),
        )
        require(
            isinstance(shape, list)
            and len(shape) >= 1
            and all(
                type(value) is int
                and value > 0
                for value in shape
            ),
            (
                "PLAN_SHAPE:"
                f"{expected_pair}"
            ),
        )

    support = base.summarize_regimes(
        items
    )
    require(
        support["support_gate_pass"]
        is True,
        "SUPPORT_GATE_NOT_PASS",
    )

    _validate_frozen_regime_counts(
        family,
        items,
    )


def _stack_plans(
    items: Sequence[Mapping[str, Any]],
    plans: Sequence[torch.Tensor],
) -> torch.Tensor:
    require(
        len(plans) == SOURCE_PAIR_COUNT,
        "PLAN_COUNT",
    )

    stacked = torch.stack(
        [
            plan.detach()
            .cpu()
            .contiguous()
            .clone()
            for plan in plans
        ],
        dim=0,
    )

    require(
        int(stacked.shape[0])
        == SOURCE_PAIR_COUNT,
        "STACKED_PLAN_COUNT",
    )
    require(
        bool(
            torch.isfinite(
                stacked
            ).all().item()
        ),
        "NONFINITE_STACKED_PLAN",
    )

    for index, row in enumerate(items):
        plan = stacked[index]

        require(
            (
                str(plan.dtype)
                .removeprefix("torch.")
            )
            == row[
                "alignment_delta_h_dtype"
            ],
            f"PLAN_DTYPE_MISMATCH:{index}",
        )
        require(
            list(plan.shape)
            == row[
                "alignment_delta_h_shape"
            ],
            f"PLAN_SHAPE_MISMATCH:{index}",
        )

    return stacked


def _write_outputs(
    output_dir: Path,
    *,
    family: str,
    items: Sequence[Mapping[str, Any]],
    plans: Sequence[torch.Tensor],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    _validate_item_rows(
        family,
        items,
    )

    stacked = _stack_plans(
        items,
        plans,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    payloads = {
        ITEM_FILE: jsonl_bytes(items),
        SUMMARY_FILE: canonical_json_bytes(
            summary
        ),
    }

    hashes: dict[str, str] = {}

    for name, raw in payloads.items():
        path = output_dir / name
        path.write_bytes(raw)
        hashes[name] = sha256_bytes(raw)

    torch.save(
        stacked,
        output_dir / PLAN_FILE,
    )
    hashes[PLAN_FILE] = sha256_file(
        output_dir / PLAN_FILE
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "files": {
            name: {
                "sha256": digest,
                "bytes": int(
                    (
                        output_dir
                        / name
                    ).stat().st_size
                ),
            }
            for name, digest in sorted(
                hashes.items()
            )
        },
    }

    manifest_raw = canonical_json_bytes(
        manifest
    )
    (
        output_dir
        / MANIFEST_FILE
    ).write_bytes(
        manifest_raw
    )

    hashes[MANIFEST_FILE] = (
        sha256_bytes(
            manifest_raw
        )
    )

    checksum_raw = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(
            hashes.items()
        )
    ).encode("utf-8")

    (
        output_dir
        / CHECKSUM_FILE
    ).write_bytes(
        checksum_raw
    )

    return hashes


def _load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_no, line in enumerate(
        path.read_text(
            encoding="utf-8-sig"
        ).splitlines(),
        1,
    ):
        if not line.strip():
            continue

        row = json.loads(line)

        require(
            isinstance(row, dict),
            (
                "JSONL_OBJECT_REQUIRED:"
                f"{line_no}"
            ),
        )
        rows.append(row)

    return rows


def validate_restartable_artifact(
    output_dir: Path,
    family: str,
) -> dict[str, Any]:
    require(
        family in FROZEN_REGIME_COUNTS,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    manifest_path = (
        output_dir
        / MANIFEST_FILE
    )
    checksum_path = (
        output_dir
        / CHECKSUM_FILE
    )

    require(
        manifest_path.is_file(),
        "MANIFEST_MISSING",
    )
    require(
        checksum_path.is_file(),
        "CHECKSUM_MISSING",
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8-sig"
        )
    )

    require(
        manifest.get("schema_version")
        == MANIFEST_SCHEMA,
        "MANIFEST_SCHEMA",
    )

    files = manifest.get("files")
    require(
        isinstance(files, dict),
        "MANIFEST_FILES",
    )

    require(
        set(files)
        == {
            ITEM_FILE,
            PLAN_FILE,
            SUMMARY_FILE,
        },
        "MANIFEST_FILE_SET",
    )

    observed_hashes: dict[str, str] = {}

    for name in (
        ITEM_FILE,
        PLAN_FILE,
        SUMMARY_FILE,
    ):
        path = output_dir / name
        require(
            path.is_file(),
            f"ARTIFACT_MISSING:{name}",
        )

        observed_sha = sha256_file(path)
        observed_bytes = int(
            path.stat().st_size
        )

        expected = files[name]

        require(
            observed_sha
            == expected["sha256"],
            f"ARTIFACT_SHA256:{name}",
        )
        require(
            observed_bytes
            == int(expected["bytes"]),
            f"ARTIFACT_BYTES:{name}",
        )

        observed_hashes[name] = (
            observed_sha
        )

    observed_hashes[MANIFEST_FILE] = (
        sha256_file(manifest_path)
    )

    checksum_rows = {}

    for line in checksum_path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if not line.strip():
            continue

        digest, name = line.split(
            "  ",
            1,
        )
        require(
            name not in checksum_rows,
            f"CHECKSUM_DUPLICATE:{name}",
        )
        checksum_rows[name] = digest

    require(
        checksum_rows
        == {
            name: digest
            for name, digest in sorted(
                observed_hashes.items()
            )
        },
        "CHECKSUM_CONTENT",
    )

    items = _load_jsonl(
        output_dir / ITEM_FILE
    )

    _validate_item_rows(
        family,
        items,
    )

    plans = torch.load(
        output_dir / PLAN_FILE,
        map_location="cpu",
        weights_only=True,
    )

    require(
        torch.is_tensor(plans),
        "PLAN_ARTIFACT_NOT_TENSOR",
    )
    require(
        int(plans.shape[0])
        == SOURCE_PAIR_COUNT,
        "PLAN_ARTIFACT_COUNT",
    )
    require(
        bool(
            torch.isfinite(
                plans
            ).all().item()
        ),
        "PLAN_ARTIFACT_NONFINITE",
    )

    for index, row in enumerate(items):
        plan = plans[index]

        require(
            (
                str(plan.dtype)
                .removeprefix("torch.")
            )
            == row[
                "alignment_delta_h_dtype"
            ],
            f"PLAN_LOAD_DTYPE:{index}",
        )
        require(
            list(plan.shape)
            == row[
                "alignment_delta_h_shape"
            ],
            f"PLAN_LOAD_SHAPE:{index}",
        )

    summary = json.loads(
        (
            output_dir
            / SUMMARY_FILE
        ).read_text(
            encoding="utf-8-sig"
        )
    )

    require(
        summary.get("schema_version")
        == SUMMARY_SCHEMA,
        "SUMMARY_SCHEMA",
    )
    require(
        summary.get("result")
        == RESULT_PASS,
        "SUMMARY_RESULT",
    )
    require(
        summary.get("family_key")
        == family,
        "SUMMARY_FAMILY",
    )
    require(
        summary.get("source_pair_count")
        == SOURCE_PAIR_COUNT,
        "SUMMARY_PAIR_COUNT",
    )
    require(
        float(summary["threshold"])
        == ALIGNMENT_SHIFT_THRESHOLD,
        "SUMMARY_THRESHOLD",
    )
    require(
        summary.get(
            "baseline_model_forward_count"
        )
        == FULL_BASELINE_FORWARD_BUDGET,
        "SUMMARY_BASELINE_BUDGET",
    )
    require(
        summary.get(
            "alignment_model_forward_count"
        )
        == 0,
        "SUMMARY_ALIGNMENT_BUDGET",
    )
    require(
        summary.get(
            "total_model_forward_count"
        )
        == FULL_BASELINE_FORWARD_BUDGET,
        "SUMMARY_TOTAL_BUDGET",
    )
    require(
        summary.get(
            "support_gate_pass"
        )
        is True,
        "SUMMARY_SUPPORT_GATE",
    )
    require(
        summary.get(
            "phase2_restartable"
        )
        is True,
        "SUMMARY_RESTARTABLE",
    )

    counts = _validate_frozen_regime_counts(
        family,
        items,
    )

    require(
        summary.get("n_LARGE")
        == counts[REGIME_LARGE],
        "SUMMARY_LARGE_COUNT",
    )
    require(
        summary.get("n_SMALL")
        == counts[REGIME_SMALL],
        "SUMMARY_SMALL_COUNT",
    )

    return {
        "summary": summary,
        "items": items,
        "alignment_delta_h": plans,
        "manifest": manifest,
    }


def run_restartable_phase1(
    *,
    family: str,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(
        family in FROZEN_REGIME_COUNTS,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    authenticate_repo(
        expected_head
    )

    gate_report = (
        base.validate_equivalence_artifact(
            family
        )
    )

    base.prevalence_eq.backend.runtime_gate()

    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    with (
        base.prevalence_eq.backend
        .parent_runtime_rebind()
    ):
        rows, encoded, event_rows = (
            base.fresh_eq.load_family_inputs(
                family,
                tokenizer_snapshot,
            )
        )

        pairs = base.fresh_eq._pair_order(
            family,
            rows,
        )

        require(
            tuple(pairs)
            == _expected_pairs(family),
            "PAIR_ORDER",
        )

        events = (
            base.prevalence_eq.parent
            .event_lookup(
                event_rows
            )
        )

        (
            base.prevalence_eq.parent
            .validate_transport_event_plan(
                pairs,
                events,
            )
        )

        row_index = (
            base.prevalence_eq.parent
            .build_row_index(
                rows
            )
        )

        trace_code, trace_line = (
            base.prevalence_eq.measurement
            ._resolve_and_validate_runtime_binding()
        )

        kernels = (
            base.prevalence_eq.kernel_compat
            .load_exact_fast_kernels()
        )

        with (
            base.prevalence_eq.kernel_compat
            .exact_transformers_kernel_loader(
                kernels
            )
        ) as constructor_kernel_calls:
            model, checkpoint_sha = (
                base.prevalence_eq.parent
                .load_representative_model_external(
                    model_snapshot=(
                        model_snapshot
                    ),
                    checkpoint_path=(
                        checkpoint_path
                    ),
                )
            )

            require(
                checkpoint_sha
                == (
                    base.prevalence_eq
                    .extraction
                    .REPRESENTATIVE_CHECKPOINT_SHA256
                ),
                "CHECKPOINT_IDENTITY",
            )

            runtime_ctx = (
                base.prevalence_eq
                .transport_runtime
                .validate_runtime_components(
                    model
                )
            )

        constructor_counts = Counter(
            constructor_kernel_calls
        )

        require(
            set(constructor_counts)
            == {
                "causal-conv1d",
                "mamba-ssm",
            },
            (
                "TRANSFORMERS_CONSTRUCTOR_"
                "KERNEL_NAMES:"
                f"{dict(constructor_counts)}"
            ),
        )
        require(
            constructor_counts[
                "causal-conv1d"
            ]
            > 0,
            (
                "TRANSFORMERS_CONSTRUCTOR_"
                "KERNEL_CALL_COUNT_ZERO"
            ),
        )
        require(
            constructor_counts[
                "causal-conv1d"
            ]
            == constructor_counts[
                "mamba-ssm"
            ],
            (
                "TRANSFORMERS_CONSTRUCTOR_"
                "KERNEL_CALL_COUNT_MISMATCH:"
                f"{dict(constructor_counts)}"
            ),
        )

        (
            base.prevalence_eq.kernel_compat
            .validate_transformers_kernel_bindings(
                kernels
            )
        )

        model.to(
            torch.device("cuda:0")
        )
        model.eval()

        require(
            all(
                parameter.device.type
                == "cuda"
                for parameter
                in model.mamba.parameters()
            ),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = (
            base.prevalence_eq.backend
            ._make_fast_capture(
                kernels
            )
        )

        original_capture = (
            base.prevalence_eq.parent
            .capture_branch
        )

        budget = (
            base.prevalence_eq.parent
            .ForwardBudget(
                FULL_BASELINE_FORWARD_BUDGET
            )
        )

        items: list[
            dict[str, Any]
        ] = []
        plans: list[
            torch.Tensor
        ] = []

        base.prevalence_eq.parent.capture_branch = (
            fast_capture
        )

        try:
            for index, pair in enumerate(
                pairs
            ):
                item, plan = (
                    _run_restartable_baseline_pair(
                        family,
                        pair,
                        model=model,
                        runtime_ctx=runtime_ctx,
                        trace_code=trace_code,
                        trace_line=trace_line,
                        encoded=encoded,
                        row_index=row_index,
                        events=events,
                        budget=budget,
                    )
                )

                item[
                    "alignment_plan_index"
                ] = index

                items.append(item)
                plans.append(plan)

            budget.assert_exact()
            torch.cuda.synchronize()

        finally:
            (
                base.prevalence_eq.parent
                .capture_branch
            ) = original_capture

    support = base.summarize_regimes(
        items
    )

    counts = (
        _validate_frozen_regime_counts(
            family,
            items,
        )
    )

    require(
        support["support_gate_pass"]
        is True,
        "SUPPORT_GATE_NOT_PASS",
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "family_key": family,
        "execution_head": expected_head,
        "baseline_artifact_freeze_commit": (
            BASELINE_ARTIFACT_FREEZE_COMMIT
        ),
        "correction_freeze_commit": (
            CORRECTION_FREEZE_COMMIT
        ),
        "implementation_scope_commit": (
            IMPLEMENTATION_SCOPE_COMMIT
        ),
        "design_freeze_commit": (
            base.DESIGN_FREEZE_COMMIT
        ),
        "equivalence_artifact_freeze_commit": (
            base.EQUIVALENCE_ARTIFACT_FREEZE_COMMIT
        ),
        "equivalence_execution_head": (
            base.EQUIVALENCE_EXECUTION_HEAD
        ),
        "equivalence_artifact_sha256": (
            base.EQUIVALENCE_ARTIFACTS[
                family
            ]["sha256"]
        ),
        "equivalence_result": (
            gate_report["result"]
        ),
        "representative_checkpoint_sha256": (
            checkpoint_sha
        ),
        "pair_id_first": pairs[0],
        "pair_id_last": pairs[-1],
        "source_pair_count": (
            SOURCE_PAIR_COUNT
        ),
        "threshold": (
            ALIGNMENT_SHIFT_THRESHOLD
        ),
        "threshold_reestimated": False,
        "n_LARGE": (
            counts[REGIME_LARGE]
        ),
        "n_SMALL": (
            counts[REGIME_SMALL]
        ),
        "p_LARGE": (
            counts[REGIME_LARGE]
            / SOURCE_PAIR_COUNT
        ),
        "p_SMALL": (
            counts[REGIME_SMALL]
            / SOURCE_PAIR_COUNT
        ),
        "minimum_group_size": (
            MIN_GROUP_SIZE
        ),
        "support_gate_pass": True,
        "prospective_regime_test": (
            base.REGIME_TEST_READY
        ),
        "baseline_model_forward_count": (
            FULL_BASELINE_FORWARD_BUDGET
        ),
        "alignment_model_forward_count": 0,
        "total_model_forward_count": (
            FULL_BASELINE_FORWARD_BUDGET
        ),
        "scientific_budget_forward_count": (
            FULL_BASELINE_FORWARD_BUDGET
        ),
        "alignment_plan_count": (
            SOURCE_PAIR_COUNT
        ),
        "alignment_plan_file": (
            PLAN_FILE
        ),
        "phase2_restartable": True,
        "baseline_only": True,
        "alignment_intervention_executed": False,
        "magnitude_intervention_executed": False,
        "response_fields_observed": False,
        "r_align_observed": False,
        "h1_test_executed": False,
        "h2_test_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion_scope": (
            "FRESH_BASELINE_REGIME_"
            "SUPPORT_ONLY"
        ),
    }

    _write_outputs(
        output_dir,
        family=family,
        items=items,
        plans=plans,
        summary=summary,
    )

    validated = (
        validate_restartable_artifact(
            output_dir,
            family,
        )
    )

    require(
        validated["summary"]["result"]
        == RESULT_PASS,
        "POSTWRITE_VALIDATION_RESULT",
    )

    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restartable fresh XG2/XG4 "
            "Phase-1 baseline freeze. "
            "Exactly 300 pairs x four "
            "baseline branches = 1200 "
            "scientific forwards per family. "
            "Persists baseline PE and the "
            "lossless alignment_delta_h plan. "
            "No alignment response, R_ALIGN, "
            "H1/H2, task head, training, "
            "or backward execution."
        )
    )

    parser.add_argument(
        "--family",
        choices=("xg2", "xg4"),
        required=True,
    )
    parser.add_argument(
        "--expected-head",
        required=True,
    )
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)

    summary = run_restartable_phase1(
        family=args.family,
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=(
            args.tokenizer_snapshot
        ),
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )

    print(
        "RESULT =",
        summary["result"],
    )
    print(
        "FAMILY =",
        summary["family_key"],
    )
    print(
        "N_LARGE =",
        summary["n_LARGE"],
    )
    print(
        "N_SMALL =",
        summary["n_SMALL"],
    )
    print(
        "BASELINE_MODEL_FORWARD_COUNT =",
        summary[
            "baseline_model_forward_count"
        ],
    )
    print(
        "ALIGNMENT_MODEL_FORWARD_COUNT =",
        summary[
            "alignment_model_forward_count"
        ],
    )
    print(
        "PHASE2_RESTARTABLE =",
        summary["phase2_restartable"],
    )


if __name__ == "__main__":
    main()
