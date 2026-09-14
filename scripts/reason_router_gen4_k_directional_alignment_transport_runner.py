#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import (
    reason_router_gen4_k_directional_alignment_transport_core
    as core,
)
from scripts import (
    reason_router_gen4_k_directional_alignment_transport_runtime
    as transport_runtime,
)
from scripts import (
    reason_router_gen4_native_mamba_state_extraction
    as extraction,
)
from scripts import (
    reason_router_gen4_native_mamba_state_measurement
    as measurement,
)
from scripts import (
    reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis
    as frozen_stats,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-directional-alignment-transport"
IMPLEMENTATION_PARENT = (
    "ac4f682acaeea5a00eafdedcd5ff097b61d3a3c0"
)

RUNNER_REL = (
    "scripts/"
    "reason_router_gen4_k_directional_alignment_transport_runner.py"
)

SOURCE_PAIR_COUNT = 300
PREFLIGHT_PAIR_COUNT = 2
FORWARDS_PER_PAIR = 8
FULL_FORWARD_BUDGET = SOURCE_PAIR_COUNT * FORWARDS_PER_PAIR
PREFLIGHT_FORWARD_BUDGET = (
    PREFLIGHT_PAIR_COUNT * FORWARDS_PER_PAIR
)

BASELINE_REPRO_TOL = 1e-12
FAMILY_ALPHA = 0.05

ITEM_SCHEMA = (
    "gen4-k-directional-alignment-transport-item-v1"
)
SUMMARY_SCHEMA = (
    "gen4-k-directional-alignment-transport-summary-v1"
)
MANIFEST_SCHEMA = (
    "gen4-k-directional-alignment-transport-manifest-v1"
)
PREFLIGHT_SCHEMA = (
    "gen4-k-directional-alignment-transport-preflight-v1"
)

ITEM_FILE = "item_metrics.jsonl"
SUMMARY_FILE = "summary.json"
MANIFEST_FILE = "manifest.json"
PREFLIGHT_FILE = "preflight.json"
CHECKSUM_FILE = "SHA256SUMS.txt"


class TransportRunnerError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise TransportRunnerError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    return b"".join(
        canonical_json_bytes(row)
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
        raise TransportRunnerError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(
    expected_head: str,
) -> str:
    branch = git(
        "branch",
        "--show-current",
    )
    head = git(
        "rev-parse",
        "HEAD",
    )

    require(
        branch == EXPECTED_BRANCH,
        f"BRANCH_MISMATCH:{branch}",
    )
    require(
        head == expected_head,
        f"HEAD_MISMATCH:{head}",
    )

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            IMPLEMENTATION_PARENT,
            head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "IMPLEMENTATION_PARENT_NOT_ANCESTOR",
    )

    require(
        git("status", "--porcelain") == "",
        "WORKTREE_NOT_CLEAN",
    )

    return head


class ForwardBudget:
    def __init__(self, limit: int):
        require(
            type(limit) is int
            and limit > 0,
            "BAD_FORWARD_BUDGET",
        )
        self.limit = limit
        self.used = 0

    def consume(self) -> None:
        require(
            self.used < self.limit,
            "FORWARD_BUDGET_EXCEEDED",
        )
        self.used += 1

    def assert_exact(self) -> None:
        require(
            self.used == self.limit,
            (
                "FORWARD_BUDGET_NOT_EXACT:"
                f"{self.used}:{self.limit}"
            ),
        )


def event_lookup(
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    for raw in event_rows:
        key = (
            str(raw["source_pair_id"]),
            str(raw["contrast_cell_id"]),
            str(raw["anchor_name"]),
        )
        require(
            key not in result,
            f"DUPLICATE_EVENT:{key}",
        )
        result[key] = dict(raw)

    return result


def validate_transport_event_plan(
    pairs: Sequence[str],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
) -> None:
    cells = (
        core.TARGET_PLUS_CELL,
        core.TARGET_MINUS_CELL,
        core.REFERENCE_PLUS_CELL,
        core.REFERENCE_MINUS_CELL,
    )

    for pair in pairs:
        for cell in cells:
            key = (
                pair,
                cell,
                core.ANCHOR_NAME,
            )
            require(
                key in events,
                f"MISSING_IDENTITY_EVENT:{key}",
            )

            event = events[key]
            anchor = int(
                event[
                    "absolute_anchor_token_index"
                ]
            )
            terminal = int(
                event["terminal_index"]
            )

            require(
                bool(event["post4_eligible"]),
                f"POST4_INELIGIBLE:{key}",
            )
            require(
                anchor + 4
                <= terminal - 1,
                f"POST4_RULE_FAILURE:{key}",
            )
            require(
                anchor + core.TARGET_OFFSET
                < terminal,
                f"TARGET_OFFSET_RANGE:{key}",
            )

        for cell in (
            core.TARGET_PLUS_CELL,
            core.TARGET_MINUS_CELL,
        ):
            identity = events[
                (
                    pair,
                    cell,
                    core.ANCHOR_NAME,
                )
            ]
            name = events[
                (
                    pair,
                    cell,
                    "A_NAME",
                )
            ]

            require(
                int(
                    identity[
                        "absolute_anchor_token_index"
                    ]
                )
                == int(
                    name[
                        "absolute_anchor_token_index"
                    ]
                ),
                (
                    "TARGET_IDENTITY_NAME_"
                    f"MISMATCH:{pair}:{cell}"
                ),
            )


def load_frozen_layer17_endpoints(
) -> dict[tuple[str, str], float]:
    path = (
        ROOT
        / core.FROZEN_LAYER17_ENDPOINT_REL
    )

    require(
        path.is_file(),
        "FROZEN_ENDPOINT_FILE_MISSING",
    )
    require(
        sha256_file(path)
        == core.FROZEN_LAYER17_ENDPOINT_SHA256,
        "FROZEN_ENDPOINT_SHA256_MISMATCH",
    )

    raw_lines = [
        line
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    require(
        len(raw_lines)
        == frozen_stats.CANONICAL_INPUT_ROWS,
        "FROZEN_ENDPOINT_ROW_COUNT",
    )

    result: dict[
        tuple[str, str],
        float,
    ] = {}

    for line in raw_lines:
        row = json.loads(line)

        if int(row["layer_index"]) != 17:
            continue
        if (
            str(row["contrast_cell_id"])
            not in {
                core.TARGET_PLUS_CELL,
                core.TARGET_MINUS_CELL,
            }
        ):
            continue

        require(
            str(row["semantic_anchor"])
            == "A_NAME",
            "FROZEN_ENDPOINT_ANCHOR",
        )

        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(
            key not in result,
            f"DUPLICATE_FROZEN_ENDPOINT:{key}",
        )

        value = float(
            row["POST4_PATH_EFFICIENCY"]
        )
        require(
            math.isfinite(value),
            "NONFINITE_FROZEN_ENDPOINT",
        )
        result[key] = value

    require(
        len(result)
        == 2 * SOURCE_PAIR_COUNT,
        "FROZEN_LAYER17_ENDPOINT_COUNT",
    )

    return result


def build_row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    result: dict[
        tuple[str, str],
        int,
    ] = {}

    for index, row in enumerate(rows):
        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(
            key not in result,
            f"DUPLICATE_PAIR_CELL:{key}",
        )
        result[key] = index

    return result


def _finite_tensor(
    value: Any,
    label: str,
):
    require(
        torch.is_tensor(value),
        f"{label}_NOT_TENSOR",
    )
    out = (
        value.detach()
        .cpu()
        .contiguous()
        .clone()
    )
    require(
        bool(torch.isfinite(out).all().item()),
        f"{label}_NONFINITE",
    )
    return out


def capture_branch(
    model: Any,
    runtime_ctx: Mapping[str, Any],
    *,
    trace_code: Any,
    trace_line: int,
    input_ids: torch.Tensor,
    anchor: int,
    budget: ForwardBudget,
    capture_states: bool,
    delta_h: Any | None = None,
    plus_branch: bool | None = None,
) -> dict[str, Any]:
    target_abs = (
        int(anchor)
        + core.TARGET_OFFSET
    )

    require(
        tuple(input_ids.shape)
        == (
            1,
            extraction.MAX_MODEL_SEQUENCE_LENGTH,
        ),
        "INPUT_SHAPE",
    )
    require(
        input_ids.device.type == "cpu",
        "INPUT_DEVICE",
    )

    layer15 = runtime_ctx["layer15"]
    norm17 = runtime_ctx["norm17"]
    mixer17 = runtime_ctx["mixer17"]

    holders: dict[str, Any] = {}
    counts = {
        "r15": 0,
        "y15": 0,
        "r17": 0,
        "x17": 0,
    }

    def layer15_pre(
        _module,
        args,
    ):
        counts["r15"] += 1
        require(
            counts["r15"] == 1,
            "DUPLICATE_R15_HOOK",
        )
        require(
            len(args) >= 1,
            "R15_ARG_COUNT",
        )
        full = _finite_tensor(
            args[0],
            "R15_FULL",
        )
        require(
            0 <= target_abs
            < full.shape[1],
            "R15_TARGET_RANGE",
        )
        holders["R"] = (
            full[0, target_abs, :]
            .contiguous()
            .clone()
        )

    def layer15_post(
        _module,
        _args,
        output,
    ):
        counts["y15"] += 1
        require(
            counts["y15"] == 1,
            "DUPLICATE_Y15_HOOK",
        )
        full = _finite_tensor(
            output,
            "Y15_FULL",
        )
        require(
            0 <= target_abs
            < full.shape[1],
            "Y15_TARGET_RANGE",
        )
        holders["Y"] = (
            full[0, target_abs, :]
            .contiguous()
            .clone()
        )

    def norm17_pre(
        _module,
        args,
    ):
        counts["r17"] += 1
        require(
            counts["r17"] == 1,
            "DUPLICATE_R17_HOOK",
        )
        require(
            len(args) == 1,
            "R17_ARG_COUNT",
        )
        full = _finite_tensor(
            args[0],
            "R17_FULL",
        )
        require(
            0 <= target_abs
            < full.shape[1],
            "R17_TARGET_RANGE",
        )
        holders["R17"] = (
            full[0, target_abs, :]
            .contiguous()
            .clone()
        )

    def norm17_post(
        _module,
        _args,
        output,
    ):
        counts["x17"] += 1
        require(
            counts["x17"] == 1,
            "DUPLICATE_X17_HOOK",
        )
        full = _finite_tensor(
            output,
            "X17_FULL",
        )
        require(
            0 <= target_abs
            < full.shape[1],
            "X17_TARGET_RANGE",
        )
        holders["X"] = (
            full[0, target_abs, :]
            .contiguous()
            .clone()
        )

    handles = [
        layer15.register_forward_pre_hook(
            layer15_pre
        ),
        layer15.mixer.register_forward_hook(
            layer15_post
        ),
        norm17.register_forward_pre_hook(
            norm17_pre
        ),
        norm17.register_forward_hook(
            norm17_post
        ),
    ]

    intervention_audit = None

    if delta_h is not None:
        require(
            plus_branch is not None,
            "INTERVENTION_BRANCH_REQUIRED",
        )
        intervention_audit = {}
        handles.append(
            transport_runtime.install_inproj_hook(
                mixer17,
                token_index=target_abs,
                strong_mask=runtime_ctx[
                    "strong_mask"
                ],
                delta_h=delta_h,
                plus_branch=bool(
                    plus_branch
                ),
                audit=intervention_audit,
            )
        )

    observer = None

    if capture_states:
        observer = measurement._TraceCollector(
            trace_code,
            trace_line,
            {
                id(mixer17): {
                    "layer_index":
                        core.INTERVENTION_LAYER,
                }
            },
            enabled=True,
        )

    prior_trace = sys.gettrace()
    budget.consume()

    try:
        model.mamba.eval()

        if observer is not None:
            with observer.capture():
                with torch.inference_mode():
                    _ = model.mamba(
                        input_ids=input_ids
                    )
        else:
            with torch.inference_mode():
                _ = model.mamba(
                    input_ids=input_ids
                )
    finally:
        for handle in reversed(handles):
            handle.remove()

    require(
        sys.gettrace() is prior_trace,
        "TRACE_RESTORATION_FAILURE",
    )
    require(
        counts
        == {
            "r15": 1,
            "y15": 1,
            "r17": 1,
            "x17": 1,
        },
        "HOOK_COUNT_FAILURE",
    )
    require(
        set(holders)
        == {
            "R",
            "Y",
            "R17",
            "X",
        },
        "HOOK_CAPTURE_MISSING",
    )

    r17 = holders["R17"].to(
        torch.float64
    )
    eps = float(
        norm17.variance_epsilon
    )

    require(
        math.isfinite(eps)
        and eps > 0.0,
        "NORM17_EPS",
    )

    scale = float(
        torch.rsqrt(
            r17.pow(2).mean()
            + eps
        ).item()
    )

    require(
        math.isfinite(scale)
        and scale > 0.0,
        "RMS_SCALE",
    )

    states = None

    if observer is not None:
        require(
            observer.snapshots is not None,
            "LAYER17_SNAPSHOTS_MISSING",
        )
        states = (
            transport_runtime
            .flatten_layer17_snapshots(
                observer.snapshots,
                token_count=int(
                    input_ids.shape[1]
                ),
            )
        )

    if delta_h is not None:
        require(
            intervention_audit is not None
            and bool(intervention_audit),
            "INTERVENTION_HOOK_NOT_OBSERVED",
        )
        require(
            intervention_audit[
                "token_index"
            ]
            == target_abs,
            "INTERVENTION_TOKEN_AUDIT",
        )

    return {
        "geometry_branch": {
            "R": holders["R"],
            "Y": holders["Y"],
            "X": holders["X"],
            "rms_scale": scale,
        },
        "states": states,
        "intervention_audit":
            intervention_audit,
        "anchor": int(anchor),
        "target_abs": target_abs,
    }


def path_efficiency(
    branch: Mapping[str, Any],
) -> float:
    states = branch["states"]
    require(
        states is not None,
        "STATES_NOT_CAPTURED",
    )
    return core.post4_path_efficiency(
        states,
        int(branch["anchor"]),
    )


def _input_row(
    encoded: Mapping[str, Any],
    row_index: Mapping[
        tuple[str, str],
        int,
    ],
    pair: str,
    cell: str,
) -> torch.Tensor:
    key = (pair, cell)
    require(
        key in row_index,
        f"MISSING_INPUT_ROW:{key}",
    )

    input_ids = encoded["input_ids"]
    require(
        torch.is_tensor(input_ids),
        "INPUT_IDS_NOT_TENSOR",
    )

    index = row_index[key]

    return (
        input_ids[
            index : index + 1
        ]
        .detach()
        .cpu()
        .contiguous()
    )


def run_pair(
    pair: str,
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[
        tuple[str, str],
        int,
    ],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    frozen_endpoint: Mapping[
        tuple[str, str],
        float,
    ],
    budget: ForwardBudget,
) -> dict[str, Any]:
    cells = {
        "tp": core.TARGET_PLUS_CELL,
        "tm": core.TARGET_MINUS_CELL,
        "rp": core.REFERENCE_PLUS_CELL,
        "rm": core.REFERENCE_MINUS_CELL,
    }

    anchors = {
        role: int(
            events[
                (
                    pair,
                    cell,
                    core.ANCHOR_NAME,
                )
            ][
                "absolute_anchor_token_index"
            ]
        )
        for role, cell in cells.items()
    }

    baseline: dict[str, Any] = {}

    for role in (
        "tp",
        "tm",
        "rp",
        "rm",
    ):
        baseline[role] = capture_branch(
            model,
            runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            input_ids=_input_row(
                encoded,
                row_index,
                pair,
                cells[role],
            ),
            anchor=anchors[role],
            budget=budget,
            capture_states=role
            in {"tp", "tm"},
        )

    target_geometry = (
        core.reconstruct_pair_geometry(
            baseline["tp"][
                "geometry_branch"
            ],
            baseline["tm"][
                "geometry_branch"
            ],
            gamma=runtime_ctx["gamma"],
            w_hidden=runtime_ctx[
                "w_hidden"
            ],
            strong_mask=runtime_ctx[
                "strong_mask"
            ],
        )
    )

    reference_geometry = (
        core.reconstruct_pair_geometry(
            baseline["rp"][
                "geometry_branch"
            ],
            baseline["rm"][
                "geometry_branch"
            ],
            gamma=runtime_ctx["gamma"],
            w_hidden=runtime_ctx[
                "w_hidden"
            ],
            strong_mask=runtime_ctx[
                "strong_mask"
            ],
        )
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

    magnitude_norm_delta, mag_core = (
        core.magnitude_delta(
            target_geometry["x"],
            target_geometry["y"],
            reference_geometry["A"],
            reference_geometry["B"],
        )
    )

    magnitude_delta_h = (
        target_geometry["d"]
        * magnitude_norm_delta
    )

    alignment: dict[str, Any] = {}
    magnitude: dict[str, Any] = {}

    for role, plus_branch in (
        ("tp", True),
        ("tm", False),
    ):
        alignment[role] = capture_branch(
            model,
            runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            input_ids=_input_row(
                encoded,
                row_index,
                pair,
                cells[role],
            ),
            anchor=anchors[role],
            budget=budget,
            capture_states=True,
            delta_h=alignment_delta_h,
            plus_branch=plus_branch,
        )

    alignment_pair_audit = (
        transport_runtime
        .paired_intervention_audit(
            alignment["tp"][
                "intervention_audit"
            ],
            alignment["tm"][
                "intervention_audit"
            ],
            alignment_delta_h,
            plus_expected_token_index=
                anchors["tp"]
                + core.TARGET_OFFSET,
            minus_expected_token_index=
                anchors["tm"]
                + core.TARGET_OFFSET,
        )
    )

    for role, plus_branch in (
        ("tp", True),
        ("tm", False),
    ):
        magnitude[role] = capture_branch(
            model,
            runtime_ctx,
            trace_code=trace_code,
            trace_line=trace_line,
            input_ids=_input_row(
                encoded,
                row_index,
                pair,
                cells[role],
            ),
            anchor=anchors[role],
            budget=budget,
            capture_states=True,
            delta_h=magnitude_delta_h,
            plus_branch=plus_branch,
        )

    magnitude_pair_audit = (
        transport_runtime
        .paired_intervention_audit(
            magnitude["tp"][
                "intervention_audit"
            ],
            magnitude["tm"][
                "intervention_audit"
            ],
            magnitude_delta_h,
            plus_expected_token_index=
                anchors["tp"]
                + core.TARGET_OFFSET,
            minus_expected_token_index=
                anchors["tm"]
                + core.TARGET_OFFSET,
        )
    )

    baseline_plus = path_efficiency(
        baseline["tp"]
    )
    baseline_minus = path_efficiency(
        baseline["tm"]
    )

    frozen_plus = float(
        frozen_endpoint[
            (
                pair,
                core.TARGET_PLUS_CELL,
            )
        ]
    )
    frozen_minus = float(
        frozen_endpoint[
            (
                pair,
                core.TARGET_MINUS_CELL,
            )
        ]
    )

    plus_repro = abs(
        baseline_plus - frozen_plus
    )
    minus_repro = abs(
        baseline_minus - frozen_minus
    )

    require(
        plus_repro
        <= BASELINE_REPRO_TOL,
        (
            "BASELINE_PLUS_REPRO:"
            f"{pair}:{plus_repro}"
        ),
    )
    require(
        minus_repro
        <= BASELINE_REPRO_TOL,
        (
            "BASELINE_MINUS_REPRO:"
            f"{pair}:{minus_repro}"
        ),
    )

    alignment_plus = path_efficiency(
        alignment["tp"]
    )
    alignment_minus = path_efficiency(
        alignment["tm"]
    )
    magnitude_plus = path_efficiency(
        magnitude["tp"]
    )
    magnitude_minus = path_efficiency(
        magnitude["tm"]
    )

    reductions = core.causal_reductions(
        baseline_plus,
        baseline_minus,
        alignment_plus,
        alignment_minus,
        magnitude_plus,
        magnitude_minus,
    )

    align_cos_resid = abs(
        float(
            align_core["realized_C"]
        )
        - float(
            align_core["target_C"]
        )
    )

    mag_cos_resid = abs(
        float(
            mag_core["realized_C"]
        )
        - float(
            mag_core["baseline_C"]
        )
    )

    return {
        "schema_version": ITEM_SCHEMA,
        "source_pair_id": pair,
        "source_block":
            core.SOURCE_BLOCK,
        "target_residual_layer":
            core.TARGET_RESIDUAL_LAYER,
        "intervention_layer":
            core.INTERVENTION_LAYER,
        "relative_coordinate":
            core.TARGET_OFFSET,
        "target_plus_cell":
            core.TARGET_PLUS_CELL,
        "target_minus_cell":
            core.TARGET_MINUS_CELL,
        "reference_plus_cell":
            core.REFERENCE_PLUS_CELL,
        "reference_minus_cell":
            core.REFERENCE_MINUS_CELL,
        "anchor_name":
            core.ANCHOR_NAME,
        "target_plus_anchor":
            anchors["tp"],
        "target_minus_anchor":
            anchors["tm"],
        "reference_plus_anchor":
            anchors["rp"],
        "reference_minus_anchor":
            anchors["rm"],
        "target_plus_intervention_token":
            anchors["tp"]
            + core.TARGET_OFFSET,
        "target_minus_intervention_token":
            anchors["tm"]
            + core.TARGET_OFFSET,
        "reference_plus_geometry_token":
            anchors["rp"]
            + core.TARGET_OFFSET,
        "reference_minus_geometry_token":
            anchors["rm"]
            + core.TARGET_OFFSET,
        "target_A":
            float(
                target_geometry["A"]
            ),
        "target_B":
            float(
                target_geometry["B"]
            ),
        "target_C":
            float(
                target_geometry["C"]
            ),
        "reference_A":
            float(
                reference_geometry["A"]
            ),
        "reference_B":
            float(
                reference_geometry["B"]
            ),
        "reference_C":
            float(
                reference_geometry["C"]
            ),
        "alignment_target_cosine":
            float(
                align_core["target_C"]
            ),
        "alignment_realized_cosine":
            float(
                align_core["realized_C"]
            ),
        "alignment_cosine_abs_residual":
            align_cos_resid,
        "alignment_A_preservation_abs_residual":
            float(
                align_core[
                    "A_preservation_abs_residual"
                ]
            ),
        "alignment_B_preservation_abs_residual":
            float(
                align_core[
                    "B_preservation_abs_residual"
                ]
            ),
        "alignment_midpoint_max_abs_residual":
            float(
                alignment_pair_audit[
                    "midpoint_max_abs_residual"
                ]
            ),
        "alignment_pair_delta_max_abs_residual":
            float(
                alignment_pair_audit[
                    "pair_delta_max_abs_residual"
                ]
            ),
        "alignment_applied_correction_max_abs_residual":
            float(
                alignment_pair_audit[
                    "applied_correction_max_abs_residual"
                ]
            ),
        "magnitude_target_A":
            float(
                mag_core["target_A"]
            ),
        "magnitude_target_B":
            float(
                mag_core["target_B"]
            ),
        "magnitude_baseline_cosine":
            float(
                mag_core["baseline_C"]
            ),
        "magnitude_realized_cosine":
            float(
                mag_core["realized_C"]
            ),
        "magnitude_cosine_abs_residual":
            mag_cos_resid,
        "magnitude_A_target_abs_residual":
            float(
                mag_core[
                    "A_target_abs_residual"
                ]
            ),
        "magnitude_B_target_abs_residual":
            float(
                mag_core[
                    "B_target_abs_residual"
                ]
            ),
        "magnitude_midpoint_max_abs_residual":
            float(
                magnitude_pair_audit[
                    "midpoint_max_abs_residual"
                ]
            ),
        "magnitude_pair_delta_max_abs_residual":
            float(
                magnitude_pair_audit[
                    "pair_delta_max_abs_residual"
                ]
            ),
        "magnitude_applied_correction_max_abs_residual":
            float(
                magnitude_pair_audit[
                    "applied_correction_max_abs_residual"
                ]
            ),
        "baseline_plus_path_efficiency":
            baseline_plus,
        "baseline_minus_path_efficiency":
            baseline_minus,
        "frozen_plus_path_efficiency":
            frozen_plus,
        "frozen_minus_path_efficiency":
            frozen_minus,
        "baseline_plus_reproduction_abs_residual":
            plus_repro,
        "baseline_minus_reproduction_abs_residual":
            minus_repro,
        "alignment_plus_path_efficiency":
            alignment_plus,
        "alignment_minus_path_efficiency":
            alignment_minus,
        "magnitude_plus_path_efficiency":
            magnitude_plus,
        "magnitude_minus_path_efficiency":
            magnitude_minus,
        **reductions,
    }


def one_sided_greater_t(
    values: Sequence[float],
) -> dict[str, float | int]:
    finite = [
        float(value)
        for value in values
    ]

    require(
        len(finite) >= 2,
        "T_TEST_N",
    )
    require(
        all(
            math.isfinite(v)
            for v in finite
        ),
        "T_TEST_NONFINITE",
    )

    n = len(finite)
    mean = statistics.fmean(finite)
    sd = statistics.stdev(finite)

    require(
        sd > 0.0,
        "T_TEST_ZERO_VARIANCE",
    )

    se = sd / math.sqrt(n)
    t = mean / se
    df = n - 1

    two_sided = (
        frozen_stats.student_t_two_sided_p(
            t,
            df,
        )
    )

    if t >= 0.0:
        raw_p = 0.5 * two_sided
    else:
        raw_p = 1.0 - 0.5 * two_sided

    return {
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "standard_error": se,
        "df": df,
        "t_statistic": t,
        "raw_p_value":
            min(
                1.0,
                max(
                    0.0,
                    float(raw_p),
                ),
            ),
        "d_z": mean / sd,
    }


def holm_two(
    p1: float,
    p2: float,
) -> tuple[float, float]:
    values = [
        float(p1),
        float(p2),
    ]

    require(
        all(
            0.0 <= p <= 1.0
            and math.isfinite(p)
            for p in values
        ),
        "HOLM_P_RANGE",
    )

    order = sorted(
        range(2),
        key=lambda i: (
            values[i],
            i,
        ),
    )

    adjusted = [0.0, 0.0]
    running = 0.0

    for rank, index in enumerate(order):
        value = min(
            1.0,
            (2 - rank)
            * values[index],
        )
        running = max(
            running,
            value,
        )
        adjusted[index] = running

    return (
        adjusted[0],
        adjusted[1],
    )


def build_full_summary(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(
        len(items) == SOURCE_PAIR_COUNT,
        "SUMMARY_ITEM_COUNT",
    )

    baseline_residual = max(
        max(
            float(
                row[
                    "baseline_plus_reproduction_abs_residual"
                ]
            ),
            float(
                row[
                    "baseline_minus_reproduction_abs_residual"
                ]
            ),
        )
        for row in items
    )

    alignment_cosine_residual = max(
        float(
            row[
                "alignment_cosine_abs_residual"
            ]
        )
        for row in items
    )
    alignment_a_residual = max(
        float(
            row[
                "alignment_A_preservation_abs_residual"
            ]
        )
        for row in items
    )
    alignment_b_residual = max(
        float(
            row[
                "alignment_B_preservation_abs_residual"
            ]
        )
        for row in items
    )
    magnitude_cosine_residual = max(
        float(
            row[
                "magnitude_cosine_abs_residual"
            ]
        )
        for row in items
    )

    magnitude_a_residual = max(
        float(
            row[
                "magnitude_A_target_abs_residual"
            ]
        )
        for row in items
    )
    magnitude_b_residual = max(
        float(
            row[
                "magnitude_B_target_abs_residual"
            ]
        )
        for row in items
    )

    alignment_midpoint = max(
        float(
            row[
                "alignment_midpoint_max_abs_residual"
            ]
        )
        for row in items
    )
    magnitude_midpoint = max(
        float(
            row[
                "magnitude_midpoint_max_abs_residual"
            ]
        )
        for row in items
    )

    alignment_applied = max(
        float(
            row[
                "alignment_applied_correction_max_abs_residual"
            ]
        )
        for row in items
    )
    magnitude_applied = max(
        float(
            row[
                "magnitude_applied_correction_max_abs_residual"
            ]
        )
        for row in items
    )

    baseline_deltas = [
        float(row["delta_baseline"])
        for row in items
    ]
    r_align = [
        float(row["R_ALIGN"])
        for row in items
    ]
    specificity = [
        float(
            row[
                "ALIGNMENT_SPECIFICITY"
            ]
        )
        for row in items
    ]

    baseline_mean = statistics.fmean(
        baseline_deltas
    )

    require(
        abs(
            baseline_mean
            - core.FROZEN_DELTA_NAME_PATH_EFFICIENCY_MEAN
        )
        <= BASELINE_REPRO_TOL,
        (
            "FROZEN_BASELINE_MEAN_REPRO:"
            f"{baseline_mean}"
        ),
    )

    h1 = one_sided_greater_t(
        r_align
    )
    h2 = one_sided_greater_t(
        specificity
    )

    h1_adj, h2_adj = holm_two(
        float(h1["raw_p_value"]),
        float(h2["raw_p_value"]),
    )

    h1_reject = (
        h1_adj < FAMILY_ALPHA
        and float(h1["mean"]) > 0.0
    )
    h2_reject = (
        h2_adj < FAMILY_ALPHA
        and float(h2["mean"]) > 0.0
    )

    manipulation_pass = (
        baseline_residual
        <= BASELINE_REPRO_TOL
        and alignment_cosine_residual
        <= core.VECTOR_TOL
        and alignment_a_residual
        <= core.VECTOR_TOL
        and alignment_b_residual
        <= core.VECTOR_TOL
        and magnitude_cosine_residual
        <= core.VECTOR_TOL
        and magnitude_a_residual
        <= core.VECTOR_TOL
        and magnitude_b_residual
        <= core.VECTOR_TOL
        and alignment_midpoint
        <= transport_runtime.MIDPOINT_TOL
        and magnitude_midpoint
        <= transport_runtime.MIDPOINT_TOL
        and alignment_applied
        <= transport_runtime.RUNTIME_CAST_TOL
        and magnitude_applied
        <= transport_runtime.RUNTIME_CAST_TOL
    )

    outcome = (
        "SUPPORTED_DIRECTIONAL_ALIGNMENT_CAUSAL_TRANSPORT"
        if (
            manipulation_pass
            and h1_reject
            and h2_reject
        )
        else
        "DIRECTIONAL_ALIGNMENT_CAUSAL_TRANSPORT_NOT_ESTABLISHED"
    )

    return {
        "schema_version":
            SUMMARY_SCHEMA,
        "population_size":
            SOURCE_PAIR_COUNT,
        "model_forward_count":
            FULL_FORWARD_BUDGET,
        "baseline_delta_name_path_efficiency_mean":
            baseline_mean,
        "frozen_baseline_delta_name_path_efficiency_mean":
            core.FROZEN_DELTA_NAME_PATH_EFFICIENCY_MEAN,
        "max_baseline_reproduction_abs_residual":
            baseline_residual,
        "max_alignment_cosine_abs_residual":
            alignment_cosine_residual,
        "max_alignment_A_preservation_abs_residual":
            alignment_a_residual,
        "max_alignment_B_preservation_abs_residual":
            alignment_b_residual,
        "max_magnitude_cosine_abs_residual":
            magnitude_cosine_residual,
        "max_magnitude_A_target_abs_residual":
            magnitude_a_residual,
        "max_magnitude_B_target_abs_residual":
            magnitude_b_residual,
        "max_alignment_midpoint_abs_residual":
            alignment_midpoint,
        "max_magnitude_midpoint_abs_residual":
            magnitude_midpoint,
        "max_alignment_applied_correction_abs_residual":
            alignment_applied,
        "max_magnitude_applied_correction_abs_residual":
            magnitude_applied,
        "all_mandatory_manipulation_checks_pass":
            manipulation_pass,
        "hypothesis_family": [
            {
                "id": "H1_R_ALIGN_GT_ZERO",
                **h1,
                "holm_adjusted_p_value":
                    h1_adj,
                "reject_holm_alpha_0_05":
                    h1_reject,
            },
            {
                "id":
                    "H2_ALIGNMENT_SPECIFICITY_GT_ZERO",
                **h2,
                "holm_adjusted_p_value":
                    h2_adj,
                "reject_holm_alpha_0_05":
                    h2_reject,
            },
        ],
        "outcome":
            outcome,
    }


def build_preflight_public(
    items: Sequence[Mapping[str, Any]],
    *,
    forward_count: int,
) -> dict[str, Any]:
    require(
        len(items)
        == PREFLIGHT_PAIR_COUNT,
        "PREFLIGHT_ITEM_COUNT",
    )

    baseline_residual = max(
        max(
            float(
                row[
                    "baseline_plus_reproduction_abs_residual"
                ]
            ),
            float(
                row[
                    "baseline_minus_reproduction_abs_residual"
                ]
            ),
        )
        for row in items
    )

    alignment_a_residual = max(
        float(
            row[
                "alignment_A_preservation_abs_residual"
            ]
        )
        for row in items
    )
    alignment_b_residual = max(
        float(
            row[
                "alignment_B_preservation_abs_residual"
            ]
        )
        for row in items
    )
    magnitude_a_residual = max(
        float(
            row[
                "magnitude_A_target_abs_residual"
            ]
        )
        for row in items
    )
    magnitude_b_residual = max(
        float(
            row[
                "magnitude_B_target_abs_residual"
            ]
        )
        for row in items
    )

    alignment_midpoint = max(
        float(
            row[
                "alignment_midpoint_max_abs_residual"
            ]
        )
        for row in items
    )
    magnitude_midpoint = max(
        float(
            row[
                "magnitude_midpoint_max_abs_residual"
            ]
        )
        for row in items
    )
    alignment_applied = max(
        float(
            row[
                "alignment_applied_correction_max_abs_residual"
            ]
        )
        for row in items
    )
    magnitude_applied = max(
        float(
            row[
                "magnitude_applied_correction_max_abs_residual"
            ]
        )
        for row in items
    )

    return {
        "schema_version":
            PREFLIGHT_SCHEMA,
        "pair_count":
            PREFLIGHT_PAIR_COUNT,
        "model_forward_count":
            forward_count,
        "max_baseline_reproduction_abs_residual":
            baseline_residual,
        "max_alignment_A_preservation_abs_residual":
            alignment_a_residual,
        "max_alignment_B_preservation_abs_residual":
            alignment_b_residual,
        "max_magnitude_A_target_abs_residual":
            magnitude_a_residual,
        "max_magnitude_B_target_abs_residual":
            magnitude_b_residual,
        "max_alignment_midpoint_abs_residual":
            alignment_midpoint,
        "max_magnitude_midpoint_abs_residual":
            magnitude_midpoint,
        "max_alignment_applied_correction_abs_residual":
            alignment_applied,
        "max_magnitude_applied_correction_abs_residual":
            magnitude_applied,
        "scientific_endpoint_values_serialized":
            False,
        "inferential_statistics_executed":
            False,
        "result":
            "PASS_BOUNDED_PREFLIGHT",
    }


def publish_bundle(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any] | None,
    preflight: Mapping[str, Any] | None,
) -> None:
    require(
        not output_dir.exists(),
        "OUTPUT_COLLISION",
    )

    staging = output_dir.with_name(
        output_dir.name + ".staging"
    )
    require(
        not staging.exists(),
        "STAGING_COLLISION",
    )

    staging.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        files: dict[str, bytes] = {
            MANIFEST_FILE:
                canonical_json_bytes(
                    manifest
                ),
        }

        if summary is not None:
            require(
                preflight is None,
                "MIXED_OUTPUT_MODE",
            )
            files[ITEM_FILE] = (
                jsonl_bytes(items)
            )
            files[SUMMARY_FILE] = (
                canonical_json_bytes(
                    summary
                )
            )
        else:
            require(
                preflight is not None,
                "PREFLIGHT_MISSING",
            )
            files[PREFLIGHT_FILE] = (
                canonical_json_bytes(
                    preflight
                )
            )

        for name, raw in files.items():
            (
                staging / name
            ).write_bytes(raw)

        checksum_lines = []

        for name in sorted(files):
            checksum_lines.append(
                f"{sha256_file(staging / name)}  {name}\n"
            )

        (
            staging / CHECKSUM_FILE
        ).write_text(
            "".join(checksum_lines),
            encoding="utf-8",
            newline="\n",
        )

        staging.replace(output_dir)

    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def run_transport(
    *,
    mode: str,
    output_dir: Path,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
) -> dict[str, Any]:
    require(
        mode in {"preflight", "full"},
        "BAD_MODE",
    )

    head = authenticate_repo(
        expected_head
    )

    # Exact frozen runtime identity before
    # tokenizer/model scientific work.
    measurement.runtime_gate()

    rows, encoded, event_rows = (
        extraction.load_scientific_inputs(
            tokenizer_snapshot
        )
    )

    pairs = list(
        extraction.canonical_pair_order(
            rows
        )
    )
    require(
        len(pairs)
        == SOURCE_PAIR_COUNT,
        "PAIR_COUNT",
    )

    events = event_lookup(
        event_rows
    )
    validate_transport_event_plan(
        pairs,
        events,
    )

    frozen_endpoint = (
        load_frozen_layer17_endpoints()
    )
    row_index = build_row_index(
        rows
    )

    model, checkpoint_sha = (
        extraction.load_representative_model(
            model_snapshot
        )
    )

    require(
        checkpoint_sha
        == extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
        "CHECKPOINT_IDENTITY",
    )

    runtime_ctx = (
        transport_runtime
        .validate_runtime_components(
            model
        )
    )

    trace_code, trace_line = (
        measurement
        ._resolve_and_validate_runtime_binding()
    )

    selected_pairs = (
        pairs[:PREFLIGHT_PAIR_COUNT]
        if mode == "preflight"
        else pairs
    )

    budget = ForwardBudget(
        PREFLIGHT_FORWARD_BUDGET
        if mode == "preflight"
        else FULL_FORWARD_BUDGET
    )

    items = []

    for pair in selected_pairs:
        items.append(
            run_pair(
                pair,
                model=model,
                runtime_ctx=runtime_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                encoded=encoded,
                row_index=row_index,
                events=events,
                frozen_endpoint=
                    frozen_endpoint,
                budget=budget,
            )
        )

    budget.assert_exact()

    runner_path = (
        ROOT / RUNNER_REL
    )

    manifest = {
        "schema_version":
            MANIFEST_SCHEMA,
        "mode":
            mode,
        "runtime_branch":
            EXPECTED_BRANCH,
        "runtime_git_head":
            head,
        "implementation_parent":
            IMPLEMENTATION_PARENT,
        "gen4_parent":
            core.GEN4_PARENT,
        "k_causal_parent":
            core.K_CAUSAL_PARENT,
        "runner_rel":
            RUNNER_REL,
        "runner_sha256":
            sha256_file(
                runner_path
            ),
        "core_rel":
            (
                "scripts/"
                "reason_router_gen4_k_"
                "directional_alignment_"
                "transport_core.py"
            ),
        "core_sha256":
            sha256_file(
                ROOT
                / "scripts"
                / "reason_router_gen4_k_directional_alignment_transport_core.py"
            ),
        "runtime_rel":
            (
                "scripts/"
                "reason_router_gen4_k_"
                "directional_alignment_"
                "transport_runtime.py"
            ),
        "runtime_sha256":
            sha256_file(
                ROOT
                / "scripts"
                / "reason_router_gen4_k_directional_alignment_transport_runtime.py"
            ),
        "checkpoint_sha256":
            checkpoint_sha,
        "frozen_endpoint_rel":
            core.FROZEN_LAYER17_ENDPOINT_REL,
        "frozen_endpoint_sha256":
            core.FROZEN_LAYER17_ENDPOINT_SHA256,
        "event_manifest_rel":
            extraction.EVENT_MANIFEST.as_posix(),
        "event_manifest_sha256":
            extraction.EVENT_MANIFEST_SHA256,
        "mamba_source_sha256":
            measurement.MAMBA_SHA256,
        "source_pair_count":
            len(selected_pairs),
        "model_forward_count":
            budget.used,
        "forwards_per_pair":
            FORWARDS_PER_PAIR,
        "source_block":
            core.SOURCE_BLOCK,
        "target_residual_layer":
            core.TARGET_RESIDUAL_LAYER,
        "intervention_layer":
            core.INTERVENTION_LAYER,
        "relative_coordinate":
            core.TARGET_OFFSET,
        "target_pair":
            [
                core.TARGET_PLUS_CELL,
                core.TARGET_MINUS_CELL,
            ],
        "reference_pair":
            [
                core.REFERENCE_PLUS_CELL,
                core.REFERENCE_MINUS_CELL,
            ],
        "anchor_name":
            core.ANCHOR_NAME,
        "strong_count":
            core.EXPECTED_STRONG_COUNT,
        "weak_count":
            core.EXPECTED_WEAK_COUNT,
        "equal_count":
            core.EXPECTED_EQUAL_COUNT,
        "strong_index_sha256":
            core.EXPECTED_STRONG_INDEX_SHA256,
        "training_executed":
            False,
        "backward_executed":
            False,
        "task_heads_executed":
            False,
        "logits_read":
            False,
        "raw_vectors_persisted":
            False,
        "tokenizer_invoked":
            True,
        "causal_intervention_executed":
            True,
        "statistical_testing":
            mode == "full",
    }

    if mode == "preflight":
        public = build_preflight_public(
            items,
            forward_count=budget.used,
        )

        publish_bundle(
            output_dir,
            manifest=manifest,
            items=(),
            summary=None,
            preflight=public,
        )

        return {
            "result":
                public["result"],
            "model_forward_count":
                budget.used,
            "scientific_values_serialized":
                False,
        }

    summary = build_full_summary(
        items
    )

    publish_bundle(
        output_dir,
        manifest=manifest,
        items=items,
        summary=summary,
        preflight=None,
    )

    return {
        "result":
            summary["outcome"],
        "model_forward_count":
            budget.used,
        "population_size":
            SOURCE_PAIR_COUNT,
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "preflight",
            "full",
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--expected-head",
        required=True,
    )
    parser.add_argument(
        "--model-snapshot",
        required=True,
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        required=True,
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)

    result = run_transport(
        mode=args.mode,
        output_dir=Path(
            args.output_dir
        ),
        expected_head=
            args.expected_head,
        model_snapshot=Path(
            args.model_snapshot
        ),
        tokenizer_snapshot=Path(
            args.tokenizer_snapshot
        ),
    )

    # Deliberately no scientific scalar
    # values are printed here.
    print(
        "RESULT =",
        result["result"],
    )
    print(
        "MODEL_FORWARD_COUNT =",
        result[
            "model_forward_count"
        ],
    )


if __name__ == "__main__":
    main()
