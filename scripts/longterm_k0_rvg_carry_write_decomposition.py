"""K0-RVG divergence-aligned layer-23 carry/write decomposition.

Scientific scope:
- frozen token IDs only; no tokenizer,
- same divergence anchors and equal-length prefix protocol,
- same CPU float32 sequential Mamba recurrence observer,
- layer 23, k=-1..+6,
- decompose S_post = carry + write where carry = G * S_prev and write = W,
- record matched/swapped L2 differences and the carry/write interaction term,
- no logits, heads, training, intervention, PCA, probes, or learned geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PARENT_FREEZE_COMMIT = "c01cdf8bd1c61f244bb761b2de624b794578a6ba"
PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_divergence_magnitude_trajectory_32b2a65_v1/"
    "summary.json"
)

MAGNITUDE_RUNNER_REL = "scripts/longterm_k0_rvg_divergence_magnitude_trajectory.py"
MAGNITUDE_RUNNER_SHA256 = (
    "a8954598b43fbdc842934441cc7252ef2585fb0e082b0e89024b159252859a3e"
)

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_FORWARD_COUNT = 1344

PRIMARY_LAYER = 23
POST_HORIZON = 6
RELATIVE_COORDINATES = tuple(range(-1, POST_HORIZON + 1))
EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"

COMMON_SIGNATURE = "DDSSSSS"
EXPECTED_COMMON_COHORT_COUNT = 330

COMPOSITION_RESIDUAL_RTOL = 1e-5


class DecompositionError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise DecompositionError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def import_module(path: Path, name: str):
    require(path.is_file(), f"MODULE_MISSING:{path}")

    spec = importlib.util.spec_from_file_location(name, path)
    require(
        spec is not None and spec.loader is not None,
        f"MODULE_SPEC_FAILURE:{path}",
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_magnitude_runner(root: Path):
    path = root / MAGNITUDE_RUNNER_REL

    require(
        sha256_bytes(path.read_bytes()) == MAGNITUDE_RUNNER_SHA256,
        "MAGNITUDE_RUNNER_SHA256_MISMATCH",
    )

    return import_module(
        path,
        "k0_rvg_magnitude_parent_runner",
    )


def authenticate_parent(
    root: Path,
    magnitude: Any,
    base: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = base.authenticate_repo(root)

    ancestor = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            PARENT_FREEZE_COMMIT,
            repo["head"],
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "PARENT_FREEZE_NOT_ANCESTOR")

    raw = base.git_bytes(
        root,
        f"{PARENT_FREEZE_COMMIT}:{PARENT_SUMMARY_REL}",
    )
    parent = json.loads(raw)

    current = root / PARENT_SUMMARY_REL
    require(current.is_file(), "PARENT_SUMMARY_MISSING")
    require(
        current.read_bytes() == raw,
        "PARENT_SUMMARY_WORKTREE_DRIFT",
    )

    require(
        parent.get("schema_version")
        == "k0-rvg-divergence-magnitude-trajectory-summary-v1",
        "PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        parent.get("item_count") == EXPECTED_ITEM_COUNT,
        "PARENT_ITEM_COUNT_MISMATCH",
    )
    require(
        parent.get("pair_role_count") == EXPECTED_PAIR_ROLE_COUNT,
        "PARENT_PAIR_ROLE_COUNT_MISMATCH",
    )
    require(
        parent.get("primary_layer") == PRIMARY_LAYER,
        "PARENT_LAYER_MISMATCH",
    )
    require(
        parent.get("state_field") == "S_post",
        "PARENT_STATE_FIELD_MISMATCH",
    )
    require(
        parent.get("relative_coordinates")
        == list(RELATIVE_COORDINATES),
        "PARENT_COORDINATES_MISMATCH",
    )
    require(
        parent.get("execution_protocol") == EXECUTION_PROTOCOL,
        "PARENT_PROTOCOL_MISMATCH",
    )
    require(
        parent.get("k_minus_1_exact_identity_count")
        == EXPECTED_PAIR_ROLE_COUNT,
        "PARENT_K_MINUS_1_MISMATCH",
    )
    require(
        parent.get("k_zero_exact_difference_count")
        == EXPECTED_PAIR_ROLE_COUNT,
        "PARENT_K_ZERO_MISMATCH",
    )

    return repo, parent


def build_plan_and_common_cohort(
    root: Path,
    magnitude: Any,
    base: Any,
):
    plan = magnitude.build_plan(root, base)

    require(
        len(plan) == EXPECTED_PAIR_ROLE_COUNT,
        "PLAN_COUNT_MISMATCH",
    )

    signatures: dict[tuple[int, str], str] = {}

    for row in plan:
        idx = int(row["local_template_index"])
        role = str(row["role"])
        anchor = int(row["anchor"])
        matched = row["matched_ids"]
        swapped = row["swapped_ids"]

        sig = "".join(
            "S"
            if int(matched[anchor + k]) == int(swapped[anchor + k])
            else "D"
            for k in range(0, 7)
        )

        signatures[(idx, role)] = sig

    corr = {
        idx
        for (idx, role), sig in signatures.items()
        if role == "corr" and sig == COMMON_SIGNATURE
    }
    ctrl = {
        idx
        for (idx, role), sig in signatures.items()
        if role == "ctrl" and sig == COMMON_SIGNATURE
    }

    require(
        len(corr) == EXPECTED_COMMON_COHORT_COUNT,
        "CORR_COMMON_COHORT_COUNT_MISMATCH",
    )
    require(
        len(ctrl) == EXPECTED_COMMON_COHORT_COUNT,
        "CTRL_COMMON_COHORT_COUNT_MISMATCH",
    )
    require(
        corr == ctrl,
        "COMMON_COHORT_ROLE_MISMATCH",
    )

    return plan, frozenset(corr), signatures


def capture_components(
    base: Any,
    observer: Any,
    model: Any,
    layer_map: Mapping[int, int],
    binding: Any,
    token_ids: Sequence[int],
    targets: Sequence[int],
) -> dict[int, dict[str, Any]]:
    import torch

    target_set = tuple(int(v) for v in targets)

    collector = observer.RawRecurrenceCollector(
        binding,
        layer_map,
        target_set,
    )

    with collector.capture():
        base.direct_backbone_forward(model, token_ids)

    require(
        collector.records is not None,
        "CAPTURE_RECORDS_MISSING",
    )

    expected = {
        (layer, token)
        for layer in range(24)
        for token in target_set
    }

    require(
        set(collector.records) == expected,
        "CAPTURE_COORDINATE_SET_MISMATCH",
    )

    out: dict[int, dict[str, Any]] = {}

    for token in target_set:
        record = collector.records[(PRIMARY_LAYER, token)]

        audit = observer.validate_recurrence_record(record)
        require(
            audit.get("recurrence_exact") == "PASS_EXACT",
            "RECURRENCE_EXACT_FAILURE",
        )

        carry = (
            record.g * record.s_prev
        ).detach().cpu().contiguous()

        require(
            torch.equal(carry + record.w, record.s_post),
            "CARRY_WRITE_EXACT_RECONSTRUCTION_FAILURE",
        )

        require(
            tuple(carry.shape) == (1, 1536, 16),
            "CARRY_SHAPE_MISMATCH",
        )
        require(
            str(carry.dtype) == "torch.float32",
            "CARRY_DTYPE_MISMATCH",
        )

        out[token] = {
            "carry": carry,
            "write": record.w,
            "s_post": record.s_post,
        }

    return out


def metric_rows_for_pair(
    row: Mapping[str, Any],
    matched: Mapping[int, Mapping[str, Any]],
    swapped: Mapping[int, Mapping[str, Any]],
    common_cohort: frozenset[int],
    token_signature: str,
) -> list[dict[str, Any]]:
    import torch

    idx = int(row["local_template_index"])
    role = str(row["role"])
    anchor = int(row["anchor"])

    provisional: list[dict[str, Any]] = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k

        mc_raw = matched[token]["carry"]
        mw_raw = matched[token]["write"]
        ms_raw = matched[token]["s_post"]

        sc_raw = swapped[token]["carry"]
        sw_raw = swapped[token]["write"]
        ss_raw = swapped[token]["s_post"]

        carry_equal = bool(torch.equal(mc_raw, sc_raw))
        write_equal = bool(torch.equal(mw_raw, sw_raw))
        state_equal = bool(torch.equal(ms_raw, ss_raw))

        if k == -1:
            require(
                carry_equal and write_equal and state_equal,
                f"K_MINUS_1_COMPONENT_IDENTITY_FAILURE:{idx}:{role}",
            )

        if k == 0:
            require(
                not state_equal,
                f"K_ZERO_STATE_NOT_DIVERGENT:{idx}:{role}",
            )

        # Scientific snapshots are float32. Metrics are accumulated in float64.
        mc = mc_raw.to(dtype=torch.float64)
        mw = mw_raw.to(dtype=torch.float64)
        ms = ms_raw.to(dtype=torch.float64)

        sc = sc_raw.to(dtype=torch.float64)
        sw = sw_raw.to(dtype=torch.float64)
        ss = ss_raw.to(dtype=torch.float64)

        dc = mc - sc
        dw = mw - sw
        ds = ms - ss

        dc_l2 = float(torch.linalg.vector_norm(dc).item())
        dw_l2 = float(torch.linalg.vector_norm(dw).item())
        ds_l2 = float(torch.linalg.vector_norm(ds).item())

        interaction = float(
            2.0 * torch.sum(dc * dw).item()
        )

        reconstructed_delta = dc + dw
        residual = ds - reconstructed_delta

        residual_l2 = float(
            torch.linalg.vector_norm(residual).item()
        )
        residual_relative = residual_l2 / max(ds_l2, 1e-12)

        require(
            all(
                math.isfinite(v)
                for v in (
                    dc_l2,
                    dw_l2,
                    ds_l2,
                    interaction,
                    residual_l2,
                    residual_relative,
                )
            ),
            f"NONFINITE_COMPONENT_METRIC:{idx}:{role}:{k}",
        )

        require(
            residual_relative <= COMPOSITION_RESIDUAL_RTOL,
            (
                f"COMPOSITION_RESIDUAL_TOO_LARGE:"
                f"{idx}:{role}:{k}:{residual_relative}"
            ),
        )

        provisional.append(
            {
                "schema_version":
                    "k0-rvg-carry-write-decomposition-row-v1",
                "local_template_index": idx,
                "stable_item_id": row["stable_item_id"],
                "role": role,
                "relative_coordinate": k,
                "token_index": token,
                "divergence_anchor_token_index": anchor,
                "token_equality_signature_k0_to_k6":
                    token_signature,
                "in_common_ddsssss_cohort":
                    idx in common_cohort,
                "carry_exact_equal": carry_equal,
                "write_exact_equal": write_equal,
                "s_post_exact_equal": state_equal,
                "delta_carry_l2": dc_l2,
                "delta_write_l2": dw_l2,
                "delta_s_post_l2": ds_l2,
                "carry_write_interaction":
                    interaction,
                "composition_residual_l2":
                    residual_l2,
                "composition_relative_residual":
                    residual_relative,
                "source_snapshot_dtype": "torch.float32",
                "metric_accumulation_dtype": "torch.float64",
            }
        )

    k0_s = next(
        float(r["delta_s_post_l2"])
        for r in provisional
        if int(r["relative_coordinate"]) == 0
    )

    require(
        k0_s > 0.0 and math.isfinite(k0_s),
        f"K_ZERO_STATE_MAGNITUDE_INVALID:{idx}:{role}",
    )

    k0_s_sq = k0_s * k0_s

    for r in provisional:
        r["delta_carry_l2_over_k0_s"] = (
            float(r["delta_carry_l2"]) / k0_s
        )
        r["delta_write_l2_over_k0_s"] = (
            float(r["delta_write_l2"]) / k0_s
        )
        r["delta_s_post_l2_over_k0_s"] = (
            float(r["delta_s_post_l2"]) / k0_s
        )
        r["carry_write_interaction_over_k0_s_sq"] = (
            float(r["carry_write_interaction"]) / k0_s_sq
        )

    return provisional


def aggregate(values: Sequence[float]) -> dict[str, float | int]:
    vals = [float(v) for v in values]

    require(bool(vals), "EMPTY_AGGREGATE")
    require(
        all(math.isfinite(v) for v in vals),
        "NONFINITE_AGGREGATE_INPUT",
    )

    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def aggregate_trajectory(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    trajectory: dict[str, dict[str, Any]] = {
        "corr": {},
        "ctrl": {},
    }

    metrics = (
        "delta_carry_l2",
        "delta_write_l2",
        "delta_s_post_l2",
        "carry_write_interaction",
        "delta_carry_l2_over_k0_s",
        "delta_write_l2_over_k0_s",
        "delta_s_post_l2_over_k0_s",
        "carry_write_interaction_over_k0_s_sq",
        "composition_relative_residual",
    )

    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            bucket = [
                r
                for r in rows
                if str(r["role"]) == role
                and int(r["relative_coordinate"]) == k
            ]

            require(
                bool(bucket),
                f"EMPTY_ROLE_COORDINATE:{role}:{k}",
            )

            trajectory[role][str(k)] = {
                metric: aggregate(
                    [float(r[metric]) for r in bucket]
                )
                for metric in metrics
            }

    return trajectory


def make_summary(
    rows: Sequence[Mapping[str, Any]],
    common_cohort: frozenset[int],
) -> dict[str, Any]:
    require(
        len(rows)
        == EXPECTED_PAIR_ROLE_COUNT * len(RELATIVE_COORDINATES),
        "ROW_COUNT_MISMATCH",
    )

    full = aggregate_trajectory(rows)

    common_rows = [
        r
        for r in rows
        if bool(r["in_common_ddsssss_cohort"])
    ]

    require(
        len(common_rows)
        == EXPECTED_COMMON_COHORT_COUNT
        * 2
        * len(RELATIVE_COORDINATES),
        "COMMON_COHORT_ROW_COUNT_MISMATCH",
    )

    common = aggregate_trajectory(common_rows)

    km1 = [
        r for r in rows
        if int(r["relative_coordinate"]) == -1
    ]

    require(
        all(
            bool(r["carry_exact_equal"])
            and bool(r["write_exact_equal"])
            and bool(r["s_post_exact_equal"])
            for r in km1
        ),
        "SUMMARY_K_MINUS_1_COMPONENT_IDENTITY_FAILURE",
    )

    return {
        "schema_version":
            "k0-rvg-carry-write-decomposition-summary-v1",
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count":
            len(common_cohort),
        "trajectory_row_count": len(rows),
        "primary_layer": PRIMARY_LAYER,
        "relative_coordinates":
            list(RELATIVE_COORDINATES),
        "execution_protocol": EXECUTION_PROTOCOL,
        "recurrence":
            "S_post = carry + write; carry = G * S_prev; write = W",
        "metrics": [
            "delta_carry_l2",
            "delta_write_l2",
            "delta_s_post_l2",
            "carry_write_interaction",
        ],
        "interaction_definition":
            "2 * dot(delta_carry, delta_write)",
        "full_336_trajectory": full,
        "common_330_ddsssss_trajectory": common,
        "max_composition_relative_residual":
            max(
                float(r["composition_relative_residual"])
                for r in rows
            ),
        "source_snapshot_dtype": "torch.float32",
        "metric_accumulation_dtype": "torch.float64",
        "raw_state_vectors_persisted": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "pca_probe_or_learned_geometry_executed": False,
        "carry_write_interaction_dot_product_computed": True,
        "training_executed": False,
        "causal_intervention_executed": False,
    }


def json_bytes(obj: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(obj),
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
        json_bytes(row)
        for row in rows
    )


def execute(
    root: Path,
    magnitude: Any,
    base: Any,
    repo: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    common_cohort: frozenset[int],
    signatures: Mapping[tuple[int, str], str],
    handoff_path: Path,
    output_dir: Path,
) -> None:
    require(
        handoff_path.is_file(),
        f"HANDOFF_MISSING:{handoff_path}",
    )

    final_dir = output_dir.resolve()
    partial_dir = Path(str(final_dir) + ".partial")

    require(
        not final_dir.exists(),
        f"OUTPUT_DIR_EXISTS:{final_dir}",
    )
    require(
        not partial_dir.exists(),
        f"PARTIAL_OUTPUT_DIR_EXISTS:{partial_dir}",
    )

    (
        observer,
        k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        snapshot,
    ) = base.resolve_runtime(
        root,
        handoff_path,
    )

    rows: list[dict[str, Any]] = []
    forward_count = 0

    for n, row in enumerate(plan, start=1):
        idx = int(row["local_template_index"])
        role = str(row["role"])
        anchor = int(row["anchor"])
        cutoff = anchor + POST_HORIZON + 1

        matched_prefix = tuple(
            row["matched_ids"][:cutoff]
        )
        swapped_prefix = tuple(
            row["swapped_ids"][:cutoff]
        )

        require(
            len(matched_prefix) == cutoff
            and len(swapped_prefix) == cutoff,
            f"PREFIX_LENGTH_FAILURE:{idx}:{role}",
        )

        require(
            matched_prefix[:anchor]
            == swapped_prefix[:anchor],
            f"PREANCHOR_PREFIX_MISMATCH:{idx}:{role}",
        )

        matched = capture_components(
            base,
            observer,
            model,
            layer_map,
            binding,
            matched_prefix,
            row["targets"],
        )
        forward_count += 1

        swapped = capture_components(
            base,
            observer,
            model,
            layer_map,
            binding,
            swapped_prefix,
            row["targets"],
        )
        forward_count += 1

        rows.extend(
            metric_rows_for_pair(
                row,
                matched,
                swapped,
                common_cohort,
                signatures[(idx, role)],
            )
        )

        if n % 16 == 0 or n == len(plan):
            print(
                f"PROGRESS pair_roles={n}/{len(plan)} "
                f"model_forwards={forward_count}",
                flush=True,
            )

    require(
        forward_count == EXPECTED_FORWARD_COUNT,
        "FORWARD_COUNT_MISMATCH",
    )

    summary = make_summary(
        rows,
        common_cohort,
    )

    partial_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metrics_path = (
        partial_dir
        / "carry_write_metrics.jsonl"
    )
    summary_path = (
        partial_dir
        / "summary.json"
    )
    manifest_path = (
        partial_dir
        / "execution_manifest.json"
    )

    metrics_path.write_bytes(
        jsonl_bytes(rows)
    )
    summary_path.write_bytes(
        json_bytes(summary)
    )

    manifest = {
        "schema_version":
            "k0-rvg-carry-write-decomposition-execution-manifest-v1",
        "runtime_git_head": repo["head"],
        "runtime_branch": repo["branch"],
        "parent_freeze_commit":
            PARENT_FREEZE_COMMIT,
        "frozen_token_evidence_commit":
            base.FROZEN_COMMIT,
        "execution_protocol":
            EXECUTION_PROTOCOL,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count":
            EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count":
            EXPECTED_COMMON_COHORT_COUNT,
        "model_forward_count":
            forward_count,
        "primary_layer": PRIMARY_LAYER,
        "relative_coordinates":
            list(RELATIVE_COORDINATES),
        "recurrence":
            "S_post = carry + write; carry = G * S_prev; write = W",
        "interaction_definition":
            "2 * dot(delta_carry, delta_write)",
        "source_snapshot_dtype":
            "torch.float32",
        "metric_accumulation_dtype":
            "torch.float64",
        "raw_state_vectors_persisted":
            False,
        "handoff_zip_sha256":
            handoff["zip_sha256"],
        "checkpoint_sha256":
            handoff["checkpoint_sha256"],
        "encoder_canonical_digest":
            encoder["canonical_digest"],
        "encoder_raw_concat_digest":
            encoder["raw_concat_digest"],
        "hf_model": k2s.HF_MODEL,
        "hf_revision": k2s.HF_REVISION,
        "hf_local_files_only": True,
        "scientific_model_forward_executed":
            True,
        "scientific_recurrent_state_read":
            True,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "pca_probe_or_learned_geometry_executed":
            False,
        "carry_write_interaction_dot_product_computed":
            True,
        "training_executed": False,
        "causal_intervention_executed":
            False,
        "runner_rel":
            Path(__file__)
            .resolve()
            .relative_to(root)
            .as_posix(),
        "runner_sha256":
            sha256_bytes(
                Path(__file__).read_bytes()
            ),
        "outputs": {
            "carry_write_metrics.jsonl":
                sha256_bytes(
                    metrics_path.read_bytes()
                ),
            "summary.json":
                sha256_bytes(
                    summary_path.read_bytes()
                ),
        },
    }

    manifest_path.write_bytes(
        json_bytes(manifest)
    )

    os.replace(
        partial_dir,
        final_dir,
    )

    common = summary[
        "common_330_ddsssss_trajectory"
    ]

    print(
        "PASS_CARRY_WRITE_DECOMPOSITION_EXECUTION"
    )
    print("output_dir =", final_dir)
    print(
        "model_forward_count =",
        forward_count,
    )

    for role in ("corr", "ctrl"):
        for k in (1, 2, 3):
            t = common[role][str(k)]

            print(
                f"{role}_k{k}_median "
                f"carry={t['delta_carry_l2_over_k0_s']['median']} "
                f"write={t['delta_write_l2_over_k0_s']['median']} "
                f"state={t['delta_s_post_l2_over_k0_s']['median']} "
                f"interaction="
                f"{t['carry_write_interaction_over_k0_s_sq']['median']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static-preflight",
        action="store_true",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
    )
    parser.add_argument(
        "--handoff",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    args = parser.parse_args()

    require(
        args.static_preflight ^ args.execute,
        "SELECT_EXACTLY_ONE_MODE",
    )

    root = Path.cwd().resolve()

    magnitude = load_magnitude_runner(root)
    base = magnitude.load_base(root)

    repo, parent = authenticate_parent(
        root,
        magnitude,
        base,
    )

    (
        plan,
        common_cohort,
        signatures,
    ) = build_plan_and_common_cohort(
        root,
        magnitude,
        base,
    )

    print(
        "=== CARRY / WRITE DECOMPOSITION PLAN ==="
    )
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print(
        "parent_freeze_commit =",
        PARENT_FREEZE_COMMIT,
    )
    print(
        "item_count =",
        EXPECTED_ITEM_COUNT,
    )
    print(
        "pair_role_count =",
        len(plan),
    )
    print(
        "common_ddsssss_item_count =",
        len(common_cohort),
    )
    print(
        "primary_layer =",
        PRIMARY_LAYER,
    )
    print(
        "relative_coordinates =",
        list(RELATIVE_COORDINATES),
    )
    print(
        "execution_protocol =",
        EXECUTION_PROTOCOL,
    )
    print(
        "recurrence = S_post = carry + write"
    )
    print(
        "carry = G * S_prev"
    )
    print(
        "write = W"
    )
    print(
        "interaction = 2 * dot(delta_carry, delta_write)"
    )
    print(
        "tokenizer_invoked = False"
    )
    print(
        "scientific_model_forward_executed = False"
    )

    if args.static_preflight:
        print(
            "PASS_CARRY_WRITE_STATIC_PREFLIGHT"
        )
        return 0

    require(
        args.handoff is not None,
        "EXECUTE_REQUIRES_HANDOFF",
    )
    require(
        args.output_dir is not None,
        "EXECUTE_REQUIRES_OUTPUT_DIR",
    )

    execute(
        root,
        magnitude,
        base,
        repo,
        plan,
        common_cohort,
        signatures,
        args.handoff.resolve(),
        args.output_dir,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DecompositionError as exc:
        print(
            f"BLOCKED: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception as exc:
        print(
            f"BLOCKED_UNEXPECTED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise