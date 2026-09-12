"""K0-RVG divergence-aligned layer-23 S_post magnitude trajectory.

Scope:
- frozen active token IDs only; no tokenizer execution,
- same matched-vs-swapped divergence anchors as the validated hash audit,
- same equal-length prefix replay through k=+6,
- same layer-23 CPU slow-path recurrence observer,
- measure S_post magnitude separation only,
- no logits, heads, PCA, probes, cosine geometry, training, or intervention.
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


PARENT_FREEZE_COMMIT = "c96b58e72f7deceb02f734c5e6c2aa5ccd23ac6a"
PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_divergence_aligned_layer23_state_audit_e322289_v1/"
    "summary.json"
)
BASE_RUNNER_REL = "scripts/longterm_k0_rvg_divergence_aligned_state_audit.py"
BASE_RUNNER_SHA256 = "926a540343e641063d2126880982a0d4c65f2fdd6ca8b553bce52c2cb9af0672"

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_FORWARD_COUNT = 1344
PRIMARY_LAYER = 23
POST_HORIZON = 6
RELATIVE_COORDINATES = tuple(range(-1, POST_HORIZON + 1))
EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"


class TrajectoryError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise TrajectoryError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def import_module(path: Path, name: str):
    require(path.is_file(), f"MODULE_MISSING:{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"MODULE_SPEC_FAILURE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base(root: Path):
    path = root / BASE_RUNNER_REL
    require(
        sha256_bytes(path.read_bytes()) == BASE_RUNNER_SHA256,
        "BASE_RUNNER_SHA256_MISMATCH",
    )
    return import_module(path, "k0_rvg_divergence_hash_base")


def authenticate_parent(root: Path, base: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = base.authenticate_repo(root)

    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", PARENT_FREEZE_COMMIT, repo["head"]],
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
    require(current.read_bytes() == raw, "PARENT_SUMMARY_WORKTREE_DRIFT")

    require(
        parent.get("schema_version")
        == "k0-rvg-divergence-aligned-layer23-state-audit-summary-v1",
        "PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(parent.get("item_count") == EXPECTED_ITEM_COUNT, "PARENT_ITEM_COUNT_MISMATCH")
    require(
        parent.get("pair_role_count") == EXPECTED_PAIR_ROLE_COUNT,
        "PARENT_PAIR_ROLE_COUNT_MISMATCH",
    )
    require(parent.get("primary_layer") == PRIMARY_LAYER, "PARENT_LAYER_MISMATCH")
    require(
        parent.get("relative_coordinates") == list(RELATIVE_COORDINATES),
        "PARENT_COORDINATES_MISMATCH",
    )
    require(
        parent.get("execution_protocol") == EXECUTION_PROTOCOL,
        "PARENT_PROTOCOL_MISMATCH",
    )
    require(
        parent.get("pair_role_state_identity_through_k_plus_6_count") == 0,
        "PARENT_IDENTITY_RESULT_MISMATCH",
    )
    require(
        parent.get("first_state_hash_difference_histogram")
        == {"corr": {"0": 336}, "ctrl": {"0": 336}},
        "PARENT_FIRST_DIFFERENCE_RESULT_MISMATCH",
    )

    token_contract = parent.get("token_contract", {})
    require(
        token_contract.get("k_minus_1_equal_pair_count") == EXPECTED_PAIR_ROLE_COUNT,
        "PARENT_K_MINUS_1_CONTRACT_MISMATCH",
    )
    require(
        token_contract.get("k_zero_different_pair_count") == EXPECTED_PAIR_ROLE_COUNT,
        "PARENT_K_ZERO_CONTRACT_MISMATCH",
    )

    return repo, parent


def build_plan(root: Path, base: Any):
    active = base.load_active_artifact(root)
    plan = base.build_static_plan(active)

    require(len(plan) == EXPECTED_PAIR_ROLE_COUNT, "PLAN_COUNT_MISMATCH")

    for row in plan:
        require(
            tuple(row["targets"])
            == tuple(int(row["anchor"]) + k for k in RELATIVE_COORDINATES),
            f"PLAN_TARGET_MISMATCH:{row['local_template_index']}:{row['role']}",
        )
        require(
            int(row["common_post_horizon"]) >= POST_HORIZON,
            f"PLAN_POST_SUPPORT_FAILURE:{row['local_template_index']}:{row['role']}",
        )

    return plan


def capture_s_post(
    base: Any,
    observer: Any,
    model: Any,
    layer_map: Mapping[int, int],
    binding: Any,
    token_ids: Sequence[int],
    targets: Sequence[int],
) -> dict[int, Any]:
    target_set = tuple(int(v) for v in targets)

    collector = observer.RawRecurrenceCollector(
        binding,
        layer_map,
        target_set,
    )

    with collector.capture():
        base.direct_backbone_forward(model, token_ids)

    require(collector.records is not None, "CAPTURE_RECORDS_MISSING")

    expected = {
        (layer, token)
        for layer in range(24)
        for token in target_set
    }
    require(
        set(collector.records) == expected,
        "CAPTURE_COORDINATE_SET_MISMATCH",
    )

    out: dict[int, Any] = {}

    for token in target_set:
        record = collector.records[(PRIMARY_LAYER, token)]
        audit = observer.validate_recurrence_record(record)

        require(
            audit.get("recurrence_exact") == "PASS_EXACT",
            "RECURRENCE_EXACT_FAILURE",
        )

        out[token] = record.s_post

    return out


def metric_rows_for_pair(
    row: Mapping[str, Any],
    matched_states: Mapping[int, Any],
    swapped_states: Mapping[int, Any],
) -> list[dict[str, Any]]:
    import torch

    idx = int(row["local_template_index"])
    role = str(row["role"])
    anchor = int(row["anchor"])
    m_ids = row["matched_ids"]
    s_ids = row["swapped_ids"]

    provisional: list[dict[str, Any]] = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k
        m_raw = matched_states[token]
        s_raw = swapped_states[token]

        require(
            tuple(m_raw.shape) == (1, 1536, 16)
            and tuple(s_raw.shape) == (1, 1536, 16),
            f"S_POST_SHAPE_MISMATCH:{idx}:{role}:{k}",
        )
        require(
            str(m_raw.dtype) == "torch.float32"
            and str(s_raw.dtype) == "torch.float32",
            f"S_POST_DTYPE_MISMATCH:{idx}:{role}:{k}",
        )

        exact_equal = bool(torch.equal(m_raw, s_raw))

        if k == -1:
            require(
                int(m_ids[token]) == int(s_ids[token]),
                f"K_MINUS_1_TOKEN_MISMATCH:{idx}:{role}",
            )
            require(exact_equal, f"K_MINUS_1_STATE_MISMATCH:{idx}:{role}")

        if k == 0:
            require(
                int(m_ids[token]) != int(s_ids[token]),
                f"K_ZERO_TOKEN_NOT_DIVERGENT:{idx}:{role}",
            )
            require(not exact_equal, f"K_ZERO_STATE_NOT_DIVERGENT:{idx}:{role}")

        # Metrics are accumulated in float64 from the validated float32 snapshots.
        m = m_raw.to(dtype=torch.float64)
        s = s_raw.to(dtype=torch.float64)
        delta = m - s

        m_l2 = float(torch.linalg.vector_norm(m).item())
        s_l2 = float(torch.linalg.vector_norm(s).item())
        delta_l2 = float(torch.linalg.vector_norm(delta).item())
        delta_rms = float(torch.sqrt(torch.mean(delta * delta)).item())

        scale = 0.5 * (m_l2 + s_l2)
        relative_l2 = delta_l2 / max(scale, 1e-12)

        require(
            all(
                math.isfinite(v)
                for v in (m_l2, s_l2, delta_l2, delta_rms, relative_l2)
            ),
            f"NONFINITE_METRIC:{idx}:{role}:{k}",
        )

        provisional.append(
            {
                "schema_version": "k0-rvg-divergence-magnitude-trajectory-row-v1",
                "local_template_index": idx,
                "stable_item_id": row["stable_item_id"],
                "role": role,
                "relative_coordinate": k,
                "token_index": token,
                "divergence_anchor_token_index": anchor,
                "matched_token_id": int(m_ids[token]),
                "swapped_token_id": int(s_ids[token]),
                "s_post_exact_equal": exact_equal,
                "matched_s_post_l2": m_l2,
                "swapped_s_post_l2": s_l2,
                "delta_s_post_l2": delta_l2,
                "delta_s_post_rms": delta_rms,
                "relative_delta_s_post_l2": relative_l2,
                "metric_accumulation_dtype": "torch.float64",
                "source_snapshot_dtype": "torch.float32",
            }
        )

    k0 = next(
        float(r["delta_s_post_l2"])
        for r in provisional
        if r["relative_coordinate"] == 0
    )
    require(k0 > 0.0 and math.isfinite(k0), f"K_ZERO_DELTA_INVALID:{idx}:{role}")

    for r in provisional:
        r["delta_s_post_l2_over_k0"] = float(r["delta_s_post_l2"]) / k0

    return provisional


def aggregate(values: Sequence[float]) -> dict[str, float | int]:
    vals = [float(v) for v in values]
    require(bool(vals), "EMPTY_AGGREGATE")
    require(all(math.isfinite(v) for v in vals), "NONFINITE_AGGREGATE_INPUT")

    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def make_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(
        len(rows) == EXPECTED_PAIR_ROLE_COUNT * len(RELATIVE_COORDINATES),
        "TRAJECTORY_ROW_COUNT_MISMATCH",
    )

    by_role_k: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    by_pair: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)

    for row in rows:
        role = str(row["role"])
        k = int(row["relative_coordinate"])
        by_role_k[(role, k)].append(row)
        by_pair[(int(row["local_template_index"]), role)].append(row)

    trajectory: dict[str, dict[str, Any]] = {"corr": {}, "ctrl": {}}

    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            bucket = by_role_k[(role, k)]
            require(len(bucket) == EXPECTED_ITEM_COUNT, f"ROLE_K_COUNT_MISMATCH:{role}:{k}")

            trajectory[role][str(k)] = {
                "delta_s_post_l2": aggregate(
                    [float(r["delta_s_post_l2"]) for r in bucket]
                ),
                "delta_s_post_rms": aggregate(
                    [float(r["delta_s_post_rms"]) for r in bucket]
                ),
                "relative_delta_s_post_l2": aggregate(
                    [float(r["relative_delta_s_post_l2"]) for r in bucket]
                ),
                "delta_s_post_l2_over_k0": aggregate(
                    [float(r["delta_s_post_l2_over_k0"]) for r in bucket]
                ),
            }

    stepwise: dict[str, dict[str, dict[str, int]]] = {
        "corr": {},
        "ctrl": {},
    }

    for role in ("corr", "ctrl"):
        for k in range(0, POST_HORIZON + 1):
            inc = dec = eq = 0

            for idx in range(EXPECTED_ITEM_COUNT):
                seq = sorted(
                    by_pair[(idx, role)],
                    key=lambda r: int(r["relative_coordinate"]),
                )
                mapping = {
                    int(r["relative_coordinate"]): float(r["delta_s_post_l2"])
                    for r in seq
                }
                prev = mapping[k - 1]
                cur = mapping[k]

                if cur > prev:
                    inc += 1
                elif cur < prev:
                    dec += 1
                else:
                    eq += 1

            require(
                inc + dec + eq == EXPECTED_ITEM_COUNT,
                f"STEPWISE_COUNT_MISMATCH:{role}:{k}",
            )

            stepwise[role][str(k)] = {
                "increase_vs_previous": inc,
                "decrease_vs_previous": dec,
                "equal_vs_previous": eq,
            }

    km1 = [r for r in rows if int(r["relative_coordinate"]) == -1]
    k0 = [r for r in rows if int(r["relative_coordinate"]) == 0]

    require(all(bool(r["s_post_exact_equal"]) for r in km1), "SUMMARY_K_MINUS_1_NOT_EQUAL")
    require(all(not bool(r["s_post_exact_equal"]) for r in k0), "SUMMARY_K_ZERO_NOT_DIFFERENT")

    return {
        "schema_version": "k0-rvg-divergence-magnitude-trajectory-summary-v1",
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "trajectory_row_count": len(rows),
        "primary_layer": PRIMARY_LAYER,
        "state_field": "S_post",
        "relative_coordinates": list(RELATIVE_COORDINATES),
        "execution_protocol": EXECUTION_PROTOCOL,
        "primary_metric": "delta_s_post_l2",
        "supporting_metrics": [
            "delta_s_post_rms",
            "relative_delta_s_post_l2",
            "delta_s_post_l2_over_k0",
        ],
        "metric_accumulation_dtype": "torch.float64",
        "source_snapshot_dtype": "torch.float32",
        "k_minus_1_exact_identity_count": sum(
            bool(r["s_post_exact_equal"]) for r in km1
        ),
        "k_zero_exact_difference_count": sum(
            not bool(r["s_post_exact_equal"]) for r in k0
        ),
        "trajectory": trajectory,
        "stepwise_delta_l2_change_counts": stepwise,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "geometry_analysis_executed": False,
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


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(json_bytes(row) for row in rows)


def execute(
    root: Path,
    base: Any,
    repo: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    handoff_path: Path,
    output_dir: Path,
) -> None:
    require(handoff_path.is_file(), f"HANDOFF_MISSING:{handoff_path}")

    final_dir = output_dir.resolve()
    partial_dir = Path(str(final_dir) + ".partial")

    require(not final_dir.exists(), f"OUTPUT_DIR_EXISTS:{final_dir}")
    require(not partial_dir.exists(), f"PARTIAL_OUTPUT_DIR_EXISTS:{partial_dir}")

    (
        observer,
        k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        snapshot,
    ) = base.resolve_runtime(root, handoff_path)

    rows: list[dict[str, Any]] = []
    forward_count = 0

    for n, row in enumerate(plan, start=1):
        anchor = int(row["anchor"])
        cutoff = anchor + POST_HORIZON + 1

        matched_prefix = tuple(row["matched_ids"][:cutoff])
        swapped_prefix = tuple(row["swapped_ids"][:cutoff])

        require(
            len(matched_prefix) == cutoff
            and len(swapped_prefix) == cutoff,
            f"PREFIX_LENGTH_FAILURE:{row['local_template_index']}:{row['role']}",
        )
        require(
            matched_prefix[:anchor] == swapped_prefix[:anchor],
            f"PREANCHOR_PREFIX_MISMATCH:{row['local_template_index']}:{row['role']}",
        )

        matched = capture_s_post(
            base,
            observer,
            model,
            layer_map,
            binding,
            matched_prefix,
            row["targets"],
        )
        forward_count += 1

        swapped = capture_s_post(
            base,
            observer,
            model,
            layer_map,
            binding,
            swapped_prefix,
            row["targets"],
        )
        forward_count += 1

        rows.extend(metric_rows_for_pair(row, matched, swapped))

        if n % 16 == 0 or n == len(plan):
            print(
                f"PROGRESS pair_roles={n}/{len(plan)} "
                f"model_forwards={forward_count}",
                flush=True,
            )

    require(forward_count == EXPECTED_FORWARD_COUNT, "FORWARD_COUNT_MISMATCH")

    summary = make_summary(rows)

    partial_dir.mkdir(parents=True, exist_ok=False)

    metrics_path = partial_dir / "trajectory_metrics.jsonl"
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "execution_manifest.json"

    metrics_path.write_bytes(jsonl_bytes(rows))
    summary_path.write_bytes(json_bytes(summary))

    manifest = {
        "schema_version": "k0-rvg-divergence-magnitude-trajectory-execution-manifest-v1",
        "runtime_git_head": repo["head"],
        "runtime_branch": repo["branch"],
        "parent_freeze_commit": PARENT_FREEZE_COMMIT,
        "frozen_token_evidence_commit": base.FROZEN_COMMIT,
        "execution_protocol": EXECUTION_PROTOCOL,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "model_forward_count": forward_count,
        "primary_layer": PRIMARY_LAYER,
        "state_field": "S_post",
        "relative_coordinates": list(RELATIVE_COORDINATES),
        "primary_metric": "delta_s_post_l2",
        "metric_accumulation_dtype": "torch.float64",
        "source_snapshot_dtype": "torch.float32",
        "raw_state_vectors_persisted": False,
        "handoff_zip_sha256": handoff["zip_sha256"],
        "checkpoint_sha256": handoff["checkpoint_sha256"],
        "encoder_canonical_digest": encoder["canonical_digest"],
        "encoder_raw_concat_digest": encoder["raw_concat_digest"],
        "hf_model": k2s.HF_MODEL,
        "hf_revision": k2s.HF_REVISION,
        "hf_local_files_only": True,
        "scientific_model_forward_executed": True,
        "scientific_recurrent_state_read": True,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "geometry_analysis_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "runner_rel": Path(__file__).resolve().relative_to(root).as_posix(),
        "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "outputs": {
            "trajectory_metrics.jsonl": sha256_bytes(metrics_path.read_bytes()),
            "summary.json": sha256_bytes(summary_path.read_bytes()),
        },
    }

    manifest_path.write_bytes(json_bytes(manifest))
    os.replace(partial_dir, final_dir)

    print("PASS_DIVERGENCE_MAGNITUDE_TRAJECTORY_EXECUTION")
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print(
        "corr_k_plus_6_over_k0_median =",
        summary["trajectory"]["corr"]["6"]["delta_s_post_l2_over_k0"]["median"],
    )
    print(
        "ctrl_k_plus_6_over_k0_median =",
        summary["trajectory"]["ctrl"]["6"]["delta_s_post_l2_over_k0"]["median"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    require(
        args.static_preflight ^ args.execute,
        "SELECT_EXACTLY_ONE_MODE",
    )

    root = Path.cwd().resolve()
    base = load_base(root)
    repo, parent = authenticate_parent(root, base)
    plan = build_plan(root, base)

    print("=== DIVERGENCE MAGNITUDE TRAJECTORY PLAN ===")
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print("parent_freeze_commit =", PARENT_FREEZE_COMMIT)
    print("item_count =", EXPECTED_ITEM_COUNT)
    print("pair_role_count =", len(plan))
    print("primary_layer =", PRIMARY_LAYER)
    print("state_field = S_post")
    print("relative_coordinates =", list(RELATIVE_COORDINATES))
    print("execution_protocol =", EXECUTION_PROTOCOL)
    print("primary_metric = delta_s_post_l2")
    print("tokenizer_invoked = False")
    print("scientific_model_forward_executed = False")

    if args.static_preflight:
        print("PASS_DIVERGENCE_MAGNITUDE_STATIC_PREFLIGHT")
        return 0

    require(args.handoff is not None, "EXECUTE_REQUIRES_HANDOFF")
    require(args.output_dir is not None, "EXECUTE_REQUIRES_OUTPUT_DIR")

    execute(
        root,
        base,
        repo,
        plan,
        args.handoff.resolve(),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrajectoryError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"BLOCKED_UNEXPECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise