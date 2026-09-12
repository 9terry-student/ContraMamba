"""K0-RVG layer-23 write-factor symmetric decomposition.

Decomposes the validated Mamba write term

    W = discrete_B * hidden_states

between matched and swapped executions using the exact symmetric bilinear
identity

    delta_W = Q_U + Q_D

    Q_U = mean(D_m, D_s) * (U_m - U_s)
    Q_D = mean(U_m, U_s) * (D_m - D_s)

where D = discrete_B and U = conv-activated hidden_states.

Scope:
- frozen token IDs only,
- same divergence anchors and equal-length-prefix protocol,
- CPU sequential Transformers 5.12.1 Mamba slow path,
- layer 23, k=-1..+6,
- common DDSSSSS 330-item cohort primary,
- no tokenizer, logits, task heads, training, intervention, PCA, or probes,
- raw vectors are not persisted.
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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence


PARENT_FREEZE_COMMIT = "aecd6d93cbb4edaa6cae6f68418a30bfecda1ceb"
PARENT_SUMMARY_REL = (
    "reports/"
    "longterm_k0_rvg_carry_write_decomposition_b1fee31_v1/"
    "summary.json"
)
PARENT_SUMMARY_SHA256 = (
    "437c615b7fd63f6f49ca6f531202371fa084f528ec6f4f52aafe133b0236cc91"
)

CARRY_WRITE_RUNNER_REL = (
    "scripts/longterm_k0_rvg_carry_write_decomposition.py"
)
CARRY_WRITE_RUNNER_SHA256 = (
    "2f9e70e25216597544da41defc3a9b6cc73572e4901691516172b0f3a7339541"
)

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FORWARD_COUNT = 1344

PRIMARY_LAYER = 23
RELATIVE_COORDINATES = tuple(range(-1, 7))
POST_HORIZON = 6
COMMON_SIGNATURE = "DDSSSSS"

EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"
FACTORIZATION = "W = discrete_B * hidden_states"
DECOMPOSITION = (
    "delta_W = Q_U + Q_D; "
    "Q_U = mean(D_m,D_s)*(U_m-U_s); "
    "Q_D = mean(U_m,U_s)*(D_m-D_s)"
)

COMPOSITION_RTOL = 1e-5


class WriteFactorError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise WriteFactorError(message)


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


def load_parent_runner(root: Path):
    path = root / CARRY_WRITE_RUNNER_REL

    require(
        sha256_bytes(path.read_bytes()) == CARRY_WRITE_RUNNER_SHA256,
        "CARRY_WRITE_RUNNER_SHA256_MISMATCH",
    )

    return import_module(
        path,
        "k0_rvg_carry_write_parent",
    )


def authenticate_parent(root: Path, parent: Any, base: Any):
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

    require(
        sha256_bytes(raw) == PARENT_SUMMARY_SHA256,
        "PARENT_SUMMARY_SHA256_MISMATCH",
    )

    current = root / PARENT_SUMMARY_REL
    require(current.is_file(), "PARENT_SUMMARY_MISSING")
    require(
        current.read_bytes() == raw,
        "PARENT_SUMMARY_WORKTREE_DRIFT",
    )

    summary = json.loads(raw)

    require(
        summary.get("schema_version")
        == "k0-rvg-carry-write-decomposition-summary-v1",
        "PARENT_SCHEMA_MISMATCH",
    )
    require(
        summary.get("item_count") == EXPECTED_ITEM_COUNT,
        "PARENT_ITEM_COUNT_MISMATCH",
    )
    require(
        summary.get("pair_role_count") == EXPECTED_PAIR_ROLE_COUNT,
        "PARENT_PAIR_ROLE_COUNT_MISMATCH",
    )
    require(
        summary.get("common_ddsssss_item_count") == EXPECTED_COMMON_COUNT,
        "PARENT_COMMON_COUNT_MISMATCH",
    )
    require(
        summary.get("primary_layer") == PRIMARY_LAYER,
        "PARENT_LAYER_MISMATCH",
    )
    require(
        summary.get("relative_coordinates")
        == list(RELATIVE_COORDINATES),
        "PARENT_COORDINATES_MISMATCH",
    )
    require(
        summary.get("execution_protocol") == EXECUTION_PROTOCOL,
        "PARENT_PROTOCOL_MISMATCH",
    )

    return repo, summary


def snapshot(value: Any, role: str):
    import torch

    require(
        isinstance(value, torch.Tensor),
        f"{role}_NOT_TENSOR",
    )
    require(
        bool(torch.isfinite(value).all().item()),
        f"{role}_NONFINITE",
    )

    out = value.detach().cpu().contiguous().clone()

    require(
        out.device.type == "cpu",
        f"{role}_NOT_CPU",
    )
    require(
        str(out.dtype) == "torch.float32",
        f"{role}_DTYPE_MISMATCH",
    )

    return out


class WriteFactorCollector:
    """Single-use layer-23 observer for D, U, and W at recurrence update."""

    def __init__(
        self,
        binding: Any,
        layer_map: Mapping[int, int],
        target_indices: Sequence[int],
    ):
        self.binding = binding
        self.layer_map = dict(layer_map)
        self.targets = frozenset(int(v) for v in target_indices)
        self.records: dict[int, dict[str, Any]] | None = None
        self.prior_trace = None
        self.used = False

    def _trace(self, frame: Any, event: str, arg: Any):
        if (
            frame.f_code is not self.binding.code
            or event != "line"
            or frame.f_lineno != self.binding.update_line
        ):
            return self._trace

        mixer = frame.f_locals.get("self")
        layer = self.layer_map.get(id(mixer))

        if layer != PRIMARY_LAYER:
            return self._trace

        i = frame.f_locals.get("i")

        require(
            type(i) is int and i >= 0,
            "AMBIGUOUS_TOKEN_INDEX",
        )

        if i not in self.targets:
            return self._trace

        require(self.records is not None, "TRACE_NOT_ACTIVE")
        require(i not in self.records, "DUPLICATE_TOKEN_CAPTURE")

        discrete_B = frame.f_locals.get("discrete_B")
        hidden_states = frame.f_locals.get("hidden_states")
        deltaB_u = frame.f_locals.get("deltaB_u")

        require(discrete_B is not None, "DISCRETE_B_MISSING")
        require(hidden_states is not None, "HIDDEN_STATES_MISSING")
        require(deltaB_u is not None, "DELTAB_U_MISSING")

        D = snapshot(
            discrete_B[:, :, i, :],
            "DISCRETE_B",
        )
        U = snapshot(
            hidden_states[:, :, i],
            "HIDDEN_STATES",
        )
        W = snapshot(
            deltaB_u[:, :, i, :],
            "DELTAB_U",
        )

        require(
            tuple(D.shape) == (1, 1536, 16),
            "DISCRETE_B_SHAPE_MISMATCH",
        )
        require(
            tuple(U.shape) == (1, 1536),
            "HIDDEN_STATES_SHAPE_MISMATCH",
        )
        require(
            tuple(W.shape) == (1, 1536, 16),
            "WRITE_SHAPE_MISMATCH",
        )

        reconstructed = D * U[:, :, None]

        require(
            torch_equal(reconstructed, W),
            "WRITE_EXACT_FACTORIZATION_FAILURE",
        )

        self.records[i] = {
            "D": D,
            "U": U,
            "W": W,
        }

        return self._trace

    @contextmanager
    def capture(self):
        require(not self.used, "TRACE_COLLECTOR_REUSE")

        self.records = {}
        self.prior_trace = sys.gettrace()
        sys.settrace(self._trace)

        completed = False
        try:
            yield self
            completed = True
        finally:
            sys.settrace(self.prior_trace)
            self.used = True

        if completed:
            require(
                set(self.records) == set(self.targets),
                "CAPTURE_COORDINATE_SET_MISMATCH",
            )


def torch_equal(a: Any, b: Any) -> bool:
    import torch
    return bool(torch.equal(a, b))


def capture_factors(
    base: Any,
    model: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    token_ids: Sequence[int],
    targets: Sequence[int],
):
    collector = WriteFactorCollector(
        binding,
        layer_map,
        targets,
    )

    with collector.capture():
        base.direct_backbone_forward(model, token_ids)

    require(
        collector.records is not None,
        "CAPTURE_RECORDS_MISSING",
    )

    return collector.records


def build_plan(root: Path, parent: Any):
    magnitude = parent.load_magnitude_runner(root)
    base = magnitude.load_base(root)

    plan, cohort, signatures = parent.build_plan_and_common_cohort(
        root,
        magnitude,
        base,
    )

    require(
        len(plan) == EXPECTED_PAIR_ROLE_COUNT,
        "PLAN_COUNT_MISMATCH",
    )
    require(
        len(cohort) == EXPECTED_COMMON_COUNT,
        "COMMON_COUNT_MISMATCH",
    )

    return magnitude, base, plan, cohort, signatures


def metric_rows_for_pair(
    row: Mapping[str, Any],
    matched: Mapping[int, Mapping[str, Any]],
    swapped: Mapping[int, Mapping[str, Any]],
    common_cohort: frozenset[int],
    signature: str,
):
    import torch

    idx = int(row["local_template_index"])
    role = str(row["role"])
    anchor = int(row["anchor"])

    provisional: list[dict[str, Any]] = []

    for k in RELATIVE_COORDINATES:
        token = anchor + k

        Dm32 = matched[token]["D"]
        Um32 = matched[token]["U"]
        Wm32 = matched[token]["W"]

        Ds32 = swapped[token]["D"]
        Us32 = swapped[token]["U"]
        Ws32 = swapped[token]["W"]

        d_equal = torch_equal(Dm32, Ds32)
        u_equal = torch_equal(Um32, Us32)
        w_equal = torch_equal(Wm32, Ws32)

        if k == -1:
            require(
                d_equal and u_equal and w_equal,
                f"K_MINUS_1_IDENTITY_FAILURE:{idx}:{role}",
            )

        Dm = Dm32.to(torch.float64)
        Um = Um32.to(torch.float64)
        Wm = Wm32.to(torch.float64)

        Ds = Ds32.to(torch.float64)
        Us = Us32.to(torch.float64)
        Ws = Ws32.to(torch.float64)

        dU = Um - Us
        dD = Dm - Ds
        dW = Wm - Ws

        mean_D = 0.5 * (Dm + Ds)
        mean_U = 0.5 * (Um + Us)

        q_u = mean_D * dU[:, :, None]
        q_d = mean_U[:, :, None] * dD

        reconstructed = q_u + q_d
        residual = dW - reconstructed

        q_u_l2 = float(torch.linalg.vector_norm(q_u).item())
        q_d_l2 = float(torch.linalg.vector_norm(q_d).item())
        dW_l2 = float(torch.linalg.vector_norm(dW).item())

        interaction = float(
            2.0 * torch.sum(q_u * q_d).item()
        )

        residual_l2 = float(
            torch.linalg.vector_norm(residual).item()
        )
        residual_relative = (
            residual_l2 / max(dW_l2, 1e-12)
        )

        values = (
            q_u_l2,
            q_d_l2,
            dW_l2,
            interaction,
            residual_l2,
            residual_relative,
        )

        require(
            all(math.isfinite(v) for v in values),
            f"NONFINITE_METRIC:{idx}:{role}:{k}",
        )

        require(
            residual_relative <= COMPOSITION_RTOL,
            (
                f"SYMMETRIC_COMPOSITION_RESIDUAL_TOO_LARGE:"
                f"{idx}:{role}:{k}:{residual_relative}"
            ),
        )

        provisional.append(
            {
                "schema_version":
                    "k0-rvg-write-factor-decomposition-row-v1",
                "local_template_index": idx,
                "stable_item_id": row["stable_item_id"],
                "role": role,
                "relative_coordinate": k,
                "token_index": token,
                "divergence_anchor_token_index": anchor,
                "token_equality_signature_k0_to_k6":
                    signature,
                "in_common_ddsssss_cohort":
                    idx in common_cohort,
                "discrete_b_exact_equal": d_equal,
                "hidden_states_exact_equal": u_equal,
                "write_exact_equal": w_equal,
                "q_u_l2": q_u_l2,
                "q_d_l2": q_d_l2,
                "delta_write_l2": dW_l2,
                "q_u_q_d_interaction": interaction,
                "composition_residual_l2":
                    residual_l2,
                "composition_relative_residual":
                    residual_relative,
                "source_snapshot_dtype":
                    "torch.float32",
                "metric_accumulation_dtype":
                    "torch.float64",
            }
        )

    k0_w = next(
        float(r["delta_write_l2"])
        for r in provisional
        if int(r["relative_coordinate"]) == 0
    )

    require(
        k0_w > 0.0 and math.isfinite(k0_w),
        f"K0_WRITE_MAGNITUDE_INVALID:{idx}:{role}",
    )

    k0_w_sq = k0_w * k0_w

    for r in provisional:
        r["q_u_l2_over_k0_w"] = (
            float(r["q_u_l2"]) / k0_w
        )
        r["q_d_l2_over_k0_w"] = (
            float(r["q_d_l2"]) / k0_w
        )
        r["delta_write_l2_over_k0_w"] = (
            float(r["delta_write_l2"]) / k0_w
        )
        r["interaction_over_k0_w_sq"] = (
            float(r["q_u_q_d_interaction"])
            / k0_w_sq
        )

    return provisional


def aggregate(values):
    vals = [float(v) for v in values]
    require(bool(vals), "EMPTY_AGGREGATE")

    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def aggregate_trajectory(rows):
    fields = (
        "q_u_l2",
        "q_d_l2",
        "delta_write_l2",
        "q_u_q_d_interaction",
        "q_u_l2_over_k0_w",
        "q_d_l2_over_k0_w",
        "delta_write_l2_over_k0_w",
        "interaction_over_k0_w_sq",
        "composition_relative_residual",
    )

    out = {"corr": {}, "ctrl": {}}

    for role in ("corr", "ctrl"):
        for k in RELATIVE_COORDINATES:
            bucket = [
                r for r in rows
                if r["role"] == role
                and int(r["relative_coordinate"]) == k
            ]

            require(
                bool(bucket),
                f"EMPTY_BUCKET:{role}:{k}",
            )

            out[role][str(k)] = {
                f: aggregate(float(r[f]) for r in bucket)
                for f in fields
            }

    return out


def make_summary(rows, cohort):
    require(
        len(rows)
        == EXPECTED_PAIR_ROLE_COUNT
        * len(RELATIVE_COORDINATES),
        "ROW_COUNT_MISMATCH",
    )

    common_rows = [
        r for r in rows
        if bool(r["in_common_ddsssss_cohort"])
    ]

    require(
        len(common_rows)
        == EXPECTED_COMMON_COUNT
        * 2
        * len(RELATIVE_COORDINATES),
        "COMMON_ROW_COUNT_MISMATCH",
    )

    km1 = [
        r for r in rows
        if int(r["relative_coordinate"]) == -1
    ]

    require(
        all(
            r["discrete_b_exact_equal"]
            and r["hidden_states_exact_equal"]
            and r["write_exact_equal"]
            for r in km1
        ),
        "K_MINUS_1_SUMMARY_IDENTITY_FAILURE",
    )

    return {
        "schema_version":
            "k0-rvg-write-factor-decomposition-summary-v1",
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count":
            len(cohort),
        "trajectory_row_count": len(rows),
        "primary_layer": PRIMARY_LAYER,
        "relative_coordinates":
            list(RELATIVE_COORDINATES),
        "execution_protocol":
            EXECUTION_PROTOCOL,
        "factorization":
            FACTORIZATION,
        "decomposition":
            DECOMPOSITION,
        "full_336_trajectory":
            aggregate_trajectory(rows),
        "common_330_ddsssss_trajectory":
            aggregate_trajectory(common_rows),
        "max_composition_relative_residual":
            max(
                float(r["composition_relative_residual"])
                for r in rows
            ),
        "source_snapshot_dtype":
            "torch.float32",
        "metric_accumulation_dtype":
            "torch.float64",
        "raw_vectors_persisted": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_probe_or_learned_geometry_executed":
            False,
    }


def json_bytes(obj):
    return (
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows):
    return b"".join(json_bytes(r) for r in rows)


def execute(
    root,
    parent,
    magnitude,
    base,
    repo,
    plan,
    cohort,
    signatures,
    handoff_path,
    output_dir,
):
    final_dir = output_dir.resolve()
    partial_dir = Path(str(final_dir) + ".partial")

    require(
        handoff_path.is_file(),
        f"HANDOFF_MISSING:{handoff_path}",
    )
    require(
        not final_dir.exists(),
        f"OUTPUT_DIR_EXISTS:{final_dir}",
    )
    require(
        not partial_dir.exists(),
        f"PARTIAL_OUTPUT_EXISTS:{partial_dir}",
    )

    (
        observer,
        k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        snapshot_info,
    ) = base.resolve_runtime(
        root,
        handoff_path,
    )

    require(
        binding.source_sha256
        == EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    rows = []
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

        matched = capture_factors(
            base,
            model,
            binding,
            layer_map,
            matched_prefix,
            row["targets"],
        )
        forward_count += 1

        swapped = capture_factors(
            base,
            model,
            binding,
            layer_map,
            swapped_prefix,
            row["targets"],
        )
        forward_count += 1

        rows.extend(
            metric_rows_for_pair(
                row,
                matched,
                swapped,
                cohort,
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

    summary = make_summary(rows, cohort)

    partial_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metrics_path = (
        partial_dir
        / "write_factor_metrics.jsonl"
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
            "k0-rvg-write-factor-decomposition-execution-manifest-v1",
        "runtime_git_head": repo["head"],
        "runtime_branch": repo["branch"],
        "parent_freeze_commit":
            PARENT_FREEZE_COMMIT,
        "execution_protocol":
            EXECUTION_PROTOCOL,
        "factorization":
            FACTORIZATION,
        "decomposition":
            DECOMPOSITION,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count":
            EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count":
            EXPECTED_COMMON_COUNT,
        "model_forward_count":
            forward_count,
        "primary_layer":
            PRIMARY_LAYER,
        "relative_coordinates":
            list(RELATIVE_COORDINATES),
        "mamba_source_sha256":
            binding.source_sha256,
        "handoff_zip_sha256":
            handoff["zip_sha256"],
        "checkpoint_sha256":
            handoff["checkpoint_sha256"],
        "encoder_canonical_digest":
            encoder["canonical_digest"],
        "encoder_raw_concat_digest":
            encoder["raw_concat_digest"],
        "source_snapshot_dtype":
            "torch.float32",
        "metric_accumulation_dtype":
            "torch.float64",
        "raw_vectors_persisted":
            False,
        "scientific_model_forward_executed":
            True,
        "scientific_write_factor_read":
            True,
        "tokenizer_invoked":
            False,
        "logits_read":
            False,
        "task_heads_executed":
            False,
        "training_executed":
            False,
        "causal_intervention_executed":
            False,
        "pca_probe_or_learned_geometry_executed":
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
            "write_factor_metrics.jsonl":
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
        "PASS_WRITE_FACTOR_DECOMPOSITION_EXECUTION"
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
                f"Q_U={t['q_u_l2_over_k0_w']['median']} "
                f"Q_D={t['q_d_l2_over_k0_w']['median']} "
                f"W={t['delta_write_l2_over_k0_w']['median']} "
                f"interaction="
                f"{t['interaction_over_k0_w_sq']['median']}"
            )


def main():
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

    parent = load_parent_runner(root)

    (
        magnitude,
        base,
        plan,
        cohort,
        signatures,
    ) = build_plan(
        root,
        parent,
    )

    repo, parent_summary = authenticate_parent(
        root,
        parent,
        base,
    )

    print("=== WRITE FACTOR DECOMPOSITION PLAN ===")
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
        len(cohort),
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
        "factorization =",
        FACTORIZATION,
    )
    print(
        "decomposition =",
        DECOMPOSITION,
    )
    print(
        "expected_mamba_source_sha256 =",
        EXPECTED_MAMBA_SOURCE_SHA256,
    )
    print("tokenizer_invoked = False")
    print(
        "scientific_model_forward_executed = False"
    )

    if args.static_preflight:
        print(
            "PASS_WRITE_FACTOR_STATIC_PREFLIGHT"
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
        parent,
        magnitude,
        base,
        repo,
        plan,
        cohort,
        signatures,
        args.handoff.resolve(),
        args.output_dir,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WriteFactorError as exc:
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