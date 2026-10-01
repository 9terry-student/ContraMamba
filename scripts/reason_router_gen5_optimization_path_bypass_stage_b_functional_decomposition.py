from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

STAGE_A_FREEZE_COMMIT = (
    "3b1adc2bd433f1fc697cfb7a268b2772cd83c1b5"
)
PHASE3A_EXECUTION_COMMIT = (
    "d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e"
)

RUN_ROOT = Path(
    "reports/reason_router_gen5_phase3a_training_runs/"
    "gen5-phase3a-contention-qualification-9cell-d58e894-retry3"
)

STAGE_A_JSON = Path(
    "reports/"
    "reason_router_gen5_optimization_path_bypass_"
    "stage_a_static_geometry_audit_v2.json"
)

EXECUTION_AUTHORITY_PATH = Path(
    "reports/"
    "reason_router_gen5_optimization_path_bypass_"
    "stage_b_forward_execution_authority_spec_candidate.md"
)

AUTHORIZED_IMPLEMENTATION_PATHS = frozenset({
    "scripts/reason_router_gen5_optimization_path_bypass_stage_b_functional_decomposition.py",
    "tests/test_reason_router_gen5_optimization_path_bypass_stage_b_functional_decomposition.py",
})

SEEDS = (6201, 6202, 6203)
PRESSURES = ("P0", "PR", "PC")
CONDITIONS = (
    "FULL",
    "R22_ONLY",
    "R22_REMOVED",
    "ZERO",
)

EXPECTED_ROWS = 840

PARENT_CHECKPOINT_SHA256 = (
    "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
)
R22_SHA256 = (
    "a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214"
)
C22_SHA256 = (
    "c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4"
)


class StageBError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StageBError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StageBError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def status_paths() -> set[str]:
    raw = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        text=True,
    )
    out: set[str] = set()
    for line in raw.splitlines():
        if not line:
            continue
        require(len(line) >= 4, f"STATUS_ROW:{line}")
        out.add(line[3:].replace("\\", "/"))
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return sha256_bytes(tensor.numpy().tobytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def authenticate_repo(
    expected_head: str,
    *,
    allow_implementation_worktree: bool,
) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, f"HEAD_MISMATCH:{head}")

    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            STAGE_A_FREEZE_COMMIT,
            expected_head,
        ) == 0,
        "STAGE_A_FREEZE_NOT_ANCESTOR",
    )

    if allow_implementation_worktree:
        observed = status_paths()
        require(
            observed.issubset(AUTHORIZED_IMPLEMENTATION_PATHS),
            f"IMPLEMENTATION_SCOPE_DRIFT:{sorted(observed)}",
        )
    else:
        require(
            git("status", "--porcelain") == "",
            "WORKTREE_NOT_CLEAN",
        )


def validate_stage_a_artifact() -> dict[str, Any]:
    path = ROOT / STAGE_A_JSON
    require(path.is_file(), f"STAGE_A_JSON_MISSING:{path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    require(
        value.get("schema_version")
        == "GEN5_OPTIMIZATION_PATH_BYPASS_STAGE_A_STATIC_GEOMETRY_AUDIT_V2",
        "STAGE_A_SCHEMA",
    )
    require(
        value.get("mode")
        == "READ_ONLY_CPU_NO_FORWARD_NO_BACKWARD_NO_TRAINING",
        "STAGE_A_MODE",
    )
    require(
        len(value.get("cells") or []) == 9,
        "STAGE_A_CELL_COUNT",
    )
    return value


def load_phase3a_artifacts() -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}

    for seed in SEEDS:
        for pressure in PRESSURES:
            cell = (
                ROOT
                / RUN_ROOT
                / "cells"
                / f"seed{seed}"
                / pressure
            )

            ckpt_path = cell / "final_correction.pt"
            report_path = cell / "training_report.json"
            provenance_path = cell / "run_provenance.json"

            for path in (
                ckpt_path,
                report_path,
                provenance_path,
            ):
                require(
                    path.is_file(),
                    f"ARTIFACT_MISSING:{seed}:{pressure}:{path.name}",
                )

            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )

            require(
                report.get("result")
                == "PASS_GEN5_PHASE3A_TRAINING_CELL",
                f"REPORT_RESULT:{seed}:{pressure}",
            )
            require(
                report.get("execution_commit")
                == PHASE3A_EXECUTION_COMMIT,
                f"REPORT_COMMIT:{seed}:{pressure}",
            )
            require(
                report.get("parent_checkpoint_sha256")
                == PARENT_CHECKPOINT_SHA256,
                f"REPORT_PARENT:{seed}:{pressure}",
            )
            require(
                int(report.get("seed", -1)) == seed,
                f"REPORT_SEED:{seed}:{pressure}",
            )
            require(
                str(report.get("pressure")) == pressure,
                f"REPORT_PRESSURE:{seed}:{pressure}",
            )

            checkpoint_sha = sha256_file(ckpt_path)
            require(
                report.get("final_correction_file_sha256")
                == checkpoint_sha,
                f"CHECKPOINT_SHA:{seed}:{pressure}",
            )
            require(
                provenance.get("final_correction_file_sha256")
                == checkpoint_sha,
                f"PROVENANCE_CHECKPOINT_SHA:{seed}:{pressure}",
            )
            require(
                provenance.get("training_report_sha256")
                == sha256_file(report_path),
                f"PROVENANCE_REPORT_SHA:{seed}:{pressure}",
            )

            payload = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=True,
            )

            require(
                payload.get("schema_version")
                == "GEN5_PHASE3A_FINAL_CORRECTION_V1",
                f"PAYLOAD_SCHEMA:{seed}:{pressure}",
            )
            require(
                payload.get("execution_commit")
                == PHASE3A_EXECUTION_COMMIT,
                f"PAYLOAD_COMMIT:{seed}:{pressure}",
            )
            require(
                payload.get("parent_checkpoint_sha256")
                == PARENT_CHECKPOINT_SHA256,
                f"PAYLOAD_PARENT:{seed}:{pressure}",
            )
            require(
                payload.get("r22_sha256") == R22_SHA256,
                f"PAYLOAD_R22:{seed}:{pressure}",
            )
            require(
                payload.get("c22_sha256") == C22_SHA256,
                f"PAYLOAD_C22:{seed}:{pressure}",
            )

            state = payload.get("state_dict")
            require(isinstance(state, Mapping), "PAYLOAD_STATE")

            a = state["A_theta.weight"]
            b = state["B_theta.weight"]

            require(
                torch.is_tensor(a)
                and tuple(a.shape) == (2, 768),
                f"A_SHAPE:{seed}:{pressure}",
            )
            require(
                torch.is_tensor(b)
                and tuple(b.shape) == (24576, 2),
                f"B_SHAPE:{seed}:{pressure}",
            )
            require(
                bool(torch.isfinite(a).all()),
                f"A_NONFINITE:{seed}:{pressure}",
            )
            require(
                bool(torch.isfinite(b).all()),
                f"B_NONFINITE:{seed}:{pressure}",
            )

            hashes = payload.get("tensor_sha256") or {}
            require(
                hashes.get("A_theta.weight") == tensor_sha256(a),
                f"A_TENSOR_SHA:{seed}:{pressure}",
            )
            require(
                hashes.get("B_theta.weight") == tensor_sha256(b),
                f"B_TENSOR_SHA:{seed}:{pressure}",
            )

            out[(seed, pressure)] = {
                "checkpoint_path": ckpt_path,
                "checkpoint_sha256": checkpoint_sha,
                "report": report,
                "payload": payload,
            }

    require(len(out) == 9, "ARTIFACT_CELL_COUNT")
    return out


def decompose_b(
    b_weight: torch.Tensor,
    r22: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    b = b_weight.detach().clone()
    r = r22.detach().to(
        device=b.device,
        dtype=b.dtype,
    )

    require(tuple(b.shape) == (24576, 2), "DECOMPOSE_B_SHAPE")
    require(tuple(r.shape) == (24576, 2), "DECOMPOSE_R_SHAPE")

    r_only = r @ (r.T @ b)
    r_removed = b - r_only
    zero = torch.zeros_like(b)

    reconstruction = r_only + r_removed

    reconstruction_max_abs = float(
        torch.max(torch.abs(reconstruction - b)).item()
    )
    removed_r22_residual = float(
        torch.max(torch.abs(r.T @ r_removed)).item()
    )

    return (
        {
            "FULL": b,
            "R22_ONLY": r_only,
            "R22_REMOVED": r_removed,
            "ZERO": zero,
        },
        {
            "reconstruction_max_abs": reconstruction_max_abs,
            "r22_removed_projection_max_abs":
                removed_r22_residual,
        },
    )


def condition_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    require(logits.ndim == 2, "LOGITS_RANK")
    require(tuple(logits.shape) == (EXPECTED_ROWS, 3), "LOGITS_SHAPE")
    require(tuple(labels.shape) == (EXPECTED_ROWS,), "LABEL_SHAPE")
    require(bool(torch.isfinite(logits).all()), "LOGITS_NONFINITE")

    per_row = F.cross_entropy(
        logits,
        labels,
        reduction="none",
    )
    mean_loss = float(per_row.mean().detach().cpu().item())
    predictions = torch.argmax(logits, dim=-1)
    accuracy = float(
        (predictions == labels)
        .to(torch.float64)
        .mean()
        .detach()
        .cpu()
        .item()
    )

    return (
        {
            "mean_final_3way_ce": mean_loss,
            "accuracy": accuracy,
        },
        per_row,
    )


def validate_execution_authority(
    *,
    expected_head: str,
    implementation_freeze_commit: str,
    execution_authority_commit: str,
) -> None:
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            implementation_freeze_commit,
            execution_authority_commit,
        ) == 0,
        "IMPLEMENTATION_FREEZE_NOT_ANCESTOR",
    )
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            execution_authority_commit,
            expected_head,
        ) == 0,
        "EXECUTION_AUTHORITY_NOT_ANCESTOR",
    )

    for rel in sorted(AUTHORIZED_IMPLEMENTATION_PATHS):
        require(
            git_rc(
                "diff",
                "--quiet",
                implementation_freeze_commit,
                expected_head,
                "--",
                rel,
            ) == 0,
            f"IMPLEMENTATION_DRIFT:{rel}",
        )

    rel = str(EXECUTION_AUTHORITY_PATH).replace("\\", "/")
    require(
        (ROOT / EXECUTION_AUTHORITY_PATH).is_file(),
        "EXECUTION_AUTHORITY_MISSING",
    )

    frozen_blob = git(
        "rev-parse",
        f"{execution_authority_commit}:{rel}",
    )
    current_blob = git(
        "rev-parse",
        f"HEAD:{rel}",
    )
    require(
        frozen_blob == current_blob,
        "EXECUTION_AUTHORITY_BLOB_DRIFT",
    )

    text = git(
        "show",
        f"{execution_authority_commit}:{rel}",
    )

    require(
        "SCIENTIFIC_EXECUTION_ALLOWED=YES_FORWARD_ONLY_STAGE_B_MATRIX"
        in text,
        "EXECUTION_AUTHORITY_NOT_OPEN",
    )
    require(
        f"IMPLEMENTATION_FREEZE_COMMIT={implementation_freeze_commit}"
        in text,
        "EXECUTION_AUTHORITY_FREEZE_BINDING",
    )
    require(
        "TRAINING_ALLOWED=NO" in text,
        "EXECUTION_AUTHORITY_TRAINING_BOUNDARY",
    )
    require(
        "BACKWARD_ALLOWED=NO" in text,
        "EXECUTION_AUTHORITY_BACKWARD_BOUNDARY",
    )
    require(
        "CONFIRMATORY_9601_9900_ALLOWED=NO" in text,
        "EXECUTION_AUTHORITY_CONFIRMATORY_BOUNDARY",
    )


def run_static_verify(args: argparse.Namespace) -> None:
    authenticate_repo(
        args.expected_head,
        allow_implementation_worktree=args.allow_opening_worktree,
    )

    stage_a = validate_stage_a_artifact()
    artifacts = load_phase3a_artifacts()

    from scripts import train_reason_router_gen5_phase3a_contention as p3a

    static = p3a.validate_static_artifacts()

    require(
        len(static["dev_rows"]) == EXPECTED_ROWS,
        "DEV_ROW_COUNT",
    )
    require(
        len(static["train_rows"]) == 3360,
        "TRAIN_ROW_COUNT",
    )

    print("GEN5_STAGE_B_STATIC_VERIFY_PASS")
    print(f"HEAD={args.expected_head}")
    print(f"PHASE3A_CELLS={len(artifacts)}")
    print(f"DEV_ROWS={len(static['dev_rows'])}")
    print(
        "STAGE_A_SCHEMA="
        + str(stage_a["schema_version"])
    )
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")


def run_matrix(args: argparse.Namespace) -> None:
    authenticate_repo(
        args.expected_head,
        allow_implementation_worktree=False,
    )

    require(
        args.implementation_freeze_commit is not None,
        "IMPLEMENTATION_FREEZE_REQUIRED",
    )
    require(
        args.execution_authority_commit is not None,
        "EXECUTION_AUTHORITY_COMMIT_REQUIRED",
    )

    validate_execution_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=
            args.implementation_freeze_commit,
        execution_authority_commit=
            args.execution_authority_commit,
    )

    output_root = Path(args.output_root)
    require(
        not output_root.exists(),
        f"OUTPUT_COLLISION:{output_root}",
    )
    output_root.mkdir(parents=True, exist_ok=False)

    artifacts = load_phase3a_artifacts()

    from contramamba.gen5_phase2_state_update_ownership import (
        load_frozen_owner_bases,
        parent_parameter_fingerprint,
    )
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    from scripts import train_reason_router_gen5_phase3a_contention as p3a

    static = p3a.validate_static_artifacts()

    snapshot = p2train.resolve_exact_snapshot(
        args.model_snapshot
    )
    parent_checkpoint = Path(args.checkpoint)
    p3a.validate_checkpoint(parent_checkpoint)

    encoded = p3a.load_runtime_encoding(
        static,
        args.tokenizer_snapshot,
    )

    dev_bundle = encoded["dev_bundle"]
    require(
        len(dev_bundle["row_ids"]) == EXPECTED_ROWS,
        "DEV_ENCODING_ROWS",
    )

    expected_dev_sha = {
        str(
            artifacts[(seed, pressure)]["report"][
                "dev_encoding_sha256"
            ]
        )
        for seed in SEEDS
        for pressure in PRESSURES
    }
    require(
        expected_dev_sha
        == {encoded["dev_encoding_sha256"]},
        "DEV_ENCODING_SHA_MISMATCH",
    )

    model, wrapper, runtime_meta, strong_mask, planes = (
        p3a._prepare_runtime_model(
            snapshot=snapshot,
            checkpoint_path=parent_checkpoint,
            seed=SEEDS[0],
        )
    )

    parent_before = runtime_meta["parent_before"]

    model.eval()
    model.mamba.config.use_cache = False

    for parameter in wrapper.correction.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    require(
        all(
            parameter.grad is None
            for parameter in model.parameters()
        ),
        "PREEXISTING_GRADIENT",
    )

    device = torch.device("cuda:0")
    features, labels, active, targets = (
        p3a._feature_batch_to_device(
            dev_bundle,
            device,
        )
    )

    r22_cpu, _c22_cpu, _basis_geometry = (
        load_frozen_owner_bases(ROOT)
    )

    row_records: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []

    condition_evaluations = 0
    streamed_chunks_total = 0

    for seed in SEEDS:
        for pressure in PRESSURES:
            artifact = artifacts[(seed, pressure)]
            payload = artifact["payload"]
            state = payload["state_dict"]

            a_cpu = (
                state["A_theta.weight"]
                .detach()
                .cpu()
                .contiguous()
            )
            b_cpu = (
                state["B_theta.weight"]
                .detach()
                .cpu()
                .contiguous()
            )

            variants, decomposition_audit = decompose_b(
                b_cpu,
                r22_cpu,
            )

            require(
                decomposition_audit["reconstruction_max_abs"]
                <= 1e-6,
                f"DECOMPOSITION_RECONSTRUCTION:{seed}:{pressure}",
            )
            require(
                decomposition_audit[
                    "r22_removed_projection_max_abs"
                ]
                <= 1e-5,
                f"DECOMPOSITION_ORTHOGONALITY:{seed}:{pressure}",
            )

            with torch.no_grad():
                wrapper.correction.A_theta.weight.copy_(
                    a_cpu.to(
                        device=device,
                        dtype=wrapper.correction.A_theta.weight.dtype,
                    )
                )

            condition_summary: dict[str, Any] = {}

            for condition in CONDITIONS:
                b_variant = variants[condition]

                with torch.no_grad():
                    wrapper.correction.B_theta.weight.copy_(
                        b_variant.to(
                            device=device,
                            dtype=
                                wrapper.correction.B_theta.weight.dtype,
                        )
                    )

                require(
                    all(
                        parameter.grad is None
                        for parameter in model.parameters()
                    ),
                    f"GRADIENT_BEFORE_FORWARD:{seed}:{pressure}:{condition}",
                )

                with torch.inference_mode():
                    output, chunks = p3a._streamed_forward(
                        model,
                        features,
                        active,
                        targets,
                        pressure=pressure,
                        strong_mask=strong_mask,
                        planes=planes,
                    )

                    logits = output["logits"]
                    metrics, per_row_ce = condition_metrics(
                        logits,
                        labels,
                    )

                condition_evaluations += 1
                streamed_chunks_total += int(chunks)

                require(
                    all(
                        parameter.grad is None
                        for parameter in model.parameters()
                    ),
                    f"GRADIENT_AFTER_FORWARD:{seed}:{pressure}:{condition}",
                )

                logits_cpu = (
                    logits.detach().cpu().to(torch.float64)
                )
                ce_cpu = (
                    per_row_ce.detach().cpu().to(torch.float64)
                )
                pred_cpu = torch.argmax(
                    logits_cpu,
                    dim=-1,
                )
                labels_cpu = labels.detach().cpu()

                condition_summary[condition] = {
                    **metrics,
                    "streamed_chunks": int(chunks),
                }

                for i in range(EXPECTED_ROWS):
                    row_records.append({
                        "seed": seed,
                        "pressure": pressure,
                        "condition": condition,
                        "row_id": dev_bundle["row_ids"][i],
                        "pair_id": dev_bundle["pair_ids"][i],
                        "contrast_cell_id":
                            dev_bundle["contrast_cell_ids"][i],
                        "label": int(labels_cpu[i].item()),
                        "prediction":
                            int(pred_cpu[i].item()),
                        "cross_entropy":
                            float(ce_cpu[i].item()),
                        "logits": [
                            float(x)
                            for x in logits_cpu[i].tolist()
                        ],
                    })

                del (
                    output,
                    logits,
                    per_row_ce,
                    logits_cpu,
                    ce_cpu,
                    pred_cpu,
                )
                torch.cuda.synchronize()

            zero_loss = float(
                condition_summary["ZERO"]["mean_final_3way_ce"]
            )
            full_loss = float(
                condition_summary["FULL"]["mean_final_3way_ce"]
            )
            r_only_loss = float(
                condition_summary["R22_ONLY"][
                    "mean_final_3way_ce"
                ]
            )
            r_removed_loss = float(
                condition_summary["R22_REMOVED"][
                    "mean_final_3way_ce"
                ]
            )

            full_gain = zero_loss - full_loss

            derived = {
                "full_gain_vs_zero": full_gain,
                "r22_only_gain_vs_zero":
                    zero_loss - r_only_loss,
                "r22_removed_gain_vs_zero":
                    zero_loss - r_removed_loss,
                "r22_removed_loss_minus_full":
                    r_removed_loss - full_loss,
                "r22_only_loss_minus_zero":
                    r_only_loss - zero_loss,
                "r22_removed_retained_gain_fraction":
                    (
                        (zero_loss - r_removed_loss)
                        / full_gain
                        if abs(full_gain) > 1e-12
                        else None
                    ),
                "r22_only_retained_gain_fraction":
                    (
                        (zero_loss - r_only_loss)
                        / full_gain
                        if abs(full_gain) > 1e-12
                        else None
                    ),
            }

            cell_summaries.append({
                "seed": seed,
                "pressure": pressure,
                "source_checkpoint_sha256":
                    artifact["checkpoint_sha256"],
                "decomposition_audit":
                    decomposition_audit,
                "conditions": condition_summary,
                "derived": derived,
            })

    require(
        condition_evaluations == 36,
        f"CONDITION_EVALUATION_COUNT:{condition_evaluations}",
    )
    require(
        len(row_records)
        == 9 * 4 * EXPECTED_ROWS,
        f"ROW_RECORD_COUNT:{len(row_records)}",
    )

    require(
        parent_parameter_fingerprint(model) == parent_before,
        "PARENT_MUTATION",
    )
    require(
        all(
            parameter.grad is None
            for parameter in model.parameters()
        ),
        "GRADIENT_CREATED",
    )

    summary = {
        "schema_version":
            "GEN5_OPTIMIZATION_PATH_BYPASS_STAGE_B_FUNCTIONAL_DECOMPOSITION_V1",
        "result":
            "PASS_GEN5_STAGE_B_FORWARD_FUNCTIONAL_DECOMPOSITION",
        "execution_head": args.expected_head,
        "implementation_freeze_commit":
            args.implementation_freeze_commit,
        "execution_authority_commit":
            args.execution_authority_commit,
        "source_phase3a_execution_commit":
            PHASE3A_EXECUTION_COMMIT,
        "population":
            "FROZEN_PHASE3A_DEV_MATCHED_PRESSURE",
        "dev_rows": EXPECTED_ROWS,
        "dev_encoding_sha256":
            encoded["dev_encoding_sha256"],
        "conditions": list(CONDITIONS),
        "cells": cell_summaries,
        "condition_evaluation_count":
            condition_evaluations,
        "streamed_chunk_forward_count":
            streamed_chunks_total,
        "model_mode": "EVAL",
        "objective": "FINAL_3WAY_CROSS_ENTROPY",
        "training_executed": False,
        "backward_executed": False,
        "optimizer_step_count": 0,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
        "parent_signature_before": parent_before,
        "parent_signature_after":
            parent_parameter_fingerprint(model),
        "runtime": runtime_meta["runtime"],
    }

    rows_path = output_root / "functional_decomposition_rows.jsonl"
    summary_path = output_root / "functional_decomposition_summary.json"

    rows_path.write_bytes(
        b"".join(
            canonical_json_bytes(row)
            for row in row_records
        )
    )
    summary_path.write_bytes(
        canonical_json_bytes(summary)
    )

    provenance = {
        "schema_version":
            "GEN5_OPTIMIZATION_PATH_BYPASS_STAGE_B_PROVENANCE_V1",
        "status": "PASS",
        "execution_head": args.expected_head,
        "implementation_freeze_commit":
            args.implementation_freeze_commit,
        "execution_authority_commit":
            args.execution_authority_commit,
        "summary_sha256":
            sha256_file(summary_path),
        "rows_sha256":
            sha256_file(rows_path),
        "training_executed": False,
        "backward_executed": False,
        "optimizer_step_count": 0,
        "confirmatory_9601_9900_loaded": False,
    }

    provenance_path = output_root / "run_provenance.json"
    provenance_path.write_bytes(
        canonical_json_bytes(provenance)
    )

    print("GEN5_STAGE_B_FORWARD_MATRIX_PASS")
    print(f"CELLS={len(cell_summaries)}")
    print(f"CONDITION_EVALUATIONS={condition_evaluations}")
    print(f"ROW_RECORDS={len(row_records)}")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_STEP_COUNT=0")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print(f"SUMMARY={summary_path}")
    print(f"ROWS={rows_path}")
    print(f"PROVENANCE={provenance_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--static-verify-only",
        action="store_true",
    )
    modes.add_argument(
        "--run-matrix",
        action="store_true",
    )

    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--allow-opening-worktree",
        action="store_true",
    )

    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)

    parser.add_argument("--implementation-freeze-commit")
    parser.add_argument("--execution-authority-commit")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.static_verify_only:
        require(
            args.model_snapshot is None,
            "STATIC_MODEL_SNAPSHOT_FORBIDDEN",
        )
        require(
            args.tokenizer_snapshot is None,
            "STATIC_TOKENIZER_SNAPSHOT_FORBIDDEN",
        )
        require(
            args.checkpoint is None,
            "STATIC_PARENT_CHECKPOINT_FORBIDDEN",
        )
        require(
            args.output_root is None,
            "STATIC_OUTPUT_ROOT_FORBIDDEN",
        )
        return

    require(
        not args.allow_opening_worktree,
        "RUNTIME_OPENING_WORKTREE_FORBIDDEN",
    )
    require(
        args.model_snapshot is not None,
        "MODEL_SNAPSHOT_REQUIRED",
    )
    require(
        args.tokenizer_snapshot is not None,
        "TOKENIZER_SNAPSHOT_REQUIRED",
    )
    require(
        args.checkpoint is not None,
        "PARENT_CHECKPOINT_REQUIRED",
    )
    require(
        args.output_root is not None,
        "OUTPUT_ROOT_REQUIRED",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

    if args.static_verify_only:
        run_static_verify(args)
    else:
        run_matrix(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
