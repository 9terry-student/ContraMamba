#!/usr/bin/env python3
"""Evaluation-only direct frame-logit export for Stage189-C."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import export_external_scalars_from_checkpoint as checkpoint_export  # noqa: E402

EXPECTED_DATA_SHA = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
EXPECTED_SIDECAR_SHA = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
STATUS_COUNTS = {"ELIGIBLE": 605, "INELIGIBLE": 716, "UNRESOLVED": 119}
LABELS = {"REFUTE", "NOT_ENTITLED", "SUPPORT"}
DIRECT_SCORE_SOURCE = 'direct output["frame_logit"]'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-provenance", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--integrity-sidecar", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", default=EXPECTED_DATA_SHA)
    parser.add_argument("--expected-sidecar-semantic-sha256", default=EXPECTED_SIDECAR_SHA)
    parser.add_argument("--expected-trainer-sha256", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-checkpoint-helper-sha256", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            rows.append(value)
    return rows


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sidecar_sha(rows: list[dict[str, Any]]) -> str:
    semantic = [{key: row[key] for key in sorted(row) if key != "created_at"} for row in rows]
    encoded = json.dumps(semantic, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str | None:
    value = row.get("id", row.get("row_id"))
    return str(value) if value is not None else None


def unique_index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row_id(row)
        if identifier is None or identifier in result:
            raise ValueError(f"{name} has missing or duplicate row IDs")
        result[identifier] = row
    return result


def finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def argv_has_option(argv: Any, option: str) -> bool:
    return isinstance(argv, list) and any(
        token == option or (isinstance(token, str) and token.startswith(option + "="))
        for token in argv
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def row_id_semantic_sha(row_ids: set[str]) -> str:
    payload = json.dumps(sorted(row_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cleanup_jsonl_artifacts(*paths: Path) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return errors


def direct_export_loop(
    model: Any,
    records: list[dict[str, Any]],
    runner_args: argparse.Namespace,
    vocab: dict[str, int] | None,
    tokenizer: Any,
    max_length: int,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    """local_direct_loop_contract: native tensors are aligned before JSON construction."""
    train = checkpoint_export.train
    transformed = train.apply_vnext_evidence_interface_to_records(
        records, getattr(runner_args, "vnext_evidence_interface", "full_evidence")
    )
    model_inputs = train._stage118_encode_inputs(
        transformed, args=runner_args, vocab=vocab, tokenizer=tokenizer,
        max_length=max_length, device=device,
    )
    exported: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(transformed), max(1, int(batch_size))):
            end = min(len(transformed), start + max(1, int(batch_size)))
            batch_records = transformed[start:end]
            batch_inputs = train._stage43_slice_inputs(model_inputs, start, end)
            temporal_flags, predicate_flags = train.extract_flags(
                batch_records, getattr(runner_args, "flag_source", "controlled_heuristic"), device
            )
            feature_inputs = train._vnext_model_feature_inputs(batch_inputs)
            train._assert_model_accepts_feature_kwargs(
                model, feature_inputs, context="Stage189-C direct frame export"
            )
            output = model(
                **feature_inputs,
                temporal_mismatch_flags=temporal_flags,
                predicate_mismatch_flags=predicate_flags,
            )
            if not isinstance(output, dict) or output.get("frame_logit") is None:
                raise RuntimeError('model output lacks output["frame_logit"]')
            if output.get("frame_prob") is None:
                raise RuntimeError('model output lacks output["frame_prob"]')
            try:
                frame_logits = output["frame_logit"].detach().cpu().reshape(-1)
                frame_probs = output["frame_prob"].detach().cpu().reshape(-1)
            except (AttributeError, TypeError, RuntimeError) as exc:
                raise RuntimeError("frame_logit/frame_prob must be direct tensors") from exc
            if int(frame_logits.numel()) != len(batch_records):
                raise RuntimeError("flattened frame-logit count differs from batch source count")
            if int(frame_probs.numel()) != len(batch_records):
                raise RuntimeError("flattened frame-prob count differs from batch source count")
            if not bool(torch.isfinite(frame_logits).all()) or not bool(torch.isfinite(frame_probs).all()):
                raise RuntimeError("non-finite native frame tensor")
            if not bool(((frame_probs >= 0.0) & (frame_probs <= 1.0)).all()):
                raise RuntimeError("native frame_prob is outside [0,1]")
            prediction_rows = train.prediction_records_v6b(batch_records, output, args=runner_args)
            if len(prediction_rows) != len(batch_records):
                raise RuntimeError("prediction/source-row alignment failure")
            for offset, (record, prediction_row) in enumerate(zip(batch_records, prediction_rows)):
                prediction = prediction_row.get("pred_final_label") or prediction_row.get("pred_label")
                exported.append({
                    "row_id": str(record["id"]),
                    "prediction": prediction,
                    "frame_logit": float(frame_logits[offset].item()),
                    "frame_prob": float(frame_probs[offset].item()),
                })
    if len(exported) != len(records):
        raise RuntimeError("direct export truncated source rows")
    return exported


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    blockers: list[str] = []
    identity_gates: list[dict[str, Any]] = []
    topology_gates: list[dict[str, Any]] = []

    def gate(collection: list[dict[str, Any]], name: str, required: Any, observed: Any,
             passed: bool, reason: str) -> None:
        collection.append({"gate": name, "required": required, "observed": observed,
                           "passed": passed, "blocking_reason": "" if passed else reason})
        if not passed:
            blockers.append(f"{name}: {reason}")

    helper_path = Path(checkpoint_export.__file__).resolve()
    helper_sha = file_sha(helper_path)
    gate(identity_gates, "checkpoint_helper_sha256", args.expected_checkpoint_helper_sha256,
         helper_sha, helper_sha == args.expected_checkpoint_helper_sha256,
         "runtime checkpoint helper bytes differ from manifest")
    report: dict[str, Any] = {
        "stage": "Stage189-C",
        "decision": "STAGE189C_POSTHOC_MARGIN_REFERENCE_BLOCKED",
        "output_jsonl": str(args.output_jsonl.resolve()),
        "output_jsonl_sha256": None,
        "row_count": 0,
        "row_id_semantic_sha256": None,
        "checkpoint_sha256": None,
        "selected_epoch": None,
        "dataset_sha256": None,
        "sidecar_semantic_sha256": None,
        "trainer_sha256": None,
        "git_commit": None,
        "blocking_reasons": blockers,
        "identity_gates": identity_gates,
        "topology_gates": topology_gates,
        "checkpoint_helper_path": str(helper_path),
        "checkpoint_helper_sha256": helper_sha,
        "local_direct_loop_contract": {
            "source": 'output["frame_logit"] -> detach() -> cpu() -> reshape(-1)',
            "probability_source": 'output["frame_prob"] -> detach() -> cpu() -> reshape(-1)',
            "batch_source_alignment_required": True,
            "inverse_logit_used": False,
            "classifier_or_loss_logits_substituted": False,
        },
        "score_source": DIRECT_SCORE_SOURCE,
        "evaluation_only": True,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "optimizer_created": False,
        "backpropagation_performed": False,
        "threshold_tuning_performed": False,
        "external_data_used": False,
        "evidence_scope": "posthoc_training_row_mechanism_diagnostic_not_generalization_evidence",
    }
    if blockers:
        return report, None

    provenance = read_json(args.run_provenance)
    data_rows = read_jsonl(args.data)
    sidecar_rows = read_jsonl(args.integrity_sidecar)
    data_sha = file_sha(args.data)
    sidecar_sha = semantic_sidecar_sha(sidecar_rows)
    checkpoint_sha = file_sha(args.selected_checkpoint)
    state, metadata, checkpoint_payload = checkpoint_export.load_checkpoint(
        args.selected_checkpoint, torch.device("cpu")
    )
    del state
    selected_epoch = metadata.get("selected_epoch")
    gate(identity_gates, "dataset_sha256", args.expected_dataset_sha256, data_sha,
         data_sha == args.expected_dataset_sha256, "dataset SHA mismatch")
    gate(identity_gates, "sidecar_semantic_sha256", args.expected_sidecar_semantic_sha256, sidecar_sha,
         sidecar_sha == args.expected_sidecar_semantic_sha256, "sidecar semantic SHA mismatch")
    gate(identity_gates, "checkpoint_selected_state", True,
         metadata.get("checkpoint_is_selected_clean_dev_state"),
         metadata.get("checkpoint_is_selected_clean_dev_state") is True,
         "checkpoint is not an unambiguous selected clean-dev state")
    gate(identity_gates, "checkpoint_selected_epoch", "positive integer", selected_epoch,
         isinstance(selected_epoch, int) and selected_epoch > 0, "selected epoch missing")
    metric_value = metadata.get("selection_metric_value")
    gate(identity_gates, "checkpoint_selection_metric", "finite final_macro_f1",
         [metadata.get("selection_metric_name"), metric_value],
         metadata.get("selection_metric_name") == "final_macro_f1" and finite(metric_value),
         "selection metric identity missing or non-finite")
    expected_model = {"architecture": "v6b_minimal", "backbone": "mamba",
                      "model_name": "state-spaces/mamba-130m-hf"}
    gate(identity_gates, "checkpoint_model_identity", expected_model,
         {key: metadata.get(key) for key in expected_model},
         {key: metadata.get(key) for key in expected_model} == expected_model,
         "checkpoint model identity mismatch")
    gate(identity_gates, "checkpoint_dataset_sha", args.expected_dataset_sha256,
         metadata.get("main_data_sha256"), metadata.get("main_data_sha256") == args.expected_dataset_sha256,
         "checkpoint dataset identity mismatch")
    gate(identity_gates, "checkpoint_dataset_path", "data/controlled_v5_v3_without_time_swap.jsonl",
         str(metadata.get("main_data_path") or ""),
         str(metadata.get("main_data_path") or "").replace(chr(92), "/").endswith(
             "data/controlled_v5_v3_without_time_swap.jsonl"),
         "checkpoint dataset path mismatch")
    gate(identity_gates, "checkpoint_trainer_sha", args.expected_trainer_sha256,
         metadata.get("trainer_sha256"), metadata.get("trainer_sha256") == args.expected_trainer_sha256,
         "checkpoint trainer SHA mismatch")
    gate(identity_gates, "checkpoint_git_commit", args.expected_git_commit,
         metadata.get("source_git_commit"), metadata.get("source_git_commit") == args.expected_git_commit,
         "checkpoint Git commit mismatch")

    parsed = provenance.get("parsed_args") or {}
    provenance_commit = ((provenance.get("source_provenance") or {}).get("git_commit"))
    provenance_trainer_sha = ((provenance.get("source_provenance") or {}).get("trainer_sha256"))
    provenance_checkpoint = ((provenance.get("finalization") or {}).get("selected_checkpoint") or {})
    gate(identity_gates, "provenance_git_commit", args.expected_git_commit, provenance_commit,
         provenance_commit == args.expected_git_commit, "run provenance Git commit mismatch")
    gate(identity_gates, "provenance_trainer_sha", args.expected_trainer_sha256,
         provenance_trainer_sha, provenance_trainer_sha == args.expected_trainer_sha256,
         "run provenance trainer SHA mismatch")
    gate(identity_gates, "provenance_model_identity", expected_model,
         {key: parsed.get(key) for key in expected_model},
         {key: parsed.get(key) for key in expected_model} == expected_model,
         "run provenance model identity mismatch")
    gate(identity_gates, "provenance_checkpoint_sha", checkpoint_sha,
         provenance_checkpoint.get("sha256"), provenance_checkpoint.get("sha256") == checkpoint_sha,
         "run provenance does not identify this checkpoint")
    gate(identity_gates, "checkpoint_seed", parsed.get("seed"), metadata.get("seed"),
         isinstance(metadata.get("seed"), int) and metadata.get("seed") == parsed.get("seed"),
         "checkpoint/provenance seed mismatch")
    gate(identity_gates, "checkpoint_selected_epoch_matches_provenance",
         (provenance.get("finalization") or {}).get("selected_epoch"), selected_epoch,
         selected_epoch == (provenance.get("finalization") or {}).get("selected_epoch"),
         "checkpoint/provenance selected epoch mismatch")
    training_args = metadata.get("training_args") or {}
    gate(identity_gates, "checkpoint_training_args", "exact provenance parsed args", "compared",
         isinstance(training_args, dict) and all(training_args.get(key) == value for key, value in parsed.items()),
         "checkpoint parsed training args do not cover provenance parsed args")

    margin = metadata.get("compatible_positive_margin") or {}
    weight = margin.get("weight")
    enabled = margin.get("enabled")
    arm = (
        "baseline" if enabled is False and weight == 0.0
        else "intervention" if enabled is True and weight == 0.05
        else None
    )
    common_margin_contract = (
        margin.get("margin_logit") == 0.0
        and margin.get("score_source") == 'output["frame_logit"]'
        and margin.get("normalization") == "eligible_row_mean"
    )
    gate(identity_gates, "checkpoint_margin_common_contract",
         {"margin_logit": 0.0, "score_source": 'output["frame_logit"]',
          "normalization": "eligible_row_mean"},
         {key: margin.get(key) for key in ("margin_logit", "score_source", "normalization")},
         common_margin_contract, "checkpoint common margin contract mismatch")
    gate(identity_gates, "checkpoint_margin_arm_identity",
         "baseline=(enabled false, weight 0.0) or intervention=(enabled true, weight 0.05)",
         {"enabled": enabled, "weight": weight}, arm is not None,
         "checkpoint enabled/weight pair does not identify an arm")
    if arm == "baseline":
        baseline_contract = (
            margin.get("enabled") is False
            and margin.get("weight") == 0.0
            and margin.get("sidecar_path") is None
            and margin.get("expected_sidecar_semantic_sha256") is None
            and not argv_has_option(provenance.get("raw_sys_argv"), "--controlled-integrity-sidecar-path")
            and not argv_has_option(
                provenance.get("raw_sys_argv"),
                "--expected-integrity-sidecar-semantic-sha256",
            )
        )
        gate(identity_gates, "checkpoint_baseline_margin_contract", True,
             baseline_contract, baseline_contract,
             "baseline checkpoint margin/sidecar isolation contract mismatch")
    elif arm == "intervention":
        sidecar_suffix = str(margin.get("sidecar_path") or "").replace(chr(92), "/")
        intervention_contract = (
            margin.get("enabled") is True
            and margin.get("weight") == 0.05
            and sidecar_suffix.endswith(
                "reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/"
                "stage185a_controlled_train_integrity_sidecar.jsonl"
            )
            and margin.get("expected_sidecar_semantic_sha256")
            == args.expected_sidecar_semantic_sha256
        )
        gate(identity_gates, "checkpoint_intervention_margin_contract", True,
             intervention_contract, intervention_contract,
             "intervention checkpoint margin/sidecar contract mismatch")

    data_index = unique_index(data_rows, "main data")
    sidecar_index = unique_index(sidecar_rows, "sidecar")
    gate(identity_gates, "sidecar_exact_dataset_row_ids", len(data_index), len(sidecar_index),
         set(data_index) == set(sidecar_index), "sidecar has missing or extra source row IDs")
    train_sidecar = {identifier: row for identifier, row in sidecar_index.items() if row.get("split") == "train"}
    counts = Counter(row.get("integrity_status") for row in train_sidecar.values())
    eligible = [row for row in train_sidecar.values() if row.get("integrity_status") == "ELIGIBLE"]
    topology = {
        "total": len(train_sidecar),
        **{key: counts.get(key, 0) for key in STATUS_COUNTS},
        "eligible_pairs": len({row.get("pair_id") for row in eligible}),
        "eligible_families": len({row.get("family_contract_id") for row in eligible}),
    }
    required_topology = {"total": 1440, **STATUS_COUNTS, "eligible_pairs": 121, "eligible_families": 5}
    gate(topology_gates, "authoritative_train_topology", required_topology, topology,
         topology == required_topology, "posthoc train topology mismatch")
    report.update({
        "seed": metadata.get("seed"), "arm": arm, "selected_epoch": selected_epoch,
        "checkpoint_sha256": checkpoint_sha, "dataset_sha256": data_sha,
        "sidecar_semantic_sha256": sidecar_sha, "trainer_sha256": metadata.get("trainer_sha256"),
        "git_commit": metadata.get("source_git_commit"), "checkpoint_identity": None,
    })
    if blockers:
        return report, None

    records = [dict(data_index[identifier]) for identifier in sorted(train_sidecar)]
    runner_cli = argparse.Namespace(
        data=args.data, architecture=None, vnext_router_mode=None, backbone=None,
        prediction_export_schema=None, device=args.device,
    )
    runner_args = checkpoint_export.merged_runner_args(runner_cli, metadata)
    device = torch.device(args.device)
    model, vocab, tokenizer, max_length = checkpoint_export.build_eval_model(
        runner_args, checkpoint_payload, device
    )
    predicted = direct_export_loop(
        model, records, runner_args, vocab, tokenizer, max_length, device,
        args.batch_size,
    )
    predicted_index = unique_index(predicted, "posthoc predictions")
    gate(topology_gates, "exported_row_ids", sorted(train_sidecar), sorted(predicted_index),
         set(predicted_index) == set(train_sidecar), "exported row-ID set differs from authoritative train IDs")
    identity = {
        "checkpoint_sha256": checkpoint_sha,
        "trainer_sha256": metadata.get("trainer_sha256"),
        "git_commit": metadata.get("source_git_commit"),
        "dataset_sha256": data_sha,
        "sidecar_semantic_sha256": sidecar_sha,
        "selected_epoch": selected_epoch,
        "seed": metadata.get("seed"),
        "arm": arm,
        "architecture": metadata.get("architecture"),
        "backbone": metadata.get("backbone"),
        "model_name": metadata.get("model_name"),
    }
    output_rows: list[dict[str, Any]] = []
    for identifier in sorted(train_sidecar):
        source, audit, predicted_row = data_index[identifier], train_sidecar[identifier], predicted_index[identifier]
        gold_label = source.get("final_label")
        prediction = predicted_row.get("prediction")
        frame_logit = predicted_row.get("frame_logit")
        frame_prob = predicted_row.get("frame_prob")
        if gold_label not in LABELS or prediction not in LABELS:
            blockers.append(f"row {identifier}: invalid gold or prediction label")
        if not finite(frame_logit) or not finite(frame_prob) or not 0.0 <= float(frame_prob) <= 1.0:
            blockers.append(f"row {identifier}: invalid direct frame scalar")
        output_rows.append({
            "row_id": identifier, "seed": metadata["seed"], "arm": arm, "split": "train",
            "pair_id": audit.get("pair_id"), "family": audit.get("family_contract_id"),
            "gold_label": gold_label, "prediction": prediction,
            "integrity_status": audit.get("integrity_status"),
            "eligible_boolean": audit.get("integrity_status") == "ELIGIBLE",
            "frame_compatible_label": source.get("frame_compatible_label"),
            "frame_logit": float(frame_logit), "frame_prob": float(frame_prob),
            "selected_epoch": selected_epoch, "checkpoint_identity": identity,
            "score_source": DIRECT_SCORE_SOURCE,
        })
    gate(topology_gates, "output_row_count", 1440, len(output_rows),
         len(output_rows) == 1440, "output row count mismatch")
    gate(topology_gates, "output_unique_row_ids", 1440, len({row["row_id"] for row in output_rows}),
         len({row["row_id"] for row in output_rows}) == 1440, "output has duplicate or missing row IDs")
    row_identity_exact = all(
        row.get("selected_epoch") == selected_epoch
        and row.get("checkpoint_identity") == identity
        for row in output_rows
    )
    gate(topology_gates, "output_selected_epoch_checkpoint_identity", True,
         row_identity_exact, row_identity_exact,
         "row selected epoch or checkpoint identity mismatch")
    gate(topology_gates, "output_row_contract", True, not blockers, not blockers,
         "one or more output rows violate label/scalar/identity contract")
    report["checkpoint_identity"] = identity
    if blockers:
        return report, None
    report["decision"] = "STAGE189C_POSTHOC_MARGIN_REFERENCE_EXPORTED"
    report["blocking_reasons"] = []
    report["row_count"] = len(output_rows)
    report["row_id_semantic_sha256"] = row_id_semantic_sha({row["row_id"] for row in output_rows})
    report["integrity_status_counts"] = dict(sorted(counts.items()))
    return report, output_rows


def main() -> int:
    args = parse_args()
    temporary = args.output_jsonl.with_name(args.output_jsonl.name + ".tmp")
    try:
        initial_cleanup_errors = cleanup_jsonl_artifacts(temporary, args.output_jsonl)
        if initial_cleanup_errors:
            raise RuntimeError(
                "initial JSONL cleanup failed: " + "; ".join(initial_cleanup_errors)
            )
        report, output_rows = execute(args)
        if output_rows is None:
            cleanup_errors = cleanup_jsonl_artifacts(temporary, args.output_jsonl)
            if cleanup_errors:
                report["blocking_reasons"].extend(
                    f"blocked cleanup failed: {item}" for item in cleanup_errors
                )
            write_json(args.output_report, report)
            return 2
        write_jsonl_atomic(args.output_jsonl, output_rows)
        report["output_jsonl_sha256"] = file_sha(args.output_jsonl)
        write_json(args.output_report, report)
        return 0
    except BaseException as exc:
        cleanup_errors = cleanup_jsonl_artifacts(temporary, args.output_jsonl)
        blocked = {
            "stage": "Stage189-C",
            "decision": "STAGE189C_POSTHOC_MARGIN_REFERENCE_BLOCKED",
            "blocking_reasons": (
                [f"{type(exc).__name__}: {exc}"]
                + [f"blocked cleanup failed: {item}" for item in cleanup_errors]
            ),
            "output_jsonl": str(args.output_jsonl.resolve()),
            "output_jsonl_sha256": None,
            "row_count": 0,
            "row_id_semantic_sha256": None,
            "identity_gates": [],
            "topology_gates": [],
            "checkpoint_sha256": None,
            "selected_epoch": None,
            "dataset_sha256": None,
            "sidecar_semantic_sha256": None,
            "trainer_sha256": None,
            "git_commit": None,
            "checkpoint_helper_sha256": None,
            "local_direct_loop_contract": {
                "source": 'output["frame_logit"] -> detach() -> cpu() -> reshape(-1)',
                "probability_source": 'output["frame_prob"] -> detach() -> cpu() -> reshape(-1)',
            },
            "evaluation_only": True,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "score_source": DIRECT_SCORE_SOURCE,
        }
        write_json(args.output_report, blocked)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
