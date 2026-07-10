from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_controlled_v6b_minimal as train  # noqa: E402
from scripts.stage43_external_factver_eval_utils import (  # noqa: E402
    get_base_prediction,
    get_composed_prediction,
    normalize_label,
)

REQUIRED_SCALAR_FIELDS = (
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "entitlement_prob",
    "compositional_entitlement_prob",
    "learned_entitlement_prob",
    "learned_entitlement_logit",
    "polarity_margin",
    "positive_energy",
    "negative_energy",
)

SAFETY_POLICY = {
    "external_data_used_for_training": False,
    "external_labels_used_for_training": False,
    "threshold_used_for_model_selection": False,
    "checkpoint_selection_modified": False,
    "shadow_diagnostics_integrated": False,
    "final_predictions_modified_by_shadow": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage160-A eval-only external scalar export from a saved checkpoint."
    )
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--external-factver-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--prediction-export-schema", default=None)
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--vnext-router-mode", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--include-gold-label", action="store_true", default=False)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=json_safe) + "\n")


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Stage160 export requires a checkpoint dict with model_state_dict")
    state = payload["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint model_state_dict is not a dict")
    metadata = dict(payload.get("metadata") or {})
    for key in (
        "architecture",
        "vnext_router_mode",
        "backbone",
        "prediction_export_schema",
        "label_list",
        "label_mapping",
        "training_args",
        "stage160_checkpoint_format_version",
    ):
        if key in payload and key not in metadata:
            metadata[key] = payload[key]
    return state, metadata, payload


def merged_runner_args(cli: argparse.Namespace, metadata: dict[str, Any]) -> argparse.Namespace:
    defaults: dict[str, Any]
    try:
        defaults = vars(train.build_parser().parse_args([]))
    except SystemExit:
        defaults = {}
    config = dict(defaults)
    training_args = metadata.get("training_args")
    if isinstance(training_args, dict):
        config.update(training_args)
    config.update({k: v for k, v in metadata.items() if k not in {"training_args"}})
    if cli.data is not None:
        config["data"] = str(cli.data)
    for key in ("architecture", "vnext_router_mode", "backbone", "prediction_export_schema"):
        value = getattr(cli, key)
        if value is not None:
            config[key] = value
    config["device"] = cli.device
    config.setdefault("architecture", "v6b_minimal")
    config.setdefault("backbone", "mamba")
    config.setdefault("vnext_router_mode", "multiplicative")
    config.setdefault("prediction_export_schema", "stage28e_v1")
    config.setdefault("model_name", metadata.get("model_name") or "state-spaces/mamba-130m-hf")
    config.setdefault("max_length", metadata.get("max_length") or 256)
    config.setdefault("flag_source", "controlled_heuristic")
    config.setdefault("vnext_evidence_interface", metadata.get("vnext_evidence_interface") or "full_evidence")
    return argparse.Namespace(**config)


def _get(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_eval_model(args: argparse.Namespace, checkpoint_payload: dict[str, Any], device: torch.device):
    vocab = checkpoint_payload.get("vocab")
    tokenizer = None
    max_length = int(_get(args, "max_length", 256) or 256)
    if _get(args, "backbone", "mamba") == "dummy":
        if vocab is None:
            data_path = _get(args, "data", None)
            if data_path is None:
                raise ValueError("dummy-backbone checkpoint export requires checkpoint vocab or --data")
            vocab = train.v5.build_vocab(train.v5.load_jsonl(data_path))
        vocab = dict(vocab)
        vocab_size = len(vocab)
        if _get(args, "architecture", "v6b_minimal") == "vnext_minimal":
            model = train.build_vnext_model(
                vocab_size,
                max_length,
                vnext_router_mode=_get(args, "vnext_router_mode", "multiplicative"),
                vnext_enable_segmented_dual_pass=bool(_get(args, "vnext_enable_segmented_dual_pass", False)),
                vnext_segmented_context_role=_get(args, "vnext_segmented_context_role", "diagnostic_only"),
                vnext_context_risk_cap_alpha=float(_get(args, "vnext_context_risk_cap_alpha", 0.0)),
                vnext_context_risk_threshold=float(_get(args, "vnext_context_risk_threshold", 0.5)),
                vnext_context_risk_source=_get(args, "vnext_context_risk_source", "context_not_entitled_prob"),
                vnext_use_slot_mismatch_head=bool(_get(args, "vnext_use_slot_mismatch_head", False)),
                vnext_slot_mismatch_detach_input=bool(_get(args, "vnext_slot_mismatch_detach_input", True)),
                vnext_slot_mismatch_input_mode=_get(args, "vnext_slot_mismatch_input_mode", "sufficiency_repr"),
                vnext_slot_mismatch_head_type=_get(args, "vnext_slot_mismatch_head_type", "linear"),
            )
        else:
            model = train.build_model(
                vocab_size,
                max_length,
                use_boundary_head=bool(_get(args, "use_boundary_loss", False)),
                use_frame_violation_head=bool(_get(args, "use_frame_violation_loss", False)),
                use_predicate_isolation_head=bool(_get(args, "use_predicate_isolation_loss", False)),
                use_preservation_entitlement_head=bool(_get(args, "use_preservation_entitlement_loss", False)),
                use_temporal_diagnostic_head=bool(_get(args, "use_temporal_diagnostic_loss", False)),
                use_temporal_residual_adapter=bool(_get(args, "use_temporal_residual_adapter", False)),
                temporal_adapter_detach_input=bool(_get(args, "temporal_adapter_detach_input", True)),
                use_temporal_channel=bool(_get(args, "use_temporal_channel", False)),
                temporal_channel_detach_input=bool(_get(args, "temporal_channel_detach_input", True)),
                use_temporal_channel_loss=bool(_get(args, "use_temporal_channel_loss", False)),
                temporal_channel_loss_weight=float(_get(args, "temporal_channel_loss_weight", 0.0)),
                temporal_channel_loss_pos_weight=float(_get(args, "temporal_channel_loss_pos_weight", 1.0)),
                use_temporal_channel_gated_penalty=bool(_get(args, "use_temporal_channel_gated_penalty", False)),
                temporal_channel_gated_penalty_scale=float(_get(args, "temporal_channel_gated_penalty_scale", 0.0)),
            )
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_get(args, "model_name", "state-spaces/mamba-130m-hf"))
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Mamba tokenizer has neither pad_token nor eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        if _get(args, "architecture", "v6b_minimal") == "vnext_minimal":
            model = train.build_vnext_mamba_model(
                _get(args, "model_name", "state-spaces/mamba-130m-hf"),
                freeze_encoder=bool(_get(args, "freeze_encoder", False)),
                freeze_a_log=bool(_get(args, "freeze_a_log", False)),
                vnext_router_mode=_get(args, "vnext_router_mode", "multiplicative"),
                vnext_enable_segmented_dual_pass=bool(_get(args, "vnext_enable_segmented_dual_pass", False)),
                vnext_segmented_context_role=_get(args, "vnext_segmented_context_role", "diagnostic_only"),
                vnext_context_risk_cap_alpha=float(_get(args, "vnext_context_risk_cap_alpha", 0.0)),
                vnext_context_risk_threshold=float(_get(args, "vnext_context_risk_threshold", 0.5)),
                vnext_context_risk_source=_get(args, "vnext_context_risk_source", "context_not_entitled_prob"),
                vnext_use_slot_mismatch_head=bool(_get(args, "vnext_use_slot_mismatch_head", False)),
                vnext_slot_mismatch_detach_input=bool(_get(args, "vnext_slot_mismatch_detach_input", True)),
                vnext_slot_mismatch_input_mode=_get(args, "vnext_slot_mismatch_input_mode", "sufficiency_repr"),
                vnext_slot_mismatch_head_type=_get(args, "vnext_slot_mismatch_head_type", "linear"),
            )
        else:
            model = train.build_mamba_model(
                _get(args, "model_name", "state-spaces/mamba-130m-hf"),
                freeze_encoder=bool(_get(args, "freeze_encoder", False)),
                freeze_a_log=bool(_get(args, "freeze_a_log", False)),
                use_boundary_head=bool(_get(args, "use_boundary_loss", False)),
                use_frame_violation_head=bool(_get(args, "use_frame_violation_loss", False)),
                use_predicate_isolation_head=bool(_get(args, "use_predicate_isolation_loss", False)),
                use_preservation_entitlement_head=bool(_get(args, "use_preservation_entitlement_loss", False)),
                use_temporal_diagnostic_head=bool(_get(args, "use_temporal_diagnostic_loss", False)),
                use_temporal_residual_adapter=bool(_get(args, "use_temporal_residual_adapter", False)),
                temporal_adapter_detach_input=bool(_get(args, "temporal_adapter_detach_input", True)),
                use_temporal_channel=bool(_get(args, "use_temporal_channel", False)),
                temporal_channel_detach_input=bool(_get(args, "temporal_channel_detach_input", True)),
                use_temporal_channel_loss=bool(_get(args, "use_temporal_channel_loss", False)),
                temporal_channel_loss_weight=float(_get(args, "temporal_channel_loss_weight", 0.0)),
                temporal_channel_loss_pos_weight=float(_get(args, "temporal_channel_loss_pos_weight", 1.0)),
                use_temporal_channel_gated_penalty=bool(_get(args, "use_temporal_channel_gated_penalty", False)),
                temporal_channel_gated_penalty_scale=float(_get(args, "temporal_channel_gated_penalty_scale", 0.0)),
            )
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, vocab, tokenizer, max_length


def load_external_records(path: Path, max_rows: int | None) -> tuple[list[dict[str, Any]], int, dict[str, int], Counter]:
    records: list[dict[str, Any]] = []
    malformed: Counter = Counter()
    gold_counts: Counter = Counter()
    seen_rows = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if max_rows is not None and seen_rows >= max_rows:
                break
            text = line.strip()
            if not text:
                continue
            seen_rows += 1
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                malformed["invalid_json"] += 1
                continue
            if not isinstance(row, dict):
                malformed["not_object"] += 1
                continue
            claim = row.get("claim")
            evidence = row.get("evidence")
            if claim is None or evidence is None:
                malformed["missing_claim_or_evidence"] += 1
                continue
            stable_id = row.get("id", row.get("example_id", row.get("uid", f"line_{line_no}")))
            raw_gold = row.get("label", row.get("gold_label"))
            gold = normalize_label(raw_gold)
            if raw_gold is not None and gold is None:
                malformed["invalid_gold_label"] += 1
            if gold is not None:
                gold_counts[gold] += 1
            record = dict(row)
            record["id"] = str(stable_id)
            record["pair_id"] = str(row.get("pair_id") or stable_id)
            record["claim"] = str(claim)
            record["evidence"] = str(evidence)
            record["gold_label"] = gold
            record["final_label"] = gold or "NOT_ENTITLED"
            record.setdefault("intervention_type", "stage160_external_factver")
            record.setdefault("normalized_intervention", "stage160_external_factver")
            record.setdefault("frame_compatible_label", 0)
            record.setdefault("predicate_covered_label", 0)
            record.setdefault("sufficiency_label", 0)
            record.setdefault("polarity_label", "NONE")
            records.append(record)
    return records, seen_rows, dict(sorted(malformed.items())), gold_counts


def export_rows(
    *,
    model: Any,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    vocab: dict[str, int] | None,
    tokenizer: Any,
    max_length: int,
    device: torch.device,
    batch_size: int,
    include_gold_label: bool,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    records = train.apply_vnext_evidence_interface_to_records(
        records, _get(args, "vnext_evidence_interface", "full_evidence")
    )
    model_inputs = train._stage118_encode_inputs(
        records,
        args=args,
        vocab=vocab,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
    )
    pred_rows = train._stage43_prediction_records_from_model(
        model=model,
        records=records,
        model_inputs=model_inputs,
        flag_source=_get(args, "flag_source", "controlled_heuristic"),
        device=device,
        args=args,
        batch_size=batch_size,
    )
    output: list[dict[str, Any]] = []
    for record, pred_row in zip(records, pred_rows):
        prediction = pred_row.get("pred_final_label") or pred_row.get("pred_label")
        composed_prediction, composed_changed, _reason = get_composed_prediction(pred_row)
        base_prediction = get_base_prediction(pred_row)
        item = {
            "id": record["id"],
            "claim": record["claim"],
            "evidence": record["evidence"],
            "prediction": composed_prediction or prediction,
            "stage160_checkpoint_path": str(checkpoint_path),
            "stage160_export_script": "scripts/export_external_scalars_from_checkpoint.py",
            "stage160_external_data_used_for_training": False,
            "stage160_external_labels_used_for_threshold_tuning": False,
            "stage160_external_used_for_checkpoint_selection": False,
            "stage160_shadow_diagnostics_integrated": False,
        }
        if include_gold_label and record.get("gold_label") is not None:
            item["gold_label"] = record["gold_label"]
        if base_prediction is not None:
            item["base_prediction"] = base_prediction
        if composed_prediction is not None:
            item["composed_prediction"] = composed_prediction
            item["changed"] = bool(composed_changed and base_prediction != composed_prediction)
        for key in REQUIRED_SCALAR_FIELDS:
            if key in pred_row:
                item[key] = pred_row[key]
        output.append(item)
    return output


def build_report(
    *,
    cli: argparse.Namespace,
    pred_path: Path,
    rows: list[dict[str, Any]],
    seen_rows: int,
    malformed: dict[str, int],
    gold_counts: Counter,
) -> dict[str, Any]:
    prediction_counts = Counter(row.get("prediction") for row in rows)
    scalar_coverage = {
        key: sum(1 for row in rows if row.get(key) is not None)
        for key in REQUIRED_SCALAR_FIELDS
    }
    required_pass = all(count == len(rows) for count in scalar_coverage.values()) if rows else False
    return {
        "checkpoint_path": str(cli.checkpoint_path),
        "external_input_path": str(cli.external_factver_jsonl),
        "output_prediction_jsonl_path": str(pred_path),
        "n_rows": len(rows),
        "input_rows_seen": seen_rows,
        "malformed_rows": malformed,
        "prediction_counts": dict(sorted((str(k), v) for k, v in prediction_counts.items())),
        "gold_counts": dict(sorted(gold_counts.items())),
        "scalar_field_coverage_counts": scalar_coverage,
        "required_scalar_coverage_pass": required_pass,
        "required_scalar_fields": list(REQUIRED_SCALAR_FIELDS),
        "safety_policy": dict(SAFETY_POLICY),
    }


def write_md_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stage160-A External Scalar Export Report",
        "",
        f"Checkpoint: `{report['checkpoint_path']}`",
        f"External input: `{report['external_input_path']}`",
        f"Prediction JSONL: `{report['output_prediction_jsonl_path']}`",
        "",
        f"Rows exported: {report['n_rows']}",
        f"Malformed rows: {sum(report['malformed_rows'].values())}",
        f"Required scalar coverage pass: {report['required_scalar_coverage_pass']}",
        "",
        "Safety policy: external data and labels are eval-only; they do not train, tune thresholds, select checkpoints, or integrate shadow diagnostics.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cli = parse_args()
    device = torch.device(cli.device)
    state, metadata, payload = load_checkpoint(cli.checkpoint_path, device)
    payload["model_state_dict"] = state
    runner_args = merged_runner_args(cli, metadata)
    model, vocab, tokenizer, max_length = build_eval_model(runner_args, payload, device)
    records, seen_rows, malformed, gold_counts = load_external_records(
        cli.external_factver_jsonl, cli.max_rows
    )
    rows = export_rows(
        model=model,
        records=records,
        args=runner_args,
        vocab=vocab,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
        batch_size=cli.batch_size,
        include_gold_label=cli.include_gold_label,
        checkpoint_path=cli.checkpoint_path,
    )
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = cli.output_dir / f"{cli.run_prefix}_external_scalar_predictions.jsonl"
    report_json = cli.output_dir / f"{cli.run_prefix}_external_scalar_export_report.json"
    report_md = cli.output_dir / f"{cli.run_prefix}_external_scalar_export_report.md"
    write_jsonl(pred_path, rows)
    report = build_report(
        cli=cli,
        pred_path=pred_path,
        rows=rows,
        seen_rows=seen_rows,
        malformed=malformed,
        gold_counts=gold_counts,
    )
    write_json(report_json, report)
    write_md_report(report_md, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())