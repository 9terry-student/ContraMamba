#!/usr/bin/env python3
"""Export local gradient-conflict sufficient statistics for one selected checkpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

EXPORTED = "STAGE190B_GRADIENT_DIAGNOSTIC_EXPORTED"
BLOCKED = "STAGE190B_GRADIENT_DIAGNOSTIC_BLOCKED"
STATUS_COUNTS = {"ELIGIBLE": 605, "INELIGIBLE": 716, "UNRESOLVED": 119}
EXPECTED_COHORTS = {"clean_dev_all": 720, "compatible_fn": 13, "incompatible_fp": 1,
                    "matched_controls": 14, "clean_model_failures": 14, **{k.lower(): v for k, v in STATUS_COUNTS.items()}}
GROUPS = ("frame_head", "decision_head", "router_and_epistemic_heads", "backbone", "other_trainable")
SHARED_GROUPS = ("backbone", "router_and_epistemic_heads", "other_trainable")
EPSILON = 1e-30
FRACTION_TOLERANCE = 1e-12
EXPECTED_OBJECTIVE_ROW_COUNTS = {
    "margin_eligible": 605,
    "ce_eligible": 605,
    "ce_clean_dev_all": 720,
    "ce_clean_dev_support": 89,
    "neg_support_vs_not_entitled_margin": 89,
    "neg_support_vs_max_other_margin": 89,
    "neg_mean_frame_logit_compatible_fn": 13,
    "neg_mean_frame_logit_matched_controls": 14,
    "neg_mean_frame_logit_ineligible": 716,
    "neg_mean_frame_logit_unresolved": 119,
}
TARGET_OBJECTIVES = tuple(name for name in EXPECTED_OBJECTIVE_ROW_COUNTS if name != "margin_eligible")
METRIC_GROUPS = (*GROUPS, "overall")
EXPECTED_GROUP_METRIC_ROWS = len(TARGET_OBJECTIVES) * len(METRIC_GROUPS)
EXPECTED_DIRECTIONAL_ROWS = len(TARGET_OBJECTIVES)
OUTPUTS = ("stage190b_parameter_inventory.csv", "stage190b_objective_summary.csv",
           "stage190b_group_gradient_metrics.csv", "stage190b_directional_derivatives.csv",
           "stage190b_cohort_topology.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--stage182b-dir", type=Path, required=True)
    p.add_argument("--stage185a-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as h:
        return json.load(h)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as h:
        for number, line in enumerate(h, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as h:
        return list(csv.DictReader(h))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sidecar_sha(rows: list[dict[str, Any]]) -> str:
    value = [{k: row[k] for k in sorted(row) if k != "created_at"} for row in rows]
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str | None:
    value = row.get("row_id", row.get("id", row.get("stable_id")))
    return str(value) if value is not None else None


def index_rows(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = row_id(row)
        if key is None or key in result:
            raise ValueError(f"{name} has a missing or duplicate row ID")
        result[key] = row
    return result


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def named_tensor_sha(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in items:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor_bytes(tensor))
    return digest.hexdigest()


def module_parameter_ids(module: Any | None) -> set[int]:
    return set() if module is None else {id(p) for p in module.parameters(recurse=True) if p.requires_grad}


def parameter_groups(model: Any) -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, torch.Tensor]]], str, dict[str, str]]:
    ordered = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not ordered:
        raise ValueError("model has no trainable parameters")
    if not hasattr(model, "frame_gate") or not hasattr(model.frame_gate, "frame_classifier"):
        raise ValueError("v6b_minimal frame_gate.frame_classifier module is not proven")
    if not hasattr(model, "decision_head") or not hasattr(model, "mamba"):
        raise ValueError("v6b_minimal decision_head or mamba module is not proven")
    owner_ids = {
        "frame_head": module_parameter_ids(model.frame_gate.frame_classifier),
        "decision_head": module_parameter_ids(model.decision_head),
        "backbone": module_parameter_ids(model.mamba),
    }
    router_modules = [getattr(model, name, None) for name in (
        "frame_gate", "predicate_coverage_head", "sufficiency_gate", "polarity_energy_head",
        "boundary_head", "frame_violation_head", "predicate_isolation_head",
        "preservation_entitlement_head", "temporal_diagnostic_head", "temporal_residual_adapter",
        "temporal_channel_v1", "router", "fusion", "gate")]
    router_ids = set().union(*(module_parameter_ids(module) for module in router_modules))
    router_ids -= owner_ids["frame_head"] | owner_ids["decision_head"] | owner_ids["backbone"]
    owner_ids["router_and_epistemic_heads"] = router_ids
    all_ids = {id(p) for _, p in ordered}
    assigned = set().union(*owner_ids.values())
    owner_ids["other_trainable"] = all_ids - assigned
    overlap = []
    for index, left in enumerate(GROUPS):
        for right in GROUPS[index + 1:]:
            overlap.extend(owner_ids[left] & owner_ids[right])
    union = set().union(*(owner_ids[group] for group in GROUPS))
    if overlap or union != all_ids:
        raise ValueError(f"parameter group overlap/uncovered: overlap={len(overlap)} uncovered={len(all_ids-union)}")
    grouped = {group: [] for group in GROUPS}
    inventory = []
    for order, (name, parameter) in enumerate(ordered):
        matches = [group for group in GROUPS if id(parameter) in owner_ids[group]]
        if len(matches) != 1:
            raise ValueError(f"parameter {name} has {len(matches)} owners")
        group = matches[0]
        grouped[group].append((name, parameter))
        inventory.append({"order": order, "parameter_name": name, "shape": json.dumps(list(parameter.shape)),
                          "numel": parameter.numel(), "dtype": str(parameter.dtype), "group": group,
                          "requires_grad": True})
    ordering_payload = [{"name": name, "shape": list(p.shape), "numel": p.numel(),
                         "group": next(group for group in GROUPS if id(p) in owner_ids[group])} for name, p in ordered]
    ordering_sha = hashlib.sha256(json.dumps(ordering_payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    justifications = {group: ("nonzero module-owned group" if grouped[group] else
        "zero-size because the selected checkpoint has no trainable parameters owned by this conceptual module set") for group in GROUPS}
    return inventory, grouped, ordering_sha, justifications


def exact_records(dataset_rows: list[dict[str, Any]], ids: set[str], name: str) -> list[dict[str, Any]]:
    selected = [dict(row) for row in dataset_rows if row_id(row) in ids]
    observed = [row_id(row) for row in selected]
    if len(observed) != len(ids) or len(set(observed)) != len(ids) or set(observed) != ids:
        raise ValueError(f"{name} exact row-ID join failed")
    return selected


def encode_records(helper: Any, runner_args: argparse.Namespace, records: list[dict[str, Any]],
                   vocab: Any, tokenizer: Any, max_length: int, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = helper.train
    transformed = train.apply_vnext_evidence_interface_to_records(
        records, getattr(runner_args, "vnext_evidence_interface", "full_evidence"))
    inputs = train._stage118_encode_inputs(transformed, args=runner_args, vocab=vocab, tokenizer=tokenizer,
                                           max_length=max_length, device=device)
    return transformed, inputs


def batch_outputs(helper: Any, model: Any, runner_args: argparse.Namespace, records: list[dict[str, Any]],
                  inputs: dict[str, Any], device: torch.device, batch_size: int):
    train = helper.train
    if not torch.is_grad_enabled():
        raise RuntimeError("Stage190-B batch forward requires gradients enabled")
    for start in range(0, len(records), batch_size):
        end = min(len(records), start + batch_size)
        batch_records = records[start:end]
        batch_inputs = train._stage43_slice_inputs(inputs, start, end)
        temporal, predicate = train.extract_flags(batch_records, getattr(runner_args, "flag_source", "controlled_heuristic"), device)
        features = train._vnext_model_feature_inputs(batch_inputs)
        train._assert_model_accepts_feature_kwargs(model, features, context="Stage190-B gradient diagnostic")
        output = model(**features, temporal_mismatch_flags=temporal, predicate_mismatch_flags=predicate)
        if not isinstance(output, dict):
            raise RuntimeError("model output must be a dictionary")
        logits = output.get("logits")
        frame_logit = output.get("frame_logit")
        labels = batch_inputs.get("final_labels")
        observed_batch = len(batch_records)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (observed_batch, 3):
            raise RuntimeError(f'output["logits"] must have shape [{observed_batch}, 3]')
        if not isinstance(frame_logit, torch.Tensor) or int(frame_logit.reshape(-1).numel()) != observed_batch:
            raise RuntimeError('output["frame_logit"] flattened count must equal batch size')
        if not isinstance(labels, torch.Tensor) or int(labels.reshape(-1).numel()) != observed_batch:
            raise RuntimeError("final_labels tensor length must equal batch size")
        label_values = {int(value) for value in labels.detach().reshape(-1).cpu().tolist()}
        allowed_label_ids = set(train.v5.FINAL_LABEL_TO_ID.values())
        if len(allowed_label_ids) != 3 or not label_values.issubset(allowed_label_ids):
            raise RuntimeError(f"final_labels contain an unproven class index: {sorted(label_values)}")
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(frame_logit).all()):
            raise RuntimeError("non-finite classification or frame logits")
        yield batch_records, batch_inputs, output

def accumulate_objective(helper: Any, model: Any, runner_args: argparse.Namespace,
                         encoded: tuple[list[dict[str, Any]], dict[str, Any]], device: torch.device,
                         batch_size: int, row_count: int,
                         loss_sum: Callable[[dict[str, Any], dict[str, Any]], torch.Tensor],
                         count_active: bool = False) -> tuple[float, int]:
    records, inputs = encoded
    if len(records) != row_count or row_count <= 0:
        raise ValueError(f"objective row count mismatch: {len(records)} != {row_count}")
    model.zero_grad(set_to_none=True)
    value_sum = 0.0
    active = 0
    with torch.enable_grad():
        if not torch.is_grad_enabled():
            raise RuntimeError("torch.enable_grad() did not enable Stage190-B gradients")
        for _rows, batch_inputs, output in batch_outputs(
            helper, model, runner_args, records, inputs, device, batch_size
        ):
            loss = loss_sum(output, batch_inputs)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise RuntimeError("objective batch sum is not a finite scalar")
            value_sum += float(loss.detach().double().cpu().item())
            if count_active:
                active += int((output["frame_logit"].detach() < 0.0).sum().cpu().item())
            loss.backward()
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            parameter.grad.div_(row_count)
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError("non-finite accumulated gradient")
    return value_sum / row_count, active

def copy_gradients(ordered: list[tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: (torch.zeros_like(p, device="cpu", dtype=torch.float32) if p.grad is None
                   else p.grad.detach().cpu().to(torch.float32).clone()) for name, p in ordered}


def gradient_stats(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor], names: list[str]) -> tuple[float, float, float, bool]:
    source_sq = target_sq = dot = 0.0
    finite = True
    for name in names:
        left, right = source[name].double(), target[name].double()
        finite = finite and bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
        source_sq += float(torch.sum(left * left).item())
        target_sq += float(torch.sum(right * right).item())
        dot += float(torch.sum(left * right).item())
    return source_sq, target_sq, dot, finite and all(math.isfinite(v) for v in (source_sq, target_sq, dot))


def execute(args: argparse.Namespace, log: Callable[[str], None]) -> dict[str, Any]:
    repo, manifest_path = args.repo_root.resolve(), args.manifest.resolve()
    manifest = read_json(manifest_path)
    blockers = []
    if manifest.get("runnable") is not True or manifest.get("blocking_reasons"):
        raise ValueError("Stage190-A manifest is not runnable")
    self_path = Path(__file__).resolve()
    helper_path = Path(manifest["checkpoint_helper_path"]).resolve()
    checkpoint_path = Path(manifest["checkpoint_path"]).resolve()
    stage182_candidate_runtime = (args.stage182b_dir.resolve() / "stage182b_candidate_localization.csv")
    stage182_controls_runtime = (args.stage182b_dir.resolve() / "stage182b_matched_control_pairs.csv")
    stage185_sidecar_runtime = (args.stage185a_dir.resolve() / "stage185a_controlled_train_integrity_sidecar.jsonl")
    identity_checks = {
        "diagnostic_script_sha256": file_sha(self_path) == manifest.get("diagnostic_script_sha256"),
        "checkpoint_helper_sha256": file_sha(helper_path) == manifest.get("checkpoint_helper_sha256"),
        "checkpoint_sha256": file_sha(checkpoint_path) == manifest.get("checkpoint_sha256"),
        "dataset_sha256": file_sha(Path(manifest["dataset_path"])) == manifest.get("dataset_sha256"),
        "training_commit": manifest.get("training_git_commit") == "bee2f5ad452d1d9f57b30f444d18835dbffdbecf",
        "trainer_sha256": manifest.get("trainer_sha256") == "24b01c5799c762772fe1700204afae59f8566898f65e7f3eefa4ac57ac6f126f",
        "stage182_candidate_path": stage182_candidate_runtime == Path(manifest["stage182b_candidate_localization_path"]).resolve(),
        "stage182_controls_path": stage182_controls_runtime == Path(manifest["stage182b_matched_control_pairs_path"]).resolve(),
        "stage185_sidecar_path": stage185_sidecar_runtime == Path(manifest["sidecar_path"]).resolve(),
    }
    for key, expected_sha in (manifest.get("artifact_hashes") or {}).items():
        path_key = {"run_provenance": "run_provenance_path", "training_report": "training_report_path",
                    "clean_dev_predictions": "clean_dev_predictions_path", "clean_dev_scalars": "clean_dev_scalars_path",
                    "stage189c_jsonl": "stage189c_jsonl_path", "stage189c_report": "stage189c_report_path",
                    "stage182b_candidate_localization": "stage182b_candidate_localization_path",
                    "stage182b_matched_control_pairs": "stage182b_matched_control_pairs_path"}.get(key)
        if path_key:
            identity_checks[f"artifact_{key}"] = file_sha(Path(manifest[path_key])) == expected_sha
    if not all(identity_checks.values()):
        raise ValueError(f"manifest identity gate failed: {[k for k,v in identity_checks.items() if not v]}")
    sidecar_rows = read_jsonl(stage185_sidecar_runtime)
    if semantic_sidecar_sha(sidecar_rows) != manifest.get("sidecar_semantic_sha256"):
        raise ValueError("sidecar semantic SHA mismatch")
    sidecar = index_rows(sidecar_rows, "Stage185-A sidecar")
    dataset_rows = read_jsonl(Path(manifest["dataset_path"]))
    dataset = index_rows(dataset_rows, "dataset")
    if set(sidecar) != set(dataset):
        raise ValueError("dataset and sidecar row-ID universes differ")
    status_ids = {status: {identifier for identifier, row in sidecar.items()
                  if row.get("split") == "train" and type(row.get("frame_compatible_label")) is int
                  and row.get("frame_compatible_label") == 1
                  and row.get("integrity_status") == status} for status in STATUS_COUNTS}
    dev_ids = {identifier for identifier, row in sidecar.items() if row.get("split") == "dev"}
    if any(len(status_ids[k]) != v for k, v in STATUS_COUNTS.items()) or len(dev_ids) != 720:
        raise ValueError("Stage185 cohort topology mismatch")
    candidate_rows = read_csv(stage182_candidate_runtime)
    control_rows = read_csv(stage182_controls_runtime)
    compatible_fn = {row["row_id"] for row in candidate_rows if row.get("native_error_direction") == "compatible_false_negative"}
    incompatible_fp = {row["row_id"] for row in candidate_rows if row.get("native_error_direction") == "incompatible_false_positive"}
    matched_controls = {row["control_row_id"] for row in control_rows if row.get("control_row_id")}
    failures = compatible_fn | incompatible_fp
    if [len(compatible_fn), len(incompatible_fp), len(matched_controls), len(failures)] != [13, 1, 14, 14]:
        raise ValueError("Stage182-B cohort topology mismatch")

    helper = load_module(helper_path, "stage190_checkpoint_helper")
    helper_trainer_path = Path(helper.train.__file__).resolve()
    if file_sha(helper_trainer_path) != manifest.get("trainer_sha256"):
        raise ValueError("runtime checkpoint helper trainer bytes differ from frozen trainer SHA")
    device = torch.device(args.device)
    state, metadata, payload = helper.load_checkpoint(checkpoint_path, device)
    payload["model_state_dict"] = state
    runner_cli = argparse.Namespace(data=Path(manifest["dataset_path"]), architecture=None,
        vnext_router_mode=None, backbone=None, prediction_export_schema=None, device=args.device)
    runner_args = helper.merged_runner_args(runner_cli, metadata)
    runtime_label_to_id = dict(helper.train.v5.FINAL_LABEL_TO_ID)
    runtime_id_to_label = dict(helper.train.v5.ID_TO_FINAL_LABEL)
    checkpoint_label_to_id = metadata.get("final_label_to_id")
    checkpoint_id_to_label = metadata.get("final_id_to_label")
    runtime_ids = list(runtime_label_to_id.values())
    label_mapping_contract_passed = (
        set(runtime_label_to_id) == {"REFUTE", "NOT_ENTITLED", "SUPPORT"}
        and len(runtime_label_to_id) == 3
        and all(type(index) is int for index in runtime_ids)
        and len(set(runtime_ids)) == 3
        and all(type(index) is int for index in runtime_id_to_label)
        and runtime_id_to_label == {
            index: label for label, index in runtime_label_to_id.items()
        }
        and checkpoint_label_to_id == runtime_label_to_id
        and checkpoint_id_to_label == runtime_id_to_label
        and isinstance(checkpoint_label_to_id, dict)
        and isinstance(checkpoint_id_to_label, dict)
        and all(type(index) is int for index in checkpoint_label_to_id.values())
        and all(type(index) is int for index in checkpoint_id_to_label)
        and all(
            checkpoint_id_to_label.get(index) == label
            and checkpoint_label_to_id.get(label) == index
            for label, index in checkpoint_label_to_id.items()
        )
    )
    if not label_mapping_contract_passed:
        raise ValueError(
            "selected-checkpoint final label mapping contract failed: "
            f"runtime_label_to_id={runtime_label_to_id!r} "
            f"runtime_id_to_label={runtime_id_to_label!r} "
            f"checkpoint_label_to_id={checkpoint_label_to_id!r} "
            f"checkpoint_id_to_label={checkpoint_id_to_label!r}"
        )
    support_id = runtime_label_to_id["SUPPORT"]
    ne_id = runtime_label_to_id["NOT_ENTITLED"]
    refute_id = runtime_label_to_id["REFUTE"]
    model, vocab, tokenizer, max_length = helper.build_eval_model(runner_args, payload, device)
    model.eval()
    inventory, grouped, ordering_sha, zero_justifications = parameter_groups(model)
    ordered = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable_before = named_tensor_sha(ordered)
    buffers_before = named_tensor_sha(model.named_buffers())

    cohorts = {
        "eligible": exact_records(dataset_rows, status_ids["ELIGIBLE"], "ELIGIBLE"),
        "ineligible": exact_records(dataset_rows, status_ids["INELIGIBLE"], "INELIGIBLE"),
        "unresolved": exact_records(dataset_rows, status_ids["UNRESOLVED"], "UNRESOLVED"),
        "clean_dev_all": exact_records(dataset_rows, dev_ids, "clean dev"),
        "compatible_fn": exact_records(dataset_rows, compatible_fn, "compatible FN"),
        "incompatible_fp": exact_records(dataset_rows, incompatible_fp, "incompatible FP"),
        "matched_controls": exact_records(dataset_rows, matched_controls, "matched controls"),
        "clean_model_failures": exact_records(dataset_rows, failures, "clean model failures"),
    }
    clean_support = [row for row in cohorts["clean_dev_all"] if row.get("final_label") == "SUPPORT"]
    if len(clean_support) != 89:
        raise ValueError(f"gold-SUPPORT clean-dev cohort must contain exactly 89 rows, got {len(clean_support)}")
    cohorts["clean_dev_support"] = clean_support
    encoded = {name: encode_records(helper, runner_args, rows, vocab, tokenizer, max_length, device)
               for name, rows in cohorts.items() if name in {"eligible", "ineligible", "unresolved", "clean_dev_all",
                                                              "clean_dev_support", "compatible_fn", "matched_controls"}}
    topology_rows = [{"cohort": name, "row_count": len(rows), "expected_row_count":
        (89 if name == "clean_dev_support" else EXPECTED_COHORTS.get(name)), "exact_join": True,
        "source_order": "dataset source order", "sampled_or_truncated": False} for name, rows in cohorts.items()]

    objectives: dict[str, tuple[str, Callable[[dict[str, Any], dict[str, Any]], torch.Tensor]]] = {
        "margin_eligible": ("eligible", lambda o, _i: F.relu(0.0 - o["frame_logit"]).sum()),
        "ce_eligible": ("eligible", lambda o, i: F.cross_entropy(o["logits"], i["final_labels"], reduction="sum")),
        "ce_clean_dev_all": ("clean_dev_all", lambda o, i: F.cross_entropy(o["logits"], i["final_labels"], reduction="sum")),
        "ce_clean_dev_support": ("clean_dev_support", lambda o, i: F.cross_entropy(o["logits"], i["final_labels"], reduction="sum")),
        "neg_support_vs_not_entitled_margin": ("clean_dev_support", lambda o, _i: -(o["logits"][:, support_id] - o["logits"][:, ne_id]).sum()),
        "neg_support_vs_max_other_margin": ("clean_dev_support", lambda o, _i: -(o["logits"][:, support_id] - torch.maximum(o["logits"][:, refute_id], o["logits"][:, ne_id])).sum()),
        "neg_mean_frame_logit_compatible_fn": ("compatible_fn", lambda o, _i: -o["frame_logit"].sum()),
        "neg_mean_frame_logit_matched_controls": ("matched_controls", lambda o, _i: -o["frame_logit"].sum()),
        "neg_mean_frame_logit_ineligible": ("ineligible", lambda o, _i: -o["frame_logit"].sum()),
        "neg_mean_frame_logit_unresolved": ("unresolved", lambda o, _i: -o["frame_logit"].sum()),
    }
    source_value, active_count = accumulate_objective(helper, model, runner_args, encoded["eligible"], device,
        args.batch_size, 605, objectives["margin_eligible"][1], count_active=True)
    source = copy_gradients(ordered)
    all_names = [name for name, _ in ordered]
    total_source_sq, _, _, source_finite = gradient_stats(source, source, all_names)
    if not source_finite or total_source_sq <= 0.0:
        raise ValueError("source margin gradient is non-finite or exactly zero")
    objective_rows = [{"objective": "margin_eligible", "cohort": "eligible", "row_count": 605,
                       "objective_mean": source_value, "gradient_norm": math.sqrt(total_source_sq),
                       "finite_gradient_gate": source_finite, "source_or_target": "source"}]
    group_rows = []
    directional_rows = []
    for objective, (cohort_name, function) in objectives.items():
        if objective == "margin_eligible":
            continue
        row_count = len(cohorts[cohort_name])
        value, _ = accumulate_objective(helper, model, runner_args, encoded[cohort_name], device,
                                         args.batch_size, row_count, function)
        target = copy_gradients(ordered)
        source_sq, target_sq, dot, finite = gradient_stats(source, target, all_names)
        cosine = dot / math.sqrt(source_sq * target_sq) if source_sq > 0.0 and target_sq > 0.0 else None
        projected = -dot
        directional_rows.append({"objective": objective, "row_count": row_count, "dot_product": dot,
            "cosine_similarity": cosine, "projected_target_change": projected,
            "projected_target_change_per_unit_margin_norm": projected / math.sqrt(source_sq) if source_sq > 0.0 else None,
            "projected_target_change_at_weight_0_05": 0.05 * projected,
            "sign_interpretation": "positive=worsens target loss; negative=improves target loss",
            "zero_source": source_sq == 0.0, "zero_target": target_sq == 0.0,
            "finite_value_gate": finite})
        objective_rows.append({"objective": objective, "cohort": cohort_name, "row_count": row_count,
            "objective_mean": value, "gradient_norm": math.sqrt(target_sq), "finite_gradient_gate": finite,
            "source_or_target": "target"})
        for group in METRIC_GROUPS:
            names = all_names if group == "overall" else [name for name, _ in grouped[group]]
            g_source_sq, g_target_sq, g_dot, g_finite = gradient_stats(source, target, names)
            group_rows.append({"objective": objective, "parameter_group": group,
                "source_gradient_squared_norm": g_source_sq, "target_gradient_squared_norm": g_target_sq,
                "dot_product": g_dot, "cosine_similarity": (g_dot / math.sqrt(g_source_sq * g_target_sq)
                    if g_source_sq > 0.0 and g_target_sq > 0.0 else None),
                "source_norm_fraction": g_source_sq / total_source_sq,
                "target_norm_fraction": g_target_sq / target_sq if target_sq > 0.0 else None,
                "zero_source": g_source_sq == 0.0, "zero_target": g_target_sq == 0.0,
                "finite_value_gate": g_finite})
        del target
        model.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    del source
    trainable_after = named_tensor_sha(ordered)
    buffers_after = named_tensor_sha(model.named_buffers())
    state_unchanged = trainable_before == trainable_after and buffers_before == buffers_after
    if not state_unchanged:
        blockers.append("model parameter or preserved-buffer state changed")
    finite_all = all(row["finite_gradient_gate"] for row in objective_rows) and all(row["finite_value_gate"] for row in group_rows + directional_rows)
    if not finite_all:
        blockers.append("one or more finite-gradient gates failed")
    # Source fractions are recovered from any target row because source statistics are target-independent.
    first_by_group = {row["parameter_group"]: row for row in group_rows if row["objective"] == "ce_eligible"}
    shared_fraction = sum(first_by_group[g]["source_gradient_squared_norm"] for g in SHARED_GROUPS) / max(total_source_sq, EPSILON)
    frame_fraction = first_by_group["frame_head"]["source_gradient_squared_norm"] / max(total_source_sq, EPSILON)
    decision_fraction = first_by_group["decision_head"]["source_gradient_squared_norm"] / max(total_source_sq, EPSILON)
    source_fraction_sum = shared_fraction + frame_fraction + decision_fraction
    source_fraction_difference = abs(source_fraction_sum - 1.0)
    source_fraction_partition_passed = (
        all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in
            (shared_fraction, frame_fraction, decision_fraction))
        and source_fraction_difference <= FRACTION_TOLERANCE
    )
    if not source_fraction_partition_passed:
        blockers.append("source-gradient fraction partition failed")
    objective_row_counts = {row["objective"]: row["row_count"] for row in objective_rows}
    if objective_row_counts != EXPECTED_OBJECTIVE_ROW_COUNTS:
        blockers.append("exact objective row-count contract failed")
    group_pairs = [(row["objective"], row["parameter_group"]) for row in group_rows]
    expected_group_pairs = {(target, group) for target in TARGET_OBJECTIVES for group in METRIC_GROUPS}
    group_grid_passed = (
        len(group_rows) == EXPECTED_GROUP_METRIC_ROWS
        and len(set(group_pairs)) == EXPECTED_GROUP_METRIC_ROWS
        and set(group_pairs) == expected_group_pairs
    )
    if not group_grid_passed:
        blockers.append("exact 54-row target/group metric grid failed")
    directional_names = [row["objective"] for row in directional_rows]
    directional_grid_passed = (
        len(directional_rows) == EXPECTED_DIRECTIONAL_ROWS
        and len(set(directional_names)) == EXPECTED_DIRECTIONAL_ROWS
        and set(directional_names) == set(TARGET_OBJECTIVES)
    )
    if not directional_grid_passed:
        blockers.append("exact nine-row directional derivative grid failed")
    trainable_parameter_count = len(ordered)
    trainable_parameter_numel = sum(int(parameter.numel()) for _, parameter in ordered)
    write_csv(args.output_dir / "stage190b_parameter_inventory.csv",
              ["order", "parameter_name", "shape", "numel", "dtype", "group", "requires_grad"], inventory)
    write_csv(args.output_dir / "stage190b_objective_summary.csv",
              ["objective", "cohort", "row_count", "objective_mean", "gradient_norm", "finite_gradient_gate", "source_or_target"], objective_rows)
    write_csv(args.output_dir / "stage190b_group_gradient_metrics.csv",
              ["objective", "parameter_group", "source_gradient_squared_norm", "target_gradient_squared_norm", "dot_product",
               "cosine_similarity", "source_norm_fraction", "target_norm_fraction", "zero_source", "zero_target", "finite_value_gate"], group_rows)
    write_csv(args.output_dir / "stage190b_directional_derivatives.csv",
              ["objective", "row_count", "dot_product", "cosine_similarity", "projected_target_change",
               "projected_target_change_per_unit_margin_norm", "projected_target_change_at_weight_0_05",
               "sign_interpretation", "zero_source", "zero_target", "finite_value_gate"], directional_rows)
    write_csv(args.output_dir / "stage190b_cohort_topology.csv",
              ["cohort", "row_count", "expected_row_count", "exact_join", "source_order", "sampled_or_truncated"], topology_rows)
    decision = EXPORTED if not blockers else BLOCKED
    report = {"stage": "Stage190-B", "decision": decision, "blocking_reasons": blockers,
        "evaluation_only": True, "training_performed": False, "optimizer_created": False,
        "optimizer_step_performed": False, "scheduler_created": False, "checkpoint_selection_performed": False,
        "threshold_tuning_performed": False, "amp_used": False, "external_data_used": False,
        "model_state_unchanged": state_unchanged, "trainable_state_sha256_before": trainable_before,
        "trainable_state_sha256_after": trainable_after, "buffer_state_sha256_before": buffers_before,
        "buffer_state_sha256_after": buffers_after, "source_score": 'direct output["frame_logit"]',
        "classifier_source": 'output["logits"]', "loss_logits_used": False,
        "selected_checkpoint_path": str(checkpoint_path), "selected_checkpoint_sha256": manifest["checkpoint_sha256"],
        "training_seed": manifest["training_seed"], "split_seed": manifest["split_seed"], "arm": manifest["arm"],
        "training_git_commit": manifest["training_git_commit"], "diagnostic_git_commit": manifest["diagnostic_git_commit"],
        "trainer_sha256": manifest["trainer_sha256"], "diagnostic_script_sha256": manifest["diagnostic_script_sha256"],
        "checkpoint_helper_sha256": manifest["checkpoint_helper_sha256"], "artifact_hashes": manifest["artifact_hashes"],
        "cohort_topology": {row["cohort"]: row["row_count"] for row in topology_rows},
        "clean_dev_support_row_count": len(clean_support),
        "gold_support_clean_dev_row_ids_sha256": hashlib.sha256("\n".join(row_id(r) or "" for r in clean_support).encode()).hexdigest(),
        "label_to_index_mapping": runtime_label_to_id,
        "index_to_label_mapping": runtime_id_to_label,
        "checkpoint_label_to_index_mapping": checkpoint_label_to_id,
        "checkpoint_index_to_label_mapping": checkpoint_id_to_label,
        "label_mapping_contract_passed": label_mapping_contract_passed,
        "parameter_ordering_sha256": ordering_sha,
        "parameter_group_contract": {"groups": list(GROUPS), "shared_groups": list(SHARED_GROUPS),
            "zero_size_justifications": zero_justifications, "disjoint": True, "exhaustive": True},
        "objective_row_counts": objective_row_counts,
        "expected_objective_row_counts": dict(EXPECTED_OBJECTIVE_ROW_COUNTS),
        "parameter_group_metric_expected_rows": EXPECTED_GROUP_METRIC_ROWS,
        "parameter_group_metric_observed_rows": len(group_rows),
        "parameter_group_metric_grid_passed": group_grid_passed,
        "directional_derivative_expected_rows": EXPECTED_DIRECTIONAL_ROWS,
        "directional_derivative_observed_rows": len(directional_rows),
        "directional_derivative_grid_passed": directional_grid_passed,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_numel": trainable_parameter_numel,
        "model_grad_enabled_during_diagnostic": True,
        "active_eligible_row_count": active_count, "source_margin_loss": source_value,
        "total_margin_gradient_norm": math.sqrt(total_source_sq), "epsilon": EPSILON,
        "shared_margin_gradient_fraction": shared_fraction, "frame_head_margin_gradient_fraction": frame_fraction,
        "decision_head_margin_gradient_fraction": decision_fraction,
        "source_fraction_sum": source_fraction_sum,
        "source_fraction_partition_difference": source_fraction_difference,
        "source_fraction_partition_tolerance": FRACTION_TOLERANCE,
        "source_fraction_partition_passed": source_fraction_partition_passed,
        "finite_gradient_gates_passed": finite_all,
        "objectives": {row["objective"]: row for row in objective_rows},
        "directional_derivatives": {row["objective"]: row for row in directional_rows},
        "group_gradient_metrics": group_rows}
    return report


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "stage190b_execution.log"
    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as h:
            h.write(message + "\n")
    report_path = output / "stage190b_gradient_report.json"
    try:
        log("Stage190-B gradient-enabled evaluation started; no training or optimizer is permitted.")
        report = execute(args, log)
        if report["decision"] != EXPORTED:
            raise RuntimeError("post-execution fail-closed gate: " + " | ".join(report["blocking_reasons"]))
        write_json(report_path, report)
        log("Stage190-B exported; model state unchanged.")
        return 0
    except BaseException as exc:
        detail = f"{type(exc).__name__}: {exc}"
        log(detail)
        log(traceback.format_exc())
        for name in OUTPUTS:
            path = output / name
            try:
                if path.exists():
                    path.unlink()
            except OSError as cleanup_exc:
                log(f"cleanup failed for {path}: {cleanup_exc}")
        fail = {"stage": "Stage190-B", "decision": BLOCKED, "blocking_reasons": [detail],
                "evaluation_only": True, "training_performed": False, "optimizer_created": False,
                "optimizer_step_performed": False, "model_state_unchanged": False,
                "source_score": 'direct output["frame_logit"]', "classifier_source": 'output["logits"]'}
        write_json(report_path, fail)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
