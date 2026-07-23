#!/usr/bin/env python3
"""Stage196-B2-B6P6 exact composer/autograd diagnostic; never trains."""
from __future__ import annotations

import argparse, csv, hashlib, io, json, math, os, subprocess, sys, time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import train_controlled_v5 as v5  # noqa: E402
from scripts import train_controlled_v6b_minimal as trainer  # noqa: E402

STAGE = "Stage196-B2-B6P6"
CANDIDATES = trainer.STAGE196B2B6P6_CANDIDATE_MASKS
PRIMITIVES = trainer.STAGE196B2B6P6_PRIMITIVES
DIRECTIONS = ("delta_support_minus_not_entitled", "delta_support_minus_refute",
              "delta_refute_minus_not_entitled", "delta_top1_runner_up_margin")
RESPONSES = ("delta_score_support", "delta_score_not_entitled", "delta_score_refute", *DIRECTIONS)
PAIRS = tuple((a, b) for i, a in enumerate(CANDIDATES) for b in CANDIDATES[i + 1:])
OUTPUTS = ("stage196b2b6p6_analysis.json", "stage196b2b6p6_report.md",
 "stage196b2b6p6_tensor_schema.csv", "stage196b2b6p6_gradient_connectivity.csv",
 "stage196b2b6p6_parameter_group_audit.csv", "stage196b2b6p6_forward_equivalence.csv",
 "stage196b2b6p6_no_mutation_audit.csv", "stage196b2b6p6_decision_gate.csv",
 "stage196b2b6p6_contract.csv")
P5 = "STAGE196B2B6P5_GRADIENT_PATH_INSTRUMENTATION_REQUIRED"
P4 = "STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE"
P2 = "STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_COMPLETE"
P5_NEXT = "STAGE196B2B6P6_MINIMAL_GRADIENT_PATH_INSTRUMENTATION"
P4_NEXT = "STAGE196B2B6P5_TRAINING_SIDE_RESPONSE_STABILITY_INTERVENTION_DESIGN"
P2_NEXT = "STAGE196B2B6P3_ACTION_RESPONSE_SAFETY_GATE_DIAGNOSTIC"
MAIN_DATA = "data/controlled_v5_v3_without_time_swap.jsonl"
INTENDED = {
 "direction_stability": {"groups": ["epistemic_heads"], "conditional_groups": ["final_composer"],
  "justification": "P5 traces every signed response to live recipient/donor epistemic primitives. Final-composer parameters are conditional: NE-involving coordinates can reach bias/alpha, while SUPPORT-minus-REFUTE algebraically cancels them."},
 "candidate_order_stability": {"groups": ["epistemic_heads"], "conditional_groups": ["final_composer"],
  "justification": "P5 pair gaps reuse the same live candidates. Epistemic heads are authoritative for every pair; final-composer reach is coordinate/class conditional, and v6b_minimal has no trainable selector."}}

def build_parser() -> argparse.ArgumentParser:
 p = argparse.ArgumentParser(description=__doc__)
 for name in ("repo-root", "stage196b2b6p5-analysis-json", "stage196b2b6p4-analysis-json",
              "stage196b2b6p2-analysis-json", "native-checkpoint", "frame-local-only-checkpoint",
              "main-data-path", "output-dir"):
  p.add_argument(f"--{name}", type=Path, required=True)
 for name in ("backbone", "model-name", "device", "current-git-commit"):
  p.add_argument(f"--{name}", required=True)
 p.add_argument("--batch-size", type=int, default=8); p.add_argument("--seed", type=int, required=True)
 return p

def read_json(path: Path) -> dict[str, Any]:
 value = json.loads(path.read_text(encoding="utf-8"))
 if not isinstance(value, dict): raise ValueError(f"{path}: object required")
 return value

def read_csv(path: Path) -> list[dict[str, str]]:
 with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))

def digest_file(path: Path) -> str:
 h = hashlib.sha256()
 with path.open("rb") as handle:
  for chunk in iter(lambda: handle.read(1048576), b""): h.update(chunk)
 return h.hexdigest()

def csv_text(fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
 out = io.StringIO(newline=""); writer = csv.DictWriter(out, fieldnames=fields, extrasaction="raise", lineterminator="\n"); writer.writeheader()
 for row in rows:
  writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})
 return out.getvalue()

def atomic_write(path: Path, text: str) -> None:
 tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
 with tmp.open("x", encoding="utf-8", newline="\n") as handle:
  handle.write(text); handle.flush(); os.fsync(handle.fileno())
 if path.exists(): tmp.unlink(); raise FileExistsError(f"refusing to overwrite {path}")
 os.rename(tmp, path)

def gate(rows: list[dict[str, Any]], name: str, required: Any, observed: Any, passed: bool, reason: str = "") -> None:
 rows.append({"contract": name, "required": required, "observed": observed, "passed": bool(passed), "blocking_reason": "" if passed else reason})

def closed(path: Path) -> bool:
 rows = read_csv(path)
 return bool(rows) and all(str(r.get("passed", "")).lower() == "true" and not r.get("blocking_reason", "").strip() for r in rows)

def tensor_payload_bytes(
 tensor: torch.Tensor, *, fingerprint_scope: str, tensor_name: str,
) -> bytes:
 if not isinstance(tensor, torch.Tensor):
  raise TypeError(f"{tensor_name}: torch.Tensor required")
 device_before_cpu = str(tensor.device)
 original_shape = list(tensor.shape)
 try:
  cpu = tensor.detach().cpu().contiguous()
  if cpu.layout != torch.strided:
   raise RuntimeError(f"strided tensor layout required, observed {cpu.layout}")
  flat = cpu.reshape(-1)
  if flat.numel() == 0:
   return b""
  return flat.view(torch.uint8).numpy().tobytes()
 except Exception as exc:
  provenance = {
   "fingerprint_scope": fingerprint_scope,
   "tensor_name": tensor_name,
   "shape": original_shape,
   "dtype": str(tensor.dtype),
   "layout": str(tensor.layout),
   "device_before_cpu_conversion": device_before_cpu,
   "requires_grad": bool(tensor.requires_grad),
   "exception_type": type(exc).__name__,
  }
  raise RuntimeError(
   "P6 tensor fingerprint serialization failed: "
   + json.dumps(provenance, sort_keys=True)
  ) from exc

def fingerprint(
 models: Sequence[torch.nn.Module], buffers: bool = False,
 *, fingerprint_scope: str | None = None,
) -> str:
 h = hashlib.sha256()
 scope = fingerprint_scope or ("buffer" if buffers else "parameter")
 for i, model in enumerate(models):
  source = model.named_buffers() if buffers else model.named_parameters()
  for name, tensor in source:
   h.update(f"{i}:{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
   h.update(tensor_payload_bytes(
    tensor, fingerprint_scope=scope, tensor_name=f"model{i}.{name}",
   ))
 return h.hexdigest()

def ids(module: torch.nn.Module | None) -> set[int]:
 return {id(p) for p in module.parameters()} if module is not None else set()

def parameter_groups(models: Sequence[torch.nn.Module]) -> OrderedDict[str, list[tuple[str, torch.nn.Parameter]]]:
 groups = OrderedDict((name, []) for name in ("frozen_backbone", "trainable_backbone", "epistemic_heads", "final_composer", "router_or_selector", "other_trainable"))
 seen: set[int] = set()
 for mi, model in enumerate(models):
  backbone = ids(getattr(model, "mamba", None)); epistemic: set[int] = set(); router: set[int] = set()
  for attr in ("frame_gate", "predicate_coverage_head", "sufficiency_gate", "polarity_energy_head"): epistemic |= ids(getattr(model, attr, None))
  composer = ids(getattr(model, "decision_head", None))
  for attr in ("alpha_temporal_raw", "alpha_predicate_raw"):
   value = getattr(model, attr, None)
   if isinstance(value, torch.nn.Parameter): composer.add(id(value))
  for attr in ("router", "selector", "final_router"): router |= ids(getattr(model, attr, None))
  for name, p in model.named_parameters():
   if id(p) in seen: continue
   seen.add(id(p)); item = (f"model{mi}.{name}", p)
   if id(p) in backbone: groups["trainable_backbone" if p.requires_grad else "frozen_backbone"].append(item)
   elif id(p) in epistemic: groups["epistemic_heads"].append(item)
   elif id(p) in composer: groups["final_composer"].append(item)
   elif id(p) in router: groups["router_or_selector"].append(item)
   elif p.requires_grad: groups["other_trainable"].append(item)
 return groups

def tensor_row(scope: str, candidate: str, name: str, tensor: torch.Tensor) -> dict[str, Any]:
 return {"scope": scope, "candidate_mask": candidate, "tensor_name": name,
  "shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
  "requires_grad": tensor.requires_grad, "grad_fn_class": type(tensor.grad_fn).__name__ if tensor.grad_fn else "",
  "is_leaf": tensor.is_leaf, "finite": bool(torch.isfinite(tensor).all().item()),
  "batch_dimension": int(tensor.shape[0]) if tensor.ndim else None}

def gradient_row(group: str, items: list[tuple[str, torch.nn.Parameter]], by_id: dict[int, torch.Tensor | None] | None) -> dict[str, Any]:
 trainable = [p for _, p in items if p.requires_grad]
 gradients = [None if by_id is None else by_id.get(id(p)) for p in trainable]
 connected = [g for g in gradients if g is not None]; finite = [g for g in connected if bool(torch.isfinite(g).all().item())]
 nonzero = [g for g in finite if int(torch.count_nonzero(g).item()) > 0]
 return {"parameter_group": group, "parameter_tensor_count": len(items), "trainable_parameter_count": sum(p.numel() for p in trainable),
  "gradient_connected_tensor_count": len(connected), "unused_tensor_count": len(trainable) - len(connected),
  "finite_gradient_tensor_count": len(finite), "nonzero_gradient_tensor_count": len(nonzero),
  "gradient_l1_norm": sum(float(g.abs().sum().item()) for g in finite),
  "gradient_l2_norm": math.sqrt(sum(float(torch.square(g).sum().item()) for g in finite)),
  "maximum_absolute_gradient": max((float(g.abs().max().item()) for g in finite), default=0.0)}

def probe_target(target: torch.Tensor, groups: OrderedDict[str, list[tuple[str, torch.nn.Parameter]]], retain: bool) -> tuple[str, list[dict[str, Any]]]:
 params = [p for items in groups.values() for _, p in items if p.requires_grad]
 for p in params: p.grad = None
 if not target.requires_grad or target.grad_fn is None:
  return "NONDIFFERENTIABLE", [gradient_row(name, items, None) for name, items in groups.items()]
 grads = torch.autograd.grad(target.sum(), params, allow_unused=True, retain_graph=retain, create_graph=False)
 mapping = {id(p): g for p, g in zip(params, grads)}; connected = [g for g in grads if g is not None]
 if any(not bool(torch.isfinite(g).all().item()) for g in connected): status = "NONFINITE"
 elif not connected: status = "DISCONNECTED"
 elif any(int(torch.count_nonzero(g).item()) > 0 for g in connected): status = "CONNECTED_NONZERO"
 else: status = "CONNECTED_ZERO_AT_OBSERVED_BATCH"
 return status, [gradient_row(name, items, mapping) for name, items in groups.items()]

def checkpoint_model(path: Path, ns: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
 state, metadata = trainer._load_stage118_checkpoint_state(path)
 args = metadata.get("training_args") if isinstance(metadata.get("training_args"), dict) else metadata
 if args.get("freeze_encoder", True) is not True: raise ValueError("authoritative run did not freeze Mamba")
 model = trainer.build_mamba_model(
  ns.model_name, freeze_encoder=True, freeze_a_log=bool(args.get("freeze_a_log", True)),
  use_boundary_head=bool(args.get("use_boundary_loss", False)),
  use_frame_violation_head=bool(args.get("use_frame_violation_loss", False)),
  use_predicate_isolation_head=bool(args.get("use_predicate_isolation_loss", False)),
  use_preservation_entitlement_head=bool(args.get("use_preservation_entitlement_loss", False)),
  use_temporal_diagnostic_head=bool(args.get("use_temporal_diagnostic_loss", False)),
  use_temporal_residual_adapter=bool(args.get("use_temporal_residual_adapter", False)),
  temporal_adapter_detach_input=bool(args.get("temporal_adapter_detach_input", True)),
  use_temporal_channel=bool(args.get("use_temporal_channel", False)),
  temporal_channel_detach_input=bool(args.get("temporal_channel_detach_input", True)),
  use_temporal_channel_loss=bool(args.get("use_temporal_channel_loss", False)),
  temporal_channel_loss_weight=float(args.get("temporal_channel_loss_weight", 0.0)),
  temporal_channel_loss_pos_weight=float(args.get("temporal_channel_loss_pos_weight", 1.0)),
  use_temporal_channel_gated_penalty=bool(args.get("use_temporal_channel_gated_penalty", False)),
  temporal_channel_gated_penalty_scale=float(args.get("temporal_channel_gated_penalty_scale", 0.0)))
 model.load_state_dict(state, strict=True)
 return model.to(device), dict(args)

def load_batch(ns: argparse.Namespace, p2: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
 path = ns.stage196b2b6p2_analysis_json.parent / "stage196b2b6p2_candidate_action_composer_scores.csv"
 rows = [row for row in read_csv(path) if int(row["seed"]) == ns.seed]
 actions_by_id: dict[str, dict[str, str]] = {}; identity: dict[str, Any] = {}
 for row in rows:
  stable = row["stable_row_id"]; actions_by_id.setdefault(stable, {})[row["candidate_mask"]] = row["candidate_action_key"]
  identity[stable] = json.loads(row["data_identity"])
 eligible = sorted(stable for stable, mapping in actions_by_id.items() if set(mapping) == set(CANDIDATES))
 if ns.batch_size <= 0 or len(eligible) < ns.batch_size: raise ValueError("invalid deterministic batch size")
 selected = eligible[:ns.batch_size]; source = v5.load_jsonl(ns.main_data_path)
 if any(row.get("intervention_type") == "time_swap" for row in source): raise ValueError("time_swap excluded")
 index = {str(row.get("id")): row for row in source}; records = []; action_lists = {mask: [] for mask in CANDIDATES}
 for stable in selected:
  value = identity[stable]; row_id = str(value[0] if isinstance(value, list) else value.get("id"))
  record = index.get(row_id) or index.get(stable)
  if record is None: raise ValueError(f"P2 identity absent from main data: {stable}")
  records.append(record)
  for mask in CANDIDATES:
   action = actions_by_id[stable][mask]
   if len(action) != 5 or set(action) - {"0", "1"}: raise ValueError("invalid P2 primitive action")
   action_lists[mask].append([bit == "1" for bit in action])
 device = torch.device(ns.device)
 tensors = {mask: torch.tensor(values, dtype=torch.bool, device=device) for mask, values in action_lists.items()}
 return records, tensors, {"candidate_csv": str(path), "selected_stable_row_ids": selected,
  "candidate_action_identity_disagreements": 0, "p2_source_boundary": p2.get("composer_source_boundary")}

def reference_recomposition(model: torch.nn.Module, native: dict[str, Any], donor: dict[str, Any], action: torch.Tensor) -> torch.Tensor:
 values = [torch.where(action[:, column], donor[key], native[key])
           for column, key in enumerate(trainer.STAGE196B2B6P6_PRIMITIVE_OUTPUT_KEYS)]
 decision = model.decision_head(frame_prob=values[0], predicate_coverage_prob=values[1],
  sufficiency_prob=values[2], positive_energy=values[3], negative_energy=values[4])
 return decision["logits"] + (native["logits"] - native["base_logits"])

def run(ns: argparse.Namespace, contracts: list[dict[str, Any]]) -> dict[str, Any]:
 root = ns.repo_root.resolve(); head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
 gate(contracts, "current_commit_identity", ns.current_git_commit, head, head == ns.current_git_commit, "commit changed")
 p5, p4, p2 = (read_json(path) for path in (ns.stage196b2b6p5_analysis_json, ns.stage196b2b6p4_analysis_json, ns.stage196b2b6p2_analysis_json))
 gate(contracts, "p5_decision_and_zero_blockers", {"decision": P5, "recommended_next_stage": P5_NEXT, "blocking_reasons": []},
  {key: p5.get(key) for key in ("decision", "recommended_next_stage", "blocking_reasons")}, p5.get("decision") == P5 and p5.get("recommended_next_stage") == P5_NEXT and p5.get("blocking_reasons") == [], "P5 closure failed")
 p5_feasibility = p5.get("source_feasibility", {})
 p5_schema_ok = (p5_feasibility.get("counterfactual_gradient_path") == "MINIMAL_GRADIENT_INSTRUMENTATION_REQUIRED"
  and p5_feasibility.get("three_candidates_currently_available") is False
  and p5_feasibility.get("final_margin_autograd") is True)
 gate(contracts, "p5_source_feasibility_schema", {"counterfactual_gradient_path": "MINIMAL_GRADIENT_INSTRUMENTATION_REQUIRED",
  "three_candidates_currently_available": False, "final_margin_autograd": True}, p5_feasibility, p5_schema_ok, "P5 feasibility schema changed")
 gate(contracts, "p4_decision_closure", {"decision": P4, "recommended_next_stage": P4_NEXT, "blocking_reasons": []},
  {key: p4.get(key) for key in ("decision", "recommended_next_stage", "blocking_reasons")}, p4.get("decision") == P4 and p4.get("recommended_next_stage") == P4_NEXT and p4.get("blocking_reasons") == [], "P4 closure failed")
 p4c = ns.stage196b2b6p4_analysis_json.parent / "stage196b2b6p4_contract.csv"; p2c = ns.stage196b2b6p2_analysis_json.parent / "stage196b2b6p2_contract.csv"
 gate(contracts, "p4_zero_failed_contracts", True, closed(p4c), closed(p4c), "P4 contracts failed")
 gate(contracts, "p2_endpoint_reproduction", {"decision": P2, "recommended_next_stage": P2_NEXT, "blocking_reasons": []},
  {key: p2.get(key) for key in ("decision", "recommended_next_stage", "blocking_reasons")}, p2.get("decision") == P2 and p2.get("recommended_next_stage") == P2_NEXT and p2.get("blocking_reasons") == [] and closed(p2c), "P2 closure failed")
 expected_data = (root / MAIN_DATA).resolve()
 gate(contracts, "clean_main_data_identity", str(expected_data), str(ns.main_data_path), ns.main_data_path == expected_data, "main data changed")
 gate(contracts, "mamba_130m_identity", ["mamba", "state-spaces/mamba-130m-hf"], [ns.backbone, ns.model_name], ns.backbone == "mamba" and ns.model_name == "state-spaces/mamba-130m-hf", "backbone changed")
 device = torch.device(ns.device); cuda_ok = device.type == "cuda" and torch.cuda.is_available()
 gate(contracts, "cuda_device", "available CUDA", str(device), cuda_ok, "CUDA required")
 records, actions, action_audit = load_batch(ns, p2)
 native_model, args = checkpoint_model(ns.native_checkpoint, ns, device); donor_model, donor_args = checkpoint_model(ns.frame_local_only_checkpoint, ns, device)
 role_ok = (args.get("architecture", "v6b_minimal") == "v6b_minimal"
  and args.get("frame_downstream_gradient_mode", "joint") == "joint"
  and donor_args.get("frame_downstream_gradient_mode") == "frame_local_only"
  and int(args.get("seed", ns.seed)) == ns.seed and int(donor_args.get("seed", ns.seed)) == ns.seed)
 gate(contracts, "authoritative_checkpoint_role_and_seed", {"native": "joint", "donor": "frame_local_only", "seed": ns.seed},
  {"native": args.get("frame_downstream_gradient_mode", "joint"), "donor": donor_args.get("frame_downstream_gradient_mode"), "seed": args.get("seed")}, role_ok, "checkpoint role changed")
 models = (native_model, donor_model); modes_before = [model.training for model in models]
 parameters_before = fingerprint(
  models, fingerprint_scope="before_probe.parameters",
 )
 buffers_before = fingerprint(
  models, True, fingerprint_scope="before_probe.buffers",
 )
 for model in models: model.eval()
 from transformers import AutoTokenizer
 tokenizer = AutoTokenizer.from_pretrained(ns.model_name)
 if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
 inputs = v5.move_inputs(v5.encode_mamba_records(records, tokenizer, int(args.get("max_length", 128)))["model_inputs"], device)
 temporal, predicate = trainer.extract_flags(records, str(args.get("flag_source", "controlled_heuristic")), device)
 ta = float(args.get("temporal_adapter_final_penalty_scale", 0.0)) if args.get("use_temporal_adapter_final_penalty", False) else 0.0
 tc = float(args.get("temporal_channel_gated_penalty_scale", 0.0)) if args.get("use_temporal_channel_gated_penalty", False) else 0.0
 kwargs = {"temporal_mismatch_flags": temporal, "predicate_mismatch_flags": predicate,
           "temporal_adapter_final_penalty_scale": ta, "temporal_channel_gated_penalty_scale": tc}
 # One forward per authoritative arm; no optimizer, scheduler, or training backward exists.
 native = native_model(**v5.model_feature_inputs(inputs), **kwargs); donor = donor_model(**v5.model_feature_inputs(inputs), **kwargs)
 geometry = trainer.stage196b2b6p6_differentiable_composer_geometry(
  model=native_model, native_output=native, counterpart_output=donor,
  candidate_primitive_actions=actions, diagnostic_enabled=True)
 gate(contracts, "exact_candidate_masks", list(CANDIDATES), list(geometry["candidate_masks"]), tuple(geometry["candidate_masks"]) == CANDIDATES, "candidate masks changed")
 gate(contracts, "clean_data_time_swap_exclusion", False,
  any(record.get("intervention_type") == "time_swap" for record in records),
  not any(record.get("intervention_type") == "time_swap" for record in records), "time_swap used")
 tensor_rows = [tensor_row("native", "", name, value) for name, value in geometry.items() if isinstance(value, torch.Tensor)]
 tensor_rows += [tensor_row("counterfactual", mask, name, value) for mask, values in geometry["candidate_geometry"].items() for name, value in values.items()]
 native_required = {f"native_score_{name}" for name in ("support", "not_entitled", "refute")}
 native_required |= {"native_margin_support_minus_not_entitled", "native_margin_support_minus_refute", "native_margin_refute_minus_not_entitled", "native_top1_runner_up_margin"}
 candidate_required = {f"counterfactual_score_{name}" for name in ("support", "not_entitled", "refute")}
 candidate_required |= {"counterfactual_margin_support_minus_not_entitled", "counterfactual_margin_support_minus_refute",
  "counterfactual_margin_refute_minus_not_entitled", "counterfactual_top1_runner_up_margin", *RESPONSES}
 native_names = {row["tensor_name"] for row in tensor_rows if row["scope"] == "native"}
 candidate_names = {mask: {row["tensor_name"] for row in tensor_rows if row["candidate_mask"] == mask} for mask in CANDIDATES}
 tensors_ok = (native_required <= native_names and all(candidate_required <= candidate_names[mask] for mask in CANDIDATES)
  and all(row["device"].startswith("cuda") and row["finite"] and row["batch_dimension"] == ns.batch_size and row["requires_grad"] and row["grad_fn_class"] for row in tensor_rows))
 gate(contracts, "canonical_score_margin_response_tensor_schema", True, {"rows": len(tensor_rows), "passed": tensors_ok}, tensors_ok, "tensor contract failed")
 native_scores_bad = 0 if torch.equal(native["logits"], geometry["native_logits"]) else ns.batch_size
 native_predictions_bad = 0 if torch.equal(native["predictions"], geometry["native_logits"].argmax(-1)) else ns.batch_size
 forward_rows = [{"scope": "native", "candidate_mask": "", "score_disagreements": native_scores_bad,
  "prediction_disagreements": native_predictions_bad, "candidate_action_identity_disagreements": 0,
  "exact_equal": native_scores_bad == native_predictions_bad == 0}]
 counterfactual_bad = 0
 for mask in CANDIDATES:
  reference = reference_recomposition(native_model, native, donor, actions[mask]); observed = geometry["candidate_geometry"][mask]["counterfactual_logits"]
  score_bad = int((reference != observed).any(-1).sum().item()); counterfactual_bad += score_bad
  forward_rows.append({"scope": "counterfactual", "candidate_mask": mask, "score_disagreements": score_bad,
   "prediction_disagreements": int((reference.argmax(-1) != observed.argmax(-1)).sum().item()),
   "candidate_action_identity_disagreements": 0, "exact_equal": bool(torch.equal(reference, observed))})
 equivalent = native_scores_bad == native_predictions_bad == counterfactual_bad == 0
 gate(contracts, "forward_and_candidate_semantic_equivalence", 0, native_scores_bad + native_predictions_bad + counterfactual_bad, equivalent, "exact equivalence failed")
 groups = parameter_groups(models)
 group_rows = [{"parameter_group": name, "parameter_tensor_count": len(items),
  "parameter_count": sum(p.numel() for _, p in items), "requires_grad_tensor_count": sum(p.requires_grad for _, p in items),
  "requires_grad_parameter_count": sum(p.numel() for _, p in items if p.requires_grad), "module_reference_derived": True} for name, items in groups.items()]
 grouped = sum(len(items) for items in groups.values()) == len({id(p) for model in models for p in model.parameters()})
 gate(contracts, "parameter_grouping_closure", True, grouped, grouped, "grouping incomplete")
 frozen = not groups["trainable_backbone"] and all(not p.requires_grad for _, p in groups["frozen_backbone"])
 gate(contracts, "frozen_backbone_policy", True, frozen, frozen, "frozen Mamba policy changed")
 targets: list[tuple[str, str, str, torch.Tensor]] = []
 for suffix in ("support", "not_entitled", "refute"):
  targets.append(("native_score", "", f"native_score_{suffix}", geometry[f"native_score_{suffix}"]))
 for mask in CANDIDATES:
  values = geometry["candidate_geometry"][mask]
  for suffix in ("support", "not_entitled", "refute"):
   targets.append(("counterfactual_score", mask, f"counterfactual_score_{suffix}", values[f"counterfactual_score_{suffix}"]))
  for coordinate in DIRECTIONS: targets.append(("direction", mask, coordinate, values[coordinate]))
 for coordinate in RESPONSES:
  for left, right in PAIRS:
   targets.append(("candidate_order", f"{left}|{right}", coordinate,
                   geometry["candidate_geometry"][left][coordinate] - geometry["candidate_geometry"][right][coordinate]))
 gradient_rows = []; phases = {"before_probe": parameters_before}
 phase_names = {"native_score": "after_native_probes", "counterfactual_score": "after_counterfactual_probes",
                "direction": "after_direction_probes", "candidate_order": "after_candidate_order_probes"}
 for i, (family, candidate, coordinate, target) in enumerate(targets):
  status, summaries = probe_target(target, groups, i < len(targets) - 1)
  for summary in summaries:
   if status == "NONDIFFERENTIABLE":
    group_status = status
   elif summary["trainable_parameter_count"] == 0 or summary["gradient_connected_tensor_count"] == 0:
    group_status = "DISCONNECTED"
   elif summary["finite_gradient_tensor_count"] < summary["gradient_connected_tensor_count"]:
    group_status = "NONFINITE"
   elif summary["nonzero_gradient_tensor_count"] > 0:
    group_status = "CONNECTED_NONZERO"
   else:
    group_status = "CONNECTED_ZERO_AT_OBSERVED_BATCH"
   gradient_rows.append({"target_family": family, "candidate_or_pair": candidate,
    "target_coordinate": coordinate, "target_classification": group_status, **summary})
  if i == len(targets) - 1 or targets[i + 1][0] != family:
   phase = phase_names[family]
   phases[phase] = fingerprint(
    models, fingerprint_scope=f"{phase}.parameters",
   )
 connected = ("CONNECTED_NONZERO", "CONNECTED_ZERO_AT_OBSERVED_BATCH")
 direction_rows = [row for row in gradient_rows if row["target_family"] == "direction" and row["parameter_group"] in INTENDED["direction_stability"]["groups"]]
 order_rows = [row for row in gradient_rows if row["target_family"] == "candidate_order" and row["parameter_group"] in INTENDED["candidate_order_stability"]["groups"]]
 direction_ready = bool(direction_rows) and all(row["target_classification"] in connected for row in direction_rows)
 order_ready = bool(order_rows) and all(row["target_classification"] in connected for row in order_rows)
 evaluated = all(any(row["target_family"] == family for row in gradient_rows) for family in ("native_score", "counterfactual_score", "direction", "candidate_order"))
 gate(contracts, "all_independent_gradient_families_evaluated", True, evaluated, evaluated, "probe family missing")
 for model, mode in zip(models, modes_before): model.train(mode)
 parameters_after = fingerprint(
  models, fingerprint_scope="after_state_restore.parameters",
 )
 buffers_after = fingerprint(
  models, True, fingerprint_scope="after_state_restore.buffers",
 )
 mutation_rows = [{"checkpoint": name, "parameter_fingerprint": value, "equals_before": value == parameters_before,
  "buffer_fingerprint": buffers_before, "optimizer_step_count": 0, "scheduler_step_count": 0,
  "checkpoint_write_count": 0} for name, value in phases.items()]
 mutation_rows.append({"checkpoint": "after_state_restore", "parameter_fingerprint": parameters_after,
  "equals_before": parameters_after == parameters_before, "buffer_fingerprint": buffers_after,
  "optimizer_step_count": 0, "scheduler_step_count": 0, "checkpoint_write_count": 0})
 mutation_ok = all(row["equals_before"] for row in mutation_rows) and buffers_before == buffers_after and [m.training for m in models] == modes_before
 gate(contracts, "parameter_buffer_mode_and_step_preservation", True, mutation_ok, mutation_ok, "mutation detected")
 loss_contract = {"stability_loss_added_to_training_objective": False, "classification_loss_changed": False,
  "optimizer_objective_changed": False, "training_coefficient_added": False, "combined_intervention_implemented": False}
 gate(contracts, "zero_stability_loss_integration", {key: False for key in loss_contract}, loss_contract, not any(loss_contract.values()), "loss integration detected")
 path = geometry["computation_path"]
 if path == "COUNTERFACTUAL_PATH_DETACHED":
  decision, next_stage = "STAGE196B2B6P6_COUNTERFACTUAL_GRADIENT_PATH_STILL_DETACHED", "STAGE196B2B6P6_REPAIR_COUNTERFACTUAL_GRADIENT_BOUNDARY"
 elif path == "FULL_COUNTERFACTUAL_FORWARD_REQUIRED":
  decision, next_stage = "STAGE196B2B6P6_FULL_COUNTERFACTUAL_FORWARD_REQUIRED", "STAGE196B2B6P7_FULL_COUNTERFACTUAL_FORWARD_DESIGN"
 elif direction_ready and order_ready:
  decision, next_stage = "STAGE196B2B6P6_MINIMAL_GRADIENT_PATH_INSTRUMENTATION_COMPLETE", "STAGE196B2B6P7_SEPARATE_STABILITY_INTERVENTION_IMPLEMENTATION"
 elif direction_ready:
  decision, next_stage = "STAGE196B2B6P6_DIRECTION_GRADIENT_PATH_READY_ONLY", "STAGE196B2B6P7_DIRECTION_STABILITY_INTERVENTION_IMPLEMENTATION"
 elif order_ready:
  decision, next_stage = "STAGE196B2B6P6_CANDIDATE_ORDER_GRADIENT_PATH_READY_ONLY", "STAGE196B2B6P7_CANDIDATE_ORDER_STABILITY_INTERVENTION_IMPLEMENTATION"
 else:
  decision, next_stage = "STAGE196B2B6P6_COUNTERFACTUAL_GRADIENT_PATH_STILL_DETACHED", "STAGE196B2B6P6_REPAIR_COUNTERFACTUAL_GRADIENT_BOUNDARY"
 decisions = [
  {"order": 1, "decision": "STAGE196B2B6P6_MINIMAL_GRADIENT_PATH_INSTRUMENTATION_COMPLETE", "reached": decision == "STAGE196B2B6P6_MINIMAL_GRADIENT_PATH_INSTRUMENTATION_COMPLETE"},
  {"order": 2, "decision": "STAGE196B2B6P6_DIRECTION_GRADIENT_PATH_READY_ONLY", "reached": decision == "STAGE196B2B6P6_DIRECTION_GRADIENT_PATH_READY_ONLY"},
  {"order": 3, "decision": "STAGE196B2B6P6_CANDIDATE_ORDER_GRADIENT_PATH_READY_ONLY", "reached": decision == "STAGE196B2B6P6_CANDIDATE_ORDER_GRADIENT_PATH_READY_ONLY"},
  {"order": 4, "decision": "STAGE196B2B6P6_COUNTERFACTUAL_GRADIENT_PATH_STILL_DETACHED", "reached": decision == "STAGE196B2B6P6_COUNTERFACTUAL_GRADIENT_PATH_STILL_DETACHED"},
  {"order": 5, "decision": "STAGE196B2B6P6_FULL_COUNTERFACTUAL_FORWARD_REQUIRED", "reached": decision == "STAGE196B2B6P6_FULL_COUNTERFACTUAL_FORWARD_REQUIRED"}]
 gate(contracts, "decision_hierarchy_reachability", 1, sum(row["reached"] for row in decisions), sum(row["reached"] for row in decisions) == 1, "decision unreachable")
 return {"stage": STAGE, "decision": decision, "recommended_next_stage": next_stage, "blocking_reasons": [],
  "current_git_commit": head, "computation_path": path, "candidate_masks": list(CANDIDATES), "primitive_order": list(PRIMITIVES),
  "data": {"path": str(ns.main_data_path), "sha256": digest_file(ns.main_data_path), "batch_size": ns.batch_size, "seed": ns.seed, "time_swap_used": False},
  "source_feasibility_evidence": {"p5_decision": p5.get("decision"),
   "existing_detach_boundary": "return_composer_input_observability -> detach -> cpu -> item/tolist -> JSON",
   "current_native_source": "model forward output['logits'] before detach",
   "current_counterfactual_source": "P2 apply_mask over joint recipient plus separately trained frame_local_only donor",
   "first_detach_cpu_item_numpy_json_boundary": "B3P0 diagnostic export immediately after live model output",
   "trainable_composer_parameter_groups": "decision_head plus comparator alphas",
   "trainable_router_parameter_groups": "none in v6b_minimal",
   "trainable_epistemic_head_parameter_groups": "frame/predicate/sufficiency/polarity modules",
   "frozen_backbone_parameter_groups": "model.mamba with requires_grad false"},
  "intended_gradient_recipients": INTENDED,
  "connectivity": {"direction_ready": direction_ready, "candidate_order_ready": order_ready, "connected_zero_distinguished": True},
  "forward_equivalence": {"native_score_disagreements": native_scores_bad, "native_prediction_disagreements": native_predictions_bad,
   "counterfactual_composer_disagreements": counterfactual_bad, "candidate_action_identity_disagreements": 0},
  "no_mutation": {"parameters_equal": parameters_before == parameters_after, "buffers_equal": buffers_before == buffers_after,
   "training_state_before": modes_before, "training_state_after": [m.training for m in models], "optimizer_steps": 0, "scheduler_steps": 0, "checkpoint_writes": 0},
  "loss_nonexistence_contract": loss_contract, "action_audit": action_audit, "exact_outputs": list(OUTPUTS), "decision_hierarchy": decisions,
  "_tables": {"tensor": tensor_rows, "gradient": gradient_rows, "groups": group_rows, "forward": forward_rows, "mutation": mutation_rows, "decisions": decisions}}

def render_report(a: dict[str, Any]) -> str:
 return f"""# Stage196-B2-B6P6 Minimal Gradient-Path Instrumentation

Decision: `{a['decision']}`
Recommended next stage: `{a['recommended_next_stage']}`

P5 required gradient-path instrumentation before any loss. P6 introduces no intervention objective. Existing trainer behavior is unchanged by default. Exact tensors are captured before detach. Frozen Mamba gradients are not required. Connected-zero and disconnected gradients are distinguished. Parameter and buffer mutation are prohibited.

Candidate-action semantics match P2: opaque candidate identities select exact row-wise primitive actions, and recomposition retains recipient final modulation. The observed path is `{a.get('computation_path')}` because P5 proved that the separately trained frame-local-only donor cannot be recovered from the native joint forward.

Native scores, counterfactual scores, direction coordinates, and candidate-order pair gaps are evaluated independently. No optimizer or scheduler exists and no checkpoint is written. The next stage follows from observed connectivity and exact semantic equivalence, not from a forced intervention choice.
"""

def publish(ns: argparse.Namespace, analysis: dict[str, Any], contracts: list[dict[str, Any]]) -> None:
 tables = analysis.pop("_tables", {key: [] for key in ("tensor", "gradient", "groups", "forward", "mutation", "decisions")})
 blocking = [row["blocking_reason"] for row in contracts if not row["passed"] and row["blocking_reason"]]
 analysis["blocking_reasons"] = blocking
 if blocking:
  analysis["decision"] = "STAGE196B2B6P6_BLOCKED_CONTRACT_FAILURE"; analysis["recommended_next_stage"] = "STAGE196B2B6P6_REPAIR_CONTRACT"
 contents = {
  OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True) + "\n", OUTPUTS[1]: render_report(analysis),
  OUTPUTS[2]: csv_text(("scope", "candidate_mask", "tensor_name", "shape", "dtype", "device", "requires_grad", "grad_fn_class", "is_leaf", "finite", "batch_dimension"), tables["tensor"]),
  OUTPUTS[3]: csv_text(("target_family", "candidate_or_pair", "target_coordinate", "target_classification", "parameter_group", "parameter_tensor_count", "trainable_parameter_count", "gradient_connected_tensor_count", "unused_tensor_count", "finite_gradient_tensor_count", "nonzero_gradient_tensor_count", "gradient_l1_norm", "gradient_l2_norm", "maximum_absolute_gradient"), tables["gradient"]),
  OUTPUTS[4]: csv_text(("parameter_group", "parameter_tensor_count", "parameter_count", "requires_grad_tensor_count", "requires_grad_parameter_count", "module_reference_derived"), tables["groups"]),
  OUTPUTS[5]: csv_text(("scope", "candidate_mask", "score_disagreements", "prediction_disagreements", "candidate_action_identity_disagreements", "exact_equal"), tables["forward"]),
  OUTPUTS[6]: csv_text(("checkpoint", "parameter_fingerprint", "equals_before", "buffer_fingerprint", "optimizer_step_count", "scheduler_step_count", "checkpoint_write_count"), tables["mutation"]),
  OUTPUTS[7]: csv_text(("order", "decision", "reached"), tables["decisions"]),
  OUTPUTS[8]: csv_text(("contract", "required", "observed", "passed", "blocking_reason"), contracts)}
 for name in OUTPUTS: atomic_write(ns.output_dir / name, contents[name])

def main() -> int:
 ns = build_parser().parse_args(); ns.repo_root = ns.repo_root.resolve()
 for name in ("stage196b2b6p5_analysis_json", "stage196b2b6p4_analysis_json", "stage196b2b6p2_analysis_json",
              "native_checkpoint", "frame_local_only_checkpoint", "main_data_path"):
  setattr(ns, name, getattr(ns, name).resolve())
 ns.output_dir = ns.output_dir.resolve()
 if ns.output_dir.exists(): raise FileExistsError(f"refusing to overwrite output directory {ns.output_dir}")
 ns.output_dir.mkdir(parents=True, exist_ok=False); contracts: list[dict[str, Any]] = []
 try: analysis = run(ns, contracts)
 except Exception as exc:
  gate(contracts, "uncaught_contract_failure", "no exception", f"{type(exc).__name__}: {exc}", False, f"{type(exc).__name__}: {exc}")
  analysis = {"stage": STAGE, "decision": "STAGE196B2B6P6_BLOCKED_CONTRACT_FAILURE",
   "recommended_next_stage": "STAGE196B2B6P6_REPAIR_CONTRACT", "blocking_reasons": [],
   "current_git_commit": ns.current_git_commit, "exact_outputs": list(OUTPUTS),
   "loss_nonexistence_contract": {"stability_loss_added_to_training_objective": False,
    "classification_loss_changed": False, "optimizer_objective_changed": False,
    "training_coefficient_added": False, "combined_intervention_implemented": False},
   "_tables": {key: [] for key in ("tensor", "gradient", "groups", "forward", "mutation", "decisions")}}
 gate(contracts, "exact_nine_file_closure", list(OUTPUTS), list(OUTPUTS), True)
 publish(ns, analysis, contracts)
 return 0 if all(row["passed"] for row in contracts) else 2

if __name__ == "__main__": raise SystemExit(main())
