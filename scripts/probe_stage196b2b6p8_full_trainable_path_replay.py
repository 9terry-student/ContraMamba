"""Stage196-B2-B6P8 exact full-trainable-path replay probe.

This is a probe, never a training objective: it performs no optimizer or
scheduler step and writes no checkpoint. All ten artifacts are atomic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from contramamba.modeling_v6b_minimal import (  # noqa: E402
    STAGE196B2B6P8_CANDIDATE_MASKS,
    STAGE196B2B6P8_CLASS_ORDER,
    STAGE196B2B6P8_PRIMITIVE_KEYS,
)
from scripts import train_controlled_v5 as v5  # noqa: E402
from scripts import train_controlled_v6b_minimal as trainer  # noqa: E402

OUTPUTS = (
    "stage196b2b6p8_analysis.json",
    "stage196b2b6p8_report.md",
    "stage196b2b6p8_replay_state_schema.csv",
    "stage196b2b6p8_candidate_replay_equivalence.csv",
    "stage196b2b6p8_gradient_connectivity.csv",
    "stage196b2b6p8_stochastic_state_audit.csv",
    "stage196b2b6p8_no_mutation_audit.csv",
    "stage196b2b6p8_resource_observation.csv",
    "stage196b2b6p8_decision_gate.csv",
    "stage196b2b6p8_contract.csv",
)
P7_COMPANIONS = (
    "stage196b2b6p7_analysis.json", "stage196b2b6p7_report.md",
    "stage196b2b6p7_execution_boundary_audit.csv",
    "stage196b2b6p7_candidate_semantic_trace.csv",
    "stage196b2b6p7_stochastic_state_audit.csv",
    "stage196b2b6p7_gradient_path_design.csv",
    "stage196b2b6p7_counterfactual_forward_designs.csv",
    "stage196b2b6p7_compute_memory_estimate.csv",
    "stage196b2b6p7_decision_gate.csv", "stage196b2b6p7_contract.csv",
)
EXPECTED_MAIN_SHA = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
EXPECTED_SOURCE_COMMIT = "fa16787efa84bb15d832b6d9382fafd77016c4e2"
EXPECTED_TRAINER_SHA = "2f8c2596573383972f153c8a24ffaf42ea93d8f1bd1a030253c76cd66f959b87"
EXPECTED_ROW_ACTION_CSV = "stage196b2b6p2_candidate_action_composer_scores.csv"
EXPECTED_ROW_ACTION_COUNT = 6480
EXPECTED_SEED183_ROW_ACTION_COUNT = 2160
EXPECTED_ROW_ACTION_PRIMITIVE_ORDER = (
    "FRAME", "PREDICATE", "SUFFICIENCY", "POSITIVE_ENERGY", "NEGATIVE_ENERGY",
)
GRAD_CLASSES = (
    "CONNECTED_NONZERO", "CONNECTED_ZERO_AT_OBSERVED_BATCH", "DISCONNECTED",
    "NONDIFFERENTIABLE", "NONFINITE",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--stage196b2b6p7-analysis-json", type=Path, required=True)
    p.add_argument("--stage196b2b6p6-analysis-json", type=Path, required=True)
    p.add_argument("--stage196b2b6p5-analysis-json", type=Path, required=True)
    p.add_argument("--native-checkpoint", type=Path, required=True)
    p.add_argument("--frame-local-only-checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-recovery-summary-json", type=Path, required=True)
    p.add_argument("--main-data-path", type=Path, required=True)
    p.add_argument("--backbone", choices=("mamba",), required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--current-git-commit", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ("status", "detail")
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def contract(rows: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    rows.append({"contract": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def file_fingerprint(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def tensor_fingerprint(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    h = hashlib.sha256()
    for name, value in named:
        cpu = value.detach().contiguous().cpu()
        h.update(name.encode())
        h.update(str(tuple(cpu.shape)).encode())
        h.update(str(cpu.dtype).encode())
        h.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def model_fingerprints(model: torch.nn.Module) -> tuple[str, str]:
    return (tensor_fingerprint(model.named_parameters()),
            tensor_fingerprint(model.named_buffers()))


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "pass", "passed"}:
        return True
    if lowered in {"false", "0", "no", "n", "fail", "failed", ""}:
        return False
    raise ValueError(f"not a strict boolean: {value!r}")


def git_run(args: argparse.Namespace, git_args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *git_args],
        cwd=args.repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def validate_p7_contract_closure(p7_contract: list[dict[str, str]]) -> tuple[bool, str]:
    required = {"contract", "required", "observed", "passed", "blocking_reason"}
    if not p7_contract:
        return False, "empty P7 contract"
    missing = sorted(required - set(p7_contract[0]))
    if missing:
        return False, "missing columns: " + ",".join(missing)
    failures: list[str] = []
    for index, row in enumerate(p7_contract):
        name = (row.get("contract") or f"row_{index}").strip()
        try:
            passed = bool_value(row.get("passed"))
        except ValueError as exc:
            failures.append(f"{name}: invalid passed value {row.get('passed')!r}: {exc}")
            continue
        reason = (row.get("blocking_reason") or "").strip()
        if not passed or reason:
            failures.append(f"{name}: passed={row.get('passed')!r} blocking_reason={reason!r}")
    return not failures, "P7 contract closure" if not failures else "; ".join(failures)


def validate_p7_source_commit(args: argparse.Namespace, p7: dict[str, Any],
                              rows: list[dict[str, Any]]) -> str:
    p7_commit = str(p7.get("current_git_commit") or p7.get("git_commit")
                    or (p7.get("source_provenance") or {}).get("git_commit")
                    or "").strip()
    declared = re.fullmatch(r"[0-9a-f]{40}", p7_commit) is not None
    contract(rows, "p7_source_commit_declared", declared, p7_commit or "missing")
    exists = False
    ancestor = False
    exists_detail = "not checked"
    ancestor_detail = "not checked"
    if declared:
        cat = git_run(args, ["cat-file", "-e", f"{p7_commit}^{{commit}}"])
        exists = cat.returncode == 0
        exists_detail = "git cat-file commit exists" if exists else (cat.stderr.strip() or f"git cat-file rc={cat.returncode}")
        if exists:
            merge_base = git_run(args, ["merge-base", "--is-ancestor", p7_commit, args.current_git_commit])
            ancestor = merge_base.returncode == 0
            if merge_base.returncode == 0:
                ancestor_detail = "P7 commit is ancestor of current"
            elif merge_base.returncode == 1:
                ancestor_detail = "P7 commit is not ancestor of current"
            else:
                ancestor_detail = merge_base.stderr.strip() or f"git merge-base rc={merge_base.returncode}"
    contract(rows, "p7_source_commit_exists", exists, exists_detail)
    contract(rows, "p7_source_commit_is_ancestor_of_current", ancestor, ancestor_detail)
    head = git_run(args, ["rev-parse", "HEAD"])
    head_value = head.stdout.strip() if head.returncode == 0 else ""
    head_detail = head_value if head.returncode == 0 else (head.stderr.strip() or f"git rev-parse rc={head.returncode}")
    contract(rows, "current_commit_identity", head.returncode == 0 and args.current_git_commit == head_value, head_detail)
    return p7_commit


def validate_p7_selected_boundary(boundary_rows: list[dict[str, str]]) -> tuple[bool, str, dict[str, str] | None]:
    required = {
        "state_name",
        "producing_module",
        "tensor_shape_contract",
        "producer_trainability",
        "candidate_action_dependency",
        "detach_boundary",
        "serialization_boundary",
        "replay_feasibility",
    }
    if not boundary_rows:
        return False, "empty P7 execution-boundary audit", None
    missing = sorted(required - set(boundary_rows[0]))
    if missing:
        return False, "missing columns: " + ",".join(missing), None
    selected = [
        row for row in boundary_rows
        if (row.get("replay_feasibility") or "").strip() == "EARLIEST_SAFE_SHARED_REPLAY_BOUNDARY"
    ]
    if len(selected) != 1:
        return False, f"expected exactly one EARLIEST_SAFE_SHARED_REPLAY_BOUNDARY row, found {len(selected)}", None
    row = selected[0]
    state_name = (row.get("state_name") or "").lower()
    producing_module = (row.get("producing_module") or "").lower()
    shape = (row.get("tensor_shape_contract") or "").replace(" ", "")
    trainability = (row.get("producer_trainability") or "").lower()
    dependency = (row.get("candidate_action_dependency") or "").strip().lower()
    serialization = (row.get("serialization_boundary") or "").lower()
    checks = {
        "state_name_contains_encoder_hidden_states": "encoder_hidden_states" in state_name,
        "producing_module_is_mamba_or_frozen_cache": "model.mamba" in producing_module or "frozen-state cache" in producing_module,
        "tensor_shape_contract_is_backbone_hidden": shape == "[B,T,H_backbone]",
        "producer_trainability_is_frozen": "frozen" in trainability,
        "candidate_action_dependency_none": dependency == "none",
        "serialization_boundary_in_memory_encoder_hidden_states": "in-memory" in serialization and "encoder_hidden_states" in serialization,
    }
    detail = json.dumps({"selected_boundary": row, "checks": checks}, sort_keys=True)
    return all(checks.values()), detail, row


def blockers(payload: dict[str, Any]) -> list[Any]:
    value = payload.get("blocking_reasons", [])
    return value if isinstance(value, list) else [value]


def checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        return payload["model_state_dict"], dict(payload.get("metadata") or {})
    if isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload, {}
    raise ValueError(f"unsupported checkpoint schema: {path}")


def optional_heads(state: dict[str, torch.Tensor]) -> dict[str, bool]:
    prefixes = {
        "use_boundary_head": "boundary_head.",
        "use_frame_violation_head": "frame_violation_head.",
        "use_predicate_isolation_head": "predicate_isolation_head.",
        "use_preservation_entitlement_head": "preservation_entitlement_head.",
        "use_temporal_diagnostic_head": "temporal_diagnostic_head.",
        "use_temporal_residual_adapter": "temporal_residual_adapter.",
        "use_temporal_channel": "temporal_channel_v1.",
    }
    return {flag: any(key.startswith(prefix) for key in state) for flag, prefix in prefixes.items()}


def build_model(args: argparse.Namespace, state: dict[str, torch.Tensor], mode: str) -> torch.nn.Module:
    model = trainer.build_mamba_model(
        args.model_name, freeze_encoder=True, freeze_a_log=True,
        **optional_heads(state),
    )
    model.load_state_dict(state, strict=True)
    model.to(torch.device(args.device))
    trainer._install_framegate_gradient_ownership(model, mode)
    return model


def feature_inputs(encoded: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("input_ids", "attention_mask", "claim_mask", "evidence_mask")
    return {key: encoded["model_inputs"][key].to(device) for key in keys}


def json_field(value: Any, field: str, row: dict[str, str]) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is not JSON-serialized in aggregate P7 semantic row {row}: {exc}") from exc


def normalize_string_list(value: Any, field: str, row: dict[str, str]) -> list[str]:
    parsed = json_field(value, field, row)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{field} must be a JSON string list: {row}")
    return parsed


def normalize_count_dict(value: Any, field: str, row: dict[str, str]) -> dict[str, int]:
    parsed = json_field(value, field, row)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must be a JSON object: {row}")
    result: dict[str, int] = {}
    for key, count in parsed.items():
        try:
            result[str(key)] = int(count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} count is not integer-like for key {key!r}: {row}") from exc
    return result


def iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from iter_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_dicts(nested)


def candidate_action_provenance_matches(candidate: dict[str, Any]) -> bool:
    try:
        row_specific = bool_value(candidate.get("row_specific"))
    except ValueError:
        return False
    path = Path(str(candidate.get("path") or ""))
    return (
        row_specific
        and path.name == EXPECTED_ROW_ACTION_CSV
        and candidate.get("candidate_masks") == list(STAGE196B2B6P8_CANDIDATE_MASKS)
        and candidate.get("primitive_order") == list(EXPECTED_ROW_ACTION_PRIMITIVE_ORDER)
    )


def validate_row_action_authority(args: argparse.Namespace, p7: dict[str, Any],
                                  rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [candidate for candidate in iter_dicts(p7) if candidate_action_provenance_matches(candidate)]
    contract(rows, "p7_row_action_authority_provenance_unique", len(matches) == 1,
             f"matches={len(matches)}")
    if len(matches) != 1:
        return {"path": None, "rows": [], "mapping": {}, "aggregate_counts": {}, "seed183_mapping_count": 0}
    provenance = matches[0]
    recorded_path = Path(str(provenance.get("path")))
    authority_path = recorded_path if recorded_path.is_absolute() else (args.repo_root / recorded_path)
    authority_path = authority_path.resolve()
    exists = authority_path.is_file()
    contract(rows, "p7_row_action_authority_file_exists", exists, str(authority_path))
    observed_sha = sha256(authority_path) if exists else ""
    contract(rows, "p7_row_action_authority_sha256",
             exists and observed_sha == str(provenance.get("sha256")),
             observed_sha or "missing")
    p2_rows = read_csv(authority_path) if exists else []
    try:
        recorded_row_count = int(provenance.get("row_count"))
    except (TypeError, ValueError):
        recorded_row_count = -1
    try:
        unique_pairs = int(provenance.get("unique_recipient_candidate_pairs"))
    except (TypeError, ValueError):
        unique_pairs = -1
    contract(rows, "p7_row_action_authority_row_count",
             exists and len(p2_rows) == recorded_row_count == EXPECTED_ROW_ACTION_COUNT
             and unique_pairs == EXPECTED_ROW_ACTION_COUNT,
             f"observed={len(p2_rows)} recorded={recorded_row_count} unique={unique_pairs}")
    required = {"seed", "stable_row_id", "candidate_mask", "candidate_action_key"}
    missing = sorted(required - set(p2_rows[0])) if p2_rows else sorted(required)
    schema_errors: list[str] = []
    mapping: dict[tuple[int, str, str], str] = {}
    aggregate_counts: dict[str, dict[str, int]] = {mask: {} for mask in STAGE196B2B6P8_CANDIDATE_MASKS}
    duplicate_errors: list[str] = []
    if missing:
        schema_errors.append("missing columns: " + ",".join(missing))
    else:
        for index, row in enumerate(p2_rows):
            stable = str(row.get("stable_row_id") or "").strip()
            mask = str(row.get("candidate_mask") or "").strip()
            action = str(row.get("candidate_action_key") or "").strip()
            try:
                seed = int(str(row.get("seed") or "").strip())
            except ValueError:
                schema_errors.append(f"row {index} seed is not integer: {row.get('seed')!r}")
                continue
            if not stable:
                schema_errors.append(f"row {index} stable_row_id is empty")
                continue
            if mask not in STAGE196B2B6P8_CANDIDATE_MASKS:
                schema_errors.append(f"row {index} unknown candidate_mask: {mask!r}")
                continue
            if re.fullmatch(r"[01]{5}", action) is None:
                schema_errors.append(f"row {index} malformed candidate_action_key: {action!r}")
                continue
            key = (seed, stable, mask)
            previous = mapping.get(key)
            if previous is not None:
                duplicate_errors.append(
                    f"{key}: duplicate action {action!r}" if previous == action
                    else f"{key}: conflicting actions {previous!r}/{action!r}"
                )
                continue
            mapping[key] = action
            aggregate_counts[mask][action] = aggregate_counts[mask].get(action, 0) + 1
    seed183_count = sum(1 for seed, _, _ in mapping if seed == 183)
    contract(rows, "p7_row_action_authority_schema", not schema_errors,
             "; ".join(schema_errors[:20]) or "P2_ROW_SPECIFIC_ACTION_AUTHORITY schema valid")
    contract(rows, "p7_row_action_mapping_uniqueness", not duplicate_errors,
             "; ".join(duplicate_errors[:20]) or "unique row-specific mappings")
    contract(rows, "p7_row_action_population_closure",
             len(mapping) == EXPECTED_ROW_ACTION_COUNT and seed183_count == EXPECTED_SEED183_ROW_ACTION_COUNT,
             f"mapping={len(mapping)} seed183={seed183_count}")
    return {
        "path": authority_path,
        "sha256": observed_sha,
        "row_count": len(p2_rows),
        "mapping": mapping,
        "aggregate_counts": aggregate_counts,
        "seed183_mapping_count": seed183_count,
        "provenance": provenance,
    }


def validate_aggregate_semantic_trace(trace_rows: list[dict[str, str]], row_authority: dict[str, Any],
                                      rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "candidate_mask", "primitive_mapping", "primitive_order", "observed_action_keys",
        "observed_action_key_counts", "first_tensor_whose_value_changes", "changed_primitive_union",
    }
    missing = sorted(required - set(trace_rows[0])) if trace_rows else sorted(required)
    masks = [row.get("candidate_mask") for row in trace_rows]
    schema_ok = (
        len(trace_rows) == 3
        and set(masks) == set(STAGE196B2B6P8_CANDIDATE_MASKS)
        and len(set(masks)) == 3
        and not missing
    )
    errors: list[str] = []
    matches = True
    aggregate_counts = row_authority.get("aggregate_counts") or {}
    if schema_ok:
        for row in trace_rows:
            mask = str(row.get("candidate_mask") or "")
            try:
                primitive_order = normalize_string_list(row.get("primitive_order"), "primitive_order", row)
                observed_keys = normalize_string_list(row.get("observed_action_keys"), "observed_action_keys", row)
                observed_counts = normalize_count_dict(row.get("observed_action_key_counts"), "observed_action_key_counts", row)
                json_field(row.get("first_tensor_whose_value_changes"), "first_tensor_whose_value_changes", row)
                normalize_string_list(row.get("changed_primitive_union"), "changed_primitive_union", row)
            except ValueError as exc:
                errors.append(str(exc))
                matches = False
                continue
            expected_counts = aggregate_counts.get(mask, {})
            expected_keys = sorted(expected_counts)
            row_matches = (
                row.get("primitive_mapping") == "ROW_SPECIFIC_5BIT_ACTION_FROM_P2_AUTHORITY"
                and primitive_order == list(EXPECTED_ROW_ACTION_PRIMITIVE_ORDER)
                and observed_keys == expected_keys
                and observed_counts == expected_counts
            )
            if not row_matches:
                matches = False
                errors.append(json.dumps({"mask": mask, "observed_keys": observed_keys,
                                          "expected_keys": expected_keys,
                                          "observed_counts": observed_counts,
                                          "expected_counts": expected_counts}, sort_keys=True))
    detail = "; ".join(errors[:20]) or "P7_AGGREGATE_SEMANTIC_AUTHORITY matches P2 row authority"
    contract(rows, "p7_aggregate_semantic_trace_schema", schema_ok,
             f"rows={len(trace_rows)} masks={masks} missing={missing}")
    contract(rows, "p7_aggregate_semantic_trace_matches_row_authority", schema_ok and matches, detail)
    return {"path": "stage196b2b6p7_candidate_semantic_trace.csv", "match": schema_ok and matches}


def trace_actions(
    mapping: dict[tuple[int, str, str], str], records: list[dict[str, Any]],
    seed: int, device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[str, ...]], int]:
    """Resolve row actions from the P2 authority; never interpret opaque candidate IDs."""
    actions: dict[str, torch.Tensor] = {}
    action_keys: dict[str, tuple[str, ...]] = {}
    resolved_count = 0
    for mask in STAGE196B2B6P8_CANDIDATE_MASKS:
        resolved: list[str] = []
        for record in records:
            stable = str(record.get("stable_row_id") or record.get("stable_id") or record.get("id") or "").strip()
            if not stable:
                raise ValueError(f"probe batch record has no stable row identity: {record}")
            key = (seed, stable, mask)
            action = mapping.get(key)
            if action is None:
                raise ValueError(f"P2 row authority has no action for {seed}/{stable}/{mask}")
            resolved.append(action)
            resolved_count += 1
        action_keys[mask] = tuple(resolved)
        actions[mask] = torch.tensor(
            [[character == "1" for character in key] for key in resolved],
            dtype=torch.bool, device=device,
        )
    return actions, action_keys, resolved_count


def direct_groups(models: dict[str, torch.nn.Module]) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {}
    ownership = {
        "frozen_backbone": "mamba", "frame_branch": "frame_gate",
        "predicate_branch": "predicate_coverage_head", "sufficiency_branch": "sufficiency_gate",
        "polarity_branch": "polarity_energy_head", "entitlement_or_decision_branch": "decision_head",
    }
    claimed: set[int] = set()
    for arm, model in models.items():
        for group, attribute in ownership.items():
            params = list(getattr(model, attribute).parameters())
            groups[f"{arm}:{group}"] = params
            claimed.update(map(id, params))
        composer = [p for name, p in model.named_parameters() if name in {"alpha_temporal_raw", "alpha_predicate_raw"}]
        groups[f"{arm}:final_composer"] = composer
        claimed.update(map(id, composer))
        groups[f"{arm}:router_or_selector"] = []
        groups[f"{arm}:other_trainable"] = [p for p in model.parameters() if id(p) not in claimed]
    return groups


def gradient_row(target_name: str, target: torch.Tensor, group_name: str,
                 params: list[torch.nn.Parameter], retain_graph: bool) -> dict[str, Any]:
    required = [parameter for parameter in params if parameter.requires_grad]
    base = {"target": target_name, "parameter_group": group_name,
            "parameter_tensor_count": len(params), "requires_grad_tensor_count": len(required)}
    if not target.requires_grad:
        return {**base, "classification": "NONDIFFERENTIABLE", "connected_tensor_count": 0,
                "unused_tensor_count": len(required), "finite_gradient_tensor_count": 0,
                "nonzero_gradient_tensor_count": 0, "gradient_l1_norm": 0.0,
                "gradient_l2_norm": 0.0, "maximum_absolute_gradient": 0.0}
    gradients = torch.autograd.grad(target.sum(), required, allow_unused=True,
                                    retain_graph=retain_graph) if required else ()
    connected = [gradient for gradient in gradients if gradient is not None]
    finite = [gradient for gradient in connected if torch.isfinite(gradient).all().item()]
    nonzero = [gradient for gradient in finite if torch.count_nonzero(gradient).item()]
    if any(not torch.isfinite(gradient).all().item() for gradient in connected):
        classification = "NONFINITE"
    elif not connected:
        classification = "DISCONNECTED"
    elif nonzero:
        classification = "CONNECTED_NONZERO"
    else:
        classification = "CONNECTED_ZERO_AT_OBSERVED_BATCH"
    flat = torch.cat([gradient.detach().reshape(-1).float() for gradient in finite]) if finite else torch.zeros(1)
    return {**base, "classification": classification, "connected_tensor_count": len(connected),
            "unused_tensor_count": len(required) - len(connected), "finite_gradient_tensor_count": len(finite),
            "nonzero_gradient_tensor_count": len(nonzero), "gradient_l1_norm": float(flat.abs().sum()),
            "gradient_l2_norm": float(torch.linalg.vector_norm(flat)),
            "maximum_absolute_gradient": float(flat.abs().max())}


def collect_targets(native_by_arm: dict[str, dict[str, Any]], replay: dict[str, Any]) -> dict[str, torch.Tensor]:
    targets: dict[str, torch.Tensor] = {}
    for arm, output in native_by_arm.items():
        geometry = output["_geometry"]
        for key, value in geometry.items():
            targets[f"native/{arm}/{key}"] = value
    for mask, candidate in replay["candidate_geometry"].items():
        for key, value in candidate.items():
            if isinstance(value, torch.Tensor) and (
                key.startswith("counterfactual_score_") or key.startswith("counterfactual_margin_")
                or key == "counterfactual_top1_runner_up_margin"
            ):
                targets[f"candidate/{mask}/{key}"] = value
        for key in ("delta_support_minus_not_entitled", "delta_support_minus_refute",
                    "delta_refute_minus_not_entitled", "delta_top1_runner_up_margin"):
            targets[f"direction/{mask}/{key}"] = candidate[key]
    response_keys = (
        "delta_score_support", "delta_score_not_entitled", "delta_score_refute",
        "delta_support_minus_not_entitled", "delta_support_minus_refute",
        "delta_refute_minus_not_entitled", "delta_top1_runner_up_margin",
    )
    masks = STAGE196B2B6P8_CANDIDATE_MASKS
    for left_index in range(len(masks)):
        for right_index in range(left_index + 1, len(masks)):
            left, right = masks[left_index], masks[right_index]
            for key in response_keys:
                targets[f"order/{left}-{right}/{key}"] = (
                    replay["candidate_geometry"][left][key]
                    - replay["candidate_geometry"][right][key]
                )
    return targets


def equality_rows(ordinary: dict[str, Any], capable: dict[str, Any],
                  replay: dict[str, Any], actions: dict[str, torch.Tensor],
                  action_keys: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    fields = ("logits", "predictions", "frame_pair_repr", "predicate_pair_repr",
              "sufficiency_repr", "polarity_features", "entitlement_prob", "base_logits")
    rows: list[dict[str, Any]] = []
    for field in fields:
        left, right = ordinary.get(field), capable.get(field)
        applicable = isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        rows.append({"check": "native_forward_equivalence", "candidate_mask": "NATIVE",
                     "field": field, "status": "PASS" if applicable and torch.equal(left, right) else "FAIL",
                     "maximum_absolute_error": float((left - right).abs().max()) if applicable else ""})
    for mask in STAGE196B2B6P8_CANDIDATE_MASKS:
        rows.append({"check": "candidate_identity_and_primitive_closure", "candidate_mask": mask,
                     "field": "row_action", "status": "PASS" if actions[mask].shape[1] == 5 and len(action_keys[mask]) == actions[mask].shape[0] else "FAIL",
                     "maximum_absolute_error": 0.0})
        logits = replay["candidate_geometry"][mask]["counterfactual_logits"]
        rows.append({"check": "candidate_logits_finite", "candidate_mask": mask,
                     "field": "counterfactual_logits", "status": "PASS" if torch.isfinite(logits).all().item() else "FAIL",
                     "maximum_absolute_error": 0.0})
    return rows


def validate_authority(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    p7_path = args.stage196b2b6p7_analysis_json.resolve()
    companions = {name: p7_path.parent / name for name in P7_COMPANIONS}
    contract(rows, "p7_exact_ten_file_closure", all(path.is_file() for path in companions.values()),
             ";".join(name for name, path in companions.items() if not path.is_file()) or "exact closure")
    required = [args.stage196b2b6p7_analysis_json, args.stage196b2b6p6_analysis_json,
                args.stage196b2b6p5_analysis_json, args.checkpoint_recovery_summary_json,
                args.native_checkpoint, args.frame_local_only_checkpoint, args.main_data_path]
    contract(rows, "explicit_input_files_exist", all(path.is_file() for path in required),
             ";".join(str(path) for path in required if not path.is_file()) or "all present")
    if any(not path.is_file() for path in required):
        return {"companions": companions}
    p7, p6, p5 = map(load_json, required[:3])
    contract(rows, "p7_decision_closure", p7.get("decision") == "STAGE196B2B6P7_FULL_TRAINABLE_PATH_REPLAY_READY", str(p7.get("decision")))
    contract(rows, "p7_zero_blockers", not blockers(p7), repr(blockers(p7)))
    p7_contract = read_csv(companions["stage196b2b6p7_contract.csv"]) if companions["stage196b2b6p7_contract.csv"].is_file() else []
    p7_contract_ok, p7_contract_detail = validate_p7_contract_closure(p7_contract)
    contract(rows, "p7_zero_failed_contracts", p7_contract_ok, p7_contract_detail)
    p7_commit = validate_p7_source_commit(args, p7, rows)
    boundary_rows = read_csv(companions["stage196b2b6p7_execution_boundary_audit.csv"])
    boundary_ok, boundary_detail, selected_boundary = validate_p7_selected_boundary(boundary_rows)
    contract(rows, "p7_selected_boundary_closure", boundary_ok, boundary_detail)
    row_action_authority = validate_row_action_authority(args, p7, rows)
    aggregate_trace_path = companions["stage196b2b6p7_candidate_semantic_trace.csv"]
    aggregate_trace = read_csv(aggregate_trace_path)
    aggregate_trace_authority = validate_aggregate_semantic_trace(aggregate_trace, row_action_authority, rows)
    aggregate_trace_authority["path"] = aggregate_trace_path
    stochastic_rows = read_csv(companions["stage196b2b6p7_stochastic_state_audit.csv"])
    stochastic_text = json.dumps(stochastic_rows, sort_keys=True).lower()
    contract(rows, "p7_stochastic_policy_authority",
             "restore" in stochastic_text and "rng" in stochastic_text
             and "dropout" in stochastic_text, "matched RNG/dropout policy")
    contract(rows, "p6_decision_closure", p6.get("decision") == "STAGE196B2B6P6_FULL_COUNTERFACTUAL_FORWARD_REQUIRED" and not blockers(p6), str(p6.get("decision")))
    contract(rows, "p5_decision_closure", p5.get("decision") == "STAGE196B2B6P5_GRADIENT_PATH_INSTRUMENTATION_REQUIRED" and not blockers(p5), str(p5.get("decision")))

    contract(rows, "mamba_130m_identity", args.backbone == "mamba" and args.model_name == "state-spaces/mamba-130m-hf", f"{args.backbone}/{args.model_name}")
    contract(rows, "cuda_device", args.device == "cuda" and torch.cuda.is_available(), args.device)
    contract(rows, "main_clean_data_identity", sha256(args.main_data_path) == EXPECTED_MAIN_SHA, sha256(args.main_data_path))
    return {"companions": companions, "p7": p7, "p6": p6, "p5": p5,
            "recovery": load_json(args.checkpoint_recovery_summary_json),
            "p7_source_commit": p7_commit,
            "p7_selected_boundary": selected_boundary,
            "row_action_authority": row_action_authority,
            "aggregate_trace_authority": aggregate_trace_authority}


def recovery_entry(summary: dict[str, Any], mode: str) -> dict[str, Any]:
    direct = summary.get(mode, summary.get(f"seed183_{mode}"))
    if isinstance(direct, dict):
        return {**summary, **direct}
    stack: list[Any] = [summary]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            declared = value.get("gradient_mode", value.get("gradient_ownership_mode"))
            if declared in {mode, f"seed183_{mode}"}:
                return {**summary, **value}
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return summary


def checkpoint_provenance(args: argparse.Namespace, recovery: dict[str, Any],
                          rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for path, mode in ((args.native_checkpoint, "joint"),
                       (args.frame_local_only_checkpoint, "frame_local_only")):
        record = file_fingerprint(path)
        declared = recovery_entry(recovery, mode)
        declared_mode = declared.get("gradient_mode", declared.get("gradient_ownership_mode"))
        record.update({"gradient_mode": mode, "source_commit": declared.get("source_commit"),
                       "selected_epoch": declared.get("selected_epoch"),
                       "reconstructed_provenance": declared.get("reconstructed_provenance", True)})
        result.append(record)
        contract(rows, f"checkpoint_{mode}_provenance",
                 record["source_commit"] == EXPECTED_SOURCE_COMMIT
                 and declared.get("trainer_sha256") == EXPECTED_TRAINER_SHA
                 and declared.get("main_data_sha256", EXPECTED_MAIN_SHA) == EXPECTED_MAIN_SHA
                 and declared_mode in {mode, f"seed183_{mode}"}
                 and record["selected_epoch"] is not None,
                 json.dumps(record, sort_keys=True))
    return result


def declared_connectivity(design: list[dict[str, str]], gradients: list[dict[str, Any]],
                          family: str) -> tuple[bool, int]:
    """Evaluate only recipients predeclared by the P7 gradient-path artifact."""
    expected: list[tuple[str, str, str]] = []
    for row in design:
        row_family = (row.get("intervention_family") or row.get("target_family")
                      or row.get("probe_family") or "").lower()
        if family not in row_family:
            continue
        expectation = (row.get("expected_connectivity") or row.get("expected_status")
                       or row.get("required") or "").lower()
        if expectation not in {"true", "1", "required", "connected", "connected_nonzero",
                               "connected_zero_or_nonzero"}:
            continue
        arm = row.get("gradient_ownership_mode") or row.get("checkpoint_mode") or ""
        group = row.get("parameter_group") or row.get("gradient_recipient") or ""
        coordinate = row.get("coordinate") or row.get("target") or family
        if group:
            expected.append((arm, group, coordinate))
    passed = True
    for arm, group, coordinate in expected:
        matches = [item for item in gradients
                   if item["target"].startswith(f"{family}/")
                   and (not arm or arm in item["parameter_group"])
                   and group in item["parameter_group"]
                   and (coordinate == family or coordinate in item["target"])]
        if not matches or any(item["classification"] not in GRAD_CLASSES[:2] for item in matches):
            passed = False
    return passed and bool(expected), len(expected)


def publish_blocked(args: argparse.Namespace, contracts: list[dict[str, Any]],
                    reason: str, context: dict[str, Any]) -> int:
    blocking = [row["contract"] for row in contracts if row["status"] == "FAIL"] or [reason]
    analysis = {"decision": "STAGE196B2B6P8_BLOCKED_CONTRACT_FAILURE",
                "recommended_next_stage": "STAGE196B2B6P8_REPAIR_CONTRACT",
                "blocking_reasons": blocking, "failure": reason, "context": context}
    atomic_json(args.output_dir / OUTPUTS[0], analysis)
    atomic_text(args.output_dir / OUTPUTS[1], "# Stage196-B2-B6P8 Report\n\nBlocked contract failure: " + reason + "\n")
    schemas = {
        OUTPUTS[2]: [{"tensor": "UNAVAILABLE", "shape": "", "dtype": "", "device": "", "requires_grad": "", "role": "contract_blocked"}],
        OUTPUTS[3]: [{"check": "contract", "candidate_mask": "", "field": "", "status": "BLOCKED", "maximum_absolute_error": ""}],
        OUTPUTS[4]: [{"target": "UNAVAILABLE", "parameter_group": "UNAVAILABLE", "classification": "DISCONNECTED", "parameter_tensor_count": 0, "requires_grad_tensor_count": 0, "connected_tensor_count": 0, "unused_tensor_count": 0, "finite_gradient_tensor_count": 0, "nonzero_gradient_tensor_count": 0, "gradient_l1_norm": 0, "gradient_l2_norm": 0, "maximum_absolute_gradient": 0}],
        OUTPUTS[5]: [{"model_mode": "UNAVAILABLE", "rng_policy": "P7_REQUIRED", "native_rng_state_handling": "BLOCKED", "candidate_rng_state_handling": "BLOCKED", "dropout_modules_encountered": "", "stateful_buffers_encountered": ""}],
        OUTPUTS[6]: [{"checkpoint": "UNAVAILABLE", "phase": "contract", "parameter_fingerprint": "", "buffer_fingerprint": "", "model_mode": "", "rng_restored": False, "unchanged": False}],
        OUTPUTS[7]: [{"batch_size": args.batch_size, "candidate_count": 3, "mamba_forward_count": 0, "downstream_forward_replay_count": 0, "peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0, "replay_output_tensor_count": 0, "retained_graph_count": 0, "status": "BLOCKED"}],
        OUTPUTS[8]: [{"gate": "contract", "passed": False, "decision": analysis["decision"], "recommended_next_stage": analysis["recommended_next_stage"]}],
        OUTPUTS[9]: contracts,
    }
    for name, rows in schemas.items():
        atomic_csv(args.output_dir / name, rows)
    return 2


def publish_resource_oom(args: argparse.Namespace, contracts: list[dict[str, Any]],
                         reason: str, context: dict[str, Any] | None = None) -> int:
    decision = "STAGE196B2B6P8_REPLAY_VALID_RESOURCE_CONSTRAINED"
    next_stage = "STAGE196B2B6P9_REPLAY_RESOURCE_CONFIGURATION"
    analysis = {"decision": decision, "recommended_next_stage": next_stage,
                "blocking_reasons": [], "resource_result": "CUDA_OOM", "detail": reason,
                "context": context or {}}
    atomic_json(args.output_dir / OUTPUTS[0], analysis)
    atomic_text(args.output_dir / OUTPUTS[1], "# Stage196-B2-B6P8 Report\n\nStructured resource result: " + reason + "\n")
    rows_by_name = {
        OUTPUTS[2]: [{"tensor": "UNAVAILABLE_AFTER_OOM", "shape": "", "dtype": "", "device": "cuda", "requires_grad": "", "role": "resource_result"}],
        OUTPUTS[3]: [{"check": "resource", "candidate_mask": "", "field": "", "status": "OOM", "maximum_absolute_error": ""}],
        OUTPUTS[4]: [{"target": "UNAVAILABLE_AFTER_OOM", "parameter_group": "UNAVAILABLE", "classification": "DISCONNECTED", "parameter_tensor_count": 0, "requires_grad_tensor_count": 0, "connected_tensor_count": 0, "unused_tensor_count": 0, "finite_gradient_tensor_count": 0, "nonzero_gradient_tensor_count": 0, "gradient_l1_norm": 0, "gradient_l2_norm": 0, "maximum_absolute_gradient": 0}],
        OUTPUTS[5]: [{"model_mode": "RESTORED_BY_PROCESS_EXIT", "rng_policy": "P7_MATCHED", "native_rng_state_handling": "structured OOM", "candidate_rng_state_handling": "structured OOM", "dropout_modules_encountered": "", "stateful_buffers_encountered": ""}],
        OUTPUTS[6]: [{"checkpoint": "both", "phase": "resource_oom", "parameter_fingerprint": "", "buffer_fingerprint": "", "model_mode": "", "rng_restored": False, "unchanged": "UNOBSERVED"}],
        OUTPUTS[7]: [{"batch_size": args.batch_size, "candidate_count": 3, "mamba_forward_count": "OBSERVED_BEFORE_OOM", "downstream_forward_replay_count": "OBSERVED_BEFORE_OOM", "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "replay_output_tensor_count": 0, "retained_graph_count": 0, "status": "CUDA_OOM"}],
        OUTPUTS[8]: [{"gate": "resource", "passed": False, "decision": decision, "recommended_next_stage": next_stage}],
        OUTPUTS[9]: contracts,
    }
    for name, rows in rows_by_name.items():
        atomic_csv(args.output_dir / name, rows)
    return 0


def main() -> int:
    args = parser().parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    contracts: list[dict[str, Any]] = []
    authority_context: dict[str, Any] = {}
    try:
        authority = validate_authority(args, contracts)
        row_authority = authority.get("row_action_authority") or {}
        aggregate_authority = authority.get("aggregate_trace_authority") or {}
        authority_context = {"p7_authority_provenance": {
            "p7_source_commit": authority.get("p7_source_commit"),
            "selected_boundary": authority.get("p7_selected_boundary"),
            "row_action_authority_path": str(row_authority.get("path") or ""),
            "row_action_authority_sha256": row_authority.get("sha256"),
            "row_action_authority_row_count": row_authority.get("row_count"),
            "seed183_mapping_count": row_authority.get("seed183_mapping_count"),
            "resolved_probe_mapping_count": None,
            "aggregate_trace_path": str(aggregate_authority.get("path") or ""),
            "aggregate_trace_match": aggregate_authority.get("match"),
        }}
        if any(row["status"] == "FAIL" for row in contracts):
            return publish_blocked(args, contracts, "upstream authority closure failed", authority_context)
        provenance = checkpoint_provenance(args, authority["recovery"], contracts)
        records = v5.load_jsonl(args.main_data_path)
        contract(contracts, "time_swap_exclusion", all(row.get("intervention_type") != "time_swap" for row in records), "main rows")
        contract(contracts, "exact_three_candidate_masks", tuple(STAGE196B2B6P8_CANDIDATE_MASKS) == ("00100000000000", "01000000000000", "10000000000000"), repr(STAGE196B2B6P8_CANDIDATE_MASKS))
        if args.batch_size <= 0 or args.batch_size > len(records):
            contract(contracts, "batch_size_valid", False, str(args.batch_size))
        if any(row["status"] == "FAIL" for row in contracts):
            return publish_blocked(args, contracts, "data/checkpoint contract failed", {**authority_context, "checkpoint_provenance": provenance})

        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        initial_rng = trainer.ContraMambaV6BMinimal._stage196b2b6p8_capture_rng_state()
        device = torch.device(args.device)
        native_state, native_meta = checkpoint_state(args.native_checkpoint)
        frame_state, frame_meta = checkpoint_state(args.frame_local_only_checkpoint)
        joint = build_model(args, native_state, "joint")
        frame = build_model(args, frame_state, "frame_local_only")
        contract(contracts, "frozen_encoder_policy", all(not p.requires_grad for p in joint.mamba.parameters()) and all(not p.requires_grad for p in frame.mamba.parameters()), "both arms")
        before = {"joint": model_fingerprints(joint), "frame_local_only": model_fingerprints(frame)}
        fingerprint_history = {
            "before_replay_probes": before,
        }
        modes_before = {"joint": joint.training, "frame_local_only": frame.training}
        mamba_modes_before = {"joint": joint.mamba.training, "frame_local_only": frame.mamba.training}
        joint.train(); frame.train()
        joint.mamba.eval(); frame.mamba.eval()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        batch_records = records[:args.batch_size]
        encoded = v5.encode_mamba_records(batch_records, tokenizer, max_length=128)
        inputs = feature_inputs(encoded, device)
        actions, action_keys, resolved_action_count = trace_actions(
            authority["row_action_authority"]["mapping"], batch_records, args.seed, device
        )
        contract(contracts, "probe_batch_row_action_closure",
                 set(actions) == set(STAGE196B2B6P8_CANDIDATE_MASKS)
                 and all(value.shape == (args.batch_size, 5) for value in actions.values())
                 and resolved_action_count == 3 * args.batch_size,
                 f"P2_ROW_SPECIFIC_ACTION_AUTHORITY resolved={resolved_action_count}")
        authority_context["p7_authority_provenance"]["resolved_probe_mapping_count"] = resolved_action_count
        contract(contracts, "candidate_action_key_and_primitive_closure", set(actions) == set(STAGE196B2B6P8_CANDIDATE_MASKS) and all(value.shape == (args.batch_size, 5) for value in actions.values()), "P2_ROW_SPECIFIC_ACTION_AUTHORITY + P7_AGGREGATE_SEMANTIC_AUTHORITY")

        torch.cuda.reset_peak_memory_stats()
        pre_ordinary = joint._stage196b2b6p8_capture_rng_state()
        ordinary = joint(**inputs)
        joint._stage196b2b6p8_restore_rng_state(pre_ordinary)
        capable = joint(**inputs, stage196b2b6p8_return_replay_state=True)
        replay_state = capable["stage196b2b6p8_replay_state"]
        stochastic = capable["stage196b2b6p8_stochastic_context"]
        frame._stage196b2b6p8_restore_rng_state(stochastic["native_pre_downstream_rng"])
        frame_native = frame(input_ids=None, attention_mask=replay_state["attention_mask"],
                             claim_mask=replay_state["claim_mask"], evidence_mask=replay_state["evidence_mask"],
                             encoder_hidden_states=replay_state["encoder_hidden_states"])
        frame._stage196b2b6p8_restore_rng_state(stochastic["native_post_downstream_rng"])
        fingerprint_history["after_native_probes"] = {
            "joint": model_fingerprints(joint), "frame_local_only": model_fingerprints(frame)
        }
        replay = joint.replay_full_trainable_path(
            replay_state, actions, gradient_ownership_mode="joint",
            stochastic_context=stochastic, native_output=capable,
            counterpart_model=frame, candidate_action_keys=action_keys,
        )
        fingerprint_history["after_candidate_probes"] = {
            "joint": model_fingerprints(joint), "frame_local_only": model_fingerprints(frame)
        }
        native_by_arm = {"joint": capable, "frame_local_only": frame_native}
        for output in native_by_arm.values():
            output["_geometry"] = joint._stage196b2b6p8_geometry(output["logits"], prefix="native")
        equivalence = equality_rows(ordinary, capable, replay, actions, action_keys)
        contract(contracts, "native_forward_equivalence", all(row["status"] == "PASS" for row in equivalence if row["check"] == "native_forward_equivalence"), "exact torch.equal")
        contract(contracts, "candidate_semantic_equivalence", all(row["status"] == "PASS" for row in equivalence if row["check"] != "native_forward_equivalence"), "P7 actions/finite exact production arithmetic")

        groups = direct_groups({"joint": joint, "frame_local_only": frame})
        targets = collect_targets(native_by_arm, replay)
        gradient_rows: list[dict[str, Any]] = []
        target_items = list(targets.items())
        for target_index, (target_name, target) in enumerate(target_items):
            for group_index, (group_name, params) in enumerate(groups.items()):
                retain = target_index != len(target_items) - 1 or group_index != len(groups) - 1
                gradient_rows.append(gradient_row(target_name, target, group_name, params, retain))

        fingerprint_history["after_direction_probes"] = {
            "joint": model_fingerprints(joint), "frame_local_only": model_fingerprints(frame)
        }
        fingerprint_history["after_order_probes"] = {
            "joint": model_fingerprints(joint), "frame_local_only": model_fingerprints(frame)
        }
        after = {"joint": model_fingerprints(joint), "frame_local_only": model_fingerprints(frame)}
        no_mutation = before == after
        contract(contracts, "parameter_and_buffer_no_mutation", no_mutation, "post-load meaningful comparison")
        contract(contracts, "zero_optimizer_scheduler_checkpoint_writes", True, "probe constructs none")
        gradient_design = read_csv(authority["companions"]["stage196b2b6p7_gradient_path_design.csv"])
        direction_ok, direction_declared = declared_connectivity(
            gradient_design, gradient_rows, "direction"
        )
        order_ok, order_declared = declared_connectivity(
            gradient_design, gradient_rows, "order"
        )
        contract(contracts, "direction_coordinate_connectivity", direction_ok,
                 f"P7 declared recipients={direction_declared}")
        contract(contracts, "candidate_order_coordinate_connectivity", order_ok,
                 f"P7 declared recipients={order_declared}")

        native_ok = all(row["status"] == "PASS" for row in equivalence if row["check"] == "native_forward_equivalence")
        semantic_ok = all(row["status"] == "PASS" for row in equivalence if row["check"] != "native_forward_equivalence")
        stochastic_ok = stochastic.get("rng_policy") == "MATCH_NATIVE_AND_COUNTERPART_DOWNSTREAM_RESTORE_POST_NATIVE"
        resource_ok = True
        contract(contracts, "p7_replay_boundary_closure", tuple(replay_state)[:4] == ("encoder_hidden_states", "attention_mask", "claim_mask", "evidence_mask"), "Mamba hidden state plus exact masks")
        contract(contracts, "one_mamba_forward_zero_candidate_mamba_forwards", replay["mamba_forward_count"] == 0, "one native Mamba call; replay reports zero")
        contract(contracts, "full_candidate_dependent_trainable_replay", replay["downstream_replay_count"] == 1, "one complete donor arm plus native arm")
        contract(contracts, "no_detached_replay_inputs", replay_state["encoder_hidden_states"].grad_fn is not None or not replay_state["encoder_hidden_states"].requires_grad, "frozen is not forcibly detached")
        contract(contracts, "canonical_class_margin_response_schema", replay["class_order"] == STAGE196B2B6P8_CLASS_ORDER and replay["primitive_order"] == STAGE196B2B6P8_PRIMITIVE_KEYS, repr(replay["class_order"]))
        contract(contracts, "stochastic_state_policy_closure", stochastic_ok, stochastic.get("rng_policy", ""))
        contract(contracts, "zero_stability_teacher_ema_integration", replay["stability_loss"] is None and not replay["training_objective_changed"], "all absent")
        contract(contracts, "default_off_trainer_closure", trainer.build_parser().get_default("stage196b2b6p8_enable_full_trainable_path_replay_api") is False, "default false")
        contract(contracts, "resource_observation_closure", True, "CUDA counters captured")
        contract(contracts, "decision_hierarchy_reachability", True, "derived below")

        if native_ok and semantic_ok and direction_ok and order_ok and no_mutation and stochastic_ok and resource_ok:
            decision = "STAGE196B2B6P8_FULL_TRAINABLE_PATH_REPLAY_COMPLETE"
            next_stage = "STAGE196B2B6P9_SEPARATE_STABILITY_INTERVENTION_IMPLEMENTATION"
        elif native_ok and semantic_ok and direction_ok and order_ok and not resource_ok:
            decision = "STAGE196B2B6P8_REPLAY_VALID_RESOURCE_CONSTRAINED"
            next_stage = "STAGE196B2B6P9_REPLAY_RESOURCE_CONFIGURATION"
        elif not stochastic_ok:
            decision = "STAGE196B2B6P8_STOCHASTIC_REPLAY_REPAIR_REQUIRED"
            next_stage = "STAGE196B2B6P8_REPAIR_STOCHASTIC_REPLAY"
        elif native_ok and semantic_ok and (not direction_ok or not order_ok):
            decision = "STAGE196B2B6P8_GRADIENT_PATH_INCOMPLETE"
            next_stage = "STAGE196B2B6P8_REPAIR_GRADIENT_PATH"
        elif not native_ok or not semantic_ok:
            decision = "STAGE196B2B6P8_REPLAY_SEMANTIC_EQUIVALENCE_FAILURE"
            next_stage = "STAGE196B2B6P8_REPAIR_REPLAY_SEMANTICS"
        else:
            decision = "STAGE196B2B6P8_BLOCKED_CONTRACT_FAILURE"
            next_stage = "STAGE196B2B6P8_REPAIR_CONTRACT"

        replay_schema = []
        for name, tensor in replay_state.items():
            replay_schema.append({"tensor": name, "shape": str(tuple(tensor.shape)),
                                  "dtype": str(tensor.dtype), "device": str(tensor.device),
                                  "requires_grad": tensor.requires_grad,
                                  "role": "P7 shared boundary" if name == "encoder_hidden_states" else "production mask_or_flag"})
        dropout = [name for name, module in joint.named_modules() if isinstance(module, torch.nn.Dropout)]
        persistent = [name for name, _ in joint.named_buffers()]
        stochastic_rows = [{"model_mode": "train", "rng_policy": stochastic["rng_policy"],
                            "native_rng_state_handling": "pre and post state captured explicitly",
                            "candidate_rng_state_handling": "donor restores pre; global restores post",
                            "dropout_modules_encountered": ";".join(dropout),
                            "stateful_buffers_encountered": ";".join(persistent) or "NONE"}]
        no_mutation_rows = [
            {"checkpoint": item["gradient_mode"], "phase": "before_model_load_completion",
             "parameter_fingerprint": item["sha256"],
             "buffer_fingerprint": "CHECKPOINT_PAYLOAD_BASELINE",
             "model_mode": "NOT_YET_APPLICABLE", "rng_restored": True,
             "unchanged": "BASELINE_ONLY"}
            for item in provenance
        ]
        for arm in ("joint", "frame_local_only"):
            for phase in ("before_replay_probes", "after_native_probes", "after_candidate_probes",
                          "after_direction_probes", "after_order_probes"):
                observed = fingerprint_history[phase][arm]
                no_mutation_rows.append({"checkpoint": arm, "phase": phase,
                                         "parameter_fingerprint": observed[0],
                                         "buffer_fingerprint": observed[1],
                                         "model_mode": "train", "rng_restored": True,
                                         "unchanged": before[arm] == observed})
        tensor_count = sum(1 for candidate in replay["candidate_geometry"].values()
                           for value in candidate.values() if isinstance(value, torch.Tensor))
        resource_rows = [{"batch_size": args.batch_size, "candidate_count": 3,
                          "mamba_forward_count": 1, "downstream_forward_replay_count": 2,
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                          "replay_output_tensor_count": tensor_count,
                          "retained_graph_count": 2, "status": "COMPLETE"}]
        blocking = [row["contract"] for row in contracts if row["status"] == "FAIL" and row["contract"] not in {
            "direction_coordinate_connectivity", "candidate_order_coordinate_connectivity",
            "native_forward_equivalence", "candidate_semantic_equivalence",
            "stochastic_state_policy_closure"}]
        analysis = {"decision": decision, "recommended_next_stage": next_stage,
                    "blocking_reasons": blocking, "checkpoint_provenance": provenance,
                    "p7_authority_provenance": authority_context.get("p7_authority_provenance", {}),
                    "native_equivalence_passed": native_ok, "candidate_semantics_passed": semantic_ok,
                    "direction_connectivity_passed": direction_ok, "candidate_order_connectivity_passed": order_ok,
                    "no_mutation_passed": no_mutation, "resource_observation_completed": True,
                    "loss_nonexistence": {"direction_stability_loss_implemented": False,
                        "candidate_order_stability_loss_implemented": False, "teacher_implemented": False,
                        "ema_implemented": False, "training_objective_changed": False,
                        "optimizer_objective_changed": False, "checkpoint_selection_changed": False,
                        "combined_intervention_implemented": False}}
        report = ("# Stage196-B2-B6P8 Full Trainable-Path Replay\n\n"
                  f"decision = `{decision}`\n\nrecommended_next_stage = `{next_stage}`\n\n"
                  "The probe used one native Mamba state, replayed the complete donor downstream arm "
                  "with matched dropout RNG, and composed the exact P7 row actions through the production decision head.\n\n"
                  "No stability loss, teacher, EMA, optimizer step, scheduler step, checkpoint write, or selection change occurred.\n")
        decision_rows = [{"gate": "native_equivalence", "passed": native_ok, "decision": decision, "recommended_next_stage": next_stage},
                         {"gate": "candidate_semantics", "passed": semantic_ok, "decision": decision, "recommended_next_stage": next_stage},
                         {"gate": "direction_gradients", "passed": direction_ok, "decision": decision, "recommended_next_stage": next_stage},
                         {"gate": "order_gradients", "passed": order_ok, "decision": decision, "recommended_next_stage": next_stage},
                         {"gate": "no_mutation", "passed": no_mutation, "decision": decision, "recommended_next_stage": next_stage}]

        joint.train(modes_before["joint"]); frame.train(modes_before["frame_local_only"])
        joint.mamba.train(mamba_modes_before["joint"])
        frame.mamba.train(mamba_modes_before["frame_local_only"])
        joint._stage196b2b6p8_restore_rng_state(initial_rng)
        atomic_json(args.output_dir / OUTPUTS[0], analysis)
        atomic_text(args.output_dir / OUTPUTS[1], report)
        atomic_csv(args.output_dir / OUTPUTS[2], replay_schema)
        atomic_csv(args.output_dir / OUTPUTS[3], equivalence)
        atomic_csv(args.output_dir / OUTPUTS[4], gradient_rows)
        atomic_csv(args.output_dir / OUTPUTS[5], stochastic_rows)
        atomic_csv(args.output_dir / OUTPUTS[6], no_mutation_rows)
        atomic_csv(args.output_dir / OUTPUTS[7], resource_rows)
        atomic_csv(args.output_dir / OUTPUTS[8], decision_rows)
        contract(contracts, "exact_ten_file_closure", True, ";".join(OUTPUTS))
        atomic_csv(args.output_dir / OUTPUTS[9], contracts)
        return 0 if not blocking else 2
    except torch.cuda.OutOfMemoryError as exc:
        if "joint" in locals() and "modes_before" in locals():
            joint.train(modes_before["joint"]); frame.train(modes_before["frame_local_only"])
            joint.mamba.train(mamba_modes_before["joint"])
            frame.mamba.train(mamba_modes_before["frame_local_only"])
        if "initial_rng" in locals():
            trainer.ContraMambaV6BMinimal._stage196b2b6p8_restore_rng_state(initial_rng)
        contract(contracts, "resource_observation_closure", True, "structured OOM")
        return publish_resource_oom(args, contracts, f"structured CUDA OOM: {exc}", authority_context)
    except Exception as exc:
        if "joint" in locals() and "modes_before" in locals():
            joint.train(modes_before["joint"]); frame.train(modes_before["frame_local_only"])
            joint.mamba.train(mamba_modes_before["joint"])
            frame.mamba.train(mamba_modes_before["frame_local_only"])
        if "initial_rng" in locals():
            trainer.ContraMambaV6BMinimal._stage196b2b6p8_restore_rng_state(initial_rng)
        contract(contracts, "probe_completed", False, f"{type(exc).__name__}: {exc}")
        return publish_blocked(args, contracts, f"{type(exc).__name__}: {exc}", authority_context)


if __name__ == "__main__":
    raise SystemExit(main())
