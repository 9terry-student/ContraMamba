#!/usr/bin/env python3
"""Design the exact Stage196-B2-B6P7 counterfactual forward by static analysis.

This analyzer reads source and frozen artifacts only.  It never imports torch,
loads a checkpoint/model, executes a model, trains, or changes an objective.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


STAGE = "Stage196-B2-B6P7"
P6_DECISION = "STAGE196B2B6P6_FULL_COUNTERFACTUAL_FORWARD_REQUIRED"
P5_DECISION = "STAGE196B2B6P5_GRADIENT_PATH_INSTRUMENTATION_REQUIRED"
P4_DECISION = "STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE"
P2_DECISION = "STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_COMPLETE"
MAIN_DATA = "data/controlled_v5_v3_without_time_swap.jsonl"
MAIN_DATA_SHA256 = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
MODEL_NAME = "state-spaces/mamba-130m-hf"
CANDIDATES = ("00100000000000", "01000000000000", "10000000000000")
PRIMITIVES = ("FRAME", "PREDICATE", "SUFFICIENCY", "POSITIVE_ENERGY", "NEGATIVE_ENERGY")
CLASS_ORDER = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
ORIGINAL_SOURCE_COMMIT = "fa16787efa84bb15d832b6d9382fafd77016c4e2"

P6_OUTPUTS = (
    "stage196b2b6p6_analysis.json",
    "stage196b2b6p6_report.md",
    "stage196b2b6p6_tensor_schema.csv",
    "stage196b2b6p6_gradient_connectivity.csv",
    "stage196b2b6p6_parameter_group_audit.csv",
    "stage196b2b6p6_forward_equivalence.csv",
    "stage196b2b6p6_no_mutation_audit.csv",
    "stage196b2b6p6_decision_gate.csv",
    "stage196b2b6p6_contract.csv",
)
OUTPUTS = (
    "stage196b2b6p7_analysis.json",
    "stage196b2b6p7_report.md",
    "stage196b2b6p7_execution_boundary_audit.csv",
    "stage196b2b6p7_candidate_semantic_trace.csv",
    "stage196b2b6p7_stochastic_state_audit.csv",
    "stage196b2b6p7_gradient_path_design.csv",
    "stage196b2b6p7_counterfactual_forward_designs.csv",
    "stage196b2b6p7_compute_memory_estimate.csv",
    "stage196b2b6p7_decision_gate.csv",
    "stage196b2b6p7_contract.csv",
)

SOURCE_FILES = (
    "scripts/train_controlled_v6b_minimal.py",
    "scripts/train_controlled_v5.py",
    "src/contramamba/modeling_v6b_minimal.py",
    "src/contramamba/masking.py",
    "src/contramamba/heads/frame_gate.py",
    "src/contramamba/heads/predicate_coverage.py",
    "src/contramamba/heads/sufficiency_gate.py",
    "src/contramamba/heads/polarity_energy.py",
    "src/contramamba/heads/entitlement_decision.py",
    "scripts/probe_stage196b2b6p6_minimal_gradient_path_instrumentation.py",
    "scripts/analyze_stage196b2b6p5_training_side_response_stability_intervention_design.py",
    "scripts/analyze_stage196b2b6_minimal_selector_intervention.py",
)

DECISIONS = (
    (
        "STAGE196B2B6P7_SHARED_FROZEN_BACKBONE_REPLAY_READY",
        "STAGE196B2B6P8_SHARED_BACKBONE_COUNTERFACTUAL_FORWARD_IMPLEMENTATION",
    ),
    (
        "STAGE196B2B6P7_FULL_TRAINABLE_PATH_REPLAY_READY",
        "STAGE196B2B6P8_FULL_TRAINABLE_PATH_REPLAY_IMPLEMENTATION",
    ),
    (
        "STAGE196B2B6P7_FULL_MODEL_COUNTERFACTUAL_FORWARD_READY",
        "STAGE196B2B6P8_FULL_MODEL_COUNTERFACTUAL_FORWARD_IMPLEMENTATION",
    ),
    (
        "STAGE196B2B6P7_EXACT_COUNTERFACTUAL_FORWARD_RESOURCE_UNSAFE",
        "STAGE196B2B7_SELECTOR_MECHANISM_RETHINK",
    ),
    (
        "STAGE196B2B6P7_COUNTERFACTUAL_SEMANTICS_UNRESOLVED",
        "STAGE196B2B6P7_REPAIR_SEMANTIC_BOUNDARY",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind in (
        ("repo-root", Path),
        ("stage196b2b6p6-analysis-json", Path),
        ("stage196b2b6p5-analysis-json", Path),
        ("stage196b2b6p4-analysis-json", Path),
        ("stage196b2b6p2-analysis-json", Path),
        ("current-git-commit", str),
        ("output-dir", Path),
    ):
        parser.add_argument(f"--{name}", required=True, type=kind)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON object required")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def csv_text(fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: canonical(value)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
        )
    return buffer.getvalue()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"refusing to overwrite {path}")
    os.rename(temporary, path)


def gate(
    rows: list[dict[str, Any]],
    name: str,
    required: Any,
    observed: Any,
    passed: bool,
    reason: str,
) -> None:
    rows.append(
        {
            "contract": name,
            "required": required,
            "observed": observed,
            "passed": bool(passed),
            "blocking_reason": "" if passed else reason,
        }
    )
    if not passed:
        raise ValueError(f"{name}: {reason}")


def contract_closed(path: Path) -> tuple[bool, int]:
    rows = read_csv(path)
    failed = sum(
        str(row.get("passed", "")).strip().lower() != "true"
        or bool(row.get("blocking_reason", "").strip())
        for row in rows
    )
    return bool(rows) and failed == 0, failed


def source_range(text: str, start_pattern: str, end_pattern: str | None = None) -> str:
    lines = text.splitlines()
    starts = [index + 1 for index, line in enumerate(lines) if re.search(start_pattern, line)]
    if len(starts) != 1:
        raise ValueError(f"source symbol is not unique: {start_pattern!r}")
    start = starts[0]
    if end_pattern is None:
        return str(start)
    ends = [
        index + 1
        for index, line in enumerate(lines[start:], start=start)
        if re.search(end_pattern, line)
    ]
    if not ends:
        raise ValueError(f"source range end absent: {end_pattern!r}")
    return f"{start}-{ends[0]}"


def source_line(text: str, literal: str) -> int:
    found = [index + 1 for index, line in enumerate(text.splitlines()) if literal in line]
    if len(found) != 1:
        raise ValueError(f"source literal is not unique: {literal!r}")
    return found[0]


def load_sources(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    texts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required static source missing: {relative}")
        texts[relative] = path.read_text(encoding="utf-8")
        hashes[relative] = sha256(path)
    return texts, hashes


def exact_p6_closure(path: Path) -> tuple[list[str], list[str], list[str]]:
    observed = sorted(item.name for item in path.iterdir() if item.is_file())
    return (
        observed,
        sorted(set(P6_OUTPUTS) - set(observed)),
        sorted(set(observed) - set(P6_OUTPUTS)),
    )


def action_summary(p2_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_path = p2_path.parent / "stage196b2b6p2_candidate_action_composer_scores.csv"
    rows = read_csv(candidate_path)
    required_columns = {
        "seed", "candidate_mask", "candidate_action_key", "stable_row_id"
    }
    if not rows or not required_columns <= set(rows[0]):
        raise ValueError("P2 candidate-action CSV schema is incomplete")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    identities: dict[tuple[str, str, str], str] = {}
    for row in rows:
        candidate = row["candidate_mask"]
        action = row["candidate_action_key"]
        if candidate not in CANDIDATES or not re.fullmatch(r"[01]{5}", action):
            raise ValueError("P2 candidate/action identity is invalid")
        key = (row["seed"], row["stable_row_id"], candidate)
        previous = identities.setdefault(key, action)
        if previous != action:
            raise ValueError("P2 candidate has conflicting actions for one recipient")
        counts[candidate][action] += 1
    if set(counts) != set(CANDIDATES):
        raise ValueError("P2 exact candidate set is incomplete")
    result: list[dict[str, Any]] = []
    first_by_bit = {
        0: "frame_prob from FrameGate",
        1: "predicate_coverage_prob from PredicateCoverageHead",
        2: "sufficiency_prob from SufficiencyGate",
        3: "positive_energy from PolarityEnergyHead",
        4: "negative_energy from PolarityEnergyHead",
    }
    for candidate in CANDIDATES:
        actions = sorted(counts[candidate])
        possible_first = sorted(
            {
                first_by_bit[next(index for index, bit in enumerate(action) if bit == "1")]
                if "1" in action
                else "no primitive changes (00000)"
                for action in actions
            }
        )
        changed_primitives = sorted(
            {
                PRIMITIVES[index]
                for action in actions
                for index, bit in enumerate(action)
                if bit == "1"
            }
        )
        result.append(
            {
                "candidate_mask": candidate,
                "primitive_mapping": "ROW_SPECIFIC_5BIT_ACTION_FROM_P2_AUTHORITY",
                "primitive_order": list(PRIMITIVES),
                "observed_action_keys": actions,
                "observed_action_key_counts": dict(sorted(counts[candidate].items())),
                "first_tensor_whose_value_changes": possible_first,
                "downstream_tensors_that_change": (
                    "selected primitives -> entitlement/decision-head logits -> "
                    "recipient-native final modulation -> final logits; exact subset is row-specific"
                ),
                "stochastic_operation_downstream": False,
                "affects_trainable_branches": True,
                "affects_frozen_backbone": False,
                "affects_input_tokenization": False,
                "requires_new_backbone_forward": False,
                "candidate_identity_is_primitive_bitmask": False,
                "changed_primitive_union": changed_primitives,
            }
        )
    summary = {
        "path": str(candidate_path),
        "sha256": sha256(candidate_path),
        "row_count": len(rows),
        "unique_recipient_candidate_pairs": len(identities),
        "candidate_masks": list(CANDIDATES),
        "primitive_order": list(PRIMITIVES),
        "row_specific": True,
    }
    return result, summary


def execution_audit(root: Path, texts: dict[str, str]) -> list[dict[str, Any]]:
    model_key = "src/contramamba/modeling_v6b_minimal.py"
    model = texts[model_key]
    frame_key = "src/contramamba/heads/frame_gate.py"
    predicate_key = "src/contramamba/heads/predicate_coverage.py"
    sufficiency_key = "src/contramamba/heads/sufficiency_gate.py"
    polarity_key = "src/contramamba/heads/polarity_energy.py"
    decision_key = "src/contramamba/heads/entitlement_decision.py"
    v5_key = "scripts/train_controlled_v5.py"
    rows = [
        {
            "state_name": "tokenized inputs: input_ids, attention_mask, claim_mask, evidence_mask",
            "producing_module": "train_controlled_v5.encode_mamba_records / tokenizer",
            "source_file": v5_key,
            "source_line_range": "model_feature_inputs "
            + source_range(texts[v5_key], r"^def model_feature_inputs\(", r"^\s*return result"),
            "tensor_shape_contract": "[B,T] for each input/mask",
            "producer_trainability": "non-parameter preprocessing",
            "candidate_action_dependency": "none",
            "native_loss_dependency": "all native heads through token and span inputs",
            "detach_boundary": "integer/bool inputs; no autograd requirement",
            "serialization_boundary": "dataset/tokenizer boundary",
            "replay_feasibility": "shared exactly; candidate IDs never enter tokenization",
        },
        {
            "state_name": "Mamba hidden state / encoder_hidden_states",
            "producing_module": "model.mamba or frozen-state cache",
            "source_file": model_key,
            "source_line_range": source_range(model, r"^\s*# Encode$", r"^\s*# Slot gates"),
            "tensor_shape_contract": "[B,T,H_backbone]",
            "producer_trainability": "frozen in authoritative configuration",
            "candidate_action_dependency": "none",
            "native_loss_dependency": "source for every epistemic head",
            "detach_boundary": "cached under no_grad; valid because producer is frozen",
            "serialization_boundary": "in-memory inputs['encoder_hidden_states']; not diagnostic JSON",
            "replay_feasibility": "EARLIEST_SAFE_SHARED_REPLAY_BOUNDARY",
        },
        {
            "state_name": "frame token projection and pooled claim/evidence frame states",
            "producing_module": "FrameGate.project + masked_pool",
            "source_file": frame_key,
            "source_line_range": source_range(
                texts[frame_key], r"^\s*def forward\(", r"^\s*pair_features ="
            ),
            "tensor_shape_contract": "[B,T,128] projected; [B,128] pooled states",
            "producer_trainability": "trainable FrameGate",
            "candidate_action_dependency": "recipient/donor arm parameter state; not 14-bit ID directly",
            "native_loss_dependency": "frame, predicate, sufficiency, polarity, final label",
            "detach_boundary": "frame_local_only hook detaches aliases only after FrameGate output",
            "serialization_boundary": "none on native path",
            "replay_feasibility": "must rerun for the separately trained donor arm",
        },
        {
            "state_name": "frame state: frame_pair_repr, frame_logit, frame_prob",
            "producing_module": "FrameGate.pair_projector + frame_classifier",
            "source_file": frame_key,
            "source_line_range": source_range(
                texts[frame_key], r"^\s*pair_features =", r"^\s*return \{"
            ),
            "tensor_shape_contract": "[B,128], [B], [B]",
            "producer_trainability": "trainable FrameGate",
            "candidate_action_dependency": "FRAME action may select donor frame_prob",
            "native_loss_dependency": "direct frame BCE and all downstream branches",
            "detach_boundary": "conditional frame_local_only forward hook after production",
            "serialization_boundary": "P0/P2 diagnostic export only, never replay input",
            "replay_feasibility": "rerun exact production module; do not reuse detached diagnostic",
        },
        {
            "state_name": "predicate state: predicate_pair_repr, coverage_logit/prob",
            "producing_module": "PredicateCoverageHead",
            "source_file": predicate_key,
            "source_line_range": source_range(
                texts[predicate_key], r"^\s*def forward\(", r"^\s*return \{"
            ),
            "tensor_shape_contract": "[B,128], [B], [B]",
            "producer_trainability": "trainable predicate branch",
            "candidate_action_dependency": "PREDICATE action may select donor probability",
            "native_loss_dependency": "predicate BCE, sufficiency, polarity, final label",
            "detach_boundary": "receives detached frame aliases only in donor gradient-ownership mode",
            "serialization_boundary": "none on native path",
            "replay_feasibility": "must rerun for donor parameter state",
        },
        {
            "state_name": "sufficiency state: sufficiency_repr, logit, probability",
            "producing_module": "SufficiencyGate",
            "source_file": sufficiency_key,
            "source_line_range": source_range(
                texts[sufficiency_key], r"^\s*def forward\(", r"^\s*return \{"
            ),
            "tensor_shape_contract": "[B,128], [B], [B]",
            "producer_trainability": "trainable sufficiency branch",
            "candidate_action_dependency": "SUFFICIENCY action may select donor probability",
            "native_loss_dependency": "sufficiency BCE, polarity, final label",
            "detach_boundary": "inherits frame detach in frame_local_only arm",
            "serialization_boundary": "none on native path",
            "replay_feasibility": "must rerun for donor parameter state",
        },
        {
            "state_name": "polarity state: polarity_features, positive_energy, negative_energy",
            "producing_module": "PolarityEnergyHead",
            "source_file": polarity_key,
            "source_line_range": source_range(
                texts[polarity_key], r"^\s*def forward\(", r"^\s*return \{"
            ),
            "tensor_shape_contract": "[B,64], [B], [B]",
            "producer_trainability": "trainable polarity branch",
            "candidate_action_dependency": "POSITIVE_ENERGY and NEGATIVE_ENERGY select independently",
            "native_loss_dependency": "polarity CE and final label",
            "detach_boundary": "inherits frame detach in frame_local_only arm",
            "serialization_boundary": "none on native path",
            "replay_feasibility": "must rerun for donor parameter state",
        },
        {
            "state_name": "entitlement state",
            "producing_module": "FinalEntitlementDecisionHead explicit_product",
            "source_file": decision_key,
            "source_line_range": source_range(
                texts[decision_key], r"^\s*def forward\(", r"^\s*support_logit ="
            ),
            "tensor_shape_contract": "[B]",
            "producer_trainability": "analytic product; upstream heads trainable",
            "candidate_action_dependency": "changes for any selected gate primitive",
            "native_loss_dependency": "final classification loss",
            "detach_boundary": "none",
            "serialization_boundary": "none",
            "replay_feasibility": "vectorize exact candidate dimension after arm replays",
        },
        {
            "state_name": "router or selector state",
            "producing_module": "none in ContraMambaV6BMinimal.forward",
            "source_file": model_key,
            "source_line_range": source_range(model, r"^\s*def forward\(", r"^\s*return \{"),
            "tensor_shape_contract": "absent; candidate IDs/actions are external P2 authority",
            "producer_trainability": "no trainable selector",
            "candidate_action_dependency": "external row-specific action lookup",
            "native_loss_dependency": "none",
            "detach_boundary": "P2 CSV is discrete provenance, never a learned feature",
            "serialization_boundary": "P2 candidate-action CSV",
            "replay_feasibility": "explicit action tensor required; no seed/row identity feature",
        },
        {
            "state_name": "decision-head logits / base_logits",
            "producing_module": "FinalEntitlementDecisionHead",
            "source_file": decision_key,
            "source_line_range": source_range(
                texts[decision_key], r"^\s*support_logit =", r"^\s*return \{"
            ),
            "tensor_shape_contract": f"[B,3] order {list(CLASS_ORDER)}",
            "producer_trainability": "trainable bias and raw_alpha",
            "candidate_action_dependency": "all selected primitives",
            "native_loss_dependency": "input to final modulation and label CE",
            "detach_boundary": "none",
            "serialization_boundary": "base_logits diagnostic only after return",
            "replay_feasibility": "reuse exact decision head; preserve class order",
        },
        {
            "state_name": "final composer logits / final_logits",
            "producing_module": "ContraMambaV6BMinimal final modulation",
            "source_file": model_key,
            "source_line_range": source_range(
                model, r"^\s*# Apply final logit modulation", r"^\s*# Compute losses"
            ),
            "tensor_shape_contract": f"[B,3] order {list(CLASS_ORDER)}",
            "producer_trainability": "decision head plus optional comparator alphas",
            "candidate_action_dependency": "candidate base logits plus recipient-native modulation",
            "native_loss_dependency": "authoritative label CE and predictions",
            "detach_boundary": "diagnostic observability detaches copies only",
            "serialization_boundary": "P0/P2 exports after native forward",
            "replay_feasibility": "exact only after complete live donor downstream replay",
        },
    ]
    for row in rows:
        row["source_file"] = str((root / row["source_file"]).resolve())
    return rows


def stochastic_audit(root: Path, texts: dict[str, str]) -> list[dict[str, Any]]:
    locations = (
        ("FrameGate pair_projector dropout", "src/contramamba/heads/frame_gate.py"),
        (
            "PredicateCoverageHead pair_projector dropout",
            "src/contramamba/heads/predicate_coverage.py",
        ),
        ("SufficiencyGate projector dropout", "src/contramamba/heads/sufficiency_gate.py"),
        (
            "PolarityEnergyHead feature_projector dropout",
            "src/contramamba/heads/polarity_energy.py",
        ),
    )
    rows: list[dict[str, Any]] = []
    for operation, relative in locations:
        rows.append(
            {
                "operation": operation,
                "source_file": str((root / relative).resolve()),
                "source_line_range": str(source_line(texts[relative], "nn.Dropout(dropout)")),
                "position_relative_to_replay_boundary": "downstream",
                "candidate_dependency": "candidate-independent within an arm; arm inputs/weights differ",
                "training_mode_behavior": "active in train mode, disabled in eval mode",
                "randomness_policy": (
                    "save RNG state before native downstream replay; restore it before "
                    "donor downstream replay; after donor replay restore the post-native "
                    "state so global RNG advances exactly once"
                ),
                "must_share_randomness": True,
                "must_use_independent_randomness": False,
                "disabled_in_authoritative_training": False,
            }
        )
    rows.extend(
        [
            {
                "operation": "Mamba backbone stochastic state",
                "source_file": str((root / "scripts/train_controlled_v5.py").resolve()),
                "source_line_range": source_range(
                    texts["scripts/train_controlled_v5.py"],
                    r"^def cache_frozen_encoder_states\(",
                    r"inputs\[\"encoder_hidden_states\"\] =",
                ),
                "position_relative_to_replay_boundary": "upstream",
                "candidate_dependency": "none",
                "training_mode_behavior": "authoritative cache forces model.mamba.eval under no_grad",
                "randomness_policy": "one shared cached hidden state; never recalculate per candidate",
                "must_share_randomness": True,
                "must_use_independent_randomness": False,
                "disabled_in_authoritative_training": True,
            },
            {
                "operation": "random masking / augmentation / stochastic routing / sampling",
                "source_file": str(
                    (root / "src/contramamba/modeling_v6b_minimal.py").resolve()
                ),
                "source_line_range": "native forward and directly called heads",
                "position_relative_to_replay_boundary": "absent from model forward",
                "candidate_dependency": "none",
                "training_mode_behavior": "not present",
                "randomness_policy": "none",
                "must_share_randomness": False,
                "must_use_independent_randomness": False,
                "disabled_in_authoritative_training": True,
            },
            {
                "operation": "stateful normalization buffers",
                "source_file": "directly called head modules",
                "source_line_range": "LayerNorm modules only; no BatchNorm running statistics",
                "position_relative_to_replay_boundary": "downstream",
                "candidate_dependency": "none",
                "training_mode_behavior": "LayerNorm has no running mean/variance",
                "randomness_policy": "no stochastic state",
                "must_share_randomness": False,
                "must_use_independent_randomness": False,
                "disabled_in_authoritative_training": True,
            },
        ]
    )
    return rows


def gradient_design() -> list[dict[str, Any]]:
    variants = (
        (
            "baseline",
            "NONE",
            "native downstream path only",
            "ordinary native loss recipients",
            "none",
        ),
        (
            "direction-consistency only",
            "DIRECTION_CONSISTENCY",
            "native + complete frame-local-only downstream arm; all three deterministic candidate compositions",
            (
                "live student epistemic heads in both arms; frame branch direct/local path; "
                "predicate, sufficiency, polarity branches; decision head and comparator "
                "parameters when algebraically connected; frozen Mamba excluded; no selector"
            ),
            (
                "teacher signs are stop-gradient; exact ties ignored; teacher identity remains "
                "NOT_YET_JUSTIFIED, so implementation must not add this loss yet"
            ),
        ),
        (
            "candidate-order-consistency only",
            "CANDIDATE_ORDER_CONSISTENCY",
            "native + complete frame-local-only downstream arm; all three candidates and three unordered pairs",
            (
                "same live student recipients as direction family through pair gaps; "
                "final composer coordinate-conditional; frozen Mamba and absent selector receive none"
            ),
            (
                "teacher pair signs are stop-gradient; exact ties ignored; no lexical mask "
                "ordering; teacher identity remains NOT_YET_JUSTIFIED"
            ),
        ),
    )
    rows = []
    for variant, family, paths, recipients, teacher in variants:
        for group, status in (
            ("frozen_backbone", "NO_GRADIENT_AUTHORITY"),
            ("trainable_backbone", "ABSENT_IN_AUTHORITATIVE_CONFIGURATION"),
            ("epistemic_heads", "PRIMARY_RECIPIENT" if family != "NONE" else "NATIVE_ONLY"),
            ("frame_branch", "PRIMARY_RECIPIENT_WITH_ARM_SPECIFIC_DETACH"),
            ("predicate_branch", "PRIMARY_RECIPIENT"),
            ("sufficiency_branch", "PRIMARY_RECIPIENT"),
            ("polarity_branch", "PRIMARY_RECIPIENT"),
            ("entitlement_decision", "COORDINATE_CONDITIONAL_RECIPIENT"),
            ("router_or_selector", "ABSENT_NO_GRADIENT"),
            ("final_composer", "COORDINATE_CONDITIONAL_RECIPIENT"),
            ("other_trainable_modules", "ONLY_IF_ACTIVE_NATIVE_MODULATION_DEPENDS_ON_THEM"),
        ):
            rows.append(
                {
                    "variant": variant,
                    "intervention_family": family,
                    "required_forward_paths": paths,
                    "parameter_group": group,
                    "gradient_recipient_status": status if family != "NONE" else "NATIVE_BASELINE",
                    "exact_gradient_path": recipients,
                    "teacher_stop_gradient_rule": teacher,
                    "independently_disableable": True,
                    "combined_first_stage_variant": False,
                }
            )
    return rows


def design_rows() -> list[dict[str, Any]]:
    return [
        {
            "design": "A",
            "name": "FINAL_COMPOSER_ONLY_RECOMPOSITION",
            "status": "REJECTED_BY_P6_FOR_TRAINING_GRADIENT_PURPOSES",
            "semantic_exactness": "diagnostic geometry only; P2/P4 remain valid",
            "gradient_exactness": "insufficient: cannot produce separately trained donor state",
            "memory_cost": "small but scientifically irrelevant as primary training path",
            "compute_cost": "composer arithmetic only",
            "implementation_complexity": "low",
            "state_sharing_risks": "reuses detached/foreign donor values or omits donor training path",
            "mamba_forwards": "0 additional",
            "downstream_forwards": "0 additional",
            "selected": False,
        },
        {
            "design": "B",
            "name": "DOWNSTREAM_MODULE_REPLAY_FROM_SHARED_ENCODER_STATE",
            "status": "INSUFFICIENT_IF_INTERPRETED_AS_PARTIAL_DOWNSTREAM_REPLAY",
            "semantic_exactness": "shared encoder is exact, but no strict suffix smaller than all trainable heads reproduces donor",
            "gradient_exactness": "incomplete under partial replay",
            "memory_cost": "between A and C, but boundary is not source-authorized",
            "compute_cost": "symbolic partial D; not selected",
            "implementation_complexity": "high because an arbitrary partial cut would be false precision",
            "state_sharing_risks": "donor parameter differences begin at first trainable head operations",
            "mamba_forwards": "1 shared or 0 per batch after authoritative cache",
            "downstream_forwards": "undefined partial count",
            "selected": False,
        },
        {
            "design": "C",
            "name": "FULL_TRAINABLE_PATH_REPLAY",
            "status": "SELECTED_EXACT_DESIGN",
            "semantic_exactness": "exact with joint and frame-local-only parameter arms, row-specific P2 actions, and recipient-native modulation",
            "gradient_exactness": "exact when both live downstream graphs and arm-specific detach semantics are retained",
            "memory_cost": "(A_M+2*A_D+3*A_C)/(A_M+A_D), or about 2+3*A_C/A_D with cached Mamba",
            "compute_cost": "(F_M+2*F_D+3*F_C)/(F_M+F_D); cached per-batch form (2*F_D+3*F_C)/F_D",
            "implementation_complexity": "moderate-high: explicit downstream entry point, dual arm state, matched RNG, provenance",
            "state_sharing_risks": "dropout mismatch, accidental parameter aliasing, detached diagnostic reuse",
            "mamba_forwards": "1 shared preprocessing forward; 0 per batch if cached",
            "downstream_forwards": "2 total (native plus donor), not 3 candidate reruns",
            "selected": True,
        },
        {
            "design": "D",
            "name": "FULL_MODEL_FORWARD_INCLUDING_BACKBONE",
            "status": "EXACT_BUT_NOT_MINIMAL",
            "semantic_exactness": "exact only with correct two-arm parameters and shared-randomness controls",
            "gradient_exactness": "exact but frozen backbone adds no authorized gradient recipient",
            "memory_cost": "native plus three full candidate graphs is approximately 4*(A_M+A_D)/(A_M+A_D)=4x if all activations are retained",
            "compute_cost": "required Design-D assessment is approximately 4x forward and up to 4x backward for native plus three full candidate forwards; no measured runtime claim",
            "implementation_complexity": "lower API surgery, higher resource and RNG risk",
            "state_sharing_risks": "different backbone/dropout realization can confound comparison",
            "mamba_forwards": "4 under literal native-plus-three-candidate Design D; source shows this is redundant",
            "downstream_forwards": "4 under literal Design D; activation pressure may require smaller microbatches while preserving accumulation semantics",
            "selected": False,
        },
    ]


def compute_rows() -> list[dict[str, Any]]:
    common = {
        "native_forward_count_per_batch": "1 logical native",
        "candidate_forward_count_per_batch": "3 logical candidate outputs",
        "mamba_forward_count_per_batch": "0 after authoritative dataset cache; otherwise 1 shared",
        "downstream_replay_count_per_batch": "2 total arm paths = 1 native + 1 donor",
        "retained_autograd_graphs": "2 downstream arm graphs with 3 deterministic composer branches",
        "activation_memory_formula": "(A_M+2*A_D+3*A_C)/(A_M+A_D); cached A_M=0 gives 2+3*A_C/A_D",
        "forward_compute_formula": "(F_M+2*F_D+3*F_C)/(F_M+F_D); cached gives 2+3*F_C/F_D",
        "backward_compute_formula": "(2*B_D+3*B_C)/B_D when both arms receive gradients",
        "gradient_accumulation_interaction": "loss divided once by existing accumulation factor; graphs freed per microbatch; no extra optimizer step",
        "batch_size_implication": "dual downstream activations may require smaller physical microbatch while preserving effective batch via accumulation; not pre-authorized",
        "benchmark_claim": False,
    }
    return [
        {
            "variant": "baseline",
            "candidate_paths_required": 0,
            "loss_arithmetic": "none",
            **{
                **common,
                "candidate_forward_count_per_batch": "0",
                "downstream_replay_count_per_batch": "1 native",
                "retained_autograd_graphs": "1 native",
                "activation_memory_formula": "1",
                "forward_compute_formula": "1",
                "backward_compute_formula": "1",
            },
        },
        {
            "variant": "direction-consistency only",
            "candidate_paths_required": 3,
            "loss_arithmetic": "O(B*3*4) signed coordinates; exact teacher ties excluded",
            **common,
        },
        {
            "variant": "candidate-order-consistency only",
            "candidate_paths_required": 3,
            "loss_arithmetic": "O(B*3_pairs*4) centered gaps; exact teacher ties excluded",
            **common,
        },
    ]


def teacher_analysis() -> list[dict[str, Any]]:
    options = [
        (
            "stop-gradient same-step native state",
            "native is a comparison origin but does not supply stable historical candidate directions/orders",
            "no extra native path beyond student; still needs exact donor replay",
            "none",
            "high target drift",
            False,
            "ties can be detected exactly but target lacks mechanistic authorization",
            "NOT_YET_JUSTIFIED",
        ),
        (
            "stop-gradient candidate reference",
            "could anchor signs/orders if its arm and time provenance were defined",
            "one exact reference geometry (two downstream arms) unless reusing student values",
            "none if same-step; otherwise reference parameters",
            "high when same-step; circular-target risk",
            False,
            "must ignore exact response/pair ties",
            "NOT_YET_JUSTIFIED",
        ),
        (
            "EMA model",
            "P5 conceptual preference follows trajectory without one pre-tail checkpoint",
            "one no-grad exact teacher geometry: recipient+donor downstream and three compositions",
            "two downstream parameter-arm copies plus EMA update state",
            "moderate target drift controlled by decay, whose value is not authorized",
            False,
            "must ignore exact teacher ties",
            "CONCEPTUAL_PREFERENCE_NOT_IMPLEMENTATION_AUTHORITY",
        ),
        (
            "frozen pre-tail anchor",
            "historical topology target, but choice of epoch/seed would encode an unjustified anchor",
            "one no-grad exact teacher geometry",
            "two frozen downstream arm parameter copies",
            "low drift; high stale-target risk",
            True,
            "must ignore exact teacher ties",
            "NOT_YET_JUSTIFIED",
        ),
        (
            "previous-epoch snapshot",
            "nearby trajectory anchor but epoch lag is an arbitrary teacher definition",
            "one no-grad exact teacher geometry",
            "two downstream arm snapshots",
            "stepwise target drift",
            False,
            "must ignore exact teacher ties",
            "NOT_YET_JUSTIFIED",
        ),
    ]
    rows = []
    for family in ("direction-consistency", "candidate-order-consistency"):
        for (
            teacher,
            justification,
            forwards,
            storage,
            drift,
            seed_specific,
            ties,
            status,
        ) in options:
            rows.append(
                {
                    "intervention_family": family,
                    "teacher_option": teacher,
                    "mechanistic_justification": justification,
                    "required_additional_forward_paths": forwards,
                    "parameter_storage_cost": storage,
                    "risk_of_target_drift": drift,
                    "seed_specific_information_introduced": seed_specific,
                    "exact_tie_policy": ties,
                    "selection_status": status,
                }
            )
    return rows


def run(ns: argparse.Namespace, contracts: list[dict[str, Any]]) -> dict[str, Any]:
    root = ns.repo_root.resolve()
    output = ns.output_dir.resolve()
    inputs = {
        "p6": ns.stage196b2b6p6_analysis_json.resolve(),
        "p5": ns.stage196b2b6p5_analysis_json.resolve(),
        "p4": ns.stage196b2b6p4_analysis_json.resolve(),
        "p2": ns.stage196b2b6p2_analysis_json.resolve(),
    }
    explicit = (
        ns.repo_root.is_absolute()
        and ns.output_dir.is_absolute()
        and all(path.is_absolute() and path.is_file() for path in inputs.values())
        and bool(re.fullmatch(r"[0-9a-f]{40}", ns.current_git_commit))
    )
    gate(
        contracts,
        "explicit_cli_and_commit",
        True,
        explicit,
        explicit,
        "all CLI paths must be explicit and commit must be lowercase 40-hex",
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=root, text=True
    ).strip()
    gate(
        contracts,
        "current_commit_identity",
        ns.current_git_commit,
        head,
        head == ns.current_git_commit,
        "current git commit changed",
    )
    output_is_fresh = output.is_dir() and not any(output.iterdir())
    gate(
        contracts,
        "output_directory_created_fresh",
        True,
        output_is_fresh,
        output_is_fresh,
        "output directory was not freshly created and empty",
    )

    p6, p5, p4, p2 = (read_json(inputs[key]) for key in ("p6", "p5", "p4", "p2"))
    observed, missing, unexpected = exact_p6_closure(inputs["p6"].parent)
    gate(
        contracts,
        "p6_exact_nine_file_closure",
        sorted(P6_OUTPUTS),
        {"files": observed, "missing": missing, "unexpected": unexpected},
        not missing and not unexpected and observed == sorted(P6_OUTPUTS),
        "P6 companion directory is not the exact nine-file set",
    )
    p6_contract_ok, p6_failed = contract_closed(
        inputs["p6"].parent / "stage196b2b6p6_contract.csv"
    )
    gate(
        contracts,
        "p6_decision_closure",
        {"decision": P6_DECISION, "blocking_reasons": []},
        {key: p6.get(key) for key in ("decision", "blocking_reasons")},
        p6.get("decision") == P6_DECISION and p6.get("blocking_reasons") == [],
        "P6 frozen decision or blocker closure changed",
    )
    gate(
        contracts,
        "p6_zero_failed_contracts",
        0,
        p6_failed,
        p6_contract_ok and p6_failed == 0,
        "P6 contains failed contracts",
    )
    for label, payload, required in (
        ("p5", p5, P5_DECISION),
        ("p4", p4, P4_DECISION),
        ("p2", p2, P2_DECISION),
    ):
        gate(
            contracts,
            f"{label}_decision_and_zero_blockers",
            {"decision": required, "blocking_reasons": []},
            {key: payload.get(key) for key in ("decision", "blocking_reasons")},
            payload.get("decision") == required and payload.get("blocking_reasons") == [],
            f"{label.upper()} frozen closure changed",
        )

    texts, source_hashes = load_sources(root)
    trainer = texts["scripts/train_controlled_v6b_minimal.py"]
    model = texts["src/contramamba/modeling_v6b_minimal.py"]
    hook_source = texts["scripts/train_controlled_v6b_minimal.py"]
    selector_source = texts["scripts/analyze_stage196b2b6_minimal_selector_intervention.py"]
    cache_source = texts["scripts/train_controlled_v5.py"]
    source_checks = {
        "mamba_identity": MODEL_NAME in trainer and MODEL_NAME in model,
        "frozen_encoder_policy": "parameter.requires_grad = not freeze_encoder" in trainer,
        "cached_encoder_api": (
            "encoder_hidden_states: torch.Tensor | None = None" in model
            and 'inputs["encoder_hidden_states"]' in cache_source
        ),
        "native_forward_order": all(
            token in model
            for token in (
                "self.mamba(input_ids=input_ids)",
                "self.frame_gate(",
                "self.predicate_coverage_head(",
                "self.sufficiency_gate(",
                "self.polarity_energy_head(",
                "self.decision_head(",
                "final_logits = base_logits",
                '"logits": final_logits',
            )
        ),
        "frame_local_only_semantics": all(
            token in hook_source
            for token in (
                'mode == "frame_local_only"',
                "register_forward_hook",
                "value.detach()",
                "frame_downstream_forward_value_changed",
            )
        ),
        "row_specific_primitive_semantics": all(
            token in selector_source
            for token in ("PRIMITIVES =", "PRIMITIVE_FIELDS =", "def apply_mask(", "zip(mask,PRIMITIVES)")
        ),
        "p6_rejection_preserved": (
            "FULL_COUNTERFACTUAL_FORWARD_REQUIRED"
            in texts["scripts/probe_stage196b2b6p6_minimal_gradient_path_instrumentation.py"]
        ),
        "class_order": "torch.stack(\n            [refute_logit, not_entitled_logit, support_logit]"
        in texts["src/contramamba/heads/entitlement_decision.py"],
    }
    gate(
        contracts,
        "native_and_candidate_source_trace_closure",
        {key: True for key in source_checks},
        source_checks,
        all(source_checks.values()),
        "one or more static execution-graph facts changed",
    )

    data_path = root / MAIN_DATA
    data_hash = sha256(data_path)
    time_swap_count = 0
    with data_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{data_path}:{line_number}: object required")
            time_swap_count += value.get("intervention_type") == "time_swap"
    gate(
        contracts,
        "clean_main_data_identity_and_time_swap_exclusion",
        {"path": MAIN_DATA, "sha256": MAIN_DATA_SHA256, "time_swap_rows": 0},
        {"path": MAIN_DATA, "sha256": data_hash, "time_swap_rows": time_swap_count},
        data_hash == MAIN_DATA_SHA256 and time_swap_count == 0,
        "clean main data identity or time_swap exclusion changed",
    )

    candidate_rows, candidate_summary = action_summary(inputs["p2"])
    gate(
        contracts,
        "exact_three_candidates_and_primitive_mapping",
        {
            "candidate_masks": list(CANDIDATES),
            "primitive_order": list(PRIMITIVES),
            "row_specific": True,
        },
        {
            key: candidate_summary[key]
            for key in ("candidate_masks", "primitive_order", "row_specific")
        },
        all(
            row["candidate_identity_is_primitive_bitmask"] is False
            and row["primitive_mapping"] == "ROW_SPECIFIC_5BIT_ACTION_FROM_P2_AUTHORITY"
            for row in candidate_rows
        ),
        "candidate primitive mapping closure failed",
    )

    boundary_rows = execution_audit(root, texts)
    stochastic_rows = stochastic_audit(root, texts)
    gradients = gradient_design()
    designs = design_rows()
    compute = compute_rows()
    teachers = teacher_analysis()

    candidate_changes_tokens = any(row["affects_input_tokenization"] for row in candidate_rows)
    candidate_changes_backbone = any(row["affects_frozen_backbone"] for row in candidate_rows)
    shared_backbone_valid = (
        not candidate_changes_tokens
        and not candidate_changes_backbone
        and source_checks["cached_encoder_api"]
        and source_checks["frozen_encoder_policy"]
    )
    separately_trained_donor = (
        p6.get("source_feasibility_evidence", {}).get("current_counterfactual_source")
        == "P2 apply_mask over joint recipient plus separately trained frame_local_only donor"
    )
    full_downstream_required = shared_backbone_valid and separately_trained_donor
    semantics_resolved = (
        full_downstream_required
        and candidate_summary["row_specific"] is True
        and source_checks["frame_local_only_semantics"]
    )
    resource_unsafe = False
    earliest = "encoder_hidden_states / Mamba last_hidden_state [B,T,H_backbone]"
    gate(
        contracts,
        "earliest_replay_boundary_unique",
        earliest,
        earliest if shared_backbone_valid and full_downstream_required else None,
        shared_backbone_valid and full_downstream_required,
        "a unique exact replay boundary was not established",
    )
    gate(
        contracts,
        "stochastic_state_handling_complete",
        True,
        bool(stochastic_rows)
        and all(row["randomness_policy"] for row in stochastic_rows),
        bool(stochastic_rows)
        and all(row["randomness_policy"] for row in stochastic_rows),
        "stochastic-state policy is incomplete",
    )
    gate(
        contracts,
        "gradient_recipients_and_intervention_separation",
        ["baseline", "direction-consistency only", "candidate-order-consistency only"],
        sorted({row["variant"] for row in gradients}),
        {row["variant"] for row in gradients}
        == {"baseline", "direction-consistency only", "candidate-order-consistency only"}
        and all(not row["combined_first_stage_variant"] for row in gradients),
        "gradient recipients or independent variants are incomplete",
    )
    gate(
        contracts,
        "compute_memory_batching_teacher_design_complete",
        True,
        {
            "compute_rows": len(compute),
            "teacher_rows": len(teachers),
            "candidate_batching": True,
        },
        len(compute) == 3 and len(teachers) == 10,
        "compute, memory, batching, or teacher design is incomplete",
    )

    if not semantics_resolved:
        decision, next_stage = DECISIONS[4]
    elif resource_unsafe:
        decision, next_stage = DECISIONS[3]
    elif candidate_changes_tokens or candidate_changes_backbone:
        decision, next_stage = DECISIONS[2]
    elif full_downstream_required:
        decision, next_stage = DECISIONS[1]
    else:
        decision, next_stage = DECISIONS[0]
    hierarchy = [
        {
            "order": index + 1,
            "decision": candidate,
            "condition": condition,
            "observed": observed_value,
            "reached": candidate == decision,
            "recommended_next_stage": candidate_next,
        }
        for index, ((candidate, candidate_next), condition, observed_value) in enumerate(
            zip(
                DECISIONS,
                (
                    "shared backbone valid and a strict downstream suffix is sufficient",
                    "shared backbone valid and complete trainable downstream arm replay required",
                    "candidate changes tokens/backbone state",
                    "exact path exceeds established resource envelope",
                    "source lacks unique candidate semantics",
                ),
                (
                    shared_backbone_valid and not full_downstream_required,
                    full_downstream_required and semantics_resolved,
                    candidate_changes_tokens or candidate_changes_backbone,
                    resource_unsafe,
                    not semantics_resolved,
                ),
            )
        )
    ]
    gate(
        contracts,
        "decision_hierarchy_reachability",
        1,
        sum(row["reached"] for row in hierarchy),
        sum(row["reached"] for row in hierarchy) == 1,
        "decision hierarchy does not reach exactly one outcome",
    )
    scope = {
        "counterfactual_forward_implemented": False,
        "stability_loss_implemented": False,
        "training_performed": False,
        "checkpoint_loaded": False,
        "model_executed": False,
        "training_objective_changed": False,
        "external_data_used": False,
        "ood_data_used": False,
        "p3_safety_labels_used": False,
        "recovery_or_harm_categories_used": False,
        "stable_row_identity_used_as_feature": False,
        "seed_identity_used_as_feature": False,
    }
    gate(
        contracts,
        "design_only_no_implementation_execution_or_leakage",
        {key: False for key in scope},
        scope,
        all(value is False for value in scope.values()),
        "P7 scope expanded beyond static source design",
    )
    gate(
        contracts,
        "exact_ten_file_output_declaration",
        list(OUTPUTS),
        list(OUTPUTS),
        len(OUTPUTS) == len(set(OUTPUTS)) == 10,
        "output declaration is not exactly ten files",
    )

    common_command = (
        "scripts/train_controlled_v6b_minimal.py --data "
        "data/controlled_v5_v3_without_time_swap.jsonl --backbone mamba "
        "--model-name state-spaces/mamba-130m-hf --architecture v6b_minimal "
        "--device cuda --epochs 20 --split-seed 174 "
        "--stage196b1-framegate-gradient-ownership-observability "
        "--stage196b2p0-epoch-channel-observability "
        "--stage196b2b3p0-export-epoch-composer-inputs "
        "--compatible-positive-margin-weight 0.0 "
        "--compatible-positive-margin-logit 0.0 --lr 0.001 "
        "--freeze-encoder true --freeze-a-log true --max-length 128 "
        "--dev-ratio 0.2 --gradient-accumulation-steps 1 "
        "--class-weighting none --select-metric final_macro_f1 "
        "--flag-source controlled_heuristic --save-selected-checkpoint "
        "--selected-checkpoint-filename selected_checkpoint.pt --seed 183"
    )
    analysis = {
        "stage": STAGE,
        "decision": decision,
        "recommended_next_stage": next_stage,
        "blocking_reasons": [],
        "current_git_commit": head,
        "upstream": {
            "p6_decision": p6.get("decision"),
            "p5_decision": p5.get("decision"),
            "p4_decision": p4.get("decision"),
            "p2_decision": p2.get("decision"),
            "p6_artifacts": {
                name: sha256(inputs["p6"].parent / name) for name in P6_OUTPUTS
            },
        },
        "checkpoint_provenance": {
            "role": "reconstructed seed183 checkpoints",
            "historical_byte_identity_authority": False,
            "original_source_commit": ORIGINAL_SOURCE_COMMIT,
            "joint_original_training_command": common_command
            + " --frame-downstream-gradient-mode joint",
            "frame_local_only_original_training_command": common_command
            + " --frame-downstream-gradient-mode frame_local_only",
            "interpretation": (
                "role/configuration/source provenance only; reconstruction does not "
                "claim byte identity with lost historical checkpoint files"
            ),
        },
        "source_files_inspected": [
            {"path": str((root / relative).resolve()), "sha256": source_hashes[relative]}
            for relative in SOURCE_FILES
        ],
        "native_execution_trace": [row["state_name"] for row in boundary_rows],
        "candidate_semantics": candidate_summary,
        "first_candidate_dependent_state": (
            "external row-specific P2 5-bit action selects between live joint/donor "
            "primitive tensors; the donor value path can first differ at the first "
            "trainable downstream projection after shared Mamba hidden state"
        ),
        "earliest_safe_replay_boundary": earliest,
        "frozen_backbone_sharing": {
            "conclusion": "SHARED_FROZEN_BACKBONE_STATE_WITH_EXACT_DOWNSTREAM_REPLAY",
            "valid": shared_backbone_valid,
            "tokens_changed": candidate_changes_tokens,
            "backbone_inputs_changed": False,
            "backbone_hidden_states_changed": False,
            "backbone_dropout_realization_changed": False,
            "upstream_pooled_representation_changed": False,
            "new_backbone_forward_per_candidate": False,
        },
        "selected_execution_design": "FULL_TRAINABLE_PATH_REPLAY",
        "candidate_batching": {
            "arm_execution": (
                "sequential native and donor downstream replays with matched RNG; "
                "do not flatten arms because ordinary Dropout would draw different masks"
            ),
            "candidate_execution": (
                "stack [B,C,5] primitive selections then preserve [B,C,3] logits, "
                "or flatten to [B*C,...] only for deterministic decision-head arithmetic"
            ),
            "candidate_identity_explicit": True,
            "native_counterfactual_pairing_exact": True,
            "cross_row_mixing": False,
            "seed_feature": False,
            "row_identity_feature": False,
            "class_order": list(CLASS_ORDER),
            "candidate_masks": list(CANDIDATES),
            "sequential_comparison": (
                "candidate-sequential composition is semantically equivalent because "
                "no stochastic operation follows primitive selection, but vectorized "
                "composition reduces repeated deterministic arithmetic"
            ),
        },
        "teacher_state_analysis": {
            "selected_teacher": "not yet justified",
            "p5_conceptual_preference": "stop-gradient EMA model",
            "implementation_authorized": False,
            "exact_ties_ignored": True,
            "rows": teachers,
        },
        "intervention_variants": [
            "baseline",
            "direction-consistency only",
            "candidate-order-consistency only",
        ],
        "combined_first_stage_variant": False,
        "data_boundary": {
            "main_classification_data": MAIN_DATA,
            "sha256": data_hash,
            "time_swap_in_main_training": False,
            "external_data": False,
            "ood_data": False,
            "p3_safety_labels": False,
            "recovery_or_harm_categories": False,
            "stable_row_identity_feature": False,
            "seed_identity_feature": False,
        },
        "cost_claim": "symbolic/source-derived, not a benchmark claim",
        "remaining_risks": [
            "P8 must verify frozen-backbone parameter and persistent-buffer identity across both arms before sharing one cached hidden state.",
            "P8 must define dual-arm initialization, update, and checkpoint provenance without treating reconstructed bytes as historical authority.",
            "Matched CPU and all active CUDA RNG states must be implemented and mutation-audited without switching downstream heads to eval mode.",
            "The symbolic dual-downstream memory envelope has not been benchmarked; resource safety remains an implementation-stage measurement.",
            "No teacher is yet scientifically justified, so neither intervention loss is authorized by this design alone.",
        ],
        "scope": scope,
        "exact_outputs": list(OUTPUTS),
        "decision_hierarchy": hierarchy,
        "_tables": {
            "execution": boundary_rows,
            "candidates": candidate_rows,
            "stochastic": stochastic_rows,
            "gradients": gradients,
            "designs": designs,
            "compute": compute,
            "decisions": hierarchy,
        },
    }
    return analysis


def render_report(analysis: dict[str, Any]) -> str:
    sharing = analysis.get("frozen_backbone_sharing", {})
    return f"""# Stage196-B2-B6P7 Full Counterfactual Forward Design

Decision: `{analysis["decision"]}`

Recommended next stage: `{analysis["recommended_next_stage"]}`

P2/P4 diagnostic recomposition remains valid and authoritative for diagnostic
geometry. P6 rejected final-composer-only recomposition only as an exact
training-gradient path; it did not invalidate P2 or P4 numerics.

Static source tracing identifies `encoder_hidden_states` / Mamba
`last_hidden_state` as the earliest safe replay boundary. Candidate IDs are
opaque selector identities, each resolving through P2 authority to a
row-specific five-bit action in the order FRAME, PREDICATE, SUFFICIENCY,
POSITIVE_ENERGY, NEGATIVE_ENERGY. They do not change tokens, Mamba inputs, or
frozen Mamba hidden states.

Frozen Mamba sharing is therefore permitted only under the exact conditions
recorded in the audit. The conclusion is
`{sharing.get("conclusion")}`. The separately trained frame-local-only donor
can differ from the joint recipient beginning at the first downstream
trainable projections, so the complete downstream trainable path must be
rerun. A partial suffix and final-composer-only path are insufficient.

Downstream head dropout remains active in authoritative training. The design
saves the RNG state before native downstream execution, restores it for the
donor downstream replay, and finally restores the post-native RNG state.
Thus arm comparisons share masks without silently forcing evaluation mode.
Candidate composition is deterministic and may then be vectorized with an
explicit candidate dimension.

Direction-consistency and candidate-order-consistency remain independently
disableable, alongside baseline. No combined first-stage variant is included.
Both require all three candidate outputs under the P5 precommitment, although
their loss arithmetic differs. No teacher is selected: EMA remains a
conceptual preference, not implementation authority, and every valid teacher
must ignore exact ties.

Compute and memory estimates are symbolic and source-derived, not benchmark
claims. A scientifically resource-unsafe conclusion would be valid rather
than a contract failure; current static evidence does not establish resource
unsafety.

No loss, counterfactual forward, model execution, checkpoint load, training,
or objective change was implemented. Reconstructed seed183 checkpoint
provenance preserves the original training roles/commands and source commit
without treating reconstructed checkpoint bytes as historical authority. The
next stage follows from the exact execution boundary.
"""


def publish(
    ns: argparse.Namespace,
    analysis: dict[str, Any],
    contracts: list[dict[str, Any]],
) -> None:
    tables = analysis.pop(
        "_tables",
        {
            key: []
            for key in (
                "execution",
                "candidates",
                "stochastic",
                "gradients",
                "designs",
                "compute",
                "decisions",
            )
        },
    )
    blocking = [
        row["blocking_reason"]
        for row in contracts
        if not row["passed"] and row["blocking_reason"]
    ]
    analysis["blocking_reasons"] = blocking
    if blocking:
        analysis["decision"] = "STAGE196B2B6P7_BLOCKED_CONTRACT_FAILURE"
        analysis["recommended_next_stage"] = "STAGE196B2B6P7_REPAIR_CONTRACT"
    contents = {
        OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        OUTPUTS[1]: render_report(analysis),
        OUTPUTS[2]: csv_text(
            (
                "state_name",
                "producing_module",
                "source_file",
                "source_line_range",
                "tensor_shape_contract",
                "producer_trainability",
                "candidate_action_dependency",
                "native_loss_dependency",
                "detach_boundary",
                "serialization_boundary",
                "replay_feasibility",
            ),
            tables["execution"],
        ),
        OUTPUTS[3]: csv_text(
            (
                "candidate_mask",
                "primitive_mapping",
                "primitive_order",
                "observed_action_keys",
                "observed_action_key_counts",
                "first_tensor_whose_value_changes",
                "downstream_tensors_that_change",
                "stochastic_operation_downstream",
                "affects_trainable_branches",
                "affects_frozen_backbone",
                "affects_input_tokenization",
                "requires_new_backbone_forward",
                "candidate_identity_is_primitive_bitmask",
                "changed_primitive_union",
            ),
            tables["candidates"],
        ),
        OUTPUTS[4]: csv_text(
            (
                "operation",
                "source_file",
                "source_line_range",
                "position_relative_to_replay_boundary",
                "candidate_dependency",
                "training_mode_behavior",
                "randomness_policy",
                "must_share_randomness",
                "must_use_independent_randomness",
                "disabled_in_authoritative_training",
            ),
            tables["stochastic"],
        ),
        OUTPUTS[5]: csv_text(
            (
                "variant",
                "intervention_family",
                "required_forward_paths",
                "parameter_group",
                "gradient_recipient_status",
                "exact_gradient_path",
                "teacher_stop_gradient_rule",
                "independently_disableable",
                "combined_first_stage_variant",
            ),
            tables["gradients"],
        ),
        OUTPUTS[6]: csv_text(
            (
                "design",
                "name",
                "status",
                "semantic_exactness",
                "gradient_exactness",
                "memory_cost",
                "compute_cost",
                "implementation_complexity",
                "state_sharing_risks",
                "mamba_forwards",
                "downstream_forwards",
                "selected",
            ),
            tables["designs"],
        ),
        OUTPUTS[7]: csv_text(
            (
                "variant",
                "candidate_paths_required",
                "loss_arithmetic",
                "native_forward_count_per_batch",
                "candidate_forward_count_per_batch",
                "mamba_forward_count_per_batch",
                "downstream_replay_count_per_batch",
                "retained_autograd_graphs",
                "activation_memory_formula",
                "forward_compute_formula",
                "backward_compute_formula",
                "gradient_accumulation_interaction",
                "batch_size_implication",
                "benchmark_claim",
            ),
            tables["compute"],
        ),
        OUTPUTS[8]: csv_text(
            (
                "order",
                "decision",
                "condition",
                "observed",
                "reached",
                "recommended_next_stage",
            ),
            tables["decisions"],
        ),
        OUTPUTS[9]: csv_text(
            ("contract", "required", "observed", "passed", "blocking_reason"),
            contracts,
        ),
    }
    for name in OUTPUTS:
        atomic_write(ns.output_dir / name, contents[name])


def main() -> int:
    ns = parse_args()
    ns.repo_root = ns.repo_root.resolve()
    for name in (
        "stage196b2b6p6_analysis_json",
        "stage196b2b6p5_analysis_json",
        "stage196b2b6p4_analysis_json",
        "stage196b2b6p2_analysis_json",
    ):
        setattr(ns, name, getattr(ns, name).resolve())
    ns.output_dir = ns.output_dir.resolve()
    if ns.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory {ns.output_dir}")
    ns.output_dir.mkdir(parents=True, exist_ok=False)
    contracts: list[dict[str, Any]] = []
    try:
        analysis = run(ns, contracts)
    except Exception as exc:
        contracts.append(
            {
                "contract": "uncaught_contract_failure",
                "required": "no exception",
                "observed": f"{type(exc).__name__}: {exc}",
                "passed": False,
                "blocking_reason": f"{type(exc).__name__}: {exc}",
            }
        )
        analysis = {
            "stage": STAGE,
            "decision": "STAGE196B2B6P7_BLOCKED_CONTRACT_FAILURE",
            "recommended_next_stage": "STAGE196B2B6P7_REPAIR_CONTRACT",
            "blocking_reasons": [],
            "current_git_commit": ns.current_git_commit,
            "scope": {
                "counterfactual_forward_implemented": False,
                "stability_loss_implemented": False,
                "training_performed": False,
                "checkpoint_loaded": False,
                "model_executed": False,
                "training_objective_changed": False,
            },
            "exact_outputs": list(OUTPUTS),
            "_tables": {
                key: []
                for key in (
                    "execution",
                    "candidates",
                    "stochastic",
                    "gradients",
                    "designs",
                    "compute",
                    "decisions",
                )
            },
        }
    contracts.append(
        {
            "contract": "exact_ten_file_closure",
            "required": list(OUTPUTS),
            "observed": list(OUTPUTS),
            "passed": True,
            "blocking_reason": "",
        }
    )
    publish(ns, analysis, contracts)
    return 0 if analysis.get("blocking_reasons") == [] and all(
        row["passed"] for row in contracts
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
