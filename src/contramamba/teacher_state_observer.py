"""Stage196-B2-B6P9-P2 teacher-state observability instrumentation.

Trainer-owned, default-off observer for previous-step, previous-epoch, and EMA
teacher state snapshots. This module is observational only: it never creates a
loss tensor, never calls backward, and never participates in optimizer state.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


SCHEMA_VERSION = "stage196b2b6p9p2_teacher_observer_state_v1"
SIDECAR_FILENAMES = (
    "teacher_observer_manifest.json",
    "teacher_observer_batch_metrics.jsonl",
    "teacher_observer_epoch_metrics.csv",
    "teacher_observer_run_summary.json",
    "teacher_observer_state_audit.json",
)
MODES = ("off", "previous_step", "previous_epoch", "ema")
TARGET_FAMILIES = ("none", "direction", "candidate_order")
CANDIDATE_MASKS = (
    "00100000000000",
    "01000000000000",
    "10000000000000",
)
CANDIDATE_ACTIONS = {
    "00100000000000": (False, False, True, False, False),
    "01000000000000": (False, True, False, False, False),
    "10000000000000": (True, False, False, False, False),
}
DIRECTION_KEYS = (
    "delta_score_support",
    "delta_score_not_entitled",
    "delta_score_refute",
    "delta_support_minus_not_entitled",
    "delta_support_minus_refute",
    "delta_refute_minus_not_entitled",
    "delta_top1_runner_up_margin",
)
UNAVAILABLE_NO_LOSS_OR_BACKWARD = {
    "available": False,
    "reason": "NO_LOSS_OR_BACKWARD_IN_OBSERVATIONAL_STAGE",
}

EPOCH_IDENTITY_FIELDS = {
    "epoch",
    "mode",
    "target_family",
    "run_name",
    "schema_version",
    "student_quantity_source",
}
EPOCH_ADDITIVE_COUNT_FIELDS = {
    "direction_teacher_total_targets",
    "direction_teacher_exact_tie_targets",
    "direction_teacher_positive_sign_targets",
    "direction_teacher_negative_sign_targets",
    "direction_student_teacher_sign_agreement_count",
    "direction_student_teacher_sign_disagreement_count",
    "direction_nonzero_loss_target_count",
    "direction_nonzero_gradient_target_count",
    "order_teacher_total_pairs",
    "order_teacher_exact_tie_pairs",
    "order_teacher_positive_pair_targets",
    "order_teacher_negative_pair_targets",
    "order_student_teacher_pair_agreement_count",
    "order_student_teacher_pair_disagreement_count",
    "order_nonzero_loss_pair_count",
    "order_nonzero_gradient_pair_count",
}
EPOCH_CUMULATIVE_COUNTER_FIELDS = {
    "teacher_state_update_count",
    "teacher_state_read_count",
    "student_forward_count",
    "teacher_forward_count",
    "successful_optimizer_step_count",
    "skipped_optimizer_step_count",
}
EPOCH_STATE_MEASUREMENT_FIELDS = {
    "teacher_state_parameter_count",
    "teacher_state_buffer_count",
    "teacher_state_bytes",
    "student_teacher_parameter_l2",
    "student_teacher_parameter_relative_l2",
    "student_teacher_buffer_mismatch_count",
    "student_teacher_exact_parameter_match_rate",
    "teacher_state_initialized",
    "teacher_state_serialized",
    "teacher_state_restored",
    "teacher_state_missing_on_resume",
    "teacher_stop_gradient",
    "teacher_eval_mode",
    "student_mode_restored",
    "rng_state_preserved",
}
EPOCH_RATE_FIELDS = {
    "direction_student_teacher_sign_agreement_rate",
    "direction_student_teacher_sign_flip_rate",
    "order_student_teacher_pair_agreement_rate",
    "order_student_teacher_pair_flip_rate",
}


def validate_teacher_observer_cli(
    mode: str, target_family: str, ema_decay: float | None
) -> None:
    if mode not in MODES:
        raise ValueError(f"--teacher-observer-mode must be one of {MODES}")
    if target_family not in TARGET_FAMILIES:
        raise ValueError(
            f"--teacher-observer-target-family must be one of {TARGET_FAMILIES}"
        )
    if mode == "off":
        if target_family != "none":
            raise ValueError(
                "mode=off requires --teacher-observer-target-family none"
            )
        if ema_decay is not None:
            raise ValueError(
                "--teacher-observer-ema-decay is forbidden when mode is off"
            )
        return
    if target_family not in {"direction", "candidate_order"}:
        raise ValueError(
            "enabled teacher observer mode requires target_family in "
            "{direction,candidate_order}"
        )
    if mode == "ema":
        if ema_decay is None:
            raise ValueError("--teacher-observer-ema-decay is required for mode=ema")
        if not 0.0 < float(ema_decay) < 1.0:
            raise ValueError("--teacher-observer-ema-decay must satisfy 0 < decay < 1")
    elif ema_decay is not None:
        raise ValueError(
            "--teacher-observer-ema-decay is forbidden for non-EMA observer modes"
        )


def teacher_observer_enabled(args: Any) -> bool:
    return getattr(args, "teacher_observer_mode", "off") != "off"


def build_teacher_observer(
    *,
    args: Any,
    student_model: nn.Module,
    output_dir: Path,
    run_name: str,
    seed: int,
    device: torch.device,
    feature_input_fn: Any | None = None,
    autocast_fn: Any | None = None,
) -> "TeacherStateObserver | None":
    mode = getattr(args, "teacher_observer_mode", "off")
    target_family = getattr(args, "teacher_observer_target_family", "none")
    ema_decay = getattr(args, "teacher_observer_ema_decay", None)
    validate_teacher_observer_cli(mode, target_family, ema_decay)
    if mode == "off":
        return None
    observer_dir = Path(
        getattr(args, "teacher_observer_output_dir", None)
        or (output_dir / "teacher_observer")
    )
    if run_name != "single":
        observer_dir = observer_dir / run_name
    return TeacherStateObserver(
        mode=mode,
        target_family=target_family,
        ema_decay=ema_decay,
        student_model=student_model,
        output_dir=observer_dir,
        run_name=run_name,
        seed=seed,
        device=device,
        feature_input_fn=feature_input_fn,
        autocast_fn=autocast_fn,
    )


def _capture_rng_state() -> dict[str, Any]:
    return {
        "cpu_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_states": tuple(
            state.clone() for state in torch.cuda.get_rng_state_all()
        ) if torch.cuda.is_available() else (),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    cpu_state = state.get("cpu_rng_state")
    cuda_states = state.get("cuda_rng_states")
    if not isinstance(cpu_state, torch.Tensor):
        raise RuntimeError("teacher observer CPU RNG state is missing")
    if not isinstance(cuda_states, tuple):
        raise RuntimeError("teacher observer CUDA RNG states are invalid")
    torch.set_rng_state(cpu_state)
    if cuda_states:
        torch.cuda.set_rng_state_all(list(cuda_states))


def _clone_cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in sorted(module.state_dict().items())
    }


def _state_bytes(state: dict[str, torch.Tensor]) -> int:
    return int(sum(value.numel() * value.element_size() for value in state.values()))


def _action_tensors(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor(actions, dtype=torch.bool, device=device).view(1, -1).expand(
            batch_size, -1
        )
        for key, actions in CANDIDATE_ACTIONS.items()
    }


def _action_keys(batch_size: int) -> dict[str, tuple[str, ...]]:
    return {
        key: tuple("".join("1" if bit else "0" for bit in actions) for _ in range(batch_size))
        for key, actions in CANDIDATE_ACTIONS.items()
    }


def _count_parameters(module: nn.Module) -> int:
    return int(sum(1 for _ in module.parameters()))


def _count_buffers(module: nn.Module) -> int:
    return int(sum(1 for _ in module.buffers()))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


class TeacherStateObserver:
    """Single-teacher observational state owner."""

    def __init__(
        self,
        *,
        mode: str,
        target_family: str,
        ema_decay: float | None,
        student_model: nn.Module,
        output_dir: Path,
        run_name: str,
        seed: int,
        device: torch.device,
        feature_input_fn: Any | None = None,
        autocast_fn: Any | None = None,
    ) -> None:
        validate_teacher_observer_cli(mode, target_family, ema_decay)
        self.mode = mode
        self.target_family = target_family
        self.ema_decay = None if ema_decay is None else float(ema_decay)
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.seed = int(seed)
        self.device = device
        self.feature_input_fn = feature_input_fn
        self.autocast_fn = autocast_fn
        self.teacher = copy.deepcopy(student_model).to(device)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher_state_initialized = True
        self.teacher_state_serialized = False
        self.teacher_state_restored = False
        self.teacher_state_missing_on_resume = False
        self.successful_step_count = 0
        self.skipped_step_count = 0
        self.read_count = 0
        self.update_count = 0
        self.student_forward_count = 0
        self.teacher_forward_count = 0
        self.epoch_boundary_metadata: dict[str, Any] = {
            "boundary": "initial_student",
            "current_epoch": 0,
            "in_epoch_progress": False,
            "mid_epoch_resume_state": "not_resumed",
        }
        self.batch_metrics: list[dict[str, Any]] = []
        self.epoch_totals: dict[int, dict[str, Any]] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_name": self.run_name,
            "seed": self.seed,
            "mode": self.mode,
            "target_family": self.target_family,
            "ema_decay": self.ema_decay,
            "candidate_masks": list(CANDIDATE_MASKS),
            "candidate_action_authority": "P8/P9-P0 exact opaque identities; no lexical sorting",
            "observational_only": True,
            "loss_implemented": False,
            "total_loss_changed": False,
            "backward_changed": False,
            "sidecars": list(SIDECAR_FILENAMES),
        }
        self._write_json(self.output_dir / "teacher_observer_manifest.json", manifest)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")

    def _forward_with_replay(
        self,
        module: nn.Module,
        inputs: dict[str, torch.Tensor],
        *,
        temporal_mismatch_flags: torch.Tensor,
        predicate_mismatch_flags: torch.Tensor,
        temporal_adapter_final_penalty_scale: float,
        temporal_channel_gated_penalty_scale: float,
        amp_enabled: bool,
    ) -> dict[str, Any]:
        if self.feature_input_fn is None or self.autocast_fn is None:
            raise RuntimeError("teacher observer requires trainer forward helper hooks")
        with self.autocast_fn(amp_enabled):
            return module(
                **self.feature_input_fn(inputs),
                temporal_mismatch_flags=temporal_mismatch_flags,
                predicate_mismatch_flags=predicate_mismatch_flags,
                temporal_adapter_final_penalty_scale=temporal_adapter_final_penalty_scale,
                temporal_channel_gated_penalty_scale=temporal_channel_gated_penalty_scale,
                stage196b2b6p8_return_replay_state=True,
            )

    def _geometry(
        self,
        module: nn.Module,
        counterpart: nn.Module,
        output: dict[str, Any],
    ) -> dict[str, Any]:
        replay_state = output.get("stage196b2b6p8_replay_state")
        stochastic_context = output.get("stage196b2b6p8_stochastic_context")
        if not isinstance(replay_state, dict) or not isinstance(stochastic_context, dict):
            raise RuntimeError("teacher observer requires exact P8 replay state")
        native_logits = output.get("logits")
        if not isinstance(native_logits, torch.Tensor):
            raise RuntimeError("teacher observer requires output['logits']")
        batch_size = int(native_logits.shape[0])
        return module.replay_full_trainable_path(
            replay_state,
            _action_tensors(batch_size, native_logits.device),
            gradient_ownership_mode="joint",
            stochastic_context=stochastic_context,
            native_output=output,
            counterpart_model=counterpart,
            candidate_action_keys=_action_keys(batch_size),
        )

    def _extract_student_geometry(
        self,
        *,
        student_model: nn.Module,
        student_output: dict[str, Any],
        inputs: dict[str, torch.Tensor],
        temporal_mismatch_flags: torch.Tensor,
        predicate_mismatch_flags: torch.Tensor,
        temporal_adapter_final_penalty_scale: float,
        temporal_channel_gated_penalty_scale: float,
        amp_enabled: bool,
    ) -> tuple[dict[str, Any], str]:
        if isinstance(student_output.get("stage196b2b6p8_replay_state"), dict):
            before_rng = _capture_rng_state()
            try:
                with torch.no_grad():
                    geometry = self._geometry(student_model, self.teacher, student_output)
            finally:
                _restore_rng_state(before_rng)
            self.student_forward_count += 1
            return geometry, "LIVE_FORWARD_REUSE"
        before_rng = _capture_rng_state()
        student_was_training = student_model.training
        try:
            student_model.eval()
            with torch.no_grad():
                replay_output = self._forward_with_replay(
                    student_model,
                    inputs,
                    temporal_mismatch_flags=temporal_mismatch_flags,
                    predicate_mismatch_flags=predicate_mismatch_flags,
                    temporal_adapter_final_penalty_scale=temporal_adapter_final_penalty_scale,
                    temporal_channel_gated_penalty_scale=temporal_channel_gated_penalty_scale,
                    amp_enabled=amp_enabled,
                )
                self.student_forward_count += 1
                geometry = self._geometry(student_model, self.teacher, replay_output)
        finally:
            student_model.train(student_was_training)
            _restore_rng_state(before_rng)
        return geometry, "OBSERVATIONAL_STUDENT_REPLAY"

    def _teacher_geometry(
        self,
        *,
        student_model: nn.Module,
        inputs: dict[str, torch.Tensor],
        temporal_mismatch_flags: torch.Tensor,
        predicate_mismatch_flags: torch.Tensor,
        temporal_adapter_final_penalty_scale: float,
        temporal_channel_gated_penalty_scale: float,
        amp_enabled: bool,
    ) -> dict[str, Any]:
        before_rng = _capture_rng_state()
        student_was_training = student_model.training
        teacher_was_training = self.teacher.training
        try:
            student_model.eval()
            self.teacher.eval()
            with torch.no_grad():
                teacher_output = self._forward_with_replay(
                    self.teacher,
                    inputs,
                    temporal_mismatch_flags=temporal_mismatch_flags,
                    predicate_mismatch_flags=predicate_mismatch_flags,
                    temporal_adapter_final_penalty_scale=temporal_adapter_final_penalty_scale,
                    temporal_channel_gated_penalty_scale=temporal_channel_gated_penalty_scale,
                    amp_enabled=amp_enabled,
                )
                self.teacher_forward_count += 1
                return self._geometry(self.teacher, student_model, teacher_output)
        finally:
            self.teacher.train(teacher_was_training)
            self.teacher.eval()
            student_model.train(student_was_training)
            _restore_rng_state(before_rng)

    def observe_batch(
        self,
        *,
        epoch: int,
        batch_index: int,
        student_model: nn.Module,
        student_output: dict[str, Any],
        inputs: dict[str, torch.Tensor],
        temporal_mismatch_flags: torch.Tensor,
        predicate_mismatch_flags: torch.Tensor,
        temporal_adapter_final_penalty_scale: float,
        temporal_channel_gated_penalty_scale: float,
        amp_enabled: bool,
    ) -> dict[str, Any]:
        self.read_count += 1
        teacher_geometry = self._teacher_geometry(
            student_model=student_model,
            inputs=inputs,
            temporal_mismatch_flags=temporal_mismatch_flags,
            predicate_mismatch_flags=predicate_mismatch_flags,
            temporal_adapter_final_penalty_scale=temporal_adapter_final_penalty_scale,
            temporal_channel_gated_penalty_scale=temporal_channel_gated_penalty_scale,
            amp_enabled=amp_enabled,
        )
        student_geometry, quantity_source = self._extract_student_geometry(
            student_model=student_model,
            student_output=student_output,
            inputs=inputs,
            temporal_mismatch_flags=temporal_mismatch_flags,
            predicate_mismatch_flags=predicate_mismatch_flags,
            temporal_adapter_final_penalty_scale=temporal_adapter_final_penalty_scale,
            temporal_channel_gated_penalty_scale=temporal_channel_gated_penalty_scale,
            amp_enabled=amp_enabled,
        )
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "run_name": self.run_name,
            "mode": self.mode,
            "target_family": self.target_family,
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "student_quantity_source": quantity_source,
            "teacher_stop_gradient": True,
            "teacher_eval_mode": True,
            "student_mode_restored": True,
            "rng_state_preserved": True,
            "teacher_state_initialized": self.teacher_state_initialized,
            "teacher_state_update_count": self.update_count,
            "teacher_state_read_count": self.read_count,
            "teacher_state_serialized": self.teacher_state_serialized,
            "teacher_state_restored": self.teacher_state_restored,
            "teacher_state_missing_on_resume": self.teacher_state_missing_on_resume,
            "student_forward_count": self.student_forward_count,
            "teacher_forward_count": self.teacher_forward_count,
            "successful_optimizer_step_count": self.successful_step_count,
            "skipped_optimizer_step_count": self.skipped_step_count,
            **self.state_distance_metrics(student_model),
        }
        if self.target_family == "direction":
            metrics.update(self._direction_metrics(student_geometry, teacher_geometry))
            metrics["direction_nonzero_loss_target_count"] = UNAVAILABLE_NO_LOSS_OR_BACKWARD
            metrics["direction_nonzero_gradient_target_count"] = UNAVAILABLE_NO_LOSS_OR_BACKWARD
        elif self.target_family == "candidate_order":
            metrics.update(self._order_metrics(student_geometry, teacher_geometry))
            metrics["order_nonzero_loss_pair_count"] = UNAVAILABLE_NO_LOSS_OR_BACKWARD
            metrics["order_nonzero_gradient_pair_count"] = UNAVAILABLE_NO_LOSS_OR_BACKWARD
        self._validate_batch_metric_closure(metrics)
        self.batch_metrics.append(metrics)
        self._accumulate_epoch(epoch, metrics)
        self._append_jsonl(
            self.output_dir / "teacher_observer_batch_metrics.jsonl", metrics
        )
        return metrics

    def _direction_metrics(
        self, student_geometry: dict[str, Any], teacher_geometry: dict[str, Any]
    ) -> dict[str, Any]:
        total = ties = pos = neg = agree = disagree = 0
        for candidate in CANDIDATE_MASKS:
            s_row = student_geometry["candidate_geometry"][candidate]
            t_row = teacher_geometry["candidate_geometry"][candidate]
            for key in DIRECTION_KEYS:
                student_delta = s_row[key].detach()
                teacher_delta = t_row[key].detach()
                teacher_sign = torch.sign(teacher_delta)
                student_sign = torch.sign(student_delta)
                teacher_active = teacher_sign != 0
                comparable_active = teacher_active & (student_sign != 0)
                ties += int((teacher_sign == 0).sum().item())
                pos += int((teacher_sign > 0).sum().item())
                neg += int((teacher_sign < 0).sum().item())
                total += int(teacher_sign.numel())
                agree += int((student_sign[comparable_active] == teacher_sign[comparable_active]).sum().item())
                disagree += int((student_sign[comparable_active] != teacher_sign[comparable_active]).sum().item())
        comparable_total = agree + disagree
        return {
            "direction_teacher_total_targets": total,
            "direction_teacher_exact_tie_targets": ties,
            "direction_teacher_positive_sign_targets": pos,
            "direction_teacher_negative_sign_targets": neg,
            "direction_student_teacher_sign_agreement_count": agree,
            "direction_student_teacher_sign_disagreement_count": disagree,
            "direction_student_teacher_sign_agreement_rate": (
                agree / comparable_total if comparable_total else None
            ),
            "direction_student_teacher_sign_flip_rate": (
                disagree / comparable_total if comparable_total else None
            ),
        }

    def _order_metrics(
        self, student_geometry: dict[str, Any], teacher_geometry: dict[str, Any]
    ) -> dict[str, Any]:
        total = ties = pos = neg = agree = disagree = 0
        pairs = ((0, 1), (0, 2), (1, 2))
        for key in DIRECTION_KEYS:
            s_values = [
                student_geometry["candidate_geometry"][candidate][key].detach()
                for candidate in CANDIDATE_MASKS
            ]
            t_values = [
                teacher_geometry["candidate_geometry"][candidate][key].detach()
                for candidate in CANDIDATE_MASKS
            ]
            for left, right in pairs:
                teacher_gap = t_values[left] - t_values[right]
                student_gap = s_values[left] - s_values[right]
                teacher_sign = torch.sign(teacher_gap)
                student_sign = torch.sign(student_gap)
                teacher_active = teacher_sign != 0
                comparable_active = teacher_active & (student_sign != 0)
                ties += int((teacher_sign == 0).sum().item())
                pos += int((teacher_sign > 0).sum().item())
                neg += int((teacher_sign < 0).sum().item())
                total += int(teacher_sign.numel())
                agree += int((student_sign[comparable_active] == teacher_sign[comparable_active]).sum().item())
                disagree += int((student_sign[comparable_active] != teacher_sign[comparable_active]).sum().item())
        comparable_total = agree + disagree
        return {
            "order_teacher_total_pairs": total,
            "order_teacher_exact_tie_pairs": ties,
            "order_teacher_positive_pair_targets": pos,
            "order_teacher_negative_pair_targets": neg,
            "order_student_teacher_pair_agreement_count": agree,
            "order_student_teacher_pair_disagreement_count": disagree,
            "order_student_teacher_pair_agreement_rate": agree / comparable_total if comparable_total else None,
            "order_student_teacher_pair_flip_rate": disagree / comparable_total if comparable_total else None,
        }

    def _validate_batch_metric_closure(self, metrics: dict[str, Any]) -> None:
        if self.target_family == "direction":
            total = int(metrics.get("direction_teacher_total_targets", 0))
            ties = int(metrics.get("direction_teacher_exact_tie_targets", 0))
            pos = int(metrics.get("direction_teacher_positive_sign_targets", 0))
            neg = int(metrics.get("direction_teacher_negative_sign_targets", 0))
            if total != ties + pos + neg:
                raise RuntimeError(
                    "teacher observer direction target closure failed: "
                    f"total={total} ties={ties} positive={pos} negative={neg}"
                )
        elif self.target_family == "candidate_order":
            total = int(metrics.get("order_teacher_total_pairs", 0))
            ties = int(metrics.get("order_teacher_exact_tie_pairs", 0))
            pos = int(metrics.get("order_teacher_positive_pair_targets", 0))
            neg = int(metrics.get("order_teacher_negative_pair_targets", 0))
            if total != ties + pos + neg:
                raise RuntimeError(
                    "teacher observer candidate-order pair closure failed: "
                    f"total={total} ties={ties} positive={pos} negative={neg}"
                )

    def _accumulate_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        total = self.epoch_totals.setdefault(int(epoch), {"epoch": int(epoch), "rows": 0})
        total["rows"] += 1
        for key, value in metrics.items():
            if key in EPOCH_RATE_FIELDS:
                continue
            if key in EPOCH_IDENTITY_FIELDS:
                if key in total and total[key] != value:
                    raise RuntimeError(
                        f"teacher observer epoch identity field {key!r} changed "
                        f"within epoch {epoch}: {total[key]!r} != {value!r}"
                    )
                total[key] = value
            elif key in EPOCH_ADDITIVE_COUNT_FIELDS:
                if isinstance(value, dict):
                    total[key] = copy.deepcopy(value)
                else:
                    total[key] = total.get(key, 0) + value
            elif key in EPOCH_CUMULATIVE_COUNTER_FIELDS:
                prior = total.get(key)
                total[key] = value if prior is None else max(prior, value)
            elif key in EPOCH_STATE_MEASUREMENT_FIELDS:
                total[key] = value
            elif key == "batch_index":
                total["last_batch_index"] = value
            else:
                total[key] = value

    def mark_optimizer_step(self, student_model: nn.Module, *, successful: bool) -> None:
        if successful:
            self.successful_step_count += 1
            if self.mode in {"previous_step", "ema"}:
                self.update_from_student(student_model, boundary="successful_optimizer_step")
        else:
            self.skipped_step_count += 1

    def on_epoch_start(self, epoch: int) -> None:
        self.epoch_boundary_metadata = {
            **self.epoch_boundary_metadata,
            "current_epoch": int(epoch),
            "in_epoch_progress": True,
            "boundary": "start_of_current_epoch",
        }

    def on_epoch_end(self, student_model: nn.Module, epoch: int) -> None:
        self.epoch_boundary_metadata = {
            **self.epoch_boundary_metadata,
            "current_epoch": int(epoch),
            "in_epoch_progress": False,
            "boundary": "after_epoch_report_before_next_epoch",
        }
        if self.mode == "previous_epoch":
            self.update_from_student(student_model, boundary="start_of_next_epoch")

    def update_from_student(self, student_model: nn.Module, *, boundary: str) -> None:
        with torch.no_grad():
            student_state = dict(student_model.state_dict())
            teacher_state = self.teacher.state_dict()
            for key in sorted(teacher_state):
                if key not in student_state:
                    raise RuntimeError(f"teacher observer missing student key {key!r}")
                source = student_state[key].detach()
                target = teacher_state[key]
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise RuntimeError(f"teacher observer state mismatch for key {key!r}")
                if self.mode == "ema" and target.is_floating_point():
                    assert self.ema_decay is not None
                    target.mul_(self.ema_decay).add_(source, alpha=1.0 - self.ema_decay)
                else:
                    target.copy_(source)
        self.teacher.eval()
        self.update_count += 1
        self.epoch_boundary_metadata = {
            **self.epoch_boundary_metadata,
            "last_update_boundary": boundary,
            "update_count": self.update_count,
            "successful_step_count": self.successful_step_count,
        }

    def state_distance_metrics(self, student_model: nn.Module) -> dict[str, Any]:
        student_state = dict(student_model.state_dict())
        teacher_state = dict(self.teacher.state_dict())
        parameter_l2_sq = 0.0
        student_l2_sq = 0.0
        exact_matches = 0
        parameter_count = 0
        buffer_mismatch_count = 0
        named_parameters = dict(student_model.named_parameters())
        for key in sorted(teacher_state):
            teacher_value = teacher_state[key].detach()
            student_value = student_state[key].detach()
            if key in named_parameters:
                parameter_count += 1
                if teacher_value.is_floating_point():
                    diff = (student_value.float() - teacher_value.float()).reshape(-1)
                    parameter_l2_sq += float(torch.dot(diff, diff).item())
                    sval = student_value.float().reshape(-1)
                    student_l2_sq += float(torch.dot(sval, sval).item())
                if torch.equal(student_value, teacher_value):
                    exact_matches += 1
            elif not torch.equal(student_value, teacher_value):
                buffer_mismatch_count += 1
        parameter_l2 = math.sqrt(parameter_l2_sq)
        student_l2 = math.sqrt(student_l2_sq)
        return {
            "teacher_state_parameter_count": _count_parameters(self.teacher),
            "teacher_state_buffer_count": _count_buffers(self.teacher),
            "teacher_state_bytes": _state_bytes(teacher_state),
            "student_teacher_parameter_l2": parameter_l2,
            "student_teacher_parameter_relative_l2": (
                parameter_l2 / student_l2 if student_l2 > 0 else None
            ),
            "student_teacher_buffer_mismatch_count": buffer_mismatch_count,
            "student_teacher_exact_parameter_match_rate": (
                exact_matches / parameter_count if parameter_count else None
            ),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        self.teacher_state_serialized = True
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "target_family": self.target_family,
            "ema_decay": self.ema_decay,
            "teacher_state": _clone_cpu_state_dict(self.teacher),
            "successful_step_count": self.successful_step_count,
            "skipped_step_count": self.skipped_step_count,
            "read_count": self.read_count,
            "update_count": self.update_count,
            "teacher_state_restored": self.teacher_state_restored,
            "teacher_state_missing_on_resume": self.teacher_state_missing_on_resume,
            "boundary_metadata": copy.deepcopy(self.epoch_boundary_metadata),
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            self.teacher_state_missing_on_resume = True
            raise RuntimeError("enabled teacher observer checkpoint state is missing")
        required = {
            "schema_version", "mode", "target_family", "teacher_state",
            "successful_step_count", "skipped_step_count", "read_count",
            "update_count", "boundary_metadata",
        }
        missing = required - set(state)
        if missing:
            self.teacher_state_missing_on_resume = True
            raise RuntimeError(f"teacher observer checkpoint state missing keys: {sorted(missing)}")
        if state["schema_version"] != SCHEMA_VERSION:
            raise RuntimeError("teacher observer checkpoint schema mismatch")
        if state["mode"] != self.mode or state["target_family"] != self.target_family:
            raise RuntimeError("teacher observer mode/target checkpoint mismatch")
        if self.mode == "ema" and float(state.get("ema_decay")) != float(self.ema_decay):
            raise RuntimeError("teacher observer EMA decay checkpoint mismatch")
        teacher_state = state["teacher_state"]
        current = self.teacher.state_dict()
        if set(teacher_state) != set(current):
            raise RuntimeError("teacher observer checkpoint key coverage mismatch")
        for key in sorted(current):
            value = teacher_state[key]
            if not torch.is_tensor(value):
                raise RuntimeError(f"teacher observer checkpoint key {key!r} is not a tensor")
            if value.shape != current[key].shape or value.dtype != current[key].dtype:
                raise RuntimeError(f"teacher observer checkpoint shape/dtype mismatch for {key!r}")
        self.teacher.load_state_dict(teacher_state)
        self.successful_step_count = int(state["successful_step_count"])
        self.skipped_step_count = int(state["skipped_step_count"])
        self.read_count = int(state["read_count"])
        self.update_count = int(state["update_count"])
        if self.successful_step_count < 0 or self.skipped_step_count < 0 or self.update_count < 0:
            raise RuntimeError("teacher observer checkpoint counters must be non-negative")
        self.epoch_boundary_metadata = copy.deepcopy(state["boundary_metadata"])
        if not isinstance(self.epoch_boundary_metadata, dict):
            raise RuntimeError("teacher observer boundary metadata must be a dict")
        self.teacher_state_restored = True
        self.teacher_state_missing_on_resume = False
        self.teacher.eval()

    def finalize(self, student_model: nn.Module) -> None:
        epoch_rows = []
        for epoch in sorted(self.epoch_totals):
            row = dict(self.epoch_totals[epoch])
            for rate_key in (
                "direction_student_teacher_sign_agreement_rate",
                "direction_student_teacher_sign_flip_rate",
                "order_student_teacher_pair_agreement_rate",
                "order_student_teacher_pair_flip_rate",
            ):
                row.pop(rate_key, None)
            if self.target_family == "direction":
                agree = row.get("direction_student_teacher_sign_agreement_count", 0)
                disagree = row.get("direction_student_teacher_sign_disagreement_count", 0)
                comparable_total = agree + disagree
                row["direction_student_teacher_sign_agreement_rate"] = (
                    agree / comparable_total if comparable_total else None
                )
                row["direction_student_teacher_sign_flip_rate"] = (
                    disagree / comparable_total if comparable_total else None
                )
            else:
                agree = row.get("order_student_teacher_pair_agreement_count", 0)
                disagree = row.get("order_student_teacher_pair_disagreement_count", 0)
                comparable_total = agree + disagree
                row["order_student_teacher_pair_agreement_rate"] = (
                    agree / comparable_total if comparable_total else None
                )
                row["order_student_teacher_pair_flip_rate"] = (
                    disagree / comparable_total if comparable_total else None
                )
            self._validate_batch_metric_closure(row)
            epoch_rows.append(row)
        self._write_epoch_csv(epoch_rows)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_name": self.run_name,
            "mode": self.mode,
            "target_family": self.target_family,
            "ema_decay": self.ema_decay,
            "decay_selected_by_implementation": False,
            "teacher_state_initialized": self.teacher_state_initialized,
            "teacher_state_update_count": self.update_count,
            "teacher_state_read_count": self.read_count,
            "teacher_state_serialized": self.teacher_state_serialized,
            "teacher_state_restored": self.teacher_state_restored,
            "teacher_state_missing_on_resume": self.teacher_state_missing_on_resume,
            "successful_optimizer_step_count": self.successful_step_count,
            "skipped_optimizer_step_count": self.skipped_step_count,
            "student_forward_count": self.student_forward_count,
            "teacher_forward_count": self.teacher_forward_count,
            "effective_teacher_age": self.update_count if self.mode == "ema" else None,
            "parameter_and_buffer_coverage": self._coverage(student_model),
            "nonzero_loss_counts": UNAVAILABLE_NO_LOSS_OR_BACKWARD,
            "nonzero_gradient_counts": UNAVAILABLE_NO_LOSS_OR_BACKWARD,
            **self.state_distance_metrics(student_model),
        }
        self._write_json(self.output_dir / "teacher_observer_run_summary.json", summary)
        audit = {
            **summary,
            "boundary_metadata": self.epoch_boundary_metadata,
            "candidate_masks": list(CANDIDATE_MASKS),
            "exact_ties_ignored_as_active_targets": True,
            "no_lexical_candidate_order": True,
            "teacher_excluded_from_optimizer": True,
            "teacher_requires_grad_false": all(
                not parameter.requires_grad for parameter in self.teacher.parameters()
            ),
            "teacher_eval_mode_at_finalize": not self.teacher.training,
        }
        self._write_json(self.output_dir / "teacher_observer_state_audit.json", audit)

    def _coverage(self, student_model: nn.Module) -> dict[str, Any]:
        student_state = dict(student_model.state_dict())
        teacher_state = dict(self.teacher.state_dict())
        return {
            "student_key_count": len(student_state),
            "teacher_key_count": len(teacher_state),
            "keys_match": sorted(student_state) == sorted(teacher_state),
            "shape_dtype_match": all(
                key in student_state
                and student_state[key].shape == teacher_state[key].shape
                and student_state[key].dtype == teacher_state[key].dtype
                for key in teacher_state
            ),
        }

    def _write_epoch_csv(self, rows: list[dict[str, Any]]) -> None:
        path = self.output_dir / "teacher_observer_epoch_metrics.csv"
        fieldnames = sorted({key for row in rows for key in row})
        if not fieldnames:
            fieldnames = ["epoch", "rows"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})











