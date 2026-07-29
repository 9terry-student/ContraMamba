from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("alpha_init must be positive")
    return math.log(math.expm1(value))



def _p2_validate_finite_nonnegative(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if (not torch.isfinite(tensor).all()) or (tensor < 0.0).any():
        raise ValueError(f"P2_ROUTER_NUMERICAL_CONTRACT_FAILED: {name}")
    return tensor


def _p2_finite_nonnegative(tensor: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(tensor).all() & (tensor >= 0.0).all()


class FinalEntitlementDecisionHead(nn.Module):
    """Final authority for REFUTE / NOT_ENTITLED / SUPPORT."""

    VALID_DECISION_MODES = {
        "explicit_product",
        "logit_sum",
        "conditional_first_blocker",
    }
    REASON_CLASS_ORDER = ("FRAME", "PREDICATE", "SUFFICIENCY", "AUTHORIZED")
    INTERNAL_CLASS_ORDER = (
        "FRAME_FAIL",
        "PREDICATE_FAIL",
        "SUFFICIENCY_FAIL",
        "REFUTE",
        "SUPPORT",
    )
    EXTERNAL_CLASS_ORDER = ("REFUTE", "NOT_ENTITLED", "SUPPORT")

    def __init__(
        self,
        not_entitled_bias_init: float = 0.0,
        alpha_init: float = 1.0,
        decision_mode: str = "explicit_product",
        router_epsilon: float = 1e-8,
        reason_bias_init: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        if decision_mode not in self.VALID_DECISION_MODES:
            raise ValueError(f"unsupported decision_mode: {decision_mode}")
        if not math.isfinite(float(router_epsilon)) or not (
            0.0 < float(router_epsilon) <= 1e-4
        ):
            raise ValueError("reason_router_epsilon must satisfy 0 < epsilon <= 1e-4")
        if len(reason_bias_init) != 3:
            raise ValueError("reason_bias_init must contain exactly 3 values")
        self.decision_mode = decision_mode
        self.router_epsilon = float(router_epsilon)
        self.not_entitled_bias = nn.Parameter(
            torch.tensor(float(not_entitled_bias_init))
        )
        self.raw_alpha = nn.Parameter(torch.tensor(_inverse_softplus(alpha_init)))
        if decision_mode == "conditional_first_blocker":
            self.reason_bias_3 = nn.Parameter(
                torch.tensor(tuple(float(value) for value in reason_bias_init))
            )
            self.not_entitled_bias.requires_grad_(False)
            self.raw_alpha.requires_grad_(False)

    def forward(
        self,
        frame_prob: torch.Tensor,
        predicate_coverage_prob: torch.Tensor,
        sufficiency_prob: torch.Tensor,
        positive_energy: torch.Tensor,
        negative_energy: torch.Tensor,
        decision_mode: str | None = None,
        *,
        return_q_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        mode = decision_mode or self.decision_mode
        if mode not in self.VALID_DECISION_MODES:
            raise ValueError(f"unsupported decision_mode: {mode}")
        if mode == "conditional_first_blocker" and not hasattr(
            self, "reason_bias_3"
        ):
            raise ValueError(
                "P2_ROUTER_PARAMETERS_NOT_INITIALIZED: "
                "conditional_first_blocker must be selected at construction time"
            )

        if mode == "explicit_product":
            entitlement_prob = (
                frame_prob * predicate_coverage_prob * sufficiency_prob
            )
        elif mode == "logit_sum":
            eps = torch.finfo(frame_prob.dtype).eps
            gate_probs = (frame_prob, predicate_coverage_prob, sufficiency_prob)
            gate_logits = [
                torch.logit(prob.clamp(min=eps, max=1.0 - eps))
                for prob in gate_probs
            ]
            entitlement_prob = torch.sigmoid(torch.stack(gate_logits).sum(dim=0))
        else:
            entitlement_prob = (
                frame_prob * predicate_coverage_prob * sufficiency_prob
            )

        support_logit = entitlement_prob * positive_energy
        refute_logit = entitlement_prob * negative_energy
        alpha = F.softplus(self.raw_alpha)
        not_entitled_logit = self.not_entitled_bias + alpha * (
            1.0 - entitlement_prob
        )
        logits = torch.stack(
            [refute_logit, not_entitled_logit, support_logit], dim=-1
        )
        product_output = {
            "entitlement_prob": entitlement_prob,
            "support_logit": support_logit,
            "refute_logit": refute_logit,
            "not_entitled_logit": not_entitled_logit,
            "logits": logits,
        }
        if mode != "conditional_first_blocker":
            if not return_q_diagnostics:
                return product_output
            q_frame = 1.0 - frame_prob
            q_predicate = frame_prob * (1.0 - predicate_coverage_prob)
            q_sufficiency = (
                frame_prob * predicate_coverage_prob * (1.0 - sufficiency_prob)
            )
            q_authorized = frame_prob * predicate_coverage_prob * sufficiency_prob
            q_masses_4 = torch.stack([q_frame, q_predicate, q_sufficiency, q_authorized], dim=-1)
            q_sum = q_frame + q_predicate + q_sufficiency + q_authorized
            return {
                **product_output,
                "q_frame": q_frame,
                "q_predicate": q_predicate,
                "q_sufficiency": q_sufficiency,
                "q_authorized": q_authorized,
                "q_masses_4": q_masses_4,
                "q_sum": q_sum,
                "q_sum_abs_error": (q_sum - 1.0).abs(),
            }

        tensors = (
            frame_prob,
            predicate_coverage_prob,
            sufficiency_prob,
            positive_energy,
            negative_energy,
        )
        if not all(tensor.ndim == 1 for tensor in tensors):
            raise ValueError("conditional_first_blocker inputs must all have shape [B]")
        if frame_prob.shape[0] == 0:
            raise ValueError("conditional_first_blocker requires B >= 1")
        if not all(tensor.shape == frame_prob.shape for tensor in tensors):
            raise ValueError("conditional_first_blocker input shapes must match")
        if not all(tensor.device == frame_prob.device for tensor in tensors):
            raise ValueError("conditional_first_blocker input devices must match")
        if not all(tensor.is_floating_point() for tensor in tensors):
            raise ValueError("conditional_first_blocker inputs must be floating point")
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("conditional_first_blocker inputs must be finite")
        probability_tolerance = 1e-6
        for name, tensor in (
            ("frame_prob", frame_prob),
            ("predicate_coverage_prob", predicate_coverage_prob),
            ("sufficiency_prob", sufficiency_prob),
        ):
            if (
                (tensor < -probability_tolerance).any()
                or (tensor > 1.0 + probability_tolerance).any()
            ):
                raise ValueError(f"{name} must be in [0, 1]")
        if (positive_energy < 0.0).any() or (negative_energy < 0.0).any():
            raise ValueError("polarity energies must be nonnegative")

        compute_dtype = (
            torch.float64
            if any(tensor.dtype == torch.float64 for tensor in tensors)
            or self.reason_bias_3.dtype == torch.float64
            else torch.float32
        )
        compute_frame = frame_prob.to(dtype=compute_dtype)
        compute_predicate = predicate_coverage_prob.to(dtype=compute_dtype)
        compute_sufficiency = sufficiency_prob.to(dtype=compute_dtype)
        compute_positive = positive_energy.to(dtype=compute_dtype)
        compute_negative = negative_energy.to(dtype=compute_dtype)

        q_frame = 1.0 - compute_frame
        q_predicate = compute_frame * (1.0 - compute_predicate)
        q_sufficiency = compute_frame * compute_predicate * (1.0 - compute_sufficiency)
        q_authorized = compute_frame * compute_predicate * compute_sufficiency
        q_stack_4 = torch.stack(
            [q_frame, q_predicate, q_sufficiency, q_authorized], dim=-1
        )
        _p2_validate_finite_nonnegative("q_masses_4", q_stack_4)
        epsilon = self.router_epsilon
        reason_bias_4 = torch.cat(
            [
                self.reason_bias_3.to(device=q_stack_4.device, dtype=compute_dtype).expand(
                    q_stack_4.shape[0], 3
                ),
                torch.zeros_like(q_authorized.unsqueeze(-1)),
            ],
            dim=-1,
        )
        reason_log_input_4 = q_stack_4 + epsilon
        _p2_validate_finite_nonnegative("reason_log_input_4", reason_log_input_4)
        reason_logits_4 = torch.log(reason_log_input_4) + reason_bias_4
        reason_probs_4 = torch.softmax(reason_logits_4, dim=-1)

        polarity_logits_2 = torch.stack([compute_negative, compute_positive], dim=-1)
        polarity_probs_2 = torch.softmax(polarity_logits_2, dim=-1)
        internal_probs_5 = torch.stack(
            [
                reason_probs_4[:, 0],
                reason_probs_4[:, 1],
                reason_probs_4[:, 2],
                reason_probs_4[:, 3] * polarity_probs_2[:, 0],
                reason_probs_4[:, 3] * polarity_probs_2[:, 1],
            ],
            dim=-1,
        )
        collapsed_probs_3 = torch.stack(
            [
                internal_probs_5[:, 3],
                (
                    internal_probs_5[:, 0]
                    + internal_probs_5[:, 1]
                    + internal_probs_5[:, 2]
                ),
                internal_probs_5[:, 4],
            ],
            dim=-1,
        )
        collapsed_log_input_3 = collapsed_probs_3 + epsilon
        _p2_validate_finite_nonnegative("collapsed_log_input_3", collapsed_log_input_3)
        collapsed_logits_3 = torch.log(collapsed_log_input_3)
        failure_denominator = collapsed_probs_3[:, 1] + epsilon
        primary_reason_posterior = (
            internal_probs_5[:, 0:3] / failure_denominator.unsqueeze(-1)
        )
        primary_reason_posterior_valid_mask = collapsed_logits_3.argmax(dim=-1) == 1
        primary_reason_posterior_sum = primary_reason_posterior.sum(dim=-1)

        q_sum = q_stack_4.sum(dim=-1)
        reason_probs_4_sum = reason_probs_4.sum(dim=-1)
        internal_probs_5_sum = internal_probs_5.sum(dim=-1)
        collapsed_probs_3_sum = collapsed_probs_3.sum(dim=-1)
        normalization_errors = torch.stack(
            [
                (q_sum - 1.0).abs(),
                (reason_probs_4_sum - 1.0).abs(),
                (internal_probs_5_sum - 1.0).abs(),
                (collapsed_probs_3_sum - 1.0).abs(),
            ],
            dim=-1,
        )
        normalization_max_abs_error = normalization_errors.max()
        normalization_tolerance = 1e-12 if compute_dtype == torch.float64 else 1e-6
        finite_nonnegative_ok = (
            _p2_finite_nonnegative(q_stack_4)
            & _p2_finite_nonnegative(reason_probs_4)
            & _p2_finite_nonnegative(internal_probs_5)
            & _p2_finite_nonnegative(collapsed_probs_3)
        )
        normalization_ok = finite_nonnegative_ok & (normalization_max_abs_error <= normalization_tolerance)

        return {
            "entitlement_prob": reason_probs_4[:, 3],
            "support_logit": collapsed_logits_3[:, 2],
            "refute_logit": collapsed_logits_3[:, 0],
            "not_entitled_logit": collapsed_logits_3[:, 1],
            "logits": collapsed_logits_3,
            "q_frame": q_frame,
            "q_predicate": q_predicate,
            "q_sufficiency": q_sufficiency,
            "q_authorized": q_authorized,
            "q_masses_4": q_stack_4,
            "reason_logits_4": reason_logits_4,
            "reason_probs_4": reason_probs_4,
            "polarity_logits_2": polarity_logits_2,
            "polarity_probs_2": polarity_probs_2,
            "internal_probs_5": internal_probs_5,
            "collapsed_probs_3": collapsed_probs_3,
            "collapsed_logits_3": collapsed_logits_3,
            "primary_reason_posterior": primary_reason_posterior,
            "primary_reason_posterior_sum": primary_reason_posterior_sum,
            "primary_reason_posterior_valid_mask": primary_reason_posterior_valid_mask,
            "original_product_logits_3": product_output["logits"].detach(),
            "q_sum": q_sum,
            "q_sum_abs_error": (q_sum - 1.0).abs(),
            "reason_probs_4_sum": reason_probs_4_sum,
            "internal_probs_5_sum": internal_probs_5_sum,
            "collapsed_probs_3_sum": collapsed_probs_3_sum,
            "normalization_max_abs_error": normalization_max_abs_error,
            "normalization_ok": normalization_ok,
        }

