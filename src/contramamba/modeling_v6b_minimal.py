"""ContraMamba-v6B-minimal: v5 base + learnable slot-level comparator alphas.

Minimal design: no composer, no dual loss paths. Just v5 with final logit modulation
for temporal and predicate mismatch signals. All CE/pairwise/intervention losses
consume the final modulated logits.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .heads import (
    FinalEntitlementDecisionHead,
    FrameGate,
    PolarityEnergyHead,
    PredicateCoverageHead,
    SufficiencyGate,
)


def _inverse_softplus(target: float) -> float:
    """Compute x such that softplus(x) ≈ target.

    softplus(x) = log(1 + exp(x))
    Solving: log(1 + exp(x)) = target
             exp(x) = exp(target) - 1
             x = log(exp(target) - 1)
    """
    if target <= 0:
        raise ValueError("target must be positive")
    return math.log(math.expm1(target))


class ContraMambaV6BMinimal(nn.Module):
    """Minimal v6B: v5 base + learnable temporal/predicate comparator alphas.

    Applies learnable slot-level logit modulation for temporal and predicate
    mismatch flags. No composer, no dual logit paths. Final logits are the only
    logits used for loss and prediction.
    """

    def __init__(
        self,
        model_name: str = "state-spaces/mamba-130m-hf",
        frame_size: int = 256,
        predicate_size: int = 256,
        sufficiency_size: int = 256,
        energy_size: int = 64,
        dropout: float = 0.1,
        freeze_a_log: bool = True,
        return_token_diagnostics: bool = False,
        decision_mode: str = "explicit_product",
        backbone: nn.Module | None = None,
        hidden_size: int | None = None,
        use_temporal_comparator: bool = False,
        use_predicate_comparator: bool = False,
        alpha_temporal_init: float = 1.25,
        alpha_predicate_init: float = 1.25,
        use_boundary_head: bool = False,
        use_frame_violation_head: bool = False,
        use_predicate_isolation_head: bool = False,
        use_preservation_entitlement_head: bool = False,
        use_temporal_diagnostic_head: bool = False,
        use_temporal_residual_adapter: bool = False,
        temporal_adapter_detach_input: bool = True,
    ) -> None:
        super().__init__()
        if backbone is None:
            try:
                from transformers import MambaConfig, MambaModel
            except ImportError as exc:
                raise ImportError(
                    "transformers is required when no backbone is supplied"
                ) from exc
            config = MambaConfig.from_pretrained(model_name)
            config.use_mamba_kernels = True
            backbone = MambaModel.from_pretrained(model_name, config=config)

        self.mamba = backbone
        inferred_hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        hidden_size = hidden_size or inferred_hidden_size
        if hidden_size is None:
            raise ValueError("hidden_size is required when the backbone has no config")

        if freeze_a_log:
            for name, parameter in self.mamba.named_parameters():
                if "A_log" in name:
                    parameter.requires_grad = False

        self.return_token_diagnostics = return_token_diagnostics
        self.use_temporal_comparator = use_temporal_comparator
        self.use_predicate_comparator = use_predicate_comparator

        # V5 heads (unchanged)
        self.frame_gate = FrameGate(
            hidden_size, frame_size, dropout, return_token_diagnostics
        )
        self.predicate_coverage_head = PredicateCoverageHead(
            hidden_size, frame_size, predicate_size, dropout, return_token_diagnostics
        )
        self.sufficiency_gate = SufficiencyGate(
            frame_size, predicate_size, sufficiency_size, dropout
        )
        self.polarity_energy_head = PolarityEnergyHead(
            frame_size, predicate_size, sufficiency_size, energy_size, dropout
        )
        self.decision_head = FinalEntitlementDecisionHead(decision_mode=decision_mode)

        # Learnable alphas for comparators (initialized near calibrated values)
        self.alpha_temporal_raw: nn.Parameter | None = None
        self.alpha_predicate_raw: nn.Parameter | None = None

        if use_temporal_comparator:
            raw_val = _inverse_softplus(alpha_temporal_init)
            self.alpha_temporal_raw = nn.Parameter(torch.tensor(raw_val, dtype=torch.float32))

        if use_predicate_comparator:
            raw_val = _inverse_softplus(alpha_predicate_init)
            self.alpha_predicate_raw = nn.Parameter(torch.tensor(raw_val, dtype=torch.float32))

        # Stage22-A: preservation boundary head — distinguishes preservation-like records
        # (none, paraphrase) from frame-mismatch records (location_swap, role_swap, etc.).
        # Input: concatenated frame_pair_repr + predicate_pair_repr + sufficiency_repr.
        # Does NOT touch output["logits"] or output["base_logits"].
        self.boundary_head: nn.Linear | None = None
        if use_boundary_head:
            self.boundary_head = nn.Linear(
                frame_size + predicate_size + sufficiency_size, 1
            )

        # Stage22-A3: frame violation head — distinguishes frame-violating interventions
        # (entity_swap, event_swap, location_swap, role_swap, title_name_swap; violation=1)
        # from non-violating interventions (none, paraphrase, sufficiency/polarity types; violation=0).
        # Same input as boundary_head. Does NOT touch output["logits"] or output["base_logits"].
        self.frame_violation_head: nn.Linear | None = None
        if use_frame_violation_head:
            self.frame_violation_head = nn.Linear(
                frame_size + predicate_size + sufficiency_size, 1
            )

        # Predicate isolation head — linear probe on predicate_pair_repr only.
        # Positive (noncoverage=1): predicate_swap records.
        # Negative (covered=0): none, paraphrase records.
        # Excluded (masked): all other intervention types.
        # Keeps predicate-noncoverage supervision on predicate_pair_repr,
        # separate from FrameGate and from the existing V5 predicate_loss
        # (which uses per-record predicate_covered_label fields).
        # Does NOT touch output["logits"] or output["base_logits"].
        self.predicate_isolation_head: nn.Linear | None = None
        if use_predicate_isolation_head:
            self.predicate_isolation_head = nn.Linear(predicate_size, 1)

        # Preservation entitlement head — linear probe on sufficiency_repr only.
        # Positive (preservation-entitled=1): none, paraphrase (should remain entitled).
        # Negative (frame-rejected=0): entity_swap, event_swap, location_swap, role_swap,
        #   title_name_swap (narrow frame-mismatch rejections).
        # Excluded (masked): predicate_swap (predicate failure path), evidence_deletion/
        #   truncation/irrelevant_evidence (sufficiency failure — masked to avoid conflating
        #   frame-rejection with sufficiency-failure), polarity_flip (polarity path).
        # Uses sufficiency_repr (entitlement-level gate combining frame+predicate info),
        # distinct from predicate_isolation_head (predicate_pair_repr only) and
        # boundary/frame_violation heads (full [frame+predicate+sufficiency] concat).
        # Does NOT touch output["logits"] or output["base_logits"].
        self.preservation_entitlement_head: nn.Linear | None = None
        if use_preservation_entitlement_head:
            self.preservation_entitlement_head = nn.Linear(sufficiency_size, 1)

        # Temporal diagnostic head — linear probe on frame_pair_repr only.
        # Positive (temporal_mismatch=1): time_swap records from the separate temporal
        #   diagnostic dataset (primary_failure_type='frame', frame_compatible_label=0).
        # Negative (temporal_control=0): none, paraphrase records from same dataset.
        # Input: frame_pair_repr (frame_size) — targets frame-level representation since
        #   time_swap is a frame-compatibility failure. Narrower than the full
        #   [frame+predicate+sufficiency] concat used by boundary/frame_violation heads.
        #   Distinct from predicate_isolation_head (predicate_pair_repr) and
        #   preservation_entitlement_head (sufficiency_repr).
        # Loaded from a SEPARATE temporal diagnostic JSONL — records must NOT be mixed
        #   into the main clean train/eval classification data.
        # Does NOT touch output["logits"] or output["base_logits"].
        # Does NOT supervise FrameGate directly; supervises a linear probe on frame_pair_repr.
        self.temporal_diagnostic_head: nn.Linear | None = None
        if use_temporal_diagnostic_head:
            self.temporal_diagnostic_head = nn.Linear(frame_size, 1)

        # Temporal residual adapter — small 2-layer MLP probing frame_pair_repr for temporal
        # mismatch, designed to absorb temporal diagnostic supervision without corrupting
        # the shared FrameGate / frame_pair_repr path.
        # By default, the input is DETACHED from the computation graph so that the adapter
        # BCE loss cannot propagate gradients back into frame_pair_repr / FrameGate.
        # This isolates temporal supervision from the preservation / predicate paths.
        # Architecture: Linear(frame_size, frame_size // 2) → GELU → Linear(frame_size // 2, 1)
        # Stage23 showed that routing TD supervision directly through frame_pair_repr
        # (as temporal_diagnostic_head does) collapses preservation. This adapter uses the
        # same input representation but blocks the gradient path by default.
        # Optionally: temporal_adapter_logit (detached) can apply a per-example NOT_ENTITLED
        # penalty to final_logits (see temporal_adapter_final_penalty_scale in forward).
        # Does NOT touch output["logits"] or output["base_logits"] unless penalty scale > 0.
        self.temporal_residual_adapter: nn.Sequential | None = None
        self.temporal_adapter_detach_input: bool = temporal_adapter_detach_input
        if use_temporal_residual_adapter:
            _ta_hidden = max(frame_size // 2, 8)
            self.temporal_residual_adapter = nn.Sequential(
                nn.Linear(frame_size, _ta_hidden),
                nn.GELU(),
                nn.Linear(_ta_hidden, 1),
            )

    def alpha_temporal(self) -> float | torch.Tensor:
        """Return softplus-constrained temporal alpha."""
        if self.alpha_temporal_raw is None:
            return 0.0
        return F.softplus(self.alpha_temporal_raw)

    def alpha_predicate(self) -> float | torch.Tensor:
        """Return softplus-constrained predicate alpha."""
        if self.alpha_predicate_raw is None:
            return 0.0
        return F.softplus(self.alpha_predicate_raw)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
        final_labels: torch.Tensor | None = None,
        frame_compatible_labels: torch.Tensor | None = None,
        predicate_covered_labels: torch.Tensor | None = None,
        sufficiency_labels: torch.Tensor | None = None,
        polarity_labels: torch.Tensor | None = None,
        polarity_mask: torch.Tensor | None = None,
        intervention_types: torch.Tensor | None = None,
        pair_ids: torch.Tensor | None = None,
        return_token_states: bool = False,
        decision_mode: str | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        temporal_mismatch_flags: torch.Tensor | None = None,
        predicate_mismatch_flags: torch.Tensor | None = None,
        temporal_adapter_final_penalty_scale: float = 0.0,
    ) -> dict[str, Any]:
        """Forward pass with optional temporal/predicate logit modulation."""
        del intervention_types, pair_ids

        # Encode
        if encoder_hidden_states is None:
            backbone_outputs = self.mamba(input_ids=input_ids)
            token_states = backbone_outputs.last_hidden_state
        else:
            if encoder_hidden_states.shape[:2] != input_ids.shape:
                raise ValueError(
                    "encoder_hidden_states must match input_ids batch/sequence dimensions"
                )
            token_states = encoder_hidden_states

        # Slot gates (unchanged from V5)
        frame = self.frame_gate(token_states, attention_mask, claim_mask, evidence_mask)
        predicate = self.predicate_coverage_head(
            token_states=token_states,
            attention_mask=attention_mask,
            claim_mask=claim_mask,
            evidence_mask=evidence_mask,
            claim_frame_state=frame["claim_frame_state"],
            evidence_frame_state=frame["evidence_frame_state"],
            frame_pair_repr=frame["frame_pair_repr"],
            frame_prob=frame["frame_prob"],
        )
        sufficiency = self.sufficiency_gate(
            frame_pair_repr=frame["frame_pair_repr"],
            predicate_pair_repr=predicate["predicate_pair_repr"],
            frame_prob=frame["frame_prob"],
            predicate_coverage_prob=predicate["predicate_coverage_prob"],
        )
        polarity = self.polarity_energy_head(
            frame_pair_repr=frame["frame_pair_repr"],
            predicate_pair_repr=predicate["predicate_pair_repr"],
            sufficiency_repr=sufficiency["sufficiency_repr"],
        )

        # Stage22-A / A3: shared slot concatenation for auxiliary diagnostic heads.
        # Computed once; reused by both boundary_head and frame_violation_head.
        # Does NOT modify final_logits or base_logits.
        _aux_input: torch.Tensor | None = None
        if self.boundary_head is not None or self.frame_violation_head is not None:
            _aux_input = torch.cat(
                [
                    frame["frame_pair_repr"],
                    predicate["predicate_pair_repr"],
                    sufficiency["sufficiency_repr"],
                ],
                dim=-1,
            )

        # Stage22-A: preservation boundary head (diagnostic/training signal only)
        # Output: boundary_logit [B], boundary_prob [B]; None when head is disabled.
        boundary_logit: torch.Tensor | None = None
        boundary_prob: torch.Tensor | None = None
        if self.boundary_head is not None and _aux_input is not None:
            boundary_logit = self.boundary_head(_aux_input).squeeze(-1)
            boundary_prob = torch.sigmoid(boundary_logit)

        # Stage22-A3: frame violation head (diagnostic/training signal only)
        # Output: frame_violation_logit [B], frame_violation_prob [B]; None when disabled.
        frame_violation_logit: torch.Tensor | None = None
        frame_violation_prob: torch.Tensor | None = None
        if self.frame_violation_head is not None and _aux_input is not None:
            frame_violation_logit = self.frame_violation_head(_aux_input).squeeze(-1)
            frame_violation_prob = torch.sigmoid(frame_violation_logit)

        # Predicate isolation head: linear probe on predicate_pair_repr for noncoverage.
        # Positive=predicate_swap (noncoverage); negative=none/paraphrase (covered).
        # None when head is disabled (safe default: downstream loss is skipped if None).
        predicate_noncoverage_logit: torch.Tensor | None = None
        predicate_noncoverage_prob: torch.Tensor | None = None
        if self.predicate_isolation_head is not None:
            predicate_noncoverage_logit = self.predicate_isolation_head(
                predicate["predicate_pair_repr"]
            ).squeeze(-1)
            predicate_noncoverage_prob = torch.sigmoid(predicate_noncoverage_logit)

        # Preservation entitlement head: linear probe on sufficiency_repr.
        # Positive=none/paraphrase (entitled); negative=frame-swaps (rejected).
        # None when head is disabled (downstream loss skipped when None).
        preservation_entitlement_logit: torch.Tensor | None = None
        preservation_entitlement_prob: torch.Tensor | None = None
        if self.preservation_entitlement_head is not None:
            preservation_entitlement_logit = self.preservation_entitlement_head(
                sufficiency["sufficiency_repr"]
            ).squeeze(-1)
            preservation_entitlement_prob = torch.sigmoid(preservation_entitlement_logit)

        # Temporal diagnostic head: linear probe on frame_pair_repr for temporal mismatch.
        # Positive=time_swap (temporal_mismatch=1); negative=none/paraphrase (temporal_control=0).
        # Loaded from a separate temporal diagnostic file; these records are not in the main
        # clean train/eval tensors — the head output is present in every forward pass
        # but the BCE loss is only computed on the separate temporal diagnostic batch.
        # None when head is disabled (downstream loss skipped when None).
        temporal_diagnostic_logit: torch.Tensor | None = None
        temporal_diagnostic_prob: torch.Tensor | None = None
        if self.temporal_diagnostic_head is not None:
            temporal_diagnostic_logit = self.temporal_diagnostic_head(
                frame["frame_pair_repr"]
            ).squeeze(-1)
            temporal_diagnostic_prob = torch.sigmoid(temporal_diagnostic_logit)

        # Temporal residual adapter: small 2-layer MLP on (optionally detached) frame_pair_repr.
        # Default: input is detached so adapter BCE loss cannot propagate gradients into
        # FrameGate / frame_pair_repr. This prevents the temporal-rejection / preservation
        # conflict observed in Stage23 (where temporal_diagnostic_head routed gradients
        # directly through frame_pair_repr, collapsing preservation and predicate disentanglement).
        # None when adapter is disabled (downstream loss skipped; no logit penalty applied).
        temporal_adapter_logit: torch.Tensor | None = None
        temporal_adapter_prob: torch.Tensor | None = None
        if self.temporal_residual_adapter is not None:
            _ta_input = (
                frame["frame_pair_repr"].detach()
                if self.temporal_adapter_detach_input
                else frame["frame_pair_repr"]
            )
            temporal_adapter_logit = self.temporal_residual_adapter(_ta_input).squeeze(-1)
            temporal_adapter_prob = torch.sigmoid(temporal_adapter_logit)

        # Base logits (V5 standard)
        decision = self.decision_head(
            frame_prob=frame["frame_prob"],
            predicate_coverage_prob=predicate["predicate_coverage_prob"],
            sufficiency_prob=sufficiency["sufficiency_prob"],
            positive_energy=polarity["positive_energy"],
            negative_energy=polarity["negative_energy"],
            decision_mode=decision_mode,
        )
        base_logits = decision["logits"]

        # Apply final logit modulation
        final_logits = base_logits
        temporal_flag_count = 0
        predicate_flag_count = 0

        if self.use_temporal_comparator and temporal_mismatch_flags is not None:
            alpha = self.alpha_temporal()
            active = temporal_mismatch_flags.bool()
            if torch.any(active):
                final_logits = final_logits.clone()
                final_logits[active, 0] -= alpha  # SUPPORT
                final_logits[active, 1] += alpha  # NOT_ENTITLED
                final_logits[active, 2] -= alpha  # REFUTE
                temporal_flag_count = int(active.sum().item())

        if self.use_predicate_comparator and predicate_mismatch_flags is not None:
            alpha = self.alpha_predicate()
            active = predicate_mismatch_flags.bool()
            if torch.any(active):
                final_logits = final_logits.clone()
                final_logits[active, 0] -= alpha  # SUPPORT
                final_logits[active, 1] += alpha  # NOT_ENTITLED
                final_logits[active, 2] -= alpha  # REFUTE
                predicate_flag_count = int(active.sum().item())

        # Optional temporal adapter per-example NOT_ENTITLED penalty.
        # Architecture-level penalty driven by adapter temporal mismatch confidence.
        # NOT OOD calibration. Stage15 is never used to set penalty scale.
        # Penalty is per-example (proportional to adapter prob), not a global shift.
        # Gradient does NOT flow to the adapter from this path (logit is detached).
        # When scale=0.0 (default) or adapter is disabled: final_logits unchanged.
        if (
            temporal_adapter_logit is not None
            and temporal_adapter_final_penalty_scale > 0.0
        ):
            _ta_penalty = torch.sigmoid(temporal_adapter_logit.detach()) * temporal_adapter_final_penalty_scale
            final_logits = final_logits.clone()
            final_logits[:, 0] -= _ta_penalty  # SUPPORT
            final_logits[:, 1] += _ta_penalty  # NOT_ENTITLED
            final_logits[:, 2] -= _ta_penalty  # REFUTE

        # Compute losses (using final_logits only)
        losses: dict[str, torch.Tensor | None] = {
            "label_loss": None,
            "frame_loss": None,
            "predicate_loss": None,
            "sufficiency_loss": None,
            "polarity_loss": None,
        }
        if final_labels is not None:
            losses["label_loss"] = F.cross_entropy(final_logits, final_labels)
        if frame_compatible_labels is not None:
            losses["frame_loss"] = F.binary_cross_entropy_with_logits(
                frame["frame_logit"], frame_compatible_labels.float()
            )
        if predicate_covered_labels is not None:
            losses["predicate_loss"] = F.binary_cross_entropy_with_logits(
                predicate["predicate_coverage_logit"],
                predicate_covered_labels.float(),
            )
        if sufficiency_labels is not None:
            losses["sufficiency_loss"] = F.binary_cross_entropy_with_logits(
                sufficiency["sufficiency_logit"], sufficiency_labels.float()
            )
        if polarity_labels is not None:
            if polarity_mask is None:
                polarity_mask = (
                    final_labels != 1
                    if final_labels is not None
                    else polarity_labels != 0
                )
            active = polarity_mask.bool()
            if torch.any(active):
                polarity_logits = torch.stack(
                    [
                        torch.zeros_like(polarity["negative_energy"]),
                        polarity["negative_energy"],
                        polarity["positive_energy"],
                    ],
                    dim=-1,
                )
                losses["polarity_loss"] = F.cross_entropy(
                    polarity_logits[active], polarity_labels[active]
                )

        active_losses = [loss for loss in losses.values() if loss is not None]
        total_loss = torch.stack(active_losses).sum() if active_losses else None

        # Return: final_logits as the only logits for downstream losses
        return {
            **decision,
            "logits": final_logits,  # ← FINAL logits for all downstream loss paths
            "base_logits": base_logits,  # ← diagnostic only
            "predictions": final_logits.argmax(dim=-1),  # ← derived from final logits
            "alpha_temporal": self.alpha_temporal(),
            "alpha_predicate": self.alpha_predicate(),
            "temporal_flag_count": temporal_flag_count,
            "predicate_flag_count": predicate_flag_count,
            "final_logits_used": True,  # ← assertion for smoke test
            # Stage22-A: boundary head outputs (None when head is disabled)
            "boundary_logit": boundary_logit,
            "boundary_prob": boundary_prob,
            # Stage22-A3: frame violation head outputs (None when head is disabled)
            "frame_violation_logit": frame_violation_logit,
            "frame_violation_prob": frame_violation_prob,
            # Predicate isolation head outputs (None when head is disabled)
            "predicate_noncoverage_logit": predicate_noncoverage_logit,
            "predicate_noncoverage_prob": predicate_noncoverage_prob,
            # Preservation entitlement head outputs (None when head is disabled)
            "preservation_entitlement_logit": preservation_entitlement_logit,
            "preservation_entitlement_prob": preservation_entitlement_prob,
            # Temporal diagnostic head outputs (None when head is disabled)
            "temporal_diagnostic_logit": temporal_diagnostic_logit,
            "temporal_diagnostic_prob": temporal_diagnostic_prob,
            # Temporal residual adapter outputs (None when adapter is disabled)
            "temporal_adapter_logit": temporal_adapter_logit,
            "temporal_adapter_prob": temporal_adapter_prob,
            **frame,
            **predicate,
            **sufficiency,
            **polarity,
            "token_states": token_states if return_token_states else None,
            "loss": total_loss,
            **losses,
        }
