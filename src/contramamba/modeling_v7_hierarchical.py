"""ContraMamba-v7-Hierarchical: Stage26-A v7 hierarchical entitlement architecture.

Clean v7 architecture path. Does not modify v6B behavior. Implements the
six-axis hierarchical judgment pipeline:

    Mamba encoder
      ↓
    FrameChannel  (novelty / frame mismatch)
      ↓
    PredicateChannel  (ambiguity / predicate noncoverage)
      ↓
    SufficiencyChannel + TemporalChannel  (ignorance / temporal invalidity)
      ↓
    EntitlementGate  (aggregates pre-entitlement channel signals)
      ↓
    PolarityChannel  (truth / contradiction — post-entitlement)
      ↓
    Final logits: REFUTE=0 / NOT_ENTITLED=1 / SUPPORT=2

Historical framing:
    ContraMamba originated the six-axis framework (-1/0/+1 expanded through
    ambiguity/ignorance/novelty). EpistemicBERT was a pragmatic detour/testbed;
    it operationalized the framework in codebook annotation order but did not
    originate it. Stage26 returns the clarified hierarchy to the original
    ContraMamba architecture.

Final logit composition (see ContraMambaV7Hierarchical.forward):
    support_score = entitlement_logit + polarity_support_logit
    refute_score  = entitlement_logit + polarity_refute_logit
    ne_score      = -entitlement_logit + ne_bias
    logits[B, 3]  = [refute_score, ne_score, support_score]
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .heads import FrameGate, PredicateCoverageHead, SufficiencyGate


# ---------------------------------------------------------------------------
# Stage26-B: v7 output contract
# ---------------------------------------------------------------------------
# All keys that ContraMambaV7Hierarchical.forward() MUST include on every call.
# Derived from what v5.controlled_losses, v5.compute_metrics,
# v5.intervention_diagnostics, v5.intervention_objective, v5.prediction_records,
# and v5.pairwise_checks actually access at runtime.
#
# Intentional exclusions:
#   v7_temporal_logit / v7_temporal_prob — present only when temporal channel is
#   active (None otherwise); no v5 utility reads them.
V7_REQUIRED_OUTPUT_KEYS: tuple[str, ...] = (
    # ── Core (inviolable) ────────────────────────────────────────────────────
    # CE loss uses "logits". "base_logits" equals "logits" in Stage26-A/B.
    "logits",
    "base_logits",
    "predictions",
    # ── v5.controlled_losses ─────────────────────────────────────────────────
    "frame_logit",
    "predicate_coverage_logit",
    "sufficiency_logit",
    "positive_energy",   # alias for v7_polarity_support_logit
    "negative_energy",   # alias for v7_polarity_refute_logit
    # ── v5.compute_metrics / intervention_diagnostics / intervention_objective
    # ── / prediction_records / pairwise_checks ───────────────────────────────
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "entitlement_prob",
    "polarity_margin",
    # ── v7 diagnostic keys (always present when architecture=v7_hierarchical) ─
    "v7_frame_logit",
    "v7_frame_prob",
    "v7_predicate_logit",
    "v7_predicate_prob",
    "v7_sufficiency_logit",
    "v7_sufficiency_prob",
    "v7_entitlement_logit",
    "v7_entitlement_prob",
    "v7_polarity_logits",
    "v7_channel_output_keys",
)


def validate_v7_output_contract(output: dict[str, Any]) -> None:
    """Raise KeyError listing all missing keys if the v7 output contract is violated.

    NOT called automatically on every forward pass — use in unit tests or a
    one-time validation pass (Stage26-C).  Calling this inside the training loop
    on every step would add overhead and is not needed once the model is verified.

    Usage:
        from contramamba.modeling_v7_hierarchical import validate_v7_output_contract
        out = model(**inputs)
        validate_v7_output_contract(out)   # raises KeyError if contract is broken

    v7_temporal_logit / v7_temporal_prob are excluded from the contract: they are
    intentionally None when --v7-disable-temporal-channel is active, and no v5
    utility reads them.
    """
    missing = [k for k in V7_REQUIRED_OUTPUT_KEYS if k not in output]
    if missing:
        raise KeyError(
            f"v7 output contract violated — missing keys: {missing}\n"
            f"Full required set: {list(V7_REQUIRED_OUTPUT_KEYS)}"
        )


class TemporalChannelV2(nn.Module):
    """Temporal invalidity probe for v7 EntitlementGate.

    Reads cat([claim_frame_state, evidence_frame_state]) — the pre-pair-projector
    pooled slot states from FrameGate's project() step, NOT frame_pair_repr.
    This separates the temporal channel from the FrameGate composition path.

    In v7, high temporal_prob ≈ temporal mismatch → reduces entitlement (via
    temporal_validity = 1 - temporal_prob fed into EntitlementGateV7).
    This routes temporal mismatch through the EntitlementGate rather than as a
    direct final-logit boost (Stage24 structural correction).

    Input:  cat([claim_frame_state, evidence_frame_state])  [B, frame_size * 2]
    Output: v7_temporal_logit [B], v7_temporal_prob [B]
    """

    def __init__(self, frame_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(frame_size // 2, 8)
        self.mlp = nn.Sequential(
            nn.Linear(frame_size * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        claim_frame_state: torch.Tensor,
        evidence_frame_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tc_input = torch.cat([claim_frame_state, evidence_frame_state], dim=-1)
        logit = self.mlp(tc_input).squeeze(-1)
        return {
            "v7_temporal_logit": logit,
            "v7_temporal_prob": torch.sigmoid(logit),
        }


class EntitlementGateV7(nn.Module):
    """Learned aggregation of pre-entitlement channel signals.

    Aggregates frame, predicate, sufficiency, and (optionally) temporal validity
    signals into a scalar entitlement logit.

    High entitlement_logit → model is entitled to make a polarity judgment.
    Low entitlement_logit → NOT_ENTITLED (one of: novelty, ambiguity, ignorance,
    temporal invalidity).

    Default mode: small learned MLP over channel probabilities (4 or 3 inputs).
    Flat arbiter mode (v7_flat_arbiter=True): explicit product formula, mimicking
    v6B's explicit_product mode for ablation comparison.

    Temporal input convention: temporal_prob is the mismatch probability
    (high = mismatch). The gate receives (1 - temporal_prob) so that a high
    temporal_prob correctly reduces the entitlement estimate.

    Architecture (default mode):
        input_dim = 4 if use_temporal_input else 3
        hidden    = max(input_dim * 4, 16)
        MLP: Linear(input_dim, hidden) → GELU → Linear(hidden, 1)
    """

    def __init__(self, use_temporal_input: bool = True) -> None:
        super().__init__()
        self.use_temporal_input = use_temporal_input
        input_dim = 4 if use_temporal_input else 3
        hidden = max(input_dim * 4, 16)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        frame_prob: torch.Tensor,
        predicate_coverage_prob: torch.Tensor,
        sufficiency_prob: torch.Tensor,
        temporal_prob: torch.Tensor | None = None,
        flat_arbiter: bool = False,
    ) -> dict[str, torch.Tensor]:
        if flat_arbiter:
            # Explicit product formula (v6B-like; ablation reference)
            if temporal_prob is not None:
                temporal_validity = 1.0 - temporal_prob
                entitlement_prob = (
                    frame_prob * predicate_coverage_prob * sufficiency_prob * temporal_validity
                )
            else:
                entitlement_prob = frame_prob * predicate_coverage_prob * sufficiency_prob
            eps = torch.finfo(frame_prob.dtype).eps
            entitlement_logit = torch.log(entitlement_prob.clamp(min=eps))
        else:
            # Learned MLP aggregation
            if self.use_temporal_input and temporal_prob is not None:
                # temporal_validity = 1 - temporal_prob so that high mismatch → low gate input
                gate_input = torch.stack(
                    [
                        frame_prob,
                        predicate_coverage_prob,
                        sufficiency_prob,
                        1.0 - temporal_prob,
                    ],
                    dim=-1,
                )
            else:
                gate_input = torch.stack(
                    [frame_prob, predicate_coverage_prob, sufficiency_prob],
                    dim=-1,
                )
            entitlement_logit = self.mlp(gate_input).squeeze(-1)
            entitlement_prob = torch.sigmoid(entitlement_logit)

        return {
            "v7_entitlement_logit": entitlement_logit,
            "v7_entitlement_prob": entitlement_prob,
        }


class PolarityChannelV7(nn.Module):
    """Post-entitlement polarity channel for v7.

    Reads from the combined frame/predicate/sufficiency representations and
    outputs separate support and refute logits. These are raw logits (not
    softplus-constrained energies like v6B's PolarityEnergyHead) so they
    participate cleanly in the additive final logit composition.

    The polarity channel's contribution to the final classification is modulated
    by the EntitlementGate via the additive composition:
        support_score = entitlement_logit + polarity_support_logit
        refute_score  = entitlement_logit + polarity_refute_logit

    When entitlement_logit is large (negative), both support_score and refute_score
    decrease; ne_score (-entitlement_logit + ne_bias) increases. Polarity signals
    are suppressed under low entitlement without explicit gating.

    Architecture:
        input_dim = frame_size + predicate_size + sufficiency_size
        projector: Linear(input_dim, polarity_size) → GELU → Dropout → LayerNorm
        support_head: Linear(polarity_size, 1)
        refute_head:  Linear(polarity_size, 1)
    """

    def __init__(
        self,
        frame_size: int,
        predicate_size: int,
        sufficiency_size: int,
        polarity_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = frame_size + predicate_size + sufficiency_size
        self.projector = nn.Sequential(
            nn.Linear(input_dim, polarity_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(polarity_size),
        )
        self.support_head = nn.Linear(polarity_size, 1)
        self.refute_head = nn.Linear(polarity_size, 1)

    def forward(
        self,
        frame_pair_repr: torch.Tensor,
        predicate_pair_repr: torch.Tensor,
        sufficiency_repr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = torch.cat(
            [frame_pair_repr, predicate_pair_repr, sufficiency_repr], dim=-1
        )
        projected = self.projector(features)
        support_logit = self.support_head(projected).squeeze(-1)
        refute_logit = self.refute_head(projected).squeeze(-1)
        return {
            "v7_polarity_support_logit": support_logit,
            "v7_polarity_refute_logit": refute_logit,
            "v7_polarity_logits": torch.stack([refute_logit, support_logit], dim=-1),
        }


class LocationBoundaryHead(nn.Module):
    """Stage28-I-A: optional independent location-boundary cap/head for v7.

    Reads concat([frame_pair_repr, predicate_pair_repr, sufficiency_repr]) —
    the same representation family used by PolarityChannelV7.

    High location_boundary_prob → location-compatible / safe (do not suppress).
    Low location_boundary_prob → potential location mismatch; cap may reduce
    SUPPORT/REFUTE entitlement when cap mode is active.

    This head does NOT use v7_entitlement_prob, learned residuals, or hybrid mixing.
    It is bounded via an independent MLP and acts as an optional post-hoc cap only.
    """

    def __init__(
        self,
        frame_size: int,
        predicate_size: int,
        sufficiency_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = frame_size + predicate_size + sufficiency_size
        hidden = max(input_dim // 4, 16)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        frame_pair_repr: torch.Tensor,
        predicate_pair_repr: torch.Tensor,
        sufficiency_repr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = torch.cat(
            [frame_pair_repr, predicate_pair_repr, sufficiency_repr], dim=-1
        )
        logit = self.mlp(features).squeeze(-1)
        prob = torch.sigmoid(logit)
        return {
            "location_boundary_logit": logit,
            "location_boundary_prob": prob,
            "v7_location_boundary_logit": logit,
            "v7_location_boundary_prob": prob,
        }


class TemporalSafetyHead(nn.Module):
    """Stage30-C2: optional independent temporal-safety head for v7.

    Reads concat([frame_pair_repr, predicate_pair_repr, sufficiency_repr]) —
    the same representation family used by PolarityChannelV7 and LocationBoundaryHead.

    High temporal_safety_prob → temporally safe / compatible (no cap needed).
    Low temporal_safety_prob → temporal mismatch risk; soft/hard cap may reduce
    H1 entitlement_for_decision when cap mode is active.

    This head does NOT use v7_entitlement_prob or the TemporalChannelV2 path.
    It is trained via an independent BCE loss on a separate temporal safety
    diagnostic dataset (none/paraphrase/time_swap only; no Stage15).
    """

    def __init__(
        self,
        frame_size: int,
        predicate_size: int,
        sufficiency_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = frame_size + predicate_size + sufficiency_size
        hidden = max(input_dim // 4, 16)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        frame_pair_repr: torch.Tensor,
        predicate_pair_repr: torch.Tensor,
        sufficiency_repr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = torch.cat(
            [frame_pair_repr, predicate_pair_repr, sufficiency_repr], dim=-1
        )
        logit = self.mlp(features).squeeze(-1)
        prob = torch.sigmoid(logit)
        return {
            "temporal_safety_logit": logit,
            "temporal_safety_prob": prob,
            "v7_temporal_safety_logit": logit,
            "v7_temporal_safety_prob": prob,
        }


class TemporalMismatchMultiHead(nn.Module):
    """Stage30-D: representation-decomposed temporal mismatch heads.

    Three independent linear/MLP heads, each reading a single representation:
        frame_head:       reads frame_pair_repr
        predicate_head:   reads predicate_pair_repr
        sufficiency_head: reads sufficiency_repr

    Temporal mismatch positive = 1 (time_swap / stage30_temporal_safe_label=0)
    Temporal safe / control     = 0 (none, paraphrase / stage30_temporal_safe_label=1)

    This is the INVERSE of TemporalSafetyHead's convention.
    """

    def __init__(
        self,
        frame_size: int,
        predicate_size: int,
        sufficiency_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        def _make_mlp(in_dim: int) -> nn.Sequential:
            hidden = max(in_dim // 4, 16)
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        self.frame_head = _make_mlp(frame_size)
        self.predicate_head = _make_mlp(predicate_size)
        self.sufficiency_head = _make_mlp(sufficiency_size)

    def forward(
        self,
        frame_pair_repr: torch.Tensor,
        predicate_pair_repr: torch.Tensor,
        sufficiency_repr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        frame_logit = self.frame_head(frame_pair_repr).squeeze(-1)
        predicate_logit = self.predicate_head(predicate_pair_repr).squeeze(-1)
        sufficiency_logit = self.sufficiency_head(sufficiency_repr).squeeze(-1)
        return {
            "temporal_frame_mismatch_logit": frame_logit,
            "temporal_predicate_mismatch_logit": predicate_logit,
            "temporal_sufficiency_mismatch_logit": sufficiency_logit,
            "temporal_frame_mismatch_prob": torch.sigmoid(frame_logit),
            "temporal_predicate_mismatch_prob": torch.sigmoid(predicate_logit),
            "temporal_sufficiency_mismatch_prob": torch.sigmoid(sufficiency_logit),
        }


class ContraMambaV7Hierarchical(nn.Module):
    """Stage26-A: v7 Hierarchical Entitlement architecture.

    Clean v7 path — does not modify v6B behavior.

    Pipeline:
        Mamba encoder
          ↓
        FrameChannel  (reuses v5 FrameGate)
          ↓
        PredicateChannel  (reuses v5 PredicateCoverageHead, conditioned on frame/pair state)
          ↓
        SufficiencyChannel + TemporalChannel  (reuses v5 SufficiencyGate; new TemporalChannelV2)
          ↓
        EntitlementGateV7  (new learned MLP aggregation)
          ↓
        PolarityChannelV7  (new separate support/refute logits)
          ↓
        Final logit composition (REFUTE=0, NOT_ENTITLED=1, SUPPORT=2):
            support_score = entitlement_logit + polarity_support_logit
            refute_score  = entitlement_logit + polarity_refute_logit
            ne_score      = -entitlement_logit + ne_bias
            logits[B, 3]  = [refute_score, ne_score, support_score]

    Ablation flags (all False = full hierarchical model):
        v7_disable_frame_channel:     EntitlementGate sees frame_prob=1.0 (no frame signal)
        v7_disable_predicate_channel: EntitlementGate sees predicate_prob=1.0
        v7_disable_sufficiency_channel: EntitlementGate sees sufficiency_prob=1.0
        v7_disable_temporal_channel:  TemporalChannelV2 not instantiated; EntitlementGate
                                      uses 3-input MLP instead of 4-input
        v7_flat_arbiter:              EntitlementGate uses explicit product (v6B-like)
                                      instead of learned MLP
        v7_no_entitlement_polarity_conditioning:
                                      Final composition ignores entitlement; polarity
                                      logits alone determine SUPPORT/REFUTE
        v7_no_aux_losses:             Stage26-A no-op (no aux losses exist yet)

    Output keys compatible with v5 training script functions:
        logits, base_logits, predictions
        frame_logit, frame_prob, predicate_coverage_logit, predicate_coverage_prob
        sufficiency_logit, sufficiency_prob, entitlement_prob, polarity_margin
        negative_energy (alias for polarity_refute_logit), positive_energy (alias for support)
        v7_* diagnostic keys
    """

    def __init__(
        self,
        model_name: str = "state-spaces/mamba-130m-hf",
        frame_size: int = 128,
        predicate_size: int = 128,
        sufficiency_size: int = 128,
        polarity_size: int = 64,
        dropout: float = 0.1,
        freeze_a_log: bool = True,
        backbone: nn.Module | None = None,
        hidden_size: int | None = None,
        # Ablation flags — all False = full hierarchical model
        v7_disable_frame_channel: bool = False,
        v7_disable_predicate_channel: bool = False,
        v7_disable_sufficiency_channel: bool = False,
        v7_disable_temporal_channel: bool = False,
        v7_flat_arbiter: bool = False,
        v7_no_entitlement_polarity_conditioning: bool = False,
        v7_no_aux_losses: bool = False,
        # Stage26-G: initialization stabilization
        # Default -0.5 to reduce early NOT_ENTITLED attractor behavior observed in Stage26-F
        # diagnostics (entitlement_prob ≈ 0.56 across all gold classes, ne_score dominating).
        # Does NOT affect v6B. Does NOT affect architecture. Does NOT use OOD or Stage15.
        v7_initial_ne_bias: float = -0.5,
        # Stage26-H1 optional bridge. Default off; preserves existing v7 behavior.
        v7_use_v6b_style_final_decision: bool = False,
        v7_use_learnable_ne_alpha: bool = False,
        v7_ne_alpha_init: float = 1.0,
        # Stage27-H2A: configurable decision-time entitlement signal for the H1 path.
        # Only consulted when v7_use_v6b_style_final_decision=True.
        # "learned" (default) preserves existing H1 behavior: uses v7_entitlement_prob
        # from the learned EntitlementGate. Other modes replace it with an explicit
        # compositional signal from the frame/predicate/sufficiency channels.
        # Valid choices: "learned", "product", "min",
        #                "frame_predicate_product", "frame_predicate_min",
        #                "product_learned_residual"
        v7_h1_entitlement_decision_signal: str = "learned",
        # Stage27-H2B: power relaxation applied only to the "product" decision signal.
        # Default 1.0 preserves exact H2A product behavior (no-op exponent).
        # Values in (0, 1) soften the product gate (less suppression of true SUPPORT).
        # Values > 1 sharpen it further. Only consulted when _eds == "product" or
        # "product_learned_residual".
        v7_h1_entitlement_product_power: float = 1.0,
        # Stage27-H2E: residual strength for "product_learned_residual" decision signal.
        # entitlement = clamp(product_base + beta * (learned - product_base.detach()), 0, 1)
        # beta=0.0 exactly recovers product_base. Default 0.25.
        v7_h1_hybrid_residual_beta: float = 0.25,
        # Stage28-I-A: optional independent location-boundary cap/head.
        # Disabled by default; preserves all existing Stage27/Stage28 behavior.
        # Only meaningful when v7_use_v6b_style_final_decision=True (H1 path active).
        v7_use_location_boundary_head: bool = False,
        v7_location_boundary_cap_mode: str = "none",  # "none" | "hard" | "soft"
        v7_location_boundary_cap_gamma: float = 1.0,
        v7_location_boundary_cap_detach: bool = False,
        # Stage30-C2: optional independent temporal-safety cap/head.
        # Disabled by default; preserves all existing Stage28-I behavior.
        # Applied after location-boundary cap when both are enabled.
        # Only meaningful when v7_use_v6b_style_final_decision=True (H1 path active).
        v7_use_temporal_safety_head: bool = False,
        v7_temporal_safety_cap_mode: str = "none",  # "none" | "hard" | "soft"
        v7_temporal_safety_cap_gamma: float = 1.0,
        v7_temporal_safety_cap_detach: bool = False,
        # Stage30-D: representation-decomposed temporal mismatch multihead.
        # Disabled by default; separate from Stage30-C2 TemporalSafetyHead.
        # Cannot combine Stage30-D cap with Stage30-C2 cap in the same run.
        # Only meaningful when v7_use_v6b_style_final_decision=True (H1 path active).
        v7_use_temporal_mismatch_multihead: bool = False,
        v7_temporal_mismatch_multihead_cap_mode: str = "none",  # "none" | "hard" | "soft"
        v7_temporal_mismatch_multihead_cap_gamma: float = 1.0,
        v7_temporal_mismatch_multihead_cap_detach: bool = False,
        v7_temporal_mismatch_multihead_fusion: str = "frame_only",
    ) -> None:
        super().__init__()

        # Backbone
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
            raise ValueError("hidden_size is required when backbone has no config")

        if freeze_a_log:
            for name, param in self.mamba.named_parameters():
                if "A_log" in name:
                    param.requires_grad = False

        # Store ablation flags for audit/introspection
        self.v7_disable_frame_channel = v7_disable_frame_channel
        self.v7_disable_predicate_channel = v7_disable_predicate_channel
        self.v7_disable_sufficiency_channel = v7_disable_sufficiency_channel
        self.v7_disable_temporal_channel = v7_disable_temporal_channel
        self.v7_flat_arbiter = v7_flat_arbiter
        self.v7_no_entitlement_polarity_conditioning = v7_no_entitlement_polarity_conditioning
        self.v7_no_aux_losses = v7_no_aux_losses

        self.v7_use_v6b_style_final_decision = v7_use_v6b_style_final_decision
        self.v7_use_learnable_ne_alpha = v7_use_learnable_ne_alpha
        self.v7_ne_alpha_init = float(v7_ne_alpha_init)
        self.v7_h1_entitlement_decision_signal = v7_h1_entitlement_decision_signal
        self.v7_h1_entitlement_product_power = float(v7_h1_entitlement_product_power)
        self.v7_h1_hybrid_residual_beta = float(v7_h1_hybrid_residual_beta)
        if not (0.0 <= self.v7_h1_hybrid_residual_beta <= 1.0):
            raise ValueError(
                f"v7_h1_hybrid_residual_beta must be in [0.0, 1.0], "
                f"got {self.v7_h1_hybrid_residual_beta!r}."
            )

        if self.v7_use_v6b_style_final_decision and self.v7_no_entitlement_polarity_conditioning:
            raise ValueError(
                "v7_use_v6b_style_final_decision is incompatible with "
                "v7_no_entitlement_polarity_conditioning. H1 bridge requires entitlement_prob gating."
            )

        if self.v7_use_learnable_ne_alpha:
            _alpha_init = torch.tensor(max(self.v7_ne_alpha_init, 1e-6))
            self.v7_raw_ne_alpha = nn.Parameter(torch.log(torch.expm1(_alpha_init)))
        else:
            self.v7_raw_ne_alpha = None

        # Channel heads — reuse existing v5/v6B heads for FrameChannel, PredicateChannel,
        # SufficiencyChannel. TemporalChannel, EntitlementGate, and PolarityChannel are new.
        self.frame_gate = FrameGate(hidden_size, frame_size, dropout)
        self.predicate_coverage_head = PredicateCoverageHead(
            hidden_size, frame_size, predicate_size, dropout
        )
        self.sufficiency_gate = SufficiencyGate(frame_size, predicate_size, sufficiency_size, dropout)

        # TemporalChannelV2 — reads cat([claim_frame_state, evidence_frame_state])
        # Not instantiated when v7_disable_temporal_channel=True; EntitlementGate uses 3-input MLP.
        self.temporal_channel: TemporalChannelV2 | None = None
        if not v7_disable_temporal_channel:
            self.temporal_channel = TemporalChannelV2(frame_size, dropout)

        # EntitlementGateV7 — learned MLP aggregation of channel probs
        # 4-input when temporal channel active; 3-input otherwise.
        _gate_use_temporal = not v7_disable_temporal_channel
        self.entitlement_gate = EntitlementGateV7(use_temporal_input=_gate_use_temporal)

        # PolarityChannelV7 — separate support/refute logits for additive composition
        self.polarity_channel = PolarityChannelV7(
            frame_size, predicate_size, sufficiency_size, polarity_size, dropout
        )

        # Learnable NOT_ENTITLED bias (scalar)
        # Stage26-G: initialized to v7_initial_ne_bias (default -0.5) rather than 0.0.
        # Rationale: Stage26-F diagnostics showed early NE dominance (≈98% NE predictions
        # at epoch 2) with entitlement_prob uniformly ≈ 0.56 across all gold classes.
        # A modest negative initialization reduces the early NE attractor without changing
        # architecture, data, CE target, or selection behavior.
        self.v7_initial_ne_bias: float = v7_initial_ne_bias
        self.ne_bias: nn.Parameter = nn.Parameter(torch.tensor(v7_initial_ne_bias))

        # Stage28-I-A: location boundary head and cap configuration.
        _valid_cap_modes = ("none", "hard", "soft")
        if v7_location_boundary_cap_mode not in _valid_cap_modes:
            raise ValueError(
                f"v7_location_boundary_cap_mode must be one of {_valid_cap_modes}, "
                f"got {v7_location_boundary_cap_mode!r}."
            )
        if v7_location_boundary_cap_gamma <= 0:
            raise ValueError(
                f"v7_location_boundary_cap_gamma must be > 0, "
                f"got {v7_location_boundary_cap_gamma!r}."
            )
        self.v7_use_location_boundary_head = v7_use_location_boundary_head
        self.v7_location_boundary_cap_mode = v7_location_boundary_cap_mode
        self.v7_location_boundary_cap_gamma = float(v7_location_boundary_cap_gamma)
        self.v7_location_boundary_cap_detach = v7_location_boundary_cap_detach

        self.location_boundary_head: LocationBoundaryHead | None = None
        if v7_use_location_boundary_head:
            self.location_boundary_head = LocationBoundaryHead(
                frame_size, predicate_size, sufficiency_size, dropout
            )

        # Stage30-C2: temporal safety head and cap configuration.
        _valid_ts_cap_modes = ("none", "hard", "soft")
        if v7_temporal_safety_cap_mode not in _valid_ts_cap_modes:
            raise ValueError(
                f"v7_temporal_safety_cap_mode must be one of {_valid_ts_cap_modes}, "
                f"got {v7_temporal_safety_cap_mode!r}."
            )
        if v7_temporal_safety_cap_gamma <= 0:
            raise ValueError(
                f"v7_temporal_safety_cap_gamma must be > 0, "
                f"got {v7_temporal_safety_cap_gamma!r}."
            )
        self.v7_use_temporal_safety_head = v7_use_temporal_safety_head
        self.v7_temporal_safety_cap_mode = v7_temporal_safety_cap_mode
        self.v7_temporal_safety_cap_gamma = float(v7_temporal_safety_cap_gamma)
        self.v7_temporal_safety_cap_detach = v7_temporal_safety_cap_detach

        self.temporal_safety_head: TemporalSafetyHead | None = None
        if v7_use_temporal_safety_head:
            self.temporal_safety_head = TemporalSafetyHead(
                frame_size, predicate_size, sufficiency_size, dropout
            )

        # Stage30-D: temporal mismatch multihead and cap configuration.
        _valid_tmm_cap_modes = ("none", "hard", "soft")
        if v7_temporal_mismatch_multihead_cap_mode not in _valid_tmm_cap_modes:
            raise ValueError(
                f"v7_temporal_mismatch_multihead_cap_mode must be one of "
                f"{_valid_tmm_cap_modes}, got {v7_temporal_mismatch_multihead_cap_mode!r}."
            )
        if v7_temporal_mismatch_multihead_cap_gamma <= 0:
            raise ValueError(
                f"v7_temporal_mismatch_multihead_cap_gamma must be > 0, "
                f"got {v7_temporal_mismatch_multihead_cap_gamma!r}."
            )
        _valid_tmm_fusions = (
            "frame_only", "predicate_only", "sufficiency_only", "max", "noisy_or", "mean"
        )
        if v7_temporal_mismatch_multihead_fusion not in _valid_tmm_fusions:
            raise ValueError(
                f"v7_temporal_mismatch_multihead_fusion must be one of "
                f"{_valid_tmm_fusions}, got {v7_temporal_mismatch_multihead_fusion!r}."
            )
        # Conflict check: Stage30-C2 cap and Stage30-D cap cannot both be active.
        if (
            v7_temporal_safety_cap_mode != "none"
            and v7_temporal_mismatch_multihead_cap_mode != "none"
        ):
            raise ValueError(
                "Stage30-C2 temporal safety cap and Stage30-D temporal mismatch multihead cap "
                "cannot both be active simultaneously.\n"
                f"  v7_temporal_safety_cap_mode={v7_temporal_safety_cap_mode!r}\n"
                f"  v7_temporal_mismatch_multihead_cap_mode="
                f"{v7_temporal_mismatch_multihead_cap_mode!r}\n"
                "Set one cap mode to 'none' to resolve this conflict."
            )
        self.v7_use_temporal_mismatch_multihead = v7_use_temporal_mismatch_multihead
        self.v7_temporal_mismatch_multihead_cap_mode = v7_temporal_mismatch_multihead_cap_mode
        self.v7_temporal_mismatch_multihead_cap_gamma = float(v7_temporal_mismatch_multihead_cap_gamma)
        self.v7_temporal_mismatch_multihead_cap_detach = v7_temporal_mismatch_multihead_cap_detach
        self.v7_temporal_mismatch_multihead_fusion = v7_temporal_mismatch_multihead_fusion

        self.temporal_mismatch_multihead: TemporalMismatchMultiHead | None = None
        if v7_use_temporal_mismatch_multihead:
            self.temporal_mismatch_multihead = TemporalMismatchMultiHead(
                frame_size, predicate_size, sufficiency_size, dropout
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Forward pass for v7 hierarchical model.

        v6B-specific kwargs (temporal_mismatch_flags, predicate_mismatch_flags,
        temporal_adapter_final_penalty_scale, temporal_channel_gated_penalty_scale,
        etc.) are accepted via **kwargs and silently ignored — v7 does not use them.

        encoder_hidden_states: optional cached encoder output (from frozen-encoder
        optimization in the training script). If provided, skips the Mamba forward.
        """
        del kwargs  # v6B-specific; unused in v7

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

        # ── FrameChannel ────────────────────────────────────────────────────────
        # Reuses v5 FrameGate: claim/evidence slot states + pair repr + frame_prob.
        # claim_frame_state, evidence_frame_state: pre-pair-projector slot states
        # frame_pair_repr: post-pair-projector joint frame representation
        frame = self.frame_gate(token_states, attention_mask, claim_mask, evidence_mask)

        if self.v7_disable_frame_channel:
            # Ablation: EntitlementGate sees frame_prob=1.0 (no frame signal).
            # The FrameGate still runs so downstream heads can use frame_pair_repr
            # and pre-projector states. Only the scalar probability is overridden.
            _gate_frame_prob = torch.ones_like(frame["frame_prob"])
        else:
            _gate_frame_prob = frame["frame_prob"]

        # ── PredicateChannel ────────────────────────────────────────────────────
        # Reuses v5 PredicateCoverageHead, conditioned on frame state and pair repr.
        predicate = self.predicate_coverage_head(
            token_states=token_states,
            attention_mask=attention_mask,
            claim_mask=claim_mask,
            evidence_mask=evidence_mask,
            claim_frame_state=frame["claim_frame_state"],
            evidence_frame_state=frame["evidence_frame_state"],
            frame_pair_repr=frame["frame_pair_repr"],
            frame_prob=_gate_frame_prob,
        )

        if self.v7_disable_predicate_channel:
            # Ablation: EntitlementGate sees predicate_prob=1.0 (no predicate signal).
            _gate_predicate_prob = torch.ones_like(predicate["predicate_coverage_prob"])
        else:
            _gate_predicate_prob = predicate["predicate_coverage_prob"]

        # ── SufficiencyChannel ──────────────────────────────────────────────────
        # Reuses v5 SufficiencyGate, conditioned on frame and predicate pair reprs.
        sufficiency = self.sufficiency_gate(
            frame_pair_repr=frame["frame_pair_repr"],
            predicate_pair_repr=predicate["predicate_pair_repr"],
            frame_prob=_gate_frame_prob,
            predicate_coverage_prob=_gate_predicate_prob,
        )

        if self.v7_disable_sufficiency_channel:
            # Ablation: EntitlementGate sees sufficiency_prob=1.0.
            _gate_sufficiency_prob = torch.ones_like(sufficiency["sufficiency_prob"])
        else:
            _gate_sufficiency_prob = sufficiency["sufficiency_prob"]

        # ── LocationBoundaryHead (Stage28-I-A) ──────────────────────────────────
        # Reads concat([frame_pair_repr, predicate_pair_repr, sufficiency_repr]).
        # Produces location_boundary_prob in [0,1]: high = location-safe (no cap),
        # low = potential location mismatch (cap reduces entitlement_for_decision).
        # Always None when head is disabled (default); cap is also silently inactive.
        v7_location_boundary_logit: torch.Tensor | None = None
        v7_location_boundary_prob: torch.Tensor | None = None
        if self.location_boundary_head is not None:
            _lb_out = self.location_boundary_head(
                frame_pair_repr=frame["frame_pair_repr"],
                predicate_pair_repr=predicate["predicate_pair_repr"],
                sufficiency_repr=sufficiency["sufficiency_repr"],
            )
            v7_location_boundary_logit = _lb_out["v7_location_boundary_logit"]
            v7_location_boundary_prob = _lb_out["v7_location_boundary_prob"]

        # ── TemporalChannel ─────────────────────────────────────────────────────
        # Reads cat([claim_frame_state, evidence_frame_state]) — pre-pair-projector slot
        # states from FrameGate's project() step. NOT frame_pair_repr (post-projector).
        # Disabled (None) when v7_disable_temporal_channel=True.
        # In Stage26-A, no temporal auxiliary loss is active; temporal channel is trained
        # only through CE loss via the EntitlementGate composition.
        v7_temporal_logit: torch.Tensor | None = None
        v7_temporal_prob: torch.Tensor | None = None
        if self.temporal_channel is not None:
            tc_out = self.temporal_channel(
                frame["claim_frame_state"], frame["evidence_frame_state"]
            )
            v7_temporal_logit = tc_out["v7_temporal_logit"]
            v7_temporal_prob = tc_out["v7_temporal_prob"]

        # ── EntitlementGate ─────────────────────────────────────────────────────
        # Aggregates pre-entitlement channel signals into a scalar entitlement logit.
        # Temporal probability is provided when temporal channel is active.
        ent_out = self.entitlement_gate(
            frame_prob=_gate_frame_prob,
            predicate_coverage_prob=_gate_predicate_prob,
            sufficiency_prob=_gate_sufficiency_prob,
            temporal_prob=v7_temporal_prob,
            flat_arbiter=self.v7_flat_arbiter,
        )
        v7_entitlement_logit = ent_out["v7_entitlement_logit"]
        v7_entitlement_prob = ent_out["v7_entitlement_prob"]

        # ── PolarityChannel ─────────────────────────────────────────────────────
        # Reads from combined frame/predicate/sufficiency representations.
        # Outputs raw logits (not softplus energies) for additive final composition.
        pol_out = self.polarity_channel(
            frame_pair_repr=frame["frame_pair_repr"],
            predicate_pair_repr=predicate["predicate_pair_repr"],
            sufficiency_repr=sufficiency["sufficiency_repr"],
        )
        v7_polarity_support = pol_out["v7_polarity_support_logit"]
        v7_polarity_refute = pol_out["v7_polarity_refute_logit"]

        # ── TemporalSafetyHead (Stage30-C2) ─────────────────────────────────────────────────
        # Independent head reading concat([frame_pair_repr, predicate_pair_repr, sufficiency_repr]).
        # High prob = temporally safe; low prob = temporal mismatch risk.
        # None when head is disabled (default).
        temporal_safety_logit: torch.Tensor | None = None
        temporal_safety_prob: torch.Tensor | None = None
        if self.temporal_safety_head is not None:
            _ts_out = self.temporal_safety_head(
                frame_pair_repr=frame["frame_pair_repr"],
                predicate_pair_repr=predicate["predicate_pair_repr"],
                sufficiency_repr=sufficiency["sufficiency_repr"],
            )
            temporal_safety_logit = _ts_out["temporal_safety_logit"]
            temporal_safety_prob = _ts_out["temporal_safety_prob"]

        # ── Stage30-D: temporal mismatch multihead ──────────────────────────────────────────────
        # Three independent heads reading separate representations.
        # Fusion mode and cap are applied in the H1 path only.
        temporal_frame_mismatch_logit: torch.Tensor | None = None
        temporal_predicate_mismatch_logit: torch.Tensor | None = None
        temporal_sufficiency_mismatch_logit: torch.Tensor | None = None
        temporal_frame_mismatch_prob: torch.Tensor | None = None
        temporal_predicate_mismatch_prob: torch.Tensor | None = None
        temporal_sufficiency_mismatch_prob: torch.Tensor | None = None
        temporal_mismatch_fused_prob: torch.Tensor | None = None
        temporal_mismatch_safe_factor: torch.Tensor | None = None
        v7_h1_entitlement_before_temporal_mismatch_cap: torch.Tensor | None = None
        v7_h1_entitlement_after_temporal_mismatch_cap: torch.Tensor | None = None

        if self.temporal_mismatch_multihead is not None:
            _tmm_out = self.temporal_mismatch_multihead(
                frame_pair_repr=frame["frame_pair_repr"],
                predicate_pair_repr=predicate["predicate_pair_repr"],
                sufficiency_repr=sufficiency["sufficiency_repr"],
            )
            temporal_frame_mismatch_logit = _tmm_out["temporal_frame_mismatch_logit"]
            temporal_predicate_mismatch_logit = _tmm_out["temporal_predicate_mismatch_logit"]
            temporal_sufficiency_mismatch_logit = _tmm_out["temporal_sufficiency_mismatch_logit"]
            temporal_frame_mismatch_prob = _tmm_out["temporal_frame_mismatch_prob"]
            temporal_predicate_mismatch_prob = _tmm_out["temporal_predicate_mismatch_prob"]
            temporal_sufficiency_mismatch_prob = _tmm_out["temporal_sufficiency_mismatch_prob"]
            # Fusion: combine per-head mismatch probabilities
            _p_f = temporal_frame_mismatch_prob
            _p_p = temporal_predicate_mismatch_prob
            _p_s = temporal_sufficiency_mismatch_prob
            _fusion = self.v7_temporal_mismatch_multihead_fusion
            if _fusion == "frame_only":
                temporal_mismatch_fused_prob = _p_f
            elif _fusion == "predicate_only":
                temporal_mismatch_fused_prob = _p_p
            elif _fusion == "sufficiency_only":
                temporal_mismatch_fused_prob = _p_s
            elif _fusion == "max":
                temporal_mismatch_fused_prob = torch.maximum(torch.maximum(_p_f, _p_p), _p_s)
            elif _fusion == "mean":
                temporal_mismatch_fused_prob = (_p_f + _p_p + _p_s) / 3.0
            elif _fusion == "noisy_or":
                temporal_mismatch_fused_prob = 1.0 - (1.0 - _p_f) * (1.0 - _p_p) * (1.0 - _p_s)
            # safe_factor = complement of mismatch probability
            temporal_mismatch_safe_factor = 1.0 - temporal_mismatch_fused_prob

        # ── Stage28-I-A: location boundary cap tracking (initialized before H1/non-H1 branch) ──
        v7_h1_entitlement_before_location_cap: torch.Tensor | None = None
        v7_h1_entitlement_after_location_cap: torch.Tensor | None = None
        v7_location_boundary_prob_for_cap: torch.Tensor | None = None

        # ── Stage30-C2: temporal safety cap tracking (initialized before H1/non-H1 branch) ──
        v7_h1_entitlement_before_temporal_cap: torch.Tensor | None = None
        v7_h1_entitlement_after_temporal_cap: torch.Tensor | None = None
        v7_temporal_safety_prob_for_cap: torch.Tensor | None = None

        # ── Final logit composition ──────────────────────────────────────────────
        # Class order matches FinalLabel in src/contramamba/labels.py:
        #   dim 0 = REFUTE        (FinalLabel.REFUTE        = 0)
        #   dim 1 = NOT_ENTITLED  (FinalLabel.NOT_ENTITLED  = 1)
        #   dim 2 = SUPPORT       (FinalLabel.SUPPORT       = 2)
        # The stack order [refute_score, ne_score, support_score] MUST NOT be
        # reordered — CE receives integer labels encoded by FinalLabel directly.
        #
        # Hierarchical (default): entitlement logit gates polarity contribution.
        #   When entitlement is high (positive), polarity can push to SUPPORT/REFUTE.
        #   When entitlement is low (negative), ne_score rises and polarity is suppressed.
        #
        # Flat (v7_no_entitlement_polarity_conditioning): polarity alone decides
        #   SUPPORT/REFUTE; NE score uses fixed ne_bias. Ablation reference only.
        if self.v7_use_v6b_style_final_decision:
            # Stage26-H1 bridge:
            # Preserve v7 hierarchy, but restore v6B-style decision geometry.
            # Raw support/refute logits remain available as diagnostics.
            positive_energy = F.softplus(v7_polarity_support)
            negative_energy = F.softplus(v7_polarity_refute)

            # Stage27-H2A: select decision-time entitlement signal.
            # "learned" (default) uses v7_entitlement_prob from the EntitlementGate.
            # Other modes use explicit compositional signals from the channel probabilities.
            # All channel probabilities are sigmoid outputs already in [0, 1], so no clamping
            # is required. The signal is kept attached (no detach) so gradients can flow.
            _eds = self.v7_h1_entitlement_decision_signal
            _fp = frame["frame_prob"]
            _pp = predicate["predicate_coverage_prob"]
            _sp = sufficiency["sufficiency_prob"]
            if _eds == "learned":
                entitlement_for_decision = v7_entitlement_prob
            elif _eds == "product":
                _raw_product = _fp * _pp * _sp
                _pwr = self.v7_h1_entitlement_product_power
                if _pwr == 1.0:
                    entitlement_for_decision = _raw_product
                else:
                    entitlement_for_decision = _raw_product.clamp_min(1e-8).pow(_pwr)
            elif _eds == "min":
                entitlement_for_decision = torch.minimum(
                    torch.minimum(_fp, _pp), _sp
                )
            elif _eds == "frame_predicate_product":
                entitlement_for_decision = _fp * _pp
            elif _eds == "frame_predicate_min":
                entitlement_for_decision = torch.minimum(_fp, _pp)
            elif _eds == "product_learned_residual":
                # Stage27-H2E: product base stays differentiable through frame/predicate/
                # sufficiency paths; learned stays differentiable through EntitlementGate.
                # Only the subtraction anchor (product_base.detach()) is stopped.
                _raw_product = _fp * _pp * _sp
                _pwr = self.v7_h1_entitlement_product_power
                if _pwr == 1.0:
                    _product_base = _raw_product
                else:
                    _product_base = _raw_product.clamp_min(1e-8).pow(_pwr)
                _beta = self.v7_h1_hybrid_residual_beta
                _residual = v7_entitlement_prob - _product_base.detach()
                entitlement_for_decision = (
                    _product_base + _beta * _residual
                ).clamp(0.0, 1.0)
            else:
                raise ValueError(
                    f"Unknown v7_h1_entitlement_decision_signal: {_eds!r}. "
                    "Valid choices: 'learned', 'product', 'min', "
                    "'frame_predicate_product', 'frame_predicate_min', "
                    "'product_learned_residual'."
                )

            # ── Stage28-I-A: location boundary cap ──────────────────────────────
            # Applied after the decision signal is selected. Cap mode "none" is
            # a guaranteed no-op; the branch is skipped entirely.
            # "hard": entitlement = min(entitlement, location_boundary_prob)
            # "soft": entitlement = entitlement * location_boundary_prob^gamma
            if (
                v7_location_boundary_prob is not None
                and self.v7_location_boundary_cap_mode != "none"
            ):
                v7_h1_entitlement_before_location_cap = entitlement_for_decision
                _cap_prob = v7_location_boundary_prob
                if self.v7_location_boundary_cap_detach:
                    _cap_prob = _cap_prob.detach()
                v7_location_boundary_prob_for_cap = _cap_prob
                if self.v7_location_boundary_cap_mode == "hard":
                    entitlement_for_decision = torch.minimum(entitlement_for_decision, _cap_prob)
                else:  # soft
                    _gamma = self.v7_location_boundary_cap_gamma
                    entitlement_for_decision = (
                        entitlement_for_decision * _cap_prob.clamp_min(1e-8).pow(_gamma)
                    )
                v7_h1_entitlement_after_location_cap = entitlement_for_decision

            # ── Stage30-C2: temporal safety cap ────────────────────────────────
            # Applied after location-boundary cap. Cap mode "none" is a no-op.
            # "hard": entitlement = min(entitlement, temporal_safety_prob)
            # "soft": entitlement = entitlement * temporal_safety_prob^gamma
            if (
                temporal_safety_prob is not None
                and self.v7_temporal_safety_cap_mode != "none"
            ):
                v7_h1_entitlement_before_temporal_cap = entitlement_for_decision
                _ts_cap_prob = temporal_safety_prob
                if self.v7_temporal_safety_cap_detach:
                    _ts_cap_prob = _ts_cap_prob.detach()
                v7_temporal_safety_prob_for_cap = _ts_cap_prob
                if self.v7_temporal_safety_cap_mode == "hard":
                    entitlement_for_decision = torch.minimum(
                        entitlement_for_decision, _ts_cap_prob
                    )
                else:  # soft
                    _ts_gamma = self.v7_temporal_safety_cap_gamma
                    entitlement_for_decision = (
                        entitlement_for_decision * _ts_cap_prob.clamp_min(1e-8).pow(_ts_gamma)
                    )
                v7_h1_entitlement_after_temporal_cap = entitlement_for_decision

            # ── Stage30-D: temporal mismatch multihead cap ────────────────────────────────
            # Applied after Stage30-C2 temporal safety cap (the two caps are mutually exclusive
            # at construction time, so at most one is ever active here).
            # safe_factor = 1 - fused_mismatch_prob.
            # "hard": entitlement = min(entitlement, safe_factor)
            # "soft": entitlement = entitlement * safe_factor.clamp_min(1e-8).pow(gamma)
            if (
                temporal_mismatch_safe_factor is not None
                and self.v7_temporal_mismatch_multihead_cap_mode != "none"
            ):
                v7_h1_entitlement_before_temporal_mismatch_cap = entitlement_for_decision
                _tmm_safe = temporal_mismatch_safe_factor
                if self.v7_temporal_mismatch_multihead_cap_detach:
                    _tmm_safe = _tmm_safe.detach()
                if self.v7_temporal_mismatch_multihead_cap_mode == "hard":
                    entitlement_for_decision = torch.minimum(
                        entitlement_for_decision, _tmm_safe
                    )
                else:  # soft
                    _tmm_gamma = self.v7_temporal_mismatch_multihead_cap_gamma
                    entitlement_for_decision = (
                        entitlement_for_decision * _tmm_safe.clamp_min(1e-8).pow(_tmm_gamma)
                    )
                v7_h1_entitlement_after_temporal_mismatch_cap = entitlement_for_decision

            support_score = entitlement_for_decision * positive_energy
            refute_score = entitlement_for_decision * negative_energy

            if self.v7_use_learnable_ne_alpha:
                if self.v7_raw_ne_alpha is None:
                    raise RuntimeError("v7_raw_ne_alpha is None while learnable NE alpha is enabled.")
                v7_ne_alpha = F.softplus(self.v7_raw_ne_alpha)
                ne_score = self.ne_bias + v7_ne_alpha * (1.0 - entitlement_for_decision)
                v7_ne_score_mode = "bias_plus_learnable_alpha_times_one_minus_entitlement_prob"
            else:
                v7_ne_alpha = None
                ne_score = self.ne_bias + (1.0 - entitlement_for_decision)
                v7_ne_score_mode = "bias_plus_one_minus_entitlement_prob"

            v7_final_logit_composition = "v6b_style_softplus_multiplicative"
            v7_polarity_energy_mode = "softplus_energy"
            v7_entitlement_decision_signal = _eds

        else:
            # Existing v7 behavior. Keep default semantics unchanged.
            if self.v7_no_entitlement_polarity_conditioning:
                support_score = v7_polarity_support
                refute_score = v7_polarity_refute
                ne_score = self.ne_bias.expand(v7_entitlement_logit.shape)

                v7_final_logit_composition = "flat"
                v7_ne_score_mode = "fixed_ne_bias"
                v7_entitlement_decision_signal = "none"
            else:
                support_score = v7_entitlement_logit + v7_polarity_support
                refute_score = v7_entitlement_logit + v7_polarity_refute
                ne_score = -v7_entitlement_logit + self.ne_bias

                v7_final_logit_composition = "hierarchical_additive"
                v7_ne_score_mode = "negative_entitlement_logit_plus_bias"
                v7_entitlement_decision_signal = "entitlement_logit"

            positive_energy = v7_polarity_support
            negative_energy = v7_polarity_refute
            v7_ne_alpha = None
            v7_polarity_energy_mode = "raw_logits"

        final_logits = torch.stack([refute_score, ne_score, support_score], dim=-1)
        polarity_margin = positive_energy - negative_energy

        # base_logits semantics in v7 Stage26-A/B:
        #   output["logits"]      — final logits used by CE loss.  INVIOLABLE.
        #   output["base_logits"] — compatibility alias only; equals logits here.
        #   In v7 Stage26-A/B there is no separate base projection, so both tensors
        #   are identical.  Do NOT treat base_logits as independent model evidence
        #   in v7 — it carries no additional diagnostic information in this stage.
        #   A future Stage26-C+ may introduce a separate baseline projection for
        #   ablation comparison, at which point base_logits would diverge from logits.
        base_logits = final_logits

        # ── Compatibility aliases ────────────────────────────────────────────────
        # These keys are required by v5.controlled_losses, v5.compute_metrics, and
        # v5.intervention_diagnostics. Do NOT remove or rename.
        #   positive_energy / negative_energy → polarity support/refute logits (aliases)
        #   polarity_margin → support_logit - refute_logit (direction signal)
        #   entitlement_prob → v7_entitlement_prob (gating probability)

        return {
            # ── Core output contract (inviolable) ──────────────────────────────
            "logits": final_logits,               # CE uses this
            "base_logits": base_logits,            # diagnostic alias (= final_logits)
            "predictions": final_logits.argmax(dim=-1),
            # ── v5 compatibility keys ──────────────────────────────────────────
            "frame_logit": frame["frame_logit"],
            "frame_prob": frame["frame_prob"],
            "claim_frame_state": frame["claim_frame_state"],
            "evidence_frame_state": frame["evidence_frame_state"],
            "frame_pair_repr": frame["frame_pair_repr"],
            "predicate_coverage_logit": predicate["predicate_coverage_logit"],
            "predicate_coverage_prob": predicate["predicate_coverage_prob"],
            "predicate_pair_repr": predicate["predicate_pair_repr"],
            "sufficiency_logit": sufficiency["sufficiency_logit"],
            "sufficiency_prob": sufficiency["sufficiency_prob"],
            "sufficiency_repr": sufficiency["sufficiency_repr"],
            "entitlement_prob": v7_entitlement_prob,  # required by intervention_diagnostics
            "positive_energy": positive_energy,        # required by controlled_losses polarity
            "negative_energy": negative_energy,        # required by controlled_losses polarity
            "polarity_margin": polarity_margin,        # required by compute_metrics
            # ── v7 diagnostic channel outputs ─────────────────────────────────
            "v7_frame_logit": frame["frame_logit"],
            "v7_frame_prob": frame["frame_prob"],
            "v7_predicate_logit": predicate["predicate_coverage_logit"],
            "v7_predicate_prob": predicate["predicate_coverage_prob"],
            "v7_sufficiency_logit": sufficiency["sufficiency_logit"],
            "v7_sufficiency_prob": sufficiency["sufficiency_prob"],
            "v7_temporal_logit": v7_temporal_logit,
            "v7_temporal_prob": v7_temporal_prob,
            "v7_entitlement_logit": v7_entitlement_logit,
            "v7_entitlement_prob": v7_entitlement_prob,
            "v7_polarity_support_logit": v7_polarity_support,
            "v7_polarity_refute_logit": v7_polarity_refute,
            "v7_polarity_logits": pol_out["v7_polarity_logits"],
            "v7_channel_output_keys": [
                "v7_frame_logit", "v7_frame_prob",
                "v7_predicate_logit", "v7_predicate_prob",
                "v7_sufficiency_logit", "v7_sufficiency_prob",
                "v7_temporal_logit", "v7_temporal_prob",
                "v7_entitlement_logit", "v7_entitlement_prob",
                "v7_polarity_support_logit", "v7_polarity_refute_logit",
            ],
            "v7_final_logit_composition": v7_final_logit_composition,
            "v7_polarity_energy_mode": v7_polarity_energy_mode,
            "v7_entitlement_decision_signal": v7_entitlement_decision_signal,
            "v7_ne_score_mode": v7_ne_score_mode,
            "v7_ne_alpha": v7_ne_alpha,
            # Stage27-H2A: H1-path decision signal diagnostics.
            # v7_h1_entitlement_for_decision is the actual tensor used as
            # entitlement_for_decision in the H1 path (None when H1 is inactive).
            # When Stage28-I-A cap is active, this is the post-cap value.
            "v7_h1_entitlement_for_decision": (
                entitlement_for_decision if self.v7_use_v6b_style_final_decision else None
            ),
            "v7_h1_entitlement_decision_signal": self.v7_h1_entitlement_decision_signal,
            "v7_h1_entitlement_product_power": self.v7_h1_entitlement_product_power,
            "v7_h1_hybrid_residual_beta": self.v7_h1_hybrid_residual_beta,

            "v7_polarity_positive_energy": positive_energy,
            "v7_polarity_negative_energy": negative_energy,
            "v7_polarity_energy_margin": polarity_margin,

            # Stage28-I-A: location boundary head outputs (None when head is disabled).
            "location_boundary_logit": v7_location_boundary_logit,
            "location_boundary_prob": v7_location_boundary_prob,
            "v7_location_boundary_logit": v7_location_boundary_logit,
            "v7_location_boundary_prob": v7_location_boundary_prob,
            # Cap tracking (None when cap mode is "none" or H1 is inactive).
            "v7_location_boundary_cap_mode": self.v7_location_boundary_cap_mode,
            "v7_location_boundary_cap_gamma": self.v7_location_boundary_cap_gamma,
            "v7_location_boundary_prob_for_cap": v7_location_boundary_prob_for_cap,
            "v7_h1_entitlement_before_location_cap": v7_h1_entitlement_before_location_cap,
            "v7_h1_entitlement_after_location_cap": v7_h1_entitlement_after_location_cap,
            # Stage30-C2: temporal safety head outputs (None when head is disabled).
            "temporal_safety_logit": temporal_safety_logit,
            "temporal_safety_prob": temporal_safety_prob,
            "v7_temporal_safety_logit": temporal_safety_logit,
            "v7_temporal_safety_prob": temporal_safety_prob,
            # Temporal safety cap tracking (None when cap mode is "none" or H1 is inactive).
            "v7_temporal_safety_cap_mode": self.v7_temporal_safety_cap_mode,
            "v7_temporal_safety_cap_gamma": self.v7_temporal_safety_cap_gamma,
            "v7_temporal_safety_prob_for_cap": v7_temporal_safety_prob_for_cap,
            "v7_h1_entitlement_before_temporal_cap": v7_h1_entitlement_before_temporal_cap,
            "v7_h1_entitlement_after_temporal_cap": v7_h1_entitlement_after_temporal_cap,
            # Stage30-D: temporal mismatch multihead outputs (None when head is disabled).
            "temporal_frame_mismatch_logit": temporal_frame_mismatch_logit,
            "temporal_predicate_mismatch_logit": temporal_predicate_mismatch_logit,
            "temporal_sufficiency_mismatch_logit": temporal_sufficiency_mismatch_logit,
            "temporal_frame_mismatch_prob": temporal_frame_mismatch_prob,
            "temporal_predicate_mismatch_prob": temporal_predicate_mismatch_prob,
            "temporal_sufficiency_mismatch_prob": temporal_sufficiency_mismatch_prob,
            "temporal_mismatch_fused_prob": temporal_mismatch_fused_prob,
            "temporal_mismatch_safe_factor": temporal_mismatch_safe_factor,
            # Stage30-D cap tracking (None when cap mode is "none" or H1 is inactive).
            "v7_temporal_mismatch_multihead_cap_mode": self.v7_temporal_mismatch_multihead_cap_mode,
            "v7_temporal_mismatch_multihead_cap_gamma": self.v7_temporal_mismatch_multihead_cap_gamma,
            "v7_temporal_mismatch_multihead_fusion": self.v7_temporal_mismatch_multihead_fusion,
            "v7_h1_entitlement_before_temporal_mismatch_cap": v7_h1_entitlement_before_temporal_mismatch_cap,
            "v7_h1_entitlement_after_temporal_mismatch_cap": v7_h1_entitlement_after_temporal_mismatch_cap,
        }
