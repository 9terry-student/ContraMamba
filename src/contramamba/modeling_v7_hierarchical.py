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
        # Initialized to 0.0; the model learns to shift NE score as needed.
        self.ne_bias: nn.Parameter = nn.Parameter(torch.zeros(()))

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
        if self.v7_no_entitlement_polarity_conditioning:
            support_score = v7_polarity_support
            refute_score = v7_polarity_refute
            ne_score = self.ne_bias.expand(v7_entitlement_logit.shape)
        else:
            support_score = v7_entitlement_logit + v7_polarity_support
            refute_score = v7_entitlement_logit + v7_polarity_refute
            ne_score = -v7_entitlement_logit + self.ne_bias

        final_logits = torch.stack([refute_score, ne_score, support_score], dim=-1)

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
        positive_energy = v7_polarity_support
        negative_energy = v7_polarity_refute
        polarity_margin = v7_polarity_support - v7_polarity_refute

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
            "v7_final_logit_composition": (
                "flat"
                if self.v7_no_entitlement_polarity_conditioning
                else "hierarchical_additive"
            ),
        }
