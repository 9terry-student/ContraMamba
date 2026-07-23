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

STAGE196B2B6P8_CANDIDATE_MASKS = (
    "00100000000000",
    "01000000000000",
    "10000000000000",
)
STAGE196B2B6P8_PRIMITIVE_KEYS = (
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "positive_energy",
    "negative_energy",
)
STAGE196B2B6P8_CLASS_ORDER = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
STAGE196B2B6P8_RNG_POLICY = (
    "MATCH_NATIVE_AND_COUNTERPART_DOWNSTREAM_RESTORE_POST_NATIVE"
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
        use_temporal_channel: bool = False,
        temporal_channel_detach_input: bool = True,
        use_temporal_channel_loss: bool = False,
        temporal_channel_loss_weight: float = 0.0,
        temporal_channel_loss_pos_weight: float = 1.0,
        use_temporal_channel_gated_penalty: bool = False,
        temporal_channel_gated_penalty_scale: float = 0.0,
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

        # TemporalChannel V1 — reads cat([claim_frame_state, evidence_frame_state]).
        # These are the pre-pair-projector pooled slot states from FrameGate's first projection
        # (Linear(hidden_size, frame_size) + pool), NOT frame_pair_repr (which is post-pair-
        # projector). This separates the temporal signal channel from the FrameGate composition
        # path, addressing the Stage23/24 gradient coupling problem.
        # With detach=True (default), TC BCE loss cannot propagate into FrameGate parameters.
        # Architecture: Linear(frame_size*2, tc_hidden) → GELU → Linear(tc_hidden, 1)
        # Optional gated penalty: scale * sigmoid(tc_logit).detach() * (1 - pe_prob).detach()
        # Requires preservation_entitlement_head when gated penalty is used.
        self.temporal_channel_v1: nn.Sequential | None = None
        self.temporal_channel_detach_input: bool = temporal_channel_detach_input
        if use_temporal_channel:
            _tc_hidden = max(frame_size // 2, 8)
            self.temporal_channel_v1 = nn.Sequential(
                nn.Linear(frame_size * 2, _tc_hidden),
                nn.GELU(),
                nn.Linear(_tc_hidden, 1),
            )

        # Store TC configuration for self-description and audit ledger.
        # These are the canonical values; the training script reads them via getattr(model, ...).
        # They do NOT change model forward behavior (which is controlled by explicit forward args);
        # they exist so the model is self-describing about its intended TC configuration.
        self.use_temporal_channel: bool = use_temporal_channel
        self.use_temporal_channel_loss: bool = use_temporal_channel_loss
        self.temporal_channel_loss_weight: float = temporal_channel_loss_weight
        self.temporal_channel_loss_pos_weight: float = temporal_channel_loss_pos_weight
        self.use_temporal_channel_gated_penalty: bool = use_temporal_channel_gated_penalty
        self.temporal_channel_gated_penalty_scale: float = temporal_channel_gated_penalty_scale

        # Constructor-time validation: catch impossible TC configurations immediately.
        if use_temporal_channel_loss and not use_temporal_channel:
            raise ValueError(
                "use_temporal_channel_loss=True requires use_temporal_channel=True. "
                "The TC head must be instantiated before its loss can be enabled."
            )
        if use_temporal_channel_gated_penalty and not use_temporal_channel:
            raise ValueError(
                "use_temporal_channel_gated_penalty=True requires use_temporal_channel=True. "
                "The gated penalty reads temporal_channel_logit, which requires the TC head."
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

    @staticmethod
    def _stage196b2b6p8_capture_rng_state() -> dict[str, Any]:
        """Capture CPU and all CUDA generator states for matched dropout."""
        return {
            "cpu_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_states": tuple(
                state.clone() for state in torch.cuda.get_rng_state_all()
            ) if torch.cuda.is_available() else (),
        }

    @staticmethod
    def _stage196b2b6p8_restore_rng_state(state: dict[str, Any]) -> None:
        cpu_state = state.get("cpu_rng_state")
        cuda_states = state.get("cuda_rng_states")
        if not isinstance(cpu_state, torch.Tensor):
            raise ValueError("Stage196-B2-B6P8 CPU RNG state is missing")
        if not isinstance(cuda_states, tuple) or not all(
            isinstance(item, torch.Tensor) for item in cuda_states
        ):
            raise ValueError("Stage196-B2-B6P8 CUDA RNG states are invalid")
        torch.set_rng_state(cpu_state)
        if cuda_states:
            if not torch.cuda.is_available():
                raise ValueError("captured CUDA RNG state requires CUDA")
            torch.cuda.set_rng_state_all(list(cuda_states))

    @staticmethod
    def _stage196b2b6p8_geometry(
        logits: torch.Tensor, *, prefix: str,
    ) -> dict[str, torch.Tensor]:
        """Use P2/P6 class order and the existing torch.topk tie semantics."""
        if logits.ndim != 2 or logits.shape[1] != len(STAGE196B2B6P8_CLASS_ORDER):
            raise ValueError(f"{prefix} logits must have shape [batch, 3]")
        refute, not_entitled, support = logits.unbind(dim=-1)
        top_two = torch.topk(
            logits, k=2, dim=-1, largest=True, sorted=True
        ).values
        return {
            f"{prefix}_score_support": support,
            f"{prefix}_score_not_entitled": not_entitled,
            f"{prefix}_score_refute": refute,
            f"{prefix}_margin_support_minus_not_entitled": (
                support - not_entitled
            ),
            f"{prefix}_margin_support_minus_refute": support - refute,
            f"{prefix}_margin_refute_minus_not_entitled": (
                refute - not_entitled
            ),
            f"{prefix}_top1_runner_up_margin": top_two[:, 0] - top_two[:, 1],
        }

    def replay_full_trainable_path(
        self,
        replay_state: dict[str, torch.Tensor],
        candidate_mask: dict[str, torch.Tensor],
        *,
        gradient_ownership_mode: str,
        stochastic_context: dict[str, Any],
        native_output: dict[str, Any],
        counterpart_model: "ContraMambaV6BMinimal",
        candidate_action_keys: dict[str, tuple[str, ...]],
    ) -> dict[str, Any]:
        """Replay the P7 full trainable path from one shared Mamba state.

        The 14-bit dictionary keys are opaque candidate identities. Their
        row-wise five-bit actions and action keys must come from the P7
        candidate semantic trace; this method never interprets candidate IDs.
        """
        if gradient_ownership_mode not in {"joint", "frame_local_only"}:
            raise ValueError("unsupported gradient ownership mode")
        if counterpart_model is self:
            raise ValueError("counterpart_model must be the separately loaded arm")
        if tuple(candidate_mask) != STAGE196B2B6P8_CANDIDATE_MASKS:
            raise ValueError("Stage196-B2-B6P8 requires exact candidate order")
        if tuple(candidate_action_keys) != STAGE196B2B6P8_CANDIDATE_MASKS:
            raise ValueError("candidate action-key closure is incomplete")
        required_state = (
            "encoder_hidden_states",
            "attention_mask",
            "claim_mask",
            "evidence_mask",
        )
        if tuple(replay_state)[:len(required_state)] != required_state:
            raise ValueError("Stage196-B2-B6P8 replay-state schema drift")
        tensors = [replay_state[name] for name in required_state]
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError("replay state may contain tensors only")
        hidden, attention, claim, evidence = tensors
        if hidden.ndim != 3 or any(
            mask.shape != hidden.shape[:2] for mask in (attention, claim, evidence)
        ):
            raise ValueError("replay-state batch/sequence dimensions disagree")
        if hidden.device.type != "cuda" or any(
            tensor.device != hidden.device for tensor in tensors
        ):
            raise ValueError("replay state must remain aligned on CUDA")
        if not hidden.is_floating_point():
            raise ValueError("encoder_hidden_states must retain floating dtype")
        if stochastic_context.get("rng_policy") != STAGE196B2B6P8_RNG_POLICY:
            raise ValueError("P7 stochastic-state policy mismatch")
        before_rng = stochastic_context.get("native_pre_downstream_rng")
        after_rng = stochastic_context.get("native_post_downstream_rng")
        if not isinstance(before_rng, dict) or not isinstance(after_rng, dict):
            raise ValueError("matched native RNG states are required")

        native_logits = native_output.get("logits")
        native_base_logits = native_output.get("base_logits")
        if not isinstance(native_logits, torch.Tensor) or not isinstance(
            native_base_logits, torch.Tensor
        ):
            raise ValueError("native output must contain live final/base logits")
        if native_logits.shape != native_base_logits.shape:
            raise ValueError("native final/base logit shapes disagree")
        batch_size = hidden.shape[0]
        if native_logits.shape != (batch_size, 3):
            raise ValueError("native logits must align with replay batch")

        counterpart_mode = counterpart_model.training
        counterpart_mamba_mode = counterpart_model.mamba.training
        counterpart_model.train(self.training)
        counterpart_model.mamba.train(counterpart_mamba_mode)
        self._stage196b2b6p8_restore_rng_state(before_rng)
        try:
            counterpart_output = counterpart_model(
                input_ids=None,
                attention_mask=attention,
                claim_mask=claim,
                evidence_mask=evidence,
                encoder_hidden_states=hidden,
                temporal_mismatch_flags=replay_state.get(
                    "temporal_mismatch_flags"
                ),
                predicate_mismatch_flags=replay_state.get(
                    "predicate_mismatch_flags"
                ),
            )
        finally:
            self._stage196b2b6p8_restore_rng_state(after_rng)
            counterpart_model.train(counterpart_mode)
            counterpart_model.mamba.train(counterpart_mamba_mode)

        native_geometry = self._stage196b2b6p8_geometry(
            native_logits, prefix="native"
        )
        native_final_modulation = native_logits - native_base_logits
        candidates: dict[str, dict[str, Any]] = {}
        for opaque_id, row_actions in candidate_mask.items():
            if (
                not isinstance(row_actions, torch.Tensor)
                or row_actions.dtype != torch.bool
                or row_actions.shape != (
                    batch_size, len(STAGE196B2B6P8_PRIMITIVE_KEYS)
                )
                or row_actions.device != hidden.device
            ):
                raise ValueError(
                    f"candidate {opaque_id} must be CUDA bool [batch, 5]"
                )
            row_keys = tuple(
                "".join("1" if value else "0" for value in row)
                for row in row_actions.detach().cpu().tolist()
            )
            if candidate_action_keys[opaque_id] != row_keys:
                raise ValueError(f"candidate action-key mismatch for {opaque_id}")
            selected: dict[str, torch.Tensor] = {}
            for column, key in enumerate(STAGE196B2B6P8_PRIMITIVE_KEYS):
                recipient = native_output.get(key)
                donor = counterpart_output.get(key)
                if (
                    not isinstance(recipient, torch.Tensor)
                    or not isinstance(donor, torch.Tensor)
                    or recipient.shape != (batch_size,)
                    or donor.shape != recipient.shape
                    or recipient.device != hidden.device
                    or donor.device != hidden.device
                ):
                    raise ValueError(f"unaligned live primitive {key!r}")
                selected[key] = torch.where(
                    row_actions[:, column], donor, recipient
                )
            recomposed = self.decision_head(
                frame_prob=selected["frame_prob"],
                predicate_coverage_prob=selected["predicate_coverage_prob"],
                sufficiency_prob=selected["sufficiency_prob"],
                positive_energy=selected["positive_energy"],
                negative_energy=selected["negative_energy"],
            )
            counterfactual_logits = (
                recomposed["logits"] + native_final_modulation
            )
            geometry = self._stage196b2b6p8_geometry(
                counterfactual_logits, prefix="counterfactual"
            )
            response = {
                "delta_score_support": geometry["counterfactual_score_support"] - native_geometry["native_score_support"],
                "delta_score_not_entitled": geometry["counterfactual_score_not_entitled"] - native_geometry["native_score_not_entitled"],
                "delta_score_refute": geometry["counterfactual_score_refute"] - native_geometry["native_score_refute"],
                "delta_support_minus_not_entitled": geometry["counterfactual_margin_support_minus_not_entitled"] - native_geometry["native_margin_support_minus_not_entitled"],
                "delta_support_minus_refute": geometry["counterfactual_margin_support_minus_refute"] - native_geometry["native_margin_support_minus_refute"],
                "delta_refute_minus_not_entitled": geometry["counterfactual_margin_refute_minus_not_entitled"] - native_geometry["native_margin_refute_minus_not_entitled"],
                "delta_top1_runner_up_margin": geometry["counterfactual_top1_runner_up_margin"] - native_geometry["native_top1_runner_up_margin"],
            }
            candidates[opaque_id] = {
                "candidate_action_keys": row_keys,
                "counterfactual_logits": counterfactual_logits,
                **selected,
                **geometry,
                **response,
            }
        return {
            "schema_version": "stage196b2b6p8_full_trainable_path_replay_v1",
            "class_order": STAGE196B2B6P8_CLASS_ORDER,
            "primitive_order": STAGE196B2B6P8_PRIMITIVE_KEYS,
            "candidate_masks": STAGE196B2B6P8_CANDIDATE_MASKS,
            "gradient_ownership_mode": gradient_ownership_mode,
            "rng_policy": STAGE196B2B6P8_RNG_POLICY,
            "native_logits": native_logits,
            **native_geometry,
            "candidate_geometry": candidates,
            "mamba_forward_count": 0,
            "downstream_replay_count": 1,
            "stability_loss": None,
            "training_objective_changed": False,
        }

    def forward(
        self,
        input_ids: torch.Tensor | None,
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
        temporal_channel_gated_penalty_scale: float = 0.0,
        return_composer_input_observability: bool = False,
        stage196b2b6p8_return_replay_state: bool = False,
    ) -> dict[str, Any]:
        """Forward pass with optional temporal/predicate logit modulation."""
        del intervention_types, pair_ids

        # Encode
        if encoder_hidden_states is None:
            if input_ids is None:
                raise ValueError("input_ids are required without encoder_hidden_states")
            backbone_outputs = self.mamba(input_ids=input_ids)
            token_states = backbone_outputs.last_hidden_state
        else:
            if input_ids is not None and encoder_hidden_states.shape[:2] != input_ids.shape:
                raise ValueError(
                    "encoder_hidden_states must match input_ids batch/sequence dimensions"
                )
            token_states = encoder_hidden_states

        stage196b2b6p8_stochastic_context: dict[str, Any] | None = None
        if stage196b2b6p8_return_replay_state:
            if token_states.device.type != "cuda":
                raise ValueError("Stage196-B2-B6P8 replay state requires CUDA")
            stage196b2b6p8_stochastic_context = {
                "rng_policy": STAGE196B2B6P8_RNG_POLICY,
                "native_pre_downstream_rng": (
                    self._stage196b2b6p8_capture_rng_state()
                ),
                "model_training": self.training,
            }

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

        # TemporalChannel V1: reads cat([claim_frame_state, evidence_frame_state]).
        # Pre-pair-projector slot states; NOT frame_pair_repr.
        # With detach=True (default): TC loss cannot propagate into FrameGate parameters.
        temporal_channel_logit: torch.Tensor | None = None
        temporal_channel_prob: torch.Tensor | None = None
        if self.temporal_channel_v1 is not None:
            _tc_base = torch.cat(
                [frame["claim_frame_state"], frame["evidence_frame_state"]], dim=-1
            )
            _tc_input = _tc_base.detach() if self.temporal_channel_detach_input else _tc_base
            temporal_channel_logit = self.temporal_channel_v1(_tc_input).squeeze(-1)
            temporal_channel_prob = torch.sigmoid(temporal_channel_logit)

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

        # Stage196-B2-B3P0: allocate exact row-level composer observations only
        # when explicitly requested by the clean-dev exporter. These tensors are
        # detached diagnostic witnesses; they never replace or modify final_logits.
        composer_observability: dict[str, torch.Tensor] = {}
        if return_composer_input_observability:
            zero_delta = torch.zeros_like(base_logits).detach()
            composer_observability = {
                "composer_temporal_mismatch_condition_input": (
                    temporal_mismatch_flags.detach()
                    if temporal_mismatch_flags is not None
                    else torch.zeros_like(frame["frame_prob"])
                ),
                "composer_predicate_mismatch_condition_input": (
                    predicate_mismatch_flags.detach()
                    if predicate_mismatch_flags is not None
                    else torch.zeros_like(frame["frame_prob"])
                ),
                "composer_temporal_adapter_final_penalty_scale": torch.full_like(
                    frame["frame_prob"], float(temporal_adapter_final_penalty_scale),
                    dtype=torch.float64,
                ).detach(),
                "composer_temporal_channel_gated_penalty_scale": torch.full_like(
                    frame["frame_prob"], float(temporal_channel_gated_penalty_scale),
                    dtype=torch.float64,
                ).detach(),
                "composer_temporal_mismatch_active": torch.zeros_like(
                    frame["frame_prob"], dtype=torch.bool
                ),
                "composer_predicate_mismatch_active": torch.zeros_like(
                    frame["frame_prob"], dtype=torch.bool
                ),
                "composer_temporal_mismatch_delta": zero_delta.clone(),
                "composer_predicate_mismatch_delta": zero_delta.clone(),
                "composer_temporal_adapter_delta": zero_delta.clone(),
                "composer_temporal_channel_delta": zero_delta.clone(),
                "composer_temporal_adapter_active": torch.zeros_like(
                    frame["frame_prob"], dtype=torch.bool
                ),
                "composer_temporal_channel_active": torch.zeros_like(
                    frame["frame_prob"], dtype=torch.bool
                ),
                "composer_temporal_adapter_effective_penalty_scale": torch.zeros_like(
                    frame["frame_prob"]
                ).detach(),
                "composer_temporal_channel_effective_scale": torch.zeros_like(
                    frame["frame_prob"]
                ).detach(),
            }

        if self.use_temporal_comparator and temporal_mismatch_flags is not None:
            alpha = self.alpha_temporal()
            active = temporal_mismatch_flags.bool()
            if return_composer_input_observability:
                composer_observability["composer_temporal_mismatch_active"] = active.detach()
            if torch.any(active):
                final_logits = final_logits.clone()
                final_logits[active, 0] -= alpha  # SUPPORT
                final_logits[active, 1] += alpha  # NOT_ENTITLED
                final_logits[active, 2] -= alpha  # REFUTE
                temporal_flag_count = int(active.sum().item())
                if return_composer_input_observability:
                    delta = composer_observability["composer_temporal_mismatch_delta"]
                    delta[active, 0] = -alpha.detach()
                    delta[active, 1] = alpha.detach()
                    delta[active, 2] = -alpha.detach()


        if self.use_predicate_comparator and predicate_mismatch_flags is not None:
            alpha = self.alpha_predicate()
            active = predicate_mismatch_flags.bool()
            if return_composer_input_observability:
                composer_observability["composer_predicate_mismatch_active"] = active.detach()
            if torch.any(active):
                final_logits = final_logits.clone()
                final_logits[active, 0] -= alpha  # SUPPORT
                final_logits[active, 1] += alpha  # NOT_ENTITLED
                final_logits[active, 2] -= alpha  # REFUTE
                predicate_flag_count = int(active.sum().item())
                if return_composer_input_observability:
                    delta = composer_observability["composer_predicate_mismatch_delta"]
                    delta[active, 0] = -alpha.detach()
                    delta[active, 1] = alpha.detach()
                    delta[active, 2] = -alpha.detach()

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
            if return_composer_input_observability:
                composer_observability["composer_temporal_adapter_active"] = torch.ones_like(
                    temporal_adapter_logit, dtype=torch.bool
                )
                composer_observability[
                    "composer_temporal_adapter_effective_penalty_scale"
                ] = _ta_penalty.detach()
                delta = composer_observability["composer_temporal_adapter_delta"]
                delta[:, 0] = -_ta_penalty.detach()
                delta[:, 1] = _ta_penalty.detach()
                delta[:, 2] = -_ta_penalty.detach()
            final_logits = final_logits.clone()
            final_logits[:, 0] -= _ta_penalty  # SUPPORT
            final_logits[:, 1] += _ta_penalty  # NOT_ENTITLED
            final_logits[:, 2] -= _ta_penalty  # REFUTE

        # TemporalChannel gated penalty: per-example NOT_ENTITLED boost.
        # Formula: scale * sigmoid(tc_logit).detach() * (1 - pe_prob).detach()
        # Only fires when TC detects temporal mismatch AND PE signals non-entitlement.
        # Suppressed on preservation-safe examples (high pe_prob) by the (1-pe_prob) gate.
        # Requires preservation_entitlement_head to be enabled.
        # No gradient flows back into TC or PE from this penalty (both logits detached).
        if temporal_channel_logit is not None and temporal_channel_gated_penalty_scale > 0.0:
            if preservation_entitlement_prob is None:
                raise RuntimeError(
                    "temporal_channel_gated_penalty_scale > 0 requires "
                    "use_preservation_entitlement_head=True; "
                    "preservation_entitlement_prob is None in this forward pass"
                )
            _tc_boost = (
                torch.sigmoid(temporal_channel_logit.detach())
                * (1.0 - preservation_entitlement_prob.detach())
                * temporal_channel_gated_penalty_scale
            )
            if return_composer_input_observability:
                composer_observability["composer_temporal_channel_active"] = torch.ones_like(
                    temporal_channel_logit, dtype=torch.bool
                )
                composer_observability[
                    "composer_temporal_channel_effective_scale"
                ] = _tc_boost.detach()
                delta = composer_observability["composer_temporal_channel_delta"]
                delta[:, 0] = -_tc_boost.detach()
                delta[:, 1] = _tc_boost.detach()
                delta[:, 2] = -_tc_boost.detach()
            final_logits = final_logits.clone()
            final_logits[:, 0] -= _tc_boost  # SUPPORT
            final_logits[:, 1] += _tc_boost  # NOT_ENTITLED
            final_logits[:, 2] -= _tc_boost  # REFUTE

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
        output: dict[str, Any] = {
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
            # TemporalChannel V1 outputs (None when channel is disabled)
            "temporal_channel_logit": temporal_channel_logit,
            "temporal_channel_prob": temporal_channel_prob,
            **composer_observability,
            **frame,
            **predicate,
            **sufficiency,
            **polarity,
            "token_states": token_states if return_token_states else None,
            "loss": total_loss,
            **losses,
        }
        if stage196b2b6p8_return_replay_state:
            assert stage196b2b6p8_stochastic_context is not None
            stage196b2b6p8_stochastic_context[
                "native_post_downstream_rng"
            ] = self._stage196b2b6p8_capture_rng_state()
            replay_state = {
                "encoder_hidden_states": token_states,
                "attention_mask": attention_mask,
                "claim_mask": claim_mask,
                "evidence_mask": evidence_mask,
            }
            if temporal_mismatch_flags is not None:
                replay_state["temporal_mismatch_flags"] = temporal_mismatch_flags
            if predicate_mismatch_flags is not None:
                replay_state[
                    "predicate_mismatch_flags"
                ] = predicate_mismatch_flags
            output["stage196b2b6p8_replay_state"] = replay_state
            output[
                "stage196b2b6p8_stochastic_context"
            ] = stage196b2b6p8_stochastic_context
        return output
