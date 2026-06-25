# Stage24 Temporal Channel Addendum

**Status:** Addendum to stage24_temporal_adapter_summary.md — records TemporalChannelV1 sanity
results and gated-penalty qualitative outcomes. No new architecture changes. No training results
beyond what is recorded here.

---

## 1. TemporalChannelV1 Loss-Only Functional Sanity

After implementing TemporalChannelV1 and fixing the temporal diagnostic data routing in the
training script, the loss-only configuration passed functional sanity:

- `active_training_losses.temporal_channel_loss.enabled = true`
- `loss_component_epoch_avg_raw.temporal_channel_loss > 0`
- `loss_component_epoch_avg_weighted.temporal_channel_loss > 0`
- `weighted temporal_channel_loss ≈ raw * temporal_channel_loss_weight`
- `active_final_logit_modifiers.temporal_channel_gated_penalty.enabled = false`
- Stage15 provenance fields: all `false`
- `time_swap` not present in main clean train/eval data

The TC head correctly reads `cat([claim_frame_state, evidence_frame_state])` — the
pre-pair-projector slot states from FrameGate — with detached input (`temporal_channel_detach_input=True`).
Gradients from the TC BCE loss do not reach FrameGate parameters.

**Loss-only conclusion:** TC BCE is a functional signal probe. It trains the TC head to detect
temporal mismatch from pre-projector frame slot states. When the penalty is disabled, final logits
are unchanged and clean dev metrics are unaffected. This confirms the architectural isolation works.

---

## 2. PE-Gated Final-Logit Penalty — Qualitative Outcomes

Three penalty scales were tested with the PE-gated formula:

```
temporal_NE_boost = scale × sigmoid(tc_logit).detach() × (1 - pe_prob).detach()
```

Applied as: `NE_logit += boost`, `SUPPORT_logit -= boost`, `REFUTE_logit -= boost`.

| Scale | Temporal mismatch effect | Surface control effect | Observation |
|-------|--------------------------|------------------------|-------------|
| p0.25 | Weak improvement | Mostly stable | Small signal; insufficient to shift temporal decision boundary |
| p0.75 | Moderate improvement | Worsens | Trade-off reappears; surface_control FNE increases |
| p1.5  | Non-monotonic / unstable | Further degraded | Penalty interacts with PE signal in non-linear ways |

The pattern is consistent with Stage24's adapter penalty findings: unconditional or weakly-gated
final-logit boosts cannot cleanly separate temporal mismatch from preservation-safe examples at
meaningful scales. Even with PE gating, the penalty fires on some surface-control and paraphrase
examples where PE probability is not close to 1.

---

## 3. Conclusion

Direct TC-gated final-logit penalty is not the preferred final path forward.

- The TC head provides real temporal mismatch signal (probe accuracy > chance on diagnostic data).
- But converting that signal into a final-logit NE boost — even when PE-gated — causes
  preservation/rejection trade-offs that cannot be resolved by tuning the scale.
- The fundamental issue is that the temporal signal is being applied as a crude post-hoc logit
  modifier, not as a structured input to the entitlement judgment.

The preferred next direction (Stage25) is to integrate the TC signal as an explicit input to a
hierarchical EntitlementGate rather than as a direct logit intervention. Temporal invalidity is
an entitlement precondition, not a polarity signal — this is a principle derived from the
ContraMamba six-axis decomposition, which the EpistemicBERT codebook operationalized in
annotation order but did not originate.
