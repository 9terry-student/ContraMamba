# Stage24 Temporal Channel Addendum

**Status:** Addendum to stage24_temporal_adapter_summary.md — records TemporalChannelV1 sanity
results and gated-penalty observed outcomes (seed1, Stage15 slot-sensitivity). No new architecture
changes. No training results beyond what is recorded here.

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

## 2. PE-Gated Final-Logit Penalty — Observed Outcomes (seed1, Stage15 slot-sensitivity)

Four configurations were evaluated with the PE-gated formula:

```
temporal_NE_boost = scale × sigmoid(tc_logit).detach() × (1 - pe_prob).detach()
```

Applied as: `NE_logit += boost`, `SUPPORT_logit -= boost`, `REFUTE_logit -= boost`.

Stage15 was used here as an evaluation oracle only. It was not used for penalty-scale selection,
threshold tuning, or checkpoint selection. All provenance fields: `stage15_used_for_*_selection = false`.

| Run | temporal_mismatch FER | surface_control FNER | temporal_erased FNER | Notes |
|-----|----------------------|---------------------|----------------------|-------|
| `tc_loss_only` | 0.87 | 0.08 | 0.00 | Auxiliary TC loss functional; final logits unchanged |
| `tc_gated_p025` | 0.85 | 0.09 | 0.00 | Penalty too weak; negligible temporal improvement |
| `tc_gated_p075` | 0.68 | 0.18 | 0.00 | Best temporal improvement; surface_control cost increases |
| `tc_gated_p150` | 0.88 | 0.01 | 0.00 | Non-monotonic: temporal improvement lost, surface preserved |

**FER** = false_entitled_rate (lower is better for temporal_mismatch — model should reject these as NOT_ENTITLED).
**FNER** = false_not_entitled_rate (lower is better for surface_control and temporal_erased — model should not over-reject these).

**Per-run observations:**

- **`tc_loss_only`:** TC BCE loss computed correctly and TC head learned a temporal signal.
  Final logits were not modified. temporal_mismatch FER remained at baseline (0.87).
  Confirms gradient isolation works but loss-only does not move final decisions.

- **`tc_gated_p025`:** Penalty scale too small to materially shift the temporal decision
  boundary. temporal_mismatch FER dropped only from 0.87 to 0.85. Surface control largely
  unaffected (FNER 0.08 → 0.09).

- **`tc_gated_p075`:** Strongest temporal improvement observed (FER 0.87 → 0.68). Trade-off
  is clear: surface_control FNER increased from 0.08 to 0.18. The penalty fires on
  surface-control examples where PE probability is not close to 1, pushing them incorrectly
  toward NOT_ENTITLED.

- **`tc_gated_p150`:** Non-monotonic result. temporal_mismatch FER returned to 0.88 — worse
  than p0.75 and nearly at baseline. surface_control FNER dropped to 0.01, suggesting the
  high-scale penalty saturated into a regime that suppresses the NE boost on most examples.
  The interaction between TC signal and PE signal at this scale is unstable.

- **`temporal_erased` across all runs:** FNER = 0.00. The TC head does not fire on
  temporal_erased examples, confirming that the TC signal is specific to temporal mismatch
  and does not over-reject preservation-safe temporal-erased cases.

The pattern is consistent with Stage24's adapter penalty findings: post-hoc final-logit boosts,
even when gated by PE probability, cannot cleanly separate temporal mismatch from
preservation-safe examples at scales sufficient to produce meaningful temporal rejection. The
scale that most reduces temporal_mismatch FER (p0.75) is also the scale that most increases
surface_control FNER. This is not a tuning problem; it is a structural property of scalar
logit modification on shared final-logit vectors.

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
