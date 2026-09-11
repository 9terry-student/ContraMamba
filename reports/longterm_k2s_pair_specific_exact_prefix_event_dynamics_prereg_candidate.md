# K2S Pair-Specific Exact-Prefix Event Dynamics Preregistration Candidate

**Status:** successor scientific/design preregistration candidate.

**Frozen prospective design verdict:** `PASS_READY_TO_FREEZE_K2S_PREREG`.

This document is intended, once committed, to be the complete scientific/design authority for bounded K2S implementation, instrumentation validation, and K2S scientific execution. No separate execution-authority document is required.

It does not reopen, amend, rescue, or reinterpret K1, K2, or K2W.

## 1. Motivation and predecessor boundary

K1 closed without native-state observation because matched observational support failed.

K2 closed without native-state observation because its full-continuation token contract had no construction-valid population.

K2W corrected the construction problem: all 300 frozen candidates were semantically valid, exact-prefix valid, event-divergence valid, and W=8 valid. All 300 reached frozen three-observer screening. However:

- N_ELIGIBLE = 0
- N_FINAL = 0
- phase_a_verdict = `INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE`

The categorical K2W observer result was:

| Seed | REFUTE | NOT_ENTITLED | SUPPORT |
| --- | ---: | ---: | ---: |
| seed180 | 50 | 244 | 6 |
| seed181 | 5 | 295 | 0 |
| seed182 | 0 | 300 | 0 |

Across the three frozen observers:

- 0 SUPPORT votes = 294
- 1 SUPPORT vote = 6
- 2 SUPPORT votes = 0
- 3 SUPPORT votes = 0

Therefore K2W established that its prospectively frozen unanimous false-commitment entrance condition had no admissible population on the frozen source.

This was not a native-state null result. Native recurrent state was never captured.

K2S is a new successor experiment. It removes A0 wrong-commitment prediction as a population-selection requirement. It does not weaken K2W after seeing its outcome.

The scientific question is now narrower and more fundamental:

**Does the native recurrent-state response to an evidence event depend on the semantic pairing between the prefix and the continuation, beyond the lexical and token-content marginals of the continuation families?**

K2S is therefore an event-response specificity experiment, not a confident-error precursor experiment.

## 2. Governing prior authorities and frozen evidence

The relevant authority/evidence chain is:

- K0 native-state kinematics hypothesis specification:
  `7bd1cf824cd53c7f6cf6215346b42cabf351b70a`

- K1 closure / K2 theory:
  `e8227945401d83b952e35b056343e3e1cb19e18f`

- K2 closure:
  `386ef0763a0dd8470c22617581af196a324f222f`

- K2W preregistration:
  `cc5386e730c333209eb070b14025c5368038e247`

- K2W Phase-A implementation:
  `c7c7a0c218bb64d083f0da86e2929990d87bc4ed`

- K2W closure:
  `b94f81b411bcbc74e32ad0ad6564b8978016eec0`

- O0c native selective-SSM state semantics:
  `ff2fb076f6e66a34a632515bb8502d8b1c90ad7f`

K0 progression remains:

Measure -> Temporalize -> Causally characterize -> Structure only if justified.

K2S remains on the measurement / temporalization side of that boundary.

It does not authorize recurrent-state intervention, state patching, overwrite, steering, necessity, sufficiency, or causal-mechanistic claims.

## 3. Frozen K2S population

The exact K2S population is the committed K2W construction-valid candidate artifact:

`reports/longterm_k2w_fixed_window_phase_a_c7c7a0c218bb/candidate_pool.jsonl`

at commit:

`b94f81b411bcbc74e32ad0ad6564b8978016eec0`

with SHA256:

`abf693d3267cc4e3dd27a8127d2948b36fdf8ba24e135a643215f0f31a26d808`

The artifact contains exactly:

- 300 rows;
- 300 construction-valid rows;
- 300 unique `stable_item_id` values;
- 300 unique `base_claim_sha256` values;
- zero duplicate exclusions.

K2S uses from each row only the prospectively frozen identity and texts:

- `stable_item_id`
- `base_claim_sha256`
- `pair_id`
- `prefix_text`
- `correction_text`
- `control_text`
- source provenance identities.

K2W `p`, `d`, `tau`, observer predictions, eligibility, confidence, and margins do not select K2S items.

All 300 items are used.

No A0 final-class prediction is an inclusion criterion.

No confidence threshold, margin threshold, voting rule, or head outcome may alter K2S population membership.

## 4. Original frozen semantic source

The underlying semantic source remains:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

from Git commit:

`8eb7386e0344117d026c0e6ab172018bb98a698e`

Physical SHA256:

`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

Semantic SHA256:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

The K2W construction already froze for item i:

P_i =
`"Claim: " + T.claim + "\nEvidence: " + T.evidence + "\nAdditional evidence:\n"`

C_i =
the decisive same-claim REFUTE continuation.

N_i =
the same-pair entity-swap NOT_ENTITLED control continuation.

K2S does not regenerate, rewrite, normalize, paraphrase, shorten, pad, or replace these texts.

## 5. Exact tokenizer and model identity

Tokenizer/model identity remains:

`state-spaces/mamba-130m-hf`

Exact Hugging Face revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Required tokenizer settings:

- `trust_remote_code = false`
- `add_special_tokens = false`
- unpadded tokenization
- fast tokenizer required.

Resolved config/tokenizer files, byte sizes, paths, and SHA256s must be recorded in the scientific manifest.

Any revision or file-identity mismatch is:

`INSTRUMENTATION_OR_PROVENANCE_FAILURE`

and stops execution.

## 6. Deterministic reciprocal block design

Sort the 300 frozen population rows by ascending `stable_item_id`.

Index them:

i = 0, 1, ..., 299.

Define deterministic reciprocal donor mapping:

`pi(i) = i XOR 1`.

Thus:

- 0 <-> 1
- 2 <-> 3
- ...
- 298 <-> 299.

This creates exactly 150 disjoint two-item blocks.

For each item i, let j = pi(i).

Require:

- i != j;
- `stable_item_id_i != stable_item_id_j`;
- `base_claim_sha256_i != base_claim_sha256_j`.

The matched branches for recipient prefix P_i are:

- M_corr(i) = P_i + C_i
- M_ctrl(i) = P_i + N_i

The swapped branches are:

- S_corr(i) = P_i + C_j
- S_ctrl(i) = P_i + N_j

Within every reciprocal block, the two prefixes, two correction continuations, and two control continuations therefore appear once under matched assignment and once under swapped assignment.

Across the complete experiment the following marginals are exactly preserved between matched and swapped assignment:

- prefix-text multiset;
- correction-continuation text multiset;
- control-continuation text multiset.

No random pairing, resampling, seed search, semantic-distance matching, or post-hoc donor choice is permitted.

## 7. Event anchor and fixed window

K2S does not use the K2W first-branch-divergence `tau` as its scientific anchor.

For each recipient prefix P_i:

`p_i = len(tokens(P_i)) - 1`.

`p_i` is the final consumed token of the common literal prefix immediately before continuation onset.

The scientific event is:

**onset of the additional-evidence continuation.**

Primary event-relative positions are fixed as:

k = 1, 2, ..., 8

corresponding to recurrent states:

S_(p_i+k).

Thus W=8 is frozen.

For all four branches of item i require:

- tokens(P_i) are an exact prefix;
- attention mask is identical through p_i;
- zero-based positions are identical through p_i;
- at least 8 post-prefix tokens exist.

There is no total continuation-length equality requirement.

There is no total continuation maximum.

There is no branch padding.

There is no truncation to equalize branches.

The first correction-vs-control token divergence may be recorded as a tokenizer diagnostic but is not the event anchor and may not select observations.

Any study item failing an exact-prefix or W=8 contract at scientific execution causes fail-closed execution rather than item dropping or replacement.

## 8. State-blind feasibility history

No native state, model forward, A0 head, checkpoint output, confidence, or scientific endpoint was inspected during K2S design feasibility.

An initial state-blind tokenizer-only audit tested a deterministic 300-cycle donor assignment. It found 300/300 matched and 300/300 deranged-valid constructions.

Before preregistration, that assignment was discarded for a statistical-design reason: the single long donor cycle creates unnecessary cross-item dependence.

No scientific state or model outcome motivated that change.

The final reciprocal two-item block rule was then fixed and audited once.

Final frozen feasibility result:

- N_total = 300
- N_blocks = 150
- N_unique_stable_id = 300
- N_unique_base_claim = 300
- N_matched_valid = 300
- N_swapped_valid = 300
- failures = none.

Matched first-divergence diagnostic relative to prefix end:

- d-p = 2 for 149 items
- d-p = 3 for 151 items.

Swapped first-divergence diagnostic:

- d-p = 2 for 149 items
- d-p = 3 for 151 items.

Matched correction post-prefix availability:

- minimum 20
- maximum 27.

Matched control post-prefix availability:

- minimum 17
- maximum 26.

Swapped correction post-prefix availability:

- minimum 20
- maximum 27.

Swapped control post-prefix availability:

- minimum 17
- maximum 26.

Prefix, correction, and control text marginals were exactly preserved.

This feasibility audit authorizes no scientific claim.

## 9. Native recurrent-state source

Use one canonical authenticated seed180 A0 realization.

Authenticated seed180 handoff:

`C:\Users\Home1\Downloads\p3w7-seed8192-a0-seed180-replacement-r1-retry1_55debe94f0d1.zip`

Handoff ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Selected checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

A0 checkpoint lineage commit:

`55debe94f0d19d16a334395e8561901fed6b52fa`

The native Mamba encoder must satisfy:

- canonical per-tensor digest:
  `48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

- secondary raw-concatenation digest:
  `968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

- exactly 242 float32 `mamba.*` tensors;
- exactly 129135360 total numel;
- exactly 516541440 raw tensor bytes.

The canonical digest and structural identity are normative.

The raw-concatenation digest is secondary provenance.

The three historical A0 realizations are known to share this exact encoder identity. K2S therefore uses seed180 as the canonical encoder realization rather than pretending that identical encoder bytes constitute independent native-state replications.

A0 task-head predictions are not used for K2S selection, primary endpoints, or promotion.

## 10. Native-state semantics

Capture only the direct selective-SSM recurrent state under the validated O0c semantics:

post-consumption recurrent state `s_t`, after token t has updated the recurrent state and before downstream readout.

Forbidden substitutes include:

- generic hidden states;
- final cache-only state;
- reconstructed recurrent state;
- learned probes;
- task-head hidden representations;
- MLP trajectory embeddings.

Capture all 24 layers for integrity and descriptive analysis.

The sole primary layer is:

layer 23.

Layers 0 through 22 are descriptive and non-rescuing.

For each branch capture enough common-prefix state to define:

- S_(p_i-1)
- S_(p_i)
- S_(p_i+1) through S_(p_i+8).

The four branches for a recipient item must have identical recurrent states through p_i within the exact numerical identity policy established by instrumentation validation.

Any failure of causal-prefix identity, finite-state integrity, fresh-state isolation, trace noninterference, or provenance is:

`INSTRUMENTATION_OR_PROVENANCE_FAILURE`

and stops the experiment.

No item may be silently dropped because instrumentation failed.

## 11. Instrumentation validation before scientific execution

Implementation must include a non-scientific instrumentation preflight before any K2S scientific artifact is produced.

The preflight must verify, using synthetic non-study text:

- exact authenticated checkpoint reconstruction;
- strict state-dict loading;
- canonical encoder fingerprint;
- direct recurrent-state capture at all 24 layers;
- expected recurrent-state shape;
- finite float32 state;
- fresh-state isolation;
- identical ordinary logits with tracing disabled versus enabled within the frozen numerical policy;
- exact causal common-prefix state identity across branches sharing a prefix;
- no hidden-state proxy substitution.

Where practical, reuse or directly port the already validated O0c state-capture semantics rather than inventing a new instrumentation convention.

Failure blocks scientific execution.

## 12. Primary native-state kinematics

Use epsilon:

`1e-12`.

For branch b and recipient item i define, at primary layer 23:

V_(p+k)^b =
S_(p+k)^b - S_(p+k-1)^b

for k = 1,...,8.

The common final pre-event velocity is:

V_p =
S_p - S_(p-1).

Define branch mean speed:

speedbar_b =
(1/8) * sum over k=1..8 of ||V_(p+k)^b||_F.

Define branch turning:

turn_(p+k)^b =
1 - cosine(V_(p+k)^b, V_(p+k-1)^b).

For k=1, the previous velocity is V_p.

A turn is undefined when either adjacent velocity norm is <= epsilon.

Define branch mean turning as the mean over valid turns.

If a branch has zero valid turns, its mean turning is undefined.

Define branch path length:

L_b =
sum over k=1..8 of ||V_(p+k)^b||_F.

Define branch displacement:

Disp_b =
||S_(p+8)^b - S_p||_F.

Define trajectory efficiency:

eta_b =
Disp_b / (L_b + epsilon).

These yield exactly three branch-level primary metric families:

- R: mean speed;
- D: mean turning;
- P: trajectory efficiency.

No other metric may join the confirmatory primary family.

## 13. Recipient-level matched and swapped contrasts

For each recipient item i define signed correction-vs-control contrasts.

Movement:

DeltaR_i_matched =
speedbar_Mcorr(i) - speedbar_Mctrl(i)

DeltaR_i_swapped =
speedbar_Scorr(i) - speedbar_Sctrl(i)

Direction:

DeltaD_i_matched =
turnbar_Mcorr(i) - turnbar_Mctrl(i)

DeltaD_i_swapped =
turnbar_Scorr(i) - turnbar_Sctrl(i)

Path:

DeltaP_i_matched =
eta_Mcorr(i) - eta_Mctrl(i)

DeltaP_i_swapped =
eta_Scorr(i) - eta_Sctrl(i)

The signed contrasts must be retained and reported descriptively.

However, K2S does not preregister that a valid semantic response must increase rather than decrease speed, turning, or efficiency.

Therefore primary pair-specificity uses contrast magnitude.

For q in {R,D,P}:

X_i_q =
abs(Deltaq_i_matched) - abs(Deltaq_i_swapped).

Positive X means that the correction-vs-control native-state contrast is larger in magnitude when the continuation pair is assigned to its own prefix than when the same continuation family is assigned to the reciprocal partner prefix.

Negative X means the swapped assignment produces the larger contrast.

This sign must not be reversed or reinterpreted after execution.

## 14. Block-level scientific unit

For reciprocal block m containing items a and b define:

B_m_q =
(X_a_q + X_b_q) / 2.

The scientific inference unit is the reciprocal block, not the individual recipient item.

There are exactly:

150 blocks.

This prevents treating the two reciprocal assignments within a block as independent observations.

For D, a block is undefined if any required matched or swapped branch mean-turn quantity for either member is undefined.

R and P must be finite for every block; otherwise execution fails instrumentation/integrity validation.

## 15. Primary inference

The primary family contains exactly:

- B_R
- B_D
- B_P.

For each endpoint:

- count positive block values;
- count negative block values;
- count exact zeros;
- count undefined values;
- report n_valid;
- exclude exact zeros from n_eff.

Use a two-sided exact binomial sign test under p=0.5.

If n_eff = 0:

raw p = 1.

A primary endpoint is promotion-eligible only if:

- n_valid >= 120 blocks;
- n_eff >= 30 blocks.

If either floor fails:

raw p = 1 for promotion purposes,
the endpoint is marked insufficient-effective-support,
and it cannot generate a positive claim.

Multiplicity correction is fixed Holm familywise correction with:

- m = 3;
- alpha = 0.05;
- deterministic tie order R < D < P.

Report for every endpoint:

- raw p;
- Holm-adjusted p;
- Holm rejection;
- n_valid;
- n_eff;
- positive count;
- negative count;
- zero count;
- undefined count;
- rank-biserial sign effect:
  `(n_positive - n_negative) / n_eff`
  when n_eff > 0.

No parametric normality assumption is primary.

## 16. Frozen positive, null, reversed, and mixed interpretation

A positive endpoint requires all of:

- valid construction and provenance;
- endpoint promotion floors satisfied;
- Holm rejection;
- positive rank-biserial sign effect.

At least one positive primary endpoint is required for:

`PAIR_SPECIFIC_EVENT_ALIGNED_NATIVE_STATE_RESPONSE_OBSERVED`

This language means only that matched semantic prefix-continuation assignment produced a stronger fixed-window native-state correction-vs-control contrast than the deterministic lexical-marginal-preserving swapped assignment.

It does not establish:

- a confident-error precursor;
- pre-evidence prediction;
- causal mechanism;
- necessity;
- sufficiency;
- epistemic state ontology;
- deployable detector.

If valid execution completes and no endpoint meets the positive rule, while no endpoint shows a Holm-significant negative effect, use:

`PRIMARY_PAIR_SPECIFICITY_NULL_AFTER_VALID_EXECUTION`

If one or more Holm-significant endpoints have negative rank-biserial sign effect and no endpoint meets the positive rule, use:

`PAIR_SPECIFICITY_REVERSED_AFTER_VALID_EXECUTION`

A reversed outcome is evidence against the K2S pair-specificity hypothesis. It must not be counted as success.

If at least one positive and at least one negative endpoint are Holm-significant, use:

`MIXED_PRIMARY_PAIR_SPECIFICITY_RESULT`

and do not promote a single-mechanism interpretation.

## 17. Secondary descriptive analyses

The following are permitted only as non-rescuing secondary descriptions:

- signed matched DeltaR / DeltaD / DeltaP distributions;
- signed swapped DeltaR / DeltaD / DeltaP distributions;
- layers 0 through 22;
- item-level X values;
- full fixed-window layer-23 trajectories;
- first correction-vs-control token-divergence diagnostics;
- continuation token identities and lengths;
- block-level heterogeneity.

They may explain a primary result.

They may not overturn a primary null, reversed, or mixed verdict.

No best-layer search, best-time search, best-window search, or metric expansion may create promotion.

## 18. Explicit non-selection and anti-rescue rules

K2S must not select or weight items by:

- A0 class prediction;
- SUPPORT vote count;
- confidence;
- margin;
- K2W eligibility;
- final answer correctness;
- native-state magnitude;
- layer response;
- token-divergence location;
- any R/D/P result.

All 300 frozen population items remain in the design unless the entire execution fails closed for construction/provenance/instrumentation reasons.

Forbidden post-hoc rescue includes:

- changing W=8;
- changing the primary layer;
- changing reciprocal blocks;
- changing donor assignment;
- choosing different lexical controls;
- changing absolute contrast specificity to another statistic after seeing state results;
- threshold sweeps;
- confidence conditioning;
- head-based filtering;
- dropping inconvenient blocks;
- replacing failed items;
- adding primary metrics.

## 19. Provenance requirements

A scientific execution manifest must bind at minimum:

- this committed K2S preregistration identity;
- implementation commit;
- runtime Git HEAD;
- runtime branch;
- dirty-state contract;
- K2W closure commit;
- candidate-pool path and SHA256;
- original source commit/path/physical SHA/semantic SHA;
- stable-ID ordering;
- every reciprocal block mapping;
- exact P/C/N texts;
- exact matched and swapped input texts;
- tokenizer revision and resolved files;
- exact token arrays;
- p_i for every recipient;
- W=8 contract;
- checkpoint ZIP identity;
- checkpoint member identity;
- checkpoint SHA;
- encoder canonical digest and structural fingerprint;
- runtime Python / torch / transformers / tokenizers / huggingface_hub versions;
- native-state instrumentation identity;
- noninterference result;
- all primary endpoint inputs;
- all primary block values;
- raw and corrected statistics;
- output artifact SHA256s.

Any provenance mismatch fails closed.

## 20. Execution boundary

Once this preregistration is independently reviewed and committed, it authorizes:

- bounded K2S implementation;
- focused implementation tests;
- synthetic/non-scientific instrumentation validation;
- one scientific K2S execution on the exact frozen 300-item population;
- primary and prespecified secondary artifact generation.

It does not authorize:

- training;
- fine-tuning;
- learned probes;
- hyperparameter sweeps;
- new thresholds;
- K3 recurrent-state interventions;
- architecture modification;
- Kaggle use unless separately required operationally after implementation is frozen.

No separate K2S execution-authority document is required.

## 21. Relationship to K3

K3 remains a later causal/mechanistic phase.

Only a valid positive K2S phenomenon may motivate a new K3 mechanistic design.

A positive K2S result does not itself authorize K3 execution.

Any future K3 study must separately freeze:

- exact recurrent-state intervention;
- intervention location/time;
- necessity/sufficiency claim boundary;
- causal controls;
- provenance.

If K2S is null or reversed after valid execution, do not jump to complex state architecture or semantic multi-stream construction as a rescue.

## 22. Claim boundary

The strongest K2S positive language is:

`PAIR_SPECIFIC_EVENT_ALIGNED_NATIVE_STATE_RESPONSE_OBSERVED`

K2S cannot by itself establish:

- `EVENT_ALIGNED_PRECURSOR_SUPPORTED`;
- confident-error prediction;
- error detection;
- epistemic authorization ontology;
- causal selective-SSM mechanism;
- native-state necessity;
- native-state sufficiency;
- deployable diagnostic behavior.

K2S asks only whether a simple native-state fixed-window response is stronger under the correct semantic prefix-continuation pairing than under an exactly marginal-preserving reciprocal mismatch control.

## 23. Final preregistration verdict

The final deterministic reciprocal-block tokenizer feasibility audit passed:

- 300/300 matched valid;
- 300/300 swapped valid;
- 150 disjoint reciprocal blocks;
- 300 unique stable IDs;
- 300 unique base claims;
- identical prefix marginal;
- identical correction-continuation marginal;
- identical control-continuation marginal;
- zero feasibility failures.

Therefore:

`PASS_READY_TO_FREEZE_K2S_PREREG`

This is design readiness, not scientific evidence.
