# K2R Claim-Disjoint Local-vs-Net Trajectory Dissociation Replication Preregistration Candidate

**Status:** successor confirmatory scientific/design preregistration candidate.

**Frozen design verdict:** `PASS_READY_TO_PREREGISTER_K2R`.

Once committed, this document is intended to be the complete scientific/design authority for bounded K2R population materialization, implementation, instrumentation validation, and one confirmatory scientific execution.

K2R does not reopen or amend K2S.

K2R does not authorize K3 intervention.

## 1. Scientific motivation

K2S produced a valid direct native-state execution on 300 claims and 150 reciprocal blocks.

Its frozen primary verdict was:

`MIXED_PRIMARY_PAIR_SPECIFICITY_RESULT`

At the preregistered primary layer 23:

- R mean-speed pair-specificity was significantly positive.
- D mean-turning pair-specificity was significantly positive.
- P trajectory-efficiency pair-specificity was significantly negative.

A prespecified secondary decomposition of the already-produced K2S artifacts showed:

- path-length specificity followed R positively;
- net displacement specificity was negative;
- efficiency specificity was strongly negative;
- local-positive and net-negative effects frequently coexisted within the same blocks.

The secondary interpretation was frozen at commit:

`624ea02130eddce416a92239a97d7728bd8aa5b1`

and explicitly did not change the K2S primary verdict.

K2R is a new confirmatory successor asking:

**Does a claim-disjoint prospective population reproduce the K2S local-versus-net trajectory dissociation under the same native-state measurement geometry?**

K2R is not a causal experiment.

## 2. Governing evidence chain

Relevant frozen anchors are:

- K0 native-state kinematics hypothesis:
  `7bd1cf824cd53c7f6cf6215346b42cabf351b70a`

- K2W closure:
  `b94f81b411bcbc74e32ad0ad6564b8978016eec0`

- K2S preregistration:
  `c9cb68c2a48a19c3874d059ca00f5c297d929ada`

- K2S implementation:
  `d6d901cb3f1ab636d4db8b6cbea0bdf1e0581631`

- K2S result archive:
  `6069213234793286e658948a2e6f3b4f1105543d`

- K2S secondary interpretation:
  `624ea02130eddce416a92239a97d7728bd8aa5b1`

K2S remains mixed.

Nothing in K2R retroactively converts K2S into a positive result.

## 3. Replication scope

K2R is a:

`SAME_GENERATOR_CLAIM_DISJOINT_PROSPECTIVE_REPLICATION`

It is not an external-distribution replication.

The new claims come from the same frozen controlled-data grammar/generator family used by the historical ContraMamba controlled datasets.

Therefore even a successful K2R replication establishes only robustness across a new deterministic claim population within that generator family.

It does not establish natural-language corpus generalization.

## 4. Frozen generator identity

K2R population generation is bound to:

Git authority commit:

`624ea02130eddce416a92239a97d7728bd8aa5b1`

Generator path:

`scripts/build_controlled_v5.py`

Generator file SHA256:

`4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c`

The implementation must additionally record the Git blob SHA of this file from the authority commit.

No generator modification is permitted for K2R population construction.

## 5. Frozen claim-disjoint population rule

Generate:

`fact_templates_for_count(600)`

Then take exactly the Python slice:

`[300:600]`

This corresponds to exactly 300 global template indices:

300 through 599 inclusive.

The exact pair-ID range must be:

`generated_fact_301`

through:

`generated_fact_600`

with no gaps, duplicates, or replacements.

The selected 300 templates must then be passed together as one standalone population to the frozen generator record-construction semantics.

The standalone population construction is required because its base polarity split is defined internally across these 300 templates.

Expected generated full source row count:

`3900`

No other global template range may be used.

No random seed, shuffle, search, semantic-distance filtering, or native-state information may modify this range.

## 6. Frozen generated-source identity

The prospective state-blind feasibility audit at authority HEAD:

`624ea02130eddce416a92239a97d7728bd8aa5b1`

produced the canonical JSONL SHA256:

`e2bf70d8d25a7dca4c590d8ccb72db74308edbcd4ef8158f0482a8730d3978a2`

for the 3900 generated source records.

Canonical JSON encoding is:

- UTF-8;
- sorted keys;
- compact separators;
- ensure_ascii = false;
- allow_nan = false;
- one LF-terminated object per row.

Any generated-source identity mismatch is:

`POPULATION_PROVENANCE_FAILURE`

and stops K2R before native-state execution.

## 7. Frozen P/C/N extraction

For each of the 300 generated pairs define:

T:
exactly one `evidence_truncation` row.

E:
exactly one `entity_swap` row.

Q:
the unique REFUTE correction row chosen by the following deterministic rule:

1. if exactly one `polarity_flip` row has final label REFUTE, use it;
2. otherwise, if exactly one `none` row has final label REFUTE, use it;
3. otherwise fail closed.

Expected correction-source counts across all 300 pairs:

- `polarity_flip` = 150
- `none` = 150.

Require for every pair:

T:
- final label = NOT_ENTITLED;
- primary failure = sufficiency;
- sufficiency label = 0.

E:
- final label = NOT_ENTITLED;
- primary failure = frame;
- polarity label = NONE.

Q:
- final label = REFUTE;
- polarity label = REFUTE.

Require byte-identical claim text across T, E, and Q.

## 8. Exact textual construction

For pair i:

P_i =
`"Claim: " + T.claim + "\nEvidence: " + T.evidence + "\nAdditional evidence:\n"`

C_i =
Q.evidence

N_i =
E.evidence

No text rewriting is allowed.

Forbidden operations include:

- Unicode normalization;
- paraphrasing;
- whitespace repair;
- branch-specific template modification;
- added EOS;
- branch padding;
- truncation;
- continuation length equalization.

## 9. Frozen K2R candidate identity

For every item construct the canonical recipe with exactly:

- `schema_version = "k2r-independent-population-v1"`
- `generator_sha256`
- `global_template_start = 300`
- `global_template_stop = 600`
- `pair_id`
- `truncation_source_id`
- `correction_source_id`
- `correction_source_intervention`
- `control_source_id`
- `prefix_text`
- `correction_text`
- `control_text`

Canonical recipe JSON uses:

- sorted keys;
- compact separators;
- UTF-8;
- ensure_ascii = false;
- allow_nan = false.

Define:

`stable_item_id = "k2r-v1:" + SHA256(canonical_recipe_json)`

Define:

`base_claim_sha256 = SHA256(exact claim UTF-8 bytes)`

Sort the 300 candidate rows lexicographically by `stable_item_id`.

The canonical JSONL SHA256 of the complete sorted K2R candidate population is frozen as:

`00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4`

Any mismatch fails closed before model execution.

## 10. Claim-disjointness from K2S

K2R must verify against the frozen K2S candidate population:

`reports/longterm_k2w_fixed_window_phase_a_c7c7a0c218bb/candidate_pool.jsonl`

with SHA256:

`abf693d3267cc4e3dd27a8127d2948b36fdf8ba24e135a643215f0f31a26d808`

The state-blind prospective audit established:

- old/new pair-ID overlap = 0;
- old/new claim SHA overlap = 0;
- old/new exact claim-text overlap = 0.

Scientific execution must revalidate all three as zero.

Any overlap is:

`REPLICATION_POPULATION_OVERLAP_FAILURE`

and stops execution.

## 11. Reciprocal block assignment

After sorting K2R items by ascending `stable_item_id`, index:

i = 0,...,299.

Define:

`pi(i) = i XOR 1`

giving:

- 0 <-> 1
- 2 <-> 3
- ...
- 298 <-> 299.

This creates exactly 150 disjoint reciprocal blocks.

For recipient i and donor j = pi(i):

Matched branches:

- M_corr(i) = P_i + C_i
- M_ctrl(i) = P_i + N_i

Swapped branches:

- S_corr(i) = P_i + C_j
- S_ctrl(i) = P_i + N_j

The complete population must preserve exactly the correction and control continuation multisets between matched and swapped assignment.

No alternate pairing is permitted.

## 12. Frozen tokenizer/model identity

Use:

`state-spaces/mamba-130m-hf`

at exact Hugging Face revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Tokenizer requirements:

- fast tokenizer;
- `trust_remote_code = false`;
- `add_special_tokens = false`;
- no padding;
- no truncation.

Resolved tokenizer/config identities and SHA256s must be written to the scientific manifest.

## 13. Event anchor and W=8

For each recipient:

`p_i = len(tokens(P_i)) - 1`

The scientific event anchor is continuation onset after the final prefix token.

Primary fixed window:

k = 1,...,8.

Therefore all primary native-state quantities use:

S_(p_i+1) through S_(p_i+8),

with pre-event state:

S_(p_i)

and pre-event velocity requiring:

S_(p_i-1).

All four branches must have the exact prefix token sequence through p_i and at least eight post-prefix tokens.

The state-blind K2R feasibility result was:

- N matched valid = 300;
- N swapped valid = 300;
- failures = none.

Matched first corr/control divergence relative to p:

- d-p = 2 for 150 items;
- d-p = 3 for 150 items.

Swapped first divergence:

- d-p = 2 for 150 items;
- d-p = 3 for 150 items.

Matched correction post-prefix availability:

23 to 28 tokens.

Matched control post-prefix availability:

20 to 27 tokens.

Swapped correction post-prefix availability:

23 to 28 tokens.

Swapped control post-prefix availability:

20 to 27 tokens.

These tokenizer diagnostics are construction evidence only.

They are not scientific endpoints.

## 14. Native recurrent-state source

Use the same canonical authenticated seed180 encoder realization as K2S.

Handoff ZIP:

`C:\Users\Home1\Downloads\p3w7-seed8192-a0-seed180-replacement-r1-retry1_55debe94f0d1.zip`

Expected ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Expected selected checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Expected encoder canonical digest:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Expected secondary raw-concatenation digest:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

Expected encoder structure:

- 242 tensors;
- 129135360 total numel;
- 516541440 raw bytes;
- float32.

A0 head predictions remain forbidden as selection or promotion criteria.

## 15. Native-state semantics

Use the same direct selective-SSM recurrent-state semantics validated in O0c and K2S:

post-consumption native recurrent state `s_t` after token t updates the recurrent state and before downstream recurrent readout.

Capture all 24 layers for integrity/descriptive output.

The only confirmatory primary layer is:

`layer 23`.

This is retained because layer 23 was already prospectively frozen as the K2S primary layer before K2S scientific outcomes were observed.

It is not selected because of the K2S secondary layer scan.

Layers 0 through 22 are descriptive only.

## 16. Instrumentation requirements

Before K2R scientific execution, synthetic non-study instrumentation preflight must verify:

- authenticated checkpoint reconstruction;
- strict state-dict load;
- canonical and raw encoder fingerprint;
- direct recurrent-state capture at all 24 layers;
- correct state shape and float32 dtype;
- finite state;
- fresh-state isolation;
- trace-on versus trace-off ordinary-logit exact noninterference;
- exact causal common-prefix recurrent-state identity across branches for every prefix token and every layer;
- direct post-consumption recurrent-state timing.

Scientific execution must fail closed on any instrumentation/provenance violation.

No item dropping is allowed.

## 17. Frozen kinematic definitions

Use:

epsilon = `1e-12`.

For each branch and primary layer 23:

V_(p+k) =
S_(p+k) - S_(p+k-1)

for k = 1,...,8.

Mean speed:

R =
mean over k of `||V_(p+k)||_F`.

Turning:

turn_(p+k) =
`1 - cosine(V_(p+k), V_(p+k-1))`

using the pre-event velocity:

V_p =
S_p - S_(p-1)

for k=1.

Mean turning D is the mean over valid turn values.

Path length:

L =
sum over k of `||V_(p+k)||_F`.

Net displacement:

DISP =
`||S_(p+8) - S_p||_F`.

Trajectory efficiency:

P =
`DISP / (L + epsilon)`.

R, D, DISP, and P are the complete confirmatory K2R endpoint family.

Path length is retained descriptively but is not a fifth confirmatory endpoint because fixed W makes it a scalar multiple of R.

## 18. Recipient-level matched/swapped contrasts

For metric q in:

- R
- D
- DISP
- P

define:

Delta_q_matched =
q(M_corr) - q(M_ctrl)

Delta_q_swapped =
q(S_corr) - q(S_ctrl)

Define pair-specificity:

X_i_q =
`abs(Delta_q_matched) - abs(Delta_q_swapped)`

Positive X means matched semantic assignment produces the larger correction-versus-control magnitude.

Negative X means reciprocal swapped assignment produces the larger magnitude.

Signed matched and swapped deltas must also be retained as descriptive data.

## 19. Block-level inference unit

For each reciprocal block m containing items a and b:

B_m_q =
`(X_a_q + X_b_q) / 2`

The block, not the recipient item, is the confirmatory inference unit.

There are exactly 150 blocks.

No block may be removed, replaced, or reweighted based on its observed native-state values.

## 20. Prospectively frozen directional replication hypothesis

K2R freezes the following four directional predictions before inspecting any K2R native state:

### R

Expected:

`B_R > 0`

Matched semantic pairing should produce stronger local movement contrast.

### D

Expected:

`B_D > 0`

Matched semantic pairing should produce stronger local directional-change contrast.

### DISP

Expected:

`B_DISP < 0`

Reciprocal mismatch should produce stronger net-displacement contrast.

### P

Expected:

`B_P < 0`

Reciprocal mismatch should produce stronger trajectory-efficiency contrast.

These four directions are directly motivated by the frozen K2S result and secondary interpretation.

They are confirmatory hypotheses for K2R, not retrospective claims about K2S.

## 21. Primary statistics

For each of the four endpoints:

- count positive blocks;
- count negative blocks;
- count exact-zero blocks;
- count undefined blocks;
- report n_valid;
- exclude exact zeros from n_eff.

Promotion floor:

- n_valid >= 120;
- n_eff >= 30.

Use a two-sided exact binomial sign test under p=0.5.

If an endpoint fails its support floor:

raw p = 1 for confirmatory purposes.

Apply Holm familywise correction across exactly four endpoints:

- R
- D
- DISP
- P

with:

- m = 4
- alpha = 0.05
- deterministic tie order R < D < DISP < P.

For every endpoint report:

- raw p;
- Holm-adjusted p;
- Holm rejection;
- positive count;
- negative count;
- zero count;
- undefined count;
- n_valid;
- n_eff;
- rank-biserial sign effect.

No additional endpoint enters the confirmatory family.

## 22. Full replication criterion

K2R achieves:

`LOCAL_VS_NET_TRAJECTORY_DISSOCIATION_REPLICATED`

if and only if all four endpoints simultaneously satisfy:

1. promotion floor PASS;
2. Holm rejection;
3. preregistered directional sign:

   - R effect > 0;
   - D effect > 0;
   - DISP effect < 0;
   - P effect < 0.

All four are required.

R/D alone are insufficient.

DISP/P alone are insufficient.

Three-of-four is insufficient for full replication.

## 23. Non-replication verdicts

If valid execution completes but the full four-endpoint criterion is not met, K2R must not be called replicated.

Use:

`LOCAL_VS_NET_DISSOCIATION_NOT_FULLY_REPLICATED`

when the full criterion fails and no endpoint is Holm-significant in the direction opposite its preregistered prediction.

Use:

`LOCAL_VS_NET_DISSOCIATION_DIRECTIONAL_CONTRADICTION`

if one or more endpoints are Holm-significant but point opposite their preregistered K2R direction.

A partial directional match may be reported descriptively but does not authorize replication promotion.

## 24. Exact-zero and undefined behavior

Exact zeros are valid observations and remain reported.

They are excluded only from n_eff according to the frozen sign-test definition.

Undefined D is allowed only when required adjacent velocity norms make turning undefined.

If an entire block lacks a valid D value because one reciprocal member is undefined, that block is D-undefined.

R, DISP, and P must always be finite after valid instrumentation.

Any nonfinite value in R, DISP, or P is an instrumentation/integrity failure.

## 25. Secondary descriptions

Permitted non-rescuing descriptions include:

- layers 0 through 22;
- signed matched/swapped contrasts;
- path length;
- per-step speed;
- per-step turning;
- token-divergence location;
- block sign patterns;
- within-block endpoint concordance;
- continuation length;
- correction-source category (`none` vs `polarity_flip`).

These may explain the K2R outcome.

They may not create replication if the frozen four-endpoint rule fails.

## 26. Explicit anti-rescue rules

K2R may not:

- change W=8;
- change layer 23;
- select k=7;
- select another favorable time step;
- select a favorable intermediate layer;
- remove DISP or P;
- redefine efficiency;
- change absolute pair-specificity;
- alter reciprocal pairing;
- filter by A0 prediction;
- filter by confidence or margin;
- filter by correction source;
- remove zero blocks;
- drop unfavorable blocks;
- replace failed items;
- sweep thresholds;
- add confirmatory metrics after observing native-state results.

Any altered design is a new experiment.

## 27. Provenance manifest

K2R scientific output must bind at minimum:

- this committed K2R prereg identity;
- implementation commit;
- runtime HEAD and branch;
- runtime dirty-state contract;
- generator authority commit;
- generator path;
- generator SHA256;
- generator Git blob SHA;
- global template range 300:600;
- generated source canonical SHA256;
- generated source row count 3900;
- canonical K2R candidate SHA256;
- old/new overlap audit;
- all 300 stable IDs;
- all 150 reciprocal block mappings;
- exact P/C/N text;
- exact token arrays;
- tokenizer revision/files;
- p_i;
- W=8;
- handoff ZIP identity;
- checkpoint identity;
- encoder fingerprints;
- runtime package versions;
- instrumentation identity;
- noninterference result;
- common-prefix recurrent-state identity;
- all recipient X values;
- all block B values;
- complete primary statistics;
- all output artifact SHA256s.

## 28. Execution boundary

After this preregistration is independently reviewed and committed, it authorizes:

- deterministic K2R population materialization;
- bounded implementation;
- focused tests;
- state-blind/tokenizer population validation;
- synthetic instrumentation validation;
- one K2R scientific execution on the exact frozen 300-item claim-disjoint population;
- prespecified primary and secondary artifact generation.

It does not authorize:

- training;
- fine-tuning;
- learned probes;
- new metrics;
- hyperparameter sweeps;
- K3 intervention;
- architecture modification.

A separate execution-authority document is not required.

## 29. K3 boundary

A successful K2R full replication would constitute stronger evidence that the local-versus-net dissociation is a reproducible within-generator native-state phenomenon.

Even then, K3 execution is not automatic.

A subsequent K3 causal-intervention preregistration would still be required.

If K2R does not fully replicate, do not rescue the dissociation by choosing favorable layers/times/endpoints.

## 30. State-blind feasibility evidence

The K2R population feasibility audit was run before any K2R native-state observation.

It used no:

- checkpoint forward pass;
- A0 head;
- confidence;
- margin;
- hidden state;
- recurrent state;
- R/D/DISP/P outcome.

Observed prospective construction identities:

- authority HEAD:
  `624ea02130eddce416a92239a97d7728bd8aa5b1`

- generator SHA256:
  `4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c`

- global template range:
  `300:600`

- new pairs:
  `300`

- first pair:
  `generated_fact_301`

- last pair:
  `generated_fact_600`

- generated source rows:
  `3900`

- generated source canonical SHA256:
  `e2bf70d8d25a7dca4c590d8ccb72db74308edbcd4ef8158f0482a8730d3978a2`

- K2R candidate canonical SHA256:
  `00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4`

- old/new pair-ID overlap:
  `0`

- old/new claim SHA overlap:
  `0`

- old/new exact claim-text overlap:
  `0`

- correction sources:
  `none = 150`
  `polarity_flip = 150`

- reciprocal blocks:
  `150`

- matched valid:
  `300/300`

- swapped valid:
  `300/300`

- prefix marginal:
  `PASS`

- correction marginal:
  `PASS`

- control marginal:
  `PASS`

- feasibility failures:
  `0`

This evidence establishes design feasibility only.

## 31. Final preregistration state

`K2R_POPULATION_CLAIM_DISJOINT_FROM_K2S = YES`

`K2R_EXTERNAL_DISTRIBUTION_REPLICATION = NO`

`K2R_NATIVE_STATE_OBSERVED = NO`

`K2R_SCIENTIFIC_RESULT_AVAILABLE = NO`

`K2R_K3_AUTHORIZATION = NO`

`PASS_READY_TO_FREEZE_K2R_PREREG`

This is preregistration readiness, not scientific evidence.
