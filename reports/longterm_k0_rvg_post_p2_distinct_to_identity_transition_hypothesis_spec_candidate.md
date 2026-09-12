# K0-RVG Post-P2 Distinct-to-Identity Transition Hypothesis Specification Candidate

**Status:** scientific hypothesis/specification candidate only.

**Date:** 2026-09-12

**Immediate parent / frozen P2 validated interpretation commit:**

`6a342810857d55da8e9d23b5910da1d27b8ea96a`

**Frozen P2 validated interpretation report:**

`reports/longterm_k0_rvg_p2_validated_scientific_interpretation_report_candidate.md`

**Frozen P2 validated interpretation report SHA256:**

`f3e3aa4df9d9ff92a61f02baf5cb1b008e513afcba5d1c041fa6359ba4733c28`

This document specifies the next scientific question after the validated K0-RVG-P2 result.

It does not authorize implementation, tokenizer execution, model construction, model forward, checkpoint loading, recurrent-state capture or reread, training, evaluation, causal intervention, learned geometry, hyperparameter search, Kaggle execution, or K4.

The historical K-series A/B fork remains context only and is not reopened.

## 1. Active scientific line

`ACTIVE_SCIENTIFIC_LINE = K0_RVG_RAW_NATIVE_VECTOR_GEOMETRY`

The frozen P1 scientific result remains:

`RAW_NATIVE_VECTOR_ORGANIZATION_NOT_ESTABLISHED`

The frozen P2 localization remains:

`P2_PRIMARY_RESULT = ENDPOINT_SUPPORTING_STATE_IDENTITY_LOCALIZES_335_OF_336_P1_ZERO_CONTRAST_ITEMS`

and:

`P2_DOMINANT_DEGENERACY_LOCALIZATION = UPSTREAM_OF_PERSISTED_ENDPOINT_SUPPORTING_STATE_WINDOW_OR_AT_ITS_INPUT_BOUNDARY`

P2 does not establish raw native vector organization and does not rescind P1.

## 2. Why the next question must be sharpened

A literal question such as:

> What is the earliest boundary at which matched and swapped are distinct?

is already trivial at the surface-text level.

The frozen P0 candidate pool preserves distinct correction and control continuation text, and P2 established:

`correction_continuation_text_relation = DIFFERENT` for `336 / 336`

`control_continuation_text_relation = DIFFERENT` for `336 / 336`

Therefore the scientifically useful question is not whether any upstream distinction exists.

The useful question is:

> Between the known distinct state-blind construction and the known endpoint-supporting state identity, where is the last verified distinction and where is the first verified identity?

This is a distinct-to-identity transition-localization problem.

`POST_P2_OBJECTIVE = DISTINCT_TO_IDENTITY_TRANSITION_LOCALIZATION`

## 3. Critical interpretation correction

The P1 event anchor `t_e` is not a matched-versus-swapped divergence anchor.

It is the first correction-versus-control token divergence within each matched or swapped pair.

The frozen P1 runner reconstructs four branches for each local item using one item-specific prefix:

- `matched_corr = item prefix + item correction`
- `matched_ctrl = item prefix + item control`
- `swapped_corr = item prefix + phase-mate correction`
- `swapped_ctrl = item prefix + phase-mate control`

Thus matched and swapped use the same local prefix but different state-blind continuation text.

The frozen P1 runner then tokenizes all four full branch strings and computes separately:

`matched_te = divergence_anchor(matched_corr_ids, matched_ctrl_ids, prefix_len)`

`swapped_te = divergence_anchor(swapped_corr_ids, swapped_ctrl_ids, prefix_len)`

The endpoint-supporting state window is:

`t_e - 1, t_e, ..., t_e + 7`

P2 compares matched versus swapped state hashes after event-relative alignment to those pair-internal `t_e` anchors.

Therefore P2 state-hash identity does not, by itself, prove that the model mapped different token sequences to identical state.

The matched-versus-swapped token sequences at the compared coordinates were not persisted in P0/P1/P2.

`P1_EVENT_ANCHOR_SEMANTICS = CORRECTION_VERSUS_CONTROL_WITHIN_PAIR`

`P1_EVENT_ANCHOR_IS_MATCHED_VERSUS_SWAPPED_DIVERGENCE = NO`

## 4. Frozen token-contract facts

The frozen P0 token contract persists, per item:

- `prefix_token_count`;
- `prefix_token_sha256`;
- matched correction/control total token counts;
- swapped correction/control total token counts;
- matched correction/control divergence anchor;
- swapped correction/control divergence anchor;
- divergence offsets from prefix;
- W=8 availability;
- phase-mate identifiers.

It does not persist:

- full `matched_corr` token-ID sequence;
- full `matched_ctrl` token-ID sequence;
- full `swapped_corr` token-ID sequence;
- full `swapped_ctrl` token-ID sequence;
- matched-versus-swapped first token-difference coordinate;
- per-coordinate matched-versus-swapped token equality over the P1 state window.

The P1 runtime did reconstruct those full token sequences transiently in order to validate the archived contract, but they were not emitted as scientific artifacts.

`FULL_MATCHED_SWAPPED_TOKEN_ID_RELATION_PERSISTED = NO`

`MATCHED_SWAPPED_FIRST_TOKEN_DIFFERENCE_PERSISTED = NO`

`MATCHED_SWAPPED_WINDOW_TOKEN_RELATION_PERSISTED = NO`

## 5. Event-window geometry of the frozen contract

The P0 contract histogram establishes that the corr-versus-ctrl divergence offset from the prefix is either 1 or 2.

For representative archived rows:

`prefix_token_count = 41`

`matched_divergence_anchor = 42`

`matched_divergence_offset_from_prefix = 1`

and the swapped anchor is the same.

Thus, when the divergence offset is 1:

- event-relative `k=-1` corresponds to the first continuation token position;
- `k=0` corresponds to the corr-versus-ctrl divergence position;
- `k=+1..+7` are the next seven positions.

The exact matched-versus-swapped token relation over those positions remains unknown from persisted artifacts alone.

## 6. Frozen P2 downstream boundary

P2 provides a strong downstream identity boundary.

For `335 / 336` items, all of the following are exact identity:

- turning components;
- response-coherence summary;
- all six persisted response/carry/write diagnostics;
- all 72 endpoint-supporting matched-versus-swapped state-hash comparisons.

The 72 state-hash comparisons are:

`2 branch-role pairs * 9 event-relative coordinates * 4 fields`

for:

`matched_corr <-> swapped_corr`

`matched_ctrl <-> swapped_ctrl`

over:

`k = -1..7`

and:

`S_prev_sha256`

`G_sha256`

`W_sha256`

`S_post_sha256`

Therefore:

`P2_DOWNSTREAM_VERIFIED_IDENTITY_BOUNDARY = LAYER23_ENDPOINT_SUPPORTING_NATIVE_RECURRENCE_WINDOW_K_MINUS_1_TO_PLUS_7`

for the dominant 335-item set.

## 7. Frozen upstream boundary

The P0 state-blind construction provides a verified upstream distinction:

- local prefix is held fixed within each item's four branch reconstructions;
- matched continuation uses the local item's correction/control text;
- swapped continuation uses the phase mate's correction/control text;
- P2 static comparison established correction text differs `336 / 336`;
- P2 static comparison established control text differs `336 / 336`.

Therefore:

`P0_UPSTREAM_VERIFIED_DISTINCT_BOUNDARY = SURFACE_CONTINUATION_TEXT`

However, surface-text difference is not equivalent to token-ID difference at a particular token coordinate.

No token-level claim may be inferred from text inequality alone.

## 8. Primary unresolved boundary

The missing boundary is:

`TOKENIZED_MATCHED_VERSUS_SWAPPED_ENDPOINT_WINDOW`

Specifically, for each item and branch role:

- correction: `matched_corr` versus `swapped_corr`;
- control: `matched_ctrl` versus `swapped_ctrl`;

the unresolved quantities are:

1. exact full token-ID equality or inequality;
2. first token-difference coordinate;
3. first token-difference coordinate relative to the appropriate `t_e`;
4. exact equality/inequality at every event-relative coordinate `k=-1..7`;
5. whether any earlier token difference exists before `k=-1`.

`POST_P2_PRIMARY_UNKNOWN = MATCHED_SWAPPED_TOKEN_WINDOW_RELATION`

## 9. Competing explanatory hypotheses

### H-TOKEN-WINDOW-IDENTITY

For the dominant 335 P2 state-identity items, matched and swapped token IDs are also exactly identical throughout the full endpoint-supporting event window for both branch roles.

Formally, for an item in the dominant set:

`matched_corr_ids[matched_te + k] == swapped_corr_ids[swapped_te + k]`

and:

`matched_ctrl_ids[matched_te + k] == swapped_ctrl_ids[swapped_te + k]`

for every:

`k in {-1,0,1,2,3,4,5,6,7}`

If this holds, the 72/72 state-hash identity can be explained at least locally by token-window input identity.

This would show that P1's observation window did not actually expose a matched-versus-swapped token distinction for those coordinates.

It would not prove that the full branch histories are identical.

`H_TOKEN_WINDOW_IDENTITY_STATUS = OPEN`

### H-PREWINDOW-TOKEN-DIFFERENCE

Matched and swapped have at least one token-ID difference before event-relative `k=-1`, but their endpoint-supporting state hashes are exact identity from `k=-1` onward.

If supported, this would bracket the distinct-to-identity transition between the earlier token history and the first persisted endpoint-supporting state boundary.

It would be evidence of a nontrivial transition in the deterministic processing chain, but not yet a causal mechanism.

`H_PREWINDOW_TOKEN_DIFFERENCE_STATUS = OPEN`

### H-INWINDOW-TOKEN-DIFFERENCE_WITH_STATE-IDENTITY

At least one matched-versus-swapped token-ID difference occurs within event-relative `k=-1..7`, while the corresponding P2 endpoint-supporting state hashes remain exact identity.

If supported, this would be the strongest static evidence that input distinction survives into the observation window but is not represented by the frozen layer-23 recurrence fields as distinct bytes at the compared coordinates.

This still must not be described as a causal "collapse" without a separately authorized mechanistic test.

`H_INWINDOW_TOKEN_DIFFERENCE_WITH_STATE_IDENTITY_STATUS = OPEN`

### H-STATIC-EVIDENCE-INSUFFICIENT

The frozen artifacts may be insufficient to reconstruct the exact token-ID relation without re-instantiating the exact frozen tokenizer.

If so, the correct result is not to guess.

The correct result is:

`EXACT_TRANSITION_NOT_IDENTIFIABLE_FROM_FROZEN_STATIC_ARTIFACTS`

followed, only under later authority, by a narrowly authenticated tokenizer-only audit.

`H_STATIC_EVIDENCE_INSUFFICIENT_STATUS = OPEN`

## 10. Required classification semantics

A future post-P2 token-boundary audit, if separately authorized, must classify each branch role and item into one of the following descriptive categories:

`TOKEN_SEQUENCES_EXACTLY_IDENTICAL_FULL_BRANCH`

`TOKEN_SEQUENCES_DIFFER_BEFORE_ENDPOINT_WINDOW`

`TOKEN_SEQUENCES_FIRST_DIFFER_AT_WINDOW_K_MINUS_1`

`TOKEN_SEQUENCES_FIRST_DIFFER_WITHIN_WINDOW_K_0_TO_PLUS_7`

`TOKEN_SEQUENCES_FIRST_DIFFER_AFTER_ENDPOINT_WINDOW`

`TOKEN_RELATION_NOT_IDENTIFIABLE_FROM_AUTHORIZED_EVIDENCE`

For the 335 P2 state-identity items, a cross-level classification may then be made as:

`TOKEN_WINDOW_IDENTITY_SUPPORTS_STATE_IDENTITY`

`PREWINDOW_TOKEN_DIFFERENCE_PRECEDES_STATE_IDENTITY`

`INWINDOW_TOKEN_DIFFERENCE_WITH_STATE_IDENTITY`

`TOKEN_BOUNDARY_UNRESOLVED`

These are localization labels, not causal labels.

## 11. Falsification logic

### H-TOKEN-WINDOW-IDENTITY is falsified for an item

if any exact matched-versus-swapped token-ID difference is authenticated at any event-relative coordinate `k=-1..7` in either correction or control role.

### H-PREWINDOW-TOKEN-DIFFERENCE is falsified for an item

if full authenticated branch token sequences are identical before `k=-1`, or if the first difference is at or after `k=-1`.

### H-INWINDOW-TOKEN-DIFFERENCE_WITH_STATE-IDENTITY is falsified for an item

if no token-ID difference occurs within `k=-1..7`, or if the item is not in the P2 state-hash-identity set.

### Exact transition localization is not established

unless the token evidence and state evidence are both provenance-authenticated and aligned under the exact frozen P1 event-anchor semantics.

No tolerance-based token equality is allowed.

## 12. Special handling of item 163

The unique P2 common exception is:

`local_template_index = 163`

It is the sole item with:

- turning difference;
- coherence difference;
- all six persisted diagnostic-summary differences;
- endpoint-supporting state-hash difference.

Its state-hash comparison has:

`61 / 72` exact equal

`11 / 72` different

with the first frozen comparison-order difference at:

`branch_role = ctrl`

`relative_coordinate = +5`

`tensor_field = G_sha256`

Item 163 is scientifically useful as a reference exception.

It is not authorized as a post-hoc subgroup for optimization or metric selection.

A future token-boundary audit should report item 163 under the same deterministic rules as all other items.

## 13. Six token-count mismatch items

The frozen P2 construction audit identified exactly six items with matched-versus-swapped correction/control total token-count mismatch:

`63, 95, 163, 231, 263, 331`

Five of those six remain endpoint-supporting state-hash identity items.

Therefore total token-count mismatch is not sufficient to explain state-hash difference.

A future audit may report these six items descriptively but may not treat them as a privileged subgroup or define a new endpoint around them.

## 14. What a future static token audit may need

The exact matched-versus-swapped token-ID relation is not stored in the frozen P0/P1/P2 artifacts.

The frozen P1 implementation shows that it was obtained by applying the authenticated tokenizer to reconstructed branch text.

Therefore a later audit may require exact reproduction of the tokenizer-only transformation.

If that later path is authorized, it must authenticate at minimum:

- frozen P0 archive SHA256 identities;
- exact P1 branch reconstruction semantics;
- exact tokenizer model ID;
- exact tokenizer revision;
- exact Transformers version or an equivalently frozen tokenizer implementation contract;
- no model construction;
- no checkpoint loading;
- no model forward;
- no recurrent-state read;
- no logits read;
- no network fallback;
- deterministic token-ID serialization.

This document does not authorize that action.

## 15. Prohibited shortcuts

The following are not scientifically valid substitutes for the missing token relation:

- inferring token equality from equal total token counts;
- inferring token difference from different surface text;
- inferring matched-versus-swapped divergence from corr-versus-ctrl `t_e`;
- retokenizing with an unpinned tokenizer;
- allowing a network-resolved latest tokenizer;
- treating a tokenizer-only result as a model-forward result;
- treating hash identity as proof of semantic identity;
- treating hash difference as a numerical distance;
- choosing a new layer/window after seeing the result;
- PCA, SVD, whitening, Mahalanobis distance, learned probes, or tuned geometry;
- subgroup optimization around item 163 or the six token-count mismatch items.

## 16. Scientific decision table

If a later authenticated token audit establishes token-window identity for the dominant 335 items:

`INTERPRETATION = P1_MATCHED_SWAPPED_OBSERVATION_WINDOW_INPUT_EQUIVALENCE`

The next question would concern construction design or observation-boundary validity, not a more complex native-state metric.

If it establishes pre-window token difference but window token identity:

`INTERPRETATION = DISTINCT_INPUT_HISTORY_PRECEDES_ENDPOINT_WINDOW_IDENTITY`

The transition is bracketed before the persisted state window.

If it establishes in-window token difference with P2 state identity:

`INTERPRETATION = DISTINCT_TOKEN_WINDOW_WITH_IDENTICAL_PERSISTED_LAYER23_STATE_SUPPORT`

This would justify a narrowly specified mechanistic localization question, but still not an immediate geometry sweep.

If exact token relation cannot be authenticated:

`INTERPRETATION = EXACT_TRANSITION_NOT_IDENTIFIABLE_FROM_FROZEN_STATIC_ARTIFACTS`

No scientific branch should be chosen by guesswork.

## 17. Current scientific conclusion

The current validated evidence supports only a transition bracket:

upstream:

`DISTINCT_SURFACE_CONTINUATION_TEXT`

downstream for 335 items:

`IDENTICAL_PERSISTED_LAYER23_ENDPOINT_SUPPORTING_STATE_HASHES`

The exact transition between them is unresolved because the matched-versus-swapped token-ID relation was not persisted.

Therefore:

`CURRENT_TRANSITION_RESOLUTION = TRANSITION_BRACKET_ONLY`

`EXACT_TRANSITION_IDENTIFIED = NO`

This is compatible with the frozen P2 statement that the dominant degeneracy lies:

`UPSTREAM_OF_PERSISTED_ENDPOINT_SUPPORTING_STATE_WINDOW_OR_AT_ITS_INPUT_BOUNDARY`

## 18. Next authorized research boundary

The next routine research step after this hypothesis specification is frozen is to draft a **static token-window audit authority specification**.

That later authority should answer only the token relation needed to discriminate the competing hypotheses above.

It should not authorize model execution.

It should prefer already persisted evidence; if exact tokenizer reproduction is required, it must be separately authenticated and tokenizer-only.

No P2 rerun is needed or authorized.

## 19. Final hypothesis markers

`HISTORICAL_AB_FORK_AUTHORITY = CONTEXT_ONLY_SUPERSEDED_BY_K0_RVG`

`P2_VALIDATED_INTERPRETATION_COMMIT = 6a342810857d55da8e9d23b5910da1d27b8ea96a`

`POST_P2_OBJECTIVE = DISTINCT_TO_IDENTITY_TRANSITION_LOCALIZATION`

`P0_UPSTREAM_VERIFIED_DISTINCT_BOUNDARY = SURFACE_CONTINUATION_TEXT`

`P2_DOWNSTREAM_VERIFIED_IDENTITY_BOUNDARY = LAYER23_ENDPOINT_SUPPORTING_NATIVE_RECURRENCE_WINDOW_K_MINUS_1_TO_PLUS_7`

`P1_EVENT_ANCHOR_SEMANTICS = CORRECTION_VERSUS_CONTROL_WITHIN_PAIR`

`P1_EVENT_ANCHOR_IS_MATCHED_VERSUS_SWAPPED_DIVERGENCE = NO`

`FULL_MATCHED_SWAPPED_TOKEN_ID_RELATION_PERSISTED = NO`

`MATCHED_SWAPPED_FIRST_TOKEN_DIFFERENCE_PERSISTED = NO`

`MATCHED_SWAPPED_WINDOW_TOKEN_RELATION_PERSISTED = NO`

`POST_P2_PRIMARY_UNKNOWN = MATCHED_SWAPPED_TOKEN_WINDOW_RELATION`

`H_TOKEN_WINDOW_IDENTITY_STATUS = OPEN`

`H_PREWINDOW_TOKEN_DIFFERENCE_STATUS = OPEN`

`H_INWINDOW_TOKEN_DIFFERENCE_WITH_STATE_IDENTITY_STATUS = OPEN`

`H_STATIC_EVIDENCE_INSUFFICIENT_STATUS = OPEN`

`CURRENT_TRANSITION_RESOLUTION = TRANSITION_BRACKET_ONLY`

`EXACT_TRANSITION_IDENTIFIED = NO`

`TOKENIZER_REEXECUTION_AUTHORIZED = NO`

`RETOKENIZATION_AUTHORIZED = NO`

`MODEL_CONSTRUCTION_AUTHORIZED = NO`

`CHECKPOINT_LOADING_AUTHORIZED = NO`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`LEARNED_OR_TUNED_GEOMETRY_AUTHORIZED = NO`

`INFERENTIAL_TESTING_AUTHORIZED = NO`

`KAGGLE_EXECUTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

`POST_P2_STATIC_TOKEN_AUDIT_IMPLEMENTATION_AUTHORIZED = NO`

`POST_P2_STATIC_TOKEN_AUDIT_EXECUTION_AUTHORIZED = NO`

`NEXT_BOUNDARY = K0_RVG_POST_P2_TOKEN_WINDOW_STATIC_AUDIT_AUTHORITY_SPECIFICATION`

This document becomes the frozen post-P2 distinct-to-identity transition hypothesis specification only after this exact document is committed and pushed as the immediate one-file child of:

`6a342810857d55da8e9d23b5910da1d27b8ea96a`
