# O0b - Token-Aligned Native Hidden-State Precursor Disambiguation

Status: STATIC SCIENTIFIC DESIGN / AUTHORITY-SPEC CANDIDATE ONLY

This document is a candidate authority/specification for a future O0b
matched-control experiment. It does not authorize implementation, dataset
generation, model download, tokenizer execution, model forward, training,
evaluation, Kaggle execution, promotion, or scientific claims.

## 1. Authority and scope boundary

Authority order for this candidate:

1. Current controller instruction for this task card.
2. Canonical O0a evidence commit:
   `f7241abea9a09b54ff3b8ee66cacbd7f4feebb14`.
3. `docs/CONTRAMAMBA_RESEARCH_HYPOTHESIS_MAP.md`.
4. `docs/CONTRAMAMBA_RESEARCH_VISION.md`.
5. Existing O0a design/execution authorities and canonical O0a artifacts only
   as prior evidence/context.

URP/reason-router authority, artifacts, attempts, checkpoints, files, and
conclusions are explicitly unrelated to O0b. O0b must not consume, modify, or
reinterpret any URP/reason-router material.

O0b preserves the O0a measurement boundary:

- frozen native pretrained Mamba only;
- Hugging Face `MambaModel` layer hidden states remain hidden-state proxies;
- not the selective-SSM recurrent state;
- no `cache_params` instrumentation;
- no direct A/B/C/Delta dynamics;
- no generation;
- no training;
- no learned classifier;
- no threshold tuning;
- no best-layer selection.

Unless a later implementation/execution authority documents a technically
necessary reason before any model execution, O0b must preserve:

```text
model/tokenizer = state-spaces/mamba-130m-hf
immutable HF revision = 5708daa364c50b880e7bd92eab456e0d34492ee9
device = CPU
dtype = float32
mode = eval / frozen / torch.inference_mode()
tokenization = add_special_tokens=False
```

## 2. Scientific question

After controlling token count, divergence position, terminal position, and
claim surface form, do native Mamba hidden-state proxies retain an early
response that distinguishes evidence sufficiency failure from semantically
sufficient controls?

The O0b target is narrower than a hallucination detector. It asks whether the
strongest O0a early hidden-state-proxy divergence, especially
`evidence_deletion`, survives a stronger matched-control design. It does not
observe model generation or an emitted unsupported commitment.

## 3. Relationship to O0a

The canonical O0a evidence commit is:

```text
f7241abea9a09b54ff3b8ee66cacbd7f4feebb14
```

O0a found early intervention-sensitive hidden-state-proxy divergence in frozen
native Mamba layer hidden states. The `evidence_deletion` clue was strong
descriptively, but it was confounded by unequal evidence/token lengths,
terminal positions, fractional-prefix alignment, and whole-pair paraphrase
controls that could change both claim and evidence wording.

O0b exists specifically to disambiguate that O0a clue. It does not
retroactively change O0a artifacts, O0a claims, O0a hashes, or O0a scientific
boundaries.

## 4. Required control structure

Each O0b matched set must contain one byte-identical claim and one byte-identical
serialization scaffold across all primary members. Only evidence wording may
change.

Required primary members per matched set:

- `reference_sufficient`: semantically sufficient evidence for the identical
  claim.
- `paraphrase_sufficient`: evidence-only paraphrase that remains semantically
  sufficient.
- `insufficient_matched`: token-count-matched evidence that removes or replaces
  conclusion-critical support while preserving frame and predicate as much as
  possible.
- `surface_null_matched`: token-count-matched surface or lexical change that
  preserves entitlement/sufficiency.

The O0a whole-pair paraphrase confound is forbidden in O0b primary comparisons:
claim text must be byte-identical within each comparison set.

The primary O0b design is evidence-sufficiency disambiguation. Predicate/entity
controls may be retained only as secondary controls if they satisfy the same
token-count, claim-identity, serialization, and first-divergence alignment
discipline. Polarity/refutation, if retained, remains an AUTHORIZED semantic
sensitivity control and must not be relabeled as a NOT_ENTITLED or sufficiency
failure case.

## 5. Dataset contract

O0b requires a small deterministic matched-control dataset, but this candidate
does not create it.

The future dataset must contain at least three independent matched set IDs:

```text
o0b_pair_001
o0b_pair_002
o0b_pair_003
```

More pair IDs may be added only by later authority before tokenization or model
execution. O0b is still a tiny mechanistic screening design; it does not
authorize population, significance, or generalization claims.

Each matched set record must specify:

- `pair_id`;
- byte-identical `claim` for every member in the set;
- `reference_sufficient` evidence;
- `paraphrase_sufficient` evidence;
- `insufficient_matched` evidence;
- `surface_null_matched` evidence;
- intended sufficiency semantics explaining why the insufficient member is
  insufficient;
- tokenizer-derived exact token counts for every serialized member;
- exact token IDs for every serialized member;
- first divergent token index for each comparison to reference;
- terminal token index for each primary comparison;
- validation result proving no accidental unmatched lengths enter primary
  comparisons.

The exact serialization must be frozen by a later implementation authority
before tokenizer validation. The default inherited O0a serialization candidate
is:

```text
Claim: <claim>
Evidence: <evidence>
```

Any later change to this serialization must be justified before dataset
validation and model execution, and the exact string template must be included
in provenance artifacts.

Dataset validation tooling must be authorized and run later before any model
execution. Manual belief that evidence lengths match is insufficient. The
validator must tokenize with the authority-bound tokenizer and
`add_special_tokens=False`, record the exact metadata above, and fail closed
before model loading if any primary matched set violates the contract.

## 6. Sufficiency semantics

The insufficient member must be insufficient because conclusion-critical support
has been removed or replaced, not because of an obvious surface shortcut.

Required insufficiency properties:

- preserve the same claim;
- preserve the same broad evidence frame when possible;
- preserve entity and predicate coverage as much as possible;
- remove or replace the fact, relation, qualifier, quantity, date, source, or
  bridge that is necessary to justify the exact claim;
- avoid empty evidence, deletion punctuation, dangling syntax, repeated filler,
  obvious truncation, or other length/surface artifacts that could be detected
  without semantic entitlement reasoning.

The surface/null member must introduce an equal-token-count wording or lexical
change that preserves entitlement/sufficiency. It estimates ordinary token
identity sensitivity under the same length and position discipline.

## 7. Token-count and position discipline

For every primary comparison within a matched set:

```text
token_count(reference_sufficient)
== token_count(paraphrase_sufficient)
== token_count(insufficient_matched)
== token_count(surface_null_matched)
```

The comparison must also satisfy:

- claim bytes are identical;
- serialization scaffold bytes are identical;
- `add_special_tokens=False`;
- terminal comparison positions are identical;
- primary terminal hidden states are never compared across unequal sequence
  lengths;
- accidental unmatched lengths are a validation failure.

Unequal-length diagnostics, if ever added by later authority, must be marked
secondary/non-inferential and may not support the O0b primary claim.

## 8. First-divergent-token alignment

O0b replaces O0a's primary fractional-prefix interpretation with exact token
index alignment around the first intervention-divergent evidence token.

For each comparison to `reference_sufficient`, the validator must find:

```text
first_divergent_token_index = first token position where token IDs differ
```

This index must occur inside the evidence region, not inside the claim or
serialization scaffold. If a comparison has no divergent token, it is invalid
for primary O0b inference unless a later authority explicitly defines it as a
separate identity sanity case.

The frozen O0b anchor schedule is:

```text
anchor_pre_minus_1 = first_divergent_token_index - 1
anchor_divergence = first_divergent_token_index
anchor_post_plus_1 = first_divergent_token_index + 1
anchor_post_plus_2 = first_divergent_token_index + 2
anchor_post_plus_4 = first_divergent_token_index + 4
anchor_terminal = final token index, only when sequence lengths are matched
```

Anchors whose indices would exceed the matched sequence length are recorded as
unavailable JSON `null` and excluded from that anchor-specific aggregate. They
must not be replaced by a different offset after seeing results.

The primary early-response anchors are `anchor_divergence`,
`anchor_post_plus_1`, `anchor_post_plus_2`, and `anchor_post_plus_4`.
`anchor_terminal` is a matched terminal-position diagnostic and may be primary
only because the primary dataset forbids unequal sequence lengths.

## 9. Pre-divergence invariant

For claim-identical comparison pairs whose serialized token sequences are
identical before the first intervention token, hidden-state distance must be
zero within a frozen deterministic tolerance before divergence.

Required tolerance:

```text
pre_divergence_abs_tolerance = 1e-6
pre_divergence_relative_tolerance = 0
```

The invariant applies to every exposed layer at `anchor_pre_minus_1` and to any
earlier audited prefix position. A violation is an execution/implementation
failure, not scientific signal. A future observer must fail closed and mark the
run invalid rather than interpreting pre-divergence nonzero distance as native
semantic sensitivity.

## 10. Primary measurement

The primary measurement is paired aligned-position hidden-state distance by
layer:

```text
D_l2(layer, anchor, member, reference)
= Euclidean distance between unit-normalized hidden-state vectors
```

The allowed primary anchors are the frozen anchors in Section 8. `anchor_terminal`
is allowed only for equal-length primary comparisons.

Cosine distance may be recorded for audit convenience, but for unit-normalized
vectors:

```text
D_l2^2 = 2 * D_cos
```

Therefore normalized L2 and cosine are algebraically redundant and must not be
counted as independent evidence.

Trajectory deltas, acceleration, terminal norm, evidence-region mean delta, and
similar quantities may remain descriptive secondary diagnostics. No unweighted
heterogeneous trajectory-summary scalar may be used as independent evidence.

O0b must not create a learned aggregate score, threshold-tuned score, selected
best layer, selected best anchor, or post hoc promotion rule.

## 11. A priori comparison logic

For every matched set, layer, and frozen anchor, O0b must record these primary
comparisons:

```text
A. D_l2(insufficient_matched, reference_sufficient)
B. D_l2(paraphrase_sufficient, reference_sufficient)
C. D_l2(surface_null_matched, reference_sufficient)
```

The scientific contrast is not a classifier. It is a structured comparison of
whether A shows a consistently larger aligned-position response than the two
semantic-preserving matched controls B and C under identical claim, identical
serialization, equal token count, matched terminal position, and frozen
first-divergent-token anchors.

Reporting must preserve pair-level and layer-level values. Aggregate summaries
may include means/medians by anchor and layer, plus counts of pair IDs where:

```text
A > B
A > C
```

These counts are descriptive only. They are not hard PASS thresholds, not
statistical tests, and not layer-selection criteria.

## 12. Interpretation and falsification matrix

O0b has no authorized hard scientific PASS threshold. The following outcomes
must be interpreted without post hoc favorable-layer selection:

- If insufficiency separation survives matched token/position controls and
  exceeds matched semantic-preserving controls consistently across multiple
  pair IDs and layers, O0b supports a narrower native sufficiency-sensitive
  precursor clue.
- If the O0a deletion effect collapses after token/position matching, O0a
  deletion divergence should be reinterpreted primarily as length, position,
  and/or surface confounding.
- If all semantic manipulations produce similar separation, native hidden-state
  distance is intervention-sensitive but not sufficiency-specific.
- If evidence-only sufficient paraphrases separate as strongly as
  insufficiency, the measured response is more likely broad wording sensitivity
  than entitlement-specific sufficiency sensitivity.
- If only isolated pair IDs, layers, or anchors respond, treat the result as
  weak/unstable and do not select favorable layers post hoc.
- If the pre-divergence invariant fails, the run is invalid as an
  execution/implementation failure.
- If any primary comparison uses unequal sequence lengths or unmatched terminal
  positions, the primary O0b claim is invalid regardless of observed distances.

No statistical-significance, population-level, or generalization claim is
authorized by this tiny toy screening design.

## 13. Required future provenance

Any later implementation/execution authority must freeze:

- exact repository commit;
- observer SHA256;
- dataset LF/runtime SHA256;
- model and tokenizer IDs;
- immutable model/tokenizer revision
  `5708daa364c50b880e7bd92eab456e0d34492ee9`;
- runtime versions;
- CPU device;
- float32 dtype;
- exact serialization;
- exact tokenizer setting `add_special_tokens=False`;
- exact tokenized pair metadata;
- exact first-divergence indices;
- exact anchor schedule;
- deterministic tolerance values;
- exact run command;
- required artifact list;
- `SHA256SUMS.txt` covering every required artifact.

The observer must fail closed before model loading on repository-head mismatch,
observer hash mismatch, dataset hash mismatch, mutable model/tokenizer revision,
non-CPU device, non-float32 dtype, invalid serialization, invalid token-count
matching, claim mismatch, first-divergence outside evidence, pre-divergence
invariant failure, unequal-length primary comparison, non-finite metric, or
incomplete artifact set.

## 14. Deferred work

Selective-SSM/cache/A/B/C/Delta instrumentation remains deferred. O0c/O1 deeper
native-state instrumentation should be considered only if O0b leaves a
matched-control signal worth explaining. M3 semantic ownership or architecture
modification remains premature.

Guiding order:

```text
Measure first.
Explain second.
Modify third.
```

## 15. Explicit non-authorizations

This candidate does not authorize:

- editing existing docs, reports, scripts, tests, source files, data files, root
  patch files, URP files, reason-router files, or canonical O0a artifacts;
- creating the O0b dataset;
- tokenizer execution;
- model loading;
- model forward pass;
- training;
- evaluation;
- generation;
- learned classifiers;
- threshold tuning;
- best-layer or best-anchor selection;
- Kaggle execution;
- commit or push.

Any later phase must receive separate explicit authority.
