# Native Recurrent Generator-Anchor Pair Localization — Scientific Design Candidate

Status: `DESIGN_READY_FOR_FREEZE`

Authority/evidence base:

- validated serialization-region evidence freeze:
  `ee61ac1f70816d7150ab9dc5af57a4b9f0714515`
- validated serialization-region evidence report blob:
  `f16d035b5f02f6e55138ca283dcdf971418eaa12`
- validated serialization-region implementation blob:
  `353663e225870d22241ca894e2774bed7a979e4d`

This document authorizes bounded implementation only after this design itself is
frozen. It does **not** authorize scientific CUDA execution.

## 1. Scientific motivation

The validated coarse serialization-region result established:

1. the PRIMARY_A-specific interference shift remains complement-side;
2. the complement deficit is not primarily EOS-mediated;
3. gaps `1-8` and `9-16` are dominated by within-CLAIM and within-EVIDENCE
   constructive-coherence loss;
4. in gaps `17-32`, `CLAIM->EVIDENCE` becomes the largest individual coarse
   region deficit;
5. the small `33-64` tail is localized almost entirely to
   `CLAIM->EVIDENCE`;
6. gaps `65+` are negligible.

The next question is therefore not another coarse region test.

The bounded follow-up question is:

> Which already-frozen XG1 generator anchors account for the short-range
> within-segment contraction and the mid-range CLAIM->EVIDENCE cross-region
> deficit?

This is a structural localization question, not a semantic-discovery exercise.

## 2. Frozen generator semantics

No new semantic labels may be introduced.

The only semantic anchor names authorized for this stage are the anchors already
defined by the frozen XG1 tokenizer-anchor machinery:

- `A_TITLE`
- `A_NAME`
- `A_ROLE`
- `A_PREDICATE`

`A_IDENTITY` is already frozen as the composite span from the start of
`A_TITLE` through the end of `A_NAME`, but it overlaps the atomic anchors and
therefore is not used as an independent partition cell. It may be reported only
as an exact post-hoc aggregate of `A_TITLE` and `A_NAME` cells.

Frozen dependency identities:

- `scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py`
  blob `6c98ce022ca134e385db28851fd364dc6daff423`;
- `scripts/build_reason_router_gen4_xg1_cross_generator_cohort.py`
  blob `c830026935a6c9f4990c6a3315c75fd5580e7264`;
- `scripts/prepare_reason_router_gen5_phase3_static.py`
  blob `5f2ec6498af475579b07e16b122be201a0ff656a`;
- Phase3 seven-cell view
  `data/reason_router_gen5_phase3_xg1_contention_training_v1/phase3_seven_cell_labeled_training.jsonl`
  blob `62c31fe06b5d83f76c1b75d4c8333aa6e31473cd`;
- Phase3 split manifest
  `data/reason_router_gen5_phase3_xg1_contention_training_v1/pair_split_manifest.json`
  blob `1f881fc46d7e4093d201eaf7cc9e98ca4294c424`.

The renderer identity already frozen by the XG1 generator is:

`During {time}, records from {location} identify {title} {name} as {role}; this person {predicate} {object}.`

The existing tokenizer-anchor implementation already derives
`A_TITLE`, `A_NAME`, `A_ROLE`, and `A_PREDICATE` character spans from generator
structure rather than by searching rendered text.

## 3. Mechanical residual class

To form an exhaustive token partition without inventing new semantic classes,
the implementation may use exactly one non-semantic mechanical class:

`RESIDUAL`

`RESIDUAL` means only:

> an active token that is not assigned to one of the four frozen atomic
> generator anchors.

It must not be interpreted as a semantic category.

It may contain renderer literals and source fields for which no pre-existing
frozen anchor is authorized in this stage, including time, location, object,
punctuation, connective text, and any explicit-denial template material outside
the four authorized anchors.

No scientific claim may be made about the internal composition of `RESIDUAL`.

## 4. Population

The population remains exactly the frozen Phase3A seed180 dev population:

- dev rows: `840`;
- valid tokens: `60,094`;
- frozen dev-order SHA256:
  `b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25`;
- frozen active dev-encoding SHA256:
  `e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51`.

No confirmatory seeds `9601/9900` may be loaded.

No VitaminC data may be loaded.

No new train/dev split may be formed.

## 5. Token-to-anchor mapping

The implementation must not inspect token strings or decoded token text.

The mapping must be derived from generator structure and tokenizer offsets.

### 5.1 Claim

Every claim in the frozen Phase3 seven-cell view is the deterministic XG1
statement render for its structured fact.

For each claim:

1. reconstruct the exact rendered claim from the frozen structured fact;
2. derive `A_TITLE`, `A_NAME`, `A_ROLE`, and `A_PREDICATE` character intervals
   using the frozen generator-declared span logic;
3. require byte-for-byte rendered-text identity with the frozen claim;
4. obtain whole-string tokenizer IDs and offsets using the exact active
   tokenizer with `add_special_tokens=False`;
5. require the tokenizer IDs, after the frozen `claim[:63]` truncation, to be
   exactly equal to the already-frozen Phase3 active claim encoding.

### 5.2 Evidence for C0-C5

For base cells `C0_SHAM` through `C5_TITLE_NAME`:

1. derive the frozen cell overrides from the XG1 cell specification;
2. reconstruct evidence through the frozen XG1 renderer;
3. derive the same four frozen anchor spans from generator structure;
4. require byte-for-byte identity with the frozen evidence;
5. obtain whole-string tokenizer IDs and offsets;
6. require the truncated IDs to equal the frozen active Phase3 evidence
   encoding exactly.

### 5.3 Evidence for C6_EXPLICIT_DENIAL

`C6_EXPLICIT_DENIAL` is generated by the already-frozen
`explicit_denial_evidence()` renderer in
`prepare_reason_router_gen5_phase3_static.py`.

The implementation may derive character intervals for the same four already
authorized source-field anchors from that frozen renderer:

- title -> `A_TITLE`
- name -> `A_NAME`
- role -> `A_ROLE`
- predicate -> `A_PREDICATE`

This is a structural materialization of already-authorized anchor semantics,
not a new semantic label.

The rendered evidence must match the frozen C6 evidence byte-for-byte and its
truncated tokenizer IDs must replay the frozen active Phase3 encoding exactly.

### 5.4 Token assignment rule

For each active non-EOS token:

- if its whole-string tokenizer offset overlaps exactly one authorized atomic
  anchor character interval, assign that token to that anchor;
- if it overlaps no authorized anchor interval, assign it to `RESIDUAL`;
- if it overlaps more than one authorized atomic anchor interval, execution
  must fail closed.

No substring search, regex search, token decoding, vocabulary-string
inspection, or semantic heuristics are allowed.

EOS remains mechanically separate and is excluded from the fine analysis.

PAD remains excluded by the frozen attention mask.

## 6. Exact partition and reconstruction

The fine token partition on each side is exactly:

`{A_TITLE, A_NAME, A_ROLE, A_PREDICATE, RESIDUAL}`.

The fine analysis is restricted to the three coarse region classes that were
scientifically implicated by the validated evidence:

1. `CLAIM->CLAIM`
2. `EVIDENCE->EVIDENCE`
3. `CLAIM->EVIDENCE`

For each coarse class, all ordered fine anchor-pair cells are evaluated.

Therefore each coarse class contains `5 x 5 = 25` fine cells, for 75 fine cells
total.

The fine cells must exactly reconstruct the already-validated coarse
serialization-region `H` vector for every gap, orientation, and component:

- fine CLAIM->CLAIM sum == coarse CLAIM->CLAIM;
- fine EVIDENCE->EVIDENCE sum == coarse EVIDENCE->EVIDENCE;
- fine CLAIM->EVIDENCE sum == coarse CLAIM->EVIDENCE.

Any reconstruction failure is a hard blocker.

`CLAIM->EOS` and `EOS->EVIDENCE` are not fine-localized in this stage because
the validated coarse evidence already showed that EOS is not the primary
localization locus.

## 7. Recurrence quantity

The scientific quantity is unchanged.

For component `c`, source token `tau`, target token `sigma`:

`I_c(tau,sigma) = 2 sum_(t >= sigma) <z_(t,tau)^c, z_(t,sigma)^c>`

and

`h_c(tau,sigma) = I_c(tau,sigma) / S_c`.

Fine anchor-pair diagnostics are additive sums of `h_c(tau,sigma)` over pairs
whose source/target token classes match the requested fine cell.

No new recurrence equation, normalization, projector, basis, or model
intervention is introduced.

## 8. Frozen windows

The distance windows remain exactly:

- `W1 = gap1_8`
- `W2 = gap9_16`
- `W3 = gap17_32`
- `W4 = gap33_64`
- `W5 = gap65_127`

Scientific emphasis follows the already-validated coarse result:

- W1/W2: short-range within-CLAIM and within-EVIDENCE;
- W3: transition regime where CLAIM->EVIDENCE became largest;
- W4: small cross-region tail control;
- W5: negligible long-range control.

No post-hoc window changes are permitted.

## 9. Required outputs

For every fine anchor-pair cell, component, and frozen window, report:

1. additive `H`;
2. exact pair-opportunity count `N`;
3. descriptive density `mean_h = H / N` when `N > 0`.

Source-matched PRIMARY_A minus CONTROL_R must be reported across all nine
source cells for:

- median `ΔH`;
- sign count;
- median descriptive `Δmean_h`.

The output must retain all 75 fine cells.

No top-k filtering may be used to define the scientific result.

Rankings may be displayed after all cells are computed, but no ranking cutoff
constitutes a mechanism threshold.

## 10. Primary scientific comparisons

The stage must answer the following bounded questions.

### Q1. Short-range CLAIM->CLAIM

Within W1 and W2, are the coarse CLAIM->CLAIM deficits concentrated in one or
more already-frozen anchors, or do they remain primarily `RESIDUAL`/distributed?

### Q2. Short-range EVIDENCE->EVIDENCE

Within W1 and W2, are the coarse EVIDENCE->EVIDENCE deficits concentrated in
one or more frozen anchors, or do they remain residual/distributed?

### Q3. Mid-range CLAIM->EVIDENCE

Within W3, which frozen claim-anchor -> evidence-anchor cells carry the
validated cross-region deficit?

### Q4. 33-64 CLAIM->EVIDENCE tail

Within W4, does the small validated tail localize to the same fine cells as W3,
or does it have a different structural profile?

These questions are descriptive localization questions.

They do not establish token-level causality.

## 11. Visible-side control

The visible component is mandatory.

The stage must report the same 75-cell decomposition for visible and
complement components.

This is especially important because the validated coarse evidence showed:

- visible CLAIM->CLAIM positive in `9/9`;
- visible EVIDENCE->EVIDENCE negative in `9/9`;
- aggregate visible `ΔlogQ` near zero due to cancellation.

A fine complement pattern must therefore be interpreted jointly with its
visible-side counterpart.

## 12. Falsification conditions

The fine-anchor localization hypothesis is not supported if any of the
following occurs:

1. the frozen active token encoding cannot be replayed exactly while obtaining
   offset mappings;
2. the fine cells fail exact reconstruction of their parent coarse region
   vectors;
3. most of the scientifically relevant deficit remains in `RESIDUAL` with no
   stable anchor-specific structure;
4. anchor-specific rankings vary substantially across the nine matched source
   cells without a coherent sign pattern;
5. the apparent complement localization is mirrored comparably in the visible
   control.

A null or residual-dominant outcome is an acceptable scientific result.

## 13. Prohibited analyses

This stage must not:

- inspect decoded token strings;
- search rendered text for anchor values;
- invent TITLE/NAME/ROLE/PREDICATE variants beyond the already-frozen anchors;
- introduce entity classes, lexical classes, syntactic classes, or manual token
  annotations;
- split `RESIDUAL` post hoc;
- add object/time/location anchors in this stage;
- fit a classifier or probe;
- refit projectors;
- modify model weights;
- train;
- optimize;
- call `.backward()`;
- load confirmatory seeds `9601/9900`;
- load VitaminC;
- use p-values;
- introduce post-hoc dominance thresholds;
- attribute percentages of nonlinear `L_interference` to fine cells;
- interpret additive `H` as Shapley attribution.

## 14. Computational design

The implementation should extend the already-validated
serialization-region reducer rather than add model forwards.

The intended execution contract is:

- same 840 dev rows;
- same two independent T4 row-sharded workers;
- same `BATCH_ROWS=32` unless a separately frozen pre-execution gate proves a
  change is necessary;
- same 243 source-gradient forwards;
- no duplicate scientific rows across workers;
- no downstream transport;
- no final-head transport;
- no training.

Fine anchor masks must be computed once per batch and reused across all source
cells and target groups.

The implementation should batch fine pair classes where memory-safe, but must
not alter the recurrence arithmetic or tolerance semantics.

## 15. Implementation gate

After this design is frozen, bounded implementation may create exactly:

- one new scientific audit script;
- one corresponding test file.

The implementation must authenticate:

- this design commit/blob;
- evidence freeze `ee61ac1f70816d7150ab9dc5af57a4b9f0714515`;
- validated serialization-region artifacts and SHA256 identities;
- frozen generator/tokenizer/static-preparation dependency blobs listed above;
- frozen Phase3 dev order and active encoding identities.

Static tests must cover at least:

1. claim anchor-span reconstruction;
2. C0-C5 evidence anchor-span reconstruction;
3. C6 explicit-denial anchor-span reconstruction;
4. whole-string tokenizer ID replay against frozen active encoding;
5. truncation at claim budget 63 and evidence budget 64;
6. no multi-anchor token assignment;
7. exhaustive five-class token partition;
8. 25 fine cells reconstruct each parent coarse region class exactly;
9. all three coarse parent classes reconstruct at every gap;
10. W1-W5 window sums;
11. pair-opportunity counts;
12. source-matched nine-cell aggregation;
13. visible-side control preservation;
14. source inspection proving no training/backward/downstream calls.

## 16. Execution boundary

Freezing this design does not authorize scientific execution.

A later execution step requires:

1. frozen implementation at a specific commit;
2. fresh Kaggle bootstrap at that commit;
3. exact runtime/snapshot/checkpoint/kernel provisioning;
4. fresh CUDA preflight and memory gate;
5. a separately frozen execution authority/gate.

No Kaggle scientific run is authorized by this document.

## 17. Expected interpretation boundary

If the stage yields stable anchor-localized deficits, the supported claim is
limited to structural generator-anchor localization within the frozen seed180
Phase3A dev population.

It does not establish:

- semantic universality;
- lexical causality;
- human-language entity causality;
- an intervention effect of individual anchors;
- a new architecture;
- a new training rule;
- generalization to confirmatory seeds.

If the result remains residual/distributed, the correct conclusion is that the
coarse serialization structure is more informative than the currently frozen
anchor vocabulary.

## 18. Stop condition

Stop after:

- this design is frozen;
- bounded implementation is completed and statically validated under the
  frozen design.

Do not start scientific execution until a separate execution gate is frozen.
