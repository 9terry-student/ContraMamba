# K0-RVG-P0 State-Blind Population / Token Contract Provisioning Specification Candidate

**Status:** state-blind provisioning specification candidate only.

**Parent preregistration commit:**

`cdad87acf664cd61e48406f9d4568b6ab206da24`

**Parent preregistration SHA256:**

`b2d4ec941c55b3a25ff3c30653784c73fbbc34fbaf31b2fb6493f09971a8c386`

**Validated raw recurrence observer implementation commit:**

`fcfe161c12f4ed8ef37aff435554cc0660e477af`

This specification authorizes no scientific model forward and no recurrent-state read.

After this specification is frozen, it may authorize one bounded state-blind implementation/provisioning phase that uses only:

- deterministic controlled-data generation;
- archived prior candidate pools;
- the frozen tokenizer;
- canonical serialization and hashing.

It may not load the Mamba model, checkpoint, task head, logits, or raw recurrence observer.

## 1. Objective

Provision and freeze the exact **scientific input contract** for K0-RVG-P before any recurrent state from the fresh population can be observed.

The provisioning phase must produce deterministic, hash-bound artifacts for:

1. generated source rows;
2. fresh candidate items;
3. phase-pair mapping;
4. matched/swapped token contracts;
5. prior-pool disjointness evidence;
6. state-blind provisioning manifest.

A successful provisioning run establishes only:

`INPUT_CONTRACT_READY_FOR_LATER_SCIENTIFIC_IMPLEMENTATION`

It is not scientific evidence.

## 2. Branch and scientific boundary

The branch state remains:

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

State-blind provisioning must not activate either branch.

Forbidden in K0-RVG-P0:

- model construction;
- checkpoint loading;
- model forward;
- logits;
- recurrent-state read;
- raw recurrence observer import/use;
- task-head inference;
- scientific endpoint computation;
- outcome inspection;
- causal intervention.

## 3. Exact implementation scope after P0 freeze

After this exact specification is frozen, implementation may create exactly two new files:

`scripts/longterm_k0_rvg_p0_state_blind_provisioning.py`

`tests/test_longterm_k0_rvg_p0_state_blind_provisioning.py`

No existing file may be modified.

Historical untracked K1 files remain untouched:

`scripts/longterm_k1_native_state_kinematics.py`

`tests/test_longterm_k1_native_state_kinematics.py`

The implementation must contain no model-forward path and no import-time provisioning execution.

## 4. Frozen generator dependency

Generator:

`scripts/build_controlled_v5.py`

Required SHA256:

`4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c`

Required Git blob:

`baee23a9f71333125f4a8735c2c92d20cab7eb4f`

The implementation must fail closed if either identity differs.

The generator's fixed seed-template count must be exactly:

`30`

The generated lexical phase period is frozen as:

`168`

## 5. Exact fresh template range

Materialize:

`fact_templates_for_count(1572)`

and take exactly:

`[1236:1572]`

Zero-based global template indices:

`1236..1571`

Expected item count:

`336`

Expected first pair ID:

`generated_fact_1237`

Expected last pair ID:

`generated_fact_1572`

No item outside this slice is eligible.

No failed item may be replaced.

## 6. Exact generated source construction

Run the frozen generator's ordinary deterministic record construction on exactly the 336 selected templates.

Expected source-row count:

`4368`

The state-blind implementation must serialize the rows as canonical UTF-8 JSONL:

- no BOM;
- LF only;
- one JSON object per line;
- `sort_keys=True`;
- compact separators;
- `ensure_ascii=False`;
- no NaN;
- final LF required.

Artifact name:

`generated_source.jsonl`

Its SHA256 is not guessed in this specification.

It must be computed by the state-blind provisioning implementation and frozen in the provisioning result.

## 7. Candidate extraction contract

For each pair ID, source rows must contain exactly:

- one `evidence_truncation`;
- one `entity_swap`;
- exactly one eligible REFUTE correction from:
  - `polarity_flip`, or
  - `none`.

The truncation row must satisfy:

- `final_label == NOT_ENTITLED`;
- `primary_failure_type == sufficiency`;
- `sufficiency_label == 0`.

The entity-swap control must satisfy:

- `final_label == NOT_ENTITLED`;
- `primary_failure_type == frame`;
- `polarity_label == NONE`.

The correction must satisfy:

- `final_label == REFUTE`;
- `polarity_label == REFUTE`.

All three rows must share the identical claim.

Any multiplicity or semantic-contract failure blocks provisioning.

## 8. Candidate item schema

For local template index `i = 0..335`, define:

`g = 1236 + i`

`phase = (g - 30) mod 168`

`cycle = i // 168`

and preserve the generator-local order.

Each candidate row must contain at least:

- `schema_version`;
- `generator_sha256`;
- `global_template_index`;
- `local_template_index`;
- `generator_phase_class`;
- `cycle_in_slice`;
- `pair_id`;
- `truncation_source_id`;
- `correction_source_id`;
- `correction_source_intervention`;
- `control_source_id`;
- `prefix_text`;
- `correction_text`;
- `control_text`;
- `base_claim_sha256`;
- `stable_item_id`.

The exact schema version is:

`k0-rvg-p0-candidate-v1`

Prefix text is exactly:

```text
Claim: <claim>
Evidence: <truncation evidence>
Additional evidence:
```

The stable item ID must be:

`k0-rvg-p0-v1:` + SHA256(canonical JSON of the candidate recipe before stable_item_id is inserted)

Candidate order is frozen as **ascending local template index**.

Hash-order sorting is forbidden.

Artifact name:

`candidate_pool.jsonl`

## 9. Phase signature audit

Use the same lexical-signature fields frozen by the K-series generated-population audit:

- title;
- name;
- alternate_title;
- alternate_name;
- role;
- alternate_role;
- predicate;
- alternate_predicate;
- time;
- alternate_time;
- location;
- alternate_location.

The 336-item slice must satisfy:

- exactly 168 phase classes;
- exactly two items per phase;
- local indices `p` and `p+168` share the same lexical signature for every `p=0..167`;
- the two pair IDs differ;
- the two claim texts differ in their generated object number and therefore are not duplicate claims.

Any phase-signature mismatch blocks provisioning.

## 10. Correction-source balance

Across 336 candidates, exact totals must be:

`polarity_flip = 168`

`none = 168`

For each phase block `p=0..167`, the two items:

`i = p`

`j = p + 168`

must contain exactly:

- one `polarity_flip` correction source;
- one `none` correction source.

Any deviation blocks provisioning.

## 11. Exact phase-pair mapping

There are exactly:

`168`

phase-paired blocks.

For block index `p`:

`item_a_local_index = p`

`item_b_local_index = p + 168`

The mapping artifact must contain, for every block:

- block index;
- phase class;
- item A local index;
- item B local index;
- item A pair ID;
- item B pair ID;
- item A stable ID;
- item B stable ID;
- item A correction-source intervention;
- item B correction-source intervention.

Artifact name:

`phase_pair_mapping.json`

The mapping is serialized as canonical compact JSON with no final whitespace ambiguity.

No reciprocal/hash/XOR remapping is allowed.

## 12. Prior-pool disjointness dependencies

Provisioning must authenticate and compare against these archived prior pools.

### K2W

Path:

`reports/longterm_k2w_fixed_window_phase_a_c7c7a0c218bb/candidate_pool.jsonl`

SHA256:

`abf693d3267cc4e3dd27a8127d2948b36fdf8ba24e135a643215f0f31a26d808`

### K2R / K3

Path:

`reports/longterm_k2r_claim_disjoint_replication_52bd363_v1/candidate_pool.jsonl`

SHA256:

`00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4`

### K3C

Path:

`reports/longterm_k3c_contribution_db75edfbf34b_v1/candidate_pool.jsonl`

SHA256:

`9603f6b20ba870807c151bb70df4c42b0957ded578729b133a37fee8aa1da83e`

### K3T

Path:

`reports/longterm_k3t_r_transportability_8c96b476da28_v1/candidate_pool.jsonl`

SHA256:

`d95d245e358ff497ea50b95e4f54d1192be64d09fec2c06538fc15f75b09ef70`

Any dependency SHA mismatch blocks provisioning.

## 13. Required prior-pool overlap checks

Against every prior pool separately, the fresh candidate set must have exact overlap count zero for:

- `pair_id`;
- exact claim text;
- canonical claim SHA256.

The implementation must report all 12 overlap counts:

`4 prior pools × 3 identity modes`

and every count must be zero.

A nonzero overlap blocks provisioning.

No candidate may be removed or replaced to repair overlap.

## 14. Frozen tokenizer identity

Tokenizer source:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers version:

`5.12.1`

Required tokenizer class:

`GPTNeoXTokenizer`

Required:

`is_fast == True`

Tokenization for contract construction must use:

`add_special_tokens=False`

The provisioning script may download/resolve tokenizer files.

It may not load model weights or instantiate the Mamba model.

## 15. Exact matched and swapped branches

For candidate `i`, let its phase mate be:

`mate(i) = i + 168` for `i < 168`

`mate(i) = i - 168` for `i >= 168`.

Define:

`M_corr(i) = prefix_i + correction_i`

`M_ctrl(i) = prefix_i + control_i`

`S_corr(i) = prefix_i + correction_mate(i)`

`S_ctrl(i) = prefix_i + control_mate(i)`

Every item therefore owns exactly two correction-control branch pairs:

- matched;
- phase-swapped.

The reciprocal construction is automatic because the mate owns the reverse donor relation.

## 16. Prefix identity audit

For each of the four full branch texts associated with item `i`:

- tokenize the full branch;
- separately tokenize `prefix_i`;
- require the full branch token IDs begin with the exact prefix token IDs.

This must hold for:

- `M_corr`;
- `M_ctrl`;
- `S_corr`;
- `S_ctrl`.

Any failure blocks provisioning.

The token contract must record:

- prefix token count;
- SHA256 of canonical prefix token-ID serialization.

## 17. Divergence-anchor definition

For each branch pair independently:

- matched: `M_corr` versus `M_ctrl`;
- swapped: `S_corr` versus `S_ctrl`;

let prefix token count be `L`.

Define divergence anchor:

`t_e = first k >= L such that corr_token[k] != ctrl_token[k]`

The divergence must exist within:

`[L, L+7]`

equivalently:

`t_e - L ∈ {0,...,7}`

If no divergence exists in that range, provisioning fails.

Matched and swapped `t_e` values are allowed to differ.

They must both be frozen in the token-contract artifact.

## 18. W=8 availability

For each branch pair and its own divergence anchor `t_e`, both correction and control branches must contain token positions:

`t_e .. t_e+7`

inclusive.

Additionally:

`t_e >= 1`

is required so incoming raw velocity:

`V_(t_e-1)`

will be defined in the later scientific execution.

Any matched or swapped failure blocks the entire population.

There is no item replacement.

## 19. Token-contract artifact

Artifact name:

`token_contracts.jsonl`

One row per candidate, in ascending local-template order.

Each row must contain at least:

- stable item ID;
- pair ID;
- local template index;
- phase block index;
- prefix token count;
- prefix token SHA256;
- matched correction token count;
- matched control token count;
- matched divergence anchor;
- matched divergence offset from prefix;
- matched W=8 availability;
- swapped correction token count;
- swapped control token count;
- swapped divergence anchor;
- swapped divergence offset from prefix;
- swapped W=8 availability;
- phase-mate stable ID;
- phase-mate pair ID.

Raw recurrent states, logits, labels predicted by the model, or scientific endpoint values are forbidden in this artifact.

## 20. State-blind artifact set

A successful provisioning run must create exactly these five scientific-input artifacts in a fresh output directory:

`generated_source.jsonl`

`candidate_pool.jsonl`

`phase_pair_mapping.json`

`token_contracts.jsonl`

`provisioning_manifest.json`

The output directory name must be descriptive and must include the eventual provisioning commit short SHA when execution occurs.

No scientific result file is allowed.

## 21. Provisioning manifest

`provisioning_manifest.json` must contain at least:

- schema version;
- runtime Git HEAD;
- P0 authority commit;
- parent preregistration commit;
- parent preregistration SHA256;
- generator path/SHA/blob;
- tokenizer model/revision/class/version;
- fresh template range;
- first/last pair ID;
- item count;
- phase block count;
- source-row count;
- correction-source counts;
- all prior-overlap counts;
- matched divergence-offset histogram;
- swapped divergence-offset histogram;
- minimum branch post-divergence availability;
- SHA256 for the other four artifacts;
- explicit flags:
  - `model_loaded = false`;
  - `checkpoint_loaded = false`;
  - `model_forward_executed = false`;
  - `logits_read = false`;
  - `recurrent_state_read = false`;
  - `observer_imported = false`.

Manifest serialization is canonical JSON plus final LF.

## 22. Required state-blind implementation tests

The bounded implementation test suite must cover at least:

1. authority / exact two-file scope constants;
2. generator identity binding;
3. exact fresh range and pair-ID boundaries;
4. expected source-row count 4368;
5. exact 168×2 phase structure;
6. exact correction-source balance;
7. exact phase-pair mapping;
8. malformed/missing source-row multiplicity fail-closed;
9. prior-pool SHA fail-closed;
10. nonzero overlap fail-closed;
11. tokenizer class/revision binding;
12. prefix identity;
13. divergence-anchor definition;
14. W=8 availability;
15. `t_e >= 1`;
16. canonical artifact serialization;
17. deterministic repeated provisioning byte identity;
18. absence of model/checkpoint/observer imports;
19. CLI absence of scientific execution options.

Tests may use tiny fabricated fixtures where practical.

Tokenizer-dependent full provisioning validation may use the frozen real tokenizer.

## 23. Required state-blind provisioning validation

After implementation, one full state-blind provisioning run must prove:

- 336 candidates;
- 4368 generated source rows;
- 168 phase blocks;
- exact 168/168 correction-source balance;
- 12/12 overlap counts equal zero;
- all 336 matched token contracts valid;
- all 336 swapped token contracts valid;
- no item replacement;
- deterministic byte-identical second materialization;
- no model/checkpoint/observer access.

The run must print and/or persist all artifact SHA256 values.

If any condition fails, scientific execution remains unauthorized.

## 24. Repository/provenance contract

Expected branch:

`longterm-k-series-native-state-kinematics`

The frozen P0 authority commit must be an ancestor of any later state-blind provisioning implementation/runtime HEAD.

During implementation validation, allowed repository dirt is limited to:

- the two authorized new P0 implementation files;
- the historical K1 untracked pair.

No other tracked/untracked change is allowed.

## 25. Scientific non-access enforcement

The P0 implementation must not import:

`scripts.longterm_k0_rvg_raw_recurrence_observer`

and must not import task-model construction functions from K2S/K3T scientific runners.

The P0 implementation may import only what is necessary for:

- frozen generator access;
- tokenizer access;
- JSON/hash/provenance utilities.

The source and tests must statically assert absence of calls/imports associated with:

- checkpoint loading;
- A0 model construction;
- model forward;
- logits;
- trace collectors;
- recurrent-state observer;
- endpoint computation.

## 26. Output interpretation

P0 PASS means:

`STATE_BLIND_INPUT_CONTRACT_VALID = YES`

It does not mean:

- raw vector organization exists;
- turning is positive;
- response coherence is positive;
- Branch A is supported;
- Branch B is supported;
- any scientific hypothesis was tested.

## 27. Next-stage boundary

After the two-file P0 implementation is frozen and a full state-blind provisioning run is independently validated and archived, the next allowed design stage is:

`K0-RVG-P1 — Scientific Raw-Vector Execution Implementation Specification`

P1 must bind the exact P0 artifact hashes.

P1 may define how the already-validated raw recurrence observer will consume those exact token contracts and compute the preregistered endpoints.

P1 still must not itself authorize scientific execution unless a later separate execution-authority artifact explicitly does so.

## 28. Authority markers

Before this specification is frozen:

`K0_RVG_P0_SPEC_FROZEN = NO`

After this exact specification is frozen as the immediate one-file child of:

`cdad87acf664cd61e48406f9d4568b6ab206da24`

the bounded state-blind implementation/provisioning authority becomes active:

`K0_RVG_P0_SPEC_FROZEN = YES`

`P0_IMPLEMENTATION_AUTHORIZED = YES`

`P0_IMPLEMENTATION_SCOPE = TWO_NEW_FILES_ONLY`

`P0_STATE_BLIND_TOKENIZER_EXECUTION_AUTHORIZED = YES`

`P0_STATE_BLIND_ARTIFACT_GENERATION_AUTHORIZED = YES`

`MODEL_LOADING_AUTHORIZED = NO`

`CHECKPOINT_LOADING_AUTHORIZED = NO`

`MODEL_FORWARD_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`RECURRENT_STATE_READ_AUTHORIZED = NO`

`RAW_RECURRENCE_OBSERVER_USE_AUTHORIZED = NO`

`SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

`SCIENTIFIC_EXECUTION_AUTHORIZED = NO`

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

`K4_EXECUTION_AUTHORIZED = NO`

Implementation must stop after the exact two-file P0 delta plus state-blind provisioning validation.
