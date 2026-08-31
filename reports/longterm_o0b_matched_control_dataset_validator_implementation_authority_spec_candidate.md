# O0b Matched-Control Dataset and Tokenizer Validator Implementation Authority

Status: STATIC IMPLEMENTATION-AUTHORITY SPECIFICATION CANDIDATE ONLY

This document is a candidate authority for a later bounded implementation phase.
It creates no dataset, validator, tests, tokenizer outputs, model outputs,
training results, evaluation results, Kaggle artifacts, commits, or scientific
claims. Its existence does not authorize execution in the current phase.

## 1. Authority and phase boundary

Authority order for this candidate:

1. Current controller instruction for this task card.
2. Frozen O0b scientific-design authority commit:
   `df461469cb087f7f5db1e41a2b08e65ea517ad8a`.
3. This candidate implementation-authority specification:
   `reports/longterm_o0b_matched_control_dataset_validator_implementation_authority_spec_candidate.md`.
4. Canonical O0a evidence commit:
   `f7241abea9a09b54ff3b8ee66cacbd7f4feebb14`.
5. `reports/longterm_o0b_token_aligned_native_hidden_state_precursor_disambiguation_authority_spec_candidate.md`.
6. `docs/CONTRAMAMBA_RESEARCH_HYPOTHESIS_MAP.md`.
7. `docs/CONTRAMAMBA_RESEARCH_VISION.md`.

URP/reason-router authority, artifacts, attempts, files, and conclusions are
unrelated to this O0b authority. They must not be consumed, modified, or
reinterpreted by the future O0b dataset/validator implementation.

Current phase: STATIC IMPLEMENTATION-AUTHORITY SPECIFICATION ONLY.

This candidate authorizes no current implementation or execution. A later
explicit implementation task may use this document as authority only if that
task preserves all boundaries below.

## 2. Scientific purpose

The future implementation must produce a human-auditable matched-control
dataset and a fail-closed tokenizer validator sufficient to establish that O0b
primary comparisons satisfy the frozen design before any Mamba model loading,
hidden-state observation, observer implementation, training, evaluation, or
scientific interpretation.

The dataset and validator exist to freeze matched O0b primary comparisons under
identical claim text, identical serialization scaffold, equal full serialized
token count, matched terminal position, and rigorously derived
first-divergent-token anchors.

## 3. Frozen constants

The future implementation must preserve:

```text
O0b scientific-design authority commit = df461469cb087f7f5db1e41a2b08e65ea517ad8a
tokenizer ID = state-spaces/mamba-130m-hf
immutable tokenizer revision = 5708daa364c50b880e7bd92eab456e0d34492ee9
tokenization = add_special_tokens=False
tokenizer trust_remote_code=False
```

The future validator may load only the tokenizer
`state-spaces/mamba-130m-hf` at immutable revision
`5708daa364c50b880e7bd92eab456e0d34492ee9`, with
`add_special_tokens=False` and `trust_remote_code=False`.
The validator must bind and validate tokenizer identity and tokenizer revision
as separate fail-closed fields:

```text
tokenizer_id == "state-spaces/mamba-130m-hf"
tokenizer_revision == "5708daa364c50b880e7bd92eab456e0d34492ee9"
```

Any tokenizer ID other than `state-spaces/mamba-130m-hf` must fail closed even
if the revision field matches. Any tokenizer revision other than
`5708daa364c50b880e7bd92eab456e0d34492ee9` must fail closed even if the
tokenizer ID matches.

The validator must prohibit repository-supplied arbitrary Python execution,
custom remote tokenizer code requiring `trust_remote_code=True`, any fallback
that enables remote code, model classes, and model weights. If tokenizer loading
at the frozen revision cannot succeed with `trust_remote_code=False`, future
implementation validation must fail closed and stop. The future implementation
must not instantiate `AutoModel`, `MambaModel`, `AutoModelForCausalLM`, or any
equivalent model class, and must not exercise any model-weight download or
request path.

The default O0b serialization is now frozen for dataset validation unless a new
authority is created before implementation:

```text
Claim: <claim>
Evidence: <evidence>
```

The exact serialized runtime string must be:

```text
Claim: <claim>\nEvidence: <evidence>
```

## 4. Future implementation file scope

Chosen future implementation file set: exactly four files.

The later bounded implementation phase may create or modify only:

1. `data/longterm_o0b_matched_controls_v1.jsonl`
2. `scripts/validate_longterm_o0b_matched_controls.py`
3. `tests/test_validate_longterm_o0b_matched_controls.py`
4. `reports/longterm_o0b_matched_controls_v1_validation.json`

No other tracked file may change unless a later authority explicitly supersedes
this one. The validation JSON is authorized as a generated, tracked validation
artifact so tokenizer-derived metadata can be frozen auditable evidence for a
subsequent execution authority without making derived values authoritative
source data in the human-authored JSONL.

The future implementation must not touch, stage, delete, rename, move, or
modify protected unrelated state, including:

- all root historical `.patch` files;
- `reports/stage180a_pass2_annotations_completed.csv`;
- `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py`;
- `tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py`;
- any URP/reason-router file;
- canonical O0a artifacts;
- the frozen O0b scientific-design authority.

## 5. Dataset file and schema

The future dataset must be:

```text
data/longterm_o0b_matched_controls_v1.jsonl
```

It must use one JSON object per matched set, not one row per condition. This
keeps claim identity shared once by construction and makes the human semantic
contract easier to audit at matched-set granularity.

For v1, the dataset must contain exactly three independent matched sets with
exactly these pair IDs, one record each:

```text
o0b_pair_001
o0b_pair_002
o0b_pair_003
```

O0a may inform domain diversity, but O0b examples must be newly authored for
the O0b matched-control design and must not blindly reuse O0a rows.

Each JSONL object must contain at minimum:

- `schema_version`
- `pair_id`
- `claim`
- `reference_sufficient`
- `paraphrase_sufficient`
- `insufficient_matched`
- `surface_null_matched`
- `insufficiency_rationale`
- `paraphrase_rationale`
- `surface_null_rationale`

Each evidence member must be a string directly. The frozen v1 schema therefore
stores `reference_sufficient`, `paraphrase_sufficient`,
`insufficient_matched`, and `surface_null_matched` as direct evidence strings,
not nested objects.

The authored JSONL is the source of human semantic content only. It must not
manually store tokenizer-derived fields as authoritative source data when those
fields can be derived deterministically by the validator.

Human-authored semantic dataset fields:

- `schema_version`
- `pair_id`
- `claim`
- four evidence strings
- three rationale strings

Tokenizer-derived validation metadata:

- full serialized token IDs;
- evidence token IDs if reported;
- evidence token counts;
- full serialized token counts;
- evidence start index;
- first divergent token index;
- terminal index;
- frozen anchor indices;
- pairwise validation statuses;
- dataset runtime-byte SHA256.

## 6. Human semantic contract

For each matched set:

`reference_sufficient` must directly and adequately support the exact claim.

`paraphrase_sufficient` must:

- keep the claim byte-identical by construction;
- preserve evidence meaning and conclusion-critical support;
- remain sufficient for the claim;
- be an evidence-only semantic-preserving paraphrase.

`insufficient_matched` must:

- keep the same claim;
- preserve frame, entity, and predicate as much as possible;
- specifically remove or replace conclusion-critical support;
- remain non-refuting unless separately marked by later authority as a
  secondary control;
- avoid empty evidence, obvious truncation, dangling syntax, deletion markers,
  repeated filler, formatting shortcuts, and syntactic degradation.

`surface_null_matched` must:

- keep the same claim;
- remain sufficient;
- introduce ordinary lexical or surface token identity change;
- avoid substantive frame, predicate, polarity, or sufficiency changes;
- avoid identity duplicates and useless near-zero wording variants.

The future dataset must avoid systematic condition-specific lexical shortcuts,
including words appearing only in all insufficient cases, repeated placeholders
or fillers, condition-specific punctuation, systematic negation only in one
condition, obvious syntactic degradation, or unique formatting patterns.

Validator/static tests must check surface/form shortcuts where mechanically
possible. Semantic sufficiency remains a required human-audited property and
cannot be fully automated.

## 7. Validator responsibilities

The future validator must:

1. Load exactly `data/longterm_o0b_matched_controls_v1.jsonl`.
2. Fail closed on malformed JSON, malformed JSONL, schema errors, wrong types,
   unexpected condition structures, missing fields, or empty required strings.
3. Require exactly the frozen pair IDs `o0b_pair_001`, `o0b_pair_002`, and
   `o0b_pair_003`, with exactly one record each.
4. Verify `pair_id` uniqueness.
5. Verify all required human-authored strings are non-empty. Normalization may
   be used only for validation checks; source bytes must not be mutated.
6. Build exact serialized strings as
   `Claim: <claim>\nEvidence: <evidence>`.
7. Verify claim bytes are shared once per matched-set object and therefore
   identical across all four members by construction.
8. Load only the tokenizer
   `state-spaces/mamba-130m-hf` at revision
   `5708daa364c50b880e7bd92eab456e0d34492ee9`.
9. Use `add_special_tokens=False` and enforce `trust_remote_code=False`.
10. Prohibit model class loading and model weight loading. The validator must
    not instantiate `MambaModel`, `AutoModel`, `AutoModelForCausalLM`, or any
    other model class.
11. Derive and record exact token IDs for every full serialized member.
12. Derive evidence-region tokenization/index metadata from the actual full
    serialized token sequence using a fail-closed boundary-safe method.
13. Verify equal full serialized token count across all four primary members:
    `reference_sufficient == paraphrase_sufficient == insufficient_matched ==
    surface_null_matched`.
14. Report evidence-region token counts separately when rigorously derivable;
    equal full serialized token count is the hard primary condition.
15. Compute pairwise first divergent token indices relative to
    `reference_sufficient` for `paraphrase_sufficient`,
    `insufficient_matched`, and `surface_null_matched`.
16. Require each first divergence to exist, fall inside the evidence region,
    avoid claim/scaffold tokens, and occur before the terminal index.
17. Compute frozen anchor indices:
    `divergence - 1`, `divergence`, `divergence + 1`,
    `divergence + 2`, `divergence + 4`, and `terminal`.
18. Represent out-of-range post-divergence anchors as JSON `null`. Alternate
    offsets must not be substituted.
19. Record terminal index for every condition.
20. Emit deterministic validation metadata to the tracked validation artifact.

The validator must fail closed on malformed or inconsistent anchor metadata.
For each reference-relative comparison, the only authorized anchor schedule is:

```text
divergence - 1
divergence
divergence + 1
divergence + 2
divergence + 4
terminal
```

The required anchor keys and values are exactly:

- `anchor_pre_minus_1` must equal `first_divergent_token_index - 1`;
- `anchor_divergence` must equal `first_divergent_token_index`;
- `anchor_post_plus_1` must equal `first_divergent_token_index + 1` when in
  range, otherwise JSON `null`;
- `anchor_post_plus_2` must equal `first_divergent_token_index + 2` when in
  range, otherwise JSON `null`;
- `anchor_post_plus_4` must equal `first_divergent_token_index + 4` when in
  range, otherwise JSON `null`;
- `anchor_terminal` must equal the common final token index for the equal-length
  matched set.

No extra anchor name is allowed. No missing required anchor key is allowed. No
substituted offset is allowed. No negative usable anchor index is allowed. No
non-null anchor may exceed terminal index. A JSON `null` anchor is valid only
when the corresponding post-divergence offset is genuinely out of range.

## 8. Token-boundary strategy

The future validator must not assume:

```text
tokenize(serialization_prefix) + tokenize(evidence)
==
tokenize(full_serialized_string)
```

The validator must derive evidence start and divergence indices from actual
full serialized token IDs and a rigorously verified prefix method. The required
fail-closed strategy is:

1. Construct the exact common serialized prefix ending immediately before the
   evidence text:
   `Claim: <claim>\nEvidence: `.
2. Tokenize the exact prefix with `add_special_tokens=False`.
3. Tokenize the full serialized string with `add_special_tokens=False`.
4. Verify that each full token sequence shares the expected prefix tokens
   exactly as token IDs.
5. Explicitly test tokenizer boundary-merging behavior with a fake or
   synthetic tokenizer fixture where prefix-plus-evidence tokenization would
   differ from naive concatenation.
6. Fail closed if prefix sharing cannot be proven for any member, or if an
   offset-based method is substituted without tests proving it agrees with the
   full-sequence prefix method.

`evidence_start_index` is an index in the actual token ID sequence produced by
tokenizing the full serialized string. For the required prefix-verification
strategy:

```text
prefix_text = "Claim: <claim>\nEvidence: "
prefix_token_ids = tokenizer(prefix_text, add_special_tokens=False)
full_token_ids = tokenizer(full_serialized_text, add_special_tokens=False)
```

Only if:

```text
full_token_ids[:len(prefix_token_ids)] == prefix_token_ids
```

may the validator define:

```text
evidence_start_index = len(prefix_token_ids)
```

This value must not be inferred before prefix-sharing proof succeeds.

If full-string tokenization produces a tokenization boundary such that
separately tokenized `prefix_token_ids` are not an exact token-ID prefix of
`full_token_ids`, the v1 validator must fail closed. It must not silently shift
`evidence_start_index`, approximate the boundary, split a token manually, infer
the index from token counts, or fall back to naive concatenated tokenization.
Offset-based recovery is not part of v1. If future work wants tokenizer offsets
to support boundary-crossing cases, that requires separate authority defining
and testing exact offset semantics before use.

## 9. Validation metadata artifact

The future validator must write:

```text
reports/longterm_o0b_matched_controls_v1_validation.json
```

The artifact must be deterministic and must contain no wall-clock timestamp.
Runtime/tool versions should be recorded later in execution provenance rather
than included here if they would make the artifact nondeterministic.

At minimum, the validation JSON must contain:

- `schema_version`
- `scientific_design_authority_commit`
- `implementation_authority_commit`
- `repository_head`
- dataset path
- dataset SHA256 over exact runtime/LF bytes used for validation
- tokenizer ID
- tokenizer revision
- `add_special_tokens`
- serialization template
- pair IDs
- per-pair, per-condition full serialized token IDs
- per-pair, per-condition full serialized token count
- per-pair, per-condition evidence-region start index
- per-pair, per-condition evidence-region token count if rigorously derivable
- first divergence relative to reference for each non-reference condition
- terminal index for every condition
- anchor indices for each comparison
- equal-length validation status
- first-divergence-in-evidence status
- overall `PASS` or `FAIL`

The validation artifact must not contain an ambiguous `authority_commit` field.
It must distinguish all authority/provenance identities:

- `scientific_design_authority_commit` is the frozen O0b scientific-design
  authority commit, exactly
  `df461469cb087f7f5db1e41a2b08e65ea517ad8a`.
- `implementation_authority_commit` is the immutable commit SHA that freezes
  this implementation-authority candidate. It must be supplied explicitly to
  the future validator or otherwise independently verified, and must not be
  inferred ambiguously from current HEAD.
- `repository_head` is the exact Git repository HEAD at which the four-file
  dataset/validator implementation and canonical validation artifact are
  produced and validated.

The validator and artifact must fail closed if either
`implementation_authority_commit` or `repository_head` is absent, malformed, or
mismatched where required. The future implementation must not conflate the
scientific-design authority commit, implementation-authority commit, or
implementation repository HEAD.

The canonical validation artifact must represent unavailable anchors as JSON
`null` and must use exactly one deterministic byte representation:

```python
json.dumps(
    payload,
    ensure_ascii=False,
    sort_keys=True,
    indent=2
) + "\n"
```

The resulting string must be encoded as UTF-8 without BOM and must use LF
newlines only. JSON object keys must be recursively emitted with deterministic
sorted ordering. Explicit pair lists or pair-entry arrays must be constructed
in this frozen semantic order even though JSON object keys are sorted lexically:

```text
o0b_pair_001
o0b_pair_002
o0b_pair_003
```

Explicit condition lists or condition-entry arrays must be constructed in this
frozen semantic order even though JSON object keys are sorted lexically:

```text
reference_sufficient
paraphrase_sufficient
insufficient_matched
surface_null_matched
```

Arrays whose semantic order is already defined must preserve that frozen order.
The artifact must contain no timestamps, random UUIDs, machine-specific
absolute paths, tokenizer cache paths, temp paths, hostname or user identity, or
environment-dependent fields not explicitly authority-bound.

## 10. Dataset hash discipline

The validator must report the dataset SHA256 calculated from the exact runtime
bytes used for validation. The future implementation should write the JSONL
with LF line endings. The validator must read bytes directly from
`data/longterm_o0b_matched_controls_v1.jsonl` and hash those bytes before JSON
parsing or newline normalization. The reported hash therefore identifies the
actual validated dataset bytes.

## 11. Tests and tokenizer access

Future tests must not require Mamba model loading.

Test structure:

- Pure unit tests must use synthetic/fake tokenizer fixtures for schema,
  duplicate/missing IDs, empty fields, unequal lengths, missing divergence,
  divergence in claim/scaffold, boundary behavior, out-of-range anchors,
  malformed condition structure, deterministic output, hash recording, and
  validation-artifact consistency.
- One explicitly bounded real-tokenizer contract test is authorized if needed
  to prove exact revision enforcement and `add_special_tokens=False` behavior.

Tokenizer-only download/cache access is allowed during the later implementation
validation only for:

```text
state-spaces/mamba-130m-hf
revision 5708daa364c50b880e7bd92eab456e0d34492ee9
```

No model weight download, model class instantiation, model forward pass,
hidden-state observation, training, evaluation, or Kaggle execution is
authorized. If the tokenizer is unavailable locally, the later implementation
validation may retrieve tokenizer files from Hugging Face for the exact
immutable revision. This is tokenizer validation only, not model execution or
scientific hidden-state execution.

The tokenizer loading call must pass and enforce `trust_remote_code=False`.
Future tests must prove that `trust_remote_code=False` is passed, configurations
or attempts requiring remote code are rejected fail-closed, no fallback enables
remote code, no model class is instantiated, and no model-weight download or
request path is exercised.

The validator must fail closed for all of the following:

- malformed JSON or JSONL;
- missing pair ID;
- duplicate pair ID;
- unexpected extra pair ID;
- missing, malformed, wrongly typed, or empty required fields;
- malformed record or condition structure;
- unequal full serialized token count;
- missing divergence;
- divergence in claim or scaffold;
- invalid terminal/divergence relation;
- unsafe or missing `trust_remote_code=False`;
- tokenizer configuration requiring remote code;
- tokenizer ID other than `state-spaces/mamba-130m-hf`;
- tokenizer revision other than
  `5708daa364c50b880e7bd92eab456e0d34492ee9`;
- `add_special_tokens` other than `False`;
- attempted model class loading;
- attempted model weight request or download;
- malformed or inconsistent anchor metadata;
- mismatch between the current dataset runtime SHA256 and the dataset SHA256
  recorded in an existing validation artifact when artifact consistency is
  checked;
- nondeterministic canonical validation artifact bytes;
- canonical JSON serialization, newline, or encoding mismatch;
- validation artifact provenance identity mismatch;
- malformed or missing `implementation_authority_commit`;
- malformed or missing `repository_head`.

Required test coverage:

Schema/data tests must explicitly cover:

- missing pair ID;
- duplicate pair ID;
- unexpected extra pair ID;
- empty fields;
- malformed record;
- malformed condition structure;
- unequal full serialized token count;
- missing divergence;
- divergence in scaffold;
- invalid terminal/divergence relation;
- wrong anchor offset;
- missing anchor;
- unexpected extra anchor;
- non-null out-of-range anchor;
- incorrect JSON `null` for an in-range anchor;
- out-of-range `+4` anchor represented as JSON `null`;
- wrong terminal anchor.

Tokenizer-boundary/security tests must explicitly cover:

- token merge across scaffold/evidence boundary;
- exact fail-closed behavior on prefix-sharing failure;
- `trust_remote_code=False` enforcement;
- remote-code-required tokenizer configuration rejected;
- wrong tokenizer ID rejected fail-closed;
- immutable revision enforcement;
- correct tokenizer ID with wrong revision rejected fail-closed;
- exact tokenizer ID plus exact revision allowed to proceed to remaining
  validation;
- `add_special_tokens=False` enforcement;
- no model class instantiation;
- no model weights requested.

Provenance/artifact tests must explicitly cover:

- exact runtime dataset SHA256 recorded;
- existing artifact dataset-hash mismatch rejected;
- `implementation_authority_commit` recorded and validated;
- `repository_head` recorded and validated;
- deterministic generation twice from identical inputs produces exact
  byte-for-byte equality and identical SHA256;
- canonical formatting, newline, and UTF-8-without-BOM encoding verified;
- validation artifact consistency with source dataset and derived tokenizer
  metadata.

## 12. Semantic review gate

Automated validation cannot establish semantic sufficiency. After future
dataset authoring and tokenizer matching, but before implementation freeze or
execution authority, a human static review is required.

The future implementation report must print all three matched sets in compact
human-readable form. For each matched set, the report must display:

- `pair_id`;
- `claim`;
- `reference_sufficient`;
- `paraphrase_sufficient`;
- `insufficient_matched`;
- `surface_null_matched`;
- `insufficiency_rationale`;
- `paraphrase_rationale`;
- `surface_null_rationale`;
- full serialized token count for each of the four conditions;
- first divergence position for `paraphrase_sufficient` versus
  `reference_sufficient`;
- first divergence position for `insufficient_matched` versus
  `reference_sufficient`;
- first divergence position for `surface_null_matched` versus
  `reference_sufficient`.

If already available in the validation metadata, the report should also display
`evidence_start_index` and terminal index. These display fields do not add a
scientific requirement beyond the frozen contract.

The report must explicitly attest for each set:

- claim identical by construction;
- reference sufficient;
- paraphrase sufficient;
- insufficient member is non-refuting and lacks conclusion-critical support;
- surface-null member is sufficient and semantic-preserving;
- no obvious condition-specific shortcut.

Tokenizer validation `PASS` and validator `PASS` cannot substitute for this
human semantic review. Without this human semantic review, tokenizer validation
`PASS` is not sufficient to freeze the dataset/validator implementation.

## 13. Scientific execution boundary

Even after future dataset/validator implementation `PASS`, this authority does
not authorize:

- Mamba model loading;
- hidden-state forward pass;
- observer implementation;
- scientific execution;
- scientific interpretation;
- Kaggle;
- training;
- evaluation.

The next transition after implementation freeze should be a separate O0b
observer implementation authority, not scientific execution directly.

## 14. Future implementation provenance report

The future implementation report must provide:

- repository HEAD;
- exact changed paths;
- dataset LF/runtime SHA256;
- validator SHA256;
- test SHA256;
- validation artifact SHA256;
- tokenizer ID and revision;
- exact validation command or commands;
- validation `PASS` output;
- tests `PASS` output;
- `git diff --check`;
- `git status --short`;
- explicit no-model/no-training/no-evaluation statement.

The future report must distinguish commands not run from commands that passed.
It must not report `PASS` for any command that did not actually run.

## 15. Static self-review checklist

This candidate explicitly checks and freezes:

- exact future file scope: four files only;
- semantic dataset contract: present, human-audited, and separated from
  tokenizer-derived metadata;
- equal full serialized token count: hard primary validator condition;
- tokenizer-boundary handling: full-sequence prefix method with fail-closed
  boundary tests;
- pairwise first-divergence calculation: reference-relative for the three
  non-reference conditions;
- anchor schedule: `-1`, `0`, `+1`, `+2`, `+4`, and terminal, with
  out-of-range post-divergence anchors as `null`;
- deterministic validation metadata: tracked JSON, stable ordering, no
  timestamp;
- tokenizer-only/no-model boundary: exact tokenizer revision allowed later,
  model classes and weights forbidden;
- semantic human-review requirement: required before freeze/execution;
- protected unrelated state: explicitly out of scope;
- no scientific execution authority: observer, model forward, training,
  evaluation, Kaggle, and interpretation remain forbidden.

## 16. Explicit non-authorizations

This candidate does not authorize current or future work outside the bounded
scope above. It specifically does not authorize:

- changing any existing file during this current authority-spec task;
- creating the dataset now;
- creating validator code now;
- creating tests now;
- tokenizer execution now;
- model loading now or in the later validator phase;
- model forward pass;
- hidden-state observation;
- observer implementation;
- training;
- evaluation;
- Kaggle;
- commit;
- push.

Final status of this candidate, if it is the only task-introduced delta after
validation:

```text
PASS_READY_FOR_INDEPENDENT_IMPLEMENTATION_AUTHORITY_VERIFICATION
```
