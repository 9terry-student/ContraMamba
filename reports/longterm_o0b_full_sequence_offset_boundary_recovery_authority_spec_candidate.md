# Long-term O0b Full-Sequence Offset Boundary Recovery Authority Specification Candidate

Status: authority candidate only; not independently verified, not frozen, and not an O0b result.

## 1. Authority and purpose

This candidate is governed by the current controller instruction, the frozen O0b scientific-design authority at commit `df461469cb087f7f5db1e41a2b08e65ea517ad8a`, the frozen O0b dataset/validator implementation authority at commit `31e6d7882586e312f783cb2fd69718eb1ee7e452`, and the independent finding `AUTHORITY_LEVEL_BOUNDARY_INCOMPATIBILITY`.

It repairs token-coordinate identifiability at the scaffold/evidence boundary only. It does not establish an O0b precursor, hidden-state separation, unsupported-commitment prediction, semantic sufficiency sensitivity, generalization, significance, or any other scientific result.

## 2. Established defect and diagnostic status

The frozen boundary rule

```text
full_token_ids[:len(prefix_token_ids)] == prefix_token_ids
```

uses `prefix_token_ids` obtained by separately tokenizing `"Claim: <claim>\nEvidence: "`. Under the exact frozen tokenizer, this rule cannot be satisfied by ordinary natural evidence starts. A systematic artificial leading LF can make it pass, but that is prohibited formatting/tokenization padding.

Therefore the old exact-prefix-sharing rule is not usable for natural O0b evidence; the leading-LF implementation is not freezeable; and the existing four-file implementation is diagnostic/provenance only. It is not scientific evidence and is not an O0b result. The prohibition on leading/trailing whitespace and formatting padding is unchanged.

## 3. Preserved constants and boundaries

The following remain exactly in force:

- tokenizer/model ID: `state-spaces/mamba-130m-hf`;
- revision: `5708daa364c50b880e7bd92eab456e0d34492ee9`;
- `add_special_tokens=False`;
- `trust_remote_code=False`;
- serialization: `Claim: <claim>\nEvidence: <evidence>`;
- exactly three matched sets and exactly four conditions per set;
- equal FULL serialized token count within each matched set;
- pairwise first divergence relative to the reference;
- pair-relative anchor schedule;
- human semantic review;
- no model classes or weights, no hidden-state forward, no training, no evaluation, and no Kaggle execution.

All unrelated scientific, dataset, tokenizer-security, equal-length, semantic-control, anchor, provenance, and no-model boundaries remain in force.

## 4. Normative full-sequence offset boundary method

The separately-tokenized-prefix rule is replaced by one uniform full-sequence offset-mapping method. The future validator MUST load the exact tokenizer as a FAST tokenizer:

```python
AutoTokenizer.from_pretrained(
    "state-spaces/mamba-130m-hf",
    revision="5708daa364c50b880e7bd92eab456e0d34492ee9",
    trust_remote_code=False,
    use_fast=True,
)
```

It MUST require `tokenizer.is_fast is True`. If a fast tokenizer or offset mapping is unavailable, validation MUST fail closed. There is no slow-tokenizer fallback.

For every full serialized member, define:

```python
full_text = f"Claim: {claim}\nEvidence: {evidence}"
evidence_char_start = len(f"Claim: {claim}\nEvidence: ")
```

Tokenize `full_text` in one operation with `add_special_tokens=False` and `return_offsets_mapping=True`. The token IDs and offset mappings from this full-string call are the sole normative source for evidence-boundary derivation. The prefix MUST NOT be separately tokenized for normative coordinates.

## 5. V1 token-coordinate-safe text domain

For O0b v1, every string that participates in the serialized tokenizer input MUST be coordinate-safe printable ASCII. The exact tokenized source fields are `claim`, `reference_sufficient`, `paraphrase_sufficient`, `insufficient_matched`, and `surface_null_matched`. Each field MUST satisfy:

```python
isinstance(text, str) and text != "" and text == text.strip() \\
    and all(0x20 <= ord(ch) <= 0x7E for ch in text)
```

This permits ordinary internal ASCII spaces and punctuation, while rejecting LF, CR, tab, all other ASCII controls, leading/trailing spaces, CJK, emoji, accented non-ASCII Latin characters, smart quotes, non-ASCII dashes, zero-width characters, and all other non-ASCII Unicode. This is an O0b v1 token-coordinate identifiability rule only. It is not a general tokenizer limitation, does not claim that Mamba cannot process Unicode, and is not a scientific result. Rationale strings that are not serialized into tokenizer input are not restricted by this token-coordinate rule. A later Unicode-capable O0b dataset requires a separate superseding authority defining an exact Unicode coordinate conversion/offset contract.

## 6. Offset-mapping validity

For every full serialized ASCII string, validation MUST require `len(input_ids) == len(offset_mapping)`. Every offset MUST contain exactly two integers `(start, end)` and be a half-open span satisfying `0 <= start < end <= len(full_text)`. Zero-length offsets are prohibited.

For each consecutive pair of token spans `i` and `i + 1`, validation MUST require `start[i] <= start[i + 1]`, `end[i] <= end[i + 1]`, and `end[i] <= start[i + 1]`. Ordinary token spans therefore MUST be monotonically ordered and non-overlapping; repeated or overlapping ordinary spans are prohibited. Any malformed, zero-length, reversed, non-monotone, or out-of-range offset MUST FAIL CLOSED. Offsets MUST NOT be normalized, repaired, merged, rewritten, or reinterpreted.

Independent verifier finding: for the exact frozen fast tokenizer `state-spaces/mamba-130m-hf`, revision `5708daa364c50b880e7bd92eab456e0d34492ee9`, tested ordinary ASCII strings produced offsets compatible with Python string-index coordinates, but permitted non-ASCII probes produced zero-length offsets, overlapping adjacent offsets, and ambiguous byte-fallback span attribution. O0b v1 therefore freezes the narrower printable-ASCII tokenized text domain instead of inventing an unverified Unicode coordinate conversion.

## 7. Evidence-start definition and recorded fields

After all strict offset checks pass, exactly one token span MUST contain `evidence_char_start`: `offset_start[j] <= evidence_char_start < offset_end[j]`. No other token span may contain that character. If no span, or more than one span, contains it, validation MUST FAIL CLOSED. Define `evidence_start_index = j`. This is equivalent to the minimum index `i` such that `offset_mapping[i].end > evidence_char_start` only after the strict ordered, non-overlapping, and unique-coverage conditions have passed.

Tokens with `end <= evidence_char_start` are strictly pre-evidence. If a token starts exactly at the boundary (`start == evidence_char_start`), it is valid and `boundary_crossing=false`. If it begins in the scaffold and extends into evidence (`start < evidence_char_start < end`), it is valid and `boundary_crossing=true`. A token ending exactly at the boundary (`end == evidence_char_start`) is strictly pre-evidence.

A token MAY cross the scaffold/evidence character boundary when `offset_start < evidence_char_start < offset_end`. Such a token is evidence-overlapping because its identity depends in part on evidence characters. This is not padding and does not modify the evidence text.

For every condition, the canonical validation artifact MUST record `evidence_char_start`, `evidence_start_index`, `evidence_start_offset_start`, `evidence_start_offset_end`, and `boundary_crossing` as `true` or `false`.

## 7. Matched-set pre-evidence invariant

Within each matched set, `evidence_start_index` MUST be identical across all four conditions. At every token position strictly before that common index, all four conditions MUST have identical token IDs and identical offset mappings.

This proves that token-coordinate differences begin no earlier than the first evidence-overlapping token. Any failure of the common-index, pre-evidence token-ID, or pre-evidence offset invariant MUST fail closed.

## 8. First divergence

Pairwise first divergence remains defined on FULL-sequence token IDs for reference versus paraphrase, reference versus insufficient, and reference versus surface-null.

`first_divergent_token_index` is the first index whose FULL-sequence token ID differs. It MUST satisfy `first_divergent_token_index >= evidence_start_index` and `first_divergent_token_index < terminal_index`.

A first divergence exactly at `evidence_start_index` is valid, including when that token crosses the scaffold/evidence character boundary, because the strictly pre-evidence token prefix has already been proven invariant. Divergence MUST NOT be redefined using character positions or semantic words.

## 9. Anchor contract

Preserve exactly the existing anchor schedule:

```text
divergence - 1
divergence
divergence + 1
divergence + 2
divergence + 4
terminal
```

Preserve all existing null/out-of-range and exact-key rules. There is no best-anchor selection and no new threshold.

## 10. Natural-evidence whitespace contract

Future authored evidence MUST satisfy `evidence == evidence.strip()` for all four evidence conditions. Explicitly reject leading LF, leading CR, leading tab, leading space, trailing whitespace, invisible Unicode padding, artificial delimiter/punctuation padding, and repeated filler inserted for tokenization.

Internal natural spaces and punctuation remain allowed. Equal token counts MUST be achieved through natural semantic wording only.

## 11. Future bounded four-file repair

Only after this candidate is independently verified and frozen may a later bounded implementation repair modify these four files and no others:

1. `data/longterm_o0b_matched_controls_v1.jsonl`
2. `scripts/validate_longterm_o0b_matched_controls.py`
3. `tests/test_validate_longterm_o0b_matched_controls.py`
4. `reports/longterm_o0b_matched_controls_v1_validation.json`

That repair MUST remove all artificial leading LF, re-author matched sets as necessary, improve pair 003 insufficiency matching if retained, implement the full-string offset method, and regenerate all tokenizer metadata, hashes, and artifact bytes. The current four-file hashes are diagnostic historical implementation identities only and MUST NOT be retained as expected repaired hashes.

Before any tokenizer invocation for a matched set/member, the future validator MUST execute this domain guard for every tokenized source field: `claim`, `reference_sufficient`, `paraphrase_sufficient`, `insufficient_matched`, and `surface_null_matched`. The normative ordering is: (1) load/parse authored source data; (2) validate source field type/content and the printable-ASCII, non-empty, stripped-text coordinate domain; (3) if any field is invalid, FAIL CLOSED immediately; and (4) only after all required fields pass may `AutoTokenizer`, tokenization, or offset derivation be invoked. The validator MUST NOT invoke the tokenizer first and reject afterward, use tokenizer output to decide whether invalid Unicode/whitespace is acceptable, tokenize invalid text for diagnostic continuation in the canonical path, or strip, normalize, transliterate, replace, or otherwise mutate invalid text before tokenization. Invalid coordinate-domain text MUST terminate validation before the tokenizer is called for that input.

## 13. Required future validation-artifact delta

The future canonical validation artifact MUST explicitly record `tokenized_text_coordinate_domain = "printable_ascii_u0020_u007e"` (or a semantically equivalent frozen constant), in addition to the five per-condition boundary fields listed in Section 7. It MUST also record matched-set checks proving the common `evidence_start_index`, exact pre-evidence token-ID invariant, and exact pre-evidence offset invariant.

All prior provenance fields remain required: `scientific_design_authority_commit`, `implementation_authority_commit`, and `repository_head`.

A later repaired implementation MUST bind separately to this authority's freeze commit, for example with `boundary_recovery_authority_commit`. This candidate MUST NOT predict that future commit SHA.

## 14. Required future test contract

Future repaired tests MUST include coordinate-domain coverage: ordinary ASCII prose, ASCII apostrophe, comma/period/colon/semicolon, internal ASCII spaces, and numeric dates such as `2021` and `1066` MUST pass; CJK, emoji, an accented character such as `é`, smart apostrophe/quotation mark, en dash/em dash, zero-width character, leading/trailing space, LF, CR, and tab MUST fail. The source-text rejection MUST be asserted to occur before normative offset coordinate derivation. Invalid input must fail closed without transliteration, Unicode normalization, punctuation replacement, whitespace stripping-and-continuing, or authored-data mutation.

For representative invalid tokenized source inputs, at minimum CJK, emoji, an accented non-ASCII character, leading LF, leading space, and trailing space, future tests MUST use a fake/mock tokenizer or equivalent call-tracking mechanism and assert both validation failure and tokenizer invocation count equal to zero. A clean printable-ASCII input test MUST assert that domain validation passes and that the tokenizer may then be called. This verifies ordering, not merely rejection.

Future repaired tests MUST include:

### Offset mapping

- fast tokenizer required;
- slow tokenizer rejected;
- missing offset mapping rejected;
- token/offset length mismatch rejected;
- zero-length span rejected;
- overlapping adjacent spans rejected;
- reversed/non-monotone start rejected;
- reversed/non-monotone end rejected;
- malformed offset rejected;
- out-of-range offset rejected;
- no token covering `evidence_char_start` rejected;
- multiple tokens covering `evidence_char_start` rejected;
- token starting exactly at the boundary accepted;
- one boundary-crossing token accepted with `boundary_crossing=true`.

### Boundary cases

- token wholly beginning at `evidence_char_start` accepted;
- token crossing the scaffold/evidence boundary accepted as evidence-overlapping;
- crossing token sets `boundary_crossing=true`;
- first evidence-overlapping token computed correctly;
- different `evidence_start_index` across conditions fails;
- differing pre-evidence token IDs fails;
- differing pre-evidence offsets fails;
- first divergence before `evidence_start_index` fails.

### Whitespace

- leading LF rejected;
- leading space rejected;
- trailing LF rejected;
- trailing space rejected;
- clean natural prose accepted.

### Regression

Preserve all existing tests for tokenizer ID/revision, `add_special_tokens=False`, `trust_remote_code=False`, no model/model weights, schema, equal full lengths, divergence, anchors, canonical artifact bytes, dataset SHA, and provenance.

## 15. Scientific interpretation boundary

This authority repairs token-coordinate identifiability only. It does not authorize or establish model execution, hidden-state separation, unsupported-commitment prediction, semantic sufficiency sensitivity, generalization, significance, or any O0b scientific conclusion.

## 16. Supersession scope

Once independently verified and frozen, this authority supersedes the separately-tokenized prefix-sharing boundary rule in implementation authority `31e6d7882586e312f783cb2fd69718eb1ee7e452`, and only the directly dependent validation, test, and artifact clauses.

It supersedes/clarifies only coordinate-domain assumptions, offset validity, and directly dependent validator, test, and artifact requirements. It does not alter the O0b scientific purpose, condition semantics, three matched sets, equal full serialized token counts, exact tokenizer/revision, `add_special_tokens=False`, `trust_remote_code=False`, semantic human review, divergence, anchor schedule, four-file repair scope, or no-model/no-training boundary.

## 17. Current implementation status and protected state

The current four untracked O0b implementation files are not freeze-ready because the dataset contains artificial leading LF evidence. They MUST remain untouched during this authority-candidate task: do not delete, rewrite, stage, commit, or regenerate them.

Unrelated state is protected, including the 75 historical root `.patch` files, `p3w7_a0_final_verify_focus_tmp/`, `p3w7_a0_final_verify_full_rs_tmp/`, `p3w7_a0_final_verify_full_tmp/`, `reports/stage180a_pass2_annotations_completed.csv`, all URP/reason-router work, existing frozen O0a/O0b authorities, and the four current diagnostic O0b implementation files.

## 18. Candidate verdict

`PASS_READY_FOR_INDEPENDENT_BOUNDARY_AUTHORITY_REVERIFICATION`

This verdict means the specification candidate is ready for the separately required independent boundary-authority verification. It is not a freeze decision, implementation approval, or scientific result.
