# Longterm O0b Scientific Interpretation Report Candidate

## 1. Status and authority

Status: candidate report for independent verification before any canonical scientific evidence freeze.

Overall scientific interpretation encoded by this candidate:

> narrow sufficiency-sensitive precursor clue supported

Authority order used:

1. Current controller instruction for this report-authoring task.
2. Frozen O0b scientific runtime-version recovery execution authority: `5079b3cc738618d9afba25397b73d499432fcfc7`.
3. Frozen repaired observer implementation: `44c5ba4f2204167f91c7f5564c6dbfcd82304035`.
4. Frozen original O0b scientific design authority: `df461469cb087f7f5db1e41a2b08e65ea517ad8`.
5. Imported and publication-validated v2 scientific artifact bundle.
6. Repository `AGENTS.md`.

This report is authored in `STATIC SCIENTIFIC INTERPRETATION REPORT AUTHORING ONLY` phase. It does not authorize implementation, training, evaluation, dataset regeneration, checkpoint mutation, model loading, tokenizer invocation, generation, Kaggle execution, best-layer selection, best-anchor selection, or architecture modification.

Canonical repository HEAD confirmed for authoring: `5079b3cc738618d9afba25397b73d499432fcfc7`.

## 2. Scientific question

The question is whether the verified O0b v2 artifact bundle supports a bounded descriptive interpretation that insufficient evidence retains a distinguishable native Mamba hidden-state-proxy response relative to semantic-preserving matched controls after controlling token count, claim bytes, serialization scaffold, terminal position, surface wording controls, and pair-relative first-divergence coordinates.

The question is not whether O0b estimates hallucination probability, detects hallucination, proves a causal mechanism, establishes statistical significance, generalizes to a population, selects an optimal layer or anchor, or authorizes any architecture change.

## 3. Frozen measurement boundary

The read-only evidence source is:

`C:\o0b-scientific-v2\reports\longterm_o0b_token_aligned_native_mamba_state_dynamics_44c5ba4_v2`

Required artifacts used as read-only evidence:

- `manifest.json`
- `anchor_observations.jsonl`
- `anchor_hidden_states.npz`
- `paired_distances.jsonl`
- `summary.json`
- `report.md`
- `SHA256SUMS.txt`

The artifacts were not modified, copied, regenerated, normalized, or rewritten for this candidate. The primary measure is normalized L2 distance. Cosine is algebraically redundant with normalized L2 under this measurement design and is not treated as independent evidence.

Required comparison terminology:

- A = `D_l2(insufficient_matched, reference_sufficient)`
- B = `D_l2(paraphrase_sufficient, reference_sufficient)`
- C = `D_l2(surface_null_matched, reference_sufficient)`

## 4. Execution and artifact provenance

Represented in the imported v2 artifacts:

- `RUN_NAME`: `longterm-o0b-token-aligned-native-mamba-state-dynamics-44c5ba4-v2`
- Execution implementation commit: `44c5ba4f2204167f91c7f5564c6dbfcd82304035`
- Observer SHA256: `fa4935a57baebf6b726b8e94c682afa7bcffc3d22b9cf6552a17eeb26ba3a63a`
- Runtime: `python 3.12.13`, `numpy 2.0.2`, `torch 2.10.0+cpu`, `transformers 5.0.0`
- Execution status: `COMPLETE`
- Pair IDs: `o0b_pair_001`, `o0b_pair_002`, `o0b_pair_003`
- Layer count: 25
- Aggregate count in `summary.json`: 150
- Pre-divergence integrity: `True`

Controller/import/publication validation facts carried into this report as frozen provenance context:

- Layer-1 command SHA256: `1a76f1c14798c27db89a55aea1a32682964487bfe35ec74cbb614ef017f455bd`
- `STARTED_UTC=2026-09-02T07:28:57Z`
- `FINISHED_UTC=2026-09-02T07:29:37Z`
- `EXIT_CODE=0`
- Run log SHA256: `6ac2d2986235413af1eea889b5ca97b41a07b12b01cb3c958febea209c9aab7a`
- Run meta SHA256: `c4a39c26478047980132609152c876662d3ffe314e1ee4457b20824bfc9ea20a`
- Handoff ZIP SHA256: `ad544879969cc786915ad7ee0aa1f23ea5fb0e9e524c715da5b980ecc72383ac`
- Import: `VALIDATED=7`, `COPIED=7`, `IDENTICAL=0`, `IMPORT PASS`
- Publication validation: `PUBLICATION_VALIDATION=PASS`, `MANIFEST_PROVENANCE=PASS`, `FILE_COUNT=7`, `EXECUTION_STATUS=COMPLETE`

## 5. Integrity result

The v2 artifacts support the integrity boundary required for this static interpretation:

- `PRE_DIVERGENCE_ALL_PASS=True`
- `PAIR_IDS=["o0b_pair_001", "o0b_pair_002", "o0b_pair_003"]`
- `LAYER_COUNT=25`
- `AGGREGATE_COUNT=150`
- Execution status is `COMPLETE`

No pre-divergence failure is present in the inspected summary integrity object. Under the frozen falsification matrix, a pre-divergence failure would invalidate the run; that invalidating condition is not observed here.

## 6. A/B/C interpretation method

For each frozen anchor and layer, the report compares whether A exceeds B and whether A exceeds C under normalized L2. The anchor-level histograms summarize how many of the three pair IDs show A>B or A>C at each layer. The frozen `BOTH_PAIR_COUNTS_*` anchor summary is interpreted as layer-level AB/AC co-support using `min(AB_count, AC_count)` at that layer. The pair-by-anchor table separately reports same-pair intersection counts, where `both` means A>B and A>C are true for the same pair and layer.

The interpretation matrix is applied without modification:

- Persistent A > B and A > C across multiple pair IDs and layers supports a narrower native sufficiency-sensitive precursor clue.
- Collapse after matching would favor length/position/surface confounding.
- Similar separation for all semantic manipulations would favor broad intervention sensitivity.
- Sufficient paraphrase as strong as insufficiency would favor wording sensitivity.
- Isolated pair/layer/anchor effects are weak/unstable and must not be promoted.
- Pre-divergence failure invalidates the run.
- No hard scientific PASS threshold exists.
- No statistical significance or population/generalization claim is authorized.

## 7. Frozen early-anchor results

`anchor_divergence`:

- `AB_HIST={1:4, 2:9, 3:12}`
- `AC_HIST={1:2, 2:17, 3:6}`
- `A_MEAN_GT_B_AND_C=19/25 layers`
- `BOTH_PAIR_COUNTS_3_OF_3=6/25`
- `BOTH_PAIR_COUNTS_GE_2_OF_3=19/25`

`anchor_post_plus_1`:

- `AB_HIST={1:18, 2:7}`
- `AC_HIST={1:7, 2:11, 3:7}`
- `A_MEAN_GT_B_AND_C=24/25`
- `BOTH_PAIR_COUNTS_3_OF_3=0/25`
- `BOTH_PAIR_COUNTS_GE_2_OF_3=7/25`

`anchor_post_plus_2`:

- `AB_HIST={0:1, 1:3, 2:5, 3:16}`
- `AC_HIST={0:1, 1:1, 2:8, 3:15}`
- `A_MEAN_GT_B_AND_C=20/25`
- `BOTH_PAIR_COUNTS_3_OF_3=11/25`
- `BOTH_PAIR_COUNTS_GE_2_OF_3=21/25`

`anchor_post_plus_4`:

- `AB_HIST={0:2, 1:17, 2:3, 3:3}`
- `AC_HIST={0:2, 1:11, 2:12}`
- `A_MEAN_GT_B_AND_C=15/25`
- `BOTH_PAIR_COUNTS_3_OF_3=0/25`
- `BOTH_PAIR_COUNTS_GE_2_OF_3=5/25`

The strongest cross-pair early consistency is at `anchor_post_plus_2`, but this is not a selected or best anchor. It is only one member of the a-priori frozen anchor schedule, and no post-hoc promotion rule is derived from it.

## 8. Pair heterogeneity

Pair-by-anchor cross-layer counts:

`anchor_divergence`:

- `o0b_pair_001`: A>B `21/25`; A>C `6/25`; both `6/25`
- `o0b_pair_002`: A>B `12/25`; A>C `25/25`; both `12/25`
- `o0b_pair_003`: A>B `25/25`; A>C `23/25`; both `23/25`

`anchor_post_plus_1`:

- `o0b_pair_001`: A>B `24/25`; A>C `18/25`; both `18/25`
- `o0b_pair_002`: A>B `8/25`; A>C `7/25`; both `6/25`
- `o0b_pair_003`: A>B `0/25`; A>C `25/25`; both `0/25`

`anchor_post_plus_2`:

- `o0b_pair_001`: A>B `21/25`; A>C `21/25`; both `21/25`
- `o0b_pair_002`: A>B `22/25`; A>C `18/25`; both `17/25`
- `o0b_pair_003`: A>B `18/25`; A>C `23/25`; both `18/25`

`anchor_post_plus_4`:

- `o0b_pair_001`: A>B `22/25`; A>C `23/25`; both `22/25`
- `o0b_pair_002`: A>B `3/25`; A>C `0/25`; both `0/25`
- `o0b_pair_003`: A>B `7/25`; A>C `12/25`; both `6/25`

These results are pair-heterogeneous. They support a bounded descriptive interpretation only because the signal is not isolated to a single pair/layer/anchor, while the observed variability prevents any uniform-persistence or generalization claim.

## 9. Matched-terminal diagnostic

`anchor_terminal`:

- `AB_HIST={2:3, 3:22}`
- `AC_HIST={2:2, 3:23}`
- `A_MEAN_GT_B_AND_C=25/25`
- `BOTH_PAIR_COUNTS_3_OF_3=20/25`
- `BOTH_PAIR_COUNTS_GE_2_OF_3=25/25`

Pair-by-anchor cross-layer counts:

- `o0b_pair_001`: A>B `25/25`; A>C `24/25`; both `24/25`
- `o0b_pair_002`: A>B `23/25`; A>C `24/25`; both `22/25`
- `o0b_pair_003`: A>B `24/25`; A>C `25/25`; both `24/25`

The matched-terminal diagnostic argues against reducing the observed separation to a pure terminal-position explanation. It remains a diagnostic, not a causal mechanism and not authority for terminal-anchor optimization.

## 10. Falsification-matrix assessment

The evidence does not collapse after the frozen matching controls. That weighs against a pure length, token-count, claim-byte, serialization-scaffold, surface-wording, first-divergence-coordinate, or terminal-position confounding explanation.

The evidence is not uniformly persistent across all early anchors. Early-anchor behavior is temporally localized, with the strongest cross-pair early consistency at `anchor_post_plus_2` inside the frozen a-priori schedule and weaker or more heterogeneous behavior at other early anchors.

The evidence is not equally strong for all semantic manipulations. A remains distinguishable from both the sufficient paraphrase and the surface-null matched control in the summarized comparisons, which weighs against interpreting the result as only broad intervention sensitivity or only wording sensitivity.

The evidence contains pair heterogeneity. Isolated pair/layer/anchor effects are not promoted, and no hard scientific PASS threshold is imposed.

## 11. Scientific conclusion

narrow sufficiency-sensitive precursor clue supported

After controlling token count, claim bytes, serialization scaffold, terminal position, surface wording controls, and pair-relative first-divergence coordinates, insufficient evidence retains a distinguishable native Mamba hidden-state-proxy response relative to semantic-preserving matched controls.

This is evidence for a precursor clue: A is often larger than both B and C across multiple pair IDs and layers, including the frozen early-anchor summaries and the matched-terminal diagnostic.

This is evidence against a pure length or terminal-position explanation: separation persists after matched controls and is visible in the matched-terminal diagnostic.

This is also evidence of temporal localization and pair heterogeneity: the early-anchor pattern is not uniformly persistent across all early anchors, and pair-level counts vary substantially by anchor.

This is not a calibrated hallucination probability, not a detector score, not a causal mechanism, not a significance result, and not a generalization result.

## 12. Explicit non-claims

This report does not claim:

- a hallucination probability;
- a detector score;
- a causal mechanism;
- statistical significance;
- population generalization;
- best-layer selection;
- best-anchor selection;
- authority for post-hoc promotion rules;
- authority for O0c/O1 implementation or execution;
- authority for architecture modification;
- authority to use external diagnostic labels for training, threshold tuning, candidate selection, or promotion.

## 13. Implication for next research phase

O0b leaves a matched-control signal worth explaining and therefore makes a later O0c/O1 native-state instrumentation authority scientifically reasonable to consider.

This report does not authorize O0c/O1 implementation or execution.

## 14. Exact evidence artifact identities

Validated artifact identities:

- `SHA256SUMS.txt`: bytes=`507`, sha256=`65b9dc03d7bba58d066cd3c064fd9175b9a23d841d0f1a9024578d300e46eec5`
- `anchor_hidden_states.npz`: bytes=`6758648`, sha256=`f683c78c563bfd9421e3b3798f4409928b5d8fd41a35a4638be55926c5fb61d5`
- `anchor_observations.jsonl`: bytes=`1248721`, sha256=`7fe8e31576ef018de948551f34116b98b437b85c281349c4d268409e5bf088c2`
- `manifest.json`: bytes=`6113`, sha256=`9eb7f4bc6c55c0fc103df9797597bf181ff23cd212f32af0afef4f778fd292ce`
- `paired_distances.jsonl`: bytes=`921248`, sha256=`9c650ed671ee2cb53de65744e619c6f282a699fc5c7bdb46c3c60c28a8f3b8f6`
- `report.md`: bytes=`177737`, sha256=`ac8d2b405fca665c800d8aef9daf83afdf1880e66a5f7a6ad37482bffb64e94c`
- `summary.json`: bytes=`179040`, sha256=`63c1d8cff5b71087746a8ca6e8dc2d860fee8aff50be48784d4b06c54db30b50`
