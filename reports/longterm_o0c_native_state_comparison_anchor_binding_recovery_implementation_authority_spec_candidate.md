# O0c Native-State Comparison-Specific Anchor-Binding Recovery Implementation Authority

## Status, authority, and phase

**Verdict:** `PASS_READY_FOR_INDEPENDENT_O0C_COMPARISON_ANCHOR_BINDING_RECOVERY_IMPLEMENTATION_AUTHORITY_VERIFICATION`

This is exactly one candidate recovery implementation authority, authored in `REPORT-ONLY O0C ANCHOR-BINDING RECOVERY IMPLEMENTATION AUTHORITY AUTHORING` phase. It authorizes neither implementation nor execution. It records the newly established comparison-specific anchor-binding defect and freezes the narrow future correction contract.

Authority precedence used:

1. Current controller instruction: O-series only.
2. Frozen O0c scientific/design authority: `242ad9ed70fc995ebda560911a7d0dfd2f18f9b3`.
3. Frozen O0c instrumentation implementation authority: `6eca52722aaffa214e8546c6b616e1f670aecf77`.
4. Frozen observer implementation: `f724b81b9b69c842652a556f850ddebf53c11987`.
5. Frozen observer `scripts/observe_longterm_o0c_selective_ssm_native_state_dynamics.py`: canonical SHA256 `ace3d6c928a2ef6b382bb9aad80c8e410d946a853fab35a45dcfb60cb5b13384`, 44023 bytes, Git blob `41027418fa0d5e117648da3ba0f4a0a08e5a3982`.
6. Frozen O0b validation artifact `reports/longterm_o0b_matched_controls_v1_validation.json`: canonical SHA256 `e8344ea3df54a3393aa8fa82dba19eb2baade9af9366687bb105f4ad348979ff`.

Required authoring HEAD is exactly `f724b81b9b69c842652a556f850ddebf53c11987`.

The failed runner-authority candidate with SHA256 `7ebc6af88b0ba79f05468b86e65531689ed9542a48fe0c1fdde20bda15f0cd55` remains context only and non-authoritative. Its verdict remains `FAIL_INPUT_VALIDATION_OR_ANCHOR_CONTRACT_AMBIGUOUS`, with secondary verdict `FAIL_ELIGIBLE_LAYER_DISCOVERY_UNDERDEFINED`.

## Established anchor-interface defect

The frozen validation artifact stores first-divergence-relative anchor indices separately for every member-to-reference comparison. Member comparisons may, and in the frozen artifact do, have different first-divergent token indices and therefore different anchor maps.

| Comparison | Member condition | First divergence for `o0b_pair_001` |
|---|---|---:|
| comparison-A | `insufficient_matched` | 25 |
| comparison-B | `paraphrase_sufficient` | 18 |
| comparison-C | `surface_null_matched` | 17 |

The frozen `measurements()` interface accepts anchors keyed by `(pair_id, condition)`. For each comparison, it obtains `anchors[(pair, member)]` and a shared `anchors[(pair, "reference_sufficient")]`, then requires matching coordinates. One shared reference map cannot represent the three distinct reference/member coordinate schedules above.

The synthetic implementation suite did not detect this defect because its anchor fixtures used identical coordinates across comparisons. This is an anchor-binding defect, not a changed scientific result.

`SCIENTIFIC_CONCLUSION: NONE`

## Exact future scope

After independent verification and activation, future implementation may modify exactly these two files and no third file:

1. `scripts/observe_longterm_o0c_selective_ssm_native_state_dynamics.py`
2. `tests/test_observe_longterm_o0c_selective_ssm_native_state_dynamics.py`

It must not modify the dataset, validation artifact, manifest schema, artifact schema, cm tooling, handoff tooling, runner files, package/runtime code, or scientific design. It must not implement a runner.

## Frozen replacement anchor API

Correct only the anchor-binding input interface used by `measurements()`. The authoritative input is `comparison_anchors`:

```text
Mapping[
    tuple[pair_id, comparison_id],
    Mapping[anchor_name, absolute_token_index],
]
```

The exact outer-key set is every `pair_id` in `PAIR_ORDER` crossed with exactly `comparison-A`, `comparison-B`, and `comparison-C`: exactly 3 x 3 = 9 comparison anchor maps. Every inner map has exactly `ANCHOR_ORDER`, with one absolute token index per anchor. Missing or extra outer keys, missing or extra anchor names, non-integer values, and indices invalid for either trajectory are rejected.

The old shapes are prohibited and must not be accepted: a shared `(pair, "reference_sufficient")` map; any condition-keyed reference map; an inferred average or union; a substituted anchor; or post-hoc anchor selection.

| Comparison ID | Member condition |
|---|---|
| `comparison-A` | `insufficient_matched` |
| `comparison-B` | `paraphrase_sufficient` |
| `comparison-C` | `surface_null_matched` |

For one `(pair_id, comparison_id)`, that map's same absolute token index is applied to the `reference_sufficient` trajectory and that comparison's member trajectory only. This applies independently to every `ANCHOR_ORDER` entry and matches the frozen matched-control design: anchors are relative to the first divergence of the specific member/reference comparison.

## Exact future `measurements()` semantics

The corrected function must:

1. validate rows and vectors exactly as before;
2. require `comparison_anchors` exact nine-key membership;
3. require each inner map's exact `ANCHOR_ORDER` membership;
4. require each anchor value to be an integer valid for both the reference and corresponding member trajectory;
5. for each pair, comparison, and layer, retrieve only `comparison_anchors[(pair, comparison_id)]`;
6. use that map's `anchor_pre_minus_1` for pre-divergence validation;
7. use each map coordinate `t` for both reference and member state lookup for that comparison only;
8. preserve `t - 1` predecessor semantics and zero-state semantics at `t = 0`;
9. preserve every existing metric formula and every measurement-row field/value semantic; and
10. never consult a shared `reference_sufficient` anchor map.

The correction must not change `COMPARISONS`, `ANCHOR_ORDER`, `MEASUREMENT_KEYS`, measurement formulas, pre-divergence tolerance, state indexing, summary semantics, the seven artifacts, or status vocabulary.

## Frozen-artifact compatibility contract

Future tests must directly use the actual frozen validation-artifact bytes, preferably by reading the repository artifact and verifying SHA256 first, or use an exact fixture extracted from those bytes. They must extract:

```text
artifact["pairs"][...]["comparisons_to_reference"][member]["anchor_indices"]
```

into the exact `(pair_id, comparison_id) -> anchor map` API. No tokenizer or model load is needed. The test must show at least one real pair with distinct A/B/C maps and accept them in corrected `measurements()` with synthetic trajectories of sufficient length. It must specifically cover frozen `o0b_pair_001` first divergences `25 / 18 / 17`. The old shared-reference representation must fail or cease to be an accepted API shape.

## Required future regression tests

In addition to all existing tests, the later two-file implementation must add or retain tests proving, without scientific model load or forward:

1. exact 9 comparison-anchor keys accepted;
2. missing comparison key rejected;
3. extra comparison key rejected;
4. missing anchor name rejected;
5. extra anchor name rejected;
6. non-integer anchor rejected;
7. out-of-range reference index rejected;
8. out-of-range member index rejected;
9. comparison-A uses only its own map;
10. comparison-B uses only its own map;
11. comparison-C uses only its own map;
12. different A/B/C maps for the same pair accepted;
13. real frozen `o0b_pair_001` `25 / 18 / 17` schedule accepted;
14. pre-divergence check uses each comparison's own `d - 1`;
15. terminal uses each comparison map's frozen terminal index;
16. measurement values still reconstruct exactly;
17. measurement ordering unchanged;
18. summary reconstruction unchanged;
19. bundle schema/bytes semantics unchanged; and
20. scientific CLI remains disabled.

## Unchanged observer semantics

All previously frozen observer properties remain required: runtime/source binding; `NativeStateObserver` semantics; source-line capture; trace restoration; fresh capture; complete `0..T` trajectory; terminal state; deterministic NPZ; measurement reconstruction; pre-divergence validation; exact manifest schema; exact seven-artifact transaction; atomic publication; success vocabulary; and CLI scientific-execution refusal.

No pretrained model/tokenizer load, scientific dataset execution, scientific forward, training, evaluation, or interpretation is authorized.

## Relationship to runner authority

The failed `7ebc6a...` candidate must not be revised in place. Required order:

1. author this anchor-binding recovery authority;
2. independent static verification;
3. freeze/commit/push authority;
4. implement exact observer/test correction;
5. independent implementation verification;
6. freeze/commit/push corrected observer;
7. author a fresh runner implementation authority against the corrected observer; and
8. in that fresh authority freeze exact O0b input serialization/token validation, `use_fast=True` tokenizer binding, comparison-specific anchor-map construction, and deterministic eligible-layer discovery.

No runner implementation is authorized before step 7.

## Boundary, validation, and next action

This candidate authorizes no model/tokenizer load, scientific dataset execution, scientific forward, Kaggle activity, runner implementation, training, evaluation, interpretation, package mutation, commit, or push. Training/evaluation, scientific execution, and Kaggle are all `NO`.

Authoring validation must establish required HEAD; exactly one untracked candidate report; no tracked changes; nothing staged; `git diff --check` on the candidate; recomputed frozen observer canonical identity; recomputed frozen validation-artifact SHA256; candidate SHA256, byte count, and Git blob; and UTF-8/BOM/LF/final-LF/trailing-whitespace facts.

`SCIENTIFIC_CONCLUSION: NONE`

`PASS_READY_FOR_INDEPENDENT_O0C_COMPARISON_ANCHOR_BINDING_RECOVERY_IMPLEMENTATION_AUTHORITY_VERIFICATION`

Exact next authorized action: independent static verification of this one candidate against the frozen authorities and required HEAD, followed only on passing verification by freeze/commit/push of the authority; do not implement, execute, or author runner work without a separately activated later authority.
