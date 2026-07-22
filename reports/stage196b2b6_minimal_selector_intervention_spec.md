# Stage196-B2-B6 Minimal Recipient-Selector Intervention Specification

## Purpose and authority

This stage converts the successful Stage196-B2-B5 cross-seed recipient-local observability result into an explicit, artifact-only selector policy. It uses Stage196-B2-B5 analyzer runtime commit `632c3be6bd9baf648cc1e43f9e91eb2940b82ff9`, the Stage196-B2-B4 primitive-action authority, and Stage196-B2-B3P0 runtime commit `fa16787efa84bb15d832b6d9382fafd77016c4e2`.

The stage authorizes no training, promotion, model loading, checkpoint loading, learned threshold, or performance-based action choice.

## Required invocation

All arguments are required and all paths are explicit:

```text
--repo-root
--stage196b2b5-analysis-json
--stage196b2b4-analysis-json
--stage196b2b3p0-run-root
--stage196b2b3p0-runtime-git-commit
--current-git-commit
--output-dir
```

The analyzer accepts only the exact analysis JSON basenames, their exact authoritative nine-file directories, and the exact P0 run directories. It performs no latest-timestamp selection, modification-time selection, recursive discovery, sibling fallback, or filename-substring search.

## Source closure

The B2-B5 source must have decision `STAGE196B2B5_CROSS_SEED_RECIPIENT_SELECTOR_LOCALIZED`, next stage `STAGE196B2B6_MINIMAL_SELECTOR_INTERVENTION_DESIGN`, no blocking reasons, the supplied analyzer commit, all contract gates passed, and exactly the nine authorized B2-B5 files. Its primary authority must contain 16 forward-direction seed-conditioned cases keyed by `(seed, stable_row_id)`: five recovery and six harm cases for seed184, and two recovery and three harm cases for seed185.

The B2-B4 source must have decision `STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION`, next stage `STAGE196B2B5_ROW_SELECTOR_OBSERVABILITY`, no blocking reasons, all contract gates passed, and exactly the nine authorized B2-B4 files. The primitive action lattice is fixed in this order:

```text
FRAME
PREDICATE
SUFFICIENCY
POSITIVE_ENERGY
NEGATIVE_ENERGY
```

The required B2-B4 counts are 20,480 primitive coalition rows, 1,024 primitive tail summaries, 32 primitive masks, 16 primary identities, and 640 directional states.

P0 must contain exactly the six named joint and frame-local-only runs for seeds 183, 184, and 185; 120 composer sidecars; 120 trajectory sidecars; 86,400 composer rows; and 86,400 trajectory rows. Composer namespaces are exact. The mixed-purpose trajectory directory is filtered only by the exact `stage196b2p0_epoch_channels_NNN.jsonl` namespace. Unrelated files are ignored, while malformed namespace-like files fail closed.

## Candidate recipient feature subsets

Every B2-B5 recipient-local subset with `feasible`, `pooled_full_pass`, and `bidirectional_cross_seed_full_pass` all true is loaded. Inclusion minimality is recomputed using exact set inclusion over semantic feature names; file order is irrelevant. Every retained feature must be available, deployment-authorized, non-diagnostic, non-outcome-derived, and an exact match for the frozen B2-B5 formula and raw source fields.

Seed, identity fields, source position, transition role, path class, subtype, counterfactual outcomes, donor-tail outcomes, B2-B4 minimal-coalition labels, and paired-treatment deltas are prohibited selector inputs.

## Independent signatures

For every candidate and every primary identity, the analyzer reconstructs the ordered epochs 18, 19, and 20 feature sequence from the P0 joint recipient composer state. It then requires exact structured equality with the stored B2-B5 pooled signature. The only numerical boundaries are zero with tolerance `1e-12` and the probability halfspace at `0.5` with tolerance `1e-12`. No learned threshold or fitted classifier exists.

## Exact action sets

For each exact signature, the analyzer intersects the B2-B4 acceptable primitive-coalition sets of all primary rows carrying that signature. It removes a coalition whenever a proper subset is also in the intersection, preserving every remaining incomparable inclusion-minimal mask.

No action is selected using accuracy, margin, effect size, donor reproduction rate, harm rate, lexicographic order, row frequency, seed frequency, or file order.

Signatures and candidates are classified purely set-theoretically:

- `UNIQUE_DETERMINISTIC`: every signature has exactly one inclusion-minimal action.
- `SET_VALUED_EXACT`: every signature has a nonempty intersection and at least one has multiple incomparable minimal actions.
- `INVALID_EMPTY_INTERSECTION`: at least one signature has no common acceptable action.

Set-valued policies remain set-valued. Each allowed action is evaluated separately; no arbitrary tie-break is introduced.

## Policy equivalence and dominance

Candidates are policy-equivalent only when all 16 primary rows receive identical inclusion-minimal action sets. Structural minima inside an equivalence class use semantic-feature set inclusion.

Candidate A dominates B only if A has a strict semantic-feature subset, identical safe primary action sets, no lower signature coverage for any audited seed, no worse safety, and both use only authorized inputs. Accuracy improvements do not participate. Every nondominated candidate is preserved.

## Primary validation

Each assigned action is re-applied from raw P0 composer inputs at tail epochs 18, 19, and 20. Only raw primitive fields are installed from the paired frame-local-only donor. Entitlement, decision-head logits, final deltas, final logits, prediction, and margin are recomputed. Recomputed tail predictions must exactly agree with B2-B4 coalition artifacts.

A valid policy has 16 primary rows, seven recovery rows reproducing donor tails, nine harm rows preserving recipient tails, and zero objective failures. A set-valued signature is safe only if every allowed action satisfies every corresponding row objective.

## Complete clean-dev audit

For each candidate, the audit includes all 720 identities for each of seeds 183, 184, and 185, yielding exactly 2,160 seed-identity states. Each state contains the exact joint recipient tail sequence and its reconstructed signature.

Signature support, not the coalition mask, determines abstention. For output-schema compatibility, a row whose signature is present in the reconstructed policy map is labeled `PRIMARY_SEEN_SIGNATURE` when its `(id, source_row_id, dev_position)` belongs to the 13-identity discovery union and `NONPRIMARY_SEEN_SIGNATURE` otherwise. These support labels do not change the separate 16-case primary authority. Any signature absent from the policy map is `UNSEEN_SIGNATURE`.

For `UNSEEN_SIGNATURE`, the audit requires `abstained=true` and records `assigned_action_set=["00000"]` as a fallback representation. The application retains the unmodified joint recipient prediction, explicitly verifies that the selector prediction equals the joint baseline prediction, and requires `prediction_changed=false`. These rows do not count as seen coverage or selector success.

For a seen signature, the audit requires `abstained=false`, a nonempty assigned action set, and exact equality between that set and the reconstructed signature action set. A seen mapping of `assigned_action_set=["00000"]` is a legitimate `SEEN_EXACT_NO_OP_POLICY`: the signature was recognized and the exact inclusion-minimal action is the empty primitive coalition. It remains seen coverage and must not be reclassified as abstention.

For seen signatures, the paired donor is matched by exact seed, epoch, id, source row id, and development position. Unique policies apply their sole action. Set-valued policies produce separate signature-action interpretations, leaving other signatures unmodified in that interpretation, so every allowed action is audited without collapsing the set.

The `unseen_signature_abstention_closure` contract records structured counts for total, unseen, seen, invalid-support, mapping-presence, abstention-state, fallback-mask, empty-action, action-set agreement, seen exact no-op, and joint-prediction-retention cases. It passes only when every violation count is zero. The legitimate `seen_exact_no_op_policy_rows` count is evidence, not a violation, and is also included in the clean-dev signature-support analysis summary and the report unseen-signature section.

Descriptive gold-label metrics use the exact epoch-20 tail endpoint, while the signature and primary-objective checks retain all three tail epochs. Metrics include accuracy, macro F1, SUPPORT recall, false NOT_ENTITLED, false entitlement, polarity errors, prediction changes, correct-to-incorrect, incorrect-to-correct, and stable-correct preservation.

## Primary cases and clean-dev data identities

The analyzer keeps two population systems separate. Primary selector validation uses 16 seed-conditioned cases keyed by `(seed, stable_row_id)`: 11 for seed184 and five for seed185. These cases remain the sole authority for the seven recovery and nine harm objectives.

The clean-dev partition instead uses data identities keyed by `(id, source_row_id, dev_position)`. The analyzer constructs the actual seed184 and seed185 identity sets from the 16 cases and requires 11 seed184 identities, five seed185 identities, their exact three-identity intersection, and their 13-identity union. The intersection and union are verified as sets; they are not inferred only from arithmetic. Overlapping identities are neither duplicated nor padded, so 16 cannot be used as a per-seed clean-dev row count.

Before policy evaluation, each seed must expose exactly the same 720 unique data identities, with every discovery-union identity present exactly once. Seed184 and seed185 are reported as `DISCOVERY_IDENTITY_UNION` (13), `NONDISCOVERY` (707), and `ALL_720` (720). Seed183 uses the same union only as `DISCOVERY_IDENTITY_UNION_CONTRAST` (13), plus `NONDISCOVERY` (707) and `ALL_720` (720). Row classification is determined only by membership of `(id, source_row_id, dev_position)` in the discovery union. Seed183 remains contrast-only: it cannot redefine recovery or harm labels, retrofit a selector, or support a formal held-out-generalization claim.

## Audit completion versus safety outcome

Audit completion and scientific safety outcome are separate contracts. Completion asks whether every required candidate and policy-action mode produced exactly one summary for the required seed and population, with the frozen population row count and all metrics present and finite where defined. It fails closed for missing, duplicate, unexpected, wrong-population-size, or malformed metric summaries.

The seed184 and seed185 completion gates use structured `required={"completed":true}` semantics and structured observed evidence. Each requires `NONDISCOVERY`, 707 rows per summary, every expected candidate-policy-mode key exactly once, and no metric defects. `safety_pass_count` and `safety_fail_count` are descriptive; their sum must close the audit, but safety failures do not fail completion.

The seed183 contrast population gate requires `DISCOVERY_IDENTITY_UNION_CONTRAST`, `NONDISCOVERY`, and `ALL_720` with row counts 13, 707, and 720, respectively. It also requires signature-support closure and computed prediction-change, correct-to-incorrect, and stable-correct-preservation metrics. Seed183 remains contrast-only and is never a hard authorization condition.

Safety outcome remains the existing exact result: `correct_to_incorrect_count == 0` and `stable_correct_preservation_rate == 1.0` determine each summary row's `safety_passed`. A completed unsafe audit proceeds to the ordered decision rules and may produce `STAGE196B2B6_PRIMARY_EXACT_CLEAN_DEV_UNSAFE`; it is not converted into a contract failure.

Other completion gates use schema-consistent structured requirements and evidence for the 2,160-state signature audit, paired recomposition, policy equivalence, ordered dominance pairs, and decision evaluation. The exact nine-output closure continues to compare the same deterministic filename list on both sides.

## Safety authorization

A unique policy is safety-authorized only if primary closure is exact and both seed184 and seed185 `NONDISCOVERY` populations of 707 rows have zero correct-to-incorrect transitions and stable-correct preservation of 1.0. The compatibility fields `seed184_nonprimary_safety_passed` and `seed185_nonprimary_safety_passed` therefore mean the complement of the 13-identity cross-seed discovery union; they do not retain the obsolete 704-row interpretation. A set-valued policy must meet those conditions for every allowed signature-action interpretation. There is no average-action authorization. Seed183 is required evidence but is not a hard authorization condition.

## Ordered decisions

The analyzer applies the frozen rules in order:

1. contract failure → `STAGE196B2B6_BLOCKED_CONTRACT_FAILURE` / `STAGE196B2B6_REPAIR`;
2. one nondominated safe unique policy class → `STAGE196B2B6_UNIQUE_MINIMAL_SAFE_SELECTOR` / `STAGE196B2B7_SELECTOR_ARCHITECTURE_INTEGRATION_DESIGN`;
3. multiple nondominated safe exact policy classes, or a remaining safe set-valued policy → `STAGE196B2B6_MULTIPLE_SAFE_SELECTOR_POLICIES` / `STAGE196B2B7_CONTROLLED_SELECTOR_POLICY_ABLATION`;
4. primary exact but all candidates unsafe on seed184 or seed185 nondiscovery rows → `STAGE196B2B6_PRIMARY_EXACT_CLEAN_DEV_UNSAFE` / `STAGE196B2B6P0_SELECTOR_SAFETY_STATE_OBSERVABILITY`;
5. primary exact with coverage dominated by unseen-signature abstention → `STAGE196B2B6_SIGNATURE_SUPPORT_INSUFFICIENT` / `STAGE196B2B6P0_SELECTOR_COVERAGE_OBSERVABILITY`;
6. all candidates set-valued with unsafe ambiguity → `STAGE196B2B6_ACTION_AMBIGUITY_UNRESOLVED` / `STAGE196B2B6P0_ACTION_DISAMBIGUATION_OBSERVABILITY`.

Any unresolved state fails closed as a contract failure.

## Outputs and atomicity

The output directory must not already exist. Exactly nine files are staged in a new temporary sibling directory, flushed and synchronized, checked for exact closure, and atomically renamed:

```text
stage196b2b6_analysis.json
stage196b2b6_report.md
stage196b2b6_candidate_feature_subsets.csv
stage196b2b6_signature_action_map.csv
stage196b2b6_primary_policy_validation.csv
stage196b2b6_clean_dev_signature_audit.csv
stage196b2b6_clean_dev_application_summary.csv
stage196b2b6_policy_dominance.csv
stage196b2b6_contract.csv
```

Structured CSV fields use deterministic sorted JSON. Contract `required` and `observed` cells are always valid structured JSON; booleans remain JSON booleans rather than string values inside those objects. Contract failure still produces the exact nine-file diagnostic bundle.

## Scientific limits

This stage does not establish formal causal mediation, external or OOD validity, unfrozen-Mamba validity, training improvement, promotion, paired-delta deployability, or safety from primary closure alone. It does not treat abstention as selector success, seed183 as an independent dataset, an arbitrary tie-break as determinism, accuracy improvement as authorization, row identity as a selector feature, or path class/subtype as inference-time information.
