# ContraMamba RFR-D2 — Independent Case-Family and Shortcut-Controlled Cohort Design

- Date: 2026-10-17 (Asia/Seoul)
- Status: `SCIENTIFIC_DATA_DESIGN_CANDIDATE / STATIC_ONLY / NO_DATA_GENERATION`
- Branch: `gen5-causal-role-state-ownership`
- Parent HEAD: `2a23f9f6f10b4b1642c1a4374641374c49638b5c`
- Parent: `reports/rfr_d2_structural_cohort_feasibility_static_audit_candidate.md`
- Semantics: D0 and D1 frozen documents and `scripts/rfr_d1_deterministic_oracle_cpu.py` are controlling contracts.
- Scope: propose an independently generated synthetic *case-family* sampling frame, prospective lexical / source / position counterbalances, group-level split rules and no-outcome feasibility gates. Do not materialize data, alter old datasets or select scientific thresholds.

## 1. Scientific question and limits

Can a model use evidence provenance and logical time to revise an initially verified A, update its consequence B = A AND C only when C permits, and preserve unrelated U through later queries and distractors? The three primary arms (`VALID_FLIP`, `VALID_SHAM`, `INVALID_FLIP`), Boolean baseline `(a0,c0,u0)`, readouts (`PRE`, `POST`, `DELAYED`), three query targets (`A`, `B`, `U`), truth labels and D1 acceptance rules are immutable.

All case families implement the **same supplied conjunction**. Different names, domains, and templates can support held-out *surface generalization*, but are not independent logical laws or evidence for general human-like belief revision. Model correctness would remain a bounded input/output result, not native recurrent-state mediation.

## 2. Family identity and sampling-frame proposal

Define a *case-family* as one independently specified synthetic micro-world consisting of:

- one family-specific nonsemantic namespace of invented entity and record identifiers; four distinct propositions A, B, C, U; and one explicit rule `B := A AND C`;
- one immutable latent Boolean world `(a0,c0,u0)` with initially authenticated A/C/U at logical timestamp 10;
- one fixed set of semantic statements about A, C and U, whose paraphrases and contrary values are available within the family;
- one source-authentication protocol, with source IDs and provenance identities explicitly shown in the model input (but not final labels);
- all three arm realizations, readouts, query targets, event-presentation permutations, truth-direction reversals, lexical paraphrases and distractor variants as **linked descendants of the same family**.

One `case_family_id` has one `split_group_id`, one split assignment and potentially multiple `case_id` descendants. The same identifiers, entity tuple, paraphrase stems, source-attestation instance and all perturbations derived from them must not cross splits. Reversing the initial truth assignments or changing the t1 arm of a semantic world is never a fresh independent statistical sample.

For *candidate source diversity*, propose two unrelated synthetic surface domains (e.g., registry approvals and laboratory instrument statuses), each with independent grammar/lexicon pools and no factual real-world truth dependencies. These are **proposed renderers**, not frozen generator code or validated independent samples. Each case must remain an explicit positive/negative proposition (false = verified false, not unmentioned). Do not use names, dates or lexical items from historical XG1 examples as a substitute for a new source family. A later lexical-domain holdout should be distinguished from a shuffled within-template holdout.

## 3. Factorial coverage and unit of analysis

Stratify at family level by all eight `(a0,c0,u0)` settings, with **equal family counts per stratum within any planned split** if a split is retained. Define symbolic count `K_s` distinct families per truth configuration in split `s`, not a chosen numerical case count; primary families per split are `8*K_s`. Under exactly one arm/readout/query realization per family, the primary answer checks would be `8*K_s*3*3*3 = 216*K_s`. Additional permutations and paraphrases increase correlated rows but **not** independent families. All `K_s`, seed, train/dev/test allocations, power and precision thresholds remain `UNSET`.

The factorial contains no primary `NOT_ENTITLED` target because A/C/U are initially known. Keep the **separate, predeclared missing-premise eligibility control** with unknown C or A, following D1 Kleene logic; do not mix it into the three-arm primary estimand, and do not artificially insert an unknown target to rebalance labels. Under the fixed primary logical design, the answer marginals are 126 REFUTE / 90 SUPPORT per 216 checks; B is 54 REFUTE / 18 SUPPORT. These proportions are design-derived and are not evidence of model behavior.

Joint case-level validity, not row-wise overall accuracy, will eventually distinguish these endpoints: A changes only under VALID_FLIP; B changes with A only when `C=T`; B remains false when `C=F`; U remains unchanged; VALID_SHAM and INVALID_FLIP preserve A/B/U; and the accepted POST state is the DELAYED oracle state.

## 4. Prospective surface and protocol counterbalances

The following are *design obligations to test before accepting generated rows*, not presently verified properties:

1. **Truth polarity / role:** distribute positive and explicit negative assertions for A/C/U across both synthetic domains and paraphrase families. Each of A/B/U is queried with both true and false expected answers wherever logically possible. Do not make one fixed word, predicate or entity predict the answer.
2. **Authority:** within each case, the VALID_FLIP and INVALID_FLIP events have identical A assertion content, logical event time and target; only source authentication / provenance differs. VALID_SHAM and VALID_FLIP are both authenticated with the same update-site structure and logical time. Across families, authentication *surface tokens* and source names rotate; authentication remains an explicit, necessary feature, not something to conceal. Source name must not by itself encode the gold label.
3. **Time and order:** timestamp 10 versus 20 semantics remain fixed in the primary arm; model-visible *presentation order* is crossed with arms and truth configurations, not with final labels. Include separately identified stale-authenticated and later-presented-untrusted controls from D1; never change event-time semantics to produce a desired label. The task cannot be solved by a last-mentioned-assertion rule.
4. **Lexical contradiction:** alternate affirmative/negative statement forms only when their meanings are unambiguous. Use opposite-polarity A records in both a valid and invalid update arm; the presence of contradiction words alone must not determine accepted truth.
5. **Logical dependency:** express `B := A AND C` in equivalent clear wordings only after defining a semantics-preserving grammar. For `C=F`, a valid A flip must not flip B. The C premise must be explicitly available to the model and not directly exposed as a gold B label.
6. **Query and delay:** vary surface positions of A/B/U queries and neutral distractors without moving or deleting authoritative facts or the rule. DELAYED distractors have no new admissible A/C/U records; a misleading fact-changing distractor is invalid, not a fourth intervention arm.
7. **No label leakage:** provenance and oracle intermediates (e.g., `target_status`, `final_label_id`, `initial_*`, `update_arm`, hidden `case_family_id`) belong to metadata, not model-facing text. Model input can contain the rule, actual certified event contents, logical timestamp, source validity evidence and query; not an answer key or data split marker.

Positive/negative assertion phrases and authenticated/untrusted cues are task-relevant information; demanding that *all* single-token correlations vanish is neither possible nor the validity criterion. Instead predeclare concrete shortcut attacks (Section 6) and require matched contrasts in which each suspected shortcut yields the **same cue but a different correct answer**, or vice versa. Template, provenance, timestamp and query position must be audited jointly, not asserted balanced from intent alone.

## 5. Identity, split and serialization contracts

Proposed structured family schema (metadata, not yet rows):

`case_family_id`, `split_group_id`, `case_id`, `surface_domain`, `renderer_family_id`, `lexicon_namespace_id`, `semantic_world_id`, `initial_A`, `initial_C`, `initial_U`, `derivation_rule_id`, `oracle_version`, `generator_version`, `family_variant_linkage`.

For future individual variants retain D1 fields plus typed event IDs, authenticated source and provenance IDs, logical event times, exact text payload, event-presentation order, query and readout IDs, distractor identity, oracle intermediate A/C for each derived B label, and checksums. `PRE` inputs must exclude *all* later event information, not merely disregard it at label time. `DELAYED` must not mutate the accepted oracle state. Do not place metadata-only truth or split fields inside a future model prompt.

`verify_split_isolation` from D1 is a structural validator; it cannot establish that two different IDs reuse the same wording/semantic micro-world. A future **independent content-identity checker** must detect duplicate canonical event structures, entity tuples, paraphrase families and near-duplicate prompts across splits. A separate held-out-renderer assessment needs renderer families that are not used during model training/selection. No assignment is made in this document; future split allocations and leakage thresholds must be frozen **before examining model outcomes**.

## 6. Required pre-execution shortcut and feasibility gates

A later generator proposal is not eligible for model execution until independently checked, *without using model responses*, for:

- `G1 LOGIC`: every variant's expected A/B/U truth equals the frozen D1 oracle; all 8 strata and all three arms remain represented; paired transitions and C=F blocking pass.
- `G2 FAMILY`: no case/group/family crosses splits, including paraphrases, reversed truth cases, alternative source attestations, and permutations; duplicate/near-duplicate surface audits and source-namespace disjointness are reported.
- `G3 SURFACE`: template/lexical/polarity/authority/time/query/distractor counterbalance tables include no deterministic unintended gold-answer cue. Expected structurally skewed B labels must be acknowledged rather than resampled into artificial balance.
- `G4 ATTACKS`: report label-only majority, t1-assertion-only, authentication-only, last-mentioned-record, question-target-only, rule-agnostic, and source-ID/template-only shortcut predictions with their **conditional** confusion patterns. None alone may be mistaken for the intended same-case revision mechanism; a weak shortcut does not prove absence of all leakage.
- `G5 SERIALIZATION`: for the *actual future pinned tokenizer*, verify the rule, source evidence/authentication, event time, updated assertion and query all survive input serialization with no decisive truncation. Token length, position and context size are `UNVERIFIED` now.
- `G6 CAPACITY`: count genuinely distinct families by truth stratum, surface renderer and split; demonstrate that the sampling frame permits intended independent test units. If only eight logical worlds with name substitutions are feasible, bound claims to surface robustness and stop a broad confirmatory claim.
- `G7 PROVENANCE`: before any data materialization, separately freeze deterministic generator code scope, renderer source/version, seed, manifests, exact schema, independent validation, and later non-leaking evaluation protocol. Artifact hashes or PASS statuses may only be claimed after execution.

These gates do **not** automatically authorize data materialization, tokenizer/model loading, training, evaluation, GPU, Kaggle, recurrent-state interventions or claim promotion. Failure is a blocker requiring a fresh prospective design decision, not threshold broadening after inspecting model answers.

## 7. Decision and next bounded step

**Static design outcome:** D1 truth semantics and D2 exact matrix permit a family-grouped, shortcut-controlled cohort *on paper*, but independent lexical capacity, data-generation quality, real split feasibility and tokenizer safety are not demonstrated. Only a bounded **CPU-only structural renderer/generator contract + in-memory, no-artifact feasibility validator design** is eligible for the next separately scoped step. It must not fabricate case instances or train/evaluate a model until distinct authority for data materialization exists.

`D2_COHORT_DESIGN=PROPOSED / REAL_COHORT_FEASIBILITY=NOT_ESTABLISHED / RFR_SCIENCE_RESULT=NONE / MODEL=OFF / GPU=OFF / PHASE3B=BLOCKED_UNCHANGED`
