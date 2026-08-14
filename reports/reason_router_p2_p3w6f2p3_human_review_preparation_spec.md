# P3-W6-F2-P3 Human Review Preparation Spec

## Decision

Previous preparation decision:

`P3W6F2P3_HUMAN_REVIEW_PREPARATION_READY_NO_CODE_CHANGE`

Augmented preparation decision:

`P3W6F2P3_HUMAN_REVIEW_PREPARATION_AUGMENTED_READY_NO_CODE_CHANGE`

AI-assisted amendment decision:

`P3W6F2P3_AI_ASSISTED_HUMAN_REVIEW_AMENDMENT_READY_NO_CODE_CHANGE`

This is a human-review operating workflow specification only. It does not begin the 119-pair F2 review, create a real WIP ledger, create an XLSX workbook, record human decisions, finalize artifacts, remediate rows, mutate data, train, evaluate, commit, or push.

This amendment permits a future non-authoritative AI prescreen layer for workload reduction. The layer may suggest per-pair semantic and grammar values, but it is not annotation authority, WIP authority, remediation authority, controlled-data authority, or a substitute for explicit human row-level action. The existing `scripts/reason_router_p3w6f2_manual_review.py` infrastructure is sufficient for authority loading, pair presentation, strict external WIP validation, deterministic decision derivation, note enforcement, correction by replacement, resume, and final Level-1 artifact generation. No production code change is required before actual P3 execution, provided the reviewer follows the confirmation, shorthand, paper-preservation, AI-prescreen, and auxiliary-workbook boundary below.

## Authority Checked

- Closed P2/P3-W6-F2-P2 HEAD: `155ae38e18ce1f632b596f64056e75cece3e245a`
- P3-W6-F2-P1 execution spec: `reports/reason_router_p2_p3w6f2p1_manual_review_execution_spec.md`
- P3-W6-F2-P1 manifest: `reports/reason_router_p2_p3w6f2p1_manual_review_execution_manifest.json`
- P3-W5 separate remediation spec: `reports/reason_router_p2_p3w5_separate_remediation_spec.md`
- P3-W5 manifest: `reports/reason_router_p2_p3w5_separate_remediation_manifest.json`
- P3-W4 F2 source template: `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_f2_manual_review.csv`
- P3-W4 pair authority: `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_pairs.jsonl`
- P3-W4 summary authority: `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_summary.json`

The P3-W5/P3-W6-F2-P1 definitions are preserved. This spec fixes the reviewer-facing operating procedure and adds paper-candidate plus XLSX auxiliary-review-surface contracts only.

## A. Reviewer Responsibilities

The reviewer must inspect one complete F2 pair at a time. A review unit is one `pair_id` containing canonical `none` REFUTE, `paraphrase` REFUTE, and `polarity_flip` SUPPORT members together.

For each pair, the reviewer directly decides exactly four classifications:

- `human_canonical_semantics`
- `human_paraphrase_semantics`
- `human_polarity_flip_semantics`
- `human_grammar_validity`

The reviewer may optionally write `human_notes`, and must write nonempty notes when required by the notes policy. The reviewer does not choose `human_authority_decision`; the frozen compatibility matrix derives it mechanically.

## B. Compact Shorthand

Human input shorthand:

- Semantic: `V = VALID`, `I = INVALID`, `U = UNCLEAR`
- Grammar: `C = CANONICAL_ONLY_DEFECT`, `M = MULTI_MEMBER_DEFECT`, `N = NO_REPRODUCIBLE_DEFECT`, `U = UNCLEAR`

Example: `V V V M` expands deterministically to:

- `VALID`
- `VALID`
- `VALID`
- `MULTI_MEMBER_DEFECT`

The shorthand is only an input alias. It must not be interpreted semantically, inferred from displayed content, inferred from root-cause class, or prefilled. The existing CLI accepts the full enum values; during the real run the operator may use a deterministic local expansion table before invoking `record`, but the WIP ledger must store only full enum strings.

## C. One-Pair Review Card Layout

Each pair view must display, in this order:

1. `pair_id`
2. `CANONICAL`
   - final label
   - claim
   - evidence
   - grammar status
   - reason codes
   - claim diff summary
   - evidence diff summary
3. `PARAPHRASE`
   - same fields
4. `POLARITY_FLIP`
   - same fields
5. `DIAGNOSTIC ONLY`
   - `automatic_root_cause_class`
   - `automatic_evidence`
6. Reviewer prompt:
   - `canonical semantics [V/I/U]:`
   - `paraphrase semantics [V/I/U]:`
   - `polarity semantics [V/I/U]:`
   - `grammar [C/M/N/U]:`
7. Notes prompt only when required after deterministic decision derivation.

No recommended semantic decision may be displayed. No `V/I/U/C/M/N/U` value may be preselected.

## D. Korean Reviewer Instructions

리뷰 단위는 하나의 `pair_id`에 속한 세 문장 묶음입니다. 반드시 canonical, paraphrase, polarity_flip 세 멤버를 함께 본 뒤 판단합니다. 자동 root-cause 정보는 참고용 진단 정보일 뿐이며, 사람이 내릴 semantic/grammar 판단을 대신하지 않습니다.

Semantic 판단:

- `VALID`: 표면적인 문법 결함이 있더라도, 해당 멤버가 controlled example에 필요한 의도된 사실/의미 관계를 보존한다고 판단되는 경우입니다.
- `INVALID`: 해당 멤버가 의도된 의미 관계를 바꾸거나, 훼손하거나, 모순시키거나, 보존하지 못한다고 판단되는 경우입니다.
- `UNCLEAR`: 의미 보존 여부를 충분한 확신으로 판단할 수 없는 경우입니다.

Grammar 판단:

- `CANONICAL_ONLY_DEFECT`: 관련 재현 가능한 문법 결함이 canonical 멤버에만 나타난다고 판단되는 경우입니다.
- `MULTI_MEMBER_DEFECT`: 관련 결함이 canonical과 하나 이상의 derivative 멤버에 함께 재현된다고 판단되는 경우입니다.
- `NO_REPRODUCIBLE_DEFECT`: 표시된 세 멤버에서 이전에 진단된 결함을 재현할 수 없다고 판단되는 경우입니다.
- `UNCLEAR`: 문법 결함의 범위를 충분한 확신으로 판단할 수 없는 경우입니다.

문법 `FAIL`은 자동으로 semantic `INVALID`가 아니며, 문법 `PASS`도 자동으로 semantic `VALID`가 아닙니다. 최종 선택은 사람이 해야 합니다.

## E. Notes Policy

The CLI derives `human_authority_decision` after the four classifications are known. Notes are required when any of these is true:

- any semantic field is `UNCLEAR`
- grammar is `UNCLEAR`
- derived decision is `SEMANTIC_CONFLICT`
- derived decision is `INSUFFICIENT_EVIDENCE_KEEP_BLOCKED`
- derived decision is `NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED`

If notes are required and blank after trim, the existing infrastructure fails closed with `MISSING_REQUIRED_NOTES`. Optional notes remain allowed for `CANONICAL_TEXTUAL_REPAIR_CANDIDATE` and `CANONICAL_REGENERATION_REQUIRED`.

AI rationale does not satisfy `human_notes`. `ai_prescreen_rationale` and `human_notes` are different fields; only human-authored notes satisfy the frozen notes policy.

## F. Save / Edit / Cancel Flow

Before any real WIP write, the reviewer must see:

- `pair_id`
- entered shorthand
- expanded enum values
- mechanically derived decision
- notes, if any

Then the reviewer must explicitly choose:

- `SAVE`: run the `record` command with full enum values and `--ack-complete-triple-reviewed`
- `EDIT`: return to the four prompts and recompute the preview
- `CANCEL`: write nothing

The current CLI writes only when `record` is executed. Therefore, in actual operation, pressing Enter on the reviewed `record` command is the explicit `SAVE`. The operator must not execute `record` until the preview has been checked.

## G. Correction Flow

For an already reviewed pair:

1. Run `show --pair-id <PAIR_ID> --wip-path <EXTERNAL_WIP>` to display the source card and existing WIP decision record.
2. Prepare proposed replacement values using the same preview flow.
3. Show old decision, proposed four expanded fields, proposed derived decision, and notes.
4. Execute `record` only after explicit `SAVE`.

The existing infrastructure replaces the pair record instead of appending a duplicate, revalidates the WIP before replacement, and updates `reviewed_at_utc` when the replacement record is created.

## H. Resume Flow

Normal loop:

1. `status`
2. `next`
3. `next --show` or `show --pair-id`
4. human entry
5. preview and `SAVE / EDIT / CANCEL`
6. `record`
7. `next`

Restart loop:

1. `status --wip-path <EXTERNAL_WIP>`
2. If status fails, stop and fix the invalid WIP ledger before reviewing more pairs.
3. `next --show --wip-path <EXTERNAL_WIP>`

The infrastructure strictly validates WIP before `status`, `next`, and `record`. No reviewed pair should be skipped when WIP is invalid.

## I. reviewer_id Convention

Use a stable, explicit, nonempty, trimmed pseudonymous identifier supplied on every `record` command, for example:

`--reviewer-id f2_primary_reviewer_01`

Do not hard-code private personal data into source or scripts. The same reviewer ID should remain identical across all 119 reviews unless a documented correction is justified.

## J. External WIP Location

The real WIP ledger must be outside the repository, persistent across sessions, outside `reports/`, not authority data, easy to back up, and never committed.

Recommended Windows pattern:

`C:\Users\Home1\ContraMambaReview\P3W6F2\p3w6f2_review_wip.jsonl`

This stage must not create the file. The directory/file should be created only at real P3 execution time.

## K. Backup Policy

After each batch, copy the external WIP ledger to a timestamped backup outside the repository, for example:

`C:\Users\Home1\ContraMambaReview\P3W6F2\backups\p3w6f2_review_wip_YYYYMMDD_HHMMSS.jsonl`

Backups must not be added to Git and must not be copied under the repository root. Before resuming from a backup, restore it to the canonical external WIP path and run `status` so strict validation checks it before any new review.

## L. AI-Assisted Prescreen Boundary

This amendment explicitly supersedes the prior ChatGPT assistance boundary that prohibited AI from proposing semantic validity or grammar scope. The conflict is narrow: the frozen preparation text allowed formatting, enum explanation, and literal-difference support, but disallowed AI semantic/grammar suggestions; the new human workload policy permits those suggestions only as a non-authoritative diagnostic prescreen.

New auxiliary protocol version:

`P3W6F2_AI_ASSISTED_PRESCREEN_V1`

The authoritative human WIP protocol remains:

`P3W5_F2_MANUAL_REVIEW_V1`

AI prescreen role:

`NON_AUTHORITATIVE_DIAGNOSTIC_PRESCREEN`

For each pair, a future AI prescreen may inspect the complete triple (`canonical`, `paraphrase`, `polarity_flip`) and propose:

- `ai_canonical_semantics_suggestion`
- `ai_paraphrase_semantics_suggestion`
- `ai_polarity_flip_semantics_suggestion`
- `ai_grammar_validity_suggestion`
- `ai_triage_status`
- `ai_prescreen_rationale`

Allowed suggestion shorthands:

- Semantic: `V`, `I`, `U`
- Grammar: `C`, `M`, `N`, `U`
- Triage: `CLEAR_SUGGESTION`, `HUMAN_REVIEW_REQUIRED`

Do not introduce numeric confidence in this version. The intended distinction is binary triage, not calibrated scoring.

`CLEAR_SUGGESTION` is allowed only when the AI can identify a straightforward semantic and grammar relation from the displayed complete triple without material ambiguity. A representative clear case is one where the claim expresses event X, canonical/paraphrase evidence unambiguously expresses negated X despite a directly observable surface grammar error, polarity flip restores affirmative X, and the grammar defect scope is directly observable.

`HUMAN_REVIEW_REQUIRED` is required when any material ambiguity exists, including uncertain semantic preservation, transformations that changed more than surface form, multiple plausible interpretations, questionable label/text relation, uncertain grammar-defect scope, inconsistency between diagnostics and literal text, any case where the AI would otherwise choose `U`, or unusual structure relative to the dominant F2 pattern. The AI should preferentially escalate uncertain cases rather than force a label.

AI suggestions are not human judgments, annotations, WIP records, remediation decisions, controlled-data authority, reviewer identity, or permission to skip review. AI fields must not be written into human fields before a human action, must not directly enter the external WIP JSONL, and must not be retrospectively edited to match the human final judgment.

Every one of the 119 pairs still requires exactly one explicit human row-level action:

- `CONFIRM`: the human reviewer inspected the complete triple and explicitly accepts the AI suggestion as their own final human judgment.
- `OVERRIDE`: the human reviewer inspected the complete triple and supplies one or more different final human classifications.

No `AUTO_ACCEPT`, blanket confirmation, batch confirmation, cluster-level approval, or approval by structural similarity is allowed. Action on one pair never approves another pair. If the human independently chooses the same values for an escalated case, `CONFIRM` remains allowed only after pair-level inspection.

For `CONFIRM`: AI proposed values -> human inspects complete triple -> explicit `CONFIRM` -> values become human-selected final classifications -> frozen compatibility matrix derives `human_authority_decision` -> notes rules are checked -> WIP may be written.

For `OVERRIDE`: AI proposed values -> human inspects complete triple -> explicit `OVERRIDE` -> human supplies final values -> frozen compatibility matrix derives `human_authority_decision` -> notes rules are checked -> WIP may be written.

## M. Recommended Batch Size

Use 10-20 pairs per session, with `status` checkpoints between batches. Accuracy is more important than speed. The WIP ledger supports stopping after any pair, because each saved record is a complete pair-level JSONL record.

## N. Paper-Candidate Registry

Reserved registry path:

`reports/reason_router_p2_p3w6f2_f2_paper_candidate_registry.json`

Purpose: preserve the 119-pair / 357-member F2 cohort as a GitHub-tracked machine-readable research asset after ContraMamba integrity remediation continues. The registry is not human annotation authority, not WIP, not final review output, and not a scientific result.

Required registry content:

- `candidate_id = F2_TRANSFORMATION_INTEGRITY_ANALYSIS_V1`
- `status = PAPER_CANDIDATE_PRESERVE_FOR_POST_REVIEW_ANALYSIS`
- `origin_stage = P3-W6-F2`
- `cohort.pair_count = 119`
- `cohort.member_count = 357`
- `cohort.member_roles = ["canonical", "paraphrase", "polarity_flip"]`

Research questions:

- RQ1: canonical generation defect가 paraphrase와 polarity-derived member에 어떤 형태로 전파되는가?
- RQ2: surface grammatical defect와 semantic validity contamination은 어떻게 분리되는가?
- RQ3: 어떤 pair가 canonical-only textual repair 후보이고, 어떤 pair가 regeneration을 요구하는가?
- RQ4: automatic integrity diagnostics와 human semantic/grammar judgment가 어느 정도 일치하는가?

Planned analysis axes:

- human semantic validity by member
- human grammar-defect scope
- derived authority decision
- automatic root-cause class
- automatic grammar statuses
- predicate / lexical signature
- automatic-vs-human agreement
- repairability category
- decision distribution
- notes/error taxonomy

Scientific nonclaims:

- no paper conclusion exists before human review
- structural similarity does not authorize bulk labeling
- 119 pairs are a paper candidate cohort, not yet a completed dataset study
- ContraMamba remediation result and independent paper claim are separate

## O. XLSX Auxiliary Review Surface

Reserved working artifact name:

`p3w6f2_f2_human_review_workbook.xlsx`

The workbook is a human-facing auxiliary review surface only. It is never frozen authority, final annotation authority, Git canonical authority, production data, final review result, or a replacement for the external WIP JSONL. The workbook must not be created in this preparation stage.

Writable authority remains the existing external WIP JSONL produced through `scripts/reason_router_p3w6f2_manual_review.py`.

Minimum workbook sheets:

- `Review`: one row per F2 pair, exactly 119 data rows. Recommended visible ordering is `review_index`, `pair_id`, source/context fields for the three members, AI prescreen fields, human final fields, `derived_authority_decision`, and `review_status`. Initial pre-prescreen state may keep all AI fields blank. After a future prescreen pass, AI fields may be populated, but `human_review_action` and all human final fields must remain blank until the human acts.
- `Authority`: exact 27 immutable source fields from frozen source authority. This sheet is not an annotation sheet. Use a visually protected/read-only convention where practical. Do not normalize, rewrite, or reparse source values.
- `Legend`: Korean reviewer instructions, shorthand, and exact frozen enum meanings. Semantic shorthand is `V/I/U`; grammar shorthand is `C/M/N/U`.
- `Summary`: human-facing convenience layout for total, reviewed, remaining, `V/V/V/C`, `V/V/V/M`, semantic conflict, insufficient evidence, no reproducible defect, and derived-decision distribution. This summary is not final scientific aggregate authority.

`Review` sheet source/context fields should include the existing display fields for canonical, paraphrase, and polarity flip: final labels, claims, evidence, automatic grammar statuses, reason codes, claim diff summaries, evidence diff summaries, `automatic_root_cause_class`, and `automatic_evidence`.

`Review` sheet AI prescreen fields are auxiliary diagnostic fields:

- `ai_canonical_semantics_suggestion`
- `ai_paraphrase_semantics_suggestion`
- `ai_polarity_flip_semantics_suggestion`
- `ai_grammar_validity_suggestion`
- `ai_triage_status`
- `ai_prescreen_rationale`

`Review` sheet human final fields are authority-intent fields that require explicit human action:

- `human_review_action`
- `human_canonical_semantics`
- `human_paraphrase_semantics`
- `human_polarity_flip_semantics`
- `human_grammar_validity`
- `human_notes`

The existing six human WIP fields remain human authority fields:

- `human_canonical_semantics`
- `human_paraphrase_semantics`
- `human_polarity_flip_semantics`
- `human_grammar_validity`
- `human_authority_decision`
- `human_notes`

AI diagnostic fields must not be stored as if they were human fields. The current production WIP schema does not need to contain the AI diagnostic fields. Prefer keeping AI diagnostics workbook-side or analysis-side only until a later authority explicitly changes the WIP schema.

Spreadsheet input validation may use dropdowns:

- semantic fields: `V`, `I`, `U`, blank
- grammar field: `C`, `M`, `N`, `U`, blank
- `ai_triage_status`: `CLEAR_SUGGESTION`, `HUMAN_REVIEW_REQUIRED`, blank
- `human_review_action`: `CONFIRM`, `OVERRIDE`, blank
- `review_status`: `UNREVIEWED`, `READY_TO_IMPORT`, `IMPORTED`

Do not permit arbitrary alternate semantic labels. An Excel formula may display a tentative `derived_authority_decision` for reviewer convenience, but it must not become final authority. The existing Python compatibility matrix must recompute the final decision before WIP import/write.

## P. Excel To WIP Authority Boundary

Spreadsheet cell entry is not an authoritative WIP record.

Future import boundary:

1. XLSX row contains complete source triple and any auxiliary AI suggestion.
2. Human reviewer inspects the complete triple.
3. Human reviewer selects exactly one `human_review_action`: `CONFIRM` or `OVERRIDE`.
4. Final human selections are resolved from explicit `CONFIRM` or `OVERRIDE`.
5. Deterministic shorthand expansion.
6. Frozen `source_record_sha256` verification.
7. Enum validation.
8. Compatibility matrix derivation.
9. Notes requirement validation.
10. Explicit reviewer confirmation/import.
11. Existing external WIP JSONL.

Only after this validation/import does a decision become review authority. Manual CLI `record` transfer is sufficient for the current preparation authority; no importer is required now.

A pair is human-reviewed only if the complete triple was available to the reviewer, `human_review_action` is `CONFIRM` or `OVERRIDE`, the four final human classification fields exist, the compatibility decision is valid, the notes requirement is satisfied, and the authoritative WIP record is successfully written. AI prescreen completion alone never increments reviewed count.

## Q. No Dual Source Of Truth

If the workbook is used:

- workbook = human workspace
- WIP JSONL = authoritative validated execution ledger

Excel and WIP must not diverge as two annotation authorities. Any imported workbook row must carry a status indicating whether it has entered WIP. Never infer WIP completion merely because the workbook has 119 filled rows.

## R. Workbook And WIP Location

Recommended external Windows working directory:

`C:\Users\Home1\ContraMambaReview\P3W6F2\`

Recommended files under that directory during future execution:

- `p3w6f2_f2_human_review_workbook.xlsx`
- `p3w6f2_review_wip.jsonl`
- `backups\`

The actual path may be adjusted by the user. Do not place active WIP under the repository root. The workbook may be backed up separately, but ordinary per-edit XLSX commits to Git are discouraged because XLSX is binary and not useful as review-authority diff history.

## S. Paper Analysis Field Preservation

The eventual validated review output should preserve enough fields for later paper analysis without re-reviewing 119 pairs. At minimum, preserve or recover:

- `pair_id`
- three member semantic judgments
- grammar scope
- derived authority decision
- automatic root-cause class
- three automatic grammar statuses
- source reason codes
- predicate/lexical signature if derivable mechanically
- review timestamp
- reviewer ID
- human notes

If a future AI prescreen is scientifically analyzed as an AI-human agreement signal, preserve or recover these additional non-authoritative AI fields separately from human fields:

- AI-suggested three semantic judgments
- AI-suggested grammar scope
- `ai_triage_status`
- AI-human agreement/disagreement
- escalated/non-escalated case status
- `human_review_action`
- AI prescreen rationale

Do not add new mandatory human judgments solely for a hypothetical paper. Paper analysis must derive from the existing integrity review where possible.

Future paper analyses may estimate AI-human agreement rate, disagreement patterns, whether ambiguous cases cluster by predicate/transformation, and the precision of `CLEAR_SUGGESTION` triage. These are future research questions, not current findings.

AI prescreen must be generated before viewing human final labels for that pair if it will later be analyzed scientifically as an AI-human agreement signal. Stored AI suggestions must not be retrospectively edited to match human judgments. If AI suggestions are regenerated later, preserve version/provenance and do not silently overwrite the original prescreen.

Minimal future AI prescreen provenance contract:

- `ai_prescreen_protocol_version`
- `ai_prescreen_model_or_system_id`
- `ai_prescreen_created_at_utc`
- `source_record_sha256`
- `pair_id`

The specific ChatGPT/model identifier must be recorded at execution time. Do not fabricate it in this preparation amendment, and do not claim model reproducibility beyond what is actually recorded.

## T. ContraMamba Track Vs Paper Track

Track A is ContraMamba F2 integrity closure. Track B is future independent transformation-integrity analysis / paper candidate.

P3 human decisions may be reused as research observations. However:

- F2 repair decisions remain governed by ContraMamba authority
- future paper hypotheses cannot influence current human labels
- no retrospective relabeling may be done to make paper categories cleaner
- remediation conclusions and paper claims must remain separately reviewed

## U. Exact Execution Sequence For P3

Real execution should use the external WIP path and stable reviewer ID consistently:

```powershell
$WIP = "C:\Users\Home1\ContraMambaReview\P3W6F2\p3w6f2_review_wip.jsonl"
$RID = "f2_primary_reviewer_01"
python scripts\reason_router_p3w6f2_manual_review.py status --wip-path $WIP
python scripts\reason_router_p3w6f2_manual_review.py next --show --wip-path $WIP
```

For each pair:

1. Read the displayed card or auxiliary workbook row.
2. Enter shorthand on scratch only, for example `V V V M`.
3. Expand deterministically to full enums.
4. Derive the decision from the frozen matrix.
5. If notes are required, write nonempty notes.
6. Show the preview and choose `SAVE / EDIT / CANCEL`.
7. On `SAVE`, run:

```powershell
python scripts\reason_router_p3w6f2_manual_review.py record `
  --wip-path $WIP `
  --pair-id <PAIR_ID> `
  --reviewer-id $RID `
  --canonical-semantics <VALID|INVALID|UNCLEAR> `
  --paraphrase-semantics <VALID|INVALID|UNCLEAR> `
  --polarity-flip-semantics <VALID|INVALID|UNCLEAR> `
  --grammar-validity <CANONICAL_ONLY_DEFECT|MULTI_MEMBER_DEFECT|NO_REPRODUCIBLE_DEFECT|UNCLEAR> `
  --notes "<NOTES_OR_EMPTY_STRING>" `
  --ack-complete-triple-reviewed
```

After each batch:

```powershell
python scripts\reason_router_p3w6f2_manual_review.py status --wip-path $WIP
```

Only after all 119 pairs have valid records under future explicit execution authority:

```powershell
python scripts\reason_router_p3w6f2_manual_review.py finalize --wip-path $WIP
```

Finalization is not part of this preparation stage.

## V. What Must Not Happen Automatically

Do not automate:

- real F2 annotation in this stage
- AI prescreen execution in this stage
- WIP creation in this stage
- XLSX workbook creation in this stage
- finalization
- remediation
- data mutation
- training or evaluation
- real WIP population
- human field population
- LLM classification execution in this stage
- heuristic pre-labeling
- grammar-based semantic autofill
- root-cause based autofill
- bulk `V/V/V` assignment
- bulk grammar assignment
- automatic acceptance of AI labels
- bulk semantic labeling
- cluster-level approval
- skipping human review for high-confidence AI cases
- AI directly populating WIP
- AI becoming `reviewer_id`
- automatic notes generation
- Excel formula output as final authority
- paper-analysis hypotheses influencing review labels

## W. Readiness Decision

Previous readiness decision: `P3W6F2P3_HUMAN_REVIEW_PREPARATION_READY_NO_CODE_CHANGE`.

Revised readiness decision: `P3W6F2P3_HUMAN_REVIEW_PREPARATION_AUGMENTED_READY_NO_CODE_CHANGE`.

AI-assisted amendment decision: `P3W6F2P3_AI_ASSISTED_HUMAN_REVIEW_AMENDMENT_READY_NO_CODE_CHANGE`.

Rationale:

- The existing CLI loads frozen authority from Git objects and verifies the 119 F2 pairs and 357 members.
- `show`/`next --show` already present one pair at a time with labels, claims, evidence, grammar status, reason codes, diff summaries, diagnostic root-cause class/evidence, source hash, and existing WIP status.
- `record` mechanically derives `human_authority_decision`, validates full enums, enforces required notes, requires explicit reviewer ID, and requires `--ack-complete-triple-reviewed`.
- `status` and `next` fail closed on invalid WIP.
- Corrections replace existing records rather than duplicating pair IDs.
- WIP paths inside the repository are rejected.
- The paper-candidate registry is documentation/research preservation only and contains no human decisions.
- XLSX review is an auxiliary human workspace only; the external WIP JSONL remains the only writable review ledger.
- AI prescreen suggestions remain auxiliary diagnostic fields and do not enter the production WIP schema.
- `CONFIRM` and `OVERRIDE` both require explicit per-pair human inspection of the complete triple.
- No batch confirmation or automatic acceptance is authorized.
- `scripts/reason_router_p3w6f2_manual_review.py` remains unchanged.

Known UX limitation accepted as operating procedure rather than code change:

- Native shorthand input and an interactive `SAVE / EDIT / CANCEL` prompt are not built into the CLI. For the real review, shorthand expansion and preview must be done deterministically before executing `record`; executing `record` is the explicit `SAVE`.
- Native XLSX import is not built into the CLI. Manual CLI `record` transfer remains sufficient unless a future reviewed authority approves an importer.
- Native AI-prescreen import/confirmation expansion is not built into the CLI. A helper/importer may be considered later, but it is not necessary for this authority amendment.

No real review, AI prescreen execution, WIP creation, XLSX creation, finalization, remediation, data mutation, training, or evaluation was begun by this spec.
