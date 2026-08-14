# P3-W6-F2-P3 Human Review Preparation Spec

## Decision

Previous preparation decision:

`P3W6F2P3_HUMAN_REVIEW_PREPARATION_READY_NO_CODE_CHANGE`

Augmented preparation decision:

`P3W6F2P3_HUMAN_REVIEW_PREPARATION_AUGMENTED_READY_NO_CODE_CHANGE`

This is a human-review operating workflow specification only. It does not begin the 119-pair F2 review, create a real WIP ledger, create an XLSX workbook, record human decisions, finalize artifacts, remediate rows, mutate data, train, evaluate, commit, or push.

The existing `scripts/reason_router_p3w6f2_manual_review.py` infrastructure is sufficient for authority loading, pair presentation, strict external WIP validation, deterministic decision derivation, note enforcement, correction by replacement, resume, and final Level-1 artifact generation. No production code change is required before actual P3 execution, provided the reviewer follows the confirmation, shorthand, paper-preservation, and auxiliary-workbook boundary below.

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

## L. ChatGPT Assistance Boundary

Allowed assistance:

- format the three members clearly
- explain enum meanings
- point out literal textual differences
- identify which words changed
- answer questions about this protocol

Not allowed:

- automatically write `V/I/U` or `C/M/N/U` into WIP
- decide semantic validity for the reviewer
- bulk approve structurally similar pairs
- infer remaining decisions from P0 cluster patterns
- use root-cause class as a substitute for human judgment

The final selection must come from the human reviewer.

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

- `Review`: one row per F2 pair, exactly 119 data rows. Columns include `review_index`, `pair_id`, the three final labels, the three claims, the three evidence fields, the three automatic grammar statuses, `automatic_root_cause_class`, the four human classification fields, `human_notes`, `derived_authority_decision`, and `review_status`. Human fields must start blank. Do not pre-label.
- `Authority`: exact 27 immutable source fields from frozen source authority. This sheet is not an annotation sheet. Use a visually protected/read-only convention where practical. Do not normalize, rewrite, or reparse source values.
- `Legend`: Korean reviewer instructions, shorthand, and exact frozen enum meanings. Semantic shorthand is `V/I/U`; grammar shorthand is `C/M/N/U`.
- `Summary`: human-facing convenience layout for total, reviewed, remaining, `V/V/V/C`, `V/V/V/M`, semantic conflict, insufficient evidence, no reproducible defect, and derived-decision distribution. This summary is not final scientific aggregate authority.

Spreadsheet input validation may use dropdowns:

- semantic fields: `V`, `I`, `U`, blank
- grammar field: `C`, `M`, `N`, `U`, blank
- `review_status`: `UNREVIEWED`, `READY_TO_IMPORT`, `IMPORTED`

Do not permit arbitrary alternate semantic labels. An Excel formula may display a tentative `derived_authority_decision` for reviewer convenience, but it must not become final authority. The existing Python compatibility matrix must recompute the final decision before WIP import/write.

## P. Excel To WIP Authority Boundary

Spreadsheet cell entry is not an authoritative WIP record.

Future import boundary:

1. XLSX human selections
2. deterministic shorthand expansion
3. frozen `source_record_sha256` verification
4. enum validation
5. compatibility matrix derivation
6. notes requirement validation
7. explicit reviewer confirmation/import
8. existing external WIP JSONL

Only after this validation/import does a decision become review authority. Manual CLI `record` transfer is sufficient for the current preparation authority; no importer is required now.

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

Do not add new mandatory human judgments solely for a hypothetical paper. Paper analysis must derive from the existing integrity review where possible.

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
- WIP creation in this stage
- XLSX workbook creation in this stage
- finalization
- remediation
- data mutation
- training or evaluation
- LLM classification
- heuristic pre-labeling
- grammar-based semantic autofill
- root-cause based autofill
- bulk `V/V/V` assignment
- bulk grammar assignment
- automatic recommended semantic decisions
- automatic notes generation
- Excel formula output as final authority
- paper-analysis hypotheses influencing review labels

## W. Readiness Decision

Previous readiness decision: `P3W6F2P3_HUMAN_REVIEW_PREPARATION_READY_NO_CODE_CHANGE`.

Revised readiness decision: `P3W6F2P3_HUMAN_REVIEW_PREPARATION_AUGMENTED_READY_NO_CODE_CHANGE`.

Rationale:

- The existing CLI loads frozen authority from Git objects and verifies the 119 F2 pairs and 357 members.
- `show`/`next --show` already present one pair at a time with labels, claims, evidence, grammar status, reason codes, diff summaries, diagnostic root-cause class/evidence, source hash, and existing WIP status.
- `record` mechanically derives `human_authority_decision`, validates full enums, enforces required notes, requires explicit reviewer ID, and requires `--ack-complete-triple-reviewed`.
- `status` and `next` fail closed on invalid WIP.
- Corrections replace existing records rather than duplicating pair IDs.
- WIP paths inside the repository are rejected.
- The paper-candidate registry is documentation/research preservation only and contains no human decisions.
- XLSX review is an auxiliary human workspace only; the external WIP JSONL remains the only writable review ledger.

Known UX limitation accepted as operating procedure rather than code change:

- Native shorthand input and an interactive `SAVE / EDIT / CANCEL` prompt are not built into the CLI. For the real review, shorthand expansion and preview must be done deterministically before executing `record`; executing `record` is the explicit `SAVE`.
- Native XLSX import is not built into the CLI. Manual CLI `record` transfer remains sufficient unless a future reviewed authority approves an importer.

No real review, WIP creation, XLSX creation, finalization, remediation, training, or evaluation was begun by this spec.
