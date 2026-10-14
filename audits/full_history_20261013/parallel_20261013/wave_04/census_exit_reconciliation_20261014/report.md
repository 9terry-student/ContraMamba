# Final historical census evidence reconciliation — 2026-10-14

Recommendation: **NOT_READY_FOR_CENSUS_CLOSURE_REVIEW**.

The complete original 222-ID inventory is reconciled. Evidence collection still includes unfinished required local documentary work; its completion is not established by classifying the missing originals. This recommendation concerns collection exit, not scientific truth, model quality or a requirement to recover every historical original.

Authority: current user instruction; frozen census and Wave03/Wave04 artifacts; RS125 reconstruction, semantic-audit qualification, bounded adjudication, downstream trace and parent recovery records; repository AGENTS.md. Phase: FINAL CENSUS EVIDENCE RECONCILIATION. Detached HEAD: `ff806bd5bd8efc62fcf6420cf8b097a63ed70571`. Only the six requested files in this directory are written. Canonical ledgers and previous overlays remain unchanged.

**1. Original criteria and exact accounting**

The original [closure report](../../wave_03/census_closure/closure_report.md) distinguishes source interpretation, documentary membership, original execution identity and scientific validity. Exact source-specific requirements control closure: available bytes, a related stage, shared pytest version, aggregate attestations and model capabilities cannot substitute for producer/version/run bindings. Data-construction and design-only records do not acquire invented model-run requirements. The original [root ledger](../../wave_03/census_closure/root_evidence_requirements.jsonl) retains unresolved shared-root dependencies and the strict snapshot chronology limitation.

For this user's bounded collection-exit question, strict completeness and review readiness are separate. Missing originals may be explicitly retained as limitations. Accessible required documentary interpretation or an unadjudicated local companion alternative cannot be silently converted into completed collection. A later reviewer may accept exact limitations; this report does not pre-authorize their acceptance.

| Accounting state | Exact satisfied IDs | Open original CENSUS_BLOCKING IDs | Legacy bounded IDs | Deferred validation IDs | Strict blockers |
|---|---:|---:|---:|---:|---:|
| Original census | 0 | 211 | 3 | 8 | 212 |
| Wave03 closure overlay | 2 | 209 | 3 | 8 | 210 |
| Frozen Wave04 consolidation | 14 | 197 | 3 | 8 | 198 |
| This full-history reconciliation | 17 | 194 | 3 | 8 | 195 |

Every row sums to 222. The legacy snapshot debt is strict, explaining the extra strict blocker. The frozen Wave04 baseline omitted two established Wave03 closures; it did not supersede their witnesses. Its producer `evidence_consolidation/consolidate.py` builds its closed-ID set from nine aliases, one upstream source and four local semantic closures, without importing the two Wave03 closures. Its 198 is preserved as historical baseline, not presented as the cumulative union.

The reproducible union is: nine path obligations + one upstream availability obligation + four Wave04 documentary-semantic obligations + two Wave03 obligations + one RS125 construction obligation = **17 distinct IDs**. Thus `212 - 17 = 195`; equivalently `198 - 1 RS125 - 2 omitted carry-forwards = 195`. Parent recovery contributes **zero** original debt-ID closures. There are 205 unsatisfied dispositions, of which eight deferred owners and two legacy nonblocking limits are not strict blockers.

| Exclusive present disposition | IDs | Strict blockers |
|---|---:|---:|
| Fully resolved documentary obligation | 17 | 0 |
| Partially resolved obligation | 7 | 7 |
| Unresolved but locally recoverable | 116 | 116 |
| Bounded historical limitation | 13 | 11 |
| Deferred scientific validation | 8 | 0 |
| Active provenance/completeness blocker | 61 | 61 |
| Total | 222 | 195 |

`debt_reconciliation.jsonl` embeds each complete original payload and frozen baseline payload; exact original line/file hashes bind them. `closure_readiness_matrix.jsonl` has one row per original ID. No ID is removed, merged, renamed or assigned closure because of neighboring evidence. Status precedence and the exact satisfied-ID sets are in `evidence_input_manifest.json#/accounting_rules`.

**2. Accepted evidence integrations**

The ten selected availability closures are carried from [documentary_debt_closures.jsonl](../evidence_consolidation/documentary_debt_closures.jsonl). The four exact documentary-semantic closures are carried from [semantic_recovery_results.jsonl](../local_semantics/semantic_recovery_results.jsonl); their full-trainer, execution and adoption boundaries remain intact.

Two omitted carry-forwards receive one credit each after rechecking their original Git object/path identities:

- `DEBT-f538a199d205c5dcb2d7` / RS044: [Wave03 source recovery line 205](../../wave_03/census_closure/source_recovery_results.jsonl), `existing_evidence_closure`. Manuscript spine `210b902247f8cbf72133ffcd80b1340c2c82e318` at `ea03cdaa5394de03389d31aa1089b51e10340a57` selects claim-disposition `04c4e8a79c443fe9dfdf6a7561ba29ef1e596a4a` and novelty-positioning `20744d8673102312d0a8a8581784540706da885c`. This satisfies documentary selection, not final novelty adoption or empirical validity.
- `DEBT-05b8fab63276f304957e` / HV000971: [Wave03 source recovery line 3](../../wave_03/census_closure/source_recovery_results.jsonl), `new_local_evidence`. Exact observer pins `e03b38fa0681fd04d8633a7de184559bd0473133` / `60a53b6f5d1db8d7cbecded5b94d5231adcfc520` bind blobs `7ac5d0d5e93742c8d76cd3e467790a1866dbade9` / `49faa5eeb659ce7fbec7004e0933deaa6fb08c3b`. The existing saved companion interpretation closes the source obligation, not a historical successful run.

RS125 `DEBT-0f63c887bfbf703cfb43` receives exactly one additional credit from [closure_adjudication.json](../rs125_bounded_closure_adjudication/closure_adjudication.json), `/satisfied_obligations`, and [debt_overlay.jsonl line 1](../rs125_bounded_closure_adjudication/debt_overlay.jsonl). Frozen structural findings authenticate the 900 rows, builder/report identities, 300 contrast sets, pair identities, train720/dev180 and zero exact-pair reuse against both pinned 200-row probe versions. Construction commit is `10376c088b9008c6a62664f695d8b269d9ffee18`; dataset blob is `d38ce42f785ff89056d702e1f3cee756b71c717b`, SHA256 `afded7607460db009171a0b10d86fb550fb083a1554e0e1bc3818290e4beb1c1`. Source identities were freshly rehashed; structural and semantic findings are reused, not recomputed.

**3. Limited integrations and RS125 boundaries**

RS125 retains the frozen qualifier `NOT_CERTIFIABLE_AS_UNIFORMLY_VALID_HARD_SUPERVISION`. Documentary satisfaction neither certifies labels nor authenticates contemporaneous builder invocation, downstream adoption or historical model execution. The original requirement expressly excludes model/checkpoint identity for this construction record. No new semantic dataset audit was conducted.

RS124 `DEBT-4632366c789bac37b222` remains open separately. [Original debt line 153](../../wave_03/study_consolidation/unresolved_census_debt.jsonl) requires exact pre/post chronology distinguishing the dry-run header from appended observed patterns, plus original Stage31B predictions and manifest/checkpoint/tokenizer/dataset/cohort/seed/command identity. Both paragraphs remain evidence. The shared Stage31 path debt `DEBT-a320ca5be694655d8686` carries its prior path-only closure with zero additional credit. RS125 reconstruction does not authenticate RS124's run or select its probe version.

All six [locally recovered parent originals](../rs125_130m_parent_original_recovery/local_parent_recovery_manifest.json) were freshly byte-hashed at their recorded absolute paths; checkpoint tensors were not loaded. Their run-provenance and training-report JSON fields agree on seed, producer, selected checkpoint/epoch and C2 non-use.

| Exact grouped-D parent | Checkpoint SHA256 | Epoch | Direct C2 documentary finding |
|---|---|---:|---|
| seed180 | `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f` | 15 | head/loss false; weight 0; path null; loaded rows 0 |
| seed181 | `afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f` | 19 | head/loss false; weight 0; path null; loaded rows 0 |

Both records bind producer `3e0e9a435068c552abf20f3a74e0c3eccca344a3`, main-data SHA256 `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`, 3,600 rows and split seed8192. Exact JSON pointers and all six hashes are in the input manifest. Original argv/run namespaces are available; full collection/retry history is not newly certified.

This narrows the prior trace's `U_PARENT_130M` and the parent audit's payload/run gaps for these **two exact parents**. It does not close other grouped arms/seeds, original C2/C3 commands/checkpoints/predictions, or inherited initialization identity. Pretrained snapshot revision remains **UNVERIFIED**. No original 222-ID debt has an exact matching run obligation for these two bundles, so no topic-based closure mapping is invented. Supplementary gaps and boundaries remain in the manifest's `supplemental_execution_binding_reconciliation` and the RS125 reconciliation row. Downstream bindings are documentary dependency evidence, not fresh downstream model checks or scientific claim certification.

**4. Remaining strict completeness blockers**

| Original kind | Remaining strict IDs |
|---|---:|
| Incomplete source semantics | 99 |
| Historical transition interpretation | 33 |
| Study identity/run/boundary | 47 |
| Missing original literal companion | 6 |
| Damaged source | 4 |
| Source relationship uncertainty | 4 |
| Original fused-streaming run identity | 1 |
| Snapshot chronology | 1 |
| Total | 195 |

Each strict ID retains its exact obligations, original source identities and root links in the reconciliation and readiness matrix. Shared pytest producer/options and endpoint derivation remain prerequisites, not additional invented debt IDs. A version/wheel alone closes none of the four REL correspondences. No numerical parity check, new training run or scientific validation is required to finish this documentary task.

**5. Bounded historical obligations**

No obligation is proven globally historically unrecoverable. Four saved damaged sources—HV000202, HV001023, HV001096 and HV001097—are unrecoverable in intended-byte dimensions **from their recorded blobs alone**. They remain strict; original pre-damage or authenticated correction evidence would be needed for full recovery. No intended characters, boundaries or authority substitutions are guessed.

Six [unlocated literal obligations](../evidence_consolidation/unlocated_originals.jsonl) retain the exact Stage35A, Stage146, Stage148, two Stage163A and Stage196 originals. Prior scoped non-location and attributed non-retention of Kaggle Outputs do not prove global destruction, production, non-execution or failure. They may be reviewed as explicit historical limits, with their strict flags retained.

Three legacy bounded obligations remain: selected ancestry versus study-wide adoption, Git-ref actor/cause/time attribution, and snapshot capture/base chronology. Only the last is strict. Thus the 13 primary bounded dispositions include 11 strict records; none is relabeled satisfied. Their acceptance would be a separate bounded closure decision.

**6. Minimum unfinished documentary work**

`unresolved_critical_obligations.jsonl` contains **123 original IDs** with exact evidence-backed remaining local/conditional work: 116 frozen UNRESOLVED source interpretations, RS062's readout-to-corrected-manuscript companion chain, five conditional companion alternatives, and RS124's distinct chronology/original-run obligation. This is a work inventory, not 123 recoverable files, experiments or guaranteed closures.

The 116 are from [local semantics](../local_semantics/semantic_recovery_results.jsonl): 103 trainer targets, twelve publication targets and one test transition remain unfinished under the frozen report. Source bytes, partial function inventories and selected clause readings already exist; they do not establish complete source-specific documentary interpretation. RS062 is line4, debt `DEBT-086155ba8f21867e8c0d`. Its upstream artifact/implementation identities and corrected manuscript bindings remain open.

The five alternatives are HV000689/P4 (`DEBT-335ce38ed09e2efa3972`), HV000688/P5 (`DEBT-d47b95603a7bc513c8ec`), HV001128/model-composer (`DEBT-342c85cf5a59fc16338a`), HV001270/encoder masks (`DEBT-f1728e470d613a23775c`), and HV000699/fused-streaming provenance (`DEBT-a7318569c3011d837819`). Their saved local-versus-external alternatives need an exact documentary disposition; minimal downstream attestations cannot recover missing original tables. These routes are not promised to resolve the full obligations locally.

The other **72 strict IDs** are separately retained: 61 active provenance requirements plus eleven strict bounded limits. They are not automatically accepted, waived or added to a demand that every original be found before review. Source-specific bounded acceptance may preserve them. The 123 identified unfinished tasks are the current minimum procedural impediment, not a guarantee that completing them would itself authorize closure or eliminate the 72 strict obligations.

Missing evidence, a source that explicitly describes non-execution, a reported failed attempt, invalid supervision and scientific falsification remain distinct. Only source-backed historical statements are preserved. No absence-based execution inference or new falsification conclusion is made. The eight deferred scientific owners remain outside collection and are not executed.

**7. Readiness and validation**

All 222 IDs are classified and traceable; collection completion remains unsupported because required accessible documentary work is still explicitly unfinished. The readiness matrix records both strict flags and bounded collection-exit assessments. Exact dispositions and residuals are exposed for review without changing canonical counts or prior reports.

The earlier parent audit's `FAIL_REQUIRED_REFS_UNCHANGED_CHECK` is retained verbatim as history. Its artifact findings are not assigned an overall PASS. Fresh byte checks here verify specific local recoveries; the current start/end ref check is a separate result and does not erase the historical failure.

Independent serialized readback **passed all 35 accounting/provenance/preservation checks**, including 2,690 evidence-locator checks. All 744 pre-existing audit files, 743 existing untracked files and 3,593 tracked files remain byte-identical; HEAD, branch, all refs, index and tracked/staged diffs are unchanged. Exactly the six authorized output names exist. `git diff --check` passed; tracked and staged diff statistics are empty. `validation.json` records the actual checks and full start/end ref states. The input manifest records 86 read inputs, 48 frozen declared output-hash checks, six parent original checks and start-state fingerprints. This PASS concerns accounting, provenance and preservation only; collection completeness and scientific validity are not certified.

Commands actually executed: read-only `git status --porcelain`, `rev-parse`, `branch --show-current`, `show-ref`, `ls-files`, `cat-file blob`, `diff --binary`, `diff --cached --binary`, `diff --check` and diff statistics; `rg --files`/`rg -n`; PowerShell `Get-Content`; inline Python standard-library JSON/JSONL, SHA256/Git-blob hashing, set/count reconciliation and independent readback; report-file writes confined to this directory. No prior helper or research program was executed. An initial oversized snapshot could not be JSON-parsed and was recaptured before output creation; broad reads were truncated and replaced with targeted reads; exploratory console encoding and missing-key reads failed before mutation and were corrected. Python emits a stale-installation-location warning; successful checks exit zero. These are probe issues, not evidence-validation failures.

Training, evaluation, model import/forward, checkpoint deserialization, semantic dataset re-audit, dataset regeneration, Kaggle/GPU, fetch, Git-ref/worktree changes, commit and push were not performed. Tracked/staged diff stats are empty. The worktree remains unclean because pre-existing audit evidence and the six new outputs are untracked.

**8. Smallest justified next action**

Separately scope a read-only documentary adjudication of **HV001270 / `DEBT-f1728e470d613a23775c`** using the already saved exact encoder-mask/gradient-ownership companion evidence. Record which original requirement it satisfies and which original binding remains missing; preserve any external fallback. This is a bounded first task within the outstanding inventory, not authority for a broader audit or scientific execution. Then process the remaining critical inventory by exact ID, permitting evidence-backed limitations instead of inventing missing originals. No follow-up work starts automatically.

Reproduction of the debt arithmetic uses only frozen JSONL records:

```python
original_ids = {r['debt_id'] for r in original_debts}
baseline_closed = {r['debt_id'] for r in frozen_wave04
                   if r['consolidated_status'] != 'OPEN_RETAINED'}
wave03_closed = {r['debt_id'] for r in frozen_wave03_recovery
                 if not r['census_blocking_after_recovery']}
rs125_closed = {r['debt_id'] for r in frozen_rs125_overlay
                if r['new_documentary_satisfaction_credit'] == 1}
closed = baseline_closed | wave03_closed | rs125_closed
assert len(original_ids) == 222 and len(closed) == 17
strict_remaining = sum(r['blocks_strict_census_completeness']
                       and r['debt_id'] not in closed
                       for r in original_debts)
assert strict_remaining == 195
```
