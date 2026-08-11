# AGENTS.md — ContraMamba Research Agent Contract

This file defines repository-wide operating rules for AI coding/research agents working on ContraMamba.
It is a workflow contract, not a substitute for stage-specific research authority documents.

## 1. Research role and scope

ContraMamba is an evidence-entitlement research project, not a generic 3-way classifier project.
Agents must preserve the distinction between final-label correctness and whether the model is internally entitled to make that decision from the supplied evidence.

Do not optimize for topline metrics by silently weakening epistemic constraints, changing evaluation semantics, or leaking external-label information into training/selection.

Prefer the smallest change that answers the current research question.

## 2. Authority resolution

Before editing or executing anything, identify the authority for the current task.
Use this precedence order:

1. Explicit constraints in the current user/task instruction.
2. The stage-specific authority/specification/manifest/report explicitly named by that task.
3. Tests and immutable artifacts explicitly designated by that authority as executable contracts.
4. This `AGENTS.md` repository-wide contract.
5. README and historical documentation as context only.

If two higher-priority authorities conflict, STOP and report the conflict. Do not guess which one to follow.

Do not infer current research authority from an older README when newer stage-specific artifacts exist.
Do not silently promote a diagnostic result into an implementation or training authority.

## 3. Phase gates

Determine the task phase before acting.

### Report-only / scope-lock phase

Allowed:
- inspect repository state and artifacts;
- reason about mechanisms, algebra, gradients, contracts, and experimental design;
- write only explicitly authorized report/specification artifacts.

Forbidden unless explicitly authorized:
- implementation changes;
- training;
- evaluation;
- dataset regeneration;
- checkpoint mutation.

### Static-audit phase

Allowed:
- repository inspection;
- static code/test/contract analysis;
- read-only commands required for the audit;
- explicitly authorized audit reports.

Forbidden unless explicitly authorized:
- production-code modification;
- training/evaluation;
- changing datasets or checkpoints.

### Implementation phase

Allowed only within the explicit file/symbol/scope whitelist from the task authority.
Do not broaden the patch because another cleanup seems useful.
Do not combine unrelated refactors with a mechanism patch.

### Execution / experiment phase

Training, evaluation, data regeneration, or external diagnostics require explicit authorization.
A successful static gate does not itself authorize execution.

## 4. Current reason-router invariants

Until explicitly superseded by a newer authority document, preserve these established decisions:

- The q-only / ordered-mass decomposition that collapses back to the existing product composer is rejected as a causal mechanism.
- The approved mechanism family is Conditional First-Blocker Reason Router + Reason-Specific Supervision + Explicit Gradient Ownership.
- Primary reason order is FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED.
- Secondary reasons are diagnostic multi-label information only; do not duplicate them into the external class target or loss unless a newer authority explicitly changes this.
- Final 3-way CE is router-only; F/P/S and polarity inputs on that CE path must obey the stage-specific detach/ownership contract.
- EMA is an observer/baseline unless a newer authority explicitly changes its role; do not silently turn it into a teacher, target, or novelty mechanism.
- Preserve required A0–A3 ablations and E0 algebraic-equivalence checks when they are part of the active authority.

When a newer authority changes any item above, follow the newer authority and report the supersession explicitly.

## 5. Research-integrity invariants

Never silently change:
- dataset identity or source path;
- train/dev/test or pair-group split semantics;
- random seeds;
- intervention family definitions;
- label semantics;
- promotion criteria;
- artifact schema;
- checkpoint identity;
- authority/provenance hashes;
- clean-vs-external evaluation separation.

External diagnostic labels must not be used for training, threshold tuning, candidate selection, or promotion unless an explicit cleanly separated protocol authorizes that use.

Treat provenance checks, immutable commit identities, expected hashes, row identities, and stage manifests as research controls, not inconveniences to bypass.

If an expected artifact is missing or inconsistent, fail closed and report the blocker.

## 6. Editing discipline

Before making a patch:

1. Inspect `git status` and current branch/HEAD when local git access is available.
2. Read the relevant authority artifacts and relevant tests before editing implementation files.
3. Identify the exact allowed files/symbols and expected behavioral delta.
4. Check whether an existing implementation already satisfies part or all of the request.

While editing:

- Keep the patch minimal and auditable.
- Preserve unrelated behavior byte-for-byte where practical.
- Do not rename/reformat unrelated code.
- Do not weaken tests merely to make a patch pass.
- Do not replace a deterministic/provenance-preserving mechanism with a heuristic shortcut without explicit authority.
- Do not add dependencies unless necessary and explicitly justified.
- Never commit secrets, credentials, API keys, private tokens, or local environment files.

Do not overwrite, discard, reset, or clean user work that you did not create.
Do not use destructive git operations (`reset --hard`, forced checkout of user changes, force-push) unless explicitly instructed.

## 7. Validation contract

After an implementation change, run the narrowest relevant validation first and broaden only as authorized/needed.

At minimum, when local git/testing is available and the task permits execution:

1. `git diff --check`
2. relevant contract/unit tests for the changed mechanism
3. any explicit static gate named by the stage authority
4. broader tests only when required by the authority or when the narrow tests reveal cross-cutting risk

If the phase forbids execution, do not run tests merely because this section lists them; report the commands that would be required instead.

Do not report PASS unless the command actually ran successfully.
Distinguish:
- not run;
- passed;
- failed because of the patch;
- failed for a pre-existing/environmental reason.

Warnings such as line-ending conversion warnings should be reported separately from functional failures.

## 8. Commit and provenance discipline

Do not commit or push unless the user/task explicitly requests it or the active workflow explicitly authorizes it.

When a commit is created:
- report the full 40-character commit SHA;
- report the commit message;
- report whether the worktree is clean afterward;
- do not amend or rewrite an existing validated commit unless explicitly instructed.

When an experiment, Kaggle run, report, or result is tied to a code state, use the full commit SHA as the authority identity rather than an ambiguous moving branch name.

## 9. Kaggle handoff

Kaggle is an execution environment, not the source of repository authority.
Git/GitHub remains the source of code identity.

When a task reaches an authorized Kaggle handoff:
- provide the local git commit/push commands when relevant;
- also provide the corresponding Kaggle Python `subprocess` sync/execution cell in the same handoff;
- pin checkout to the new full commit SHA;
- if the SHA is not known yet, use an obvious placeholder and state exactly what must be replaced after committing;
- verify the checked-out SHA before running the stage command;
- do not silently execute a later stage just because synchronization succeeded.

## 10. Agent role separation

For nontrivial research changes, prefer explicit role separation:

### Auditor
Read-only. Determine repository state, applicable authority, algebraic/gradient equivalence risks, scope, and blockers.

### Implementer
Modify only authorized scope. Do not self-expand the research claim.

### Verifier
Independently inspect the resulting diff and repository state. Re-run authorized gates rather than trusting the implementer's summary.

When subagents or parallel work are used, give each agent a non-overlapping responsibility and merge only after independent verification.

## 11. Result reporting

Every substantive task report should make it possible to reconstruct what happened without relying on agent confidence.

Include, as applicable:
- decision/status;
- authority artifacts used;
- files inspected;
- files modified;
- concise semantic description of each modification;
- commands actually executed;
- test/static-gate results;
- blockers or unresolved assumptions;
- git diff/stat summary;
- commit SHA if one was created;
- whether training/evaluation was run;
- next authorized action, without automatically performing it.

Do not hide uncertainty behind a PASS label.
Do not claim a research conclusion that is stronger than the executed evidence supports.

## 12. Default stop conditions

STOP and ask/report rather than improvising when:
- the active authority is ambiguous;
- requested scope conflicts with an immutable/stage authority;
- a required provenance hash/commit/artifact does not match;
- a task would require external-label leakage;
- a requested implementation is algebraically equivalent to a previously rejected mechanism unless the task explicitly asks to test that equivalence;
- the only way forward would require changing seeds, data, labels, split semantics, or promotion criteria without explicit authorization;
- the phase boundary would be crossed (for example static audit -> training) without explicit permission.

The objective is not maximum autonomous activity. The objective is a reproducible, authority-preserving research workflow in which every mechanism change and experimental conclusion remains auditable.