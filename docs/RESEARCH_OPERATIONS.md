# ContraMamba Research Operations

This handbook is the stable operating layer for ContraMamba work across
ChatGPT, Codex, PowerShell/local shell, and Kaggle. It is process guidance, not
stage-specific scientific authority.

For repository-wide agent rules, read `AGENTS.md`. For current project state,
read `reports/RESEARCH_STATE.md`. For curated artifact navigation, read
`reports/ARTIFACT_INDEX.md`.

## Role Boundary

ChatGPT:

- formulates scientific questions and hypotheses;
- designs experimental comparisons and interpretation plans;
- decides what scientific question should be tested next;
- drafts bounded Codex tasks when repository work is needed;
- must not treat unvalidated runtime output as established evidence.

Codex:

- inspects repository state and artifacts;
- implements bounded authorized changes;
- performs static audits and validation tooling;
- constructs exact commands and provenance checks;
- checks artifacts, hashes, dirty state, and import contracts;
- must not independently invent scientific hypotheses, hyperparameters, or
  experimental scope unless explicitly delegated.

PowerShell/local shell:

- is the operator interface for git/status inspection, local preflight
  invocation, exact command execution when instructed, hash/file checks, and
  artifact handoff;
- must not silently alter scientific inputs, commands, paths, or artifacts.

Kaggle:

- is a compute environment only;
- may perform dependency/environment setup, CPU preflight, authorized GPU
  execution, and raw artifact production;
- makes no autonomous scientific decisions;
- must not run ad hoc hyperparameter search or "result looks bad, try another
  setting" behavior.

## Normal Flow

Scientific decision -> bounded authority/spec -> implementation/static
validation -> local preflight -> Kaggle execution -> collection/import ->
provenance validation -> scientific analysis -> next scientific decision.

Each step should preserve exact command identity, Git HEAD, input identity,
artifact paths, hashes, and interpretation boundary.

## Evidence Levels

1. `PLAN`
2. `FROZEN_AUTHORITY`
3. `IMPLEMENTED_AND_STATICALLY_VALIDATED`
4. `RUNTIME_EXECUTED`
5. `IMPORTED_AND_PROVENANCE_VALIDATED`
6. `SCIENTIFICALLY_INTERPRETED`

Later levels cannot be inferred from earlier levels. A frozen plan is not
execution. Static validation is not a runtime result. A runtime exit code is not
provenance validation. Provenance validation is not scientific interpretation.

## Pre-URP Boundary

The current pre-URP period may be used for infrastructure preparation,
repository organization, environment checks, command dry contracts, synthetic
analysis tooling, and provenance automation.

Formal P3-W7-A0 scientific execution begins only after the research-credit
period starts and a separate P3-W7-A0 execution authority is created,
independently verified, and frozen. Pre-URP outputs must not be represented as
P3-W7 scientific evidence.

## Immutable Evidence Rules

- Raw runtime artifacts are not manually edited.
- Frozen historical authority is not rewritten.
- Historical paths are not casually renamed.
- Derived summaries and figures stay separate from raw evidence.
- Failed, blocked, negative, or non-promoted runs may be retained when
  scientifically relevant.
- Exit code 0 alone does not establish valid evidence.
- Historical evidence remains historical unless a newer frozen authority
  explicitly changes its role.

## Codex Prompt Protocol

Default future Codex tasks should be short and bounded:

```text
Stage:
Read:
Goal:
Allowed delta:
Forbidden:
Validation:
Return:
```

Stable repository-wide rules should be referenced from this file and
`AGENTS.md` rather than repeated in every prompt. Stage-specific instructions
should state the exact authority artifacts and allowed delta.

## Token Efficiency

- Do not paste the full historical lineage when `reports/RESEARCH_STATE.md` is
  sufficient.
- Reference authority artifacts by path.
- Do not dump full inventories unless specifically requested.
- Report diffs, blockers, commands, and validation results compactly.
- Distinguish stable rules from stage-specific instructions.
- Prefer `reports/artifact_manifest.json` lookup over repeated repository
  archaeology when machine-readable pointers are enough.

## New Chat Guidance

Start a new Codex chat when:

- a major experimental workstream changes;
- the project formally transitions from pre-URP preparation to P3-W7-A0;
- context becomes stale, conflicted, or too large to audit safely.

Otherwise, continue the current bounded workstream chat so local decisions and
artifact context remain close at hand.
