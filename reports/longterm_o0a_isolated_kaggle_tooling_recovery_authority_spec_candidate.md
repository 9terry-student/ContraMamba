# Long-term O0a Isolated-Kaggle Tooling-Recovery Authority Specification Candidate

**Status:** PRE-IMPLEMENTATION TOOLING RECOVERY AUTHORITY CANDIDATE / NOT YET ACTIVE

This candidate is documentation only. It authorizes no launcher modification, scientific execution, Kaggle execution, training, evaluation, model download, model forward, dataset regeneration, checkpoint mutation, staging, commit, or push.

## 1. Blocker

Long-term O0a scientific execution authority storage identity:

`7f52e21e809c4cd68d914f9e55fef5ec34f02f14`

Authorized execution implementation identity:

`9f595e64f8a6aaec5bb1975521a6ee86e2ab1401`

The local execution worktree `C:\o0a-lf` has:

- `HEAD = 9f595e64f8a6aaec5bb1975521a6ee86e2ab1401`
- clean status;
- observer SHA256 `2937223733750f72344f18d30f3686d4ffc96cde5b397524560443a9e885489f`;
- dataset SHA256 `17e07f4ca9d3a11511c67493b7b9e9ff9ef6b1ebdca4c9d779705ad72e7c04dc`.

The observed external launcher is `C:\Users\Home1\.contramamba\cm.ps1`. Its `$Repo` assignment hardcodes `C:\Users\Home1\Desktop\ContraMamba`. Consequently, `cm context` from `C:\o0a-lf` reports canonical-main `HEAD = 7f52e21e809c4cd68d914f9e55fef5ec34f02f14` rather than the detached implementation identity.

Generated Kaggle cells hardcode `REPO="/kaggle/working/ContraMamba"`. That canonical Kaggle repository is dirty because of unrelated URP state at `reports/reason_router_p3w7_a0_current_lineage_runs/`. This conflicts with isolated O0a operation. No model forward occurred.

## 2. Required tooling capability

Authorize only a bounded repair to the external launcher adding these two explicit optional overrides:

- `CONTRAMAMBA_REPO_ROOT`: selects the local repository/worktree used by `cm`.
- `CONTRAMAMBA_KAGGLE_REPO`: selects the Kaggle repository path embedded in generated `kaggle`, `run`, and `collect` cells.

When either variable is absent, behavior must remain byte-semantically equivalent to the current defaults:

- local repository: `C:\Users\Home1\Desktop\ContraMamba`;
- Kaggle repository: `/kaggle/working/ContraMamba`.

No automatic current-directory inference is authorized. An invalid explicit override must never silently fall back to a default.

## 2A. Default-compatibility protected invariants

When both `CONTRAMAMBA_REPO_ROOT` and `CONTRAMAMBA_KAGGLE_REPO` are absent, any
future launcher repair MUST preserve existing behavior for all of the
following:

1. Command names: `status`, `context`, `gate`, `ship`, `kaggle`, `run`,
   `collect`, and `import`.
2. Canonical local repository default: `C:\Users\Home1\Desktop\ContraMamba`.
3. Canonical Kaggle repository default: `/kaggle/working/ContraMamba`.
4. Existing gate behavior and registered gate targets.
5. Existing ship behavior and file-state reporting semantics.
6. Existing run-command exact-byte transport semantics.
7. Existing run-command SHA256 verification semantics.
8. Existing run-registry command binding semantics, except for the narrowly
   authorized addition of `kaggle_repo` for new entries.
9. Existing handoff schema and handoff manifest semantics.
10. Existing collect checksum/provenance behavior.
11. Existing import validation, collision checks, registry binding, audit
    recording, and provenance logic.
12. Legacy registry behavior: entries lacking `kaggle_repo` continue to mean
    exactly `/kaggle/working/ContraMamba`.

The tooling repair is NOT authority to redesign, simplify, refactor, rename,
or otherwise alter these behaviors. The only authorized semantic additions are:

- explicit local repository override plumbing;
- explicit Kaggle repository override plumbing;
- fail-closed validation of those overrides;
- `kaggle_repo` persistence for new run-registry entries;
- `run`/`collect` use of that persisted repository identity.

## 3. Local repository override contract

When `CONTRAMAMBA_REPO_ROOT` is set, the launcher must resolve that explicit path, require it to exist, require it to be a Git worktree/repository, and use that exact path for status, context, gate, ship, kaggle, run, collect, and import repository operations wherever the existing command uses `$Repo`.

An invalid explicit local override must fail closed and must not use the canonical main repository.

The intended O0a value is `C:\o0a-lf`. This is operational identity only. Scientific execution identity remains `9f595e64f8a6aaec5bb1975521a6ee86e2ab1401`.

## 4. Kaggle repository override contract

When `CONTRAMAMBA_KAGGLE_REPO` is set, generated safe-bootstrap, fresh-bootstrap, pinned-run, and collector cells must use it instead of `/kaggle/working/ContraMamba`.

The intended O0a value is exactly `/kaggle/working/ContraMamba-o0a-9f595e6`.

The override must be validated before interpolation. It must be a safe absolute Kaggle working path with no shell metacharacters or newline injection. At minimum it must:

- begin with `/kaggle/working/`;
- not equal `/kaggle/working`;
- reject newline, carriage return, quotes, backticks, command substitution, semicolon, pipe, ampersand, and traversal components.

An invalid explicit override must be rejected before cell generation and must not fall back silently.

## 5. Run-registry binding

At `cm run save <name>`, every new registry entry must include an explicit field equivalent to `kaggle_repo`, binding the Kaggle repository path used for that run.

`cm run <name>` must use the registry-bound Kaggle repository, not a later environment-variable value. `cm collect <name>` must use the same registry-bound path. This prevents bootstrap on one Kaggle repository followed by run or collection on another.

Existing registry entries must remain readable. An old entry lacking `kaggle_repo` must preserve legacy default behavior using `/kaggle/working/ContraMamba`; it must not be silently reinterpreted as the O0a alternate repository.

## 6. O0a intended tooling identity

After repair and separate verification, O0a will use:

```text
CONTRAMAMBA_REPO_ROOT=C:\o0a-lf
CONTRAMAMBA_KAGGLE_REPO=/kaggle/working/ContraMamba-o0a-9f595e6
```

Expected results:

- `cm context` resolves `HEAD = 9f595e64f8a6aaec5bb1975521a6ee86e2ab1401`;
- `cm kaggle` generates `EXPECTED_COMMIT = 9f595e64f8a6aaec5bb1975521a6ee86e2ab1401`;
- generated cells use `REPO = /kaggle/working/ContraMamba-o0a-9f595e6`.

This alternate path deliberately avoids touching `/kaggle/working/ContraMamba` and its unrelated URP artifacts.

## 7. Protected scientific runtime identity

This tooling repair MUST NOT change the following protected scientific values:

- Execution implementation commit: `9f595e64f8a6aaec5bb1975521a6ee86e2ab1401`.
- Observer SHA256: `2937223733750f72344f18d30f3686d4ffc96cde5b397524560443a9e885489f`.
- Dataset SHA256: `17e07f4ca9d3a11511c67493b7b9e9ff9ef6b1ebdca4c9d779705ad72e7c04dc`.
- Model ID: `state-spaces/mamba-130m-hf`.
- Tokenizer ID: `state-spaces/mamba-130m-hf`.
- Exact model/tokenizer revision: `5708daa364c50b880e7bd92eab456e0d34492ee9`.
- Runtime device: `cpu`.
- Runtime dtype: `float32`.

No alternative HF revision is authorized; mutable `main` is not authorized.
No GPU/CUDA execution is authorized. No `float16` or `bfloat16` is
authorized. Tooling repair cannot reinterpret these scientific execution
values.

## 8. Scientific authority preservation

This repair does not authorize changes to the O0a observer, O0a tests, dataset, model or tokenizer identity, HF revision, runtime observer SHA, CPU/float32 settings, scientific command, output directory, run name, metrics, interpretation boundary, or scientific claim maximum.

The exact O0a run name remains `longterm-o0a-native-hidden-proxy-screen-v1`. The exact scientific command remains unchanged.

## 9. URP isolation

O0a must not delete `/kaggle/working/ContraMamba`; use `cm kaggle fresh` against the canonical Kaggle repository; delete, move, collect, or import `reports/reason_router_p3w7_a0_current_lineage_runs/`; modify URP local files; consume URP registry entries; or alter URP attempts or evidence. The alternate Kaggle path exists to avoid that interference.

## 10. Launcher baseline provenance

During the later implementation task, before modifying `C:\Users\Home1\.contramamba\cm.ps1`, the implementer must compute and report its exact SHA256. The implementation authority must bind that observed baseline hash. If launcher bytes change between authority verification and implementation, implementation must block and return the new hash.

The verifier-observed pre-implementation baseline is
`94a4333037cf434b895fbee08e70dc1254b1e9ea233d6bad424dd4a4b34ecdaf`.
The future implementation task MUST recompute the hash before modification. If
the recomputed bytes differ, it MUST BLOCK before modification and report the
new hash. This recorded value is not permission to ignore drift.

The later implementation report must record the post-repair launcher SHA256 and include the exact bounded diff/hunks needed to reproduce the repair, because `cm.ps1` is external to Git. No unrelated launcher cleanup or refactor is authorized.

## 11. Required future implementation tests

Future implementation must test without Kaggle execution:

- **A — default mode:** no overrides; canonical local and Kaggle defaults remain selected.
- **B — valid local override:** `C:\o0a-lf` is selected and context resolves `9f595e6...`.
- **C — invalid local override:** fail closed with no fallback to main.
- **D — valid Kaggle override:** safe bootstrap, fresh bootstrap, run, and collect all use the alternate path.
- **E — unsafe Kaggle override:** rejected before cell generation.
- **F — registry:** new entries record `kaggle_repo`; later environment drift cannot alter the registered path; collect uses it; legacy entries without the field retain the canonical legacy default.
- **G — O0a dry provenance:** generated bootstrap identity is exactly `9f595e64f8a6aaec5bb1975521a6ee86e2ab1401` plus `/kaggle/working/ContraMamba-o0a-9f595e6`.

No Kaggle cell is executed during tooling validation.

## 12. Implementation boundary

The later implementation may modify only `C:\Users\Home1\.contramamba\cm.ps1`, plus temporary test artifacts outside the Git repository that are removed after validation. No repository source file may be modified by that implementation task. This authority candidate is handled separately in Git.

No training, evaluation, model download, model forward, or Kaggle execution is authorized by this tooling authority.

## 13. Freeze and explicit tooling-authority activation

This candidate does not authorize launcher modification yet. The required sequence is:

1. create this candidate;
2. while this candidate is uncommitted, it is NOT ACTIVE;
3. independently verify PASS over the repaired exact candidate bytes;
4. run the existing `cm ship` file-state check;
5. commit the exact candidate unchanged;
6. push that unchanged commit;
7. controller records the resulting commit as the tooling-recovery
   authority-freeze identity;
8. only after steps 3–7, THE TOOLING RECOVERY AUTHORITY BECOMES ACTIVE;
9. no textual modification after freeze is required for activation;
10. only after that explicit activation may the bounded `cm.ps1`
   implementation task begin;
11. after launcher implementation, independent launcher verification is still
    required;
12. O0a bootstrap may resume only after launcher verification PASS.

Activation of this tooling authority does NOT activate or execute a Kaggle run.
No implementation permission exists before explicit activation, and tooling
activation alone grants no Kaggle execution permission.

No scientific execution occurs during steps 1–12.

The expected Git delta for this documentation task is only:

```text
?? reports/longterm_o0a_isolated_kaggle_tooling_recovery_authority_spec_candidate.md
```

The existing unrelated worktree state, including URP artifacts, patch files, and other user files, must be preserved. In particular, do not change any existing O0a file, `docs/`, `data/`, `scripts/`, `tests/`, `src/`, `AGENTS.md`, URP/reason-router files, `reports/stage180a_pass2_annotations_completed.csv`, root `.patch` files, or `C:\Users\Home1\.contramamba\cm.ps1`.

## 14. Candidate verification record

The candidate author must report:

1. exact file created;
2. current HEAD and drift assessment;
3. blocker statement;
4. intended local repository override;
5. intended Kaggle repository override;
6. default compatibility rule;
7. fail-closed override validation;
8. run-registry binding rule;
9. legacy registry compatibility;
10. URP isolation rule;
11. launcher baseline-provenance rule;
12. future test matrix;
13. scientific-authority preservation;
14. validation outputs;
15. Git diff/no-index stat;
16. `git status --short`;
17. confirmation that `cm.ps1` and all other files were untouched;
18. confirmation that no Kaggle/model/run/collect/import/staging/commit/push occurred;
19. the final readiness token.

For this candidate-creation task, the readiness token is:

`LONGTERM_O0A_ISOLATED_KAGGLE_TOOLING_RECOVERY_AUTHORITY_CANDIDATE_READY_FOR_VERIFICATION`
