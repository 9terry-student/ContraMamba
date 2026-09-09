# P3-W7 Seed8192 A0 Seed180 R2 Recovery Local-Preflight Boundary Correction Authority Specification Candidate

## Verdict and narrow supersession

`PASS_READY_FOR_INDEPENDENT_SEED8192_A0_SEED180_R2_LOCAL_PREFLIGHT_BOUNDARY_CORRECTION_AUTHORITY_VERIFICATION`

This REPORT-ONLY recovery execution-boundary correction supersedes **only** the
active recovery-handoff authority's impossible requirement that the Kaggle
recovery-packaging command inspect the current Windows-local helper and
Windows-local run registry. It does **not** waive either check. Both move,
unchanged in substance, to mandatory Windows-local Stage L immediately before
any Kaggle recovery execution is authorized.

All Kaggle-accessible fail-closed evidence, artifact, staging, manifest, ZIP,
collision, and source-immutability checks remain mandatory Stage K checks.
Stage K must not claim to inspect the current `cm.ps1` helper or current
Windows-local registry, and must not compare hard-coded constants and describe
that as verification of current local state. A Kaggle constant is not a local
control-plane read.

This correction authorizes no command execution, Stage L, Kaggle execution,
recovery packaging, ZIP creation, `cm import`, helper/importer/registry/marker
mutation, source or artifact mutation, checkpoint access, training, evaluation,
commit, or push. It changes neither scientific nor artifact identity and
creates no historical r2 evidence.

## Bound authority and immutable identities

| Binding | Required value |
| --- | --- |
| Active recovery-handoff authority HEAD | `48d258fc5e8b09e163e9252f33c86936244ef872` |
| Active authority report | `reports/reason_router_p3w7_seed8192_a0_seed180_r2_provenance_recovery_handoff_execution_authority_spec_candidate.md` |
| Existing failed candidate | `reports/reason_router_p3w7_seed8192_a0_seed180_r2_provenance_recovery_handoff_command_freeze_candidate.md` |
| Run name | `p3w7-seed8192-a0-seed180-r2` |
| R2 execution / expected / actual commit | `abd85a088c274678004432160625d42208112848` |
| Original registered command SHA256 | `82be0c377e305228609e8ce9f75a6b3e8b8a6f83f9999a1543914f9fc517c1f4` |
| Frozen importer helper bytes / SHA256 | `91498` / `09097e460ce9f05d5ead09ad3ee9499ac6b9da4298d0a69e8abde6006e5facae` |
| `run.log` SHA256 | `a0d4b015cc77e8060184a5035333fe46e101f52d931af3101950382e24407f4e` |
| `run.meta` SHA256 | `23989109951bdf2b3fdfc17f9a7739c34e2875131bc76c0ccb1e313858fcdc27` |
| Handoff schema / discovery / file count | `contramamba-handoff-v3` / `fixed_authorized_path_sha256_recovery` / `5` |

The five authorized paths and frozen SHA256 values, wrapper semantics,
timestamps, output pathname, exact ZIP namespace, and later importer boundary
remain exactly as in the active recovery-handoff authority. This correction
does not add, remove, substitute, discover, regenerate, or rehash an artifact.

## Corrected two-stage fail-closed boundary

### Stage L — Windows-local control-plane preflight

Stage L runs on the user's Windows machine immediately before the controller
may authorize Stage K. Its exact PowerShell command must be authored in a
future correction-bound command candidate, independently reviewed, and frozen
as exact UTF-8 bytes and SHA256 before it runs. It is a read-only machine check:
it must not use `cm run`, create wrapper evidence, write a registry, touch the
helper, write an attestation file, create a ZIP, or represent output as r2
historical wrapper metadata.

The exact Stage L command must fail closed unless it independently verifies:

1. The active authority worktree it actually reads is at exact HEAD
   `48d258fc5e8b09e163e9252f33c86936244ef872`, with no tracked or staged
   changes, as required by the eventual execution workflow.
2. The helper path is resolved through `$HOME/.contramamba/cm.ps1` (not a
   hard-coded profile); the resolved regular file is exactly `91498` bytes and
   SHA256 `09097e460ce9f05d5ead09ad3ee9499ac6b9da4298d0a69e8abde6006e5facae`.
3. The object-keyed local registry at `$HOME/.contramamba/run-registry.json`
   contains exactly the authentic original entry
   `p3w7-seed8192-a0-seed180-r2`, with `head`
   `abd85a088c274678004432160625d42208112848` and `command_sha256`
   `82be0c377e305228609e8ce9f75a6b3e8b8a6f83f9999a1543914f9fc517c1f4`.
4. Its stored `command` is recomputed from its exact in-memory string via
   `[System.Text.Encoding]::UTF8.GetBytes($runCommand)` and lowercase SHA-256,
   the same `cm.ps1` semantics. No trim, newline normalization, code-page
   conversion, serialization round-trip, or hard-coded surrogate is allowed.
5. The exact Stage K payload about to enter the normal ContraMamba workflow has
   passed independent freeze: Stage L reads its exact bytes and requires its
   byte count and SHA256 to equal separately recorded frozen values.

Only complete controller-observable machine PASS output, including resolved
paths, identities, and Stage-K payload bytes/SHA256, satisfies Stage L. A user
assertion, partial output, prior PASS, or manually typed value cannot substitute.
A mismatch is terminal: no Stage K authorization follows and no repair or
mutation of helper, registry, or evidence is authorized.

Stage L output is control-plane authorization evidence only. It is not placed
in the v3 ZIP, `run.log`, `run.meta`, `command.sh`, r2 provenance, or claimed
historical wrapper metadata. No persistent local attestation is required.

### Stage K — Kaggle recovery packaging

Stage K runs only in Kaggle against authentic r2 evidence and recovery output.
Its exact Bash/Python payload must be authored separately from Stage L,
independently reviewed, frozen as exact UTF-8 bytes with final LF and SHA256,
and bound by Stage L. It may execute only after controller receipt and
acceptance of complete Stage L PASS evidence through normal ContraMamba Kaggle
run workflow.

Stage K must retain every truthful Kaggle-side fail-closed check in the active
authority: authentic `run.log`, `run.meta`, and `command.sh` bytes, hashes, and
semantic fields; exactly five authorized paths and frozen SHA256 values; source
hash before size/copy/manifest; staged hash/size; exact v3 manifest semantics
and `artifact_discovery`; exact ZIP namespace; post-ZIP member/path/manifest/
hash/size checks; collision/failure handling; and source immutability.

Stage K must not inspect or claim verification of `$HOME/.contramamba/cm.ps1`,
`C:\\Users\\Home1\\.contramamba\\cm.ps1`, or Windows-local `run-registry.json`.
Hard-coded helper size/SHA256, registry HEAD, or registry command hash may not
be presented as current-local verification. Those are exclusively Stage L
facts; Stage K verifies only wrapper and evidence facts present in Kaggle.

## Required sequencing and gates

1. Author exact Stage L local-preflight command.
2. Author exact Stage K Kaggle-recovery command.
3. Independently verify both, including boundary ownership and exact payload
   semantics.
4. Freeze both byte counts and SHA256 values in repository evidence.
5. Commit/push correction and freeze reports and remotely authenticate them.
6. User runs frozen Stage L locally.
7. Stage L must PASS.
8. User returns complete PASS evidence to controller.
9. Only then may controller authorize frozen Stage K through normal
   ContraMamba Kaggle run workflow.
10. Stage K generates the recovery handoff.
11. Real `cm import` remains a later, separately authorized boundary.

No Stage L PASS means no Kaggle recovery execution. Real import remains subject
to unchanged importer checks and a separate worktree pinned to
`abd85a088c274678004432160625d42208112848`; at import time the helper must
again be exactly the frozen `91498` bytes and SHA256 above.

## Existing failed candidate and preservation requirements

The existing failed command-freeze candidate is **not validated, corrected, or
activated** by this report. It must be re-reviewed under this corrected
boundary. Any later modification of its report or command changes its identity
and requires fresh exact byte-count/SHA256 verification; its prior identity
cannot carry forward.

At authoring start it was exactly `14367` bytes, SHA256
`9e8049251409741bbe3a1a4a072d0dcbad77666182e08daa248d08e9cb121a82`, and Git
blob `65ea058f122a4ec9cdc07fa4538d916c91350b90`. It must remain byte-identical
throughout this task. This report neither edits it nor treats its payload as
eligible to execute.

No helper, importer, registry, original evidence, artifact contract, scientific
identity, split, seed, checkpoint, or future import guard may be modified to
enact this correction. Any condition requiring one is a STOP requiring new
authority.

## Read-only authoring record

Read-only inspection confirmed the active authority HEAD, the frozen helper at
the `$HOME`-resolved path, and authentic local registry entry. The registry
command recomputed under helper UTF-8 semantics to its registered SHA256. The
existing failed candidate was checked before authoring and must be rechecked
after authoring.

Recovery execution: NOT PERFORMED. Kaggle execution: NOT PERFORMED. Real import:
NOT PERFORMED. Training/evaluation: NOT PERFORMED. Commit/push: NOT PERFORMED.
