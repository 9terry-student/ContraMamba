# Longterm O0c Runtime-Source Provenance Preflight Command-Transport Recovery Execution Authority Spec Candidate

## 1. Verdict

PASS_READY_FOR_REVERIFICATION

This is a narrow execution-recovery authority candidate for the ContraMamba O0c exact-runtime source-provenance preflight. It authorizes exactly one replacement run registration/execution after the v1 command was incorrectly registered due to clipboard transport.

This corrective authoring task accepts the independent verifier finding that the prior `2ded70...` / `3612` byte command identity was invalid. That identity is removed from normative recovery requirements and must not be defended or reproduced.

This authoring task does not register, execute, collect, import, train, evaluate, mutate packages, mutate implementation, modify `cm.ps1`, commit, or push.

## 2. Authority Chain

Current controller instruction in this task card is the active highest authority for this candidate.

Frozen O0c runtime-source provenance preflight execution authority:

`4726f2ab1540f6fe6148e89d11200ac0469f286f`

Frozen O0c runtime-source provenance preflight implementation:

`de874a22df4f60adbdc5efbcf294961c7b3a48a5`

Frozen implementation authority:

`811ae9c843564e8cddb5fc373761afb618cb7cfd`

Frozen runtime-source provenance preflight authority:

`8c6a0ccf2a8583b9b7accbdb5ab757d722b6e328`

Frozen O0c native-state instrumentation authority:

`242ad9ed70fc995ebda560911a7d0dfd2f18f9b3`

Repository-wide authority:

`AGENTS.md` and Kaggle/failure-recovery runbooks.

Canonical repository:

`C:\Users\Home1\Desktop\ContraMamba`

Canonical HEAD verified for this authoring task:

`4726f2ab1540f6fe6148e89d11200ac0469f286f`

## 3. Failure Classification

The observed v1 failure is frozen as:

`PRE-SEMANTIC COMMAND-TRANSPORT / COMMAND-REGISTRATION FAILURE`

Observed run:

`longterm-o0c-runtime-source-provenance-preflight-de874a2-v1`

Pinned execution commit:

`de874a22df4f60adbdc5efbcf294961c7b3a48a5`

Registered command SHA256:

`50a2bdcca75266a8fa3cc561e6d5febb7789f1bcb28ae37da56015e4562039d3`

Observed wrapper evidence showed `COMMAND_BASE64` decoded to a Windows PowerShell wrapper beginning materially with:

`$oldNativePreference = $PSNativeCommandUseErrorActionPreference`

and containing:

`$env:CONTRAMAMBA_REPO_ROOT = 'C:\o0c-preflight-de874a2'`

`cm run save ...`

`cm run ...`

The decoded command was then invoked by Kaggle through:

`bash -x "$COMMAND_FILE"`

and failed immediately with shell errors including:

`command not found`

`cd: C:o0c-preflight-de874a2: No such file or directory`

`cm: command not found`

Final wrapper result:

`EXIT_CODE=2`

`CONTRAMAMBA RUN FAIL`

The frozen O0c preflight Python script was never semantically invoked. There was no script-emitted `preflight_status=...`, and there was no O0c canonical preflight result.

This v1 failure must not be classified as:

- `PASS_SOURCE_IDENTITY_FROZEN`;
- `BLOCKED_RUNTIME_VERSION_MISMATCH`;
- `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED`;
- `BLOCKED_BACKEND_PATH_UNRESOLVED`;
- any other semantic/provenance preflight status;
- scientific evidence.

## 4. Root Cause

The root cause is narrowly:

The intended frozen bash command had been copied to the clipboard, but the subsequent local PowerShell wrapper used to invoke `cm run save` / `cm run` was then copied, overwriting clipboard contents.

`cm run save` therefore correctly registered the bytes present on the clipboard, but those bytes were not the frozen execution command.

This is command transport/registration error, not frozen preflight implementation failure.

Do not modify `cm.ps1` for this failure.

## 5. V1 Immutability

The failed v1 provenance must be preserved.

Do not:

- overwrite the v1 run registry entry;
- reuse v1 as the replacement execution name;
- delete or replace its Kaggle log;
- delete or replace its meta file;
- delete or replace its command file;
- reinterpret v1 as a semantic preflight attempt.

The existing v1 evidence may remain in Kaggle run-log storage. Collection/import of v1 is not required merely to permit recovery, because the complete failure evidence is command-transport infrastructure evidence rather than a scientific/preflight artifact.

Do not destroy v1 evidence.

## 6. Recovery Basis

The frozen execution authority at `4726f2ab1540f6fe6148e89d11200ac0469f286f`, in `reports/longterm_o0c_runtime_source_provenance_preflight_execution_authority_spec_candidate.md`, permits a rerun for:

- clearly transient execution-infrastructure failure;
- command transport failure;
- notebook/session failure before script semantics executed.

The v1 failure occurred before semantic preflight completion and before the frozen Python script was semantically invoked. Recovery is therefore permitted without changing preflight semantics.

This recovery changes only:

- run identity from v1 to v2;
- local run-registry binding to the correct frozen command bytes.

It changes none of:

- implementation;
- execution commit;
- CLI;
- expected runtime;
- script identity;
- output path;
- PASS/BLOCKED semantics;
- CPU/GPU policy;
- model/tokenizer prohibition;
- package mutation prohibition.

## 7. Replacement Run Name

The replacement run name is frozen exactly as:

`longterm-o0c-runtime-source-provenance-preflight-de874a2-v2`

This is a new retry identity. It must not overwrite or alias v1.

The `de874a2` abbreviation continues to identify the exact frozen execution commit:

`de874a22df4f60adbdc5efbcf294961c7b3a48a5`

## 8. Exact Execution Commit

The replacement v2 remains pinned to exactly:

`de874a22df4f60adbdc5efbcf294961c7b3a48a5`

No execution is authorized against:

`4726f2ab1540f6fe6148e89d11200ac0469f286f`

or any later commit.

This recovery authority itself may be frozen on a later canonical main commit, but scientific/preflight execution identity remains `de874a22df4f60adbdc5efbcf294961c7b3a48a5`.

## 9. Exact Frozen Script

Path:

`scripts/preflight_longterm_o0c_runtime_source_provenance.py`

SHA256:

`73ffff2681928389933867caa9586b3f54260b2f64bdb1a44c50b77e4accdbfc`

Bytes:

`32034`

No implementation change is authorized.

## 10. Exact Command Recovery

The frozen execution authority read for exact command recovery is:

Commit:

`4726f2ab1540f6fe6148e89d11200ac0469f286f`

File:

`reports/longterm_o0c_runtime_source_provenance_preflight_execution_authority_spec_candidate.md`

The exact normative command was independently re-extracted from the `BEGIN_EXACT_COMMAND` / `END_EXACT_COMMAND` section according to the frozen rule:

- use the content between `BEGIN_EXACT_COMMAND` and `END_EXACT_COMMAND`;
- remove only the opening ` ```bash` and closing ` ``` ` fence lines;
- encode as UTF-8;
- use LF line endings;
- add no final LF.

Exact recovered frozen command SHA256:

`bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`

Exact recovered frozen command byte count:

`3505`

The recovered command SHA256 differs from the erroneous v1 registered PowerShell-wrapper SHA256:

`50a2bdcca75266a8fa3cc561e6d5febb7789f1bcb28ae37da56015e4562039d3`

The replacement v2 run must use exactly these recovered frozen command bytes. Do not rewrite, reformat, normalize, or reconstruct the command from memory.

Block if independent extraction cannot reproduce exactly:

- SHA256 `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`;
- byte count `3505`.

Block if the recovered command SHA unexpectedly equals the erroneous v1 SHA, unless source inspection proves an impossible identity collision.

## 11. Representation Boundary

The corrected recovery authority distinguishes three representations.

A. Authority canonical command bytes:

- UTF-8;
- LF line endings;
- no final LF;
- SHA256 `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`;
- `3505` bytes.

B. Windows clipboard text representation:

- may be subject to clipboard/API newline behavior;
- is not authoritative merely because it is on the clipboard;
- must be measured immediately after the intended future clipboard write.

C. `cm run save` registry input:

- is exactly what live `cm.ps1` obtains from `(Get-Clipboard -Raw).Trim()`;
- is then subject to the live `cm.ps1` `%%bash` handling, if present;
- is UTF-8 encoded, SHA256 hashed, and stored in the run registry.

Do not assert A == B == C without measurement.

## 12. Live cm.ps1 Run-Save Semantics

The live external tool inspected for run-save semantics is:

`C:\Users\Home1\.contramamba\cm.ps1`

Verified SHA256:

`b15d70832e7c76c05fea6a9955bd199edcf9fb633fe0fe34266c44788260f570`

Relevant `run save` behavior:

- validates the run name;
- resolves the Kaggle repo override before accepting a run;
- sets `$runCommand = (Get-Clipboard -Raw).Trim()`;
- rejects an empty command;
- blocks if the command begins with a Markdown code fence;
- splits on CRLF or LF and removes a leading `%%bash` line only when the first trimmed line equals `%%bash`, then trims again;
- optionally supports `CONTRAMAMBA_RUN_COMMAND_BYTE_MODE=utf8-final-lf-v1`, but that mode requires exactly one logical line and is not applicable to this multi-line bash command;
- UTF-8 encodes `$runCommand`;
- computes SHA256 over those exact UTF-8 bytes;
- stores `head`, `commit_subject`, `command_sha256`, `saved_at`, `command`, and `kaggle_repo` in the registry;
- writes the registry JSON outside the Git repository;
- reports `HASH : <commandHash>` after saving.

No `%%bash` line is present in the authority canonical command. No `%%bash` normalization should occur for v2 registration.

`cm.ps1` must not be modified for this recovery.

## 13. Clipboard Round-Trip Probe

A deterministic non-registering clipboard round-trip probe was authorized and performed. It did not call `cm run save`, `cm run`, `cm kaggle`, Kaggle, or the preflight script.

Probe mechanism:

- extracted the authority canonical command from Git object `4726f2ab1540f6fe6148e89d11200ac0469f286f:reports/longterm_o0c_runtime_source_provenance_preflight_execution_authority_spec_candidate.md`;
- removed only the ` ```bash` and closing ` ``` ` fence lines;
- joined command lines with LF and no added final LF;
- verified authority SHA256 `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92` and `3505` bytes;
- saved the existing text clipboard in memory;
- wrote the exact command with `Set-Clipboard -Value $command`;
- immediately read `Get-Clipboard -Raw`;
- applied `.Trim()` exactly as the live `cm.ps1` run-save path does;
- UTF-8 encoded the registry-input string and compared it byte-for-byte with the authority canonical bytes;
- restored the prior text clipboard content.

Clipboard raw/read-back facts:

- raw SHA256: `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`;
- raw bytes: `3505`;
- raw CR count: `0`;
- raw LF count: `107`;
- raw ends with LF: `False`;
- raw ends with CR: `False`.

Registry-input facts after `(Get-Clipboard -Raw).Trim()`:

- registry-input SHA256: `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`;
- registry-input bytes: `3505`;
- registry-input CR count: `0`;
- registry-input LF count: `107`;
- registry-input ends with LF: `False`;
- registry-input ends with CR: `False`;
- registry input == authority canonical command: `TRUE`.

Future v2 registration is authorized only if the same byte-preservation checks pass immediately before `cm run save`.

Required successful registry identity:

- SHA256 `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`;
- bytes `3505`;
- CR count `0`.

If the intended clipboard method yields CRLF or any other byte difference:

`BLOCKED_CLIPBOARD_COMMAND_BYTES_NOT_PRESERVED`

Do not:

- accept a CRLF registry command;
- change the authority command to CRLF;
- silently normalize after registration;
- change the expected hash to a transport-mutated form.

The registry bytes become the Kaggle `.command.sh` bytes, so CRLF would not be the frozen LF bash command.

## 14. Safe Future Registration Procedure

The safe registration procedure must prevent recurrence of clipboard overwrite.

The future procedure must be performed in one local PowerShell invocation, so no second command block is copied after clipboard setup. It must explicitly select and enter the frozen execution worktree before any Git identity check, dirty-state check, authority extraction, clipboard write, `cm run save`, or `cm run`. It must programmatically extract the command, set the clipboard, verify the live registry input, invoke `cm run save`, verify the saved hash, and only then invoke `cm run`.

The frozen execution worktree path is:

`C:\o0c-preflight-de874a2`

The procedure must not create a replacement worktree. If this path is missing or is not a Git worktree/repository, it must block and return to controller review or normal `cm kaggle` preparation.

The authority extraction source remains the already validated deterministic Git object:

`4726f2ab1540f6fe6148e89d11200ac0469f286f:reports/longterm_o0c_runtime_source_provenance_preflight_execution_authority_spec_candidate.md`

The execution worktree may also contain the same authority path, but the command extraction below intentionally uses the explicit frozen authority Git object so the command identity remains `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92` / `3505` bytes / `CR=0` / `LF=107`.

Concrete one-invocation future procedure:

```powershell
$ErrorActionPreference = 'Stop'
$runName = 'longterm-o0c-runtime-source-provenance-preflight-de874a2-v2'
$executionWorktree = 'C:\o0c-preflight-de874a2'
$executionCommit = 'de874a22df4f60adbdc5efbcf294961c7b3a48a5'
$authorityCommit = '4726f2ab1540f6fe6148e89d11200ac0469f286f'
$authorityFile = 'reports/longterm_o0c_runtime_source_provenance_preflight_execution_authority_spec_candidate.md'
$expectedCommandSha256 = 'bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92'
$expectedCommandBytes = 3505
$expectedCommandLfCount = 107
$oldLocation = Get-Location
$oldRepoRoot = [string]$env:CONTRAMAMBA_REPO_ROOT
$oldNativePreference = $PSNativeCommandUseErrorActionPreference

try {
    if (-not (Test-Path -LiteralPath $executionWorktree -PathType Container)) {
        throw "Execution worktree does not exist: $executionWorktree"
    }

    $env:CONTRAMAMBA_REPO_ROOT = $executionWorktree
    Set-Location -LiteralPath $executionWorktree

    $insideWorkTree = git rev-parse --is-inside-work-tree
    if ($LASTEXITCODE -ne 0 -or $insideWorkTree -ne 'true') {
        throw "Execution path is not a Git worktree/repository: $executionWorktree"
    }

    $actualHead = git rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $actualHead -ne $executionCommit) {
        throw "Execution worktree HEAD mismatch: $actualHead"
    }

    $status = git status --porcelain
    if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect execution worktree status' }
    if ($status) { throw 'Execution worktree is not clean' }

    $localAuthorityPathExists = Test-Path -LiteralPath $authorityFile -PathType Leaf
    Write-Host "LOCAL_AUTHORITY_PATH_EXISTS: $localAuthorityPathExists"
    Write-Host "AUTHORITY_SOURCE: ${authorityCommit}:${authorityFile}"

    $authorityText = git show "${authorityCommit}:${authorityFile}"
    if ($LASTEXITCODE -ne 0) { throw 'Unable to read frozen execution authority from Git object' }
    $authorityText = $authorityText -join "`n"

    $match = [regex]::Match($authorityText, '(?s)BEGIN_EXACT_COMMAND\n(.*?)\nEND_EXACT_COMMAND')
    if (-not $match.Success) { throw 'BEGIN_EXACT_COMMAND / END_EXACT_COMMAND block not found' }

    $block = $match.Groups[1].Value
    $lines = $block -split "`n"
    if ($lines[0] -ne '```bash') { throw "Unexpected opening fence: $($lines[0])" }
    if ($lines[$lines.Length - 1] -ne '```') { throw "Unexpected closing fence: $($lines[$lines.Length - 1])" }

    $command = ($lines[1..($lines.Length - 2)] -join "`n")
    $commandBytes = [System.Text.Encoding]::UTF8.GetBytes($command)
    $commandSha256 = [System.BitConverter]::ToString(
        [System.Security.Cryptography.SHA256]::HashData($commandBytes)
    ).Replace('-', '').ToLowerInvariant()

    if ($commandSha256 -ne $expectedCommandSha256) { throw "Authority command SHA mismatch: $commandSha256" }
    if ($commandBytes.Length -ne $expectedCommandBytes) { throw "Authority command byte mismatch: $($commandBytes.Length)" }
    if ($command.Contains("`r")) { throw 'Authority command contains CR' }
    if ((([regex]::Matches($command, "`n")).Count) -ne $expectedCommandLfCount) {
        throw "Authority command LF count mismatch: $(([regex]::Matches($command, "`n")).Count)"
    }

    Set-Clipboard -Value $command

    $registryInput = (Get-Clipboard -Raw).Trim()
    $registryBytes = [System.Text.Encoding]::UTF8.GetBytes($registryInput)
    $registrySha256 = [System.BitConverter]::ToString(
        [System.Security.Cryptography.SHA256]::HashData($registryBytes)
    ).Replace('-', '').ToLowerInvariant()

    if ($registrySha256 -ne $expectedCommandSha256) { throw "Clipboard registry-input SHA mismatch: $registrySha256" }
    if ($registryBytes.Length -ne $expectedCommandBytes) { throw "Clipboard registry-input byte mismatch: $($registryBytes.Length)" }
    if ($registryInput.Contains("`r")) { throw 'Clipboard registry input contains CR' }
    if ((([regex]::Matches($registryInput, "`n")).Count) -ne $expectedCommandLfCount) {
        throw "Clipboard registry-input LF count mismatch: $(([regex]::Matches($registryInput, "`n")).Count)"
    }
    if (-not [System.Linq.Enumerable]::SequenceEqual([byte[]]$commandBytes, [byte[]]$registryBytes)) {
        throw 'Clipboard registry input differs from authority canonical command bytes'
    }

    $PSNativeCommandUseErrorActionPreference = $false
    $saveOutput = cm run save $runName 2>&1
    $saveOutput | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw 'cm run save failed' }
    if (-not (($saveOutput -join "`n") -match "HASH\s*:\s*$expectedCommandSha256")) {
        throw 'cm run save did not report the expected command hash'
    }

    cm run $runName
    if ($LASTEXITCODE -ne 0) { throw 'cm run failed' }
}
finally {
    $PSNativeCommandUseErrorActionPreference = $oldNativePreference
    if ($null -eq $oldRepoRoot -or $oldRepoRoot -eq '') {
        Remove-Item Env:\CONTRAMAMBA_REPO_ROOT -ErrorAction SilentlyContinue
    }
    else {
        $env:CONTRAMAMBA_REPO_ROOT = $oldRepoRoot
    }
    Set-Location -LiteralPath $oldLocation
}
```

No intervening clipboard operation is allowed after `Set-Clipboard` and before `cm run save`.

This authoring task does not run the procedure.

## 15. Kaggle Bootstrap Reuse

The existing Kaggle repository was successfully bootstrapped before v1:

`EXPECTED_COMMIT: de874a22df4f60adbdc5efbcf294961c7b3a48a5`

`ACTUAL_COMMIT: de874a22df4f60adbdc5efbcf294961c7b3a48a5`

and was clean before wrapper execution.

The malformed command did not successfully enter the Windows local path and did not invoke `cm`; it failed as bash.

Recovery authority may reuse the existing Kaggle bootstrap only if the v2 pinned wrapper independently verifies before execution:

- exact HEAD `de874a22df4f60adbdc5efbcf294961c7b3a48a5`;
- clean Kaggle repository.

If either condition fails, stop and return to normal `cm kaggle` bootstrap handling.

Do not use `cm kaggle fresh` merely because v1 failed.

## 16. Output Collision Policy

The v1 malformed command never invoked the preflight script.

Nevertheless, the v2 frozen command retains the original fail-closed output collision check for:

`reports/longterm_o0c_runtime_source_provenance_preflight.json`

If this artifact exists for any reason before v2:

`BLOCK`

Do not delete it automatically.

## 17. Semantic Contract Unchanged

The replacement v2 semantic contract remains exactly:

Python:

`3.12.13`

NumPy:

`2.0.2`

torch:

`2.10.0+cpu`

Transformers:

`5.0.0`

Output:

`reports/longterm_o0c_runtime_source_provenance_preflight.json`

PASS:

`PASS_SOURCE_IDENTITY_FROZEN`

All frozen `BLOCKED_*` statuses and fail-closed semantics remain unchanged.

## 18. CPU / Model / Package Boundary

Replacement v2 remains:

- CPU-only;
- Kaggle Accelerator=None;
- GPU OFF.

No:

- model;
- tokenizer;
- dataset;
- forward;
- generation;
- training;
- evaluation;
- CUDA;
- pip;
- conda;
- package/environment repair;
- implementation mutation.

## 19. Recovery Success Condition

This recovery authority succeeds when v2 is correctly registered with:

- run name `longterm-o0c-runtime-source-provenance-preflight-de874a2-v2`;
- exact execution HEAD `de874a22df4f60adbdc5efbcf294961c7b3a48a5`;
- exact recovered frozen command SHA256 `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`;
- exact recovered frozen command byte count `3505`;
- exact recovered frozen command bytes extracted from the frozen authority.

Only then may the v2 pinned Kaggle cell be generated.

This authoring task itself does not register or execute v2.

## 20. Future V2 Execution Interpretation

Once correctly registered, v2 becomes the replacement authorized semantic preflight attempt.

If v2 executes the frozen Python script and emits a semantic/provenance `BLOCKED_*`, that is no longer a transport failure and must be interpreted under the original frozen execution authority.

If v2 emits `PASS_SOURCE_IDENTITY_FROZEN`, validate/import under the original authority.

Do not create v3 automatically.

## 21. Existing V1 Evidence

Preserve the distinction:

`v1: PRE-SEMANTIC COMMAND-TRANSPORT FAILURE`

`v2: replacement semantic attempt, only after correct command registration`

Do not merge their logs or identities.

Do not claim v1 tested Transformers `5.0.0` static semantics.

## 22. Authoring Delta

This task creates exactly one task-attributable file:

`reports/longterm_o0c_runtime_source_provenance_preflight_command_transport_recovery_execution_authority_spec_candidate.md`

No other task-attributable file change is authorized.

Do not modify:

- `cm.ps1`;
- original execution authority;
- implementation;
- tests;
- any O0b/O0c scientific report;
- run registry;
- Kaggle;
- protected temp state.

## 23. Validation Record Required

Before authoring, verify:

- canonical HEAD equals `4726f2ab1540f6fe6148e89d11200ac0469f286f`;
- nothing staged;
- original execution-authority file exists and is tracked;
- frozen script identity matches SHA256 `73ffff2681928389933867caa9586b3f54260b2f64bdb1a44c50b77e4accdbfc` and bytes `32034`.

After authoring, verify:

- candidate SHA256 and byte count;
- recovered frozen command SHA256 `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92` and byte count `3505`;
- clipboard round-trip probe facts;
- registry-input SHA256, byte count, CR/LF facts, and byte equality result;
- recovered command differs from erroneous v1 command SHA;
- `git diff --check`;
- `git diff --name-status`;
- `git diff --cached --name-status`;
- `git status --short`;
- task-attributable delta exactly one untracked candidate;
- nothing staged.

Training/evaluation allowed:

`NO`

Scientific execution:

`NO`

Preflight execution:

`NO`

Kaggle execution:

`NO`

`cm run save` / `cm run`:

`NO`

Package/environment mutation:

`NO`

Implementation:

`NO`

`cm.ps1` modification:

`NO`

Commit/push:

`NO`

## 24. Stop Conditions

Stop and report `BLOCKED` if:

- original authority command extraction is ambiguous;
- authority canonical command does not equal SHA256 `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92` and `3505` bytes;
- recovered exact command cannot be reproduced byte-for-byte;
- recovered command unexpectedly equals erroneous v1 PowerShell command SHA;
- clipboard round-trip cannot preserve exact LF/no-final-LF command bytes through the same `Get-Clipboard -Raw` / `.Trim()` path;
- registry-input CR count is not `0`;
- v1 would need to be overwritten;
- recovery requires implementation/CLI/runtime changes;
- recovery requires `cm.ps1` change;
- any unrelated task-attributable delta is introduced.

## 25. Required Independent Verification Report Fields

An independent verifier should report:

1. Overall verdict: `PASS_READY_FOR_REVERIFICATION` or `BLOCKED`.
2. Canonical HEAD.
3. Candidate path.
4. Candidate SHA256 and bytes.
5. Frozen execution authority identity.
6. Frozen script SHA256 and bytes.
7. v1 failure classification.
8. v1 erroneous command SHA.
9. Exact recovered command SHA256 and bytes.
10. Proof recovered command is the exact frozen authority command.
11. Exact `cm.ps1` run-save behavior.
12. Clipboard probe mechanism used.
13. Clipboard raw/read-back facts.
14. Registry-input SHA256 and bytes.
15. Registry-input CR/LF facts.
16. Exact byte equality result.
17. Replacement run name.
18. Safe one-invocation clipboard/registration procedure, if byte equality is true.
19. Kaggle bootstrap reuse conditions.
20. Output collision policy.
21. Semantic contract unchanged.
22. Exact task-attributable state.
23. Confirmation that v1 was not overwritten or reused.
24. Confirmation that v2 was not registered by this authoring task.
25. Confirmation that nothing was staged, committed, or pushed.
26. Confirmation that no preflight/Kaggle/`cm run save`/`cm run`/model/tokenizer/training/evaluation/package mutation occurred.
