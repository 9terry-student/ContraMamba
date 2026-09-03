# Longterm O0c Runtime-Source Provenance Preflight Command-Transport Recovery PowerShell Hash Compatibility Correction Authority Spec Candidate

## 1. Verdict

`PASS_READY_FOR_INDEPENDENT_VERIFICATION`

This is a narrow PowerShell compatibility correction authority candidate for the already frozen ContraMamba O0c preflight command-transport recovery procedure.

It authorizes exactly one procedure-level compatibility substitution in the frozen v2 registration procedure:

- replace executable future-procedure uses of `[System.Security.Cryptography.SHA256]::HashData(...)`;
- with the already proven compatible `[System.Security.Cryptography.SHA256]::Create().ComputeHash(...)` mechanism through the helper frozen in Section 4.

No other correction is authorized.

This authoring task did not register v2, execute v2, run the preflight, invoke Kaggle, call `cm run save`, call `cm run`, mutate packages, mutate implementation, stage, commit, or push.

## 2. Authority Chain

Authority order used:

1. Current controller instruction for this task.
2. Frozen command-transport recovery authority: `a9f588c00cad90050dd0e38a3b52a2b03fab98ae`.
3. Original frozen O0c preflight execution authority: `4726f2ab1540f6fe6148e89d11200ac0469f286f`.
4. Frozen implementation: `de874a22df4f60adbdc5efbcf294961c7b3a48a5`.
5. Repository runbooks and `AGENTS.md`.

Canonical repository:

`C:\Users\Home1\Desktop\ContraMamba`

Canonical HEAD verified before authoring:

`a9f588c00cad90050dd0e38a3b52a2b03fab98ae`

Nothing was staged before authoring.

## 3. Failure / Blocker Classification

The newly observed blocker is frozen as:

`PRE-REGISTRATION LOCAL POWERSHELL/.NET API COMPATIBILITY BLOCKER`

Observed actual local error from the same machine:

`[System.Security.Cryptography.SHA256] has no method named HashData`

Exception class:

`MethodNotFound / RuntimeException`

The blocker occurred earlier when a local commit-verification block used:

```powershell
[System.Security.Cryptography.SHA256]::HashData($bytes)
```

No v2 registration occurred.

No v2 execution occurred.

No preflight semantics executed.

This blocker must not be classified as:

- command-byte mismatch;
- clipboard fidelity failure;
- `cm.ps1` failure;
- Kaggle failure;
- preflight `BLOCKED_*`;
- scientific evidence.

## 4. Exact Allowed Substitution

The corrected future procedure must define one helper equivalent to:

```powershell
function Get-Sha256Hex {
    param([byte[]]$Bytes)

    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $hasher.ComputeHash($Bytes)
    }
    finally {
        $hasher.Dispose()
    }

    return ([System.BitConverter]::ToString($digest)).Replace('-', '').ToLowerInvariant()
}
```

The corrected future procedure must use this helper for both:

- `$commandBytes`;
- `$registryBytes`.

No use of `[System.Security.Cryptography.SHA256]::HashData(...)` is permitted in the executable future procedure.

The intended substitutions are exactly:

```powershell
$commandSha256 = Get-Sha256Hex -Bytes $commandBytes
```

and:

```powershell
$registrySha256 = Get-Sha256Hex -Bytes $registryBytes
```

## 5. Frozen Recovery Authority Defect

Frozen recovery authority read:

Commit:

`a9f588c00cad90050dd0e38a3b52a2b03fab98ae`

File:

`reports/longterm_o0c_runtime_source_provenance_preflight_command_transport_recovery_execution_authority_spec_candidate.md`

The concrete one-invocation future procedure in that frozen recovery authority uses:

```powershell
[System.Security.Cryptography.SHA256]::HashData($commandBytes)
```

for the authority command hash, and:

```powershell
[System.Security.Cryptography.SHA256]::HashData($registryBytes)
```

for the clipboard registry-input hash.

This exact static API is not supported by the user's observed local PowerShell/.NET runtime for the earlier commit-verification block. That is the only currently authorized defect.

No other defect in the frozen recovery authority is presently authorized for correction.

## 6. Required Identity Preservation

The substitution is implementation-equivalent hashing only.

It must still produce exactly:

Authority canonical command SHA256:

`bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`

Authority canonical command bytes:

`3505`

Authority canonical command CR count:

`0`

Authority canonical command LF count:

`107`

Registry input must remain byte-for-byte equal to the authority canonical command bytes.

Do not change:

- authority canonical command bytes;
- expected command SHA;
- expected byte count;
- expected LF count;
- expected CR count.

## 7. V1 / V2 Boundary

Preserve exactly:

v1 run:

`longterm-o0c-runtime-source-provenance-preflight-de874a2-v1`

v1 command SHA256:

`50a2bdcca75266a8fa3cc561e6d5febb7789f1bcb28ae37da56015e4562039d3`

v1 classification:

`PRE-SEMANTIC COMMAND-TRANSPORT FAILURE`

v2 run:

`longterm-o0c-runtime-source-provenance-preflight-de874a2-v2`

v2 execution commit:

`de874a22df4f60adbdc5efbcf294961c7b3a48a5`

No v1 overwrite is authorized.

No automatic v3 is authorized.

## 8. Frozen Semantic Contract

Do not change the frozen script:

`scripts/preflight_longterm_o0c_runtime_source_provenance.py`

Script SHA256:

`73ffff2681928389933867caa9586b3f54260b2f64bdb1a44c50b77e4accdbfc`

Script bytes:

`32034`

Runtime contract:

- Python `3.12.13`;
- NumPy `2.0.2`;
- torch `2.10.0+cpu`;
- Transformers `5.0.0`.

Output:

`reports/longterm_o0c_runtime_source_provenance_preflight.json`

PASS status:

`PASS_SOURCE_IDENTITY_FROZEN`

CPU only / GPU OFF remains required.

All preflight semantics, fail-closed statuses, provenance checks, runtime checks, source checks, output collision behavior, artifact schema, and PASS/BLOCKED interpretation remain unchanged.

## 9. Procedure Preservation

Except for SHA256 implementation compatibility, preserve the frozen recovery procedure exactly in meaning and ordering:

- `$executionWorktree = 'C:\o0c-preflight-de874a2'`;
- `CONTRAMAMBA_REPO_ROOT` set to that worktree;
- `Set-Location` before HEAD/status/authority/clipboard/`cm`;
- exact HEAD `de874a22df4f60adbdc5efbcf294961c7b3a48a5`;
- clean worktree;
- authority source `4726f2ab1540f6fe6148e89d11200ac0469f286f:reports/longterm_o0c_runtime_source_provenance_preflight_execution_authority_spec_candidate.md`;
- normative extraction from `BEGIN_EXACT_COMMAND` / `END_EXACT_COMMAND`;
- `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92` / `3505` / CR `0` / LF `107` checks;
- `Set-Clipboard`;
- immediate `Get-Clipboard -Raw` plus `.Trim()`;
- byte equality;
- `cm run save longterm-o0c-runtime-source-provenance-preflight-de874a2-v2`;
- saved hash verification;
- `cm run longterm-o0c-runtime-source-provenance-preflight-de874a2-v2`;
- `finally` restoration;
- `PSNativeCommandUseErrorActionPreference` session-local only.

Do not alter:

- command bytes;
- hashes;
- run names;
- execution commits;
- clipboard semantics;
- registry semantics;
- Kaggle semantics;
- preflight semantics.

## 10. Local Non-Registering Compatibility Probe

A local non-registering compatibility probe was run during this authoring task.

It did not call:

- `cm run save`;
- `cm run`;
- `cm kaggle`;
- Kaggle;
- the preflight script.

The probe extracted the canonical command from:

`4726f2ab1540f6fe6148e89d11200ac0469f286f:reports/longterm_o0c_runtime_source_provenance_preflight_execution_authority_spec_candidate.md`

using the frozen extraction rule:

- content between `BEGIN_EXACT_COMMAND` and `END_EXACT_COMMAND`;
- remove only the opening ```` ```bash ```` and closing ```` ``` ```` fence lines;
- encode as UTF-8;
- LF line endings;
- no added final LF.

Replacement helper result for authority canonical command:

- SHA256: `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`;
- bytes: `3505`;
- CR count: `0`;
- LF count: `107`.

Replacement helper result for clipboard registry-input bytes after `Set-Clipboard`, immediate `Get-Clipboard -Raw`, and `.Trim()`:

- SHA256: `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`;
- bytes: `3505`;
- CR count: `0`;
- LF count: `107`.

Byte equality:

`TRUE`

If a future independent verifier cannot reproduce these values with `SHA256.Create().ComputeHash(...)`, the correction is blocked.

## 11. API Compatibility Spot Check

Local non-mutating spot-check results:

- `Set-Clipboard`: available and successfully used in the non-registering probe;
- `Get-Clipboard`: available and successfully used in the non-registering probe;
- `[System.Linq.Enumerable]::SequenceEqual`: available and returned `TRUE`;
- `[System.Security.Cryptography.SHA256]::Create().ComputeHash(...)`: available and reproduced both expected hashes.

No redesign of these APIs is authorized.

If another required API is demonstrably unavailable in the actual future execution shell, report `BLOCKED` rather than widening this correction.

## 12. Exact Delta

This task creates exactly one task-attributable untracked candidate:

`reports/longterm_o0c_runtime_source_provenance_preflight_command_transport_recovery_powershell_hash_compatibility_correction_authority_spec_candidate.md`

No modification is authorized to:

- frozen recovery authority;
- original execution authority;
- `cm.ps1`;
- run registry;
- v1;
- implementation;
- tests;
- Kaggle;
- protected unrelated state.

## 13. Validation Record

Before authoring:

- HEAD exact: `a9f588c00cad90050dd0e38a3b52a2b03fab98ae`;
- nothing staged.

Authoring-time compatibility probe:

- independent canonical command SHA256 via `ComputeHash`: `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`;
- independent canonical command bytes: `3505`;
- independent canonical command CR count: `0`;
- independent canonical command LF count: `107`;
- clipboard registry-input SHA256 via `ComputeHash`: `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`;
- clipboard registry-input bytes: `3505`;
- clipboard registry-input CR count: `0`;
- clipboard registry-input LF count: `107`;
- `SequenceEqual` availability/result: available / `TRUE`;
- clipboard API availability/result: available / successful round trip;
- `SHA256.Create().ComputeHash(...)` availability/result: available / reproduced expected hashes.

After-authoring validation must report:

- candidate SHA256 and bytes;
- `git diff --check`;
- `git diff --name-status`;
- `git diff --cached --name-status`;
- `git status --short`.

## 14. Explicit Non-Execution Attestation

Training/evaluation allowed:

`NO`

Preflight execution:

`NO`

Kaggle execution:

`NO`

`cm run save` / `cm run`:

`NO`

Package/environment mutation:

`NO`

`cm.ps1` modification:

`NO`

Implementation/test modification:

`NO`

Commit/push:

`NO`

No v2 registration occurred.

No v2 execution occurred.

No preflight semantics executed.

## 15. Required Independent Verification Report Fields

An independent verifier should report:

1. Verdict: `PASS_READY_FOR_INDEPENDENT_VERIFICATION` or `BLOCKED`.
2. HEAD.
3. Candidate SHA256 and bytes.
4. Frozen recovery authority identity.
5. Exact unsupported API finding.
6. Exact replacement helper.
7. Canonical command SHA/bytes via `ComputeHash`.
8. Registry-input SHA/bytes via `ComputeHash`.
9. `SequenceEqual` availability/result.
10. Clipboard API availability/result.
11. Confirmation that no semantic/provenance contract changed.
12. Exact task-attributable state.
13. Confirmation that nothing was staged, committed, or pushed.
14. Confirmation that no v2 registration or execution occurred.

## 16. Next Authorized Action

The next authorized action is independent verification of this candidate.

Only after independent verification and controller activation may the future v2 registration/execution procedure be re-issued with the `Get-Sha256Hex` compatibility helper substituted for both executable SHA256 hash sites.
