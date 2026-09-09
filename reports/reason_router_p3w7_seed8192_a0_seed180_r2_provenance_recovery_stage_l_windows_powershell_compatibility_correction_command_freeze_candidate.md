# P3-W7 Seed8192 A0 Seed180 R2 Stage-L Windows PowerShell Compatibility Correction Command Freeze Candidate

## Verdict and authority lineage

`PASS_READY_FOR_INDEPENDENT_SEED8192_A0_SEED180_R2_STAGE_L_WINDOWS_POWERSHELL_COMPATIBILITY_CORRECTION_FREEZE_VERIFICATION`

This successor is authorized by recovery-handoff authority
`48d258fc5e8b09e163e9252f33c86936244ef872`, local-preflight boundary
correction `7e8b909e57c0e716f5154cc1d3083c06ef8f2d5a`, and Stage-L/K freeze
`bdb8202331f20dfb850924912133c3dd8871feb2`.  It supersedes the historical
Stage-L freeze **FOR EXECUTION DUE TO HOST RUNTIME COMPATIBILITY** only.  The
historical report remains byte-identical and valid as a historical freeze:
`REPORT_BLOB=4efbe3ac701dad3fa42bad7535d2c436ef41489e`,
`OLD_PAYLOAD_BYTES=6164`, and
`OLD_PAYLOAD_SHA256=939835a040905cd816b45f28218356c8cdd32a169461a9a1d7ae3dcad4a2f0a4`.

The frozen Stage-L command was **NOT invoked**.  A host-compatibility check
failed before invocation because Windows PowerShell / classic .NET does not
provide `[System.Security.Cryptography.SHA256]::HashData(...)` or
`[System.Convert]::ToHexString(...)` (observed errors: SHA256 has no method
named `HashData`; Convert has no method named `ToHexString`).

## Successor ready-to-copy payload

The following is the one designated PowerShell fence.  Its only substantive
implementation delta from the historical payload is `Get-Sha256Hex`: it uses
classic .NET `SHA256.Create().ComputeHash()` and `BitConverter` formatting.
It hashes the unchanged exact input `byte[]`, returns lowercase 64-hex, and
introduces no normalization.

```powershell
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$AuthorityWorktree,
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$StageKPayloadPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-CheckedGit {
    param([string[]]$GitArgs)
    $output = @(& git @GitArgs 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed: $($output -join [Environment]::NewLine)"
    }
    return @($output | ForEach-Object { $_.ToString() })
}

function Get-Sha256Hex {
    param([byte[]]$Bytes)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hash = $sha.ComputeHash($Bytes)
    }
    finally {
        $sha.Dispose()
    }
    return ([System.BitConverter]::ToString($hash)).Replace('-', '').ToLowerInvariant()
}

function Resolve-RegularFile {
    param([string]$Path, [string]$Label)
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    $item = Get-Item -LiteralPath $resolved -Force -ErrorAction Stop
    if (($item -isnot [System.IO.FileInfo]) -or $item.PSIsContainer) {
        throw "$Label must be a regular file: $resolved"
    }
    return $item
}

$expectedAuthorityHead = '48d258fc5e8b09e163e9252f33c86936244ef872'
$expectedHelperBytes = 91498L
$expectedHelperSha256 = '09097e460ce9f05d5ead09ad3ee9499ac6b9da4298d0a69e8abde6006e5facae'
$expectedRunName = 'p3w7-seed8192-a0-seed180-r2'
$expectedRegistryHead = 'abd85a088c274678004432160625d42208112848'
$expectedCommandSha256 = '82be0c377e305228609e8ce9f75a6b3e8b8a6f83f9999a1543914f9fc517c1f4'
$expectedStageKPayloadBytes = 11386L
$expectedStageKPayloadSha256 = 'e4e7f9e8a15082b7b0a93faefaf5d60245674a65360432c204f2131b498a309c'

$authorityItem = Get-Item -LiteralPath $AuthorityWorktree -Force -ErrorAction Stop
if (-not $authorityItem.PSIsContainer) {
    throw "Authority worktree path is not a directory: $AuthorityWorktree"
}
$authorityPath = (Resolve-Path -LiteralPath $authorityItem.FullName -ErrorAction Stop).Path
if ((Invoke-CheckedGit @('-C', $authorityPath, 'rev-parse', '--is-inside-work-tree'))[0] -ne 'true') {
    throw "Authority path is not a Git worktree: $authorityPath"
}
$gitTopLevel = (Invoke-CheckedGit @('-C', $authorityPath, 'rev-parse', '--show-toplevel'))[0]
if ((Resolve-Path -LiteralPath $gitTopLevel -ErrorAction Stop).Path -ne $authorityPath) {
    throw "Authority worktree argument must resolve to its Git top level: $authorityPath"
}
$remoteConfig = Invoke-CheckedGit @('-C', $authorityPath, 'config', '--get-regexp', '^remote\..*\.url$')
if (-not (@($remoteConfig | Where-Object { $_ -match '(?i)contramamba' }).Count -gt 0)) {
    throw "Authority worktree is not configured for ContraMamba: $authorityPath"
}
$authorityHead = (Invoke-CheckedGit @('-C', $authorityPath, 'rev-parse', 'HEAD'))[0]
if ($authorityHead -ne $expectedAuthorityHead) {
    throw "Authority HEAD mismatch: $authorityHead"
}
$statusLines = Invoke-CheckedGit @('-C', $authorityPath, 'status', '--porcelain=v1', '-uno')
$stagedLines = @($statusLines | Where-Object { $_.Length -ge 2 -and $_[0] -ne ' ' })
$trackedModificationLines = @($statusLines | Where-Object { $_.Length -ge 2 -and $_[1] -ne ' ' })
if ($stagedLines.Count -ne 0) {
    throw "Authority worktree has staged files: $($stagedLines.Count)"
}
if ($trackedModificationLines.Count -ne 0) {
    throw "Authority worktree has tracked unstaged modifications: $($trackedModificationLines.Count)"
}

$helperItem = Resolve-RegularFile -Path (Join-Path $HOME '.contramamba/cm.ps1') -Label 'Helper'
$helperBytes = [System.IO.File]::ReadAllBytes($helperItem.FullName)
$helperSha256 = Get-Sha256Hex $helperBytes
if (($helperItem.Length -ne $expectedHelperBytes) -or ($helperSha256 -ne $expectedHelperSha256)) {
    throw "Helper identity mismatch: $($helperItem.FullName)"
}

$registryItem = Resolve-RegularFile -Path (Join-Path $HOME '.contramamba/run-registry.json') -Label 'Registry'
try {
    $registry = [System.IO.File]::ReadAllText($registryItem.FullName, [System.Text.UTF8Encoding]::new($false)) | ConvertFrom-Json -ErrorAction Stop
} catch {
    throw "Registry is not readable JSON: $($registryItem.FullName)"
}
$entryProperty = $registry.PSObject.Properties[$expectedRunName]
if ($null -eq $entryProperty) {
    throw "Registry entry is absent: $expectedRunName"
}
$entry = $entryProperty.Value
foreach ($propertyName in @('head', 'command_sha256', 'command')) {
    if ($null -eq $entry.PSObject.Properties[$propertyName]) {
        throw "Registry entry lacks $propertyName"
    }
}
if (($entry.head -isnot [string]) -or ($entry.command_sha256 -isnot [string]) -or ($entry.command -isnot [string])) {
    throw 'Registry entry fields must be JSON strings'
}
if (($entry.head -ne $expectedRegistryHead) -or ($entry.command_sha256 -ne $expectedCommandSha256)) {
    throw 'Registry frozen fields mismatch'
}
$runCommand = $entry.command
$recomputedCommandSha256 = Get-Sha256Hex ([System.Text.Encoding]::UTF8.GetBytes($runCommand))
if ($recomputedCommandSha256 -ne $expectedCommandSha256) {
    throw "Registry command SHA256 mismatch: $recomputedCommandSha256"
}

$stageKPayloadItem = Resolve-RegularFile -Path $StageKPayloadPath -Label 'Stage-K payload'
$stageKPayloadBytes = [System.IO.File]::ReadAllBytes($stageKPayloadItem.FullName)
$stageKPayloadSha256 = Get-Sha256Hex $stageKPayloadBytes
if (($stageKPayloadItem.Length -ne $expectedStageKPayloadBytes) -or ($stageKPayloadSha256 -ne $expectedStageKPayloadSha256)) {
    throw "Stage-K payload identity mismatch: $($stageKPayloadItem.FullName)"
}

@(
    'STAGE_L_PREFLIGHT=PASS'
    "AUTHORITY_WORKTREE=$authorityPath"
    "AUTHORITY_HEAD=$authorityHead"
    'STAGED_COUNT=0'
    'TRACKED_MODIFICATIONS=0'
    "HELPER_PATH=$($helperItem.FullName)"
    "HELPER_BYTES=$($helperItem.Length)"
    "HELPER_SHA256=$helperSha256"
    "REGISTRY_PATH=$($registryItem.FullName)"
    "RUN_NAME=$expectedRunName"
    "REGISTRY_HEAD=$($entry.head)"
    "REGISTERED_COMMAND_SHA256=$($entry.command_sha256)"
    "RECOMPUTED_COMMAND_SHA256=$recomputedCommandSha256"
    "STAGE_K_PAYLOAD_PATH=$($stageKPayloadItem.FullName)"
    "STAGE_K_PAYLOAD_BYTES=$($stageKPayloadItem.Length)"
    "STAGE_K_PAYLOAD_SHA256=$stageKPayloadSha256"
) | Write-Output
```

## Exact-byte successor freeze and validation

The fenced payload was independently extracted as UTF-8 bytes.  It has no BOM
and exactly one final LF.  Its exact byte count is `6301` and its SHA256 is
`0fd716a3ea98ec1f09304ca351217c289aa5623b03782deeece758f444203343`.
Stage-K remains unchanged:
`REPORT_BLOB=65ea058f122a4ec9cdc07fa4538d916c91350b90`,
`PAYLOAD_BYTES=11386`, and
`PAYLOAD_SHA256=e4e7f9e8a15082b7b0a93faefaf5d60245674a65360432c204f2131b498a309c`.

Windows PowerShell parser validation of the extracted successor reports zero
errors.  The corrected primitive was run alone (not Stage L) against exact
bytes `01 02 03` and returned
`039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81`;
it matches the required SHA-256 and the result matches `^[0-9a-f]{64}$`.
Static text/AST comparison finds the sole substantive implementation delta is
the compatibility body of `Get-Sha256Hex`; all arguments, gates, byte inputs,
expected identities, controller PASS fields, and read-only boundary are
otherwise retained.

Execution NOT PERFORMED: Stage L; Stage K; recovery; Kaggle; real import;
training; evaluation; commit; push.  No helper, registry, Git, artifact, or
checkpoint mutation was performed.
