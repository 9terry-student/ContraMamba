# P3-W7 Seed8192 A0 Seed180 R2 Stage-L PowerShell Scalar-Output Correction Command Freeze Candidate

## Verdict and authority lineage

`PASS_READY_FOR_INDEPENDENT_SEED8192_A0_SEED180_R2_STAGE_L_POWERSHELL_SCALAR_OUTPUT_CORRECTION_FREEZE_VERIFICATION`

This v2 successor is authorized by recovery-handoff authority
`48d258fc5e8b09e163e9252f33c86936244ef872`, local-preflight boundary
correction `7e8b909e57c0e716f5154cc1d3083c06ef8f2d5a`, Stage-L/K freeze
`bdb8202331f20dfb850924912133c3dd8871feb2`, and the active Windows
PowerShell Stage-L successor freeze
`a8f7a968ce7493002ef04061cccf70e4332f5768`. The required repository HEAD
was exactly `a8f7a968ce7493002ef04061cccf70e4332f5768` before authoring, with
zero staged files and zero tracked unstaged modifications.

The frozen v1 successor is the committed report
`reports/reason_router_p3w7_seed8192_a0_seed180_r2_provenance_recovery_stage_l_windows_powershell_compatibility_correction_command_freeze_candidate.md`,
whose Git blob is `9d810afb3d7897d9ce43ff476f4a445638eedefe`. Its extracted
ready-to-copy payload has `BYTES=6301` and
`SHA256=0fd716a3ea98ec1f09304ca351217c289aa5623b03782deeece758f444203343`.

The observed attempted Stage-L precheck failed at:

```powershell
if ((Invoke-CheckedGit @(
   '-C', $authorityPath,
   'rev-parse', '--is-inside-work-tree'
))[0] -ne 'true') {
   throw "Authority path is not a Git worktree: $authorityPath"
}
```

with `Authority path is not a Git worktree: C:\w\r2a`. This was an
output-shape/indexing defect: a single pipeline result is a scalar
`System.String` at the caller, so `[0]` indexes the first character. It is
not evidence that `C:\w\r2a` is not a worktree.

## V2 ready-to-copy PowerShell payload

The only substantive v1-to-v2 delta is array-subexpression wrapping before
`[0]` at the three single-result `Invoke-CheckedGit` consumers. `Get-Sha256Hex`
is text-identical to v1, including its classic-.NET `SHA256.Create()`,
`ComputeHash()`, `Dispose()`, and `BitConverter` formatting body. The
remote-config consumer deliberately remains unwrapped and multi-line.

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
if ((@(Invoke-CheckedGit @('-C', $authorityPath, 'rev-parse', '--is-inside-work-tree')))[0] -ne 'true') {
    throw "Authority path is not a Git worktree: $authorityPath"
}
$gitTopLevel = (@(Invoke-CheckedGit @('-C', $authorityPath, 'rev-parse', '--show-toplevel')))[0]
if ((Resolve-Path -LiteralPath $gitTopLevel -ErrorAction Stop).Path -ne $authorityPath) {
    throw "Authority worktree argument must resolve to its Git top level: $authorityPath"
}
$remoteConfig = Invoke-CheckedGit @('-C', $authorityPath, 'config', '--get-regexp', '^remote\..*\.url$')
if (-not (@($remoteConfig | Where-Object { $_ -match '(?i)contramamba' }).Count -gt 0)) {
    throw "Authority worktree is not configured for ContraMamba: $authorityPath"
}
$authorityHead = (@(Invoke-CheckedGit @('-C', $authorityPath, 'rev-parse', 'HEAD')))[0]
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

## Exact payload identity and isolated validation

The extracted v2 fence is UTF-8 without BOM, has no CR bytes, and has exactly
one final LF. `PAYLOAD_BYTES=6310` and
`PAYLOAD_SHA256=c31a811f396bcd6e1d0eb321be381cb678c676c89ac9bb6e88702ff7e248aec4`.

The PowerShell parser was run on the extracted v2 payload and reported
`PARSER_ERRORS=0`. No controller was invoked. The isolated
`Invoke-CheckedGit` function was run only with the listed read-only Git
commands against `C:\w\r2a`:

- Raw `git -C C:\w\r2a rev-parse --is-inside-work-tree` returned `true`.
- The v1 expression `(Invoke-CheckedGit ...)[0]` reproduced `t`.
- The v2 expression `(@(Invoke-CheckedGit ...))[0]` returned complete `true`.
- The corrected show-toplevel expression returned complete `C:/w/r2a`, the
  normal Windows Git path spelling of `C:\w\r2a`.
- The corrected HEAD expression returned complete 40-character
  `48d258fc5e8b09e163e9252f33c86936244ef872`.
- The unmodified remote-config consumer returned one output line,
  `remote.origin.url https://github.com/9terry-student/ContraMamba.git`; its
  ContraMamba match count was one.

Static normalized-line comparison found exactly three changed source lines:
the three `rev-parse` single-result consumers above. All authority gates,
explicit identities, helper and registry handling, UTF-8 registry-command
recomputation, Stage-K byte binding, controller PASS fields, read-only
behavior, and the complete `Get-Sha256Hex` body are otherwise identical to
v1. In-memory normalized payload comparison found no other source changes.
`git diff --no-index --check` reported its expected difference exit status
without whitespace diagnostics.

Stage-K remains unchanged: `REPORT_BLOB=65ea058f122a4ec9cdc07fa4538d916c91350b90`,
`PAYLOAD_BYTES=11386`, and
`PAYLOAD_SHA256=e4e7f9e8a15082b7b0a93faefaf5d60245674a65360432c204f2131b498a309c`.

The report's final byte count, SHA256, and Git blob are necessarily recorded
by the independent verifier after this report is written: embedding a final
self-hash or self-blob in the bytes being hashed would change that identity.
The final repository delta is exactly this one untracked report; no tracked
or staged files were changed.

NOT EXECUTED: full Stage L; Stage K; Kaggle; recovery; real import; training;
evaluation; commit; push.
