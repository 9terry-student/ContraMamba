# P3-W7 Seed8192 A0 Seed180 R2 Stage-L Local-Preflight Command Freeze Candidate

## Verdict and authority lineage

`PASS_READY_FOR_INDEPENDENT_SEED8192_A0_SEED180_R2_STAGE_L_COMMAND_FREEZE_VERIFICATION`

This report-only candidate freezes the exact Stage-L Windows PowerShell
local-control-plane preflight command required by the active recovery-handoff
authority `48d258fc5e8b09e163e9252f33c86936244ef872` and the activated boundary
correction authority `7e8b909e57c0e716f5154cc1d3083c06ef8f2d5a`.

The command is intentionally authored on the boundary-correction checkout, but
it must **not** execute there. At a later separately authorized execution, its
first mandatory argument is the explicit path to a valid ContraMamba authority
worktree pinned at exactly `48d258fc5e8b09e163e9252f33c86936244ef872`. Its
second mandatory argument is the explicit path to the independently frozen
Stage-K payload file. That payload may be outside the authority worktree and is
authenticated as file bytes only; this command neither reconstructs, parses,
nor executes it.

## Frozen identities and execution boundary

| Binding | Exact value |
| --- | --- |
| Authority-worktree HEAD | `48d258fc5e8b09e163e9252f33c86936244ef872` |
| Run name | `p3w7-seed8192-a0-seed180-r2` |
| Registry entry HEAD | `abd85a088c274678004432160625d42208112848` |
| Registered/recomputed command SHA256 | `82be0c377e305228609e8ce9f75a6b3e8b8a6f83f9999a1543914f9fc517c1f4` |
| Helper bytes / SHA256 | `91498` / `09097e460ce9f05d5ead09ad3ee9499ac6b9da4298d0a69e8abde6006e5facae` |
| Stage-K payload bytes / SHA256 | `11386` / `e4e7f9e8a15082b7b0a93faefaf5d60245674a65360432c204f2131b498a309c` |

The exact ready-to-copy payload is the one and only designated code fence in
this report:

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
    return [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
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

## Exact-byte freeze

The fenced PowerShell payload is UTF-8, has no UTF-8 BOM, and has exactly one
final LF. Its exact byte count is `6164` and its SHA256 is
`939835a040905cd816b45f28218356c8cdd32a169461a9a1d7ae3dcad4a2f0a4`.
Those values are computed from the independently extracted fence bytes after
this report is written; no Markdown reconstruction is used by the payload
itself.

## Static scope assessment

The payload only resolves paths, invokes read-only Git inspection commands,
reads helper/registry/payload bytes, parses the registry JSON, hashes bytes,
and emits its controller block after all gates pass. It contains no `cm run`,
`cm collect`, `cm import`, Kaggle, checkout, reset, clean, stash, add, commit,
push, ZIP, wrapper-evidence, artifact, checkpoint, training, or evaluation
operation. It deliberately permits untracked files in the authority worktree:
Git status is requested with `-uno`, while staged and tracked-unstaged states
are both fail-closed.

Execution NOT PERFORMED: Stage L; Stage K; recovery; real import; Kaggle;
training; evaluation; commit; push.
