Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Name)
    Write-Output ""
    Write-Output $Name
}

function Invoke-Git {
    param(
        [string[]]$GitArgs,
        [switch]$AllowFailure
    )
    try {
        $output = & git @GitArgs 2>$null
        if ($LASTEXITCODE -ne 0) {
            if ($AllowFailure) { return $null }
            throw "git $($GitArgs -join ' ') failed with exit code $LASTEXITCODE"
        }
        return $output
    } catch {
        if ($AllowFailure) { return $null }
        throw
    }
}

function Get-RepoRoot {
    $root = Invoke-Git -GitArgs @("rev-parse", "--show-toplevel") -AllowFailure
    if ($null -eq $root) { return $null }
    return ($root | Select-Object -First 1)
}

function Count-Lines {
    param($Value)
    if ($null -eq $Value) { return 0 }
    if ($Value -is [array]) { return $Value.Count }
    if ([string]::IsNullOrWhiteSpace([string]$Value)) { return 0 }
    return 1
}

function Get-OptionalProperty {
    param(
        $Object,
        [string]$Name
    )
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Get-GitRelation {
    param([string]$Head, [string]$Remote)
    if ([string]::IsNullOrWhiteSpace($Head) -or [string]::IsNullOrWhiteSpace($Remote)) {
        return "UNKNOWN"
    }
    $counts = Invoke-Git -GitArgs @("rev-list", "--left-right", "--count", "origin/main...HEAD") -AllowFailure
    if ($null -eq $counts) { return "UNKNOWN" }
    $parts = (($counts | Select-Object -First 1) -split "\s+")
    if ($parts.Count -lt 2) { return "UNKNOWN" }
    $behind = [int]$parts[0]
    $ahead = [int]$parts[1]
    if ($ahead -eq 0 -and $behind -eq 0) { return "ALIGNED" }
    if ($ahead -gt 0 -and $behind -eq 0) { return "AHEAD" }
    if ($ahead -eq 0 -and $behind -gt 0) { return "BEHIND" }
    if ($ahead -gt 0 -and $behind -gt 0) { return "DIVERGED" }
    return "UNKNOWN"
}

function Get-MarkdownSectionLines {
    param(
        [string[]]$Lines,
        [string]$Heading
    )
    $pattern = "^##\s+$([regex]::Escape($Heading))\s*$"
    $start = -1
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i] -match $pattern) {
            $start = $i + 1
            break
        }
    }
    if ($start -lt 0) { return @() }
    $end = $Lines.Count
    for ($i = $start; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i] -match "^##\s+") {
            $end = $i
            break
        }
    }
    return $Lines[$start..($end - 1)] | Where-Object { $_.Trim() -ne "" }
}

function Clean-MarkdownValue {
    param([string[]]$Lines)
    if ($null -eq $Lines -or $Lines.Count -eq 0) { return "UNKNOWN" }
    $joined = ($Lines | ForEach-Object {
        ($_ -replace "^\s*-\s*", "" -replace '`', "").Trim()
    }) -join " "
    return ($joined -replace "\s+", " ").Trim()
}

function Get-PathStatus {
    param(
        [string]$RepoRoot,
        [string]$RelativePath
    )
    $full = Join-Path $RepoRoot $RelativePath
    return [pscustomobject]@{
        FullPath = $full
        Present = Test-Path -LiteralPath $full
    }
}

function Get-HashStatus {
    param(
        [string]$Path,
        [string]$ExpectedSha256,
        [bool]$Present
    )
    if (-not $Present) { return "ABSENT" }
    if ([string]::IsNullOrWhiteSpace($ExpectedSha256)) { return "NOT_CHECKED" }
    try {
        $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -eq $ExpectedSha256.ToLowerInvariant()) { return "VERIFIED" }
        return "MISMATCH"
    } catch {
        return "NOT_CHECKED"
    }
}

$attention = New-Object System.Collections.Generic.List[string]

$repoRoot = Get-RepoRoot
if ($null -eq $repoRoot) {
    Write-Output "REPOSITORY"
    Write-Output "root: UNKNOWN"
    Write-Output "FINAL_SUMMARY: NEEDS_ATTENTION"
    exit 1
}

Set-Location -LiteralPath $repoRoot

$branch = Invoke-Git -GitArgs @("branch", "--show-current") -AllowFailure
if ($null -eq $branch -or [string]::IsNullOrWhiteSpace(($branch | Select-Object -First 1))) {
    $branch = Invoke-Git -GitArgs @("rev-parse", "--short", "HEAD") -AllowFailure
    $branch = "DETACHED@$($branch | Select-Object -First 1)"
} else {
    $branch = ($branch | Select-Object -First 1)
}

$head = Invoke-Git -GitArgs @("rev-parse", "HEAD") -AllowFailure
$head = if ($null -eq $head) { "UNKNOWN" } else { $head | Select-Object -First 1 }
$remote = Invoke-Git -GitArgs @("rev-parse", "origin/main") -AllowFailure
$remote = if ($null -eq $remote) { "UNKNOWN" } else { $remote | Select-Object -First 1 }
$relation = Get-GitRelation -Head $head -Remote $remote
if ($relation -eq "UNKNOWN") {
    $attention.Add("git relation to origin/main could not be determined")
}

$trackedModified = Count-Lines (Invoke-Git -GitArgs @("diff", "--name-only") -AllowFailure)
$staged = Count-Lines (Invoke-Git -GitArgs @("diff", "--cached", "--name-only") -AllowFailure)
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$porcelain = & git status --porcelain --untracked-files=all 2>$null
$ErrorActionPreference = $previousErrorActionPreference
$untracked = Count-Lines ($porcelain | Where-Object { $_ -like "?? *" })
$reviewPatchCount = @(Get-ChildItem -LiteralPath $repoRoot -Filter "reason_router_*.patch" -File -ErrorAction SilentlyContinue).Count
$stage180DuplicatePath = Join-Path $repoRoot "reports/stage180a_pass2_annotations_completed.csv"
$stage180DuplicatePresent = Test-Path -LiteralPath $stage180DuplicatePath

$statePath = Join-Path $repoRoot "reports/RESEARCH_STATE.md"
$manifestPath = Join-Path $repoRoot "reports/artifact_manifest.json"

$stateLines = $null
if (Test-Path -LiteralPath $statePath) {
    $stateLines = Get-Content -LiteralPath $statePath
} else {
    $attention.Add("reports/RESEARCH_STATE.md missing")
}

$manifest = $null
if (Test-Path -LiteralPath $manifestPath) {
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    } catch {
        $attention.Add("reports/artifact_manifest.json malformed")
    }
} else {
    $attention.Add("reports/artifact_manifest.json missing")
}

Write-Section "REPOSITORY"
Write-Output "root: $repoRoot"
Write-Output "branch: $branch"
Write-Output "head: $head"
Write-Output "origin_main: $remote"
Write-Output "head_vs_origin_main: $relation"

Write-Section "GIT HYGIENE"
Write-Output "tracked_modifications_count: $trackedModified"
Write-Output "staged_changes_count: $staged"
Write-Output "untracked_count: $untracked"
Write-Output "untracked_review_patch_count: $reviewPatchCount"
Write-Output "stage180a_duplicate_untracked_present: $stage180DuplicatePresent"
if ($trackedModified -gt 0 -or $staged -gt 0 -or $untracked -gt 0) {
    Write-Output "warning: DIRTY_WORKTREE_OR_UNTRACKED_FILES_PRESENT"
} else {
    Write-Output "warning: none"
}

Write-Section "RESEARCH STATE"
if ($null -ne $stateLines) {
    Write-Output ("current_phase: " + (Clean-MarkdownValue (Get-MarkdownSectionLines -Lines $stateLines -Heading "Current Phase")))
    Write-Output ("active_research_topic: " + (Clean-MarkdownValue (Get-MarkdownSectionLines -Lines $stateLines -Heading "Active Research Topic")))
    Write-Output ("next_formal_research_step: " + (Clean-MarkdownValue (Get-MarkdownSectionLines -Lines $stateLines -Heading "Next Formal Research Step")))
} else {
    Write-Output "current_phase: UNKNOWN"
    Write-Output "active_research_topic: UNKNOWN"
    Write-Output "next_formal_research_step: UNKNOWN"
}

Write-Section "CANONICAL DEPENDENCIES"
if ($null -ne $manifest) {
    $futureA0 = @($manifest.artifacts | Where-Object { $_.required_for_future_a0 -eq $true })
    foreach ($artifact in $futureA0) {
        $pathStatus = Get-PathStatus -RepoRoot $repoRoot -RelativePath $artifact.path
        $presentText = if ($pathStatus.Present) { "present" } else { "absent" }
        $trackedText = if ($artifact.tracked -eq $true) { "tracked" } elseif ($artifact.tracked -eq $false) { "untracked_or_external" } else { "unknown" }
        $expectedSha256 = Get-OptionalProperty -Object $artifact -Name "physical_sha256"
        $hashStatus = Get-HashStatus -Path $pathStatus.FullPath -ExpectedSha256 $expectedSha256 -Present $pathStatus.Present
        Write-Output "$($artifact.id): $presentText; $trackedText; hash=$hashStatus; path=$($artifact.path)"
        if ($artifact.tracked -eq $true -and -not $pathStatus.Present) {
            $attention.Add("required current tracked control file missing: $($artifact.path)")
        }
        if ($hashStatus -eq "MISMATCH") {
            $attention.Add("hash mismatch: $($artifact.path)")
        }
    }
} else {
    Write-Output "manifest: unavailable"
}

Write-Section "P4-L STATUS"
if ($null -ne $manifest) {
    $sidecar = $manifest.artifacts | Where-Object { $_.id -eq "canonical_p4l_sidecar_absent" } | Select-Object -First 1
    $provenance = $manifest.artifacts | Where-Object { $_.id -eq "canonical_p4l_provenance_absent" } | Select-Object -First 1
    foreach ($artifact in @($sidecar, $provenance)) {
        if ($null -ne $artifact) {
            $pathStatus = Get-PathStatus -RepoRoot $repoRoot -RelativePath $artifact.path
            $state = if ($pathStatus.Present) { "present" } else { "absent" }
            Write-Output "$($artifact.id): WARN_$($state.ToUpperInvariant()); path=$($artifact.path)"
        }
    }
    Write-Output "external_exact_byte_source: UNRESOLVED"
    Write-Output "interpretation: WARN_ONLY_NOT_HISTORICAL_CORRUPTION"
} else {
    Write-Output "p4l_status: UNKNOWN"
}

Write-Section "OPERATING BOUNDARY"
Write-Output "FORMAL_P3W7_A0_AUTHORITY = NOT_CREATED"
Write-Output "FORMAL_TRAINING_ALLOWED = NO"
Write-Output "KAGGLE_GPU_FORMAL_EXECUTION_ALLOWED = NO"

Write-Section "FINAL SUMMARY"
if ($attention.Count -gt 0) {
    Write-Output "status: NEEDS_ATTENTION"
    Write-Output ("attention: " + ($attention -join "; "))
} else {
    Write-Output "status: READY_FOR_PRE_URP_INFRASTRUCTURE"
}
