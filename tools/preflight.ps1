param(
    [string]$Mode = "PRE_URP_INFRASTRUCTURE",
    [string]$ExpectedHead,
    [string]$AuthorityPath,
    [string]$RunName,
    [int]$Seed,
    [string[]]$AllowedSeeds,
    [string]$OutputRoot,
    [string[]]$RequiredInputPath,
    [string[]]$RequiredInputSha256,
    [string[]]$ExactOutputTarget,
    [string]$CommandString,
    [string]$ExpectedCommandSha256
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$script:Failures = New-Object System.Collections.Generic.List[string]
$script:Warnings = New-Object System.Collections.Generic.List[string]

function Write-Section {
    param([string]$Name)
    Write-Output ""
    Write-Output $Name
}

function Add-Result {
    param(
        [string]$Status,
        [string]$Key,
        [string]$Detail
    )
    [Console]::Out.WriteLine("$Status`t$Key`t$Detail")
    if ($Status -eq "FAIL") { $script:Failures.Add("${Key}: $Detail") }
    if ($Status -eq "WARN") { $script:Warnings.Add("${Key}: $Detail") }
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

function Get-RepoRoot {
    $root = Invoke-Git -GitArgs @("rev-parse", "--show-toplevel") -AllowFailure
    if ($null -eq $root) { return $null }
    return ($root | Select-Object -First 1)
}

function Get-GitRelation {
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

function Resolve-RepoPath {
    param(
        [string]$RepoRoot,
        [string]$Path
    )
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}

function Test-InsideRepo {
    param(
        [string]$RepoRoot,
        [string]$Path
    )
    $rootFull = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\', '/')
    $pathFull = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    return ($pathFull -eq $rootFull -or $pathFull.StartsWith($rootFull + [System.IO.Path]::DirectorySeparatorChar))
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-StringSha256 {
    param([string]$Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes)) -replace "-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Test-ControlFile {
    param(
        [string]$RepoRoot,
        [string]$RelativePath,
        [switch]$Json
    )
    $full = Resolve-RepoPath -RepoRoot $RepoRoot -Path $RelativePath
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
        Add-Result "FAIL" $RelativePath "missing"
        return $null
    }
    if ($Json) {
        try {
            $parsed = Get-Content -LiteralPath $full -Raw | ConvertFrom-Json
            Add-Result "PASS" $RelativePath "present; json_parse=PASS"
            return $parsed
        } catch {
            Add-Result "FAIL" $RelativePath "malformed_json"
            return $null
        }
    }
    Add-Result "PASS" $RelativePath "present"
    return $true
}

function Test-FileSha {
    param(
        [string]$RepoRoot,
        [string]$Path,
        [string]$ExpectedSha256
    )
    $full = Resolve-RepoPath -RepoRoot $RepoRoot -Path $Path
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
        return "ABSENT"
    }
    if ([string]::IsNullOrWhiteSpace($ExpectedSha256)) {
        return "NOT_CHECKED"
    }
    $actual = Get-Sha256 -Path $full
    if ($actual -eq $ExpectedSha256.ToLowerInvariant()) {
        return "VERIFIED"
    }
    return "MISMATCH"
}

function Test-PathNonexistent {
    param(
        [string]$RepoRoot,
        [string]$Path,
        [string]$Key
    )
    $full = Resolve-RepoPath -RepoRoot $RepoRoot -Path $Path
    if (Test-Path -LiteralPath $full) {
        Add-Result "FAIL" $Key "path_exists; path=$Path"
    } else {
        Add-Result "PASS" $Key "path_absent; path=$Path"
    }
}

function Get-AllowedSeedValues {
    param([string[]]$Values)
    $parsed = New-Object System.Collections.Generic.List[int]
    if ($null -eq $Values) { return $parsed }
    foreach ($value in $Values) {
        foreach ($part in ([string]$value -split ",")) {
            $trimmed = $part.Trim()
            if ($trimmed -eq "") { continue }
            $seedValue = 0
            if (-not [int]::TryParse($trimmed, [ref]$seedValue)) {
                throw "invalid allowed seed value: $trimmed"
            }
            $parsed.Add($seedValue)
        }
    }
    return $parsed
}

function Write-NotChecked {
    param([string]$Key)
    Add-Result "NOT_CHECKED" $Key "parameter_not_supplied"
}

$repoRoot = Get-RepoRoot
Write-Section "REPOSITORY"
if ($null -eq $repoRoot) {
    Add-Result "FAIL" "repository" "not_a_git_repository"
    Write-Section "FINAL_STATUS"
    Write-Output "BLOCKED"
    exit 1
}

Set-Location -LiteralPath $repoRoot

$branch = Invoke-Git -GitArgs @("branch", "--show-current") -AllowFailure
if ($null -eq $branch -or [string]::IsNullOrWhiteSpace(($branch | Select-Object -First 1))) {
    $shortHead = Invoke-Git -GitArgs @("rev-parse", "--short", "HEAD") -AllowFailure
    $branch = "DETACHED@$($shortHead | Select-Object -First 1)"
} else {
    $branch = $branch | Select-Object -First 1
}
$head = Invoke-Git -GitArgs @("rev-parse", "HEAD") -AllowFailure
$head = if ($null -eq $head) { "UNKNOWN" } else { $head | Select-Object -First 1 }
$remote = Invoke-Git -GitArgs @("rev-parse", "origin/main") -AllowFailure
$remote = if ($null -eq $remote) { "UNKNOWN" } else { $remote | Select-Object -First 1 }
$relation = Get-GitRelation

$trackedModified = Count-Lines (Invoke-Git -GitArgs @("diff", "--name-only") -AllowFailure)
$stagedModified = Count-Lines (Invoke-Git -GitArgs @("diff", "--cached", "--name-only") -AllowFailure)
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$porcelain = & git status --porcelain --untracked-files=all 2>$null
$ErrorActionPreference = $previousErrorActionPreference
$untracked = Count-Lines ($porcelain | Where-Object { $_ -like "?? *" })
$reviewPatches = @(Get-ChildItem -LiteralPath $repoRoot -Filter "reason_router_*.patch" -File -ErrorAction SilentlyContinue).Count
$stage180Duplicate = Test-Path -LiteralPath (Join-Path $repoRoot "reports/stage180a_pass2_annotations_completed.csv")

Add-Result "PASS" "root" $repoRoot
Add-Result "PASS" "branch" $branch
Add-Result "PASS" "head" $head
if ($remote -eq "UNKNOWN") { Add-Result "WARN" "origin_main" "UNKNOWN" } else { Add-Result "PASS" "origin_main" $remote }
if ($relation -eq "UNKNOWN") { Add-Result "WARN" "head_vs_origin_main" "UNKNOWN" } else { Add-Result "PASS" "head_vs_origin_main" $relation }
if ($trackedModified -gt 0) { Add-Result "FAIL" "tracked_dirty_state" "tracked_modifications_count=$trackedModified" } else { Add-Result "PASS" "tracked_dirty_state" "tracked_modifications_count=0" }
if ($stagedModified -gt 0) { Add-Result "FAIL" "staged_state" "staged_changes_count=$stagedModified" } else { Add-Result "PASS" "staged_state" "staged_changes_count=0" }
if ($untracked -gt 0) {
    Add-Result "WARN" "untracked_state" "untracked_count=$untracked; review_patch_count=$reviewPatches; stage180a_duplicate_present=$stage180Duplicate"
} else {
    Add-Result "PASS" "untracked_state" "untracked_count=0"
}

Write-Section "CONTROL_FILES"
$agents = Test-ControlFile -RepoRoot $repoRoot -RelativePath "AGENTS.md"
$ops = Test-ControlFile -RepoRoot $repoRoot -RelativePath "docs/RESEARCH_OPERATIONS.md"
$state = Test-ControlFile -RepoRoot $repoRoot -RelativePath "reports/RESEARCH_STATE.md"
$manifest = Test-ControlFile -RepoRoot $repoRoot -RelativePath "reports/artifact_manifest.json" -Json

Write-Section "CANONICAL_ARTIFACTS"
if ($null -eq $manifest) {
    Add-Result "FAIL" "canonical_artifacts" "manifest_unavailable"
} else {
    $requiredArtifacts = @($manifest.artifacts | Where-Object { $_.required_for_future_a0 -eq $true })
    foreach ($artifact in $requiredArtifacts) {
        $path = [string]$artifact.path
        $full = Resolve-RepoPath -RepoRoot $repoRoot -Path $path
        $present = Test-Path -LiteralPath $full
        $tracked = Get-OptionalProperty -Object $artifact -Name "tracked"
        $expectedSha = Get-OptionalProperty -Object $artifact -Name "physical_sha256"
        $hashStatus = Test-FileSha -RepoRoot $repoRoot -Path $path -ExpectedSha256 $expectedSha
        $status = "PASS"
        if (-not $present -and ($artifact.id -eq "canonical_p4l_sidecar_absent" -or $artifact.id -eq "canonical_p4l_provenance_absent")) {
            $status = "WARN"
            $hashStatus = "WARN_ABSENT_EXPECTED_PRE_URP"
        } elseif (-not $present -and $tracked -eq $true) {
            $status = "FAIL"
        } elseif ($hashStatus -eq "MISMATCH") {
            $status = "FAIL"
        }
        $presenceText = if ($present) { "present" } else { "absent" }
        Add-Result $status $artifact.id "path=$path; present=$presenceText; tracked=$tracked; physical_sha256=$hashStatus"
    }
}

Write-Section "AUTHORITY"
Add-Result "PASS" "FORMAL_EXECUTION_AUTHORITY" "NOT_AVAILABLE"
Add-Result "PASS" "FORMAL_TRAINING" "NOT_ALLOWED"
Add-Result "PASS" "FORMAL_EVALUATION" "NOT_ALLOWED"
Add-Result "PASS" "FORMAL_KAGGLE_GPU_EXECUTION" "NOT_ALLOWED"
Add-Result "PASS" "mode" $Mode

Write-Section "RUN_SAFETY"
if ([string]::IsNullOrWhiteSpace($ExpectedHead)) {
    Write-NotChecked "expected_head"
} elseif ($head -eq $ExpectedHead) {
    Add-Result "PASS" "expected_head" "matches"
} else {
    Add-Result "FAIL" "expected_head" "expected=$ExpectedHead; actual=$head"
}

if ([string]::IsNullOrWhiteSpace($AuthorityPath)) {
    Write-NotChecked "authority_path"
} else {
    $fullAuthority = Resolve-RepoPath -RepoRoot $repoRoot -Path $AuthorityPath
    if ((Test-InsideRepo -RepoRoot $repoRoot -Path $fullAuthority) -and (Test-Path -LiteralPath $fullAuthority -PathType Leaf)) {
        Add-Result "PASS" "authority_path" "exists; path=$AuthorityPath"
    } else {
        Add-Result "FAIL" "authority_path" "missing_or_outside_repo; path=$AuthorityPath"
    }
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    Write-NotChecked "run_name"
} else {
    Add-Result "PASS" "run_name" "provided=$RunName"
}

if ($PSBoundParameters.ContainsKey("Seed")) {
    $seedAllowlistParseFailed = $false
    try {
        $allowedSeedValues = Get-AllowedSeedValues -Values $AllowedSeeds
    } catch {
        Add-Result "FAIL" "seed_allowlist" $_.Exception.Message
        $seedAllowlistParseFailed = $true
        $allowedSeedValues = @()
    }
    if ($allowedSeedValues.Count -eq 0 -and -not $seedAllowlistParseFailed) {
        Add-Result "NOT_CHECKED" "seed_allowlist" "seed=$Seed; allowed_seed_set_not_supplied"
    } elseif ($allowedSeedValues -contains $Seed) {
        Add-Result "PASS" "seed_allowlist" "seed=$Seed"
    } else {
        Add-Result "FAIL" "seed_allowlist" "seed=$Seed not_in_allowed_set"
    }
} else {
    Write-NotChecked "seed_allowlist"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    Write-NotChecked "output_root_collision"
} else {
    Test-PathNonexistent -RepoRoot $repoRoot -Path $OutputRoot -Key "output_root_collision"
    $fullOutputRoot = Resolve-RepoPath -RepoRoot $repoRoot -Path $OutputRoot
    if (Test-InsideRepo -RepoRoot $repoRoot -Path $fullOutputRoot) {
        Add-Result "PASS" "output_root_location" "inside_repository"
    } else {
        Add-Result "WARN" "output_root_location" "outside_repository_or_external_runtime_root; path=$OutputRoot"
    }
}

if ([string]::IsNullOrWhiteSpace($RunName) -or [string]::IsNullOrWhiteSpace($OutputRoot)) {
    Write-NotChecked "run_name_collision"
} else {
    $runPath = Join-Path $OutputRoot $RunName
    Test-PathNonexistent -RepoRoot $repoRoot -Path $runPath -Key "run_name_collision"
}

if ($null -eq $RequiredInputPath -or $RequiredInputPath.Count -eq 0) {
    Write-NotChecked "required_input_paths"
} else {
    for ($i = 0; $i -lt $RequiredInputPath.Count; $i++) {
        $path = $RequiredInputPath[$i]
        $full = Resolve-RepoPath -RepoRoot $repoRoot -Path $path
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
            Add-Result "FAIL" "required_input_$i" "missing; path=$path"
            continue
        }
        $expected = $null
        if ($null -ne $RequiredInputSha256 -and $i -lt $RequiredInputSha256.Count) {
            $expected = $RequiredInputSha256[$i]
        }
        $hashStatus = Test-FileSha -RepoRoot $repoRoot -Path $path -ExpectedSha256 $expected
        if ($hashStatus -eq "MISMATCH") {
            Add-Result "FAIL" "required_input_$i" "present; hash=MISMATCH; path=$path"
        } else {
            Add-Result "PASS" "required_input_$i" "present; hash=$hashStatus; path=$path"
        }
    }
}

if ($null -eq $ExactOutputTarget -or $ExactOutputTarget.Count -eq 0) {
    Write-NotChecked "exact_output_targets"
} else {
    for ($i = 0; $i -lt $ExactOutputTarget.Count; $i++) {
        Test-PathNonexistent -RepoRoot $repoRoot -Path $ExactOutputTarget[$i] -Key "exact_output_target_$i"
    }
}

Write-Section "COMMAND_IDENTITY"
if ([string]::IsNullOrWhiteSpace($CommandString) -and [string]::IsNullOrWhiteSpace($ExpectedCommandSha256)) {
    Add-Result "NOT_CHECKED" "COMMAND_IDENTITY" "command_and_expected_hash_not_supplied"
} elseif ([string]::IsNullOrWhiteSpace($CommandString) -or [string]::IsNullOrWhiteSpace($ExpectedCommandSha256)) {
    Add-Result "FAIL" "COMMAND_IDENTITY" "command_or_expected_hash_missing"
} else {
    $commandSha = Get-StringSha256 -Text $CommandString
    if ($commandSha -eq $ExpectedCommandSha256.ToLowerInvariant()) {
        Add-Result "PASS" "COMMAND_IDENTITY" "sha256=$commandSha"
    } else {
        Add-Result "FAIL" "COMMAND_IDENTITY" "expected=$ExpectedCommandSha256; actual=$commandSha"
    }
}

Write-Section "WARNINGS"
if ($script:Warnings.Count -eq 0) {
    Write-Output "none"
} else {
    foreach ($warning in $script:Warnings) {
        Write-Output $warning
    }
}

Write-Section "FINAL_STATUS"
if ($script:Failures.Count -gt 0) {
    Write-Output "BLOCKED"
    foreach ($failure in $script:Failures) {
        Write-Output "failure: $failure"
    }
    exit 1
}

Write-Output "PRE_URP_INFRASTRUCTURE_READY"
exit 0
