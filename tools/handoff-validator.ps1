[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string]$HandoffZip,
    [string]$ImportedRoot,
    [string]$SourceArtifactRoot,
    [string]$ExpectedRunName,
    [string]$ExpectedHead,
    [string]$ExpectedAuthoritySha,
    [string]$ExpectedCommandSha256,
    [string]$ExpectedSeed,
    [string]$ExpectedSplitSeed,
    [string]$ExpectedArm,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RequiredFile
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$script:Failures = New-Object System.Collections.Generic.List[string]
$script:Warnings = New-Object System.Collections.Generic.List[string]
$script:ZipEntries = @{}
$script:ZipFileEntries = @{}
$script:ExtractRoot = $null
$script:RequiredFiles = New-Object System.Collections.Generic.List[string]

if ($null -ne $RequiredFile) {
    foreach ($value in $RequiredFile) {
        foreach ($part in (([string]$value) -split "[,;]")) {
            if (-not [string]::IsNullOrWhiteSpace($part)) {
                $script:RequiredFiles.Add($part.Trim())
            }
        }
    }
}

function Write-Section {
    param([string]$Name)
    [Console]::Out.WriteLine("")
    [Console]::Out.WriteLine($Name)
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

function Resolve-FullPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Test-SafeChildPath {
    param(
        [string]$Root,
        [string]$Candidate
    )
    $rootFull = (Resolve-FullPath -Path $Root).TrimEnd('\', '/')
    $candidateFull = (Resolve-FullPath -Path $Candidate).TrimEnd('\', '/')
    $comparison = [System.StringComparison]::OrdinalIgnoreCase
    return (
        $candidateFull.Equals($rootFull, $comparison) -or
        $candidateFull.StartsWith($rootFull + [System.IO.Path]::DirectorySeparatorChar, $comparison) -or
        $candidateFull.StartsWith($rootFull + [System.IO.Path]::AltDirectorySeparatorChar, $comparison)
    )
}

function Get-SafeChildPath {
    param(
        [string]$Root,
        [string]$RelativePath
    )
    if ([string]::IsNullOrWhiteSpace($RelativePath)) {
        throw "empty_relative_path"
    }
    $candidate = Resolve-FullPath -Path (Join-Path $Root $RelativePath)
    if (-not (Test-SafeChildPath -Root $Root -Candidate $candidate)) {
        throw "path_escapes_root=$RelativePath"
    }
    return $candidate
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-StreamSha256 {
    param([System.IO.Stream]$Stream)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($Stream)) -replace "-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-ZipEntrySha256 {
    param($Entry)
    $stream = $Entry.Open()
    try {
        return Get-StreamSha256 -Stream $stream
    } finally {
        $stream.Dispose()
    }
}

function Normalize-ZipEntryName {
    param([string]$Name)
    if ([string]::IsNullOrWhiteSpace($Name)) { throw "empty_zip_entry_name" }
    if ($Name.StartsWith("/") -or $Name.StartsWith("\")) { throw "absolute_zip_entry=$Name" }
    if ($Name -match "^[A-Za-z]:") { throw "drive_qualified_zip_entry=$Name" }
    if ($Name.StartsWith("//") -or $Name.StartsWith("\\")) { throw "unc_zip_entry=$Name" }
    $converted = $Name.Replace("\", "/")
    $parts = New-Object System.Collections.Generic.List[string]
    foreach ($part in ($converted -split "/")) {
        if ([string]::IsNullOrWhiteSpace($part)) { continue }
        if ($part -eq ".") { continue }
        if ($part -eq "..") { throw "traversal_zip_entry=$Name" }
        $parts.Add($part)
    }
    if ($parts.Count -eq 0) { throw "empty_normalized_zip_entry=$Name" }
    return (($parts -join "/").ToLowerInvariant())
}

function Test-ZipStructure {
    param([string]$Path)
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = $null
    try {
        $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
        $seen = New-Object System.Collections.Generic.HashSet[string]
        $entryCount = 0
        $fileCount = 0
        foreach ($entry in $zip.Entries) {
            $entryCount += 1
            try {
                $normalized = Normalize-ZipEntryName -Name $entry.FullName
            } catch {
                Add-Result "FAIL" "zip_entry_path" $_.Exception.Message
                continue
            }
            if (-not $seen.Add($normalized)) {
                Add-Result "FAIL" "zip_duplicate_entry" "duplicate_normalized_path=$normalized"
                continue
            }
            $script:ZipEntries[$normalized] = $entry
            if (-not [string]::IsNullOrEmpty($entry.Name)) {
                $script:ZipFileEntries[$normalized] = $entry
                $fileCount += 1
            }
        }
        Add-Result "PASS" "zip_structure" "entries=$entryCount; files=$fileCount"
        return $zip
    } catch {
        if ($null -ne $zip) { $zip.Dispose() }
        Add-Result "FAIL" "zip_structure" "malformed_or_unreadable_zip=$($_.Exception.Message)"
        return $null
    }
}

function Copy-ZipEntryToFile {
    param(
        $Entry,
        [string]$Destination
    )
    $parent = [System.IO.Path]::GetDirectoryName($Destination)
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $inputStream = $Entry.Open()
    try {
        $outputStream = [System.IO.File]::Open($Destination, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        try {
            $inputStream.CopyTo($outputStream)
        } finally {
            $outputStream.Dispose()
        }
    } finally {
        $inputStream.Dispose()
    }
}

function Expand-ZipSafely {
    param($Zip)
    $tempBase = [System.IO.Path]::GetTempPath()
    $root = Join-Path $tempBase ("NON_SCIENTIFIC_PRE_URP_INFRASTRUCTURE_TEST_handoff_" + [System.Guid]::NewGuid().ToString("N"))
    [System.IO.Directory]::CreateDirectory($root) | Out-Null
    $script:ExtractRoot = $root
    foreach ($name in $script:ZipFileEntries.Keys) {
        $entry = $script:ZipFileEntries[$name]
        $destination = Get-SafeChildPath -Root $root -RelativePath $name
        Copy-ZipEntryToFile -Entry $entry -Destination $destination
    }
    return $root
}

function Get-RepoRootFromScript {
    $scriptDir = Split-Path -Parent $MyInvocation.ScriptName
    if ([string]::IsNullOrWhiteSpace($scriptDir)) {
        $scriptDir = Split-Path -Parent $PSCommandPath
    }
    return Resolve-FullPath -Path (Join-Path $scriptDir "..")
}

function Get-ArtifactValidatorPath {
    $repoRoot = Get-RepoRootFromScript
    return Join-Path (Join-Path $repoRoot "tools") "artifact-validator.ps1"
}

function Get-CommonArtifactNames {
    return @(
        "training_report.json",
        "training_report_predictions.jsonl",
        "clean_dev_predictions.json",
        "selected_checkpoint.pt",
        "run_provenance.json",
        "collection_manifest.json",
        "handoff_manifest.json",
        "import_manifest.json"
    )
}

function Normalize-RelativeArtifactPath {
    param([string]$Path)
    return (Normalize-ZipEntryName -Name $Path)
}

function Get-CompareTargets {
    $targets = New-Object System.Collections.Generic.HashSet[string]
    foreach ($name in (Get-CommonArtifactNames)) {
        try { [void]$targets.Add((Normalize-RelativeArtifactPath -Path $name)) } catch {}
    }
    if ($script:RequiredFiles.Count -gt 0) {
        foreach ($name in $script:RequiredFiles) {
            if (-not [string]::IsNullOrWhiteSpace($name)) {
                [void]$targets.Add((Normalize-RelativeArtifactPath -Path $name))
            }
        }
    }
    return $targets
}

function Test-ArtifactCompleteness {
    Write-Section "ARTIFACT_COMPLETENESS"
    if ($script:RequiredFiles.Count -eq 0) {
        Add-Result "NOT_CHECKED" "required_files" "no_required_files_supplied"
    } else {
        foreach ($required in $script:RequiredFiles) {
            try {
                $normalized = Normalize-RelativeArtifactPath -Path $required
                if ($script:ZipFileEntries.ContainsKey($normalized)) {
                    Add-Result "PASS" $required "present_in_zip"
                } else {
                    Add-Result "FAIL" $required "missing_required_zip_artifact"
                }
            } catch {
                Add-Result "FAIL" $required "invalid_required_path=$($_.Exception.Message)"
            }
        }
    }
    foreach ($artifact in (Get-CommonArtifactNames)) {
        $normalized = Normalize-RelativeArtifactPath -Path $artifact
        if ($script:ZipFileEntries.ContainsKey($normalized)) {
            Add-Result "PASS" $artifact "detected_optional_artifact"
        } else {
            Add-Result "NOT_CHECKED" $artifact "optional_not_present"
        }
    }
}

function Invoke-InternalArtifactValidation {
    param([string]$Root)
    Write-Section "PROVENANCE"
    $validator = Get-ArtifactValidatorPath
    if (-not (Test-Path -LiteralPath $validator -PathType Leaf)) {
        Add-Result "FAIL" "artifact_validator" "missing=$validator"
        return
    }
    $args = @("-ExecutionPolicy", "Bypass", "-File", $validator, "-ArtifactRoot", $Root)
    if (-not [string]::IsNullOrWhiteSpace($ExpectedRunName)) { $args += @("-ExpectedRunName", $ExpectedRunName) }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedHead)) { $args += @("-ExpectedHead", $ExpectedHead) }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedAuthoritySha)) { $args += @("-ExpectedAuthoritySha", $ExpectedAuthoritySha) }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedCommandSha256)) { $args += @("-ExpectedCommandSha256", $ExpectedCommandSha256) }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedSeed)) { $args += @("-ExpectedSeed", $ExpectedSeed) }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedSplitSeed)) { $args += @("-ExpectedSplitSeed", $ExpectedSplitSeed) }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedArm)) { $args += @("-ExpectedArm", $ExpectedArm) }
    $output = & powershell @args 2>&1
    $exit = $LASTEXITCODE
    $final = ($output | Select-String -Pattern "ARTIFACT_VALIDATION_(PASS|WARN|FAIL)" | Select-Object -Last 1).Line
    if ([string]::IsNullOrWhiteSpace($final)) { $final = "artifact_validator_final_status_absent" }
    if ($exit -ne 0) {
        $failureDetails = @($output | Select-String -Pattern "^failure:" | Select-Object -First 3)
        $detail = "exit_code=$exit; final=$final"
        if ($null -ne $failureDetails -and $failureDetails.Count -gt 0) {
            $detail = $detail + "; " + (($failureDetails | ForEach-Object { $_.Line }) -join " | ")
        }
        Add-Result "FAIL" "artifact_validator" $detail
    } elseif ($final -match "WARN") {
        Add-Result "WARN" "artifact_validator" "exit_code=0; final=$final"
    } else {
        Add-Result "PASS" "artifact_validator" "exit_code=0; final=$final"
    }
}

function Compare-FileWithZipEntry {
    param(
        [string]$FilePath,
        $Entry
    )
    $fileItem = Get-Item -LiteralPath $FilePath
    $fileSha = Get-Sha256 -Path $FilePath
    $zipSha = Get-ZipEntrySha256 -Entry $Entry
    $sizeMatch = ($fileItem.Length -eq $Entry.Length)
    $shaMatch = ($fileSha -eq $zipSha)
    return [pscustomobject]@{
        SizeMatch = $sizeMatch
        ShaMatch = $shaMatch
        FileSize = $fileItem.Length
        EntrySize = $Entry.Length
        FileSha = $fileSha
        EntrySha = $zipSha
    }
}

function Compare-RootToZip {
    param(
        [string]$Root,
        [string]$SectionName,
        [bool]$FailRequiredMissingOnly
    )
    Write-Section $SectionName
    if ([string]::IsNullOrWhiteSpace($Root)) {
        Add-Result "NOT_CHECKED" $SectionName "root_not_supplied"
        return "NOT_CHECKED"
    }
    $rootFull = Resolve-FullPath -Path $Root
    if (-not (Test-Path -LiteralPath $rootFull -PathType Container)) {
        Add-Result "FAIL" $SectionName "root_missing_or_not_directory=$rootFull"
        return "FAIL"
    }
    $sectionStatus = "PASS"
    $requiredSet = New-Object System.Collections.Generic.HashSet[string]
    if ($script:RequiredFiles.Count -gt 0) {
        foreach ($required in $script:RequiredFiles) {
            [void]$requiredSet.Add((Normalize-RelativeArtifactPath -Path $required))
        }
    }
    foreach ($target in (Get-CompareTargets)) {
        $sourcePath = Get-SafeChildPath -Root $rootFull -RelativePath $target
        $sourceExists = Test-Path -LiteralPath $sourcePath -PathType Leaf
        $zipExists = $script:ZipFileEntries.ContainsKey($target)
        $isRequired = $requiredSet.Contains($target)
        if ($sourceExists -and $zipExists) {
            $comparison = Compare-FileWithZipEntry -FilePath $sourcePath -Entry $script:ZipFileEntries[$target]
            if ($comparison.SizeMatch -and $comparison.ShaMatch) {
                Add-Result "PASS" $target "MATCH; size_bytes=$($comparison.FileSize); sha256=$($comparison.FileSha)"
            } else {
                Add-Result "FAIL" $target "MISMATCH; source_sha256=$($comparison.FileSha); zip_sha256=$($comparison.EntrySha)"
                $sectionStatus = "FAIL"
            }
        } elseif ($sourceExists -and -not $zipExists) {
            if ($isRequired) {
                Add-Result "FAIL" $target "ABSENT_IN_ZIP"
                $sectionStatus = "FAIL"
            } else {
                Add-Result "WARN" $target "ABSENT_IN_ZIP_OPTIONAL_SOURCE_PRESENT"
                if ($sectionStatus -ne "FAIL") { $sectionStatus = "WARN" }
            }
        } elseif (-not $sourceExists -and $zipExists) {
            if ($isRequired -and -not $FailRequiredMissingOnly) {
                Add-Result "FAIL" $target "ABSENT_IN_SOURCE"
                $sectionStatus = "FAIL"
            } else {
                Add-Result "WARN" $target "ABSENT_IN_SOURCE_ZIP_PRESENT"
                if ($sectionStatus -ne "FAIL") { $sectionStatus = "WARN" }
            }
        } elseif ($isRequired) {
            Add-Result "FAIL" $target "ABSENT_IN_SOURCE_AND_ZIP"
            $sectionStatus = "FAIL"
        } else {
            Add-Result "NOT_CHECKED" $target "absent_in_both"
        }
    }
    Add-Result $sectionStatus $SectionName "summary=$sectionStatus"
    return $sectionStatus
}

function Compare-ZipToImport {
    param([string]$Root)
    Write-Section "ZIP_TO_IMPORT"
    if ([string]::IsNullOrWhiteSpace($Root)) {
        Add-Result "NOT_CHECKED" "ZIP_TO_IMPORT" "imported_root_not_supplied"
        return "NOT_CHECKED"
    }
    $rootFull = Resolve-FullPath -Path $Root
    if (-not (Test-Path -LiteralPath $rootFull -PathType Container)) {
        Add-Result "FAIL" "ZIP_TO_IMPORT" "imported_root_missing_or_not_directory=$rootFull"
        return "FAIL"
    }
    $sectionStatus = "PASS"
    $requiredSet = New-Object System.Collections.Generic.HashSet[string]
    if ($script:RequiredFiles.Count -gt 0) {
        foreach ($required in $script:RequiredFiles) {
            [void]$requiredSet.Add((Normalize-RelativeArtifactPath -Path $required))
        }
    }
    foreach ($target in (Get-CompareTargets)) {
        $importPath = Get-SafeChildPath -Root $rootFull -RelativePath $target
        $importExists = Test-Path -LiteralPath $importPath -PathType Leaf
        $zipExists = $script:ZipFileEntries.ContainsKey($target)
        $isRequired = $requiredSet.Contains($target)
        if ($zipExists -and $importExists) {
            $comparison = Compare-FileWithZipEntry -FilePath $importPath -Entry $script:ZipFileEntries[$target]
            if ($comparison.SizeMatch -and $comparison.ShaMatch) {
                Add-Result "PASS" $target "MATCH; size_bytes=$($comparison.FileSize); sha256=$($comparison.FileSha)"
            } else {
                Add-Result "FAIL" $target "MISMATCH; zip_sha256=$($comparison.EntrySha); import_sha256=$($comparison.FileSha)"
                $sectionStatus = "FAIL"
            }
        } elseif ($isRequired -and $zipExists -and -not $importExists) {
            Add-Result "FAIL" $target "ABSENT_IN_IMPORT"
            $sectionStatus = "FAIL"
        } elseif ($zipExists -and -not $importExists) {
            Add-Result "WARN" $target "ABSENT_IN_IMPORT_OPTIONAL_ZIP_PRESENT"
            if ($sectionStatus -ne "FAIL") { $sectionStatus = "WARN" }
        } elseif ($isRequired -and -not $zipExists) {
            Add-Result "FAIL" $target "ABSENT_IN_ZIP"
            $sectionStatus = "FAIL"
        } else {
            Add-Result "NOT_CHECKED" $target "zip_artifact_not_present"
        }
    }
    Add-Result $sectionStatus "ZIP_TO_IMPORT" "summary=$sectionStatus"
    return $sectionStatus
}

function Write-CrossRunSection {
    Write-Section "CROSS_RUN"
    Add-Result "NOT_CHECKED" "cm_specific_identity_contract" "no_concrete_cm_collect_import_schema_established_in_repository"
    Add-Result "PASS" "cross_file_identity" "delegated_to_artifact_validator_when_zip_payload_artifacts_present"
}

function Write-FinalStatus {
    param(
        [string]$SourceToZip,
        [string]$ZipToImport
    )
    Write-Section "WARNINGS"
    if ($script:Warnings.Count -eq 0) {
        Add-Result "PASS" "warnings" "none"
    } else {
        Add-Result "WARN" "warnings" "count=$($script:Warnings.Count)"
    }
    Write-Section "FINAL_STATUS"
    $e2e = "NOT_CHECKED"
    if ($SourceToZip -ne "NOT_CHECKED" -and $ZipToImport -ne "NOT_CHECKED") {
        if ($SourceToZip -eq "FAIL" -or $ZipToImport -eq "FAIL") {
            $e2e = "FAIL"
        } elseif ($SourceToZip -eq "WARN" -or $ZipToImport -eq "WARN") {
            $e2e = "WARN"
        } else {
            $e2e = "PASS"
        }
    }
    Add-Result "PASS" "HANDOFF_INTEGRITY_SCOPE" "byte_structure_and_provenance_only"
    Add-Result "PASS" "SCIENTIFIC_SUCCESS" "NOT_ESTABLISHED"
    Add-Result $e2e "END_TO_END" "SOURCE_TO_ZIP=$SourceToZip; ZIP_TO_IMPORT=$ZipToImport"
    if ($script:Failures.Count -gt 0) {
        Write-Output "HANDOFF_VALIDATION_FAIL"
        exit 1
    }
    if ($script:Warnings.Count -gt 0) {
        Write-Output "HANDOFF_VALIDATION_WARN"
        exit 0
    }
    Write-Output "HANDOFF_VALIDATION_PASS"
    exit 0
}

$zip = $null
try {
    Write-Section "HANDOFF"
    $zipPath = Resolve-FullPath -Path $HandoffZip
    if (-not (Test-Path -LiteralPath $zipPath -PathType Leaf)) {
        Add-Result "FAIL" "handoff_zip" "missing_or_not_file=$zipPath"
        Write-FinalStatus -SourceToZip "NOT_CHECKED" -ZipToImport "NOT_CHECKED"
    }
    $zipItem = Get-Item -LiteralPath $zipPath
    Add-Result "PASS" "handoff_zip" "path=$zipPath; size_bytes=$($zipItem.Length); sha256=$(Get-Sha256 -Path $zipPath)"

    Write-Section "ZIP_STRUCTURE"
    $zip = Test-ZipStructure -Path $zipPath
    if ($null -eq $zip) {
        Write-FinalStatus -SourceToZip "NOT_CHECKED" -ZipToImport "NOT_CHECKED"
    }

    Test-ArtifactCompleteness

    if ($script:Failures.Count -eq 0) {
        $extractRoot = Expand-ZipSafely -Zip $zip
        Invoke-InternalArtifactValidation -Root $extractRoot
    } else {
        Write-Section "PROVENANCE"
        Add-Result "NOT_CHECKED" "artifact_validator" "skipped_due_to_prior_zip_or_completeness_failure"
    }

    $sourceToZip = Compare-RootToZip -Root $SourceArtifactRoot -SectionName "SOURCE_TO_ZIP" -FailRequiredMissingOnly $false
    $zipToImport = Compare-ZipToImport -Root $ImportedRoot
    Write-CrossRunSection
    Write-FinalStatus -SourceToZip $sourceToZip -ZipToImport $zipToImport
} finally {
    if ($null -ne $zip) { $zip.Dispose() }
    if (-not [string]::IsNullOrWhiteSpace($script:ExtractRoot) -and (Test-Path -LiteralPath $script:ExtractRoot)) {
        try {
            Remove-Item -LiteralPath $script:ExtractRoot -Recurse -Force -ErrorAction Stop
        } catch {
            [Console]::Error.WriteLine("WARN`ttemp_cleanup`tfailed=$($script:ExtractRoot); $($_.Exception.Message)")
        }
    }
}
