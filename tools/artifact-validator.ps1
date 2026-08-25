param(
    [Parameter(Mandatory = $true)]
    [string]$ArtifactRoot,
    [string]$ExpectedHead,
    [string]$ExpectedAuthoritySha,
    [string]$ExpectedRunName,
    [string]$ExpectedSeed,
    [string]$ExpectedSplitSeed,
    [string]$ExpectedArm,
    [string]$ExpectedCommandSha256,
    [string]$ExpectedDatasetSha256,
    [string]$ExpectedSidecarSha256,
    [string]$ExpectedRowCount,
    [string[]]$RequiredFile,
    [string[]]$ExpectedFileSha256
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$script:Failures = New-Object System.Collections.Generic.List[string]
$script:Warnings = New-Object System.Collections.Generic.List[string]
$script:ParsedJsonByPath = @{}
$script:PredictionRowsByPath = @{}
$script:IdentityRecords = New-Object System.Collections.Generic.List[object]

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

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Resolve-SafePath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Get-ChildArtifactPath {
    param(
        [string]$Root,
        [string]$RelativePath
    )
    $rootFull = Resolve-SafePath -Path $Root
    $candidate = Resolve-SafePath -Path (Join-Path $rootFull $RelativePath)
    $rootTrimmed = $rootFull.TrimEnd('\', '/')
    $candidateTrimmed = $candidate.TrimEnd('\', '/')
    $comparison = [System.StringComparison]::OrdinalIgnoreCase
    $inside = (
        $candidateTrimmed.Equals($rootTrimmed, $comparison) -or
        $candidateTrimmed.StartsWith($rootTrimmed + [System.IO.Path]::DirectorySeparatorChar, $comparison)
    )
    if (-not $inside) {
        throw "path escapes artifact root: $RelativePath"
    }
    return $candidate
}

function Get-FileSize {
    param([string]$Path)
    return (Get-Item -LiteralPath $Path).Length
}

function ConvertTo-CompactValue {
    param($Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [bool]) { return $Value.ToString().ToLowerInvariant() }
    return ([string]$Value).Trim()
}

function Add-IdentityValue {
    param(
        [string]$Field,
        $Value,
        [string]$Source
    )
    $compact = ConvertTo-CompactValue -Value $Value
    if ([string]::IsNullOrWhiteSpace($compact)) { return }
    $script:IdentityRecords.Add([pscustomobject]@{
        Field = $Field
        Value = $compact
        Source = $Source
    })
}

function Get-PropertyValue {
    param(
        $Object,
        [string]$Name
    )
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Find-JsonValues {
    param(
        $Node,
        [string[]]$Names,
        [int]$Depth = 0
    )
    $values = New-Object System.Collections.Generic.List[object]
    if ($null -eq $Node -or $Depth -gt 16) { return $values }
    if ($Node -is [string] -or $Node -is [ValueType]) { return $values }
    if ($Node -is [System.Collections.IDictionary]) {
        foreach ($key in $Node.Keys) {
            $value = $Node[$key]
            if ($Names -contains [string]$key) { $values.Add($value) }
            foreach ($nested in (Find-JsonValues -Node $value -Names $Names -Depth ($Depth + 1))) {
                $values.Add($nested)
            }
        }
        return $values
    }
    foreach ($property in $Node.PSObject.Properties) {
        if ($Names -contains $property.Name) { $values.Add($property.Value) }
        foreach ($nested in (Find-JsonValues -Node $property.Value -Names $Names -Depth ($Depth + 1))) {
            $values.Add($nested)
        }
    }
    return $values
}

function Read-JsonFile {
    param([string]$Path)
    try {
        $json = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        $script:ParsedJsonByPath[$Path] = $json
        Add-Result "PASS" ([System.IO.Path]::GetFileName($Path)) "json_parse=PASS"
        return $json
    } catch {
        Add-Result "FAIL" ([System.IO.Path]::GetFileName($Path)) "malformed_json"
        return $null
    }
}

function Read-JsonLinesFile {
    param([string]$Path)
    $lineNumber = 0
    $rowCount = 0
    $stableIds = New-Object System.Collections.Generic.HashSet[string]
    $ids = New-Object System.Collections.Generic.HashSet[string]
    $duplicateStableIds = New-Object System.Collections.Generic.List[string]
    $duplicateIds = New-Object System.Collections.Generic.List[string]
    $sampleRows = New-Object System.Collections.Generic.List[object]
    try {
        $reader = [System.IO.File]::OpenText($Path)
        try {
            while ($null -ne ($line = $reader.ReadLine())) {
                $lineNumber += 1
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                try {
                    $row = $line | ConvertFrom-Json
                } catch {
                    Add-Result "FAIL" ([System.IO.Path]::GetFileName($Path)) "malformed_jsonl_line=$lineNumber"
                    return $null
                }
                $rowCount += 1
                if ($sampleRows.Count -lt 20) { $sampleRows.Add($row) }
                $stableIdValue = Get-PropertyValue -Object $row -Name "stable_id"
                $compactStableId = ConvertTo-CompactValue -Value $stableIdValue
                if (-not [string]::IsNullOrWhiteSpace($compactStableId)) {
                    if (-not $stableIds.Add($compactStableId)) {
                        $duplicateStableIds.Add($compactStableId)
                    }
                }
                $idValue = Get-PropertyValue -Object $row -Name "id"
                $compactId = ConvertTo-CompactValue -Value $idValue
                if (-not [string]::IsNullOrWhiteSpace($compactId)) {
                    if (-not $ids.Add($compactId)) { $duplicateIds.Add($compactId) }
                }
            }
        } finally {
            $reader.Close()
        }
    } catch {
        Add-Result "FAIL" ([System.IO.Path]::GetFileName($Path)) "jsonl_read_failed=$($_.Exception.Message)"
        return $null
    }
    $script:PredictionRowsByPath[$Path] = [pscustomobject]@{
        RowCount = $rowCount
        UniqueStableIdCount = $stableIds.Count
        UniqueIdCount = $ids.Count
        DuplicateStableIds = $duplicateStableIds
        DuplicateIds = $duplicateIds
        SampleRows = $sampleRows
    }
    Add-Result "PASS" ([System.IO.Path]::GetFileName($Path)) "jsonl_parse=PASS; non_empty_line_count=$rowCount"
    if ($duplicateStableIds.Count -gt 0) {
        $first = $duplicateStableIds | Select-Object -First 5
        Add-Result "FAIL" ([System.IO.Path]::GetFileName($Path)) "duplicate_stable_id_detected; count=$($duplicateStableIds.Count); examples=$($first -join ',')"
    }
    if ($duplicateIds.Count -gt 0) {
        $first = $duplicateIds | Select-Object -First 5
        Add-Result "FAIL" ([System.IO.Path]::GetFileName($Path)) "duplicate_id_detected; count=$($duplicateIds.Count); examples=$($first -join ',')"
    }
    return $script:PredictionRowsByPath[$Path]
}

function Check-File {
    param(
        [string]$Root,
        [string]$RelativePath,
        [bool]$Required,
        [string]$ExpectedSha
    )
    try {
        $full = Get-ChildArtifactPath -Root $Root -RelativePath $RelativePath
    } catch {
        Add-Result "FAIL" $RelativePath $_.Exception.Message
        return $null
    }
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
        if ($Required) {
            Add-Result "FAIL" $RelativePath "missing_required_file"
        } else {
            Add-Result "NOT_CHECKED" $RelativePath "not_present_optional"
        }
        return $null
    }
    $size = Get-FileSize -Path $full
    $shaText = "NOT_CHECKED"
    if (-not [string]::IsNullOrWhiteSpace($ExpectedSha)) {
        $actual = Get-Sha256 -Path $full
        if ($actual -eq $ExpectedSha.ToLowerInvariant()) {
            $shaText = "VERIFIED"
        } else {
            Add-Result "FAIL" $RelativePath "sha256_mismatch; expected=$ExpectedSha; actual=$actual; size_bytes=$size"
            return $full
        }
    }
    Add-Result "PASS" $RelativePath "present; size_bytes=$size; sha256=$shaText"
    return $full
}

function Get-ExpectedShaForRequiredFile {
    param(
        [int]$Index,
        [string[]]$Values
    )
    if ($null -eq $Values -or $Index -ge $Values.Count) { return $null }
    return $Values[$Index]
}

function Test-IdentityExpectation {
    param(
        [string]$Field,
        [object]$Expected
    )
    if ($null -eq $Expected -or [string]::IsNullOrWhiteSpace([string]$Expected)) {
        Add-Result "NOT_CHECKED" $Field "expectation_not_supplied"
        return
    }
    $expectedText = ConvertTo-CompactValue -Value $Expected
    $values = @()
    $values = @($script:IdentityRecords | Where-Object { $_.Field -eq $Field })
    if ($values.Count -eq 0) {
        Add-Result "FAIL" $Field "ABSENT; expected=$expectedText"
        return
    }
    $observedList = New-Object System.Collections.Generic.List[string]
    foreach ($entry in $values) {
        $observedText = ConvertTo-CompactValue -Value $entry.Value
        if (
            -not [string]::IsNullOrWhiteSpace($observedText) -and
            -not $observedList.Contains($observedText)
        ) {
            $observedList.Add($observedText)
        }
    }
    $observed = @($observedList | Sort-Object)
    if ($observed.Count -eq 1 -and [string]$observed[0] -eq $expectedText) {
        Add-Result "PASS" $Field "MATCH; value=$expectedText"
    } else {
        Add-Result "FAIL" $Field "MISMATCH; expected=$expectedText; observed=$($observed -join ',')"
    }
}

function Test-StringIdentityExpectation {
    param(
        [string]$Field,
        [string]$ExpectedText
    )
    if ([string]::IsNullOrWhiteSpace($ExpectedText)) {
        Add-Result "NOT_CHECKED" $Field "expectation_not_supplied"
        return
    }
    $values = @()
    $values = @($script:IdentityRecords | Where-Object { $_.Field -eq $Field })
    if ($values.Count -eq 0) {
        Add-Result "FAIL" $Field "ABSENT; expected=$ExpectedText"
        return
    }
    $observedList = New-Object System.Collections.Generic.List[string]
    foreach ($entry in $values) {
        $observedText = ConvertTo-CompactValue -Value $entry.Value
        if (
            -not [string]::IsNullOrWhiteSpace($observedText) -and
            -not $observedList.Contains($observedText)
        ) {
            $observedList.Add($observedText)
        }
    }
    $observed = @($observedList | Sort-Object)
    if ($observed.Count -eq 1 -and $observed[0] -eq $ExpectedText) {
        Add-Result "PASS" $Field "MATCH; value=$ExpectedText"
    } else {
        Add-Result "FAIL" $Field "MISMATCH; expected=$ExpectedText; observed=$($observed -join ',')"
    }
}

function Harvest-IdentitiesFromJson {
    param(
        $Json,
        [string]$Source
    )
    $map = @{
        run_name = @("run_name", "run_identifier", "run_id")
        execution_head = @("execution_head", "execution_commit", "current_git_commit", "git_commit", "head", "source_git_commit")
        authority_sha = @("authority_sha", "authority_sha256", "authority_file_sha256", "expected_authority_sha", "declared_authority_sha256")
        command_sha256 = @("command_sha256", "command_sha", "command_identity_sha256", "expected_command_sha256")
        seed = @("seed", "training_seed")
        split_seed = @("split_seed", "configured_split_seed", "resolved_split_seed")
        arm = @("arm", "reason_router_arm")
        dataset_sha256 = @("dataset_sha256", "main_data_sha256", "source_dataset_sha256", "expected_dataset_sha256")
        sidecar_sha256 = @("sidecar_sha256", "integrity_sidecar_sha256", "controlled_integrity_sidecar_sha256", "expected_integrity_sidecar_sha256", "expected_integrity_sidecar_semantic_sha256")
    }
    foreach ($field in $map.Keys) {
        foreach ($value in (Find-JsonValues -Node $Json -Names $map[$field])) {
            Add-IdentityValue -Field $field -Value $value -Source $Source
        }
    }
}

function Harvest-IdentitiesFromSampleRows {
    param(
        $Rows,
        [string]$Source
    )
    foreach ($row in $Rows.SampleRows) {
        Harvest-IdentitiesFromJson -Json $row -Source $Source
    }
}

$rootStatus = "PASS"
$artifactRootFull = $null

Write-Section "ARTIFACT_ROOT"
try {
    $artifactRootFull = Resolve-SafePath -Path $ArtifactRoot
    if (-not (Test-Path -LiteralPath $artifactRootFull)) {
        Add-Result "FAIL" "artifact_root" "missing; path=$ArtifactRoot"
        $rootStatus = "FAIL"
    } elseif (-not (Test-Path -LiteralPath $artifactRootFull -PathType Container)) {
        Add-Result "FAIL" "artifact_root" "not_a_directory; path=$ArtifactRoot"
        $rootStatus = "FAIL"
    } else {
        Add-Result "PASS" "artifact_root" "directory_present; resolved_path=$artifactRootFull"
    }
} catch {
    Add-Result "FAIL" "artifact_root" "resolve_failed=$($_.Exception.Message)"
    $rootStatus = "FAIL"
}

$commonArtifacts = @(
    @{ Name = "training_report.json"; Type = "json" },
    @{ Name = "training_report_predictions.jsonl"; Type = "jsonl" },
    @{ Name = "clean_dev_predictions.json"; Type = "json" },
    @{ Name = "selected_checkpoint.pt"; Type = "binary" },
    @{ Name = "run_provenance.json"; Type = "json" }
)

Write-Section "FILES"
$presentFiles = New-Object System.Collections.Generic.List[object]
if ($rootStatus -eq "PASS") {
    foreach ($artifact in $commonArtifacts) {
        $full = Check-File -Root $artifactRootFull -RelativePath $artifact.Name -Required $false -ExpectedSha $null
        if ($null -ne $full) {
            $presentFiles.Add([pscustomobject]@{
                Path = $full
                Name = $artifact.Name
                Type = $artifact.Type
            })
        }
    }
    if ($null -ne $RequiredFile) {
        for ($i = 0; $i -lt $RequiredFile.Count; $i++) {
            $relative = $RequiredFile[$i]
            $expectedSha = Get-ExpectedShaForRequiredFile -Index $i -Values $ExpectedFileSha256
            $full = Check-File -Root $artifactRootFull -RelativePath $relative -Required $true -ExpectedSha $expectedSha
            if ($null -ne $full -and -not ($presentFiles | Where-Object { $_.Path -eq $full })) {
                $extension = [System.IO.Path]::GetExtension($full).ToLowerInvariant()
                $type = if ($extension -eq ".json") { "json" } elseif ($extension -eq ".jsonl") { "jsonl" } else { "binary" }
                $presentFiles.Add([pscustomobject]@{
                    Path = $full
                    Name = $relative
                    Type = $type
                })
            }
        }
    }
} else {
    Add-Result "NOT_CHECKED" "files" "artifact_root_unavailable"
}

Write-Section "STRUCTURE"
if ($presentFiles.Count -eq 0) {
    Add-Result "WARN" "structure" "no_known_or_required_files_present"
} else {
    foreach ($file in $presentFiles) {
        if ($file.Type -eq "json") {
            $json = Read-JsonFile -Path $file.Path
            if ($null -ne $json) { Harvest-IdentitiesFromJson -Json $json -Source $file.Name }
        } elseif ($file.Type -eq "jsonl") {
            $rows = Read-JsonLinesFile -Path $file.Path
            if ($null -ne $rows) { Harvest-IdentitiesFromSampleRows -Rows $rows -Source $file.Name }
        } else {
            Add-Result "PASS" $file.Name "binary_structure_not_loaded"
        }
    }
}

Write-Section "IDENTITY"
Test-IdentityExpectation -Field "run_name" -Expected $ExpectedRunName
Test-IdentityExpectation -Field "execution_head" -Expected $ExpectedHead
Test-IdentityExpectation -Field "authority_sha" -Expected $ExpectedAuthoritySha
Test-IdentityExpectation -Field "command_sha256" -Expected $ExpectedCommandSha256
if ($PSBoundParameters.ContainsKey("ExpectedSeed")) {
    Test-StringIdentityExpectation -Field "seed" -ExpectedText $ExpectedSeed
} else {
    Add-Result "NOT_CHECKED" "seed" "expectation_not_supplied"
}
if ($PSBoundParameters.ContainsKey("ExpectedSplitSeed")) {
    Test-StringIdentityExpectation -Field "split_seed" -ExpectedText $ExpectedSplitSeed
} else {
    Add-Result "NOT_CHECKED" "split_seed" "expectation_not_supplied"
}
Test-IdentityExpectation -Field "arm" -Expected $ExpectedArm
Test-IdentityExpectation -Field "dataset_sha256" -Expected $ExpectedDatasetSha256
Test-IdentityExpectation -Field "sidecar_sha256" -Expected $ExpectedSidecarSha256

Write-Section "CROSS_FILE"
foreach ($field in @("run_name", "execution_head", "authority_sha", "command_sha256", "seed", "split_seed", "arm", "dataset_sha256", "sidecar_sha256")) {
    $fieldRecords = @($script:IdentityRecords | Where-Object { $_.Field -eq $field })
    if ($fieldRecords.Count -eq 0) {
        Add-Result "NOT_CHECKED" "cross_file_$field" "identity_absent"
        continue
    }
    $valueList = New-Object System.Collections.Generic.List[string]
    foreach ($entry in $fieldRecords) {
        $text = ConvertTo-CompactValue -Value $entry.Value
        if (-not [string]::IsNullOrWhiteSpace($text) -and -not $valueList.Contains($text)) {
            $valueList.Add($text)
        }
    }
    $values = @($valueList | Sort-Object)
    if ($values.Count -le 1) {
        Add-Result "PASS" "cross_file_$field" "consistent_or_single_source; value=$($values -join ',')"
    } else {
        Add-Result "FAIL" "cross_file_$field" "inconsistent_values=$($values -join ',')"
    }
}

$predictionCounts = @($script:PredictionRowsByPath.Keys | ForEach-Object { $script:PredictionRowsByPath[$_].RowCount })
if ($PSBoundParameters.ContainsKey("ExpectedRowCount")) {
    if ($predictionCounts.Count -eq 0) {
        Add-Result "FAIL" "prediction_row_count" "ABSENT; expected=$ExpectedRowCount"
    } else {
        foreach ($path in $script:PredictionRowsByPath.Keys) {
            $actualCount = $script:PredictionRowsByPath[$path].RowCount
            $name = [System.IO.Path]::GetFileName($path)
            if ($actualCount -eq $ExpectedRowCount) {
                Add-Result "PASS" "prediction_row_count_$name" "MATCH; value=$actualCount"
            } else {
                Add-Result "FAIL" "prediction_row_count_$name" "MISMATCH; expected=$ExpectedRowCount; actual=$actualCount"
            }
        }
    }
} else {
    Add-Result "NOT_CHECKED" "prediction_row_count" "expectation_not_supplied"
}

if ($predictionCounts.Count -gt 1) {
    $uniqueCounts = @($predictionCounts | Select-Object -Unique)
    if ($uniqueCounts.Count -eq 1) {
        Add-Result "PASS" "prediction_row_count_consistency" "consistent; value=$($uniqueCounts[0])"
    } else {
        Add-Result "WARN" "prediction_row_count_consistency" "multiple_prediction_row_counts=$($uniqueCounts -join ',')"
    }
} else {
    Add-Result "NOT_CHECKED" "prediction_row_count_consistency" "fewer_than_two_prediction_files"
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
Write-Output "ARTIFACT_INTEGRITY"
Write-Output "PROVENANCE_CONSISTENCY"
if ($script:Failures.Count -gt 0) {
    Write-Output "ARTIFACT_VALIDATION_FAIL"
    foreach ($failure in $script:Failures) {
        Write-Output "failure: $failure"
    }
    exit 1
}
if ($script:Warnings.Count -gt 0) {
    Write-Output "ARTIFACT_VALIDATION_WARN"
    exit 0
}
Write-Output "ARTIFACT_VALIDATION_PASS"
exit 0
