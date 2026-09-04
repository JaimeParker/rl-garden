[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RoboTwinPath,
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
$bundleRoot = $PSScriptRoot
$expectedBaseCommit = '0008ae6800df9f75fc8de7098bacb01735fd8fd2'
$expectedTargetCommit = '964a4e4b1c434d62a5d106a8fbc543210641a8d9'

$files = @(
    @{
        RelativePath = 'envs/_base_task.py'
        BaseSha256 = 'aa9d717ad214b9d68c6eeb83ed03f2491db6addc98deaa988a48a3c074178bdd'
        TargetSha256 = 'ce1072e90a3d9083333181bf5bd53f15d2d86c540b5720738ad151ac98fb2bbc'
    },
    @{
        RelativePath = 'envs/robot/planner.py'
        BaseSha256 = '178a72a7b6ededee66e7f78ebc35e7c39c4a855a1d713b90a01a32868d49f24a'
        TargetSha256 = 'c170726233650ba3fc54feb89e3f41df40081f6b2feffdc82b4e10cb7c586c7c'
    }
)

function Join-RelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )
    return Join-Path $Root ($RelativePath -replace '/', [System.IO.Path]::DirectorySeparatorChar)
}

function Get-Sha256OrNull {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

$resolvedRoboTwin = (Resolve-Path -LiteralPath $RoboTwinPath).Path
Write-Host "RoboTwin path: $resolvedRoboTwin"
Write-Host "Expected base commit: $expectedBaseCommit"
Write-Host "Expected target commit: $expectedTargetCommit"

$states = @()
foreach ($file in $files) {
    $relative = $file.RelativePath
    $targetPath = Join-RelativePath -Root $resolvedRoboTwin -RelativePath $relative
    $sourcePath = Join-RelativePath -Root (Join-Path $bundleRoot 'files') -RelativePath $relative
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Bundled target file missing: $sourcePath"
    }
    $sourceHash = Get-Sha256OrNull -Path $sourcePath
    if ($sourceHash -cne $file.TargetSha256) {
        throw "Bundled target hash mismatch for $relative. Expected $($file.TargetSha256), got $sourceHash."
    }
    $currentHash = Get-Sha256OrNull -Path $targetPath
    if ($null -eq $currentHash) {
        throw "RoboTwin file missing: $targetPath"
    }
    $state =
        if ($currentHash -ceq $file.TargetSha256) {
            'target'
        } elseif ($currentHash -ceq $file.BaseSha256) {
            'base'
        } else {
            'unknown'
        }
    $states += [PSCustomObject]@{
        RelativePath = $relative
        CurrentSha256 = $currentHash
        State = $state
    }
}

$states | Format-Table -AutoSize

$unknown = @($states | Where-Object { $_.State -eq 'unknown' })
if ($unknown.Count -gt 0) {
    $bad = ($unknown | ForEach-Object { "$($_.RelativePath)=$($_.CurrentSha256)" }) -join ', '
    throw "Refusing to apply: files are neither the expected base nor target contents: $bad"
}

$baseFiles = @($states | Where-Object { $_.State -eq 'base' })
$targetFiles = @($states | Where-Object { $_.State -eq 'target' })
if ($baseFiles.Count -gt 0 -and $targetFiles.Count -gt 0) {
    throw "Refusing to apply: RoboTwin has a mixed base/target state. Restore one consistent version first."
}

if ($targetFiles.Count -eq $files.Count) {
    Write-Host 'RoboTwin already has the target delta-EE executor runtime patch.'
    exit 0
}

if (-not $Apply) {
    Write-Host 'DRY RUN: both RoboTwin files match the expected base contents.'
    Write-Host 'Re-run with -Apply to copy the target files from this bundle.'
    exit 0
}

if (Test-Path -LiteralPath (Join-Path $resolvedRoboTwin '.git')) {
    foreach ($file in $files) {
        $relative = $file.RelativePath
        $status = (& git -C $resolvedRoboTwin status --porcelain -- $relative | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "git status failed for $relative"
        }
        if ($status) {
            throw "Refusing to overwrite a dirty RoboTwin file: $relative status=$status"
        }
    }
}

foreach ($file in $files) {
    $relative = $file.RelativePath
    $targetPath = Join-RelativePath -Root $resolvedRoboTwin -RelativePath $relative
    $sourcePath = Join-RelativePath -Root (Join-Path $bundleRoot 'files') -RelativePath $relative
    Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
}

foreach ($file in $files) {
    $relative = $file.RelativePath
    $targetPath = Join-RelativePath -Root $resolvedRoboTwin -RelativePath $relative
    $currentHash = Get-Sha256OrNull -Path $targetPath
    if ($currentHash -cne $file.TargetSha256) {
        throw "Post-apply hash mismatch for $relative. Expected $($file.TargetSha256), got $currentHash."
    }
}

Write-Host 'Applied RoboTwin delta-EE executor runtime patch successfully.'
