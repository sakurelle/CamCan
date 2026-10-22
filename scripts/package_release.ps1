param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^v\d+\.\d+\.\d+$')]
    [string]$Tag
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = Join-Path $ProjectRoot "dist\MarkerQC"
$Exe = Join-Path $DistRoot "MarkerQC.exe"
$Internal = Join-Path $DistRoot "_internal"
if (-not (Test-Path -LiteralPath $Exe)) { throw "MarkerQC.exe не найден: $Exe" }
if (-not (Test-Path -LiteralPath $Internal -PathType Container)) { throw "Каталог _internal не найден: $Internal" }

$VersionLine = Select-String -LiteralPath (Join-Path $ProjectRoot "pyproject.toml") -Pattern '^version\s*=\s*"([^\"]+)"$' | Select-Object -First 1
if (-not $VersionLine) { throw "Не удалось прочитать версию из pyproject.toml" }
$ProjectVersion = $VersionLine.Matches[0].Groups[1].Value
if ($Tag -ne "v$ProjectVersion") { throw "Тег $Tag не соответствует версии проекта v$ProjectVersion" }

$ReleaseDir = Join-Path $ProjectRoot "release"
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
$ArchiveName = "MarkerQC-$Tag-win64.zip"
$ArchivePath = Join-Path $ReleaseDir $ArchiveName
$ChecksumPath = "$ArchivePath.sha256"
Compress-Archive -LiteralPath $DistRoot -DestinationPath $ArchivePath -CompressionLevel Optimal -Force
$Hash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath $ChecksumPath -Value "$Hash  $ArchiveName" -Encoding ascii

$FolderBytes = (Get-ChildItem -LiteralPath $DistRoot -File -Recurse | Measure-Object -Property Length -Sum).Sum
$ZipBytes = (Get-Item -LiteralPath $ArchivePath).Length
Write-Output "RELEASE_ZIP=$ArchivePath"
Write-Output "RELEASE_SHA256=$ChecksumPath"
Write-Output "APP_FOLDER_BYTES=$FolderBytes"
Write-Output "ZIP_BYTES=$ZipBytes"
