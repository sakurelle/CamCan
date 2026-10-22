param(
    [string]$PythonExecutable = "",
    [switch]$OneFile,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not $PythonExecutable) {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $PythonExecutable = & py -3.12 -c "import sys; print(sys.executable)"
        if ($LASTEXITCODE -ne 0) { throw "Python 3.12 x64 не найден. Установите его или задайте -PythonExecutable." }
    } else {
        $PythonExecutable = (Get-Command python -ErrorAction Stop).Source
    }
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $PythonExecutable -m venv .venv
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Не удалось установить зависимости из requirements.txt." }

if (-not $SkipTests) {
    & $VenvPython -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Тесты не прошли; сборка остановлена." }
}

if ($OneFile) {
    & $VenvPython -m PyInstaller --noconfirm --clean MarkerQC-onefile.spec
} else {
    & $VenvPython -m PyInstaller --noconfirm --clean MarkerQC.spec
}

if ($LASTEXITCODE -ne 0) { throw "PyInstaller завершился с ошибкой." }

if (-not $OneFile) {
    $DistRoot = Join-Path $ProjectRoot "dist\MarkerQC"
    $DistDocs = Join-Path $DistRoot "docs"
    New-Item -ItemType Directory -Force -Path $DistDocs | Out-Null
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "README.md") -Destination $DistRoot -Force
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs\OPERATOR_GUIDE.md") -Destination $DistDocs -Force
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs\METROLOGY_VALIDATION.md") -Destination $DistDocs -Force
}

Write-Host "Сборка готова в каталоге dist." -ForegroundColor Green

