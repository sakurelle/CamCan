param(
    [string]$PythonExecutable = "",
    [switch]$OneFile,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$OperatorGuide = Join-Path $ProjectRoot "docs\Руководство оператора.pdf"

if (-not (Test-Path -LiteralPath $OperatorGuide -PathType Leaf)) {
    throw "Актуальное руководство оператора не найдено: $OperatorGuide"
}

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
    Copy-Item -LiteralPath $OperatorGuide -Destination $DistDocs -Force

    $ReleaseDocs = @(Get-ChildItem -LiteralPath $DistDocs -File)
    if ($ReleaseDocs.Count -ne 1 -or $ReleaseDocs[0].Name -ne "Руководство оператора.pdf") {
        throw "В каталоге релизной документации должен находиться только файл Руководство оператора.pdf"
    }
    $PdfCopies = @(Get-ChildItem -LiteralPath $DistRoot -File -Recurse | Where-Object Name -eq "Руководство оператора.pdf")
    if ($PdfCopies.Count -ne 1 -or $PdfCopies[0].FullName -ne (Join-Path $DistDocs "Руководство оператора.pdf")) {
        throw "PDF-инструкция должна находиться в релизе ровно один раз, в каталоге docs"
    }
}

Write-Host "Сборка готова в каталоге dist." -ForegroundColor Green

