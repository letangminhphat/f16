$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$hlsInclude = if ($env:VITIS_HLS_INCLUDE) {
    $env:VITIS_HLS_INCLUDE
} else {
    'D:\Xilinx\Vitis_HLS\2023.1\include'
}
if (-not (Test-Path (Join-Path $hlsInclude 'ap_int.h'))) {
    throw "Set VITIS_HLS_INCLUDE to a Vitis HLS include directory"
}
if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
    throw 'g++ is required for the code-only syntax check'
}

Push-Location $root
try {
    foreach ($file in Get-ChildItem '.\code\*.cpp') {
        Write-Host "Checking $($file.Name)"
        & g++ -std=c++14 -Wno-unknown-pragmas '-I.\code' `
            "-I$hlsInclude" -fsyntax-only $file.FullName
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & python -c "import ast,pathlib; ast.parse(pathlib.Path(r'scripts/check_budget.py').read_text(encoding='utf-8'))"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & python '.\scripts\check_architecture.py'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $placeholder = & rg -n -i 'TODO|FIXME|placeholder|pseudocode' `
        code config 2>$null
    if ($LASTEXITCODE -eq 0) {
        $placeholder
        throw 'Placeholder text remains in implementation files'
    }
    Write-Host 'PASS: all canonical code/ sources compile and no placeholders remain.'
} finally {
    Pop-Location
}
