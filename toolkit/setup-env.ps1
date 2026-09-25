# Builds the pinned export venv for audiosr 0.0.7 on Windows.
# Usage: pwsh -File toolkit/setup-env.ps1
$ErrorActionPreference = 'Continue'
$repo = Split-Path $PSScriptRoot -Parent
$venv = Join-Path $repo '.venv'

if (-not (Test-Path $venv)) {
    py -3.11 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
$python = Join-Path $venv 'Scripts\python.exe'

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt') -c (Join-Path $PSScriptRoot 'constraints.txt')
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

# Verify the pins survived resolution (audiosr deps must not bump numpy).
& $python -c "import numpy; assert numpy.__version__ == '1.23.5', numpy.__version__; print('numpy', numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw "numpy pin broken" }
& $python -c "import audiosr; print('audiosr OK')"
if ($LASTEXITCODE -ne 0) { throw "audiosr import failed" }
& $python -c "import onnxscript, onnx, onnxruntime, torch; print('torch', torch.__version__, 'onnx', onnx.__version__, 'onnxscript', onnxscript.__version__, 'onnxruntime', onnxruntime.__version__)"
if ($LASTEXITCODE -ne 0) { throw "export stack import failed" }
& $python (Join-Path $PSScriptRoot 'check_pins.py') (Join-Path $PSScriptRoot 'constraints.txt')
if ($LASTEXITCODE -ne 0) { throw "installed versions drifted from constraints.txt" }
Write-Host 'Environment ready.'
