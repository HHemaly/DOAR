# DOAR — Windows setup (PowerShell). Requires Python 3.11.
# Does NOT install PyTorch (see the CUDA note); it detects an existing torch and
# never replaces a working CUDA build. It never claims CUDA was verified — run
# `python main.py gpu-smoke` for that.

$ErrorActionPreference = "Stop"

Write-Host "=== DOAR Windows setup ===" -ForegroundColor Cyan

# 1. Require Python 3.11, avoiding a RoboDK-bundled Python.
$pyCmd = $null
foreach ($cand in @("py -3.11", "python")) {
    try {
        $ver = & cmd /c "$cand --version" 2>&1
        if ($ver -match "3\.11\.") {
            $exe = & cmd /c "$cand -c ""import sys;print(sys.executable)""" 2>&1
            if ($exe -match "(?i)robodk") {
                Write-Host "Skipping RoboDK Python: $exe" -ForegroundColor Yellow
                continue
            }
            $pyCmd = $cand; break
        }
    } catch { }
}
if (-not $pyCmd) { throw "Python 3.11 not found (and not RoboDK's). Install from python.org." }
Write-Host "Using: $pyCmd" -ForegroundColor Green

# 2. Create/reuse .venv.
if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv ..." -ForegroundColor Cyan
    & cmd /c "$pyCmd -m venv .venv"
} else {
    Write-Host ".venv already exists — reusing." -ForegroundColor Green
}
$py = ".\.venv\Scripts\python.exe"

# 3. Install non-PyTorch project extras.
& $py -m pip install --upgrade pip
& $py -m pip install -e ".[ml,cv,dev]"

# 4. Detect existing torch; never replace a working CUDA build.
$torchInfo = & $py -c "import importlib.util as u; s=u.find_spec('torch');
import sys
if s is None:
    print('MISSING')
else:
    import torch
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1
Write-Host "PyTorch: $torchInfo" -ForegroundColor Cyan
if ($torchInfo -match "MISSING") {
    Write-Host @"
PyTorch is NOT installed. Do not blindly install a CUDA wheel.
  1. Run: nvidia-smi   (note the CUDA Version shown top-right)
  2. Pick the matching wheel from https://pytorch.org/get-started/locally/
     Your Quadro P3200 (Pascal, CC 6.1) is supported by current torch.
     If the driver supports CUDA >= 12.1:
       $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     Otherwise use the cu118 or CPU wheel per the selector.
  3. Verify with: $py main.py gpu-smoke --output outputs\gpu
"@ -ForegroundColor Yellow
} else {
    Write-Host "Existing torch detected — leaving it untouched." -ForegroundColor Green
}

# 5. Verify imports + print interpreter.
& $py -c "import numpy, PIL, sklearn; print('core imports OK')"
& $py -c "import sys; print('interpreter:', sys.executable)"

# 6. Readiness checks (does not require the dataset).
& $py main.py check-training-readiness --output outputs\readiness

Write-Host "=== Setup complete. GPU path is NOT verified until gpu-smoke runs on CUDA. ===" -ForegroundColor Cyan
