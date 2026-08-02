@echo off
REM ═══════════════════════════════════════════════════════════════
REM  DOAR — Windows Setup Script
REM  Run this ONCE from the DOAR-main folder in VS Code terminal.
REM  It creates a virtual environment and installs all dependencies.
REM ═══════════════════════════════════════════════════════════════

echo.
echo ══════════════════════════════════════════════════
echo  DOAR v2 — Windows Setup
echo ══════════════════════════════════════════════════

REM Step 1: Create virtual environment
echo.
echo [1/4] Creating Python virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Could not create virtual environment.
    echo Make sure Python 3.11 is installed: https://python.org/downloads
    pause
    exit /b 1
)

REM Step 2: Activate venv
echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Step 3: Upgrade pip
echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

REM Step 4: Install requirements
echo.
echo [4/4] Installing required packages...
pip install opencv-python Pillow numpy matplotlib
pip install easyocr
pip install sentence-transformers
pip install deep-translator
pip install gradio
pip install tqdm

echo.
echo ══════════════════════════════════════════════════
echo  Setup complete!
echo ══════════════════════════════════════════════════
echo.
echo Next steps:
echo   1. Activate the environment:
echo      venv\Scripts\activate
echo.
echo   2. Edit pipeline.py lines 25-26 to set your paths:
echo      DATASET_ROOT = "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing"
echo      OUTPUT_DIR   = "outputs"
echo.
echo   3. Test on one image:
echo      python pipeline.py --image "path\to\drawing.jpg"
echo.
echo   4. Run on the full dataset:
echo      python analyze_dataset.py --max 5
echo.
echo   5. Launch the UI:
echo      python ui\app.py
echo.
pause
