@echo off
cd /d "%~dp0"
set PYTHONPATH=%CD%\src
set TF_CPP_MIN_LOG_LEVEL=2

if not exist ".venv\Scripts\python.exe" (
  echo The local Python environment is missing.
  echo See the Local GUI section in README.md.
  pause
  exit /b 1
)

if not exist "results\phase4\learn-5x5\checkpoints\iteration-0019-inference.h5" (
  echo The local inference checkpoint is missing.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m breakthrough_zero.cli gui ^
  --checkpoint "results\phase4\learn-5x5\checkpoints\iteration-0019-inference.h5" ^
  --simulations 100

if errorlevel 1 pause
