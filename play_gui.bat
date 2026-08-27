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

if not exist "checkpoints\phase8" (
  echo The local 8x8 checkpoints are missing.
  echo Copy the accepted models into checkpoints\phase8 first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m breakthrough_zero.cli gui ^
  --checkpoint "checkpoints\phase8\iteration-0107.h5" ^
  --models-dir "checkpoints\phase8" ^
  --board-size 8 ^
  --simulations 256

if errorlevel 1 pause
