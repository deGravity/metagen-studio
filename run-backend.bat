@echo off
REM Launch the studio FastAPI backend. Called by run.bat (or run directly).
REM
REM Reads from the parent environment if set:
REM   STUDIO_PY            python interpreter path
REM   STUDIO_BACKEND_PORT  port (default 8000)
REM   METAGEN_ANTHROPIC_API_KEY  enables /api/chat tab in the UI

setlocal EnableDelayedExpansion

set "STUDIO_DIR=%~dp0"

if "%STUDIO_PY%"=="" (
  if not "%CONDA_PREFIX%"=="" (
    set "STUDIO_PY=%CONDA_PREFIX%\python.exe"
  ) else (
    set "STUDIO_PY=%USERPROFILE%\anaconda3\envs\metamaterials-dev-windows-gpu\python.exe"
  )
)
if not exist "%STUDIO_PY%" (
  echo Python interpreter not found at "%STUDIO_PY%".
  echo Set STUDIO_PY env var or activate the metamaterials-dev-windows-gpu env.
  pause
  exit /b 1
)

REM Derive the conda env root from STUDIO_PY (env_root\python.exe).
for %%I in ("%STUDIO_PY%") do set "STUDIO_ENV=%%~dpI"
if "%STUDIO_ENV:~-1%"=="\" set "STUDIO_ENV=%STUDIO_ENV:~0,-1%"

REM Prepend the env's Library\bin to PATH so the uvicorn --reload worker
REM subprocess (which inherits this env) can find conda-installed DLLs
REM when importing metagen_kernel / metagen_simulator / numpy/MKL etc.
REM Without this the worker crashes silently during import and the
REM reloader stays alive but never serves.
set "PATH=%STUDIO_ENV%;%STUDIO_ENV%\Library\bin;%STUDIO_ENV%\Library\mingw-w64\bin;%STUDIO_ENV%\Scripts;%PATH%"

if "%STUDIO_BACKEND_PORT%"=="" set "STUDIO_BACKEND_PORT=8000"

echo [studio backend] %STUDIO_PY% on port %STUDIO_BACKEND_PORT%
REM cd into the backend so uvicorn's --reload watcher anchors there.
REM Without this, --reload watches CWD (whatever opened the window) and
REM if that happens to be a directory with node_modules / build outputs
REM the WatchFiles tree-walk prevents the worker from starting.
cd /d "%STUDIO_DIR%backend"
"%STUDIO_PY%" -m uvicorn studio_backend.main:app ^
  --host 0.0.0.0 --port %STUDIO_BACKEND_PORT% ^
  --reload --reload-dir "%STUDIO_DIR%backend\studio_backend"

REM If uvicorn exits, pause so the user can see the error.
echo.
echo Backend exited. Press any key to close this window.
pause >nul
endlocal
