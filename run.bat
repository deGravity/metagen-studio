@echo off
REM Windows launcher for the metaDSL Studio (counterpart of run.sh).
REM
REM Opens two new console windows — one for the FastAPI backend (uvicorn
REM with --reload) and one for the Vite frontend dev server. Close either
REM window to stop that half. Closing this launcher window does NOT stop
REM the children (run.sh has the same property).
REM
REM Env vars (optional; same names as run.sh):
REM   STUDIO_PY              full path to the Python interpreter
REM                          (default: %CONDA_PREFIX%\python.exe if set,
REM                          else %USERPROFILE%\anaconda3\envs\metamaterials-dev-windows-gpu\python.exe)
REM   STUDIO_BACKEND_PORT    backend port (default 8000)
REM   STUDIO_FRONTEND_PORT   frontend port (default 5173)
REM   METAGEN_ANTHROPIC_API_KEY  enables the Copilot tab

setlocal EnableDelayedExpansion

set "STUDIO_DIR=%~dp0"

if "%STUDIO_BACKEND_PORT%"==""  set "STUDIO_BACKEND_PORT=8000"
if "%STUDIO_FRONTEND_PORT%"=="" set "STUDIO_FRONTEND_PORT=5173"

echo [studio] backend  -^> http://localhost:%STUDIO_BACKEND_PORT%
echo [studio] frontend -^> http://localhost:%STUDIO_FRONTEND_PORT%
echo.
echo Two new console windows will open. Close either window to stop that
echo server. Re-run this launcher to restart.
echo.

start "studio backend"  "%STUDIO_DIR%run-backend.bat"
start "studio frontend" "%STUDIO_DIR%run-frontend.bat"

endlocal
exit /b 0
