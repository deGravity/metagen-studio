@echo off
REM Launch the studio Vite dev server. Called by run.bat (or run directly).
REM Requires `npm install` to have been run once in studio/frontend.
REM
REM Env:
REM   STUDIO_FRONTEND_PORT  port (default 5173)

setlocal EnableDelayedExpansion

set "STUDIO_DIR=%~dp0"

if "%STUDIO_FRONTEND_PORT%"=="" set "STUDIO_FRONTEND_PORT=5173"

cd /d "%STUDIO_DIR%frontend"
if not exist node_modules (
  echo node_modules not found. Run `npm install` in studio\frontend first.
  pause
  exit /b 1
)

echo [studio frontend] vite on port %STUDIO_FRONTEND_PORT%
call npm run dev -- --port %STUDIO_FRONTEND_PORT% --host

echo.
echo Frontend exited. Press any key to close this window.
pause >nul
endlocal
