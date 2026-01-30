@echo off
REM ╔═══════════════════════════════════════════════════════════╗
REM ║   WOLF_AI Command Center - Windows Launcher               ║
REM ╚═══════════════════════════════════════════════════════════╝

cd /d E:\WOLF_AI

echo.
echo   🐺 WOLF_AI Command Center
echo   ═════════════════════════════════════
echo.
echo   1. API Server only
echo   2. API + Tunnel (for phone access)
echo   3. API + Telegram Bot
echo   4. Everything (API + Tunnel + Telegram)
echo   5. Just Telegram Bot
echo.
set /p choice="   Select option (1-5): "

if "%choice%"=="1" (
    echo Starting API Server...
    python run_server.py
)
if "%choice%"=="2" (
    echo Starting API Server with Tunnel...
    python run_server.py --tunnel
)
if "%choice%"=="3" (
    echo Starting API Server with Telegram Bot...
    python run_server.py --telegram
)
if "%choice%"=="4" (
    echo Starting Everything...
    python run_server.py --all
)
if "%choice%"=="5" (
    echo Starting Telegram Bot only...
    python -m telegram.bot
)

pause
