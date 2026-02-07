@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ==============================================
REM BIG BOSS CEO - TRADING DAY LAUNCH WRAPPER
REM 1. Runs preflight_trading_readiness.py and checks PASS
REM 2. Optionally disables self-test after first success (if ENABLE_SELFTEST=1)
REM 3. Launches grizzlies_bot.py with timestamped log
REM 4. Keeps console open on failure for inspection
REM ==============================================

REM Delayed expansion already enabled above
REM --- Derive safe date/time stamp (YYYYMMDD_HHMMSS) using PowerShell (wmic removed on modern Windows) ---
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%I

REM --- Detect python executable (prefer python, fallback py) ---
REM Detect python safely (avoid complex && || nesting that can trip cmd parsing)
where python >nul 2>&1
if %ERRORLEVEL%==0 (
    set PY=python
) else (
    where py >nul 2>&1
    if %ERRORLEVEL%==0 (set PY=py) else (set PY=python)
)
@echo off
REM Simplified launcher to avoid CMD parsing issues
setlocal EnableExtensions

REM Timestamp
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%I
if not defined STAMP set STAMP=NO_TIME

REM Python detect
set PY=python
where %PY% >nul 2>&1 || (set PY=py)

set LOGDIR=logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set PREFLIGHT_LOG=%LOGDIR%\preflight_%STAMP%.log
set BOT_LOG=%LOGDIR%\bot_%STAMP%.log

echo ======================================================
echo  BIG BOSS CEO TRADING DAY LAUNCH
echo  Timestamp: %STAMP%
echo ======================================================
echo [1/2] Preflight...
%PY% preflight_trading_readiness.py > "%PREFLIGHT_LOG%" 2>&1
if errorlevel 1 goto preflight_fail
findstr /C:"PREFLIGHT_STATUS=PASS" "%PREFLIGHT_LOG%" >nul || goto preflight_fail
echo Preflight PASS (see %PREFLIGHT_LOG%)
goto launch_bot

:preflight_fail
echo Preflight FAILED. See %PREFLIGHT_LOG%
powershell -NoProfile -Command "Get-Content '%PREFLIGHT_LOG%' -Tail 40"
pause
goto :eof

:launch_bot
REM Self-test handling: Only disable after it has passed once (flag present)
if exist logs\selftest_passed.flag (
    echo Self-test flag found - logs\selftest_passed.flag - disabling future self-tests.
    powershell -NoProfile -Command "(Get-Content '.env') -replace '^ENABLE_SELFTEST=1$','ENABLE_SELFTEST=0' | Set-Content '.env'"
) else (
    echo No self-test flag found yet - keeping ENABLE_SELFTEST=1 for this launch.
)

echo [2/2] Starting bot (CTRL+C to stop)...
echo Logging to %BOT_LOG%
%PY% grizzlies_bot.py >> "%BOT_LOG%" 2>&1
set BOT_EXIT=%ERRORLEVEL%
echo Bot exited with code %BOT_EXIT%
if NOT %BOT_EXIT%==0 (
    powershell -NoProfile -Command "Get-Content '%BOT_LOG%' -Tail 40"
)
pause
endlocal
    echo Bot closed cleanly. Log archived at %BOT_LOG%
