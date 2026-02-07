@echo off
REM ==============================================
REM BIG BOSS CEO - SELFTEST STATUS HELP SCRIPT
REM Shows latest bot log self-test lines and flag.
REM Usage:  check_selftest_status.bat
REM ==============================================
if not exist logs (
  echo logs folder missing.
  exit /b 1
)
set BOTLOG=
for /f "delims=" %%F in ('dir /b /o-d logs\bot_*.log 2^>nul') do (
  set "BOTLOG=logs\%%F"
  goto gotlog
)
:gotlog
if not defined BOTLOG (
  echo No bot logs found.
  exit /b 1
)
echo Latest bot log: %BOTLOG%
findstr /C:"SELFTEST config" /C:"SELFTEST trigger" /C:"SELFTEST PASS" /C:"SELFTEST FAILED" "%BOTLOG%" 2>nul
if exist logs\selftest_passed.flag (
  echo --- Flag file contents ---
  type logs\selftest_passed.flag
) else (
  echo Flag file not present yet.
)
exit /b 0
