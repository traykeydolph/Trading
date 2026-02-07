@echo off
setlocal

rem --- Session env (safe defaults for today) ---
set LOG_LEVEL=DEBUG
set PAPER_TRADING=true
set ASSET_CLASS_MODE=both
rem Lower threshold so borderline signals can execute during debug
set GRIZZLIES_CONFIDENCE_THRESHOLD=0.5
rem Reasonable paper size so Binance min notional is met on majors
set GRIZZLIES_POSITION_SIZE=100

:loop
echo Starting BIG BOSS CEO bot...
"C:\Users\User\BIG BOSS CEO\.venv\Scripts\python.exe" grizzlies_bot.py
echo Bot exited. Restarting in 10 seconds...
timeout /t 10 /nobreak >nul
goto loop