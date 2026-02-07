@echo off
echo 🧹 CLEARING OLD POSITIONS FOR NEW ANALYSIS SESSION
echo ================================================

REM Backup current state (just in case)
if exist "data\active_positions_state.json" (
    echo 💾 Backing up current positions...
    copy "data\active_positions_state.json" "data\positions_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%.json" >nul
    echo ✅ Backup saved
)

REM Clear the state file
if exist "data\active_positions_state.json" (
    del "data\active_positions_state.json"
    echo ✅ Old positions cleared
) else (
    echo ℹ️ No existing positions to clear
)

echo.
echo 🔥 Ready for fresh analysis session!
echo    - All old positions removed
echo    - Bot will start with clean slate
echo    - Previous positions backed up (if any)
echo.
echo ⏭️ Now run 'restart_demo_bot.bat' to start fresh
pause
