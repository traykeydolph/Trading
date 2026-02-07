@echo off
echo 📊 BIG BOSS CEO QUICK STATUS CHECK
echo ================================

REM Check if bot is running
tasklist /fi "ImageName eq python.exe" /fo csv | find /i "python.exe" >nul
if %ERRORLEVEL%==0 (
    echo ✅ Bot Status: RUNNING
) else (
    echo ❌ Bot Status: NOT RUNNING
)

REM Check recent activity
echo.
echo 📝 Recent Activity (Last 10 lines):
echo --------------------------------
tail -n 10 "logs\big_boss_ceo_20250804.log" 2>nul

REM Check active positions
echo.
echo 🧠 Checking Position Tracker State...
echo --------------------------------
findstr /i "Total Tracked" "logs\big_boss_ceo_20250804.log" | tail -n 1 2>nul

echo.
echo 💡 Tips:
echo - Run 'save_state.bat' before unplugging
echo - Run 'restore_state.bat' after moving
echo - Your active positions will be preserved!
echo ================================
pause
