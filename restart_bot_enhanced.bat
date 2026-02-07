@echo off
echo 🔄 Restarting Big Boss CEO Bot with Enhanced Image Processing...
echo.

REM Kill any existing Python processes
taskkill /f /im python.exe >nul 2>&1

REM Wait a moment
timeout /t 2 >nul

REM Start the bot with virtual environment
echo 🚀 Starting bot with enhanced image processing...
start "Big Boss CEO - Enhanced" ".venv\Scripts\python.exe" grizzlies_bot.py

echo.
echo ✅ Bot restarted with enhanced image processing!
echo 🖼️ Now extracts: PnL, ROI, Mark Price, Entry Price, Position Size, Leverage
echo 📊 Watch for detailed "EXTRACTED TRADING DATA" logs
echo.
pause
