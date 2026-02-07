@echo off
echo 🔒 MANUAL POSITION CLOSER
echo ========================

if "%1"=="" (
    echo Usage: close_position.bat [TICKER]
    echo Example: close_position.bat FUN/USDT
    echo Example: close_position.bat MSFT
    echo.
    echo Current positions:
    if exist "data\active_positions_state.json" (
        type "data\active_positions_state.json" | findstr "ticker" | findstr /v "timestamp"
    ) else (
        echo No active positions found
    )
    pause
    exit /b
)

set TICKER=%1
echo.
echo 🔒 Manually closing position: %TICKER%
echo.

REM Create a close signal in the logs
echo %date% %time% - INFO - MANUAL_CLOSE - 🔒 Manual close signal: %TICKER% >> logs\big_boss_ceo_20250804.log

REM Update the state file to mark position as closed
python -c "
import json
from pathlib import Path
from datetime import datetime

state_file = Path('data/active_positions_state.json')
if state_file.exists():
    with open(state_file, 'r') as f:
        data = json.load(f)
    
    closed_count = 0
    for pos_id, pos in data['active_positions'].items():
        ticker = pos['ticker']
        if (ticker == '%TICKER%' or 
            ticker.split('/')[0] == '%TICKER%' or 
            '%TICKER%'.split('/')[0] == ticker or
            ticker.replace('/', '') == '%TICKER%'.replace('/', '')):
            pos['status'] = 'CLOSED'
            closed_count += 1
            print(f'✅ Closed position: {pos_id} ({ticker})')
    
    if closed_count > 0:
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'💾 State updated: {closed_count} position(s) marked as closed')
    else:
        print('❌ No matching positions found for: %TICKER%')
else:
    print('❌ No state file found')
"

echo.
echo ✅ Position close request completed
echo 🔄 Restart bot to apply changes and clean up closed positions
pause
