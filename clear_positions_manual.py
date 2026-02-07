#!/usr/bin/env python3
"""
Clear all active positions and reset the bot state
"""
import json
import os
from datetime import datetime

def clear_all_positions():
    """Clear all active positions and reset counters"""
    
    # Path to the state file
    state_file = "data/active_positions_state.json"
    
    # Create clean state
    clean_state = {
        "timestamp": datetime.now().isoformat(),
        "total_signals": 0,
        "successful_trades": 0,
        "paper_balance": 10000.0,
        "realized_pnl_today": 0.0,
        "start_of_day_balance": 10000.0,
        "daily_loss_limit_pct": 0.1,
        "circuit_breaker_state": None,
        "active_positions": {}
    }
    
    # Write the clean state
    try:
        with open(state_file, 'w') as f:
            json.dump(clean_state, f, indent=2)
        print("✅ All active positions cleared!")
        print("✅ State counters reset!")
        print("✅ Paper balance reset to $10,000")
        print(f"✅ State file updated: {state_file}")
    except Exception as e:
        print(f"❌ Error clearing positions: {e}")

if __name__ == "__main__":
    clear_all_positions()
