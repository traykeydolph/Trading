"""
🎯 OPTIONS TRADING ENGINE - PROPER OPTIONS HANDLING
==================================================
Based on comprehensive options trading framework
Handles parsing, position sizing, and execution correctly
==================================================
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("options_engine")

@dataclass
class OptionsSignal:
    """Properly parsed options signal"""
    symbol: str           # e.g., "INTC"
    strike: float         # e.g., 25.0
    option_type: str      # "call" or "put"
    expiry: datetime      # e.g., 2025-08-22
    premium: float        # e.g., 0.50
    raw_signal: str       # Original signal text
    confidence: float = 85.0

class OptionsParser:
    """Parse options signals correctly"""
    
    def __init__(self):
        # Regex patterns for options signals (expanded for better detection)
        self.patterns = [
            # Standard formats
            # INTC 25c 8/22/25 @0.50
            r'([A-Z]{1,5})\s+(\d+(?:\.\d+)?)([cp])\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+@(\d+(?:\.\d+)?)',
            # INTC $25 call 8/22/25 @0.50
            r'([A-Z]{1,5})\s+\$?(\d+(?:\.\d+)?)\s+(call|put)\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+@(\d+(?:\.\d+)?)',
            
            # More flexible patterns for incomplete signals
            # HOOD 25c (assume near expiry if no date)
            r'([A-Z]{1,5})\s+(\d+(?:\.\d+)?)([cp])\s*$',
            # HOOD $25 call (assume near expiry)
            r'([A-Z]{1,5})\s+\$?(\d+(?:\.\d+)?)\s+(call|put)\s*$',
            # HOOD calls (generic call signal, use ATM strike)
            r'([A-Z]{1,5})\s+(calls?|puts?)\s*$',
        ]
        
        # Default values for incomplete signals
        self.default_premium = 0.50  # Default premium if not specified
        self.default_days_to_expiry = 7  # Default to weekly options
    
    def parse_options_signal(self, message: str) -> Optional[OptionsSignal]:
        """Parse options signal from Discord message with flexible pattern matching"""
        try:
            message = message.strip().upper()
            
            # Pattern 1: Complete format with expiry and premium
            # INTC 25c 8/22/25 @0.50
            pattern1 = r'([A-Z]{1,5})\s+(\d+(?:\.\d+)?)([cp])\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+@(\d+(?:\.\d+)?)'
            match = re.search(pattern1, message, re.IGNORECASE)
            if match:
                symbol = match.group(1)
                strike = float(match.group(2))
                option_type = 'call' if match.group(3).lower() == 'c' else 'put'
                expiry = self._parse_expiry_date(match.group(4))
                premium = float(match.group(5))
                
                return OptionsSignal(symbol=symbol, strike=strike, option_type=option_type, 
                                   expiry=expiry, premium=premium, raw_signal=message)
            
            # Pattern 2: Complete format with call/put words
            # INTC $25 call 8/22/25 @0.50
            pattern2 = r'([A-Z]{1,5})\s+\$?(\d+(?:\.\d+)?)\s+(call|put)\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+@(\d+(?:\.\d+)?)'
            match = re.search(pattern2, message, re.IGNORECASE)
            if match:
                symbol = match.group(1)
                strike = float(match.group(2))
                option_type = match.group(3).lower()
                expiry = self._parse_expiry_date(match.group(4))
                premium = float(match.group(5))
                
                return OptionsSignal(symbol=symbol, strike=strike, option_type=option_type, 
                                   expiry=expiry, premium=premium, raw_signal=message)
            
            # Pattern 3: Strike and type only (no expiry/premium)
            # HOOD 25c or HOOD $25 call
            pattern3 = r'([A-Z]{1,5})\s+\$?(\d+(?:\.\d+)?)([cp]|call|put)\s*$'
            match = re.search(pattern3, message, re.IGNORECASE)
            if match:
                symbol = match.group(1)
                strike = float(match.group(2))
                option_type_raw = match.group(3).lower()
                
                if option_type_raw in ['c', 'call']:
                    option_type = 'call'
                elif option_type_raw in ['p', 'put']:
                    option_type = 'put'
                else:
                    return None
                
                # Use defaults for missing data
                expiry = datetime.now() + timedelta(days=self.default_days_to_expiry)
                premium = self.default_premium
                
                logger.info(f"🎯 Parsed incomplete options signal: {symbol} ${strike} {option_type} (using defaults)")
                return OptionsSignal(symbol=symbol, strike=strike, option_type=option_type, 
                                   expiry=expiry, premium=premium, raw_signal=message)
            
            # Pattern 4: Generic calls/puts (determine strike from current price)
            # HOOD calls or HOOD puts
            pattern4 = r'([A-Z]{1,5})\s+(calls?|puts?)\s*$'
            match = re.search(pattern4, message, re.IGNORECASE)
            if match:
                symbol = match.group(1)
                option_type = 'call' if 'call' in match.group(2).lower() else 'put'
                
                # Try to get current stock price for ATM strike
                try:
                    from exchange_api import UnifiedExchangeAPI
                    exchange = UnifiedExchangeAPI()
                    current_price = exchange.get_current_price(symbol)
                    
                    if current_price > 0:
                        # Round to nearest $5 for options strike
                        strike = round(current_price / 5) * 5
                    else:
                        # Fallback strike
                        strike = 25.0
                except:
                    strike = 25.0  # Default strike if price lookup fails
                
                expiry = datetime.now() + timedelta(days=self.default_days_to_expiry)
                premium = self.default_premium
                
                logger.info(f"🎯 Parsed generic options signal: {symbol} ${strike} {option_type} (estimated strike)")
                return OptionsSignal(symbol=symbol, strike=strike, option_type=option_type, 
                                   expiry=expiry, premium=premium, raw_signal=message)
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing options signal: {e}")
            return None
    
    def _parse_expiry_date(self, date_str: str) -> datetime:
        """Parse expiry date from various formats"""
        try:
            # Handle formats like 8/22/25, 8/22/2025
            parts = date_str.split('/')
            month, day = int(parts[0]), int(parts[1])
            year = int(parts[2])
            
            # Handle 2-digit vs 4-digit years
            if year < 100:
                year += 2000
            
            return datetime(year, month, day)
            
        except Exception as e:
            logger.error(f"Error parsing expiry date {date_str}: {e}")
            # Default to 30 days from now if parsing fails
            return datetime.now() + timedelta(days=30)

class OptionsPositionSizer:
    """Calculate proper position sizes for options based on risk management"""
    
    def __init__(self):
        self.fixed_contracts = os.getenv('OPTIONS_FIXED_CONTRACTS')
        self.risk_per_trade_pct = float(os.getenv('OPTIONS_RISK_PER_TRADE_PCT', '0.02'))
        self.max_contracts = int(os.getenv('OPTIONS_MAX_CONTRACTS', '10'))
    
    def calculate_position_size(self, signal: OptionsSignal, account_balance: float) -> Dict:
        """Calculate position size using proper risk management
        
        Returns:
            Dict with contracts, cost, risk_amount, etc.
        """
        try:
            # Contract cost = premium × 100 (multiplier)
            contract_cost = signal.premium * 100
            
            # Check if fixed contracts mode is enabled
            if self.fixed_contracts:
                contracts = int(self.fixed_contracts)
                logger.info(f"📊 FIXED CONTRACT MODE: Using {contracts} contract(s)")
            else:
                # Risk allocation (e.g., 2% of account)
                risk_amount = account_balance * self.risk_per_trade_pct
                
                # Max contracts based on risk
                max_contracts_by_risk = int(risk_amount / contract_cost)
                
                # Apply safety limits
                contracts = min(max_contracts_by_risk, self.max_contracts)
                
                # Ensure at least 1 contract if risk allows
                contracts = max(contracts, 1) if contract_cost <= risk_amount else 0
            
            total_cost = contracts * contract_cost
            
            logger.info(f"📊 Options Position Sizing:")
            if self.fixed_contracts:
                logger.info(f"   Mode: FIXED CONTRACTS")
                logger.info(f"   Contracts: {contracts} (fixed)")
            else:
                risk_amount = account_balance * self.risk_per_trade_pct
                logger.info(f"   Risk Amount: ${risk_amount:.2f} ({self.risk_per_trade_pct*100:.1f}%)")
            logger.info(f"   Contract Cost: ${contract_cost:.2f}")
            logger.info(f"   Total Cost: ${total_cost:.2f}")
            
            return {
                'contracts': contracts,
                'contract_cost': contract_cost,
                'total_cost': total_cost,
                'risk_amount': total_cost,  # In fixed mode, risk = total cost
                'risk_pct': (total_cost / account_balance * 100) if account_balance > 0 else 0,
                'underlying_exposure': contracts * 100 * signal.strike,
                'effective_leverage': (contracts * 100 * signal.strike) / total_cost if total_cost > 0 else 0,
                'mode': 'fixed' if self.fixed_contracts else 'risk_based'
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {'contracts': 0, 'error': str(e)}

class OptionsOrderManager:
    """Manage options order placement with proper formatting"""
    
    def __init__(self):
        self.use_limit_orders = os.getenv('OPTIONS_USE_LIMIT_ORDERS', 'true').lower() == 'true'
        self.stop_loss_pct = float(os.getenv('OPTIONS_STOP_LOSS_PCT', '0.50'))
        self.tp1_pct = float(os.getenv('OPTIONS_TP1_PCT', '1.00'))
        self.tp2_pct = float(os.getenv('OPTIONS_TP2_PCT', '2.00'))
    
    def format_options_symbol(self, signal: OptionsSignal) -> str:
        """Format options symbol for Alpaca API
        
        Example: INTC250822C00025000
        Format: SYMBOL + YYMMDD + C/P + STRIKE (8 digits)
        """
        try:
            # Symbol (padded to ensure proper length)
            symbol = signal.symbol.ljust(6)[:6]
            
            # Date in YYMMDD format
            date_str = signal.expiry.strftime('%y%m%d')
            
            # Call/Put
            cp = 'C' if signal.option_type == 'call' else 'P'
            
            # Strike price (8 digits: 5 before decimal, 3 after, padded with zeros)
            strike_formatted = f"{int(signal.strike * 1000):08d}"
            
            return f"{symbol}{date_str}{cp}{strike_formatted}"
            
        except Exception as e:
            logger.error(f"Error formatting options symbol: {e}")
            return f"{signal.symbol}_{signal.option_type}_{signal.strike}"
    
    def create_options_order(self, signal: OptionsSignal, position_info: Dict) -> Dict:
        """Create properly formatted options order"""
        try:
            if position_info['contracts'] <= 0:
                return {'status': 'error', 'message': 'Invalid position size'}
            
            # Format symbol for Alpaca
            options_symbol = self.format_options_symbol(signal)
            
            # Create order parameters
            order = {
                'symbol': options_symbol,
                'qty': position_info['contracts'],
                'side': 'buy',  # Always buying options to open
                'type': 'limit' if self.use_limit_orders else 'market',
                'time_in_force': 'day',
                'limit_price': signal.premium if self.use_limit_orders else None,
                # Risk management levels
                'stop_loss_price': signal.premium * (1 - self.stop_loss_pct),
                'take_profit_1': signal.premium * (1 + self.tp1_pct),
                'take_profit_2': signal.premium * (1 + self.tp2_pct),
                # Metadata
                'signal_info': {
                    'underlying': signal.symbol,
                    'strike': signal.strike,
                    'expiry': signal.expiry.isoformat(),
                    'option_type': signal.option_type,
                    'premium': signal.premium
                }
            }
            
            logger.info(f"📋 Options Order Created:")
            logger.info(f"   Symbol: {options_symbol}")
            logger.info(f"   Qty: {position_info['contracts']} contracts")
            logger.info(f"   Premium: ${signal.premium}")
            logger.info(f"   Stop Loss: ${order['stop_loss_price']:.2f}")
            logger.info(f"   Take Profit 1: ${order['take_profit_1']:.2f}")
            logger.info(f"   Take Profit 2: ${order['take_profit_2']:.2f}")
            
            return {'status': 'success', 'order': order}
            
        except Exception as e:
            logger.error(f"Error creating options order: {e}")
            return {'status': 'error', 'message': str(e)}

class OptionsTradeManager:
    """Complete options trading workflow"""
    
    def __init__(self, account_balance: float = 10000):
        self.parser = OptionsParser()
        self.sizer = OptionsPositionSizer()
        self.order_manager = OptionsOrderManager()
        self.account_balance = account_balance
    
    def process_options_signal(self, message: str) -> Dict:
        """Complete options signal processing workflow"""
        try:
            # 1. Parse the signal
            signal = self.parser.parse_options_signal(message)
            if not signal:
                return {'status': 'not_options', 'message': 'Not an options signal'}
            
            logger.info(f"🎯 Options Signal Detected:")
            logger.info(f"   {signal.symbol} ${signal.strike} {signal.option_type} {signal.expiry.strftime('%m/%d/%y')} @${signal.premium}")
            
            # 2. Calculate position size
            position_info = self.sizer.calculate_position_size(signal, self.account_balance)
            if position_info['contracts'] <= 0:
                return {'status': 'error', 'message': 'Position size too small or invalid'}
            
            # 3. Create order
            order_result = self.order_manager.create_options_order(signal, position_info)
            if order_result['status'] != 'success':
                return order_result
            
            # 4. Return complete trade package
            return {
                'status': 'success',
                'signal': signal,
                'position_info': position_info,
                'order': order_result['order'],
                'trade_summary': {
                    'symbol': signal.symbol,
                    'strike': signal.strike,
                    'option_type': signal.option_type,
                    'expiry': signal.expiry.strftime('%m/%d/%Y'),
                    'premium': signal.premium,
                    'contracts': position_info['contracts'],
                    'total_cost': position_info['total_cost'],
                    'risk_pct': position_info['risk_pct'],
                    'underlying_exposure': position_info['underlying_exposure']
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing options signal: {e}")
            return {'status': 'error', 'message': str(e)}

# Test function to validate the framework
def test_options_framework():
    """Test the options framework with sample signals"""
    manager = OptionsTradeManager(account_balance=10000)
    
    test_signals = [
        "INTC 25c 8/22/25 @0.50",
        "AAPL $150 call 9/15/25 @2.25",
        "TSLA 200p 10/20/25 @5.00"
    ]
    
    for signal in test_signals:
        print(f"\n🧪 Testing: {signal}")
        result = manager.process_options_signal(signal)
        
        if result['status'] == 'success':
            summary = result['trade_summary']
            print(f"✅ SUCCESS:")
            print(f"   Contracts: {summary['contracts']}")
            print(f"   Total Cost: ${summary['total_cost']:.2f}")
            print(f"   Risk: {summary['risk_pct']:.1f}%")
            print(f"   Exposure: ${summary['underlying_exposure']:,.0f}")
        else:
            print(f"❌ FAILED: {result['message']}")

if __name__ == "__main__":
    test_options_framework()
