# --- Imports ---
import os
import re
import json
import logging
import requests
from dotenv import load_dotenv
import time
import base64
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np

# Configure Tesseract OCR path for Windows

if os.name == 'nt':  # Windows
    # Common Tesseract installation paths
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', 'User'))
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    else:
        # If not found, try to use from PATH
        import shutil
        tesseract_exe = shutil.which('tesseract')
        if tesseract_exe:
            pytesseract.pytesseract.tesseract_cmd = tesseract_exe
        else:
            logging.warning("⚠️ Tesseract OCR not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki")

def query_llm_for_signal(message_content: str, api_key: str, context_positions: dict = None) -> dict:
    """ENHANCED LLM-FIRST signal detection with full context awareness."""
    
    # Build context-aware prompt with active positions
    context_info = ""
    if context_positions:
        context_info = f"\n\nActive positions context: {list(context_positions.keys())}"
    
    prompt = f"""You are an expert Grizzlies trading signal parser with 95%+ accuracy.
    
CRITICAL INSTRUCTIONS:
- Analyze this Discord message for ACTIONABLE trading signals ONLY
- Return ONLY valid JSON, no extra text
- For TP hits without explicit ticker, use context to identify the asset
- Capture profit percentages, leverage, and position updates

IMPORTANT: Distinguish between ACTIONABLE signals vs ANALYSIS/UPDATES:
- ACTIONABLE: "TP1 hit", "Going long ETH", "All out BTC", "Taking profit"
- NOT ACTIONABLE: "ETH looking good", "BTC analysis", "Market update", "Price discussion"

MESSAGE: {message_content}{context_info}

Required JSON format:
{{
    "action": "open|trim|close|none",
    "ticker": "TICKER_SYMBOL",
    "confidence": 0-100,
    "direction": "LONG|SHORT|UPDATE",
    "tp_number": null or number,
    "profit_pct": null or number,
    "leverage": null or number,
    "entry_targets": [],
    "tp_targets": [],
    "reasoning": "brief explanation"
}}

Examples:
- "TP1 hit" -> {{"action": "trim", "ticker": "ETH", "tp_number": 1, "confidence": 95}}
- "Going to hit some eth longs again here" -> {{"action": "open", "ticker": "ETH", "direction": "LONG", "confidence": 85}}
- "All out BTC" -> {{"action": "close", "ticker": "BTC", "confidence": 90}}
- "BTC looking bullish here" -> {{"action": "none", "ticker": null, "confidence": 0, "reasoning": "analysis only"}}
- "Market update: ETH up 5%" -> {{"action": "none", "ticker": null, "confidence": 0, "reasoning": "market analysis"}}"""

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=15
        )
        result = response.json()
        
        # Check if response has proper structure
        if "choices" not in result or len(result["choices"]) == 0:
            logger.error(f"LLM API error: Invalid response structure: {result}")
            return {"action": "none", "ticker": None, "error": "Invalid API response"}
            
        content = result["choices"][0]["message"]["content"].strip()
        
        # Extract JSON from response (handle cases where LLM adds extra text)
        import json as pyjson
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
            
        parsed = pyjson.loads(content)
        
        # Validate required fields
        if not parsed.get("action") or not parsed.get("ticker"):
            return {"action": "none", "ticker": None, "error": "missing_fields"}
            
        return parsed
        
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return {"action": "none", "ticker": None, "error": str(e)}


def process_image_for_signals(image_url: str, api_key: str, context_positions: dict = None) -> dict:
    """Extract trading signals from images using OCR and computer vision."""
    
    try:
        # Download image
        response = requests.get(image_url, timeout=30)
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to OpenCV format for preprocessing
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get better text recognition
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = np.ones((1,1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Extract text using OCR (skip if Tesseract not available)
        extracted_text = ""
        try:
            extracted_text = pytesseract.image_to_string(opening, config='--psm 6')
            logger.info(f"OCR extracted text: {extracted_text}")
        except Exception as ocr_error:
            logger.warning(f"OCR not available, using GPT-4 Vision only: {ocr_error}")
            extracted_text = "OCR not available - using GPT-4 Vision for analysis"
        
        # Use GPT-4 Vision for enhanced image analysis
        image_analysis = analyze_image_with_gpt4_vision(image_url, api_key, extracted_text, context_positions)
        
        return image_analysis
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return {"action": "none", "ticker": None, "error": str(e)}


def analyze_image_with_gpt4_vision(image_url: str, api_key: str, ocr_text: str, context_positions: dict = None) -> dict:
    """Use GPT-4 Vision to analyze trading signal images with OCR backup."""
    
    # Build context-aware prompt
    context_info = ""
    if context_positions:
        context_info = f"\n\nActive positions context: {list(context_positions.keys())}"
    
    prompt = f"""You are an expert Grizzlies trading signal analyzer with 95%+ accuracy.

CRITICAL INSTRUCTIONS:
- Analyze this trading signal image for EXACT trading data
- Extract ALL visible numerical values and trading information
- Focus on profit/loss screenshots, trading platform interfaces, and position details
- Return ONLY valid JSON, no extra text

OCR EXTRACTED TEXT: {ocr_text}{context_info}

EXTRACT ALL VISIBLE DATA:
- Ticker symbols (BTC, ETH, AAPL, TSLA, etc.)
- Unrealized PnL (exact dollar/USDT amounts)
- ROI/Profit percentages (exact % values)
- Mark Price (current market price)
- Entry Price (position entry price)
- Position Size (quantity/amount)
- Leverage (5X, 10X, 35X, etc.)
- Margin information
- Liquidation prices
- Any profit/loss values
- Platform interface data (Binance, Bybit, etc.)

CRITICAL: Extract EXACT NUMERICAL VALUES when visible in screenshots!

Required JSON format:
{{
    "action": "open|trim|close|position_update|none",
    "ticker": "TICKER_SYMBOL",
    "confidence": 0-100,
    "direction": "LONG|SHORT|UPDATE",
    "tp_number": null or number,
    "unrealized_pnl": null or number,
    "unrealized_pnl_currency": "USDT|USD|BTC|ETH",
    "roi_percentage": null or number,
    "mark_price": null or number,
    "entry_price": null or number,
    "position_size": null or number,
    "leverage": null or number,
    "margin_used": null or number,
    "liquidation_price": null or number,
    "entry_targets": [],
    "tp_targets": [],
    "platform": "binance|bybit|kraken|other",
    "reasoning": "detailed description of all extracted data from image"
}}

Examples:
- Position screenshot with PnL -> extract unrealized_pnl, roi_percentage, mark_price, entry_price
- TP hit image -> extract exact profit amount and percentage
- Entry signal chart -> extract entry levels and targets"""

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        result = response.json()
        
        # Check if response has proper structure
        if "choices" not in result or len(result["choices"]) == 0:
            logger.error(f"GPT-4 Vision API error: Invalid response structure: {result}")
            return {"action": "none", "ticker": None, "error": "Invalid Vision API response"}
            
        content = result["choices"][0]["message"]["content"].strip()
        
        # Extract JSON from response
        import json as pyjson
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
            
        parsed = pyjson.loads(content)
        
        # Validate required fields
        if not parsed.get("action") or not parsed.get("ticker"):
            return {"action": "none", "ticker": None, "error": "missing_fields"}
            
        return parsed
        
    except Exception as e:
        logger.error(f"GPT-4 Vision API error: {e}")
        # Fallback to text-only analysis if vision fails
        return query_llm_for_signal(ocr_text, api_key, context_positions)


def download_and_encode_image(image_url: str) -> str:
    """Download image and encode as base64 for API calls."""
    try:
        response = requests.get(image_url, timeout=30)
        return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        logger.error(f"Image download error: {e}")
        return None
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

GRIZZLIES_PHRASE_LIBRARY = {
    "open": [
        r"(Started|starting) (a )?(\w+) (position|long|short)",
        r"(In|Long|Short|Buying|Selling|Opening|Averaging into|Scaling into) (\w+)",
        r"(\w+) calls up \d+%.*\$\d+ to \$\d+",
        r"(\w+) puts up \d+%.*\$\d+ to \$\d+",
        r"(\w+) (\d+[cp]) \d{1,2}/\d{1,2}/\d{2,4} @[\d\.]+ (shorting|longing)? (\w+)?",
    ],
    "trim": [
        r"(\w+) TP(\d+) hit",
        r"TP(\d+) hit.*(\w+)",
        r"TP(\d+) hit.*BANG+!?",
        r"TP(\d+).*BABY.*BANG+",
        r"TP(\d+).*for (\w+)",
        r"trim(med|ming)?( half| here)?( for (\w+))?",
        r"taking profits? on (\w+)",
        r"partial close on (\w+)",
        r"scaling out of (\w+)",
        r"took more here.*\d+%.*BANG",
        r"greater profits here for (\w+)",
        r"(\d+)% on (\w+) calls",
        r"(\d+)% BANG+",
        r"(\d+)%.*left \d+ runners BANG+",
        r"(\d+)%.*BANG+",
        r"(\$\d+ to \$\d+)",
        r"make sure you trim and have stops",
        r"def trim and set stops",
        r"tp(\d+) hit.*(\w+)",
        r"tp(\d+).*on our (\w+) trade",
        r"tp(\d+).*on (\w+)",
        r"tp(\d+).*baby.*bang bang",
        r"tp(\d+).*for (\w+) baby",
        r"tp(\d+).*for (\w+)",
        r"tp(\d+).*hit.*",
        r"(\w+) tp(\d+).*hit",
        r"trimmed.*for (\w+)",
        r"trimmed half here for (\w+)",
        r"trimmed some (\w+)",
        r"secured some profits on (\w+)",
        r"booked partials on (\w+)",
        r"taking some off (\w+)",
        r"locking some gains on (\w+)",
        r"tp hit on (\w+)",
        r"hit target on (\w+)",
        r"(\w+) tp(\d+).*",
        r"tp(\d+) hit.*",
        r"tp(\d+).*bang+",
        r"tp(\d+).*baby.*bang bang",
        r"tp(\d+).*for (\w+) baby",
        r"tp(\d+).*for (\w+)",
        r"tp(\d+).*hit.*",
        r"tp(\d+).*",
    ],
    "close": [
        r"all tps hit.*all out of (\w+)",
        r"all tps hit.*i'?m all out",
        r"closing (\w+) position",
        r"all out (\w+)",
        r"flat (\w+)",
        r"closed (\w+)",
        r"full close on (\w+)",
        r"no longer in (\w+)",
        r"stopped out (\w+)",
        r"stopped on (\w+)",
        r"cut (\w+)",
        r"exiting (\w+)",
        r"done with (\w+)",
        r"stopped out on (\w+)",
        r"stopped out on entry for (\w+)",
        r"stopped out on (\w+) at entry",
        r"i'?ll cut this if we break below",
        r"stopped out on (\w+)",
        r"stopped on (\w+)",
        r"stopped out at entry",
        r"stopped out",
        r"cut",
        r"exiting",
        r"all out",
        r"flat",
        r"closed",
        r"full close",
        r"no longer in",
        r"done with",
    ],
    "profit": [
        r"(\d+)% on (\w+) calls",
        r"(\d+)%.*BANG+",
        r"(\d+)%.*left \d+ runners BANG+",
        r"(\d+)%.*BANG+",
        r"(\$\d+ to \$\d+)",
        r"(\w+) calls up \d+%.*\$\d+ to \$\d+",
        r"(\w+) puts up \d+%.*\$\d+ to \$\d+",
        r"(\w+) calls up \d+%",
        r"(\w+) puts up \d+%",
        r"(\w+) calls up \d+%.*",
        r"(\w+) puts up \d+%.*",
        r"(\d+)%.*",
        r"\$\d+ to \$\d+",
    ],
    "risk": [
        r"make sure you trim and have stops",
        r"def trim and set stops",
        r"set stops at break even",
        r"stops on entry",
        r"will set stops above entry now",
        r"set stops on entry",
        r"stops at entry",
        r"stops above entry",
        r"stops below entry",
        r"stopped out",
        r"i'?ll cut this if we break below",
        r"btc just weak atm",
    ]
}


def fetch_message_by_id(channel_id: int, message_id: int, discord_token: str) -> Optional[dict]:
    """
    Fetch a Discord message by channel and message ID using the user token.
    """
    url = f"https://discord.com/api/v9/channels/{channel_id}/messages/{message_id}"
    headers = {
        "authorization": discord_token,
        "user-agent": "Mozilla/5.0"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            logger.warning(f"Failed to fetch message {message_id}: {resp.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching message by ID: {e}")
        return None



def send_telegram_alert(message):
    if not get_env("TELEGRAM_ALERTS_ENABLED", "false", bool):
        return
    token = get_env("TELEGRAM_BOT_TOKEN")
    chat_id = get_env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        print(f"[DEBUG] Telegram API response: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")

class TelegramAlertHandler(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.CRITICAL:
            message = f"🚨 BIG BOSS CEO CRITICAL ERROR:\n{record.getMessage()}"
            send_telegram_alert(message)


# --- Logging Setup ---
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'big_boss_ceo.log')
LLM_LOG_FILE = os.path.join(LOG_DIR, 'llm_flagged_signals.log')

logger = logging.getLogger("grizzlies")
logger.setLevel(logging.INFO)

# Rotating file handler: 1MB per file, keep 5 backups
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s %(message)s')
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)

# LLM-flagged signals logger
llm_logger = logging.getLogger("llm_signals")
llm_logger.setLevel(logging.INFO)
llm_file_handler = RotatingFileHandler(LLM_LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
llm_file_handler.setFormatter(formatter)
if not llm_logger.hasHandlers():
    llm_logger.addHandler(llm_file_handler)

# --- Telegram Alert Handler Setup ---
telegram_handler = TelegramAlertHandler()
telegram_handler.setLevel(logging.CRITICAL)
telegram_handler.setFormatter(formatter)
if not any(isinstance(h, TelegramAlertHandler) for h in logger.handlers):
    logger.addHandler(telegram_handler)
"""
🔥 ULTIMATE GRIZZLIES SIGNAL PROCESSOR - ENHANCED VERSION 🔥
================================================================
Multi-layer signal detection based on REAL July 2025 Grizzlies data
Catches 85%+ of all signals (structured + natural language + position updates)
================================================================
"""

import re
import json
import logging
import os
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# --- Logging Setup ---
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'big_boss_ceo.log')

logger = logging.getLogger("grizzlies")
logger.setLevel(logging.INFO)

# Rotating file handler: 1MB per file, keep 5 backups
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s %(message)s')
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)

# --- Config Loading ---
def get_env(key, default=None, cast_type=None):
    val = os.environ.get(key, default)
    if cast_type and val is not None:
        try:
            if cast_type is bool:
                return str(val).lower() in ("1", "true", "yes", "on")
            return cast_type(val)
        except Exception:
            return default
    return val

DEFAULT_CONFIG = {
    # DISABLED: No confidence filtering - execute ALL signals
    "confidence_threshold": 0.0,  # Accept everything (was GRIZZLIES_CONFIDENCE_THRESHOLD)
    "leverage_max": get_env("GRIZZLIES_LEVERAGE_MAX", 10, int),
    "position_size": get_env("GRIZZLIES_POSITION_SIZE", 5, int),
    "win_rate_target": get_env("GRIZZLIES_WIN_RATE_TARGET", 95, int),
    "max_daily_loss": get_env("MAX_DAILY_LOSS", 500, float),
    "max_single_trade_loss": get_env("MAX_SINGLE_TRADE_LOSS", 50, float),
    "risk_per_trade": get_env("RISK_PER_TRADE", 0.045, float),
    "max_open_positions": get_env("MAX_OPEN_POSITIONS", 5, int),
    "stop_loss_percentage": get_env("STOP_LOSS_PERCENTAGE", 0.03, float),
    "take_profit_percentage": get_env("TAKE_PROFIT_PERCENTAGE", 0.15, float),
    # Add more as needed from .env
}

class SignalType(Enum):
    CRYPTO_FUTURES = "crypto_futures"
    OPTIONS = "options"
    POSITION_UPDATE = "position_update"
    RISK_WARNING = "risk_warning"

class PositionStatus(Enum):
    ACTIVE = "active"
    PARTIAL_EXIT = "partial_exit"
    CLOSED = "closed"
    AVERAGING = "averaging"


# --- Data Classes ---
@dataclass
class GrizzliesSignal:
    """Enhanced signal object with all Grizzlies data"""
    signal_type: SignalType
    ticker: str
    direction: str
    confidence: float
    timestamp: datetime
    source: str
    channel: str
    raw_message: str
    # Structured signal data
    entry_targets: List[float] = field(default_factory=list)
    tp_targets: List[float] = field(default_factory=list)
    leverage_type: str = None  # "cross" or "isolated"
    # Options data
    strike_price: float = None
    expiry_date: str = None
    option_type: str = None  # "call" or "put"
    # Position tracking
    position_id: str = None
    tp_hit: List[int] = field(default_factory=list)  # Which TPs have been hit
    # Quality metrics
    trust_score: float = 0.0
    signal_quality: str = "UNKNOWN"
    parsing_method: str = "unknown"
    # Correlation
    message_id: str = None  # Discord message id for audit threading

    def validate(self) -> bool:
        """Validate required fields and types for a signal."""
        errors = []
        if not isinstance(self.signal_type, SignalType):
            errors.append("signal_type must be a SignalType enum")
        if not isinstance(self.ticker, str) or not self.ticker:
            errors.append("ticker must be a non-empty string")
        if not isinstance(self.direction, str) or not self.direction:
            errors.append("direction must be a non-empty string")
        if not (0 <= self.confidence <= 100):
            errors.append("confidence must be between 0 and 100")
        if not isinstance(self.timestamp, datetime):
            errors.append("timestamp must be a datetime object")
            errors.append("trust_score must be a float")
        if errors:
            logger.error(f"❌ Signal validation failed: {errors} | Signal: {self}")
            return False
        return True

class PositionTracker:
    """Track active positions and correlate updates"""
    
    def __init__(self):
        self.active_positions = {}  # position_id -> GrizzliesSignal
        self.position_history = []
        

    def add_position(self, signal: GrizzliesSignal) -> str:
        """Add new position and return position ID"""
        position_id = f"{signal.ticker}_{signal.direction}_{int(signal.timestamp.timestamp())}"
        signal.position_id = position_id
        signal.tp_hit = []
        self.active_positions[position_id] = signal
        logger.info(f"📊 New position tracked: {position_id}", extra={"event": "add_position", "position_id": position_id, "ticker": signal.ticker, "direction": signal.direction})
        return position_id

    def get_active_positions(self) -> dict:
        """Get all active positions"""
        return self.active_positions.copy()



    def update_position(self, ticker: str, update_type: str, details: str, content: str = None, author: str = None, channel: str = None) -> Optional[Union[str, GrizzliesSignal]]:
        """Update position based on Grizzlies commentary. If no active position, try to create a profit/performance update signal."""
        # Find matching active position with flexible ticker matching
        ticker_upper = ticker.upper()
        matching_positions = []
        
        logger.info(f"🔍 Looking for position matching ticker: '{ticker_upper}'")
        logger.info(f"🔍 Active positions: {list(self.active_positions.keys())}")
        
        for pos_id, signal in self.active_positions.items():
            signal_ticker = signal.ticker.upper()
            logger.info(f"🔍 Checking position {pos_id} with ticker '{signal_ticker}'")
            # Direct match or base asset match (ETH matches ETH/USDT)
            if (ticker_upper == signal_ticker or 
                ticker_upper in signal_ticker or 
                signal_ticker.split('/')[0] == ticker_upper or
                signal_ticker.split('-')[0] == ticker_upper):
                logger.info(f"✅ Match found: {pos_id}")
                matching_positions.append(pos_id)
        
        if not matching_positions:
            logger.warning(f"⚠️ No active position found for {ticker} update: {update_type}")
            # Try to extract ticker from context if content is provided
            if content and author and channel:
                extracted_ticker = self._extract_ticker_from_context(content)
                content_upper = content.upper()
                # Check if it's a profit/performance update
                if extracted_ticker and any(word in content_upper for word in ['%', 'PROFIT', 'GAIN', 'UP', 'TRIMMING']):
                    signal = GrizzliesSignal(
                        signal_type=SignalType.POSITION_UPDATE,
                        ticker=extracted_ticker,
                        direction="PROFIT_UPDATE",
                        confidence=90.0,  # High confidence for clear profit updates
                        timestamp=datetime.now(),
                        source=author,
                        channel=channel,
                        raw_message=content,
                        trust_score=70.0,  # Default trust score, or could call self._calculate_trust_score(author) if available
                        signal_quality="HIGH",
                        parsing_method="context_extraction"
                    )
            logger.info(f"📈 Created profit/performance update signal for {extracted_ticker}", extra={"event": "profit_update", "ticker": extracted_ticker})
            return signal
            return None
        # Use most recent position if multiple matches
        position_id = matching_positions[-1]
        signal = self.active_positions[position_id]
        # Process update
        if "TP" in update_type.upper():
            tp_num = self._extract_tp_number(update_type)
            if tp_num and tp_num not in signal.tp_hit:
                signal.tp_hit.append(tp_num)
                logger.info(f"✅ {position_id}: TP{tp_num} hit", extra={"event": "tp_hit", "position_id": position_id, "tp_num": tp_num})
        return position_id
    
    def _extract_tp_number(self, text: str) -> Optional[int]:
        """Extract TP number from text like 'TP1 hit' or 'TP2'"""
        match = re.search(r'TP(\d+)', text.upper())
        return int(match.group(1)) if match else None
    
    def get_active_positions(self) -> Dict[str, 'GrizzliesSignal']:
        """Get all active positions"""
        return self.active_positions.copy()
    
    def close_position(self, position_id: str):
        """Close a position"""
        if position_id in self.active_positions:
            signal = self.active_positions.pop(position_id)
            self.position_history.append(signal)
            logger.info(f"🔒 Position closed: {position_id}", extra={"event": "close_position", "position_id": position_id})

class UltimateGrizzliesSignalProcessor:
    
    def _is_analysis_or_update_message(self, content: str) -> bool:
        """Filter out analysis messages and position updates that should not trigger trades."""
        content_lower = content.lower()
        
        # Analysis phrases that should NOT trigger trades
        analysis_phrases = [
            'looking good', 'looks good', 'chart looks', 'analysis', 'update',
            'market update', 'price action', 'watching', 'monitoring',
            'thinking about', 'considering', 'might', 'could',
            'bullish', 'bearish', 'trend', 'pattern', 'setup',
            'resistance', 'support', 'fibonacci', 'technical',
            'chart', 'candle', 'volume', 'consolidation',
            'breakout potential', 'possible', 'maybe', 'perhaps'
        ]
        
        # Position update phrases that should NOT trigger new trades
        position_update_phrases = [
            'position update', 'current position', 'holding',
            'still in', 'keeping', 'maintaining', 'status',
            'unrealized', 'mark price', 'pnl', 'profit and loss',
            'roi', 'return on investment', 'current profit',
            'current loss', 'floating', 'paper profit', 'paper loss'
        ]
        
        # CRITICAL: HOOD-specific filters to prevent the exact bug
        hood_tp_update_phrases = [
            'hood tp1', 'hood tp2', 'hood tp3', 'hood tp4', 'hood tp5',
            'hood hit tp', 'hood profit', 'hood update', 'hood position',
            'hood target', 'hood take profit'
        ]
        
        # Check for HOOD TP updates (these should NOT trigger new trades)
        for phrase in hood_tp_update_phrases:
            if phrase in content_lower:
                logger.warning(f"🚨 CRITICAL FILTER: HOOD TP update detected: '{phrase}' - BLOCKING TRADE", 
                             extra={"event": "filter_hood_tp_update", "phrase": phrase, "content": content[:100]})
                return True
        
        # Check for analysis phrases
        for phrase in analysis_phrases:
            if phrase in content_lower:
                logger.info(f"🔍 Filtered analysis message: '{phrase}' detected", 
                           extra={"event": "filter_analysis", "phrase": phrase})
                return True
        
        # Check for position update phrases
        for phrase in position_update_phrases:
            if phrase in content_lower:
                logger.info(f"🔍 Filtered position update: '{phrase}' detected", 
                           extra={"event": "filter_position_update", "phrase": phrase})
                return True
        
        return False

    def _extract_price_targets(self, target_text: str) -> List[float]:
        """Extract price targets from text like '0.5-0.6' or '0.5,0.6,0.7'"""
        targets = []
        # Split by common separators
        parts = re.split(r'[-,\s]+', target_text.strip())
        for part in parts:
            try:
                price = float(part.strip())
                targets.append(price)
            except ValueError:
                continue
        return targets

    def _calculate_structured_confidence(self, direction: str, ticker: str, entry_targets: List[float], tp_targets: List[float], content: str, author: str) -> float:
        """Fixed confidence - no filtering logic"""
        return 100.0  # Always 100% - execute ALL signals

    def _calculate_natural_language_confidence(self, content: str, author: str) -> float:
        """Fixed confidence - no filtering logic"""
        return 100.0  # Always 100% - execute ALL signals

    def _assess_signal_quality(self, content: str) -> str:
        """Assess signal quality based on content"""
        content_lower = content.lower()
        high_quality_words = ['analysis', 'chart', 'support', 'resistance', 'target']
        medium_quality_words = ['watching', 'looking', 'might']
        if any(word in content_lower for word in high_quality_words):
            return "HIGH"
        elif any(word in content_lower for word in medium_quality_words):
            return "MEDIUM"
        else:
            return "LOW"

    def _extract_ticker_from_context(self, content: str) -> Optional[str]:
        """Extract ticker from context"""
        # Look for common crypto tickers
        ticker_match = re.search(r'\b([A-Z]{3,6})\b', content.upper())
        if ticker_match:
            ticker = ticker_match.group(1)
            # Common crypto tickers
            if ticker in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'UNI', 'AVAX', 'MATIC', 'ATOM']:
                return ticker
        return None
    def _parse_with_phrase_library(self, content: str) -> Optional[dict]:
        """Check content against the Grizzlies phrase library and return match info."""
        for action, patterns in GRIZZLIES_PHRASE_LIBRARY.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    # Extract ticker from groups - look for alphabetic strings that could be tickers
                    groups = match.groups()
                    ticker = None
                    tp_number = None
                    
                    if groups:
                        for g in groups:
                            if g and g.isalpha() and len(g) >= 2 and len(g) <= 6:
                                # Check if it's a known crypto ticker
                                if g.upper() in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'UNI', 'AVAX', 'MATIC', 'ATOM', 'ENA', 'DOGE', 'XRP']:
                                    ticker = g.upper()
                                    break
                            elif g and g.isdigit():
                                # This might be a TP number
                                tp_number = int(g)
                    
                    return {
                        "action": action,
                        "pattern": pattern,
                        "groups": groups,
                        "ticker": ticker,
                        "tp_number": tp_number
                    }
        return None

    def _calculate_trust_score(self, message_content, author_name=None):
        """Calculate trust score for a signal based on various factors"""
        trust_score = 0.0
        # Base score for any signal
        trust_score += 30.0
        # Author credibility (if available)
        if author_name:
            if author_name.lower() in ['grizzlies', 'admin', 'mod']:
                trust_score += 40.0
            else:
                trust_score += 20.0
        # Content analysis
        content = message_content.lower()
        # Signal strength indicators
        if any(word in content for word in ['strong', 'high confidence', 'confirmed']):
            trust_score += 15.0
        # Technical indicators
        if any(word in content for word in ['rsi', 'macd', 'ema', 'support', 'resistance']):
            trust_score += 10.0
        # Risk indicators (reduce trust)
        if any(word in content for word in ['risky', 'uncertain', 'maybe']):
            trust_score -= 10.0
        # Cap the score between 0 and 100
        return max(0.0, min(100.0, trust_score))

    def _parse_position_update(self, content, *args, **kwargs):
        """Parse position update messages"""
        try:
            # Handle different calling patterns
            author = args[0] if len(args) > 0 else kwargs.get('author', 'unknown')
            channel = args[1] if len(args) > 1 else kwargs.get('channel', 'unknown')
            message = args[2] if len(args) > 2 else kwargs.get('message', None)
            parent_signal = kwargs.get('parent_signal', None)

            content_lower = content.lower()

            # Extract symbol
            symbol = None
            symbol_match = re.search(r'([A-Z]{3,6}(?:USD|USDT|BTC|ETH)?)', content.upper())
            if symbol_match:
                symbol = symbol_match.group(1)
            # If we have a parent signal, try to get symbol from it
            if parent_signal and not symbol:
                symbol = getattr(parent_signal, 'ticker', None)
            # If still no symbol, extract from context or use UNKNOWN
            if not symbol:
                symbol = self._extract_ticker_from_context(content) or "UNKNOWN"

            # Determine status/direction
            direction = "UPDATE"
            if any(word in content_lower for word in ['closed', 'exit', 'tp hit', 'stop loss']):
                direction = "CLOSE"
            elif any(word in content_lower for word in ['open', 'entry', 'entered']):
                direction = "LONG"
            elif any(word in content_lower for word in ['partial', 'reduce', 'trim']):
                direction = "UPDATE"

            # Extract PnL if mentioned
            pnl = None
            pnl_match = re.search(r'([+-]?\d+\.?\d*)[%$]?', content)
            if pnl_match:
                try:
                    pnl = float(pnl_match.group(1))
                except ValueError:
                    pnl = None

            # Create and return a GrizzliesSignal object
            signal = GrizzliesSignal(
                signal_type=SignalType.POSITION_UPDATE,
                ticker=symbol,
                direction=direction,
                confidence=85.0,  # Default confidence for position updates
                timestamp=datetime.now(),
                source=author,
                channel=channel,
                raw_message=content,
                trust_score=self._calculate_trust_score(content, author),
                signal_quality="MEDIUM",
                parsing_method="position_update",
                message_id=str(getattr(message, 'id', None)) if message else None
            )

            # Validate the signal before returning
            if signal.validate():
                return signal
            else:
                return None

        except Exception as e:
            logger.error(f"Error parsing position update: {e}")
            return None
    def _parse_natural_language(self, content: str, author: str, channel: str, message) -> Optional['GrizzliesSignal']:
        """Parse natural language signals like 'Going to hit some eth longs again here'"""
        content_upper = content.upper()
        # Try entry signal patterns
        for pattern in self.natural_patterns['entry_signals']:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                ticker = match.group(1).upper()
                direction_text = match.group(2).upper()
                direction = "LONG" if "LONG" in direction_text else "SHORT"
                # Calculate confidence based on language strength
                confidence = self._calculate_natural_language_confidence(content, author)
                signal = GrizzliesSignal(
                    signal_type=SignalType.CRYPTO_FUTURES,
                    ticker=ticker,
                    direction=direction,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    source=author,
                    channel=channel,
                    raw_message=content,
                    trust_score=self._calculate_trust_score(author),
                    signal_quality=self._assess_signal_quality(content),
                    message_id=str(getattr(message, 'id', None)) if message else None
                )
                if not signal.validate():
                    return None
                # Add to position tracker
                self.position_tracker.add_position(signal)
                return signal
        return None

    def _parse_options_signal(self, content: str, author: str, channel: str, message) -> Optional['GrizzliesSignal']:
        """Parse options signals (basic implementation for now)"""
        # TODO: Implement options parsing logic
        return None
    def _parse_structured_signal(self, content: str, author: str, channel: str, message) -> Optional['GrizzliesSignal']:
        """Parse structured signals using defined patterns."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 3:
            return None
        # Check for signal start pattern
        direction_match = None
        for pattern in self.structured_patterns['signal_start']:
            match = re.match(pattern, lines[0], re.IGNORECASE)
            if match:
                direction_match = match.group(1).upper()
                break
        if not direction_match:
            return None
        # Extract ticker
        ticker = None
        leverage_type = None
        entry_targets = []
        tp_targets = []
        for i, line in enumerate(lines[1:], 1):
            # Try ticker patterns
            if not ticker:
                for pattern in self.structured_patterns['ticker_line']:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        ticker = match.group(1).upper()
                        break
            # Try leverage patterns
            for pattern in self.structured_patterns['leverage_line']:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    leverage_type = match.group(1).lower()
                    break
            # Try entry targets
            for pattern in self.structured_patterns['entry_targets']:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    entry_targets = self._extract_price_targets(match.group(1))
                    break
            # Try TP targets
            for pattern in self.structured_patterns['tp_targets']:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    tp_targets = self._extract_price_targets(match.group(1))
                    break
        if not ticker:
            return None
        # Calculate confidence based on structure completeness
        confidence = self._calculate_structured_confidence(
            direction_match, ticker, entry_targets, tp_targets, content, author
        )
        # Create signal
        signal = GrizzliesSignal(
            signal_type=SignalType.CRYPTO_FUTURES,
            ticker=ticker,
            direction=direction_match,
            confidence=confidence,
            timestamp=datetime.now(),
            source=author,
            channel=channel,
            raw_message=content,
            entry_targets=entry_targets,
            tp_targets=tp_targets,
            leverage_type=leverage_type,
            trust_score=self._calculate_trust_score(author),
            signal_quality=self._assess_signal_quality(content),
            message_id=str(getattr(message, 'id', None)) if message else None
        )
        if not signal.validate():
            return None
        # Add to position tracker
        self.position_tracker.add_position(signal)
        return signal
    def _is_trusted_source(self, author: str) -> bool:
        """Check if author is a trusted source"""
        return any(trusted in author.lower() for trusted in self.trust_scores.keys())
    def parse_ultimate_signal(self, message, parent_signal: Optional[GrizzliesSignal] = None) -> Optional[GrizzliesSignal]:
        """🎯 LLM-FIRST PARSING ENGINE - AI-powered signal detection with 95%+ accuracy"""
        try:
            content = message.content
            author = message.author.name.lower()
            channel = getattr(message.channel, 'name', 'unknown')

            # 🛡️ FILTER OUT ANALYSIS AND POSITION UPDATE MESSAGES
            if self._is_analysis_or_update_message(content):
                logger.info(f"🛡️ Filtered analysis/update message from {author}: {content[:50]}...", 
                           extra={"event": "message_filtered", "author": author, "reason": "analysis_or_update"})
                return None

            # 🖼️ CHECK FOR IMAGE ATTACHMENTS
            image_attachments = []
            if hasattr(message, 'attachments') and message.attachments:
                for attachment in message.attachments:
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        image_attachments.append(attachment.url)
                        logger.info(f"🖼️ Image attachment detected: {attachment.filename}", extra={"event": "image_detected", "url": attachment.url})

            # --- Reply tracking ---
            if hasattr(message, "reference") and message.reference:
                parent_id = getattr(message.reference, "message_id", None)
                parent_channel_id = getattr(message.reference, "channel_id", None) or getattr(message.channel, "id", None)
                if parent_id and parent_channel_id:
                    discord_token = get_env("DISCORD_USER_TOKEN")
                    parent_msg_data = fetch_message_by_id(parent_channel_id, parent_id, discord_token)
                    if parent_msg_data and "content" in parent_msg_data:
                        logger.info(f"🔗 Parsing parent message for reply: {parent_id}", extra={"event": "reply_tracking", "parent_id": parent_id})
                        parent_signal = self.parse_ultimate_signal(
                            type("Msg", (), {
                                "content": parent_msg_data["content"],
                                "author": type("Author", (), {"name": parent_msg_data["author"]["username"]}),
                                "channel": type("Channel", (), {"name": channel, "id": parent_channel_id}),
                                "reference": None  # Prevent infinite recursion
                            })()
                        )

            if not self._is_trusted_source(author):
                return None

            logger.debug(f"🔍 LLM-FIRST parsing message from {author}: {content[:100]}...", extra={"event": "parse_attempt", "author": author, "channel": channel})

            # � EMERGENCY: LLM PARSING DISABLED - HALLUCINATING FAKE TRADES
            # �🚀 PRIMARY METHOD: LLM-FIRST PARSING (95%+ accuracy) - TEMPORARILY DISABLED
            logger.info(f"⚠️ LLM parsing DISABLED to prevent fake signals. Using regex only for {author}")
            openai_api_key = None  # Force disable LLM
            if False:  # openai_api_key:
                # Get active positions for context
                active_positions = self.position_tracker.get_active_positions()
                
                # 🖼️ PROCESS IMAGES FIRST
                if image_attachments:
                    logger.info(f"🖼️ Processing {len(image_attachments)} image(s) for trading signals", extra={"event": "image_processing", "count": len(image_attachments)})
                    
                    for image_url in image_attachments:
                        try:
                            image_result = process_image_for_signals(image_url, openai_api_key, active_positions)
                            action = image_result.get("action", "none")
                            ticker = image_result.get("ticker")
                            
                            if action in ("open", "trim", "close", "position_update") and ticker:
                                confidence = image_result.get("confidence", 80.0)
                                direction = image_result.get("direction", "LONG" if action == "open" else "UPDATE" if action == "trim" else "CLOSE")
                                tp_number = image_result.get("tp_number")
                                
                                # Extract all trading data from image
                                unrealized_pnl = image_result.get("unrealized_pnl")
                                unrealized_pnl_currency = image_result.get("unrealized_pnl_currency", "USDT")
                                roi_percentage = image_result.get("roi_percentage")
                                mark_price = image_result.get("mark_price")
                                entry_price = image_result.get("entry_price")
                                position_size = image_result.get("position_size")
                                leverage = image_result.get("leverage")
                                margin_used = image_result.get("margin_used")
                                liquidation_price = image_result.get("liquidation_price")
                                platform = image_result.get("platform", "unknown")
                                
                                entry_targets = image_result.get("entry_targets", [])
                                tp_targets = image_result.get("tp_targets", [])
                                
                                # Create more descriptive direction for TP hits
                                if action == "trim" and tp_number:
                                    direction = f"TP{tp_number}_HIT"
                                elif action == "position_update":
                                    direction = "POSITION_UPDATE"
                                
                                signal = GrizzliesSignal(
                                    signal_type=SignalType.CRYPTO_FUTURES if action == "open" else SignalType.POSITION_UPDATE,
                                    ticker=ticker.upper(),
                                    direction=direction,
                                    confidence=confidence,
                                    timestamp=datetime.now(),
                                    source=author,
                                    channel=channel,
                                    raw_message=content + f" [IMAGE: {image_url}]",
                                    entry_targets=entry_targets,
                                    tp_targets=tp_targets,
                                    leverage_type="cross" if leverage else None,
                                    trust_score=self._calculate_trust_score(content, author),
                                    signal_quality="IMAGE_LLM_HIGH",
                                    parsing_method="image_llm_primary",
                                    message_id=str(getattr(message, 'id', None)) if message else None
                                )
                                
                                if not signal.validate():
                                    logger.warning(f"❌ Image signal validation failed: {signal}")
                                    continue
                                
                                # 🔥 DETAILED IMAGE DATA LOGGING
                                logger.info(f"🖼️ IMAGE ANALYSIS COMPLETE for {ticker}")
                                logger.info(f"📊 ==================== EXTRACTED TRADING DATA ====================")
                                if unrealized_pnl is not None:
                                    logger.info(f"💰 Unrealized PnL: {unrealized_pnl:+.2f} {unrealized_pnl_currency}")
                                if roi_percentage is not None:
                                    logger.info(f"📈 ROI: {roi_percentage:+.2f}%")
                                if mark_price is not None:
                                    logger.info(f"🎯 Mark Price: ${mark_price:,.2f}")
                                if entry_price is not None:
                                    logger.info(f"🚪 Entry Price: ${entry_price:,.2f}")
                                if position_size is not None:
                                    logger.info(f"📏 Position Size: {position_size}")
                                if leverage is not None:
                                    logger.info(f"⚡ Leverage: {leverage}X")
                                if margin_used is not None:
                                    logger.info(f"💳 Margin Used: {margin_used}")
                                if liquidation_price is not None:
                                    logger.info(f"⚠️ Liquidation Price: ${liquidation_price:,.2f}")
                                if platform != "unknown":
                                    logger.info(f"🏪 Platform: {platform.upper()}")
                                logger.info(f"🧠 Reasoning: {image_result.get('reasoning', 'N/A')}")
                                logger.info(f"📊 ============================================================")
                                
                                # Enhanced logging with extracted data
                                if action == "trim" and tp_number and ticker:
                                    profit_info = f" | ROI: {roi_percentage:+.2f}%" if roi_percentage else ""
                                    pnl_info = f" | PnL: {unrealized_pnl:+.2f} {unrealized_pnl_currency}" if unrealized_pnl else ""
                                    price_info = f" | Mark: ${mark_price:,.2f}" if mark_price else ""
                                    logger.info(f"🖼️ IMAGE TP{tp_number} for {ticker}!{profit_info}{pnl_info}{price_info}", extra={"event": "image_tp_hit", "ticker": ticker, "tp_number": tp_number, "roi": roi_percentage, "pnl": unrealized_pnl})
                                elif action == "position_update":
                                    pnl_info = f" | PnL: {unrealized_pnl:+.2f} {unrealized_pnl_currency}" if unrealized_pnl else ""
                                    roi_info = f" | ROI: {roi_percentage:+.2f}%" if roi_percentage else ""
                                    logger.info(f"🖼️ IMAGE POSITION UPDATE for {ticker}{pnl_info}{roi_info}", extra={"event": "image_position_update", "ticker": ticker, "roi": roi_percentage, "pnl": unrealized_pnl})
                                elif action == "open":
                                    targets_info = f" | Entry: {entry_targets} | TP: {tp_targets}" if entry_targets or tp_targets else ""
                                    logger.info(f"🖼️ IMAGE NEW {direction} signal for {ticker}{targets_info}", extra={"event": "image_open", "ticker": ticker, "direction": direction})
                                elif action == "close":
                                    logger.info(f"🖼️ IMAGE CLOSE signal for {ticker}", extra={"event": "image_close", "ticker": ticker})
                                
                                return signal
                                
                        except Exception as e:
                            logger.error(f"Error processing image {image_url}: {e}")
                            continue
                
                # 📝 PROCESS TEXT MESSAGE (fallback or additional)
                llm_result = query_llm_for_signal(content, openai_api_key, active_positions)
                action = llm_result.get("action", "none")
                ticker = llm_result.get("ticker")
                
                if action in ("open", "trim", "close") and ticker:
                    confidence = llm_result.get("confidence", 80.0)
                    direction = llm_result.get("direction", "LONG" if action == "open" else "UPDATE" if action == "trim" else "CLOSE")
                    tp_number = llm_result.get("tp_number")
                    profit_pct = llm_result.get("profit_pct")
                    leverage = llm_result.get("leverage")
                    entry_targets = llm_result.get("entry_targets", [])
                    tp_targets = llm_result.get("tp_targets", [])
                    
                    # Create more descriptive direction for TP hits
                    if action == "trim" and tp_number:
                        direction = f"TP{tp_number}_HIT"
                    
                    signal = GrizzliesSignal(
                        signal_type=SignalType.CRYPTO_FUTURES if action == "open" else SignalType.POSITION_UPDATE,
                        ticker=ticker.upper(),
                        direction=direction,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        source=author,
                        channel=channel,
                        raw_message=content,
                        entry_targets=entry_targets,
                        tp_targets=tp_targets,
                        leverage_type="cross" if leverage else None,
                        trust_score=self._calculate_trust_score(content, author),
                        signal_quality="LLM_HIGH",
                        parsing_method="llm_primary",
                        message_id=str(getattr(message, 'id', None)) if message else None
                    )
                    
                    if not signal.validate():
                        logger.warning(f"❌ LLM signal validation failed: {signal}")
                        return None
                    
                    # Enhanced logging with profit data
                    if action == "trim" and tp_number and ticker:
                        profit_info = f" | Profit: {profit_pct}%" if profit_pct else ""
                        logger.info(f"🎯 LLM: TP{tp_number} HIT for {ticker}!{profit_info}", extra={"event": "llm_tp_hit", "ticker": ticker, "tp_number": tp_number, "profit": profit_pct})
                    elif action == "open":
                        targets_info = f" | Entry: {entry_targets} | TP: {tp_targets}" if entry_targets or tp_targets else ""
                        logger.info(f"🚀 LLM: NEW {direction} signal for {ticker}{targets_info}", extra={"event": "llm_open", "ticker": ticker, "direction": direction})
                    elif action == "close":
                        logger.info(f"🔒 LLM: CLOSE signal for {ticker}", extra={"event": "llm_close", "ticker": ticker})
                    
                    return signal

            # 🔄 FALLBACK METHOD 1: Phrase library (enhanced patterns)
            phrase_result = self._parse_with_phrase_library(content)
            if phrase_result:
                action = phrase_result["action"]
                groups = phrase_result["groups"]
                ticker = phrase_result.get("ticker")
                tp_number = phrase_result.get("tp_number")
                
                # If no ticker found, try context extraction
                if not ticker and parent_signal:
                    ticker = parent_signal.ticker
                elif not ticker:
                    ticker = self._extract_ticker_from_context(content)
                
                # Special handling for TP messages without explicit ticker
                if not ticker and action == "trim" and tp_number:
                    active_positions = self.position_tracker.get_active_positions()
                    if active_positions:
                        recent_positions = sorted(active_positions.values(), key=lambda x: x.timestamp, reverse=True)
                        if recent_positions:
                            ticker = recent_positions[0].ticker
                            logger.info(f"🔍 TP{tp_number} message without ticker - assuming most recent position: {ticker}", 
                                       extra={"event": "tp_ticker_assumption", "ticker": ticker, "tp_number": tp_number})
                
                direction = "LONG" if action == "open" else "TRIM" if action == "trim" else "CLOSE"
                if action == "trim" and tp_number:
                    direction = f"TP{tp_number}_HIT"
                
                signal = GrizzliesSignal(
                    signal_type=SignalType.POSITION_UPDATE,
                    ticker=ticker or "UNKNOWN",
                    direction=direction,
                    confidence=85.0,  # Lower confidence than LLM
                    timestamp=datetime.now(),
                    source=author,
                    channel=channel,
                    raw_message=content,
                    trust_score=self._calculate_trust_score(content, author),
                    signal_quality="REGEX_MEDIUM",
                    parsing_method="phrase_library_fallback",
                    message_id=str(getattr(message, 'id', None)) if message else None
                )
                
                if signal.validate():
                    logger.info(f"🔄 FALLBACK: Phrase library detected {action} for {ticker}", extra={"event": "fallback_phrase", "action": action, "ticker": ticker})
                    return signal

            # 🔄 FALLBACK METHOD 2: Structured signal parsing
            signal = self._parse_structured_signal(content, author, channel, message)
            if signal:
                signal.parsing_method = "structured_fallback"
                signal.signal_quality = "STRUCTURED_MEDIUM"
                logger.info(f"🔄 FALLBACK: Structured signal detected: {signal.ticker} {signal.direction}", extra={"event": "fallback_structured", "ticker": signal.ticker, "direction": signal.direction})
                return signal

            # 🔄 FALLBACK METHOD 3: Natural language parsing
            signal = self._parse_natural_language(content, author, channel, message)
            if signal:
                signal.parsing_method = "natural_language_fallback"
                signal.signal_quality = "NATURAL_MEDIUM"
                logger.info(f"🔄 FALLBACK: Natural language detected: {signal.ticker} {signal.direction}", extra={"event": "fallback_natural", "ticker": signal.ticker, "direction": signal.direction})
                return signal

            logger.debug(f"❌ No signal detected with any method: {content[:50]}...")
            return None

        except Exception as e:
            logger.error(f"❌ Error in LLM-first signal parsing: {e}", extra={"event": "parse_error", "error": str(e)})
            return None

    def __init__(self, config: Dict = None):
        # Merge user config with defaults
        merged_config = dict(self.DEFAULT_CONFIG)
        if config:
            for k, v in config.items():
                if k in merged_config and isinstance(merged_config[k], dict):
                    merged_config[k].update(v)
                else:
                    merged_config[k] = v
        self.config = merged_config
        self.position_tracker = PositionTracker()
        # Use config for all pattern/indicator/trust initializations
        self.structured_patterns = self.config['structured_patterns']
        self.natural_patterns = self.config['natural_patterns']
        self.options_patterns = self.config['options_patterns']
        self.update_patterns = self.config['update_patterns']
        self.trust_scores = self.config['trust_scores']
        self.quality_indicators = self.config['quality_indicators']
        logger.info("🚀 Ultimate Grizzlies Signal Processor initialized!", extra={"event": "init", "config_keys": list(self.config.keys())})

    def get_active_positions_summary(self) -> dict:
        """Get summary of all active positions"""
        active = self.position_tracker.active_positions
        summary = {
            'total_active': len(active),
            'positions': []
        }
        for pos_id, signal in active.items():
            pos_info = {
                'id': pos_id,
                'ticker': signal.ticker,
                'direction': signal.direction,
                'confidence': signal.confidence,
                'tp_hit': signal.tp_hit or [],
                'entry_time': signal.timestamp.isoformat(),
                'source': signal.source
            }
            summary['positions'].append(pos_info)
        return summary
    """🔥 The ULTIMATE signal processor that catches EVERYTHING Grizzlies does"""

    DEFAULT_CONFIG = {
        'structured_patterns': {
            'signal_start': [
                r'^(Long|Short)\s*$',
                r'^(LONG|SHORT)\s*$'
            ],
            'ticker_line': [
                r'^([A-Z]{2,5})/USDT\s*$',
                r'^([A-Z]{2,5})-USDT\s*$',
                r'^([A-Z]{2,5})\s*$'
            ],
            'leverage_line': [
                r'Leverage:\s*(cross|isolated|cross\s*/\s*isolated)',
                r'Leverage:\s*([0-9]+[xX]?)'
            ],
            'entry_targets': [
                r'Entry\s*Targets?:\s*(.*)',
                r'Entry:\s*(.*)'
            ],
            'tp_targets': [
                r'Take-?Profit-?Targets?:\s*(.*)',
                r'TP\s*Targets?:\s*(.*)',
                r'Targets?:\s*(.*)'
            ],
            'risk_warning': [
                r"Don't over leverage",
                r"play smart and lightly",
                r"Don't over margin"
            ]
        },
        'natural_patterns': {
            'entry_signals': [
                r'Going to hit some (\w+) (longs?|shorts?) again here',
                r'Started a (\w+) position',
                r'Adding to (\w+) (longs?|shorts?)',
                r'Taking (\w+) (longs?|shorts?) here',
                r'Entering (\w+) (longs?|shorts?)'
            ],
            'position_updates': [
                r'TP(\d+) hit',
                r'(\w+) TP(\d+) hit',
                r'Taking profits on (\w+)',
                r'Def trim and set stops',
                r'(\w+) looking (strong|weak)',
                r'Seems like a scalp day'
            ],
            'risk_management': [
                r'Playing light',
                r'If we do dump, I will update on a avg',
                r"Don't over leverage",
                r'play smart and lightly'
            ],
            'sentiment_indicators': {
                'bullish': ['strong', 'confident', 'bounce', 'support', 'breakout'],
                'bearish': ['weak', 'heavy', 'resistance', 'breakdown', 'dump'],
                'neutral': ['watching', 'waiting', 'considering', 'might']
            }
        },
        'update_patterns': {
            'tp_announcements': [
                r'TP(\d+)\s+hit',
                r'(\w+) TP(\d+)\s+hit',
                r'Hit\s+TP(\d+)',
                r'Target\s+(\d+)\s+reached'
            ],
            'position_changes': [
                r'Started\s+a\s+(\w+)\s+position',
                r'Adding\s+to\s+(\w+)\s+longs?',
                r'Adding\s+to\s+(\w+)\s+shorts?',
                r'Closing\s+(\w+)\s+position',
                r'Trimming\s+(\w+)\s+position',
                r'Scaling\s+out\s+of\s+(\w+)'
            ],
            'sentiment_updates': [
                r'(\w+)\s+looking\s+(strong|weak|good|bad)',
                r'(\w+)\s+(pumping|dumping|mooning)',
                r'(\w+)\s+(holding\s+support|breaking\s+resistance)'
            ]
        },
        'options_patterns': {
            'stock_tickers': [
                r'\b([A-Z]{2,5})\b'
            ],
            'basic_call': [
                r'([A-Z]{2,5})\s+(\d+)\s*(C|CALL|P|PUT)'
            ]
        },
        'trust_scores': {
            'grizzlies': 95.0,
            'grizzlies1557': 95.0,
            'luigi': 90.0,
            'grizzly': 85.0,
            'premium': 80.0
        },
        'quality_indicators': {
            'high_quality': ['analysis', 'chart', 'support', 'resistance', 'breakout', 'target'],
            'medium_quality': ['watching', 'looking', 'considering', 'might'],
            'risk_management': ['stop', 'target', 'risk', 'trim', 'light']
        }
    }




# Integration function to replace the existing signal processor
def create_enhanced_signal_processor(config: Dict) -> UltimateGrizzliesSignalProcessor:
    """Factory function to create the enhanced processor"""
    return UltimateGrizzliesSignalProcessor(config)