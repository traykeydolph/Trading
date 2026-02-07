"""
🔥 EXCHANGE API INTEGRATION - REAL TRADING ENGINE 🔥
==================================================
Multi-exchange support with Kraken/Binance integration
Paper trading + Real money execution
Advanced position tracking and profit management
==================================================
"""

import hmac
import hashlib
import base64
import time
import requests
import json
import logging
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime, UTC
import os
from dotenv import load_dotenv
from requests import RequestException

# ==================================================
# Robust Error Handling Layer (Checkpoint 1A)
# ==================================================
class ExchangeError(Exception):
    """Base exchange error"""

class NetworkError(ExchangeError):
    pass

class APITimeoutError(NetworkError):
    pass

class InvalidSymbolError(ExchangeError):
    pass

class InsufficientFundsError(ExchangeError):
    pass

class ExchangeDownError(ExchangeError):
    pass

class AlertManager:
    """Multi-transport alert dispatcher (log / Telegram / email).

    Environment variables:
        ALERTS_ENABLED=1                Master on/off
        ALERTS_TRANSPORT=log,telegram   Comma list of transports
        ALERT_MIN_LEVEL=INFO            Min level to emit (INFO|WARNING|ERROR|CRITICAL)
        # Telegram
        TELEGRAM_BOT_TOKEN=xxx
        TELEGRAM_CHAT_ID=123456
        # Email (basic SMTP)
        ALERT_EMAIL_FROM=bot@example.com
        ALERT_EMAIL_TO=ops@example.com,dev@example.com
        ALERT_EMAIL_SMTP=smtp.example.com:587
        ALERT_EMAIL_USER=user
        ALERT_EMAIL_PASS=pass
        ALERT_EMAIL_STARTTLS=1
        # Rate limiting
        ALERTS_PER_MINUTE=30
    """
    LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

    def __init__(self):
        self.enabled = os.getenv("ALERTS_ENABLED", "1") not in ("0", "false", "False")
        self.transports = [t.strip() for t in os.getenv("ALERTS_TRANSPORT", "log").split(',') if t.strip()]
        self.min_level = os.getenv("ALERT_MIN_LEVEL", "INFO").upper()
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        # Email config
        self.email_from = os.getenv("ALERT_EMAIL_FROM")
        self.email_to = [e.strip() for e in os.getenv("ALERT_EMAIL_TO", "").split(',') if e.strip()]
        self.email_smtp = os.getenv("ALERT_EMAIL_SMTP")
        self.email_user = os.getenv("ALERT_EMAIL_USER")
        self.email_pass = os.getenv("ALERT_EMAIL_PASS")
        self.email_starttls = os.getenv("ALERT_EMAIL_STARTTLS", "1") not in ("0", "false", "False")
        # Rate limiting
        self.max_per_minute = int(os.getenv("ALERTS_PER_MINUTE", "30"))
        self._recent_times: List[float] = []

    # --------------- Public API ---------------
    def send(self, level: str, message: str, context: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        level_upper = level.upper()
        if self.LEVEL_ORDER.get(level_upper, 100) < self.LEVEL_ORDER.get(self.min_level, 20):
            return

        now = time.time()
        # Prune timestamps older than 60s
        self._recent_times = [t for t in self._recent_times if now - t < 60]
        if len(self._recent_times) >= self.max_per_minute:
            if level_upper in ("CRITICAL", "ERROR"):
                logger.warning("Alert rate limit exceeded; emitting high-priority alert anyway")
            else:
                return
        self._recent_times.append(now)

        context = context or {}
        payload = {"level": level_upper, "message": message, "context": context, "ts": datetime.now(UTC).isoformat()}

        for transport in self.transports:
            try:
                if transport == 'log':
                    self._emit_log(level_upper, message, context)
                elif transport == 'telegram':
                    self._emit_telegram(payload)
                elif transport == 'email':
                    self._emit_email(payload)
            except Exception as e:  # Never let alert failures break trading
                logger.error(f"Alert transport '{transport}' failed: {e}")

    # --------------- Transports ---------------
    def _emit_log(self, level: str, message: str, context: Dict[str, Any]):
        if level in ("CRITICAL", "ERROR"):
            logger.error(f"🔔 ALERT [{level}] {message} | ctx={context}")
        elif level == "WARNING":
            logger.warning(f"🔔 ALERT [{level}] {message} | ctx={context}")
        else:
            logger.info(f"🔔 ALERT [{level}] {message} | ctx={context}")

    def _emit_telegram(self, payload: Dict[str, Any]):
        if not self.telegram_token or not self.telegram_chat_id:
            return  # Silently skip if not configured
        text = self._format_payload_text(payload)
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": self.telegram_chat_id, "text": text, "parse_mode": "Markdown"}, timeout=5)
        except Exception as e:
            logger.warning(f"Telegram alert failed: {e}")

    def _emit_email(self, payload: Dict[str, Any]):
        if not (self.email_from and self.email_to and self.email_smtp):
            return
        try:
            import smtplib
            from email.message import EmailMessage
            msg = EmailMessage()
            msg['Subject'] = f"[BOT {payload['level']}] {payload['message'][:80]}"
            msg['From'] = self.email_from
            msg['To'] = ', '.join(self.email_to)
            msg.set_content(self._format_payload_text(payload))
            host, port = (self.email_smtp.split(':') + ['587'])[:2]
            with smtplib.SMTP(host, int(port), timeout=7) as s:
                if self.email_starttls:
                    s.starttls()
                if self.email_user and self.email_pass:
                    s.login(self.email_user, self.email_pass)
                s.send_message(msg)
        except Exception as e:
            logger.warning(f"Email alert failed: {e}")

    def _format_payload_text(self, payload: Dict[str, Any]) -> str:
        ctx = payload.get('context') or {}
        ctx_str_parts = []
        for k, v in list(ctx.items())[:8]:  # limit to avoid huge messages
            ctx_str_parts.append(f"{k}={v}")
        ctx_str = ' | '.join(ctx_str_parts)
        return f"{payload['ts']} [{payload['level']}] {payload['message']}{(' \n' + ctx_str) if ctx_str else ''}"

alert_manager = AlertManager()

load_dotenv()

logger = logging.getLogger("exchange_api")

@dataclass
class ExchangePosition:
    """Real exchange position with live tracking"""
    position_id: str
    ticker: str
    side: str  # "buy" or "sell"
    size: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    leverage: int
    timestamp: datetime
    tp_levels: List[float] = None
    sl_level: float = None
    status: str = "active"  # active, partial_closed, closed
    fees_paid: float = 0.0

class KrakenFuturesAPI:
    """🔥 Kraken Futures API for REAL trading execution"""
    def __init__(self, api_key: str = None, api_secret: str = None, sandbox: bool = True):
        """Initialize Kraken Futures API client."""
        self.api_key = api_key or os.getenv("KRAKEN_API_KEY")
        self.api_secret = api_secret or os.getenv("KRAKEN_API_SECRET")
        self.sandbox = sandbox

        if sandbox:
            self.base_url = "https://demo-futures.kraken.com"
            logger.info("🧪 Kraken SANDBOX mode - Paper trading enabled")
        else:
            self.base_url = "https://futures.kraken.com"
            logger.info("💰 Kraken LIVE mode - REAL MONEY trading!")

        # Runtime state
        self.session = requests.Session()
        self.max_retries = 3
        self.backoff_base = 1.0
        self.failure_count = 0
        self.status = "INIT"
        self.last_ok = None

        self._test_connection()
    
    def _test_connection(self):
        """Test API connection"""
        try:
            response = self._make_request("GET", "/derivatives/api/v3/instruments")
            if response.get("result") == "success":
                logger.info("✅ Kraken API connection successful!")
                return True
            else:
                logger.error(f"❌ Kraken API connection failed: {response}")
                return False
        except Exception as e:
            logger.error(f"❌ Kraken connection error: {e}")
            return False
    
    def _generate_signature(self, endpoint: str, nonce: str, post_data: str = "") -> str:
        """Generate Kraken API signature"""
        if not self.api_secret:
            return ""
            
        message = post_data + nonce + endpoint
        secret_decoded = base64.b64decode(self.api_secret)
        signature = hmac.new(secret_decoded, message.encode(), hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated API request with retry/backoff and classification"""
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        if self.api_key and endpoint.startswith("/derivatives/api/v3/") and method != "GET":
            nonce = str(int(time.time() * 1000))
            post_data = ""
            if params:
                post_data = "&".join([f"{k}={v}" for k, v in params.items()])

            signature = self._generate_signature(endpoint, nonce, post_data)
            headers.update({
                "APIKey": self.api_key,
                "Nonce": nonce,
                "Authent": signature
            })

        for attempt in range(1, self.max_retries + 1):
            try:
                if method == "GET":
                    response = self.session.get(url, params=params, headers=headers, timeout=10)
                else:
                    response = self.session.post(url, data=params, headers=headers, timeout=10)

                data = response.json()
                if isinstance(data, dict) and data.get("result") == "success":
                    self.failure_count = 0
                    self.status = "OK"
                    self.last_ok = time.time()
                else:
                    # Treat as failure but still return
                    self.failure_count += 1
                return data
            except requests.Timeout as e:
                self.failure_count += 1
                if attempt == self.max_retries:
                    alert_manager.send("ERROR", "Kraken request timeout", {"endpoint": endpoint})
                    return {"result": "error", "error": "timeout"}
            except RequestException as e:
                self.failure_count += 1
                if attempt == self.max_retries:
                    alert_manager.send("ERROR", "Kraken network error", {"endpoint": endpoint, "err": str(e)})
                    return {"result": "error", "error": str(e)}
            except Exception as e:
                self.failure_count += 1
                if attempt == self.max_retries:
                    alert_manager.send("ERROR", "Kraken unknown error", {"endpoint": endpoint, "err": str(e)})
                    return {"result": "error", "error": str(e)}
            # Backoff before retry
            time.sleep(self.backoff_base * attempt)

        # If many failures accumulate mark as DOWN
        if self.failure_count >= self.max_retries:
            self.status = "DOWN"
        return {"result": "error", "error": "unhandled"}
    
    def get_account_info(self) -> Dict:
        """Get account balance and margin info"""
        return self._make_request("GET", "/derivatives/api/v3/accounts")
    
    def get_positions(self) -> Dict:
        """Get all open positions"""
        return self._make_request("GET", "/derivatives/api/v3/openpositions")
    
    def place_order(self, symbol: str, side: str, size: float, order_type: str = "mkt", 
                   leverage: int = 1, stop_loss: float = None, take_profit: float = None) -> Dict:
        """
        Place trading order with advanced parameters
        
        Args:
            symbol: Trading pair (e.g., "PI_XBTUSD", "PI_ETHUSD")
            side: "buy" or "sell"
            size: Order size in contracts
            order_type: "mkt" (market) or "lmt" (limit)
            leverage: Leverage multiplier
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        params = {
            "orderType": order_type,
            "symbol": symbol,
            "side": side,
            "size": int(size)
        }
        
        if leverage > 1:
            params["leverage"] = leverage
        
        if stop_loss:
            params["stopPrice"] = stop_loss
        
        if take_profit:
            params["limitPrice"] = take_profit
        
        return self._make_request("POST", "/derivatives/api/v3/sendorder", params)
    
    def close_position(self, symbol: str, size: float = None) -> Dict:
        """Close position (partial or full)"""
        positions = self.get_positions()
        if positions.get("result") != "success":
            return {"result": "error", "error": "Could not get positions"}
        
        # Find the position
        for pos in positions.get("openPositions", []):
            if pos["symbol"] == symbol:
                close_size = size or abs(pos["size"])
                close_side = "sell" if pos["side"] == "buy" else "buy"
                
                return self.place_order(symbol, close_side, close_size)
        
        return {"result": "error", "error": "Position not found"}
    
    def get_market_price(self, symbol: str) -> float:
        """Get current market price"""
        response = self._make_request("GET", f"/derivatives/api/v3/tickers")
        if response.get("result") == "success":
            for ticker in response.get("tickers", []):
                if ticker["symbol"] == symbol:
                    return float(ticker["last"])
        return 0.0

class BinanceFuturesAPI:
    """🔥 Binance Futures API for REAL trading execution"""
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        """Initialize Binance Futures API client."""
        self.testnet = testnet

        if testnet:
            self.api_key = api_key or os.getenv("BINANCE_TESTNET_API_KEY")
            self.api_secret = api_secret or os.getenv("BINANCE_TESTNET_SECRET")
            self.base_url = os.getenv("BINANCE_TESTNET_URL", "https://testnet.binancefuture.com")
            logger.info("🧪 Binance TESTNET mode - Paper trading enabled")
        else:
            self.api_key = api_key or os.getenv("BINANCE_API_KEY")
            self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
            self.base_url = "https://fapi.binance.com"
            logger.info("💰 Binance LIVE mode - REAL MONEY trading!")

        # Runtime state
        self.session = requests.Session()
        self.max_retries = 3
        self.backoff_base = 1.0
        self.failure_count = 0
        self.status = "INIT"
        self.last_ok = None

        # Symbol filters cache (step size, min qty, tick size, min notional)
        self._symbol_filters: Dict[str, Dict[str, float]] = {}

        self._test_connection()
        # Load exchange info lazily upon first use to reduce startup time on slow networks

    def _load_exchange_info(self):
        """Fetch and cache exchange filters for all symbols."""
        try:
            data = self._make_request("GET", "/fapi/v1/exchangeInfo")
            if isinstance(data, dict) and data.get("symbols"):
                for s in data["symbols"]:
                    sym = s.get("symbol")
                    flt = {"stepSize": None, "minQty": None, "tickSize": None, "minNotional": None}
                    for f in s.get("filters", []):
                        ftype = f.get("filterType")
                        if ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                            flt["stepSize"] = float(f.get("stepSize", 0))
                            flt["minQty"] = float(f.get("minQty", 0))
                        elif ftype == "PRICE_FILTER":
                            flt["tickSize"] = float(f.get("tickSize", 0))
                        elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                            # Futures sometimes use NOTIONAL with minNotional
                            flt["minNotional"] = float(f.get("minNotional", f.get("notional", 0)))
                    self._symbol_filters[sym] = flt
            else:
                logger.warning(f"Could not load Binance exchangeInfo: {data}")
        except Exception as e:
            logger.warning(f"Failed to fetch Binance exchangeInfo: {e}")

    def _get_filters(self, symbol: str) -> Dict[str, float]:
        if not self._symbol_filters:
            self._load_exchange_info()
        return self._symbol_filters.get(symbol.upper(), {})

    def _adjust_quantity(self, symbol: str, qty: float, price: float = None) -> float:
        """Adjust quantity to meet symbol precision and minimums by flooring to step size."""
        try:
            filters = self._get_filters(symbol) or {}
            step = filters.get("stepSize") or 0.0
            min_qty = filters.get("minQty") or 0.0
            min_notional = filters.get("minNotional") or 0.0

            adj = float(qty)
            if step and step > 0:
                # Use Decimal for accurate flooring to step
                step_d = Decimal(str(step))
                qd = Decimal(str(adj))
                # floor to nearest multiple: floor(q/step) * step
                multiples = (qd / step_d).to_integral_value(rounding=ROUND_DOWN)
                adj = float(multiples * step_d)

            if min_qty and adj < min_qty:
                adj = float(min_qty)

            if price and min_notional:
                notional = adj * price
                if notional < min_notional:
                    needed = min_notional / max(price, 1e-9)
                    # round up to meet notional, then re-floor by step
                    adj = needed
                    if step and step > 0:
                        step_d = Decimal(str(step))
                        multiples = (Decimal(str(adj)) / step_d).to_integral_value(rounding=ROUND_DOWN)
                        adj = float(multiples * step_d)

            # Ensure > 0 after flooring
            if adj <= 0:
                adj = 0.0
            return adj
        except Exception:
            return max(0.0, float(qty))
    
    def _test_connection(self):
        """Test API connection"""
        try:
            response = self._make_request("GET", "/fapi/v1/ping")
            if response == {}:  # Binance ping returns empty object on success
                logger.info("✅ Binance API connection successful!")
                return True
            else:
                logger.error(f"❌ Binance API connection failed: {response}")
                return False
        except Exception as e:
            logger.error(f"❌ Binance connection error: {e}")
            return False
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate Binance API signature"""
        if not self.api_secret:
            return ""
        return hmac.new(
            self.api_secret.encode(), 
            query_string.encode(), 
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make authenticated API request with retry/backoff & classification"""
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}

        if params is None:
            params = {}

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            params["signature"] = self._generate_signature(query_string)

        for attempt in range(1, self.max_retries + 1):
            try:
                if method == "GET":
                    response = self.session.get(url, params=params, headers=headers, timeout=10)
                else:
                    response = self.session.post(url, data=params, headers=headers, timeout=10)
                data = response.json()

                if "code" in data and data.get("code") not in (0, None):
                    # Detect insufficient balance specific code -2010
                    if data.get("code") == -2010 or ("insufficient" in data.get("msg", "").lower()):
                        alert_manager.send("WARNING", "Binance insufficient funds", {"endpoint": endpoint, "resp": data})
                    self.failure_count += 1
                else:
                    self.failure_count = 0
                    self.status = "OK"
                    self.last_ok = time.time()
                return data
            except requests.Timeout:
                self.failure_count += 1
                if attempt == self.max_retries:
                    alert_manager.send("ERROR", "Binance request timeout", {"endpoint": endpoint})
                    return {"code": -1, "msg": "timeout"}
            except RequestException as e:
                self.failure_count += 1
                if attempt == self.max_retries:
                    alert_manager.send("ERROR", "Binance network error", {"endpoint": endpoint, "err": str(e)})
                    return {"code": -1, "msg": str(e)}
            except Exception as e:
                self.failure_count += 1
                if attempt == self.max_retries:
                    alert_manager.send("ERROR", "Binance unknown error", {"endpoint": endpoint, "err": str(e)})
                    return {"code": -1, "msg": str(e)}
            time.sleep(self.backoff_base * attempt)

        if self.failure_count >= self.max_retries:
            self.status = "DOWN"
        return {"code": -1, "msg": "unhandled"}
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        return self._make_request("GET", "/fapi/v2/account", signed=True)
    
    def get_positions(self) -> Dict:
        """Get position information"""
        return self._make_request("GET", "/fapi/v2/positionRisk", signed=True)
    
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET",
                   leverage: int = 1, stop_price: float = None, take_profit: float = None, _price_hint: float = None) -> Dict:
        """Place trading order with precision-safe sizing."""
        # Adjust quantity first using exchange filters
        adj_qty = self._adjust_quantity(symbol, quantity, price=_price_hint)
        if adj_qty <= 0:
            return {"code": -1000, "msg": f"Quantity below minimum after adjustment (orig={quantity})"}

        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type,
            "quantity": adj_qty
        }
        
        if order_type == "LIMIT":
            params["timeInForce"] = "GTC"
        
        resp = self._make_request("POST", "/fapi/v1/order", params, signed=True)
        # If precision error, try one more time flooring to fewer decimals
        if isinstance(resp, dict) and resp.get("code") == -1111:
            # Try a secondary floor by trimming to 6 decimal places as a safe bound
            try:
                adj_qty2 = float(Decimal(str(adj_qty)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))
                if adj_qty2 > 0 and adj_qty2 != adj_qty:
                    params["quantity"] = adj_qty2
                    resp = self._make_request("POST", "/fapi/v1/order", params, signed=True)
            except Exception:
                pass
        return resp
    
    def close_position(self, symbol: str, quantity: float = None) -> Dict:
        """Close position"""
        # Get current position first
        positions = self.get_positions()
        for pos in positions:
            if pos["symbol"] == symbol and float(pos["positionAmt"]) != 0:
                position_amt = float(pos["positionAmt"])
                close_side = "SELL" if position_amt > 0 else "BUY"
                close_qty = quantity or abs(position_amt)
                
                return self.place_order(symbol, close_side, close_qty)
        
        return {"code": -1, "msg": "No position found"}
    
    def get_market_price(self, symbol: str) -> float:
        """Get current market price"""
        response = self._make_request("GET", "/fapi/v1/ticker/price", {"symbol": symbol.upper()})
        if "price" in response:
            return float(response["price"])
        return 0.0

class AlpacaAPI:
    """🏦 Alpaca API for STOCK trading execution"""
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper_trading: bool = True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.paper_trading = paper_trading
        
        if paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
            logger.info("🧪 Alpaca PAPER mode - Paper trading enabled")
        else:
            self.base_url = "https://api.alpaca.markets"
            logger.info("💰 Alpaca LIVE mode - REAL MONEY trading!")
        
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json"
        })
        self.max_retries = 3
        self.backoff_base = 1.0
        self.failure_count = 0
        self.status = "INIT"
        self.last_ok = None
        self._test_connection()
    
    def _test_connection(self):
        """Test API connection"""
        try:
            response = self._make_request("GET", "/v2/account")
            if response.get("id"):
                logger.info("✅ Alpaca API connection successful!")
                return True
            else:
                logger.error(f"❌ Alpaca API connection failed: {response}")
                return False
        except Exception as e:
            logger.error(f"❌ Alpaca connection error: {e}")
            return False
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated API request with retry/backoff & error classification"""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(1, self.max_retries + 1):
            try:
                if method == "GET":
                    response = self.session.get(url, params=params, timeout=10)
                elif method == "POST":
                    response = self.session.post(url, json=params, timeout=10)
                elif method == "DELETE":
                    response = self.session.delete(url, timeout=10)
                else:
                    raise ValueError("Unsupported method")

                try:
                    data = response.json()
                except ValueError:
                    # Non-JSON or empty body; capture raw for diagnostics
                    raw_text = response.text[:200]
                    data = {"message": "non_json_response", "status_code": response.status_code, "raw": raw_text, "endpoint": endpoint}
                if isinstance(data, dict) and data.get("code") in (40010001,):  # hypothetical insufficient buying power code
                    alert_manager.send("WARNING", "Alpaca insufficient buying power", {"endpoint": endpoint, "resp": data})
                # Mark success heuristically if an id or list returned
                if response.status_code < 500:
                    self.failure_count = 0
                    self.status = "OK"
                    self.last_ok = time.time()
                else:
                    self.failure_count += 1
                return data
            except requests.Timeout:
                self.failure_count += 1
                if attempt == self.max_retries:
                    alert_manager.send("ERROR", "Alpaca request timeout", {"endpoint": endpoint})
                    return {"message": "timeout"}
            except RequestException as e:
                self.failure_count += 1
                if attempt == self.max_retries:
                    alert_manager.send("ERROR", "Alpaca network error", {"endpoint": endpoint, "err": str(e)})
                    return {"message": str(e)}
            except Exception as e:
                self.failure_count += 1
                if attempt == self.max_retries:
                    alert_manager.send("ERROR", "Alpaca unknown error", {"endpoint": endpoint, "err": str(e)})
                    return {"message": str(e)}
            time.sleep(self.backoff_base * attempt)

        if self.failure_count >= self.max_retries:
            self.status = "DOWN"
        return {"message": "unhandled"}
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        return self._make_request("GET", "/v2/account")
    
    def get_positions(self) -> Dict:
        """Get all positions"""
        return self._make_request("GET", "/v2/positions")
    
    def place_order(self, symbol: str, side: str, qty: float, order_type: str = "market") -> Dict:
        """Place stock order"""
        params = {
            "symbol": symbol.upper(),
            "qty": str(qty),
            "side": side.lower(),
            "type": order_type,
            "time_in_force": "day"
        }
        return self._make_request("POST", "/v2/orders", params)

    def get_order(self, order_id: str) -> Dict:
        """Fetch a single order by id (used to confirm persistence)."""
        if not order_id:
            return {"message": "missing_order_id"}
        try:
            return self._make_request("GET", f"/v2/orders/{order_id}")
        except Exception as e:
            return {"message": str(e)}
    
    def close_position(self, symbol: str, qty: float = None) -> Dict:
        """Close stock position"""
        if qty:
            # Close partial position
            positions = self.get_positions()
            for pos in positions:
                if pos["symbol"] == symbol:
                    current_qty = float(pos["qty"])
                    close_side = "sell" if current_qty > 0 else "buy"
                    return self.place_order(symbol, close_side, qty)
        else:
            # Close entire position
            return self._make_request("DELETE", f"/v2/positions/{symbol.upper()}")
        
        return {"message": "Position not found"}
    
    def get_market_price(self, symbol: str) -> float:
        """Get current market price (Alpaca Stocks)

        Tries quote endpoint then trades fallback. Normalizes field names across API versions.
        """
        sym = symbol.upper()
        # New style endpoint
        resp = self._make_request("GET", f"/v2/stocks/{sym}/quotes/latest")
        try:
            if isinstance(resp, dict) and resp.get('quote'):
                q = resp['quote']
                ask = q.get('ap') or q.get('ask_price') or q.get('ask')
                bid = q.get('bp') or q.get('bid_price') or q.get('bid')
                if ask and bid:
                    return (float(ask) + float(bid)) / 2.0
        except Exception:
            pass
        # Fallback: last trade price
        trade = self._make_request("GET", f"/v2/stocks/{sym}/trades/latest")
        try:
            if isinstance(trade, dict) and trade.get('trade'):
                price = trade['trade'].get('p') or trade['trade'].get('price')
                if price:
                    return float(price)
        except Exception:
            pass
        return 0.0

class UnifiedExchangeAPI:
    """🔥 UNIFIED EXCHANGE API - DUAL-EXCHANGE ROUTING ENGINE"""
    def __init__(self, exchange: str = "binance", paper_trading: bool = True):
        """Initialize unified routing API for crypto (Binance) + stocks (Alpaca)."""
        self.primary_exchange = exchange.lower()
        self.paper_trading = paper_trading
        self.positions: Dict[str, ExchangePosition] = {}
        self.paper_balance = 10000.0  # Starting paper balance

        # Initialize CRYPTO exchange (Binance)
        self.crypto_api = BinanceFuturesAPI(testnet=paper_trading)
        self.crypto_symbols = {
            "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
            "ADA": "ADAUSDT", "DOT": "DOTUSDT", "AVAX": "AVAXUSDT",
            "BITCOIN": "BTCUSDT", "ETHEREUM": "ETHUSDT", "SOLANA": "SOLUSDT",
            "XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "MATIC": "MATICUSDT",
            # Additional bases observed in Grizzlies signals
            "LINK": "LINKUSDT", "ENA": "ENAUSDT"
        }
        # Initialize STOCK exchange (Alpaca) – follow global paper_trading flag to allow future live cutover
        self.stock_api = AlpacaAPI(paper_trading=self.paper_trading)
        self.stock_symbols = {
            "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX",
            "HOOD", "GME", "AMC", "PLTR", "COIN", "SQ", "PYPL", "ROKU",
            "SNAP", "TWTR", "UBER", "LYFT", "ABNB", "SHOP", "CRM", "ORCL",
            "SPY", "INTC", "NIO", "ADBE", "IREN", "ADM", "CDLR", "CMG", "DUOL"
        }

        # Risk / tracking state
        self.max_position_pct = float(os.getenv("MAX_POSITION_PCT", "0.2"))
        self.daily_loss_limit_pct = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.1"))
        self.realized_pnl_today = 0.0
        self.start_of_day_balance = self.paper_balance
        self._last_day = datetime.now(UTC).date()

        # Fee configuration (simple taker fee assumptions)
        self.crypto_fee_rate = float(os.getenv("FEE_RATE_CRYPTO", "0.0004"))  # 4 bps
        self.stock_fee_rate = float(os.getenv("FEE_RATE_STOCK", "0.0005"))   # 5 bps

        logger.info(
            f"🚀 DUAL-EXCHANGE API initialized: CRYPTO→Binance | STOCKS→Alpaca | Paper: {self.paper_trading}"
        )
    
    def _extract_base_quote(self, ticker: str) -> tuple[str, str | None]:
        """Extract base and quote assets from a ticker in various common formats.
        Examples:
        - ETH/USDT -> (ETH, USDT)
        - BTCUSDT  -> (BTC, USDT)
        - SOL-USD  -> (SOL, USD)
        - ADA      -> (ADA, None)
        """
        t = (ticker or "").upper().strip()
        # Normalize common delimiters
        t = t.replace(" ", "").replace("-", "/")

        # If explicit base/quote delimiter present
        if "/" in t:
            parts = t.split("/")
            if len(parts) >= 2 and parts[0]:
                return parts[0], parts[1]
            # Fallback if malformed
            return parts[0] if parts and parts[0] else t, None

        # Suffix-based detection for common quotes
        for q in ("USDT", "USDC", "USD"):
            if t.endswith(q) and len(t) > len(q):
                return t[: -len(q)], q

        # Perp markers sometimes appear, treat as base-only
        if t.endswith("PERP") and len(t) > 4:
            return t[: -4], "PERP"

        # Default: treat entire string as base
        return t, None
    
    def _is_crypto_symbol(self, ticker: str) -> bool:
        """Determine if ticker is crypto or stock"""
        base_asset, quote_asset = self._extract_base_quote(ticker)

        # Enhanced crypto detection:
        # 1. If quote is crypto quote (USDT/USDC), definitely crypto
        # 2. If base is in our crypto symbols, definitely crypto  
        # 3. If ticker contains /, assume crypto (BTC/USDT, ETH/USDT format)
        # 4. If ends with USDT/USDC without /, assume crypto (BTCUSDT format)
        # 5. Otherwise, assume stock
        crypto_quotes = {"USDT", "USDC", "USD"}
        
        # Enhanced logic for better detection
        is_crypto = (
            (quote_asset and quote_asset in crypto_quotes) or 
            (base_asset in self.crypto_symbols) or
            ("/" in ticker and quote_asset in crypto_quotes) or
            ticker.upper().endswith(("USDT", "USDC"))
        )
        
        logger.info(f"🎯 Symbol routing: {ticker} → {'CRYPTO (Binance)' if is_crypto else 'STOCK (Alpaca)'}")
        logger.debug(f"   Base: {base_asset}, Quote: {quote_asset}, InCryptoList: {base_asset in self.crypto_symbols}")
        return is_crypto
    
    def _get_exchange_for_symbol(self, ticker: str):
        """Route symbol to correct exchange"""
        if self._is_crypto_symbol(ticker):
            return self.crypto_api, "crypto"
        else:
            return self.stock_api, "stock"
    
    def _map_crypto_symbol(self, ticker: str) -> str:
        """Map crypto ticker to Binance format"""
        base_asset, quote_asset = self._extract_base_quote(ticker)
        # We standardize to USDT-margined perpetuals on Binance Futures
        # Even if input is USDC/USD, default to USDT unless an explicit override exists
        mapped_symbol = self.crypto_symbols.get(base_asset, f"{base_asset}USDT")
        logger.debug(f"🎯 Crypto mapping: {ticker} → {mapped_symbol} (base={base_asset}, quote={quote_asset})")
        return mapped_symbol
    
    def _map_stock_symbol(self, ticker: str) -> str:
        """Map stock ticker to Alpaca format"""
        clean_ticker = ticker.upper().strip()
        logger.debug(f"🎯 Stock mapping: {ticker} → {clean_ticker}")
        return clean_ticker
    
    def _map_symbol(self, ticker: str) -> str:
        """Unified mapping wrapper choosing crypto or stock mapping."""
        if self._is_crypto_symbol(ticker):
            return self._map_crypto_symbol(ticker)
        return self._map_stock_symbol(ticker)
    
    def open_position(self, ticker: str, direction: str, size: float, leverage: int = 5, 
                     tp_levels: List[float] = None, sl_level: float = None) -> Dict:
        """
        🚀 DUAL-EXCHANGE POSITION OPENING
        Routes crypto→Binance, stocks→Alpaca automatically
        """
        try:
            # Daily reset if date changed
            today = datetime.now(UTC).date()
            if today != self._last_day:
                self._last_day = today
                self.start_of_day_balance = self.paper_balance
                self.realized_pnl_today = 0.0

            # --- Basic validation & risk gate (Checkpoint 2A previews) ---
            # Daily loss limit gate
            current_drawdown = (self.start_of_day_balance - (self.paper_balance + self.realized_pnl_today))
            if self.daily_loss_limit_pct > 0 and current_drawdown > self.start_of_day_balance * self.daily_loss_limit_pct:
                alert_manager.send("ERROR", "Daily loss limit triggered - blocking new trades", {
                    "drawdown": current_drawdown,
                    "limit_pct": self.daily_loss_limit_pct
                })
                return {"status": "error", "message": "Daily loss limit reached - trading halted"}
            if size <= 0:
                return {"status": "error", "message": "Size must be positive"}
            if direction not in ("LONG", "SHORT"):
                return {"status": "error", "message": "Direction must be LONG or SHORT"}

            # Position sizing limit (percentage of balance) – for paper / risk pre-check
            if self.paper_trading:
                if size > self.paper_balance * self.max_position_pct:
                    return {"status": "error", "message": f"Position size ${size:.2f} exceeds max pct {self.max_position_pct*100:.1f}% of balance"}
                if size > self.paper_balance:
                    return {"status": "error", "message": "Insufficient paper balance"}

            # Get correct exchange for this symbol
            exchange_api, exchange_type = self._get_exchange_for_symbol(ticker)

            # Symbol validation
            if exchange_type == "crypto":
                base, _quote = self._extract_base_quote(ticker)
                if base not in self.crypto_symbols:
                    logger.warning(f"Unknown crypto base '{base}' from '{ticker}' – defaulting to {base}USDT mapping")
                # Do not block; mapping will fall back to {base}USDT
            else:
                if ticker.upper() not in self.stock_symbols:
                    return {"status": "error", "message": f"Invalid or unsupported stock symbol: {ticker}"}

            # Exchange health check
            if hasattr(exchange_api, 'status') and exchange_api.status == "DOWN":
                alert_manager.send("ERROR", "Attempt to trade on DOWN exchange", {"ticker": ticker, "exchange": exchange_type})
                return {"status": "error", "message": f"Exchange {exchange_type} currently down"}
            
            if exchange_type == "crypto":
                symbol = self._map_crypto_symbol(ticker)
                side = "buy" if direction == "LONG" else "sell"
                
                # Get current crypto price
                current_price = exchange_api.get_market_price(symbol)
                if current_price == 0:
                    return {"status": "error", "message": "Could not get crypto market price"}
                
                # Calculate crypto position size (USD notional -> contracts)
                contract_size = size / current_price
                logger.debug(f"🔢 Computed crypto qty before exchange adjust: {contract_size} for ${size:.2f} at price {current_price}")
                
                if self.paper_trading:
                    # Binance Testnet execution
                    result = exchange_api.place_order(symbol, side.upper(), contract_size, "MARKET", _price_hint=current_price)
                    
                    if "orderId" in result:
                        logger.info(f"🚀 CRYPTO EXECUTED: {direction} {ticker} | Size: ${size:.2f} | Binance Testnet | Order: {result['orderId']}")
                        # Create paper position record
                        position_id = f"{ticker}-{int(time.time()*1000)}"
                        self.positions[position_id] = ExchangePosition(
                            position_id=position_id,
                            ticker=ticker.upper(),
                            side=side,
                            size=contract_size,  # store quantity (contracts)
                            entry_price=current_price,
                            current_price=current_price,
                            pnl=0.0,
                            pnl_pct=0.0,
                            leverage=leverage,
                            timestamp=datetime.now(UTC),
                            tp_levels=tp_levels or [],
                            sl_level=sl_level,
                            status="active",
                            fees_paid=size * self.crypto_fee_rate  # entry fee on notional USD size
                        )
                        return {
                            "status": "success",
                            "exchange": "binance_testnet",
                            "order_id": result["orderId"],
                            "ticker": ticker,
                            "direction": direction,
                            "usd_size": size,
                            "position_qty": contract_size,
                            "price": current_price,
                            "leverage": leverage,
                            "position_id": position_id
                        }
                    else:
                        msg = result.get("msg") if isinstance(result, dict) else str(result)
                        if isinstance(result, dict) and (result.get("code") == -2010 or (msg and 'insufficient' in msg.lower())):
                            alert_manager.send("WARNING", "Binance insufficient funds", {"ticker": ticker, "resp": result})
                            return {"status": "error", "message": "Insufficient funds (Binance)"}
                        logger.error(f"❌ Binance order failed: {result}")
                        alert_manager.send("ERROR", "Binance order execution failure", {"ticker": ticker, "resp": result})
                        return {"status": "error", "message": f"Binance order failed: {msg}"}
                
            else:  # Stock
                symbol = self._map_stock_symbol(ticker)
                side = "buy" if direction == "LONG" else "sell"
                
                # Get current stock price (may fail if data subscription missing)
                current_price = exchange_api.get_market_price(symbol)
                if current_price == 0:
                    logger.warning(f"Alpaca price unavailable for {symbol} - proceeding with 1 share test order (paper price fallback)")
                    current_price = 1.0  # placeholder to compute fees / PnL heuristics
                
                # Calculate stock position size (number of shares)
                shares = int(size / max(current_price, 0.01))
                if shares <= 0:
                    shares = 1  # Ensure at least 1 share
                
                # Alpaca Paper execution
                result = exchange_api.place_order(symbol, side, shares, "market")

                confirmed = False
                order_status = None
                if "id" in result:
                    # Poll up to 3 times to confirm order presence & status
                    for attempt in range(1, 4):
                        details = exchange_api.get_order(result["id"]) if hasattr(exchange_api, 'get_order') else {}
                        if isinstance(details, dict) and details.get("id") == result["id"]:
                            order_status = details.get("status")
                            confirmed = True
                            logger.info(f"🏦 Alpaca order confirmed (attempt {attempt}) status={order_status}")
                            break
                        time.sleep(1)
                    if not confirmed:
                        logger.warning(f"Alpaca order id {result['id']} not confirmed after polling")
                    logger.info(f"🏦 STOCK EXECUTED: {direction} {ticker} | Shares: {shares} | Alpaca Paper | Order: {result['id']}")
                    # Create paper position record (stocks)
                    position_id = f"{ticker}-{int(time.time()*1000)}"
                    self.positions[position_id] = ExchangePosition(
                        position_id=position_id,
                        ticker=ticker.upper(),
                        side=side,
                        size=shares,  # number of shares
                        entry_price=current_price,
                        current_price=current_price,
                        pnl=0.0,
                        pnl_pct=0.0,
                        leverage=1,
                        timestamp=datetime.now(UTC),
                        tp_levels=tp_levels or [],
                        sl_level=sl_level,
                        status="active",
                        fees_paid=(shares * current_price) * self.stock_fee_rate
                    )
                    return {
                        "status": "success",
                        "exchange": "alpaca_paper" if self.paper_trading else "alpaca_live",
                        "order_id": result["id"],
                        "ticker": ticker,
                        "direction": direction,
                        "shares": shares,
                        "price": current_price,
                        "position_id": position_id,
                        "confirmed": confirmed,
                        "order_status": order_status
                    }
                else:
                    msg = result.get("message") if isinstance(result, dict) else str(result)
                    if msg and 'insufficient' in msg.lower():
                        alert_manager.send("WARNING", "Alpaca insufficient buying power", {"ticker": ticker, "resp": result})
                        return {"status": "error", "message": "Insufficient buying power (Alpaca)"}
                    logger.error(f"❌ Alpaca order failed: {result}")
                    alert_manager.send("ERROR", "Alpaca order execution failure", {"ticker": ticker, "resp": result})
                    return {"status": "error", "message": f"Alpaca order failed: {msg}"}
                    
        except Exception as e:
            logger.error(f"❌ Position opening error: {e}")
            alert_manager.send("ERROR", "Unhandled exception in open_position", {"ticker": ticker, "err": str(e)})
            return {"status": "error", "message": str(e)}

    def close_position(self, ticker: str, percentage: float = 100.0, reason: str = "manual") -> Dict:
        """Close position (full or partial)"""
        try:
            if self.paper_trading:
                # Find and close paper position
                for pos_id, position in self.positions.items():
                    if position.ticker == ticker and position.status == "active":
                        current_price = self.get_current_price(ticker)
                        if current_price == 0:
                            # Fallback to entry price to avoid artificial PnL distortion when feed unavailable
                            current_price = position.entry_price
                        
                        # Calculate PnL
                        if position.side == "buy":  # LONG
                            pnl = (current_price - position.entry_price) * position.size * position.leverage
                        else:  # SHORT
                            pnl = (position.entry_price - current_price) * position.size * position.leverage
                        
                        pnl_pct = (pnl / (position.entry_price * position.size)) * 100
                        
                        # Fee on exit notional (portion closed)
                        fee_rate = self.crypto_fee_rate if position.ticker in self.crypto_symbols else self.stock_fee_rate
                        exit_notional = current_price * position.size * (percentage / 100.0)
                        exit_fee = exit_notional * fee_rate
                        # Adjust pnl for proportional fees (add previously paid proportional entry fee part)
                        # Entry fee proportional to portion closed
                        entry_fee_portion = (position.fees_paid if position.fees_paid else 0.0) * (percentage / 100.0)
                        pnl_after_fees = pnl * (percentage / 100.0) - (exit_fee + entry_fee_portion)
                        # Update paper balance (only realized portion)
                        self.paper_balance += pnl_after_fees
                        # Track realized PnL (full closure adds entire PnL; partial closure proportion)
                        realized_component = pnl if percentage >= 100 else pnl * (percentage / 100.0)
                        self.realized_pnl_today += realized_component
                        
                        # Update position status
                        if percentage >= 100:
                            position.status = "closed"
                        else:
                            position.status = "partial_closed"
                            position.size *= (1 - percentage / 100)
                        
                        # Accumulate fees
                        position.fees_paid += exit_fee
                        position.pnl = pnl_after_fees if percentage >= 100 else pnl_after_fees  # realized portion
                        position.pnl_pct = pnl_pct
                        position.current_price = current_price
                        
                        logger.info(f"📊 PAPER TRADE: Closed {percentage:.1f}% of {ticker} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}")
                        # Daily loss limit post-check (in case closure increases loss)
                        if self.daily_loss_limit_pct > 0:
                            drawdown = (self.start_of_day_balance - (self.paper_balance + self.realized_pnl_today))
                            if drawdown > self.start_of_day_balance * self.daily_loss_limit_pct:
                                alert_manager.send("CRITICAL", "Daily loss limit breached after close", {
                                    "ticker": ticker,
                                    "drawdown": drawdown,
                                    "limit_pct": self.daily_loss_limit_pct
                                })
                        
                        return {
                            "status": "success",
                            "position_id": pos_id,
                            "ticker": ticker,
                            "percentage": percentage,
                            "pnl": pnl_after_fees,
                            "pnl_pct": pnl_pct,
                            "fees_paid": position.fees_paid,
                            "new_balance": self.paper_balance,
                            "reason": reason,
                            "mode": "paper"
                        }
                
                return {"status": "error", "message": f"No active position found for {ticker}"}
            else:
                # Close real position (route by symbol)
                exchange_api, exchange_type = self._get_exchange_for_symbol(ticker)
                if exchange_type == "crypto":
                    symbol = self._map_crypto_symbol(ticker)
                else:
                    symbol = self._map_stock_symbol(ticker)
                result = exchange_api.close_position(symbol)

                success = False
                if exchange_type == "crypto" and isinstance(result, dict) and ("orderId" in result or result.get("result") == "success"):
                    success = True
                elif exchange_type == "stock" and isinstance(result, dict) and not result.get("message"):
                    success = True

                if success:
                    logger.info(f"💰 REAL TRADE: Closed position for {ticker} | Reason: {reason}")
                    return {"status": "success", "ticker": ticker, "percentage": percentage, "reason": reason, "mode": "live"}
                logger.error(f"❌ Failed to close position: {result}")
                return {"status": "error", "message": str(result)}
                    
        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_current_price(self, ticker: str) -> float:
        """Get current market price"""
        exchange_api, exchange_type = self._get_exchange_for_symbol(ticker)
        if exchange_type == "crypto":
            symbol = self._map_crypto_symbol(ticker)
        else:
            symbol = self._map_stock_symbol(ticker)
        return exchange_api.get_market_price(symbol)
    
    def get_positions_summary(self) -> Dict:
        """Get summary of all positions"""
        if self.paper_trading:
            active_positions = [pos for pos in self.positions.values() if pos.status == "active"]
            total_pnl = sum(pos.pnl for pos in self.positions.values())
            
            return {
                "total_positions": len(active_positions),
                "total_pnl": total_pnl,
                "balance": self.paper_balance,
                "positions": [
                    {
                        "ticker": pos.ticker,
                        "side": pos.side,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                        "current_price": pos.current_price,
                        "pnl": pos.pnl,
                        "pnl_pct": pos.pnl_pct,
                        "leverage": pos.leverage
                    }
                    for pos in active_positions
                ]
            }
        else:
            # Aggregate positions from both exchanges
            crypto_positions = self.crypto_api.get_positions() if hasattr(self, 'crypto_api') else []
            stock_positions = self.stock_api.get_positions() if hasattr(self, 'stock_api') else []
            return {"crypto": crypto_positions, "stocks": stock_positions}
    
    def update_position_prices(self):
        """Update current prices for all paper positions"""
        if self.paper_trading:
            for position in self.positions.values():
                if position.status == "active":
                    current_price = self.get_current_price(position.ticker)
                    position.current_price = current_price
                    
                    # Update PnL
                    if position.side == "buy":  # LONG
                        position.pnl = (current_price - position.entry_price) * position.size * position.leverage
                    else:  # SHORT
                        position.pnl = (position.entry_price - current_price) * position.size * position.leverage
                    
                    position.pnl_pct = (position.pnl / (position.entry_price * position.size)) * 100
