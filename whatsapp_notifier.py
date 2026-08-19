"""WhatsApp Notifier — push alerts, trade confirmations, daily summaries via Callmebot API.

Usage:
  1. Add +34 623 78 64 49 to phone contacts
  2. WhatsApp that number: "I allow callmebot to send me messages"
  3. Receive API key by return message
  4. Set phone + apikey in config.yaml:
       whatsapp:
         enabled: false
         phone: "+919XXXXXXXXX"
         apikey: "your_apikey"
"""

import asyncio, logging, httpx
from datetime import datetime
from config_manager import get_config

logger = logging.getLogger("whatsapp")

CALLMEBOT_URL = "https://api.callmebot.com/whatsapp.php"


def _load_config():
    cfg = get_config()
    enabled = cfg.get("whatsapp.enabled", False)
    phone = cfg.get("whatsapp.phone", "")
    apikey = cfg.get("whatsapp.apikey", "")
    return enabled, phone, apikey


async def send_message(text):
    """Send a WhatsApp message via Callmebot. Fire-and-forget."""
    enabled, phone, apikey = _load_config()
    if not enabled or not phone or not apikey:
        return False

    try:
        params = {
            "phone": phone.replace("+", "").replace(" ", ""),
            "text": text[:4000],
            "apikey": apikey,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(CALLMEBOT_URL, params=params)
            if resp.status_code == 200:
                return True
            logger.warning(f"Callmebot error {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        logger.debug(f"WhatsApp send failed: {e}")
        return False


async def send_alert(ticker, direction, confidence, price, reasons):
    emoji = "BUY" if direction == "BUY" else "SELL"
    text = (
        f"*Trade Signal: {ticker}*\n"
        f"{emoji} *{direction}* at Rs{price:.2f}\n"
        f"Confidence: {confidence:.0f}%\n"
        f"Reasons: {', '.join(reasons[:3])}"
    )
    return await send_message(text)


async def send_trade_execution(symbol, direction, quantity, price, order_id):
    text = (
        f"*Trade Executed*\n"
        f"{symbol} {direction} x{quantity}\n"
        f"Price: Rs{price:.2f}\n"
        f"Order: {order_id}"
    )
    return await send_message(text)


async def send_trade_close(symbol, reason, entry, exit_price, pnl):
    emoji = "PROFIT" if pnl >= 0 else "LOSS"
    text = (
        f"*Trade Closed: {symbol}*\n"
        f"Reason: {reason}\n"
        f"Entry: Rs{entry:.2f} -> Exit: Rs{exit_price:.2f}\n"
        f"{emoji} P&L: Rs{pnl:.2f}"
    )
    return await send_message(text)


async def send_daily_summary():
    try:
        from market_overview import MarketOverview
        from market_regime import MarketRegime
        from memory_manager import MemoryManager

        mo = MarketOverview()
        mr = MarketRegime()
        mm = MemoryManager()

        breadth = mo.get_breadth()
        regime = mr.get_current()
        stats = mm.get_stats()
        accuracy = mm.get_prediction_accuracy()

        text = f"*Daily Market Summary*\n{datetime.now().strftime('%Y-%m-%d')}\n\n"
        if regime:
            text += f"Regime: {regime.get('regime_label', 'N/A')}\n"
        if breadth:
            text += f"Advancing: {breadth.get('advancing', 0)} | Declining: {breadth.get('declining', 0)}\n"
        text += f"\n*AI Stats*\nPatterns: {stats.get('total_knowledge', 0)}\nPredictions: {stats.get('total_predictions', 0)}\n"
        if accuracy and accuracy[0] > 0:
            text += f"Accuracy: {accuracy[1]:.1f}%"

        await send_message(text)
        return True
    except Exception as e:
        logger.warning(f"Daily summary failed: {e}")
        return False


async def send_regime_update(regime_dict):
    if not regime_dict or regime_dict.get("regime") in (None, "unknown"):
        return False
    text = (
        f"*Market Regime Update*\n"
        f"{regime_dict.get('regime_label', 'N/A')}\n"
        f"{regime_dict.get('details', '')}"
    )
    return await send_message(text)


async def test_connection():
    enabled, phone, apikey = _load_config()
    if not phone or not apikey:
        return {"ok": False, "error": "Phone or API key not configured"}
    ok = await send_message("Nifty AI Bot is connected and working!")
    return {"ok": ok, "error": None if ok else "Message send failed. Check phone/apikey."}
