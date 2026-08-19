"""Telegram Bot — push alerts, trade confirmations, daily summaries to Telegram.

Uses Telegram Bot API via HTTP (no heavy dependencies, just httpx).
Config in config.yaml:
  telegram:
    enabled: false
    bot_token: ""  # From @BotFather
    chat_id: ""   # Your Telegram user/group chat ID
"""

import asyncio, logging
from datetime import datetime
import httpx
from config_manager import get_config

logger = logging.getLogger("telegram_bot")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

# Check if httpx is available
try:
    _ = httpx.Client
except ImportError:
    httpx = None
    logger.warning("httpx not available — Telegram bot disabled")


def _load_config():
    cfg = get_config()
    enabled = cfg.get("telegram.enabled", False)
    token = cfg.get("telegram.bot_token", "")
    chat_id = cfg.get("telegram.chat_id", "")
    return enabled, token, chat_id


async def send_message(text, parse_mode="HTML"):
    """Send a message to the configured Telegram chat. Fire-and-forget."""
    enabled, token, chat_id = _load_config()
    if not enabled or not token or not chat_id or not httpx:
        return False

    try:
        url = TELEGRAM_API.format(token=token)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json={
                "chat_id": chat_id,
                "text": text[:4000],  # Telegram 4096 char limit
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            })
            if resp.status_code == 200:
                return True
            data = resp.json()
            if not data.get("ok"):
                logger.warning(f"Telegram API error: {data.get('description', 'unknown')}")
                if data.get("error_code") == 401:
                    logger.warning("Telegram: Invalid bot token")
                return False
            return True
    except Exception as e:
        logger.debug(f"Telegram send failed: {e}")
        return False


async def send_alert(ticker, direction, confidence, price, reasons):
    """Send a trade alert to Telegram."""
    emoji = "🟢" if direction == "BUY" else "🔴"
    direction_arrow = "↗️" if direction == "BUY" else "↘️"
    text = (
        f"{emoji} <b>Trade Signal: {ticker}</b>\n"
        f"{direction_arrow} <b>{direction}</b> at ₹{price:.2f}\n"
        f"🎯 Confidence: {confidence:.0f}%\n"
        f"📊 Reasons: {', '.join(reasons[:3])}"
    )
    ok = await send_message(text)
    if ok:
        logger.info(f"Telegram alert sent: {ticker} {direction}")
    return ok


async def send_trade_execution(symbol, direction, quantity, price, order_id):
    """Send trade execution notification."""
    text = (
        f"✅ <b>Trade Executed</b>\n"
        f"📈 {symbol} {direction} x{quantity}\n"
        f"💵 Price: ₹{price:.2f}\n"
        f"🆔 Order: <code>{order_id}</code>"
    )
    return await send_message(text)


async def send_trade_close(symbol, reason, entry, exit_price, pnl):
    """Send trade close notification."""
    emoji = "✅" if pnl >= 0 else "❌"
    pnl_arrow = "📈" if pnl >= 0 else "📉"
    text = (
        f"{emoji} <b>Trade Closed: {symbol}</b>\n"
        f"📋 Reason: {reason}\n"
        f"📥 Entry: ₹{entry:.2f} → 📤 Exit: ₹{exit_price:.2f}\n"
        f"{pnl_arrow} P&L: ₹{pnl:.2f}"
    )
    return await send_message(text)


async def send_daily_summary():
    """Generate and send a daily market summary."""
    try:
        from market_overview import MarketOverview
        from market_regime import MarketRegime
        from memory_manager import MemoryManager

        mo = MarketOverview()
        mr = MarketRegime()
        mm = MemoryManager()

        # Market breadth
        breadth = mo.get_breadth()
        regime = mr.get_current()
        stats = mm.get_stats()
        accuracy = mm.get_prediction_accuracy()

        text = (
            f"📊 <b>Daily Market Summary</b>\n"
            f"📅 {datetime.now().strftime('%Y-%m-%d')}\n\n"
        )

        if regime:
            text += f"🌡 <b>Regime:</b> {regime.get('regime_label', 'N/A')}\n"

        if breadth:
            adv = breadth.get("advancing", 0)
            dec = breadth.get("declining", 0)
            text += f"📈 Advancing: {adv} | 📉 Declining: {dec}\n"

        text += f"\n🧠 <b>AI Stats</b>\n"
        text += f"📚 Patterns: {stats.get('total_knowledge', 0)}\n"
        text += f"🎯 Predictions: {stats.get('total_predictions', 0)}\n"
        if accuracy and accuracy[0] > 0:
            text += f"✅ Accuracy: {accuracy[1]:.1f}%"

        await send_message(text)
        logger.info("Daily summary sent to Telegram")
        return True
    except Exception as e:
        logger.warning(f"Daily summary failed: {e}")
        return False


async def send_regime_update(regime_dict):
    """Send market regime change notification."""
    if not regime_dict or regime_dict.get("regime") in (None, "unknown"):
        return False

    emojis = {
        "strong_bull": "🚀", "weak_bull": "📈", "ranging": "➡️",
        "weak_bear": "📉", "strong_bear": "💀", "high_volatility": "🎢",
        "low_volatility": "😴",
    }
    emoji = emojis.get(regime_dict.get("regime", ""), "❓")
    text = (
        f"{emoji} <b>Market Regime Update</b>\n"
        f"📊 {regime_dict.get('regime_label', 'N/A')}\n"
        f"📋 {regime_dict.get('details', '')}"
    )
    return await send_message(text)


async def test_connection():
    """Test if Telegram bot is configured and working."""
    enabled, token, chat_id = _load_config()
    if not token:
        return {"ok": False, "error": "No bot token configured"}
    if not chat_id:
        return {"ok": False, "error": "No chat ID configured"}
    ok = await send_message("🤖 <b>Nifty AI Bot</b> is connected and working!")
    return {"ok": ok, "error": None if ok else "Message send failed"}
