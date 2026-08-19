import asyncio, logging, json, httpx, random
from datetime import datetime, time as dtime
import duckdb
import pandas as pd
import numpy as np
from dhan_client import DhanClient, AuthExpiredError
from data_manager import DataManager
from downloader import DataDownloader
from memory_manager import MemoryManager
from agent_tools import ToolRouter
from confidence_engine import ConfidenceEngine, get_engine
from auto_trader import AutoTrader, get_auto_trader
from config_manager import get_config

logger = logging.getLogger("auto_learner")

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

class AutoLearner:
    def __init__(self, dhan: DhanClient, db: DataManager, memory: MemoryManager,
                 downloader: DataDownloader, tool_router: ToolRouter):
        self.dhan = dhan
        self.db = db
        self.memory = memory
        self.downloader = downloader
        self.tools = tool_router
        self.auto_trader = get_auto_trader()
        self._task = None
        self.running = False
        self._current_state = "idle"
        self._feed = None

        self._param_db_path = None
        self._param_con = None

    @property
    def state(self):
        return self._current_state

    @property
    def _param_db(self):
        if self._param_con is None:
            cfg = get_config()
            self._param_db_path = cfg.get("databases.memory", "/Volumes/Untitled/market_data/memory.duckdb")
            self._param_con = duckdb.connect(self._param_db_path)
            self._param_con.execute("""
                CREATE TABLE IF NOT EXISTS strategy_params (
                    id INTEGER PRIMARY KEY,
                    date DATE NOT NULL DEFAULT CURRENT_DATE,
                    strategy_id VARCHAR(50),
                    ticker VARCHAR(20),
                    params_json VARCHAR(500),
                    accuracy DOUBLE DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    win_rate DOUBLE DEFAULT 0,
                    sharpe DOUBLE DEFAULT 0,
                    profit_factor DOUBLE DEFAULT 0,
                    max_dd DOUBLE DEFAULT 0,
                    trades_reviewed INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._param_con.execute("""
                CREATE INDEX IF NOT EXISTS idx_sp_strategy ON strategy_params(strategy_id, ticker)
            """)
        return self._param_con

    def _track_strategy_result(self, strategy_id, ticker, params, accuracy, total_trades, win_rate, sharpe=0, pf=0, max_dd=0):
        """Log a strategy+params combo result for accuracy tracking."""
        try:
            con = self._param_db
            existing = con.execute(
                "SELECT id, total_trades, trades_reviewed, accuracy FROM strategy_params WHERE strategy_id = ? AND ticker = ? AND params_json = ?",
                [strategy_id, ticker, json.dumps(params, sort_keys=True)]
            ).fetchone()
            if existing:
                pid, old_trades, reviewed, old_accuracy = existing
                new_reviewed = reviewed + 1
                # Exponentially weighted average (decay old, favor new)
                alpha = 0.3  # weight for new observation
                new_accuracy = accuracy * alpha + (1 - alpha) * old_accuracy
                con.execute("""
                    UPDATE strategy_params SET
                        accuracy = ?, total_trades = ?, win_rate = ?,
                        sharpe = ?, profit_factor = ?, max_dd = ?,
                        trades_reviewed = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, [round(new_accuracy, 4), total_trades, round(win_rate, 2),
                     round(sharpe, 4), round(pf, 4), round(max_dd, 4),
                     new_reviewed, pid])
            else:
                max_id = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM strategy_params").fetchone()[0]
                con.execute("""
                    INSERT INTO strategy_params
                        (id, strategy_id, ticker, params_json, accuracy, total_trades, win_rate,
                         sharpe, profit_factor, max_dd, trades_reviewed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, [max_id, strategy_id, ticker, json.dumps(params, sort_keys=True),
                     round(accuracy, 4), total_trades, round(win_rate, 2),
                     round(sharpe, 4), round(pf, 4), round(max_dd, 4)])
        except Exception as e:
            logger.debug(f"Track strategy result failed: {e}")

    def _select_best_params(self, strategy_id, ticker, default_params, epsilon=0.2, metric="accuracy"):
        """Epsilon-greedy: 80% best known params, 20% explore (random).
        'explore' returns default params to try something new.
        """
        try:
            con = self._param_db
            rows = con.execute("""
                SELECT params_json, {} FROM strategy_params
                WHERE strategy_id = ? AND ticker = ? AND trades_reviewed >= 2
                ORDER BY {} DESC LIMIT 5
            """.format(metric, metric), [strategy_id, ticker]).fetchall()
            if rows and random.random() > epsilon:
                best_params = json.loads(rows[0][0])
                logger.info(f"  Exploit best {strategy_id}/{ticker} params: {best_params} ({metric}={rows[0][1]:.3f})")
                return best_params
        except Exception as e:
            logger.debug(f"Select best params failed: {e}")

        logger.info(f"  Explore new params for {strategy_id}/{ticker} (default)")
        return dict(default_params)  # explore

    def _llm_suggest_param_adjustment(self, strategy_id, recent_results):
        """Query LLM to suggest param adjustments based on recent trade outcomes."""
        if len(recent_results) < 3:
            return {}
        try:
            summary = "\n".join(
                f"Params: {r['params']}, Accuracy: {r['accuracy']:.1f}%, Trades: {r['trades']}"
                for r in recent_results[-10:]
            )
            prompt = f"""Maine {strategy_id} strategy ke saath trading ki. Results:
{summary}

In results ke basis par, kya maine params mein changes karne chahiye? Sirf numeric param changes batao.
Format: {{"param_name": new_value, ...}} ya {{}} agar koi change nahi chahiye."""

            cfg = get_config()
            h = cfg.get("llama_server.host", "127.0.0.1")
            p = cfg.get("llama_server.port", 8080)
            url = f"http://{h}:{p}/v1/chat/completions"
            import httpx
            resp = httpx.post(url, json={
                "messages": [
                    {"role": "system", "content": "You are a strategy optimizer. Only respond with JSON."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.3,
            }, timeout=30)
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"]
                import re
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    suggested = json.loads(match.group())
                    if isinstance(suggested, dict) and len(suggested) > 0:
                        logger.info(f"  LLM suggests param change: {suggested}")
                        return suggested
        except Exception as e:
            logger.debug(f"LLM param suggestion failed: {e}")
        return {}

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            self.auto_trader.start()
            logger.info("Auto-learner started")

    def stop(self):
        self.running = False
        self.auto_trader.stop()
        if self._task:
            self._task.cancel()
        logger.info("Auto-learner stopped")

    async def _loop(self):
        self.running = True
        while self.running:
            try:
                now = datetime.now()
                is_market_hours = self._is_market_hours(now)
                is_end_of_day = now.hour >= 15 and now.hour <= 16 and now.weekday() < 5
                is_midnight = now.hour == 0 and now.minute < 30

                if is_market_hours:
                    self._current_state = "live_feed"
                    self._publish_state()
                    await self._check_live_feed()

                    self._current_state = "downloading"
                    self._publish_state()
                    logger.info("Auto: Downloading new data...")
                    await self.downloader.download_incremental()

                    self._current_state = "computing_indicators"
                    self._publish_state()
                    logger.info("Auto: Computing indicators...")
                    await self.downloader.download_all_indicators()

                if is_end_of_day:
                    self._current_state = "daily_analysis"
                    self._publish_state()
                    logger.info("Auto: Running daily analysis...")
                    await self._analyze_movers()
                    await self._daily_review()
                    await self._resolve_predictions()
                    await self._analyze_sectors()
                    await self._analyze_options()
                    try:
                        from whatsapp_notifier import send_daily_summary
                        await send_daily_summary()
                    except Exception as e:
                        logger.debug(f"WhatsApp summary failed: {e}")

                if is_midnight:
                    self._current_state = "deep_analysis"
                    logger.info("Auto: Running deep analysis on all stocks...")
                    await self._deep_analyze_all()
                    await self._analyze_multi_timeframe()
                    await self._fetch_market_headlines()
                    await self._store_knowledge_summary()
                    await self._analyze_regime()
                    await self._download_intraday_data()
                    await self._download_expired_options()
                    await self._analyze_expired_options()
                    await self._adapt_parameters()
                    self.memory.expire_old_predictions()
                    try:
                        self.memory.compute_correlations()
                    except:
                        pass

                self._current_state = "sleeping"
                self._publish_state()
                wait = 900 if is_market_hours else 3600
                await asyncio.sleep(wait)

            except AuthExpiredError:
                logger.warning("Auto-learner: Dhan API token expired! Stopping.")
                self.memory.store_knowledge(
                    "API Token Expired",
                    "Dhan API access token expired. Regenerate at https://web.dhan.co",
                    category="system", source="auto_learner"
                )
                self.running = False
                self._current_state = "token_expired"
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Auto-learner error: {e}")
                self._current_state = "error"
                await asyncio.sleep(300)

    async def _analyze_movers(self):
        try:
            df = self.db.get_significant_movers(threshold_pct=2.0)
            if len(df) == 0:
                logger.info("  No significant movers today")
                return

            confidence = get_engine()
            for _, row in df.head(5).iterrows():
                ticker = row["symbol"]
                change = row["change_pct"]
                indicators = await self.tools._exec_get_indicators(ticker)

                if isinstance(indicators, dict):
                    data = await self.tools._exec_get_market_data(ticker, days=5)
                    price = data.get("latest_close") if isinstance(data, dict) else None
                    buy = confidence.score_buy_signal(indicators, price)
                    sell = confidence.score_sell_signal(indicators, price)

                    fact = f"{ticker}: {change:+.2f}% | Buy={buy['signal']}({buy['confidence']}%) | Sell={sell['signal']}({sell['confidence']}%)"
                    self.memory.store_fact(ticker, "mover_analysis", fact,
                        confidence=min(abs(change) / 10, 1.0), period="daily")

                    if buy["signal"] in ("strong", "moderate"):
                        dir_ = "BUY"
                        sig = buy
                    elif sell["signal"] in ("strong", "moderate"):
                        dir_ = "SELL"
                        sig = sell
                    else:
                        dir_ = "NEUTRAL"
                        sig = buy

                    if dir_ != "NEUTRAL" and price:
                        # Log to fine-tune collector
                        try:
                            from finetune_collector import get_collector
                            get_collector().log_prediction(ticker, dir_, price,
                                price * 1.05 if dir_ == "BUY" else price * 0.95,
                                price * 0.98 if dir_ == "BUY" else price * 1.02,
                                sig["confidence"], sig["summary"],
                                features={k: str(v) for k, v in indicators.items() if v is not None})
                        except Exception as e:
                            logger.debug(f"Finetune collect failed for {ticker}: {e}")
                        self.memory.store_prediction(
                                ticker, dir_, price, price * 1.05 if dir_ == "BUY" else price * 0.95,
                                price * 0.98 if dir_ == "BUY" else price * 1.02,
                                sig["confidence"], sig["summary"],
                                indicators_used={k: str(v) for k, v in indicators.items() if v is not None},
                                strategy_id="smc_signal",
                                params_used={"min_confidence": 50},
                            )

                    # Desktop alert for strong signals
                    from alert_system import get_alert_system
                    if buy["signal"] == "strong" or sell["signal"] == "strong":
                        get_alert_system().check_and_alert(ticker, indicators, price)

                    # Auto-trade for strong signals
                    if dir_ != "NEUTRAL" and price:
                        await self.auto_trader.evaluate_and_trade(ticker, indicators, price, sig)

                    # Fetch news for top movers
                    if abs(change) >= 3.0:
                        try:
                            from news_fetcher import get_news_fetcher
                            news = await get_news_fetcher().stock_news(ticker, max_results=2)
                            if news:
                                headlines = "; ".join(n["title"] for n in news)
                                self.memory.store_fact(ticker, "news",
                                    f"News: {headlines[:500]}",
                                    confidence=0.6, source="auto_news")
                                logger.info(f"  News for {ticker}: {headlines[:100]}")
                        except Exception as ne:
                            logger.debug(f"News fetch for {ticker}: {ne}")

                logger.info(f"  Auto-analyzed {ticker}: {change:+.2f}%")
        except Exception as e:
            logger.warning(f"Analyze movers failed: {e}")

    async def _daily_review(self):
        try:
            stats = self.memory.get_stats()
            accuracy = self.memory.get_prediction_accuracy()
            review = (
                f"End of day review. "
                f"Patterns: {stats.get('total_knowledge', 0)}, "
                f"Facts: {stats.get('total_facts', 0)}, "
                f"Predictions: {stats.get('total_predictions', 0)}, "
                f"Accuracy: {accuracy[1]:.1f}%" if accuracy and accuracy[0] > 0
                else f"Predictions: {stats.get('total_predictions', 0)}, no resolutions yet"
            )
            self.memory.store_fact("ALL", "daily_review", review, source="auto_learner")
            logger.info(f"Daily review: {review}")
        except Exception as e:
            logger.warning(f"Daily review failed: {e}")

    async def _analyze_sectors(self):
        """Compute sector performance, trends (weekly/monthly), and detect rotation."""
        try:
            from sector_analysis import SectorAnalyzer
            sa = SectorAnalyzer()
            perf = sa.compute_sector_performance()
            if len(perf) > 0:
                sa.compute_sector_trends()
                rotation = sa.detect_rotation()
                if rotation:
                    self.memory.store_knowledge(
                        f"Sector Rotation: {rotation['date']}",
                        rotation["details"],
                        category="sector_rotation", source="auto_learner",
                        tags=f"leading:{rotation['leading_sector']},lagging:{rotation['lagging_sector']}"
                    )
                    logger.info(f"Sector rotation: {rotation['details']}")
                summary = sa.get_sector_summary(as_dict=True)
                if isinstance(summary, dict) and summary.get("sectors"):
                    top = summary["sectors"][:3]
                    bot = summary["sectors"][-3:]
                    self.memory.store_fact("ALL", "sector_performance",
                        f"Top sectors: {', '.join(s['name'] for s in top)} | "
                        f"Bottom: {', '.join(s['name'] for s in bot)}",
                        confidence=0.8, source="auto_learner")
                    logger.info(f"Sector analysis: {top[0]['name']} leading, {bot[0]['name']} lagging")
                for period in ["weekly", "monthly"]:
                    trends = sa.get_sector_trends(period)
                    if trends.get("sectors"):
                        top3 = trends["sectors"][:3]
                        bot3 = trends["sectors"][-3:]
                        self.memory.store_fact("ALL", f"sector_trend_{period}",
                            f"{period.title()} — Top: {', '.join(s['name'] for s in top3)} | "
                            f"Bottom: {', '.join(s['name'] for s in bot3)}",
                            confidence=0.7, source="auto_learner")
                        logger.info(f"  {period} sector trend: top={top3[0]['name']}, bot={bot3[0]['name']}")
        except Exception as e:
            logger.warning(f"Sector analysis failed: {e}")

    async def _analyze_options(self):
        """Fetch option chain for NIFTY and BANKNIFTY, store signals."""
        oa = None
        try:
            from options_analyzer import OptionsAnalyzer
            oa = OptionsAnalyzer()
            for underlying in ["NIFTY", "BANKNIFTY"]:
                result = await oa.fetch_and_analyze(underlying)
                if result.get("error"):
                    logger.warning(f"  Options {underlying}: {result['error']}")
                    continue
                summary = (
                    f"{underlying}: PCR={result.get('pcr_oi', '?')}, Spot=₹{result.get('spot', 0)}, "
                    f"MaxPain=₹{result.get('max_pain', '?')}, IV_Skew={result.get('iv_skew', 0):+.1f}, "
                    f"Resistance={result.get('top_ce_resistance',[])}, Support={result.get('top_pe_support',[])}"
                )
                self.memory.store_fact(underlying, "options_summary", summary,
                    confidence=0.7, source="options_analyzer")
                self.memory.store_knowledge(
                    f"Options Analysis {underlying} ({result.get('expiry', '?')})",
                    result.get("interpretation", summary),
                    category="options_analysis", ticker=underlying, source="options_analyzer"
                )
                logger.info(f"  Options {underlying}: {summary[:120]}")
        except Exception as e:
            logger.warning(f"Options analysis failed: {e}")
        finally:
            if oa is not None:
                oa.close()

    async def _resolve_predictions(self):
        try:
            unresolved = self.memory.get_predictions_for_review(days_back=7)
            if not unresolved:
                return

            for pred in unresolved:
                pid, ticker, direction = pred[0], pred[1], pred[2]
                entry, target, stop = pred[3], pred[4], pred[5]
                confidence, reasoning = pred[6], pred[7]
                time_frame = pred[8]
                try:
                    data = await self.tools._exec_get_market_data(ticker, days=5)
                    if isinstance(data, dict) and "latest_close" in data:
                        current_price = data["latest_close"]
                        if direction == "BUY":
                            hit_target = current_price >= (target or float("inf"))
                            hit_stop = current_price <= (stop or 0)
                        else:
                            hit_target = current_price <= (target or 0)
                            hit_stop = current_price >= (stop or float("inf"))

                        if hit_target:
                            outcome = "correct"
                            acc = 100.0
                        elif hit_stop:
                            outcome = "incorrect"
                            acc = 0.0
                        else:
                            continue

                        # LLM self-review first (so review text is available)
                        review_text = await self._self_review_prediction(
                            ticker, direction, entry, target, stop,
                            confidence, reasoning, current_price, outcome, acc
                        )

                        resolved = self.memory.resolve_prediction(pid, outcome, acc, self_review=review_text)
                        if resolved and resolved.get("strategy_id"):
                            self._track_strategy_result(
                                strategy_id=resolved["strategy_id"],
                                ticker=resolved["ticker"],
                                params=resolved.get("params_used", {}),
                                accuracy=acc,
                                total_trades=1,
                                win_rate=100.0 if outcome == "correct" else 0.0,
                            )
                        self.memory.store_fact(ticker, "prediction_resolved",
                            f"{ticker} {direction} prediction resolved: {outcome} (acc={acc:.0f}%)",
                            confidence=0.9, source="auto_learner")

                        # Log outcome to fine-tune collector
                        try:
                            from finetune_collector import get_collector
                            fc = get_collector()
                            idx = fc.find_prediction(ticker, direction, entry)
                            if idx >= 0:
                                fc.log_outcome(idx, current_price, outcome, acc)
                        except Exception as e:
                            logger.debug(f"Finetune outcome log failed: {e}")

                        logger.info(f"  Resolved {ticker} {direction}: {outcome}")
                except Exception as e:
                    logger.debug(f"Resolve prediction for {ticker}: {e}")
                    continue
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Resolve predictions failed: {e}")

    async def _self_review_prediction(self, ticker, direction, entry, target, stop,
                                       confidence, reasoning, current_price, outcome, accuracy):
        """Query LLM to review a resolved prediction and extract lessons."""
        try:
            prompt = f"""Aapne {ticker} ke liye {direction} prediction diya tha:
- Entry: {entry}
- Target: {target}
- Stop Loss: {stop}
- Confidence: {confidence}%
- Reasoning: {reasoning}

Actual outcome: {outcome} (accuracy: {accuracy}%)
Current price: {current_price}

Questions to answer in Hinglish:
1. Yeh prediction sahi/glt kyun hui? Kya socha tha vs kya hua?
2. Indicators ya market structure mein kya signal tha jo miss hua?
3. Isse aapne kya seekha? (Learning point)
4. Future mein same pattern aaye to kya karna chahiye?"""

            cfg = get_config()
            h = cfg.get("llama_server.host", "127.0.0.1")
            p = cfg.get("llama_server.port", 8080)
            url = f"http://{h}:{p}/v1/chat/completions"
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, json={
                    "messages": [
                        {"role": "system", "content": "Aap ek self-learning AI stock market analyst hain. Aap apne predictions ka review karte hain aur seekhte hain."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.4,
                })
                if resp.status_code == 200:
                    review = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    if review:
                        self.memory.store_knowledge(
                            f"Self-Review: {ticker} {direction} ({outcome})",
                            f"Prediction review for {ticker}:\n{review}",
                            category="self_review", ticker=ticker,
                            source="auto_review",
                            tags=f"{outcome},{direction.lower()}"
                        )
                        logger.info(f"  Self-review stored for {ticker} {direction}")
                        return review
                return None
        except Exception as e:
            logger.warning(f"Self-review LLM call failed: {e}")
            return None

    async def _deep_analyze_all(self):
        try:
            stocks = self.db.get_all_securities()
            count = 0
            confidence = get_engine()
            for _, row in stocks.iterrows():
                if not self.running:
                    break
                ticker = row["symbol"]
                try:
                    indicators = await self.tools._exec_get_indicators(ticker)
                    if not isinstance(indicators, dict):
                        continue
                    rsi14 = indicators.get("rsi_14") if indicators else None
                    if rsi14 is not None and (rsi14 > 70 or rsi14 < 30):
                        data = await self.tools._exec_get_market_data(ticker, days=5)
                        price = data.get("latest_close") if isinstance(data, dict) else None
                        signal = confidence.score_buy_signal(indicators, price) if rsi14 < 30 else confidence.score_sell_signal(indicators, price)
                        fact = f"RSI(14)={rsi14} extreme. {signal['summary']}"
                        self.memory.store_fact(ticker, "extremes", fact, confidence=0.7, source="auto_deep")
                        count += 1
                        if signal["signal"] in ("strong", "moderate"):
                            data = await self.tools._exec_get_market_data(ticker, days=5)
                            price = data.get("latest_close") if isinstance(data, dict) else None
                            dir_ = "BUY" if rsi14 < 30 else "SELL"
                            if price:
                                try:
                                    from finetune_collector import get_collector
                                    get_collector().log_prediction(ticker, dir_, price,
                                        price * 1.05 if dir_ == "BUY" else price * 0.95,
                                        price * 0.98 if dir_ == "BUY" else price * 1.02,
                                        signal["confidence"], signal["summary"],
                                        features={k: str(v) for k, v in indicators.items() if v is not None},
                                        timeframe="1d")
                                except Exception as e:
                                    logger.debug(f"Finetune collect in deep analysis for {ticker}: {e}")
                                self.memory.store_prediction(ticker, dir_, price,
                                    price * 1.05 if dir_ == "BUY" else price * 0.95,
                                    price * 0.98 if dir_ == "BUY" else price * 1.02,
                                    signal["confidence"], signal["summary"],
                                    indicators_used={k: str(v) for k, v in indicators.items() if v is not None},
                                    time_frame="1d",
                                    strategy_id="smc_signal",
                                    params_used={"min_confidence": 50})
                                if signal["signal"] == "strong":
                                    from alert_system import get_alert_system
                                    get_alert_system().check_and_alert(ticker, indicators, price)
                                await self.auto_trader.evaluate_and_trade(ticker, indicators, price, signal)
                except Exception as e:
                    logger.debug(f"Deep analysis failed for {ticker}: {e}")
                await asyncio.sleep(0.1)
            logger.info(f"Deep analysis: {count} extreme observations found")
        except Exception as e:
            logger.warning(f"Deep analysis failed: {e}")

    async def _analyze_multi_timeframe(self):
        """Compute and store weekly/monthly features for key indices."""
        try:
            from feature_engine import compute_multi_timeframe
            for ticker in ["NIFTY", "BANKNIFTY"]:
                sid = self.tools._lookup_security_id(ticker)
                df = self.db.get_daily_asc(sid, limit=1000)
                if len(df) < 30:
                    continue
                df["symbol"] = ticker
                mtf = compute_multi_timeframe(df)
                summary = f"{ticker}: "
                for tf, feats in mtf.items():
                    if feats:
                        summary += f"{tf}: RSI={feats.get('rsi_14','?')}, Trend={feats.get('trend','?')}, "
                self.memory.store_fact(ticker, "multi_timeframe", summary, confidence=0.7, source="auto_deep")
                logger.info(f"  MTF {ticker}: {summary[:100]}")
        except Exception as e:
            logger.warning(f"Multi-timeframe analysis failed: {e}")

    async def _fetch_market_headlines(self):
        """Fetch general market headlines and store in memory."""
        try:
            from news_fetcher import get_news_fetcher
            news = await get_news_fetcher().search_headlines(max_results=8)
            if news:
                headlines = [n["title"] for n in news if n.get("title")]
                summary = " | ".join(headlines[:6])
                self.memory.store_fact("ALL", "market_headlines",
                    f"Market headlines ({datetime.now().strftime('%Y-%m-%d')}): {summary[:1000]}",
                    confidence=0.5, source="auto_news")
                self.memory.store_knowledge(
                    f"Market Headlines {datetime.now().strftime('%Y-%m-%d')}",
                    "\n".join(f"- {h}" for h in headlines[:10]),
                    category="news", source="auto_news", importance=3
                )
                logger.info(f"Stored {len(headlines)} market headlines")
        except Exception as e:
            logger.debug(f"Market headlines fetch: {e}")

    async def _store_knowledge_summary(self):
        try:
            stats = self.memory.get_stats()
            accuracy = self.memory.get_prediction_accuracy()
            summary = (
                f"Nightly knowledge summary: "
                f"{stats.get('total_knowledge', 0)} patterns, "
                f"{stats.get('total_facts', 0)} market facts, "
                f"{stats.get('total_predictions', 0)} total predictions across "
                f"{stats.get('total_sessions', 0)} analysis sessions."
            )
            if accuracy and accuracy[0] > 0:
                summary += f" Overall accuracy: {accuracy[1]:.1f}%"
            self.memory.store_knowledge("Nightly Summary", summary,
                category="system", source="auto_learner", importance=3)
        except Exception as e:
            logger.debug(f"Nightly knowledge summary failed: {e}")

    async def _analyze_regime(self):
        """Analyze and store market regime."""
        try:
            from market_regime import MarketRegime
            mr = MarketRegime()
            result = await mr.analyze()
            mr.close()
            if result and "error" not in result:
                self.memory.store_fact("NIFTY", "market_regime",
                    f"Market regime: {result['regime_label']} | ADX={result['adx']}, BB={result['bb_width_pct']}%, RSI={result['rsi_14']}",
                    confidence=0.8, source="market_regime")
                self.memory.store_knowledge(
                    f"Market Regime ({result['date']})",
                    result['details'],
                    category="market_regime", ticker="NIFTY", source="market_regime",
                    tags=result['regime']
                )
                logger.info(f"Market regime: {result['regime_label']}")
        except Exception as e:
            logger.warning(f"Regime analysis failed: {e}")

    @staticmethod
    def _is_market_hours(now):
        if now.weekday() >= 5:
            return False
        start = now.replace(hour=9, minute=15, second=0)
        end = now.replace(hour=15, minute=30, second=0)
        return start <= now <= end

    def _publish_state(self):
        """Publish state change via event bus for WebSocket broadcast."""
        try:
            from event_bus import get_event_bus
            get_event_bus().emit("learner_state", {"state": self._current_state})
        except Exception as e:
            logger.debug(f"State publish failed: {e}")

    def trigger_manual_download(self):
        if not self._task or self._task.done():
            self.start()
        asyncio.create_task(self._manual_download())

    async def _manual_download(self):
        await self.downloader.download_incremental()
        await self.downloader.download_all_indicators()

    async def _download_intraday_data(self):
        logger.info("Auto: Downloading intraday data...")
        try:
            result = await self.downloader.download_all_intraday(intervals=(15, 60))
            if result.get("status") == "success":
                logger.info(f"  Intraday: {result['rows_downloaded']} rows")
        except Exception as e:
            logger.warning(f"  Intraday download failed: {e}")

    async def _download_expired_options(self):
        """Download expired options history for NIFTY & BANKNIFTY ATM strikes."""
        logger.info("Auto: Downloading expired options data...")
        indices = [("NIFTY", "13"), ("BANKNIFTY", "25")]
        total = 0
        try:
            for name, sid in indices:
                for strike_pos in ["ATM", "ATM+1", "ATM-1"]:
                    total += await self.downloader.download_expired_options(
                        name, sid, "NSE_FNO", "OPTIDX",
                        expiry_flag="WEEK", expiry_code=1,
                        strike=strike_pos, interval=5,
                    )
                    await asyncio.sleep(0.5)
            logger.info(f"  Expired options: {total} rows")
        except Exception as e:
            logger.warning(f"  Expired options download failed: {e}")

    async def _analyze_expired_options(self):
        """Analyze expired options history for IV trend / sentiment signals."""
        oa = None
        try:
            from options_analyzer import OptionsAnalyzer
            oa = OptionsAnalyzer()
            for u in ["NIFTY", "BANKNIFTY"]:
                result = await asyncio.to_thread(oa.analyze_expired_history, u, 30)
                if result.get("latest"):
                    latest = result["latest"]
                    sent = result.get("sentiment", "neutral")
                    summary = (
                        f"{u} ExpOpt: IV_spread={latest.get('iv_spread',0):+.1f} "
                        f"OI_ratio={latest.get('oi_ratio',0):.2f} "
                        f"Premium_ratio={latest.get('premium_ratio',0):.2f} "
                        f"→ {sent}"
                    )
                    self.memory.store_fact(u, "expired_options_sentiment", summary,
                        confidence=0.6, source="auto_learner")
                    self.memory.store_knowledge(
                        f"Expired Options {u}",
                        summary + f" | Trend: {result.get('iv_spread_trend_dir', 'stable')}",
                        category="options_analysis", ticker=u, source="auto_learner",
                    )
                    logger.info(f"  {summary}")
        except Exception as e:
            logger.warning(f"  Expired options analysis failed: {e}")
        finally:
            if oa is not None:
                oa.close()

    async def _adapt_parameters(self):
        """Midnight: query LLM to suggest param adjustments, validate, and auto-apply."""
        try:
            con = self._param_db
            for sid in ["rsi_mean_reversion", "sma_crossover", "bollinger_bounce", "vwap_reversion", "smc_signal"]:
                if not self.running:
                    break
                rows = con.execute("""
                    SELECT ticker, params_json, accuracy, trades_reviewed, sharpe
                    FROM strategy_params WHERE strategy_id = ? AND trades_reviewed >= 2
                    ORDER BY accuracy DESC LIMIT 5
                """, [sid]).fetchall()
                if len(rows) < 2:
                    continue
                recent = [{"ticker": r[0], "params": r[1], "accuracy": r[2],
                           "trades": r[3], "sharpe": r[4]} for r in rows]
                suggested = self._llm_suggest_param_adjustment(sid, recent)
                if suggested:
                    self.memory.store_knowledge(
                        f"Param Suggestion: {sid}",
                        json.dumps(suggested, indent=2),
                        category="strategy_optimization",
                        source="auto_learner",
                        tags=f"suggestion,{sid}",
                    )
                    logger.info(f"  {sid}: LLM suggests {suggested}")
                    # Auto-apply to auto-trader if valid
                    if sid == "smc_signal":
                        auto_trader = get_auto_trader()
                        applied = []
                        for k, v in suggested.items():
                            if k in ("sl_atr",) and isinstance(v, (int, float)) and 1 <= v <= 5:
                                auto_trader.sl_atr = float(v)
                                applied.append(f"{k}={v}")
                            elif k in ("tp_atr",) and isinstance(v, (int, float)) and 1 <= v <= 10:
                                auto_trader.tp_atr = float(v)
                                applied.append(f"{k}={v}")
                            elif k in ("min_confidence",) and isinstance(v, (int, float)) and 20 <= v <= 95:
                                auto_trader.min_confidence = float(v)
                                applied.append(f"{k}={v}")
                        if applied:
                            logger.info(f"  Auto-applied to auto-trader: {', '.join(applied)}")
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.warning(f"Parameter adaptation failed: {e}")

    async def _check_live_feed(self):
        if self._feed is None:
            try:
                from dhan_feed import get_live_feed
                self._feed = get_live_feed()
            except Exception as e:
                logger.debug(f"Feed init failed: {e}")
                return
        if not self._feed.latest:
            return
        # Verify data is fresh (within 120s)
        try:
            age = (datetime.now() - self._feed.latest_updated).total_seconds()
        except AttributeError:
            age = 999
        if age > 120:
            logger.debug(f"Feed data stale ({age:.0f}s old), reinitializing")
            self._feed = None
            return
        prices = self._feed.get_all_prices()
        if not prices:
            return
        big_movers = []
        try:
            for sym, ltp in prices.items():
                sid = self.db.get_security_id(sym)
                if not sid:
                    continue
                ind = self.db.get_indicators(sid, limit=1)
                if ind is not None and len(ind) > 0:
                    r = ind.iloc[0]
                    sma20 = r.get("sma_20", 0)
                    rvol = r.get("rvol", 0)
                    prev_close = float(self._feed.latest.get(
                        str(sid), {}).get("close", 0))
                    chg_pct = ((ltp - prev_close) / prev_close * 100) if prev_close else 0
                    if abs(chg_pct) > 2 or (rvol and rvol > 2.0):
                        big_movers.append(f"{sym} LTP={ltp} Chg={chg_pct:+.2f}% RVOL={rvol}")
            if big_movers:
                fact = " | ".join(big_movers[:5])
                self.memory.store_fact("LIVE", "live_movers", fact,
                    confidence=min(len(big_movers) / 10, 1.0), source="dhan_feed")
        except:
            pass
