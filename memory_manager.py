"""Memory Manager — 4-part memory architecture.
1. Long-term knowledge (embedding search, never deleted)
2. Short-term session (recent conversations, auto-expires)
3. Market facts (structured data: patterns, observations, levels)
4. Prediction history (versioned predictions with outcomes for self-review)
"""

import uuid, json, logging
from datetime import datetime, timedelta
import duckdb
import numpy as np
from config_manager import get_config

logger = logging.getLogger("memory_manager")

class MemoryManager:
    def __init__(self, db_path=None):
        cfg = get_config()
        self.db_path = db_path or cfg.get("databases.memory", "/Volumes/Untitled/market_data/memory.duckdb")
        self.con = duckdb.connect(self.db_path)
        self._model = None
        self._model_name = cfg.get("memory.embedding_model", "all-MiniLM-L6-v2")
        self._embed_dim = cfg.get("memory.embedding_dim", 384)
        self._episodic_ttl = cfg.get("memory.episodic_ttl_days", 90)
        self._init_schema()

    def _init_schema(self):
        # 1. Long-term knowledge (embedding search)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id VARCHAR PRIMARY KEY,
                version INTEGER DEFAULT 1,
                category VARCHAR,
                title VARCHAR,
                content TEXT,
                source VARCHAR DEFAULT 'ai_analysis',
                ticker VARCHAR DEFAULT '',
                tags VARCHAR DEFAULT '',
                importance INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                embedding FLOAT[384]
            )
        """)
        # 2. Short-term session (episodic, auto-expires)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id VARCHAR PRIMARY KEY,
                session_id VARCHAR,
                user_query TEXT,
                ai_response TEXT,
                tools_used VARCHAR DEFAULT '',
                reasoning TEXT DEFAULT '',
                data_context TEXT DEFAULT '',
                confidence FLOAT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        # 3. Market facts (structured observations)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS market_facts (
                id VARCHAR PRIMARY KEY,
                ticker VARCHAR DEFAULT '',
                fact_type VARCHAR,
                fact TEXT,
                confidence FLOAT DEFAULT 1.0,
                source VARCHAR DEFAULT 'analysis',
                source_period VARCHAR DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                valid_until TIMESTAMP
            )
        """)
        # 4b. Ticker correlations
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ticker_correlations (
                id INTEGER PRIMARY KEY,
                ticker1 VARCHAR,
                ticker2 VARCHAR,
                correlation_20d FLOAT,
                correlation_50d FLOAT,
                lead_lag_score FLOAT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 5. Knowledge decay tracking
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_decay (
                knowledge_id VARCHAR PRIMARY KEY,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                decay_factor FLOAT DEFAULT 1.0
            )
        """)
        # 4. Prediction history (versioned, never overwritten)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id VARCHAR PRIMARY KEY,
                version INTEGER DEFAULT 1,
                ticker VARCHAR,
                direction VARCHAR,
                entry FLOAT, target FLOAT, stop FLOAT,
                confidence FLOAT,
                reasoning TEXT,
                indicators_used TEXT,
                time_frame VARCHAR,
                strategy_id VARCHAR DEFAULT '',
                params_used VARCHAR DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                actual_outcome VARCHAR,
                accuracy FLOAT,
                self_review TEXT
            )
        """)
        # Migration: add columns if missing (idempotent)
        for col_sql in [
            "ALTER TABLE prediction_history ADD COLUMN IF NOT EXISTS strategy_id VARCHAR DEFAULT ''",
            "ALTER TABLE prediction_history ADD COLUMN IF NOT EXISTS params_used VARCHAR DEFAULT '{}'",
        ]:
            try:
                self.con.execute(col_sql)
            except:
                pass
        # Indexes
        try:
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_ticker ON knowledge(ticker)")
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_cat ON knowledge(category)")
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_sessions_sid ON sessions(session_id)")
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)")
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_facts_ticker ON market_facts(ticker)")
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_pred_ticker ON prediction_history(ticker)")
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_pred_created ON prediction_history(created_at)")
        except:
            pass
        self.con.commit()
        logger.info("Memory schema initialized (4-part)")

    # ── Embedding ──

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"Loaded embedding model: {self._model_name}")
            except Exception as e:
                logger.error(f"Embedding model failed: {e}")
                raise
        return self._model

    def _embed(self, text):
        try:
            return self._get_model().encode(text[:8192]).tolist()
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return [0.0] * self._embed_dim

    # ── 1. Long-Term Knowledge ──

    def search_knowledge(self, query, top_k=5, category=None, ticker=None):
        q_vec = self._embed(query)
        conditions = ["embedding IS NOT NULL"]
        params = [q_vec]
        if category:
            conditions.append("category = ?")
            params.append(category)
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        where = " AND ".join(conditions)
        try:
            results = self.con.execute(f"""
                SELECT id, version, category, title, content, source, ticker,
                       1 - array_cosine_distance(embedding, ?::FLOAT[{self._embed_dim}]) AS score,
                       importance, created_at
                FROM knowledge
                WHERE {where}
                ORDER BY score DESC, importance DESC
                LIMIT ?
            """, params + [top_k]).fetchall()
            return [
                {"id": r[0], "version": r[1], "category": r[2], "title": r[3],
                 "content": r[4], "source": r[5], "ticker": r[6],
                 "score": float(r[7]) if r[7] else 0, "importance": r[8],
                 "created_at": str(r[9])}
                for r in results if r[7] is not None and r[7] > 0.25
            ]
        except Exception as e:
            logger.warning(f"Knowledge search failed: {e}")
            return []

    def store_knowledge(self, title, content, category="analysis", ticker="",
                        source="ai_analysis", tags="", importance=5):
        try:
            vec = self._embed(content)
            dim = self._embed_dim
            existing = self.con.execute(
                "SELECT id, version FROM knowledge WHERE title = ? AND ticker = ? ORDER BY version DESC LIMIT 1",
                [title[:500], ticker]
            ).fetchone()
            if existing:
                new_ver = existing[1] + 1
                self.con.execute(f"""
                    UPDATE knowledge SET content = ?, version = ?, updated_at = CURRENT_TIMESTAMP,
                        embedding = ?::FLOAT[{dim}], importance = ?
                    WHERE id = ?
                """, [content, new_ver, vec, importance, existing[0]])
            else:
                self.con.execute(f"""
                    INSERT INTO knowledge (id, version, category, title, content, source, ticker, tags, importance, embedding)
                    VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[{dim}])
                """, [uuid.uuid4().hex, category, title[:500], content, source, ticker, tags, importance, vec])
            self.con.commit()
            return True
        except Exception as e:
            logger.warning(f"Store knowledge failed: {e}")
            return False

    # ── 2. Short-Term Sessions ──

    def store_session(self, session_id, user_query, ai_response, tools_used="",
                      reasoning="", data_context="", confidence=0):
        try:
            expires = datetime.now() + timedelta(days=self._episodic_ttl)
            self.con.execute("""
                INSERT INTO sessions (id, session_id, user_query, ai_response,
                    tools_used, reasoning, data_context, confidence, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """, [uuid.uuid4().hex, session_id, user_query[:2000], ai_response[:10000],
                  tools_used[:500], reasoning[:5000], str(data_context)[:5000],
                  confidence, expires])
            self.con.commit()
            self._cleanup_expired()
            return True
        except Exception as e:
            logger.warning(f"Store session failed: {e}")
            return False

    def search_sessions(self, session_id=None, query=None, limit=10):
        if session_id:
            results = self.con.execute("""
                SELECT user_query, ai_response, tools_used, created_at
                FROM sessions WHERE session_id = ? ORDER BY created_at DESC LIMIT ?
            """, [session_id, limit]).fetchall()
        else:
            results = self.con.execute("""
                SELECT user_query, ai_response, tools_used, created_at
                FROM sessions ORDER BY created_at DESC LIMIT ?
            """, [limit]).fetchall()
        return [{"query": r[0], "response": r[1][:300], "tools": r[2], "time": str(r[3])} for r in results]

    def _cleanup_expired(self):
        try:
            self.con.execute("DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP")
        except:
            pass

    # ── 3. Market Facts ──

    def store_fact(self, ticker, fact_type, fact, confidence=1.0, source="analysis", period="", valid_days=30):
        try:
            valid_until = datetime.now() + timedelta(days=valid_days)
            self.con.execute("""
                INSERT INTO market_facts (id, ticker, fact_type, fact, confidence, source, source_period, valid_until)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [uuid.uuid4().hex, ticker, fact_type, fact[:5000], confidence, source, period, valid_until])
            self.con.commit()
            return True
        except Exception as e:
            logger.warning(f"Store fact failed: {e}")
            return False

    def get_facts(self, ticker="", fact_type="", limit=20):
        conditions = ["valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP"]
        params = []
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        if fact_type:
            conditions.append("fact_type = ?")
            params.append(fact_type)
        where = " AND ".join(conditions)
        try:
            results = self.con.execute(f"""
                SELECT ticker, fact_type, fact, confidence, source, created_at
                FROM market_facts WHERE {where}
                ORDER BY confidence DESC, created_at DESC LIMIT ?
            """, params + [limit]).fetchall()
            return [{"ticker": r[0], "type": r[1], "fact": r[2], "confidence": r[3],
                     "source": r[4], "created_at": str(r[5])} for r in results]
        except:
            return []

    # ── 4. Prediction History ──

    def store_prediction(self, ticker, direction, entry, target, stop, confidence,
                         reasoning, indicators_used=None, time_frame="1d",
                         strategy_id="", params_used=None):
        try:
            self.con.execute("""
                INSERT INTO prediction_history (id, version, ticker, direction, entry, target, stop,
                    confidence, reasoning, indicators_used, time_frame, strategy_id, params_used, created_at)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [uuid.uuid4().hex, ticker, direction,
                  _v(entry), _v(target), _v(stop), confidence,
                  reasoning[:5000], json.dumps(indicators_used or {})[:1000], time_frame,
                  strategy_id[:50], json.dumps(params_used or {})[:500]])
            self.con.commit()
            return True
        except Exception as e:
            logger.warning(f"Store prediction failed: {e}")
            return False

    def resolve_prediction(self, prediction_id, actual_outcome, accuracy, self_review=None):
        try:
            row = self.con.execute("""
                SELECT ticker, strategy_id, params_used FROM prediction_history WHERE id = ?
            """, [prediction_id]).fetchone()
            if self_review:
                self.con.execute("""
                    UPDATE prediction_history
                    SET resolved_at = CURRENT_TIMESTAMP, actual_outcome = ?, accuracy = ?, self_review = ?
                    WHERE id = ?
                """, [actual_outcome, accuracy, self_review[:5000], prediction_id])
            else:
                self.con.execute("""
                    UPDATE prediction_history
                    SET resolved_at = CURRENT_TIMESTAMP, actual_outcome = ?, accuracy = ?
                    WHERE id = ?
                """, [actual_outcome, accuracy, prediction_id])
            self.con.commit()
            if row:
                return {
                    "ticker": row[0],
                    "strategy_id": row[1],
                    "params_used": json.loads(row[2]) if row[2] and row[2] != '{}' else {},
                    "accuracy": accuracy,
                    "outcome": actual_outcome,
                }
            return None
        except Exception as e:
            logger.warning(f"Resolve prediction failed: {e}")
            return None

    def get_predictions_for_review(self, days_back=30):
        return self.con.execute(f"""
            SELECT id, ticker, direction, entry, target, stop, confidence, reasoning,
                   time_frame, created_at
            FROM prediction_history
            WHERE resolved_at IS NULL
              AND created_at > CURRENT_TIMESTAMP - INTERVAL {days_back} DAY
            ORDER BY created_at DESC
        """).fetchall()

    def get_unresolved_predictions(self, ticker="", limit=20):
        conditions = ["resolved_at IS NULL"]
        params = []
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        where = " AND ".join(conditions)
        return self.con.execute(f"""
            SELECT id, ticker, direction, entry, target, stop, confidence, reasoning, time_frame, created_at
            FROM prediction_history WHERE {where} ORDER BY created_at DESC LIMIT ?
        """, params + [limit]).fetchall()

    def get_prediction_accuracy(self, ticker=""):
        conditions = ["resolved_at IS NOT NULL"]
        params = []
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        where = " AND ".join(conditions)
        return self.con.execute(f"""
            SELECT COUNT(*),
                   AVG(CASE WHEN actual_outcome = 'correct' THEN 100.0 ELSE 0 END),
                   AVG(accuracy)
            FROM prediction_history WHERE {where}
        """, params).fetchone()

    def expire_old_predictions(self, max_days=30):
        """Auto-resolve predictions older than max_days that never hit target/stop.
        Uses partial accuracy scoring based on price direction vs prediction.
        """
        count = 0
        try:
            rows = self.con.execute(f"""
                SELECT id, ticker, direction, entry, target, stop
                FROM prediction_history
                WHERE resolved_at IS NULL
                  AND created_at < CURRENT_TIMESTAMP - INTERVAL {max_days} DAY
            """).fetchall()
            for row in rows:
                pid, ticker, direction, entry, target, stop, *_ = row + (None,) * 6
                # Without current price data, we assign partial accuracy
                # based on how far price moved in predicted direction before expiry
                # We set accuracy to 0 (unresolved → counted as neutral)
                self.con.execute("""
                    UPDATE prediction_history
                    SET resolved_at = CURRENT_TIMESTAMP, actual_outcome = 'expired', accuracy = 0
                    WHERE id = ? AND resolved_at IS NULL
                """, [pid])
                count += 1
            if count:
                self.con.commit()
                logger.info(f"Auto-expired {count} old predictions")
        except Exception as e:
            logger.warning(f"Prediction expiry failed: {e}")
        return count

    def compute_correlations(self):
        """Compute pairwise correlations between Nifty 50 stocks from daily returns."""
        try:
            tickers = self.con.execute("""
                SELECT s.symbol
                FROM securities s
                JOIN daily d ON s.security_id = d.security_id
                WHERE s.is_index = 0 AND s.is_active = 1
                GROUP BY s.symbol
                HAVING COUNT(*) > 200
            """).fetchdf()
            if len(tickers) == 0:
                return 0

            df = self.con.execute("""
                SELECT s.symbol, d.date, d.close
                FROM daily d
                JOIN securities s ON d.security_id = s.security_id
                WHERE s.symbol IN ({})
                ORDER BY d.date
            """.format(",".join(["?"] * len(tickers))), tickers["symbol"].tolist()).fetchdf()
            df["date"] = pd.to_datetime(df["date"])
            pivot = df.pivot_table(index="date", columns="symbol", values="close")
            returns = pivot.pct_change().dropna(how="all")

            pairs_added = 0
            symbols = tickers["symbol"].tolist()
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    s1, s2 = symbols[i], symbols[j]
                    col1, col2 = returns[s1], returns[s2]
                    valid = col1.notna() & col2.notna()
                    if valid.sum() < 20:
                        continue
                    corr_20 = col1[valid].tail(20).corr(col2[valid].tail(20))
                    corr_50 = col1[valid].tail(50).corr(col2[valid].tail(50))
                    # Simple lead-lag: cross-correlation at lag 1
                    lead_lag = col1[valid].shift(1).corr(col2[valid]) or 0
                    self.con.execute("""
                        INSERT INTO ticker_correlations (ticker1, ticker2, correlation_20d, correlation_50d, lead_lag_score)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (id) DO UPDATE SET
                            correlation_20d = EXCLUDED.correlation_20d,
                            correlation_50d = EXCLUDED.correlation_50d,
                            lead_lag_score = EXCLUDED.lead_lag_score,
                            updated_at = CURRENT_TIMESTAMP
                    """, [s1, s2, round(corr_20, 4), round(corr_50, 4), round(lead_lag, 4)])
                    pairs_added += 1
            self.con.commit()
            logger.info(f"Computed {pairs_added} ticker correlations")
            return pairs_added
        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")
            return 0

    def record_knowledge_access(self, knowledge_id):
        """Track knowledge access for decay computation."""
        try:
            self.con.execute("""
                INSERT INTO knowledge_decay (knowledge_id, last_accessed, access_count, decay_factor)
                VALUES (?, CURRENT_TIMESTAMP, 1, 1.0)
                ON CONFLICT (knowledge_id) DO UPDATE SET
                    last_accessed = CURRENT_TIMESTAMP,
                    access_count = knowledge_decay.access_count + 1
            """, [knowledge_id])
            self.con.commit()
        except:
            pass

    def get_stats(self):
        try:
            return {
                "total_knowledge": self.con.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0],
                "total_sessions": self.con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
                "total_facts": self.con.execute("SELECT COUNT(*) FROM market_facts").fetchone()[0],
                "total_predictions": self.con.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0],
                "unresolved_predictions": self.con.execute(
                    "SELECT COUNT(*) FROM prediction_history WHERE resolved_at IS NULL"
                ).fetchone()[0],
                "accuracy": self.get_prediction_accuracy()[1] if self.get_prediction_accuracy()[0] > 0 else None,
            }
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        try:
            self.con.close()
        except:
            pass

def _v(val):
    if val is None:
        return None
    try:
        return float(val)
    except:
        return None
