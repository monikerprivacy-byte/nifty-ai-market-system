"""Configuration Manager — loads from config.yaml. No hardcoded values anywhere."""
import os, yaml, logging
from pathlib import Path

CONFIG_PATH = Path("/Volumes/Untitled/market_data/config.yaml")
_config = None

class Config:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        if name in self._data:
            val = self._data[name]
            return Config(val) if isinstance(val, dict) else val
        raise AttributeError(f"Config key '{name}' not found")

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def to_dict(self):
        return self._data

def load_config(path=None):
    global _config
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        data = yaml.safe_load(f)

    env_overrides = {
        "dhan_api.access_token": os.environ.get("DHAN_ACCESS_TOKEN"),
        "dhan_api.client_id": os.environ.get("DHAN_CLIENT_ID"),
    }
    for dotted_key, value in env_overrides.items():
        if value:
            keys = dotted_key.split(".")
            node = data
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = value

    _config = Config(data)
    # Validate required keys
    required_keys = [
        "app.data_dir", "app.log_level",
        "dhan_api.client_id", "dhan_api.access_token",
        "databases.market", "databases.memory",
        "llama_server.host", "llama_server.port",
        "data_download.start_date", "data_download.interval_seconds",
        "memory.embedding_model", "memory.embedding_dim",
        "confidence.min_confirmation_signals", "confidence.max_conflicting_signals",
        "trading.mode", "trading.paper_capital", "trading.max_position_pct",
        "trading.max_daily_loss_pct", "trading.max_concentration_pct",
        "model.primary", "model.pre_filter_confidence", "model.llm_only_for_review",
        "model.skip_llm_on_weak_signals",
    ]
    missing = [k for k in required_keys if _config.get(k) is None]
    if missing:
        logger = logging.getLogger("config")
        logger.warning(f"Missing config keys: {', '.join(missing)}")
    # Ensure directories exist
    temp_dir = _config.get("app.temp_dir", "/tmp/nifty_ai")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    data_dir = _config.get("app.data_dir", "")
    if data_dir:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        Path(data_dir + "/backups").mkdir(parents=True, exist_ok=True)
    return _config

def get_config():
    if _config is None:
        return load_config()
    return _config
