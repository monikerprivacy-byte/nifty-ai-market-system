"""Model Downloader + Speed Optimizer.

Downloads smaller/faster quantized models and manages model selection.
Also provides a pre-filter that avoids invoking the LLM unnecessarily.

Available models:
- Qwen3.5-9B-Q6 (current): 6.9 GB, 9B params, Q6_K — best quality, slowest
- Qwen2.5-7B-Q4: ~4.5 GB, 7B params, Q4_K_M — 2x faster, 90% quality
- Qwen2.5-3B-Q4: ~2 GB, 3B params, Q4_K_M — 5x faster, 75% quality
- Qwen2.5-1.5B-Q4: ~1 GB, 1.5B params — 10x faster, good for routing

Strategy (configurable):
  model.primary: Full analysis (deep, multi-timeframe)
  model.router: Quick classification (routing, pre-filter)
  model.pre_filter: Skip LLM if confidence < threshold
"""

import asyncio, logging, os, json, urllib.request
from pathlib import Path
from config_manager import get_config

logger = logging.getLogger("model_manager")

MODELS_DIR = Path("/Volumes/Untitled/market_data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace GGUF repos for fast models
MODEL_SOURCES = {
    "qwen2.5-7b-q4": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.5,
        "params": "7B",
        "quant": "Q4_K_M",
        "description": "2x faster than 9B, ~90% quality",
    },
    "qwen2.5-3b-q4": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_gb": 2.0,
        "params": "3B",
        "quant": "Q4_K_M",
        "description": "5x faster, good for routine analysis",
    },
    "qwen2.5-1.5b-q4": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_gb": 1.0,
        "params": "1.5B",
        "quant": "Q4_K_M",
        "description": "10x faster, ideal for routing/pre-filter",
    },
}

# ├─ Pre-filter Logic ──
# Skip LLM entirely if confidence engine signals are weak.
# Only invoke LLM for self-review and deep analysis.
# This reduces LLM calls by ~80%.

PRE_FILTER_RULES = {
    "skip_llm_on_weak_signals": True,       # Don't call LLM for < 55% confidence
    "skip_llm_for_routine_updates": True,    # Daily data updates don't need LLM
    "llm_only_for_review_and_deep": True,    # Only use LLM for self-review + deep analysis
    "confidence_engine_only_for_trading": True,  # Use confidence engine directly for trading decisions
}


def get_available_models():
    """List all downloaded GGUF models (in root dir or subdirectories)."""
    models = []
    for f in MODELS_DIR.rglob("*.gguf"):
        size_gb = f.stat().st_size / (1024**3) if f.exists() else 0
        models.append({
            "path": str(f),
            "name": f.stem,
            "size_gb": round(size_gb, 1),
        })
    return models


def get_recommended_model():
    """Return the best available model. Prefers smaller fast model if available."""
    available = get_available_models()
    if not available:
        return None

    # Prefer fastest model that's available (3B > 7B > 9B > 1.5B)
    preferred_order = ["qwen2.5-3b", "qwen2.5-7b", "qwen3.5-9b", "qwen2.5-1.5b"]
    for name in preferred_order:
        for m in available:
            if name in m["name"].lower():
                return m

    # Fallback to smallest (fastest)
    available.sort(key=lambda x: x["size_gb"])
    return available[0]


async def download_model(model_id, progress_callback=None):
    """Download a model from HuggingFace with progress reporting (non-blocking)."""
    import asyncio, concurrent.futures

    if model_id not in MODEL_SOURCES:
        return {"error": f"Unknown model: {model_id}"}

    info = MODEL_SOURCES[model_id]
    dest = MODELS_DIR / model_id / f"{model_id}.gguf"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        actual_gb = dest.stat().st_size / (1024**3)
        expected_gb = info["size_gb"]
        if actual_gb >= expected_gb * 0.9:
            return {"status": "exists", "path": str(dest), "size_gb": round(actual_gb, 1)}
        else:
            logger.warning(f"Partial download detected ({actual_gb:.1f}/{expected_gb} GB), re-downloading...")
            dest.unlink()

    logger.info(f"Downloading {model_id} ({info['size_gb']} GB)...")

    def report(block_count, block_size, total_size):
        downloaded = block_count * block_size / (1024**3)
        total = total_size / (1024**3) if total_size > 0 else 0
        pct = min(downloaded / total * 100, 100) if total > 0 else 0
        if progress_callback:
            progress_callback(round(pct, 1), downloaded, total)
        if block_count % 100 == 0:
            logger.info(f"  Download: {downloaded:.1f}/{total:.1f} GB ({pct:.0f}%)")

    def _sync_download():
        try:
            urllib.request.urlretrieve(info["url"], str(dest), report)
            return {"status": "downloaded", "path": str(dest), "size_gb": round(dest.stat().st_size / (1024**3), 1)}
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if dest.exists():
                dest.unlink()
            return {"error": str(e)}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _sync_download)
    logger.info(f"Download complete: {model_id} ({dest})")
    return result


def create_model_runner_script(model_path, port=8081):
    """Create a run script for a specific model on a specific port."""
    script = f"""#!/bin/bash
# Auto-generated: run {Path(model_path).name}
"/Volumes/Untitled/untitled folder 4/llama-b9861/llama-server" \\
    -m "{model_path}" \\
    --host 127.0.0.1 --port {port} \\
    --n-gpu-layers 99 \\
    -c 8192 \\
    --mlock \\
    --no-kv-offload
"""
    script_path = MODELS_DIR / f"run_model_{port}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    logger.info(f"Runner script created: {script_path}")
    return str(script_path)


# ── LLM Pre-Filter ──

class LLMPreFilter:
    """Decides whether to invoke the LLM or use rule-based analysis directly.

    Goal: reduce LLM calls by 80%+ by using confidence engine for routine decisions.
    """

    def __init__(self):
        cfg = get_config()
        self.min_confidence_for_llm = float(cfg.get("model.pre_filter_confidence", 55))
        self.llm_for_review_only = cfg.get("model.llm_only_for_review", True)

    def should_call_llm(self, context_type, confidence=None, signal=None):
        """Quick decision: call LLM or skip?"""
        # Always call LLM for:
        if context_type == "self_review":
            return True, "Self-review needs reasoning"
        if context_type == "deep_analysis":
            return True, "Deep analysis benefits from LLM"

        # Usually skip LLM for:
        if context_type == "routine_update":
            return False, "Routine — no LLM needed"

        if context_type == "trading_signal":
            if self.llm_for_review_only:
                return False, "Use confidence engine directly for trading"

            if confidence is not None and confidence < self.min_confidence_for_llm:
                return False, f"Confidence {confidence}% < threshold {self.min_confidence_for_llm}%"

            if signal in ("neutral", "weak"):
                return False, f"Signal too weak ({signal})"

            return True, f"Strong signal ({signal}, {confidence}%) needs LLM"

        if context_type == "market_analysis":
            return False, "Market analysis via confidence engine is sufficient"

        # Default: skip
        return False, "No context match"

    def get_routed_config(self, llm_host="127.0.0.1", llm_port=8080):
        """Return API config for the LLM server."""
        return {
            "host": llm_host,
            "port": llm_port,
            "url": f"http://{llm_host}:{llm_port}/v1/chat/completions",
        }


# Singleton
_instance = None
_pre_filter = None


def get_model_manager():
    global _instance
    if _instance is None:
        _instance = type("ModelManager", (), {
            "get_available_models": get_available_models,
            "get_recommended_model": get_recommended_model,
            "download_model": download_model,
            "create_runner_script": create_model_runner_script,
        })()
    return _instance


def get_pre_filter():
    global _pre_filter
    if _pre_filter is None:
        _pre_filter = LLMPreFilter()
    return _pre_filter
