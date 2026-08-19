"""Fine-Tuning Data Collector — captures every prediction + outcome as training pairs.

Flow:
1. Each prediction stored by memory_manager is also logged here with full context
2. When prediction resolves, outcome is linked back
3. Data is saved as JSONL (one example per line) suitable for LoRA fine-tuning
4. Format: {"instruction": "...", "input": "...", "output": "..."}

Usage:
  collector = FineTuneCollector()
  collector.log_prediction(ticker, features, prediction, reasoning)
  collector.log_outcome(ticker, prediction_id, actual_price, pnl, correct)

  Then export: collector.export_jsonl("training_data.jsonl")
"""

import json, logging, os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("finetune_collector")

DATA_DIR = Path("/Volumes/Untitled/market_data/finetune_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


class FineTuneCollector:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._examples = []
        self._load_existing()

    def _load_existing(self):
        """Load existing training examples from disk."""
        path = self.data_dir / "training_examples.jsonl"
        if path.exists():
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._examples.append(json.loads(line))
                logger.info(f"Loaded {len(self._examples)} existing training examples")
            except Exception as e:
                logger.warning(f"Failed to load existing examples: {e}")

    def _save(self):
        """Write all examples to disk."""
        path = self.data_dir / "training_examples.jsonl"
        with open(path, "w") as f:
            for ex in self._examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def log_prediction(self, ticker, direction, entry_price, target, stop_loss,
                       confidence, reasoning, features=None, timeframe="1d"):
        """Log a new prediction as a training example (without outcome yet)."""
        example = {
            "type": "prediction",
            "ticker": ticker,
            "direction": direction,
            "entry_price": entry_price,
            "target": target,
            "stop_loss": stop_loss,
            "confidence": confidence,
            "reasoning": reasoning,
            "timeframe": timeframe,
            "features": {k: v for k, v in (features or {}).items() if v is not None},
            "timestamp": datetime.now().isoformat(),
            "training_format": {
                "instruction": f"Analyze {ticker} and give a {'buy' if direction == 'BUY' else 'sell'} signal.",
                "input": json.dumps(features or {}, ensure_ascii=False),
                "output": f"{direction} signal at ₹{entry_price}. "
                          f"Target: ₹{target}, Stop: ₹{stop_loss}. "
                          f"Confidence: {confidence}%. "
                          f"Reasoning: {reasoning}",
            },
            "outcome": None,  # Will be filled by log_outcome
        }
        self._examples.append(example)
        self._save()
        return len(self._examples) - 1  # Return index

    def log_outcome(self, prediction_index, actual_price, outcome, accuracy, pnl=None):
        """Link an outcome to a previous prediction."""
        if prediction_index < 0 or prediction_index >= len(self._examples):
            logger.warning(f"Invalid prediction index: {prediction_index}")
            return False

        example = self._examples[prediction_index]
        example["outcome"] = {
            "actual_price": actual_price,
            "outcome": outcome,  # "correct", "incorrect", "partial"
            "accuracy": accuracy,
            "pnl": pnl,
            "resolved_at": datetime.now().isoformat(),
        }

        # Update training format with outcome
        example["training_format"]["output"] += (
            f"\nOutcome: {outcome} (accuracy: {accuracy}%). "
            f"Actual price: ₹{actual_price}."
        )

        self._save()
        logger.info(f"Outcome logged for {example['ticker']} {example['direction']}: {outcome}")
        return True

    def export_jsonl(self, filename="training_data.jsonl"):
        """Export in standard JSONL format for LoRA fine-tuning."""
        path = self.data_dir / filename
        count = 0
        with open(path, "w") as f:
            for ex in self._examples:
                if ex.get("outcome") is not None:  # Only export resolved predictions
                    tf = ex.get("training_format", {})
                    line = {
                        "instruction": tf.get("instruction", ""),
                        "input": tf.get("input", ""),
                        "output": tf.get("output", ""),
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    count += 1

        logger.info(f"Exported {count} training examples to {path}")
        return {"path": str(path), "count": count}

    def export_chat_format(self, filename="chat_format.jsonl"):
        """Export in chat format for llama.cpp fine-tuning."""
        path = self.data_dir / filename
        count = 0
        with open(path, "w") as f:
            for ex in self._examples:
                if ex.get("outcome") is not None:
                    tf = ex.get("training_format", {})
                    chat = {
                        "messages": [
                            {"role": "system", "content": "You are a stock market analyst. Analyze data and give clear buy/sell signals with reasoning."},
                            {"role": "user", "content": tf.get("instruction", "") + "\n" + tf.get("input", "")},
                            {"role": "assistant", "content": tf.get("output", "")},
                        ]
                    }
                    f.write(json.dumps(chat, ensure_ascii=False) + "\n")
                    count += 1
        logger.info(f"Exported {count} chat-format examples to {path}")
        return {"path": str(path), "count": count}

    def find_prediction(self, ticker, direction, entry_price):
        """Find a prediction by ticker, direction, and entry price. Returns index or -1."""
        for i, ex in enumerate(self._examples):
            if (ex.get("ticker") == ticker
                    and ex.get("direction") == direction
                    and ex.get("entry_price") == entry_price):
                return i
        return -1

    def get_stats(self):
        """Get collector statistics."""
        total = len(self._examples)
        resolved = sum(1 for e in self._examples if e.get("outcome") is not None)
        correct = sum(1 for e in self._examples
                      if e.get("outcome") and e["outcome"].get("outcome") == "correct")
        return {
            "total_predictions_logged": total,
            "resolved": resolved,
            "correct": correct,
            "accuracy_pct": round(correct / max(resolved, 1) * 100, 1),
            "data_dir": str(self.data_dir),
        }


# Singleton
_instance = None

def get_collector():
    global _instance
    if _instance is None:
        _instance = FineTuneCollector()
    return _instance
