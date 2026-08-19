#!/bin/bash
# Nifty AI - auto-start script for launchd
set -e

MODEL="/Volumes/Untitled/market_data/models/qwen2.5-3b-q4/qwen2.5-3b-q4.gguf"
LLAMA_BIN="/Volumes/Untitled/untitled folder 4/llama-b9861/llama-server"
NIFTY_APP="/Volumes/Untitled/untitled folder 4/nifty_app.py"
MARKET_DIR="/Volumes/Untitled/market_data"
LOG_DIR="/Volumes/Untitled/market_data/logs"
mkdir -p "$LOG_DIR"

# Start llama-server
nohup "$LLAMA_BIN" \
    -m "$MODEL" \
    --host 127.0.0.1 --port 8080 \
    --n-gpu-layers 99 -c 8192 --mlock --no-kv-offload --no-jinja \
    > "$LOG_DIR/llama_server.log" 2>&1 &
LLAMA_PID=$!
echo "llama-server PID: $LLAMA_PID"

# Wait for llama-server to be ready
echo "Waiting for llama-server..."
for i in $(seq 1 30); do
    if curl -s --max-time 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q '"ok"'; then
        echo "llama-server ready after ${i}s"
        break
    fi
    sleep 2
done

# Start nifty_app (runs in foreground so launchd can manage it)
echo "Starting nifty_app..."
cd "$MARKET_DIR"
export PYTHONPATH="$MARKET_DIR:$PYTHONPATH"
exec python3 "$NIFTY_APP"
