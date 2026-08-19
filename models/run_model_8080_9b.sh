#!/bin/bash
# Auto-generated: run Qwen3.5-9B-Q6.gguf
"/Volumes/Untitled/untitled folder 4/llama-b9861/llama-server" \
    -m "/Volumes/Untitled/market_data/models/Qwen3.5-9B-Q6.gguf" \
    --host 127.0.0.1 --port 8080 \
    --n-gpu-layers 99 \
    -c 4096 \
    --flash-attn on \
    --no-warmup
