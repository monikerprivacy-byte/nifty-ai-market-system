#!/bin/bash
"/Volumes/Untitled/untitled folder 4/llama-b9861/llama-server" \
    -m "/Volumes/Untitled/market_data/models/qwen2.5-3b-q4/qwen2.5-3b-q4.gguf" \
    --host 127.0.0.1 --port 8080 \
    --n-gpu-layers 50 \
    -c 4096 \
    --flash-attn on \
    --no-warmup
