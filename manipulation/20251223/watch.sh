#!/bin/bash

# Simple Training Monitor - One Command Start
echo "🚀 Starting Training Monitor..."
echo "Press Ctrl+C to stop"
echo "================================"

# Dynamic monitoring logic
MONITOR_COUNT=0

while true; do
    clear
    echo "🔍 Training Monitor - Check #${MONITOR_COUNT}"
    echo "📅 $(date '+%H:%M:%S')"
    echo "================================"

    # Quick process check
    if pgrep -f "re-orient.py" > /dev/null; then
        echo "✅ Status: TRAINING"

        # GPU info
        if command -v nvidia-smi &> /dev/null; then
            GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1)
            echo "📊 GPU: ${GPU_UTIL}% | Memory: $(echo $GPU_UTIL | cut -d',' -f2)MB"
        fi

        # Log progress
        if [ -f "training_progress.json" ]; then
            LINES=$(wc -l < training_progress.json)
            echo "📈 Progress: ${LINES} entries"
        fi

        # Latest log line
        if [ -f "re-orient_training.log" ]; then
            LAST_LINE=$(tail -1 re-orient_training.log 2>/dev/null)
            if [[ $LAST_LINE == *"Step"* ]]; then
                echo "🏆 ${LAST_LINE}"
            fi
        fi

    else
        echo "❌ Status: NOT RUNNING"
    fi

    echo ""
    echo "Next check in: ${INTERVAL}sec | Ctrl+C to stop"
    echo "================================"

    MONITOR_COUNT=$((MONITOR_COUNT + 1))

    # Dynamic interval: more frequent at start, less frequent later
    if [ $MONITOR_COUNT -lt 10 ]; then
        INTERVAL=15      # First 10 checks: 15s
    elif [ $MONITOR_COUNT -lt 30 ]; then
        INTERVAL=30      # Next 20 checks: 30s
    else
        INTERVAL=60      # After that: 60s
    fi

    sleep $INTERVAL
done