#!/bin/bash

# Quick Status Check for Training
echo "🔍 Quick Training Status Check"
echo "================================"

# Check if training process is running
if pgrep -f "re-orient.py" > /dev/null; then
    echo "✅ Training Process: RUNNING"
    PID=$(pgrep -f "re-orient.py")
    echo "   PID: $PID"

    # Check if process is using CPU/GPU
    if ps -p $PID -o %cpu --no-headers > /dev/null 2>&1; then
        CPU_USAGE=$(ps -p $PID -o %cpu --no-headers | tr -d ' ')
        echo "   CPU Usage: ${CPU_USAGE}%"
    fi
else
    echo "❌ Training Process: NOT RUNNING"
fi

echo ""

# GPU Status
echo "📊 GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    awk -F',' '{
        printf "   %s\n", $1
        printf "   📈 Utilization: %s%%\n", $2
        printf "   💾 Memory: %sMB / %sMB (%.1fGB)\n", $3, $4, ($3/1024)
    }'
else
    echo "   ⚠️ nvidia-smi not available"
fi

echo ""

# Log file status
echo "📝 Log Files:"
if [ -f "re-orient_training.log" ]; then
    LOG_SIZE=$(du -h re-orient_training.log | cut -f1)
    LOG_LINES=$(wc -l < re-orient_training.log)
    echo "   Training Log: ${LOG_SIZE}, ${LOG_LINES} lines"

    # Show last important line
    LAST_LINE=$(tail -1 re-orient_training.log 2>/dev/null)
    if [[ $LAST_LINE == *"Step"* ]]; then
        echo "   🏆 Latest: ${LAST_LINE}"
    fi
fi

if [ -f "training_progress.json" ]; then
    PROG_LINES=$(wc -l < training_progress.json)
    echo "   Progress File: ${PROG_LINES} entries"
fi

echo ""

# Recent important log entries
echo "🕐 Recent Activity:"
tail -20 re-orient_training.log 2>/dev/null | \
grep -E "(✓|🚀|✅|GPU|Step|Reward|STARTING|Error|ERROR)" | \
tail -3 | sed 's/^/   /'

echo "================================"
echo "📅 Checked at: $(date '+%Y-%m-%d %H:%M:%S')"