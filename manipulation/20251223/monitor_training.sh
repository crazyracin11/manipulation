#!/bin/bash

# Auto Monitor Script for Re-Orient Training
# Usage: ./monitor_training.sh [update_interval]

DEFAULT_INTERVAL=30
INTERVAL=${1:-$DEFAULT_INTERVAL}

echo "🚀 Starting Re-Orient Training Monitor"
echo "Update interval: ${INTERVAL} seconds"
echo "Press Ctrl+C to stop monitoring"
echo "================================================"

# Check if training is running
check_training() {
    if pgrep -f "re-orient.py" > /dev/null; then
        echo "✅ Training process is running"
        return 0
    else
        echo "❌ Training process NOT found"
        return 1
    fi
}

# Get GPU status
get_gpu_status() {
    if command -v nvidia-smi &> /dev/null; then
        echo "📊 GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | \
        awk -F',' '{
            printf "  📈 GPU Util: %s%%\n", $1
            printf "  💾 Memory: %sMB / %sMB (%.1fGB used)\n", $2, $3, ($2/1024)
            printf "  🌡️  Temp: %s°C\n", $4
            printf "  ⚡ Power: %sW\n", $5
        }'
    else
        echo "⚠️ nvidia-smi not available"
    fi
}

# Check training progress
check_progress() {
    if [ -f "training_progress.json" ]; then
        local lines=$(wc -l < training_progress.json)
        echo "📈 Progress file: ${lines} entries"

        if [ $lines -gt 0 ]; then
            echo "🕐 Latest progress:"
            tail -2 training_progress.json 2>/dev/null | tail -1 | \
            python3 -c "
import sys, json
try:
    line = sys.stdin.read().strip()
    if line:
        data = json.loads(line)
        ts = data.get('timestamp', 'Unknown')
        steps = data.get('num_steps', 'N/A')
        reward = data.get('episode_reward', 'N/A')
        gpu_info = data.get('gpu_memory_info', '')
        print(f'  Time: {ts}')
        print(f'  Step: {steps}, Reward: {reward}')
        if gpu_info:
            print(f'  GPU Memory: {gpu_info}')
except:
    pass
"
        fi
    else
        echo "📈 Progress file not ready yet"
    fi
}

# Show recent training log
show_recent_log() {
    if [ -f "re-orient_training.log" ]; then
        echo "📝 Recent training log:"
        tail -10 re-orient_training.log | \
        awk '{
            if ($0 ~ /✓|🚀|✅|GPU|Step|Reward|STARTING/) {
                print "  " $0
            } else if ($0 ~ /WARNING|Error|✗/) {
                print "  ⚠️ " $0
            }
        }' | tail -5
    fi
}

# Monitor loop
monitor_count=0
while true; do
    clear
    echo "🚀 Re-Orient Training Monitor - Update #${monitor_count}"
    echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================"

    # Check if training is still running
    if ! check_training; then
        echo ""
        echo "❌ Training has stopped or not started yet"
        echo ""
        echo "🔍 Checking for errors..."
        if [ -f "re-orient_training.log" ]; then
            echo "Last 10 lines of log:"
            tail -10 re-orient_training.log
        fi
        break
    fi

    echo ""
    get_gpu_status
    echo ""
    check_progress
    echo ""
    show_recent_log
    echo ""

    monitor_count=$((monitor_count + 1))
    echo "⏰ Next update in ${INTERVAL} seconds (Update #${monitor_count + 1})"
    echo "================================================"

    sleep $INTERVAL
done

echo ""
echo "🏁 Monitoring stopped"