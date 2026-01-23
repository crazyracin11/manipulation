#!/bin/bash
# Notebook Training Monitor - For re-orient_notebook.py
# Usage: ./watch_notebook.sh [update_interval]

DEFAULT_INTERVAL=30
INTERVAL=${1:-$DEFAULT_INTERVAL}

echo "🚀 Starting Notebook Training Monitor"
echo "Update interval: ${INTERVAL} seconds"
echo "Press Ctrl+C to stop monitoring"
echo "================================================"

# Check if training is running
check_training() {
    if pgrep -f "re-orient_notebook.py" > /dev/null; then
        echo "✅ Notebook training process is running"
        return 0
    else
        echo "❌ Notebook training process NOT found"
        return 1
    fi
}

# Get GPU status
get_gpu_status() {
    if command -v nvidia-smi &> /dev/null; then
        echo "📊 GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
        awk -F',' '{
            printf "  📈 GPU Util: %s%%\n", $1
            printf "  💾 Memory: %sMB / %sMB (%.1fGB used)\n", $2, $3, ($2/1024)
            printf "  🌡  Temp: %s°C\n", $4
            printf "  ⚡ Power: %sW\n", $5
        }'
    else
        echo "⚠️ nvidia-smi not available"
    fi
}

# Check training progress from log file
check_progress() {
    if [ -f "re-orient_notebook_training.log" ]; then
        echo "📈 Training Log Analysis:"
        lines=$(wc -l < re-orient_notebook_training.log)
        echo "  Log file: ${lines} lines"

        # Look for key training milestones
        if grep -q "CELL 8" re-orient_notebook_training.log; then
            echo "  ✅ Environment setup completed"
        fi

        if grep -q "CELL 19: TRAIN POLICY SETUP" re-orient_notebook_training.log; then
            echo "  ✅ Training configuration ready"
        fi

        if grep -q "CELL 20: EXECUTE TRAINING" re-orient_notebook_training.log; then
            echo "  ✅ Training started"
        fi

        if grep -q "time to train:" re-orient_notebook_training.log; then
            echo "  ✅ Training timing completed"

            # Extract training time
            train_time=$(grep "time to train:" re-orient_notebook_training.log | tail -1 | cut -d' ' -f4)
            jit_time=$(grep "time to jit:" re-orient_notebook_training.log | tail -1 | cut -d' ' -f4)
            echo "  📊 JIT Time: $jit_time"
            echo "  📊 Train Time: $train_time"
        fi

        if grep -q "CELL 22: ROLLOUT AND VIDEO GENERATION" re-orient_notebook_training.log; then
            echo "  ✅ Rollout and video generation started"
        fi

        # Look for progress indicators
        if grep -q "Step.*Reward.*=" re-orient_notebook_training.log; then
            echo "  📈 Progress tracking active"
            latest_step=$(grep "Step.*Reward.*=" re-orient_notebook_training.log | tail -1 | grep -o "Step [0-9]*")
            latest_reward=$(grep "Step.*Reward.*=" re-orient_notebook_training.log | tail -1 | grep -o "Reward = [0-9.-]*")
            if [ ! -z "$latest_step" ] && [ ! -z "$latest_reward" ]; then
                echo "    Latest: $latest_step $latest_reward"
            fi
        fi

        # Check for video generation
        if grep -q "Video saved as:" re-orient_notebook_training.log; then
            echo "  🎬 Video generation completed"
            video_file=$(grep "Video saved as:" re-orient_notebook_training.log | tail -1 | cut -d' ' -f4)
            if [ -f "$video_file" ]; then
                size=$(ls -lh "$video_file" | awk '{print $5}')
                echo "    📹 File: $video_file ($size)"
            fi
        fi

        # Check for errors
        if grep -q "ERROR\|Failed\|Traceback" re-orient_notebook_training.log; then
            echo "  ⚠️ Errors detected in log"
            echo "    Last 5 error lines:"
            grep -E "ERROR|Failed|Traceback" re-orient_notebook_training.log | tail -5 | sed 's/^/    /'
        fi

    else
        echo "📈 Log file not found yet"
    fi
}

# Check generated files
check_files() {
    echo "📁 Generated Files:"

    # Check for training progress files
    if [ -f "training_progress.json" ]; then
        entries=$(wc -l < training_progress.json)
        echo "  📊 Progress data: $entries entries"
    fi

    # Check for training plots
    plot_files=$(ls training_progress_*.png 2>/dev/null | wc -l)
    if [ $plot_files -gt 0 ]; then
        echo "  📈 Training plots: $plot_files files"
        latest_plot=$(ls -t training_progress_*.png 2>/dev/null | head -1)
        if [ -f "$latest_plot" ]; then
            size=$(ls -lh "$latest_plot" | awk '{print $5}')
            echo "    Latest: $latest_plot ($size)"
        fi
    fi

    # Check for model files
    model_files=$(ls leap_cube_reorient_model_* 2>/dev/null)
    if [ -n "$model_files" ]; then
        echo "  🤖 Model files:"
        ls -lh leap_cube_reorient_model_* 2>/dev/null | awk '{print "    " $9 " $5}'
    fi

    # Check for video files
    video_files=$(ls *rollout.mp4 2>/dev/null)
    if [ -n "$video_files" ]; then
        echo "  🎬 Video files:"
        ls -lh *rollout.mp4 2>/dev/null | awk '{print "    " $9 " $5}'
    fi

    # Check for frame files
    frame_dirs=$(find . -name "*_frame_*" -type f 2>/dev/null | wc -l)
    if [ $frame_dirs -gt 0 ]; then
        echo "  🖼️ Frame files: $frame_dirs files"
    fi
}

# Show recent log tail
show_recent_log() {
    if [ -f "re-orient_notebook_training.log" ]; then
        echo "📝 Recent Log (last 10 lines):"
        tail -10 re-orient_notebook_training.log 2>/dev/null | \
        grep -E "✅|❌|⚠️|🚀|📊|🎬|CELL|time to|Episode|Video|Rollout|Generated" | \
        sed 's/^/  /' ||
        tail -10 re-orient_notebook_training.log 2>/dev/null |
        sed 's/^/  /'
    else
        echo "📝 No log file found yet"
    fi
}

# Monitor loop
monitor_count=0

while true; do
    clear
    echo "🚀 Notebook Training Monitor - Check #${monitor_count}"
    echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================"

    # Check if training is still running
    if ! check_training; then
        echo ""
        echo "❌ Training has stopped or not started yet"
        echo ""
        echo "🔍 Checking final status..."
        check_progress
        check_files

        echo ""
        echo "📝 Final log output:"
        if [ -f "re-orient_notebook_training.log" ]; then
            echo "Last 20 lines of log:"
            tail -20 re-orient_notebook_training.log
        fi

        break
    fi

    echo ""
    get_gpu_status
    echo ""
    check_progress
    echo ""
    check_files
    echo ""
    show_recent_log

    monitor_count=$((monitor_count + 1))

    # Dynamic interval for different phases
    if [ $monitor_count -lt 20 ]; then
        INTERVAL=30      # First 20 checks: 30s (startup phase)
    elif [ $monitor_count -lt 60 ]; then
        INTERVAL=60      # Next 40 checks: 60s (training phase)
    else
        INTERVAL=120     # After that: 2min (stable phase)
    fi

    echo ""
    echo "⏰ Next check in ${INTERVAL}sec | Ctrl+C to stop"
    echo "================================================"

    sleep $INTERVAL
done

echo ""
echo "🏁 Monitoring stopped"
echo "📊 Final results are saved in the generated files"