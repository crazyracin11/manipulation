#!/bin/bash
# Stop Notebook Training Monitor and Training

echo "🛑 Stopping Notebook Training Monitor and Training..."

# Stop the monitor if running
pkill -f "watch_notebook.py" 2>/dev/null && echo "✅ Monitor stopped" || echo "⚠️ No monitor running"

# Optionally stop training too (uncomment if needed)
pkill -f "re-orient_notebook.py" 2>/dev/null && echo "✅ Training stopped" || echo "⚠️ No training process running"

echo "Done! 🎯"