#!/bin/bash

# Stop Training Monitor
echo "🛑 Stopping Training Monitor..."

# Kill the monitor script
pkill -f "watch.sh" 2>/dev/null && echo "✅ Monitor stopped" || echo "⚠️ No monitor running"

# Optionally also stop training (uncomment if needed)
# pkill -f "re-orient.py" 2>/dev/null && echo "✅ Training stopped"

echo "Done! 🎯"