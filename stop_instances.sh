#!/bin/bash

# RAG System Multi-Instance Stop Script
# =====================================
# This script stops all running RAG instances

echo "🛑 Stopping all RAG instances..."

SCRIPT_DIR="/home/vane/Downloads/paany_instance"
cd "$SCRIPT_DIR"

# Function to stop instance by PID file
stop_instance() {
    local pid_file=$1
    local instance_name=$(basename "$pid_file" .pid)
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        
        if ps -p $pid > /dev/null 2>&1; then
            echo "⏹️  Stopping $instance_name (PID: $pid)..."
            kill $pid
            
            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! ps -p $pid > /dev/null 2>&1; then
                    echo "✅ $instance_name stopped gracefully"
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo "⚠️  Force killing $instance_name..."
                kill -9 $pid
                sleep 1
            fi
        else
            echo "ℹ️  $instance_name was not running"
        fi
        
        rm -f "$pid_file"
    fi
}

# Stop all instances by PID files
if [ -d "pids" ]; then
    for pid_file in pids/*.pid; do
        if [ -f "$pid_file" ]; then
            stop_instance "$pid_file"
        fi
    done
fi

# Also kill any remaining python processes running main.py
echo "🔍 Checking for remaining RAG processes..."
remaining=$(ps aux | grep "python3 main.py" | grep -v grep | awk '{print $2}')

if [ ! -z "$remaining" ]; then
    echo "⚠️  Found remaining processes, cleaning up..."
    echo "$remaining" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Clean up PID directory
rm -rf pids/

echo ""
echo "✅ All RAG instances have been stopped"
echo "🔍 Verifying no processes remain..."

# Final verification
remaining_check=$(ps aux | grep "python3 main.py" | grep -v grep)
if [ -z "$remaining_check" ]; then
    echo "✅ Clean shutdown confirmed - no RAG processes running"
else
    echo "⚠️  Warning: Some processes may still be running:"
    echo "$remaining_check"
fi
