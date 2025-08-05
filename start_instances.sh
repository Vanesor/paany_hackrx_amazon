#!/bin/bash

# RAG System Multi-Instance Deployment Script
# ===========================================
# This script starts multiple RAG instances for nginx load balancing

echo "🚀 Starting RAG System with Load Balancing..."

# Configuration
INSTANCES=2  # Number of instances to start
BASE_PORT=8000
SCRIPT_DIR="/home/vane/Downloads/paany_instance"

# Function to start a single instance
start_instance() {
    local port=$1
    local instance_id=$((port - BASE_PORT + 1))
    
    echo "⚡ Starting RAG Instance $instance_id on port $port..."
    
    cd "$SCRIPT_DIR"
    
    # Start instance in background with logging
    nohup python3 main.py $port > "logs/rag_instance_${instance_id}.log" 2>&1 &
    
    local pid=$!
    echo $pid > "pids/rag_instance_${instance_id}.pid"
    
    echo "✅ Instance $instance_id started (PID: $pid, Port: $port)"
    
    # Wait a moment for startup
    sleep 2
}

# Function to check if instance is running
check_instance() {
    local port=$1
    local instance_id=$((port - BASE_PORT + 1))
    
    echo "🔍 Checking Instance $instance_id (port $port)..."
    
    # Test HTTP connection
    if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "✅ Instance $instance_id is responding"
        return 0
    else
        echo "❌ Instance $instance_id is not responding"
        return 1
    fi
}

# Create necessary directories
mkdir -p logs pids

# Stop any existing instances
echo "🛑 Stopping any existing instances..."
if [ -f "stop_instances.sh" ]; then
    bash stop_instances.sh
fi

# Wait for ports to be released
sleep 3

# Start instances
echo "🔄 Starting $INSTANCES RAG instances..."
for i in $(seq 1 $INSTANCES); do
    port=$((BASE_PORT + i - 1))
    start_instance $port
done

# Wait for all instances to start up
echo "⏳ Waiting for instances to initialize..."
sleep 15

# Health check all instances
echo "🏥 Performing health checks..."
all_healthy=true
for i in $(seq 1 $INSTANCES); do
    port=$((BASE_PORT + i - 1))
    if ! check_instance $port; then
        all_healthy=false
    fi
done

if [ "$all_healthy" = true ]; then
    echo ""
    echo "🎉 All RAG instances are running successfully!"
    echo ""
    echo "📋 Instance Summary:"
    for i in $(seq 1 $INSTANCES); do
        port=$((BASE_PORT + i - 1))
        echo "   Instance $i: http://localhost:$port"
    done
    echo ""
    echo "🌐 Ready for nginx load balancing!"
    echo "   Configure nginx to proxy to: localhost:$BASE_PORT, localhost:$((BASE_PORT + 1))"
    echo ""
    echo "📝 To check logs: tail -f logs/rag_instance_*.log"
    echo "🛑 To stop all: bash stop_instances.sh"
else
    echo ""
    echo "⚠️  Some instances failed to start. Check logs in the logs/ directory."
    echo "🛑 Stopping all instances..."
    bash stop_instances.sh
    exit 1
fi
