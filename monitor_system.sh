#!/bin/bash

# System Monitoring and Health Check Script
# Use this to monitor your system during load tests

echo "🔍 RAG System Health and Concurrency Monitor"
echo "============================================"

# Function to check system health
check_health() {
    echo "🏥 Health Check:"
    curl -s "http://localhost:8000/health" | jq '.'
    echo ""
}

# Function to check concurrency status
check_concurrency() {
    echo "⚡ Concurrency Status:"
    curl -s -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
         "http://localhost:8000/concurrency/status" | jq '.'
    echo ""
}

# Function to check performance stats
check_performance() {
    echo "📊 Performance Stats:"
    curl -s -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
         "http://localhost:8000/performance/stats" | jq '.performance_stats'
    echo ""
}

# Function to check Redis status
check_redis() {
    echo "🔴 Redis Status:"
    curl -s -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
         "http://localhost:8000/redis-status" | jq '.redis_connected, .cache_stats'
    echo ""
}

# Function to monitor system resources
check_system_resources() {
    echo "💻 System Resources:"
    echo "Memory usage:"
    free -h
    echo ""
    echo "CPU usage:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2 $3 $4 $5 $6 $7 $8}'
    echo ""
    echo "Docker containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
}

# Main monitoring function
monitor_system() {
    while true; do
        clear
        echo "🕐 $(date)"
        echo "RAG System Monitor - Press Ctrl+C to stop"
        echo "========================================="
        echo ""
        
        check_health
        check_concurrency
        check_system_resources
        
        echo "Refreshing in 5 seconds..."
        sleep 5
    done
}

# Check command line arguments
case "${1}" in
    "health")
        check_health
        ;;
    "concurrency")
        check_concurrency
        ;;
    "performance")
        check_performance
        ;;
    "redis")
        check_redis
        ;;
    "resources")
        check_system_resources
        ;;
    "monitor")
        monitor_system
        ;;
    *)
        echo "Usage: $0 {health|concurrency|performance|redis|resources|monitor}"
        echo ""
        echo "Commands:"
        echo "  health      - Check basic system health"
        echo "  concurrency - Check current concurrency status"
        echo "  performance - Check performance statistics"
        echo "  redis       - Check Redis connection and cache stats"
        echo "  resources   - Check system resources (memory, CPU, containers)"
        echo "  monitor     - Continuous monitoring (press Ctrl+C to stop)"
        echo ""
        echo "Examples:"
        echo "  $0 health"
        echo "  $0 monitor  # For real-time monitoring during load tests"
        ;;
esac
