#!/bin/bash

# Quick Performance Benchmark Script
# This script runs progressively increasing loads to find your optimal concurrency

echo "🎯 RAG System Performance Benchmark"
echo "===================================="
echo "This will test 1, 2, 3, and 5 concurrent requests to find optimal performance"
echo ""

# Function to run benchmark with N concurrent requests
run_benchmark() {
    local num_requests=$1
    local test_name="benchmark_${num_requests}"
    
    echo "📊 Testing $num_requests concurrent request(s)..."
    
    # Start timestamp
    local start_time=$(date +%s)
    
    # Launch concurrent requests
    for i in $(seq 1 $num_requests); do
        curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
          -H "Content-Type: application/json" \
          -H "Accept: application/json" \
          -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
          -s --max-time 120 \
          -d '{
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": ["What is the grace period for premium payment?"]
          }' > "${test_name}_${i}.json" 2>&1 &
    done
    
    # Wait for all requests to complete
    wait
    
    # End timestamp
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Count successful responses
    local success_count=0
    for i in $(seq 1 $num_requests); do
        if [ -f "${test_name}_${i}.json" ] && jq -e '.answers' "${test_name}_${i}.json" > /dev/null 2>&1; then
            ((success_count++))
        fi
    done
    
    # Calculate metrics
    local success_rate=$((success_count * 100 / num_requests))
    local avg_time_per_request=$((duration > 0 ? duration / num_requests : 0))
    local requests_per_second=$(( num_requests > 0 && duration > 0 ? (num_requests * 60) / duration : 0 ))
    
    echo "  ⏱️  Duration: ${duration}s"
    echo "  ✅ Success: ${success_count}/${num_requests} (${success_rate}%)"
    echo "  📈 Avg time per request: ${avg_time_per_request}s"
    echo "  🚀 Requests per minute: ${requests_per_second}"
    echo ""
    
    # Store results for summary
    echo "${num_requests},${duration},${success_count},${success_rate},${avg_time_per_request}" >> benchmark_results.csv
    
    # Clean up test files
    rm -f ${test_name}_*.json
    
    # Brief pause between tests
    sleep 3
}

# Initialize results file
echo "concurrent_requests,duration_seconds,successful_requests,success_rate_percent,avg_time_per_request" > benchmark_results.csv

echo "🚀 Starting performance benchmark..."
echo ""

# Run benchmarks with increasing concurrency
run_benchmark 1
run_benchmark 2  
run_benchmark 3
run_benchmark 5

echo "🏁 Benchmark completed!"
echo ""

# Display summary
echo "📊 BENCHMARK SUMMARY:"
echo "===================="
echo ""
echo "Requests | Duration | Success | Rate | Avg Time"
echo "---------|----------|---------|------|----------"

while IFS=, read -r requests duration success rate avg_time; do
    if [ "$requests" != "concurrent_requests" ]; then
        printf "%-8s | %-8s | %-7s | %3s%% | %4ss\n" "$requests" "${duration}s" "$success" "$rate" "$avg_time"
    fi
done < benchmark_results.csv

echo ""

# Find optimal concurrency
echo "💡 RECOMMENDATIONS:"
echo "==================="

# Read results and find best performance
best_rate=0
best_requests=1
optimal_requests=1

while IFS=, read -r requests duration success rate avg_time; do
    if [ "$requests" != "concurrent_requests" ] && [ "$rate" -gt "$best_rate" ]; then
        best_rate=$rate
        best_requests=$requests
    fi
    
    if [ "$requests" != "concurrent_requests" ] && [ "$rate" -ge 90 ] && [ "$requests" -gt "$optimal_requests" ]; then
        optimal_requests=$requests
    fi
done < benchmark_results.csv

echo "🎯 Best success rate: $best_requests concurrent requests ($best_rate% success)"
echo "⚡ Optimal concurrency: $optimal_requests concurrent requests (90%+ success)"
echo ""

if [ "$optimal_requests" -ge 5 ]; then
    echo "🎉 EXCELLENT: Your system can handle high concurrency!"
    echo "   Consider using 5+ concurrent requests in production"
elif [ "$optimal_requests" -ge 3 ]; then
    echo "👍 GOOD: Your system performs well with moderate concurrency"
    echo "   Recommended: 3-5 concurrent requests in production"
else
    echo "⚠️  LIMITED: Your system works best with low concurrency"
    echo "   Recommended: 1-2 concurrent requests, consider hardware upgrade"
fi

echo ""
echo "📁 Detailed results saved to: benchmark_results.csv"
echo "🔍 View results: cat benchmark_results.csv"
