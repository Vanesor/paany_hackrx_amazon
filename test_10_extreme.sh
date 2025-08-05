#!/bin/bash

# EXTREME Test - 10 Concurrent Requests (WILL CAUSE QUEUEING)
# This tests what happens when you exceed your system limits

echo "💥 EXTREME TEST: 10 concurrent requests (BEYOND LIMITS)"
echo "⚠️  WARNING: This WILL cause significant queueing and slower responses"
echo "📊 Expected: Heavy queueing, some timeouts possible, high memory usage"
echo "🎯 Purpose: Test your semaphore limits and queue behavior"
echo ""

read -p "This will stress your system heavily. Continue? (y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Test cancelled."
    exit 0
fi

# Function to make a lightweight request
make_extreme_request() {
    local request_id=$1
    
    echo "🔄 Starting extreme request $request_id..."
    
    # Use only 1 question per request to minimize memory per request
    if [ $((request_id % 2)) -eq 1 ]; then
        curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
          -H "Content-Type: application/json" \
          -H "Accept: application/json" \
          -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
          --max-time 180 \
          -d '{
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": ["What is the grace period for premium payment?"]
          }' > "extreme_response_${request_id}.json" 2>&1
    else
        curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
          -H "Content-Type: application/json" \
          -H "Accept: application/json" \
          -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
          --max-time 180 \
          -d '{
            "documents": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
            "questions": ["What are Newton'\''s three laws of motion?"]
          }' > "extreme_response_${request_id}.json" 2>&1
    fi
    
    echo "✅ Extreme request $request_id completed (or timed out)"
}

# Start timestamp
start_time=$(date +%s)
echo "⏰ Start time: $(date)"
echo "🚀 Launching 10 concurrent requests..."

# Launch 10 concurrent requests
for i in {1..10}; do
    make_extreme_request "$i" &
    sleep 0.1  # Small delay to prevent overwhelming the system instantly
done

echo "⏳ Waiting for all requests to complete (max 3 minutes per request)..."

# Wait for all requests to complete
wait

# End timestamp
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "🏁 All extreme test requests completed!"
echo "⏱️  Total duration: ${duration} seconds"

# Analyze results
echo ""
echo "📋 EXTREME TEST RESULTS:"
success_count=0
timeout_count=0
error_count=0

for i in {1..10}; do
    if [ -f "extreme_response_${i}.json" ]; then
        echo "Request $i:"
        if jq -e '.answers' "extreme_response_${i}.json" > /dev/null 2>&1; then
            echo "  ✅ Success"
            ((success_count++))
        elif grep -q "timeout" "extreme_response_${i}.json" 2>/dev/null; then
            echo "  ⏰ Timeout"
            ((timeout_count++))
        else
            echo "  ❌ Error"
            ((error_count++))
        fi
    else
        echo "Request $i: ❌ No response file"
        ((error_count++))
    fi
done

echo ""
echo "📊 EXTREME TEST SUMMARY:"
echo "  Success rate: $success_count/10 ($(( success_count * 10 ))%)"
echo "  Timeout rate: $timeout_count/10 ($(( timeout_count * 10 ))%)"
echo "  Error rate: $error_count/10 ($(( error_count * 10 ))%)"
echo "  Average time: $((duration / 10)) seconds per request"
echo "  Total duration: ${duration} seconds"

echo ""
if [ $success_count -ge 8 ]; then
    echo "🎉 AMAZING: Your system handled extreme load very well!"
elif [ $success_count -ge 6 ]; then
    echo "👍 GOOD: System degraded gracefully under extreme load"
elif [ $success_count -ge 4 ]; then
    echo "⚠️  MODERATE: Some failures under extreme load (expected)"
else
    echo "❌ POOR: System struggled with extreme load (consider optimizations)"
fi

echo ""
echo "💡 RECOMMENDATIONS based on results:"
if [ $success_count -ge 8 ]; then
    echo "  - Your system can handle more than expected! Consider increasing limits slightly"
elif [ $success_count -ge 6 ]; then
    echo "  - Current configuration is well-tuned for your hardware"
else
    echo "  - Stick to 3-5 concurrent requests for optimal performance"
    echo "  - Consider upgrading to 4GB RAM for better concurrent handling"
fi

echo ""
echo "🧹 Cleanup extreme test files:"
echo "rm extreme_response_*.json"
