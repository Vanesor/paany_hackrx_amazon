#!/bin/bash

# Stress Test - 5 Concurrent Requests (MAXIMUM RECOMMENDED)
# This tests the maximum recommended concurrent load for your 2GB Lightsail

echo "🔥 STRESS TEST: 5 concurrent requests (MAXIMUM for 2GB RAM)"
echo "⚠️  This pushes your system to the limit"
echo "📊 Expected: Some queueing, but all should succeed"
echo ""

# Function to make a request with smaller question sets to reduce memory pressure
make_stress_request() {
    local request_id=$1
    
    echo "🔄 Starting stress request $request_id..."
    
    # Alternate between documents to test different scenarios
    if [ $((request_id % 2)) -eq 1 ]; then
        # Policy document with 2 questions (lighter load)
        curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
          -H "Content-Type: application/json" \
          -H "Accept: application/json" \
          -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
          -d '{
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": [
              "What is the grace period for premium payment?",
              "What is the waiting period for pre-existing diseases?"
            ]
          }' > "stress_response_${request_id}.json" 2>&1
    else
        # Newton document with 2 questions (lighter load)
        curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
          -H "Content-Type: application/json" \
          -H "Accept: application/json" \
          -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
          -d '{
            "documents": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
            "questions": [
              "What are Newton'\''s three laws of motion?",
              "How does Newton explain gravitational force?"
            ]
          }' > "stress_response_${request_id}.json" 2>&1
    fi
    
    echo "✅ Stress request $request_id completed"
}

# Start timestamp
start_time=$(date +%s)
echo "⏰ Start time: $(date)"

# Launch 5 concurrent requests
for i in {1..5}; do
    make_stress_request "$i" &
done

# Wait for all requests to complete
wait

# End timestamp
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "🏁 All stress test requests completed!"
echo "⏱️  Total duration: ${duration} seconds"
echo "📊 Check response files: stress_response_1.json to stress_response_5.json"

# Display summaries
echo ""
echo "📋 Stress test results:"
success_count=0
for i in {1..5}; do
    if [ -f "stress_response_${i}.json" ]; then
        echo "Request $i:"
        if jq -e '.answers' "stress_response_${i}.json" > /dev/null 2>&1; then
            echo "  ✅ Success - $(jq '.answers | length' stress_response_${i}.json) answers"
            ((success_count++))
        else
            echo "  ❌ Error - $(head -n 1 stress_response_${i}.json)"
        fi
    fi
done

echo ""
echo "📊 STRESS TEST SUMMARY:"
echo "  Success rate: $success_count/5 ($(( success_count * 20 ))%)"
echo "  Average time per request: $((duration / 5)) seconds"

if [ $success_count -eq 5 ]; then
    echo "  🎉 EXCELLENT: All requests succeeded under stress!"
elif [ $success_count -ge 4 ]; then
    echo "  👍 GOOD: Most requests succeeded"
else
    echo "  ⚠️  CONCERNING: Many requests failed - consider reducing concurrent load"
fi

echo ""
echo "🔍 Check system status:"
echo "curl -H 'Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca' http://localhost:8000/concurrency/status"
