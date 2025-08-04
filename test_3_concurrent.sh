#!/bin/bash

# Concurrent Request Test - 3 Requests (OPTIMAL for 2GB RAM)
# This tests the optimal number of concurrent requests for your Lightsail setup

echo "🚀 Testing 3 concurrent requests (OPTIMAL for 2GB RAM)..."
echo "⚡ This should work smoothly with your current semaphore limits"
echo "📊 Expected: All requests succeed, queue time minimal"
echo ""

# Function to make a single request
make_request() {
    local request_id=$1
    local doc_type=$2
    
    echo "🔄 Starting request $request_id ($doc_type)..."
    
    if [ "$doc_type" == "policy" ]; then
        curl -X POST "https://reverse-proxy-tj2t.onrender.com/api/v1/hackrx/run" \
          -H "Content-Type: application/json" \
          -H "Accept: application/json" \
          -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
          -d '{
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": [
              "What is the grace period for premium payment?",
              "What is the waiting period for pre-existing diseases?",
              "Does this policy cover maternity expenses?"
            ]
          }' > "response_${request_id}.json" 2>&1
    else
        curl -X POST "https://reverse-proxy-tj2t.onrender.com/api/v1/hackrx/run" \
          -H "Content-Type: application/json" \
          -H "Accept: application/json" \
          -H "Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca" \
          -d '{
            "documents": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
            "questions": [
              "What are Newton'\''s three laws of motion?",
              "How does Newton explain gravitational force?",
              "What is the concept of centripetal force?"
            ]
          }' > "response_${request_id}.json" 2>&1
    fi
    
    echo "✅ Request $request_id completed"
}

# Start timestamp
start_time=$(date +%s)
echo "⏰ Start time: $(date)"

# Launch 3 concurrent requests
make_request "1" "policy" &
make_request "2" "newton" &
make_request "3" "policy" &

# Wait for all requests to complete
wait

# End timestamp
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "🏁 All requests completed!"
echo "⏱️  Total duration: ${duration} seconds"
echo "📊 Check response files: response_1.json, response_2.json, response_3.json"

# Display summaries
echo ""
echo "📋 Response summaries:"
for i in {1..3}; do
    if [ -f "response_${i}.json" ]; then
        echo "Response $i:"
        if jq -e '.answers' "response_${i}.json" > /dev/null 2>&1; then
            echo "  ✅ Success - $(jq '.answers | length' response_${i}.json) answers received"
        else
            echo "  ❌ Error - $(head -n 1 response_${i}.json)"
        fi
    fi
done

echo ""
echo "🔍 To check server concurrency status:"
echo "curl -H 'Authorization: Bearer 6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca' https://reverse-proxy-tj2t.onrender.com/concurrency/status"
