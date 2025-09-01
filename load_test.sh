#!/bin/bash

# Load test configuration
URL="https://mannequin-segmenter-new-234382015820.europe-west4.run.app/infer"
IMAGE_URL="https://media.remix.eu/files/12-2025/Majka-bluza-Tommy-Hilfiger-131315547b.jpg"
NUM_REQUESTS=50
TIMEOUT=300

echo "🚀 Starting load test with $NUM_REQUESTS concurrent requests"
echo "📸 Image: $IMAGE_URL"
echo "🎯 Target: $URL"
echo "⏱️  Timeout: ${TIMEOUT}s per request"
echo "=================================="

# Create results directory
mkdir -p load_test_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="load_test_results/load_test_$TIMESTAMP.log"

# Function to make a single request
make_request() {
    local request_id=$1
    local start_time=$(date +%s.%3N)
    
    local response=$(curl -s -X POST "$URL" \
        -H "Content-Type: application/json" \
        -d "{\"image_url\": \"$IMAGE_URL\", \"upload_gcs\": true}" \
        --max-time $TIMEOUT \
        -w "%{http_code}|%{time_total}" \
        2>/dev/null)
    
    local end_time=$(date +%s.%3N)
    local wall_time=$(echo "$end_time - $start_time" | bc -l)
    
    # Parse response
    local http_code=$(echo "$response" | tail -c 10 | cut -d'|' -f1)
    local curl_time=$(echo "$response" | tail -c 10 | cut -d'|' -f2)
    local body=$(echo "$response" | head -c -10)
    
    # Log result
    echo "Request_$request_id|$http_code|$curl_time|$wall_time|$(date '+%H:%M:%S')" >> "$RESULTS_FILE"
    
    # Console output
    if [[ "$http_code" == "200" ]]; then
        echo "✅ Request $request_id: ${curl_time}s (HTTP $http_code)"
    else
        echo "❌ Request $request_id: ${curl_time}s (HTTP $http_code)"
    fi
}

# Export function for parallel execution
export -f make_request
export URL IMAGE_URL TIMEOUT RESULTS_FILE

# Create header for results file
echo "Request_ID|HTTP_Code|Response_Time|Wall_Time|Timestamp" > "$RESULTS_FILE"

# Start load test - launch all requests in parallel
echo "🔥 Launching $NUM_REQUESTS concurrent requests..."
seq 1 $NUM_REQUESTS | xargs -n 1 -P $NUM_REQUESTS -I {} bash -c 'make_request {}'

echo "=================================="
echo "📊 Load test completed! Analyzing results..."

# Analyze results
if [[ -f "$RESULTS_FILE" ]]; then
    echo ""
    echo "📈 PERFORMANCE ANALYSIS:"
    echo "========================"
    
    # Count success/failure
    local total_requests=$(tail -n +2 "$RESULTS_FILE" | wc -l)
    local success_count=$(tail -n +2 "$RESULTS_FILE" | grep "|200|" | wc -l)
    local error_count=$((total_requests - success_count))
    
    echo "📊 Total requests: $total_requests"
    echo "✅ Successful: $success_count"
    echo "❌ Failed: $error_count"
    echo "📈 Success rate: $(echo "scale=1; $success_count * 100 / $total_requests" | bc -l)%"
    
    if [[ $success_count -gt 0 ]]; then
        echo ""
        echo "⏱️  RESPONSE TIME ANALYSIS (successful requests only):"
        echo "======================================================"
        
        # Extract response times from successful requests
        tail -n +2 "$RESULTS_FILE" | grep "|200|" | cut -d'|' -f3 | sort -n > /tmp/response_times.txt
        
        # Calculate statistics
        local min_time=$(head -n1 /tmp/response_times.txt)
        local max_time=$(tail -n1 /tmp/response_times.txt)
        local median_time=$(tail -n +2 "$RESULTS_FILE" | grep "|200|" | cut -d'|' -f3 | sort -n | awk 'NR==int(NR/2)+1')
        local avg_time=$(tail -n +2 "$RESULTS_FILE" | grep "|200|" | cut -d'|' -f3 | awk '{sum+=$1} END {print sum/NR}')
        
        echo "🚀 Fastest response: ${min_time}s"
        echo "�� Slowest response: ${max_time}s"
        echo "📊 Average response: ${avg_time}s"
        echo "📊 Median response: ${median_time}s"
        
        # Percentiles
        local p95=$(tail -n +2 "$RESULTS_FILE" | grep "|200|" | cut -d'|' -f3 | sort -n | awk 'NR==int(0.95*NR)+1')
        local p99=$(tail -n +2 "$RESULTS_FILE" | grep "|200|" | cut -d'|' -f3 | sort -n | awk 'NR==int(0.99*NR)+1')
        
        echo "📊 95th percentile: ${p95}s"
        echo "📊 99th percentile: ${p99}s"
        
        # Response time distribution
        echo ""
        echo "📊 RESPONSE TIME DISTRIBUTION:"
        echo "=============================="
        echo "< 10s:  $(tail -n +2 "$RESULTS_FILE" | grep "|200|" | cut -d'|' -f3 | awk '$1<10' | wc -l) requests"
        echo "10-30s: $(tail -n +2 "$RESULTS_FILE" | grep "|200|" | cut -d'|' -f3 | awk '$1>=10 && $1<30' | wc -l) requests"
        echo "30-60s: $(tail -n +2 "$RESULTS_FILE" | grep "|200|" | cut -d'|' -f3 | awk '$1>=30 && $1<60' | wc -l) requests"
        echo "> 60s:  $(tail -n +2 "$RESULTS_FILE" | grep "|200|" | cut -d'|' -f3 | awk '$1>=60' | wc -l) requests"
        
        # Cleanup
        rm -f /tmp/response_times.txt
    fi
    
    echo ""
    echo "💾 Detailed results saved to: $RESULTS_FILE"
    echo "📄 First 10 results:"
    head -n 11 "$RESULTS_FILE" | column -t -s'|'
else
    echo "❌ Results file not found!"
fi
