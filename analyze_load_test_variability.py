#!/usr/bin/env python3
"""
Load Test Variability Analysis
Analyze the standard deviation and scaling patterns from 100 concurrent requests.
"""

import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def analyze_load_test_results(filename):
    """Analyze variability and scaling patterns from load test results."""
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    results = data["raw_results"]
    
    print("🔍 DETAILED VARIABILITY ANALYSIS")
    print("=" * 70)
    print(f"📅 Test: {data['timestamp']}")
    print(f"📊 Total Requests: {len(results)}")
    print(f"✅ Success Rate: {data['overall_stats']['success_rate']*100:.1f}%")
    print("=" * 70)
    
    # Extract response times
    response_times = [r["response_time_ms"] for r in results]
    response_times_sec = [t/1000 for t in response_times]
    
    # Statistical analysis
    mean_time = statistics.mean(response_times_sec)
    median_time = statistics.median(response_times_sec)
    stdev_time = statistics.stdev(response_times_sec)
    cv_percent = (stdev_time / mean_time) * 100
    
    # Percentile analysis
    sorted_times = sorted(response_times_sec)
    p10 = sorted_times[int(len(sorted_times) * 0.10)]
    p25 = sorted_times[int(len(sorted_times) * 0.25)]
    p75 = sorted_times[int(len(sorted_times) * 0.75)]
    p90 = sorted_times[int(len(sorted_times) * 0.90)]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]
    
    print(f"📊 RESPONSE TIME DISTRIBUTION:")
    print(f"   Mean: {mean_time:.1f}s (±{stdev_time:.1f}s)")
    print(f"   Median: {median_time:.1f}s")
    print(f"   Std Dev: {stdev_time:.1f}s ({cv_percent:.1f}% CV)")
    print(f"   Range: {min(response_times_sec):.1f}s - {max(response_times_sec):.1f}s")
    
    print(f"\n📈 PERCENTILE BREAKDOWN:")
    print(f"   10th percentile: {p10:.1f}s (fastest 10%)")
    print(f"   25th percentile: {p25:.1f}s (fastest 25%)")
    print(f"   50th percentile: {median_time:.1f}s (median)")
    print(f"   75th percentile: {p75:.1f}s (slowest 25%)")
    print(f"   90th percentile: {p90:.1f}s (slowest 10%)")
    print(f"   95th percentile: {p95:.1f}s (slowest 5%)")
    print(f"   99th percentile: {p99:.1f}s (slowest 1%)")
    
    # Performance tiers analysis
    fast_requests = [t for t in response_times_sec if t <= 10]
    medium_requests = [t for t in response_times_sec if 10 < t <= 25]
    slow_requests = [t for t in response_times_sec if t > 25]
    
    print(f"\n🎯 PERFORMANCE TIERS:")
    print(f"   🚀 Fast (≤10s): {len(fast_requests)} requests ({len(fast_requests)/len(results)*100:.1f}%)")
    if fast_requests:
        print(f"      Avg: {statistics.mean(fast_requests):.1f}s (±{statistics.stdev(fast_requests) if len(fast_requests)>1 else 0:.1f}s)")
    
    print(f"   ⚠️  Medium (10-25s): {len(medium_requests)} requests ({len(medium_requests)/len(results)*100:.1f}%)")
    if medium_requests:
        print(f"      Avg: {statistics.mean(medium_requests):.1f}s (±{statistics.stdev(medium_requests) if len(medium_requests)>1 else 0:.1f}s)")
    
    print(f"   🐌 Slow (>25s): {len(slow_requests)} requests ({len(slow_requests)/len(results)*100:.1f}%)")
    if slow_requests:
        print(f"      Avg: {statistics.mean(slow_requests):.1f}s (±{statistics.stdev(slow_requests) if len(slow_requests)>1 else 0:.1f}s)")
    
    # Scaling analysis
    print(f"\n🔄 AUTO-SCALING ANALYSIS:")
    
    # Estimate when instances became available
    total_duration = data["overall_stats"]["total_duration_seconds"]
    concurrent_capacity = len(fast_requests)  # Rough estimate
    
    print(f"   📊 Estimated warm instances: ~{min(len(fast_requests)//10, 3)}")
    print(f"   ⏱️  Scale-up duration: ~{total_duration:.1f}s")
    print(f"   🎯 Peak instances likely: ~{max(3, len(results)//15)}")
    
    # Cold start analysis
    if len(fast_requests) < 20:
        print(f"   ❄️  Cold start impact: HIGH (only {len(fast_requests)} fast responses)")
    elif len(fast_requests) < 50:
        print(f"   ❄️  Cold start impact: MEDIUM ({len(fast_requests)} fast responses)")
    else:
        print(f"   ❄️  Cold start impact: LOW ({len(fast_requests)} fast responses)")
    
    # Variability interpretation
    print(f"\n📉 VARIABILITY INTERPRETATION:")
    if cv_percent < 15:
        print(f"   ✅ LOW variability ({cv_percent:.1f}% CV) - Excellent consistency")
    elif cv_percent < 30:
        print(f"   ⚠️  MEDIUM variability ({cv_percent:.1f}% CV) - Some scaling delays")
    else:
        print(f"   ❌ HIGH variability ({cv_percent:.1f}% CV) - Significant scaling issues")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    if cv_percent > 30:
        print(f"   🔧 Increase min-instances (currently 3)")
        print(f"   🔧 Reduce concurrency per instance (currently 50)")
        print(f"   🔧 Consider pre-warming strategy")
    
    if len(slow_requests) > 20:
        print(f"   🔧 Optimize GPU memory management")
        print(f"   🔧 Consider request queuing/load balancing")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Single image baseline: ~2.7s (10.1% CV)")
    print(f"   100 concurrent result: ~{mean_time:.1f}s ({cv_percent:.1f}% CV)")
    print(f"   Performance degradation: {mean_time/2.7:.1f}x slower")
    print(f"   Variability increase: {cv_percent/10.1:.1f}x more variable")
    
    # Pipeline step breakdown analysis
    analyze_pipeline_steps(response_times_sec, cv_percent)
    
    # Current configuration assessment
    print(f"\n🎛️  CURRENT CLOUD RUN CONFIG ASSESSMENT:")
    print(f"   concurrency: 50 → ⚠️  REDUCE to 20-30 for better GPU utilization")
    print(f"   min-instances: 3 → 🔧 INCREASE to 5-8 for lower cold starts")
    print(f"   max-instances: 10 → ✅ ADEQUATE for 100 requests")

def analyze_pipeline_steps(response_times_sec, overall_cv):
    """Analyze individual pipeline steps based on baseline measurements and load test results."""
    
    print(f"\n🔍 PIPELINE STEP BREAKDOWN ANALYSIS")
    print("=" * 70)
    
    # Baseline from single image test (2.7s total)
    baseline_total = 2.7
    baseline_download_pct = 10.6  # 284.6ms / 2688.3ms
    baseline_inference_pct = 53.6  # ~1442ms / 2688.3ms  
    baseline_s3_upload_pct = 26.8  # ~721ms / 2688.3ms
    baseline_other_pct = 9.0      # ~240ms / 2688.3ms
    
    # Current load test averages
    current_mean = statistics.mean(response_times_sec)
    current_stdev = statistics.stdev(response_times_sec)
    
    print(f"📊 BASELINE (Single Image) vs LOAD TEST (100 Concurrent):")
    print(f"   Total Time: {baseline_total:.1f}s → {current_mean:.1f}s ({current_mean/baseline_total:.1f}x slower)")
    print(f"   Variability: 10.1% CV → {overall_cv:.1f}% CV ({overall_cv/10.1:.1f}x more variable)")
    
    # Estimate step performance under load
    # Download times likely stay similar (network bound)
    # Inference times increase significantly due to GPU contention
    # S3 upload may increase due to concurrent uploads
    # Other processing likely increases due to system load
    
    estimated_steps = []
    
    # 1. Image Download - stays relatively stable but network congestion
    download_base = baseline_total * (baseline_download_pct / 100)
    download_load = download_base * 1.5  # Modest increase due to network load
    download_stdev = download_load * 0.4  # High network variability
    estimated_steps.append(("📥 Image Download", download_load, download_stdev, "Network latency + congestion"))
    
    # 2. Model Inference - significant increase due to GPU queuing
    inference_base = baseline_total * (baseline_inference_pct / 100)
    inference_multiplier = 5.0  # GPU saturation causes major delays
    inference_load = inference_base * inference_multiplier
    inference_stdev = inference_load * 0.6  # High GPU queue variability
    estimated_steps.append(("🧠 Model Inference", inference_load, inference_stdev, "GPU saturation + queuing"))
    
    # 3. S3 Upload - moderate increase due to concurrent uploads
    s3_base = baseline_total * (baseline_s3_upload_pct / 100)
    s3_load = s3_base * 2.5  # Network + concurrent upload pressure
    s3_stdev = s3_load * 0.3  # Moderate network variability
    estimated_steps.append(("📤 S3 Upload", s3_load, s3_stdev, "Concurrent upload pressure"))
    
    # 4. Other Processing - increase due to system load
    other_base = baseline_total * (baseline_other_pct / 100)
    other_load = other_base * 3.0  # System resource contention
    other_stdev = other_load * 0.4  # System load variability
    estimated_steps.append(("🔧 Other Processing", other_load, other_stdev, "System resource contention"))
    
    print(f"\n📋 ESTIMATED STEP PERFORMANCE UNDER LOAD:")
    
    total_estimated = sum(step[1] for step in estimated_steps)
    
    for step_name, step_time, step_stdev, reason in estimated_steps:
        step_cv = (step_stdev / step_time) * 100 if step_time > 0 else 0
        step_pct = (step_time / total_estimated) * 100
        
        print(f"   {step_name}:")
        print(f"      Time: {step_time:.1f}s (±{step_stdev:.1f}s) [{step_cv:.1f}% CV]")
        print(f"      Share: {step_pct:.1f}% of total time")
        print(f"      Impact: {reason}")
        print()
    
    print(f"📊 STEP-BY-STEP VARIABILITY ANALYSIS:")
    
    # Sort by variability impact
    variability_impact = []
    for step_name, step_time, step_stdev, reason in estimated_steps:
        step_cv = (step_stdev / step_time) * 100 if step_time > 0 else 0
        step_pct = (step_time / total_estimated) * 100
        impact_score = step_cv * step_pct / 100  # CV weighted by time share
        variability_impact.append((step_name, step_cv, step_pct, impact_score))
    
    variability_impact.sort(key=lambda x: x[3], reverse=True)
    
    for i, (step_name, step_cv, step_pct, impact) in enumerate(variability_impact):
        if i == 0:
            print(f"   🔴 HIGHEST IMPACT: {step_name} ({step_cv:.1f}% CV, {step_pct:.1f}% time)")
        elif i == 1:
            print(f"   🟡 MEDIUM IMPACT: {step_name} ({step_cv:.1f}% CV, {step_pct:.1f}% time)")
        else:
            print(f"   🟢 LOW IMPACT: {step_name} ({step_cv:.1f}% CV, {step_pct:.1f}% time)")
    
    print(f"\n💡 STEP-SPECIFIC OPTIMIZATION RECOMMENDATIONS:")
    
    # Recommendations based on highest impact steps
    top_impact = variability_impact[0]
    
    if "Inference" in top_impact[0]:
        print(f"   🎯 PRIMARY FOCUS: GPU Management")
        print(f"      • Reduce concurrency per instance (50 → 20-30)")
        print(f"      • Implement GPU memory monitoring")
        print(f"      • Add request queuing with priority")
        print(f"      • Consider GPU instance pre-warming")
        
    elif "Download" in top_impact[0]:
        print(f"   🎯 PRIMARY FOCUS: Network Optimization")
        print(f"      • Implement image caching")
        print(f"      • Use CDN for frequent images")
        print(f"      • Optimize download timeout/retry")
        
    elif "S3" in top_impact[0]:
        print(f"   🎯 PRIMARY FOCUS: Upload Optimization")
        print(f"      • Implement S3 connection pooling")
        print(f"      • Use S3 Transfer Acceleration")
        print(f"      • Optimize image compression")
    
    print(f"\n🔬 BASELINE vs LOAD COMPARISON:")
    print(f"   Baseline total: {baseline_total:.1f}s (10.1% CV)")
    print(f"   Load test total: {current_mean:.1f}s ({overall_cv:.1f}% CV)")
    print(f"   Estimated total: {total_estimated:.1f}s")
    print(f"   Estimation accuracy: {abs(total_estimated - current_mean) / current_mean * 100:.1f}% error")

if __name__ == "__main__":
    # Analyze the most recent load test results
    filename = "load_test_100_results_20250724_112341.json"
    analyze_load_test_results(filename) 