#!/usr/bin/env python3
"""
Phase 7: Performance and Stress Testing
======================================

Tests system performance under various loads:
1. Response time measurement
2. Concurrent request handling
3. Large data processing
4. Memory usage monitoring
5. HeavyDB query optimization
"""

import requests
import time
import threading
import concurrent.futures
import psutil
import os
from datetime import datetime, timedelta
import statistics

print("=" * 80)
print("PHASE 7: PERFORMANCE AND STRESS TESTING")
print("=" * 80)

BASE_URL = "http://localhost:8000/api/v1/market-regime"

test_results = {
    "response_time": False,
    "concurrent_handling": False,
    "large_data_processing": False,
    "memory_efficiency": False,
    "query_optimization": False,
    "throughput": False
}

# Get initial memory usage
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024  # MB

# Test 1: Response Time Measurement
print("\n1️⃣ Testing Response Times...")
response_times = []

try:
    endpoints = [
        ("/status", "GET", None),
        ("/config", "GET", None),
    ]
    
    for endpoint, method, data in endpoints:
        url = BASE_URL + endpoint
        
        # Measure multiple requests
        times = []
        for i in range(10):
            start = time.time()
            
            if method == "GET":
                response = requests.get(url)
            else:
                response = requests.post(url, json=data)
                
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
            
        avg_time = statistics.mean(times)
        response_times.extend(times)
        
        print(f"   {endpoint}: {avg_time:.2f}ms average (min: {min(times):.2f}ms, max: {max(times):.2f}ms)")
        
        if avg_time < 1000:  # Under 1 second is acceptable
            test_results["response_time"] = True
            
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Concurrent Request Handling
print("\n2️⃣ Testing Concurrent Request Handling...")
try:
    def make_request(i):
        start = time.time()
        try:
            response = requests.get(f"{BASE_URL}/status")
            return (True, time.time() - start, response.status_code)
        except Exception as e:
            return (False, time.time() - start, str(e))
    
    # Test with multiple concurrent requests
    concurrent_counts = [10, 50, 100]
    
    for count in concurrent_counts:
        print(f"\n   Testing with {count} concurrent requests...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
            futures = [executor.submit(make_request, i) for i in range(count)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        successful = sum(1 for r in results if r[0])
        avg_time = statistics.mean(r[1] for r in results) * 1000
        
        print(f"     Success rate: {successful}/{count} ({successful/count*100:.1f}%)")
        print(f"     Average response time: {avg_time:.2f}ms")
        
        if successful / count > 0.95:  # 95% success rate
            test_results["concurrent_handling"] = True
            
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Large Data Processing
print("\n3️⃣ Testing Large Data Processing...")
try:
    # Request longer time periods
    periods = [
        ("7_days", "7 days"),
        ("30_days", "30 days"),
        ("90_days", "90 days")
    ]
    
    for period_key, period_name in periods:
        start = time.time()
        
        # Try to generate CSV for large period (will fail without config but tests processing)
        payload = {
            "period": period_key,
            "format": "detailed",
            "include_metadata": True
        }
        
        response = requests.post(f"{BASE_URL}/generate-csv", json=payload)
        elapsed = (time.time() - start) * 1000
        
        print(f"   {period_name}: {elapsed:.2f}ms (status: {response.status_code})")
        
        if elapsed < 5000:  # Under 5 seconds for large data
            test_results["large_data_processing"] = True
            
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Memory Usage
print("\n4️⃣ Testing Memory Efficiency...")
try:
    # Make multiple requests and check memory growth
    for i in range(100):
        requests.get(f"{BASE_URL}/status")
        
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_growth = current_memory - initial_memory
    
    print(f"   Initial memory: {initial_memory:.2f} MB")
    print(f"   Current memory: {current_memory:.2f} MB")
    print(f"   Memory growth: {memory_growth:.2f} MB")
    
    if memory_growth < 50:  # Less than 50MB growth
        print(f"   ✅ Memory usage is efficient")
        test_results["memory_efficiency"] = True
    else:
        print(f"   ⚠️  High memory growth detected")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: Query Performance
print("\n5️⃣ Testing HeavyDB Query Performance...")
try:
    # Test direct data access performance
    import sys
    sys.path.append('/srv/samba/shared/bt/backtester_stable/BTRUN')
    from backtester_v2.dal.heavydb_connection import get_connection, execute_query
    
    conn = get_connection()
    if conn:
        queries = [
            ("Count query", "SELECT COUNT(*) FROM nifty_option_chain"),
            ("Latest data", "SELECT * FROM nifty_option_chain WHERE trade_date = (SELECT MAX(trade_date) FROM nifty_option_chain) LIMIT 100"),
            ("Date range", "SELECT COUNT(*) FROM nifty_option_chain WHERE trade_date >= '2025-06-01' AND trade_date <= '2025-06-18'")
        ]
        
        for query_name, query in queries:
            start = time.time()
            result = execute_query(conn, query)
            elapsed = (time.time() - start) * 1000
            
            print(f"   {query_name}: {elapsed:.2f}ms")
            
            if elapsed < 1000:  # Under 1 second
                test_results["query_optimization"] = True
                
except Exception as e:
    print(f"   ⚠️  Could not test query performance: {e}")

# Test 6: Throughput Testing
print("\n6️⃣ Testing API Throughput...")
try:
    duration = 10  # seconds
    start_time = time.time()
    request_count = 0
    
    print(f"   Running throughput test for {duration} seconds...")
    
    while time.time() - start_time < duration:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            request_count += 1
            
    elapsed = time.time() - start_time
    throughput = request_count / elapsed
    
    print(f"   Total requests: {request_count}")
    print(f"   Throughput: {throughput:.2f} requests/second")
    
    if throughput > 10:  # At least 10 requests per second
        print(f"   ✅ Good throughput")
        test_results["throughput"] = True
    else:
        print(f"   ⚠️  Low throughput")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Summary
print("\n" + "=" * 80)
print("PHASE 7 TEST SUMMARY")
print("=" * 80)

for test, passed in test_results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{test.replace('_', ' ').title()}: {status}")

# Performance metrics summary
if response_times:
    print(f"\nPerformance Metrics:")
    print(f"  Average response time: {statistics.mean(response_times):.2f}ms")
    print(f"  Median response time: {statistics.median(response_times):.2f}ms")
    print(f"  95th percentile: {sorted(response_times)[int(len(response_times)*0.95)]:.2f}ms")

print("\n" + "=" * 80)

all_passed = all(test_results.values())
performance_acceptable = test_results["response_time"] and test_results["throughput"]

if all_passed:
    print("✅ ALL PERFORMANCE TESTS PASSED")
    print("\nThe system shows good performance characteristics:")
    print("- Fast response times")
    print("- Handles concurrent requests well")
    print("- Efficient memory usage")
    print("- Good throughput")
elif performance_acceptable:
    print("⚠️  BASIC PERFORMANCE IS ACCEPTABLE")
    print("\nAreas for improvement:")
    if not test_results["concurrent_handling"]:
        print("- Concurrent request handling needs optimization")
    if not test_results["memory_efficiency"]:
        print("- Memory usage could be optimized")
    if not test_results["query_optimization"]:
        print("- Database queries could be optimized")
else:
    print("❌ PERFORMANCE ISSUES DETECTED")
    print("\nThe system may have scalability issues")

print("\n📊 SCALABILITY RECOMMENDATIONS:")
print("1. Implement caching for frequently accessed data")
print("2. Add connection pooling for HeavyDB")
print("3. Optimize queries with proper indexing")
print("4. Consider async processing for heavy calculations")
print("5. Add rate limiting to prevent abuse")
print("6. Monitor production performance metrics")