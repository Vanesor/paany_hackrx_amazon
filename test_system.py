#!/usr/bin/env python3
"""
Test script for the Accuracy RAG System
"""

import asyncio
import aiohttp
import json
import time

# Test configuration
API_URL = "http://localhost:8000"
API_TOKEN = "your-secure-token-here"  # Update with your actual token
TEST_PDF_URL = "https://example.com/sample.pdf"  # Update with actual PDF URL

async def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ Health check passed: {data}")
                    return True
                else:
                    print(f"✗ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"✗ Health check error: {e}")
            return False

async def test_main_endpoint():
    """Test the main RAG endpoint"""
    print("Testing main RAG endpoint...")
    
    # Test payload
    payload = {
        "documents": TEST_PDF_URL,
        "questions": [
            "What is the main topic of this document?",
            "What are the key points mentioned?"
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            start_time = time.time()
            
            async with session.post(
                f"{API_URL}/api/v1/hackrx/run",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=35)  # 35 second timeout
            ) as response:
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ Main endpoint test passed in {processing_time:.2f}s")
                    print(f"  Answers received: {len(data.get('answers', []))}")
                    
                    for i, answer in enumerate(data.get('answers', []), 1):
                        print(f"  Answer {i}: {answer[:100]}...")
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"✗ Main endpoint failed: {response.status}")
                    print(f"  Error: {error_text}")
                    return False
                    
        except asyncio.TimeoutError:
            print("✗ Main endpoint timeout (>35s)")
            return False
        except Exception as e:
            print(f"✗ Main endpoint error: {e}")
            return False

async def test_system():
    """Run all system tests"""
    print("=" * 50)
    print("Accuracy RAG System Test Suite")
    print("=" * 50)
    
    # Test health check
    health_ok = await test_health_check()
    print()
    
    if not health_ok:
        print("⚠️  Health check failed - system may not be ready")
        return
    
    # Test main endpoint (only if you have a valid PDF URL and API token)
    if TEST_PDF_URL != "https://example.com/sample.pdf":
        main_ok = await test_main_endpoint()
    else:
        print("⚠️  Skipping main endpoint test - update TEST_PDF_URL and API_TOKEN")
        main_ok = True
    
    print()
    print("=" * 50)
    if health_ok and main_ok:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_system())
