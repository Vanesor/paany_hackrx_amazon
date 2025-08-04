# Test the API endpoints
import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your AWS instance IP
API_TOKEN = "6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca"

def test_health():
    """Test basic health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_api_health():
    """Test comprehensive API health"""
    print("\n🔍 Testing comprehensive API health...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def test_main_api():
    """Test main RAG API endpoint"""
    print("\n🚀 Testing main RAG API...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    payload = {
        "query": "What is artificial intelligence and how does it work?",
        "documents": [
            "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior.",
            "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
            "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data."
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/hackrx/run",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Response:")
            print(f"Answer: {result.get('answer', 'No answer')[:200]}...")
            print(f"Sources: {len(result.get('sources', []))} documents")
            print(f"Processing time: {result.get('processing_time_seconds', 0):.2f}s")
            return True
        else:
            print(f"❌ API Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Main API test failed: {e}")
        return False

def test_redis_status():
    """Test Redis connection status"""
    print("\n📊 Testing Redis status...")
    try:
        response = requests.get(f"{BASE_URL}/redis-status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Redis status check failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Running API Tests for Redis-Enhanced RAG System")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health),
        ("API Health Check", test_api_health),
        ("Redis Status", test_redis_status),
        ("Main RAG API", test_main_api)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔧 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print(f"{'✅ PASSED' if result else '❌ FAILED'}")
    
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs and configuration.")

if __name__ == "__main__":
    run_all_tests()
