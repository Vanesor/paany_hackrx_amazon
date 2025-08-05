#!/usr/bin/env python3
"""
FastAPI Reverse Proxy for PAANY RAG System
Deployed on Render to proxy requests to AWS Lightsail instance
"""

import os
import time
import logging
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import psutil
import threading

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - Update these with your AWS Lightsail details
AWS_LIGHTSAIL_URL = os.getenv('AWS_LIGHTSAIL_URL', 'http://YOUR_AWS_IP:8000')
PROXY_TOKEN = os.getenv('PROXY_TOKEN', '6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca')
TIMEOUT_SECONDS = int(os.getenv('TIMEOUT_SECONDS', '120'))

# Health monitoring configuration
HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '10'))  # seconds
ENABLE_CONTINUOUS_HEALTH_CHECK = os.getenv('ENABLE_CONTINUOUS_HEALTH_CHECK', 'true').lower() == 'true'

# Global health monitoring state
class HealthMonitor:
    def __init__(self):
        self.server_start_time = time.time()
        self.last_health_check = time.time()
        self.health_status = "starting"
        self.request_count = 0
        self.error_count = 0
        self.aws_connection_status = "unknown"
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.uptime_seconds = 0
        self.is_monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if not self.is_monitoring and ENABLE_CONTINUOUS_HEALTH_CHECK:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info(f"🔄 Started continuous health monitoring (every {HEALTH_CHECK_INTERVAL}s)")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("🛑 Stopped continuous health monitoring")
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.is_monitoring:
            try:
                self._update_system_stats()
                self.last_health_check = time.time()
                self.health_status = "healthy"
                logger.debug(f"✅ Health check: CPU {self.cpu_usage:.1f}%, Memory {self.memory_usage:.1f}%")
            except Exception as e:
                logger.error(f"❌ Health monitoring error: {e}")
                self.health_status = "degraded"
            
            time.sleep(HEALTH_CHECK_INTERVAL)
    
    def _update_system_stats(self):
        """Update system statistics"""
        try:
            # CPU and memory usage
            self.cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            self.memory_usage = memory.percent
            
            # Uptime
            self.uptime_seconds = time.time() - self.server_start_time
        except Exception as e:
            logger.warning(f"Failed to update system stats: {e}")
    
    def increment_request(self):
        """Increment request counter"""
        self.request_count += 1
    
    def increment_error(self):
        """Increment error counter"""
        self.error_count += 1
    
    def update_aws_status(self, status: str):
        """Update AWS connection status"""
        self.aws_connection_status = status
    
    def get_health_data(self) -> Dict[str, Any]:
        """Get current health data"""
        return {
            "server_status": self.health_status,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "uptime_formatted": self._format_uptime(self.uptime_seconds),
            "last_health_check": self.last_health_check,
            "last_health_check_ago": round(time.time() - self.last_health_check, 2),
            "system_stats": {
                "cpu_usage_percent": round(self.cpu_usage, 2),
                "memory_usage_percent": round(self.memory_usage, 2),
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": round((self.error_count / max(self.request_count, 1)) * 100, 2)
            },
            "aws_connection_status": self.aws_connection_status,
            "monitoring_enabled": self.is_monitoring,
            "monitoring_interval_seconds": HEALTH_CHECK_INTERVAL,
            "timestamp": time.time()
        }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {secs}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

# Global health monitor instance
health_monitor = HealthMonitor()

# FastAPI app
app = FastAPI(
    title="PAANY RAG Reverse Proxy",
    version="1.0.0",
    description="Reverse proxy for PAANY RAG system deployed on AWS Lightsail",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class ProxyHealthResponse(BaseModel):
    proxy_status: str
    aws_instance_status: str
    aws_instance_url: str
    timestamp: float
    response_time_ms: Optional[float] = None

class ServerHealthResponse(BaseModel):
    server_status: str
    uptime_seconds: float
    uptime_formatted: str
    last_health_check: float
    last_health_check_ago: float
    system_stats: Dict[str, Any]
    aws_connection_status: str
    monitoring_enabled: bool
    monitoring_interval_seconds: int
    timestamp: float

# Global HTTP session
http_session: Optional[aiohttp.ClientSession] = None

async def get_http_session() -> aiohttp.ClientSession:
    """Get or create HTTP session"""
    global http_session
    if http_session is None or http_session.closed:
        timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
        http_session = aiohttp.ClientSession(timeout=timeout)
    return http_session

async def forward_request(
    endpoint: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, str]] = None
) -> tuple[int, Dict[str, Any]]:
    """Forward request to AWS Lightsail instance"""
    health_monitor.increment_request()
    
    try:
        session = await get_http_session()
        url = f"{AWS_LIGHTSAIL_URL}{endpoint}"
        
        # Prepare headers
        request_headers = {
            "Content-Type": "application/json",
            "User-Agent": "PAANY-Reverse-Proxy/1.0"
        }
        if headers:
            request_headers.update(headers)
        
        logger.info(f"🔄 Forwarding {method} request to: {url}")
        
        async with session.request(
            method=method,
            url=url,
            headers=request_headers,
            json=json_data,
            params=params
        ) as response:
            try:
                response_data = await response.json()
            except:
                response_data = {"message": await response.text()}
            
            logger.info(f"✅ AWS response: {response.status}")
            
            # Update AWS connection status
            if response.status == 200:
                health_monitor.update_aws_status("healthy")
            else:
                health_monitor.update_aws_status("unhealthy")
            
            return response.status, response_data
            
    except aiohttp.ClientTimeout:
        health_monitor.increment_error()
        health_monitor.update_aws_status("timeout")
        logger.error(f"⏰ Timeout forwarding request to {url}")
        return 504, {"error": "Gateway timeout - AWS instance took too long to respond"}
    except aiohttp.ClientError as e:
        health_monitor.increment_error()
        health_monitor.update_aws_status("unreachable")
        logger.error(f"🔌 Connection error: {e}")
        return 502, {"error": f"Bad gateway - Cannot connect to AWS instance: {str(e)}"}
    except Exception as e:
        health_monitor.increment_error()
        health_monitor.update_aws_status("error")
        logger.error(f"❌ Unexpected error: {e}")
        return 500, {"error": f"Internal proxy error: {str(e)}"}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with proxy information"""
    return {
        "service": "PAANY RAG Reverse Proxy",
        "version": "1.0.0",
        "description": "Reverse proxy for PAANY RAG system on AWS Lightsail",
        "aws_target": AWS_LIGHTSAIL_URL,
        "server_health": health_monitor.get_health_data(),
        "endpoints": {
            "main_rag_api": "/api/v1/hackrx/run",
            "health_check": "/health",
            "api_health": "/api/health",
            "proxy_health": "/proxy/health",
            "server_health": "/server/health",
            "redis_status": "/redis-status"
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        }
    }

# Main RAG API endpoint - the primary endpoint you'll use
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_document_and_answer(
    request: QueryRequest,
    authorization: str = Header(None)
):
    """Main RAG API endpoint - forwards to AWS Lightsail instance"""
    try:
        headers = {}
        if authorization:
            headers["Authorization"] = authorization
        else:
            # Use default token if none provided
            headers["Authorization"] = f"Bearer {PROXY_TOKEN}"
        
        status_code, response_data = await forward_request(
            endpoint="/api/v1/hackrx/run",
            method="POST",
            headers=headers,
            json_data=request.dict()
        )
        
        if status_code == 200:
            return QueryResponse(**response_data)
        else:
            # NEW, MORE ROBUST CODE
            detail = response_data.get("detail", response_data.get("error", "An unknown error occurred on the RAG server."))
            raise HTTPException(status_code=status_code, detail=detail)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in main API endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check - forwards to AWS instance"""
    status_code, response_data = await forward_request("/health")
    
    if status_code == 200:
        return response_data
    else:
        raise HTTPException(status_code=status_code, detail="AWS instance health check failed")

@app.get("/api/health")
async def api_health_check():
    """Comprehensive API health - forwards to AWS instance"""
    status_code, response_data = await forward_request("/api/health")
    
    if status_code == 200:
        return response_data
    else:
        raise HTTPException(status_code=status_code, detail="AWS instance API health check failed")

@app.get("/proxy/health", response_model=ProxyHealthResponse)
async def proxy_health_check():
    """Check both proxy and AWS instance health"""
    start_time = time.time()
    
    try:
        # Test connection to AWS instance
        status_code, aws_response = await forward_request("/health")
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        
        if status_code == 200:
            aws_status = "healthy"
        else:
            aws_status = "unhealthy"
        
        return ProxyHealthResponse(
            proxy_status="healthy",
            aws_instance_status=aws_status,
            aws_instance_url=AWS_LIGHTSAIL_URL,
            timestamp=time.time(),
            response_time_ms=response_time_ms
        )
        
    except Exception as e:
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f"❌ Proxy health check failed: {e}")
        
        return ProxyHealthResponse(
            proxy_status="healthy",
            aws_instance_status="unreachable",
            aws_instance_url=AWS_LIGHTSAIL_URL,
            timestamp=time.time(),
            response_time_ms=response_time_ms
        )

@app.get("/server/health", response_model=ServerHealthResponse)
async def server_health_check():
    """Get detailed server health information with continuous monitoring stats"""
    try:
        # Get current health data from monitor
        health_data = health_monitor.get_health_data()
        
        return ServerHealthResponse(**health_data)
        
    except Exception as e:
        logger.error(f"❌ Server health check failed: {e}")
        # Return basic health data even if monitoring fails
        return ServerHealthResponse(
            server_status="error",
            uptime_seconds=time.time() - health_monitor.server_start_time,
            uptime_formatted="unknown",
            last_health_check=0,
            last_health_check_ago=999999,
            system_stats={
                "cpu_usage_percent": 0,
                "memory_usage_percent": 0,
                "request_count": health_monitor.request_count,
                "error_count": health_monitor.error_count,
                "error_rate": 0
            },
            aws_connection_status="unknown",
            monitoring_enabled=health_monitor.is_monitoring,
            monitoring_interval_seconds=HEALTH_CHECK_INTERVAL,
            timestamp=time.time()
        )

# Redis status endpoint
@app.get("/redis-status")
async def redis_status(authorization: str = Header(None)):
    """Get Redis status from AWS instance"""
    headers = {}
    if authorization:
        headers["Authorization"] = authorization
    else:
        headers["Authorization"] = f"Bearer {PROXY_TOKEN}"
    
    status_code, response_data = await forward_request(
        endpoint="/redis-status",
        headers=headers
    )
    
    if status_code == 200:
        return response_data
    else:
        raise HTTPException(status_code=status_code, detail="Failed to get Redis status")

# Performance stats endpoint
@app.get("/performance/stats")
async def performance_stats(authorization: str = Header(None)):
    """Get performance statistics from AWS instance"""
    headers = {}
    if authorization:
        headers["Authorization"] = authorization
    else:
        headers["Authorization"] = f"Bearer {PROXY_TOKEN}"
    
    status_code, response_data = await forward_request(
        endpoint="/performance/stats",
        headers=headers
    )
    
    if status_code == 200:
        return response_data
    else:
        raise HTTPException(status_code=status_code, detail="Failed to get performance stats")

# System info endpoint
@app.get("/performance/system-info")
async def system_info(authorization: str = Header(None)):
    """Get system information from AWS instance"""
    headers = {}
    if authorization:
        headers["Authorization"] = authorization
    else:
        headers["Authorization"] = f"Bearer {PROXY_TOKEN}"
    
    status_code, response_data = await forward_request(
        endpoint="/performance/system-info",
        headers=headers
    )
    
    if status_code == 200:
        return response_data
    else:
        raise HTTPException(status_code=status_code, detail="Failed to get system info")

# Debug endpoints
@app.post("/debug/document-structure")
async def debug_document_structure(
    request: QueryRequest,
    authorization: str = Header(None)
):
    """Debug document structure - forwards to AWS instance"""
    headers = {}
    if authorization:
        headers["Authorization"] = authorization
    else:
        headers["Authorization"] = f"Bearer {PROXY_TOKEN}"
    
    status_code, response_data = await forward_request(
        endpoint="/debug/document-structure",
        method="POST",
        headers=headers,
        json_data=request.dict()
    )
    
    if status_code == 200:
        return response_data
    else:
        raise HTTPException(status_code=status_code, detail="Debug request failed")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize health monitoring on startup"""
    logger.info("🚀 Starting PAANY RAG Reverse Proxy...")
    health_monitor.start_monitoring()
    logger.info("✅ Server startup complete")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("🔄 Shutting down PAANY RAG Reverse Proxy...")
    
    # Stop health monitoring
    health_monitor.stop_monitoring()
    
    # Close HTTP session
    global http_session
    if http_session and not http_session.closed:
        await http_session.close()
    
    logger.info("🔄 Proxy server shutdown complete")

# Main execution
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Render uses PORT env variable
    
    logger.info(f"🚀 Starting PAANY RAG Reverse Proxy on port {port}")
    logger.info(f"🎯 Target AWS instance: {AWS_LIGHTSAIL_URL}")
    logger.info(f"🔄 Health monitoring: {'Enabled' if ENABLE_CONTINUOUS_HEALTH_CHECK else 'Disabled'}")
    if ENABLE_CONTINUOUS_HEALTH_CHECK:
        logger.info(f"⏱️  Health check interval: {HEALTH_CHECK_INTERVAL} seconds")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )