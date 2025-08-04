#!/bin/bash

# Start the Redis-Enhanced RAG System

echo "🚀 Starting Redis-Enhanced RAG System..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create it from .env.example"
    echo "   cp .env.example .env"
    echo "   nano .env  # Add your GOOGLE_API_KEY"
    exit 1
fi

# Check if GOOGLE_API_KEY is set
if ! grep -q "GOOGLE_API_KEY=.*[^=]" .env; then
    echo "❌ GOOGLE_API_KEY not set in .env file"
    echo "   Please edit .env and add your Google API key"
    exit 1
fi

# Create directories if they don't exist
mkdir -p cache downloads auto_downloads

# Start services
echo "🐳 Starting Docker services..."
sudo docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to initialize..."
sleep 15

# Check service health
echo "🏥 Checking service health..."

# Check Redis
if sudo docker exec rag-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis health check failed"
    sudo docker-compose logs redis
fi

# Check RAG System
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ RAG System is running"
else
    echo "❌ RAG System health check failed"
    echo "Checking logs..."
    sudo docker-compose logs rag-system
fi

# Display status
echo ""
echo "🎉 Services Status:"
echo "=================="
sudo docker-compose ps

echo ""
echo "🌐 Access URLs:"
echo "==============="
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com)
echo "RAG System: http://$PUBLIC_IP:8000"
echo "Health Check: http://$PUBLIC_IP:8000/health"
echo "API Health: http://$PUBLIC_IP:8000/api/health"
echo "Main API: http://$PUBLIC_IP:8000/api/v1/hackrx/run"

echo ""
echo "📋 Useful Commands:"
echo "=================="
echo "View logs: sudo docker-compose logs -f"
echo "Stop services: sudo docker-compose down"
echo "Restart: sudo docker-compose restart"
