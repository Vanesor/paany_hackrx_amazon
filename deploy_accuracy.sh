#!/bin/bash

# Quick deployment script for accuracy-optimized RAG system
# Run this after setup.sh to deploy the Redis-free accuracy system

set -e

echo "🎯 Deploying Accuracy-Optimized RAG System (No Redis)"
echo "====================================================="

# Stop any existing services
echo "🛑 Stopping existing services..."
sudo docker-compose down 2>/dev/null || true

# Clean up old containers and images
echo "🧹 Cleaning up old containers..."
sudo docker system prune -f

# Create necessary directories
echo "📁 Creating directories for BGE-base models..."
mkdir -p cache downloads models auto_downloads
chmod 755 cache downloads models auto_downloads

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Creating .env from template..."
    cp .env.example .env
    echo "🔑 IMPORTANT: Edit .env file and add your GOOGLE_API_KEY"
    echo "   Run: nano .env"
    echo "   Press Ctrl+X, then Y, then Enter to save"
fi

# Build and start the accuracy system
echo "🏗️  Building accuracy-optimized system..."
sudo docker-compose build --no-cache

echo "🚀 Starting accuracy system..."
sudo docker-compose up -d

# Wait for BGE-base models to load
echo "⏳ Waiting for BGE-base models to load (this takes 2-3 minutes)..."
for i in {1..36}; do
    echo -n "."
    sleep 5
done
echo ""

# Test the system
echo "🧪 Testing accuracy system..."
sleep 10

if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Accuracy-optimized RAG System is running!"
    
    # Show system info
    echo ""
    echo "📊 System Information:"
    echo "Container: $(sudo docker ps --filter name=rag-accuracy-system --format 'table {{.Names}}\t{{.Status}}')"
    echo "Memory: $(sudo docker stats rag-accuracy-system --no-stream --format 'table {{.MemUsage}}\t{{.CPUPerc}}')"
    
    echo ""
    echo "🌐 Access URLs:"
    echo "Health Check: http://localhost:8000/health"
    echo "API Docs: http://localhost:8000/docs"
    
    echo ""
    echo "🔧 Useful Commands:"
    echo "View logs: sudo docker-compose logs -f rag-accuracy-system"
    echo "Monitor: sudo docker stats rag-accuracy-system"
    echo "Restart: sudo docker-compose restart"
    
else
    echo "❌ System failed to start properly"
    echo "Check logs: sudo docker-compose logs rag-accuracy-system"
    exit 1
fi

echo ""
echo "🎯 Accuracy-optimized RAG system deployed successfully!"
echo "System is ready for high-quality document processing with BGE-base embeddings."
