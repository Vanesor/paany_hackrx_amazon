#!/bin/bash

# Stop the Redis-Enhanced RAG System

echo "🛑 Stopping Redis-Enhanced RAG System..."

# Stop and remove containers
sudo docker-compose down

echo "✅ Services stopped"

# Optional: Remove volumes (uncomment if you want to clear all data)
# echo "🗑️  Removing volumes..."
# sudo docker-compose down -v

echo ""
echo "📋 Service Status:"
echo "=================="
sudo docker-compose ps

echo ""
echo "To start again: ./start.sh"
