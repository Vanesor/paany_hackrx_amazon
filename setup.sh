#!/bin/bash

# AWS Instance Setup Script for Redis-Enhanced RAG System
# This script sets up the entire environment on a fresh AWS EC2 instance

set -e  # Exit on any error

echo "🚀 Starting AWS Instance Setup for Redis-Enhanced RAG System"
echo "============================================================"

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install required system packages
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    docker.io \
    docker-compose \
    curl \
    wget \
    git \
    htop \
    vim \
    unzip \
    build-essential \
    python3 \
    python3-pip

# Start and enable Docker
echo "🐳 Setting up Docker..."
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install Docker Compose (latest version)
echo "📋 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create application directory
echo "📁 Setting up application directory..."
APP_DIR="/home/ubuntu/rag-system"
mkdir -p $APP_DIR
cd $APP_DIR

# Copy application files (assuming they're uploaded to the instance)
echo "📋 Application files should be in the current directory"
echo "Current directory contents:"
ls -la

# Set up environment file
echo "🔐 Setting up environment configuration..."
# if [ ! -f .env ]; then
#     cp .env.example .env
#     echo "⚠️  IMPORTANT: Edit .env file and add your GOOGLE_API_KEY"
#     echo "   Run: nano .env"
# fi

# Create required directories
echo "📂 Creating application directories..."
mkdir -p cache downloads auto_downloads

# Build and start services
echo "🏗️  Building and starting services..."
# sudo docker-compose build
sudo docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Test health endpoints
echo "🏥 Testing health endpoints..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ RAG System health check passed"
else
    echo "❌ RAG System health check failed"
    echo "Check logs with: sudo docker-compose logs rag-system"
fi

# Check Redis
if sudo docker exec rag-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis health check passed"
else
    echo "❌ Redis health check failed"
    echo "Check logs with: sudo docker-compose logs redis"
fi

# Display status
echo ""
echo "🎉 Setup Complete!"
echo "==================="
echo "RAG System URL: http://$(curl -s http://checkip.amazonaws.com):8000"
echo "Health Check: http://$(curl -s http://checkip.amazonaws.com):8000/health"
echo "API Health: http://$(curl -s http://checkip.amazonaws.com):8000/api/health"
echo ""
echo "Management Commands:"
echo "  Start services:  sudo docker-compose up -d"
echo "  Stop services:   sudo docker-compose down"
echo "  View logs:       sudo docker-compose logs -f"
echo "  Restart:         sudo docker-compose restart"
echo ""
echo "⚠️  Don't forget to:"
echo "  1. Edit .env file with your GOOGLE_API_KEY"
echo "  2. Configure AWS Security Groups to allow port 8000"
echo "  3. Consider setting up SSL/TLS for production"

# Show next steps
echo ""
echo "Next Steps:"
echo "1. Edit environment: nano .env"
echo "2. Restart services: sudo docker-compose restart"
echo "3. Test API: curl http://localhost:8000/api/v1/hackrx/run -X POST -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"documents\":[\"test doc\"]}'"
