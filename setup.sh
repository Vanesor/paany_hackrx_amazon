#!/bin/bash

# AWS Instance Setup Script for Accuracy-Optimized RAG System
# This script sets up the entire environment on a fresh AWS EC2 instance
# Optimized for BGE-base model without Redis dependency

set -e  # Exit on any error

echo "🎯 Starting AWS Instance Setup for Accuracy-Optimized RAG System"
echo "================================================================"

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install required system packages for accuracy system
echo "🔧 Installing system dependencies for BGE-base system..."
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
    python3-pip \
    python3-dev

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

# Create required directories for accuracy system
echo "📂 Creating application directories for BGE-base models..."
mkdir -p cache downloads models auto_downloads

# Build and start accuracy-optimized services
echo "🏗️  Building and starting accuracy-optimized services..."
# sudo docker-compose build
sudo docker-compose up --build -d

# Wait for BGE-base models to load (longer wait time)
echo "⏳ Waiting for BGE-base models to load (this may take 2-3 minutes)..."
sleep 120

# Test health endpoints
echo "🏥 Testing accuracy system health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Accuracy-optimized RAG System health check passed"
else
    echo "❌ RAG System health check failed"
    echo "Check logs with: sudo docker-compose logs rag-system"
fi

# Display status
echo ""
echo "🎉 Setup Complete!"
echo "==================="
echo "RAG System URL: http://$(curl -s http://checkip.amazonaws.com):8000"
echo "Accuracy System Health: http://$(curl -s http://checkip.amazonaws.com):8000/health"
echo "API Health: http://$(curl -s http://checkip.amazonaws.com):8000/api/health"
echo ""
echo "Management Commands:"
echo "  Start services:  sudo docker-compose up -d"
echo "  Stop services:   sudo docker-compose down"
echo "  View logs:       sudo docker-compose logs -f rag-accuracy-system"
echo "  Restart:         sudo docker-compose restart"
echo "  Monitor memory:  sudo docker stats rag-accuracy-system"
echo ""
echo "⚠️  Don't forget to:"
echo "  1. Edit .env file with your GOOGLE_API_KEY"
echo "  2. Configure AWS Security Groups to allow port 8000"
echo "  3. Monitor memory usage (BGE-base uses more RAM)"
echo "  4. Consider setting up SSL/TLS for production"

# Show next steps for accuracy system
echo ""
echo "Next Steps for Accuracy-Optimized System:"
echo "1. Edit environment: nano .env"
echo "2. Restart services: sudo docker-compose restart"
echo "3. Test API: curl http://localhost:8000/api/v1/hackrx/run -X POST -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"documents\":[\"test doc\"]}'"
echo "4. Monitor BGE-base model performance with: sudo docker stats"
