#!/bin/bash

# Installation script for Python 3.10.12 accuracy-optimized RAG system
# This script installs dependencies in the correct order to avoid conflicts

set -e  # Exit on any error

echo "🔧 Installing dependencies for Python 3.10.12 accuracy-optimized RAG system..."
echo "============================================================================="

# Check Python version
python_version=$(python3 --version)
echo "📍 Detected Python version: $python_version"

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Remove any existing problematic packages
echo "🧹 Removing any existing problematic packages..."
pip uninstall torch transformers sentence-transformers numpy scipy -y || true

# Install numpy first (critical for other dependencies)
echo "📊 Installing numpy==1.23.5..."
pip install --no-cache-dir numpy==1.23.5

# Install PyTorch 2.0.1 (CPU version) to fix uint64 error
echo "🔥 Installing PyTorch 2.0.1 (CPU version)..."
pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Install scipy with compatible version
echo "🧮 Installing scipy==1.9.3..."
pip install --no-cache-dir scipy==1.9.3

# Install transformers and sentence-transformers
echo "🤗 Installing transformers==4.30.2..."
pip install --no-cache-dir transformers==4.30.2

echo "📝 Installing sentence-transformers==2.2.2..."
pip install --no-cache-dir sentence-transformers==2.2.2

# Install remaining dependencies from requirements.txt
echo "📋 Installing remaining dependencies..."
pip install --no-cache-dir -r requirements.txt

# Test critical imports
echo "🧪 Testing critical imports..."
python3 -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} imported successfully')
    
    import transformers
    print(f'✅ Transformers {transformers.__version__} imported successfully')
    
    import sentence_transformers
    print(f'✅ Sentence-transformers {sentence_transformers.__version__} imported successfully')
    
    import numpy as np
    print(f'✅ NumPy {np.__version__} imported successfully')
    
    import fastapi
    print(f'✅ FastAPI {fastapi.__version__} imported successfully')
    
    import google.generativeai as genai
    print('✅ Google Generative AI imported successfully')
    
    import fitz  # PyMuPDF
    print('✅ PyMuPDF imported successfully')
    
    print('🎉 All critical dependencies installed and working!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    exit(1)
"

echo ""
echo "✅ Installation completed successfully!"
echo "🚀 Ready to run the accuracy-optimized RAG system!"
echo ""
echo "Next steps:"
echo "1. Set up your .env file with GOOGLE_API_KEY"
echo "2. Run: python main.py"
echo "   or"
echo "3. Run: docker-compose up --build -d"
