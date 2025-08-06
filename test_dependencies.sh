#!/bin/bash

# Quick dependency test script for Python 3.10.12
echo "🧪 Testing Python 3.10.12 dependencies..."

python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    # Test NumPy first
    import numpy as np
    print(f'✅ NumPy {np.__version__} - OK')
    
    # Test SciPy
    import scipy
    print(f'✅ SciPy {scipy.__version__} - OK')
    
    # Test HuggingFace Hub
    import huggingface_hub
    print(f'✅ HuggingFace Hub {huggingface_hub.__version__} - OK')
    
    # Test PyTorch
    import torch
    print(f'✅ PyTorch {torch.__version__} - OK')
    
    # Test Transformers
    import transformers
    print(f'✅ Transformers {transformers.__version__} - OK')
    
    # Test Sentence Transformers
    import sentence_transformers
    print(f'✅ Sentence Transformers {sentence_transformers.__version__} - OK')
    
    # Test FastAPI
    import fastapi
    print(f'✅ FastAPI {fastapi.__version__} - OK')
    
    # Test Google GenAI
    import google.generativeai as genai
    print(f'✅ Google Generative AI - OK')
    
    # Test PyMuPDF
    import fitz
    print(f'✅ PyMuPDF (fitz) - OK')
    
    print('\\n🎉 All dependencies loaded successfully!')
    
except ImportError as e:
    print(f'❌ Import Error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"
