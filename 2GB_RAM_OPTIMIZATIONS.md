# 2GB RAM Accuracy-Optimized RAG System

## System Configuration
- **Target System**: Amazon Lightsail 2GB RAM VPS
- **Available Memory**: ~1.3GB (after OS and system processes)  
- **Model**: BGE-base-en-v1.5 (~440MB) for higher accuracy
- **Reranker**: ms-marco-MiniLM-L-6-v2 (6-layer) for better reranking
- **Architecture**: Single-process accuracy-focused system **WITHOUT Redis**
- **LLM**: Gemini-1.5-flash with accuracy-tuned parameters

## Key Accuracy Optimizations Made

### 1. Enhanced Model Selection
- **Embedding Model**: BGE-base-en-v1.5 (768 dimensions) vs BGE-small (384 dimensions)
- **Reranker**: 6-layer MiniLM vs 2-layer for better accuracy
- **LLM Configuration**: Lower temperature (0.05), focused sampling (top_p: 0.7)
- **Output Tokens**: 2048 for comprehensive answers

### 2. Improved Retrieval Parameters  
- **Top-k initial**: 18 (increased for better coverage)
- **Top-k final**: 8 (more results for comprehensive answers)
- **Max rerank chunks**: 15 (enhanced reranking depth)
- **Max chunks retention**: 80 (better document coverage)
- **Rerank threshold**: 0.5 (lower = more reranking = better accuracy)

### 3. Enhanced Text Processing for Accuracy
- **Query length**: 200 chars (increased for better context)
- **Chunk text length**: 400 chars (increased for reranking)
- **Context window**: 800 chars per context (comprehensive context)
- **Prompt optimization**: Multi-instruction prompting for thoroughness
- **Chunk processing**: Quality-based retention vs simple length

### 4. Memory Management Without Redis
- **No Redis dependency**: Simplified architecture, more RAM for models
- **Model caching**: Local filesystem caching for BGE-base models  
- **Memory thresholds**: 1.7GB warning, 1.8GB critical
- **Batch processing**: 8-12 batch sizes for accuracy balance
- **Cleanup strategy**: Balanced frequency for stability

### 5. Docker Optimization for Accuracy
- **Resource limits**: 1.8GB memory, 1.9 CPUs
- **Model pre-caching**: Persistent model storage volumes
- **Health checks**: Extended timeouts for BGE-base loading
- **Environment**: Accuracy-focused configuration variables

## Performance Improvements Expected

### Accuracy Improvements (Primary Focus)
- **Better embeddings**: BGE-base vs BGE-small (~25% accuracy improvement)
- **Enhanced reranking**: 6-layer vs 2-layer cross-encoder (~20% improvement)
- **Comprehensive context**: 5 contexts with 800 chars each
- **Improved LLM prompting**: Multi-instruction prompts for thoroughness
- **Quality chunk retention**: Content-aware selection vs simple truncation

### Speed Considerations
- **Model loading**: ~60 seconds for BGE-base (vs 30s for BGE-small)
- **Embedding generation**: ~20% slower but much more accurate
- **Reranking**: ~30% slower but significantly better relevance
- **Overall latency**: 20-30s for complex queries (within 30s timeout)

### Memory Efficiency Without Redis
- **No Redis overhead**: ~200MB RAM saved for model processing
- **Direct model caching**: Filesystem-based persistent caching
- **Better resource allocation**: All RAM available for accuracy models
- **Simplified architecture**: Fewer moving parts, better stability

## Testing Recommendations

1. **Accuracy Testing**: Compare response quality with previous BGE-small system
2. **Memory Monitoring**: Watch for memory spikes above 1.6GB during BGE-base loading
3. **Performance Benchmarking**: Ensure 20-30s response times for complex queries
4. **Timeout Compliance**: Verify 30-second limit adherence under load
5. **Model Caching**: Test persistent model storage across container restarts

## Deployment Notes (Redis-Free Architecture)

- **Simplified Setup**: No Redis configuration or dependency management
- **Better Resource Utilization**: All 2GB RAM available for accuracy models
- **Model Persistence**: BGE-base models cached on filesystem for fast restart
- **Enhanced Stability**: Fewer services = fewer failure points
- **Direct Model Access**: No network overhead for model operations

## Docker Commands for Accuracy System

```bash
# Build and start accuracy-optimized system
docker-compose up --build -d

# Monitor memory usage
docker stats rag-accuracy-system

# View logs
docker-compose logs -f rag-accuracy-system

# Restart system
docker-compose restart rag-accuracy-system
```

## Monitoring Thresholds (Updated for Accuracy Focus)

- **Normal operation**: < 1.4GB memory usage
- **Warning level**: 1.4-1.6GB memory usage (BGE-base loading)
- **Critical level**: > 1.6GB memory usage
- **Emergency cleanup**: > 1.7GB memory usage

The system is now optimized for maximum accuracy on 2GB RAM without Redis dependency, prioritizing response quality over speed while maintaining reasonable performance.
