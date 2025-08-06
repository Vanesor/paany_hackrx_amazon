#!/usr/bin/env python3
"""
Accuracy-Optimized RAG System for 1.3GB RAM
Single request processing with 30-second timeout
Using Gemini-2.0-Flash and BGE-base for maximum accuracy
"""

import asyncio
import gc
import hashlib
import logging
import math
import os
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import fitz  # PyMuPDF
import numpy as np
import psutil
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, Header
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Memory and Performance Configuration for 1.3GB RAM
RESPONSE_TIMEOUT = 30  # 30 second hard timeout
PARTIAL_TIMEOUT = 28   # Start returning partial results at 28 seconds
MAX_MEMORY_GB = 1.2    # Maximum memory usage (1.2GB out of 1.3GB available)
CHUNK_SIZE = 800       # Smaller chunks for accuracy
OVERLAP_SIZE = 100     # Overlap for context preservation

# Model Configuration for Accuracy
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Best accuracy for English
EMBEDDING_DIM = 768    # BGE-base dimension
DEVICE = "cpu"         # Force CPU for stability on limited RAM
CPU_COUNT = min(4, os.cpu_count() or 4)

# Initialize Google Generative AI
try:
    import google.generativeai as genai
    # Configure with Gemini 2.0 Flash (most accurate and fast)
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY', 'your-api-key-here'))
    GEMINI_MODEL = 'gemini-2.0-flash-exp'  # Latest and most accurate
except ImportError:
    logger.error("Google Generative AI not available")
    genai = None

# Global executor for async operations
global_executor = ThreadPoolExecutor(max_workers=2)  # Minimal threads for 1.3GB RAM

class TimeoutManager:
    """Manages request timeout and partial result handling"""
    
    def __init__(self, timeout_seconds: int = RESPONSE_TIMEOUT):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()
        self.partial_results = []
        
    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    def get_remaining_time(self) -> float:
        return max(0, self.timeout_seconds - self.get_elapsed_time())
    
    def is_timeout_approaching(self, buffer_seconds: float = 2.0) -> bool:
        return self.get_remaining_time() <= buffer_seconds
    
    def is_partial_timeout(self) -> bool:
        return self.get_elapsed_time() >= PARTIAL_TIMEOUT
    
    def add_partial_result(self, result: str):
        self.partial_results.append(result)

def get_memory_usage() -> float:
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class DocumentElement:
    """Represents a document element with hierarchical structure"""
    
    def __init__(self, element_id: str, content: str, element_type: str, 
                 page_number: int, level: int = 0, title: str = "", 
                 parent_id: Optional[str] = None):
        self.element_id = element_id
        self.content = content
        self.element_type = element_type
        self.page_number = page_number
        self.level = level
        self.title = title
        self.parent_id = parent_id
        self.children_ids = []

class AccuracyRAGSystem:
    """Accuracy-optimized RAG system for single request processing"""
    
    def __init__(self):
        self.embedding_model = None
        self.embedding_model_name = EMBEDDING_MODEL
        self.performance_stats = {
            'total_embeddings_computed': 0,
            'memory_cleanups': 0,
            'requests_processed': 0
        }
        
        # Initialize Gemini model
        if genai:
            self.llm_model = genai.GenerativeModel(GEMINI_MODEL)
        else:
            self.llm_model = None
            logger.error("Gemini model not available")

    def load_models(self):
        """Load embedding model optimized for accuracy"""
        logger.info("Loading BGE-base model for maximum accuracy...")
        
        try:
            # Load BGE-base model for best accuracy
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=DEVICE,
                cache_folder="./model_cache"
            )
            
            # Optimize for accuracy over speed
            self.embedding_model.eval()
            
            # Warm up with sample text
            warmup_text = ["This is a warmup text for the BGE model to ensure optimal performance."]
            _ = self.embedding_model.encode(warmup_text, show_progress_bar=False)
            
            logger.info(f"BGE-base model loaded successfully on {DEVICE}")
            logger.info(f"Memory usage after loading: {get_memory_usage():.2f}GB")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _extract_pdf_text(self, content: bytes) -> List[Tuple[int, str]]:
        """Extract text from PDF with memory optimization"""
        pages = []
        logger.info("Starting PDF text extraction...")
        
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            
            for page_num in range(len(doc)):
                if get_memory_usage() > MAX_MEMORY_GB:
                    logger.warning(f"Memory limit reached at page {page_num}")
                    break
                    
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():  # Only add pages with content
                    pages.append((page_num + 1, text))
                
                # Clean up page immediately
                page = None
            
            doc.close()
            logger.info(f"Extracted {len(pages)} pages successfully")
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
        
        return pages

    async def download_and_extract(self, url: str) -> List[Tuple[int, str]]:
        """Download and extract PDF content"""
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download PDF: {response.status}")
                
                content = await response.read()
                
        return await asyncio.get_event_loop().run_in_executor(
            global_executor, self._extract_pdf_text, content
        )

    def _create_smart_chunks(self, pages: List[Tuple[int, str]], timeout_manager: TimeoutManager) -> List[Dict[str, Any]]:
        """Create intelligent chunks optimized for accuracy"""
        chunks = []
        
        for page_num, text in pages:
            if timeout_manager.is_timeout_approaching():
                logger.warning("Timeout approaching during chunking")
                break
                
            # Clean and prepare text
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                continue
            
            # Smart chunking: split by sentences and paragraphs for better context
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = ""
            
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > CHUNK_SIZE:
                    if current_chunk:
                        # Create chunk with metadata
                        chunk = {
                            'text': current_chunk.strip(),
                            'page': page_num,
                            'word_count': len(current_chunk.split()),
                            'char_count': len(current_chunk),
                            'chunk_id': f"page_{page_num}_chunk_{len(chunks)}"
                        }
                        chunks.append(chunk)
                    
                    # Start new chunk
                    current_chunk = sentence
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            # Add final chunk if it has content
            if current_chunk.strip():
                chunk = {
                    'text': current_chunk.strip(),
                    'page': page_num,
                    'word_count': len(current_chunk.split()),
                    'char_count': len(current_chunk),
                    'chunk_id': f"page_{page_num}_chunk_{len(chunks)}"
                }
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} intelligent chunks")
        return chunks

    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with memory optimization"""
        if not texts:
            return np.array([])
        
        try:
            # Process in small batches to manage memory
            batch_size = 8  # Small batches for 1.3GB RAM
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Generate embeddings for batch
                embeddings = self.embedding_model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=DEVICE
                )
                
                all_embeddings.append(embeddings)
                
                # Memory check
                if get_memory_usage() > MAX_MEMORY_GB:
                    force_memory_cleanup()
            
            # Combine all embeddings
            if all_embeddings:
                final_embeddings = np.vstack(all_embeddings)
            else:
                final_embeddings = np.zeros((len(texts), EMBEDDING_DIM))
            
            self.performance_stats['total_embeddings_computed'] += len(texts)
            return final_embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros((len(texts), EMBEDDING_DIM))

    async def retrieve_relevant_chunks(self, query: str, chunks: List[Dict], 
                                     chunk_embeddings: np.ndarray, 
                                     timeout_manager: TimeoutManager) -> List[Dict]:
        """Retrieve most relevant chunks using semantic similarity"""
        if len(chunks) == 0 or timeout_manager.is_timeout_approaching():
            return []
        
        # Generate query embedding
        query_embedding = await self._generate_embeddings([query])
        
        if query_embedding.size == 0:
            return []
        
        # Calculate similarities
        query_embedding = query_embedding.flatten()
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        # Get top relevant chunks (more for accuracy)
        top_k = min(8, len(chunks))  # Retrieve top 8 for accuracy
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Create result chunks with scores
        relevant_chunks = []
        for idx in top_indices:
            chunk = chunks[idx].copy()
            chunk['similarity_score'] = float(similarities[idx])
            relevant_chunks.append(chunk)
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
        return relevant_chunks

    def _create_context_for_llm(self, chunks: List[Dict]) -> str:
        """Create well-formatted context for the LLM"""
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks):
            page_info = f"[Page {chunk['page']}]"
            score_info = f"[Relevance: {chunk.get('similarity_score', 0):.3f}]"
            
            context_part = f"Context {i+1} {page_info} {score_info}:\n{chunk['text']}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)

    async def generate_answer(self, question: str, relevant_chunks: List[Dict], 
                            timeout_manager: TimeoutManager) -> str:
        """Generate accurate answer using Gemini"""
        if not self.llm_model:
            return "LLM model not available"
        
        if not relevant_chunks:
            return "No relevant information found in the document to answer this question."
        
        # Create context
        context = self._create_context_for_llm(relevant_chunks)
        
        # Create accuracy-focused prompt
        prompt = f"""You are an expert document analyst. Answer the question based ONLY on the provided context.

INSTRUCTIONS:
1. Provide a precise, comprehensive answer based solely on the given context
2. Include specific details, numbers, and relevant information from the context
3. If the context doesn't contain enough information, state what is available and what is missing
4. Use clear, professional language without emojis or formatting symbols
5. Include page references in parentheses like (Page X) when citing information

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        try:
            # Check timeout before generation
            if timeout_manager.is_timeout_approaching():
                return "Timeout approaching during answer generation."
            
            # Generate answer with Gemini
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.llm_model.generate_content, prompt
                ),
                timeout=max(3, timeout_manager.get_remaining_time() - 1)
            )
            
            return response.text.strip()
            
        except asyncio.TimeoutError:
            return "Answer generation timeout reached."
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}"

    async def process_document_and_questions(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Process document and answer questions with timeout management"""
        timeout_manager = TimeoutManager()
        answers = []
        
        try:
            # Download and extract PDF
            logger.info("Starting document processing...")
            pages = await self.download_and_extract(pdf_url)
            
            if timeout_manager.is_timeout_approaching():
                return ["Timeout during PDF extraction"] * len(questions)
            
            # Create intelligent chunks
            chunks = self._create_smart_chunks(pages, timeout_manager)
            
            if timeout_manager.is_timeout_approaching():
                return ["Timeout during document chunking"] * len(questions)
            
            # Generate embeddings for chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            chunk_embeddings = await self._generate_embeddings(chunk_texts)
            
            if timeout_manager.is_timeout_approaching():
                return ["Timeout during embedding generation"] * len(questions)
            
            # Process each question
            for i, question in enumerate(questions):
                if timeout_manager.is_partial_timeout():
                    # Return partial results if we've reached 28 seconds
                    remaining_questions = len(questions) - len(answers)
                    answers.extend([""] * remaining_questions)
                    break
                
                # Retrieve relevant chunks
                relevant_chunks = await self.retrieve_relevant_chunks(
                    question, chunks, chunk_embeddings, timeout_manager
                )
                
                # Generate answer
                answer = await self.generate_answer(question, relevant_chunks, timeout_manager)
                answers.append(answer)
                
                logger.info(f"Processed question {i+1}/{len(questions)}")
            
            # Update performance stats
            self.performance_stats['requests_processed'] += 1
            
            return answers
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            # Return error messages for all questions
            error_message = f"Processing error: {str(e)}"
            return [error_message] * len(questions)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.performance_stats,
            "model_info": {
                "embedding_model": self.embedding_model_name,
                "llm_model": GEMINI_MODEL,
                "device": DEVICE,
                "memory_limit_gb": MAX_MEMORY_GB
            },
            "current_memory_gb": get_memory_usage()
        }

# Global RAG system instance
rag_system = AccuracyRAGSystem()

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Accuracy-Optimized RAG System...")
    await asyncio.get_event_loop().run_in_executor(global_executor, rag_system.load_models)
    logger.info("System ready for processing")
    
    yield
    
    logger.info("Shutting down system...")
    force_memory_cleanup()
    global_executor.shutdown(wait=True)

app = FastAPI(
    title="Accuracy RAG System",
    version="1.0.0",
    description="Single-request accuracy-optimized RAG system with 30-second timeout",
    lifespan=lifespan
)

# API Models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Expected token for authentication
EXPECTED_TOKEN = os.getenv('API_TOKEN', 'your-secure-token-here')

def verify_token(authorization: str = Header(None)):
    """Verify API token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    token = authorization.split("Bearer ")[1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

# Main API endpoint
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_document_and_answer(
    request: QueryRequest,
    authorization: str = Header(None)
):
    """Main endpoint for document processing and question answering"""
    verify_token(authorization)
    
    request_id = f"req_{int(time.time() * 1000)}"
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    logger.info(f"[{request_id}] Starting request processing")
    logger.info(f"[{request_id}] Questions: {len(request.questions)}")
    logger.info(f"[{request_id}] Initial memory: {initial_memory:.2f}GB")
    
    # Memory check
    if initial_memory > MAX_MEMORY_GB:
        logger.error(f"[{request_id}] Memory too high: {initial_memory:.2f}GB")
        raise HTTPException(status_code=503, detail="Server memory too high")
    
    try:
        # Process document and questions
        answers = await rag_system.process_document_and_questions(
            request.documents, request.questions
        )
        
        # Performance logging
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        
        logger.info(f"[{request_id}] Request completed in {total_time:.2f}s")
        logger.info(f"[{request_id}] Memory: {initial_memory:.2f}GB -> {final_memory:.2f}GB")
        logger.info(f"[{request_id}] Answers returned: {len([a for a in answers if a])}")
        
        # Cleanup
        force_memory_cleanup()
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"[{request_id}] Request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "memory_usage_gb": get_memory_usage(),
        "memory_limit_gb": MAX_MEMORY_GB,
        "model_loaded": rag_system.embedding_model is not None
    }

# Performance stats endpoint
@app.get("/performance/stats")
async def get_performance_stats(authorization: str = Header(None)):
    """Get performance statistics"""
    verify_token(authorization)
    return rag_system.get_performance_stats()

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "Accuracy RAG System",
        "version": "1.0.0",
        "description": "Single-request accuracy-optimized RAG system",
        "model": rag_system.embedding_model_name,
        "llm_model": GEMINI_MODEL,
        "timeout_seconds": RESPONSE_TIMEOUT,
        "memory_limit_gb": MAX_MEMORY_GB,
        "endpoints": {
            "main": "/api/v1/hackrx/run",
            "health": "/health",
            "performance": "/performance/stats"
        }
    }

# Server execution
if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid port: {sys.argv[1]}, using 8000")
    
    logger.info("Starting Accuracy RAG Server for 1.3GB RAM...")
    logger.info(f"Device: {DEVICE} | CPU cores: {CPU_COUNT}")
    logger.info(f"Model: {rag_system.embedding_model_name}")
    logger.info(f"LLM: {GEMINI_MODEL}")
    logger.info(f"Timeout: {RESPONSE_TIMEOUT}s | Memory limit: {MAX_MEMORY_GB}GB")
    logger.info(f"Port: {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        loop="asyncio",
        log_level="info",
        timeout_keep_alive=45,
        limit_concurrency=1,  # Single request processing
        limit_max_requests=100,
        access_log=True
    )
