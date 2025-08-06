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
import dotenv
dotenv.load_dotenv()
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure enhanced logging with colors and more detailed formatting
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and enhanced detail"""
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m\033[1m', # Bold Red
        'RESET': '\033[0m'    # Reset
    }
    
    def format(self, record):
        # Add memory usage to log record
        if not hasattr(record, 'memory'):
            record.memory = f"{get_memory_usage():.2f}GB"
        
        # Add elapsed time if not present
        if not hasattr(record, 'elapsed'):
            record.elapsed = f"{time.time() - START_TIME:.1f}s"
            
        levelname = record.levelname
        message = super().format(record)
        if levelname in self.COLORS:
            message = f"{self.COLORS[levelname]}{message}{self.COLORS['RESET']}"
        return message

# Start time for elapsed time calculation
START_TIME = time.time()

# Configure root logger with enhanced format
log_format = '%(asctime)s | [%(elapsed)s] | %(levelname)-8s | [🧠 %(memory)s] | %(message)s'
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(log_format))

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler]
)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info(f"Starting RAG system with BGE-small and enhanced accuracy settings")

# Memory and Performance Configuration for 1.3GB RAM
RESPONSE_TIMEOUT = 30  # 30 second hard timeout
PARTIAL_TIMEOUT = 28   # Start returning partial results at 28 seconds
MAX_MEMORY_GB = 1.2    # Maximum memory usage (1.2GB out of 1.3GB available)
CHUNK_SIZE = 800       # Smaller chunks for accuracy
OVERLAP_SIZE = 100     # Overlap for context preservation

# Model Configuration for Accuracy
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Faster with good accuracy for English
EMBEDDING_DIM = 384    # BGE-small dimension (reduced from 768)
DEVICE = "cpu"         # Force CPU for stability on limited RAM
CPU_COUNT = min(4, os.cpu_count() or 4)

# Enhanced accuracy settings
CHUNK_OVERLAP_FACTOR = 0.15  # 15% overlap between chunks for better context
TOP_K_CHUNKS = 8      # Number of chunks to retrieve for each question
RERANK_FACTOR = 1.2   # Boost score for chunks with exact keyword matches

# Initialize Google Generative AI
try:
    import google.generativeai as genai
    # Configure with Gemini Flash (most accurate and fast)
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY', 'your-api-key-here'))
    GEMINI_MODEL = 'gemini-2.5-flash-lite'  # Stable and fast model
except ImportError:
    logger.error("Google Generative AI not available")
    genai = None

# Global executors for parallel processing
global_executor = ThreadPoolExecutor(max_workers=2)  # Main executor for PDF processing
embedding_executor = ThreadPoolExecutor(max_workers=3)  # Dedicated executor for embeddings
chunking_executor = ThreadPoolExecutor(max_workers=2)  # Dedicated executor for chunking

# Single request processing lock
processing_lock = asyncio.Lock()

class TimeoutManager:
    """Manages request timeout and partial result handling with enhanced timing"""
    
    def __init__(self, timeout_seconds: int = RESPONSE_TIMEOUT):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()
        self.partial_results = []
        self.stage_timings = {}
        self.current_stage = None
        self.stage_start_time = None
        logger.info(f"Timeout manager initialized with {timeout_seconds}s timeout, partial results at {PARTIAL_TIMEOUT}s")
        
    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    def get_remaining_time(self) -> float:
        remaining = max(0, self.timeout_seconds - self.get_elapsed_time())
        if remaining < 5 and remaining > 0:
            logger.warning(f"Only {remaining:.2f}s remaining until timeout")
        return remaining
    
    def is_timeout_approaching(self, buffer_seconds: float = 2.0) -> bool:
        remaining = self.get_remaining_time()
        is_approaching = remaining <= buffer_seconds
        if is_approaching:
            logger.warning(f"Timeout approaching! Only {remaining:.2f}s left")
        return is_approaching
    
    def is_partial_timeout(self) -> bool:
        elapsed = self.get_elapsed_time()
        is_partial = elapsed >= PARTIAL_TIMEOUT
        if is_partial:
            logger.warning(f"Partial timeout reached at {elapsed:.2f}s (limit: {PARTIAL_TIMEOUT}s)")
        return is_partial
    
    def add_partial_result(self, result: str):
        self.partial_results.append(result)
    
    def start_stage(self, stage_name: str):
        """Start timing a processing stage"""
        if self.current_stage:
            self.end_stage()
        
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        logger.info(f"Starting stage: {stage_name} (elapsed: {self.get_elapsed_time():.2f}s)")
    
    def end_stage(self):
        """End timing the current stage"""
        if not self.current_stage or not self.stage_start_time:
            return
        
        duration = time.time() - self.stage_start_time
        self.stage_timings[self.current_stage] = round(duration, 2)
        logger.info(f"Completed stage: {self.current_stage} in {duration:.2f}s")
        
        self.current_stage = None
        self.stage_start_time = None
    
    def get_timing_report(self):
        """Get a complete timing report"""
        return {
            "total_elapsed": round(self.get_elapsed_time(), 2),
            "stages": self.stage_timings,
            "remaining": round(self.get_remaining_time(), 2)
        }

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
        # Initialize cache for embeddings to avoid recomputation
        self.embedding_cache = {}
        
        # Initialize Gemini model
        if genai:
            # Configure Gemini with optimized settings
            generation_config = {
                "temperature": 0.1,  # Lower temperature for more accurate responses
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ]
            self.llm_model = genai.GenerativeModel(
                GEMINI_MODEL,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        else:
            self.llm_model = None
            logger.error("Gemini model not available")
            
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embedding quality"""
        # Remove extra whitespace and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        # Keep informative punctuation but normalize it
        text = re.sub(r'[^\w\s\.\,\:\;\?\!]', ' ', text)
        # Add special handling for key information patterns
        text = re.sub(r'(\d+[\.:])', r' \1 ', text)  # Add space around numbers with punctuation
        
        return text

    def load_models(self):
        """Load embedding model optimized for accuracy and performance"""
        logger.info(f"Loading {self.embedding_model_name} model...")
        load_start = time.perf_counter()
        
        try:
            # Memory before loading
            mem_before = get_memory_usage()
            logger.info(f"Memory before model loading: {mem_before:.2f}GB")
            
            # Load the BGE model with optimized settings
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=DEVICE,
                cache_folder="./model_cache"
            )
            
            # Optimize for inference
            self.embedding_model.eval()
            
            # Apply quantization to reduce memory usage if on CPU
            if DEVICE == "cpu":
                logger.info("Applying quantization to reduce memory footprint...")
                from torch.quantization import quantize_dynamic
                try:
                    self.embedding_model = quantize_dynamic(
                        self.embedding_model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("Model quantization applied successfully")
                except Exception as qe:
                    logger.warning(f"Quantization failed, using full precision model: {qe}")
            
            # Warm up with real examples for better initial performance
            warmup_texts = [
                "What is the coverage for pre-existing conditions?",
                "The policy states that maternity benefits have a waiting period.",
                "According to section 3.2, liability coverage extends to household members."
            ]
            
            logger.info("Warming up embedding model with realistic examples...")
            warmup_start = time.perf_counter()
            _ = self.embedding_model.encode(warmup_texts, show_progress_bar=False)
            warmup_time = time.perf_counter() - warmup_start
            logger.info(f"Model warm-up completed in {warmup_time:.2f}s")
            
            # Memory after loading
            mem_after = get_memory_usage()
            mem_diff = mem_after - mem_before
            
            # Log detailed model information
            load_time = time.perf_counter() - load_start
            logger.info(f"✅ {self.embedding_model_name} loaded in {load_time:.2f}s on {DEVICE}")
            logger.info(f"📊 Model stats: {EMBEDDING_DIM} dimensions, memory usage: {mem_diff:.2f}GB")
            logger.info(f"🧠 Total memory: {mem_after:.2f}GB/{MAX_MEMORY_GB}GB ({(mem_after/MAX_MEMORY_GB)*100:.1f}%)")
            
            # Initialize cache for frequently used embeddings
            self.performance_stats['model_load_time'] = load_time
            self.performance_stats['model_memory_usage'] = mem_diff
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            logger.error("Check your model cache and internet connection")
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
        """Create intelligent chunks optimized for accuracy and performance"""
        chunks = []
        start_time = time.time()
        
        # Enhanced pattern for better sentence and paragraph splitting
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
        
        # Identify important sections (headings, titles) using regex patterns
        heading_patterns = [
            re.compile(r'^[A-Z][A-Z\s]{3,}$', re.MULTILINE),  # ALL CAPS HEADINGS
            re.compile(r'^(?:\d+\.)+\s+[A-Z]', re.MULTILINE),  # Numbered headings (1.2.3)
            re.compile(r'^[A-Z][a-z]+\s+\d+[\.:]\s', re.MULTILINE)  # Section/Article headings
        ]
        
        # Process pages in parallel using executor
        all_page_chunks = []
        
        # Split text processing into smaller operations to prevent timeouts
        max_pages_per_batch = 50
        page_batches = [pages[i:i+max_pages_per_batch] for i in range(0, len(pages), max_pages_per_batch)]
        
        for batch_index, page_batch in enumerate(page_batches):
            # Check timeout at the batch level
            if timeout_manager.is_timeout_approaching():
                logger.warning(f"Timeout approaching during chunking (batch {batch_index+1}/{len(page_batches)})")
                break
                
            batch_start = time.time()
            batch_chunks = []
            
            for page_num, text in page_batch:
                # Clean and prepare text
                text = re.sub(r'\s+', ' ', text).strip()
                
                if not text:
                    continue
                
                # First, try to identify structural elements like headings
                segments = []
                last_pos = 0
                
                # Find headings and split text at those points
                for pattern in heading_patterns:
                    for match in pattern.finditer(text):
                        start, end = match.span()
                        if start > last_pos:
                            # Add text before heading
                            segments.append(text[last_pos:start])
                        # Add heading as separate segment (gives it more weight)
                        segments.append(text[start:end])
                        last_pos = end
                
                # Add final segment
                if last_pos < len(text):
                    segments.append(text[last_pos:])
                
                # If no segments were created, use the whole text
                if not segments:
                    segments = [text]
                
                # Process each segment
                for segment in segments:
                    # Smart chunking: split by sentences for better context
                    sentences = sentence_pattern.split(segment)
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
                                    'chunk_id': f"page_{page_num}_chunk_{len(batch_chunks)+len(chunks)}"
                                }
                                batch_chunks.append(chunk)
                            
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
                            'chunk_id': f"page_{page_num}_chunk_{len(batch_chunks)+len(chunks)}"
                        }
                        batch_chunks.append(chunk)
            
            # Add enhanced overlap between chunks for better context preservation
            enhanced_batch_chunks = []
            for i, chunk in enumerate(batch_chunks):
                # Keep the original chunk
                enhanced_batch_chunks.append(chunk)
                
                # For important chunks (determined by length, position or content patterns)
                # create an overlapping chunk with neighboring content
                if i > 0 and i < len(batch_chunks) - 1:
                    # If this chunk might contain important information based on content patterns
                    if (len(chunk['text']) > CHUNK_SIZE * 0.8 or
                        re.search(r'(definition|policy|section|article|clause)', chunk['text'].lower())):
                        
                        # Create an overlapping chunk that combines parts of previous, current and next chunk
                        prev_text = batch_chunks[i-1]['text'][-OVERLAP_SIZE:] if i > 0 else ""
                        next_text = batch_chunks[i+1]['text'][:OVERLAP_SIZE] if i < len(batch_chunks) - 1 else ""
                        
                        overlap_chunk = {
                            'text': f"{prev_text} {chunk['text']} {next_text}".strip(),
                            'page': chunk['page'],
                            'word_count': len(f"{prev_text} {chunk['text']} {next_text}".split()),
                            'char_count': len(f"{prev_text} {chunk['text']} {next_text}"),
                            'chunk_id': f"overlap_{chunk['chunk_id']}",
                            'is_overlap': True
                        }
                        enhanced_batch_chunks.append(overlap_chunk)
            
            # Add all enhanced chunks from this batch
            chunks.extend(enhanced_batch_chunks)
            
            batch_time = time.time() - batch_start
            logger.info(f"Processed batch {batch_index+1}/{len(page_batches)} with {len(enhanced_batch_chunks)} chunks in {batch_time:.2f}s")
        
        # Final optimization: deduplicate near-identical chunks to save on embeddings
        unique_chunks = {}
        for chunk in chunks:
            # Use first 100 chars as a signature
            signature = chunk['text'][:100]
            if signature not in unique_chunks:
                unique_chunks[signature] = chunk
            elif chunk.get('is_overlap', False) and not unique_chunks[signature].get('is_overlap', False):
                # If this is an overlap chunk replacing a regular chunk, use the overlap one
                unique_chunks[signature] = chunk
        
        final_chunks = list(unique_chunks.values())
        
        total_time = time.time() - start_time
        logger.info(f"Created {len(final_chunks)} intelligent chunks in {total_time:.2f}s ({len(pages)} pages)")
        logger.info(f"Average processing speed: {len(pages)/max(0.001, total_time):.1f} pages/second")
        
        return final_chunks

    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with memory optimization and caching"""
        if not texts:
            return np.array([])
        
        try:
            # Use embedding cache to avoid recomputing identical texts
            cache_hits = 0
            texts_to_embed = []
            cached_embeddings = []
            text_indices = []
            
            # Check cache first
            for i, text in enumerate(texts):
                # Use a hash of the text as cache key to save memory
                text_hash = hashlib.md5(text.encode()).hexdigest()
                
                if text_hash in self.embedding_cache:
                    cached_embeddings.append(self.embedding_cache[text_hash])
                    cache_hits += 1
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
            
            # If we have texts to embed
            if texts_to_embed:
                # Process in efficient batches
                batch_size = 16  # Larger batches for better performance
                all_new_embeddings = []
                
                # Use dedicated embedding executor to generate embeddings
                def embed_batch(batch):
                    return self.embedding_model.encode(
                        batch,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        device=DEVICE
                    )
                
                # Process batches in parallel
                embedding_futures = []
                for i in range(0, len(texts_to_embed), batch_size):
                    batch = texts_to_embed[i:i + batch_size]
                    # Submit batch to embedding executor
                    future = embedding_executor.submit(embed_batch, batch)
                    embedding_futures.append(future)
                
                # Collect results
                for i, future in enumerate(embedding_futures):
                    start_idx = i * batch_size
                    batch = texts_to_embed[start_idx:start_idx + batch_size]
                    embeddings = future.result()
                    all_new_embeddings.append(embeddings)
                    
                    # Update cache with new embeddings
                    for j, text in enumerate(batch):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        # Store in cache
                        self.embedding_cache[text_hash] = embeddings[j]
                
                # Combine all new embeddings
                if all_new_embeddings:
                    new_embeddings = np.vstack(all_new_embeddings)
                else:
                    new_embeddings = np.zeros((0, EMBEDDING_DIM))
                
                self.performance_stats['total_embeddings_computed'] += len(texts_to_embed)
            else:
                new_embeddings = np.zeros((0, EMBEDDING_DIM))
            
            # Now reconstruct the final embeddings array in the original order
            if cache_hits > 0:
                # Combine cached and new embeddings
                final_embeddings = np.zeros((len(texts), EMBEDDING_DIM))
                
                # Place cached embeddings
                cached_idx = 0
                for i in range(len(texts)):
                    if i not in text_indices:  # This was a cached embedding
                        final_embeddings[i] = cached_embeddings[cached_idx]
                        cached_idx += 1
                
                # Place new embeddings
                new_idx = 0
                for i in text_indices:
                    final_embeddings[i] = new_embeddings[new_idx]
                    new_idx += 1
                    
                logger.info(f"Embedding cache hits: {cache_hits}/{len(texts)} ({cache_hits/len(texts)*100:.1f}%)")
            else:
                final_embeddings = new_embeddings
            
            # Perform memory cleanup if needed
            current_memory = get_memory_usage()
            if current_memory > MAX_MEMORY_GB * 0.9:
                # Clean up embedding cache if it's getting too large (keep most recent 100 entries)
                if len(self.embedding_cache) > 100:
                    keys = list(self.embedding_cache.keys())
                    for key in keys[:-100]:  # Remove oldest entries
                        del self.embedding_cache[key]
                force_memory_cleanup()
                self.performance_stats['memory_cleanups'] += 1
            
            return final_embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros((len(texts), EMBEDDING_DIM))

    async def retrieve_relevant_chunks(self, query: str, chunks: List[Dict], 
                                     chunk_embeddings: np.ndarray, 
                                     timeout_manager: TimeoutManager) -> List[Dict]:
        """Retrieve most relevant chunks using enhanced semantic similarity and re-ranking"""
        if len(chunks) == 0 or timeout_manager.is_timeout_approaching():
            return []
            
        query_start = time.perf_counter()
        logger.info(f"Finding relevant information for: '{query[:50]}...'")
        
        # Extract key terms from the query for lexical matching
        query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
        
        # Generate query embedding
        query_embedding = await self._generate_embeddings([query])
        
        if query_embedding.size == 0:
            logger.warning("Failed to generate query embedding")
            return []
        
        # Calculate similarities
        query_embedding = query_embedding.flatten()
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        # First-pass selection: Get more chunks than needed for re-ranking
        pre_rerank_k = min(TOP_K_CHUNKS * 2, len(chunks))
        pre_rerank_indices = np.argsort(similarities)[-pre_rerank_k:][::-1]
        
        # Re-rank using additional signals
        candidates = []
        for idx in pre_rerank_indices:
            chunk = chunks[idx].copy()
            base_score = float(similarities[idx])
            chunk['similarity_score'] = base_score
            
            # Lexical match bonus: check for presence of key query terms
            text_lower = chunk['text'].lower()
            matching_terms = [term for term in query_terms if term in text_lower]
            term_match_ratio = len(matching_terms) / max(1, len(query_terms))
            
            # Apply lexical bonus to semantic score
            if term_match_ratio > 0:
                lexical_bonus = term_match_ratio * RERANK_FACTOR
                chunk['lexical_match_score'] = lexical_bonus
                chunk['final_score'] = base_score * (1 + lexical_bonus)
                chunk['matching_terms'] = matching_terms
            else:
                chunk['lexical_match_score'] = 0
                chunk['final_score'] = base_score
                
            # Boost scores for chunks with numeric data when query contains numeric references
            if re.search(r'\b\d+\b', query) and re.search(r'\b\d+\b', chunk['text']):
                chunk['final_score'] *= 1.1
                chunk['has_numeric_match'] = True
                
            # Boost scores for chunks with heading or title indicators
            if re.search(r'\b(section|chapter|article|clause|definition)\b', chunk['text'].lower()):
                chunk['final_score'] *= 1.15
                chunk['is_structural'] = True
                
            # Page number relevance - boost earlier pages for definitional questions
            if "what is" in query.lower() or "definition" in query.lower() or "defined" in query.lower():
                page_position_factor = 1 + (0.1 * (1 / max(1, chunk['page'])))
                chunk['final_score'] *= page_position_factor
            
            candidates.append(chunk)
        
        # Sort by final re-ranked score
        candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Select top K after re-ranking
        top_k = min(TOP_K_CHUNKS, len(candidates))
        relevant_chunks = candidates[:top_k]
        
        # Add diversification: if we have very similar top chunks, replace some with more diverse ones
        if len(relevant_chunks) >= 4:
            # Check similarity between top chunks
            top_texts = [c['text'][:100] for c in relevant_chunks[:3]]
            for i in range(3, len(candidates)):
                if i >= len(relevant_chunks):
                    break
                candidate = candidates[i]
                # If this chunk adds diversity (comes from different page or has different matching terms)
                if (candidate['page'] not in [c['page'] for c in relevant_chunks[:3]] or
                    set(candidate.get('matching_terms', [])).difference(*[set(c.get('matching_terms', [])) for c in relevant_chunks[:3]])):
                    # Replace the lowest scoring chunk in positions 3+ with this diverse chunk
                    min_idx = min(range(3, len(relevant_chunks)), key=lambda i: relevant_chunks[i]['final_score'])
                    relevant_chunks[min_idx] = candidate
                    break
        
        # Log detailed information about chunk selection
        query_time = time.perf_counter() - query_start
        
        # Create score summary for logging
        score_summary = [f"{c.get('final_score', 0):.3f}" for c in relevant_chunks[:3]]
        matching_terms = set().union(*[set(c.get('matching_terms', [])) for c in relevant_chunks])
        pages = sorted(set(c['page'] for c in relevant_chunks))
        
        logger.info(f"Retrieved {len(relevant_chunks)}/{len(chunks)} chunks in {query_time:.3f}s")
        logger.info(f"Top scores: {', '.join(score_summary)} | Pages: {pages} | Terms: {list(matching_terms)[:5]}")
        
        return relevant_chunks

    def _create_context_for_llm(self, chunks: List[Dict]) -> str:
        """Create enhanced well-formatted context for the LLM with structural indicators"""
        if not chunks:
            return ""
            
        logger.info(f"Creating context from {len(chunks)} chunks")
        
        # First, sort chunks by page number for better context coherence
        chunks_by_page = sorted(chunks, key=lambda x: (x['page'], -x.get('final_score', 0)))
        
        # Group chunks by page
        page_groups = {}
        for chunk in chunks_by_page:
            page = chunk['page']
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(chunk)
            
        # Create context with page structure maintained
        context_parts = []
        
        # Add metadata header with document overview
        pages = sorted(list(page_groups.keys()))
        context_parts.append(f"DOCUMENT OVERVIEW: Content spans {len(pages)} pages, with information from pages: {', '.join(map(str, pages))}\n")
        
        # Process each page group
        for page, page_chunks in sorted(page_groups.items()):
            # Add page header
            context_parts.append(f"--- PAGE {page} ---")
            
            # Add chunks from this page
            for i, chunk in enumerate(page_chunks):
                # Create a rich content header with metadata
                score_info = f"[Relevance: {chunk.get('final_score', 0):.3f}]"
                
                # Include metadata about the chunk that might help the LLM
                meta_info = []
                if chunk.get('is_structural', False):
                    meta_info.append("STRUCTURAL SECTION")
                if chunk.get('has_numeric_match', True):
                    meta_info.append("CONTAINS NUMERIC DATA")
                if chunk.get('is_overlap', False):
                    meta_info.append("CONTEXTUAL OVERLAP")
                if chunk.get('matching_terms'):
                    terms = ", ".join(chunk.get('matching_terms', [])[:3])
                    meta_info.append(f"KEY TERMS: {terms}")
                    
                meta_string = f" [{' | '.join(meta_info)}]" if meta_info else ""
                
                # Format the chunk content
                text = chunk['text'].strip()
                # Add paragraph breaks for better readability
                text = re.sub(r'(\. )', r'.\n', text, flags=re.MULTILINE)
                
                # Create the final context part
                context_part = f"EXTRACT {i+1} {score_info}{meta_string}:\n{text}"
                context_parts.append(context_part)
        
        # Log statistics about the context
        total_chars = sum(len(part) for part in context_parts)
        logger.info(f"Created context with {len(context_parts)} sections, {total_chars} characters from {len(pages)} pages")
        
        return "\n\n".join(context_parts)

    async def generate_answer(self, question: str, relevant_chunks: List[Dict], 
                            timeout_manager: TimeoutManager) -> str:
        """Generate accurate answer using Gemini with optimized context handling"""
        if not self.llm_model:
            return "LLM model not available"
        
        if not relevant_chunks:
            return "No relevant information found in the document to answer this question."
        
        # Calculate available time for generation
        available_time = timeout_manager.get_remaining_time() - 1.5  # Buffer for response handling
        
        try:
            # Smart context selection based on relevance and question type
            # Sort chunks by similarity score
            sorted_chunks = sorted(relevant_chunks, key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            # Analyze question to determine information needs
            is_factual = re.search(r'(what|when|where|who|how many|how much|define)', question.lower()) is not None
            is_comparative = re.search(r'(compare|difference|versus|vs|similar|differ)', question.lower()) is not None
            needs_examples = re.search(r'(example|illustrate|demonstrate)', question.lower()) is not None
            
            # Select chunks based on question type
            selected_chunks = []
            
            if is_factual:
                # For factual questions, prioritize highest relevance chunks
                selected_chunks = sorted_chunks[:min(5, len(sorted_chunks))]
            elif is_comparative:
                # For comparative questions, include more diverse chunks
                selected_chunks = sorted_chunks[:min(8, len(sorted_chunks))]
            elif needs_examples:
                # For questions needing examples, include more chunks
                selected_chunks = sorted_chunks[:min(8, len(sorted_chunks))]
            else:
                # Default selection
                selected_chunks = sorted_chunks[:min(6, len(sorted_chunks))]
            
            # Create optimized context
            context = self._create_context_for_llm(selected_chunks)
            
            # Create accuracy-focused prompt with enhanced instructions
            system_prompt = """You are an expert document analyst with exceptional precision. Answer the question based ONLY on the provided context."""
            
            user_prompt = f"""IMPORTANT INSTRUCTIONS:
1. Answer the question using ONLY information from the provided context
2. Include specific details, numbers, and relevant information from the context
3. If the context doesn't have enough information, clearly state what is available and what is missing
4. Use clear, professional language without emojis
5. Include page references in parentheses like (Page X) when citing specific information
6. Be concise yet thorough - focus on accuracy above all
7. Never make up information that isn't in the context

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
            
            # Check timeout before generation
            if timeout_manager.is_timeout_approaching():
                return "Timeout approaching during answer generation."
            
            # Generate answer with Gemini using structured approach
            try:
                # First try with system prompt for better accuracy
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.llm_model.generate_content(
                            [
                                {"role": "system", "parts": [system_prompt]},
                                {"role": "user", "parts": [user_prompt]}
                            ]
                        )
                    ),
                    timeout=max(3, available_time)
                )
            except:
                # Fall back to simpler prompt if system prompt fails
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.llm_model.generate_content(user_prompt)
                    ),
                    timeout=max(2, available_time * 0.8)
                )
            
            answer = response.text.strip()
            
            # Post-process answer for better formatting
            # Remove any markdown code blocks
            answer = re.sub(r'```[a-z]*\n|```', '', answer)
            
            # Ensure page references are properly formatted
            answer = re.sub(r'page (\d+)', r'Page \1', answer, flags=re.IGNORECASE)
            
            return answer
            
        except asyncio.TimeoutError:
            # More informative timeout message
            return "The answer generation reached the time limit. Based on the available information, the document appears to contain relevant content, but a complete answer could not be generated within the time constraint."
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # More helpful error message
            return f"Unable to generate a complete answer: {str(e)}. Please try rephrasing your question or checking if the document contains the relevant information."

    async def process_document_and_questions(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Process document and answer questions with timeout management"""
        timeout_manager = TimeoutManager()
        answers = []
        
        try:
            # Process Start
            logger.info(f"Starting document processing at {time.strftime('%H:%M:%S.%f')[:-3]}")
            logger.info(f"Target timeout: {timeout_manager.timeout_seconds}s, partial results at {PARTIAL_TIMEOUT}s")
            
            # Download and extract PDF
            timeout_manager.start_stage("pdf_extraction")
            pages = await self.download_and_extract(pdf_url)
            timeout_manager.end_stage()
            
            logger.info(f"PDF extraction: {len(pages)} pages extracted in {timeout_manager.stage_timings['pdf_extraction']}s")
            
            if timeout_manager.is_timeout_approaching():
                logger.error(f"Timeout after PDF extraction at {timeout_manager.get_elapsed_time():.2f}s")
                return ["Timeout during PDF extraction"] * len(questions)
                
            # Parallel processing for chunking and initial embedding tasks
            timeout_manager.start_stage("parallel_processing")
            
            # Create function to chunk in background
            async def chunk_in_background():
                return await asyncio.get_event_loop().run_in_executor(
                    chunking_executor, self._create_smart_chunks, pages, timeout_manager
                )
                
            # Start chunking task (will run in parallel)
            chunking_task = asyncio.create_task(chunk_in_background())
            
            # While chunking is happening, prepare embedding model and cache
            # This helps reduce total processing time by doing work in parallel
            if hasattr(self, 'embedding_model') and self.embedding_model:
                # Warm up embedding model while chunking happens
                warmup_text = ["This document contains important information that needs to be processed efficiently."]
                await asyncio.get_event_loop().run_in_executor(
                    embedding_executor, 
                    lambda: self.embedding_model.encode(
                        warmup_text, 
                        normalize_embeddings=True, 
                        show_progress_bar=False
                    )
                )
                
            # Wait for chunking to complete
            chunks = await chunking_task
            
            # Process chunk texts immediately after getting chunks
            chunk_texts = [self._preprocess_text(chunk['text']) for chunk in chunks]
            
            # Split chunks into batches for parallel embedding processing
            batch_size = 16  # Process more chunks at once for efficiency
            chunk_batches = [chunk_texts[i:i+batch_size] for i in range(0, len(chunk_texts), batch_size)]
            
            # Generate embeddings for chunks in parallel
            embedding_tasks = []
            for batch in chunk_batches:
                task = asyncio.create_task(self._generate_embeddings(batch))
                embedding_tasks.append(task)
            
            # Wait for all embedding tasks to complete
            batch_embeddings = await asyncio.gather(*embedding_tasks)
            
            # Combine all batch embeddings
            if batch_embeddings:
                chunk_embeddings = np.vstack(batch_embeddings)
            else:
                chunk_embeddings = np.zeros((0, EMBEDDING_DIM))
            
            timeout_manager.end_stage()
            
            # Log performance metrics
            processing_time = timeout_manager.stage_timings['parallel_processing']
            logger.info(f"Parallel processing completed in {processing_time:.2f}s")
            logger.info(f"Document chunking: {len(chunks)} chunks created")
            logger.info(f"Average chars per chunk: {sum(len(c['text']) for c in chunks) / max(1, len(chunks)):.1f}")
            logger.info(f"Generated {len(chunk_texts)} embeddings, processing rate: {len(chunk_texts)/max(0.001, processing_time):.1f} chunks/second")
            
            if timeout_manager.is_timeout_approaching():
                return ["Timeout during embedding generation"] * len(questions)
            
            # Process each question
            timeout_manager.start_stage("question_processing")
            question_timings = []
            
            # Log remaining time before questions processing
            remaining_before_questions = timeout_manager.get_remaining_time()
            logger.info(f"Starting question processing with {remaining_before_questions:.2f}s remaining")
            logger.info(f"Processing {len(questions)} questions, approx {remaining_before_questions/max(1, len(questions)):.2f}s per question")
            
            for i, question in enumerate(questions):
                q_start = time.perf_counter()
                question_id = f"Q{i+1}"
                logger.info(f"[{question_id}] Processing: '{question[:50]}...' at {timeout_manager.get_elapsed_time():.2f}s elapsed")
                
                if timeout_manager.is_partial_timeout():
                    # Return partial results if we've reached 28 seconds
                    elapsed = timeout_manager.get_elapsed_time()
                    logger.warning(f"⚠️ Partial timeout at {elapsed:.2f}s, processed {len(answers)}/{len(questions)} questions")
                    logger.warning(f"⏱️ Time per question: {elapsed/max(1, len(answers)):.2f}s, remaining time: {timeout_manager.get_remaining_time():.2f}s")
                    remaining_questions = len(questions) - len(answers)
                    answers.extend([""] * remaining_questions)
                    break
                
                # Retrieve relevant chunks
                retrieval_start = time.perf_counter()
                relevant_chunks = await self.retrieve_relevant_chunks(
                    question, chunks, chunk_embeddings, timeout_manager
                )
                retrieval_time = time.perf_counter() - retrieval_start
                
                # Generate answer
                generation_start = time.perf_counter()
                answer = await self.generate_answer(question, relevant_chunks, timeout_manager)
                generation_time = time.perf_counter() - generation_start
                answers.append(answer)
                
                # Calculate and log question timing
                q_elapsed = time.perf_counter() - q_start
                question_timings.append({
                    'question_idx': i,
                    'question': question[:30] + "...",
                    'retrieval_time': round(retrieval_time, 2),
                    'generation_time': round(generation_time, 2),
                    'total_time': round(q_elapsed, 2),
                    'chunks_retrieved': len(relevant_chunks),
                    'answer_length': len(answer) if answer else 0,
                    'elapsed_at_completion': round(timeout_manager.get_elapsed_time(), 2)
                })
                
                # Detailed timing log
                logger.info(f"[{question_id}] Completed in {q_elapsed:.2f}s → retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s")
                logger.info(f"[{question_id}] Retrieved {len(relevant_chunks)} chunks, answer length: {len(answer) if answer else 0}")
                
                # Early warning if we're taking too long per question
                remaining = timeout_manager.get_remaining_time()
                questions_left = len(questions) - (i + 1)
                if questions_left > 0:
                    time_per_q = timeout_manager.get_elapsed_time() / (i + 1)
                    estimated_time_needed = time_per_q * questions_left
                    if estimated_time_needed > remaining:
                        logger.warning(f"⚠️ Time warning: {estimated_time_needed:.1f}s needed for remaining {questions_left} questions, but only {remaining:.1f}s left")
            
            # End question processing timing
            timeout_manager.end_stage()
            
            # Update performance stats
            self.performance_stats['requests_processed'] += 1
            self.performance_stats['questions_processed'] = self.performance_stats.get('questions_processed', 0) + len(questions)
            self.performance_stats['questions_completed'] = self.performance_stats.get('questions_completed', 0) + len([a for a in answers if a])
            
            # Create detailed timing and accuracy report
            total_elapsed = timeout_manager.get_elapsed_time()
            completed_questions = len([a for a in answers if a])
            answer_lengths = [len(a) for a in answers if a]
            
            # Calculate answer quality metrics
            avg_answer_length = sum(answer_lengths) / max(1, len(answer_lengths)) if answer_lengths else 0
            completion_rate = (completed_questions / len(questions)) * 100 if questions else 0
            questions_per_second = completed_questions / total_elapsed if total_elapsed > 0 else 0
            
            # Summary stats with enhanced formatting
            logger.info("=" * 60)
            logger.info(f"📊 PROCESSING SUMMARY")
            logger.info(f"🎯 Accuracy: {len(questions)} questions | {completed_questions} completed ({completion_rate:.1f}%)")
            logger.info(f"⏱️ Timing: {total_elapsed:.2f}s/{RESPONSE_TIMEOUT}s ({(total_elapsed/RESPONSE_TIMEOUT)*100:.1f}% of timeout)")
            logger.info(f"📈 Performance: {questions_per_second:.2f} q/s | Avg answer: {avg_answer_length:.0f} chars")
            
            # Memory stats
            current_mem = get_memory_usage()
            logger.info(f"🧠 Memory: {current_mem:.2f}GB/{MAX_MEMORY_GB}GB ({(current_mem/MAX_MEMORY_GB)*100:.1f}%)")
            
            # Stage breakdown with enhanced formatting
            logger.info("-" * 60)
            logger.info("⏱️ PROCESSING STAGES BREAKDOWN")
            
            stages = timeout_manager.get_timing_report()['stages']
            total_stages_time = sum(stages.values())
            
            # Sort stages by duration
            sorted_stages = sorted(stages.items(), key=lambda x: x[1], reverse=True)
            for stage, duration in sorted_stages:
                percentage = (duration / total_elapsed) * 100
                bar_length = int(percentage / 5)  # 20 chars = 100%
                progress_bar = "█" * bar_length + "░" * (20 - bar_length)
                logger.info(f"  {stage:20s} │ {progress_bar} │ {duration:.2f}s ({percentage:.1f}%)")
            
            # Question stats with detailed analysis
            if question_timings:
                logger.info("-" * 60)
                logger.info("📝 QUESTION PROCESSING ANALYSIS")
                
                retrieval_times = [q['retrieval_time'] for q in question_timings]
                generation_times = [q['generation_time'] for q in question_timings]
                total_times = [q['total_time'] for q in question_timings]
                
                avg_retrieval = sum(retrieval_times) / len(retrieval_times)
                avg_generation = sum(generation_times) / len(generation_times)
                avg_total = sum(total_times) / len(total_times)
                
                # Calculate time distribution
                retrieval_pct = (avg_retrieval / avg_total) * 100 if avg_total > 0 else 0
                generation_pct = (avg_generation / avg_total) * 100 if avg_total > 0 else 0
                
                logger.info(f"  Average time per question: {avg_total:.2f}s")
                logger.info(f"  ├─ Retrieval:   {avg_retrieval:.2f}s ({retrieval_pct:.1f}%)")
                logger.info(f"  └─ Generation: {avg_generation:.2f}s ({generation_pct:.1f}%)")
                
                # Find fastest and slowest questions
                if len(question_timings) > 1:
                    fastest_q = min(question_timings, key=lambda q: q['total_time'])
                    slowest_q = max(question_timings, key=lambda q: q['total_time'])
                    
                    logger.info(f"  Fastest Q{fastest_q['question_idx']+1}: {fastest_q['total_time']:.2f}s | '{fastest_q['question']}'")
                    logger.info(f"  Slowest Q{slowest_q['question_idx']+1}: {slowest_q['total_time']:.2f}s | '{slowest_q['question']}'")
                
                # Accuracy completion analysis
                if completed_questions < len(questions):
                    remaining_time = RESPONSE_TIMEOUT - total_elapsed
                    remaining_questions = len(questions) - completed_questions
                    est_time_needed = avg_total * remaining_questions
                    
                    logger.info("-" * 60)
                    logger.info(f"⚠️ INCOMPLETE PROCESSING: {completed_questions}/{len(questions)} questions processed")
                    logger.info(f"  Remaining questions: {remaining_questions}")
                    logger.info(f"  Estimated time needed: {est_time_needed:.2f}s")
                    logger.info(f"  Time available: {remaining_time:.2f}s")
                    logger.info(f"  Time deficit: {est_time_needed - remaining_time:.2f}s")
                    
                    # Make recommendations for better performance
                    if est_time_needed > RESPONSE_TIMEOUT:
                        logger.info("  💡 RECOMMENDATION: Consider breaking document into smaller parts")
                        logger.info("  💡 RECOMMENDATION: Limit number of questions per request")
            
            # Save performance metrics for future analysis
            completion_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            perf_metrics = {
                'timestamp': completion_timestamp,
                'questions_total': len(questions),
                'questions_completed': completed_questions,
                'completion_rate': completion_rate,
                'total_time': total_elapsed,
                'avg_time_per_question': avg_total if 'avg_total' in locals() else None,
                'avg_retrieval_time': avg_retrieval if 'avg_retrieval' in locals() else None,
                'avg_generation_time': avg_generation if 'avg_generation' in locals() else None,
                'memory_usage': current_mem,
                'memory_usage_pct': (current_mem/MAX_MEMORY_GB)*100,
                'stages': stages
            }
            self.performance_stats['recent_metrics'] = perf_metrics
            
            logger.info("=" * 60)
            
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
    startup_start = time.time()
    await asyncio.get_event_loop().run_in_executor(global_executor, rag_system.load_models)
    startup_time = time.time() - startup_start
    logger.info(f"System ready for processing in {startup_time:.2f}s")
    logger.info(f"Memory usage: {get_memory_usage():.2f}GB/{MAX_MEMORY_GB}GB")
    
    yield
    
    logger.info("Shutting down system...")
    # Gracefully shutdown all executors
    for executor, name in [
        (global_executor, "main"),
        (embedding_executor, "embedding"),
        (chunking_executor, "chunking")
    ]:
        logger.info(f"Shutting down {name} executor...")
        executor.shutdown(wait=False)
    
    # Final cleanup
    force_memory_cleanup()

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
EXPECTED_TOKEN = os.getenv('API_TOKEN', '6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca')

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
    
    # Generate unique request ID for tracking
    request_id = f"req_{int(time.time() * 1000)}"
    
    # Single request processing - check if system is busy
    if processing_lock.locked():
        logger.warning(f"[{request_id}] System busy - request rejected")
        raise HTTPException(
            status_code=503, 
            detail="System is currently processing another request. Please try again later."
        )
    
    # Create request tracking logs
    start_time = time.time()
    request_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    document_url = request.documents.split('?')[0] if '?' in request.documents else request.documents
    document_type = "Unknown"
    
    # Try to identify document type from URL
    if "policy" in document_url.lower():
        document_type = "Policy"
    elif "newton" in document_url.lower() or "principia" in document_url.lower():
        document_type = "Scientific"
    
    # Log request details with enhanced format
    logger.info("=" * 60)
    logger.info(f"📥 NEW REQUEST [{request_id}] at {request_timestamp}")
    logger.info(f"📄 Document: {document_type} | {document_url}")
    logger.info(f"❓ Questions: {len(request.questions)}")
    
    # Sample first few questions for context
    if request.questions:
        sample_questions = request.questions[:min(3, len(request.questions))]
        for i, q in enumerate(sample_questions):
            logger.info(f"   Q{i+1}: {q[:60]}..." if len(q) > 60 else f"   Q{i+1}: {q}")
        if len(request.questions) > 3:
            logger.info(f"   ...and {len(request.questions)-3} more questions")
    
    # Check system status
    initial_memory = get_memory_usage()
    logger.info(f"🧠 Memory: {initial_memory:.2f}GB/{MAX_MEMORY_GB}GB ({(initial_memory/MAX_MEMORY_GB)*100:.1f}%)")
    
    # Memory check with enhanced logging
    if initial_memory > MAX_MEMORY_GB * 0.95:
        memory_pct = (initial_memory/MAX_MEMORY_GB)*100
        logger.error(f"❌ [{request_id}] Memory critical: {initial_memory:.2f}GB ({memory_pct:.1f}%)")
        force_memory_cleanup()
        
        # Check if cleanup helped
        new_memory = get_memory_usage()
        if new_memory > MAX_MEMORY_GB * 0.9:
            logger.error(f"❌ [{request_id}] Memory still too high after cleanup: {new_memory:.2f}GB")
            raise HTTPException(status_code=503, 
                               detail=f"Server memory at capacity: {memory_pct:.1f}% used. Please try again later.")
        else:
            logger.info(f"✅ [{request_id}] Memory cleanup successful: {initial_memory:.2f}GB → {new_memory:.2f}GB")
            initial_memory = new_memory
    
    # Track errors for better diagnostics
    error_info = None
    
    try:
        async with processing_lock:
            logger.info(f"🔒 [{request_id}] Processing lock acquired")
            
            # Time tracking
            request_start = time.perf_counter()
            processing_start = time.time()
            
            # Main processing
            answers = await rag_system.process_document_and_questions(
                request.documents, request.questions
            )
            
            # Performance metrics
            request_duration = time.perf_counter() - request_start
            wall_time = time.time() - start_time
            
            # Memory stats
            final_memory = get_memory_usage()
            memory_change = final_memory - initial_memory
            memory_change_pct = (memory_change / initial_memory) * 100 if initial_memory > 0 else 0
            
            # Answer quality metrics
            answers_provided = len([a for a in answers if a])
            completion_rate = (answers_provided / len(request.questions)) * 100 if request.questions else 0
            avg_answer_len = sum(len(a) for a in answers if a) / max(1, answers_provided) if answers_provided else 0
            
            # Enhanced completion logging
            logger.info("=" * 60)
            logger.info(f"✅ REQUEST COMPLETED [{request_id}]")
            logger.info(f"⏱️ Timing: {wall_time:.2f}s total | {request_duration:.2f}s processing time")
            logger.info(f"📊 Results: {answers_provided}/{len(request.questions)} answers ({completion_rate:.1f}%)")
            logger.info(f"📝 Answers: {avg_answer_len:.0f} chars average length")
            
            # Memory usage visualization
            mem_pct_before = (initial_memory/MAX_MEMORY_GB)*100
            mem_pct_after = (final_memory/MAX_MEMORY_GB)*100
            mem_bar_before = "█" * int(mem_pct_before/5) + "░" * (20 - int(mem_pct_before/5))
            mem_bar_after = "█" * int(mem_pct_after/5) + "░" * (20 - int(mem_pct_after/5))
            
            logger.info(f"🧠 Memory Before: {mem_bar_before} {initial_memory:.2f}GB ({mem_pct_before:.1f}%)")
            logger.info(f"🧠 Memory After:  {mem_bar_after} {final_memory:.2f}GB ({mem_pct_after:.1f}%)")
            logger.info(f"🧠 Memory Change: {memory_change:.2f}GB ({memory_change_pct:+.1f}%)")
            
            # Performance tracking
            questions_per_second = len(request.questions) / wall_time if wall_time > 0 else 0
            logger.info(f"⚙️ Processing speed: {questions_per_second:.2f} questions/second")
            
            # Cleanup
            cleanup_start = time.time()
            force_memory_cleanup()
            cleanup_time = time.time() - cleanup_start
            
            logger.info(f"🧹 Memory cleanup: {cleanup_time:.2f}s")
            logger.info(f"🔓 Processing lock released")
            logger.info("=" * 60)
            
            # Store request stats
            rag_system.performance_stats['last_request'] = {
                'request_id': request_id,
                'timestamp': request_timestamp,
                'document_type': document_type,
                'questions_count': len(request.questions),
                'answers_count': answers_provided,
                'completion_rate': completion_rate,
                'processing_time': wall_time,
                'memory_usage': {
                    'before': initial_memory,
                    'after': final_memory,
                    'change': memory_change,
                    'change_pct': memory_change_pct
                }
            }
            
            return QueryResponse(answers=answers)
            
    except Exception as e:
        # Enhanced error logging
        error_time = time.time() - start_time
        error_info = {
            'error_type': type(e).__name__,
            'error_msg': str(e),
            'time_to_error': error_time
        }
        
        logger.error(f"❌ [{request_id}] Request failed after {error_time:.2f}s: {type(e).__name__}: {str(e)}")
        
        # Include stack trace for better debugging
        import traceback
        trace = traceback.format_exc()
        logger.error(f"Stack trace:\n{trace}")
        
        # Force cleanup on error
        force_memory_cleanup()
        
        # Track error in performance stats
        if not hasattr(rag_system.performance_stats, 'errors'):
            rag_system.performance_stats['errors'] = []
            
        rag_system.performance_stats['errors'].append({
            'request_id': request_id,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'error_type': type(e).__name__,
            'error_msg': str(e),
            'time_to_error': error_time
        })
        
        # Return appropriate error response
        if isinstance(e, HTTPException):
            raise e
        elif "memory" in str(e).lower():
            raise HTTPException(status_code=503, detail=f"Memory error: {str(e)}")
        elif "timeout" in str(e).lower():
            raise HTTPException(status_code=504, detail=f"Processing timeout: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Ensure lock is released in case of error in the async context
        if processing_lock.locked():
            logger.warning(f"[{request_id}] Forcing lock release after error")
            # We can't release the lock from outside the context
            # But we can log this situation

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "memory_usage_gb": get_memory_usage(),
        "memory_limit_gb": MAX_MEMORY_GB,
        "model_loaded": rag_system.embedding_model is not None,
        "system_busy": processing_lock.locked()
    }

# System status endpoint
@app.get("/status")
async def system_status():
    """System status endpoint"""
    return {
        "system_busy": processing_lock.locked(),
        "memory_usage_gb": get_memory_usage(),
        "memory_limit_gb": MAX_MEMORY_GB,
        "requests_processed": rag_system.performance_stats.get('requests_processed', 0),
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
        reload=False,  # Disable reload in production for better performance
        workers=1,
        loop="asyncio",
        log_level="info",
        timeout_keep_alive=60,  # Longer keep-alive for stability
        access_log=True,
        use_colors=True,  # Make logs more readable
        limit_concurrency=2  # Limit concurrent connections for stability
    )
