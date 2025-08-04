#!/usr/bin/env python3
"""
ENHANCED PRODUCTION-GRADE RAG SYSTEM (v9) - JINAAI OPTIMIZED FOR MAXIMUM SPEED
- jinaai/jina-embedding-b-en-v1 model with GPU acceleration
- Advanced batch processing with dynamic sizing
- Memory-efficient processing with garbage collection
- Multi-level caching system (embedding + query cache)
- Parallel processing with optimized thread pools
- Smart chunking with content-aware sizing
"""

# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import os
import gc
import asyncio
import aiohttp
import fitz
import re
import time
import logging
import hashlib
import math
import uvicorn
from typing import List, Dict, Any, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import threading
from functools import lru_cache

import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import torch
from transformers import AutoTokenizer

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# Redis imports for advanced caching
import redis.asyncio as redis
import json
import pickle
import base64

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Secrets and Critical Configurations ---
GOOGLE_API_KEY = 'AIzaSyB1pi8BtnI-yAzmx6DLpUmwj5TPYjHmhJ0'
EXPECTED_TOKEN = "6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca"

if not GOOGLE_API_KEY:
    logger.critical("FATAL: GOOGLE_API_KEY environment variable not set.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# Optimized thread pool configuration
CPU_COUNT = os.cpu_count() or 4
global_executor = ThreadPoolExecutor(max_workers=min(CPU_COUNT * 2, 16))

# Redis Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
REDIS_EXPIRE_TIME = int(os.getenv('REDIS_EXPIRE_TIME', 86400))  # 24 hours default
redis_client = None

# Advanced device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    # Get GPU memory info for optimization
    GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"🔥 GPU detected: {torch.cuda.get_device_name(0)} ({GPU_MEMORY_GB:.1f}GB)")
else:
    GPU_MEMORY_GB = 0

logger.info(f"🔥 Using device: {DEVICE} | CPU cores: {CPU_COUNT}")


# 2. ENHANCED DATA STRUCTURES FOR HIERARCHICAL CHUNKING
# ==============================================================================
class ChunkType(Enum):
    DOCUMENT = "document"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"

@dataclass
class DocumentElement:
    """Represents a hierarchical element in the document structure"""
    element_id: str
    element_type: ChunkType
    title: str
    content: str
    page_numbers: List[int]
    start_char: int
    end_char: int
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    level: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}


# 3. REDIS-ENHANCED MULTI-LEVEL CACHING SYSTEM
# ==============================================================================

async def get_redis_client() -> redis.Redis:
    """Get or create Redis client with connection pooling"""
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=False,  # Keep binary for numpy arrays
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            await redis_client.ping()
            logger.info(f"✅ Redis connected successfully at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Using memory-only cache.")
            redis_client = None
    return redis_client

class RedisEmbeddingCache:
    """Hybrid Redis + Memory caching system for embeddings with intelligent fallback"""
    
    def __init__(self, max_memory_size: int = 10000):
        # Memory cache for ultra-fast access
        self.memory_cache = {}
        self.memory_access_count = {}
        self.memory_access_order = []
        self.max_memory_size = max_memory_size
        self.lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            'memory_hits': 0,
            'redis_hits': 0,
            'misses': 0,
            'redis_errors': 0,
            'total_sets': 0
        }
    
    def _get_hash(self, text: str) -> str:
        """Generate consistent hash for text content"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy array for Redis storage"""
        return base64.b64encode(pickle.dumps(embedding)).decode('utf-8').encode()
    
    def _deserialize_embedding(self, data: bytes) -> np.ndarray:
        """Deserialize numpy array from Redis"""
        return pickle.loads(base64.b64decode(data.decode('utf-8')))
    
    async def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding with hybrid memory+Redis lookup"""
        text_hash = self._get_hash(text)
        
        # 1. Check memory cache first (fastest)
        with self.lock:
            if text_hash in self.memory_cache:
                self.memory_access_count[text_hash] = self.memory_access_count.get(text_hash, 0) + 1
                if text_hash in self.memory_access_order:
                    self.memory_access_order.remove(text_hash)
                self.memory_access_order.append(text_hash)
                self.stats['memory_hits'] += 1
                return self.memory_cache[text_hash].copy()
        
        # 2. Check Redis cache (persistent)
        try:
            redis_conn = await get_redis_client()
            if redis_conn:
                redis_key = f"embedding:{text_hash}"
                cached_data = await redis_conn.get(redis_key)
                if cached_data:
                    embedding = self._deserialize_embedding(cached_data)
                    # Store in memory cache for future ultra-fast access
                    await self._set_memory_cache(text_hash, embedding)
                    self.stats['redis_hits'] += 1
                    return embedding
        except Exception as e:
            logger.debug(f"Redis get error: {e}")
            self.stats['redis_errors'] += 1
        
        # 3. Cache miss
        self.stats['misses'] += 1
        return None
    
    async def set(self, text: str, embedding: np.ndarray):
        """Store embedding in both memory and Redis"""
        text_hash = self._get_hash(text)
        self.stats['total_sets'] += 1
        
        # 1. Store in memory cache
        await self._set_memory_cache(text_hash, embedding)
        
        # 2. Store in Redis for persistence
        try:
            redis_conn = await get_redis_client()
            if redis_conn:
                redis_key = f"embedding:{text_hash}"
                serialized_data = self._serialize_embedding(embedding)
                await redis_conn.setex(redis_key, REDIS_EXPIRE_TIME, serialized_data)
        except Exception as e:
            logger.debug(f"Redis set error: {e}")
            self.stats['redis_errors'] += 1
    
    async def _set_memory_cache(self, text_hash: str, embedding: np.ndarray):
        """Manage memory cache with intelligent eviction"""
        with self.lock:
            # Evict if memory cache is full
            if len(self.memory_cache) >= self.max_memory_size and text_hash not in self.memory_cache:
                await self._evict_memory_items(int(self.max_memory_size * 0.1))
            
            self.memory_cache[text_hash] = embedding.copy()
            self.memory_access_count[text_hash] = self.memory_access_count.get(text_hash, 0) + 1
            if text_hash not in self.memory_access_order:
                self.memory_access_order.append(text_hash)
    
    async def _evict_memory_items(self, num_items: int):
        """Intelligent memory cache eviction"""
        if not self.memory_access_order:
            return
        
        # Sort by access count (ascending) - evict least accessed first
        candidates = []
        for text_hash in self.memory_access_order:
            if text_hash in self.memory_cache:
                score = self.memory_access_count.get(text_hash, 1)
                candidates.append((score, text_hash))
        
        candidates.sort()  # Least accessed first
        
        for _, text_hash in candidates[:num_items]:
            if text_hash in self.memory_cache:
                del self.memory_cache[text_hash]
            if text_hash in self.memory_access_count:
                del self.memory_access_count[text_hash]
            if text_hash in self.memory_access_order:
                self.memory_access_order.remove(text_hash)
    
    async def clear(self):
        """Clear all caches"""
        # Clear memory cache
        with self.lock:
            self.memory_cache.clear()
            self.memory_access_count.clear()
            self.memory_access_order.clear()
        
        # Clear Redis cache
        try:
            redis_conn = await get_redis_client()
            if redis_conn:
                # Delete all embedding keys
                keys = await redis_conn.keys("embedding:*")
                if keys:
                    await redis_conn.delete(*keys)
                logger.info("🗑️ Redis embedding cache cleared")
        except Exception as e:
            logger.debug(f"Redis clear error: {e}")
            self.stats['redis_errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.stats['memory_hits'] + self.stats['redis_hits'] + self.stats['misses']
        memory_hit_rate = (self.stats['memory_hits'] / max(1, total_requests)) * 100
        redis_hit_rate = (self.stats['redis_hits'] / max(1, total_requests)) * 100
        overall_hit_rate = ((self.stats['memory_hits'] + self.stats['redis_hits']) / max(1, total_requests)) * 100
        
        with self.lock:
            total_memory_accesses = sum(self.memory_access_count.values())
            avg_memory_access = total_memory_accesses / max(len(self.memory_access_count), 1)
            
            return {
                "memory_cache_size": len(self.memory_cache),
                "max_memory_size": self.max_memory_size,
                "memory_hit_rate": round(memory_hit_rate, 2),
                "redis_hit_rate": round(redis_hit_rate, 2),
                "overall_hit_rate": round(overall_hit_rate, 2),
                "total_requests": total_requests,
                "redis_errors": self.stats['redis_errors'],
                "avg_memory_access": round(avg_memory_access, 2),
                **self.stats
            }


class RedisQueryCache:
    """Redis-backed query cache with memory buffer"""
    
    def __init__(self, max_memory_size: int = 500):
        self.memory_cache = {}
        self.memory_access_order = []
        self.max_memory_size = max_memory_size
        self.lock = threading.RLock()
        
        self.stats = {
            'memory_hits': 0,
            'redis_hits': 0,
            'misses': 0,
            'redis_errors': 0
        }
    
    def _get_key(self, query: str, context_hash: str) -> str:
        combined = f"{query}||{context_hash}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    async def get(self, query: str, context_hash: str) -> Optional[str]:
        key = self._get_key(query, context_hash)
        
        # Check memory cache first
        with self.lock:
            if key in self.memory_cache:
                self.memory_access_order.remove(key)
                self.memory_access_order.append(key)
                self.stats['memory_hits'] += 1
                return self.memory_cache[key]
        
        # Check Redis cache
        try:
            redis_conn = await get_redis_client()
            if redis_conn:
                redis_key = f"query:{key}"
                cached_answer = await redis_conn.get(redis_key)
                if cached_answer:
                    answer = cached_answer.decode('utf-8')
                    # Store in memory for fast future access
                    await self._set_memory_cache(key, answer)
                    self.stats['redis_hits'] += 1
                    return answer
        except Exception as e:
            logger.debug(f"Redis query get error: {e}")
            self.stats['redis_errors'] += 1
        
        self.stats['misses'] += 1
        return None
    
    async def set(self, query: str, context_hash: str, answer: str):
        key = self._get_key(query, context_hash)
        
        # Store in memory
        await self._set_memory_cache(key, answer)
        
        # Store in Redis
        try:
            redis_conn = await get_redis_client()
            if redis_conn:
                redis_key = f"query:{key}"
                await redis_conn.setex(redis_key, REDIS_EXPIRE_TIME, answer.encode('utf-8'))
        except Exception as e:
            logger.debug(f"Redis query set error: {e}")
            self.stats['redis_errors'] += 1
    
    async def _set_memory_cache(self, key: str, answer: str):
        with self.lock:
            if len(self.memory_cache) >= self.max_memory_size and key not in self.memory_cache:
                oldest = self.memory_access_order.pop(0)
                del self.memory_cache[oldest]
            
            self.memory_cache[key] = answer
            if key not in self.memory_access_order:
                self.memory_access_order.append(key)
    
    async def clear(self):
        # Clear memory cache
        with self.lock:
            self.memory_cache.clear()
            self.memory_access_order.clear()
        
        # Clear Redis cache
        try:
            redis_conn = await get_redis_client()
            if redis_conn:
                keys = await redis_conn.keys("query:*")
                if keys:
                    await redis_conn.delete(*keys)
                logger.info("🗑️ Redis query cache cleared")
        except Exception as e:
            logger.debug(f"Redis query clear error: {e}")
            self.stats['redis_errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.stats['memory_hits'] + self.stats['redis_hits'] + self.stats['misses']
        hit_rate = ((self.stats['memory_hits'] + self.stats['redis_hits']) / max(1, total_requests)) * 100
        
        return {
            "memory_cache_size": len(self.memory_cache),
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests,
            **self.stats
        }


# 4. ENHANCED SEMANTIC PROCESSOR CLASS
# ==============================================================================
class HierarchicalSemanticProcessor:
    """Advanced semantic processor with content-aware chunking"""
    
    def __init__(self):
        # Enhanced pattern matching for document structure
        self.heading_patterns = [
            # Chapter patterns
            (r'^(Chapter\s+\d+[:\.]?\s*.*?)$', ChunkType.CHAPTER, 1),
            (r'^(CHAPTER\s+[IVXLCDM]+[:\.]?\s*.*?)$', ChunkType.CHAPTER, 1),
            (r'^(\d+\.\s+[A-Z][^.]*?)$', ChunkType.CHAPTER, 1),
            
            # Section patterns
            (r'^(\d+\.\d+\s+[A-Z][^.]*?)$', ChunkType.SECTION, 2),
            (r'^(Section\s+\d+[:\.]?\s*.*?)$', ChunkType.SECTION, 2),
            (r'^([A-Z][A-Z\s]{3,}[^a-z]*?)$', ChunkType.SECTION, 2),
            
            # Subsection patterns
            (r'^(\d+\.\d+\.\d+\s+[A-Z][^.]*?)$', ChunkType.SUBSECTION, 3),
            (r'^([a-z]\)\s+.*?)$', ChunkType.SUBSECTION, 3),
            (r'^([A-Z]\.\s+.*?)$', ChunkType.SUBSECTION, 3),
        ]
        
        self.sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        self.paragraph_separators = ['\n\n', '\n\r\n', '\r\n\r\n']
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern, re.MULTILINE), chunk_type, level)
            for pattern, chunk_type, level in self.heading_patterns
        ]

    @lru_cache(maxsize=100)
    def analyze_document_structure(self, text_hash: str, word_count: int, text_sample: str) -> Dict[str, Any]:
        """Analyze document structure with caching"""
        if word_count == 0:
            return {"document_type": "article", "size_category": "standard", "structure_complexity": "simple"}
        
        # Use sample for analysis to improve performance
        text = text_sample[:10000]  # Use first 10k chars for analysis
        
        # Count structural elements
        indicators = {
            "chapters": len(re.findall(r'\b[Cc]hapter\s+\d+', text)),
            "sections": len(re.findall(r'\b[Ss]ection\s+\d+', text)),
            "subsections": len(re.findall(r'\d+\.\d+\.\d+', text)),
            "numbered_lists": len(re.findall(r'^\d+\.\s+', text, re.MULTILINE)),
            "bullet_points": len(re.findall(r'^[•\-\*]\s+', text, re.MULTILINE)),
            "headings": len(re.findall(r'^[A-Z][A-Z\s]{3,}$', text, re.MULTILINE)),
        }
        
        # Determine document type
        doc_type = "article"
        structure_complexity = "simple"
        
        text_lower = text.lower()
        if "whereas" in text_lower or "hereby" in text_lower or indicators["sections"] > 10:
            doc_type = "legal"
            structure_complexity = "complex"
        elif "abstract" in text_lower[:1000] or "introduction" in text_lower[:1500]:
            doc_type = "academic"
            structure_complexity = "moderate"
        elif indicators["chapters"] > 0:
            doc_type = "book"
            structure_complexity = "complex"
        elif indicators["sections"] > 5 or indicators["subsections"] > 10:
            structure_complexity = "moderate"
        
        size_category = "large" if word_count > 100000 else "medium" if word_count > 20000 else "standard"
        
        return {
            "document_type": doc_type,
            "size_category": size_category,
            "structure_complexity": structure_complexity,
            "structural_indicators": indicators,
            "word_count": word_count
        }

    def extract_hierarchical_structure(self, pages: List[Tuple[int, str]], request_id: str) -> List[DocumentElement]:
        """Extract hierarchical document structure with optimized processing"""
        # Combine all pages with character mapping
        full_text, char_to_page_map = "", {}
        for page_num, page_text in pages:
            start_char = len(full_text)
            clean_page_text = page_text + "\n\n"
            full_text += clean_page_text
            for i in range(start_char, len(full_text), max(1, len(clean_page_text) // 100)):
                char_to_page_map[i] = page_num
        
        logger.info(f"[{request_id}] 🔍 Analyzing document structure...")
        
        # Create hash for caching analysis
        text_hash = hashlib.sha256(full_text[:10000].encode()).hexdigest()[:16]
        word_count = len(full_text.split())
        text_sample = full_text[:10000]
        
        analysis = self.analyze_document_structure(text_hash, word_count, text_sample)
        logger.info(f"[{request_id}] 📊 Document analysis: {analysis}")
        
        # Parallel heading detection for large documents
        if len(full_text) > 100000:
            headings = self._parallel_heading_detection(full_text, char_to_page_map)
        else:
            headings = self._sequential_heading_detection(full_text, char_to_page_map)
        
        # Create hierarchical structure
        elements = self._build_hierarchy(headings, full_text, char_to_page_map, analysis, pages)
        
        logger.info(f"[{request_id}] 🏗️ Created {len(elements)} hierarchical elements")
        return elements

    def _parallel_heading_detection(self, full_text: str, char_to_page_map: Dict[int, int]) -> List[Dict]:
        """Parallel heading detection for large documents"""
        chunk_size = len(full_text) // CPU_COUNT
        chunks = []
        
        for i in range(0, len(full_text), chunk_size):
            end_pos = min(i + chunk_size + 1000, len(full_text))  # Overlap for boundary headings
            chunks.append((i, full_text[i:end_pos]))
        
        all_headings = []
        with ThreadPoolExecutor(max_workers=CPU_COUNT) as thread_executor:
            futures = []
            for start_pos, chunk in chunks:
                future = thread_executor.submit(self._detect_headings_in_chunk, chunk, start_pos, char_to_page_map)
                futures.append(future)
            
            for future in as_completed(futures):
                headings = future.result()
                all_headings.extend(headings)
        
        # Remove duplicates and sort
        unique_headings = {}
        for heading in all_headings:
            key = (heading['start'], heading['title'])
            if key not in unique_headings or heading['level'] < unique_headings[key]['level']:
                unique_headings[key] = heading
        
        return sorted(unique_headings.values(), key=lambda x: x['start'])

    def _sequential_heading_detection(self, full_text: str, char_to_page_map: Dict[int, int]) -> List[Dict]:
        """Sequential heading detection for smaller documents"""
        return self._detect_headings_in_chunk(full_text, 0, char_to_page_map)

    def _detect_headings_in_chunk(self, text: str, offset: int, char_to_page_map: Dict[int, int]) -> List[Dict]:
        """Detect headings in a text chunk"""
        headings = []
        for pattern, chunk_type, level in self.compiled_patterns:
            for match in pattern.finditer(text):
                start_pos = match.start() + offset
                end_pos = match.end() + offset
                title = match.group(1).strip()
                page_num = char_to_page_map.get(start_pos, 1)
                
                headings.append({
                    'title': title,
                    'type': chunk_type,
                    'level': level,
                    'start': start_pos,
                    'end': end_pos,
                    'page': page_num
                })
        
        return headings

    def _build_hierarchy(self, headings: List[Dict], full_text: str, char_to_page_map: Dict[int, int],
                        analysis: Dict[str, Any], pages: List[Tuple[int, str]]) -> List[DocumentElement]:
        """Build hierarchical structure from detected headings"""
        elements = []
        element_counter = 0
        
        # Create root document element
        root_element = DocumentElement(
            element_id=f"doc_0",
            element_type=ChunkType.DOCUMENT,
            title="Document Root",
            content="",
            page_numbers=list(range(1, len(pages) + 1)),
            start_char=0,
            end_char=len(full_text),
            level=0,
            metadata=analysis
        )
        elements.append(root_element)
        element_counter += 1
        
        # Process headings to create hierarchy
        current_parents = {0: "doc_0"}  # level -> parent_id mapping
        
        for i, heading in enumerate(headings):
            # Determine content boundaries
            content_start = heading['end']
            content_end = headings[i + 1]['start'] if i + 1 < len(headings) else len(full_text)
            
            # Extract content
            content = full_text[content_start:content_end].strip()
            
            # Get page numbers for this section
            section_pages = self._get_pages_for_range(heading['start'], content_end, char_to_page_map)
            
            # Find parent
            parent_id = self._find_parent_id(heading['level'], current_parents)
            
            # Create element
            element_id = f"elem_{element_counter}"
            element = DocumentElement(
                element_id=element_id,
                element_type=heading['type'],
                title=heading['title'],
                content=content,
                page_numbers=section_pages,
                start_char=heading['start'],
                end_char=content_end,
                parent_id=parent_id,
                level=heading['level'],
                metadata={
                    'heading_match': heading['title'],
                    'content_length': len(content),
                    'word_count': len(content.split())
                }
            )
            
            elements.append(element)
            
            # Update parent-child relationships
            parent_element = next((e for e in elements if e.element_id == parent_id), None)
            if parent_element:
                parent_element.children_ids.append(element_id)
            
            # Update current parents mapping
            current_parents[heading['level']] = element_id
            current_parents = {k: v for k, v in current_parents.items() if k <= heading['level']}
            
            element_counter += 1
        
        # Create paragraph-level chunks with content-aware sizing
        self._create_content_aware_chunks(elements, full_text, char_to_page_map, element_counter, analysis)
        
        return elements

    def _get_pages_for_range(self, start_char: int, end_char: int, char_to_page_map: Dict[int, int]) -> List[int]:
        """Get page numbers for character range"""
        pages = set()
        step = max(1, (end_char - start_char) // 50)
        for pos in range(start_char, end_char, step):
            pages.add(char_to_page_map.get(pos, 1))
        return sorted(list(pages))

    def _find_parent_id(self, level: int, current_parents: Dict[int, str]) -> str:
        """Find appropriate parent ID for given level"""
        for parent_level in range(level - 1, -1, -1):
            if parent_level in current_parents:
                return current_parents[parent_level]
        return "doc_0"

    def _create_content_aware_chunks(self, elements: List[DocumentElement], full_text: str,
                                   char_to_page_map: Dict[int, int], counter_start: int,
                                   analysis: Dict[str, Any]) -> None:
        """Create content-aware chunks with dynamic sizing"""
        element_counter = counter_start
        new_elements = []
        
        # Get optimal chunk size based on document characteristics
        base_chunk_size = self._get_optimal_chunk_size(analysis)
        
        for element in elements:
            if element.element_type in [ChunkType.SECTION, ChunkType.SUBSECTION, ChunkType.CHAPTER]:
                content = element.content
                
                # Dynamic chunk sizing based on content density
                content_density = self._calculate_content_density(content)
                adjusted_chunk_size = int(base_chunk_size * content_density)
                
                if len(content) > adjusted_chunk_size * 1.3:  # Only chunk if significantly larger
                    paragraphs = self._split_content_aware(
                        content, adjusted_chunk_size, element.start_char,
                        char_to_page_map, element.element_id, element_counter
                    )
                    
                    for para in paragraphs:
                        new_elements.append(para)
                        element.children_ids.append(para.element_id)
                        element_counter += 1
        
        elements.extend(new_elements)

    def _calculate_content_density(self, content: str) -> float:
        """Calculate content density to adjust chunk size"""
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if not non_empty_lines:
            return 1.0
        
        # Calculate various density metrics
        avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        line_density = len(non_empty_lines) / len(lines)
        
        # Adjust chunk size based on content characteristics
        if avg_line_length > 100:  # Dense technical content
            return 0.8
        elif avg_line_length < 30:  # Sparse content (lists, etc.)
            return 1.2
        else:
            return 1.0 * line_density

    def _split_content_aware(self, content: str, chunk_size: int, base_char_pos: int,
                           char_to_page_map: Dict[int, int], parent_id: str,
                           counter_start: int) -> List[DocumentElement]:
        """Split content with awareness of semantic boundaries"""
        paragraphs = []
        position = 0
        counter = counter_start
        overlap = max(50, chunk_size // 10)  # Dynamic overlap
        
        while position < len(content):
            end_pos = min(position + chunk_size, len(content))
            
            # Find optimal break point
            if end_pos < len(content):
                break_point = self._find_semantic_break(content, position, end_pos)
                end_pos = break_point if break_point > position else end_pos
            
            chunk_text = content[position:end_pos].strip()
            if chunk_text and len(chunk_text) > 50:  # Minimum chunk size
                chunk_start_char = base_char_pos + position
                chunk_end_char = base_char_pos + end_pos
                chunk_pages = self._get_pages_for_range(chunk_start_char, chunk_end_char, char_to_page_map)
                
                paragraph = DocumentElement(
                    element_id=f"para_{counter}",
                    element_type=ChunkType.PARAGRAPH,
                    title=f"Paragraph {counter - counter_start + 1}",
                    content=chunk_text,
                    page_numbers=chunk_pages,
                    start_char=chunk_start_char,
                    end_char=chunk_end_char,
                    parent_id=parent_id,
                    level=4,
                    metadata={
                        'chunk_index': counter - counter_start,
                        'word_count': len(chunk_text.split()),
                        'char_count': len(chunk_text),
                        'content_density': self._calculate_content_density(chunk_text)
                    }
                )
                paragraphs.append(paragraph)
                counter += 1
            
            # Move position with overlap consideration
            position = max(position + 1, end_pos - overlap)
        
        return paragraphs

    def _find_semantic_break(self, content: str, start: int, preferred_end: int) -> int:
        """Find the best semantic break point within a range"""
        search_start = max(start, preferred_end - 200)
        search_end = min(len(content), preferred_end + 100)
        
        # Priority order for break points
        break_patterns = [
            (r'\n\n', 2),      # Paragraph breaks (highest priority)
            (r'\. ', 1),       # Sentence endings
            (r'[.!?]\n', 1),   # Sentence endings with newline
            (r', ', 0.5),      # Comma breaks (lowest priority)
            (r' ', 0.1)        # Word boundaries (fallback)
        ]
        
        best_break = preferred_end
        best_score = 0
        
        for pattern, score in break_patterns:
            for match in re.finditer(pattern, content[search_start:search_end]):
                break_pos = search_start + match.end()
                distance_penalty = abs(break_pos - preferred_end) / 200
                final_score = score - distance_penalty
                
                if final_score > best_score:
                    best_score = final_score
                    best_break = break_pos
        
        return best_break

    def _get_optimal_chunk_size(self, analysis: Dict[str, Any]) -> int:
        """Determine optimal chunk size based on document analysis"""
        doc_type = analysis.get('document_type', 'article')
        complexity = analysis.get('structure_complexity', 'simple')
        size_category = analysis.get('size_category', 'standard')
        
        base_sizes = {
            'legal': 1800,
            'academic': 1600,
            'book': 1200,
            'article': 1000
        }
        
        complexity_multipliers = {
            'simple': 1.0,
            'moderate': 1.2,
            'complex': 1.4
        }
        
        size_multipliers = {
            'standard': 1.0,
            'medium': 1.1,
            'large': 1.3
        }
        
        base_size = base_sizes.get(doc_type, 1000)
        complexity_mult = complexity_multipliers.get(complexity, 1.0)
        size_mult = size_multipliers.get(size_category, 1.0)
        
        return int(base_size * complexity_mult * size_mult)

    def get_chunks_for_retrieval(self, elements: List[DocumentElement]) -> List[Dict[str, Any]]:
        """Convert hierarchical elements to chunks suitable for retrieval"""
        chunks = []
        
        for element in elements:
            # Skip document root and empty content
            if element.element_type == ChunkType.DOCUMENT or not element.content.strip():
                continue
            
            # Create enriched chunk with hierarchical context
            parent_context = ""
            if element.parent_id:
                parent = next((e for e in elements if e.element_id == element.parent_id), None)
                if parent and parent.title and parent.title != "Document Root":
                    parent_context = f"Section: {parent.title}\n"
            
            enriched_content = f"{parent_context}{element.title}\n\n{element.content}".strip()
            
            chunk = {
                'text': enriched_content,
                'page': element.page_numbers[0] if element.page_numbers else 1,
                'pages': element.page_numbers,
                'element_id': element.element_id,
                'element_type': element.element_type.value,
                'title': element.title,
                'level': element.level,
                'parent_id': element.parent_id,
                'metadata': element.metadata,
                'word_count': len(element.content.split()),
                'char_count': len(element.content)
            }
            chunks.append(chunk)
        
        return chunks


# 5. ULTRA-FAST RAG SYSTEM WITH JINAAI EMBEDDING
# ==============================================================================
class JinaAIOptimizedRAGSystem:
    def __init__(self):
        self.processor = HierarchicalSemanticProcessor()
        self.embedding_model = None
        self.tokenizer = None
        self.reranker_model = None
        
        # JinaAI Embedding Configuration
        self.embedding_model_name = 'jinaai/jina-embedding-s-en-v1'
        self.reranker_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        self.llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Retrieval parameters
        self.top_k_initial = 20
        self.top_k_final = 8
        
        # Advanced Redis-enabled caching system
        self.embedding_cache = RedisEmbeddingCache(max_memory_size=10000)
        self.query_cache = RedisQueryCache(max_memory_size=500)
        
        # Dynamic batch processing settings
        self.base_batch_size = 32
        self.max_batch_size = self._calculate_optimal_batch_size()
        self.enable_half_precision = DEVICE == "cuda" and GPU_MEMORY_GB > 4
        
        # Performance monitoring
        self.performance_stats = {
            'total_embeddings_computed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_batch_size': 0,
            'gpu_memory_usage': 0
        }

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available resources"""
        if DEVICE == "cuda":
            # Adjust batch size based on GPU memory
            if GPU_MEMORY_GB >= 16:
                return 128
            elif GPU_MEMORY_GB >= 8:
                return 64
            elif GPU_MEMORY_GB >= 4:
                return 32
            else:
                return 16
        else:
            # CPU-based processing
            return min(16, CPU_COUNT * 2)

    def load_models(self):
        """Load all ML models with JinaAI optimizations"""
        logger.info("--- Loading JinaAI Embedding Model with Advanced Optimizations ---")
        
        # Load JinaAI embedding model with optimizations
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=DEVICE
        )
        
        # Load tokenizer for advanced text preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name
        )
        
        # Apply optimizations for GPU
        if self.enable_half_precision:
            logger.info("🚀 Enabling half precision (FP16) for GPU acceleration")
            self.embedding_model = self.embedding_model.half()
        
        # Optimize model for inference
        if hasattr(self.embedding_model, 'eval'):
            self.embedding_model.eval()
        
        # Load reranker with optimizations
        self.reranker_model = CrossEncoder(self.reranker_model_name)
        
        # Model warmup with progressively larger batches
        logger.info("🔥 Warming up models with progressive batch sizes...")
        warmup_texts = [
            f"This is warmup text number {i} for testing the embedding model performance."
            for i in range(1, 17)
        ]
        
        # Progressive warmup
        for batch_size in [1, 4, 8, min(16, self.max_batch_size)]:
            batch = warmup_texts[:batch_size]
            _ = self.embedding_model.encode(
                batch, 
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
        
        # Warmup reranker
        _ = self.reranker_model.predict([["warmup query", "warmup text"]], show_progress_bar=False)
        
        # Force garbage collection after model loading
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"--- JinaAI Models loaded successfully | Max batch size: {self.max_batch_size} ---")

    def _extract_pdf_text(self, content: bytes) -> List[Tuple[int, str]]:
        """Optimized PDF text extraction with parallel processing"""
        pages = []
        with fitz.open(stream=content, filetype="pdf") as doc:
            if len(doc) > 10:  # Use parallel processing for larger PDFs
                with ThreadPoolExecutor(max_workers=min(CPU_COUNT, 8)) as pdf_executor:
                    futures = []
                    for page_num in range(len(doc)):
                        future = pdf_executor.submit(self._extract_single_page, doc, page_num)
                        futures.append((page_num, future))
                    
                    for page_num, future in futures:
                        text = future.result()
                        if text and text.strip():
                            pages.append((page_num + 1, text))
            else:
                # Sequential processing for smaller PDFs
                for page_num, page in enumerate(doc):
                    text = page.get_text("text")
                    if text and text.strip():
                        pages.append((page_num + 1, text))
        
        return sorted(pages, key=lambda x: x[0])

    def _extract_single_page(self, doc, page_num: int) -> str:
        """Extract text from a single page"""
        return doc[page_num].get_text("text")

    async def download_and_extract(self, url: str) -> List[Tuple[int, str]]:
        """Optimized PDF download and extraction"""
        timeout = aiohttp.ClientTimeout(total=300, connect=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                content = await response.read()
        
        return await asyncio.get_event_loop().run_in_executor(
            global_executor, self._extract_pdf_text, content
        )

    def _preprocess_texts_for_jina(self, texts: List[str]) -> List[str]:
        """Advanced text preprocessing optimized for JinaAI with vectorized operations"""
        if not texts:
            return []
        
        processed_texts = []
        
        # Batch tokenization for efficiency
        if self.tokenizer and len(texts) > 1:
            # Use batch encoding for multiple texts
            try:
                encoded_batch = self.tokenizer.batch_encode_plus(
                    texts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=True,
                    max_length=500,
                    return_tensors=None
                )
                
                for i, tokens in enumerate(encoded_batch['input_ids']):
                    if len(tokens) > 500:
                        tokens = tokens[:500]
                    cleaned_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    # Fast whitespace normalization
                    cleaned_text = ' '.join(cleaned_text.split())
                    processed_texts.append(cleaned_text)
                    
            except Exception:
                # Fallback to individual processing
                for text in texts:
                    cleaned_text = ' '.join(text.strip().split())
                    tokens = self.tokenizer.encode(cleaned_text, add_special_tokens=False)
                    if len(tokens) > 500:
                        tokens = tokens[:500]
                        cleaned_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    processed_texts.append(cleaned_text)
        else:
            # Single text or no tokenizer
            for text in texts:
                cleaned_text = ' '.join(text.strip().split())
                if self.tokenizer:
                    tokens = self.tokenizer.encode(cleaned_text, add_special_tokens=False)
                    if len(tokens) > 500:
                        tokens = tokens[:500]
                        cleaned_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                processed_texts.append(cleaned_text)
        
        return processed_texts

    async def _embed_with_advanced_caching(self, texts: List[str]) -> np.ndarray:
        """Ultra-fast embedding with advanced caching and JinaAI optimizations"""
        if not texts:
            return np.array([])
        
        # Preprocess texts for JinaAI
        processed_texts = self._preprocess_texts_for_jina(texts)
        
        embeddings = []
        texts_to_embed = []
        cache_indices = []
        
        # Check cache with batch optimization
        cache_hits = 0
        for i, text in enumerate(processed_texts):
            cached_embedding = await self.embedding_cache.get(text)
            if cached_embedding is not None:
                embeddings.append((i, cached_embedding))
                cache_hits += 1
            else:
                texts_to_embed.append(text)
                cache_indices.append(i)
        
        # Update performance stats
        self.performance_stats['cache_hits'] += cache_hits
        self.performance_stats['cache_misses'] += len(texts_to_embed)
        
        # Embed uncached texts with dynamic batching
        if texts_to_embed:
            logger.info(f"⚡ JinaAI embedding {len(texts_to_embed)} texts ({cache_hits} from cache)")
            
            # Dynamic batch size adjustment based on text lengths
            avg_text_length = sum(len(text) for text in texts_to_embed) / len(texts_to_embed)
            dynamic_batch_size = self._adjust_batch_size_for_length(avg_text_length)
            
            new_embeddings = []
            for i in range(0, len(texts_to_embed), dynamic_batch_size):
                batch = texts_to_embed[i:i + dynamic_batch_size]
                
                # Use JinaAI-optimized encoding
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=len(batch),
                    convert_to_numpy=True,
                    device=DEVICE
                )
                
                new_embeddings.extend(batch_embeddings)
                
                # Clear GPU cache periodically for large batches
                if DEVICE == "cuda" and len(new_embeddings) % 200 == 0:
                    torch.cuda.empty_cache()
            
            # Cache new embeddings (async)
            for text, embedding in zip(texts_to_embed, new_embeddings):
                await self.embedding_cache.set(text, embedding)
            
            # Add to results
            for i, embedding in enumerate(new_embeddings):
                embeddings.append((cache_indices[i], embedding))
            
            # Update performance stats
            self.performance_stats['total_embeddings_computed'] += len(new_embeddings)
            self.performance_stats['average_batch_size'] = (
                (self.performance_stats['average_batch_size'] * 0.9) + 
                (dynamic_batch_size * 0.1)
            )
        
        # Sort by original order and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])
        
        # Update GPU memory usage stats
        if DEVICE == "cuda":
            self.performance_stats['gpu_memory_usage'] = torch.cuda.memory_allocated() / 1e9
        
        return result

    def _adjust_batch_size_for_length(self, avg_length: float) -> int:
        """Dynamically adjust batch size based on average text length"""
        # JinaAI models can handle larger batches with shorter sequences
        if avg_length > 1000:  # Long texts
            return max(4, self.max_batch_size // 8)  # Smaller batches for long texts
        elif avg_length > 500:  # Medium texts
            return max(8, self.max_batch_size // 4)  # Moderate batches
        elif avg_length > 200:  # Short-medium texts
            return max(16, self.max_batch_size // 2)  # Larger batches
        else:  # Very short texts
            return self.max_batch_size  # Maximum batches for short texts

    def _rerank_with_optimization(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimized reranking with batch processing"""
        if not chunks:
            return chunks
        
        # Prepare reranker input
        reranker_input = [[query, chunk['text'][:1000]] for chunk in chunks]  # Truncate for speed
        
        # Batch reranking for efficiency
        batch_size = min(32, len(reranker_input))
        cross_scores = []
        
        for i in range(0, len(reranker_input), batch_size):
            batch = reranker_input[i:i + batch_size]
            batch_scores = self.reranker_model.predict(batch, show_progress_bar=False)
            cross_scores.extend(batch_scores)
        
        # Apply scores and sort
        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(cross_scores[i])
        
        return sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)

    async def retrieve_and_rerank(self, query: str, chunks_with_metadata: List[Dict], 
                                  chunk_embeddings: np.ndarray, request_id: str) -> List[Dict[str, Any]]:
        """Advanced retrieval with semantic search and hierarchical boosting"""
        if len(chunks_with_metadata) == 0:
            return []
        
        # Create context hash for query caching
        context_hash = hashlib.sha256(str(chunk_embeddings.tobytes()).encode()).hexdigest()[:16]
        
        # Check query cache
        cached_result = await self.query_cache.get(query, context_hash)
        if cached_result:
            logger.info(f"[{request_id}] 🎯 Using cached query result")
            return eval(cached_result)  # Safe since we control the cached content
        
        # Fast query embedding
        query_embedding = await self._embed_with_advanced_caching([query])
        
        if query_embedding.size == 0:
            return []
        
        # Ultra-fast similarity computation with optimizations
        query_embedding = query_embedding.flatten()
        
        # Optimized similarity computation with enhanced GPU utilization
        if DEVICE == "cuda" and chunk_embeddings.shape[0] > 1000:
            # Enhanced GPU computation with memory management
            try:
                # Clear GPU cache first
                torch.cuda.empty_cache()
                
                # Use smaller batches for very large datasets to prevent OOM
                batch_size = min(2000, chunk_embeddings.shape[0])
                all_similarities = []
                
                for i in range(0, chunk_embeddings.shape[0], batch_size):
                    end_idx = min(i + batch_size, chunk_embeddings.shape[0])
                    batch_embeddings = torch.from_numpy(chunk_embeddings[i:end_idx]).to(DEVICE, dtype=torch.float16)
                    query_embedding_gpu = torch.from_numpy(query_embedding).to(DEVICE, dtype=torch.float16).unsqueeze(1)
                    
                    batch_similarities = torch.mm(batch_embeddings, query_embedding_gpu).squeeze().cpu().numpy()
                    all_similarities.append(batch_similarities)
                    
                    # Clear batch from GPU memory
                    del batch_embeddings, query_embedding_gpu
                    torch.cuda.empty_cache()
                
                similarities = np.concatenate(all_similarities)
                
            except Exception as e:
                logger.warning(f"GPU computation failed ({e}), falling back to CPU")
                # Fallback to CPU with numpy optimization
                similarities = np.dot(chunk_embeddings, query_embedding)
        else:
            # Enhanced CPU optimization with threading for very large datasets
            if chunk_embeddings.shape[0] > 2000:
                # Use parallel processing for large datasets
                from concurrent.futures import ThreadPoolExecutor
                import math
                
                def compute_batch_similarity(batch_data):
                    start_idx, end_idx = batch_data
                    return np.dot(chunk_embeddings[start_idx:end_idx], query_embedding)
                
                n_threads = min(4, os.cpu_count() or 4)
                batch_size = math.ceil(chunk_embeddings.shape[0] / n_threads)
                batches = [(i, min(i + batch_size, chunk_embeddings.shape[0])) 
                          for i in range(0, chunk_embeddings.shape[0], batch_size)]
                
                with ThreadPoolExecutor(max_workers=n_threads) as executor:
                    batch_results = list(executor.map(compute_batch_similarity, batches))
                    similarities = np.concatenate(batch_results)
            elif chunk_embeddings.shape[0] > 500:
                # Use numpy's optimized BLAS for medium datasets
                similarities = np.dot(chunk_embeddings, query_embedding)
            else:
                # Use einsum for smaller datasets (often faster due to less overhead)
                similarities = np.einsum('ij,j->i', chunk_embeddings, query_embedding)
        
        # Advanced candidate selection with hierarchical boosting
        candidate_indices = self._select_diverse_candidates(
            similarities, chunks_with_metadata, self.top_k_initial
        )
        
        candidate_chunks = [chunks_with_metadata[i] for i in candidate_indices]
        
        # Apply hierarchical and semantic boosting
        for i, chunk in enumerate(candidate_chunks):
            base_score = similarities[candidate_indices[i]]
            
            # Hierarchical boosting (prefer higher-level sections)
            level_boost = max(0, 4 - chunk.get('level', 4)) * 0.03
            
            # Content quality boosting
            content_boost = self._calculate_content_quality_boost(chunk)
            
            # Page proximity boosting (prefer content from similar pages)
            page_boost = self._calculate_page_proximity_boost(chunk, candidate_chunks)
            
            chunk['semantic_score'] = base_score + level_boost + content_boost + page_boost
        
        # Re-rank with optimized cross-encoder
        reranked_chunks = await asyncio.get_event_loop().run_in_executor(
            None, self._rerank_with_optimization, query, candidate_chunks
        )
        
        final_chunks = reranked_chunks[:self.top_k_final]
        
        # Cache the result
        await self.query_cache.set(query, context_hash, str(final_chunks))
        
        logger.info(f"[{request_id}] 🎯 Retrieval complete | Top score: {final_chunks[0]['rerank_score']:.4f}")
        return final_chunks

    def _select_diverse_candidates(self, similarities: np.ndarray, chunks: List[Dict], 
                                 top_k: int) -> List[int]:
        """Enhanced diverse candidate selection with semantic clustering"""
        # Get top candidates based on similarity (more than needed for diversity filtering)
        top_indices = np.argsort(similarities)[-top_k * 3:][::-1]
        
        selected_indices = []
        similarity_threshold = 0.05  # Minimum similarity difference required
        
        for idx in top_indices:
            if len(selected_indices) >= top_k:
                break
            
            chunk = chunks[idx]
            is_diverse = True
            current_similarity = similarities[idx]
            
            # Enhanced diversity checks
            for selected_idx in selected_indices:
                selected_chunk = chunks[selected_idx]
                selected_similarity = similarities[selected_idx]
                
                # 1. Avoid identical or very similar content
                if (chunk.get('element_id') == selected_chunk.get('element_id') or
                    chunk.get('title') == selected_chunk.get('title')):
                    is_diverse = False
                    break
                
                # 2. Avoid very similar similarity scores (likely duplicate content)
                if abs(current_similarity - selected_similarity) < similarity_threshold:
                    # Check content overlap
                    chunk_words = set(chunk.get('text', '').lower().split()[:20])
                    selected_words = set(selected_chunk.get('text', '').lower().split()[:20])
                    overlap = len(chunk_words & selected_words) / max(len(chunk_words | selected_words), 1)
                    if overlap > 0.7:  # High content overlap
                        is_diverse = False
                        break
                
                # 3. Page proximity check with more nuance
                chunk_pages = set(chunk.get('pages', [chunk.get('page', 0)]))
                selected_pages = set(selected_chunk.get('pages', [selected_chunk.get('page', 0)]))
                page_overlap = len(chunk_pages & selected_pages)
                
                # Allow some page overlap if similarity scores are very different
                if (page_overlap > 0 and 
                    abs(current_similarity - selected_similarity) < similarity_threshold * 2 and
                    chunk.get('level', 4) == selected_chunk.get('level', 4)):
                    is_diverse = False
                    break
            
            if is_diverse:
                selected_indices.append(idx)
        
        # If we don't have enough diverse candidates, fill with best remaining
        if len(selected_indices) < top_k:
            remaining_candidates = [idx for idx in top_indices if idx not in selected_indices]
            needed = top_k - len(selected_indices)
            selected_indices.extend(remaining_candidates[:needed])
        
        return selected_indices

    def _calculate_content_quality_boost(self, chunk: Dict) -> float:
        """Calculate content quality boost based on chunk characteristics"""
        content = chunk.get('text', '')
        
        # Boost for structured content
        structure_indicators = len(re.findall(r'^\d+\.|\n\d+\.|\n[A-Z]\.', content, re.MULTILINE))
        structure_boost = min(0.02, structure_indicators * 0.005)
        
        # Boost for appropriate length
        word_count = chunk.get('word_count', len(content.split()))
        if 50 <= word_count <= 300:
            length_boost = 0.01
        elif 300 < word_count <= 600:
            length_boost = 0.02
        else:
            length_boost = 0.0
        
        return structure_boost + length_boost

    def _calculate_page_proximity_boost(self, chunk: Dict, all_chunks: List[Dict]) -> float:
        """Calculate boost based on page proximity to other selected chunks"""
        chunk_page = chunk.get('page', 1)
        
        # Count nearby chunks
        nearby_count = 0
        for other_chunk in all_chunks:
            other_page = other_chunk.get('page', 1)
            if abs(chunk_page - other_page) <= 2 and chunk != other_chunk:
                nearby_count += 1
        
        return min(0.01, nearby_count * 0.003)

    def _create_enriched_context(self, contexts: List[Dict]) -> str:
        """Create enriched context with advanced formatting"""
        context_parts = []
        seen_pages = set()
        
        for i, ctx in enumerate(contexts):
            pages = ctx.get('pages', [ctx.get('page', 1)])
            page_range = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"
            
            # Create unique identifiers for context parts
            section_info = f"Section: {ctx.get('title', 'Unknown')}" if ctx.get('title') else ""
            level_info = f"Level {ctx.get('level', 'Unknown')}" if ctx.get('level') is not None else ""
            relevance_score = f"Relevance: {ctx.get('rerank_score', 0):.3f}"
            
            # Build context header
            header_parts = [page_range]
            if section_info:
                header_parts.append(section_info)
            if level_info:
                header_parts.append(level_info)
            header_parts.append(relevance_score)
            
            header = f"--- Context {i+1} | {' | '.join(header_parts)} ---"
            
            # Add content with smart truncation
            content = ctx['text']
            if len(content) > 1500:  # Truncate very long content
                content = content[:1400] + "... [truncated]"
            
            context_parts.append(f"{header}\n{content}")
            seen_pages.update(pages)
        
        return "\n\n".join(context_parts)

    def _generate_answer_sync(self, prompt: str) -> str:
        """Generate answer with optimized prompting"""
        try:
            response = self.llm_model.generate_content(
                prompt, 
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024
                }
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer. Please try again."

    async def answer_question(self, question: str, contexts: List[Dict], q_request_id: str) -> str:
        """Generate comprehensive answer with advanced context utilization"""
        if not contexts:
            return "The answer to this question could not be found in the provided document."
        
        # Create enriched context
        context_for_prompt = self._create_enriched_context(contexts)
        
        # Advanced prompt engineering for better results
        prompt = f"""You are an advanced AI assistant with expertise in analyzing hierarchically structured documents. Your task is to provide a precise, comprehensive answer based SOLELY on the provided context.

CRITICAL INSTRUCTIONS:
1. Analyze ALL provided contexts, paying attention to relevance scores and hierarchical levels
2. Synthesize information from multiple contexts when they complement each other
3. Provide a direct, well-structured answer that addresses the question completely
4. MANDATORY CITATIONS: Always cite your sources using this format:
   - Single page: "(Page 15)"
   - Multiple pages: "(Pages 15-17)"
   - Specific sections: "(Section 2.1, Page 15)"
   - Multiple sources: "(Pages 12, 15-16; Section 3.2, Page 20)"
5. If information comes from different hierarchical levels, indicate the most relevant level
6. If the question cannot be answered from the provided context, state: "The answer to this question could not be found in the provided document."
7. Do NOT use external knowledge - rely ONLY on the provided context

HIERARCHICAL DOCUMENT CONTEXT:
{context_for_prompt}

QUESTION TO ANSWER:
{question}

COMPREHENSIVE ANSWER WITH PRECISE CITATIONS:"""
        
        return await asyncio.get_event_loop().run_in_executor(
            None, self._generate_answer_sync, prompt
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        cache_stats = self.embedding_cache.get_stats()
        query_cache_stats = self.query_cache.get_stats()
        
        return {
            **self.performance_stats,
            **cache_stats,
            "query_cache": query_cache_stats,
            "model_info": {
                "embedding_model": self.embedding_model_name,
                "device": DEVICE,
                "gpu_memory_gb": GPU_MEMORY_GB,
                "half_precision": self.enable_half_precision,
                "max_batch_size": self.max_batch_size
            },
            "cache_efficiency": {
                "embedding_hit_rate": cache_stats.get("overall_hit_rate", 0),
                "query_hit_rate": query_cache_stats.get("hit_rate", 0),
                "embedding_cache_size": cache_stats.get("memory_cache_size", 0),
                "query_cache_size": query_cache_stats.get("memory_cache_size", 0)
            }
        }

    async def clear_all_caches(self):
        """Clear all caches and reset performance stats"""
        await self.embedding_cache.clear()
        await self.query_cache.clear()
        
        # Reset performance stats
        self.performance_stats = {
            'total_embeddings_computed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_batch_size': 0,
            'gpu_memory_usage': 0
        }
        
        # Clear GPU cache if available
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("🗑️ All caches cleared and memory optimized")


# Global RAG system instance
rag_system = JinaAIOptimizedRAGSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan management"""
    # Startup
    logger.info("🚀 Starting JinaAI-Optimized RAG Server...")
    await asyncio.get_event_loop().run_in_executor(global_executor, rag_system.load_models)
    logger.info("✅ Server ready for requests")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down server...")
    await rag_system.clear_all_caches()
    global_executor.shutdown(wait=True)

# 6. FASTAPI APPLICATION SERVER & STARTUP EVENT
# ==============================================================================
app = FastAPI(
    title="JinaAI-Optimized Ultra-Fast RAG System", 
    version="9.0.0",
    description="Production-grade RAG system with JinaAI embeddings and advanced optimizations",
    lifespan=lifespan
)


# --- API Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class PerformanceStatsResponse(BaseModel):
    performance_stats: Dict[str, Any]

class SystemInfoResponse(BaseModel):
    system_info: Dict[str, Any]


def verify_token(authorization: str = Header(None)):
    """Verify API token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.split("Bearer ")[1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


# --- Main API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_document_and_answer(
    request: QueryRequest, 
    background_tasks: BackgroundTasks,
    authorization: str = Header(None)
):
    """Main endpoint for document processing and question answering"""
    verify_token(authorization)
    request_id = f"req_{int(time.time() * 1000)}"
    start_time = time.time()
    
    logger.info(f"[{request_id}] 🚀 New request | Questions: {len(request.questions)} | Model: JinaAI")

    try:
        # PDF extraction with timing
        pdf_start = time.time()
        pages = await rag_system.download_and_extract(request.documents)
        pdf_time = time.time() - pdf_start
        logger.info(f"[{request_id}] 📄 PDF extracted: {len(pages)} pages in {pdf_time:.2f}s")
        
        # Document structure analysis
        structure_start = time.time()
        hierarchical_elements = rag_system.processor.extract_hierarchical_structure(pages, request_id)
        structure_time = time.time() - structure_start
        
        # Convert to retrieval chunks
        chunk_start = time.time()
        chunks_with_metadata = rag_system.processor.get_chunks_for_retrieval(hierarchical_elements)
        chunk_time = time.time() - chunk_start
        
        logger.info(f"[{request_id}] 📚 Created {len(chunks_with_metadata)} optimized chunks")
        
        # JinaAI embedding generation
        embed_start = time.time()
        chunk_texts = [c['text'] for c in chunks_with_metadata]
        chunk_embeddings = await rag_system._embed_with_advanced_caching(chunk_texts)
        embed_time = time.time() - embed_start
        
        logger.info(f"[{request_id}] ⚡ JinaAI embeddings completed in {embed_time:.2f}s")

        # Process questions in parallel
        async def process_single_question(question: str, q_idx: int) -> str:
            q_request_id = f"{request_id}_q{q_idx+1}"
            contexts = await rag_system.retrieve_and_rerank(
                question, chunks_with_metadata, chunk_embeddings, q_request_id
            )
            return await rag_system.answer_question(question, contexts, q_request_id)

        qa_start = time.time()
        tasks = [process_single_question(q, i) for i, q in enumerate(request.questions)]
        answers = await asyncio.gather(*tasks)
        qa_time = time.time() - qa_start

        total_time = time.time() - start_time
        
        logger.info(f"[{request_id}] ✅ Request completed in {total_time:.2f}s | JinaAI embedding: {embed_time:.2f}s")
        
        # Schedule background cleanup if needed
        if background_tasks and total_time > 30:
            background_tasks.add_task(cleanup_memory)
        
        return QueryResponse(answers=answers)
    
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- Performance and Monitoring Endpoints ---
@app.get("/performance/stats", response_model=PerformanceStatsResponse)
async def get_performance_stats(authorization: str = Header(None)):
    """Get comprehensive performance statistics"""
    verify_token(authorization)
    return PerformanceStatsResponse(performance_stats=rag_system.get_performance_stats())

@app.get("/performance/system-info", response_model=SystemInfoResponse)
async def get_system_info(authorization: str = Header(None)):
    """Get detailed system information"""
    verify_token(authorization)
    
    system_info = {
        "hardware": {
            "device": DEVICE,
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory_gb": GPU_MEMORY_GB,
            "cpu_count": CPU_COUNT
        },
        "model_config": {
            "embedding_model": rag_system.embedding_model_name,
            "reranker_model": rag_system.reranker_model_name,
            "max_batch_size": rag_system.max_batch_size,
            "half_precision_enabled": rag_system.enable_half_precision
        },
        "cache_config": {
            "embedding_cache_max_size": rag_system.embedding_cache.max_memory_size,
            "query_cache_max_size": rag_system.query_cache.max_memory_size
        }
    }
    
    return SystemInfoResponse(system_info=system_info)

@app.post("/performance/clear-cache")
async def clear_all_caches(authorization: str = Header(None)):
    """Clear all caches and optimize memory"""
    verify_token(authorization)
    await rag_system.clear_all_caches()
    return {"message": "All caches cleared and memory optimized"}

@app.post("/performance/optimize-memory")
async def optimize_memory(authorization: str = Header(None)):
    """Perform comprehensive memory optimization"""
    verify_token(authorization)
    
    initial_memory = 0
    if DEVICE == "cuda":
        initial_memory = torch.cuda.memory_allocated() / 1e9
    
    # Clear caches
    await rag_system.clear_all_caches()
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU memory if available
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    final_memory = 0
    if DEVICE == "cuda":
        final_memory = torch.cuda.memory_allocated() / 1e9
        memory_freed = initial_memory - final_memory
    else:
        memory_freed = 0
    
    return {
        "message": "Memory optimization completed",
        "initial_memory_gb": round(initial_memory, 3),
        "final_memory_gb": round(final_memory, 3),
        "memory_freed_gb": round(memory_freed, 3)
    }


# --- Debug and Development Endpoints ---
@app.post("/debug/document-structure")
async def debug_document_structure(request: QueryRequest, authorization: str = Header(None)):
    """Debug endpoint to inspect document hierarchical structure"""
    verify_token(authorization)
    request_id = f"debug_{int(time.time() * 1000)}"
    
    try:
        # Extract and analyze document
        pages = await rag_system.download_and_extract(request.documents)
        hierarchical_elements = rag_system.processor.extract_hierarchical_structure(pages, request_id)
        
        # Create detailed structure analysis
        structure_analysis = {
            "document_stats": {
                "total_pages": len(pages),
                "total_elements": len(hierarchical_elements),
                "element_types": {}
            },
            "hierarchy": [],
            "content_distribution": {}
        }
        
        # Analyze element types
        for element in hierarchical_elements:
            element_type = element.element_type.value
            if element_type not in structure_analysis["document_stats"]["element_types"]:
                structure_analysis["document_stats"]["element_types"][element_type] = 0
            structure_analysis["document_stats"]["element_types"][element_type] += 1
        
        # Build hierarchy view
        for element in hierarchical_elements[:50]:  # Limit for readability
            element_info = {
                "id": element.element_id,
                "type": element.element_type.value,
                "title": element.title[:100] + "..." if len(element.title) > 100 else element.title,
                "level": element.level,
                "parent_id": element.parent_id,
                "children_count": len(element.children_ids),
                "pages": element.page_numbers,
                "content_stats": {
                    "char_count": len(element.content),
                    "word_count": len(element.content.split()),
                    "line_count": len(element.content.split('\n'))
                },
                "content_preview": element.content[:200] + "..." if len(element.content) > 200 else element.content
            }
            structure_analysis["hierarchy"].append(element_info)
        
        # Analyze content distribution
        for level in range(5):
            level_elements = [e for e in hierarchical_elements if e.level == level]
            if level_elements:
                total_content = sum(len(e.content) for e in level_elements)
                avg_content = total_content / len(level_elements)
                structure_analysis["content_distribution"][f"level_{level}"] = {
                    "count": len(level_elements),
                    "total_chars": total_content,
                    "avg_chars": round(avg_content, 2)
                }
        
        return structure_analysis
    
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Debug error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")


@app.post("/debug/embedding-analysis")
async def debug_embedding_analysis(
    request: QueryRequest, 
    authorization: str = Header(None),
    sample_size: int = 10
):
    """Analyze embedding quality and similarity patterns"""
    verify_token(authorization)
    request_id = f"embed_debug_{int(time.time() * 1000)}"
    
    try:
        # Process document
        pages = await rag_system.download_and_extract(request.documents)
        hierarchical_elements = rag_system.processor.extract_hierarchical_structure(pages, request_id)
        chunks_with_metadata = rag_system.processor.get_chunks_for_retrieval(hierarchical_elements)
        
        # Sample chunks for analysis
        sample_chunks = chunks_with_metadata[:sample_size]
        chunk_texts = [c['text'][:500] for c in sample_chunks]  # Truncate for speed
        
        # Generate embeddings
        embeddings = await rag_system._embed_with_advanced_caching(chunk_texts)
        
        # Analyze embedding patterns
        analysis = {
            "embedding_stats": {
                "dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
                "sample_size": len(embeddings),
                "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))) if len(embeddings) > 0 else 0,
                "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))) if len(embeddings) > 0 else 0
            },
            "similarity_matrix": [],
            "chunk_analysis": []
        }
        
        # Compute similarity matrix for sample
        if len(embeddings) > 1:
            similarity_matrix = np.dot(embeddings, embeddings.T)
            # Convert to list for JSON serialization
            analysis["similarity_matrix"] = similarity_matrix.tolist()
        
        # Analyze individual chunks
        for i, (chunk, embedding) in enumerate(zip(sample_chunks, embeddings)):
            chunk_analysis = {
                "index": i,
                "title": chunk.get('title', 'Unknown'),
                "type": chunk.get('element_type', 'unknown'),
                "word_count": chunk.get('word_count', 0),
                "embedding_norm": float(np.linalg.norm(embedding)),
                "text_preview": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            }
            analysis["chunk_analysis"].append(chunk_analysis)
        
        return analysis
    
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Embedding analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding analysis error: {str(e)}")


@app.post("/benchmark/qwen3-performance")
async def benchmark_qwen3_performance(authorization: str = Header(None)):
    """Comprehensive Qwen3 embedding performance benchmark"""
    verify_token(authorization)
    
    # Test different scenarios
    test_scenarios = [
        {
            "name": "short_texts",
            "texts": [f"Short text sample {i}." for i in range(1, 21)],
            "description": "20 short texts (5-10 words each)"
        },
        {
            "name": "medium_texts", 
            "texts": [f"Medium length text sample {i} with multiple sentences and varied content to test embedding performance with moderate sized inputs." for i in range(1, 21)],
            "description": "20 medium texts (20-30 words each)"
        },
        {
            "name": "long_texts",
            "texts": [f"Long text sample {i} containing extensive content with multiple paragraphs, detailed explanations, and comprehensive information to thoroughly test the Qwen3 embedding model performance with substantial input lengths. " * 5 for i in range(1, 11)],
            "description": "10 long texts (100+ words each)"
        },
        {
            "name": "mixed_batch",
            "texts": ["Short.", "Medium length text with several words and phrases for testing purposes.", "Very long comprehensive text sample " * 20],
            "description": "Mixed length texts in single batch"
        }
    ]
    
    benchmark_results = {}
    
    for scenario in test_scenarios:
        scenario_name = scenario["name"]
        texts = scenario["texts"]
        
        # Clear cache for accurate timing
        await rag_system.embedding_cache.clear()
        
        # Benchmark without cache
        start_time = time.time()
        embeddings_no_cache = await rag_system._embed_with_advanced_caching(texts)
        no_cache_time = time.time() - start_time
        
        # Benchmark with cache (second run)
        start_time = time.time()
        embeddings_cached = await rag_system._embed_with_advanced_caching(texts)
        cached_time = time.time() - start_time
        
        # Calculate statistics
        total_chars = sum(len(text) for text in texts)
        total_words = sum(len(text.split()) for text in texts)
        
        benchmark_results[scenario_name] = {
            "description": scenario["description"],
            "metrics": {
                "text_count": len(texts),
                "total_characters": total_chars,
                "total_words": total_words,
                "avg_chars_per_text": round(total_chars / len(texts), 2),
                "avg_words_per_text": round(total_words / len(texts), 2)
            },
            "performance": {
                "no_cache_time": round(no_cache_time, 4),
                "cached_time": round(cached_time, 4),
                "speedup_factor": round(no_cache_time / max(cached_time, 0.0001), 2),
                "chars_per_second": round(total_chars / no_cache_time, 2),
                "texts_per_second": round(len(texts) / no_cache_time, 2)
            },
            "embedding_stats": {
                "dimension": embeddings_no_cache.shape[1] if len(embeddings_no_cache) > 0 else 0,
                "mean_norm": float(np.mean(np.linalg.norm(embeddings_no_cache, axis=1))) if len(embeddings_no_cache) > 0 else 0
            }
        }
    
    # Overall system stats
    system_stats = {
        "model_info": {
            "embedding_model": rag_system.embedding_model_name,
            "device": DEVICE,
            "gpu_memory_gb": GPU_MEMORY_GB,
            "max_batch_size": rag_system.max_batch_size,
            "half_precision": rag_system.enable_half_precision
        },
        "cache_stats": rag_system.embedding_cache.get_stats(),
        "performance_stats": rag_system.get_performance_stats()
    }
    
    return {
        "benchmark_results": benchmark_results,
        "system_stats": system_stats,
        "timestamp": time.time()
    }


# --- Background Task Functions ---
async def cleanup_memory():
    """Background task for memory cleanup"""
    try:
        # Intelligent cache cleanup - get cache stats
        cache_stats = rag_system.embedding_cache.get_stats()
        memory_cache_size = cache_stats.get("memory_cache_size", 0)
        max_memory_size = rag_system.embedding_cache.max_memory_size
        
        if memory_cache_size > max_memory_size * 0.8:
            # Redis cache handles its own eviction, so just clear memory cache if needed
            await rag_system.embedding_cache._evict_memory_items(int(max_memory_size * 0.2))
        
        # Clear GPU memory periodically
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("🧹 Background memory cleanup completed")
    except Exception as e:
        logger.error(f"Error in background cleanup: {e}")


# --- Health Check and Status Endpoints ---
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "RAG System",
            "version": "9.0.0"
        }
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

@app.get("/api/health")
async def api_health_checkpoint():
    """Comprehensive API health checkpoint with system validation"""
    try:
        checkpoint_start = time.time()
        
        # Test model availability
        model_status = {
            "embedding_model_loaded": rag_system.embedding_model is not None,
            "reranker_model_loaded": rag_system.reranker_model is not None,
            "tokenizer_loaded": rag_system.tokenizer is not None
        }
        
        # Test embedding functionality
        embedding_test_passed = False
        embedding_time = 0
        try:
            embed_start = time.time()
            test_embedding = await rag_system._embed_with_advanced_caching(["API health check test"])
            embedding_time = time.time() - embed_start
            embedding_test_passed = len(test_embedding) > 0 if hasattr(test_embedding, '__len__') else True
        except Exception as e:
            logger.warning(f"Embedding test failed: {e}")
        
        # Test Redis connection
        redis_status = {
            "connected": False,
            "error": None
        }
        try:
            redis_conn = await get_redis_client()
            if redis_conn:
                await redis_conn.ping()
                redis_status["connected"] = True
        except Exception as e:
            redis_status["error"] = str(e)
        
        # Get cache statistics
        cache_stats = rag_system.embedding_cache.get_stats()
        
        # System resources
        system_resources = {
            "device": DEVICE,
            "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False,
            "cpu_count": CPU_COUNT
        }
        
        if DEVICE == "cuda":
            try:
                system_resources["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1e9
                system_resources["gpu_memory_total_gb"] = GPU_MEMORY_GB
            except Exception:
                pass
        
        checkpoint_time = time.time() - checkpoint_start
        
        # Determine overall health
        overall_status = "healthy"
        if not model_status["embedding_model_loaded"] or not embedding_test_passed:
            overall_status = "degraded"
        
        health_data = {
            "status": overall_status,
            "timestamp": time.time(),
            "service": "Redis-Enhanced RAG System",
            "version": "9.0.0",
            "api_version": "v1",
            "checkpoint_duration_ms": round(checkpoint_time * 1000, 2),
            "models": model_status,
            "embedding_test": {
                "passed": embedding_test_passed,
                "duration_ms": round(embedding_time * 1000, 2)
            },
            "redis": redis_status,
            "cache": {
                "memory_cache_size": cache_stats.get("memory_cache_size", 0),
                "overall_hit_rate": cache_stats.get("overall_hit_rate", 0),
                "total_requests": cache_stats.get("total_requests", 0)
            },
            "system": system_resources,
            "endpoints": {
                "main": "/api/v1/hackrx/run",
                "health": "/health",
                "api_health": "/api/health",
                "performance": "/performance/stats",
                "redis_status": "/redis-status"
            }
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"API health checkpoint failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "service": "Redis-Enhanced RAG System",
            "version": "9.0.0",
            "error": str(e)
        }

@app.get("/redis-status")
async def redis_status(authorization: str = Header(None)):
    """Get Redis connection status and cache statistics"""
    verify_token(authorization)
    
    try:
        redis_conn = await get_redis_client()
        redis_connected = False
        redis_info = {}
        
        if redis_conn:
            try:
                await redis_conn.ping()
                redis_connected = True
                # Get Redis info
                info = await redis_conn.info()
                redis_info = {
                    "redis_version": info.get("redis_version", "unknown"),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                }
                
                # Get key counts
                embedding_keys = await redis_conn.keys("embedding:*")
                query_keys = await redis_conn.keys("query:*")
                redis_info.update({
                    "embedding_keys_count": len(embedding_keys),
                    "query_keys_count": len(query_keys)
                })
                
            except Exception as e:
                redis_connected = False
                redis_info = {"error": str(e)}
        
        # Get cache statistics
        embedding_cache_stats = rag_system.embedding_cache.get_stats()
        query_cache_stats = rag_system.query_cache.get_stats()
        
        return {
            "redis_connected": redis_connected,
            "redis_config": {
                "host": REDIS_HOST,
                "port": REDIS_PORT,
                "db": REDIS_DB,
                "expire_time": REDIS_EXPIRE_TIME
            },
            "redis_info": redis_info,
            "cache_stats": {
                "embedding_cache": embedding_cache_stats,
                "query_cache": query_cache_stats
            }
        }
        
    except Exception as e:
        logger.error(f"Redis status check failed: {e}")
        return {
            "redis_connected": False,
            "error": str(e),
            "cache_stats": {
                "embedding_cache": rag_system.embedding_cache.get_stats(),
                "query_cache": rag_system.query_cache.get_stats()
            }
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Qwen3-Optimized Ultra-Fast RAG System",
        "version": "9.0.0",
        "description": "Production-grade RAG system with JinaAI embeddings and advanced optimizations",
        "model": rag_system.embedding_model_name,
        "device": DEVICE,
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health",
            "performance": "/performance/stats",
            "system_info": "/performance/system-info",
            "debug": "/debug/document-structure",
            "benchmark": "/benchmark/qwen3-performance"
        }
    }


# 7. SERVER EXECUTION
# ==============================================================================
if __name__ == "__main__":
    logger.info("🚀 Starting Qwen3-Optimized Ultra-Fast RAG Server...")
    logger.info(f"🔥 Hardware: {DEVICE} | CPU cores: {CPU_COUNT} | GPU memory: {GPU_MEMORY_GB:.1f}GB")
    logger.info(f"⚡ Model: {rag_system.embedding_model_name}")
    
    uvicorn.run(
        "final_2:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # Disable reload for production
        workers=1,     # Single worker for GPU efficiency
        loop="asyncio",
        log_level="info"
    )
