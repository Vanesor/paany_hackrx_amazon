#!/usr/bin/env python3
"""
SPEED-OPTIMIZED RAG SYSTEM (v13) - FAST & ACCURATE FOR 2GB RAM AMAZON LIGHTSAIL
- BGE-base model for superior accuracy (~220MB)
- Optimized for ACCURACY with 1.3GB available RAM buffer
- Enhanced retrieval depth and context processing
- 30-second timeout with comprehensive results
- Maximum accuracy within memory constraints
"""

# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import os
import ast
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
import sys
from typing import List, Dict, Any, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import threading
import signal

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
import psutil
import signal
import dotenv
dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Secrets and Critical Configurations ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
EXPECTED_TOKEN = "6e8b43cca9d29b261843a3b1c53382bdaa5b2c9e96db92da679278c6dc0042ca"

if not GOOGLE_API_KEY:
    logger.critical("FATAL: GOOGLE_API_KEY environment variable not set.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# SINGLE PROCESS configuration for 1-1.2GB RAM
CPU_COUNT = os.cpu_count() or 4
# Only use ThreadPoolExecutor for embedding/chunking speedup - limited workers for memory
global_executor = ThreadPoolExecutor(max_workers=min(6, CPU_COUNT))  # Increased for 2GB RAM

# OPTIMIZED processing for 2GB RAM (1GB available)
REQUEST_SEMAPHORE = asyncio.Semaphore(1)  # Single request for stability
PROCESSING_SEMAPHORE = asyncio.Semaphore(1)  # Single document processing
CHUNK_PROCESSING_SEMAPHORE = asyncio.Semaphore(3)  # Increased for 2GB RAM
EMBEDDING_SEMAPHORE = asyncio.Semaphore(1)  # Single embedding batch

# 30-second timeout configuration
RESPONSE_TIMEOUT = 30  # Maximum time before returning partial results
TIMEOUT_WARNING = 27   # Send partial results after this time

# 30-second timeout configuration
RESPONSE_TIMEOUT = 30  # Maximum time before returning partial results
TIMEOUT_WARNING = 27   # Send partial results after this time

# Lightweight memory management for 1GB RAM
def get_memory_usage() -> float:
    """Lightweight memory usage tracking"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024 / 1024
    except:
        return 0.0

def force_memory_cleanup():
    """Memory cleanup optimized for 2GB RAM systems"""
    try:
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available (though we force CPU mode)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info(f"🗑️ Memory cleanup complete: {get_memory_usage():.2f}GB")
    except Exception as e:
        logger.warning(f"Memory cleanup warning: {e}")

def check_memory_threshold() -> bool:
    """Check if memory usage is approaching limits for 2GB RAM system"""
    current = get_memory_usage()
    return current > 1.5  # Warning at 1.5GB for 2GB systems (leave 0.5GB buffer)

# Force CPU-only processing for memory efficiency
DEVICE = "cpu"  # Force CPU for memory efficiency
GPU_MEMORY_GB = 0

logger.info(f"🔥 Using device: {DEVICE} | CPU cores: {CPU_COUNT} (Memory-optimized for 1GB)")


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
    children_ids: List[str] = field(default_factory=list)
    level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}


# 3. TIMEOUT-AWARE PROCESSING
# ==============================================================================
class TimeoutManager:
    """Manages request timeout and partial results"""
    
    def __init__(self, timeout_seconds: int = RESPONSE_TIMEOUT):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()
        self.partial_results = []
        
    def is_timeout_approaching(self, warning_threshold: int = TIMEOUT_WARNING) -> bool:
        """Check if we're approaching timeout"""
        elapsed = time.time() - self.start_time
        return elapsed >= warning_threshold
        
    def is_timeout_exceeded(self) -> bool:
        """Check if timeout is exceeded"""
        elapsed = time.time() - self.start_time
        return elapsed >= self.timeout_seconds
        
    def get_elapsed_time(self) -> float:
        """Get elapsed time"""
        return time.time() - self.start_time
        
    def add_partial_result(self, result: Any):
        """Add a partial result"""
        self.partial_results.append(result)
        
    def get_partial_results(self) -> List[Any]:
        """Get all partial results"""
        return self.partial_results


# 4. ENHANCED SEMANTIC PROCESSOR CLASS
# ==============================================================================
class HierarchicalSemanticProcessor:
    """Lightweight semantic processor optimized for 1GB RAM"""
    
    def __init__(self):
        # Simplified pattern matching for minimal memory usage
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

    def analyze_document_structure(self, text_hash: str, word_count: int, text_sample: str) -> Dict[str, Any]:
        """Analyze document structure without caching for memory efficiency"""
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
        """Extract hierarchical document structure with optimized PARALLEL processing"""
        # Combine all pages with character mapping using PARALLEL processing
        full_text, char_to_page_map = self._combine_pages_parallel(pages)
        
        logger.info(f"[{request_id}] 🔍 Analyzing document structure with PARALLEL processing...")
        
        # Create hash for caching analysis
        text_hash = hashlib.sha256(full_text[:10000].encode()).hexdigest()[:16]
        word_count = len(full_text.split())
        text_sample = full_text[:10000]
        
        analysis = self.analyze_document_structure(text_hash, word_count, text_sample)
        logger.info(f"[{request_id}] 📊 Document analysis: {analysis}")
        
        # PARALLEL heading detection for better performance (lowered threshold)
        if len(full_text) > 50000:  # Lowered from 100000 for more parallel processing
            headings = self._parallel_heading_detection(full_text, char_to_page_map)
        else:
            headings = self._sequential_heading_detection(full_text, char_to_page_map)
        
        # Create hierarchical structure with PARALLEL chunk processing
        elements = self._build_hierarchy_parallel(headings, full_text, char_to_page_map, analysis, pages, request_id)
        
        logger.info(f"[{request_id}] 🏗️ Created {len(elements)} hierarchical elements with PARALLEL processing")
        return elements

    def _combine_pages_parallel(self, pages: List[Tuple[int, str]]) -> Tuple[str, Dict[int, int]]:
        """Combine pages in parallel for better performance"""
        if len(pages) <= 4:
            # Sequential for small documents
            full_text, char_to_page_map = "", {}
            for page_num, page_text in pages:
                start_char = len(full_text)
                clean_page_text = page_text + "\n\n"
                full_text += clean_page_text
                for i in range(start_char, len(full_text), max(1, len(clean_page_text) // 100)):
                    char_to_page_map[i] = page_num
            return full_text, char_to_page_map
        
        # Parallel processing for larger documents
        def process_page_chunk(page_group):
            text_parts = []
            char_maps = []
            current_offset = 0
            
            for page_num, page_text in page_group:
                start_char = current_offset
                clean_page_text = page_text + "\n\n"
                text_parts.append(clean_page_text)
                end_char = current_offset + len(clean_page_text)
                
                page_map = {}
                for i in range(start_char, end_char, max(1, len(clean_page_text) // 100)):
                    page_map[i] = page_num
                char_maps.append(page_map)
                current_offset = end_char
            
            return "".join(text_parts), char_maps
        
        # Split pages into chunks for parallel processing
        chunk_size = max(2, len(pages) // min(4, len(pages)))
        page_chunks = [pages[i:i + chunk_size] for i in range(0, len(pages), chunk_size)]
        
        # Process chunks in parallel with memory-aware limits
        max_workers = min(6, len(page_chunks))  # Increased for ultra-fast speed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(process_page_chunk, page_chunks))
        
        # Combine results efficiently
        full_text = ""
        char_to_page_map = {}
        
        for chunk_text, chunk_maps in chunk_results:
            text_offset = len(full_text)
            full_text += chunk_text
            
            # Adjust character mappings efficiently
            for char_map in chunk_maps:
                for char_pos, page_num in char_map.items():
                    char_to_page_map[char_pos + text_offset] = page_num
        
        return full_text, char_to_page_map

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
                    'text': title,  # Use 'text' for consistency
                    'title': title,  # Keep 'title' for backward compatibility
                    'type': chunk_type,
                    'level': level,
                    'start': start_pos,
                    'end': end_pos,
                    'page': page_num,
                    'pattern': pattern.pattern  # Add pattern info for debugging
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

    def _build_hierarchy_parallel(self, headings: List[Dict], full_text: str, char_to_page_map: Dict[int, int],
                                analysis: Dict[str, Any], pages: List[Tuple[int, str]], request_id: str) -> List[DocumentElement]:
        """Build hierarchical structure with PARALLEL chunk processing for better performance"""
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
        
        # Process headings to create hierarchy (sequential for structure)
        current_parents = {0: "doc_0"}  # level -> parent_id mapping
        
        for i, heading in enumerate(headings):
            try:
                # Debug heading structure
                logger.debug(f"[{request_id}] Processing heading {i}: {heading}")
                
                # Ensure heading has required fields
                if 'text' not in heading:
                    heading['text'] = heading.get('title', 'Unknown')
                if 'start' not in heading or 'end' not in heading:
                    logger.warning(f"[{request_id}] Heading missing start/end: {heading}")
                    continue
                    
                # Determine content boundaries
                content_start = heading['end']
                content_end = headings[i + 1]['start'] if i + 1 < len(headings) else len(full_text)
                
                # Extract content for this heading
                heading_content = full_text[content_start:content_end].strip()
                pages_for_heading = self._get_pages_for_range(heading['start'], content_end, char_to_page_map)
                
                # Create element
                element_id = f"elem_{element_counter}"
                parent_id = self._find_parent_id(heading['level'], current_parents)
                
                element = DocumentElement(
                    element_id=element_id,
                    element_type=heading['type'],
                    title=heading['text'],
                    content=heading_content,
                    page_numbers=pages_for_heading,
                    start_char=heading['start'],
                    end_char=content_end,
                    parent_id=parent_id,
                    children_ids=[],
                    level=heading['level'],
                    metadata={"heading_pattern": heading.get('pattern', 'unknown')}
                )
                elements.append(element)
                
                # Update current parents mapping
                current_parents[heading['level']] = element_id
                current_parents = {k: v for k, v in current_parents.items() if k <= heading['level']}
                
                element_counter += 1
                
            except Exception as e:
                logger.error(f"[{request_id}] Error processing heading {i}: {e} | Heading: {heading}")
                continue
        
        # Create paragraph-level chunks with PARALLEL content-aware processing
        logger.info(f"[{request_id}] 🔄 Starting PARALLEL chunk processing...")
        self._create_content_aware_chunks_parallel(elements, full_text, char_to_page_map, element_counter, analysis, request_id)
        
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
        """Determine optimal chunk size for SPEED with good accuracy"""
        doc_type = analysis.get('document_type', 'article')
        complexity = analysis.get('structure_complexity', 'simple')
        size_category = analysis.get('size_category', 'standard')
        
        # SPEED-OPTIMIZED chunk sizes - larger chunks for faster processing
        base_sizes = {
            'legal': 1600,     # Reduced for speed while maintaining quality
            'academic': 1400,  # Reduced for faster processing
            'book': 1200,      # Reduced for speed
            'article': 1000    # Reduced for quick processing
        }
        
        complexity_multipliers = {
            'simple': 1.0,     # Standard sizing
            'moderate': 1.1,   # Slightly larger (reduced from 1.2)
            'complex': 1.2     # Reduced from 1.4 for speed
        }
        
        size_multipliers = {
            'standard': 1.0,   # Standard sizing
            'medium': 1.1,     # Reduced from 1.2 for speed
            'large': 1.2       # Reduced from 1.3 for speed
        }
        
        base_size = base_sizes.get(doc_type, 1000)  # Reduced default for speed
        complexity_mult = complexity_multipliers.get(complexity, 1.0)
        size_mult = size_multipliers.get(size_category, 1.0)
        
        final_size = int(base_size * complexity_mult * size_mult)
        # Enhanced cap for accuracy - larger chunks = better context
        return min(final_size, 2500)  # Increased from 2000 for better accuracy

    def get_chunks_for_retrieval(self, elements: List[DocumentElement]) -> List[Dict[str, Any]]:
        """Convert hierarchical elements to chunks suitable for retrieval - SPEED OPTIMIZED"""
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
        
        # ACCURACY-FOCUSED chunk retention for 2GB RAM
        max_chunks = 80  # Increased for better coverage
        if len(chunks) > max_chunks:
            # Keep highest quality chunks for better accuracy
            chunks = sorted(chunks, key=lambda x: x.get('word_count', 0), reverse=True)[:max_chunks]
            logger.info(f"� Accuracy optimization: Limited to {len(chunks)} highest quality chunks")
        
        return chunks

    def _create_content_aware_chunks_parallel(self, elements: List[DocumentElement], full_text: str,
                                            char_to_page_map: Dict[int, int], counter_start: int,
                                            analysis: Dict[str, Any], request_id: str) -> None:
        """Create content-aware chunks with PARALLEL processing for better performance"""
        element_counter = counter_start
        new_elements = []
        
        # Get optimal chunk size based on document characteristics
        base_chunk_size = self._get_optimal_chunk_size(analysis)
        
        # Filter elements that need chunking
        elements_to_chunk = [
            elem for elem in elements 
            if elem.element_type in [ChunkType.SECTION, ChunkType.SUBSECTION, ChunkType.CHAPTER] 
            and len(elem.content) > base_chunk_size * 1.3
        ]
        
        if not elements_to_chunk:
            logger.info(f"[{request_id}] ⚡ No elements need chunking - skipping parallel processing")
            return
        
        logger.info(f"[{request_id}] 🔄 Processing {len(elements_to_chunk)} elements in parallel...")
        
        # Define chunk processing function for parallel execution
        def process_element_chunks(element_data):
            element, element_idx = element_data
            content = element.content
            
            # Dynamic chunk sizing based on content density
            content_density = self._calculate_content_density(content)
            adjusted_chunk_size = int(base_chunk_size * content_density)
            
            # Split content with semantic awareness
            chunk_elements = self._split_content_aware(
                content, adjusted_chunk_size, element.start_char,
                char_to_page_map, element.element_id, counter_start + element_idx * 100
            )
            
            return element.element_id, chunk_elements
        
        # Process elements in parallel with memory-conscious batching
        max_workers = min(8, len(elements_to_chunk))  # Increased for ultra-fast speed with higher RAM
        
        async def process_chunks_with_semaphore():
            # Use semaphore to control concurrent chunk processing
            async with CHUNK_PROCESSING_SEMAPHORE:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Prepare data for parallel processing
                    element_data = [(elem, idx) for idx, elem in enumerate(elements_to_chunk)]
                    
                    # Execute parallel processing
                    results = list(executor.map(process_element_chunks, element_data))
                    
                    return results
        
        # Run the parallel processing (this will be called from within async context)
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use thread executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    element_data = [(elem, idx) for idx, elem in enumerate(elements_to_chunk)]
                    results = list(executor.map(process_element_chunks, element_data))
            else:
                # Sync context
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    element_data = [(elem, idx) for idx, elem in enumerate(elements_to_chunk)]
                    results = list(executor.map(process_element_chunks, element_data))
        except Exception as e:
            logger.warning(f"[{request_id}] ⚠️ Parallel processing failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            results = []
            for idx, element in enumerate(elements_to_chunk):
                result = process_element_chunks((element, idx))
                results.append(result)
        
        # Combine results and update parent elements
        total_new_chunks = 0
        for element_id, chunk_elements in results:
            # Find the parent element and update its children
            parent_element = next((e for e in elements if e.element_id == element_id), None)
            if parent_element:
                for chunk_elem in chunk_elements:
                    new_elements.append(chunk_elem)
                    parent_element.children_ids.append(chunk_elem.element_id)
                    total_new_chunks += 1
        
        elements.extend(new_elements)
        logger.info(f"[{request_id}] ✅ PARALLEL chunking complete: {total_new_chunks} new chunks created")


# 5. LIGHTWEIGHT RAG SYSTEM FOR 1GB RAM
# ==============================================================================
class LightweightRAGSystem:
    def __init__(self):
        self.processor = HierarchicalSemanticProcessor()
        self.embedding_model = None
        self.tokenizer = None
        self.reranker_model = None
        
        # ACCURACY-OPTIMIZED models for 2GB RAM - better quality answers
        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'  # Better model (~440MB) for higher accuracy
        self.reranker_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'  # More accurate reranker (6 layers)
        self.llm_model = genai.GenerativeModel('gemini-1.5-flash',
            generation_config={
                'temperature': 0.05,  # Lower temperature for more accurate responses
                'top_p': 0.7,         # More focused sampling
                'top_k': 20,          # Limited candidates for consistency
                'max_output_tokens': 2048,  # Longer responses for comprehensive answers
                'candidate_count': 1
            }
        )
        
        # ACCURACY-OPTIMIZED retrieval parameters 
        self.top_k_initial = 18   # More candidates for better coverage
        self.top_k_final = 8      # More final results for comprehensive answers
        
        # ACCURACY-FOCUSED batch processing for 2GB RAM
        self.base_batch_size = 8     # Smaller batches for better accuracy
        self.max_batch_size = 12     # Moderate size for accuracy vs speed balance
        self.enable_half_precision = False  # Keep disabled for stability
        
        # ACCURACY-OPTIMIZED settings
        self.skip_reranking = False  # Always use reranking for accuracy
        self.rerank_threshold = 0.5  # Lower threshold = more reranking = better accuracy
        self.max_chunks_for_speed = 80  # More chunks for comprehensive coverage
        
        # Performance monitoring (minimal)
        self.performance_stats = {
            'total_embeddings_computed': 0,
            'average_batch_size': 0,
            'memory_cleanups': 0
        }

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size for BGE-base model on 2GB RAM"""
        current_memory = get_memory_usage()
        
        # BGE-base optimized for 2GB RAM system
        if current_memory > 1.5:  # If using more than 1.5GB
            return 2  # Conservative when approaching limits
        elif DEVICE == "cuda":
            # GPU with CPU offloading - moderate batches for 2GB RAM
            return min(6, CPU_COUNT // 2)
        else:
            # CPU-only processing - optimized for 2GB RAM
            return min(4, max(2, CPU_COUNT // 4))  # Better batching with more RAM

    def load_models(self):
        """Load BGE-base models optimized for ACCURACY on 2GB RAM"""
        logger.info("--- Loading BGE-base Models for ACCURACY on 2GB RAM ---")
        
        try:
            # Optimize for accuracy
            torch.set_num_threads(6)  # More threads for better processing
            
            # Load BGE-base embedding model (better accuracy)
            logger.info(f"Loading BGE-base embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device="cpu",  # Force CPU
                cache_folder="/tmp/sentence_transformers",
            #    trust_remote_code=True  # Required for BGE models
            )
            
            # Disable gradient tracking for speed
            if hasattr(self.embedding_model, 'eval'):
                self.embedding_model.eval()
            
            for param in self.embedding_model.parameters():
                param.requires_grad = False
            
            # Skip tokenizer loading to save memory and time
            self.tokenizer = None
            
            # Load more accurate reranker (6-layer instead of 2-layer)
            logger.info(f"Loading accurate reranker: {self.reranker_model_name}")
            self.reranker_model = CrossEncoder(
                self.reranker_model_name,
                device="cpu",
                max_length=128  # Increased for better context understanding
            )
            
            # Disable gradients for reranker
            if hasattr(self.reranker_model.model, 'eval'):
                self.reranker_model.model.eval()
            
            for param in self.reranker_model.model.parameters():
                param.requires_grad = False
            
            # Minimal warmup for BGE-base (accuracy focused)
            logger.info("🔥 Quick warmup for BGE-base model...")
            with torch.no_grad():
                _ = self.embedding_model.encode(
                    ["Test"],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=1,
                    convert_to_numpy=True
                )
            
            # Quick warmup reranker
            with torch.no_grad():
                _ = self.reranker_model.predict([["test", "test"]], show_progress_bar=False)
            
            # Quick cleanup
            gc.collect()
            
            logger.info(f"✅ BGE-base models loaded for ACCURACY | Max batch size: {self.max_batch_size}")
            logger.info(f"💾 Memory usage after loading: {get_memory_usage():.2f}GB")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise

    async def download_and_extract(self, pdf_url: str) -> List[Tuple[int, str]]:
        """Download and extract PDF with memory efficiency"""
        logger.info(f"📥 Downloading PDF from: {pdf_url[:100]}...")
        
        start_time = time.time()
        
        try:
            # Download with memory streaming
            timeout = aiohttp.ClientTimeout(total=15)  # Quick timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(pdf_url) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail=f"PDF download failed: {response.status}")
                    
                    pdf_data = await response.read()
            
            # Extract text with minimal memory usage
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            pages = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                pages.append((page_num + 1, text))
                
                # Memory cleanup every few pages
                if page_num % 5 == 0:
                    gc.collect()
            
            doc.close()
            del pdf_data
            gc.collect()
            
            download_time = time.time() - start_time
            logger.info(f"📄 PDF extracted: {len(pages)} pages in {download_time:.2f}s")
            
            return pages
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

    async def _embed_texts_lightweight(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings optimized for ACCURACY with 2GB RAM"""
        if not texts:
            return np.array([])
        
        try:
            # Process balanced batches for accuracy
            all_embeddings = []
            batch_size = self.max_batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Balanced memory check for accuracy focus
                if get_memory_usage() > 1.7:  # Higher threshold for accuracy
                    force_memory_cleanup()
                    batch_size = max(4, batch_size // 2)  # Still reasonable for accuracy
                
                with torch.no_grad():
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        normalize_embeddings=True,  # Critical for BGE model accuracy
                        show_progress_bar=False,
                        batch_size=len(batch_texts),
                        convert_to_numpy=True,
                        device="cpu",
                        precision='float32'  # Higher precision for better accuracy
                    )
                    all_embeddings.append(batch_embeddings)
                
                # Regular cleanup for accuracy balance
                if i % batch_size == 0:  # Regular cleanup
                    gc.collect()
            
            # Combine embeddings
            if all_embeddings:
                embeddings = np.vstack(all_embeddings)
            else:
                embeddings = np.zeros((len(texts), 768))  # BGE-base dimension (increased from 384)
            
            self.performance_stats['total_embeddings_computed'] += len(texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return np.zeros((len(texts), 768))  # BGE-base dimension

    def _rerank_minimal(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """SPEED-OPTIMIZED reranking for 2GB RAM"""
        if not chunks:
            return chunks
        
        # Enhanced limits for accuracy
        max_rerank_chunks = 15  # Increased from 10 for better accuracy
        if len(chunks) > max_rerank_chunks:
            chunks = chunks[:max_rerank_chunks]
        
        logger.info(f"� Reranking {len(chunks)} chunks (SPEED MODE)")
        
        try:
            # Enhanced reranker input with larger text limits for accuracy
            reranker_input = [[query[:200], chunk['text'][:400]] for chunk in chunks]  # Increased context for accuracy
            
            with torch.no_grad():
                cross_scores = self.reranker_model.predict(
                    reranker_input, 
                    show_progress_bar=False,
                    batch_size=4  # Smaller batch for better accuracy
                )
            
            # Apply scores
            for i, chunk in enumerate(chunks):
                chunk['rerank_score'] = float(cross_scores[i])
            
            return sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Reranking error: {e}")
            # Fallback to similarity scores
            for chunk in chunks:
                chunk['rerank_score'] = chunk.get('semantic_score', 0.0)
            return sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)

    async def retrieve_and_rerank_with_timeout(self, query: str, chunks_with_metadata: List[Dict], 
                                               chunk_embeddings: np.ndarray, request_id: str,
                                               timeout_manager: TimeoutManager) -> List[Dict[str, Any]]:
        """Fast retrieval with timeout awareness"""
        start_time = time.time()
        
        # Check timeout before starting
        if timeout_manager.is_timeout_exceeded():
            logger.warning(f"[{request_id}] ⏰ Timeout exceeded before retrieval")
            return []
        
        if len(chunks_with_metadata) == 0:
            return []
        
        # Quick query embedding
        query_embedding = await self._embed_texts_lightweight([query])
        if query_embedding.size == 0:
            return []
        
        query_embedding = query_embedding.flatten()
        
        # Enhanced similarity computation for accuracy
        # Normalize embeddings for better cosine similarity
        chunk_embeddings_norm = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        query_embedding_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute cosine similarities for better semantic matching
        similarities = np.dot(chunk_embeddings_norm, query_embedding_norm)
        
        # Apply semantic enhancement based on text characteristics
        for i, chunk in enumerate(chunks_with_metadata):
            # Boost relevance for chunks with question keywords
            text_lower = chunk['text'].lower()
            query_lower = query.lower()
            
            # Keyword overlap boost
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            overlap_ratio = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
            similarities[i] += overlap_ratio * 0.1  # 10% boost for keyword overlap
            
            # Length penalty for very short chunks (often incomplete)
            if len(chunk['text']) < 50:
                similarities[i] *= 0.8  # Penalize very short chunks
        
        # Quick candidate selection
        top_indices = np.argsort(similarities)[-self.top_k_initial:][::-1]
        candidate_chunks = [chunks_with_metadata[i] for i in top_indices]
        
        # Apply semantic scores
        for i, chunk in enumerate(candidate_chunks):
            chunk['semantic_score'] = similarities[top_indices[i]]
        
        # Check timeout before reranking
        if timeout_manager.is_timeout_approaching():
            logger.warning(f"[{request_id}] ⏰ Timeout approaching, skipping reranking")
            candidate_chunks.sort(key=lambda x: x['semantic_score'], reverse=True)
            final_chunks = candidate_chunks[:self.top_k_final]
            for chunk in final_chunks:
                chunk['rerank_score'] = chunk['semantic_score']
            return final_chunks
        
        # Rerank if time allows
        final_chunks = self._rerank_minimal(query, candidate_chunks[:self.top_k_final])
        
        return final_chunks

    async def answer_question_with_timeout(self, question: str, contexts: List[Dict], 
                                           request_id: str, timeout_manager: TimeoutManager) -> str:
        """Generate answer with timeout awareness - ACCURACY OPTIMIZED"""
        
        # Check timeout before answering
        if timeout_manager.is_timeout_exceeded():
            logger.warning(f"[{request_id}] ⏰ Timeout exceeded, returning partial answer")
            return "⏰ Response timeout reached. Please try again with a simpler query."
        
        if not contexts:
            return "I couldn't find relevant information to answer your question."
        
        # Build comprehensive context for accuracy
        context_text = ""
        context_count = min(len(contexts), 5)  # More context for accuracy
        for i, ctx in enumerate(contexts[:context_count]):
            context_text += f"Context {i+1} (Page {ctx.get('page', 'N/A')}, Relevance: {ctx.get('rerank_score', 0):.3f}):\n{ctx['text'][:800]}\n\n"  # Longer context
        
        # Build accuracy-focused prompt
        prompt = f"""You are an expert document analyst. Based on the following comprehensive context from the document, provide a thorough, accurate, and well-reasoned answer to the question.

DOCUMENT CONTEXT:
{context_text}

QUESTION: {question}

INSTRUCTIONS FOR MAXIMUM ACCURACY:
- Analyze ALL provided context segments carefully
- Synthesize information across multiple contexts when relevant
- Provide specific details, numbers, dates, and quotes when available
- If information is incomplete, clearly state what is known and what might be missing
- Cross-reference information between contexts to ensure consistency
- Give preference to more recent or authoritative information if there are conflicts
- Structure your answer logically with clear reasoning
- Include relevant page references where helpful

COMPREHENSIVE ANSWER:"""
        
        try:
            # Check timeout one more time
            if timeout_manager.is_timeout_approaching():
                logger.warning(f"[{request_id}] ⏰ Timeout approaching during answer generation")
                return f"{contexts[0]['text'][:200]}..."
            
            # Generate answer with more time for comprehensive responses
            response = await asyncio.wait_for(
                self._generate_answer_async(prompt),
                timeout=max(8, RESPONSE_TIMEOUT - timeout_manager.get_elapsed_time())  # More time for accuracy
            )
            
            return response.text.strip()
            
        except asyncio.TimeoutError:
            logger.warning(f"[{request_id}] ⏰ Answer generation timeout")
            return f"{contexts[0]['text'][:200]}..."
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Answer generation failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}"

    async def _generate_answer_async(self, prompt: str):
        """Generate answer asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.llm_model.generate_content, prompt)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get minimal performance statistics"""
        return {
            "total_embeddings": self.performance_stats['total_embeddings_computed'],
            "memory_cleanups": self.performance_stats['memory_cleanups'],
            "current_memory_gb": get_memory_usage()
        }
        
        # Warmup reranker
        _ = self.reranker_model.predict([["warmup query", "warmup text"]], show_progress_bar=False)
        
        # Force garbage collection after model loading
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"--- JinaAI Models loaded successfully | Max batch size: {self.max_batch_size} ---")

    def _extract_pdf_text(self, content: bytes) -> List[Tuple[int, str]]:
        """Memory-optimized PDF text extraction for heavy files under 1.7GB RAM"""
        pages = []
        initial_memory = get_memory_usage()
        logger.info(f"📄 Starting PDF extraction | Memory: {initial_memory:.2f}GB")
        
        try:
            with fitz.open(stream=content, filetype="pdf") as doc:
                total_pages = len(doc)
                logger.info(f"📄 Processing {total_pages} pages")
                
                # For heavy files, process in larger batches with memory cleanup
                if total_pages > 20 or len(content) > 20_000_000:  # Heavy file detection
                    batch_size = 24  # Process 24 pages at a time for maximum speed with 4GB RAM
                    for batch_start in range(0, total_pages, batch_size):
                        batch_end = min(batch_start + batch_size, total_pages)
                        
                        # Process batch
                        for page_num in range(batch_start, batch_end):
                            try:
                                text = doc[page_num].get_text("text")
                                if text and text.strip():
                                    # Truncate very long pages to save memory
                                    if len(text) > 50000:
                                        text = text[:50000] + "...[truncated]"
                                    pages.append((page_num + 1, text))
                            except Exception as e:
                                logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        
                        # Memory cleanup after each batch - optimized for 4GB RAM
                        gc.collect()
                        current_memory = get_memory_usage()
                        if current_memory > 3.5:  # Increased threshold for 4GB RAM
                            logger.warning(f"⚠️ High memory usage: {current_memory:.2f}GB - forcing cleanup")
                            force_memory_cleanup()
                        
                        logger.info(f"📄 Processed batch {batch_start+1}-{batch_end} | Memory: {current_memory:.2f}GB")
                else:
                    # Normal processing for smaller files
                    for page_num, page in enumerate(doc):
                        try:
                            text = page.get_text("text")
                            if text and text.strip():
                                pages.append((page_num + 1, text))
                        except Exception as e:
                            logger.warning(f"Error extracting page {page_num + 1}: {e}")
        
        except Exception as e:
            logger.error(f"❌ PDF extraction failed: {e}")
            return []
        
        finally:
            # Final cleanup
            gc.collect()
            final_memory = get_memory_usage()
            logger.info(f"📄 PDF extraction complete: {len(pages)} pages | Memory: {final_memory:.2f}GB")
        
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
        """ULTRA-OPTIMIZED embedding generation with concurrent processing"""
        import torch
        start_time = time.time()
        
        if not texts:
            return np.array([])
        
        # Memory monitoring with relaxed thresholds for speed
        current_memory = get_memory_usage()
        logger.info(f"💾 Memory before embedding: {current_memory:.2f}GB")
        
        # Early cleanup if memory is very high (increased threshold)
        if current_memory > 3.0:
            logger.warning(f"⚠️ High memory before embedding - forcing cleanup")
            force_memory_cleanup()
            current_memory = get_memory_usage()
        
        # Optimized text preprocessing for speed
        processed_texts = []
        for text in texts:
            # Increased limits for better accuracy while maintaining speed
            cleaned = ' '.join(text.strip().split())[:512]  # Increased from 128 to 512
            processed_texts.append(cleaned)
        
        logger.info(f"⚡ Processing {len(processed_texts)} texts with ULTRA-FAST concurrent settings")
        
        # Generate embeddings with CONCURRENT processing
        try:
            async with EMBEDDING_SEMAPHORE:
                # Use optimized batch processing for speed
                embeddings = await self._generate_embeddings_batch_parallel(
                    processed_texts, batch_size=min(24, len(processed_texts))  # Increased batch size for higher RAM
                )
                
                embedding_time = time.time() - start_time
                final_memory = get_memory_usage()
                logger.info(f"⚡ CONCURRENT embedding complete: {embedding_time:.3f}s | Memory: {final_memory:.2f}GB")
                
                return embeddings
                
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return np.zeros((len(texts), 768))  # BGE-base dimension

    async def _generate_embeddings_batch_parallel(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Generate embeddings with CONCURRENT processing and optimized memory management"""
        if not texts:
            return np.array([])
        
        try:
            all_embeddings = []
            
            # Process in CONCURRENT batches for maximum speed
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Memory check with increased thresholds for speed
                current_memory = get_memory_usage()
                if current_memory > 3.2:  # Increased threshold for better performance
                    # Use smaller batches only when memory is very high
                    smaller_batch_size = max(1, batch_size // 2)
                    for j in range(0, len(batch_texts), smaller_batch_size):
                        small_batch = batch_texts[j:j + smaller_batch_size]
                        batch_embeddings = self.embedding_model.encode(
                            small_batch,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            batch_size=len(small_batch),
                            convert_to_numpy=True,
                            device=DEVICE  # Use optimal device
                        )
                        all_embeddings.append(batch_embeddings)
                else:
                    # FAST batch processing for good memory conditions
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=len(batch_texts),  # Process full batch
                        convert_to_numpy=True,
                        device=DEVICE  # Use optimal device for speed
                    )
                    all_embeddings.append(batch_embeddings)
            
            # Combine all embeddings efficiently
            if all_embeddings:
                embeddings = np.vstack(all_embeddings)
            else:
                embeddings = np.zeros((len(texts), 768))  # BGE-base dimension
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Parallel embedding batch failed: {e}")
            return np.zeros((len(texts), 768))  # BGE-base dimension

    async def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts optimized for SPEED"""
        if not texts:
            return np.array([])
        
        try:
            # Use sentence-transformers with SPEED-OPTIMIZED settings
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=min(24, len(texts)),  # Increased for higher RAM capacity
                convert_to_numpy=True,
                device=DEVICE  # Use GPU if available for speed
            )
            return embeddings
        except Exception as e:
            logger.error(f"🔥 Batch embedding error: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), 768))  # BGE-base dimension

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
        """Ultra-optimized reranking for heavy files under 1.7GB RAM"""
        import torch
        
        if not chunks:
            return chunks
        
        # Limit reranking for heavy files to save memory
        max_rerank_chunks = 8  # Severely reduced from more chunks
        if len(chunks) > max_rerank_chunks:
            chunks = chunks[:max_rerank_chunks]
        
        logger.info(f"🔄 Reranking {len(chunks)} chunks with ultra-conservative settings")
        
        # Prepare reranker input with ultra-aggressive truncation for heavy files
        reranker_input = [[query[:100], chunk['text'][:200]] for chunk in chunks]  # Severely reduced limits
        
        try:
            with torch.no_grad():  # Disable gradients
                # Process one item at a time for heavy files
                cross_scores = []
                
                for i, input_pair in enumerate(reranker_input):
                    # Process single item
                    batch_scores = self.reranker_model.predict(
                        [input_pair], 
                        show_progress_bar=False,
                        batch_size=1  # Force single item processing
                    )
                    cross_scores.extend(batch_scores)
                    
                    # Memory cleanup optimized for 4GB RAM
                    if i % 4 == 0 and i > 0:  # Less frequent cleanup for speed
                        gc.collect()
                        current_memory = get_memory_usage()
                        if current_memory > 3.5:  # Higher threshold for 4GB RAM
                            logger.warning(f"⚠️ Memory during reranking: {current_memory:.2f}GB - forcing cleanup")
                            force_memory_cleanup()
        
            # Apply scores and sort
            for i, chunk in enumerate(chunks):
                chunk['rerank_score'] = float(cross_scores[i])
            
            return sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Heavy file reranking error: {e}")
            # Force cleanup on error
            force_memory_cleanup()
            # Fallback: return chunks with similarity scores as rerank scores
            for chunk in chunks:
                chunk['rerank_score'] = chunk.get('semantic_score', 0.0)
            return sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)

    async def retrieve_and_rerank(self, query: str, chunks_with_metadata: List[Dict], 
                                  chunk_embeddings: np.ndarray, request_id: str) -> List[Dict[str, Any]]:
        """Fast retrieval with semantic search and intelligent reranking"""
        start_time = time.time()
        
        if len(chunks_with_metadata) == 0:
            return []
        
        # Fast query embedding
        embed_start = time.time()
        query_embedding = await self._embed_with_advanced_caching([query])
        embed_time = time.time() - embed_start
        logger.info(f"[{request_id}] ⚡ Query embedding: {embed_time:.3f}s")
        
        if query_embedding.size == 0:
            return []
        
        # Ultra-fast similarity computation
        sim_start = time.time()
        query_embedding = query_embedding.flatten()
        
        # Optimized similarity computation based on dataset size
        if chunk_embeddings.shape[0] > 1000:
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
        else:
            # Direct computation for smaller datasets
            similarities = np.dot(chunk_embeddings, query_embedding)
        
        sim_time = time.time() - sim_start
        logger.info(f"[{request_id}] ⚡ Similarity computation: {sim_time:.3f}s")
        
        # Advanced candidate selection
        select_start = time.time()
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
            
            # Page proximity boosting
            page_boost = self._calculate_page_proximity_boost(chunk, candidate_chunks)
            
            chunk['semantic_score'] = base_score + level_boost + content_boost + page_boost
        
        select_time = time.time() - select_start
        logger.info(f"[{request_id}] ⚡ Candidate selection: {select_time:.3f}s")
        
        # Smart selective reranking for accuracy
        rerank_start = time.time()
        top_score = max(similarities[candidate_indices]) if candidate_indices else 0
        
        if top_score < self.rerank_threshold:
            # Low confidence - use reranking for accuracy
            logger.info(f"[{request_id}] 🎯 Low confidence ({top_score:.3f}), applying reranking")
            reranked_chunks = await asyncio.get_event_loop().run_in_executor(
                None, self._rerank_with_optimization, query, candidate_chunks
            )
            final_chunks = reranked_chunks[:self.top_k_final]
        else:
            # High confidence - skip reranking for speed
            logger.info(f"[{request_id}] 🚀 High confidence ({top_score:.3f}), skipping reranking")
            candidate_chunks.sort(key=lambda x: x['semantic_score'], reverse=True)
            final_chunks = candidate_chunks[:self.top_k_final]
            
            # Add semantic scores as rerank scores for compatibility
            for chunk in final_chunks:
                chunk['rerank_score'] = chunk['semantic_score']
        
        rerank_time = time.time() - rerank_start
        logger.info(f"[{request_id}] ⚡ Reranking: {rerank_time:.3f}s")
        
        total_time = time.time() - start_time
        top_final_score = final_chunks[0]['rerank_score'] if final_chunks else 0
        logger.info(f"[{request_id}] 🎯 Retrieval complete: {total_time:.3f}s | Top score: {top_final_score:.4f}")
        
        return final_chunks

    def _select_diverse_candidates(self, similarities: np.ndarray, chunks: List[Dict], 
                                 top_k: int) -> List[int]:
        """Enhanced diverse candidate selection with semantic clustering for accuracy"""
        # Get top candidates based on similarity (more candidates for better diversity)
        top_indices = np.argsort(similarities)[-top_k * 4:][::-1]  # Increased multiplier
        
        selected_indices = []
        similarity_threshold = 0.03  # Tighter threshold for better accuracy
        
        for idx in top_indices:
            if len(selected_indices) >= top_k:
                break
            
            chunk = chunks[idx]
            is_diverse = True
            current_similarity = similarities[idx]
            
            # Enhanced diversity checks with more granular analysis
            for selected_idx in selected_indices:
                selected_chunk = chunks[selected_idx]
                selected_similarity = similarities[selected_idx]
                
                # 1. Avoid identical or very similar content
                if (chunk.get('element_id') == selected_chunk.get('element_id') or
                    chunk.get('title') == selected_chunk.get('title')):
                    is_diverse = False
                    break
                
                # 2. Enhanced semantic similarity check
                if abs(current_similarity - selected_similarity) < similarity_threshold:
                    # More sophisticated content overlap detection
                    chunk_words = set(chunk.get('text', '').lower().split()[:30])  # Increased window
                    selected_words = set(selected_chunk.get('text', '').lower().split()[:30])
                    overlap = len(chunk_words & selected_words) / max(len(chunk_words | selected_words), 1)
                    if overlap > 0.6:  # Slightly reduced threshold for better diversity
                        is_diverse = False
                        break
                
                # 3. Enhanced hierarchical diversity
                chunk_level = chunk.get('level', 4)
                selected_level = selected_chunk.get('level', 4)
                if (chunk_level == selected_level and 
                    chunk.get('parent_id') == selected_chunk.get('parent_id') and
                    abs(current_similarity - selected_similarity) < similarity_threshold * 1.5):
                    is_diverse = False
                    break
                
                # 4. Page proximity with content type consideration
                chunk_pages = set(chunk.get('pages', [chunk.get('page', 0)]))
                selected_pages = set(selected_chunk.get('pages', [selected_chunk.get('page', 0)]))
                page_overlap = len(chunk_pages & selected_pages)
                
                # More nuanced page overlap handling
                if (page_overlap > 0 and 
                    abs(current_similarity - selected_similarity) < similarity_threshold * 2 and
                    chunk.get('element_type') == selected_chunk.get('element_type')):
                    # Check if content is truly different despite page overlap
                    content_diff = abs(len(chunk.get('text', '')) - len(selected_chunk.get('text', '')))
                    if content_diff < 100:  # Similar length content on same page
                        is_diverse = False
                        break
            
            if is_diverse:
                selected_indices.append(idx)
        
        # Enhanced fallback with semantic clustering
        if len(selected_indices) < top_k:
            remaining_candidates = [idx for idx in top_indices if idx not in selected_indices]
            needed = top_k - len(selected_indices)
            
            # Apply semantic clustering to remaining candidates
            clustered_candidates = self._apply_semantic_clustering(
                remaining_candidates, similarities, chunks, needed
            )
            selected_indices.extend(clustered_candidates)
        
        return selected_indices

    def _apply_semantic_clustering(self, candidate_indices: List[int], similarities: np.ndarray, 
                                 chunks: List[Dict], needed: int) -> List[int]:
        """Apply semantic clustering to select diverse candidates"""
        if not candidate_indices or needed <= 0:
            return []
        
        # Group candidates by similarity ranges for better diversity
        similarity_groups = {}
        for idx in candidate_indices:
            sim_score = similarities[idx]
            # Create similarity bins
            sim_bin = round(sim_score, 1)  # Group by 0.1 similarity intervals
            if sim_bin not in similarity_groups:
                similarity_groups[sim_bin] = []
            similarity_groups[sim_bin].append(idx)
        
        # Select from different similarity groups
        selected = []
        sorted_groups = sorted(similarity_groups.keys(), reverse=True)
        
        # Round-robin selection from similarity groups
        group_index = 0
        while len(selected) < needed and any(similarity_groups.values()):
            current_group = sorted_groups[group_index % len(sorted_groups)]
            if similarity_groups[current_group]:
                # Select best candidate from this group
                group_candidates = similarity_groups[current_group]
                best_idx = max(group_candidates, key=lambda x: similarities[x])
                selected.append(best_idx)
                similarity_groups[current_group].remove(best_idx)
                
            group_index += 1
            
            # Clean up empty groups
            if not similarity_groups[current_group]:
                del similarity_groups[current_group]
                sorted_groups = [g for g in sorted_groups if g in similarity_groups]
        
        return selected[:needed]

    def _calculate_content_quality_boost(self, chunk: Dict) -> float:
        """Enhanced content quality boost with multiple quality indicators"""
        content = chunk.get('text', '')
        
        # Enhanced structure analysis
        structure_indicators = (
            len(re.findall(r'^\d+\.|\n\d+\.|\n[A-Z]\.', content, re.MULTILINE)) +
            len(re.findall(r'^\s*[-•]\s+', content, re.MULTILINE)) +  # Bullet points
            len(re.findall(r'^\s*[A-Z][A-Z\s]{10,}$', content, re.MULTILINE))  # Headings
        )
        structure_boost = min(0.03, structure_indicators * 0.008)  # Increased weight
        
        # Enhanced length scoring with optimal ranges
        word_count = chunk.get('word_count', len(content.split()))
        if 75 <= word_count <= 200:      # Sweet spot for detailed answers
            length_boost = 0.025
        elif 200 < word_count <= 400:   # Good for comprehensive content
            length_boost = 0.02
        elif 50 <= word_count < 75:     # Decent for specific facts
            length_boost = 0.015
        elif 400 < word_count <= 600:   # Still good but longer
            length_boost = 0.01
        else:
            length_boost = 0.0
        
        # Content density and quality indicators
        sentences = len(re.findall(r'[.!?]+', content))
        avg_sentence_length = word_count / max(sentences, 1)
        
        # Prefer moderate sentence lengths (indicates good readability)
        if 10 <= avg_sentence_length <= 25:
            readability_boost = 0.01
        else:
            readability_boost = 0.0
        
        # Boost for content with definitions, examples, or explanations
        quality_indicators = (
            len(re.findall(r'\b(defined as|means|refers to|such as|for example|including)\b', content, re.IGNORECASE)) +
            len(re.findall(r'\b(because|therefore|thus|hence|consequently)\b', content, re.IGNORECASE)) +
            len(re.findall(r'\b(section|subsection|clause|paragraph)\s+\d+', content, re.IGNORECASE))
        )
        quality_boost = min(0.02, quality_indicators * 0.005)
        
        return structure_boost + length_boost + readability_boost + quality_boost

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
        """Create enriched context with advanced formatting for better LLM comprehension"""
        context_parts = []
        seen_pages = set()
        
        for i, ctx in enumerate(contexts):
            pages = ctx.get('pages', [ctx.get('page', 1)])
            page_range = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"
            
            # Enhanced metadata for better context
            section_info = f"Section: {ctx.get('title', 'Unknown')}" if ctx.get('title') else ""
            level_info = f"Level {ctx.get('level', 'Unknown')}" if ctx.get('level') is not None else ""
            relevance_score = f"Relevance: {ctx.get('rerank_score', 0):.3f}"
            element_type = f"Type: {ctx.get('element_type', 'Unknown')}"
            
            # Build enhanced context header
            header_parts = [page_range]
            if section_info and "Unknown" not in section_info:
                header_parts.append(section_info)
            if level_info and "Unknown" not in level_info:
                header_parts.append(level_info)
            if element_type and "Unknown" not in element_type:
                header_parts.append(element_type)
            header_parts.append(relevance_score)
            
            header = f"--- Context {i+1} | {' | '.join(header_parts)} ---"
            
            # Enhanced content processing
            content = ctx['text']
            
            # Add hierarchical context if available
            if ctx.get('parent_id') and section_info:
                content = f"[Context: {section_info}]\n{content}"
            
            # Smart truncation that preserves sentence boundaries
            if len(content) > 1800:  # Increased limit for better accuracy
                sentences = re.split(r'[.!?]+', content)
                truncated_sentences = []
                current_length = 0
                
                for sentence in sentences:
                    if current_length + len(sentence) > 1700:
                        break
                    truncated_sentences.append(sentence)
                    current_length += len(sentence)
                
                if truncated_sentences:
                    content = '.'.join(truncated_sentences) + "."
                    if len(content) < len(ctx['text']) * 0.8:  # Only add truncation note if significantly cut
                        content += "\n[...content continues...]"
                else:
                    content = content[:1700] + "... [truncated]"
            
            context_parts.append(f"{header}\n{content}")
            seen_pages.update(pages)
        
        # Add summary footer for better LLM understanding
        total_pages = len(seen_pages)
        summary_footer = f"\n--- Summary: {len(contexts)} relevant contexts from {total_pages} page(s) ---"
        
        return "\n\n".join(context_parts) + summary_footer

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
        
        return {
            **self.performance_stats,
            "model_info": {
                "embedding_model": self.embedding_model_name,
                "device": DEVICE,
                "gpu_memory_gb": GPU_MEMORY_GB,
                "half_precision": self.enable_half_precision,
                "max_batch_size": self.max_batch_size
            }
        }

    async def clear_all_caches(self):
        """Simplified cleanup for maximum speed"""
        
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
        logger.info("🗑️ Memory optimized for maximum speed")


# Global RAG system instance
rag_system = LightweightRAGSystem()

async def cleanup_memory():
    """Background task for memory cleanup"""
    logger.info("🗑️ Starting background memory cleanup...")
    start_time = time.time()
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if DEVICE == "cuda" and torch:
        torch.cuda.empty_cache()
    
    cleanup_time = time.time() - start_time
    memory_after = get_memory_usage()
    logger.info(f"🗑️ Background cleanup complete: {cleanup_time:.3f}s | Memory: {memory_after:.2f}GB")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan management"""
    # Startup
    logger.info("🚀 Starting BGE-base RAG Server for 2GB RAM...")
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
    title="Lightweight RAG System", 
    version="11.0.0",
    description="BGE-base optimized RAG system with 30-second timeout for 2GB RAM Amazon Lightsail",
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
    """Main endpoint for document processing and question answering optimized for 2GB RAM with 30-second timeout"""
    verify_token(authorization)
    request_id = f"req_{int(time.time() * 1000)}"
    
    # Initialize timeout manager
    timeout_manager = TimeoutManager()
    start_time = time.time()
    initial_memory = get_memory_usage()
    logger.info(f"[{request_id}] � Starting request | Memory: {initial_memory:.2f}GB | Timeout: {RESPONSE_TIMEOUT}s")
    
    # Early memory check - reject if already too high for 2GB systems
    if initial_memory > 1.6:
        logger.error(f"[{request_id}] ❌ Memory too high ({initial_memory:.2f}GB) - rejecting request")
        raise HTTPException(status_code=503, detail=f"Server memory too high: {initial_memory:.2f}GB")
    
    # SINGLE REQUEST processing optimized for 2GB RAM
    async with REQUEST_SEMAPHORE:
        logger.info(f"[{request_id}] 📝 Processing {len(request.questions)} questions (2GB RAM optimized mode)")

        try:
            # Check initial timeout
            if timeout_manager.is_timeout_exceeded():
                logger.error(f"[{request_id}] ⏰ Initial timeout exceeded")
                return QueryResponse(answers=["⏰ Request timeout before processing started."])
            
            # Memory check for 2GB system
            current_memory = get_memory_usage()
            if current_memory > 1.4:
                logger.warning(f"[{request_id}] ⚠️ Memory at {current_memory:.2f}GB - forcing cleanup")
                force_memory_cleanup()
                await asyncio.sleep(0.1)
                current_memory = get_memory_usage()
                if current_memory > 1.3:
                    raise HTTPException(status_code=503, detail=f"Memory too high for processing: {current_memory:.2f}GB")
            
            # PDF extraction with timeout awareness
            pdf_start = time.time()
            logger.info(f"[{request_id}] � Starting PDF extraction...")
            
            if timeout_manager.is_timeout_approaching():
                logger.warning(f"[{request_id}] ⏰ Timeout approaching before PDF extraction")
                return QueryResponse(answers=["⏰ Timeout approaching. Please try with a smaller document."])
            
            pages = await rag_system.download_and_extract(request.documents)
            pdf_time = time.time() - pdf_start
            pdf_memory = get_memory_usage()
            logger.info(f"[{request_id}] 📄 PDF extraction: {len(pages)} pages in {pdf_time:.3f}s | Memory: {pdf_memory:.2f}GB")
            
            # Check timeout after PDF extraction
            if timeout_manager.is_timeout_approaching():
                logger.warning(f"[{request_id}] ⏰ Timeout approaching after PDF extraction")
                return QueryResponse(answers=["⏰ Timeout reached during document processing. Please try with a smaller document."])
            
            # Memory cleanup for 2GB system
            if pdf_memory > 1.4:
                logger.warning(f"[{request_id}] ⚠️ Memory at {pdf_memory:.2f}GB after PDF - cleanup")
                force_memory_cleanup()
                pdf_memory = get_memory_usage()
            
            # Document structure analysis with timeout
            structure_start = time.time()
            logger.info(f"[{request_id}] 🏗️ Starting structure analysis...")
            
            if timeout_manager.is_timeout_approaching():
                logger.warning(f"[{request_id}] ⏰ Timeout approaching before structure analysis")
                return QueryResponse(answers=["⏰ Timeout reached. Partial processing completed."])
            
            hierarchical_elements = rag_system.processor.extract_hierarchical_structure(pages, request_id)
            
            # Clear pages immediately to save memory
            del pages
            gc.collect()
            
            structure_time = time.time() - structure_start
            structure_memory = get_memory_usage()
            logger.info(f"[{request_id}] 🏗️ Structure analysis: {structure_time:.3f}s | {len(hierarchical_elements)} elements | Memory: {structure_memory:.2f}GB")
            
            # Check timeout after structure analysis
            if timeout_manager.is_timeout_approaching():
                logger.warning(f"[{request_id}] ⏰ Timeout approaching after structure analysis")
                return QueryResponse(answers=["⏰ Timeout reached during document analysis."])
            
            # Memory cleanup for 2GB system
            if structure_memory > 1.4:
                logger.warning(f"[{request_id}] ⚠️ Memory at {structure_memory:.2f}GB after structure - cleanup")
                force_memory_cleanup()
            
            # Convert to retrieval chunks with timeout awareness
            chunk_start = time.time()
            logger.info(f"[{request_id}] 📚 Creating retrieval chunks...")
            
            if timeout_manager.is_timeout_approaching():
                logger.warning(f"[{request_id}] ⏰ Timeout approaching before chunking")
                return QueryResponse(answers=["⏰ Timeout reached during document preparation."])
            
            chunks_with_metadata = rag_system.processor.get_chunks_for_retrieval(hierarchical_elements)
            
            # Clear hierarchical elements to save memory
            del hierarchical_elements
            gc.collect()
            
            chunk_time = time.time() - chunk_start
            chunk_memory = get_memory_usage()
            logger.info(f"[{request_id}] 📚 Chunk creation: {chunk_time:.3f}s | {len(chunks_with_metadata)} chunks | Memory: {chunk_memory:.2f}GB")
            
            # Check timeout before embedding
            if timeout_manager.is_timeout_approaching():
                logger.warning(f"[{request_id}] ⏰ Timeout approaching before embedding")
                return QueryResponse(answers=["⏰ Timeout reached during chunk preparation."])
            
            # Memory check before embedding for 2GB system
            if chunk_memory > 1.4:
                logger.warning(f"[{request_id}] ⚠️ Memory at {chunk_memory:.2f}GB before embedding - cleanup")
                force_memory_cleanup()
            
            # Embedding generation with timeout awareness
            embed_start = time.time()
            chunk_texts = [c['text'] for c in chunks_with_metadata]
            logger.info(f"[{request_id}] 🔢 Generating embeddings for {len(chunk_texts)} texts...")
            
            if timeout_manager.is_timeout_approaching():
                logger.warning(f"[{request_id}] ⏰ Timeout approaching before embedding")
                return QueryResponse(answers=["⏰ Timeout reached before embedding generation."])
            
            chunk_embeddings = await rag_system._embed_texts_lightweight(chunk_texts)
            embed_time = time.time() - embed_start
            embed_memory = get_memory_usage()
            
            # Final memory cleanup after embedding
            if check_memory_threshold():
                force_memory_cleanup()
                rag_system.performance_stats['memory_cleanups'] += 1
            
            processing_time = time.time() - pdf_start
            logger.info(f"[{request_id}] ⚡ Document processing complete: {processing_time:.3f}s (PDF: {pdf_time:.3f}s, Structure: {structure_time:.3f}s, Chunks: {chunk_time:.3f}s, Embedding: {embed_time:.3f}s) | Memory: {embed_memory:.2f}GB")

            # Process questions with timeout awareness
            answers = []
            for q_idx, question in enumerate(request.questions):
                q_request_id = f"{request_id}_q{q_idx+1}"
                q_start = time.time()
                
                # Check timeout before each question
                if timeout_manager.is_timeout_exceeded():
                    logger.warning(f"[{q_request_id}] ⏰ Timeout exceeded, adding timeout answer")
                    timeout_manager.add_partial_result("⏰ Request timeout reached. Please try again with fewer questions.")
                    break
                
                if timeout_manager.is_timeout_approaching():
                    logger.warning(f"[{q_request_id}] ⏰ Timeout approaching, processing with reduced context")
                    # Use reduced processing for speed
                    contexts = await rag_system.retrieve_and_rerank_with_timeout(
                        question, chunks_with_metadata[:50], chunk_embeddings[:50], q_request_id, timeout_manager
                    )
                else:
                    # Normal processing
                    contexts = await rag_system.retrieve_and_rerank_with_timeout(
                        question, chunks_with_metadata, chunk_embeddings, q_request_id, timeout_manager
                    )
                
                # Generate answer with timeout awareness
                answer = await rag_system.answer_question_with_timeout(question, contexts, q_request_id, timeout_manager)
                answers.append(answer)
                timeout_manager.add_partial_result(answer)
                
                q_time = time.time() - q_start
                logger.info(f"[{q_request_id}] ✅ Question processed in {q_time:.3f}s")
                
                # Break if timeout exceeded
                if timeout_manager.is_timeout_exceeded():
                    logger.warning(f"[{request_id}] ⏰ Timeout exceeded after question {q_idx+1}")
                    break
            
            # If no answers were generated due to timeout, use partial results
            if not answers and timeout_manager.get_partial_results():
                answers = timeout_manager.get_partial_results()
            elif not answers:
                answers = ["Request timeout reached before any questions could be processed."]
            
            # Ensure we have answers for all questions
            while len(answers) < len(request.questions):
                answers.append("timeout constraints.")
            
            # Final timing summary
            total_time = time.time() - start_time
            final_memory = get_memory_usage()
            memory_delta = final_memory - initial_memory
            
            logger.info(f"[{request_id}] ✅ REQUEST COMPLETE (1GB RAM Optimized):")
            logger.info(f"[{request_id}]   📊 Total time: {total_time:.3f}s")
            logger.info(f"[{request_id}]   📄 Document processing: {processing_time:.3f}s")
            logger.info(f"[{request_id}]   🤖 Questions processed: {len(answers)}/{len(request.questions)}")
            logger.info(f"[{request_id}]   🧠 Memory: {initial_memory:.2f}GB → {final_memory:.2f}GB (Δ{memory_delta:+.2f}GB)")
            logger.info(f"[{request_id}]   🗑️ Memory cleanups: {rag_system.performance_stats.get('memory_cleanups', 0)}")
            
            # Force final cleanup if memory usage is high for 2GB system
            if final_memory > 1.3:
                force_memory_cleanup()
                logger.info(f"[{request_id}] 🗑️ Final cleanup | Memory: {get_memory_usage():.2f}GB")
            
            return QueryResponse(answers=answers)
        
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"[{request_id}] ❌ Error after {error_time:.3f}s: {e}")
            # Return timeout message if error occurred near timeout
            if timeout_manager.is_timeout_approaching() or timeout_manager.is_timeout_exceeded():
                return QueryResponse(answers=[f"⏰ Request timeout: {str(e)}"])
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


# --- Concurrency Monitoring Endpoint ---
@app.get("/concurrency/status")
async def get_concurrency_status(authorization: str = Header(None)):
    """Get current concurrency status and queue information"""
    verify_token(authorization)
    
    return {
        "concurrency_limits": {
            "max_concurrent_requests": 12,  # Updated for ultra-fast speed
            "max_concurrent_processing": 12,  # Updated for ultra-fast speed 
            "current_active_requests": 12 - REQUEST_SEMAPHORE._value,
            "current_processing_requests": 12 - PROCESSING_SEMAPHORE._value,
            "requests_waiting": max(0, len(getattr(REQUEST_SEMAPHORE, '_waiters', []))),
            "processing_waiting": max(0, len(getattr(PROCESSING_SEMAPHORE, '_waiters', [])))
        },
        "resource_info": {
            "cpu_cores": CPU_COUNT,
            "device": DEVICE,
            "thread_pool_size": global_executor._max_workers
        },
        "recommendations": {
            "optimal_concurrent_requests": "4-6 for heavy files under 4GB RAM",
            "memory_per_request_estimate": "500MB-1GB for heavy files",
            "expected_queue_time": "15-45 seconds for heavy files"
        }
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
        
        # Benchmark direct embedding (no cache)
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
        # Simple memory cleanup without cache management
        
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
        
        # System resources
        system_resources = {
            "device": DEVICE,
            "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False,
            "cpu_count": CPU_COUNT,
            "memory_usage_gb": get_memory_usage()
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
        
        # System ready - no cache stats needed for speed
        
        health_data = {
            "status": overall_status,
            "timestamp": time.time(),
            "service": "Lightweight RAG System",
            "version": "11.0.0",
            "api_version": "v1",
            "checkpoint_duration_ms": round(checkpoint_time * 1000, 2),
            "models": model_status,
            "embedding_test": {
                "passed": embedding_test_passed,
                "duration_ms": round(embedding_time * 1000, 2)
            },
            "system": system_resources,
            "timeout_config": {
                "response_timeout": RESPONSE_TIMEOUT,
                "warning_threshold": TIMEOUT_WARNING
            },
            "concurrency": {
                "max_requests": REQUEST_SEMAPHORE._value,
                "max_processing": PROCESSING_SEMAPHORE._value
            },
            "endpoints": {
                "main": "/api/v1/hackrx/run",
                "health": "/health",
                "api_health": "/api/health",
                "performance": "/performance/stats"
            }
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"API health checkpoint failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "service": "Fast Parallel RAG System",
            "version": "10.0.0",
            "error": str(e)
        }

# End of API endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Lightweight RAG System",
        "version": "11.0.0",
        "description": "Memory-optimized RAG system with 30-second timeout for 1GB RAM environments",
        "model": rag_system.embedding_model_name,
        "device": DEVICE,
        "timeout_seconds": RESPONSE_TIMEOUT,
        "memory_target": "1GB",
        "concurrency": {
            "max_requests": REQUEST_SEMAPHORE._value,
            "max_processing": PROCESSING_SEMAPHORE._value
        },
        "endpoints": {
            "main": "/api/v1/hackrx/run",
            "health": "/health",
            "performance": "/performance/stats",
            "system_info": "/performance/system-info"
        }
    }


# 7. SERVER EXECUTION
# ==============================================================================
if __name__ == "__main__":
    import sys
    
    # Support configurable port via command line argument
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid port argument: {sys.argv[1]}, using default 8000")
    
    logger.info("🚀 Starting Lightweight RAG Server (30s timeout, 1GB RAM optimized)...")
    logger.info(f"🔥 Hardware: {DEVICE} | CPU cores: {CPU_COUNT} | Memory target: 1GB")
    logger.info(f"⚡ Models: {rag_system.embedding_model_name} (lightweight)")
    logger.info(f"🔄 Single Process Mode: {REQUEST_SEMAPHORE._value} concurrent request")
    logger.info(f"🌐 Server port: {port} | TIMEOUT-AWARE PROCESSING")
    logger.info(f"🧠 Memory-conservative processing with 30-second timeout")
    logger.info(f"⏰ Features: Timeout management, partial results, 1GB RAM optimization")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,
        workers=1,     # Single worker for memory efficiency
        loop="asyncio",
        log_level="info",
        # Optimized for 1GB RAM with timeout processing
        timeout_keep_alive=60,        # Reasonable timeout for memory-constrained system
        limit_concurrency=2,          # Very limited for 1GB RAM
        limit_max_requests=100,       # Lower limit for memory efficiency
        access_log=True,              # Enable for monitoring requests
        # Conservative settings for 1GB RAM
        h11_max_incomplete_event_size=16384,  # Smaller buffer for memory efficiency
        ws_max_size=16777216,         # Smaller websocket for memory efficiency
        lifespan="on"
    )