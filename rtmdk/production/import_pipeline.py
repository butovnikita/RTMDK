"""
rtmdk/production/import_pipeline.py — Multi-Format Data Import Pipeline.

Imports data from various sources into RTMDK memory.
Features:
- JSON, CSV, TSV import
- PDF text extraction (if pdfplumber available)
- Web page extraction (via requests + regex)
- Auto-batch indexing with progress tracking
"""

import os
import json
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Iterator
from dataclasses import dataclass


@dataclass
class ImportResult:
    """Result of an import operation."""
    total_items: int
    imported_items: int
    failed_items: int
    duration_seconds: float
    errors: List[str]


class ImportPipeline:
    """Imports data from various formats into RTMDK memory.
    
    Usage:
        pipeline = ImportPipeline(memory)
        
        # Import JSON
        result = pipeline.import_json("data.json", text_field="content")
        
        # Import CSV
        result = pipeline.import_csv("data.csv", text_column="description")
        
        # Import from web
        result = pipeline.import_url("https://example.com/article")
    """
    
    def __init__(self, memory, batch_size: int = 100, session_id: str = "import"):
        self.memory = memory
        self.batch_size = batch_size
        self.session_id = session_id
    
    def import_json(
        self,
        filepath: str,
        text_field: str = "text",
        query_field: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
    ) -> ImportResult:
        """Import from JSON file.
        
        Expected format: [{"text": "...", "query": "...", ...}, ...]
        or {"records": [{"text": "..."}, ...]}
        """
        t0 = time.time()
        path = Path(filepath)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle nested format
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        elif isinstance(data, dict):
            data = [data]
        
        return self._import_items(data, text_field, query_field, metadata_fields, time.time() - t0)
    
    def import_csv(
        self,
        filepath: str,
        text_column: str = "text",
        query_column: Optional[str] = None,
        delimiter: str = ",",
    ) -> ImportResult:
        """Import from CSV file."""
        t0 = time.time()
        items = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                items.append(row)
        
        return self._import_items(items, text_column, query_column, list(items[0].keys()) if items else None, time.time() - t0)
    
    def import_text(
        self,
        filepath: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> ImportResult:
        """Import plain text file, splitting into chunks."""
        t0 = time.time()
        path = Path(filepath)
        
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks
        chunks = self._chunk_text(text, chunk_size, overlap)
        
        imported = 0
        errors = []
        
        for i, chunk in enumerate(chunks):
            try:
                self.memory.save_context(
                    {"input": chunk, "session_id": self.session_id},
                    {"output": chunk}
                )
                imported += 1
            except Exception as e:
                errors.append(f"Chunk {i}: {e}")
        
        return ImportResult(
            total_items=len(chunks),
            imported_items=imported,
            failed_items=len(errors),
            duration_seconds=time.time() - t0,
            errors=errors,
        )
    
    def import_url(
        self,
        url: str,
        extract_text: bool = True,
    ) -> ImportResult:
        """Import content from a URL.
        
        Extracts text from HTML pages.
        """
        t0 = time.time()
        errors = []
        
        try:
            import requests
            
            resp = requests.get(url, timeout=30)
            html = resp.text
            
            # Extract text (simple approach)
            if extract_text:
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', html)
                # Remove extra whitespace
                text = re.sub(r'\s+', ' ', text).strip()
            else:
                text = html
            
            # Chunk and import
            chunks = self._chunk_text(text, 500, 50)
            imported = 0
            
            for chunk in chunks:
                try:
                    self.memory.save_context(
                        {"input": chunk, "session_id": self.session_id, "source": url},
                        {"output": chunk}
                    )
                    imported += 1
                except Exception as e:
                    errors.append(str(e))
            
            return ImportResult(
                total_items=len(chunks),
                imported_items=imported,
                failed_items=len(errors),
                duration_seconds=time.time() - t0,
                errors=errors,
            )
        except ImportError:
            return ImportResult(0, 0, 1, time.time() - t0, ["requests not installed"])
        except Exception as e:
            return ImportResult(0, 0, 1, time.time() - t0, [str(e)])
    
    def _import_items(
        self,
        items: List[Dict],
        text_field: str,
        query_field: Optional[str],
        metadata_fields: Optional[List[str]],
        setup_time: float,
    ) -> ImportResult:
        """Generic import from list of dicts."""
        t0 = time.time()
        imported = 0
        errors = []
        
        for i, item in enumerate(items):
            try:
                text = item.get(text_field, "")
                if not text:
                    errors.append(f"Item {i}: empty text field '{text_field}'")
                    continue
                
                query = item.get(query_field, text) if query_field else text
                
                # Build metadata
                metadata = {}
                if metadata_fields:
                    for field in metadata_fields:
                        if field in item and field != text_field:
                            metadata[field] = item[field]
                
                self.memory.save_context(
                    {"input": query, "session_id": self.session_id, **metadata},
                    {"output": text}
                )
                imported += 1
                
                # Progress
                if (i + 1) % self.batch_size == 0:
                    elapsed = time.time() - t0
                    print(f"  Imported {i+1}/{len(items)} ({elapsed:.0f}s)")
                    
            except Exception as e:
                errors.append(f"Item {i}: {e}")
        
        return ImportResult(
            total_items=len(items),
            imported_items=imported,
            failed_items=len(errors),
            duration_seconds=time.time() - t0 + setup_time,
            errors=errors[:10],  # First 10 errors
        )
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
