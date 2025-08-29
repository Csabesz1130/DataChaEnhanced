"""
Bulletproof Pipeline - Robust document processing and chunking system
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of document chunks"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    FORMULA = "formula"
    MIXED = "mixed"


class ProcessingMethod(Enum):
    """Methods used for processing chunks"""
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"
    MANUAL = "manual"


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    chunk_id: str
    content: str
    chunk_type: ChunkType
    source_location: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    processing_method: ProcessingMethod = ProcessingMethod.RULE_BASED
    embedding_vector: Optional[List[float]] = None
    parent_document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"chunk_{datetime.now().timestamp()}_{hash(self.content) % 10000}"
        
        if not self.metadata:
            self.metadata = {}


@dataclass
class IntermediateRepresentation:
    """Intermediate representation of a document during processing"""
    doc_id: str
    source_path: str
    doc_type: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_chunk(self, chunk: DocumentChunk):
        """Add a chunk to the representation"""
        chunk.parent_document_id = self.doc_id
        chunk.chunk_index = len(self.chunks)
        self.chunks.append(chunk)
        self.updated_at = datetime.now()
    
    def get_chunks_by_type(self, chunk_type: ChunkType) -> List[DocumentChunk]:
        """Get chunks of a specific type"""
        return [chunk for chunk in self.chunks if chunk.chunk_type == chunk_type]
    
    def get_high_confidence_chunks(self, min_confidence: float = 0.7) -> List[DocumentChunk]:
        """Get chunks with confidence above threshold"""
        return [chunk for chunk in self.chunks if chunk.confidence >= min_confidence]


class BulletproofPipeline:
    """Robust document processing pipeline with error handling and fallbacks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.processing_stats = {
            'total_documents': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'total_chunks': 0,
            'average_confidence': 0.0
        }
    
    def process_document(self, file_path: str, doc_type: str = "auto") -> IntermediateRepresentation:
        """Process a document and return intermediate representation"""
        try:
            self.logger.info(f"Processing document: {file_path}")
            
            # Create intermediate representation
            ir = IntermediateRepresentation(
                doc_id=self._generate_doc_id(file_path),
                source_path=file_path,
                doc_type=doc_type or self._detect_doc_type(file_path)
            )
            
            # Process based on document type
            if doc_type == "excel" or file_path.endswith(('.xlsx', '.xls')):
                ir = self._process_excel(file_path, ir)
            elif doc_type == "csv" or file_path.endswith('.csv'):
                ir = self._process_csv(file_path, ir)
            elif doc_type == "text" or file_path.endswith('.txt'):
                ir = self._process_text(file_path, ir)
            else:
                ir = self._process_generic(file_path, ir)
            
            # Update statistics
            self.processing_stats['total_documents'] += 1
            self.processing_stats['successful_processing'] += 1
            self.processing_stats['total_chunks'] += len(ir.chunks)
            
            if ir.chunks:
                avg_confidence = sum(chunk.confidence for chunk in ir.chunks) / len(ir.chunks)
                self.processing_stats['average_confidence'] = avg_confidence
            
            self.logger.info(f"Successfully processed {file_path} into {len(ir.chunks)} chunks")
            return ir
            
        except Exception as e:
            self.logger.error(f"Failed to process document {file_path}: {str(e)}")
            self.processing_stats['failed_processing'] += 1
            
            # Return minimal representation on failure
            return IntermediateRepresentation(
                doc_id=self._generate_doc_id(file_path),
                source_path=file_path,
                doc_type=doc_type or "unknown",
                processing_status="failed",
                metadata={"error": str(e)}
            )
    
    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a unique document ID"""
        import hashlib
        return hashlib.md5(file_path.encode()).hexdigest()[:16]
    
    def _detect_doc_type(self, file_path: str) -> str:
        """Detect document type from file extension"""
        ext = file_path.lower().split('.')[-1]
        if ext in ['xlsx', 'xls']:
            return 'excel'
        elif ext == 'csv':
            return 'csv'
        elif ext == 'txt':
            return 'text'
        else:
            return 'unknown'
    
    def _process_excel(self, file_path: str, ir: IntermediateRepresentation) -> IntermediateRepresentation:
        """Process Excel files"""
        try:
            import pandas as pd
            
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    df = df.convert_dtypes()
                    
                    # Create chunks from each row
                    for row_idx, row in df.iterrows():
                        chunk_content = self._create_row_chunk(df.columns, row, sheet_name)
                        chunk = DocumentChunk(
                            chunk_id=f"{ir.doc_id}_sheet_{sheet_name}_row_{row_idx}",
                            content=chunk_content,
                            chunk_type=ChunkType.TABLE,
                            source_location=f"{file_path}:{sheet_name}:{row_idx}",
                            confidence=0.9,
                            metadata={
                                "sheet": sheet_name,
                                "row_index": int(row_idx),
                                "columns": list(df.columns),
                                "source": file_path
                            }
                        )
                        ir.add_chunk(chunk)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process sheet {sheet_name}: {str(e)}")
                    
        except ImportError:
            self.logger.warning("Pandas not available, Excel processing skipped")
            chunk = DocumentChunk(
                chunk_id=f"{ir.doc_id}_error",
                content="Excel processing requires pandas",
                chunk_type=ChunkType.TEXT,
                source_location=file_path,
                confidence=0.0,
                metadata={"error": "pandas_not_available"}
            )
            ir.add_chunk(chunk)
        
        return ir
    
    def _process_csv(self, file_path: str, ir: IntermediateRepresentation) -> IntermediateRepresentation:
        """Process CSV files"""
        try:
            import pandas as pd
            
            df = pd.read_csv(file_path)
            df = df.convert_dtypes()
            
            # Create chunks from each row
            for row_idx, row in df.iterrows():
                chunk_content = self._create_row_chunk(df.columns, row, "CSV")
                chunk = DocumentChunk(
                    chunk_id=f"{ir.doc_id}_row_{row_idx}",
                    content=chunk_content,
                    chunk_type=ChunkType.TABLE,
                    source_location=f"{file_path}:{row_idx}",
                    confidence=0.9,
                    metadata={
                        "row_index": int(row_idx),
                        "columns": list(df.columns),
                        "source": file_path
                    }
                )
                ir.add_chunk(chunk)
                
        except ImportError:
            self.logger.warning("Pandas not available, CSV processing skipped")
            chunk = DocumentChunk(
                chunk_id=f"{ir.doc_id}_error",
                content="CSV processing requires pandas",
                chunk_type=ChunkType.TEXT,
                source_location=file_path,
                confidence=0.0,
                metadata={"error": "pandas_not_available"}
            )
            ir.add_chunk(chunk)
        
        return ir
    
    def _process_text(self, file_path: str, ir: IntermediateRepresentation) -> IntermediateRepresentation:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into paragraphs
            paragraphs = content.split('\n\n')
            for i, para in enumerate(paragraphs):
                if para.strip():
                    chunk = DocumentChunk(
                        chunk_id=f"{ir.doc_id}_para_{i}",
                        content=para.strip(),
                        chunk_type=ChunkType.TEXT,
                        source_location=f"{file_path}:paragraph:{i}",
                        confidence=0.8,
                        metadata={
                            "paragraph_index": i,
                            "source": file_path
                        }
                    )
                    ir.add_chunk(chunk)
                    
        except Exception as e:
            self.logger.error(f"Failed to read text file {file_path}: {str(e)}")
            chunk = DocumentChunk(
                chunk_id=f"{ir.doc_id}_error",
                content=f"Error reading file: {str(e)}",
                chunk_type=ChunkType.TEXT,
                source_location=file_path,
                confidence=0.0,
                metadata={"error": str(e)}
            )
            ir.add_chunk(chunk)
        
        return ir
    
    def _process_generic(self, file_path: str, ir: IntermediateRepresentation) -> IntermediateRepresentation:
        """Process generic files"""
        chunk = DocumentChunk(
            chunk_id=f"{ir.doc_id}_generic",
            content=f"Generic file: {file_path}",
            chunk_type=ChunkType.TEXT,
            source_location=file_path,
            confidence=0.5,
            metadata={
                "source": file_path,
                "processing_method": "generic"
            }
        )
        ir.add_chunk(chunk)
        return ir
    
    def _create_row_chunk(self, columns: List[str], row: Any, sheet_name: str) -> str:
        """Create a text representation of a data row"""
        parts = []
        for col in columns:
            val = row[col]
            try:
                if pd.isna(val):
                    continue
                parts.append(f"{col}={val}")
            except Exception:
                continue
        
        if sheet_name != "CSV":
            return f"Sheet {sheet_name} | Row: " + "; ".join(parts)
        else:
            return "Row: " + "; ".join(parts)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()
