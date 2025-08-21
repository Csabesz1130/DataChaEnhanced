# src/cognitive/rag_engine.py
"""
Cognitive RAG Engine - Pillar 3: Advanced Semantic Processing
Implements sophisticated RAG architecture with hybrid search and adaptive chunking
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json
import pickle
from pathlib import Path

# Vector and ML libraries
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Language processing
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel

# Database and caching
import redis
from sqlalchemy import create_engine, Column, String, Text, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..processors.bulletproof_pipeline import DocumentChunk, ChunkType, IntermediateRepresentation

logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class StoredChunk(Base):
    __tablename__ = 'document_chunks'
    
    chunk_id = Column(String(64), primary_key=True)
    file_id = Column(String(64), index=True)
    content = Column(Text)
    embedding_vector = Column(Text)  # JSON serialized
    chunk_type = Column(String(32))
    source_location = Column(String(256))
    confidence = Column(Float)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class SearchStrategy(Enum):
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class SearchResult:
    chunk_id: str
    content: str
    score: float
    source_location: str
    metadata: Dict[str, Any]
    search_method: str
    confidence: float

@dataclass
class QueryContext:
    query: str
    user_id: Optional[str] = None
    file_filter: Optional[List[str]] = None
    chunk_type_filter: Optional[List[ChunkType]] = None
    min_confidence: float = 0.3
    max_results: int = 10
    search_strategy: SearchStrategy = SearchStrategy.ADAPTIVE

class CognitiveRAGEngine:
    """
    Advanced RAG engine with hybrid search, adaptive chunking, and semantic understanding
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = None
        self.tokenizer = None
        self.nlp = None
        
        # Vector stores
        self.faiss_index = None
        self.chunk_id_map = {}  # Maps FAISS index to chunk IDs
        
        # Keyword search
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunk_contents = []
        
        # Caching
        self.redis_client = None
        self.cache_enabled = config.get('cache_enabled', True)
        
        # Database
        self.db_engine = None
        self.db_session = None
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'vector_searches': 0,
            'keyword_searches': 0,
            'hybrid_searches': 0,
            'cache_hits': 0,
            'average_response_time': 0.0
        }
        
        # Initialize components
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize all components asynchronously"""
        try:
            await self._init_ai_models()
            await self._init_vector_stores()
            await self._init_database()
            await self._init_cache()
            logger.info("Cognitive RAG Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise
    
    async def _init_ai_models(self):
        """Initialize AI models and NLP components"""
        
        # Embedding model
        model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        
        # Advanced tokenizer for query analysis
        tokenizer_name = self.config.get('tokenizer_model', 'distilbert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # SpaCy for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Question-answering pipeline
        self.qa_pipeline = pipeline("question-answering", 
                                   model="distilbert-base-cased-distilled-squad")
        
        logger.info("AI models initialized")
    
    async def _init_vector_stores(self):
        """Initialize FAISS vector store and TF-IDF for hybrid search"""
        
        # Initialize FAISS index (will be populated when chunks are added)
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1,
            max_df=0.9
        )
        
        logger.info("Vector stores initialized")
    
    async def _init_database(self):
        """Initialize database connection"""
        db_url = self.config.get('database_url', 'sqlite:///nexus_cognition.db')
        self.db_engine = create_engine(db_url)
        Base.metadata.create_all(self.db_engine)
        
        Session = sessionmaker(bind=self.db_engine)
        self.db_session = Session()
        
        logger.info("Database initialized")
    
    async def _init_cache(self):
        """Initialize Redis cache"""
        if self.cache_enabled:
            try:
                redis_url = self.config.get('redis_url', 'redis://localhost:6379/1')
                self.redis_client = redis.from_url(redis_url)
                await self.redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis cache unavailable: {e}")
                self.cache_enabled = False
    
    async def ingest_document(self, ir: IntermediateRepresentation) -> Dict[str, Any]:
        """
        Ingest document with advanced chunking and metadata enrichment
        """
        start_time = datetime.now()
        
        # Enhance chunks with metadata and context
        enhanced_chunks = await self._enhance_chunks(ir.chunks, ir)
        
        # Generate embeddings
        embeddings = await self._generate_embeddings(enhanced_chunks)
        
        # Store in vector database
        await self._store_chunks_vector(enhanced_chunks, embeddings, ir)
        
        # Store in database
        await self._store_chunks_db(enhanced_chunks, embeddings, ir)
        
        # Update search indices
        await self._update_search_indices(enhanced_chunks)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'file_id': ir.file_id,
            'chunks_processed': len(enhanced_chunks),
            'embeddings_generated': len(embeddings),
            'processing_time': processing_time,
            'status': 'success'
        }
        
        logger.info(f"Document ingested: {ir.file_id}, {len(enhanced_chunks)} chunks")
        return result
    
    async def _enhance_chunks(
        self, 
        chunks: List[DocumentChunk], 
        ir: IntermediateRepresentation
    ) -> List[DocumentChunk]:
        """
        Enhance chunks with metadata enrichment and context
        """
        enhanced_chunks = []
        
        for chunk in chunks:
            # Add file context to content
            enhanced_content = self._add_contextual_metadata(chunk, ir)
            
            # Create enhanced chunk
            enhanced_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                type=chunk.type,
                content=enhanced_content,
                source_location=chunk.source_location,
                confidence=chunk.confidence,
                metadata={
                    **chunk.metadata,
                    'file_type': ir.file_type.value,
                    'original_filename': ir.original_filename,
                    'file_id': ir.file_id,
                    'enhancement_timestamp': datetime.now().isoformat()
                },
                processing_method=chunk.processing_method
            )
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _add_contextual_metadata(self, chunk: DocumentChunk, ir: IntermediateRepresentation) -> str:
        """Add file and context metadata to chunk content for better embedding"""
        
        # Create contextual prefix
        context_parts = []
        
        # File type context
        context_parts.append(f"[File Type: {ir.file_type.value.upper()}]")
        
        # File name context
        if ir.original_filename:
            context_parts.append(f"[File: {ir.original_filename}]")
        
        # Chunk type context
        context_parts.append(f"[Content Type: {chunk.type.value}]")
        
        # Source location context
        if chunk.source_location:
            context_parts.append(f"[Location: {chunk.source_location}]")
        
        # Combine with original content
        context_prefix = " ".join(context_parts)
        enhanced_content = f"{context_prefix} {chunk.content}"
        
        return enhanced_content
    
    async def _generate_embeddings(self, chunks: List[DocumentChunk]) -> List[np.ndarray]:
        """Generate embeddings for chunks"""
        
        contents = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batches for efficiency
        batch_size = self.config.get('embedding_batch_size', 32)
        embeddings = []
        
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def _store_chunks_vector(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: List[np.ndarray],
        ir: IntermediateRepresentation
    ):
        """Store chunks and embeddings in FAISS vector store"""
        
        if not embeddings:
            return
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        
        # Get current index size
        start_idx = self.faiss_index.ntotal
        
        # Add to FAISS index
        self.faiss_index.add(embedding_matrix)
        
        # Update chunk ID mapping
        for i, chunk in enumerate(chunks):
            self.chunk_id_map[start_idx + i] = chunk.chunk_id
    
    async def _store_chunks_db(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: List[np.ndarray],
        ir: IntermediateRepresentation
    ):
        """Store chunks in database"""
        
        for chunk, embedding in zip(chunks, embeddings):
            stored_chunk = StoredChunk(
                chunk_id=chunk.chunk_id,
                file_id=ir.file_id,
                content=chunk.content,
                embedding_vector=json.dumps(embedding.tolist()),
                chunk_type=chunk.type.value,
                source_location=chunk.source_location,
                confidence=chunk.confidence,
                metadata=chunk.metadata
            )
            
            self.db_session.merge(stored_chunk)
        
        self.db_session.commit()
    
    async def _update_search_indices(self, chunks: List[DocumentChunk]):
        """Update TF-IDF index for keyword search"""
        
        # Add new content to corpus
        new_contents = [chunk.content for chunk in chunks]
        self.chunk_contents.extend(new_contents)
        
        # Refit TF-IDF (in production, use incremental updates)
        if len(self.chunk_contents) > 0:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunk_contents)
    
    async def search(self, query_context: QueryContext) -> List[SearchResult]:
        """
        Advanced search with hybrid vector + keyword approach
        """
        start_time = datetime.now()
        
        # Check cache first
        cache_key = self._generate_cache_key(query_context)
        if self.cache_enabled and self.redis_client:
            cached_result = await self.redis_client.get(cache_key)
            if cached_result:
                self.search_stats['cache_hits'] += 1
                return json.loads(cached_result)
        
        # Determine search strategy
        strategy = await self._determine_search_strategy(query_context)
        
        # Execute search based on strategy
        if strategy == SearchStrategy.VECTOR_ONLY:
            results = await self._vector_search(query_context)
        elif strategy == SearchStrategy.KEYWORD_ONLY:
            results = await self._keyword_search(query_context)
        elif strategy == SearchStrategy.HYBRID:
            results = await self._hybrid_search(query_context)
        else:  # ADAPTIVE
            results = await self._adaptive_search(query_context)
        
        # Apply filters
        filtered_results = await self._apply_filters(results, query_context)
        
        # Rank and limit results
        final_results = await self._rank_and_limit(filtered_results, query_context)
        
        # Cache results
        if self.cache_enabled and self.redis_client:
            await self.redis_client.setex(
                cache_key, 
                300,  # 5 minutes
                json.dumps([r.__dict__ for r in final_results])
            )
        
        # Update stats
        self._update_search_stats(strategy, start_time)
        
        return final_results
    
    async def _determine_search_strategy(self, query_context: QueryContext) -> SearchStrategy:
        """Intelligently determine the best search strategy"""
        
        if query_context.search_strategy != SearchStrategy.ADAPTIVE:
            return query_context.search_strategy
        
        query = query_context.query.lower()
        
        # Use keyword search for specific identifiers
        if any(pattern in query for pattern in ['id:', 'number:', 'code:', 'ref:']):
            return SearchStrategy.KEYWORD_ONLY
        
        # Use vector search for conceptual queries
        if any(pattern in query for pattern in ['similar', 'like', 'related', 'about']):
            return SearchStrategy.VECTOR_ONLY
        
        # Use hybrid for most other queries
        return SearchStrategy.HYBRID
    
    async def _vector_search(self, query_context: QueryContext) -> List[SearchResult]:
        """Pure vector similarity search"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_context.query])
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(
            query_embedding, 
            min(query_context.max_results * 2, self.faiss_index.ntotal)
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            chunk_id = self.chunk_id_map.get(idx)
            if chunk_id:
                chunk = self._get_chunk_by_id(chunk_id)
                if chunk:
                    results.append(SearchResult(
                        chunk_id=chunk_id,
                        content=chunk.content,
                        score=float(score),
                        source_location=chunk.source_location,
                        metadata=chunk.metadata,
                        search_method='vector',
                        confidence=chunk.confidence
                    ))
        
        return results
    
    async def _keyword_search(self, query_context: QueryContext) -> List[SearchResult]:
        """TF-IDF based keyword search"""
        
        if self.tfidf_matrix is None or len(self.chunk_contents) == 0:
            return []
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query_context.query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-query_context.max_results * 2:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                # Map back to chunk (this is simplified - in production, maintain proper mapping)
                chunk_content = self.chunk_contents[idx]
                
                # Find corresponding chunk in database
                chunk = self.db_session.query(StoredChunk).filter(
                    StoredChunk.content == chunk_content
                ).first()
                
                if chunk:
                    results.append(SearchResult(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        score=float(similarities[idx]),
                        source_location=chunk.source_location,
                        metadata=json.loads(chunk.metadata) if chunk.metadata else {},
                        search_method='keyword',
                        confidence=chunk.confidence
                    ))
        
        return results
    
    async def _hybrid_search(self, query_context: QueryContext) -> List[SearchResult]:
        """Combine vector and keyword search results"""
        
        # Get results from both methods
        vector_results = await self._vector_search(query_context)
        keyword_results = await self._keyword_search(query_context)
        
        # Combine and re-rank
        combined_results = {}
        
        # Add vector results with weight
        vector_weight = self.config.get('vector_weight', 0.7)
        for result in vector_results:
            combined_results[result.chunk_id] = SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score * vector_weight,
                source_location=result.source_location,
                metadata=result.metadata,
                search_method='hybrid_vector',
                confidence=result.confidence
            )
        
        # Add keyword results with weight
        keyword_weight = self.config.get('keyword_weight', 0.3)
        for result in keyword_results:
            if result.chunk_id in combined_results:
                # Combine scores
                combined_results[result.chunk_id].score += result.score * keyword_weight
                combined_results[result.chunk_id].search_method = 'hybrid_combined'
            else:
                combined_results[result.chunk_id] = SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=result.score * keyword_weight,
                    source_location=result.source_location,
                    metadata=result.metadata,
                    search_method='hybrid_keyword',
                    confidence=result.confidence
                )
        
        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x.score, 
            reverse=True
        )
        
        return sorted_results
    
    async def _adaptive_search(self, query_context: QueryContext) -> List[SearchResult]:
        """Adaptive search that analyzes query characteristics"""
        
        # Analyze query using NLP
        query_analysis = await self._analyze_query(query_context.query)
        
        # Choose strategy based on analysis
        if query_analysis['is_factual']:
            return await self._keyword_search(query_context)
        elif query_analysis['is_conceptual']:
            return await self._vector_search(query_context)
        else:
            return await self._hybrid_search(query_context)
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics using NLP"""
        
        analysis = {
            'is_factual': False,
            'is_conceptual': False,
            'entities': [],
            'intent': 'general'
        }
        
        if self.nlp:
            doc = self.nlp(query)
            
            # Extract entities
            analysis['entities'] = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Detect factual vs conceptual queries
            factual_indicators = ['what', 'when', 'where', 'who', 'how many', 'which']
            conceptual_indicators = ['why', 'how', 'explain', 'describe', 'similar']
            
            query_lower = query.lower()
            
            if any(indicator in query_lower for indicator in factual_indicators):
                analysis['is_factual'] = True
                analysis['intent'] = 'factual'
            elif any(indicator in query_lower for indicator in conceptual_indicators):
                analysis['is_conceptual'] = True
                analysis['intent'] = 'conceptual'
        
        return analysis
    
    async def _apply_filters(
        self, 
        results: List[SearchResult], 
        query_context: QueryContext
    ) -> List[SearchResult]:
        """Apply various filters to search results"""
        
        filtered_results = results
        
        # Confidence filter
        if query_context.min_confidence > 0:
            filtered_results = [
                r for r in filtered_results 
                if r.confidence >= query_context.min_confidence
            ]
        
        # File filter
        if query_context.file_filter:
            filtered_results = [
                r for r in filtered_results
                if r.metadata.get('file_id') in query_context.file_filter
            ]
        
        # Chunk type filter
        if query_context.chunk_type_filter:
            type_values = [ct.value for ct in query_context.chunk_type_filter]
            filtered_results = [
                r for r in filtered_results
                if self._get_chunk_by_id(r.chunk_id).type.value in type_values
            ]
        
        return filtered_results
    
    async def _rank_and_limit(
        self, 
        results: List[SearchResult], 
        query_context: QueryContext
    ) -> List[SearchResult]:
        """Final ranking and limiting of results"""
        
        # Sort by score (already done in most cases, but ensure)
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        # Limit results
        limited_results = sorted_results[:query_context.max_results]
        
        return limited_results
    
    def _get_chunk_by_id(self, chunk_id: str) -> Optional[StoredChunk]:
        """Get chunk from database by ID"""
        return self.db_session.query(StoredChunk).filter(
            StoredChunk.chunk_id == chunk_id
        ).first()
    
    def _generate_cache_key(self, query_context: QueryContext) -> str:
        """Generate cache key for query"""
        key_parts = [
            query_context.query,
            str(query_context.max_results),
            str(query_context.min_confidence),
            str(query_context.search_strategy.value)
        ]
        
        if query_context.file_filter:
            key_parts.append("|".join(sorted(query_context.file_filter)))
        
        if query_context.chunk_type_filter:
            key_parts.append("|".join(sorted([ct.value for ct in query_context.chunk_type_filter])))
        
        key_string = "|".join(key_parts)
        return f"search_cache:{hash(key_string)}"
    
    def _update_search_stats(self, strategy: SearchStrategy, start_time: datetime):
        """Update search performance statistics"""
        self.search_stats['total_searches'] += 1
        
        if strategy == SearchStrategy.VECTOR_ONLY:
            self.search_stats['vector_searches'] += 1
        elif strategy == SearchStrategy.KEYWORD_ONLY:
            self.search_stats['keyword_searches'] += 1
        else:
            self.search_stats['hybrid_searches'] += 1
        
        response_time = (datetime.now() - start_time).total_seconds()
        self.search_stats['average_response_time'] = (
            (self.search_stats['average_response_time'] * (self.search_stats['total_searches'] - 1) + response_time) /
            self.search_stats['total_searches']
        )
    
    async def answer_question(self, question: str, context_chunks: List[SearchResult]) -> Dict[str, Any]:
        """Use QA pipeline to answer questions based on retrieved context"""
        
        if not context_chunks:
            return {
                'answer': 'No relevant information found.',
                'confidence': 0.0,
                'sources': []
            }
        
        # Combine context from top chunks
        context = "\n\n".join([chunk.content for chunk in context_chunks[:5]])
        
        # Use QA pipeline
        try:
            result = self.qa_pipeline(question=question, context=context)
            
            return {
                'answer': result['answer'],
                'confidence': result['score'],
                'sources': [
                    {
                        'chunk_id': chunk.chunk_id,
                        'source_location': chunk.source_location,
                        'relevance_score': chunk.score
                    }
                    for chunk in context_chunks[:3]
                ]
            }
        except Exception as e:
            logger.error(f"QA pipeline failed: {e}")
            return {
                'answer': 'Unable to generate answer from the available context.',
                'confidence': 0.0,
                'sources': []
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance and usage statistics"""
        return {
            'search_stats': self.search_stats,
            'vector_store_size': self.faiss_index.ntotal if self.faiss_index else 0,
            'total_chunks': len(self.chunk_contents),
            'cache_enabled': self.cache_enabled,
            'models_loaded': {
                'embedding_model': self.embedding_model is not None,
                'qa_pipeline': hasattr(self, 'qa_pipeline'),
                'nlp_model': self.nlp is not None
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.db_session:
            self.db_session.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Cognitive RAG Engine cleanup completed")


# Global engine instance
cognitive_engine: Optional[CognitiveRAGEngine] = None

def get_cognitive_engine() -> CognitiveRAGEngine:
    """Get or create the global Cognitive RAG Engine instance"""
    global cognitive_engine
    if cognitive_engine is None:
        config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'tokenizer_model': 'distilbert-base-uncased',
            'database_url': 'sqlite:///nexus_cognition.db',
            'redis_url': 'redis://localhost:6379/1',
            'cache_enabled': True,
            'embedding_batch_size': 32,
            'vector_weight': 0.7,
            'keyword_weight': 0.3
        }
        cognitive_engine = CognitiveRAGEngine(config)
    
    return cognitive_engine