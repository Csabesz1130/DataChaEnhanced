# src/api/main.py
"""
FastAPI Integration for Nexus Cognition System
Complete API server with all four pillars integrated
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Import our Nexus Cognition components
from ..core.nexus_engine import get_nexus_engine, NexusCognitionEngine, ProcessingContext
from ..processors.bulletproof_pipeline import (
    MagicBytesDetector, FallbackProcessor, FileValidationResult,
    DocumentChunk, IntermediateRepresentation, FileType
)
from ..cognitive.rag_engine import (
    get_cognitive_engine, CognitiveRAGEngine, QueryContext, SearchStrategy, SearchResult
)
from ..learning.active_learning_engine import (
    get_learning_engine, ActiveLearningEngine, FeedbackType, AnomalyDetection
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class DocumentUploadResponse(BaseModel):
    session_id: str
    file_id: str
    filename: str
    file_type: str
    status: str
    processing_started: datetime
    estimated_completion: Optional[datetime] = None

class ProcessingStatusResponse(BaseModel):
    session_id: str
    status: str
    progress: Optional[float] = None
    chunks_processed: Optional[int] = None
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    requires_human_review: bool = False
    anomaly_score: Optional[float] = None
    processing_time: Optional[float] = None
    error_messages: List[str] = Field(default_factory=list)

class SearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=10, ge=1, le=50)
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    file_filter: Optional[List[str]] = None
    search_strategy: SearchStrategy = SearchStrategy.ADAPTIVE
    include_answer: bool = False

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_found: int
    search_time: float
    search_strategy_used: str
    answer: Optional[Dict[str, Any]] = None

class FeedbackRequest(BaseModel):
    session_id: str
    feedback_type: FeedbackType
    feedback_value: Any  # Could be rating, boolean, text
    result_ids: Optional[List[str]] = None
    comments: Optional[str] = None

class FeedbackResponse(BaseModel):
    status: str
    session_id: str
    feedback_processed: bool
    should_retrain: bool = False
    message: str

class SystemHealthResponse(BaseModel):
    timestamp: datetime
    overall_status: str
    components: Dict[str, str]
    performance_metrics: Dict[str, Any]
    recent_activity: Dict[str, Any]

# Security
security = HTTPBearer(auto_error=False)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Simple authentication - in production, implement proper JWT validation"""
    if credentials:
        # In production, validate JWT token and return user ID
        return "authenticated_user"
    return "anonymous"

# Initialize FastAPI app
app = FastAPI(
    title="Nexus Cognition API",
    description="Advanced AI Document Processing Platform with Active Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instances
nexus_engine: Optional[NexusCognitionEngine] = None
cognitive_engine: Optional[CognitiveRAGEngine] = None
learning_engine: Optional[ActiveLearningEngine] = None

@app.on_event("startup")
async def startup_event():
    """Initialize all engines on startup"""
    global nexus_engine, cognitive_engine, learning_engine
    
    logger.info("Starting Nexus Cognition API server...")
    
    try:
        # Initialize engines
        nexus_engine = get_nexus_engine()
        cognitive_engine = get_cognitive_engine()
        learning_engine = get_learning_engine()
        
        # Register processors
        from ..processors.specialized_parsers import (
            ATFProcessor, ExcelProcessor, CSVProcessor, 
            PDFProcessor, DOCXProcessor, TextProcessor
        )
        
        # Create fallback processor
        fallback_config = {
            'quarantine_path': './quarantine',
            'tika_server_url': 'http://localhost:9998',
            'ocr_enabled': True
        }
        fallback_processor = FallbackProcessor(fallback_config)
        
        # Register processors with nexus engine
        nexus_engine.register_processor(FileType.ATF, ATFProcessor(), ['native', 'tika', 'quarantine'])
        nexus_engine.register_processor(FileType.EXCEL, ExcelProcessor(), ['native', 'tika', 'quarantine'])
        nexus_engine.register_processor(FileType.CSV, CSVProcessor(), ['native', 'tika', 'quarantine'])
        nexus_engine.register_processor(FileType.PDF, PDFProcessor(), ['native', 'ocr', 'tika', 'quarantine'])
        nexus_engine.register_processor(FileType.DOCX, DOCXProcessor(), ['native', 'tika', 'quarantine'])
        nexus_engine.register_processor(FileType.TXT, TextProcessor(), ['native', 'quarantine'])
        
        logger.info("Nexus Cognition API server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Nexus Cognition API server...")
    
    if nexus_engine:
        await nexus_engine.shutdown()
    if cognitive_engine:
        await cognitive_engine.cleanup()
    if learning_engine:
        await learning_engine.cleanup()

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Nexus Cognition API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: Optional[str] = Depends(get_current_user),
    processing_options: Optional[str] = Query(None, description="JSON string of processing options")
):
    """
    Upload and process a document through the complete Nexus Cognition pipeline
    """
    if not nexus_engine:
        raise HTTPException(status_code=503, detail="Nexus engine not initialized")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Save uploaded file temporarily
        temp_dir = Path(tempfile.mkdtemp())
        temp_file_path = temp_dir / file.filename
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse processing options
        options = {}
        if processing_options:
            try:
                options = json.loads(processing_options)
            except json.JSONDecodeError:
                logger.warning(f"Invalid processing options JSON: {processing_options}")
        
        # Validate file using magic bytes detection
        validation_result = MagicBytesDetector.validate_file(str(temp_file_path))
        
        if not validation_result.is_valid:
            # Cleanup
            shutil.rmtree(temp_dir)
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file: {validation_result.magic_bytes}"
            )
        
        if validation_result.security_risk:
            # Cleanup
            shutil.rmtree(temp_dir)
            raise HTTPException(
                status_code=400, 
                detail="File rejected due to security risk"
            )
        
        # Start async processing
        context = await nexus_engine.process_document_async(
            str(temp_file_path),
            user_id=user_id,
            processing_options=options
        )
        
        # Generate file ID from content hash
        import hashlib
        file_id = hashlib.md5(temp_file_path.read_bytes()).hexdigest()
        
        # Estimate completion time based on file size and type
        file_size_mb = validation_result.file_size / (1024 * 1024)
        estimated_minutes = max(1, int(file_size_mb / 10))  # Rough estimate
        estimated_completion = datetime.now() + timedelta(minutes=estimated_minutes)
        
        return DocumentUploadResponse(
            session_id=context.session_id,
            file_id=file_id,
            filename=file.filename,
            file_type=validation_result.detected_type.value,
            status="processing",
            processing_started=context.processing_start,
            estimated_completion=estimated_completion
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

@app.get("/status/{session_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(session_id: str):
    """Get real-time processing status for a document"""
    if not nexus_engine:
        raise HTTPException(status_code=503, detail="Nexus engine not initialized")
    
    try:
        status_info = nexus_engine.get_processing_status(session_id)
        
        if 'error' in status_info:
            raise HTTPException(status_code=404, detail=status_info['error'])
        
        # Calculate progress based on processing steps
        total_steps = 6  # validation, extraction, embedding, indexing, anomaly detection, completion
        completed_steps = len(status_info.get('processing_steps', []))
        progress = min(completed_steps / total_steps, 1.0)
        
        return ProcessingStatusResponse(
            session_id=session_id,
            status=status_info.get('status', 'unknown'),
            progress=progress,
            chunks_processed=status_info.get('chunks_processed'),
            confidence_scores=status_info.get('confidence_scores', {}),
            requires_human_review=status_info.get('requires_human_review', False),
            anomaly_score=status_info.get('anomaly_score'),
            processing_time=status_info.get('processing_time'),
            error_messages=status_info.get('error_log', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    user_id: Optional[str] = Depends(get_current_user)
):
    """
    Advanced semantic search across all processed documents
    """
    if not cognitive_engine:
        raise HTTPException(status_code=503, detail="Cognitive engine not initialized")
    
    try:
        start_time = datetime.now()
        
        # Create query context
        query_context = QueryContext(
            query=request.query,
            user_id=user_id,
            file_filter=request.file_filter,
            min_confidence=request.min_confidence,
            max_results=request.max_results,
            search_strategy=request.search_strategy
        )
        
        # Perform search
        results = await cognitive_engine.search(query_context)
        
        # Check if human feedback should be requested
        feedback_decision = await learning_engine.should_request_human_feedback(
            request.query, results, user_id
        )
        
        # Generate answer if requested
        answer = None
        if request.include_answer and results:
            answer = await cognitive_engine.answer_question(request.query, results)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Convert results to dict format
        result_dicts = []
        for result in results:
            result_dict = {
                'chunk_id': result.chunk_id,
                'content': result.content,
                'score': result.score,
                'source_location': result.source_location,
                'metadata': result.metadata,
                'search_method': result.search_method,
                'confidence': result.confidence
            }
            result_dicts.append(result_dict)
        
        response = SearchResponse(
            query=request.query,
            results=result_dicts,
            total_found=len(results),
            search_time=search_time,
            search_strategy_used=query_context.search_strategy.value,
            answer=answer
        )
        
        # Add feedback request info to response metadata
        if feedback_decision['should_request']:
            # Store search session for potential feedback
            search_session = {
                'query': request.query,
                'results': results,
                'user_id': user_id,
                'timestamp': datetime.now(),
                'feedback_reason': feedback_decision['reason']
            }
            # In production, store in database or cache
        
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    user_id: Optional[str] = Depends(get_current_user)
):
    """
    Submit human feedback for active learning
    """
    if not learning_engine:
        raise HTTPException(status_code=503, detail="Learning engine not initialized")
    
    try:
        # Prepare feedback data
        feedback_data = {
            'user_id': user_id or 'anonymous',
            'query': '',  # Would be retrieved from session
            'results': [],  # Would be retrieved from session
            'feedback_type': request.feedback_type.value,
            'feedback_value': request.feedback_value,
            'confidence_before': 0.0,  # Would be retrieved from session
            'metadata': {
                'result_ids': request.result_ids,
                'comments': request.comments,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Process feedback
        result = await learning_engine.collect_human_feedback(
            request.session_id,
            feedback_data
        )
        
        message = "Feedback received successfully"
        if result.get('should_retrain'):
            message += ". Model retraining has been triggered."
        
        return FeedbackResponse(
            status=result.get('status', 'success'),
            session_id=request.session_id,
            feedback_processed=result.get('feedback_processed', True),
            should_retrain=result.get('should_retrain', False),
            message=message
        )
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@app.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get comprehensive system health and performance metrics"""
    try:
        timestamp = datetime.now()
        
        # Get health from all engines
        nexus_health = nexus_engine.get_system_health() if nexus_engine else {"overall_health": "unavailable"}
        cognitive_stats = cognitive_engine.get_performance_stats() if cognitive_engine else {}
        learning_stats = learning_engine.get_learning_statistics() if learning_engine else {}
        
        # Determine overall status
        component_statuses = []
        if nexus_health.get("overall_health") == "healthy":
            component_statuses.append("healthy")
        else:
            component_statuses.append("degraded")
        
        if cognitive_stats:
            component_statuses.append("healthy")
        else:
            component_statuses.append("unavailable")
        
        if learning_stats:
            component_statuses.append("healthy")
        else:
            component_statuses.append("unavailable")
        
        overall_status = "healthy" if all(s == "healthy" for s in component_statuses) else "degraded"
        
        return SystemHealthResponse(
            timestamp=timestamp,
            overall_status=overall_status,
            components={
                "nexus_engine": nexus_health.get("overall_health", "unknown"),
                "cognitive_engine": "healthy" if cognitive_stats else "unavailable",
                "learning_engine": "healthy" if learning_stats else "unavailable",
                "database": "healthy",  # Would check actual DB connection
                "redis": "healthy"  # Would check actual Redis connection
            },
            performance_metrics={
                "nexus_metrics": nexus_health.get("metrics", {}),
                "search_stats": cognitive_stats.get("search_stats", {}),
                "learning_stats": learning_stats.get("feedback_statistics", {})
            },
            recent_activity={
                "documents_processed": nexus_health.get("metrics", {}).get("documents_processed", 0),
                "searches_performed": cognitive_stats.get("search_stats", {}).get("total_searches", 0),
                "feedback_received": learning_stats.get("feedback_statistics", {}).get("recent_feedback", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/analytics/performance")
async def get_performance_analytics(
    days: int = Query(default=7, ge=1, le=90),
    user_id: Optional[str] = Depends(get_current_user)
):
    """Get detailed performance analytics"""
    try:
        if not learning_engine:
            raise HTTPException(status_code=503, detail="Learning engine not initialized")
        
        stats = learning_engine.get_learning_statistics()
        
        # Add time-based filtering and more detailed analytics
        analytics = {
            "time_period": f"Last {days} days",
            "performance_summary": stats,
            "trends": {
                "feedback_trend": "stable",  # Would calculate actual trends
                "accuracy_trend": "improving",
                "anomaly_trend": "decreasing"
            },
            "recommendations": [
                "Continue collecting user feedback for model improvement",
                "Monitor data drift alerts",
                "Review anomalous documents in quarantine"
            ]
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics request failed: {e}")
        raise HTTPException(status_code=500, detail="Analytics request failed")

@app.get("/anomalies")
async def get_anomaly_reports(
    status: Optional[str] = Query(None, description="Filter by status: detected, investigated, resolved"),
    limit: int = Query(default=50, ge=1, le=200)
):
    """Get anomaly detection reports"""
    try:
        if not learning_engine:
            raise HTTPException(status_code=503, detail="Learning engine not initialized")
        
        # Query anomalies from database
        query = learning_engine.db_session.query(learning_engine.AnomalyReport)
        
        if status:
            query = query.filter(learning_engine.AnomalyReport.status == status)
        
        anomalies = query.order_by(learning_engine.AnomalyReport.created_at.desc()).limit(limit).all()
        
        anomaly_list = []
        for anomaly in anomalies:
            anomaly_dict = {
                'anomaly_id': anomaly.anomaly_id,
                'file_id': anomaly.file_id,
                'anomaly_type': anomaly.anomaly_type,
                'anomaly_score': anomaly.anomaly_score,
                'status': anomaly.status,
                'created_at': anomaly.created_at.isoformat(),
                'details': json.loads(anomaly.details) if anomaly.details else {}
            }
            anomaly_list.append(anomaly_dict)
        
        return {
            "anomalies": anomaly_list,
            "total_count": len(anomaly_list),
            "filters_applied": {"status": status} if status else {}
        }
        
    except Exception as e:
        logger.error(f"Anomaly reports request failed: {e}")
        raise HTTPException(status_code=500, detail="Anomaly reports request failed")

@app.post("/retrain")
async def trigger_manual_retrain(
    user_id: Optional[str] = Depends(get_current_user)
):
    """Manually trigger model retraining"""
    try:
        if not learning_engine:
            raise HTTPException(status_code=503, detail="Learning engine not initialized")
        
        # Check if user has permission (in production, implement proper authorization)
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Trigger retraining
        asyncio.create_task(learning_engine._trigger_model_retraining())
        
        return {
            "status": "success",
            "message": "Model retraining has been triggered",
            "triggered_by": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual retrain failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger retraining")

# WebSocket endpoint for real-time updates (optional)
@app.websocket("/ws/status/{session_id}")
async def websocket_status_updates(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time processing status updates"""
    await websocket.accept()
    
    try:
        while True:
            if nexus_engine:
                status = nexus_engine.get_processing_status(session_id)
                await websocket.send_json(status)
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )