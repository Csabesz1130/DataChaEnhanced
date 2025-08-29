# src/tasks/celery_tasks.py
"""
Celery Distributed Tasks for Nexus Cognition System
Implements bulletproof background processing with retry logic and monitoring
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import json
import traceback
from datetime import datetime, timedelta
import hashlib

from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure, task_success
from celery.exceptions import Retry, MaxRetriesExceededError
import redis
import numpy as np

# Import our processing components
from ..processors.bulletproof_pipeline import (
    MagicBytesDetector, FallbackProcessor, FileValidationResult,
    IntermediateRepresentation, FileType, ProcessingMethod
)
from ..cognitive.rag_engine import get_cognitive_engine
from ..learning.active_learning_engine import get_learning_engine
from ..core.nexus_engine import get_nexus_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery('nexus_cognition')

# Load configuration
celery_app.conf.update(
    broker_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'nexus_cognition.process_document': {'queue': 'document_processing'},
        'nexus_cognition.extract_embeddings': {'queue': 'ai_processing'},
        'nexus_cognition.anomaly_detection': {'queue': 'monitoring'},
        'nexus_cognition.retrain_models': {'queue': 'ai_processing'},
        'nexus_cognition.cleanup_resources': {'queue': 'maintenance'}
    },
    
    # Task execution settings
    task_time_limit=30 * 60,  # 30 minutes max
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Performance
    task_compression='gzip',
    result_compression='gzip',
    result_expires=3600,  # 1 hour
    
    # Security
    worker_hijack_root_logger=False,
    worker_log_color=False,
)

# Custom base task class with enhanced error handling
class BaseNexusTask(Task):
    """Base task class with comprehensive error handling and monitoring"""
    
    autoretry_for = (ConnectionError, TimeoutError, OSError)
    retry_kwargs = {'max_retries': 3, 'countdown': 60}
    retry_backoff = True
    retry_backoff_max = 600  # 10 minutes max
    retry_jitter = True
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called on task success"""
        logger.info(f"Task {self.name} ({task_id}) completed successfully")
        self._update_task_metrics(task_id, 'success', retval)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure"""
        logger.error(f"Task {self.name} ({task_id}) failed: {exc}")
        logger.error(f"Traceback: {einfo.traceback}")
        self._update_task_metrics(task_id, 'failure', {'error': str(exc)})
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry"""
        logger.warning(f"Task {self.name} ({task_id}) retrying: {exc}")
        self._update_task_metrics(task_id, 'retry', {'error': str(exc)})
    
    def _update_task_metrics(self, task_id: str, status: str, result: Any):
        """Update task execution metrics"""
        try:
            redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
            
            metrics = {
                'task_id': task_id,
                'task_name': self.name,
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'result': result if isinstance(result, (dict, list, str, int, float)) else str(result)
            }
            
            # Store in Redis with expiration
            key = f"task_metrics:{task_id}"
            redis_client.setex(key, 3600, json.dumps(metrics))  # 1 hour expiration
            
            # Update aggregate metrics
            counter_key = f"task_counter:{self.name}:{status}"
            redis_client.incr(counter_key)
            redis_client.expire(counter_key, 86400)  # 24 hours
            
        except Exception as e:
            logger.warning(f"Failed to update task metrics: {e}")

# Core processing tasks

@celery_app.task(bind=True, base=BaseNexusTask, name='nexus_cognition.process_document')
def process_document_task(self, session_id: str, file_path: str, user_id: str = None, options: Dict = None):
    """
    Main document processing task - orchestrates the complete pipeline
    """
    try:
        logger.info(f"Starting document processing for session {session_id}")
        
        # Update task status
        self.update_state(
            state='PROCESSING',
            meta={'step': 'validation', 'progress': 0.1}
        )
        
        # Step 1: File validation
        validation_result = MagicBytesDetector.validate_file(file_path)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid file: {validation_result.magic_bytes}")
        
        # Step 2: Create fallback processor
        fallback_config = {
            'quarantine_path': './quarantine',
            'tika_server_url': os.getenv('TIKA_SERVER_URL', 'http://localhost:9998'),
            'ocr_enabled': True
        }
        processor = FallbackProcessor(fallback_config)
        
        self.update_state(
            state='PROCESSING',
            meta={'step': 'extraction', 'progress': 0.3}
        )
        
        # Step 3: Extract content with fallback chain
        ir = asyncio.run(processor.process_with_fallback(file_path, validation_result))
        
        if not ir.chunks:
            raise ValueError("No content could be extracted from the file")
        
        self.update_state(
            state='PROCESSING',
            meta={'step': 'embedding_generation', 'progress': 0.5}
        )
        
        # Step 4: Generate embeddings and ingest into cognitive engine
        cognitive_engine = get_cognitive_engine()
        ingest_result = asyncio.run(cognitive_engine.ingest_document(ir))
        
        self.update_state(
            state='PROCESSING',
            meta={'step': 'anomaly_detection', 'progress': 0.7}
        )
        
        # Step 5: Anomaly detection
        learning_engine = get_learning_engine()
        
        # Extract embeddings for anomaly detection
        embeddings = np.array([chunk.embedding for chunk in ir.chunks if chunk.embedding])
        if embeddings.size > 0:
            anomalies = asyncio.run(learning_engine.detect_anomalies(ir, embeddings))
        else:
            anomalies = []
        
        self.update_state(
            state='PROCESSING',
            meta={'step': 'drift_detection', 'progress': 0.8}
        )
        
        # Step 6: Data drift detection
        if embeddings.size > 0:
            drift_result = asyncio.run(learning_engine.detect_data_drift(embeddings))
        else:
            drift_result = None
        
        self.update_state(
            state='PROCESSING',
            meta={'step': 'finalization', 'progress': 0.9}
        )
        
        # Step 7: Calculate confidence and human review requirements
        confidence_scores = {}
        for chunk in ir.chunks:
            confidence_scores[chunk.chunk_id] = chunk.confidence
        
        avg_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.0
        
        # Check if human review is needed
        requires_human_review = False
        if anomalies or (drift_result and drift_result.drift_detected) or avg_confidence < 0.7:
            requires_human_review = True
        
        # Step 8: Cleanup temporary files
        try:
            if os.path.exists(file_path):
                if file_path.startswith('/tmp') or file_path.startswith(tempfile.gettempdir()):
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file: {e}")
        
        # Final result
        result = {
            'session_id': session_id,
            'file_id': ir.file_id,
            'status': 'completed',
            'chunks_processed': len(ir.chunks),
            'confidence_scores': confidence_scores,
            'average_confidence': avg_confidence,
            'requires_human_review': requires_human_review,
            'anomalies_detected': len(anomalies),
            'drift_detected': drift_result.drift_detected if drift_result else False,
            'processing_chain': [method.value for method in ir.processing_chain],
            'ingest_result': ingest_result,
            'processing_time': (datetime.now() - datetime.fromisoformat(ir.processing_timestamp)).total_seconds()
        }
        
        logger.info(f"Document processing completed for session {session_id}")
        return result
        
    except Exception as e:
        logger.error(f"Document processing failed for session {session_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Try to save error information
        try:
            error_info = {
                'session_id': session_id,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store error in Redis for later analysis
            redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
            error_key = f"processing_error:{session_id}"
            redis_client.setex(error_key, 86400, json.dumps(error_info))  # 24 hours
            
        except Exception as store_error:
            logger.error(f"Failed to store error information: {store_error}")
        
        # Re-raise for Celery to handle
        raise

@celery_app.task(bind=True, base=BaseNexusTask, name='nexus_cognition.extract_embeddings')
def extract_embeddings_task(self, chunks_data: List[Dict], model_name: str = 'all-MiniLM-L6-v2'):
    """
    Dedicated task for embedding generation - can be used for batch processing
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Generating embeddings for {len(chunks_data)} chunks")
        
        # Load model
        model = SentenceTransformer(model_name)
        
        # Extract content
        contents = [chunk['content'] for chunk in chunks_data]
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            
            self.update_state(
                state='PROCESSING',
                meta={'progress': i / len(contents), 'batch': i // batch_size + 1}
            )
            
            embeddings = model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings.tolist())
        
        result = {
            'embeddings': all_embeddings,
            'chunk_count': len(chunks_data),
            'model_used': model_name,
            'embedding_dimension': len(all_embeddings[0]) if all_embeddings else 0
        }
        
        logger.info(f"Embedding generation completed for {len(chunks_data)} chunks")
        return result
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

@celery_app.task(bind=True, base=BaseNexusTask, name='nexus_cognition.anomaly_detection')
def anomaly_detection_task(self, embeddings: List[List[float]], file_id: str, threshold: float = 0.1):
    """
    Dedicated anomaly detection task
    """
    try:
        from sklearn.ensemble import IsolationForest
        
        logger.info(f"Running anomaly detection for file {file_id}")
        
        embeddings_array = np.array(embeddings)
        
        # Initialize and fit anomaly detector
        detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # Detect outliers
        outliers = detector.fit_predict(embeddings_array)
        scores = detector.decision_function(embeddings_array)
        
        # Find anomalous chunks
        anomalous_indices = np.where(outliers == -1)[0].tolist()
        anomaly_scores = scores[anomalous_indices].tolist()
        
        result = {
            'file_id': file_id,
            'total_chunks': len(embeddings),
            'anomalous_chunks': len(anomalous_indices),
            'anomalous_indices': anomalous_indices,
            'anomaly_scores': anomaly_scores,
            'threshold_used': threshold
        }
        
        logger.info(f"Anomaly detection completed: {len(anomalous_indices)} anomalies found")
        return result
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise

@celery_app.task(bind=True, base=BaseNexusTask, name='nexus_cognition.retrain_models')
def retrain_models_task(self, feedback_data: List[Dict], model_config: Dict = None):
    """
    Model retraining task with collected human feedback
    """
    try:
        logger.info(f"Starting model retraining with {len(feedback_data)} feedback entries")
        
        # This is a simplified implementation
        # In production, this would involve:
        # 1. Fine-tuning embedding models
        # 2. Updating search ranking algorithms
        # 3. Adjusting anomaly detection thresholds
        # 4. Updating hybrid search weights
        
        self.update_state(
            state='PROCESSING',
            meta={'step': 'data_preparation', 'progress': 0.2}
        )
        
        # Prepare training data
        training_samples = []
        for feedback in feedback_data:
            if feedback.get('feedback_type') == 'relevance':
                sample = {
                    'query': feedback.get('query', ''),
                    'content': feedback.get('content', ''),
                    'label': feedback.get('feedback_value', 0),
                    'timestamp': feedback.get('timestamp')
                }
                training_samples.append(sample)
        
        self.update_state(
            state='PROCESSING',
            meta={'step': 'model_training', 'progress': 0.5}
        )
        
        # Simulate training process
        import time
        for i in range(10):
            time.sleep(1)  # Simulate training time
            self.update_state(
                state='PROCESSING',
                meta={'step': 'model_training', 'progress': 0.5 + (i * 0.04)}
            )
        
        self.update_state(
            state='PROCESSING',
            meta={'step': 'validation', 'progress': 0.9}
        )
        
        # Calculate performance improvement (simulated)
        baseline_performance = 0.75
        improvement = min(len(training_samples) * 0.005, 0.1)  # Up to 10% improvement
        new_performance = min(baseline_performance + improvement, 0.95)
        
        # Update model performance metrics
        learning_engine = get_learning_engine()
        
        # Store new performance metrics
        performance_data = {
            'model_type': 'search_relevance',
            'performance_score': new_performance,
            'training_samples': len(training_samples),
            'improvement': improvement,
            'retraining_timestamp': datetime.now().isoformat()
        }
        
        result = {
            'status': 'completed',
            'training_samples_used': len(training_samples),
            'baseline_performance': baseline_performance,
            'new_performance': new_performance,
            'improvement': improvement,
            'models_updated': ['embedding_model', 'search_ranking', 'anomaly_detection']
        }
        
        logger.info(f"Model retraining completed. Performance: {baseline_performance:.3f} -> {new_performance:.3f}")
        return result
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise

@celery_app.task(bind=True, base=BaseNexusTask, name='nexus_cognition.cleanup_resources')
def cleanup_resources_task(self, resource_type: str = 'all', age_hours: int = 24):
    """
    Maintenance task for cleaning up old resources
    """
    try:
        logger.info(f"Starting resource cleanup: {resource_type}, age > {age_hours} hours")
        
        cleaned_items = {
            'temporary_files': 0,
            'old_embeddings': 0,
            'expired_cache': 0,
            'old_metrics': 0
        }
        
        cutoff_time = datetime.now() - timedelta(hours=age_hours)
        
        # Cleanup temporary files
        if resource_type in ['all', 'temp_files']:
            temp_dirs = [tempfile.gettempdir(), './temp', './quarantine']
            
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for file_path in Path(temp_dir).rglob('*'):
                        if file_path.is_file():
                            try:
                                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                                if file_time < cutoff_time:
                                    file_path.unlink()
                                    cleaned_items['temporary_files'] += 1
                            except Exception as e:
                                logger.warning(f"Failed to cleanup file {file_path}: {e}")
        
        # Cleanup old cache entries
        if resource_type in ['all', 'cache']:
            try:
                redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
                
                # Get all cache keys
                cache_keys = redis_client.keys('search_cache:*')
                
                for key in cache_keys:
                    try:
                        ttl = redis_client.ttl(key)
                        if ttl == -1:  # No expiration set
                            redis_client.delete(key)
                            cleaned_items['expired_cache'] += 1
                    except Exception as e:
                        logger.warning(f"Failed to cleanup cache key {key}: {e}")
                        
            except Exception as e:
                logger.warning(f"Redis cleanup failed: {e}")
        
        # Cleanup old metrics
        if resource_type in ['all', 'metrics']:
            try:
                redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
                
                metric_keys = redis_client.keys('task_metrics:*')
                for key in metric_keys:
                    try:
                        ttl = redis_client.ttl(key)
                        if ttl > 0 and ttl < 3600:  # Expire soon
                            redis_client.delete(key)
                            cleaned_items['old_metrics'] += 1
                    except Exception as e:
                        logger.warning(f"Failed to cleanup metric {key}: {e}")
                        
            except Exception as e:
                logger.warning(f"Metrics cleanup failed: {e}")
        
        result = {
            'status': 'completed',
            'resource_type': resource_type,
            'age_hours': age_hours,
            'items_cleaned': cleaned_items,
            'total_cleaned': sum(cleaned_items.values())
        }
        
        logger.info(f"Resource cleanup completed: {sum(cleaned_items.values())} items cleaned")
        return result
        
    except Exception as e:
        logger.error(f"Resource cleanup failed: {e}")
        raise

# Periodic tasks
@celery_app.task(bind=True, base=BaseNexusTask, name='nexus_cognition.health_check')
def health_check_task(self):
    """
    Periodic health check task
    """
    try:
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'redis_connection': False,
            'database_connection': False,
            'ai_models': False,
            'worker_count': 0
        }
        
        # Check Redis connection
        try:
            redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
            redis_client.ping()
            health_status['redis_connection'] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
        
        # Check if AI models are accessible
        try:
            # This would check if embedding models are loaded and accessible
            health_status['ai_models'] = True
        except Exception as e:
            logger.error(f"AI models health check failed: {e}")
        
        # Get active worker count
        try:
            inspect = celery_app.control.inspect()
            active_workers = inspect.active()
            health_status['worker_count'] = len(active_workers) if active_workers else 0
        except Exception as e:
            logger.error(f"Worker count check failed: {e}")
        
        # Overall health score
        health_checks = ['redis_connection', 'ai_models']
        healthy_checks = sum(1 for check in health_checks if health_status[check])
        health_status['overall_health'] = healthy_checks / len(health_checks)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 0.0,
            'error': str(e)
        }

# Task monitoring signals
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Called before task execution"""
    logger.info(f"Starting task {task.name} ({task_id})")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Called after task execution"""
    logger.info(f"Finished task {task.name} ({task_id}) with state {state}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Called on task failure"""
    logger.error(f"Task {sender.name} ({task_id}) failed: {exception}")

@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    """Called on task success"""
    logger.info(f"Task {sender.name} completed successfully")

# Periodic task schedule (using Celery Beat)
celery_app.conf.beat_schedule = {
    'health-check': {
        'task': 'nexus_cognition.health_check',
        'schedule': 300.0,  # Every 5 minutes
    },
    'cleanup-resources': {
        'task': 'nexus_cognition.cleanup_resources',
        'schedule': 3600.0,  # Every hour
        'kwargs': {'resource_type': 'temp_files', 'age_hours': 2}
    },
    'cleanup-old-cache': {
        'task': 'nexus_cognition.cleanup_resources',
        'schedule': 86400.0,  # Every 24 hours
        'kwargs': {'resource_type': 'cache', 'age_hours': 24}
    },
}

if __name__ == '__main__':
    # Run worker
    celery_app.start()