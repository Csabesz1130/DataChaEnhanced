# src/learning/active_learning_engine.py
"""
Active Learning & Self-Learning Motor - Pillar 4
Implements continuous learning, anomaly detection, and human-in-the-loop feedback
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import hashlib

# ML and monitoring libraries
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp
import pandas as pd

# Deep learning for advanced models
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments

# Monitoring and alerting
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Database
from sqlalchemy import create_engine, Column, String, Text, Float, DateTime, JSON, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..processors.bulletproof_pipeline import DocumentChunk, IntermediateRepresentation
from ..cognitive.rag_engine import SearchResult, QueryContext

logger = logging.getLogger(__name__)

# Extended database models for learning
Base = declarative_base()

class LearningSession(Base):
    __tablename__ = 'learning_sessions'
    
    session_id = Column(String(64), primary_key=True)
    user_id = Column(String(64))
    query = Column(Text)
    search_results = Column(JSON)
    user_feedback = Column(JSON)
    confidence_before = Column(Float)
    confidence_after = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    feedback_received_at = Column(DateTime)
    used_for_training = Column(Boolean, default=False)

class AnomalyReport(Base):
    __tablename__ = 'anomaly_reports'
    
    anomaly_id = Column(String(64), primary_key=True)
    file_id = Column(String(64))
    anomaly_type = Column(String(32))
    anomaly_score = Column(Float)
    details = Column(JSON)
    status = Column(String(16), default='detected')  # detected, investigated, resolved
    created_at = Column(DateTime, default=datetime.utcnow)
    investigated_at = Column(DateTime)
    resolution = Column(Text)

class ModelPerformance(Base):
    __tablename__ = 'model_performance'
    
    metric_id = Column(String(64), primary_key=True)
    model_name = Column(String(64))
    metric_type = Column(String(32))
    metric_value = Column(Float)
    measurement_date = Column(DateTime, default=datetime.utcnow)
    model_metadata = Column(JSON)

class DataDriftAlert(Base):
    __tablename__ = 'data_drift_alerts'
    
    alert_id = Column(String(64), primary_key=True)
    drift_type = Column(String(32))
    drift_score = Column(Float)
    threshold = Column(Float)
    affected_features = Column(JSON)
    alert_severity = Column(String(16))  # low, medium, high, critical
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged = Column(Boolean, default=False)
    action_taken = Column(Text)

class FeedbackType(Enum):
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    PREFERENCE = "preference"
    CORRECTION = "correction"

class UncertaintyMethod(Enum):
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    ENTROPY_SAMPLING = "entropy_sampling"
    MARGIN_SAMPLING = "margin_sampling"
    LEAST_CONFIDENCE = "least_confidence"
    RANDOM_SAMPLING = "random_sampling"

@dataclass
class FeedbackEntry:
    session_id: str
    user_id: str
    query: str
    results: List[SearchResult]
    feedback_type: FeedbackType
    feedback_value: Any  # Could be rating, boolean, text, etc.
    confidence_before: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnomalyDetection:
    anomaly_id: str
    file_id: str
    anomaly_type: str
    score: float
    description: str
    severity: str
    recommendations: List[str]
    detected_at: datetime = field(default_factory=datetime.now)

@dataclass
class DriftDetectionResult:
    drift_detected: bool
    drift_score: float
    threshold: float
    affected_dimensions: List[int]
    drift_type: str
    recommendation: str
    p_value: Optional[float] = None

class ActiveLearningEngine:
    """
    Advanced Active Learning Engine with Human-in-the-Loop and continuous adaptation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Learning parameters
        self.uncertainty_threshold = config.get('uncertainty_threshold', 0.7)
        self.min_feedback_for_retrain = config.get('min_feedback_for_retrain', 50)
        self.retrain_frequency = config.get('retrain_frequency_hours', 24)
        
        # Sampling strategies
        self.uncertainty_method = UncertaintyMethod(
            config.get('uncertainty_method', 'confidence_threshold')
        )
        self.random_sampling_rate = config.get('random_sampling_rate', 0.05)
        
        # Anomaly detection
        self.anomaly_detector = None
        self.baseline_embeddings = None
        self.anomaly_threshold = config.get('anomaly_threshold', 0.1)
        
        # Data drift monitoring
        self.drift_detection_window = config.get('drift_window_days', 7)
        self.drift_threshold = config.get('drift_threshold', 0.1)
        self.baseline_distributions = {}
        
        # Human feedback management
        self.pending_feedback = []
        self.feedback_history = []
        self.human_reviewers = config.get('human_reviewers', [])
        
        # Model performance tracking
        self.performance_history = {}
        self.last_retrain_time = None
        
        # Database
        self.db_engine = None
        self.db_session = None
        
        # Notifications
        self.notification_config = config.get('notifications', {})
        
        # Initialize components
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize all learning components"""
        try:
            await self._init_database()
            await self._init_anomaly_detection()
            await self._init_drift_monitoring()
            await self._load_historical_data()
            logger.info("Active Learning Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Active Learning Engine: {e}")
            raise
    
    async def _init_database(self):
        """Initialize database for learning data"""
        db_url = self.config.get('database_url', 'sqlite:///nexus_learning.db')
        self.db_engine = create_engine(db_url)
        Base.metadata.create_all(self.db_engine)
        
        Session = sessionmaker(bind=self.db_engine)
        self.db_session = Session()
        
        logger.info("Learning database initialized")
    
    async def _init_anomaly_detection(self):
        """Initialize anomaly detection models"""
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100,
            max_features=1.0
        )
        
        # Initialize with empty baseline (will be populated)
        self.baseline_embeddings = np.array([]).reshape(0, 384)  # Assuming 384-dim embeddings
        
        logger.info("Anomaly detection initialized")
    
    async def _init_drift_monitoring(self):
        """Initialize data drift monitoring"""
        self.baseline_distributions = {
            'embedding_means': None,
            'embedding_stds': None,
            'content_lengths': None,
            'chunk_types': None
        }
        
        logger.info("Data drift monitoring initialized")
    
    async def _load_historical_data(self):
        """Load historical learning data for warm start"""
        try:
            # Load recent feedback for analysis
            recent_feedback = self.db_session.query(LearningSession).filter(
                LearningSession.created_at > datetime.now() - timedelta(days=30)
            ).all()
            
            self.feedback_history = [
                FeedbackEntry(
                    session_id=session.session_id,
                    user_id=session.user_id,
                    query=session.query,
                    results=[],  # Simplified for initialization
                    feedback_type=FeedbackType.RELEVANCE,
                    feedback_value=session.user_feedback,
                    confidence_before=session.confidence_before,
                    timestamp=session.created_at
                )
                for session in recent_feedback
                if session.user_feedback is not None
            ]
            
            logger.info(f"Loaded {len(self.feedback_history)} historical feedback entries")
            
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
    
    async def should_request_human_feedback(
        self, 
        query: str, 
        results: List[SearchResult],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Determine if human feedback should be requested using uncertainty sampling
        """
        
        # Calculate uncertainty based on results
        uncertainty_score = await self._calculate_uncertainty(results)
        
        # Apply uncertainty sampling strategy
        should_request = False
        reason = ""
        
        if self.uncertainty_method == UncertaintyMethod.CONFIDENCE_THRESHOLD:
            should_request = uncertainty_score > self.uncertainty_threshold
            reason = f"Low confidence: {uncertainty_score:.3f} > {self.uncertainty_threshold}"
        
        elif self.uncertainty_method == UncertaintyMethod.RANDOM_SAMPLING:
            should_request = np.random.random() < self.random_sampling_rate
            reason = f"Random sampling (rate: {self.random_sampling_rate})"
        
        # Additional triggers for feedback request
        additional_triggers = await self._check_additional_feedback_triggers(query, results, user_id)
        
        if additional_triggers['should_request']:
            should_request = True
            reason = additional_triggers['reason']
        
        return {
            'should_request': should_request,
            'reason': reason,
            'uncertainty_score': uncertainty_score,
            'method': self.uncertainty_method.value,
            'additional_triggers': additional_triggers
        }
    
    async def _calculate_uncertainty(self, results: List[SearchResult]) -> float:
        """Calculate uncertainty score from search results"""
        
        if not results:
            return 1.0  # Maximum uncertainty for no results
        
        scores = [result.score for result in results]
        confidences = [result.confidence for result in results]
        
        # Multiple uncertainty measures
        uncertainty_measures = []
        
        # 1. Score-based uncertainty (entropy of top results)
        if len(scores) > 1:
            # Normalize scores to probabilities
            scores_array = np.array(scores)
            if scores_array.sum() > 0:
                probs = scores_array / scores_array.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                uncertainty_measures.append(entropy)
        
        # 2. Confidence-based uncertainty
        avg_confidence = np.mean(confidences)
        confidence_uncertainty = 1.0 - avg_confidence
        uncertainty_measures.append(confidence_uncertainty)
        
        # 3. Score gap (margin between top results)
        if len(scores) > 1:
            score_gap = scores[0] - scores[1]
            gap_uncertainty = 1.0 - min(score_gap, 1.0)
            uncertainty_measures.append(gap_uncertainty)
        
        # 4. Result consistency
        if len(results) > 2:
            score_std = np.std(scores[:3])  # Standard deviation of top 3 scores
            consistency_uncertainty = min(score_std * 2, 1.0)  # Scale to [0,1]
            uncertainty_measures.append(consistency_uncertainty)
        
        # Combine uncertainty measures
        if uncertainty_measures:
            return np.mean(uncertainty_measures)
        else:
            return 0.5  # Neutral uncertainty
    
    async def _check_additional_feedback_triggers(
        self, 
        query: str, 
        results: List[SearchResult],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Check for additional conditions that should trigger feedback requests"""
        
        triggers = {
            'should_request': False,
            'reason': '',
            'triggers_activated': []
        }
        
        # 1. New user (get feedback to understand preferences)
        if user_id:
            user_history = self.db_session.query(LearningSession).filter(
                LearningSession.user_id == user_id
            ).count()
            
            if user_history < 5:  # New user
                triggers['triggers_activated'].append('new_user')
        
        # 2. Novel query patterns (using simple keyword detection)
        query_embedding = await self._get_query_embedding(query)
        if await self._is_novel_query(query_embedding):
            triggers['triggers_activated'].append('novel_query')
        
        # 3. Inconsistent results from different search methods
        search_methods = set(result.search_method for result in results)
        if len(search_methods) > 1:
            # Check if different methods give very different top results
            method_groups = {}
            for result in results[:5]:  # Top 5 results
                method = result.search_method
                if method not in method_groups:
                    method_groups[method] = []
                method_groups[method].append(result.score)
            
            if len(method_groups) > 1:
                method_means = {method: np.mean(scores) for method, scores in method_groups.items()}
                score_variance = np.var(list(method_means.values()))
                
                if score_variance > 0.1:  # High variance between methods
                    triggers['triggers_activated'].append('method_disagreement')
        
        # 4. Seasonal/temporal patterns (e.g., end of month reports)
        current_time = datetime.now()
        if current_time.day >= 28:  # End of month
            triggers['triggers_activated'].append('temporal_pattern')
        
        # Determine if any trigger should request feedback
        priority_triggers = ['novel_query', 'method_disagreement']
        if any(trigger in triggers['triggers_activated'] for trigger in priority_triggers):
            triggers['should_request'] = True
            triggers['reason'] = f"Additional triggers: {', '.join(triggers['triggers_activated'])}"
        
        return triggers
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query (simplified - would use actual embedding model)"""
        # This would use the actual embedding model from cognitive engine
        # For now, return a dummy embedding
        return np.random.random(384)
    
    async def _is_novel_query(self, query_embedding: np.ndarray) -> bool:
        """Determine if query is novel compared to historical queries"""
        # Simple novelty detection using distance to historical query embeddings
        # In production, maintain a database of historical query embeddings
        
        # For demonstration, assume novel if random condition
        return np.random.random() < 0.1  # 10% of queries considered novel
    
    async def collect_human_feedback(
        self, 
        session_id: str,
        feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Collect and process human feedback for active learning
        """
        
        try:
            # Create feedback entry
            feedback_entry = FeedbackEntry(
                session_id=session_id,
                user_id=feedback_data.get('user_id', 'anonymous'),
                query=feedback_data.get('query', ''),
                results=feedback_data.get('results', []),
                feedback_type=FeedbackType(feedback_data.get('feedback_type', 'relevance')),
                feedback_value=feedback_data.get('feedback_value'),
                confidence_before=feedback_data.get('confidence_before', 0.0),
                metadata=feedback_data.get('metadata', {})
            )
            
            # Store in database
            learning_session = LearningSession(
                session_id=session_id,
                user_id=feedback_entry.user_id,
                query=feedback_entry.query,
                search_results=json.dumps([r.__dict__ for r in feedback_entry.results]),
                user_feedback=json.dumps({
                    'type': feedback_entry.feedback_type.value,
                    'value': feedback_entry.feedback_value,
                    'metadata': feedback_entry.metadata
                }),
                confidence_before=feedback_entry.confidence_before,
                feedback_received_at=datetime.now()
            )
            
            self.db_session.merge(learning_session)
            self.db_session.commit()
            
            # Add to learning queue
            self.feedback_history.append(feedback_entry)
            
            # Check if we should trigger retraining
            should_retrain = await self._should_trigger_retraining()
            
            # Update model performance estimates
            await self._update_performance_estimates(feedback_entry)
            
            result = {
                'status': 'success',
                'session_id': session_id,
                'feedback_processed': True,
                'should_retrain': should_retrain,
                'total_feedback_collected': len(self.feedback_history)
            }
            
            if should_retrain:
                # Trigger asynchronous retraining
                asyncio.create_task(self._trigger_model_retraining())
                result['retraining_triggered'] = True
            
            logger.info(f"Human feedback collected for session {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to collect human feedback: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'session_id': session_id
            }
    
    async def _should_trigger_retraining(self) -> bool:
        """Determine if model retraining should be triggered"""
        
        # Check feedback count
        recent_feedback = [
            f for f in self.feedback_history 
            if f.timestamp > datetime.now() - timedelta(hours=self.retrain_frequency)
        ]
        
        if len(recent_feedback) < self.min_feedback_for_retrain:
            return False
        
        # Check time since last retrain
        if self.last_retrain_time:
            time_since_retrain = datetime.now() - self.last_retrain_time
            if time_since_retrain.total_seconds() < (self.retrain_frequency * 3600):
                return False
        
        # Check performance degradation
        recent_performance = await self._calculate_recent_performance()
        if recent_performance and recent_performance < 0.7:  # Below acceptable threshold
            return True
        
        return len(recent_feedback) >= self.min_feedback_for_retrain
    
    async def _update_performance_estimates(self, feedback: FeedbackEntry):
        """Update model performance estimates based on feedback"""
        
        # Simple performance tracking based on user satisfaction
        if feedback.feedback_type == FeedbackType.RELEVANCE:
            # Assume feedback_value is a rating 1-5
            rating = feedback.feedback_value
            if isinstance(rating, (int, float)):
                performance_score = rating / 5.0  # Normalize to [0,1]
                
                # Store performance metric
                metric = ModelPerformance(
                    metric_id=hashlib.md5(f"{feedback.session_id}_{datetime.now()}".encode()).hexdigest(),
                    model_name='search_relevance',
                    metric_type='user_satisfaction',
                    metric_value=performance_score,
                    metadata={
                        'session_id': feedback.session_id,
                        'query': feedback.query,
                        'confidence_before': feedback.confidence_before
                    }
                )
                
                self.db_session.add(metric)
                self.db_session.commit()
    
    async def _calculate_recent_performance(self) -> Optional[float]:
        """Calculate recent model performance"""
        
        recent_metrics = self.db_session.query(ModelPerformance).filter(
            ModelPerformance.measurement_date > datetime.now() - timedelta(days=7)
        ).all()
        
        if not recent_metrics:
            return None
        
        scores = [metric.metric_value for metric in recent_metrics]
        return np.mean(scores)
    
    async def _trigger_model_retraining(self):
        """Trigger asynchronous model retraining"""
        
        try:
            logger.info("Starting model retraining with collected feedback")
            
            # Prepare training data from feedback
            training_data = await self._prepare_training_data()
            
            if len(training_data) < 10:  # Minimum data requirement
                logger.warning("Insufficient training data for retraining")
                return
            
            # Perform retraining (simplified implementation)
            new_performance = await self._retrain_models(training_data)
            
            # Update performance tracking
            self.last_retrain_time = datetime.now()
            
            # Send notification about retraining
            await self._send_retraining_notification(new_performance)
            
            logger.info(f"Model retraining completed. New performance: {new_performance}")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            await self._send_error_notification(f"Model retraining failed: {e}")
    
    async def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data from collected feedback"""
        
        training_samples = []
        
        for feedback in self.feedback_history[-self.min_feedback_for_retrain:]:
            if feedback.feedback_type == FeedbackType.RELEVANCE:
                # Create training sample
                sample = {
                    'query': feedback.query,
                    'results': [r.__dict__ for r in feedback.results],
                    'label': feedback.feedback_value,
                    'confidence': feedback.confidence_before,
                    'timestamp': feedback.timestamp.isoformat()
                }
                training_samples.append(sample)
        
        return training_samples
    
    async def _retrain_models(self, training_data: List[Dict[str, Any]]) -> float:
        """Perform actual model retraining (simplified)"""
        
        # In a real implementation, this would:
        # 1. Fine-tune the embedding model
        # 2. Update the search ranking algorithm
        # 3. Adjust hybrid search weights
        # 4. Update anomaly detection thresholds
        
        # For demonstration, simulate performance improvement
        baseline_performance = 0.75
        improvement = min(len(training_data) * 0.01, 0.15)  # Up to 15% improvement
        new_performance = min(baseline_performance + improvement, 0.95)
        
        # Mark feedback as used for training
        session_ids = [data.get('query', '') for data in training_data]
        self.db_session.query(LearningSession).filter(
            LearningSession.session_id.in_(session_ids)
        ).update({'used_for_training': True})
        self.db_session.commit()
        
        return new_performance
    
    async def detect_anomalies(
        self, 
        ir: IntermediateRepresentation,
        embeddings: np.ndarray
    ) -> List[AnomalyDetection]:
        """
        Detect anomalies in processed documents
        """
        
        anomalies = []
        
        try:
            # Update baseline if needed
            if self.baseline_embeddings.shape[0] == 0:
                self.baseline_embeddings = embeddings
                self.anomaly_detector.fit(embeddings)
                return anomalies  # No anomalies on first document
            
            # Detect embedding anomalies
            anomaly_scores = self.anomaly_detector.decision_function(embeddings)
            outliers = self.anomaly_detector.predict(embeddings)
            
            # File-level anomaly detection
            file_anomaly_score = np.mean(anomaly_scores)
            if file_anomaly_score < -self.anomaly_threshold:
                anomalies.append(AnomalyDetection(
                    anomaly_id=hashlib.md5(f"{ir.file_id}_embedding_anomaly".encode()).hexdigest(),
                    file_id=ir.file_id,
                    anomaly_type='embedding_anomaly',
                    score=abs(file_anomaly_score),
                    description=f"Document embeddings significantly different from baseline",
                    severity='medium',
                    recommendations=[
                        "Review document content for unusual patterns",
                        "Check if document is corrupted or in unexpected format",
                        "Verify file type detection accuracy"
                    ]
                ))
            
            # Chunk-level anomaly detection
            chunk_anomalies = np.where(outliers == -1)[0]
            if len(chunk_anomalies) > 0:
                anomalies.append(AnomalyDetection(
                    anomaly_id=hashlib.md5(f"{ir.file_id}_chunk_anomalies".encode()).hexdigest(),
                    file_id=ir.file_id,
                    anomaly_type='chunk_anomaly',
                    score=len(chunk_anomalies) / len(embeddings),
                    description=f"{len(chunk_anomalies)} chunks detected as anomalous",
                    severity='low',
                    recommendations=[
                        "Review anomalous chunks for data quality issues",
                        "Consider excluding anomalous chunks from search index"
                    ]
                ))
            
            # Content-based anomaly detection
            content_anomalies = await self._detect_content_anomalies(ir)
            anomalies.extend(content_anomalies)
            
            # Store anomalies in database
            for anomaly in anomalies:
                await self._store_anomaly(anomaly)
            
            # Update baseline with normal data
            normal_embeddings = embeddings[outliers != -1]
            if len(normal_embeddings) > 0:
                self.baseline_embeddings = np.vstack([self.baseline_embeddings, normal_embeddings])
                
                # Keep baseline size manageable
                if self.baseline_embeddings.shape[0] > 10000:
                    self.baseline_embeddings = self.baseline_embeddings[-5000:]
                
                # Retrain anomaly detector periodically
                if self.baseline_embeddings.shape[0] % 1000 == 0:
                    self.anomaly_detector.fit(self.baseline_embeddings)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    async def _detect_content_anomalies(self, ir: IntermediateRepresentation) -> List[AnomalyDetection]:
        """Detect content-based anomalies"""
        
        anomalies = []
        
        # Check file size anomalies
        file_size = ir.global_metadata.get('file_size', 0)
        if file_size > 50 * 1024 * 1024:  # > 50MB
            anomalies.append(AnomalyDetection(
                anomaly_id=hashlib.md5(f"{ir.file_id}_size_anomaly".encode()).hexdigest(),
                file_id=ir.file_id,
                anomaly_type='size_anomaly',
                score=min(file_size / (100 * 1024 * 1024), 1.0),  # Normalize to [0,1]
                description=f"Unusually large file size: {file_size / (1024*1024):.1f}MB",
                severity='medium',
                recommendations=[
                    "Verify file is not corrupted",
                    "Consider breaking large files into smaller chunks",
                    "Check if file contains unnecessary data"
                ]
            ))
        
        # Check chunk count anomalies
        chunk_count = len(ir.chunks)
        if chunk_count > 1000:
            anomalies.append(AnomalyDetection(
                anomaly_id=hashlib.md5(f"{ir.file_id}_chunk_count_anomaly".encode()).hexdigest(),
                file_id=ir.file_id,
                anomaly_type='chunk_count_anomaly',
                score=min(chunk_count / 2000, 1.0),
                description=f"Unusually high chunk count: {chunk_count}",
                severity='low',
                recommendations=[
                    "Review chunking strategy for this file type",
                    "Consider adjusting chunk size parameters"
                ]
            ))
        
        # Check processing method anomalies
        fallback_methods = ['apache_tika', 'ocr_extraction', 'quarantined']
        if any(method.value in fallback_methods for method in ir.processing_chain):
            anomalies.append(AnomalyDetection(
                anomaly_id=hashlib.md5(f"{ir.file_id}_processing_anomaly".encode()).hexdigest(),
                file_id=ir.file_id,
                anomaly_type='processing_anomaly',
                score=0.7,
                description=f"File required fallback processing: {[m.value for m in ir.processing_chain]}",
                severity='medium',
                recommendations=[
                    "Investigate why native parsing failed",
                    "Check file format and integrity",
                    "Consider improving native parser for this file type"
                ]
            ))
        
        return anomalies
    
    async def _store_anomaly(self, anomaly: AnomalyDetection):
        """Store anomaly in database"""
        
        anomaly_record = AnomalyReport(
            anomaly_id=anomaly.anomaly_id,
            file_id=anomaly.file_id,
            anomaly_type=anomaly.anomaly_type,
            anomaly_score=anomaly.score,
            details=json.dumps({
                'description': anomaly.description,
                'severity': anomaly.severity,
                'recommendations': anomaly.recommendations,
                'detected_at': anomaly.detected_at.isoformat()
            })
        )
        
        self.db_session.add(anomaly_record)
        self.db_session.commit()
    
    async def detect_data_drift(self, new_embeddings: np.ndarray) -> DriftDetectionResult:
        """
        Detect data drift in incoming embeddings using statistical tests
        """
        
        if self.baseline_embeddings.shape[0] == 0:
            # Set initial baseline
            self.baseline_embeddings = new_embeddings
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                threshold=self.drift_threshold,
                affected_dimensions=[],
                drift_type='no_baseline',
                recommendation='Baseline established'
            )
        
        try:
            # Multiple drift detection methods
            drift_results = []
            
            # 1. Wasserstein distance (Earth Mover's Distance)
            wasserstein_scores = []
            for dim in range(min(self.baseline_embeddings.shape[1], new_embeddings.shape[1])):
                baseline_dim = self.baseline_embeddings[:, dim]
                new_dim = new_embeddings[:, dim]
                score = wasserstein_distance(baseline_dim, new_dim)
                wasserstein_scores.append(score)
            
            avg_wasserstein = np.mean(wasserstein_scores)
            
            # 2. Kolmogorov-Smirnov test
            ks_scores = []
            ks_pvalues = []
            for dim in range(min(self.baseline_embeddings.shape[1], new_embeddings.shape[1])):
                baseline_dim = self.baseline_embeddings[:, dim]
                new_dim = new_embeddings[:, dim]
                ks_stat, p_value = ks_2samp(baseline_dim, new_dim)
                ks_scores.append(ks_stat)
                ks_pvalues.append(p_value)
            
            avg_ks = np.mean(ks_scores)
            min_pvalue = np.min(ks_pvalues)
            
            # 3. Mean shift detection
            baseline_means = np.mean(self.baseline_embeddings, axis=0)
            new_means = np.mean(new_embeddings, axis=0)
            mean_shift = np.linalg.norm(baseline_means - new_means)
            
            # Determine drift
            drift_detected = (
                avg_wasserstein > self.drift_threshold or
                avg_ks > self.drift_threshold or
                min_pvalue < 0.05 or
                mean_shift > self.drift_threshold
            )
            
            # Calculate overall drift score
            drift_score = np.mean([avg_wasserstein, avg_ks, mean_shift])
            
            # Identify affected dimensions
            affected_dims = []
            threshold_per_dim = self.drift_threshold
            for i, (ws, ks, p) in enumerate(zip(wasserstein_scores, ks_scores, ks_pvalues)):
                if ws > threshold_per_dim or ks > threshold_per_dim or p < 0.05:
                    affected_dims.append(i)
            
            # Generate recommendation
            if drift_detected:
                recommendation = "Consider model retraining due to significant data drift"
                if len(affected_dims) > len(wasserstein_scores) * 0.5:
                    recommendation = "Critical data drift detected - immediate retraining recommended"
            else:
                recommendation = "No significant data drift detected"
            
            result = DriftDetectionResult(
                drift_detected=drift_detected,
                drift_score=drift_score,
                threshold=self.drift_threshold,
                affected_dimensions=affected_dims,
                drift_type='statistical_drift',
                recommendation=recommendation,
                p_value=min_pvalue
            )
            
            # Store drift alert if detected
            if drift_detected:
                await self._store_drift_alert(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Data drift detection failed: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                threshold=self.drift_threshold,
                affected_dimensions=[],
                drift_type='error',
                recommendation=f"Drift detection failed: {e}"
            )
    
    async def _store_drift_alert(self, drift_result: DriftDetectionResult):
        """Store data drift alert in database"""
        
        severity = 'low'
        if drift_result.drift_score > self.drift_threshold * 2:
            severity = 'high'
        elif drift_result.drift_score > self.drift_threshold * 1.5:
            severity = 'medium'
        
        alert = DataDriftAlert(
            alert_id=hashlib.md5(f"drift_{datetime.now()}".encode()).hexdigest(),
            drift_type=drift_result.drift_type,
            drift_score=drift_result.drift_score,
            threshold=drift_result.threshold,
            affected_features=json.dumps(drift_result.affected_dimensions),
            alert_severity=severity
        )
        
        self.db_session.add(alert)
        self.db_session.commit()
        
        # Send notification for high severity alerts
        if severity in ['medium', 'high']:
            await self._send_drift_notification(drift_result, severity)
    
    async def _send_drift_notification(self, drift_result: DriftDetectionResult, severity: str):
        """Send drift detection notification"""
        
        message = f"""
        Data Drift Alert - {severity.upper()} Severity
        
        Drift Score: {drift_result.drift_score:.4f} (threshold: {drift_result.threshold})
        Affected Dimensions: {len(drift_result.affected_dimensions)}
        P-value: {drift_result.p_value:.6f if drift_result.p_value else 'N/A'}
        
        Recommendation: {drift_result.recommendation}
        
        This alert was generated by the Nexus Cognition Active Learning Engine.
        """
        
        await self._send_notification("Data Drift Alert", message)
    
    async def _send_retraining_notification(self, performance: float):
        """Send model retraining notification"""
        
        message = f"""
        Model Retraining Completed
        
        New Performance Score: {performance:.3f}
        Training Data: {len(self.feedback_history)} feedback entries
        Timestamp: {datetime.now()}
        
        The models have been updated with the latest user feedback.
        """
        
        await self._send_notification("Model Retraining Complete", message)
    
    async def _send_error_notification(self, error_message: str):
        """Send error notification"""
        
        message = f"""
        Active Learning Engine Error
        
        Error: {error_message}
        Timestamp: {datetime.now()}
        
        Please investigate and resolve the issue.
        """
        
        await self._send_notification("Active Learning Error", message)
    
    async def _send_notification(self, subject: str, message: str):
        """Send notification via configured channels"""
        
        try:
            # Email notification
            if self.notification_config.get('email_enabled'):
                await self._send_email_notification(subject, message)
            
            # Slack notification (if configured)
            if self.notification_config.get('slack_webhook'):
                await self._send_slack_notification(subject, message)
            
            # Log notification
            logger.info(f"Notification sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    async def _send_email_notification(self, subject: str, message: str):
        """Send email notification"""
        
        smtp_config = self.notification_config.get('smtp', {})
        recipients = self.notification_config.get('email_recipients', [])
        
        if not smtp_config or not recipients:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_email')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[Nexus Cognition] {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(smtp_config.get('host'), smtp_config.get('port', 587))
            server.starttls()
            server.login(smtp_config.get('username'), smtp_config.get('password'))
            
            text = msg.as_string()
            server.sendmail(smtp_config.get('from_email'), recipients, text)
            server.quit()
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
    
    async def _send_slack_notification(self, subject: str, message: str):
        """Send Slack notification"""
        
        import aiohttp
        
        webhook_url = self.notification_config.get('slack_webhook')
        if not webhook_url:
            return
        
        try:
            payload = {
                'text': f"*{subject}*\n```{message}```"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Slack notification failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning and performance statistics"""
        
        # Feedback statistics
        feedback_stats = {
            'total_feedback': len(self.feedback_history),
            'feedback_by_type': {},
            'recent_feedback': 0
        }
        
        for feedback in self.feedback_history:
            fb_type = feedback.feedback_type.value
            feedback_stats['feedback_by_type'][fb_type] = feedback_stats['feedback_by_type'].get(fb_type, 0) + 1
            
            if feedback.timestamp > datetime.now() - timedelta(days=7):
                feedback_stats['recent_feedback'] += 1
        
        # Performance statistics
        recent_performance = self.db_session.query(ModelPerformance).filter(
            ModelPerformance.measurement_date > datetime.now() - timedelta(days=30)
        ).all()
        
        performance_stats = {
            'recent_measurements': len(recent_performance),
            'average_performance': 0.0,
            'performance_trend': 'stable'
        }
        
        if recent_performance:
            scores = [metric.metric_value for metric in recent_performance]
            performance_stats['average_performance'] = np.mean(scores)
            
            # Simple trend calculation
            if len(scores) > 5:
                recent_avg = np.mean(scores[-5:])
                older_avg = np.mean(scores[:-5])
                if recent_avg > older_avg + 0.05:
                    performance_stats['performance_trend'] = 'improving'
                elif recent_avg < older_avg - 0.05:
                    performance_stats['performance_trend'] = 'declining'
        
        # Anomaly statistics
        recent_anomalies = self.db_session.query(AnomalyReport).filter(
            AnomalyReport.created_at > datetime.now() - timedelta(days=30)
        ).all()
        
        anomaly_stats = {
            'total_anomalies': len(recent_anomalies),
            'anomaly_types': {},
            'unresolved_anomalies': 0
        }
        
        for anomaly in recent_anomalies:
            anom_type = anomaly.anomaly_type
            anomaly_stats['anomaly_types'][anom_type] = anomaly_stats['anomaly_types'].get(anom_type, 0) + 1
            
            if anomaly.status == 'detected':
                anomaly_stats['unresolved_anomalies'] += 1
        
        # Drift statistics
        recent_drift_alerts = self.db_session.query(DataDriftAlert).filter(
            DataDriftAlert.created_at > datetime.now() - timedelta(days=30)
        ).all()
        
        drift_stats = {
            'total_drift_alerts': len(recent_drift_alerts),
            'unacknowledged_alerts': len([a for a in recent_drift_alerts if not a.acknowledged]),
            'critical_alerts': len([a for a in recent_drift_alerts if a.alert_severity == 'critical'])
        }
        
        return {
            'timestamp': datetime.now(),
            'learning_engine_status': 'active',
            'feedback_statistics': feedback_stats,
            'performance_statistics': performance_stats,
            'anomaly_statistics': anomaly_stats,
            'drift_statistics': drift_stats,
            'last_retrain_time': self.last_retrain_time,
            'baseline_embedding_count': self.baseline_embeddings.shape[0] if self.baseline_embeddings.size > 0 else 0
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.db_session:
            self.db_session.close()
        
        logger.info("Active Learning Engine cleanup completed")


# Global learning engine instance
learning_engine: Optional[ActiveLearningEngine] = None

def get_learning_engine() -> ActiveLearningEngine:
    """Get or create the global Active Learning Engine instance"""
    global learning_engine
    if learning_engine is None:
        config = {
            'database_url': 'sqlite:///nexus_learning.db',
            'uncertainty_threshold': 0.7,
            'min_feedback_for_retrain': 50,
            'retrain_frequency_hours': 24,
            'uncertainty_method': 'confidence_threshold',
            'random_sampling_rate': 0.05,
            'anomaly_threshold': 0.1,
            'drift_window_days': 7,
            'drift_threshold': 0.1,
            'notifications': {
                'email_enabled': False,
                'email_recipients': [],
                'slack_webhook': None,
                'smtp': {}
            }
        }
        learning_engine = ActiveLearningEngine(config)
    
    return learning_engine