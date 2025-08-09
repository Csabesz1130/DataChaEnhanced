#!/usr/bin/env python3
"""
Background Processing System for AI Excel Learning

This module provides background processing capabilities for learning from Excel files
and notifying users when the learning is complete. It includes task queuing, progress
tracking, and notification systems.
"""

import os
import sys
import time
import json
import logging
import threading
import queue
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .learning_pipeline import LearningPipeline
from .formula_learner import FormulaLearner, FormulaLogic, FilterCondition
from .excel_analyzer import ExcelAnalyzer
from .ml_models import ExcelMLModels
from .chart_learner import ChartLearner
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Status of background learning tasks"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class NotificationType(Enum):
    """Types of notifications"""
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    PROGRESS_UPDATE = "progress_update"
    SYSTEM_READY = "system_ready"

@dataclass
class LearningTask:
    """Represents a background learning task"""
    task_id: str
    file_path: str
    task_type: str  # 'excel_analysis', 'formula_learning', 'chart_learning', 'full_pipeline'
    priority: int = 1  # 1=low, 5=high
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Dict[str, Any] = None
    error: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Notification:
    """Represents a notification to the user"""
    notification_id: str
    task_id: str
    notification_type: NotificationType
    message: str
    timestamp: datetime
    data: Dict[str, Any] = None
    read: bool = False
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}

class BackgroundProcessor:
    """
    Background processor for learning from Excel files
    
    Features:
    - Queue-based task processing
    - Progress tracking and notifications
    - Multi-threaded processing
    - Task prioritization
    - Persistent storage of results
    - Real-time status updates
    """
    
    def __init__(self, 
                 max_workers: int = 2,
                 storage_path: str = "background_learning",
                 enable_notifications: bool = True):
        """
        Initialize the background processor
        
        Args:
            max_workers: Maximum number of concurrent workers
            storage_path: Path to store learning results and metadata
            enable_notifications: Whether to enable notification system
        """
        self.max_workers = max_workers
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.learning_pipeline = LearningPipeline()
        self.model_manager = ModelManager()
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, LearningTask] = {}
        self.completed_tasks: Dict[str, LearningTask] = {}
        self.task_counter = 0
        
        # Notification system
        self.enable_notifications = enable_notifications
        self.notifications: List[Notification] = []
        self.notification_handlers: List[Callable] = []
        
        # Processing control
        self.running = False
        self.workers: List[threading.Thread] = []
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_processing_time': 0,
            'average_processing_time': 0
        }
        
        # Load existing data
        self._load_persistent_data()
        
        logger.info(f"Background processor initialized with {max_workers} workers")
    
    def _load_persistent_data(self):
        """Load persistent data from storage"""
        try:
            # Load completed tasks
            tasks_file = self.storage_path / "completed_tasks.json"
            if tasks_file.exists():
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                    for task_data in tasks_data:
                        task = self._dict_to_task(task_data)
                        self.completed_tasks[task.task_id] = task
            
            # Load notifications
            notifications_file = self.storage_path / "notifications.json"
            if notifications_file.exists():
                with open(notifications_file, 'r') as f:
                    notifications_data = json.load(f)
                    for notif_data in notifications_data:
                        notification = self._dict_to_notification(notif_data)
                        self.notifications.append(notification)
            
            # Load statistics
            stats_file = self.storage_path / "statistics.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.stats.update(json.load(f))
                    
            logger.info(f"Loaded {len(self.completed_tasks)} completed tasks and {len(self.notifications)} notifications")
            
        except Exception as e:
            logger.warning(f"Error loading persistent data: {e}")
    
    def _save_persistent_data(self):
        """Save persistent data to storage"""
        try:
            # Save completed tasks
            tasks_data = [asdict(task) for task in self.completed_tasks.values()]
            with open(self.storage_path / "completed_tasks.json", 'w') as f:
                json.dump(tasks_data, f, default=str, indent=2)
            
            # Save notifications
            notifications_data = [asdict(notification) for notification in self.notifications]
            with open(self.storage_path / "notifications.json", 'w') as f:
                json.dump(notifications_data, f, default=str, indent=2)
            
            # Save statistics
            with open(self.storage_path / "statistics.json", 'w') as f:
                json.dump(self.stats, f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving persistent data: {e}")
    
    def add_notification_handler(self, handler: Callable[[Notification], None]):
        """Add a notification handler function"""
        self.notification_handlers.append(handler)
        logger.info(f"Added notification handler: {handler.__name__}")
    
    def _send_notification(self, notification: Notification):
        """Send a notification to all handlers"""
        if not self.enable_notifications:
            return
            
        self.notifications.append(notification)
        
        # Call all notification handlers
        for handler in self.notification_handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Error in notification handler {handler.__name__}: {e}")
        
        # Save notifications
        self._save_persistent_data()
    
    def submit_learning_task(self, 
                           file_path: str, 
                           task_type: str = "full_pipeline",
                           priority: int = 3,
                           metadata: Dict[str, Any] = None) -> str:
        """
        Submit a new learning task to the background processor
        
        Args:
            file_path: Path to the Excel file to learn from
            task_type: Type of learning task ('excel_analysis', 'formula_learning', 'chart_learning', 'full_pipeline')
            priority: Task priority (1=low, 5=high)
            metadata: Additional metadata for the task
            
        Returns:
            Task ID for tracking
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with self.lock:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{int(time.time())}"
            
            task = LearningTask(
                task_id=task_id,
                file_path=file_path,
                task_type=task_type,
                priority=priority,
                metadata=metadata or {}
            )
            
            # Add to queue with priority (lower number = higher priority)
            self.task_queue.put((5 - priority, task))
            self.active_tasks[task_id] = task
            
            # Send notification
            notification = Notification(
                notification_id=f"notif_{task_id}",
                task_id=task_id,
                notification_type=NotificationType.TASK_STARTED,
                message=f"Learning task submitted: {os.path.basename(file_path)}",
                timestamp=datetime.now(),
                data={'task_type': task_type, 'priority': priority}
            )
            self._send_notification(notification)
            
            self.stats['total_tasks'] += 1
            logger.info(f"Submitted learning task {task_id} for {file_path}")
            
            return task_id
    
    def start_processing(self):
        """Start the background processing workers"""
        if self.running:
            logger.warning("Background processor is already running")
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"BackgroundWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} background workers")
        
        # Send system ready notification
        notification = Notification(
            notification_id=f"system_ready_{int(time.time())}",
            task_id="",
            notification_type=NotificationType.SYSTEM_READY,
            message="Background learning system is ready and processing tasks",
            timestamp=datetime.now()
        )
        self._send_notification(notification)
    
    def stop_processing(self):
        """Stop the background processing workers"""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        self.workers.clear()
        logger.info("Background processing stopped")
    
    def _worker_loop(self):
        """Main worker loop for processing tasks"""
        while self.running:
            try:
                # Get next task from queue
                try:
                    priority, task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process the task
                self._process_task(task)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)
    
    def _process_task(self, task: LearningTask):
        """Process a single learning task"""
        logger.info(f"Processing task {task.task_id}: {task.task_type}")
        
        # Update task status
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now()
        
        # Send progress notification
        notification = Notification(
            notification_id=f"progress_{task.task_id}",
            task_id=task.task_id,
            notification_type=NotificationType.PROGRESS_UPDATE,
            message=f"Started processing: {os.path.basename(task.file_path)}",
            timestamp=datetime.now(),
            data={'progress': 0.0}
        )
        self._send_notification(notification)
        
        try:
            # Process based on task type
            if task.task_type == "excel_analysis":
                result = self._process_excel_analysis(task)
            elif task.task_type == "formula_learning":
                result = self._process_formula_learning(task)
            elif task.task_type == "chart_learning":
                result = self._process_chart_learning(task)
            elif task.task_type == "full_pipeline":
                result = self._process_full_pipeline(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Update task with results
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 100.0
            task.result = result
            
            # Calculate processing time
            processing_time = (task.completed_at - task.started_at).total_seconds()
            self.stats['total_processing_time'] += processing_time
            self.stats['completed_tasks'] += 1
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['completed_tasks']
            )
            
            # Move to completed tasks
            with self.lock:
                self.completed_tasks[task.task_id] = task
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
            
            # Send completion notification
            notification = Notification(
                notification_id=f"completed_{task.task_id}",
                task_id=task.task_id,
                notification_type=NotificationType.TASK_COMPLETED,
                message=f"Learning completed: {os.path.basename(task.file_path)}",
                timestamp=datetime.now(),
                data={
                    'processing_time': processing_time,
                    'result_summary': self._summarize_result(result)
                }
            )
            self._send_notification(notification)
            
            logger.info(f"Task {task.task_id} completed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)
            
            self.stats['failed_tasks'] += 1
            
            # Send failure notification
            notification = Notification(
                notification_id=f"failed_{task.task_id}",
                task_id=task.task_id,
                notification_type=NotificationType.TASK_FAILED,
                message=f"Learning failed: {os.path.basename(task.file_path)}",
                timestamp=datetime.now(),
                data={'error': str(e)}
            )
            self._send_notification(notification)
            
            logger.error(f"Task {task.task_id} failed: {e}")
        
        # Save persistent data
        self._save_persistent_data()
    
    def _process_excel_analysis(self, task: LearningTask) -> Dict[str, Any]:
        """Process Excel analysis task"""
        # Update progress
        task.progress = 25.0
        
        # Analyze Excel file
        analyzer = ExcelAnalyzer()
        structure = analyzer.analyze_excel_file(task.file_path)
        
        task.progress = 75.0
        
        # Extract key insights
        insights = {
            'sheets_count': len(structure.sheets),
            'total_cells': sum(len(sheet.cells) for sheet in structure.sheets),
            'formulas_count': sum(len(sheet.formulas) for sheet in structure.sheets),
            'charts_count': sum(len(sheet.charts) for sheet in structure.sheets),
            'data_ranges': structure.data_ranges,
            'patterns_detected': len(structure.patterns)
        }
        
        return {
            'structure': structure,
            'insights': insights,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _process_formula_learning(self, task: LearningTask) -> Dict[str, Any]:
        """Process formula learning task"""
        # Update progress
        task.progress = 20.0
        
        # Analyze Excel file for formulas
        analyzer = ExcelAnalyzer()
        structure = analyzer.analyze_excel_file(task.file_path)
        
        task.progress = 40.0
        
        # Learn formula logic
        formula_learner = FormulaLearner()
        learned_formulas = []
        
        for sheet in structure.sheets:
            for formula in sheet.formulas:
                try:
                    formula_logic = formula_learner.learn_formula_logic(formula.formula)
                    learned_formulas.append({
                        'cell': formula.cell,
                        'formula': formula.formula,
                        'logic': formula_logic
                    })
                except Exception as e:
                    logger.warning(f"Error learning formula {formula.formula}: {e}")
        
        task.progress = 80.0
        
        # Learn filtering operations
        filter_patterns = []
        for pattern in structure.patterns:
            if 'filter' in pattern.pattern_type.lower():
                filter_patterns.append(pattern)
        
        return {
            'learned_formulas': learned_formulas,
            'filter_patterns': filter_patterns,
            'total_formulas': len(learned_formulas),
            'learning_timestamp': datetime.now().isoformat()
        }
    
    def _process_chart_learning(self, task: LearningTask) -> Dict[str, Any]:
        """Process chart learning task"""
        # Update progress
        task.progress = 30.0
        
        # Analyze Excel file for charts
        analyzer = ExcelAnalyzer()
        structure = analyzer.analyze_excel_file(task.file_path)
        
        task.progress = 60.0
        
        # Learn chart patterns
        chart_learner = ChartLearner()
        learned_charts = []
        
        for sheet in structure.sheets:
            for chart in sheet.charts:
                try:
                    chart_pattern = chart_learner.learn_chart_pattern(chart)
                    learned_charts.append({
                        'sheet': sheet.name,
                        'chart': chart,
                        'pattern': chart_pattern
                    })
                except Exception as e:
                    logger.warning(f"Error learning chart: {e}")
        
        return {
            'learned_charts': learned_charts,
            'total_charts': len(learned_charts),
            'chart_types': list(set(chart['pattern'].chart_type for chart in learned_charts)),
            'learning_timestamp': datetime.now().isoformat()
        }
    
    def _process_full_pipeline(self, task: LearningTask) -> Dict[str, Any]:
        """Process full pipeline task (analysis + learning + generation)"""
        # Update progress
        task.progress = 10.0
        
        # Step 1: Excel Analysis
        analyzer = ExcelAnalyzer()
        structure = analyzer.analyze_excel_file(task.file_path)
        
        task.progress = 30.0
        
        # Step 2: Formula Learning
        formula_learner = FormulaLearner()
        learned_formulas = []
        
        for sheet in structure.sheets:
            for formula in sheet.formulas:
                try:
                    formula_logic = formula_learner.learn_formula_logic(formula.formula)
                    learned_formulas.append({
                        'cell': formula.cell,
                        'formula': formula.formula,
                        'logic': formula_logic
                    })
                except Exception as e:
                    logger.warning(f"Error learning formula {formula.formula}: {e}")
        
        task.progress = 50.0
        
        # Step 3: Chart Learning
        chart_learner = ChartLearner()
        learned_charts = []
        
        for sheet in structure.sheets:
            for chart in sheet.charts:
                try:
                    chart_pattern = chart_learner.learn_chart_pattern(chart)
                    learned_charts.append({
                        'sheet': sheet.name,
                        'chart': chart,
                        'pattern': chart_pattern
                    })
                except Exception as e:
                    logger.warning(f"Error learning chart: {e}")
        
        task.progress = 70.0
        
        # Step 4: Model Training
        ml_models = ExcelMLModels()
        training_results = ml_models.train_on_structure(structure)
        
        task.progress = 90.0
        
        # Step 5: Generate sample output
        data_generator = self.learning_pipeline.data_generator
        excel_generator = self.learning_pipeline.excel_generator
        
        # Generate sample data based on learned patterns
        sample_data = data_generator.generate_data(
            patterns=structure.patterns,
            config=GenerationConfig(
                num_records=100,
                include_formulas=True,
                include_charts=True
            )
        )
        
        # Generate sample Excel file
        output_path = self.storage_path / f"generated_{task.task_id}.xlsx"
        generation_config = ExcelGenerationConfig(
            output_path=str(output_path),
            include_charts=True,
            include_formatting=True
        )
        
        excel_result = excel_generator.generate_excel(
            data=sample_data,
            structure=structure,
            config=generation_config
        )
        
        return {
            'structure': structure,
            'learned_formulas': learned_formulas,
            'learned_charts': learned_charts,
            'training_results': training_results,
            'generated_file': str(output_path),
            'generation_result': excel_result,
            'total_formulas': len(learned_formulas),
            'total_charts': len(learned_charts),
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the processing result"""
        summary = {
            'result_type': 'unknown',
            'key_metrics': {}
        }
        
        if 'insights' in result:
            summary['result_type'] = 'excel_analysis'
            summary['key_metrics'] = {
                'sheets_count': result['insights'].get('sheets_count', 0),
                'formulas_count': result['insights'].get('formulas_count', 0),
                'charts_count': result['insights'].get('charts_count', 0)
            }
        elif 'learned_formulas' in result:
            summary['result_type'] = 'formula_learning'
            summary['key_metrics'] = {
                'total_formulas': result.get('total_formulas', 0),
                'filter_patterns': len(result.get('filter_patterns', []))
            }
        elif 'learned_charts' in result:
            summary['result_type'] = 'chart_learning'
            summary['key_metrics'] = {
                'total_charts': result.get('total_charts', 0),
                'chart_types': result.get('chart_types', [])
            }
        elif 'generated_file' in result:
            summary['result_type'] = 'full_pipeline'
            summary['key_metrics'] = {
                'total_formulas': result.get('total_formulas', 0),
                'total_charts': result.get('total_charts', 0),
                'generated_file': result.get('generated_file', '')
            }
        
        return summary
    
    def get_task_status(self, task_id: str) -> Optional[LearningTask]:
        """Get the status of a specific task"""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        return None
    
    def get_all_tasks(self) -> Dict[str, LearningTask]:
        """Get all tasks (active and completed)"""
        all_tasks = {}
        all_tasks.update(self.active_tasks)
        all_tasks.update(self.completed_tasks)
        return all_tasks
    
    def get_notifications(self, unread_only: bool = False) -> List[Notification]:
        """Get notifications, optionally filtered to unread only"""
        if unread_only:
            return [n for n in self.notifications if not n.read]
        return self.notifications.copy()
    
    def mark_notification_read(self, notification_id: str):
        """Mark a notification as read"""
        for notification in self.notifications:
            if notification.notification_id == notification_id:
                notification.read = True
                break
        self._save_persistent_data()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        stats.update({
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'queue_size': self.task_queue.qsize(),
            'system_running': self.running,
            'workers_count': len(self.workers)
        })
        return stats
    
    def _dict_to_task(self, task_dict: Dict[str, Any]) -> LearningTask:
        """Convert dictionary to LearningTask"""
        task_dict['status'] = TaskStatus(task_dict['status'])
        task_dict['created_at'] = datetime.fromisoformat(task_dict['created_at'])
        if task_dict.get('started_at'):
            task_dict['started_at'] = datetime.fromisoformat(task_dict['started_at'])
        if task_dict.get('completed_at'):
            task_dict['completed_at'] = datetime.fromisoformat(task_dict['completed_at'])
        return LearningTask(**task_dict)
    
    def _dict_to_notification(self, notif_dict: Dict[str, Any]) -> Notification:
        """Convert dictionary to Notification"""
        notif_dict['notification_type'] = NotificationType(notif_dict['notification_type'])
        notif_dict['timestamp'] = datetime.fromisoformat(notif_dict['timestamp'])
        return Notification(**notif_dict)
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old completed tasks and notifications"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up old completed tasks
        old_task_ids = [
            task_id for task_id, task in self.completed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_date
        ]
        for task_id in old_task_ids:
            del self.completed_tasks[task_id]
        
        # Clean up old notifications
        self.notifications = [
            n for n in self.notifications
            if n.timestamp > cutoff_date
        ]
        
        # Save cleaned data
        self._save_persistent_data()
        
        logger.info(f"Cleaned up {len(old_task_ids)} old tasks and old notifications")
    
    def export_learning_results(self, output_path: str):
        """Export all learning results to a file"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'statistics': self.stats,
                'completed_tasks': [asdict(task) for task in self.completed_tasks.values()],
                'notifications': [asdict(notif) for notif in self.notifications]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, default=str, indent=2)
            
            logger.info(f"Exported learning results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting learning results: {e}")
            raise
