# src/analysis/ai_realtime_integration.py
"""
Real-Time AI Integration Module

This module provides seamless integration between the AI system and the main
application, enabling real-time predictions, batch processing, and continuous learning.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import threading
import queue
from pathlib import Path
import json
from typing import Optional, Dict, List, Tuple
from src.utils.logger import app_logger
from src.analysis.ai_curve_learning import CurveFittingAI, ExcelAnalysisCollector
from src.analysis.ai_confidence_validation import PredictionConfidenceEstimator, ModelValidator, ActiveLearningSelector
from src.excel_charted.enhanced_excel_export_with_charts import EnhancedExcelExporter

class AIIntegrationManager:
    """
    Central manager for AI-enhanced curve analysis workflow.
    
    Coordinates:
    - Real-time predictions during data acquisition
    - Batch processing of multiple experiments
    - Continuous learning from expert feedback
    - Model versioning and A/B testing
    """
    
    def __init__(self, main_app):
        self.main_app = main_app
        self.ai_system = CurveFittingAI()
        self.confidence_estimator = PredictionConfidenceEstimator(self.ai_system)
        self.validator = ModelValidator(self.ai_system)
        self.active_learner = ActiveLearningSelector(self.confidence_estimator)
        self.excel_collector = ExcelAnalysisCollector(self.ai_system)
        
        # Processing queues
        self.prediction_queue = queue.Queue()
        self.training_queue = queue.Queue()
        
        # State management
        self.is_running = False
        self.current_mode = 'manual'  # 'manual', 'assisted', 'automatic'
        self.model_versions = {}
        self.active_model_version = None
        
        # Performance tracking
        self.performance_metrics = {
            'predictions_made': 0,
            'high_confidence_predictions': 0,
            'expert_overrides': 0,
            'processing_times': []
        }
        
        # Initialize
        self._load_latest_model()
        app_logger.info("AI Integration Manager initialized")
    
    def _load_latest_model(self):
        """Load the most recent trained model."""
        if self.ai_system.load_latest_models():
            self.active_model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_versions[self.active_model_version] = {
                'loaded_at': datetime.now(),
                'training_examples': len(self.ai_system.training_data)
            }
            app_logger.info(f"Loaded model version: {self.active_model_version}")
    
    def start_real_time_processing(self):
        """Start real-time processing threads."""
        if not self.is_running:
            self.is_running = True
            
            # Start prediction thread
            self.prediction_thread = threading.Thread(
                target=self._prediction_worker,
                daemon=True
            )
            self.prediction_thread.start()
            
            # Start training thread
            self.training_thread = threading.Thread(
                target=self._training_worker,
                daemon=True
            )
            self.training_thread.start()
            
            app_logger.info("Real-time processing started")
    
    def stop_real_time_processing(self):
        """Stop real-time processing threads."""
        self.is_running = False
        app_logger.info("Real-time processing stopped")
    
    def _prediction_worker(self):
        """Worker thread for processing prediction requests."""
        while self.is_running:
            try:
                # Get prediction request (timeout to check is_running)
                request = self.prediction_queue.get(timeout=1.0)
                
                if request is None:
                    continue
                
                # Process prediction
                start_time = datetime.now()
                result = self._process_prediction_request(request)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Track performance
                self.performance_metrics['predictions_made'] += 1
                self.performance_metrics['processing_times'].append(processing_time)
                
                # Send result back to main app
                if result and 'callback' in request:
                    request['callback'](result)
                
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"Prediction worker error: {str(e)}")
    
    def _training_worker(self):
        """Worker thread for processing training updates."""
        while self.is_running:
            try:
                # Get training request
                request = self.training_queue.get(timeout=5.0)
                
                if request is None:
                    continue
                
                # Process training update
                if request['type'] == 'add_example':
                    self._add_training_example(request['data'])
                elif request['type'] == 'retrain':
                    self._retrain_models()
                elif request['type'] == 'scan_directory':
                    self._scan_for_training_data(request['path'])
                
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"Training worker error: {str(e)}")
    
    def _process_prediction_request(self, request):
        """Process a single prediction request."""
        try:
            curve_data = request['curve_data']
            
            # Get predictions with confidence
            results = self.confidence_estimator.predict_with_confidence(curve_data)
            
            if results:
                # Check confidence level
                for curve_type, result in results.items():
                    if result['confidence']['level'] == 'high':
                        self.performance_metrics['high_confidence_predictions'] += 1
                
                # Add processing metadata
                results['metadata'] = {
                    'model_version': self.active_model_version,
                    'processing_mode': self.current_mode,
                    'timestamp': datetime.now().isoformat()
                }
            
            return results
            
        except Exception as e:
            app_logger.error(f"Prediction processing error: {str(e)}")
            return None
    
    def request_prediction(self, curve_data, callback=None, priority=False):
        """
        Request AI prediction for curve data.
        
        Args:
            curve_data: Dictionary with 'hyperpol' and 'depol' curve data
            callback: Function to call with results
            priority: If True, process immediately
        """
        request = {
            'curve_data': curve_data,
            'callback': callback,
            'timestamp': datetime.now()
        }
        
        if priority and self.ai_system.is_trained:
            # Process immediately
            return self._process_prediction_request(request)
        else:
            # Queue for processing
            self.prediction_queue.put(request)
            return None
    
    def add_expert_feedback(self, curve_data, expert_selections, parameters):
        """
        Add expert feedback for continuous learning.
        
        Args:
            curve_data: Original curve data
            expert_selections: Expert's point selections
            parameters: Extracted parameters
        """
        training_data = {
            'curves': curve_data,
            'expert_selections': expert_selections,
            'extracted_parameters': parameters,
            'curve_features': self.ai_system._extract_curve_features(curve_data),
            'timestamp': datetime.now()
        }
        
        # Queue for training
        self.training_queue.put({
            'type': 'add_example',
            'data': training_data
        })
        
        app_logger.info("Expert feedback queued for training")
    
    def process_batch(self, file_list, output_dir, mode='automatic'):
        """
        Process a batch of files with AI assistance.
        
        Args:
            file_list: List of file paths to process
            output_dir: Directory for output files
            mode: 'automatic', 'assisted', or 'manual'
        """
        app_logger.info(f"Starting batch processing in {mode} mode")
        
        results = []
        review_candidates = []
        
        for file_path in file_list:
            try:
                # Load data
                curve_data = self._load_curve_data(file_path)
                
                if mode == 'automatic' and self.ai_system.is_trained:
                    # Full automatic processing
                    prediction_results = self.request_prediction(
                        curve_data, 
                        priority=True
                    )
                    
                    if prediction_results:
                        # Generate Excel report automatically
                        self._generate_automated_report(
                            curve_data, 
                            prediction_results, 
                            output_dir, 
                            file_path
                        )
                        results.append({
                            'file': file_path,
                            'status': 'completed',
                            'mode': 'automatic'
                        })
                    else:
                        review_candidates.append(file_path)
                        
                elif mode == 'assisted':
                    # Get AI suggestions but require confirmation
                    prediction_results = self.request_prediction(
                        curve_data, 
                        priority=True
                    )
                    
                    results.append({
                        'file': file_path,
                        'status': 'pending_review',
                        'predictions': prediction_results,
                        'mode': 'assisted'
                    })
                    
                else:  # manual mode
                    # Just prepare for manual analysis
                    results.append({
                        'file': file_path,
                        'status': 'manual_required',
                        'mode': 'manual'
                    })
                    
            except Exception as e:
                app_logger.error(f"Error processing {file_path}: {str(e)}")
                results.append({
                    'file': file_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Use active learning to select files for review
        if review_candidates and mode == 'automatic':
            selected_for_review = self._select_for_expert_review(review_candidates)
            for file_path in selected_for_review:
                results.append({
                    'file': file_path,
                    'status': 'selected_for_review',
                    'reason': 'active_learning'
                })
        
        return results
    
    def _load_curve_data(self, file_path):
        """Load curve data from file."""
        # Implementation depends on your file format
        # This is a placeholder
        return {
            'hyperpol': {
                'time': np.array([]),
                'current': np.array([])
            },
            'depol': {
                'time': np.array([]),
                'current': np.array([])
            }
        }
    
    def _generate_automated_report(self, curve_data, predictions, output_dir, source_file):
        """Generate automated Excel report with AI predictions."""
        try:
            # Create output filename
            source_name = Path(source_file).stem
            output_file = Path(output_dir) / f"{source_name}_AI_analysis.xlsx"
            
            # Initialize Excel exporter
            exporter = EnhancedExcelExporter(str(output_file))
            
            # Add curve data
            exporter.add_curve_data(
                curve_data['hyperpol']['time'],
                curve_data['hyperpol']['current'],
                curve_data['depol']['time'],
                curve_data['depol']['current']
            )
            
            # Apply AI predictions
            for curve_type in ['hyperpol', 'depol']:
                if curve_type in predictions:
                    pred = predictions[curve_type]['predictions']
                    
                    # Set up analysis with AI predictions
                    exporter.setup_analysis_sheet(
                        curve_type,
                        linear_point1=pred.get('linear_start', 10),
                        linear_point2=pred.get('linear_end', 50),
                        exp_start_point=pred.get('exp_start', 60)
                    )
            
            # Add metadata
            exporter.add_metadata_sheet({
                'source_file': str(source_file),
                'analysis_mode': 'AI_automatic',
                'model_version': self.active_model_version,
                'confidence_scores': {
                    ct: predictions[ct]['confidence']['overall'] 
                    for ct in predictions
                },
                'timestamp': datetime.now().isoformat()
            })
            
            # Save file
            exporter.save()
            app_logger.info(f"Generated automated report: {output_file}")
            
        except Exception as e:
            app_logger.error(f"Error generating automated report: {str(e)}")
    
    def _select_for_expert_review(self, file_list):
        """Select files that would benefit most from expert review."""
        curve_batch = []
        
        for file_path in file_list:
            try:
                curve_data = self._load_curve_data(file_path)
                curve_batch.append(curve_data)
            except:
                continue
        
        # Use active learning selector
        selected = self.active_learner.select_for_review(
            curve_batch, 
            max_selections=min(5, len(curve_batch))
        )
        
        # Return corresponding file paths
        selected_files = []
        for item in selected:
            index = item['index']
            if index < len(file_list):
                selected_files.append(file_list[index])
        
        return selected_files
    
    def _add_training_example(self, training_data):
        """Add a new training example to the AI system."""
        self.ai_system.add_training_example(training_data)
        
        # Check if we should retrain
        if len(self.ai_system.training_data) % 10 == 0:
            app_logger.info("Triggering automatic retraining")
            self.training_queue.put({'type': 'retrain'})
    
    def _retrain_models(self):
        """Retrain AI models with accumulated data."""
        if self.ai_system.train_models():
            # Create new version
            new_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_versions[new_version] = {
                'trained_at': datetime.now(),
                'training_examples': len(self.ai_system.training_data),
                'validation_results': self.validator.perform_cross_validation()
            }
            
            # Update active version
            self.active_model_version = new_version
            app_logger.info(f"Models retrained - new version: {new_version}")
    
    def _scan_for_training_data(self, directory_path):
        """Scan directory for training data."""
        self.excel_collector.scan_directory_for_analyses(directory_path)
        
        # Retrain if we found new examples
        if len(self.ai_system.training_data) > 0:
            self.training_queue.put({'type': 'retrain'})
    
    def get_performance_report(self):
        """Generate performance report for the AI system."""
        report = {
            'summary': {
                'total_predictions': self.performance_metrics['predictions_made'],
                'high_confidence_rate': (
                    self.performance_metrics['high_confidence_predictions'] / 
                    max(1, self.performance_metrics['predictions_made'])
                ),
                'avg_processing_time': np.mean(self.performance_metrics['processing_times'])
                    if self.performance_metrics['processing_times'] else 0
            },
            'model_versions': self.model_versions,
            'active_version': self.active_model_version,
            'training_data_size': len(self.ai_system.training_data)
        }
        
        # Add validation results if available
        if hasattr(self.validator, 'validation_results'):
            report['validation'] = self.validator.validation_results
        
        return report
    
    def export_training_dataset(self, output_path):
        """Export collected training dataset for analysis."""
        try:
            training_df = []
            
            for example in self.ai_system.training_data:
                row = {
                    'timestamp': example.get('timestamp', ''),
                    'file_path': example.get('file_path', '')
                }
                
                # Add selections
                for curve_type in ['hyperpol', 'depol']:
                    if curve_type in example.get('expert_selections', {}):
                        selections = example['expert_selections'][curve_type]
                        for key, value in selections.items():
                            row[f'{curve_type}_{key}'] = value
                
                # Add features
                if 'curve_features' in example:
                    for curve_type in ['hyperpol', 'depol']:
                        if curve_type in example['curve_features']:
                            features = example['curve_features'][curve_type]
                            for key, value in features.items():
                                row[f'{curve_type}_feature_{key}'] = value
                
                training_df.append(row)
            
            # Save to CSV
            df = pd.DataFrame(training_df)
            df.to_csv(output_path, index=False)
            app_logger.info(f"Training dataset exported to: {output_path}")
            
        except Exception as e:
            app_logger.error(f"Error exporting training dataset: {str(e)}")
    
    def switch_mode(self, mode):
        """
        Switch between operational modes.
        
        Args:
            mode: 'manual', 'assisted', or 'automatic'
        """
        if mode in ['manual', 'assisted', 'automatic']:
            old_mode = self.current_mode
            self.current_mode = mode
            app_logger.info(f"Switched from {old_mode} to {mode} mode")
            
            # Notify main app of mode change
            if hasattr(self.main_app, 'on_ai_mode_change'):
                self.main_app.on_ai_mode_change(mode)