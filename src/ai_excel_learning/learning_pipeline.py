"""
Learning Pipeline for Excel AI

This module orchestrates the entire AI learning process from Excel analysis
to model training and file generation.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import time

from .excel_analyzer import ExcelAnalyzer, ExcelStructure
from .ml_models import ExcelMLModels
from .chart_learner import ChartLearner, ChartPattern, ChartTemplate
from .data_generator import DataGenerator, GenerationConfig
from .excel_generator import ExcelGenerator, ExcelGenerationConfig
from .formula_learner import FormulaLearner, FormulaLogic, FilterCondition, CalculationPattern

logger = logging.getLogger(__name__)

@dataclass
class LearningSession:
    """Represents a learning session"""
    session_id: str
    timestamp: str
    input_files: List[str]
    models_trained: List[str]
    templates_created: List[str]
    performance_metrics: Dict[str, Any]

@dataclass
class PipelineConfig:
    """Configuration for the learning pipeline"""
    models_dir: str = "models"
    templates_dir: str = "templates"
    output_dir: str = "output"
    analysis_dir: str = "analysis"
    enable_charts: bool = True
    enable_formatting: bool = True
    enable_formulas: bool = True
    model_types: List[str] = None
    chart_types: List[str] = None

class LearningPipeline:
    """
    Orchestrates the complete AI learning process for Excel files
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.analyzer = ExcelAnalyzer()
        self.ml_models = ExcelMLModels(self.config.models_dir)
        self.chart_learner = ChartLearner(self.config.templates_dir)
        self.data_generator = DataGenerator()
        self.excel_generator = ExcelGenerator()
        
        # Formula learning capabilities
        self.formula_learner = FormulaLearner()
        
        # Create directories
        self._create_directories()
        
        # Session tracking
        self.sessions = []
        self.current_session = None
        
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.models_dir,
            self.config.templates_dir,
            self.config.output_dir,
            self.config.analysis_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def start_learning_session(self, session_name: str = None) -> str:
        """
        Start a new learning session
        
        Args:
            session_name: Optional name for the session
            
        Returns:
            Session ID
        """
        session_id = session_name or f"session_{int(time.time())}"
        
        self.current_session = LearningSession(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            input_files=[],
            models_trained=[],
            templates_created=[],
            performance_metrics={}
        )
        
        logger.info(f"Started learning session: {session_id}")
        return session_id
    
    def learn_from_excel_files(self, excel_files: List[str], 
                              session_id: str = None) -> Dict[str, Any]:
        """
        Learn from multiple Excel files
        
        Args:
            excel_files: List of Excel file paths
            session_id: Optional session ID
            
        Returns:
            Learning results
        """
        if session_id is None:
            session_id = self.start_learning_session()
        
        logger.info(f"Learning from {len(excel_files)} Excel files")
        
        results = {
            'session_id': session_id,
            'files_processed': [],
            'models_trained': [],
            'templates_created': [],
            'performance_metrics': {}
        }
        
        start_time = time.time()
        
        # Process each file
        for excel_file in excel_files:
            try:
                file_result = self._process_single_file(excel_file, session_id)
                results['files_processed'].append(file_result)
                
                # Track in current session
                if self.current_session:
                    self.current_session.input_files.append(excel_file)
                
            except Exception as e:
                logger.error(f"Error processing file {excel_file}: {str(e)}")
                results['files_processed'].append({
                    'file': excel_file,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Train models on combined data
        models_trained = self._train_models_from_analysis(results['files_processed'])
        results['models_trained'] = models_trained
        
        # Create templates
        templates_created = self._create_templates_from_analysis(results['files_processed'])
        results['templates_created'] = templates_created
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        results['performance_metrics'] = {
            'processing_time_seconds': processing_time,
            'files_per_second': len(excel_files) / processing_time,
            'models_trained_count': len(models_trained),
            'templates_created_count': len(templates_created)
        }
        
        # Update current session
        if self.current_session:
            self.current_session.models_trained = models_trained
            self.current_session.templates_created = templates_created
            self.current_session.performance_metrics = results['performance_metrics']
            self.sessions.append(self.current_session)
        
        logger.info(f"Learning session completed. Processed {len(excel_files)} files in {processing_time:.2f} seconds")
        return results
    
    def _process_single_file(self, excel_file: str, session_id: str) -> Dict[str, Any]:
        """Process a single Excel file"""
        logger.info(f"Processing file: {excel_file}")
        
        # Analyze the file
        structure = self.analyzer.analyze_excel_file(excel_file)
        
        # Save analysis
        analysis_path = Path(self.config.analysis_dir) / f"{Path(excel_file).stem}_{session_id}_analysis.json"
        self.analyzer.save_analysis(structure, str(analysis_path))
        
        # Learn chart patterns if enabled
        chart_patterns = []
        if self.config.enable_charts:
            chart_patterns = self.chart_learner.learn_chart_patterns(structure.__dict__)
        
        return {
            'file': excel_file,
            'status': 'success',
            'structure': structure.__dict__,
            'analysis_path': str(analysis_path),
            'chart_patterns': [pattern.__dict__ for pattern in chart_patterns],
            'data_ranges_count': len(structure.data_ranges),
            'charts_count': sum(len(charts) for charts in structure.charts.values()),
            'formulas_count': sum(len(formulas) for formulas in structure.formulas.values())
        }
    
    def _train_models_from_analysis(self, file_results: List[Dict[str, Any]]) -> List[str]:
        """Train models from analysis results"""
        trained_models = []
        
        # Collect all data patterns
        all_patterns = []
        for result in file_results:
            if result['status'] == 'success':
                for sheet_name, sheet_data in result['structure']['data_ranges'].items():
                    for range_data in sheet_data.get('ranges', []):
                        all_patterns.extend(range_data.get('patterns', []))
        
        # Train models for different data types
        numeric_patterns = [p for p in all_patterns if p.get('data_type') == 'numeric']
        categorical_patterns = [p for p in all_patterns if p.get('data_type') == 'categorical']
        
        # Train numeric models
        if numeric_patterns:
            for i, pattern in enumerate(numeric_patterns[:5]):  # Limit to first 5 patterns
                try:
                    # Create synthetic data for training
                    training_data = self._create_training_data_from_pattern(pattern)
                    
                    model_name = f"numeric_model_{i}_{int(time.time())}"
                    result = self.ml_models.train_numeric_model(
                        training_data, 
                        'target', 
                        model_name,
                        'neural_network'
                    )
                    trained_models.append(model_name)
                    
                except Exception as e:
                    logger.warning(f"Failed to train numeric model {i}: {str(e)}")
        
        # Train categorical models
        if categorical_patterns:
            for i, pattern in enumerate(categorical_patterns[:5]):  # Limit to first 5 patterns
                try:
                    # Create synthetic data for training
                    training_data = self._create_training_data_from_pattern(pattern)
                    
                    model_name = f"categorical_model_{i}_{int(time.time())}"
                    result = self.ml_models.train_categorical_model(
                        training_data,
                        'target',
                        model_name,
                        'random_forest'
                    )
                    trained_models.append(model_name)
                    
                except Exception as e:
                    logger.warning(f"Failed to train categorical model {i}: {str(e)}")
        
        return trained_models
    
    def _create_training_data_from_pattern(self, pattern: Dict[str, Any]) -> pd.DataFrame:
        """Create training data from a pattern"""
        # Generate synthetic data based on pattern statistics
        stats = pattern.get('statistics', {})
        
        if pattern.get('data_type') == 'numeric':
            mean = stats.get('mean', 0)
            std = stats.get('std', 1)
            count = stats.get('count', 100)
            
            # Generate features
            feature1 = np.random.normal(mean, std, count)
            feature2 = np.random.normal(mean * 0.5, std * 0.5, count)
            
            # Generate target with some relationship
            target = feature1 * 0.7 + feature2 * 0.3 + np.random.normal(0, std * 0.1, count)
            
            return pd.DataFrame({
                'feature1': feature1,
                'feature2': feature2,
                'target': target
            })
        
        else:  # categorical
            count = stats.get('count', 100)
            
            # Generate features
            feature1 = np.random.choice(['A', 'B', 'C'], count)
            feature2 = np.random.choice(['X', 'Y', 'Z'], count)
            
            # Generate target based on features
            target = []
            for f1, f2 in zip(feature1, feature2):
                if f1 == 'A' and f2 == 'X':
                    target.append('High')
                elif f1 == 'B' and f2 == 'Y':
                    target.append('Medium')
                else:
                    target.append('Low')
            
            return pd.DataFrame({
                'feature1': feature1,
                'feature2': feature2,
                'target': target
            })
    
    def _create_templates_from_analysis(self, file_results: List[Dict[str, Any]]) -> List[str]:
        """Create templates from analysis results"""
        created_templates = []
        
        for result in file_results:
            if result['status'] == 'success':
                # Create chart templates
                for pattern in result.get('chart_patterns', []):
                    try:
                        chart_pattern = ChartPattern(**pattern)
                        template_name = f"chart_template_{len(created_templates)}_{int(time.time())}"
                        
                        template = self.chart_learner.create_chart_template(chart_pattern, template_name)
                        created_templates.append(template_name)
                        
                    except Exception as e:
                        logger.warning(f"Failed to create chart template: {str(e)}")
        
        return created_templates
    
    def generate_excel_file(self, session_id: str, 
                           output_filename: str,
                           num_rows: int = 100,
                           num_columns: int = 5) -> str:
        """
        Generate an Excel file using learned models and templates
        
        Args:
            session_id: Session ID to use for generation
            output_filename: Name for the output file
            num_rows: Number of rows to generate
            num_columns: Number of columns to generate
            
        Returns:
            Path to the generated file
        """
        logger.info(f"Generating Excel file using session {session_id}")
        
        # Find the session
        session = self._find_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Generate data using trained models
        data = self._generate_data_using_models(session, num_rows, num_columns)
        
        # Create generation config
        config = self._create_generation_config(session)
        
        # Generate Excel file
        output_path = Path(self.config.output_dir) / output_filename
        generated_path = self.excel_generator.generate_excel_file(
            data, config, str(output_path)
        )
        
        logger.info(f"Excel file generated: {generated_path}")
        return generated_path
    
    def _find_session(self, session_id: str) -> Optional[LearningSession]:
        """Find a learning session by ID"""
        for session in self.sessions:
            if session.session_id == session_id:
                return session
        return None
    
    def _generate_data_using_models(self, session: LearningSession, 
                                  num_rows: int, num_columns: int) -> pd.DataFrame:
        """Generate data using trained models"""
        # Load available models
        available_models = self.ml_models.list_models()
        
        if not available_models:
            # Fallback to pattern-based generation
            return self._generate_fallback_data(num_rows, num_columns)
        
        # Generate data using the first available model
        model_name = available_models[0]
        self.ml_models.load_model(model_name)
        
        try:
            data = self.ml_models.generate_data(model_name, num_rows)
            return data
        except Exception as e:
            logger.warning(f"Failed to generate data using model {model_name}: {str(e)}")
            return self._generate_fallback_data(num_rows, num_columns)
    
    def _generate_fallback_data(self, num_rows: int, num_columns: int) -> pd.DataFrame:
        """Generate fallback data when no models are available"""
        config = GenerationConfig(
            num_rows=num_rows,
            num_columns=num_columns,
            data_types=['numeric', 'categorical'],
            patterns=['random', 'sequential'],
            constraints={},
            relationships=[]
        )
        
        # Create dummy patterns
        patterns = []
        for i in range(num_columns):
            pattern = {
                'pattern_type': 'random' if i % 2 == 0 else 'sequential',
                'data_type': 'numeric' if i % 2 == 0 else 'categorical',
                'statistics': {
                    'mean': i * 10,
                    'std': 5,
                    'min': 0,
                    'max': 100
                }
            }
            patterns.append(pattern)
        
        return self.data_generator.generate_data_from_patterns(patterns, config)
    
    def _create_generation_config(self, session: LearningSession) -> ExcelGenerationConfig:
        """Create generation configuration from session"""
        # Load available templates
        available_templates = self.chart_learner.list_templates()
        
        config = ExcelGenerationConfig(
            filename="generated_file",
            sheets=[{
                'name': 'Sheet1',
                'start_row': 1,
                'start_col': 1,
                'include_headers': True,
                'formatting': {}
            }],
            charts=[],
            formatting={
                'default_styles': {
                    'header_style': {'bold': True, 'size': 12},
                    'data_style': {'size': 11}
                }
            },
            metadata={
                'properties': {
                    'title': 'AI Generated Excel File',
                    'creator': 'DataChaEnhanced AI',
                    'created': datetime.now()
                }
            }
        )
        
        # Add charts if templates are available
        if available_templates:
            template_name = available_templates[0]
            template = self.chart_learner.load_template(template_name)
            
            chart_config = {
                'sheet': 'Sheet1',
                'type': template.chart_type,
                'title': template.template_config['visual_config'].get('title', 'Chart'),
                'position': {'anchor': 'D1'},
                'series': template.template_config['series_config']
            }
            config.charts.append(chart_config)
        
        return config
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a learning session"""
        session = self._find_session(session_id)
        if not session:
            return {}
        
        return {
            'session_id': session.session_id,
            'timestamp': session.timestamp,
            'input_files_count': len(session.input_files),
            'models_trained_count': len(session.models_trained),
            'templates_created_count': len(session.templates_created),
            'performance_metrics': session.performance_metrics
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all learning sessions"""
        return [self.get_session_summary(session.session_id) for session in self.sessions]
    
    def save_pipeline_state(self, file_path: str):
        """Save the current pipeline state"""
        state = {
            'sessions': [session.__dict__ for session in self.sessions],
            'config': self.config.__dict__,
            'models': self.ml_models.list_models(),
            'templates': self.chart_learner.list_templates()
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Pipeline state saved to: {file_path}")
    
    def load_pipeline_state(self, file_path: str):
        """Load pipeline state from file"""
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        # Restore sessions
        self.sessions = [LearningSession(**session_data) for session_data in state['sessions']]
        
        # Load models
        for model_name in state['models']:
            try:
                self.ml_models.load_model(model_name)
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {str(e)}")
        
        # Load templates
        for template_name in state['templates']:
            try:
                self.chart_learner.load_template(template_name)
            except Exception as e:
                logger.warning(f"Failed to load template {template_name}: {str(e)}")
        
        logger.info(f"Pipeline state loaded from: {file_path}")
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        if not self.sessions:
            return {}
        
        total_sessions = len(self.sessions)
        total_files = sum(len(session.input_files) for session in self.sessions)
        total_models = sum(len(session.models_trained) for session in self.sessions)
        total_templates = sum(len(session.templates_created) for session in self.sessions)
        
        avg_processing_time = np.mean([
            session.performance_metrics.get('processing_time_seconds', 0) 
            for session in self.sessions
        ])
        
        return {
            'total_sessions': total_sessions,
            'total_files_processed': total_files,
            'total_models_trained': total_models,
            'total_templates_created': total_templates,
            'average_processing_time_seconds': avg_processing_time,
            'average_files_per_session': total_files / total_sessions,
            'average_models_per_session': total_models / total_sessions
        }
    
    def learn_formula_logic(self, formula: str, context_data: pd.DataFrame = None) -> FormulaLogic:
        """
        Learn the logic behind an Excel formula
        
        Args:
            formula: The Excel formula string
            context_data: Optional DataFrame for context analysis
            
        Returns:
            FormulaLogic object representing the learned logic
        """
        return self.formula_learner.learn_formula_logic(formula, context_data)
    
    def apply_learned_formula_logic(self, formula_logic: FormulaLogic, data: pd.DataFrame, 
                                  target_context: str = None) -> Any:
        """
        Apply learned formula logic to new data
        
        Args:
            formula_logic: The learned formula logic
            data: DataFrame to apply logic to
            target_context: Optional target context for the operation
            
        Returns:
            Result of applying the learned logic
        """
        return self.formula_learner.apply_learned_logic(formula_logic, data, target_context)
    
    def learn_filtering_operations(self, data: pd.DataFrame, filter_conditions: List[Dict[str, Any]]) -> List[FilterCondition]:
        """
        Learn filtering operations from data and conditions
        
        Args:
            data: DataFrame being filtered
            filter_conditions: List of filter conditions applied
            
        Returns:
            List of learned FilterCondition objects
        """
        return self.formula_learner.learn_filtering_operations(data, filter_conditions)
    
    def apply_learned_filters(self, data: pd.DataFrame, filters: List[FilterCondition] = None) -> pd.DataFrame:
        """
        Apply learned filtering operations to data
        
        Args:
            data: DataFrame to filter
            filters: List of FilterCondition objects (uses learned filters if None)
            
        Returns:
            Filtered DataFrame
        """
        return self.formula_learner.apply_learned_filters(data, filters)
    
    def get_learned_formula_patterns(self) -> Dict[str, CalculationPattern]:
        """Get all learned formula patterns"""
        return self.formula_learner.get_learned_patterns()
    
    def get_learned_filter_patterns(self) -> List[FilterCondition]:
        """Get all learned filter patterns"""
        return self.formula_learner.get_filter_patterns()
    
    def save_learned_patterns(self, filepath: str):
        """Save learned patterns to file"""
        self.formula_learner.save_learned_patterns(filepath)
    
    def load_learned_patterns(self, filepath: str):
        """Load learned patterns from file"""
        self.formula_learner.load_learned_patterns(filepath)
    
    def demonstrate_formula_learning(self, sample_formulas: List[str] = None) -> Dict[str, Any]:
        """
        Demonstrate formula learning capabilities
        
        Args:
            sample_formulas: List of sample formulas to learn from
            
        Returns:
            Dictionary with demonstration results
        """
        if sample_formulas is None:
            sample_formulas = [
                "=SUM(A1:A10)",
                "=AVERAGE(B2:B50)",
                "=IF(C1>250, 'High', 'Low')",
                "=FILTER(A1:C100, B1:B100>250)",
                "=A1*B1+C1"
            ]
        
        results = {
            'learned_patterns': {},
            'demonstrations': {}
        }
        
        # Create sample data for demonstrations
        sample_data = pd.DataFrame({
            'A': range(1, 101),
            'B': np.random.uniform(100, 300, 100),
            'C': np.random.choice(['Low', 'Medium', 'High'], 100)
        })
        
        for formula in sample_formulas:
            try:
                # Learn formula logic
                formula_logic = self.learn_formula_logic(formula, sample_data)
                
                # Store learned pattern
                results['learned_patterns'][formula] = {
                    'formula_type': formula_logic.formula_type.value,
                    'operation': formula_logic.operation,
                    'source_range': formula_logic.source_range,
                    'conditions': formula_logic.conditions,
                    'parameters': formula_logic.parameters,
                    'confidence': formula_logic.confidence
                }
                
                # Demonstrate application
                if formula_logic.confidence > 0:
                    applied_result = self.apply_learned_formula_logic(formula_logic, sample_data)
                    results['demonstrations'][formula] = {
                        'result_type': type(applied_result).__name__,
                        'result_preview': str(applied_result)[:100] if applied_result is not None else None
                    }
                
            except Exception as e:
                logger.error(f"Error demonstrating formula {formula}: {e}")
                results['learned_patterns'][formula] = {'error': str(e)}
        
        return results
    
    def create_chart_with_formula_logic(self, chart_template_name: str, data: pd.DataFrame,
                                      formula_logic: FormulaLogic = None,
                                      filter_conditions: List[FilterCondition] = None,
                                      output_path: str = None) -> Dict[str, Any]:
        """
        Create a chart that incorporates learned formula logic and filtering
        
        Args:
            chart_template_name: Name of the chart template to use
            data: DataFrame with data for the chart
            formula_logic: Optional learned formula logic to apply
            filter_conditions: Optional filter conditions to apply
            output_path: Optional path to save the chart image
            
        Returns:
            Dictionary with chart configuration and metadata
        """
        try:
            # Create a temporary template with formula logic
            if chart_template_name not in self.chart_learner.templates:
                # Create a basic template if it doesn't exist
                basic_template = ChartTemplate(
                    name=chart_template_name,
                    chart_type='line',
                    template_config={
                        'visual_config': {
                            'title': f'Chart with Formula Logic: {formula_logic.formula_type.value if formula_logic else "None"}',
                            'x_axis': {'title': 'Index'},
                            'y_axis': {'title': 'Values'}
                        },
                        'series_config': [{'name': 'Data Series'}]
                    },
                    data_requirements={'min_series': 1, 'min_data_points': 5},
                    formula_logic=formula_logic,
                    filter_conditions=filter_conditions
                )
                self.chart_learner.templates[chart_template_name] = basic_template
            
            # Generate chart with formula logic and filters applied
            chart_config = self.chart_learner.generate_chart(
                chart_template_name, 
                data, 
                output_path=output_path,
                apply_formula_logic=True,
                apply_filters=True
            )
            
            logger.info(f"Created chart with formula logic: {chart_template_name}")
            return chart_config
            
        except Exception as e:
            logger.error(f"Error creating chart with formula logic: {e}")
            return {'error': str(e)}
    
    def demonstrate_chart_with_formulas(self, sample_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Demonstrate creating charts that incorporate learned formula logic and filtering
        
        Args:
            sample_data: Optional sample data to use
            
        Returns:
            Dictionary with demonstration results
        """
        if sample_data is None:
            # Create sample data with current measurements
            sample_data = pd.DataFrame({
                'Current_mA': np.random.uniform(100, 300, 100),
                'Voltage_V': np.random.uniform(3.0, 5.0, 100),
                'Temperature_C': np.random.uniform(20, 80, 100),
                'Status': np.random.choice(['OK', 'Warning', 'Error'], 100),
                'Timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
            })
        
        results = {
            'charts_created': [],
            'formula_applications': [],
            'filter_applications': []
        }
        
        # Example 1: Chart with SUM formula
        try:
            sum_formula = self.learn_formula_logic("=SUM(A1:A10)")
            chart1 = self.create_chart_with_formula_logic(
                "sum_chart", 
                sample_data, 
                formula_logic=sum_formula
            )
            results['charts_created'].append({
                'name': 'sum_chart',
                'formula_type': 'SUM',
                'config': chart1
            })
            results['formula_applications'].append('SUM formula applied to chart data')
        except Exception as e:
            logger.error(f"Error creating SUM chart: {e}")
        
        # Example 2: Chart with filtering (values above 250 mA)
        try:
            filter_conditions = [
                FilterCondition(column='Current_mA', operator='>', value=250)
            ]
            chart2 = self.create_chart_with_formula_logic(
                "filtered_chart", 
                sample_data, 
                filter_conditions=filter_conditions
            )
            results['charts_created'].append({
                'name': 'filtered_chart',
                'filter_type': 'Current > 250mA',
                'config': chart2
            })
            results['filter_applications'].append('Filter applied: Current_mA > 250')
        except Exception as e:
            logger.error(f"Error creating filtered chart: {e}")
        
        # Example 3: Chart with conditional logic
        try:
            if_formula = self.learn_formula_logic("=IF(A1>250, 'High', 'Low')")
            chart3 = self.create_chart_with_formula_logic(
                "conditional_chart", 
                sample_data, 
                formula_logic=if_formula
            )
            results['charts_created'].append({
                'name': 'conditional_chart',
                'formula_type': 'IF',
                'config': chart3
            })
            results['formula_applications'].append('IF formula applied to chart data')
        except Exception as e:
            logger.error(f"Error creating conditional chart: {e}")
        
        return results
