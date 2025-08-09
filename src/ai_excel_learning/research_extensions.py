#!/usr/bin/env python3
"""
Research Extensions for AI Excel Learning System

This module provides advanced capabilities specifically designed for researchers:
- Batch processing of multiple Excel files
- Advanced data validation and quality checks
- Statistical analysis and pattern recognition
- Collaboration and sharing features
- Research workflow automation
- Data reproducibility and versioning
- Integration with research tools and databases
"""

import os
import sys
import json
import logging
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import zipfile
import yaml

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .background_processor import BackgroundProcessor, LearningTask, TaskStatus
from .learning_pipeline import LearningPipeline
from .formula_learner import FormulaLearner
from .excel_analyzer import ExcelAnalyzer
from .ml_models import ExcelMLModels

logger = logging.getLogger(__name__)

@dataclass
class ResearchProject:
    """Represents a research project with multiple Excel files"""
    project_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    file_paths: List[str]
    metadata: Dict[str, Any]
    collaborators: List[str]
    tags: List[str]
    status: str = "active"
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.collaborators is None:
            self.collaborators = []
        if self.tags is None:
            self.tags = []

@dataclass
class DataQualityReport:
    """Report on data quality and validation results"""
    file_path: str
    validation_timestamp: datetime
    total_cells: int
    empty_cells: int
    formula_cells: int
    chart_cells: int
    data_types: Dict[str, int]
    anomalies: List[Dict[str, Any]]
    quality_score: float
    recommendations: List[str]

@dataclass
class StatisticalAnalysis:
    """Statistical analysis results for Excel data"""
    file_path: str
    analysis_timestamp: datetime
    descriptive_stats: Dict[str, Any]
    correlations: Dict[str, float]
    patterns: List[Dict[str, Any]]
    outliers: List[Dict[str, Any]]
    trends: List[Dict[str, Any]]

class ResearchExtensions:
    """
    Advanced research extensions for the AI Excel Learning System
    
    Features:
    - Batch processing of multiple Excel files
    - Data quality validation and reporting
    - Statistical analysis and pattern recognition
    - Research project management
    - Collaboration and sharing
    - Data reproducibility and versioning
    - Integration with research databases
    """
    
    def __init__(self, 
                 base_path: str = "research_data",
                 max_workers: int = 4,
                 enable_database: bool = True):
        """
        Initialize research extensions
        
        Args:
            base_path: Base path for storing research data
            max_workers: Maximum number of workers for batch processing
            enable_database: Whether to enable SQLite database for metadata
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.background_processor = BackgroundProcessor(
            max_workers=max_workers,
            storage_path=str(self.base_path / "background_learning")
        )
        self.learning_pipeline = LearningPipeline()
        
        # Research-specific paths
        self.projects_path = self.base_path / "projects"
        self.projects_path.mkdir(exist_ok=True)
        
        self.analysis_path = self.base_path / "analysis"
        self.analysis_path.mkdir(exist_ok=True)
        
        self.collaboration_path = self.base_path / "collaboration"
        self.collaboration_path.mkdir(exist_ok=True)
        
        # Database for metadata
        self.enable_database = enable_database
        if enable_database:
            self.db_path = self.base_path / "research_metadata.db"
            self._init_database()
        
        # Research projects
        self.projects: Dict[str, ResearchProject] = {}
        self._load_projects()
        
        logger.info(f"Research extensions initialized at {self.base_path}")
    
    def _init_database(self):
        """Initialize SQLite database for research metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT,
                    collaborators TEXT,
                    tags TEXT,
                    status TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_analysis (
                    file_path TEXT PRIMARY KEY,
                    project_id TEXT,
                    analysis_timestamp TEXT,
                    quality_score REAL,
                    total_cells INTEGER,
                    formulas_count INTEGER,
                    charts_count INTEGER,
                    data_types TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_results (
                    result_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    task_id TEXT,
                    result_type TEXT,
                    result_data TEXT,
                    created_at TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Research database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _load_projects(self):
        """Load existing research projects"""
        try:
            projects_file = self.projects_path / "projects.json"
            if projects_file.exists():
                with open(projects_file, 'r') as f:
                    projects_data = json.load(f)
                    for project_data in projects_data:
                        project = ResearchProject(
                            project_id=project_data['project_id'],
                            name=project_data['name'],
                            description=project_data['description'],
                            created_at=datetime.fromisoformat(project_data['created_at']),
                            updated_at=datetime.fromisoformat(project_data['updated_at']),
                            file_paths=project_data['file_paths'],
                            metadata=project_data['metadata'],
                            collaborators=project_data['collaborators'],
                            tags=project_data['tags'],
                            status=project_data['status']
                        )
                        self.projects[project.project_id] = project
                
                logger.info(f"Loaded {len(self.projects)} research projects")
                
        except Exception as e:
            logger.warning(f"Error loading projects: {e}")
    
    def _save_projects(self):
        """Save research projects to file"""
        try:
            projects_data = []
            for project in self.projects.values():
                project_dict = asdict(project)
                project_dict['created_at'] = project.created_at.isoformat()
                project_dict['updated_at'] = project.updated_at.isoformat()
                projects_data.append(project_dict)
            
            with open(self.projects_path / "projects.json", 'w') as f:
                json.dump(projects_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving projects: {e}")
    
    def create_research_project(self, 
                              name: str,
                              description: str,
                              file_paths: List[str],
                              collaborators: List[str] = None,
                              tags: List[str] = None,
                              metadata: Dict[str, Any] = None) -> str:
        """
        Create a new research project
        
        Args:
            name: Project name
            description: Project description
            file_paths: List of Excel file paths
            collaborators: List of collaborator emails/names
            tags: List of tags for categorization
            metadata: Additional metadata
            
        Returns:
            Project ID
        """
        # Validate file paths
        valid_paths = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                valid_paths.append(file_path)
            else:
                logger.warning(f"File not found: {file_path}")
        
        if not valid_paths:
            raise ValueError("No valid Excel files provided")
        
        # Generate project ID
        project_id = f"project_{int(datetime.now().timestamp())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        # Create project
        project = ResearchProject(
            project_id=project_id,
            name=name,
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            file_paths=valid_paths,
            metadata=metadata or {},
            collaborators=collaborators or [],
            tags=tags or []
        )
        
        # Store project
        self.projects[project_id] = project
        self._save_projects()
        
        # Create project directory
        project_dir = self.projects_path / project_id
        project_dir.mkdir(exist_ok=True)
        
        # Copy files to project directory
        for file_path in valid_paths:
            dest_path = project_dir / os.path.basename(file_path)
            shutil.copy2(file_path, dest_path)
        
        logger.info(f"Created research project {project_id}: {name}")
        return project_id
    
    def batch_process_project(self, 
                            project_id: str,
                            task_type: str = "full_pipeline",
                            priority: int = 3) -> List[str]:
        """
        Submit all files in a project for batch processing
        
        Args:
            project_id: Project ID to process
            task_type: Type of learning task
            priority: Task priority
            
        Returns:
            List of task IDs
        """
        if project_id not in self.projects:
            raise ValueError(f"Project not found: {project_id}")
        
        project = self.projects[project_id]
        task_ids = []
        
        logger.info(f"Starting batch processing for project {project_id}: {len(project.file_paths)} files")
        
        # Submit tasks for each file
        for file_path in project.file_paths:
            try:
                task_id = self.background_processor.submit_learning_task(
                    file_path=file_path,
                    task_type=task_type,
                    priority=priority,
                    metadata={
                        'project_id': project_id,
                        'project_name': project.name,
                        'batch_processing': True
                    }
                )
                task_ids.append(task_id)
                
            except Exception as e:
                logger.error(f"Error submitting task for {file_path}: {e}")
        
        # Update project
        project.updated_at = datetime.now()
        project.metadata['last_batch_processing'] = {
            'timestamp': datetime.now().isoformat(),
            'task_ids': task_ids,
            'task_type': task_type
        }
        self._save_projects()
        
        logger.info(f"Submitted {len(task_ids)} tasks for batch processing")
        return task_ids
    
    def analyze_data_quality(self, file_path: str) -> DataQualityReport:
        """
        Analyze data quality of an Excel file
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Data quality report
        """
        logger.info(f"Analyzing data quality for {file_path}")
        
        try:
            # Analyze Excel file
            analyzer = ExcelAnalyzer()
            structure = analyzer.analyze_excel_file(file_path)
            
            # Calculate quality metrics
            total_cells = sum(len(sheet.cells) for sheet in structure.sheets)
            empty_cells = sum(
                sum(1 for cell in sheet.cells if not cell.value or str(cell.value).strip() == '')
                for sheet in structure.sheets
            )
            formula_cells = sum(len(sheet.formulas) for sheet in structure.sheets)
            chart_cells = sum(len(sheet.charts) for sheet in structure.sheets)
            
            # Analyze data types
            data_types = {}
            for sheet in structure.sheets:
                for cell in sheet.cells:
                    if cell.value is not None:
                        cell_type = type(cell.value).__name__
                        data_types[cell_type] = data_types.get(cell_type, 0) + 1
            
            # Detect anomalies
            anomalies = self._detect_anomalies(structure)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                total_cells, empty_cells, formula_cells, chart_cells, anomalies
            )
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(
                total_cells, empty_cells, formula_cells, chart_cells, anomalies
            )
            
            report = DataQualityReport(
                file_path=file_path,
                validation_timestamp=datetime.now(),
                total_cells=total_cells,
                empty_cells=empty_cells,
                formula_cells=formula_cells,
                chart_cells=chart_cells,
                data_types=data_types,
                anomalies=anomalies,
                quality_score=quality_score,
                recommendations=recommendations
            )
            
            logger.info(f"Data quality analysis completed. Score: {quality_score:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing data quality: {e}")
            raise
    
    def _detect_anomalies(self, structure) -> List[Dict[str, Any]]:
        """Detect anomalies in Excel structure"""
        anomalies = []
        
        for sheet in structure.sheets:
            # Check for empty sheets
            if len(sheet.cells) == 0:
                anomalies.append({
                    'type': 'empty_sheet',
                    'sheet': sheet.name,
                    'severity': 'low',
                    'description': 'Sheet contains no data'
                })
            
            # Check for inconsistent data types
            data_types_by_column = {}
            for cell in sheet.cells:
                if cell.value is not None:
                    col = cell.cell[0] if len(cell.cell) > 0 else 'A'
                    if col not in data_types_by_column:
                        data_types_by_column[col] = set()
                    data_types_by_column[col].add(type(cell.value).__name__)
            
            for col, types in data_types_by_column.items():
                if len(types) > 3:  # More than 3 different types in one column
                    anomalies.append({
                        'type': 'inconsistent_data_types',
                        'sheet': sheet.name,
                        'column': col,
                        'severity': 'medium',
                        'description': f'Column {col} has {len(types)} different data types: {types}'
                    })
            
            # Check for broken formulas
            for formula in sheet.formulas:
                if 'ERROR' in str(formula.value).upper():
                    anomalies.append({
                        'type': 'broken_formula',
                        'sheet': sheet.name,
                        'cell': formula.cell,
                        'severity': 'high',
                        'description': f'Formula in {formula.cell} returns error: {formula.value}'
                    })
        
        return anomalies
    
    def _calculate_quality_score(self, total_cells, empty_cells, formula_cells, chart_cells, anomalies) -> float:
        """Calculate overall data quality score"""
        if total_cells == 0:
            return 0.0
        
        # Base score from data completeness
        completeness_score = 1.0 - (empty_cells / total_cells)
        
        # Bonus for formulas and charts (indicates structured data)
        structure_bonus = min(0.2, (formula_cells + chart_cells) / max(total_cells, 1) * 0.5)
        
        # Penalty for anomalies
        anomaly_penalty = min(0.3, len(anomalies) * 0.05)
        
        # Calculate final score
        score = completeness_score + structure_bonus - anomaly_penalty
        return max(0.0, min(1.0, score))
    
    def _generate_quality_recommendations(self, total_cells, empty_cells, formula_cells, chart_cells, anomalies) -> List[str]:
        """Generate recommendations for improving data quality"""
        recommendations = []
        
        if empty_cells / max(total_cells, 1) > 0.3:
            recommendations.append("Consider removing or filling empty cells to improve data completeness")
        
        if formula_cells == 0:
            recommendations.append("Consider adding formulas for data validation and calculations")
        
        if chart_cells == 0:
            recommendations.append("Consider adding charts for better data visualization")
        
        for anomaly in anomalies:
            if anomaly['type'] == 'broken_formula':
                recommendations.append(f"Fix broken formula in {anomaly['cell']}: {anomaly['description']}")
            elif anomaly['type'] == 'inconsistent_data_types':
                recommendations.append(f"Standardize data types in column {anomaly['column']}")
        
        return recommendations
    
    def perform_statistical_analysis(self, file_path: str) -> StatisticalAnalysis:
        """
        Perform statistical analysis on Excel data
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Statistical analysis results
        """
        logger.info(f"Performing statistical analysis for {file_path}")
        
        try:
            # Read Excel file as DataFrame
            excel_data = pd.read_excel(file_path, sheet_name=None)
            
            # Combine all sheets for analysis
            all_data = []
            for sheet_name, df in excel_data.items():
                df['_sheet_name'] = sheet_name
                all_data.append(df)
            
            if not all_data:
                raise ValueError("No data found in Excel file")
            
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove non-numeric columns for statistical analysis
            numeric_df = combined_df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                raise ValueError("No numeric data found for statistical analysis")
            
            # Descriptive statistics
            descriptive_stats = {
                'count': numeric_df.count().to_dict(),
                'mean': numeric_df.mean().to_dict(),
                'std': numeric_df.std().to_dict(),
                'min': numeric_df.min().to_dict(),
                'max': numeric_df.max().to_dict(),
                'median': numeric_df.median().to_dict()
            }
            
            # Correlations
            correlations = {}
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                for col1 in corr_matrix.columns:
                    for col2 in corr_matrix.columns:
                        if col1 != col2:
                            key = f"{col1}_vs_{col2}"
                            correlations[key] = float(corr_matrix.loc[col1, col2])
            
            # Detect patterns
            patterns = self._detect_patterns(numeric_df)
            
            # Detect outliers
            outliers = self._detect_outliers(numeric_df)
            
            # Detect trends
            trends = self._detect_trends(numeric_df)
            
            analysis = StatisticalAnalysis(
                file_path=file_path,
                analysis_timestamp=datetime.now(),
                descriptive_stats=descriptive_stats,
                correlations=correlations,
                patterns=patterns,
                outliers=outliers,
                trends=trends
            )
            
            logger.info(f"Statistical analysis completed. Found {len(patterns)} patterns, {len(outliers)} outliers")
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing statistical analysis: {e}")
            raise
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect patterns in numeric data"""
        patterns = []
        
        for column in df.columns:
            series = df[column].dropna()
            if len(series) < 3:
                continue
            
            # Check for linear patterns
            if len(series) > 5:
                # Simple linear trend detection
                x = np.arange(len(series))
                correlation = np.corrcoef(x, series)[0, 1]
                
                if abs(correlation) > 0.7:
                    patterns.append({
                        'type': 'linear_trend',
                        'column': column,
                        'correlation': float(correlation),
                        'direction': 'increasing' if correlation > 0 else 'decreasing',
                        'strength': 'strong' if abs(correlation) > 0.8 else 'moderate'
                    })
            
            # Check for periodicity (simple check)
            if len(series) > 10:
                # Check for repeating patterns
                autocorr = series.autocorr()
                if abs(autocorr) > 0.5:
                    patterns.append({
                        'type': 'periodic',
                        'column': column,
                        'autocorrelation': float(autocorr),
                        'description': 'Data shows periodic patterns'
                    })
            
            # Check for clustering
            if len(series) > 5:
                # Simple clustering detection using standard deviation
                mean_val = series.mean()
                std_val = series.std()
                within_1std = ((series >= mean_val - std_val) & (series <= mean_val + std_val)).sum()
                clustering_ratio = within_1std / len(series)
                
                if clustering_ratio > 0.8:
                    patterns.append({
                        'type': 'clustering',
                        'column': column,
                        'clustering_ratio': float(clustering_ratio),
                        'description': 'Data shows clustering around the mean'
                    })
        
        return patterns
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect outliers in numeric data"""
        outliers = []
        
        for column in df.columns:
            series = df[column].dropna()
            if len(series) < 3:
                continue
            
            # Use IQR method for outlier detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = series[(series < lower_bound) | (series > upper_bound)].index
            
            for idx in outlier_indices:
                outliers.append({
                    'column': column,
                    'index': int(idx),
                    'value': float(series[idx]),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'deviation': float(abs(series[idx] - series.median()) / series.std())
                })
        
        return outliers
    
    def _detect_trends(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect trends in numeric data"""
        trends = []
        
        for column in df.columns:
            series = df[column].dropna()
            if len(series) < 5:
                continue
            
            # Simple trend detection
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]
            
            if abs(slope) > series.std() * 0.1:  # Significant trend
                trends.append({
                    'column': column,
                    'slope': float(slope),
                    'direction': 'increasing' if slope > 0 else 'decreasing',
                    'magnitude': 'strong' if abs(slope) > series.std() * 0.2 else 'moderate',
                    'description': f"Data shows {'increasing' if slope > 0 else 'decreasing'} trend"
                })
        
        return trends
    
    def batch_analyze_project(self, project_id: str) -> Dict[str, Any]:
        """
        Perform batch analysis on all files in a project
        
        Args:
            project_id: Project ID to analyze
            
        Returns:
            Analysis results for all files
        """
        if project_id not in self.projects:
            raise ValueError(f"Project not found: {project_id}")
        
        project = self.projects[project_id]
        results = {
            'project_id': project_id,
            'project_name': project.name,
            'analysis_timestamp': datetime.now().isoformat(),
            'files_analyzed': 0,
            'quality_reports': [],
            'statistical_analyses': [],
            'overall_quality_score': 0.0,
            'summary': {}
        }
        
        logger.info(f"Starting batch analysis for project {project_id}: {len(project.file_paths)} files")
        
        # Analyze each file
        quality_scores = []
        
        for file_path in project.file_paths:
            try:
                # Data quality analysis
                quality_report = self.analyze_data_quality(file_path)
                results['quality_reports'].append(asdict(quality_report))
                quality_scores.append(quality_report.quality_score)
                
                # Statistical analysis
                try:
                    stat_analysis = self.perform_statistical_analysis(file_path)
                    results['statistical_analyses'].append(asdict(stat_analysis))
                except Exception as e:
                    logger.warning(f"Statistical analysis failed for {file_path}: {e}")
                
                results['files_analyzed'] += 1
                
            except Exception as e:
                logger.error(f"Analysis failed for {file_path}: {e}")
        
        # Calculate overall quality score
        if quality_scores:
            results['overall_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        # Generate summary
        results['summary'] = self._generate_analysis_summary(results)
        
        # Save analysis results
        analysis_file = self.analysis_path / f"analysis_{project_id}_{int(datetime.now().timestamp())}.json"
        with open(analysis_file, 'w') as f:
            json.dump(results, f, default=str, indent=2)
        
        logger.info(f"Batch analysis completed. Analyzed {results['files_analyzed']} files")
        return results
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of analysis results"""
        summary = {
            'total_files': results['files_analyzed'],
            'average_quality_score': results['overall_quality_score'],
            'total_anomalies': 0,
            'total_patterns': 0,
            'total_outliers': 0,
            'data_types_found': set(),
            'recommendations': []
        }
        
        # Aggregate statistics from quality reports
        for report in results['quality_reports']:
            summary['total_anomalies'] += len(report['anomalies'])
            summary['data_types_found'].update(report['data_types'].keys())
            summary['recommendations'].extend(report['recommendations'])
        
        # Aggregate statistics from statistical analyses
        for analysis in results['statistical_analyses']:
            summary['total_patterns'] += len(analysis['patterns'])
            summary['total_outliers'] += len(analysis['outliers'])
        
        # Remove duplicate recommendations
        summary['recommendations'] = list(set(summary['recommendations']))
        summary['data_types_found'] = list(summary['data_types_found'])
        
        return summary
    
    def export_project_report(self, project_id: str, output_path: str):
        """
        Export comprehensive project report
        
        Args:
            project_id: Project ID
            output_path: Output file path
        """
        if project_id not in self.projects:
            raise ValueError(f"Project not found: {project_id}")
        
        project = self.projects[project_id]
        
        # Get project analysis
        analysis_results = self.batch_analyze_project(project_id)
        
        # Get learning results
        learning_results = []
        all_tasks = self.background_processor.get_all_tasks()
        for task in all_tasks.values():
            if (task.metadata and 
                task.metadata.get('project_id') == project_id and 
                task.status == TaskStatus.COMPLETED):
                learning_results.append({
                    'task_id': task.task_id,
                    'file_path': task.file_path,
                    'task_type': task.task_type,
                    'processing_time': (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else 0,
                    'result_summary': task.result
                })
        
        # Create comprehensive report
        report = {
            'project_info': asdict(project),
            'analysis_results': analysis_results,
            'learning_results': learning_results,
            'export_timestamp': datetime.now().isoformat(),
            'system_info': {
                'version': '1.0.0',
                'extensions': 'research_extensions'
            }
        }
        
        # Export to file
        with open(output_path, 'w') as f:
            json.dump(report, f, default=str, indent=2)
        
        logger.info(f"Project report exported to {output_path}")
    
    def create_collaboration_package(self, project_id: str, output_path: str):
        """
        Create a collaboration package for sharing with other researchers
        
        Args:
            project_id: Project ID
            output_path: Output ZIP file path
        """
        if project_id not in self.projects:
            raise ValueError(f"Project not found: {project_id}")
        
        project = self.projects[project_id]
        
        # Create temporary directory for package
        temp_dir = Path("temp_collaboration_package")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Copy project files
            project_files_dir = temp_dir / "project_files"
            project_files_dir.mkdir()
            
            for file_path in project.file_paths:
                if os.path.exists(file_path):
                    dest_path = project_files_dir / os.path.basename(file_path)
                    shutil.copy2(file_path, dest_path)
            
            # Create project metadata
            metadata = {
                'project_info': asdict(project),
                'export_timestamp': datetime.now().isoformat(),
                'system_version': '1.0.0',
                'instructions': [
                    "This package contains Excel files and analysis results for collaborative research.",
                    "Use the AI Excel Learning System to process these files.",
                    "Contact the project owner for additional information."
                ]
            }
            
            with open(temp_dir / "project_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create README
            readme_content = f"""
# Research Project: {project.name}

## Description
{project.description}

## Files
{chr(10).join(f"- {os.path.basename(fp)}" for fp in project.file_paths)}

## Collaborators
{chr(10).join(f"- {collab}" for collab in project.collaborators)}

## Tags
{chr(10).join(f"- {tag}" for tag in project.tags)}

## Usage
1. Extract this package
2. Use the AI Excel Learning System to analyze the Excel files
3. Review the project_metadata.json for additional information

## Contact
For questions about this project, contact the project owner.
            """
            
            with open(temp_dir / "README.md", 'w') as f:
                f.write(readme_content)
            
            # Create ZIP file
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_dir)
                        zipf.write(file_path, arcname)
            
            logger.info(f"Collaboration package created: {output_path}")
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get comprehensive research statistics"""
        stats = {
            'total_projects': len(self.projects),
            'active_projects': len([p for p in self.projects.values() if p.status == 'active']),
            'total_files': sum(len(p.file_paths) for p in self.projects.values()),
            'total_collaborators': len(set(
                collab for p in self.projects.values() for collab in p.collaborators
            )),
            'projects_by_tag': {},
            'recent_activity': []
        }
        
        # Projects by tag
        for project in self.projects.values():
            for tag in project.tags:
                stats['projects_by_tag'][tag] = stats['projects_by_tag'].get(tag, 0) + 1
        
        # Recent activity (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        for project in self.projects.values():
            if project.updated_at > cutoff_date:
                stats['recent_activity'].append({
                    'project_id': project.project_id,
                    'project_name': project.name,
                    'updated_at': project.updated_at.isoformat()
                })
        
        # Background processing stats
        bg_stats = self.background_processor.get_statistics()
        stats.update({
            'background_processing': bg_stats,
            'system_uptime': 'active' if bg_stats['system_running'] else 'inactive'
        })
        
        return stats
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old research data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up old analysis files
        old_analysis_files = [
            f for f in self.analysis_path.glob("analysis_*.json")
            if f.stat().st_mtime < cutoff_date.timestamp()
        ]
        
        for file_path in old_analysis_files:
            try:
                file_path.unlink()
                logger.info(f"Deleted old analysis file: {file_path}")
            except Exception as e:
                logger.warning(f"Error deleting {file_path}: {e}")
        
        # Clean up background processor data
        self.background_processor.cleanup_old_data(days_to_keep)
        
        logger.info(f"Cleanup completed. Deleted {len(old_analysis_files)} old analysis files")
