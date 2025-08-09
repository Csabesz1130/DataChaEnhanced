"""
AI Excel Learning Module

This module provides advanced AI/ML capabilities for learning from Excel files
and generating similar files automatically. It includes:

- Excel file analysis and pattern recognition
- Machine learning models for data generation
- Chart and visualization learning
- Automated Excel file generation
- Continuous learning and model refinement
"""

from .excel_analyzer import ExcelAnalyzer
from .ml_models import ExcelMLModels
from .chart_learner import ChartLearner
from .data_generator import DataGenerator
from .excel_generator import ExcelGenerator
from .learning_pipeline import LearningPipeline
from .model_manager import ModelManager
from .formula_learner import FormulaLearner, FormulaLogic, FilterCondition, CalculationPattern, FormulaType

__version__ = "1.0.0"
__author__ = "DataChaEnhanced AI Team"

__all__ = [
    "ExcelAnalyzer",
    "ExcelMLModels",
    "ChartLearner",
    "DataGenerator",
    "ExcelGenerator",
    "LearningPipeline",
    "ModelManager",
    "FormulaLearner",
    "FormulaLogic",
    "FilterCondition",
    "CalculationPattern",
    "FormulaType"
]
