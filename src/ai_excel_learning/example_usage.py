"""
Example Usage of AI Excel Learning System

This script demonstrates how to use the complete AI Excel learning system
to analyze Excel files, train models, and generate new files.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

from .learning_pipeline import LearningPipeline, PipelineConfig
from .excel_analyzer import ExcelAnalyzer
from .ml_models import ExcelMLModels
from .chart_learner import ChartLearner
from .data_generator import DataGenerator, GenerationConfig
from .excel_generator import ExcelGenerator, ExcelGenerationConfig
from .model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_excel_file(output_path: str):
    """Create a sample Excel file for demonstration"""
    logger.info("Creating sample Excel file...")
    
    # Create sample data
    data = pd.DataFrame({
        'Index': range(1, 101),
        'Value_A': np.random.normal(50, 10, 100),
        'Value_B': np.random.normal(30, 5, 100),
        'Category': np.random.choice(['A', 'B', 'C'], 100),
        'Date': pd.date_range('2023-01-01', periods=100, freq='D')
    })
    
    # Add some relationships
    data['Value_C'] = data['Value_A'] * 0.7 + data['Value_B'] * 0.3 + np.random.normal(0, 2, 100)
    
    # Create Excel file with charts
    generator = ExcelGenerator()
    
    config = ExcelGenerationConfig(
        filename="sample_data",
        sheets=[{
            'name': 'Data',
            'start_row': 1,
            'start_col': 1,
            'include_headers': True,
            'formatting': {
                'columns': [
                    {'column': 'A', 'width': 10},
                    {'column': 'B', 'width': 15},
                    {'column': 'C', 'width': 15},
                    {'column': 'D', 'width': 15},
                    {'column': 'E', 'width': 15},
                    {'column': 'F', 'width': 15}
                ]
            }
        }],
        charts=[{
            'sheet': 'Data',
            'type': 'scatter',
            'title': 'Value A vs Value B',
            'position': {'anchor': 'H2'},
            'series': [{
                'title': 'Scatter Plot',
                'x_values': 'B2:B101',
                'y_values': 'C2:C101'
            }]
        }],
        formatting={
            'default_styles': {
                'header_style': {'bold': True, 'size': 12},
                'data_style': {'size': 11}
            }
        },
        metadata={
            'properties': {
                'title': 'Sample Data for AI Learning',
                'creator': 'DataChaEnhanced AI',
                'created': datetime.now()
            }
        }
    )
    
    generator.generate_excel_file(data, config, output_path)
    logger.info(f"Sample Excel file created: {output_path}")

def demonstrate_learning_pipeline():
    """Demonstrate the complete learning pipeline"""
    logger.info("=== AI Excel Learning Pipeline Demonstration ===")
    
    # Create sample Excel files
    sample_files = []
    for i in range(3):
        sample_path = f"sample_data_{i+1}.xlsx"
        create_sample_excel_file(sample_path)
        sample_files.append(sample_path)
    
    # Initialize pipeline
    config = PipelineConfig(
        models_dir="ai_models",
        templates_dir="ai_templates", 
        output_dir="ai_output",
        analysis_dir="ai_analysis",
        enable_charts=True,
        enable_formatting=True,
        enable_formulas=True
    )
    
    pipeline = LearningPipeline(config)
    
    # Start learning session
    session_id = pipeline.start_learning_session("demo_session")
    logger.info(f"Started learning session: {session_id}")
    
    # Learn from Excel files
    logger.info("Learning from Excel files...")
    results = pipeline.learn_from_excel_files(sample_files, session_id)
    
    # Print results
    logger.info("Learning Results:")
    logger.info(f"  Files processed: {len(results['files_processed'])}")
    logger.info(f"  Models trained: {len(results['models_trained'])}")
    logger.info(f"  Templates created: {len(results['templates_created'])}")
    logger.info(f"  Processing time: {results['performance_metrics']['processing_time_seconds']:.2f} seconds")
    
    # Generate new Excel file
    logger.info("Generating new Excel file...")
    output_file = pipeline.generate_excel_file(
        session_id=session_id,
        output_filename="ai_generated_file.xlsx",
        num_rows=50,
        num_columns=4
    )
    logger.info(f"Generated Excel file: {output_file}")
    
    # Get session summary
    summary = pipeline.get_session_summary(session_id)
    logger.info("Session Summary:")
    logger.info(json.dumps(summary, indent=2))
    
    return pipeline, session_id

def demonstrate_individual_components():
    """Demonstrate individual components"""
    logger.info("=== Individual Components Demonstration ===")
    
    # 1. Excel Analyzer
    logger.info("1. Testing Excel Analyzer...")
    analyzer = ExcelAnalyzer()
    
    # Create a sample file for analysis
    sample_path = "test_analysis.xlsx"
    create_sample_excel_file(sample_path)
    
    # Analyze the file
    structure = analyzer.analyze_excel_file(sample_path)
    logger.info(f"  Analyzed file with {len(structure.sheet_names)} sheets")
    logger.info(f"  Found {sum(len(charts) for charts in structure.charts.values())} charts")
    
    # 2. ML Models
    logger.info("2. Testing ML Models...")
    ml_models = ExcelMLModels("test_models")
    
    # Create sample training data
    training_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'target': np.random.normal(0, 1, 100)
    })
    
    # Train a model
    result = ml_models.train_numeric_model(
        training_data, 'target', 'test_model', 'neural_network'
    )
    logger.info(f"  Trained model with RMSE: {result['rmse']:.4f}")
    
    # Generate data
    generated_data = ml_models.generate_data('test_model', 20)
    logger.info(f"  Generated {len(generated_data)} rows of data")
    
    # 3. Chart Learner
    logger.info("3. Testing Chart Learner...")
    chart_learner = ChartLearner("test_charts")
    
    # Learn chart patterns
    chart_patterns = chart_learner.learn_chart_patterns(structure.__dict__)
    logger.info(f"  Learned {len(chart_patterns)} chart patterns")
    
    # Create template
    if chart_patterns:
        template = chart_learner.create_chart_template(chart_patterns[0], "test_template")
        logger.info(f"  Created chart template: {template.name}")
    
    # 4. Data Generator
    logger.info("4. Testing Data Generator...")
    data_generator = DataGenerator()
    
    # Create generation config
    gen_config = GenerationConfig(
        num_rows=100,
        num_columns=5,
        data_types=['numeric', 'categorical'],
        patterns=['random', 'sequential'],
        constraints={},
        relationships=[]
    )
    
    # Generate data
    patterns = [
        {
            'pattern_type': 'random',
            'data_type': 'numeric',
            'statistics': {'mean': 50, 'std': 10, 'min': 0, 'max': 100}
        },
        {
            'pattern_type': 'sequential',
            'data_type': 'numeric',
            'statistics': {'min': 1, 'step': 1}
        }
    ]
    
    generated_data = data_generator.generate_data_from_patterns(patterns, gen_config)
    logger.info(f"  Generated data shape: {generated_data.shape}")
    
    # 5. Excel Generator
    logger.info("5. Testing Excel Generator...")
    excel_generator = ExcelGenerator()
    
    # Create generation config
    excel_config = ExcelGenerationConfig(
        filename="test_generated",
        sheets=[{
            'name': 'Generated Data',
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
                'title': 'Test Generated File',
                'creator': 'DataChaEnhanced AI'
            }
        }
    )
    
    # Generate Excel file
    output_path = excel_generator.generate_excel_file(generated_data, excel_config, "test_generated.xlsx")
    logger.info(f"  Generated Excel file: {output_path}")
    
    # 6. Model Manager
    logger.info("6. Testing Model Manager...")
    model_manager = ModelManager("test_model_versions")
    
    # Create model version
    version_id = model_manager.create_model_version(
        model_name="test_model",
        model_type="neural_network",
        model_path="test_models/test_model.h5",
        training_data=training_data,
        performance_metrics={'rmse': result['rmse']},
        metadata={'description': 'Test model for demonstration'}
    )
    logger.info(f"  Created model version: {version_id}")
    
    # Deploy model
    deployment_id = model_manager.deploy_model(version_id, "test")
    logger.info(f"  Deployed model: {deployment_id}")
    
    # Get statistics
    stats = model_manager.get_model_statistics()
    logger.info(f"  Model statistics: {stats}")

def demonstrate_advanced_features():
    """Demonstrate advanced features"""
    logger.info("=== Advanced Features Demonstration ===")
    
    # 1. Time Series Data Generation
    logger.info("1. Time Series Data Generation...")
    data_generator = DataGenerator()
    
    # Generate different types of time series
    trend_data = data_generator.generate_time_series_data(
        100, 'trend', start_value=0, end_value=100, noise_std=2
    )
    seasonal_data = data_generator.generate_time_series_data(
        365, 'seasonal', start_date='2023-01-01', noise_std=1
    )
    
    logger.info(f"  Generated trend data: {trend_data.shape}")
    logger.info(f"  Generated seasonal data: {seasonal_data.shape}")
    
    # 2. Correlated Data Generation
    logger.info("2. Correlated Data Generation...")
    
    # Create correlation matrix
    correlation_matrix = np.array([
        [1.0, 0.8, 0.3],
        [0.8, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])
    
    correlated_data = data_generator.generate_correlated_data(
        1000, correlation_matrix, ['X', 'Y', 'Z']
    )
    logger.info(f"  Generated correlated data: {correlated_data.shape}")
    
    # 3. Clustered Data Generation
    logger.info("3. Clustered Data Generation...")
    
    clustered_data = data_generator.generate_clustered_data(
        300, num_clusters=3
    )
    logger.info(f"  Generated clustered data: {clustered_data.shape}")
    
    # 4. Text Data Generation
    logger.info("4. Text Data Generation...")
    
    text_patterns = [
        {
            'column_name': 'names',
            'pattern_type': 'names',
            'names': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
        },
        {
            'column_name': 'emails',
            'pattern_type': 'emails',
            'domains': ['gmail.com', 'yahoo.com', 'hotmail.com']
        }
    ]
    
    text_data = data_generator.generate_text_data(50, text_patterns)
    logger.info(f"  Generated text data: {text_data.shape}")


def demonstrate_formula_learning():
    """Demonstrate formula learning capabilities"""
    logger.info("=== Formula Learning Demonstration ===")
    
    # Initialize the learning pipeline
    pipeline = LearningPipeline()
    
    # Sample formulas to learn from
    sample_formulas = [
        "=SUM(A1:A10)",
        "=AVERAGE(B2:B50)", 
        "=IF(C1>250, 'High', 'Low')",
        "=FILTER(A1:C100, B1:B100>250)",
        "=A1*B1+C1",
        "=COUNTIF(D1:D50, '>100')",
        "=VLOOKUP(E1, A1:B100, 2, FALSE)"
    ]
    
    # Demonstrate formula learning
    logger.info("Learning formula logic...")
    results = pipeline.demonstrate_formula_learning(sample_formulas)
    
    # Display results
    logger.info("Formula Learning Results:")
    for formula, pattern in results['learned_patterns'].items():
        if 'error' not in pattern:
            logger.info(f"  {formula}:")
            logger.info(f"    Type: {pattern['formula_type']}")
            logger.info(f"    Operation: {pattern['operation']}")
            logger.info(f"    Source Range: {pattern['source_range']}")
            logger.info(f"    Confidence: {pattern['confidence']}")
            
            if formula in results['demonstrations']:
                demo = results['demonstrations'][formula]
                logger.info(f"    Result Type: {demo['result_type']}")
                logger.info(f"    Result Preview: {demo['result_preview']}")
        else:
            logger.warning(f"  {formula}: Error - {pattern['error']}")
    
    # Demonstrate filtering operations
    logger.info("\nDemonstrating filtering operations...")
    sample_data = pd.DataFrame({
        'Current_mA': np.random.uniform(100, 300, 100),
        'Voltage_V': np.random.uniform(3.0, 5.0, 100),
        'Temperature_C': np.random.uniform(20, 80, 100),
        'Status': np.random.choice(['OK', 'Warning', 'Error'], 100)
    })
    
    # Learn filtering operations
    filter_conditions = [
        {'column': 'Current_mA', 'operator': '>', 'value': 250},
        {'column': 'Temperature_C', 'operator': '>', 'value': 60},
        {'column': 'Status', 'operator': '==', 'value': 'Warning'}
    ]
    
    learned_filters = pipeline.learn_filtering_operations(sample_data, filter_conditions)
    logger.info(f"Learned {len(learned_filters)} filter patterns")
    
    # Apply learned filters
    filtered_data = pipeline.apply_learned_filters(sample_data, learned_filters)
    logger.info(f"Applied filters: {len(sample_data)} -> {len(filtered_data)} rows")
    
    # Show some filtered results
    if len(filtered_data) > 0:
        logger.info("Sample filtered results:")
        logger.info(filtered_data.head().to_string())
    
    return results


def main():
    """Main demonstration function"""
    logger.info("Starting AI Excel Learning System Demonstration")
    
    try:
        # Demonstrate individual components
        demonstrate_individual_components()
        
        # Demonstrate advanced features
        demonstrate_advanced_features()
        
        # Demonstrate formula learning capabilities
        formula_results = demonstrate_formula_learning()
        
        # Demonstrate complete pipeline
        pipeline, session_id = demonstrate_learning_pipeline()
        
        # Save pipeline state
        pipeline.save_pipeline_state("pipeline_state.json")
        logger.info("Pipeline state saved")
        
        # Get performance statistics
        stats = pipeline.get_performance_statistics()
        logger.info("Overall Performance Statistics:")
        logger.info(json.dumps(stats, indent=2))
        
        logger.info("Demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}")
        raise

if __name__ == "__main__":
    main()
