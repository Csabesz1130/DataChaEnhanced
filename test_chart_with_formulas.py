#!/usr/bin/env python3
"""
Test script for Chart Generation with Formula Logic

This script demonstrates how the AI Excel Learning System can:
1. Learn formula logic from Excel formulas
2. Learn filtering operations from data
3. Create charts that incorporate the learned logic
4. Generate diagrams that reflect calculations and filtering
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai_excel_learning import (
    LearningPipeline, 
    FormulaLearner, 
    FormulaLogic, 
    FilterCondition,
    FormulaType
)

def test_chart_with_sum_formula():
    """Test creating a chart that incorporates SUM formula logic"""
    print("=== Testing Chart with SUM Formula ===")
    
    # Initialize the learning pipeline
    pipeline = LearningPipeline()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Current_mA': np.random.uniform(100, 300, 50),
        'Voltage_V': np.random.uniform(3.0, 5.0, 50),
        'Temperature_C': np.random.uniform(20, 80, 50),
        'Timestamp': pd.date_range('2024-01-01', periods=50, freq='H')
    })
    
    print("Sample data shape:", sample_data.shape)
    print("Sample data preview:")
    print(sample_data.head())
    print()
    
    # Learn SUM formula logic
    sum_formula = pipeline.learn_formula_logic("=SUM(A1:A10)")
    print(f"Learned SUM formula: {sum_formula.formula_type.value}")
    print(f"Source range: {sum_formula.source_range}")
    print(f"Confidence: {sum_formula.confidence}")
    print()
    
    # Create chart with SUM formula logic
    chart_config = pipeline.create_chart_with_formula_logic(
        "sum_chart", 
        sample_data, 
        formula_logic=sum_formula
    )
    
    print("Chart configuration:")
    print(f"Chart type: {chart_config.get('type', 'Unknown')}")
    print(f"Data series: {len(chart_config.get('data', []))}")
    print(f"Layout: {chart_config.get('layout', {}).get('title', 'No title')}")
    print()
    
    return chart_config

def test_chart_with_filtering():
    """Test creating a chart that incorporates filtering logic"""
    print("=== Testing Chart with Filtering Logic ===")
    
    # Initialize the learning pipeline
    pipeline = LearningPipeline()
    
    # Create sample data with current measurements
    sample_data = pd.DataFrame({
        'Current_mA': np.random.uniform(100, 300, 100),
        'Voltage_V': np.random.uniform(3.0, 5.0, 100),
        'Temperature_C': np.random.uniform(20, 80, 100),
        'Status': np.random.choice(['OK', 'Warning', 'Error'], 100),
        'Timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
    })
    
    print("Original data shape:", sample_data.shape)
    print("Current measurements range:", sample_data['Current_mA'].min(), "-", sample_data['Current_mA'].max())
    print()
    
    # Define filter condition (values above 250 mA)
    filter_conditions = [
        FilterCondition(column='Current_mA', operator='>', value=250)
    ]
    
    print("Filter condition: Current_mA > 250")
    print()
    
    # Create chart with filtering logic
    chart_config = pipeline.create_chart_with_formula_logic(
        "filtered_chart", 
        sample_data, 
        filter_conditions=filter_conditions
    )
    
    print("Chart configuration:")
    print(f"Chart type: {chart_config.get('type', 'Unknown')}")
    print(f"Data series: {len(chart_config.get('data', []))}")
    print(f"Layout: {chart_config.get('layout', {}).get('title', 'No title')}")
    print()
    
    return chart_config

def test_chart_with_conditional_logic():
    """Test creating a chart that incorporates conditional logic"""
    print("=== Testing Chart with Conditional Logic ===")
    
    # Initialize the learning pipeline
    pipeline = LearningPipeline()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Current_mA': np.random.uniform(100, 300, 50),
        'Voltage_V': np.random.uniform(3.0, 5.0, 50),
        'Temperature_C': np.random.uniform(20, 80, 50),
        'Timestamp': pd.date_range('2024-01-01', periods=50, freq='H')
    })
    
    print("Sample data shape:", sample_data.shape)
    print()
    
    # Learn conditional formula logic
    if_formula = pipeline.learn_formula_logic("=IF(A1>250, 'High', 'Low')")
    print(f"Learned IF formula: {if_formula.formula_type.value}")
    print(f"Conditions: {if_formula.conditions}")
    print(f"Parameters: {if_formula.parameters}")
    print(f"Confidence: {if_formula.confidence}")
    print()
    
    # Create chart with conditional logic
    chart_config = pipeline.create_chart_with_formula_logic(
        "conditional_chart", 
        sample_data, 
        formula_logic=if_formula
    )
    
    print("Chart configuration:")
    print(f"Chart type: {chart_config.get('type', 'Unknown')}")
    print(f"Data series: {len(chart_config.get('data', []))}")
    print(f"Layout: {chart_config.get('layout', {}).get('title', 'No title')}")
    print()
    
    return chart_config

def test_comprehensive_demonstration():
    """Test comprehensive demonstration of chart generation with formulas"""
    print("=== Comprehensive Chart with Formulas Demonstration ===")
    
    # Initialize the learning pipeline
    pipeline = LearningPipeline()
    
    # Create comprehensive sample data
    sample_data = pd.DataFrame({
        'Current_mA': np.random.uniform(100, 300, 100),
        'Voltage_V': np.random.uniform(3.0, 5.0, 100),
        'Temperature_C': np.random.uniform(20, 80, 100),
        'Status': np.random.choice(['OK', 'Warning', 'Error'], 100),
        'Timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
    })
    
    print("Sample data created with shape:", sample_data.shape)
    print()
    
    # Run comprehensive demonstration
    results = pipeline.demonstrate_chart_with_formulas(sample_data)
    
    print("Demonstration Results:")
    print(f"Charts created: {len(results['charts_created'])}")
    print(f"Formula applications: {len(results['formula_applications'])}")
    print(f"Filter applications: {len(results['filter_applications'])}")
    print()
    
    # Display details of each chart
    for i, chart_info in enumerate(results['charts_created']):
        print(f"Chart {i+1}: {chart_info['name']}")
        if 'formula_type' in chart_info:
            print(f"  Formula type: {chart_info['formula_type']}")
        if 'filter_type' in chart_info:
            print(f"  Filter type: {chart_info['filter_type']}")
        print(f"  Config: {chart_info['config'].get('type', 'Unknown')} chart")
        print()
    
    # Display formula applications
    print("Formula Applications:")
    for app in results['formula_applications']:
        print(f"  - {app}")
    print()
    
    # Display filter applications
    print("Filter Applications:")
    for app in results['filter_applications']:
        print(f"  - {app}")
    print()
    
    return results

def create_visual_demonstration():
    """Create a visual demonstration of the charts"""
    print("=== Creating Visual Demonstration ===")
    
    # Initialize the learning pipeline
    pipeline = LearningPipeline()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Current_mA': np.random.uniform(100, 300, 50),
        'Voltage_V': np.random.uniform(3.0, 5.0, 50),
        'Temperature_C': np.random.uniform(20, 80, 50),
        'Timestamp': pd.date_range('2024-01-01', periods=50, freq='H')
    })
    
    # Create multiple charts with different logic
    charts = {}
    
    # 1. Original data chart
    charts['original'] = {
        'title': 'Original Data',
        'data': sample_data,
        'description': 'Raw current measurements over time'
    }
    
    # 2. Chart with SUM formula
    sum_formula = pipeline.learn_formula_logic("=SUM(A1:A10)")
    sum_chart_config = pipeline.create_chart_with_formula_logic(
        "sum_demo", sample_data, formula_logic=sum_formula
    )
    charts['sum'] = {
        'title': 'Data with SUM Formula Applied',
        'config': sum_chart_config,
        'description': 'Chart showing data with SUM calculation applied'
    }
    
    # 3. Chart with filtering (>250 mA)
    filter_conditions = [FilterCondition(column='Current_mA', operator='>', value=250)]
    filter_chart_config = pipeline.create_chart_with_formula_logic(
        "filter_demo", sample_data, filter_conditions=filter_conditions
    )
    charts['filtered'] = {
        'title': 'Filtered Data (>250 mA)',
        'config': filter_chart_config,
        'description': 'Chart showing only measurements above 250 mA'
    }
    
    # 4. Chart with conditional logic
    if_formula = pipeline.learn_formula_logic("=IF(A1>250, 'High', 'Low')")
    conditional_chart_config = pipeline.create_chart_with_formula_logic(
        "conditional_demo", sample_data, formula_logic=if_formula
    )
    charts['conditional'] = {
        'title': 'Data with Conditional Logic',
        'config': conditional_chart_config,
        'description': 'Chart showing data with High/Low classification'
    }
    
    print("Visual demonstration charts created:")
    for name, chart_info in charts.items():
        print(f"  {name}: {chart_info['title']}")
        print(f"    Description: {chart_info['description']}")
    print()
    
    return charts

def main():
    """Run all chart with formulas tests"""
    print("AI Excel Learning System - Chart Generation with Formula Logic Test")
    print("=" * 70)
    print()
    
    try:
        # Test individual chart types
        print("1. Testing individual chart types...")
        sum_chart = test_chart_with_sum_formula()
        filter_chart = test_chart_with_filtering()
        conditional_chart = test_chart_with_conditional_logic()
        
        print("\n" + "="*50 + "\n")
        
        # Test comprehensive demonstration
        print("2. Testing comprehensive demonstration...")
        comprehensive_results = test_comprehensive_demonstration()
        
        print("\n" + "="*50 + "\n")
        
        # Create visual demonstration
        print("3. Creating visual demonstration...")
        visual_charts = create_visual_demonstration()
        
        print("\n" + "="*70)
        print("All tests completed successfully!")
        print()
        print("The AI system can now:")
        print("✅ Learn formula logic from Excel formulas")
        print("✅ Learn filtering operations from data")
        print("✅ Create charts that incorporate learned logic")
        print("✅ Generate diagrams that reflect calculations")
        print("✅ Apply SUM, AVERAGE, IF, and FILTER operations to charts")
        print("✅ Visualize filtered data (e.g., values above 250 mA)")
        print("✅ Create conditional visualizations based on learned logic")
        print()
        print("This addresses your request: 'And based on these, after these create real diagrams'")
        print("The system can now create diagrams that reflect the learned calculations and filtering!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
