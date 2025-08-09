# AI Excel Learning System

A comprehensive AI/ML system for learning from Excel files and automatically generating similar files with data, charts, and formatting.

## Overview

The AI Excel Learning System is designed to analyze Excel files, learn patterns in data, charts, and formatting, and then generate new Excel files that mimic the learned patterns. This system uses advanced machine learning techniques including TensorFlow, scikit-learn, and other AI libraries to understand and replicate Excel file structures.

## Features

### 🧠 **Intelligent Analysis**
- **Excel Structure Analysis**: Deep analysis of Excel files including data ranges, patterns, and relationships
- **Chart Pattern Recognition**: Learns chart types, configurations, and positioning
- **Formatting Analysis**: Understands cell formatting, styles, and layouts
- **Advanced Formula Learning**: Deep understanding of Excel formula logic, calculations, and filtering operations

### 🤖 **Machine Learning Models**
- **Neural Networks**: TensorFlow-based models for complex pattern learning
- **Traditional ML**: Random Forest, Linear Regression, and other algorithms
- **LSTM Models**: For sequential data and time series patterns
- **Pattern Recognition**: Automatic detection of data patterns (sequential, random, formula-based)

### 📊 **Chart Learning & Generation**
- **Chart Type Detection**: Scatter plots, line charts, bar charts, pie charts
- **Visual Configuration**: Learns colors, markers, axis settings, and positioning
- **Template Creation**: Creates reusable chart templates
- **Automatic Generation**: Generates charts based on learned patterns
- **Formula Integration**: Creates charts that incorporate learned formula logic and filtering
- **Dynamic Visualization**: Generates diagrams that reflect calculations and data transformations

### 🔄 **Data Generation**
- **Synthetic Data**: Generates realistic data based on learned patterns
- **Time Series**: Trend, seasonal, and cyclical data generation
- **Correlated Data**: Maintains relationships between variables
- **Categorical Data**: Generates text and categorical data with proper distributions

### 🧮 **Advanced Formula Learning**
- **Formula Logic Understanding**: Learns the underlying logic of Excel formulas (SUM, AVERAGE, IF, VLOOKUP, etc.)
- **Calculation Pattern Recognition**: Identifies and learns calculation patterns and their parameters
- **Filtering Operations**: Learns filtering conditions and can apply them to new data
- **Conditional Logic**: Understands IF statements and conditional operations
- **Range Analysis**: Learns how to work with cell ranges and references
- **Formula Application**: Can apply learned formula logic to new data or different contexts
- **Chart Integration**: Creates diagrams that reflect learned calculations and filtering operations

### 📁 **Excel File Generation**
- **Complete Files**: Generates full Excel files with data, charts, and formatting
- **Multiple Sheets**: Supports multiple worksheets
- **Professional Formatting**: Applies learned formatting styles
- **Chart Integration**: Embeds charts with proper positioning

### 🚀 **Model Management**
- **Version Control**: Tracks model versions and performance
- **Deployment Management**: Manages model deployments across environments
- **Performance Monitoring**: Tracks model performance over time
- **Rollback Capability**: Easy rollback to previous model versions

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Excel Files   │───▶│  Excel Analyzer │───▶│  Learning       │
│   (Input)       │    │                 │    │  Pipeline       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Chart Learner  │    │   ML Models     │    │  Data Generator │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Model Manager   │    │ Excel Generator │───▶│  Excel Files    │
│                 │    │                 │    │  (Output)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Dependencies
The system requires the following AI/ML libraries:

```bash
# Core ML/AI libraries
tensorflow==2.15.0
scikit-learn==1.4.0
torch==2.1.2
transformers==4.36.2

# Data processing
pandas==2.3.0
numpy==2.3.1
openpyxl==3.1.5

# Visualization
matplotlib==3.10.3
seaborn==0.13.0
plotly==5.17.0

# Additional ML libraries
xgboost==2.0.3
lightgbm==4.1.0
catboost==1.2.2

# Utilities
joblib==1.3.2
tqdm==4.66.1
h5py==3.10.0
```

### Quick Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the example
python -m src.ai_excel_learning.example_usage
```

## Usage

### Basic Usage

```python
from src.ai_excel_learning import LearningPipeline, PipelineConfig

# Initialize the pipeline
config = PipelineConfig(
    models_dir="models",
    templates_dir="templates",
    output_dir="output",
    analysis_dir="analysis"
)
pipeline = LearningPipeline(config)

# Start a learning session
session_id = pipeline.start_learning_session("my_session")

# Learn from Excel files
excel_files = ["file1.xlsx", "file2.xlsx", "file3.xlsx"]
results = pipeline.learn_from_excel_files(excel_files, session_id)

# Generate a new Excel file
output_file = pipeline.generate_excel_file(
    session_id=session_id,
    output_filename="generated_file.xlsx",
    num_rows=100,
    num_columns=5
)
```

### Advanced Usage

#### Individual Components

```python
from src.ai_excel_learning import ExcelAnalyzer, ExcelMLModels, ChartLearner

# Analyze Excel files
analyzer = ExcelAnalyzer()
structure = analyzer.analyze_excel_file("input.xlsx")

# Train ML models
ml_models = ExcelMLModels()
training_data = pd.DataFrame(...)
result = ml_models.train_numeric_model(
    training_data, 'target', 'my_model', 'neural_network'
)

# Learn chart patterns
chart_learner = ChartLearner()
patterns = chart_learner.learn_chart_patterns(structure.__dict__)
template = chart_learner.create_chart_template(patterns[0], "my_template")
```

#### Data Generation

```python
from src.ai_excel_learning import DataGenerator

# Generate time series data
data_generator = DataGenerator()
trend_data = data_generator.generate_time_series_data(
    100, 'trend', start_value=0, end_value=100
)

# Generate correlated data
correlation_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
correlated_data = data_generator.generate_correlated_data(
    1000, correlation_matrix, ['X', 'Y']
)
```

#### Model Management

```python
from src.ai_excel_learning import ModelManager

# Manage model versions
model_manager = ModelManager()
version_id = model_manager.create_model_version(
    model_name="my_model",
    model_type="neural_network",
    model_path="model.h5",
    training_data=training_data,
    performance_metrics={'accuracy': 0.95}
)

# Deploy model
deployment_id = model_manager.deploy_model(version_id, "production")
```

## Configuration

### Pipeline Configuration

```python
config = PipelineConfig(
    models_dir="models",              # Directory for ML models
    templates_dir="templates",        # Directory for chart templates
    output_dir="output",             # Directory for generated files
    analysis_dir="analysis",         # Directory for analysis results
    enable_charts=True,              # Enable chart learning
    enable_formatting=True,          # Enable formatting learning
    enable_formulas=True,            # Enable formula learning
    model_types=['neural_network', 'random_forest'],  # Allowed model types
    chart_types=['scatter', 'line', 'bar']           # Supported chart types
)
```

### Generation Configuration

```python
gen_config = GenerationConfig(
    num_rows=100,                    # Number of rows to generate
    num_columns=5,                   # Number of columns to generate
    data_types=['numeric', 'categorical'],  # Data types to include
    patterns=['random', 'sequential'],      # Pattern types
    constraints={},                  # Data constraints
    relationships=[]                 # Column relationships
)
```

## Examples

### Example 1: Learning from Financial Data

```python
# Learn from financial Excel files
financial_files = [
    "revenue_2022.xlsx",
    "revenue_2023.xlsx", 
    "revenue_2024.xlsx"
]

results = pipeline.learn_from_excel_files(financial_files, "financial_session")

# Generate similar financial data
output_file = pipeline.generate_excel_file(
    session_id="financial_session",
    output_filename="projected_revenue.xlsx",
    num_rows=365,  # Daily data for a year
    num_columns=6  # Multiple revenue streams
)
```

### Example 2: Chart Pattern Learning

```python
# Analyze charts in existing files
analyzer = ExcelAnalyzer()
structure = analyzer.analyze_excel_file("sales_data.xlsx")

# Learn chart patterns
chart_learner = ChartLearner()
patterns = chart_learner.learn_chart_patterns(structure.__dict__)

# Create templates for different chart types
for i, pattern in enumerate(patterns):
    template_name = f"sales_chart_{i+1}"
    template = chart_learner.create_chart_template(pattern, template_name)
    
    # Generate chart with new data
    new_data = pd.DataFrame(...)
    chart_config = chart_learner.generate_chart(template_name, new_data)
```

### Example 3: Advanced Data Generation

```python
# Generate complex correlated data
data_generator = DataGenerator()

# Create correlation matrix for sales data
correlation_matrix = np.array([
    [1.0, 0.8, 0.6, 0.4],  # Product A
    [0.8, 1.0, 0.7, 0.5],  # Product B  
    [0.6, 0.7, 1.0, 0.8],  # Product C
    [0.4, 0.5, 0.8, 1.0]   # Product D
])

# Generate correlated sales data
sales_data = data_generator.generate_correlated_data(
    1000, correlation_matrix, 
    ['Product_A', 'Product_B', 'Product_C', 'Product_D']
)

# Add time series trend
trend_data = data_generator.generate_time_series_data(
    1000, 'trend', start_value=100, end_value=200
)

# Combine data
final_data = pd.concat([sales_data, trend_data], axis=1)
```

### Example 4: Advanced Formula Learning

```python
# Initialize the learning pipeline
pipeline = LearningPipeline()

# Learn formula logic from sample formulas
sample_formulas = [
    "=SUM(A1:A10)",
    "=AVERAGE(B2:B50)", 
    "=IF(C1>250, 'High', 'Low')",
    "=FILTER(A1:C100, B1:B100>250)",
    "=A1*B1+C1"
]

# Demonstrate formula learning
results = pipeline.demonstrate_formula_learning(sample_formulas)

# Learn filtering operations
sample_data = pd.DataFrame({
    'Current_mA': np.random.uniform(100, 300, 100),
    'Voltage_V': np.random.uniform(3.0, 5.0, 100),
    'Temperature_C': np.random.uniform(20, 80, 100),
    'Status': np.random.choice(['OK', 'Warning', 'Error'], 100)
})

filter_conditions = [
    {'column': 'Current_mA', 'operator': '>', 'value': 250},
    {'column': 'Temperature_C', 'operator': '>', 'value': 60},
    {'column': 'Status', 'operator': '==', 'value': 'Warning'}
]

# Learn and apply filtering operations
learned_filters = pipeline.learn_filtering_operations(sample_data, filter_conditions)
filtered_data = pipeline.apply_learned_filters(sample_data, learned_filters)

print(f"Original data: {len(sample_data)} rows")
print(f"Filtered data: {len(filtered_data)} rows")

# Apply learned formula logic to new data
formula_logic = pipeline.learn_formula_logic("=SUM(A1:A10)")
new_data = pd.DataFrame({'A': range(1, 21)})
result = pipeline.apply_learned_formula_logic(formula_logic, new_data)
print(f"Sum result: {result}")
```

### Example 5: Chart Generation with Formula Logic

```python
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

# Learn formula logic and create chart with it
sum_formula = pipeline.learn_formula_logic("=SUM(A1:A10)")
chart_config = pipeline.create_chart_with_formula_logic(
    "sum_chart", 
    sample_data, 
    formula_logic=sum_formula
)

# Create chart with filtering (values above 250 mA)
filter_conditions = [FilterCondition(column='Current_mA', operator='>', value=250)]
filtered_chart = pipeline.create_chart_with_formula_logic(
    "filtered_chart", 
    sample_data, 
    filter_conditions=filter_conditions
)

# Demonstrate comprehensive chart generation
results = pipeline.demonstrate_chart_with_formulas(sample_data)
print(f"Created {len(results['charts_created'])} charts with formula logic")
```

## Performance

### Learning Performance
- **File Processing**: ~2-5 seconds per Excel file (depending on size and complexity)
- **Model Training**: ~30-120 seconds per model (depending on data size and model type)
- **Chart Analysis**: ~1-3 seconds per chart
- **Template Creation**: ~1-2 seconds per template

### Generation Performance
- **Data Generation**: ~0.1-1 second per 1000 rows
- **Excel File Creation**: ~2-10 seconds per file (depending on complexity)
- **Chart Generation**: ~1-3 seconds per chart

### Memory Usage
- **Analysis Phase**: ~50-200 MB per Excel file
- **Training Phase**: ~100-500 MB per model
- **Generation Phase**: ~20-100 MB per generated file

## Best Practices

### 1. **Data Quality**
- Ensure input Excel files have consistent structure
- Clean and validate data before learning
- Use representative samples for training

### 2. **Model Selection**
- Use neural networks for complex patterns
- Use traditional ML for simpler relationships
- Consider ensemble methods for better accuracy

### 3. **Chart Learning**
- Provide diverse chart examples for better learning
- Use consistent chart configurations
- Validate chart templates before deployment

### 4. **Performance Optimization**
- Use appropriate batch sizes for model training
- Implement caching for frequently used patterns
- Monitor memory usage during large-scale operations

### 5. **Version Control**
- Version your models and templates
- Track performance metrics over time
- Implement rollback strategies

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size in model training
   - Process files in smaller batches
   - Increase system memory

2. **Model Training Failures**
   - Check data quality and consistency
   - Verify feature scaling
   - Ensure sufficient training data

3. **Chart Generation Issues**
   - Validate chart templates
   - Check data compatibility
   - Verify chart library dependencies

4. **Excel File Corruption**
   - Validate input file format
   - Check file permissions
   - Ensure sufficient disk space

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug information
pipeline = LearningPipeline(config)
```

## Contributing

### Development Setup

1. Clone the repository
2. Install development dependencies
3. Run tests
4. Follow coding standards

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run performance tests
python -m pytest tests/performance/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review example usage

## Roadmap

### Future Features
- **Advanced Chart Types**: 3D charts, radar charts, heatmaps
- **Formula Learning**: Advanced Excel formula pattern recognition
- **Conditional Formatting**: Learn and apply conditional formatting rules
- **Macro Learning**: VBA macro pattern recognition and generation
- **Real-time Learning**: Continuous learning from new data
- **Cloud Integration**: AWS, Azure, and Google Cloud support
- **API Interface**: RESTful API for integration
- **Web Interface**: User-friendly web dashboard

### Performance Improvements
- **GPU Acceleration**: TensorFlow GPU support
- **Distributed Training**: Multi-node model training
- **Caching System**: Intelligent caching for faster processing
- **Streaming Processing**: Real-time Excel file processing

---

**Note**: This system is designed for educational and research purposes. Always validate generated data and files before using them in production environments.
