"""
Chart Learner for Excel AI

This module learns chart patterns and configurations from Excel files
to enable automatic chart generation in similar files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import openpyxl
from openpyxl.chart import ScatterChart, LineChart, BarChart, PieChart
from openpyxl.chart.series import DataPoint
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

@dataclass
class ChartPattern:
    """Represents a learned chart pattern"""
    chart_type: str
    data_patterns: Dict[str, Any]
    visual_config: Dict[str, Any]
    positioning: Dict[str, Any]
    series_config: List[Dict[str, Any]]

@dataclass
class ChartTemplate:
    """Represents a chart template for generation"""
    name: str
    chart_type: str
    template_config: Dict[str, Any]
    data_requirements: Dict[str, Any]

class ChartLearner:
    """
    Learns chart patterns and configurations from Excel files
    """
    
    def __init__(self, templates_dir: str = "chart_templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        
        self.chart_patterns = {}
        self.templates = {}
        self.chart_types = {
            'scatter': ScatterChart,
            'line': LineChart,
            'bar': BarChart,
            'column': BarChart,
            'pie': PieChart
        }
        
    def learn_chart_patterns(self, excel_structure: Dict[str, Any]) -> List[ChartPattern]:
        """
        Learn chart patterns from Excel structure
        
        Args:
            excel_structure: Structure from ExcelAnalyzer
            
        Returns:
            List of learned chart patterns
        """
        logger.info("Learning chart patterns from Excel structure")
        
        patterns = []
        
        for sheet_name, sheet_data in excel_structure['charts'].items():
            for chart_info in sheet_data:
                pattern = self._extract_chart_pattern(chart_info, excel_structure, sheet_name)
                if pattern:
                    patterns.append(pattern)
                    self.chart_patterns[f"{sheet_name}_{chart_info.get('type', 'unknown')}"] = pattern
        
        logger.info(f"Learned {len(patterns)} chart patterns")
        return patterns
    
    def _extract_chart_pattern(self, chart_info: Dict[str, Any], 
                              excel_structure: Dict[str, Any], 
                              sheet_name: str) -> Optional[ChartPattern]:
        """Extract pattern from a single chart"""
        try:
            chart_type = self._normalize_chart_type(chart_info.get('type', ''))
            
            # Extract data patterns
            data_patterns = self._extract_data_patterns(chart_info, excel_structure, sheet_name)
            
            # Extract visual configuration
            visual_config = self._extract_visual_config(chart_info)
            
            # Extract positioning
            positioning = self._extract_positioning(chart_info)
            
            # Extract series configuration
            series_config = self._extract_series_config(chart_info)
            
            pattern = ChartPattern(
                chart_type=chart_type,
                data_patterns=data_patterns,
                visual_config=visual_config,
                positioning=positioning,
                series_config=series_config
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error extracting chart pattern: {str(e)}")
            return None
    
    def _normalize_chart_type(self, chart_type: str) -> str:
        """Normalize chart type names"""
        chart_type_lower = chart_type.lower()
        
        if 'scatter' in chart_type_lower:
            return 'scatter'
        elif 'line' in chart_type_lower:
            return 'line'
        elif 'bar' in chart_type_lower:
            return 'bar'
        elif 'column' in chart_type_lower:
            return 'column'
        elif 'pie' in chart_type_lower:
            return 'pie'
        else:
            return 'unknown'
    
    def _extract_data_patterns(self, chart_info: Dict[str, Any], 
                              excel_structure: Dict[str, Any], 
                              sheet_name: str) -> Dict[str, Any]:
        """Extract data patterns used in the chart"""
        data_patterns = {
            'series_count': len(chart_info.get('series', [])),
            'data_ranges': [],
            'data_types': [],
            'correlations': []
        }
        
        # Analyze series data
        for series in chart_info.get('series', []):
            series_pattern = self._analyze_series_data(series, excel_structure, sheet_name)
            data_patterns['data_ranges'].append(series_pattern)
            
            # Determine data types
            if series_pattern.get('x_data') is not None:
                data_patterns['data_types'].append(self._determine_data_type(series_pattern['x_data']))
            if series_pattern.get('y_data') is not None:
                data_patterns['data_types'].append(self._determine_data_type(series_pattern['y_data']))
        
        # Find correlations between series
        if len(data_patterns['data_ranges']) > 1:
            data_patterns['correlations'] = self._find_series_correlations(data_patterns['data_ranges'])
        
        return data_patterns
    
    def _analyze_series_data(self, series: Dict[str, Any], 
                           excel_structure: Dict[str, Any], 
                           sheet_name: str) -> Dict[str, Any]:
        """Analyze data for a single series"""
        series_pattern = {
            'name': series.get('name', ''),
            'x_data': None,
            'y_data': None,
            'statistics': {}
        }
        
        # Extract X and Y data if available
        if 'x_values' in series and series['x_values']:
            series_pattern['x_data'] = self._extract_data_from_range(series['x_values'], excel_structure, sheet_name)
        
        if 'y_values' in series and series['y_values']:
            series_pattern['y_data'] = self._extract_data_from_range(series['y_values'], excel_structure, sheet_name)
        
        # Calculate statistics
        if series_pattern['x_data'] is not None:
            series_pattern['statistics']['x_stats'] = self._calculate_data_statistics(series_pattern['x_data'])
        
        if series_pattern['y_data'] is not None:
            series_pattern['statistics']['y_stats'] = self._calculate_data_statistics(series_pattern['y_data'])
        
        return series_pattern
    
    def _extract_data_from_range(self, range_ref: str, 
                                excel_structure: Dict[str, Any], 
                                sheet_name: str) -> Optional[List[float]]:
        """Extract data from a range reference"""
        try:
            # This is a simplified implementation
            # In a real scenario, you'd parse the range reference and extract actual data
            # For now, we'll return None and handle it in the calling function
            return None
        except Exception as e:
            logger.warning(f"Could not extract data from range {range_ref}: {str(e)}")
            return None
    
    def _determine_data_type(self, data: List[float]) -> str:
        """Determine the type of data"""
        if not data:
            return 'unknown'
        
        # Check if sequential
        if len(data) > 1:
            diffs = np.diff(data)
            if np.std(diffs) < 0.1:
                return 'sequential'
        
        # Check if categorical (few unique values)
        unique_ratio = len(set(data)) / len(data)
        if unique_ratio < 0.3:
            return 'categorical'
        
        return 'continuous'
    
    def _calculate_data_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate statistics for data"""
        if not data:
            return {}
        
        data_array = np.array(data)
        return {
            'mean': float(np.mean(data_array)),
            'std': float(np.std(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'count': len(data_array)
        }
    
    def _find_series_correlations(self, data_ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find correlations between series"""
        correlations = []
        
        for i in range(len(data_ranges)):
            for j in range(i + 1, len(data_ranges)):
                if (data_ranges[i].get('y_data') and 
                    data_ranges[j].get('y_data') and
                    len(data_ranges[i]['y_data']) == len(data_ranges[j]['y_data'])):
                    
                    corr = np.corrcoef(data_ranges[i]['y_data'], data_ranges[j]['y_data'])[0, 1]
                    correlations.append({
                        'series1': i,
                        'series2': j,
                        'correlation': float(corr)
                    })
        
        return correlations
    
    def _extract_visual_config(self, chart_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract visual configuration from chart"""
        visual_config = {
            'title': chart_info.get('title', ''),
            'x_axis': self._extract_axis_config(chart_info.get('x_axis', {})),
            'y_axis': self._extract_axis_config(chart_info.get('y_axis', {})),
            'colors': [],
            'markers': [],
            'line_styles': []
        }
        
        # Extract series visual properties
        for series in chart_info.get('series', []):
            if 'marker' in series:
                visual_config['markers'].append(series['marker'])
            
            # Extract color information if available
            if 'color' in series:
                visual_config['colors'].append(series['color'])
        
        return visual_config
    
    def _extract_axis_config(self, axis_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract axis configuration"""
        return {
            'title': axis_info.get('title', ''),
            'min': axis_info.get('min'),
            'max': axis_info.get('max'),
            'major_unit': axis_info.get('major_unit'),
            'minor_unit': axis_info.get('minor_unit')
        }
    
    def _extract_positioning(self, chart_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract chart positioning information"""
        position = chart_info.get('position', {})
        return {
            'left': position.get('left'),
            'top': position.get('top'),
            'width': position.get('width'),
            'height': position.get('height')
        }
    
    def _extract_series_config(self, chart_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract series configuration"""
        series_config = []
        
        for series in chart_info.get('series', []):
            config = {
                'name': series.get('name', ''),
                'x_values_ref': series.get('x_values', ''),
                'y_values_ref': series.get('y_values', ''),
                'marker': series.get('marker', ''),
                'line_style': series.get('line_style', '')
            }
            series_config.append(config)
        
        return series_config
    
    def create_chart_template(self, pattern: ChartPattern, template_name: str) -> ChartTemplate:
        """
        Create a chart template from a learned pattern
        
        Args:
            pattern: Learned chart pattern
            template_name: Name for the template
            
        Returns:
            Chart template
        """
        template_config = {
            'chart_type': pattern.chart_type,
            'visual_config': pattern.visual_config,
            'positioning': pattern.positioning,
            'series_config': pattern.series_config,
            'data_requirements': self._extract_data_requirements(pattern)
        }
        
        template = ChartTemplate(
            name=template_name,
            chart_type=pattern.chart_type,
            template_config=template_config,
            data_requirements=template_config['data_requirements']
        )
        
        self.templates[template_name] = template
        self._save_template(template)
        
        return template
    
    def _extract_data_requirements(self, pattern: ChartPattern) -> Dict[str, Any]:
        """Extract data requirements from pattern"""
        requirements = {
            'min_series': pattern.data_patterns['series_count'],
            'data_types': list(set(pattern.data_patterns['data_types'])),
            'correlation_threshold': 0.7,
            'min_data_points': 5
        }
        
        # Analyze data ranges for requirements
        for data_range in pattern.data_patterns['data_ranges']:
            if data_range.get('statistics', {}).get('y_stats'):
                y_stats = data_range['statistics']['y_stats']
                requirements['data_range'] = {
                    'min': y_stats.get('min', 0),
                    'max': y_stats.get('max', 100),
                    'mean': y_stats.get('mean', 50)
                }
                break
        
        return requirements
    
    def generate_chart(self, template_name: str, data: pd.DataFrame, 
                      output_path: str = None) -> Dict[str, Any]:
        """
        Generate a chart using a template and data
        
        Args:
            template_name: Name of the template to use
            data: Data for the chart
            output_path: Optional path to save the chart
            
        Returns:
            Chart configuration
        """
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")
        
        template = self.templates[template_name]
        
        # Validate data against requirements
        self._validate_data_requirements(data, template.data_requirements)
        
        # Generate chart based on type
        if template.chart_type == 'scatter':
            chart_config = self._generate_scatter_chart(template, data)
        elif template.chart_type == 'line':
            chart_config = self._generate_line_chart(template, data)
        elif template.chart_type == 'bar':
            chart_config = self._generate_bar_chart(template, data)
        elif template.chart_type == 'pie':
            chart_config = self._generate_pie_chart(template, data)
        else:
            raise ValueError(f"Unsupported chart type: {template.chart_type}")
        
        # Save chart if output path provided
        if output_path:
            self._save_chart_image(chart_config, output_path)
        
        return chart_config
    
    def _validate_data_requirements(self, data: pd.DataFrame, requirements: Dict[str, Any]):
        """Validate data against template requirements"""
        if len(data.columns) < requirements['min_series']:
            raise ValueError(f"Data must have at least {requirements['min_series']} columns")
        
        if len(data) < requirements['min_data_points']:
            raise ValueError(f"Data must have at least {requirements['min_data_points']} rows")
    
    def _generate_scatter_chart(self, template: ChartTemplate, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate scatter chart"""
        chart_config = {
            'type': 'scatter',
            'data': [],
            'layout': {
                'title': template.template_config['visual_config']['title'],
                'xaxis': template.template_config['visual_config']['x_axis'],
                'yaxis': template.template_config['visual_config']['y_axis']
            }
        }
        
        # Add series data
        for i, series_config in enumerate(template.template_config['series_config']):
            if i < len(data.columns) - 1:  # Need at least 2 columns for x and y
                x_col = data.columns[i]
                y_col = data.columns[i + 1]
                
                series_data = {
                    'x': data[x_col].tolist(),
                    'y': data[y_col].tolist(),
                    'mode': 'markers',
                    'name': series_config.get('name', f'Series {i+1}'),
                    'marker': {'symbol': series_config.get('marker', 'circle')}
                }
                chart_config['data'].append(series_data)
        
        return chart_config
    
    def _generate_line_chart(self, template: ChartTemplate, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate line chart"""
        chart_config = {
            'type': 'line',
            'data': [],
            'layout': {
                'title': template.template_config['visual_config']['title'],
                'xaxis': template.template_config['visual_config']['x_axis'],
                'yaxis': template.template_config['visual_config']['y_axis']
            }
        }
        
        # Add series data
        for i, series_config in enumerate(template.template_config['series_config']):
            if i < len(data.columns):
                series_data = {
                    'x': data.index.tolist(),
                    'y': data.iloc[:, i].tolist(),
                    'mode': 'lines+markers',
                    'name': series_config.get('name', f'Series {i+1}'),
                    'line': {'style': series_config.get('line_style', 'solid')}
                }
                chart_config['data'].append(series_data)
        
        return chart_config
    
    def _generate_bar_chart(self, template: ChartTemplate, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate bar chart"""
        chart_config = {
            'type': 'bar',
            'data': [],
            'layout': {
                'title': template.template_config['visual_config']['title'],
                'xaxis': template.template_config['visual_config']['x_axis'],
                'yaxis': template.template_config['visual_config']['y_axis']
            }
        }
        
        # Add series data
        for i, series_config in enumerate(template.template_config['series_config']):
            if i < len(data.columns):
                series_data = {
                    'x': data.index.tolist(),
                    'y': data.iloc[:, i].tolist(),
                    'name': series_config.get('name', f'Series {i+1}'),
                    'type': 'bar'
                }
                chart_config['data'].append(series_data)
        
        return chart_config
    
    def _generate_pie_chart(self, template: ChartTemplate, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate pie chart"""
        chart_config = {
            'type': 'pie',
            'data': [{
                'labels': data.index.tolist(),
                'values': data.iloc[:, 0].tolist(),
                'type': 'pie',
                'name': template.template_config['series_config'][0].get('name', 'Pie Chart')
            }],
            'layout': {
                'title': template.template_config['visual_config']['title']
            }
        }
        
        return chart_config
    
    def _save_chart_image(self, chart_config: Dict[str, Any], output_path: str):
        """Save chart as image"""
        try:
            fig = go.Figure(data=chart_config['data'], layout=chart_config['layout'])
            fig.write_image(output_path)
            logger.info(f"Chart saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving chart: {str(e)}")
    
    def _save_template(self, template: ChartTemplate):
        """Save template to file"""
        template_path = self.templates_dir / f"{template.name}.json"
        
        template_data = {
            'name': template.name,
            'chart_type': template.chart_type,
            'template_config': template.template_config,
            'data_requirements': template.data_requirements
        }
        
        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        logger.info(f"Template saved: {template_path}")
    
    def load_template(self, template_name: str) -> ChartTemplate:
        """Load template from file"""
        template_path = self.templates_dir / f"{template_name}.json"
        
        if not template_path.exists():
            raise ValueError(f"Template {template_name} not found")
        
        with open(template_path, 'r') as f:
            template_data = json.load(f)
        
        template = ChartTemplate(**template_data)
        self.templates[template_name] = template
        
        return template
    
    def list_templates(self) -> List[str]:
        """List all available templates"""
        template_files = list(self.templates_dir.glob("*.json"))
        return [f.stem for f in template_files]
    
    def delete_template(self, template_name: str):
        """Delete a template"""
        template_path = self.templates_dir / f"{template_name}.json"
        
        if template_path.exists():
            template_path.unlink()
        
        if template_name in self.templates:
            del self.templates[template_name]
        
        logger.info(f"Template {template_name} deleted")
    
    def analyze_chart_similarity(self, chart1: ChartPattern, chart2: ChartPattern) -> float:
        """
        Analyze similarity between two chart patterns
        
        Args:
            chart1: First chart pattern
            chart2: Second chart pattern
            
        Returns:
            Similarity score (0-1)
        """
        similarity_score = 0.0
        
        # Compare chart types
        if chart1.chart_type == chart2.chart_type:
            similarity_score += 0.3
        
        # Compare data patterns
        data_similarity = self._compare_data_patterns(chart1.data_patterns, chart2.data_patterns)
        similarity_score += data_similarity * 0.4
        
        # Compare visual config
        visual_similarity = self._compare_visual_config(chart1.visual_config, chart2.visual_config)
        similarity_score += visual_similarity * 0.3
        
        return similarity_score
    
    def _compare_data_patterns(self, patterns1: Dict[str, Any], patterns2: Dict[str, Any]) -> float:
        """Compare data patterns"""
        if patterns1['series_count'] == patterns2['series_count']:
            return 1.0
        else:
            return 0.5
    
    def _compare_visual_config(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> float:
        """Compare visual configurations"""
        similarity = 0.0
        
        # Compare titles
        if config1.get('title') == config2.get('title'):
            similarity += 0.3
        
        # Compare axis configurations
        x_axis_similarity = self._compare_axis_config(config1.get('x_axis', {}), config2.get('x_axis', {}))
        y_axis_similarity = self._compare_axis_config(config1.get('y_axis', {}), config2.get('y_axis', {}))
        
        similarity += (x_axis_similarity + y_axis_similarity) * 0.35
        
        return similarity
    
    def _compare_axis_config(self, axis1: Dict[str, Any], axis2: Dict[str, Any]) -> float:
        """Compare axis configurations"""
        if axis1.get('title') == axis2.get('title'):
            return 1.0
        return 0.0
