#!/usr/bin/env python3
"""
Quality Assurance System for AI Excel Learning

This module ensures high-quality learning by assessing data quality,
filtering low-quality inputs, and providing quality metrics for
continuous improvement.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for Excel data"""
    data_completeness: float      # 0.0 - 1.0
    data_consistency: float       # 0.0 - 1.0
    formula_complexity: float     # 0.0 - 1.0
    chart_quality: float          # 0.0 - 1.0
    noise_ratio: float            # 0.0 - 1.0
    overall_quality: float        # 0.0 - 1.0
    quality_score: str            # 'excellent', 'good', 'fair', 'poor'
    recommendations: List[str]

@dataclass
class QualityThresholds:
    """Configurable quality thresholds"""
    min_data_points: int = 100
    min_formula_complexity: float = 0.3
    min_chart_quality: float = 0.7
    max_noise_ratio: float = 0.15
    min_completeness: float = 0.8
    min_consistency: float = 0.7

class QualityAssurance:
    """
    Ensures high-quality learning by assessing and filtering data
    """
    
    def __init__(self, thresholds: QualityThresholds = None):
        self.thresholds = thresholds or QualityThresholds()
        self.quality_history = []
        
    def assess_excel_quality(self, excel_file: str, 
                           analysis_result: Dict[str, Any]) -> QualityMetrics:
        """
        Comprehensive quality assessment of Excel file
        
        Args:
            excel_file: Path to Excel file
            analysis_result: Result from ExcelAnalyzer
            
        Returns:
            QualityMetrics object with detailed quality information
        """
        logger.info(f"Assessing quality of {excel_file}")
        
        # Extract basic metrics
        data_completeness = self._assess_data_completeness(analysis_result)
        data_consistency = self._assess_data_consistency(analysis_result)
        formula_complexity = self._assess_formula_complexity(analysis_result)
        chart_quality = self._assess_chart_quality(analysis_result)
        noise_ratio = self._assess_noise_ratio(analysis_result)
        
        # Calculate overall quality
        weights = {
            'completeness': 0.25,
            'consistency': 0.25,
            'formula_complexity': 0.2,
            'chart_quality': 0.2,
            'noise_ratio': 0.1
        }
        
        overall_quality = (
            data_completeness * weights['completeness'] +
            data_consistency * weights['consistency'] +
            formula_complexity * weights['formula_complexity'] +
            chart_quality * weights['chart_quality'] +
            (1 - noise_ratio) * weights['noise_ratio']
        )
        
        # Determine quality score
        if overall_quality >= 0.9:
            quality_score = 'excellent'
        elif overall_quality >= 0.7:
            quality_score = 'good'
        elif overall_quality >= 0.5:
            quality_score = 'fair'
        else:
            quality_score = 'poor'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            data_completeness, data_consistency, formula_complexity,
            chart_quality, noise_ratio
        )
        
        metrics = QualityMetrics(
            data_completeness=data_completeness,
            data_consistency=data_consistency,
            formula_complexity=formula_complexity,
            chart_quality=chart_quality,
            noise_ratio=noise_ratio,
            overall_quality=overall_quality,
            quality_score=quality_score,
            recommendations=recommendations
        )
        
        # Store in history
        self.quality_history.append({
            'file': excel_file,
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics
        })
        
        return metrics
    
    def _assess_data_completeness(self, analysis_result: Dict[str, Any]) -> float:
        """Assess data completeness across sheets"""
        sheets = analysis_result.get('sheets', [])
        if not sheets:
            return 0.0
        
        total_cells = sum(sheet.get('cells_with_data', 0) for sheet in sheets)
        max_possible_cells = sum(
            sheet.get('max_row', 0) * sheet.get('max_column', 0) 
            for sheet in sheets
        )
        
        if max_possible_cells == 0:
            return 0.0
        
        completeness = total_cells / max_possible_cells
        return min(completeness, 1.0)
    
    def _assess_data_consistency(self, analysis_result: Dict[str, Any]) -> float:
        """Assess data consistency across sheets"""
        sheets = analysis_result.get('sheets', [])
        if len(sheets) < 2:
            return 1.0
        
        # Check column count consistency
        column_counts = [sheet.get('max_column', 0) for sheet in sheets]
        column_variance = np.var(column_counts) if len(column_counts) > 1 else 0
        
        # Check row count consistency
        row_counts = [sheet.get('max_row', 0) for sheet in sheets]
        row_variance = np.var(row_counts) if len(row_counts) > 1 else 0
        
        # Normalize variances (lower is better)
        max_expected_variance = 100  # Reasonable threshold
        column_consistency = max(0, 1 - (column_variance / max_expected_variance))
        row_consistency = max(0, 1 - (row_variance / max_expected_variance))
        
        return (column_consistency + row_consistency) / 2
    
    def _assess_formula_complexity(self, analysis_result: Dict[str, Any]) -> float:
        """Assess formula complexity and sophistication"""
        formulas = analysis_result.get('formulas', [])
        if not formulas:
            return 0.0
        
        # Analyze formula complexity
        complexity_scores = []
        for formula in formulas:
            formula_text = formula.get('formula', '')
            score = self._calculate_formula_complexity(formula_text)
            complexity_scores.append(score)
        
        if not complexity_scores:
            return 0.0
        
        return np.mean(complexity_scores)
    
    def _calculate_formula_complexity(self, formula: str) -> float:
        """Calculate complexity score for a single formula"""
        if not formula:
            return 0.0
        
        # Basic complexity indicators
        complexity_factors = {
            'functions': len([f for f in ['SUM', 'AVERAGE', 'IF', 'VLOOKUP', 'INDEX', 'MATCH'] 
                            if f in formula.upper()]),
            'operators': len([op for op in ['+', '-', '*', '/', '^', '&'] if op in formula]),
            'references': len([ref for ref in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] 
                             if ref in formula.upper()]),
            'parentheses': formula.count('(') + formula.count(')'),
            'length': len(formula)
        }
        
        # Weighted complexity score
        weights = {
            'functions': 0.3,
            'operators': 0.2,
            'references': 0.2,
            'parentheses': 0.15,
            'length': 0.15
        }
        
        total_score = sum(
            min(factor / 10, 1.0) * weight 
            for factor, weight in complexity_factors.items()
        )
        
        return min(total_score, 1.0)
    
    def _assess_chart_quality(self, analysis_result: Dict[str, Any]) -> float:
        """Assess chart quality and sophistication"""
        charts = analysis_result.get('charts', [])
        if not charts:
            return 0.0
        
        # Analyze chart quality
        quality_scores = []
        for chart in charts:
            score = self._calculate_chart_quality(chart)
            quality_scores.append(score)
        
        if not quality_scores:
            return 0.0
        
        return np.mean(quality_scores)
    
    def _calculate_chart_quality(self, chart: Dict[str, Any]) -> float:
        """Calculate quality score for a single chart"""
        # Basic chart quality indicators
        quality_factors = {
            'has_title': 0.2,
            'has_axis_labels': 0.2,
            'has_legend': 0.15,
            'has_grid': 0.1,
            'data_series_count': min(chart.get('series_count', 1) / 5, 1.0) * 0.2,
            'chart_type_sophistication': self._assess_chart_type_sophistication(chart) * 0.15
        }
        
        total_score = sum(quality_factors.values())
        return min(total_score, 1.0)
    
    def _assess_chart_type_sophistication(self, chart: Dict[str, Any]) -> float:
        """Assess sophistication of chart type"""
        chart_type = chart.get('chart_type', '').lower()
        
        sophistication_scores = {
            'line': 0.6,
            'bar': 0.5,
            'column': 0.5,
            'scatter': 0.8,
            'area': 0.7,
            'pie': 0.4,
            'doughnut': 0.6,
            'bubble': 0.9,
            'surface': 0.9,
            'radar': 0.8
        }
        
        return sophistication_scores.get(chart_type, 0.5)
    
    def _assess_noise_ratio(self, analysis_result: Dict[str, Any]) -> float:
        """Assess noise ratio in data"""
        sheets = analysis_result.get('sheets', [])
        if not sheets:
            return 0.0
        
        # Simple noise estimation based on data patterns
        total_cells = sum(sheet.get('cells_with_data', 0) for sheet in sheets)
        total_formulas = analysis_result.get('total_formulas', 0)
        
        # Higher formula ratio suggests more structured data (less noise)
        if total_cells == 0:
            return 0.0
        
        formula_ratio = total_formulas / total_cells
        # Inverse relationship: more formulas = less noise
        noise_ratio = max(0, 1 - formula_ratio * 2)
        
        return min(noise_ratio, 1.0)
    
    def _generate_recommendations(self, 
                                completeness: float,
                                consistency: float,
                                formula_complexity: float,
                                chart_quality: float,
                                noise_ratio: float) -> List[str]:
        """Generate improvement recommendations based on quality metrics"""
        recommendations = []
        
        if completeness < self.thresholds.min_completeness:
            recommendations.append(
                f"Adat kitöltöttség növelése: jelenleg {completeness:.1%}, "
                f"ajánlott: {self.thresholds.min_completeness:.1%}"
            )
        
        if consistency < self.thresholds.min_consistency:
            recommendations.append(
                f"Adat konzisztencia javítása: jelenleg {consistency:.1%}, "
                f"ajánlott: {self.thresholds.min_consistency:.1%}"
            )
        
        if formula_complexity < self.thresholds.min_formula_complexity:
            recommendations.append(
                f"Formulák komplexitásának növelése: jelenleg {formula_complexity:.1%}, "
                f"ajánlott: {self.thresholds.min_formula_complexity:.1%}"
            )
        
        if chart_quality < self.thresholds.min_chart_quality:
            recommendations.append(
                f"Diagramok minőségének javítása: jelenleg {chart_quality:.1%}, "
                f"ajánlott: {self.thresholds.min_chart_quality:.1%}"
            )
        
        if noise_ratio > self.thresholds.max_noise_ratio:
            recommendations.append(
                f"Zaj arány csökkentése: jelenleg {noise_ratio:.1%}, "
                f"ajánlott: <{self.thresholds.max_noise_ratio:.1%}"
            )
        
        if not recommendations:
            recommendations.append("Kiváló adatminőség! Nincs javaslat.")
        
        return recommendations
    
    def filter_low_quality_data(self, 
                               data_items: List[Dict[str, Any]], 
                               min_quality: float = 0.6) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter data based on quality thresholds
        
        Returns:
            Tuple of (high_quality_items, low_quality_items)
        """
        high_quality = []
        low_quality = []
        
        for item in data_items:
            if hasattr(item, 'overall_quality'):
                quality = item.overall_quality
            elif isinstance(item, dict) and 'overall_quality' in item:
                quality = item['overall_quality']
            else:
                # Default to medium quality if can't determine
                quality = 0.5
            
            if quality >= min_quality:
                high_quality.append(item)
            else:
                low_quality.append(item)
        
        logger.info(f"Filtered {len(data_items)} items: "
                   f"{len(high_quality)} high quality, {len(low_quality)} low quality")
        
        return high_quality, low_quality
    
    def get_quality_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        if not self.quality_history:
            return {}
        
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        recent_history = [
            item for item in self.quality_history
            if item['timestamp'] >= cutoff_date
        ]
        
        if not recent_history:
            return {}
        
        # Calculate trends
        metrics_df = pd.DataFrame([
            {
                'timestamp': item['timestamp'],
                'overall_quality': item['metrics'].overall_quality,
                'data_completeness': item['metrics'].data_completeness,
                'formula_complexity': item['metrics'].formula_complexity,
                'chart_quality': item['metrics'].chart_quality
            }
            for item in recent_history
        ])
        
        trends = {
            'total_files': len(recent_history),
            'average_quality': metrics_df['overall_quality'].mean(),
            'quality_trend': 'improving' if len(recent_history) > 1 else 'stable',
            'best_quality': metrics_df['overall_quality'].max(),
            'worst_quality': metrics_df['overall_quality'].min(),
            'quality_std': metrics_df['overall_quality'].std()
        }
        
        # Determine trend direction
        if len(recent_history) > 1:
            recent_quality = metrics_df['overall_quality'].iloc[-5:].mean()
            earlier_quality = metrics_df['overall_quality'].iloc[:-5].mean()
            if recent_quality > earlier_quality + 0.05:
                trends['quality_trend'] = 'improving'
            elif recent_quality < earlier_quality - 0.05:
                trends['quality_trend'] = 'declining'
            else:
                trends['quality_trend'] = 'stable'
        
        return trends
    
    def export_quality_report(self, output_path: str):
        """Export comprehensive quality report"""
        report = {
            'summary': {
                'total_files_assessed': len(self.quality_history),
                'average_quality': np.mean([item['metrics'].overall_quality 
                                          for item in self.quality_history]),
                'quality_distribution': {
                    'excellent': len([item for item in self.quality_history 
                                    if item['metrics'].quality_score == 'excellent']),
                    'good': len([item for item in self.quality_history 
                               if item['metrics'].quality_score == 'good']),
                    'fair': len([item for item in self.quality_history 
                               if item['metrics'].quality_score == 'fair']),
                    'poor': len([item for item in self.quality_history 
                               if item['metrics'].quality_score == 'poor'])
                }
            },
            'trends': self.get_quality_trends(),
            'recent_assessments': [
                {
                    'file': item['file'],
                    'timestamp': item['timestamp'].isoformat(),
                    'quality_score': item['metrics'].quality_score,
                    'overall_quality': item['metrics'].overall_quality
                }
                for item in self.quality_history[-10:]  # Last 10 assessments
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Quality report exported to {output_path}")
