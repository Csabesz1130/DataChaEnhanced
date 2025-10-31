"""
Data Generator for Excel AI

This module generates synthetic data based on learned patterns and models
to create realistic Excel files.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import random
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for data generation"""
    num_rows: int
    num_columns: int
    data_types: List[str]
    patterns: List[str]
    constraints: Dict[str, Any]
    relationships: List[Dict[str, Any]]

class DataGenerator:
    """
    Generates synthetic data based on learned patterns and models
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        self.generation_history = []
        
    def generate_data_from_patterns(self, patterns: List[Dict[str, Any]], 
                                   config: GenerationConfig) -> pd.DataFrame:
        """
        Generate data based on learned patterns
        
        Args:
            patterns: List of data patterns from ExcelAnalyzer
            config: Generation configuration
            
        Returns:
            Generated DataFrame
        """
        logger.info(f"Generating data with {config.num_rows} rows and {config.num_columns} columns")
        
        # Initialize DataFrame
        data = pd.DataFrame()
        
        # Generate each column based on patterns
        for i in range(config.num_columns):
            if i < len(patterns):
                pattern = patterns[i]
                column_data = self._generate_column_from_pattern(pattern, config.num_rows)
            else:
                # Generate random column if no pattern available
                column_data = self._generate_random_column(config.num_rows)
            
            column_name = f"Column_{i+1}"
            data[column_name] = column_data
        
        # Apply relationships and constraints
        data = self._apply_relationships(data, config.relationships)
        data = self._apply_constraints(data, config.constraints)
        
        # Record generation
        self.generation_history.append({
            'timestamp': datetime.now().isoformat(),
            'config': config.__dict__,
            'generated_shape': data.shape
        })
        
        return data
    
    def _generate_column_from_pattern(self, pattern: Dict[str, Any], num_rows: int) -> pd.Series:
        """Generate a single column based on pattern"""
        pattern_type = pattern.get('pattern_type', 'random')
        data_type = pattern.get('data_type', 'numeric')
        statistics = pattern.get('statistics', {})
        
        if pattern_type == 'sequential':
            return self._generate_sequential_data(num_rows, statistics)
        elif pattern_type == 'formula':
            return self._generate_formula_data(num_rows, statistics)
        elif pattern_type == 'categorical':
            return self._generate_categorical_data(num_rows, statistics)
        elif pattern_type == 'random':
            return self._generate_random_data(num_rows, data_type, statistics)
        else:
            return self._generate_random_data(num_rows, data_type, statistics)
    
    def _generate_sequential_data(self, num_rows: int, statistics: Dict[str, Any]) -> pd.Series:
        """Generate sequential data"""
        start_value = statistics.get('min', 0)
        step = statistics.get('step', 1)
        
        if 'mean' in statistics and 'std' in statistics:
            # Generate with some noise around the trend
            trend = np.arange(start_value, start_value + num_rows * step, step)
            noise = np.random.normal(0, statistics['std'] * 0.1, num_rows)
            return pd.Series(trend + noise)
        else:
            return pd.Series(np.arange(start_value, start_value + num_rows * step, step))
    
    def _generate_formula_data(self, num_rows: int, statistics: Dict[str, Any]) -> pd.Series:
        """Generate formula-based data"""
        # Try to identify the formula pattern
        if 'coefficients' in statistics:
            # Polynomial formula
            x = np.arange(num_rows)
            coefficients = statistics['coefficients']
            result = np.polyval(coefficients, x)
            return pd.Series(result)
        else:
            # Linear formula
            slope = statistics.get('slope', 1)
            intercept = statistics.get('intercept', 0)
            x = np.arange(num_rows)
            return pd.Series(slope * x + intercept)
    
    def _generate_categorical_data(self, num_rows: int, statistics: Dict[str, Any]) -> pd.Series:
        """Generate categorical data"""
        categories = statistics.get('categories', ['A', 'B', 'C', 'D'])
        probabilities = statistics.get('probabilities', None)
        
        if probabilities and len(probabilities) == len(categories):
            return pd.Series(np.random.choice(categories, num_rows, p=probabilities))
        else:
            return pd.Series(np.random.choice(categories, num_rows))
    
    def _generate_random_data(self, num_rows: int, data_type: str, 
                            statistics: Dict[str, Any]) -> pd.Series:
        """Generate random data"""
        if data_type == 'numeric':
            mean = statistics.get('mean', 0)
            std = statistics.get('std', 1)
            min_val = statistics.get('min', mean - 3 * std)
            max_val = statistics.get('max', mean + 3 * std)
            
            data = np.random.normal(mean, std, num_rows)
            data = np.clip(data, min_val, max_val)
            return pd.Series(data)
        
        elif data_type == 'date':
            start_date = statistics.get('start_date', datetime.now())
            end_date = statistics.get('end_date', start_date + timedelta(days=num_rows))
            
            date_range = pd.date_range(start=start_date, end=end_date, periods=num_rows)
            return pd.Series(date_range)
        
        elif data_type == 'boolean':
            p_true = statistics.get('p_true', 0.5)
            return pd.Series(np.random.choice([True, False], num_rows, p=[p_true, 1-p_true]))
        
        else:  # text
            words = statistics.get('words', ['word1', 'word2', 'word3', 'word4', 'word5'])
            return pd.Series(np.random.choice(words, num_rows))
    
    def _generate_random_column(self, num_rows: int) -> pd.Series:
        """Generate a completely random column"""
        data_type = random.choice(['numeric', 'categorical', 'boolean'])
        
        if data_type == 'numeric':
            return pd.Series(np.random.normal(0, 1, num_rows))
        elif data_type == 'categorical':
            categories = ['A', 'B', 'C', 'D', 'E']
            return pd.Series(np.random.choice(categories, num_rows))
        else:  # boolean
            return pd.Series(np.random.choice([True, False], num_rows))
    
    def _apply_relationships(self, data: pd.DataFrame, 
                           relationships: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply relationships between columns"""
        for rel in relationships:
            source_col = rel.get('source_column')
            target_col = rel.get('target_column')
            rel_type = rel.get('type', 'correlation')
            strength = rel.get('strength', 0.8)
            
            if source_col in data.columns and target_col in data.columns:
                if rel_type == 'correlation':
                    data[target_col] = self._apply_correlation(data[source_col], data[target_col], strength)
                elif rel_type == 'formula':
                    formula = rel.get('formula', 'x * 2')
                    data[target_col] = self._apply_formula(data[source_col], formula)
        
        return data
    
    def _apply_correlation(self, source: pd.Series, target: pd.Series, 
                          strength: float) -> pd.Series:
        """Apply correlation between two columns"""
        # Normalize both series
        source_norm = (source - source.mean()) / source.std()
        target_norm = (target - target.mean()) / target.std()
        
        # Create correlated series
        correlated = strength * source_norm + np.sqrt(1 - strength**2) * target_norm
        
        # Denormalize
        return correlated * target.std() + target.mean()
    
    def _apply_formula(self, source: pd.Series, formula: str) -> pd.Series:
        """Apply formula to source column"""
        try:
            # Simple formula evaluation (in production, use a safer method)
            x = source
            return eval(formula)
        except:
            # Fallback to simple multiplication
            return source * 2
    
    def _apply_constraints(self, data: pd.DataFrame, 
                          constraints: Dict[str, Any]) -> pd.DataFrame:
        """Apply constraints to the data"""
        # Range constraints
        for col in data.columns:
            if col in constraints.get('ranges', {}):
                range_constraint = constraints['ranges'][col]
                min_val = range_constraint.get('min')
                max_val = range_constraint.get('max')
                
                if min_val is not None:
                    data[col] = data[col].clip(lower=min_val)
                if max_val is not None:
                    data[col] = data[col].clip(upper=max_val)
        
        # Uniqueness constraints
        for col in constraints.get('unique_columns', []):
            if col in data.columns:
                data[col] = data[col].drop_duplicates().reset_index(drop=True)
        
        return data
    
    def generate_time_series_data(self, num_points: int, 
                                 pattern_type: str = 'trend',
                                 **kwargs) -> pd.DataFrame:
        """
        Generate time series data
        
        Args:
            num_points: Number of data points
            pattern_type: Type of pattern ('trend', 'seasonal', 'cyclical', 'random')
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with time series data
        """
        dates = pd.date_range(start=kwargs.get('start_date', '2020-01-01'), 
                             periods=num_points, freq=kwargs.get('freq', 'D'))
        
        if pattern_type == 'trend':
            trend = np.linspace(kwargs.get('start_value', 0), 
                               kwargs.get('end_value', 100), num_points)
            noise = np.random.normal(0, kwargs.get('noise_std', 1), num_points)
            values = trend + noise
        
        elif pattern_type == 'seasonal':
            base_trend = np.linspace(0, 50, num_points)
            seasonal = 10 * np.sin(2 * np.pi * np.arange(num_points) / 365.25)
            noise = np.random.normal(0, kwargs.get('noise_std', 1), num_points)
            values = base_trend + seasonal + noise
        
        elif pattern_type == 'cyclical':
            trend = np.linspace(0, 30, num_points)
            cyclical = 15 * np.sin(2 * np.pi * np.arange(num_points) / 30)
            noise = np.random.normal(0, kwargs.get('noise_std', 1), num_points)
            values = trend + cyclical + noise
        
        else:  # random
            values = np.random.normal(kwargs.get('mean', 0), 
                                    kwargs.get('std', 1), num_points)
        
        return pd.DataFrame({
            'date': dates,
            'value': values
        })
    
    def generate_correlated_data(self, num_rows: int, 
                                correlation_matrix: np.ndarray,
                                column_names: List[str] = None) -> pd.DataFrame:
        """
        Generate correlated data using a correlation matrix
        
        Args:
            num_rows: Number of rows to generate
            correlation_matrix: Correlation matrix (n x n)
            column_names: Names for the columns
            
        Returns:
            DataFrame with correlated data
        """
        n_cols = correlation_matrix.shape[0]
        
        if column_names is None:
            column_names = [f'Column_{i+1}' for i in range(n_cols)]
        
        # Generate uncorrelated data
        uncorrelated = np.random.normal(0, 1, (num_rows, n_cols))
        
        # Apply correlation using Cholesky decomposition
        try:
            L = np.linalg.cholesky(correlation_matrix)
            correlated = uncorrelated @ L.T
        except np.linalg.LinAlgError:
            # If correlation matrix is not positive definite, use SVD
            U, s, Vt = np.linalg.svd(correlation_matrix)
            L = U @ np.sqrt(np.diag(s))
            correlated = uncorrelated @ L.T
        
        return pd.DataFrame(correlated, columns=column_names)
    
    def generate_clustered_data(self, num_points: int, 
                               num_clusters: int = 3,
                               cluster_centers: List[List[float]] = None,
                               cluster_sizes: List[int] = None) -> pd.DataFrame:
        """
        Generate clustered data
        
        Args:
            num_points: Total number of points
            num_clusters: Number of clusters
            cluster_centers: Centers of clusters
            cluster_sizes: Size of each cluster
            
        Returns:
            DataFrame with clustered data
        """
        if cluster_centers is None:
            cluster_centers = np.random.uniform(-10, 10, (num_clusters, 2))
        
        if cluster_sizes is None:
            cluster_sizes = [num_points // num_clusters] * num_clusters
            cluster_sizes[0] += num_points % num_clusters
        
        data = []
        labels = []
        
        for i in range(num_clusters):
            cluster_data = np.random.normal(cluster_centers[i], 1, (cluster_sizes[i], 2))
            data.extend(cluster_data)
            labels.extend([i] * cluster_sizes[i])
        
        return pd.DataFrame(data, columns=['X', 'Y']).assign(cluster=labels)
    
    def generate_text_data(self, num_rows: int, 
                          text_patterns: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate text data based on patterns
        
        Args:
            num_rows: Number of rows to generate
            text_patterns: List of text generation patterns
            
        Returns:
            DataFrame with text data
        """
        data = {}
        
        for pattern in text_patterns:
            column_name = pattern.get('column_name', 'text_column')
            pattern_type = pattern.get('pattern_type', 'random')
            
            if pattern_type == 'names':
                names = pattern.get('names', ['John', 'Jane', 'Bob', 'Alice'])
                data[column_name] = np.random.choice(names, num_rows)
            
            elif pattern_type == 'emails':
                domains = pattern.get('domains', ['gmail.com', 'yahoo.com', 'hotmail.com'])
                usernames = [f"user{i}" for i in range(num_rows)]
                data[column_name] = [f"{username}@{np.random.choice(domains)}" 
                                   for username in usernames]
            
            elif pattern_type == 'addresses':
                streets = pattern.get('streets', ['Main St', 'Oak Ave', 'Pine Rd'])
                cities = pattern.get('cities', ['New York', 'Los Angeles', 'Chicago'])
                data[column_name] = [f"{np.random.randint(1, 9999)} {np.random.choice(streets)}, {np.random.choice(cities)}" 
                                   for _ in range(num_rows)]
            
            else:  # random
                words = pattern.get('words', ['word1', 'word2', 'word3', 'word4', 'word5'])
                data[column_name] = np.random.choice(words, num_rows)
        
        return pd.DataFrame(data)
    
    def save_generation_history(self, file_path: str):
        """Save generation history to file"""
        with open(file_path, 'w') as f:
            json.dump(self.generation_history, f, indent=2)
        
        logger.info(f"Generation history saved to: {file_path}")
    
    def load_generation_history(self, file_path: str):
        """Load generation history from file"""
        with open(file_path, 'r') as f:
            self.generation_history = json.load(f)
        
        logger.info(f"Generation history loaded from: {file_path}")
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about generation history"""
        if not self.generation_history:
            return {}
        
        total_generations = len(self.generation_history)
        total_rows = sum(gen['generated_shape'][0] for gen in self.generation_history)
        total_columns = sum(gen['generated_shape'][1] for gen in self.generation_history)
        
        return {
            'total_generations': total_generations,
            'total_rows_generated': total_rows,
            'total_columns_generated': total_columns,
            'average_rows_per_generation': total_rows / total_generations,
            'average_columns_per_generation': total_columns / total_generations
        }
