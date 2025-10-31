#!/usr/bin/env python3
"""
Intelligent Sampling System for AI Excel Learning

This module provides intelligent sampling strategies to select optimal
training data sets for maximum learning efficiency.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random

logger = logging.getLogger(__name__)

@dataclass
class SamplingStrategy:
    """Configuration for sampling strategy"""
    name: str
    description: str
    min_sample_size: int
    max_sample_size: int
    diversity_weight: float = 0.4
    complexity_weight: float = 0.3
    quality_weight: float = 0.3
    random_seed: int = 42

@dataclass
class SamplingResult:
    """Result of intelligent sampling"""
    selected_indices: List[int]
    selected_data: List[Any]
    excluded_indices: List[int]
    sampling_metrics: Dict[str, float]
    strategy_used: str
    diversity_score: float
    complexity_score: float
    quality_score: float

class IntelligentSampling:
    """
    Intelligent sampling system for optimal training data selection
    """
    
    def __init__(self):
        self.strategies = {
            'diversity_based': SamplingStrategy(
                name='diversity_based',
                description='Maximize data diversity while maintaining quality',
                min_sample_size=50,
                max_sample_size=1000,
                diversity_weight=0.6,
                complexity_weight=0.2,
                quality_weight=0.2
            ),
            'complexity_based': SamplingStrategy(
                name='complexity_based',
                description='Focus on complex patterns for advanced learning',
                min_sample_size=30,
                max_sample_size=500,
                diversity_weight=0.3,
                complexity_weight=0.6,
                quality_weight=0.1
            ),
            'quality_based': SamplingStrategy(
                name='quality_based',
                description='Prioritize high-quality data for reliable learning',
                min_sample_size=100,
                max_sample_size=2000,
                diversity_weight=0.2,
                complexity_weight=0.2,
                quality_weight=0.6
            ),
            'balanced': SamplingStrategy(
                name='balanced',
                description='Balanced approach considering all factors equally',
                min_sample_size=75,
                max_sample_size=1500,
                diversity_weight=0.33,
                complexity_weight=0.33,
                quality_weight=0.34
            ),
            'adaptive': SamplingStrategy(
                name='adaptive',
                description='Adaptive sampling based on current model performance',
                min_sample_size=50,
                max_sample_size=1000,
                diversity_weight=0.4,
                complexity_weight=0.3,
                quality_weight=0.3
            )
        }
        
        self.scaler = StandardScaler()
        self.random_state = random.Random(42)
        
    def select_optimal_training_set(self, 
                                  available_data: List[Any], 
                                  target_size: int,
                                  strategy_name: str = 'balanced',
                                  quality_scores: Optional[List[float]] = None,
                                  complexity_scores: Optional[List[float]] = None) -> SamplingResult:
        """
        Select optimal training set using intelligent sampling
        
        Args:
            available_data: List of available data items
            target_size: Desired size of training set
            strategy_name: Name of sampling strategy to use
            quality_scores: Optional quality scores for each data item
            complexity_scores: Optional complexity scores for each data item
            
        Returns:
            SamplingResult with selected data and metrics
        """
        if not available_data:
            raise ValueError("No data available for sampling")
        
        if strategy_name not in self.strategies:
            strategy_name = 'balanced'
            logger.warning(f"Unknown strategy '{strategy_name}', using 'balanced'")
        
        strategy = self.strategies[strategy_name]
        
        # Adjust target size to strategy limits
        target_size = max(strategy.min_sample_size, 
                         min(target_size, strategy.max_sample_size, len(available_data)))
        
        logger.info(f"Selecting {target_size} items using {strategy_name} strategy "
                   f"from {len(available_data)} available items")
        
        # Calculate scores if not provided
        if quality_scores is None:
            quality_scores = self._calculate_default_quality_scores(available_data)
        
        if complexity_scores is None:
            complexity_scores = self._calculate_default_complexity_scores(available_data)
        
        # Calculate diversity scores
        diversity_scores = self._calculate_diversity_scores(available_data)
        
        # Combine scores according to strategy weights
        combined_scores = self._combine_scores(
            diversity_scores, complexity_scores, quality_scores, strategy
        )
        
        # Select optimal subset
        selected_indices = self._select_optimal_subset(
            combined_scores, target_size, strategy
        )
        
        # Create result
        selected_data = [available_data[i] for i in selected_indices]
        excluded_indices = [i for i in range(len(available_data)) if i not in selected_indices]
        
        # Calculate sampling metrics
        sampling_metrics = self._calculate_sampling_metrics(
            selected_indices, diversity_scores, complexity_scores, quality_scores
        )
        
        result = SamplingResult(
            selected_indices=selected_indices,
            selected_data=selected_data,
            excluded_indices=excluded_indices,
            sampling_metrics=sampling_metrics,
            strategy_used=strategy_name,
            diversity_score=sampling_metrics['diversity_score'],
            complexity_score=sampling_metrics['complexity_score'],
            quality_score=sampling_metrics['quality_score']
        )
        
        logger.info(f"Sampling completed: {len(selected_data)} items selected, "
                   f"diversity: {sampling_metrics['diversity_score']:.3f}, "
                   f"complexity: {sampling_metrics['complexity_score']:.3f}, "
                   f"quality: {sampling_metrics['quality_score']:.3f}")
        
        return result
    
    def _calculate_default_quality_scores(self, data: List[Any]) -> List[float]:
        """Calculate default quality scores for data items"""
        quality_scores = []
        
        for item in data:
            if hasattr(item, 'overall_quality'):
                quality_scores.append(item.overall_quality)
            elif isinstance(item, dict) and 'overall_quality' in item:
                quality_scores.append(item['overall_quality'])
            elif isinstance(item, dict) and 'quality_score' in item:
                # Convert quality score string to numeric
                score_str = item['quality_score']
                if score_str == 'excellent':
                    quality_scores.append(0.95)
                elif score_str == 'good':
                    quality_scores.append(0.8)
                elif score_str == 'fair':
                    quality_scores.append(0.6)
                elif score_str == 'poor':
                    quality_scores.append(0.3)
                else:
                    quality_scores.append(0.5)
            else:
                # Default quality score
                quality_scores.append(0.5)
        
        return quality_scores
    
    def _calculate_default_complexity_scores(self, data: List[Any]) -> List[float]:
        """Calculate default complexity scores for data items"""
        complexity_scores = []
        
        for item in data:
            if hasattr(item, 'formula_complexity'):
                complexity_scores.append(item.formula_complexity)
            elif isinstance(item, dict) and 'formula_complexity' in item:
                complexity_scores.append(item['formula_complexity'])
            elif isinstance(item, dict) and 'formulas' in item:
                # Estimate complexity from formula count
                formula_count = len(item.get('formulas', []))
                complexity_scores.append(min(formula_count / 10, 1.0))
            else:
                # Default complexity score
                complexity_scores.append(0.5)
        
        return complexity_scores
    
    def _calculate_diversity_scores(self, data: List[Any]) -> List[float]:
        """Calculate diversity scores using clustering and similarity analysis"""
        try:
            # Extract features for diversity calculation
            features = self._extract_diversity_features(data)
            
            if len(features) < 2:
                return [0.5] * len(data)
            
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Use K-means clustering to identify diverse groups
            n_clusters = min(5, len(features) // 10)
            if n_clusters < 2:
                n_clusters = 2
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Calculate diversity scores based on cluster distribution
            diversity_scores = []
            for i, label in enumerate(cluster_labels):
                # Items in smaller clusters get higher diversity scores
                cluster_size = list(cluster_labels).count(label)
                diversity_score = 1.0 / (cluster_size + 1)  # +1 to avoid division by zero
                diversity_scores.append(diversity_score)
            
            # Normalize to 0-1 range
            max_score = max(diversity_scores)
            if max_score > 0:
                diversity_scores = [score / max_score for score in diversity_scores]
            
            return diversity_scores
            
        except Exception as e:
            logger.warning(f"Error calculating diversity scores: {e}, using random scores")
            return [self.random_state.random() for _ in data]
    
    def _extract_diversity_features(self, data: List[Any]) -> List[List[float]]:
        """Extract features for diversity calculation"""
        features = []
        
        for item in data:
            feature_vector = []
            
            # Extract basic features
            if isinstance(item, dict):
                # Sheet count
                feature_vector.append(len(item.get('sheets', [])) / 10)
                
                # Total cells
                total_cells = sum(sheet.get('cells_with_data', 0) for sheet in item.get('sheets', []))
                feature_vector.append(min(total_cells / 1000, 1.0))
                
                # Formula count
                formula_count = item.get('total_formulas', 0)
                feature_vector.append(min(formula_count / 50, 1.0))
                
                # Chart count
                chart_count = len(item.get('charts', []))
                feature_vector.append(min(chart_count / 10, 1.0))
                
                # Data type diversity
                data_types = set()
                for sheet in item.get('sheets', []):
                    if 'data_ranges' in sheet:
                        for range_data in sheet['data_ranges'].get('ranges', []):
                            data_types.add(range_data.get('data_type', 'unknown'))
                feature_vector.append(len(data_types) / 5)
                
            else:
                # Default features for unknown data types
                feature_vector.extend([0.5, 0.5, 0.5, 0.5, 0.5])
            
            features.append(feature_vector)
        
        return features
    
    def _combine_scores(self, 
                        diversity_scores: List[float],
                        complexity_scores: List[float],
                        quality_scores: List[float],
                        strategy: SamplingStrategy) -> List[float]:
        """Combine individual scores according to strategy weights"""
        combined_scores = []
        
        for i in range(len(diversity_scores)):
            combined_score = (
                diversity_scores[i] * strategy.diversity_weight +
                complexity_scores[i] * strategy.complexity_weight +
                quality_scores[i] * strategy.quality_weight
            )
            combined_scores.append(combined_score)
        
        return combined_scores
    
    def _select_optimal_subset(self, 
                              scores: List[float], 
                              target_size: int,
                              strategy: SamplingStrategy) -> List[int]:
        """Select optimal subset based on combined scores"""
        if target_size >= len(scores):
            return list(range(len(scores)))
        
        # Create index-score pairs and sort by score
        index_score_pairs = [(i, score) for i, score in enumerate(scores)]
        index_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top scores
        selected_indices = [pair[0] for pair in index_score_pairs[:target_size]]
        
        # Add some randomness for diversity (if not quality-focused)
        if strategy.quality_weight < 0.5 and len(selected_indices) > 10:
            # Replace 10% of lowest scores with random selection from top 50%
            replace_count = max(1, int(target_size * 0.1))
            top_50_percent = int(target_size * 0.5)
            
            for _ in range(replace_count):
                if len(selected_indices) > 1:
                    # Remove one of the lowest scores
                    selected_indices.pop()
                    # Add random selection from top 50%
                    random_index = self.random_state.randint(0, top_50_percent)
                    if random_index < len(index_score_pairs):
                        selected_indices.append(index_score_pairs[random_index][0])
        
        return selected_indices
    
    def _calculate_sampling_metrics(self,
                                   selected_indices: List[int],
                                   diversity_scores: List[float],
                                   complexity_scores: List[float],
                                   quality_scores: List[float]) -> Dict[str, float]:
        """Calculate metrics for the selected subset"""
        if not selected_indices:
            return {
                'diversity_score': 0.0,
                'complexity_score': 0.0,
                'quality_score': 0.0,
                'coverage_score': 0.0
            }
        
        # Calculate average scores for selected items
        selected_diversity = np.mean([diversity_scores[i] for i in selected_indices])
        selected_complexity = np.mean([complexity_scores[i] for i in selected_indices])
        selected_quality = np.mean([quality_scores[i] for i in selected_indices])
        
        # Calculate coverage (how well the selection covers the full range)
        all_scores = list(zip(diversity_scores, complexity_scores, quality_scores))
        selected_scores = [all_scores[i] for i in selected_indices]
        
        # Coverage based on how well selected scores span the full range
        coverage_score = self._calculate_coverage_score(all_scores, selected_scores)
        
        return {
            'diversity_score': selected_diversity,
            'complexity_score': selected_complexity,
            'quality_score': selected_quality,
            'coverage_score': coverage_score
        }
    
    def _calculate_coverage_score(self, 
                                 all_scores: List[Tuple[float, float, float]],
                                 selected_scores: List[Tuple[float, float, float]]) -> float:
        """Calculate how well the selection covers the full range of scores"""
        if not all_scores or not selected_scores:
            return 0.0
        
        # Calculate ranges for each dimension
        all_ranges = []
        selected_ranges = []
        
        for dim in range(3):  # diversity, complexity, quality
            all_values = [score[dim] for score in all_scores]
            selected_values = [score[dim] for score in selected_scores]
            
            all_range = max(all_values) - min(all_values)
            selected_range = max(selected_values) - min(selected_values)
            
            if all_range > 0:
                all_ranges.append(all_range)
                selected_ranges.append(selected_range)
        
        if not all_ranges:
            return 0.0
        
        # Coverage is the ratio of selected range to full range
        coverage_ratios = [selected_ranges[i] / all_ranges[i] for i in range(len(all_ranges))]
        return np.mean(coverage_ratios)
    
    def adaptive_sampling(self,
                         available_data: List[Any],
                         current_performance: Dict[str, float],
                         target_size: int) -> SamplingResult:
        """
        Adaptive sampling based on current model performance
        
        Args:
            available_data: Available data items
            current_performance: Current model performance metrics
            target_size: Desired training set size
            
        Returns:
            SamplingResult with adaptively selected data
        """
        # Analyze performance to determine strategy
        if current_performance.get('accuracy', 0.5) < 0.6:
            # Low accuracy - focus on quality
            strategy_name = 'quality_based'
        elif current_performance.get('accuracy', 0.5) > 0.8:
            # High accuracy - focus on diversity and complexity
            strategy_name = 'diversity_based'
        else:
            # Medium accuracy - balanced approach
            strategy_name = 'balanced'
        
        logger.info(f"Adaptive sampling: performance={current_performance.get('accuracy', 0.5):.3f}, "
                   f"selected strategy={strategy_name}")
        
        return self.select_optimal_training_set(
            available_data, target_size, strategy_name
        )
    
    def get_sampling_recommendations(self, 
                                   data_size: int,
                                   target_size: int,
                                   current_performance: Optional[Dict[str, float]] = None) -> List[str]:
        """Get recommendations for sampling strategy selection"""
        recommendations = []
        
        if target_size > data_size * 0.8:
            recommendations.append(
                f"Target size ({target_size}) is very large compared to available data ({data_size}). "
                "Consider reducing target size or collecting more data."
            )
        
        if target_size < 50:
            recommendations.append(
                "Target size is very small. Consider increasing to at least 50 items "
                "for reliable learning."
            )
        
        if current_performance:
            accuracy = current_performance.get('accuracy', 0.5)
            if accuracy < 0.6:
                recommendations.append(
                    "Low accuracy detected. Use 'quality_based' strategy to focus on "
                    "high-quality training examples."
                )
            elif accuracy > 0.8:
                recommendations.append(
                    "High accuracy achieved. Use 'diversity_based' strategy to expand "
                    "model capabilities."
                )
        
        if not recommendations:
            recommendations.append(
                "Data size and target size are well-balanced. "
                "Use 'balanced' strategy for optimal results."
            )
        
        return recommendations
