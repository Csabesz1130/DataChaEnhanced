"""
Linear Fit Subtraction Module for DataChaEnhanced
================================================
Location: src/analysis/linear_fit_subtractor.py

This module provides functionality to subtract linearly fitted curves from 
hyperpol and depol purple curves separately, then reload the plot.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class LinearFitSubtractor:
    """
    Handles subtraction of linear fits from purple curves (hyperpol and depol).
    """
    
    def __init__(self):
        """Initialize the linear fit subtractor."""
        self.fitted_curves = {
            'hyperpol': {
                'linear_params': None,
                'linear_curve': None,
                'r_squared': None
            },
            'depol': {
                'linear_params': None,
                'linear_curve': None,
                'r_squared': None
            }
        }
        
        self.subtracted_data = {
            'hyperpol': {
                'original_data': None,
                'original_times': None,
                'subtracted_data': None,
                'subtracted_times': None
            },
            'depol': {
                'original_data': None,
                'original_times': None,
                'subtracted_data': None,
                'subtracted_times': None
            }
        }
    
    def set_fitted_curves(self, curve_type: str, linear_params: Dict, 
                         linear_curve: Dict, r_squared: float):
        """
        Set the fitted linear curve data for a specific curve type.
        
        Args:
            curve_type: 'hyperpol' or 'depol'
            linear_params: Dictionary containing slope, intercept, start_idx, end_idx
            linear_curve: Dictionary containing times and data arrays
            r_squared: R-squared value of the fit
        """
        if curve_type not in ['hyperpol', 'depol']:
            raise ValueError("curve_type must be 'hyperpol' or 'depol'")
        
        self.fitted_curves[curve_type] = {
            'linear_params': linear_params,
            'linear_curve': linear_curve,
            'r_squared': r_squared
        }
        
        logger.info(f"Set linear fit for {curve_type}: "
                   f"y = {linear_params['slope']:.6f}x + {linear_params['intercept']:.6f}, "
                   f"R² = {r_squared:.4f}")
    
    def set_original_data(self, curve_type: str, data: np.ndarray, times: np.ndarray):
        """
        Set the original data for a curve type.
        
        Args:
            curve_type: 'hyperpol' or 'depol'
            data: Original curve data
            times: Original time data
        """
        if curve_type not in ['hyperpol', 'depol']:
            raise ValueError("curve_type must be 'hyperpol' or 'depol'")
        
        self.subtracted_data[curve_type]['original_data'] = data.copy()
        self.subtracted_data[curve_type]['original_times'] = times.copy()
        
        logger.info(f"Set original data for {curve_type}: {len(data)} points")
    
    def subtract_linear_fit(self, curve_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subtract the linear fit from the original curve data.
        
        Args:
            curve_type: 'hyperpol' or 'depol'
            
        Returns:
            Tuple of (subtracted_data, times)
        """
        if curve_type not in ['hyperpol', 'depol']:
            raise ValueError("curve_type must be 'hyperpol' or 'depol'")
        
        # Check if we have the necessary data
        if not self._has_required_data(curve_type):
            raise ValueError(f"Missing required data for {curve_type} subtraction")
        
        original_data = self.subtracted_data[curve_type]['original_data']
        original_times = self.subtracted_data[curve_type]['original_times']
        linear_params = self.fitted_curves[curve_type]['linear_params']
        
        # Generate the linear fit over the full time range
        linear_fit = linear_params['slope'] * original_times + linear_params['intercept']
        
        # Subtract the linear fit from the original data
        subtracted_data = original_data - linear_fit
        
        # Store the results
        self.subtracted_data[curve_type]['subtracted_data'] = subtracted_data.copy()
        self.subtracted_data[curve_type]['subtracted_times'] = original_times.copy()
        
        logger.info(f"Subtracted linear fit from {curve_type}: "
                   f"mean change = {np.mean(linear_fit):.6f}")
        
        return subtracted_data, original_times
    
    def subtract_both_curves(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Subtract linear fits from both hyperpol and depol curves.
        
        Returns:
            Dictionary with 'hyperpol' and 'depol' keys, each containing (data, times) tuples
        """
        results = {}
        
        for curve_type in ['hyperpol', 'depol']:
            if self._has_required_data(curve_type):
                try:
                    data, times = self.subtract_linear_fit(curve_type)
                    results[curve_type] = (data, times)
                except Exception as e:
                    logger.error(f"Failed to subtract linear fit from {curve_type}: {str(e)}")
            else:
                logger.warning(f"Skipping {curve_type} - missing required data")
        
        return results
    
    def get_subtracted_data(self, curve_type: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the subtracted data for a specific curve type.
        
        Args:
            curve_type: 'hyperpol' or 'depol'
            
        Returns:
            Tuple of (subtracted_data, times) or None if not available
        """
        if curve_type not in ['hyperpol', 'depol']:
            raise ValueError("curve_type must be 'hyperpol' or 'depol'")
        
        if (self.subtracted_data[curve_type]['subtracted_data'] is not None and
            self.subtracted_data[curve_type]['subtracted_times'] is not None):
            return (self.subtracted_data[curve_type]['subtracted_data'],
                    self.subtracted_data[curve_type]['subtracted_times'])
        
        return None
    
    def get_all_subtracted_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get all available subtracted data.
        
        Returns:
            Dictionary with available subtracted data
        """
        results = {}
        for curve_type in ['hyperpol', 'depol']:
            data = self.get_subtracted_data(curve_type)
            if data is not None:
                results[curve_type] = data
        
        return results
    
    def reset_subtraction(self, curve_type: Optional[str] = None):
        """
        Reset subtraction data for a specific curve type or all curves.
        
        Args:
            curve_type: 'hyperpol', 'depol', or None for all curves
        """
        if curve_type is None:
            # Reset all curves
            for ct in ['hyperpol', 'depol']:
                self.subtracted_data[ct]['subtracted_data'] = None
                self.subtracted_data[ct]['subtracted_times'] = None
            logger.info("Reset all subtraction data")
        elif curve_type in ['hyperpol', 'depol']:
            # Reset specific curve
            self.subtracted_data[curve_type]['subtracted_data'] = None
            self.subtracted_data[curve_type]['subtracted_times'] = None
            logger.info(f"Reset subtraction data for {curve_type}")
        else:
            raise ValueError("curve_type must be 'hyperpol', 'depol', or None")
    
    def _has_required_data(self, curve_type: str) -> bool:
        """
        Check if we have all required data for subtraction.
        
        Args:
            curve_type: 'hyperpol' or 'depol'
            
        Returns:
            True if all required data is available
        """
        # Check original data
        if (self.subtracted_data[curve_type]['original_data'] is None or
            self.subtracted_data[curve_type]['original_times'] is None):
            return False
        
        # Check fitted curve
        if (self.fitted_curves[curve_type]['linear_params'] is None or
            self.fitted_curves[curve_type]['linear_curve'] is None):
            return False
        
        return True
    
    def get_fit_info(self, curve_type: str) -> Optional[Dict]:
        """
        Get information about the fitted curve.
        
        Args:
            curve_type: 'hyperpol' or 'depol'
            
        Returns:
            Dictionary with fit information or None if not available
        """
        if curve_type not in ['hyperpol', 'depol']:
            raise ValueError("curve_type must be 'hyperpol' or 'depol'")
        
        if not self._has_required_data(curve_type):
            return None
        
        params = self.fitted_curves[curve_type]['linear_params']
        r_squared = self.fitted_curves[curve_type]['r_squared']
        
        return {
            'slope': params['slope'],
            'intercept': params['intercept'],
            'r_squared': r_squared,
            'equation': f"y = {params['slope']:.6f}x + {params['intercept']:.6f}",
            'start_time': params['start_time'],
            'end_time': params['end_time']
        }
