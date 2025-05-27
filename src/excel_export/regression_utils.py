"""
Regression utilities for signal processing.
This module provides functions for computing and applying linear regression
to correct baseline drift in electrophysiological data.
"""

import numpy as np
from src.utils.logger import app_logger

def compute_regression_params(time_ms, current_pA, fit_start_ms=None, fit_end_ms=None):
    """
    Compute slope and intercept for linear regression on part of the data.
    If fit_start_ms/fit_end_ms are provided, only that segment is used for fitting.
    
    Args:
        time_ms (np.ndarray): time array in ms
        current_pA (np.ndarray): current array in pA
        fit_start_ms (float): optional start time of the fitting region
        fit_end_ms (float): optional end time of the fitting region

    Returns:
        (float, float): (slope, intercept) from y = slope*x + intercept
    """
    try:
        # Optionally select the sub-region for fitting
        if fit_start_ms is not None and fit_end_ms is not None:
            mask = (time_ms >= fit_start_ms) & (time_ms <= fit_end_ms)
            x_fit = time_ms[mask]
            y_fit = current_pA[mask]
        else:
            # If not specified, use the entire array
            x_fit = time_ms
            y_fit = current_pA

        if len(x_fit) < 2:
            app_logger.error(f"Not enough points for regression: {len(x_fit)} points")
            raise ValueError("Not enough points in the chosen region for linear regression.")

        # Fit a line: slope, intercept
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        app_logger.info(f"Computed regression parameters: slope={slope:.6f}, intercept={intercept:.6f}")
        return slope, intercept
        
    except Exception as e:
        app_logger.error(f"Error computing regression parameters: {str(e)}")
        raise

def apply_curve_correction(time_ms, current_pA, slope, intercept, shift=0):
    """
    Apply linear regression correction to a signal.
    
    For Scenario B, subtract the fitted line from each data point:
        corrected = original - (slope*(time-shift) + intercept)
    
    Args:
        time_ms (np.ndarray): time array in ms
        current_pA (np.ndarray): current array in pA
        slope (float): slope of the regression line
        intercept (float): intercept of the regression line
        shift (float): optional time shift for regression formula
        
    Returns:
        np.ndarray: corrected current values
    """
    try:
        if len(time_ms) != len(current_pA):
            raise ValueError("Time and current arrays must have the same length")
            
        corrected_pA = current_pA.copy()
        # Apply correction: original - (slope*(time-shift) + intercept)
        corrected_pA = corrected_pA - (slope * (time_ms - shift) + intercept)
        
        app_logger.info(f"Applied curve correction with slope={slope:.6f}, intercept={intercept:.6f}, shift={shift}")
        return corrected_pA
        
    except Exception as e:
        app_logger.error(f"Error applying curve correction: {str(e)}")
        raise

def calculate_integral_with_correction(time_ms, current_pA, integral_val=1.0, 
                                       use_regression=False, 
                                       fit_start_ms=None, fit_end_ms=None,
                                       shift=0):
    """
    Calculate the integral of a signal with optional regression correction.
    
    Args:
        time_ms (np.ndarray): time array in ms
        current_pA (np.ndarray): current array in pA
        integral_val (float): scaling factor for the integral
        use_regression (bool): whether to apply regression correction
        fit_start_ms (float): optional start time for regression fitting
        fit_end_ms (float): optional end time for regression fitting
        shift (float): optional time shift for regression formula
        
    Returns:
        float: calculated integral value
    """
    try:
        # Apply regression correction if requested
        if use_regression:
            # Compute regression parameters
            slope, intercept = compute_regression_params(
                time_ms, current_pA, fit_start_ms, fit_end_ms
            )
            
            # Apply correction
            corrected_pA = apply_curve_correction(time_ms, current_pA, slope, intercept, shift)
        else:
            corrected_pA = current_pA.copy()
            
        # Calculate time step in seconds (assuming time_ms is in ms)
        if len(time_ms) > 1:
            time_step_ms = np.mean(np.diff(time_ms))
            time_step_s = time_step_ms / 1000.0
        else:
            # Default to 0.5 ms if only one point
            time_step_s = 0.0005
            
        # Calculate integral using trapezoidal rule
        integral = np.trapz(corrected_pA, time_ms) * 0.001  # Convert ms to s for integration
        
        # Alternative calculation method (matches Excel SUM approach)
        # SUM(values) / 2 / 1000 / integral_val - assuming 0.5ms steps
        sum_integral = np.sum(corrected_pA) * time_step_s / integral_val
        
        app_logger.info(f"Calculated integral: {integral:.6f} (trapz), {sum_integral:.6f} (sum)")
        return sum_integral
        
    except Exception as e:
        app_logger.error(f"Error calculating integral with correction: {str(e)}")
        raise