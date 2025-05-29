"""
Integration calculator for signal processing.
This module provides functions for resampling and integrating electrophysiological data
with support for different integration methods (direct or regression-corrected).
"""

import numpy as np
from scipy.interpolate import interp1d
from src.utils.logger import app_logger
from src.excel_export.regression_utils import compute_regression_params, apply_curve_correction

def resample_data(time_s, current_pA, start_time_ms, duration_ms=12, step_ms=0.5):
    """
    Extract a segment from time_s (seconds) and current_pA (pA) starting at 
    start_time_ms (milliseconds) for a duration of duration_ms, then resample
    it to 0..(duration_ms - step_ms) in step_ms steps.

    Args:
        time_s (np.ndarray): Original time array in seconds.
        current_pA (np.ndarray): Original current array in pA.
        start_time_ms (float): Start time in milliseconds for the segment.
        duration_ms (float): Duration of the segment in ms (default 12 ms).
        step_ms (float): The resampling interval in ms (default 0.5 ms).

    Returns:
        (np.ndarray, np.ndarray):
            - new_time_ms: resampled time array, 0 to duration_ms in step_ms increments
            - new_current_pA: resampled current array
    """
    try:
        # Convert time to milliseconds
        time_ms = time_s * 1000.0

        # Select the segment from start_time_ms to start_time_ms+duration_ms
        mask = (time_ms >= start_time_ms) & (time_ms <= start_time_ms + duration_ms)
        segment_time = time_ms[mask] - start_time_ms  # Shift to start at 0
        segment_current = current_pA[mask]

        if len(segment_time) < 2:
            app_logger.error(f"Insufficient data points for resampling: {len(segment_time)} points")
            raise ValueError(
                f"Insufficient data points between {start_time_ms} ms and {start_time_ms + duration_ms} ms. "
                f"Found {len(segment_time)}."
            )

        # Create an interpolation function
        interp_func = interp1d(segment_time, segment_current, kind='linear', 
                              bounds_error=False, fill_value="extrapolate")

        # Build new time axis [0 .. duration_ms) in steps of step_ms
        new_time_ms = np.arange(0, duration_ms, step_ms)
        new_current_pA = interp_func(new_time_ms)

        app_logger.info(f"Resampled data from {len(segment_time)} to {len(new_time_ms)} points "
                       f"({start_time_ms}-{start_time_ms+duration_ms} ms)")
        return new_time_ms, new_current_pA
        
    except Exception as e:
        app_logger.error(f"Error resampling data: {str(e)}")
        raise

def calculate_integral_scenario_a(current_pA, time_ms=None, integral_val=1.0):
    """
    Calculate integral using direct summation method (Scenario A).
    Formula: SUM(current_values) / 2 / 1000 / integral_val
    
    Args:
        current_pA (np.ndarray): Current array in pA
        time_ms (np.ndarray): Time array in ms (optional, used for time step)
        integral_val (float): Scaling factor for the integral
        
    Returns:
        float: Calculated integral value
    """
    try:
        # Calculate time step in ms if time_ms is provided
        if time_ms is not None and len(time_ms) > 1:
            time_step_ms = np.mean(np.diff(time_ms))
        else:
            # Default to 0.5 ms steps
            time_step_ms = 0.5
            
        # Calculate the sum-based integral
        time_factor = 1.0 / (time_step_ms / 0.5)  # Normalize for time step (default is 0.5 ms)
        integral = np.sum(current_pA) / time_factor / 1000.0 / integral_val
        
        app_logger.info(f"Calculated Scenario A integral: {integral:.6f} (time step: {time_step_ms:.3f} ms)")
        return integral
        
    except Exception as e:
        app_logger.error(f"Error calculating Scenario A integral: {str(e)}")
        raise

def calculate_integral_scenario_b(time_ms, current_pA, integral_val, 
                                 fit_start_ms=None, fit_end_ms=None, shift=0):
    """
    Calculate integral using regression correction method (Scenario B).
    
    Args:
        time_ms (np.ndarray): Time array in ms
        current_pA (np.ndarray): Current array in pA
        integral_val (float): Scaling factor for the integral
        fit_start_ms (float): Start time for regression fitting
        fit_end_ms (float): End time for regression fitting
        shift (float): Time shift for regression formula
        
    Returns:
        float: Calculated integral value
    """
    try:
        # 1. Find regression parameters automatically
        slope, intercept = compute_regression_params(
            time_ms, current_pA, 
            fit_start_ms=fit_start_ms, 
            fit_end_ms=fit_end_ms
        )

        # 2. Apply correction to entire dataset
        corrected_pA = apply_curve_correction(time_ms, current_pA, slope, intercept, shift)

        # 3. Calculate time step in ms
        if len(time_ms) > 1:
            time_step_ms = np.mean(np.diff(time_ms))
        else:
            # Default to 0.5 ms steps
            time_step_ms = 0.5
            
        # 4. Calculate the sum-based integral
        time_factor = 1.0 / (time_step_ms / 0.5)  # Normalize for time step (default is 0.5 ms)
        integral = np.sum(corrected_pA) / time_factor / 1000.0 / integral_val
        
        app_logger.info(f"Calculated Scenario B integral: {integral:.6f} (slope: {slope:.6f}, "
                       f"intercept: {intercept:.6f}, shift: {shift})")
        return integral, slope, intercept
        
    except Exception as e:
        app_logger.error(f"Error calculating Scenario B integral: {str(e)}")
        raise