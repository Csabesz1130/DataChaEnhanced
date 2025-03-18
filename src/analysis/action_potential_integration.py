"""
Integration methods for action potential analysis.
This module provides functions for calculating integrals and capacitance from electrophysiology data.
"""

import numpy as np
from src.utils.logger import app_logger

def calculate_purple_integrals(hyperpol_data, hyperpol_times, depol_data, depol_times, voltage_diff):
    """
    Calculate integrals for both hyperpolarization and depolarization curves.
    
    Args:
        hyperpol_data (np.array): Hyperpolarization current data (pA)
        hyperpol_times (np.array): Hyperpolarization time data (s)
        depol_data (np.array): Depolarization current data (pA)
        depol_times (np.array): Depolarization time data (s)
        voltage_diff (float): Voltage difference (mV)
        
    Returns:
        dict: Dictionary with integral results
    """
    try:
        # Ensure we have valid inputs
        if hyperpol_data is None or depol_data is None:
            app_logger.error("Missing hyperpol or depol data for integration")
            return {
                'error': 'No data available for integration'
            }
            
        if len(hyperpol_data) < 2 or len(depol_data) < 2:
            app_logger.error("Insufficient data points for integration")
            return {
                'error': 'Insufficient data points'
            }
            
        # Convert times from seconds to milliseconds for integration
        hyperpol_ms = hyperpol_times * 1000
        depol_ms = depol_times * 1000
        
        # Calculate hyperpol integral (pA * ms = pC)
        hyperpol_pC = np.trapz(hyperpol_data, x=hyperpol_ms)
        
        # Calculate depol integral (pA * ms = pC)
        depol_pC = np.trapz(depol_data, x=depol_ms)
        
        # Calculate capacitance: C = Q/V (pC/mV = pF, we convert to nF)
        capacitance_pF = abs((hyperpol_pC - depol_pC) / voltage_diff)
        capacitance_nF = capacitance_pF / 1000
        
        # Format results for display
        return {
            'purple_integral_value': f"Hyperpol: {hyperpol_pC:.2f} pC, Depol: {depol_pC:.2f} pC",
            'hyperpol_area': f"{hyperpol_pC:.2f} pC",
            'depol_area': f"{depol_pC:.2f} pC",
            'capacitance_nF': f"{capacitance_nF:.2f} nF",
            'raw_values': {
                'hyperpol_pC': hyperpol_pC,
                'depol_pC': depol_pC,
                'capacitance_pF': capacitance_pF,
                'capacitance_nF': capacitance_nF
            }
        }
        
    except Exception as e:
        app_logger.error(f"Error calculating purple integrals: {str(e)}")
        return {
            'error': f"Integration error: {str(e)}"
        }

def integrate_curve_segment(data, times, start_idx, end_idx, baseline_correction=True):
    """
    Integrate a specific segment of curve data.
    
    Args:
        data (np.array): Current data (pA)
        times (np.array): Time data (s)
        start_idx (int): Starting index
        end_idx (int): Ending index
        baseline_correction (bool): Whether to apply baseline correction
        
    Returns:
        float: Integral value in pC
    """
    try:
        # Extract segment
        segment_data = data[start_idx:end_idx]
        segment_times = times[start_idx:end_idx]
        
        # Baseline correction
        if baseline_correction:
            # Use first few points for baseline
            baseline = np.median(segment_data[:min(10, len(segment_data))])
            segment_data = segment_data - baseline
        
        # Convert times to ms
        segment_times_ms = segment_times * 1000
        
        # Integrate (pA * ms = pC)
        integral_pC = np.trapz(segment_data, x=segment_times_ms)
        
        return integral_pC
        
    except Exception as e:
        app_logger.error(f"Error integrating curve segment: {str(e)}")
        return 0.0

def integrate_ranges(processor, ranges):
    """
    Integrate specific ranges of the purple curves.
    
    Args:
        processor: An ActionPotentialProcessor instance
        ranges: Dictionary with 'hyperpol' and 'depol' ranges, each containing 'start' and 'end' indices
        
    Returns:
        dict: Results dictionary with formatted integral values
    """
    try:
        # Make sure we have the purple curves
        if (not hasattr(processor, 'modified_hyperpol') or
            not hasattr(processor, 'modified_depol') or
            processor.modified_hyperpol is None or
            processor.modified_depol is None):
            return {
                'error': 'No purple curves available'
            }
            
        # Get hyperpolarization range
        hyperpol_range = ranges.get('hyperpol', {'start': 0, 'end': 199})
        hyperpol_start = hyperpol_range['start']
        hyperpol_end = hyperpol_range['end']
        
        # Get depolarization range
        depol_range = ranges.get('depol', {'start': 0, 'end': 199})
        depol_start = depol_range['start']
        depol_end = depol_range['end']
        
        # Make sure ranges are valid
        if hyperpol_start >= hyperpol_end or depol_start >= depol_end:
            app_logger.error(f"Invalid integration ranges: hyperpol={hyperpol_start}-{hyperpol_end}, depol={depol_start}-{depol_end}")
            return {
                'error': 'Invalid integration ranges'
            }
            
        # Check array bounds
        if (hyperpol_end > len(processor.modified_hyperpol) or 
            depol_end > len(processor.modified_depol)):
            app_logger.error(f"Integration range exceeds curve length: hyperpol={len(processor.modified_hyperpol)}, depol={len(processor.modified_depol)}")
            return {
                'error': 'Range exceeds curve length'
            }
        
        # Calculate the integrals for both ranges
        hyperpol_pC = integrate_curve_segment(
            processor.modified_hyperpol,
            processor.modified_hyperpol_times,
            hyperpol_start,
            hyperpol_end
        )
        
        depol_pC = integrate_curve_segment(
            processor.modified_depol,
            processor.modified_depol_times,
            depol_start,
            depol_end
        )
        
        # Calculate capacitance
        voltage_diff = abs(processor.params['V2'] - processor.params['V0'])
        capacitance_pF = abs(hyperpol_pC - depol_pC) / voltage_diff
        capacitance_nF = capacitance_pF / 1000
        
        # Format results
        return {
            'hyperpol_area': f"{hyperpol_pC:.2f} pC",
            'depol_area': f"{depol_pC:.2f} pC",
            'capacitance_nF': f"{capacitance_nF:.2f} nF",
            'integral_value': f"{abs(hyperpol_pC - depol_pC):.2f} pC",
            'raw_values': {
                'hyperpol_pC': hyperpol_pC,
                'depol_pC': depol_pC,
                'capacitance_pF': capacitance_pF,
                'capacitance_nF': capacitance_nF
            }
        }
        
    except Exception as e:
        app_logger.error(f"Error in integrate_ranges: {str(e)}")
        return {
            'error': f"Integration error: {str(e)}"
        }