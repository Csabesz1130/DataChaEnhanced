"""
Spike removal utility for electrophysiology data.
This module provides functions to detect and remove periodic spikes from various signal types.
"""

import numpy as np
from src.utils.logger import app_logger

def remove_periodic_spikes(data, times=None, processor=None):
    """
    Identify and remove spikes at (n + 200*i) in the data, replacing each
    with its preceding point. 'n' is taken from 'processor.params' if available,
    otherwise defaults to 28.
    
    Args:
        data (np.array): Signal data with spikes to be removed
        times (np.array, optional): Corresponding time values (unused here, but left for compatibility)
        processor (ActionPotentialProcessor, optional): If provided, we try
            to read 'n' from processor.params['normalization_points']['seg1'][0].
            
    Returns:
        np.array: Corrected signal with spikes removed
    """
    if data is None or len(data) < 50:
        app_logger.warning("Data too short for spike removal")
        return data

    try:
        # 1) Determine the starting index n from the processor or default to 28
        if processor and hasattr(processor, 'params'):
            norm_points = processor.params.get('normalization_points', {})
            if 'seg1' in norm_points and len(norm_points['seg1']) > 0:
                n = norm_points['seg1'][0]
            else:
                n = 28
                app_logger.debug("No 'seg1' found in params; using default n=28")
        else:
            n = 28
            app_logger.debug("No processor or no params; using default n=28")

        # 2) Create a copy so we don't modify the original data
        cleaned_data = np.array(data).copy()

        # 3) Remove spikes every 200 points starting at n
        period = 200
        i = 0
        replaced_count = 0
        while True:
            spike_index = n + period * i
            if spike_index >= len(cleaned_data):
                break  # Out of bounds, stop

            # The "spike" is replaced by the preceding point if possible
            if spike_index > 0:
                old_val = cleaned_data[spike_index]
                cleaned_data[spike_index] = cleaned_data[spike_index - 1]
                replaced_count += 1
                app_logger.debug(
                    f"Spike at {spike_index} replaced: {old_val:.2f} -> {cleaned_data[spike_index]:.2f}"
                )
            i += 1

        if replaced_count > 0:
            app_logger.info(
                f"Periodic spike removal complete: replaced {replaced_count} points "
                f"(start={n}, period=200)."
            )
        else:
            app_logger.info("No spikes replaced (none found at n + 200*i).")

        return cleaned_data

    except Exception as e:
        app_logger.error(f"Error during spike removal: {str(e)}")
        return data


def estimate_affected_points(data, spike_index, threshold):
    """
    Estimate how many points are affected by a spike.
    
    Args:
        data (np.array): Signal data
        spike_index (int): Index of the spike
        threshold (float): Threshold for detecting abnormal values
        
    Returns:
        int: Number of affected points
    """
    if spike_index >= len(data) - 1:
        return 5  # Default
        
    # Calculate differences
    diff_data = np.abs(np.diff(data))
    
    # Count consecutive points that deviate significantly
    count = 0
    for i in range(spike_index, min(spike_index + 50, len(data) - 1)):
        if diff_data[i] > threshold / 3:  # Lower threshold for consecutive points
            count += 1
        else:
            if count >= 3:  # Only count if we have at least 3 consecutive points
                break
            count = 0
    
    # Return at least 5 points, but not more than we found plus safety margin
    return max(5, count + 2)

def apply_spike_removal_to_curves(curves_dict):
    """
    Apply spike removal to multiple curves contained in a dictionary.
    
    Args:
        curves_dict (dict): Dictionary of curves with structure:
                           {'curve_name': (data_array, times_array)}
                           
    Returns:
        dict: Dictionary with same structure but with spikes removed
    """
    results = {}
    for name, (data, times) in curves_dict.items():
        if data is not None and len(data) > 0:
            app_logger.info(f"Applying spike removal to curve: {name}")
            cleaned_data = remove_periodic_spikes(data, times)
            results[name] = (cleaned_data, times)
        else:
            results[name] = (data, times)
    
    return results

# Example usage for ActionPotentialProcessor:
def process_curves_in_processor(processor):
    """
    Apply spike removal to all curves in an ActionPotentialProcessor instance.
    
    Args:
        processor: An ActionPotentialProcessor instance
        
    Returns:
        bool: True if processing was successful
    """
    try:
        # Create dictionary of curves to process
        curves = {}
        
        # Add orange curve if it exists
        if hasattr(processor, 'orange_curve') and processor.orange_curve is not None:
            curves['orange'] = (processor.orange_curve, processor.orange_curve_times)
            
        # Add normalized curve if it exists
        if hasattr(processor, 'normalized_curve') and processor.normalized_curve is not None:
            curves['normalized'] = (processor.normalized_curve, processor.normalized_curve_times)
            
        # Add average curve if it exists
        if hasattr(processor, 'average_curve') and processor.average_curve is not None:
            curves['average'] = (processor.average_curve, processor.average_curve_times)
            
        # Add modified curves if they exist
        if hasattr(processor, 'modified_hyperpol') and processor.modified_hyperpol is not None:
            curves['hyperpol'] = (processor.modified_hyperpol, processor.modified_hyperpol_times)
            
        if hasattr(processor, 'modified_depol') and processor.modified_depol is not None:
            curves['depol'] = (processor.modified_depol, processor.modified_depol_times)
        
        # Process all curves
        processed_curves = apply_spike_removal_to_curves(curves)
        
        # Update processor with cleaned curves
        for name, (cleaned_data, times) in processed_curves.items():
            if name == 'orange':
                processor.orange_curve = cleaned_data
            elif name == 'normalized':
                processor.normalized_curve = cleaned_data
            elif name == 'average':
                processor.average_curve = cleaned_data
            elif name == 'hyperpol':
                processor.modified_hyperpol = cleaned_data
            elif name == 'depol':
                processor.modified_depol = cleaned_data
        
        return True
        
    except Exception as e:
        app_logger.error(f"Error processing curves: {str(e)}")
        return False

# Function to fix the specific segments for purple curves
def apply_correct_segments(processor, n=None):
    """
    Apply correct segment indices for generating purple curves.
    This fixes the segment ranges to match the expected indices.
    
    Args:
        processor: An ActionPotentialProcessor instance
        n (int): Starting point (default: None, will use existing starting point)
        
    Returns:
        tuple: (modified_hyperpol, modified_hyperpol_times, modified_depol, modified_depol_times)
    """
    try:
        # Get starting point
        if n is None:
            if 'normalization_points' in processor.params:
                n = processor.params['normalization_points']['seg1'][0]
            else:
                n = 35  # Default

        # Fixed segment indices based on the logs
        depol_start = n + 835
        depol_end = n + 1034
        hyperpol_start = n + 1035
        hyperpol_end = n + 1234

        # Store the slice indices for later validation
        processor._depol_slice = (depol_start, depol_end)
        processor._hyperpol_slice = (hyperpol_start, hyperpol_end)

        # Make sure orange curve exists and is long enough
        if not hasattr(processor, 'orange_curve') or processor.orange_curve is None:
            app_logger.error("No orange curve available")
            return None, None, None, None
            
        if len(processor.orange_curve) < hyperpol_end:
            app_logger.error(f"Orange curve too short ({len(processor.orange_curve)} points)")
            return None, None, None, None

        # Extract segments
        depol_data = processor.orange_curve[depol_start:depol_end].copy()
        depol_times = processor.orange_curve_times[depol_start:depol_end]
        hyperpol_data = processor.orange_curve[hyperpol_start:hyperpol_end].copy()
        hyperpol_times = processor.orange_curve_times[hyperpol_start:hyperpol_end]

        # Make sure we have average curve
        if not hasattr(processor, 'average_curve') or processor.average_curve is None:
            app_logger.error("No average curve available")
            return None, None, None, None

        # Make average curve match segment length
        if len(processor.average_curve) < len(depol_data):
            pad_length = len(depol_data) - len(processor.average_curve)
            scaled_curve = np.pad(processor.average_curve, (0, pad_length), mode='edge')
        else:
            scaled_curve = processor.average_curve[:len(depol_data)]

        # Apply voltage scaling
        voltage_diff = abs(processor.params['V2'] - processor.params['V0'])
        scaled_curve = scaled_curve * voltage_diff

        # Apply modifications
        hyperpol_modified = hyperpol_data + scaled_curve
        depol_modified = depol_data - scaled_curve

        # Store results
        processor.modified_hyperpol = hyperpol_modified
        processor.modified_hyperpol_times = hyperpol_times
        processor.modified_depol = depol_modified
        processor.modified_depol_times = depol_times

        return (
            processor.modified_hyperpol,
            processor.modified_hyperpol_times,
            processor.modified_depol,
            processor.modified_depol_times
        )

    except Exception as e:
        app_logger.error(f"Error applying correct segments: {str(e)}")
        return None, None, None, None