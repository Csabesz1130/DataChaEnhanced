"""
Enhanced spike removal utility with higher sensitivity.
Use this version if you need to detect subtler spikes.
"""

import numpy as np
from scipy.signal import find_peaks
from src.utils.logger import app_logger

def remove_spikes_from_processor(processor):
    """
    Adaptively detect and remove spikes from action potential curves.
    This version uses more sensitive thresholds to detect subtler spikes.
    
    Args:
        processor: The ActionPotentialProcessor object containing the curves
    
    Returns:
        tuple: (bool success, dict results)
    """
    if not processor:
        app_logger.error("No processor provided to remove_spikes")
        return False, {"error": "No processor provided"}
    
    results = {"replaced_points": {}}
    total_replaced = 0
    
    try:
        # === 1. Process each curve if it exists ===
        curve_names = ['orange_curve', 'normalized_curve', 'average_curve', 
                      'modified_hyperpol', 'modified_depol']
        
        for curve_name in curve_names:
            if hasattr(processor, curve_name) and getattr(processor, curve_name) is not None:
                curve = getattr(processor, curve_name)
                if len(curve) < 3:  # Need at least 3 points for detection
                    continue
                
                # Find the corresponding time array if available
                time_array = None
                time_attr = f"{curve_name}_times"
                if hasattr(processor, time_attr) and getattr(processor, time_attr) is not None:
                    time_array = getattr(processor, time_attr)
                
                # Detect and replace spikes in this curve with more sensitive settings
                replaced_count, modified_curve = detect_and_replace_spikes(
                    curve, 
                    time_array=time_array, 
                    curve_name=curve_name,
                    sensitivity='high'  # Use high sensitivity mode
                )
                
                # Update the processor's curve with the modified one
                if replaced_count > 0:
                    setattr(processor, curve_name, modified_curve)
                    results["replaced_points"][curve_name] = replaced_count
                    total_replaced += replaced_count
        
        # Update the processor with the modified curves
        if total_replaced > 0:
            results["total_replaced"] = total_replaced
            app_logger.info(f"Successfully removed {total_replaced} spikes across all curves")
            return True, results
        else:
            app_logger.info("No problematic spikes found matching criteria")
            return True, {"message": "No problematic spikes found matching criteria"}
    
    except Exception as e:
        app_logger.error(f"Error removing spikes: {str(e)}")
        return False, {"error": str(e)}

def detect_and_replace_spikes(curve, time_array=None, curve_name="unknown", sensitivity='medium'):
    """
    Detect and replace spikes with adjustable sensitivity.
    
    Args:
        curve: numpy array containing the curve data
        time_array: optional corresponding time points
        curve_name: name of the curve for logging
        sensitivity: 'low', 'medium', or 'high' to adjust detection thresholds
        
    Returns:
        tuple: (count of replaced points, modified curve)
    """
    # Set thresholds based on sensitivity level
    if sensitivity == 'high':
        # Highly sensitive thresholds - will detect more subtle spikes
        rate_change_threshold = 4.0  # standard deviations (was 8)
        value_outlier_threshold = 4.0  # standard deviations (was 10)
        neighbor_diff_threshold = 3.0  # standard deviations (was 5)
    elif sensitivity == 'low':
        # Conservative thresholds - fewer false positives
        rate_change_threshold = 10.0
        value_outlier_threshold = 12.0
        neighbor_diff_threshold = 8.0
    else:  # 'medium' (default)
        rate_change_threshold = 8.0
        value_outlier_threshold = 10.0
        neighbor_diff_threshold = 5.0
    
    # Make a copy to avoid modifying the input
    cleaned_curve = np.copy(curve)
    replaced_count = 0
    
    try:
        # Calculate global statistics for adaptive thresholding
        global_median = np.median(curve)
        global_std = np.std(curve)
        
        # Method 1: Detect spikes using first-derivative (rate of change)
        diff = np.abs(np.diff(curve))
        median_diff = np.median(diff) 
        std_diff = np.std(diff)
        
        # Find spike candidates where rate of change is exceptionally high
        spike_candidates = np.where(diff > (median_diff + rate_change_threshold * std_diff))[0] + 1
        
        # Method 2: Find points that are extreme outliers in value
        diff_from_median = np.abs(curve - global_median)
        value_outliers = np.where(diff_from_median > value_outlier_threshold * global_std)[0]
        
        # Combine both methods for final spike detection
        all_candidates = np.unique(np.concatenate((spike_candidates, value_outliers)))
        
        # Process each candidate
        for i in all_candidates:
            if i <= 0 or i >= len(curve) - 1:
                continue  # Skip endpoints
                
            # Check if point is actually a spike by comparing to neighbors
            prev_point = curve[i-1]
            next_point = curve[i+1]
            current_point = curve[i]
            
            # A spike should be significantly different from both neighbors
            diff_prev = abs(current_point - prev_point)
            diff_next = abs(current_point - next_point)
            
            # Point is a spike if it's far from both neighbors and they are closer to each other
            neighbor_diff = abs(prev_point - next_point)
            if (diff_prev > neighbor_diff_threshold * std_diff and 
                diff_next > neighbor_diff_threshold * std_diff and 
                neighbor_diff < max(diff_prev, diff_next) * 0.5):
                
                # Replace with previous point (most effective for these specific spikes)
                old_value = cleaned_curve[i]
                cleaned_curve[i] = prev_point
                
                # Get time info for better logging
                time_info = ""
                if time_array is not None and i < len(time_array):
                    time_info = f" at time {time_array[i]*1000:.2f}ms"
                
                app_logger.info(f"Replaced spike in {curve_name} at index {i}{time_info}: "
                               f"{old_value:.2f} → {prev_point:.2f}")
                replaced_count += 1
                
        # Additional check for step-specific spikes in known curves with segmented structure
        if curve_name in ['orange_curve', 'normalized_curve'] and len(curve) > 100:
            # Check for spikes at likely voltage step transitions
            segment_size = estimate_segment_size(curve)
            
            if segment_size > 0:
                app_logger.debug(f"Estimated segment size for {curve_name}: {segment_size}")
                
                # Check positions at the start of each segment
                for start_idx in range(0, len(curve), segment_size):
                    end_check = min(start_idx + 5, len(curve) - 1)  # Check first few points of each segment
                    
                    for i in range(max(1, start_idx), end_check):
                        # Similar check as above but focused on segment boundaries
                        prev_point = cleaned_curve[i-1]
                        current_point = cleaned_curve[i]
                        
                        if i < len(cleaned_curve) - 1:
                            next_point = cleaned_curve[i+1]
                            
                            # Check for abrupt, isolated jumps
                            diff_prev = abs(current_point - prev_point)
                            diff_next = abs(current_point - next_point)
                            
                            if diff_prev > neighbor_diff_threshold * global_std and diff_next > neighbor_diff_threshold * global_std:
                                # Replace with previous point
                                old_value = cleaned_curve[i]
                                cleaned_curve[i] = prev_point
                                
                                time_info = ""
                                if time_array is not None and i < len(time_array):
                                    time_info = f" at time {time_array[i]*1000:.2f}ms"
                                
                                app_logger.info(f"Replaced segment boundary spike in {curve_name} at index {i}{time_info}: "
                                              f"{old_value:.2f} → {prev_point:.2f}")
                                replaced_count += 1
        
        return replaced_count, cleaned_curve
    
    except Exception as e:
        app_logger.error(f"Error detecting spikes in {curve_name}: {str(e)}")
        return 0, curve  # Return original curve on error

def estimate_segment_size(curve):
    """
    Estimate the segment size in a curve by finding repeating patterns.
    
    Args:
        curve: numpy array containing the curve data
        
    Returns:
        int: Estimated segment size or 0 if cannot determine
    """
    try:
        # Use autocorrelation to find repeating patterns
        if len(curve) < 50:
            return 0
            
        # Method 1: Look for repeating patterns in the first derivative
        diff = np.diff(curve)
        
        # Compute autocorrelation (simplified approach)
        corr = np.correlate(diff, diff, mode='full')
        corr = corr[len(corr)//2:]  # Take only the positive lags
        
        # Find peaks in the autocorrelation
        peaks, _ = find_peaks(corr, height=0.5*np.max(corr), distance=20)
        
        if len(peaks) > 0:
            # First peak after zero lag is likely the segment size
            segment_size = peaks[0]
            if 50 <= segment_size <= 300:  # Reasonable range for segments
                return segment_size
        
        # Method 2: Try common segment sizes directly
        common_sizes = [200, 100, 150, 250, 300]
        for size in common_sizes:
            if size < len(curve) // 2:
                # Check correlation between segments
                segments = [curve[i:i+size] for i in range(0, len(curve)-size, size)]
                if len(segments) >= 2:
                    correlations = []
                    for i in range(len(segments)-1):
                        corr = np.corrcoef(segments[i], segments[i+1])[0,1]
                        correlations.append(corr)
                    
                    avg_corr = np.mean(correlations)
                    if avg_corr > 0.7:  # Strong correlation between segments
                        return size
        
        # Fall back to common size if can't determine
        return 200
    
    except Exception as e:
        app_logger.error(f"Error estimating segment size: {str(e)}")
        return 200  # Default common segment size