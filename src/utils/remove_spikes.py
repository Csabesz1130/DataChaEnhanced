"""
Spike removal functionality for the Signal Analyzer application.
This module provides tools to detect and remove periodic spikes from
electrophysiological data without requiring complex external dependencies.
"""

import numpy as np
from src.utils.logger import app_logger
import tkinter as tk
from tkinter import ttk, messagebox

class RemoveSpikesToolbar:
    """
    Toolbar for spike removal tools in the action potential tab
    """
    def __init__(self, parent_tab, container_frame, main_app):
        """
        Initialize the spike removal toolbar.
        
        Args:
            parent_tab: Parent ActionPotentialTab instance
            container_frame: Frame to add toolbar to
            main_app: Reference to main application
        """
        self.parent = parent_tab
        self.main_app = main_app
        
        # Create frame for toolbar
        self.frame = ttk.LabelFrame(container_frame, text="Spike Removal Tools")
        self.frame.pack(fill='x', padx=5, pady=5)
        
        # Add description
        ttk.Label(
            self.frame,
            text="Remove periodic spikes from data signals that can impact analysis",
            wraplength=250,
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=(0, 5))
        
        # Create button frame
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        # Create remove button
        self.remove_btn = ttk.Button(
            button_frame,
            text="Remove Spikes",
            command=self.on_remove_spikes_click
        )
        self.remove_btn.pack(side='left', padx=5)
        
        # Create status indicator
        status_frame = ttk.Frame(self.frame)
        status_frame.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(status_frame, text="Status:").pack(side='left')
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side='left', padx=5)
        
        # Options frame
        options_frame = ttk.Frame(self.frame)
        options_frame.pack(fill='x', padx=5, pady=2)
        
        # Add threshold option
        ttk.Label(options_frame, text="Threshold:").pack(side='left')
        self.threshold_var = tk.DoubleVar(value=5.0)
        threshold_entry = ttk.Entry(options_frame, textvariable=self.threshold_var, width=5)
        threshold_entry.pack(side='left', padx=5)
        
        # Add apply to all curves checkbox
        self.apply_all = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, 
            text="Apply to all curves", 
            variable=self.apply_all
        ).pack(side='left', padx=10)

    def on_remove_spikes_click(self):
        """Handle click on the Remove Spikes button"""
        try:
            # First, check if data is available
            app = self.main_app
            if app.data is None:
                messagebox.showwarning("No Data", "Please load data first")
                return
                
            # Disable button while processing
            self.remove_btn.state(['disabled'])
            self.status_label.config(text="Removing spikes...")
            
            # Get threshold
            threshold = self.threshold_var.get()
            
            # Apply spike removal to original data
            app.data = remove_periodic_spikes(app.data, threshold_factor=threshold)
            
            # Apply to filtered data if it exists
            if app.filtered_data is not None:
                app.filtered_data = remove_periodic_spikes(app.filtered_data, threshold_factor=threshold)
                
            # If apply to all curves is checked, process other curves too
            if self.apply_all.get() and app.action_potential_processor is not None:
                # Use helper function to process all curves in the processor
                process_curves_in_processor(app.action_potential_processor, threshold)
                
            # Update plots
            app.update_plot()
            
            # Set status
            self.status_label.config(text="Spikes removed")
            
            # If analysis was already done, re-run it
            if (hasattr(app, 'action_potential_processor') and 
                app.action_potential_processor is not None and
                hasattr(self.parent, 'analyze_signal')):
                # Ask user if they want to re-run analysis
                if messagebox.askyesno("Reanalyze", 
                                      "Would you like to re-run the analysis with the cleaned data?"):
                    self.parent.analyze_signal()
                
            # Re-enable the button
            self.remove_btn.state(['!disabled'])
                
        except Exception as e:
            app_logger.error(f"Error removing spikes: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")
            self.remove_btn.state(['!disabled'])
            messagebox.showerror("Error", f"Failed to remove spikes: {str(e)}")

def remove_periodic_spikes(data_array, processor=None, period=None, auto_detect=True, 
                          threshold_factor=5.0):
    """
    Remove periodic spikes from data array.
    
    Args:
        data_array: NumPy array of data values
        processor: Optional processor instance for context
        period: Optional known period between spikes
        auto_detect: Whether to auto-detect the period
        threshold_factor: Multiplier for median difference to detect spikes
        
    Returns:
        Cleaned data array
    """
    if data_array is None or len(data_array) < 100:
        return data_array
        
    # Make a copy to avoid modifying the original
    cleaned_data = np.array(data_array.copy())
    
    # Calculate differences between consecutive points
    diffs = np.abs(np.diff(cleaned_data))
    
    # Calculate threshold for spike detection
    median_diff = np.median(diffs)
    threshold = threshold_factor * median_diff
    
    # Find spikes (large differences)
    spike_indices = np.where(diffs > threshold)[0] + 1
    
    if len(spike_indices) < 2:
        app_logger.info(f"No significant spikes detected (threshold={threshold:.2f})")
        return cleaned_data
        
    app_logger.info(f"Found {len(spike_indices)} potential spikes above threshold {threshold:.2f}")
    
    # Try to determine if there's a pattern in the spike positions
    if auto_detect and len(spike_indices) >= 3:
        # Calculate intervals between consecutive spikes
        intervals = np.diff(spike_indices)
        
        # Find most common interval if the intervals are similar
        if np.std(intervals) / np.mean(intervals) < 0.2:  # Low variation
            detected_period = int(np.median(intervals))
            app_logger.info(f"Detected regular spike pattern with period {detected_period}")
            period = detected_period
    
    # If no period detected or provided, just replace individual spikes
    if period is None or period <= 0:
        replaced_count = 0
        for idx in spike_indices:
            if idx > 1 and idx < len(cleaned_data) - 1:
                # Replace with average of surrounding points
                replacement = (cleaned_data[idx-2] + cleaned_data[idx-1]) / 2
                cleaned_data[idx] = replacement
                replaced_count += 1
                
        app_logger.info(f"Replaced {replaced_count} individual spikes")
        return cleaned_data
    
    # For periodic spikes, get the position of the first spike
    first_spike = spike_indices[0]
    app_logger.info(f"First spike at position {first_spike}, using period {period}")
    
    # Replace spikes at regular intervals
    replaced_count = 0
    for idx in range(first_spike, len(cleaned_data), period):
        if idx >= len(cleaned_data):
            break
            
        # Replace with average of surrounding points
        if idx > 1 and idx < len(cleaned_data) - 1:
            # Use points before the spike for replacement
            replacement = (cleaned_data[idx-2] + cleaned_data[idx-1]) / 2
            cleaned_data[idx] = replacement
            replaced_count += 1
    
    app_logger.info(f"Replaced {replaced_count} spikes at regular intervals")
    return cleaned_data

def process_curves_in_processor(processor, threshold_factor=5.0):
    """
    Apply spike removal to all relevant curves in an ActionPotentialProcessor
    
    Args:
        processor: ActionPotentialProcessor instance
        threshold_factor: Threshold factor for spike detection
    """
    if processor is None:
        return False
        
    # Track which curves were modified
    modified_curves = []
    
    # Orange curve
    if hasattr(processor, 'orange_curve') and processor.orange_curve is not None:
        processor.orange_curve = remove_periodic_spikes(
            processor.orange_curve, 
            processor=processor,
            threshold_factor=threshold_factor
        )
        modified_curves.append('orange')
        
    # Normalized curve
    if hasattr(processor, 'normalized_curve') and processor.normalized_curve is not None:
        processor.normalized_curve = remove_periodic_spikes(
            processor.normalized_curve,
            processor=processor,
            threshold_factor=threshold_factor
        )
        modified_curves.append('normalized')
        
    # Average curve
    if hasattr(processor, 'average_curve') and processor.average_curve is not None:
        processor.average_curve = remove_periodic_spikes(
            processor.average_curve,
            processor=processor,
            threshold_factor=threshold_factor
        )
        modified_curves.append('average')
        
    # Apply to purple curves last
    apply_correct_segments(processor, threshold_factor)
    
    app_logger.info(f"Applied spike removal to all curves: {', '.join(modified_curves)}")
    return True

def apply_correct_segments(processor, threshold_factor=5.0):
    """
    Apply spike removal to modified (purple) curve segments and re-apply
    average to peaks to ensure consistency.
    
    Args:
        processor: ActionPotentialProcessor instance
        threshold_factor: Threshold factor for spike detection
    """
    if processor is None:
        return False
        
    # Modified (purple) curves
    if hasattr(processor, 'modified_hyperpol') and processor.modified_hyperpol is not None:
        processor.modified_hyperpol = remove_periodic_spikes(
            processor.modified_hyperpol,
            processor=processor,
            threshold_factor=threshold_factor
        )
        
    if hasattr(processor, 'modified_depol') and processor.modified_depol is not None:
        processor.modified_depol = remove_periodic_spikes(
            processor.modified_depol,
            processor=processor,
            threshold_factor=threshold_factor
        )
        
    # Re-apply average to peaks to ensure consistency
    if (hasattr(processor, 'average_curve') and processor.average_curve is not None and
        hasattr(processor, 'apply_average_to_peaks')):
        app_logger.info("Re-applying average to peaks after spike removal")
        processor.apply_average_to_peaks()
        
    return True