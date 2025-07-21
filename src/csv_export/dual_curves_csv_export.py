# src/csv_export/dual_curves_csv_export.py
"""
CSV export functionality for both purple and red curves.
This provides a simple CSV export option as an alternative to Excel.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
from tkinter import filedialog, messagebox
from src.utils.logger import app_logger

def export_dual_curves_to_csv(processor, app, filename=None):
    """
    Export both purple and red curves to CSV format.
    
    Args:
        processor: ActionPotentialProcessor instance with purple curve data
        app: Main application instance with red curve data
        filename: Optional filename, will prompt user if not provided
        
    Returns:
        str: Path to the created CSV file
    """
    try:
        app_logger.info("Starting dual curves CSV export")
        
        # Get filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"dual_curves_data_{timestamp}.csv"
            
            filename = filedialog.asksaveasfilename(
                title="Save Dual Curves CSV File",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=default_filename
            )
            
            if not filename:
                app_logger.info("CSV export cancelled by user")
                return None
        
        # Validate data availability
        if not _validate_dual_curve_data_csv(processor, app):
            raise ValueError("Missing purple or red curve data for CSV export")
        
        # Prepare purple curve data
        hyperpol_times_ms = processor.modified_hyperpol_times * 1000
        depol_times_ms = processor.modified_depol_times * 1000
        
        # Prepare red curve data (extract relevant sections)
        red_time_ms = app.time_data * 1000
        red_data = app.filtered_data
        
        # Extract red curve sections corresponding to purple curves
        if hasattr(processor, '_hyperpol_slice') and hasattr(processor, '_depol_slice'):
            hyperpol_slice = processor._hyperpol_slice
            depol_slice = processor._depol_slice
        else:
            # Default slices
            hyperpol_slice = (1035, 1235)
            depol_slice = (835, 1035)
        
        # Extract red curve sections
        red_hyperpol_data = red_data[hyperpol_slice[0]:hyperpol_slice[1]]
        red_hyperpol_times = red_time_ms[hyperpol_slice[0]:hyperpol_slice[1]]
        red_depol_data = red_data[depol_slice[0]:depol_slice[1]]
        red_depol_times = red_time_ms[depol_slice[0]:depol_slice[1]]
        
        # Create unified data structure
        max_len = max(len(processor.modified_hyperpol), len(processor.modified_depol),
                     len(red_hyperpol_data), len(red_depol_data))
        
        # Pad arrays to same length
        def pad_array(arr, target_len):
            padded = np.full(target_len, np.nan)
            padded[:len(arr)] = arr
            return padded
        
        # Create DataFrame with all data
        df = pd.DataFrame({
            'Purple_Hyperpol_Time_ms': pad_array(hyperpol_times_ms, max_len),
            'Purple_Hyperpol_Current_pA': pad_array(processor.modified_hyperpol, max_len),
            'Purple_Depol_Time_ms': pad_array(depol_times_ms, max_len),
            'Purple_Depol_Current_pA': pad_array(processor.modified_depol, max_len),
            'Red_Hyperpol_Time_ms': pad_array(red_hyperpol_times, max_len),
            'Red_Hyperpol_Current_pA': pad_array(red_hyperpol_data, max_len),
            'Red_Depol_Time_ms': pad_array(red_depol_times, max_len),
            'Red_Depol_Current_pA': pad_array(red_depol_data, max_len)
        })
        
        # Add metadata header
        metadata_lines = [
            f"# Dual Curves Data Export",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Purple curves: Modified/processed data",
            f"# Red curves: Original filtered/denoised data",
            f"# Data points: {max_len}",
            f"#"
        ]
        
        # Write metadata and data
        with open(filename, 'w') as f:
            # Write metadata
            for line in metadata_lines:
                f.write(line + '\n')
            
            # Write DataFrame
            df.to_csv(f, index=False)
        
        app_logger.info(f"Dual curves CSV export completed: {filename}")
        return filename
        
    except Exception as e:
        app_logger.error(f"Error in dual curves CSV export: {str(e)}")
        raise

def export_purple_curves_only_csv(processor, filename=None):
    """
    Export only purple curves to CSV format.
    
    Args:
        processor: ActionPotentialProcessor instance with purple curve data
        filename: Optional filename, will prompt user if not provided
        
    Returns:
        str: Path to the created CSV file
    """
    try:
        app_logger.info("Starting purple curves only CSV export")
        
        # Get filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"purple_curves_data_{timestamp}.csv"
            
            filename = filedialog.asksaveasfilename(
                title="Save Purple Curves CSV File",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=default_filename
            )
            
            if not filename:
                app_logger.info("Purple curves CSV export cancelled by user")
                return None
        
        # Validate purple curve data
        if not _validate_purple_curve_data_csv(processor):
            raise ValueError("Missing purple curve data for CSV export")
        
        # Prepare data
        hyperpol_times_ms = processor.modified_hyperpol_times * 1000
        depol_times_ms = processor.modified_depol_times * 1000
        
        # Create unified data structure
        max_len = max(len(processor.modified_hyperpol), len(processor.modified_depol))
        
        # Pad arrays
        def pad_array(arr, target_len):
            padded = np.full(target_len, np.nan)
            padded[:len(arr)] = arr
            return padded
        
        # Create DataFrame
        df = pd.DataFrame({
            'Hyperpol_Time_ms': pad_array(hyperpol_times_ms, max_len),
            'Hyperpol_Current_pA': pad_array(processor.modified_hyperpol, max_len),
            'Depol_Time_ms': pad_array(depol_times_ms, max_len),
            'Depol_Current_pA': pad_array(processor.modified_depol, max_len)
        })
        
        # Add metadata header
        metadata_lines = [
            f"# Purple Curves Data Export",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Contains modified/processed curve data",
            f"# Data points: {max_len}",
            f"#"
        ]
        
        # Write metadata and data
        with open(filename, 'w') as f:
            # Write metadata
            for line in metadata_lines:
                f.write(line + '\n')
            
            # Write DataFrame
            df.to_csv(f, index=False)
        
        app_logger.info(f"Purple curves CSV export completed: {filename}")
        return filename
        
    except Exception as e:
        app_logger.error(f"Error in purple curves CSV export: {str(e)}")
        raise

def _validate_dual_curve_data_csv(processor, app):
    """Validate that both purple and red curve data are available for CSV export."""
    # Check purple curve data
    purple_valid = (hasattr(processor, 'modified_hyperpol') and 
                   hasattr(processor, 'modified_depol') and
                   processor.modified_hyperpol is not None and 
                   processor.modified_depol is not None and
                   len(processor.modified_hyperpol) > 0 and 
                   len(processor.modified_depol) > 0)
    
    # Check red curve data
    red_valid = (hasattr(app, 'filtered_data') and 
                hasattr(app, 'time_data') and
                app.filtered_data is not None and 
                app.time_data is not None and
                len(app.filtered_data) > 0 and 
                len(app.time_data) > 0)
    
    app_logger.info(f"CSV data validation - Purple curves: {purple_valid}, Red curves: {red_valid}")
    return purple_valid and red_valid

def _validate_purple_curve_data_csv(processor):
    """Validate that purple curve data is available for CSV export."""
    return (hasattr(processor, 'modified_hyperpol') and 
            hasattr(processor, 'modified_depol') and
            processor.modified_hyperpol is not None and 
            processor.modified_depol is not None and
            len(processor.modified_hyperpol) > 0 and 
            len(processor.modified_depol) > 0)

def add_csv_export_buttons(app):
    """
    Add CSV export buttons to the application.
    This can be used as an alternative or addition to Excel export.
    """
    try:
        from tkinter import ttk
        
        # Create CSV export frame
        if not hasattr(app, 'csv_export_frame'):
            app.csv_export_frame = ttk.LabelFrame(app.control_frame, text="CSV Export")
            app.csv_export_frame.pack(fill='x', padx=5, pady=5)
        
        # Purple curves only CSV button
        purple_csv_button = ttk.Button(
            app.csv_export_frame,
            text="Export Purple Curves to CSV",
            command=lambda: _export_purple_csv_wrapper(app)
        )
        purple_csv_button.pack(pady=2)
        
        # Dual curves CSV button
        dual_csv_button = ttk.Button(
            app.csv_export_frame,
            text="Export Both Curves to CSV",
            command=lambda: _export_dual_csv_wrapper(app)
        )
        dual_csv_button.pack(pady=2)
        
        app_logger.info("CSV export buttons added successfully")
        
    except Exception as e:
        app_logger.error(f"Error adding CSV export buttons: {str(e)}")
        messagebox.showerror("Error", f"Failed to add CSV export buttons: {str(e)}")

def _export_purple_csv_wrapper(app):
    """Wrapper function for purple curves CSV export with error handling."""
    try:
        processor = app.action_potential_processor
        if not processor:
            messagebox.showwarning("No Data", "Please run analysis to generate purple curves first.")
            return
        
        filename = export_purple_curves_only_csv(processor)
        if filename:
            messagebox.showinfo("Export Success", f"Purple curves exported to CSV:\n{filename}")
            
    except Exception as e:
        app_logger.error(f"Error in purple curves CSV export wrapper: {str(e)}")
        messagebox.showerror("Export Error", f"Failed to export purple curves to CSV: {str(e)}")

def _export_dual_csv_wrapper(app):
    """Wrapper function for dual curves CSV export with error handling."""
    try:
        processor = app.action_potential_processor
        
        if not processor:
            messagebox.showwarning("No Data", "Please run analysis to generate purple curves first.")
            return
        
        if not hasattr(app, 'filtered_data') or app.filtered_data is None:
            messagebox.showwarning("No Filtered Data", 
                                 "No red curves (filtered data) available. Please load and filter data first.")
            return
        
        filename = export_dual_curves_to_csv(processor, app)
        if filename:
            messagebox.showinfo("Export Success", 
                              f"Both purple and red curves exported to CSV:\n{filename}")
            
    except Exception as e:
        app_logger.error(f"Error in dual curves CSV export wrapper: {str(e)}")
        messagebox.showerror("Export Error", f"Failed to export dual curves to CSV: {str(e)}")