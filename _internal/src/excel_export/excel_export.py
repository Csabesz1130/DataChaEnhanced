"""
Excel export module for the Signal Analyzer application.
This module handles exporting signal data and analysis results to Excel format,
supporting both direct integration (Scenario A) and regression-corrected integration
(Scenario B).
"""

import openpyxl
import numpy as np
from tkinter import filedialog, messagebox
from src.utils.logger import app_logger
from src.excel_export.regression_utils import compute_regression_params, apply_curve_correction
from src.excel_export.integration_calculator import (
    resample_data, 
    calculate_integral_scenario_a,
    calculate_integral_scenario_b
)

def export_to_excel(app):
    """
    Export signal data and analysis results to Excel.
    
    This function:
    1. Extracts and resamples the depolarization and hyperpolarization data
    2. Calculates integrals using either direct or regression-corrected methods
    3. Creates an Excel workbook with metadata, analysis results, and signal data
    4. Saves the workbook to a user-specified location
    
    Args:
        app: The main application instance with action_potential_processor
        
    Returns:
        bool: True if export successful, False otherwise
    """
    # Check if the app has an ActionPotentialProcessor with purple curves
    if (not hasattr(app, 'action_potential_processor') or 
        app.action_potential_processor is None):
        messagebox.showwarning("No Data", "Please run analysis first.")
        return False

    processor = app.action_potential_processor

    # Check if purple curves exist
    if (not hasattr(processor, 'modified_hyperpol') or
        not hasattr(processor, 'modified_depol') or
        processor.modified_hyperpol is None or
        processor.modified_depol is None):
        messagebox.showwarning("Missing Data", 
                               "Analysis does not contain required curves. "
                               "Please run analysis with 'Show Modified Peaks' enabled.")
        return False

    # Get the integral scaling factor
    integral_val = 1.0  # Default value
    if hasattr(processor, 'params'):
        voltage_diff = abs(processor.params.get('V2', 0) - processor.params.get('V0', -80))
        if voltage_diff > 0:
            integral_val = voltage_diff

    # Extract filename & V2 if available
    filename = "Unknown"
    if hasattr(app, 'current_file'):
        import os
        filename = os.path.basename(app.current_file)
    v2 = processor.params.get('V2', 'N/A') if hasattr(processor, 'params') else 'N/A'

    # Prompt for the save location
    filepath = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        title="Export Analysis to Excel"
    )
    if not filepath:
        return False  # User canceled

    # Log export start
    app_logger.info(f"Starting Excel export: {filepath}")

    try:
        # Resample data to 0-11.5ms in 0.5ms steps (24 points)
        # or 0-100ms in 0.5ms steps (200 points) if using full curve
        use_full_curve = False  # Change to True to use 200 points (100ms)
        
        duration_ms = 100 if use_full_curve else 12
        
        # Determine appropriate start times for the segments
        # These should be adjusted based on the actual data
        # Find a flat/stable region for beginning of each curve
        hyperpol_start_ms = 0  # Will be converted to ms from s in resample_data
        depol_start_ms = 0     # Will be converted to ms from s in resample_data
        
        if hasattr(processor, 'modified_hyperpol_times') and processor.modified_hyperpol_times is not None:
            hyperpol_start_ms = processor.modified_hyperpol_times[0] * 1000
            
        if hasattr(processor, 'modified_depol_times') and processor.modified_depol_times is not None:
            depol_start_ms = processor.modified_depol_times[0] * 1000
            
        app_logger.info(f"Using start times: hyperpol={hyperpol_start_ms:.2f}ms, "
                        f"depol={depol_start_ms:.2f}ms")

        # Resample the data
        hyperpol_time_ms, hyperpol_current_pA = resample_data(
            processor.modified_hyperpol_times, 
            processor.modified_hyperpol,
            start_time_ms=hyperpol_start_ms,
            duration_ms=duration_ms
        )
        
        depol_time_ms, depol_current_pA = resample_data(
            processor.modified_depol_times, 
            processor.modified_depol,
            start_time_ms=depol_start_ms,
            duration_ms=duration_ms
        )
        
        # Get scenario selection from app
        # You can add checkboxes or auto-detection logic in the main app
        use_regression_hyperpol = getattr(app, 'use_regression_hyperpol', False)
        use_regression_depol = getattr(app, 'use_regression_depol', False)
        
        # Calculate integrals for both curves
        # For Scenario A (direct integration)
        hyperpol_integral_a = calculate_integral_scenario_a(
            hyperpol_current_pA, hyperpol_time_ms, integral_val)
        depol_integral_a = calculate_integral_scenario_a(
            depol_current_pA, depol_time_ms, integral_val)
            
        # For Scenario B (with regression correction)
        # For hyperpolarization
        hyperpol_slope = hyperpol_intercept = 0
        depol_slope = depol_intercept = 0
        
        if use_regression_hyperpol:
            # Define fitting region (e.g., first third of the curve)
            fit_start = 0
            fit_end = hyperpol_time_ms[-1] / 3
            hyperpol_integral_b, hyperpol_slope, hyperpol_intercept = calculate_integral_scenario_b(
                hyperpol_time_ms, hyperpol_current_pA, integral_val,
                fit_start_ms=fit_start, fit_end_ms=fit_end
            )
        else:
            hyperpol_integral_b = hyperpol_integral_a
            
        # For depolarization
        if use_regression_depol:
            # Define fitting region (e.g., first third of the curve)
            fit_start = 0
            fit_end = depol_time_ms[-1] / 3
            depol_integral_b, depol_slope, depol_intercept = calculate_integral_scenario_b(
                depol_time_ms, depol_current_pA, integral_val,
                fit_start_ms=fit_start, fit_end_ms=fit_end
            )
        else:
            depol_integral_b = depol_integral_a
            
        # Choose which integral values to use based on scenario selection
        hyperpol_integral = hyperpol_integral_b if use_regression_hyperpol else hyperpol_integral_a
        depol_integral = depol_integral_b if use_regression_depol else depol_integral_a
        
        # Create linear capacitance value
        linear_capacitance = abs(hyperpol_integral - depol_integral)
        
        app_logger.info(f"Calculated integrals - Hyperpol: {hyperpol_integral:.6f}, "
                        f"Depol: {depol_integral:.6f}, Capacitance: {linear_capacitance:.6f}")

        # Create Excel workbook
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Analysis Results"

        # Write metadata
        ws.append(["Filename", "V2 (mV)", "Integral Value", "Hyperpol Integral", 
                   "Depol Integral", "Linear Capacitance"])
        ws.append([filename, v2, integral_val, hyperpol_integral, 
                   depol_integral, linear_capacitance])
        
        # Add regression parameters if used
        if use_regression_hyperpol or use_regression_depol:
            ws.append([])
            ws.append(["Regression Parameters"])
            ws.append(["Curve", "Scenario", "Slope", "Intercept"])
            
            if use_regression_hyperpol:
                ws.append(["Hyperpolarization", "B", hyperpol_slope, hyperpol_intercept])
            else:
                ws.append(["Hyperpolarization", "A", "N/A", "N/A"])
                
            if use_regression_depol:
                ws.append(["Depolarization", "B", depol_slope, depol_intercept])
            else:
                ws.append(["Depolarization", "A", "N/A", "N/A"])

        # Add space before data
        ws.append([])

        # Write hyperpolarization data
        ws.append(["Hyperpolarization"])
        ws.append(["Time (ms)", "Current (pA)"])
        for t, c in zip(hyperpol_time_ms, hyperpol_current_pA):
            ws.append([t, c])

        # Add space before next section
        ws.append([])

        # Write depolarization data
        ws.append(["Depolarization"])
        ws.append(["Time (ms)", "Current (pA)"])
        for t, c in zip(depol_time_ms, depol_current_pA):
            ws.append([t, c])

        # Save the workbook
        wb.save(filepath)
        app_logger.info(f"Excel export completed successfully: {filepath}")
        
        # Show success message
        messagebox.showinfo("Export Successful", 
                           f"Data exported to:\n{filepath}\n\n"
                           f"Hyperpol: {hyperpol_integral:.4f} pC\n"
                           f"Depol: {depol_integral:.4f} pC\n"
                           f"Linear Capacitance: {linear_capacitance:.4f} nF")
        
        # Update results in app if needed
        if hasattr(app, 'update_excel_results'):
            app.update_excel_results(hyperpol_integral, depol_integral, linear_capacitance)
            
        return True
        
    except Exception as e:
        app_logger.error(f"Error exporting to Excel: {str(e)}")
        messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
        return False

def add_excel_export_to_app(app):
    """
    Add Excel export functionality to the app's interface.
    
    This function should be called during app initialization to add the
    Excel export button to the toolbar.
    
    Args:
        app: The main application instance
    """
    from tkinter import ttk
    
    # Find the appropriate toolbar or menu to add the button
    if hasattr(app, 'toolbar_frame'):
        # Create Excel export button
        export_excel_btn = ttk.Button(
            app.toolbar_frame,
            text="Export to Excel",
            command=lambda: export_to_excel(app)
        )
        export_excel_btn.pack(side='left', padx=2)
        
        app_logger.info("Added Excel export button to toolbar")
    else:
        app_logger.warning("Could not find toolbar_frame to add Excel export button")

def update_excel_results(app, hyperpol_integral, depol_integral, linear_capacitance):
    """
    Update the app's UI with the Excel export results.
    
    This function can be implemented in the main app class and called after
    successful export to update any result displays.
    
    Args:
        app: The main application instance
        hyperpol_integral (float): Hyperpolarization integral value
        depol_integral (float): Depolarization integral value
        linear_capacitance (float): Linear capacitance value
    """
    # Example implementation - adjust based on your app's structure
    if hasattr(app, 'action_potential_tab'):
        tab = app.action_potential_tab
        
        if hasattr(tab, 'hyperpol_result') and hasattr(tab, 'depol_result'):
            tab.hyperpol_result.set(f"{hyperpol_integral:.4f} pC")
            tab.depol_result.set(f"{depol_integral:.4f} pC")
            
        if hasattr(tab, 'capacitance_result'):
            tab.capacitance_result.set(f"{linear_capacitance:.4f} nF")
            
        app_logger.info("Updated UI with Excel export results")