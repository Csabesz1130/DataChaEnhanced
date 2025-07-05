# src/excel_export_enhanced_integration.py
"""
Enhanced Excel export integration that maintains backward compatibility
while adding automatic chart generation and manual curve fitting framework.

This file replaces the original excel_export.py import to seamlessly upgrade
the export functionality without breaking existing code.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from datetime import datetime
import os
from src.utils.logger import app_logger

# Import both old and new export systems
try:
    from src.excel_charted.enhanced_excel_export_with_charts import export_purple_curves_with_charts
    ENHANCED_EXPORT_AVAILABLE = True
    app_logger.info("Enhanced Excel export with charts is available")
except ImportError as e:
    ENHANCED_EXPORT_AVAILABLE = False
    app_logger.warning(f"Enhanced Excel export not available: {e}")

try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False
    app_logger.warning("xlsxwriter not available - install with: pip install xlsxwriter")

def add_excel_export_to_app(app):
    """
    Add enhanced Excel export functionality to the application.
    
    This function maintains backward compatibility while adding enhanced features
    when the enhanced export system is available.
    
    Args:
        app: The main application instance
    """
    app_logger.info("Adding enhanced Excel export functionality to application")
    
    # Create export frame if it doesn't exist
    if not hasattr(app, 'export_frame'):
        app.export_frame = ttk.LabelFrame(app.control_frame, text="Data Export")
        app.export_frame.pack(fill='x', padx=5, pady=5)
    
    # Clear any existing export buttons
    for widget in app.export_frame.winfo_children():
        widget.destroy()
    
    # Add enhanced export button
    export_button = ttk.Button(
        app.export_frame,
        text="Export Purple Curves to Excel (Enhanced)", 
        command=lambda: enhanced_export_purple_curves(app)
    )
    export_button.pack(pady=5)
    
    # Add status label for export feedback
    app.export_status_label = ttk.Label(
        app.export_frame, 
        text="Ready to export purple curve data",
        font=('TkDefaultFont', 9)
    )
    app.export_status_label.pack(pady=2)
    
    # Add requirements check button
    req_button = ttk.Button(
        app.export_frame,
        text="Check Export Requirements",
        command=check_export_requirements
    )
    req_button.pack(pady=2)
    
    app_logger.info("Enhanced Excel export functionality added successfully")

def enhanced_export_purple_curves(app):
    """
    Enhanced purple curve export with automatic chart generation and manual analysis framework.
    
    This function provides the enhanced export functionality while maintaining
    backward compatibility with the existing system.
    
    Args:
        app: The main application instance
    """
    try:
        app_logger.info("Starting enhanced purple curve export")
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Checking data availability...")
            app.master.update_idletasks()
        
        # Check if action potential processor exists and has data
        if not hasattr(app, 'action_potential_processor') or app.action_potential_processor is None:
            messagebox.showerror("Export Error", 
                               "No action potential analysis data available.\n"
                               "Please run the action potential analysis first.")
            return
        
        processor = app.action_potential_processor
        
        # Validate purple curve data
        if not _validate_purple_curve_data(processor):
            messagebox.showerror("Export Error",
                               "Purple curve data not available.\n"
                               "Please ensure the action potential analysis has been completed "
                               "and purple curves have been generated.")
            return
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Selecting export location...")
            app.master.update_idletasks()
        
        # Get export filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"purple_curves_analysis_{timestamp}.xlsx"
        
        filename = filedialog.asksaveasfilename(
            title="Save Enhanced Excel Analysis File",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialname=default_filename
        )
        
        if not filename:
            if hasattr(app, 'export_status_label'):
                app.export_status_label.config(text="Export cancelled")
            return
        
        # Check if enhanced export is available
        if ENHANCED_EXPORT_AVAILABLE and XLSXWRITER_AVAILABLE:
            # Use enhanced export with charts
            _perform_enhanced_export(app, processor, filename)
        else:
            # Fall back to basic export with upgrade message
            _perform_basic_export_with_upgrade_info(app, processor, filename)
            
    except Exception as e:
        app_logger.error(f"Error in enhanced purple curve export: {str(e)}")
        error_msg = f"Export failed: {str(e)}"
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Export failed")
        messagebox.showerror("Export Error", error_msg)

def _perform_enhanced_export(app, processor, filename):
    """Perform the enhanced export with charts and manual analysis framework."""
    try:
        app_logger.info("Performing enhanced export with automatic charts")
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Creating Excel file with charts...")
            app.master.update_idletasks()
        
        # Call the enhanced export function
        result_filename = export_purple_curves_with_charts(processor, filename)
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Export completed successfully!")
        
        # Show success message with enhanced features info
        success_msg = (
            f"Enhanced Excel file created successfully!\n\n"
            f"File: {result_filename}\n\n"
            f"✅ ENHANCED FEATURES INCLUDED:\n"
            f"• Automatic interactive charts (like your image!)\n"
            f"• Manual curve fitting framework\n"
            f"• Point selection tools for linear fitting\n"
            f"• Exponential parameter extraction\n"
            f"• Step-by-step analysis workflow\n"
            f"• Training data preparation tools\n\n"
            f"📊 Excel file contains 6 worksheets:\n"
            f"1. Purple_Curve_Data - Your data\n"
            f"2. Charts - Interactive charts 🆕\n"
            f"3. Hyperpol_Analysis - Manual fitting 🆕\n"
            f"4. Depol_Analysis - Manual fitting 🆕\n"
            f"5. Manual_Fitting_Tools - Solver helpers 🆕\n"
            f"6. Instructions - Complete workflow guide 🆕"
        )
        
        messagebox.showinfo("Enhanced Export Successful", success_msg)
        app_logger.info(f"Enhanced export completed: {result_filename}")
        
    except Exception as e:
        app_logger.error(f"Error in enhanced export: {str(e)}")
        raise

def _perform_basic_export_with_upgrade_info(app, processor, filename):
    """Perform basic export and show upgrade information."""
    try:
        app_logger.info("Performing basic export with upgrade information")
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Creating basic Excel file...")
            app.master.update_idletasks()
        
        # Create basic Excel file
        _create_basic_excel_export(processor, filename)
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Basic export completed")
        
        # Show upgrade message
        upgrade_msg = (
            f"Basic Excel file created: {filename}\n\n"
            f"⚠️ ENHANCED FEATURES NOT AVAILABLE\n\n"
            f"To get automatic charts and manual analysis framework:\n"
            f"1. Install xlsxwriter: pip install xlsxwriter\n"
            f"2. Restart the application\n"
            f"3. Re-export for enhanced features\n\n"
            f"Enhanced features include:\n"
            f"• Automatic interactive charts\n"
            f"• Manual curve fitting framework\n"
            f"• Point selection tools\n"
            f"• Exponential parameter extraction\n"
            f"• Training data preparation"
        )
        
        messagebox.showinfo("Basic Export Completed", upgrade_msg)
        app_logger.info(f"Basic export completed: {filename}")
        
    except Exception as e:
        app_logger.error(f"Error in basic export: {str(e)}")
        raise

def _create_basic_excel_export(processor, filename):
    """Create a basic Excel export without charts."""
    try:
        # Prepare data
        hyperpol_times_ms = processor.modified_hyperpol_times * 1000
        depol_times_ms = processor.modified_depol_times * 1000
        
        # Create DataFrames
        max_len = max(len(processor.modified_hyperpol), len(processor.modified_depol))
        
        # Pad shorter arrays
        hyperpol_data_padded = np.full(max_len, np.nan)
        hyperpol_times_padded = np.full(max_len, np.nan)
        depol_data_padded = np.full(max_len, np.nan)
        depol_times_padded = np.full(max_len, np.nan)
        
        hyperpol_data_padded[:len(processor.modified_hyperpol)] = processor.modified_hyperpol
        hyperpol_times_padded[:len(hyperpol_times_ms)] = hyperpol_times_ms
        depol_data_padded[:len(processor.modified_depol)] = processor.modified_depol
        depol_times_padded[:len(depol_times_ms)] = depol_times_ms
        
        # Create DataFrame
        df = pd.DataFrame({
            'Hyperpol_Time_ms': hyperpol_times_padded,
            'Hyperpol_Current_pA': hyperpol_data_padded,
            'Depol_Time_ms': depol_times_padded,
            'Depol_Current_pA': depol_data_padded
        })
        
        # Export to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Purple_Curve_Data', index=False)
            
            # Add a basic instructions sheet
            instructions_df = pd.DataFrame({
                'Instructions': [
                    'BASIC PURPLE CURVE DATA EXPORT',
                    '',
                    'This file contains the purple curve data from your analysis.',
                    'For enhanced features with automatic charts and manual fitting:',
                    '1. Install xlsxwriter: pip install xlsxwriter',
                    '2. Restart the application',
                    '3. Re-export for enhanced features',
                    '',
                    'Enhanced features include:',
                    '• Automatic interactive charts',
                    '• Manual curve fitting framework',
                    '• Point selection tools for linear fitting',
                    '• Exponential parameter extraction',
                    '• Step-by-step analysis workflow',
                    '• Training data preparation tools'
                ]
            })
            instructions_df.to_excel(writer, sheet_name='README', index=False)
        
        app_logger.info("Basic Excel export completed")
        
    except Exception as e:
        app_logger.error(f"Error creating basic Excel export: {str(e)}")
        raise

def _validate_purple_curve_data(processor):
    """Validate that purple curve data is available for export."""
    required_attrs = [
        'modified_hyperpol', 'modified_depol',
        'modified_hyperpol_times', 'modified_depol_times'
    ]
    
    for attr in required_attrs:
        if not hasattr(processor, attr):
            app_logger.error(f"Missing attribute: {attr}")
            return False
        
        value = getattr(processor, attr)
        if value is None:
            app_logger.error(f"Attribute {attr} is None")
            return False
        
        if isinstance(value, (list, np.ndarray)) and len(value) == 0:
            app_logger.error(f"Attribute {attr} is empty")
            return False
    
    app_logger.debug("Purple curve data validation passed")
    return True

def check_export_requirements():
    """Check and display export requirements and capabilities."""
    try:
        requirements_status = []
        
        # Check enhanced export availability
        if ENHANCED_EXPORT_AVAILABLE:
            requirements_status.append("✅ Enhanced export module: Available")
        else:
            requirements_status.append("❌ Enhanced export module: Not available")
        
        # Check xlsxwriter
        if XLSXWRITER_AVAILABLE:
            requirements_status.append("✅ xlsxwriter library: Available")
        else:
            requirements_status.append("❌ xlsxwriter library: Not installed")
        
        # Overall capability
        if ENHANCED_EXPORT_AVAILABLE and XLSXWRITER_AVAILABLE:
            capabilities = (
                "🎉 FULL ENHANCED EXPORT AVAILABLE\n\n"
                "Your exports will include:\n"
                "• Automatic interactive charts\n"
                "• Manual curve fitting framework\n"
                "• Point selection tools\n"
                "• Exponential parameter extraction\n"
                "• Step-by-step analysis workflow\n"
                "• Training data preparation tools"
            )
        else:
            capabilities = (
                "⚠️ BASIC EXPORT ONLY\n\n"
                "To enable enhanced features:\n"
                "1. Install xlsxwriter: pip install xlsxwriter\n"
                "2. Restart the application\n\n"
                "Enhanced features include automatic charts and manual analysis framework."
            )
        
        # Combine status and capabilities
        full_message = (
            "EXPORT REQUIREMENTS CHECK\n\n" +
            "\n".join(requirements_status) + "\n\n" +
            capabilities
        )
        
        messagebox.showinfo("Export Requirements", full_message)
        
    except Exception as e:
        app_logger.error(f"Error checking export requirements: {str(e)}")
        messagebox.showerror("Error", f"Failed to check requirements: {str(e)}")

# Additional utility functions for backward compatibility

def export_data_to_excel(data, filename, sheet_name="Data"):
    """
    Backward compatibility function for basic data export.
    
    Args:
        data: Data to export (DataFrame or dict)
        filename: Output filename
        sheet_name: Name of the Excel sheet
    """
    try:
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        app_logger.info(f"Data exported to Excel: {filename}")
        return filename
        
    except Exception as e:
        app_logger.error(f"Error exporting data to Excel: {str(e)}")
        raise

def get_export_capabilities():
    """
    Get information about current export capabilities.
    
    Returns:
        dict: Dictionary with capability information
    """
    return {
        'enhanced_export_available': ENHANCED_EXPORT_AVAILABLE,
        'xlsxwriter_available': XLSXWRITER_AVAILABLE,
        'charts_supported': ENHANCED_EXPORT_AVAILABLE and XLSXWRITER_AVAILABLE,
        'manual_fitting_supported': ENHANCED_EXPORT_AVAILABLE and XLSXWRITER_AVAILABLE
    }