# src/excel_charted/dual_curves_export_integration.py
"""
Dual curves Excel export integration that provides both purple and red curve export functionality.
This extends the existing export system to include original filtered data alongside processed data.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from datetime import datetime
import os
from src.utils.logger import app_logger

# Import both export systems
try:
    from src.excel_charted.enhanced_excel_export_with_charts_dual import export_both_curves_with_charts
    DUAL_EXPORT_AVAILABLE = True
    app_logger.info("Dual curves Excel export with charts is available")
except ImportError as e:
    DUAL_EXPORT_AVAILABLE = False
    app_logger.warning(f"Dual curves Excel export not available: {e}")

try:
    from src.excel_charted.enhanced_excel_export_with_charts import export_purple_curves_with_charts
    PURPLE_EXPORT_AVAILABLE = True
except ImportError as e:
    PURPLE_EXPORT_AVAILABLE = False
    app_logger.warning(f"Purple curves Excel export not available: {e}")

try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False
    app_logger.warning("xlsxwriter not available - install with: pip install xlsxwriter")

def add_dual_excel_export_to_app(app):
    """
    Add dual curves Excel export functionality to the application.
    Provides options for exporting purple curves only or both purple and red curves.
    
    Args:
        app: The main application instance
    """
    app_logger.info("Adding dual curves Excel export functionality to application")
    
    # Create export frame if it doesn't exist
    if not hasattr(app, 'export_frame'):
        app.export_frame = ttk.LabelFrame(app.control_frame, text="Data Export")
        app.export_frame.pack(fill='x', padx=5, pady=5)
    
    # Clear any existing export buttons
    for widget in app.export_frame.winfo_children():
        widget.destroy()
    
    # Add purple curves only export button
    purple_button = ttk.Button(
        app.export_frame,
        text="Export Purple Curves Only", 
        command=lambda: export_purple_curves_only(app)
    )
    purple_button.pack(pady=2)
    
    # Add dual curves export button (NEW FEATURE)
    dual_button = ttk.Button(
        app.export_frame,
        text="Export Both Purple & Red Curves (Enhanced)", 
        command=lambda: export_dual_curves_enhanced(app)
    )
    dual_button.pack(pady=2)
    
    # Add status label for export feedback
    app.export_status_label = ttk.Label(
        app.export_frame, 
        text="Ready to export curve data",
        font=('TkDefaultFont', 9)
    )
    app.export_status_label.pack(pady=2)
    
    # Add requirements check button
    req_button = ttk.Button(
        app.export_frame,
        text="Check Export Requirements",
        command=check_dual_export_requirements
    )
    req_button.pack(pady=2)
    
    app_logger.info("Dual curves Excel export functionality added successfully")

def export_purple_curves_only(app):
    """Export only purple curves using the existing system."""
    try:
        processor = app.action_potential_processor
        if not processor or not hasattr(processor, 'modified_hyperpol'):
            messagebox.showwarning("No Data", "Please run analysis to generate purple curves first.")
            return
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Selecting export location...")
            app.master.update_idletasks()
        
        # Get export filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"purple_curves_only_{timestamp}.xlsx"
        
        filename = filedialog.asksaveasfilename(
            title="Save Purple Curves Excel File",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialname=default_filename
        )
        
        if not filename:
            if hasattr(app, 'export_status_label'):
                app.export_status_label.config(text="Export cancelled")
            return
        
        # Check if purple export is available
        if PURPLE_EXPORT_AVAILABLE and XLSXWRITER_AVAILABLE:
            # Use enhanced export with charts (purple only)
            _perform_purple_only_export(app, processor, filename)
        else:
            # Fall back to basic export
            _perform_basic_purple_export(app, processor, filename)
            
    except Exception as e:
        app_logger.error(f"Error in purple curves export: {str(e)}")
        error_msg = f"Export failed: {str(e)}"
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Export failed")
        messagebox.showerror("Export Error", error_msg)

def export_dual_curves_enhanced(app):
    """
    Export both purple and red curves with enhanced Excel functionality.
    This is the NEW FEATURE requested by the user.
    """
    try:
        processor = app.action_potential_processor
        
        # Validate purple curve data
        if not processor or not hasattr(processor, 'modified_hyperpol'):
            messagebox.showwarning("No Data", "Please run analysis to generate purple curves first.")
            return
        
        # Validate red curve data
        if not hasattr(app, 'filtered_data') or app.filtered_data is None:
            messagebox.showwarning("No Filtered Data", 
                                 "No red curves (filtered data) available. Please load and filter data first.")
            return
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Selecting export location...")
            app.master.update_idletasks()
        
        # Get export filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"dual_curves_analysis_{timestamp}.xlsx"
        
        filename = filedialog.asksaveasfilename(
            title="Save Dual Curves Excel Analysis File",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialname=default_filename
        )
        
        if not filename:
            if hasattr(app, 'export_status_label'):
                app.export_status_label.config(text="Export cancelled")
            return
        
        # Check if dual export is available
        if DUAL_EXPORT_AVAILABLE and XLSXWRITER_AVAILABLE:
            # Use enhanced dual export with charts (NEW FEATURE)
            _perform_dual_export(app, processor, filename)
        else:
            # Fall back to basic dual export
            _perform_basic_dual_export(app, processor, filename)
            
    except Exception as e:
        app_logger.error(f"Error in dual curves export: {str(e)}")
        error_msg = f"Export failed: {str(e)}"
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Export failed")
        messagebox.showerror("Export Error", error_msg)

def _perform_purple_only_export(app, processor, filename):
    """Perform the enhanced export with charts (purple curves only)."""
    try:
        app_logger.info("Performing enhanced export with automatic charts (purple only)")
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Creating Excel file with purple curves...")
            app.master.update_idletasks()
        
        # Call the enhanced export function (existing)
        result_filename = export_purple_curves_with_charts(processor, filename)
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Purple curves export completed!")
        
        # Show success message
        success_msg = (
            f"Purple Curves Excel file created successfully!\n\n"
            f"File: {result_filename}\n\n"
            f"Contains only purple curves (modified data) with interactive charts."
        )
        
        messagebox.showinfo("Export Success", success_msg)
        
    except Exception as e:
        app_logger.error(f"Error creating purple curves Excel export: {str(e)}")
        raise

def _perform_dual_export(app, processor, filename):
    """Perform the enhanced dual export with charts (NEW FEATURE)."""
    try:
        app_logger.info("Performing enhanced dual export with automatic charts")
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Creating Excel file with both curve datasets...")
            app.master.update_idletasks()
        
        # Call the enhanced dual export function (NEW)
        result_filename = export_both_curves_with_charts(processor, app, filename)
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Dual curves export completed successfully!")
        
        # Show success message with enhanced features info
        success_msg = (
            f"Dual Curves Excel file created successfully!\n\n"
            f"File: {result_filename}\n\n"
            f"✅ ENHANCED FEATURES INCLUDED:\n"
            f"• Both Purple curves (modified data) AND Red curves (filtered data)\n"
            f"• Side-by-side comparison charts\n"
            f"• Statistical analysis of both datasets\n"
            f"• Manual curve fitting framework\n"
            f"• Automatic interactive charts\n"
            f"• Processing effects analysis\n\n"
            f"📊 Excel file contains 6 worksheets:\n"
            f"1. Dual_Curve_Data - Both datasets\n"
            f"2. Interactive_Charts - Visual comparisons\n"
            f"3. Hyperpol_Analysis - Hyperpol curves comparison\n"
            f"4. Depol_Analysis - Depol curves comparison\n"
            f"5. Manual_Fitting - Fitting framework\n"
            f"6. Instructions - Usage guide\n\n"
            f"🔬 Perfect for validating processing effects!"
        )
        
        messagebox.showinfo("Export Success", success_msg)
        
    except Exception as e:
        app_logger.error(f"Error creating dual curves Excel export: {str(e)}")
        raise

def _perform_basic_dual_export(app, processor, filename):
    """Perform basic dual export without charts when xlsxwriter is not available."""
    try:
        app_logger.info("Performing basic dual export (no charts)")
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Creating basic Excel file with both datasets...")
            app.master.update_idletasks()
        
        # Prepare data
        hyperpol_times_ms = processor.modified_hyperpol_times * 1000
        depol_times_ms = processor.modified_depol_times * 1000
        
        # Get red curve sections
        if hasattr(processor, '_hyperpol_slice') and hasattr(processor, '_depol_slice'):
            hyperpol_slice = processor._hyperpol_slice
            depol_slice = processor._depol_slice
        else:
            hyperpol_slice = (1035, 1235)
            depol_slice = (835, 1035)
        
        red_time_ms = app.time_data * 1000
        red_hyperpol_data = app.filtered_data[hyperpol_slice[0]:hyperpol_slice[1]]
        red_hyperpol_times = red_time_ms[hyperpol_slice[0]:hyperpol_slice[1]]
        red_depol_data = app.filtered_data[depol_slice[0]:depol_slice[1]]
        red_depol_times = red_time_ms[depol_slice[0]:depol_slice[1]]
        
        # Create unified DataFrame
        max_len = max(len(processor.modified_hyperpol), len(processor.modified_depol),
                     len(red_hyperpol_data), len(red_depol_data))
        
        # Pad arrays
        def pad_array(arr, target_len):
            padded = np.full(target_len, np.nan)
            padded[:len(arr)] = arr
            return padded
        
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
        
        # Export to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Dual_Curve_Data', index=False)
            
            # Add instructions sheet
            instructions_df = pd.DataFrame({
                'Instructions': [
                    'BASIC DUAL CURVES DATA EXPORT',
                    '',
                    'This file contains both purple and red curve data.',
                    'Purple curves: Modified/processed data',
                    'Red curves: Original filtered/denoised data',
                    '',
                    'For enhanced features with automatic charts:',
                    '1. Install xlsxwriter: pip install xlsxwriter',
                    '2. Restart the application',
                    '3. Re-export for enhanced features',
                    '',
                    'Enhanced features include:',
                    '• Side-by-side comparison charts',
                    '• Statistical analysis of both datasets',
                    '• Processing effects analysis',
                    '• Manual curve fitting framework'
                ]
            })
            instructions_df.to_excel(writer, sheet_name='README', index=False)
        
        # Update status
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Basic dual export completed!")
        
        success_msg = (
            f"Basic Dual Curves Excel file created!\n\n"
            f"File: {filename}\n\n"
            f"Contains both Purple and Red curve data.\n"
            f"Install xlsxwriter for enhanced features with charts."
        )
        
        messagebox.showinfo("Export Success", success_msg)
        
        app_logger.info("Basic dual Excel export completed")
        
    except Exception as e:
        app_logger.error(f"Error creating basic dual Excel export: {str(e)}")
        raise

def _perform_basic_purple_export(app, processor, filename):
    """Perform basic purple export without charts when xlsxwriter is not available."""
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
            
            # Add instructions sheet
            instructions_df = pd.DataFrame({
                'Instructions': [
                    'BASIC PURPLE CURVE DATA EXPORT',
                    '',
                    'This file contains only the purple curve data.',
                    'For enhanced features with charts:',
                    '1. Install xlsxwriter: pip install xlsxwriter',
                    '2. Restart the application',
                    '3. Use enhanced export options'
                ]
            })
            instructions_df.to_excel(writer, sheet_name='README', index=False)
        
        if hasattr(app, 'export_status_label'):
            app.export_status_label.config(text="Basic purple export completed!")
        
        messagebox.showinfo("Export Success", f"Basic Purple Curves file created: {filename}")
        
    except Exception as e:
        app_logger.error(f"Error creating basic purple Excel export: {str(e)}")
        raise

def check_dual_export_requirements():
    """Check and display requirements for dual curves export."""
    try:
        requirements_status = []
        
        # Check xlsxwriter
        if XLSXWRITER_AVAILABLE:
            requirements_status.append("✅ xlsxwriter: Available")
        else:
            requirements_status.append("❌ xlsxwriter: Missing (install with: pip install xlsxwriter)")
        
        # Check pandas
        try:
            import pandas
            requirements_status.append("✅ pandas: Available")
        except ImportError:
            requirements_status.append("❌ pandas: Missing (install with: pip install pandas)")
        
        # Check numpy
        try:
            import numpy
            requirements_status.append("✅ numpy: Available")
        except ImportError:
            requirements_status.append("❌ numpy: Missing (install with: pip install numpy)")
        
        # Check export modules
        if DUAL_EXPORT_AVAILABLE:
            requirements_status.append("✅ Dual curves export: Available")
        else:
            requirements_status.append("❌ Dual curves export: Not available")
            
        if PURPLE_EXPORT_AVAILABLE:
            requirements_status.append("✅ Purple curves export: Available")
        else:
            requirements_status.append("❌ Purple curves export: Not available")
        
        status_text = "\n".join(requirements_status)
        
        if XLSXWRITER_AVAILABLE and DUAL_EXPORT_AVAILABLE:
            status_text += "\n\n🎉 All requirements met for enhanced dual curves export!"
        else:
            status_text += "\n\n⚠️ Some requirements missing. Install missing packages for full functionality."
        
        messagebox.showinfo("Export Requirements", status_text)
        
    except Exception as e:
        app_logger.error(f"Error checking requirements: {str(e)}")
        messagebox.showerror("Error", f"Failed to check requirements: {str(e)}")