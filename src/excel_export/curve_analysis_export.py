"""
Curve Analysis Excel Export Module
Exports curve fitting results to Excel format
"""

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from datetime import datetime
from src.utils.logger import app_logger

def export_curve_analysis_to_excel(export_data, filepath):
    """Export curve analysis results to Excel.
    
    Args:
        export_data: Dictionary containing all export data
        filepath: Path where Excel file should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create workbook
        wb = openpyxl.Workbook()
        
        # Create sheets
        ws_info = wb.active
        ws_info.title = "File Information"
        
        ws_linear = wb.create_sheet("Linear Fitting")
        ws_exp = wb.create_sheet("Exponential Fitting")
        ws_integration = wb.create_sheet("Integration")
        ws_capacitance = wb.create_sheet("Capacitance")
        ws_summary = wb.create_sheet("Summary")
        
        # Fill sheets
        _fill_info_sheet(ws_info, export_data)
        _fill_linear_sheet(ws_linear, export_data)
        _fill_exponential_sheet(ws_exp, export_data)
        _fill_integration_sheet(ws_integration, export_data)
        _fill_capacitance_sheet(ws_capacitance, export_data)
        _fill_summary_sheet(ws_summary, export_data)
        
        # Save workbook
        wb.save(filepath)
        app_logger.info(f"Excel export saved to: {filepath}")
        return True
        
    except Exception as e:
        app_logger.error(f"Failed to export to Excel: {str(e)}")
        return False

def _get_header_style():
    """Get header cell style."""
    return {
        'font': Font(bold=True, size=12),
        'fill': PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid'),
        'alignment': Alignment(horizontal='left', vertical='center')
    }

def _get_subheader_style():
    """Get subheader cell style."""
    return {
        'font': Font(bold=True, size=11),
        'fill': PatternFill(start_color='B4C7E7', end_color='B4C7E7', fill_type='solid'),
        'alignment': Alignment(horizontal='left', vertical='center')
    }

def _apply_style(cell, style_dict):
    """Apply style to cell."""
    for attr, value in style_dict.items():
        setattr(cell, attr, value)

def _fill_info_sheet(ws, data):
    """Fill file information sheet."""
    # Title
    ws['A1'] = 'FILE INFORMATION'
    _apply_style(ws['A1'], _get_header_style())
    
    # Information
    row = 3
    info_items = [
        ('File Name', data.get('filename', 'Unknown')),
        ('Voltage (mV)', data.get('voltage', 'N/A')),
        ('Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    ]
    
    for label, value in info_items:
        ws[f'A{row}'] = label
        ws[f'B{row}'] = value
        _apply_style(ws[f'A{row}'], {'font': Font(bold=True)})
        row += 1
    
    # Auto-size columns
    ws.column_dimensions['A'].width = 20
    ws.column_dimensions['B'].width = 40

def _fill_linear_sheet(ws, data):
    """Fill linear fitting sheet."""
    # Title
    ws['A1'] = 'LINEAR FITTING'
    _apply_style(ws['A1'], _get_header_style())
    
    # Headers
    ws['A3'] = 'Parameter'
    ws['B3'] = 'Hyperpolarization'
    ws['C3'] = 'Depolarization'
    for col in ['A3', 'B3', 'C3']:
        _apply_style(ws[col], _get_subheader_style())
    
    # Data
    linear_data = data.get('linear_fitting', {})
    hyperpol = linear_data.get('hyperpol', {})
    depol = linear_data.get('depol', {})
    
    rows = [
        ('Slope (pA/ms)', 
         hyperpol.get('slope', 'N/A'), 
         depol.get('slope', 'N/A')),
        ('Y-intercept (pA)', 
         hyperpol.get('intercept', 'N/A'), 
         depol.get('intercept', 'N/A')),
        ('R²', 
         hyperpol.get('r_squared', 'N/A'), 
         depol.get('r_squared', 'N/A')),
        ('Equation', 
         hyperpol.get('equation', 'N/A'), 
         depol.get('equation', 'N/A'))
    ]
    
    row = 4
    for label, hyp_val, dep_val in rows:
        ws[f'A{row}'] = label
        ws[f'B{row}'] = hyp_val
        ws[f'C{row}'] = dep_val
        _apply_style(ws[f'A{row}'], {'font': Font(bold=True)})
        row += 1
    
    # Auto-size columns
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 30
    ws.column_dimensions['C'].width = 30

def _fill_exponential_sheet(ws, data):
    """Fill exponential fitting sheet."""
    # Title
    ws['A1'] = 'EXPONENTIAL FITTING'
    _apply_style(ws['A1'], _get_header_style())
    
    # Headers
    ws['A3'] = 'Parameter'
    ws['B3'] = 'Hyperpolarization'
    ws['C3'] = 'Depolarization'
    for col in ['A3', 'B3', 'C3']:
        _apply_style(ws[col], _get_subheader_style())
    
    # Data
    exp_data = data.get('exponential_fitting', {})
    hyperpol = exp_data.get('hyperpol', {})
    depol = exp_data.get('depol', {})
    
    rows = [
        ('Tau (ms)', 
         hyperpol.get('tau', 'N/A'), 
         depol.get('tau', 'N/A')),
        ('Amplitude (pA)', 
         hyperpol.get('A', 'N/A'), 
         depol.get('A', 'N/A')),
        ('Offset (pA)', 
         hyperpol.get('C', 'N/A'), 
         depol.get('C', 'N/A')),
        ('R²', 
         hyperpol.get('r_squared', 'N/A'), 
         depol.get('r_squared', 'N/A')),
        ('Equation', 
         hyperpol.get('equation', 'N/A'), 
         depol.get('equation', 'N/A'))
    ]
    
    row = 4
    for label, hyp_val, dep_val in rows:
        ws[f'A{row}'] = label
        ws[f'B{row}'] = hyp_val
        ws[f'C{row}'] = dep_val
        _apply_style(ws[f'A{row}'], {'font': Font(bold=True)})
        row += 1
    
    # Auto-size columns
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 30
    ws.column_dimensions['C'].width = 30

def _fill_integration_sheet(ws, data):
    """Fill integration sheet."""
    # Title
    ws['A1'] = 'INTEGRATION'
    _apply_style(ws['A1'], _get_header_style())
    
    # Headers
    ws['A3'] = 'Parameter'
    ws['B3'] = 'Hyperpolarization'
    ws['C3'] = 'Depolarization'
    for col in ['A3', 'B3', 'C3']:
        _apply_style(ws[col], _get_subheader_style())
    
    # Data
    integration_data = data.get('integration', {})
    hyperpol = integration_data.get('hyperpol', {})
    depol = integration_data.get('depol', {})
    
    rows = [
        ('Integral (pA·s)', 
         hyperpol.get('integral', 'N/A'), 
         depol.get('integral', 'N/A')),
        ('Start Point Index', 
         hyperpol.get('start_point', 'N/A'), 
         depol.get('start_point', 'N/A')),
        ('End Point Index', 
         hyperpol.get('end_point', 'N/A'), 
         depol.get('end_point', 'N/A'))
    ]
    
    row = 4
    for label, hyp_val, dep_val in rows:
        ws[f'A{row}'] = label
        ws[f'B{row}'] = hyp_val
        ws[f'C{row}'] = dep_val
        _apply_style(ws[f'A{row}'], {'font': Font(bold=True)})
        row += 1
    
    # Auto-size columns
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 30
    ws.column_dimensions['C'].width = 30

def _fill_capacitance_sheet(ws, data):
    """Fill capacitance sheet."""
    # Title
    ws['A1'] = 'CAPACITANCE'
    _apply_style(ws['A1'], _get_header_style())
    
    # Data
    ws['A3'] = 'Linear Capacitance'
    ws['B3'] = data.get('capacitance', {}).get('linear_capacitance', 'N/A')
    _apply_style(ws['A3'], {'font': Font(bold=True)})
    
    # Auto-size columns
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 30

def _fill_summary_sheet(ws, data):
    """Fill summary sheet."""
    # Title
    ws['A1'] = 'SUMMARY'
    _apply_style(ws['A1'], _get_header_style())
    
    row = 3
    
    # Hyperpolarization section
    ws[f'A{row}'] = '=== HYPERPOLARIZATION ==='
    _apply_style(ws[f'A{row}'], _get_subheader_style())
    row += 1
    
    linear_hyp = data.get('linear_fitting', {}).get('hyperpol', {})
    integration_hyp = data.get('integration', {}).get('hyperpol', {})
    
    hyp_items = [
        ('Linear Slope (pA/ms)', linear_hyp.get('slope', 'N/A')),
        ('Linear Y0 (pA)', linear_hyp.get('intercept', 'N/A')),
        ('Linear R²', linear_hyp.get('r_squared', 'N/A')),
        ('Integral (pA·s)', integration_hyp.get('integral', 'N/A'))
    ]
    
    for label, value in hyp_items:
        ws[f'A{row}'] = label
        ws[f'B{row}'] = value
        _apply_style(ws[f'A{row}'], {'font': Font(bold=True)})
        row += 1
    
    row += 1
    
    # Depolarization section
    ws[f'A{row}'] = '=== DEPOLARIZATION ==='
    _apply_style(ws[f'A{row}'], _get_subheader_style())
    row += 1
    
    linear_dep = data.get('linear_fitting', {}).get('depol', {})
    integration_dep = data.get('integration', {}).get('depol', {})
    
    dep_items = [
        ('Linear Slope (pA/ms)', linear_dep.get('slope', 'N/A')),
        ('Linear Y0 (pA)', linear_dep.get('intercept', 'N/A')),
        ('Linear R²', linear_dep.get('r_squared', 'N/A')),
        ('Integral (pA·s)', integration_dep.get('integral', 'N/A'))
    ]
    
    for label, value in dep_items:
        ws[f'A{row}'] = label
        ws[f'B{row}'] = value
        _apply_style(ws[f'A{row}'], {'font': Font(bold=True)})
        row += 1
    
    row += 1
    
    # Overall section
    ws[f'A{row}'] = '=== OVERALL ==='
    _apply_style(ws[f'A{row}'], _get_subheader_style())
    row += 1
    
    ws[f'A{row}'] = 'Linear Capacitance'
    ws[f'B{row}'] = data.get('capacitance', {}).get('linear_capacitance', 'N/A')
    _apply_style(ws[f'A{row}'], {'font': Font(bold=True)})
    
    # Auto-size columns
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 30
