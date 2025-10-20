"""
Curve Analysis Excel Export Module for DataChaEnhanced
====================================================
Location: src/excel_export/curve_analysis_export.py

This module handles exporting curve analysis results to Excel format.
"""

import openpyxl
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter
from pathlib import Path
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def export_curve_analysis_to_excel(export_data: Dict[str, Any], filepath: str) -> bool:
    """
    Export curve analysis data to Excel file.
    
    Args:
        export_data: Dictionary containing all analysis data
        filepath: Path where to save the Excel file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create new workbook
        wb = openpyxl.Workbook()
        ws = wb.active
        
        # Get filename for sheet name (remove extension and limit length)
        filename = export_data.get('filename', 'Unknown')
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        # Excel sheet names are limited to 31 characters
        sheet_name = filename[:31] if len(filename) > 31 else filename
        ws.title = sheet_name
        
        # Get depolarization voltage
        v2_voltage = export_data.get('v2_voltage', 0)
        
        # Add header information
        ws['A1'] = f"Curve Analysis Results - {filename}"
        ws['A1'].font = Font(bold=True, size=14)
        
        ws['A2'] = f"Depolarization Voltage: {v2_voltage} mV"
        ws['A2'].font = Font(bold=True)
        
        # Add timestamp
        ws['A3'] = f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ws['A3'].font = Font(italic=True)
        
        # Create table headers
        headers = ['Parameter', 'Hyperpol', 'Depol']
        header_row = 5
        
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=header_row, column=col, value=header)
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center')
            cell.fill = PatternFill(start_color='E0E0E0', end_color='E0E0E0', fill_type='solid')
        
        # Define table data
        table_data = [
            ('Linear Slope', 'linear_slope'),
            ('Linear R²', 'linear_r2'),
            ('Exp Amplitude (A)', 'exp_amplitude'),
            ('Exp Tau (ms)', 'exp_tau'),
            ('Exp R²', 'exp_r2'),
            ('Integral (pC)', 'integral'),
            ('Linear Capacitance (nF)', 'linear_capacitance')
        ]
        
        # Add table data
        for row_idx, (param_name, data_key) in enumerate(table_data, header_row + 1):
            # Parameter name
            ws.cell(row=row_idx, column=1, value=param_name).font = Font(bold=True)
            
            # Hyperpol value
            hyperpol_data = export_data.get('hyperpol', {})
            hyperpol_value = hyperpol_data.get(data_key, 'N/A')
            if isinstance(hyperpol_value, (int, float)):
                hyperpol_value = round(hyperpol_value, 3)
            ws.cell(row=row_idx, column=2, value=hyperpol_value)
            
            # Depol value (except for linear capacitance which is overall)
            if data_key == 'linear_capacitance':
                ws.cell(row=row_idx, column=3, value='-')
            else:
                depol_data = export_data.get('depol', {})
                depol_value = depol_data.get(data_key, 'N/A')
                if isinstance(depol_value, (int, float)):
                    depol_value = round(depol_value, 3)
                ws.cell(row=row_idx, column=3, value=depol_value)
        
        # Add borders to table
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Apply borders to table area
        for row in range(header_row, header_row + len(table_data) + 1):
            for col in range(1, 4):
                ws.cell(row=row, column=col).border = thin_border
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            ws.column_dimensions[column_letter].width = adjusted_width
        
        # Center align all data cells
        for row in range(header_row + 1, header_row + len(table_data) + 1):
            for col in range(2, 4):  # Only hyperpol and depol columns
                ws.cell(row=row, column=col).alignment = Alignment(horizontal='center')
        
        # Save the workbook
        wb.save(filepath)
        logger.info(f"Successfully exported curve analysis to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        return False

def format_export_data(fitting_results: Dict, integrals: Dict, capacitance: str, v2_voltage: float, filename: str) -> Dict[str, Any]:
    """
    Format data for export in the required structure.
    
    Args:
        fitting_results: Results from curve fitting manager
        integrals: Integration results from range manager
        capacitance: Linear capacitance value
        v2_voltage: Depolarization voltage (V2)
        filename: Current filename
        
    Returns:
        Dict: Formatted data for export
    """
    try:
        export_data = {
            'filename': filename,
            'v2_voltage': v2_voltage,
            'hyperpol': {},
            'depol': {}
        }
        
        # Process fitting results
        for curve_type in ['hyperpol', 'depol']:
            if curve_type in fitting_results:
                curve_data = fitting_results[curve_type]
                
                # Linear fit data
                if 'linear' in curve_data:
                    linear = curve_data['linear']
                    export_data[curve_type]['linear_slope'] = linear.get('slope', 0)
                    export_data[curve_type]['linear_r2'] = linear.get('r_squared', 0)
                
                # Exponential fit data
                if 'exponential' in curve_data:
                    exp = curve_data['exponential']
                    export_data[curve_type]['exp_amplitude'] = exp.get('A', 0)
                    export_data[curve_type]['exp_tau'] = exp.get('tau_ms', 0)  # Convert to ms
                    export_data[curve_type]['exp_r2'] = exp.get('r_squared', 0)
        
        # Process integration results
        if 'hyperpol_integral' in integrals:
            export_data['hyperpol']['integral'] = integrals['hyperpol_integral']
        if 'depol_integral' in integrals:
            export_data['depol']['integral'] = integrals['depol_integral']
        
        # Process linear capacitance (overall value)
        if capacitance and capacitance != "---":
            # Extract numeric value from string like "1.234 nF"
            try:
                cap_value = float(capacitance.replace(' nF', '').replace(' pF', ''))
                # Convert pF to nF if needed
                if 'pF' in capacitance:
                    cap_value = cap_value / 1000
                export_data['linear_capacitance'] = cap_value
            except (ValueError, AttributeError):
                export_data['linear_capacitance'] = 0
        else:
            export_data['linear_capacitance'] = 0
        
        return export_data
        
    except Exception as e:
        logger.error(f"Error formatting export data: {str(e)}")
        return {
            'filename': filename,
            'v2_voltage': v2_voltage,
            'hyperpol': {},
            'depol': {},
            'linear_capacitance': 0
        }
