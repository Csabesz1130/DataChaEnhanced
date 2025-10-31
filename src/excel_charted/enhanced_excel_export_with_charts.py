# src/excel_charted/enhanced_excel_export_with_charts_dual.py
"""
Enhanced Excel export with automatic chart generation and manual curve fitting framework.
Creates Excel files with interactive charts and comprehensive analysis tools for BOTH purple and red curves.
"""

import numpy as np
import pandas as pd
import xlsxwriter
from datetime import datetime
import os
from src.utils.logger import app_logger

class EnhancedExcelExporterDual:
    """Enhanced Excel exporter with automatic chart generation for both purple and red curves"""
    
    def __init__(self):
        self.workbook = None
        self.filename = None
        self.charts_created = []
        
    def export_both_curves_with_charts(self, processor, app, filename=None):
        """
        Export both purple and red curves to Excel with automatic chart generation.
        
        Args:
            processor: ActionPotentialProcessor instance with purple curve data
            app: Main application instance with red curve data (filtered_data, time_data)
            filename: Optional filename, will auto-generate if not provided
        """
        try:
            app_logger.info("Starting enhanced Excel export with automatic charts for both datasets")
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dual_curves_analysis_{timestamp}.xlsx"
            
            self.filename = filename
            
            # Validate data availability
            if not self._validate_dual_curve_data(processor, app):
                raise ValueError("Missing purple or red curve data for export")
            
            # Create Excel workbook using xlsxwriter for advanced chart features
            self.workbook = xlsxwriter.Workbook(filename)
            
            # Create worksheets in order
            self._create_dual_data_worksheet(processor, app)
            self._create_dual_charts_worksheet(processor, app)
            self._create_dual_hyperpol_analysis_worksheet(processor, app)
            self._create_dual_depol_analysis_worksheet(processor, app)
            self._create_dual_manual_fitting_worksheet()
            self._create_dual_instructions_worksheet()
            
            # Close workbook
            self.workbook.close()
            
            app_logger.info(f"Enhanced dual curves Excel export completed: {filename}")
            return filename
            
        except Exception as e:
            app_logger.error(f"Error in enhanced dual curves Excel export: {str(e)}")
            if self.workbook:
                self.workbook.close()
            raise
    
    def _validate_dual_curve_data(self, processor, app):
        """Validate that both purple and red curve data are available."""
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
        
        app_logger.info(f"Data validation - Purple curves: {purple_valid}, Red curves: {red_valid}")
        return purple_valid and red_valid
    
    def _create_dual_data_worksheet(self, processor, app):
        """Create worksheet with both purple and red curve data."""
        worksheet = self.workbook.add_worksheet('Dual_Curve_Data')
        
        # Create formats
        header_format = self.workbook.add_format({
            'bold': True, 
            'font_size': 12, 
            'bg_color': '#4F81BD', 
            'font_color': 'white'
        })
        subheader_format = self.workbook.add_format({
            'bold': True, 
            'font_size': 11, 
            'bg_color': '#B1C5E7'
        })
        
        # Write headers
        worksheet.write('A1', 'DUAL CURVE DATA EXPORT', header_format)
        worksheet.write('A2', f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        worksheet.write('A3', 'Purple curves: Modified/processed data')
        worksheet.write('A4', 'Red curves: Original filtered/denoised data')
        
        # Prepare purple curve data
        hyperpol_times_ms = processor.modified_hyperpol_times * 1000
        depol_times_ms = processor.modified_depol_times * 1000
        
        # Prepare red curve data (continuous, not split into sections)
        red_time_ms = app.time_data * 1000
        red_data = app.filtered_data
        
        # Create unified data structure
        max_len = max(len(processor.modified_hyperpol), len(processor.modified_depol),
                     len(red_data))
        
        # Pad all arrays to same length
        def pad_array(arr, target_len):
            padded = np.full(target_len, np.nan)
            padded[:len(arr)] = arr
            return padded
        
        # Purple curve data (padded)
        purple_hyperpol_times = pad_array(hyperpol_times_ms, max_len)
        purple_hyperpol_data = pad_array(processor.modified_hyperpol, max_len)
        purple_depol_times = pad_array(depol_times_ms, max_len)
        purple_depol_data = pad_array(processor.modified_depol, max_len)
        
        # Red curve data (continuous, padded)
        red_times_padded = pad_array(red_time_ms, max_len)
        red_data_padded = pad_array(red_data, max_len)
        
        # Column headers
        col_headers = [
            'Purple_Hyperpol_Time_ms', 'Purple_Hyperpol_Current_pA',
            'Purple_Depol_Time_ms', 'Purple_Depol_Current_pA',
            'Red_Time_ms', 'Red_Current_pA'
        ]
        
        # Write column headers
        for col, header in enumerate(col_headers):
            worksheet.write(5, col, header, subheader_format)
        
        # Write data with number format
        number_format = self.workbook.add_format({'num_format': '0.000'})
        
        data_arrays = [
            purple_hyperpol_times, purple_hyperpol_data,
            purple_depol_times, purple_depol_data,
            red_times_padded, red_data_padded
        ]
        
        for row in range(max_len):
            for col, data_array in enumerate(data_arrays):
                if row < len(data_array) and not np.isnan(data_array[row]):
                    worksheet.write(row + 6, col, data_array[row], number_format)
        
        # Set column widths
        worksheet.set_column('A:F', 18)
        
        app_logger.info(f"Dual data worksheet created with {max_len} data points")
        
    def _create_dual_charts_worksheet(self, processor, app):
        """Create worksheet with interactive charts for both purple and red curves."""
        worksheet = self.workbook.add_worksheet('Interactive_Charts')
        
        # Create formats
        title_format = self.workbook.add_format({
            'bold': True, 
            'font_size': 14, 
            'bg_color': '#4F81BD', 
            'font_color': 'white'
        })
        
        worksheet.write('A1', 'INTERACTIVE CHARTS - DUAL CURVES', title_format)
        worksheet.write('A2', 'Charts show both Purple (modified) and Red (filtered) curves')
        
        # Create hyperpolarization comparison chart
        hyperpol_chart = self.workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
        
        # Add purple hyperpol series
        hyperpol_chart.add_series({
            'name': 'Purple Hyperpol (Modified)',
            'categories': ['Dual_Curve_Data', 6, 0, 300, 0],  # Time column A
            'values': ['Dual_Curve_Data', 6, 1, 300, 1],      # Current column B
            'line': {'color': 'purple', 'width': 2},
            'marker': {'type': 'none'}
        })
        
        # Add red hyperpol series
        hyperpol_chart.add_series({
            'name': 'Red Hyperpol (Filtered)',
            'categories': ['Dual_Curve_Data', 6, 4, 300, 4],  # Time column E
            'values': ['Dual_Curve_Data', 6, 5, 300, 5],      # Current column F
            'line': {'color': 'red', 'width': 2},
            'marker': {'type': 'none'}
        })
        
        hyperpol_chart.set_title({'name': 'Hyperpolarization Curves Comparison'})
        hyperpol_chart.set_x_axis({'name': 'Time (ms)'})
        hyperpol_chart.set_y_axis({'name': 'Current (pA)'})
        hyperpol_chart.set_size({'width': 600, 'height': 400})
        worksheet.insert_chart('A4', hyperpol_chart)
        
        # Create depolarization comparison chart
        depol_chart = self.workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
        
        # Add purple depol series
        depol_chart.add_series({
            'name': 'Purple Depol (Modified)',
            'categories': ['Dual_Curve_Data', 6, 2, 300, 2],  # Time column C
            'values': ['Dual_Curve_Data', 6, 3, 300, 3],      # Current column D
            'line': {'color': 'purple', 'width': 2},
            'marker': {'type': 'none'}
        })
        
        # Add red depol series
        depol_chart.add_series({
            'name': 'Red Depol (Filtered)',
            'categories': ['Dual_Curve_Data', 6, 6, 300, 6],  # Time column G
            'values': ['Dual_Curve_Data', 6, 7, 300, 7],      # Current column H
            'line': {'color': 'red', 'width': 2},
            'marker': {'type': 'none'}
        })
        
        depol_chart.set_title({'name': 'Depolarization Curves Comparison'})
        depol_chart.set_x_axis({'name': 'Time (ms)'})
        depol_chart.set_y_axis({'name': 'Current (pA)'})
        depol_chart.set_size({'width': 600, 'height': 400})
        worksheet.insert_chart('A25', depol_chart)
        
        app_logger.info("Dual curves charts worksheet created with comparison charts")
        
    def _create_dual_hyperpol_analysis_worksheet(self, processor, app):
        """Create analysis worksheet for hyperpolarization curves comparison."""
        worksheet = self.workbook.add_worksheet('Hyperpol_Analysis')
        
        header_format = self.workbook.add_format({
            'bold': True, 
            'font_size': 12, 
            'bg_color': '#4F81BD', 
            'font_color': 'white'
        })
        
        worksheet.write('A1', 'HYPERPOLARIZATION CURVES ANALYSIS', header_format)
        worksheet.write('A3', 'Comparison between Purple (modified) and Red (filtered) hyperpol curves')
        
        # Calculate basic statistics for both curves
        purple_hyperpol = processor.modified_hyperpol
        purple_stats = {
            'mean': np.mean(purple_hyperpol),
            'std': np.std(purple_hyperpol),
            'min': np.min(purple_hyperpol),
            'max': np.max(purple_hyperpol),
            'peak_to_peak': np.max(purple_hyperpol) - np.min(purple_hyperpol)
        }
        
        # Get red hyperpol data
        if hasattr(processor, '_hyperpol_slice'):
            hyperpol_slice = processor._hyperpol_slice
        else:
            hyperpol_slice = (1035, 1235)
        
        red_hyperpol = app.filtered_data[hyperpol_slice[0]:hyperpol_slice[1]]
        red_stats = {
            'mean': np.mean(red_hyperpol),
            'std': np.std(red_hyperpol),
            'min': np.min(red_hyperpol),
            'max': np.max(red_hyperpol),
            'peak_to_peak': np.max(red_hyperpol) - np.min(red_hyperpol)
        }
        
        # Write comparison table
        worksheet.write('A5', 'Statistic')
        worksheet.write('B5', 'Purple Curve')
        worksheet.write('C5', 'Red Curve')
        worksheet.write('D5', 'Difference')
        worksheet.write('E5', '% Change')
        
        row = 6
        for stat, label in [('mean', 'Mean'), ('std', 'Std Dev'), ('min', 'Minimum'), 
                           ('max', 'Maximum'), ('peak_to_peak', 'Peak-to-Peak')]:
            worksheet.write(row, 0, label)
            worksheet.write(row, 1, purple_stats[stat])
            worksheet.write(row, 2, red_stats[stat])
            diff = purple_stats[stat] - red_stats[stat]
            worksheet.write(row, 3, diff)
            if red_stats[stat] != 0:
                pct_change = (diff / red_stats[stat]) * 100
                worksheet.write(row, 4, f"{pct_change:.2f}%")
            row += 1
        
        worksheet.set_column('A:E', 15)
        
    def _create_dual_depol_analysis_worksheet(self, processor, app):
        """Create analysis worksheet for depolarization curves comparison."""
        worksheet = self.workbook.add_worksheet('Depol_Analysis')
        
        header_format = self.workbook.add_format({
            'bold': True, 
            'font_size': 12, 
            'bg_color': '#4F81BD', 
            'font_color': 'white'
        })
        
        worksheet.write('A1', 'DEPOLARIZATION CURVES ANALYSIS', header_format)
        worksheet.write('A3', 'Comparison between Purple (modified) and Red (filtered) depol curves')
        
        # Calculate basic statistics for both curves
        purple_depol = processor.modified_depol
        purple_stats = {
            'mean': np.mean(purple_depol),
            'std': np.std(purple_depol),
            'min': np.min(purple_depol),
            'max': np.max(purple_depol),
            'peak_to_peak': np.max(purple_depol) - np.min(purple_depol)
        }
        
        # Get red depol data
        if hasattr(processor, '_depol_slice'):
            depol_slice = processor._depol_slice
        else:
            depol_slice = (835, 1035)
        
        red_depol = app.filtered_data[depol_slice[0]:depol_slice[1]]
        red_stats = {
            'mean': np.mean(red_depol),
            'std': np.std(red_depol),
            'min': np.min(red_depol),
            'max': np.max(red_depol),
            'peak_to_peak': np.max(red_depol) - np.min(red_depol)
        }
        
        # Write comparison table
        worksheet.write('A5', 'Statistic')
        worksheet.write('B5', 'Purple Curve')
        worksheet.write('C5', 'Red Curve')
        worksheet.write('D5', 'Difference')
        worksheet.write('E5', '% Change')
        
        row = 6
        for stat, label in [('mean', 'Mean'), ('std', 'Std Dev'), ('min', 'Minimum'), 
                           ('max', 'Maximum'), ('peak_to_peak', 'Peak-to-Peak')]:
            worksheet.write(row, 0, label)
            worksheet.write(row, 1, purple_stats[stat])
            worksheet.write(row, 2, red_stats[stat])
            diff = purple_stats[stat] - red_stats[stat]
            worksheet.write(row, 3, diff)
            if red_stats[stat] != 0:
                pct_change = (diff / red_stats[stat]) * 100
                worksheet.write(row, 4, f"{pct_change:.2f}%")
            row += 1
        
        worksheet.set_column('A:E', 15)
        
    def _create_dual_manual_fitting_worksheet(self):
        """Create worksheet with manual fitting framework for dual curves."""
        worksheet = self.workbook.add_worksheet('Manual_Fitting')
        
        title_format = self.workbook.add_format({
            'bold': True, 
            'font_size': 14, 
            'bg_color': '#4F81BD', 
            'font_color': 'white'
        })
        
        worksheet.write('A1', 'MANUAL CURVE FITTING FRAMEWORK - DUAL CURVES', title_format)
        
        # Instructions
        instructions = [
            "This worksheet provides tools for manual curve fitting on both datasets:",
            "",
            "PURPLE CURVES (Modified Data):",
            "• Use columns A-B for hyperpol time/current",
            "• Use columns C-D for depol time/current", 
            "",
            "RED CURVES (Filtered Data):",
            "• Use columns E-F for hyperpol time/current",
            "• Use columns G-H for depol time/current",
            "",
            "ANALYSIS WORKFLOW:",
            "1. Compare curve characteristics between datasets",
            "2. Identify regions for linear/exponential fitting",
            "3. Select point ranges for each dataset separately",
            "4. Calculate fitting parameters for comparison",
            "5. Document differences in curve behaviors"
        ]
        
        row = 3
        for instruction in instructions:
            worksheet.write(row, 0, instruction)
            row += 1
        
        worksheet.set_column('A:A', 50)
        
    def _create_dual_instructions_worksheet(self):
        """Create comprehensive instructions worksheet for dual curve analysis."""
        worksheet = self.workbook.add_worksheet('Instructions')
        
        title_format = self.workbook.add_format({
            'bold': True, 
            'font_size': 16, 
            'bg_color': '#4F81BD', 
            'font_color': 'white'
        })
        
        section_format = self.workbook.add_format({
            'bold': True, 
            'font_size': 12, 
            'bg_color': '#FFFFCC'
        })
        
        worksheet.write('A1', 'DUAL CURVES ANALYSIS INSTRUCTIONS', title_format)
        
        instructions = [
            "",
            "OVERVIEW:",
            "This Excel file contains both Purple and Red curve datasets for comparison:",
            "• Purple Curves: Modified/processed data from your analysis",
            "• Red Curves: Original filtered/denoised data",
            "",
            "STEP 1: DATA COMPARISON",
            "• Use the 'Interactive_Charts' worksheet to visually compare curves",
            "• Review statistical comparisons in analysis worksheets",
            "• Note differences in curve characteristics",
            "",
            "STEP 2: CURVE ANALYSIS",
            "• Hyperpol_Analysis: Compare hyperpolarization responses",
            "• Depol_Analysis: Compare depolarization responses",
            "• Look for processing effects on signal characteristics",
            "",
            "STEP 3: MANUAL FITTING (Optional)",
            "• Use Manual_Fitting worksheet for detailed analysis",
            "• Compare fitting parameters between datasets",
            "• Document processing effects on curve parameters",
            "",
            "DATA INTERPRETATION:",
            "• Purple curves show post-processing effects",
            "• Red curves represent raw filtered signals",
            "• Differences indicate processing impact",
            "• Use for validation and method development"
        ]
        
        row = 2
        for instruction in instructions:
            if instruction.startswith("STEP") or instruction.startswith("OVERVIEW") or instruction.startswith("DATA INTERPRETATION"):
                worksheet.write(row, 0, instruction, section_format)
            else:
                worksheet.write(row, 0, instruction)
            row += 1
        
        worksheet.set_column('A:A', 60)


def export_both_curves_with_charts(processor, app, filename=None):
    """
    Enhanced export function that creates Excel file with both purple and red curves.
    
    Args:
        processor: ActionPotentialProcessor instance
        app: Main application instance with filtered data
        filename: Optional filename for the Excel file
        
    Returns:
        str: Path to the created Excel file
    """
    exporter = EnhancedExcelExporterDual()
    return exporter.export_both_curves_with_charts(processor, app, filename)