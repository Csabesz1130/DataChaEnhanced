# src/analysis/enhanced_excel_export_with_charts.py
"""
Enhanced Excel export with automatic chart generation and manual curve fitting framework.
Creates Excel files with interactive charts and comprehensive analysis tools.
"""

import numpy as np
import pandas as pd
import xlsxwriter
from datetime import datetime
import os
from src.utils.logger import app_logger

class EnhancedExcelExporter:
    """Enhanced Excel exporter with automatic chart generation and manual curve fitting framework"""
    
    def __init__(self):
        self.workbook = None
        self.filename = None
        self.charts_created = []
        
    def export_purple_curves_with_charts(self, processor, filename=None):
        """
        Export purple curves to Excel with automatic chart generation and manual analysis framework.
        
        Args:
            processor: ActionPotentialProcessor instance with purple curve data
            filename: Optional filename, will auto-generate if not provided
        """
        try:
            app_logger.info("Starting enhanced Excel export with automatic charts")
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"purple_curves_analysis_{timestamp}.xlsx"
            
            self.filename = filename
            
            # Check if we have purple curve data
            if not self._validate_purple_curve_data(processor):
                raise ValueError("No purple curve data available for export")
            
            # Create Excel workbook using xlsxwriter for advanced chart features
            self.workbook = xlsxwriter.Workbook(filename)
            
            # Create worksheets in order
            self._create_data_worksheet(processor)
            self._create_charts_worksheet(processor)  # This creates the instant charts
            self._create_hyperpol_analysis_worksheet(processor)
            self._create_depol_analysis_worksheet(processor)
            self._create_manual_fitting_worksheet()
            self._create_instructions_worksheet()
            
            # Close workbook
            self.workbook.close()
            
            app_logger.info(f"Enhanced Excel export completed with charts: {filename}")
            return filename
            
        except Exception as e:
            app_logger.error(f"Error in enhanced Excel export: {str(e)}")
            if self.workbook:
                self.workbook.close()
            raise
    
    def _validate_purple_curve_data(self, processor):
        """Validate that purple curve data is available."""
        required_attrs = [
            'modified_hyperpol', 'modified_depol',
            'modified_hyperpol_times', 'modified_depol_times'
        ]
        
        for attr in required_attrs:
            if not hasattr(processor, attr) or getattr(processor, attr) is None:
                app_logger.error(f"Missing required attribute: {attr}")
                return False
        return True
    
    def _create_data_worksheet(self, processor):
        """Create the main data worksheet with purple curve data."""
        worksheet = self.workbook.add_worksheet('Purple_Curve_Data')
        
        # Define formats
        header_format = self.workbook.add_format({
            'bold': True,
            'font_color': 'white',
            'bg_color': '#4F81BD',
            'border': 1,
            'align': 'center'
        })
        
        hyperpol_format = self.workbook.add_format({
            'bg_color': '#E6F3FF',
            'border': 1,
            'num_format': '0.0000000'
        })
        
        depol_format = self.workbook.add_format({
            'bg_color': '#FFE6E6',
            'border': 1,
            'num_format': '0.0000000'
        })
        
        # Write headers
        headers = [
            'Hyperpol_Time_ms', 'Hyperpol_Current_pA', 
            'Depol_Time_ms', 'Depol_Current_pA'
        ]
        
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Write data
        hyperpol_times_ms = processor.modified_hyperpol_times * 1000  # Convert to ms
        depol_times_ms = processor.modified_depol_times * 1000
        
        max_rows = max(len(processor.modified_hyperpol), len(processor.modified_depol))
        
        for row in range(max_rows):
            # Hyperpolarization data
            if row < len(processor.modified_hyperpol):
                worksheet.write(row + 1, 0, hyperpol_times_ms[row], hyperpol_format)
                worksheet.write(row + 1, 1, processor.modified_hyperpol[row], hyperpol_format)
            
            # Depolarization data
            if row < len(processor.modified_depol):
                worksheet.write(row + 1, 2, depol_times_ms[row], depol_format)
                worksheet.write(row + 1, 3, processor.modified_depol[row], depol_format)
        
        # Set column widths
        worksheet.set_column('A:D', 15)
        
        # Store data range for chart creation
        self.data_range = {
            'hyperpol_rows': len(processor.modified_hyperpol),
            'depol_rows': len(processor.modified_depol)
        }
    
    def _create_charts_worksheet(self, processor):
        """Create worksheet with automatic charts like in the user's image."""
        worksheet = self.workbook.add_worksheet('Charts')
        
        # Add title
        title_format = self.workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center',
            'font_color': '#4F81BD'
        })
        
        worksheet.merge_range('A1:H1', 'Purple Curves Analysis - Automatic Charts', title_format)
        
        # Create Hyperpolarization chart (like in the image)
        hyperpol_chart = self.workbook.add_chart({'type': 'scatter', 'subtype': 'smooth_with_markers'})
        
        # Configure hyperpol chart
        hyperpol_chart.add_series({
            'name': 'Hyperpolarization Purple Curve',
            'categories': ['Purple_Curve_Data', 1, 0, self.data_range['hyperpol_rows'], 0],
            'values': ['Purple_Curve_Data', 1, 1, self.data_range['hyperpol_rows'], 1],
            'line': {'color': '#8B00FF', 'width': 2},  # Purple color
            'marker': {
                'type': 'circle', 
                'size': 4, 
                'border': {'color': '#8B00FF'}, 
                'fill': {'color': '#8B00FF'}
            }
        })
        
        hyperpol_chart.set_title({
            'name': 'Purple Hyperpolarization Curve',
            'name_font': {'size': 14, 'bold': True}
        })
        hyperpol_chart.set_x_axis({
            'name': 'Time (ms)',
            'major_gridlines': {'visible': True},
            'name_font': {'size': 12}
        })
        hyperpol_chart.set_y_axis({
            'name': 'Current (pA)',
            'major_gridlines': {'visible': True},
            'name_font': {'size': 12}
        })
        hyperpol_chart.set_legend({'position': 'top'})
        hyperpol_chart.set_size({'width': 600, 'height': 400})
        
        # Insert hyperpol chart
        worksheet.insert_chart('A3', hyperpol_chart)
        
        # Create Depolarization chart
        depol_chart = self.workbook.add_chart({'type': 'scatter', 'subtype': 'smooth_with_markers'})
        
        # Configure depol chart
        depol_chart.add_series({
            'name': 'Depolarization Purple Curve',
            'categories': ['Purple_Curve_Data', 1, 2, self.data_range['depol_rows'], 2],
            'values': ['Purple_Curve_Data', 1, 3, self.data_range['depol_rows'], 3],
            'line': {'color': '#8B00FF', 'width': 2},  # Purple color
            'marker': {
                'type': 'circle', 
                'size': 4, 
                'border': {'color': '#8B00FF'}, 
                'fill': {'color': '#8B00FF'}
            }
        })
        
        depol_chart.set_title({
            'name': 'Purple Depolarization Curve',
            'name_font': {'size': 14, 'bold': True}
        })
        depol_chart.set_x_axis({
            'name': 'Time (ms)',
            'major_gridlines': {'visible': True},
            'name_font': {'size': 12}
        })
        depol_chart.set_y_axis({
            'name': 'Current (pA)',
            'major_gridlines': {'visible': True},
            'name_font': {'size': 12}
        })
        depol_chart.set_legend({'position': 'top'})
        depol_chart.set_size({'width': 600, 'height': 400})
        
        # Insert depol chart
        worksheet.insert_chart('A25', depol_chart)
        
        # Add combined chart (both curves)
        combined_chart = self.workbook.add_chart({'type': 'scatter', 'subtype': 'smooth_with_markers'})
        
        # Add both series to combined chart
        combined_chart.add_series({
            'name': 'Hyperpolarization',
            'categories': ['Purple_Curve_Data', 1, 0, self.data_range['hyperpol_rows'], 0],
            'values': ['Purple_Curve_Data', 1, 1, self.data_range['hyperpol_rows'], 1],
            'line': {'color': '#0000FF', 'width': 2},  # Blue for hyperpol
            'marker': {'type': 'circle', 'size': 3, 'border': {'color': '#0000FF'}, 'fill': {'color': '#0000FF'}}
        })
        
        combined_chart.add_series({
            'name': 'Depolarization',
            'categories': ['Purple_Curve_Data', 1, 2, self.data_range['depol_rows'], 2],
            'values': ['Purple_Curve_Data', 1, 3, self.data_range['depol_rows'], 3],
            'line': {'color': '#FF0000', 'width': 2},  # Red for depol
            'marker': {'type': 'circle', 'size': 3, 'border': {'color': '#FF0000'}, 'fill': {'color': '#FF0000'}}
        })
        
        combined_chart.set_title({
            'name': 'Combined Purple Curves - Manual Analysis',
            'name_font': {'size': 14, 'bold': True}
        })
        combined_chart.set_x_axis({
            'name': 'Time (ms)',
            'major_gridlines': {'visible': True},
            'name_font': {'size': 12}
        })
        combined_chart.set_y_axis({
            'name': 'Current (pA)',
            'major_gridlines': {'visible': True},
            'name_font': {'size': 12}
        })
        combined_chart.set_legend({'position': 'top'})
        combined_chart.set_size({'width': 600, 'height': 400})
        
        # Insert combined chart
        worksheet.insert_chart('A47', combined_chart)
        
        # Add instructions for chart usage
        instruction_format = self.workbook.add_format({
            'bold': True,
            'font_size': 11,
            'bg_color': '#FFFFCC',
            'border': 1,
            'text_wrap': True
        })
        
        worksheet.merge_range('K3:O15', 
            'CHART USAGE INSTRUCTIONS:\n\n'
            '1. Use these charts to visually identify points for curve fitting\n'
            '2. Click on chart points to see coordinates\n'
            '3. Note the time values (x-axis) for linear fitting\n'
            '4. Go to analysis worksheets to enter point selections\n'
            '5. Use row numbers from Purple_Curve_Data sheet\n'
            '6. Charts show the actual data you will be analyzing', 
            instruction_format)
    
    def _create_hyperpol_analysis_worksheet(self, processor):
        """Create hyperpolarization analysis worksheet with manual curve fitting."""
        worksheet = self.workbook.add_worksheet('Hyperpol_Analysis')
        
        # Title format
        title_format = self.workbook.add_format({
            'bold': True,
            'font_size': 16,
            'font_color': '#0000FF',
            'align': 'center'
        })
        
        # Section header format
        section_format = self.workbook.add_format({
            'bold': True,
            'font_size': 12,
            'bg_color': '#D9E2F3',
            'border': 1,
            'align': 'center'
        })
        
        # Input format for user entries
        input_format = self.workbook.add_format({
            'bg_color': '#FFF2CC',
            'border': 1,
            'align': 'center',
            'num_format': '0'
        })
        
        # Result format
        result_format = self.workbook.add_format({
            'bg_color': '#E2EFDA',
            'border': 1,
            'num_format': '0.0000'
        })
        
        row = 0
        
        # Title
        worksheet.merge_range(row, 0, row, 7, 'HYPERPOLARIZATION PURPLE CURVE ANALYSIS', title_format)
        row += 2
        
        # STEP 1: Manual Point Selection for Linear Fitting
        worksheet.merge_range(row, 0, row, 7, 'STEP 1: SELECT TWO POINTS FOR LINEAR FITTING', section_format)
        row += 1
        
        worksheet.write(row, 0, 'Linear Fitting Points (Row Numbers):', self.workbook.add_format({'bold': True}))
        row += 1
        
        worksheet.write(row, 0, 'Point 1 (Start Row #):')
        worksheet.write(row, 1, '', input_format)  # User input cell B
        worksheet.write(row, 2, 'Point 2 (End Row #):')
        worksheet.write(row, 3, '', input_format)   # User input cell D
        
        # Add data validation to restrict to valid row numbers
        worksheet.data_validation(row, 1, row, 1, {
            'validate': 'integer',
            'criteria': 'between',
            'minimum': 1,
            'maximum': len(processor.modified_hyperpol),
            'error_message': 'Must be a valid row number from Purple_Curve_Data sheet'
        })
        
        worksheet.data_validation(row, 3, row, 3, {
            'validate': 'integer', 
            'criteria': 'between',
            'minimum': 1,
            'maximum': len(processor.modified_hyperpol),
            'error_message': 'Must be a valid row number from Purple_Curve_Data sheet'
        })
        row += 2
        
        # STEP 2: Linear Fitting Results 
        worksheet.merge_range(row, 0, row, 7, 'STEP 2: LINEAR FITTING RESULTS (y = mx + a)', section_format)
        row += 1
        
        worksheet.write(row, 0, 'Slope (m):', self.workbook.add_format({'bold': True}))
        # Formula to calculate slope based on selected points
        worksheet.write_formula(row, 1, 
            '=IF(AND(B4<>"",D4<>""),SLOPE(OFFSET(Purple_Curve_Data!B:B,B4,0,D4-B4+1,1),OFFSET(Purple_Curve_Data!A:A,B4,0,D4-B4+1,1)),"")', 
            result_format)
        worksheet.write(row, 2, 'pA/ms')
        row += 1
        
        worksheet.write(row, 0, 'Intercept (a):', self.workbook.add_format({'bold': True}))
        worksheet.write_formula(row, 1, 
            '=IF(AND(B4<>"",D4<>""),INTERCEPT(OFFSET(Purple_Curve_Data!B:B,B4,0,D4-B4+1,1),OFFSET(Purple_Curve_Data!A:A,B4,0,D4-B4+1,1)),"")', 
            result_format)
        worksheet.write(row, 2, 'pA')
        row += 1
        
        worksheet.write(row, 0, 'R-squared:', self.workbook.add_format({'bold': True}))
        worksheet.write_formula(row, 1, 
            '=IF(AND(B4<>"",D4<>""),RSQ(OFFSET(Purple_Curve_Data!B:B,B4,0,D4-B4+1,1),OFFSET(Purple_Curve_Data!A:A,B4,0,D4-B4+1,1)),"")', 
            result_format)
        row += 2
        
        # STEP 3: Third Point Selection
        worksheet.merge_range(row, 0, row, 7, 'STEP 3: SELECT THIRD POINT FOR EXPONENTIAL FITTING', section_format)
        row += 1
        
        worksheet.write(row, 0, 'Exponential Start Point (Row #):')
        worksheet.write(row, 1, '', input_format)  # User input cell
        
        worksheet.data_validation(row, 1, row, 1, {
            'validate': 'integer',
            'criteria': 'between', 
            'minimum': 1,
            'maximum': len(processor.modified_hyperpol),
            'error_message': 'Must be a valid row number from Purple_Curve_Data sheet'
        })
        row += 2
        
        # STEP 4: Trend Removal/Addition
        worksheet.merge_range(row, 0, row, 7, 'STEP 4: TREND REMOVAL (HYPERPOL = SUBTRACT LINEAR)', section_format)
        row += 1
        
        worksheet.write(row, 0, 'Linear Trend Operation:', self.workbook.add_format({'bold': True}))
        worksheet.write(row, 1, 'SUBTRACT linear trend from hyperpol curve')
        worksheet.write(row, 0, 'Formula: Corrected_Current = Original_Current - (m × time + a)')
        row += 2
        
        # STEP 5: Exponential Fitting Results
        worksheet.merge_range(row, 0, row, 7, 'STEP 5: EXPONENTIAL FITTING RESULTS (y = A×exp(-t/τ) + C)', section_format)
        row += 1
        
        worksheet.write(row, 0, 'Amplitude (A):', self.workbook.add_format({'bold': True}))
        worksheet.write(row, 1, '', result_format)  # Manual entry - requires Solver
        worksheet.write(row, 2, 'pA')
        row += 1
        
        worksheet.write(row, 0, 'Time Constant (τ):', self.workbook.add_format({'bold': True}))
        worksheet.write(row, 1, '', result_format)  # Manual entry - requires Solver
        worksheet.write(row, 2, 'ms')
        row += 1
        
        worksheet.write(row, 0, 'Offset (C):', self.workbook.add_format({'bold': True}))
        worksheet.write(row, 1, '', result_format)  # Manual entry - requires Solver
        worksheet.write(row, 2, 'pA')
        row += 1
        
        worksheet.write(row, 0, 'R-squared:', self.workbook.add_format({'bold': True}))
        worksheet.write(row, 1, '', result_format)  # Manual entry
        row += 2
        
        # STEP 6: Parameter Summary
        worksheet.merge_range(row, 0, row, 7, 'STEP 6: EXTRACTED PARAMETERS SUMMARY', section_format)
        row += 1
        
        summary_format = self.workbook.add_format({'bg_color': '#E6F3FF', 'border': 1})
        
        worksheet.write(row, 0, 'Linear Parameters:', self.workbook.add_format({'bold': True}))
        row += 1
        worksheet.write(row, 0, 'Slope (m):', summary_format)
        worksheet.write_formula(row, 1, '=B8', summary_format)
        worksheet.write(row, 2, 'pA/ms', summary_format)
        row += 1
        worksheet.write(row, 0, 'Intercept (a):', summary_format)
        worksheet.write_formula(row, 1, '=B9', summary_format)
        worksheet.write(row, 2, 'pA', summary_format)
        row += 1
        
        worksheet.write(row, 0, 'Exponential Parameters:', self.workbook.add_format({'bold': True}))
        row += 1
        worksheet.write(row, 0, 'Amplitude (A):', summary_format)
        worksheet.write_formula(row, 1, '=B19', summary_format)
        worksheet.write(row, 2, 'pA', summary_format)
        row += 1
        worksheet.write(row, 0, 'Time Constant (τ):', summary_format)
        worksheet.write_formula(row, 1, '=B20', summary_format)
        worksheet.write(row, 2, 'ms', summary_format)
        row += 1
        worksheet.write(row, 0, 'Offset (C):', summary_format)
        worksheet.write_formula(row, 1, '=B21', summary_format)
        worksheet.write(row, 2, 'pA', summary_format)
        
        # Set column widths
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:G', 12)
    
    def _create_depol_analysis_worksheet(self, processor):
        """Create depolarization analysis worksheet (similar structure to hyperpol)."""
        worksheet = self.workbook.add_worksheet('Depol_Analysis')
        
        # Title format
        title_format = self.workbook.add_format({
            'bold': True,
            'font_size': 16,
            'font_color': '#FF0000',
            'align': 'center'
        })
        
        # Section header format
        section_format = self.workbook.add_format({
            'bold': True,
            'font_size': 12,
            'bg_color': '#F2DCDB',
            'border': 1,
            'align': 'center'
        })
        
        # Input format
        input_format = self.workbook.add_format({
            'bg_color': '#FFF2CC',
            'border': 1,
            'align': 'center',
            'num_format': '0'
        })
        
        # Result format
        result_format = self.workbook.add_format({
            'bg_color': '#E2EFDA',
            'border': 1,
            'num_format': '0.0000'
        })
        
        row = 0
        
        # Title
        worksheet.merge_range(row, 0, row, 7, 'DEPOLARIZATION PURPLE CURVE ANALYSIS', title_format)
        row += 2
        
        # STEP 1: Manual Point Selection for Linear Fitting
        worksheet.merge_range(row, 0, row, 7, 'STEP 1: SELECT TWO POINTS FOR LINEAR FITTING', section_format)
        row += 1
        
        worksheet.write(row, 0, 'Linear Fitting Points (Row Numbers):', self.workbook.add_format({'bold': True}))
        row += 1
        
        worksheet.write(row, 0, 'Point 1 (Start Row #):')
        worksheet.write(row, 1, '', input_format)
        worksheet.write(row, 2, 'Point 2 (End Row #):')
        worksheet.write(row, 3, '', input_format)
        
        # Data validation
        worksheet.data_validation(row, 1, row, 1, {
            'validate': 'integer',
            'criteria': 'between',
            'minimum': 1,
            'maximum': len(processor.modified_depol),
            'error_message': 'Must be a valid row number from Purple_Curve_Data sheet'
        })
        
        worksheet.data_validation(row, 3, row, 3, {
            'validate': 'integer',
            'criteria': 'between',
            'minimum': 1,
            'maximum': len(processor.modified_depol),
            'error_message': 'Must be a valid row number from Purple_Curve_Data sheet'
        })
        row += 2
        
        # STEP 2: Linear Fitting Results (using depol data - columns C and D)
        worksheet.merge_range(row, 0, row, 7, 'STEP 2: LINEAR FITTING RESULTS (y = mx + a)', section_format)
        row += 1
        
        worksheet.write(row, 0, 'Slope (m):', self.workbook.add_format({'bold': True}))
        worksheet.write_formula(row, 1, 
            '=IF(AND(B4<>"",D4<>""),SLOPE(OFFSET(Purple_Curve_Data!D:D,B4,0,D4-B4+1,1),OFFSET(Purple_Curve_Data!C:C,B4,0,D4-B4+1,1)),"")', 
            result_format)
        worksheet.write(row, 2, 'pA/ms')
        row += 1
        
        worksheet.write(row, 0, 'Intercept (a):', self.workbook.add_format({'bold': True}))
        worksheet.write_formula(row, 1, 
            '=IF(AND(B4<>"",D4<>""),INTERCEPT(OFFSET(Purple_Curve_Data!D:D,B4,0,D4-B4+1,1),OFFSET(Purple_Curve_Data!C:C,B4,0,D4-B4+1,1)),"")', 
            result_format)
        worksheet.write(row, 2, 'pA')
        row += 1
        
        worksheet.write(row, 0, 'R-squared:', self.workbook.add_format({'bold': True}))
        worksheet.write_formula(row, 1, 
            '=IF(AND(B4<>"",D4<>""),RSQ(OFFSET(Purple_Curve_Data!D:D,B4,0,D4-B4+1,1),OFFSET(Purple_Curve_Data!C:C,B4,0,D4-B4+1,1)),"")', 
            result_format)
        row += 2
        
        # STEP 3: Third Point Selection
        worksheet.merge_range(row, 0, row, 7, 'STEP 3: SELECT THIRD POINT FOR EXPONENTIAL FITTING', section_format)
        row += 1
        
        worksheet.write(row, 0, 'Exponential Start Point (Row #):')
        worksheet.write(row, 1, '', input_format)
        
        worksheet.data_validation(row, 1, row, 1, {
            'validate': 'integer',
            'criteria': 'between',
            'minimum': 1,
            'maximum': len(processor.modified_depol),
            'error_message': 'Must be a valid row number from Purple_Curve_Data sheet'
        })
        row += 2
        
        # STEP 4: Trend Addition
        worksheet.merge_range(row, 0, row, 7, 'STEP 4: TREND ADDITION (DEPOL = ADD LINEAR)', section_format)
        row += 1
        
        worksheet.write(row, 0, 'Linear Trend Operation:', self.workbook.add_format({'bold': True}))
        worksheet.write(row, 1, 'ADD linear trend to depol curve')
        worksheet.write(row, 0, 'Formula: Corrected_Current = Original_Current + (m × time + a)')
        row += 2
        
        # STEP 5: Exponential Fitting Results (different form for depol)
        worksheet.merge_range(row, 0, row, 7, 'STEP 5: EXPONENTIAL FITTING RESULTS (y = A×(1-exp(-t/τ)) + C)', section_format)
        row += 1
        
        worksheet.write(row, 0, 'Amplitude (A):', self.workbook.add_format({'bold': True}))
        worksheet.write(row, 1, '', result_format)
        worksheet.write(row, 2, 'pA')
        row += 1
        
        worksheet.write(row, 0, 'Time Constant (τ):', self.workbook.add_format({'bold': True}))
        worksheet.write(row, 1, '', result_format)
        worksheet.write(row, 2, 'ms')
        row += 1
        
        worksheet.write(row, 0, 'Offset (C):', self.workbook.add_format({'bold': True}))
        worksheet.write(row, 1, '', result_format)
        worksheet.write(row, 2, 'pA')
        row += 1
        
        worksheet.write(row, 0, 'R-squared:', self.workbook.add_format({'bold': True}))
        worksheet.write(row, 1, '', result_format)
        row += 2
        
        # STEP 6: Parameter Summary
        worksheet.merge_range(row, 0, row, 7, 'STEP 6: EXTRACTED PARAMETERS SUMMARY', section_format)
        row += 1
        
        summary_format = self.workbook.add_format({'bg_color': '#FFE6E6', 'border': 1})
        
        worksheet.write(row, 0, 'Linear Parameters:', self.workbook.add_format({'bold': True}))
        row += 1
        worksheet.write(row, 0, 'Slope (m):', summary_format)
        worksheet.write_formula(row, 1, '=B8', summary_format)
        worksheet.write(row, 2, 'pA/ms', summary_format)
        row += 1
        worksheet.write(row, 0, 'Intercept (a):', summary_format)
        worksheet.write_formula(row, 1, '=B9', summary_format)
        worksheet.write(row, 2, 'pA', summary_format)
        row += 1
        
        worksheet.write(row, 0, 'Exponential Parameters:', self.workbook.add_format({'bold': True}))
        row += 1
        worksheet.write(row, 0, 'Amplitude (A):', summary_format)
        worksheet.write_formula(row, 1, '=B19', summary_format)
        worksheet.write(row, 2, 'pA', summary_format)
        row += 1
        worksheet.write(row, 0, 'Time Constant (τ):', summary_format)
        worksheet.write_formula(row, 1, '=B20', summary_format)
        worksheet.write(row, 2, 'ms', summary_format)
        row += 1
        worksheet.write(row, 0, 'Offset (C):', summary_format)
        worksheet.write_formula(row, 1, '=B21', summary_format)
        worksheet.write(row, 2, 'pA', summary_format)
        
        # Set column widths
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:G', 12)
    
    def _create_manual_fitting_worksheet(self):
        """Create worksheet with manual fitting tools and helpers."""
        worksheet = self.workbook.add_worksheet('Manual_Fitting_Tools')
        
        title_format = self.workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center',
            'bg_color': '#366092',
            'font_color': 'white'
        })
        
        worksheet.merge_range('A1:F1', 'MANUAL CURVE FITTING TOOLS AND EXAMPLES', title_format)
        
        # Add helper formulas and examples
        example_format = self.workbook.add_format({'bg_color': '#F0F8FF', 'border': 1})
        
        row = 3
        worksheet.write(row, 0, 'EXPONENTIAL FITTING HELPER', self.workbook.add_format({'bold': True, 'font_size': 14}))
        row += 2
        
        worksheet.write(row, 0, 'For HYPERPOLARIZATION (decay): y = A × exp(-t/τ) + C', self.workbook.add_format({'bold': True}))
        row += 1
        worksheet.write(row, 0, 'For DEPOLARIZATION (growth): y = A × (1 - exp(-t/τ)) + C', self.workbook.add_format({'bold': True}))
        row += 2
        
        worksheet.write(row, 0, 'SOLVER SETUP INSTRUCTIONS:', self.workbook.add_format({'bold': True, 'bg_color': '#FFFFCC'}))
        row += 1
        worksheet.write(row, 0, '1. Create helper column calculating predicted values')
        row += 1
        worksheet.write(row, 0, '2. Create column calculating residuals (actual - predicted)²')
        row += 1
        worksheet.write(row, 0, '3. Sum the squared residuals')
        row += 1
        worksheet.write(row, 0, '4. Use Solver to minimize sum of squared residuals')
        row += 1
        worksheet.write(row, 0, '5. Change variables: A, τ, C parameters')
        row += 2
        
        worksheet.write(row, 0, 'PARAMETER CONSTRAINTS:', self.workbook.add_format({'bold': True, 'bg_color': '#E6F3FF'}))
        row += 1
        worksheet.write(row, 0, 'Time constant (τ) > 0.1 ms')
        row += 1
        worksheet.write(row, 0, 'Amplitude (A): For hyperpol > 0, for depol can be negative')
        row += 1
        worksheet.write(row, 0, 'Offset (C): Usually close to baseline current')
        row += 2
        
        # Example calculation area
        worksheet.write(row, 0, 'EXAMPLE CALCULATION AREA:', self.workbook.add_format({'bold': True, 'bg_color': '#E6FFE6'}))
        row += 1
        worksheet.write(row, 0, 'Time (ms)', example_format)
        worksheet.write(row, 1, 'Current (pA)', example_format)
        worksheet.write(row, 2, 'A (parameter)', example_format)
        worksheet.write(row, 3, 'τ (parameter)', example_format)
        worksheet.write(row, 4, 'C (parameter)', example_format)
        worksheet.write(row, 5, 'Predicted', example_format)
        worksheet.write(row, 6, 'Residual²', example_format)
        
        # Set column widths
        worksheet.set_column('A:G', 15)
    
    def _create_instructions_worksheet(self):
        """Create comprehensive instructions worksheet."""
        worksheet = self.workbook.add_worksheet('Instructions')
        
        title_format = self.workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center',
            'bg_color': '#2F5233',
            'font_color': 'white'
        })
        
        header_format = self.workbook.add_format({
            'bold': True,
            'font_size': 14,
            'bg_color': '#70AD47',
            'font_color': 'white'
        })
        
        text_format = self.workbook.add_format({
            'text_wrap': True,
            'valign': 'top'
        })
        
        worksheet.merge_range('A1:F1', 'PURPLE CURVE MANUAL ANALYSIS WORKFLOW', title_format)
        
        row = 3
        worksheet.merge_range(f'A{row}:F{row}', 'COMPLETE WORKFLOW GUIDE', header_format)
        row += 2
        
        instructions = [
            "STEP 1: EXAMINE CHARTS",
            "• Go to 'Charts' worksheet to see your purple curves",
            "• Use the interactive charts to identify fitting regions",
            "• Note time values where linear behavior occurs",
            "• Identify transition points for exponential fitting",
            "",
            "STEP 2: LINEAR FITTING (for both Hyperpol and Depol)",
            "• Go to respective analysis worksheet (Hyperpol_Analysis or Depol_Analysis)",
            "• Enter TWO row numbers for linear fitting region",
            "• Row numbers correspond to Purple_Curve_Data sheet",
            "• Linear parameters (slope, intercept, R²) auto-calculate",
            "• Record these parameters: y = mx + a",
            "",
            "STEP 3: TREND REMOVAL/ADDITION",
            "• HYPERPOLARIZATION: SUBTRACT linear trend from curve",
            "• DEPOLARIZATION: ADD linear trend to curve", 
            "• This prepares data for exponential fitting",
            "• Formula provided in analysis worksheets",
            "",
            "STEP 4: EXPONENTIAL FITTING",
            "• Select THIRD point where exponential behavior starts",
            "• Use Manual_Fitting_Tools worksheet for Solver setup",
            "• Fit appropriate exponential function:",
            "  - Hyperpol: y = A×exp(-t/τ) + C (decay)",
            "  - Depol: y = A×(1-exp(-t/τ)) + C (growth)",
            "",
            "STEP 5: PARAMETER EXTRACTION",
            "• Record all parameters from both fits:",
            "• Linear: slope (m), intercept (a), R²",
            "• Exponential: amplitude (A), time constant (τ), offset (C), R²",
            "• These parameters can be used for model training",
            "",
            "STEP 6: VALIDATION AND DOCUMENTATION",
            "• Check R² values for fit quality (>0.95 is excellent)",
            "• Document point selection rationale",
            "• Save parameter values for training dataset",
            "• Repeat process for multiple files to build training data"
        ]
        
        for instruction in instructions:
            if instruction.startswith("STEP"):
                worksheet.write(row, 0, instruction, self.workbook.add_format({'bold': True, 'font_size': 12, 'bg_color': '#FFFFCC'}))
            else:
                worksheet.write(row, 0, instruction, text_format)
            row += 1
        
        # Set column widths and row heights
        worksheet.set_column('A:F', 20)
        for i in range(3, row):
            worksheet.set_row(i, 20)


def export_purple_curves_with_charts(processor, filename=None):
    """
    Enhanced export function that creates Excel file with automatic charts and manual analysis framework.
    This replaces the previous export function but maintains all current functionality.
    
    Args:
        processor: ActionPotentialProcessor instance
        filename: Optional filename for the Excel file
        
    Returns:
        str: Path to the created Excel file
    """
    exporter = EnhancedExcelExporter()
    return exporter.export_purple_curves_with_charts(processor, filename)