"""
Excel export package for Signal Analyzer application.
This package provides functionality for exporting signal data to Excel,
with support for different integration methods and regression correction.
"""

from .excel_export import export_to_excel, add_excel_export_to_app, update_excel_results
from .regression_utils import compute_regression_params, apply_curve_correction
from .integration_calculator import (
    resample_data,
    calculate_integral_scenario_a,
    calculate_integral_scenario_b
)
from .export_backup_manager import backup_manager
from .curve_analysis_export import export_curve_analysis_to_excel
from .set_based_export import export_sets_to_excel

__all__ = [
    'export_to_excel',
    'add_excel_export_to_app',
    'update_excel_results',
    'compute_regression_params',
    'apply_curve_correction',
    'resample_data',
    'calculate_integral_scenario_a',
    'calculate_integral_scenario_b',
    'backup_manager',
    'export_curve_analysis_to_excel',
    'export_sets_to_excel'
]