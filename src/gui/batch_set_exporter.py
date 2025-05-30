"""
Batch Set Exporter - Building on the working single-file export method
Location: src/gui/batch_set_exporter.py
"""

import os
import csv
import numpy as np
from pathlib import Path
import datetime
import re
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from collections import defaultdict
from src.utils.logger import app_logger
from src.io_utils.io_utils import ATFHandler
from src.analysis.action_potential import ActionPotentialProcessor
from src.filtering.filtering import combined_filter

class BatchSetExporter:
    """Batch exporter based on the working single-file method"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        
    def export_folder_by_sets(self):
        """Export ATF files organized by sets to Excel files"""
        try:
            # Select folder
            folder_path = filedialog.askdirectory(
                title="Select folder containing ATF files"
            )
            
            if not folder_path:
                return False
            
            folder_path = Path(folder_path)
            
            # Find all ATF files
            atf_files = list(folder_path.glob("*.atf"))
            
            if not atf_files:
                messagebox.showwarning("No Files", "No ATF files found")
                return False
            
            # Organize by sets
            file_sets = self._organize_files_by_sets(atf_files)
            
            if not file_sets:
                messagebox.showerror("Error", "Could not parse file names")
                return False
            
            # Select output directory
            output_dir = filedialog.askdirectory(
                title="Select output directory"
            )
            
            if not output_dir:
                return False
            
            output_dir = Path(output_dir) / "ExportedSets"
            output_dir.mkdir(exist_ok=True)
            
            # Process each set
            results = []
            for set_number, files_info in file_sets.items():
                success = self._process_set(set_number, files_info, output_dir)
                if success:
                    results.append(f"✓ Set {set_number}: Success")
                else:
                    results.append(f"✗ Set {set_number}: Failed")
            
            # Show results
            messagebox.showinfo("Export Complete", "\n".join(results))
            return True
            
        except Exception as e:
            app_logger.error(f"Error in batch export: {str(e)}")
            messagebox.showerror("Error", f"Export failed: {str(e)}")
            return False
    
    def _organize_files_by_sets(self, atf_files):
        """Parse filenames and organize by set number"""
        file_sets = defaultdict(list)
        # Megengedőbb minta: 8 számjegy, aláhúzás, 3-4 számjegyű sorszám, aláhúzás,
        # tetszőleges hosszú set-szám, aláhúzás, előjeles feszültség, végül mV majd .atf (kis- vagy nagybetűvel)
        # A `re.IGNORECASE` zászlóval az ATF kiterjesztés lehet nagybetűs is.
        pattern = re.compile(r'(\d{8}_\d{3,4})_(\d+)_([+-]?\d+)mV\.atf$', re.IGNORECASE)

        def _fallback_parse(name: str):
            """Tartalék parser, ha a regex nem illeszkedik.
            Visszatér (file_number, set_number, voltage) vagy None.
            Elv: a név aláhúzás mentén bontása.
            Példa: 20250528_000_1_-50mV.atf ->
            parts = ['20250528', '000', '1', '-50mV']"""
            try:
                base, ext = os.path.splitext(name)
                if ext.lower() != '.atf':
                    return None
                parts = base.split('_')
                if len(parts) < 4:
                    return None
                file_number = f"{parts[0]}_{parts[1]}"  # dátum + sorszám
                set_number = int(parts[2])
                voltage_str = parts[3]
                # Távolítsuk el a 'mV' vagy 'mv' végét, majd konvertáljuk int-re
                voltage = int(voltage_str.replace('mV', '').replace('mv', ''))
                return file_number, set_number, voltage
            except Exception:
                return None
        
        for file_path in atf_files:
            name = file_path.name
            match = pattern.match(name)
            if match:
                file_number = match.group(1)
                set_number = int(match.group(2))
                voltage = int(match.group(3))
            else:
                # Próbálkozunk a tartalék parserrel
                parsed = _fallback_parse(name)
                if parsed is None:
                    app_logger.warning(f"Nem sikerült értelmezni a fájlnevet: {name}")
                    continue
                file_number, set_number, voltage = parsed

            file_sets[set_number].append({
                'file_path': file_path,
                'filename': name,
                'file_number': file_number,
                'voltage': voltage,
                'sheet_name': file_number
            })
        
        # Sort files within sets
        for set_number in file_sets:
            file_sets[set_number].sort(key=lambda x: x['file_number'])
        
        return dict(file_sets)
    
    def _process_set(self, set_number, files_info, output_dir):
        """Process a single set and create Excel file"""
        try:
            # Create workbook
            wb = Workbook()
            
            # Process each file
            processed_count = 0
            for file_info in files_info:
                if self._process_file_to_sheet(wb, file_info):
                    processed_count += 1
            
            if processed_count == 0:
                app_logger.error(f"No files processed for set {set_number}")
                return False
            
            # Remove default sheet
            if "Sheet" in wb.sheetnames:
                wb.remove(wb["Sheet"])
            
            # Save workbook
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"Set_{set_number}_{timestamp}.xlsx"
            wb.save(filename)
            
            app_logger.info(f"Set {set_number} exported: {processed_count} files")
            return True
            
        except Exception as e:
            app_logger.error(f"Error processing set {set_number}: {str(e)}")
            return False
    
    def _process_file_to_sheet(self, wb, file_info):
        """Process single file and add to workbook - using the working method"""
        try:
            # Load file
            atf_handler = ATFHandler(file_info['file_path'])
            atf_handler.load_atf()
            
            time_data = atf_handler.get_column("Time")
            raw_data = atf_handler.get_column("#1")
            
            if time_data is None or raw_data is None:
                return False
            
            # Get sampling rate
            sampling_interval = atf_handler.get_sampling_rate()
            fs_hz = 1.0 / sampling_interval if sampling_interval > 0 else 10000.0
            
            # Apply filters (same as working version)
            filter_params = {
                'savgol_params': {'window_length': 101, 'polyorder': 3},
                'butter_params': {'cutoff': 2000, 'fs': fs_hz, 'order': 2}
            }
            filtered_data = combined_filter(raw_data, **filter_params)
            
            # Create processor
            processor_params = {
                'n_cycles': 2,
                't0': 20,
                't1': 100,
                't2': 100,
                't3': 1000,
                'V0': -80,
                'V1': -100,
                'V2': file_info['voltage'],
                'cell_area_cm2': 0.0001,
                'time_constant_ms': 10,
                'threshold_percent': 10,
                'use_alternative_method': True,
                'show_modified_peaks': True,
            }
            
            processor = ActionPotentialProcessor(
                data=filtered_data, 
                time_data=time_data, 
                params=processor_params
            )
            
            # Process data (call methods directly like your working version does)
            processor.baseline_correction_initial()
            processor.advanced_baseline_normalization()
            processor.process_orange_curve_with_spike_removal()
            processor.normalized_curve, processor.normalized_curve_times = processor.calculate_normalized_curve()
            processor.average_curve, processor.average_curve_times = processor.calculate_segment_average()
            
            # Generate purple curves
            result = processor.apply_average_to_peaks()
            if result and result[0] is not None:
                processor.modified_hyperpol = result[0]
                processor.modified_hyperpol_times = result[1]
                processor.modified_depol = result[2]
                processor.modified_depol_times = result[3]
            else:
                app_logger.error(f"Failed to generate purple curves for {file_info['filename']}")
                return False
            
            # Calculate integrals (simple approach like your working version)
            hyperpol_integral = 0
            depol_integral = 0
            
            if (hasattr(processor, 'modified_hyperpol') and 
                hasattr(processor, 'modified_hyperpol_times')):
                hyperpol_integral = np.trapz(
                    processor.modified_hyperpol, 
                    x=processor.modified_hyperpol_times * 1000
                )
            
            if (hasattr(processor, 'modified_depol') and 
                hasattr(processor, 'modified_depol_times')):
                depol_integral = np.trapz(
                    processor.modified_depol,
                    x=processor.modified_depol_times * 1000
                )
            
            # Store integral values on processor (like your working version expects)
            processor.integral_value = f"{hyperpol_integral + depol_integral:.2f} pC"
            processor.hyperpol_area = f"{hyperpol_integral:.2f} pC"
            processor.depol_area = f"{depol_integral:.2f} pC"
            processor.purple_integral_value = (
                f"Hyperpol: {hyperpol_integral:.2f} pC, "
                f"Depol: {depol_integral:.2f} pC"
            )
            
            # Add sheet with data
            self._write_sheet(wb, file_info, processor)
            return True
            
        except Exception as e:
            app_logger.error(f"Error processing {file_info['filename']}: {str(e)}")
            return False
    
    def _write_sheet(self, wb, file_info, processor):
        """Write data to Excel sheet - matching your working CSV format"""
        # Create sheet
        sheet_name = file_info['sheet_name']
        if sheet_name in wb.sheetnames:
            sheet_name = f"{sheet_name}_v2"
        ws = wb.create_sheet(title=sheet_name)
        
        # Get integral values (same as your working version)
        results_dict = {
            'integral_value': getattr(processor, 'integral_value', 'N/A'),
            'hyperpol_area': getattr(processor, 'hyperpol_area', 'N/A'),
            'depol_area': getattr(processor, 'depol_area', 'N/A'),
            'purple_integral_value': getattr(processor, 'purple_integral_value', 'N/A')
        }
        
        row = 1
        
        # File info
        ws[f'A{row}'] = "Original ATF File:"
        ws[f'B{row}'] = file_info['filename']
        row += 1
        
        ws[f'A{row}'] = "V2 Voltage (mV):"
        ws[f'B{row}'] = file_info['voltage']
        row += 2
        
        # Overall integral
        ws[f'A{row}'] = "Overall Integral (pC):"
        ws[f'B{row}'] = results_dict['integral_value']
        ws[f'A{row}'].font = Font(bold=True)
        row += 2
        
        # Purple Hyperpol Section
        ws[f'A{row}'] = "PURPLE HYPERPOL CURVE"
        ws[f'A{row}'].font = Font(bold=True)
        row += 1
        
        ws[f'A{row}'] = "Hyperpol Integral from the ap:"
        ws[f'B{row}'] = results_dict['hyperpol_area']
        row += 1
        
        # Headers
        ws[f'A{row}'] = "Index"
        ws[f'B{row}'] = "Hyperpol_pA"
        ws[f'C{row}'] = "Hyperpol_time_ms"
        for col in ['A', 'B', 'C']:
            ws[f'{col}{row}'].font = Font(bold=True)
        row += 1
        
        # Data
        if hasattr(processor, 'modified_hyperpol') and hasattr(processor, 'modified_hyperpol_times'):
            hyperpol = processor.modified_hyperpol
            hyperpol_times = processor.modified_hyperpol_times
            for i in range(len(hyperpol)):
                ws[f'A{row}'] = i + 1
                ws[f'B{row}'] = float(f"{hyperpol[i]:.7f}")
                ws[f'C{row}'] = float(f"{hyperpol_times[i]*1000:.7f}")
                row += 1
        row += 1
        
        # Purple Depol Section
        ws[f'A{row}'] = "PURPLE DEPOL CURVE"
        ws[f'A{row}'].font = Font(bold=True)
        row += 1
        
        ws[f'A{row}'] = "Depol Integral from the ap:"
        ws[f'B{row}'] = results_dict['depol_area']
        row += 1
        
        # Headers
        ws[f'A{row}'] = "Index"
        ws[f'B{row}'] = "Depol_pA"
        ws[f'C{row}'] = "Depol_time_ms"
        for col in ['A', 'B', 'C']:
            ws[f'{col}{row}'].font = Font(bold=True)
        row += 1
        
        # Data
        if hasattr(processor, 'modified_depol') and hasattr(processor, 'modified_depol_times'):
            depol = processor.modified_depol
            depol_times = processor.modified_depol_times
            for i in range(len(depol)):
                ws[f'A{row}'] = i + 1
                ws[f'B{row}'] = float(f"{depol[i]:.7f}")
                ws[f'C{row}'] = float(f"{depol_times[i]*1000:.7f}")
                row += 1
        
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
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width


# Button class for integration
class BatchSetExportButton:
    """Batch set export button for integration with the main app"""
    
    def __init__(self, parent_app, toolbar_frame):
        self.parent_app = parent_app
        self.exporter = BatchSetExporter(parent_app)
        
        # Add export button to toolbar
        self.export_button = tk.Button(
            toolbar_frame, 
            text="📊 Export Sets", 
            command=self.export_sets,
            relief="raised",
            bg="#2ECC71",
            fg="white",
            font=("Arial", 9, "bold"),
            padx=8,
            pady=2
        )
        self.export_button.pack(side='left', padx=2)
        
        app_logger.info("Batch set export button added to toolbar")
    
    def export_sets(self):
        """Export ATF file sets"""
        try:
            self.exporter.export_folder_by_sets()
        except Exception as e:
            app_logger.error(f"Error in set export: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export sets:\n{str(e)}")


# This is the function being imported by your app
def add_set_export_to_toolbar(parent_app, toolbar_frame):
    """
    Add the batch set export button to an existing toolbar.
    This is the function your app is trying to import.
    
    Args:
        parent_app: The main application instance
        toolbar_frame: The tkinter frame where the button should be added
    
    Returns:
        BatchSetExportButton: The created button instance
    """
    return BatchSetExportButton(parent_app, toolbar_frame)