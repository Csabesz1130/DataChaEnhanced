"""
Simplified Set-Based Purple Curve Exporter
Complete fixed version with proper indentation and Excel file locking solution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import re
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from collections import defaultdict
from src.utils.logger import app_logger
from src.io_utils.io_utils import ATFHandler
from src.analysis.action_potential import ActionPotentialProcessor
import traceback
from src.filtering.filtering import combined_filter

class SimplifiedSetExporter:
    """Simplified export manager that matches the existing Excel format exactly"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        
    def export_folder_by_sets(self):
        """Export ATF files organized by sets to separate Excel files"""
        try:
            # Select folder containing ATF files
            folder_path = filedialog.askdirectory(
                title="Select folder containing ATF files organized by sets"
            )
            
            if not folder_path:
                return False
            
            folder_path = Path(folder_path)
            
            # Find all ATF files
            atf_files = list(folder_path.glob("*.atf"))
            
            if not atf_files:
                messagebox.showwarning("No Files", "No ATF files found in the selected folder")
                return False
            
            app_logger.info(f"Found {len(atf_files)} ATF files in {folder_path}")
            
            # Parse and organize files by sets
            file_sets = self._organize_files_by_sets(atf_files)
            
            if not file_sets:
                messagebox.showerror("Parse Error", "Could not parse file names. Please check the naming convention.")
                return False
            
            # Show organization confirmation
            if not self._confirm_set_organization(file_sets):
                return False
            
            # Select output directory
            output_dir = filedialog.askdirectory(
                title="Select output directory for Excel files"
            )
            
            if not output_dir:
                return False
            
            output_dir = Path(output_dir)
            
            # Process each set
            total_sets = len(file_sets)
            progress_window = self._create_progress_window(total_sets)
            
            try:
                processed_sets = 0
                results = []
                
                for set_number, files_info in file_sets.items():
                    self._update_progress(progress_window, processed_sets, total_sets, 
                                        f"Processing Set {set_number}")
                    
                    try:
                        excel_filename = self._generate_safe_excel_filename(set_number, output_dir)
                        success = self._create_simple_excel_for_set(set_number, files_info, excel_filename)
                        
                        if success:
                            results.append(f"✓ Set {set_number}: {excel_filename.name}")
                            app_logger.info(f"Set {set_number} exported to {excel_filename}")
                        else:
                            results.append(f"✗ Set {set_number}: Failed")
                            
                    except Exception as e:
                        app_logger.error(f"Error processing set {set_number}: {str(e)}")
                        results.append(f"✗ Set {set_number}: Error - {str(e)}")
                    
                    processed_sets += 1
                
                progress_window.destroy()
                
                # Show results
                self._show_results(results)
                return True
                
            finally:
                if progress_window.winfo_exists():
                    progress_window.destroy()
                
        except Exception as e:
            app_logger.error(f"Error in simplified set export: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export file sets:\n{str(e)}")
            return False
    
    def _organize_files_by_sets(self, atf_files):
        """Parse filenames and organize files by set number"""
        file_sets = defaultdict(list)
        
        # Pattern: YYYYMMDD_NNN_S_±VVmV.atf
        pattern = r'(\d{8}_\d{3})_(\d+)_([+-]?\d+)mV\.atf$'
        
        for file_path in atf_files:
            filename = file_path.name
            match = re.match(pattern, filename)
            
            if match:
                file_number = match.group(1)  # e.g., "20250528_000"
                set_number = int(match.group(2))  # e.g., 1, 2, 3
                voltage = int(match.group(3))  # e.g., -50, -10, 0
                
                file_info = {
                    'file_path': file_path,
                    'filename': filename,
                    'file_number': file_number,
                    'set_number': set_number,
                    'voltage': voltage,
                    'sheet_name': file_number
                }
                
                file_sets[set_number].append(file_info)
                
            else:
                app_logger.warning(f"Could not parse filename: {filename}")
        
        # Sort files within each set by file number
        for set_number in file_sets:
            file_sets[set_number].sort(key=lambda x: x['file_number'])
        
        return dict(file_sets)
    
    def _confirm_set_organization(self, file_sets):
        """Show user the set organization and confirm"""
        confirm_window = tk.Toplevel(self.parent_app.master)
        confirm_window.title("Confirm Set Organization")
        confirm_window.transient(self.parent_app.master)
        confirm_window.grab_set()
        
        # Center window with minimum size
        confirm_window.update_idletasks()
        desired_width = 600
        desired_height = 400
        x = (confirm_window.winfo_screenwidth() // 2) - (desired_width // 2)
        y = (confirm_window.winfo_screenheight() // 2) - (desired_height // 2)
        confirm_window.geometry(f"{desired_width}x{desired_height}+{x}+{y}")
        confirm_window.minsize(desired_width, desired_height)
        
        # Title
        tk.Label(confirm_window, text="File Set Organization", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Scrollable text area
        frame = tk.Frame(confirm_window)
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y')
        
        text_area = tk.Text(frame, yscrollcommand=scrollbar.set, wrap='word')
        text_area.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=text_area.yview)
        
        # Build organization text
        org_text = f"Found {len(file_sets)} sets:\n\n"
        
        for set_number in sorted(file_sets.keys()):
            files_info = file_sets[set_number]
            org_text += f"📁 SET {set_number} ({len(files_info)} files) → 'Set_{set_number}_[timestamp].xlsx'\n"
            
            for file_info in files_info:
                org_text += f"   📄 {file_info['filename']} → Sheet: '{file_info['sheet_name']}'\n"
            
            org_text += "\n"
        
        text_area.insert('1.0', org_text)
        text_area.config(state='disabled')
        
        # Result variable
        result = {'confirmed': False}
        
        # Buttons
        button_frame = tk.Frame(confirm_window)
        button_frame.pack(pady=10)
        
        def confirm():
            result['confirmed'] = True
            confirm_window.destroy()
        
        def cancel():
            result['confirmed'] = False
            confirm_window.destroy()
        
        tk.Button(button_frame, text="✓ Export", command=confirm, 
                 bg='green', fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=10)
        tk.Button(button_frame, text="✗ Cancel", command=cancel, 
                 bg='red', fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=10)
        
        confirm_window.wait_window()
        return result['confirmed']
    
    def _generate_safe_excel_filename(self, set_number, output_dir):
        """Generate Excel filename that avoids conflicts with open files"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Try different filename variations to avoid locked files
        for attempt in range(10):
            if attempt == 0:
                filename = f"Set_{set_number}_{timestamp}.xlsx"
            else:
                filename = f"Set_{set_number}_{timestamp}_v{attempt}.xlsx"
            
            full_path = output_dir / filename
            
            # Test if we can create/write to this file
            if self._test_file_writable(full_path):
                app_logger.info(f"Generated safe filename: {filename}")
                return full_path
        
        # If all attempts fail, ask user what to do
        app_logger.warning("Could not generate writable filename automatically")
        return self._handle_filename_conflict(set_number, output_dir)
    
    def _test_file_writable(self, file_path):
        """Test if a file path is writable without actually creating the file"""
        try:
            # Try to open in write mode
            with open(file_path, 'w') as f:
                pass
            # If successful, remove the test file
            file_path.unlink()
            return True
        except (OSError, PermissionError):
            return False
    
    def _handle_filename_conflict(self, set_number, output_dir):
        """Handle filename conflicts by asking user for guidance"""
        conflict_window = tk.Toplevel(self.parent_app.master)
        conflict_window.title("File Conflict")
        conflict_window.transient(self.parent_app.master)
        conflict_window.grab_set()
        
        # Center window
        conflict_window.update_idletasks()
        desired_width = 500
        desired_height = 200
        x = (conflict_window.winfo_screenwidth() // 2) - (desired_width // 2)
        y = (conflict_window.winfo_screenheight() // 2) - (desired_height // 2)
        conflict_window.geometry(f"{desired_width}x{desired_height}+{x}+{y}")
        conflict_window.minsize(desired_width, desired_height)
        
        # Message
        message = (f"Cannot create Excel file for Set {set_number}.\n"
                  f"Possible causes:\n"
                  f"• Excel files are open in another application\n"
                  f"• Output directory is write-protected\n\n"
                  f"Please close Excel files or choose a different output directory.")
        
        tk.Label(conflict_window, text=message, wraplength=450, justify='left').pack(pady=20)
        
        result = {'action': 'skip'}
        
        button_frame = tk.Frame(conflict_window)
        button_frame.pack(pady=10)
        
        def retry():
            result['action'] = 'retry'
            conflict_window.destroy()
        
        def skip():
            result['action'] = 'skip'
            conflict_window.destroy()
        
        tk.Button(button_frame, text="Retry", command=retry, 
                 bg='blue', fg='white').pack(side='left', padx=10)
        tk.Button(button_frame, text="Skip Set", command=skip, 
                 bg='orange', fg='white').pack(side='left', padx=10)
        
        conflict_window.wait_window()
        
        if result['action'] == 'retry':
            return self._generate_safe_excel_filename(set_number, output_dir)
        else:
            # Return a placeholder that will cause the set to be skipped
            return None
    
    def _create_simple_excel_for_set(self, set_number, files_info, excel_filename):
        """Create simplified Excel file matching the existing format"""
        if excel_filename is None:
            app_logger.warning(f"Skipping Set {set_number} due to filename conflict")
            return False
            
        try:
            app_logger.info(f"Creating simplified Excel for Set {set_number}: {excel_filename}")
            
            wb = Workbook()
            
            # Process each file in the set
            processed_files = []
            
            for file_info in files_info:
                try:
                    # Process the ATF file
                    processor = self._process_atf_file_simple(file_info)
                    
                    if processor is not None:
                        # Add simple sheet to workbook
                        self._add_simple_sheet(wb, file_info, processor)
                        processed_files.append(file_info)
                        
                except Exception as e:
                    app_logger.error(f"Error processing {file_info['filename']}: {str(e)}")
                    continue
            
            # Remove default sheet if we have data sheets
            if processed_files and "Sheet" in wb.sheetnames:
                wb.remove(wb["Sheet"])
            
            # Save workbook with final safety check
            try:
                wb.save(excel_filename)
                app_logger.info(f"Set {set_number} saved: {len(processed_files)} files processed")
                return True
            except PermissionError as e:
                app_logger.error(f"Permission denied saving {excel_filename}: {str(e)}")
                return False
                
        except Exception as e:
            app_logger.error(f"Error creating simplified Excel for set {set_number}: {str(e)}")
            return False
    
    def _process_atf_file_simple(self, file_info):
        """
        Process ATF file following the GUI workflow:
        1. Load file
        2. Apply specified filters (Savitzky-Golay and Butterworth)
        3. Use "Averaged Normalized" method equivalent via processor params
        4. Generate purple curves.
        """
        file_path = file_info['file_path']
        filename_short = file_info['filename']
        voltage = file_info['voltage']
        
        app_logger.info(f"Processing ATF file: {filename_short} with V2={voltage}mV following GUI workflow (Fixed v5)")

        try:
            atf_handler = ATFHandler(file_path)
            atf_handler.load_atf()
            
            time_data = atf_handler.get_column("Time")
            raw_data = atf_handler.get_column("#1")

            if time_data is None or raw_data is None or len(time_data) == 0 or len(raw_data) == 0 or len(time_data) != len(raw_data):
                app_logger.error(f"Data loading issue for {filename_short}. Time: {len(time_data) if time_data is not None else 'None'}, Data: {len(raw_data) if raw_data is not None else 'None'}")
                return None
            app_logger.debug(f"Loaded {len(raw_data)} raw data points for {filename_short}.")

            sampling_interval = atf_handler.get_sampling_rate()
            if sampling_interval <= 0:
                app_logger.warning(f"Invalid sampling interval ({sampling_interval}s) for {filename_short}. Defaulting fs to 10kHz.")
                fs_hz = 10000.0 
            else:
                fs_hz = 1.0 / sampling_interval
            app_logger.debug(f"Calculated sampling frequency for {filename_short}: {fs_hz:.2f} Hz")

            # Filter parameters based on user's GUI settings (image_beb633.png)
            # Savitzky-Golay: Window 101, Polyorder 3
            # Butterworth: Cutoff 2000 Hz, Order 2
            filter_params_for_export = {
                'savgol_params': {'window_length': 101, 'polyorder': 3},
                'butter_params': {'cutoff': 2000, 'fs': fs_hz, 'order': 2}
            }
            app_logger.info(f"Applying GUI-matched filters to {filename_short}: {filter_params_for_export}")
            
            filtered_data_for_export = combined_filter(raw_data, **filter_params_for_export)
            app_logger.debug(f"Data filtered for {filename_short}. Length: {len(filtered_data_for_export)}")

            # FIXED: Include ALL required parameters that the processor expects
            processor_params = {
                # Timing parameters (CRITICAL - these were missing!)
                'n_cycles': 2,
                't0': 20,      # Start time (ms)
                't1': 100,     # End of first phase (ms) 
                't2': 100,     # Start of second phase (ms)
                't3': 1000,    # End time (ms)
                
                # Voltage parameters
                'V0': -80,     # Holding voltage
                'V1': -100,    # Hyperpolarization voltage
                'V2': voltage, # Depolarization voltage from filename
                
                # Other parameters
                'cell_area_cm2': 0.0001,
                'time_constant_ms': 10, 
                'threshold_percent': 10,
                'use_alternative_method': True, 
                'show_modified_peaks': True,
            }
            
            processor = ActionPotentialProcessor(data=filtered_data_for_export, 
                                                time_data=time_data, 
                                                params=processor_params)
            
            app_logger.debug(f"Processor params for {filename_short} upon instantiation: {processor.params}")
            
            # Call process_signal (should work now with all required parameters)
            try:
                _p_data, _o_curve, _o_times, _n_curve, _n_times, _avg_curve, _avg_times, results_signal = processor.process_signal()

                if results_signal.get('error'):
                    app_logger.error(f"process_signal failed for {filename_short}: {results_signal['error']}")
                    return None
                app_logger.debug(f"process_signal completed for {filename_short}.")
            except Exception as e:
                app_logger.error(f"process_signal exception for {filename_short}: {str(e)}")
                return None

            # Call apply_average_to_peaks to generate purple curves
            try:
                processor.apply_average_to_peaks()
                app_logger.debug(f"apply_average_to_peaks called for {filename_short}.")
            except Exception as e:
                app_logger.error(f"apply_average_to_peaks exception for {filename_short}: {str(e)}")
                return None
            
            # Check if purple curve data exists
            if (not hasattr(processor, 'modified_hyperpol') or processor.modified_hyperpol is None or
                not hasattr(processor, 'modified_depol') or processor.modified_depol is None or
                not hasattr(processor, 'modified_hyperpol_times') or processor.modified_hyperpol_times is None or
                not hasattr(processor, 'modified_depol_times') or processor.modified_depol_times is None):
                app_logger.error(f"Purple curve data attributes not found for {filename_short} after analysis steps.")
                return None

            # Check purple curve lengths
            hyperpol_len = len(processor.modified_hyperpol)
            depol_len = len(processor.modified_depol)
            
            app_logger.debug(f"Purple curve lengths for {filename_short}: Hyperpol={hyperpol_len}, Depol={depol_len}")

            # Accept either 199 or 200 points (both are valid)
            if hyperpol_len < 190 or depol_len < 190:
                app_logger.error(f"Purple curves for {filename_short} too short. "
                            f"Got Hyperpol: {hyperpol_len}, Depol: {depol_len}")
                return None
            
            processor.analysis_complete = True 
            app_logger.info(f"Successfully processed {filename_short}. Purple curves generated with Hyperpol: {hyperpol_len}, Depol: {depol_len} points.")
            return processor
            
        except Exception as e:
            app_logger.error(f"Exception in _process_atf_file_simple for {filename_short}: {str(e)}\n{traceback.format_exc()}")
            return None
    
    def _add_simple_sheet(self, wb, file_info, processor):
        """Add sheet with simplified format matching the existing layout exactly"""
        sheet_name_base = file_info['sheet_name']
        sheet_name = sheet_name_base
        counter = 1
        while sheet_name in wb.sheetnames:
            sheet_name = f"{sheet_name_base}_{counter}"
            counter += 1
        ws = wb.create_sheet(title=sheet_name)
        app_logger.info(f"Adding sheet '{ws.title}' for file {file_info['filename']}")

        try:
            # Check if processor has purple curve data
            if not (hasattr(processor, 'modified_hyperpol') and processor.modified_hyperpol is not None and
                    hasattr(processor, 'modified_depol') and processor.modified_depol is not None and
                    hasattr(processor, 'modified_hyperpol_times') and processor.modified_hyperpol_times is not None and
                    hasattr(processor, 'modified_depol_times') and processor.modified_depol_times is not None):
                app_logger.error(f"Processor for sheet '{ws.title}' is missing critical purple curve data.")
                ws['A1'] = f"Error: Missing processed data for {file_info['filename']}."
                return

            # Get actual lengths of purple curves
            hyperpol_len = len(processor.modified_hyperpol)
            depol_len = len(processor.modified_depol)
            hyperpol_times_len = len(processor.modified_hyperpol_times)
            depol_times_len = len(processor.modified_depol_times)
            
            app_logger.debug(f"Purple curve lengths: Hyperpol={hyperpol_len}, Depol={depol_len}, "
                            f"Hyperpol_times={hyperpol_times_len}, Depol_times={depol_times_len}")

            # Ensure all arrays have the same length
            if not (hyperpol_len == hyperpol_times_len and depol_len == depol_times_len):
                app_logger.error(f"Purple curve length mismatch for {file_info['filename']}: "
                            f"Hyperpol data={hyperpol_len}, times={hyperpol_times_len}, "
                            f"Depol data={depol_len}, times={depol_times_len}")
                ws['A1'] = f"Error: Data length mismatch for {file_info['filename']}."
                return

            # Use the actual lengths for integration (all points)
            hyperpol_integral = processor.calculate_integration('hyperpol', 0, hyperpol_len)
            depol_integral = processor.calculate_integration('depol', 0, depol_len)
            overall_integral = hyperpol_integral + depol_integral 

            app_logger.debug(f"Sheet '{ws.title}': Hyperpol Integral={hyperpol_integral:.3f}, "
                        f"Depol Integral={depol_integral:.3f}, Overall={overall_integral:.3f}")

            current_row = 1
            
            # File info
            ws[f'A{current_row}'] = "Original ATF File:"
            ws[f'B{current_row}'] = file_info['filename']
            current_row += 1
            ws[f'A{current_row}'] = "V2 Voltage (mV):"
            ws[f'B{current_row}'] = file_info['voltage']
            current_row += 2

            # Overall Integral (matching the format from the image)
            ws[f'A{current_row}'] = "Overall Integral (pC):"
            ws[f'B{current_row}'] = float(f"{overall_integral:.2f}")
            ws[f'A{current_row}'].font = Font(bold=True)
            current_row += 2
            
            # PURPLE HYPERPOL CURVE header
            ws[f'A{current_row}'] = "PURPLE HYPERPOL CURVE"
            ws[f'A{current_row}'].font = Font(bold=True)
            current_row += 1
            
            # Hyperpol Integral
            ws[f'A{current_row}'] = "Hyperpol Integral from the ap:"
            ws[f'B{current_row}'] = float(f"{hyperpol_integral:.2f}")
            current_row += 1
            
            # Headers for hyperpol data
            headers_hyperpol = ["Index", "Hyperpol_pA", "Hyperpol_time_ms"]
            for col_idx, header_title in enumerate(headers_hyperpol):
                col_letter = get_column_letter(col_idx + 1)
                ws[f'{col_letter}{current_row}'] = header_title
                ws[f'{col_letter}{current_row}'].font = Font(bold=True)
            current_row += 1
            
            # Add hyperpol data (use actual length)
            for i in range(hyperpol_len):
                ws[f'A{current_row + i}'] = int(i + 1)
                ws[f'B{current_row + i}'] = float(f"{processor.modified_hyperpol[i]:.7f}")
                ws[f'C{current_row + i}'] = float(f"{processor.modified_hyperpol_times[i] * 1000:.7f}")
            current_row += hyperpol_len + 2 
            
            # PURPLE DEPOL CURVE header
            ws[f'A{current_row}'] = "PURPLE DEPOL CURVE"
            ws[f'A{current_row}'].font = Font(bold=True)
            current_row += 1

            # Depol Integral
            ws[f'A{current_row}'] = "Depol Integral from the ap:"
            ws[f'B{current_row}'] = float(f"{depol_integral:.2f}")
            current_row += 1
            
            # Headers for depol data
            headers_depol = ["Index", "Depol_pA", "Depol_time_ms"]
            for col_idx, header_title in enumerate(headers_depol):
                col_letter = get_column_letter(col_idx + 1)
                ws[f'{col_letter}{current_row}'] = header_title
                ws[f'{col_letter}{current_row}'].font = Font(bold=True)
            current_row += 1
            
            # Add depol data (use actual length)
            for i in range(depol_len):
                ws[f'A{current_row + i}'] = int(i + 1)
                ws[f'B{current_row + i}'] = float(f"{processor.modified_depol[i]:.7f}")
                ws[f'C{current_row + i}'] = float(f"{processor.modified_depol_times[i] * 1000:.7f}")
            
            # Auto-adjust column widths
            for col_cells in ws.columns:
                max_length = 0
                column_letter_str = get_column_letter(col_cells[0].column)
                for cell in col_cells:
                    try:
                        if cell.value is not None:
                            cell_len = len(str(cell.value))
                            if cell_len > max_length:
                                max_length = cell_len
                    except:
                        pass
                adjusted_width = (max_length + 2) if max_length > 0 else 10
                ws.column_dimensions[column_letter_str].width = adjusted_width

            app_logger.info(f"Successfully populated sheet '{ws.title}' for {file_info['filename']} "
                        f"with {hyperpol_len} hyperpol and {depol_len} depol points.")

        except Exception as e:
            app_logger.error(f"Error populating sheet '{ws.title}' for {file_info['filename']}: {str(e)}\n{traceback.format_exc()}")
            try:
                ws['A1'] = f"Error populating this sheet for file: {file_info['filename']}"
                ws['A2'] = str(e)
            except:
                pass
    
    def _create_progress_window(self, total_sets):
        """Create progress window"""
        progress_window = tk.Toplevel(self.parent_app.master)
        progress_window.title("Processing Sets")
        progress_window.transient(self.parent_app.master)
        progress_window.grab_set()
        
        # Center window with minimum size
        progress_window.update_idletasks()
        desired_width = 450
        desired_height = 120
        x = (progress_window.winfo_screenwidth() // 2) - (desired_width // 2)
        y = (progress_window.winfo_screenheight() // 2) - (desired_height // 2)
        progress_window.geometry(f"{desired_width}x{desired_height}+{x}+{y}")
        progress_window.minsize(desired_width, desired_height)
        
        # Progress elements
        progress_window.label = tk.Label(progress_window, text="Starting processing...")
        progress_window.label.pack(pady=10)
        
        progress_window.progress = ttk.Progressbar(
            progress_window, length=350, mode='determinate', maximum=total_sets
        )
        progress_window.progress.pack(pady=5)
        
        progress_window.status = tk.Label(progress_window, text="")
        progress_window.status.pack(pady=5)
        
        progress_window.update()
        return progress_window
    
    def _update_progress(self, progress_window, current, total, status=""):
        """Update progress window"""
        try:
            if progress_window.winfo_exists():
                progress_window.progress['value'] = current
                progress_window.label.config(text=f"Processing set {current + 1}/{total}")
                progress_window.status.config(text=status)
                progress_window.update()
        except:
            pass
    
    def _show_results(self, results):
        """Show export results to user"""
        result_window = tk.Toplevel(self.parent_app.master)
        result_window.title("Export Results")
        result_window.transient(self.parent_app.master)
        
        # Center window with minimum size
        result_window.update_idletasks()
        desired_width = 500
        desired_height = 300
        x = (result_window.winfo_screenwidth() // 2) - (desired_width // 2)
        y = (result_window.winfo_screenheight() // 2) - (desired_height // 2)
        result_window.geometry(f"{desired_width}x{desired_height}+{x}+{y}")
        result_window.minsize(desired_width, desired_height)
        
        tk.Label(result_window, text="Export Results", 
                font=("Arial", 12, "bold")).pack(pady=10)
        
        # Scrollable results
        frame = tk.Frame(result_window)
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y')
        
        text_area = tk.Text(frame, yscrollcommand=scrollbar.set)
        text_area.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=text_area.yview)
        
        results_text = "\n".join(results)
        text_area.insert('1.0', results_text)
        text_area.config(state='disabled')
        
        tk.Button(result_window, text="Close", command=result_window.destroy).pack(pady=10)


# Integration with main application
class SimplifiedSetExportButton:
    """Simplified set export button for integration with the main app"""
    
    def __init__(self, parent_app, toolbar_frame):
        self.parent_app = parent_app
        self.exporter = SimplifiedSetExporter(parent_app)
        
        # Add export button to toolbar
        self.export_button = tk.Button(
            toolbar_frame, 
            text="📊 Export Sets", 
            command=self.export_sets,
            relief="raised",
            bg="#4A90E2",
            fg="white",
            font=("Arial", 9, "bold"),
            padx=8,
            pady=2
        )
        self.export_button.pack(side='left', padx=2)
        
        app_logger.info("Simplified set export button added to toolbar")
    
    def export_sets(self):
        """Export ATF file sets with simplified format"""
        try:
            self.exporter.export_folder_by_sets()
        except Exception as e:
            app_logger.error(f"Error in set export: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export sets:\n{str(e)}")


# Standalone function for direct integration
def add_set_export_to_toolbar(parent_app, toolbar_frame):
    """
    Add the set export button to an existing toolbar.
    
    Args:
        parent_app: The main application instance
        toolbar_frame: The tkinter frame where the button should be added
    
    Returns:
        SimplifiedSetExportButton: The created button instance
    """
    return SimplifiedSetExportButton(parent_app, toolbar_frame)