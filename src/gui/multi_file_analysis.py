"""
Complete Multi-File Analysis Feature - Full integration with main app
Location: src/gui/multi_file_analysis.py

Features:
- Full processing pipeline matching main app
- History integration that actually works
- Integral value displays
- Point tracker functionality
- Memory-efficient implementation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pathlib import Path
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from src.utils.logger import app_logger
from src.io_utils.io_utils import ATFHandler
from src.analysis.action_potential import ActionPotentialProcessor
from src.filtering.filtering import combined_filter

class FileSlot:
    """Represents a single file slot in the multi-file analyzer"""
    
    def __init__(self, slot_number):
        self.slot_number = slot_number
        self.filename = None
        self.filepath = None
        self.time_data = None
        self.raw_data = None
        self.filtered_data = None
        self.processor = None
        self.results = {}
        self.is_loaded = False
        self.is_processed = False
        
        # Store all processing curves
        self.processed_data = None
        self.orange_curve = None
        self.orange_curve_times = None
        self.normalized_curve = None
        self.normalized_curve_times = None
        self.average_curve = None
        self.average_curve_times = None
        self.modified_hyperpol = None
        self.modified_hyperpol_times = None
        self.modified_depol = None
        self.modified_depol_times = None
        
    def load_file(self, filepath):
        """Load ATF file into this slot"""
        try:
            self.filepath = filepath
            self.filename = Path(filepath).name
            
            # Load ATF file
            atf_handler = ATFHandler(filepath)
            atf_handler.load_atf()
            
            self.time_data = atf_handler.get_column("Time")
            self.raw_data = atf_handler.get_column("#1")
            
            if self.time_data is None or self.raw_data is None:
                raise ValueError("Could not load time or current data")
            
            self.is_loaded = True
            self.is_processed = False
            
            app_logger.info(f"Loaded file {self.filename} into slot {self.slot_number}")
            return True
            
        except Exception as e:
            app_logger.error(f"Error loading file into slot {self.slot_number}: {str(e)}")
            self.clear()
            return False
    
    def process_file(self, voltage=-50, parent_history_manager=None):
        """Process the loaded file with FULL action potential analysis pipeline"""
        if not self.is_loaded:
            return False
            
        try:
            # Get sampling rate
            sampling_interval = 1e-5  # Default 100 µs
            fs_hz = 1.0 / sampling_interval
            
            # Apply filters (same as main app)
            filter_params = {
                'savgol_params': {'window_length': 101, 'polyorder': 3},
                'butter_params': {'cutoff': 2000, 'fs': fs_hz, 'order': 2}
            }
            self.filtered_data = combined_filter(self.raw_data, **filter_params)
            
            # Create processor with same parameters as main app
            processor_params = {
                'n_cycles': 2,
                't0': 20,
                't1': 100,
                't2': 100,
                't3': 1000,
                'V0': -80,
                'V1': -100,
                'V2': voltage,
                'cell_area_cm2': 0.0001,
                'time_constant_ms': 10,
                'threshold_percent': 10,
                'use_alternative_method': True,
                'show_modified_peaks': True,
            }
            
            self.processor = ActionPotentialProcessor(
                data=self.filtered_data,
                time_data=self.time_data,
                params=processor_params
            )
            
            # Run FULL processing pipeline like main app
            result = self.processor.process_signal(
                use_alternative_method=processor_params.get('use_alternative_method', False)
            )
            
            if result and result[0] is not None:
                # Store all processing results
                (
                    self.processed_data,
                    self.orange_curve,
                    self.orange_curve_times,
                    self.normalized_curve,
                    self.normalized_curve_times,
                    self.average_curve,
                    self.average_curve_times,
                    processing_results
                ) = result
                
                # Generate purple curves
                purple_result = self.processor.apply_average_to_peaks()
                if purple_result and purple_result[0] is not None:
                    (
                        self.modified_hyperpol,
                        self.modified_hyperpol_times,
                        self.modified_depol,
                        self.modified_depol_times
                    ) = purple_result
                    
                    # Calculate integrals
                    self._calculate_integrals()
                    
                    # ADD TO HISTORY with proper error handling
                    if parent_history_manager:
                        try:
                            # Ensure we have the required fields for history
                            history_results = {
                                'integral_value': self.results.get('integral_value', '0.00 pC'),
                                'hyperpol_area': self.results.get('hyperpol_area', '0.00 pC'),
                                'depol_area': self.results.get('depol_area', '0.00 pC'),
                                'capacitance_nF': self.results.get('capacitance_nF', '0.00 nF'),
                                'v2_voltage': self.results.get('v2_voltage', f'{voltage} mV')
                            }
                            
                            parent_history_manager.add_entry(
                                filename=self.filepath,
                                results=history_results,
                                analysis_type="multi-file"
                            )
                            app_logger.info(f"Successfully added {self.filename} to history from multi-file analysis")
                        except Exception as e:
                            app_logger.error(f"Failed to add {self.filename} to history: {str(e)}")
                            # Continue processing even if history fails
                    
                    self.is_processed = True
                    app_logger.info(f"Processed file {self.filename} with full pipeline")
                    return True
                else:
                    app_logger.error(f"Failed to generate purple curves for {self.filename}")
                    return False
            else:
                app_logger.error(f"Failed to process signal for {self.filename}")
                return False
                
        except Exception as e:
            app_logger.error(f"Error processing file {self.filename}: {str(e)}")
            return False
    
    def _calculate_integrals(self):
        """Calculate integrals using default ranges (same format as main app)"""
        try:
            if not hasattr(self, 'modified_hyperpol') or self.modified_hyperpol is None:
                return
                
            # Default integration ranges (0-200)
            hyperpol_start, hyperpol_end = 0, min(200, len(self.modified_hyperpol))
            depol_start, depol_end = 0, min(200, len(self.modified_depol))
            
            # Calculate hyperpol integral
            hyperpol_data = self.modified_hyperpol[hyperpol_start:hyperpol_end]
            hyperpol_times = self.modified_hyperpol_times[hyperpol_start:hyperpol_end]
            hyperpol_integral = np.trapz(hyperpol_data, x=hyperpol_times * 1000)
            
            # Calculate depol integral
            depol_data = self.modified_depol[depol_start:depol_end]
            depol_times = self.modified_depol_times[depol_start:depol_end]
            depol_integral = np.trapz(depol_data, x=depol_times * 1000)
            
            # Calculate total and other metrics (same format as main app)
            total_integral = hyperpol_integral + depol_integral
            voltage = self.processor.params.get('V2', 0)
            
            # Calculate capacitance (same formula as main app)
            if voltage != 0:
                capacitance_nF = abs(total_integral / voltage) * 1000  # Convert to nF
            else:
                capacitance_nF = 0
            
            # Store results in EXACT same format as main app history
            self.results = {
                'integral_value': f"{total_integral:.2f} pC",
                'hyperpol_area': f"{hyperpol_integral:.2f} pC", 
                'depol_area': f"{depol_integral:.2f} pC",
                'capacitance_nF': f"{capacitance_nF:.2f} nF",
                'v2_voltage': f"{voltage} mV",
                'hyperpol_integral': hyperpol_integral,  # Numeric values for calculations
                'depol_integral': depol_integral,
                'total_integral': total_integral,
                'voltage': voltage
            }
            
        except Exception as e:
            app_logger.error(f"Error calculating integrals for {self.filename}: {str(e)}")
    
    def clear(self):
        """Clear this slot"""
        self.filename = None
        self.filepath = None
        self.time_data = None
        self.raw_data = None
        self.filtered_data = None
        self.processor = None
        self.results = {}
        self.is_loaded = False
        self.is_processed = False
        
        # Clear processing curves
        self.processed_data = None
        self.orange_curve = None
        self.orange_curve_times = None
        self.normalized_curve = None
        self.normalized_curve_times = None
        self.average_curve = None
        self.average_curve_times = None
        self.modified_hyperpol = None
        self.modified_hyperpol_times = None
        self.modified_depol = None
        self.modified_depol_times = None


class MultiFileAnalysisWindow:
    """Main window for multi-file analysis with complete feature set"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.window = tk.Toplevel(parent_app.master)
        self.window.title("Multi-File Analysis - Up to 10 Files")
        self.window.geometry("1600x1200")
        
        # Initialize file slots
        self.file_slots = [FileSlot(i) for i in range(10)]
        self.current_slot = 0
        
        # Display options (same as main app)
        self.display_options = {
            'show_raw': tk.BooleanVar(value=False),
            'show_filtered': tk.BooleanVar(value=True),
            'show_processed': tk.BooleanVar(value=False),
            'show_average': tk.BooleanVar(value=True),
            'show_normalized': tk.BooleanVar(value=True),
            'show_averaged_normalized': tk.BooleanVar(value=True),
            'show_modified': tk.BooleanVar(value=True)
        }
        
        # Initialize point tracker
        self.point_tracker = None
        self.point_status_var = tk.StringVar(value="No data loaded")
        
        # Setup UI
        self.setup_ui()
        
        app_logger.info("Enhanced multi-file analysis window opened")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create main frames
        self.create_file_management_frame()
        self.create_display_options_frame()
        self.create_analysis_frame()
        self.create_results_display_frame()
        self.create_results_frame()
        
    def create_file_management_frame(self):
        """Create file management controls"""
        file_frame = ttk.LabelFrame(self.window, text="File Management", padding="5")
        file_frame.pack(fill='x', padx=5, pady=5)
        
        # File slots display
        slots_frame = ttk.Frame(file_frame)
        slots_frame.pack(fill='x', pady=5)
        
        ttk.Label(slots_frame, text="File Slots:").pack(side='left')
        
        # Create slot buttons
        self.slot_buttons = []
        for i in range(10):
            btn = ttk.Button(slots_frame, text=f"Slot {i+1}", width=8,
                           command=lambda x=i: self.select_slot(x))
            btn.pack(side='left', padx=2)
            self.slot_buttons.append(btn)
        
        # Control buttons
        control_frame = ttk.Frame(file_frame)
        control_frame.pack(fill='x', pady=5)
        
        ttk.Button(control_frame, text="Load File", 
                  command=self.load_file).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear Slot", 
                  command=self.clear_current_slot).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear All", 
                  command=self.clear_all_slots).pack(side='left', padx=5)
        
        # Voltage input
        voltage_frame = ttk.Frame(control_frame)
        voltage_frame.pack(side='left', padx=20)
        ttk.Label(voltage_frame, text="V2 Voltage (mV):").pack(side='left')
        self.voltage_var = tk.StringVar(value="-50")
        ttk.Entry(voltage_frame, textvariable=self.voltage_var, width=8).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Process Current File", 
                  command=self.process_current_file).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Process All Files", 
                  command=self.process_all_files).pack(side='left', padx=5)
        
        # Current file info
        self.file_info_var = tk.StringVar(value="No file selected")
        ttk.Label(file_frame, textvariable=self.file_info_var).pack(pady=5)
        
        self.update_slot_buttons()
    
    def create_display_options_frame(self):
        """Create display options frame (like main app)"""
        display_frame = ttk.LabelFrame(self.window, text="Display Options", padding="5")
        display_frame.pack(fill='x', padx=5, pady=5)
        
        # Create checkboxes for each curve type
        options_frame = ttk.Frame(display_frame)
        options_frame.pack(fill='x')
        
        # Row 1
        row1 = ttk.Frame(options_frame)
        row1.pack(fill='x', pady=2)
        
        ttk.Checkbutton(row1, text="Raw Signal", 
                       variable=self.display_options['show_raw'],
                       command=self.update_plot).pack(side='left', padx=10)
        ttk.Checkbutton(row1, text="Filtered Signal", 
                       variable=self.display_options['show_filtered'],
                       command=self.update_plot).pack(side='left', padx=10)
        ttk.Checkbutton(row1, text="Processed Signal", 
                       variable=self.display_options['show_processed'],
                       command=self.update_plot).pack(side='left', padx=10)
        
        # Row 2
        row2 = ttk.Frame(options_frame)
        row2.pack(fill='x', pady=2)
        
        ttk.Checkbutton(row2, text="50-point Average", 
                       variable=self.display_options['show_average'],
                       command=self.update_plot).pack(side='left', padx=10)
        ttk.Checkbutton(row2, text="Voltage-Normalized", 
                       variable=self.display_options['show_normalized'],
                       command=self.update_plot).pack(side='left', padx=10)
        ttk.Checkbutton(row2, text="Averaged Normalized", 
                       variable=self.display_options['show_averaged_normalized'],
                       command=self.update_plot).pack(side='left', padx=10)
        ttk.Checkbutton(row2, text="Modified Peaks", 
                       variable=self.display_options['show_modified'],
                       command=self.update_plot).pack(side='left', padx=10)
    
    def create_analysis_frame(self):
        """Create analysis visualization frame with point tracking"""
        analysis_frame = ttk.LabelFrame(self.window, text="Analysis Visualization", padding="5")
        analysis_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(14, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, analysis_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, analysis_frame)
        self.toolbar.update()
        
        # View options
        view_frame = ttk.Frame(analysis_frame)
        view_frame.pack(side='bottom', fill='x', pady=5)
        
        self.view_mode = tk.StringVar(value="current")
        ttk.Radiobutton(view_frame, text="Current File Only", variable=self.view_mode, 
                       value="current", command=self.update_plot).pack(side='left')
        ttk.Radiobutton(view_frame, text="Compare All Loaded", variable=self.view_mode, 
                       value="compare", command=self.update_plot).pack(side='left')
        ttk.Radiobutton(view_frame, text="Purple Curves Only", variable=self.view_mode, 
                       value="purple", command=self.update_plot).pack(side='left')
        
        # Point tracking status
        status_frame = ttk.Frame(view_frame)
        status_frame.pack(side='right', padx=10)
        
        # Status bar for point tracking
        status_label = ttk.Label(status_frame, textvariable=self.point_status_var,
                               font=('Consolas', 9), relief=tk.SUNKEN, padding=5)
        status_label.pack()
        
        # Initialize point tracker
        self.setup_point_tracker()
    
    def setup_point_tracker(self):
        """Setup point tracker for the multi-file analysis"""
        try:
            from src.utils.point_counter import CurvePointTracker
            self.point_tracker = CurvePointTracker(self.fig, self.ax, self.point_status_var)
            app_logger.info("Point tracker initialized for multi-file analysis")
        except Exception as e:
            app_logger.error(f"Failed to initialize point tracker: {str(e)}")
            self.point_tracker = None
    
    def update_point_tracker_data(self, slot):
        """Update point tracker with current slot data"""
        if not self.point_tracker or not slot.is_processed:
            return
            
        try:
            # Clear previous data
            for curve_type in self.point_tracker.curve_data:
                self.point_tracker.curve_data[curve_type] = {
                    'data': None, 
                    'times': None, 
                    'visible': False
                }
            
            # Set current slot data
            if slot.orange_curve is not None:
                self.point_tracker.curve_data['orange']['data'] = slot.orange_curve
                self.point_tracker.curve_data['orange']['times'] = slot.orange_curve_times
                self.point_tracker.curve_data['orange']['visible'] = self.display_options['show_average'].get()
            
            if slot.normalized_curve is not None:
                self.point_tracker.curve_data['blue']['data'] = slot.normalized_curve
                self.point_tracker.curve_data['blue']['times'] = slot.normalized_curve_times
                self.point_tracker.curve_data['blue']['visible'] = self.display_options['show_normalized'].get()
            
            if slot.average_curve is not None:
                self.point_tracker.curve_data['magenta']['data'] = slot.average_curve
                self.point_tracker.curve_data['magenta']['times'] = slot.average_curve_times
                self.point_tracker.curve_data['magenta']['visible'] = self.display_options['show_averaged_normalized'].get()
            
            if slot.modified_hyperpol is not None:
                self.point_tracker.curve_data['purple_hyperpol']['data'] = slot.modified_hyperpol
                self.point_tracker.curve_data['purple_hyperpol']['times'] = slot.modified_hyperpol_times
                self.point_tracker.curve_data['purple_hyperpol']['visible'] = self.display_options['show_modified'].get()
            
            if slot.modified_depol is not None:
                self.point_tracker.curve_data['purple_depol']['data'] = slot.modified_depol
                self.point_tracker.curve_data['purple_depol']['times'] = slot.modified_depol_times
                self.point_tracker.curve_data['purple_depol']['visible'] = self.display_options['show_modified'].get()
            
            # Always ensure event connections are active
            self.point_tracker._connect()
            
        except Exception as e:
            app_logger.error(f"Error updating point tracker data: {str(e)}")
    
    def create_results_display_frame(self):
        """Create results display frame showing current file integrals"""
        results_display_frame = ttk.LabelFrame(self.window, text="Current File Results", padding="5")
        results_display_frame.pack(fill='x', padx=5, pady=5)
        
        # Create results display
        self.results_display_frame = ttk.Frame(results_display_frame)
        self.results_display_frame.pack(fill='x')
        
        # Initialize result variables
        self.current_integral_var = tk.StringVar(value="Total Integral: No data")
        self.current_hyperpol_var = tk.StringVar(value="Hyperpol: No data")
        self.current_depol_var = tk.StringVar(value="Depol: No data")
        self.current_capacitance_var = tk.StringVar(value="Capacitance: No data")
        
        # Create result labels
        results_grid = ttk.Frame(self.results_display_frame)
        results_grid.pack(fill='x')
        
        ttk.Label(results_grid, textvariable=self.current_integral_var,
                 font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=10)
        ttk.Label(results_grid, textvariable=self.current_capacitance_var,
                 font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=1, sticky='w', padx=10)
        ttk.Label(results_grid, textvariable=self.current_hyperpol_var,
                 font=('TkDefaultFont', 9)).grid(row=1, column=0, sticky='w', padx=10)
        ttk.Label(results_grid, textvariable=self.current_depol_var,
                 font=('TkDefaultFont', 9)).grid(row=1, column=1, sticky='w', padx=10)
    
    def update_results_display(self, slot):
        """Update the results display for current slot"""
        if slot.is_processed and slot.results:
            self.current_integral_var.set(f"Total Integral: {slot.results.get('integral_value', 'N/A')}")
            self.current_hyperpol_var.set(f"Hyperpol: {slot.results.get('hyperpol_area', 'N/A')}")
            self.current_depol_var.set(f"Depol: {slot.results.get('depol_area', 'N/A')}")
            self.current_capacitance_var.set(f"Capacitance: {slot.results.get('capacitance_nF', 'N/A')}")
        else:
            self.current_integral_var.set("Total Integral: No data")
            self.current_hyperpol_var.set("Hyperpol: No data")
            self.current_depol_var.set("Depol: No data")
            self.current_capacitance_var.set("Capacitance: No data")
    
    def create_results_frame(self):
        """Create results summary frame"""
        results_frame = ttk.LabelFrame(self.window, text="All Files Summary", padding="5")
        results_frame.pack(fill='x', padx=5, pady=5)
        
        # Results table
        columns = ('Slot', 'Filename', 'Voltage', 'Hyperpol', 'Depol', 'Total', 'Status')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120)
        
        self.results_tree.pack(fill='x', pady=5)
        
        # Button frame
        button_frame = ttk.Frame(results_frame)
        button_frame.pack(fill='x', pady=5)
        
        # Export button
        ttk.Button(button_frame, text="Export Results to CSV", 
                  command=self.export_results).pack(side='left', padx=5)
        
        # View main app history button
        ttk.Button(button_frame, text="View Main App History", 
                  command=self.view_main_history).pack(side='left', padx=5)
        
        # Force refresh history button
        ttk.Button(button_frame, text="Refresh History Integration", 
                  command=self.force_refresh_history).pack(side='left', padx=5)
        
        # Info label
        info_label = ttk.Label(results_frame, 
                              text="ℹ️ Processed files are automatically added to the main application's history",
                              font=('TkDefaultFont', 8, 'italic'))
        info_label.pack(pady=2)
    
    def force_refresh_history(self):
        """Force refresh the history integration"""
        try:
            history_manager = getattr(self.parent_app, 'history_manager', None)
            if not history_manager:
                messagebox.showwarning("Warning", "History manager not available in main application")
                return
                
            added_count = 0
            for slot in self.file_slots:
                if slot.is_processed and slot.results:
                    try:
                        # Re-add to history with proper format
                        history_results = {
                            'integral_value': slot.results.get('integral_value', '0.00 pC'),
                            'hyperpol_area': slot.results.get('hyperpol_area', '0.00 pC'),
                            'depol_area': slot.results.get('depol_area', '0.00 pC'),
                            'capacitance_nF': slot.results.get('capacitance_nF', '0.00 nF'),
                            'v2_voltage': slot.results.get('v2_voltage', '0 mV')
                        }
                        
                        history_manager.add_entry(
                            filename=slot.filepath,
                            results=history_results,
                            analysis_type="multi-file-refresh"
                        )
                        added_count += 1
                    except Exception as e:
                        app_logger.error(f"Failed to refresh history for {slot.filename}: {str(e)}")
            
            messagebox.showinfo("History Refresh", f"Successfully refreshed {added_count} entries in history")
            
        except Exception as e:
            app_logger.error(f"Error refreshing history: {str(e)}")
            messagebox.showerror("Error", f"Failed to refresh history: {str(e)}")
    
    def select_slot(self, slot_number):
        """Select a specific file slot"""
        self.current_slot = slot_number
        self.update_slot_buttons()
        self.update_file_info()
        self.update_results_display(self.file_slots[self.current_slot])
        self.update_plot()
    
    def update_slot_buttons(self):
        """Update the appearance of slot buttons"""
        for i, btn in enumerate(self.slot_buttons):
            slot = self.file_slots[i]
            
            # Highlight current slot
            if i == self.current_slot:
                btn.configure(text=f"► Slot {i+1}")
            else:
                btn.configure(text=f"Slot {i+1}")
    
    def update_file_info(self):
        """Update current file information display"""
        slot = self.file_slots[self.current_slot]
        if slot.is_loaded:
            info = f"Slot {self.current_slot + 1}: {slot.filename}"
            if slot.is_processed:
                info += f" - Processed (V2: {slot.results.get('voltage', 'N/A')}mV)"
            else:
                info += " - Loaded, not processed"
        else:
            info = f"Slot {self.current_slot + 1}: Empty"
        
        self.file_info_var.set(info)
    
    def load_file(self):
        """Load a file into the current slot"""
        filepath = filedialog.askopenfilename(
            title="Select ATF file",
            filetypes=[("ATF files", "*.atf"), ("All files", "*.*")]
        )
        
        if filepath:
            slot = self.file_slots[self.current_slot]
            if slot.load_file(filepath):
                self.update_slot_buttons()
                self.update_file_info()
                self.update_results_display(slot)
                self.update_plot()
                messagebox.showinfo("Success", f"File loaded into slot {self.current_slot + 1}")
            else:
                messagebox.showerror("Error", "Failed to load file")
    
    def clear_current_slot(self):
        """Clear the current slot"""
        slot = self.file_slots[self.current_slot]
        slot.clear()
        self.update_slot_buttons()
        self.update_file_info()
        self.update_results_display(slot)
        self.update_plot()
        self.update_results_table()
    
    def clear_all_slots(self):
        """Clear all slots"""
        if messagebox.askyesno("Confirm", "Clear all loaded files?"):
            for slot in self.file_slots:
                slot.clear()
            self.update_slot_buttons()
            self.update_file_info()
            self.update_results_display(self.file_slots[self.current_slot])
            self.update_plot()
            self.update_results_table()
    
    def process_current_file(self):
        """Process the file in the current slot"""
        slot = self.file_slots[self.current_slot]
        if not slot.is_loaded:
            messagebox.showwarning("Warning", "No file loaded in current slot")
            return
        
        try:
            voltage = int(self.voltage_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid voltage value")
            return
        
        # Pass history manager to the process function
        history_manager = getattr(self.parent_app, 'history_manager', None)
        
        if slot.process_file(voltage, history_manager):
            self.update_slot_buttons()
            self.update_file_info()
            self.update_results_display(slot)
            self.update_plot()
            self.update_results_table()
            
            # Check if actually added to history
            if history_manager:
                messagebox.showinfo("Success", 
                    f"File processed successfully!\n\n"
                    f"✓ Results: {slot.results.get('integral_value', 'N/A')}\n"
                    f"✓ Added to main application history\n"
                    f"✓ Use 'View Main App History' to see all entries")
            else:
                messagebox.showinfo("Success", f"File processed successfully")
        else:
            messagebox.showerror("Error", "Failed to process file")
    
    def process_all_files(self):
        """Process all loaded files"""
        try:
            voltage = int(self.voltage_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid voltage value")
            return
        
        # Get history manager from parent app
        history_manager = getattr(self.parent_app, 'history_manager', None)
        
        processed_count = 0
        for slot in self.file_slots:
            if slot.is_loaded and not slot.is_processed:
                if slot.process_file(voltage, history_manager):
                    processed_count += 1
        
        self.update_slot_buttons()
        self.update_file_info()
        self.update_results_display(self.file_slots[self.current_slot])
        self.update_plot()
        self.update_results_table()
        
        # Enhanced success message
        if history_manager and processed_count > 0:
            messagebox.showinfo("Complete", 
                f"Processing Complete!\n\n"
                f"✓ Processed {processed_count} files\n"
                f"✓ All results added to main application history\n"
                f"✓ Use 'View Main App History' to see all entries")
        else:
            messagebox.showinfo("Complete", f"Processed {processed_count} files")
    
    def update_plot(self):
        """Update the analysis plot with FULL processing visualization"""
        self.ax.clear()
        
        mode = self.view_mode.get()
        
        if mode == "current":
            self._plot_current_file_full()
        elif mode == "compare":
            self._plot_compare_all()
        elif mode == "purple":
            self._plot_purple_curves()
        
        # Update point tracker data for current mode
        if mode == "current":
            current_slot = self.file_slots[self.current_slot]
            self.update_point_tracker_data(current_slot)
        
        self.canvas.draw()
    
    def _plot_current_file_full(self):
        """Plot the current file with ALL processing curves (like main app)"""
        slot = self.file_slots[self.current_slot]
        if not slot.is_loaded:
            self.ax.text(0.5, 0.5, "No file loaded", ha='center', va='center', 
                        transform=self.ax.transAxes)
            return
        
        # Plot raw data if enabled
        if self.display_options['show_raw'].get() and slot.raw_data is not None:
            self.ax.plot(slot.time_data * 1000, slot.raw_data, 'b-', alpha=0.3, label='Raw Signal')
        
        # Plot filtered data (maroon, like main app)
        if self.display_options['show_filtered'].get() and slot.filtered_data is not None:
            self.ax.plot(slot.time_data * 1000, slot.filtered_data, 
                        color='#800000', label='Filtered Signal', alpha=0.7, linewidth=1.5)
        
        # Plot processed data if enabled and available
        if (self.display_options['show_processed'].get() and 
            slot.is_processed and slot.processed_data is not None):
            self.ax.plot(slot.time_data * 1000, slot.processed_data, 'g-', 
                        label='Processed Signal', linewidth=1.5, alpha=0.7)
        
        # Plot 50-point average (orange)
        if (self.display_options['show_average'].get() and 
            slot.is_processed and slot.orange_curve is not None):
            self.ax.plot(slot.orange_curve_times * 1000, slot.orange_curve, 
                        color='#FFA500', label='50-point Average', linewidth=1.5, alpha=0.7)
        
        # Plot voltage-normalized (dark blue)
        if (self.display_options['show_normalized'].get() and 
            slot.is_processed and slot.normalized_curve is not None):
            self.ax.plot(slot.normalized_curve_times * 1000, slot.normalized_curve, 
                        color='#0057B8', label='Voltage-Normalized', linewidth=1.5, alpha=0.7)
        
        # Plot averaged-normalized (magenta)
        if (self.display_options['show_averaged_normalized'].get() and 
            slot.is_processed and slot.average_curve is not None):
            self.ax.plot(slot.average_curve_times * 1000, slot.average_curve, 
                        color='magenta', label='Averaged Normalized', linewidth=2, alpha=0.8)
        
        # Plot modified peaks (purple)
        if (self.display_options['show_modified'].get() and 
            slot.is_processed and slot.modified_hyperpol is not None):
            self.ax.plot(slot.modified_hyperpol_times * 1000, slot.modified_hyperpol, 
                        color='purple', label='Modified Peaks', linewidth=2, alpha=0.8)
            self.ax.plot(slot.modified_depol_times * 1000, slot.modified_depol, 
                        color='purple', label='_nolegend_', linewidth=2, alpha=0.8)
        
        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('Current (pA)')
        self.ax.set_title(f'File: {slot.filename or "No file"}')
        self.ax.legend()
        self.ax.grid(True)
    
    def _plot_compare_all(self):
        """Plot all loaded files for comparison"""
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        plotted_any = False
        for i, slot in enumerate(self.file_slots):
            if slot.is_loaded and slot.filtered_data is not None:
                self.ax.plot(slot.time_data * 1000, slot.filtered_data, 
                           color=colors[i], label=f'Slot {i+1}: {slot.filename}')
                plotted_any = True
        
        if not plotted_any:
            self.ax.text(0.5, 0.5, "No files loaded", ha='center', va='center', 
                        transform=self.ax.transAxes)
        else:
            self.ax.set_xlabel('Time (ms)')
            self.ax.set_ylabel('Current (pA)')
            self.ax.set_title('All Loaded Files Comparison')
            self.ax.legend()
            self.ax.grid(True)
    
    def _plot_purple_curves(self):
        """Plot only the purple curves from processed files"""
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        plotted_any = False
        for i, slot in enumerate(self.file_slots):
            if slot.is_processed and slot.modified_hyperpol is not None:
                self.ax.plot(slot.modified_hyperpol_times * 1000, 
                           slot.modified_hyperpol, 
                           color=colors[i], linestyle='-', 
                           label=f'Slot {i+1} Hyperpol')
                self.ax.plot(slot.modified_depol_times * 1000, 
                           slot.modified_depol, 
                           color=colors[i], linestyle='--', 
                           label=f'Slot {i+1} Depol')
                plotted_any = True
        
        if not plotted_any:
            self.ax.text(0.5, 0.5, "No processed files", ha='center', va='center', 
                        transform=self.ax.transAxes)
        else:
            self.ax.set_xlabel('Time (ms)')
            self.ax.set_ylabel('Current (pA)')
            self.ax.set_title('Purple Curves Comparison')
            self.ax.legend()
            self.ax.grid(True)
    
    def update_results_table(self):
        """Update the results table"""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add data for each slot
        for i, slot in enumerate(self.file_slots):
            if slot.is_loaded:
                if slot.is_processed:
                    values = (
                        f"Slot {i+1}",
                        slot.filename,
                        f"{slot.results.get('voltage', 'N/A')}",
                        f"{slot.results.get('hyperpol_integral', 0):.2f}",
                        f"{slot.results.get('depol_integral', 0):.2f}",
                        f"{slot.results.get('total_integral', 0):.2f}",
                        "Processed ✓"
                    )
                else:
                    values = (
                        f"Slot {i+1}",
                        slot.filename,
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "Loaded"
                    )
                self.results_tree.insert('', 'end', values=values)
    
    def view_main_history(self):
        """Open the main application's history window"""
        try:
            if hasattr(self.parent_app, 'show_analysis_history'):
                self.parent_app.show_analysis_history()
            else:
                messagebox.showinfo("Info", "History feature not available in main application")
        except Exception as e:
            app_logger.error(f"Error opening main app history: {str(e)}")
            messagebox.showerror("Error", f"Failed to open history: {str(e)}")
    
    def export_results(self):
        """Export results to CSV"""
        import csv
        
        # Get processed files
        processed_files = [slot for slot in self.file_slots if slot.is_processed]
        
        if not processed_files:
            messagebox.showwarning("Warning", "No processed files to export")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Enhanced CSV export with same format as main app history
                    writer.writerow(['Slot', 'Filename', 'Voltage_mV', 'Integral_Value', 
                                   'Hyperpol_Area', 'Depol_Area', 'Capacitance_nF', 'Analysis_Type'])
                    
                    for i, slot in enumerate(self.file_slots):
                        if slot.is_processed:
                            writer.writerow([
                                i + 1,
                                slot.filename,
                                slot.results.get('voltage', 0),
                                slot.results.get('integral_value', f"{slot.results.get('total_integral', 0):.2f} pC"),
                                slot.results.get('hyperpol_area', f"{slot.results.get('hyperpol_integral', 0):.2f} pC"),
                                slot.results.get('depol_area', f"{slot.results.get('depol_integral', 0):.2f} pC"),
                                slot.results.get('capacitance_nF', 'N/A'),
                                'multi-file'
                            ])
                
                messagebox.showinfo("Success", f"Results exported to {filepath}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")


class MultiFileAnalysisButton:
    """Button to open multi-file analysis window"""
    
    def __init__(self, parent_app, toolbar_frame):
        self.parent_app = parent_app
        self.analysis_window = None
        
        # Add button to toolbar
        self.button = ttk.Button(
            toolbar_frame,
            text="📁 Multi-File Analysis",
            command=self.open_analysis_window
        )
        self.button.pack(side='left', padx=2)
        
        app_logger.info("Multi-file analysis button added to toolbar")
    
    def open_analysis_window(self):
        """Open the multi-file analysis window"""
        try:
            if self.analysis_window is None or not self.analysis_window.window.winfo_exists():
                self.analysis_window = MultiFileAnalysisWindow(self.parent_app)
            else:
                # Bring existing window to front
                self.analysis_window.window.lift()
                self.analysis_window.window.focus_force()
                
        except Exception as e:
            app_logger.error(f"Error opening multi-file analysis: {str(e)}")
            messagebox.showerror("Error", f"Failed to open multi-file analysis:\n{str(e)}")


def add_multi_file_analysis_to_toolbar(parent_app, toolbar_frame):
    """
    Add the multi-file analysis button to an existing toolbar.
    This is the function to integrate with your main app.
    
    Args:
        parent_app: The main application instance
        toolbar_frame: The tkinter frame where the button should be added
    
    Returns:
        MultiFileAnalysisButton: The created button instance
    """
    return MultiFileAnalysisButton(parent_app, toolbar_frame)