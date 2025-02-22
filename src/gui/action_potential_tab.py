import tkinter as tk
from tkinter import ttk, messagebox
from src.utils.logger import app_logger
import numpy as np

class ActionPotentialTab:
    def __init__(self, parent, callback):
        """Initialize the action potential analysis tab."""
        self.parent = parent
        self.update_callback = callback
        
        # Create main frame
        self.frame = ttk.LabelFrame(parent, text="Action Potential Analysis")
        self.frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create canvas and scrollbar for scrolling
        self.canvas = tk.Canvas(self.frame, width=260)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack scrollbar components
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Initialize variables
        self.init_variables()
        
        # Setup all UI components in correct order
        self.setup_results_display()  # First, setup results display
        self.setup_parameter_controls()
        self.setup_normalization_points()
        self.setup_integration_range_controls()  # Add integration range controls
        self.setup_regression_controls()
        self.setup_analysis_controls()
        
        # Configure mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Configure canvas resize
        self.frame.bind('<Configure>', self._on_frame_configure)

        self._analysis_results = {}
        
        app_logger.debug("Action potential analysis tab initialized")

    def _on_frame_configure(self, event=None):
        """Handle frame resizing"""
        width = max(260, self.frame.winfo_width() - 25)
        self.canvas.configure(width=width)
        self.canvas.itemconfig('window', width=width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def setup_normalization_points(self):
        """Setup single input field for the starting point"""
        norm_frame = ttk.LabelFrame(self.scrollable_frame, text="Normalization Point")
        norm_frame.pack(fill='x', padx=5, pady=5)

        # Single point input
        point_frame = ttk.Frame(norm_frame)
        point_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(point_frame, text="Starting Point:").pack(side='left')
        norm_entry = ttk.Entry(point_frame, textvariable=self.norm_point, width=10)
        norm_entry.pack(side='left', padx=5)
        
        # Info label
        ttk.Label(
            norm_frame,
            text="Leave blank to use default value (35)",
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=2)

    def get_normalization_points(self):
        """Get segments based on single starting point"""
        try:
            start_point = self.norm_point.get().strip()
            
            if not start_point:
                return None
                
            try:
                n = int(start_point)
                if n < 1:
                    messagebox.showwarning("Invalid Input", "Starting point must be positive")
                    return None
                    
                return {
                    'seg1': (n, n + 199),
                    'seg2': (n + 200, n + 399),
                    'seg3': (n + 400, n + 599),
                    'seg4': (n + 600, n + 799)
                }
                
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid number")
                return None

        except Exception as e:
            app_logger.error(f"Error getting normalization points: {str(e)}")
            return None
        
    def setup_regression_controls(self):
        """Setup regression range selection controls."""
        reg_frame = ttk.LabelFrame(self.scrollable_frame, text="Range Selection")
        reg_frame.pack(fill='x', padx=5, pady=5)

        # Show points toggle
        self.show_points = tk.BooleanVar(value=False)
        points_check = ttk.Checkbutton(
            reg_frame,
            text="Enable Points & Regression",
            variable=self.show_points,
            command=self.on_show_points_change
        )
        points_check.pack(pady=5)

        # Info label
        ttk.Label(
            reg_frame,
            text="Enable to show points and drag directly on the purple curves",
            wraplength=250,
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=2)

        # Current range display
        self.range_display = ttk.Label(reg_frame, text="")
        self.range_display.pack(pady=5)

    def on_show_points_change(self):
        """Handle toggling of range selection mode."""
        show_points = self.show_points.get()
        
        # Check if we have processed data and purple curves
        processor = getattr(self.parent.master, 'action_potential_processor', None)
        has_purple_curves = False
        
        if processor is not None:
            has_purple_curves = (
                hasattr(processor, 'modified_hyperpol') and 
                processor.modified_hyperpol is not None and
                len(processor.modified_hyperpol) > 0 and
                hasattr(processor, 'modified_depol') and 
                processor.modified_depol is not None and
                len(processor.modified_depol) > 0
            )
            app_logger.debug(f"Purple curves check - processor exists: True, has curves: {has_purple_curves}")
        else:
            app_logger.debug("No action potential processor found")

        if show_points and not has_purple_curves:
            app_logger.debug("Showing points requested but no purple curves exist")
            self.show_points.set(False)
            messagebox.showinfo(
                "Analysis Required",
                "Please run 'Analyze Signal' first to generate the purple curves."
            )
            return

        # Update plot display mode for purple curves
        if show_points:
            app_logger.debug("Enabling point selection mode")
            self.modified_display_mode.set("all_points")
            
            # Enable sliders
            for slider in [self.hyperpol_start_slider, self.hyperpol_end_slider,
                         self.depol_start_slider, self.depol_end_slider]:
                slider.configure(state='normal')
        else:
            app_logger.debug("Disabling point selection mode")
            self.modified_display_mode.set("line")
            
            # Disable sliders
            for slider in [self.hyperpol_start_slider, self.hyperpol_end_slider,
                         self.depol_start_slider, self.depol_end_slider]:
                slider.configure(state='disabled')

        # Toggle span selectors in main app
        if hasattr(self.parent, 'master'):
            self.parent.master.toggle_span_selectors(show_points)

        # Update display with current ranges
        self.update_range_display()

        # Update integration with current ranges
        self.on_integration_interval_change()

    def update_range_display(self):
        """Update the display of current integration ranges."""
        if not hasattr(self, 'range_display'):
            return
            
        hyperpol_start = self.hyperpol_start.get()
        hyperpol_end = self.hyperpol_end.get()
        depol_start = self.depol_start.get()
        depol_end = self.depol_end.get()

        self.range_display.config(
            text=f"Hyperpol: {hyperpol_start}-{hyperpol_end}\n"
                f"Depol: {depol_start}-{depol_end}"
        )

    def get_intervals(self):
        """Get current interval settings"""
        try:
            # Get integration intervals for both hyperpol and depol
            integration_ranges = self.get_integration_ranges()

            # Get regression interval if points are shown
            show_points = self.show_points.get() if hasattr(self, 'show_points') else False

            return {
                'integration_ranges': integration_ranges,
                'show_points': show_points
            }
        except Exception as e:
            app_logger.error(f"Error getting intervals: {str(e)}")
            return {
                'integration_ranges': {
                    'hyperpol': {'start': 0, 'end': 200},
                    'depol': {'start': 0, 'end': 200}
                },
                'show_points': False
            }

    def on_method_change(self, *args):
        """Handle changes in integration method"""
        try:
            # Trigger re-analysis with new method if we have already analyzed
            if self.integral_result.get() != "---":
                app_logger.debug(f"Integration method changed, triggering reanalysis")
                self.analyze_signal()
        except Exception as e:
            app_logger.error(f"Error handling method change: {str(e)}")

    def init_variables(self):
        """Initialize control variables with validation"""
        try:
            # Analysis parameters
            self.n_cycles = tk.IntVar(value=2)
            self.t0 = tk.DoubleVar(value=20.0)
            self.t1 = tk.DoubleVar(value=100.0)
            self.t2 = tk.DoubleVar(value=100.0)
            self.V0 = tk.DoubleVar(value=-80.0)
            self.V1 = tk.DoubleVar(value=-100.0)
            self.V2 = tk.DoubleVar(value=-20.0)  

            # Integration method
            self.integration_method = tk.StringVar(value="traditional")
            
            # Results display
            self.integral_result = tk.StringVar(value="---")
            self.hyperpol_result = tk.StringVar(value="---")
            self.depol_result = tk.StringVar(value="---")
            self.capacitance_result = tk.StringVar(value="---")
            self.status_text = tk.StringVar(value="Ready")
            self.progress_var = tk.DoubleVar()
            
            # Display mode variables
            self.show_noisy_original = tk.BooleanVar(value=False)
            self.show_red_curve = tk.BooleanVar(value=True)
            self.show_processed = tk.BooleanVar(value=True)
            self.show_average = tk.BooleanVar(value=True)
            self.show_normalized = tk.BooleanVar(value=True)
            self.show_modified = tk.BooleanVar(value=True)
            self.show_averaged_normalized = tk.BooleanVar(value=True)
            
            # Display modes for each curve type
            self.processed_display_mode = tk.StringVar(value="line")
            self.average_display_mode = tk.StringVar(value="line")
            self.normalized_display_mode = tk.StringVar(value="line")
            self.modified_display_mode = tk.StringVar(value="line")
            self.averaged_normalized_display_mode = tk.StringVar(value="line")

            # Integration and regression intervals
            self.hyperpol_start = tk.IntVar(value=0)
            self.hyperpol_end = tk.IntVar(value=200)
            self.depol_start = tk.IntVar(value=0)
            self.depol_end = tk.IntVar(value=200)
            self.show_points = tk.BooleanVar(value=False)
            
            # Add validation traces
            self.n_cycles.trace_add("write", self.validate_n_cycles)
            self.t0.trace_add("write", self.validate_time_constant)
            self.t1.trace_add("write", self.validate_time_constant)
            self.t2.trace_add("write", self.validate_time_constant)
            self.V2.trace_add("write", self.validate_voltage)
            self.integration_method.trace_add("write", self.on_method_change)
            
            # Normalization points
            self.norm_points = {
                'seg1_start': tk.StringVar(),
                'seg1_end': tk.StringVar(),
                'seg2_start': tk.StringVar(),
                'seg2_end': tk.StringVar(),
                'seg3_start': tk.StringVar(),
                'seg3_end': tk.StringVar(),
                'seg4_start': tk.StringVar(),
                'seg4_end': tk.StringVar()
            }
            
            # Single normalization point
            self.norm_point = tk.StringVar()
            
        except Exception as e:
            app_logger.error(f"Error initializing variables: {str(e)}")
            raise

    def setup_parameter_controls(self):
        """Setup parameter input controls"""
        try:
            param_frame = ttk.LabelFrame(self.scrollable_frame, text="Parameters")
            param_frame.pack(fill='x', padx=5, pady=2)
            
            # Integration method selection
            method_frame = ttk.LabelFrame(param_frame, text="Integration Method")
            method_frame.pack(fill='x', padx=5, pady=(2, 5))
            
            ttk.Radiobutton(method_frame, text="Traditional Method",
                        variable=self.integration_method,
                        value="traditional").pack(anchor='w', padx=5, pady=2)
            
            ttk.Radiobutton(method_frame, text="Averaged Normalized Method",
                        variable=self.integration_method,
                        value="alternative").pack(anchor='w', padx=5, pady=2)
            
            # Method description
            ttk.Label(method_frame, 
                     text="Note: Alternative method uses averaged normalized curves",
                     font=('TkDefaultFont', 8, 'italic')).pack(anchor='w', padx=5, pady=(0, 2))
            
            # Number of cycles
            cycles_frame = ttk.Frame(param_frame)
            cycles_frame.pack(fill='x', padx=5, pady=1)
            ttk.Label(cycles_frame, text="Number of Cycles:").pack(side='left')
            ttk.Entry(cycles_frame, textvariable=self.n_cycles,
                    width=10).pack(side='right')
            
            # Time constants
            time_frame = ttk.Frame(param_frame)
            time_frame.pack(fill='x', padx=5, pady=1)
            time_frame.grid_columnconfigure(1, weight=1)
            time_frame.grid_columnconfigure(3, weight=1)
            time_frame.grid_columnconfigure(5, weight=1)
            
            # t0, t1, t2 controls
            ttk.Label(time_frame, text="t0 (ms):").grid(row=0, column=0, padx=2)
            ttk.Entry(time_frame, textvariable=self.t0, width=8).grid(row=0, column=1, padx=2)
            
            ttk.Label(time_frame, text="t1 (ms):").grid(row=0, column=2, padx=2)
            ttk.Entry(time_frame, textvariable=self.t1, width=8).grid(row=0, column=3, padx=2)
            
            ttk.Label(time_frame, text="t2 (ms):").grid(row=0, column=4, padx=2)
            ttk.Entry(time_frame, textvariable=self.t2, width=8).grid(row=0, column=5, padx=2)
            
            ttk.Label(time_frame, 
                     text="t0: baseline, t1: hyperpolarization, t2: depolarization",
                     font=('TkDefaultFont', 8, 'italic')).grid(row=1, column=0, columnspan=6, pady=(0, 2), sticky='w')
            
            # Voltage levels
            volt_frame = ttk.Frame(param_frame)
            volt_frame.pack(fill='x', padx=5, pady=1)
            volt_frame.grid_columnconfigure(1, weight=1)
            volt_frame.grid_columnconfigure(3, weight=1)
            volt_frame.grid_columnconfigure(5, weight=1)
            
            # V0, V1, V2 controls
            ttk.Label(volt_frame, text="V0 (mV):").grid(row=0, column=0, padx=2)
            ttk.Entry(volt_frame, textvariable=self.V0, width=8).grid(row=0, column=1, padx=2)
            
            ttk.Label(volt_frame, text="V1 (mV):").grid(row=0, column=2, padx=2)
            ttk.Entry(volt_frame, textvariable=self.V1, width=8).grid(row=0, column=3, padx=2)
            
            ttk.Label(volt_frame, text="V2 (mV):").grid(row=0, column=4, padx=2)
            ttk.Entry(volt_frame, textvariable=self.V2, width=8).grid(row=0, column=5, padx=2)
            
            ttk.Label(volt_frame, 
                     text="V0: baseline, V1: hyperpolarization, V2: depolarization",
                     font=('TkDefaultFont', 8, 'italic')).grid(row=1, column=0, columnspan=6, pady=(0, 2), sticky='w')
            
        except Exception as e:
            app_logger.error(f"Error setting up parameter controls: {str(e)}")
            raise

    def setup_integration_range_controls(self):
        """Setup sliders for hyperpolarization and depolarization integration ranges."""
        range_frame = ttk.LabelFrame(self.scrollable_frame, text="Integration Ranges")
        range_frame.pack(fill='x', padx=5, pady=5)

        # Hyperpolarization range
        hyperpol_frame = ttk.LabelFrame(range_frame, text="Hyperpolarization")
        hyperpol_frame.pack(fill='x', padx=5, pady=5)

        # Start slider for hyperpol
        hyperpol_start_frame = ttk.Frame(hyperpol_frame)
        hyperpol_start_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(hyperpol_start_frame, text="Start:").pack(side='left')
        self.hyperpol_start_display = ttk.Label(hyperpol_start_frame, width=5)
        self.hyperpol_start_display.pack(side='right')
        
        self.hyperpol_start = tk.IntVar(value=0)
        self.hyperpol_start_slider = ttk.Scale(
            hyperpol_frame, 
            from_=0, 
            to=199,
            variable=self.hyperpol_start,
            orient='horizontal',
            command=lambda v: self.update_slider_display(v, self.hyperpol_start_display, 'hyperpol_start')
        )
        self.hyperpol_start_slider.pack(fill='x', padx=5)
        self.hyperpol_start_slider.configure(state='disabled')  # Initially disabled

        # End slider for hyperpol
        hyperpol_end_frame = ttk.Frame(hyperpol_frame)
        hyperpol_end_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(hyperpol_end_frame, text="End:").pack(side='left')
        self.hyperpol_end_display = ttk.Label(hyperpol_end_frame, width=5)
        self.hyperpol_end_display.pack(side='right')
        
        self.hyperpol_end = tk.IntVar(value=200)
        self.hyperpol_end_slider = ttk.Scale(
            hyperpol_frame, 
            from_=1, 
            to=200,
            variable=self.hyperpol_end,
            orient='horizontal',
            command=lambda v: self.update_slider_display(v, self.hyperpol_end_display, 'hyperpol_end')
        )
        self.hyperpol_end_slider.pack(fill='x', padx=5)
        self.hyperpol_end_slider.configure(state='disabled')  # Initially disabled

        # Depolarization range
        depol_frame = ttk.LabelFrame(range_frame, text="Depolarization")
        depol_frame.pack(fill='x', padx=5, pady=5)

        # Start slider for depol
        depol_start_frame = ttk.Frame(depol_frame)
        depol_start_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(depol_start_frame, text="Start:").pack(side='left')
        self.depol_start_display = ttk.Label(depol_start_frame, width=5)
        self.depol_start_display.pack(side='right')
        
        self.depol_start = tk.IntVar(value=0)
        self.depol_start_slider = ttk.Scale(
            depol_frame, 
            from_=0, 
            to=199,
            variable=self.depol_start,
            orient='horizontal',
            command=lambda v: self.update_slider_display(v, self.depol_start_display, 'depol_start')
        )
        self.depol_start_slider.pack(fill='x', padx=5)
        self.depol_start_slider.configure(state='disabled')  # Initially disabled

        # End slider for depol
        depol_end_frame = ttk.Frame(depol_frame)
        depol_end_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(depol_end_frame, text="End:").pack(side='left')
        self.depol_end_display = ttk.Label(depol_end_frame, width=5)
        self.depol_end_display.pack(side='right')
        
        self.depol_end = tk.IntVar(value=200)
        self.depol_end_slider = ttk.Scale(
            depol_frame, 
            from_=1, 
            to=200,
            variable=self.depol_end,
            orient='horizontal',
            command=lambda v: self.update_slider_display(v, self.depol_end_display, 'depol_end')
        )
        self.depol_end_slider.pack(fill='x', padx=5)
        self.depol_end_slider.configure(state='disabled')  # Initially disabled

        # Help text
        ttk.Label(
            range_frame,
            text="Adjust sliders to set integration range for each curve (0.5ms per point)",
            wraplength=250,
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=5)

    def setup_results_display(self):
        """Setup the results display section."""
        results_frame = ttk.LabelFrame(self.scrollable_frame, text="Results")
        results_frame.pack(fill='x', padx=5, pady=5)

        # Main integral value
        integral_frame = ttk.Frame(results_frame)
        integral_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(integral_frame, text="Integral Value:", anchor='w').pack(side='left')
        ttk.Label(integral_frame, textvariable=self.integral_result).pack(side='right')

        # Purple curve integrals
        purple_frame = ttk.Frame(results_frame)
        purple_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(purple_frame, text="Purple Curves:", anchor='w').pack(side='left')
        purple_right = ttk.Frame(purple_frame)
        purple_right.pack(side='right')

        # Hyperpolarization (indented)
        hyperpol_frame = ttk.Frame(purple_right)
        hyperpol_frame.pack(fill='x')
        ttk.Label(hyperpol_frame, text="Hyperpol:", width=10, anchor='e').pack(side='left')
        ttk.Label(hyperpol_frame, textvariable=self.hyperpol_result).pack(side='right')

        # Depolarization (indented)
        depol_frame = ttk.Frame(purple_right)
        depol_frame.pack(fill='x')
        ttk.Label(depol_frame, text="Depol:", width=10, anchor='e').pack(side='left')
        ttk.Label(depol_frame, textvariable=self.depol_result).pack(side='right')

        # Linear capacitance
        cap_frame = ttk.Frame(results_frame)
        cap_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(cap_frame, text="Linear Capacitance:", anchor='w').pack(side='left')
        ttk.Label(cap_frame, textvariable=self.capacitance_result).pack(side='right')

        # Progress and status
        progress_frame = ttk.Frame(results_frame)
        progress_frame.pack(fill='x', padx=5, pady=2)
        self.progress = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            mode='determinate'
        )
        self.progress.pack(fill='x', pady=2)
        ttk.Label(progress_frame, textvariable=self.status_text).pack(anchor='w')

    def update_results(self, results):
        """Update displayed results with proper formatting."""
        try:
            if isinstance(results, dict):
                # Update main integral value
                if 'integral_value' in results:
                    self.integral_result.set(results['integral_value'])
                
                # Update hyperpolarization result
                if 'hyperpol_area' in results:
                    # Extract numeric value from string if it ends with ' pC'
                    hyperpol_str = str(results['hyperpol_area'])
                    if ' pC' in hyperpol_str:
                        hyperpol_val = float(hyperpol_str.replace(' pC', ''))
                    else:
                        hyperpol_val = float(hyperpol_str)
                    self.hyperpol_result.set(f"{hyperpol_val:.2f} pC")
                
                # Update depolarization result
                if 'depol_area' in results:
                    # Extract numeric value from string if it ends with ' pC'
                    depol_str = str(results['depol_area'])
                    if ' pC' in depol_str:
                        depol_val = float(depol_str.replace(' pC', ''))
                    else:
                        depol_val = float(depol_str)
                    self.depol_result.set(f"{depol_val:.2f} pC")
                
                # Update linear capacitance
                if 'capacitance_nF' in results:
                    # Extract numeric value from string if it ends with ' nF'
                    cap_str = str(results['capacitance_nF'])
                    if ' nF' in cap_str:
                        cap_val = float(cap_str.replace(' nF', ''))
                    else:
                        cap_val = float(cap_str)
                    self.capacitance_result.set(f"{cap_val:.2f} nF")
                
                # Update status
                self.status_text.set(results.get('status', 'Results updated'))
                
                # Store full results for later access
                self._analysis_results = results.copy()
                
            else:
                self.integral_result.set("Error")
                self.hyperpol_result.set("Error")
                self.depol_result.set("Error")
                self.capacitance_result.set("Error")
                self.status_text.set("Error in results")
                
        except Exception as e:
            app_logger.error(f"Error updating results: {str(e)}")
            self.integral_result.set("Error")
            self.hyperpol_result.set("Error")
            self.depol_result.set("Error")
            self.capacitance_result.set("Error")
            self.status_text.set(f"Error: {str(e)}")

    def update_slider_display(self, value, label, slider_type):
        """Update display label for a slider and handle range validation."""
        try:
            val = int(float(value))
            label.config(text=str(val))
            
            # Ensure end is always greater than start
            if slider_type.endswith('_start'):
                end_var = getattr(self, slider_type.replace('start', 'end'))
                if val >= end_var.get():
                    end_var.set(val + 1)
            elif slider_type.endswith('_end'):
                start_var = getattr(self, slider_type.replace('end', 'start'))
                if val <= start_var.get():
                    start_var.set(val - 1)
                    
            self.on_integration_interval_change()
            
        except Exception as e:
            app_logger.error(f"Error updating slider display: {str(e)}")

    def get_integration_ranges(self):
        """Get current integration ranges for both curves."""
        return {
            'hyperpol': {
                'start': self.hyperpol_start.get(),
                'end': self.hyperpol_end.get()
            },
            'depol': {
                'start': self.depol_start.get(),
                'end': self.depol_end.get()
            }
        }
    
    def setup_analysis_controls(self):
        """Setup analysis controls and display options."""
        analysis_frame = ttk.LabelFrame(self.scrollable_frame, text="Analysis")
        analysis_frame.pack(fill='x', padx=5, pady=5)
        
        # Display options frame
        display_frame = ttk.LabelFrame(analysis_frame, text="Display Options")
        display_frame.pack(fill='x', padx=5, pady=5)
        
        # Show/hide signal checkbuttons
        signals = [
            ("Show Processed Signal", self.show_processed),
            ("Show Red Curve", self.show_red_curve),
            ("Show 50-point Average", self.show_average),
            ("Show Voltage-Normalized", self.show_normalized),
            ("Show Averaged Normalized", self.show_averaged_normalized),
            ("Show Modified Peaks", self.show_modified)
        ]
        
        for text, var in signals:
            ttk.Checkbutton(display_frame, text=text,
                           variable=var,
                           command=self.on_display_change).pack(anchor='w', padx=5)
        
        # Display modes frame
        modes_frame = ttk.Frame(display_frame)
        modes_frame.pack(fill='x', padx=5, pady=5)
        
        # Create mode controls grid
        control_frames = ttk.Frame(modes_frame)
        control_frames.pack(fill='x', expand=True)
        control_frames.grid_columnconfigure(0, weight=1)
        control_frames.grid_columnconfigure(1, weight=1)
        
        # Setup mode controls for each signal type
        mode_configs = [
            ("Processed", self.processed_display_mode, 0, 0),
            ("Average", self.average_display_mode, 0, 1),
            ("Normalized", self.normalized_display_mode, 1, 0),
            ("Modified", self.modified_display_mode, 1, 1),
            ("Avg Normalized", self.averaged_normalized_display_mode, 2, 0)
        ]
        
        for title, var, row, col in mode_configs:
            frame = ttk.LabelFrame(control_frames, text=title)
            frame.grid(row=row, column=col, padx=2, pady=2, sticky='nsew')
            for mode in ["line", "points", "all_points"]:
                ttk.Radiobutton(frame, 
                               text=mode.replace('_', ' ').title(),
                               variable=var,
                               value=mode,
                               command=self.on_display_change).pack(anchor='w')
        
        # Analysis button
        self.analyze_button = ttk.Button(analysis_frame, 
                                       text="Analyze Signal",
                                       command=self.analyze_signal)
        self.analyze_button.pack(pady=5)

    def analyze_signal(self):
        """Perform signal analysis with current parameters."""
        try:
            # Disable button during analysis
            self.analyze_button.state(['disabled'])
            self.status_text.set("Analyzing...")
            self.progress_var.set(0)
            
            # Get parameters and normalization points
            params = self.get_parameters()
            norm_points = self.get_normalization_points()
            if norm_points:
                params['normalization_points'] = norm_points
            
            # Update progress
            self.progress_var.set(50)
            
            # Call callback with parameters
            self.update_callback(params)
            
            # Reset UI
            self.progress_var.set(100)
            self.status_text.set("Analysis complete")
            self.analyze_button.state(['!disabled'])
            
        except Exception as e:
            app_logger.error(f"Error in signal analysis: {str(e)}")
            self.status_text.set(f"Error: {str(e)}")
            self.analyze_button.state(['!disabled'])
            messagebox.showerror("Analysis Error", str(e))

    def on_display_change(self):
        """Handle changes in display options."""
        try:
            params = self.get_parameters()
            self.update_callback(params)
        except Exception as e:
            app_logger.error(f"Error handling display change: {str(e)}")

    def on_integration_interval_change(self, *args):
        """Handle changes to integration interval."""
        try:
            ranges = self.get_integration_ranges()
            if self.update_callback:
                self.update_callback({
                    'integration_ranges': ranges,
                    'show_points': self.show_points.get()
                })
        except Exception as e:
            app_logger.error(f"Error handling integration interval change: {str(e)}")

    def get_parameters(self):
        """Get current analysis parameters with validation."""
        try:
            params = {
                'n_cycles': self.n_cycles.get(),
                't0': self.t0.get(),
                't1': self.t1.get(),
                't2': self.t2.get(),
                'V0': self.V0.get(),
                'V1': self.V1.get(),
                'V2': self.V2.get(),
                'use_alternative_method': self.integration_method.get() == "alternative",
                'integration_ranges': self.get_integration_ranges(),
                'display_options': {
                    'show_noisy_original': self.show_noisy_original.get(),
                    'show_red_curve': self.show_red_curve.get(),
                    'show_processed': self.show_processed.get(),
                    'show_average': self.show_average.get(),
                    'show_normalized': self.show_normalized.get(),
                    'show_modified': self.show_modified.get(),
                    'show_averaged_normalized': self.show_averaged_normalized.get(),
                    'processed_mode': self.processed_display_mode.get(),
                    'average_mode': self.average_display_mode.get(),
                    'normalized_mode': self.normalized_display_mode.get(),
                    'modified_mode': self.modified_display_mode.get(),
                    'averaged_normalized_mode': self.averaged_normalized_display_mode.get()
                }
            }
            
            # Validate parameters
            if params['n_cycles'] < 1:
                raise ValueError("Number of cycles must be positive")
            if params['t0'] <= 0 or params['t1'] <= 0 or params['t2'] <= 0:
                raise ValueError("Time constants must be positive")
            
            return params
            
        except Exception as e:
            app_logger.error(f"Error getting parameters: {str(e)}")
            raise

    def validate_voltage(self, *args):
        """Validate voltage values and trigger reanalysis."""
        try:
            v2 = self.V2.get()
            if abs(v2) > 100:
                messagebox.showwarning("Validation", "Voltage should be between -100mV and +100mV")
                self.V2.set(-20.0)
                return
            
            # If valid and already analyzed once, trigger reanalysis
            if self.integral_result.get() != "---":
                app_logger.debug(f"V2 changed to {v2}mV, triggering reanalysis")
                self.analyze_signal()
                
        except tk.TclError:
            self.V2.set(-20.0)

    def validate_time_constant(self, *args):
        """Validate time constants."""
        try:
            t1, t2 = self.t1.get(), self.t2.get()
            if t1 <= 0 or t2 <= 0:
                messagebox.showwarning("Validation", "Time constants must be positive")
                if t1 <= 0: self.t1.set(100.0)
                if t2 <= 0: self.t2.set(100.0)
        except tk.TclError:
            self.t1.set(100.0)
            self.t2.set(100.0)

    def validate_n_cycles(self, *args):
        """Validate number of cycles."""
        try:
            value = self.n_cycles.get()
            if value < 1:
                self.n_cycles.set(1)
                messagebox.showwarning("Validation", "Number of cycles must be at least 1")
            elif value > 10:
                self.n_cycles.set(10)
                messagebox.showwarning("Validation", "Maximum number of cycles is 10")
        except tk.TclError:
            self.n_cycles.set(2)

    def reset(self):
        """Reset the tab to initial state."""
        try:
            # Reset parameters
            self.n_cycles.set(2)
            self.t1.set(100.0)
            self.t2.set(100.0)
            self.V0.set(-80.0)
            self.V1.set(-100.0)
            self.V2.set(-20.0)
            
            # Reset results
            self.integral_result.set("---")
            self.hyperpol_result.set("---")
            self.depol_result.set("---")
            self.capacitance_result.set("---")
            self.status_text.set("Ready")
            self.progress_var.set(0)
            
            # Reset display options
            self.show_processed.set(True)
            self.show_red_curve.set(True)
            self.show_average.set(True)
            self.show_normalized.set(True)
            self.show_modified.set(True)
            self.show_averaged_normalized.set(True)
            
            # Reset display modes
            for var in [self.processed_display_mode, self.average_display_mode,
                       self.normalized_display_mode, self.modified_display_mode,
                       self.averaged_normalized_display_mode]:
                var.set("line")
            
            # Reset ranges
            self.hyperpol_start.set(0)
            self.hyperpol_end.set(200)
            self.depol_start.set(0)
            self.depol_end.set(200)
            
            # Enable analyze button
            self.analyze_button.state(['!disabled'])
            
        except Exception as e:
            app_logger.error(f"Error resetting tab: {str(e)}")
            raise