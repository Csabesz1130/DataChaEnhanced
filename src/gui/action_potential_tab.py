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
        
        # Create controls within scrollable_frame
        self.setup_parameter_controls()
        self.setup_normalization_points()
        
        # Add interval controls BEFORE analysis controls
        self.setup_interval_controls()  # Add this line
        
        self.setup_analysis_controls()
        
        # Configure mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Configure canvas resize
        self.frame.bind('<Configure>', self._on_frame_configure)

        self._analysis_results = {}
        
        app_logger.debug("Action potential analysis tab initialized")

    def _on_frame_configure(self, event=None):
        """Handle frame resizing"""
        # Update the canvas size while maintaining minimum width
        width = max(260, self.frame.winfo_width() - 25)  # -25 for scrollbar and padding
        self.canvas.configure(width=width)
        
        # Update the scrollable frame width
        self.canvas.itemconfig('window', width=width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    # Add to src/gui/action_potential_tab.py

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

        # Add help text
        ttk.Label(
            range_frame,
            text="Adjust sliders to set integration range for each curve (0.5ms per point)",
            wraplength=250,
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=5)

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

    def setup_interval_controls(self):
        """Setup sliders for controlling integration and regression intervals"""
        # Create a frame after parameter controls and before analysis controls
        interval_frame = ttk.LabelFrame(self.scrollable_frame, text="Purple Curve Analysis")
        interval_frame.pack(fill='x', padx=5, pady=5)

        # Show points control in its own frame
        points_frame = ttk.Frame(interval_frame)
        points_frame.pack(fill='x', padx=5, pady=2)

        # Add explanatory text
        ttk.Label(
            points_frame,
            text="First run analysis, then enable points to show regression lines",
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=(0,5))

        # Add "Show Points & Regression" checkbox with larger text and padding
        self.show_points = tk.BooleanVar(value=False)
        points_check = ttk.Checkbutton(
            points_frame, 
            text="Show Points & Fit Regression Lines",
            variable=self.show_points,
            command=self.on_show_points_change,
            style='Bold.TCheckbutton'  # Use bold style for emphasis
        )
        points_check.pack(pady=5, padx=5)

        # Regression interval controls
        regression_frame = ttk.LabelFrame(interval_frame, text="Regression Range")
        regression_frame.pack(fill='x', padx=5, pady=5)

        # Start slider with formatted display
        reg_start_frame = ttk.Frame(regression_frame)
        reg_start_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(reg_start_frame, text="Start Point:").pack(side='left')
        
        self.reg_start_display = ttk.Label(reg_start_frame, width=5)
        self.reg_start_display.pack(side='right')
        
        self.regression_start = tk.IntVar(value=0)
        self.reg_start_slider = ttk.Scale(
            reg_start_frame, 
            from_=0, 
            to=199,
            variable=self.regression_start,
            orient='horizontal',
            command=lambda v: self.update_slider_display(v, self.reg_start_display)
        )
        self.reg_start_slider.pack(side='left', fill='x', expand=True)

        # End slider with formatted display
        reg_end_frame = ttk.Frame(regression_frame)
        reg_end_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(reg_end_frame, text="End Point:").pack(side='left')
        
        self.reg_end_display = ttk.Label(reg_end_frame, width=5)
        self.reg_end_display.pack(side='right')
        
        self.regression_end = tk.IntVar(value=200)
        self.reg_end_slider = ttk.Scale(
            reg_end_frame, 
            from_=1, 
            to=200,
            variable=self.regression_end,
            orient='horizontal',
            command=lambda v: self.update_slider_display(v, self.reg_end_display)
        )
        self.reg_end_slider.pack(side='left', fill='x', expand=True)

        # Add instruction text
        ttk.Label(
            regression_frame,
            text="Drag on plot or use sliders to adjust regression range",
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=(0,5))

    # Add to ActionPotentialTab class:

    def setup_regression_controls(self):
        """Setup regression range selection controls."""
        reg_frame = ttk.LabelFrame(self.scrollable_frame, text="Regression Range")
        reg_frame.pack(fill='x', padx=5, pady=5)

        # Show points toggle
        self.show_points = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            reg_frame,
            text="Enable Range Selection",
            variable=self.show_points,
            command=self.on_show_points_change
        ).pack(pady=5)

        # Info label
        ttk.Label(
            reg_frame,
            text="Drag directly on the purple curves to select integration ranges",
            wraplength=250,
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=2)

        # Display current ranges
        self.range_display = ttk.Label(reg_frame, text="")
        self.range_display.pack(pady=5)

    def on_show_points_change(self):
        """Handle toggling of range selection mode."""
        show_points = self.show_points.get()
        
        if show_points and not hasattr(self.parent.master, 'action_potential_processor'):
            self.show_points.set(False)
            messagebox.showinfo(
                "Analysis Required",
                "Please run 'Analyze Signal' first to generate the purple curves."
            )
            return

        # Update plot display mode
        if show_points:
            self.modified_display_mode.set("all_points")

        # Toggle span selectors in main app
        if hasattr(self.parent, 'master'):
            self.parent.master.toggle_span_selectors(show_points)

        # Update integration with current ranges
        self.on_integration_interval_change()

    def update_range_display(self):
        """Update the display of current integration ranges."""
        hyperpol_start = self.hyperpol_start.get()
        hyperpol_end = self.hyperpol_end.get()
        depol_start = self.depol_start.get()
        depol_end = self.depol_end.get()

        self.range_display.config(
            text=f"Hyperpol: {hyperpol_start}-{hyperpol_end}\n"
                f"Depol: {depol_start}-{depol_end}"
        )

    def update_slider_ranges(self, data_length):
        """Update slider ranges based on data length"""
        try:
            # Update integration sliders
            self.int_start_slider.configure(to=data_length-1)
            self.int_end_slider.configure(to=data_length-1)
            self.integration_end.set(data_length-1)

            # Update regression sliders
            self.reg_start_slider.configure(to=data_length-1)
            self.reg_end_slider.configure(to=data_length-1)
            self.regression_end.set(data_length-1)

        except Exception as e:
            app_logger.error(f"Error updating slider ranges: {str(e)}")

    def on_integration_interval_change(self, *args):
        """Handle changes to integration interval"""
        try:
            start = self.integration_start.get()
            end = self.integration_end.get()

            # Ensure end is greater than start
            if start >= end:
                self.integration_end.set(start + 1)
                end = start + 1

            if self.update_callback:
                self.update_callback({
                    'integration_interval': (start, end),
                    'show_points': self.show_points.get()
                })

        except Exception as e:
            app_logger.error(f"Error handling integration interval change: {str(e)}")

    def on_regression_interval_change(self, *args):
        """Handle changes to regression interval"""
        try:
            start = self.regression_start.get()
            end = self.regression_end.get()

            # Ensure end is greater than start
            if start >= end:
                self.regression_end.set(start + 1)
                end = start + 1

            # Only trigger update if show_points is enabled
            if self.show_points.get() and self.update_callback:
                params = self.get_parameters()
                params.update({
                    'show_points': True,
                    'regression_interval': (start, end)
                })
                self.update_callback(params)

        except Exception as e:
            app_logger.error(f"Error handling regression interval change: {str(e)}")

    def update_slider_display(self, value, label):
        """Update the display label for a slider with formatted integer."""
        try:
            val = int(float(value))
            label.config(text=str(val))
            self.on_regression_interval_change()
        except:
            pass

    def get_intervals(self):
        """Get current interval settings"""
        return {
            'integration_interval': (
                self.integration_start.get(),
                self.integration_end.get()
            ),
            'regression_interval': (
                self.regression_start.get(),
                self.regression_end.get()
            ),
            'show_points': self.show_points.get()
        }

    # Update the init_variables method in src/gui/action_potential_tab.py

    def init_variables(self):
        """Initialize control variables with validation"""
        try:
            # Analysis parameters with validation
            self.n_cycles = tk.IntVar(value=2)
            self.t0 = tk.DoubleVar(value=20.0)
            self.t1 = tk.DoubleVar(value=100.0)
            self.t2 = tk.DoubleVar(value=100.0)
            self.V0 = tk.DoubleVar(value=-80.0)
            self.V1 = tk.DoubleVar(value=-100.0)
            self.V2 = tk.DoubleVar(value=-20.0)  

            # Integration method selection
            self.integration_method = tk.StringVar(value="traditional")
            
            # Results display
            self.integral_value = tk.StringVar(value="No analysis performed")
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

            # Integration and regression interval variables
            self.integration_start = tk.IntVar(value=0)
            self.integration_end = tk.IntVar(value=200)
            self.regression_start = tk.IntVar(value=0)
            self.regression_end = tk.IntVar(value=200)
            self.show_points = tk.BooleanVar(value=False)
            
            # Add validation traces
            self.n_cycles.trace_add("write", self.validate_n_cycles)
            self.t0.trace_add("write", self.validate_time_constant)
            self.t1.trace_add("write", self.validate_time_constant)
            self.t2.trace_add("write", self.validate_time_constant)
            self.V2.trace_add("write", self.validate_voltage)
            self.integration_method.trace_add("write", self.on_method_change)
            
            # Initialize variables for normalization points
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
            
            app_logger.debug("Variables initialized successfully")
            
        except Exception as e:
            app_logger.error(f"Error initializing variables: {str(e)}")
            raise

    def validate_voltage(self, *args):
        """Validate voltage values and trigger reanalysis"""
        try:
            v2 = self.V2.get()
            # Typical range check (-100mV to +100mV)
            if abs(v2) > 100:
                messagebox.showwarning("Validation", "Voltage should be between -100mV and +100mV")
                self.V2.set(-20.0)  # Reset to default
                return
            
            # If voltage is valid and we've already performed analysis once,
            # trigger reanalysis with new voltage
            if self.integral_value.get() != "No analysis performed":
                app_logger.debug(f"V2 changed to {v2}mV, triggering reanalysis")
                self.analyze_signal()  # This will recalculate everything with new V2
                
        except tk.TclError:
            # Invalid float - reset to default
            self.V2.set(-20.0)

    def setup_parameter_controls(self):
        """Setup parameter input controls"""
        try:
            param_frame = ttk.LabelFrame(self.scrollable_frame, text="Parameters")
            param_frame.pack(fill='x', padx=5, pady=2)
            
            # Integration method selection at the top
            method_frame = ttk.LabelFrame(param_frame, text="Integration Method")
            method_frame.pack(fill='x', padx=5, pady=(2, 5))
            
            ttk.Radiobutton(method_frame, text="Traditional Method",
                        variable=self.integration_method,
                        value="traditional").pack(anchor='w', padx=5, pady=2)
            
            ttk.Radiobutton(method_frame, text="Averaged Normalized Method",
                        variable=self.integration_method,
                        value="alternative").pack(anchor='w', padx=5, pady=2)
            
            # Method description label
            method_desc = ttk.Label(method_frame, text="Note: Alternative method uses averaged normalized curves",
                                font=('TkDefaultFont', 8, 'italic'))
            method_desc.pack(anchor='w', padx=5, pady=(0, 2))
            
            # Number of cycles
            cycles_frame = ttk.Frame(param_frame)
            cycles_frame.pack(fill='x', padx=5, pady=1)
            ttk.Label(cycles_frame, text="Number of Cycles:").pack(side='left')
            ttk.Entry(cycles_frame, textvariable=self.n_cycles,
                    width=10).pack(side='right')
            
            # Time constants - Grid layout for compactness
            time_frame = ttk.Frame(param_frame)
            time_frame.pack(fill='x', padx=5, pady=1)
            
            time_frame.grid_columnconfigure(1, weight=1)
            time_frame.grid_columnconfigure(3, weight=1)
            time_frame.grid_columnconfigure(5, weight=1)
            
            # t0 controls
            ttk.Label(time_frame, text="t0 (ms):").grid(row=0, column=0, padx=2)
            ttk.Entry(time_frame, textvariable=self.t0, width=8).grid(row=0, column=1, padx=2)
            
            # t1 controls
            ttk.Label(time_frame, text="t1 (ms):").grid(row=0, column=2, padx=2)
            ttk.Entry(time_frame, textvariable=self.t1, width=8).grid(row=0, column=3, padx=2)
            
            # t2 controls
            ttk.Label(time_frame, text="t2 (ms):").grid(row=0, column=4, padx=2)
            ttk.Entry(time_frame, textvariable=self.t2, width=8).grid(row=0, column=5, padx=2)
            
            # Add time constants description
            time_desc = ttk.Label(time_frame, 
                                text="t0: baseline, t1: hyperpolarization, t2: depolarization",
                                font=('TkDefaultFont', 8, 'italic'))
            time_desc.grid(row=1, column=0, columnspan=6, pady=(0, 2), sticky='w')
            
            # Voltage levels - Grid layout
            volt_frame = ttk.Frame(param_frame)
            volt_frame.pack(fill='x', padx=5, pady=1)
            
            volt_frame.grid_columnconfigure(1, weight=1)
            volt_frame.grid_columnconfigure(3, weight=1)
            volt_frame.grid_columnconfigure(5, weight=1)
            
            # V0 controls
            ttk.Label(volt_frame, text="V0 (mV):").grid(row=0, column=0, padx=2)
            ttk.Entry(volt_frame, textvariable=self.V0, width=8).grid(row=0, column=1, padx=2)
            
            # V1 controls
            ttk.Label(volt_frame, text="V1 (mV):").grid(row=0, column=2, padx=2)
            ttk.Entry(volt_frame, textvariable=self.V1, width=8).grid(row=0, column=3, padx=2)
            
            # V2 controls
            ttk.Label(volt_frame, text="V2 (mV):").grid(row=0, column=4, padx=2)
            ttk.Entry(volt_frame, textvariable=self.V2, width=8).grid(row=0, column=5, padx=2)
            
            # Add voltage description
            volt_desc = ttk.Label(volt_frame, 
                                text="V0: baseline, V1: hyperpolarization, V2: depolarization",
                                font=('TkDefaultFont', 8, 'italic'))
            volt_desc.grid(row=1, column=0, columnspan=6, pady=(0, 2), sticky='w')
            
        except Exception as e:
            app_logger.error(f"Error setting up parameter controls: {str(e)}")
            raise

    def setup_analysis_controls(self):
        """Setup analysis controls and results display"""
        try:
            analysis_frame = ttk.LabelFrame(self.scrollable_frame, text="Analysis")
            analysis_frame.pack(fill='x', padx=5, pady=5)
            
            # Display options frame
            display_frame = ttk.LabelFrame(analysis_frame, text="Display Options")
            display_frame.pack(fill='x', padx=5, pady=5)
            
            # Show/hide signals
            ttk.Checkbutton(display_frame, text="Show Processed Signal",
                        variable=self.show_processed,
                        command=self.on_display_change).pack(anchor='w', padx=5)
            ttk.Checkbutton(display_frame, text="Show Red Curve",
                    variable=self.show_red_curve,
                    command=self.on_display_change).pack(anchor='w', padx=5)
            ttk.Checkbutton(display_frame, text="Show 50-point Average",
                        variable=self.show_average,
                        command=self.on_display_change).pack(anchor='w', padx=5)
            ttk.Checkbutton(display_frame, text="Show Voltage-Normalized",
                        variable=self.show_normalized,
                        command=self.on_display_change).pack(anchor='w', padx=5)
            ttk.Checkbutton(display_frame, text="Show Averaged Normalized",  # New option
                        variable=self.show_averaged_normalized,
                        command=self.on_display_change).pack(anchor='w', padx=5)
            ttk.Checkbutton(display_frame, text="Show Modified Peaks",
                        variable=self.show_modified,
                        command=self.on_display_change).pack(anchor='w', padx=5)
            
            # Display modes frame
            modes_frame = ttk.Frame(display_frame)
            modes_frame.pack(fill='x', padx=5, pady=5)
            
            # Create a grid of mode controls
            control_frames = ttk.Frame(modes_frame)
            control_frames.pack(fill='x', expand=True)
            
            # Configure grid weights for even spacing
            control_frames.grid_columnconfigure(0, weight=1)
            control_frames.grid_columnconfigure(1, weight=1)
            
            # Processed signal display mode
            proc_frame = ttk.LabelFrame(control_frames, text="Processed")
            proc_frame.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
            for mode in ["line", "points", "all_points"]:
                ttk.Radiobutton(proc_frame, text=mode.replace('_', ' ').title(),
                            variable=self.processed_display_mode,
                            value=mode,
                            command=self.on_display_change).pack(anchor='w')
            
            # Average signal display mode
            avg_frame = ttk.LabelFrame(control_frames, text="Average")
            avg_frame.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
            for mode in ["line", "points", "all_points"]:
                ttk.Radiobutton(avg_frame, text=mode.replace('_', ' ').title(),
                            variable=self.average_display_mode,
                            value=mode,
                            command=self.on_display_change).pack(anchor='w')
            
            # Normalized signal display mode
            norm_frame = ttk.LabelFrame(control_frames, text="Normalized")
            norm_frame.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
            for mode in ["line", "points", "all_points"]:
                ttk.Radiobutton(norm_frame, text=mode.replace('_', ' ').title(),
                            variable=self.normalized_display_mode,
                            value=mode,
                            command=self.on_display_change).pack(anchor='w')
            
            # Modified peaks display mode
            mod_frame = ttk.LabelFrame(control_frames, text="Modified")
            mod_frame.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
            for mode in ["line", "points", "all_points"]:
                ttk.Radiobutton(mod_frame, text=mode.replace('_', ' ').title(),
                            variable=self.modified_display_mode,
                            value=mode,
                            command=self.on_display_change).pack(anchor='w')
            
            # Averaged normalized display mode (new)
            avg_norm_frame = ttk.LabelFrame(control_frames, text="Avg Normalized")
            avg_norm_frame.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
            for mode in ["line", "points", "all_points"]:
                ttk.Radiobutton(avg_norm_frame, text=mode.replace('_', ' ').title(),
                            variable=self.averaged_normalized_display_mode,
                            value=mode,
                            command=self.on_display_change).pack(anchor='w')
            
            # Analysis button and progress bar
            self.analyze_button = ttk.Button(analysis_frame, 
                                        text="Analyze Signal",
                                        command=self.analyze_signal)
            self.analyze_button.pack(pady=5)
            
            self.progress = ttk.Progressbar(analysis_frame, 
                                        variable=self.progress_var,
                                        mode='determinate')
            self.progress.pack(fill='x', padx=5, pady=5)
            
            # Results display
            results_frame = ttk.LabelFrame(analysis_frame, text="Results")
            results_frame.pack(fill='x', padx=5, pady=5)
            
            ttk.Label(results_frame, text="Integral Value:").pack(side='left')
            ttk.Label(results_frame, textvariable=self.integral_value,
                    width=20).pack(side='left', padx=5)
            
            self.extra_info_var = tk.StringVar(value="")
            ttk.Label(results_frame, textvariable=self.extra_info_var, width=40).pack(side='left', padx=5)
            
            # Status display
            status_frame = ttk.Frame(self.scrollable_frame)
            status_frame.pack(fill='x', padx=5, pady=5)
            ttk.Label(status_frame, textvariable=self.status_text).pack(side='left')
            
        except Exception as e:
            app_logger.error(f"Error setting up analysis controls: {str(e)}")
            raise

    def debug_curve_data(self):
        """Debug method to log curve data"""
        if hasattr(self, 'modified_hyperpol') and self.modified_hyperpol is not None:
            app_logger.debug(f"Modified hyperpol curve: {len(self.modified_hyperpol)} points")
            app_logger.debug(f"Range: [{np.min(self.modified_hyperpol):.2f}, {np.max(self.modified_hyperpol):.2f}]")
        else:
            app_logger.debug("No modified hyperpol curve")

        if hasattr(self, 'modified_depol') and self.modified_depol is not None:
            app_logger.debug(f"Modified depol curve: {len(self.modified_depol)} points")
            app_logger.debug(f"Range: [{np.min(self.modified_depol):.2f}, {np.max(self.modified_depol):.2f}]")
        else:
            app_logger.debug("No modified depol curve")

    def on_display_change(self):
        """Handle changes in display options"""
        try:
            self.debug_curve_data()  # Add this line for debugging
            params = self.get_parameters()
            self.update_callback(params)
        except Exception as e:
            app_logger.error(f"Error handling display change: {str(e)}")
            raise

    def setup_normalization_points(self):
        """Setup single input field for the starting point"""
        norm_frame = ttk.LabelFrame(self.scrollable_frame, text="Normalization Point")
        norm_frame.pack(fill='x', padx=5, pady=5)

        # Single point input
        point_frame = ttk.Frame(norm_frame)
        point_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(point_frame, text="Starting Point:").pack(side='left')
        self.norm_point = tk.StringVar()
        ttk.Entry(point_frame, textvariable=self.norm_point, width=10).pack(side='left', padx=5)
        
        # Info label
        ttk.Label(norm_frame, text="Leave blank to use default value (35)", 
                font=('TkDefaultFont', 8, 'italic')).pack(pady=2)

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
    
    def create_tooltip(self, widget, text):
        """Create tooltip for a widget"""
        def enter(event):
            self.tooltip = tk.Toplevel()
            self.tooltip.wm_overrideredirect(True)
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(self.tooltip, text=text, 
                            justify='left', background="#ffffe0", 
                            relief='solid', borderwidth=1)
            label.pack()
            
        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
                
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    def validate_n_cycles(self, *args):
        """Validate number of cycles"""
        try:
            value = self.n_cycles.get()
            if value < 1:
                self.n_cycles.set(1)
                messagebox.showwarning("Validation", "Number of cycles must be at least 1")
            elif value > 10:
                self.n_cycles.set(10)
                messagebox.showwarning("Validation", "Maximum number of cycles is 10")
        except tk.TclError:
            # Invalid integer - reset to default
            self.n_cycles.set(2)

    def validate_time_constant(self, *args):
        """Validate time constants"""
        try:
            t1 = self.t1.get()
            t2 = self.t2.get()
            if t1 <= 0:
                self.t1.set(1.0)
                messagebox.showwarning("Validation", "Time constants must be positive")
            if t2 <= 0:
                self.t2.set(1.0)
                messagebox.showwarning("Validation", "Time constants must be positive")
        except tk.TclError:
            # Invalid float - reset to defaults
            self.t1.set(100.0)
            self.t2.set(100.0)

    def analyze_signal(self):
        """Perform signal analysis with current parameters"""
        try:
            # Disable button during analysis
            self.analyze_button.state(['disabled'])
            self.status_text.set("Analyzing...")
            self.progress_var.set(0)
            
            # Get parameters
            params = self.get_parameters()
            
            # Get normalization points if provided
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
            self.integral_value.set("Error")
            self.analyze_button.state(['!disabled'])
            messagebox.showerror("Analysis Error", str(e))

    def on_method_change(self, *args):
        """Handle changes in integration method"""
        try:
            # Trigger re-analysis with new method if we have already analyzed
            if self.integral_value.get() != "No analysis performed":
                self.analyze_signal()
        except Exception as e:
            app_logger.error(f"Error handling method change: {str(e)}")

    # Modify get_parameters() to include integration method
    def get_parameters(self):
        """Get current analysis parameters with validation"""
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
            
            app_logger.debug(f"Parameters validated: {params}")
            return params
            
        except Exception as e:
            app_logger.error(f"Error getting parameters: {str(e)}")
            raise

    def update_results(self, results):
        """
        Update displayed integral_value, status_text, and store the dictionary
        so it can be retrieved by get_results_dict().
        """
        try:
            if isinstance(results, dict):
                # Show main integral in the existing label
                self.integral_value.set(results.get('integral_value', 'N/A'))
                # Optionally set the status text
                self.status_text.set(results.get('status', 'Results updated'))
                
                # Collect extra lines for other keys
                extra_lines = []
                if 'purple_integral_value' in results:
                    extra_lines.append(f"Purple: {results['purple_integral_value']}")
                for key in ('capacitance_nF', 'hyperpol_area', 'depol_area'):
                    if key in results:
                        extra_lines.append(f"{key}: {results[key]}")
                
                self.extra_info_var.set(",  ".join(extra_lines))
                
                # Store a copy of results for later retrieval
                self._analysis_results = results.copy()
            else:
                self.integral_value.set("Unexpected format")
                self.extra_info_var.set("")
                self.status_text.set("Results updated")
        except Exception as e:
            app_logger.error(f"Error updating results: {str(e)}")
            self.status_text.set("Error updating results")
            self.extra_info_var.set("")
            raise

    def get_results_dict(self):
        """
        Return the last analysis results dictionary so the main GUI
        can export integrals or other data to CSV.
        """
        return self._analysis_results

    def reset(self):
        """Reset the tab to initial state"""
        try:
            # Reset parameters
            self.n_cycles.set(2)
            self.t1.set(100.0)
            self.t2.set(100.0)
            self.V0.set(-80.0)
            self.V1.set(100.0)
            self.V2.set(10.0)
            
            # Reset results
            self.integral_value.set("No analysis performed")
            self.status_text.set("Ready")
            self.progress_var.set(0)
            self.analyze_button.state(['!disabled'])
            
            # Reset display options
            self.show_processed.set(True)
            self.show_average.set(True)
            self.show_normalized.set(True)
            self.processed_display_mode.set("line")
            self.average_display_mode.set("line")
            self.normalized_display_mode.set("line")
            
            # Reset normalization points
            for key in self.norm_points:
                self.norm_points[key].set("")
            
        except Exception as e:
            app_logger.error(f"Error resetting tab: {str(e)}")
            raise

    def get_display_options(self):
        """Get current display options"""
        return {
            'show_red_curve': self.show_red_curve.get(), 
            'show_processed': self.show_processed.get(),
            'show_average': self.show_average.get(),
            'show_normalized': self.show_normalized.get(),
            'processed_mode': self.processed_display_mode.get(),
            'average_mode': self.average_display_mode.get(),
            'normalized_mode': self.normalized_display_mode.get()
        }