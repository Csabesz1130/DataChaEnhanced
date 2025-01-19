import tkinter as tk
from tkinter import ttk, messagebox
from src.utils.logger import app_logger

class ActionPotentialTab:
    def __init__(self, parent, callback):
        """
        Initialize the action potential analysis tab.
        
        Args:
            parent: Parent widget
            callback: Function to call when analysis settings change
        """
        self.parent = parent
        self.update_callback = callback
        
        # Create main frame
        self.frame = ttk.LabelFrame(parent, text="Action Potential Analysis")
        self.frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize variables
        self.init_variables()
        
        # Create controls
        self.setup_parameter_controls()
        self.setup_normalization_points()
        self.setup_analysis_controls()
        
        app_logger.debug("Action potential analysis tab initialized")

    def init_variables(self):
        """Initialize control variables with validation"""
        try:
            # Analysis parameters with validation
            self.n_cycles = tk.IntVar(value=2)
            self.t0 = tk.DoubleVar(value=20.0)  # Add t0
            self.t1 = tk.DoubleVar(value=100.0)
            self.t2 = tk.DoubleVar(value=100.0)
            self.V0 = tk.DoubleVar(value=-80.0)
            self.V1 = tk.DoubleVar(value=100.0)
            self.V2 = tk.DoubleVar(value=10.0)
            
            # Results display
            self.integral_value = tk.StringVar(value="No analysis performed")
            self.status_text = tk.StringVar(value="Ready")
            
            # Display mode variables
            self.show_processed = tk.BooleanVar(value=True)
            self.show_average = tk.BooleanVar(value=True)
            self.show_normalized = tk.BooleanVar(value=True)
            self.processed_display_mode = tk.StringVar(value="line")
            self.average_display_mode = tk.StringVar(value="line")
            self.normalized_display_mode = tk.StringVar(value="line")
            
            # Add validation traces
            self.n_cycles.trace_add("write", self.validate_n_cycles)
            self.t0.trace_add("write", self.validate_time_constant)  # Add t0 validation
            self.t1.trace_add("write", self.validate_time_constant)
            self.t2.trace_add("write", self.validate_time_constant)
            
            app_logger.debug("Variables initialized successfully")
            
        except Exception as e:
            app_logger.error(f"Error initializing variables: {str(e)}")
            raise

    def setup_normalization_points(self):
        """Setup input fields for normalization segment points"""
        norm_frame = ttk.LabelFrame(self.frame, text="Normalization Points (Optional)")
        norm_frame.pack(fill='x', padx=5, pady=5)

        # Create a frame for segment inputs with grid layout
        segments_frame = ttk.Frame(norm_frame)
        segments_frame.pack(fill='x', padx=5, pady=5)

        # Initialize variables for storing point values
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

        # Column headers
        ttk.Label(segments_frame, text="Segment").grid(row=0, column=0, padx=5)
        ttk.Label(segments_frame, text="Start").grid(row=0, column=1, padx=5)
        ttk.Label(segments_frame, text="End").grid(row=0, column=2, padx=5)
        ttk.Label(segments_frame, text="Type").grid(row=0, column=3, padx=5)

        # Add segment rows
        segments = [
            ("1", "Hyperpol"),
            ("2", "Depol"),
            ("3", "Hyperpol"),
            ("4", "Depol")
        ]

        for i, (seg_num, seg_type) in enumerate(segments, 1):
            ttk.Label(segments_frame, text=f"{seg_num}").grid(row=i, column=0, padx=5, pady=2)
            
            # Start point entry
            start_entry = ttk.Entry(segments_frame, width=8, 
                                  textvariable=self.norm_points[f'seg{seg_num}_start'])
            start_entry.grid(row=i, column=1, padx=2, pady=2)
            
            # End point entry
            end_entry = ttk.Entry(segments_frame, width=8,
                                textvariable=self.norm_points[f'seg{seg_num}_end'])
            end_entry.grid(row=i, column=2, padx=2, pady=2)
            
            # Type label
            ttk.Label(segments_frame, text=seg_type).grid(row=i, column=3, padx=5, pady=2)

        # Add info label
        ttk.Label(norm_frame, text="Leave blank to use default values",
                 font=('TkDefaultFont', 8, 'italic')).pack(pady=2)

    def setup_parameter_controls(self):
        """Setup parameter input controls"""
        try:
            param_frame = ttk.LabelFrame(self.frame, text="Parameters")
            param_frame.pack(fill='x', padx=5, pady=5)
            
            # Number of cycles
            cycle_frame = ttk.Frame(param_frame)
            cycle_frame.pack(fill='x', padx=5, pady=2)
            ttk.Label(cycle_frame, text="Number of Cycles:").pack(side='left')
            ttk.Entry(cycle_frame, textvariable=self.n_cycles,
                    width=10).pack(side='right')
            
            # Time constants
            time_frame = ttk.Frame(param_frame)
            time_frame.pack(fill='x', padx=5, pady=2)
            ttk.Label(time_frame, text="t0 (ms):").pack(side='left')  # Add t0
            ttk.Entry(time_frame, textvariable=self.t0,
                    width=10).pack(side='left', padx=5)
            ttk.Label(time_frame, text="t1 (ms):").pack(side='left')
            ttk.Entry(time_frame, textvariable=self.t1,
                    width=10).pack(side='left', padx=5)
            ttk.Label(time_frame, text="t2 (ms):").pack(side='left')
            ttk.Entry(time_frame, textvariable=self.t2,
                    width=10).pack(side='right')
            
            # Voltage levels
            volt_frame = ttk.Frame(param_frame)
            volt_frame.pack(fill='x', padx=5, pady=2)
            ttk.Label(volt_frame, text="V0 (mV):").pack(side='left')
            ttk.Entry(volt_frame, textvariable=self.V0,
                    width=10).pack(side='left', padx=5)
            ttk.Label(volt_frame, text="V1 (mV):").pack(side='left')
            ttk.Entry(volt_frame, textvariable=self.V1,
                    width=10).pack(side='left', padx=5)
            ttk.Label(volt_frame, text="V2 (mV):").pack(side='left')
            ttk.Entry(volt_frame, textvariable=self.V2,
                    width=10).pack(side='right')
            
            # Add tooltips
            self.create_tooltip(cycle_frame, "Number of action potential cycles to analyze")
            self.create_tooltip(time_frame, "Time constants for baseline (t0), repolarization (t1) and depolarization (t2)")
            self.create_tooltip(volt_frame, "Voltage levels for baseline (V0), repolarization (V1), and depolarization (V2)")
            
        except Exception as e:
            app_logger.error(f"Error setting up parameter controls: {str(e)}")
            raise

    def setup_analysis_controls(self):
        """Setup analysis controls and results display"""
        try:
            analysis_frame = ttk.LabelFrame(self.frame, text="Analysis")
            analysis_frame.pack(fill='x', padx=5, pady=5)
            
            # Display options frame
            display_frame = ttk.LabelFrame(analysis_frame, text="Display Options")
            display_frame.pack(fill='x', padx=5, pady=5)
            
            # Show/hide signals with update callbacks
            ttk.Checkbutton(display_frame, text="Show Processed Signal",
                        variable=self.show_processed,
                        command=self.on_display_change).pack(anchor='w', padx=5)
            ttk.Checkbutton(display_frame, text="Show 50-point Average",
                        variable=self.show_average,
                        command=self.on_display_change).pack(anchor='w', padx=5)
            ttk.Checkbutton(display_frame, text="Show Voltage-Normalized",
                        variable=self.show_normalized,
                        command=self.on_display_change).pack(anchor='w', padx=5)
            
            # Display modes frame
            modes_frame = ttk.Frame(display_frame)
            modes_frame.pack(fill='x', padx=5, pady=5)
            
            # Processed signal display mode
            proc_frame = ttk.LabelFrame(modes_frame, text="Processed")
            proc_frame.pack(side='left', fill='x', expand=True, padx=2)
            for mode in ["line", "points", "all_points"]:
                ttk.Radiobutton(proc_frame, text=mode.replace('_', ' ').title(),
                            variable=self.processed_display_mode,
                            value=mode,
                            command=self.on_display_change).pack(anchor='w')
            
            # Average signal display mode
            avg_frame = ttk.LabelFrame(modes_frame, text="Average")
            avg_frame.pack(side='left', fill='x', expand=True, padx=2)
            for mode in ["line", "points", "all_points"]:
                ttk.Radiobutton(avg_frame, text=mode.replace('_', ' ').title(),
                            variable=self.average_display_mode,
                            value=mode,
                            command=self.on_display_change).pack(anchor='w')
            
            # Normalized signal display mode
            norm_frame = ttk.LabelFrame(modes_frame, text="Normalized")
            norm_frame.pack(side='left', fill='x', expand=True, padx=2)
            for mode in ["line", "points", "all_points"]:
                ttk.Radiobutton(norm_frame, text=mode.replace('_', ' ').title(),
                            variable=self.normalized_display_mode,
                            value=mode,
                            command=self.on_display_change).pack(anchor='w')
            
            # Analysis controls
            self.analyze_button = ttk.Button(analysis_frame, 
                                        text="Analyze Signal",
                                        command=self.analyze_signal)
            self.analyze_button.pack(pady=5)
            
            # Progress bar
            self.progress_var = tk.DoubleVar()
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
            
            # Status display
            status_frame = ttk.Frame(self.frame)
            status_frame.pack(fill='x', padx=5, pady=5)
            ttk.Label(status_frame, textvariable=self.status_text).pack(side='left')
            
        except Exception as e:
            app_logger.error(f"Error setting up analysis controls: {str(e)}")
            raise

    def on_display_change(self):
        """Handle changes in display options"""
        try:
            # Get current display settings
            params = self.get_parameters()
            # Call callback to update display
            self.update_callback(params)
        except Exception as e:
            app_logger.error(f"Error handling display change: {str(e)}")
            raise

    def get_normalization_points(self):
        """
        Get normalization points if all fields are filled in correctly.
        Returns None if any field is empty or invalid.
        """
        try:
            # Check if all fields are filled
            all_points = []
            for seg_num in range(1, 5):
                start = self.norm_points[f'seg{seg_num}_start'].get().strip()
                end = self.norm_points[f'seg{seg_num}_end'].get().strip()
                
                # If any field is empty, return None
                if not start or not end:
                    return None
                    
                try:
                    start_val = int(start)
                    end_val = int(end)
                    all_points.extend([start_val, end_val])
                except ValueError:
                    return None

            # Return points only if all values are valid
            return {
                'seg1': (all_points[0], all_points[1]),
                'seg2': (all_points[2], all_points[3]),
                'seg3': (all_points[4], all_points[5]),
                'seg4': (all_points[6], all_points[7])
            }

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

    def get_parameters(self):
        """Get current analysis parameters with validation"""
        try:
            params = {
                'n_cycles': self.n_cycles.get(),
                't0': self.t0.get(),  # Add t0
                't1': self.t1.get(),
                't2': self.t2.get(),
                'V0': self.V0.get(),
                'V1': self.V1.get(),
                'V2': self.V2.get(),
                'display_options': {
                    'show_processed': self.show_processed.get(),
                    'show_average': self.show_average.get(),
                    'show_normalized': self.show_normalized.get(),
                    'processed_mode': self.processed_display_mode.get(),
                    'average_mode': self.average_display_mode.get(),
                    'normalized_mode': self.normalized_display_mode.get()
                }
            }
            
            # Validate parameters
            if params['n_cycles'] < 1:
                raise ValueError("Number of cycles must be positive")
            if params['t0'] <= 0 or params['t1'] <= 0 or params['t2'] <= 0:  # Add t0 validation
                raise ValueError("Time constants must be positive")
            
            app_logger.debug(f"Parameters validated: {params}")
            return params
            
        except Exception as e:
            app_logger.error(f"Error getting parameters: {str(e)}")
            raise

    def update_results(self, results):
        """Update displayed results"""
        try:
            if isinstance(results, dict):
                self.integral_value.set(results.get('integral_value', 'N/A'))
                self.status_text.set(results.get('status', 'Results updated'))
            else:
                self.integral_value.set(f"{results:.6f} µC/cm²")
                self.status_text.set("Results updated")
            
        except Exception as e:
            app_logger.error(f"Error updating results: {str(e)}")
            self.status_text.set("Error updating results")
            raise

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
            'show_processed': self.show_processed.get(),
            'show_average': self.show_average.get(),
            'show_normalized': self.show_normalized.get(),
            'processed_mode': self.processed_display_mode.get(),
            'average_mode': self.average_display_mode.get(),
            'normalized_mode': self.normalized_display_mode.get()
        }