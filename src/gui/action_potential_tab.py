import tkinter as tk
from tkinter import ttk, messagebox
from src.utils.logger import app_logger

class ActionPotentialTab:
    def __init__(self, parent, callback):
        """
        A Tkinter tab for multi-step patch clamp analysis.
        Args:
            parent: parent widget (e.g. a Notebook)
            callback: function(params) => results
        """
        self.parent = parent
        self.update_callback = callback
        
        # Main container
        self.frame = ttk.LabelFrame(parent, text="Action Potential Analysis")
        self.frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.init_variables()
        self.setup_parameter_controls()
        self.setup_analysis_controls()
        self.setup_visibility_controls()
        
        app_logger.debug("Action potential analysis tab initialized")

    def init_variables(self):
        """Initialize tkinter variables with defaults."""
        try:
            # Analysis parameters
            self.n_cycles = tk.IntVar(value=2)
            self.t0 = tk.DoubleVar(value=20.0)
            self.t1 = tk.DoubleVar(value=100.0)
            self.t2 = tk.DoubleVar(value=100.0)
            self.t3 = tk.DoubleVar(value=1000.0)

            self.V0 = tk.DoubleVar(value=-80.0)
            self.V1 = tk.DoubleVar(value=-100.0)
            self.V2 = tk.DoubleVar(value=10.0)

            # Visibility controls
            self.show_processed = tk.BooleanVar(value=True)
            self.show_average = tk.BooleanVar(value=True)

            # Display variables
            self.integral_value = tk.StringVar(value="No analysis performed")
            self.status_text = tk.StringVar(value="Ready")

            # validation
            self.n_cycles.trace_add("write", self.validate_n_cycles)
            for var in [self.t0, self.t1, self.t2, self.t3]:
                var.trace_add("write", self.validate_time_constant)
            
            app_logger.debug("Variables initialized successfully.")
        except Exception as e:
            app_logger.error(f"Error initializing variables: {str(e)}")
            raise

    def validate_n_cycles(self, *args):
        """Ensure 1 <= n_cycles <= 10."""
        try:
            val = self.n_cycles.get()
            if val < 1:
                self.n_cycles.set(1)
                messagebox.showwarning("Validation", "Number of cycles must be >= 1.")
            elif val > 10:
                self.n_cycles.set(10)
                messagebox.showwarning("Validation", "Maximum cycles is 10.")
        except tk.TclError:
            self.n_cycles.set(2)

    def validate_time_constant(self, *args):
        """Ensure t0, t1, t2, t3 > 0."""
        try:
            for var in [self.t0, self.t1, self.t2, self.t3]:
                if var.get() <= 0:
                    var.set(1.0)
                    messagebox.showwarning("Validation", "Time constants must be > 0.")
        except tk.TclError:
            self.t0.set(20.0)
            self.t1.set(100.0)
            self.t2.set(100.0)
            self.t3.set(1000.0)

    def setup_visibility_controls(self):
        """Create visibility toggle controls for curves."""
        try:
            visibility_frame = ttk.LabelFrame(self.frame, text="Show/Hide Curves")
            visibility_frame.pack(fill='x', padx=5, pady=5)
            
            ttk.Checkbutton(visibility_frame, 
                           text="Show Processed Signal (Green)",
                           variable=self.show_processed,
                           command=self.on_visibility_change).pack(pady=2)
            
            ttk.Checkbutton(visibility_frame, 
                           text="Show 50-point Average (Orange)",
                           variable=self.show_average,
                           command=self.on_visibility_change).pack(pady=2)
            
        except Exception as e:
            app_logger.error(f"Error setting up visibility controls: {str(e)}")
            raise

    def setup_parameter_controls(self):
        """Create UI for the time intervals and voltages."""
        try:
            param_frame = ttk.LabelFrame(self.frame, text="Parameters")
            param_frame.pack(fill='x', padx=5, pady=5)
            
            # cycles
            cyc_frame = ttk.Frame(param_frame)
            cyc_frame.pack(fill='x', padx=5, pady=2)
            ttk.Label(cyc_frame, text="Number of Cycles:").pack(side='left')
            ttk.Entry(cyc_frame, textvariable=self.n_cycles, width=4).pack(side='left', padx=5)

            # times
            time_frame = ttk.Frame(param_frame)
            time_frame.pack(fill='x', padx=5, pady=2)

            ttk.Label(time_frame, text="t0 (ms):").pack(side='left')
            ttk.Entry(time_frame, textvariable=self.t0, width=6).pack(side='left', padx=2)
            
            ttk.Label(time_frame, text="t1 (ms):").pack(side='left')
            ttk.Entry(time_frame, textvariable=self.t1, width=6).pack(side='left', padx=2)

            ttk.Label(time_frame, text="t2 (ms):").pack(side='left')
            ttk.Entry(time_frame, textvariable=self.t2, width=6).pack(side='left', padx=2)

            ttk.Label(time_frame, text="t3 (ms):").pack(side='left')
            ttk.Entry(time_frame, textvariable=self.t3, width=6).pack(side='left', padx=2)

            # voltages
            volt_frame = ttk.Frame(param_frame)
            volt_frame.pack(fill='x', padx=5, pady=2)

            ttk.Label(volt_frame, text="V0 (mV):").pack(side='left')
            ttk.Entry(volt_frame, textvariable=self.V0, width=6).pack(side='left', padx=5)

            ttk.Label(volt_frame, text="V1 (mV):").pack(side='left')
            ttk.Entry(volt_frame, textvariable=self.V1, width=6).pack(side='left', padx=5)

            ttk.Label(volt_frame, text="V2 (mV):").pack(side='left')
            ttk.Entry(volt_frame, textvariable=self.V2, width=6).pack(side='left', padx=5)
            
        except Exception as e:
            app_logger.error(f"Error setting up parameter controls: {str(e)}")
            raise

    def setup_analysis_controls(self):
        """Create button for 'Analyze Signal' plus a progress bar, plus results display."""
        try:
            analysis_frame = ttk.LabelFrame(self.frame, text="Analysis")
            analysis_frame.pack(fill='x', padx=5, pady=5)
            
            # Analysis button
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

            ttk.Label(results_frame, text="Integral/Cap.:").pack(side='left')
            ttk.Label(results_frame, textvariable=self.integral_value, 
                     width=30).pack(side='left', padx=5)

            # Status display
            status_frame = ttk.Frame(self.frame)
            status_frame.pack(fill='x', padx=5, pady=5)
            ttk.Label(status_frame, textvariable=self.status_text).pack(side='left')
            
        except Exception as e:
            app_logger.error(f"Error setting up analysis controls: {str(e)}")
            raise

    def on_visibility_change(self):
        """Handle visibility toggle changes."""
        try:
            # Create visibility parameters dictionary
            params = {
                'visibility_update': True,  # Flag to indicate this is a visibility update
                'show_processed': self.show_processed.get(),
                'show_average': self.show_average.get()
            }
            
            # Call the callback with visibility settings
            self.update_callback(params)
            
        except Exception as e:
            app_logger.error(f"Error updating visibility: {str(e)}")
            raise

    def analyze_signal(self):
        """Collect user parameters, pass to callback, show results or errors."""
        try:
            self.analyze_button.state(['disabled'])
            self.status_text.set("Analyzing...")
            self.progress_var.set(0)
            
            # gather parameters
            params = self.get_parameters()
            self.progress_var.set(50)
            
            # call processing
            results = self.update_callback(params)
            
            # display results
            if results:
                self.update_results(results)
                self.status_text.set("Analysis complete")
                # Reset visibility controls to show both curves
                self.show_processed.set(True)
                self.show_average.set(True)
            else:
                self.integral_value.set("No analysis results")
                self.status_text.set("No results returned")
            
            self.progress_var.set(100)
            self.analyze_button.state(['!disabled'])
            
        except Exception as e:
            app_logger.error(f"Error analyzing signal: {str(e)}")
            self.status_text.set("Analysis failed")
            self.integral_value.set("Error in analysis")
            self.analyze_button.state(['!disabled'])
            messagebox.showerror("Analysis Error", str(e))

    def get_parameters(self):
        """Assemble user entries into a dictionary."""
        try:
            params = {
                'n_cycles': self.n_cycles.get(),
                't0': self.t0.get(),
                't1': self.t1.get(),
                't2': self.t2.get(),
                't3': self.t3.get(),
                'V0': self.V0.get(),
                'V1': self.V1.get(),
                'V2': self.V2.get(),
                'cell_area_cm2': 1e-4
            }
            if params['n_cycles'] < 1:
                raise ValueError("n_cycles must be >= 1.")
            for seg in ['t0','t1','t2','t3']:
                if params[seg] <= 0:
                    raise ValueError(f"{seg} must be > 0.")
            return params
        except Exception as e:
            messagebox.showerror("Parameter Error", str(e))
            raise

    def update_results(self, results):
        """Update the integral_value text with the final results from the processor."""
        if not results:
            self.integral_value.set("No analysis results.")
            return
        
        lines = []
        lines.append("Analysis Results:")
        if 'integral_value' in results:
            lines.append(f"Charge: {results['integral_value']}")
        if 'capacitance_uF_cm2' in results:
            lines.append(f"Cap.:   {results['capacitance_uF_cm2']}")
        # raw
        if 'raw_values' in results:
            rv = results['raw_values']
            lines.append(f"Charge(C)= {rv['charge_C']:.2e}")
            lines.append(f"Cap(F)=    {rv['capacitance_F']:.2e}")
            lines.append(f"Area=      {rv['area_cm2']:.2e} cm²")
        
        self.integral_value.set("\n".join(lines))

    def reset(self):
        """Reset to defaults."""
        self.n_cycles.set(2)
        self.t0.set(20.0)
        self.t1.set(100.0)
        self.t2.set(100.0)
        self.t3.set(1000.0)
        self.V0.set(-80.0)
        self.V1.set(-100.0)
        self.V2.set(10.0)
        self.integral_value.set("No analysis performed")
        self.status_text.set("Ready")
        self.progress_var.set(0)
        self.analyze_button.state(['!disabled'])
