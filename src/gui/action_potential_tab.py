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
        self.setup_analysis_controls()
        
        app_logger.debug("Action potential analysis tab initialized")

    def init_variables(self):
        """Initialize control variables with validation"""
        try:
            # Analysis parameters with validation
            self.n_cycles = tk.IntVar(value=2)
            self.t1 = tk.DoubleVar(value=100.0)
            self.t2 = tk.DoubleVar(value=100.0)
            self.V0 = tk.DoubleVar(value=-80.0)
            self.V1 = tk.DoubleVar(value=100.0)
            self.V2 = tk.DoubleVar(value=10.0)
            
            # Results display
            self.integral_value = tk.StringVar(value="No analysis performed")
            self.status_text = tk.StringVar(value="Ready")
            
            # Add validation traces
            self.n_cycles.trace_add("write", self.validate_n_cycles)
            self.t1.trace_add("write", self.validate_time_constant)
            self.t2.trace_add("write", self.validate_time_constant)
            
            app_logger.debug("Variables initialized successfully")
            
        except Exception as e:
            app_logger.error(f"Error initializing variables: {str(e)}")
            raise

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
            self.create_tooltip(time_frame, "Time constants for repolarization (t1) and depolarization (t2)")
            self.create_tooltip(volt_frame, "Voltage levels for baseline (V0), repolarization (V1), and depolarization (V2)")
            
        except Exception as e:
            app_logger.error(f"Error setting up parameter controls: {str(e)}")
            raise

    def setup_analysis_controls(self):
        """Setup analysis controls and results display"""
        try:
            analysis_frame = ttk.LabelFrame(self.frame, text="Analysis")
            analysis_frame.pack(fill='x', padx=5, pady=5)
            
            # Analysis button with progress indication
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
            
            # Integral value display
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

    def get_parameters(self):
        """Get current analysis parameters with validation"""
        try:
            params = {
                'n_cycles': self.n_cycles.get(),
                't1': self.t1.get(),
                't2': self.t2.get(),
                'V0': self.V0.get(),
                'V1': self.V1.get(),
                'V2': self.V2.get()
            }
            
            # Validate parameters
            if params['n_cycles'] < 1:
                raise ValueError("Number of cycles must be positive")
            if params['t1'] <= 0 or params['t2'] <= 0:
                raise ValueError("Time constants must be positive")
            
            app_logger.debug(f"Parameters validated: {params}")
            return params
            
        except Exception as e:
            app_logger.error(f"Error getting parameters: {str(e)}")
            raise

    def analyze_signal(self):
        """Perform signal analysis with current parameters"""
        try:
            # Disable button during analysis
            self.analyze_button.state(['disabled'])
            self.status_text.set("Analyzing...")
            self.progress_var.set(0)
            
            # Get parameters
            params = self.get_parameters()
            
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

    def update_results(self, integral_value):
        """Update displayed results"""
        try:
            self.integral_value.set(f"{integral_value:.6f} µC/cm²")
            self.status_text.set("Results updated")
            
        except Exception as e:
            app_logger.error(f"Error updating results: {str(e)}")
            self.status_text.set("Error updating results")
            raise

    def reset(self):
        """Reset the tab to initial state"""
        try:
            self.n_cycles.set(2)
            self.t1.set(100.0)
            self.t2.set(100.0)
            self.V0.set(-80.0)
            self.V1.set(100.0)
            self.V2.set(10.0)
            self.integral_value.set("No analysis performed")
            self.status_text.set("Ready")
            self.progress_var.set(0)
            self.analyze_button.state(['!disabled'])
            
        except Exception as e:
            app_logger.error(f"Error resetting tab: {str(e)}")
            raise