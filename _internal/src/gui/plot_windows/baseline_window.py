# src/gui/plot_windows/baseline_correction_window.py

import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from src.utils.logger import app_logger
from .plot_window_base import PlotWindowBase

class BaselineCorrectionWindow(PlotWindowBase):
    def __init__(self, parent, callback=None):
        super().__init__(parent, title="Baseline Correction")
        self.callback = callback
        
        # Initialize baseline parameters
        self.baseline_method = tk.StringVar(value="window")
        self.constant_value = tk.DoubleVar(value=0.0)
        self.window_start = tk.DoubleVar(value=0.0)
        self.window_end = tk.DoubleVar(value=1.0)
        self.window_size = tk.IntVar(value=100)
        self.polynomial_order = tk.IntVar(value=1)
        self.show_baseline = tk.BooleanVar(value=True)
        self.show_window = tk.BooleanVar(value=True)
        self.auto_update = tk.BooleanVar(value=True)
        
        # Initialize results
        self.baseline_data = None
        self.stats_var = tk.StringVar(value="No data")
        
        # Setup parameter traces
        self.setup_variable_traces()
        
        # Setup controls
        self.setup_baseline_controls()
        
        app_logger.debug("Baseline correction window initialized")

    def setup_variable_traces(self):
        """Setup variable traces for auto-update"""
        for var in [self.constant_value, self.window_start, self.window_end,
                   self.window_size, self.polynomial_order]:
            var.trace_add("write", self.on_parameter_change)

    def setup_baseline_controls(self):
        """Setup baseline correction specific controls"""
        # Method selection
        method_frame = ttk.LabelFrame(self.control_frame, text="Baseline Method")
        method_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Radiobutton(method_frame, text="Window Average",
                       variable=self.baseline_method, value="window",
                       command=self.update_plot).pack(pady=2)
        
        ttk.Radiobutton(method_frame, text="Constant Value",
                       variable=self.baseline_method, value="constant",
                       command=self.update_plot).pack(pady=2)
        
        ttk.Radiobutton(method_frame, text="Polynomial Fit",
                       variable=self.baseline_method, value="polynomial",
                       command=self.update_plot).pack(pady=2)
        
        # Constant value control
        const_frame = ttk.LabelFrame(self.control_frame, text="Constant Value")
        const_frame.pack(fill='x', padx=5, pady=5)
        
        value_frame = ttk.Frame(const_frame)
        value_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(value_frame, text="Value:").pack(side='left')
        ttk.Entry(value_frame, textvariable=self.constant_value,
                 width=10).pack(side='right')
        ttk.Scale(const_frame, from_=-1000, to=1000,
                 variable=self.constant_value,
                 orient='horizontal').pack(fill='x')
        
        # Window controls
        window_frame = ttk.LabelFrame(self.control_frame, text="Window Settings")
        window_frame.pack(fill='x', padx=5, pady=5)
        
        # Start time control
        start_frame = ttk.Frame(window_frame)
        start_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(start_frame, text="Start Time (s):").pack(side='left')
        ttk.Entry(start_frame, textvariable=self.window_start,
                 width=10).pack(side='right')
        ttk.Scale(window_frame, from_=0, to=1,
                 variable=self.window_start,
                 orient='horizontal').pack(fill='x')
        
        # End time control
        end_frame = ttk.Frame(window_frame)
        end_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(end_frame, text="End Time (s):").pack(side='left')
        ttk.Entry(end_frame, textvariable=self.window_end,
                 width=10).pack(side='right')
        ttk.Scale(window_frame, from_=0, to=1,
                 variable=self.window_end,
                 orient='horizontal').pack(fill='x')
        
        # Window size for moving average
        size_frame = ttk.Frame(window_frame)
        size_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(size_frame, text="Window Size:").pack(side='left')
        ttk.Entry(size_frame, textvariable=self.window_size,
                 width=10).pack(side='right')
        ttk.Scale(window_frame, from_=1, to=1000,
                 variable=self.window_size,
                 orient='horizontal').pack(fill='x')
        
        # Polynomial controls
        poly_frame = ttk.LabelFrame(self.control_frame, text="Polynomial Settings")
        poly_frame.pack(fill='x', padx=5, pady=5)
        
        order_frame = ttk.Frame(poly_frame)
        order_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(order_frame, text="Order:").pack(side='left')
        ttk.Entry(order_frame, textvariable=self.polynomial_order,
                 width=5).pack(side='right')
        ttk.Scale(poly_frame, from_=1, to=10,
                 variable=self.polynomial_order,
                 orient='horizontal').pack(fill='x')
        
        # Display options
        display_frame = ttk.LabelFrame(self.control_frame, text="Display Options")
        display_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(display_frame, text="Show Baseline",
                       variable=self.show_baseline,
                       command=self.update_plot).pack(pady=2)
        
        ttk.Checkbutton(display_frame, text="Show Window",
                       variable=self.show_window,
                       command=self.update_plot).pack(pady=2)
        
        ttk.Checkbutton(display_frame, text="Auto Update",
                       variable=self.auto_update).pack(pady=2)
        
        # Statistics display
        stats_frame = ttk.LabelFrame(self.control_frame, text="Statistics")
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(stats_frame, textvariable=self.stats_var,
                 wraplength=200, justify='left').pack(pady=5)
        
        # Control buttons
        ttk.Button(self.control_frame, text="Update",
                  command=self.update_plot).pack(pady=2)
        ttk.Button(self.control_frame, text="Reset",
                  command=self.reset_baseline).pack(pady=2)

    def set_data(self, time_data, data):
        """Override set_data to update slider ranges"""
        super().set_data(time_data, data)
        
        if time_data is not None:
            t_min = np.min(time_data)
            t_max = np.max(time_data)
            
            # Update time range sliders
            for widget in self.control_frame.winfo_children():
                if isinstance(widget, ttk.Scale):
                    if widget.cget('variable') in ['window_start', 'window_end']:
                        widget.configure(from_=t_min, to=t_max)
            
            self.window_start.set(t_min)
            self.window_end.set(t_max)

    def calculate_baseline(self):
        """Calculate baseline based on selected method"""
        try:
            if self.baseline_method.get() == "constant":
                return np.full_like(self.data, self.constant_value.get())
            
            elif self.baseline_method.get() == "window":
                start_idx = np.searchsorted(self.time_data, self.window_start.get())
                end_idx = np.searchsorted(self.time_data, self.window_end.get())
                window = self.data[start_idx:end_idx]
                
                # Calculate moving average
                window_size = min(self.window_size.get(), len(window))
                kernel = np.ones(window_size) / window_size
                baseline = np.convolve(window, kernel, mode='same')
                
                return np.mean(baseline)
            
            else:  # polynomial fit
                # Get window indices
                start_idx = np.searchsorted(self.time_data, self.window_start.get())
                end_idx = np.searchsorted(self.time_data, self.window_end.get())
                
                # Fit polynomial to window
                t = self.time_data[start_idx:end_idx]
                y = self.data[start_idx:end_idx]
                coeffs = np.polyfit(t, y, self.polynomial_order.get())
                
                # Generate baseline
                baseline = np.polyval(coeffs, self.time_data)
                return baseline
            
        except Exception as e:
            app_logger.error(f"Error calculating baseline: {str(e)}")
            raise

    def update_statistics(self):
        """Update statistics display"""
        if self.processed_data is None:
            return
            
        try:
            stats = {
                'mean': np.mean(self.processed_data),
                'std': np.std(self.processed_data),
                'baseline': np.mean(self.baseline_data) if self.baseline_data is not None else 0,
                'offset': np.mean(self.processed_data - self.data)
            }
            
            stats_text = (
                f"Statistics:\n"
                f"Mean: {stats['mean']:.2f}\n"
                f"Std Dev: {stats['std']:.2f}\n"
                f"Baseline: {stats['baseline']:.2f}\n"
                f"Offset: {stats['offset']:.2f}"
            )
            
            self.stats_var.set(stats_text)
            
        except Exception as e:
            app_logger.error(f"Error updating statistics: {str(e)}")
            self.stats_var.set(f"Error: {str(e)}")

    def on_parameter_change(self, *args):
        """Handle parameter changes"""
        if self.auto_update.get():
            self.update_plot()

    def update_plot(self):
        """Update plot with baseline correction"""
        if self.data is None:
            return
            
        try:
            # Calculate baseline
            self.baseline_data = self.calculate_baseline()
            
            # Apply baseline correction
            if isinstance(self.baseline_data, np.ndarray):
                self.processed_data = self.data - self.baseline_data
            else:
                self.processed_data = self.data - self.baseline_data
            
            # Update statistics
            self.update_statistics()
            
            # Clear previous plot
            self.ax.clear()
            
            # Plot original data
            self.ax.plot(self.time_data, self.data, 'b-',
                        label='Original', alpha=0.5)
            
            # Plot baseline-corrected data
            self.ax.plot(self.time_data, self.processed_data, 'r-',
                        label='Corrected', linewidth=1.5)
            
            # Plot baseline if enabled
            if self.show_baseline.get():
                if isinstance(self.baseline_data, np.ndarray):
                    self.ax.plot(self.time_data, self.baseline_data, 'g--',
                               label='Baseline', alpha=0.7)
                else:
                    self.ax.axhline(y=self.baseline_data, color='g',
                                  linestyle='--', label='Baseline')
            
            # Show window region if enabled
            if self.show_window.get() and self.baseline_method.get() != "constant":
                start = self.window_start.get()
                end = self.window_end.get()
                self.ax.axvspan(start, end, alpha=0.2, color='yellow',
                              label='Window')
            
            # Set labels and grid
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Signal")
            self.ax.legend()
            self.ax.grid(True)
            
            # Update plot
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Store view state
            self.store_current_view()
            
            app_logger.debug("Baseline correction plot updated")
            
        except Exception as e:
            app_logger.error(f"Error updating baseline plot: {str(e)}")
            raise

    def reset_baseline(self):
        """Reset baseline parameters to defaults"""
        self.baseline_method.set("window")
        self.constant_value.set(0.0)
        self.window_start.set(self.time_data[0] if self.time_data is not None else 0.0)
        self.window_end.set(self.time_data[-1] if self.time_data is not None else 1.0)
        self.window_size.set(100)
        self.polynomial_order.set(1)
        self.show_baseline.set(True)
        self.show_window.set(True)
        self.update_plot()

    def on_accept(self):
        """Handle accept button click"""
        if self.callback:
            result = {
                'processed_data': self.processed_data,
                'baseline_data': self.baseline_data,
                'method': self.baseline_method.get(),
                'parameters': {
                    'constant_value': self.constant_value.get(),
                    'window_start': self.window_start.get(),
                    'window_end': self.window_end.get(),
                    'window_size': self.window_size.get(),
                    'polynomial_order': self.polynomial_order.get()
                }
            }
            self.callback(result)
        self.destroy()