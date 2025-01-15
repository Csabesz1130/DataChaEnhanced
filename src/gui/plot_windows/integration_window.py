import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.integrate import simps, trapz
from src.utils.logger import app_logger
from .plot_window_base import PlotWindowBase

class IntegrationWindow(PlotWindowBase):
    def __init__(self, parent, callback=None):
        super().__init__(parent, title="Signal Integration")
        self.callback = callback
        
        # Initialize integration parameters
        self.integration_method = tk.StringVar(value="simps")
        self.use_range = tk.BooleanVar(value=False)
        self.range_start = tk.DoubleVar(value=0.0)
        self.range_end = tk.DoubleVar(value=1.0)
        self.show_filled = tk.BooleanVar(value=True)
        
        # Integration results
        self.integral_value = tk.StringVar(value="Not calculated")
        self.integral_data = None
        
        # Setup specific controls
        self.setup_integration_controls()
        
        app_logger.debug("Integration window initialized")

    def setup_integration_controls(self):
        """Setup integration specific controls"""
        # Method selection
        method_frame = ttk.LabelFrame(self.control_frame, text="Integration Method")
        method_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Radiobutton(method_frame, text="Simpson's Rule",
                       variable=self.integration_method, value="simps",
                       command=self.update_plot).pack(pady=2)
        
        ttk.Radiobutton(method_frame, text="Trapezoidal Rule",
                       variable=self.integration_method, value="trapz",
                       command=self.update_plot).pack(pady=2)
        
        # Range selection
        range_frame = ttk.LabelFrame(self.control_frame, text="Integration Range")
        range_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(range_frame, text="Use Custom Range",
                       variable=self.use_range,
                       command=self.update_plot).pack(pady=2)
        
        ttk.Label(range_frame, text="Start Time (s):").pack()
        ttk.Entry(range_frame, textvariable=self.range_start).pack(pady=2)
        ttk.Scale(range_frame, from_=0, to=1,
                 variable=self.range_start, orient='horizontal',
                 command=lambda *args: self.update_plot()).pack(fill='x', pady=2)
        
        ttk.Label(range_frame, text="End Time (s):").pack()
        ttk.Entry(range_frame, textvariable=self.range_end).pack(pady=2)
        ttk.Scale(range_frame, from_=0, to=1,
                 variable=self.range_end, orient='horizontal',
                 command=lambda *args: self.update_plot()).pack(fill='x', pady=2)
        
        # Display options
        display_frame = ttk.LabelFrame(self.control_frame, text="Display Options")
        display_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(display_frame, text="Show Filled Area",
                       variable=self.show_filled,
                       command=self.update_plot).pack(pady=2)
        
        # Results display
        results_frame = ttk.LabelFrame(self.control_frame, text="Integration Results")
        results_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(results_frame, textvariable=self.integral_value,
                 wraplength=200, justify='left').pack(pady=5)
        
        # Add calculate button
        ttk.Button(self.control_frame, text="Calculate",
                  command=self.calculate_integral).pack(pady=5)

    def set_data(self, time_data, data):
        """Override set_data to update range scales"""
        super().set_data(time_data, data)
        
        # Update range scales with actual time limits
        t_min = np.min(time_data)
        t_max = np.max(time_data)
        
        for scale in self.control_frame.findall('ttk.Scale'):
            scale.configure(from_=t_min, to=t_max)
        
        self.range_start.set(t_min)
        self.range_end.set(t_max)

    def calculate_integral(self):
        """Calculate the integral of the signal"""
        try:
            if self.processed_data is None:
                return
            
            # Get data range
            if self.use_range.get():
                start_idx = np.searchsorted(self.time_data, self.range_start.get())
                end_idx = np.searchsorted(self.time_data, self.range_end.get())
                t = self.time_data[start_idx:end_idx]
                y = self.processed_data[start_idx:end_idx]
            else:
                t = self.time_data
                y = self.processed_data
            
            # Calculate integral
            if self.integration_method.get() == "simps":
                integral = simps(y, t)
            else:  # trapz
                integral = trapz(y, t)
            
            # Calculate cumulative integral
            if self.integration_method.get() == "simps":
                self.integral_data = np.array([simps(y[:i+1], t[:i+1]) 
                                             for i in range(len(t))])
            else:
                self.integral_data = np.array([trapz(y[:i+1], t[:i+1])
                                             for i in range(len(t))])
            
            # Update display
            self.integral_value.set(
                f"Integral Value: {integral:.6f}\n"
                f"Method: {self.integration_method.get()}\n"
                f"Range: {t[0]:.3f} to {t[-1]:.3f} s"
            )
            
            self.update_plot()
            
            app_logger.debug(f"Integral calculated: {integral:.6f}")
            return integral
            
        except Exception as e:
            app_logger.error(f"Error calculating integral: {str(e)}")
            raise

    def update_plot(self):
        """Update plot with integration visualization"""
        if self.processed_data is None:
            return
            
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Get data range
            if self.use_range.get():
                start_idx = np.searchsorted(self.time_data, self.range_start.get())
                end_idx = np.searchsorted(self.time_data, self.range_end.get())
                t = self.time_data[start_idx:end_idx]
                y = self.processed_data[start_idx:end_idx]
            else:
                t = self.time_data
                y = self.processed_data
            
            # Plot original signal
            self.ax.plot(t, y, 'b-', label='Signal')
            
            # Show filled area if enabled
            if self.show_filled.get() and self.integral_data is not None:
                self.ax.fill_between(t, y, alpha=0.3)
            
            # Plot cumulative integral if calculated
            if self.integral_data is not None:
                ax2 = self.ax.twinx()
                ax2.plot(t, self.integral_data, 'r-', label='Cumulative Integral')
                ax2.set_ylabel('Integral', color='r')
            
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Signal")
            self.ax.legend(loc='upper left')
            if self.integral_data is not None:
                ax2.legend(loc='upper right')
            
            self.ax.grid(True)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            app_logger.debug("Integration plot updated")
            
        except Exception as e:
            app_logger.error(f"Error updating integration plot: {str(e)}")
            raise

    def on_accept(self):
        """Handle accept button click"""
        if self.callback and self.integral_data is not None:
            self.callback({
                'integral_value': float(self.integral_value.get().split()[2]),
                'integral_data': self.integral_data,
                'time_data': self.time_data,
                'original_data': self.processed_data
            })
        self.destroy()