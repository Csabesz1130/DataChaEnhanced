# src/gui/plot_windows/integration_window.py

import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.integrate import simpson
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
        self.show_cumulative = tk.BooleanVar(value=True)
        
        # Integration results
        self.integral_value = tk.StringVar(value="Not calculated")
        self.integral_data = None
        
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
        
        # Start time control with slider
        start_frame = ttk.Frame(range_frame)
        start_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(start_frame, text="Start:").pack(side='left')
        ttk.Entry(start_frame, textvariable=self.range_start,
                 width=10).pack(side='right')
        ttk.Scale(range_frame, from_=0, to=1,
                 variable=self.range_start,
                 orient='horizontal',
                 command=lambda *args: self.update_plot()).pack(fill='x')
        
        # End time control with slider
        end_frame = ttk.Frame(range_frame)
        end_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(end_frame, text="End:").pack(side='left')
        ttk.Entry(end_frame, textvariable=self.range_end,
                 width=10).pack(side='right')
        ttk.Scale(range_frame, from_=0, to=1,
                 variable=self.range_end,
                 orient='horizontal',
                 command=lambda *args: self.update_plot()).pack(fill='x')
        
        # Display options
        display_frame = ttk.LabelFrame(self.control_frame, text="Display Options")
        display_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(display_frame, text="Show Filled Area",
                       variable=self.show_filled,
                       command=self.update_plot).pack(pady=2)
        
        ttk.Checkbutton(display_frame, text="Show Cumulative",
                       variable=self.show_cumulative,
                       command=self.update_plot).pack(pady=2)
        
        # Results display
        results_frame = ttk.LabelFrame(self.control_frame, text="Integration Results")
        results_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(results_frame, textvariable=self.integral_value,
                 wraplength=200, justify='left').pack(pady=5)
        
        # Calculate button
        ttk.Button(self.control_frame, text="Calculate",
                  command=self.calculate_integral).pack(pady=5)

    def set_data(self, time_data, data):
        """Override set_data to update range scales"""
        super().set_data(time_data, data)
        
        if time_data is not None:
            t_min = np.min(time_data)
            t_max = np.max(time_data)
            
            # Update range scales
            for widget in self.control_frame.winfo_children():
                if isinstance(widget, ttk.Scale):
                    widget.configure(from_=t_min, to=t_max)
            
            self.range_start.set(t_min)
            self.range_end.set(t_max)

    # Add these methods to the IntegrationWindow class

    def set_normalized_data(self, normalized_data, normalization_params):
        """Set normalized data and its parameters"""
        self.normalization_params = normalization_params
        self.original_data = self.data  # Store original data
        self.data = normalized_data
        self.processed_data = normalized_data.copy()
        self.update_plot()
        app_logger.debug("Set normalized data with parameters: " + str(normalization_params))

    # In IntegrationWindow class:

    def calculate_integral(self):
        """Calculate the integral using proper time units"""
        try:
            if self.processed_data is None:
                return
            
            # Get data range in real time units
            if self.use_range.get():
                start_idx = np.searchsorted(self.time_data, self.range_start.get())
                end_idx = np.searchsorted(self.time_data, self.range_end.get())
                t = self.time_data[start_idx:end_idx]
                y = self.processed_data[start_idx:end_idx]
            else:
                t = self.time_data
                y = self.processed_data
            
            # Calculate integral in proper units
            if self.integration_method.get() == "simps":
                # Simpson's integration using real time intervals
                integral = simpson(y, t)
                if self.show_cumulative.get():
                    self.integral_data = np.array([
                        simpson(y[:i+1], t[:i+1]) for i in range(len(t))
                    ])
            else:  # trapz
                # Trapezoidal integration using real time intervals
                integral = np.trapz(y, t)
                if self.show_cumulative.get():
                    self.integral_data = np.array([
                        np.trapz(y[:i+1], t[:i+1]) for i in range(len(t))
                    ])
            
            # Update results with proper time units
            duration = t[-1] - t[0]
            self.integral_value.set(
                f"Integration Results:\n"
                f"Method: {self.integration_method.get()}\n"
                f"Integral: {integral:.6e} pA·s\n"
                f"Duration: {duration:.3f} s\n"
                f"Mean Rate: {integral/duration:.2e} pA"
            )
            
            self.update_plot()
            return integral
            
        except Exception as e:
            app_logger.error(f"Error calculating integral: {str(e)}")
            raise

    # In integration_window.py, update the plotting methods

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
            
            # Plot signal with consistent styling
            self.ax.plot(t, y, 'b-', label='Signal', linewidth=1.5)
            
            # Show filled area if enabled (make it match the main plot style)
            if self.show_filled.get():
                self.ax.fill_between(t, y, alpha=0.3, color='blue')
            
            # Plot cumulative integral if calculated and enabled
            if self.show_cumulative.get() and self.integral_data is not None:
                ax2 = self.ax.twinx()
                ax2.plot(t, self.integral_data, 'r-',
                        label='Cumulative Integral', linewidth=1.5)
                ax2.set_ylabel('Integral', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                
                # Add legend for second axis
                lines2, labels2 = ax2.get_legend_handles_labels()
                self.ax.legend(loc='upper left')
                ax2.legend(lines2, labels2, loc='upper right')
            else:
                self.ax.legend()
            
            # Set consistent grid style
            self.ax.grid(True, which='both', color='gray', linestyle='-', alpha=0.2)
            
            # Set axis labels with proper units
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Current (pA)')
            
            # Ensure axis limits match main plot style
            if hasattr(self, 'ylim'):
                self.ax.set_ylim(self.ylim)
            
            # Match main plot time formatting
            self.format_time_axis()
            
            # Update layout and draw
            self.fig.tight_layout()
            self.canvas.draw()
            
            app_logger.debug("Integration plot updated")
            
        except Exception as e:
            app_logger.error(f"Error updating integration plot: {str(e)}")
            raise

    def format_time_axis(self):
        """Format time axis to match main plot"""
        xlim = self.ax.get_xlim()
        time_range = xlim[1] - xlim[0]
        
        if time_range < 0.1:  # Less than 100ms
            self.ax.xaxis.set_major_formatter(lambda x, p: f"{x*1000:.1f} ms")
        elif time_range < 1:  # Less than 1s
            self.ax.xaxis.set_major_formatter(lambda x, p: f"{x*1000:.0f} ms")
        elif time_range < 60:  # Less than 1min
            self.ax.xaxis.set_major_formatter(lambda x, p: f"{x:.1f} s")
        else:  # More than 1min
            self.ax.xaxis.set_major_formatter(lambda x, p: f"{x/60:.1f} min")

    def set_data(self, time_data, data, ylim=None):
        """Set data with optional y-axis limits"""
        super().set_data(time_data, data)
        if ylim is not None:
            self.ylim = ylim
            self.ax.set_ylim(ylim)


    def on_accept(self):
        """Handle accept button click"""
        if self.callback and self.integral_data is not None:
            result = {
                'integral_value': float(self.integral_value.get().split()[-2]),
                'integral_data': self.integral_data,
                'time_data': self.time_data,
                'original_data': self.processed_data,
                'range': {
                    'start': self.range_start.get(),
                    'end': self.range_end.get()
                } if self.use_range.get() else None
            }
            self.callback(result)
        self.destroy()