import tkinter as tk
from tkinter import ttk
import numpy as np
from src.utils.logger import app_logger
from .plot_window_base import PlotWindowBase

class BaselineCorrectionWindow(PlotWindowBase):
    def __init__(self, parent, callback=None):
        super().__init__(parent, title="Baseline Correction")
        self.callback = callback
        
        # Initialize baseline parameters
        self.baseline_method = tk.StringVar(value="constant")
        self.baseline_window_size = tk.IntVar(value=100)
        self.constant_value = tk.DoubleVar(value=0.0)
        self.window_start = tk.IntVar(value=0)
        self.window_end = tk.IntVar(value=100)
        
        # Setup specific controls
        self.setup_baseline_controls()
        
        app_logger.debug("Baseline correction window initialized")

    def setup_baseline_controls(self):
        """Setup baseline correction specific controls"""
        # Method selection
        method_frame = ttk.LabelFrame(self.control_frame, text="Baseline Method")
        method_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Radiobutton(method_frame, text="Constant Value",
                       variable=self.baseline_method, value="constant",
                       command=self.update_plot).pack(pady=2)
        
        ttk.Radiobutton(method_frame, text="Window Average",
                       variable=self.baseline_method, value="window",
                       command=self.update_plot).pack(pady=2)
        
        # Constant value control
        const_frame = ttk.LabelFrame(self.control_frame, text="Constant Value")
        const_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Entry(const_frame, textvariable=self.constant_value).pack(pady=2)
        
        # Window controls
        window_frame = ttk.LabelFrame(self.control_frame, text="Window Settings")
        window_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(window_frame, text="Start Index:").pack()
        ttk.Entry(window_frame, textvariable=self.window_start).pack(pady=2)
        
        ttk.Label(window_frame, text="End Index:").pack()
        ttk.Entry(window_frame, textvariable=self.window_end).pack(pady=2)
        
        ttk.Label(window_frame, text="Window Size:").pack()
        ttk.Entry(window_frame, textvariable=self.baseline_window_size).pack(pady=2)
        
        # Add update button
        ttk.Button(self.control_frame, text="Update",
                  command=self.update_plot).pack(pady=5)

    def calculate_baseline(self):
        """Calculate baseline based on selected method"""
        if self.baseline_method.get() == "constant":
            return np.full_like(self.data, self.constant_value.get())
            
        else:  # window method
            start = self.window_start.get()
            end = min(self.window_end.get(), len(self.data))
            window = self.data[start:end]
            return np.mean(window)

    def update_plot(self):
        """Update plot with baseline correction"""
        if self.data is None:
            return
            
        try:
            # Calculate baseline
            baseline = self.calculate_baseline()
            
            # Apply baseline correction
            if self.baseline_method.get() == "constant":
                self.processed_data = self.data - baseline
            else:
                self.processed_data = self.data - np.mean(baseline)
            
            # Clear previous plot
            self.ax.clear()
            
            # Plot original data
            self.ax.plot(self.time_data, self.data, 'b-', 
                        label='Original', alpha=0.5)
            
            # Plot baseline-corrected data
            self.ax.plot(self.time_data, self.processed_data, 'r-',
                        label='Baseline Corrected')
            
            # Plot baseline if using window method
            if self.baseline_method.get() == "window":
                start = self.window_start.get()
                end = self.window_end.get()
                self.ax.axhline(y=np.mean(baseline), color='g', 
                              linestyle='--', label='Baseline')
                self.ax.axvspan(self.time_data[start], self.time_data[end],
                              alpha=0.2, color='yellow', label='Baseline Window')
            
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Signal")
            self.ax.legend()
            self.ax.grid(True)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            app_logger.debug("Baseline correction plot updated")
            
        except Exception as e:
            app_logger.error(f"Error updating baseline plot: {str(e)}")
            raise

    def on_accept(self):
        """Handle accept button click"""
        if self.callback:
            self.callback(self.processed_data)
        self.destroy()