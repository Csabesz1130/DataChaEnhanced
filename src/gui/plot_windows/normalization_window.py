import tkinter as tk
from tkinter import ttk
import numpy as np
from src.utils.logger import app_logger
from src.gui.plot_windows.plot_window_base import PlotWindowBase

class NormalizationWindow(PlotWindowBase):
    def __init__(self, parent, callback=None):
        super().__init__(parent, title="Signal Normalization")
        self.callback = callback
        
        # Initialize normalization parameters
        self.norm_method = tk.StringVar(value="minmax")
        self.v0 = tk.DoubleVar(value=-80.0)  # Initial voltage
        self.v1 = tk.DoubleVar(value=20.0)   # Final voltage
        self.custom_min = tk.DoubleVar(value=0.0)
        self.custom_max = tk.DoubleVar(value=1.0)
        
        # Add trace to variables for auto-update
        self.v0.trace_add("write", lambda *args: self.update_plot())
        self.v1.trace_add("write", lambda *args: self.update_plot())
        self.custom_min.trace_add("write", lambda *args: self.update_plot())
        self.custom_max.trace_add("write", lambda *args: self.update_plot())
        
        # Setup specific controls
        self.setup_normalization_controls()
        
        app_logger.debug("Normalization window initialized")

    def setup_normalization_controls(self):
        """Setup normalization specific controls"""
        # Method selection
        method_frame = ttk.LabelFrame(self.control_frame, text="Normalization Method")
        method_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Radiobutton(method_frame, text="Min-Max",
                       variable=self.norm_method, value="minmax",
                       command=self.update_plot).pack(pady=2)
        
        ttk.Radiobutton(method_frame, text="Voltage Range",
                       variable=self.norm_method, value="voltage",
                       command=self.update_plot).pack(pady=2)
        
        # Voltage range controls
        voltage_frame = ttk.LabelFrame(self.control_frame, text="Voltage Range")
        voltage_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(voltage_frame, text="V0 (mV):").pack()
        ttk.Entry(voltage_frame, textvariable=self.v0).pack(pady=2)
        ttk.Scale(voltage_frame, from_=-100, to=0,
                 variable=self.v0, orient='horizontal').pack(fill='x', pady=2)
        
        ttk.Label(voltage_frame, text="V1 (mV):").pack()
        ttk.Entry(voltage_frame, textvariable=self.v1).pack(pady=2)
        ttk.Scale(voltage_frame, from_=0, to=100,
                 variable=self.v1, orient='horizontal').pack(fill='x', pady=2)
        
        # Custom range controls
        custom_frame = ttk.LabelFrame(self.control_frame, text="Custom Range")
        custom_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(custom_frame, text="Min Value:").pack()
        ttk.Entry(custom_frame, textvariable=self.custom_min).pack(pady=2)
        ttk.Scale(custom_frame, from_=-10, to=10,
                 variable=self.custom_min, orient='horizontal').pack(fill='x', pady=2)
        
        ttk.Label(custom_frame, text="Max Value:").pack()
        ttk.Entry(custom_frame, textvariable=self.custom_max).pack(pady=2)
        ttk.Scale(custom_frame, from_=-10, to=10,
                 variable=self.custom_max, orient='horizontal').pack(fill='x', pady=2)
        
        # Statistics display
        stats_frame = ttk.LabelFrame(self.control_frame, text="Statistics")
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        self.stats_var = tk.StringVar(value="No data")
        stats_label = ttk.Label(stats_frame, textvariable=self.stats_var,
                              wraplength=200, justify='left')
        stats_label.pack(pady=5)
        
        # Reference lines toggle
        self.show_refs = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Show Reference Lines",
                       variable=self.show_refs,
                       command=self.update_plot).pack(pady=5)
        
        # Reset button
        ttk.Button(self.control_frame, text="Reset",
                  command=self.reset_normalization).pack(pady=5)

    def normalize_data(self):
        """Normalize data based on selected method"""
        try:
            if self.norm_method.get() == "minmax":
                # Min-max normalization to custom range
                min_val = np.min(self.data)
                max_val = np.max(self.data)
                custom_min = self.custom_min.get()
                custom_max = self.custom_max.get()
                
                normalized = (self.data - min_val) * (custom_max - custom_min)
                normalized = normalized / (max_val - min_val) + custom_min
                
            else:  # voltage range normalization
                v0 = self.v0.get()
                v1 = self.v1.get()
                
                # Scale data to voltage range
                data_min = np.min(self.data)
                data_max = np.max(self.data)
                scale = (v1 - v0) / (data_max - data_min)
                normalized = (self.data - data_min) * scale + v0
            
            return normalized
            
        except Exception as e:
            app_logger.error(f"Error in normalization: {str(e)}")
            raise

    def update_statistics(self):
        """Update statistics display"""
        if self.processed_data is None:
            return
            
        stats = {
            'min': np.min(self.processed_data),
            'max': np.max(self.processed_data),
            'mean': np.mean(self.processed_data),
            'std': np.std(self.processed_data),
            'range': np.ptp(self.processed_data)
        }
        
        stats_text = (
            f"Statistics:\n"
            f"Min: {stats['min']:.2f}\n"
            f"Max: {stats['max']:.2f}\n"
            f"Mean: {stats['mean']:.2f}\n"
            f"Std Dev: {stats['std']:.2f}\n"
            f"Range: {stats['range']:.2f}"
        )
        
        self.stats_var.set(stats_text)

    def update_plot(self):
        """Update plot with normalized data"""
        if self.data is None:
            return
            
        try:
            # Normalize data
            self.processed_data = self.normalize_data()
            
            # Update statistics
            self.update_statistics()
            
            # Clear previous plot
            self.ax.clear()
            
            # Plot original data
            self.ax.plot(self.time_data, self.data, 'b-', 
                        label='Original', alpha=0.5)
            
            # Plot normalized data
            self.ax.plot(self.time_data, self.processed_data, 'r-',
                        label='Normalized')
            
            # Add reference lines if enabled
            if self.show_refs.get():
                if self.norm_method.get() == "voltage":
                    self.ax.axhline(y=self.v0.get(), color='g',
                                  linestyle='--', label='V0')
                    self.ax.axhline(y=self.v1.get(), color='g',
                                  linestyle='--', label='V1')
                else:
                    self.ax.axhline(y=self.custom_min.get(), color='g',
                                  linestyle='--', label='Min')
                    self.ax.axhline(y=self.custom_max.get(), color='g',
                                  linestyle='--', label='Max')
            
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Signal")
            self.ax.legend()
            self.ax.grid(True)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            app_logger.debug("Normalization plot updated")
            
        except Exception as e:
            app_logger.error(f"Error updating normalization plot: {str(e)}")
            raise

    def reset_normalization(self):
        """Reset normalization parameters to defaults"""
        self.norm_method.set("minmax")
        self.v0.set(-80.0)
        self.v1.set(20.0)
        self.custom_min.set(0.0)
        self.custom_max.set(1.0)
        self.show_refs.set(True)
        self.update_plot()

    def on_accept(self):
        """Handle accept button click"""
        if self.callback:
            self.callback(self.processed_data)
        self.destroy()