# src/gui/plot_windows/normalization_window.py

import tkinter as tk
from tkinter import ttk
import numpy as np
from src.utils.logger import app_logger
from .plot_window_base import PlotWindowBase

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
        self.show_reference = tk.BooleanVar(value=True)
        self.show_stats = tk.BooleanVar(value=True)
        self.auto_update = tk.BooleanVar(value=True)
        
        # Initialize statistics
        self.stats_var = tk.StringVar(value="No data")
        
        # Setup parameter traces
        self.setup_variable_traces()
        
        # Setup controls
        self.setup_normalization_controls()
        
        app_logger.debug("Normalization window initialized")

    def setup_variable_traces(self):
        """Setup variable traces for auto-update"""
        for var in [self.v0, self.v1, self.custom_min, self.custom_max]:
            var.trace_add("write", self.on_parameter_change)

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
        
        # V0 controls
        v0_frame = ttk.Frame(voltage_frame)
        v0_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(v0_frame, text="V0 (mV):").pack(side='left')
        ttk.Entry(v0_frame, textvariable=self.v0, width=10).pack(side='right')
        ttk.Scale(voltage_frame, from_=-100, to=0,
                 variable=self.v0, orient='horizontal').pack(fill='x')
        
        # V1 controls
        v1_frame = ttk.Frame(voltage_frame)
        v1_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(v1_frame, text="V1 (mV):").pack(side='left')
        ttk.Entry(v1_frame, textvariable=self.v1, width=10).pack(side='right')
        ttk.Scale(voltage_frame, from_=0, to=100,
                 variable=self.v1, orient='horizontal').pack(fill='x')
        
        # Custom range controls
        custom_frame = ttk.LabelFrame(self.control_frame, text="Custom Range")
        custom_frame.pack(fill='x', padx=5, pady=5)
        
        # Min value controls
        min_frame = ttk.Frame(custom_frame)
        min_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(min_frame, text="Min:").pack(side='left')
        ttk.Entry(min_frame, textvariable=self.custom_min, width=10).pack(side='right')
        ttk.Scale(custom_frame, from_=-10, to=10,
                 variable=self.custom_min, orient='horizontal').pack(fill='x')
        
        # Max value controls
        max_frame = ttk.Frame(custom_frame)
        max_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(max_frame, text="Max:").pack(side='left')
        ttk.Entry(max_frame, textvariable=self.custom_max, width=10).pack(side='right')
        ttk.Scale(custom_frame, from_=-10, to=10,
                 variable=self.custom_max, orient='horizontal').pack(fill='x')
        
        # Display options
        display_frame = ttk.LabelFrame(self.control_frame, text="Display Options")
        display_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(display_frame, text="Show Reference Lines",
                       variable=self.show_reference,
                       command=self.update_plot).pack(pady=2)
        
        ttk.Checkbutton(display_frame, text="Show Statistics",
                       variable=self.show_stats,
                       command=self.update_statistics).pack(pady=2)
        
        ttk.Checkbutton(display_frame, text="Auto Update",
                       variable=self.auto_update).pack(pady=2)
        
        # Statistics display
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Statistics")
        self.stats_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(self.stats_frame, textvariable=self.stats_var,
                 wraplength=200, justify='left').pack(pady=5)
        
        # Control buttons
        ttk.Button(self.control_frame, text="Update",
                  command=self.update_plot).pack(pady=2)
        ttk.Button(self.control_frame, text="Reset",
                  command=self.reset_normalization).pack(pady=2)

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
        if self.processed_data is None or not self.show_stats.get():
            self.stats_var.set("No data")
            return
            
        try:
            orig_stats = {
                'min': np.min(self.data),
                'max': np.max(self.data),
                'mean': np.mean(self.data),
                'std': np.std(self.data)
            }
            
            norm_stats = {
                'min': np.min(self.processed_data),
                'max': np.max(self.processed_data),
                'mean': np.mean(self.processed_data),
                'std': np.std(self.processed_data)
            }
            
            stats_text = (
                f"Original Data:\n"
                f"  Min: {orig_stats['min']:.2f}\n"
                f"  Max: {orig_stats['max']:.2f}\n"
                f"  Mean: {orig_stats['mean']:.2f}\n"
                f"  Std: {orig_stats['std']:.2f}\n\n"
                f"Normalized Data:\n"
                f"  Min: {norm_stats['min']:.2f}\n"
                f"  Max: {norm_stats['max']:.2f}\n"
                f"  Mean: {norm_stats['mean']:.2f}\n"
                f"  Std: {norm_stats['std']:.2f}"
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
                        label='Normalized', linewidth=1.5)
            
            # Add reference lines if enabled
            if self.show_reference.get():
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
        self.show_reference.set(True)
        self.show_stats.set(True)
        self.update_plot()

    def on_accept(self):
        """Handle accept button click"""
        if self.callback:
            result = {
                'processed_data': self.processed_data,
                'method': self.norm_method.get(),
                'parameters': {
                    'v0': self.v0.get(),
                    'v1': self.v1.get(),
                    'custom_min': self.custom_min.get(),
                    'custom_max': self.custom_max.get()
                },
                'statistics': {
                    'original_min': np.min(self.data),
                    'original_max': np.max(self.data),
                    'normalized_min': np.min(self.processed_data),
                    'normalized_max': np.max(self.processed_data)
                }
            }
            self.callback(result)
        self.destroy()