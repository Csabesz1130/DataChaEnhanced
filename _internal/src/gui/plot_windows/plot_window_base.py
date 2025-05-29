# src/gui/plot_windows/plot_window_base.py

import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from src.utils.logger import app_logger

class PlotWindowBase(tk.Toplevel):
    """Enhanced base class for all plot windows with navigation capabilities"""
    
    def __init__(self, parent, title="Plot Window", size=(800, 600)):
        super().__init__(parent)
        self.title(title)
        self.geometry(f"{size[0]}x{size[1]}")
        
        # Initialize data attributes
        self.data = None
        self.time_data = None
        self.processed_data = None
        self.time_info = None
        
        # View history for navigation
        self.view_history = []
        self.current_view_index = -1
        
        # Create main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create plot frame
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(fill='both', expand=True, side='left')
        
        # Create control frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill='y', side='right', padx=5)
        
        # Setup matplotlib
        self.setup_plot()
        
        # Setup navigation controls
        self.setup_navigation_controls()
        
        # Setup view controls
        self.setup_view_controls()
        
        # Setup additional controls
        self.setup_controls()
        
        # Add accept/cancel buttons
        self.setup_action_buttons()
        
        # Bind window close event
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        app_logger.debug(f"Initialized {title} window")

    def setup_plot(self):
        """Setup matplotlib figure and canvas with enhanced navigation"""
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        # Connect event handlers
        self.canvas.mpl_connect('button_release_event', self.on_plot_interaction)

    def setup_navigation_controls(self):
        """Setup navigation control panel"""
        nav_frame = ttk.LabelFrame(self.control_frame, text="Navigation")
        nav_frame.pack(fill='x', padx=5, pady=5)
        
        # Add navigation buttons
        button_frame = ttk.Frame(nav_frame)
        button_frame.pack(fill='x', padx=5, pady=2)
        
        ttk.Button(button_frame, text="◀", width=3,
                  command=self.go_back).pack(side='left', padx=2)
        ttk.Button(button_frame, text="▶", width=3,
                  command=self.go_forward).pack(side='left', padx=2)
        ttk.Button(button_frame, text="Reset",
                  command=self.reset_view).pack(side='right', padx=2)

    def setup_view_controls(self):
        """Setup enhanced view control panel with time-based navigation"""
        view_frame = ttk.LabelFrame(self.control_frame, text="View")
        view_frame.pack(fill='x', padx=5, pady=5)
        
        # Add time range selector
        time_frame = ttk.Frame(view_frame)
        time_frame.pack(fill='x', padx=5, pady=2)
        
        # Time range presets
        self.view_range = tk.StringVar(value="Full")
        time_ranges = ["Full", "Last 10s", "Last 1s", "Last 100ms", "Custom"]
        ttk.Label(time_frame, text="Time Range:").pack(side='left')
        ttk.Combobox(time_frame, textvariable=self.view_range,
                    values=time_ranges, state='readonly',
                    width=15).pack(side='right')
        
        # Custom time range controls
        custom_frame = ttk.Frame(view_frame)
        custom_frame.pack(fill='x', padx=5, pady=2)
        
        self.custom_start = tk.DoubleVar(value=0.0)
        self.custom_end = tk.DoubleVar(value=1.0)
        
        ttk.Label(custom_frame, text="Start (s):").pack(side='left')
        ttk.Entry(custom_frame, textvariable=self.custom_start,
                 width=10).pack(side='left', padx=2)
        ttk.Label(custom_frame, text="End (s):").pack(side='left')
        ttk.Entry(custom_frame, textvariable=self.custom_end,
                 width=10).pack(side='left', padx=2)
        
        ttk.Button(view_frame, text="Apply Range",
                  command=self.update_view_range).pack(fill='x', padx=5, pady=2)
        
        # Time info display
        self.time_info_var = tk.StringVar(value="No data")
        ttk.Label(view_frame, textvariable=self.time_info_var,
                 wraplength=200, justify='left').pack(pady=2)

    def setup_controls(self):
        """Setup additional controls - to be implemented by subclasses"""
        pass

    def setup_action_buttons(self):
        """Setup accept/cancel buttons"""
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(side='bottom', fill='x', pady=10)
        
        ttk.Button(button_frame, text="Accept", 
                  command=self.on_accept).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.on_cancel).pack(side='right', padx=5)

    def set_data(self, time_data, data):
        """Set the data and initialize view with proper time handling"""
        self.time_data = np.array(time_data)
        self.data = np.array(data)
        self.processed_data = np.copy(data)
        
        # Calculate time interval statistics
        self.time_info = self.analyze_time_data()
        
        # Update custom range limits
        self.custom_start.set(self.time_info['start'])
        self.custom_end.set(self.time_info['end'])
        
        # Reset view history
        self.view_history = []
        self.current_view_index = -1
        
        # Update plot with correct time scaling
        self.update_plot()
        
        # Store initial view
        self.store_current_view()

    def analyze_time_data(self):
        """Analyze time data to determine intervals and scaling"""
        if self.time_data is None or len(self.time_data) < 2:
            return None

        time_info = {
            'start': self.time_data[0],
            'end': self.time_data[-1],
            'duration': self.time_data[-1] - self.time_data[0],
            'mean_interval': np.mean(np.diff(self.time_data)),
            'sampling_rate': 1.0 / np.mean(np.diff(self.time_data))
        }
        
        app_logger.debug(f"Time analysis: duration={time_info['duration']:.3f}s, "
                        f"sampling rate={time_info['sampling_rate']:.1f}Hz")
        return time_info

    def store_current_view(self):
        """Store current view limits in history"""
        view = {
            'xlim': self.ax.get_xlim(),
            'ylim': self.ax.get_ylim()
        }
        
        # Remove any forward history if we're not at the end
        if self.current_view_index < len(self.view_history) - 1:
            self.view_history = self.view_history[:self.current_view_index + 1]
        
        self.view_history.append(view)
        self.current_view_index = len(self.view_history) - 1

    def go_back(self):
        """Navigate to previous view"""
        if self.current_view_index > 0:
            self.current_view_index -= 1
            view = self.view_history[self.current_view_index]
            self.ax.set_xlim(view['xlim'])
            self.ax.set_ylim(view['ylim'])
            self.canvas.draw()

    def go_forward(self):
        """Navigate to next view"""
        if self.current_view_index < len(self.view_history) - 1:
            self.current_view_index += 1
            view = self.view_history[self.current_view_index]
            self.ax.set_xlim(view['xlim'])
            self.ax.set_ylim(view['ylim'])
            self.canvas.draw()

    def reset_view(self):
        """Reset to full view"""
        if self.time_info is not None:
            self.ax.set_xlim(self.time_info['start'], self.time_info['end'])
            margin = 0.1 * (np.max(self.data) - np.min(self.data))
            self.ax.set_ylim(np.min(self.data) - margin, np.max(self.data) + margin)
            self.canvas.draw()
            self.store_current_view()

    def update_view_range(self):
        """Update view based on selected time range"""
        if self.time_data is None or self.time_info is None:
            return
            
        range_type = self.view_range.get()
        t_max = self.time_info['end']
        
        try:
            if range_type == "Full":
                self.ax.set_xlim(self.time_info['start'], t_max)
            elif range_type == "Last 10s":
                self.ax.set_xlim(max(t_max - 10, self.time_info['start']), t_max)
            elif range_type == "Last 1s":
                self.ax.set_xlim(max(t_max - 1, self.time_info['start']), t_max)
            elif range_type == "Last 100ms":
                self.ax.set_xlim(max(t_max - 0.1, self.time_info['start']), t_max)
            elif range_type == "Custom":
                start = max(self.custom_start.get(), self.time_info['start'])
                end = min(self.custom_end.get(), t_max)
                if start < end:
                    self.ax.set_xlim(start, end)
            
            # Update time info display
            view_start, view_end = self.ax.get_xlim()
            self.time_info_var.set(
                f"View Range: {view_end - view_start:.3f}s\n"
                f"Sampling Rate: {self.time_info['sampling_rate']:.1f}Hz"
            )
            
            self.canvas.draw()
            self.store_current_view()
            
        except Exception as e:
            app_logger.error(f"Error updating view range: {str(e)}")
            raise

    def on_plot_interaction(self, event):
        """Handle plot interaction events"""
        if event.inaxes == self.ax:
            self.store_current_view()

    def update_plot(self):
        """Base update plot function with proper time handling"""
        if self.data is None or self.time_data is None:
            return
            
        try:
            self.ax.clear()
            
            # Plot with actual time values
            self.ax.plot(self.time_data, self.data, 'b-', 
                        label='Original', alpha=0.5)
            
            if self.processed_data is not None:
                self.ax.plot(self.time_data, self.processed_data, 'r-',
                           label='Processed', linewidth=1.5)
            
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Signal")
            self.ax.legend()
            self.ax.grid(True)
            
            # Update time info
            if self.time_info:
                self.time_info_var.set(
                    f"Duration: {self.time_info['duration']:.3f}s\n"
                    f"Sampling Rate: {self.time_info['sampling_rate']:.1f}Hz"
                )
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            app_logger.error(f"Error in base plot update: {str(e)}")
            raise

    def on_accept(self):
        """Handle accept button click - to be implemented by subclasses"""
        pass

    def on_cancel(self):
        """Handle cancel button click"""
        self.destroy()

    def get_processed_data(self):
        """Return the processed data"""
        return self.processed_data