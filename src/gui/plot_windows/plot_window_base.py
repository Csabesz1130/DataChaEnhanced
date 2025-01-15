import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from src.utils.logger import app_logger

class PlotWindowBase(tk.Toplevel):
    """Base class for all plot windows"""
    
    def __init__(self, parent, title="Plot Window", size=(800, 600)):
        super().__init__(parent)
        self.title(title)
        self.geometry(f"{size[0]}x{size[1]}")
        
        # Initialize data
        self.data = None
        self.time_data = None
        self.processed_data = None
        
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
        
        # Setup controls
        self.setup_controls()
        
        # Add accept/cancel buttons
        self.setup_action_buttons()
        
        # Bind window close event
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        app_logger.debug(f"Initialized {title} window")

    def setup_plot(self):
        """Setup matplotlib figure and canvas"""
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def setup_controls(self):
        """Setup control panel - to be implemented by subclasses"""
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
        """Set the data to be plotted"""
        self.time_data = time_data
        self.data = data
        self.processed_data = np.copy(data)
        self.update_plot()

    def update_plot(self):
        """Update the plot - to be implemented by subclasses"""
        pass

    def on_accept(self):
        """Handle accept button click - to be implemented by subclasses"""
        pass

    def on_cancel(self):
        """Handle cancel button click"""
        self.destroy()

    def get_processed_data(self):
        """Return the processed data"""
        return self.processed_data